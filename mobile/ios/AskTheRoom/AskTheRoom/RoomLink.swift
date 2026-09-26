import CoreBluetooth
import Foundation
import os

/// GATT layout from spec section 3.
enum RigGATT {
    static let service = CBUUID(string: "8A1E0001-6B7F-4C2B-9E3A-2F5D7C1A0001")
    static let question = CBUUID(string: "8A1E0002-6B7F-4C2B-9E3A-2F5D7C1A0002")
    static let answer = CBUUID(string: "8A1E0003-6B7F-4C2B-9E3A-2F5D7C1A0003")
    static let state = CBUUID(string: "8A1E0004-6B7F-4C2B-9E3A-2F5D7C1A0004")
    static let status = CBUUID(string: "8A1E0005-6B7F-4C2B-9E3A-2F5D7C1A0005")
    static let incoming = [answer, state, status]
}

/// The Bluetooth link to the rig (spec section 5, "Connect"): scan for the service,
/// connect to the strongest rig, remember it, and reconnect forever with backoff up to 5 s.
@MainActor
final class RoomLink: NSObject, RoomTransport {
    static let rememberedKey = "rigPeripheralID"
    /// How long to keep listening after the first rig is heard, to pick the strongest.
    static let chooseWindow: Duration = .milliseconds(800)
    static let maxBackoff = 5.0

    private let log = Logger(subsystem: "com.asktheroom.app", category: "RoomLink")
    private weak var store: RoomStore?
    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var questionCharacteristic: CBCharacteristic?
    private var reassemblers: [CBUUID: Reassembler] = [:]
    private var heard: [UUID: (peripheral: CBPeripheral, rssi: Int)] = [:]
    private var chooseTask: Task<Void, Never>?
    private var retryTask: Task<Void, Never>?
    private var backoff = 0.5
    private var stopped = false
    /// A question asked while the link was down; sent as soon as it's back.
    private var waiting: Question?

    init(store: RoomStore) {
        self.store = store
        super.init()
        central = CBCentralManager(delegate: self, queue: nil,
                                   options: [CBCentralManagerOptionShowPowerAlertKey: false])
    }

    // MARK: RoomTransport

    func send(_ question: Question) {
        guard let data = question.encoded() else { return }
        guard let peripheral, peripheral.state == .connected, let characteristic = questionCharacteristic else {
            waiting = question
            return
        }
        waiting = nil
        peripheral.writeValue(data, for: characteristic, type: .withResponse)
    }

    func stop() {
        stopped = true
        chooseTask?.cancel()
        retryTask?.cancel()
        if central.state == .poweredOn { central.stopScan() }
        if let peripheral { central.cancelPeripheralConnection(peripheral) }
        peripheral = nil
    }

    // MARK: Finding the rig

    private var rememberedID: UUID? {
        get { UserDefaults.standard.string(forKey: Self.rememberedKey).flatMap(UUID.init(uuidString:)) }
        set { UserDefaults.standard.set(newValue?.uuidString, forKey: Self.rememberedKey) }
    }

    private func findRig() {
        guard !stopped, central.state == .poweredOn, peripheral == nil else { return }
        store?.linkChanged(.searching)

        // Already connected by the system (e.g. after the app relaunches), or remembered:
        // a pending connect completes whenever the rig comes back in range.
        let known = central.retrieveConnectedPeripherals(withServices: [RigGATT.service]).first
            ?? rememberedID.flatMap { central.retrievePeripherals(withIdentifiers: [$0]).first }
        if let known {
            connect(known)
        }
        // Scan as well, in case the rig's identity changed or none is remembered.
        heard = [:]
        central.scanForPeripherals(withServices: [RigGATT.service])
    }

    private func discovered(_ p: CBPeripheral, rssi: Int) {
        guard peripheral == nil || peripheral?.state != .connected else { return }
        heard[p.identifier] = (p, rssi)
        if p.identifier == rememberedID {
            connect(p)
            return
        }
        guard chooseTask == nil else { return }
        chooseTask = Task { [weak self] in
            try? await Task.sleep(for: Self.chooseWindow)
            guard let self, !Task.isCancelled else { return }
            self.chooseTask = nil
            if let best = self.heard.values.max(by: { $0.rssi < $1.rssi }) {
                self.connect(best.peripheral)
            }
        }
    }

    private func connect(_ p: CBPeripheral) {
        if let current = peripheral {
            if current.identifier == p.identifier, current.state != .disconnected { return }
            if current.identifier != p.identifier { central.cancelPeripheralConnection(current) }
        }
        peripheral = p
        p.delegate = self
        store?.linkChanged(.connecting)
        central.connect(p)
    }

    private func connected(_ p: CBPeripheral) {
        guard p.identifier == peripheral?.identifier else { return }
        central.stopScan()
        chooseTask?.cancel()
        chooseTask = nil
        rememberedID = p.identifier
        backoff = 0.5
        log.info("connected; MTU \(p.maximumWriteValueLength(for: .withoutResponse) + 3)")
        p.discoverServices([RigGATT.service])
    }

    private func lost(_ p: CBPeripheral, error: Error?) {
        guard p.identifier == peripheral?.identifier else { return }
        log.info("disconnected: \(error?.localizedDescription ?? "no error")")
        peripheral = nil
        questionCharacteristic = nil
        reassemblers = [:]
        store?.linkChanged(.searching)
        retryLater()
    }

    private func retryLater() {
        retryTask?.cancel()
        let delay = backoff
        backoff = min(Self.maxBackoff, backoff * 2)
        retryTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            self?.findRig()
        }
    }

    // MARK: Services

    private func foundServices(_ p: CBPeripheral) {
        guard let service = p.services?.first(where: { $0.uuid == RigGATT.service }) else {
            log.error("rig has no Ask the Room service")
            central.cancelPeripheralConnection(p)
            return
        }
        p.discoverCharacteristics([RigGATT.question] + RigGATT.incoming, for: service)
    }

    private func foundCharacteristics(_ p: CBPeripheral, service: CBService) {
        for c in service.characteristics ?? [] {
            switch c.uuid {
            case RigGATT.question:
                questionCharacteristic = c
            case RigGATT.answer, RigGATT.state:
                p.setNotifyValue(true, for: c)
            case RigGATT.status:
                p.setNotifyValue(true, for: c)
                p.readValue(for: c)
            default:
                break
            }
        }
        store?.linkChanged(.connected)
        if let waiting { send(waiting) }
    }

    // MARK: Incoming

    private func received(_ value: Data, on uuid: CBUUID) {
        // A status read may come back as plain JSON rather than a framed chunk.
        if uuid == RigGATT.status, value.first == UInt8(ascii: "{"), let status = Wire.decode(RigStatus.self, from: value) {
            store?.receive(status: status)
            return
        }
        guard let message = reassemblers[uuid, default: Reassembler()].add(value) else { return }
        switch uuid {
        case RigGATT.answer:
            guard let answer = Wire.decode(Answer.self, from: message) else { return dropped(message, uuid) }
            store?.receive(answer: answer)
        case RigGATT.state:
            guard let state = Wire.decode(Snapshot.self, from: message) else { return dropped(message, uuid) }
            store?.receive(state: state)
        case RigGATT.status:
            guard let status = Wire.decode(RigStatus.self, from: message) else { return dropped(message, uuid) }
            store?.receive(status: status)
        default:
            break
        }
    }

    private func dropped(_ message: Data, _ uuid: CBUUID) {
        log.error("dropped malformed JSON on \(uuid.uuidString, privacy: .public) (\(message.count) bytes)")
    }
}

// CoreBluetooth calls back on the main queue (queue: nil above).

extension RoomLink: CBCentralManagerDelegate {
    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        let state = central.state
        MainActor.assumeIsolated {
            switch state {
            case .poweredOn: findRig()
            case .poweredOff: store?.linkChanged(.bluetoothOff)
            case .unauthorized: store?.linkChanged(.unauthorized)
            case .unsupported: store?.linkChanged(.unsupported)
            default: break
            }
            if state != .poweredOn {
                peripheral = nil
                questionCharacteristic = nil
            }
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                                    advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let rssi = RSSI.intValue
        MainActor.assumeIsolated {
            // 127 means "unknown"; treat it as weakest.
            discovered(peripheral, rssi: rssi == 127 ? -127 : rssi)
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        MainActor.assumeIsolated { connected(peripheral) }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        MainActor.assumeIsolated { lost(peripheral, error: error) }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        MainActor.assumeIsolated { lost(peripheral, error: error) }
    }
}

extension RoomLink: CBPeripheralDelegate {
    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        MainActor.assumeIsolated { foundServices(peripheral) }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        MainActor.assumeIsolated { foundCharacteristics(peripheral, service: service) }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard error == nil, let value = characteristic.value else { return }
        let uuid = characteristic.uuid
        MainActor.assumeIsolated { received(value, on: uuid) }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let error else { return }
        let message = error.localizedDescription
        MainActor.assumeIsolated { log.error("question write failed: \(message, privacy: .public)") }
    }

    /// The rig's app restarted and re-registered its service: find it again.
    nonisolated func peripheral(_ peripheral: CBPeripheral, didModifyServices invalidatedServices: [CBService]) {
        guard invalidatedServices.contains(where: { $0.uuid == RigGATT.service }) else { return }
        MainActor.assumeIsolated {
            questionCharacteristic = nil
            reassemblers = [:]
            peripheral.discoverServices([RigGATT.service])
        }
    }
}
