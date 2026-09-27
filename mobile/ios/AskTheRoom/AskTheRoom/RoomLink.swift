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
    /// In the order to subscribe: state last, because subscribing to it sends a snapshot at once.
    static let incoming = [answer, status, state]

    static func channel(for uuid: CBUUID) -> LinkChannel? {
        switch uuid {
        case answer: return .answer
        case state: return .state
        case status: return .status
        default: return nil
        }
    }
}

/// The Bluetooth link to the rig (spec section 5, "Connect"): scan for the service,
/// connect to the strongest rig, remember it, and reconnect forever: at once after a drop,
/// then with backoff up to 5 s.
///
/// It keeps going in the background (`bluetooth-central` background mode), and with state
/// restoration iOS relaunches the app when the rig comes back, so the map is current whenever
/// the rig is on. The pending connect to the remembered rig never times out.
@MainActor
final class RoomLink: NSObject, RoomTransport {
    static let rememberedKey = "rigPeripheralID"
    static let restoreID = "AskTheRoomCentral"
    /// How long to keep listening after the first rig is heard, to pick the strongest.
    static let chooseWindow: Duration = .milliseconds(800)
    static let maxBackoff = 5.0
    /// The bridge sends state at least every 5 s. Nothing for three heartbeats means the link is up but
    /// the bridge hung or lost our subscription, so reconnect.
    static let staleAfter: Duration = .seconds(15)
    /// Diagnostics go to the store at most this often, so the settings screen doesn't redraw constantly.
    static let statsEvery: Duration = .seconds(1)
    static let watchdogReason = "watchdog: nothing for 15 s"

    private let log = Logger(subsystem: "com.asktheroom.app", category: "RoomLink")
    private weak var store: RoomStore?
    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var questionCharacteristic: CBCharacteristic?
    private var meter = LinkMeter()
    private var statsTask: Task<Void, Never>?
    private var lastStatsPublish: ContinuousClock.Instant?
    /// Why we're about to cancel the connection ourselves, for the disconnect that follows.
    private var disconnectReason: String?
    private var heard: [UUID: (peripheral: CBPeripheral, rssi: Int)] = [:]
    private var chooseTask: Task<Void, Never>?
    private var retryTask: Task<Void, Never>?
    private var watchTask: Task<Void, Never>?
    private var lastHeard = ContinuousClock.now
    private var backoff = 0.5
    private var stopped = false
    /// A question asked while the link was down; sent as soon as it's back.
    private var waiting: Question?
    /// The rig iOS was still connected or connecting to when it relaunched the app.
    private var restored: CBPeripheral?

    init(store: RoomStore) {
        self.store = store
        super.init()
        central = CBCentralManager(delegate: self, queue: nil,
                                   options: [CBCentralManagerOptionShowPowerAlertKey: false,
                                             CBCentralManagerOptionRestoreIdentifierKey: Self.restoreID])
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

    /// Written to the question characteristic (PROTOCOL.md section 5a); no answer comes back.
    /// Sent on every connect too, so a rig that restarted gets it again.
    func send(voice: VoiceSettings) {
        guard let data = voice.encoded(), let peripheral, peripheral.state == .connected,
              let characteristic = questionCharacteristic else { return }
        peripheral.writeValue(data, for: characteristic, type: .withResponse)
    }

    /// Where the person sits, written like the voice settings: raw JSON on the question
    /// characteristic, no answer. The rig pushes a state with the new `view` instead.
    /// Sent on every connect too, so a rig that restarted turns the map again; a reset to the
    /// rig's default goes out on one connect only.
    func send(orient: OrientSettings) {
        guard let data = orient.encoded(), let peripheral, peripheral.state == .connected,
              let characteristic = questionCharacteristic else { return }
        peripheral.writeValue(data, for: characteristic, type: .withResponse)
    }

    func stop() {
        stopped = true
        chooseTask?.cancel()
        retryTask?.cancel()
        watchTask?.cancel()
        statsTask?.cancel()
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

    /// Picks up where iOS left off after relaunching the app: a live link is set up again,
    /// a pending connect is left to complete, and anything else starts over.
    private func adopt(_ p: CBPeripheral) {
        guard !stopped else { return }
        switch p.state {
        case .connected:
            peripheral = p
            p.delegate = self
            store?.linkChanged(.connecting)
            connected(p)
        case .connecting:
            peripheral = p
            p.delegate = self
            store?.linkChanged(.connecting)
        default:
            findRig()
        }
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
        let reason = disconnectReason ?? error?.localizedDescription ?? "the rig disconnected"
        disconnectReason = nil
        log.info("disconnected: \(reason, privacy: .public)")
        let wasWorking = questionCharacteristic != nil
        peripheral = nil
        questionCharacteristic = nil
        meter.disconnected(reason: reason)
        publishStats(now: true)
        watchTask?.cancel()
        store?.linkChanged(.searching)
        // A working link that dropped: try again straight away. The pending connect to the remembered
        // rig completes as soon as it advertises again. Only failed attempts back off.
        if wasWorking {
            backoff = 0.5
            findRig()
        } else {
            retryLater()
        }
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
        var found: [CBUUID: CBCharacteristic] = [:]
        for c in service.characteristics ?? [] { found[c.uuid] = c }
        questionCharacteristic = found[RigGATT.question]
        // PROTOCOL.md section 3: read status first, which tells the bridge this link's MTU, so the
        // first snapshot comes in as few chunks as the link allows. CoreBluetooth runs these in order.
        if let status = found[RigGATT.status] { p.readValue(for: status) }
        for uuid in RigGATT.incoming {
            if let c = found[uuid] { p.setNotifyValue(true, for: c) }
        }
        store?.linkChanged(.connected)
        meter.connected(mtu: p.maximumWriteValueLength(for: .withoutResponse) + 3)
        publishStats(now: true)
        if let question = questionCharacteristic {
            for data in Self.connectWrites(voice: Speaker.voiceSettings, orient: Seat.orientForConnect()) {
                p.writeValue(data, for: question, type: .withResponse)
            }
        }
        if let waiting { send(waiting) }
        watch(p)
    }

    /// What to write to the question characteristic on every connect, in order: the hello first,
    /// so the bridge knows it may compress before it sends anything big, then voice and seat.
    static func connectWrites(voice: VoiceSettings, orient: OrientSettings?) -> [Data] {
        [Hello.current.encoded(), voice.encoded(), orient?.encoded()].compactMap { $0 }
    }

    /// Reconnects when nothing has arrived for `staleAfter` on a link that still looks connected.
    private func watch(_ p: CBPeripheral) {
        let id = p.identifier
        lastHeard = .now
        watchTask?.cancel()
        watchTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled, let current = self.peripheral, current.identifier == id else { return }
                if ContinuousClock.now - self.lastHeard > Self.staleAfter {
                    self.log.info("nothing from the rig for \(Self.staleAfter.components.seconds) s; reconnecting")
                    self.disconnectReason = Self.watchdogReason
                    self.central.cancelPeripheralConnection(current)
                    return
                }
            }
        }
    }

    // MARK: Incoming

    private func received(_ value: Data, on uuid: CBUUID) {
        lastHeard = .now
        defer { publishStats() }
        // A status read may come back as plain JSON rather than a framed chunk.
        if uuid == RigGATT.status, value.first == UInt8(ascii: "{"), let status = Wire.decode(RigStatus.self, from: value) {
            store?.receive(status: status)
            return
        }
        guard let channel = RigGATT.channel(for: uuid) else { return }
        let droppedBefore = meter.stats[channel].dropped
        let message = meter.receive(value, on: channel)
        let dropped = meter.stats[channel].dropped
        if dropped > droppedBefore {
            log.notice("dropped an incomplete message on \(uuid.uuidString, privacy: .public) (\(dropped) so far)")
        }
        guard let message else { return }
        switch channel {
        case .answer:
            guard let answer = Wire.decode(Answer.self, from: message.data) else { return malformed(message, channel) }
            store?.receive(answer: answer)
        case .state:
            guard let state = Wire.decode(Snapshot.self, from: message.data) else { return malformed(message, channel) }
            meter.stateDecoded(tx: state.tx, chunks: message.chunks)
            store?.receive(state: state)
        case .status:
            guard let status = Wire.decode(RigStatus.self, from: message.data) else { return malformed(message, channel) }
            store?.receive(status: status)
        }
    }

    private func malformed(_ message: Reassembler.Message, _ channel: LinkChannel) {
        meter.malformed(on: channel)
        log.error("dropped malformed JSON on \(String(describing: channel), privacy: .public) (\(message.data.count) bytes)")
    }

    /// Hands the diagnostics to the store, at most once per `statsEvery`; a change in between
    /// goes out when the second is up.
    private func publishStats(now: Bool = false) {
        let clock = ContinuousClock.now
        if now || lastStatsPublish.map({ clock - $0 >= Self.statsEvery }) ?? true {
            statsTask?.cancel()
            statsTask = nil
            lastStatsPublish = clock
            store?.receive(linkStats: meter.stats)
            return
        }
        guard statsTask == nil, let last = lastStatsPublish else { return }
        statsTask = Task { [weak self] in
            try? await Task.sleep(until: last + Self.statsEvery, clock: .continuous)
            guard let self, !Task.isCancelled else { return }
            self.statsTask = nil
            self.lastStatsPublish = .now
            self.store?.receive(linkStats: self.meter.stats)
        }
    }
}

// CoreBluetooth calls back on the main queue (queue: nil above).

extension RoomLink: CBCentralManagerDelegate {
    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        let state = central.state
        MainActor.assumeIsolated {
            switch state {
            case .poweredOn:
                if let p = restored {
                    restored = nil
                    adopt(p)
                } else {
                    findRig()
                }
            case .poweredOff: store?.linkChanged(.bluetoothOff)
            case .unauthorized: store?.linkChanged(.unauthorized)
            case .unsupported: store?.linkChanged(.unsupported)
            default: break
            }
            if state != .poweredOn {
                meter.disconnected(reason: "Bluetooth turned off")
                publishStats(now: true)
                peripheral = nil
                questionCharacteristic = nil
            }
        }
    }

    /// Called before `centralManagerDidUpdateState` when iOS relaunches the app for the rig.
    nonisolated func centralManager(_ central: CBCentralManager, willRestoreState dict: [String: Any]) {
        let rig = (dict[CBCentralManagerRestoredStatePeripheralsKey] as? [CBPeripheral])?.first
        MainActor.assumeIsolated {
            log.info("restored by iOS: \(rig == nil ? "no rig" : "rig", privacy: .public)")
            restored = rig
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

    nonisolated func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic, error: Error?) {
        guard let error else { return }
        let uuid = characteristic.uuid.uuidString
        let message = error.localizedDescription
        MainActor.assumeIsolated { log.error("subscribing to \(uuid, privacy: .public) failed: \(message, privacy: .public)") }
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
            meter.resetReassembly()
            peripheral.discoverServices([RigGATT.service])
        }
    }
}
