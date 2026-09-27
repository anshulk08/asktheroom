# Ask the Room: phone ↔ rig over Bluetooth LE

The iPhone app talks straight to the Jetson over BLE, one phone to one rig, with no cloud and no Wi-Fi.
You ask a question on the phone. The rig speaks the answer and the laser points, the same as a question
asked from the dashboard. The phone gets the answer text and the target position, and keeps a live copy
of the table state.

```
iPhone (CoreBluetooth central)  ──BLE──  ble_bridge.py (Jetson host, BlueZ peripheral)  ──HTTP──  room app :8000 (container)
```

| File | What it is |
|---|---|
| `PROTOCOL.md` | The wire protocol: UUIDs, framing, and every JSON key. The iOS app implements this. |
| `bridge/bleproto.py` | Protocol reference implementation (framing, reassembly, compaction). Python 3.10, stdlib only |
| `bridge/ble_bridge.py` | The Jetson bridge: GATT server and advertisement over D-Bus, and the relay to `/ask` and `/state` |
| `bridge/run_bridge.sh` | start / stop / restart / status / log, in the background, no sudo |
| `bridge/askroom-ble.service` | Optional systemd unit (runs as root; see section 2) |
| `bridge/test_client.py` | Mac stand-in for the phone (`bleak`): scan, connect, ask, and measure |
| `../tests/test_mobile_protocol.py` | Unit tests (framing, compaction, bridge logic with HTTP and BLE faked) |

The iOS app is being built separately (teammate with Xcode) against `PROTOCOL.md`.

## 1. Install and run the bridge on the Jetson

The bridge runs on the Jetson host, not in the container. It needs nothing beyond JetPack 6: system
`python3` 3.10, `python3-dbus`, `python3-gi` and BlueZ 5.64 are all present. No pip packages are needed,
which matters because the Jetson has no internet.

From the Mac (the Jetson is `192.168.55.1` over USB):

```bash
rsync -a --exclude __pycache__ ~/askroom/mobile/ guru@192.168.55.1:askroom/mobile/
ssh guru@192.168.55.1
```

On the Jetson:

```bash
# 1. The Realtek workaround. Once after every boot, or after Bluetooth restarts. Without it, the phone
#    connects and then hangs for 30 s and disconnects. See PROTOCOL.md section 10.
sudo hcitool -i hci0 cmd 0x08 0x0001 DF 1F 0A 00 00 00 00 00
#    The last line printed should be "02 01 20 00" or "01 01 20 00". The final 00 means success.

# 2. Start the bridge. It relays to the room app on localhost:8000, which must already be running.
~/askroom/mobile/bridge/run_bridge.sh start      # log: ~/askroom/data/ble_bridge.log
~/askroom/mobile/bridge/run_bridge.sh status
~/askroom/mobile/bridge/run_bridge.sh log        # tail -f
~/askroom/mobile/bridge/run_bridge.sh stop
```

A healthy start looks like this:

```
askroom.ble INFO room app is up
askroom.ble INFO GATT application registered on /org/bluez/hci0 (service 8a1e0001-…)
askroom.ble INFO advertising as 'AskTheRoom' with service 8a1e0001-… (interval (100, 150) ms)
```

A `WARNING LE event mask NOT set … Operation not permitted` line is expected when the bridge runs as
`guru`. The bridge tries the workaround itself, but that needs root. It is harmless once you have run the
`sudo hcitool` line above.

**Permissions:** `guru` does not need to be in the `bluetooth` group, and no D-Bus policy change is needed.
Ubuntu's `/etc/dbus-1/system.d/bluetooth.conf` lets anyone send to `org.bluez`, and both
`RegisterApplication` and `RegisterAdvertisement` succeed as `guru` (checked on the rig). The only root
step is the `hcitool` line.

## 2. Keep it running across reboots (optional, one sudo)

Pick **one** of these. Don't run both, or two copies will advertise.

**A. systemd, as root.** This handles reboots, Bluetooth restarts and adapter power cycles on its own,
because a root bridge can re-apply the event mask itself.

```bash
~/askroom/mobile/bridge/run_bridge.sh stop
sudo cp ~/askroom/mobile/bridge/askroom-ble.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now askroom-ble
journalctl -u askroom-ble -f
```

**B. Stay as guru, and let it fix the mask itself.** Give `hcitool` the raw-HCI capability once, then use
`run_bridge.sh`:

```bash
sudo setcap cap_net_raw+ep /usr/bin/hcitool     # untested on the rig; if the bridge log still says
                                                # "Operation not permitted", use A or the sudo line
```

## 3. Test from the Mac without an iPhone

```bash
cd ~/askroom
.venv/bin/pip install bleak                       # already in .venv
.venv/bin/python mobile/bridge/test_client.py                          # scan, connect, 3 questions
.venv/bin/python mobile/bridge/test_client.py -q "where are my keys?" --watch 20 -v
.venv/bin/python mobile/bridge/test_client.py --reconnects 3 --json
```

The client does what the phone does. It scans by service UUID, connects, reads `status` (which tells
the bridge the MTU), subscribes to all three notify characteristics and reassembles them, then asks.
It prints the connect time, the MTU, the snapshot size and delivery time, and each question's round
trip. macOS asks once for Bluetooth permission for the terminal app. Note that the questions really are
spoken on the rig and move the laser.

Unit tests: `.venv/bin/python -m pytest -q tests/test_mobile_protocol.py`

## 4. The iPhone side (for whoever runs the app)

1. Install Xcode, open the project, and under Signing & Capabilities set your **Team**. Change the
   bundle id if it clashes.
2. On the iPhone, go to Settings → Privacy & Security → **Developer Mode** → On (the phone restarts),
   then trust the Mac when prompted.
3. Plug in the phone, select it as the run destination, and Run. Allow Bluetooth when asked.
   `NSBluetoothAlwaysUsageDescription` must be in Info.plist.
4. The app scans for service `8A1E0001-6B7F-4C2B-9E3A-2F5D7C1A0001` and connects. It should show
   `status.app == "up"` and a state snapshot within about a second.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Phone or Mac connects, then about 30 s later disconnects without discovering services (bleak: `BleakError: disconnected`) | The event-mask workaround is missing. It resets on reboot, `systemctl restart bluetooth`, or an adapter power cycle. Run the `sudo hcitool …` line again (section 1) |
| Nothing found when scanning | Run `run_bridge.sh status` and check the log for `advertising as 'AskTheRoom'`. Run `bluetoothctl show` and check it says `Powered: yes` and `ActiveInstances: 0x01`. Make sure only one bridge is running (`pgrep -af ble_bridge`) |
| Found, but the name is `guru-desktop` or blank | Normal. Scan by service UUID; the name only arrives in the scan response |
| After changing UUIDs or characteristics, the phone sees old or missing characteristics | iOS caches the GATT table. Forget the device if it is listed under Settings → Bluetooth, toggle Bluetooth off and on, and relaunch the app. The Mac caches too: toggle Bluetooth |
| Answer is "The room isn't running right now." | The container app isn't serving `localhost:8000` (`curl localhost:8000/healthz`). The bridge keeps running and recovers by itself when the app comes back |
| Answers arrive but the laser doesn't move | `status.laser_cal` is false (no `laser_cal.json`), or the actuator in `config.yaml` is `fake` |
| `status.cal` false, and every object `X` | The table isn't calibrated (ArUco markers 0–3 not all visible) |
| bluetoothd restarted | The bridge re-registers on its own (`bluetoothd is on the bus … registering`). Re-apply the event mask unless you use systemd (2A) |
