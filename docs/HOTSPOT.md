# Switching the rig to the phone hotspot (and back)

Grok needs the internet: room naming ("is it one of these?"), open answers and the thinking cue's
follow-up all stop when the venue Wi-Fi drops. Offline, the rig still answers from the rules (the
table), and room answers already made keep working, but no new room object gets a name (spec 0010).
Test this switch once the night before and once in the morning (§8, step 3).

**Do it at the rig's keyboard, or over USB-C (`ssh guru@192.168.55.1`).** An SSH session over Wi-Fi
(`10.90.84.178`) drops when the Wi-Fi changes, and on the hotspot the Jetson gets a new address (the
phone's hotspot screen lists it, or run `hostname -I` at the rig). The room app keeps running: it does
not need a restart.

## Before the expo (once)

1. The phone: turn on the hotspot. Note its name (SSID) and password. 2.4 GHz is fine; keep the phone
   plugged in, near the rig and not in someone's pocket.
2. The Jetson: find the Wi-Fi interface and the venue connection's name:

   ```bash
   nmcli device status                                      # the TYPE wifi row: e.g. wlP1p1s0 on the Orin Nano
   nmcli -t -f NAME,TYPE,DEVICE connection show --active    # the venue Wi-Fi's connection NAME
   ```

3. Save the hotspot as a connection (asks for the sudo password):

   ```bash
   sudo nmcli device wifi connect "<hotspot SSID>" password "<password>" ifname <wifi interface> name askroom-hotspot
   sudo nmcli connection modify askroom-hotspot connection.autoconnect no   # only when we say so
   sudo nmcli connection up "<venue connection NAME>"                     # back to the venue for now
   ```

## Switch to the hotspot

```bash
sudo nmcli connection up askroom-hotspot
```

## Verify (both networks, every time)

```bash
nmcli -t -f ACTIVE,SSID device wifi | grep '^yes'              # which network we're on
ping -c 2 -W 2 1.1.1.1                                          # internet at all
scripts/dock.sh python3 demo_check.py --live --only 6 9 13      # network, clock (NTP), Grok with round-trip ms
scripts/room_app.sh status                                      # the app still up
```

- Check 13 prints `Grok reachable, N ms round trip`. Over a phone hotspot expect 150-600 ms; over
  1500 ms it says so (answers will lean on the thinking cue).
- The running app notices within `net.interval_s` (5 s): the dashboard's online dot turns green and
  the app re-warms its Grok connection by itself (`main.warm_on_connect`).
- Ask one room question that needs Grok (put an object in a zone and ask where it is): the answer
  names the zone.
- The clock: check 9 must pass. The hotspot gives NTP; if the Jetson was offline all night it may
  have been behind (`timedatectl` shows `System clock synchronized: yes`).

## Switch back to the venue Wi-Fi

```bash
sudo nmcli connection up "<venue connection NAME>"
```

Then the same **Verify** block. If the venue Wi-Fi is unreliable during judging, stay on the hotspot:
it is the tested path.

## If something goes wrong

- `nmcli connection up` fails with "Secrets were required": the password is wrong; delete and redo
  step 3 (`sudo nmcli connection delete askroom-hotspot`).
- Online but check 13 fails with HTTP 401/403: the key, not the network (`XAI_API_KEY` in `.env`).
- Check 13 times out on the hotspot: the phone may have mobile data off, or a captive portal on the
  venue network. Open any page from a laptop on the same network.
- Lost SSH after the switch: expected. Reconnect to the new address, or use USB-C.
