#!/usr/bin/env python3
"""Ask the Room BLE bridge: a BlueZ GATT peripheral that lets the iPhone app talk to the rig
directly over Bluetooth LE (mobile/PROTOCOL.md). Runs on the Jetson HOST (not in the container),
next to the app's HTTP API on localhost:8000.

    question write   -> POST /ask {"text", "source": "dashboard"} in a worker thread -> answer notify
    GET /state (4 Hz) -> state notify: on change at most 2 Hz, heartbeat every 5 s, and right after
                        the phone subscribes
    status           -> read (unframed JSON) + notify on change (app up/down, fps, online, calibration)

Source "dashboard" is the /ask source that is spoken AND aimed (main.Room.ask_and_act answers every
source except "sms"/"n8n" out loud and moves the laser; server/app.py accepts only
{"dashboard", "n8n"}).

Needs only what JetPack 6 ships: python3 (3.10), python3-dbus, python3-gi, BlueZ 5.64.

    python3 mobile/bridge/ble_bridge.py                 # foreground, logs to stderr
    python3 mobile/bridge/ble_bridge.py --log data/ble_bridge.log
    mobile/bridge/run_bridge.sh start|stop|status|log   # background
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bleproto as P  # noqa: E402

log = logging.getLogger("askroom.ble")

ASK_SOURCE = "phone"          # spoken + laser, like "dashboard" (see module docstring); /state 'answers' tags it
REPO_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BUSY_TEXT = "I'm still answering your last question."
SLOW_TEXT = "Sorry, that took too long. Please ask again."
FAIL_TEXT = "Sorry, something went wrong answering that."
CHARS = ("answer", "state", "status")


# ---------------------------------------------------------------- HTTP to the room app

class RoomDown(Exception):
    """The app's HTTP API is not reachable (container stopped, still starting, crashed)."""


class RoomHTTP:
    def __init__(self, base: str = "http://127.0.0.1:8000"):
        self.base = base.rstrip("/")

    def _open(self, req, timeout: float):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, OSError) as ex:
            if isinstance(ex, TimeoutError) or "timed out" in str(ex):
                raise TimeoutError(str(ex)) from ex
            raise RoomDown(str(ex)) from ex

    def get_json(self, path: str, timeout: float = 1.0):
        return self._open(urllib.request.Request(self.base + path), timeout)

    def post_json(self, path: str, body: dict, timeout: float = 12.0):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        return self._open(req, timeout)


def read_config(repo: str) -> tuple[tuple[float, float], str, str]:
    """(table size cm, table_cal path, laser_cal path) from the repo's config.yaml, with the device's
    config.local.yaml over it (core/config.py)."""
    size, tcal, lcal = (90.0, 60.0), "table_cal.json", "laser_cal.json"
    path = os.path.join(repo, "config.yaml")
    try:
        import yaml  # present on JetPack; the regex below covers its absence
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        local = os.path.join(repo, "config.local.yaml")
        if os.path.isfile(local):
            with open(local) as f:
                over = yaml.safe_load(f) or {}
            for k in ("table", "paths"):
                cfg[k] = dict(cfg.get(k) or {}, **(over.get(k) or {}))
        s = (cfg.get("table") or {}).get("size_cm") or size
        size = (float(s[0]), float(s[1]))
        paths = cfg.get("paths") or {}
        tcal, lcal = paths.get("table_cal", tcal), paths.get("laser_cal", lcal)
    except ImportError:
        try:
            txt = open(path).read()
            m = re.search(r"size_cm:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", txt)
            if m:
                size = (float(m.group(1)), float(m.group(2)))
        except OSError:
            pass
    except Exception as ex:
        log.warning("config.yaml not read (%s); table %sx%s cm", ex, *size)
    tcal = os.path.join(repo, tcal)
    try:                                   # the calibration's own size wins (one-tag mode measures the area)
        with open(tcal) as f:
            s = json.load(f).get("size_cm")
        if s:
            size = (float(s[0]), float(s[1]))
    except (OSError, ValueError, TypeError, IndexError):
        pass
    return size, tcal, os.path.join(repo, lcal)


# ---------------------------------------------------------------- controller workaround

# The Jetson's Realtek controller (LE features BD 5F 66 00: extended advertising, no LL Privacy)
# reports a connection made to an extended-advertising set ONLY as "LE Enhanced Connection
# Complete". Kernel 5.15 unmasks that event only for LL-Privacy controllers, so the connection
# complete never reaches the kernel: the phone connects, its first ATT request goes unanswered and
# it disconnects after 30 s (seen in btmon: "LE Channel Selection Algorithm" for an unknown handle,
# ACL RX, no ACL TX). Fix: re-send LE Set Event Mask with the kernel's own bits plus bit 9
# (Enhanced Connection Complete). Needs root or CAP_NET_RAW; lost whenever the adapter re-inits.
LE_EVENT_MASK = "DF1F0A0000000000"


def le_event_mask_cmd(hci: str, mask_hex: str = LE_EVENT_MASK) -> list[str]:
    b = bytes.fromhex(mask_hex)
    if len(b) != 8:
        raise ValueError("LE event mask must be 8 bytes (16 hex digits)")
    return ["hcitool", "-i", hci, "cmd", "0x08", "0x0001"] + [f"{x:02X}" for x in b]


def apply_le_event_mask(hci: str, mask_hex: str = LE_EVENT_MASK, run=None) -> bool:
    """Send LE Set Event Mask via hcitool. True when the controller answered status 0x00."""
    import subprocess
    run = run or subprocess.run
    cmd = le_event_mask_cmd(hci, mask_hex)
    try:
        r = run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as ex:
        log.warning("LE event mask not set (%s); run once: sudo %s", ex, " ".join(cmd))
        return False
    out = (r.stdout or "") + (r.stderr or "")
    # hcitool prints the Command Complete as "> HCI Event: 0x0e plen 4\n  01 01 20 00": last byte = status
    tail = out.strip().split()[-4:] if out.strip() else []
    ok = r.returncode == 0 and len(tail) == 4 and [t.upper() for t in tail[1:]] == ["01", "20", "00"]
    if ok:
        log.info("LE event mask set on %s (%s): Enhanced Connection Complete unmasked", hci, mask_hex)
    else:
        log.warning("LE event mask NOT set on %s (exit %s: %s). Phones will fail to connect on this "
                    "Realtek controller until you run: sudo %s", hci, r.returncode,
                    " ".join(out.split())[-160:], " ".join(cmd))
    return ok


# ---------------------------------------------------------------- bridge logic (no D-Bus)

class BridgeCore:
    """Everything except BlueZ. emit(char, chunks) hands framed notification values to the BLE layer
    (thread safe; the D-Bus layer marshals it onto the GLib loop). All HTTP goes through `http`."""

    def __init__(self, http, emit: Callable[[str, list], None], table_cm=(90.0, 60.0),
                 table_cal: Optional[str] = None, laser_cal: Optional[str] = None,
                 source: str = ASK_SOURCE, clock: Callable[[], float] = time.time,
                 mono: Callable[[], float] = time.monotonic, state_min_s: float = 0.5,
                 heartbeat_s: float = 5.0, status_min_s: float = 1.0, ask_timeout_s: float = 12.0,
                 default_mtu: int = P.DEFAULT_MTU):
        self.http, self.emit = http, emit
        self.table_cm = (float(table_cm[0]), float(table_cm[1]))
        self.table_cal, self.laser_cal = table_cal, laser_cal
        self._cal_mtime: Optional[float] = None      # table_cal.json we last took the size from
        self.source = source
        self.seen = {"answers": None, "notices": None}   # newest seq / notice id already handled (None: first poll)
        self.clock, self.mono = clock, mono
        self.state_min_s, self.heartbeat_s, self.status_min_s = state_min_s, heartbeat_s, status_min_s
        self.ask_timeout_s = ask_timeout_s
        self.default_mtu = int(default_mtu)
        self.lock = threading.RLock()
        self.counter = P.MsgCounter()
        self.notifying = {c: False for c in CHARS}
        self.mtus: dict[str, int] = {}
        self.app_up = False
        self.latest_state: Optional[dict] = None       # raw WorldState from GET /state
        self.last_sent_state: Optional[dict] = None    # compact, as sent
        self.state_sent_at = float("-inf")
        self.force_state = False
        self.status: dict = P.status_msg(False, None, False, False)
        self.last_sent_status: Optional[dict] = None
        self.status_sent_at = float("-inf")
        self.asks: "queue.Queue[tuple[int, str, float]]" = queue.Queue(maxsize=1)
        self.stop_ev = threading.Event()
        self.threads: list[threading.Thread] = []
        self.stats = {"questions": 0, "answers": 0, "state_msgs": 0, "state_bytes": 0, "status_msgs": 0,
                      "poll_fail": 0}

    # -- link facts from the BLE layer

    def note_mtu(self, device: Optional[str], mtu) -> None:
        try:
            m = int(mtu)
        except (TypeError, ValueError):
            return
        if m < P.MIN_MTU:
            return
        key = str(device or "?")
        with self.lock:
            old = self.mtu()
            if self.mtus.get(key) == m:
                return
            self.mtus[key] = m
            log.info("MTU %d from %s", m, key)
            if m < old and self.notifying["state"]:
                self.force_state = True          # the last snapshot may have been cut: resend

    def forget_device(self, device: str) -> None:
        with self.lock:
            self.mtus.pop(str(device), None)

    def mtu(self) -> int:
        with self.lock:
            return min(self.mtus.values()) if self.mtus else self.default_mtu

    def set_notifying(self, char: str, on: bool) -> None:
        with self.lock:
            self.notifying[char] = bool(on)
            log.info("%s notifications %s", char, "on" if on else "off")
            if on and char == "state":
                self.force_state = True
            if on and char == "status":
                self.last_sent_status = None
        if on and char in ("state", "status"):
            self.push_updates()

    # -- sending

    def _send(self, char: str, obj: dict) -> list[bytes]:
        payload = P.dumps(obj)
        with self.lock:
            chunks = P.frame(self.counter.next(char), payload, self.mtu())
            self.emit(char, chunks)
        return chunks

    # -- state and status

    def calibrated(self, state: Optional[dict]) -> tuple[bool, bool]:
        tcal = bool(self.table_cal and os.path.exists(self.table_cal))
        if isinstance(state, dict) and state.get("t") is not None:
            tcal = True                          # perception only updates the world once calibrated
        lcal = bool(self.laser_cal and os.path.exists(self.laser_cal))
        return tcal, lcal

    def poll_once(self) -> None:
        """One GET /state, then whatever notifications are due."""
        try:
            body = self.http.get_json("/state", timeout=1.0)
            st = body.get("state") if isinstance(body, dict) else None
            if not isinstance(st, dict):
                raise ValueError("no state in /state")
            with self.lock:
                if not self.app_up:
                    log.info("room app is up")
                self.app_up, self.latest_state = True, st
            self.room_messages(body)
        except Exception as ex:
            with self.lock:
                self.stats["poll_fail"] += 1
                if self.app_up:
                    log.warning("room app unreachable: %s", ex)
                self.app_up = False
        self.push_updates()

    def room_messages(self, body: dict) -> None:
        """New answers from the room (not the phone's own) and new notices, on the answer characteristic
        with id null (PROTOCOL.md 6a). What happened before the first poll is not replayed."""
        def rows(key, idk):
            return [r for r in (body.get(key) or []) if isinstance(r, dict) and isinstance(r.get(idk), int)
                    and not isinstance(r.get(idk), bool)]
        out = []
        for key, idk in (("answers", "seq"), ("notices", "id")):
            rs = sorted(rows(key, idk), key=lambda r: r[idk])
            last = self.seen[key]
            if rs:
                self.seen[key] = max(rs[-1][idk], last or 0)
            if last is None:
                self.seen[key] = self.seen[key] or 0
                continue
            for r in rs:
                if r[idk] <= last or not r.get("text"):
                    continue
                target = P.target_of(self.latest_state, r.get("point_at"))
                if key == "answers" and r.get("src") != self.source:
                    out.append(P.room_msg(str(r.get("src") or "voice"), r["text"], r.get("point_at"),
                                          r.get("action"), target, q=str(r.get("q") or "")))
                elif key == "notices":
                    out.append(P.room_msg("notice", r["text"], r.get("point_at"), r.get("action"), target,
                                          nid=r["id"], kind=r.get("kind")))
        with self.lock:
            live = self.notifying["answer"]
        for m in out if live else []:
            self._send("answer", m)
            self.stats["room_msgs"] = self.stats.get("room_msgs", 0) + 1

    def compact_now(self) -> dict:
        with self.lock:
            return P.compact_state(self.latest_state, self.current_table_cm(), self.clock())

    def current_table_cm(self) -> tuple[float, float]:
        """The table size, re-read when table_cal.json changes: a one-tag recalibration measures a new
        tracked area, and the phone's map should follow it without restarting the bridge."""
        try:
            m = os.path.getmtime(self.table_cal) if self.table_cal else None
        except OSError:
            m = None
        if m is not None and m != self._cal_mtime:
            self._cal_mtime = m
            try:
                with open(self.table_cal) as f:
                    s = json.load(f).get("size_cm")
                if s:
                    self.table_cm = (float(s[0]), float(s[1]))
            except (OSError, ValueError, TypeError, IndexError, AttributeError):
                pass
        return self.table_cm

    def push_updates(self) -> None:
        now = self.mono()
        with self.lock:
            tcal, lcal = self.calibrated(self.latest_state if self.app_up else None)
            self.status = P.status_msg(self.app_up, self.latest_state, tcal, lcal)
            if self.notifying["status"] and P.status_changed(self.last_sent_status, self.status) \
                    and (self.last_sent_status is None or now - self.status_sent_at >= self.status_min_s
                         or self.last_sent_status.get("app") != self.status["app"]):
                self._send("status", self.status)
                self.last_sent_status, self.status_sent_at = dict(self.status), now
                self.stats["status_msgs"] += 1
            if not self.notifying["state"]:
                return
            cur = self.compact_now()
            due = self.force_state or now - self.state_sent_at >= self.heartbeat_s or (
                now - self.state_sent_at >= self.state_min_s and P.state_changed(self.last_sent_state, cur))
            if due:
                chunks = self._send("state", cur)
                self.last_sent_state, self.state_sent_at, self.force_state = cur, now, False
                self.stats["state_msgs"] += 1
                self.stats["state_bytes"] = sum(len(c) - P.HEADER_LEN for c in chunks)

    def status_read(self) -> bytes:
        with self.lock:
            return P.dumps(self.status)

    # -- questions

    def on_question(self, value: bytes) -> None:
        """question write (called on the GLib thread: must not block)."""
        t0 = self.mono()
        qid, text, err = P.parse_question(bytes(value))
        self.stats["questions"] += 1
        if err:
            log.info("question %d rejected: %s", qid, err)
            self._send("answer", P.answer_msg(qid, False, err, ms=0))
            return
        log.info("question %d: %r", qid, text)
        try:
            self.asks.put_nowait((qid, text, t0))
        except queue.Full:
            self._send("answer", P.answer_msg(qid, False, BUSY_TEXT, ms=0))

    def answer(self, qid: int, text: str, t0: Optional[float] = None) -> dict:
        """POST /ask and build the answer message (blocking; worker thread)."""
        t0 = self.mono() if t0 is None else t0
        try:
            r = self.http.post_json("/ask", {"text": text, "source": self.source}, timeout=self.ask_timeout_s)
            point_at, action = r.get("point_at"), r.get("action")
            target = None
            if point_at:
                try:
                    st = self.http.get_json("/state", timeout=1.0).get("state")
                    with self.lock:
                        self.latest_state = st or self.latest_state
                except Exception:
                    st = self.latest_state
                target = P.target_of(st, point_at)
            msg = P.answer_msg(qid, True, str(r.get("text") or ""), point_at, action, target,
                               round((self.mono() - t0) * 1000))
        except RoomDown:
            msg = P.answer_msg(qid, False, P.ROOM_DOWN_TEXT, ms=round((self.mono() - t0) * 1000))
        except TimeoutError:
            msg = P.answer_msg(qid, False, SLOW_TEXT, ms=round((self.mono() - t0) * 1000))
        except Exception as ex:
            log.warning("ask failed: %s", ex)
            msg = P.answer_msg(qid, False, FAIL_TEXT, ms=round((self.mono() - t0) * 1000))
        return msg

    def ask_worker(self) -> None:
        while not self.stop_ev.is_set():
            try:
                qid, text, t0 = self.asks.get(timeout=0.5)
            except queue.Empty:
                continue
            msg = self.answer(qid, text, t0)
            self.stats["answers"] += 1
            log.info("answer %d (%d ms, ok=%s): %s point_at=%s target=%s", qid, msg["ms"], msg["ok"],
                     msg["text"], msg["point_at"], msg["target"])
            self._send("answer", msg)
            self.push_updates()                 # the laser field likely changed

    def poll_loop(self, period: float = 0.25) -> None:
        while not self.stop_ev.is_set():
            t0 = self.mono()
            try:
                self.poll_once()
            except Exception:
                log.exception("poll failed")
            self.stop_ev.wait(max(0.05, period - (self.mono() - t0)))

    def start(self, poll_s: float = 0.25) -> None:
        for target, name, args in ((self.poll_loop, "poll", (poll_s,)), (self.ask_worker, "ask", ())):
            t = threading.Thread(target=target, args=args, name=name, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        self.stop_ev.set()


# ---------------------------------------------------------------- BlueZ D-Bus layer

BLUEZ = "org.bluez"
OM_IFACE = "org.freedesktop.DBus.ObjectManager"
PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_MGR = "org.bluez.GattManager1"
ADV_MGR = "org.bluez.LEAdvertisingManager1"
SVC_IFACE = "org.bluez.GattService1"
CHRC_IFACE = "org.bluez.GattCharacteristic1"
ADV_IFACE = "org.bluez.LEAdvertisement1"
DEVICE_IFACE = "org.bluez.Device1"
APP_PATH = "/org/askroom"


def run_ble(core: BridgeCore, adapter_hint: Optional[str] = None, pump_ms: int = 5,
            chunks_per_tick: int = 4, adv_interval_ms: Optional[tuple] = (100, 150),
            event_mask: Optional[str] = LE_EVENT_MASK) -> int:
    import dbus
    import dbus.exceptions
    import dbus.mainloop.glib
    import dbus.service
    from gi.repository import GLib

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    outbox = P.Outbox()
    state = {"adapter": None, "app_ok": False, "adv_ok": False, "retry": None,
             "adv_interval": tuple(adv_interval_ms) if adv_interval_ms else None, "powered": None}
    pump = {"id": None}

    class InvalidArgs(dbus.exceptions.DBusException):
        _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"

    class NotSupported(dbus.exceptions.DBusException):
        _dbus_error_name = "org.bluez.Error.NotSupported"

    class Characteristic(dbus.service.Object):
        def __init__(self, index: int, uuid: str, flags: list, name: str, service):
            self.path = f"{service.path}/char{index}"
            self.uuid, self.flags, self.name, self.service = uuid, flags, name, service
            self.notifying = False
            super().__init__(bus, self.path)

        def props(self) -> dict:
            return {CHRC_IFACE: {"Service": dbus.ObjectPath(self.service.path), "UUID": self.uuid,
                                 "Flags": dbus.Array(self.flags, signature="s")}}

        @dbus.service.method(PROP_IFACE, in_signature="s", out_signature="a{sv}")
        def GetAll(self, interface):
            if interface != CHRC_IFACE:
                raise InvalidArgs()
            return self.props()[CHRC_IFACE]

        @dbus.service.method(CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
        def ReadValue(self, options):
            core.note_mtu(options.get("device"), options.get("mtu"))
            if self.name != "status":
                raise NotSupported()
            value = core.status_read()
            off = int(options.get("offset", 0))
            return dbus.Array([dbus.Byte(b) for b in value[off:]], signature="y")

        @dbus.service.method(CHRC_IFACE, in_signature="aya{sv}")
        def WriteValue(self, value, options):
            core.note_mtu(options.get("device"), options.get("mtu"))
            if self.name != "question":
                raise NotSupported()
            core.on_question(bytes(value))

        @dbus.service.method(CHRC_IFACE)
        def StartNotify(self):
            if not self.notifying:
                self.notifying = True
                core.set_notifying(self.name, True)

        @dbus.service.method(CHRC_IFACE)
        def StopNotify(self):
            if self.notifying:
                self.notifying = False
                outbox.clear(self.name)
                core.set_notifying(self.name, False)

        @dbus.service.signal(PROP_IFACE, signature="sa{sv}as")
        def PropertiesChanged(self, interface, changed, invalidated):
            pass

        def notify(self, value: bytes) -> None:
            self.PropertiesChanged(CHRC_IFACE, {"Value": dbus.Array([dbus.Byte(b) for b in value],
                                                                    signature="y")}, [])

    class Service(dbus.service.Object):
        def __init__(self):
            self.path = APP_PATH + "/service0"
            super().__init__(bus, self.path)
            self.chars = [Characteristic(0, P.QUESTION_UUID, ["write"], "question", self),
                          Characteristic(1, P.ANSWER_UUID, ["notify"], "answer", self),
                          Characteristic(2, P.STATE_UUID, ["notify"], "state", self),
                          Characteristic(3, P.STATUS_UUID, ["read", "notify"], "status", self)]
            self.by_name = {c.name: c for c in self.chars}

        def props(self) -> dict:
            return {SVC_IFACE: {"UUID": P.SERVICE_UUID, "Primary": True,
                                "Characteristics": dbus.Array([dbus.ObjectPath(c.path) for c in self.chars],
                                                              signature="o")}}

        @dbus.service.method(PROP_IFACE, in_signature="s", out_signature="a{sv}")
        def GetAll(self, interface):
            if interface != SVC_IFACE:
                raise InvalidArgs()
            return self.props()[SVC_IFACE]

    class Application(dbus.service.Object):
        def __init__(self, service: Service):
            self.service = service
            super().__init__(bus, APP_PATH)

        @dbus.service.method(OM_IFACE, out_signature="a{oa{sa{sv}}}")
        def GetManagedObjects(self):
            out = {dbus.ObjectPath(self.service.path): self.service.props()}
            for c in self.service.chars:
                out[dbus.ObjectPath(c.path)] = c.props()
            return out

    class Advertisement(dbus.service.Object):
        def __init__(self):
            self.path = APP_PATH + "/adv0"
            super().__init__(bus, self.path)

        def props(self) -> dict:
            # flags (3) + 128-bit service UUID (18) fit the 31-byte advertisement; BlueZ moves the
            # local name into the scan response (iOS scans actively in the foreground).
            props = {"Type": "peripheral",
                     "ServiceUUIDs": dbus.Array([P.SERVICE_UUID], signature="s"),
                     "LocalName": dbus.String(P.LOCAL_NAME),
                     "Discoverable": dbus.Boolean(True)}
            if state["adv_interval"]:
                # BlueZ's default is 1.28 s, which makes the phone take seconds to find the rig
                props["MinInterval"] = dbus.UInt32(state["adv_interval"][0])
                props["MaxInterval"] = dbus.UInt32(state["adv_interval"][1])
            return props

        @dbus.service.method(PROP_IFACE, in_signature="s", out_signature="a{sv}")
        def GetAll(self, interface):
            if interface != ADV_IFACE:
                raise InvalidArgs()
            return self.props()

        @dbus.service.method(ADV_IFACE)
        def Release(self):
            log.info("advertisement released by BlueZ")

    service = Service()
    app = Application(service)
    adv = Advertisement()

    # -- notifications: core threads -> GLib loop -> Outbox -> paced PropertiesChanged

    def drain() -> bool:
        for char, value in outbox.pop(chunks_per_tick):
            ch = service.by_name[char]
            if ch.notifying:
                ch.notify(value)
        if outbox.pending():
            return True
        pump["id"] = None
        return False

    def enqueue(char: str, chunks: list) -> bool:
        if service.by_name[char].notifying:
            outbox.push(char, chunks)
            if pump["id"] is None:
                drain()
                if outbox.pending() and pump["id"] is None:
                    pump["id"] = GLib.timeout_add(pump_ms, drain)
        return False

    core.emit = lambda char, chunks: GLib.idle_add(enqueue, char, list(chunks))

    # -- registration (again whenever bluetoothd restarts)

    def find_adapter() -> Optional[str]:
        om = dbus.Interface(bus.get_object(BLUEZ, "/"), OM_IFACE)
        for path, ifaces in om.GetManagedObjects().items():
            if GATT_MGR in ifaces and ADV_MGR in ifaces and (not adapter_hint or path.endswith(adapter_hint)):
                return str(path)
        return None

    def register() -> bool:
        state["retry"] = None
        try:
            path = find_adapter()
        except dbus.exceptions.DBusException as ex:
            log.error("BlueZ not reachable: %s", ex)
            path = None
        if not path:
            log.error("no Bluetooth LE adapter with GATT + advertising (%s); retrying in 5 s",
                      adapter_hint or "any")
            state["retry"] = GLib.timeout_add_seconds(5, register)
            return False
        state["adapter"] = path
        obj = bus.get_object(BLUEZ, path)
        props = dbus.Interface(obj, PROP_IFACE)
        try:
            if not props.Get("org.bluez.Adapter1", "Powered"):
                log.info("powering on %s", path)
                props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(True))
        except dbus.exceptions.DBusException as ex:
            log.warning("could not power on %s: %s", path, ex)
        if event_mask:
            apply_le_event_mask(path.rsplit("/", 1)[-1], event_mask)

        def app_ok():
            state["app_ok"] = True
            log.info("GATT application registered on %s (service %s)", path, P.SERVICE_UUID)

        def adv_ok():
            state["adv_ok"] = True
            log.info("advertising as %r with service %s (interval %s ms)", P.LOCAL_NAME, P.SERVICE_UUID,
                     state["adv_interval"] or "BlueZ default")

        def failed(what, key):
            def cb(err):
                name = err.get_dbus_name()
                if name == "org.bluez.Error.AlreadyExists":
                    state[key] = True
                    return
                log.error("%s failed: %s: %s", what, name, err.get_dbus_message())
                if what == "RegisterAdvertisement" and state["adv_interval"]:
                    log.warning("retrying the advertisement without MinInterval/MaxInterval")
                    state["adv_interval"] = None
                if name == "org.freedesktop.DBus.Error.AccessDenied":
                    log.error("the D-Bus policy denies this user; see mobile/README.md (permissions)")
                if state["retry"] is None:
                    state["retry"] = GLib.timeout_add_seconds(5, register)
            return cb

        if not state["app_ok"]:
            dbus.Interface(obj, GATT_MGR).RegisterApplication(
                APP_PATH, {}, reply_handler=app_ok, error_handler=failed("RegisterApplication", "app_ok"))
        if not state["adv_ok"]:
            dbus.Interface(obj, ADV_MGR).RegisterAdvertisement(
                adv.path, {}, reply_handler=adv_ok, error_handler=failed("RegisterAdvertisement", "adv_ok"))
        return False

    def owner_changed(owner: str) -> None:
        if owner:
            log.info("bluetoothd is on the bus (%s); registering", owner)
            state["app_ok"] = state["adv_ok"] = False
            for c in service.chars:
                if c.notifying:
                    c.notifying = False
                    core.set_notifying(c.name, False)
            GLib.timeout_add(500, register)
        else:
            log.warning("bluetoothd left the bus; waiting for it")
            state["app_ok"] = state["adv_ok"] = False

    def device_changed(interface, changed, invalidated, path=None):
        if interface != DEVICE_IFACE or "Connected" not in changed:
            return
        if changed["Connected"]:
            log.info("central connected: %s", path)
        else:
            log.info("central disconnected: %s", path)
            core.forget_device(path)

    def adapter_changed(interface, changed, invalidated, path=None):
        if interface != "org.bluez.Adapter1" or "Powered" not in changed:
            return
        on = bool(changed["Powered"])
        log.info("adapter %s powered %s", path, "on" if on else "off")
        if on and event_mask:            # the kernel re-sent its own event mask on power-on
            GLib.timeout_add(500, lambda: apply_le_event_mask(str(path).rsplit("/", 1)[-1], event_mask) and False)

    bus.add_signal_receiver(adapter_changed, dbus_interface=PROP_IFACE, signal_name="PropertiesChanged",
                            arg0="org.bluez.Adapter1", path_keyword="path")
    bus.add_signal_receiver(device_changed, dbus_interface=PROP_IFACE, signal_name="PropertiesChanged",
                            arg0=DEVICE_IFACE, path_keyword="path")
    bus.watch_name_owner(BLUEZ, owner_changed)     # fires now with the current owner -> register()

    loop = GLib.MainLoop()
    import signal as _signal
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, lambda *_: (loop.quit(), False)[1])
    core.start()

    def heartbeat() -> bool:
        s = core.stats
        log.info("alive: app=%s mtu=%d subs=%s q=%d a=%d state_msgs=%d (%d B last) status_msgs=%d",
                 "up" if core.app_up else "down", core.mtu(),
                 ",".join(c for c in CHARS if core.notifying[c]) or "-", s["questions"], s["answers"],
                 s["state_msgs"], s["state_bytes"], s["status_msgs"])
        return True

    GLib.timeout_add_seconds(60, heartbeat)
    try:
        loop.run()
    finally:
        core.stop()
        try:
            if state["adapter"]:
                obj = bus.get_object(BLUEZ, state["adapter"])
                dbus.Interface(obj, ADV_MGR).UnregisterAdvertisement(adv.path)
                dbus.Interface(obj, GATT_MGR).UnregisterApplication(APP_PATH)
        except Exception:
            pass
        log.info("bridge stopped")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the Room BLE bridge (BlueZ GATT peripheral)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="the room app's HTTP API")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="askroom checkout (config.yaml, calibration files)")
    ap.add_argument("--adapter", default=None, help="e.g. hci0 (default: first LE adapter)")
    ap.add_argument("--source", default=ASK_SOURCE, help="/ask source (dashboard = spoken + laser)")
    ap.add_argument("--mtu", type=int, default=P.DEFAULT_MTU, help="assumed ATT MTU until a read/write reports it")
    ap.add_argument("--adv-interval", default="100,150", help="advertising interval min,max ms ('' = BlueZ default 1.28 s)")
    ap.add_argument("--le-event-mask", default=LE_EVENT_MASK,
                    help="LE event mask to re-apply via hcitool at start and after adapter power-on "
                         "(needs root or CAP_NET_RAW on hcitool); 'none' to skip")
    ap.add_argument("--log", default=None, help="also append logs to this file")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers)
    table_cm, tcal, lcal = read_config(args.repo)
    log.info("bridge starting: api %s, table %gx%g cm, source %r", args.url, *table_cm, args.source)
    core = BridgeCore(RoomHTTP(args.url), emit=lambda c, ch: None, table_cm=table_cm, table_cal=tcal,
                      laser_cal=lcal, source=args.source, default_mtu=args.mtu)
    interval = tuple(int(x) for x in args.adv_interval.split(",")) if args.adv_interval else None
    mask = None if str(args.le_event_mask).lower() in ("", "none", "off") else args.le_event_mask
    return run_ble(core, args.adapter, adv_interval_ms=interval, event_mask=mask)


if __name__ == "__main__":
    raise SystemExit(main())
