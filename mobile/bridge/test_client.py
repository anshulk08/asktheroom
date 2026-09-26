#!/usr/bin/env python3
"""Mac stand-in for the iPhone app: talks to the Jetson BLE bridge exactly as the phone does
(mobile/PROTOCOL.md) and measures the link.

    .venv/bin/python mobile/bridge/test_client.py                       # scan, connect, ask 3 questions
    .venv/bin/python mobile/bridge/test_client.py -q "where are my keys?" --watch 20
    .venv/bin/python mobile/bridge/test_client.py --reconnects 3        # disconnect / reconnect cycles
    .venv/bin/python mobile/bridge/test_client.py --json                # machine-readable summary

Needs `bleak` (pip install bleak). macOS asks once for Bluetooth permission for the terminal app.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bleproto as P  # noqa: E402

DEFAULT_QUESTIONS = ["where are my keys?", "where is my wallet?", "what changed?"]


def now() -> float:
    return time.perf_counter()


class Link:
    """One connection's reassemblers and message log."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.asm = {c: P.Reassembler() for c in ("answer", "state", "status")}
        self.first_chunk_t: dict[str, float] = {}
        self.chunks: dict[str, int] = {}
        self.messages: dict[str, list[dict]] = {c: [] for c in self.asm}
        self.waiters: dict[str, list[asyncio.Future]] = {c: [] for c in self.asm}
        self.bad_json = 0

    def handler(self, char: str):
        def on_notify(_sender, data: bytearray):
            t = now()
            if len(data) >= 2 and data[1] == 0:
                self.first_chunk_t[char] = t
                self.chunks[char] = 0
            self.chunks[char] = self.chunks.get(char, 0) + 1
            msg = self.asm[char].feed(bytes(data))
            if msg is None:
                return
            try:
                obj = json.loads(msg.decode("utf-8"))
            except ValueError:
                self.bad_json += 1
                print(f"  ! {char}: {len(msg)} bytes that are not JSON")
                return
            rec = {"t": t, "t0": self.first_chunk_t.get(char, t), "bytes": len(msg),
                   "chunks": self.chunks.get(char, 1), "obj": obj}
            self.messages[char].append(rec)
            if self.verbose:
                print(f"  <- {char} {len(msg)} B in {rec['chunks']} chunk(s): {msg[:160].decode(errors='replace')}")
            for fut in self.waiters[char]:
                if not fut.done():
                    fut.set_result(rec)
            self.waiters[char] = [f for f in self.waiters[char] if not f.done()]
        return on_notify

    async def next(self, char: str, timeout: float, pred=None) -> Optional[dict]:
        end = now() + timeout
        while True:
            fut = asyncio.get_running_loop().create_future()
            self.waiters[char].append(fut)
            try:
                rec = await asyncio.wait_for(fut, max(0.01, end - now()))
            except asyncio.TimeoutError:
                return None
            if pred is None or pred(rec):
                return rec


def describe_state(obj: dict) -> str:
    ents = obj.get("e", [])
    by = {}
    for e in ents:
        by.setdefault(e["s"], []).append(e["n"])
    parts = [f"{s}:{','.join(n)}" for s, n in sorted(by.items())]
    return (f"table {obj.get('table')} online={obj.get('online')} laser={obj.get('laser')} "
            f"{len(ents)} entities [{' '.join(parts)}]")


async def session(args, results: dict, questions: list[str], qid0: int) -> int:
    from bleak import BleakClient, BleakScanner

    t_scan = now()
    dev = await BleakScanner.find_device_by_filter(
        lambda d, adv: P.SERVICE_UUID in [u.lower() for u in (adv.service_uuids or [])],
        timeout=args.scan_timeout)
    if dev is None:
        print(f"no device advertising {P.SERVICE_UUID} within {args.scan_timeout} s")
        results["error"] = "not found"
        return qid0
    scan_s = now() - t_scan
    print(f"found {dev.name!r} ({dev.address}) in {scan_s:.2f} s")
    results.setdefault("scan_s", []).append(round(scan_s, 3))

    link = Link(args.verbose)
    t_conn = now()
    async with BleakClient(dev, timeout=20.0) as client:
        connect_s = now() - t_conn             # includes GATT service discovery
        mtu = client.mtu_size
        print(f"connected + services discovered in {connect_s:.2f} s, MTU {mtu}")
        results.setdefault("connect_s", []).append(round(connect_s, 3))
        results["mtu"] = mtu
        svc = client.services.get_service(P.SERVICE_UUID)
        if svc is None:
            print("service missing after discovery")
            results["error"] = "service missing"
            return qid0
        # read status first: tells the bridge our MTU (options["mtu"]) before any big notification
        t = now()
        raw = await client.read_gatt_char(P.STATUS_UUID)
        print(f"status read ({(now() - t) * 1000:.0f} ms): {raw.decode()}")
        results["status_read"] = json.loads(raw.decode())
        t_sub = now()
        for char, uuid in (("answer", P.ANSWER_UUID), ("status", P.STATUS_UUID), ("state", P.STATE_UUID)):
            await client.start_notify(uuid, link.handler(char))
        rec = await link.next("state", 5.0)
        if rec is None:
            print("no state snapshot within 5 s of subscribing")
            results["error"] = "no snapshot"
        else:
            first_s = rec["t"] - t_sub
            deliver_ms = (rec["t"] - rec["t0"]) * 1000
            print(f"state snapshot {rec['bytes']} B in {rec['chunks']} chunks: {first_s * 1000:.0f} ms after "
                  f"subscribing, {deliver_ms:.0f} ms first->last chunk")
            print("   ", describe_state(rec["obj"]))
            results.setdefault("snapshot", []).append({"bytes": rec["bytes"], "chunks": rec["chunks"],
                                                       "after_subscribe_ms": round(first_s * 1000),
                                                       "delivery_ms": round(deliver_ms, 1)})
        qid = qid0
        for q in questions:
            qid = (qid + 1) & 0xFFFF
            body = P.dumps({"id": qid, "q": q})
            t0 = now()
            await client.write_gatt_char(P.QUESTION_UUID, body, response=True)
            t_written = now()
            rec = await link.next("answer", args.answer_timeout, pred=lambda r, i=qid: r["obj"].get("id") == i)
            if rec is None:
                print(f"Q{qid} {q!r}: no answer in {args.answer_timeout} s")
                results.setdefault("answers", []).append({"q": q, "error": "timeout"})
                continue
            a = rec["obj"]
            rt = (rec["t"] - t0) * 1000
            print(f"Q{qid} {q!r} -> {rt:.0f} ms round trip (write ack {(t_written - t0) * 1000:.0f} ms, "
                  f"bridge {a['ms']} ms, {rec['bytes']} B / {rec['chunks']} chunks)")
            print(f"    ok={a['ok']} text={a['text']!r} point_at={a['point_at']} action={a['action']} "
                  f"target={a['target']}")
            results.setdefault("answers", []).append({"q": q, "round_trip_ms": round(rt), "bridge_ms": a["ms"],
                                                      "write_ack_ms": round((t_written - t0) * 1000),
                                                      "bytes": rec["bytes"], "chunks": rec["chunks"],
                                                      "answer": a})
            await asyncio.sleep(args.gap)
        if args.watch > 0:
            n0 = len(link.messages["state"])
            print(f"watching state for {args.watch} s ...")
            await asyncio.sleep(args.watch)
            recs = link.messages["state"][n0:]
            if recs:
                gaps = [b["t"] - a["t"] for a, b in zip(recs, recs[1:])]
                print(f"  {len(recs)} state messages, sizes {min(r['bytes'] for r in recs)}-"
                      f"{max(r['bytes'] for r in recs)} B, delivery max "
                      f"{max((r['t'] - r['t0']) * 1000 for r in recs):.0f} ms"
                      + (f", interval median {statistics.median(gaps):.2f} s" if gaps else ""))
                print("   last:", describe_state(recs[-1]["obj"]))
            results.setdefault("watch", []).append({"seconds": args.watch, "state_msgs": len(recs)})
        st = link.messages["status"]
        if st:
            print("last status notify:", st[-1]["obj"])
        results["dropped_partials"] = sum(a.dropped for a in link.asm.values())
        results["bad_json"] = link.bad_json
    return qid


async def amain(args) -> int:
    results: dict = {}
    questions = args.question or DEFAULT_QUESTIONS
    qid = int(time.time()) & 0xFF00
    for i in range(1 + args.reconnects):
        if i:
            print(f"\n--- reconnect {i} ---")
            await asyncio.sleep(args.reconnect_gap)
        qid = await session(args, results, questions if i == 0 else questions[:1], qid)
        if results.get("error"):
            break
    rts = [a["round_trip_ms"] for a in results.get("answers", []) if "round_trip_ms" in a]
    print("\nsummary:")
    if results.get("connect_s"):
        print(f"  connect (incl. discovery): {', '.join(f'{c:.2f}' for c in results['connect_s'])} s")
    if rts:
        print(f"  question -> answer: median {statistics.median(rts):.0f} ms, min {min(rts)} ms, max {max(rts)} ms "
              f"over {len(rts)}")
    for s in results.get("snapshot", []):
        print(f"  snapshot: {s['bytes']} B, {s['chunks']} chunks, {s['delivery_ms']} ms first->last chunk")
    if args.json:
        print(json.dumps(results, indent=1, default=str))
    return 1 if results.get("error") or not rts else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BLE test client for the Ask the Room bridge")
    ap.add_argument("-q", "--question", action="append", help="question to ask (repeatable)")
    ap.add_argument("--watch", type=float, default=0, help="seconds to keep receiving state afterwards")
    ap.add_argument("--reconnects", type=int, default=0, help="disconnect/reconnect cycles after the first")
    ap.add_argument("--reconnect-gap", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.5, help="seconds between questions")
    ap.add_argument("--scan-timeout", type=float, default=15.0)
    ap.add_argument("--answer-timeout", type=float, default=15.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
