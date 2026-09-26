"""EventLog (spec 3.6): SQLite event history + background JPEG snapshots.

Python 3.10 compatible. db_path ':memory:' gives one in-memory database shared by every thread
(replays, the fake-world dev server, tests).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import queue
import sqlite3
import threading
import time

import cv2

from core.types import Event, EventType, Frame

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, t REAL, wall REAL, obj TEXT, type TEXT,
  from_x REAL, from_y REAL, to_x REAL, to_y REAL, parent TEXT, edge TEXT, confidence REAL, snapshot TEXT);
CREATE TABLE IF NOT EXISTS state_snapshots (t REAL, json TEXT);
CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY, t REAL, text TEXT, intent TEXT, obj TEXT,
  answer TEXT, online INTEGER, latency_ms INTEGER);
CREATE INDEX IF NOT EXISTS events_obj_t ON events(obj, t);
CREATE INDEX IF NOT EXISTS events_wall ON events(wall);
"""

_COLS = 't, wall, obj, type, from_x, from_y, to_x, to_y, parent, edge, confidence, snapshot'


def _row_to_event(r) -> Event:
    t, wall, obj, type_, fx, fy, tx, ty, parent, edge, conf, snap = r
    return Event(t=t, wall=wall, obj=obj, type=EventType(type_),
                 from_cm=None if fx is None else (fx, fy),
                 to_cm=None if tx is None else (tx, ty),
                 parent=parent, edge=edge, confidence=conf, snapshot=snap)


def _json_default(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)   # e.g. world passes {name: Entity}
    raise TypeError(f'{type(o).__name__} is not JSON serialisable')


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        log.warning('could not delete snapshot %s', path, exc_info=True)


class EventLog:
    """Event history in SQLite, shared by the world thread (writes) and voice/server (reads).

    Threading: every thread gets its own connection (threading.local). WAL lets those readers
    run concurrently with the writer and see each commit immediately. Snapshot JPEGs are
    encoded on one background thread fed by a queue, so add() never waits on cv2.imwrite.
    """

    def __init__(self, db_path: str, snap_dir: str, prune_on_start_h: float | None = 24):
        self.db_path = db_path
        self.snap_dir = snap_dir
        self._local = threading.local()
        self._conns: dict[threading.Thread, sqlite3.Connection] = {}
        self._conns_lock = threading.Lock()
        self._closed = False
        self._mem: sqlite3.Connection | None = None
        self._mem_lock = threading.RLock()
        if db_path == ':memory:':
            # one connection for all threads, or each thread would see its own empty database
            self._mem = sqlite3.connect(':memory:', check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        os.makedirs(snap_dir, exist_ok=True)
        self._conn().executescript(SCHEMA)
        # Unbounded: events are rare (a handful per second at worst) and imwrite keeps up.
        self._snapq: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_snapshots, name='eventlog-snapshots',
                                        daemon=True)
        self._writer.start()
        if prune_on_start_h is not None:
            self.prune(prune_on_start_h)

    def _write_snapshots(self) -> None:
        while True:
            item = self._snapq.get()
            try:
                if item is None:
                    return
                path, img = item
                if not cv2.imwrite(path, img):
                    log.warning('snapshot write failed: %s', path)
            except Exception:
                log.exception('snapshot write failed')
            finally:
                self._snapq.task_done()

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection. One per thread, so no connection is ever shared.

        check_same_thread=False only so close() can close connections owned by long-lived
        reader threads (voice, server); in normal use each is touched by its own thread only.
        """
        if self._mem is not None:
            if self._closed:
                raise RuntimeError('EventLog is closed')
            return self._mem
        c = getattr(self._local, 'conn', None)
        if c is None:
            if self._closed:
                raise RuntimeError('EventLog is closed')
            c = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
            c.execute('PRAGMA journal_mode=WAL')     # readers never block the writer
            c.execute('PRAGMA synchronous=NORMAL')   # WAL + NORMAL: no fsync per commit
            self._local.conn = c
            with self._conns_lock:
                # Close connections left by finished threads (e.g. per-request server threads).
                for th in [th for th in self._conns if not th.is_alive()]:
                    self._conns.pop(th).close()
                self._conns[threading.current_thread()] = c
        return c

    def _locked(self):
        """Serialises the shared ':memory:' connection. File databases keep one connection per
        thread under WAL, so they need no lock."""
        return self._mem_lock if self._mem is not None else contextlib.nullcontext()

    def add(self, ev: Event, frame: Frame | None = None) -> None:
        """Log ev (world thread). JPEG encoding happens on the writer thread, never here.

        With an image, the snapshot path is decided now, stored in the row and also set on
        ev.snapshot so the caller can pass the event on with its path.
        """
        if frame is not None and frame.img is not None:
            ev.snapshot = os.path.join(
                self.snap_dir, f'{int(ev.wall * 1000)}_{ev.obj}_{EventType(ev.type).value}.jpg')
            # Copy (~1 ms at 720p) so a capture loop that reuses its buffer cannot change the
            # pixels before the writer encodes them.
            self._snapq.put((ev.snapshot, frame.img.copy()))
        fx, fy = ev.from_cm if ev.from_cm is not None else (None, None)
        tx, ty = ev.to_cm if ev.to_cm is not None else (None, None)
        # Commit every add. With WAL + synchronous=NORMAL a commit is an append to the -wal file
        # with no fsync (~20 us; 1,000 adds measured ~25 ms), so batching buys nothing, and a
        # per-add commit means readers on any thread see the event as soon as add() returns.
        # Trade-off: a power cut can lose the last few commits; the DB stays consistent.
        with self._locked():
            c = self._conn()
            c.execute(f'INSERT INTO events ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                      (ev.t, ev.wall, ev.obj, EventType(ev.type).value, fx, fy, tx, ty,
                       ev.parent, ev.edge, ev.confidence, ev.snapshot))
            c.commit()

    def _rows(self, sql: str, args=()) -> list:
        with self._locked():
            return self._conn().execute(sql, args).fetchall()

    def last(self, obj: str, n: int = 3) -> list[Event]:
        """Most recent n events for obj, newest first."""
        rows = self._rows(f'SELECT {_COLS} FROM events WHERE obj = ? ORDER BY t DESC, id DESC LIMIT ?',
                          (obj, n))
        return [_row_to_event(r) for r in rows]

    def max_number(self, prefix: str) -> int:
        """Highest N among logged objs named prefix + N ('thing:' -> 12 for thing:12), else 0."""
        rows = self._rows('SELECT MAX(CAST(substr(obj, ?) AS INTEGER)) FROM events WHERE obj LIKE ?',
                          (len(prefix) + 1, prefix + '%'))
        return int(rows[0][0] or 0)

    def since(self, wall: float) -> list[Event]:
        """Events with wall time >= wall, oldest first. Wall, not monotonic t: callers pass clock times."""
        rows = self._rows(f'SELECT {_COLS} FROM events WHERE wall >= ? ORDER BY wall, id', (wall,))
        return [_row_to_event(r) for r in rows]

    def last_of_type(self, obj: str, types: list[str]) -> Event | None:
        """Most recent event for obj whose type is in types (strings or EventType members)."""
        names = [EventType(x).value for x in types]
        if not names:
            return None
        marks = ','.join('?' * len(names))
        rows = self._rows(f'SELECT {_COLS} FROM events WHERE obj = ? AND type IN ({marks}) '
                          'ORDER BY t DESC, id DESC LIMIT 1', (obj, *names))
        return _row_to_event(rows[0]) if rows else None

    def log_question(self, text: str, intent: str, obj: str | None, answer: str,
                     online: bool, latency_ms: int, t: float | None = None) -> None:
        """t is wall time; defaults to now."""
        with self._locked():
            c = self._conn()
            c.execute('INSERT INTO questions (t, text, intent, obj, answer, online, latency_ms) '
                      'VALUES (?,?,?,?,?,?,?)',
                      (time.time() if t is None else t, text, intent, obj, answer,
                       int(bool(online)), int(latency_ms)))
            c.commit()

    def save_state(self, state: dict, t: float | None = None) -> None:
        """t defaults to state['t'], else now."""
        if t is None:
            t = state.get('t') or time.time()
        blob = json.dumps(state, default=_json_default)
        with self._locked():
            c = self._conn()
            c.execute('INSERT INTO state_snapshots (t, json) VALUES (?, ?)', (t, blob))
            c.commit()

    def prune(self, max_age_h: float = 24) -> None:
        """Delete snapshot JPEGs older than max_age_h (wall time); keep the event rows.

        Rows whose wall is older than the cutoff get snapshot = NULL after their file is removed.
        JPEGs in snap_dir with an old mtime that no row points at (e.g. the DB was deleted) go too.
        """
        cutoff = time.time() - max_age_h * 3600.0
        with self._locked():
            c = self._conn()
            rows = c.execute('SELECT id, snapshot FROM events WHERE wall < ? AND snapshot IS NOT NULL',
                             (cutoff,)).fetchall()
            for _, path in rows:
                _remove_quietly(path)
            c.executemany('UPDATE events SET snapshot = NULL WHERE id = ?', [(i,) for i, _ in rows])
            c.execute('DELETE FROM state_snapshots WHERE t < ?', (cutoff,))
            c.commit()
        for entry in os.scandir(self.snap_dir):
            if entry.name.endswith('.jpg') and entry.is_file() and entry.stat().st_mtime < cutoff:
                _remove_quietly(entry.path)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every queued snapshot is on disk. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        with self._snapq.all_tasks_done:
            while self._snapq.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._snapq.all_tasks_done.wait(remaining)
        return True

    def close(self) -> None:
        """Write any queued snapshots, stop the writer thread, close every thread's connection."""
        if self._closed:
            return
        self._closed = True
        self._snapq.put(None)          # sentinel goes after pending snapshots, so they are written
        self._writer.join(timeout=10)
        with self._conns_lock:
            conns, self._conns = list(self._conns.values()), {}
        if self._mem is not None:
            conns.append(self._mem)
        for c in conns:
            c.close()
