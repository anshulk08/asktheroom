import sqlite3
import threading
import time

import numpy as np
import pytest

from core.events import EventLog
from core.types import Event, EventType, Frame


@pytest.fixture
def log(tmp_path):
    el = EventLog(str(tmp_path / 'events.db'), str(tmp_path / 'snaps'))
    yield el
    el.close()


def ev(obj='keys', type_=EventType.MOVED, t=1.0, wall=None, **kw):
    return Event(t=t, wall=wall if wall is not None else time.time(), obj=obj, type=type_, **kw)


def test_added_event_round_trips_all_fields(log):
    e = Event(t=12.5, wall=1_700_000_000.25, obj='keys', type=EventType.PUT_INSIDE,
              from_cm=(1.5, 2.0), to_cm=(30.0, 40.25), parent='mug', edge='left', confidence=0.7)
    log.add(e, None)
    got = log.last('keys', n=1)
    assert got == [e]
    assert isinstance(got[0].type, EventType)
    assert isinstance(got[0].from_cm, tuple) and isinstance(got[0].to_cm, tuple)


def test_event_without_positions_round_trips_as_none(log):
    log.add(ev(type_=EventType.EXITED_VIEW, edge='right'), None)
    got = log.last('keys', n=1)[0]
    assert got.from_cm is None and got.to_cm is None and got.snapshot is None


def test_last_returns_newest_first_limited_to_n(log):
    for t in (1.0, 2.0, 3.0, 4.0):
        log.add(ev(t=t), None)
    assert [e.t for e in log.last('keys', n=3)] == [4.0, 3.0, 2.0]


def test_last_only_returns_the_requested_object(log):
    log.add(ev(obj='keys', t=1.0), None)
    log.add(ev(obj='wallet', t=2.0), None)
    assert [e.obj for e in log.last('keys')] == ['keys']


def test_since_filters_and_orders_by_wall_time(log):
    # wall, not monotonic t: voice and the dashboard ask "since 10 minutes ago" in clock time
    log.add(ev(obj='keys', t=1.0, wall=3000.0), None)
    log.add(ev(obj='wallet', t=2.0, wall=1000.0), None)
    log.add(ev(obj='phone', t=3.0, wall=2000.0), None)
    assert [(e.obj, e.wall) for e in log.since(2000.0)] == [('phone', 2000.0), ('keys', 3000.0)]


def test_memory_db_is_shared_across_threads(tmp_path):
    el = EventLog(':memory:', str(tmp_path / 'snaps'))
    th = threading.Thread(target=lambda: el.add(ev(t=1.0), None))
    th.start()
    th.join()
    assert [e.t for e in el.last('keys')] == [1.0]
    el.close()


def test_add_frame_defaults_to_none(log):
    log.add(ev(t=1.0))
    assert log.last('keys')[0].snapshot is None


def test_last_of_type_returns_most_recent_matching_event(log):
    log.add(ev(t=1.0, type_=EventType.PICKED_UP), None)
    log.add(ev(t=2.0, type_=EventType.COVERED, parent='cup'), None)
    log.add(ev(t=3.0, type_=EventType.PICKED_UP), None)
    log.add(ev(t=4.0, type_=EventType.MOVED), None)
    got = log.last_of_type('keys', ['COVERED', 'PICKED_UP'])
    assert (got.t, got.type) == (3.0, EventType.PICKED_UP)


def test_last_of_type_accepts_event_type_members(log):
    log.add(ev(t=1.0, type_=EventType.COVERED), None)
    assert log.last_of_type('keys', [EventType.COVERED]).t == 1.0


def test_last_of_type_is_none_when_nothing_matches(log):
    log.add(ev(obj='keys', type_=EventType.MOVED), None)
    log.add(ev(obj='wallet', type_=EventType.COVERED), None)
    assert log.last_of_type('keys', ['COVERED']) is None
    assert log.last_of_type('keys', []) is None


def test_database_has_spec_tables_and_obj_t_index(tmp_path, log):
    db = sqlite3.connect(str(tmp_path / 'events.db'))
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')")}
    idx_cols = [r[2] for r in db.execute("PRAGMA index_info('events_obj_t')")]
    db.close()
    assert {'events', 'state_snapshots', 'questions', 'events_obj_t'} <= names
    assert idx_cols == ['obj', 't']


def test_database_uses_wal_journal(tmp_path, log):
    db = sqlite3.connect(str(tmp_path / 'events.db'))
    mode = db.execute('PRAGMA journal_mode').fetchone()[0]
    db.close()
    assert mode == 'wal'


def test_connections_use_synchronous_normal(log):
    # synchronous is per-connection, so check the log's own connection. 1 == NORMAL.
    assert log._conn().execute('PRAGMA synchronous').fetchone()[0] == 1


def test_thousand_adds_without_images_take_under_a_second(log):
    start = time.perf_counter()
    for i in range(1000):
        log.add(ev(t=float(i), from_cm=(1.0, 2.0), to_cm=(3.0, 4.0)), None)
    assert time.perf_counter() - start < 1.0
    assert len(log.since(0.0)) == 1000


def _poll(fn, timeout=0.1):
    deadline = time.monotonic() + timeout
    while True:
        got = fn()
        if got or time.monotonic() > deadline:
            return got
        time.sleep(0.005)


def _in_thread(fn):
    out = {}

    def run():
        try:
            out['value'] = fn()
        except BaseException as e:   # surface errors from the thread in the test
            out['error'] = e

    th = threading.Thread(target=run)
    th.start()
    th.join(5)
    assert not th.is_alive()
    if 'error' in out:
        raise out['error']
    return out['value']


def test_reader_thread_sees_event_added_on_another_thread(log):
    log.add(ev(t=7.0, type_=EventType.COVERED, parent='cup'), None)
    got = _in_thread(lambda: _poll(lambda: log.last('keys')))
    assert [(e.t, e.parent) for e in got] == [(7.0, 'cup')]


def test_events_added_on_world_thread_are_visible_to_main_thread_within_100ms(log):
    _in_thread(lambda: [log.add(ev(t=float(i)), None) for i in range(5)])
    assert [e.t for e in _poll(lambda: log.since(0.0))] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_reads_on_many_threads_while_writing(log):
    stop = threading.Event()
    errors = []

    def reader():
        try:
            while not stop.is_set():
                log.last('keys'), log.since(0.0), log.last_of_type('keys', ['MOVED'])
        except BaseException as e:
            errors.append(e)

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for r in readers:
        r.start()
    for i in range(200):
        log.add(ev(t=float(i)), None)
    stop.set()
    for r in readers:
        r.join(5)
    assert errors == []
    assert len(log.since(0.0)) == 200


def _frame(img, wall=1_700_000_000.123):
    return Frame(t=1.0, wall=wall, img=img, idx=0)


def test_add_with_image_stores_snapshot_path_named_by_wall_ms_obj_type(tmp_path, log):
    img = np.zeros((72, 128, 3), np.uint8)
    log.add(ev(wall=1_700_000_000.123, type_=EventType.PICKED_UP), _frame(img))
    expected = str(tmp_path / 'snaps' / '1700000000123_keys_PICKED_UP.jpg')
    assert log.last('keys')[0].snapshot == expected


def test_snapshot_jpeg_is_written_by_background_writer(log):
    import cv2
    img = np.full((72, 128, 3), 200, np.uint8)
    log.add(ev(), _frame(img))
    log.flush()
    written = cv2.imread(log.last('keys')[0].snapshot)
    assert written is not None and written.shape == (72, 128, 3)


def test_frame_without_image_gives_no_snapshot(log):
    log.add(ev(), _frame(None))
    assert log.last('keys')[0].snapshot is None


def test_add_does_not_wait_for_jpeg_encoding(log):
    # Random noise at 1280x720 is expensive to JPEG-encode (~10+ ms each even on a laptop),
    # so 20 synchronous imwrites could not fit in 100 ms.
    img = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    start = time.perf_counter()
    for i in range(20):
        log.add(ev(t=float(i), wall=1_700_000_000.0 + i), _frame(img))
    elapsed = time.perf_counter() - start
    log.flush()
    assert elapsed < 0.1


def test_flush_waits_until_all_queued_snapshots_exist(log):
    import os
    img = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for i in range(5):
        log.add(ev(t=float(i), wall=1_700_000_000.0 + i), _frame(img))
    log.flush()
    assert all(os.path.exists(e.snapshot) for e in log.since(0.0))


def _writer_threads():
    return [t for t in threading.enumerate() if t.name == 'eventlog-snapshots']


def test_close_stops_the_snapshot_writer_thread(tmp_path):
    # this log's own writer: other tests' logs (fake worlds) may still have theirs running
    el = EventLog(str(tmp_path / 'e.db'), str(tmp_path / 's'))
    assert el._writer in _writer_threads()
    el.close()
    assert el._writer not in _writer_threads()


def test_close_finishes_queued_snapshots_first(tmp_path):
    import os
    el = EventLog(str(tmp_path / 'e.db'), str(tmp_path / 's'))
    el.add(ev(), _frame(np.zeros((72, 128, 3), np.uint8)))
    path = el.last('keys')[0].snapshot
    el.close()
    assert os.path.exists(path)


def test_close_twice_is_harmless(tmp_path):
    el = EventLog(str(tmp_path / 'e.db'), str(tmp_path / 's'))
    el.close()
    el.close()


def test_close_releases_connections_held_by_live_reader_threads(tmp_path):
    import os
    db = str(tmp_path / 'e.db')
    el = EventLog(db, str(tmp_path / 's'))
    el.add(ev(), None)
    has_read, release = threading.Event(), threading.Event()

    def long_lived_reader():   # like the voice/server threads: reads, then stays alive
        el.last('keys')
        has_read.set()
        release.wait(5)

    th = threading.Thread(target=long_lived_reader)
    th.start()
    has_read.wait(5)
    try:
        el.close()
        # SQLite checkpoints and deletes the -wal file only when the last connection closes.
        assert not os.path.exists(db + '-wal')
    finally:
        release.set()
        th.join(5)


HOUR = 3600.0
TINY = np.zeros((8, 8, 3), np.uint8)


def _add_snapped(el, obj, wall):
    el.add(ev(obj=obj, wall=wall), _frame(TINY, wall=wall))
    el.flush()
    return el.last(obj, n=1)[0].snapshot


def test_prune_deletes_old_snapshots_and_clears_their_paths(log):
    import os
    path = _add_snapped(log, 'keys', time.time() - 30 * HOUR)
    log.prune(max_age_h=24)
    assert not os.path.exists(path)
    got = log.last('keys')
    assert len(got) == 1 and got[0].snapshot is None


def test_prune_keeps_recent_snapshots(log):
    import os
    path = _add_snapped(log, 'wallet', time.time() - 1 * HOUR)
    log.prune(max_age_h=24)
    assert os.path.exists(path)
    assert log.last('wallet')[0].snapshot == path


def test_prune_deletes_old_orphan_jpegs_by_mtime(tmp_path, log):
    import os
    snaps = tmp_path / 'snaps'
    old, new = snaps / 'old_orphan.jpg', snaps / 'new_orphan.jpg'
    old.write_bytes(b'x')
    new.write_bytes(b'x')
    past = time.time() - 30 * HOUR
    os.utime(old, (past, past))
    log.prune(max_age_h=24)
    assert not old.exists() and new.exists()


def test_prune_tolerates_already_missing_snapshot_file(log):
    import os
    path = _add_snapped(log, 'keys', time.time() - 30 * HOUR)
    os.remove(path)
    log.prune(max_age_h=24)
    assert log.last('keys')[0].snapshot is None


def test_construction_prunes_old_snapshots_by_default(tmp_path):
    import os
    db, snaps = str(tmp_path / 'e.db'), str(tmp_path / 's')
    el = EventLog(db, snaps, prune_on_start_h=None)
    path = _add_snapped(el, 'keys', time.time() - 30 * HOUR)
    el.close()
    el = EventLog(db, snaps)
    try:
        assert not os.path.exists(path)
        assert el.last('keys')[0].snapshot is None
    finally:
        el.close()


def test_construction_without_prune_keeps_old_snapshots(tmp_path):
    import os
    db, snaps = str(tmp_path / 'e.db'), str(tmp_path / 's')
    el = EventLog(db, snaps, prune_on_start_h=None)
    path = _add_snapped(el, 'keys', time.time() - 30 * HOUR)
    el.close()
    el = EventLog(db, snaps, prune_on_start_h=None)
    try:
        assert os.path.exists(path)
    finally:
        el.close()


def _rows(tmp_path, sql):
    db = sqlite3.connect(str(tmp_path / 'events.db'))
    try:
        return db.execute(sql).fetchall()
    finally:
        db.close()


def test_log_question_stores_a_row(tmp_path, log):
    log.log_question('where are my keys?', 'WHERE', 'keys', 'Under the red cup.', True, 840, t=5.0)
    log.log_question('what moved?', 'SUMMARY', None, 'Nothing.', False, 120, t=6.0)
    assert _rows(tmp_path, 'SELECT t, text, intent, obj, answer, online, latency_ms FROM questions '
                           'ORDER BY id') == [
        (5.0, 'where are my keys?', 'WHERE', 'keys', 'Under the red cup.', 1, 840),
        (6.0, 'what moved?', 'SUMMARY', None, 'Nothing.', 0, 120),
    ]


def test_save_state_stores_json(tmp_path, log):
    import json
    state = {'keys': {'status': 'UNDER', 'parent': 'cup', 'pos_cm': [10.0, 20.5]}}
    log.save_state(state, t=9.0)
    [(t, blob)] = _rows(tmp_path, 'SELECT t, json FROM state_snapshots')
    assert t == 9.0 and json.loads(blob) == state


def test_save_state_serialises_enums_and_tuples(tmp_path, log):
    import json
    from core.types import Status
    log.save_state({'keys': {'status': Status.HELD, 'pos_cm': (1.0, 2.0)}}, t=1.0)
    [(blob,)] = _rows(tmp_path, 'SELECT json FROM state_snapshots')
    assert json.loads(blob) == {'keys': {'status': 'HELD', 'pos_cm': [1.0, 2.0]}}


def test_save_state_accepts_entity_dataclasses(tmp_path, log):
    import json
    from core.types import Entity, Status
    log.save_state({'keys': Entity(name='keys', kind='target', status=Status.UNDER, parent='cup')}, t=1.0)
    [(blob,)] = _rows(tmp_path, 'SELECT json FROM state_snapshots')
    got = json.loads(blob)['keys']
    assert (got['name'], got['status'], got['parent']) == ('keys', 'UNDER', 'cup')


def test_failed_snapshot_write_does_not_stop_later_snapshots(log):
    import os
    unencodable = np.zeros((8, 8, 5), np.uint8)   # 5 channels: cv2.imwrite raises cv2.error
    log.add(ev(obj='keys', wall=1_700_000_000.0), _frame(unencodable))
    log.add(ev(obj='wallet', wall=1_700_000_001.0), _frame(TINY))
    assert log.flush(timeout=5.0)
    assert os.path.exists(log.last('wallet')[0].snapshot)


def test_connections_of_finished_threads_are_not_accumulated(log):
    # A threaded HTTP server may use a fresh thread per request; each opens a connection.
    log.add(ev(), None)
    for _ in range(20):
        _in_thread(lambda: log.last('keys'))
    _in_thread(lambda: log.last('keys'))
    assert len(log._conns) <= 2   # this thread's + at most the latest reader's
