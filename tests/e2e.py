"""End-to-end key-path tests using the stdlib runner (no third-party deps).

Run:  python3 -m tests.e2e
Both services are started in-process on real HTTP ports with isolated SQLite
databases; assertions cover every rule from the requirements.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.http_runner import run_alert_service, run_notifier_service  # noqa: E402

ALERT_PORT = 18080
NOTIFIER_PORT = 18081
BASE = f"http://127.0.0.1:{ALERT_PORT}"
NOTIFIER = f"http://127.0.0.1:{NOTIFIER_PORT}"


def http(method: str, url: str, body=None, expect=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode())
            status = resp.status
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode())
        status = exc.code
    if expect is not None:
        assert status == expect, f"{method} {url} -> {status} (want {expect}): {payload}"
    return status, payload


def wait_until(desc, predicate, timeout=10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for: {desc} (last={last!r})")


def alert_status(fp):
    _, data = http("GET", f"{BASE}/api/alerts")
    for a in data["alerts"]:
        if a["fingerprint"] == fp:
            return a
    return None


def outbox_rows(fp):
    _, data = http("GET", f"{BASE}/api/notifications")
    return [n for n in data["notifications"] if n["fingerprint"] == fp]


def received_nos():
    _, data = http("GET", f"{NOTIFIER}/api/notifications")
    return {n["notification_no"]: n for n in data["notifications"]}


def wait_delivered(fp, expected_count, timeout=10.0):
    def check():
        rows = sorted(outbox_rows(fp), key=lambda r: r["id"])
        if len(rows) != expected_count:
            return None
        return rows if all(r["status"] == "delivered" for r in rows) else None
    return wait_until(f"{fp} {expected_count} delivered", check, timeout)


def console_snapshot():
    # The single request a page refresh makes: alerts, silence and outbox all
    # come from the same committed snapshot on the server.
    return http("GET", f"{BASE}/api/alerts", expect=200)[1]


class SlowNotifier:
    """Notifier replacement that can stall deliveries to create interleavings.

    ``hold`` blocks deliveries for fingerprints matching ``gate`` (prefix)
    until released, which lets a test race event ingestion and silence changes
    against one outbox row that is sitting 'in_transit' while the page is
    refreshed. Other fingerprints answer immediately so earlier tests' leftovers
    never block the worker.
    """

    def __init__(self, port):
        self.port = port
        self.hold = threading.Event()
        self.hold.set()
        self.gate = None
        self.seen = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode())
                if outer.gate is None or payload["fingerprint"].startswith(outer.gate):
                    outer.hold.wait(30)
                with outer._lock:
                    outer.seen.append(payload["notification_no"])
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def release(self):
        self.hold.set()

    def close(self):
        self.hold.set()
        self.server.shutdown()
        self.server.server_close()


class E2ETestCase(unittest.TestCase):
    tmpdir: str
    alert_server = None
    notifier_server = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="alert-e2e-")
        os.environ["ALERT_DB_PATH"] = os.path.join(cls.tmpdir, "alerts.db")
        os.environ["NOTIFIER_DB_PATH"] = os.path.join(cls.tmpdir, "notifier.db")
        os.environ["NOTIFIER_URL"] = NOTIFIER
        os.environ["WORKER_INTERVAL_SECONDS"] = "0.1"
        cls.notifier_server = run_notifier_service("127.0.0.1", NOTIFIER_PORT)
        cls.alert_server = run_alert_service("127.0.0.1", ALERT_PORT)
        wait_until("services healthy", lambda: (
            http("GET", f"{BASE}/health")[0] == 200
            and http("GET", f"{NOTIFIER}/health")[0] == 200
        ))

    @classmethod
    def tearDownClass(cls):
        cls.alert_server.stop_event.set()
        cls.alert_server.shutdown()
        cls.alert_server.server_close()
        cls.notifier_server.shutdown()
        cls.notifier_server.server_close()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_01_health_and_pages(self):
        self.assertEqual(http("GET", f"{BASE}/health", expect=200)[1]["service"],
                         "alert-console")
        self.assertEqual(http("GET", f"{NOTIFIER}/health", expect=200)[1]["service"],
                         "local-notifier")
        for url in (f"{BASE}/", f"{NOTIFIER}/"):
            with urllib.request.urlopen(url, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn(b"text/html", resp.headers["Content-Type"].encode())
        with urllib.request.urlopen(f"{BASE}/static/app.js", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn(b"javascript", resp.headers["Content-Type"].encode())
        with urllib.request.urlopen(f"{NOTIFIER}/static/notifier.js", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
        # path traversal must not escape the static directory (403 or 404 both safe)
        traversal_status, _ = http("GET", f"{BASE}/static/../../common/config.py")
        self.assertIn(traversal_status, (403, 404))

    def test_02_basic_firing_recovery_and_lost_receipt(self):
        fp = "svc/basic"
        s, r1 = http("POST", f"{BASE}/api/events",
                     {"fingerprint": fp, "seq": 1, "kind": "firing", "summary": "cpu high"},
                     expect=200)
        self.assertEqual(r1["result"], "accepted")
        self.assertEqual(r1["alert"]["status"], "firing")
        self.assertEqual(r1["alert"]["round"], 1)

        # identical replay keeps the same result, creates nothing new
        s, r2 = http("POST", f"{BASE}/api/events",
                     {"fingerprint": fp, "seq": 1, "kind": "firing", "summary": "cpu high"},
                     expect=200)
        self.assertEqual(r2["result"], "duplicate_replayed")

        # same seq different body -> rejected
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 1, "kind": "firing", "summary": "changed text"},
             expect=409)

        # recovery
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 2, "kind": "recovery", "summary": "cpu back"},
             expect=200)
        self.assertEqual(alert_status(fp)["status"], "recovered")

        # stale seq cannot rewrite state
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 2, "kind": "firing", "summary": "late trigger"},
             expect=409)
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 1, "kind": "recovery", "summary": "old recovery"},
             expect=409)
        self.assertEqual(alert_status(fp)["status"], "recovered")

        rows = wait_delivered(fp, 2)
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds, ["firing", "recovery"])

        # notifier: each number stored exactly once despite retried receipts
        recv = received_nos()
        firing_no = next(r["notification_no"] for r in rows if r["kind"] == "firing")
        rec_no = next(r["notification_no"] for r in rows if r["kind"] == "recovery")
        self.assertIn(firing_no, recv)
        self.assertIn(rec_no, recv)
        self.assertGreaterEqual(recv[firing_no]["attempts_seen"], 1)
        self.assertEqual(recv[firing_no]["status"], "accepted")
        # ordering on the receiver side: firing arrived before recovery
        self.assertLess(recv[firing_no]["first_seen_at"], recv[rec_no]["first_seen_at"])

        # outbox attempts show the lost receipt was retried with the same number
        firing_row = next(r for r in rows if r["kind"] == "firing")
        self.assertGreaterEqual(firing_row["attempts"], 2)

    def test_03_silence_parks_new_firing_but_recovery_still_notifies(self):
        fp = "svc/silenced"
        # fault announced before the maintenance window
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 1, "kind": "firing", "summary": "disk full"},
             expect=200)
        wait_delivered(fp, 1)

        # enable silence
        _, sil = http("POST", f"{BASE}/api/silence",
                      {"duration_seconds": 30, "reason": "maintenance"}, expect=200)
        self.assertTrue(sil["active"])

        # new trigger during silence: alert updates, no notification enqueued
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 2, "kind": "firing", "summary": "disk fuller"},
             expect=200)
        a = alert_status(fp)
        self.assertEqual(a["status"], "firing")
        self.assertEqual(a["summary"], "disk fuller")
        time.sleep(0.3)
        self.assertEqual(len(outbox_rows(fp)), 1)

        # recovery of an already-announced fault MUST still be notified
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 3, "kind": "recovery", "summary": "disk ok"},
             expect=200)
        rows = wait_delivered(fp, 2)
        self.assertEqual([r["kind"] for r in rows], ["firing", "recovery"])

        # receiver shows both; no extra firing notification appeared
        recv = received_nos()
        self.assertEqual(
            sorted(k for k in recv if k.startswith(fp)),
            sorted(r["notification_no"] for r in rows),
        )
        http("DELETE", f"{BASE}/api/silence", expect=200)

    def test_04_silence_expiry_replays_only_still_firing_unannounced(self):
        fp = "svc/parked"
        http("POST", f"{BASE}/api/silence",
             {"duration_seconds": 1, "reason": "short window"}, expect=200)

        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 1, "kind": "firing", "summary": "latency"},
             expect=200)
        # also a second fingerprint that recovers during the silence
        fp2 = "svc/parked-recovered"
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp2, "seq": 1, "kind": "firing", "summary": "temp spike"},
             expect=200)
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp2, "seq": 2, "kind": "recovery", "summary": "spike gone"},
             expect=200)

        time.sleep(0.3)
        # nothing announced yet: fp2 recovered without ever being notified,
        # so it must never produce a notification for that round
        self.assertEqual(outbox_rows(fp), [])
        self.assertEqual(outbox_rows(fp2), [])

        # wait for the silence to expire and the worker sweep to release it
        rows = wait_delivered(fp, 1, timeout=8)
        self.assertEqual(rows[0]["kind"], "firing")
        time.sleep(0.3)
        self.assertEqual(outbox_rows(fp2), [], "recovered-during-silence round stays unannounced")

    def test_05_retrigger_after_recovery_is_a_new_round(self):
        fp = "svc/rounds"
        for seq, kind, text in [
            (1, "firing", "round1 fire"),
            (2, "recovery", "round1 rec"),
            (3, "firing", "round2 fire"),
            (4, "recovery", "round2 rec"),
        ]:
            http("POST", f"{BASE}/api/events",
                 {"fingerprint": fp, "seq": seq, "kind": kind, "summary": text}, expect=200)

        a = alert_status(fp)
        self.assertEqual(a["round"], 2)
        self.assertEqual(a["status"], "recovered")
        rows = wait_delivered(fp, 4)
        self.assertEqual([(r["round"], r["kind"]) for r in rows],
                         [(1, "firing"), (1, "recovery"), (2, "firing"), (2, "recovery")])
        # notification numbers are unique per round/kind
        self.assertEqual(len({r["notification_no"] for r in rows}), 4)

    def test_06_per_fingerprint_ordering_under_loss(self):
        fp = "svc/order"
        for seq, kind, text in [
            (1, "firing", "f1"),
            (2, "recovery", "r1"),
            (3, "firing", "f2"),
        ]:
            http("POST", f"{BASE}/api/events",
                 {"fingerprint": fp, "seq": seq, "kind": kind, "summary": text}, expect=200)
        rows = wait_delivered(fp, 3)
        self.assertEqual([r["kind"] for r in rows], ["firing", "recovery", "firing"])
        recv = received_nos()
        # Receiver first-seen timestamps must follow the outbox (id) order.
        times = [recv[r["notification_no"]]["first_seen_at"] for r in rows]
        self.assertEqual(times, sorted(times))

    def test_07_invalid_payloads(self):
        # stdlib runner maps validation rejects to 409; the FastAPI app uses
        # Pydantic and answers 422 for the same malformed input.
        http("POST", f"{BASE}/api/events", {"seq": 1, "kind": "firing", "summary": "x"},
             expect=409)
        http("POST", f"{BASE}/api/events",
             {"fingerprint": "x", "seq": -1, "kind": "firing", "summary": "x"}, expect=409)
        http("POST", f"{BASE}/api/events",
             {"fingerprint": "x", "seq": 1, "kind": "explode", "summary": "x"}, expect=409)
        http("POST", f"{BASE}/api/silence", {"duration_seconds": -3}, expect=422)

    def test_08_restart_does_not_lose_or_reorder(self):
        # Persist a partially delivered outbox, restart both servers, verify
        # everything converges with no duplicates on the receiver.
        fp = "svc/restart"
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 1, "kind": "firing", "summary": "before restart"},
             expect=200)
        wait_delivered(fp, 1)
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 2, "kind": "firing", "summary": "still firing text"},
             expect=200)  # same round refresh, no new notification

        before = received_nos()
        before_nos = {k: v["attempts_seen"] for k, v in before.items() if k.startswith(fp)}

        # restart both processes on the same database files
        self.alert_server.stop_event.set()
        self.alert_server.shutdown()
        self.alert_server.server_close()
        self.notifier_server.shutdown()
        self.notifier_server.server_close()
        time.sleep(0.2)
        type(self).notifier_server = run_notifier_service("127.0.0.1", NOTIFIER_PORT)
        type(self).alert_server = run_alert_service("127.0.0.1", ALERT_PORT)
        wait_until("services healthy after restart", lambda: (
            http("GET", f"{BASE}/health")[0] == 200
            and http("GET", f"{NOTIFIER}/health")[0] == 200
        ))

        # state survived
        self.assertEqual(alert_status(fp)["summary"], "still firing text")

        # post a recovery after restart; it must flow through and no duplicate
        # firing notification may be produced
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 3, "kind": "recovery", "summary": "after restart rec"},
             expect=200)
        rows = wait_delivered(fp, 2)
        self.assertEqual([r["kind"] for r in rows], ["firing", "recovery"])

        after = received_nos()
        for no, seen in before_nos.items():
            # the pre-restart notification number is still a single stored row
            self.assertIn(no, after)
            self.assertEqual(after[no]["first_seen_at"], before[no]["first_seen_at"])

        # a late, older trigger must not overwrite the recovered state
        http("POST", f"{BASE}/api/events",
             {"fingerprint": fp, "seq": 2, "kind": "firing", "summary": "late again"},
             expect=409)
        self.assertEqual(alert_status(fp)["status"], "recovered")


ALERT_PORT2 = 18090
ALERT_BASE2 = f"http://127.0.0.1:{ALERT_PORT2}"
SLOW_NOTIFIER_PORT = 18091


class ConsistencyInterleavingTest(unittest.TestCase):
    """Reads must show only committed data, and only from one commit instant.

    A slow notifier holds the outbox worker inside a network call while events
    and silences are applied, so a refresh racing those writes would expose
    uncommitted rows if reads shared the writer connection, or splice state
    from different commits if they were fetched in separate requests.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="alert-interleave-")
        cls.alert_db = os.path.join(cls.tmpdir, "alerts.db")
        cls.notifier_db = os.path.join(cls.tmpdir, "notifier.db")
        os.environ["ALERT_DB_PATH"] = cls.alert_db
        os.environ["NOTIFIER_DB_PATH"] = cls.notifier_db
        os.environ["NOTIFIER_URL"] = f"http://127.0.0.1:{SLOW_NOTIFIER_PORT}"
        os.environ["WORKER_INTERVAL_SECONDS"] = "0.05"
        os.environ["NOTIFY_TIMEOUT_SECONDS"] = "30"
        cls.slow = SlowNotifier(SLOW_NOTIFIER_PORT)
        cls.alert_server = run_alert_service("127.0.0.1", ALERT_PORT2)
        wait_until("alert console healthy", lambda: (
            http("GET", f"{ALERT_BASE2}/health")[0] == 200))

    @classmethod
    def tearDownClass(cls):
        cls.alert_server.stop_event.set()
        cls.alert_server.shutdown()
        cls.alert_server.server_close()
        cls.slow.close()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        self.slow.hold.clear()  # park deliveries until the test releases them

    def tearDown(self):
        self.slow.gate = None
        self.slow.release()

    def _post(self, fp, seq, kind, summary):
        return http("POST", f"{ALERT_BASE2}/api/events",
                    {"fingerprint": fp, "seq": seq, "kind": kind, "summary": summary},
                    expect=200)

    def _snapshot(self):
        return http("GET", f"{ALERT_BASE2}/api/alerts", expect=200)[1]

    def test_01_uncommitted_recovery_and_silence_changes_are_never_visible(self):
        fp = "race/snapshot"
        self.slow.gate = fp
        self._post(fp, 1, "firing", "cpu high")
        # Park the firing delivery in_transit (worker is inside the HTTP call).
        wait_until("firing in transit", lambda: any(
            n["status"] == "in_transit" for n in outbox_rows_for(fp, ALERT_BASE2)))

        writer = sqlite3.connect(self.alert_db, timeout=30, isolation_level=None)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE alerts SET status='recovered', current_seq=2, summary='cpu ok',"
                       " updated_at=? WHERE fingerprint=?", (time.time(), fp))
        writer.execute("INSERT INTO events(fingerprint, seq, kind, summary, received_at)"
                       " VALUES (?,?,?,?,?)", (fp, 2, "recovery", "cpu ok", time.time()))
        writer.execute("INSERT INTO notifications(notification_no, fingerprint, round, kind,"
                       " summary, status, next_retry_at, enqueued_at)"
                       " VALUES (?,?,?,?,?,'pending',?,?)",
                       (f"{fp}#r1:recovery", fp, 1, "recovery", "cpu ok", 0, time.time()))
        writer.execute("INSERT INTO silences(id, until_ts, reason, updated_at) VALUES (1,?,?,?)",
                       (time.time() + 600, "uncommitted maintenance", time.time()))
        try:
            # While that transaction is open, repeated refreshes must keep
            # showing the last committed state: firing, plain outbox, no silence.
            for _ in range(5):
                snap = self._snapshot()
                alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
                self.assertEqual(alert["status"], "firing")
                self.assertEqual(alert["current_seq"], 1)
                self.assertEqual(alert["summary"], "cpu high")
                self.assertEqual(alert["recovery_notify_status"], None)
                self.assertFalse(snap["silence"]["active"])
                self.assertEqual(
                    [n["notification_no"] for n in snap["notifications"]
                     if n["fingerprint"] == fp], [f"{fp}#r1:firing"])
                time.sleep(0.05)
        finally:
            writer.execute("ROLLBACK")
            writer.close()

        # After rollback the visible state is unchanged -- and still committed
        # state, not the rolled-back "recovered" row.
        snap = self._snapshot()
        alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
        self.assertEqual(alert["status"], "firing")
        self.assertFalse(snap["silence"]["active"])
        self.assertNotIn(f"{fp}#r1:recovery",
                         [n["notification_no"] for n in snap["notifications"]
                          if n["fingerprint"] == fp])

        # A real committed recovery then becomes visible in one step: alert,
        # recovery outbox row and notification states together.
        self._post(fp, 2, "recovery", "cpu ok")
        snap = self._snapshot()
        alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
        self.assertEqual(alert["status"], "recovered")
        nos = [n["notification_no"] for n in snap["notifications"]
               if n["fingerprint"] == fp]
        self.assertIn(f"{fp}#r1:recovery", nos)
        self.assertEqual(alert["recovery_notify_status"], "pending")

    def test_02_reads_do_not_wait_on_slow_notifier(self):
        fp = "race/slow-wire"
        self.slow.gate = fp
        self._post(fp, 1, "firing", "slow delivery")
        wait_until("delivery parked", lambda: any(
            n["status"] == "in_transit" for n in outbox_rows_for(fp, ALERT_BASE2)))
        started = time.time()
        snap = self._snapshot()
        elapsed = time.time() - started
        # The worker is blocked inside the notifier for 30s; the page read must
        # still return promptly from the committed snapshot.
        self.assertLess(elapsed, 2.0)
        alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
        self.assertEqual(alert["status"], "firing")
        self.assertEqual(alert["firing_notify_status"], "in_transit")

    def test_03_silence_change_and_event_are_one_snapshot(self):
        fp = "race/silent"
        self.slow.gate = fp
        self._post(fp, 1, "firing", "before silence")
        wait_until("firing pending", lambda: outbox_rows_for(fp, ALERT_BASE2))
        http("POST", f"{ALERT_BASE2}/api/silence",
             {"duration_seconds": 600, "reason": "window"}, expect=200)
        snap = self._snapshot()
        self.assertTrue(snap["silence"]["active"])
        self.assertEqual(snap["silence"]["reason"], "window")
        # Silence suppresses new notifications, but the alert still updates in
        # the same committed state the banner is read from.
        self._post(fp, 2, "firing", "during silence")
        snap = self._snapshot()
        alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
        self.assertEqual(alert["summary"], "during silence")
        self.assertTrue(snap["silence"]["active"])
        self.assertEqual(len([n for n in snap["notifications"] if n["fingerprint"] == fp]), 1)
        http("DELETE", f"{ALERT_BASE2}/api/silence", expect=200)
        self.assertFalse(self._snapshot()["silence"]["active"])


    def test_04_rolled_back_ingest_is_never_shown_as_recovered(self):
        """The exact reported race, through the real HTTP handlers.

        An incoming recovery fails while enqueueing its notification, so the
        whole ingest transaction rolls back. Reads arriving in that window must
        keep showing the last committed firing state -- not the dirty
        'recovered' row -- and must not resurrect it after the rollback.
        """
        fp = "race-rollback"  # no slash: path params do not span segments
        self.slow.gate = fp
        self._post(fp, 1, "firing", "cpu high")
        wait_until("firing row exists", lambda: outbox_rows_for(fp, ALERT_BASE2))

        from common import service as core_service
        original = core_service._enqueue_notification
        entered = threading.Event()
        failures = []

        def flaky(conn, fp_arg, round_no, kind, summary, now):
            if fp_arg == fp and kind == "recovery":
                entered.set()
                time.sleep(0.4)  # hold the write open while pages refresh
                raise sqlite3.OperationalError("injected failure while enqueueing")
            return original(conn, fp_arg, round_no, kind, summary, now)

        def attempt_recovery():
            try:
                http("POST", f"{ALERT_BASE2}/api/events",
                     {"fingerprint": fp, "seq": 2, "kind": "recovery", "summary": "cpu ok"})
            except Exception as exc:  # connection dropped after the server-side raise
                failures.append(repr(exc))

        core_service._enqueue_notification = flaky
        try:
            poster = threading.Thread(target=attempt_recovery)
            poster.start()
            self.assertTrue(entered.wait(5), "injected failure never ran")

            # The write transaction is open and doomed. Every refresh must still
            # render the committed firing state as one coherent snapshot.
            for _ in range(5):
                snap = self._snapshot()
                alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
                self.assertEqual(alert["status"], "firing")
                self.assertEqual(alert["current_seq"], 1)
                self.assertEqual(alert["summary"], "cpu high")
                self.assertIsNone(alert["recovery_notify_status"])
                self.assertFalse(snap["silence"]["active"])
                self.assertEqual(
                    [n["notification_no"] for n in snap["notifications"]
                     if n["fingerprint"] == fp], [f"{fp}#r1:firing"])
                time.sleep(0.05)
            poster.join(10)
        finally:
            core_service._enqueue_notification = original

        self.assertTrue(failures, "the injected enqueue failure should have surfaced")
        # After the rollback the state is still the committed firing state and
        # no ghost recovery row or event exists.
        snap = self._snapshot()
        alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
        self.assertEqual(alert["status"], "firing")
        self.assertEqual(alert["current_seq"], 1)
        self.assertNotIn(f"{fp}#r1:recovery",
                         [n["notification_no"] for n in snap["notifications"]])
        _, ev = http("GET", f"{ALERT_BASE2}/api/alerts/{fp}/events", expect=200)
        self.assertEqual([e["seq"] for e in ev["events"]], [1])
        # The event can be accepted cleanly afterwards, with both notifications.
        self._post(fp, 2, "recovery", "cpu ok")
        snap = self._snapshot()
        alert = next(a for a in snap["alerts"] if a["fingerprint"] == fp)
        self.assertEqual(alert["status"], "recovered")
        self.assertEqual(alert["current_seq"], 2)


def outbox_rows_for(fp, base):
    _, data = http("GET", f"{base}/api/notifications")
    return [n for n in data["notifications"] if n["fingerprint"] == fp]


if __name__ == "__main__":
    unittest.main(verbosity=2)
