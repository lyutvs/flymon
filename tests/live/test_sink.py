"""Event sinks never block or raise; the HTTP sink delivers in order when a viewer listens."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flymon.live.sink import HttpEventSink, NullSink, RecordingSink


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]            # nothing listens here once the socket closes


@pytest.fixture
def receiver():
    """A bare POST receiver: records each request body's events, in arrival order."""
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.extend(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", got
    srv.shutdown()
    srv.server_close()


def _ev(i):
    return {"v": 1, "fly": "fm-a", "type": "battle_start", "battle_tag": f"b{i}", "turn": 0}


def test_null_and_recording_sinks():
    NullSink().emit(_ev(0))
    rec = RecordingSink()
    rec.emit(_ev(0)); rec.emit(_ev(1))
    assert [e["battle_tag"] for e in rec.events] == ["b0", "b1"]


def test_unreachable_viewer_costs_nothing_and_counts_drops():
    sink = HttpEventSink(f"http://127.0.0.1:{_closed_port()}")
    t0 = time.perf_counter()
    for i in range(1000):
        sink.emit(_ev(i))
    assert time.perf_counter() - t0 < 0.5
    sink.close(timeout_s=10)
    assert (sink.sent, sink.dropped) == (0, 1000)


def test_full_queue_and_unserialisable_events_are_dropped():
    sink = HttpEventSink("http://127.0.0.1:1", max_queue=5, autostart=False)
    for i in range(8):
        sink.emit(_ev(i))
    sink.emit({**_ev(9), "detail": {"x": float("nan")}})
    sink.emit({**_ev(9), "detail": {1, 2}})
    assert sink.q.qsize() == 5
    assert sink.dropped == 3 + 2


def test_delivers_in_order_and_snapshots_on_emit(receiver):
    url, got = receiver
    sink = HttpEventSink(url, batch=7)
    ev = _ev(0)
    sink.emit(ev)
    ev["battle_tag"] = "mutated after emit"
    for i in range(1, 50):
        sink.emit(_ev(i))
    sink.close(timeout_s=10)
    assert [e["battle_tag"] for e in got] == [f"b{i}" for i in range(50)]
    assert (sink.sent, sink.dropped) == (50, 0)
