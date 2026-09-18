"""Viewer server: loopback only, events in -> snapshot + SSE out in order, bounded memory, bad input survives."""
import http.client
import json
import urllib.error
import urllib.request

import pytest

from flymon.live.server import Hub, ViewerServer


def _ev(fly, tag, etype="battle_start", turn=0, **f):
    return {"v": 1, "fly": fly, "type": etype, "battle_tag": tag, "turn": turn, **f}


def _post(url, payload, raw: bytes | None = None):
    body = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(url + "/events", data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as r:
        return r.status, r.headers["Content-Type"], r.read()


class SSE:
    """Reads `event:`/`data:` messages from /stream, skipping `: ping` comments."""

    def __init__(self, url):
        host, port = url.removeprefix("http://").split(":")
        self.conn = http.client.HTTPConnection(host, int(port), timeout=5)
        self.conn.request("GET", "/stream")
        self.resp = self.conn.getresponse()
        assert self.resp.status == 200
        assert self.resp.getheader("Content-Type") == "text/event-stream"

    def next(self):
        name, data = "message", None
        while True:
            line = self.resp.fp.readline().decode().rstrip("\n")
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
            elif line == "" and data is not None:
                return name, data

    def close(self):
        self.conn.close()


@pytest.fixture
def viewer(tmp_path):
    (tmp_path / "index.html").write_text("<h1>live</h1>")
    (tmp_path / "app.js").write_text("// app")
    (tmp_path / "secret.txt").write_text("no")
    with ViewerServer(port=0, keep_battles=2, web_root=tmp_path) as srv:
        yield srv


def test_binds_loopback_only(viewer):
    assert viewer.httpd.server_address[0] == "127.0.0.1"
    assert viewer.url.startswith("http://127.0.0.1:")


def test_post_then_state_keeps_arrival_order(viewer):
    evs = [_ev("fm-a", "b1"), _ev("fm-a", "b1", "decision", 1, decider="fly", coach_kind="attack",
                                     candidates=["x", "y"], chosen="x"), _ev("fm-b", "b2")]
    assert _post(viewer.url, evs[:2]) == (200, {"accepted": 2, "ignored": 0})
    assert _post(viewer.url, evs[2]) == (200, {"accepted": 1, "ignored": 0})
    status, ctype, body = _get(viewer.url, "/state")
    assert (status, ctype) == (200, "application/json")
    assert json.loads(body) == {"v": 1, "ignored": 0, "events": evs}


def test_stream_sends_snapshot_then_live_events_in_order(viewer):
    _post(viewer.url, [_ev("fm-a", "b1")])
    sse = SSE(viewer.url)
    try:
        name, snap = sse.next()
        assert name == "state" and snap["events"] == [_ev("fm-a", "b1")]
        live = [_ev("fm-a", "b1", "decision", t, decider="coach", coach_kind="switch", candidates=[], chosen=None)
                for t in range(1, 6)]
        for ev in live:
            _post(viewer.url, [ev])
        assert [sse.next() for _ in live] == [("message", ev) for ev in live]
    finally:
        sse.close()


def test_keeps_only_the_most_recent_battles_per_fly(viewer):
    for tag in ("b1", "b2", "b3"):
        _post(viewer.url, [_ev("fm-a", tag), _ev("fm-b", tag)])
    tags = {(e["fly"], e["battle_tag"]) for e in viewer.hub.state()["events"]}
    assert tags == {("fm-a", "b2"), ("fm-a", "b3"), ("fm-b", "b2"), ("fm-b", "b3")}


def test_bad_input_is_refused_and_the_server_keeps_working(viewer):
    status, body = _post(viewer.url, None, raw=b"{not json")
    assert status == 400 and "bad JSON" in body["error"]
    assert _post(viewer.url, [{"v": 2, "fly": "fm-a", "type": "battle_start", "battle_tag": "b"},
                              {"v": 1, "fly": "fm-a", "type": "spikes", "battle_tag": "b"},
                              "nope", _ev("fm-a", "b9")]) == (200, {"accepted": 1, "ignored": 3})
    assert viewer.hub.state()["ignored"] == 3
    assert _post(viewer.url, [_ev("fm-a", "b10")])[0] == 200


def test_static_files_and_no_escape_from_web_root(viewer):
    assert _get(viewer.url, "/") == (200, "text/html; charset=utf-8", b"<h1>live</h1>")
    assert _get(viewer.url, "/app.js")[1] == "text/javascript; charset=utf-8"
    for path in ("/secret.txt", "/../server.py", "/%2e%2e/secret.txt", "/missing.js"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(viewer.url, path)
        assert e.value.code == 404


def test_a_full_subscriber_is_dropped_not_blocking():
    hub = Hub(keep_battles=3, subscriber_queue=2)
    sub, _ = hub.subscribe()
    assert hub.add([_ev("fm-a", "b1", turn=t) for t in range(5)]) == (5, 0)
    assert sub.closed and not hub.subscribers
