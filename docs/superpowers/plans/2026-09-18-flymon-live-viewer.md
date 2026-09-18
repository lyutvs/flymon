# FlyMon 라이브 뷰어 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 배틀이 도는 동안 한 마리의 배틀 화면과 그 턴의 결정·귀속 기록을 한 페이지에서 보고, 뇌(M3)는 뷰어 수정 없이 붙일 수 있게 한다.

**Architecture:** `FlyCoachPlayer`와 (나중에) 뇌가 이벤트를 `EventSink`로 내보내고, `HttpEventSink`가 로컬 `ViewerServer`에 POST 한다.
서버는 마리별 최근 배틀을 들고 SSE로 페이지에 흘려보낸다. 페이지는 Showdown의 공식 리플레이 렌더러(`replay-embed.js`)에 원시 프로토콜을
먹여 배틀을 그리고, 결정·귀속·뇌 trace 패널을 함께 그린다.

**Tech Stack:** Python 3.13 표준 라이브러리(`http.server`, `queue`, `threading`, `urllib`) + numpy, poke-env 0.16.1,
브라우저(빌드 도구·프레임워크 없음, SVG 직접 그리기), Showdown `replay-embed.js`(Smogon 호스팅).

**Spec:** `docs/superpowers/specs/2026-09-17-flymon-live-viewer-design.md` — 실행자는 계획과 스펙을 함께 읽는다.

## Global Constraints

- 이 작업은 **실험이 아니다**: `results/`에 판정용 산출물을 쓰지 않고, M0d·M2의 어떤 판정에도 쓰지 않는다.
- **관찰 불변**: `event_sink=None`·`turn_delay_s=0`이면 `FlyCoachPlayer`의 동작과 JSONL은 이전과 같다. `emit`은 절대 예외를 올리지 않는다.
- `pilot_no_brain.py`(M1 게이트)와 `flymon/brain/`은 건드리지 않는다.
- 새 파이썬 의존성 금지(표준 라이브러리 + 이미 있는 numpy·poke-env). 페이지에 외부 JS 라이브러리 금지.
- 서버는 `127.0.0.1`에만 바인딩한다.
- 페이지 스크립트에서 전역 이름 `$`를 만들지 않는다(렌더러의 jQuery를 가려 `Replays.init()`이 죽는다).
- 브랜치 `live-viewer`에서만 작업한다(`open-fly-brain-connectome`에는 다른 세션이 M0d를 커밋 중).
- 확인용 배틀은 몇 판만 돌린다(같은 기계에서 M0d 실데이터 실행이 돈다).
- 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

**이 계획의 코드는 전부 스크래치패드 사본에서 실제로 실행해 통과시킨 것이다**(새 테스트 35개가 통과하고, 완료 기준 1–3을 실제로 확인했다). 그대로 옮기면 된다.

---

### Task 1: 이벤트와 싱크

**Files:**
- Create: `flymon/live/__init__.py`, `flymon/live/events.py`, `flymon/live/sink.py`
- Test: `tests/live/test_events.py`, `tests/live/test_sink.py`

**Interfaces:**
- Consumes: 없음
- Produces: `make_event(etype, fly, battle_tag, turn, **fields) -> dict`, `to_json(obj) -> str`, `VERSION = 1`,
  `REQUIRED`(type → 필수 필드), `PHASES`, `SERIES_KINDS`; `EventSink`(프로토콜, `emit(event) -> None`), `NullSink`,
  `RecordingSink`(`.events`), `HttpEventSink(url, max_queue=10_000, batch=200, timeout_s=1.0, autostart=True)`
  (`.emit`, `.close(timeout_s)`, `.dropped`, `.sent`, `.q`)

- [ ] **Step 1: 실패하는 테스트 쓰기**

`tests/live/test_events.py`:

```python
"""Viewer event contract: required fields per type, version, JSON with numpy values."""
import json

import numpy as np
import pytest

from flymon.live.events import REQUIRED, VERSION, make_event, to_json


def test_common_fields_and_version():
    ev = make_event("decision", "fm-a", "battle-gen1ou-1", np.int64(3), decider="fly", coach_kind="attack",
                    candidates=["surf", "earthquake"], chosen="surf")
    assert VERSION == 1
    assert ev == {"v": 1, "fly": "fm-a", "type": "decision", "battle_tag": "battle-gen1ou-1", "turn": 3,
                  "decider": "fly", "coach_kind": "attack", "candidates": ["surf", "earthquake"], "chosen": "surf"}
    assert type(ev["turn"]) is int


@pytest.mark.parametrize("etype", sorted(t for t, req in REQUIRED.items() if req))
def test_each_required_field_is_enforced(etype):
    full = {"decider": "fly", "coach_kind": "attack", "candidates": [], "chosen": None, "outcome": {},
            "phase": "decide", "slot": 0, "series": [], "won": True, "turns": 5, "lines": ["|turn|1"]}
    fields = {k: full[k] for k in REQUIRED[etype]}
    make_event(etype, "fm-a", "b", 1, **fields)
    for k in REQUIRED[etype]:
        with pytest.raises(ValueError, match=k):
            make_event(etype, "fm-a", "b", 1, **{f: v for f, v in fields.items() if f != k})


def test_protocol_lines_must_be_strings():
    make_event("protocol", "fm-a", "b", 1, lines=["|init|battle", "|turn|1"])
    for bad in ("|turn|1", [b"|turn|1"], [["|turn|1"]], None):
        with pytest.raises(ValueError, match="protocol lines"):
            make_event("protocol", "fm-a", "b", 1, lines=bad)


def test_unknown_type_and_bad_trace_are_rejected():
    with pytest.raises(ValueError, match="unknown event type"):
        make_event("spikes", "fm-a", "b", 1)
    ok = {"path": "rate_hz/kc", "kind": "scalar", "points": [[0.0, 1.0]]}
    make_event("trace", "fm-a", "b", 1, phase="reinforce", slot=None, series=[ok])
    with pytest.raises(ValueError, match="phase"):
        make_event("trace", "fm-a", "b", 1, phase="read", slot=0, series=[ok])
    for bad in ({**ok, "kind": "raster"}, {**ok, "path": 3}, {**ok, "points": None}, "rate_hz/kc"):
        with pytest.raises(ValueError, match="bad trace series"):
            make_event("trace", "fm-a", "b", 1, phase="decide", slot=0, series=[bad])


def test_detail_travels_as_given():
    detail = {"V": [0.5, -0.25], "readout": {"mbon": "MBON13"}}
    ev = make_event("decision", "fm-a", "b", 2, decider="fly", coach_kind="attack", candidates=["a", "b"],
                    chosen="a", detail=detail)
    assert ev["detail"] is detail


def test_to_json_accepts_numpy_and_refuses_nan():
    text = to_json({"V": np.array([0.5, 1.0], np.float32), "n": np.int32(4), "x": np.float64(0.25)})
    assert json.loads(text) == {"V": [0.5, 1.0], "n": 4, "x": 0.25}
    with pytest.raises(ValueError):
        to_json({"V": float("nan")})
    with pytest.raises(TypeError):
        to_json({"s": {1, 2}})
```

`tests/live/test_sink.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/live -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.live'`

- [ ] **Step 3: 구현**

`flymon/live/__init__.py`:

```python
"""Live viewer: battle and brain events streamed to a local page while flies play (design 2026-09-17)."""
```

`flymon/live/events.py`:

```python
"""Viewer events: one JSON object per thing that happened, joined on (fly, battle_tag, turn).

This is the contract the brain plugs into (live viewer design, section 4). Adding a field does not
bump VERSION; the page ignores fields and types it does not know.
"""
from __future__ import annotations

import json

import numpy as np

VERSION = 1
REQUIRED = {
    "battle_start": (),
    "decision": ("decider", "coach_kind", "candidates", "chosen"),
    "outcome": ("outcome",),
    "protocol": ("lines",),
    "trace": ("phase", "slot", "series"),
    "battle_end": ("won", "turns"),
}
PHASES = ("decide", "reinforce")
SERIES_KINDS = ("scalar", "bars", "text")


def make_event(etype: str, fly: str, battle_tag: str, turn: int, **fields) -> dict:
    if etype not in REQUIRED:
        raise ValueError(f"unknown event type {etype!r}")
    missing = [k for k in REQUIRED[etype] if k not in fields]
    if missing:
        raise ValueError(f"{etype} event is missing {missing}")
    if etype == "protocol":
        lines = fields["lines"]
        if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
            raise ValueError("protocol lines must be a list of strings")
    if etype == "trace":
        if fields["phase"] not in PHASES:
            raise ValueError(f"trace phase must be one of {PHASES}, got {fields['phase']!r}")
        for s in fields["series"]:
            if (not isinstance(s, dict) or s.get("kind") not in SERIES_KINDS or not isinstance(s.get("path"), str)
                    or not isinstance(s.get("points"), list)):
                raise ValueError(f"bad trace series {s!r}")
    return {"v": VERSION, "fly": fly, "type": etype, "battle_tag": battle_tag, "turn": int(turn), **fields}


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def to_json(obj) -> str:
    """json.dumps that also accepts numpy scalars and arrays (the brain's detail values are numpy)."""
    return json.dumps(obj, default=_plain, allow_nan=False)
```

`flymon/live/sink.py`:

```python
"""Event sinks. `emit` never blocks and never raises: a battle must not wait on, or die for, the viewer."""
from __future__ import annotations

import queue
import threading
import urllib.request
from typing import Protocol

from .events import to_json


class EventSink(Protocol):
    def emit(self, event: dict) -> None: ...


class NullSink:
    dropped = 0

    def emit(self, event: dict) -> None:
        pass

    def close(self, timeout_s: float = 5.0) -> None:
        pass


class RecordingSink:
    """Keeps every event in memory (tests)."""

    dropped = 0

    def __init__(self):
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)

    def close(self, timeout_s: float = 5.0) -> None:
        pass


class HttpEventSink:
    """Serialises on emit (a snapshot: later mutation of the event does not leak), queues the text, and a
    daemon thread POSTs batches to `<url>/events`. A full queue, an unserialisable event or an unreachable
    viewer drops events and counts them in `dropped`."""

    def __init__(self, url: str, max_queue: int = 10_000, batch: int = 200, timeout_s: float = 1.0,
                 autostart: bool = True):
        self.url = url.rstrip("/") + "/events"
        self.batch, self.timeout_s = batch, timeout_s
        self.q: queue.Queue[str] = queue.Queue(max_queue)
        self.dropped = 0
        self.sent = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="flymon-live-sink", daemon=True)
        if autostart:
            self.start()

    def start(self) -> None:
        self._thread.start()

    def emit(self, event: dict) -> None:
        try:
            self.q.put_nowait(to_json(event))
        except Exception:                    # queue.Full, TypeError, ValueError (NaN)
            self._count("dropped", 1)

    def close(self, timeout_s: float = 5.0) -> None:
        """Send what is queued (best effort within `timeout_s`), then stop the thread."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout_s)

    def _count(self, name: str, n: int) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def _run(self) -> None:
        while True:
            try:
                items = [self.q.get(timeout=0.1)]
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            while len(items) < self.batch:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            self._post(items)

    def _post(self, items: list[str]) -> None:
        body = ("[" + ",".join(items) + "]").encode()
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                r.read()
            self._count("sent", len(items))
        except Exception:
            self._count("dropped", len(items))
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/live -q`
Expected: PASS (14 passed)

- [ ] **Step 5: 커밋**

```bash
git add flymon/live/__init__.py flymon/live/events.py flymon/live/sink.py tests/live/test_events.py tests/live/test_sink.py
git commit -m "feat(live): viewer event contract and non-blocking sinks"
```

---

### Task 2: WebSink (뇌 trace 접속점)

**Files:**
- Create: `flymon/live/trace.py`
- Test: `tests/live/test_trace.py`

**Interfaces:**
- Consumes: `make_event`(Task 1), `RecordingSink`(Task 1), `flymon.brain.viz.SpikeTap`·`RerunSink`(기존)
- Produces: `WebSink(event_sink, fly)` — `scalar(path, t_ms, value)`, `bars(path, t_ms, values)`, `text(path, t_ms, msg)`,
  `flush(battle_tag, turn, phase, slot) -> dict | None`

- [ ] **Step 1: 실패하는 테스트 쓰기**

`tests/live/test_trace.py`:

```python
"""WebSink speaks the viz.py sink interface and turns one presentation into one trace event."""
import inspect

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.viz import RerunSink, SpikeTap
from flymon.live.sink import RecordingSink
from flymon.live.trace import WebSink


def test_same_method_signatures_as_rerun_sink():
    for name in ("scalar", "bars", "text"):
        ours = list(inspect.signature(getattr(WebSink, name)).parameters)
        theirs = list(inspect.signature(getattr(RerunSink, name)).parameters)
        assert ours == theirs, name


def test_flush_sends_one_trace_and_empties_the_buffer():
    rec = RecordingSink()
    web = WebSink(rec, "fm-a")
    assert web.flush("b", 1, "decide", 0) is None and rec.events == []
    web.scalar("rate_hz/kc", 100, np.float32(4.5))
    web.scalar("rate_hz/kc", 200, 5)
    web.bars("cell_hz/mbon", 200, np.array([1, 2, 3], np.int64))
    web.text("events", 150, "odour on")
    ev = web.flush("battle-gen1ou-3", 7, "decide", 1)
    assert rec.events == [ev]
    assert {k: ev[k] for k in ("type", "fly", "battle_tag", "turn", "phase", "slot")} == \
        {"type": "trace", "fly": "fm-a", "battle_tag": "battle-gen1ou-3", "turn": 7, "phase": "decide", "slot": 1}
    assert ev["series"] == [
        {"path": "rate_hz/kc", "kind": "scalar", "points": [[100.0, 4.5], [200.0, 5.0]]},
        {"path": "cell_hz/mbon", "kind": "bars", "points": [[200.0, [1.0, 2.0, 3.0]]]},
        {"path": "events", "kind": "text", "points": [[150.0, "odour on"]]},
    ]
    assert web.flush("battle-gen1ou-3", 7, "decide", 2) is None
    assert len(rec.events) == 1


def _engine(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    return pops, Engine(c, pops, p, seed=0)


def test_tapping_into_a_web_sink_leaves_spikes_bit_identical(synthetic_connectome):
    runs = []
    for tapped in (False, True):
        pops, eng = _engine(synthetic_connectome)
        seen = []
        eng.on_step = lambda e, fired, seen=seen: seen.append(fired.copy())
        rec = RecordingSink()
        web = WebSink(rec, "fm-a")
        if tapped:
            SpikeTap(eng, {"kc": pops.kc, "mbon": pops.mbon}, web, every=10, per_cell={"mbon": pops.mbon})
        eng.set_ext(pops.kc[:10], 60.0)
        eng.run(50)
        runs.append((seen, web.flush("b", 1, "decide", 0)))
    (plain, none), (observed, ev) = runs
    assert none is None
    assert len(plain) == len(observed) == 50
    assert all(np.array_equal(a, b) for a, b in zip(plain, observed))
    assert [len(s["points"]) for s in ev["series"]] == [5, 5, 5]
    assert np.concatenate(observed).size > 0      # the stimulus really made spikes
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/live/test_trace.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.live.trace'`

- [ ] **Step 3: 구현**

`flymon/live/trace.py`:

```python
"""WebSink: the viz.py sink interface, buffered per presentation and sent as one `trace` event.

M3 wires it as `SpikeTap(engine, groups, WebSink(event_sink, fly))` and calls `flush(...)` when a
presentation ends. The tap's clock runs across engine.reset(); the page rebases each trace to 0.
"""
from __future__ import annotations

import numpy as np

from .events import make_event


class WebSink:
    def __init__(self, event_sink, fly: str):
        self.event_sink, self.fly = event_sink, fly
        self._series: dict[tuple[str, str], list] = {}

    def scalar(self, path: str, t_ms: float, value) -> None:
        self._add("scalar", path, t_ms, float(value))

    def bars(self, path: str, t_ms: float, values) -> None:
        self._add("bars", path, t_ms, [float(x) for x in np.asarray(values, np.float64).ravel()])

    def text(self, path: str, t_ms: float, msg: str) -> None:
        self._add("text", path, t_ms, str(msg))

    def _add(self, kind: str, path: str, t_ms: float, value) -> None:
        self._series.setdefault((path, kind), []).append([float(t_ms), value])

    def flush(self, battle_tag: str, turn: int, phase: str, slot: int | None) -> dict | None:
        """Send everything buffered since the last flush as one trace event; nothing buffered sends nothing."""
        if not self._series:
            return None
        series = [{"path": p, "kind": k, "points": pts} for (p, k), pts in self._series.items()]
        self._series = {}
        event = make_event("trace", self.fly, battle_tag, turn, phase=phase, slot=slot, series=series)
        self.event_sink.emit(event)
        return event
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/live/test_trace.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add flymon/live/trace.py tests/live/test_trace.py
git commit -m "feat(live): WebSink turns one presentation into one trace event"
```

---

### Task 3: 뷰어 서버

**Files:**
- Create: `flymon/live/server.py`
- Test: `tests/live/test_server.py`

**Interfaces:**
- Consumes: `REQUIRED`·`VERSION`·`to_json`(Task 1)
- Produces: `Hub(keep_battles=3, subscriber_queue=10_000)`(`add(events) -> (accepted, ignored)`, `state()`,
  `subscribe() -> (Subscriber, snapshot)`, `unsubscribe(sub)`, `close_all()`), `Subscriber`(`.q`, `.closed`),
  `ViewerServer(port=8765, host="127.0.0.1", keep_battles=3, web_root=WEB_ROOT)`(`.start()`, `.stop()`, `.url`, `.hub`,
  컨텍스트 매니저), `WEB_ROOT`

- [ ] **Step 1: 실패하는 테스트 쓰기**

`tests/live/test_server.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/live/test_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.live.server'`

- [ ] **Step 3: 구현**

`flymon/live/server.py`:

```python
"""Local viewer server: flies POST events, pages subscribe over SSE. Loopback only, standard library only.

  POST /events   a JSON event or an array of them -> {"accepted": n, "ignored": m}
  GET  /stream   SSE: first an `event: state` snapshot, then each accepted event as `data:`
  GET  /state    the same snapshot as JSON
  GET  /…        static files from web/live (index.html for /)

The snapshot and the subscription are taken under one lock, so a page never misses or repeats an event.
"""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .events import REQUIRED, VERSION, to_json

WEB_ROOT = Path(__file__).resolve().parents[2] / "web" / "live"
MAX_BODY = 32 * 1024 * 1024
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}


class Subscriber:
    def __init__(self, max_queue: int):
        self.q: queue.Queue[dict] = queue.Queue(max_queue)
        self.closed = False


class Hub:
    """Each fly's most recent `keep_battles` battles, and the live subscribers."""

    def __init__(self, keep_battles: int = 3, subscriber_queue: int = 10_000):
        self.keep, self.sub_queue = keep_battles, subscriber_queue
        self.lock = threading.Lock()
        self.flies: dict[str, dict[str, list[dict]]] = {}
        self.subscribers: set[Subscriber] = set()
        self.ignored = 0

    @staticmethod
    def valid(ev) -> bool:
        return (isinstance(ev, dict) and ev.get("v") == VERSION and ev.get("type") in REQUIRED
                and isinstance(ev.get("fly"), str) and isinstance(ev.get("battle_tag"), str))

    def add(self, events: list) -> tuple[int, int]:
        accepted = ignored = 0
        with self.lock:
            for ev in events:
                if not self.valid(ev):
                    ignored += 1
                    continue
                accepted += 1
                battles = self.flies.setdefault(ev["fly"], {})
                battles.setdefault(ev["battle_tag"], []).append(ev)
                while len(battles) > self.keep:
                    del battles[next(iter(battles))]
                for sub in list(self.subscribers):
                    try:
                        sub.q.put_nowait(ev)
                    except queue.Full:           # a stuck page: drop it, EventSource reconnects and resyncs
                        sub.closed = True
                        self.subscribers.discard(sub)
            self.ignored += ignored
        return accepted, ignored

    def _snapshot(self) -> dict:
        events = [ev for battles in self.flies.values() for evs in battles.values() for ev in evs]
        return {"v": VERSION, "ignored": self.ignored, "events": events}

    def state(self) -> dict:
        with self.lock:
            return self._snapshot()

    def subscribe(self) -> tuple[Subscriber, dict]:
        with self.lock:
            sub = Subscriber(self.sub_queue)
            self.subscribers.add(sub)
            return sub, self._snapshot()

    def unsubscribe(self, sub: Subscriber) -> None:
        with self.lock:
            sub.closed = True
            self.subscribers.discard(sub)

    def close_all(self) -> None:
        with self.lock:
            for sub in self.subscribers:
                sub.closed = True
            self.subscribers.clear()


class _Handler(BaseHTTPRequestHandler):
    hub: Hub
    web_root: Path
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, to_json(obj).encode(), "application/json")

    def do_POST(self):
        if urlsplit(self.path).path != "/events":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self.close_connection = True
            return self._json(413, {"error": "body too large"})
        try:
            payload = json.loads(self.rfile.read(n))
        except (ValueError, UnicodeDecodeError) as e:
            return self._json(400, {"error": f"bad JSON: {e}"})
        events = payload if isinstance(payload, list) else [payload]
        accepted, ignored = self.hub.add(events)
        self._json(200, {"accepted": accepted, "ignored": ignored})

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/stream":
            return self._stream()
        if path == "/state":
            return self._json(200, self.hub.state())
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        root = self.web_root.resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file() or target.suffix not in CONTENT_TYPES:
            return self._json(404, {"error": "not found"})
        self._send(200, target.read_bytes(), CONTENT_TYPES[target.suffix])

    def _stream(self) -> None:
        sub, snapshot = self.hub.subscribe()
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"event: state\ndata: " + to_json(snapshot).encode() + b"\n\n")
            self.wfile.flush()
            while not sub.closed:
                try:
                    ev = sub.q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                else:
                    self.wfile.write(b"data: " + to_json(ev).encode() + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.hub.unsubscribe(sub)


class ViewerServer:
    def __init__(self, port: int = 8765, host: str = "127.0.0.1", keep_battles: int = 3,
                 web_root: Path = WEB_ROOT):
        self.hub = Hub(keep_battles)
        handler = type("Handler", (_Handler,), {"hub": self.hub, "web_root": Path(web_root)})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "ViewerServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="flymon-viewer", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.hub.close_all()
        if self._thread is not None:
            self.httpd.shutdown()
            self._thread.join(5)
            self._thread = None
        self.httpd.server_close()

    def __enter__(self) -> "ViewerServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/live/test_server.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add flymon/live/server.py tests/live/test_server.py
git commit -m "feat(live): loopback viewer server with SSE fan-out and bounded history"
```

---

### Task 4: 플레이어 훅 (이벤트·detail·프로토콜·관전 지연)

**Files:**
- Modify: `flymon/battle/fly_coach_player.py`
- Test: `tests/battle/test_fly_coach_player.py`(추가), `tests/battle/test_barrier.py`(추가)

**Interfaces:**
- Consumes: `make_event`(Task 1), `RecordingSink`(Task 1)
- Produces: `FlyCoachPlayer(..., event_sink=None, turn_delay_s=0.0)`, `.emit_errors`;
  이벤트 `battle_start`/`protocol`/`decision`(+`detail`)/`outcome`/`battle_end`

- [ ] **Step 1: 실패하는 테스트 쓰기**

`tests/battle/test_fly_coach_player.py` — 맨 위 import에 `import time`과
`from flymon.live.sink import RecordingSink`를 더하고, 파일 끝에 붙인다:

```python
# ---- live viewer events -----------------------------------------------------------------------
def _record(event):
    """A viewer event rewritten as the JSONL record it mirrors."""
    rec = {"battle_tag": event["battle_tag"], "turn": event["turn"], "kind": event["type"]}
    rec.update({k: v for k, v in event.items() if k not in ("v", "fly", "type", "battle_tag", "turn", "detail")})
    return rec


@requires_server
async def test_events_mirror_the_jsonl_log_between_start_and_end(server, tmp_path):
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    sink = RecordingSink()
    me = FlyCoachPlayer(provider=RandomProvider(3), coach=Coach(), barrier=None, log_path=tmp_path / "ev.jsonl",
                        event_sink=sink, account_configuration=AccountConfiguration("fm-test-ev", None),
                        battle_format="gen1ou", server_configuration=cfg, team=team_export(POOL[:6]),
                        max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-ev-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    try:
        await me.battle_against(opp, n_battles=2)
        recs = [json.loads(line) for line in (tmp_path / "ev.jsonl").read_text().splitlines()]
        evs = sink.events
        assert all(e["v"] == 1 and e["fly"] == "fm-test-ev" for e in evs)
        assert [_record(e) for e in evs if e["type"] in ("decision", "outcome")] == recs
        assert me.emit_errors == 0
        for tag, battle in me.battles.items():
            mine = [e for e in evs if e["battle_tag"] == tag]
            types = [e["type"] for e in mine]
            assert types.count("battle_start") == 1 and types.count("battle_end") == 1
            assert mine[0]["type"] == "battle_start"
            end = types.index("battle_end")                     # the server may still send protocol after it
            assert not ({"decision", "outcome", "trace"} & set(types[end + 1:]))
            assert mine[end]["won"] == battle.won and mine[end]["turns"] == mine[end]["turn"] == battle.turn
            assert all("detail" not in e for e in mine)           # RandomProvider reports no detail
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()


class _DetailProvider:
    async def decide(self, battle, cands, ctx):
        ctx["detail"] = {"V": [0.1 * i for i in range(len(cands))]}
        return len(cands) - 1


def _offline(name, **kw):
    return FlyCoachPlayer(coach=Coach(), account_configuration=AccountConfiguration(name, None),
                          battle_format="gen1ou", start_listening=False, **kw)


async def test_without_a_sink_no_event_is_built(make_battle, monkeypatch):
    import flymon.battle.fly_coach_player as fcp

    def boom(*a, **k):
        raise AssertionError("make_event called without a sink")

    monkeypatch.setattr(fcp, "make_event", boom)
    player = _offline("fm-offline-nosink", provider=_DetailProvider())
    assert player.event_sink is None and player.turn_delay_s == 0.0
    await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert player.emit_errors == 0


async def test_provider_detail_rides_on_the_decision_event(make_battle):
    sink = RecordingSink()
    player = _offline("fm-offline-detail", provider=_DetailProvider(), event_sink=sink)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await player.choose_move(battle)
    (ev,) = sink.events
    assert ev["type"] == "decision" and ev["decider"] == "fly" and len(ev["candidates"]) == 3
    assert ev["detail"] == {"V": [0.0, 0.1, 0.2]} and ev["chosen"] == ev["candidates"][2]


async def test_detail_through_the_barrier(make_battle):
    async def run_batch(reqs):
        for r in reqs:
            r.context["detail"] = {"p": [1.0, 0.0, 0.0]}
        return [0] * len(reqs)

    sink = RecordingSink()
    player = _offline("fm-offline-bar", provider=RandomProvider(0), barrier=BatchBarrier(run_batch, deadline_ms=50),
                      event_sink=sink)
    await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    (ev,) = sink.events
    assert ev["detail"] == {"p": [1.0, 0.0, 0.0]} and ev["chosen"] == ev["candidates"][0]


async def test_a_raising_sink_does_not_stop_the_turn(make_battle):
    class Broken:
        def emit(self, event):
            raise RuntimeError("viewer exploded")

    player = _offline("fm-offline-broken", provider=_DetailProvider(), event_sink=Broken())
    order = await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert order.message.startswith("/choose move ")
    assert player.emit_errors == 1


async def test_turn_delay_holds_the_order_back_after_the_event(make_battle):
    sink = RecordingSink()
    player = _offline("fm-offline-delay", provider=_DetailProvider(), event_sink=sink, turn_delay_s=0.2)
    t0 = time.perf_counter()
    await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert time.perf_counter() - t0 >= 0.2
    assert [e["type"] for e in sink.events] == ["decision"]


@requires_server
async def test_protocol_events_carry_the_battle_for_the_renderer(server, tmp_path):
    """The page replays these lines in Showdown's own renderer, so they must be the raw protocol minus `|request|`."""
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    sink = RecordingSink()
    me = FlyCoachPlayer(provider=RandomProvider(4), coach=Coach(), barrier=None, event_sink=sink,
                        account_configuration=AccountConfiguration("fm-test-proto", None), battle_format="gen1ou",
                        server_configuration=cfg, team=team_export(POOL[:6]), max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-proto-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    try:
        await me.battle_against(opp, n_battles=1)
        (tag,) = list(me.battles)
        lines = [line for e in sink.events if e["type"] == "protocol" and e["battle_tag"] == tag for line in e["lines"]]
        assert lines[0] == "|init|battle"
        assert "|turn|1" in lines
        assert any(line.startswith("|move|") for line in lines)
        assert any(line.startswith("|win|") or line.startswith("|tie") for line in lines)
        assert all(not line.startswith("|request|") for line in lines)
        assert all(line.startswith("|") for line in lines)
        first_start = next(i for i, e in enumerate(sink.events) if e["type"] == "battle_start")
        first_proto = next(i for i, e in enumerate(sink.events) if e["type"] == "protocol")
        assert first_start < first_proto
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()
```

`tests/battle/test_barrier.py` 끝에 붙인다(현재 배리어가 이미 만족하는 계약을 못 박는 테스트):

```python
async def test_detail_filled_by_the_batch_reaches_the_submitter_context():
    """M3's batch runner reports what the brain saw through each request's context (live viewer detail)."""
    async def run_batch(reqs):
        for i, r in enumerate(reqs):
            r.context["detail"] = {"V": [0.5, -0.5], "slot": i}
        return [0] * len(reqs)

    bar = BatchBarrier(run_batch, deadline_ms=10_000)
    bar.register("a")
    ctx = {"battle_tag": "b", "turn": 4}
    assert await bar.submit("a", None, ["x", "y"], ctx) == 0
    assert ctx == {"battle_tag": "b", "turn": 4, "detail": {"V": [0.5, -0.5], "slot": 0}}
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/battle/test_fly_coach_player.py -q`
Expected: FAIL — 새 테스트 7개가 `TypeError: __init__() got an unexpected keyword argument 'event_sink'` 등으로 실패

- [ ] **Step 3: 구현**

`flymon/battle/fly_coach_player.py`에 이 diff를 적용한다:

```diff
--- a/flymon/battle/fly_coach_player.py
+++ b/flymon/battle/fly_coach_player.py
@@ -1,6 +1,7 @@
 """poke-env player: coach decides strategy, a DecisionProvider picks among attack candidates."""
 from __future__ import annotations
 
+import asyncio
 import json
 from dataclasses import asdict
 from pathlib import Path
@@ -8,6 +9,7 @@
 from poke_env.battle import AbstractBattle
 from poke_env.player import Player
 
+from ..live.events import make_event
 from .attribution import TurnAttributor
 from .barrier import BatchBarrier
 from .coach import Coach
@@ -29,12 +31,20 @@
     A forced switch (after a faint) makes the server ask again within the same `turn`, so that turn
     carries a second `"decision"` record with the same `(battle_tag, turn)`: the join between
     decision and outcome records is many-to-one, not one-to-one.
+
+    With an `event_sink` (flymon.live), every JSONL record is also emitted as a viewer event with the same
+    content, bracketed by `battle_start`/`battle_end`; a provider that fills `context["detail"]` has it
+    carried on the decision event. `turn_delay_s` holds each order back so a person can watch the battle.
+    Neither changes what is decided or logged; a sink that raises is counted in `emit_errors`.
     """
 
     def __init__(self, provider: DecisionProvider, coach: Coach, barrier: BatchBarrier | None = None,
-                 log_path: Path | None = None, **player_kwargs):
+                 log_path: Path | None = None, event_sink=None, turn_delay_s: float = 0.0, **player_kwargs):
         super().__init__(**player_kwargs)
         self.provider, self.coach, self.barrier = provider, coach, barrier
+        self.event_sink, self.turn_delay_s = event_sink, float(turn_delay_s)
+        self.emit_errors = 0
+        self._started: set[str] = set()
         self.log_path = Path(log_path) if log_path else None
         if self.log_path:
             self.log_path.parent.mkdir(parents=True, exist_ok=True)
@@ -50,7 +60,11 @@
     async def _handle_battle_message(self, split_messages):
         tag = split_messages[0][0].lstrip(">")
         if not split_messages[0][0].startswith(">game"):  # best-of rooms carry no battle protocol
+            if tag not in self._started:
+                self._started.add(tag)
+                self._emit("battle_start", tag, 0)
             att = self.attributors.setdefault(tag, TurnAttributor())
+            self._emit_protocol(tag, split_messages[1:])
             battle = self._battles.get(tag)
             if battle is not None and battle.player_role:
                 att.my_side = battle.player_role
@@ -64,6 +78,15 @@
         # a `|request|` line in this message makes super() call choose_move synchronously
         await super()._handle_battle_message(split_messages)
 
+    def _emit_protocol(self, battle_tag: str, lines: list) -> None:
+        """Mirror the raw protocol to the viewer, which replays it in the official battle renderer.
+        `|request|` carries my own team sheet and the renderer ignores it, so it stays out."""
+        if self.event_sink is None:
+            return
+        out = ["|".join(line) for line in lines if len(line) >= 2 and line[1] != "request"]
+        if out:
+            self._emit("protocol", battle_tag, self.turns.get(battle_tag, 0), lines=out)
+
     def _close_turn(self, battle_tag: str) -> None:
         """End the turn's attribution block, if I acted, and log its `Outcome` against that turn."""
         att = self.attributors.get(battle_tag)
@@ -71,14 +94,16 @@
             return
         outcome = att.end_turn()
         self.outcomes.setdefault(battle_tag, []).append(outcome)
-        self._write({"battle_tag": battle_tag, "turn": self.turns.get(battle_tag, 0), "kind": "outcome",
-                     "outcome": asdict(outcome)})
+        rec = {"battle_tag": battle_tag, "turn": self.turns.get(battle_tag, 0), "kind": "outcome",
+               "outcome": asdict(outcome)}
+        self._write(rec)
+        self._emit("outcome", battle_tag, rec["turn"], outcome=rec["outcome"])
 
     # ---- decision -------------------------------------------------------------------------
     async def choose_move(self, battle: AbstractBattle):
         decision = self.coach.decide(battle)
         who, cands = route(decision, candidates(battle))
-        chosen = None
+        chosen, detail = None, None
         if who == "fly":
             ctx = {"battle_tag": battle.battle_tag, "turn": battle.turn}
             if self.barrier is not None:
@@ -89,23 +114,38 @@
             if not 0 <= idx < len(cands):
                 raise ValueError(f"provider returned index {idx} for {len(cands)} candidates")
             chosen = cands[idx]
+            detail = ctx.get("detail")
             order = self.create_order(chosen)
         else:
             order = decision.order
-        self._log(battle, who, decision, cands, chosen)
+        self._log(battle, who, decision, cands, chosen, detail)
+        if self.turn_delay_s > 0:
+            await asyncio.sleep(self.turn_delay_s)
         return order
 
-    def _log(self, battle, who, decision, cands, chosen) -> None:
+    def _log(self, battle, who, decision, cands, chosen, detail=None) -> None:
         rec = {"battle_tag": battle.battle_tag, "turn": battle.turn, "kind": "decision", "decider": who,
                "coach_kind": decision.kind, "candidates": [m.id for m in cands],
                "chosen": chosen.id if chosen else (decision.move.id if decision.move else None)}
         self.turn_log.setdefault(battle.battle_tag, []).append(rec)
         self._write(rec)
+        fields = {k: rec[k] for k in ("decider", "coach_kind", "candidates", "chosen")}
+        if detail is not None:
+            fields["detail"] = detail
+        self._emit("decision", rec["battle_tag"], rec["turn"], **fields)
 
     def _write(self, rec: dict) -> None:
         if self.log_path:
             with open(self.log_path, "a") as f:
                 f.write(json.dumps(rec) + "\n")
+
+    def _emit(self, etype: str, battle_tag: str, turn: int, **fields) -> None:
+        if self.event_sink is None:
+            return
+        try:
+            self.event_sink.emit(make_event(etype, self.username, battle_tag, turn, **fields))
+        except Exception:
+            self.emit_errors += 1
 
     # ---- stats ------------------------------------------------------------------------------
     def battle_stats(self, battle_tag: str) -> dict:
@@ -123,5 +163,6 @@
 
     def _battle_finished_callback(self, battle: AbstractBattle) -> None:
         self._close_turn(battle.battle_tag)   # a battle that ends without `|upkeep|` still logs its last turn
+        self._emit("battle_end", battle.battle_tag, battle.turn, won=battle.won, turns=battle.turn)
         if self.barrier is not None:
             self.barrier.unregister(self.player_id)
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/battle/test_fly_coach_player.py tests/battle/test_barrier.py -q`
Expected: PASS (23 passed — 기존 15개 + 새 8개)

- [ ] **Step 5: 커밋**

```bash
git add flymon/battle/fly_coach_player.py tests/battle/test_fly_coach_player.py tests/battle/test_barrier.py
git commit -m "feat(battle): player emits viewer events, provider detail and raw protocol"
```

---

### Task 5: 가짜 뇌와 실행 스크립트

**Files:**
- Create: `flymon/live/fake_brain.py`, `scripts/live_viewer.py`, `scripts/live_battles.py`
- Test: `tests/live/test_fake_brain.py`, `tests/live/test_live_battles.py`

**Interfaces:**
- Consumes: `WebSink`(Task 2), `ViewerServer`(Task 3), `HttpEventSink`·`NullSink`(Task 1), 플레이어 훅(Task 4)
- Produces: `FakeBrainProvider(inner, event_sink, fly, seed=0, ...)`;
  `scripts/live_battles.py`의 `parse_args(argv)`, `fly_name(fly_id)`, `free_port()`, `make_provider(kind, fly_id, sink)`,
  `run_fly(args, fly_id, sbs, cfg, sink)`, `run_battles(args, cfg, sink)`, `main(argv)`

- [ ] **Step 1: 실패하는 테스트 쓰기**

`tests/live/test_fake_brain.py`:

```python
"""The fake brain walks the brain's viewer path: same choice as its inner provider, detail + one trace per candidate."""
import numpy as np

from flymon.live.events import to_json
from flymon.live.fake_brain import FakeBrainProvider
from flymon.live.sink import RecordingSink


class Fixed:
    async def decide(self, battle, candidates, context):
        return 1


async def test_keeps_the_choice_and_reports_detail_and_traces():
    sink = RecordingSink()
    brain = FakeBrainProvider(Fixed(), sink, "fm-live-f00", seed=0)
    ctx = {"battle_tag": "battle-gen1ou-5", "turn": 3}
    assert await brain.decide(None, ["surf", "earthquake", "blizzard"], ctx) == 1
    d = ctx["detail"]
    assert set(d) == {"V", "p", "kc_active_frac"} and all(len(v) == 3 for v in d.values())
    assert int(np.argmax(d["V"])) == 1 and abs(float(np.sum(d["p"])) - 1.0) < 0.01
    to_json(d)                                                   # numpy detail serialises
    assert [(e["type"], e["slot"], e["phase"], e["turn"]) for e in sink.events] == \
        [("trace", s, "decide", 3) for s in range(3)]
    paths = {s["path"]: s for s in sink.events[0]["series"]}
    assert set(paths) == {"events", "rate_hz/alpn", "rate_hz/kc", "rate_hz/mbon", "rate_hz/apl", "cell_hz/mbon"}
    assert len(paths["rate_hz/kc"]["points"]) == 14 and paths["cell_hz/mbon"]["kind"] == "bars"
    starts = [e["series"][0]["points"][0][0] for e in sink.events]
    assert starts == sorted(starts) and starts[1] > starts[0]    # one running clock, like SpikeTap
```

`tests/live/test_live_battles.py`:

```python
"""scripts/live_battles.py end to end: one fake-brain battle reaches a real viewer server over HTTP."""
import importlib.util
from pathlib import Path

import pytest
from poke_env.ps_client import ServerConfiguration

from flymon.battle.server import NODE_BIN, ShowdownServer
from flymon.live.server import ViewerServer
from flymon.live.sink import HttpEventSink

_spec = importlib.util.spec_from_file_location("live_battles", Path("scripts/live_battles.py"))
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)


def test_parse_args_defaults_and_names():
    a = lb.parse_args([])
    assert (a.flies, a.battles, a.provider, a.watch_fly, a.turn_delay, a.viewer) == (2, 2, "fake-brain", 0, 0.0, None)
    assert lb.fly_name(3) == "fm-live-f03"
    with pytest.raises(SystemExit):
        lb.parse_args(["--provider", "brain"])


@pytest.mark.skipif(not NODE_BIN.exists(), reason="run scripts/install_showdown.sh first")
async def test_one_fake_brain_battle_reaches_the_viewer(tmp_path):
    args = lb.parse_args(["--flies", "1", "--battles", "1", "--provider", "fake-brain", "--opponent", "random",
                          "--log-dir", str(tmp_path)])
    with ViewerServer(port=0) as viewer, ShowdownServer(port=lb.free_port()) as srv:
        sink = HttpEventSink(viewer.url)
        cfg = ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")
        (result,) = await lb.run_battles(args, cfg, sink)
        sink.close(timeout_s=10)
        events = viewer.hub.state()["events"]
    assert result == {"fly": "fm-live-f00", "battles": 1, "emit_errors": 0}
    assert sink.dropped == 0 and sink.sent == len(events)
    assert {e["type"] for e in events} == {"battle_start", "protocol", "decision", "outcome", "trace", "battle_end"}
    fly_decisions = [e for e in events if e["type"] == "decision" and e["decider"] == "fly"]
    assert fly_decisions and all(len(e["detail"]["V"]) == len(e["candidates"]) for e in fly_decisions)
    assert (tmp_path / "fly00.jsonl").exists()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/live/test_fake_brain.py tests/live/test_live_battles.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.live.fake_brain'`

- [ ] **Step 3: 구현**

`flymon/live/fake_brain.py`:

```python
"""A stand-in for the M3 brain on the viewer path only. Never use it in an experiment arm.

It keeps the inner provider's choice, then does what the brain will do for the viewer: fills
`context["detail"]` and sends one trace per candidate presentation through a WebSink. The numbers are
made up; what is real is the path they travel.
"""
from __future__ import annotations

import numpy as np

from .trace import WebSink

POPULATIONS = ("alpn", "kc", "mbon", "apl")


class FakeBrainProvider:
    def __init__(self, inner, event_sink, fly: str, seed: int = 0, settle_ms: float = 800.0,
                 read_ms: float = 600.0, every_ms: float = 100.0, n_mbon: int = 6):
        self.inner, self.fly = inner, fly
        self.web = WebSink(event_sink, fly)
        self.rng = np.random.default_rng(seed)
        self.settle_ms, self.read_ms, self.every_ms, self.n_mbon = settle_ms, read_ms, every_ms, n_mbon
        self.clock_ms = 0.0                 # runs on across presentations, like SpikeTap's clock

    async def decide(self, battle, candidates, context) -> int:
        idx = await self.inner.decide(battle, candidates, context)
        n = len(candidates)
        V = self.rng.normal(0.0, 0.5, n)
        V[idx] = V.max() + 0.25             # the made-up values agree with the choice
        p = np.exp(V) / np.exp(V).sum()
        kc = self.rng.uniform(0.04, 0.08, n)
        context["detail"] = {"V": np.round(V, 3), "p": np.round(p, 3), "kc_active_frac": np.round(kc, 4)}
        for slot in range(n):
            self._present(slot, kc[slot])
            self.web.flush(context["battle_tag"], context["turn"], "decide", slot)
        return idx

    def _present(self, slot: int, kc_frac: float) -> None:
        on = self.clock_ms
        steps = int(round((self.settle_ms + self.read_ms) / self.every_ms))
        self.web.text("events", on, f"candidate {slot} on")
        cells = np.zeros(self.n_mbon)
        for k in range(1, steps + 1):
            t = on + k * self.every_ms
            rise = min(1.0, k * self.every_ms / 300.0)
            base = {"alpn": 12.0, "kc": 40.0 * kc_frac, "mbon": 3.5, "apl": 150.0}
            for pop in POPULATIONS:
                self.web.scalar(f"rate_hz/{pop}", t, base[pop] * (0.4 + 0.6 * rise) * self.rng.uniform(0.85, 1.15))
            cells = np.clip(3.5 + self.rng.normal(0.0, 1.5, self.n_mbon) + 4.0 * rise * (slot % 2), 0.0, None)
        self.web.bars("cell_hz/mbon", on + steps * self.every_ms, cells)
        self.clock_ms = on + steps * self.every_ms
```

`scripts/live_viewer.py`:

```python
#!/usr/bin/env python
"""Serve the live viewer page on 127.0.0.1 until Ctrl-C.

    uv run python scripts/live_viewer.py --port 8765
"""
from __future__ import annotations

import argparse
import sys
import time

from flymon.live.server import ViewerServer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--keep-battles", type=int, default=3, help="recent battles kept per fly for late pages")
    args = ap.parse_args(argv)
    with ViewerServer(port=args.port, keep_battles=args.keep_battles) as srv:
        print(f"viewer: {srv.url}/  (live_battles.py prints the URL with ?fly= and &showdown=)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/live_battles.py`:

```python
#!/usr/bin/env python
"""Brain-free battles streamed to the live viewer (not an experiment: nothing here is a result).

    uv run python scripts/live_viewer.py                                   # terminal 1
    uv run python scripts/live_battles.py --flies 2 --battles 2 --provider fake-brain \\
        --watch-fly 0 --turn-delay 2 --viewer http://127.0.0.1:8765        # terminal 2, open the printed URL

Only the watched fly waits `--turn-delay` seconds per order. Without `--viewer` events go nowhere.
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import sys
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from flymon.battle.coach import Coach
from flymon.battle.fly_coach_player import FlyCoachPlayer
from flymon.battle.opponents import make_opponent
from flymon.battle.pool import by_species, team_export
from flymon.battle.providers import MaxDamageProvider, RandomProvider
from flymon.battle.schedule import make_schedule
from flymon.battle.server import ShowdownServer
from flymon.live.fake_brain import FakeBrainProvider
from flymon.live.sink import HttpEventSink, NullSink

PROVIDERS = ("rnd", "max", "fake-brain")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flies", type=int, default=2)
    ap.add_argument("--battles", type=int, default=2)
    ap.add_argument("--provider", choices=PROVIDERS, default="fake-brain")
    ap.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    ap.add_argument("--watch-fly", type=int, default=0)
    ap.add_argument("--turn-delay", type=float, default=0.0)
    ap.add_argument("--viewer", default=None, help="viewer base URL, e.g. http://127.0.0.1:8765")
    ap.add_argument("--showdown-port", type=int, default=0, help="0 = any free port")
    ap.add_argument("--log-dir", default=None, help="write per-fly JSONL here (default: no log)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def fly_name(fly_id: int) -> str:
    return f"fm-live-f{fly_id:02d}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _team(species: list[str]) -> str:
    return team_export([by_species[s] for s in species])


def make_provider(kind: str, fly_id: int, sink):
    if kind == "rnd":
        return RandomProvider(seed=fly_id)
    if kind == "max":
        return MaxDamageProvider()
    return FakeBrainProvider(RandomProvider(seed=fly_id), sink, fly_name(fly_id), seed=fly_id)


async def run_fly(args, fly_id: int, sbs: list, cfg: ServerConfiguration, sink) -> dict:
    me = FlyCoachPlayer(provider=make_provider(args.provider, fly_id, sink), coach=Coach(), barrier=None,
                        log_path=Path(args.log_dir) / f"fly{fly_id:02d}.jsonl" if args.log_dir else None,
                        event_sink=sink, turn_delay_s=args.turn_delay if fly_id == args.watch_fly else 0.0,
                        account_configuration=AccountConfiguration(fly_name(fly_id), None), battle_format="gen1ou",
                        server_configuration=cfg, team=_team(sbs[0].my_team), max_concurrent_battles=1)
    opp = make_opponent(sbs[0].opponent, fly_id, cfg, _team(sbs[0].opp_team))
    try:
        for sb in sbs:
            me.update_team(_team(sb.my_team))
            opp.update_team(_team(sb.opp_team))
            await me.battle_against(opp, n_battles=1)
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()
    return {"fly": fly_name(fly_id), "battles": me.n_finished_battles, "emit_errors": me.emit_errors}


async def run_battles(args, cfg: ServerConfiguration, sink) -> list:
    sched = make_schedule(args.flies, args.battles, args.opponent, seed=args.seed)
    by_fly = {f: [s for s in sched if s.fly_id == f] for f in range(args.flies)}
    return await asyncio.gather(*(run_fly(args, f, sbs, cfg, sink) for f, sbs in by_fly.items()),
                                return_exceptions=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    sink = HttpEventSink(args.viewer) if args.viewer else NullSink()
    port = args.showdown_port or free_port()
    with ShowdownServer(port=port) as srv:
        cfg = ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")
        if args.viewer:
            print(f"open: {args.viewer.rstrip('/')}/?fly={fly_name(args.watch_fly)}", flush=True)
        results = asyncio.run(run_battles(args, cfg, sink))
    sink.close(timeout_s=10)
    failed = [r for r in results if isinstance(r, BaseException)]
    for r in results:
        print(f"fly failed: {r!r}" if isinstance(r, BaseException) else
              f"{r['fly']}: battles={r['battles']} emit_errors={r['emit_errors']}", flush=True)
    print(f"dropped_events={sink.dropped}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/live -q`
Expected: PASS (27 passed)

- [ ] **Step 5: 커밋**

```bash
git add flymon/live/fake_brain.py scripts/live_viewer.py scripts/live_battles.py tests/live/test_fake_brain.py tests/live/test_live_battles.py
git commit -m "feat(live): fake brain on the viewer path, viewer and battle runner scripts"
```

---

### Task 6: 페이지 (`web/live`)

**Files:**
- Create: `web/live/index.html`, `web/live/app.js`, `web/live/style.css`

**Interfaces:**
- Consumes: `GET /stream`(`event: state` 스냅샷 → `data:` 이벤트), `GET /state`(Task 3), 이벤트 형식(Task 1)
- Produces: 없음(페이지가 끝점)

- [ ] **Step 1: 페이지 쓰기**

`web/live/index.html`:

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlyMon Live</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="bar">
    <strong>FlyMon Live</strong>
    <label>보는 마리 <select id="fly"></select></label>
    <span id="battle" class="muted">배틀 없음</span>
    <span id="turn"></span>
    <button id="follow" type="button" hidden>최신 턴 따라가기</button>
    <span id="ignored" class="muted" hidden></span>
    <span id="conn" class="conn off">연결 안 됨</span>
  </header>
  <main>
    <section class="battle-view">
      <!-- the official battle renderer draws into these four divs (replay-embed.js from Showdown) -->
      <div class="wrapper replay-wrapper">
        <div class="battle"></div>
        <div class="battle-log"></div>
        <div class="replay-controls"></div>
        <div class="replay-controls-2"></div>
        <script type="text/plain" class="battle-log-data"></script>
      </div>
      <p id="battle-note" class="hint muted">배틀 화면을 불러오는 중</p>
    </section>
    <section class="side">
      <div class="panel"><h2>① 이번 턴 결정</h2><div id="decision" class="muted">아직 결정 없음</div></div>
      <div class="panel"><h2>② 이번 턴 결과 (귀속)</h2><div id="outcome" class="muted">결과 대기 중</div></div>
    </section>
    <section class="panel wide"><h2>③ 뇌 활동</h2><div id="brain" class="muted">뇌 미연결</div></section>
    <section class="panel wide"><h2>④ 턴 타임라인</h2><div id="timeline" class="chips"></div></section>
  </main>
  <script>
    // Showdown serves the renderer with a daily cache key, the same way its own replay pages do.
    var daily = Math.floor(Date.now() / 1000 / 60 / 60 / 24);
    document.write('<script src="https://play.pokemonshowdown.com/js/replay-embed.js?version' + daily + '"></' + 'script>');
  </script>
  <script src="app.js"></script>
</body>
</html>
```

`web/live/style.css`:

```css
:root {
  --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #6b7385; --line: #dde1e8;
  --fly: #1f7a4d; --coach: #5b6475; --bar: #3d7bd9; --neg: #d9563d; --ok: #1f7a4d; --bad: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171d; --panel: #1c2028; --text: #e6e9ef; --muted: #9aa3b5; --line: #2c323d;
    --fly: #4cc38a; --coach: #a3acbd; --bar: #6aa5ff; --neg: #ff8a70; --ok: #4cc38a; --bad: #ff6b60;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, sans-serif; }
.bar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; padding: 10px 16px;
       background: var(--panel); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 1; }
.muted { color: var(--muted); }
.conn { margin-left: auto; font-size: 12px; }
.conn::before { content: "●"; margin-right: 4px; }
.conn.on::before { color: var(--ok); }
.conn.off::before { color: var(--bad); }
main { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 12px; padding: 12px 16px; }
.wide { grid-column: 1 / -1; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.side { display: flex; flex-direction: column; gap: 12px; }
h2 { font-size: 13px; margin: 0 0 8px; color: var(--muted); font-weight: 600; }
.battle-view { min-width: 0; }
.battle-view .wrapper { max-width: 100%; }
.battle-view .battle { margin: 0 auto; }
.hint { margin: 4px 0 0; font-size: 12px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; color: #fff; }
.badge.fly { background: var(--fly); }
.badge.coach { background: var(--coach); }
.cands { margin: 6px 0 0; padding: 0; list-style: none; display: flex; flex-wrap: wrap; gap: 6px; }
.cands li { border: 1px solid var(--line); border-radius: 6px; padding: 1px 8px; }
.cands li.chosen { border-color: var(--fly); font-weight: 600; }
.decision + .decision { margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--line); }
.detail { margin-top: 8px; }
.row { display: flex; gap: 8px; }
.row .k { color: var(--muted); min-width: 110px; }
.sub { margin: 6px 0 0 8px; }
.subtitle { color: var(--muted); font-size: 12px; }
.bars { margin: 6px 0; }
.bars .title { color: var(--muted); font-size: 12px; }
.bars .line { display: grid; grid-template-columns: 110px 1fr 60px; gap: 6px; align-items: center; font-size: 12px; }
.bars .track { position: relative; height: 10px; background: var(--bg); border-radius: 3px; }
.bars .fill { position: absolute; top: 0; height: 10px; border-radius: 3px; background: var(--bar); }
.bars .fill.neg { background: var(--neg); }
.flags { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 4px; font-size: 12px; }
.groups { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
.group { border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
.group .title { font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.series { display: grid; grid-template-columns: 96px 1fr 52px; gap: 6px; align-items: center; font-size: 11px; }
.series svg { width: 100%; height: 28px; }
.series polyline { fill: none; stroke: var(--bar); stroke-width: 1.5; }
.series rect { fill: var(--bar); }
.texts { font-size: 11px; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 12px;
        padding: 2px 10px; font-size: 12px; cursor: pointer; }
.chip.fly { border-color: var(--fly); }
.chip.selected { outline: 2px solid var(--bar); }
@media (max-width: 1100px) {
  main { grid-template-columns: minmax(0, 1fr); }
}
```

`web/live/app.js`:

```javascript
// FlyMon Live: renders viewer events (flymon/live/events.py, version 1) for one fly at a time.
// Unknown event types and fields are ignored, so the brain can add data without touching this file.
"use strict";

const params = new URLSearchParams(location.search);
const COACH_KIND = { attack: "공격", support: "보조기", switch: "교체", default: "기본" };
const EFFECT = { super: "효과 굉장함", neutral: "보통", resisted: "효과 별로", immune: "효과 없음", unknown: "상성 알 수 없음" };

const view = { flies: new Map(), fly: params.get("fly"), follow: true, turn: null, ignored: 0 };
// NOT `$`: the Showdown renderer on this page uses jQuery's global `$`, and a global const would shadow it.
const el = (id) => document.getElementById(id);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmt(v) {
  return typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v);
}

// ---- state --------------------------------------------------------------------------------------
function battleOf(ev) {
  let fly = view.flies.get(ev.fly);
  if (!fly) {
    fly = { battles: new Map(), last: null };
    view.flies.set(ev.fly, fly);
    if (!view.fly) view.fly = ev.fly;
  }
  let b = fly.battles.get(ev.battle_tag);
  if (!b) {
    b = { tag: ev.battle_tag, turns: new Map(), maxTurn: 0, end: null, lines: [], lastLineTurn: 0 };
    fly.battles.set(ev.battle_tag, b);
    fly.last = ev.battle_tag;
  }
  return b;
}

function turnOf(b, turn) {
  let t = b.turns.get(turn);
  if (!t) {
    t = { decisions: [], outcomes: [], traces: [] };
    b.turns.set(turn, t);
  }
  b.maxTurn = Math.max(b.maxTurn, turn);
  return t;
}

function ingest(ev) {
  if (!ev || ev.v !== 1 || typeof ev.fly !== "string" || typeof ev.battle_tag !== "string") {
    view.ignored += 1;
    return;
  }
  const kinds = { decision: "decisions", outcome: "outcomes", trace: "traces" };
  if (ev.type === "battle_start") {
    battleOf(ev);
    if (ev.fly === view.fly) view.follow = true;
  } else if (ev.type === "battle_end") {
    battleOf(ev).end = ev;
  } else if (ev.type === "protocol") {
    const b = battleOf(ev);
    b.lines.push(...ev.lines);
    for (const line of ev.lines) {
      const m = /^\|turn\|(\d+)/.exec(line);
      if (m) b.lastLineTurn = Number(m[1]);
    }
  } else if (kinds[ev.type]) {
    turnOf(battleOf(ev), ev.turn)[kinds[ev.type]].push(ev);
  } else {
    view.ignored += 1;
  }
}

function currentBattle() {
  const fly = view.flies.get(view.fly);
  return fly && fly.last ? fly.battles.get(fly.last) : null;
}

// ---- the official battle renderer (replay-embed.js) -------------------------------------------
const renderer = {
  tag: null,
  fed: 0,
  ok: null,
  available() {
    const b = window.Replays && window.Replays.battle;
    return !!(b && typeof b.add === "function" && typeof b.instantAdd === "function" && typeof Replays.init === "function");
  },
  show(battle) {
    if (!this.available()) {
      this.ok = false;
      return false;
    }
    this.ok = true;
    if (this.tag !== battle.tag) {
      try { Replays.battle.destroy(); } catch (e) { /* older renderer: the new init replaces it */ }
      document.querySelector("script.battle-log-data").textContent = "";
      Replays.init();
      this.tag = battle.tag;
      this.fed = 0;
      for (const line of battle.lines) Replays.battle.instantAdd(line);
      this.fed = battle.lines.length;
      Replays.battle.play();
      this.catchUp(battle);                                                 // start at the live turn
      return true;
    }
    for (let i = this.fed; i < battle.lines.length; i++) Replays.battle.add(battle.lines[i]);
    this.fed = battle.lines.length;
    if (Replays.battle.paused) Replays.battle.play();
    this.catchUp(battle);
    return true;
  },
  catchUp(battle) {
    // animation is slower than play: if it falls behind, jump to the newest turn instead of drifting
    if (battle.lastLineTurn && Replays.battle.turn < battle.lastLineTurn - 1) {
      try { Replays.battle.seekTurn(battle.lastLineTurn); } catch (e) { /* older renderer: let it animate */ }
    }
  }
};

// ---- panels -------------------------------------------------------------------------------------
function barsHTML(title, values, labels) {
  const lo = Math.min(0, ...values), hi = Math.max(0, ...values), span = hi - lo || 1;
  const zero = ((0 - lo) / span) * 100;
  const lines = values.map((v, i) => {
    const w = (Math.abs(v) / span) * 100, left = v >= 0 ? zero : zero - w;
    return `<div class="line"><span>${esc(labels[i])}</span><span class="track">` +
      `<span class="fill${v < 0 ? " neg" : ""}" style="left:${left}%;width:${w}%"></span></span><span>${fmt(v)}</span></div>`;
  });
  return `<div class="bars"><div class="title">${esc(title)}</div>${lines.join("")}</div>`;
}

function detailHTML(detail, cands, depth = 0) {
  return Object.entries(detail).map(([k, v]) => {
    if (Array.isArray(v) && v.length > 0 && v.length === cands.length && v.every((x) => typeof x === "number")) {
      return barsHTML(k, v, cands);
    }
    if (v !== null && typeof v === "object" && !Array.isArray(v) && depth < 2) {
      return `<div class="sub"><div class="subtitle">${esc(k)}</div>${detailHTML(v, cands, depth + 1)}</div>`;
    }
    const text = v === null || ["number", "string", "boolean"].includes(typeof v) ? fmt(v) : JSON.stringify(v);
    return `<div class="row"><span class="k">${esc(k)}</span><span>${esc(text)}</span></div>`;
  }).join("");
}

function decisionHTML(d) {
  const who = d.decider === "fly" ? '<span class="badge fly">초파리</span>' : '<span class="badge coach">코치</span>';
  const cands = d.candidates.map((c) => `<li class="${c === d.chosen ? "chosen" : ""}">${esc(c)}${c === d.chosen ? " ✔" : ""}</li>`);
  const chosen = d.candidates.includes(d.chosen) || d.chosen === null ? "" : `<div class="muted">선택: ${esc(d.chosen)}</div>`;
  const detail = d.detail && typeof d.detail === "object" ? `<div class="detail">${detailHTML(d.detail, d.candidates)}</div>` : "";
  return `<div class="decision">${who} 코치 주문: ${esc(COACH_KIND[d.coach_kind] || d.coach_kind)}` +
    (cands.length ? `<ul class="cands">${cands.join("")}</ul>` : "") + chosen + detail + "</div>";
}

function outcomeHTML(ev) {
  const o = ev.outcome || {};
  const flag = (label, on) => `<span>${label} ${on ? "✔" : "✗"}</span>`;
  return `<div><strong>${esc(o.move_id || "?")}</strong> → 직접 피해 ${Math.round((o.dealt_frac || 0) * 100)}%` +
    ` · ${esc(EFFECT[o.effectiveness] || o.effectiveness)}</div>` +
    `<div class="flags">${flag("기절", o.target_fainted_by_me)}${flag("빗나감", o.missed)}` +
    `${flag("행동 불가", o.no_action)}${flag("불확실", o.uncertain)}</div>` +
    (o.notes && o.notes.length ? `<div class="muted">${esc(o.notes.join(" · "))}</div>` : "");
}

function sparkHTML(points, t0) {
  const xs = points.map((p) => p[0] - t0), ys = points.map((p) => Number(p[1]));
  const xmax = Math.max(...xs) || 1, ymin = Math.min(...ys), yspan = Math.max(...ys) - ymin || 1;
  const pts = xs.map((x, i) => `${((x / xmax) * 100).toFixed(1)},${(26 - ((ys[i] - ymin) / yspan) * 24).toFixed(1)}`);
  return `<svg viewBox="0 0 100 28" preserveAspectRatio="none"><polyline points="${pts.join(" ")}"/></svg>`;
}

function barSvgHTML(values) {
  const vmax = Math.max(...values, 0) || 1, w = 100 / values.length;
  const rects = values.map((v, i) => {
    const h = (Math.max(v, 0) / vmax) * 26;
    return `<rect x="${(i * w + w * 0.1).toFixed(1)}" y="${(27 - h).toFixed(1)}" width="${(w * 0.8).toFixed(1)}" height="${h.toFixed(1)}"/>`;
  });
  return `<svg viewBox="0 0 100 28" preserveAspectRatio="none">${rects.join("")}</svg>`;
}

function traceHTML(tr, cands) {
  const all = tr.series.flatMap((s) => s.points.map((p) => p[0]));
  const t0 = all.length ? Math.min(...all) : 0;                  // the tap clock runs on: rebase to 0
  const slot = tr.slot === null || tr.slot === undefined ? "" : ` · 후보 ${tr.slot}${cands[tr.slot] ? " " + cands[tr.slot] : ""}`;
  const body = tr.series.map((s) => {
    if (!s.points.length) return "";
    const last = s.points[s.points.length - 1];
    if (s.kind === "scalar") {
      return `<div class="series"><span>${esc(s.path)}</span>${sparkHTML(s.points, t0)}<span>${fmt(Number(last[1]))}</span></div>`;
    }
    if (s.kind === "bars") {
      return `<div class="series"><span>${esc(s.path)}</span>${barSvgHTML(last[1].map(Number))}<span>${last[1].length}개</span></div>`;
    }
    if (s.kind === "text") {
      return `<div class="texts">${s.points.map((p) => `${Math.round(p[0] - t0)} ms: ${esc(p[1])}`).join("<br>")}</div>`;
    }
    return "";
  }).join("");
  return `<div class="group"><div class="title">${esc(tr.phase)}${esc(slot)}</div>${body}</div>`;
}

function chipHTML(turn, t, selected) {
  const fly = t.decisions.some((d) => d.decider === "fly");
  const o = t.outcomes.length ? t.outcomes[t.outcomes.length - 1].outcome || {} : null;
  let mark = "";
  if (o) {
    if (o.missed) mark = " ✗빗나감";
    else if (o.no_action) mark = " 행동불가";
    else if (o.uncertain) mark = " ?";
    else if (o.target_fainted_by_me) mark = " 기절";
    else if (o.dealt_frac > 0) mark = ` ✔${Math.round(o.dealt_frac * 100)}%`;
  }
  return `<button type="button" class="chip${fly ? " fly" : ""}${selected ? " selected" : ""}" data-turn="${turn}">` +
    `${turn} ${fly ? "초파리" : "코치"}${mark}</button>`;
}

// ---- render -------------------------------------------------------------------------------------
function render() {
  const sel = el("fly");
  const names = [...view.flies.keys()].sort();
  if (sel.options.length !== names.length || [...sel.options].some((o, i) => o.value !== names[i])) {
    sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  }
  sel.value = view.fly || "";
  el("ignored").hidden = view.ignored === 0;
  el("ignored").textContent = `무시한 이벤트 ${view.ignored}`;

  const b = currentBattle();
  el("battle").textContent = b ? b.tag + (b.end ? ` (끝남: ${b.end.won === true ? "승" : b.end.won === false ? "패" : "무"})` : "") : "배틀 없음";
  const note = el("battle-note");
  if (!b) {
    note.textContent = "배틀 없음";
  } else if (renderer.show(b)) {
    note.textContent = `${b.tag} · 프로토콜 ${b.lines.length}줄`;
  } else {
    note.textContent = "배틀 화면을 불러오지 못했습니다 (인터넷 연결 또는 Showdown 렌더러 확인). 결정·결과 패널은 그대로 동작합니다.";
  }

  const turns = b ? [...b.turns.keys()].sort((x, y) => x - y) : [];
  const turn = view.follow ? (turns.length ? turns[turns.length - 1] : null) : view.turn;
  const t = b && turn !== null ? b.turns.get(turn) : null;
  el("turn").textContent = turn === null ? "" : `턴 ${turn}`;
  el("follow").hidden = view.follow;

  el("decision").innerHTML = t && t.decisions.length ? t.decisions.map(decisionHTML).join("") : "아직 결정 없음";
  el("outcome").innerHTML = t && t.outcomes.length ? t.outcomes.map(outcomeHTML).join("") : "결과 대기 중";
  const cands = t && t.decisions.length ? t.decisions[t.decisions.length - 1].candidates : [];
  el("brain").innerHTML = t && t.traces.length ?
    `<div class="groups">${t.traces.map((tr) => traceHTML(tr, cands)).join("")}</div>` : "뇌 미연결";
  el("timeline").innerHTML = turns.map((n) => chipHTML(n, b.turns.get(n), n === turn)).join("");
}

// ---- wiring -------------------------------------------------------------------------------------
el("fly").addEventListener("change", (e) => { view.fly = e.target.value; view.follow = true; render(); });
el("follow").addEventListener("click", () => { view.follow = true; render(); });
el("timeline").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  view.follow = false;
  view.turn = Number(chip.dataset.turn);
  render();
});

const stream = new EventSource("/stream");
stream.addEventListener("state", (e) => {
  view.flies.clear();
  view.ignored = 0;
  for (const ev of JSON.parse(e.data).events) ingest(ev);
  render();
});
stream.onmessage = (e) => { ingest(JSON.parse(e.data)); render(); };
stream.onopen = () => { el("conn").className = "conn on"; el("conn").textContent = "연결됨"; };
stream.onerror = () => { el("conn").className = "conn off"; el("conn").textContent = "재연결 중"; };
render();
```

- [ ] **Step 2: 문법 확인**

Run: `node --check web/live/app.js`
Expected: 출력 없음(성공). 페이지 동작은 Task 7의 완료 기준으로 확인한다.

- [ ] **Step 3: 커밋**

```bash
git add web/live/index.html web/live/app.js web/live/style.css
git commit -m "feat(web): live page renders the battle in Showdown's own renderer beside the panels"
```

---

### Task 7: 완료 기준 확인 (컨트롤러가 직접 실행)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-17-flymon-live-viewer-design.md`(3절에 확인 결과 한 줄)

**Interfaces:**
- Consumes: Task 1–6 전부
- Produces: 없음

- [ ] **Step 1: 전체 테스트**

Run: `uv sync && bash scripts/install_showdown.sh && uv run pytest -q`
Expected: 기존 테스트 + 새 테스트 전부 통과(새 테스트 35개: live 27 + 플레이어 7 + 배리어 1)

- [ ] **Step 2: 완료 기준 1 (가짜 뇌)**

```bash
uv run python scripts/live_viewer.py --port 8765 &
uv run python scripts/live_battles.py --flies 2 --battles 2 --provider fake-brain \
    --watch-fly 0 --turn-delay 2 --viewer http://127.0.0.1:8765
```
출력된 `open:` 주소를 브라우저로 연다. 확인: 페이지 안에 배틀이 그려지고 렌더러 턴이 헤더 턴보다 2턴 이상 뒤처지지 않음,
① 후보별 막대(V·p·kc_active_frac), ③ 후보별 차트, ④ 턴 칩이 늘어남, 드롭다운으로 `fm-live-f01`을 고르면 그 마리 배틀로 바뀜.
스크린샷을 `results/live-viewer/`(git 제외)에 남긴다.

- [ ] **Step 3: 완료 기준 2 (뇌 없음)**

```bash
uv run python scripts/live_battles.py --flies 1 --battles 1 --provider rnd --turn-delay 0.3 --viewer http://127.0.0.1:8765
```
확인: ③이 "뇌 미연결", ①에 막대 없음.

- [ ] **Step 4: 완료 기준 3 (뷰어가 죽어도 배틀은 산다)**

배틀이 도는 중에 뷰어 서버를 `kill -TERM` 한다(백그라운드 프로세스는 SIGINT를 무시한다).
확인: 스크립트가 `exit 0`, `dropped_events=<0보다 큰 수>` 출력, 배틀은 끝까지 진행.

- [ ] **Step 5: 완료 기준 4 (렌더러 없음)**

페이지 콘솔에서 `Replays.battle = null` 로 만든 뒤 이벤트가 오면, 배틀 화면 자리에 안내 문구가 뜨고 ①②③④는 계속 동작.

- [ ] **Step 6: 설계 문서에 결과 기록하고 커밋**

```bash
git add docs/superpowers/specs/2026-09-17-flymon-live-viewer-design.md
git commit -m "docs: live viewer acceptance results"
```
