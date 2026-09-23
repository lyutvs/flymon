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
