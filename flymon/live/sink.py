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
