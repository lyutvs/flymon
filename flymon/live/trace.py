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
