"""Live view of a running simulation: watch training while it runs instead of after the JSON lands.

SpikeTap         takes engine.on_step, calls the hook it replaced first (Plasticity keeps its rule
                 there), then only counts spikes; every `every` steps it sends population mean rates
                 and per-cell rates to a sink. It never writes engine state, so an observed run is
                 bit-identical. Install it after Plasticity, which assigns engine.on_step directly.
ConditioningViz  on_event handler for conditioning.run_arm: compartment weights, dopamine pulses,
                 probe counts and D, stamped with the tap's clock.
RerunSink        writes to a rerun.RecordingStream on the "sim" duration timeline. rerun is the
                 optional `viz` extra and is imported only there.

Sink interface: scalar(path, t_ms, value), bars(path, t_ms, values), text(path, t_ms, msg).
The tap's clock keeps running across engine.reset(), so each presentation gets its own stretch of
the timeline.
"""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .plasticity import Plasticity


class SpikeTap:
    def __init__(self, engine: Engine, groups: dict, sink, every: int = 100, per_cell: dict | None = None):
        self.sink, self.every, self.dt = sink, every, engine.p.dt
        self.masks = {}
        for name, idx in groups.items():
            idx = np.asarray(idx, np.int64)
            if idx.size == 0:
                raise ValueError(f"group {name!r} is empty")
            mask = np.zeros(engine.N, bool); mask[idx] = True
            self.masks[name] = (mask, idx.size)
        self.rows = {}
        for name, idx in (per_cell or {}).items():
            idx = np.asarray(idx, np.int64)
            if idx.size == 0:
                raise ValueError(f"per-cell group {name!r} is empty")
            row = np.full(engine.N, -1, np.int64); row[idx] = np.arange(idx.size)
            self.rows[name] = row
        self.counts = dict.fromkeys(self.masks, 0)
        self.cell_counts = {name: np.zeros(len(per_cell[name]), np.int64) for name in self.rows}
        self.steps = 0
        self.sim_ms = 0.0
        self.inner = engine.on_step
        engine.on_step = self.on_step

    def on_step(self, engine: Engine, fired: np.ndarray) -> None:
        if self.inner is not None:
            self.inner(engine, fired)
        for name, (mask, _) in self.masks.items():
            self.counts[name] += int(np.count_nonzero(mask[fired]))
        for name, row in self.rows.items():
            r = row[fired]
            r = r[r >= 0]
            if r.size:
                self.cell_counts[name] += np.bincount(r, minlength=self.cell_counts[name].size)
        self.steps += 1
        self.sim_ms = self.steps * self.dt
        if self.steps % self.every == 0:
            self._emit()

    def _emit(self) -> None:
        window_s = self.every * self.dt / 1000.0
        for name, (_, n) in self.masks.items():
            self.sink.scalar(f"rate_hz/{name}", self.sim_ms, self.counts[name] / (n * window_s))
            self.counts[name] = 0
        for name, c in self.cell_counts.items():
            self.sink.bars(f"cell_hz/{name}", self.sim_ms, c / window_s)
            c[:] = 0


class ConditioningViz:
    def __init__(self, sink, clock: SpikeTap, pl: Plasticity, channels):
        self.sink, self.clock, self.pl = sink, clock, pl
        # Plasticity keeps each type's DAN->MBON weight vector restricted to its core compartment
        self.cores = {name: pl.pops.mbon[pl.types[name][1] > 0] for name in channels}

    def __call__(self, kind: str, **f) -> None:
        t, s = self.clock.sim_ms, self.sink
        if kind == "arm_start":
            s.text("events", t, f"arm {f['arm']} seed {f['seed']} start")
        elif kind == "probe":
            for k in ("A", "P"):
                s.scalar(f"probe/{f['phase']}/{f['cs']}/{k}", t, f[k])
        elif kind == "presentation":
            for name, core in self.cores.items():
                s.scalar(f"weights_frac/{name}", t, self.pl.weights_frac_by_mbon_set(core))
                s.scalar(f"dan_pulse/{name}", t, 1.0 if f["dan"] == name else 0.0)
        elif kind == "arm_end":
            for k in ("D_pre", "D_post", "dD"):
                s.scalar(f"{k}/{f['arm']}", t, f[k])
            s.text("events", t, f"arm {f['arm']} seed {f['seed']} dD {f['dD']:+.3f}")
        else:
            raise ValueError(f"unknown conditioning event {kind!r}")


def attach_conditioning_viz(engine: Engine, pl: Plasticity, pops: Populations, sink, channels,
                            every: int = 100) -> ConditioningViz:
    """Tap the olfactory -> mushroom body path and return the on_event handler for run_arm.
    Call after Plasticity(...), which assigns engine.on_step."""
    groups = {"alpn": pops.alpn, "kc": pops.kc, "mbon": pops.mbon, "apl": pops.apl,
              **{f"dan/{name}": pl.types[name][0] for name in channels}}
    tap = SpikeTap(engine, groups, sink, every=every, per_cell={"mbon": pops.mbon})
    return ConditioningViz(sink, tap, pl, channels)


class RerunSink:
    def __init__(self, rec, prefix: str = ""):
        import rerun as rr   # optional `viz` extra
        self.rr, self.rec, self.prefix = rr, rec, prefix

    def _at(self, t_ms: float) -> None:
        self.rec.set_time("sim", duration=t_ms / 1000.0)

    def scalar(self, path: str, t_ms: float, value) -> None:
        self._at(t_ms)
        self.rec.log(self.prefix + path, self.rr.Scalars(float(value)))

    def bars(self, path: str, t_ms: float, values) -> None:
        self._at(t_ms)
        self.rec.log(self.prefix + path, self.rr.BarChart(np.asarray(values, np.float32)))

    def text(self, path: str, t_ms: float, msg: str) -> None:
        self._at(t_ms)
        self.rec.log(self.prefix + path, self.rr.TextLog(msg))
