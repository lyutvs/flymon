"""W's only measurement file (W.0, W.1, W.8, W.9.6 P2-10 / P2-12, W.9.8 H9): the sequential F.2 learning job on U's rig
(u_measure.u_rig, the combined lever L_V or "none"), and the cache plumbing around it. The shared measurement files
(r_measure.R_MEASURE_FILES), T's (t_measure.T_MEASURE_FILES) and U's (u_measure.U_MEASURE_FILES) are imported, never
edited; this file joins them in the W measurement key (= U's measurement key 8a4e0930… + this file), which keys W's
cache and every W block.
- w_learn_job: one (pair, fly, brain) unit. The rig's weights are reset; the probes (X then Y per probe seed,
  n_jobs._present — the presentation sequence of P's arm job and of the oracle's decide: reset, traces, drive, DAN
  quiet, present, settle + read; plasticity off) run at "pre", then after every phase. A phase is o_jobs.train_x
  unchanged — X alone, `trials` trials, train seed = seed_base + fly·seed_stride + t, settle with the weights frozen,
  the phase's DAN (None: no injection) for present_ms, the gap, recovery after a pulse — with plasticity on iff
  `plastic`, the phasic dopamine integrated per compartment (o_jobs.DaMeter). After each phase: the plastic weights'
  sha256 and the weight fractions (all, the punishment / reward cores). The rig is restored on exit.
  F.2's R = [PAM08, PPL105], N = [None, None], G.5's RN = [PAM08, None] (w_spec.phases); each brain is its own job
  from the naive rig (RN never shares R's state, W.9.6 P2-10). With phases [[PPL105, 12, base]], probe seed = seed and
  P's windows / timings, it is u_arm_job's punish arm (W.3 2 (i)); with phases [] it is the naive probe (W.3 2 (ii)).
- WMeasurer: one cache entry ("w_learn") per unit, keyed by the W measurement key + every job input + pair, brain and
  block (H9: pair · fly · brain · stage list · K · probe seeds · training arguments · engine); a round of n_workers
  units is written as soon as it returns, so a stage resumes with the missing units only. `run` skips the cache (the
  path gate measures afresh)."""
from __future__ import annotations

import sys
import time

from . import o_jobs
from .h3_store import code_key
from .n_jobs import _present, readout_cells
from .r_measure import R_MEASURE_FILES
from .t_measure import T_MEASURE_FILES
from .u_measure import U_MEASURE_FILES, u_rig

W_MEASURE_FILES = ("flymon/brain/w_measure.py",)
KIND = "w_learn"


def w_measure_key(npz: str) -> dict:
    """W.1: the U measurement key's files plus W's measurement file."""
    return code_key(npz, files=tuple(dict.fromkeys(R_MEASURE_FILES + T_MEASURE_FILES + U_MEASURE_FILES
                                                   + W_MEASURE_FILES)))


def w_learn_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, odor_x: dict, odor_y: dict, readout: dict,
                reward_type: str, punish_type: str, phases, probe_seeds, strength: float, settle_ms: float,
                read_ms: float, window_ms: int, present_ms: float, gap_ms: float, train_settle_ms: float, fly: int,
                seed_stride: int, plastic: bool) -> dict:
    e, p, c, sha, n_edit, blocks = u_rig(eng.conn, pops, params, edit, p_type)
    a_cells, p_cells = readout_cells(e.conn, readout)
    a_core, p_core = c[punish_type].core, c[reward_type].core
    t0 = time.perf_counter()
    walls = dict(probe_s=0.0, train_s=0.0)

    def probes() -> dict:
        t = time.perf_counter()
        was = p.enabled
        p.set_enabled(False)
        try:
            out = {"x": [], "y": []}
            for s in probe_seeds:
                for k, o in (("x", odor_x), ("y", odor_y)):
                    out[k].append(_present(e, p, pops, o, int(s), strength, settle_ms, read_ms, window_ms, a_cells,
                                           p_cells)[0])
            return out
        finally:
            p.set_enabled(was)
            walls["probe_s"] += time.perf_counter() - t

    def snap(name: str, pr: dict, da) -> dict:
        return dict(stage=name, x=pr["x"], y=pr["y"], w_sha256=o_jobs.weights_sha256(e.csc.w[p.edges]),
                    weights_frac=float(p.weights_frac()), weights_frac_A=float(p.weights_frac_by_mbon_set(a_core)),
                    weights_frac_P=float(p.weights_frac_by_mbon_set(p_core)), da_integral=da)

    meter = o_jobs.DaMeter(e, p)
    try:
        p.reset_weights()
        w0 = o_jobs.weights_sha256(p.w0)
        stages = [snap("pre", probes(), None)]
        for i, (dan, trials, base) in enumerate(phases):
            t = time.perf_counter()
            meter.acc[:] = 0
            p.set_enabled(bool(plastic))
            with meter:
                o_jobs.train_x(e, p, pops, odor_x, strength, int(fly), dan, int(trials), present_ms, gap_ms,
                               train_settle_ms, int(base), int(seed_stride), meter)
            p.set_enabled(True)
            walls["train_s"] += time.perf_counter() - t
            stages.append(snap(f"S{i + 1}", probes(), meter.by_compartment(c)))
    finally:
        e.on_step = p.on_step
        p.reset_weights()
        p.set_enabled(True)
    return dict(edit=edit, csc_sha256=sha, edit_edges=int(n_edit), block_edges=dict(blocks), w0_sha256=w0,
                fly=int(fly), plastic=bool(plastic), probe_seeds=[int(s) for s in probe_seeds], stages=stages,
                wall_s=time.perf_counter() - t0, **walls)


class WMeasurer:
    """w_learn_job over a pool (FlyPool.run_jobs) behind a cache (w_store.WCache)."""

    def __init__(self, pool, cache, params, readout: dict, p_type: str, reward_type: str, punish_type: str,
                 windows: dict, timing: dict):
        """windows = {strength, settle_ms, read_ms, window_ms}; timing = {present_ms, gap_ms, train_settle_ms,
        seed_stride}."""
        self.pool, self.cache, self.params = pool, cache, params
        self.common = dict(params=params, p_type=p_type, readout=dict(readout), reward_type=reward_type,
                           punish_type=punish_type, **{k: windows[k] for k in ("strength", "settle_ms", "read_ms")},
                           window_ms=int(windows["window_ms"]), **timing)
        self.last_wall_s, self.last_jobs = 0.0, 0

    def kwargs(self, u: dict, override: dict | None = None) -> dict:
        return dict(self.common, **(override or {}), edit=u["edit"], odor_x=dict(u["odor_x"]), odor_y=dict(u["odor_y"]),
                    phases=[list(ph) for ph in u["phases"]], probe_seeds=[int(s) for s in u["probe_seeds"]],
                    fly=int(u["fly"]), plastic=bool(u["plastic"]))

    def inputs(self, u: dict, block: str) -> dict:
        return dict(self.kwargs(u), pair=u["pair"], brain=u["brain"], block=block)

    def run(self, units: list, **override) -> list:
        """No cache (the path gate): rows in order; `override` replaces common arguments (P's windows and timings for
        W.3 2 (i))."""
        n_w = max(1, int(self.pool.n_workers))
        out, t0 = [], time.perf_counter()
        for a in range(0, len(units), n_w):
            out += self.pool.run_jobs(w_learn_job, [self.kwargs(u, override) for u in units[a:a + n_w]])
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(units)
        return out

    def learn(self, units: list, block: str, check=None) -> list:
        """Every unit through the cache; check(done, todo) after each round may raise to stop the stage (the budget
        ledger). Rows in order, each with its cache key and file."""
        ins = [self.inputs(u, block) for u in units]
        todo = [i for i, x in enumerate(ins) if self.cache.get(KIND, x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        t0 = time.perf_counter()
        if todo:
            print(f"w learn {block}: {len(todo)}/{len(units)} units to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            if check is not None:
                check(a, len(todo))
            batch = todo[a:a + n_w]
            for i, r in zip(batch, self.pool.run_jobs(w_learn_job, [self.kwargs(units[i]) for i in batch])):
                self.cache.put(KIND, ins[i], r, [self.params])
            print(f"w learn {block}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for u, x in zip(units, ins):
            got = self.cache.get(KIND, x)
            if got is None:
                raise RuntimeError(f"w learn {block}: {u['pair']} fly {u['fly']} {u['brain']} missing after the run")
            out.append(dict(unit=u, result=got, cache_key=self.cache.key(KIND, x),
                            cache_file=str(self.cache._path(KIND, x))))
        return out
