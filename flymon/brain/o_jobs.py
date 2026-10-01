"""FlyPool worker jobs of spec appendix O (O.7.1, O.7.3). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle them; the worker's default engine only lends its connectome.

O's rig is cached per worker under (Params, edit), edit in EDITS = n_jobs.EDITS + ("apl_to_nonkc_zero",):
- N's three edits are made by n_jobs.apply_edit, unchanged;
- "apl_to_nonkc_zero" (O.7.1): every APL out-edge whose target is not a KC set to 0 (APL -> MBON05 included). With
  "apl_to_kc_zero" it partitions "apl_all_zero"'s edge set (test).
The edit is made in place on a freshly built engine's CSC (the graded-APL out-edge views follow). The cache is O's own:
n_jobs._RIG is never touched, and no N module changes (O.5).

presentation_job: O1's presentations, N0f's procedure (n_jobs._present: reset, clear_drive, present, settle, read,
plasticity off), scalars only.

arm_job: one (X, arm, seed) of O2 (O.7.3) through train_x, n_jobs.train_plus_only's loop with a punish flag; the
dopamine-zero arm (dopamine_zero) and the phasic-dopamine meter (DaMeter); every arm's plastic weights hashed before
and after training (O.7.4-5, O.7.8: arms 2 and 4 must not move them)."""
from __future__ import annotations

import contextlib
import hashlib
import time

import numpy as np

from . import n_jobs
from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .n_spec import train_seed
from .plasticity import Plasticity
from .stimuli import present

NONKC = "apl_to_nonkc_zero"
EDITS = n_jobs.EDITS + (NONKC,)
_RIG: dict = {}          # (Params, edit) -> (Engine, Plasticity, comps, csc sha256); one entry per worker


def apply_edit(eng, pops, edit: str) -> str:
    """N's edits through n_jobs.apply_edit; NONKC zeroes every APL -> non-KC edge in place. The sha256 of csc.w.
    Adapted from n_jobs:apply_edit (copied pattern, O.5 forbids editing N)."""
    if edit != NONKC:
        return n_jobs.apply_edit(eng, pops, edit)
    is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
    is_kc = np.zeros(eng.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
    m = is_apl[edge_sources(eng.csc)] & ~is_kc[eng.csc.tgt.astype(np.int64)]
    eng.csc.w[m] = np.float32(0.0)
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


def rig(conn, pops, params, edit: str):
    """The worker's rig for (params, edit), built once and reused while jobs keep asking for it (one entry: a switch
    rebuilds, it never edits a cached engine).
    Copied from n_jobs:rig (O's own cache; O.5 forbids editing N)."""
    key = (params, edit)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha = apply_edit(eng, pops, edit)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha)
    return _RIG[key]


def presentation_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                     settle_ms: float, read_ms: float, window_ms: int) -> list:
    """O1's presentations on O's rig: n_jobs._present per seed, plasticity off, weights at w0; scalars only.
    Copied from n_jobs:presentation_job (O.5 forbids editing N)."""
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = n_jobs.readout_cells(e.conn, readout)
    p.reset_weights()
    was = p.enabled
    p.set_enabled(False)
    try:
        return [dict(n_jobs._present(e, p, pops, odor, s, strength, settle_ms, read_ms, window_ms, a_cells,
                                     p_cells)[0], edit=edit, csc_sha256=sha) for s in seeds]
    finally:
        p.set_enabled(was)


# ================================================================ O2: the training loop and the arms (O.7.3)
class DaMeter:
    """Installed as the engine's step hook while a training runs: the plasticity rule runs unchanged
    (Plasticity.on_step), then, while `on`, the phasic dopamine max(da - da_base, 0) is integrated per MBON (x dt).
    Reading the traces changes nothing; leaving the context restores e.on_step = p.on_step."""

    def __init__(self, e, p):
        self.e, self.p, self.on = e, p, False
        self.acc = np.zeros(p.da.shape, np.float64)

    def __call__(self, eng, fired) -> None:
        self.p.on_step(eng, fired)
        if self.on:
            self.acc += np.maximum(self.p.da - self.p.da_base, 0.0) * eng.p.dt

    def __enter__(self):
        self.e.on_step = self
        return self

    def __exit__(self, *exc) -> None:
        self.e.on_step = self.p.on_step

    def by_compartment(self, comps) -> dict:
        """Reading 11: per DAN type, the mean over its core MBONs of the integral."""
        return {name: float(self.acc[self.p.mb_local[cp.core]].mean()) for name, cp in sorted(comps.items())
                if cp.core.size}


@contextlib.contextmanager
def dopamine_zero(p):
    """Arm 4 (O.7.3, reading 10): every DAN type's MBON weight vector in the rule set to 0, so p.da and p.da_base stay
    exactly 0 every step. The DAN cells, their drive and their spikes are untouched (drive_dan / quiet_dan read only
    the cells). Restored on exit (O.7.8: with the rule gated by phasic dopamine only, no weight can move here)."""
    saved = p.types
    p.types = {k: (cells, np.zeros_like(w)) for k, (cells, w) in saved.items()}
    p.da[:] = 0; p.da_base[:] = 0
    try:
        yield
    finally:
        p.types = saved


def train_x(e, p, pops, cs, strength: float, seed: int, punish, trials: int, present_ms: float, gap_ms: float,
            settle_ms: float, seed_base: int, seed_stride: int, meter=None) -> None:
    """X alone, n_jobs.train_plus_only's loop with a punish flag (reading 9; copied from n_jobs:train_plus_only, O.5
    forbids editing N). Per trial:
    - reset to train_seed(seed, trial);
    - settle with the weights frozen;
    - the punishment DAN for present_ms if `punish` is a DAN type (None: no injection);
    - the gap, then one recovery step only after a pulse (conditioning.train_block's dan-None rule).
    The meter integrates over present + gap."""
    for t in range(int(trials)):
        e.reset(train_seed(seed, t, seed_base, seed_stride))
        p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, cs, strength)
        was = p.enabled
        p.set_enabled(False); e.run(settle_ms); p.set_enabled(was)
        if punish is not None:
            p.drive_dan(punish, e.p.dan_drive_mv)
        if meter is not None:
            meter.on = True
        e.run(present_ms)
        p.quiet_dan(); e.clear_drive(); e.run(gap_ms)
        if meter is not None:
            meter.on = False
        if punish is not None:
            p.recover_pulse()


def weights_sha256(w) -> str:
    """sha256 of the plastic weights as float32 bytes (the bit-identity plumbing check, O.7.4-5)."""
    return hashlib.sha256(np.ascontiguousarray(w, np.float32).tobytes()).hexdigest()


def arm_job(eng, pl, pops, comps, ro, params, edit: str, odor_x: dict, odor_y: dict, seed: int, arm: str,
            punish: bool, plastic: bool, da_zero: bool, readout: dict, punish_type: str, reward_type: str,
            strength: float, settle_ms: float, read_ms: float, window_ms: int, trials: int, present_ms: float,
            gap_ms: float, train_settle_ms: float, seed_base: int, seed_stride: int) -> dict:
    """One (X, arm, seed) of O2 in the edit's rig (O.7.3); adapted from n_jobs:absolute_arm_job (O.5):
    - probes of X and Y at `seed`, plasticity off (n_jobs._present rows, as absolute_arm_job);
    - training of X alone (train_x): punishment iff `punish`, weights frozen unless `plastic`, the rule's dopamine
      held at 0 iff `da_zero`; the phasic dopamine integrated per compartment;
    - the probes again; weights_frac, and weights_frac_by_mbon_set of the A·P cores as conditioning.run_arm reads them
      (A = comps[punish_type].core, P = comps[reward_type].core: the taught compartments, not the readout cells);
    - the sha256 of the plastic weights before (w0) and after training, every arm (O.7.4-5, O.7.8).
    With punish and plastic on this is absolute_arm_job's result (regression test). Weights, the step hook and the
    rule's dopamine weights are restored."""
    e, p, c, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = n_jobs.readout_cells(e.conn, readout)
    a_core, p_core = c[punish_type].core, c[reward_type].core
    t0 = time.perf_counter()

    def probes():
        was = p.enabled
        p.set_enabled(False)
        try:
            return {k: n_jobs._present(e, p, pops, o, seed, strength, settle_ms, read_ms, window_ms, a_cells,
                                       p_cells)[0] for k, o in (("x", odor_x), ("y", odor_y))}
        finally:
            p.set_enabled(was)

    meter = DaMeter(e, p)
    try:
        p.reset_weights()
        w0_sha = weights_sha256(p.w0)
        pre = probes()
        p.set_enabled(bool(plastic))
        with meter, (dopamine_zero(p) if da_zero else contextlib.nullcontext()):
            train_x(e, p, pops, odor_x, strength, int(seed), punish_type if punish else None, trials, present_ms,
                    gap_ms, train_settle_ms, seed_base, seed_stride, meter)
        p.set_enabled(True)
        post = probes()
        w_post = weights_sha256(e.csc.w[p.edges])
        wf = float(p.weights_frac())
        wa, wp = float(p.weights_frac_by_mbon_set(a_core)), float(p.weights_frac_by_mbon_set(p_core))
        da = meter.by_compartment(c)
    finally:
        e.on_step = p.on_step
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), edit=edit, arm=arm, punish=bool(punish), plastic=bool(plastic),
                da_zero=bool(da_zero), csc_sha256=sha, pre=pre, post=post, weights_frac=wf, weights_frac_A=wa,
                weights_frac_P=wp, w0_sha256=w0_sha, w_post_sha256=w_post, da_integral=da,
                wall_s=time.perf_counter() - t0)
