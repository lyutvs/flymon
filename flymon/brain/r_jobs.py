"""FlyPool worker jobs of spec appendix R (R.9.6, R.9.7). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle them; the worker's engine only lends its connectome. Every job
takes its rig from q_jobs.q_rig(conn, pops, params, edit, p_type) — Q's edit-capable rig (built as h4_jobs.rig_for /
o_jobs.rig build theirs, then the edit in place; Q's reproduction gate 02650a3 pinned it), cached per worker under
(Params, edit, P type). q_jobs, k_jobs, o_jobs, n_jobs and h4_jobs are imported, never edited.
- kc_activity_job: k_jobs.activity_job's body (h4_jobs._present_kc per item, plasticity off, weights reset) on Q's
  rig; each row also carries the CSC sha and the edited edge count (gate ①; with edit "none" it is activity_job's
  rows — test — and reproduces encoder ③'s entries, R.9.6).
- r_arm_job: o_jobs.arm_job copied onto Q's rig (O's rig knows no apl_to_mbon05_zero); with edit "none" it is
  arm_job's result (minus wall_s) plus "r" (test; the reproduction gate checks it against P's committed entries).
- r_oracle_job: q_jobs.q_oracle_job called unchanged (R.9.6: Q's verified oracle path), then the P readout type's
  per-cell naive probe on the report seeds — presentation.decide with idx = the type's cells, the weights at w0, the
  oracle's own pre-probe sequence, so the per-cell counts sum to report.pre.P (R.9.7 per-cell saturation)."""
from __future__ import annotations

import contextlib
import time

import numpy as np

from . import o_jobs, q_jobs
from .h4_jobs import _present_kc, type_cells
from .n_jobs import _present, readout_cells
from .presentation import decide


def kc_activity_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, items, strength: float,
                    settle_ms: float, read_ms: float, window_ms: int) -> list:
    e, p, _, sha, n_edit = q_jobs.q_rig(eng.conn, pops, params, edit, p_type)
    out = []
    try:
        p.reset_weights()
        p.set_enabled(False)
        for i, odor, seed in items:
            o = _present_kc(e, p, pops, odor, int(seed), strength, settle_ms, read_ms, int(window_ms))
            nz = np.flatnonzero(o["read"])
            out.append(dict(i=int(i), seed=int(seed), kc=nz.tolist(), n=o["read"][nz].astype(int).tolist(),
                            max_win=int(o["max_win"]), csc_sha256=sha, edit_edges=int(n_edit)))
    finally:
        p.reset_weights()
        p.set_enabled(True)
    return out


def r_arm_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, odor_x: dict, odor_y: dict, seed: int,
              arm: str, punish: bool, plastic: bool, da_zero: bool, readout: dict, punish_type: str, reward_type: str,
              strength: float, settle_ms: float, read_ms: float, window_ms: int, trials: int, present_ms: float,
              gap_ms: float, train_settle_ms: float, seed_base: int, seed_stride: int) -> dict:
    """o_jobs:arm_job (copied; O2's one (X, arm, seed)) on Q's rig. Probes of X and Y at `seed` (plasticity off),
    training of X alone (punishment iff `punish`, weights frozen unless `plastic`, the rule's dopamine at 0 iff
    `da_zero`), the probes again, the weight fractions, the phasic dopamine per compartment, and the plastic-weight
    sha256 before and after. Weights, the step hook and the rule's dopamine weights are restored."""
    e, p, c, sha, n_edit = q_jobs.q_rig(eng.conn, pops, params, edit, p_type)
    a_cells, p_cells = readout_cells(e.conn, readout)
    a_core, p_core = c[punish_type].core, c[reward_type].core
    t0 = time.perf_counter()

    def probes():
        was = p.enabled
        p.set_enabled(False)
        try:
            return {k: _present(e, p, pops, o, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells)[0]
                    for k, o in (("x", odor_x), ("y", odor_y))}
        finally:
            p.set_enabled(was)

    meter = o_jobs.DaMeter(e, p)
    try:
        p.reset_weights()
        w0_sha = o_jobs.weights_sha256(p.w0)
        pre = probes()
        p.set_enabled(bool(plastic))
        with meter, (o_jobs.dopamine_zero(p) if da_zero else contextlib.nullcontext()):
            o_jobs.train_x(e, p, pops, odor_x, strength, int(seed), punish_type if punish else None, trials,
                           present_ms, gap_ms, train_settle_ms, seed_base, seed_stride, meter)
        p.set_enabled(True)
        post = probes()
        w_post = o_jobs.weights_sha256(e.csc.w[p.edges])
        wf = float(p.weights_frac())
        wa, wp = float(p.weights_frac_by_mbon_set(a_core)), float(p.weights_frac_by_mbon_set(p_core))
        da = meter.by_compartment(c)
    finally:
        e.on_step = p.on_step
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), edit=edit, arm=arm, punish=bool(punish), plastic=bool(plastic),
                da_zero=bool(da_zero), csc_sha256=sha, pre=pre, post=post, weights_frac=wf, weights_frac_A=wa,
                weights_frac_P=wp, w0_sha256=w0_sha, w_post_sha256=w_post, da_integral=da,
                wall_s=time.perf_counter() - t0, r=dict(edit_edges=int(n_edit), p_type=p_type))


def r_oracle_job(eng, pl, pops, comps, ro, **kw) -> dict:
    """q_jobs.q_oracle_job(**kw) unchanged, then "r": the P readout type's per-cell read-window counts of X and Y on
    every report seed with the weights at w0 (q_oracle_job leaves them there), plus what the saturation record needs
    (read window, dt, refractory steps of the job's Params)."""
    res = q_jobs.q_oracle_job(eng, pl, pops, comps, ro, **kw)
    p_type = kw["readout"]["P"]
    e, p, _, _, _ = q_jobs.q_rig(eng.conn, pops, kw["params"], kw["edit"], p_type)
    cells = type_cells(e.conn, [p_type])[p_type]
    xs, ys = [], []
    try:
        p.reset_weights()
        for s in kw["report_seeds"]:
            cnt = decide(e, p, pops, [kw["odor_x"], kw["odor_y"]], kw["strength"], int(s), kw["settle_ms"],
                         kw["read_ms"], idx=cells)
            xs.append(np.asarray(cnt[0]).astype(int).tolist())
            ys.append(np.asarray(cnt[1]).astype(int).tolist())
    finally:
        p.reset_weights()
        p.set_enabled(True)
    res["r"] = dict(p_type=p_type, n_cells=int(cells.size), p_cells={"x": xs, "y": ys}, read_ms=float(kw["read_ms"]),
                    dt=float(e.p.dt), refrac_steps=int(e.p.refrac_steps()))
    return res
