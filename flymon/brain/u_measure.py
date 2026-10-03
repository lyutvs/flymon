"""U's only measurement file (U.1, U.8, U.9.1, U.9.3, U.9.4 P2-5): the partial lever L_f (the 2 APL->MBON05 CSC edges
of q_jobs.apply_q_edit's mask, weight × f), the mechanism contrast edits, and U's copies of every job U runs with them.
The shared measurement files (r_measure.R_MEASURE_FILES) and T's (t_measure.T_MEASURE_FILES) are not touched; this file
joins them in the U measurement key, which keys U's cache and every U block.
- Edits (strings, so f is in every job kwarg and cache key): u_edit(f) = "u_apl_mbon05_x<f:.1f>" for f in F_ALLOWED
  (the 9-point scan grid and the endpoints 0.0 / 1.0), optionally + "+out_block" (MBON03 -> MBON13 and CRE055 -> MBON13
  set to 0) or "+chain_entry" (MBON05 -> MBON09 / MBON11 / MBON01 set to 0). apply_u_edit multiplies the APL -> P
  edges by float32(f) and then adds float32(0) to every edited weight, so f = 0 stores +0.0 exactly as
  q_jobs.apply_q_edit's zero (the weights are negative; -w × 0 is -0.0, whose bytes — and CSC sha — differ). It returns
  the CSC sha, the APL -> P mask size (2 on the real connectome, for every f — the edit's edge count, U.1) and the
  per-pair counts of a contrast block. "none" stays q_jobs' "none" (no change, 0 edges).
- Jobs (U.9.4 P2-5: oracle, P arm, KC activity, reference set / rest): u_kc_activity_job, u_arm_job and u_oracle_job
  are r_jobs.kc_activity_job / r_arm_job / r_oracle_job (r_oracle_job = q_jobs.q_oracle_job + "r") copied onto U's rig
  u_rig (q_jobs.q_rig's construction, U's edit); with edit "none" each one calls the R job itself, so C and E0 run R's
  verified path. u_ref_job / u_rest_job are t_measure.t_ref_job / t_rest_job's call sequence on u_engine (h3_jobs.engine_for's
  construction, U's edit) with U.6's mechanism reads added inside the read loop (reading the membrane and the release
  does not change the engine): every listed type's summed read-window count (all cells of the type, h4_jobs.type_cells —
  MBON and non-MBON), the active fraction of each listed KC subtype and of all KCs, and the graded APL release per read
  step and APL cell (q_jobs._present_kc_apl's reading).
- UPool: the pool adapter RMeasurer runs through — run_jobs(r_jobs.X, kws) runs U's copy of X on the real pool; any
  other job refuses. RMeasurer itself is used unchanged (one oracle path, T's Reading 4).
- UZMeasurer: u_ref_job / u_rest_job over a pool in h3_measure.chunks, no cache (U.9.1, U.3 3 (a))."""
from __future__ import annotations

import contextlib
import hashlib
import time

import numpy as np

from . import o_jobs, q_jobs, r_jobs
from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .h3_measure import chunks
from .h3_store import code_key
from .h4_formula import dprime, dv
from .h4_jobs import _present_kc, type_cells
from .n_jobs import _present, readout_cells
from .plasticity import Plasticity
from .presentation import decide
from .r_measure import R_MEASURE_FILES
from .stimuli import present
from .t_measure import T_MEASURE_FILES

U_MEASURE_FILES = ("flymon/brain/u_measure.py",)
NONE = q_jobs.NONE
PREFIX = "u_apl_mbon05_x"
F_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
F_ALLOWED = (0.0,) + F_GRID + (1.0,)
BLOCKS = {"out_block": (("MBON03", "MBON13"), ("CRE055", "MBON13")),
          "chain_entry": (("MBON05", "MBON09"), ("MBON05", "MBON11"), ("MBON05", "MBON01"))}
_RIG: dict = {}          # (Params, edit, p_type) -> (Engine, Plasticity, comps, sha, n, blocks); one per worker
_ZRIG: dict = {}         # (Params, edit, p_type, types, kc_types) -> (Engine, cells, kc cells, sha, n, blocks)


def u_measure_key(npz: str) -> dict:
    """U.8: the T measurement key's files plus U's measurement file."""
    return code_key(npz, files=tuple(dict.fromkeys(R_MEASURE_FILES + T_MEASURE_FILES + U_MEASURE_FILES)))


def u_edit(f: float, block: str | None = None) -> str:
    f = float(f)
    if f not in F_ALLOWED:
        raise ValueError(f"f {f!r} is not one of U's {F_ALLOWED}")
    if block is not None and block not in BLOCKS:
        raise ValueError(f"unknown contrast block {block!r}; U has {sorted(BLOCKS)}")
    return f"{PREFIX}{f:.1f}" + ("" if block is None else f"+{block}")


def is_u_edit(edit: str) -> bool:
    return isinstance(edit, str) and edit.startswith(PREFIX)


def parse_u_edit(edit: str) -> tuple:
    """(f, block or None); ValueError for anything u_edit does not write."""
    if not is_u_edit(edit):
        raise ValueError(f"{edit!r} is not a U edit")
    head, _, block = edit[len(PREFIX):].partition("+")
    f = float(head)
    if u_edit(f, block or None) != edit:
        raise ValueError(f"{edit!r} is not in u_edit's form")
    return f, (block or None)


def apply_u_edit(eng, pops, edit: str, p_type: str) -> tuple:
    """In place on eng.csc.w (graded-APL views follow): (sha256 of csc.w, APL -> p_type edge count, {pair: count})."""
    if edit == NONE:
        return hashlib.sha256(eng.csc.w.tobytes()).hexdigest(), 0, {}
    f, block = parse_u_edit(edit)
    src, tgt = edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
    t = np.asarray(eng.conn.type).astype(str)
    is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
    m = is_apl[src] & (t[tgt] == p_type)
    eng.csc.w[m] = eng.csc.w[m] * np.float32(f) + np.float32(0.0)
    counts = {}
    for a, b in BLOCKS.get(block, ()):
        mb = (t[src] == a) & (t[tgt] == b)
        eng.csc.w[mb] = np.float32(0.0)
        counts[f"{a}->{b}"] = int(mb.sum())
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest(), int(m.sum()), counts


def u_rig(conn, pops, params, edit: str, p_type: str):
    """q_jobs.q_rig's construction (Engine, the edit in place, compartments, Plasticity after the edit) with U's edit;
    one rig per worker, rebuilt on a switch."""
    key = (params, edit, p_type)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha, n, blocks = apply_u_edit(eng, pops, edit, p_type)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha, n, blocks)
    return _RIG[key]


# ================================================================ copies of R's jobs on U's rig (U.9.4 P2-5)
def u_kc_activity_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, items, strength: float,
                      settle_ms: float, read_ms: float, window_ms: int) -> list:
    """r_jobs.kc_activity_job on U's rig ("none": R's job itself)."""
    if not is_u_edit(edit):
        return r_jobs.kc_activity_job(eng, pl, pops, comps, ro, params=params, edit=edit, p_type=p_type, items=items,
                                      strength=strength, settle_ms=settle_ms, read_ms=read_ms, window_ms=window_ms)
    e, p, _, sha, n_edit, _ = u_rig(eng.conn, pops, params, edit, p_type)
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


def u_arm_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, odor_x: dict, odor_y: dict, seed: int,
              arm: str, punish: bool, plastic: bool, da_zero: bool, readout: dict, punish_type: str, reward_type: str,
              strength: float, settle_ms: float, read_ms: float, window_ms: int, trials: int, present_ms: float,
              gap_ms: float, train_settle_ms: float, seed_base: int, seed_stride: int) -> dict:
    """r_jobs.r_arm_job on U's rig ("none": R's job itself)."""
    kw = dict(params=params, edit=edit, p_type=p_type, odor_x=odor_x, odor_y=odor_y, seed=seed, arm=arm,
              punish=punish, plastic=plastic, da_zero=da_zero, readout=readout, punish_type=punish_type,
              reward_type=reward_type, strength=strength, settle_ms=settle_ms, read_ms=read_ms, window_ms=window_ms,
              trials=trials, present_ms=present_ms, gap_ms=gap_ms, train_settle_ms=train_settle_ms,
              seed_base=seed_base, seed_stride=seed_stride)
    if not is_u_edit(edit):
        return r_jobs.r_arm_job(eng, pl, pops, comps, ro, **kw)
    e, p, c, sha, n_edit, _ = u_rig(eng.conn, pops, params, edit, p_type)
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


def _q_oracle(e, p, c, sha, n_edit, pops, edit, odor_x, odor_y, readout, z, types, act_seeds, select_seeds,
              report_seeds, alphas, fixed_alphas, active_fx, strength, settle_ms, read_ms, window_ms, punish_type,
              reward_type) -> dict:
    """q_jobs.q_oracle_job's body after its rig line (copied unchanged)."""
    cells = type_cells(e.conn, types)
    idx = np.concatenate([cells[n] for n in types])
    bounds = np.cumsum([0] + [len(cells[n]) for n in types])
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win, apl = [], [], [], []
            for s in act_seeds:
                o = q_jobs._present_kc_apl(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"]); apl.append(o["apl_out_per_step"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}, apl

        def probe(seeds):
            out = {n: [] for n in types}
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                for j, n in enumerate(types):
                    out[n].append(cnt[:, bounds[j]:bounds[j + 1]].sum(1).tolist())
            return out

        def ap(pr):
            return {"A": pr[readout["A"]], "P": pr[readout["P"]]}

        fx, kc_x, apl_x = activity(odor_x)
        fy, kc_y, apl_y = activity(odor_y)
        rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core]); pun = np.isin(p.post_mb, p.mb_local[c[punish_type].core])
        w = e.csc.w

        def set_w(a_r, a_p=None):
            wv = p.w0.copy()
            if a_r is not None:
                wv[rew] = p.w0[rew] * (1.0 - a_r * fx)[p.pre_kc[rew]]
            if a_p is not None:
                wv[pun] = p.w0[pun] * (1.0 - a_p * fx)[p.pre_kc[pun]]
            w[p.edges] = wv

        set_w(None); pre_sel = probe(select_seeds)
        reward = {}
        for a in alphas:
            set_w(a); r1 = probe(select_seeds)
            reward[str(a)] = {"R1": r1, "change": dprime(dv(ap(r1), z) - dv(ap(pre_sel), z))}
        a_r = max(alphas, key=lambda a: (reward[str(a)]["change"], -a))
        punish = {}
        for a in alphas:
            set_w(a_r, a); r2 = probe(select_seeds)
            punish[str(a)] = {"R2": r2, "change": dprime(dv(ap(r2), z) - dv(ap(reward[str(a_r)]["R1"]), z))}
        a_p = min(alphas, key=lambda a: (punish[str(a)]["change"], a))
        set_w(None); pre = probe(report_seeds)
        set_w(a_r); R1 = probe(report_seeds)
        set_w(a_r, a_p); R2 = probe(report_seeds)
        fixed = {}
        for a in fixed_alphas:
            set_w(float(a)); fa = probe(report_seeds)
            fixed[str(float(a))] = {"counts": fa, "report": ap(fa)}
        w[p.edges] = p.w0
        nz = np.flatnonzero(fx)
        return {"select": {"pre": pre_sel, "reward": reward, "punish": punish},
                "alpha_reward": a_r, "alpha_punish": a_p,
                "counts": {"pre": pre, "R1": R1, "R2": R2},
                "report": {"pre": ap(pre), "R1": ap(R1), "R2": ap(R2)},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))},
                "q": {"edit": edit, "csc_sha256": sha, "edit_edges": n_edit, "fixed": fixed,
                      "apl_out": {"x": apl_x, "y": apl_y},
                      "fx": {"idx": nz.tolist(), "val": fx[nz].tolist()},
                      "reach": q_jobs.reach(e, p, c, fx, reward_type, readout["P"], active_fx)}}
    finally:
        p.reset_weights(); p.set_enabled(True)


def u_oracle_job(eng, pl, pops, comps, ro, **kw) -> dict:
    """r_jobs.r_oracle_job (q_jobs.q_oracle_job, then the P type's per-cell naive probe "r") on U's rig ("none": R's
    job itself)."""
    if not is_u_edit(kw["edit"]):
        return r_jobs.r_oracle_job(eng, pl, pops, comps, ro, **kw)
    p_type = kw["readout"]["P"]
    e, p, c, sha, n_edit, _ = u_rig(eng.conn, pops, kw["params"], kw["edit"], p_type)
    res = _q_oracle(e, p, c, sha, n_edit, pops, **{k: v for k, v in kw.items() if k != "params"})
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


JOBS = {r_jobs.kc_activity_job: u_kc_activity_job, r_jobs.r_arm_job: u_arm_job, r_jobs.r_oracle_job: u_oracle_job}


class UPool:
    """The pool RMeasurer runs U's measurements through: R's job function in, U's copy out (U.9.4 P2-5)."""

    def __init__(self, pool):
        self.pool = pool
        self.n_workers = 1 if pool is None else pool.n_workers

    def run_jobs(self, fn, kws):
        if fn not in JOBS:
            raise ValueError(f"U has no copy of {getattr(fn, '__name__', fn)}")
        return self.pool.run_jobs(JOBS[fn], kws)


# ================================================================ reference set / rest with U.6's reads
def u_engine(conn, pops, params, edit: str, p_type: str, types, kc_types):
    """t_measure.z_engine's construction (Engine(conn, pops, params, seed=0), no plasticity) with U's edit; the cells of
    every listed type (all neurons of the type) and of every listed KC subtype."""
    key = (params, edit, p_type, tuple(types), tuple(kc_types))
    if key not in _ZRIG:
        _ZRIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha, n, blocks = apply_u_edit(eng, pops, edit, p_type)
        t = np.asarray(conn.type).astype(str)
        kc = np.asarray(pops.kc, np.int64)
        _ZRIG[key] = (eng, type_cells(conn, list(types)), {k: kc[t[kc] == k] for k in kc_types}, sha, n, blocks)
    return _ZRIG[key]


def _read(e, pops, steps: int, kc_cells: dict) -> tuple:
    """t_measure._read's loop (counts per neuron over `steps` steps) plus the graded APL release per read step and APL
    cell (q_jobs._present_kc_apl's reading; spiking APL: its read-window spikes)."""
    counts = np.zeros(e.N, np.int32)
    apl = np.asarray(pops.apl, np.int64)
    graded = e.p.apl_mode == "graded"
    rel = 0.0
    for _ in range(int(steps)):
        counts[e.step()] += 1
        if graded:
            rel += float(e.apl_release(e.v[apl]).sum())
    if not graded:
        rel = float(counts[apl].sum())
    kc = counts[pops.kc]
    mech = dict(kc_sub={k: float((counts[v] > 0).mean()) for k, v in kc_cells.items()},
                apl_out_per_step=rel / (max(int(steps), 1) * max(apl.size, 1)))
    return counts, kc, mech


def u_ref_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, types, kc_types, odors, strength: float,
              settle_ms: float, read_steps: int) -> list:
    """t_measure.t_ref_job's sequence (reset / clear_drive / present / run(settle) / step × read) on u_engine; each row
    = t_ref_job's row (types, kc_active_frac, csc_sha256, edit_edges) plus "mech" and the contrast block counts."""
    e, cells, kc_cells, sha, n, blocks = u_engine(eng.conn, pops, params, edit, p_type, types, kc_types)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], strength); e.run(settle_ms)
            counts, kc, mech = _read(e, pops, read_steps, kc_cells)
            out.append(dict(odor=o["name"], seed=int(seed), types={t: int(counts[cells[t]].sum()) for t in types},
                            kc_active_frac=float((kc > 0).mean()), csc_sha256=sha, edit_edges=int(n),
                            block_edges=dict(blocks), mech=mech))
    return out


def u_rest_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, types, kc_types, seeds, settle_ms: float,
               read_steps: int) -> list:
    """t_measure.t_rest_job's sequence (no odour) on u_engine, with the same reads."""
    e, cells, kc_cells, sha, n, blocks = u_engine(eng.conn, pops, params, edit, p_type, types, kc_types)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); e.run(settle_ms)
        counts, kc, mech = _read(e, pops, read_steps, kc_cells)
        out.append(dict(seed=int(seed), types={t: int(counts[cells[t]].sum()) for t in types},
                        kc_active_frac=float((kc > 0).mean()), csc_sha256=sha, edit_edges=int(n),
                        block_edges=dict(blocks), mech=mech))
    return out


class UZMeasurer:
    """u_ref_job / u_rest_job over a pool in h3_measure.chunks; no cache (each scan point is minutes of work and is
    written once under results/u/ by the runner)."""

    def __init__(self, pool, params, p_type: str, types, kc_types):
        self.pool, self.params, self.p_type = pool, params, p_type
        self.types, self.kc_types = [str(t) for t in types], [str(t) for t in kc_types]

    def _common(self, edit: str, settle_ms: float, read_steps: int) -> dict:
        return dict(params=self.params, edit=edit, p_type=self.p_type, types=list(self.types),
                    kc_types=list(self.kc_types), settle_ms=float(settle_ms), read_steps=int(read_steps))

    def reference(self, edit: str, odors: list, strength: float, settle_ms: float, read_steps: int) -> list:
        common = dict(self._common(edit, settle_ms, read_steps), strength=float(strength))
        parts = self.pool.run_jobs(u_ref_job, [dict(common, odors=c) for c in chunks(odors, self.pool.n_workers)])
        return [r for part in parts for r in part]

    def rest(self, edit: str, seeds: list, settle_ms: float, read_steps: int) -> list:
        common = self._common(edit, settle_ms, read_steps)
        seeds = [int(s) for s in seeds]
        parts = self.pool.run_jobs(u_rest_job, [dict(common, seeds=c) for c in chunks(seeds, self.pool.n_workers)])
        return [r for part in parts for r in part]
