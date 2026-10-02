"""FlyPool worker job of spec appendix Q (Q.3, Q.6.2, Q.6.6). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle it; the worker's default engine only lends its connectome.

q_oracle_job is h4_jobs.oracle_job's call sequence (G.14.3) copied onto Q's rig (h4_jobs is imported, never edited):
activity of X and Y on act_seeds, alpha_r chosen on select_seeds (reward change max, then punishment change min, ties to
the smaller alpha), pre / R1 / R2 on report_seeds. After R2 it adds Q's records under key "q":
- the fixed reward-only arms (Q.6.2): w0 * (1 - alpha * fx) on the reward core at each fixed alpha, probed on the
  report seeds;
- X's KC firing probabilities fx (sparse), and reach (Q.6.2 W_X = sum w0 * fx over the reward core's edges onto the P
  readout type; Q.3 f_X = share of KCs with fx >= active_fx having such an edge; per-type sums);
- the APL output of every activity presentation (read window; graded: summed release per step; spiking: spikes),
  read as n_jobs._present reads it (reading the membrane does not change the engine).
With edit "none" the result minus "q" equals oracle_job's (test; the reproduction gate checks it against the encoder
cache, Q.6.8).

Edits (Q.6.6), applied in place on a freshly built engine's CSC (graded-APL views follow), one rig per worker cached
under (Params, edit, P type): "none"; "apl_to_mbon05_zero" — every APL out-edge onto a cell of the P readout type
(MBON05: 2 CSC edges on the real connectome) set to 0; "apl_to_nonkc_zero" — o_jobs.apply_edit, unchanged."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from . import o_jobs
from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .h4_formula import dprime, dv
from .h4_jobs import type_cells
from .plasticity import Plasticity
from .presentation import decide
from .stimuli import present

NONE = "none"
MBON05 = "apl_to_mbon05_zero"
NONKC = o_jobs.NONKC
EDITS = (NONE, MBON05, NONKC)
_RIG: dict = {}          # (Params, edit, p_type) -> (Engine, Plasticity, comps, csc sha256, edges changed)


def apply_q_edit(eng, pops, edit: str, p_type: str) -> tuple:
    """Q's edit in place on eng.csc.w (Q.6.6); (sha256 of csc.w, number of CSC edges changed).
    Adapted from o_jobs:apply_edit (MBON05 uses its mask pattern; NONKC calls it unchanged; O.5 / Q forbid editing O)."""
    if edit not in EDITS:
        raise ValueError(f"unknown Q edit {edit!r}; Q has {EDITS}")
    before = eng.csc.w.copy()
    if edit == MBON05:
        is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
        t = np.asarray(eng.conn.type).astype(str)
        m = is_apl[edge_sources(eng.csc)] & (t[eng.csc.tgt.astype(np.int64)] == p_type)
        eng.csc.w[m] = np.float32(0.0)
    elif edit == NONKC:
        o_jobs.apply_edit(eng, pops, NONKC)
    n = int((before != eng.csc.w).sum())
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest(), n


def q_rig(conn, pops, params, edit: str, p_type: str):
    """The worker's rig for (params, edit, p_type), built once and reused; a switch rebuilds, a cached engine is never
    re-edited. A different mv_per_synapse (mv_scale, Q.6.5) is a different Params, hence a different rig.
    Copied from o_jobs:rig (Q's own cache; h4_jobs._RIG and o_jobs._RIG are never touched)."""
    key = (params, edit, p_type)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha, n = apply_q_edit(eng, pops, edit, p_type)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha, n)
    return _RIG[key]


def _present_kc_apl(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms: int) -> dict:
    """h4_jobs:_present_kc (copied; same step sequence, so the KC reads are identical) with the read-window APL output
    added as n_jobs:_present reads it (graded: summed apl_release per read step; spiking: APL read-window spikes),
    per read step and APL cell."""
    kc = pops.kc
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    apl = np.asarray(pops.apl, np.int64)
    graded = e.p.apl_mode == "graded"
    e.reset(seed); p.reset_traces(); e.clear_drive(); p.quiet_dan()
    present(e, pops, odor, strength)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32)
    apl_out = 0.0
    n_settle, n_total = int(round(settle_ms / e.p.dt)), int(round((settle_ms + read_ms) / e.p.dt))
    for step in range(n_total):
        fired = e.step()
        f = pos[fired]; f = f[f >= 0]
        win[f] += 1; hist.append(f)
        if len(hist) > window_ms:
            win[hist.popleft()] -= 1
        if f.size:
            max_win = max(max_win, int(win.max()))
        if step >= n_settle:
            read[f] += 1
            if graded:
                apl_out += float(e.apl_release(e.v[apl]).sum())
            else:
                apl_out += float(np.isin(fired, apl).sum())
    return {"read": read, "max_win": max_win,
            "apl_out_per_step": apl_out / ((n_total - n_settle) * max(apl.size, 1))}


def reach(e, p, c, fx, reward_type: str, p_type: str, active_fx: float) -> dict:
    """Q.6.2 / Q.3 reach of X onto the reward core: W_X = sum w0 * fx over the reward core's KC -> p_type edges (on the
    real connectome both MBON05 cells are in PAM08's core, so this is every KC -> MBON05 edge); n_edges with fx > 0;
    n_active_kc (fx >= active_fx) and f_X, their share with such an edge; by_type / total (MBON21's share = by_type /
    total) as records."""
    t = np.asarray(e.conn.type).astype(str)
    post_t = t[e.csc.tgt[p.edges].astype(np.int64)]
    rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core])
    contrib = p.w0.astype(np.float64) * fx[p.pre_kc]
    on_p = rew & (post_t == p_type)
    by_type = {str(k): float(contrib[rew & (post_t == k)].sum()) for k in sorted(set(post_t[rew].tolist()))}
    total = float(contrib[rew].sum())
    active = fx >= active_fx
    has = np.zeros(len(fx), bool); has[p.pre_kc[on_p]] = True
    n_active = int(active.sum())
    return dict(W_X=float(contrib[on_p].sum()), n_edges=int((on_p & (fx[p.pre_kc] > 0)).sum()),
                n_active_kc=n_active, f_X=(float((active & has).sum() / n_active) if n_active else None),
                by_type=by_type, total=total)


def q_oracle_job(eng, pl, pops, comps, ro, params, edit: str, odor_x: dict, odor_y: dict, readout: dict, z: dict,
                 types, act_seeds, select_seeds, report_seeds, alphas, fixed_alphas, active_fx: float,
                 strength: float, settle_ms: float, read_ms: float, window_ms: int, punish_type: str,
                 reward_type: str) -> dict:
    """h4_jobs:oracle_job (copied, G.14.3) on Q's rig, plus "q" (module docstring). z comes from the caller (block
    h4) in every condition; alphas is the selection set only — fixed_alphas (0.8, 1.0) are report arms that never
    enter the selection. strength is the presentation s the caller passes ((c) lowers it, Q.6.9)."""
    e, p, c, sha, n_edit = q_rig(eng.conn, pops, params, edit, readout["P"])
    cells = type_cells(e.conn, types)
    idx = np.concatenate([cells[n] for n in types])
    bounds = np.cumsum([0] + [len(cells[n]) for n in types])
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win, apl = [], [], [], []
            for s in act_seeds:
                o = _present_kc_apl(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
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
                      "reach": reach(e, p, c, fx, reward_type, readout["P"], active_fx)}}
    finally:
        p.reset_weights(); p.set_enabled(True)
