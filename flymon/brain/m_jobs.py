"""FlyPool worker jobs of spec appendix M (M.10.1, readings 4-5). reference_cells_job / rest_cells_job are
h3_jobs.reference_job / rest_job's presentation sequence with the read counts of a cell list (every MBON cell) instead
of type sums; pre_job is h4_jobs.oracle_job up to its unedited probes (KC activity of X then Y on the act seeds, then
decide() on the select and on the report seeds, at w0, plasticity off) with per-cell counts; edit_job is the rest of
oracle_job — the reward alpha on the select seeds, the punishment alpha given it, then R1 and R2 on the report seeds —
with the readout V built from named cell groups. The edited KC -> MBON edges are those onto the taught DANs' core cells,
and edit_job refuses unless those cells equal the readout group's cells (reward -> P, punish -> A): a partial-type core
is edited and read on its core cells only. Composed on type-cell groups they are oracle_job (test). Signature
fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them."""
from __future__ import annotations

import numpy as np

from .h3_jobs import engine_for
from .h4_formula import dprime, dv
from .h4_jobs import _present_kc, rig_for
from .presentation import decide
from .stimuli import present


def group_counts(counts, cells, groups: dict) -> dict:
    """counts: [seed][2][len(cells)] -> {group: [[x, y] per seed]} (sums over the group's cells)."""
    pos = {int(c): k for k, c in enumerate(cells)}
    ix = {g: [pos[int(i)] for i in idx] for g, idx in groups.items()}
    arr = np.asarray(counts, np.int64)
    return {g: arr[:, :, k].sum(axis=2).tolist() for g, k in ix.items()}


def reference_cells_job(eng, pl, pops, comps, ro, params, odors, strength: float, settle_ms: float, read_steps: int,
                        cells) -> list:
    """h3_jobs.reference_job's presentations (reset, clear_drive, present, settle, read_steps steps); per-cell counts."""
    e, _ = engine_for(eng.conn, pops, params)
    idx = np.asarray(cells, np.int64)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], strength); e.run(settle_ms)
            counts = np.zeros(e.N, np.int32)
            for _ in range(read_steps):
                counts[e.step()] += 1
            out.append(dict(odor=o["name"], seed=int(seed), counts=counts[idx].tolist()))
    return out


def rest_cells_job(eng, pl, pops, comps, ro, params, seeds, settle_ms: float, read_steps: int, cells) -> list:
    """h3_jobs.rest_job's window with no odour; per-cell counts."""
    e, _ = engine_for(eng.conn, pops, params)
    idx = np.asarray(cells, np.int64)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); e.run(settle_ms)
        counts = np.zeros(e.N, np.int32)
        for _ in range(read_steps):
            counts[e.step()] += 1
        out.append(dict(seed=int(seed), counts=counts[idx].tolist()))
    return out


def _probe(e, p, pops, odor_x, odor_y, seeds, strength, settle_ms, read_ms, idx) -> list:
    return [decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx).tolist() for s in seeds]


def pre_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, cells, act_seeds, select_seeds, report_seeds,
            strength: float, settle_ms: float, read_ms: float, window_ms: int) -> dict:
    """oracle_job through its unedited probes: fx / fy on act_seeds, then select and report probes at w0 (per cell)."""
    e, p, c = rig_for(eng.conn, pops, params)
    idx = np.asarray(cells, np.int64)
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win = [], [], []
            for s in act_seeds:
                o = _present_kc(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
        e.csc.w[p.edges] = p.w0
        pre_sel = _probe(e, p, pops, odor_x, odor_y, select_seeds, strength, settle_ms, read_ms, idx)
        pre_rep = _probe(e, p, pops, odor_x, odor_y, report_seeds, strength, settle_ms, read_ms, idx)
        nx, ny = np.flatnonzero(fx), np.flatnonzero(fy)
        return {"fx_idx": nx.tolist(), "fx_val": fx[nx].tolist(), "fy_idx": ny.tolist(), "fy_val": fy[ny].tolist(),
                "pre_sel": pre_sel, "pre_rep": pre_rep,
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}
    finally:
        p.reset_weights(); p.set_enabled(True)


def edit_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, fx_idx, fx_val, cells, groups: dict,
             readout: dict, z: dict, pre_sel, select_seeds, report_seeds, alphas, strength: float, settle_ms: float,
             read_ms: float, reward_type: str, punish_type: str) -> dict:
    """oracle_job after its unedited probes: alpha_reward on select, alpha_punish given it, R1 / R2 on report. The
    reward edit's cells must equal groups[readout["P"]], the punish edit's groups[readout["A"]] (ValueError)."""
    e, p, c = rig_for(eng.conn, pops, params)
    idx = np.asarray(cells, np.int64)
    fx = np.zeros(len(pops.kc)); fx[np.asarray(fx_idx, np.int64)] = np.asarray(fx_val, float)
    rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core]); pun = np.isin(p.post_mb, p.mb_local[c[punish_type].core])
    mbon = np.asarray(pops.mbon, np.int64)
    edited = {}
    for arm, mask, key in (("reward", rew, "P"), ("punish", pun, "A")):          # taught cells == readout cells
        got = sorted(int(i) for i in mbon[np.unique(p.post_mb[mask])])
        want = sorted(int(i) for i in groups[readout[key]])
        if got != want:
            raise ValueError(f"{arm} edits cells {got} but readout {key} = {readout[key]!r} reads {want}")
        edited[arm] = {"group": readout[key], "cells": got}

    def ap(counts):
        g = group_counts(counts, cells, {"A": groups[readout["A"]], "P": groups[readout["P"]]})
        return {"A": g["A"], "P": g["P"]}

    try:
        p.reset_weights(); p.set_enabled(False)
        w = e.csc.w

        def set_w(a_r, a_p=None):
            wv = p.w0.copy()
            if a_r is not None:
                wv[rew] = p.w0[rew] * (1.0 - a_r * fx)[p.pre_kc[rew]]
            if a_p is not None:
                wv[pun] = p.w0[pun] * (1.0 - a_p * fx)[p.pre_kc[pun]]
            w[p.edges] = wv

        probe = lambda seeds: _probe(e, p, pops, odor_x, odor_y, seeds, strength, settle_ms, read_ms, idx)
        reward = {}
        for a in alphas:
            set_w(a); r1 = probe(select_seeds)
            reward[str(a)] = {"R": r1, "change": dprime(dv(ap(r1), z) - dv(ap(pre_sel), z))}
        a_r = max(alphas, key=lambda a: (reward[str(a)]["change"], -a))
        punish = {}
        for a in alphas:
            set_w(a_r, a); r2 = probe(select_seeds)
            punish[str(a)] = {"R": r2, "change": dprime(dv(ap(r2), z) - dv(ap(reward[str(a_r)]["R"]), z))}
        a_p = min(alphas, key=lambda a: (punish[str(a)]["change"], a))
        set_w(a_r); R1 = probe(report_seeds)
        set_w(a_r, a_p); R2 = probe(report_seeds)
        w[p.edges] = p.w0
        return {"reward": reward, "punish": punish, "alpha_reward": a_r, "alpha_punish": a_p, "R1_rep": R1, "R2_rep": R2,
                "readout": dict(readout), "edited": edited}
    finally:
        p.reset_weights(); p.set_enabled(True)
