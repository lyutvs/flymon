"""FlyPool worker jobs for the M0d H.4 runner (spec appendix H.4): the conditioning arms of the readout reselection and
the oracle.

Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them.
Every job builds (or reuses) its own rig — Engine, Plasticity, compartments — from the `Params` it is given
(h3_jobs' pattern); the worker's default engine only lends its connectome. The call sequences are the committed
ones, so the recorded numbers reproduce: `teach_job` is `conditioning.run_arm` (the M0c arm, as
`pool_jobs.conditioning_arm_job` runs it) with per-type counts; `oracle_job` is `m2_engine_probe.oracle_job` (G.14.3)
with the readout types, their z constants and the combination's engine passed in instead of MBON13 / MBON05, F.3's
constants and `configure()`.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .circuits import compartments
from .conditioning import arms, train_block
from .engine_cpu import Engine
from .h4_formula import dprime, dv
from .plasticity import Plasticity
from .presentation import decide
from .stimuli import design_odor_pair, present

_RIG: dict = {}          # Params -> (Engine, Plasticity, comps); one entry per worker


def rig_for(conn, pops, params):
    """The worker's engine, plasticity and compartments for `params`, built as FlyPool's worker builds them
    (fly_pool._WorkerState.get) and reused while jobs keep asking for the same Params."""
    if params not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[params] = (eng, Plasticity(eng, pops, comps), comps)
    return _RIG[params]


def type_cells(conn, types) -> dict:
    t = np.asarray(conn.type).astype(str)
    return {n: np.flatnonzero(t == n) for n in types}


# ================================================================ readout reselection: the M0c single-channel arms
def teach_job(eng, pl, pops, comps, ro, params, seed: int, arm: str, order: str, types, punish_type: str,
              reward_type: str, k: int, odor_seed: int, strength: float, trials: int, present_ms: float, gap_ms: float,
              settle_ms: float, read_ms: float) -> dict:
    """conditioning.run_arm at one seed — probes (settle, read) of CS+ and CS- with the probe seed, the training block,
    the probes again — with every listed MBON type's count. order "ab": CS+ = odour a, CS- = odour b (M0c);
    "ba" exchanges them. The punish type is paired with the CS+ slot, the reward type with the CS- slot (arms())."""
    e, p, c = rig_for(eng.conn, pops, params)
    cells = type_cells(e.conn, types)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    cs_plus, cs_minus = (a, b) if order == "ab" else (b, a)

    def probe(odor):
        was = p.enabled
        p.set_enabled(False)
        e.reset(seed); p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, odor, strength)
        e.run(settle_ms)
        counts = e.run(read_ms)
        p.set_enabled(was)
        return {n: int(counts[i].sum()) for n, i in cells.items()}

    punish, reward, plastic = arms(punish_type, reward_type)[arm]
    try:
        p.reset_weights()
        pre = {"plus": probe(cs_plus), "minus": probe(cs_minus)}
        p.set_enabled(plastic)
        train_block(e, p, pops, cs_plus, cs_minus, strength, seed, punish, reward, trials, present_ms, gap_ms, settle_ms)
        p.set_enabled(True)
        post = {"plus": probe(cs_plus), "minus": probe(cs_minus)}
    finally:
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), arm=arm, order=order, pre=pre, post=post)


# ================================================================ the oracle (G.14.3)
def _present_kc(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms: int) -> dict:
    """m2_engine_probe.present_kc: one presentation; KC counts of the read window and the maximum sliding-window KC
    count over the whole presentation (G.8's window, recorded)."""
    kc = pops.kc
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    e.reset(seed); p.reset_traces(); e.clear_drive(); p.quiet_dan()
    present(e, pops, odor, strength)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32)
    n_settle, n_total = int(round(settle_ms / e.p.dt)), int(round((settle_ms + read_ms) / e.p.dt))
    for step in range(n_total):
        f = pos[e.step()]; f = f[f >= 0]
        win[f] += 1; hist.append(f)
        if len(hist) > window_ms:
            win[hist.popleft()] -= 1
        if f.size:
            max_win = max(max_win, int(win.max()))
        if step >= n_settle:
            read[f] += 1
    return {"read": read, "max_win": max_win}


def oracle_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, readout: dict, z: dict, types,
               act_seeds, select_seeds, report_seeds, alphas, strength: float, settle_ms: float, read_ms: float,
               window_ms: int, punish_type: str, reward_type: str) -> dict:
    """One pair: KC activity f_i of X on act_seeds; alpha picked on select_seeds (reward change max, then punishment
    change min, ties -> smaller alpha); pre / R1 / R2 on report_seeds. Probe counts are kept per listed type; V is
    built from readout = {"A": type, "P": type} with z = {"A": (mean, sd), "P": (mean, sd)}."""
    e, p, c = rig_for(eng.conn, pops, params)
    cells = type_cells(e.conn, types)
    idx = np.concatenate([cells[n] for n in types])
    bounds = np.cumsum([0] + [len(cells[n]) for n in types])
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win = [], [], []
            for s in act_seeds:
                o = _present_kc(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}

        def probe(seeds):
            out = {n: [] for n in types}
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                for j, n in enumerate(types):
                    out[n].append(cnt[:, bounds[j]:bounds[j + 1]].sum(1).tolist())
            return out

        def ap(pr):
            return {"A": pr[readout["A"]], "P": pr[readout["P"]]}

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
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
        w[p.edges] = p.w0
        return {"select": {"pre": pre_sel, "reward": reward, "punish": punish},
                "alpha_reward": a_r, "alpha_punish": a_p,
                "counts": {"pre": pre, "R1": R1, "R2": R2},
                "report": {"pre": ap(pre), "R1": ap(R1), "R2": ap(R2)},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}
    finally:
        p.reset_weights(); p.set_enabled(True)
