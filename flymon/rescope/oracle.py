"""The reward-only oracle and the pair qualification decision of the re-scoped claim (spec 10).

`reward_oracle_job` is the reward half of `flymon.brain.h4_jobs.oracle_job`, copied with the punishment sweep and R2
removed; tests/rescope/test_oracle.py pins it to the original (alpha_reward, select reward block, report pre / R1, kc).
`qualify` / `qual_x` are pure.
"""
from __future__ import annotations

import numpy as np

from ..brain import h4_formula
from ..brain.b_rules import dprime
from ..brain.h4_formula import dv
from ..brain.h4_jobs import _present_kc, decide, rig_for, type_cells


def reward_oracle_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, readout: dict, z: dict, types,
                      act_seeds, select_seeds, report_seeds, alphas, strength: float, settle_ms: float,
                      read_ms: float, window_ms: int, reward_type: str) -> dict:
    """Reward half of flymon.brain.h4_jobs.oracle_job (accepted copy, pinned by tests/rescope/test_oracle.py): the G.14.3 'freq' edit on reward-core KC->MBON edges only."""
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
        rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core])
        w = e.csc.w

        def set_w(a_r):
            wv = p.w0.copy()
            if a_r is not None:
                wv[rew] = p.w0[rew] * (1.0 - a_r * fx)[p.pre_kc[rew]]
            w[p.edges] = wv

        set_w(None); pre_sel = probe(select_seeds)
        reward = {}
        for a in alphas:
            set_w(a); r1 = probe(select_seeds)
            reward[str(a)] = {"R1": r1, "change": h4_formula.dprime(dv(ap(r1), z) - dv(ap(pre_sel), z))}
        a_r = max(alphas, key=lambda a: (reward[str(a)]["change"], -a))
        set_w(None); pre = probe(report_seeds)
        set_w(a_r); R1 = probe(report_seeds)
        w[p.edges] = p.w0
        return {"select": {"pre": pre_sel, "reward": reward},
                "alpha_reward": a_r,
                "counts": {"pre": pre, "R1": R1},
                "report": {"pre": ap(pre), "R1": ap(R1)},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}
    finally:
        p.reset_weights(); p.set_enabled(True)


def qual_x(naive: dict, spec) -> str:
    """X = the odour with the larger naive MBON05 median over the qualification seeds; a tie goes to spec.x_tie."""
    a, b = float(np.median(naive["a"])), float(np.median(naive["b"]))
    return spec.x_tie if a == b else ("a" if a > b else "b")


def qualify(x: str, report: dict, naive: dict, spec, z) -> dict:
    """Qualified <=> naive MBON05 >= spec.floor_spikes on every seed for both odours and the reward-oracle change
    r = d'(dV_R1 - dV_pre) >= spec.oracle_min. report = reward_oracle_job's report {"pre", "R1"} ({"A", "P"} rows)."""
    floor_ok = bool(min(min(naive["a"]), min(naive["b"])) >= spec.floor_spikes)
    r = dprime(dv(report["R1"], z) - dv(report["pre"], z))
    reasons = ([] if floor_ok else ["naive MBON05 below floor on some seed"]) + \
              ([] if (r is not None and r >= spec.oracle_min) else [f"oracle reward change d' {r} < {spec.oracle_min}"])
    return dict(qualified=not reasons, r=r, floor_ok=floor_ok, x=x, reasons=reasons)
