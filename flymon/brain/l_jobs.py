"""The FlyPool job of spec appendix L's screening (L.11.3): the naive half of h4_jobs.oracle_job with the identical
call sequence — weights reset, plasticity off, X's then Y's KC activity on the act seeds, weights at w0, the
two-odour decision probes on the select seeds — so a screened pair's features equal the ones the oracle would compute.
An accepted copy (no existing module changes); test_l_measure pins it to oracle_job's select-seed pre and kc.
Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level for the spawn pool."""
from __future__ import annotations

import numpy as np

from .h4_jobs import _present_kc, rig_for, type_cells
from .presentation import decide


def naive_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, types, act_seeds, select_seeds,
              strength: float, settle_ms: float, read_ms: float, window_ms: int) -> dict:
    """fx / fy as sparse (index, fire fraction) lists, pre = {type: [[x, y] per select seed]}, kc as oracle_job's."""
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

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
        e.csc.w[p.edges] = p.w0                      # oracle_job's set_w(None)
        pre = {n: [] for n in types}
        for s in select_seeds:
            cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
            for j, n in enumerate(types):
                pre[n].append(cnt[:, bounds[j]:bounds[j + 1]].sum(1).tolist())
        nzx, nzy = np.flatnonzero(fx), np.flatnonzero(fy)
        return {"fx_idx": nzx.tolist(), "fx_val": fx[nzx].tolist(), "fy_idx": nzy.tolist(), "fy_val": fy[nzy].tolist(),
                "pre": pre, "kc": {"x": kc_x, "y": kc_y,
                                   "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}
    finally:
        p.reset_weights(); p.set_enabled(True)
