"""The metrics of spec appendix K (K.8.3), pure functions over per-KC fire fractions f (fraction of act seeds with at
least one read-window spike, H.4a.7) and synapse-count weights (edges >= min_weight).

N_p = sum w13 f_x [f_y = 0] (X-only drive onto the target readout), D13 = sum w13 f_x, S = N_p / D13 (0 without drive),
N05 = the reward readout's N. An engine's N is the sum over the fixed pair list: no pair is dropped. Everything else is
recorded, never judged."""
from __future__ import annotations

import numpy as np

LOBES = {"apbp": "KCa'b'", "g": "KCg", "ab": "KCab"}


def _kc_pos(conn, pops) -> np.ndarray:
    pos = np.full(conn.N, -1, np.int64)
    pos[np.asarray(pops.kc, np.int64)] = np.arange(len(pops.kc))
    return pos


def readout_weights(conn, pops, mbon_type: str, min_weight: int) -> np.ndarray:
    pos = _kc_pos(conn, pops)
    cells = np.flatnonzero(np.asarray(conn.type).astype(str) == mbon_type)
    sel = (pos[conn.pre] >= 0) & np.isin(conn.post, cells) & (conn.w >= min_weight)
    return np.bincount(pos[conn.pre[sel]], weights=conn.w[sel].astype(np.float64), minlength=len(pops.kc))


def lobe_masks(conn, pops) -> dict:
    t = np.asarray(conn.type).astype(str)[np.asarray(pops.kc, np.int64)]
    return {k: np.char.startswith(t, prefix) for k, prefix in LOBES.items()}


def kc_kc_edges(conn, pops, min_weight: int) -> tuple:
    pos = _kc_pos(conn, pops)
    sel = (pos[conn.pre] >= 0) & (pos[conn.post] >= 0) & (conn.w >= min_weight)
    return pos[conn.pre[sel]], pos[conn.post[sel]], conn.w[sel].astype(np.float64)


def pair_metrics(fx, fy, w13, w05) -> dict:
    only = fx * (fy == 0)
    n, d = float(np.dot(w13, only)), float(np.dot(w13, fx))
    return dict(N=n, D13=d, S=n / d if d > 0 else 0.0, N05=float(np.dot(w05, only)))


def removed_fraction(fx, fy) -> float:
    """H.4a.8's mask measure: the share of X's activity on KCs Y also fires."""
    s = float(fx.sum())
    return float(fx[fy > 0].sum()) / s if s > 0 else 0.0


def engine_metrics(acts: list, w13, w05, lobes: dict) -> dict:
    per = [pair_metrics(a["fx"], a["fy"], w13, w05) for a in acts]
    jac = []
    for a in acts:
        x, y = a["fx"] > 0, a["fy"] > 0
        u = int((x | y).sum())
        jac.append(int((x & y).sum()) / u if u else 0.0)
    fx_all = np.concatenate([a["fx"] for a in acts]) if acts else np.zeros(0)
    act = fx_all[fx_all > 0]
    eighths = np.rint(act * 8).astype(int)
    return dict(N=float(sum(m["N"] for m in per)), N05=float(sum(m["N05"] for m in per)),
                S_median=float(np.median([m["S"] for m in per])) if per else 0.0,
                S=[m["S"] for m in per], D13=[m["D13"] for m in per], N_p=[m["N"] for m in per],
                jaccard_median=float(np.median(jac)) if jac else 0.0,
                lobes={k: float(np.mean([float((a["fx"][m] > 0).mean()) if m.any() else 0.0 for a in acts]))
                       for k, m in lobes.items()},
                reliability=dict(n_active=int(act.size), frac_always=float((act == 1.0).mean()) if act.size else 0.0,
                                 hist={str(k): int((eighths == k).sum()) for k in range(1, 9)}))


def kc_kc_input(counts, edges, g: float, mv_per_synapse: float) -> np.ndarray:
    """Per-KC inhibitory charge proxy: sum over KC->KC in-edges of the presynaptic mean count x synapses x |g| x mV."""
    pre, post, w = edges
    return np.bincount(post, weights=counts[pre] * w * abs(g) * mv_per_synapse, minlength=len(counts))
