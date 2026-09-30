"""The pure rules of spec appendix N as amended by N.8. No engine, no pool.
- The presentation state (N.8.8).
- N0f's per-cell statistics and operating-point rule (N.8.3), with the state-share flags.
- N0's similarity gate (N.8.4).
- N1's per-pair oracle reading and gate (N.3, N.8.5).
- N2's raw-unit statistics, validity checks and verdict (N.8.6).
- N2.0's thresholds, budget and outcome.
- The design view N2.0 may read from block n0f (blinding), and the sentences."""
from __future__ import annotations

import numpy as np

from .h4_formula import dprime, dv

OPERATING_POINT = "OPERATING_POINT"
STOP_DATA_MISMATCH = "STOP_DATA_MISMATCH"
STOP_NO_OPERATING_POINT = "STOP_NO_OPERATING_POINT"
SIMILARITY_GO = "SIMILARITY_GO"
STOP_SIMILARITY_ORDER = "STOP_SIMILARITY_ORDER"
N1_GO = "N1_GO"
STOP_UNTESTABLE = "STOP_UNTESTABLE"
N2_0_GO = "N2_0_GO"
STOP_POWER = "STOP_POWER"
STOP_BUDGET = "STOP_BUDGET"
SUPPORTED = "SUPPORTED"
NOT_REPLICATED = "NOT_REPLICATED"
NO_LEARNING = "NO_LEARNING"
INVALID = "INVALID"


def state(P: int, spec) -> str:
    return "firing" if int(P) >= spec.state_p_min else "silent"


# ================================================================ N0f (N.8.3)
def cell_stats(rows: list, spec) -> dict:
    """One (odour, condition) block of presentation rows -> N.8.3's numbers."""
    kc = np.array([r["kc_frac"] for r in rows], float)
    hz = np.array([r["kc_max_win_hz"] for r in rows], float)
    A = np.array([r["A"] for r in rows], float); P = np.array([r["P"] for r in rows], float)
    return dict(n=len(rows), kc_frac_median=float(np.median(kc)), kc_frac=kc.tolist(),
                runaway_share=float((hz > spec.runaway_hz).mean()), max_win_hz_max=float(hz.max()),
                A_median=float(np.median(A)), P_median=float(np.median(P)),
                A_zero_share=float((A == 0).mean()), P_zero_share=float((P == 0).mean()),
                firing_share=float((P >= spec.state_p_min).mean()),
                apl_out_mean=float(np.mean([r["apl_out_per_step"] for r in rows])),
                wall_s_median=float(np.median([r["wall_s"] for r in rows])))


def wall_per_step(rows: list) -> float:
    return float(np.median([r["wall_s"] / r["steps"] for r in rows]))


def point_key(g: float, c_delta: float) -> str:
    return f"{g:g}|{c_delta:g}"


def point_checks(cells: dict, spec) -> dict:
    """cells {stimulus: {"on" | "block" | "all": cell_stats}} at one (g, c_δ). Returns N.8.3's clauses ① ② ③ on the
    judged stimuli (IA / EB alone are records) and the distance of the on medians' mean from the target."""
    lo, hi = spec.kc_band
    on = [cells[s]["on"] for s in spec.judged_stimuli]
    both = on + [cells[s]["block"] for s in spec.judged_stimuli]
    band = all(lo <= c["kc_frac_median"] <= hi for c in on)
    runaway = all(c["kc_frac_median"] <= spec.kc_block_max and c["runaway_share"] <= spec.runaway_share_max
                  for c in both)
    floor = all(c["A_zero_share"] <= spec.zero_share_max and c["P_zero_share"] <= spec.zero_share_max for c in both)
    mean_on = float(np.mean([c["kc_frac_median"] for c in on]))
    return dict(band=bool(band), runaway=bool(runaway), floor=bool(floor), ok=bool(band and runaway and floor),
                mean_on=mean_on, dist=abs(mean_on - spec.kc_target))


def select_point(grid: dict, spec) -> dict:
    """grid {(g, c_δ): cells} -> the passing point nearest the target; ties -> smaller g, then smaller c_δ."""
    checks = {k: point_checks(v, spec) for k, v in grid.items()}
    table = {point_key(*k): c for k, c in checks.items()}
    ok = [k for k, c in checks.items() if c["ok"]]
    if not ok:
        return dict(outcome=STOP_NO_OPERATING_POINT, selected=None, checks=table)
    g, c = min(ok, key=lambda k: (checks[k]["dist"], k[0], k[1]))
    return dict(outcome=OPERATING_POINT, selected=dict(g=float(g), c_delta=float(c)), checks=table)


def state_flags(cells: dict, spec) -> list:
    """Every (stimulus, block condition) whose firing share differs from the on share by more than 0.25 (a record)."""
    out = []
    for s, by in cells.items():
        for cond in ("block", "all"):
            if abs(by[cond]["firing_share"] - by["on"]["firing_share"]) > spec.state_diff_flag:
                out.append(dict(stimulus=s, condition=cond, on=by["on"]["firing_share"],
                                share=by[cond]["firing_share"]))
    return out


# ================================================================ N0 (N.8.4)
def _pearson_rows(a, b) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True); b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), np.nan)


def similarity(fired: dict, n_kc: int, spec, draws: int | None = None, seed: int | None = None) -> dict:
    """fired {stimulus: [fired KC positions per seed]} (the same seeds in the same order) -> r(4:1, 1:4),
    r(4:1, δ-DL) of the KC firing-probability vectors, Δr, and the common-seed bootstrap 95% CI. SIMILARITY_GO iff
    the lower bound is finite and > 0; an undefined r (a silent vector) fails."""
    x, y_sim = spec.pair_stimuli()["sim"]
    y_dis = spec.pair_stimuli()["dis"][1]
    n = len(fired[x])
    if n < 2 or any(len(fired[s]) != n for s in (y_sim, y_dis)):
        raise ValueError("similarity needs the same >= 2 seeds for every stimulus")
    M = {}
    for s in (x, y_sim, y_dis):
        m = np.zeros((n, int(n_kc)))
        for i, f in enumerate(fired[s]):
            m[i, np.asarray(f, np.int64)] = 1.0
        M[s] = m
    f = {s: M[s].mean(0) for s in M}
    r_sim, r_dis = float(_pearson_rows(f[x], f[y_sim])), float(_pearson_rows(f[x], f[y_dis]))
    draws = spec.boot_draws if draws is None else int(draws)
    idx = np.random.default_rng(spec.boot_seed if seed is None else seed).integers(0, n, (draws, n))
    wt = np.stack([np.bincount(row, minlength=n) for row in idx]) / n                  # draws x seeds weights
    d = np.empty(draws)
    for a in range(0, draws, 500):
        fb = {s: wt[a:a + 500] @ M[s] for s in M}
        d[a:a + 500] = _pearson_rows(fb[x], fb[y_sim]) - _pearson_rows(fb[x], fb[y_dis])
    if np.isfinite(d).all():
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    else:
        lo = hi = float("nan")
    ok = bool(np.isfinite(lo) and lo > 0)
    return dict(outcome=SIMILARITY_GO if ok else STOP_SIMILARITY_ORDER, r_sim=r_sim, r_dis=r_dis,
                delta_r=r_sim - r_dis, ci95=[lo, hi], draws=draws, n_seeds=n)


# ================================================================ N1 (N.3, N.8.5)
def n1_pair(row: dict, z: dict, spec) -> dict:
    """p0 = d'(dV_P - dV_pre) on the report seeds; testable iff p0 <= -testable_min and sd(dV_P - dV_pre) > 0;
    o = mean(dV_P - dV_pre) in z units (N.8.6's c1)."""
    pre, post = dv(row["report"]["pre"], z), dv(row["report"]["P"], z)
    d = post - pre
    p0 = dprime(d)
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    A = np.asarray(row["report"]["pre"]["A"], float); P = np.asarray(row["report"]["pre"]["P"], float)
    return dict(p0=p0, sd=sd, o=float(d.mean()), d_pre=dprime(pre),
                testable=bool(p0 is not None and p0 <= -spec.testable_min and sd > 0), alpha=row["alpha_punish"],
                changes={a: v["change"] for a, v in row["select"]["punish"].items()}, jaccard=row["kc"]["jaccard"],
                floor=dict(A_zero_share=float((A == 0).mean()), P_zero_share=float((P == 0).mean())))


def n1_outcome(pairs: dict) -> dict:
    t = {k: bool(v["testable"]) for k, v in pairs.items()}
    if all(t.values()):
        return dict(outcome=N1_GO, note=None)
    note = ("다른 쌍만 시험 가능: APL이 있어도 비슷한 쌍의 변별이 사전 지정 편집 프로토콜에서 서지 않는다"
            if t.get("dis") and not t.get("sim") else None)
    return dict(outcome=STOP_UNTESTABLE, note=note)
