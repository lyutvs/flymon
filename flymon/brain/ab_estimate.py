"""AB's estimators, intervals, calibration and verdict (AB.4 6697–6737, AB.5 6739–6766, AB.6 6768–6775): pure
functions, sealed before the first 2nd-gen measurement (AB.7 0d). A pair's raw counts are {stage: int [F, K, 2, 2]}
(w_verdict's layout); the four gates are w_verdict.GATES with directions SIGNS (+, −, +, −).
- Three statistics per gate (AB.4): D = the ±10-winsorized fly d′ (±∞ included; aa_estimate.pair_arrays), D_fin =
  the same with ±∞ flies set to NaN (finite |d′| > 10 stays ±10), R = the unstandardised contrast (probe mean per fly,
  z units). A pair's estimate is the fly mean (NaN-mean for D_fin); the set point is the unweighted pair mean.
- Two interval methods, both required (AB.5): TS = aa_estimate.boot_two_stage over the X-label groups
  (y_rules.merge_groups), B = boot_b, numpy percentile (linear) at α and 1 − α — the four gates share one draw, one
  stream per statistic ("pool" / "pool_drop" / "pool_raw", tag "S1"); CG = the two-way cluster-robust interval
  (X label × opponent type set, V* = max(V_X + V_T − V_XT, V_X, V_T), G/(G − 1) per definition, df = min(G_X, G_T) − 1).
  D_fin's TS keeps the full S1 group list and the shared draw (a pair without a finite fly is NaN in that gate and
  drops out of every replicate by nanmean; limits by nanpercentile) — plan Reading 6; its CG uses the remaining pairs.
- Calibration (AB.5): the structure is a tuple of (X group, type-set group) int pairs and nothing else (TypeError);
  24 cells, truth from truth_pairs simulated pairs, selection n_sel / verification n_ver reps on separate streams
  ("cal", tag, "sel" | "truth" | "ver", cell), joint miss of both methods, Clopper–Pearson one-sided upper bound,
  largest α of the grid that passes every cell; verification only at the chosen α. Reps run in chunks of cal_chunk and
  a chunk's end state (the PCG64 state) resumes the same stream bit for bit (plan Reading 9).
- Verdict (AB.4 판정 1–5): STOP_MACHINE → STOP_PROTOCOL → FAIL → PASS → UNDECIDED; comparisons rounded to 9 digits,
  strict; D / D_fin compared on the Hedges scale (× J(8)), R against ±raw_min."""
from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np
from scipy import stats

from . import aa_estimate as AE
from . import w_verdict as WV
from . import y_rules
from .ab_spec import SPEC as AB

GATES = WV.GATES
SIGNS = WV.SIGNS
STATS = ("D", "Dfin", "R")
METHODS = ("TS", "CG")
FLOW = dict(D="pool", Dfin="pool_drop", R="pool_raw")
PASS, FAIL, UNDECIDED = "PASS", "FAIL", "UNDECIDED"
STOP_MACHINE, STOP_PROTOCOL = "STOP_MACHINE", "STOP_PROTOCOL"
CAUSE_FAIL_UNCAL = "FAIL 쪽 보정 미도달"


def hedges(s=AB) -> float:
    return AE.hedges_j(s.probes)


def delta(s=AB) -> float:
    """AB.5: 1 / (J(8) × c(7)) — the cell-(1) D truth sits on the Hedges bar 1."""
    return 1.0 / (AE.hedges_j(s.probes) * AE.c_k(s.probes))


def _r(x: float, s) -> float:
    return round(float(x), s.round_digits)


# ================================================================ one pair (AB.4 세 통계)
def raw4(d: dict, z: dict) -> np.ndarray:
    """[F, 4] probe-mean contrasts of the four gates (z units), GATES order; columns 2–3 = aa_estimate.raw_contrasts."""
    r1, r2, n1 = WV.dv(d["R1"], z), WV.dv(d["R2"], z), WV.dv(d["N1"], z)
    rn1, rn2 = WV.dv(d["RN1"], z), WV.dv(d["RN2"], z)
    return np.stack([r1.mean(-1), (r2 - r1).mean(-1), (r1 - n1).mean(-1), ((r2 - r1) - (rn2 - rn1)).mean(-1)], -1)


def pair_stats(d: dict, z: dict, s=AB) -> dict:
    """{D, Dfin, R, raw}: [F, 4] each. A NaN fly d′ raises ValueError (AB.4 판정 1: STOP_MACHINE upstream)."""
    a = AE.pair_arrays(d, z, s)
    return dict(D=a["w"], Dfin=np.where(np.isinf(a["raw"]), np.nan, a["w"]), R=raw4(d, z), raw=a["raw"])


def stats_from_dprime(dp: np.ndarray, contrast_mean: np.ndarray, s=AB) -> dict:
    """The same three statistics from fly d′ [..., F] (and the fly probe-mean contrast) — calibration's path."""
    D = AE.winsorize(dp, s)
    return dict(D=D, Dfin=np.where(np.isinf(dp), np.nan, D), R=np.asarray(contrast_mean, float))


# ================================================================ structure
def structure(keys: list, tsets: list) -> tuple:
    """((X group, type-set group), …) per pair, in keys order; X groups = y_rules.merge_groups (first appearance),
    type-set groups numbered by first appearance of the type-set string (plan Reading 3)."""
    gx = [0] * len(keys)
    for j, g in enumerate(y_rules.merge_groups(list(keys))):
        for i in g:
            gx[i] = j
    order = list(dict.fromkeys(tsets))
    return tuple((int(gx[i]), int(order.index(t))) for i, t in enumerate(tsets))


def check_structure(st) -> tuple:
    if not isinstance(st, tuple) or not st or not all(
            isinstance(p, tuple) and len(p) == 2 and all(type(v) is int and v >= 0 for v in p) for p in st):
        raise TypeError("calibrate takes the structure only: a tuple of (X group, type-set group) int pairs (AB.5)")
    return st


def _labels(st) -> tuple:
    a = np.asarray(st, int)
    return a[:, 0], a[:, 1]


def groups_of(gx) -> list:
    out = {}
    for i, g in enumerate(np.asarray(gx).tolist()):
        out.setdefault(int(g), []).append(i)
    return list(out.values())


def rep_structure(sizes: tuple, s=AB) -> tuple:
    """AB.5 대표 구조: pairs in group order, type sets assigned round-robin (pair number mod 9; 해석 13)."""
    out, i = [], 0
    for j, n in enumerate(sizes):
        for _ in range(int(n)):
            out.append((j, i % s.n_type_sets))
            i += 1
    return tuple(out)


def n_groups(st) -> tuple:
    gx, gt = _labels(st)
    return len(set(gx.tolist())), len(set(gt.tolist()))


# ================================================================ the two interval methods (AB.5)
def ts_reps(X, groups, r, s=AB, B=None) -> np.ndarray:
    """[B, n] two-stage replicates (aa_estimate.boot_two_stage; X [P, F, n], NaN flies excluded)."""
    return AE.boot_two_stage(np.asarray(X, float), groups, r, s.boot_b if B is None else int(B), s.boot_chunk)


def ts_limits(reps, levels, s=AB) -> tuple:
    """(lo [L, n], hi [L, n]) = nanpercentile at 100·α and 100·(1 − α) (linear; = numpy.percentile without NaN)."""
    lv = np.asarray(levels, float)
    with warnings.catch_warnings():                       # an all-NaN column (a D_fin gate without finite flies)
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(reps, s.pct_scale * lv, axis=0).reshape(len(lv), -1),
                np.nanpercentile(reps, s.pct_scale * (1 - lv), axis=0).reshape(len(lv), -1))


def _vh(e: np.ndarray, lab: np.ndarray, k: int) -> tuple:
    _u, inv = np.unique(lab, return_inverse=True)
    G = int(inv.max()) + 1
    sums = np.bincount(inv, e, G)
    return (G / (G - 1)) * float((sums ** 2).sum()) / k ** 2 if G > 1 else float("nan"), G


def cg_parts(theta, gx, gt) -> dict:
    """AB.5 CG on the pairs with a finite θ̂: μ̂, V_X, V_T, V_XT, V*, G_X, G_T, df, k."""
    th = np.asarray(theta, float)
    keep = np.isfinite(th)
    th, gx, gt = th[keep], np.asarray(gx)[keep], np.asarray(gt)[keep]
    k = int(th.size)
    mu = float(th.mean()) if k else float("nan")
    e = th - mu
    vx, Gx = _vh(e, gx, k) if k else (float("nan"), 0)
    vt, Gt = _vh(e, gt, k) if k else (float("nan"), 0)
    cells = np.unique(np.stack([gx, gt], 1), axis=0, return_inverse=True)[1].reshape(-1) if k else gx
    vi, Gi = _vh(e, cells, k) if k else (float("nan"), 0)
    v = vx + vt - vi
    vs = max(v, vx, vt)
    return dict(mu=mu, V_X=vx, V_T=vt, V_XT=vi, V=v, V_star=vs, G_X=Gx, G_T=Gt, G_XT=Gi, df=min(Gx, Gt) - 1, k=k,
                psd_fix=bool(vs != v))


def cg_limits(theta, gx, gt, levels) -> tuple:
    """(lo [L], hi [L]) = μ̂ ∓ t_{1−α, df} √V* (scipy.stats.t)."""
    p = cg_parts(theta, gx, gt)
    lv = np.asarray(levels, float)
    if p["df"] < 1 or not np.isfinite(p["V_star"]):
        nan = np.full(lv.size, np.nan)
        return nan, nan
    h = stats.t.ppf(1 - lv, p["df"]) * math.sqrt(p["V_star"])
    return p["mu"] - h, p["mu"] + h


# ================================================================ the set (AB.4 · AB.5, judgement)
def set_limits(per_pair: dict, keys: list, tsets: list, levels: dict, s=AB, B=None, rngs=None) -> dict:
    """per_pair = {key: pair_stats(...)}; keys in c order; levels = {stat: [α, …]} (D carries α_P^D then α_F).
    rngs = {stat: Generator} overrides the judgement streams (boot_seed, FLOW[stat], "S1") — the synthetic validation
    only. Returns {stat: {point [4], levels, k_fin [4], gx_fin [4], gt_fin [4], ts: {lo, hi} [L, 4],
    cg: {lo, hi} [L, 4], _reps [B, 4]}}."""
    st = structure(keys, tsets)
    gx, gt = _labels(st)
    groups = y_rules.merge_groups(list(keys))
    out = {}
    for name in STATS:
        X = np.stack([np.asarray(per_pair[k][name], float) for k in keys])           # [P, F, 4]
        theta = AE._nanmean(X, 1)                                                    # [P, 4]
        lv = list(levels[name])
        r = rngs[name] if rngs is not None else AE.stream(s.boot_seed, FLOW[name], "S1")
        reps = ts_reps(X, groups, r, s, B)
        lo, hi = ts_limits(reps, lv, s)
        clo, chi = np.empty((len(lv), 4)), np.empty((len(lv), 4))
        kf, gxf, gtf = [], [], []
        for g in range(4):
            clo[:, g], chi[:, g] = cg_limits(theta[:, g], gx, gt, lv)
            fin = np.isfinite(theta[:, g])
            kf.append(int(fin.sum()))
            gxf.append(len(set(gx[fin].tolist())))
            gtf.append(len(set(gt[fin].tolist())))
        out[name] = dict(point=AE._nanmean(theta, 0).tolist(), levels=lv, k_fin=kf, gx_fin=gxf, gt_fin=gtf,
                         ts=dict(lo=lo.tolist(), hi=hi.tolist()), cg=dict(lo=clo.tolist(), hi=chi.tolist()),
                         _reps=reps)
    return out


def _bar(name: str, s) -> float:
    return s.raw_min if name == "R" else s.bar


def _scale(name: str, s) -> float:
    return 1.0 if name == "R" else hedges(s)


def beyond(v: float, sign: float, name: str, s) -> bool:
    """The expected-direction end beyond the bar: + → J·v > bar; − → J·v < −bar (rounded, strict)."""
    x = _scale(name, s) * float(v)
    return bool(np.isfinite(x)) and (_r(x - _bar(name, s), s) > 0 if sign > 0 else _r(x + _bar(name, s), s) < 0)


def short_of(v: float, sign: float, s) -> bool:
    """D's FAIL side: + → J·hi < bar; − → J·lo > −bar (rounded, strict)."""
    x = hedges(s) * float(v)
    return bool(np.isfinite(x)) and (_r(x - s.bar, s) < 0 if sign > 0 else _r(x + s.bar, s) > 0)


def mech_bad(per_pair_mech: dict, s=AB) -> list:
    """AB.4 판정 2: pairs whose fly median of either sign fraction (w_verdict.mech_fractions) is < 0.75."""
    out = []
    for k, fr in per_pair_mech.items():
        med = np.median(np.asarray(fr, float), axis=0)
        if any(_r(m - s.mech_min, s) < 0 for m in med):
            out.append(dict(key=k, medians=[float(m) for m in med]))
    return out


def verdict(lim: dict, machine: list, mech: dict, alpha: dict, s=AB) -> dict:
    """AB.4 판정 (1–5). lim = set_limits(...) with levels D = [α_P^D, α_F] (α_F may be absent → FAIL disabled),
    Dfin = [α_P^F], R = [α_P^R]; alpha = {"D": α_P^D, "Dfin": α_P^F, "R": α_P^R, "F": α_F or None}."""
    if machine:
        return dict(label=STOP_MACHINE, causes=list(machine))
    bad = mech_bad(mech, s)
    if bad:
        return dict(label=STOP_PROTOCOL, causes=bad)
    gates = {}
    fail, causes = [], []
    for gi, g in enumerate(GATES):
        sign = SIGNS[gi]
        row = {}
        for name in STATS:
            L = lim[name]
            a = alpha[name]
            j = L["levels"].index(a)
            end = "lo" if sign > 0 else "hi"
            ts_v, cg_v = L["ts"][end][j][gi], L["cg"][end][j][gi]
            ok_ts, ok_cg = beyond(ts_v, sign, name, s), beyond(cg_v, sign, name, s)
            fin_ok = True
            if name == "Dfin":
                fin_ok = (L["k_fin"][gi] >= s.k_min and L["gx_fin"][gi] >= s.g_min and L["gt_fin"][gi] >= s.g_min)
            row[name] = dict(TS=ts_v, CG=cg_v, ok_TS=ok_ts, ok_CG=ok_cg, fin_ok=fin_ok, point=L["point"][gi])
            if not fin_ok:
                causes.append(f"관문 {g}의 유한 마리판이 쌍 · 묶음 하한 미달({L['k_fin'][gi]}쌍, 묶음 "
                              f"{L['gx_fin'][gi]} · {L['gt_fin'][gi]})")
            elif ok_ts != ok_cg:
                causes.append(f"관문 {g}의 {name} 두 방식이 갈림(TS {'넘음' if ok_ts else '걸침'} · CG "
                              f"{'넘음' if ok_cg else '걸침'})")
            elif not ok_ts:
                causes.append(f"관문 {g}의 {name} 구간이 {'±0.25 z' if name == 'R' else '막대'}를 걸침")
        if alpha.get("F") is not None:
            L = lim["D"]
            j = L["levels"].index(alpha["F"])
            end = "hi" if sign > 0 else "lo"
            f_ts, f_cg = L["ts"][end][j][gi], L["cg"][end][j][gi]
            row["fail"] = dict(TS=f_ts, CG=f_cg, both=short_of(f_ts, sign, s) and short_of(f_cg, sign, s),
                               past_zero=all(np.isfinite(v) and _r(sign * v, s) < 0 for v in (f_ts, f_cg)))
            if row["fail"]["both"]:
                fail.append(g)
        gates[g] = row
    if fail:
        return dict(label=FAIL, causes=[], fail_gates=fail, gates=gates)
    ok = all(r[n]["ok_TS"] and r[n]["ok_CG"] and r[n]["fin_ok"] for r in gates.values() for n in STATS)
    if ok:
        return dict(label=PASS, causes=[], gates=gates)
    if alpha.get("F") is None:
        causes.append(CAUSE_FAIL_UNCAL)
    return dict(label=UNDECIDED, causes=causes, gates=gates)


# ================================================================ judgement helpers and records (AB.4 기록, AB.6)
def levels_for(alpha: dict) -> dict:
    """set_limits' levels from the 7b block's α: D = [α_P^D] (+ [α_F] when reached), D_fin = [α_P^F], R = [α_P^R]."""
    return dict(D=[alpha["D"]] + ([alpha["F"]] if alpha.get("F") is not None else []), Dfin=[alpha["Dfin"]],
                R=[alpha["R"]])


def fly_counts(per_pair: dict, keys: list, s=AB) -> dict:
    """Per gate over the 8k S1 flies: the share with expected-direction raw d′ ≥ 1, the ±10 (|d′| > 10, ±∞
    included) and ±∞ counts (AB.4 기록, AB.8 PASS 문장)."""
    R = np.stack([np.asarray(per_pair[k]["raw"], float) for k in keys])
    out = {}
    for gi, g in enumerate(GATES):
        x = R[..., gi]
        with np.errstate(invalid="ignore"):
            ge1 = np.round(SIGNS[gi] * x - 1.0, s.round_digits) >= 0
        out[g] = dict(share_ge1=float(ge1.mean()), n_clipped=int((np.abs(x) > s.winsor).sum()),
                      n_inf=int(np.isinf(x).sum()), n_flies=int(x.size))
    return out


def record_cis(per_pair: dict, keys: list, s=AB, B=None) -> dict:
    """Record-only fly-level and pair-level CIs of D (streams ("pool_fly", "S1"), ("pool_pair", "S1"); 2.5 / 97.5)."""
    X = np.stack([np.asarray(per_pair[k]["D"], float) for k in keys])
    groups = y_rules.merge_groups(list(keys))
    B = s.boot_b if B is None else int(B)
    out = {}
    for m, fn in (("fly", AE.boot_fly), ("pair", AE.boot_pair)):
        reps = fn(X, groups, AE.stream(s.boot_seed, f"pool_{m}", "S1"), B, s.boot_chunk)
        q = np.nanpercentile(reps, s.pct, axis=0)
        out[m] = {g: [float(q[0, i]), float(q[1, i])] for i, g in enumerate(GATES)}
    return out


def naive_record(pre_by_pair: dict, keys: list, z: dict, s=AB, B=None) -> dict:
    """S1 통합 순진 d′ (record, plan Reading 13): per fly the ±10-winsorized d′ of the naive pre ΔV
    (w_verdict.dprime over probes), pair = fly mean, point = pair mean, CI = two-stage 2.5 / 97.5 on ("naive", "S1")."""
    X = np.stack([AE.winsorize(WV.dprime(WV.dv(np.asarray(pre_by_pair[k], float), z)), s) for k in keys])[..., None]
    reps = AE.boot_two_stage(X, y_rules.merge_groups(list(keys)), AE.stream(s.boot_seed, "naive", "S1"),
                             s.boot_b if B is None else int(B), s.boot_chunk)
    q = np.nanpercentile(reps[:, 0], s.pct)
    return dict(point=float(X.mean(1)[:, 0].mean()), ci=[float(q[0]), float(q[1])])


def floor_subset(floors_by_pair: dict, keys: list, s=AB) -> list:
    """AB.6 바닥 민감도: S1 pairs with φ_R < 0.10 ∧ φ_P < 0.10 (records only, never the verdict)."""
    return [k for k in keys if _r(floors_by_pair[k]["phi_R"] - s.floor_cut, s) < 0
            and _r(floors_by_pair[k]["phi_P"] - s.floor_cut, s) < 0]
