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


def limits(X, groups, gx, gt, levels, r, s=AB, B=None) -> tuple:
    """The one code path of both interval methods, shared by the judgement (set_limits) and the calibration
    (rep_misses) — AB.5 "판정과 같은 코드 경로". X [P, F, n] fly values (NaN = excluded fly); groups = the X-label
    groups (index lists); gx, gt = per-pair labels; r = the stream (TS consumes it). Returns (θ̂ [P, n], TS reps
    [B, n], TS lo, hi [L, n], CG lo, hi [L, n])."""
    X = np.asarray(X, float)
    lv = list(levels)
    theta = AE._nanmean(X, 1)
    reps = ts_reps(X, groups, r, s, B)
    lo, hi = ts_limits(reps, lv, s)
    n = X.shape[-1]
    clo, chi = np.empty((len(lv), n)), np.empty((len(lv), n))
    for g in range(n):
        clo[:, g], chi[:, g] = cg_limits(theta[:, g], gx, gt, lv)
    return theta, reps, lo, hi, clo, chi


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
        lv = list(levels[name])
        r = rngs[name] if rngs is not None else AE.stream(s.boot_seed, FLOW[name], "S1")
        theta, reps, lo, hi, clo, chi = limits(X, groups, gx, gt, lv, r, s, B)
        kf, gxf, gtf = [], [], []
        for g in range(4):
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
    probe = fail_row_uncal(lim, alpha, s) if alpha.get("F") is None else None
    if probe is not None and probe["in_fail_row"]:
        causes.append(CAUSE_FAIL_UNCAL)
    return dict(label=UNDECIDED, causes=causes, gates=gates, fail_uncal=probe)


def fail_row_uncal(lim: dict, alpha: dict, s=AB) -> dict:
    """AB.4 판정 3 (FAIL side not calibrated in 7b): would the result fall in the FAIL row? Read at the largest
    FAIL-grid α among D's computed levels (levels_for(..., fail_probe=True) appends f_grid[0]): some gate whose D is
    short of the bar under both methods there. No FAIL-grid level computed → undeterminable, counted as in the row
    (the cause is disclosed rather than hidden). Never a FAIL: the label stays UNDECIDED."""
    L = lim["D"]
    have = [a for a in s.f_grid if a in L["levels"]]
    if not have:
        return dict(alpha=None, gates=None, in_fail_row=True)
    a = max(have)
    j = L["levels"].index(a)
    short = []
    for gi, g in enumerate(GATES):
        end = "hi" if SIGNS[gi] > 0 else "lo"
        if short_of(L["ts"][end][j][gi], SIGNS[gi], s) and short_of(L["cg"][end][j][gi], SIGNS[gi], s):
            short.append(g)
    return dict(alpha=a, gates=short, in_fail_row=bool(short))


# ================================================================ judgement helpers and records (AB.4 기록, AB.6)
def levels_for(alpha: dict, s=AB, fail_probe: bool = False) -> dict:
    """set_limits' levels from the 7b block's α: D = [α_P^D] (+ [α_F] when reached), D_fin = [α_P^F], R = [α_P^R].
    fail_probe (the judgement, order 9): when α_F was not reached, D also carries f_grid[0] so the verdict can tell
    whether the result falls in the FAIL row (AB.4 판정 3, cause "FAIL 쪽 보정 미도달"); the futility path (0f)
    leaves it off."""
    d = [alpha["D"]] + ([alpha["F"]] if alpha.get("F") is not None else [])
    if fail_probe and alpha.get("F") is None:
        d.append(s.f_grid[0])
    return dict(D=d, Dfin=[alpha["Dfin"]], R=[alpha["R"]])


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


# ================================================================ calibration (AB.5)
def _skew(zv, sd):
    return sd * (np.exp(zv) - math.exp(1 / 2)) / math.sqrt(math.e * (math.e - 1))


def effects(cell: tuple, gx, gt, r, s=AB) -> np.ndarray:
    """u_p of one replicate (structure-aware draws; draw order fixed by the cell kind)."""
    kind, a = cell
    P, Gx, Gt = len(gx), int(np.max(gx)) + 1, int(np.max(gt)) + 1
    if kind == "X":
        return r.normal(0.0, a, Gx)[gx]
    if kind == "T":
        return r.normal(0.0, a, Gt)[gt]
    if kind == "pair":
        return r.normal(0.0, a, P)
    if kind == "XT":
        return r.normal(0.0, a, Gx)[gx] + r.normal(0.0, a, Gt)[gt]
    if kind == "Xpair":
        return r.normal(0.0, a, Gx)[gx] + r.normal(0.0, a, P)
    if kind == "skX":
        return a * _skew(r.standard_normal(Gx), s.skew_sd)[gx]
    if kind == "skT":
        return a * _skew(r.standard_normal(Gt), s.skew_sd)[gt]
    return np.zeros(P)


def marginal_effects(cell: tuple, n: int, r, s=AB) -> np.ndarray:
    """u for n independent pairs from the cell's marginal (the truth's superpopulation, plan Reading 8)."""
    kind, a = cell
    if kind in ("X", "T", "pair"):
        return r.normal(0.0, a, n)
    if kind in ("XT", "Xpair"):
        return r.normal(0.0, a, n) + r.normal(0.0, a, n)
    if kind in ("skX", "skT"):
        return a * _skew(r.standard_normal(n), s.skew_sd)
    return np.zeros(n)


def probes(cell: tuple, u, r, dlt: float, s=AB) -> np.ndarray:
    """[P, F, K] probe contrasts = δ + u_p + m_{p,f} + N(0, 1); cell (23)/(24): flies 0–1 constant ±0.05."""
    u = np.asarray(u, float)
    P = u.size
    fe = r.standard_normal((P, s.flies, 1)) if cell[0] == "fly" else 0.0
    x = dlt + u[:, None, None] + fe + r.standard_normal((P, s.flies, s.probes))
    if cell[0] == "floor":
        x[:, :s.floor_flies, :] = s.floor_const * cell[1]
    return x


def synth_stats(x, s=AB) -> dict:
    return stats_from_dprime(WV.dprime(x), x.mean(-1), s)


def truth(st, tag: str, ci: int, s=AB, n=None) -> dict:
    """AB.5 참값: the mean pair statistic over n simulated pairs (stream ("cal", tag, "truth", ci))."""
    check_structure(st)
    n = s.truth_pairs if n is None else int(n)
    r = AE.stream(s.cal_seed, "cal", tag, "truth", int(ci))
    cell, dl = s.cell(ci), delta(s)
    acc, cnt, m = dict(D=0.0, Dfin=0.0, R=0.0), dict(D=0, Dfin=0, R=0), 0
    while m < n:
        c = min(s.truth_chunk, n - m)
        x = probes(cell, marginal_effects(cell, c, r, s), r, dl, s)
        st3 = synth_stats(x, s)
        for name in STATS:
            th = AE._nanmean(st3[name], 1)                      # a pair without a finite fly leaves D_fin's mean
            acc[name] += float(np.nansum(th))
            cnt[name] += int(np.isfinite(th).sum())
        m += c
    return {k: acc[k] / cnt[k] for k in STATS}


def rep_misses(st, cell, r, tru: dict, s=AB, B=None) -> dict:
    """One replicate: {stat: {P: joint [Lp], P_TS, P_CG, (D) F: joint [Lf], F_TS, F_CG}} booleans."""
    gx, gt = _labels(st)
    groups = groups_of(gx)
    x = probes(cell, effects(cell, gx, gt, r, s), r, delta(s), s)
    st3 = synth_stats(x, s)
    out = {}
    for name in STATS:
        lv = list(s.p_grid) + (list(s.f_grid) if name == "D" else [])
        _th, _reps, lo, hi, clo, chi = limits(st3[name][..., None], groups, gx, gt, lv, r, s, B)
        t = tru[name]
        up_ts, up_cg = lo[:, 0] > t, clo[:, 0] > t
        dn_ts, dn_cg = hi[:, 0] < t, chi[:, 0] < t
        n_p = len(s.p_grid)
        row = dict(P=(up_ts & up_cg)[:n_p], P_TS=up_ts[:n_p], P_CG=up_cg[:n_p])
        if name == "D":
            row.update(F=(dn_ts & dn_cg)[n_p:], F_TS=dn_ts[n_p:], F_CG=dn_cg[n_p:])
        out[name] = row
    return out


def _zero_counts(s) -> dict:
    out = {}
    for name in STATS:
        out[name] = {k: [0] * len(s.p_grid) for k in ("P", "P_TS", "P_CG")}
        if name == "D":
            out[name].update({k: [0] * len(s.f_grid) for k in ("F", "F_TS", "F_CG")})
    return out


def _gen_from(state: dict | None, s, tag, phase, ci):
    r = AE.stream(s.cal_seed, "cal", tag, phase, int(ci))
    if state is not None:
        r.bit_generator.state = state
    return r


def cal_run(st, tag: str, phase: str, ci: int, tru: dict, s=AB, n=None, resume=None, on_chunk=None,
            B=None, only=None) -> dict:
    """Reps of one (structure tag, phase, cell) on stream ("cal", tag, phase, ci), in chunks of cal_chunk. resume =
    a previous payload {done, counts, state}; on_chunk(payload) is called after every chunk (the runner's atomic
    checkpoint). only = {stat: {side: α}} (verification): only those levels are counted, so the verification stream
    never informs the selection (AB.5 (나)). Returns the final payload. Bit-identical whether run at once or
    resumed."""
    check_structure(st)
    if phase not in ("sel", "ver"):
        raise ValueError(phase)
    n = (s.n_sel if phase == "sel" else s.n_ver) if n is None else int(n)
    if resume is not None and int(resume.get("n", -1)) != n:
        raise ValueError(f"resume payload was for n = {resume.get('n')}, not the requested n = {n}")
    pay = dict(done=0, n=n, counts=_zero_counts(s), state=None) if resume is None else copy.deepcopy(resume)
    r = _gen_from(pay["state"], s, tag, phase, ci)
    cell = s.cell(ci)
    while pay["done"] < n:
        m = min(s.cal_chunk, n - pay["done"])
        for _ in range(m):
            got = rep_misses(st, cell, r, tru, s, B)
            for name, row in got.items():
                for k, v in row.items():
                    keep = None
                    if only is not None:
                        a = (only.get(name) or {}).get(k[0])
                        keep = None if a is None else _grid(k[0], s).index(a)
                        if keep is None:
                            continue
                    c = pay["counts"][name][k]
                    for i, b in enumerate(np.asarray(v, bool).tolist()):
                        if keep is None or i == keep:
                            c[i] += int(b)
        pay = dict(pay, done=pay["done"] + m, state=r.bit_generator.state)
        if on_chunk is not None:
            on_chunk(copy.deepcopy(pay))
    return pay


def cp_upper(x: int, n: int, s=AB) -> float:
    """Clopper–Pearson one-sided upper bound at cp_level (scipy.stats.beta); x = n → 1."""
    return 1.0 if x >= n else float(stats.beta.ppf(s.cp_level, x + 1, n - x))


def _grid(side: str, s) -> tuple:
    return s.p_grid if side == "P" else s.f_grid


def _target(side: str, s) -> float:
    return s.p_target if side == "P" else s.f_target


def select(counts: dict, n: int, s=AB) -> dict:
    """(가) per stat and side: the largest α (grids run large → small) whose CP upper bound ≤ target in every cell.
    counts = {ci: payload counts}. Returns {stat: {P: α | None, (D) F: α | None}} + the per-cell bounds."""
    out = {}
    for name in STATS:
        out[name] = {}
        for side in (("P", "F") if name == "D" else ("P",)):
            pick = None
            for i, a in enumerate(_grid(side, s)):
                if all(_r(cp_upper(c[name][side][i], n, s) - _target(side, s), s) <= 0 for c in counts.values()):
                    pick = a
                    break
            out[name][side] = pick
    return out


def verify(counts: dict, chosen: dict, n: int, s=AB) -> dict:
    """{stat: {side: {alpha, ok, worst: {cell, x, cp}}}} at the chosen α only (None = not reached)."""
    out = {}
    for name, sides in chosen.items():
        out[name] = {}
        for side, a in sides.items():
            if a is None:
                out[name][side] = dict(alpha=None, ok=False, worst=None)
                continue
            i = _grid(side, s).index(a)
            cps = {ci: cp_upper(c[name][side][i], n, s) for ci, c in counts.items()}
            w = max(cps, key=lambda k: cps[k])
            out[name][side] = dict(alpha=a, ok=all(_r(v - _target(side, s), s) <= 0 for v in cps.values()),
                                   worst=dict(cell=int(w), x=int(counts[w][name][side][i]), cp=float(cps[w])))
    return out


def _serial_cell(kind, args):
    """The default cell executor (no checkpoint): kind "truth" → truth(*args), "sel"/"ver" → cal_run(*args)."""
    if kind == "truth":
        return truth(*args)
    return cal_run(*args)


def calibrate_many(items: list, s=AB, cell=None) -> list:
    """AB.5 선택과 검증 on several structures at once: items = [(structure, tag)] (structures only — anything else
    TypeError). cell(kind, args) runs one job (the runner passes a checkpointing, pooled executor with .many); each
    phase runs across every item together: 24 truths per item, then 24 selection cells, then 24 verification cells
    (only at the chosen α). Returns one summary per item, in order."""
    if not isinstance(items, list) or not items:
        raise TypeError("calibrate_many takes a non-empty list of (structure, tag)")
    for st, tag in items:
        check_structure(st)
        if not isinstance(tag, str):
            raise TypeError("the structure tag is a string")
    run = cell or _serial_cell
    cis = list(range(1, len(s.cells) + 1))
    idx = [(i, ci) for i in range(len(items)) for ci in cis]
    got = run_many(run, [("truth", (items[i][0], items[i][1], ci, s)) for i, ci in idx])
    tr = [{} for _ in items]
    for (i, ci), v in zip(idx, got):
        tr[i][ci] = v
    got = run_many(run, [("sel", (items[i][0], items[i][1], "sel", ci, tr[i][ci], s)) for i, ci in idx])
    sel = [{} for _ in items]
    for (i, ci), v in zip(idx, got):
        sel[i][ci] = v["counts"]
    chosen = [select(sel[i], s.n_sel, s) for i in range(len(items))]
    only = [{n: {sd: a for sd, a in v.items() if a is not None} for n, v in ch.items()} for ch in chosen]
    need = [i for i in range(len(items)) if any(a is not None for sd in chosen[i].values() for a in sd.values())]
    vidx = [(i, ci) for i in need for ci in cis]
    got = run_many(run, [("ver", (items[i][0], items[i][1], "ver", ci, tr[i][ci], s, None, None, None, None, only[i]))
                         for i, ci in vidx])
    ver = [{} for _ in items]
    for (i, ci), v in zip(vidx, got):
        ver[i][ci] = v["counts"]
    out = []
    for i, (st, tag) in enumerate(items):
        checked = verify(ver[i], chosen[i], s.n_ver, s) if i in need else {
            n: {sd: dict(alpha=None, ok=False, worst=None) for sd in v} for n, v in chosen[i].items()}
        alpha = {n: {sd: (v["alpha"] if v["ok"] else None) for sd, v in sides.items()} for n, sides in checked.items()}
        sel_cp = {n: {sd: {ci: [cp_upper(x, s.n_sel, s) for x in sel[i][ci][n][sd]] for ci in cis}
                      for sd in (("P", "F") if n == "D" else ("P",))} for n in STATS}
        out.append(dict(tag=tag, structure=[list(p) for p in st], k=len(st), groups=list(n_groups(st)), truth=tr[i],
                        sel_counts=sel[i], ver_counts=ver[i], chosen=chosen[i], verified=checked, alpha=alpha,
                        pass_ok=all(alpha[n]["P"] is not None for n in STATS), fail_ok=alpha["D"]["F"] is not None,
                        sel_cp=sel_cp, n_sel=s.n_sel, n_ver=s.n_ver))
    return out


def calibrate(st, tag: str, s=AB, cell=None) -> dict:
    """AB.5 on one structure (tuple of (X group, type-set group) int pairs only; anything else TypeError)."""
    check_structure(st)
    return calibrate_many([(st, tag)], s, cell)[0]


def run_many(run, jobs: list) -> list:
    """Jobs through the executor: a callable with .many(jobs) runs them together (the runner's pool), else one by
    one."""
    many = getattr(run, "many", None)
    return many(jobs) if many is not None else [run(k, a) for k, a in jobs]


# ================================================================ bench and the synthetic validation (AB.7 0, AB.5)
def bench(s=AB) -> dict:
    """AB.7 0 비용 측정: g 7 cells (17) · (23), bench_reps reps each, B = boot_b, seconds per rep (stream ("bench",
    cell) under cal_seed; never a calibration count)."""
    st = rep_structure(s.rep_sizes("g7"), s)
    out = {}
    for ci in s.bench_cells:
        r = AE.stream(s.cal_seed, "bench", int(ci))
        tru = dict(D=1.0, Dfin=1.0, R=1.0)
        t = time.perf_counter()
        for _ in range(s.bench_reps):
            rep_misses(st, s.cell(ci), r, tru, s)
        out[str(ci)] = (time.perf_counter() - t) / s.bench_reps
    return dict(per_rep_s=out, max_s=max(out.values()), structure="g7", B=s.boot_b, reps=s.bench_reps)


def synth_configs(s=AB) -> list:
    """(name, per-gate effect in the expected direction) — AB.5 합성 검증 (1)–(3); (1) puts reward_level at the bar
    (by the mirror, any gate gives the same distribution — plan Reading 10)."""
    d0 = delta(s)
    cfg = [("bar_one", (d0,) + (s.synth_other,) * 3), ("bar_all", (d0,) * 4)]
    cfg += [(f"eff{e}", (float(e),) * 4) for e in s.synth_effects]
    return cfg


def synth_rep(st, cell, eff: tuple, alpha: dict, r, s=AB, B=None) -> str:
    """One synthetic replicate of the whole verdict: four independent gates (sign × (effect + u + noise)), then
    set_limits on stream r and verdict with no machine / mechanism problems; returns the label."""
    gx, gt = _labels(st)
    keys = [f"a|{i}|X{int(gx[i])}|Y" for i in range(len(gx))]
    tsets = [f"T{int(gt[i])}" for i in range(len(gx))]
    per = {k: dict(D=np.empty((s.flies, 4)), Dfin=np.empty((s.flies, 4)), R=np.empty((s.flies, 4))) for k in keys}
    for g in range(4):
        x = SIGNS[g] * probes(cell, effects(cell, gx, gt, r, s), r, float(eff[g]), s)
        st3 = synth_stats(x, s)
        for i, k in enumerate(keys):
            for name in STATS:
                per[k][name][:, g] = st3[name][i]
    lv = dict(D=[alpha["D"]] + ([alpha["F"]] if alpha.get("F") is not None else []), Dfin=[alpha["Dfin"]],
              R=[alpha["R"]])
    lim = set_limits(per, keys, tsets, lv, s, B, rngs={name: r for name in STATS})
    return verdict(lim, [], {}, alpha, s)["label"]


# ================================================================ the futility gate (AB.7 0f, AB.9.3)
from . import w_records  # noqa: E402  (part C only: the AA source goes through the judgement path)

RECORDS_REASON = "records 상한"                       # = ab_rules.RECORDS_REASON (AB.7 예산; ab_estimate imports no rules)


def futility_inputs(by_pair: dict, keys: list, z: dict, s=AB) -> dict:
    """AB.7 0f AA 원천: by_pair = {pair key: [{unit: {fly, brain}, result}]} (AA S1's 16 pairs × 8 flies × R · N · RN,
    read from AA's learn cache by the runner); z = Y pilot's full-precision z_V (AA's). Per pair, through the judgement
    path (w_records.pair_data → pair_stats): fly d′ [F, 4] (unclipped, ±∞ kept) and the four raw contrasts [F, 4].
    JSON-able lists (the cell executor keys and pickles them)."""
    dp, raw = [], []
    for k in keys:
        st = pair_stats(w_records.pair_data(by_pair[k], list(range(s.flies))), z, s)
        dp.append(np.asarray(st["raw"], float).tolist())
        raw.append(np.asarray(st["R"], float).tolist())
    return dict(keys=list(keys), dp=dp, raw=raw)


def _col_mean(x) -> float:
    return float(np.ascontiguousarray(x, float).mean())


def fut_check(arr: dict, pairs: dict, s1: dict, s=AB) -> list:
    """AB.7 0f: the source must reproduce AA bit for bit — the winsorized fly mean of each gate = records.pairs[key]
    .gates[g].mean, the two associations' fly contrasts = raw[g].fly; and (rounded 9) the pair means' mean = s1_records
    .point / raw.point, the D pair SD (ddof 1) = sd_pairs. A difference is a sealed-code defect (restart_preseal)."""
    why = []
    if list(arr["keys"]) != list(s1["keys"]) or len(arr["keys"]) != s.fut_src_pairs:
        why.append(f"원천 쌍 {len(arr['keys'])}개 · 순서 ≠ AA s1_records.keys")
        return why
    D = AE.winsorize(np.asarray(arr["dp"], float), s)
    R = np.asarray(arr["raw"], float)
    for i, k in enumerate(arr["keys"]):
        for gi, g in enumerate(GATES):
            if _col_mean(D[i, :, gi]) != pairs[k]["gates"][g]["mean"]:
                why.append(f"{k} {g}: 잘라낸 마리 평균 ≠ AA records")
        for g in AE.RAW:
            if R[i, :, GATES.index(g)].tolist() != list(pairs[k]["raw"][g]["fly"]):
                why.append(f"{k} {g}: 마리 비표준화 대조 ≠ AA records")
    mD = D.mean(1)
    for gi, g in enumerate(GATES):
        if _r(mD[:, gi].mean() - s1["point"][g], s) != 0:
            why.append(f"{g}: 쌍 추정 평균 ≠ s1_records.point")
        if _r(np.std(mD[:, gi], ddof=1) - s1["sd_pairs"][g], s) != 0:
            why.append(f"{g}: 쌍 추정 SD ≠ s1_records.sd_pairs")
    for g in AE.RAW:
        if _r(R.mean(1)[:, GATES.index(g)].mean() - s1["raw"]["point"][g], s) != 0:
            why.append(f"{g}: 비표준화 대조 평균 ≠ s1_records.raw.point")
    return why


def _icc(y: np.ndarray, groups: list) -> float:
    """One-way random-effects moment estimate (unbalanced n₀): clamp(σ̂²_b / (σ̂²_b + MSW), 0, 1)."""
    N, G = y.size, len(groups)
    n = np.array([len(g) for g in groups], float)
    gm = np.array([y[g].mean() for g in groups])
    msb = float((n * (gm - y.mean()) ** 2).sum()) / (G - 1)
    msw = float(sum(((y[g] - gm[j]) ** 2).sum() for j, g in enumerate(groups))) / (N - G)
    n0 = (N - float((n ** 2).sum()) / N) / (G - 1)
    s2b = max(0.0, (msb - msw) / n0)
    return float(min(1.0, s2b / (s2b + msw))) if s2b + msw > 0 else 0.0


def fut_model(arr: dict, s1: dict, s=AB) -> dict:
    """AB.7 0f model from the source: μ [8] (D pair means' mean on the uncorrected K 8 scale, then R), τ [8] (D: AA's
    DL τ̂ where available, else the S1 pair-estimate SD — s1_records.sd_pairs; R: the pair-mean SD, ddof 1), its
    source labels, ρ [8] (one-way MoM over AA's X-label groups), the 8 × 8 Pearson C of the pair means and its
    Cholesky factor (C + eps·I). Returns the source arrays too (dp, raw, mD, mR) — the simulation's only input."""
    D = AE.winsorize(np.asarray(arr["dp"], float), s)
    R = np.asarray(arr["raw"], float)
    mD, mR = D.mean(1), R.mean(1)
    M = np.concatenate([mD, mR], 1)                                                    # [P, 8]
    tau, src = [], []
    for gi, g in enumerate(GATES):
        dl = s1["dl"][g]
        if dl.get("available"):
            tau.append(float(dl["tau"]))
            src.append("dl")
        else:
            tau.append(float(s1["sd_pairs"][g]))
            src.append("sd")
    tau += [float(v) for v in np.std(mR, axis=0, ddof=1)]
    src += ["sd"] * 4
    groups = y_rules.merge_groups(list(arr["keys"]))
    rho = [_icc(M[:, j], groups) for j in range(M.shape[1])]
    C = np.corrcoef(M.T)
    L = np.linalg.cholesky(C + s.fut_chol_eps * np.eye(C.shape[0]))
    return dict(keys=list(arr["keys"]), dp=arr["dp"], raw=arr["raw"], mD=mD.tolist(), mR=mR.tolist(),
                mu=M.mean(0).tolist(), tau=tau, tau_src=src, rho=rho, C=C.tolist(), chol=L.tolist(),
                x_sizes=[len(g) for g in groups])


def fut_rho(inp: dict, alloc: str) -> np.ndarray:
    rho = np.asarray(inp["rho"], float)
    if alloc == "icc":
        return rho
    if alloc == "pair":
        return np.zeros_like(rho)
    if alloc == "group":
        return np.ones_like(rho)
    raise ValueError(alloc)


def fut_alpha(alpha: dict) -> dict:
    """The 0e levels of one structure (cal_gate alpha {D {P, F}, Dfin {P}, R {P}}) → verdict's α, FAIL side off."""
    return dict(D=alpha["D"]["P"], Dfin=alpha["Dfin"]["P"], R=alpha["R"]["P"], F=None)


def fut_draw(st, inp: dict, alloc: str, r, s=AB) -> tuple:
    """Steps (1)–(2) of one replicate, in this draw order: a [G, 8], e [k, 8] (Cholesky of C × N(0, I₈), scaled by
    τ√ρ and τ√(1 − ρ)), then the source pair q [k] and the flies f [k, F] (with replacement). θ = μ + a_X(p) + e_p."""
    gx, _gt = _labels(st)
    k, G = len(gx), int(gx.max()) + 1
    L = np.asarray(inp["chol"], float)
    tau, rho = np.asarray(inp["tau"], float), fut_rho(inp, alloc)
    nd = L.shape[0]
    a = (r.standard_normal((G, nd)) @ L.T) * (tau * np.sqrt(rho))
    e = (r.standard_normal((k, nd)) @ L.T) * (tau * np.sqrt(1.0 - rho))
    th = np.asarray(inp["mu"], float) + a[gx] + e                                      # [k, 8]
    q = r.integers(0, len(inp["dp"]), k)
    f = r.integers(0, s.flies, (k, s.flies))
    return th, q, f


def fut_flies(th, q, f, inp: dict, s=AB) -> tuple:
    """Step (3): the fly values of the three statistics [k, F, 4] — D: θ + (clip(source, ±10) − the source pair's
    winsorized mean), a ±∞ source fly kept (+10 / −10 in D, NaN in D_fin); R: θ^R + (source contrast − source pair
    mean)."""
    dp, raw = np.asarray(inp["dp"], float), np.asarray(inp["raw"], float)
    mD, mR = np.asarray(inp["mD"], float), np.asarray(inp["mR"], float)
    src = dp[q[:, None], f]                                                            # [k, F, 4]
    inf = np.isinf(src)
    new = np.where(inf, src, th[:, None, :4] + (AE.winsorize(src, s) - mD[q][:, None, :]))
    D = AE.winsorize(new, s)
    return D, np.where(inf, np.nan, D), th[:, None, 4:] + (raw[q[:, None], f] - mR[q][:, None, :])


def fut_rep(st, inp: dict, alpha: dict, alloc: str, r, s=AB, B=None) -> tuple:
    """One replicate (AB.7 0f 모의 한 회): (1) group and pair effects (Cholesky × N(0, I₈) ⊙ τ√ρ, ⊙ τ√(1 − ρ)), θ_p =
    μ + a_X(p) + e_p; (2) per pair a source pair q (uniform over the source) and 8 of its flies with replacement;
    (3) fly values — D: θ + (clip(source, ±10) − the source pair's winsorized mean), a ±∞ source fly kept as is; R:
    θ^R + (source contrast − source pair mean); (4) set_limits + verdict on stream r (D, D_fin, R in that order) at
    the structure's 0e levels, no machine / mechanism rows, FAIL off. Returns (PASS?, the verdict's gate rows)."""
    gx, gt = _labels(st)
    th, q, f = fut_draw(st, inp, alloc, r, s)
    D, Dfin, Rf = fut_flies(th, q, f, inp, s)
    keys = [f"a|{i}|X{int(gx[i])}|Y" for i in range(len(gx))]
    tsets = [f"T{int(gt[i])}" for i in range(len(gx))]
    per = {kk: dict(D=D[i], Dfin=Dfin[i], R=Rf[i]) for i, kk in enumerate(keys)}
    al = fut_alpha(alpha)
    lim = set_limits(per, keys, tsets, levels_for(al), s, B, rngs={name: r for name in STATS})
    v = verdict(lim, [], {}, al, s)
    return v["label"] == PASS, v["gates"]


def _blocked_zero() -> dict:
    return {g: {n: {"TS": 0, "CG": 0} for n in STATS} for g in GATES}


def _fut_gen(state, s, stream):
    r = AE.stream(s.synth_seed, *stream)
    if state is not None:
        r.bit_generator.state = state
    return r


def fut_run(st, tag: str, alloc: str, inp: dict, alpha: dict, s=AB, n=None, resume=None, on_chunk=None, B=None,
            stream=None) -> dict:
    """n replicates on stream ("fut", tag, alloc) (or `stream`, the grid's ("fut", "grid", scen, g, k)) under the
    synth root 88_030_000, in chunks of cal_chunk with the PCG64 state checkpointed (cal_run's form, bit-identical when
    resumed). Payload {done, n, passed, blocked {gate: {stat: {TS, CG}}}, state}: blocked counts a replicate where that
    one condition (gate × statistic × method; a D_fin pair / group floor blocks both methods) did not clear its bar."""
    check_structure(st)
    n = s.fut_reps if n is None else int(n)
    stream = ("fut", tag, alloc) if stream is None else tuple(stream)
    pay = dict(done=0, n=n, passed=0, blocked=_blocked_zero(), state=None) if resume is None else copy.deepcopy(resume)
    r = _fut_gen(pay["state"], s, stream)
    while pay["done"] < n:
        m = min(s.cal_chunk, n - pay["done"])
        for _ in range(m):
            ok, gates = fut_rep(st, inp, alpha, alloc, r, s, B)
            pay["passed"] += int(ok)
            for g, row in gates.items():
                for name in STATS:
                    c = row[name]
                    for meth in ("TS", "CG"):
                        pay["blocked"][g][name][meth] += int(not (c[f"ok_{meth}"] and c["fin_ok"]))
        pay = dict(pay, done=pay["done"] + m, state=r.bit_generator.state)
        if on_chunk is not None:
            on_chunk(copy.deepcopy(pay))
    return pay


def cp_two(x: int, n: int, s=AB) -> list:
    """Clopper–Pearson two-sided interval at fut_ci_tails (record only; the decision is the point P̂)."""
    lo_t, hi_t = s.fut_ci_tails
    lo = 0.0 if x <= 0 else float(stats.beta.ppf(lo_t, x, n - x + 1))
    hi = 1.0 if x >= n else float(stats.beta.ppf(hi_t, x + 1, n - x))
    return [lo, hi]


def fut_row(pay: dict, s=AB) -> dict:
    n = int(pay["n"])
    x = int(pay["passed"])
    share = {g: {nm: {m: pay["blocked"][g][nm][m] / n for m in ("TS", "CG")} for nm in STATS} for g in GATES}
    worst = max(((g, nm, m) for g in GATES for nm in STATS for m in ("TS", "CG")),
                key=lambda t: share[t[0]][t[1]][t[2]])
    return dict(passed=x, n=n, p_hat=x / n, cp95=cp_two(x, n, s), blocked=share,
                worst=dict(gate=worst[0], stat=worst[1], method=worst[2], share=share[worst[0]][worst[1]][worst[2]]))


def _serial_fut(kind, args):
    if kind != "fut":
        raise ValueError(kind)
    return fut_run(*args)


def futility(inp: dict, alpha_by_tag: dict, s=AB, cell=None) -> dict:
    """AB.7 0f 가망 관문: 9 rows (fut_tags × fut_allocs; structure = rep_structure of the 0e representative sizes, levels
    = 0e's verified α of that structure), fut_reps replicates each, through cell(kind, args) (run_many). The decision
    row is fut_judge = (g6, icc): P̂ = PASS / fut_reps, rounded 9, P̂ < fut_threshold → futile. Takes the AA-source
    model and 0e's levels only — never a Gen-2 value."""
    run = cell or _serial_fut
    rows = [(t, a) for t in s.fut_tags for a in s.fut_allocs]
    jobs = [("fut", (rep_structure(s.rep_sizes(t), s), t, a, inp, alpha_by_tag[t], s)) for t, a in rows]
    got = run_many(run, jobs)
    out = {f"{t}|{a}": dict(fut_row(p, s), tag=t, alloc=a, stream=["fut", t, a],
                            alpha=fut_alpha(alpha_by_tag[t])) for (t, a), p in zip(rows, got)}
    j = out["|".join(s.fut_judge)]
    futile = _r(j["p_hat"] - s.fut_threshold, s) < 0
    return dict(rows=out, judge=dict(row="|".join(s.fut_judge), p_hat=j["p_hat"], passed=j["passed"], n=j["n"],
                                     threshold=s.fut_threshold, futile=bool(futile)))


# ---------------------------------------------------------------- the record-only grid (STOP_FUTILE only)
def even_sizes(k: int, g: int) -> tuple:
    """k pairs in g groups as evenly as possible; the first k mod g groups get one more."""
    q, m = divmod(int(k), int(g))
    return tuple(q + 1 if j < m else q for j in range(int(g)))


def grid_cells(s=AB) -> list:
    return [(g, k) for g in s.fut_grid_g for k in s.fut_grid_k if k >= g]


def nearest_tag(g: int, s=AB) -> str:
    """g ≤ 5 → g5, g = 6 → g6, g ≥ 7 → g7: the representative structure whose stored 0e selection counts are used."""
    sizes = {t: len(s.rep_sizes(t)) for t in s.fut_tags}
    lo, hi = min(sizes.values()), max(sizes.values())
    gg = min(max(int(g), lo), hi)
    return next(t for t, n in sizes.items() if n == gg)


def scenario_cells(tag: str, s=AB) -> list:
    out = dict(s.fut_scenarios)[tag]
    return [ci for ci in range(1, len(s.cells) + 1) if ci not in out]


def reselect(sel_counts: dict, cells: list, s=AB) -> dict | None:
    """AB.5 (가) on the scenario's cells only, PASS side, from 0e's stored selection counts ({ci: counts}; JSON keys may
    be strings). Returns the fut_alpha form, or None when some statistic reaches no α."""
    sub = {int(ci): c for ci, c in sel_counts.items() if int(ci) in cells}
    ch = select(sub, s.n_sel, s)
    if any(ch[n]["P"] is None for n in STATS):
        return None
    return {n: {"P": ch[n]["P"]} for n in STATS}


def pareto_min(points: list) -> list:
    """The (g, k) cells no other listed cell dominates (g′ ≤ g ∧ k′ ≤ k, not equal) — sorted."""
    pts = sorted(set((int(g), int(k)) for g, k in points))
    return [p for p in pts if not any(q != p and q[0] <= p[0] and q[1] <= p[1] for q in pts)]


def futility_grid(inp: dict, sel_by_tag: dict, s=AB, cell=None, stop=None) -> dict:
    """AB.7 0f 기록 격자 (STOP_FUTILE only; record, never changes AB): per scenario set, every (g, k) of grid_cells,
    even group sizes (rep_structure: type sets by pair number mod 9), the α reselected from the nearest
    representative structure's stored 0e selection counts on the scenario's cells, fut_grid_reps replicates on stream
    ("fut", "grid", scen, g, k) with the fut_grid_alloc allocation. One run_many per scenario; stop() is checked
    before each scenario (the records cap) — past it the remaining cells are null with RECORDS_REASON. Returns
    {scen: {cells: [...], pareto: [[g, k], …]}}."""
    run = cell or _serial_fut
    out = {}
    for scen, _ex in s.fut_scenarios:
        cis = scenario_cells(scen, s)
        cells, jobs, idx = [], [], []
        for g, k in grid_cells(s):
            tag = nearest_tag(g, s)
            al = reselect(sel_by_tag[tag], cis, s)
            row = dict(g=g, k=k, sizes=list(even_sizes(k, g)), alpha_from=tag,
                       alpha=None if al is None else fut_alpha(al))
            if al is not None:
                idx.append(len(cells))
                jobs.append(("fut", (rep_structure(even_sizes(k, g), s), tag, s.fut_grid_alloc, inp, al, s,
                                     s.fut_grid_reps, None, None, None, ["fut", "grid", scen, g, k])))
            else:
                row.update(p_hat=None, reason="α 미도달")
            cells.append(row)
        if stop is not None and stop():
            for row in cells:
                row.update(p_hat=None, reason=RECORDS_REASON)
            out[scen] = dict(cells=cells, pareto=None, reason=RECORDS_REASON)
            continue
        for i, p in zip(idx, run_many(run, jobs)):
            r_ = fut_row(p, s)
            cells[i].update(passed=r_["passed"], n=r_["n"], p_hat=r_["p_hat"], cp95=r_["cp95"])
        ok = [(c["g"], c["k"]) for c in cells if c.get("p_hat") is not None
              and _r(c["p_hat"] - s.fut_threshold, s) >= 0]
        out[scen] = dict(cells=cells, pareto=[list(p) for p in pareto_min(ok)])
    return out
