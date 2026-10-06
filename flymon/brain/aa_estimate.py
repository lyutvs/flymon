"""AA's estimators (AA.4 6199–6214, AA.5 6218–6233, AA.6 6239–6240, AA.9 6282–6284, AA.9.1 6304–6305): pure functions,
sealed before the naive screen (AA.7 3). A pair's raw counts are {stage: int [F, K, 2, 2]} (w_verdict's layout).
- Fly statistic = w_verdict.gate_stats (four gates, GATES order); winsorized to [−winsor, winsor] (±∞ included); the
  pair estimate = the mean of the winsorized fly d′ (sign kept); F < min_flies → null; a NaN fly is a measurement
  defect (ValueError → STOP_MACHINE upstream). Records: the median, ±∞ / clipped counts, the ±∞-dropped mean, the t
  interval, the raw (unstandardised) contrasts, the floor shares (φ_R = R1 MBON05(X) = 0, φ_P = R2 MBON13(X) = 0;
  floor_both = pilot_record's "both").
- Resampling (plan Readings 4, 5): flies are resampled, never probes; the four gates (and the raw contrasts of a pair CI)
  share one index draw; draws come in chunks of boot_chunk — two-stage: groups (n, G) then flies (n, G, M, F);
  fly-level: (n, P, F); pair-level: pairs (n, P) then flies (n, P, F). NaN (a dropped ±∞ fly in the record variant)
  is excluded by nanmean; a pair occurrence without a finite fly is excluded from that replicate.
- Small-k rule (AA.5 6222–6226): k 0 none, k 1 pair study, groups ≥ min_groups two-stage primary, else fly-level
  primary + pair range + "적은 묶음" (two-stage a record when groups ≥ 2)."""
from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
from scipy import stats

from . import w_verdict as WV
from . import x_oc, y_oc, y_rules
from .aa_spec import SPEC as AA
from .w_spec import SPEC as W_SPEC
from .y_spec import SPEC as Y_SPEC

GATES = WV.GATES
GI = {g: i for i, g in enumerate(GATES)}
PRIMARY = ("reward_assoc", "punish_assoc")
RECORDED = ("reward_level", "punish_drop")
RAW = ("reward_assoc", "punish_assoc")
COLS = GATES + tuple(f"raw_{g}" for g in RAW)
METHODS = ("two_stage", "fly", "pair")
NOTE_BAR = "마리 관문 막대이며 쌍 평균 기준이 아님"
TARGET_LOW = "목표 ≤ 거짓 통과 효과"


def stream(root: int, *tags) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [int(root)] + [y_oc.tag(t) if isinstance(t, str) else int(t) for t in tags]))


def c_k(k: int) -> float:
    nu = k - 1
    return math.sqrt(nu / 2) * math.exp(math.lgamma((nu - 1) / 2) - math.lgamma(nu / 2))


def hedges_j(k: int) -> float:
    return 1 - 3 / (4 * (k - 1) - 1)


def _nanmean(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis)


# ================================================================ one pair (AA.4)
def fly_dprimes(d: dict, z: dict) -> np.ndarray:
    return np.asarray(WV.gate_stats(d, z), float)


def winsorize(x, s=AA) -> np.ndarray:
    return np.clip(np.asarray(x, float), -s.winsor, s.winsor)


def raw_contrasts(d: dict, z: dict) -> np.ndarray:
    r1, r2, n1 = WV.dv(d["R1"], z), WV.dv(d["R2"], z), WV.dv(d["N1"], z)
    rn1, rn2 = WV.dv(d["RN1"], z), WV.dv(d["RN2"], z)
    return np.stack([(r1 - n1).mean(-1), ((r2 - r1) - (rn2 - rn1)).mean(-1)], -1)


def floors(d: dict) -> dict:
    a = {s: np.asarray(d[s]) for s in WV.STAGES}
    per = {s: dict(MBON13_X=float((a[s][..., WV.A, WV.X] == 0).mean()),
                   MBON05_X=float((a[s][..., WV.P, WV.X] == 0).mean())) for s in WV.STAGES}
    r1, r2 = a["R1"], a["R2"]
    both = ((r1[..., WV.P, WV.X] == 0) & (r1[..., WV.P, WV.Y] == 0)) | \
           ((r2[..., WV.A, WV.X] == 0) & (r2[..., WV.A, WV.Y] == 0))
    return dict(stages=per, phi_R=float((r1[..., WV.P, WV.X] == 0).mean()),
                phi_P=float((r2[..., WV.A, WV.X] == 0).mean()), floor_both=float(both.mean()))


def _lab(v: float):
    return ("+inf" if v > 0 else "-inf") if np.isinf(v) else float(v)


def t_interval(x, s=AA):
    x = np.asarray(x, float)
    h = float(stats.t.ppf((1 + s.ci_level) / 2, x.size - 1)) * float(x.std(ddof=1)) / math.sqrt(x.size)
    return [float(x.mean()) - h, float(x.mean()) + h]


def pair_arrays(d: dict, z: dict, s=AA) -> dict:
    raw = fly_dprimes(d, z)
    if np.isnan(raw).any():
        raise ValueError("NaN 마리 d′ (AA.4: 측정 결함)")
    return dict(w=winsorize(raw, s), raw=raw, rc=raw_contrasts(d, z))


def pair_record(d: dict, z: dict, s=AA) -> dict:
    a = pair_arrays(d, z, s)
    F = a["raw"].shape[0]
    ok = F >= s.min_flies
    gates = {}
    for i, g in enumerate(GATES):
        x, xr = a["w"][:, i], a["raw"][:, i]
        fin = xr[np.isfinite(xr)]
        gates[g] = dict(fly=[_lab(v) for v in xr], mean=float(x.mean()) if ok else None, median=float(np.median(x)),
                        sd=float(x.std(ddof=1)) if ok else None, n_inf=int(np.isinf(xr).sum()),
                        n_clipped=int((np.abs(xr) > s.winsor).sum()), t_ci=t_interval(x, s) if ok else None,
                        drop_mean=float(fin.mean()) if fin.size >= s.min_flies else None, n_finite=int(fin.size))
    raw = {g: dict(mean=float(a["rc"][:, j].mean()), fly=a["rc"][:, j].tolist()) for j, g in enumerate(RAW)}
    return dict(gates=gates, raw=raw, floors=floors(d), F=F)


def pct(reps, s=AA) -> np.ndarray:
    return np.percentile(np.asarray(reps, float), s.pct, axis=0)


def pair_ci(a: dict, c: int, s=AA, B: int | None = None) -> dict:
    """AA.5 6218: the pair's fly bootstrap, stream ("pair", c); the 4 gates and 2 raw contrasts share the draw."""
    V = np.concatenate([a["w"], a["rc"]], 1)
    B = s.boot_b if B is None else int(B)
    r = stream(s.boot_seed, "pair", c)
    F = V.shape[0]
    reps = np.empty((B, V.shape[1]))
    for o in range(0, B, s.boot_chunk):
        n = min(s.boot_chunk, B - o)
        reps[o:o + n] = V[r.integers(0, F, size=(n, F))].mean(1)
    q = pct(reps, s)
    return dict(ci={c_: [float(q[0, j]), float(q[1, j])] for j, c_ in enumerate(COLS)}, reps=reps)


# ================================================================ pooled resampling (AA.5)
def boot_two_stage(D, groups, r, B, chunk) -> np.ndarray:
    D = np.asarray(D, float)
    P, F, n = D.shape
    G, M = len(groups), max(len(g) for g in groups)
    pad = np.full((G, M), -1, int)
    for j, g in enumerate(groups):
        pad[j, :len(g)] = g
    out = np.empty((B, n))
    for o in range(0, B, chunk):
        m = min(chunk, B - o)
        gi = r.integers(0, G, size=(m, G))
        fi = r.integers(0, F, size=(m, G, M, F))
        p = pad[gi]
        ok = p >= 0
        th = _nanmean(D[np.where(ok, p, 0)[..., None], fi], 3)
        th[~ok] = np.nan
        out[o:o + m] = _nanmean(th.reshape(m, G * M, n), 1)
    return out


def boot_fly(D, groups, r, B, chunk) -> np.ndarray:
    D = np.asarray(D, float)
    P, F, n = D.shape
    out = np.empty((B, n))
    for o in range(0, B, chunk):
        m = min(chunk, B - o)
        fi = r.integers(0, F, size=(m, P, F))
        out[o:o + m] = _nanmean(_nanmean(D[np.arange(P)[None, :, None], fi], 2), 1)
    return out


def boot_pair(D, groups, r, B, chunk) -> np.ndarray:
    D = np.asarray(D, float)
    P, F, n = D.shape
    out = np.empty((B, n))
    for o in range(0, B, chunk):
        m = min(chunk, B - o)
        pi = r.integers(0, P, size=(m, P))
        fi = r.integers(0, F, size=(m, P, F))
        out[o:o + m] = _nanmean(_nanmean(D[pi[..., None], fi], 2), 1)
    return out


BOOT = dict(two_stage=boot_two_stage, fly=boot_fly, pair=boot_pair)


def ci_plan(k: int, n_groups: int, s=AA) -> dict:
    if k == 0:
        return dict(kind="none", primary=None, records=[], flag=None)
    if k == 1:
        return dict(kind="pair_study", primary=None, records=[], flag="쌍별 연구")
    if n_groups >= s.min_groups:
        return dict(kind="pooled", primary="two_stage", records=["fly", "pair"], flag=None)
    rec = (["two_stage"] if n_groups >= 2 else []) + ["pair"]
    return dict(kind="pooled", primary="fly", records=rec, flag="적은 묶음")


def _tags(method: str, set_tag: str, s) -> tuple:
    return (s.flow(set_tag), set_tag) if method == "two_stage" else (f"pool_{method}", set_tag)


def _ci_of(reps, names, s) -> dict:
    q = pct(reps, s)
    return {g: [float(q[0, i]), float(q[1, i])] for i, g in enumerate(names)}


def estimate_set(arrs: dict, keys: list, set_tag: str, s=AA, B: int | None = None) -> dict:
    """arrs = {key: pair_arrays}; keys in c order. Point = unweighted mean of the pair estimates; CIs by the small-k
    rule (primary + records), the raw-contrast and ±∞-dropped records on the primary method, DL / HK per gate."""
    B = s.boot_b if B is None else int(B)
    groups = y_rules.merge_groups(list(keys))
    plan = ci_plan(len(keys), len(groups), s)
    out = dict(set=set_tag, k=len(keys), n_groups=len(groups), sizes=sorted((len(g) for g in groups), reverse=True),
               plan=plan, keys=list(keys))
    if plan["kind"] != "pooled":
        return out
    W_ = np.stack([arrs[x]["w"] for x in keys])
    R_ = np.stack([arrs[x]["raw"] for x in keys])
    RC = np.stack([arrs[x]["rc"] for x in keys])
    theta = W_.mean(1)
    out["point"] = {g: float(theta[:, i].mean()) for i, g in enumerate(GATES)}
    out["range"] = {g: [float(theta[:, i].min()), float(theta[:, i].max())] for i, g in enumerate(GATES)}
    out["ci"] = {}
    for m in [plan["primary"]] + plan["records"]:
        reps = BOOT[m](W_, groups, stream(s.boot_seed, *_tags(m, set_tag, s)), B, s.boot_chunk)
        out["ci"][m] = _ci_of(reps, GATES, s)
        if m == plan["primary"]:
            out["_reps"] = reps
    rr = BOOT[plan["primary"]](RC, groups, stream(s.boot_seed, "pool_raw", set_tag), B, s.boot_chunk)
    out["raw"] = dict(point={g: float(RC.mean(1)[:, j].mean()) for j, g in enumerate(RAW)}, ci=_ci_of(rr, RAW, s),
                      method=plan["primary"])
    out["drop"] = {}
    for i, g in enumerate(GATES):
        fin = np.isfinite(R_[..., i])
        keep = [j for j in range(len(keys)) if fin[j].sum() >= s.min_flies]
        sub = [keys[j] for j in keep]
        sg = y_rules.merge_groups(sub)
        p2 = ci_plan(len(sub), len(sg), s)
        Dd = np.where(fin, R_[..., i], np.nan)[keep][..., None]
        row = dict(k=len(sub), plan=p2, n_excluded_pairs=len(keys) - len(sub))
        if len(sub):
            row["point"] = float(_nanmean(_nanmean(Dd, 1), 0)[0])
        if p2["kind"] == "pooled":
            reps = BOOT[p2["primary"]](Dd, sg, stream(s.boot_seed, "pool_drop", set_tag, i), B, s.boot_chunk)
            row["ci"] = [float(v) for v in np.nanpercentile(reps[:, 0], s.pct)]
        out["drop"][g] = row
    v = W_.var(1, ddof=1) / W_.shape[1]
    out["dl"] = {g: dl_hk(theta[:, i], v[:, i], s) for i, g in enumerate(GATES)}
    out["sd_pairs"] = {g: float(np.std(theta[:, i], ddof=1)) for i, g in enumerate(GATES)}
    return out


def dl_hk(theta, v, s=AA) -> dict:
    th, v = np.asarray(theta, float), np.asarray(v, float)
    k = th.size
    if k < 2:
        return dict(available=False, reason="k < 2")
    if (v <= 0).any():
        return dict(available=False, reason="v_p = 0인 쌍이 있다")
    w = 1 / v
    mw = (w * th).sum() / w.sum()
    Q = float((w * (th - mw) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    ws = 1 / (v + tau2)
    mu = float((ws * th).sum() / ws.sum())
    se = math.sqrt(1 / ws.sum())
    zq = float(stats.norm.ppf((1 + s.ci_level) / 2))
    tq = float(stats.t.ppf((1 + s.ci_level) / 2, k - 1))
    se_hk = math.sqrt(float((ws * (th - mu) ** 2).sum()) / (k - 1) / ws.sum())
    return dict(available=True, Q=Q, tau2=float(tau2), tau=math.sqrt(tau2), mu=mu, se=se, ci=[mu - zq * se, mu + zq * se],
                I2=max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, hk_ci=[mu - tq * se_hk, mu + tq * se_hk])


def min_of_gates(point4, reps4, s=AA) -> dict:
    """AA.9.1 6304: signs (+, −, +, −) = w_verdict.SIGNS on GATES first, then the minimum; the CI from the per-
    replicate minimum of the primary resample (never the minimum of marginal CI ends)."""
    sp = WV.SIGNS * np.asarray(point4, float)
    dist = (WV.SIGNS * np.asarray(reps4, float)).min(1)
    lo, hi = np.percentile(dist, s.pct)
    return dict(point=float(sp.min()), gate=GATES[int(sp.argmin())], lo=float(lo), hi=float(hi))


def _pos(x: float, r: float, s) -> str:
    d = round(x - r, s.round_digits)
    return "위" if d > 0 else ("아래" if d < 0 else "같음")


def compare_table(point: float, lo: float, hi: float, s=AA) -> list:
    """AA.9 6282–6284: the point and CI against 0, ±0.5, ±1, ±1.5 on the uncorrected (K 8 fly d′) and the Hedges
    (× J(K)) scales side by side; the ±1 rows carry NOTE_BAR."""
    j, out = hedges_j(s.probes), []
    for r in s.compare_refs:
        for scale, f in (("raw", 1.0), ("hedges", j)):
            p, l_, h = f * point, f * lo, f * hi
            ci = "아래" if round(h - r, s.round_digits) < 0 else ("위" if round(l_ - r, s.round_digits) > 0 else "걸침")
            out.append(dict(ref=r, scale=scale, point_value=p, lo=l_, hi=h, point=_pos(p, r, s), ci=ci,
                            note=NOTE_BAR if abs(r) == s.bar_ref else ""))
    return out


def phi_quantiles(values, s=AA) -> list:
    return [float(v) for v in np.quantile(np.asarray(values, float), s.floor_qs)] if len(values) else []


def g6_inputs(mog: dict, s=AA) -> list:
    """AA.9.1 6304–6308: rows point / lo / hi × scale (Hedges first, then raw); target ≤ d_false (rounded) → status
    TARGET_LOW (not computed)."""
    j, out = hedges_j(s.probes), []
    for scale in s.g6_scales:
        f = j if scale == "hedges" else 1.0
        for row in s.g6_rows:
            t = f * float(mog[row])
            out.append(dict(row=row, scale=scale, target=t,
                            status=TARGET_LOW if round(t - s.d_false, s.round_digits) <= 0 else "compute"))
    return out


# ================================================================ structure-matched coverage (AA.5 6227–6231)
def cov_cells(s=AA) -> list:
    return [(float(d), h) for d in s.cov_deltas for h in s.cov_hets]


def target(delta: float, method: str, s=AA) -> float:
    """The covered value: δ · c(K) for the d′ methods (K-probe fly d′ expectation), δ for the raw contrasts."""
    return float(delta) if method.startswith("raw_") else float(delta) * c_k(s.probes)


def _groups(sizes) -> list:
    if not isinstance(sizes, tuple) or not all(type(n) is int and n >= 1 for n in sizes):
        raise TypeError("coverage takes the group-size structure only: a tuple of positive ints (AA.5)")
    out, pos = [], 0
    for n in sizes:
        out.append(list(range(pos, pos + n)))
        pos += n
    return out


def _share(groups, method, r, delta, het, sd, reps, B, s) -> float:
    P = sum(len(g) for g in groups)
    gid = np.concatenate([[j] * len(g) for j, g in enumerate(groups)]).astype(int)
    raw = method.startswith("raw_")
    boot = BOOT[method[len("raw_"):] if raw else method]
    tgt = target(delta, method, s)
    hits = 0
    for _ in range(reps):
        if het == "group":
            u = r.normal(0.0, sd, len(groups))[gid]
        elif het == "pair":
            u = r.normal(0.0, sd, P)
        else:
            u = np.zeros(P)
        x = delta + u[:, None, None] + r.standard_normal((P, s.flies, s.probes))
        D = (x.mean(-1) if raw else winsorize(WV.dprime(x), s))[..., None]
        lo, hi = np.percentile(boot(D, groups, r, B, s.boot_chunk)[:, 0], s.pct)
        hits += int(lo <= tgt <= hi)
    return hits / reps


def coverage_cell(sizes, method, set_tag, ci, s=AA, reps=None, B=None) -> float:
    """One cell (δ, heterogeneity) on stream SeedSequence([synth_seed, "cov", set_tag, method, ci])."""
    groups = _groups(sizes)
    delta, het = cov_cells(s)[int(ci)]
    return _share(groups, method, stream(s.synth_seed, "cov", set_tag, method, int(ci)), delta, het, s.cov_het_sd,
                  s.cov_reps if reps is None else int(reps), s.cov_b if B is None else int(B), s)


def coverage_summary(shares: list, s=AA) -> dict:
    cells = [dict(delta=d, het=h, share=float(v)) for (d, h), v in zip(cov_cells(s), shares)]
    m = min(c["share"] for c in cells)
    return dict(cells=cells, min=m, under=bool(round(m - s.cov_bar, s.round_digits) < 0))


def coverage(sizes, method, set_tag, s=AA, reps=None, B=None) -> dict:
    _groups(sizes)
    return dict(coverage_summary([coverage_cell(sizes, method, set_tag, i, s, reps, B)
                                  for i in range(len(cov_cells(s)))], s), sizes=list(sizes), method=method)


def synth_validation(s=AA, reps=None, B=None) -> dict:
    """AA.7 0 6247 (record): 20 independent pairs × 8 × 8, δ {0, 1, 1.5} × pair SD {0, 0.5}, 1000 reps, the primary
    two-stage CI (B = cov_b, plan Reading 6), stream (synth_seed, "synth", cell)."""
    groups = _groups((1,) * s.synth_pairs)
    cells = []
    for i, (d, sd) in enumerate((d, sd) for d in s.synth_deltas for sd in s.synth_sds):
        sh = _share(groups, "two_stage", stream(s.synth_seed, "synth", i), float(d), "pair" if sd else "none",
                    float(sd), s.synth_reps if reps is None else int(reps), s.cov_b if B is None else int(B), s)
        cells.append(dict(delta=float(d), sd=float(sd), share=sh))
    return dict(cells=cells, pairs=s.synth_pairs, method="two_stage", label="기록",
                note="결과로 방법을 바꾸지 않는다(AA.7 0); 실제 구조의 포함 확률은 순서 4b")


# ================================================================ G.6 recompute (AA.9.1, record only)
def g6_spec(tgt: float, s=AA, ys=Y_SPEC):
    return dataclasses.replace(ys, d_power=float(tgt), precheck_seed=s.g6_seed)


def _designs(power, false, fill_bad, yd) -> list:
    fs = list(range(yd.f_min, yd.f_max + 1))
    out = []
    for kr in [tuple(int(v) for v in k) for k in yd.k_ranges]:
        ok = y_oc.range_ok(power, false, fill_bad, yd, kr, False)
        out += [dict(p_set=yd.p_set_grid[i[0]], q=yd.q_grid[i[1]], K=yd.k_grid[i[2]], F=fs[i[3]], k_range=list(kr))
                for i in np.ndindex(ok.shape) if ok[i]]
    return out


def _rank(rows: list, costs: dict, n_naive: int, yd) -> list:
    for r in rows:
        r["cost_h"] = float(y_rules.design_cost_y(costs, r, yd, W_SPEC, n_naive, r["k_range"][1], False)["total_h"])
    return sorted(rows, key=lambda r: (-r["p_set"], r["cost_h"], -r["q"], r["K"], r["F"], r["k_range"][0]))


def g6_summary(pc: dict, yd, costs: dict, n_naive: int, s=AA) -> dict:
    power, false = np.asarray(pc["power"], float), np.asarray(pc["false"], float)
    fb = np.asarray(pc["fill_bad"], bool)
    rows = _rank(_designs(power, false, fb, yd), costs, n_naive, yd)
    pt = {k: np.asarray(v, float) for k, v in pc["point"].items()}
    pb = np.min([pt[f"g{g}|min|base"] for g in yd.cluster_grid], 0)
    fb_ = np.max([pt[f"g{g}|max|base"] for g in yd.cluster_grid], 0)
    base = _designs(pb, fb_, fb, yd)
    ps, q, K, F, kr = s.g6_design
    i = (yd.p_set_grid.index(ps), yd.q_grid.index(q), yd.k_grid.index(K), F - yd.f_min)
    counts = {f"{a}-{b}|{p}": sum(1 for r in rows if r["k_range"] == [a, b] and r["p_set"] == p)
              for a, b in yd.k_ranges for p in yd.p_set_grid}
    return dict(n_pass=len(rows), counts=counts, first=rows[0] if rows else None, designs=rows,
                n_pass_base_only=len(base), base_minus_all=len(base) - len(rows),
                target_design_power={k: float(power[i][k - yd.k_min]) for k in range(kr[0], kr[1] + 1)})


def g6_row(theta, z, tgt, costs, n_naive, s=AA, ys=Y_SPEC, cell=x_oc.run_cell) -> dict:
    yd = g6_spec(tgt, s, ys)
    pc = y_oc.precheck_y(theta, z, yd, [tuple(int(v) for v in k) for k in ys.k_ranges], cell=cell)
    c = pc["calibration"]["min"]
    if not c["ok"]:
        return dict(status=c["failure"]["status"], target=float(tgt))
    return dict(status="ok", target=float(tgt), false=pc["false"], fill_bad=pc["fill_bad"],
                **g6_summary(pc, yd, costs, n_naive, s))


def g6_hetero(theta, z, mu, tau, ref: dict, costs, n_naive, s=AA, ys=Y_SPEC) -> dict:
    """AA.9.1 6306 (plan Reading 13): half the pairs at μ − τ̂, half at μ + τ̂ (alternating over the k_cap slots),
    power on streams (g6_seed, "het" | "mix_het", g, scenario), false side and fill exclusion from `ref` (the same θ's
    point-row precheck, whose false target 0.5 does not depend on d_power)."""
    yd = g6_spec(mu, s, ys)
    idx = y_oc.rng(yd.precheck_seed, y_oc.TAG_CAL).integers(0, len(theta["resid"]), yd.cal_reps)
    cals = [y_oc.calibrate_y(theta, t, "min", idx, z, yd) for t in (mu - tau, mu + tau)]
    for c in cals:
        if not c["ok"]:
            return dict(status=c["failure"]["status"], mu=float(mu), tau=float(tau))
    mix = [cals[j % 2]["corners"] for j in range(yd.k_cap)]
    ones = [1.0] * yd.f_max
    pts, fill_bad = [], np.asarray(ref["fill_bad"], bool).copy()
    for gi, g in enumerate(yd.cluster_grid):
        for si, sc in enumerate(y_oc.SCENARIOS):
            r = y_oc.evaluate_y(theta, y_oc.rng(yd.precheck_seed, y_oc.tag("het"), gi, si),
                                y_oc.rng(yd.precheck_seed, y_oc.tag("mix_het"), gi, si), yd.precheck_reps, mix, ones,
                                ones, g, z, yd, sc)
            pts.append(r["p"])
            fill_bad |= np.round(np.asarray(r["fill_by_k"]) - yd.fill_max, yd.round_digits) > 0
    rows = _rank(_designs(np.min(pts, 0), np.asarray(ref["false"], float), fill_bad, yd), costs, n_naive, yd)
    return dict(status="ok", mu=float(mu), tau=float(tau), n_pass=len(rows), first=rows[0] if rows else None)
