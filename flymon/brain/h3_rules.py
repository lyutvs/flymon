"""The M0d H.3 decision rules (spec appendix H.3a.4-H.3a.6) as pure functions over measured rows.

Nothing here runs the engine. Every rule takes the rows a measurement returned (or a callback that measures) and the
configuration object, so each rule has a synthetic-input unit test (spec H.3a.11). The statistics copy the
arithmetic of the committed diagnostics that produced the adopted operating point (`candidate_feasibility.py`
90c36f9, `d4_margin_protocol.py` e936434, `align_inputs.py` 19188ab, `readout_floor_guard.py` cde8bff), so the H.3
run reproduces their numbers bit for bit instead of approximately.
"""
from __future__ import annotations

import math

import numpy as np

# ---- candidate statuses (H.3a.4: each is recorded as a different state) ---------------------------------------
SEARCH_FAILED = "search_failed"                  # no evaluated point within the membrane tolerance (H.3a.5 step 1)
QUAL_FAILED = "qualification_failed"             # a qualification or the membrane recheck failed outright
INDETERMINATE = "indeterminate"                  # overlap CI still holds 0 after the extra seeds, or the guard halves split
GATE_FAILED = "stage4_failed"                    # baseline CI or runaway (H.3a.5 step 4)
NOT_REACHED = "not_reached"                      # a higher-ranked candidate was adopted first
ADOPTED = "adopted"
HOMEOSTASIS_UNCONVERGED = "homeostasis_unconverged"   # C3: 40 iterations without both stop conditions
BOUNDARY_LIMITED = "boundary_limited"            # C3: more than 10% of updated KCs at a clip bound
STALLED = "stalled"                              # C3: a cycle moved membrane < 0.2 mV and KC < 0.2 pp
CYCLES_EXHAUSTED = "cycles_exhausted"            # C3: 3 cycles and the membrane still outside 11 +- 1 mV
# ---- combination statuses ---------------------------------------------------------------------------------------
COMBO_ADOPTED = "adopted"
COMBO_DROPPED = "dropped"                        # candidates exhausted: stop, the user decides (H.3a.4)
COMBO_ABORTED = "compute_aborted"                # the budget ran out before every candidate was seen: not a drop


# ================================================================ bisections (H.3a.5 steps 1 and 2)
def log_mid(lo: float, hi: float) -> float:
    return float(math.exp(0.5 * (math.log(lo) + math.log(hi))))


def bisect_log(evaluate, lo: float, hi: float, steps: int, target: float, tol: float, key: str = "median_mv") -> dict:
    """Step 1: log bisection of a rising quantity. Both ends are evaluated first (the bracket is recorded), then
    `steps` midpoints. The accepted point is the EVALUATED point with the smallest |value - target| (ends first,
    then the trace, first minimum wins) and it converges when that distance is <= tol. `evaluate(x)` -> dict."""
    ends = [dict(evaluate(lo), x=float(lo)), dict(evaluate(hi), x=float(hi))]
    bracketed = bool(ends[0][key] <= target <= ends[1][key])
    trace = []
    for i in range(steps):
        mid = log_mid(lo, hi)
        st = dict(evaluate(mid), step=i + 1, x=mid, bracket_before=[lo, hi])
        if st[key] < target:
            lo = mid
        else:
            hi = mid
        st["bracket_after"] = [lo, hi]
        trace.append(st)
    evaluated = [ends[0], ends[1]] + trace
    best = min(evaluated, key=lambda s: abs(s[key] - target))
    return dict(endpoints=ends, trace=trace, bracketed=bracketed, final_bracket=[lo, hi], accepted=best,
                error=float(best[key] - target), converged=bool(abs(best[key] - target) <= tol))


def bisect_mid(evaluate, lo: float, hi: float, steps: int, target: float, key: str = "hz") -> dict:
    """Step 2: linear bisection with no early stop; the accepted value is the final interval's midpoint (it is
    never measured here, which is why this rule differs from step 1's). Ends are evaluated for the record."""
    ends = [dict(evaluate(lo), x=float(lo)), dict(evaluate(hi), x=float(hi))]
    bracketed = bool(ends[0][key] <= target <= ends[1][key])
    trace = []
    for i in range(steps):
        mid = 0.5 * (lo + hi)
        st = dict(evaluate(mid), step=i + 1, x=float(mid), bracket_before=[float(lo), float(hi)])
        if st[key] < target:
            lo = mid
        else:
            hi = mid
        st["bracket_after"] = [float(lo), float(hi)]
        trace.append(st)
    return dict(endpoints=ends, trace=trace, bracketed=bracketed, final_bracket=[float(lo), float(hi)],
                accepted=float(0.5 * (lo + hi)))


# ================================================================ reference-set statistics
def reference_stats(rows: list[dict], quasi_linear: tuple) -> dict:
    """candidate_feasibility.point_stats: medians over the presentations (not over odours)."""
    v = np.array([p["apl_v_mean"] for p in rows], float)
    kc = 100.0 * np.array([p["kc_active_frac"] for p in rows], float)
    q1, q3 = (float(x) for x in np.percentile(v, [25, 75]))
    out = dict(n_pres=len(rows), median_mv=float(np.median(v)), q1_mv=q1, q3_mv=q3, iqr_mv=q3 - q1,
               median_kc_pct=float(np.median(kc)), kc_q1_pct=float(np.percentile(kc, 25)),
               kc_q3_pct=float(np.percentile(kc, 75)))
    if rows and rows[0].get("release_frac") is not None:
        rel = np.array([p["release_frac"] for p in rows], float)
        lo, hi = quasi_linear
        out.update(median_release_frac=float(np.median(rel)), rel_share_below=float((rel < lo).mean()),
                   rel_share_in=float(((rel >= lo) & (rel <= hi)).mean()), rel_share_above=float((rel > hi).mean()))
    return out


def membrane_check(stats: dict, target: float, tol: float) -> dict:
    err = float(stats["median_mv"] - target)
    return dict(median_mv=float(stats["median_mv"]), error=err, ok=bool(abs(err) <= tol))


# ================================================================ D.4 on the design pair (qualifications 1 and 2)
D4_KEYS = ("frac_active_A", "frac_active_B", "jaccard", "chance", "kc_hz_A", "kc_hz_B", "mbon_hz_A", "mbon_hz_B")


def d4_stats(rows: list[dict], band: tuple) -> dict:
    """d4_margin_protocol.design_stats: seed means; sparsity margin in percentage points and overlap margin."""
    m = {k: float(np.mean([r[k] for r in rows])) for k in D4_KEYS}
    lo, hi = 100 * band[0], 100 * band[1]
    pa, pb = 100 * m["frac_active_A"], 100 * m["frac_active_B"]
    m.update(n_seeds=len(rows), seeds=[int(r["seed"]) for r in rows], pct_A=pa, pct_B=pb,
             band_ok=bool(lo <= pa <= hi and lo <= pb <= hi),
             margin_pp=float(min(hi - max(pa, pb), min(pa, pb) - lo)),
             overlap_margin=float(m["chance"] - m["jaccard"]),
             overlap_margin_per_seed=[float(r["chance"]) - float(r["jaccard"]) for r in rows])
    return m


def boot_mean_ci(x, draws: int, seed: int) -> dict:
    """align_inputs.boot_overlap: 95% percentile CI of the mean, resampling seeds, a fresh generator per call."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(draws, x.size))].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return dict(n=int(x.size), draws=int(draws), seed=int(seed), mean=float(x.mean()), ci=[lo, hi],
                holds_zero=bool(lo <= 0.0 <= hi))


def overlap_clause(ci: list) -> str:
    """'pass' when the whole CI is above 0, 'fail' when it is below, 'holds_zero' otherwise (then: more seeds)."""
    lo, hi = ci
    return "pass" if lo > 0.0 else "fail" if hi < 0.0 else "holds_zero"


# ================================================================ readout-floor guard (qualification 3)
def mbon_type_stats(stim: list[dict], rest_by_seed: dict, name: str, med_min: float, zero_max: float) -> dict:
    """readout_floor_guard.type_stats: point estimates over the presentations."""
    s = np.array([p["types"][name] for p in stim], float)
    r = np.array([rest_by_seed[p["seed"]]["types"][name] for p in stim], float)
    d = s - r
    med, zero = float(np.median(d)), float((s == 0).mean())
    return dict(median_delta=med, zero_share=zero, median_stim=float(np.median(s)), median_rest=float(np.median(r)),
                mean_stim=float(s.mean()), n_pres=len(stim), passes=bool(med >= med_min and zero <= zero_max))


def mbon_type_boot(stim: list[dict], rest_by_seed: dict, name: str, odor_names: list, draws: int, seed: int) -> dict:
    """Recorded only (H.3a.4: the guard judges on point estimates): odour-cluster percentile CIs."""
    cl_s = np.array([[float(p["types"][name]) for p in stim if p["odor"] == o] for o in odor_names], float)
    cl_r = np.array([[float(rest_by_seed[p["seed"]]["types"][name]) for p in stim if p["odor"] == o]
                     for o in odor_names], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(odor_names), size=(draws, len(odor_names)))
    s = cl_s[idx].reshape(draws, -1)
    d = (cl_s - cl_r)[idx].reshape(draws, -1)
    zl, zh = (float(x) for x in np.percentile((s == 0).mean(axis=1), [2.5, 97.5]))
    ml, mh = (float(x) for x in np.percentile(np.median(d, axis=1), [2.5, 97.5]))
    return dict(zero_share_ci=[zl, zh], median_delta_ci=[ml, mh], draws=int(draws), seed=int(seed))


def guard_pool(stim: list[dict], rest_by_seed: dict, types: list, odor_names: list, spec) -> dict:
    """At least one type of the pool passes. With exactly one passing type its zero-share is re-measured on each
    half of the odours (0..guard_half-1, guard_half..) and both halves must pass; split halves are INDETERMINATE."""
    stats = {n: mbon_type_stats(stim, rest_by_seed, n, spec.guard_med_delta_min, spec.guard_zero_share_max)
             for n in types}
    passing = [n for n in types if stats[n]["passes"]]
    out = dict(types=stats, passing=passing, halves=None)
    if not passing:
        out["verdict"] = "fail"
    elif len(passing) > 1:
        out["verdict"] = "pass"
    else:
        first = set(odor_names[:spec.guard_half])
        halves = []
        for part in (True, False):
            sub = [p for p in stim if (p["odor"] in first) == part]
            z = float(np.mean([p["types"][passing[0]] == 0 for p in sub]))
            halves.append(dict(n_pres=len(sub), zero_share=z, ok=bool(z <= spec.guard_zero_share_max)))
        out["halves"] = halves
        out["verdict"] = "pass" if all(h["ok"] for h in halves) else "indeterminate"
    return out


# ================================================================ stage 4 (H.3a.5 step 4)
def baseline_stats(per_seed_hz) -> dict:
    x = np.asarray(per_seed_hz, float)
    sd = float(x.std(ddof=1))
    return dict(n=int(x.size), mean_hz=float(x.mean()), sd_hz=sd, se_hz=float(sd / math.sqrt(x.size)),
                per_seed=[float(v) for v in x])


def baseline_ci_verdict(stats: dict, z: float, band: tuple) -> dict:
    """Pass when the 95% CI (mean +- z SE) overlaps the band: a point band fails a calibrated engine on noise."""
    lo, hi = stats["mean_hz"] - z * stats["se_hz"], stats["mean_hz"] + z * stats["se_hz"]
    return dict(ci=[float(lo), float(hi)], band=list(band), ok=bool(hi >= band[0] and lo <= band[1]))


def runaway_verdict(rest_kc_over: list, odor_kc_over: list, n_rest: int, n_odor: int) -> dict:
    """D.4: no KC above the rest threshold in any rest seed and none above the odour threshold in any odour-B seed;
    the sample sizes are part of the definition, so a short run cannot pass."""
    ok = bool(len(rest_kc_over) >= n_rest and len(odor_kc_over) >= n_odor
              and max(rest_kc_over, default=1) == 0 and max(odor_kc_over, default=1) == 0)
    return dict(rest_kc_over_per_seed=[int(v) for v in rest_kc_over],
                odor_kc_over_per_seed=[int(v) for v in odor_kc_over], ok=ok)


# ================================================================ ranking (H.3a.5 step 3)
def rank_score(margin_pp: float, overlap_margin: float, overlap_sd: float) -> float:
    """The normalised minimum of the sparsity margin (pp) and the overlap margin (in pooled-SD units)."""
    return float(min(margin_pp, overlap_margin / overlap_sd))


def qualification_verdict(sparsity_ok: bool, overlap: str, guard_verdicts: list, membrane_ok: bool | None) -> tuple:
    """(status or None, reasons). A definite failure outranks an indeterminate clause; None means qualified."""
    fails, undecided = [], []
    if not sparsity_ok:
        fails.append("sparsity")
    if overlap == "fail":
        fails.append("overlap")
    elif overlap != "pass":
        undecided.append("overlap")
    for pool, v in guard_verdicts:
        if v == "fail":
            fails.append(f"guard_{pool}")
        elif v != "pass":
            undecided.append(f"guard_{pool}")
    if membrane_ok is False:
        fails.append("membrane")
    if fails:
        return QUAL_FAILED, fails + undecided
    if undecided:
        return INDETERMINATE, undecided
    return None, []
