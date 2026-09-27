"""The primary (reward-side machine control) verdict — spec 10.4. Pure functions over b_runner probe records."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import betabinom

from ..brain.b_rules import check_records, dprime, noplast_ok, v_of

PASS, FAIL, INVALID, NOT_CONSTRUCTIBLE = "PASS", "FAIL", "INVALID", "NOT_CONSTRUCTIBLE"
STOP_MACHINE, STOP_PROTOCOL, STOP_CONTROL_INVALID, STOP_FEW_PAIRS, OK = (
    "STOP_MACHINE", "STOP_PROTOCOL", "STOP_CONTROL_INVALID", "STOP_FEW_PAIRS", "OK")
BRAINS = ("Rr", "N", "N2", "noplast")


def _other(x): return "b" if x == "a" else "a"


def _index(records):
    return {(r["brain"], r["fly"], r["stage"], r["seed"]): r["counts"] for r in records}


def naive_p(records, x, spec) -> float:
    return float(np.median([r["counts"][x][spec.p_type] for r in records if r["brain"] == "Rr" and r["stage"] == "pre"]))


def choose_x(records, spec) -> str:
    a, b = naive_p(records, "a", spec), naive_p(records, "b", spec)
    return spec.x_tie if a == b else ("a" if a > b else "b")


def _spill(e_x, e_y) -> float:
    return math.inf if e_x <= 0 else abs(e_y) / e_x


def _beats_null(med, null_max) -> bool:
    return med > null_max


def _floor_odours(x, y) -> tuple:
    return (x, y)


def fly_stats(records, x, spec, fly) -> dict | None:
    y, idx = _other(x), _index(records)
    seeds = sorted({r["seed"] for r in records if r["brain"] == "Rr" and r["fly"] == fly and r["stage"] == "pre"})
    try:
        V = {(b, st): np.array([[v_of(idx[(b, fly, st, s)][o], spec) for s in seeds] for o in (x, y)])
             for b in ("Rr", "N", "N2") for st in ("pre", "S1")}
        P = {b: np.array([idx[(b, fly, "S1", s)][x][spec.p_type] for s in seeds]) for b in ("Rr", "N")}
    except KeyError:
        return None
    dv = {b: V[(b, "S1")][0] - V[(b, "S1")][1] for b in ("Rr", "N", "N2")}
    assoc, null, level = dprime(dv["Rr"] - dv["N"]), dprime(dv["N2"] - dv["N"]), dprime(dv["Rr"])
    if assoc is None or null is None:
        return None
    e_x = float(np.mean(V[("Rr", "S1")][0] - V[("N", "S1")][0]))
    e_y = float(np.mean(V[("Rr", "S1")][1] - V[("N", "S1")][1]))
    sign = float(np.mean(np.where(P["Rr"] < P["N"], 1.0, np.where(P["Rr"] == P["N"], 0.5, 0.0))))
    return dict(assoc=assoc, null=null, e_x=e_x, e_y=e_y, spill=_spill(e_x, e_y), level=level, sign=sign)


def _seeds_of(spec, pair):
    return lambda f: spec.probe_seeds(pair, f)


def pair_verdict(records, x, spec, pair) -> dict:
    out = dict(pair=pair, x=x, reasons=[], n_valid=0, assoc_median=None, null_max=None, spill_median=None,
               level_median=None, sign_median=None)
    reasons = check_records(records, BRAINS, spec, _seeds_of(spec, pair))
    if reasons:
        return dict(out, status=INVALID, reasons=reasons)
    if not noplast_ok(records):
        return dict(out, status=STOP_MACHINE, reasons=["noplast counts moved"])
    if x != choose_x(records, spec):
        return dict(out, status=INVALID, reasons=[f"X {x} is not the rule's {choose_x(records, spec)}"])
    if min(naive_p(records, o, spec) for o in _floor_odours(x, _other(x))) < spec.floor_spikes:
        return dict(out, status=NOT_CONSTRUCTIBLE, reasons=["naive MBON05 median below floor"])
    stats = [fly_stats(records, x, spec, f) for f in range(spec.n_flies)]
    valid = [s for s in stats if s is not None and not any(isinstance(v, float) and math.isnan(v) for v in s.values())]
    out["n_valid"] = len(valid)
    if len(valid) < spec.valid_min:
        return dict(out, status=INVALID, reasons=[f"{len(valid)} valid flies < {spec.valid_min}"])
    med = lambda k: float(np.median([s[k] for s in valid]))
    out.update(assoc_median=med("assoc"), null_max=float(max(s["null"] for s in valid)), spill_median=med("spill"),
               level_median=med("level"), sign_median=med("sign"))
    ok = (out["assoc_median"] >= spec.assoc_min and _beats_null(out["assoc_median"], out["null_max"])
          and out["spill_median"] <= spec.spill_max)
    return dict(out, status=PASS if ok else FAIL)


def control_verdict(records, x, spec, qualified: bool) -> dict:
    if not qualified:
        return dict(status=STOP_CONTROL_INVALID, pair=spec.control, sign_median=None)
    v = pair_verdict(records, x, spec, spec.control)
    if v["status"] in (INVALID, STOP_MACHINE, NOT_CONSTRUCTIBLE):
        return dict(status=v["status"], pair=spec.control, sign_median=None, reasons=v["reasons"])
    status = OK if v["sign_median"] >= spec.sign_min else STOP_PROTOCOL
    return dict(status=status, pair=spec.control, sign_median=v["sign_median"])


def false_pass(t: int, m: int, q: float, rho: float) -> float:
    if rho <= 0:
        from scipy.stats import binom
        return float(binom.sf(t - 1, m, q))
    ab = (1 - rho) / rho
    return float(betabinom.sf(t - 1, m, q * ab, (1 - q) * ab))


def threshold_t(m: int, spec) -> int | None:
    t = math.ceil(2 * m / 3)
    while t <= m and false_pass(t, m, spec.q_null, spec.rho) > spec.fp_cap:
        t += 1
    return t if t <= m else None


def oc_table(spec) -> dict:
    qs = (0.2, 0.4, 0.6, 0.8, 0.9)
    return {str(m): {"t": threshold_t(m, spec),
                     "p_pass": {str(q): [false_pass(threshold_t(m, spec), m, q, 1e-4),
                                         false_pass(threshold_t(m, spec), m, q, spec.rho)] for q in qs}}
            for m in range(spec.min_pairs, spec.n_pairs + 1)}


def overall(control: dict, pairs: dict, spec) -> dict:
    statuses = [p["status"] for p in pairs.values()]
    m = len(pairs)
    base = dict(m=m, t=None, n_pass=statuses.count(PASS))
    if STOP_MACHINE in statuses or control["status"] == STOP_MACHINE:
        return dict(base, status=STOP_MACHINE)
    for st in (INVALID, NOT_CONSTRUCTIBLE):
        if control["status"] == st:
            return dict(base, status=st)
    if control["status"] in (STOP_CONTROL_INVALID, STOP_PROTOCOL):
        return dict(base, status=control["status"])
    for st in (INVALID, NOT_CONSTRUCTIBLE):
        if st in statuses:
            return dict(base, status=st)
    if m < spec.min_pairs:
        return dict(base, status=STOP_FEW_PAIRS)
    t = threshold_t(m, spec)
    return dict(base, t=t, status=PASS if base["n_pass"] >= t else FAIL)


# ---- spec 10.4 item 6: recorded only, never judged -------------------------------------------------------------
def _med(xs):
    xs = [float(v) for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(xs)) if xs else None


def _fly_dv(idx, spec, x, brain, fly, stage, seeds):
    """Per probe seed V(X) - V(Y) of one brain / fly / stage (None when a row is missing)."""
    y = _other(x)
    try:
        return np.array([v_of(idx[(brain, fly, stage, s)][x], spec) - v_of(idx[(brain, fly, stage, s)][y], spec)
                         for s in seeds])
    except KeyError:
        return None


def _choice(dv):
    """Mean over probe seeds of [V(X) > V(Y)], ties 1/2."""
    return None if dv is None or dv.size == 0 else float(np.mean(np.where(dv > 0, 1.0, np.where(dv == 0, 0.5, 0.0))))


def recorded_items(records, x, spec, xcore=None, floor_frac=None) -> dict:
    """Spec 10.4 item 6 (recorded, no verdict effect), on the probe records of one pair.

    naive_dprime: d'(dV_Rr,pre) over every fly's Rr pre probe seeds pooled (the naive brains are identical; F's
      pooled naive d'), plus the per-fly d' and their median.
    choice: c_s[b][stage] for b in Rr / N / N2, stage pre / S1 = per fly the mean over probe seeds of [V(X) > V(Y)]
      (ties 1/2): per-fly values and their median; rr_minus_n_S1 = per fly c_Rr,S1 - c_N,S1 and its median.
    self_change[b] for b in N / N2 = per fly d'(dV_b,S1 - dV_b,pre) (probe seed by seed; J.12.9): median and max.
    xcore (when given {"Rr": [per-fly w/w0 arrays], "N": [...]} on the X-core MBON05 edge mask): per fly the median
      w/w0, per brain the median over flies, ratio Rr / N of those medians; floor_contact = per fly the share of the
      mask's edges at w/w0 <= floor_frac (min_weight_frac), per brain the median over flies.
    dV = V(X) - V(Y) at each probe seed, V = z_A - z_P (b_rules.v_of)."""
    idx = _index(records)
    flies = sorted({r["fly"] for r in records if r["brain"] == "Rr" and r["stage"] == "pre"})
    seeds = {f: sorted({r["seed"] for r in records if r["brain"] == "Rr" and r["fly"] == f and r["stage"] == "pre"})
             for f in flies}
    dv = {(b, f, st): _fly_dv(idx, spec, x, b, f, st, seeds[f])
          for b in ("Rr", "N", "N2") for f in flies for st in ("pre", "S1")}

    pooled = [v for f in flies if dv[("Rr", f, "pre")] is not None for v in dv[("Rr", f, "pre")]]
    per_fly_naive = [None if dv[("Rr", f, "pre")] is None else dprime(dv[("Rr", f, "pre")]) for f in flies]
    naive = dict(pooled=dprime(pooled), n_pooled=len(pooled), per_fly=per_fly_naive, fly_median=_med(per_fly_naive))

    choice = {}
    for b in ("Rr", "N", "N2"):
        choice[b] = {}
        for st in ("pre", "S1"):
            c = [_choice(dv[(b, f, st)]) for f in flies]
            choice[b][st] = dict(per_fly=c, median=_med(c))
    diff = [None if None in (choice["Rr"]["S1"]["per_fly"][i], choice["N"]["S1"]["per_fly"][i])
            else choice["Rr"]["S1"]["per_fly"][i] - choice["N"]["S1"]["per_fly"][i] for i in range(len(flies))]
    choice["rr_minus_n_S1"] = dict(per_fly=diff, median=_med(diff))

    self_change = {}
    for b in ("N", "N2"):
        d = [None if dv[(b, f, "S1")] is None or dv[(b, f, "pre")] is None
             else dprime(dv[(b, f, "S1")] - dv[(b, f, "pre")]) for f in flies]
        ok = [v for v in d if v is not None and not math.isnan(v)]
        self_change[b] = dict(per_fly=d, median=_med(d), max=float(max(ok)) if ok else None)

    out = dict(naive_dprime=naive, choice=choice, self_change=self_change, flies=flies, xcore=None)
    if xcore is not None:
        xc = {}
        for b in ("Rr", "N"):
            arrs = [np.asarray(a, float) for a in xcore.get(b, [])]
            meds = [float(np.median(a)) if a.size else None for a in arrs]
            fl = ([float(np.mean(a <= floor_frac + 1e-9)) if a.size else None for a in arrs]
                  if floor_frac is not None else [None] * len(arrs))
            xc[b] = dict(per_fly_median=meds, median=_med(meds), floor_contact_per_fly=fl, floor_contact=_med(fl))
        rr, n = xc["Rr"]["median"], xc["N"]["median"]
        n_edges = int(np.asarray(xcore["Rr"][0]).size) if xcore.get("Rr") else 0
        out["xcore"] = dict(xc, ratio_rr_over_n=None if rr is None or not n else rr / n, floor_frac=floor_frac,
                            n_edges=n_edges)
    return out
