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
