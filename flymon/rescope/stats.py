"""Evaluation-block win rates, fly -> battle paired hierarchical bootstrap (Readings R7), (2a)/(2b) verdict (spec 4.5).

Inputs are Task 8's arm result.json dicts ({flies, eval, per_fly: [{fly, invalid, eval_battles: [{won, finished}],
RS: donor_invalid, residual_frac, donor_sha256}], ...}). A battle counts as a win only when `won is True`; an
unfinished battle counts as a loss (as in pilot_no_brain). A fly is INVALID in an arm when its row says `invalid`, or
(RS) `donor_invalid` (FLY k was invalid - blocks.fly_result does not fold it in), a residual above spec.residual_max,
or eval weights that changed; a comparison drops fly k when it is INVALID in either arm (an INVALID pair).

The verdict is INVALID - never PASS / FAIL - when an arm is missing or malformed (missing / duplicate fly rows, a
`won` that is not True / False / None, NaN, eval battle counts that differ), or when a comparison keeps fewer valid
pairs than min_valid_pairs(F) = ceil(F * spec.valid_min / spec.n_flies), floor 2 (plan R10; that shortage makes only the
verdict using the comparison INVALID).
"""
from __future__ import annotations

import math

import numpy as np

from .spec import SPEC

PASS, FAIL, INVALID, OK = "PASS", "FAIL", "INVALID", "OK"
ARMS = ("FLY", "RS", "COFF", "RND", "MAX")
COMPARISONS = {"2a": ("FLY", "RND"), "vs_coff": ("FLY", "COFF"), "vs_rs": ("FLY", "RS")}


# ---- per-arm tables ---------------------------------------------------------------------------
def _bad_number(x) -> bool:
    return isinstance(x, float) and not math.isfinite(x)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def row_invalid(row: dict, spec=SPEC) -> bool:
    """A fly row is INVALID for pairing: its own flag, the RS donor's (FLY k), residual > max, weights changed."""
    if row.get("invalid") or row.get("donor_invalid"):
        return True
    if "residual_frac" in row:
        r = row["residual_frac"]
        if not _is_number(r) or _bad_number(float(r)) or r > spec.residual_max:
            return True
    return row.get("weights_bit_identical_across_eval") is False


def win_table(result: dict, spec=SPEC) -> dict:
    """fly -> 0/1 array of eval-block wins (unfinished = loss), INVALID flies excluded."""
    return {int(f["fly"]): np.array([1.0 if b["won"] is True else 0.0 for b in f["eval_battles"]])
            for f in result["per_fly"] if not row_invalid(f, spec)}


def check_arm(name: str, result, spec=SPEC) -> list:
    """Reasons this arm's result cannot be judged (empty = well-formed)."""
    if not isinstance(result, dict):
        return [f"{name}: result missing"]
    rows = result.get("per_fly")
    if not isinstance(rows, list) or not rows:
        return [f"{name}: per_fly missing or empty"]
    out = []
    if "complete" in result and result["complete"] is not True:
        out.append(f"{name}: arm not complete")
    try:
        flies = [int(r["fly"]) for r in rows]
    except (KeyError, TypeError, ValueError):
        return out + [f"{name}: a per_fly row has no integer fly"]
    n = result.get("flies", len(rows))
    if sorted(flies) != list(range(n)):
        out.append(f"{name}: fly rows {sorted(flies)} are not exactly 0..{n - 1} (missing or duplicate rows)")
    e = result.get("eval")
    for r in rows:
        f = r.get("fly")
        if not isinstance(r.get("invalid", False), bool):
            out.append(f"{name} fly {f}: invalid flag {r.get('invalid')!r} is not a bool")
        if "residual_frac" in r and (not _is_number(r["residual_frac"]) or _bad_number(float(r["residual_frac"]))):
            out.append(f"{name} fly {f}: residual_frac {r['residual_frac']!r}")
        ev = r.get("eval_battles")
        if not isinstance(ev, list):
            out.append(f"{name} fly {f}: eval_battles missing")
            continue
        if row_invalid(r, spec):
            continue                                   # an INVALID fly is dropped; its battles are not judged
        if not ev:
            out.append(f"{name} fly {f}: no eval battles")
        if e is not None and len(ev) != e:
            out.append(f"{name} fly {f}: {len(ev)} eval battles != eval {e}")
        for b in ev:
            if not isinstance(b, dict) or "won" not in b or not (b["won"] is None or isinstance(b["won"], bool)):
                out.append(f"{name} fly {f}: eval battle won {b.get('won') if isinstance(b, dict) else b!r} is not "
                           "True / False / None")
                break
    return out


def min_valid_pairs(n_flies: int, spec=SPEC) -> int:
    return max(2, math.ceil(n_flies * spec.valid_min / spec.n_flies))


# ---- the paired bootstrap ---------------------------------------------------------------------
def paired_boot(a: dict, b: dict, draws: int, seed: int) -> dict:
    """Flies k valid in both tables are resampled jointly (one pick of k serves both arms), then each arm's battles
    within the picked fly are resampled with replacement; statistic = mean over flies of rate(a) - rate(b), 95% CI
    from the 2.5 / 97.5 percentiles. Wins are 0/1, so a with-replacement resample of fly k's n battles has mean
    Binomial(n, p_k) / n exactly - that is how the battle level is drawn (vectorised, same distribution)."""
    ks = sorted(set(a) & set(b))
    if not ks:
        return dict(diff=None, lo=None, hi=None, n_pairs=0, flies=[])
    for t in (a, b):
        for k in ks:
            v = t[k]
            if v.size == 0 or not np.all((v == 0.0) | (v == 1.0)):
                raise ValueError(f"fly {k}: eval wins must be a non-empty 0/1 array")
    na = np.array([a[k].size for k in ks]); pa = np.array([a[k].mean() for k in ks])
    nb = np.array([b[k].size for k in ks]); pb = np.array([b[k].mean() for k in ks])
    point = float(np.mean(pa - pb))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ks), size=(draws, len(ks)))
    ra = rng.binomial(na[pick], pa[pick]) / na[pick]
    rb = rng.binomial(nb[pick], pb[pick]) / nb[pick]
    stat = (ra - rb).mean(axis=1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return dict(diff=point, lo=float(lo), hi=float(hi), n_pairs=len(ks), flies=[int(k) for k in ks])


# ---- the verdict ------------------------------------------------------------------------------
def _arm_invalid(result: dict, spec) -> list:
    return sorted(int(f["fly"]) for f in result["per_fly"] if row_invalid(f, spec))


def _rate(t: dict):
    return float(np.mean([v.mean() for v in t.values()])) if t else None


def m4_verdict(arms: dict, spec=SPEC) -> dict:
    """{status, reasons, 2a: {verdict, pass, reasons, diff, lo, hi, n_pairs}, 2b: {verdict, pass, reasons, vs_coff,
    vs_rs}, invalid_pairs, invalid_flies, min_valid_pairs, recorded}. (2a) passes iff lo(FLY - RND) > 0, (2b) iff
    lo(FLY - COFF) > 0 and lo(FLY - RS) > 0.
    status INVALID (a missing / malformed arm, fly or eval counts that differ) makes both verdicts INVALID. A pair
    shortage (< min_valid_pairs) or a non-finite statistic is per verdict (plan R10): (2a) INVALID only from FLY/RND,
    (2b) INVALID from FLY/COFF or FLY/RS; the other verdict is still judged and status stays OK. An INVALID verdict
    has pass None, never PASS / FAIL."""
    reasons = []
    for name in ARMS:
        reasons += check_arm(name, arms.get(name), spec)
    if reasons:
        return dict(status=INVALID, reasons=reasons, **{"2a": dict(verdict=INVALID, **{"pass": None}),
                                                         "2b": dict(verdict=INVALID, **{"pass": None})})
    nf = {name: arms[name].get("flies", len(arms[name]["per_fly"])) for name in ARMS}
    ne = {name: arms[name].get("eval") for name in ARMS}
    if len(set(nf.values())) != 1:
        reasons.append(f"arms have different fly counts {nf}")
    if len({e for e in ne.values() if e is not None}) > 1:
        reasons.append(f"arms have different eval counts {ne}")
    F = max(nf.values())
    need = min_valid_pairs(F, spec)
    t = {name: win_table(arms[name], spec) for name in ARMS}
    cmp, short = {}, {}
    for key, (x, y) in COMPARISONS.items():
        c = paired_boot(t[x], t[y], spec.boot_draws, spec.boot_seed)
        short[key] = []
        if c["n_pairs"] < need:
            short[key].append(f"{key} ({x} - {y}): {c['n_pairs']} valid pairs < {need}")
        elif any(v is None or _bad_number(v) for v in (c["diff"], c["lo"], c["hi"])):
            short[key].append(f"{key} ({x} - {y}): non-finite statistic {c}")
        cmp[key] = c
    invalid_pairs = {key: dict(x=x, y=y, flies=sorted(set(range(F)) - set(cmp[key]["flies"])),
                               n=F - cmp[key]["n_pairs"])
                     for key, (x, y) in COMPARISONS.items()}
    status = INVALID if reasons else OK

    why = {"2a": short["2a"], "2b": short["vs_coff"] + short["vs_rs"]}

    def verdict(key: str, ok: bool):
        bad = status == INVALID or why[key]
        return (INVALID, None) if bad else ((PASS, True) if ok else (FAIL, False))

    va, pa = verdict("2a", cmp["2a"]["lo"] is not None and cmp["2a"]["lo"] > 0)
    vb, pb = verdict("2b", all(cmp[k]["lo"] is not None and cmp[k]["lo"] > 0 for k in ("vs_coff", "vs_rs")))
    rate = {name: _rate(t[name]) for name in ARMS}
    fly, rnd, mx = rate["FLY"], rate["RND"], rate["MAX"]
    recorded = dict(win_rate=rate, n_valid_flies={name: len(t[name]) for name in ARMS},
                    max_minus_fly=None if None in (mx, fly) else mx - fly,
                    fly_max_ratio=None if not mx or fly is None else fly / mx,
                    fly_rnd_ratio=None if not rnd or fly is None else fly / rnd)
    return {"status": status, "reasons": reasons, "min_valid_pairs": need, "n_flies": F,
            "2a": dict(cmp["2a"], verdict=va, reasons=why["2a"], **{"pass": pa}),
            "2b": {"verdict": vb, "pass": pb, "reasons": why["2b"], "vs_coff": cmp["vs_coff"], "vs_rs": cmp["vs_rs"]},
            "invalid_pairs": invalid_pairs,
            "invalid_flies": {name: _arm_invalid(arms[name], spec) for name in ARMS},
            "recorded": recorded}


# ---- recorded only: the information-turn type match (original spec 4.3 criterion 2 statistic) -------------
def info_turn_match(records) -> float | None:
    """The match rate of info_turn_stats (None when no informative turn)."""
    return info_turn_stats(records)["rate"]


def info_turn_stats(records) -> dict:
    """Among fly-decided turns whose candidates' type multipliers do not all equal the maximum (an informative
    turn), the fraction whose chosen move has the maximal multiplier. A decision record needs `multipliers` (one per
    candidate, same order as `candidates`); the logschema decision fields do not carry it, so records without it are
    counted in n_without_multipliers and skipped. Returns {rate, n_info, n_fly_turns, n_without_multipliers,
    info_frac}; rate / info_frac are None when undefined."""
    n_fly = n_info = n_hit = n_missing = 0
    for r in records:
        if r.get("kind") != "decision" or r.get("decider") != "fly":
            continue
        n_fly += 1
        cands, mult, chosen = r.get("candidates") or [], r.get("multipliers"), r.get("chosen")
        if mult is None or len(mult) != len(cands) or chosen not in cands:
            n_missing += 1
            continue
        m = [float(x) for x in mult]
        top = max(m)
        if all(x == top for x in m):
            continue
        n_info += 1
        n_hit += m[cands.index(chosen)] == top
    usable = n_fly - n_missing
    return dict(rate=n_hit / n_info if n_info else None, n_info=n_info, n_fly_turns=n_fly,
                n_without_multipliers=n_missing, info_frac=n_info / usable if usable else None)


# ---- recorded only: the other spec 4.5 records ---------------------------------------------------------
def learn_curve(result: dict, spec=SPEC, bin_size: int = 10) -> dict | None:
    """Learning-block win curve over the valid flies' learn_records (schedule order; unfinished = loss): the win
    rate at each battle number and in bins of bin_size. None for an arm without a learning block."""
    rows = [r for r in result.get("per_fly") or [] if r.get("learn_records") and not row_invalid(r, spec)]
    if not rows:
        return None
    n = min(len(r["learn_records"]) for r in rows)
    m = np.array([[1.0 if b.get("won") is True else 0.0 for b in r["learn_records"][:n]] for r in rows])
    per = m.mean(axis=0)
    bins = [float(m[:, i:i + bin_size].mean()) for i in range(0, n, bin_size)]
    return dict(per_battle=[float(x) for x in per], bins=bins, bin_size=bin_size, n_flies=len(rows),
                n_battles=n, overall=float(m.mean()))


def rs_mismatch(result: dict) -> dict:
    """RS turn / pulse mismatch per fly (YokedQueue.summary in the row's `yoke`): bundles dropped (donor pulses
    left over), turns with the queue exhausted (RS turns beyond the donor's), residual share; and the totals."""
    per = {}
    for r in result.get("per_fly") or []:
        y = r.get("yoke") or {}
        per[int(r["fly"])] = dict(dropped_bundles=y.get("dropped_bundles"), exhausted_turns=y.get("exhausted_turns"),
                                  delivered_bundles=y.get("delivered_bundles"), n_bundles=y.get("n_bundles"),
                                  residual_frac=r.get("residual_frac"))
    tot = lambda k: sum(v[k] for v in per.values() if _is_number(v[k]))
    return dict(per_fly=per, dropped_bundles=tot("dropped_bundles"), exhausted_turns=tot("exhausted_turns"))


def weight_medians(result: dict) -> dict:
    """fly -> final plastic weight median / w0 (the brain arm's end), as the row carries it."""
    return {int(r["fly"]): r.get("final_weight_median_ratio") for r in result.get("per_fly") or []}


def decision_fraction(records) -> dict:
    """Fly-decided share of all decision records (coach turns and forced switches included in the denominator)."""
    dec = [r for r in records if r.get("kind") == "decision"]
    n_fly = sum(1 for r in dec if r.get("decider") == "fly")
    return dict(n_decisions=len(dec), n_fly=n_fly, frac=n_fly / len(dec) if dec else None)
