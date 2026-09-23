"""Spec J.12's decision rules as pure functions over probe records (J.12.3 statistics and verdicts, J.12.5 thresholds).

A probe record: {"brain": "R" | "N" | "N2" | "noplast", "fly": f, "stage": "pre" | "S1" | "S2", "seed": s,
"counts": {"a": {A type: n, P type: n}, "b": {...}}} — S1 after the reward block, S2 after the punishment block.
Nothing here runs the engine. The code is the authoritative form of J.12's rules (F.5's convention).
"""
from __future__ import annotations

import math

import numpy as np

PASS, FAIL = "PASS", "FAIL"
NOT_CONSTRUCTIBLE = "NOT_CONSTRUCTIBLE"
STOP_MACHINE = "STOP_MACHINE"
INVALID = "INVALID"
STOP = "STOP"                                    # the B test's non-verdict outcome (the user decides)
# ---- the calibration pilot's own stops (J.12.5-2) ------------------------------------------------------------------
NO_EFFECT = "NO_EFFECT"
NOT_SPECIFIC = "NOT_SPECIFIC"
UNDERPOWERED = "UNDERPOWERED"
CALIBRATED = "CALIBRATED"
STAGES = ("pre", "S1", "S2")


def dprime(x) -> float | None:
    """mean / sd (ddof 1). sd = 0 is the limit (F.5): 0.0 if the mean is 0, else +-inf. Fewer than 2 values: None."""
    x = np.asarray(list(x), float)
    if x.size < 2:
        return None
    m, s = float(x.mean()), float(x.std(ddof=1))
    if s == 0.0:
        return 0.0 if m == 0.0 else math.copysign(math.inf, m)
    return m / s


def v_of(counts: dict, spec) -> float:
    """V = z_A - z_P of one odour's readout counts (F.3)."""
    (ma, sa), (mp, sp) = spec.z_a, spec.z_p
    return (counts[spec.a_type] - ma) / sa - (counts[spec.p_type] - mp) / sp


def check_records(records: list, brains: tuple, spec, seeds_of) -> list:
    """INVALID reasons: every (brain, fly, stage, seed) of the declared layout exactly once, finite counts."""
    reasons = []
    keys = [(r["brain"], int(r["fly"]), r["stage"], int(r["seed"])) for r in records]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate probe records")
    want = {(b, f, st, s) for b in brains for f in (range(1) if b == "noplast" else range(spec.n_flies))
            for st in STAGES for s in seeds_of(f)}
    got = set(keys)
    if got != want:
        reasons.append(f"records cover {len(got & want)} of {len(want)} declared cells, {len(got - want)} foreign")
    for r in records:
        vals = [r["counts"][o][t] for o in ("a", "b") for t in (spec.a_type, spec.p_type)]
        if any(not isinstance(v, (int, np.integer)) or v < 0 for v in vals):
            reasons.append(f"non-count readout values in {r['brain']} fly {r['fly']} {r['stage']} seed {r['seed']}")
            break
    return reasons


def _index(records):
    return {(r["brain"], int(r["fly"]), r["stage"], int(r["seed"])): r["counts"] for r in records}


def noplast_ok(records: list) -> bool:
    """The plasticity-off fly's probe counts after both blocks equal its naive counts exactly."""
    idx = _index(records)
    pre = {k[3]: c for k, c in idx.items() if k[0] == "noplast" and k[2] == "pre"}
    return all(idx[("noplast", 0, st, s)] == c for s, c in pre.items() for st in ("S1", "S2"))


def naive_x(records: list, x: str, spec) -> dict:
    """Median naive counts of X's two readout types over every pre probe of the R brains."""
    pre = [r["counts"][x] for r in records if r["brain"] == "R" and r["stage"] == "pre"]
    return {t: float(np.median([c[t] for c in pre])) for t in (spec.a_type, spec.p_type)}


def choose_x(records: list, spec, fixed: str | None) -> str:
    """J.12.1's naive rule: the odour with the larger median naive MBON13 count (tie: spec.x_tie), unless fixed."""
    if fixed is not None:
        return fixed
    pre = [r["counts"] for r in records if r["brain"] == "R" and r["stage"] == "pre"]
    med = {o: float(np.median([c[o][spec.a_type] for c in pre])) for o in ("a", "b")}
    return spec.x_tie if med["a"] == med["b"] else max(med, key=med.get)


def fly_items(records: list, x: str, spec, fly: int, brain: str = "R", ref: str = "N") -> dict | None:
    """The four J.12.3 statistics of one fly, `brain` against its no-DAN reference `ref` (R vs N for the effect,
    N vs N2 for the pilot's null). None when any is undefined."""
    y = "a" if x == "b" else "b"
    idx = _index(records)
    seeds = sorted(s for (b, f, st, s) in idx if b == brain and f == fly and st == "pre")
    V = {(b, st): np.array([[v_of(idx[(b, fly, st, s)][o], spec) for s in seeds] for o in (x, y)])
         for b in (brain, ref) for st in STAGES}
    dv = {k: v[0] - v[1] for k, v in V.items()}
    reward = dprime(dv[(brain, "S1")] - dv[(ref, "S1")])
    punish = dprime((dv[(brain, "S2")] - dv[(brain, "S1")]) - (dv[(ref, "S2")] - dv[(ref, "S1")]))

    def spill(num_y, num_x):
        """Y's R - N change over X's, negatives at 0. X unmoved relative to N (a zero denominator) is +inf: without an
        association there is no specificity to pass (the association items fail too)."""
        den = float(np.mean(num_x))
        return math.inf if den == 0.0 else max(0.0, float(np.mean(num_y)) / den)
    step = lambda b, o: V[(b, "S2")][o] - V[(b, "S1")][o]
    spill_r = spill(V[(brain, "S1")][1] - V[(ref, "S1")][1], V[(brain, "S1")][0] - V[(ref, "S1")][0])
    spill_p = spill(step(brain, 1) - step(ref, 1), step(brain, 0) - step(ref, 0))
    c = {k: float(np.mean(np.where(v[0] > v[1], 1.0, np.where(v[0] == v[1], 0.5, 0.0)))) for k, v in V.items()}
    choice = (c[(brain, "S1")] - c[(brain, "S2")]) - (c[(ref, "S1")] - c[(ref, "S2")])
    out = dict(reward=reward, punish=punish, spill_reward=spill_r, spill_punish=spill_p, choice=choice,
               r=dprime(dv[(brain, "S1")] - dv[(brain, "pre")]), p=dprime(dv[(brain, "S2")] - dv[(brain, "S1")]),
               c_pre=c[(brain, "pre")])
    return None if any(out[k] is None for k in ("reward", "punish", "spill_reward", "spill_punish")) else out


def pair_verdict(records: list, x: str, thresholds: dict, spec) -> dict:
    """J.12.3's pair verdict. thresholds: {"reward": t_R, "punish": t_P, "choice": t_C} (the calibration's)."""
    reasons = check_records(records, ("R", "N", "noplast"), spec, lambda f: sorted(
        {int(r["seed"]) for r in records if r["brain"] == "R" and int(r["fly"]) == f}))
    if reasons:
        return dict(status=INVALID, reasons=reasons)
    if not noplast_ok(records):
        return dict(status=STOP_MACHINE, reasons=["noplast counts moved"])
    nx = naive_x(records, x, spec)
    if min(nx.values()) < spec.floor_spikes:
        return dict(status=NOT_CONSTRUCTIBLE, naive_x=nx)
    flies = {f: fly_items(records, x, spec, f) for f in range(spec.n_flies)}
    valid = {f: it for f, it in flies.items() if it is not None}
    if len(valid) < spec.valid_min:
        return dict(status=INVALID, reasons=[f"{len(valid)} valid flies < {spec.valid_min}"], naive_x=nx)
    med = {k: float(np.median([it[k] for it in valid.values()])) for k in next(iter(valid.values()))}
    checks = dict(reward=med["reward"] >= thresholds["reward"], punish=med["punish"] <= -thresholds["punish"],
                  spill_reward=med["spill_reward"] <= spec.spill_max, spill_punish=med["spill_punish"] <= spec.spill_max,
                  choice=med["choice"] >= thresholds["choice"])
    return dict(status=PASS if all(checks.values()) else FAIL, checks=checks, medians=med, naive_x=nx,
                n_valid=len(valid), flies={str(f): it for f, it in flies.items()})


def b_verdict(exploration: dict, confirmation: dict) -> dict:
    """J.12.3: PASS iff both test pairs PASS; FAIL if either FAILs; otherwise STOP (the user decides)."""
    st = (exploration["status"], confirmation["status"])
    out = PASS if st == (PASS, PASS) else FAIL if FAIL in st else STOP
    return dict(outcome=out, exploration=st[0], confirmation=st[1])


# ================================================================ J.12.5: the calibration pilot
def _boot_median(values, draws, rng) -> np.ndarray:
    v = np.asarray(values, float)
    return np.median(v[rng.integers(0, v.size, size=(draws, v.size))], axis=1)


def threshold(effect: list, null: list, step: float, spec, rng, cap: float) -> dict:
    """The smallest grid t >= 0 with P_null(median >= t) <= null_max; then the power at half the effect (the effect
    values minus half their median, noise kept). `effect` and `null` are signed so that larger is better."""
    nb, hb = _boot_median(null, spec.boot_draws, rng), _boot_median(
        np.asarray(effect, float) - float(np.median(effect)) / 2.0, spec.boot_draws, rng)
    t, n = None, 0
    while n * step <= cap:
        cand = round(n * step, 10)
        if float((nb >= cand).mean()) <= spec.null_max:
            t = cand
            break
        n += 1
    if t is None:
        return dict(t=None, p_null=None, power=None, ok=False)
    power = float((hb >= t).mean())
    return dict(t=t, p_null=float((nb >= t).mean()), power=power, ok=bool(power >= spec.power_min))


def calibrate(records: list, x: str, spec) -> dict:
    """J.12.5: the pilot's own stops, then t_R, t_P, t_C from R vs N (effect) and N vs N2 (null)."""
    reasons = check_records(records, ("R", "N", "N2", "noplast"), spec, lambda f: sorted(
        {int(r["seed"]) for r in records if r["brain"] == "R" and int(r["fly"]) == f}))
    if reasons:
        return dict(status=INVALID, reasons=reasons)
    if not noplast_ok(records):
        return dict(status=STOP_MACHINE)
    nx = naive_x(records, x, spec)
    if min(nx.values()) < spec.floor_spikes:
        return dict(status=NOT_CONSTRUCTIBLE, naive_x=nx)
    eff = [fly_items(records, x, spec, f) for f in range(spec.n_flies)]
    nul = [fly_items(records, x, spec, f, brain="N", ref="N2") for f in range(spec.n_flies)]
    pairs = [(e, n) for e, n in zip(eff, nul) if e is not None and n is not None]
    if len(pairs) < spec.valid_min:
        return dict(status=INVALID, reasons=[f"{len(pairs)} valid flies < {spec.valid_min}"])
    E = {k: [e[k] for e, _ in pairs] for k in ("reward", "punish", "spill_reward", "spill_punish", "choice")}
    N = {k: [n[k] for _, n in pairs] for k in ("reward", "punish", "choice")}
    med = {k: float(np.median(v)) for k, v in E.items()}
    base = dict(naive_x=nx, effect_medians=med, effect=E, null=N, n_valid=len(pairs))
    if med["reward"] <= 0 or med["punish"] >= 0:
        return dict(base, status=NO_EFFECT)
    if med["spill_reward"] > spec.spill_max or med["spill_punish"] > spec.spill_max:
        return dict(base, status=NOT_SPECIFIC)
    rng = np.random.default_rng(spec.boot_seed)
    th = dict(reward=threshold(E["reward"], N["reward"], spec.grid_dprime, spec, rng, spec.grid_max_dprime),
              punish=threshold([-v for v in E["punish"]], [-v for v in N["punish"]], spec.grid_dprime, spec, rng,
                               spec.grid_max_dprime),
              choice=threshold(E["choice"], N["choice"], spec.grid_choice, spec, rng, 2.0))
    ok = all(v["ok"] for v in th.values())
    return dict(base, status=CALIBRATED if ok else UNDERPOWERED, thresholds=th,
                values={k: v["t"] for k, v in th.items()})
