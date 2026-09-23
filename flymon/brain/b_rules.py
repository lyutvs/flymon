"""Spec J.12's decision rules as pure functions over probe records (J.12.3 statistics and verdicts, J.12.5 thresholds).

A probe record: {"brain": "Rr" | "Rp" | "N" | "N2" | "noplast", "fly": f, "stage": "pre" | "S1", "seed": s,
"counts": {"a": {A type: n, P type: n}, "b": {...}}}. J.12.7: independent arms from naive (F.1 reading b) — Rr gets the
reward block, Rp the punishment block, N (and the pilot's N2) the same presentations without DAN; S1 is after the block.
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
STAGES = ("pre", "S1")
ARMS = ("Rr", "Rp", "N")


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
    """INVALID reasons: every (brain, fly, stage, seed) of the declared layout exactly once, finite counts. seeds_of(f)
    is the declaration (spec.probe_seeds(pair, f)); the noplast brain is fly 0's seeds. A malformed record (a missing
    key, counts without an odour or a readout type) is a reason too, never an exception."""
    for r in records:
        try:
            [(r["brain"], int(r["fly"]), r["stage"], int(r["seed"]))]
            [r["counts"][o][t] for o in ("a", "b") for t in (spec.a_type, spec.p_type)]
        except (KeyError, TypeError, ValueError, IndexError) as e:
            return [f"malformed probe record (missing or bad {e!r}): {str(r)[:120]}"]
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
    """The plasticity-off fly's probe counts after its block equal its naive counts exactly."""
    idx = _index(records)
    pre = {k[3]: c for k, c in idx.items() if k[0] == "noplast" and k[2] == "pre"}
    return all(idx[("noplast", 0, "S1", s)] == c for s, c in pre.items())


def naive_x(records: list, x: str, spec) -> dict:
    """Median naive counts of X's two readout types over every pre probe of the reward-arm brains (all naive brains
    are identical)."""
    pre = [r["counts"][x] for r in records if r["brain"] == "Rr" and r["stage"] == "pre"]
    return {t: float(np.median([c[t] for c in pre])) for t in (spec.a_type, spec.p_type)}


def choose_x(records: list, spec, fixed: str | None) -> str:
    """J.12.1's naive rule: the odour with the larger median naive MBON13 count (tie: spec.x_tie), unless fixed."""
    if fixed is not None:
        return fixed
    pre = [r["counts"] for r in records if r["brain"] == "Rr" and r["stage"] == "pre"]
    med = {o: float(np.median([c[o][spec.a_type] for c in pre])) for o in ("a", "b")}
    return spec.x_tie if med["a"] == med["b"] else max(med, key=med.get)


def fly_items(records: list, x: str, spec, fly: int, reward_arm: str = "Rr", punish_arm: str = "Rp",
              ref: str = "N") -> dict | None:
    """J.12.7's statistics of one fly: each arm against the no-DAN reference `ref` at S1 (Rr, Rp vs N for the effect;
    N, N vs N2 for the pilot's null). None when any judged statistic is undefined."""
    y = "a" if x == "b" else "b"
    idx = _index(records)
    seeds = sorted(s for (b, f, st, s) in idx if b == reward_arm and f == fly and st == "pre")
    V = {(b, st): np.array([[v_of(idx[(b, fly, st, s)][o], spec) for s in seeds] for o in (x, y)])
         for b in {reward_arm, punish_arm, ref} for st in STAGES}
    dv = {k: v[0] - v[1] for k, v in V.items()}
    reward = dprime(dv[(reward_arm, "S1")] - dv[(ref, "S1")])
    punish = dprime(dv[(punish_arm, "S1")] - dv[(ref, "S1")])

    def spill(arm):
        """Y's arm - N change over X's, negatives at 0. X unmoved relative to N (a zero denominator) is +inf: without an
        association there is no specificity to pass (the association items fail too)."""
        num_x = V[(arm, "S1")][0] - V[(ref, "S1")][0]
        num_y = V[(arm, "S1")][1] - V[(ref, "S1")][1]
        den = float(np.mean(num_x))
        return math.inf if den == 0.0 else max(0.0, float(np.mean(num_y)) / den)
    c = {k: float(np.mean(np.where(v[0] > v[1], 1.0, np.where(v[0] == v[1], 0.5, 0.0)))) for k, v in V.items()}
    out = dict(reward=reward, punish=punish, spill_reward=spill(reward_arm), spill_punish=spill(punish_arm),
               choice=c[(ref, "S1")] - c[(punish_arm, "S1")],                   # punishment lowers X's choice vs N
               choice_reward=c[(reward_arm, "S1")] - c[(ref, "S1")],            # recorded only
               r=dprime(dv[(reward_arm, "S1")] - dv[(reward_arm, "pre")]),
               p=dprime(dv[(punish_arm, "S1")] - dv[(punish_arm, "pre")]), c_pre=c[(reward_arm, "pre")])
    return None if any(out[k] is None for k in ("reward", "punish", "spill_reward", "spill_punish")) else out


def recorded(records: list, x: str, spec) -> dict:
    """J.12.3's record-only items: naive d' of V(X) - V(Y) over every reward-arm pre probe (all flies pooled), and the
    share of S1 probes whose taught X cell is at 0 — the P type in the reward arm, the A type in the punishment arm."""
    y = "a" if x == "b" else "b"
    at = lambda b, st: [r["counts"] for r in records if r["brain"] == b and r["stage"] == st]
    share = lambda b, t: float(np.mean([c[x][t] == 0 for c in at(b, "S1")])) if at(b, "S1") else None
    return dict(naive_dprime=dprime([v_of(c[x], spec) - v_of(c[y], spec) for c in at("Rr", "pre")]),
                floor_share_reward=share("Rr", spec.p_type), floor_share_punish=share("Rp", spec.a_type))


def pair_verdict(records: list, x: str, thresholds: dict, spec, pair: str) -> dict:
    """J.12.3's pair verdict. thresholds: {"reward": t_R, "punish": t_P, "choice": t_C} (the calibration's). The records
    must cover exactly `pair`'s declared probe seeds (spec.probe_seeds) and x must be the rule's (choose_x, with the
    pair's fixed X if any)."""
    reasons = check_records(records, ARMS + ("noplast",), spec, lambda f: spec.probe_seeds(pair, f))
    if not reasons and x != choose_x(records, spec, dict(spec.fixed_x).get(pair)):
        reasons.append("X is not the rule's")
    if reasons:
        return dict(status=INVALID, reasons=reasons)
    if not noplast_ok(records):
        return dict(status=STOP_MACHINE, reasons=["noplast counts moved"])
    nx = naive_x(records, x, spec)
    rec = recorded(records, x, spec)
    if min(nx.values()) < spec.floor_spikes:
        return dict(status=NOT_CONSTRUCTIBLE, naive_x=nx, recorded=dict(naive_dprime=rec["naive_dprime"]))
    flies = {f: fly_items(records, x, spec, f) for f in range(spec.n_flies)}
    valid = {f: it for f, it in flies.items() if it is not None}
    if len(valid) < spec.valid_min:
        return dict(status=INVALID, reasons=[f"{len(valid)} valid flies < {spec.valid_min}"], naive_x=nx)
    med = {k: float(np.median([it[k] for it in valid.values()])) for k in next(iter(valid.values()))}
    checks = dict(reward=med["reward"] >= thresholds["reward"], punish=med["punish"] <= -thresholds["punish"],
                  spill_reward=med["spill_reward"] <= spec.spill_max, spill_punish=med["spill_punish"] <= spec.spill_max,
                  choice=med["choice"] >= thresholds["choice"])
    return dict(status=PASS if all(checks.values()) else FAIL, checks=checks, medians=med, naive_x=nx,
                n_valid=len(valid), flies={str(f): it for f, it in flies.items()}, recorded=rec)


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
    values minus half their median, noise kept). `effect` and `null` are signed so that larger is better.
    Non-finite values (d' at its sd = 0 limit) can make a bootstrap median NaN (+inf and -inf averaged): a NaN null
    median counts as reaching every t and a NaN half-effect median as reaching none (both conservative); the counts of
    non-finite inputs are recorded."""
    nf = dict(n_nonfinite_null=int((~np.isfinite(np.asarray(null, float))).sum()),
              n_nonfinite_effect=int((~np.isfinite(np.asarray(effect, float))).sum()))
    with np.errstate(invalid="ignore"):                                 # inf - inf is the NaN handled below
        nb, hb = _boot_median(null, spec.boot_draws, rng), _boot_median(
            np.asarray(effect, float) - float(np.median(effect)) / 2.0, spec.boot_draws, rng)
    p_null = lambda t: float(((nb >= t) | np.isnan(nb)).mean())
    t, n = None, 0
    while n * step <= cap:
        cand = round(n * step, 10)
        if p_null(cand) <= spec.null_max:
            t = cand
            break
        n += 1
    if t is None:
        return dict(t=None, p_null=None, power=None, ok=False, **nf)
    power = float((hb >= t).mean())                                 # NaN >= t is False: a NaN half effect never passes
    return dict(t=t, p_null=p_null(t), power=power, ok=bool(power >= spec.power_min), **nf)


def calibrate(records: list, x: str, spec, pair: str = "calibration") -> dict:
    """J.12.5 / J.12.7: the pilot's own stops, then t_R, t_P, t_C from the arms vs N (effect) and N vs N2 (null). The
    records must cover exactly `pair`'s declared probe seeds."""
    reasons = check_records(records, ARMS + ("N2", "noplast"), spec, lambda f: spec.probe_seeds(pair, f))
    if reasons:
        return dict(status=INVALID, reasons=reasons)
    if not noplast_ok(records):
        return dict(status=STOP_MACHINE)
    nx = naive_x(records, x, spec)
    if min(nx.values()) < spec.floor_spikes:
        return dict(status=NOT_CONSTRUCTIBLE, naive_x=nx)
    eff = [fly_items(records, x, spec, f) for f in range(spec.n_flies)]
    nul = [fly_items(records, x, spec, f, reward_arm="N", punish_arm="N", ref="N2") for f in range(spec.n_flies)]
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
              choice=threshold(E["choice"], N["choice"], spec.grid_choice, spec, rng,
                               spec.grid_max_choice))
    ok = all(v["ok"] for v in th.values())
    return dict(base, status=CALIBRATED if ok else UNDERPOWERED, thresholds=th,
                values={k: v["t"] for k, v in th.items()})
