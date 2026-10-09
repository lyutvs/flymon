"""AD.3's learning-check records (record only; the only gate is the 40-battle futility check, Task 8), per arm (FLY-L,
FLY-OS) and point, over the valid (confirm, own) records: the right-direction share (diag.best_contrast > 0), centred V
by multiplier class (with n), the switch curve (arm mean and per fly), the self-noise null sum q*q (diag.empirical_null;
outside point 0 a pooled approximation, AC.11), floor contact and the cumulative pulses, reward / punishment totals per
interval between points (count and ms, per arm and per fly; AD.0 6: the same rule, different realised rewards), the
multiplier share where the power and multiplier picks conflict, and the changed plastic-edge fraction. floor_reading:
the larger arm mean per point; the first point above 0.5 and the last point at or below it (AD.3's pre-registered
reading; not a STOP)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..ac import diag
from . import validity
from .inputs import battle_index
from .spec import LABEL

ARMS = ("FLYL", "FLYOS")


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _recs(sits, arm, p):
    return validity.records(sits, point=p, arm=arm)


def right_share(sits, pairs, points) -> dict:
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for p in points:
            r = _recs(sits, arm, p)
            out[arm][int(p)] = dict(share=validity.share(r, pairs), n=len(r))
    return out


def centred_v(sits, pairs, points) -> dict:
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for p in points:
            acc = defaultdict(list)
            for r in _recs(sits, arm, p).values():
                for s in r["situations"]:
                    pr = pairs[s["pair"]]
                    mults = pr["mults1"] if s["side"] == 0 else pr["mults2"]
                    for v, m in zip(diag.centred(s["v"]), mults):
                        acc[diag.mult_class(m)].append(float(v))
            out[arm][int(p)] = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(acc.items())}
    return out


def switch_curve(sits, points) -> dict:
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for p in points:
            r = _recs(sits, arm, p)
            per = {g: float(np.mean(x["switched"])) for g, x in sorted(r.items())}
            out[arm][int(p)] = dict(mean=_mean(per.values()), per_fly=per)
    return out


def null_curve(sits, pairs, points) -> dict:
    return {arm: {int(p): (diag.empirical_null(list(r.values()), pairs)["mean"] if (r := _recs(sits, arm, p)) else None)
                  for p in points} for arm in ARMS}


def conflict_share(sits, pairs, move_info, points) -> dict:
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for p in points:
            items = []
            for r in _recs(sits, arm, p).values():
                for s in r["situations"]:
                    pr = pairs[s["pair"]]
                    mults = pr["mults1"] if s["side"] == 0 else pr["mults2"]
                    items.append((s["pick"], diag.rule_picks([move_info[c][1] for c in pr["cands"]], mults)))
            c = diag.rule_agreement(items)["conflict"]
            out[arm][int(p)] = dict(n=c["n"], mult_share=c["mult_share"], power_share=c["power_share"])
    return out


def frac_changed(sits, points) -> dict:
    return {arm: {int(p): _mean((r.get("weights") or {}).get("frac_changed") for r in _recs(sits, arm, p).values())
                  for p in points} for arm in ARMS}


def floor_curve(sits, points) -> dict:
    return {arm: {int(p): _mean(r.get("floor_contact") for r in _recs(sits, arm, p).values()) for p in points}
            for arm in ARMS}


def _zero() -> dict:
    return dict(n_reward=0, ms_reward=0.0, n_punish=0, ms_punish=0.0)


def pulse_tables(learn_logs: dict, lay, points, reward: str = "PAM08", punish: str = "PPL105") -> dict:
    pts = sorted(int(p) for p in points)
    names = [f"{a}-{b}" for a, b in zip(pts, pts[1:])]
    by_fly, cum = {}, {}
    for i, recs in learn_logs.items():
        arm, _, g = lay[int(i)]
        by_fly[g] = {n: _zero() for n in names}
        idx = []
        for r in recs:
            if r.get("kind") != "reinforce":
                continue
            b = battle_index(r["battle_id"])
            idx += [b] * len(r.get("pulses") or [])
            for a_, b_, n in zip(pts, pts[1:], names):
                if a_ <= b < b_:
                    for dan, ms in r.get("pulses") or []:
                        key = "reward" if dan == reward else "punish" if dan == punish else None
                        if key:
                            by_fly[g][n][f"n_{key}"] += 1
                            by_fly[g][n][f"ms_{key}"] += float(ms)
        cum[g] = {p: float(sum(1 for b in idx if b < p)) for p in pts}
    by_arm = {}
    cumulative = {}
    for arm in ARMS:
        flies = [g for a, _, g in lay if a == arm and g in by_fly]
        by_arm[arm] = {n: {k: sum(by_fly[g][n][k] for g in flies) for k in _zero()} for n in names}
        cumulative[arm] = {p: _mean(cum[g][p] for g in flies) if flies else None for p in pts}
    return dict(by_arm=by_arm, by_fly=by_fly, cumulative=cumulative)


def floor_reading(floor: dict, read: float) -> dict:
    pts = sorted({int(p) for v in floor.values() for p in v})
    values = {}
    for p in pts:
        xs = [floor[a].get(p) for a in floor if floor[a].get(p) is not None]
        values[p] = max(xs) if xs else None
    over = [p for p in pts if values[p] is not None and values[p] > read]
    first = over[0] if over else None
    ok = [p for p in pts if values[p] is not None and values[p] <= read and (first is None or p < first)]
    return dict(values=values, first_over=first, last_ok=ok[-1] if ok else None, read=float(read))


def learncheck(sits, pairs, move_info, learn_logs, lay, points, read: float) -> dict:
    floor = floor_curve(sits, points)
    return dict(right_share=right_share(sits, pairs, points), centred_v=centred_v(sits, pairs, points),
                switch_curve=switch_curve(sits, points), null=null_curve(sits, pairs, points),
                conflict=conflict_share(sits, pairs, move_info, points), frac_changed=frac_changed(sits, points),
                floor=floor, floor_reading=floor_reading(floor, read), pulses=pulse_tables(learn_logs, lay, points),
                label=LABEL)
