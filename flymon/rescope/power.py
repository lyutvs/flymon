"""Stage 4 (spec 10.7): pilot variance components -> joint (2b) power -> the cheapest (F, E) -> budget.

Readings R8: normal approximation with variance components (between-fly variance x spec.var_margin, binomial within),
joint (2b) power by Monte Carlo of the two z statistics FLY - COFF and FLY - RS, correlated through FLY's sampling
noise (drawn once per draw and shared), spec.power_draws draws, seed spec.boot_seed. Power is computed at the MIE
alternative: both (2b) differences are spec.mie; (2a)'s power uses the pilot's measured FLY - RND gap and is
recorded only.
"""
from __future__ import annotations

import numpy as np

STATUSES = ("SIZED", "STOP_BUDGET", "STOP_POWER")
GRID_F = (8, 12, 16, 24, 32)
GRID_E = (20, 40, 60, 80, 100, 150, 200, 300)


def components(table: dict) -> dict:
    """Method of moments over fly -> 0/1 win arrays: var_between = max(0, var(per-fly rates) - mean(p_k(1-p_k)/n_k)).
    Needs at least two flies, each with at least one battle (a variance over flies is otherwise undefined)."""
    if len(table) < 2 or any(np.asarray(v).size == 0 for v in table.values()):
        raise ValueError(f"components need >= 2 flies with >= 1 battle each, got {len(table)} flies")
    rates = np.array([np.asarray(v, dtype=float).mean() for v in table.values()])
    ns = np.array([np.asarray(v).size for v in table.values()])
    p = float(rates.mean())
    within = float(np.mean(rates * (1 - rates) / ns))
    return dict(p=p, var_between=float(max(0.0, rates.var(ddof=1) - within)), n_flies=len(rates),
                n_battles=int(ns.mean()))


def _se2(c, F, E, spec) -> float:
    return c["var_between"] * spec.var_margin / F + c["p"] * (1 - c["p"]) / (F * E)


def joint_power(comp: dict, F: int, E: int, spec) -> dict:
    rng = np.random.default_rng(spec.boot_seed)
    n = spec.power_draws
    fly = dict(comp["FLY"], p=min(comp["FLY"]["p"] + spec.mie, 1.0))
    e_fly = rng.normal(0, np.sqrt(_se2(fly, F, E, spec)), n)
    out = {}
    for name in ("COFF", "RS", "RND"):
        other = comp[name]
        e_o = rng.normal(0, np.sqrt(_se2(other, F, E, spec)), n)
        centre = spec.mie if name != "RND" else comp["FLY"]["p"] - other["p"]   # (2b): FLY = other + MIE; (2a): pilot gap
        diff = centre + e_fly - e_o
        se = np.sqrt(_se2(fly, F, E, spec) + _se2(other, F, E, spec))
        out[name] = diff / se > 1.96 if se > 0 else np.full(n, centre > 0)      # no noise at all: the sign decides
    return dict(p_coff=float(out["COFF"].mean()), p_rs=float(out["RS"].mean()),
                p_2b=float((out["COFF"] & out["RS"]).mean()), p_2a=float(out["RND"].mean()))


def wall_hours(F, E, rates, spec) -> float:
    """FLY and RS (serial, after FLY) learn + eval, COFF eval only, RND and MAX eval only on the no-brain rate."""
    b, s = rates["sec_per_fly_battle_brain"], rates["sec_per_battle_nobrain"]
    L = spec.learn_battles
    return (2 * F * (L + E) * b + F * E * b + 2 * F * E * s) / 3600.0


def choose(comp, rates, spec, grid_F=GRID_F, grid_E=GRID_E) -> dict:
    table = []
    for F in grid_F:
        for E in grid_E:
            jp = joint_power(comp, F, E, spec)
            table.append(dict(F=F, E=E, hours=wall_hours(F, E, rates, spec), **jp))
    ok = [r for r in table if r["p_2b"] >= spec.power_target]
    if not ok:
        return dict(status="STOP_POWER", F=None, E=None, hours=None, p_2b=None, table=table)
    best = min(ok, key=lambda r: (r["hours"], r["F"]))
    status = "SIZED" if best["hours"] <= spec.budget_hours else "STOP_BUDGET"
    return dict(status=status, F=best["F"], E=best["E"], hours=best["hours"], p_2b=best["p_2b"], table=table)
