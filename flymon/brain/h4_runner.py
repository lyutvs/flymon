"""M0d H.4 procedure (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a): per combination the readout reselection,
the z constants and the oracle; then the selection over the combinations. H.4a.2 took the APL->MBON ablation of
H.3a.9 (2) out of the verdict, so there is no ablated oracle; its reason travels with the result (SPEC.notes).

    every combination first: reactivity (the H.3 guard measurement) + teachability (M0c arms) -> readout (A, P), z
       |- A or P empty              -> dropped_no_readout: no oracle, not eligible
       '- two types pass in a pool  -> stop_multiple_types: the run stops before any oracle (H.4 does not combine them)
    then every combination with a readout: oracle -> statistics and records
    INVALID rows? -> selection (F_a >= 2; top T_b; tie band; C0 > C1 > C3)

Reselecting every combination before the first oracle makes a reactivity mismatch or a stop surface within minutes,
not after an hour of oracle work.

The measurer is an interface (h4_measure.H4Measurer in a run, a scripted stand-in in the tests).
"""
from __future__ import annotations

import dataclasses

from .h4_rules import (INVALID, READOUT_SELECTED, STOP_MULTI_TYPE, combo_records, combo_stats, pick_readout, reactivity,
                       select, teach_choice, z_constants)


@dataclasses.dataclass
class Context:
    spec: object
    combos: dict                     # {name: adopted Params} in spec.combos order
    pools: dict                      # {"A": [MBON types], "P": [MBON types]}
    probe_seeds: list                # the reference set's seeds, generator order (the rest measurement's seeds)
    expected: list                   # [(axis, turn, x, y)] of the pair list
    h3_guard: dict                   # {name: {type: H.3 guard stats}} from block "h3": reactivity must equal it
    log: object = print


def reselect(m, ctx: Context, name: str, params) -> dict:
    spec = ctx.spec
    ref = m.reference(params)
    rest_by_seed = {r["seed"]: r for r in m.rest(params, ctx.probe_seeds)}
    types = [t for k in ("A", "P") for t in ctx.pools[k]]
    react = reactivity(ref, rest_by_seed, types, spec)
    for t in types:                          # the same measurement as H.3's guard, so the same numbers
        g = ctx.h3_guard[name].get(t)
        if g is None or (react[t]["median_delta"], react[t]["zero_share"]) != (g["median_delta"], g["zero_share"]):
            raise RuntimeError(f"{name} {t}: reactivity {react[t]} differs from the H.3 guard {g}")
    rows = m.teach(params)
    teach = {t: teach_choice(rows, t, "punish_only" if t in ctx.pools["A"] else "reward_only", spec) for t in types}
    pick = pick_readout(react, teach, ctx.pools)
    z = z_constants(ref, pick["readout"], spec.z_ddof) if pick["status"] == READOUT_SELECTED else None
    return dict(reactivity=react, teach=teach, **pick, z=z)


def run_oracle(m, ctx: Context, name: str, params, r: dict) -> dict:
    rows = m.oracle(params, r["readout"], r["z"])
    r["oracle"] = combo_stats(rows, r["z"], ctx.expected, ctx.spec)
    r["records"] = None if r["oracle"]["reasons"] else combo_records(rows, r["z"], ctx.spec)    # none on INVALID rows
    agg = r["oracle"]["aggregate"]
    ctx.log(f"{name} oracle: " + (f"T_b {agg['testable_b']}/{agg['n_b']}  F_a {agg['F_a']}" if agg
                                  else f"INVALID {r['oracle']['reasons']}"))
    return r


def run_h4(m, ctx: Context) -> dict:
    out = dict(combos={}, selection=None, outcome=None)
    for name, params in ctx.combos.items():
        r = out["combos"][name] = reselect(m, ctx, name, params)
        ctx.log(f"{name}: {r['status']} readout {r['readout']} z {r['z']}")
        if r["status"] == STOP_MULTI_TYPE:
            out["outcome"] = STOP_MULTI_TYPE
            return out
    for name, params in ctx.combos.items():
        if out["combos"][name]["status"] == READOUT_SELECTED:
            run_oracle(m, ctx, name, params, out["combos"][name])
    bad = {n: c["oracle"]["reasons"] for n, c in out["combos"].items()
           if c["status"] == READOUT_SELECTED and c["oracle"]["reasons"]}
    if bad:
        out.update(outcome=INVALID, invalid=bad)
        return out
    live = {n: c for n, c in out["combos"].items() if c["status"] == READOUT_SELECTED}
    out["selection"] = select({n: (live[n]["oracle"]["aggregate"] if n in live else None) for n in ctx.combos}, ctx.spec)
    out["outcome"] = out["selection"]["outcome"]
    return out
