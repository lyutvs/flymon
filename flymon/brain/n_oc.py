"""Spec N.8.6 (decision ⑥): N2.0's operating characteristic of the whole judgement rule (clauses ①-④, bootstrap
included), from the APL-on pilot only.
- The on arm is a resample of the pilot's per-seed Δ, with the same indices for both pairs.
- The block arm has its mean shifted by D and its deviations scaled by each of spec.sd_mults, under each of
  spec.oc_pairings: "same" = built from the on arm's own resample (N.8.6's wording: on and block move together seed by
  seed), "independent" = built from a resample of its own (plan reading 6: no within-seed correlation).
- Null: D_sim = D_dis = 0. Alternative: D_sim = alt_frac x ℓ̂_sim,on, D_dis = 0.
- n = the smallest grid value with null <= null_max and alternative >= power_min in EVERY spread x pairing scenario,
  else None (STOP_POWER). Each scenario's own minimal n is recorded beside it (scenario_n), for N.8b.
- Resolution (a disclosed approximation, plan reading 7): each simulated experiment bootstraps spec.oc_boot draws; the
  judgement itself uses spec.boot_draws. Both numbers are in the output."""
from __future__ import annotations

import math

import numpy as np

from .n_rules import SUPPORTED, n2_stats, scenario_key, verdict

_BLOCK_IDX = {"same": lambda on, off: on, "independent": lambda on, off: off}   # which resample the block arm is built from
HYPOTHESES = ("null", "alt")


def off_arm(values, idx, mu: float, shift: float, mult: float) -> np.ndarray:
    """Block-arm Δ: mean mu + shift (ℓ_off = ℓ_on - D), deviations of the resampled pilot values scaled by mult."""
    v = np.asarray(values, float)[np.asarray(idx, np.int64)]
    return mu + shift + mult * (v - mu)


def _pairing(name: str):
    if name not in _BLOCK_IDX:
        raise ValueError(f"unknown pairing {name!r} (known: {sorted(_BLOCK_IDX)})")
    return _BLOCK_IDX[name]


def arms(sim, dis, mu: dict, on, off, d_sim: float, mult: float, pairing: str) -> dict:
    """One simulated experiment's four Δ arrays. on / off are seed-index resamples; the block arm reads `on` under
    "same" and `off` under "independent". D_dis is 0 under both hypotheses."""
    blk = _pairing(pairing)(on, off)
    return {"sim_on": sim[on], "dis_on": dis[on], "sim_off": off_arm(sim, blk, mu["sim"], d_sim, mult),
            "dis_off": off_arm(dis, blk, mu["dis"], 0.0, mult)}


def simulate(pilot: dict, c1: dict, delta_min: float, eps: float, spec, pairings=None) -> dict:
    """The OC table over pairings (default: every spec.oc_pairings) x spec.sd_mults x {null, alt} x spec.n_grid, and
    the n it implies. Deterministic (spec.oc_seed)."""
    pairings = list(spec.oc_pairings if pairings is None else pairings)
    if not pairings:
        raise ValueError("no pairing to simulate")
    for p in pairings:
        _pairing(p)
    sim, dis = np.asarray(pilot["sim"], float), np.asarray(pilot["dis"], float)
    if sim.ndim != 1 or sim.shape != dis.shape or sim.size < 2 or not (np.isfinite(sim).all() and np.isfinite(dis).all()):
        raise ValueError("the pilot needs the same >= 2 seeds for both pairs, every Δ a number")
    m = sim.size
    mu = {"sim": float(sim.mean()), "dis": float(dis.mean())}
    alt = spec.alt_frac * -mu["sim"]
    rng = np.random.default_rng(spec.oc_seed)
    rows = []
    for pairing in pairings:
        for mult in spec.sd_mults:
            for hyp, d_sim in zip(HYPOTHESES, (0.0, alt)):
                for n in spec.n_grid:
                    wins = 0
                    for _ in range(spec.oc_draws):
                        on, off = rng.integers(0, m, n), rng.integers(0, m, n)
                        d = arms(sim, dis, mu, on, off, d_sim, mult, pairing)
                        st = n2_stats(d, spec, draws=spec.oc_boot, seed=int(rng.integers(2 ** 62)))
                        wins += verdict(st, c1, delta_min, eps)["verdict"] == SUPPORTED
                    p = wins / spec.oc_draws
                    rows.append(dict(pairing=pairing, sd_mult=float(mult), hyp=hyp, n=int(n), p_supported=p,
                                     mc_se=math.sqrt(p * (1 - p) / spec.oc_draws)))
    return dict(rows=rows, n=choose_n(rows, spec, pairings), scenario_n=scenario_n(rows, spec, pairings),
                draws=spec.oc_draws, boot=spec.oc_boot, judge_boot=spec.boot_draws,
                boot_note=(f"each simulated experiment bootstraps {spec.oc_boot} draws; the judgement uses "
                           f"{spec.boot_draws} (an approximation of the judged rule's CIs)"),
                alt_D_sim=alt, pairings=pairings,
                pairing=("same: block arm = the on arm's own resample, mean shifted, deviations scaled (N.8.6); "
                         "independent: block arm = a resample of its own from the pilot pool (plan reading 6)"))


def _cells(rows: list, spec, pairings) -> dict:
    """{(pairing, mult, n): passes} for every declared cell; ValueError if a cell or one of its hypotheses is absent."""
    if pairings is None:
        pairings = sorted({r.get("pairing") for r in rows}, key=str)
    out = {}
    for p in pairings:
        for mult in spec.sd_mults:
            for n in spec.n_grid:
                cell = [r for r in rows if r.get("pairing") == p and r["sd_mult"] == mult and r["n"] == n]
                by = {h: [r["p_supported"] for r in cell if r["hyp"] == h] for h in HYPOTHESES}
                if not all(by.values()):
                    raise ValueError(f"incomplete OC table: pairing {p}, spread {mult:g}x, n {n} lacks "
                                     f"{[h for h in HYPOTHESES if not by[h]]}")
                out[(p, float(mult), int(n))] = bool(all(x <= spec.null_max for x in by["null"])
                                                     and all(x >= spec.power_min for x in by["alt"]))
    return out


def scenario_n(rows: list, spec, pairings=None) -> dict:
    """{scenario_key: that scenario's own smallest passing n, or None} (the record N.8b reports)."""
    ok = _cells(rows, spec, pairings) if rows else {}
    out = {}
    for (p, mult, n), good in ok.items():
        k = scenario_key(p, mult)
        out.setdefault(k, None)
        if good and out[k] is None:
            out[k] = n                                              # spec.n_grid order: the first is the smallest
    return out


def choose_n(rows: list, spec, pairings=None) -> int | None:
    """The smallest grid n at which every scenario passes both hypotheses (when passing is monotone in n, the largest
    of scenario_n). No rows: None. A missing cell: ValueError, never a pass."""
    if not rows:
        return None
    ok = _cells(rows, spec, pairings)
    for n in sorted(spec.n_grid):
        if all(good for (_, _, m), good in ok.items() if m == n):
            return int(n)
    return None
