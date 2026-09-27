"""Stage 1 (spec 10.6): pick recovery_per_pulse without battles — 1,000 synthetic pulses, the runtime safety population
(median w/w0 over all plastic KC->MBON edges, flymon.agent.swarm.BrainSwarm.median_ratio), path minimum >= 0.5.

Two flies per recovery value: fly 0 ("alt", the selection trajectory) is pulsed on the synthetic odours in turn
(pulse i on odours[i % n]); fly 1 ("same", recorded only) gets every pulse on odours[0]. Both get the same DAN plan
(reward, punish, reward, ...) and the same pulse seeds (taurec_seed_base + i).

Sampling rule: a sample is taken after pulse i (1-based) when i % taurec_sample_every == 0, and after the last pulse
if that pulse was not already sampled; no sample is taken before the first pulse. So P pulses sampled every k give
ceil(P / k) samples (1,000 every 10 -> 100; 6 every 2 -> 3; 5 every 2 -> 3 at pulses 2, 4, 5). The path minimum of a
recovery value is the minimum of fly 0's sampled median ratios.
"""
from __future__ import annotations

import numpy as np

from ..agent.encode import Encoder
from ..battle.pool import POOL
from ..brain import h4_pairs

FLOOR_RATIO = 0.2          # a taught edge counts as floored at w/w0 <= this (Params.min_weight_frac: default and C3 both 0.2),
                           # with a relative tolerance 1e-6: the engine floors in float32, so w/w0 = 0.2 (1 +- ~6e-8)


def synthetic_odours(pops, n: int, seed: int) -> list:
    """n battle-encoder odours: my / opponent species drawn from POOL (with replacement), one of my attacks, hp
    fractions uniform in (0, 1] (1 - U[0, 1)), all from one generator seeded `seed`."""
    enc, rng, out = Encoder(pops), np.random.default_rng(seed), []
    for _ in range(n):
        me, opp = rng.choice(len(POOL), 2, replace=True)
        mon = POOL[int(me)]
        move = mon.attacks[int(rng.integers(len(mon.attacks)))]
        mtype, bp = enc.move_info[move]
        my_hp, opp_hp = 1.0 - float(rng.random()), 1.0 - float(rng.random())
        out.append(h4_pairs.odour(pops, enc.chan, list(enc.species_types[mon.species]),
                                  list(enc.species_types[POOL[int(opp)].species]), mtype, bp, my_hp, opp_hp))
    return out


def pulse_plan(spec) -> list:
    """[(dan, pulse_ms)] of length taurec_pulses, alternating reward / punish, starting with reward."""
    one = [(spec.reward_dan, float(spec.taurec_reward_ms)), (spec.punish_dan, float(spec.taurec_punish_ms))]
    return [one[i % 2] for i in range(spec.taurec_pulses)]


def pre_kc_job(eng, pl, pops, comps, ro):
    """The local KC index (into pops.kc) of every plastic edge, in pl.edges order."""
    return np.asarray(pl.pre_kc).copy()


def taught_mask(pool, odour: dict, spec) -> tuple:
    """(bool mask over the plastic edges, number of active KCs): edges in the reward or punish DAN's core compartment
    whose presynaptic KC spikes at least once in a naive read of `odour` (fly 0, seed taurec_seed_base - 1)."""
    from ..agent.jobs import edge_compartments_job
    kc = _kc_index(pool)
    counts = pool.decide_batch([(0, [odour], spec.taurec_seed_base - 1)], spec.strength, spec.probe_settle_ms,
                               spec.probe_read_ms, idx=kc)[0][0]
    active = np.flatnonzero(np.asarray(counts) > 0)
    comp = pool.run_jobs(edge_compartments_job, [dict(types=[spec.reward_dan, spec.punish_dan])])[0]
    pre_kc = pool.run_jobs(pre_kc_job, [{}])[0]
    mask = (np.asarray(comp[spec.reward_dan]) | np.asarray(comp[spec.punish_dan])) & np.isin(pre_kc, active)
    return mask, int(active.size)


def _kc_index(pool):
    return pool.run_jobs(_kc_job, [{}])[0]


def _kc_job(eng, pl, pops, comps, ro):
    return np.asarray(pops.kc).copy()


def _sample(pool, fly, mask) -> tuple:
    r = pool.w[fly] / pool.w0[pool.flies[fly].shuffle_seed]
    taught = r[mask]
    if taught.size == 0:
        return float(np.median(r)), None, None
    return float(np.median(r)), float(np.quantile(taught, 0.1)), float(np.mean(taught <= FLOOR_RATIO * (1 + 1e-6)))


def trajectory(pool, odours, plan, spec, taught_mask) -> dict:
    """{"alt": {...}, "same": {...}}, each {"ratio", "q10_taught", "floor_frac_taught", "pulse"} sampled by the rule in
    the module docstring. ratio = median w/w0 over all plastic edges (the runtime safety population)."""
    out = {k: {"ratio": [], "q10_taught": [], "floor_frac_taught": [], "pulse": []} for k in ("alt", "same")}
    k = int(spec.taurec_sample_every)
    for i, (dan, ms) in enumerate(plan):
        seed = spec.taurec_seed_base + i
        pool.reinforce_batch([(0, odours[i % len(odours)], dan, ms, seed), (1, odours[0], dan, ms, seed)],
                             spec.strength, spec.train_settle_ms, spec.train_gap_ms)
        if (i + 1) % k == 0 or i + 1 == len(plan):
            for fly, key in ((0, "alt"), (1, "same")):
                med, q10, fl = _sample(pool, fly, taught_mask)
                o = out[key]
                o["ratio"].append(med); o["q10_taught"].append(q10); o["floor_frac_taught"].append(fl)
                o["pulse"].append(i + 1)
    return out


def select(results: dict, spec) -> dict:
    """The smallest recovery value whose alt path minimum is >= median_floor; none -> STOP_NO_RECOVERY."""
    path_min = {str(r): float(min(res["alt"]["ratio"])) for r, res in results.items()}
    ok = sorted(r for r, res in results.items() if min(res["alt"]["ratio"]) >= spec.median_floor)
    if not ok:
        return dict(status="STOP_NO_RECOVERY", recovery_per_pulse=None, path_min=path_min)
    return dict(status="SELECTED", recovery_per_pulse=ok[0], path_min=path_min)
