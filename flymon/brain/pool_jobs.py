"""Functions that run on a FlyPool worker (`FlyPool.run_jobs`): phase timing, the M0 conditioning arm,
the M0 sparsity seed, the resting baseline with the runaway set, and the peak RSS of the worker. Module-level so the pool can pickle them;
signature fn(engine, plasticity, pops, comps, readout, **kwargs); every job leaves the worker's weights reset."""
from __future__ import annotations

import time

from .conditioning import run_arm
from .measure import chance_jaccard, jaccard, kc_sparsity
from .stimuli import design_odor_pair, present


# ---- worker-side jobs (module-level so the pool can pickle them) --------------------------------
def phase_timing_job(eng, pl, pops, comps, ro, odor, strength: float, steps: int, warm: int,
                     reward_type: str = "PAM08") -> dict:
    """ms per step of the decision phase (odour on, plasticity off) and the reinforcement phase
    (plasticity on, the reward DAN driven). Leaves the worker's weights reset."""
    dt = eng.p.dt
    pl.reset_weights()
    pl.set_enabled(False)
    eng.reset(0)
    eng.clear_drive()
    pl.quiet_dan()
    present(eng, pops, odor, strength)
    eng.run(warm * dt)
    t0 = time.perf_counter()
    eng.run(steps * dt)
    ms_dec = (time.perf_counter() - t0) / steps * 1000.0
    pl.set_enabled(True)
    pl.drive_dan(reward_type, eng.p.dan_drive_mv)
    eng.run(warm * dt)
    t0 = time.perf_counter()
    eng.run(steps * dt)
    ms_rein = (time.perf_counter() - t0) / steps * 1000.0
    pl.quiet_dan()
    pl.reset_weights()
    return {"ms_decision": ms_dec, "ms_reinforce": ms_rein}


def conditioning_arm_job(eng, pl, pops, comps, ro, seed: int, arm: str, strength: float = 0.35, k: int = 8,
                         odor_seed: int = 0, trials: int = 12, present_ms: float = 800.0, settle_ms: float = 800.0,
                         punish_type: str = "PPL105", reward_type: str = "PAM08") -> dict:
    """One (seed, arm) of the M0 conditioning, exactly as scripts/reproduce_flybrain_measurements.py runs it."""
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    return run_arm(eng, pl, pops, ro, a, b, strength, seed, arm, trials=trials, present_ms=present_ms,
                   settle_ms=settle_ms, punish_type=punish_type, reward_type=reward_type)


def sparsity_job(eng, pl, pops, comps, ro, seed: int, strength: float = 0.35, k: int = 8, odor_seed: int = 0) -> dict:
    """One seed of the M0 sparsity row (measure.kc_sparsity for both odours), plasticity off."""
    pl.reset_weights()
    pl.set_enabled(False)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    ra = kc_sparsity(eng, pops, a, strength, seed=seed)
    rb = kc_sparsity(eng, pops, b, strength, seed=seed)
    pl.set_enabled(True)
    return {"frac_active_A": ra["frac_active"], "frac_active_B": rb["frac_active"],
            "jaccard": jaccard(ra["active"], rb["active"]), "chance": chance_jaccard(ra["frac_active"], rb["frac_active"]),
            "mbon_hz_A": ra["mbon_hz"], "mbon_hz_B": rb["mbon_hz"]}


def baseline_job(eng, pl, pops, comps, ro, seed: int, ms: float = 3000.0, sat_hz: float = 100.0) -> dict:
    """Resting MBON rates (measure.mbon_baseline) plus the runaway set: neurons above sat_hz, how many
    are Kenyon cells, and their share of all spikes (spec 5 M0b; STD revisit condition, A.5)."""
    pl.reset_weights()
    pl.set_enabled(False)
    eng.reset(seed)
    eng.clear_drive()
    pl.quiet_dan()
    counts = eng.run(ms)
    pl.set_enabled(True)
    sec = ms / 1000.0
    hz = counts[pops.mbon] / sec
    keep = hz <= sat_hz
    over = (counts / sec) > sat_hz
    types = eng.conn.type[pops.mbon]
    return {"mbon_hz": float(hz.mean()), "mbon_hz_trimmed": float(hz[keep].mean()) if keep.any() else 0.0,
            "n_saturated": int((~keep).sum()), "n_types_active": int(len(set(types[hz > 0].tolist()))),
            "runaway": {"sat_hz": sat_hz, "n_over_sat": int(over.sum()), "n_kc_over_sat": int(over[pops.kc].sum()),
                        "spike_share_over_sat": float(counts[over].sum() / max(int(counts.sum()), 1))}}


def rss_job(eng, pl, pops, comps, ro) -> float:
    """Peak resident set size of this worker process in GB; the engine for the requested wiring variant
    is built before the job runs, so calling it before and after a new variant measures that variant."""
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9      # macOS reports bytes, Linux kB
