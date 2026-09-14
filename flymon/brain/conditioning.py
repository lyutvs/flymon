"""Paired-seed olfactory conditioning with reversal: the engine acceptance gate (spec 5, M0).

Protocol (reproduction target: flybrain conditioning4c):
  pre-test  : probe CS+ and CS- with the same noise seed `seed` (plasticity off; training uses the
              disjoint seed block 1_000_000 + seed * 1000 + trial so probes never see a training seed)
  training  : N trials of CS+ paired with one DAN type (punishment, default PPL105; reward,
              default PAM08 - the real-data gate uses PPL101 for punishment, see README),
              then CS- paired with the other. The "reversed" arm keeps the odour identities and
              the readout frame fixed and exchanges which dopamine type is paired with which
              odour, so the sign of dD must flip.
  post-test : same probes, same seed -> the no-plasticity arm is exactly 0.0
Readout D = disc over the punishment type's core (approach MBONs) minus disc over the reward type's
core (avoidance MBONs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .plasticity import Plasticity
from .stimuli import present


@dataclass
class Readout:
    """Approach/avoidance readout frame: `a_core` is the punishment type's core compartment
    (approach MBONs), `p_core` the reward type's core (avoidance MBONs)."""
    a_core: np.ndarray
    p_core: np.ndarray

    @classmethod
    def from_compartments(cls, comps: dict, punish_type: str = "PPL105",
                          reward_type: str = "PAM08") -> "Readout":
        missing = [t for t in (punish_type, reward_type) if t not in comps]
        if missing:
            raise ValueError(f"missing dopamine compartments {missing}: the readout needs both "
                             f"{punish_type} and {reward_type}")
        a_core, p_core = comps[punish_type].core, comps[reward_type].core
        overlap = set(a_core.tolist()) & set(p_core.tolist())
        if overlap:
            raise ValueError(f"{punish_type} and {reward_type} core compartments overlap on "
                             f"{len(overlap)} MBONs: the readout would not separate approach from avoidance")
        return cls(a_core=a_core, p_core=p_core)


def disc(x_plus: float, x_minus: float) -> float:
    return float((x_plus - x_minus) / (x_plus + x_minus + 1e-9))


def D(ro: Readout, plus: dict, minus: dict) -> float:
    return disc(plus["A"], minus["A"]) - disc(plus["P"], minus["P"])


def probe(engine: Engine, pl: Plasticity, pops: Populations, ro: Readout, odor, strength: float, seed: int,
          settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    was = pl.enabled
    pl.set_enabled(False)
    engine.reset(seed)
    pl.reset_traces()
    engine.clear_drive()
    pl.quiet_dan()
    present(engine, pops, odor, strength)
    engine.run(settle_ms)
    counts = engine.run(read_ms)
    pl.set_enabled(was)
    return {"A": int(counts[ro.a_core].sum()), "P": int(counts[ro.p_core].sum())}


def train_block(engine: Engine, pl: Plasticity, pops: Populations, cs_plus, cs_minus, strength: float, seed: int,
                punish: str | None, reward: str | None, trials: int = 12, present_ms: float = 800.0,
                gap_ms: float = 200.0) -> None:
    p = engine.p
    for t in range(trials):
        for odor, dan in ((cs_plus, punish), (cs_minus, reward)):
            engine.reset(1_000_000 + seed * 1000 + t)   # disjoint from the probe seeds
            pl.reset_traces()
            engine.clear_drive()
            pl.quiet_dan()
            present(engine, pops, odor, strength)
            if dan is not None:
                pl.drive_dan(dan, p.dan_drive_mv)
            engine.run(present_ms)
            pl.quiet_dan()
            engine.clear_drive()
            engine.run(gap_ms)
            if dan is not None:
                pl.recover_pulse()   # spec 3.1: recovery is per dopamine pulse, not per presentation


def arms(punish_type: str = "PPL105", reward_type: str = "PAM08") -> dict:
    """The five protocol arms for a chosen pair of dopamine channels.

    arm -> (DAN type paired with the odour in the CS+ slot, DAN type paired with the CS- slot,
    plasticity on). "reversed" keeps the odours and the readout frame and exchanges the channels,
    so the sign of dD must flip.
    """
    return {
        "both": (punish_type, reward_type, True),
        "reversed": (reward_type, punish_type, True),
        "noplast": (punish_type, reward_type, False),
        "punish_only": (punish_type, None, True),
        "reward_only": (None, reward_type, True),
    }


ARMS = arms()   # default channels, kept module-level for backward compatibility


def run_arm(engine: Engine, pl: Plasticity, pops: Populations, ro: Readout, cs_plus, cs_minus, strength: float,
            seed: int, arm: str, trials: int = 12, present_ms: float = 800.0, gap_ms: float = 200.0,
            settle_ms: float = 200.0, read_ms: float = 600.0,
            punish_type: str = "PPL105", reward_type: str = "PAM08") -> dict:
    punish, reward, plastic = arms(punish_type, reward_type)[arm]
    pl.reset_weights()
    pre = D(ro, probe(engine, pl, pops, ro, cs_plus, strength, seed, settle_ms, read_ms),
            probe(engine, pl, pops, ro, cs_minus, strength, seed, settle_ms, read_ms))
    pl.set_enabled(plastic)
    train_block(engine, pl, pops, cs_plus, cs_minus, strength, seed, punish, reward, trials, present_ms, gap_ms)
    pl.set_enabled(True)
    post = D(ro, probe(engine, pl, pops, ro, cs_plus, strength, seed, settle_ms, read_ms),
             probe(engine, pl, pops, ro, cs_minus, strength, seed, settle_ms, read_ms))
    return {"arm": arm, "seed": seed, "D_pre": pre, "D_post": post, "dD": post - pre, "weights_frac": pl.weights_frac()}


def reversal_test(engine, pl, pops, ro, cs_plus, cs_minus, strength, seeds,
                  punish_type: str = "PPL105", reward_type: str = "PAM08", **kw) -> dict:
    """Run all five arms at every seed for the chosen dopamine channels.

    The readout frame follows the channels: the approach set is the punishment type's core and the
    avoidance set is the reward type's core (`Readout.from_compartments`). On real data the gate uses
    PPL101 (gamma1pedc, core MBON11) as the punishment channel because the PPL105 core is
    odour-selective or silent in our engine - see README ("우리가 정한 것").
    """
    names = arms(punish_type, reward_type)
    per_seed = {s: {arm: run_arm(engine, pl, pops, ro, cs_plus, cs_minus, strength, s, arm,
                                 punish_type=punish_type, reward_type=reward_type, **kw)
                    for arm in names} for s in seeds}
    return summarise(per_seed)


def summarise(per_seed: dict) -> dict:
    seeds = list(per_seed)
    flips = sum(1 for s in seeds if np.sign(per_seed[s]["both"]["dD"]) == -np.sign(per_seed[s]["reversed"]["dD"])
                and per_seed[s]["both"]["dD"] != 0)
    # arm names do not depend on the channels, so the default ARMS keys are the full set
    per_arm = {arm: {"mean_dD": float(np.mean([per_seed[s][arm]["dD"] for s in seeds])),
                     "sd_dD": float(np.std([per_seed[s][arm]["dD"] for s in seeds])),
                     "mean_weights_frac": float(np.mean([per_seed[s][arm]["weights_frac"] for s in seeds]))}
               for arm in ARMS}
    return {"n_seeds": len(seeds), "n_flip": flips,
            "noplast_max_abs_dD": float(max(abs(per_seed[s]["noplast"]["dD"]) for s in seeds)),
            "arms": per_arm, "per_seed": {str(s): per_seed[s] for s in seeds}}
