"""Paired-seed olfactory conditioning with reversal: the engine acceptance gate (spec 5, M0).

Protocol (reproduction target: flybrain conditioning4c):
  pre-test  : probe CS+ and CS- with the same noise seed (plasticity off)
  training  : N trials of CS+ paired with one DAN type (punishment PPL105 or reward PAM08),
              then CS- paired with the other
  post-test : same probes, same seed -> the no-plasticity arm is exactly 0.0
Readout D = disc over the PPL105 core (approach MBONs) minus disc over the PAM08 core (avoidance MBONs).
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
    a_core: np.ndarray
    p_core: np.ndarray


def disc(x_plus: float, x_minus: float) -> float:
    return float((x_plus - x_minus) / (x_plus + x_minus + 1e-9))


def D(ro: Readout, plus: dict, minus: dict) -> float:
    return disc(plus["A"], minus["A"]) - disc(plus["P"], minus["P"])


def probe(engine: Engine, pl: Plasticity, pops: Populations, ro: Readout, odor, strength: float, seed: int,
          settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    was = pl.enabled
    pl.set_enabled(False)
    engine.reset(seed)
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
            engine.reset(seed * 1000 + t)
            engine.clear_drive()
            pl.quiet_dan()
            present(engine, pops, odor, strength)
            if dan is not None:
                pl.drive_dan(dan, p.dan_drive_mv)
            engine.run(present_ms)
            pl.quiet_dan()
            engine.clear_drive()
            engine.run(gap_ms)
            pl.recover_pulse()


ARMS = {
    # arm: (DAN type paired with the odour in the CS+ slot, DAN type paired with the CS- slot, plasticity on)
    "both": ("PPL105", "PAM08", True),
    "reversed": ("PPL105", "PAM08", True),   # same dopamine, but the two odours swap slots -> sign must flip
    "noplast": ("PPL105", "PAM08", False),
    "punish_only": ("PPL105", None, True),
    "reward_only": (None, "PAM08", True),
}
SWAP_ODOURS = {"reversed"}


def run_arm(engine: Engine, pl: Plasticity, pops: Populations, ro: Readout, cs_plus, cs_minus, strength: float,
            seed: int, arm: str, trials: int = 12, present_ms: float = 800.0, gap_ms: float = 200.0,
            settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    punish, reward, plastic = ARMS[arm]
    if arm in SWAP_ODOURS:
        cs_plus, cs_minus = cs_minus, cs_plus        # reversal = swap which odour gets which channel
    pl.reset_weights()
    pre = D(ro, probe(engine, pl, pops, ro, cs_plus, strength, seed, settle_ms, read_ms),
            probe(engine, pl, pops, ro, cs_minus, strength, seed, settle_ms, read_ms))
    pl.set_enabled(plastic)
    train_block(engine, pl, pops, cs_plus, cs_minus, strength, seed, punish, reward, trials, present_ms, gap_ms)
    pl.set_enabled(True)
    post = D(ro, probe(engine, pl, pops, ro, cs_plus, strength, seed, settle_ms, read_ms),
             probe(engine, pl, pops, ro, cs_minus, strength, seed, settle_ms, read_ms))
    return {"arm": arm, "seed": seed, "D_pre": pre, "D_post": post, "dD": post - pre, "weights_frac": pl.weights_frac()}


def reversal_test(engine, pl, pops, ro, cs_plus, cs_minus, strength, seeds, **kw) -> dict:
    per_seed = {s: {arm: run_arm(engine, pl, pops, ro, cs_plus, cs_minus, strength, s, arm, **kw) for arm in ARMS} for s in seeds}
    return summarise(per_seed)


def summarise(per_seed: dict) -> dict:
    seeds = list(per_seed)
    flips = sum(1 for s in seeds if np.sign(per_seed[s]["both"]["dD"]) == -np.sign(per_seed[s]["reversed"]["dD"])
                and per_seed[s]["both"]["dD"] != 0)
    arms = {arm: {"mean_dD": float(np.mean([per_seed[s][arm]["dD"] for s in seeds])),
                  "sd_dD": float(np.std([per_seed[s][arm]["dD"] for s in seeds])),
                  "mean_weights_frac": float(np.mean([per_seed[s][arm]["weights_frac"] for s in seeds]))}
            for arm in ARMS}
    return {"n_seeds": len(seeds), "n_flip": flips,
            "noplast_max_abs_dD": float(max(abs(per_seed[s]["noplast"]["dD"]) for s in seeds)),
            "arms": arms, "per_seed": {str(s): per_seed[s] for s in seeds}}
