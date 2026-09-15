"""Paired-seed olfactory conditioning with reversal: the engine acceptance gate (spec 5, M0).

Protocol (reproduction target: flybrain conditioning4c):
  pre-test  : probe CS+ and CS- with the same noise seed `seed` (plasticity off; training uses the
              disjoint seed block 1_000_000 + seed * 1000 + trial so probes never see a training seed)
  training  : N trials of CS+ paired with one DAN type (punishment PPL105, reward PAM08; the
              real-data gate uses the same pair - PPL101 was tried and rejected, see README),
              then CS- paired with the other. The "reversed" arm keeps the odour identities and
              the readout frame fixed and exchanges which dopamine type is paired with which
              odour, so the sign of dD must flip.
  post-test : same probes, same seed -> the no-plasticity arm is exactly 0.0
Readout D = the discrimination index over the punishment type's core (approach MBONs) minus the
same index over the reward type's core (avoidance MBONs). The primary index is the *graded* one
(`disc_graded`: CS+ minus CS- over the naive pre-training total of that set) because the classic
`(x+ - x-)/(x+ + x-)` saturates at +-1 as soon as one count is 0, which in the sparse regime is the
normal case and hides real graded learning. The classic index is still computed and reported as
`*_disc` for comparison.
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
    """The classic (saturating) discrimination index. Kept for comparison only: see `disc_graded`."""
    return float((x_plus - x_minus) / (x_plus + x_minus + 1e-9))


def disc_graded(x_plus: float, x_minus: float, norm: float) -> float:
    """The primary index: the CS+ / CS- difference in units of the *naive* total on the same set.

    `disc` divides by the current total, so it pins to +-1 whenever one count is 0 and then cannot
    move at all. In the sparse regime that is the normal case (the PPL105 core's only responsive
    cell answers exactly one odour: 0 vs 27-41 spikes / 600 ms), so a real, graded depression of the
    responsive odour (A- falling 33 -> 10) leaves disc_A = -1 unchanged - a measurement artefact,
    not absence of learning. Normalising by a fixed pre-training total keeps the index graded.
    """
    return float((x_plus - x_minus) / max(norm, 1))


def D(ro: Readout, plus: dict, minus: dict) -> float:
    return disc(plus["A"], minus["A"]) - disc(plus["P"], minus["P"])


def D_graded(plus: dict, minus: dict, norm_a: float, norm_p: float) -> float:
    """Graded readout D, with `norm_a`/`norm_p` the naive (pre-training) totals of the same arm."""
    return disc_graded(plus["A"], minus["A"], norm_a) - disc_graded(plus["P"], minus["P"], norm_p)


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
                gap_ms: float = 200.0, settle_ms: float = 800.0, on_event=None) -> None:
    """Train `trials` paired presentations of each odour.

    Teaching signal = *phasic* dopamine, i.e. DAN activity above the odour-evoked baseline, not the
    absolute DAN level. DANs (PPL101 in particular) fire endogenously while an odour is on, so each
    presentation starts with a `settle_ms` window: the odour is on, no DAN is driven, the traces and
    `da_base` keep integrating, but plasticity is disabled so the weights are frozen. The default
    800 ms is four `Params.da_baseline_ms` time constants (200 ms), so ~98% of the odour-evoked DAN
    level sits in the baseline before the pulse and only the pulse counts as surprise. Three time
    constants is not enough: at 600 ms an adapted tonic DAN still causes 52% of the depression a
    phasic pulse causes (synthetic fixture), against 22% at 800 ms. Both constants
    matter: with a 1 s baseline time constant the baseline never catches up inside a presentation and
    every arm depresses its punishment compartment non-specifically (README, item D).
    """
    p = engine.p
    for t in range(trials):
        for cs, odor, dan in (("plus", cs_plus, punish), ("minus", cs_minus, reward)):
            engine.reset(1_000_000 + seed * 1000 + t)   # disjoint from the probe seeds
            pl.reset_traces()
            engine.clear_drive()
            pl.quiet_dan()
            present(engine, pops, odor, strength)
            was = pl.enabled                            # settle: baseline adapts, weights frozen
            pl.set_enabled(False)
            engine.run(settle_ms)
            pl.set_enabled(was)
            if dan is not None:
                pl.drive_dan(dan, p.dan_drive_mv)
            engine.run(present_ms)
            pl.quiet_dan()
            engine.clear_drive()
            engine.run(gap_ms)
            if dan is not None:
                pl.recover_pulse()   # spec 3.1: recovery is per dopamine pulse, not per presentation
            if on_event is not None:
                on_event("presentation", trial=t, cs=cs, dan=dan)


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
            settle_ms: float = 800.0, read_ms: float = 600.0,
            punish_type: str = "PPL105", reward_type: str = "PAM08", on_event=None) -> dict:
    """One arm at one seed. `on_event(kind, **fields)`, when given, only observes: "arm_start",
    each "probe" (phase, cs, A, P), each "presentation" (trial, cs, dan) and "arm_end" (the result).

    "D_pre"/"D_post"/"dD" are the *graded* index (`D_graded`), normalised by this arm's own naive
    pre-training totals; the saturating classic index is kept alongside as "D_pre_disc"/"D_post_disc"/
    "dD_disc", and the four raw probe count pairs under "counts"."""
    emit = on_event or (lambda kind, **fields: None)

    def probes(phase: str) -> dict:
        counts = {cs: probe(engine, pl, pops, ro, odor, strength, seed, settle_ms, read_ms)
                  for cs, odor in (("plus", cs_plus), ("minus", cs_minus))}
        for cs, c in counts.items():
            emit("probe", phase=phase, cs=cs, **c)
        return counts

    punish, reward, plastic = arms(punish_type, reward_type)[arm]
    emit("arm_start", arm=arm, seed=seed)
    pl.reset_weights()
    pre = probes("pre")
    pl.set_enabled(plastic)
    train_block(engine, pl, pops, cs_plus, cs_minus, strength, seed, punish, reward, trials, present_ms, gap_ms,
                settle_ms, on_event=on_event)
    pl.set_enabled(True)
    post = probes("post")
    # the naive totals of this arm and seed: a fixed yardstick, so post-training change is graded
    norm_a = pre["plus"]["A"] + pre["minus"]["A"]
    norm_p = pre["plus"]["P"] + pre["minus"]["P"]
    g_pre = D_graded(pre["plus"], pre["minus"], norm_a, norm_p)
    g_post = D_graded(post["plus"], post["minus"], norm_a, norm_p)
    d_pre = D(ro, pre["plus"], pre["minus"])
    d_post = D(ro, post["plus"], post["minus"])
    result = {"arm": arm, "seed": seed,
              "D_pre": g_pre, "D_post": g_post, "dD": g_post - g_pre,
              "D_pre_disc": d_pre, "D_post_disc": d_post, "dD_disc": d_post - d_pre,
              "counts": {"pre_plus": pre["plus"], "pre_minus": pre["minus"],
                         "post_plus": post["plus"], "post_minus": post["minus"]},
              "weights_frac": pl.weights_frac(),
              "w_frac_a_core": pl.weights_frac_by_mbon_set(ro.a_core),
              "w_frac_p_core": pl.weights_frac_by_mbon_set(ro.p_core)}
    emit("arm_end", **result)
    return result


def reversal_test(engine, pl, pops, ro, cs_plus, cs_minus, strength, seeds,
                  punish_type: str = "PPL105", reward_type: str = "PAM08", **kw) -> dict:
    """Run all five arms at every seed for the chosen dopamine channels.

    The readout frame follows the channels: the approach set is the punishment type's core and the
    avoidance set is the reward type's core (`Readout.from_compartments`). The real-data gate keeps
    PPL105/PAM08: PPL105 is endogenously quiet and odour-specific, while PPL101 fires tonically
    (~119 Hz at rest) and carries no phasic contrast - see README ("우리가 정한 것").
    """
    names = arms(punish_type, reward_type)
    per_seed = {s: {arm: run_arm(engine, pl, pops, ro, cs_plus, cs_minus, strength, s, arm,
                                 punish_type=punish_type, reward_type=reward_type, **kw)
                    for arm in names} for s in seeds}
    return summarise(per_seed)


def summarise(per_seed: dict) -> dict:
    """Gate statistics on the graded `dD`, with the saturating index reported alongside
    (`n_flip_disc`, per-arm `mean_dD_disc`) so the two can be compared run to run."""
    seeds = list(per_seed)

    def n_flip(key: str) -> int:
        return sum(1 for s in seeds
                   if np.sign(per_seed[s]["both"][key]) == -np.sign(per_seed[s]["reversed"][key])
                   and per_seed[s]["both"][key] != 0)

    # arm names do not depend on the channels, so the default ARMS keys are the full set
    per_arm = {arm: {"mean_dD": float(np.mean([per_seed[s][arm]["dD"] for s in seeds])),
                     "sd_dD": float(np.std([per_seed[s][arm]["dD"] for s in seeds])),
                     "mean_dD_disc": float(np.mean([per_seed[s][arm]["dD_disc"] for s in seeds])),
                     "mean_weights_frac": float(np.mean([per_seed[s][arm]["weights_frac"] for s in seeds]))}
               for arm in ARMS}
    return {"n_seeds": len(seeds), "n_flip": n_flip("dD"), "n_flip_disc": n_flip("dD_disc"),
            "noplast_max_abs_dD": float(max(abs(per_seed[s]["noplast"]["dD"]) for s in seeds)),
            "arms": per_arm, "per_seed": {str(s): per_seed[s] for s in seeds}}


def _drop(pre: float, post: float) -> float:
    """Fraction of a naive probe response lost after training (0 when there was nothing to lose)."""
    return 0.0 if pre == 0 else (pre - post) / pre


def channel_specific_seeds(per_seed: dict) -> int:
    """Seeds where both dopamine channels depress the odour they were actually paired with.

    Read off the raw probe counts, not the composite index. Per seed, all three must hold:
      - `both` (reward PAM08 on the CS-): the approach core's minus-odour response drops more
        than its plus-odour response;
      - `reversed` (reward on the CS+): the other way round;
      - `reversed` (punishment PPL105 on the CS-): the aversive core's minus-odour response drops
        more than it does in `both`, where the same channel was paired with the other odour.
    """
    n = 0
    for rec in per_seed.values():
        both, rev = rec["both"]["counts"], rec["reversed"]["counts"]
        p_minus_both = _drop(both["pre_minus"]["P"], both["post_minus"]["P"])
        p_plus_both = _drop(both["pre_plus"]["P"], both["post_plus"]["P"])
        p_plus_rev = _drop(rev["pre_plus"]["P"], rev["post_plus"]["P"])
        p_minus_rev = _drop(rev["pre_minus"]["P"], rev["post_minus"]["P"])
        a_minus_both = _drop(both["pre_minus"]["A"], both["post_minus"]["A"])
        a_minus_rev = _drop(rev["pre_minus"]["A"], rev["post_minus"]["A"])
        if p_minus_both > p_plus_both and p_plus_rev > p_minus_rev and a_minus_rev > a_minus_both:
            n += 1
    return n
