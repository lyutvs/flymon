import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import (D, D_graded, Readout, arms, disc, disc_graded, probe, run_arm,
                                       train_block)
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair


def _setup(synthetic_connectome):
    c = synthetic_connectome()
    # kc_thresh 0.5 so the tiny synthetic olfactory pathway reliably drives Kenyon cells
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=0)
    comps = compartments(c, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    ro = Readout.from_compartments(comps)
    a, b = design_odor_pair(pops, k=2, exclude=("ORN_DA1",))
    return c, pops, eng, pl, ro, a, b


def test_disc_and_D():
    assert disc(3, 1) == pytest.approx(0.5)
    assert disc(0, 0) == pytest.approx(0.0)
    ro = Readout(a_core=np.array([0]), p_core=np.array([1]))
    assert D(ro, {"A": 3, "P": 1}, {"A": 1, "P": 3}) == pytest.approx(0.5 - (-0.5))


def test_disc_graded_is_not_saturated():
    """The classic index pins to -1 as soon as one count is 0; the graded one still reports how
    big the surviving response is relative to the naive total."""
    assert disc(0, 10) == pytest.approx(-1.0)
    assert disc_graded(0, 10, 40) == pytest.approx(-0.25)
    assert disc_graded(0, 40, 40) == pytest.approx(-1.0)
    assert disc_graded(0, 0, 0) == pytest.approx(0.0)       # empty naive total -> no signal, not a divide by zero
    assert disc_graded(5, 5, 0) == pytest.approx(0.0)
    ro = Readout(a_core=np.array([0]), p_core=np.array([1]))
    assert D_graded({"A": 0, "P": 10}, {"A": 10, "P": 0}, 40, 20) == pytest.approx(-0.25 - 0.5)
    assert D(ro, {"A": 0, "P": 10}, {"A": 10, "P": 0}) == pytest.approx(-1.0 - 1.0)


def test_arms_use_requested_types():
    a = arms("PPL101", "PAM08")
    assert a["both"] == ("PPL101", "PAM08", True)
    assert a["reversed"] == ("PAM08", "PPL101", True)      # channels exchanged, not odours
    assert a["noplast"] == ("PPL101", "PAM08", False)
    assert a["punish_only"] == ("PPL101", None, True)
    assert a["reward_only"] == (None, "PAM08", True)
    assert set(a) == set(arms())                            # same five arms whatever the channels
    assert arms()["both"] == ("PPL105", "PAM08", True)      # defaults stay the flybrain pair


def test_readout_from_compartments_rejects_overlap_and_missing(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    comps = compartments(c, pops, Params().core_frac)       # synthetic fixture has PPL105 and PAM08 only
    ro = Readout.from_compartments(comps, "PPL105", "PAM08")
    assert ro.a_core.tolist() == comps["PPL105"].core.tolist()
    assert ro.p_core.tolist() == comps["PAM08"].core.tolist()
    with pytest.raises(ValueError, match="PPL101"):
        Readout.from_compartments(comps, "PPL101", "PAM08")
    with pytest.raises(ValueError, match="PAM11"):
        Readout.from_compartments(comps, "PPL105", "PAM11")
    overlapping = dict(comps)
    overlapping["PAM08"] = dataclasses.replace(comps["PAM08"], core=comps["PPL105"].core)
    with pytest.raises(ValueError, match="overlap"):
        Readout.from_compartments(overlapping, "PPL105", "PAM08")


def test_probe_is_paired_by_seed_and_leaves_weights(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    r1 = probe(eng, pl, pops, ro, a, 1.0, seed=5, settle_ms=50, read_ms=100)
    r2 = probe(eng, pl, pops, ro, a, 1.0, seed=5, settle_ms=50, read_ms=100)
    assert r1 == r2
    assert pl.weights_frac() == pytest.approx(1.0)
    assert pl.enabled is True     # restored after the probe


def test_noplast_arm_gives_exactly_zero(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    r = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=2, arm="noplast", trials=2, present_ms=100, gap_ms=20,
                settle_ms=50, read_ms=100)
    assert r["dD"] == 0.0 and r["dD_disc"] == 0.0 and r["weights_frac"] == pytest.approx(1.0)


def test_run_arm_reports_both_indices_and_counts(synthetic_connectome):
    """Both indices and the raw probe counts they are computed from travel with every arm."""
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    r = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=2, arm="noplast", trials=2, present_ms=100, gap_ms=20,
                settle_ms=50, read_ms=100)
    assert set(r) == {"arm", "seed", "D_pre", "D_post", "dD", "D_pre_disc", "D_post_disc", "dD_disc",
                      "counts", "weights_frac", "w_frac_a_core", "w_frac_p_core"}
    assert r["dD"] == 0.0 and r["dD_disc"] == 0.0           # no plasticity: both indices are exactly 0
    assert set(r["counts"]) == {"pre_plus", "pre_minus", "post_plus", "post_minus"}
    assert all(set(v) == {"A", "P"} for v in r["counts"].values())
    assert r["counts"]["pre_plus"] == r["counts"]["post_plus"]   # same seed, frozen weights
    pre, post = r["counts"]["pre_plus"], r["counts"]["pre_minus"]
    norm_a = pre["A"] + post["A"]
    norm_p = pre["P"] + post["P"]
    assert r["D_pre"] == pytest.approx(D_graded(pre, post, norm_a, norm_p))
    assert r["D_pre_disc"] == pytest.approx(D(ro, pre, post))
    assert r["w_frac_a_core"] == pytest.approx(1.0) and r["w_frac_p_core"] == pytest.approx(1.0)


def test_train_block_changes_weights_when_dan_driven(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish="PPL105", reward="PAM08", trials=2, present_ms=200, gap_ms=20)
    assert pl.weights_frac() < 1.0


def test_settle_window_freezes_weights_but_adapts_baseline(synthetic_connectome):
    """Phasic, not absolute, dopamine teaches: a DAN level the baseline has settled on during the
    settle window depresses far less than a pulse delivered after the settle. Both arms deliver
    dopamine on all four presentations, so only the timing differs."""
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    kw = dict(trials=2, present_ms=300, gap_ms=20, settle_ms=800)   # settle 800 = 4 x da_baseline_ms

    # tonic/endogenous DAN activity: driven before train_block and never quieted, so it is already
    # on during the settle window and the baseline adapts to it (quiet_dan is a no-op here)
    pl.quiet_dan = lambda: None
    pl.drive_dan("PAM08", 70.0)
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish=None, reward=None, **kw)
    wf_tonic = pl.weights_frac()

    # the normal protocol: the same DAN, same number of presentations, driven only after the settle
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish="PAM08", reward="PAM08", **kw)
    wf_phasic = pl.weights_frac()

    assert wf_tonic > wf_phasic
    assert (1.0 - wf_tonic) < 0.5 * (1.0 - wf_phasic)   # adapted-to DAN teaches at most half as much


def test_recovery_applied_once_per_dopamine_pulse(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    calls = []
    orig = pl.recover_pulse
    pl.recover_pulse = lambda: (calls.append(1), orig())

    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish="PPL105", reward=None, trials=2, present_ms=50, gap_ms=10)
    assert len(calls) == 2          # one pulse per trial
    calls.clear()
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish="PPL105", reward="PAM08", trials=2, present_ms=50, gap_ms=10)
    assert len(calls) == 4          # two pulses per trial
    calls.clear()
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish=None, reward=None, trials=2, present_ms=50, gap_ms=10)
    assert len(calls) == 0          # unpaired presentations recover nothing


def test_training_traces_do_not_leak_into_probes(synthetic_connectome):
    """Traces are per-presentation state: a dopamine pulse leaves da > 0, but the next probe
    (same seed, plasticity off) must return exactly the pre-training counts."""
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    before = probe(eng, pl, pops, ro, a, 1.0, seed=7, settle_ms=50, read_ms=100)
    pl.set_enabled(False)          # weights frozen: only the traces can carry anything over
    train_block(eng, pl, pops, a, b, 1.0, seed=7, punish="PPL105", reward="PAM08", trials=2,
                present_ms=200, gap_ms=20)
    assert pl.da.max() > 0.0       # the pulse really did charge the dopamine trace
    pl.set_enabled(True)
    after = probe(eng, pl, pops, ro, a, 1.0, seed=7, settle_ms=50, read_ms=100)
    assert after == before
    assert pl.weights_frac() == pytest.approx(1.0)


def test_reversed_arm_flips_sign_with_odour_specific_learning(synthetic_connectome):
    """Disjoint KC codes make the contingency learnable, so exchanging the dopamine channels
    (same odours, same readout frame) must flip the sign of dD."""
    c = synthetic_connectome(disjoint_kc=True)
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=0)
    comps = compartments(c, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    ro = Readout.from_compartments(comps)
    a = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
    b = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
    # read_ms 600, not 100: with a 100 ms readout the synthetic MBONs emit 0-1 spikes per probe
    # and D is quantised to 0 for both arms (measured: dD_both = 0.0 at every seed)
    kw = dict(trials=3, present_ms=150, gap_ms=20, settle_ms=50, read_ms=600)
    arms = {arm: run_arm(eng, pl, pops, ro, a, b, 1.0, seed=1, arm=arm, **kw)
            for arm in ("noplast", "both", "reversed")}
    assert arms["noplast"]["dD"] == 0.0
    assert arms["both"]["dD"] != 0.0
    assert np.sign(arms["both"]["dD"]) == -np.sign(arms["reversed"]["dD"])


@pytest.mark.skipif(not Path("results/m0/conditioning.json").exists(), reason="gate not yet run on real data")
def test_real_conditioning_gate():
    d = json.loads(Path("results/m0/conditioning.json").read_text())
    assert d["n_seeds"] == 8 and d["n_flip"] == 8
    assert d["noplast_max_abs_dD"] == 0.0
    both, rev = d["arms"]["both"]["mean_dD"], d["arms"]["reversed"]["mean_dD"]
    assert abs(both) >= 0.3 and abs(rev) >= 0.3 and np.sign(both) == -np.sign(rev)
    assert np.sign(d["arms"]["punish_only"]["mean_dD"] + d["arms"]["reward_only"]["mean_dD"]) == np.sign(both)
