"""Spec J.12.4: the B test's judge, verified before any run — fixtures with a known answer and mutations that must
change a verdict (a surviving mutation means the rule it removes is not tested)."""
import dataclasses
import math

import numpy as np
import pytest

from flymon.brain import b_rules as B
from flymon.brain.b_spec import SPEC

from b_fixtures import records

TH = {"reward": 1.0, "punish": 1.0, "choice": 0.25}


def verdict(recs, x="b", th=TH, spec=SPEC, pair="calibration"):
    return B.pair_verdict(recs, x, th, spec, pair)


def test_spec_is_j12():
    assert dict(SPEC.pair_seeds) == {"calibration": 23, "exploration": 0, "confirmation": 7}
    assert dict(SPEC.fixed_x) == {"exploration": "b"}
    assert (SPEC.n_flies, SPEC.n_probe, SPEC.trials, SPEC.pulse_ms, SPEC.reward_dan, SPEC.punish_dan) == \
        (8, 8, 20, 400.0, "PAM08", "PPL105")
    assert (SPEC.z_a, SPEC.z_p) == ((21.8293, 18.1032), (40.6433, 24.0891))
    assert (SPEC.floor_spikes, SPEC.valid_min, SPEC.spill_max, SPEC.null_max, SPEC.power_min) == (5.0, 6, 0.5, 0.05, 0.9)
    assert (SPEC.grid_dprime, SPEC.grid_choice, SPEC.grid_max_dprime, SPEC.grid_max_choice) == (0.05, 0.125, 50.0, 2.0)
    assert SPEC.probe_seeds("exploration", 2)[:2] == [410_200, 410_201]
    assert SPEC.train_seed("confirmation", 1, 3) == 4_201_003 and SPEC.train_seed("calibration", 1, 3, True) == 4_501_003


def test_dprime_keeps_f5_s_limits():
    assert B.dprime([1.0]) is None and B.dprime([0.0, 0.0]) == 0.0 and B.dprime([2.0, 2.0]) == math.inf
    assert math.isclose(B.dprime([1.0, 3.0]), 2.0 / math.sqrt(2.0))


# ---- fixtures with a known answer ----------------------------------------------------------------------------------------
def test_specific_associative_learning_passes():
    """Reward 10 spikes of MBON05(X), then punishment 25 of MBON13(X): X's value falls below Y's, so the choice flips."""
    v = verdict(records(reward=10, punish=25))
    assert v["status"] == B.PASS and all(v["checks"].values()) and v["n_valid"] == 8


def test_presentation_drift_without_dan_fails():
    """F.6's first path: X-only drift passes spec 5's literal level test; the paired no-DAN contrast stops it."""
    v = verdict(records(drift_r=20, drift_p=20))
    assert v["status"] == B.FAIL and not v["checks"]["reward"] and not v["checks"]["punish"]
    assert v["medians"]["r"] > 1                                        # the uncontrasted change would have passed


def test_full_generalisation_fails_on_y_specificity():
    v = verdict(records(reward=10, punish=25, spill=1.0))
    assert v["status"] == B.FAIL and not v["checks"]["spill_reward"] and not v["checks"]["spill_punish"]


def test_reward_only_fails():
    v = verdict(records(reward=10, punish=0))
    assert v["status"] == B.FAIL and v["checks"]["reward"] and not v["checks"]["punish"]


def test_a_punishment_that_does_not_change_the_choice_fails():
    """X far above Y naive (V gap ~ 3.5 z) and a small, consistent punishment: association passes, choice does not."""
    v = verdict(records(naive=dict(ax=90, px=10, ay=10, py=40), reward=5, punish=6, noise=1.0))
    assert v["checks"]["punish"] and not v["checks"]["choice"] and v["status"] == B.FAIL


def test_a_floor_readout_on_x_is_not_constructible():
    """X is the rule's (its naive MBON13 median 2 beats Y's 0) and sits below the floor."""
    v = verdict(records(naive=dict(ax=2, px=40, ay=0, py=40), reward=20, punish=20, noise=0.5))
    assert v["status"] == B.NOT_CONSTRUCTIBLE and v["naive_x"][SPEC.a_type] < SPEC.floor_spikes
    assert set(v["recorded"]) == {"naive_dprime"}


def test_a_noplast_count_that_moved_stops_the_machine():
    assert verdict(records(reward=10, punish=25, noplast_move=True))["status"] == B.STOP_MACHINE


def test_a_missing_or_duplicate_record_is_invalid():
    recs = records(reward=10, punish=25)
    assert verdict(recs[1:])["status"] == B.INVALID
    assert verdict(recs + recs[:1])["status"] == B.INVALID


def test_a_fly_missing_in_every_brain_is_invalid():
    assert verdict([r for r in records(reward=10, punish=25) if r["fly"] != 7])["status"] == B.INVALID


def test_another_pairs_seeds_are_invalid():
    recs = records(reward=10, punish=25, pair="exploration")
    assert verdict(recs, pair="exploration")["status"] == B.PASS         # the same records under their own pair
    assert verdict(recs)["status"] == B.INVALID


def test_one_seed_missing_in_every_brain_of_a_fly_is_invalid():
    s = SPEC.probe_seeds("calibration", 3)[2]
    assert verdict([r for r in records(reward=10, punish=25) if not (r["fly"] == 3 and r["seed"] == s)])["status"] \
        == B.INVALID


def test_an_x_that_is_not_the_rules_is_invalid():
    recs = records(reward=10, punish=25, pair="exploration")          # the exploration pair's X is fixed to b
    v = verdict(recs, x="a", pair="exploration")
    assert v["status"] == B.INVALID and v["reasons"] == ["X is not the rule's"]


def test_the_recorded_items_on_a_hand_checkable_pair():
    """One fly, two probe seeds. R pre: X MBON13 30 / 40, Y 20 / 20, MBON05 40 everywhere, so dV = 10 / sA, 20 / sA and
    d' = 15 / (10 / sqrt 2) = 1.5 sqrt 2. R S1: X MBON05 0 on one seed of two (1/2); R S2: X MBON13 0 on both (1)."""
    spec = dataclasses.replace(SPEC, n_flies=1, n_probe=2, valid_min=1)
    s0, s1 = spec.probe_seeds("calibration", 0)
    A, P = spec.a_type, spec.p_type
    pre = {s0: {"b": {A: 30, P: 40}, "a": {A: 20, P: 40}}, s1: {"b": {A: 40, P: 40}, "a": {A: 20, P: 40}}}
    R = {"pre": pre, "S1": {s0: {"b": {A: 30, P: 0}, "a": {A: 20, P: 40}}, s1: pre[s1]},
         "S2": {s: {"b": {A: 0, P: 0}, "a": {A: 20, P: 40}} for s in (s0, s1)}}
    recs = [dict(brain=b, fly=0, stage=st, seed=s, counts=(R[st][s] if b == "R" else pre[s]))
            for b in ("R", "N", "noplast") for st in B.STAGES for s in (s0, s1)]
    v = verdict(recs, spec=spec)
    assert v["status"] in (B.PASS, B.FAIL)
    assert math.isclose(v["recorded"]["naive_dprime"], 1.5 * math.sqrt(2.0))
    assert v["recorded"]["floor_share_s1"] == 0.5 and v["recorded"]["floor_share_s2"] == 1.0


def test_too_few_valid_flies_is_invalid(monkeypatch):
    """A fly is invalid when a statistic is undefined (d' on fewer than 2 probe seeds); three leave 5 < 6 valid."""
    orig = B.fly_items
    monkeypatch.setattr(B, "fly_items", lambda recs, x, spec, f, **kw: None if f < 3 else orig(recs, x, spec, f, **kw))
    v = verdict(records(reward=10, punish=25))
    assert v["status"] == B.INVALID and "valid flies" in v["reasons"][0]


def test_no_association_at_all_fails_rather_than_invalidates():
    """X unmoved relative to N: the Y ratio is +inf (not specific), the fly stays valid and the pair FAILs."""
    v = verdict(records(brain_noise=0.0))
    assert v["status"] == B.FAIL and v["n_valid"] == 8 and v["medians"]["spill_reward"] == math.inf


def test_the_b_verdict_needs_both_test_pairs():
    P, F, N = ({"status": s} for s in (B.PASS, B.FAIL, B.NOT_CONSTRUCTIBLE))
    assert B.b_verdict(P, P)["outcome"] == B.PASS
    assert B.b_verdict(P, F)["outcome"] == B.FAIL and B.b_verdict(F, N)["outcome"] == B.FAIL
    assert B.b_verdict(P, N)["outcome"] == B.STOP


def test_choose_x_is_the_larger_naive_a_type_median_and_a_fixed_x_wins():
    recs = records(naive=dict(ax=30, px=40, ay=20, py=40))                  # x = b carries the larger MBON13
    assert B.choose_x(recs, SPEC, None) == "b" and B.choose_x(recs, SPEC, "a") == "a"
    recs = records(x="a", naive=dict(ax=30, px=40, ay=20, py=40))
    assert B.choose_x(recs, SPEC, None) == "a"


# ---- mutations: each must turn a fixture's verdict --------------------------------------------------------------------------
def test_mutation_without_the_no_dan_contrast_lets_drift_pass(monkeypatch):
    orig = B.fly_items

    def no_contrast(*a, **kw):
        out = orig(*a, **kw)
        return None if out is None else dict(out, reward=out["r"], punish=out["p"])
    monkeypatch.setattr(B, "fly_items", no_contrast)
    assert verdict(records(drift_r=20, drift_p=20), th={**TH, "choice": -1.0})["status"] == B.PASS


def test_mutation_with_a_flipped_sign_fails_real_learning(monkeypatch):
    monkeypatch.setattr(B, "v_of", lambda c, s: -((c[s.a_type] - s.z_a[0]) / s.z_a[1] - (c[s.p_type] - s.z_p[0]) / s.z_p[1]))
    assert verdict(records(reward=10, punish=25))["status"] == B.FAIL


def test_partial_generalisation_fails_on_y_specificity_alone():
    """Y takes 70 % of X's loss: the associations still pass (30 % remains, consistently), Y specificity does not."""
    v = verdict(records(reward=20, punish=25, spill=0.7))
    assert v["checks"]["reward"] and v["checks"]["punish"] and not v["checks"]["spill_reward"] and v["status"] == B.FAIL


def test_mutation_without_y_specificity_lets_generalisation_pass():
    assert verdict(records(reward=20, punish=25, spill=0.7), spec=dataclasses.replace(SPEC, spill_max=math.inf),
                   th={**TH, "choice": -1.0})["status"] == B.PASS


def test_a_failure_of_either_test_pair_fails_b():
    """Dropping either pair from b_verdict turns one of these into a PASS."""
    assert B.b_verdict({"status": B.PASS}, {"status": B.FAIL})["outcome"] == B.FAIL
    assert B.b_verdict({"status": B.FAIL}, {"status": B.PASS})["outcome"] == B.FAIL


def test_mutation_without_the_noplast_check_passes_a_moved_machine(monkeypatch):
    monkeypatch.setattr(B, "noplast_ok", lambda recs: True)
    assert verdict(records(reward=10, punish=25, noplast_move=True))["status"] == B.PASS


def test_mutation_that_reverts_the_sd_zero_limit_loses_a_noiseless_pass(monkeypatch):
    recs = records(reward=10, punish=25, noise=0.0, brain_noise=0.0)
    assert verdict(recs)["status"] == B.PASS                                  # +-inf d' from constant differences
    orig = B.dprime
    monkeypatch.setattr(B, "dprime", lambda x: None if np.std(np.asarray(list(x), float)) == 0 else orig(x))
    assert verdict(recs)["status"] == B.INVALID


def test_mutation_without_the_choice_item_passes_an_unchanged_choice():
    recs = records(naive=dict(ax=90, px=10, ay=10, py=40), reward=5, punish=6, noise=1.0)
    assert verdict(recs, th={**TH, "choice": -1.0})["status"] == B.PASS


# ---- J.12.5: the calibration pilot's rule ------------------------------------------------------------------------------------
def test_a_learning_pilot_calibrates_and_its_thresholds_meet_both_targets():
    c = B.calibrate(records(reward=10, punish=25, n2=True, noise=4.0), "b", SPEC)
    assert c["status"] == B.CALIBRATED
    for k, th in c["thresholds"].items():
        assert th["p_null"] <= SPEC.null_max and th["power"] >= SPEC.power_min and th["t"] >= 0, k
    again = B.calibrate(records(reward=10, punish=25, n2=True, noise=4.0), "b", SPEC)
    assert again["values"] == c["values"]                                    # the bootstrap seed is fixed


def test_the_pilot_stops_without_an_effect_without_specificity_and_when_underpowered():
    assert B.calibrate(records(n2=True), "b", SPEC)["status"] == B.NO_EFFECT
    assert B.calibrate(records(reward=10, punish=25, spill=1.0, n2=True), "b", SPEC)["status"] == B.NO_EFFECT
    assert B.calibrate(records(reward=20, punish=25, spill=0.7, n2=True), "b", SPEC)["status"] == B.NOT_SPECIFIC
    weak = B.calibrate(records(reward=1, punish=1, n2=True, noise=6.0, seed=3), "b", SPEC)
    assert weak["status"] in (B.UNDERPOWERED, B.NO_EFFECT, B.NOT_SPECIFIC)          # a weak pilot never calibrates
    assert B.calibrate(records(reward=10, punish=25, n2=True, noplast_move=True), "b", SPEC)["status"] == B.STOP_MACHINE


def test_threshold_is_the_smallest_grid_point_meeting_the_null_bound():
    rng = np.random.default_rng(0)
    th = B.threshold([5.0] * 8, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], 0.05, SPEC, rng, 50.0)
    assert th["t"] == 0.05 and th["p_null"] <= 0.05 and th["power"] == 1.0   # median 0 unless >= 4 of 8 draws hit 3.0


def test_a_threshold_that_half_the_effect_cannot_clear_is_not_ok():
    """Null medians reach ~1 (wide null), the effect's median is 1.2: at the null-bound t the half effect (~0.6)
    passes far less than 90 % of the time -> ok False, which calibrate() reports as UNDERPOWERED."""
    rng = np.random.default_rng(0)
    null = [-1.0, -0.5, 0.0, 0.3, 0.6, 0.9, 1.1, 1.3]
    th = B.threshold([1.0, 1.1, 1.15, 1.2, 1.2, 1.25, 1.3, 1.4], null, 0.05, SPEC, rng, 50.0)
    assert th["t"] is not None and th["p_null"] <= SPEC.null_max and th["power"] < SPEC.power_min and not th["ok"]


def test_a_record_without_counts_an_odour_or_a_readout_type_is_invalid_not_an_exception():
    import copy
    base = records(reward=10, punish=25, n2=True, noise=4.0)
    for cut in (lambda r: r.pop("counts"), lambda r: r["counts"].pop("a"), lambda r: r["counts"]["b"].pop(SPEC.p_type),
                lambda r: r.pop("seed")):
        recs = copy.deepcopy(base)
        cut(recs[5])
        v = verdict(recs)
        assert v["status"] == B.INVALID and "malformed" in v["reasons"][0]
        c = B.calibrate(recs, "b", SPEC)
        assert c["status"] == B.INVALID and "malformed" in c["reasons"][0]


def test_non_finite_null_medians_count_against_the_threshold():
    """Five -inf and one +inf: a bootstrap median is +inf with P ~ 0.009 and NaN (+inf and -inf averaged) with P ~ 0.054.
    Counting NaN as not reaching t (the old rule) gives t = 0 with P_null ~ 0.009; counting it as reaching every t
    leaves no grid t with P_null <= 0.05 (conservative)."""
    null = [-math.inf] * 5 + [math.inf]
    th = B.threshold([5.0] * 8, null, 0.05, SPEC, np.random.default_rng(0), 50.0)
    assert th["t"] is None and not th["ok"]
    assert th["n_nonfinite_null"] == 6 and th["n_nonfinite_effect"] == 0
    rng = np.random.default_rng(0)
    with np.errstate(invalid="ignore"):
        nb = np.median(np.asarray(null)[rng.integers(0, 6, size=(SPEC.boot_draws, 6))], axis=1)
    assert float((nb >= 0).mean()) <= SPEC.null_max < float(((nb >= 0) | np.isnan(nb)).mean())   # the rules differ here
    eff = B.threshold([math.inf, -math.inf] * 4, [0.0] * 8, 0.05, SPEC, np.random.default_rng(0), 50.0)
    assert eff["t"] == 0.05 and eff["power"] == 0.0 and not eff["ok"]            # a NaN half effect never passes
    assert eff["n_nonfinite_effect"] == 8 and eff["n_nonfinite_null"] == 0
