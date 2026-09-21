"""The H.4 configuration object restates nothing it could take from committed code: the M0c arm, G.14's oracle driver,
the H.3 guard, F.3's z convention and the spec's selection rule (spec H.4, H.3a.1, H.3a.9, H.4a)."""
import importlib.util
import inspect
import sys
from pathlib import Path

from flymon.brain import conditioning, pool_jobs
from flymon.brain.h3_spec import SPEC as H3
from flymon.brain.h4_spec import SPEC

CAL = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/m2-calibration-g"


def _g14_driver():
    sys.path.insert(0, str(CAL))
    try:
        s = importlib.util.spec_from_file_location("m2_engine_probe", CAL / "m2_engine_probe.py")
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(str(CAL))


def _defaults(fn):
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect._empty}


def test_the_combinations_and_their_tie_order():
    assert SPEC.combos == ("C0", "C1", "C3")                    # H.3a.1: C2 dropped; C0 > C1 > C3


def test_reactivity_is_the_h3_guard_and_uses_h3s_reference_set():
    assert (SPEC.react_med_delta_min, SPEC.react_zero_share_max) == (H3.guard_med_delta_min, H3.guard_zero_share_max)
    assert SPEC.h3 == H3


def test_the_teaching_arm_is_m0cs():
    job, arm = _defaults(pool_jobs.conditioning_arm_job), _defaults(conditioning.run_arm)
    assert SPEC.teach_seeds == tuple(range(8, 16)) and SPEC.teach_min_decreased == 6      # D.4 seeds; H.3a.9 (1)
    assert (SPEC.teach_trials, SPEC.teach_present_ms, SPEC.teach_window.settle_ms) == \
        (job["trials"], job["present_ms"], job["settle_ms"])
    assert (SPEC.teach_gap_ms, SPEC.teach_window.read_ms) == (arm["gap_ms"], arm["read_ms"])
    assert (job["strength"], job["k"], job["odor_seed"]) == (H3.strength, H3.design_k, H3.design_odor_seed)
    assert (job["punish_type"], job["reward_type"]) == (H3.punish_type, H3.reward_type)
    assert SPEC.teach_orders == ("ab", "ba")                                               # H.4a.1


def test_the_oracle_is_g14s_driver():
    d = _g14_driver()
    assert list(SPEC.act_seeds) == d.ACT_SEEDS and list(SPEC.select_seeds) == d.SELECT_SEEDS
    assert list(SPEC.report_seeds) == d.REPORT_SEEDS and list(SPEC.oracle_alphas) == d.ALPHAS
    assert H3.strength == d.STRENGTH and SPEC.kc_window_ms == d.WINDOW_MS
    assert (SPEC.oracle_window.settle_ms, SPEC.oracle_window.read_ms) == (d.SETTLE_MS, d.READ_MS)


def test_the_selection_rule_and_the_notes():
    assert (SPEC.f_a_min, SPEC.t_b_min, SPEC.tie_pairs, SPEC.z_ddof) == (2, 0.5, 2, 0)
    assert len(SPEC.notes) == 3 and "H.3a.10" in SPEC.notes[0] and "H.4a.2" in SPEC.notes[1] and "H.4a.4" in SPEC.notes[2]
