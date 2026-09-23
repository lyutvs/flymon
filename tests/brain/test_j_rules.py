"""Spec appendix J's rules (J.11.3 scan, J.11.4 reading, J.11.4-5 D.6) as pure functions."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import d6a
from flymon.brain import j_rules as R
from flymon.brain.j_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
G8 = ROOT / "results/m2/d6a/g8.json"


def row(g, seed, kc=10, kc_on=5, pn=100, uni_hz=20.0, orn_hz=60.0, std_r=None):
    return dict(g=g, seed=seed, kc=kc, kc_on=kc_on, pn=pn, uni_hz=uni_hz, orn_hz=orn_hz, std_r=std_r)


def test_spec_is_the_declared_scan_and_judgement():
    assert (SPEC.std_f, SPEC.std_tau_ms) == ((0.78, 0.90, 0.95), (893.0, 300.0, 100.0))
    assert (SPEC.lit_f, SPEC.lit_tau_ms, SPEC.scale_bracket, SPEC.alpn_match_tol) == (0.78, 893.0, (1.0, 64.0), 0.05)
    assert (SPEC.kc_band_pct, SPEC.tie_tol, SPEC.first_kc, SPEC.homeo_target, SPEC.max_settings) == \
        ((3.0, 9.0), 0.01, 1.65, 0.060, 3)
    assert (SPEC.h4.t_b_min, SPEC.h4.f_a_min, SPEC.h4.n_pairs_b) == (0.5, 2, 21)
    assert SPEC.h4.h3.all51_seeds == tuple(range(200, 208))


def test_glom_means_average_seeds_and_keep_missing_measures_missing():
    m = R.glom_means([row("A", 1, kc=10), row("A", 2, kc=20), row("B", 1, uni_hz=None)])
    assert m["A"]["kc"] == 15.0 and m["B"]["uni_hz"] is None and m["A"]["std_r"] is None


def test_log10_var_floors_counts_at_half_a_spike():
    assert R.log10_var([0, 0.5], 0.5) == 0.0
    assert math.isclose(R.log10_var([1, 100], 0.5), 1.0)


def test_scan_metrics_name_the_glomeruli_without_unipns():
    rows = [row("A", 1, kc_on=1, pn=100), row("B", 1, kc_on=100, pn=300, uni_hz=None), row("C", 1, kc_on=10, pn=200)]
    m = R.scan_metrics(rows, 0.5)
    assert m["alpn_median"] == 200.0 and m["no_uni"] == ["B"]
    assert math.isclose(m["log10_var"]["kc_on"], float(np.var([0.0, 2.0, 1.0])))
    assert m["log10_var"]["uni_hz"] == 0.0


def _setting(f, tau, red, restorable=True, feasible=True):
    return dict(f=f, tau_ms=tau, reduction=red, restorable=restorable, feasible=feasible)


def test_select_order_takes_the_largest_reduction_and_skips_unusable_settings():
    s = [_setting(0.78, 893.0, 0.10), _setting(0.9, 300.0, 0.30), _setting(0.95, 100.0, 0.50, restorable=False),
         _setting(0.9, 100.0, 0.40, feasible=False), _setting(0.95, 300.0, -0.2)]
    assert R.select_order(s, SPEC) == [1, 0, 4]


def test_a_tie_within_tie_tol_goes_to_the_literature_side():
    s = [_setting(0.95, 100.0, 0.305), _setting(0.9, 300.0, 0.300), _setting(0.78, 893.0, 0.296)]
    assert R.select_order(s, SPEC) == [2, 1, 0]            # all three within 0.01 of 0.305: the literature point first
    s[2]["reduction"] = 0.2949                             # outside the band (0.305 - 0.2949 > 0.01)
    assert R.select_order(s, SPEC) == [1, 0, 2]            # 0.300 ties 0.305 and is nearer 0.78 / 893


@pytest.mark.parametrize("tb, fa, naive, out", [(10, 2, 4, "B"), (11, 2, 4, "SELECTED"), (11, 1, 4, "B"),
                                                (21, 2, 2, "SELECTED"), (14, 0, 1, "B")])
def test_stage2_reading_is_the_m2_bar(tb, fa, naive, out):
    agg = dict(T_b=tb / 21, testable_b=tb, n_b=21, F_a=fa, naive_a=naive)
    r = R.stage2_reading(agg, SPEC.h4)
    assert r["outcome"] == out and r["f_a_possible"] == (naive >= 2)


def test_d6b_is_e2_s_ratio_rule():
    raw = {"rows": [dict(turn=0, seed=1, kc_spikes=[10, 30]), dict(turn=0, seed=2, kc_spikes=[10, 15]),
                    dict(turn=2, seed=1, kc_spikes=[0, 2]), dict(turn=2, seed=2, kc_spikes=[5, 5])]}
    b = R.d6b(raw, 2.0)
    assert b["fired"] and b["turns_over"] == [0] and b["per_turn"]["2"]["max"] == 2.0 and b["ratio_max"] == 3.0


@pytest.mark.skipif(not G8.exists(), reason="G.8's raw record (git-excluded) is not here")
def test_the_declared_engine_judge_is_g8_s_rule_on_g8_s_own_record():
    g8 = json.loads(G8.read_text())
    assert R.judge_d6a_declared(g8, d6a.engine_params()) == d6a.judge(g8)
    with pytest.raises(ValueError, match="declared engine"):
        R.judge_d6a_declared(g8, {**d6a.engine_params(), "kc_thresh": 1.65})
    with pytest.raises(ValueError, match="seeds"):
        R.judge_d6a_declared({**g8, "seeds": g8["seeds"][:-1]}, d6a.engine_params())


def test_c3_mismatch_compares_the_threshold_sha_not_the_path():
    want = dict(kc_thresh=1.65, apl_input_scale=0.6, mbon_hold_frac=0.84, kc_thresh_sha256="s", apl_mode="graded",
                apl_r_max=0.333, kc_thresh_file="results/m0d/h3/thresholds/a.npz")
    assert R.c3_mismatch({**want, "kc_thresh_file": "results/m0d/j/std/thresholds/a.npz"}, want) == []
    assert R.c3_mismatch({**want, "kc_thresh_sha256": "t", "mbon_hold_frac": 0.85}, want) == \
        ["mbon_hold_frac", "kc_thresh_sha256"]


def test_cycle_divergence_names_the_first_differing_cycle():
    cyc = lambda x, sha: dict(search=dict(accepted=dict(x=x)), homeostasis=dict(params=dict(kc_thresh_sha256=sha)))
    want = [dict(cycles=[cyc(0.1, "a"), cyc(0.7, "b"), cyc(0.6, "c")])]
    assert R.cycle_divergence([dict(cycles=[cyc(0.1, "a"), cyc(0.7, "b"), cyc(0.6, "c")])], want) is None
    d = R.cycle_divergence([dict(cycles=[cyc(0.1, "a"), cyc(0.7, "x"), cyc(0.6, "c")])], want)
    assert (d["cell"], d["cycle"], d["threshold_sha256"]) == (0, 2, ["x", "b"])
    assert R.cycle_divergence([dict(cycles=[cyc(0.1, "a")])], want)["cycles"] == [1, 3]


def test_pair_mismatch_is_exact():
    s = dict(d_pre=0.1, r=2.5, p=-3.0, m=2.5, testable=True)
    assert R.pair_mismatch(dict(s), s) == [] and R.pair_mismatch({**s, "r": 2.5000001}, s) == ["r"]
