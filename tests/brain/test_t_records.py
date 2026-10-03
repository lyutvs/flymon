"""T's records: z_side is h4_rules.z_constants (mean, population SD) and H.4's guard statistic on the readout types,
with the zero-SD types named; z_ratios are σ_lever / σ_h4; P_L read on z_lever is a record beside the gate (ℓ_L under
z_lever over ℓ_C under h4 z); the cluster table counts passes per T.9.1 cluster; the α-fixed sensitivity reads L's raw
on h4 z; gate ②'s ratio and OC are S's functions."""
import numpy as np
import pytest

from flymon.brain import s_records
from flymon.brain import t_records as TR
from flymon.brain.p_rules import p_judge
from flymon.brain.t_spec import SPEC
from tests.brain.r_fixtures import Z, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, p_rows
from tests.brain.t_fixtures import counts, ref_rows, rest_rows

READOUT = {"A": "MBON13", "P": "MBON05"}


def test_z_side_is_population_mean_sd_and_the_guard():
    a, p = counts(96, 8), counts(96, 20, zeros=10)
    side = TR.z_side(ref_rows(a, p), rest_rows(96, rest=2), READOUT, SPEC)
    assert side["z"] == {"A": [float(np.mean(a)), float(np.std(a))], "P": [float(np.mean(p)), float(np.std(p))]}
    g = side["guard"]
    assert g["MBON13"]["median_delta"] == float(np.median(np.array(a) - 2)) and g["MBON13"]["zero_share"] == 0.0
    assert g["MBON05"]["zero_share"] == pytest.approx(10 / 96) and g["MBON05"]["passes"]
    assert (side["n_ref"], side["n_rest"], side["edit_edges"], side["csc_sha256"], side["zero_sd"]) == (
        96, 96, [0], ["sha-C"], [])
    flat = TR.z_side(ref_rows([5] * 96, p), rest_rows(96), READOUT, SPEC)
    assert flat["z"] is None and flat["zero_sd"] == ["MBON13"] and "zero SD" in flat["why"]
    assert TR.z_ratios({"A": [6.0, 4.0], "P": [80.0, 40.0]}, {"A": (10.0, 8.0), "P": (26.0, 20.0)}) == {"A": 0.5,
                                                                                                         "P": 2.0}


def test_p_zlever_is_a_record_beside_the_h4_gate():
    rl = p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2)
    rc = p_rows(SPEC.p_c, "none", 16)
    res_c = p_judge(rc, Z, C1, SPEC.p_c)
    zl = {"A": (10.0, 4.5), "P": (26.0, 19.0)}                    # σA halved: ℓ_L doubles on z_lever
    rec = TR.p_zlever(rl, zl, C1, res_c, SPEC)
    res_l = p_judge(rl, Z, C1, SPEC.p)
    for d in SPEC.p.directions:
        assert rec["ell"][d] == pytest.approx(2 * res_l["directions"][d]["ell"])
        assert rec["ratio_to_C_h4"][d] == pytest.approx(rec["ell"][d] / res_c["directions"][d]["ell"])
    assert rec["label"] == "LEARNS_CONFIRMATORY"
    assert TR.ratio is s_records.ratio and TR.gate2_oc is s_records.gate2_oc


def test_clusters_count_per_cluster_and_condition():
    pairs = {"L": [dict(key="b|1|x|y", testable=True, reward_pass=True, punish_pass=False),
                   dict(key="a|2|x|y", testable=False, reward_pass=True, punish_pass=True)],
             "C": [dict(key="b|1|x|y", testable=False, reward_pass=False, punish_pass=True)]}
    labels = {"b|1|x|y": "GROUND* 대 NORMAL", "a|2|x|y": "FIRE*"}
    assert TR.clusters(pairs, labels) == {
        "GROUND* 대 NORMAL": {"L": dict(n=1, testable=1, reward_pass=1, punish_pass=0),
                              "C": dict(n=1, testable=0, reward_pass=0, punish_pass=1)},
        "FIRE*": {"L": dict(n=1, testable=0, reward_pass=1, punish_pass=1)}}


def test_alpha_fixed_reads_l_on_h4_z():
    from flymon.brain.r_records import cond_summary
    rows = fake_rows(21, 43, turn0=0)
    keys = [key_of(r) for r in rows]
    seeds = SPEC.judge_seeds()
    plan = {k: (True, True, False) for k in keys[:14]}
    gl = got_for(rows, plan, edges=2, edit=SPEC.lever_edit, sha="sha-L")
    gc = got_for(rows, {k: (True, True, False) for k in keys[:8]})
    C = cond_summary(gc, SPEC.cond("C"), SPEC, Z, keys, seeds)
    out = TR.alpha_fixed(gl, C, keys, seeds, Z, SPEC)
    assert (out["n"], out["c"], out["band"]) == (14, 8, "B_Fa") and out["g_fail"]["g_fail"] is False
    assert "판정 아님" in out["note"]
