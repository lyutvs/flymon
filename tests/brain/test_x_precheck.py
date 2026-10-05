"""The point-θ precheck (X.9.1.1): point values worst over g, the pass rule at every k, the best-power design and the
records target with their tie order, a failed calibration filled and named in the sentence, the verbatim STOP
sentence; the variants (X.4.4: V0 = W's pair_cov, V1 |d′| < 5, V2 aweights, the < 6 rule); the point records'
layout; P1-5's residual comparison; x_rules' reuse and diagnosis reasons."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import x_oc, x_rules
from flymon.brain.x_spec import SPEC as XS

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ND = np.array([0.08, -0.05, -0.32, 1.09, 1.43, 2.72, 4.13, 5.9, 6.0, 7.98, -9.89, 10.7, -12.88, 13.3, 14.5, -16.07])
PREFIX = ("점 θ 사전 점검(X.9.1.1)에서 파일럿 잡음의 점 추정으로도 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다 — p_set · "
          "q · K · F 어느 설계도 k = 4–8 모두에서 점 검정력 ≥ 0.80과 점 거짓 통과 ≤ 0.05를 함께 만족하지 않는다(")
SUFFIX = "). 부트스트랩 작동 특성은 계산하지 않았고, 주 세트를 쓰지 않고 멈춘다."


@pytest.fixture(scope="module")
def pilot():
    return w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))


@pytest.fixture(scope="module")
def theta(pilot):
    return w_oc.fit(pilot)


def test_variants(theta):
    pm = theta["pair_means"]
    assert np.array_equal(x_oc.pair_cov_variant(pm, ND, "V0", XS), theta["pair_cov"])
    x = pm.reshape(16, 4)
    assert np.allclose(x_oc.pair_cov_variant(pm, ND, "V1", XS), np.cov(x[np.abs(ND) < 5.0], rowvar=False))
    assert (np.abs(ND) < 5.0).sum() == 7
    assert np.allclose(x_oc.pair_cov_variant(pm, ND, "V2", XS), np.cov(x, rowvar=False, aweights=1 / (1 + np.abs(ND))))
    f1 = x_oc.pair_cov_variant(pm, ND, "F1", XS)                       # 3 pairs < 6: diagonal + V0's correlations
    v0 = theta["pair_cov"]
    r0 = v0 / np.outer(np.sqrt(np.diag(v0)), np.sqrt(np.diag(v0)))
    sd = np.sqrt(np.diag(np.cov(x[np.abs(ND) < 0.5], rowvar=False)))
    assert np.allclose(f1, r0 * np.outer(sd, sd))
    with pytest.raises(ValueError):
        x_oc.pair_cov_variant(pm, ND, "V9", XS)


def test_pick_rules_and_ties():
    shp = x_oc.grid_shape(XS)
    pw, fp = np.zeros(shp), np.ones(shp)
    pw[1, 2, 0, 3] = 0.9                                                 # best power, false fails
    pw[0, 0, 1, 5], fp[0, 0, 1, 5] = 0.5, 0.0                           # false ok
    fs = list(range(8, 33))
    assert x_oc.pick(pw, fp, XS, fs, False)["index"] == [1, 2, 0, 3]
    t = x_oc.pick(pw, fp, XS, fs, True)
    assert t["rule"] == "false_ok_max_power" and t["index"] == [0, 0, 1, 5]
    fp[0, 0, 1, 5] = 0.2
    fp[3, 1, 0, 0] = 0.1
    assert x_oc.pick(pw, fp, XS, fs, True)["index"] == [3, 1, 0, 0]  # none ok → smallest max_k false
    pw2 = np.full(shp, 0.5)                                            # all tied: larger p_set, larger q, small K, F
    assert x_oc.pick(pw2, np.zeros(shp), XS, fs, False)["index"] == [4, 2, 0, 0]


def test_precheck_layout_and_rule(theta):
    pc = x_oc.precheck(theta, Z, XS, n_rep=40)
    pw, fp = np.asarray(pc["power"]), np.asarray(pc["false"])
    pts = {k: np.asarray(v) for k, v in pc["point"].items()}
    assert np.array_equal(pw, np.min([pts[f"g{g}|min"] for g in XS.cluster_grid], 0))
    assert np.array_equal(fp, np.max([pts[f"g{g}|max"] for g in XS.cluster_grid], 0))
    ok = ((np.round(pw - 0.80, 9) >= 0) & (np.round(fp - 0.05, 9) <= 0)).all(-1)
    assert pc["passed"] == bool(ok.any()) and len(pc["passing"]) == int(ok.sum())
    assert len(pc["at_f32"]) == 5 * 3 * 2 and pc["at_f32"][0]["F"] == 32 and pc["seed"] == 43_200_000
    assert pc["best_power"]["rule"] == "max_power" and pc["records_target"]["rule"] != "max_power"


def test_precheck_pass_and_stop_decisions(theta):
    easy = dataclasses.replace(XS, p_power=0.0, p_false=1.0)
    assert x_rules.precheck_decision(x_oc.precheck(theta, Z, easy, n_rep=8))["outcome"] == x_rules.PASS
    hard = dataclasses.replace(XS, p_power=1.01)
    dec = x_rules.precheck_decision(x_oc.precheck(theta, Z, hard, n_rep=8))
    assert dec["outcome"] == x_rules.STOP_OC_UNREACHABLE
    s = dec["sentence"]
    assert s.startswith(PREFIX) and s.endswith(SUFFIX) and "점 검정력 최대 설계 p_set " in s
    assert s.count("·F 32: k 4 점 검정력 ") == 30


def test_precheck_calibration_failure_fills_and_sentence(theta):
    bad = dataclasses.replace(XS, d_power=40.0)
    pc = x_oc.precheck(theta, Z, bad, n_rep=8)
    assert not pc["calibration"]["min"]["ok"] and np.all(np.asarray(pc["power"]) == 0.0) and not pc["passed"]
    s = x_rules.precheck_decision(pc)["sentence"]
    assert s.startswith(PREFIX + "보정 불가 — min: a ") and "; max: a ok, b " in s and s.endswith(SUFFIX)


def test_point_records_layout(theta):
    target = dict(p_set=0.75, q=0.75, K=8, F=8)
    r = x_oc.point_records(theta, np.abs(ND), target, Z, XS, n_rep=6)
    assert set(r["variants"]) == {"V0", "V1", "V2"} and r["seed"] == 43_200_000
    e = r["variants"]["V1"]["g0.5"]
    assert set(e["true_dprime"]) == {"0.5", "1.0", "1.5", "2.0"} and len(e["true_dprime"]["1.5"]) == 5
    assert set(e["heterogeneous_w"]) == {"one_pair_0.5", "half_pairs_0.5", "all_1.0"}
    assert set(e["pair_cov_scaled"]) == {f"{s}|{m}" for s in (0.5, 1.0, 2.0) for m in ("power", "false")}
    assert len(e["mixed_flies"]) == 5 and len(r["tags"]) == 4 + 4 + 6


def test_residual_compare(pilot):
    out = x_oc.residual_compare(pilot, np.abs(ND) < 0.5, XS)
    assert out["n_pairs"] == dict(balanced=3, rest=13)
    g = out["groups"]["balanced"]
    assert len(g) == 6 * 2 * 2 and set(g["R1|MBON05|X"]) == {"sd", "quantiles", "zero_share", "n"}
    assert g["R1|MBON05|X"]["n"] == 3 * 8 * 8 and list(g["pre|MBON13|Y"]["quantiles"]) == ["5.0", "25.0", "50.0",
                                                                                          "75.0", "95.0"]


def _w_doc(keys):
    return dict(stage0=dict(decision_files={"flymon/brain/w_measure.py": XS.w_measure_file_sha}),
                pilot=dict(outcome="PASS", w_measure_key=XS.w_measure_key, pairs=list(keys), manifest=[{}] * 384,
                           record=dict(pairs={k: dict(naive_d=float(d)) for k, d in zip(keys, ND)})),
                oc=dict(w_measure_key=XS.w_measure_key, detail_sha256=XS.w_oc_detail_sha))


def test_reuse_reasons():
    keys = list(XS.balanced_pairs) + [f"p{j}" for j in range(13)]
    doc = _w_doc(keys)
    k = dict(w_measure_key=XS.w_measure_key, pipeline_key=XS.w_pipeline_key, w_measure_sha=XS.w_measure_file_sha)
    facts = dict(last="f30ae35" + "0" * 33, ancestors={"5fbc4c8": True, "f30ae35": True},
                 git=dict(tracked=True, dirty=False))
    assert x_rules.w_reuse_reasons(doc, k, facts, XS.w_oc_detail_sha, XS) == []
    assert x_rules.pair_fact_reasons(doc, keys, XS) == []
    why = x_rules.w_reuse_reasons(doc, dict(k, w_measure_key="z" * 64), dict(facts, ancestors={}), "y" * 64, XS)
    assert any(w.startswith("W 측정 키 ") for w in why) and any("5fbc4c8" in w for w in why)
    assert any("results/w/oc.json" in w for w in why)
    s = x_rules.reuse_stop(why)
    assert s["outcome"] == x_rules.STOP_REUSE and s["records_unavailable"] is True
    assert s["sentence"].startswith("X 재사용 조건(X.5 1 · X.4.3 진단)이 깨졌다(") and s["sentence"].endswith(
        "X는 W 파일럿·경로와 V 블록을 다시 재는 경로를 갖지 않으므로 주 세트를 쓰지 않고 멈춘다 — 사용자 몫.")


def test_diagnosis_reasons():
    ok = dict(diffs=[], n_draws=200, counts=dict(total=29, power=14, false=18, both=3))
    assert x_rules.diagnosis_reasons(ok, XS) == []
    assert x_rules.diagnosis_reasons(dict(ok, counts=dict(total=28, power=14, false=17, both=3)), XS)
    assert x_rules.diagnosis_reasons(dict(ok, diffs=[dict(draw=4, side="min")]), XS)
    assert x_rules.diagnosis_reasons(dict(ok, n_draws=3, counts=dict(total=0, power=0, false=0, both=0)), XS) == []
