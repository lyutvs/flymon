"""The V chain from smoke to gate ② (V.3 8-10, V.6, V.9.2): smoke on 24_609_xxx / 25_409_xxx checks L_V's edges, CSC
(block path's) and z_V and C / E0's h4 z and R's repro CSC; the OCs (independent with V's notes, T's cluster method on
block set's clusters written to results/v/oc_cluster.json with its sha); gate ② labels both P conditions on h4 z, reads
the h4 ratio first and then ℓ_L(z_V) / ℓ_C(h4 z), each ≥ 0.5 in both directions; a smoke problem or any gate ② STOP
blocks the marker and the judgement; gate ②'s one INVALID rerun; one measurer per z."""
from pathlib import Path

import pytest

from flymon.brain import v_rules
from flymon.brain import v_runner as VR
from flymon.brain.v_spec import LEVER_V
from tests.brain.v_world import SPEC, World, ZScripted, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to(w, m, zm, last):
    r = w.runner(m, zm)
    for s in VR.ORDER[:VR.ORDER.index(last)]:
        getattr(r, f"stage_{s}")()
    return r


def test_chain_to_gate2(w):
    m, zm = w.scripted(), ZScripted()
    through(w, m, zm=zm)
    d = doc()
    assert set(d) == set(VR.ORDER[:VR.ORDER.index("measurement_started")])
    orc = d["smoke"]["oracle"]
    assert orc["L"]["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and orc["L"]["edit"] == LEVER_V
    assert orc["C"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} == orc["E0"]["z"]
    oc = d["oc"]
    assert Path(SPEC.oc_cluster_out).exists() and oc["cluster_path"] == SPEC.oc_cluster_out
    assert oc["cluster"]["clusters_b"] == [21] and oc["cluster"]["clusters_a"] == [43]
    assert set(oc["cluster_values"]) == {"cl_null", "cl_harm1", "cl_harm2"}
    assert any("U 기전 기록" in n for n in oc["independent"]["notes"])
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"], g2["seeds"], g2["scale"]) == (
        "LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY", list(SPEC.p.seeds), None)
    assert g2["edit_edges_L"] == [2] and g2["ratio_z_V"]["r1"]["ratio"] == pytest.approx(3 * g2["ratio"]["r1"]["ratio"])
    assert g2["p_L_on_z_V"]["ell"]["r1"] == pytest.approx(g2["ratio_z_V"]["r1"]["ell_L"])
    assert w.judgement_calls == 0 and w.odour_calls == 1
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]



def test_smoke_problem_blocks_later_stages(w):
    m = w.scripted(override={("smoke", "L"): dict(sha="sha-C")})
    out = _to(w, m, ZScripted(), "smoke").stage_smoke()
    assert out["problems"] and any("block path's L_V" in p for p in out["problems"])
    with pytest.raises(SystemExit):
        w.runner(m).stage_oc()


@pytest.mark.parametrize("drop,outcome", [({LEVER_V: 2}, "STOP_PUNISH_BROKEN"), ({"none": 2}, "STOP_P_REFERENCE"),
                                          ({LEVER_V: 6}, "STOP_PUNISH_WEAKENED")])
def test_gate2_stops_on_h4_block_the_judgement(w, drop, outcome):
    m = w.scripted(drop=drop)
    out = _to(w, m, ZScripted(), "gate2").stage_gate2()
    assert out["outcome"] == outcome
    if outcome == "STOP_PUNISH_WEAKENED":
        assert out["scale"] == "h4"
    with pytest.raises(SystemExit):
        w.runner(m)._require("measurement_started")                     # STOP blocks the marker


def test_gate2_stop_on_the_z_v_scale_alone(w):
    """V.9.2: σA_V = 30 makes ℓ_L(z_V) / ℓ_C(h4 z) = 13/30 / (17/9) ≈ 0.23 while the h4 ratio 13/17 passes."""
    lever = ([5, 65] * 48, [60, 100] * 48)
    zm = ZScripted(counts={LEVER_V: lever})
    w.u_rows = dict(edit=LEVER_V, ref=zm.reference(LEVER_V, [None] * 48, 0.35, 800.0, 600),
                    rest=zm.rest(LEVER_V, list(range(96)), 800.0, 600))
    m = w.scripted()
    out = _to(w, m, zm, "gate2").stage_gate2()
    assert out["outcome"] == "STOP_PUNISH_WEAKENED" and out["scale"] == "z_V" and set(out["low"]) == {"r1", "r2"}
    assert all(out["ratio"][d]["ratio"] >= 0.5 for d in ("r1", "r2"))
    assert out["failed_scales"] == [["z_V", "r1"], ["z_V", "r2"]]
    h, v = out["ratio"], out["ratio_z_V"]                               # V.9.7 1: V's sentence, both scales
    assert out["sentence"] == ("조합 지렛대 아래 같은 시드 P에서 처벌 학습량 비가 0.5에 못 미쳤다("
                               f"h4 z 비 ℓ_r1 {h['r1']['ratio']:.3f}·ℓ_r2 {h['r2']['ratio']:.3f}, "
                               f"z_V 비 ℓ_r1 {v['r1']['ratio']:.3f}·ℓ_r2 {v['r2']['ratio']:.3f}; "
                               "미달: z_V ℓ_r1, z_V ℓ_r2).")


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    _to(w, w.scripted(), ZScripted(), "gate2")
    r = w.runner(w.scripted(arm_edges={LEVER_V: 3}))                    # only gate ②'s P arms carry 3 edges
    assert r.stage_gate2()["outcome"] == v_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                                        # same pipeline key
    m2 = w.scripted()
    r2 = w.runner(m2, pipeline={"key": "q" * 64})
    out = r2.stage_gate2(rerun=True)
    assert out["outcome"] == "PASS" and out["rerun_of"] and "gate2_invalid" in doc()
    with pytest.raises(SystemExit):
        r2.stage_gate2(rerun=True)


def test_one_measurer_per_z(w):
    m = w.scripted()
    made = []

    def measure(z):
        made.append(dict(z))
        return m.at(z)
    r = VR.Runner(measure, ZScripted(), w.ctx, SPEC, code={"key": SPEC.r_shared_key}, tcode={"key": "t" * 64},
                  ucode={"key": "u" * 64}, pipeline={"key": "p" * 64}, archive_root=w.archive)
    through(w, m, r=r)
    assert sorted(tuple(z["A"]) for z in made) == [(6.0, 3.0), (10.0, 9.0)]
