"""The T stage chain up to gate ③ (T.1-T.3, T.6, T.9.1-T.9.4, T.9.6, T.9.7; plan Readings 3-12): reuse -> set -> z ->
gate1s -> smoke -> oc -> gate2_oc -> gate2 -> gate3; each stage refuses (exit 2, nothing written) when an earlier block
is missing, a later one exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted
or a hashed file is dirty; STOP_REUSE, STOP_SET_SHORT, STOP_Z_REPRO (the lever then unmeasured), STOP_Z_DEGENERATE,
STOP_STRENGTH_LEVER, a smoke problem, gate ② STOPs / INVALID, STOP_EVEN_REPRO (L then unmeasured) and
STOP_EVEN_LOW_LEVER block every later stage; R's reuse condition broken after `reuse` refuses with exit 7; L's oracle
runs on z_lever and C / E0's on block h4's z; gate ② reads both P conditions on h4 z; the judgement set is touched only
through the set's odours (gate ①'s supplement) before the judgement stages."""
import json
from pathlib import Path

import pytest

from flymon.brain import t_rules
from flymon.brain import t_runner as TR
from tests.brain.t_world import SPEC, ZL, ZScripted, World, doc, through_gate3


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate3(w):
    m = w.scripted()
    through_gate3(w, m)
    d = doc()
    assert set(d) == set(TR.ORDER[:TR.ORDER.index("jm:L")])
    assert all(d[b]["pipeline_key"] == "p" * 64 and d[b]["t_measure_key"] == "t" * 64 for b in d)
    assert d["reuse"]["records"]["r_gate3"] == dict(testable_b=16, c_even=7)
    assert d["set"]["set"]["last_turn"] == 103 and w.set_calls == 1
    z = d["z"]
    assert z["z_lever"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and z["none"]["z"] == {"A": [10.0, 9.0],
                                                                                      "P": [26.0, 19.0]}
    assert z["sd_ratio_lever_over_h4"] == {"A": 3.0 / 9.0, "P": 20.0 / 19.0}
    assert z["lever"]["guard"]["MBON13"]["median_delta"] == 6.0 and Path(SPEC.z_detail).exists()
    assert d["gate1s"]["record"]["n_odours"] == 55 and w.odour_calls == 1
    orc = d["smoke"]["oracle"]
    assert orc["L"]["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert orc["C"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} == orc["E0"]["z"]
    assert set(d["smoke"]["cost"]) == {"gate2_h", "gate3_L_h", "jm_per_condition_h", "note"}
    assert d["oc"]["cluster_fixture_sha256"] and round(d["oc"]["cluster"]["g_fail"][0]["p"], 3) == 0.401
    assert d["gate2_oc"]["rows"][1]["p_stop_weakened"] == pytest.approx(0.75)
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"], g2["seeds"]) == ("LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY",
                                                            list(SPEC.p.seeds))
    assert g2["p_L_on_z_lever"]["ell"]["r1"] == pytest.approx(3 * g2["ratio"]["r1"]["ell_L"])  # σA 9 -> 3
    g3 = d["gate3"]
    assert (g3["testable_b"], g3["c_even"], g3["L_h4_R"]["testable_b"], g3["L_h4_R"]["matches_r_gate3"]) == (
        12, 7, 16, True)
    assert w.judgement_calls == 0
    ev = [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    assert ev == [("oracle", "even", "L", 39, ZL["A"])]                      # L only, on z_lever; C from R's raw
    sm = [c for c in m.calls if c[0] == "oracle" and c[1] == "smoke"]
    assert [(c[2], c[4]) for c in sm] == [("L", ZL["A"]), ("C", (10.0, 9.0)), ("E0", (10.0, 9.0))]
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]


def test_reuse_stop_and_broken_reuse_refuses_with_7(w):
    out = w.runner(w.scripted(), code={"key": "f" * 64}).stage_reuse()
    assert out["outcome"] == t_rules.STOP_REUSE and "공유 측정 키" in out["sentence"]
    with pytest.raises(SystemExit) as e:
        w.runner(w.scripted()).stage_set()
    assert e.value.code == 2 and w.set_calls == 0


def test_reuse_broken_after_reuse(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.r["gate1"]["outcome"] = "STOP_STRENGTH_LEVER"
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == TR.EXIT_KEY and "set" not in doc() and w.set_calls == 0


def test_set_short_and_mismatch(w):
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.js = dict(w.js, digest_keys="0" * 64)
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == 2 and "set" not in doc()
    w.js = dict(w.js, status="STOP_SET_SHORT", n_a=40, last_turn=1985)
    out = r.stage_set()
    assert out["outcome"] == t_rules.STOP_SET_SHORT and "(21·40쌍)" in out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_z()


def _to_z(w, m, zm):
    r = w.runner(m, zm)
    r.stage_reuse()
    r.stage_set()
    return r


def test_z_repro_stop_leaves_the_lever_unmeasured(w):
    zm = ZScripted(counts={SPEC.no_edit: ([1, 19] * 47 + [1, 20], [7, 45] * 48)})
    r = _to_z(w, w.scripted(), zm)
    out = r.stage_z()
    assert out["outcome"] == t_rules.STOP_Z_REPRO and out["lever"] is None and "z_lever" not in out
    assert "z 절차 결함" in out["sentence"]
    assert {c[1] for c in zm.calls} == {SPEC.no_edit}
    with pytest.raises(SystemExit):
        r.stage_gate1s()


@pytest.mark.parametrize("counts,failed", [
    (([0, 6] * 48, [60, 100] * 48), ["MBON13"]),                 # MBON13 median Δ 3 < 5
    (([3, 9] * 48, [0] * 96), ["MBON05"]),                       # silent MBON05: guard and zero SD
])
def test_z_degenerate_stops(w, counts, failed):
    zm = ZScripted(counts={SPEC.lever_edit: counts})
    r = _to_z(w, w.scripted(), zm)
    out = r.stage_z()
    assert out["outcome"] == t_rules.STOP_Z_DEGENERATE and out["failed"] == failed and "z_lever" not in out
    with pytest.raises(SystemExit):
        r.stage_gate1s()


def test_z_invalid_when_the_lever_edits_other_than_2_edges(w):
    r = _to_z(w, w.scripted(), ZScripted(edges={SPEC.lever_edit: 1}))
    assert r.stage_z()["outcome"] == t_rules.INVALID


def test_gate1s_stop_names_the_odours(w):
    m = w.scripted(kc={SPEC.lever_edit: {"M3|T": 0.02}})
    r = _to_z(w, m, ZScripted())
    r.stage_z()
    out = r.stage_gate1s()
    assert out["outcome"] == t_rules.STOP_STRENGTH_LEVER and out["failed"] == ["M3|T"]
    assert "T 세트 냄새 M3|T 0.0200" in out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_smoke()


def _to_gate2(w, m):
    r = w.runner(m)
    for s in ("reuse", "set", "z", "gate1s", "smoke", "oc", "gate2_oc"):
        getattr(r, f"stage_{s}")()
    return r


@pytest.mark.parametrize("drop,outcome", [
    ({SPEC.lever_edit: 6}, t_rules.STOP_PUNISH_WEAKENED), ({SPEC.lever_edit: 3}, t_rules.STOP_PUNISH_BROKEN),
    ({SPEC.no_edit: 3}, t_rules.STOP_P_REFERENCE)])
def test_gate2_stops_block_the_judgement(w, drop, outcome):
    r = _to_gate2(w, w.scripted(drop=drop))
    out = r.stage_gate2()
    assert out["outcome"] == outcome and out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_gate3()
    assert w.judgement_calls == 0


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    m = w.scripted()
    r = _to_gate2(w, m)
    m.arm_edges[SPEC.lever_edit] = 1
    assert r.stage_gate2()["outcome"] == t_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)
    m.arm_edges[SPEC.lever_edit] = SPEC.lever_edges
    fixed = w.runner(m, pipeline={"key": "q" * 64})
    assert fixed.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["rerun_of"] == d["gate2_invalid"]["written_at"]
    with pytest.raises(SystemExit):
        fixed.stage_gate2(rerun=True)


def test_gate3_even_repro_stop_leaves_l_unmeasured(w):
    m = w.scripted()
    r = _to_gate2(w, m)
    r.stage_gate2()
    w.r_even_plan["C"] = dict(list(w.r_even_plan["C"].items())[:6])          # R's C raw reads 6, not 7
    out = r.stage_gate3()
    assert out["outcome"] == t_rules.STOP_EVEN_REPRO and out["L"] is None
    assert out["sentence"] == "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L 16, C 6)."
    assert not [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    with pytest.raises(SystemExit):
        r._require("jm:L")


def test_gate3_low_lever_stop(w):
    m = w.scripted()
    m.plan[("even", "L")] = {}
    r = _to_gate2(w, m)
    r.stage_gate2()
    out = r.stage_gate3()
    assert out["outcome"] == t_rules.STOP_EVEN_LOW_LEVER and out["testable_b"] == 0
    with pytest.raises(SystemExit):
        r._require("jm:L")


def test_gate3_refuses_when_rs_even_raw_is_missing(w):
    r = _to_gate2(w, w.scripted())
    r.stage_gate2()

    def missing(rows, name):
        raise ValueError("R's even raw for C lacks 1 pair(s)")
    w.ctx["r_even"] = missing
    with pytest.raises(SystemExit) as e:
        r.stage_gate3()
    assert e.value.code == 2 and "gate3" not in doc()


def test_smoke_problem_blocks_later_stages(w):
    m = w.scripted(override={("smoke", "L"): dict(edges=1)})
    r = _to_z(w, m, ZScripted())
    r.stage_z()
    r.stage_gate1s()
    assert r.stage_smoke()["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_oc_refuses_a_cluster_table_other_than_the_fixture(w, monkeypatch):
    r = _to_z(w, w.scripted(), ZScripted())
    for s in ("z", "gate1s", "smoke"):
        getattr(r, f"stage_{s}")()
    real = TR.t_rules.oc_cluster
    monkeypatch.setattr(TR.t_rules, "oc_cluster", lambda spec: dict(real(spec), sha256="0" * 64))
    with pytest.raises(SystemExit):
        r.stage_oc()
    assert "oc" not in doc()


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    r = w.runner(w.scripted())
    with pytest.raises(SystemExit):
        r.stage_set()
    r.stage_reuse()
    with pytest.raises(SystemExit):
        r.stage_reuse()
    monkeypatch.setattr(TR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_set()
    monkeypatch.setattr(TR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(TR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/t_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_set()
    assert set(doc()) == {"reuse"}
    assert json.loads(Path(SPEC.summary).read_text())["reuse"]["outcome"] == "PASS"


# ---- additions to the brief (carried review findings) -----------------------------------------------------------------
def test_z_invalid_without_96_presentations_leaves_the_lever_unmeasured(w):
    zm = ZScripted(counts={SPEC.no_edit: ([1, 19] * 47, [7, 45] * 47)})       # same z, 94 presentations
    r = _to_z(w, w.scripted(), zm)
    out = r.stage_z()
    assert out["outcome"] == t_rules.INVALID and out["n_presentations"] == 96 == TR.n_presentations()
    assert out["lever"] is None and {c[1] for c in zm.calls} == {SPEC.no_edit}
    with pytest.raises(SystemExit):
        r.stage_gate1s()


def test_one_measurer_per_z(w):
    made = []
    m = w.scripted()

    def measure(z):
        made.append(tuple(sorted(z.items())))
        return m.at(z)
    r = TR.Runner(measure, ZScripted(), w.ctx, SPEC, code={"key": SPEC.r_shared_key}, tcode={"key": "t" * 64},
                  pipeline={"key": "p" * 64}, archive_root=w.archive)
    through_gate3(w, m, r=r)
    assert len(made) == len(set(made)) == 2


class _Cache:
    def __init__(self, have):
        self.have, self.gets = have, []

    def get(self, kind, inputs):
        self.gets.append(inputs["pair"])
        return {"result": 1} if inputs["pair"] in self.have else None


class _RM:
    def __init__(self):
        self.oracle_calls = 0

    def inputs(self, row, cond, block, seeds):
        return dict(pair=TR.row_key(row), condition=cond.name, block=block)

    def oracle(self, rows, cond, block, seeds):
        self.oracle_calls += 1
        return ["got"] * len(rows)


def test_r_even_reader_gets_every_entry_first_and_never_computes():
    rows = [dict(axis="b", turn=i, x="X", y="Y") for i in range(5)]
    keys = [TR.row_key(r) for r in rows]
    cache, rm = _Cache({keys[0], keys[2], keys[4]}), _RM()
    rd = TR.r_even_reader(cache, rm, SPEC)
    with pytest.raises(ValueError, match=r"lacks 2 pair\(s\).*'b\|1\|X\|Y', 'b\|3\|X\|Y'"):
        rd(rows, "C")
    assert cache.gets == keys and rm.oracle_calls == 0
    cache.have = set(keys)
    assert rd(rows, "C") == ["got"] * 5 and rm.oracle_calls == 1


def test_runtime_z_must_equal_the_declared_z_h4(tmp_path):
    z = SPEC.z_h4_dict()
    TR.check_z_matches_spec(SPEC, dict(z))
    bad = {k: (v[0] + 1e-9, v[1]) for k, v in z.items()}
    with pytest.raises(SystemExit) as e:
        TR.check_z_matches_spec(SPEC, bad)
    assert e.value.code == 2
    bad_sd = {k: (v[0], v[1] * 1.0000001) for k, v in z.items()}
    with pytest.raises(SystemExit) as e:
        TR.check_z_matches_spec(SPEC, bad_sd)
    assert e.value.code == 2
