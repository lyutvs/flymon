"""The U stage chain up to gate ② (U.3 as ordered by U.9.5; U.6, U.9.1-U.9.4): reuse -> path -> set -> scan -> kc ->
even -> choose -> smoke -> oc -> gate2_oc -> gate2; each stage refuses (exit 2, nothing written) when an earlier block
is missing, a later one exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted
or a hashed file is dirty; STOP_REUSE, STOP_U_PATH_REPRO, STOP_NO_QUALIFIED_F (guard or KC), STOP_EVEN_REPRO (no L_f
then measured), STOP_EVEN_PUNISH, STOP_EVEN_LOW_LEVER, a smoke problem and gate ② STOPs block every later stage; the
reuse condition broken after `reuse` refuses with exit 7; L_f's oracle runs on z_f and C / E0's on block h4's z; gate ②
reads both P conditions on h4 z; the judgement set is touched only through the set's odours (the KC band) before the
judgement stages; the mechanism contrast is recorded and never decides."""
import json
from pathlib import Path

import pytest

from flymon.brain import u_rules
from flymon.brain import u_runner as UR
from flymon.brain.u_measure import u_edit
from tests.brain.u_world import SPEC, ZF, ZScripted, World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate2(w):
    m, zm = w.scripted(), ZScripted()
    through(w, m, zm=zm)
    d = doc()
    assert set(d) == set(UR.ORDER[:UR.ORDER.index("jm:L")])
    assert all(d[b]["pipeline_key"] == "p" * 64 and d[b]["u_measure_key"] == "u" * 64 for b in d)
    assert d["reuse"]["records"]["t_z"]["outcome"] == "STOP_Z_DEGENERATE"
    pa = d["path"]
    assert [(c["f"], c["diffs"]) for c in pa["checks"]] == [("1.0", []), ("0.0", []), ("0.0", []), ("1.0", [])]
    assert pa["sides"]["1.0"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} and Path(SPEC.path_detail).exists()
    assert set(pa["sides"]["1.0"]["mech"]["types"]) == set(w.ctx["rec_types"])
    sc = d["scan"]
    assert sc["candidates"] == [0.5, 0.6, 0.7, 0.8, 0.9] and sc["checked"] == [0.5, 0.6, 0.7]
    assert sc["points"]["0.3"]["outcome"] == "STOP_Z_DEGENERATE" and sc["points"]["0.3"]["z"] is None
    assert sc["points"]["0.6"]["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert sc["points"]["0.6"]["record"] == dict(p_mean_ratio=80.0 / 26.0, sd_ratio={"A": 3.0 / 9.0, "P": 20.0 / 19.0})
    rd = sc["contrast"]["readings"]
    assert (rd["d_none"], rd["block"]["d_block"], rd["block"]["reading"]) == (9.0, 8.0, u_rules.CHAIN_AGAINST)
    assert rd["entry"]["reading"] == u_rules.KC_SIDE and sc["contrast"]["block_f0"]["invalid"] == []
    assert len([c for c in zm.calls if c[0] == "reference"]) == 2 + 9 + 3
    assert d["kc"]["qualified"] == [0.5, 0.6, 0.7] and set(d["kc"]["points"]) == {"0.5", "0.6", "0.7"}
    assert w.odour_calls == 1 and d["kc"]["points"]["0.6"]["record_set"]["n_odours"] == 55
    ev = d["even"]
    assert ev["c_even"] == 7 and ev["L_h4_R"] == dict(testable_b=16, matches_r_gate3=True)
    assert {k: v["record"]["testable_b"] for k, v in ev["per_f"].items()} == {"0.5": 12, "0.6": 13, "0.7": 13}
    assert ev["per_f"]["0.7"]["record"]["drops"]["a"]["net_drop"] == 2
    ch = d["choose"]
    assert (ch["f_star"], ch["kept"], ch["testable_b"]) == (0.6, [0.5, 0.6], 13)
    assert ch["z_f_star"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    evc = [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    assert evc == [("oracle", "even", "L", u_edit(f), 39, ZF["A"]) for f in (0.5, 0.6, 0.7)]
    pc = [c for c in m.calls if c[0] == "oracle" and c[1] == "path"]
    assert pc == [("oracle", "path", "L", u_edit(0.0), 3, (10.0, 9.0)), ("oracle", "path", "L", u_edit(1.0), 3,
                                                                          (10.0, 9.0))]
    orc = d["smoke"]["oracle"]
    assert orc["L"]["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and orc["L"]["edit"] == u_edit(0.6)
    assert orc["C"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} == orc["E0"]["z"]
    assert set(d["smoke"]["cost"]) == {"gate2_h", "jm_per_condition_h", "note"}
    assert d["oc"]["cluster"]["sha256"] == SPEC.oc_cluster_sha256 and any("P3-10" in n for n in
                                                                          d["oc"]["independent"]["notes"])
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"], g2["f_star"], g2["seeds"]) == (
        "LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY", 0.6, list(SPEC.p.seeds))
    assert g2["edit_edges_L"] == [2] and g2["p_L_on_z_f"]["ell"]["r1"] == pytest.approx(3 * g2["ratio"]["r1"]["ell_L"])
    assert w.judgement_calls == 0
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]


def test_reuse_stop_and_later_stages_refuse(w):
    out = w.runner(w.scripted(), tcode={"key": "x" * 64}).stage_reuse()
    assert out["outcome"] == u_rules.STOP_REUSE and "T 측정 키" in out["sentence"]
    with pytest.raises(SystemExit) as e:
        w.runner(w.scripted()).stage_path()
    assert e.value.code == 2


def test_reuse_broken_after_reuse_refuses_with_7(w):
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.t_git["dirty"] = True
    with pytest.raises(SystemExit) as e:
        r.stage_path()
    assert e.value.code == UR.EXIT_KEY and "path" not in doc()


def _to(w, m, zm, last):
    r = w.runner(m, zm)
    for s in UR.ORDER[:UR.ORDER.index(last)]:
        getattr(r, f"stage_{s}")()
    return r


def test_path_stop_on_a_reference_row(w):
    a, p = [1, 19] * 47 + [1, 20], [7, 45] * 48                         # one MBON13 count off at f = 1
    zm = ZScripted(counts={1.0: (a, p)})
    r = _to(w, w.scripted(), zm, "path")
    out = r.stage_path()
    assert out["outcome"] == u_rules.STOP_U_PATH_REPRO and out["failed"][0]["f"] == 1.0
    assert out["sentence"].startswith("U 부분 편집 경로가 끝점 1.0에서 T 편집 없는 엔진 기준 집합 행을 재현하지 못했다(기준 집합: "
                                      "1 row(s) differ")
    with pytest.raises(SystemExit):
        r.stage_set()


def test_path_stop_on_the_even_oracle_and_mutation_without_the_edit(w):
    m = w.scripted()
    m.plan[("path", u_edit(0.0))] = {}                                  # U's f = 0 oracle ≠ R's L raw
    r = _to(w, m, ZScripted(), "path")
    out = r.stage_path()
    assert out["outcome"] == u_rules.STOP_U_PATH_REPRO and out["failed"] == [
        dict(f=0.0, ref="R 짝수 L 원자료 3쌍", diffs=out["failed"][0]["diffs"])]


def test_path_mutation_an_oracle_on_the_unedited_csc_fails(w):
    m = w.scripted(override={("path", "L"): dict(sha="sha-C")})          # the edit never reached the engine
    out = _to(w, m, ZScripted(), "path").stage_path()
    assert out["outcome"] == u_rules.STOP_U_PATH_REPRO and {x["ref"] for x in out["failed"]} == {
        "R 짝수 L 원자료 3쌍"}


def test_path_invalid_without_96_presentations(w):
    r = _to(w, w.scripted(), ZScripted(n=95), "path")
    assert r.stage_path()["outcome"] == u_rules.INVALID


def test_scan_stop_when_no_f_passes_the_guard(w):
    fail = ([0, 6] * 48, [60, 100] * 48)
    zm = ZScripted(counts={f: fail for f in SPEC.f_grid})
    r = _to(w, w.scripted(), zm, "scan")
    out = r.stage_scan()
    assert out["outcome"] == u_rules.STOP_NO_QUALIFIED_F and out["kind"] == u_rules.NQ_GUARD
    assert out["sentence"].startswith("f ∈ {0.1, …, 0.9} 9점 모두 판독 반응성 가드에서 떨어졌다(f 0.1: MBON05 Δ 80.0")
    assert out["contrast"]["readings"]["entry"]["reading"] == u_rules.KC_SIDE      # recorded even on a STOP (U.9.3 6)
    with pytest.raises(SystemExit):
        r.stage_kc()


def test_scan_candidates_need_no_monotonicity(w):
    fail = ([0, 6] * 48, [60, 100] * 48)
    from tests.brain.u_world import PASS_COUNTS
    zm = ZScripted(counts={**{f: fail for f in SPEC.f_grid}, 0.2: PASS_COUNTS, 0.8: PASS_COUNTS})
    r = _to(w, w.scripted(), zm, "scan")
    out = r.stage_scan()
    assert out["outcome"] == "PASS" and out["candidates"] == [0.2, 0.8] and out["checked"] == [0.2, 0.8]


def test_scan_invalid_point_and_contrast_invalid_is_only_a_record(w):
    zm = ZScripted(edges={u_edit(0.3): 1})
    r = _to(w, w.scripted(), zm, "scan")
    assert r.stage_scan()["outcome"] == u_rules.INVALID


def test_scan_contrast_with_other_edge_counts_is_invalid_and_only_a_record(w):
    zm = ZScripted(blocks={u_edit(0.0, "chain_entry"): {"MBON05->MBON09": 6, "MBON05->MBON11": 2,
                                                        "MBON05->MBON01": 2}})
    out = _to(w, w.scripted(), zm, "scan").stage_scan()
    assert out["outcome"] == "PASS" and out["contrast"]["readings"]["entry"]["reading"] == u_rules.INVALID
    assert out["contrast"]["entry_f0"]["invalid"] and out["contrast"]["readings"]["block"]["reading"] != "INVALID"


def test_kc_qualifies_a_subset_without_substitutes(w):
    m = w.scripted(kc={u_edit(0.5): {"M3|T": 0.02}})
    r = _to(w, m, ZScripted(), "kc")
    out = r.stage_kc()
    assert out["outcome"] == "PASS" and out["qualified"] == [0.6, 0.7]
    assert out["points"]["0.5"]["failed"] == ["T 세트 M3|T"]


def test_kc_stops_when_no_checked_f_qualifies(w):
    low = {u_edit(f): {f"C{i}|T": 0.01 for i in range(112)} for f in (0.5, 0.6, 0.7)}
    out = _to(w, w.scripted(kc=low), ZScripted(), "kc").stage_kc()
    assert out["outcome"] == u_rules.STOP_NO_QUALIFIED_F and out["kind"] == u_rules.NQ_KC
    assert out["sentence"] == ("가드 스캔 통과 후보 {0.5, 0.6, 0.7, 0.8, 0.9} 가운데 작은 f부터 검사한 {0.5, 0.6, 0.7}이 모두 "
                               "KC 대역에서 떨어졌다(검사하지 않은 후보 {0.8, 0.9}).")


def test_even_repro_stop_leaves_l_unmeasured(w):
    m = w.scripted()
    r = _to(w, m, ZScripted(), "even")
    w.r_even_plan["C"] = dict(list(w.r_even_plan["C"].items())[:6])
    out = r.stage_even()
    assert out["outcome"] == u_rules.STOP_EVEN_REPRO and out["per_f"] == {}
    assert out["sentence"] == "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L 16, C 6)."
    assert not [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    with pytest.raises(SystemExit):
        r.stage_choose()


def test_even_refuses_when_rs_even_raw_is_missing(w):
    r = _to(w, w.scripted(), ZScripted(), "even")

    def missing(rows, name):
        raise ValueError("R's even raw for C lacks 1 pair(s)")
    w.ctx["r_even"] = missing
    with pytest.raises(SystemExit) as e:
        r.stage_even()
    assert e.value.code == 2 and "even" not in doc()


def test_even_invalid_on_wrong_edges(w):
    m = w.scripted(override={("even", "L"): dict(edges=1)})
    r = _to(w, m, ZScripted(), "even")
    assert r.stage_even()["outcome"] == u_rules.INVALID


def test_choose_punish_stop(w):
    m = w.scripted()
    eb, ea = w.keys(w.even, "b"), w.keys(w.even, "a")
    for f in (0.5, 0.6):
        m.plan[("even", u_edit(f))] = {**{k: (True, True, False) for k in eb[:13]},
                                       **{k: (False, False, False) for k in eb[13:16]}}
    r = _to(w, m, ZScripted(), "choose")
    out = r.stage_choose()
    assert out["outcome"] == u_rules.STOP_EVEN_PUNISH and out["kept"] == []
    assert out["sentence"].endswith("(f 0.5: (b) 3·(a) 0; f 0.6: (b) 3·(a) 0; f 0.7: (b) 0·(a) 2).")
    with pytest.raises(SystemExit):
        r.stage_smoke()


def test_choose_low_lever_names_f_star_and_the_checked_f(w):
    m = w.scripted()
    for f in (0.5, 0.6):
        m.plan[("even", u_edit(f))] = {k: (True, True, False) for k in w.keys(w.even, "b")[:10]}
    r = _to(w, m, ZScripted(), "choose")
    out = r.stage_choose()
    assert out["outcome"] == u_rules.STOP_EVEN_LOW_LEVER and out["f_star"] == 0.6
    assert out["sentence"].endswith("(f* = 0.6, 검사한 f {0.5, 0.6, 0.7})")
    with pytest.raises(SystemExit):
        r.stage_smoke()


def test_smoke_problem_blocks_later_stages(w):
    m = w.scripted(override={("smoke", "L"): dict(edges=1)})
    r = _to(w, m, ZScripted(), "smoke")
    assert r.stage_smoke()["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_oc_refuses_a_cluster_table_other_than_ts_fixture(w, monkeypatch):
    r = _to(w, w.scripted(), ZScripted(), "oc")
    real = UR.u_rules.oc_cluster
    monkeypatch.setattr(UR.u_rules, "oc_cluster", lambda spec: dict(real(spec), sha256="0" * 64))
    with pytest.raises(SystemExit):
        r.stage_oc()
    assert "oc" not in doc()


@pytest.mark.parametrize("drop,outcome", [
    ({u_edit(0.6): 6}, u_rules.STOP_PUNISH_WEAKENED), ({u_edit(0.6): 3}, u_rules.STOP_PUNISH_BROKEN),
    ({"none": 3}, u_rules.STOP_P_REFERENCE)])
def test_gate2_stops_block_the_judgement(w, drop, outcome):
    r = _to(w, w.scripted(drop=drop), ZScripted(), "gate2")
    out = r.stage_gate2()
    assert out["outcome"] == outcome and out["sentence"]
    with pytest.raises(SystemExit):
        r._require("jm:L")
    assert w.judgement_calls == 0


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    m = w.scripted()
    r = _to(w, m, ZScripted(), "gate2")
    m.arm_edges[u_edit(0.6)] = 1
    assert r.stage_gate2()["outcome"] == u_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)
    del m.arm_edges[u_edit(0.6)]
    fixed = w.runner(m, pipeline={"key": "q" * 64})
    assert fixed.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["rerun_of"] == d["gate2_invalid"]["written_at"]
    with pytest.raises(SystemExit):
        fixed.stage_gate2(rerun=True)


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    r = w.runner(w.scripted())
    with pytest.raises(SystemExit):
        r.stage_path()
    r.stage_reuse()
    with pytest.raises(SystemExit):
        r.stage_reuse()
    monkeypatch.setattr(UR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_path()
    monkeypatch.setattr(UR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(UR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/u_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_path()
    assert set(doc()) == {"reuse"}
    assert json.loads(Path(SPEC.summary).read_text())["reuse"]["outcome"] == "PASS"


def test_one_measurer_per_z(w):
    made = []
    m = w.scripted()

    def measure(z):
        made.append(tuple(sorted(z.items())))
        return m.at(z)
    r = UR.Runner(measure, ZScripted(), w.ctx, SPEC, code={"key": SPEC.r_shared_key}, tcode={"key": "t" * 64},
                  ucode={"key": "u" * 64}, pipeline={"key": "p" * 64}, archive_root=w.archive)
    through(w, m, r=r)
    assert len(made) == len(set(made)) == 2                              # h4 z and z_f (0.5-0.7 share z_f here)


def test_hashed_and_decision_files():
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import T_MEASURE_FILES
    from flymon.brain.t_runner import T_HASHED_FILES
    from flymon.brain.u_measure import U_MEASURE_FILES
    assert set(T_HASHED_FILES) <= set(UR.U_HASHED_FILES)
    assert set(UR.U_PIPELINE_FILES) | set(U_MEASURE_FILES) | {"results/summary/t_lever.json"} <= set(UR.U_HASHED_FILES)
    assert not set(UR.DECISION_FILES) & (set(R_MEASURE_FILES) | set(T_MEASURE_FILES) | set(U_MEASURE_FILES))
    assert "flymon/brain/u_rules.py" in UR.DECISION_FILES and "tests/brain/fixtures/t_oc_cluster.json" in \
        UR.DECISION_FILES


ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not (NPZ.exists() and (ROOT / "results/r/cache/r_oracle").exists()),
                    reason="no connectome or no R raw cache")
def test_rs_even_raw_is_found_by_content_key_through_r_raw_spec():
    """U.9.1 / U.3 4: R's even L and C raw (h4 z, H.4 seeds) are found by content key under U's numbers with T's lever
    edit (r_raw_spec), with no pool and nothing written; U's own unresolved lever would miss every L entry."""
    from flymon.agent.config import load_c3_config
    from flymon.brain import q_pairs, r_pairs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
    from flymon.brain.t_runner import r_even_reader
    from flymon.brain.u_spec import SPEC as U_SPEC
    from flymon.brain.u_store import RReadCache
    cfg = load_c3_config(U_SPEC.m0d_summary)
    enc = json.loads((ROOT / U_SPEC.encoder_summary).read_text())
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = r_pairs.even_rows(pops, rc, enc, U_SPEC)
    q_pairs.check_strength(enc, U_SPEC)
    m0d = json.loads((ROOT / U_SPEC.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    cache = RReadCache(str(ROOT / U_SPEC.r_cache_dir), code_key(str(NPZ), files=R_MEASURE_FILES))
    m = RMeasurer(None, cache, U_SPEC, cfg.params, cfg.readout, cfg.z, types, len(pops.kc))
    read = r_even_reader(cache, m, UR.r_raw_spec(U_SPEC))
    for n in ("L", "C"):
        got = read(rows[:U_SPEC.path_even_n], n)
        assert len(got) == 3 and m.last_jobs == 0
    assert [g["result"]["q"]["edit"] for g in read(rows, "L")] == ["apl_to_mbon05_zero"] * 39
    with pytest.raises(ValueError, match="lacks 39 pair"):
        r_even_reader(cache, m, U_SPEC)(rows, "L")
