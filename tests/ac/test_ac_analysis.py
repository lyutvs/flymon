"""AC stage-1 reading (AC.5, AC.8): criterion 1 (FLY - C-off and FLY - FLY-TB at battle 40, >= 0.15 and CI above 0),
criteria 2 and 3 (> 0 and CI above 0, descriptive), 4 and 5 not measured, the minimum valid flies, the sentences."""
import pytest

from flymon.ac import analysis as an
from flymon.ac.spec import FORBIDDEN, LABEL, SPEC


def rows_for(arm, n, start=0, won=None, invalid=()):
    return [dict(fly=start + i, arm=arm, k=i, invalid=(i in invalid),
                 eval_battles=[dict(won=(won(i, b) if won else False), finished=True) for b in range(20)])
            for i in range(n)]


def sit(arm, fly, point, switched):
    return dict(kind="situation_eval", arm=arm, fly=fly, point=point, switched=switched, frozen=True)


def brain_world(fly_rate=0.8, coff_rate=0.2, tb_rate=0.1, fly_invalid=()):
    rows = rows_for("FLY", 12, invalid=fly_invalid) + rows_for("COFF", 6, 12) + rows_for("TB", 6, 18)
    sits = []
    for r in rows:
        rate = {"FLY": fly_rate, "COFF": coff_rate, "TB": tb_rate}[r["arm"]]
        sw = [1] * int(round(20 * rate)) + [0] * (20 - int(round(20 * rate)))
        sits.append(sit(r["arm"], r["fly"], 0, [0] * 20))
        if r["arm"] != "COFF":
            sits += [sit(r["arm"], r["fly"], p, sw) for p in (10, 20, 30, 40)]
        else:
            sits[-1] = sit("COFF", r["fly"], 0, sw)
    return rows, sits


def test_c1_units_take_point_40_and_coff_point_0():
    rows, sits = brain_world()
    u = an.c1_units(sits, rows, SPEC)
    assert len(u["FLY"]) == 12 and sum(u["FLY"][0]) == 16 and sum(u["COFF"][12]) == 4 and sum(u["TB"][18]) == 2


def test_criterion_1_met_and_sentences_clean():
    rows, sits = brain_world()
    res = {"BRAIN": dict(per_fly=rows, mode="run", eval_digest="e"),
           "RND": dict(per_fly=rows_for("RND", 16), mode="run", eval_digest="e"),
           "MAX": dict(per_fly=rows_for("MAX", 16, won=lambda i, b: b % 2 == 0), mode="run", eval_digest="e")}
    doc = an.analyse(res, sits, [], [], SPEC, weak={"WEAK-RND": 0.145, "WEAK-MAX": 0.485})
    c1 = [c for c in doc["criteria"] if c["criterion"] == 1]
    assert [(c["a"], c["b"], c["status"]) for c in c1] == [("FLY", "COFF", "충족"), ("FLY", "TB", "충족")]
    assert c1[0]["diff"] == pytest.approx(0.6) and c1[1]["diff"] == pytest.approx(0.7)
    assert {c["criterion"]: c["status"] for c in doc["criteria"] if c["criterion"] in (4, 5)} == {4: "미측정",
                                                                                                    5: "미측정"}
    assert doc["c1_summary"].startswith("배틀로 학습한 마리에서 상성 조건부 선택이 관찰됐다")
    assert doc["label"] == LABEL and doc["weak_pilot"] == {"WEAK-RND": 0.145, "WEAK-MAX": 0.485}
    for s in doc["sentences"] + [doc["c1_summary"]]:
        assert LABEL in s and not any(w in s for w in FORBIDDEN), s


def test_too_few_valid_flies_is_measurement_failure():
    rows, sits = brain_world(fly_invalid=(0, 1, 2, 3))           # FLY valid 8 < 9
    res = {"BRAIN": dict(per_fly=rows, mode="run", eval_digest="e"),
           "RND": dict(per_fly=rows_for("RND", 16), mode="run", eval_digest="e"),
           "MAX": dict(per_fly=rows_for("MAX", 16), mode="run", eval_digest="e")}
    doc = an.analyse(res, sits, [], [], SPEC)
    c1 = [c for c in doc["criteria"] if c["criterion"] == 1]
    assert all(c["status"] == "측정 불성립" and c["n_a"] == 8 and c["min_a"] == 9 for c in c1)
    assert "측정 불성립" in doc["c1_summary"]


def test_criterion_1_not_met():
    rows, sits = brain_world(fly_rate=0.3, coff_rate=0.2, tb_rate=0.25)
    res = {"BRAIN": dict(per_fly=rows, mode="run", eval_digest="e"),
           "RND": dict(per_fly=rows_for("RND", 16), mode="run", eval_digest="e"),
           "MAX": dict(per_fly=rows_for("MAX", 16), mode="run", eval_digest="e")}
    doc = an.analyse(res, sits, [], [], SPEC)
    assert all(c["status"] == "미충족" for c in doc["criteria"] if c["criterion"] == 1)
    assert doc["c1_summary"].startswith("배틀로 학습한 마리의 상성 조건부 선택이 탐색 기준에 못 미쳤다")


def test_c2_counts_informative_fly_turns_only():
    rows = rows_for("FLY", 1) + rows_for("COFF", 1, 1)
    recs = [dict(kind="decision", decider="fly", fly=0, candidates=["a", "b"], chosen="a", multipliers=[2.0, 1.0]),
            dict(kind="decision", decider="fly", fly=0, candidates=["a", "b"], chosen="b", multipliers=[2.0, 1.0]),
            dict(kind="decision", decider="fly", fly=0, candidates=["a", "b"], chosen="b", multipliers=[1.0, 1.0]),
            dict(kind="decision", decider="coach", fly=0, candidates=[], chosen="x"),
            dict(kind="decision", decider="fly", fly=1, candidates=["a", "b", "c"], chosen="c",
                 multipliers=[0.5, 1.0, 2.0])]
    u = an.c2_units(recs, rows)
    assert u == {"FLY": {0: [1, 0]}, "COFF": {1: [1]}}


def test_kc_ratio_by_arm():
    rows = rows_for("FLY", 1) + rows_for("TB", 1, 1)
    recs = [dict(kind="decision", decider="fly", fly=0, kc_ratio_gt2=True),
            dict(kind="decision", decider="fly", fly=0, kc_ratio_gt2=False),
            dict(kind="decision", decider="coach", fly=0)]
    assert an.kc_ratio_by_arm(recs, rows) == {"FLY": 0.5, "TB": None}


def test_digest_mismatch_refused():
    ok = dict(mode="run", eval_digest="e")
    an.check_digests({"BRAIN": ok, "RND": ok, "MAX": ok})
    with pytest.raises(SystemExit, match="evaluation schedule"):
        an.check_digests({"BRAIN": ok, "RND": dict(ok, eval_digest="x"), "MAX": ok})
    with pytest.raises(SystemExit, match="mode"):
        an.check_digests({"BRAIN": dict(ok, mode="smoke"), "RND": ok, "MAX": ok})


# ---- controller rulings (Task 17) ------------------------------------------------------------------------------------
import dataclasses  # noqa: E402

FAST = dataclasses.replace(SPEC, boot_draws=200)


def world_results(rows, rnd=None, maxr=None, book=None):
    return {"BRAIN": dict(per_fly=rows, mode="run", eval_digest="e", book=book or {"invalid": {}, "stopped": {}}),
            "RND": dict(per_fly=rnd or rows_for("RND", 16), mode="run", eval_digest="e"),
            "MAX": dict(per_fly=maxr or rows_for("MAX", 16), mode="run", eval_digest="e")}


def test_boot_draws_and_seed_are_the_spec_values():
    rows, sits = brain_world()
    doc = an.analyse(world_results(rows), sits, [], [], SPEC)
    assert doc["boot"] == dict(draws=10_000, seed=304)
    assert any("6마리 팔의 CI는 불안정할 수 있다" in n for n in doc["notes"])


def test_zero_valid_flies_never_reach_the_bootstrap(monkeypatch):
    rows, sits = brain_world()
    for r in rows:
        if r["arm"] == "COFF":
            r["invalid"] = True
    called = []
    monkeypatch.setattr(an.boot, "two_stage_diff", lambda *a, **k: called.append(a) or {})
    c = an.contrast(1, "primary", "FLY", "COFF", an.c1_units(sits, rows, FAST), FAST, FAST.min_effect)
    assert c["status"] == "측정 불성립" and c["n_b"] == 0 and not called


def test_stopped_arm_is_measurement_failure_with_its_invalid_count_at_stop():
    rows, sits = brain_world()
    for r in rows:
        if r["arm"] == "TB":
            r["invalid"], r["invalid_reason"] = True, ("AL-f18-b003: unfinished after 3 retries" if r["fly"] in (18, 19, 20)
                                                       else "arm_stopped")
    book = {"invalid": {}, "stopped": {"TB": dict(status="STOP_INFRA", n_invalid=3, size=6, min_valid=4)}}
    doc = an.analyse(world_results(rows, book=book), sits, [], [], FAST)
    tb = [c for c in doc["criteria"] if c["criterion"] == 1 and c["b"] == "TB"][0]
    assert tb["status"] == "측정 불성립" and tb["stopped_b"] is True
    assert doc["stopped"]["TB"]["n_invalid"] == 3
    assert doc["n_valid"]["TB"] is None                                  # never read for a stopped arm
    assert doc["invalid_reasons"]["TB"] == {"unfinished_after_retries": 3}   # arm_stopped skips are not failures
    assert doc["n_arm_stopped"]["TB"] == 3
    assert "STOP_INFRA" in [s for s in doc["sentences"] if "FLY − FLY-TB" in s][0]


def test_nobrain_rows_are_keyed_by_arm_and_k():
    rows, _ = brain_world()
    u = an.c3_units(rows + rows_for("RND", 16) + rows_for("MAX", 16))
    assert set(u["RND"]) == {("RND", k) for k in range(16)} and len(u["FLY"]) == 12
    assert all(len(v) == 20 for arm in u.values() for v in arm.values())


def test_criteria_4_and_5_are_never_judged_and_sentences_say_not_measured():
    rows, sits = brain_world()
    doc = an.analyse(world_results(rows), sits, [], [], FAST)
    c45 = [c for c in doc["criteria"] if c["criterion"] in (4, 5)]
    assert all(c["status"] == "미측정" and "diff" not in c for c in c45)
    s45 = [s for s in doc["sentences"] if "4.3 기준 4" in s or "4.3 기준 5" in s]
    assert len(s45) == 2 and all("탐색 기준 미측정" in s and LABEL in s for s in s45)


def test_upper_bound_and_secondary_sentences():
    rows, sits = brain_world()
    for r in rows:
        if r["arm"] == "FLY":
            r["eval_battles"] = [dict(won=(b % 4 == 0), finished=True) for b in range(20)]
    maxr = rows_for("MAX", 16, won=lambda i, b: b % 2 == 0)
    doc = an.analyse(world_results(rows, maxr=maxr), sits, [], [], FAST)
    by = {(c["criterion"], c["a"], c["b"]): c for c in doc["criteria"] if "a" in c}
    assert by[(3, "MAX", "FLY")]["status"] == "기술" and by[(3, "MAX", "FLY")]["diff"] == pytest.approx(0.25)
    assert by[(3, "FLY", "RND")]["status"] == "충족"
    s3 = [s for s in doc["sentences"] if s.startswith("FLY − RND")][0]
    assert "(기술 통계, 다중 비교 미보정)" in s3 and "6마리 팔" not in s3
    s1 = [s for s in doc["sentences"] if s.startswith("FLY − C-off")][0]
    assert "(6마리 팔의 CI는 불안정할 수 있다)" in s1 and "다중 비교 미보정" not in s1
    assert by[(2, "FLY", "COFF")]["status"] == "측정 불성립"            # no evaluation decision records given


def test_kc_ratio_reaches_every_sentence():
    rows, sits = brain_world()
    recs = [dict(kind="decision", decider="fly", fly=0, kc_ratio_gt2=True),
            dict(kind="decision", decider="fly", fly=1, kc_ratio_gt2=False)]
    doc = an.analyse(world_results(rows), sits, [], recs, FAST)
    assert doc["kc_ratio_gt2"]["FLY"] == 0.5
    assert all("KC 비율 > 2 턴 비율(FLY): 0.500" in s for s in doc["sentences"])


def test_c2_skips_malformed_decisions():
    rows = rows_for("FLY", 1)
    recs = [dict(kind="decision", decider="fly", fly=0, candidates=["a", "b"], chosen=None, multipliers=[2.0, 1.0]),
            dict(kind="decision", decider="fly", fly=0, candidates=["a", "b"], chosen="a", multipliers=[2.0]),
            dict(kind="decision", decider="fly", fly=0, candidates=["a", "b"], chosen="a", multipliers=None)]
    assert an.c2_units(recs, rows) == {}


def _script():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts/analyze_ac_m4.py"
    spec = importlib.util.spec_from_file_location("analyze_ac_m4", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_stage1(tmp_path, complete=True):
    import json
    rows, sits = brain_world()
    root = tmp_path / "results/m4/stage1"
    per = {"BRAIN": rows, "RND": rows_for("RND", 16), "MAX": rows_for("MAX", 16)}
    for arm, pf in per.items():
        (root / arm / "logs").mkdir(parents=True)
        (root / arm / "result.json").write_text(json.dumps(dict(arm=arm, mode="run", eval_digest="e", per_fly=pf,
                                                                book={"invalid": {}, "stopped": {}},
                                                                complete=complete, label=LABEL)))
    lines = lambda recs: "".join(json.dumps(r) + "\n" for r in recs)          # noqa: E731
    (root / "BRAIN/logs/situations_init.jsonl").write_text(lines([s for s in sits if s["point"] == 0]))
    (root / "BRAIN/logs/situations.jsonl").write_text(lines([s for s in sits if s["point"] != 0]))
    (root / "BRAIN/logs/fly00.jsonl").write_text(lines([dict(kind="decision", decider="fly", fly=0,
                                                             kc_ratio_gt2=True)]))
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({"arms": {"WEAK-RND": {"win_rate": 0.145}, "WEAK-MAX": {"win_rate": 0.485}}}))
    return m1


def test_script_writes_the_summary_with_label(tmp_path, monkeypatch):
    import json
    m1 = _write_stage1(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _script().main(["--m1", str(m1)]) == 0
    doc = json.loads((tmp_path / an.RESULT).read_text())
    assert doc["label"] == LABEL and doc["weak_pilot"] == {"WEAK-RND": 0.145, "WEAK-MAX": 0.485}
    assert [c["status"] for c in doc["criteria"] if c["criterion"] == 1] == ["충족", "충족"]
    assert doc["kc_ratio_gt2"]["FLY"] == 1.0 and set(doc["sources"]) == {"BRAIN", "RND", "MAX"}


def test_script_refuses_incomplete_arm(tmp_path, monkeypatch):
    m1 = _write_stage1(tmp_path, complete=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="not complete"):
        _script().main(["--m1", str(m1)])
