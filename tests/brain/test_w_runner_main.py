"""W's stage chain from the main set to the judgement (W.9.9 순서 6-14): the set check, the oracle on z_V with
24_700_xxx (the set is used from here), the naive screen in declared order stopping at 8 gate pairs (no pair beyond
them measured; unbalanced pairs skipped), STOP_FEW_PAIRS, every RN unit carrying both phases from its own pre, the
learning / band / record manifests holding no statistic, the seal and one judgement (PASS and FAIL)."""
import dataclasses
import json
import re
from pathlib import Path

import pytest

from flymon.brain import w_rules
from flymon.brain import w_runner as WR
from flymon.brain.r_pairs import row_key
from tests.brain.w_world import MAIN, SPEC, World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_judge_pass(w):
    out = through(w, "judge")
    d = doc()
    assert set(d) >= set(WR.ORDER) and len(d["ledger"]) == len(WR.ORDER)
    assert all(d[b]["w_measure_key"] == "w" * 64 for b in WR.ORDER)
    assert [c["diffs"] for c in d["path"]["checks"]] == [[]] * len(d["path"]["checks"])
    assert len(d["path"]["checks"]) == 8 + 6 + 1
    assert d["pilot"]["outcome"] == "PASS" and d["pilot"]["exploratory"]["label"] == "탐색"
    assert d["oc"]["selected"] == dict(q=0.75, K=8, F=8, cost_h=d["oc"]["selected"]["cost_h"])
    assert d["budget"]["plan"]["with_c"] is True
    assert d["set"]["set"]["n_b"] == 167 and d["oracle"]["n"] == 249
    assert ("screen", "L", 249, tuple(d["reuse"]["z_V"]["A"])) in w.oracle_calls
    testable = [p for p in d["oracle"]["pairs"] if p["testable"]]
    g = d["gates"]["gates"]
    assert [x["c"] for x in g] == [p["c"] for p in testable[:8]]
    assert [x["c"] for x in d["naive"]["screened"]] == [p["c"] for p in testable[:8]]
    assert all("stats" not in d[s] for s in ("learn", "band", "records"))
    assert d["learn"]["n_units"] == 8 * 8 * 3 and d["band"]["n_units"] == 8 * 8 * 3
    assert d["records"]["n_units"] == 8 * 8 * 3 + 2 * 2
    assert d["seal"]["status"] == "SEALED" and Path(d["seal"]["archive"]["dir"]).exists()
    assert out["verdict"] == "PASS" and out["sentence"].startswith("조합 지렛대 모델(")
    assert "설계 q 0.75 · K 8 · F 8 · k 상한 8" in out["sentence"]
    assert all(n["counts_equal"] and n["weights_equal"] for n in out["records"]["noplast"])
    assert Path(WR.DONE_MARKER).exists()
    # W.9.10 2 / W.10: the drift-only level-gate d′ and the mixed-fly P(PASS) reach the stored blocks and the judge
    assert d["oc"]["drift_dprime"]["population"] == {"reward_level": 0.1}
    assert out["records"]["oc"]["mixed_flies"] == {"g0.0": [0.1] * 5}
    assert out["records"]["oc"]["drift_dprime"] == d["oc"]["drift_dprime"]


def test_estimate_recosts_the_plan_design_at_stage_8(w):
    """W.9.9 P1-5: block estimate re-costs only the plan's design, [with C, without C], on the real testable count
    (the oracle already done); C is dropped first, then STOP_BUDGET."""
    through(w, "estimate")
    d = doc()
    e = d["estimate"]
    assert e["budget_stage"] == "8" and e["outcome"] == "PASS" and e["plan"]["with_c"] is True
    assert e["plan"]["design"] == d["budget"]["plan"]["design"] and e["plan"]["parts_h"]["oracle"] == 0.0
    assert e["n_testable"] == d["oracle"]["n_testable"]


def test_estimate_drops_c_then_stops(w, monkeypatch):
    from flymon.brain import w_records
    through(w, "oracle")
    calls = []
    real = w_rules.budget

    def spy(elapsed, opts, spec, stage):
        calls.append([(o["design"], o["with_c"]) for o in opts])
        return real(elapsed, opts, spec, stage)
    monkeypatch.setattr(WR.w_rules, "budget", spy)
    c_h = lambda wc: w_records.design_cost(doc()["smoke"]["costs"], 8, 8, SPEC, 0, doc()["oracle"]["n_testable"], wc)  # noqa
    tight = dataclasses.replace(SPEC, budget_h=WR.w_records.elapsed_h(doc()["ledger"]) + c_h(False)["total_h"]
                                + 0.5 * c_h(True)["parts_h"]["c"])
    e = w.runner(spec=tight).stage_estimate()
    assert e["outcome"] == "PASS" and e["plan"]["with_c"] is False
    assert [w_ for _, w_ in calls[0]] == [True, False] and calls[0][0][0] == calls[0][1][0]


def test_naive_stops_at_eight_gate_pairs_and_unbalanced_pairs_skip(w):
    testable = [r for r in MAIN if r["c"] % 3 == 0]
    for r in testable[:3]:
        w.model.offset[row_key(r)] = 40.0                               # naive-unbalanced: not a gate pair
    through(w, "gates")
    d = doc()
    sc = d["naive"]["screened"]
    assert [x["gate"] for x in sc[:3]] == [False] * 3 and len(sc) == 11 and sum(x["gate"] for x in sc) == 8
    measured = {json.loads(Path(m["cache_file"]).read_text())["inputs"]["pair"] for m in d["naive"]["manifest"]}
    assert measured == {x["key"] for x in sc}


def test_few_pairs_stop(w):
    for r in MAIN:
        w.model.offset[row_key(r)] = 40.0
    through(w, "naive")
    out = w.runner().stage_gates()
    assert out["outcome"] == w_rules.STOP_FEW_PAIRS and out["sentence"] == (
        "W 주 세트 249쌍에서 순진 프로브 균형·오라클 시험 가능 쌍이 0개로 최소 4에 못 미쳤다.")      # W.9.10 4


def test_rn_units_run_both_phases_from_pre(w):
    through(w, "gates")
    r = w.runner()
    us = r.learn_units(doc())
    rn = [u for u in us if u["brain"] == "RN"]
    assert rn and all([p[0] for p in u["phases"]] == ["PAM08", None] for u in rn)
    assert all(u["phases"][0][2] == SPEC.train_base(u["idx"], 0) and u["probe_seeds"][0] ==
               SPEC.probe_seeds(u["idx"], u["fly"], 1)[0] for u in rn)                # RN starts from its own pre
    r0 = [u for u in us if u["brain"] == "R"][0]
    c = r0["idx"]
    assert r0["phases"] == [["PAM08", 20, 28_000_000 + c * 40_000], ["PPL105", 20, 28_000_000 + c * 40_000 + 20]]
    assert r0["probe_seeds"] == [26_000_000 + c * 4_000 + k for k in range(8)]
    b = r.band_units(doc())
    assert b[0]["probe_seeds"] == [26_000_000 + c * 4_000 + k for k in range(8, 16)]


def test_judge_fail_and_once(w):
    gates_c = [r for r in MAIN if r["c"] % 3 == 0][:8]
    w.model.effects[row_key(gates_c[0])] = (30.0, 0.0)                 # one pair learns no punishment
    out = through(w, "judge")
    assert out["verdict"] == "FAIL" and out["judgement"]["failing"] == [row_key(gates_c[0])]
    assert out["sentence"].startswith("F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부(")
    w.judge_commits.append("x")
    with pytest.raises(SystemExit):
        w.runner().stage_judge()


# W.9.8 H3: "배치마다 원장 갱신, 추정이 상한을 넘으면 그 시점에 STOP_BUDGET" — naive / learn / band / records alike
H_STOP = re.compile(r"^남은 추정 비용 누적 (\d+\.\d\d) h \+ 남은 (\d+\.\d\d) h = (\d+\.\d\d) h가 W 상한 24 h를 넘는다\.$")
IN_STAGE = {"naive": (("naive",), ("learn", "band", "c", "noplast"), "gates"),
            "learn": (("learn",), ("band", "c", "noplast"), "band"),
            "band": (("band",), ("c", "noplast"), "records"),
            "records": (("c", "noplast"), (), "seal")}


@pytest.mark.parametrize("stage", list(IN_STAGE))
def test_in_stage_budget_stop(w, monkeypatch, stage):
    """Each pool round costs one scripted hour; the budget admits the stage's estimate + the later stages' + 0.5 h, so
    the first round passes and the ledger stops the stage at the next one: a STOP_BUDGET block with the same 〈h〉
    format as w_rules.budget, the stage's progress kept, and W stopped there."""
    parts, after, nxt = IN_STAGE[stage]
    through(w, WR.ORDER[WR.ORDER.index(stage) - 1])
    r0 = w.runner()
    base = WR.w_records.elapsed_h(doc()["ledger"])
    need = r0._later_h(doc(), parts) + r0._later_h(doc(), after)
    tight = dataclasses.replace(SPEC, budget_h=base + need + 0.5)
    clock, real = [0.0], WR.time.perf_counter
    monkeypatch.setattr(WR.time, "perf_counter", lambda: real() + clock[0])
    run_jobs = w.pool.run_jobs

    def hour(fn, kws):
        clock[0] += 3600.0
        return run_jobs(fn, kws)
    monkeypatch.setattr(w.pool, "run_jobs", hour)
    jobs0 = w.pool.jobs
    out = getattr(w.runner(spec=tight), f"stage_{stage}")()
    assert out["outcome"] == w_rules.STOP_BUDGET and stage in WR.GATES
    m = H_STOP.match(out["sentence"])
    assert m, out["sentence"]
    x, y, z = map(float, m.groups())
    assert x >= round(base, 2) + 1.0 and abs(x + y - z) <= 0.011 and z > tight.budget_h
    assert out["reasons"] == [f"누적 {m.group(1)} h + 남은 {m.group(2)} h = {m.group(3)} h"]
    assert 0 < w.pool.jobs - jobs0 < {"naive": 8 * 11}.get(stage, 10 ** 6)
    assert doc()[stage]["outcome"] == w_rules.STOP_BUDGET and "manifest" not in doc()[stage]
    assert w.runner()._prog(stage) >= 3600.0
    with pytest.raises(SystemExit):
        getattr(w.runner(), f"stage_{nxt}")()
