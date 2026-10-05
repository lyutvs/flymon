"""W's stage chain up to the budget gate (W.9.9 순서 0-5a): stage0 -> reuse -> path -> pilot -> oc -> smoke -> budget;
refusals (missing / later / own block, a stopped gate, uncommitted summary), exit 7 when the reuse condition breaks
after reuse, STOP_REUSE, the path gate's three checks (V gate ②'s punish-arm rows, V's oracle naive counts, the reward
path) with STOP_W_PATH_REPRO and INVALID, the pilot's H6 stop, STOP_OC_UNREACHABLE, smoke on its own seed block and
the budget gate's order (C dropped first, then the alternative design, then STOP_BUDGET)."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain import w_rules
from flymon.brain import w_runner as WR
from flymon.brain.r_pairs import row_key
from flymon.brain.w_spec import smoke
from tests.brain.w_world import MAIN, PILOT, SPEC, World, doc, fake_oc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_budget(w):
    through(w, "budget")
    d = doc()
    assert set(d) == set(WR.ORDER[:WR.ORDER.index("budget") + 1]) | {"ledger"}
    assert [e["stage"] for e in d["ledger"]] == list(WR.ORDER[:WR.ORDER.index("budget") + 1])
    assert all(d[b]["w_measure_key"] == "w" * 64 and d[b]["pipeline_key"] == "p" * 64 for b in WR.ORDER[:7])
    assert d["stage0"]["outcome"] == "PASS" and d["stage0"]["tables"]["protocol"]["pulse_ms"] == 400.0
    assert len(d["path"]["checks"]) == 8 + 6 + 1 and all(c["diffs"] == [] for c in d["path"]["checks"])
    assert d["pilot"]["n_units"] == 16 * 8 * 3 and d["pilot"]["exploratory"]["label"] == "탐색"
    assert d["oc"]["selected"]["F"] == 8 and d["budget"]["plan"]["with_c"] is True
    assert "set" not in d and w.oracle_calls == [("smoke", "L", 1, tuple(d["reuse"]["z_V"]["A"]))]


def test_refusals_and_order(w):
    with pytest.raises(SystemExit) as e:
        w.runner().stage_reuse()
    assert e.value.code == 2
    through(w, "reuse")
    with pytest.raises(SystemExit):
        w.runner().stage_reuse()
    w.facts["ancestors"]["928eaad"] = False
    with pytest.raises(SystemExit) as e:
        w.runner().stage_path()
    assert e.value.code == 7


def test_reuse_stop(w):
    w.v["judge"]["band"] = "B_Tb"
    through(w, "stage0")
    out = w.runner().stage_reuse()
    assert out["outcome"] == w_rules.STOP_REUSE and "V 판정 B_Tb" in out["sentence"]
    with pytest.raises(SystemExit):
        w.runner().stage_path()


def test_path_stop_on_v_rows_and_naive(w):
    through(w, "reuse")
    real = w.v_gate2_rows
    w.ctx["v_gate2_rows"] = lambda items: [dict(r, w_post_sha256="x") if i == 0 else r
                                           for i, r in enumerate(real(items))]
    out = w.runner().stage_path()
    assert out["outcome"] == w_rules.STOP_W_PATH_REPRO
    assert out["sentence"].startswith("W 학습 경로가 V 관문 ② 처벌 팔 행(L_V r1 25400000)을 재현하지 못했다(")


def test_path_naive_mismatch_and_invalid_edit(w, monkeypatch):
    through(w, "reuse")

    def jm(name, n):
        out = World.v_jm(w, name, n)
        if name == "C":
            out[1][1]["report"]["pre"]["A"][0][0] += 1
        return out
    w.ctx["v_jm"] = jm
    out = w.runner().stage_path()
    assert out["outcome"] == w_rules.STOP_W_PATH_REPRO and [c["ref"] for c in out["failed"]] == [
        f"V 오라클 순진 카운트(C {row_key(World.v_jm(w, 'C', 3)[1][0])})"]
    real = w.model.job
    monkeypatch.setattr(w.model, "job", lambda kw, pair: dict(real(kw, pair), edit_edges=3))
    doc_ = doc()
    doc_.pop("path")
    Path(SPEC.summary).write_text(json.dumps(doc_))
    out = w.runner().stage_path()
    assert out["outcome"] == w_rules.INVALID and "edit_edges 3" in out["reasons"][0]


def test_pilot_no_effect_stops(w):
    for r in w.ctx["pilot_rows"]():
        w.model.effects[row_key(r)] = (30.0, 0.0)                    # reward learns, punishment does nothing
    through(w, "path")
    out = w.runner().stage_pilot()
    assert out["outcome"] == w_rules.STOP_PILOT_NO_EFFECT and out["sentence"].startswith(
        "파일럿에서 조합 지렛대의 F.2 학습 효과가 (i) 보상 연합 d′ ≥ 0.5 쌍 비율 1.000·처벌 연합 d′ ≤ −0.5 쌍 비율 0.000")
    with pytest.raises(SystemExit):
        w.runner().stage_oc()


def test_oc_unreachable(w):
    through(w, "pilot")
    fake_oc.none = True
    out = w.runner().stage_oc()
    assert out["outcome"] == w_rules.STOP_OC_UNREACHABLE
    assert out["sentence"].startswith("파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다(")
    assert "q 0.5·K 8·F 32: k 4 검정력 하한 0.500 / 거짓 통과 상한 0.010" in out["sentence"]   # W.9.10 3


def test_budget_drops_c_then_alt_then_stops(w):
    through(w, "smoke")
    d = doc()
    r = w.runner()
    full = r._options(d, 249, None)
    assert [(o["design"]["F"], o["with_c"]) for o in full] == [(8, True), (8, False), (9, False)]
    elapsed = r._elapsed_h(d)
    assert elapsed == sum(e["wall_s"] for e in d["ledger"]) / 3600 and elapsed < SPEC.budget_h
    tight = dataclasses.replace(SPEC, budget_h=elapsed + (full[0]["total_h"] + full[1]["total_h"]) / 2)
    out = w_rules.budget(elapsed, full, tight, stage="5a")
    assert out["outcome"] == "PASS" and out["plan"]["with_c"] is False
    out = w_rules.budget(elapsed, full, dataclasses.replace(SPEC, budget_h=elapsed + full[1]["total_h"] / 2),
                         stage="5a")
    assert out["outcome"] == w_rules.STOP_BUDGET and out["sentence"].startswith("남은 추정 비용 누적 ")


def test_smoke_spec_only_for_smoke(w):
    through(w, "oc")
    with pytest.raises(SystemExit):
        w.runner().stage_smoke()
    out = w.runner(smoke(SPEC)).stage_smoke()
    assert out["problems"] == [] and out["cost"]["total_h"] > 0 and out["design"]["K"] == 8
    assert all(42_100_000 <= s < 42_200_000 for s in doc()["smoke"]["seeds"]["probe"]["0"])
    assert ("smoke", "L", 1, tuple(doc()["reuse"]["z_V"]["A"])) in w.oracle_calls


def test_stage0_synthetic_validation_reps(w):
    through(w, "stage0")
    assert w.synth_calls == [SPEC.synth_reps] and SPEC.synth_reps >= 1000
    assert doc()["stage0"]["tables"]["oc"]["synth_reps"] == SPEC.synth_reps


def test_rn_sharing_r_state_is_caught(w):
    """Mutation (W.9.6 P2-10): RN continues on R's trained brain instead of its own naive rig — the pilot's machine
    check refuses it (RN's pre ≠ R's pre, RN's weights after the reward phase ≠ R's), and so does the smoke."""
    through(w, "path")
    w.model.share_state = True
    out = w.runner().stage_pilot()
    assert out["outcome"] == w_rules.INVALID
    assert any("RN pre ≠ R pre" in m for m in out["reasons"])
    assert any("RN 보상 뒤 가중치 ≠ R" in m for m in out["reasons"])
    with pytest.raises(SystemExit):
        w.runner().stage_oc()


def test_rn_sharing_r_state_fails_the_smoke(w):
    through(w, "oc")
    w.model.share_state = True
    out = w.runner(smoke(SPEC)).stage_smoke()
    assert "fly 0: RN pre ≠ R pre" in out["problems"] and "fly 0: RN 보상 뒤 가중치 ≠ R" in out["problems"]
    with pytest.raises(SystemExit):
        w.runner().stage_budget()


def test_every_rn_unit_runs_both_phases_from_its_own_pre(w):
    """W.9.6 P2-10: RN is its own job from the naive rig — reward phase with R's training seeds, then no DAN."""
    r = w.runner()
    for where, rows in (("main", MAIN[:2]), ("pilot", PILOT[:2]), ("smoke", PILOT[:1])):
        us = r.units(rows, where, 8, 2, "x")
        by = {(u["pair"], u["fly"], u["brain"]): u for u in us}
        for (pair, fly, brain), u in by.items():
            if brain != "RN":
                continue
            ph, rph = u["phases"], by[(pair, fly, "R")]["phases"]
            assert [p[0] for p in ph] == [SPEC.reward_dan, None] and ph[0] == rph[0]
            assert ph[1][1:] == rph[1][1:] and u["probe_seeds"] == by[(pair, fly, "R")]["probe_seeds"]
