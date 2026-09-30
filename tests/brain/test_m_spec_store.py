"""Spec M.10: one configuration object; guarded writes only under results/m0d/m/ and the M summary."""
import dataclasses
import json

import pytest

from flymon.brain import m_store
from flymon.brain.l_spec import SPEC as L_SPEC
from flymon.brain.m_spec import SPEC, smoke


def test_the_declared_candidates_and_incumbents():
    assert len(SPEC.reward_candidates) == 13 and len(SPEC.punish_candidates) == 8
    assert SPEC.incumbent_reward in SPEC.reward_candidates and SPEC.incumbent_punish in SPEC.punish_candidates
    assert "PAM07" not in SPEC.reward_candidates and "PAM09" not in SPEC.reward_candidates
    assert len(SPEC.reward_candidates) - 1 + len(SPEC.punish_candidates) - 1 == 19


def test_the_numbers_are_m10s():
    assert (SPEC.top_k, SPEC.judge_from_turn, SPEC.judge_n_b, SPEC.judge_n_a) == (2, 4, 21, 18)
    assert SPEC.judge_b_digest.startswith("f55be2df") and SPEC.judge_a_digest.startswith("a978f054")
    assert SPEC.oc_q_b == (0.33, 0.5, 0.6, 0.7) and SPEC.oc_c == (2, 4, 7) and SPEC.oc_naive_a == tuple(range(9))
    assert SPEC.j.h4.testable_min == 2.0 and SPEC.j.h4.z_ddof == 0 and SPEC.j.stage2_select_testable_b == 11
    assert SPEC.dict_readout_c3() == {"A": "MBON13", "P": "MBON05"}


def test_c3_readout_m0d_path_and_attempts_come_from_l():
    assert SPEC.l is L_SPEC and SPEC.readout_c3 == L_SPEC.readout and SPEC.m0d_path == L_SPEC.m0d_path
    assert SPEC.attempts[:3] == L_SPEC.attempts[:3] and [a[0] for a in SPEC.attempts] == ["I", "J", "K", "L", "M"]
    assert SPEC.attempts[3] == ("L", "naive-readout screen SCREEN_IMPRECISE 7/15", "c3d2e25")
    assert SPEC.attempts[4] == ("M", "this declaration", "1441eba")
    other = dataclasses.replace(L_SPEC, readout=(("A", "X"), ("P", "Y")), m0d_path="elsewhere.json")
    s = dataclasses.replace(SPEC, l=other)
    assert s.dict_readout_c3() == {"A": "X", "P": "Y"} and s.m0d_path == "elsewhere.json"


def test_smoke_keeps_the_rules_and_leaves_the_judgement_set_alone():
    s = smoke(SPEC)
    assert s.judge_from_turn == 60 and s.judge_n_b == 2                   # M.10.8: L set turns >= 60 (of 64)
    assert SPEC.l.n_turns == 64 and s.judge_n_b == 2 and s.judge_b_digest == "" and s.judge_a_digest == ""
    assert s.incumbent_reward in s.reward_candidates and s.incumbent_punish in s.punish_candidates
    assert (s.j, s.k, s.l, s.top_k) == (SPEC.j, SPEC.k, SPEC.l, SPEC.top_k)


def test_smoke_candidates_are_a_declared_subset_in_declared_order():
    s = smoke(SPEC)
    for got, declared in ((s.reward_candidates, SPEC.reward_candidates), (s.punish_candidates, SPEC.punish_candidates)):
        assert len(got) == 2 and set(got) <= set(declared)
        assert list(got) == [n for n in declared if n in got]           # a filter of the declared list
    assert set(s.reward_candidates) == {"PAM08", "PAM10"} and set(s.punish_candidates) == {"PPL103", "PPL105"}
    assert (s.n_even_b, s.n_even_a, s.n_odd) == (2, 2, 2) and (SPEC.n_even_b, SPEC.n_even_a, SPEC.n_odd) == (None,) * 3


def test_the_spec_is_frozen_and_hashable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPEC.top_k = 3
    assert hash(SPEC) == hash(dataclasses.replace(SPEC))


@pytest.mark.parametrize("path", ["results/m0d/l/x.json", "results/summary/l_screen.json", "results/m0d/m/../k/x",
                                  "results/m0d/mx/x.json", "/tmp/x.json"])
def test_the_guard_refuses_other_paths(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        m_store.guard(path, [])


def test_summary_blocks_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m_store.write_summary_block(m_store.SUMMARY, "a", {"x": 1}, [])
    m_store.write_summary_block(m_store.SUMMARY, "b", {"y": 2}, [])
    m_store.write_json("results/m0d/m/r.json", {"z": 3}, [])
    assert json.loads((tmp_path / m_store.SUMMARY).read_text()) == {"a": {"x": 1}, "b": {"y": 2}}
    assert json.loads((tmp_path / "results/m0d/m/r.json").read_text()) == {"z": 3}
