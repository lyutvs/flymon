"""AC's numbers (spec appendix AC.2-AC.8) and where AC may write."""
import dataclasses
import json

import pytest

from flymon.ac import store
from flymon.ac.spec import FORBIDDEN, LABEL, SPEC, bench, smoke


def test_arms_seeds_and_thresholds():
    assert dict(SPEC.brain_arms) == {"FLY": 12, "COFF": 6, "TB": 6} and SPEC.n_brain() == 24
    assert dict(SPEC.nobrain_arms) == {"RND": 16, "MAX": 16}
    assert SPEC.min_valid_of() == {"FLY": 9, "COFF": 4, "TB": 4, "RND": 12, "MAX": 12}
    assert SPEC.seeds() == {"gen": 301, "learn": 302, "eval": 303, "boot": 304, "tie": 305}
    assert (SPEC.learn_battles, SPEC.eval_battles, SPEC.sit_points) == (40, 20, (0, 10, 20, 30, 40))
    assert (SPEC.n_pairs, SPEC.boot_draws, SPEC.min_effect) == (20, 10_000, 0.15)
    assert (SPEC.learn_rows(), SPEC.eval_rows()) == (12, 16)
    assert SPEC.lever_edit == "u_apl_mbon05_x0.0+chain_entry"
    assert SPEC.lever_sha == "2d359b8b6947d542348b658db977e1edbe0db5919e7f4b1187598cb10ef1253a"
    assert (SPEC.strength, SPEC.a_type, SPEC.p_type, SPEC.codebook_config) == (1.0, "MBON13", "MBON05", "k2-norm")
    assert (SPEC.budget_total_h, SPEC.budget_stage1_h, SPEC.session_cap_h, SPEC.margin) == (60.0, 48.0, 24.0, 1.3)
    assert SPEC.rescope_s_per_batch_battle == 431.1


def test_label_is_ac5_verbatim_and_clean():
    assert LABEL == ("M2 정식 판정 없음 — 근거는 AA 효과 크기 추정(보상 연합 d′ 6.02 [4.83, 7.49], 처벌 −4.52 "
                     "[−5.89, −3.23], 넓힌 풀 16쌍, 커버리지 0.855 명목 미달)뿐. 탐색 단계(부록 AC).")
    assert FORBIDDEN == ("PASS", "FAIL", "학습된다", "학습 안 됨")
    assert not any(w in LABEL for w in FORBIDDEN)


def test_smoke_and_bench_change_scale_and_seeds_only():
    sm, be = smoke(), bench()
    for s in (sm, be):
        for f in ("smoke_max_tie", "smoke_max_a0", "smoke_max_p0", "min_effect", "lever_edit", "strength", "boot_seed",
                  "tie_seed", "kc_ratio", "margin"):
            assert getattr(s, f) == getattr(SPEC, f), f
    assert dict(sm.brain_arms) == {"FLY": 2, "COFF": 2, "TB": 2} and (sm.learn_battles, sm.eval_battles) == (2, 2)
    assert sm.sit_points == (0, 1, 2) and sm.n_pairs == 4 and sm.gen_seed == 305 and sm.tags == ("SL", "SE")
    assert sm.learn_seed != sm.eval_seed and sm.mode == "smoke" and sm.out_root == "results/m4-smoke"
    assert be.brain_arms == SPEC.brain_arms and (be.learn_battles, be.eval_battles) == (2, 0) and be.sit_points == (0,)
    assert be.n_pairs == 20 and be.gen_seed == 305 and be.learn_seed == sm.learn_seed and be.tags == ("BL", "BE")
    assert be.mode == "bench" and be.out_root == "results/m4-bench"


def test_guard_allows_only_ac_trees(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for ok in ("results/m4/stage1/BRAIN/result.json", "results/m4-smoke/x.json", "results/m4-bench/b.json",
               "results/summary/ac_inputs.json"):
        assert store.guard(ok)
    for bad in ("results/rescope/x.json", "results/summary/m0d.json", "results/summary/ac_x.txt", "x.json"):
        with pytest.raises(SystemExit, match="AC outputs"):
            store.guard(bad)


def test_write_json_and_digest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = store.write_json("results/m4/a.json", {"b": 1, "a": [1, 2]})
    assert json.loads(p.read_text()) == {"a": [1, 2], "b": 1}
    assert store.digest({"a": 1, "b": 2}) == store.digest({"b": 2, "a": 1}) != store.digest({"a": 2, "b": 2})
    assert store.sha256_file(p) == store.sha256_file(p) and len(store.sha256_file(p)) == 64


def test_spec_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPEC.n_pairs = 3
