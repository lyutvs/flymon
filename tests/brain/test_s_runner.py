# tests/brain/test_s_runner.py
"""The S stage chain up to gate ② (S.2, S.3 ①-④, S.9.3, S.9.5, S.9.7; Readings 4-6, 9): reuse -> set -> smoke -> oc
-> gate2_oc -> gate2; each stage refuses (exit 2, nothing written) when an earlier block is missing, a later one
exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted or a hashed file is
dirty; a STOP_REUSE, STOP_SET_SHORT, smoke problem or gate ② STOP / INVALID blocks every later stage; R's reuse
condition broken after `reuse` refuses every later stage with exit 7; the judgement set is never measured before the
judgement stages."""
import json
from pathlib import Path

import pytest

from flymon.brain import s_rules
from flymon.brain import s_runner as SR
from flymon.brain.s_spec import SPEC
from tests.brain.s_world import Scripted, World, doc, through_gate2


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate2_never_touches_the_judgement_set(w):
    m = Scripted()
    through_gate2(w, m)
    d = doc()
    assert set(d) == set(SR.ORDER[:SR.ORDER.index("jm:L")])
    assert d["reuse"]["records"]["k_even"] == 16 and d["reuse"]["records"]["repro_csc_sha256_none"] == "sha-C"
    assert d["reuse"]["shared_key"] == SPEC.r_shared_key and d["reuse"]["code_key"] == SPEC.r_shared_key
    assert all(d[b]["pipeline_key"] == "p" * 64 for b in d)
    assert d["set"]["set"]["last_turn"] == 177 and w.set_calls == 1
    assert d["smoke"]["p"]["L"]["edit_edges"] == [2] and d["smoke"]["p"]["C"]["edit_edges"] == [0]
    assert d["smoke"]["p"]["L"]["label"] == d["smoke"]["p"]["C"]["label"] == "LEARNS_CONFIRMATORY"
    assert Path(SPEC.smoke_detail).exists()
    assert d["gate2_oc"]["rows"][1]["rho"] == 0.5 and d["gate2_oc"]["rows"][1]["p_stop_weakened"] == pytest.approx(0.75)
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"]) == ("LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY")
    assert g2["seeds"] == list(SPEC.p.seeds) and g2["edit_edges_L"] == [2] and g2["edit_edges_C"] == [0]
    assert 0.5 <= g2["ratio"]["r1"]["ratio"] < 1 and set(g2["ell_L_minus_r_gate2"]) == {"r1", "r2"}
    assert g2["ell_L_minus_r_gate2"]["r1"] == pytest.approx(g2["ratio"]["r1"]["ell_L"] - 1.087)
    assert w.judgement_calls == 0
    assert not any(c[1] == "judge" for c in m.calls if c[0] == "oracle")
    assert all(c[3] == 2 for c in m.calls if c[0] == "oracle" and c[1] == "smoke")
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]


def test_reuse_stop_when_the_shared_key_differs(w):
    out = w.runner(Scripted(), code={"key": "f" * 64}).stage_reuse()
    assert out["outcome"] == s_rules.STOP_REUSE and "공유 측정 키" in out["sentence"]
    assert doc()["reuse"]["outcome"] == s_rules.STOP_REUSE
    with pytest.raises(SystemExit) as e:
        w.runner(Scripted()).stage_set()
    assert e.value.code == 2 and w.set_calls == 0


@pytest.mark.parametrize("mutate", ["gate1", "dirty", "key"])
def test_reuse_broken_after_reuse_refuses_with_7(w, mutate):
    r = w.runner(Scripted())
    r.stage_reuse()
    if mutate == "gate1":
        w.r["gate1"]["outcome"] = "STOP_STRENGTH_LEVER"
    elif mutate == "dirty":
        w.r_git["dirty"] = True
    else:
        r = w.runner(Scripted(), code={"key": "f" * 64})
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == SR.EXIT_REUSE and "set" not in doc() and w.set_calls == 0


def test_set_stop_set_short_blocks_later_stages(w):
    w.js = dict(w.js, status="STOP_SET_SHORT", n_b=19, last_turn=209)
    r = w.runner(Scripted())
    r.stage_reuse()
    out = r.stage_set()
    assert out["outcome"] == s_rules.STOP_SET_SHORT and "(19쌍)" in out["sentence"]
    with pytest.raises(SystemExit) as e:
        r.stage_smoke()
    assert e.value.code == 2


def test_set_mismatch_refuses_without_a_block(w):
    w.js = dict(w.js, digest_keys="0" * 64)
    r = w.runner(Scripted())
    r.stage_reuse()
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == 2 and "set" not in doc()


def _to_gate2(w, m):
    r = w.runner(m)
    r.stage_reuse()
    r.stage_set()
    r.stage_smoke()
    r.stage_oc()
    r.stage_gate2_oc()
    return r


@pytest.mark.parametrize("drop,outcome", [
    ({SPEC.lever_edit: 6}, s_rules.STOP_PUNISH_WEAKENED),          # ℓ_L ≈ 0.78 ≥ c1, ratio ≈ 7/17 < 0.5
    ({SPEC.lever_edit: 3}, s_rules.STOP_PUNISH_BROKEN),            # ℓ_L ≈ 0.44 < c1
    ({SPEC.no_edit: 3}, s_rules.STOP_P_REFERENCE),                 # ℓ_C ≈ 0.44 < c1, L fine
])
def test_gate2_stops_block_the_judgement(w, drop, outcome):
    r = _to_gate2(w, Scripted(drop=drop))
    out = r.stage_gate2()
    assert out["outcome"] == outcome and out["sentence"]
    with pytest.raises(SystemExit) as e:
        r._require("jm:L")                                         # every judgement stage starts here
    assert e.value.code == 2 and w.judgement_calls == 0


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    m = Scripted()
    r = _to_gate2(w, m)
    m.arm_edges[SPEC.lever_edit] = 1                               # the lever changed 1 edge in gate ② only
    assert r.stage_gate2()["outcome"] == s_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                                  # same pipeline key: fix first
    m.arm_edges[SPEC.lever_edit] = SPEC.lever_edges
    fixed = w.runner(m, pipeline={"key": "q" * 64})
    assert fixed.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["rerun_of"] == d["gate2_invalid"]["written_at"]
    with pytest.raises(SystemExit):
        fixed.stage_gate2(rerun=True)                              # once


def test_gate2_invalid_when_the_p_judgement_is_invalid(w):
    m = Scripted()
    r = _to_gate2(w, m)
    orig = m.arms
    m.arms = lambda items, *a: [dict(x, csc_sha256="sha-other") if x["seed"] == SPEC.p.seeds[0] else x
                                for x in orig(items, *a)]
    out = r.stage_gate2()
    assert out["outcome"] == s_rules.INVALID and out["ratio"] is None


def test_smoke_problem_blocks_later_stages(w):
    m = Scripted(override={("smoke", "L"): dict(edges=1)})
    r = w.runner(m)
    r.stage_reuse()
    r.stage_set()
    assert r.stage_smoke()["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    r = w.runner(Scripted())
    with pytest.raises(SystemExit):
        r.stage_set()                                              # reuse missing
    r.stage_reuse()
    with pytest.raises(SystemExit):
        r.stage_reuse()                                            # own block
    monkeypatch.setattr(SR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_set()                                              # uncommitted summary
    monkeypatch.setattr(SR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(SR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/s_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_set()                                              # dirty hashed file
    assert set(doc()) == {"reuse"}


def test_gate2_oc_refuses_without_ps_block(w):
    r = w.runner(Scripted())
    r.stage_reuse()
    r.stage_set()
    r.stage_smoke()
    r.stage_oc()
    w.ctx["p_ref"] = lambda wanted: {}
    with pytest.raises(SystemExit):
        r.stage_gate2_oc()
    assert "gate2_oc" not in doc()


def _smoke(w, m):
    r = w.runner(m)
    r.stage_reuse()
    r.stage_set()
    return r.stage_smoke()["problems"]


def _p_invalid_in_smoke(m):
    from flymon.brain.s_spec import smoke
    orig, s0 = m.arms, smoke(SPEC).p.seeds[0]
    m.arms = lambda items, *a: [dict(x, csc_sha256="sha-other") if x["seed"] == s0 else x for x in orig(items, *a)]
    return m


@pytest.mark.parametrize("make,expect", [
    (lambda: Scripted(arm_edges={SPEC.no_edit: 1}), "P arms C: edges [1]"),
    (lambda: _p_invalid_in_smoke(Scripted()), "P arms L: INVALID"),
    (lambda: Scripted(override={("smoke", "L"): dict(sha="sha-C")}), "L and C ran on the same CSC weights"),
    (lambda: Scripted(override={("smoke", "C"): dict(sha="sha-X")}), "C / E0 CSC sha-X / sha-C is not R's repro"),
    (lambda: Scripted(override={("smoke", "E0"): dict(sha="sha-X")}), "C / E0 CSC sha-C / sha-X is not R's repro"),
])
def test_smoke_problem_branches(w, make, expect):
    problems = _smoke(w, make())
    assert any(p.startswith(expect) for p in problems), problems


@pytest.mark.parametrize("pipeline", ["p" * 64, "q" * 64])
def test_gate2_rerun_refuses_with_7_when_the_shared_key_changed(w, pipeline):
    """S.9.5: a 'fix' in a shared measurement file changes the shared key (the pipeline key may stay) — exit 7."""
    m = Scripted()
    r = _to_gate2(w, m)
    m.arm_edges[SPEC.lever_edit] = 1
    assert r.stage_gate2()["outcome"] == s_rules.INVALID
    before = doc()
    m.arm_edges[SPEC.lever_edit] = SPEC.lever_edges
    with pytest.raises(SystemExit) as e:
        w.runner(m, code={"key": "f" * 64}, pipeline={"key": pipeline}).stage_gate2(rerun=True)
    assert e.value.code == SR.EXIT_REUSE and doc() == before and "gate2_invalid" not in doc()


def test_stop_reuse_writes_only_its_own_block(w):
    out = w.runner(Scripted(), code={"key": "f" * 64}).stage_reuse()
    assert out["outcome"] == s_rules.STOP_REUSE
    assert set(doc()) == {"reuse"} and not Path(SPEC.smoke_detail).exists()
    assert not Path(SPEC.cache_dir).exists() and w.set_calls == 0 and w.judgement_calls == 0
