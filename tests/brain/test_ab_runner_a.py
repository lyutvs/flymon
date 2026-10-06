"""ab_runner (A): closure / seal / env, AB seed formulas (c = the AB candidate number), the AA differential test (cache
hits only, bit for bit, mutants caught), stage0, reuse (every AB.2 0b item incl. the 0f source), generate (0b's list
only, both tallies, STOP_SET_MISMATCH), seal_code (seal + five stream vectors), chain / env / seal refusals, the
learn-archive STOP_MACHINE rule (AB.7 8), restart_preseal (plan Reading 15) and the candidate states."""
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import aa_runner, ab_estimate, ab_rules, ab_store
from flymon.brain import ab_runner as R
from flymon.brain.ab_spec import SPEC
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.y_spec import SPEC as Y
from tests.brain.ab_world import AA_ROWS, MAIN, S1_C, World, doc

ROOT = Path(__file__).resolve().parents[2]


def _code(fn):
    with pytest.raises(SystemExit) as e:
        fn()
    return e.value.code


def _note(tmp_path, **kw):
    p = tmp_path / "note.json"
    p.write_text(json.dumps(dict(dict(symptom="Stage II code added (staged build)", clause="AB.7 0 (order-0 "
                                      "completeness)", cause="plan reading 33"), **kw)))
    return str(p)


# ================================================================ closure, seal, seeds
def test_closure_follows_levels_1_and_2():
    c = R.closure(["flymon/brain/ab_pairs.py"])
    for f in ("flymon/agent/e_pairs.py", "flymon/battle/pool.py", "flymon/brain/v_pairs.py",
              "flymon/brain/ab_pairs.py", "flymon/brain/ab_spec.py"):
        assert f in c, f
    assert all((ROOT / f).exists() for f in c)
    sf = R.seal_files(SPEC)
    assert set(R.closure(SPEC.seal_files)) <= set(sf) and set(SPEC.seal_files) <= set(sf)
    assert "flymon/brain/ab_runner.py" not in sf and "flymon/brain/aa_estimate.py" in sf
    imp = R.imported_files()
    assert not set(imp) & set(R.ab_files()) and "flymon/brain/aa_runner.py" in imp


def test_env_now_fields():
    e = R.env_now()
    assert set(e) == {"uv_lock_sha256", "python", "numpy", "scipy", "poke_env", "platform", "ab_files", "imported"}
    assert "flymon/brain/ab_runner.py" in e["ab_files"] and "scripts/run_ab.py" in e["ab_files"]
    assert "flymon/brain/w_measure.py" in e["imported"]


def test_ab_w_spec_seed_formulas():
    ws = R.ab_w_spec()
    assert ws.probe_seeds(5, 2, 8)[3] == 88_100_000 + 5 * 4_000 + 2 * 100 + 3
    assert ws.train_base(5, 1) == 89_500_000 + 5 * 40_000 + 20
    assert ws.oracle_seeds()["select"][0] == 88_000_100
    assert ws.oracle_seeds()["act"] == list(range(88_000_000, 88_000_008))
    assert ws.oracle_seeds()["report"][-1] == 88_000_207
    assert ws.smoke_probe_seeds(1, 8)[0] == 88_040_100 and ws.smoke_train_base(1) == 88_050_020
    ss = R.smoke_seed_set()
    assert min(ss) == 88_040_000 and 88_043_199 in ss and 88_060_207 in ss and 88_060_008 not in ss
    assert all(88_040_000 <= x < 88_070_000 for x in ss)


def test_units_main_seed_formula_uses_c():
    r = dict(MAIN[0], c=7)
    us = aa_runner.units([r], "main", 8, 2, V.lever_edit, ws=R.ab_w_spec())
    for u in us:
        assert u["idx"] == 7
        assert u["probe_seeds"] == [88_100_000 + 7 * 4_000 + u["fly"] * 100 + k for k in range(8)]
        b = 89_500_000 + 7 * 40_000
        assert u["phases"][0][2] == b and u["phases"][1][2] == b + 20
    pos = aa_runner.units([dict(r, c=0)], "main", 8, 2, V.lever_edit, ws=R.ab_w_spec())   # a position, not c
    assert [u["probe_seeds"] for u in pos] != [u["probe_seeds"] for u in us]


# ================================================================ the differential test
def test_differential_bit_equal_on_world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = R.differential(w.ctx, w.s)
    assert out["ok"], out["reasons"][:5]
    assert out["n_units"] == out["n_hits"] == 2 * 8 * 3
    first = [row_key(r) for r in AA_ROWS if r["c"] in S1_C][:2]
    assert sorted(out["pairs"]) == sorted(first)
    assert all(all(v.values()) for v in out["pairs"].values())
    assert out["ts"] == {"reward_assoc": True, "punish_assoc": True}
    assert w.pool.jobs == w.pool_jobs_after_aa                          # nothing measured


def _mut_seed(orig):
    return lambda: dataclasses.replace(orig(), probe_seed0=orig().probe_seed0 + 1)


def _mut_brain(orig):
    def f(rows, where, K, F, edit, **kw):
        out = orig(rows, where, K, F, edit, **kw)
        ph = {(u["pair"], u["fly"], u["brain"]): u["phases"] for u in out}
        sw = dict(R="RN", RN="R", N="N")
        return [dict(u, phases=ph[(u["pair"], u["fly"], sw[u["brain"]])]) for u in out]
    return f


def _mut_stage(orig):
    def f(rows, flies):
        d = orig(rows, flies)
        return dict(d, R1=d["R2"], R2=d["R1"])
    return f


@pytest.mark.parametrize("name", ["probe seed root", "brain order", "stage order", "z", "ab seeds"])
def test_differential_catches_mutants(tmp_path, monkeypatch, name):
    w = World(tmp_path, monkeypatch)
    if name == "probe seed root":
        monkeypatch.setattr(R, "aa_seeds", _mut_seed(R.aa_seeds))
    elif name == "ab seeds":                                            # AB's judged seeds instead of AA's
        monkeypatch.setattr(R, "aa_seeds", lambda: R.ab_w_spec(w.s))
    elif name == "brain order":
        monkeypatch.setattr(R, "units", _mut_brain(R.units))
    elif name == "stage order":
        monkeypatch.setattr(R, "pair_counts", _mut_stage(R.pair_counts))
    else:
        w.wz = dict(w.wz, A=(w.wz["A"][0] + 1e-9, w.wz["A"][1]))
    out = R.differential(w.ctx, w.s)
    assert out["ok"] is False and out["reasons"], name
    assert w.pool.jobs == w.pool_jobs_after_aa


def test_differential_cache_miss_is_failure(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    first = row_key([r for r in AA_ROWS if r["c"] in S1_C][0])
    man = json.loads(Path(SPEC.aa_learn_detail).read_text())["manifest"]
    Path(next(m for m in man if m["key"].startswith(first + "|"))["cache_file"]).unlink()
    out = R.differential(w.ctx, w.s)
    assert out["ok"] is False and any("캐시 적중 실패" in r for r in out["reasons"])
    assert w.pool.jobs == w.pool_jobs_after_aa


def test_fut_source_reproduces_aa_records(tmp_path, monkeypatch):
    """The world's 0f source holds fut_check (Task 8a's futility stage runs on it)."""
    w = World(tmp_path, monkeypatch)
    src = w.ctx["fut_source"]()
    assert src["n_units"] == w.s.fut_src_units == 144 and src["n_pairs"] == 6 and not src["missing"] + src["bad_sha"]
    arr = ab_estimate.futility_inputs(src["by_pair"], src["keys"], src["z"], w.s)
    assert ab_estimate.fut_check(arr, src["pairs"], src["s1"], w.s) == []
    m = ab_estimate.fut_model(arr, src["s1"], w.s)                      # non-degenerate source: C finite, unit diagonal
    C = np.asarray(m["C"])
    assert C.shape == (8, 8) and np.isfinite(C).all() and np.allclose(np.diag(C), 1.0)
    assert np.isfinite(np.asarray(m["chol"])).all()


# ================================================================ stage0
def test_stage0_pass_block(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = w.runner().run("stage0")
    assert out["outcome"] == "PASS", out.get("reasons")
    blk = doc()["stage0"]
    assert blk["tests"]["last_line"] == "exit 0" and re.search(r"\d+ passed", blk["tests"]["passed_line"])
    b = blk["bench"]
    assert set(b["per_rep_s"]) == {"17", "23"} and b["max_s"] > 0 and b["B"] == w.s.boot_b
    assert blk["seal"]["key"] == R.seal_now(w.s)["key"] and blk["differential"]["ok"]
    assert blk["env"]["ab_files"] and blk["env"]["imported"] and blk["order"] == "0a" and "archive" in blk
    assert set(R.ab_files()) <= set(blk["decision_files"]) and "flymon/brain/ab_estimate.py" in blk["decision_files"]
    assert blk["numbers"] == ab_store.to_json(w.s)
    for k in ("git", "written_at", "w_measure_key", "u_measure_key", "detail_sha256"):
        assert k in blk
    assert doc()["candidates"] == [] and doc()["budget"]["ledger"][-1]["stage"] == "stage0"
    assert Path(w.s.archive_root, "stage0").exists()


@pytest.mark.parametrize("text,why", [
    ("exit 0\n", "통과 개수 줄"),
    ("..........\nexit 0\n", "통과 개수 줄"),
    ("1 failed, 1233 passed in 600.00s\nexit 0\n", "실패 · 오류 요약"),
    ("1233 passed, 2 errors in 600.00s\nexit 0\n", "실패 · 오류 요약"),
    ("1234 passed in 600.00s\nexit 1\n", "마지막 줄"),
    ("\x1b[31m\x1b[1m1 failed\x1b[0m, \x1b[32m1233 passed\x1b[0m\x1b[31m in 600.00s\x1b[0m\nexit 0\n", "실패 · 오류 요약"),
    ("\x1b[32m1233 passed\x1b[0m, \x1b[31m\x1b[1m2 errors\x1b[0m in 600.00s\nexit 0\n", "실패 · 오류 요약"),
])
def test_stage0_requires_passed_line_and_no_failures(tmp_path, monkeypatch, text, why):
    w = World(tmp_path, monkeypatch)
    w.write_log(text)
    out = w.runner().run("stage0")
    assert out["outcome"] == "INVALID" and any(why in r for r in out["reasons"]), out["reasons"]
    assert "archive" not in out


def test_stage0_coloured_passed_line_is_found(tmp_path, monkeypatch):
    """pytest colours the summary on a terminal: "\x1b[32m1234 passed\x1b[0m" is still the "N passed" line."""
    w = World(tmp_path, monkeypatch)
    w.write_log("\x1b[32m\x1b[1m1234\x1b[0m\x1b[32m passed\x1b[0m\x1b[32m in 600.00s\x1b[0m\nexit 0\n")
    out = w.runner().run("stage0")
    assert out["outcome"] == "PASS", out.get("reasons")
    assert doc()["stage0"]["tests"]["passed_line"] == "1234 passed in 600.00s"


def test_differential_needs_diff_pairs_pairs(tmp_path, monkeypatch):
    """AA S1 shorter than diff_pairs is a failure (never a shorter differential test)."""
    w = World(tmp_path, monkeypatch)
    out = R.differential(w.ctx, dataclasses.replace(w.s, diff_pairs=len(S1_C) + 1))
    assert out["ok"] is False and any("차등 시험 쌍 수" in r for r in out["reasons"]), out["reasons"]
    assert w.pool.jobs == w.pool_jobs_after_aa


def test_stage0_invalid_on_differential(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "pair_counts", _mut_stage(R.pair_counts))
    out = w.runner().run("stage0")
    assert out["outcome"] == "INVALID" and any(r.startswith("차등 시험:") for r in out["reasons"])
    assert out["differential"]["ok"] is False and "archive" not in out
    assert _code(lambda: w.runner().archive("stage0")) == 2


# ================================================================ reuse (AB.2 0b)
def _break(w, case):
    s = w.s
    if case == "v_commit":
        w.facts_ok = lambda c: c != s.v_commits[2][1]
    elif case == "aa_commit":
        w.facts_ok = lambda c: c != s.aa_blocks[-1][1]
    elif case == "kc_sha":
        Path(V.kc_input_detail).write_text(Path(V.kc_input_detail).read_text() + " ")
    elif case == "w_digest":
        w.w_digest = dict(w.w_digest, digest_keys="0" * 64)
    elif case == "aa_outcome":
        d = json.loads(Path(SPEC.aa_summary).read_text())
        d["estimate"]["outcome"] = "STOP_X"
        Path(SPEC.aa_summary).write_text(json.dumps(d))
    elif case == "lv_count":                                            # one AA trained key fewer
        d = json.loads(Path(SPEC.aa_summary).read_text())
        next(x for x in d["candidates"] if x["state"] == "trained")["state"] = "screened"
        Path(SPEC.aa_summary).write_text(json.dumps(d))
    elif case == "v_check":
        w.base_error = "V's set does not reproduce V's block set: row 3"
    elif case == "keys":
        w.ctx["keys"] = lambda: dict(w_measure_key="0" * 64, u_measure_key=Y.u_measure_key)
    elif case == "file_sha":
        real = w.ctx["decl_sha"]

        def decl(files, commit):
            d = real(files, commit)
            d["flymon/brain/w_measure.py"] = "x" * 64
            return d
        w.ctx["decl_sha"] = decl
    else:
        man = json.loads(Path(SPEC.aa_learn_detail).read_text())
        s1 = {row_key(r) for r in AA_ROWS if r["c"] in S1_C}
        i = next(j for j, m in enumerate(man["manifest"]) if m["key"].rsplit("|", 2)[0] in s1)
        if case == "fut_unit_removed":
            man["manifest"].pop(i)
            Path(SPEC.aa_learn_detail).write_text(json.dumps(man))
        elif case == "fut_file_deleted":
            Path(man["manifest"][i]["cache_file"]).unlink()
        elif case == "fut_bytes_changed":
            p = Path(man["manifest"][i]["cache_file"])
            p.write_text(p.read_text() + " ")
        else:
            raise AssertionError(case)


REUSE_BREAKS = ["v_commit", "aa_commit", "kc_sha", "w_digest", "aa_outcome", "lv_count", "v_check", "keys",
                "file_sha", "fut_unit_removed", "fut_file_deleted", "fut_bytes_changed"]
REUSE_WHY = dict(v_commit="V 블록", aa_commit="AA 블록 records 커밋", kc_sha="kc_input.json", w_digest="W 주 세트",
                 aa_outcome="AA 블록 estimate의 결과", lv_count="AA trained", v_check="V 세트:", keys="w_measure_key",
                 file_sha="w_measure.py", fut_unit_removed="가망 관문 원천 단위", fut_file_deleted="캐시 파일 없음",
                 fut_bytes_changed="sha256 불일치")


@pytest.mark.parametrize("case", [None] + REUSE_BREAKS)
def test_reuse_pass_and_each_break(tmp_path, monkeypatch, case):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    if case:
        _break(w, case)
    out = w.runner().run("reuse")
    blk = doc()["reuse"]
    if case is None:
        assert out["outcome"] == "PASS", out.get("reasons")
        lv = blk["lv_odour"]
        assert lv["n"] == w.s.lv_odour_n == len(lv["odours"]) and len(lv["sha256"]) == 64
        assert lv["odours"] == sorted(lv["odours"]) and set(lv["per_source"]) == {"aa31", "learned", "v_set"}
        assert blk["fut_source"] == dict(n_units=144, n_pairs=6, missing=[], bad_sha=[])
        return
    assert out["outcome"] == ab_rules.STOP_REUSE and out["stop_stage"] == "0b", out.get("reasons")
    assert any(REUSE_WHY[case] in r for r in out["reasons"]), out["reasons"]
    assert out["sentence"].startswith("AB 재사용 조건(AB.2)이 깨졌다(") and out["sentence"].endswith(ab_rules.CLOSURE)
    assert "`untouched` 0 · `oracled` 0 · `screened` 0 · `training` 0 · `trained` 0" in out["sentence"]
    assert "lv_odour" not in blk and blk["fut_source"]["n_pairs"] == 6
    fs = blk["fut_source"]
    if case == "fut_file_deleted":
        assert len(fs["missing"]) == 1 and not fs["bad_sha"]
    elif case == "fut_bytes_changed":
        assert len(fs["bad_sha"]) == 1 and not fs["missing"]
    elif case == "fut_unit_removed":
        assert fs["n_units"] == 143
    else:
        assert fs == dict(n_units=144, n_pairs=6, missing=[], bad_sha=[])
    assert _code(lambda: w.runner().run("generate")) == 2                   # AB stops there


# ================================================================ generate (0c)
def test_generate_reads_reuse_list_and_records_both_tallies(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    lv = doc()["reuse"]["lv_odour"]["odours"]
    w.ctx["lv"] = lambda: pytest.fail("0c must not recompute lv_odour")
    out = w.runner().run("generate")
    assert out["outcome"] == "PASS", out.get("reasons")
    assert w.gen_calls == [lv]
    blk = doc()["generate"]
    assert blk["skipped_inorder"] == w.gen["skipped_inorder"] and blk["skipped_declared"] == w.gen["skipped_declared"]
    assert blk["skipped_inorder"]["lv_odour"] == 7 and "lv_odour" not in blk["skipped_declared"]
    assert "rows" not in blk and blk["keys"] == w.gen["keys"] and blk["selftest"]["ok"] is True
    det = json.loads(Path(blk["detail_path"]).read_text())
    assert det["digest_keys"] == w.gen["digest_keys"] and ab_store.to_json(det["selftest"]) == blk["selftest"]


@pytest.mark.parametrize("field,value", [("decl_b_x_groups", 99), ("decl_pre_kc", (("a", 30), ("b", 11)))])
def test_generate_mismatch_is_stop_set_mismatch(tmp_path, monkeypatch, field, value):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    w.s = dataclasses.replace(w.s, **{field: value})
    out = w.runner().run("generate")
    assert out["outcome"] == ab_rules.STOP_SET_MISMATCH and out["stop_stage"] == "0c", out.get("reasons")
    assert out["sentence"].startswith("AB 생성원(AB.3)이 선언값") and out["sentence"].endswith(ab_rules.CLOSURE)
    assert _code(lambda: w.runner().run("seal_code")) == 2


def test_generate_selftest_failure_is_a_mismatch(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    w.ctx["selftest"] = lambda: dict(ok=False, digest_keys="0" * 64)
    out = w.runner().run("generate")
    assert out["outcome"] == ab_rules.STOP_SET_MISMATCH and any("자가 시험" in r for r in out["reasons"])


# ================================================================ seal_code (0d)
def test_seal_code_records_seal_and_stream_vectors(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = w.chain("seal_code")
    blk = doc()["seal_code"]
    assert blk["seal"]["key"] == R.seal_now(w.s)["key"] == out["seal"]["key"]
    assert "flymon/brain/ab_estimate.py" in blk["seal"]["files"] and "flymon/brain/aa_estimate.py" in blk["seal"]["files"]
    assert blk["seal"]["numbers"]["fut_threshold"] == w.s.fut_threshold
    vec = blk["stream_vectors"]
    assert len(vec) == 5 and [tuple(t) for t, _v in vec] == [tuple(t) for t, _v in SPEC.stream_vectors]
    assert [tuple(v) for _t, v in vec] == [tuple(v) for _t, v in SPEC.stream_vectors]
    assert vec[-1][0] == ["synth", "fut", "g6", "icc"]


def test_seal_code_refuses_other_stream_vectors(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("generate")
    sv = list(w.s.stream_vectors)
    sv[0] = (sv[0][0], (1, 2, 3))
    w.s = dataclasses.replace(w.s, stream_vectors=tuple(sv))
    assert _code(lambda: w.runner().run("seal_code")) == 2
    assert "seal_code" not in doc()


def test_seal_check_exit7_on_change(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    r = w.runner()
    now = R.seal_now(w.s)
    r._seal_check(dict(seal_code=dict(seal=now)))
    assert _code(lambda: r._seal_check(dict(seal_code=dict(seal=dict(now, key="0" * 64))))) == R.EXIT_SEAL == 7
    r._seal_check(dict(seal_code=dict(seal=dict(now, key="0" * 64)), reseal_1=dict(seal=dict(key="1" * 64)),
                       reseal_2=dict(seal=now)))
    assert _code(lambda: r._seal_check(dict(seal_code=dict(seal=now), reseal_2=dict(seal=dict(key="2" * 64))))) == 7
    # a sealed stage (after seal_code) checks the seal before anything else that could start
    w.chain("seal_code")
    w.s = dataclasses.replace(w.s, fut_reps=w.s.fut_reps + 1)
    assert _code(lambda: w.runner()._require("cal_gate")) == 7


def test_seal_key_changes_with_a_futility_number(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    k = R.seal_now(w.s)["key"]
    assert R.seal_now(dataclasses.replace(w.s, fut_threshold=0.6))["key"] != k
    assert R.seal_now(dataclasses.replace(w.s, fut_grid_k=(8, 16)))["key"] != k
    assert R.seal_now(dataclasses.replace(w.s, cli_print_chars=10))["key"] == k     # not a sealed number


# ================================================================ chain, env, archive
def test_chain_refusals(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    assert _code(lambda: w.runner().run("reuse")) == 2                       # order
    w.chain("stage0")
    assert _code(lambda: w.runner().run("stage0")) == 2                      # rewrite
    assert w.runner().run("reuse")["outcome"] == "PASS"
    assert _code(lambda: w.runner().run("reuse")) == 2
    d = doc()
    d["seal_code"] = dict(outcome="PASS")                                     # a later block
    Path(w.s.summary).write_text(json.dumps(d))
    assert _code(lambda: w.runner().run("generate")) == 2
    d.pop("seal_code")
    Path(w.s.summary).write_text(json.dumps(d))
    monkeypatch.setattr(R, "summary_git", lambda p: dict(tracked=True, dirty=True))
    assert _code(lambda: w.runner().run("generate")) == 2                    # dirty summary
    monkeypatch.setattr(R, "summary_git", lambda p: dict(tracked=True, dirty=False))
    monkeypatch.setattr(R, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/ab_spec.py"]))
    assert _code(lambda: w.runner().run("generate")) == 2                    # dirty hashed file
    monkeypatch.setattr(R, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
    assert w.runner().run("generate")["outcome"] == "PASS"
    assert _code(lambda: w.runner().run("judge")) == 2                       # not an AB stage


def test_later_stage_needs_matching_keys(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    w.ctx["keys"] = lambda: dict(w_measure_key="0" * 64, u_measure_key=Y.u_measure_key)
    assert _code(lambda: w.runner()._require("generate")) == 2


def test_env_mismatch_exit_6_and_resume(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    real = R.env_now
    monkeypatch.setattr(R, "env_now", lambda: dict(real(), numpy="0.0.0"))
    assert _code(lambda: w.runner().run("reuse")) == R.EXIT_ENV == 6
    led = doc()["budget"]["ledger"][-1]
    assert led["stage"] == "reuse" and any("numpy" in x for x in led["env_mismatch"]) and "reuse" not in doc()
    f = "flymon/brain/ab_runner.py"                                           # an ab_* fix after a committed block
    monkeypatch.setattr(R, "env_now", lambda: (lambda d: dict(d, ab_files=dict(d["ab_files"], **{f: "f" * 64})))(real()))
    assert _code(lambda: w.runner().run("reuse")) == 6
    monkeypatch.setattr(R, "env_now", lambda: (lambda d: dict(d, ab_files=dict(d["ab_files"], **{
        "flymon/brain/ab_new.py": "n" * 64})))(real()))                       # a file added later is not a mismatch
    assert w.runner().run("reuse")["outcome"] == "PASS"


def test_env_resume_compares_first_start(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    ab_store.write_json(str(Path(w.s.progress_dir) / "reuse.env.json"), dict(R.env_now(), python="2.7"), [])
    assert _code(lambda: w.runner().run("reuse")) == 6


def test_archive_before_block_and_idempotent(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = w.runner().run("stage0")
    assert out["archive"] and all(Path(a["dst"]).exists() for a in out["archive"])
    assert w.runner().archive("stage0") == out["archive"]
    assert _code(lambda: w.runner().archive("reuse")) == 2


def test_learn_archive_sha_mismatch_is_stop_machine(tmp_path, monkeypatch):
    """AB.7 8: an archive copy is never overwritten — same sha skips, another sha is STOP_MACHINE〈8〉 (not exit 2)."""
    w = World(tmp_path, monkeypatch)
    raw = Path(w.s.cache_dir, "w_learn", "u1.json")
    ab_store.write_json(str(raw), dict(key="k", result={}), [])
    ab_store.write_json(w.runner()._detail("learn"), dict(manifest=[dict(cache_file=str(raw))]), [])
    files = w.runner()._files("learn")
    ab_store.archive(files, "learn", w.s)                                     # an earlier copy, same bytes
    blk = w.runner()._write("learn", dict(outcome="PASS", reasons=[], n_units=1))
    assert blk["outcome"] == "PASS" and len(blk["archive"]) == 2
    d = doc()
    d.pop("learn")
    Path(w.s.summary).write_text(json.dumps(d))
    dst = next(Path(a["dst"]) for a in blk["archive"] if a["src"] == str(raw))
    dst.write_text("{}")                                                      # the archive now holds other bytes
    blk = w.runner()._write("learn", dict(outcome="PASS", reasons=[], n_units=1))
    assert blk["outcome"] == ab_rules.STOP_MACHINE and blk["stop_stage"] == "8" and "archive" not in blk
    assert blk["n_units"] == 1 and blk["archive_clash"] and dst.read_text() == "{}"
    other = Path(w.s.archive_root, "screen")                                  # any other stage still refuses
    ab_store.write_json(w.runner()._detail("screen"), dict(manifest=[]), [])
    other.mkdir(parents=True)
    (other / "x.json").write_text("{}")
    assert _code(lambda: w.runner()._write("screen", dict(outcome="PASS", reasons=[]))) == 2


def test_unbuilt_stage_refused(tmp_path, monkeypatch, capsys):
    """Stage I (plan "Staged build"): a stage without stage_<name> refuses (exit 2). Task 8b deletes this test."""
    w = World(tmp_path, monkeypatch)
    capsys.readouterr()
    assert _code(lambda: w.runner().run("kc_input")) == 2
    assert "not implemented yet" in capsys.readouterr().err


# ================================================================ restart_preseal (plan Reading 15)
def _fixed(monkeypatch):
    real = R.env_now
    f = "flymon/brain/ab_runner.py"
    monkeypatch.setattr(R, "env_now", lambda: (lambda d: dict(d, ab_files=dict(d["ab_files"], **{f: "f" * 64})))(
        real()))


def test_restart_preseal_before_measurement(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("seal_code")
    before = doc()
    led0 = list(before["budget"]["ledger"])
    seal = before["seal_code"]["seal"]["key"]
    monkeypatch.setattr(R, "_head_summary", lambda path: Path(path).read_bytes())     # committed at HEAD
    _fixed(monkeypatch)                                                                # Stage II code added
    assert _code(lambda: w.runner()._require("cal_gate")) == R.EXIT_ENV
    before = doc()
    res = w.runner().restart_preseal(_note(tmp_path))
    assert res["outcome"] == "PASS" and res["sentence"].startswith("사전 측정 재시작 1:")
    d = doc()
    assert set(d) == {"restart_preseal_1", "budget"}
    rb = d["restart_preseal_1"]
    assert rb["outcome"] == "PASS" and rb["old_seal"] == seal and rb["clause"].startswith("AB.7 0")
    assert sorted(rb["invalidated"]) == sorted(["stage0", "reuse", "generate", "seal_code"])
    assert all(v["outcome"] == "INVALID" and v["outcome_was"] == "PASS" for v in rb["invalidated"].values())
    assert d["budget"]["ledger"][:len(led0)] == led0 and "env_mismatch" in d["budget"]["ledger"][-1]
    inv = Path(f"{w.s.raw_dir}.invalid-1")
    arch = Path(w.s.archive_root)
    assert not Path(w.s.raw_dir).exists() and (inv / "stage0.json").exists() and (inv / "tests_0a.log").exists()
    assert json.loads((inv / Path(w.s.summary).name).read_text()) == before
    assert not arch.exists() and (arch.with_name(arch.name + ".invalid-1") / "generate").exists()
    assert json.loads((inv / "restart_preseal.json").read_text())["n"] == 1
    w.write_log("1240 passed in 700.00s\nexit 0\n")                                    # §2 again, then stage0
    out = w.chain("seal_code")
    assert out["outcome"] == "PASS" and doc()["restart_preseal_1"] == rb
    assert doc()["stage0"]["env"]["ab_files"]["flymon/brain/ab_runner.py"] == "f" * 64
    assert doc()["seal_code"]["seal"]["key"] == seal                                   # same seal (Reading 33)
    assert len(doc()["budget"]["ledger"]) == len(led0) + 1 + 4
    res2 = w.runner().restart_preseal(_note(tmp_path))                                 # slot 2, record kept
    assert res2["restart"]["n"] == 2 and set(doc()) == {"restart_preseal_1", "restart_preseal_2", "budget"}


def test_restart_preseal_untracked_without_blocks_removes_summary(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.write_log("exit 1\n")
    _fixed(monkeypatch)
    ab_store.append_ledger(w.s.summary, dict(stage="stage0", env_mismatch=["x"], wall_s=0.0), [])
    res = w.runner().restart_preseal(_note(tmp_path))
    assert res["restart"]["summary"].startswith("removed") and not Path(w.s.summary).exists()


@pytest.mark.parametrize("case", ["kc_input", "kc_input_v1", "head_kc_input", "defect_1", "cache", "stop"])
def test_restart_preseal_refused_after_kc_input_or_cache(tmp_path, monkeypatch, case, capsys):
    w = World(tmp_path, monkeypatch)
    w.chain("seal_code")
    d = doc()
    if case in ("kc_input", "kc_input_v1", "defect_1"):
        d[case] = dict(outcome="PASS")
        Path(w.s.summary).write_text(json.dumps(d))
    elif case == "head_kc_input":
        monkeypatch.setattr(R, "_head_summary", lambda path: json.dumps(dict(d, kc_input=dict(outcome="PASS"))).encode())
    elif case == "cache":
        ab_store.write_json(str(Path(w.s.cache_dir, "r_act", "x.json")), {}, [])
    else:
        d["generate"]["outcome"] = ab_rules.STOP_SET_MISMATCH
        Path(w.s.summary).write_text(json.dumps(d))
    before = Path(w.s.summary).read_bytes()
    capsys.readouterr()
    assert _code(lambda: w.runner().restart_preseal(_note(tmp_path))) == 2
    assert "restart_preseal" in capsys.readouterr().err
    assert Path(w.s.raw_dir).exists() and not Path(f"{w.s.raw_dir}.invalid-1").exists()
    assert Path(w.s.archive_root).exists() and Path(w.s.summary).read_bytes() == before


@pytest.mark.parametrize("note", [None, "missing", dict(symptom="", clause="c", cause="x"), dict(symptom="s"), "bad"])
def test_restart_preseal_needs_note(tmp_path, monkeypatch, note):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    if note is None:
        path = None
    elif note == "missing":
        path = str(tmp_path / "nope.json")
    elif note == "bad":
        (tmp_path / "bad.json").write_text("{")
        path = str(tmp_path / "bad.json")
    else:
        (tmp_path / "n.json").write_text(json.dumps(note))
        path = str(tmp_path / "n.json")
    assert _code(lambda: w.runner().restart_preseal(path)) == 2
    assert Path(w.s.raw_dir).exists() and "stage0" in doc()


# ================================================================ candidates (AB.3 6691, plan Reading 17)
def test_candidate_states_move_forward_only(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    keys = [f"b|{i}|x{i}|y{i}" for i in range(6)]
    d = doc()
    d["candidates"] = [dict(key=k, c=i, axis="b", state="untouched") for i, k in enumerate(keys)]
    Path(w.s.summary).write_text(json.dumps(d))
    r = w.runner()
    r._cands_set("oracle", keys, "oracled")
    r._cands_set("oracle", keys[:4], "screened")
    r._cands_set("oracle", keys[:3], "training")
    r._cands_set("oracle", keys[:2], "trained")
    r._cands_set("oracle", keys[:3], "oracled")                                       # never backwards
    r._cands_set("oracle", keys[:1], "training")
    c = ab_rules.counts(r._cands(doc(), "oracle"))
    assert c == dict(untouched=0, oracled=2, screened=1, training=1, trained=2)
    assert ab_rules.counts(doc()["candidates"])["untouched"] == 6                     # the block: at the next write
    assert r._counts(doc()) == ab_rules.counts(doc()["candidates"])
