"""aa_runner (A): units on Y / AA seeds (c = the main-set candidate number), the differential test against Y's pilot
(cache hits only, mutants caught), stage0, reuse (each AA.7 1 item), chain refusals, env mismatch, archives."""
import json
import re
from pathlib import Path

import pytest

from flymon.brain import aa_rules
from flymon.brain import aa_runner as AR
from flymon.brain import aa_store
from flymon.brain.aa_spec import SPEC as AA
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.y_spec import SPEC as Y
from tests.brain.aa_world import LENIENT_C, MAIN, PILOT, World, doc

ROOT = Path(__file__).resolve().parents[2]


def _run_aa():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_aa", ROOT / "scripts/run_aa.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ================================================================ units
def test_units_main_seed_formula_uses_c():
    rows = [MAIN[5], MAIN[200]]
    nv = AR.units(rows, "main", 8, 2, V.lever_edit, brains=("naive",))
    for u in nv:
        c = int(rows[0]["c"]) if u["pair"] == row_key(rows[0]) else int(rows[1]["c"])
        assert u["phases"] == [] and u["idx"] == c
        assert u["probe_seeds"] == [62_000_000 + c * 4_000 + u["fly"] * 100 + k for k in range(8)]
    us = AR.units(rows, "main", 8, 2, V.lever_edit)
    for u in us:
        c = u["idx"]
        b = 64_000_000 + c * 40_000
        ph = dict(R=[["PAM08", 20, b], ["PPL105", 20, b + 20]], N=[[None, 20, b], [None, 20, b + 20]],
                  RN=[["PAM08", 20, b], [None, 20, b + 20]])[u["brain"]]
        assert u["phases"] == ph
        assert u["probe_seeds"] == [62_000_000 + c * 4_000 + u["fly"] * 100 + k for k in range(8)]
    # mutant "c 대신 31쌍 안의 번호": the lenient position gives other seeds
    r = MAIN[200]
    pos = sorted(LENIENT_C).index(r["c"]) if r["c"] in LENIENT_C else 7
    a = AR.units([r], "main", 8, 1, V.lever_edit)
    b = AR.units([dict(r, c=pos)], "main", 8, 1, V.lever_edit)
    assert all(u["idx"] == r["c"] for u in a)
    assert [u["probe_seeds"] for u in a] != [u["probe_seeds"] for u in b]
    assert [u["phases"] for u in a] != [u["phases"] for u in b]


def test_units_pilot_and_smoke_seeds():
    us = AR.units(PILOT, "pilot", 8, 8, V.lever_edit)
    assert len(us) == 4 * 8 * 3
    for u in us:
        j = u["idx"]
        assert u["probe_seeds"] == [60_000_000 + j * 4_000 + u["fly"] * 100 + k for k in range(8)]
        assert u["phases"][0][2] == 61_000_000 + j * 40_000 and u["phases"][1][2] == 61_000_000 + j * 40_000 + 20
    sm = AR.units(PILOT[:1], "smoke", 8, 1, V.lever_edit)
    ss = AR.smoke_seed_set(AA)
    for u in sm:
        assert u["idx"] == "smoke"
        assert u["probe_seeds"] == [87_100_000 + u["fly"] * 100 + k for k in range(8)]
        assert u["phases"][0][2] == 87_110_000 and u["phases"][1][2] == 87_110_020
        assert set(u["probe_seeds"]) <= ss
    main = AR.units(MAIN, "main", 8, 8, V.lever_edit, brains=("naive",))
    assert not any(s in ss for u in main for s in u["probe_seeds"])


def test_declared_and_manifest_helpers():
    d = AR.declared(V.lever_edit, {0: [1]})
    assert d["csc_sha256"] == V.sha_combined and d["edit_edges"] == V.lever_edges and d["probe_seeds"] == {0: [1]}
    assert AR.declared("none", {})["edit_edges"] == 0
    rows = [dict(unit=dict(pair="p", fly=0, brain="R")), dict(unit=dict(pair="q", fly=0, brain="R"))]
    assert list(AR.by_pair(rows)) == ["p", "q"]


# ================================================================ the differential test
def test_differential_bit_equal_on_y_path(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = AR.differential(w.ctx, w.s)
    assert out["ok"], out["reasons"][:5]
    assert out["n_units"] == 96 and out["n_hits"] == 96
    assert set(out["pairs"]) == set(Y.pilot_v_pairs)
    for p in out["pairs"].values():
        assert p == dict(counts_equal=True, gate_stats_equal=True, record_equal=True, admission_equal=True)
    assert w.pool.jobs == w.pool_jobs_after_y


def _mut_seed(orig):
    def f(rows, where, K, F, edit, **kw):
        out = orig(rows, where, K, F, edit, **kw)
        return [dict(u, probe_seeds=[s + 1 for s in u["probe_seeds"]]) for u in out]
    return f


def _mut_brain(orig):
    def f(rows, where, K, F, edit, **kw):
        out = orig(rows, where, K, F, edit, **kw)
        ph = {}
        for u in out:
            ph[(u["pair"], u["fly"], u["brain"])] = u["phases"]
        sw = dict(R="RN", RN="R", N="N")
        return [dict(u, phases=ph[(u["pair"], u["fly"], sw[u["brain"]])]) if u["brain"] in sw else u for u in out]
    return f


def _mut_stage(orig):
    def f(rows, flies):
        d = orig(rows, flies)
        return dict(d, R1=d["R2"], R2=d["R1"])
    return f


def _mut_main(orig):
    def f(rows, where, K, F, edit, **kw):
        if where == "pilot":
            return orig([dict(r, c=j) for j, r in enumerate(rows)], "main", K, F, edit, **kw)
        return orig(rows, where, K, F, edit, **kw)
    return f


@pytest.mark.parametrize("name,attr,mk", [("seed+1", "units", _mut_seed), ("brain order", "units", _mut_brain),
                                          ("stage order", "pair_counts", _mut_stage),
                                          ("main formula", "units", _mut_main)])
def test_differential_catches_mutants(tmp_path, monkeypatch, name, attr, mk):
    w = World(tmp_path, monkeypatch)
    monkeypatch.setattr(AR, attr, mk(getattr(AR, attr)))
    out = AR.differential(w.ctx, w.s)
    assert out["ok"] is False and out["reasons"], name
    assert w.pool.jobs == w.pool_jobs_after_y


def test_differential_cache_miss_is_failure(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    f = sorted(Path(Y.cache_dir, "w_learn").glob("*.json"))[0]
    f.unlink()
    out = AR.differential(w.ctx, w.s)
    assert out["ok"] is False and any("캐시 적중 실패" in r for r in out["reasons"])
    assert w.pool.jobs == w.pool_jobs_after_y


# ================================================================ stage0
def test_stage0_pass_block(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = w.runner().run("stage0")
    assert out["outcome"] == "PASS", out.get("reasons")
    blk = doc()["stage0"]
    assert blk["tests"]["last_line"] == "exit 0" and re.search(r"\d+ passed", blk["tests"]["passed_line"])
    assert blk["differential"]["ok"] and blk["synth"]["label"] == "기록"
    assert set(AR.aa_files()) <= set(blk["decision_files"])
    assert blk["numbers"] == aa_store.to_json(w.s)
    assert blk["env"]["aa_files"] and blk["order"] == "0a" and "archive" in blk
    for k in ("git", "written_at", "w_measure_key", "u_measure_key"):
        assert k in blk
    c = doc()["candidates"]
    assert len(c) == 249 and all(x["state"] == "untouched" for x in c)
    assert [x["key"] for x in c] == [row_key(r) for r in MAIN] and [x["c"] for x in c] == list(range(249))
    assert doc()["budget"]["ledger"][-1]["stage"] == "stage0"


def test_differential_duplicate_unit_key_is_clean_failure(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    orig = AR.units
    monkeypatch.setattr(AR, "units", lambda *a, **k: (lambda U: U + U[:1])(orig(*a, **k)))
    out = AR.differential(w.ctx, w.s)
    assert out["ok"] is False and any("AA 단위 키 중복 1개" in r for r in out["reasons"])


@pytest.mark.parametrize("text,why", [
    ("exit 0\n", "통과 개수 줄"),                                   # -qq: no "N passed" line, exit 0 alone
    ("..........\nexit 0\n", "통과 개수 줄"),
    ("1 failed, 1233 passed in 600.00s\nexit 0\n", "실패 · 오류 요약"),
    ("1233 passed, 2 errors in 600.00s\nexit 0\n", "실패 · 오류 요약"),
])
def test_stage0_requires_passed_line_and_no_failures(tmp_path, monkeypatch, text, why):
    w = World(tmp_path, monkeypatch)
    Path(w.s.tests_log).write_text(text)
    out = w.runner().run("stage0")
    assert out["outcome"] == "INVALID" and any(why in r for r in out["reasons"]), out["reasons"]


@pytest.mark.parametrize("case", ["no_log", "exit_1", "mutant"])
def test_stage0_invalid_cases(tmp_path, monkeypatch, case):
    w = World(tmp_path, monkeypatch)
    if case == "no_log":
        Path(w.s.tests_log).unlink()
    elif case == "exit_1":
        Path(w.s.tests_log).write_text("3 failed, 10 passed\nexit 1\n")
    else:
        monkeypatch.setattr(AR, "units", _mut_seed(AR.units))
    out = w.runner().run("stage0")
    assert out["outcome"] == "INVALID" and out["reasons"] and "archive" not in out
    assert _run_aa().exit_code(out) == 5
    with pytest.raises(SystemExit) as e:
        w.runner().archive("stage0")
    assert e.value.code == 2


# ================================================================ reuse
def _break(w, case):
    if case == "y_anc":
        w.facts_ok = False
    elif case == "y_outcome":
        d = json.loads(Path(Y.summary).read_text())
        d["precheck"]["outcome"] = "STOP_X"
        Path(Y.summary).write_text(json.dumps(d))
    elif case == "key":
        w.ctx["keys"] = lambda: dict(w_measure_key="0" * 64, u_measure_key=Y.u_measure_key)
    elif case == "oracle_bytes":
        Path(Y.oracle_detail).write_text(Path(Y.oracle_detail).read_text() + " ")
    elif case == "gates_block":
        d = json.loads(Path(Y.summary).read_text())
        d["gates"] = dict(outcome="PASS")
        Path(Y.summary).write_text(json.dumps(d))
    elif case == "learn_progress":
        Path("results/y/progress").mkdir(parents=True, exist_ok=True)
        Path("results/y/progress/learn.json").write_text("{}")
    elif case == "z_anc":
        zc = {c for _, c in AA.z_blocks}
        w.ctx["git_facts"] = lambda path, commits: dict(last="x", ancestors={c: c not in zc for c in commits})
    elif case == "decl":
        real = w.ctx["decl_sha"]

        def decl(files, commit):
            d = real(files, commit)
            d["flymon/brain/y_rules.py"] = "x" * 64
            return d
        w.ctx["decl_sha"] = decl


@pytest.mark.parametrize("case", [None, "y_anc", "y_outcome", "key", "oracle_bytes", "gates_block", "learn_progress",
                                  "z_anc", "decl"])
def test_reuse_pass_and_each_break(tmp_path, monkeypatch, case):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    if case:
        _break(w, case)
    out = w.runner().run("reuse")
    if case is None:
        assert out["outcome"] == "PASS", out.get("reasons")
        assert out["lenient_axes"] == {"b": 11, "a": 20}
        assert "keys" not in json.dumps(doc()["reuse"]).replace('"w_measure_key"', "").replace('"u_measure_key"', "")
        return
    assert out["outcome"] == aa_rules.STOP_REUSE and out["reasons"]
    assert out["sentence"].startswith("AA 재사용 조건(AA.7 1)이 깨졌다(")
    assert out["sentence"].endswith("(후보 상태: untouched 249 · screened 0 · trained 0)")
    if case == "oracle_bytes":
        assert any("oracle.json" in r for r in out["reasons"])
    assert _run_aa().exit_code(out) == 3


# ================================================================ chain, env, archive
def test_chain_refusals(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        w.runner().run("reuse")
    assert e.value.code == 2
    w.chain("stage0")
    with pytest.raises(SystemExit) as e:
        w.runner().run("stage0")
    assert e.value.code == 2
    monkeypatch.setattr(AR, "summary_git", lambda p: dict(tracked=True, dirty=True))
    with pytest.raises(SystemExit) as e:
        w.runner().run("reuse")
    assert e.value.code == 2


def test_later_stage_needs_matching_keys(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    w.ctx["keys"] = lambda: dict(w_measure_key="0" * 64, u_measure_key=Y.u_measure_key)
    with pytest.raises(SystemExit) as e:
        w.runner()._require("smoke")
    assert e.value.code == 2


def test_env_mismatch_exit_6_and_resume(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    real = AR.env_now
    monkeypatch.setattr(AR, "env_now", lambda: dict(real(), numpy="0.0.0"))
    with pytest.raises(SystemExit) as e:
        w.runner().run("reuse")
    assert e.value.code == AR.EXIT_ENV == 6
    led = doc()["budget"]["ledger"][-1]
    assert led["stage"] == "reuse" and any("numpy" in x for x in led["env_mismatch"])
    assert "reuse" not in doc()
    # an aa_* file added later is not a mismatch
    monkeypatch.setattr(AR, "env_now", lambda: (lambda d: dict(d, aa_files=dict(d["aa_files"], **{
        "flymon/brain/aa_new.py": "n" * 64})))(real()))
    assert w.runner().run("reuse")["outcome"] == "PASS"


def test_env_resume_compares_first_start(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    real = AR.env_now
    p = Path(w.s.progress_dir) / "reuse.env.json"
    aa_store.write_json(str(p), dict(real(), python="2.7"), [])
    with pytest.raises(SystemExit) as e:
        w.runner().run("reuse")
    assert e.value.code == 6


def test_archive_before_block_and_idempotent(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = w.runner().run("stage0")
    assert out["archive"] and all(Path(a["dst"]).exists() for a in out["archive"])
    again = w.runner().archive("stage0")
    assert again == out["archive"]
    with pytest.raises(SystemExit) as e:
        w.runner().archive("reuse")
    assert e.value.code == 2


def test_unknown_or_unimplemented_stage_refuses(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        w.runner().run("judge")
    assert e.value.code == 2


def test_seal_check(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    r = w.runner()
    now = AR.seal_now(w.s)
    r._seal_check(dict(seal_code=dict(seal=now)))
    with pytest.raises(SystemExit) as e:
        r._seal_check(dict(seal_code=dict(seal=dict(now, key="0" * 64))))
    assert e.value.code == AR.EXIT_SEAL == 7
    r._seal_check(dict(seal_code=dict(seal=dict(now, key="0" * 64)), reseal_1=dict(seal=dict(key="1" * 64)),
                       reseal_2=dict(seal=now)))
    with pytest.raises(SystemExit):
        r._seal_check(dict(seal_code=dict(seal=now), reseal_1=dict(seal=now), reseal_2=dict(seal=dict(key="2" * 64))))


def test_candidate_states_move_forward_only(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("stage0")
    r = w.runner()
    keys = [row_key(MAIN[c]) for c in LENIENT_C]
    r._cands_set("screen", keys, "screened")
    r._cands_set("screen", keys[:2], "trained")
    r._cands_set("screen", keys[:3], "screened")
    c = aa_rules.candidate_counts(r._cands(doc(), "screen"))
    assert c == dict(untouched=218, screened=29, trained=2)
    assert aa_rules.candidate_counts(doc()["candidates"])["untouched"] == 249
