"""aa_runner (B): smoke (+ budget), seal_code, the naive screen (machine checks, S1 manifest, candidates,
STOP_NO_PAIRS, resume), coverage 4b (structure only, raw re-read, cell resume)."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import aa_estimate, aa_rules, w_records, y_rules
from flymon.brain import aa_runner as AR
from flymon.brain.r_pairs import row_key
from flymon.brain.y_spec import SPEC as Y
from tests.brain.aa_world import LENIENT_C, MAIN, World, doc

LKEYS = [row_key(MAIN[c]) for c in LENIENT_C]
C_OF = {row_key(r): r["c"] for r in MAIN}


def _world(tmp_path, monkeypatch, **kw):
    w = World(tmp_path, monkeypatch)
    if kw:
        w.s = dataclasses.replace(w.s, **kw)
    return w


def _counts(c):
    return aa_rules.candidate_counts(c)


def _screen_detail(w):
    return json.loads(Path(w.s.raw_dir, "screen.json").read_text())


# ================================================================ order 2: smoke
def test_smoke_pass_and_costs(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    out = w.chain("smoke")
    blk = doc()["smoke"]
    assert out["outcome"] == "PASS" and blk["problems"] == []
    assert blk["costs"]["trial_s"] > 0 and blk["costs"]["naive_job_s"] > 0 and blk["costs"]["learn_job_max_s"] > 0
    assert set(blk["estimates_h"]) == {"screen", "coverage", "learn", "estimate"}
    assert blk["budget"]["ok"] is True and blk["order"] == "2"
    assert not Path(w.s.cache_dir).exists()
    ss = AR.smoke_seed_set(w.s)
    assert len(blk["manifest"]) == 4                                 # R · N · RN + naive, one fly
    for m in blk["manifest"]:
        assert Path(m["cache_file"]).resolve().is_relative_to(Path(w.s.smoke_cache_dir).resolve())
        ins = json.loads(Path(m["cache_file"]).read_text())["inputs"]
        assert ins["pair"] == Y.pilot_v_pairs[0] and set(ins["probe_seeds"]) <= ss
        assert ins["probe_seeds"] == [87_100_000 + k for k in range(8)]
    assert doc()["budget"]["ledger"][-1]["stage"] == "smoke"


def test_smoke_budget_stop(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("reuse")
    w.model.train_s = 10.0                    # learn job ≈ 400 s; ⌈744 / 4⌉ × 400 s ≈ 20.7 h
    out = w.runner().run("smoke")
    assert out["outcome"] == aa_rules.STOP_BUDGET
    assert "(2)" in out["sentence"] and "주 세트 미측정" in out["sentence"]
    assert out["sentence"].endswith("(후보 상태: untouched 249 · screened 0 · trained 0)")
    assert out["budget"]["ok"] is False and out["estimates_h"]["learn"] > 12


def _naive_seed_mutant(orig):
    def f(rows, where, K, F, edit, **kw):
        out = orig(rows, where, K, F, edit, **kw)
        if kw.get("brains") == ("naive",):
            return [dict(u, probe_seeds=[x + 1 for x in u["probe_seeds"]]) for u in out]
        return out
    return f


@pytest.mark.parametrize("case", ["share_state", "naive_seeds"])
def test_smoke_pre_mismatch_is_a_problem(tmp_path, monkeypatch, case):
    w = _world(tmp_path, monkeypatch)
    w.chain("reuse")
    if case == "share_state":
        w.model.share_state = True
    else:
        monkeypatch.setattr(AR, "units", _naive_seed_mutant(AR.units))
    out = w.runner().run("smoke")
    assert out["outcome"] == aa_rules.INVALID and out["problems"] and "archive" not in out
    if case == "naive_seeds":
        assert any("순진" in p for p in out["problems"])
    else:
        assert any("RN" in p for p in out["problems"])


# ================================================================ order 3: seal_code
def test_seal_code_records_hashes_and_then_binds(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, workers=1)
    w.chain("seal_code")
    blk = doc()["seal_code"]
    assert blk["outcome"] == "PASS" and blk["order"] == "3"
    assert set(blk["seal"]["files"]) == set(w.s.seal_files)
    assert blk["seal"]["numbers"] == json.loads(json.dumps(AR._js(w.s.seal_fields())))
    assert blk["seal"] == AR.seal_now(w.s)
    real = AR.seal_now
    monkeypatch.setattr(AR, "seal_now", lambda s=w.s: dict(real(s), key="0" * 64))
    assert w.runner().run("screen")["outcome"] == "PASS"            # the screen is not sealed
    with pytest.raises(SystemExit) as e:
        w.runner().run("coverage")
    assert e.value.code == AR.EXIT_SEAL == 7
    assert "coverage" not in doc()


# ================================================================ order 4: screen
def test_screen_pass_s1_manifest_and_candidates(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    for c in LENIENT_C[:3]:                                           # three fail the balance filter
        w.model.offset[row_key(MAIN[c])] = 40.0
    out = w.runner().run("screen")
    assert out["outcome"] == "PASS"
    blk = doc()["screen"]
    assert [p["key"] for p in blk["pairs"]] == LKEYS and [p["c"] for p in blk["pairs"]] == list(LENIENT_C)
    det = _screen_detail(w)
    pre, dig, bad = w.runner()._screen_raw(doc())
    assert bad == [] and dig == blk["raw_digest"] and set(pre) == set(LKEYS)
    for p in blk["pairs"]:
        f = y_rules.final_filter(pre[p["key"]], w.z, Y)
        assert (p["naive_d"], p["L_A"], p["L_P"], p["passed"]) == (f["d"], f["L_A"], f["L_P"], f["passed"])
        assert p["x_odour"] == y_rules.x_odour(p["key"]) and p["axis"] == p["key"].split("|")[0]
        assert pre[p["key"]].shape == (8, 8, 2, 2)
    assert [p["passed"] for p in blk["pairs"][:3]] == [False] * 3
    assert blk["k"] == sum(p["passed"] for p in blk["pairs"]) == 28
    s1 = [p["key"] for p in blk["pairs"] if p["passed"]]
    out_ = [p["key"] for p in blk["pairs"] if not p["passed"]]
    for name, keys in (("S1", s1), ("S31", LKEYS), ("out", out_)):
        g = blk["groups"][name]
        assert g["n"] == len(keys)
        assert g["sizes"] == sorted((len(x) for x in y_rules.merge_groups(keys)), reverse=True)
    assert blk["reasons_count"] == dict(balance=3, a_floor=0, p_floor=0)
    assert blk["n_units"] == 248 and blk["seeds_ok"] is True
    assert blk["detail_sha256"] == AR.sha256_file(Path(w.s.raw_dir, "screen.json"))
    assert len(det["manifest"]) == 248
    c = doc()["candidates"]
    assert _counts(c) == dict(untouched=218, screened=31, trained=0)
    assert {x["key"] for x in c if x["state"] == "screened"} == set(LKEYS)
    assert all(set(x) == {"key", "c", "axis", "state"} for x in c)
    # no per-unit count anywhere in the block
    txt = json.dumps(blk)
    assert '"counts"' not in txt and '"pre"' not in txt and '"manifest"' not in txt
    # the archive holds the detail and every raw file
    assert len(blk["archive"]) == 249


def test_screen_seeds_use_c_and_main_block(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("screen")
    det = _screen_detail(w)
    seen = set()
    for m in det["manifest"]:
        ins = json.loads(Path(m["cache_file"]).read_text())["inputs"]
        c = C_OF[ins["pair"]]
        assert ins["brain"] == "naive" and ins["block"] == "screen" and ins["phases"] == []
        assert ins["probe_seeds"] == [62_000_000 + c * 4_000 + ins["fly"] * 100 + k for k in range(8)]
        seen.add((ins["pair"], ins["fly"]))
    assert seen == {(k, f) for k in LKEYS for f in range(8)}


def _units_mutant(case):
    orig = AR.units

    def f(rows, where, K, F, edit, **kw):
        out = orig(rows, where, K, F, edit, **kw)
        if where != "main":
            return out
        if case == "dup":
            return out + [dict(out[5])]
        if case == "k7":
            return [dict(out[0], probe_seeds=out[0]["probe_seeds"][:7])] + out[1:]
        if case == "seed":
            return [dict(out[0], probe_seeds=[x + 1 for x in out[0]["probe_seeds"]])] + out[1:]
        if case == "pos":                     # mutant "c 대신 31쌍 안의 번호"
            pos = {r["c"]: j for j, r in enumerate(rows)}
            return orig([dict(r, c=pos[r["c"]]) for r in rows], where, K, F, edit, **kw)
        return out
    return f


@pytest.mark.parametrize("case", ["dup", "k7", "seed", "pos", "nan"])
def test_screen_machine_checks(tmp_path, monkeypatch, case):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    if case == "nan":
        real = w.model.job
        bad = LKEYS[4]

        def job(kw, pair):
            r = real(kw, pair)
            if pair == bad and kw["fly"] == 2:
                r["stages"][0]["x"][3]["A"] = float("nan")
            return r
        w.model.job = job
    else:
        monkeypatch.setattr(AR, "units", _units_mutant(case))
    out = w.runner().run("screen")
    assert out["outcome"] == aa_rules.STOP_MACHINE, (case, out.get("reasons"))
    assert out["seeds_ok"] is (case not in ("seed", "k7", "pos"))      # from the seed comparison, not reason text
    assert "AA 순진 거름 측정에서 기계 검사가 맞지 않았다(" in out["sentence"]
    assert "학습 측정 없음, 31쌍 `screened`" in out["sentence"]
    assert out["sentence"].endswith("(후보 상태: untouched 218 · screened 31 · trained 0)")
    assert "pairs" not in out and "k" not in out
    with pytest.raises(SystemExit) as e:
        w.runner().run("coverage")
    assert e.value.code == 2


def test_seeds_match_direct():
    want = {("p", 0): [1, 2], ("p", 1): [3, 4]}
    ok = [dict(pair="p", fly=0, probe_seeds=[1, 2]), dict(pair="p", fly=1, probe_seeds=[3, 4])]
    assert AR.seeds_match(ok, want) is True
    assert AR.seeds_match([ok[0], dict(ok[1], probe_seeds=[3, 5])], want) is False
    assert AR.seeds_match([ok[0], dict(ok[1], probe_seeds=[3])], want) is False
    assert AR.seeds_match([dict(ok[0], fly=2)], want) is False                     # no formula list for the key


@pytest.mark.parametrize("k0", [False, True])
def test_screen_rereads_raw_right_after_block(tmp_path, monkeypatch, k0):
    """AA.7 4 "블록 커밋 직후 한 번 더 읽어 같음": the screen stage itself re-reads its raw after the block is written,
    so the k = 0 path (no coverage stage follows) is verified too."""
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    if k0:
        for r in MAIN:
            w.model.offset[row_key(r)] = 40.0
    real_write, seen = AR.Runner._write, []

    def write(self, stage, body, *a, **kw):
        out = real_write(self, stage, body, *a, **kw)
        if stage == "screen":                    # the raw changes between the block write and the re-read
            f = Path(_screen_detail(w)["manifest"][7]["cache_file"])
            f.write_text(json.dumps(json.loads(f.read_text()), indent=1))
            seen.append(out["outcome"])
        return out
    monkeypatch.setattr(AR.Runner, "_write", write)
    out = w.runner().run("screen")
    assert seen == [aa_rules.STOP_NO_PAIRS if k0 else aa_rules.PASS]
    assert out["outcome"] == aa_rules.STOP_MACHINE
    assert "원자료 digest 재확인 불일치(블록 직후)" in out["sentence"] and "31쌍 `screened`" in out["sentence"]
    assert out["sentence"].endswith("(후보 상태: untouched 218 · screened 31 · trained 0)")
    blk = doc()["screen"]
    assert blk["outcome"] == aa_rules.STOP_MACHINE and "pairs" not in blk and "k" not in blk
    assert blk["raw_digest_now"] != blk["raw_digest"] and len(blk["archive"]) == 249
    assert [x["stage"] for x in doc()["budget"]["ledger"]].count("screen") == 1
    with pytest.raises(SystemExit) as e:
        w.runner().run("coverage")
    assert e.value.code == 2


def test_screen_reread_runs_on_pass(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    real, n = AR.Runner._screen_raw, dict(i=0)

    def spy(self, d):
        n["i"] += 1
        assert d["screen"]["outcome"] == aa_rules.PASS                # read after the block is written
        return real(self, d)
    monkeypatch.setattr(AR.Runner, "_screen_raw", spy)
    assert w.runner().run("screen")["outcome"] == aa_rules.PASS
    assert n["i"] == 1 and doc()["screen"]["outcome"] == aa_rules.PASS


def test_screen_k0_stop_no_pairs(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    for r in MAIN:
        w.model.offset[row_key(r)] = 40.0
    out = w.runner().run("screen")
    assert out["outcome"] == aa_rules.STOP_NO_PAIRS
    cnt = dict(untouched=218, screened=31, trained=0)
    assert out["sentence"] == aa_rules.no_pairs_stop(dict(balance=31, a_floor=0, p_floor=0), cnt)["sentence"]
    assert out["reasons_count"] == dict(balance=31, a_floor=0, p_floor=0) and out["k"] == 0
    assert out["groups"]["S1"] == dict(n=0, sizes=[])
    assert _counts(doc()["candidates"]) == cnt
    for st in ("coverage", "learn"):
        with pytest.raises(SystemExit) as e:
            w.runner().run(st)
        assert e.value.code == 2


def test_screen_resumes_from_cache(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    real, n = w.model.job, dict(i=0)

    def job(kw, pair):
        n["i"] += 1
        if n["i"] > 50:
            raise RuntimeError("killed")
        return real(kw, pair)
    w.model.job = job
    with pytest.raises(RuntimeError):
        w.runner().run("screen")
    assert "screen" not in doc()
    cached = len(list(Path(w.s.cache_dir, "w_learn").glob("*.json")))
    assert 0 < cached < 248
    prog = json.loads(Path(w.s.progress_dir, "screen.json").read_text())
    assert prog["core"] > 0
    pc = json.loads(Path(w.s.progress_dir, "screen.candidates.json").read_text())
    assert _counts(pc) == dict(untouched=218, screened=31, trained=0)
    w.model.job = real
    j0 = w.pool.jobs
    assert w.runner().run("screen")["outcome"] == "PASS"
    assert w.pool.jobs - j0 == 248 - cached
    assert _counts(doc()["candidates"]) == dict(untouched=218, screened=31, trained=0)
    led = doc()["budget"]["ledger"][-1]
    assert led["stage"] == "screen" and led["wall_s"] >= prog["core"]


# ================================================================ order 4b: coverage
def test_coverage_block_structure_only(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, workers=2)                     # the spawn-pool path
    w.chain("coverage")
    sc, cv = doc()["screen"], doc()["coverage"]
    assert cv["outcome"] == "PASS" and cv["order"] == "4b"
    k = sc["k"]
    assert k == 31 and sc["groups"]["out"]["n"] == 0
    for st in ("S1", "S31"):
        g = sc["groups"][st]
        p = aa_estimate.ci_plan(g["n"], len(g["sizes"]), w.s)
        want = [p["primary"]] + p["records"] + [f"raw_{p['primary']}"]
        assert sorted(cv["by_set"][st]) == sorted(want)
        for m in want:
            e = cv["by_set"][st][m]
            assert len(e["cells"]) == 9 and isinstance(e["under"], bool)
            assert e == AR._js(aa_estimate.coverage(tuple(g["sizes"]), m, st, w.s))
    assert cv["by_set"]["out"] is None and cv["null_reason"]["out"]
    assert "S1" not in cv["null_reason"]


def test_coverage_small_sets(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, workers=1)
    w.chain("seal_code")
    keep = {row_key(MAIN[c]) for c in LENIENT_C[11:14]}               # (a) pairs only: few X-odour groups
    for k in LKEYS:
        if k not in keep:
            w.model.offset[k] = 40.0
    w.runner().run("screen")
    cv = w.runner().run("coverage")
    sc = doc()["screen"]
    g = sc["groups"]["S1"]
    p = aa_estimate.ci_plan(g["n"], len(g["sizes"]), w.s)
    assert g["n"] == 3 and p["primary"] == "fly"
    assert sorted(cv["by_set"]["S1"]) == sorted([p["primary"]] + p["records"] + ["raw_fly"])
    assert cv["by_set"]["out"] is not None and sc["groups"]["out"]["n"] == 28


def test_coverage_rereads_screen_raw(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, workers=1)
    w.chain("screen")
    f = Path(_screen_detail(w)["manifest"][7]["cache_file"])
    f.write_text(json.dumps(json.loads(f.read_text()), indent=1))
    out = w.runner().run("coverage")
    assert out["outcome"] == aa_rules.STOP_MACHINE
    assert "원자료 digest" in out["sentence"] and "31쌍 `screened`" in out["sentence"]
    assert out["sentence"].endswith("(후보 상태: untouched 218 · screened 31 · trained 0)")
    assert "by_set" not in out


@pytest.mark.parametrize("bad", [[2.0, 1], [0, 3], [True, 2], "ab", [2, 2]])
def test_coverage_validates_sizes_before_cells(tmp_path, monkeypatch, bad):
    """The group-size structure is validated (aa_estimate._groups + sum = n) before any coverage_cell call."""
    w = _world(tmp_path, monkeypatch, workers=1)
    w.chain("screen")
    d = doc()
    d["screen"]["groups"]["S1"] = dict(n=3, sizes=bad)
    Path(AR.AA.summary).write_text(json.dumps(d))
    calls = []
    monkeypatch.setattr(aa_estimate, "coverage_cell", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(AR, "_cov_job", lambda *a: calls.append(a))
    with pytest.raises(SystemExit) as e:
        w.runner().run("coverage")
    assert e.value.code == 2 and calls == [] and "coverage" not in doc()


def test_coverage_resumes_cells(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, workers=1)
    w.chain("screen")
    real, n = AR._cov_job, dict(i=0)

    def job(*a):
        n["i"] += 1
        if n["i"] > 10:
            raise RuntimeError("killed")
        return real(*a)
    monkeypatch.setattr(AR, "_cov_job", job)
    with pytest.raises(RuntimeError):
        w.runner().run("coverage")
    assert "coverage" not in doc()
    cells = list(Path(w.s.progress_dir, "coverage").glob("*.npz"))
    assert len(cells) == 10
    n["i"] = -10_000
    out = w.runner().run("coverage")
    assert out["outcome"] == "PASS"
    total = sum(len(v) for v in out["by_set"].values() if v) * 9
    assert n["i"] + 10_000 == total - 10


def test_raw_digest_shape_and_order():
    a = dict(stages=[dict(x=[dict(A=1, P=2)], y=[dict(A=3, P=4)])])
    b = dict(stages=[dict(x=[dict(A=3, P=4)], y=[dict(A=1, P=2)])])
    assert AR.raw_digest([a, b]) != AR.raw_digest([b, a])
    assert AR.raw_digest([a]) == AR.raw_digest([dict(a)])
    c = AR.counts_f(dict(x=[dict(A=float("nan"), P=1)], y=[dict(A=1, P=1)]))
    assert c.shape == (1, 2, 2) and np.isnan(c[0, 0, 0])
    assert np.array_equal(AR.counts_f(a["stages"][0]), w_records.counts(a["stages"][0]))
