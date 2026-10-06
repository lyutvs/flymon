"""aa_runner (C): learn (all 31 pairs in c order, `trained` at round start, reservation and measured-cap budget stops,
pre = naive pre / RN1 = R1 machine checks, resume) and estimate (S1 from the screen block, small-k paths, repeat bit
check, result sentence, comparison table, seal refusal)."""
import json
import time
from pathlib import Path

import pytest

from flymon.brain import aa_estimate, aa_rules, y_rules
from flymon.brain import aa_runner as AR
from flymon.brain.h3_store import canonical
from flymon.brain.r_pairs import row_key
from tests.brain.aa_world import LENIENT_C, MAIN, World, doc

LKEYS = [row_key(MAIN[c]) for c in LENIENT_C]
STAT_KEYS = {"d", "mean", "naive_d", "gate", "gates", "point", "ci", "L_A", "L_P"}


def _world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    import dataclasses
    w.s = dataclasses.replace(w.s, workers=1)                         # cell stages in-process (speed)
    return w


def _counts(c):
    return aa_rules.candidate_counts(c)


def _keys(o):
    if isinstance(o, dict):
        return set(o) | {k for v in o.values() for k in _keys(v)}
    if isinstance(o, list):
        return {k for v in o for k in _keys(v)}
    return set()


def _edit_summary(fn):
    p = Path(AR.AA.summary)
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def _learn_job(w, fn):
    """Wrap the model's job for learn jobs only (phases non-empty, block-free: the model never sees the block)."""
    real = w.model.job

    def job(kw, pair):
        if kw["phases"]:
            return fn(kw, pair, real)
        return real(kw, pair)
    w.model.job = job
    return real


def _cache_learn(w):
    out = 0
    for f in Path(w.s.cache_dir, "w_learn").glob("*.json"):
        if json.loads(f.read_text())["inputs"]["block"] == "learn":
            out += 1
    return out


# ================================================================ order 5: learn
def test_learn_measures_all_31_in_c_order_and_marks_trained(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    for c in LENIENT_C[:3]:                                           # failing pairs are learned too
        w.model.offset[row_key(MAIN[c])] = 40.0
    w.chain("coverage")
    assert doc()["screen"]["k"] == 28
    seen, order = [], []
    cp = Path(w.s.progress_dir, "learn.candidates.json")

    def fn(kw, pair, real):
        st = {x["key"]: x["state"] for x in json.loads(cp.read_text())} if cp.exists() else {}
        seen.append(st.get(pair))
        if not order or order[-1] != pair:
            order.append(pair)
        return real(kw, pair)
    _learn_job(w, fn)
    j0 = w.pool.jobs
    out = w.runner().run("learn")
    assert out["outcome"] == "PASS", (out.get("reasons"), out.get("sentence"))
    assert w.pool.jobs - j0 == 744 and out["n_units"] == 744 and out["n_trained"] == 31
    assert order == LKEYS                                             # c ascending, every lenient pair
    assert seen and all(x == "trained" for x in seen)                 # marked before the pair's first unit ran
    blk = doc()["learn"]
    assert not (_keys(blk) & STAT_KEYS), _keys(blk) & STAT_KEYS
    assert blk["order"] == "5" and blk["detail_sha256"] == AR.sha256_file(Path(w.s.raw_dir, "learn.json"))
    det = json.loads(Path(w.s.raw_dir, "learn.json").read_text())
    assert len(det["manifest"]) == 744
    assert blk["manifest_sha256"] == AR.hashlib.sha256(canonical(det["manifest"]).encode()).hexdigest()
    brains = {}
    for m in det["manifest"]:
        ins = json.loads(Path(m["cache_file"]).read_text())["inputs"]
        assert ins["block"] == "learn"
        brains.setdefault(ins["pair"], set()).add((ins["fly"], ins["brain"]))
    assert set(brains) == set(LKEYS)
    assert all(v == {(f, b) for f in range(8) for b in ("R", "N", "RN")} for v in brains.values())
    c = doc()["candidates"]
    assert _counts(c) == dict(untouched=218, screened=0, trained=31)
    assert len(blk["archive"]) == 745
    assert doc()["budget"]["ledger"][-1]["stage"] == "learn"


@pytest.mark.parametrize("case", ["share_state", "pre_noise"])
def test_learn_pre_equals_naive_pre_and_rn1(tmp_path, monkeypatch, case):
    w = _world(tmp_path, monkeypatch)
    w.chain("coverage")
    if case == "share_state":
        w.model.share_state = True
    else:
        bad = LKEYS[6]

        def fn(kw, pair, real):                                      # the learn path's pre ≠ the screen's pre
            r = real(kw, pair)
            if pair == bad:
                r["stages"][0]["x"][2]["A"] += 1
            return r
        _learn_job(w, fn)
    out = w.runner().run("learn")
    assert out["outcome"] == aa_rules.STOP_MACHINE, out.get("reasons")
    assert "AA 학습 측정에서 기계 검사가 맞지 않았다(" in out["sentence"]
    assert "학습 31쌍 `trained`" in out["sentence"]
    assert out["sentence"].endswith("(후보 상태: untouched 218 · screened 0 · trained 31)")
    txt = out["reasons"][0]
    if case == "share_state":
        assert "RN" in txt
    else:
        assert "학습 pre ≠ 순진 거름 프로브" in txt and LKEYS[6] in txt
    with pytest.raises(SystemExit) as e:
        w.runner().run("estimate")
    assert e.value.code == 2


def test_learn_reservation_stop_before_start(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("coverage")

    def pad(d):
        d["budget"]["ledger"].append(dict(stage="pad", wall_s=11.9 * 3600, at="x"))
        d["smoke"]["estimates_h"]["learn"] = 1.0
    _edit_summary(pad)
    j0 = w.pool.jobs
    out = w.runner().run("learn")
    assert out["outcome"] == aa_rules.STOP_BUDGET
    assert "(5 전)" in out["sentence"] and "순진 거름만 측정(31쌍 `screened`, 학습 0)" in out["sentence"]
    assert out["sentence"].endswith("(후보 상태: untouched 218 · screened 31 · trained 0)")
    assert w.pool.jobs == j0 and _cache_learn(w) == 0
    assert _counts(doc()["candidates"]) == dict(untouched=218, screened=31, trained=0)


def test_learn_measured_cap_stop(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("coverage")
    margin_s = 1.5

    def pad(d):
        d["budget"]["ledger"].append(dict(stage="pad", wall_s=12 * 3600 - sum(e["wall_s"] for e in d["budget"]["ledger"])
                                          - margin_s, at="x"))
        d["smoke"]["estimates_h"]["learn"] = 0.0
        d["smoke"]["estimates_h"]["estimate"] = 0.0
    _edit_summary(pad)
    n = dict(i=0)

    def fn(kw, pair, real):                                          # the first pair is fast, then slow
        n["i"] += 1
        if n["i"] > 24:
            time.sleep(0.6)
        return real(kw, pair)
    _learn_job(w, fn)
    out = w.runner().run("learn")
    assert out["outcome"] == aa_rules.STOP_BUDGET, out.get("sentence")
    assert "(5)" in out["sentence"]
    done = AR.Runner(w.ctx, w.s, pool=w.pool)._pairs_complete(
        AR.units([MAIN[c] for c in LENIENT_C], "main", 8, 8, AR.V_SPEC.lever_edit), w.ctx["measurer"](None))
    assert done == 1 and "학습 1쌍 완료" in out["sentence"]
    assert n["i"] < 744
    tr = _counts(doc()["candidates"])["trained"]
    assert tr >= 1 and out["sentence"].endswith(f"(후보 상태: untouched 218 · screened {31 - tr} · trained {tr})")
    assert "manifest_sha256" not in out
    with pytest.raises(SystemExit) as e:                             # no partial estimate (AA.9.2 해석 12)
        w.runner().run("estimate")
    assert e.value.code == 2


def test_learn_resumes_and_keeps_trained_marks(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("coverage")
    n = dict(i=0)

    def fn(kw, pair, real):
        n["i"] += 1
        if n["i"] > 100:
            raise RuntimeError("killed")
        return real(kw, pair)
    real = _learn_job(w, fn)
    with pytest.raises(RuntimeError):
        w.runner().run("learn")
    assert "learn" not in doc()
    cached = _cache_learn(w)
    assert 0 < cached < 744
    pc = json.loads(Path(w.s.progress_dir, "learn.candidates.json").read_text())
    tr = {x["key"] for x in pc if x["state"] == "trained"}
    assert tr == set(LKEYS[:len(tr)]) and 0 < len(tr) < 31            # the first pairs in c order
    assert json.loads(Path(w.s.progress_dir, "learn.json").read_text())["core"] > 0
    w.model.job = real
    j0 = w.pool.jobs
    out = w.runner().run("learn")
    assert out["outcome"] == "PASS"
    assert w.pool.jobs - j0 == 744 - cached
    assert _counts(doc()["candidates"]) == dict(untouched=218, screened=0, trained=31)


def test_learn_refuses_after_no_pairs(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    for r in MAIN:
        w.model.offset[row_key(r)] = 40.0
    assert w.runner().run("screen")["outcome"] == aa_rules.STOP_NO_PAIRS
    j0 = w.pool.jobs
    with pytest.raises(SystemExit) as e:
        w.runner().run("learn")
    assert e.value.code == 2 and w.pool.jobs == j0


# ================================================================ order 6: estimate
def test_estimate_primary_two_stage(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    for c in LENIENT_C[:2]:                                           # two (b) pairs fail: S1 ≠ S31
        w.model.offset[row_key(MAIN[c])] = 40.0
    out = w.chain("estimate")
    d = doc()
    pr = out["primary"]
    s1 = [p["key"] for p in d["screen"]["pairs"] if p["passed"]]
    assert len(s1) == 29 and [p["key"] for p in pr["pairs"]] == s1
    assert pr["plan"]["primary"] == "two_stage" and pr["kind"] == "pooled" and pr["flag"] is None
    assert pr["k"] == 29 and pr["n_groups"] >= 5
    data = w.runner()._learn_data(d)
    arrs = {k: aa_estimate.pair_arrays(data[k], w.z, w.s) for k in s1}
    est = aa_estimate.estimate_set(arrs, s1, "S1", w.s)
    for g in aa_estimate.PRIMARY:
        assert pr["pooled"][g]["point"] == est["point"][g]
        assert pr["pooled"][g]["ci"] == est["ci"]["two_stage"][g] and pr["pooled"][g]["method"] == "two_stage"
        assert pr["range"][g] == est["range"][g]
    p0 = pr["pairs"][0]
    ci = aa_estimate.pair_ci(arrs[p0["key"]], p0["c"], w.s)["ci"]
    assert p0["reward_assoc"]["ci"] == ci["reward_assoc"]
    cov = d["coverage"]["by_set"]["S1"]["two_stage"]
    assert pr["coverage"] == cov["min"] and pr["under"] == cov["under"]
    s = out["sentence"]
    assert "묶음-마리 두 단계" in s and f"통과한 29쌍(X 냄새 묶음 {pr['n_groups']}개)" in s
    assert not any(x in s.replace("PASS/FAIL이 아니다", "") for x in aa_rules.FORBIDDEN_RESULT_WORDS)
    tau = est["dl"]["reward_assoc"]
    assert (f"쌍 간 SD 보상 {aa_rules._n(tau['tau'] if tau['available'] else None)}") in s
    assert out["repeat_equal"] is True and out["seal_key"] == AR.seal_now(w.s)["key"]
    assert set(out["compare"]) == set(aa_estimate.PRIMARY)
    assert out["compare"]["reward_assoc"] == AR._js(aa_estimate.compare_table(
        est["point"]["reward_assoc"], *est["ci"]["two_stage"]["reward_assoc"], w.s))
    assert len(out["compare"]["punish_assoc"]) == 14
    assert d["budget"]["ledger"][-1]["stage"] == "estimate" and "archive" in d["estimate"]


@pytest.mark.parametrize("k", [1, 3])
def test_estimate_small_k_paths(tmp_path, monkeypatch, k):
    w = _world(tmp_path, monkeypatch)
    keep = {row_key(MAIN[c]) for c in LENIENT_C[11:11 + k]}           # (a) pairs, 3 distinct X odours
    for x in LKEYS:
        if x not in keep:
            w.model.offset[x] = 40.0
    out = w.chain("estimate")
    pr = out["primary"]
    assert pr["k"] == k and [p["key"] for p in pr["pairs"]] == [x for x in LKEYS if x in keep]
    if k == 1:
        assert pr["kind"] == "pair_study" and "pooled" not in pr and "range" not in pr
        assert out["sentence"].startswith(f"쌍별 연구(k = 1) — 쌍 {pr['pairs'][0]['key']}의 쌍별 추정")
        assert "통합" not in out["sentence"]
        p = pr["pairs"][0]
        assert out["compare"]["punish_assoc"] == AR._js(aa_estimate.compare_table(
            p["punish_assoc"]["mean"], *p["punish_assoc"]["ci"], w.s))
    else:
        assert pr["kind"] == "pooled" and pr["plan"]["primary"] == "fly" and pr["flag"] == "적은 묶음"
        assert pr["n_groups"] == 3 and set(pr["range"]) == set(aa_estimate.PRIMARY)
        assert pr["coverage"] == doc()["coverage"]["by_set"]["S1"]["fly"]["min"]
        s = out["sentence"]
        assert "'적은 묶음' 표식" in s and "마리 단계(묶음 < 5, 쌍별 값 범위 보상 " in s
        (a0, a1), (p0, p1) = (pr["range"][g] for g in aa_estimate.PRIMARY)       # plan Reading 27
        n = aa_rules._n
        assert f"쌍별 값 범위 보상 {n(a0)}–{n(a1)} · 처벌 {n(p0)}–{n(p1)})" in s


def test_s1_never_changes_with_learning_values(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    fail = LKEYS[0]
    w.model.offset[fail] = 40.0
    w.runner().run("screen")
    w.runner().run("coverage")
    del w.model.offset[fail]                     # after the screen the failing pair's naive level would now pass
    w.model.effects[fail] = (90.0, 90.0)         # and its learning values are extreme
    _learn_job(w, lambda kw, pair, real: real(kw, pair))
    out = w.runner().run("learn")
    assert out["outcome"] == aa_rules.STOP_MACHINE                   # learn pre ≠ naive pre for that pair
    # the pair-selection rule itself: estimate never reruns the filter and S1 = screen passers
    (tmp_path / "b").mkdir()
    w2 = _world(tmp_path / "b", monkeypatch)
    w2.model.offset[fail] = 40.0
    w2.chain("learn")

    def boom(*a, **k):
        raise AssertionError("final_filter after the screen")
    monkeypatch.setattr(y_rules, "final_filter", boom)
    est = w2.runner().run("estimate")
    sc = {p["key"]: p for p in doc()["screen"]["pairs"]}
    assert fail not in [p["key"] for p in est["primary"]["pairs"]]
    assert [p["key"] for p in est["primary"]["pairs"]] == [k for k, p in sc.items() if p["passed"]]
    for p in est["primary"]["pairs"]:
        assert (p["naive_d"], p["L_A"], p["L_P"]) == (sc[p["key"]]["naive_d"], sc[p["key"]]["L_A"],
                                                      sc[p["key"]]["L_P"])


def test_estimate_seal_mismatch_exit7(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("learn")
    real = AR.seal_now
    monkeypatch.setattr(AR, "seal_now", lambda s=w.s: dict(real(s), key="0" * 64))
    with pytest.raises(SystemExit) as e:
        w.runner().run("estimate")
    assert e.value.code == AR.EXIT_SEAL == 7
    assert "estimate" not in doc()


def test_estimate_repeat_mismatch_invalid(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("learn")
    real, n = aa_estimate.estimate_set, dict(i=0)

    def flaky(*a, **k):
        n["i"] += 1
        out = real(*a, **k)
        if n["i"] == 2:
            out["point"] = {g: v + 1e-12 for g, v in out["point"].items()}
        return out
    monkeypatch.setattr(aa_estimate, "estimate_set", flaky)
    out = w.runner().run("estimate")
    assert out["outcome"] == aa_rules.INVALID and out["repeat_equal"] is False
    assert "archive" not in out and "primary" not in out and "sentence" not in out
    assert not Path(w.s.archive_root, "estimate").exists()
    with pytest.raises(SystemExit):
        w.runner().archive("estimate")
    assert n["i"] == 2
