"""Y's 0p chain: digest (keys, V reuse through W's _reuse_dec, the main-set digest) then oracle (the count table only in
the block, per-pair values only in results/y/oracle.json); STOP_REUSE, early STOP_FEW_PAIRS, INVALID; refusals; an
interrupted oracle writes no block; archives only for committed blocks (PASS / STOP)."""
import dataclasses
import json
from pathlib import Path

import shutil

import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y

GOOD_Q = dict(edit=V.lever_edit, csc_sha256=V.sha_combined, edit_edges=V.lever_edges)


def _rep(la, lp, st):
    return {"pre": {"A": [[la, 0]] * 8, "P": [[lp, 0]] * 8}, "R1": {}, "R2": {}, "_st": st}


def _st(d, testable=True):
    return dict(d_pre=d, r=3.0, p=-3.0, m=3.0, testable=testable)


# (axis, d_pre, L_A, L_P, testable) — 6 lenient passes by default (Y.9.2 P1-7: [4, 8] stays, [6, 10] drops)
PAIRS = [("b", 0.1, 25, 50, True), ("b", 0.7, 17, 35, True), ("a", 0.2, 40, 100, True), ("a", -0.9, 30, 60, True),
         ("b", 0.0, 22, 44, True), ("a", 3.0, 50, 200, True), ("b", 0.1, 25, 50, False), ("a", 0.3, 18, 40, True)]


class FakeM:
    def __init__(self, pairs, q=GOOD_Q, raise_after=None, none_at=()):
        self.pairs, self.q, self.raise_after, self.none_at, self.calls = pairs, q, raise_after, set(none_at), []

    def oracle(self, rows, cond, block, seeds):
        self.calls.append((cond.name, block, seeds))
        if self.raise_after is not None:
            raise RuntimeError("pool died")
        out = []
        for i, (r, (ax, d, la, lp, t)) in enumerate(zip(rows, self.pairs)):
            p = Path(f"results/y/cache/r_oracle/{i}.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            res = dict(q=dict(self.q), report=_rep(la, lp, None if i in self.none_at else _st(d, t)))
            p.write_text(json.dumps(dict(key=f"k{i}", kind="r_oracle", inputs={}, result=res)))
            out.append(dict(key=f"{ax}|{r['turn']}|x|y", cache_file=str(p), cache_key=f"k{i}", result=res))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch, pairs=PAIRS):
        monkeypatch.chdir(tmp_path)
        self.pairs = pairs
        self.ys = dataclasses.replace(Y, archive_root=str(tmp_path / "archive/y"))
        self.arch = tmp_path / "archive/y"
        self.js = dict(n_b=167, n_a=82, n=249, first_turn=306, last_turn=1967, skipped={}, clusters_b=[],
                       clusters_a=[], digest_keys=Y.main_digest_keys, digest_e0_b="b" * 64, digest_e0_a="a" * 64,
                       n_odours=155, all_off_pool=True, keys=[f"k{i}" for i in range(249)], rows=[])
        self.keys = dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key)
        self.reuse = dict(outcome="PASS", reasons=[])
        self.set_ok = True
        rows = [dict(axis=ax, turn=306 + i, c=i) for i, (ax, *_rest) in enumerate(pairs)]

        def main_rows(blk):
            if not self.set_ok:
                raise ValueError("W set keys differ from block set")
            assert blk["digest_keys"] == Y.main_digest_keys
            return rows
        self.m = FakeM(pairs)
        self.ctx = dict(keys=lambda: dict(self.keys), reuse=lambda: dict(self.reuse), w_set=lambda: dict(self.js),
                        main_rows=main_rows, params=lambda: Params())
        monkeypatch.setattr(YR, "summary_git", lambda p: dict(tracked=True, dirty=False, judged=[]))
        monkeypatch.setattr(YR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(YR, "pair_stats", lambda rep, z, t: rep["_st"])

    def runner(self):
        return YR.Runner(self.ctx, self.ys, measure=lambda: self.m)


def doc():
    return json.loads(Path(Y.summary).read_text())


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_digest_pass_records_the_set(w):
    out = w.runner().stage_digest()
    assert out["outcome"] == R.PASS and out == doc()["digest"]
    assert out["set"]["digest_keys"] == Y.main_digest_keys and len(out["set"]["keys"]) == 249
    assert "rows" not in out["set"] and doc()["budget"]["ledger"][0]["stage"] == "digest"
    assert out["archive"] == [] and not (w.arch / "digest").exists()  # no detail file at digest


@pytest.mark.parametrize("breaker", ["w_key", "u_key", "reuse", "digest", "n_b", "last_turn", "gen"])
def test_digest_stop_reuse(w, breaker):
    if breaker == "w_key":
        w.keys["w_measure_key"] = "x"
    elif breaker == "u_key":
        w.keys["u_measure_key"] = "x"
    elif breaker == "reuse":
        w.reuse = dict(outcome="STOP_REUSE", reasons=["V 커밋 cf0b3b2(set)가 HEAD 이력에 없음"])
    elif breaker == "digest":
        w.js["digest_keys"] = "0" * 64
    elif breaker == "n_b":
        w.js["n_b"] = 166
    elif breaker == "last_turn":
        w.js["last_turn"] = 1985
    else:
        def boom():
            raise ValueError("V's set does not reproduce")
        w.ctx["w_set"] = boom
    out = w.runner().stage_digest()
    assert out["outcome"] == R.STOP_REUSE and out["records_unavailable"] is True
    assert out["sentence"].startswith("Y 재사용 조건(Y.7 0p-b)이 깨졌다(") and out["archive"] == []
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oracle()
    assert e.value.code == 2


def test_oracle_counts_only_in_block_and_values_in_detail(w):
    w.runner().stage_digest()
    out = w.runner().stage_oracle()
    assert out["outcome"] == R.PASS and out == doc()["oracle"]
    assert w.m.calls == [("L", "screen", W.oracle_seeds())]
    det = json.loads(Path(Y.oracle_detail).read_text())
    assert out["derived"] == R.derive(det["pairs"], Y) and out["detail_reused"] is False
    assert out["derived"]["counts"]["all"]["y_lenient"] == 6 and out["derived"]["counts"]["all"]["testable"] == 7
    assert out["derived"]["yield_rule"]["kept"] == [[4, 8]] == out["k_ranges"]
    assert out["derived"]["lenient_levels"]["n"] == 6 and len(out["derived"]["lenient_levels"]["A"]) == 5
    assert out["detail_sha256"] == sha256_file(Y.oracle_detail) and len(det["manifest"]) == len(PAIRS)
    text = json.dumps(out)
    for k in ("d_pre", "L_A", "L_P", '"pairs"', "manifest", "cache_file"):
        assert k not in text, k
    assert det["pairs"][0]["L_A"] == 25.0 and det["pairs"][0]["d_pre"] == 0.1
    for rec in det["pairs"]:
        assert {"axis", "value", "failure", "testable", "d_pre", "L_A", "L_P"} <= set(rec)
    assert [e["sha256"] for e in out["archive"]] == [out["detail_sha256"]]
    assert sha256_file(w.arch / "oracle/y/oracle.json") == out["detail_sha256"]


@pytest.mark.parametrize("n_len,rule,text", [(3, "early", "통과한 쌍이 3개로 최소 관문 쌍 수 4에 못 미쳤다"),
                                           (5, "yield", "통과한 쌍이 5개로, 수율 규칙(Y.9.2 P1-7, c 1.5)")])
def test_oracle_early_stop_few_pairs(tmp_path, monkeypatch, n_len, rule, text):
    w = World(tmp_path, monkeypatch, pairs=PAIRS[:n_len] + PAIRS[5:7])
    w.runner().stage_digest()
    out = w.runner().stage_oracle()
    assert out["outcome"] == R.STOP_FEW_PAIRS and out["stop_stage"] == "early" and out["stage"] == "oracle"
    assert out["rule"] == rule and text in out["sentence"] and out["derived"]["yield_rule"]["kept"] == []
    assert (w.arch / "oracle/y/oracle.json").exists() and len(out["archive"]) == 1     # a STOP is archived


def test_oracle_lever_mismatch_is_invalid_and_no_value_is_counted(w):
    w.runner().stage_digest()
    w.m = FakeM(PAIRS, none_at={5})                                    # a non-lenient pair: still 6
    out = w.runner().stage_oracle()
    assert out["derived"]["counts"]["all"]["no_value"] == 1 and out["outcome"] == R.PASS
    shutil.rmtree(w.arch / "oracle")                                  # the PASS run's archive (test only)
    Path(Y.oracle_detail).unlink()                                    # else the cached detail is re-derived
    Path(Y.summary).write_text(json.dumps({k: v for k, v in doc().items() if k != "oracle"}))
    w.m = FakeM(PAIRS, q=dict(GOOD_Q, edit_edges=-1))
    out = w.runner().stage_oracle()
    assert out["outcome"] == R.INVALID and out["derived"]["counts"]["all"]["failures"] == len(PAIRS)
    assert "archive" not in out and not (w.arch / "oracle").exists()   # INVALID is never archived


def test_interrupted_oracle_writes_no_block_and_keeps_wall(w):
    w.runner().stage_digest()
    w.m = FakeM(PAIRS, raise_after=0)
    with pytest.raises(RuntimeError):
        w.runner().stage_oracle()
    assert "oracle" not in doc() and not Path(Y.oracle_detail).exists() and not w.arch.joinpath("oracle").exists()
    prog = json.loads(Path(Y.progress_dir, "oracle.json").read_text())["wall_s"]
    assert prog >= 0.0
    w.m = FakeM(PAIRS)
    assert w.runner().stage_oracle()["outcome"] == R.PASS
    assert json.loads(Path(Y.progress_dir, "oracle.json").read_text())["wall_s"] >= prog
    assert doc()["budget"]["ledger"][-1]["stage"] == "oracle"


def test_refusals(w, monkeypatch):
    with pytest.raises(SystemExit):
        w.runner().stage_oracle()                                  # no digest
    w.runner().stage_digest()
    with pytest.raises(SystemExit):
        w.runner().stage_digest()                                  # block exists
    w.set_ok = False
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oracle()                                  # set no longer reproduces block digest
    assert e.value.code == 2 and "oracle" not in doc()
    w.set_ok = True
    monkeypatch.setattr(YR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/w_pairs.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        w.runner().stage_oracle()


def test_hashed_files_cover_y_and_the_w_code_it_imports():
    for f in ("flymon/brain/y_spec.py", "flymon/brain/y_rules.py", "flymon/brain/y_store.py",
              "flymon/brain/y_runner.py", "scripts/run_y.py", "flymon/brain/w_pairs.py", "flymon/brain/w_runner.py",
              "flymon/brain/w_measure.py", "results/summary/v_lever.json"):
        assert f in YR.Y_HASHED_FILES, f


def test_oracle_refuses_when_the_measurement_keys_moved_after_digest(w):
    w.runner().stage_digest()
    w.keys["w_measure_key"] = "x"
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oracle()
    assert e.value.code == 2 and "oracle" not in doc() and w.m.calls == []


def test_existing_different_archive_refuses_before_the_block(w):
    w.runner().stage_digest()
    (w.arch / "oracle/y").mkdir(parents=True)
    (w.arch / "oracle/y/oracle.json").write_text("other")
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oracle()
    assert e.value.code == 2 and "oracle" not in doc()


def _drop_block(stage):
    Path(Y.summary).write_text(json.dumps({k: v for k, v in doc().items() if k != stage}))


def test_oracle_rederives_from_an_existing_oracle_json_without_measuring(w):
    """Y.9.2 P2-11: only oracle.json is bit-bound; with oracle.json matching the cache the stage re-derives the block
    (count table, yield rule, quantiles) and never calls the measurer; the archive repeat is idempotent."""
    w.runner().stage_digest()
    first = w.runner().stage_oracle()
    sha = sha256_file(Y.oracle_detail)
    _drop_block("oracle")
    w.m = FakeM(PAIRS, raise_after=0)                                 # measuring would raise
    out = w.runner().stage_oracle()
    assert out["detail_reused"] is True and w.m.calls == [] and out["replaced_detail_sha256"] is None
    assert out["detail_sha256"] == sha == sha256_file(Y.oracle_detail)
    assert out["derived"] == first["derived"] and out["archive"] == first["archive"]


@pytest.mark.parametrize("breaker", ["cache_sha", "record", "rows"])
def test_oracle_measures_when_oracle_json_does_not_match_the_cache(w, breaker):
    w.runner().stage_digest()
    w.runner().stage_oracle()
    _drop_block("oracle")
    shutil.rmtree(w.arch / "oracle")
    det = json.loads(Path(Y.oracle_detail).read_text())
    if breaker == "cache_sha":
        Path(det["manifest"][0]["cache_file"]).write_text("{}")
    elif breaker == "record":
        det["pairs"][0]["d_pre"] = 0.11
        Path(Y.oracle_detail).write_text(json.dumps(det))
    else:
        det["pairs"], det["manifest"] = det["pairs"][:-1], det["manifest"][:-1]
        Path(Y.oracle_detail).write_text(json.dumps(det))
    old = sha256_file(Y.oracle_detail)
    w.m = FakeM(PAIRS)
    out = w.runner().stage_oracle()
    assert out["detail_reused"] is False and len(w.m.calls) == 1 and out["outcome"] == R.PASS
    if breaker == "cache_sha":                                        # re-measured to the same bits: nothing replaced
        assert out["replaced_detail_sha256"] is None and out["detail_sha256"] == old
    else:
        assert out["replaced_detail_sha256"] == old != out["detail_sha256"]


def test_runner_archive_repeats_after_commit_and_refuses_without_a_block(w):
    with pytest.raises(SystemExit):
        w.runner().archive("oracle")                                  # no block
    w.runner().stage_digest()
    assert w.runner().archive("digest") == []
    out = w.runner().stage_oracle()
    assert w.runner().archive("oracle") == out["archive"] == w.runner().archive("oracle")
    _drop_block("oracle")
    Path(Y.oracle_detail).unlink()
    w.m = FakeM(PAIRS, q=dict(GOOD_Q, edit_edges=-1))
    shutil.rmtree(w.arch / "oracle")
    assert w.runner().stage_oracle()["outcome"] == R.INVALID
    with pytest.raises(SystemExit) as e:
        w.runner().archive("oracle")                                  # INVALID is never archived
    assert e.value.code == 2 and not (w.arch / "oracle").exists()
