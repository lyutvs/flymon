"""The qualification and primary CLIs: refusals, the naive / oracle calls the qualification makes, and the primary
driver over qualified pairs on the fake pool (never the real connectome)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.config import Params
from flymon.rescope import primary, rules
from flymon.rescope.spec import SPEC

from .test_primary import CELLS, ODORS, SMALL, fake

ROOT = Path(__file__).resolve().parents[2]
QUIET = lambda s: None


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_primary_refuses_without_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])


def test_primary_refuses_on_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": "STOP_FEW_PAIRS", "control_qualified": True, "qualified": []}))
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])


def test_primary_refuses_on_unqualified_control(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": None, "control_qualified": False, "qualified": ["p1000"] * 4}))
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])


def test_primary_smoke_reads_the_smoke_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": None, "control_qualified": True, "qualified": ["p1000"] * 4}))
    with pytest.raises(SystemExit):                     # the real one exists, the smoke one does not
        load("run_rescope_primary").main(["--smoke", "--allow-dirty"])


def test_qualify_refuses_outside_rescope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load("run_rescope_qualify").main(["--out", "results/b/x", "--allow-dirty"])


def test_smoke_paths():
    q = load("run_rescope_qualify")
    out, summ = q.paths("results/rescope/qualify", smoke=True, stage="qualify")
    assert str(out).startswith("results/rescope-smoke/") and str(summ) == "results/rescope-smoke/summary/rescope_qualify.json"
    out, summ = q.paths("results/rescope/qualify", smoke=False, stage="qualify")
    assert str(out) == "results/rescope/qualify" and str(summ) == "results/summary/rescope_qualify.json"


class NaivePool:
    n_flies = 3

    def __init__(self):
        self.calls = []

    def decide_batch(self, reqs, strength, settle_ms, read_ms, idx):
        self.calls.append(dict(reqs=reqs, strength=strength, settle_ms=settle_ms, read_ms=read_ms, idx=idx))
        return [np.array([[10 + s % 7, 1], [20, 2]]) for _, _, s in reqs]      # a: 11+s%7, b: 22


class OraclePool:
    def __init__(self, good):
        self.good, self.calls = good, []

    def run_jobs(self, fn, kwargs_list):
        self.calls.append((fn, kwargs_list))
        pre = {"A": [[10, 10]] * 8, "P": [[30, 30]] * 8}
        r1 = {"A": [[10, 10]] * 8, "P": [[30 - 12 - (i % 2), 30] for i in range(8)]}
        return [dict(alpha_reward=0.5, report={"pre": pre, "R1": r1 if self.good(kw) else pre}) for kw in kwargs_list]


def test_naive_covers_every_select_and_report_seed():
    q = load("run_rescope_qualify")
    pool = NaivePool()
    idx = np.array([5, 6])
    nv = q.naive_counts(pool, SPEC, "p1001", ODORS, idx)
    seeds = SPEC.qual_seeds("p1001")
    assert len(nv["a"]) == len(nv["b"]) == len(seeds["select"]) + len(seeds["report"])
    got = [s for c in pool.calls for _, _, s in c["reqs"]]
    assert got == seeds["select"] + seeds["report"]
    assert all(cands == [ODORS["a"], ODORS["b"]] for c in pool.calls for _, cands, _ in c["reqs"])
    assert all(c["strength"] == SPEC.strength and c["settle_ms"] == SPEC.probe_settle_ms
               and c["read_ms"] == SPEC.probe_read_ms and np.array_equal(c["idx"], idx) for c in pool.calls)
    assert nv["a"] == [11 + s % 7 for s in got] and nv["b"] == [22] * len(got)


def test_measure_passes_c3_z_and_qualifies():
    from flymon.rescope.oracle import reward_oracle_job
    q = load("run_rescope_qualify")
    names = SPEC.pair_names()
    odors_of = {n: {"a": {f"ORN_{n}_a": 1.0}, "b": {f"ORN_{n}_b": 1.0}} for n in names}
    bad = {"p1004", "p1005"}
    oracle = OraclePool(good=lambda kw: not any(n in next(iter(kw["odor_x"])) for n in bad))
    params = Params()
    pairs = q.measure(NaivePool(), oracle, SPEC, params, odors_of, np.array([1]), log=QUIET)
    fn, kws = oracle.calls[0]
    assert fn is reward_oracle_job and len(kws) == len(names)
    for n, kw in zip(names, kws):
        s = SPEC.qual_seeds(n)
        x = pairs[n]["x"]
        assert x == "b"                                   # naive b (22) > a (<= 17)
        assert kw["odor_x"] == odors_of[n][x] and kw["odor_y"] == odors_of[n]["a"]
        assert kw["z"] == {"A": SPEC.z_a, "P": SPEC.z_p} and kw["readout"] == {"A": SPEC.a_type, "P": SPEC.p_type}
        assert kw["params"] is params and kw["reward_type"] == SPEC.reward_dan
        assert list(kw["act_seeds"]) == s["act"] and list(kw["select_seeds"]) == s["select"]
        assert list(kw["report_seeds"]) == s["report"] and tuple(kw["alphas"]) == SPEC.oracle_alphas
        assert len(pairs[n]["naive"]["a"]) == len(s["select"]) + len(s["report"])
        assert pairs[n]["alpha_reward"] == 0.5
    summ = q.summarize(SPEC, pairs)
    assert summ["qualified"] == ["p1000", "p1001", "p1002", "p1003"] and summ["m"] == 4 and summ["stop"] is None
    assert summ["control_qualified"] is True
    pairs["p1003"]["qualified"] = False
    assert q.summarize(SPEC, pairs)["stop"] == "STOP_FEW_PAIRS"


def test_primary_driver_runs_control_then_qualified_and_resets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = load("run_rescope_primary")
    qual = {"qualified": ["p1000", "p1001"], "pairs": {"seed0": {"x": "a"}, "p1000": {"x": "b"}, "p1001": {"x": "b"}}}
    pool = fake()
    res = p.run_all(pool, SMALL, qual, {n: ODORS for n in SMALL.pair_names()}, CELLS, Path("results/rescope/primary"),
                    "abc", [Params()], {"commit": "abc"}, log=QUIET)
    assert set(res["pairs"]) == {"p1000", "p1001"} and res["control"]["pair"] == "seed0"
    ref = primary.run_pair(fake(), SMALL, "p1001", ODORS, CELLS, log=QUIET)
    raw = json.loads(Path("results/rescope/primary/p1001/records.json").read_text())
    assert raw["records"] == ref["records"]                # the pool was reset to naive between pairs
    assert res["overall"] == rules.overall(res["control"], res["pairs"], SMALL)
    rec = res["recorded"]
    for n in ("seed0", "p1000", "p1001"):
        assert set(rec[n]) == {"level", "sign", "naive", "qual_x_vs_x"}
        assert rec[n]["qual_x_vs_x"] == (qual["pairs"][n]["x"] == raw["x"] if n == "p1001" else rec[n]["qual_x_vs_x"])
    assert Path("results/rescope/primary/seed0/records.json").exists()
