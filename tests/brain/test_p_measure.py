# tests/brain/test_p_measure.py
"""Spec P.4 / P.6.5: one atomic cache file per (P, direction, arm, seed) with the code key and the spec commit; rows carry
direction, X, Y and point; a rerun resumes without a pool and an interrupted run computes only the missing items; P
writes only under results/p/ and results/summary/p_learning.json (never O's or N's paths)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flymon.brain import o_jobs, o_measure
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.p_measure import (HASHED_FILES, MEASURE_FILES, PCache, PMeasurer, write_json,
                                    write_summary_block)
from flymon.brain.p_spec import SPEC

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}
X, Y = {"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}
CODE = {"key": "k" * 64}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


class InProcessPool:
    n_workers = 3

    def __init__(self, conn, pops):
        self.conn, self.pops, self.rounds, self.jobs = conn, pops, [], 0

    def run_jobs(self, fn, jobs):
        self.rounds.append(len(jobs)); self.jobs += len(jobs)
        return [fn(_Stub(self.conn), None, self.pops, None, None, **j) for j in jobs]


class NoPool:
    n_workers = 3

    def run_jobs(self, fn, jobs):
        raise AssertionError("a resumed run must not touch the pool")


def _spec():
    n = SPEC.o.n
    h4 = replace(n.h4, teach_trials=1, teach_present_ms=100.0, teach_gap_ms=20.0,
                 teach_window=replace(n.h4.teach_window, settle_ms=50.0),
                 oracle_window=replace(n.h4.oracle_window, settle_ms=30.0, read_ms=50.0))
    return replace(SPEC, o=replace(SPEC.o, n=replace(n, l=replace(n.l, j=replace(n.l.j, h4=h4)))))


@pytest.fixture
def world(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome(disjoint_kc=True)
    o_jobs._RIG.clear()
    return c, Populations.from_connectome(c)


def _items(seeds=(5,)):
    return [dict(direction=d, x="X" if d == "r1" else "Y", y="Y" if d == "r1" else "X", edit="none", arm=a, punish=pu,
                 plastic=pl, da_zero=dz, odor_x=X if d == "r1" else Y, odor_y=Y if d == "r1" else X, seed=s,
                 point=(0.25, 8.0))
            for d in SPEC.directions for a, pu, pl, dz in SPEC.o.o2_arms for s in seeds]


def test_one_file_per_arm_with_code_key_spec_commit_and_resume(world):
    c, pops = world
    pool = InProcessPool(c, pops)
    rows = PMeasurer(pool, _spec(), PCache("results/p/cache", CODE, "r1", "c0ffee")).p_arms(P, _items(), RO, "PPL105")
    assert [(r["direction"], r["arm"]) for r in rows] == [(d, a) for d in SPEC.directions for a in SPEC.arms]
    assert all(r["point"] == [0.25, 8.0] for r in rows) and pool.rounds == [3, 3]
    files = sorted(Path("results/p/cache/p_arm").glob("*.json"))
    assert len(files) == 6
    d = json.loads(files[0].read_text())
    assert d["spec_commit"] == "c0ffee" and d["code_key"] == "k" * 64 and d["inputs"]["stage"] == "p"
    assert d["inputs"]["reward_type"] == N_SPEC.h3.reward_type and "point" not in d["result"]
    frozen = [r for r in rows if r["arm"] == "frozen"]
    assert all(r["w_post_sha256"] == r["w0_sha256"] for r in frozen)
    again = PMeasurer(NoPool(), _spec(), PCache("results/p/cache", CODE, "r2", "c0ffee")).p_arms(
        P, _items(), RO, "PPL105")
    assert [r["w_post_sha256"] for r in again] == [r["w_post_sha256"] for r in rows]


def test_an_interrupted_run_computes_only_the_missing_items(world):
    c, pops = world
    PMeasurer(InProcessPool(c, pops), _spec(), PCache("results/p/cache", CODE, "r1", "x")).p_arms(
        P, _items()[:2], RO, "PPL105")
    pool = InProcessPool(c, pops)
    PMeasurer(pool, _spec(), PCache("results/p/cache", CODE, "r2", "x")).p_arms(P, _items(), RO, "PPL105")
    assert pool.jobs == 4


def test_writes_only_under_results_p(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(Path("results/p/x/a.json"), {"a": 1}, [P])
    write_summary_block(Path("results/summary/p_learning.json"), "p", {"outcome": "JUDGED"}, [P])
    assert json.loads(Path("results/summary/p_learning.json").read_text()) == {"p": {"outcome": "JUDGED"}}
    for bad in ("results/o/x.json", o_measure.SUMMARY, "results/summary/n_real_odour.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            write_json(Path(bad), {}, [P])
        assert e.value.code == 2


def test_file_lists_extend_os():
    assert set(o_measure.MEASURE_FILES) <= set(MEASURE_FILES) and "flymon/brain/p_measure.py" in MEASURE_FILES
    assert set(MEASURE_FILES) <= set(HASHED_FILES) and set(o_measure.HASHED_FILES) <= set(HASHED_FILES)
    assert {"flymon/brain/p_spec.py", "flymon/brain/p_rules.py", "flymon/brain/p_cli.py", "scripts/run_p.py",
            "scripts/p_oc.py", "flymon/brain/o_rules.py", "flymon/brain/n_rules.py"} <= set(HASHED_FILES)
