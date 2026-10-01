# tests/brain/test_o_measure.py
"""Spec O.7.5: one atomic cache file per (O1, condition, cell, seed) and per (O2, X, arm, seed), each carrying the code
key and the spec commit; a rerun resumes without a pool and an interrupted one computes only the missing items; O writes
only under results/o/ and results/summary/o_states.json; rounds of one item per worker."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flymon.brain import n_measure, o_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_measure import (HASHED_FILES, MEASURE_FILES, OCache, OMeasurer, write_json,
                                    write_summary_block)
from flymon.brain.o_spec import SPEC

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
    h4 = replace(N_SPEC.h4, teach_trials=1, teach_present_ms=100.0, teach_gap_ms=20.0,
                 teach_window=replace(N_SPEC.h4.teach_window, settle_ms=50.0),
                 oracle_window=replace(N_SPEC.h4.oracle_window, settle_ms=30.0, read_ms=50.0))
    return replace(SPEC, n=replace(N_SPEC, l=replace(N_SPEC.l, j=replace(N_SPEC.l.j, h4=h4))))


@pytest.fixture
def world(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome(disjoint_kc=True)
    o_jobs._RIG.clear()
    return c, Populations.from_connectome(c)


def _o1_items():
    return [dict(cond=cond, edit=edit, g=1.0, stim="X", seed=s, odor=X)
            for cond, edit in (("on", "none"), ("kc", "apl_to_kc_zero")) for s in (5, 6)]


def _o2_items():
    return [dict(x="X", y="Y", edit="none", arm=a, punish=pu, plastic=pl, da_zero=dz, odor_x=X, odor_y=Y, seed=5)
            for a, pu, pl, dz in SPEC.o2_arms]


def test_o1_one_file_per_item_with_code_key_and_spec_commit_and_resume(world):
    c, pops = world
    pool = InProcessPool(c, pops)
    rows = OMeasurer(pool, _spec(), OCache("results/o/cache", CODE, "run1", "c0ffee")).o1_presentations(
        P, _o1_items(), RO)
    assert [(r["cond"], r["seed"]) for r in rows] == [("on", 5), ("on", 6), ("kc", 5), ("kc", 6)]
    assert pool.rounds == [3, 1]
    files = sorted(Path("results/o/cache/o1_pres").glob("*.json"))
    assert len(files) == 4
    d = json.loads(files[0].read_text())
    assert d["spec_commit"] == "c0ffee" and d["code_key"] == "k" * 64 and d["inputs"]["stage"] == "o1"
    again = OMeasurer(NoPool(), _spec(), OCache("results/o/cache", CODE, "run2", "c0ffee")).o1_presentations(
        P, _o1_items(), RO)
    assert json.dumps(again, sort_keys=True) == json.dumps(json.loads(json.dumps(rows)), sort_keys=True)


def test_an_interrupted_run_computes_only_the_missing_items(world):
    c, pops = world
    OMeasurer(InProcessPool(c, pops), _spec(), OCache("results/o/cache", CODE, "run1", "x")).o1_presentations(
        P, _o1_items()[:2], RO)
    pool = InProcessPool(c, pops)
    OMeasurer(pool, _spec(), OCache("results/o/cache", CODE, "run2", "x")).o1_presentations(P, _o1_items(), RO)
    assert pool.jobs == 2


def test_o2_one_file_per_arm_and_resume(world):
    c, pops = world
    rows = OMeasurer(InProcessPool(c, pops), _spec(), OCache("results/o/cache", CODE, "r", "x")).o2_arms(
        P, _o2_items(), RO, "PPL105")
    assert [r["arm"] for r in rows] == [a for a, *_ in SPEC.o2_arms] and {r["x"] for r in rows} == {"X"}
    assert len(list(Path("results/o/cache/o2_arm").glob("*.json"))) == 4        # the arms never share an entry
    again = OMeasurer(NoPool(), _spec(), OCache("results/o/cache", CODE, "r2", "x")).o2_arms(
        P, _o2_items(), RO, "PPL105")
    assert [r["w_post_sha256"] for r in again] == [r["w_post_sha256"] for r in rows]


def test_writes_only_under_results_o(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(Path("results/o/x/a.json"), {"a": 1}, [P])
    write_summary_block(Path("results/summary/o_states.json"), "o1", {"outcome": "JUDGED"}, [P])
    assert json.loads(Path("results/summary/o_states.json").read_text()) == {"o1": {"outcome": "JUDGED"}}
    for bad in ("results/n/x.json", "results/summary/n_real_odour.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            write_json(Path(bad), {}, [P])
        assert e.value.code == 2


def test_file_lists_extend_ns():
    assert set(n_measure.MEASURE_FILES) <= set(MEASURE_FILES)
    assert {"flymon/brain/o_jobs.py", "flymon/brain/o_measure.py"} <= set(MEASURE_FILES)
    assert set(MEASURE_FILES) <= set(HASHED_FILES)
    assert set(n_measure.HASHED_FILES) <= set(HASHED_FILES)
    assert {"flymon/brain/o_spec.py", "flymon/brain/o_rules.py", "flymon/brain/o_cli.py", "scripts/run_o1.py",
            "scripts/run_o2.py", "flymon/brain/n_rules.py"} <= set(HASHED_FILES)


def test_reward_type_is_part_of_the_o2_cache_key(world):
    c, pops = world
    root = "results/o/cache"
    OMeasurer(InProcessPool(c, pops), _spec(), OCache(root, CODE, "r", "x")).o2_arms(P, _o2_items()[:1], RO, "PPL105")
    d = json.loads(next(Path(root, "o2_arm").glob("*.json")).read_text())
    assert d["inputs"]["reward_type"] == _spec().n.h3.reward_type
