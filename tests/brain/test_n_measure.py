"""Spec N.8.7: one cache entry per (condition, seed) arm (the judge's checkpoint), resumable without a pool; APL-on and
APL-block items never share an entry; caches written only under results/n/; rounds of one item per worker."""
import json
from pathlib import Path

import pytest

from flymon.brain import n_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT
from flymon.brain.n_measure import HASHED_FILES, MEASURE_FILES, NCache, NMeasurer
from flymon.brain.n_spec import SPEC

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}
X, Y = {"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


class InProcessPool:
    """Runs the real job in this process on the synthetic connectome, recording the round sizes."""
    n_workers = 3

    def __init__(self, conn, pops):
        self.conn, self.pops, self.rounds = conn, pops, []

    def run_jobs(self, fn, jobs):
        self.rounds.append(len(jobs))
        return [fn(_Stub(self.conn), None, self.pops, None, None, **j) for j in jobs]


class NoPool:
    n_workers = 3

    def run_jobs(self, fn, jobs):
        raise AssertionError("a resumed run must not touch the pool")


def _spec():
    from dataclasses import replace
    h4 = replace(SPEC.h4, teach_trials=1, teach_present_ms=100.0, teach_gap_ms=20.0,
                 teach_window=replace(SPEC.h4.teach_window, settle_ms=50.0),
                 oracle_window=replace(SPEC.h4.oracle_window, settle_ms=30.0, read_ms=50.0))
    return replace(SPEC, l=replace(SPEC.l, j=replace(SPEC.l.j, h4=h4)))


@pytest.fixture
def world(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome(disjoint_kc=True)
    n_jobs._RIG.clear()
    return c, Populations.from_connectome(c)


def _items():
    return [dict(cond=cond, edit=edit, odor_x=X, odor_y=Y, seed=s, plastic=True)
            for cond, edit in (("sim_on", "none"), ("sim_off", "apl_to_kc_zero")) for s in (5, 6)]


def test_one_entry_per_condition_and_seed_and_a_rerun_resumes(world):
    c, pops = world
    code = {"key": "k" * 64}
    pool = InProcessPool(c, pops)
    m = NMeasurer(pool, _spec(), NCache("results/n/cache", code, "run1"))
    rows = m.arms(P, _items(), RO, "PPL105")
    assert [r["cond"] for r in rows] == ["sim_on", "sim_on", "sim_off", "sim_off"]
    assert pool.rounds == [3, 1]                                   # rounds of one item per worker
    files = sorted(Path("results/n/cache/n_arm").glob("*.json"))
    assert len(files) == 4                                         # on and block never share an entry
    again = NMeasurer(NoPool(), _spec(), NCache("results/n/cache", code, "run2")).arms(P, _items(), RO, "PPL105")
    assert [json.dumps(r, sort_keys=True) for r in again] == [json.dumps(json.loads(json.dumps(r)), sort_keys=True)
                                                               for r in rows]


def test_presentations_and_kc_vectors_are_cached_per_edit_and_odour(world):
    c, pops = world
    m = NMeasurer(InProcessPool(c, pops), _spec(), NCache("results/n/cache", {"key": "k" * 64}, "r"))
    rows = m.presentations(P, [("none", X), ("apl_to_kc_zero", X)], (5, 6), RO)
    assert [len(r) for r in rows] == [2, 2] and rows[0][0]["edit"] == "none" and "kc_fired" not in rows[1][0]
    kcv = m.kc_vectors(P, [("none", X)], (5,), RO)
    assert "kc_fired" in kcv[0][0]
    assert len(list(Path("results/n/cache/n_pres").glob("*.json"))) == 2
    assert m.params_seen == [P]


def test_punish_oracle_is_one_entry_per_pair(world):
    c, pops = world
    from dataclasses import replace
    s = replace(_spec(), act_seeds=(1, 2), select_seeds=(3, 4), report_seeds=(5, 6))
    m = NMeasurer(InProcessPool(c, pops), s, NCache("results/n/cache", {"key": "k" * 64}, "r"))
    out = m.punish_oracle(P, [dict(name="sim", odor_x=X, odor_y=Y)], RO, {"A": (1.0, 2.0), "P": (1.0, 2.0)}, "PPL105")
    assert out[0]["alpha_punish"] in SPEC.h4.oracle_alphas
    assert len(list(Path("results/n/cache/n_oracle").glob("*.json"))) == 1


def test_the_cache_refuses_paths_outside_results_n(world):
    c, pops = world
    m = NMeasurer(InProcessPool(c, pops), _spec(), NCache("results/m0d/n/cache", {"key": "k" * 64}, "r"))
    with pytest.raises(SystemExit):
        m.presentations(P, [("none", X)], (5,), RO)


def test_the_file_keys_cover_the_data_and_every_n_file():
    assert "data/odor/hallem2006_subset.csv" in MEASURE_FILES and "flymon/brain/n_jobs.py" in MEASURE_FILES
    for f in ("flymon/brain/n_rules.py", "flymon/brain/n_oc.py", "flymon/brain/n_cli.py", "scripts/run_n2_judge.py"):
        assert f in HASHED_FILES
    assert set(MEASURE_FILES) <= set(HASHED_FILES)
    assert all((ROOT / f).exists() for f in MEASURE_FILES)
