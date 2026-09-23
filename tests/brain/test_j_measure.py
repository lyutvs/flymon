"""Spec J.10.8 / J.11.6: J's measurements are cached by content and resumable; the key covers the depression."""
import dataclasses

from flymon.brain import d6a
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.j_measure import HASHED_FILES, MEASURE_FILES, JMeasurer
from flymon.brain.j_params import with_std
from flymon.brain.j_spec import SPEC


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, jobs):
        self.calls.append((fn.__name__, len(jobs)))
        if fn.__name__ == "all51_uni_job":
            return [[dict(g=g, seed=s, kc=1, kc_on=1, pn=1, uni_n=1, uni_hz=1.0, orn_hz=1.0, std_r=None)
                     for g, s in j["items"]] for j in jobs]
        return [dict(seed=j["seed"], max_win=[1] * len(j["odors"]), kc_active_frac=[0.05] * len(j["odors"]),
                     kc_spikes=[10] * len(j["odors"])) for j in jobs]


def _m(tmp_path, pool):
    return JMeasurer(pool, SPEC, (["G1", "G2", "G3"], 34.0), MeasureCache("results/m0d/j/std/cache", dict(key="k"), "run"))


def test_the_key_files_cover_the_engine_the_jobs_and_d6_and_the_procedure_is_hashed_only():
    for f in ("flymon/brain/engine_cpu.py", "flymon/brain/config.py", "flymon/brain/j_params.py",
              "flymon/brain/j_jobs.py", "flymon/brain/d6a.py", "flymon/brain/h4_jobs.py", "flymon/brain/h3_jobs.py"):
        assert f in MEASURE_FILES
    for f in ("flymon/brain/j_rules.py", "flymon/brain/j_runner.py", "flymon/brain/j_setup.py", "flymon/brain/h3_c3.py",
              "scripts/run_j_std_scan.py", "scripts/run_j_std_judge.py"):
        assert f not in MEASURE_FILES and f in HASHED_FILES


def test_all51_uni_is_cached_per_params_and_the_depression_is_part_of_the_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = _m(tmp_path, pool)
    base = Params()
    rows = m.all51_uni(base)
    assert len(rows) == 3 * len(SPEC.h4.h3.all51_seeds)
    m.all51_uni(base)
    assert pool.calls == [("all51_uni_job", 2)]                              # the second call is a cache hit
    m.all51_uni(with_std(base, 0.9, 300.0, 2.0))
    m.all51_uni(with_std(base, 0.9, 300.0, 2.5))
    assert len(pool.calls) == 3


def test_d6_is_cached_per_seed_block_so_a_resume_loses_at_most_one_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = _m(tmp_path, pool)
    by_turn = {0: [{"ORN_A": 1.0}, {"ORN_B": 1.0}], 2: [{"ORN_C": 1.0}, {"ORN_D": 1.0}]}
    rows = m.d6(Params(), by_turn, d6a.SEEDS[:16])
    assert [(r["turn"], r["seed"]) for r in rows][:3] == [(0, 400), (0, 401), (0, 402)] and len(rows) == 32
    assert pool.calls == [("d6_job", 16), ("d6_job", 16)]
    again = _m(tmp_path, FakePool())
    assert again.d6(Params(), by_turn, d6a.SEEDS[:16]) == rows and again.pool.calls == []
