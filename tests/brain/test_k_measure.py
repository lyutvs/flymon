"""Spec K.8.2: per-KC activity measured once per (odour, seed), cached under a content key, shared by duplicate odours."""
import numpy as np
from flymon.brain import k_jobs
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.k_measure import HASHED_FILES, MEASURE_FILES, KMeasurer
from flymon.brain.k_spec import SPEC


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kwargs_list):
        assert fn is k_jobs.activity_job
        self.calls.append(kwargs_list)
        out = []
        for kw in kwargs_list:
            part = []
            for i, odor, seed in kw["items"]:
                k = int(sum(odor.values()) * 10) % 5                    # deterministic fake: KC k fires on even seeds
                part.append(dict(i=i, seed=seed, kc=[k] if seed % 2 == 0 else [], n=[3] if seed % 2 == 0 else [],
                                 max_win=1))
            out.append(part)
        return out


PAIRS = [dict(axis="b", turn=1, x="a", y="b", odor_x={"g1": 0.1}, odor_y={"g2": 0.2}),
         dict(axis="b", turn=3, x="c", y="d", odor_x={"g2": 0.2}, odor_y={"g3": 0.3})]     # odour g2 appears twice


def test_files_extend_js_and_name_k_modules():
    assert "flymon/brain/k_jobs.py" in MEASURE_FILES and "flymon/brain/j_jobs.py" in MEASURE_FILES
    assert "flymon/brain/k_metrics.py" not in MEASURE_FILES and "flymon/brain/k_metrics.py" in HASHED_FILES
    assert {"scripts/run_k_scan.py", "scripts/run_k_judge.py", "flymon/brain/k_rules.py"} <= set(HASHED_FILES)


def test_duplicate_odours_are_measured_once_and_fractions_are_per_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                        # h3_store.guard: writes only under results/m0d/
    code = {"key": "k" * 64}
    pool = FakePool()
    km = KMeasurer(pool, SPEC, PAIRS, 5, MeasureCache("results/m0d/k/cache", code, "r1"))
    assert len(km.odours) == 3 and km.index == [(0, 1), (1, 2)]
    acts = km.activity(Params())
    n_items = sum(len(kw["items"]) for kw in pool.calls[0])
    assert n_items == 3 * len(SPEC.j.h4.act_seeds)
    k1 = int(0.1 * 10) % 5
    assert acts[0]["fx"][k1] == 0.5 and acts[0]["cx"][k1] == 1.5      # 4 of 8 seeds, 3 spikes each
    assert np.array_equal(acts[0]["fy"], acts[1]["fx"])                # the shared odour is one measurement
    KMeasurer(pool, SPEC, PAIRS, 5, MeasureCache("results/m0d/k/cache", code, "r2")).activity(Params())
    assert len(pool.calls) == 1                                        # the rerun is a cache hit (resume)

