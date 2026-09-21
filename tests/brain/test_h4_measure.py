"""The H.4 measurement layer: teach and the oracle through the content-addressed cache (one entry per oracle pair, run
in rounds of one pair per worker), reference and rest delegated to H.3's measurer."""
import pytest

from flymon.brain import h4_jobs
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_spec import SPEC

PAIRS = [dict(axis=ax, turn=2 * j, x=f"{ax}x{j}", y=f"{ax}y{j}", odor_x={"ORN_A": 1.0}, odor_y={"ORN_B": 1.0 + j})
         for ax in ("a", "b") for j in range(3)]
POOLS = {"A": ["MA1"], "P": ["MP1"]}


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kwargs_list):
        self.calls.append((fn.__name__, len(kwargs_list)))
        return [dict(tag=fn.__name__, b=kw.get("odor_y", {}).get("ORN_B"), seed=kw.get("seed"), x="job's own x")
                for kw in kwargs_list]


class FakeH3:
    def __init__(self):
        self.calls = []

    def reference(self, params):
        self.calls.append("reference"); return ["ref"]

    def rest(self, params, seeds):
        self.calls.append(("rest", tuple(seeds))); return ["rest"]


@pytest.fixture
def m(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    cache = MeasureCache("results/m0d/h4/cache", dict(key="k" * 64), "run")
    return H4Measurer(pool, SPEC, PAIRS, POOLS, cache, FakeH3())


def test_the_oracle_runs_missing_pairs_in_rounds_and_resumes_per_pair(m, tmp_path):
    p = Params(kc_thresh=1.6)
    z = {"A": (1.0, 2.0), "P": (3.0, 4.0)}
    rows = m.oracle(p, {"A": "MA1", "P": "MP1"}, z)
    assert m.pool.calls == [("oracle_job", 2), ("oracle_job", 2), ("oracle_job", 2)]
    assert [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows] == [(q["axis"], q["turn"], q["x"], q["y"]) for q in PAIRS]
    assert [r["b"] for r in rows] == [q["odor_y"]["ORN_B"] for q in PAIRS] and rows[0]["tag"] == "oracle_job"
    files = sorted((tmp_path / "results/m0d/h4/cache/oracle").glob("*.json"))
    assert len(files) == len(PAIRS) and m.cache.misses == len(PAIRS)
    m.pool.calls.clear()
    assert m.oracle(p, {"A": "MA1", "P": "MP1"}, z) == rows and m.pool.calls == []
    files[0].unlink()
    assert m.oracle(p, {"A": "MA1", "P": "MP1"}, z) == rows and m.pool.calls == [("oracle_job", 1)]
    assert m.oracle(p, {"A": "MA1", "P": "MP1"}, {"A": (1.0, 2.5), "P": (3.0, 4.0)}) != [] and \
        m.pool.calls[-3:] == [("oracle_job", 2)] * 3                   # other z constants are another measurement


def test_teach_runs_every_arm_order_and_seed_once_and_is_cached(m):
    p = Params()
    rows = m.teach(p)
    assert m.pool.calls == [("teach_job", 2 * 2 * len(SPEC.teach_seeds))]
    assert len(rows) == 32 and m.teach(p) == rows and len(m.pool.calls) == 1


def test_reference_and_rest_are_h3s_measurements(m):
    p = Params(kc_thresh=1.7)
    assert m.reference(p) == ["ref"] and m.rest(p, [5, 6]) == ["rest"]
    assert m.h3.calls == ["reference", ("rest", (5, 6))] and m.params_seen == [p]
