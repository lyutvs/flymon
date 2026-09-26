"""Spec L.11.1 / L.11.3: pinned inputs are refused on a hash / engine / seed / order mismatch; the naive job is
oracle_job's first half; the naive measurer is cached per pair and runs in rounds of one pair per worker."""
import dataclasses
import hashlib
import json

import numpy as np
import pytest

from flymon.brain import h4_jobs as J
from flymon.brain import l_measure as lm
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.l_spec import SPEC

C3 = {"kc_thresh": 1.65}


def _spec(**kw):
    """SPEC with 2 act seeds and 2 select seeds, and the given pinned paths / hashes."""
    h4 = dataclasses.replace(SPEC.j.h4, act_seeds=(500, 501), select_seeds=(600, 601))
    return dataclasses.replace(SPEC, j=dataclasses.replace(SPEC.j, h4=h4), **kw)


def _pin(path, doc, key):
    path.write_text(json.dumps(doc))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return _spec(**{f"{key}_path": str(path), f"{key}_sha256": sha})


# ================================================================ small pieces
def test_check_pinned_refuses_another_hash(tmp_path):
    f = tmp_path / "x.json"; f.write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        lm.check_pinned(f, "0" * 64)
    assert lm.check_pinned(f, hashlib.sha256(b"{}").hexdigest()) == {}


def test_dense_restores_the_vector():
    assert lm.dense([1, 3], [0.5, 1.0], 5).tolist() == [0.0, 0.5, 0.0, 1.0, 0.0]
    assert lm.dense([], [], 3).tolist() == [0.0, 0.0, 0.0]


def test_the_file_lists_extend_ks_and_name_every_l_file():
    assert set(lm.K_MEASURE_FILES) < set(lm.MEASURE_FILES) and set(lm.MEASURE_FILES) < set(lm.HASHED_FILES)
    assert {"flymon/brain/l_jobs.py", "flymon/brain/l_measure.py"} <= set(lm.MEASURE_FILES)
    assert {f"flymon/brain/l_{m}.py" for m in ("spec", "store", "pairs", "screen", "rules", "measure", "jobs", "oc",
                                                 "cli")} <= set(lm.HASHED_FILES)
    assert len(set(lm.HASHED_FILES)) == len(lm.HASHED_FILES)


# ================================================================ K's activity cache (odd pairs)
def _k_doc(odours, seeds=(500, 501), params=C3, result=None, **inputs):
    base = dict(odours=odours, params=params, seeds=list(seeds), strength=SPEC.j.h4.h3.strength,
                settle_ms=SPEC.j.h4.oracle_window.settle_ms, read_ms=SPEC.j.h4.oracle_window.read_ms,
                window_ms=int(SPEC.j.h4.kc_window_ms))
    base.update(inputs)
    if result is None:
        result = [{"i": 0, "seed": 500, "kc": [0], "n": [2], "max_win": 1},
                  {"i": 0, "seed": 501, "kc": [0, 1], "n": [1, 1], "max_win": 1},
                  {"i": 1, "seed": 500, "kc": [], "n": [], "max_win": 0},
                  {"i": 1, "seed": 501, "kc": [1], "n": [3], "max_win": 1}]
    return {"key": "k", "kind": "k_act", "inputs": base, "result": result}


ODD = [dict(axis="b", turn=1, x="a", y="b", odor_x={"g1": 1.0}, odor_y={"g2": 1.0})]


def test_odd_activity_rebuilds_the_fire_fractions(tmp_path):
    spec = _pin(tmp_path / "k.json", _k_doc([{"g1": 1.0}, {"g2": 1.0}]), "k_act")
    got = lm.load_odd_activity(spec, ODD, 3, C3)
    assert got[0]["fx"].tolist() == [1.0, 0.5, 0.0] and got[0]["fy"].tolist() == [0.0, 0.5, 0.0]


def test_odd_activity_refuses_another_hash_or_engine(tmp_path):
    spec = _pin(tmp_path / "k.json", _k_doc([{"g1": 1.0}, {"g2": 1.0}]), "k_act")
    with pytest.raises(ValueError, match="engine"):
        lm.load_odd_activity(spec, ODD, 3, {"kc_thresh": 1.7})
    with pytest.raises(ValueError, match="sha256"):
        lm.load_odd_activity(dataclasses.replace(spec, k_act_sha256="0" * 64), ODD, 3, C3)
    other = _pin(tmp_path / "k2.json", _k_doc([{"g1": 1.0}, {"g2": 1.0}], read_ms=100.0), "k_act")
    with pytest.raises(ValueError, match="presentation"):
        lm.load_odd_activity(other, ODD, 3, C3)


def test_odd_activity_refuses_another_odour_order_with_both_digests(tmp_path):
    """Review focus 1: the same odours in another order would attach f vectors to the wrong pair."""
    spec = _pin(tmp_path / "k.json", _k_doc([{"g1": 1.0}, {"g2": 1.0}]), "k_act")
    swapped = [dict(ODD[0], odor_x={"g2": 1.0}, odor_y={"g1": 1.0})]
    with pytest.raises(ValueError, match="odour") as e:
        lm.load_odd_activity(spec, swapped, 3, C3)
    want = lm.odours_digest([{"g2": 1.0}, {"g1": 1.0}])
    got = lm.odours_digest([{"g1": 1.0}, {"g2": 1.0}])
    assert want != got and want in str(e.value) and got in str(e.value)


@pytest.mark.parametrize("seeds, result", [
    ((500,), None),                                                     # another declared seed list
    ((500, 501), [{"i": 0, "seed": 500, "kc": [0], "n": [1], "max_win": 1},
                  {"i": 0, "seed": 501, "kc": [0], "n": [1], "max_win": 1},
                  {"i": 1, "seed": 500, "kc": [1], "n": [1], "max_win": 1}]),          # odour 1 lacks seed 501
    ((500, 501), [{"i": 0, "seed": 500, "kc": [0], "n": [1], "max_win": 1},
                  {"i": 0, "seed": 500, "kc": [0], "n": [1], "max_win": 1},
                  {"i": 1, "seed": 500, "kc": [1], "n": [1], "max_win": 1},
                  {"i": 1, "seed": 501, "kc": [1], "n": [1], "max_win": 1}]),          # odour 0: seed 500 twice
])
def test_odd_activity_refuses_other_act_seeds(tmp_path, seeds, result):
    spec = _pin(tmp_path / "k.json", _k_doc([{"g1": 1.0}, {"g2": 1.0}], seeds=seeds, result=result), "k_act")
    with pytest.raises(ValueError, match="seed"):
        lm.load_odd_activity(spec, ODD, 3, C3)


def test_odd_activity_refuses_a_kc_index_outside_n_kc(tmp_path):
    spec = _pin(tmp_path / "k.json", _k_doc([{"g1": 1.0}, {"g2": 1.0}]), "k_act")
    with pytest.raises(ValueError, match="KC"):
        lm.load_odd_activity(spec, ODD, 1, C3)


# ================================================================ the ceiling raw file (even pairs)
def _row(axis, turn, x, y, n_sel=2, n_act=2, mode="x_only"):
    pre = {t: [[5 + i, 6 + i] for i in range(n_sel)] for t in ("MBON13", "MBON18", "MBON05", "MBON21")}
    kc = {k: {"frac": [0.1] * n_act, "spikes": [3] * n_act, "max_win": [1] * n_act} for k in ("x", "y")}
    return dict(mode=mode, axis=axis, turn=turn, x=x, y=y, fx=[0.0, 0.5], fy=[1.0, 0.0], select={"pre": pre}, kc=kc)


def _ceiling(rows=None, **kw):
    d = dict(pairs_digest=SPEC.j.h4.pairs_digest, complete=True, smoke=False, dirty_hashed=[],
             params={"C3": C3, "C0": {"kc_thresh": 1.0}}, readout={"C3": dict(SPEC.readout)},
             spec={"act_seeds": [500, 501], "select_seeds": [600, 601]},
             rows={"C3": rows if rows is not None else
                   [_row("b", 0, "m vs o", "m vs p", mode="oracle"), _row("a", 0, "m", "n"),
                    _row("b", 0, "m vs o", "m vs p"), _row("b", 2, "q vs o", "q vs r")]})
    d.update(kw)
    return d


def test_even_reads_the_x_only_b_rows_in_order(tmp_path):
    spec = _pin(tmp_path / "c.json", _ceiling(), "ceiling")
    got = lm.load_even(spec, C3)
    assert [g["key"] for g in got] == [("b", 0, "m vs o", "m vs p"), ("b", 2, "q vs o", "q vs r")]
    assert got[0]["fx"].tolist() == [0.0, 0.5] and got[0]["fy"].tolist() == [1.0, 0.0]
    assert got[0]["pre"]["MBON13"] == [[5, 6], [6, 7]]


@pytest.mark.parametrize("change, match", [
    (dict(pairs_digest="0" * 64), "complete"), (dict(complete=False), "complete"), (dict(smoke=True), "complete"),
    (dict(dirty_hashed=["x.py"]), "complete"), (dict(params={"C3": {"kc_thresh": 1.7}}), "engine"),
    (dict(readout={"C3": {"A": "MBON18", "P": "MBON05"}}), "readout"),
    (dict(spec={"act_seeds": [500, 501], "select_seeds": [600]}), "seed"),
    (dict(spec={"act_seeds": [500], "select_seeds": [600, 601]}), "seed"),
])
def test_even_refuses_another_run(tmp_path, change, match):
    spec = _pin(tmp_path / "c.json", _ceiling(**change), "ceiling")
    with pytest.raises(ValueError, match=match):
        lm.load_even(spec, C3)


@pytest.mark.parametrize("row", [_row("b", 0, "m vs o", "m vs p", n_sel=1), _row("b", 0, "m vs o", "m vs p", n_sel=0),
                                 _row("b", 0, "m vs o", "m vs p", n_act=1)])
def test_even_refuses_a_row_with_other_seed_counts(tmp_path, row):
    """An empty select list would pass guard G silently (Task 3's carried finding)."""
    spec = _pin(tmp_path / "c.json", _ceiling(rows=[row]), "ceiling")
    with pytest.raises(ValueError, match="seed"):
        lm.load_even(spec, C3)


def test_even_refuses_a_row_missing_a_guard_type(tmp_path):
    row = _row("b", 0, "m vs o", "m vs p"); del row["select"]["pre"]["MBON05"]
    spec = _pin(tmp_path / "c.json", _ceiling(rows=[row]), "ceiling")
    with pytest.raises(ValueError, match="MBON05"):
        lm.load_even(spec, C3)


# ================================================================ the naive job (oracle_job's first half)
P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")              # synthetic PPL105 core, then PAM08 core (test_h4_jobs)
COMMON = dict(types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602), strength=3.0, settle_ms=50.0,
              read_ms=100.0, window_ms=20)


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    from flymon.brain.circuits import Populations
    c = synthetic_connectome(disjoint_kc=True)
    return c, Populations.from_connectome(c)


def test_the_naive_job_matches_the_oracle_jobs_first_half(conn_pops):
    from flymon.brain.l_jobs import naive_job
    from flymon.brain.stimuli import design_odor_pair
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    n = naive_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **COMMON)
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled
    J._RIG.clear()
    o = J.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, readout={"A": "MBON03", "P": "MBON01"},
                     z={"A": (5.0, 3.0), "P": (8.0, 4.0)}, report_seeds=(608,), alphas=(0.2, 0.5),
                     punish_type="PPL105", reward_type="PAM08", **COMMON)
    assert n["pre"] == o["select"]["pre"] and n["kc"] == o["kc"]
    assert any(x for row in n["pre"]["MBON01"] for x in row), "the synthetic regime must drive the MBONs"
    fx = lm.dense(n["fx_idx"], n["fx_val"], len(pops.kc))
    assert fx.any() and fx.mean() == pytest.approx(np.mean(n["kc"]["x"]["frac"]))      # both = mean fire rate
    assert all(v > 0 for v in n["fx_val"] + n["fy_val"])


def test_the_naive_job_in_the_pool_equals_in_process(conn_pops, synthetic_npz):
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.l_jobs import naive_job
    from flymon.brain.stimuli import design_odor_pair
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    ref = naive_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **COMMON)
    with FlyPool(synthetic_npz, Params(), [{}], workers=1) as pool:
        got = pool.run_jobs(naive_job, [dict(params=P, odor_x=a, odor_y=b, **COMMON)])[0]
    assert got == ref


# ================================================================ the measurer
PAIRS = [dict(axis="b", turn=j, x=f"x{j}", y=f"y{j}", odor_x={"ORN_A": 1.0}, odor_y={"ORN_B": 1.0 + j}) for j in range(5)]


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kwargs_list):
        self.calls.append((fn.__name__, len(kwargs_list)))
        return [dict(b=kw["odor_y"]["ORN_B"], sel=list(kw["select_seeds"]), act=list(kw["act_seeds"]),
                     types=list(kw["types"])) for kw in kwargs_list]


@pytest.fixture
def m(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return lm.LMeasurer(FakePool(), SPEC, ("MBON13", "MBON05"), MeasureCache("results/m0d/l/cache", dict(key="k" * 64), "run"))


def test_the_measurer_runs_missing_pairs_in_rounds_and_resumes_per_pair(m, tmp_path):
    p = Params(kc_thresh=1.6)
    rows = m.naive(p, PAIRS)
    assert m.pool.calls == [("naive_job", 2), ("naive_job", 2), ("naive_job", 1)] and m.params_seen == [p]
    assert [r["b"] for r in rows] == [q["odor_y"]["ORN_B"] for q in PAIRS]
    assert rows[0]["sel"] == list(SPEC.j.h4.select_seeds) and rows[0]["act"] == list(SPEC.j.h4.act_seeds)
    files = sorted((tmp_path / "results/m0d/l/cache/l_naive").glob("*.json"))
    assert len(files) == len(PAIRS) and m.cache.misses == len(PAIRS)
    m.pool.calls.clear()
    assert m.naive(p, PAIRS) == rows and m.pool.calls == []
    files[0].unlink()
    assert m.naive(p, PAIRS) == rows and m.pool.calls == [("naive_job", 1)]
    assert m.naive(p, PAIRS[::-1]) == rows[::-1] and m.pool.calls == [("naive_job", 1)]     # pair order, cache hits
    m.naive(Params(kc_thresh=1.7), PAIRS[:1])
    assert m.pool.calls[-1] == ("naive_job", 1)                                              # other engine, other entry


def test_a_failed_probe_writes_nothing(m, tmp_path):
    with pytest.raises(lm._Missing):
        m.cache.get_or_compute("l_naive", {"a": 1}, lm._missing, [Params()])
    assert not (tmp_path / "results/m0d/l/cache").exists()
