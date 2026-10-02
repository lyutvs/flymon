# tests/brain/test_q_store_measure.py
"""Q.6.8 storage: one atomic content-key file per (pair, condition, seed block) under results/q/cache, resume after an
interruption, a corrupt or foreign file recomputed, writes refused outside results/q/ + q_reward.json and inside
results/q/q0_cache; the job inputs are the encoder's oracle inputs plus edit / fixed arms / active_fx (Reading 3)."""
import json
from pathlib import Path

import pytest

from flymon.agent.e_measure import EMeasurer
from flymon.agent.e_spec import SPEC as E
from flymon.agent.e_store import ECache
from flymon.brain import q_store
from flymon.brain.config import Params
from flymon.brain.q_measure import Q_MEASURE_FILES, QMeasurer, cond_params
from flymon.brain.q_spec import SPEC

READOUT = {"A": "MBON13", "P": "MBON05"}
Z = {"A": (10.78125, 9.41), "P": (26.25, 19.31)}
TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
ROW = dict(axis="b", turn=0, x="Surf", y="Ice Beam", odor_x={"ORN_DM1": 1.2, "ORN_VA2": 0.8},
           odor_y={"ORN_DM6": 1.0, "ORN_VC1": 1.0})
CODE = {"key": "k" * 64}


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn.__name__ == "activity_job":
            return [[dict(i=i, seed=s, kc=[0, 1, 2], n=[1, 1, 1], max_win=3) for i, _, s in kw["items"]] for kw in kws]
        return [{"echo": kw["odor_x"], "edit": kw["edit"], "strength": kw["strength"]} for kw in kws]


def _m(pool, root="results/q/cache"):
    return QMeasurer(pool, q_store.QCache(root, CODE), SPEC, Params(), READOUT, Z, TYPES, n_kc=100)


@pytest.mark.parametrize("path", ["results/encoder/cache/oracle/x.json", "results/summary/encoder_grid.json",
                                  "results/q/q0_cache/x.json", "results/o/x.json", "elsewhere.json"])
def test_guard_refuses(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        q_store.write_json(path, {}, [Params()])
    assert e.value.code == 2 and not (tmp_path / path).exists()


def test_guard_allows_q_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    q_store.write_json("results/q/cache/oracle/a.json", {"a": 1}, [Params()])
    q_store.write_summary_block(q_store.SUMMARY, "s_c", {"s": 1.0}, [Params()])
    assert json.loads((tmp_path / q_store.SUMMARY).read_text()) == {"s_c": {"s": 1.0}}
    assert not list((tmp_path / "results/q/cache/oracle").glob(".*.tmp"))


def test_kwargs_are_the_encoder_oracle_inputs_plus_q_fields():
    seeds = SPEC.repro_seeds()
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    mine = _m(None).kwargs(ROW, base, seeds)
    enc = EMeasurer(None, ECache("unused", CODE), Params(), 100, E, guard_params=False)._oracle_kw(
        ROW, 1.0, READOUT, Z, TYPES, seeds)
    extra = {"edit", "fixed_alphas", "active_fx"}
    assert {k: v for k, v in mine.items() if k not in extra} == enc
    assert mine["edit"] == "none" and mine["fixed_alphas"] == [0.8, 1.0] and mine["active_fx"] == 0.5


def test_cond_params_scales_mv_only():
    p = Params()
    lo = SPEC.conditions(1.0, 0.8, 1.25)[4]
    q = cond_params(p, lo)
    assert q.mv_per_synapse == p.mv_per_synapse * 0.8 and q.kc_thresh_file == p.kc_thresh_file
    assert cond_params(p, SPEC.conditions(1.0, 0.8, 1.25)[0]) is p


def test_one_file_per_pair_condition_block_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    rows = [dict(ROW, x=f"x{i}") for i in range(5)]
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    out = _m(pool).run(rows, base, "q1", SPEC.q1_seeds())
    assert [o["key"] for o in out] == [f"b|0|x{i}|Ice Beam" for i in range(5)]
    assert pool.calls == [("q_oracle_job", 2), ("q_oracle_job", 2), ("q_oracle_job", 1)]
    files = sorted((tmp_path / "results/q/cache/oracle").glob("*.json"))
    assert len(files) == 5
    pool.calls.clear()
    files[0].write_text("{trunc")                                   # a corrupt entry
    again = _m(pool).run(rows, base, "q1", SPEC.q1_seeds())
    assert pool.calls == [("q_oracle_job", 1)] and [o["result"] for o in again] == [o["result"] for o in out]
    pool.calls.clear()
    _m(pool).run(rows, base, "repro", SPEC.repro_seeds())            # another seed block: new entries
    assert sum(n for _, n in pool.calls) == 5


def test_foreign_key_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    _m(pool).run([ROW], base, "q1", SPEC.q1_seeds())
    f = next((tmp_path / "results/q/cache/oracle").glob("*.json"))
    d = json.loads(f.read_text()); d["key"] = "other"; f.write_text(json.dumps(d))
    pool.calls.clear()
    _m(pool).run([ROW], base, "q1", SPEC.q1_seeds())
    assert pool.calls == [("q_oracle_job", 1)]


def test_kc_probe_fractions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = _m(FakePool(), "results/q/smoke/cache").kc_probe({"a": ROW["odor_x"], "b": ROW["odor_y"]}, Params(), 1.0, (24_008_007, 24_008_008))
    assert got == {"a": [0.03, 0.03], "b": [0.03, 0.03]}


def test_measure_files_cover_the_job_imports():
    for f in ("flymon/brain/q_jobs.py", "flymon/brain/o_jobs.py", "flymon/brain/n_jobs.py", "flymon/brain/h4_jobs.py",
              "flymon/brain/q_measure.py", "flymon/brain/q_store.py", "flymon/brain/k_jobs.py"):
        assert f in Q_MEASURE_FILES
    root = Path(__file__).resolve().parents[2]
    assert all((root / f).exists() for f in Q_MEASURE_FILES)


def test_smoke_and_real_roots_are_separate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    smoke_seeds = dict(act=(24_008_000,), select=(24_008_001,), report=(24_008_002,))
    with pytest.raises(SystemExit):
        _m(FakePool()).run([ROW], base, "smoke", smoke_seeds)                      # smoke seeds, real root
    with pytest.raises(SystemExit):
        _m(FakePool(), "results/q/smoke/cache").run([ROW], base, "q1", SPEC.q1_seeds())   # real seeds, smoke root
    _m(FakePool(), "results/q/smoke/cache").run([ROW], base, "smoke", smoke_seeds)
    assert not (tmp_path / "results/q/cache").exists()


def test_key_covers_strength_edit_scale_condition_alphas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m, seeds = _m(None), SPEC.q1_seeds()
    cs = SPEC.conditions(0.7, 0.8, 1.25)
    keys = {m.cache.key("oracle", m.inputs(ROW, c, "q1", seeds)) for c in cs}
    assert len(keys) == len(cs)
    base = m.inputs(ROW, cs[0], "q1", seeds)
    assert m.cache.key("oracle", dict(base, fixed_alphas=[0.8])) != m.cache.key("oracle", base)
    assert m.cache.key("oracle", dict(base, z={"A": [1.0, 2.0], "P": [3.0, 4.0]})) != m.cache.key("oracle", base)
