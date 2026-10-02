# tests/brain/test_r_store_measure.py
"""R's writer and measurer (R.5, R.8, R.9.7): writes only under results/r/ (results/r/ref/ read-only) and
results/summary/r_lever.json, atomically; RCache keeps smoke and real entries apart and treats a truncated or foreign
entry as missing; load_manifest re-checks every raw file's sha256 and key; the archive copy goes only under its root,
never over an existing directory. The measurer writes one entry per unit and resumes with only the missing units; its
oracle inputs are the encoder's (plus edit / no fixed arm / active_fx) and its arm inputs are P's (plus p_type)."""
import json
from pathlib import Path

import pytest

from flymon.agent.e_measure import EMeasurer
from flymon.agent.e_spec import SPEC as E
from flymon.agent.e_store import ECache
from flymon.brain import p_measure, r_jobs
from flymon.brain import r_store as S
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.r_spec import SPEC

READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]


def test_guard_and_summary_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/q/x.json", "results/p/x.json", "results/encoder/x.json", "results/r/ref/activity/x.json",
                "results/summary/q_reward.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            S.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(S.write_json("results/r/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    S.write_summary_block(S.SUMMARY, "repro", {"n": 1}, [])
    S.write_summary_block(S.SUMMARY, "smoke", {"m": 2}, [])
    S.move_block(S.SUMMARY, "smoke", "smoke_old", [])
    assert S.read_summary() == {"repro": {"n": 1}, "smoke_old": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))


def test_cache_scope_and_bad_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = S.RCache("results/r/cache", {"key": "k"}), S.RCache("results/r/smoke/cache", {"key": "k"})
    real.put("r_oracle", {"act_seeds": [500]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [500]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_009_100})                   # a smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [500]})                  # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [25_008_000, 500]})          # mixed
    sm.put("r_arm", {"seed": 25_009_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_009_100}) == {"v": 2}
    p = real._path("r_oracle", {"act_seeds": [500]})
    p.write_text("{trunc")
    assert real.get("r_oracle", {"act_seeds": [500]}) is None
    p.write_text(json.dumps({"key": "other", "kind": "r_oracle", "result": {}}))
    assert real.get("r_oracle", {"act_seeds": [500]}) is None


def test_load_manifest_rechecks_sha_and_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = Path("results/r/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    man = [dict(key="b|0|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))]
    got, bad = S.load_manifest(man)
    assert bad == [] and got == [dict(key="b|0|x|y", result={"v": 1}, cache_key="K", cache_file=str(f))]
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 2}}))
    assert S.load_manifest(man)[1] and S.load_manifest([dict(man[0], cache_file="results/r/none.json")])[1]


def test_archive_copy_only_under_its_root_and_never_over(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = Path("results/r/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text("x")
    root = tmp_path / "arch"
    out = S.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == "x" and out[0]["sha256"] == sha256_file(f)
    assert Path(out[0]["dst"]).parent.name == "r_oracle"
    with pytest.raises(SystemExit):
        S.archive_copy([str(f)], root / "seal1", root)            # exists
    with pytest.raises(SystemExit):
        S.archive_copy([str(f)], tmp_path / "elsewhere", root)    # outside the root


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is r_jobs.kc_activity_job:
            return [[dict(i=i, seed=s, kc=[0, 1], n=[1, 1], max_win=3, csc_sha256="sha-L", edit_edges=2)
                     for i, _, s in kw["items"]] for kw in kws]
        if fn is r_jobs.r_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


def _m(tmp_path, monkeypatch, n=2):
    monkeypatch.chdir(tmp_path)
    pool = FakePool(n)
    return RMeasurer(pool, S.RCache("results/r/cache", {"key": "k"}), SPEC, Params(), READOUT, Z, TYPES, 100), pool


ROWS = [dict(axis="b", turn=0, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_activity_writes_one_entry_per_odour_and_resumes(tmp_path, monkeypatch):
    m, pool = _m(tmp_path, monkeypatch)
    od = {f"o{i}": {"G": 1.0} for i in range(3)}
    got = m.activity(od, SPEC.lever_edit, 1.0, (500, 501, 502, 503), "gate1")
    assert set(got) == set(od) and got["o0"]["frac"] == [0.02] * 4 and got["o0"]["edit_edges"] == [2]
    assert got["o0"]["csc_sha256"] == ["sha-L"] and got["o0"]["max_win"] == [3] * 4
    n = len(pool.calls)
    assert m.activity(od, SPEC.lever_edit, 1.0, (500, 501, 502, 503), "gate1") == got and len(pool.calls) == n


def test_oracle_resumes_only_missing_pairs(tmp_path, monkeypatch):
    m, pool = _m(tmp_path, monkeypatch)
    L = SPEC.cond("L")
    got = m.oracle(ROWS, L, "jm", SPEC.h4_seeds())
    assert [g["key"] for g in got] == ["b|0|x0|y0", "b|0|x1|y1", "b|0|x2|y2"]
    assert all(Path(g["cache_file"]).exists() for g in got) and got[0]["result"]["edit"] == SPEC.lever_edit
    Path(got[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = m.oracle(ROWS, L, "jm", SPEC.h4_seeds())
    assert pool.calls == [("r_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in got]
    e0 = m.oracle(ROWS[:1], SPEC.cond("E0"), "jm", SPEC.h4_seeds())
    assert e0[0]["result"]["echo"] == {"E": 1.0}


def test_oracle_inputs_are_the_encoders_plus_edit(tmp_path, monkeypatch):
    m, _ = _m(tmp_path, monkeypatch)
    em = EMeasurer(None, ECache("results/encoder/cache", {"key": "k"}), Params(), 100, E)
    ref = em._oracle_kw(ROWS[0], 1.0, READOUT, Z, TYPES, SPEC.h4_seeds())
    mine = m.kwargs(ROWS[0], SPEC.cond("C"), SPEC.h4_seeds())
    assert {k: v for k, v in mine.items() if k not in ("edit", "fixed_alphas", "active_fx")} == ref
    assert mine["edit"] == "none" and mine["fixed_alphas"] == [] and mine["active_fx"] == SPEC.active_fx


class _PassCache:
    def get_or_compute(self, kind, inputs, compute, params_list):
        return compute()


def test_arm_inputs_are_ps_plus_p_type(tmp_path, monkeypatch):
    m, _ = _m(tmp_path, monkeypatch)
    seen = []

    class Capture:
        n_workers = 1

        def run_jobs(self, fn, kws):
            seen.extend(kws)
            return [{} for _ in kws]

    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=23_000_000, point=(0.25, 8.0))
    p_measure.PMeasurer(Capture(), P_SPEC, _PassCache()).p_arms(Params(), [item], READOUT, "PPL105")
    common = m.arm_common(P_SPEC.o.n, READOUT, "PPL105")
    assert {k: v for k, v in common.items() if k != "p_type"} == {k: v for k, v in seen[0].items()
                                                                  if k in common}
    assert set(seen[0]) - set(common) == {"edit", "odor_x", "odor_y", "seed", "arm", "punish", "plastic", "da_zero"}
    rows = m.arms([item], READOUT, "PPL105", "repro", P_SPEC.o.n)
    assert rows[0]["direction"] == "r1" and rows[0]["point"] == [0.25, 8.0] and rows[0]["r"] == {"edit_edges": 0}


def test_measure_files_cover_every_job_module():
    for f in ("flymon/brain/q_jobs.py", "flymon/brain/o_jobs.py", "flymon/brain/n_jobs.py", "flymon/brain/h4_jobs.py",
              "flymon/brain/r_jobs.py", "flymon/brain/r_measure.py", "flymon/brain/r_store.py"):
        assert f in R_MEASURE_FILES, f
    assert set(p_measure.MEASURE_FILES) <= set(R_MEASURE_FILES)
