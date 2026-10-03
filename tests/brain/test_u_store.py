"""U's writer and caches (U.5, U.8): writes only under results/u/ and results/summary/u_lever.json, atomically — every
other track's path (results/t/, results/r/, results/s/, results/p/, T's, R's and S's summaries) refuses; UCache is
RCache with U's writer and U's smoke seeds; RReadCache is T's (R's root, read only); r_store's load_manifest and
archive_copy are reused; the unchanged RMeasurer over U's pool adapter, one per z, writes U entries under results/u/
whose inputs carry f in the edit string, so two f differ in their cache keys."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_store, t_store
from flymon.brain import u_measure as UM
from flymon.brain import u_store as U
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.r_measure import RMeasurer
from flymon.brain.u_spec import SPEC, at, smoke

READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]
ZF = {"A": (7.0, 6.0), "P": (60.0, 25.0)}


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/t/x.json", "results/t/z.json", "results/r/cache/r_oracle/a.json", "results/s/x.json",
                "results/p/x.json", "results/summary/t_lever.json", "results/summary/r_lever.json",
                "results/summary/s_lever.json", "results/ux/a.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            U.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(U.write_json("results/u/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    U.write_summary_block(U.SUMMARY, "reuse", {"n": 1}, [])
    U.write_summary_block(U.SUMMARY, "path", {"m": 2}, [])
    assert U.read_summary() == {"reuse": {"n": 1}, "path": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert U.SUMMARY == SPEC.summary


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/u").mkdir(parents=True)
    outside.mkdir()
    (root / "results/u/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        U.write_json("results/u/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_ucache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = U.UCache(SPEC.cache_dir, {"key": "u"}), U.UCache(SPEC.smoke_cache_dir, {"key": "u"})
    assert isinstance(real, r_store.RCache)
    assert U.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_500_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_500_000]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_309_100})                   # U's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_500_000]})           # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [24_409_000]})               # T's smoke seed is not U's
    sm.put("r_arm", {"seed": 25_309_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_309_100}) == {"v": 2}
    with pytest.raises(SystemExit):
        U.UCache("results/t/cache", {"key": "u"}).put("r_oracle", {"act_seeds": [24_500_000]}, {"v": 1}, [])
    assert {k for k in vars(U.UCache) if not k.startswith("__")} == {"put"}


def test_reused_store_parts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert U.RReadCache is t_store.RReadCache
    assert U.load_manifest is r_store.load_manifest and U.archive_copy is r_store.archive_copy
    f = Path("results/u/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    got, bad = U.load_manifest([dict(key="b|0|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))])
    assert bad == [] and got[0]["result"] == {"v": 1}
    root = tmp_path / "arch-u"
    out = U.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == f.read_text()


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is UM.u_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"], z=kw["z"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


ROWS = [dict(axis="b", turn=i, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_one_measurer_per_z_over_upool_and_ucache_and_f_in_the_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool, cache = FakePool(), U.UCache(SPEC.cache_dir, {"key": "u"})
    up = UM.UPool(pool)
    s3, s5 = at(SPEC, 0.3), at(SPEC, 0.5)
    m3, m5 = RMeasurer(up, cache, SPEC, Params(), READOUT, ZF, TYPES, 100), RMeasurer(up, cache, SPEC, Params(),
                                                                                       READOUT, ZF, TYPES, 100)
    mc = RMeasurer(up, cache, SPEC, Params(), READOUT, Z, TYPES, 100)
    g3 = m3.oracle(ROWS, s3.cond("L"), "judge", SPEC.judge_seeds())
    g5 = m5.oracle(ROWS, s5.cond("L"), "judge", SPEC.judge_seeds())
    gc = mc.oracle(ROWS, s3.cond("C"), "judge", SPEC.judge_seeds())
    assert pool.calls == [("u_oracle_job", 2), ("u_oracle_job", 1)] * 3
    i3, i5, ic = (json.loads(Path(g[0]["cache_file"]).read_text())["inputs"] for g in (g3, g5, gc))
    assert i3["edit"] == "u_apl_mbon05_x0.3" and i5["edit"] == "u_apl_mbon05_x0.5" and ic["edit"] == "none"
    assert {k for k in i3 if i3[k] != i5[k]} == {"edit"}
    assert {k for k in i3 if i3[k] != ic[k]} == {"z", "edit", "condition"}
    assert g3[0]["cache_key"] != g5[0]["cache_key"]
    assert all(g["cache_file"].startswith("results/u/cache/r_oracle/") for g in g3 + g5 + gc)
    Path(g3[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = m3.oracle(ROWS, s3.cond("L"), "judge", SPEC.judge_seeds())
    assert pool.calls == [("u_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in g3]
    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=25_300_000, point=(0.25, 8.0))
    from flymon.brain.p_spec import SPEC as P_SPEC
    rows = mc.arms([item, dict(item, edit=s3.lever_edit)], READOUT, "PPL105", "gate2", P_SPEC.o.n)
    assert [r["edit"] for r in rows] == ["none", "u_apl_mbon05_x0.3"] and pool.calls[-1] == ("u_arm_job", 2)
