"""S's writer (S.5, S.8; plan Reading 3): writes only under results/s/ and results/summary/s_lever.json, atomically —
every other track's path (results/r/, results/p/, results/q/, results/encoder/, R's summary) refuses; SCache is
RCache with S's writer and S's smoke seeds (smoke and real entries never share a root; a truncated or foreign entry is
missing); r_store's load_manifest and archive_copy are reused as they are (archive only under S's root); and the
unchanged RMeasurer writes S entries under results/s/ and resumes with only the missing units."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_store
from flymon.brain import s_store as S
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.s_spec import SPEC, smoke

READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/r/x.json", "results/r/cache/r_oracle/a.json", "results/p/x.json", "results/q/x.json",
                "results/encoder/x.json", "results/summary/r_lever.json", "results/summary/q_reward.json",
                "results/summary/p_learning.json", "results/sx/a.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            S.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(S.write_json("results/s/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    S.write_summary_block(S.SUMMARY, "reuse", {"n": 1}, [])
    S.write_summary_block(S.SUMMARY, "set", {"m": 2}, [])
    assert S.read_summary() == {"reuse": {"n": 1}, "set": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert S.SUMMARY == SPEC.summary


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/s").mkdir(parents=True)
    outside.mkdir()
    (root / "results/s/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        S.write_json("results/s/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_scache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = S.SCache(SPEC.cache_dir, {"key": "k"}), S.SCache(SPEC.smoke_cache_dir, {"key": "k"})
    assert isinstance(real, r_store.RCache)
    assert S.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_300_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_300_000]}) == {"v": 1}
    d = json.loads(real._path("r_oracle", {"act_seeds": [24_300_000]}).read_text())
    assert d["key"] == real.key("r_oracle", {"act_seeds": [24_300_000]}) and d["inputs"] == {"act_seeds": [24_300_000]}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_109_100})                   # S's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_300_000]})           # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [24_309_000, 24_300_000]})   # mixed
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [25_008_000]})               # R's smoke seed is not S's
    sm.put("r_arm", {"seed": 25_109_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_109_100}) == {"v": 2}
    p = real._path("r_oracle", {"act_seeds": [24_300_000]})
    p.write_text("{trunc")
    assert real.get("r_oracle", {"act_seeds": [24_300_000]}) is None
    p.write_text(json.dumps({"key": "other", "kind": "r_oracle", "result": {}}))
    assert real.get("r_oracle", {"act_seeds": [24_300_000]}) is None
    with pytest.raises(SystemExit):
        S.SCache("results/r/cache", {"key": "k"}).put("r_oracle", {"act_seeds": [24_300_000]}, {"v": 1}, [])


def test_scache_keys_equal_rcaches():
    """Same key formula and layout as R (ECache over the shared code key) — only the root and the writer differ."""
    a, b = S.SCache("results/s/cache", {"key": "k"}), r_store.RCache("results/r/cache", {"key": "k"})
    ins = {"act_seeds": [500]}
    assert a.key("r_oracle", ins) == b.key("r_oracle", ins)
    assert a._path("r_oracle", ins).name == b._path("r_oracle", ins).name


def test_manifest_and_archive_are_r_stores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert S.load_manifest is r_store.load_manifest and S.archive_copy is r_store.archive_copy
    f = Path("results/s/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    man = [dict(key="b|104|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))]
    got, bad = S.load_manifest(man)
    assert bad == [] and got[0]["result"] == {"v": 1}
    root = tmp_path / "arch-s"
    out = S.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == f.read_text()
    with pytest.raises(SystemExit):
        S.archive_copy([str(f)], tmp_path / "arch-r" / "x", root)


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is r_jobs.r_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


ROWS = [dict(axis="b", turn=104 + i, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_the_unchanged_measurer_writes_s_entries_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = RMeasurer(pool, S.SCache(SPEC.cache_dir, {"key": "k"}), SPEC, Params(), READOUT, Z, TYPES, 100)
    got = m.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    assert all(g["cache_file"].startswith("results/s/cache/r_oracle/") for g in got)
    stored = json.loads(Path(got[0]["cache_file"]).read_text())["inputs"]
    assert stored["act_seeds"] == SPEC.judge_seeds()["act"] and stored["edit"] == SPEC.lever_edit
    Path(got[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = m.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    assert pool.calls == [("r_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in got]
    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=25_100_000, point=(0.25, 8.0))
    rows = m.arms([item, dict(item, edit=SPEC.lever_edit)], READOUT, "PPL105", "gate2", P_SPEC.o.n)
    assert [r["edit"] for r in rows] == ["none", SPEC.lever_edit] and len(list(Path("results/s/cache/r_arm").glob(
        "*.json"))) == 2
