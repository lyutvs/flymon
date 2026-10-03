# tests/brain/test_t_store.py
"""T's writer and caches (T.5, T.8, T.9.2; plan Readings 3, 5): writes only under results/t/ and
results/summary/t_lever.json, atomically — every other track's path (results/r/, results/s/, results/p/, R's and S's
summaries) refuses; TCache is RCache with T's writer and T's smoke seeds; RReadCache reads R's root by content key and
never writes; r_store's load_manifest and archive_copy are reused as they are; the unchanged RMeasurer, one per z,
writes T entries under results/t/ whose inputs differ only by z; and R's committed even raw for C (and L) is found by
content key under results/r/cache with block h4's z (gate ③'s reuse)."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_store
from flymon.brain import t_store as T
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.t_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]
ZL = {"A": (7.0, 6.0), "P": (80.0, 30.0)}


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/r/x.json", "results/r/cache/r_oracle/a.json", "results/s/x.json", "results/s/cache/a.json",
                "results/p/x.json", "results/q/x.json", "results/summary/r_lever.json",
                "results/summary/s_lever.json", "results/summary/p_learning.json", "results/tx/a.json",
                "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            T.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(T.write_json("results/t/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    T.write_summary_block(T.SUMMARY, "reuse", {"n": 1}, [])
    T.write_summary_block(T.SUMMARY, "set", {"m": 2}, [])
    assert T.read_summary() == {"reuse": {"n": 1}, "set": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert T.SUMMARY == SPEC.summary


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/t").mkdir(parents=True)
    outside.mkdir()
    (root / "results/t/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        T.write_json("results/t/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_tcache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = T.TCache(SPEC.cache_dir, {"key": "k"}), T.TCache(SPEC.smoke_cache_dir, {"key": "k"})
    assert isinstance(real, r_store.RCache)
    assert T.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_400_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_400_000]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_209_100})                   # T's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_400_000]})           # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [24_309_000]})               # S's smoke seed is not T's
    sm.put("r_arm", {"seed": 25_209_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_209_100}) == {"v": 2}
    p = real._path("r_oracle", {"act_seeds": [24_400_000]})
    p.write_text("{trunc")
    assert real.get("r_oracle", {"act_seeds": [24_400_000]}) is None
    with pytest.raises(SystemExit):
        T.TCache("results/r/cache", {"key": "k"}).put("r_oracle", {"act_seeds": [24_400_000]}, {"v": 1}, [])


def test_rreadcache_reads_rs_root_and_never_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = r_store.RCache("results/r/cache", {"key": "k"})
    w.put("r_oracle", {"act_seeds": [500]}, {"v": 3}, [])
    ro = T.RReadCache("results/r/cache", {"key": "k"})
    assert ro.get("r_oracle", {"act_seeds": [500]}) == {"v": 3}
    assert ro.key("r_oracle", {"act_seeds": [500]}) == w.key("r_oracle", {"act_seeds": [500]})
    before = sorted(Path("results/r").rglob("*"))
    with pytest.raises(SystemExit):
        ro.put("r_oracle", {"act_seeds": [501]}, {"v": 4}, [])
    assert sorted(Path("results/r").rglob("*")) == before
    assert {k for k in vars(T.RReadCache) if not k.startswith("__")} == {"put"}
    assert {k for k in vars(T.TCache) if not k.startswith("__")} == {"put"}


def test_manifest_and_archive_are_r_stores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert T.load_manifest is r_store.load_manifest and T.archive_copy is r_store.archive_copy
    f = Path("results/t/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    man = [dict(key="b|0|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))]
    got, bad = T.load_manifest(man)
    assert bad == [] and got[0]["result"] == {"v": 1}
    root = tmp_path / "arch-t"
    out = T.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == f.read_text()
    with pytest.raises(SystemExit):
        T.archive_copy([str(f)], tmp_path / "arch-s" / "x", root)


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is r_jobs.r_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"], z=kw["z"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


ROWS = [dict(axis="b", turn=i, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_one_measurer_per_z_over_one_cache(tmp_path, monkeypatch):
    """Plan Reading 4: L's measurer carries z_lever, C's h4 z; same cache, entries differ only by z and condition."""
    monkeypatch.chdir(tmp_path)
    pool, cache = FakePool(), T.TCache(SPEC.cache_dir, {"key": "k"})
    ml = RMeasurer(pool, cache, SPEC, Params(), READOUT, ZL, TYPES, 100)
    mc = RMeasurer(pool, cache, SPEC, Params(), READOUT, Z, TYPES, 100)
    gl = ml.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    gc = mc.oracle(ROWS, SPEC.cond("C"), "judge", SPEC.judge_seeds())
    il, ic = (json.loads(Path(g[0]["cache_file"]).read_text())["inputs"] for g in (gl, gc))
    assert il["z"] == {k: list(v) for k, v in ZL.items()} and ic["z"] == {k: list(v) for k, v in Z.items()}
    assert {k for k in il if il[k] != ic[k]} == {"z", "edit", "condition"}
    assert all(g["cache_file"].startswith("results/t/cache/r_oracle/") for g in gl + gc)
    Path(gl[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = ml.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    assert pool.calls == [("r_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in gl]
    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=25_200_000, point=(0.25, 8.0))
    rows = mc.arms([item, dict(item, edit=SPEC.lever_edit)], READOUT, "PPL105", "gate2", P_SPEC.o.n)
    assert [r["edit"] for r in rows] == ["none", SPEC.lever_edit]


NPZ = ROOT / "data/malecns.npz"
R_CACHE = ROOT / SPEC.r_cache_dir / "r_oracle"


@pytest.mark.skipif(not (NPZ.exists() and R_CACHE.exists()), reason="no connectome or no R raw cache")
def test_rs_even_raw_is_found_by_content_key():
    """T.9.2: with block h4's z, T's spec builds the same oracle inputs R's even stage stored, so every L and C even
    entry is read from results/r/cache with no pool and nothing written."""
    from flymon.agent.config import load_c3_config
    from flymon.brain import q_pairs, r_pairs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    cfg = load_c3_config(SPEC.m0d_summary)
    enc = json.loads((ROOT / SPEC.encoder_summary).read_text())
    q_pairs.check_strength(enc, SPEC)
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = r_pairs.even_rows(pops, rc, enc, SPEC)
    m0d = json.loads((ROOT / SPEC.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    cache = T.RReadCache(str(ROOT / SPEC.r_cache_dir), code_key(str(NPZ), files=R_MEASURE_FILES))
    m = RMeasurer(None, cache, SPEC, cfg.params, cfg.readout, cfg.z, types, len(pops.kc))
    for n in ("C", "L"):
        ins = [m.inputs(r, SPEC.cond(n), "even", SPEC.h4_seeds()) for r in rows]
        assert all(cache.get("r_oracle", x) is not None for x in ins), n
    got = m.oracle(rows, SPEC.cond("C"), "even", SPEC.h4_seeds())
    assert len(got) == 39 and m.last_jobs == 0
