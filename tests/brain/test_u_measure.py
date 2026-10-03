# tests/brain/test_u_measure.py
"""U's measurement file (U.1, U.8, U.9.1, U.9.3, U.9.4 P2-5): the edit grammar (f in every edit string, hence in every
cache key), the U measurement key (T's files + u_measure.py), the pool adapter and UZMeasurer's chunks; and on the real
connectome: the APL -> MBON05 weights become w × f on exactly the 2 edges with every other weight unchanged, f = 0 gives
R's L CSC 860cba4f… (+0.0, not -0.0) and f = 1 the unedited 1aee8398…, the contrast blocks hit 4 / 17 and 7 / 2 / 2
edges, and every U job copy equals the R / T job it copies at both endpoints (f = 1 ≡ "none", f = 0 ≡
apl_to_mbon05_zero, the edit labels aside) — while a copy whose edit is not applied fails that reproduction."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.brain import r_jobs
from flymon.brain import t_measure as TM
from flymon.brain import u_measure as UM
from flymon.brain.h3_spec import SPEC as H3
from flymon.brain.h3_spec import make_odors
from flymon.brain.r_measure import R_MEASURE_FILES

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
SHA_NONE = "1aee839811b8aa662588fbfab3050cee00962ddc8ecd4d2fd1361fd0c72692d5"
SHA_ZERO = "860cba4f9eead9d7a85b632f2c93c51790c5082fe80460975cb1300239c73e46"
LEVER = "apl_to_mbon05_zero"
TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
KC_TYPES = ["KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m"]


def test_edit_grammar():
    assert UM.F_GRID == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert UM.F_ALLOWED == (0.0,) + UM.F_GRID + (1.0,)
    assert UM.u_edit(0.3) == "u_apl_mbon05_x0.3" and UM.u_edit(0) == "u_apl_mbon05_x0.0"
    assert UM.u_edit(1.0, "out_block") == "u_apl_mbon05_x1.0+out_block"
    assert UM.parse_u_edit("u_apl_mbon05_x0.0+chain_entry") == (0.0, "chain_entry")
    assert len({UM.u_edit(f) for f in UM.F_ALLOWED}) == 11
    for bad in (0.25, 0.75, -0.1, 1.1):
        with pytest.raises(ValueError):
            UM.u_edit(bad)
    with pytest.raises(ValueError):
        UM.u_edit(0.5, "other")
    for bad in ("none", LEVER, "u_apl_mbon05_x0.30", "u_apl_mbon05_x0.3+x", "u_apl_mbon05_x0.25"):
        with pytest.raises(ValueError):
            UM.parse_u_edit(bad)
    assert UM.is_u_edit(UM.u_edit(0.5)) and not UM.is_u_edit("none") and not UM.is_u_edit(LEVER)
    assert UM.BLOCKS == {"out_block": (("MBON03", "MBON13"), ("CRE055", "MBON13")),
                         "chain_entry": (("MBON05", "MBON09"), ("MBON05", "MBON11"), ("MBON05", "MBON01"))}


def test_u_measure_key_is_t_keys_files_plus_u_measure():
    assert UM.U_MEASURE_FILES == ("flymon/brain/u_measure.py",)
    assert not set(UM.U_MEASURE_FILES) & (set(R_MEASURE_FILES) | set(TM.T_MEASURE_FILES))
    if NPZ.exists():
        k, t = UM.u_measure_key(str(NPZ)), TM.t_measure_key(str(NPZ))
        assert set(k["files"]) == set(t["files"]) | set(UM.U_MEASURE_FILES)
        assert k["key"] != t["key"] and len(k["key"]) == 64


class FakePool:
    n_workers = 3

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, [len(kw.get("odors", kw.get("seeds", []))) for kw in kws], kws[0]))
        if fn is UM.u_ref_job:
            return [[dict(odor=o["name"], seed=s) for o in kw["odors"] for s in o["seeds"]] for kw in kws]
        if fn is UM.u_rest_job:
            return [[dict(seed=s) for s in kw["seeds"]] for kw in kws]
        return [fn.__name__ for _ in kws]


def test_upool_maps_rs_jobs_to_us_copies_and_refuses_others():
    pool = FakePool()
    up = UM.UPool(pool)
    assert up.n_workers == 3 and UM.UPool(None).n_workers == 1
    assert up.run_jobs(r_jobs.r_oracle_job, [{}]) == ["u_oracle_job"]
    assert up.run_jobs(r_jobs.r_arm_job, [{}, {}]) == ["u_arm_job"] * 2
    assert up.run_jobs(r_jobs.kc_activity_job, [{}]) == ["u_kc_activity_job"]
    with pytest.raises(ValueError, match="no copy"):
        up.run_jobs(TM.t_ref_job, [{}])


def test_uzmeasurer_chunks_and_arguments():
    pool = FakePool()
    zm = UM.UZMeasurer(pool, {"p": 1}, "MBON05", TYPES, KC_TYPES)
    odors = [dict(name=f"R{j:02d}", seeds=[2 * j, 2 * j + 1], strengths={}) for j in range(7)]
    ref = zm.reference(UM.u_edit(0.4), odors, 0.35, 800.0, 600)
    assert [(r["odor"], r["seed"]) for r in ref] == [(o["name"], s) for o in odors for s in o["seeds"]]
    name, sizes, kw = pool.calls[0]
    assert name == "u_ref_job" and sizes == [3, 3, 1]
    assert {k: kw[k] for k in ("edit", "p_type", "types", "kc_types", "strength", "settle_ms", "read_steps")} == dict(
        edit="u_apl_mbon05_x0.4", p_type="MBON05", types=TYPES, kc_types=KC_TYPES, strength=0.35, settle_ms=800.0,
        read_steps=600)
    rest = zm.rest(UM.u_edit(0.4), list(range(14)), 800.0, 600)
    assert [r["seed"] for r in rest] == list(range(14)) and pool.calls[1][0] == "u_rest_job"


# ================================================================ the real connectome
@pytest.fixture(scope="module")
def real():
    if not NPZ.exists():
        pytest.skip("no connectome")
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.r_spec import SPEC as R
    conn = Connectome.load(str(NPZ))
    pops = Populations.from_connectome(conn)
    return SimpleNamespace(conn=conn, pops=pops, params=load_c3_config(R.m0d_summary).params,
                           eng=SimpleNamespace(conn=conn), od=make_odors(pops, H3.reference)[:1])


def _engine(real, edit):
    from flymon.brain.engine_cpu import Engine
    e = Engine(real.conn, real.pops, real.params, seed=0)
    w0 = e.csc.w.copy()
    return e, w0, UM.apply_u_edit(e, real.pops, edit, "MBON05")


def test_partial_edit_scales_exactly_the_two_edges(real):
    """U.9.1: the edited APL out-edge view = original × f on the 2 edges, every other weight unchanged; f = 0 is R's L
    CSC (+0.0), f = 1 the unedited CSC; "none" changes nothing."""
    from flymon.brain.h3_jobs import edge_sources
    for f in UM.F_ALLOWED:
        e, w0, (sha, n, blocks) = _engine(real, UM.u_edit(f))
        src, tgt = edge_sources(e.csc), e.csc.tgt.astype(np.int64)
        t = np.asarray(real.conn.type).astype(str)
        m = np.isin(src, np.asarray(real.pops.apl)) & (t[tgt] == "MBON05")
        assert n == 2 == int(m.sum()) and blocks == {}
        assert np.array_equal(e.csc.w[~m], w0[~m])
        assert np.array_equal(e.csc.w[m], w0[m] * np.float32(f) + np.float32(0.0))
        assert not np.signbit(e.csc.w[m]).any() if f == 0.0 else np.all(w0[m] < 0)
        views = np.concatenate([w for _, w in e._apl_edges])
        assert np.array_equal(np.sort(views), np.sort(e.csc.w[np.isin(src, np.asarray(real.pops.apl))]))
        if f == 0.0:
            assert sha == SHA_ZERO
        elif f == 1.0:
            assert sha == SHA_NONE
        else:
            assert sha not in (SHA_ZERO, SHA_NONE)
    e, w0, (sha, n, blocks) = _engine(real, "none")
    assert (sha, n, blocks) == (SHA_NONE, 0, {}) and np.array_equal(e.csc.w, w0)


def test_contrast_blocks_hit_the_declared_edges(real):
    """U.9.3 4 (CSC, min_weight 5): MBON03 -> MBON13 4, CRE055 -> MBON13 17; MBON05 -> MBON09 7, -> MBON11 2, -> MBON01 2;
    the block edges become 0, the APL -> MBON05 edges are f's."""
    _, _, (sha_o, n, blocks) = _engine(real, UM.u_edit(0.0, "out_block"))
    assert n == 2 and blocks == {"MBON03->MBON13": 4, "CRE055->MBON13": 17}
    _, _, (sha_c, n, blocks) = _engine(real, UM.u_edit(0.0, "chain_entry"))
    assert n == 2 and blocks == {"MBON05->MBON09": 7, "MBON05->MBON11": 2, "MBON05->MBON01": 2}
    _, _, (sha_1, _, _) = _engine(real, UM.u_edit(1.0, "out_block"))
    assert len({sha_o, sha_c, sha_1, SHA_ZERO, SHA_NONE}) == 5


def _strip(x):
    """A job result without the edit labels (the edit string and its edge count): what U.9.1 compares bit for bit."""
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in ("edit", "edit_edges", "wall_s")}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _odours(real):
    o = real.od[0]["strengths"]
    keys = sorted(o)
    return {k: o[k] for k in keys[: len(keys) // 2]}, {k: o[k] for k in keys[len(keys) // 2:]}


def _oracle_kw(real, edit):
    from flymon.brain.r_spec import SPEC as R
    ox, oy = _odours(real)
    return dict(params=real.params, odor_x=ox, odor_y=oy, readout={"A": "MBON13", "P": "MBON05"},
                z={"A": [10.78125, 9.412096743243064], "P": [26.25, 19.30889259728101]}, types=TYPES,
                act_seeds=[11], select_seeds=[12], report_seeds=[13], alphas=[0.5, 1.0], strength=1.0,
                settle_ms=R.settle_ms, read_ms=R.read_ms, window_ms=R.window_ms, punish_type=R.punish_type,
                reward_type=R.reward_type, edit=edit, fixed_alphas=[], active_fx=R.active_fx)


@pytest.mark.parametrize("f,r_edit,sha", [(1.0, "none", SHA_NONE), (0.0, LEVER, SHA_ZERO)])
def test_oracle_copy_reproduces_rs_job_at_both_endpoints(real, f, r_edit, sha):
    mine = UM.u_oracle_job(real.eng, None, real.pops, None, None, **_oracle_kw(real, UM.u_edit(f)))
    ref = r_jobs.r_oracle_job(real.eng, None, real.pops, None, None, **_oracle_kw(real, r_edit))
    assert _strip(mine) == _strip(ref)
    assert mine["q"]["csc_sha256"] == sha and mine["q"]["edit_edges"] == 2 and mine["q"]["edit"] == UM.u_edit(f)


def test_none_goes_to_rs_job_itself(monkeypatch):
    """C and E0 run R's verified jobs: with edit "none" every U copy calls the R job it copies, unchanged."""
    seen = []
    for name in ("r_oracle_job", "kc_activity_job", "r_arm_job"):
        monkeypatch.setattr(r_jobs, name, lambda *a, _n=name, **kw: seen.append((_n, kw["edit"])) or _n)
    eng = SimpleNamespace(conn=None)
    assert UM.u_oracle_job(eng, None, None, None, None, edit="none", params=None) == "r_oracle_job"
    assert UM.u_kc_activity_job(eng, None, None, None, None, params=None, edit="none", p_type="MBON05", items=[],
                                strength=1.0, settle_ms=1.0, read_ms=1.0, window_ms=1) == "kc_activity_job"
    kw = {k: None for k in ("odor_x", "odor_y", "seed", "arm", "punish", "plastic", "da_zero", "readout",
                            "punish_type", "reward_type", "strength", "settle_ms", "read_ms", "window_ms", "trials",
                            "present_ms", "gap_ms", "train_settle_ms", "seed_base", "seed_stride")}
    assert UM.u_arm_job(eng, None, None, None, None, params=None, edit="none", p_type="MBON05", **kw) == "r_arm_job"
    assert seen == [("r_oracle_job", "none"), ("kc_activity_job", "none"), ("r_arm_job", "none")]


def test_mutation_an_edit_that_is_not_applied_fails_the_reproduction(real, monkeypatch):
    """U.9.1: a copy whose edit never reaches the engine (the scale skipped) no longer reproduces R's L at f = 0."""
    real_apply = UM.apply_u_edit

    def skipped(eng, pops, edit, p_type):
        sha, n, blocks = real_apply(eng, pops, "none", p_type)
        return sha, 2, blocks
    monkeypatch.setattr(UM, "apply_u_edit", skipped)
    UM._RIG.clear()
    mine = UM.u_oracle_job(real.eng, None, real.pops, None, None, **_oracle_kw(real, UM.u_edit(0.0)))
    UM._RIG.clear()
    ref = r_jobs.r_oracle_job(real.eng, None, real.pops, None, None, **_oracle_kw(real, LEVER))
    assert _strip(mine) != _strip(ref) and mine["q"]["csc_sha256"] != SHA_ZERO


@pytest.mark.parametrize("f,r_edit", [(1.0, "none"), (0.0, LEVER)])
def test_kc_activity_copy_reproduces_rs_job(real, f, r_edit):
    from flymon.brain.r_spec import SPEC as R
    ox, _ = _odours(real)
    kw = dict(params=real.params, p_type="MBON05", items=[(0, ox, 21), (1, ox, 22)], strength=1.0,
              settle_ms=R.settle_ms, read_ms=R.read_ms, window_ms=R.window_ms)
    mine = UM.u_kc_activity_job(real.eng, None, real.pops, None, None, edit=UM.u_edit(f), **kw)
    ref = r_jobs.kc_activity_job(real.eng, None, real.pops, None, None, edit=r_edit, **kw)
    assert _strip(mine) == _strip(ref) and {r["edit_edges"] for r in mine} == {2}


@pytest.mark.parametrize("f,r_edit", [(1.0, "none"), (0.0, LEVER)])
def test_arm_copy_reproduces_rs_job(real, f, r_edit):
    from flymon.brain.p_spec import SPEC as P
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.r_spec import SPEC as R
    ox, oy = _odours(real)
    common = RMeasurer(None, None, R, real.params, {"A": "MBON13", "P": "MBON05"}, {}, TYPES, 1).arm_common(
        P.o.n, {"A": "MBON13", "P": "MBON05"}, P.o.n.h3.punish_type)
    kw = dict(common, odor_x=ox, odor_y=oy, seed=31, arm="punish", punish=True, plastic=True, da_zero=False, trials=1)
    mine = UM.u_arm_job(real.eng, None, real.pops, None, None, **dict(kw, edit=UM.u_edit(f)))
    ref = r_jobs.r_arm_job(real.eng, None, real.pops, None, None, **dict(kw, edit=r_edit))
    assert _strip(mine) == _strip(ref) and mine["r"]["edit_edges"] == 2


@pytest.mark.parametrize("f,t_edit", [(1.0, "none"), (0.0, LEVER)])
def test_reference_and_rest_copies_reproduce_ts_jobs(real, f, t_edit):
    w = H3.reference_window
    types = TYPES + ["MBON09", "MBON11", "MBON01", "MBON03", "CRE055", "MBON30", "LHMB1"]
    mine = UM.u_ref_job(real.eng, None, real.pops, None, None, real.params, UM.u_edit(f), "MBON05", types, KC_TYPES,
                        real.od, H3.strength, w.settle_ms, int(w.read_ms))
    ref = TM.t_ref_job(real.eng, None, real.pops, None, None, real.params, t_edit, "MBON05", TYPES, real.od,
                       H3.strength, w.settle_ms, int(w.read_ms))
    assert [{t: m["types"][t] for t in TYPES} for m in mine] == [r["types"] for r in ref]
    assert [(m["odor"], m["seed"], m["kc_active_frac"], m["csc_sha256"]) for m in mine] == [
        (r["odor"], r["seed"], r["kc_active_frac"], r["csc_sha256"]) for r in ref]
    assert {m["edit_edges"] for m in mine} == {2}
    for m in mine:
        assert set(m["mech"]["kc_sub"]) == set(KC_TYPES) and all(0.0 <= v <= 1.0 for v in m["mech"]["kc_sub"].values())
        assert m["mech"]["apl_out_per_step"] >= 0.0 and m["block_edges"] == {}
    seed = real.od[0]["seeds"][0]
    r1 = UM.u_rest_job(real.eng, None, real.pops, None, None, real.params, UM.u_edit(f), "MBON05", types, KC_TYPES,
                       [seed], w.settle_ms, int(w.read_ms))
    r2 = TM.t_rest_job(real.eng, None, real.pops, None, None, real.params, t_edit, "MBON05", TYPES, [seed],
                       w.settle_ms, int(w.read_ms))
    assert {t: r1[0]["types"][t] for t in TYPES} == r2[0]["types"] and r1[0]["csc_sha256"] == r2[0]["csc_sha256"]


def test_type_cells_equal_mbon_index_for_the_mbon_types(real):
    from flymon.brain.h3_jobs import mbon_type_index
    from flymon.brain.h4_jobs import type_cells
    idx, cells = mbon_type_index(real.conn, real.pops), type_cells(real.conn, TYPES + ["MBON09", "MBON03"])
    for t in TYPES + ["MBON09", "MBON03"]:
        assert np.array_equal(np.sort(cells[t]), idx[t])
