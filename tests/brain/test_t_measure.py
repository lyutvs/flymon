# tests/brain/test_t_measure.py
"""T's measurement file (T.1, T.8, T.9.4, T.9.6): the T measurement key is the shared key's files plus t_measure.py
and h3_spec.py and differs from the shared key; ZMeasurer splits the reference set and the rest seeds over the pool
as h3_measure.chunks does; and on the real connectome t_ref_job / t_rest_job with edit "none" give exactly
h3_jobs.reference_job's / rest_job's counts (the unedited z can reproduce block h4's z bit for bit), while the lever
changes 2 CSC edges (R's L CSC 860cba4f…, the unedited 1aee8398…)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from flymon.brain import h3_jobs
from flymon.brain import t_measure as TM
from flymon.brain.h3_spec import SPEC as H3
from flymon.brain.h3_spec import make_odors
from flymon.brain.r_measure import R_MEASURE_FILES
from flymon.brain.t_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


def test_t_measure_files_and_key():
    assert TM.T_MEASURE_FILES == ("flymon/brain/t_measure.py", "flymon/brain/h3_spec.py")
    assert not set(TM.T_MEASURE_FILES) & set(R_MEASURE_FILES)
    if NPZ.exists():
        k = TM.t_measure_key(str(NPZ))
        assert set(k["files"]) - {"npz:malecns.npz"} == set(R_MEASURE_FILES) | set(TM.T_MEASURE_FILES)
        assert k["key"] != SPEC.r_shared_key and len(k["key"]) == 64


class FakePool:
    n_workers = 3

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, [len(kw.get("odors", kw.get("seeds", []))) for kw in kws], kws[0]))
        if fn is TM.t_ref_job:
            return [[dict(odor=o["name"], seed=s) for o in kw["odors"] for s in o["seeds"]] for kw in kws]
        return [[dict(seed=s) for s in kw["seeds"]] for kw in kws]


def test_zmeasurer_chunks_and_arguments():
    pool = FakePool()
    zm = TM.ZMeasurer(pool, {"p": 1}, "MBON05", ["MBON13", "MBON05"])
    odors = [dict(name=f"R{j:02d}", seeds=[2 * j, 2 * j + 1], strengths={}) for j in range(7)]
    ref = zm.reference(SPEC.lever_edit, odors, 0.35, 800.0, 600)
    assert [(r["odor"], r["seed"]) for r in ref] == [(o["name"], s) for o in odors for s in o["seeds"]]
    name, sizes, kw = pool.calls[0]
    assert name == "t_ref_job" and sizes == [3, 3, 1]
    assert {k: kw[k] for k in ("edit", "p_type", "types", "strength", "settle_ms", "read_steps")} == dict(
        edit=SPEC.lever_edit, p_type="MBON05", types=["MBON13", "MBON05"], strength=0.35, settle_ms=800.0,
        read_steps=600)
    rest = zm.rest("none", list(range(14)), 800.0, 600)
    assert [r["seed"] for r in rest] == list(range(14)) and pool.calls[1][0] == "t_rest_job"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_unedited_jobs_equal_h3s_reference_and_rest_jobs():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    conn = Connectome.load(str(NPZ))
    pops = Populations.from_connectome(conn)
    params = load_c3_config(SPEC.m0d_summary).params
    od, w, eng = make_odors(pops, H3.reference)[:1], H3.reference_window, SimpleNamespace(conn=conn)
    types = ["MBON13", "MBON18", "MBON05", "MBON21"]
    mine = TM.t_ref_job(eng, None, pops, None, None, params, "none", "MBON05", types, od, H3.strength,
                        w.settle_ms, int(w.read_ms))
    ref = h3_jobs.reference_job(eng, None, pops, None, None, params, od, H3.strength, w.settle_ms, int(w.read_ms),
                                tuple(H3.apl_v_quantiles), tuple(H3.callout_types))
    assert [m["types"] for m in mine] == [{t: r["types"][t] for t in types} for r in ref]
    assert [m["kc_active_frac"] for m in mine] == [r["kc_active_frac"] for r in ref]
    assert {m["edit_edges"] for m in mine} == {0} and mine[0]["csc_sha256"].startswith("1aee8398")
    seed = od[0]["seeds"][0]
    r1 = TM.t_rest_job(eng, None, pops, None, None, params, "none", "MBON05", types, [seed], w.settle_ms,
                       int(w.read_ms))
    r2 = h3_jobs.rest_job(eng, None, pops, None, None, params, [seed], w.settle_ms, int(w.read_ms))
    assert r1[0]["types"] == {t: r2[0]["types"][t] for t in types}
    lv = TM.t_rest_job(eng, None, pops, None, None, params, SPEC.lever_edit, "MBON05", types, [seed], w.settle_ms,
                       int(w.read_ms))
    assert lv[0]["edit_edges"] == SPEC.lever_edges and lv[0]["csc_sha256"].startswith("860cba4f")
