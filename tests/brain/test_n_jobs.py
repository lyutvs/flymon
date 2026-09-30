"""Spec N.8.3 / N.8.7: the rig is cached per worker under (Params, edit); "apl_to_kc_zero" zeroes exactly the APL->KC
edges, "apl_all_zero" equals Params(apl_scale=0) by value; graded-APL views follow the edit; APL-on and APL-block jobs
mixed in one worker never see each other's engine; a presentation row is scalars only (blinding) and its KC and readout
counts are h4_jobs._present_kc's and presentation.decide's."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_jobs import edge_sources
from flymon.brain.h4_jobs import _present_kc, rig_for
from flymon.brain.presentation import decide

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
PG = replace(P, apl_mode="graded")
RO = {"A": "MBON03", "P": "MBON01"}                  # synthetic: PPL105 core = MBON03/04, PAM08 core = MBON01/02
X = {"ORN_DM1": 1.0, "ORN_DA1": 0.5}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    t = np.asarray(c.type).astype(str)
    apl, mbon = int(np.flatnonzero(t == "APL")[0]), int(np.flatnonzero(t == "MBON01")[0])
    c = replace(c, pre=np.append(c.pre, np.int32(apl)), post=np.append(c.post, np.int32(mbon)),
                w=np.append(c.w, np.int32(20)))                 # one APL -> MBON edge, so the two edits differ
    N._RIG.clear(); H4._RIG.clear()                     # h4_jobs.rig_for is keyed by Params only: never share it
    return c, Populations.from_connectome(c)


def _apl_masks(e, pops):
    src, tgt = edge_sources(e.csc), e.csc.tgt.astype(np.int64)
    is_apl = np.isin(src, pops.apl)
    return is_apl & np.isin(tgt, pops.kc), is_apl & ~np.isin(tgt, pops.kc)


def test_apl_to_kc_zero_zeroes_exactly_the_apl_kc_edges(conn_pops):
    c, pops = conn_pops
    e0 = Engine(c, pops, P, seed=0)
    e1 = Engine(c, pops, P, seed=0)
    sha = N.apply_edit(e1, pops, "apl_to_kc_zero")
    to_kc, to_other = _apl_masks(e1, pops)
    assert to_kc.any() and to_other.any()
    assert np.all(e1.csc.w[to_kc] == 0) and np.all(e0.csc.w[to_kc] != 0)
    assert np.array_equal(e1.csc.w[~to_kc], e0.csc.w[~to_kc])
    assert sha != N.apply_edit(e0, pops, "none")


def test_apl_all_zero_equals_apl_scale_zero_by_value(conn_pops):
    c, pops = conn_pops
    e = Engine(c, pops, P, seed=0)
    N.apply_edit(e, pops, "apl_all_zero")
    assert np.array_equal(e.csc.w, Engine(c, pops, replace(P, apl_scale=0.0), seed=0).csc.w)
    with pytest.raises(ValueError, match="unknown edit"):
        N.apply_edit(e, pops, "apl_half")


def test_graded_apl_views_follow_the_edit(conn_pops):
    c, pops = conn_pops
    e, _, _, _ = N.rig(c, pops, PG, "apl_to_kc_zero")
    is_kc = np.zeros(e.N, bool); is_kc[pops.kc] = True
    for tgt, w in e._apl_edges:
        assert np.all(w[is_kc[tgt]] == 0) and np.any(w[~is_kc[tgt]] != 0)


def test_rig_is_keyed_by_params_and_edit_and_holds_one_entry(conn_pops):
    c, pops = conn_pops
    a = N.rig(c, pops, P, "none")
    b = N.rig(c, pops, P, "apl_to_kc_zero")
    assert a[0] is not b[0] and a[3] != b[3] and len(N._RIG) == 1
    assert N.rig(c, pops, P, "apl_to_kc_zero")[0] is b[0]


@pytest.mark.parametrize("params", [P, PG])
def test_on_block_on_in_one_worker(conn_pops, params):
    c, pops = conn_pops
    kw = dict(params=params, odor=X, seeds=[3, 4], readout=RO, **W)
    on1 = N.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    blk = N.presentation_job(_Stub(c), None, pops, None, None, edit="apl_to_kc_zero", **kw)
    on2 = N.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    assert strip(on1) == strip(on2)
    N._RIG.clear()
    fresh = N.presentation_job(_Stub(c), None, pops, None, None, edit="apl_to_kc_zero", **kw)
    assert strip(blk) == strip(fresh)
    assert {r["csc_sha256"] for r in on1} != {r["csc_sha256"] for r in blk}
    assert [r["edit"] for r in blk] == ["apl_to_kc_zero"] * 2


def test_a_presentation_row_is_scalars_only(conn_pops):
    c, pops = conn_pops
    rows = N.presentation_job(_Stub(c), None, pops, None, None, params=P, edit="apl_to_kc_zero", odor=X, seeds=[3],
                              readout=RO, **W)
    assert set(rows[0]) == set(N.PRESENTATION_KEYS) | {"edit", "csc_sha256"}
    assert all(np.isscalar(v) for v in rows[0].values())


def test_kc_and_readout_counts_are_present_kc_and_decide(conn_pops):
    c, pops = conn_pops
    row = N.presentation_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor=X, seeds=[7],
                             readout=RO, **W)[0]
    e, p, _ = rig_for(c, pops, P)                                     # h4's rig: same Params, unedited engine
    p.set_enabled(False)
    ref = _present_kc(e, p, pops, X, 7, W["strength"], W["settle_ms"], W["read_ms"], W["window_ms"])
    t = np.asarray(c.type).astype(str)
    cnt = decide(e, p, pops, [X], W["strength"], 7, W["settle_ms"], W["read_ms"])[0]
    assert row["kc_frac"] == float((ref["read"] > 0).mean()) and row["kc_spikes"] == int(ref["read"].sum())
    assert row["kc_max_win_hz"] == ref["max_win"] * 1000.0 / W["window_ms"]
    assert (row["A"], row["P"]) == (int(cnt[t == "MBON03"].sum()), int(cnt[t == "MBON01"].sum()))
    assert row["steps"] == 150


def test_kc_vectors_carry_the_fired_positions(conn_pops):
    c, pops = conn_pops
    rows = N.kc_vectors_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor=X, seeds=[3, 4],
                            readout=RO, **W)
    for r in rows:
        assert r["n_kc"] == len(pops.kc) and len(r["kc_fired"]) == round(r["kc_frac"] * len(pops.kc))
        assert all(0 <= i < len(pops.kc) for i in r["kc_fired"])
