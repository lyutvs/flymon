"""Spec O.7.1: "apl_to_nonkc_zero" zeroes exactly the APL -> non-KC edges (APL -> MBON included); with "apl_to_kc_zero" it
partitions "apl_all_zero"'s edge set; graded-APL views follow; O's rig cache is its own (n_jobs._RIG never touched) and
delegates N's edits to n_jobs.apply_edit; an O1 presentation row is N0f's row (n_jobs._present on O's rig)."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain import o_jobs as O
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_jobs import edge_sources

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
PG = replace(P, apl_mode="graded")
RO = {"A": "MBON03", "P": "MBON01"}
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
                w=np.append(c.w, np.int32(20)))                 # one APL -> MBON edge: the non-KC edit has a target
    O._RIG.clear(); N._RIG.clear(); H4._RIG.clear()
    return c, Populations.from_connectome(c)


def _zeroed(c, pops, params, edit) -> set:
    e0, e1 = Engine(c, pops, params, seed=0), Engine(c, pops, params, seed=0)
    O.apply_edit(e1, pops, edit)
    return set(np.flatnonzero(e0.csc.w != e1.csc.w).tolist())


@pytest.mark.parametrize("params", [P, PG])
def test_the_two_partial_edits_partition_the_all_output_edit(conn_pops, params):
    c, pops = conn_pops
    kc, nonkc, all_ = (_zeroed(c, pops, params, e) for e in ("apl_to_kc_zero", O.NONKC, "apl_all_zero"))
    assert kc and nonkc and not (kc & nonkc) and (kc | nonkc) == all_


def test_all_output_is_every_apl_out_edge(conn_pops):
    c, pops = conn_pops
    e = Engine(c, pops, P, seed=0)
    assert _zeroed(c, pops, P, "apl_all_zero") == set(np.flatnonzero(np.isin(edge_sources(e.csc), pops.apl)).tolist())


def test_nonkc_zeroes_exactly_the_apl_to_non_kc_edges(conn_pops):
    c, pops = conn_pops
    e0, e1 = Engine(c, pops, P, seed=0), Engine(c, pops, P, seed=0)
    sha = O.apply_edit(e1, pops, O.NONKC)
    src, tgt = edge_sources(e1.csc), e1.csc.tgt.astype(np.int64)
    is_apl, to_kc = np.isin(src, pops.apl), np.isin(tgt, pops.kc)
    to_other = is_apl & ~to_kc
    t = np.asarray(c.type).astype(str)
    assert to_other.any() and np.isin(tgt[to_other], np.flatnonzero(t == "MBON01")).any()
    assert np.all(e1.csc.w[to_other] == 0) and np.all(e0.csc.w[to_other] != 0)
    assert np.array_equal(e1.csc.w[~to_other], e0.csc.w[~to_other])
    assert sha not in (O.apply_edit(e0, pops, "none"),)


def test_graded_views_follow_the_nonkc_edit(conn_pops):
    c, pops = conn_pops
    e, _, _, _ = O.rig(c, pops, PG, O.NONKC)
    is_kc = np.zeros(e.N, bool); is_kc[pops.kc] = True
    for tgt, w in e._apl_edges:
        assert np.all(w[~is_kc[tgt]] == 0) and np.any(w[is_kc[tgt]] != 0)


def test_ns_edits_are_delegated_and_unknown_edits_refused(conn_pops):
    c, pops = conn_pops
    for edit in N.EDITS:
        assert O.apply_edit(Engine(c, pops, P, seed=0), pops, edit) == N.apply_edit(Engine(c, pops, P, seed=0), pops,
                                                                                     edit)
    assert O.EDITS == N.EDITS + (O.NONKC,)
    with pytest.raises(ValueError, match="unknown edit"):
        O.apply_edit(Engine(c, pops, P, seed=0), pops, "apl_half")


def test_rig_is_keyed_by_params_and_edit_and_never_touches_ns_cache(conn_pops):
    c, pops = conn_pops
    a = O.rig(c, pops, P, "none")
    b = O.rig(c, pops, P, O.NONKC)
    assert a[0] is not b[0] and a[3] != b[3] and len(O._RIG) == 1
    assert O.rig(c, pops, P, O.NONKC)[0] is b[0] and len(N._RIG) == 0


@pytest.mark.parametrize("params", [P, PG])
def test_o1_rows_are_n0f_rows_for_ns_edits(conn_pops, params):
    c, pops = conn_pops
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    for edit in N.EDITS:
        kw = dict(params=params, edit=edit, odor=X, seeds=[3, 4], readout=RO, **W)
        assert strip(O.presentation_job(_Stub(c), None, pops, None, None, **kw)) == \
            strip(N.presentation_job(_Stub(c), None, pops, None, None, **kw))


@pytest.mark.parametrize("params", [P, PG])
def test_on_nonkc_on_in_one_worker(conn_pops, params):
    c, pops = conn_pops
    kw = dict(params=params, odor=X, seeds=[3, 4], readout=RO, **W)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    on1 = O.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    blk = O.presentation_job(_Stub(c), None, pops, None, None, edit=O.NONKC, **kw)
    on2 = O.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    assert strip(on1) == strip(on2)
    O._RIG.clear()
    assert strip(blk) == strip(O.presentation_job(_Stub(c), None, pops, None, None, edit=O.NONKC, **kw))
    assert {r["csc_sha256"] for r in on1} != {r["csc_sha256"] for r in blk}
    assert [r["edit"] for r in blk] == [O.NONKC] * 2


def test_an_o1_row_is_scalars_only(conn_pops):
    c, pops = conn_pops
    rows = O.presentation_job(_Stub(c), None, pops, None, None, params=P, edit=O.NONKC, odor=X, seeds=[3],
                              readout=RO, **W)
    assert set(rows[0]) == set(N.PRESENTATION_KEYS) | {"edit", "csc_sha256"}
    assert all(np.isscalar(v) for v in rows[0].values())
