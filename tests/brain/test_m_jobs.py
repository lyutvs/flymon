"""Spec M.10.1 reading 4-5: the per-cell reference / rest jobs are H.3's protocol (their type sums equal
reference_job / rest_job's); pre_job + edit_job composed on type-cell groups equal H.4's oracle_job exactly; edit_job
edits exactly the readout cells of each taught group (a partial-type core reads and edits only its core cells)."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h3_jobs as H3
from flymon.brain import h4_jobs as J
from flymon.brain import m_jobs as M
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.h4_formula import dprime, dv
from flymon.brain.stimuli import design_odor_pair

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")
ODORS = [dict(name="S00", types=["ORN_DM1"], strengths={"ORN_DM1": 1.0}, seeds=[11, 12]),
         dict(name="S01", types=["ORN_DA1"], strengths={"ORN_DA1": 1.0}, seeds=[13])]
COMMON = dict(act_seeds=(500, 501), select_seeds=(600, 601, 602), report_seeds=(608, 609), strength=3.0,
              settle_ms=50.0, read_ms=100.0)
EDIT = dict(select_seeds=COMMON["select_seeds"], report_seeds=COMMON["report_seeds"], alphas=(0.2, 0.5, 0.8),
            strength=3.0, settle_ms=50.0, read_ms=100.0, reward_type="PAM08", punish_type="PPL105")


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    return c, Populations.from_connectome(c)


def _type_groups(c):
    t = np.asarray(c.type).astype(str)
    return {n: np.flatnonzero(t == n).tolist() for n in TYPES}


def _drop_dan_edges(c, pops, pairs):
    """Remove DAN -> MBON edges (no current in the CSC; only the compartments change)."""
    t = np.asarray(c.type).astype(str)
    keep = np.ones(len(c.pre), bool)
    for dan, mbon in pairs:
        keep &= ~((t[c.pre] == dan) & (t[c.post] == mbon))
    return replace(c, pre=c.pre[keep], post=c.post[keep], w=c.w[keep])


def test_reference_and_rest_cells_sum_to_h3s_type_counts(conn_pops):
    c, pops = conn_pops
    cells = [int(i) for i in pops.mbon]
    kw = dict(params=P, strength=3.0, settle_ms=50.0, read_steps=100)
    H3._CACHE.clear()
    ref = H3.reference_job(_Stub(c), None, pops, None, None, odors=ODORS, quantiles=(50,), callout=(), **kw)
    H3._CACHE.clear()
    got = M.reference_cells_job(_Stub(c), None, pops, None, None, odors=ODORS, cells=cells, **kw)
    pos = {i: k for k, i in enumerate(cells)}
    assert len(got) == len(ref) == 3
    for r, g in zip(ref, got):
        assert (g["odor"], g["seed"]) == (r["odor"], r["seed"])
        for n, idx in _type_groups(c).items():
            assert sum(g["counts"][pos[i]] for i in idx) == r["types"][n]
    H3._CACHE.clear()
    rest = H3.rest_job(_Stub(c), None, pops, None, None, params=P, seeds=[1, 2], settle_ms=50.0, read_steps=100)
    H3._CACHE.clear()
    rc = M.rest_cells_job(_Stub(c), None, pops, None, None, params=P, seeds=[1, 2], settle_ms=50.0, read_steps=100,
                          cells=cells)
    assert [g["seed"] for g in rc] == [1, 2]
    for r, g in zip(rest, rc):
        for n, idx in _type_groups(c).items():
            assert sum(g["counts"][pos[i]] for i in idx) == r["types"][n]


def test_pre_and_edit_compose_to_oracle_job(conn_pops):
    from flymon.brain.m_measure import compose
    c, pops = conn_pops
    # single-type cores (PAM08 -> MBON01, PPL105 -> MBON03), so the type readout is exactly the taught core cells
    c = _drop_dan_edges(c, pops, [("PAM08", "MBON02"), ("PPL105", "MBON04")])
    a, b = design_odor_pair(pops, k=2, seed=0)
    cells = [int(i) for i in pops.mbon]
    groups = _type_groups(c)
    readout, z = {"A": "MBON03", "P": "MBON01"}, {"A": (5.0, 3.0), "P": (8.0, 4.0)}
    J._RIG.clear()
    o = J.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, readout=readout, z=z, types=TYPES,
                     alphas=(0.2, 0.5, 0.8), window_ms=20, punish_type="PPL105", reward_type="PAM08", **COMMON)
    J._RIG.clear()
    pre = M.pre_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, cells=cells, window_ms=20, **COMMON)
    ed = M.edit_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, fx_idx=pre["fx_idx"],
                    fx_val=pre["fx_val"], cells=cells, groups=groups, readout=readout, z=z, pre_sel=pre["pre_sel"],
                    **EDIT)
    assert compose(pre, ed, cells, groups) == o
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled          # weights and plasticity restored
    assert ed["edited"] == {"reward": {"group": "MBON01", "cells": groups["MBON01"]},
                            "punish": {"group": "MBON03", "cells": groups["MBON03"]}}


def test_a_partial_core_group_reads_only_its_cells(conn_pops):
    c, pops = conn_pops
    cells = [int(i) for i in pops.mbon]
    counts = [[[1, 2, 3, 4][: len(cells)], [5, 6, 7, 8][: len(cells)]]]
    g = M.group_counts(counts, cells, {"part": [cells[0]], "all": cells})
    assert g["part"] == [[1, 5]] and g["all"] == [[sum(counts[0][0]), sum(counts[0][1])]]


def test_a_partial_core_is_edited_and_read_on_its_core_cells_only(conn_pops):
    """MBON02's cell relabelled MBON03: type MBON03 = {mbon[1], mbon[2]}; PAM08's core {mbon[0], mbon[1]} and PPL105's
    {mbon[2], mbon[3]} each hold one of its two cells. The edits and the readout use the cores, never the type."""
    c, pops = conn_pops
    mb = [int(i) for i in pops.mbon]
    t = np.asarray(c.type).astype(object).copy(); t[mb[1]] = "MBON03"
    c = replace(c, type=np.asarray(t, dtype=c.type.dtype))
    comps = compartments(c, pops, P.core_frac)
    core = {n: sorted(int(i) for i in comps[n].core) for n in ("PAM08", "PPL105")}
    typ = np.flatnonzero(np.asarray(c.type).astype(str) == "MBON03").tolist()
    assert core == {"PAM08": [mb[0], mb[1]], "PPL105": [mb[2], mb[3]]} and typ == [mb[1], mb[2]]
    a, b = design_odor_pair(pops, k=2, seed=0)
    readout, z = {"A": "PPL105", "P": "PAM08"}, {"A": (5.0, 3.0), "P": (8.0, 4.0)}
    J._RIG.clear()
    pre = M.pre_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, cells=mb, window_ms=20, **COMMON)
    kw = dict(params=P, odor_x=a, odor_y=b, fx_idx=pre["fx_idx"], fx_val=pre["fx_val"], cells=mb, readout=readout,
              z=z, pre_sel=pre["pre_sel"], **EDIT)
    ed = M.edit_job(_Stub(c), None, pops, None, None, groups=core, **kw)
    assert ed["edited"] == {"reward": {"group": "PAM08", "cells": core["PAM08"]},
                            "punish": {"group": "PPL105", "cells": core["PPL105"]}}
    assert mb[2] not in ed["edited"]["reward"]["cells"] and mb[1] not in ed["edited"]["punish"]["cells"]
    ap = lambda cnt: {"A": M.group_counts(cnt, mb, core)["PPL105"], "P": M.group_counts(cnt, mb, core)["PAM08"]}
    for s, v in ed["reward"].items():                                  # the change is read on the core sums
        assert v["change"] == dprime(dv(ap(v["R"]), z) - dv(ap(pre["pre_sel"]), z))
    # a readout group that is the type (not the taught core) is refused
    with pytest.raises(ValueError, match="edits"):
        M.edit_job(_Stub(c), None, pops, None, None, groups={"PAM08": core["PAM08"], "PPL105": typ}, **kw)
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled
