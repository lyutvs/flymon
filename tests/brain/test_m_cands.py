"""Spec M.10.1 / M.10.2: candidates are core CELL sets merged across DANs, named by the DAN with most cells; the
specificity weights are K's readout weights on a cell list; the pre-check passes a candidate only strictly above its
arm incumbent."""
import dataclasses
import itertools

import numpy as np
import pytest

from flymon.brain import m_cands as mc
from flymon.brain.circuits import Compartment, Populations
from flymon.brain.k_metrics import pair_metrics, readout_weights
from flymon.brain.m_spec import SPEC, smoke

TYPES = np.array(["MBON05", "MBON05", "MBON21", "MBON09", "MBON13", "MBON18", "MBON10", "MBON10", "MBON10"])


def _cp(fam, n_cells, core):
    return Compartment(family=fam, cells=np.arange(n_cells), w_mbon=np.zeros(len(TYPES)), core=np.array(core))


def _declared_comps():
    """Every declared representative with its own distinct core, plus PAM07 = PAM08 / PAM09 = PAM10 (fewer cells)."""
    sets = [[i] for i in range(len(TYPES))] + [list(p) for p in itertools.combinations(range(len(TYPES)), 2)]
    comps = {n: _cp("PAM", 20, sets[i]) for i, n in enumerate(SPEC.reward_candidates)}
    comps.update({n: _cp("PPL1", 4, sets[i]) for i, n in enumerate(SPEC.punish_candidates)})
    comps["PAM07"] = _cp("PAM", 14, comps["PAM08"].core)
    comps["PAM09"] = _cp("PAM", 3, comps["PAM10"].core[::-1])
    return comps


def test_groups_merge_equal_cell_sets_and_name_the_largest_dan():
    comps = {"PAM07": _cp("PAM", 14, [0, 1, 2]), "PAM08": _cp("PAM", 50, [2, 1, 0]), "PAM13": _cp("PAM", 16, [3, 6]),
             "PPL105": _cp("PPL1", 2, [4, 5])}
    g = mc.core_groups(TYPES, comps, "PAM")
    assert [x["name"] for x in g] == ["PAM08", "PAM13"]
    assert g[0]["dans"] == ["PAM07", "PAM08"] and g[0]["cells"] == [0, 1, 2] and g[0]["types"] == ["MBON05", "MBON21"]
    assert g[1]["cells"] == [3, 6] and g[1]["types"] == ["MBON09", "MBON10"]      # 1 of 3 MBON10 cells: a partial type
    assert g[0]["digest"] == mc.cells_digest([2, 0, 1]) and g[0]["n_dan_cells"] == 50
    assert [x["name"] for x in mc.core_groups(TYPES, comps, "PPL1")] == ["PPL105"]


def test_groups_tie_goes_to_the_first_name_and_empty_cores_are_skipped():
    comps = {"PAM02": _cp("PAM", 5, [1]), "PAM01": _cp("PAM", 5, [1]), "PAM03": _cp("PAM", 9, [])}
    g = mc.core_groups(TYPES, comps, "PAM")
    assert [x["name"] for x in g] == ["PAM01"] and g[0]["dans"] == ["PAM01", "PAM02"]


def test_candidates_are_the_declared_lists():
    got = mc.candidates(TYPES, _declared_comps(), SPEC)
    assert tuple(c["name"] for c in got["reward"]) == SPEC.reward_candidates
    assert tuple(c["name"] for c in got["punish"]) == SPEC.punish_candidates
    assert next(c for c in got["reward"] if c["name"] == "PAM08")["dans"] == ["PAM07", "PAM08"]
    assert next(c for c in got["reward"] if c["name"] == "PAM10")["dans"] == ["PAM09", "PAM10"]


def test_candidates_under_smoke_check_the_declared_lists_then_filter():
    comps = _declared_comps()
    got = mc.candidates(TYPES, comps, smoke(SPEC))
    assert [c["name"] for c in got["reward"]] == ["PAM08", "PAM10"]
    assert [c["name"] for c in got["punish"]] == ["PPL103", "PPL105"]
    full = mc.candidates(TYPES, comps, SPEC)
    assert got["reward"][0] == next(c for c in full["reward"] if c["name"] == "PAM08")


def test_candidates_refuse_other_names():
    comps = {"PAM08": _cp("PAM", 50, [0]), "PPL105": _cp("PPL1", 2, [4])}
    spec = dataclasses.replace(SPEC, reward_candidates=("PAM08",), punish_candidates=("PPL105",))
    with pytest.raises(ValueError, match="PAM"):
        mc.candidates(TYPES, comps, SPEC)
    with pytest.raises(ValueError, match="PAM"):                     # a narrowed spec does not narrow the check
        mc.candidates(TYPES, comps, spec)
    merged = _declared_comps()
    merged["PAM12"] = _cp("PAM", 20, merged["PAM11"].core)           # two declared names on one cell set
    with pytest.raises(ValueError, match="reward"):
        mc.candidates(TYPES, merged, SPEC)
    with pytest.raises(ValueError, match="PAM99"):                   # a spec naming an undeclared candidate
        mc.candidates(TYPES, _declared_comps(), dataclasses.replace(SPEC, reward_candidates=("PAM08", "PAM99")))


def test_overlap_is_cell_intersection():
    assert mc.overlap({"cells": [1, 2]}, {"cells": [2, 3]}) and not mc.overlap({"cells": [1]}, {"cells": [3]})


def test_cell_weights_equal_readout_weights_on_a_whole_type(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    cells = np.flatnonzero(np.asarray(c.type).astype(str) == "MBON01")
    assert np.array_equal(mc.cell_weights(c, pops, cells, 1), readout_weights(c, pops, "MBON01", 1))
    assert mc.cell_weights(c, pops, cells[:0], 1).sum() == 0


def test_cell_weights_on_a_cell_list_use_only_those_cells(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    t = np.asarray(c.type).astype(str)
    two = np.flatnonzero((t == "MBON01") | (t == "MBON03"))       # one cell per type here: a cell list across types
    got = mc.cell_weights(c, pops, two, 1)
    assert np.array_equal(got, readout_weights(c, pops, "MBON01", 1) + readout_weights(c, pops, "MBON03", 1))
    sel = np.isin(c.pre, np.asarray(pops.kc)) & np.isin(c.post, two) & (c.w >= 1)
    assert got.sum() == c.w[sel].sum() > 0
    assert mc.cell_weights(c, pops, two, 10**6).sum() == 0         # min_weight filters edges


def test_lobe_shares_sum_to_the_masked_weight():
    w = np.array([1.0, 3.0, 0.0, 4.0])
    lobes = {"g": np.array([1, 0, 0, 0], bool), "ab": np.array([0, 1, 0, 1], bool)}
    assert mc.lobe_shares(w, lobes) == {"g": 0.125, "ab": 0.875}
    assert mc.lobe_shares(np.zeros(4), lobes) == {"g": 0.0, "ab": 0.0}


def test_s_is_k_s_on_the_candidate_weights():
    fx, fy, w = np.array([1.0, 0.5, 0.0]), np.array([0.0, 0.5, 1.0]), np.array([2.0, 1.0, 5.0])
    assert mc.pair_s(fx, fy, w) == pair_metrics(fx, fy, w, w)["S"] == pytest.approx(2.0 / 2.5)


def test_spec_check_passes_only_strictly_above_the_incumbent():
    spec = dataclasses.replace(SPEC, reward_candidates=("PAM08", "PAM10", "PAM12"), punish_candidates=("PPL103", "PPL105"))
    vals = {"PAM08": [0.2, 0.3, 0.4], "PAM10": [0.3, 0.4, 0.5], "PAM12": [0.1, 0.3, 0.6],
            "PPL105": [0.5, 0.5, 0.5], "PPL103": [0.5, 0.5, 0.5]}
    r = mc.spec_check(vals, spec)
    assert r["outcome"] == mc.SPEC_GO and r["passing"] == {"reward": ["PAM10"], "punish": []}
    assert r["medians"]["PAM12"] == 0.3 and r["incumbent_median"] == {"reward": 0.3, "punish": 0.5}
    none = mc.spec_check({k: [0.0] * 3 for k in vals}, spec)
    assert none["outcome"] == mc.STOP_NO_SPECIFICITY and none["passing"] == {"reward": [], "punish": []}


def test_spec_check_one_empty_arm_still_goes_and_missing_values_refuse():
    spec = dataclasses.replace(SPEC, reward_candidates=("PAM08", "PAM10"), punish_candidates=("PPL103", "PPL105"))
    vals = {"PAM08": [0.3], "PAM10": [0.3], "PPL105": [0.1], "PPL103": [0.2]}
    r = mc.spec_check(vals, spec)
    assert r["outcome"] == mc.SPEC_GO and r["passing"] == {"reward": [], "punish": ["PPL103"]}
    with pytest.raises(KeyError):
        mc.spec_check({k: v for k, v in vals.items() if k != "PAM10"}, spec)
