"""Spec M.10.1: group reactivity is H.4's rule on the group sum; z needs a finite mean and sd > 0; per-cell records;
the measurer caches per pair / per combination and runs in rounds of one item per worker (M.10.3)."""
import numpy as np
import pytest

from flymon.brain import m_measure as mm
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.m_spec import SPEC

CELLS = [10, 11, 12]
REF = [dict(odor=f"R{i}", seed=100 + i, counts=[i, 2 * i, 0]) for i in range(8)]
REST = [dict(seed=100 + i, counts=[0, 1, 0]) for i in range(8)]


def test_group_rows_sum_the_groups_cells():
    g = mm.group_rows(REF[:2], CELLS, {"a": [10, 11], "b": [12]})
    assert g[1]["types"] == {"a": 3, "b": 0} and (g[1]["odor"], g[1]["seed"]) == ("R1", 101)


def test_a_partial_group_sums_only_its_cells():
    g = mm.group_rows(REF, CELLS, {"core": [11]})                     # a core that is part of a type: its cells only
    assert [r["types"]["core"] for r in g] == [2 * i for i in range(8)]


def test_group_react_is_h4s_rule_on_the_sum():
    r = mm.group_react(REF, REST, CELLS, {"a": [10, 11]}, SPEC.j.h4)
    s = np.array([3 * i for i in range(8)], float); d = s - 1
    assert r["a"]["median_delta"] == float(np.median(d)) and r["a"]["zero_share"] == 1 / 8
    assert r["a"]["passes"] == bool(np.median(d) >= 5 and 1 / 8 <= 0.25)


def test_z_is_population_sd_and_invalid_at_zero_sd():
    assert mm.group_z(REF, CELLS, [10, 11], 0) == pytest.approx((10.5, float(np.std([3 * i for i in range(8)]))))
    assert mm.group_z(REF, CELLS, [12], 0) is None                                  # all zero: sd 0 -> invalid
    assert mm.group_z(REF[:1], CELLS, [10, 11], 1) is None                          # one row, ddof 1: sd nan -> invalid


def test_cell_records_carry_every_cell_and_the_largest_variance_share():
    rec = mm.cell_records(REF, REST, CELLS, [10, 11], SPEC.j.h4)
    assert [c["cell"] for c in rec["per_cell"]] == [10, 11]
    assert rec["max_var_share"] == pytest.approx(2 / 3)                            # cov(cell 11, sum) / var(sum)
    assert rec["per_cell"][1]["median_delta"] == float(np.median([2 * i - 1 for i in range(8)]))


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, jobs):
        self.calls.append((fn.__name__, len(jobs)))
        if fn.__name__ == "edit_job":
            return [_fake_edit(j) for j in jobs]
        if fn.__name__ == "reference_cells_job":                        # the real jobs return a row list per item
            return [[dict(odor=o["name"], seed=s, counts=[0, 0, 0]) for o in j["odors"] for s in o["seeds"]] for j in jobs]
        if fn.__name__ == "rest_cells_job":
            return [[dict(seed=s, counts=[0, 0, 0]) for s in j["seeds"]] for j in jobs]
        return [{"fn": fn.__name__, "x": j.get("odor_x")} for j in jobs]


def _fake_edit(j):
    per = [[[1, 2, 3], [4, 5, 6]]]
    return {"reward": {"0.5": {"R": per, "change": 1.0}}, "punish": {"0.5": {"R": per, "change": -1.0}},
            "alpha_reward": 0.5, "alpha_punish": 0.5, "R1_rep": per, "R2_rep": per, "readout": dict(j["readout"]),
            "edited": {"reward": {"group": j["readout"]["P"], "cells": j["groups"][j["readout"]["P"]]},
                       "punish": {"group": j["readout"]["A"], "cells": j["groups"][j["readout"]["A"]]}}}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                         # h3_store.guard is cwd-relative
    return MeasureCache("results/m0d/m/cache", {"key": "k" * 64, "files": {}}, "r")


PAIRS = [dict(axis="b", turn=j, x=f"x{j}", y=f"y{j}", odor_x={"O": j}, odor_y={"O": -j}) for j in range(3)]


def test_pre_is_cached_per_pair_in_rounds(cache):
    pool = FakePool()
    m = mm.MMeasurer(pool, SPEC, cache)
    first = m.pre(Params(), PAIRS, CELLS)
    assert pool.calls == [("pre_job", 2), ("pre_job", 1)] and [r["x"] for r in first] == [{"O": 0}, {"O": 1}, {"O": 2}]
    m.pre(Params(), PAIRS, CELLS)
    assert len(pool.calls) == 2                                                     # all cached
    assert m.params_seen == [Params()]


def test_edit_composes_per_pair_surfaces_the_edited_cells_and_keys_on_the_combination(cache):
    pool = FakePool()
    m = mm.MMeasurer(pool, SPEC, cache)
    pres = [dict(fx_idx=[0], fx_val=[1.0], fy_idx=[], fy_val=[], pre_sel=[[[0, 0, 0], [0, 0, 0]]],
                 pre_rep=[[[0, 0, 1], [0, 1, 0]]], kc={"j": j}) for j in range(3)]
    groups, readout, z = {"g1": [10, 11], "g2": [12]}, {"A": "g2", "P": "g1"}, {"A": (1.0, 2.0), "P": (3.0, 4.0)}
    args = (Params(), PAIRS, pres, CELLS, groups, readout, z, "PAM08", "PPL105")
    rows = m.edit(*args)
    assert pool.calls == [("edit_job", 2), ("edit_job", 1)]
    assert [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows] == [("b", j, f"x{j}", f"y{j}") for j in range(3)]
    assert rows[0]["counts"]["R1"] == {"g1": [[3, 9]], "g2": [[3, 6]]} and rows[2]["kc"] == {"j": 2}
    assert rows[0]["report"]["R1"] == {"A": [[3, 6]], "P": [[3, 9]]}
    assert rows[0]["edited"] == {"reward": {"group": "g1", "cells": [10, 11]}, "punish": {"group": "g2", "cells": [12]}}
    assert m.edit(*args) == rows and len(pool.calls) == 2                           # cached per pair
    m.edit(Params(), PAIRS, pres, CELLS, groups, readout, {"A": (1.0, 2.5), "P": (3.0, 4.0)}, "PAM08", "PPL105")
    m.edit(Params(), PAIRS, pres, CELLS, groups, readout, z, "PAM10", "PPL105")
    m.edit(Params(), PAIRS, pres, CELLS, {"g1": [10], "g2": [12]}, readout, z, "PAM08", "PPL105")
    assert len(pool.calls) == 8                                                     # z, DAN, cells: new keys


def test_reference_rest_and_teach_are_cached_per_item(cache):
    pool = FakePool()
    m = mm.MMeasurer(pool, SPEC, cache)
    odors = [dict(name=f"S{i}", types=["ORN_A"], strengths={"ORN_A": 1.0}, seeds=[i, i + 10]) for i in range(3)]
    ref = m.reference(Params(), odors, CELLS)
    assert pool.calls == [("reference_cells_job", 2), ("reference_cells_job", 1)] and len(ref) == 6
    assert [(r["odor"], r["seed"]) for r in ref[:2]] == [("S0", 0), ("S0", 10)]
    rest = m.rest(Params(), [1, 2], CELLS)
    assert pool.calls[-1] == ("rest_cells_job", 2) and len(rest) == 2
    t = m.teach(Params(), ("MBON05",), "reward_only", "PPL105", "PAM08")
    n = len(SPEC.j.h4.teach_orders) * len(SPEC.j.h4.teach_seeds)
    assert len(t) == n and sum(k for f, k in pool.calls if f == "teach_job") == n
    before = len(pool.calls)
    assert m.reference(Params(), odors, CELLS) == ref and m.rest(Params(), [1, 2], CELLS) == rest
    assert m.teach(Params(), ("MBON05",), "reward_only", "PPL105", "PAM08") == t and len(pool.calls) == before


def test_measure_and_hashed_files_name_every_m_file():
    for f in ("flymon/brain/m_jobs.py", "flymon/brain/m_measure.py", "flymon/brain/h3_jobs.py"):
        assert f in mm.MEASURE_FILES
    for f in ("flymon/brain/m_spec.py", "flymon/brain/m_cands.py", "flymon/brain/m_rules.py", "flymon/brain/m_oc.py",
              "flymon/brain/m_cli.py", "flymon/brain/m_store.py", "scripts/run_m_spec.py", "scripts/run_m_stage0.py",
              "scripts/run_m_oc.py", "scripts/run_m_stage1.py", "scripts/run_m_stage2.py", "scripts/run_m_list.py",
              "scripts/run_m_stage3.py"):
        assert f in mm.HASHED_FILES
