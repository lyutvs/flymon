"""AA.5's structure-matched coverage and the order-0 synthetic validation: coverage takes the group-size structure only
(no value argument — mutant "coverage에 결과값을 넘김"), is bit-reproducible on its stream, covers δ · c(7) for the
d′ methods and δ for the raw ones, reports the minimum of 9 cells and flags < 0.90."""
import dataclasses
import inspect

import numpy as np
import pytest

from flymon.brain import aa_estimate as E
from flymon.brain.aa_spec import SPEC

S = dataclasses.replace(SPEC, cov_reps=12, cov_b=60, boot_chunk=50, synth_reps=6)


def test_signature_takes_structure_only():
    assert list(inspect.signature(E.coverage).parameters) == ["sizes", "method", "set_tag", "s", "reps", "B"]
    assert list(inspect.signature(E.coverage_cell).parameters) == ["sizes", "method", "set_tag", "ci", "s", "reps", "B"]
    with pytest.raises(TypeError):
        E.coverage(np.array([1.5, 2.0]), "fly", "S1", S)
    with pytest.raises(TypeError):
        E.coverage((2, 1.5), "fly", "S1", S)


def test_cells_reproducible_and_stream_per_set_and_method():
    a = E.coverage((3, 2, 1, 1, 1), "two_stage", "S1", S)
    assert a == E.coverage((3, 2, 1, 1, 1), "two_stage", "S1", S)
    assert len(a["cells"]) == 9 and a["min"] == min(c["share"] for c in a["cells"])
    assert [(c["delta"], c["het"]) for c in a["cells"]] == E.cov_cells(S)
    assert E.stream(S.synth_seed, "cov", "S1", "two_stage", 0).integers(0, 1 << 30) != \
        E.stream(S.synth_seed, "cov", "S31", "two_stage", 0).integers(0, 1 << 30)


def test_target_is_delta_times_c7_for_dprime_and_delta_for_raw():
    assert E.target(1.5, "fly", S) == pytest.approx(1.5 * E.c_k(8))
    assert E.target(1.5, "raw_fly", S) == 1.5 and E.target(0.0, "two_stage", S) == 0.0


def test_flag_below_bar():
    s = dataclasses.replace(S, cov_bar=1.01)
    assert E.coverage((1, 1), "fly", "S1", s)["under"] is True
    s = dataclasses.replace(S, cov_bar=0.0)
    assert E.coverage((1, 1), "fly", "S1", s)["under"] is False


def test_two_stage_covers_on_independent_pairs():
    """Sanity: 20 singletons, no heterogeneity, δ 1.5 — the two-stage CI covers δ · c(7) in most replicates."""
    s = dataclasses.replace(SPEC, cov_reps=200, cov_b=500, boot_chunk=500)
    i = E.cov_cells(s).index((1.5, "none"))
    assert E.coverage_cell((1,) * 20, "two_stage", "S1", i, s) >= 0.85


def test_group_heterogeneity_hurts_the_fly_level_ci():
    s = dataclasses.replace(SPEC, cov_reps=150, cov_b=300, boot_chunk=300)
    i = E.cov_cells(s).index((1.0, "group"))
    assert E.coverage_cell((4, 4, 4), "fly", "S1", i, s) < E.coverage_cell((4, 4, 4), "two_stage", "S1", i, s)


def test_synth_validation_record():
    v = E.synth_validation(S)
    assert len(v["cells"]) == 6 and v["pairs"] == 20 and v["method"] == "two_stage" and v["label"] == "기록"
    assert v == E.synth_validation(S)


def test_flag_boundary_equal_to_bar_is_not_flagged():
    """Spec AA.5 6230: "최솟값 < 0.90이면 … 명목 미달" — strictly below; a minimum exactly at the bar is not flagged."""
    assert E.coverage_summary([0.95] * 8 + [0.90], S)["under"] is False
    assert E.coverage_summary([0.95] * 8 + [0.899], S)["under"] is True
    a = E.coverage((1, 1), "fly", "S1", S)
    assert E.coverage((1, 1), "fly", "S1", dataclasses.replace(S, cov_bar=a["min"]))["under"] is False


class _SignRng:
    """normal() returns scale × (+1, −1, +1, …) and standard_normal() zeros; integers() is a real draw."""

    def __init__(self):
        self.r = np.random.default_rng(0)

    def normal(self, loc, scale, size):
        return loc + scale * np.where(np.arange(size) % 2 == 0, 1.0, -1.0)

    def standard_normal(self, size):
        return np.zeros(size)

    def integers(self, *a, **k):
        return self.r.integers(*a, **k)


@pytest.mark.parametrize("het,u", [("group", [0.5, 0.5, 0.5, -0.5]), ("pair", [0.5, -0.5, 0.5, -0.5]),
                                   ("none", [0.0, 0.0, 0.0, 0.0])])
def test_group_and_pair_heterogeneity_are_distinct(het, u, monkeypatch):
    """Groups (3, 1), δ 0, raw (no probe noise) so the data the bootstrap sees are δ + u exactly: group-level u is
    shared inside a group (+, +, +, −), pair-level u is one draw per pair (+, −, +, −). Swapping or dropping either
    branch changes the captured data."""
    seen = []

    def spy(D, groups, r, B, chunk):
        seen.append(np.asarray(D)[..., 0].copy())
        return np.zeros((B, 1))
    monkeypatch.setitem(E.BOOT, "fly", spy)
    E._share(E._groups((3, 1)), "raw_fly", _SignRng(), 0.0, het, 0.5, 2, 5, S)
    assert len(seen) == 2 and all(np.array_equal(d, np.tile(np.asarray(u)[:, None], (1, S.flies))) for d in seen)
