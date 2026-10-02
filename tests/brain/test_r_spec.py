# tests/brain/test_r_spec.py
"""Spec R.2 / R.8 / R.9: every R number in r_spec; R's new blocks (P replication 25_000_000+i, P smoke 25_009_xxx, R
smoke 25_008_xxx) collide with no declared seed and no declared seed's training seeds (test_p_spec's collector); the
reused blocks (encoder ③ 24_002_xxx, H.4 500-615, judgement 24_100_xxx) are exactly their owners' and live in methods,
so the encoder track's collision test (field defaults of every brain *_spec module) never sees them; R's P spec differs
from P's only in its seeds and the edit p_judge's gate expects; no R file holds a reused block as a literal."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.p_spec import SPEC as P
from flymon.brain.q_spec import SPEC as Q
from flymon.brain.r_spec import R_P, SPEC, Cond, LeverSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_r() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/r_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


R_SEEDS = set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.smoke_seeds)
R_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds())


def test_r_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/r_spec.py") == "flymon.brain.r_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.p.seeds == tuple(range(25_000_000, 25_000_032))
    assert len(SM.p.seeds) == 4 and all(25_009_000 <= s < 25_010_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(25_008_000, 25_008_100))
    sm = set(SM.kc_seeds()) | {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 9


def test_new_blocks_collide_with_nothing_declared():
    declared = _declared_without_r()
    for s in (500, 615, 24_002_000, 24_100_000, 24_100_207, 23_000_000, 24_000_000, 24_008_000, 22_000_000):
        assert s in declared, s
    assert not R_SEEDS & declared
    assert not {s for s in R_SEEDS if (s - BASE) // STRIDE in declared}     # no declared s trains on an R seed
    assert not R_TRAIN & declared
    assert not {s for s in R_TRAIN if (s - BASE) // STRIDE in declared}
    assert not R_TRAIN & R_SEEDS


def test_reused_blocks_are_read_from_their_owners():
    assert SPEC.kc_seeds() == E.strength_seeds == tuple(range(24_002_000, 24_002_008))
    h4 = dict(act=list(range(500, 508)), select=list(range(600, 608)), report=list(range(608, 616)))
    assert SPEC.h4_seeds() == SPEC.even_seeds() == h4 == SM.h4_seeds()
    assert SPEC.judge_seeds() == dict(act=list(E.judge_act_seeds), select=list(E.judge_select_seeds),
                                      report=list(E.judge_report_seeds))
    assert SPEC.p_repro_seeds() == (23_000_000, 23_000_015, 23_000_031) and set(SPEC.p_repro_seeds()) <= set(P.seeds)
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_no_encoder_seed_sits_in_an_r_field():
    assert not _module_seeds("flymon.brain.r_spec") & e_spec.track_seeds(E)


def test_gate2_spec_differs_from_p_only_in_seeds_and_the_edit():
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(R_P.o, f.name) != getattr(P.o, f.name)}
    assert diff == {"o2_seed0", "smoke_seed0", "o1_conditions"}
    assert R_P.o.on_edit == SPEC.lever_edit == "apl_to_mbon05_zero"
    assert R_P.o.o1_conditions[1:] == P.o.o1_conditions[1:]
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(R_P, f.name) == getattr(P, f.name), f.name
    assert SPEC.p is R_P and R_P.pairs() == P.pairs() and R_P.flags() == P.flags()


def test_conditions_and_numbers():
    L, C, E0 = SPEC.conditions()
    assert L == Cond("L", "apl_to_mbon05_zero", "egrid", 1.0)
    assert C == Cond("C", "none", "egrid", 1.0)
    assert E0 == Cond("E0", "none", "e0", E.e0_strength) and E.e0_strength == 0.35
    assert SPEC.cond("C") == C and SPEC.cond_names == ("L", "C", "E0")
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.n_b, SPEC.n_a, SPEC.n_a_even) == (11, 2, 2, 21, 32, 18)
    assert SPEC.valid_band == Q.kc_band == (0.03, 0.15)
    assert (SPEC.g_fail_drop, SPEC.g_fail_flips, SPEC.c_even_expected, SPEC.lever_edges) == (2, 3, 7, 2)
    assert (SPEC.settle_ms, SPEC.read_ms, SPEC.window_ms, SPEC.alphas) == (800.0, 600.0, 200, (0.2, 0.5, 0.8))
    assert SPEC.fixed_alphas == () and SPEC.last_turn == 103 and SPEC.n_calib == 112
    assert SPEC.digest_e0_b.startswith("4b9bccc9") and SPEC.digest_e0_a.startswith("78509547")
    assert SPEC.digest_keys.startswith("33be39a1")


def test_paths_match_their_owners():
    from flymon.brain import p_cli, p_measure
    assert SPEC.p_summary == p_measure.SUMMARY
    assert SPEC.p_cache_dir == p_cli.RUN_OUT + "/cache/p_arm"
    assert SPEC.encoder_summary == Q.encoder_summary
    assert SPEC.m0d_summary == E.m0d_summary


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(LeverSpec):
        if f.name not in ("p", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.p.o.on_edit == SPEC.lever_edit and SM.workers == 4


ALLOWED_R_SPEC = {0, 1, 2, 3, 4, 6, 7, 9, 15, 16, 18, 20, 31, 32, 103, 112, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0,
                  95.0, 1000.0, 3600.0, 25_000_000, 25_009_000, 25_008_000, 25_008_100}


def test_r_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/r_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_R_SPEC, nums - ALLOWED_R_SPEC


def test_no_r_file_holds_a_reused_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("r_*.py")) + sorted((ROOT / "scripts").glob("run_r*.py"))
    assert files
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_002_000 <= i < 24_003_000}
        assert not bad, (p.name, bad)
