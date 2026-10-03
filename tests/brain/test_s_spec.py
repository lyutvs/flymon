# tests/brain/test_s_spec.py
"""Spec S.2 / S.3 / S.8 / S.9.5 / S.9.6: every S number in s_spec; S's new blocks (judgement 24_300_xxx, smoke
24_309_xxx, gate ② 25_100_000+i and its smoke 25_109_xxx, and their training seeds) collide with no declared seed of
any other spec module (test_p_spec's collector, R included); the two gate-② P specs differ from each other in the one
edit only and from P in their seed blocks; R's reused blocks are not S fields; no S file holds R's judgement, gate-①
or gate-② block as a literal; the shared measurement key over R_MEASURE_FILES is still R's 3c2699c7…."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.p_spec import SPEC as P
from flymon.brain.r_spec import SPEC as R
from flymon.brain.r_spec import LeverSpec
from flymon.brain.s_spec import P_C, P_L, SPEC, LastSetSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_s() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/s_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


S_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds))
S_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_s_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/s_spec.py") == "flymon.brain.s_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_300_000, 24_300_008)),
                                      select=list(range(24_300_100, 24_300_108)),
                                      report=list(range(24_300_200, 24_300_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_100_000, 25_100_032))
    assert len(SPEC.p.seeds) == P.o.o2_n_seeds
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_109_000 <= s < 25_110_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_309_000, 24_309_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """S.9.6: S's blocks and their derived training seeds against every other declared block (R, P, Q, the encoder,
    H.4 ... d6a) — no shared seed, no declared seed training on an S seed, no S training seed declared."""
    declared = _declared_without_s()
    for s in (500, 615, 24_002_000, 24_100_000, 24_100_207, 25_000_000, 25_008_000, 25_009_000, 23_000_000,
              24_000_000, 22_000_000):
        assert s in declared, s
    assert not S_SEEDS & declared
    assert not {s for s in S_SEEDS if (s - BASE) // STRIDE in declared}
    assert not S_TRAIN & declared
    assert not {s for s in S_TRAIN if (s - BASE) // STRIDE in declared}
    assert not S_TRAIN & S_SEEDS


def test_no_encoder_or_r_reused_seed_sits_in_an_s_field():
    fields = _module_seeds("flymon.brain.s_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & set(R.p.seeds) and not fields & set(range(500, 616))
    assert SPEC.judge_seeds() != R.judge_seeds()


def test_gate2_specs_differ_in_the_one_edit_only():
    """S.9.6: P_L and P_C are the same P spec but for o1_conditions[0] (the edit p_judge's gate expects)."""
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_L.o, f.name) != getattr(P_C.o, f.name)}
    assert diff == {"o1_conditions"}
    assert P_L.o.o1_conditions[1:] == P_C.o.o1_conditions[1:] == P.o.o1_conditions[1:]
    assert (P_L.o.on_edit, P_C.o.on_edit) == ("apl_to_mbon05_zero", "none") == (SPEC.lever_edit, SPEC.no_edit)
    diff_p = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_C.o, f.name) != getattr(P.o, f.name)}
    assert diff_p == {"o2_seed0", "smoke_seed0"}
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(P_L, f.name) == getattr(P_C, f.name) == getattr(P, f.name), f.name
    assert SPEC.p is P_L and SPEC.p_c is P_C


def test_numbers():
    assert isinstance(SPEC, LeverSpec) and isinstance(SPEC, LastSetSpec)
    assert (SPEC.first_turn, SPEC.set_last_turn, SPEC.last_turn, SPEC.n_b, SPEC.n_a) == (104, 209, 177, 21, 43)
    assert SPEC.digest_e0_b == "2b92f40a448fe5552598ac8ab265fb5cdc50424e8de945a8a5be587c44553456"
    assert SPEC.digest_e0_a == "7b53467569821e3c3ffdbe2eb317843b22de033041717b9ec8d48fbc40cf8cc7"
    assert SPEC.digest_keys == "884d49cc5d7c29a9a37e17166a1220c52ccbc1675b79f14f7e5abe2e0ca91231"
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.testable_min, SPEC.naive_max) == (11, 2, 2, 2.0, 0.5)
    assert (SPEC.g_fail_drop, SPEC.g_fail_flips, SPEC.p_ratio_min) == (3, None, 0.5)
    assert SPEC.gate2_oc_rhos == (0.4, 0.5, 0.61, 0.83, 1.0)
    assert SPEC.oc_discord == (0.02, 0.05, 0.1, 0.15)
    assert SPEC.oc_harm == ((0.1, 0.02), (0.15, 0.02), (0.2, 0.05), (0.1, 0.05))
    assert SPEC.r_shared_key == "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    assert SPEC.r_summary == R.summary == "results/summary/r_lever.json"
    assert SPEC.r_reused == ("repro", "gate1", "gate3")
    assert dict(SPEC.r_commits) == dict(repro="22c934b", gate1="6ad2201", gate2="78f514a", gate3="a83977f")
    assert (SPEC.summary, SPEC.cache_dir, SPEC.archive_root) == ("results/summary/s_lever.json", "results/s/cache",
                                                                 "~/flymon-archive/s")
    assert SPEC.conditions() == R.conditions() and SPEC.cond_names == ("L", "C", "E0")
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "lever_edges",
              "config", "strength", "e0_strength", "oc_q", "oc_c", "oc_naive_max", "sat_fracs", "smoke_pairs"):
        assert getattr(SPEC, f) == getattr(R, f), f


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(LastSetSpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4 and SM.p.o.on_edit == SPEC.lever_edit and SM.p_c.o.on_edit == SPEC.no_edit


ALLOWED_S_SPEC = {0.4, 0.5, 0.61, 0.83, 1.0, 0.1, 0.02, 0.15, 0.2, 0.05, 3, 43, 104, 177, 24_300_000, 24_300_008,
                  24_300_100, 24_300_108, 24_300_200, 24_300_208, 24_309_000, 24_309_100, 25_100_000, 25_109_000, 4}


def test_s_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/s_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_S_SPEC, nums - ALLOWED_S_SPEC


def test_no_s_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("s_*.py")) + sorted((ROOT / "scripts").glob("run_s*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_002_000 <= i < 24_003_000
               or 25_000_000 <= i < 25_010_000 or 23_000_000 <= i < 23_010_000}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_measurement_key_is_still_rs():
    """S.3 ① / S.9.5: the key over r_measure.R_MEASURE_FILES (+ the connectome, Python and NumPy versions) equals the
    key every R block carries; S edits none of those files."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
