# tests/brain/test_t_spec.py
"""Spec T.1 / T.2 (T.9.1) / T.3 / T.8 / T.9.6: every T number in t_spec; T's new blocks (judgement 24_400_xxx, smoke
24_409_xxx, gate ② 25_200_000+i and its smoke 25_209_xxx, their training seeds, the set generator's 20261004 and the
cluster OC's 20261005) collide with no declared seed of any other spec module (test_p_spec's collector, R and S
included); the two gate-② P specs differ from each other in the one edit only and from P in their seed blocks; every
seed field S declared is overridden; z_h4 is block h4's C3 z and the guard thresholds are H.4's; no T file holds
another track's block as a literal; the shared measurement key over R_MEASURE_FILES is still R's 3c2699c7…."""
import ast
import dataclasses
import importlib
import json
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.h4_spec import SPEC as H4
from flymon.brain.p_spec import SPEC as P
from flymon.brain.r_spec import SPEC as R
from flymon.brain.s_spec import SPEC as S
from flymon.brain.s_spec import LastSetSpec
from flymon.brain.t_spec import P_C, P_L, SPEC, TSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_t() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/t_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


T_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds)
           | {SPEC.set_rng_seed, SPEC.oc_cluster_seed})
T_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_t_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/t_spec.py") == "flymon.brain.t_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_400_000, 24_400_008)),
                                      select=list(range(24_400_100, 24_400_108)),
                                      report=list(range(24_400_200, 24_400_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_200_000, 25_200_032))
    assert len(SPEC.p.seeds) == P.o.o2_n_seeds
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_209_000 <= s < 25_210_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_409_000, 24_409_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    assert (SPEC.set_rng_seed, SPEC.oc_cluster_seed) == (20261004, 20261005)
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """T.9.6 (P3-11): T's blocks, the generator and OC seeds and the derived training seeds against every other
    declared block (S, R, P, Q, the encoder, H.4 ... d6a)."""
    declared = _declared_without_t()
    for s in (500, 615, 24_002_000, 24_100_000, 24_300_000, 24_309_000, 25_000_000, 25_100_000, 25_109_000, 23_000_000,
              24_000_000, 22_000_000):
        assert s in declared, s
    assert not T_SEEDS & declared
    assert not {s for s in T_SEEDS if (s - BASE) // STRIDE in declared}
    assert not T_TRAIN & declared
    assert not {s for s in T_TRAIN if (s - BASE) // STRIDE in declared}
    assert not T_TRAIN & T_SEEDS


def test_every_s_seed_field_is_overridden():
    s_seed_fields = {f.name for f in dataclasses.fields(LastSetSpec) if "seed" in f.name} | {"p", "p_c"}
    for name in s_seed_fields:
        assert getattr(SPEC, name) != getattr(S, name), name
    fields = _module_seeds("flymon.brain.t_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & set(R.p.seeds) and not fields & set(range(500, 616))
    assert not fields & (set(S.judge_act_seeds) | set(S.smoke_seeds) | set(S.p.seeds))


def test_gate2_specs_differ_in_the_one_edit_only():
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
    assert isinstance(SPEC, LastSetSpec) and isinstance(SPEC, TSpec)
    assert (SPEC.opp_num_max, SPEC.n_opp, SPEC.n_combos, SPEC.first_turn, SPEC.set_last_turn) == (151, 127, 1986, 0,
                                                                                                    1985)
    assert (SPEC.n_b, SPEC.n_a, SPEC.last_turn_b, SPEC.last_turn, SPEC.n_set_odours) == (21, 43, 17, 103, 55)
    assert SPEC.digest_e0_b == "8c9729bfba529a7bea8d783a629c1c1f54380af79ffc8032e46f657d669a0c71"
    assert SPEC.digest_e0_a == "e31b552636bc237f1d9fc85034432849b8a922e0d8b5fd17b1f87274b725e989"
    assert SPEC.digest_keys == "37dde1ca0c9d0bf33772b42359e45257726b46d552837ffb4cde75e8beb03c97"
    assert dict(SPEC.skipped_declared) == dict(cap=9, collision=0, used=164, glom_dup=0, in_set=48, pool_only=57)
    assert sum(n for *_, n in SPEC.clusters_b) == 21 and len(SPEC.clusters_b) == 9
    assert sum(n for _, n in SPEC.clusters_a) == 43 and len(SPEC.clusters_a) == 8
    assert SPEC.z_h4_dict() == {"A": (10.78125, 9.412096743243064), "P": (26.25, 19.30889259728101)}
    assert (SPEC.z_guard_med_min, SPEC.z_guard_zero_max, SPEC.z_ddof) == (5.0, 0.25, 0) == (
        H4.react_med_delta_min, H4.react_zero_share_max, H4.z_ddof)
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.testable_min, SPEC.naive_max) == (11, 2, 2, 2.0, 0.5)
    assert (SPEC.g_fail_drop, SPEC.g_fail_flips, SPEC.p_ratio_min, SPEC.c_even_expected, SPEC.r_even_L_h4) == (
        3, None, 0.5, 7, 16)
    assert SPEC.valid_band == (0.03, 0.15) and SPEC.kc_seeds() == tuple(E.strength_seeds)
    assert SPEC.gate2_oc_rhos == (0.4, 0.5, 0.61, 0.83, 1.0) and SPEC.oc_harm == S.oc_harm
    assert (SPEC.oc_cluster_icc, SPEC.oc_cluster_draws) == (0.3, 100_000)
    assert SPEC.oc_cluster_g == (("null", 0.1, 0.1), ("harm", 0.1, 0.02), ("harm", 0.1, 0.05))
    assert SPEC.r_shared_key == "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    assert (SPEC.r_summary, SPEC.r_cache_dir, SPEC.s_summary) == ("results/summary/r_lever.json", "results/r/cache",
                                                                  "results/summary/s_lever.json")
    assert SPEC.r_reused == ("repro", "gate1")
    assert (SPEC.summary, SPEC.cache_dir, SPEC.archive_root, SPEC.z_detail) == (
        "results/summary/t_lever.json", "results/t/cache", "~/flymon-archive/t", "results/t/z.json")
    assert SPEC.conditions() == R.conditions() and SPEC.cond_names == ("L", "C", "E0")
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "lever_edges",
              "config", "strength", "e0_strength", "oc_q", "oc_c", "oc_naive_max", "sat_fracs", "smoke_pairs"):
        assert getattr(SPEC, f) == getattr(R, f), f


def test_z_h4_is_block_h4s_c3_z():
    from flymon.agent.config import load_c3_config
    assert load_c3_config(SPEC.m0d_summary).z == SPEC.z_h4_dict()
    h4 = json.loads((ROOT / SPEC.m0d_summary).read_text())["h4"]["h4"]["combos"]["C3"]["z"]
    assert {k: tuple(v) for k, v in h4.items()} == SPEC.z_h4_dict()


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(TSpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4 and SM.p.o.on_edit == SPEC.lever_edit and SM.p_c.o.on_edit == SPEC.no_edit


ALLOWED_T_SPEC = {151, 127, 1986, 20261004, 0, 1985, 43, 17, 103, 9, 164, 48, 57, 55, 1, 3, 4, 7, 8, 2, 5,
                  10.78125, 9.412096743243064, 26.25, 19.30889259728101, 24_400_000, 24_400_008, 24_400_100,
                  24_400_108, 24_400_200, 24_400_208, 24_409_000, 24_409_100, 16, 20261005, 0.3, 100_000, 0.1, 0.02,
                  0.05, 25_200_000, 25_209_000}


def test_t_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/t_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_T_SPEC, nums - ALLOWED_T_SPEC


def test_no_t_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("t_*.py")) + sorted((ROOT / "scripts").glob("run_t*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_300_000 <= i < 24_310_000
               or 24_002_000 <= i < 24_003_000 or 25_000_000 <= i < 25_010_000 or 25_100_000 <= i < 25_110_000
               or 23_000_000 <= i < 23_010_000 or 800_000 <= i < 800_200}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_measurement_key_is_still_rs():
    """T.3 1 / T.8: the key over r_measure.R_MEASURE_FILES equals the key every R block carries; T edits none."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
