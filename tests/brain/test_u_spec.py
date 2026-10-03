# tests/brain/test_u_spec.py
"""Spec U.1-U.3 / U.6 / U.9: every U number in u_spec; U's new blocks (judgement 24_500_xxx, smoke 24_509_xxx, gate ②
25_300_000+i and its smoke 25_309_xxx, their training seeds) collide with no declared seed of any other spec module (T
included); every seed field T declared is overridden while T's set and cluster OC are inherited; the lever is
unresolved until at(spec, f) sets u_edit(f) on L and on the lever P spec; the two gate-② P specs differ in the one edit
only; no U file holds another track's block as a literal; the shared key is still R's."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.p_spec import SPEC as P
from flymon.brain.t_spec import SPEC as T
from flymon.brain.t_spec import TSpec
from flymon.brain.u_measure import BLOCKS, F_GRID, u_edit
from flymon.brain.u_spec import P_C, P_L, SPEC, UNRESOLVED, USpec, at, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000


def _declared_without_u() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/u_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


U_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds))
U_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_u_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/u_spec.py") == "flymon.brain.u_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_500_000, 24_500_008)),
                                      select=list(range(24_500_100, 24_500_108)),
                                      report=list(range(24_500_200, 24_500_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_300_000, 25_300_032))
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_309_000 <= s < 25_310_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_509_000, 24_509_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """U's blocks and their derived training seeds against every other declared block (T, S, R, P, Q, the encoder,
    H.4 ... d6a)."""
    declared = _declared_without_u()
    for s in (500, 615, 24_002_000, 24_300_000, 24_400_000, 24_409_000, 25_000_000, 25_100_000, 25_200_000,
              25_209_000, 23_000_000, 24_000_000, 22_000_000):
        assert s in declared, s
    assert not U_SEEDS & declared
    assert not {s for s in U_SEEDS if (s - BASE) // STRIDE in declared}
    assert not U_TRAIN & declared
    assert not {s for s in U_TRAIN if (s - BASE) // STRIDE in declared}
    assert not U_TRAIN & U_SEEDS


def test_every_t_seed_field_is_overridden_and_ts_set_and_oc_numbers_are_inherited():
    t_seed_fields = {f.name for f in dataclasses.fields(TSpec) if "seed" in f.name} | {"p", "p_c"}
    for name in t_seed_fields:
        assert getattr(SPEC, name) != getattr(T, name), name
    assert SPEC.set_rng_seed == () == SPEC.oc_cluster_seed           # T's set / cluster OC via T's spec
    assert (T.set_rng_seed, T.oc_cluster_seed) == (20261004, 20261005)
    for name in ("digest_e0_b", "digest_e0_a", "digest_keys", "last_turn",
                 "last_turn_b", "n_a", "skipped_declared", "clusters_b", "clusters_a", "n_set_odours",
                 "oc_cluster_fixture", "oc_cluster_icc", "oc_cluster_draws", "oc_cluster_g", "z_h4"):
        assert getattr(SPEC, name) == getattr(T, name), name
    fields = _module_seeds("flymon.brain.u_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & (set(T.judge_act_seeds) | set(T.smoke_seeds) | set(T.p.seeds))


def test_the_lever_is_unresolved_until_at():
    assert SPEC.lever_edit == UNRESOLVED == "" and SPEC.f is None and SPEC.p is P_L and SPEC.p_c is P_C
    assert SPEC.cond("L").edit == ""
    for f in F_GRID + (0.0, 1.0):
        s = at(SPEC, f)
        assert (s.f, s.lever_edit, s.cond("L").edit, s.p.o.on_edit) == (f, u_edit(f), u_edit(f), u_edit(f))
        assert s.cond("C").edit == s.cond("E0").edit == "none" == s.p_c.o.on_edit
        assert s.p.seeds == SPEC.p.seeds and s.lever_edges == 2
        assert smoke(s).p.o.on_edit == u_edit(f) and at(SM, f).p.o.on_edit == u_edit(f)
    for bad in (0.25, 0.5000001):
        with pytest.raises(ValueError):
            at(SPEC, bad)


def test_gate2_specs_differ_in_the_one_edit_only():
    s = at(SPEC, 0.3)
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(s.p.o, f.name) != getattr(P_C.o, f.name)}
    assert diff == {"o1_conditions"}
    assert s.p.o.o1_conditions[1:] == P_C.o.o1_conditions[1:] == P.o.o1_conditions[1:]
    diff_p = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_C.o, f.name) != getattr(P.o, f.name)}
    assert diff_p == {"o2_seed0", "smoke_seed0"}
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(s.p, f.name) == getattr(P_C, f.name) == getattr(P, f.name), f.name


def test_numbers():
    assert isinstance(SPEC, TSpec) and isinstance(SPEC, USpec)
    assert SPEC.f_grid == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) == F_GRID
    assert (SPEC.f_ends, SPEC.n_candidates_max, SPEC.path_even_n) == ((0.0, 1.0), 3, 3)
    assert SPEC.sha_none == "1aee839811b8aa662588fbfab3050cee00962ddc8ecd4d2fd1361fd0c72692d5"
    assert SPEC.sha_zero == "860cba4f9eead9d7a85b632f2c93c51790c5082fe80460975cb1300239c73e46"
    assert (SPEC.t_none_edit, SPEC.t_lever_edit) == ("none", T.lever_edit) == ("none", "apl_to_mbon05_zero")
    assert SPEC.t_z_detail_sha256 == "fc8bb380eb4050c286624bf697422d6b57cb109ba55b13f91bf7d5f6c127762c"
    assert SPEC.t_measure_key_t == "7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374"
    assert (SPEC.t_summary, SPEC.t_z_detail, dict(SPEC.t_commits)) == (
        "results/summary/t_lever.json", "results/t/z.json", {"z": "0eb642e"})
    assert SPEC.chain_types == ("MBON09", "MBON11", "MBON01", "MBON03", "CRE055", "MBON30", "LHMB1")
    assert SPEC.kc_types == ("KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m")
    assert SPEC.contrast_declared() == {
        "out_block": {"MBON03->MBON13": 4, "CRE055->MBON13": 17},
        "chain_entry": {"MBON05->MBON09": 7, "MBON05->MBON11": 2, "MBON05->MBON01": 2}}
    assert {b: {f"{a}->{t}" for a, t in prs} for b, prs in BLOCKS.items()} == {
        b: set(v) for b, v in SPEC.contrast_declared().items()}
    assert sum(SPEC.contrast_declared()["chain_entry"].values()) == 11
    assert SPEC.contrast_points == (("block_f1", 1.0, "out_block"), ("block_f0", 0.0, "out_block"),
                                    ("entry_f0", 0.0, "chain_entry"))
    assert (SPEC.chain_support_max, SPEC.chain_against_min, SPEC.kc_side_delta_max) == (0.5, 0.8, 1.0)
    assert (SPEC.even_drop_b_lt, SPEC.even_drop_a_le) == (3, 1)
    assert SPEC.oc_cluster_sha256 == "83a4d0bdce784536908e3b0e2e00efe8cb6610dec43ee6fca2be4060c7ecffb0"
    assert (SPEC.r_reused, dict(SPEC.r_commits)) == (("repro",), {"repro": "22c934b", "gate3": "a83977f"})
    assert (SPEC.summary, SPEC.cache_dir, SPEC.smoke_cache_dir, SPEC.archive_root) == (
        "results/summary/u_lever.json", "results/u/cache", "results/u/smoke/cache", "~/flymon-archive/u")
    assert (SPEC.path_detail, SPEC.scan_detail, SPEC.smoke_detail) == (
        "results/u/path.json", "results/u/scan.json", "results/u/smoke.json")
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.g_fail_drop, SPEC.p_ratio_min, SPEC.c_even_expected) == (
        11, 2, 2, 3, 0.5, 7)
    assert (SPEC.z_guard_med_min, SPEC.z_guard_zero_max, SPEC.z_ddof) == (5.0, 0.25, 0)
    assert SPEC.valid_band == (0.03, 0.15) and SPEC.kc_seeds() == tuple(E.strength_seeds) and SPEC.n_calib == 112
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "lever_edges", "config",
              "strength", "e0_strength", "oc_q", "oc_c", "oc_naive_max", "smoke_pairs", "r_shared_key", "r_cache_dir",
              "r_summary", "gate2_oc_rhos", "oc_harm", "r_even_L_h4"):
        assert getattr(SPEC, f) == getattr(T, f), f


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(USpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4


ALLOWED_U_SPEC = {25_300_000, 25_309_000, 0.0, 1.0, 3, 4, 17, 7, 2, 0.5, 0.8, 1, 24_500_000, 24_500_008, 24_500_100,
                  24_500_108, 24_500_200, 24_500_208, 24_509_000, 24_509_100}


def test_u_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/u_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_U_SPEC, nums - ALLOWED_U_SPEC


def test_no_u_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("u_*.py")) + sorted((ROOT / "scripts").glob("run_u*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_300_000 <= i < 24_310_000
               or 24_400_000 <= i < 24_410_000 or 24_002_000 <= i < 24_003_000 or 25_000_000 <= i < 25_010_000
               or 25_100_000 <= i < 25_110_000 or 25_200_000 <= i < 25_210_000 or 23_000_000 <= i < 23_010_000
               or 800_000 <= i < 800_200 or i in (20261004, 20261005)}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_and_t_measurement_keys_are_still_rs_and_ts():
    """U.3 1 / U.8: the key over R_MEASURE_FILES is R's and the T measurement key is T's z block's; U edits neither."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
    assert t_measure_key(str(NPZ))["key"] == SPEC.t_measure_key_t
