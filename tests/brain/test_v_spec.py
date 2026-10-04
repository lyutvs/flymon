# tests/brain/test_v_spec.py
"""Spec V.1-V.3 / V.6 / V.9: every V number in v_spec; V's new blocks (judgement 24_600_xxx, smoke 24_609_xxx, gate ②
25_400_000+i and its smoke 25_409_xxx, their training seeds) collide with no declared seed of any other spec module (U
and T included); every seed field U declared is overridden while T's generator / cluster seeds stay T's; the lever is
the fixed string u_edit(0.0, "chain_entry") on L and on the lever P spec, and the edit string is in every cache key;
T's declared set values are cleared; the two gate-② P specs differ in the one edit only; no V file holds another
track's block as a literal; the shared, T and V measurement keys are R's, T's and U's."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.config import Params
from flymon.brain.p_spec import SPEC as P
from flymon.brain.r_measure import RMeasurer
from flymon.brain.t_spec import SPEC as T
from flymon.brain.u_measure import u_edit
from flymon.brain.u_spec import SPEC as U
from flymon.brain.u_spec import USpec
from flymon.brain.v_spec import LEVER_V, P_C, P_L, SPEC, VSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000


def _declared_without_v() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/v_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


V_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds))
V_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_v_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/v_spec.py") == "flymon.brain.v_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_600_000, 24_600_008)),
                                      select=list(range(24_600_100, 24_600_108)),
                                      report=list(range(24_600_200, 24_600_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_400_000, 25_400_032))
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_409_000 <= s < 25_410_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_609_000, 24_609_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """V's blocks and their derived training seeds against every other declared block (U, T, S, R, P, Q, the encoder,
    H.4 ... d6a)."""
    declared = _declared_without_v()
    for s in (500, 615, 24_002_000, 24_400_000, 24_500_000, 24_509_000, 25_200_000, 25_300_000, 25_309_000,
              23_000_000, 24_000_000, 22_000_000):
        assert s in declared, s
    assert not V_SEEDS & declared
    assert not {s for s in V_SEEDS if (s - BASE) // STRIDE in declared}
    assert not V_TRAIN & declared
    assert not {s for s in V_TRAIN if (s - BASE) // STRIDE in declared}
    assert not V_TRAIN & V_SEEDS


def test_every_u_seed_field_is_overridden_and_ts_generator_seeds_stay_ts():
    u_seed_fields = {f.name for f in dataclasses.fields(USpec) if "seed" in f.name} - {"set_rng_seed",
                                                                                         "oc_cluster_seed"}
    for name in u_seed_fields | {"p", "p_c"}:
        assert getattr(SPEC, name) != getattr(U, name), name
    assert SPEC.set_rng_seed == () == SPEC.oc_cluster_seed           # V generates with T's spec (v_pairs / v_rules)
    assert (T.set_rng_seed, T.oc_cluster_seed) == (20261004, 20261005)
    fields = _module_seeds("flymon.brain.v_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & (set(U.judge_act_seeds) | set(U.smoke_seeds) | set(U.p.seeds))


def test_ts_declared_set_values_are_cleared():
    assert (SPEC.digest_e0_b, SPEC.digest_e0_a, SPEC.digest_keys) == ("", "", "")
    assert (SPEC.last_turn, SPEC.last_turn_b, SPEC.n_set_odours) == (None, None, None)
    assert SPEC.skipped_declared == SPEC.clusters_b == SPEC.clusters_a == ()
    assert (SPEC.first_turn, SPEC.n_b, SPEC.n_a) == (0, 21, 43)


def test_the_lever_is_the_combined_edit_everywhere():
    assert LEVER_V == u_edit(0.0, "chain_entry") == "u_apl_mbon05_x0.0+chain_entry"
    assert SPEC.lever_edit == LEVER_V and SPEC.cond("L").edit == LEVER_V and SPEC.p.o.on_edit == LEVER_V
    assert SPEC.p is P_L and SPEC.p_c is P_C and P_C.o.on_edit == "none"
    assert SPEC.cond("C").edit == SPEC.cond("E0").edit == "none" and SPEC.lever_edges == 2
    assert smoke(SPEC).p.o.on_edit == LEVER_V


def test_the_edit_is_in_every_cache_key():
    """V.3 2 unit test: the lever string is an input of every KC-activity and oracle cache entry."""
    m = RMeasurer(None, None, SPEC, Params(), {"A": "MBON13", "P": "MBON05"}, {"A": (1.0, 1.0), "P": (1.0, 1.0)},
                  ["MBON13", "MBON05"], 100)
    a = m._act_inputs("X|Y", {"G": 1.0}, SPEC.lever_edit, 1.0, [1], "kc_input")
    n = m._act_inputs("X|Y", {"G": 1.0}, SPEC.no_edit, 1.0, [1], "kc_input")
    assert a["edit"] == LEVER_V and n["edit"] == "none" and a != n
    row = dict(axis="b", turn=0, x="m1", y="m2", odor_x={"G": 1.0}, odor_y={"G": 2.0})
    assert m.inputs(row, SPEC.cond("L"), "even", SPEC.h4_seeds())["edit"] == LEVER_V


def test_gate2_specs_differ_in_the_one_edit_only():
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(SPEC.p.o, f.name) != getattr(P_C.o, f.name)}
    assert diff == {"o1_conditions"}
    diff_p = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_C.o, f.name) != getattr(P.o, f.name)}
    assert diff_p == {"o2_seed0", "smoke_seed0"}
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(SPEC.p, f.name) == getattr(P_C, f.name) == getattr(P, f.name), f.name


def test_numbers():
    assert isinstance(SPEC, USpec) and isinstance(SPEC, VSpec)
    assert SPEC.sha_combined == "2d359b8b6947d542348b658db977e1edbe0db5919e7f4b1187598cb10ef1253a"
    assert SPEC.sha_none == "1aee839811b8aa662588fbfab3050cee00962ddc8ecd4d2fd1361fd0c72692d5"
    assert SPEC.sha_zero == "860cba4f9eead9d7a85b632f2c93c51790c5082fe80460975cb1300239c73e46"
    assert (SPEC.n_lever_edges_total, SPEC.mbon05_apl_edges, SPEC.mbon05_apl_weights, SPEC.mbon05_apl_weight_tol) == (
        13, 2, (-7.865, -12.929), 0.001)
    assert SPEC.contrast_declared()["chain_entry"] == {"MBON05->MBON09": 7, "MBON05->MBON11": 2, "MBON05->MBON01": 2}
    assert SPEC.u_measure_key_u == "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    assert dict(SPEC.u_commits) == {"reuse": "d00fdf2", "path": "3b83824", "scan": "d732b20", "kc": "c0d09a7"}
    assert (SPEC.u_summary, SPEC.u_path_detail, SPEC.u_scan_detail, SPEC.u_entry_point) == (
        "results/summary/u_lever.json", "results/u/path.json", "results/u/scan.json", "entry_f0")
    assert SPEC.u_path_detail_sha256 == "57ae0c4ed11396421e21a91ae0bd5f88a20c2e8215e612a1225d6599b94be329"
    assert SPEC.u_scan_detail_sha256 == "a4cddfd7fbdec1dd8a95e93088aeecff2c1e9eca389454bcd8d81d4f45c0ab7e"
    assert SPEC.t_measure_key_t == "7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374"
    assert SPEC.r_shared_key == "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    assert (SPEC.even_drop_b_lt, SPEC.even_drop_a_le, SPEC.path_even_n) == (3, 1, 3)
    assert (SPEC.summary, SPEC.cache_dir, SPEC.smoke_cache_dir, SPEC.archive_root) == (
        "results/summary/v_lever.json", "results/v/cache", "results/v/smoke/cache", "~/flymon-archive/v")
    assert (SPEC.path_detail, SPEC.kc_input_detail, SPEC.smoke_detail, SPEC.oc_cluster_out) == (
        "results/v/path.json", "results/v/kc_input.json", "results/v/smoke.json", "results/v/oc_cluster.json")
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.g_fail_drop, SPEC.p_ratio_min, SPEC.c_even_expected) == (
        11, 2, 2, 3, 0.5, 7)
    assert (SPEC.z_guard_med_min, SPEC.z_guard_zero_max, SPEC.z_ddof) == (5.0, 0.25, 0)
    assert SPEC.valid_band == (0.03, 0.15) and SPEC.kc_seeds() == tuple(E.strength_seeds) and SPEC.n_calib == 112
    assert SPEC.kc_seeds() == tuple(range(24_002_000, 24_002_008))
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "config", "strength",
              "e0_strength", "oc_q", "oc_c", "oc_naive_max", "smoke_pairs", "r_cache_dir", "r_summary",
              "gate2_oc_rhos", "oc_harm", "r_even_L_h4", "z_h4", "oc_cluster_icc", "oc_cluster_draws",
              "oc_cluster_g"):
        assert getattr(SPEC, f) == getattr(U, f) == getattr(T, f), f
    for f in ("chain_types", "kc_types", "t_z_detail_sha256", "t_summary", "t_z_detail", "t_none_edit",
              "t_lever_edit", "contrast_edges"):
        assert getattr(SPEC, f) == getattr(U, f), f


def test_u_kc_none_reads_one_agreeing_map():
    pts = {k: {"record_set": {"per_odour_none": {"A|B": 0.05}}} for k in SPEC.u_kc_points}
    assert SPEC.u_kc_none({"kc": {"points": pts}}) == {"A|B": 0.05}
    pts["0.7"]["record_set"]["per_odour_none"] = {"A|B": 0.06}
    with pytest.raises(ValueError):
        SPEC.u_kc_none({"kc": {"points": pts}})


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(VSpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4


ALLOWED_V_SPEC = {25_400_000, 25_409_000, 0.0, 13, 2, 4, -7.865, -12.929, 0.001, 0, 43, 3, 1, 24_600_000, 24_600_008,
                  24_600_100, 24_600_108, 24_600_200, 24_600_208, 24_609_000, 24_609_100}


def test_v_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/v_spec.py").read_text())
    nums = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
            nums.add(-n.operand.value)
        elif isinstance(n, ast.Constant) and type(n.value) in (int, float):
            nums.add(n.value)
    assert nums - {7.865, 12.929} <= ALLOWED_V_SPEC, nums - ALLOWED_V_SPEC


def test_no_v_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("v_*.py")) + sorted((ROOT / "scripts").glob("run_v*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_300_000 <= i < 24_310_000
               or 24_400_000 <= i < 24_410_000 or 24_500_000 <= i < 24_510_000 or 24_002_000 <= i < 24_003_000
               or 25_000_000 <= i < 25_010_000 or 25_100_000 <= i < 25_110_000 or 25_200_000 <= i < 25_210_000
               or 25_300_000 <= i < 25_310_000 or 23_000_000 <= i < 23_010_000 or 800_000 <= i < 800_200
               or i in (20261004, 20261005)}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_t_and_v_measurement_keys_are_rs_ts_and_us():
    """V.3 1 / V.9.5 P1-3: the key over R_MEASURE_FILES is R's, the T measurement key is T's z block's and the V
    measurement key (u_measure.u_measure_key, unchanged) is U's literal; V edits none of their files."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import u_measure_key
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
    assert t_measure_key(str(NPZ))["key"] == SPEC.t_measure_key_t
    assert u_measure_key(str(NPZ))["key"] == SPEC.u_measure_key_u
