"""Y's 0p numbers (Y.2 · Y.3.2 · Y.3.3 · Y.7 0p): thresholds, keys, the main-set facts, seed roots that collide with
nothing declared, and no Y file but y_spec holding a number (literal guard)."""
import ast
import dataclasses
import importlib
import json
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC, YSpec
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]


def test_y_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/y_spec.py") == "flymon.brain.y_spec"


def test_thresholds():
    assert (SPEC.naive_max, SPEC.c_a, SPEC.c_p) == (0.5, 20.0, 43.0)
    assert (SPEC.lenient_d, SPEC.lenient_a, SPEC.lenient_p) == (1.0, 16.0, 34.4)
    assert round(0.8 * SPEC.c_a, 9) == SPEC.lenient_a and round(0.8 * SPEC.c_p, 9) == SPEC.lenient_p
    assert SPEC.round_digits == 9 and V.testable_min == 2.0
    assert SPEC.f2_0 == (8.875, 51.640625) and SPEC.f2_25 == (30.7421875, 94.421875)
    assert (SPEC.user_a, SPEC.user_onesided_d) == (31.0, -1.0)
    assert SPEC.k_ranges == ((4, 8), (6, 10)) and min(lo for lo, _ in SPEC.k_ranges) == 4
    assert SPEC.budget_h == 24.0


def test_y92_yield_and_sentence_facts():
    """Y.9.2 P1-7: c = 1 / (2/3) from W pilot 16 -> 3 oracle-lenient -> 2 pilot-strict; P1-4 quantiles; Y.8 facts."""
    assert SPEC.yield_c == 1.5 and SPEC.yield_basis_n == 16 and len(SPEC.yield_basis_pairs) == 3
    assert SPEC.yield_c == len(SPEC.yield_basis_pairs) / sum(ok for _, ok in SPEC.yield_basis_pairs)
    assert SPEC.level_quantiles == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert (SPEC.pilot_w, SPEC.pilot_v, SPEC.pilot_min_sigma) == (3, 4, 5)
    assert SPEC.pool_turns == (W.first_turn, W.last_turn) == (306, 1985)


def test_f2_values_are_xs_precheck_diag():
    """Task 1 review M3: the F2 columns come from X's precheck_diag.json sha256 33f83895…; missing fails."""
    from flymon.brain.h3_store import sha256_file
    p = ROOT / SPEC.x_precheck_diag
    assert p.exists(), f"{SPEC.x_precheck_diag} (X detail, git-ignored) is required"
    assert sha256_file(p) == SPEC.x_precheck_diag_sha256
    assert SPEC.x_precheck_diag_sha256.startswith("33f83895")
    f = json.loads(p.read_text())["filters"]
    assert SPEC.f2_0 == (f["F2(0)"]["pass_counts"]["c_A"], f["F2(0)"]["pass_counts"]["c_P"])
    assert SPEC.f2_25 == (f["F2(25)"]["pass_counts"]["c_A"], f["F2(25)"]["pass_counts"]["c_P"])


def test_keys_and_main_set_facts():
    assert SPEC.u_measure_key == W.u_measure_key_u == \
        "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    assert SPEC.w_measure_key == "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    assert SPEC.main_digest_keys == "65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a"
    assert (SPEC.main_n_b, SPEC.main_n_a, SPEC.main_last_turn) == (167, 82, 1967)
    assert (SPEC.summary, SPEC.raw_dir, SPEC.cache_dir, SPEC.oracle_detail) == (
        "results/summary/y_learning.json", "results/y", "results/y/cache", "results/y/oracle.json")


def test_seed_roots_are_new():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {60_000_000, 61_000_000, 62_000_000, 64_000_000, 77_000_000, 77_100_000, 77_110_000,
                   77_150_000, 77_200_000, 77_300_000, 78_000_000, 78_100_000, 78_200_000}
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(YSpec))
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/y_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    assert not any(60_000_000 <= s < 78_300_000 for s in declared)
    o = W.oracle_seeds()
    assert not any(60_000_000 <= s < 78_300_000 for s in o["act"] + o["select"] + o["report"])


def _numbers(path: Path) -> set:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def test_no_y_file_but_y_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("y_*.py")) if p.name != "y_spec.py"]
    if (ROOT / "scripts/run_y.py").exists():
        files.append(ROOT / "scripts/run_y.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)


SHARED = ("bar", "band_width", "naive_max", "mech_min", "round_digits", "min_gate_pairs", "min_pass_pairs",
          "p_set_grid", "q_grid", "k_grid", "f_min", "f_max", "k_min", "envelope", "envelope_solo_from", "d_power",
          "p_power", "d_false", "p_false", "oc_reps", "cal_reps", "cal_tol", "cal_iter", "boot_draws", "boot_reps",
          "boot_level", "cluster_grid", "record_dprimes", "oc_chunk", "cal_floor_rule", "chol_jitter",
          "w_rejection_tries", "synth_reps", "synth_null_max", "synth_big_min", "synth_big_dprime",
          "synth_drift_dprime_min", "simple_normal_fs", "simple_normal_reps", "synth_base", "synth_sd", "synth_corr",
          "synth_learn", "synth_drift", "het_scales", "het_low_dprime", "het_all_dprime", "precheck_reps",
          "p26_reps", "budget_h")


def test_shared_numbers_equal_x_spec():
    from flymon.brain.x_spec import SPEC as XS
    for n in SHARED:
        assert getattr(SPEC, n) == getattr(XS, n), n
    assert SPEC.k_cap == max(hi for _, hi in SPEC.k_ranges) == 10 and XS.k_cap == SPEC.k_ranges[0][1] == 8
    assert SPEC.k_min == min(lo for lo, _ in SPEC.k_ranges)


def test_phase_a_numbers():
    assert (SPEC.grid_steps, SPEC.widen, SPEC.refine_delta, SPEC.refine_parts, SPEC.knob_tol) == (
        200, (1, 2, 4), 0.25, 32, 1e-3)
    assert (SPEC.coarse_step_flag, SPEC.tries, SPEC.fill_max) == (0.1, 2000, 0.01)
    assert (SPEC.small_boot_draws, SPEC.floor_share, SPEC.floor_resid_n, SPEC.thr_step) == (50, 0.10, 1024, 0.25)
    assert SPEC.thr_expected == (19.25, 42.0) and (SPEC.cost_margin, SPEC.reconfirm_reps, SPEC.reconfirm_max) == (
        1.3, 1600, 5)
    assert (SPEC.reconfirm_seed, SPEC.small_boot_seed, SPEC.records_seed) == (78_000_000, 78_100_000, 78_200_000)
    assert SPEC.sigma2_scale == 2.0 and SPEC.compare_reps == 4000 and SPEC.mix_reps == 4000
    assert (SPEC.pilot_flies, SPEC.pilot_probes) == (8, 8)
    assert len(SPEC.pilot_w_pairs) == SPEC.pilot_w and len(SPEC.pilot_v_pairs) == SPEC.pilot_v
    from flymon.brain.x_spec import SPEC as XS
    assert SPEC.pilot_w_pairs == XS.balanced_pairs
    assert SPEC.pilot_v_pairs == ("b|17|Thunderbolt vs Nidoran-M|Thunderbolt vs Clefairy", "a|218|Earthquake|Strength",
                                  "a|305|Earthquake|Rock Slide", "a|223|Surf|Earthquake")
    assert SPEC.x_commits == (("stage0", "868771a"), ("precheck", "4b81035"))
    assert [c for _, c in SPEC.w_commits] == ["744d1bc", "5620f95", "5fbc4c8", "f30ae35"]
    assert dict(SPEC.m_table)[0.625] == (3, 4, 4, 5, 5, 6, 7) and len(SPEC.set_fixtures) == 14


def test_pilot_seed_layout_is_ws_with_y_roots():
    import dataclasses as dc
    yw = dc.replace(W, pilot_probe_seed0=SPEC.pilot_probe_seed0, pilot_train_seed0=SPEC.pilot_train_seed0)
    assert yw.pilot_probe_seeds(1, 2, 3) == [60_000_000 + 4_000 + 200 + k for k in range(3)]
    assert yw.pilot_train_base(3, 1) == 61_000_000 + 3 * 40_000 + 20
