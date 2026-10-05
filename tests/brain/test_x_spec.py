"""X's numbers (X.2 · X.3 · X.4.1 · X.9.1): W's shared numbers unchanged, X's own grids / roots / calibration rule /
diagnostic grid, the W facts X re-checks; X's seed roots collide with nothing declared; no X file but x_spec holds a
number (literal guard)."""
import ast
import dataclasses
import importlib
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.w_spec import SPEC as W
from flymon.brain.x_spec import SPEC, W_SHARED, XSpec
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
BASE, STRIDE = 1_000_000, 1000


def test_x_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/x_spec.py") == "flymon.brain.x_spec"


def test_w_shared_numbers_are_ws():
    for f in W_SHARED:
        assert getattr(SPEC, f) == getattr(W, f), f
    assert {"q_grid", "k_grid", "f_min", "f_max", "k_min", "k_cap", "cal_reps", "cal_tol", "cal_iter", "oc_chunk",
            "naive_max", "bar", "band_width", "round_digits", "mech_min", "min_gate_pairs", "cal_floor_rule",
            "cluster_grid", "d_power", "d_false", "p_power", "p_false", "oc_reps", "record_dprimes"} <= set(W_SHARED)


def test_x_numbers():
    assert SPEC.p_set_grid == (0.5, 0.625, 0.75, 0.875, 1.0) and SPEC.min_pass_pairs == 3
    assert (SPEC.bracket_mult, SPEC.w_bracket_mult, SPEC.cal_retry_iter) == (4.0, 1.0, 160)
    assert SPEC.w_rejection_tries == 200 and SPEC.chol_jitter == 1e-9
    assert (SPEC.precheck_reps, SPEC.diag_reps, SPEC.p26_reps) == (4000, 4000, 200)
    assert SPEC.het_scales == (0.5, 1.0, 2.0) and (SPEC.het_low_dprime, SPEC.het_all_dprime) == (0.5, 1.0)
    assert SPEC.variants == ("V0", "V1", "V2") and SPEC.v1_exclude_from == 5.0 and SPEC.min_variant_pairs == 6
    assert SPEC.v1_pairs == 7 and len(SPEC.balanced_pairs) == 3
    assert SPEC.resid_quantiles == (5.0, 25.0, 50.0, 75.0, 95.0)
    assert (SPEC.diag_pairs, SPEC.diag_f_max, SPEC.diag_fs, SPEC.diag_k_max) == (16, 64, (8, 16, 32, 64), 16)
    assert SPEC.diag_k_los == (4, 6, 8, 10, 12) and SPEC.diag_k_width == 4 and SPEC.diag_pairs >= SPEC.diag_k_max
    assert (SPEC.diag_floor_share, SPEC.diag_tries, SPEC.diag_fill_flag) == (0.10, 2000, 0.01)
    assert SPEC.diag_levels == (0.0, 25.0, 50.0, 75.0)
    assert SPEC.w_measure_key == "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    assert SPEC.w_pipeline_key == "9636cf4c7f868338f512cd7cf1a722027bd842c37528188859c14bebb04a2853"
    assert SPEC.w_measure_file_sha == "e23009146ee04e3f6f328445a51685f8f045f6bb194749bf4ebd8f80c6e2be63"
    assert SPEC.w_oc_detail_sha == "7a0a65f3f6b969d47b29f9a41ae45f2d1cfc4bf9bb76d028bf7172a9a6fbeb77"
    assert dict(SPEC.w_commits) == {"pilot": "5fbc4c8", "oc": "f30ae35"} and SPEC.w_summary_commit == "f30ae35"
    assert (SPEC.w_boot_fail_total, SPEC.w_boot_fail_power, SPEC.w_boot_fail_false, SPEC.w_boot_fail_both) == (
        29, 14, 18, 3)
    assert (SPEC.w_pilot_pairs, SPEC.w_pilot_units) == (16, 384)
    assert (SPEC.summary, SPEC.raw_dir, SPEC.archive_root) == ("results/summary/x_learning.json", "results/x",
                                                              "~/flymon-archive/x")
    assert SPEC.diag_detail == "results/x/precheck_diag.json"


def test_seed_roots_are_new():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {43_000_000, 43_200_000, 43_300_000}
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(XSpec))
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/x_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    _collect(importlib.import_module("flymon.brain.p_spec").SPEC, declared, seen)
    assert not any(43_000_000 <= s < 43_400_000 for s in declared)
    assert not any((s - BASE) // STRIDE in declared for s in range(43_000_000, 43_400_000, STRIDE))


def _numbers(path: Path) -> set:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def test_no_x_file_but_x_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("x_*.py")) if p.name != "x_spec.py"]
    if (ROOT / "scripts/run_x.py").exists():
        files.append(ROOT / "scripts/run_x.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
