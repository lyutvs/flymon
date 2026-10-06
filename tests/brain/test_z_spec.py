"""Z's numbers (Z.2, Z.3, Z.7, Z.9.2): the gate / rule numbers as declared, Z's seed blocks colliding with nothing
declared (W · earlier appendices · X · Y), the roots Z reuses only for Y's reproduction kept out of the collectors,
and no Z file but z_spec holding a number (literal guard)."""
import ast
import dataclasses
import importlib
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.y_spec import SPEC as Y
from flymon.brain.z_spec import SPEC, ZSpec
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]


def test_z_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/z_spec.py") == "flymon.brain.z_spec"


def test_order0_numbers():
    assert SPEC.order0 == SPEC.stages[:5] == ("stage0", "ydiag", "reuse", "sens", "split")
    assert (SPEC.n_len, dict(SPEC.n_len_axes)) == (31, {"b": 11, "a": 20})
    assert SPEC.oracle_sha256.startswith("fa4750ac") and SPEC.decl_commit == "7c4bde5"
    assert [c for _, c in SPEC.y_blocks] == ["36df057", "2a9425a", "b1279d2", "2b82a8f", "eaffc5a", "279e916",
                                             "547c3b0"]
    assert SPEC.y_admitted == Y.pilot_w_pairs + Y.pilot_v_pairs[:3] and SPEC.y_n_sigma == 5
    assert (SPEC.y_cells_n, SPEC.redraw_n) == (200, 50)
    assert SPEC.diag_design == (0.5, 0.5, 16, 8, (4, 8))
    assert SPEC.gate_by_k == (0.047, 0.087, 0.159, 0.050, 0.082) and (SPEC.gate_sim, SPEC.gate_false) == (0.047, 0.0)
    assert SPEC.sens_deltas == (0.0, -2.41, -4.82) and SPEC.sens_ns == (12, 24)
    assert (SPEC.sens_draws, SPEC.sens_reps, SPEC.sens_cont_delta, SPEC.plan_bar) == (100, 400, 1, 0.80)
    assert SPEC.plan_bar == Y.p_power
    assert (SPEC.j_min, SPEC.n_star, SPEC.se_max) == (12, (12, 24), 1.62)
    assert SPEC.strata == ("b", "a") and SPEC.halves == ("C", "P")
    assert (SPEC.budget_h, SPEC.records_budget_h, SPEC.cost_margin) == (24.0, 6.0, 1.3)
    assert (SPEC.diag_root, SPEC.psize_root, SPEC.cf_roots) == (Y.oc_seed, Y.oc_seed + 7, (1, 2, 3, 4, 5, 6))


def test_printed_names_are_unique_and_digits_sane():
    names = [n for n, _, _ in SPEC.printed]
    assert len(names) == len(set(names))
    assert all(isinstance(d, int) and 0 <= d <= 3 for _, _, d in SPEC.printed)


def test_z_seed_blocks_collide_with_nothing_declared():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    roots = {SPEC.split_seed, SPEC.pilot_probe_seed0, SPEC.pilot_train_seed0, SPEC.oc_seed, SPEC.smoke_probe_seed0,
             SPEC.smoke_train_seed0, SPEC.smoke_oracle_seed0, SPEC.precheck_seed, SPEC.compare_seed,
             SPEC.reconfirm_seed, SPEC.small_boot_seed, SPEC.records_seed, SPEC.sens_seed}
    assert roots <= out
    assert all(s >= 79_000_000 for s in out)                    # nothing of Z inside W / X / Y's blocks
    assert SPEC.diag_root not in out and SPEC.psize_root not in out      # Y's roots are not Z declarations
    blocks = SPEC.seed_blocks
    assert all(any(lo <= r < hi for lo, hi in blocks) for r in roots)
    for (a, b), (c, d) in zip(blocks, blocks[1:]):
        assert b <= c
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/z_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    assert not any(lo <= s < hi for s in declared for lo, hi in blocks)
    assert SPEC.pilot_train_seed0 + (SPEC.pilot_j_max - 1) * 40_000 + 7 * 1_000 + 39 < 83_000_000   # Z.9.1 해석 3 (f ≤ 7, t < 40)
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(ZSpec))


def _numbers(path: Path) -> set:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def test_no_z_file_but_z_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("z_*.py")) if p.name != "z_spec.py"]
    if (ROOT / "scripts/run_z.py").exists():
        files.append(ROOT / "scripts/run_z.py")
    assert files
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
