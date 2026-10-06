"""AA's numbers (AA.2–AA.9.1): the declared values, AA's seed blocks colliding with nothing declared, the sealed subset,
and no AA file but aa_spec holding a number (Z's literal guard)."""
import ast
import dataclasses
import importlib
import math
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.aa_spec import SPEC, AASpec
from flymon.brain.y_spec import SPEC as Y
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]


def test_aa_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/aa_spec.py") == "flymon.brain.aa_spec"


def test_chain_and_reuse_facts():
    assert SPEC.stages == ("stage0", "reuse", "smoke", "seal_code", "screen", "coverage", "learn", "estimate", "records")
    assert dict(SPEC.stage_labels)["coverage"] == "4b" and dict(SPEC.stage_labels)["seal_code"] == "3"
    assert [c for _, c in SPEC.y_blocks] == ["36df057", "2a9425a", "b1279d2", "2b82a8f", "eaffc5a", "279e916",
                                             "547c3b0"]
    assert [c for _, c in SPEC.z_blocks] == ["94c6935", "300938f", "2603508", "403b4a3"]
    assert SPEC.oracle_sha256.startswith("fa4750ac") and (SPEC.n_len, dict(SPEC.n_len_axes)) == (31, {"b": 11, "a": 20})
    assert SPEC.n_main == 249 and SPEC.decl_commit == "c87f4df"
    assert SPEC.y_absent_blocks == ("gates", "learn")


def test_design_and_estimation_numbers():
    assert (SPEC.flies, SPEC.probes) == (8, 8)
    assert (SPEC.winsor, SPEC.boot_b, SPEC.pct, SPEC.min_flies, SPEC.min_groups) == (10.0, 10_000, (2.5, 97.5), 2, 5)
    assert (SPEC.cov_deltas, SPEC.cov_hets, SPEC.cov_het_sd) == ((0.0, 1.0, 1.5), ("none", "group", "pair"), 0.5)
    assert (SPEC.cov_reps, SPEC.cov_b, SPEC.cov_bar) == (1000, 2000, 0.90)
    assert (SPEC.synth_pairs, SPEC.synth_deltas, SPEC.synth_sds, SPEC.synth_reps) == (20, (0.0, 1.0, 1.5), (0.0, 0.5), 1000)
    assert SPEC.floor_cut == 0.10 and SPEC.compare_refs == (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5)
    nu = SPEC.probes - 1
    assert round(math.sqrt(nu / 2) * math.gamma((nu - 1) / 2) / math.gamma(nu / 2), 4) == SPEC.c7_declared == 1.1259
    assert round(1 - 3 / (4 * nu - 1), 4) == SPEC.hedges_j_declared == 0.8889
    assert (SPEC.core_cap_h, SPEC.records_cap_h, SPEC.g6_cap_h, SPEC.cost_margin) == (12.0, 4.0, 2.0, 1.3)
    assert (SPEC.noplast_pairs, SPEC.noplast_flies) == (2, 2)
    assert SPEC.g6_design == (0.5, 0.5, 16, 8, (4, 8)) and SPEC.d_false == Y.d_false
    assert SPEC.set_flow == (("S1", "pool"), ("S31", "pool31"), ("out", "pool_out"), ("sens", "sens_floor"))


def test_aa_seed_blocks_collide_with_nothing_declared():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    roots = {SPEC.boot_seed, SPEC.smoke_probe_seed0, SPEC.smoke_train_seed0, SPEC.g6_seed, SPEC.synth_seed}
    assert roots == {87_000_000, 87_100_000, 87_110_000, 87_200_000, 87_300_000} and roots <= out
    assert all(87_000_000 <= x <= 87_400_000 for x in out)        # the roots and the declared block bounds only
    assert all(any(lo <= r < hi for lo, hi in SPEC.seed_blocks) for r in roots)
    for (a, b), (c, d) in zip(SPEC.seed_blocks, SPEC.seed_blocks[1:]):
        assert b <= c
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/aa_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    assert not any(lo <= s < hi for s in declared for lo, hi in SPEC.seed_blocks)
    # smoke training seeds stay in the smoke block (f < 32, t < 40)
    assert SPEC.smoke_train_seed0 + 31 * 1_000 + 39 < SPEC.seed_blocks[1][1]
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(AASpec))


def test_seal_fields_are_the_estimation_numbers():
    f = SPEC.seal_fields()
    for k in ("winsor", "boot_b", "pct", "boot_chunk", "min_flies", "min_groups", "boot_seed", "synth_seed",
              "set_flow", "cov_deltas", "cov_hets", "cov_het_sd", "cov_reps", "cov_b", "cov_bar", "floor_cut",
              "g6_rows", "g6_scales", "d_false", "g6_seed", "compare_refs"):
        assert k in f, k
    assert "flymon/brain/aa_estimate.py" in SPEC.seal_files and "flymon/brain/aa_spec.py" in SPEC.seal_files


def test_seal_files_cover_aa_estimate_imports():
    """AA.7 3: the seal hashes aa_estimate and every flymon file it imports (transitively inside flymon.brain)."""
    seen, todo = set(), ["flymon/brain/aa_estimate.py"]
    while todo:
        p = todo.pop()
        if p in seen or not (ROOT / p).exists():
            continue
        seen.add(p)
        for n in ast.walk(ast.parse((ROOT / p).read_text())):
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                names = [n.module] if n.module else [a.name for a in n.names]
                todo += [f"flymon/brain/{m}.py" for m in names]
    assert seen <= set(SPEC.seal_files), sorted(seen - set(SPEC.seal_files))


def _numbers(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            out.add(n.value)
    return out


def test_no_aa_file_but_aa_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("aa_*.py")) if p.name != "aa_spec.py"]
    if (ROOT / "scripts/run_aa.py").exists():
        files.append(ROOT / "scripts/run_aa.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
