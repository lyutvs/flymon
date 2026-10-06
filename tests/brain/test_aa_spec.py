"""AA's numbers (AA.2–AA.9.1): the declared values, AA's seed blocks colliding with nothing declared, the sealed subset,
and no AA file but aa_spec holding a number (Z's literal guard)."""
import ast
import hashlib
import json
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
    # AA.2 6172: the oracle detail sha256 is Y block oracle's detail_sha256 (the file is hashed, never parsed)
    y_oracle = json.loads((ROOT / "results/summary/y_learning.json").read_text())["oracle"]
    assert SPEC.oracle_sha256 == y_oracle["detail_sha256"] == "fa4750ac80883bcdc571e8e5b6ba7cc3d25f320549988dc0c3ab8823f75768bb"
    assert hashlib.sha256((ROOT / y_oracle["detail_path"]).read_bytes()).hexdigest() == SPEC.oracle_sha256
    assert (SPEC.n_len, dict(SPEC.n_len_axes)) == (31, {"b": 11, "a": 20})
    assert SPEC.n_main == 249 and SPEC.decl_commit == "c87f4df"
    assert SPEC.y_absent_blocks == ("gates", "learn")
    # AA.7 1 6252: Y's gates / learn blocks and their progress files are absent
    assert SPEC.y_absent_files == ("results/y/gates.json", "results/y/learn.json", "results/y/progress/gates.json",
                                   "results/y/progress/learn.json")
    # AA.0 6143 / Z.10: Y and Z stopped at their last block; every earlier block passed
    assert [b for b, _ in SPEC.y_outcomes] == [b for b, _ in SPEC.y_blocks]
    assert dict(SPEC.y_outcomes) == {"digest": "PASS", "oracle": "PASS", "stage0": "PASS", "reuse": "PASS",
                                     "pilot": "PASS", "precheck": "PASS", "oc": "STOP_OC_UNREACHABLE"}
    assert [b for b, _ in SPEC.z_outcomes] == [b for b, _ in SPEC.z_blocks]
    assert dict(SPEC.z_outcomes) == {"stage0": "PASS", "ydiag": "PASS", "reuse": "PASS",
                                     "sens": "STOP_PLAN_UNREACHABLE"}


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
    assert SPEC.boot_chunk == 1_000 and SPEC.boot_b % SPEC.boot_chunk == 0     # plan Reading (B in chunks of 1 000)
    assert SPEC.ci_level == 0.95 and SPEC.pct == (round(50 * (1 - SPEC.ci_level), 9), round(50 * (1 + SPEC.ci_level), 9))
    assert SPEC.floor_qs == (0.0, 0.25, 0.5, 0.75, 1.0)                     # AA.6 6239: min · 25 · 50 · 75% · max
    assert SPEC.bar_ref == 1.0 and SPEC.bar_ref in SPEC.compare_refs        # AA.9 6284: the ±1 row note
    # AA.5 6231 "집합 셋 × CI 방식"; plan: raw contrasts under raw_two_stage / raw_fly
    assert SPEC.cov_methods == ("two_stage", "fly", "pair", "raw_two_stage", "raw_fly")
    assert SPEC.g6_rows == ("point", "lo", "hi")                            # AA.9.1 6304: 점 · CI 하한끝 · 상한끝
    assert SPEC.g6_scales == ("hedges", "raw")                              # AA.9.1 6305: Hedges = 주 행, first
    # AA.7 2 6253: smoke = Y pilot pair b|17 (j 0, AA.7 0 6248), one fly
    assert (SPEC.smoke_pair, SPEC.smoke_flies) == (0, 1)
    # smoke probe seeds root + f·100 + k (W layout, AA.2 6173), f < 32: stay below the smoke training root
    assert (SPEC.smoke_probe_span, SPEC.smoke_fly_span) == (100, 32)
    assert SPEC.smoke_probe_seed0 + SPEC.smoke_fly_span * SPEC.smoke_probe_span <= SPEC.smoke_train_seed0
    assert SPEC.smoke_flies <= SPEC.smoke_fly_span and SPEC.probes <= SPEC.smoke_probe_span


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
              "g6_rows", "g6_scales", "d_false", "g6_seed", "compare_refs", "ci_level", "floor_qs", "bar_ref",
              "cov_methods", "c7_declared", "hedges_j_declared"):
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
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                if n.module == "flymon.brain":
                    todo += [f"flymon/brain/{a.name}.py" for a in n.names]
                elif n.module.startswith("flymon.brain."):
                    todo.append(n.module.replace(".", "/") + ".py")
            elif isinstance(n, ast.Import):
                todo += [a.name.replace(".", "/") + ".py" for a in n.names if a.name.startswith("flymon.brain.")]
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
