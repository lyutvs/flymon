"""AB's numbers (AB.2–AB.9.1): the declared values, AB's seed blocks colliding with nothing declared, the stream test
vectors, the sealed subset, and no AB file but ab_spec holding a number (Z's literal guard)."""
import ast
import dataclasses
import importlib
import math
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.ab_spec import SPEC, ABSpec
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]


def test_ab_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/ab_spec.py") == "flymon.brain.ab_spec"


def test_chain():
    assert SPEC.stages == ("stage0", "reuse", "generate", "seal_code", "cal_gate", "futility", "kc_input", "set",
                           "smoke", "oracle", "screen", "calibrate", "learn", "verdict", "records")
    lab = dict(SPEC.stage_labels)
    assert [lab[s] for s in SPEC.stages] == ["0a", "0b", "0c", "0d", "0e", "0f", "3", "4", "5", "6", "7", "7b", "8", "9",
                                             "10"]
    assert SPEC.sealed_stages()[:2] == ("cal_gate", "futility")
    assert SPEC.sealed_stages()[0] == "cal_gate" and SPEC.first_measure == "kc_input"


def test_reuse_and_generator_facts():
    assert SPEC.kc_input_sha256.startswith("f197492a") and SPEC.w_digest_keys == Y.main_digest_keys
    assert (SPEC.w_n_b, SPEC.w_n_a, SPEC.w_last_turn) == (Y.main_n_b, Y.main_n_a, Y.main_last_turn)
    assert [c for _, c in SPEC.v_commits] == ["928eaad", "7dc199d", "cf0b3b2", "a279a56", "69f7b2d"]
    assert len(SPEC.aa_blocks) == 9 and SPEC.decl_commit == "ea408a2"
    assert (SPEC.n_opp, SPEC.n_combos, SPEC.shuffle_seed) == (73, 1168, 20261007)
    assert sum(dict(SPEC.decl_pre_kc).values()) == 209 and sum(dict(SPEC.decl_post_kc_max).values()) == 184
    assert sum(dict(SPEC.decl_lv).values()) == 127 and dict(SPEC.decl_odours) == {"all": 107, "v_cache": 36, "new": 71}
    assert sum(v for _, v in SPEC.decl_a_x_groups) == 139 and sum(v for _, v in SPEC.decl_a_t_groups) == 139
    assert len(SPEC.kc_repro) == 4 and SPEC.lv_odour_n == 108


def test_criterion_and_calibration_numbers():
    nu = SPEC.probes - 1
    c7 = math.sqrt(nu / 2) * math.gamma((nu - 1) / 2) / math.gamma(nu / 2)
    j = 1 - 3 / (4 * nu - 1)
    assert round(c7, 4) == SPEC.c7_declared and round(j, 4) == SPEC.hedges_j_declared
    assert round(1 / (j * c7), 6) == SPEC.delta_declared
    assert (SPEC.bar, SPEC.raw_min, SPEC.k_min, SPEC.g_min, SPEC.winsor) == (1.0, 0.25, 8, 5, 10.0)
    assert SPEC.p_grid == (0.025, 0.0125, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001)
    assert SPEC.f_grid == (0.00625, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.00002, 0.00001)
    assert (SPEC.p_target, SPEC.f_target, SPEC.f_target * 4) == (0.025, 0.00625, 0.025)
    assert (SPEC.n_sel, SPEC.n_ver, SPEC.truth_pairs, SPEC.boot_b) == (5_000, 10_000, 1_000_000, 10_000)
    assert len(SPEC.cells) == 24 and SPEC.synth_cells == (1, 4, 12, 19) and SPEC.bench_cells == (17, 23)
    assert (SPEC.floor_const, SPEC.floor_flies, SPEC.skew_sd) == (0.05, 2, 2.0)
    assert (SPEC.core_cap_h, SPEC.records_cap_h, SPEC.cost_margin, SPEC.smoke_k, SPEC.smoke_lenient) == (24.0, 4.0, 2.0, 26, 40)


def test_ab_seed_blocks_collide_with_nothing_declared():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    roots = {SPEC.oracle_act_seed0, SPEC.oracle_select_seed0, SPEC.oracle_report_seed0, SPEC.boot_seed, SPEC.cal_seed,
             SPEC.synth_seed, SPEC.smoke_probe_seed0, SPEC.smoke_train_seed0, SPEC.smoke_oracle_seed0,
             SPEC.probe_seed0, SPEC.train_seed0}
    assert roots <= out and all(any(lo <= r < hi for lo, hi in SPEC.seed_blocks) for r in roots)
    for (a, b), (c, d) in zip(SPEC.seed_blocks, SPEC.seed_blocks[1:]):
        assert b <= c
    # the judged probe / training formulas stay in their blocks for c < 300
    assert SPEC.probe_seed0 + 299 * W.cand_probe_stride + 7 * W.fly_probe_stride + 7 < 89_300_000
    assert SPEC.train_seed0 + 299 * W.cand_train_stride + 7 * W.fly_train_stride + 39 < 101_500_000
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/ab_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    assert not any(lo <= x < hi for x in declared for lo, hi in SPEC.seed_blocks)
    assert set(V.kc_seeds()) == set(range(24_002_000, 24_002_008))       # reused on purpose (AB.2)
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(ABSpec))


def test_oracle_seed_lists():
    o = SPEC.oracle_seeds()
    assert o["act"][0] == 88_000_000 and o["select"][-1] == 88_000_107 and o["report"][0] == 88_000_200
    assert SPEC.smoke_oracle_seeds()["report"][-1] == 88_060_207


def test_futility_numbers():
    """AB.7 0f · AB.9.3 (the model, the reps, the threshold, the grid) and the fifth stream vector (AB.2)."""
    assert (SPEC.fut_src_pairs, SPEC.fut_src_units, SPEC.fut_x_sizes) == (16, 16 * 8 * 3, (5, 2, 2, 2, 2, 2, 1))
    assert SPEC.fut_decl_point == (5.263, -3.984, 6.021, -4.517) and SPEC.fut_decl_raw_assoc == (2.573, -1.449)
    assert SPEC.fut_decl_tau == (2.485, 1.676, 2.117, 2.038) and SPEC.fut_decl_tau_src == ("sd", "dl", "sd", "dl")
    assert (SPEC.fut_chol_eps, SPEC.fut_reps, SPEC.fut_threshold, SPEC.fut_ci_tails) == (1e-9, 2_000, 0.5, (0.025, 0.975))
    assert SPEC.fut_tags == ("g5", "g6", "g7") and SPEC.fut_allocs == ("icc", "pair", "group")
    assert SPEC.fut_judge == ("g6", "icc") and SPEC.rep_sizes("g6") == (5, 4, 3, 3, 2, 1)
    assert SPEC.fut_grid_g == tuple(range(5, 13)) and SPEC.fut_grid_k == tuple(range(8, 41, 4))
    assert (SPEC.fut_grid_reps, SPEC.fut_grid_alloc) == (500, "icc")
    sc = dict(SPEC.fut_scenarios)
    assert list(sc) == ["all", "noskew", "sd2", "sd1"] and sc["all"] == () and sc["noskew"] == (18, 19, 20, 21)
    assert sc["sd2"] == (5, 9, 13, 16) and sc["sd1"] == (4, 5, 8, 9, 12, 13) + tuple(range(15, 22))
    n = sum(1 for g in SPEC.fut_grid_g for k in SPEC.fut_grid_k if k >= g)
    assert n == 68 and SPEC.fut_grid_cells == 4 * n == 272
    assert SPEC.stream_vectors[-1][0] == ("synth", "fut", "g6", "icc") and len(SPEC.stream_vectors) == 5


# AB.5 칸 24개, typed from the spec text (not from ab_spec): (kind, value, SD of the heterogeneity it adds). The
# value is the SD for the SD cells, the direction (+1 / −1) for the skew and floor cells, whose SD is skew_sd 2 / 0.
AB5_CELLS = (
    ("none", 0.0, 0.0),                                                                   # (1)
    ("X", 0.5, 0.5), ("X", 1.0, 1.0), ("X", 2.0, 2.0), ("X", 3.0, 3.0),                   # (2)–(5)
    ("T", 0.5, 0.5), ("T", 1.0, 1.0), ("T", 2.0, 2.0), ("T", 3.0, 3.0),                   # (6)–(9)
    ("pair", 0.5, 0.5), ("pair", 1.0, 1.0), ("pair", 2.0, 2.0), ("pair", 3.0, 3.0),       # (10)–(13)
    ("XT", 1.0, 1.0), ("XT", 2.0, 2.0), ("XT", 3.0, 3.0),                                 # (14)–(16)
    ("Xpair", 2.0, 2.0),                                                                  # (17)
    ("skX", 1.0, 2.0), ("skX", -1.0, 2.0),                                                # (18)–(19)
    ("skT", 1.0, 2.0), ("skT", -1.0, 2.0),                                                # (20)–(21)
    ("fly", 1.0, 1.0),                                                                    # (22)
    ("floor", 1.0, 0.0), ("floor", -1.0, 0.0))                                            # (23)–(24)


def test_cells_typed_from_ab5_and_the_futility_scenarios():
    assert SPEC.cells == tuple((k, v) for k, v, _ in AB5_CELLS)
    assert [SPEC.cell(i) for i in (1, 5, 17, 18, 24)] == [("none", 0.0), ("X", 3.0), ("Xpair", 2.0), ("skX", 1.0),
                                                          ("floor", -1.0)]
    # AB.7 0f 기록 격자: noskew drops the skew cells (18)–(21), sd2 the SD 3 cells, sd1 every cell with SD > 1
    idx = range(1, len(AB5_CELLS) + 1)
    want = dict(all=(), noskew=tuple(i for i in idx if AB5_CELLS[i - 1][0].startswith("sk")),
                sd2=tuple(i for i in idx if AB5_CELLS[i - 1][2] > 2), sd1=tuple(i for i in idx if AB5_CELLS[i - 1][2] > 1))
    assert want["noskew"] == (18, 19, 20, 21) and want["sd2"] == (5, 9, 13, 16)
    assert want["sd1"] == (4, 5, 8, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21)
    assert SPEC.fut_scenarios == tuple(want.items())
    assert SPEC.skew_sd == 2.0


# AB.9.1 결과 전 고정 목록, as far as ab_spec holds it (Y's filter and V's KC numbers seal through their own files)
AB91_FIXED = ("n_opp", "n_combos", "shuffle_seed", "lv_odour_n", "kc_repro", "flies", "probes", "winsor",
              "hedges_j_declared", "bar", "raw_min", "boot_b", "p_grid", "p_target", "f_grid", "f_target", "cp_level",
              "cells", "n_sel", "n_ver", "delta_declared", "truth_pairs", "rep_structures", "k_min", "g_min",
              "core_cap_h", "records_cap_h", "cost_margin", "smoke_k", "smoke_lenient", "seed_blocks",
              "stream_vectors", "stages", "fut_src_pairs", "fut_src_units", "fut_judge", "fut_allocs", "fut_reps",
              "fut_threshold", "fut_tags", "fut_grid_g", "fut_grid_k", "fut_scenarios", "fut_grid_reps",
              "fut_grid_alloc", "fut_grid_cells", "fut_chol_eps")


def test_seal_fields():
    f = SPEC.seal_fields()
    assert len(set(SPEC.seal_names)) == len(SPEC.seal_names)
    names = [x.name for x in dataclasses.fields(ABSpec)]
    decl = [n for n in names if n.startswith("decl_") and n != "decl_commit"]          # the AB.3 declared values
    roots = [n for n in names if "seed" in n and type(getattr(SPEC, n)) is int and not n.startswith("aa_")]
    assert len(decl) == 17 and len(roots) == 13                                     # AB's roots (+ shuffle, n seeds)
    for k in AB91_FIXED + tuple(decl) + tuple(roots):
        assert k in f, k
    assert all(f[k] == getattr(SPEC, k) for k in f)


def _numbers(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            out.add(n.value)
    return out


def test_no_ab_file_but_ab_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("ab_*.py")) if p.name != "ab_spec.py"]
    if (ROOT / "scripts/run_ab.py").exists():
        files.append(ROOT / "scripts/run_ab.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
