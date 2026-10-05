"""Spec W.1-W.3 / W.9: every W number in w_spec; W's blocks (main 26_000_000 / 28_000_000 by candidate, pilot
40_000_000 / 41_000_000, smoke 42_1xx_xxx, OC 42_000_000, oracle 24_700_xxx) as formulas, inside their layout and
colliding with no declared seed of any other spec module (nor with their P-style training seeds); W declares none of
V's seeds (WSpec is not a VSpec); the phases of R / N / RN; the design grid; the shared, T, U and W measurement keys."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.brain import d6a
from flymon.brain.w_spec import BRAINS, SPEC, WSpec, smoke
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
BASE, STRIDE = 1_000_000, 1000


def _declared_without_w() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/w_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


def w_seeds(n_cand=SPEC.n_cand_max, n_pilot=SPEC.n_pilot_max, k_max=32) -> tuple:
    probes, trains = set(), set()
    for c in range(n_cand):
        for f in range(SPEC.f_max):
            probes |= set(SPEC.probe_seeds(c, f, k_max))
            trains |= {SPEC.train_base(c, p) + f * SPEC.fly_train_stride + t for p in (0, 1)
                       for t in range(SPEC.trials)}
    for j in range(n_pilot):
        for f in range(SPEC.pilot_flies):
            probes |= set(SPEC.pilot_probe_seeds(j, f, SPEC.pilot_probes))
            trains |= {SPEC.pilot_train_base(j, p) + f * SPEC.fly_train_stride + t for p in (0, 1)
                       for t in range(SPEC.trials)}
    probes |= set(SPEC.smoke_seed_set()) | {s for v in SPEC.oracle_seeds().values() for s in v}
    trains |= {SPEC.smoke_train_base(p) + f * SPEC.fly_train_stride + t for p in (0, 1) for f in range(SPEC.f_max)
               for t in range(SPEC.trials)}
    return probes, trains


def test_w_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/w_spec.py") == "flymon.brain.w_spec"


def test_seed_formulas():
    assert SPEC.probe_seeds(5, 3, 8) == [26_000_000 + 5 * 4_000 + 3 * 100 + k for k in range(8)]
    assert SPEC.probe_seeds(5, 3, 16, k0=8) == [26_000_000 + 20_000 + 300 + k for k in range(8, 16)]
    assert SPEC.train_base(5, 0) == 28_000_000 + 5 * 40_000 and SPEC.train_base(5, 1) == 28_000_000 + 200_020
    assert SPEC.pilot_probe_seeds(2, 1, 8) == [40_000_000 + 8_000 + 100 + k for k in range(8)]
    assert SPEC.pilot_train_base(2, 1) == 41_000_000 + 80_000 + 20
    assert SPEC.oracle_seeds() == dict(act=list(range(24_700_000, 24_700_008)),
                                       select=list(range(24_700_100, 24_700_108)),
                                       report=list(range(24_700_200, 24_700_208)))
    sm = smoke(SPEC).oracle_seeds()
    assert all(42_100_000 <= s < 42_200_000 for v in sm.values() for s in v)
    assert SPEC.oc_seed == 42_000_000
    for bad in (lambda: SPEC.probe_seeds(300, 0, 8), lambda: SPEC.probe_seeds(0, 32, 8),
                lambda: SPEC.probe_seeds(0, 0, 101), lambda: SPEC.pilot_probe_seeds(25, 0, 8)):
        with pytest.raises(ValueError):
            bad()


def test_layout_has_no_internal_overlap():
    probes, trains = w_seeds()
    assert not probes & trains
    main = {s for s in probes | trains if s < 42_000_000}
    smoke_ = (probes | trains) - main
    assert min(probes) >= 24_700_000 and max(main) < 42_000_000
    assert all(42_100_000 <= s < 42_200_000 for s in smoke_)
    assert all(not 27_200_000 <= s < 28_000_000 for s in main)


def test_global_seed_collision():
    declared = _declared_without_w()
    for s in (500, 24_002_000, 24_600_000, 25_400_000, 23_000_000, 22_000_000):
        assert s in declared, s
    probes, trains = w_seeds()
    mine = probes | trains
    assert not mine & declared
    assert not {s for s in mine if (s - BASE) // STRIDE in declared}       # no declared s trains P-style on W's


def test_wspec_carries_no_v_seed():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {26_000_000, 28_000_000, 40_000_000, 41_000_000, 42_100_000, 42_110_000, 42_150_000, 42_000_000} \
        | set(range(24_700_000, 24_700_008)) | set(range(24_700_100, 24_700_108)) | set(range(24_700_200, 24_700_208))
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(WSpec))


def test_phases_and_designs():
    assert BRAINS == ("R", "N", "RN")
    assert SPEC.phases("R", 10, 30) == [["PAM08", 20, 10], ["PPL105", 20, 30]]
    assert SPEC.phases("N", 10, 30) == [[None, 20, 10], [None, 20, 30]]
    assert SPEC.phases("RN", 10, 30) == [["PAM08", 20, 10], [None, 20, 30]]
    d = SPEC.designs()
    assert len(d) == 3 * 2 * 25 and d[0] == (0.5, 8, 8) and d[-1] == (0.75, 16, 32)


def test_numbers():
    assert (SPEC.bar, SPEC.band_width, SPEC.naive_max, SPEC.mech_min, SPEC.round_digits) == (1.0, 0.2, 0.5, 0.75, 9)
    assert (SPEC.d_power, SPEC.p_power, SPEC.d_false, SPEC.p_false) == (1.5, 0.80, 0.5, 0.05)
    assert (SPEC.oc_reps, SPEC.cal_reps, SPEC.cal_tol, SPEC.cal_iter, SPEC.boot_draws) == (4000, 2000, 0.02, 40, 200)
    assert SPEC.cluster_grid == (0.0, 0.5, 1.0) and SPEC.boot_level == 0.95 and SPEC.boot_reps == 400
    assert (SPEC.k_cap, SPEC.k_min, SPEC.envelope, SPEC.envelope_solo_from) == (8, 4, 3, 29)
    assert (SPEC.trials, SPEC.pulse_ms, SPEC.train_settle_ms, SPEC.gap_ms, SPEC.strength) == (20, 400.0, 800.0,
                                                                                              200.0, 1.0)
    assert (SPEC.first_turn, SPEC.last_turn, SPEC.n_b_expected, SPEC.n_a_expected) == (306, 1985, 167, 82)
    assert SPEC.budget_h == 24.0 and SPEC.min_gate_pairs == 4 and SPEC.cal_floor_rule == "zero"
    assert SPEC.synth_reps >= 1000
    assert SPEC.z_v() == {"A": (16.917, 12.484), "P": (80.167, 29.775)}
    assert dict(SPEC.v_commits) == {"z": "928eaad", "kc_input": "7dc199d", "set": "cf0b3b2", "judge": "a279a56"}
    assert (SPEC.summary, SPEC.cache_dir, SPEC.archive_root) == (
        "results/summary/w_learning.json", "results/w/cache", "~/flymon-archive/w")


def test_smoke_changes_scale_only():
    sm = smoke(SPEC)
    for f in dataclasses.fields(WSpec):
        if f.name not in ("smoke", "workers"):
            assert getattr(sm, f.name) == getattr(SPEC, f.name), f.name
    assert sm.smoke and sm.workers == 4


def test_no_w_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("w_*.py")) + [ROOT / "scripts/run_w.py"]
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_000_000 <= i < 24_700_000 or 24_710_000 <= i < 26_000_000
               or i in (20261004, 20261005) or 800_000 <= i < 800_200}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_t_and_u_keys_are_the_reused_ones():
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import u_measure_key
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
    assert t_measure_key(str(NPZ))["key"] == SPEC.t_measure_key_t
    assert u_measure_key(str(NPZ))["key"] == SPEC.u_measure_key_u
