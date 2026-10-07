"""AC.7 2a: tau_rec on L_V reuses the re-scope stage-1 code with E-grid odours, the lever pool and strength 1.0;
the grid and the selection rule are unchanged."""
import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np

from flymon.agent import encode_grid as eg
from flymon.battle.pool import POOL
from flymon.rescope import taurec
from flymon.rescope.spec import SPEC
from tests.ac.ac_fakes import CB, MI, ST, stub_grid

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("rt", ROOT / "scripts/run_rescope_taurec.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_synthetic_grid_odours_are_e_grid_cell_odours():
    enc = stub_grid()
    got = taurec.synthetic_grid_odours(enc, 6, 990_000)
    rng = np.random.default_rng(990_000)
    for o in got:
        me, opp = rng.choice(len(POOL), 2, replace=True)
        mon = POOL[int(me)]
        move = mon.attacks[int(rng.integers(len(mon.attacks)))]
        assert o == eg.odour(enc.rc, CB, MI[move][0], tuple(sorted(ST[POOL[int(opp)].species])), "norm")
    assert got == taurec.synthetic_grid_odours(enc, 6, 990_000)


def test_lv_setup_changes_strength_stage_and_default_out_only():
    rt = load()
    a = Namespace(lv=True, smoke=False, out=rt.DEFAULT_OUT)
    spec, stage = rt.lv_setup(a)
    assert stage == "taurec_lv" and spec.strength == 1.0 and a.out == "results/rescope/taurec_lv"
    assert spec.recovery_grid == SPEC.recovery_grid and spec.taurec_pulses == SPEC.taurec_pulses
    assert spec.taurec_taught_floor_max == SPEC.taurec_taught_floor_max and spec.median_floor == SPEC.median_floor
    assert rt.paths(a.out, False, stage)[1] == Path("results/summary/rescope_taurec_lv.json")
    assert rt.paths(a.out, True, stage)[1] == Path("results/rescope-smoke/summary/rescope_taurec_lv.json")
    b = Namespace(lv=False, smoke=False, out=rt.DEFAULT_OUT)
    spec_b, stage_b = rt.lv_setup(b)
    assert stage_b == "taurec" and spec_b == SPEC and b.out == "results/rescope/taurec"
    s, _ = rt.lv_setup(Namespace(lv=True, smoke=True, out=rt.DEFAULT_OUT))
    assert s.strength == 1.0 and s.recovery_grid == (0.0, 0.02) and s.taurec_pulses == 20


def test_carry_keys_add_the_lever_fields_only_on_lv():
    rt = load()
    assert rt.carry_keys(False) == rt.CARRY_KEYS
    assert set(rt.carry_keys(True)) - set(rt.CARRY_KEYS) == {"lever_edit", "lever_sha", "codebook_digest", "encoder",
                                                               "strength"}
