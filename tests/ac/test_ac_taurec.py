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


# ---- AC.5 label on every --lv output (fix round 1): fake pools, tmp dirs only ------------------------------------
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from flymon.ac.spec import LABEL  # noqa: E402


class FakePool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def lever_sha(self):
        return "fake-sha"


def _traj():
    return {"alt": {"ratio": [1.0, 0.9], "floor_frac_taught": [0.1, 0.2]},
            "same": {"ratio": [1.0, 0.9], "floor_frac_taught": [0.1, 0.2]}}


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """main() with every pool, connectome, encoder and trajectory replaced; writes under tmp_path only."""
    import flymon.ac.config as ac_config
    import flymon.agent.config as agent_config
    import flymon.brain.circuits as circuits
    import flymon.brain.connectome as connectome
    import flymon.brain.fly_pool as fp
    import flymon.brain.lv_pool as lvp
    from flymon.brain.config import Params
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(params=Params())
    monkeypatch.setattr(agent_config, "load_c3_config", lambda *a, **k: cfg)
    monkeypatch.setattr(ac_config, "load_lv_config", lambda *a, **k: cfg)
    monkeypatch.setattr(ac_config, "grid_encoder", lambda pops: "enc")
    monkeypatch.setattr(ac_config, "lv_codebook", lambda *a, **k: (None, "fake-digest"))
    monkeypatch.setattr(connectome.Connectome, "load", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(circuits.Populations, "from_connectome", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(fp, "FlyPool", FakePool)
    monkeypatch.setattr(lvp, "LeverFlyPool", FakePool)
    monkeypatch.setattr(taurec, "synthetic_odours", lambda pops, n, seed: [{}] * n)
    monkeypatch.setattr(taurec, "synthetic_grid_odours", lambda enc, n, seed: [{}] * n)
    monkeypatch.setattr(taurec, "taught_mask", lambda pool, od, spec: (np.ones(3, bool), 7))
    monkeypatch.setattr(taurec, "trajectory", lambda *a, **k: _traj())
    rt = load()
    monkeypatch.setattr(rt, "git_provenance", lambda files=(): {"commit": "test", "dirty": False, "dirty_files": []})
    return tmp_path, rt


def test_lv_run_labels_summary_and_every_r_file(run_env):
    root, rt = run_env
    assert rt.main(["--lv", "--smoke", "--allow-dirty"]) == 0
    summ = json.loads((root / "results/rescope-smoke/summary/rescope_taurec_lv.json").read_text())
    assert summ["label"] == LABEL and summ["lever_edit"] and summ["strength"] == 1.0
    assert summ["lever_sha"] == "fake-sha" and summ["codebook_digest"] == "fake-digest"
    files = sorted((root / "results/rescope-smoke/taurec_lv").glob("r_*.json"))
    assert len(files) == 2 and all(json.loads(f.read_text())["label"] == LABEL for f in files)


def test_non_lv_run_has_no_label_or_lever_keys(run_env):
    root, rt = run_env
    assert rt.main(["--smoke", "--allow-dirty"]) == 0
    summ = json.loads((root / "results/rescope-smoke/summary/rescope_taurec.json").read_text())
    assert "label" not in summ and not set(rt.LV_KEYS) & set(summ)
    files = sorted((root / "results/rescope-smoke/taurec").glob("r_*.json"))
    assert len(files) == 2 and all(set(json.loads(f.read_text())) == {"recovery_per_pulse", "trajectory"}
                                   for f in files)


def test_reselect_lv_labels_and_carries_lever_keys(run_env):
    root, rt = run_env
    assert rt.main(["--lv", "--smoke", "--allow-dirty"]) == 0
    assert rt.main(["--lv", "--smoke", "--allow-dirty", "--reselect"]) == 0
    summ = json.loads((root / "results/rescope-smoke/summary/rescope_taurec_lv.json").read_text())
    assert summ["label"] == LABEL and summ["provenance"]["reselect"] is True
    assert summ["lever_sha"] == "fake-sha" and summ["codebook_digest"] == "fake-digest"


def test_label_helper_and_carry_keys_exclude_label():
    rt = load()
    assert rt.lv_label(True) == {"label": LABEL} and rt.lv_label(False) == {}
    assert "label" not in rt.carry_keys(False) and "label" not in rt.CARRY_KEYS
