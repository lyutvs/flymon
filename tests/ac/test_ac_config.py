"""The L_V agent configuration (AC.2): C3 Params with r replaced, z_V, MBON13 / MBON05, E-grid k2-norm at 1.0."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.ac.config import grid_encoder, load_lv_config, load_recovery, lv_codebook
from flymon.ac.spec import SPEC
from flymon.agent import e_codebook
from flymon.agent import encode_grid as eg
from flymon.agent.config import config_hash
from flymon.brain.j_store import load_c3

needs_summary = pytest.mark.skipif(not Path("results/summary/m0d.json").exists(), reason="needs results/summary/m0d.json")


@needs_summary
def test_lv_config_numbers():
    cfg = load_lv_config(0.002)
    base, _, _ = load_c3(json.loads(Path("results/summary/m0d.json").read_text()))
    assert cfg.z == {"A": (16.917, 12.484), "P": (80.167, 29.775)}
    assert cfg.readout == {"A": "MBON13", "P": "MBON05"}
    assert (cfg.strength, cfg.settle_ms, cfg.read_ms) == (1.0, 800.0, 600.0)
    assert (cfg.reward_type, cfg.punish_type) == ("PAM08", "PPL105")
    assert (cfg.tau_start, cfg.tau_end, cfg.tau_battles) == (1.0, 0.2, 20)
    assert cfg.params.recovery_per_pulse == 0.002
    assert dataclasses.replace(cfg.params, recovery_per_pulse=base.recovery_per_pulse) == base


@needs_summary
def test_config_hash_tracks_recovery():
    assert config_hash(load_lv_config(0.001)) != config_hash(load_lv_config(0.002))
    assert config_hash(load_lv_config(0.002)) == config_hash(load_lv_config(0.002))


def test_codebook_digest_matches_the_block():
    cb, dg = lv_codebook()
    assert dg == "76500f42964b24c4e4a934fb70fb1692e75fc0ff353cb78486e9034a2cd2c858"
    assert e_codebook.digest(cb.words) == dg and cb.k == 2


def test_grid_encoder_battle_odour_is_the_cell_odour():
    cb, _ = lv_codebook()
    gloms = sorted({g for w in cb.words for g in w})

    class Pops:
        receptor_types = {g: list(range(1 + i % 4)) for i, g in enumerate(gloms)}

    class Mon:
        def __init__(self, s):
            self.species, self.current_hp_fraction = s, 1.0

    class Battle:
        active_pokemon, opponent_active_pokemon = Mon("Blastoise"), Mon("Charizard")

    class Mv:
        id = "surf"

    enc = grid_encoder(Pops())
    rc = {g: len(v) for g, v in Pops.receptor_types.items()}
    assert enc.odour(Battle(), Mv()) == eg.odour(rc, cb, "WATER", ("FIRE", "FLYING"), "norm")


def test_load_recovery_refusals(tmp_path):
    p = tmp_path / "t.json"
    with pytest.raises(SystemExit, match="does not exist"):
        load_recovery(p)
    p.write_text(json.dumps({"status": "STOP_NO_RECOVERY", "lever_edit": SPEC.lever_edit}))
    with pytest.raises(SystemExit, match="not SELECTED"):
        load_recovery(p)
    p.write_text(json.dumps({"status": "SELECTED", "recovery_per_pulse": 0.005, "lever_edit": "none"}))
    with pytest.raises(SystemExit, match="lever"):
        load_recovery(p)
    p.write_text(json.dumps({"status": "SELECTED", "recovery_per_pulse": 0.005, "lever_edit": SPEC.lever_edit}))
    r, info = load_recovery(p)
    assert r == 0.005 and info["path"] == str(p) and len(info["sha256"]) == 64
