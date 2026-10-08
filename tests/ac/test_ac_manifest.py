"""The model manifest (spec AC.1 (i)): frozen after AC.7 2a; M4 runs and resumes verify it."""
import pytest

from flymon.ac.manifest import build_model_manifest, check_model
from flymon.ac.spec import LABEL, SPEC
from flymon.agent.config import AgentConfig, config_hash
from flymon.brain.config import Params
from flymon.brain.w_spec import SPEC as W

INFO = {"path": "results/summary/rescope_taurec_lv.json", "sha256": "f" * 64}


def cfg(r=0.002, strength=1.0, z=None):
    return AgentConfig(params=Params(recovery_per_pulse=r), z=z or W.z_v(), readout={"A": "MBON13", "P": "MBON05"},
                       strength=strength)


def taurec(**kw):
    d = dict(status="SELECTED", recovery_per_pulse=0.002, lever_edit=SPEC.lever_edit, lever_sha=SPEC.lever_sha,
             rule="10.6 amendment 2026-09-28")
    d.update(kw)
    return d


def test_frozen_manifest_fields():
    c = cfg()
    m = build_model_manifest(taurec(), INFO, c, "d" * 64, "e" * 64)
    assert m["status"] == "FROZEN" and m["lever_sha"] == SPEC.lever_sha and m["lever_edit"] == SPEC.lever_edit
    assert m["z_V"] == {"A": [16.917, 12.484], "P": [80.167, 29.775]} and m["strength"] == 1.0
    assert m["recovery_per_pulse"] == 0.002 and m["codebook_digest"] == "d" * 64 and m["encoder_grid_sha256"] == "e" * 64
    assert m["seeds"] == {"gen": 301, "learn": 302, "eval": 303, "boot": 304, "tie": 305}
    assert m["config_hash"] == config_hash(c) and m["taurec"] == INFO and m["label"] == LABEL
    assert m["readout"] == {"A": "MBON13", "P": "MBON05"} and m["tau"] == {"start": 1.0, "end": 0.2, "battles": 20}


def test_no_recovery_is_recorded_not_frozen():
    m = build_model_manifest(taurec(status="STOP_NO_RECOVERY", recovery_per_pulse=None, failed={"0.0": ["b"]}),
                             INFO, cfg(0.0), "d" * 64, "e" * 64)
    assert m["status"] == "STOP_NO_RECOVERY" and m["failed"] == {"0.0": ["b"]} and "recovery_per_pulse" not in m


def test_refuses_another_lever_or_r():
    with pytest.raises(ValueError, match="lever"):
        build_model_manifest(taurec(lever_sha="0" * 64), INFO, cfg(), "d" * 64, "e" * 64)
    with pytest.raises(ValueError, match="recovery"):
        build_model_manifest(taurec(recovery_per_pulse=0.005), INFO, cfg(0.002), "d" * 64, "e" * 64)


def test_check_model_names_each_mismatch():
    c = cfg()
    m = build_model_manifest(taurec(), INFO, c, "d" * 64, "e" * 64)
    assert check_model(m, cfg=c, codebook_digest="d" * 64) == []
    assert check_model(m, cfg=cfg(0.005), codebook_digest="d" * 64) == ["recovery_per_pulse", "config_hash"]
    assert "codebook_digest" in check_model(m, cfg=c, codebook_digest="x" * 64)
    assert "strength" in check_model(m, cfg=cfg(strength=0.35), codebook_digest="d" * 64)
    assert "z_V" in check_model(m, cfg=cfg(z={"A": (1.0, 1.0), "P": (1.0, 1.0)}), codebook_digest="d" * 64)
    assert check_model(dict(m, status="STOP_NO_RECOVERY"), cfg=c, codebook_digest="d" * 64)[0] == "status"


def test_script_lever_mismatch_is_a_one_line_stop(tmp_path, monkeypatch, capsys):
    """Final review M6: a lever sha mismatch exits 2 with one STOP line, no traceback, no manifest."""
    import importlib.util
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts/write_ac_model_manifest.py"
    sp = importlib.util.spec_from_file_location("write_ac_model_manifest", p)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    t = tmp_path / "taurec.json"
    t.write_text(json.dumps(taurec(lever_sha="0" * 64)))
    assert mod.main(["--taurec", str(t)]) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("STOP_LEVER_MISMATCH")
    assert not (tmp_path / "results/summary/ac_model_manifest.json").exists()
