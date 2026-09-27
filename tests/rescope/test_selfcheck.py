"""The self-check CLI: all_ok is the AND of its four checks and a failing check exits 1 (checks monkeypatched; the real
ones need the connectome and are run by the controller)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("rescope_selfcheck", ROOT / "scripts" / "rescope_selfcheck.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _patch(monkeypatch, mod, oks):
    names = ("check_pairs_digest", "check_oracle_copy", "check_pool_equivalence", "check_naive_vs_oracle_pre")
    for name, ok in zip(names, oks):
        monkeypatch.setattr(mod, name, lambda ok=ok: {"ok": ok})


def test_one_failing_check_fails_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load()
    _patch(monkeypatch, mod, (True, False, True, True))
    with pytest.raises(SystemExit) as e:
        mod.main(["--allow-dirty"])
    assert e.value.code == 1
    d = json.loads((tmp_path / "results/rescope/selfcheck.json").read_text())
    assert d["all_ok"] is False
    assert set(d["checks"]) == {"pairs_digest", "oracle_copy_real", "pool_equivalence_c3", "naive_vs_oracle_pre"}
    assert d["checks"]["oracle_copy_real"]["ok"] is False and d["checks"]["pairs_digest"]["ok"] is True


def test_all_checks_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load()
    _patch(monkeypatch, mod, (True, True, True, True))
    assert mod.main(["--allow-dirty"]) == 0
    assert json.loads((tmp_path / "results/rescope/selfcheck.json").read_text())["all_ok"] is True


def test_raising_check_is_recorded_not_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load()
    _patch(monkeypatch, mod, (True, True, True, True))

    def boom():
        raise RuntimeError("no connectome")
    monkeypatch.setattr(mod, "check_naive_vs_oracle_pre", boom)
    with pytest.raises(SystemExit) as e:
        mod.main(["--allow-dirty"])
    assert e.value.code == 1
    c = json.loads((tmp_path / "results/rescope/selfcheck.json").read_text())["checks"]["naive_vs_oracle_pre"]
    assert c["ok"] is False and "no connectome" in c["error"]
