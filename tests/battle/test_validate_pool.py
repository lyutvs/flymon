"""Pool validation reporting when Node (and therefore `validate-team gen1ou`) is absent."""
from pathlib import Path

from flymon.battle import validate_pool

MISSING_NODE = Path("/nonexistent/showdown/node")


def test_legality_is_unchecked_not_ok_without_node(monkeypatch, capsys):
    monkeypatch.setattr(validate_pool, "NODE_BIN", MISSING_NODE)
    assert validate_pool.main([]) == 0
    out = capsys.readouterr().out
    rows = [l for l in out.splitlines() if l.startswith("| Blastoise")]
    assert rows and all(l.rstrip().endswith("| unchecked |") for l in rows)
    # the legality column (last cell) never reads `ok` when nothing was checked
    assert not any(l.rstrip().endswith("| ok |") for l in out.splitlines() if l.startswith("| "))
    assert "legality was not checked" in out


def test_write_docs_refuses_without_node(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(validate_pool, "NODE_BIN", MISSING_NODE)
    monkeypatch.setattr(validate_pool, "DOCS_POOL", tmp_path / "pool.md")
    assert validate_pool.main(["--write-docs"]) == 2
    assert "refusing --write-docs" in capsys.readouterr().out
    assert not (tmp_path / "pool.md").exists()


def test_legality_cell_states():
    assert validate_pool._legality_cell(True, "") == "ok"
    assert validate_pool._legality_cell(False, "bad") == "FAIL: bad"
    assert validate_pool._legality_cell(None, "node missing") == "unchecked"
