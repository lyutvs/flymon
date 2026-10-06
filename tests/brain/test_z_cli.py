"""scripts/run_z.py: argument refusals, exit codes, and output without split keys."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_z", ROOT / "scripts/run_z.py")
run_z = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_z)


def test_name_goes_with_archive_only(capsys):
    assert run_z.main(["--stage", "archive"]) == 2
    assert run_z.main(["--stage", "split", "--name", "split"]) == 2


def test_unknown_stage_is_an_argparse_error():
    with pytest.raises(SystemExit) as e:
        run_z.main(["--stage", "oracle"])
    assert e.value.code == 2


def test_cwd_refusal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_z.main(["--stage", "stage0"]) == 2


def test_exit_codes_and_report():
    assert run_z.exit_code(dict(outcome="PASS")) == 0
    assert run_z.exit_code(dict(outcome="INVALID")) == 5
    assert run_z.exit_code(dict(outcome="STOP_PILOT_FEW")) == 3
    out = dict(outcome="PASS", split=dict(C=dict(n=16)), env={"x": 1}, archive=[1], sentence="s")
    lines = run_z.report_lines("split", out, 2000)
    assert lines[0] == "stage split: outcome PASS (exit 0)" and lines[1] == "s"
    body = json.loads(lines[2])
    assert "env" not in body and "archive" not in body and body["split"]["C"]["n"] == 16
