"""scripts/run_aa.py: argument refusals, exit codes, and the printed report (QUIET fields left out)."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import aa_rules
from flymon.brain.aa_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_aa", ROOT / "scripts/run_aa.py")
run_aa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_aa)


def test_exit_codes():
    assert run_aa.exit_code(dict(outcome="PASS")) == 0
    assert run_aa.exit_code(dict(outcome="INVALID")) == 5
    for s in aa_rules.STOPS:
        assert run_aa.exit_code(dict(outcome=s)) == 3


def test_report_lines():
    out = dict(outcome="PASS", sentence="s", env={"x": 1}, archive=[1], git={}, decision_files={}, numbers={},
               synth={}, candidates=[], differential={}, lenient_axes=dict(b=11, a=20), filler="z" * 5000)
    lines = run_aa.report_lines("reuse", out, SPEC.cli_print_chars)
    assert lines[0] == "stage reuse: outcome PASS (exit 0)" and lines[1] == "s"
    assert len(lines[2]) <= SPEC.cli_print_chars
    for q in run_aa.QUIET:
        assert f'"{q}"' not in lines[2]
    short = run_aa.report_lines("reuse", dict(out, filler=""), SPEC.cli_print_chars)
    assert json.loads(short[2])["lenient_axes"] == dict(b=11, a=20)


def test_name_goes_with_archive_only():
    assert run_aa.main(["--stage", "archive"]) == 2
    assert run_aa.main(["--stage", "stage0", "--name", "stage0"]) == 2


def test_unknown_stage_is_an_argparse_error():
    with pytest.raises(SystemExit) as e:
        run_aa.main(["--stage", "judge"])
    assert e.value.code == 2


def test_cwd_refusal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_aa.main(["--stage", "stage0"]) == 2
