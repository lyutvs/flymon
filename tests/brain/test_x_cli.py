"""scripts/run_x.py: stages equal the runner's ORDER, exit codes, refusals outside the repository root."""
import importlib.util
from pathlib import Path

from flymon.brain import x_runner as XR

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_x", ROOT / "scripts/run_x.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


def test_stages_and_exit_codes():
    assert tuple(CLI.STAGES) == XR.ORDER
    assert CLI.exit_code(dict(outcome="PASS")) == 0
    assert CLI.exit_code(dict(outcome="INVALID")) == 5
    assert CLI.exit_code(dict(outcome="STOP_REUSE")) == 3 and CLI.exit_code(dict(outcome="STOP_OC_UNREACHABLE")) == 3


def test_refuses_outside_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert CLI.main(["--stage", "stage0"]) == 2
