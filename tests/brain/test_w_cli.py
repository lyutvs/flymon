"""scripts/run_w.py and W's files: arguments, exit codes, stage lists equal to the runner's; no W file imports the
frozen F v3 verdict.py or loads code by path; the only W measurement file is w_measure.py and no W pipeline module
defines a job; the CLI's pool code sits under `if __name__ == "__main__":`."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from flymon.brain import w_runner as WR

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_w", ROOT / "scripts/run_w.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)
W_FILES = sorted((ROOT / "flymon/brain").glob("w_*.py")) + [ROOT / "scripts/run_w.py"]


def test_args_and_exit_codes():
    a = lambda **k: SimpleNamespace(**dict(dict(stage=None, note=None, workers=None), **k))  # noqa: E731
    assert CLI.check_args(a()) == "--stage is required"
    assert CLI.check_args(a(stage="recompute")).startswith("--note is required")
    assert CLI.check_args(a(stage="path", note="x")).startswith("--note goes with")
    assert CLI.check_args(a(stage="path")) is None
    assert CLI.exit_code("path", dict(outcome="PASS")) == 0
    assert CLI.exit_code("path", dict(outcome="INVALID")) == 5
    assert CLI.exit_code("budget", dict(outcome="STOP_BUDGET")) == 3
    assert CLI.exit_code("smoke", dict(problems=["x"])) == 6 and CLI.exit_code("smoke", dict(problems=[])) == 0
    assert CLI.exit_code("seal", dict(status="NOT_READ")) == 6 and CLI.exit_code("judge", dict(verdict="FAIL")) == 0
    assert set(CLI.GATE_STAGES) == set(WR.GATES) and set(WR.ORDER) <= set(CLI.STAGES)


def test_no_frozen_verdict_and_no_code_by_path():
    for p in W_FILES:
        src = p.read_text()
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "m2-learning-f-v3" not in src and "spec_from_file_location" not in names, p.name
        assert not {"exec", "runpy", "exec_module"} & names, p.name


def test_only_w_measure_defines_a_job():
    for p in W_FILES:
        tree = ast.parse(p.read_text())
        jobs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.endswith("_job")]
        assert jobs == (["w_learn_job"] if p.name == "w_measure.py" else []), p.name


def test_main_guard():
    tree = ast.parse((ROOT / "scripts/run_w.py").read_text())
    last = tree.body[-1]
    assert isinstance(last, ast.If) and "__name__" in ast.unparse(last.test)
