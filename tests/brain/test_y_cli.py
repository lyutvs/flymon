"""run_y.py: arguments, cwd refusal, exit codes, quiet output (no per-pair key printed; the oracle stage prints\nits outcome label and exit code only — Y red-team P0-1)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_y", ROOT / "scripts/run_y.py")
run_y = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_y)


def test_stages_and_exit_codes():
    assert run_y.STAGES == ("digest", "oracle", "stage0", "reuse", "pilot", "precheck")
    assert run_y.POOL_STAGES == ("oracle", "pilot")
    assert run_y.exit_code({"outcome": "PASS"}) == 0
    assert run_y.exit_code({"outcome": "STOP_FEW_PAIRS"}) == 3 and run_y.exit_code({"outcome": "STOP_REUSE"}) == 3
    assert run_y.exit_code({"outcome": "INVALID"}) == 5


def test_refuses_outside_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_y.main(["--stage", "digest"]) == 2


def test_quiet_keys_hide_records():
    assert {"set", "reuse", "git", "seeds", "z_V"} <= set(run_y.QUIET)


def test_oracle_prints_the_label_only():
    out = dict(outcome="STOP_FEW_PAIRS", sentence="... 3개로 ...", counts={"all": {"y_lenient": 3}}, n=249,
               detail_sha256="f" * 64, archive=[dict(dst="x", sha256="f" * 64)])
    lines = run_y.report_lines("oracle", out, 2000)
    assert lines == ["stage oracle: outcome STOP_FEW_PAIRS (exit 3)"]
    assert run_y.report_lines("oracle", dict(outcome="PASS", counts={}), 2000) == ["stage oracle: outcome PASS (exit 0)"]


def test_digest_prints_sentence_and_quiet_block():
    out = dict(outcome="STOP_REUSE", sentence="Y 재사용 조건", reasons=["r"], set={"keys": ["k"]}, git={}, archive=[])
    lines = run_y.report_lines("digest", out, 2000)
    assert lines[0] == "stage digest: outcome STOP_REUSE (exit 3)" and lines[1] == "Y 재사용 조건"
    assert '"set"' not in lines[2] and '"git"' not in lines[2] and '"reasons"' in lines[2]


def test_archive_stage_arguments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_y.ARCHIVE == "archive"
    assert run_y.main(["--stage", "archive"]) == 2                      # needs --name
    assert run_y.main(["--stage", "oracle", "--name", "oracle"]) == 2    # --name only with archive
    assert run_y.main(["--stage", "archive", "--name", "oracle"]) == 2   # outside the root
