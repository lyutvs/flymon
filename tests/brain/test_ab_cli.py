"""scripts/run_ab.py: exit codes, the printed report (QUIET fields left out), argument refusals, the cwd refusal and
the spawn guard."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import ab_rules
from flymon.brain.ab_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_ab", ROOT / "scripts/run_ab.py")
run_ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_ab)


def test_exit_codes():
    assert run_ab.exit_code(dict(outcome="PASS")) == 0
    assert run_ab.exit_code(dict(outcome="INVALID")) == 5
    for s in ab_rules.STOPS:
        assert run_ab.exit_code(dict(outcome=s)) == 3
    for s in (ab_rules.FAIL, ab_rules.UNDECIDED):                    # a verdict label that is not PASS stops AB
        assert run_ab.exit_code(dict(outcome=s)) == 3


def test_report_lines_quiet():
    out = dict(outcome="STOP_FUTILE", sentence="s", filler="z" * 5000, stop_stage="0f",
               **{q: {"x": 1} for q in run_ab.QUIET})
    lines = run_ab.report_lines("futility", out, SPEC.cli_print_chars)
    assert lines[0] == "stage futility: outcome STOP_FUTILE (exit 3)" and lines[1] == "s"
    assert len(lines[2]) <= SPEC.cli_print_chars
    for q in run_ab.QUIET:
        assert f'"{q}"' not in lines[2]
    short = run_ab.report_lines("futility", dict(out, filler=""), SPEC.cli_print_chars)
    assert json.loads(short[2]) == dict(outcome="STOP_FUTILE", filler="", stop_stage="0f")
    for q in ("cal", "truth", "sel_counts", "ver_counts", "pairs", "manifest", "per_pair", "detail", "structures",
              "model", "rows", "grid"):
        assert q in run_ab.QUIET


@pytest.mark.parametrize("kw,ok", [
    (dict(stage="restart_preseal", note="n.json", name=None), True),
    (dict(stage="restart_preseal", note=None, name=None), False),
    (dict(stage="restart_preseal", note="n.json", name="1"), False),
    (dict(stage="defect", note="n.json", name=None), True),
    (dict(stage="defect", note=None, name=None), False),
    (dict(stage="reseal", note=None, name="1"), True),
    (dict(stage="reseal", note=None, name="x"), False),
    (dict(stage="recompute", note=None, name="verdict"), True),
    (dict(stage="recompute", note=None, name="cal_gate"), False),        # 0e / 0f: restart_preseal, never recompute
    (dict(stage="archive", note=None, name="set_v1"), True),
    (dict(stage="archive", note=None, name="judge"), False),
    (dict(stage="stage0", note=None, name="stage0"), False),
    (dict(stage="stage0", note="n.json", name=None), False),
    (dict(stage="futility", note=None, name=None), True),
])
def test_name_and_note_arguments(kw, ok):
    assert (run_ab.arg_reasons(argparse.Namespace(**kw), SPEC) == []) is ok, kw


def test_argument_refusals_exit_2():
    assert run_ab.main(["--stage", "archive"]) == 2
    assert run_ab.main(["--stage", "restart_preseal"]) == 2
    assert run_ab.main(["--stage", "stage0", "--note", "n.json"]) == 2


def test_unknown_stage():
    with pytest.raises(SystemExit) as e:
        run_ab.main(["--stage", "judge"])
    assert e.value.code == 2


def test_every_stage_is_accepted_by_the_parser():
    for st in SPEC.stages:
        assert run_ab.arg_reasons(argparse.Namespace(stage=st, note=None, name=None), SPEC) == []


def test_cwd_refusal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_ab.main(["--stage", "stage0"]) == 2
    assert run_ab.main(["--stage", "restart_preseal", "--note", "n.json"]) == 2


def test_main_guard():
    src = (ROOT / "scripts/run_ab.py").read_text().rstrip().splitlines()
    assert src[-2:] == ['if __name__ == "__main__":', "    sys.exit(main())"]
