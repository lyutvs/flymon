"""AC.6: extrapolation from the larger of the benchmark and rescope's 431.1 s/batch-battle, x 1.3, against 48 h;
sessions capped at min(24 h, the 60 h budget minus what stage 1 has spent); wall-clock sessions per arm."""
import json

import pytest

from flymon.ac import budget
from flymon.ac.spec import LABEL, SPEC


def test_counts():
    assert budget.batches(SPEC) == 60 and budget.n_sit_evals(SPEC) == 96


def test_extrapolation_uses_the_larger_rate():
    r = budget.extrapolate({"s_per_batch_battle": 500.0, "sit_eval_s_per_fly": 10.0})
    assert r["status"] == "OK" and r["s_per_batch_battle_used"] == 500.0
    assert r["hours"] == pytest.approx((500 * 60 + 10 * 96) * 1.3 / 3600)
    r = budget.extrapolate({"s_per_batch_battle": 100.0, "sit_eval_s_per_fly": 10.0})
    assert r["s_per_batch_battle_used"] == 431.1
    assert r["hours"] == pytest.approx((431.1 * 60 + 960) * 1.3 / 3600)


def test_over_48_hours_is_stop_budget():
    r = budget.extrapolate({"s_per_batch_battle": 3000.0, "sit_eval_s_per_fly": 10.0})
    assert r["status"] == "STOP_BUDGET" and r["hours"] > 48 and r["limit_h"] == 48.0


def test_session_cap_uses_the_remaining_budget():
    assert budget.session_cap_s(SPEC, 0.0) == 24 * 3600
    assert budget.session_cap_s(SPEC, 50 * 3600) == pytest.approx(10 * 3600)
    assert budget.session_cap_s(SPEC, 61 * 3600) <= 0


def test_sessions_are_recorded_and_summed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results/m4/stage1/BRAIN"
    i = budget.open_session(out, 1000.0)
    assert json.loads((out / "wall_clock.json").read_text())["sessions"][i]["status"] == "running"
    total = budget.close_session(out, i, 1000.0, "aborted", None)
    j = budget.open_session(out, 2000.0)
    total = budget.close_session(out, j, 2000.0, "ok", {"complete": True, "played": {"L": [1, 2]}})
    d = json.loads((out / "wall_clock.json").read_text())
    assert [s["status"] for s in d["sessions"]] == ["aborted", "ok"] and d["sessions"][1]["played"] == {"L": 2}
    assert d["label"] == LABEL
    assert total == pytest.approx(sum(s["seconds"] for s in d["sessions"]))
    rnd = tmp_path / "results/m4/stage1/RND"
    rnd.mkdir(parents=True)
    (rnd / "wall_clock.json").write_text(json.dumps({"sessions": [{"seconds": 7.0}, {"seconds": None}]}))
    assert budget.spent_s("results/m4/stage1") == pytest.approx(total + 7.0)
    assert budget.spent_s("results/m4/none") == 0.0


def test_script_writes_labelled_budget_and_exit_codes(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec_ = importlib.util.spec_from_file_location(
        "write_ac_budget", Path(__file__).resolve().parents[2] / "scripts/write_ac_budget.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    b = tmp_path / "bench.json"
    b.write_text(json.dumps({"s_per_batch_battle": 100.0, "sit_eval_s_per_fly": 10.0}))
    assert mod.main(["--bench", str(b)]) == 0
    d = json.loads((tmp_path / budget.BUDGET).read_text())
    assert d["label"] == LABEL and d["status"] == "OK" and d["bench_sha256"]
    b.write_text(json.dumps({"s_per_batch_battle": 3000.0, "sit_eval_s_per_fly": 10.0}))
    assert mod.main(["--bench", str(b)]) == 2
    with pytest.raises(SystemExit):
        mod.main(["--bench", str(tmp_path / "missing.json")])
