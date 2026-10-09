"""AD's numbers (spec appendix AD.1-AD.8) and where AD may write."""
import subprocess

import pytest

from flymon.ac.spec import LABEL as AC_LABEL
from flymon.ad import store
from flymon.ad.spec import FORBIDDEN, LABEL, SPEC, bench, smoke


def test_arms_layout_and_ids():
    assert dict(SPEC.arms) == {"FLYL": 12, "FLYOS": 12} and SPEC.n_flies() == 24 and SPEC.n_rows() == 12
    lay = SPEC.layout()
    assert lay[0] == ("FLYL", 0, 24) and lay[11] == ("FLYL", 11, 35)
    assert lay[12] == ("FLYOS", 0, 36) and lay[23] == ("FLYOS", 11, 47)
    assert SPEC.tags == ("DL", "DE") and SPEC.out_root == "results/m4/ad"
    assert SPEC.min_valid_of() == {"FLYL": 9, "FLYOS": 9, "COFF": 4, "FLY1": 9, "RND": 12, "MAX": 12}
    assert SPEC.coff_flies == (12, 13, 14, 15, 16, 17) and SPEC.fly1_flies == tuple(range(12))


def test_seeds_and_thresholds():
    assert SPEC.seeds() == {"gen": 301, "eval": 303, "boot": 304, "tie": 305, "learn": 306, "os": 307, "new": 308}
    assert 302 not in SPEC.seeds().values()
    assert (SPEC.boot_draws, SPEC.min_effect, SPEC.retry_max) == (10_000, 0.15, 3)
    assert (SPEC.os_gate_draws, SPEC.os_freq_tol, SPEC.os_chi2_p, SPEC.strength_tol) == (16_000, 0.01, 0.001, 1e-9)
    assert (SPEC.kc_band_min, SPEC.kc_band_sd, SPEC.calib_reps) == (0.15, 3.0, (0, 1, 2))
    assert (SPEC.floor_read, SPEC.taurec_pulses) == (0.5, 1000)


def test_points_and_learning_length():
    assert (SPEC.learn_battles, SPEC.fallback_battles, SPEC.futility_point, SPEC.eval_battles) == (120, 80, 40, 20)
    assert SPEC.points() == (0, 20, 40, 60, 80, 100, 120) and SPEC.new_points() == (0, 40, 120)
    assert SPEC.points(80) == (0, 20, 40, 60, 80) and SPEC.new_points(80) == (0, 40, 80)
    assert SPEC.points(40) == (0, 20, 40) and SPEC.new_points(40) == (0, 40)
    assert SPEC.futility_checked(120) and SPEC.futility_checked(80) and not SPEC.futility_checked(40)


def test_budget_numbers():
    assert round(SPEC.budget_limit_h(), 3) == 47.418
    assert (SPEC.sit_evals(120, True), SPEC.sit_evals(120, False)) == (324, 252)
    assert (SPEC.sit_evals(80, True), SPEC.sit_evals(80, False)) == (252, 180)
    assert (SPEC.batches(120), SPEC.batches(80)) == (140, 100)
    assert SPEC.sit_evals_at(40, 120, True) == 60 and SPEC.sit_evals_at(20, 120, True) == 36
    assert SPEC.sit_evals_at(40, 120, False) == 36 and SPEC.sit_evals_at(30, 120, True) == 0
    assert (SPEC.bench_valid, SPEC.bench_max_attempts, SPEC.busy_cpu, SPEC.busy_samples) == (3, 6, 50.0, 3)
    assert (SPEC.worst_factor, SPEC.stage1_spent_s, SPEC.reserve_h) == (2.0, 41_695.3, 1.0)


def test_smoke_and_bench():
    s = smoke()
    assert dict(s.arms) == {"FLYL": 2, "FLYOS": 2} and s.layout()[2] == ("FLYOS", 0, 26)
    assert s.points() == (0, 1, 2) and s.new_points() == (0, 1, 2) and s.futility_checked()
    assert (s.mode, s.out_root, s.tags, s.n_pairs, s.n_new_pairs) == ("smoke", "results/m4-smoke/ad", ("XL", "XE"), 4, 4)
    assert s.gen_seed == 305 and s.min_effect == SPEC.min_effect and s.os_seed == SPEC.os_seed
    b = bench(2)
    assert dict(b.arms) == dict(SPEC.arms) and b.learn_battles == 2 and b.eval_battles == 0
    assert b.points() == (0,) and b.new_points() == (0,) and not b.futility_checked()
    assert (b.mode, b.out_root, b.tags, b.bench_attempt) == ("bench", "results/m4-bench/ad-2", ("YL", "YE"), 2)


def test_label_is_ac5_verbatim():
    assert LABEL == AC_LABEL and FORBIDDEN == ("PASS", "FAIL", "학습된다", "학습 안 됨")


def test_guard_allows_only_ad_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store.write_json("results/m4/ad/x.json", {"b": 1, "a": 2})
    assert (tmp_path / "results/m4/ad/x.json").read_text().startswith('{\n "a": 2')
    store.write_json("results/summary/ad_x.json", {})
    store.write_json("results/m4-bench/ad-1/bench.json", {})
    store.write_json("results/m4-smoke/ad/result.json", {})
    for bad in ("results/summary/ac_x.json", "results/m4/stage1/x.json", "results/summary/ad_x.txt", "x.json"):
        with pytest.raises(SystemExit):
            store.write_json(bad, {})


def test_code_changes_since(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (tmp_path / "flymon").mkdir(); (tmp_path / "flymon/a.py").write_text("x = 1\n")
    (tmp_path / "notes.md").write_text("n\n")
    run("add", "."); run("commit", "-qm", "c1")
    c1 = store.head_commit()
    assert store.code_changes_since(c1, ["flymon"]) == []
    (tmp_path / "notes.md").write_text("changed\n")
    run("commit", "-qam", "c2")
    assert store.code_changes_since(c1, ["flymon"]) == []            # a later commit outside the code paths
    (tmp_path / "flymon/a.py").write_text("x = 2\n")
    assert store.code_changes_since(c1, ["flymon"]) == ["flymon/a.py"]   # uncommitted
    run("commit", "-qam", "c3")
    assert store.code_changes_since(c1, ["flymon"]) == ["flymon/a.py"]   # committed after c1
    (tmp_path / "flymon/b.py").write_text("y = 1\n")
    assert store.code_changes_since(c1, ["flymon"]) == ["flymon/a.py", "flymon/b.py"]   # untracked
