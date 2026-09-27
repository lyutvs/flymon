"""M4 statistics: win tables, the paired fly -> battle bootstrap, the (2a)/(2b) verdict (INVALID on partial or
malformed input, never PASS / FAIL), the recorded information-turn match and the write_rescope_m4 CLI."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.rescope import stats
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]


def arm(rates, n=200, seed=0, invalid=()):
    rng = np.random.default_rng(seed)
    return {"per_fly": [{"fly": k, "invalid": k in invalid,
                         "eval_battles": [{"won": bool(w), "finished": True} for w in rng.random(n) < r]}
                        for k, r in enumerate(rates)]}


def five(n_flies=12, n=200, fly=0.5, coff=0.2, rs=0.5, rnd=0.15, mx=0.47):
    return {"FLY": arm([fly] * n_flies, n, seed=1), "COFF": arm([coff] * n_flies, n, seed=2),
            "RS": arm([rs] * n_flies, n, seed=3), "RND": arm([rnd] * n_flies, n, seed=4),
            "MAX": arm([mx] * n_flies, n, seed=5)}


# ---- the brief's tests ----------------------------------------------------------------------------
def test_clear_difference_detected():
    out = stats.paired_boot(stats.win_table(arm([0.5] * 12, seed=1)), stats.win_table(arm([0.2] * 12, seed=2)), 2000, 0)
    assert out["lo"] > 0 and abs(out["diff"] - 0.3) < 0.05


def test_no_difference_ci_contains_zero():
    out = stats.paired_boot(stats.win_table(arm([0.3] * 12, seed=3)), stats.win_table(arm([0.3] * 12, seed=4)), 2000, 0)
    assert out["lo"] < 0 < out["hi"]


def test_invalid_flies_dropped_pairwise():
    a = stats.win_table(arm([0.5] * 4, invalid=(1,)))
    b = stats.win_table(arm([0.2] * 4))
    assert stats.paired_boot(a, b, 200, 0)["n_pairs"] == 3


def test_2b_needs_both():
    v = stats.m4_verdict(five(), SPEC)
    assert v["status"] == stats.OK
    assert v["2a"]["pass"] is True and v["2a"]["verdict"] == stats.PASS
    assert v["2b"]["vs_coff"]["lo"] > 0 and v["2b"]["pass"] is False and v["2b"]["verdict"] == stats.FAIL


# ---- win table and bootstrap ----------------------------------------------------------------------
def test_unfinished_counts_as_loss():
    res = {"per_fly": [{"fly": 0, "invalid": False, "eval_battles": [
        {"won": True, "finished": True}, {"won": None, "finished": False}, {"won": False, "finished": True}]}]}
    assert stats.win_table(res)[0].tolist() == [1.0, 0.0, 0.0]


def test_boot_is_deterministic_and_paired():
    a, b = stats.win_table(arm([0.5] * 8, 40, seed=1)), stats.win_table(arm([0.3] * 8, 40, seed=2))
    assert stats.paired_boot(a, b, 1000, 7) == stats.paired_boot(a, b, 1000, 7)
    # identical arms: every draw picks the same flies for both, so a zero-variance fly level gives diff 0 exactly
    same = stats.paired_boot(a, a, 500, 0)
    assert same["diff"] == 0.0 and same["lo"] < 0 < same["hi"]      # battle level still resampled per arm


def test_fly_level_resampled_jointly():
    """A big between-fly spread shared by both arms cancels in the paired difference: the CI is tight."""
    rates = np.linspace(0.1, 0.9, 12)
    a = stats.win_table(arm(list(rates + 0.05), 400, seed=1))
    b = stats.win_table(arm(list(rates - 0.05), 400, seed=2))
    out = stats.paired_boot(a, b, 2000, 0)
    assert out["lo"] > 0 and out["hi"] - out["lo"] < 0.1


# ---- invalid pairs and malformed input ------------------------------------------------------------
def test_rs_donor_invalid_drops_the_pair():
    arms = five()
    arms["RS"]["per_fly"][3]["donor_invalid"] = True          # FLY 3 invalid upstream; the RS row itself is not
    arms["RS"]["per_fly"][3]["invalid"] = False
    v = stats.m4_verdict(arms, SPEC)
    assert 3 not in v["2b"]["vs_rs"]["flies"] and v["2b"]["vs_rs"]["n_pairs"] == 11
    assert v["invalid_pairs"]["vs_rs"]["flies"] == [3] and v["invalid_pairs"]["vs_rs"]["n"] == 1
    assert v["invalid_pairs"]["vs_coff"]["n"] == 0 and 3 in v["invalid_flies"]["RS"]


def test_fly_row_invalid_drops_every_fly_pair():
    arms = five()
    arms["FLY"]["per_fly"][5]["invalid"] = True
    v = stats.m4_verdict(arms, SPEC)
    assert all(v["invalid_pairs"][k]["flies"] == [5] for k in ("2a", "vs_coff", "vs_rs"))


def test_rs_residual_over_max_is_invalid_even_if_unflagged():
    arms = five()
    arms["RS"]["per_fly"][0].update(residual_frac=0.06, invalid=False)
    assert stats.m4_verdict(arms, SPEC)["invalid_pairs"]["vs_rs"]["flies"] == [0]


def test_too_few_valid_pairs_invalid():
    arms = five(n_flies=8)
    for k in (0, 1, 2):
        arms["RS"]["per_fly"][k]["donor_invalid"] = True       # 5 valid pairs < 6 (6/8)
    v = stats.m4_verdict(arms, SPEC)
    assert v["status"] == stats.INVALID and v["2b"]["verdict"] == stats.INVALID and v["2b"]["pass"] is None
    assert v["2a"]["verdict"] == stats.INVALID and v["2a"]["pass"] is None
    arms["RS"]["per_fly"][2]["donor_invalid"] = False          # 6 valid pairs: judged
    assert stats.m4_verdict(arms, SPEC)["status"] == stats.OK


@pytest.mark.parametrize("break_it", [
    lambda a: a.pop("MAX"),
    lambda a: a["RS"].__setitem__("per_fly", []),
    lambda a: a["COFF"]["per_fly"].pop(4),
    lambda a: a["COFF"]["per_fly"].append(copy.deepcopy(a["COFF"]["per_fly"][0])),
    lambda a: a["FLY"]["per_fly"][2]["eval_battles"][0].__setitem__("won", float("nan")),
    lambda a: a["FLY"]["per_fly"][2]["eval_battles"][0].__setitem__("won", 1),
    lambda a: a["FLY"]["per_fly"][2]["eval_battles"][0].pop("won"),
    lambda a: a["RND"]["per_fly"][1].__setitem__("eval_battles", a["RND"]["per_fly"][1]["eval_battles"][:-1]),
    lambda a: a["RS"]["per_fly"][1].__setitem__("residual_frac", float("nan")),
    lambda a: a["FLY"].__setitem__("complete", False),
    lambda a: a["FLY"]["per_fly"][0].__setitem__("invalid", "no"),
], ids=["missing-arm", "empty-arm", "missing-row", "duplicate-row", "nan-won", "int-won", "no-won",
        "short-eval", "nan-residual", "incomplete", "non-bool-invalid"])
def test_malformed_input_is_invalid(break_it):
    arms = five(n_flies=8, n=40)
    for d in arms.values():
        d.update(flies=8, eval=40, complete=True)
    assert stats.m4_verdict(arms, SPEC)["status"] == stats.OK
    break_it(arms)
    v = stats.m4_verdict(arms, SPEC)
    assert v["status"] == stats.INVALID and v["reasons"]
    assert v["2a"]["pass"] is None and v["2b"]["pass"] is None
    assert v["2a"]["verdict"] == v["2b"]["verdict"] == stats.INVALID


def test_different_fly_counts_invalid():
    arms = five()
    arms["MAX"] = arm([0.47] * 10, seed=5)
    assert stats.m4_verdict(arms, SPEC)["status"] == stats.INVALID


def test_recorded():
    rec = stats.m4_verdict(five(), SPEC)["recorded"]
    wr = rec["win_rate"]
    assert rec["max_minus_fly"] == pytest.approx(wr["MAX"] - wr["FLY"])
    assert rec["fly_rnd_ratio"] == pytest.approx(wr["FLY"] / wr["RND"])
    assert rec["fly_max_ratio"] == pytest.approx(wr["FLY"] / wr["MAX"])


# ---- information-turn match (recorded only) -------------------------------------------------------
def test_info_turn_match():
    recs = [dict(kind="decision", decider="fly", candidates=["a", "b"], multipliers=[2.0, 1.0], chosen="a"),   # hit
            dict(kind="decision", decider="fly", candidates=["a", "b", "c"], multipliers=[1.0, 0.5, 2.0], chosen="b"),
            dict(kind="decision", decider="fly", candidates=["a", "b"], multipliers=[1.0, 1.0], chosen="b")]   # no info
    assert stats.info_turn_match(recs) == pytest.approx(0.5)
    s = stats.info_turn_stats(recs + [dict(kind="decision", decider="coach", candidates=[], chosen="x"),
                                      dict(kind="decision", decider="fly", candidates=["a", "b"], chosen="a")])
    assert (s["n_info"], s["n_fly_turns"], s["n_without_multipliers"]) == (2, 4, 1)
    assert s["info_frac"] == pytest.approx(2 / 3)
    assert stats.info_turn_match([]) is None


# ---- the CLI ---------------------------------------------------------------------------------------
def load(dirty=False):
    spec = importlib.util.spec_from_file_location("wm4", ROOT / "scripts/write_rescope_m4.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.git_provenance = lambda files=(): {"commit": "c" * 40, "dirty": dirty, "dirty_files": ["x.py"] if dirty else []}
    return mod


def test_cli_refuses_dirty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    with pytest.raises(SystemExit):
        load(dirty=True).main(["--phase", "judge"])
    assert load(dirty=True).main(["--phase", "judge", "--allow-dirty"]) == 0


def lay_out(tmp_path, phase="judge", smoke=False, n_flies=8, n=40, **kw):
    root = tmp_path / ("results/rescope-smoke" if smoke else "results/rescope") / phase
    for name, d in five(n_flies, n, **kw).items():
        d.update(arm=name, phase=phase, smoke=smoke, complete=True, flies=n_flies, eval=n,
                 schedule_digests={"learn": None, "eval": "e" * 64})
        if name == "RS":
            for r in d["per_fly"]:
                r.update(donor_invalid=False, residual_frac=0.01, donor_sha256="d" * 64)
        p = root / name / "result.json"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    (root / "eval_schedule.json").write_text(json.dumps({"digest": "e" * 64, "schedule": []}))
    return root


def test_cli_judge_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    log = tmp_path / "results/rescope/judge/FLY/logs/eval/fly00.jsonl"; log.parent.mkdir(parents=True)
    log.write_text(json.dumps(dict(kind="decision", decider="fly", candidates=["a", "b"], multipliers=[2, 1],
                                   chosen="a")) + "\n")
    assert load().main(["--phase", "judge", "--allow-dirty"]) == 0
    s = json.loads((tmp_path / "results/summary/rescope_m4.json").read_text())
    assert s["status"] == "OK" and s["2a"]["verdict"] == "PASS" and s["2b"]["verdict"] == "FAIL"
    assert s["arms"]["RS"]["rs"]["0"]["donor_sha256"] == "d" * 64 and s["arms"]["FLY"]["n_eval"] == 40
    assert s["recorded"]["info_turn_match"]["FLY"]["rate"] == 1.0
    assert s["boot_draws"] == SPEC.boot_draws and s["boot_seed"] == SPEC.boot_seed
    with pytest.raises(SystemExit):                              # the judged file is not overwritten silently
        load().main(["--phase", "judge", "--allow-dirty"])
    assert not (tmp_path / "results/rescope-smoke").exists()


def test_cli_smoke_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, phase="pilot", smoke=True, n_flies=2, n=2)
    load().main(["--smoke", "--phase", "pilot", "--allow-dirty"])
    assert (tmp_path / "results/rescope-smoke/summary/rescope_m4.json").exists()
    assert not (tmp_path / "results/summary").exists()


def test_cli_refuses_pilot_without_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, phase="pilot")
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--allow-dirty"])


@pytest.mark.parametrize("break_it", ["missing", "digest", "schedule-file", "phase", "unreadable"])
def test_cli_bad_input_writes_invalid(tmp_path, monkeypatch, break_it):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    p = root / "COFF" / "result.json"
    if break_it == "missing":
        p.unlink()
    elif break_it == "unreadable":
        p.write_text("{not json")
    elif break_it == "schedule-file":
        (root / "eval_schedule.json").write_text(json.dumps({"digest": "f" * 64}))
    else:
        d = json.loads(p.read_text())
        if break_it == "digest":
            d["schedule_digests"]["eval"] = "f" * 64
        else:
            d["phase"] = "pilot"
        p.write_text(json.dumps(d))
    assert load().main(["--phase", "judge", "--allow-dirty"]) == 2
    s = json.loads((tmp_path / "results/summary/rescope_m4.json").read_text())
    assert s["status"] == "INVALID" and s["2a"]["pass"] is None and s["2b"]["pass"] is None
    assert s["2a"]["verdict"] == s["2b"]["verdict"] == "INVALID"
