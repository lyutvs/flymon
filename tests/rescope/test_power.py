"""Stage 4 sizing: variance components, joint (2b) power, the cheapest (F, E), budget / power stops and the
write_rescope_power CLI (refusal on partial / malformed pilot input, never a silent size)."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.rescope import power
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]


def comp(p, vb=0.0): return {"p": p, "var_between": vb, "n_flies": 6, "n_battles": 20}


# ---- the brief's tests ----------------------------------------------------------------------------
def test_components_moment_estimate():
    rng = np.random.default_rng(0)
    table = {k: (rng.random(400) < 0.3).astype(float) for k in range(40)}
    c = power.components(table)
    assert abs(c["p"] - 0.3) < 0.02 and c["var_between"] < 0.002


def test_power_grows_with_E():
    cs = {"FLY": comp(0.3), "RS": comp(0.3), "COFF": comp(0.3), "RND": comp(0.15)}
    lo = power.joint_power(cs, 12, 40, SPEC)["p_2b"]; hi = power.joint_power(cs, 12, 400, SPEC)["p_2b"]
    assert hi > lo and 0 <= lo <= 1


def test_joint_below_single():
    cs = {"FLY": comp(0.3, 0.001), "RS": comp(0.3, 0.001), "COFF": comp(0.3), "RND": comp(0.15)}
    r = power.joint_power(cs, 16, 200, SPEC)
    assert r["p_2b"] <= min(r["p_coff"], r["p_rs"]) + 1e-9


def test_choose_min_wall_and_stops():
    cs = {"FLY": comp(0.3), "RS": comp(0.3), "COFF": comp(0.3), "RND": comp(0.15)}
    rates = {"sec_per_batch_battle_brain": 1.0, "sec_per_batch_battle_nobrain": 0.1}
    out = power.choose(cs, rates, SPEC)
    assert out["status"] == "SIZED" and out["p_2b"] >= 0.8
    slow = {"sec_per_batch_battle_brain": 600.0, "sec_per_batch_battle_nobrain": 1.0}
    assert power.choose(cs, slow, SPEC)["status"] == "STOP_BUDGET"
    tiny = dict(cs, FLY=comp(0.3, 0.2), RS=comp(0.3, 0.2))
    assert power.choose(tiny, rates, SPEC, grid_F=(2,), grid_E=(20,))["status"] == "STOP_POWER"


# ---- extra unit checks ----------------------------------------------------------------------------
def test_components_between_fly_variance_and_guard():
    rng = np.random.default_rng(1)
    ps = rng.uniform(0.1, 0.5, 60)
    c = power.components({k: (rng.random(2000) < p).astype(float) for k, p in enumerate(ps)})
    assert c["var_between"] == pytest.approx(ps.var(ddof=1), rel=0.25) and c["n_battles"] == 2000
    with pytest.raises(ValueError):
        power.components({0: np.ones(5)})
    with pytest.raises(ValueError):
        power.components({0: np.ones(5), 1: np.array([])})


def test_joint_power_matches_normal_theory_and_is_deterministic():
    cs = {"FLY": comp(0.3), "RS": comp(0.3), "COFF": comp(0.3), "RND": comp(0.15)}
    r = power.joint_power(cs, 16, 200, SPEC)
    se = np.sqrt(0.35 * 0.65 / 3200 + 0.3 * 0.7 / 3200)
    from math import erf, sqrt
    single = 1 - 0.5 * (1 + erf((1.96 - SPEC.mie / se) / sqrt(2)))
    assert r["p_coff"] == pytest.approx(single, abs=0.01) and r["p_rs"] == pytest.approx(single, abs=0.01)
    assert r["p_2b"] > r["p_coff"] * r["p_rs"]                 # shared FLY noise: positively correlated
    assert r == power.joint_power(cs, 16, 200, SPEC)
    assert r["p_2a"] > 0.99                                     # pilot gap 0.15


def test_wall_hours_formula():
    rates = {"sec_per_batch_battle_brain": 2.0, "sec_per_batch_battle_nobrain": 0.5}
    L = SPEC.learn_battles
    one = ((L + 20) * 2.0 * 2 + 20 * 2.0 + 2 * 20 * 0.5) / 3600          # one batch
    assert power.wall_hours(8, 20, rates, SPEC, 16) == pytest.approx(one)
    assert power.wall_hours(17, 20, rates, SPEC, 16) == pytest.approx(2 * one)
    assert power.wall_hours(32, 20, rates, SPEC, 8) == pytest.approx(4 * one)
    with pytest.raises(ValueError):
        power.wall_hours(8, 20, rates, SPEC, 0)


def test_partial_batch_costs_a_full_batch():
    """Batch model (R11): with W = 16, F = 6 and F = 16 both take one batch - equal hours."""
    rates = {"sec_per_batch_battle_brain": 1.5, "sec_per_batch_battle_nobrain": 0.2}
    assert power.wall_hours(6, 80, rates, SPEC, 16) == power.wall_hours(16, 80, rates, SPEC, 16)
    assert power.wall_hours(24, 80, rates, SPEC, 16) == 2 * power.wall_hours(16, 80, rates, SPEC, 16)


def test_choose_uses_the_judge_workers():
    cs = {"FLY": comp(0.3, 0.002), "RS": comp(0.3, 0.002), "COFF": comp(0.3), "RND": comp(0.15)}
    rates = {"sec_per_batch_battle_brain": 1.0, "sec_per_batch_battle_nobrain": 0.1}
    w16, w8 = power.choose(cs, rates, SPEC, workers=16), power.choose(cs, rates, SPEC, workers=8)
    for out, W in ((w16, 16), (w8, 8)):
        assert all(r["hours"] == pytest.approx(power.wall_hours(r["F"], r["E"], rates, SPEC, W)) for r in out["table"])
    assert w8["hours"] >= w16["hours"]


def test_choose_picks_cheapest_powered_point():
    cs = {"FLY": comp(0.3, 0.002), "RS": comp(0.3, 0.002), "COFF": comp(0.3), "RND": comp(0.15)}
    rates = {"sec_per_batch_battle_brain": 1.0, "sec_per_batch_battle_nobrain": 0.1}
    out = power.choose(cs, rates, SPEC)
    ok = [r for r in out["table"] if r["p_2b"] >= SPEC.power_target]
    assert out["hours"] == min(r["hours"] for r in ok)
    assert len(out["table"]) == len(power.GRID_F) * len(power.GRID_E)


def test_stop_budget_still_reports_best():
    cs = {"FLY": comp(0.3), "RS": comp(0.3), "COFF": comp(0.3), "RND": comp(0.15)}
    out = power.choose(cs, {"sec_per_batch_battle_brain": 600.0, "sec_per_batch_battle_nobrain": 1.0}, SPEC)
    assert out["status"] == "STOP_BUDGET" and out["F"] and out["E"] and out["hours"] > SPEC.budget_hours


# ---- the CLI ---------------------------------------------------------------------------------------
def load(dirty=False):
    spec = importlib.util.spec_from_file_location("wpow", ROOT / "scripts/write_rescope_power.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.git_provenance = lambda files=(): {"commit": "c" * 40, "dirty": dirty, "dirty_files": ["x.py"] if dirty else []}
    return mod


RATES = {"FLY": 0.3, "RS": 0.3, "COFF": 0.3, "RND": 0.15}
WALL = {"FLY": 3600.0, "RS": 3600.0, "COFF": 1200.0, "RND": 12.0}


def sessions(*secs, status="ok"):
    return {"sessions": [dict(started_utc="2026-09-28T00:00:00+00:00", ended_utc="2026-09-28T01:00:00+00:00",
                              seconds=x, status=status, played=None, complete=True) for x in secs]}


def write_sessions(root, arm, d):
    (root / arm / "wall_clock.json").write_text(json.dumps(d))


def lay_out(tmp_path, smoke=False, n_flies=6, learn=40, n=20, seed=0, logs=True):
    mod = load()
    root = tmp_path / ("results/rescope-smoke" if smoke else "results/rescope") / "pilot"
    dig = mod.expected_digests(n_flies, learn, n)
    rng = np.random.default_rng(seed)
    sha = {}
    for k in range(n_flies):                                     # FLY k's learning log (the RS donor), always there
        f = root / "FLY" / "logs" / f"fly{k:02d}.jsonl"; f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps(dict(kind="decision", decider="fly")) + "\n" for _ in range(3 * learn))
                     + json.dumps(dict(kind="decision", decider="coach")) + "\n")
        sha[k] = hashlib.sha256(f.read_bytes()).hexdigest()
    for name, p in RATES.items():
        lr = learn if name in ("FLY", "RS") else 0
        rows = [{"fly": k, "invalid": False,
                 "eval_battles": [{"won": bool(w), "finished": True} for w in rng.random(n) < p]}
                for k in range(n_flies)]
        if name == "FLY":
            for r in rows:
                r.update(learn_log_sha256=sha[r["fly"]])
        if name == "RS":
            for r in rows:
                r.update(donor_invalid=False, residual_frac=0.01, donor_sha256=sha[r["fly"]])
        d = dict(arm=name, phase="pilot", smoke=smoke, complete=True, flies=n_flies, learn=lr, eval=n,
                 per_fly=rows, wall_clock_s=WALL[name], m_overlap_note=None, workers=16,
                 schedule_digests={"learn": dig["learn"] if lr else None, "eval": dig["eval"]},
                 logs=dict(learn="logs" if lr else None, eval="logs/eval", retries="<logs>/retries"))
        a = root / name; a.mkdir(parents=True, exist_ok=True)
        (a / "result.json").write_text(json.dumps(d))
        write_sessions(root, name, sessions(WALL[name]))
        if logs and name in ("FLY", "RS", "COFF"):
            for k in range(n_flies):
                for sub, nb in (("logs/eval", n), ("logs", lr)):
                    if nb:
                        f = a / sub / f"fly{k:02d}.jsonl"; f.parent.mkdir(parents=True, exist_ok=True)
                        f.write_text("".join(json.dumps(dict(kind="decision", decider="fly")) + "\n"
                                             for _ in range(3 * nb))
                                     + json.dumps(dict(kind="decision", decider="coach")) + "\n")
    (root / "eval_schedule.json").write_text(json.dumps({"digest": dig["eval"], "schedule": []}))
    return root


def edit(root, arm, fn):
    p = root / arm / "result.json"
    d = json.loads(p.read_text()); fn(d); p.write_text(json.dumps(d))


def test_cli_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    code = load().main(["--m-overlap", "yes", "--allow-dirty"])
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    assert s["status"] in power.STATUSES and code == (0 if s["status"] == "SIZED" else 2)
    assert set(s["components"]) == {"FLY", "RS", "COFF", "RND"} and s["components"]["FLY"]["n_flies"] == 6
    assert s["rates"]["sec_per_batch_battle_brain"] == pytest.approx(3600.0 / (1 * 60))   # 6 flies, 16 workers: 1 batch
    assert s["rates"]["sec_per_batch_battle_nobrain"] == pytest.approx(12.0 / (1 * 20))
    assert s["workers"] == 16 and s["rate_record"]["per_arm"]["FLY"]["batches"] == 1
    assert s["rate_record"]["sources"]["sec_per_batch_battle_brain"] == "FLY"
    assert s["m_overlap"] == "yes" and s["m_overlap_hours_note"]
    assert "사전 추정" in s["pre_estimate"]["text"] and "40–50시간" in s["pre_estimate"]["text"]
    dec = s["fly_decisions"]["per_arm"]["FLY"]
    assert dec["eval"]["per_battle"] == 3.0 and dec["learn"]["per_battle"] == 3.0 and not dec["missing_logs"]
    assert s["power_draws"] == SPEC.power_draws and s["seed"] == SPEC.boot_seed
    assert len(s["table"]) == 40
    with pytest.raises(SystemExit, match="exists"):             # not overwritten silently
        load().main(["--m-overlap", "yes", "--allow-dirty"])
    assert load().main(["--m-overlap", "no", "--allow-dirty", "--force"]) in (0, 2)
    assert json.loads((tmp_path / "results/summary/rescope_power.json").read_text())["m_overlap"] == "no"
    assert not (tmp_path / "results/rescope-smoke").exists()


def test_cli_sized_status_feeds_judge(tmp_path, monkeypatch):
    """A low-variance, fast pilot is SIZED with integer F / E the judge battle CLI can check."""
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    for arm in ("FLY", "RS", "COFF", "RND"):
        edit(root, arm, lambda d: d.update(wall_clock_s=36.0))
        write_sessions(root, arm, sessions(36.0))
    assert load().main(["--m-overlap", "no", "--allow-dirty"]) == 0
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    assert s["status"] == "SIZED" and s["F"] in power.GRID_F and s["E"] in power.GRID_E
    assert s["hours"] <= SPEC.budget_hours and s["fly_decisions"]["projected_at_choice"] > 0


def test_cli_smoke_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, smoke=True, n_flies=2, learn=2, n=2)
    load().main(["--smoke", "--m-overlap", "no", "--allow-dirty"])
    assert (tmp_path / "results/rescope-smoke/summary/rescope_power.json").exists()
    assert not (tmp_path / "results/summary").exists()


def test_cli_requires_m_overlap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    with pytest.raises(SystemExit):
        load().main(["--allow-dirty"])
    with pytest.raises(SystemExit):
        load().main(["--allow-dirty", "--m-overlap", "maybe"])


def test_cli_refuses_dirty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    with pytest.raises(SystemExit, match="uncommitted"):
        load(dirty=True).main(["--m-overlap", "no"])
    assert not (tmp_path / "results/summary").exists()


def _drop_row(d): d["per_fly"].pop()
def _nan_won(d): d["per_fly"][0]["eval_battles"][0]["won"] = "yes"
def _incomplete(d): d["complete"] = False
def _other_phase(d): d["phase"] = "judge"
def _bad_eval_digest(d): d["schedule_digests"]["eval"] = "f" * 64
def _bad_learn_digest(d): d["schedule_digests"]["learn"] = "f" * 64
def _wall_none(d): d["wall_clock_s"] = None
def _wall_missing(d): d.pop("wall_clock_s")
def _wall_nan(d): d["wall_clock_s"] = float("nan")
def _wall_zero(d): d["wall_clock_s"] = 0
def _flies_str(d): d["flies"] = "6"
def _no_workers(d): d.pop("workers")
def _zero_workers(d): d["workers"] = 0


@pytest.mark.parametrize("arm, fn, match", [
    ("RS", _drop_row, "missing or duplicate"), ("COFF", _nan_won, "won"), ("RND", _incomplete, "complete"),
    ("FLY", _other_phase, "phase"), ("COFF", _bad_eval_digest, "eval schedule digest"),
    ("RS", _bad_learn_digest, "learn schedule digest"), ("FLY", _wall_none, "wall_clock_s"),
    ("RND", _wall_missing, "wall_clock_s"), ("FLY", _wall_nan, "wall_clock_s"), ("RND", _wall_zero, "wall_clock_s"),
    ("FLY", _flies_str, "integer"), ("FLY", _no_workers, "workers"), ("RND", _zero_workers, "workers")])
def test_cli_refuses_malformed_pilot(tmp_path, monkeypatch, arm, fn, match):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    edit(root, arm, fn)
    with pytest.raises(SystemExit, match=match):
        load().main(["--m-overlap", "no", "--allow-dirty"])
    assert not (tmp_path / "results/summary").exists()


def test_cli_refuses_missing_arm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    (root / "COFF" / "result.json").unlink()
    with pytest.raises(SystemExit, match="COFF.*does not exist"):
        load().main(["--m-overlap", "no", "--allow-dirty"])


def test_cli_refuses_non_r9_pilot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, n_flies=4)
    with pytest.raises(SystemExit, match="R9"):
        load().main(["--m-overlap", "no", "--allow-dirty"])


def test_cli_refuses_differing_sizes_and_eval_schedule_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    (root / "eval_schedule.json").write_text(json.dumps({"digest": "0" * 64}))
    with pytest.raises(SystemExit, match="eval_schedule.json"):
        load().main(["--m-overlap", "no", "--allow-dirty"])
    root = lay_out(tmp_path)
    edit(root, "COFF", lambda d: d.update(learn=40))
    with pytest.raises(SystemExit, match="learn counts"):
        load().main(["--m-overlap", "no", "--allow-dirty"])


def test_cli_refuses_too_few_valid_flies(tmp_path, monkeypatch):
    """RS flies whose donor was INVALID or whose residual is over max are excluded (stats.row_invalid); fewer than
    two valid flies leaves no between-fly variance -> refusal, not a size."""
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)

    def spoil(d):
        for r in d["per_fly"][:3]:
            r["donor_invalid"] = True
        for r in d["per_fly"][3:5]:
            r["residual_frac"] = 0.2
    edit(root, "RS", spoil)
    with pytest.raises(SystemExit, match="RS: 1 valid flies"):
        load().main(["--m-overlap", "no", "--allow-dirty"])


def test_cli_invalid_flies_excluded_from_components(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    edit(root, "RS", lambda d: d["per_fly"][0].update(donor_invalid=True))
    load().main(["--m-overlap", "no", "--allow-dirty"])
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    assert s["components"]["RS"]["n_flies"] == 5 and s["pilot"]["RS"]["invalid_flies"] == [0]


def test_cli_rs_coff_wall_clock_optional_but_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    edit(root, "RS", _wall_none)
    load().main(["--m-overlap", "no", "--allow-dirty"])
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    assert s["rate_record"]["per_arm"]["RS"]["sec_per_batch_battle"] is None
    assert s["rate_record"]["sources"]["sec_per_batch_battle_brain"] == "FLY"


def test_cli_missing_logs_recorded_not_sized_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, logs=False)
    load().main(["--m-overlap", "no", "--allow-dirty"])
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    assert s["fly_decisions"]["per_arm"]["FLY"]["missing_logs"] and s["fly_decisions"]["projected_at_choice"] is None


# ---- session log (fix round 1: an aborted session is counted, a session without an end refuses) ---------
@pytest.mark.parametrize("arm", ["FLY", "RND"])
def test_cli_refuses_session_without_end(tmp_path, monkeypatch, arm):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    d = sessions(WALL[arm] / 2, WALL[arm] / 2)
    d["sessions"].append(dict(started_utc="2026-09-28T02:00:00+00:00", ended_utc=None, seconds=None,
                              status="running", played=None, complete=None))
    write_sessions(root, arm, d)
    with pytest.raises(SystemExit, match="no end record"):
        load().main(["--m-overlap", "no", "--allow-dirty"])
    assert not (tmp_path / "results/summary").exists()


@pytest.mark.parametrize("arm", ["FLY", "RND"])
def test_cli_refuses_missing_session_log(tmp_path, monkeypatch, arm):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    (root / arm / "wall_clock.json").unlink()
    with pytest.raises(SystemExit, match="session log"):
        load().main(["--m-overlap", "no", "--allow-dirty"])


def test_cli_refuses_sessions_not_summing_to_wall(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    write_sessions(root, "FLY", sessions(1000.0))
    with pytest.raises(SystemExit, match="sum to"):
        load().main(["--m-overlap", "no", "--allow-dirty"])


def test_cli_aborted_session_is_counted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    d = sessions(1000.0, status="aborted")
    d["sessions"] += sessions(2600.0)["sessions"]
    write_sessions(root, "FLY", d)                              # 1000 + 2600 == FLY wall_clock_s 3600
    load().main(["--m-overlap", "no", "--allow-dirty"])
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    r = s["rate_record"]["per_arm"]["FLY"]
    assert r["sessions"] == 2 and r["aborted_sessions"] == 1 and r["sec_per_batch_battle"] == pytest.approx(3600 / 60)


def test_cli_judge_workers_recorded_and_scale_hours(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    load().main(["--m-overlap", "no", "--allow-dirty", "--workers", "4"])
    s = json.loads((tmp_path / "results/summary/rescope_power.json").read_text())
    rates = s["rates"]
    assert s["workers"] == 4
    assert all(r["hours"] == pytest.approx(power.wall_hours(r["F"], r["E"], rates, SPEC, 4)) for r in s["table"])


# ---- run_rescope_battles session bookkeeping ------------------------------------------------------------
def load_battles():
    spec = importlib.util.spec_from_file_location("rrb", ROOT / "scripts/run_rescope_battles.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_battles_sessions_count_aborted_and_leave_killed_open(tmp_path, monkeypatch):
    import time
    from flymon.brain.config import Params
    monkeypatch.chdir(tmp_path)
    mod = load_battles()
    out = Path("results/rescope/pilot/FLY")
    t0 = time.time() - 10.0
    i = mod.open_session(out, t0, [Params()])
    mod.close_session(out, i, t0, "aborted", None, [Params()])          # an exception in session 0
    k = mod.open_session(out, time.time(), [Params()])                     # session 1 killed hard: never closed
    t2 = time.time() - 5.0
    j = mod.open_session(out, t2, [Params()])
    run = {"played": {"L": ["a", "b"], "E": ["c"]}, "complete": True}
    total = mod.close_session(out, j, t2, "ok", run, [Params()])
    ss = json.loads((out / "wall_clock.json").read_text())["sessions"]
    assert [x["status"] for x in ss] == ["aborted", "running", "ok"] and (i, k, j) == (0, 1, 2)
    assert ss[1]["ended_utc"] is None and ss[2]["played"] == {"L": 2, "E": 1} and ss[2]["complete"] is True
    assert total == pytest.approx(ss[0]["seconds"] + ss[2]["seconds"]) and total >= 15.0
    assert load().session_reasons("FLY", out / "wall_clock.json", total)          # the killed session refuses


# ---- the FLY <-> RS donor trace (spec 10.8; final-review finding 2) -----------------------------------------
@pytest.mark.parametrize("spoil", ["rs_sha", "fly_sha", "log_changed", "learn_digest"])
def test_cli_refuses_donor_mismatch(tmp_path, monkeypatch, spoil):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    if spoil == "rs_sha":
        edit(root, "RS", lambda d: d["per_fly"][2].update(donor_sha256="0" * 64))
    elif spoil == "fly_sha":
        edit(root, "FLY", lambda d: d["per_fly"][2].update(learn_log_sha256="0" * 64))
    elif spoil == "log_changed":
        (root / "FLY" / "logs" / "fly02.jsonl").write_text("{}\n")
    else:                                                        # RS learned on another schedule than its donor
        edit(root, "RS", lambda d: d["schedule_digests"].update(learn="0" * 64))
    with pytest.raises(SystemExit, match="donor mismatch"):
        load().main(["--m-overlap", "no", "--allow-dirty"])
    assert not (tmp_path / "results/summary/rescope_power.json").exists()
