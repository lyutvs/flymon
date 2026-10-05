"""Y.7 7 / 7a: the smoke (pilot pair j 0, one fly, the smoke seed block, lever / z / RN1 / seed checks, unit costs)
and the worst-case budget gate (×1.3; C first, then the reconfirmed alternatives, at most five reconfirmations in all,
then STOP_BUDGET), plus run_y's phase-B stages."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.y_spec import SPEC as Y
from tests.brain import y_world_b as YW
from tests.brain.y_world_b import World, doc, through

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_smoke_pass_seeds_costs_and_archive(w):
    through(w, "oc")
    out = w.runner().stage_smoke()
    assert out["outcome"] == R.PASS and out["reasons"] == [] and out["pair"] == "b|0|sx0|sy0"
    assert out["seeds"]["probe"]["0"] == list(range(77_100_000, 77_100_008))
    assert out["seeds"]["train"] == [77_110_000, 77_110_020] and out["seeds"]["oracle"]["act"][0] == 77_150_000
    assert out["costs"]["workers"] == w.ys.workers and out["costs_used"]["oracle_round_s"] >= YW.COSTS["oracle_round_s"]
    files = json.loads(Path(Y.smoke_detail).read_text())["manifest"]
    assert len(files) == 3 + 3 + 1 + 1 and all("/smoke/" in m["cache_file"] for m in files)
    assert {a["src"] for a in out["archive"]} == {Y.oracle_detail, Y.smoke_detail} | {m["cache_file"] for m in files}


def test_smoke_invalid_on_a_lever_mismatch_and_rn_sharing(w):
    through(w, "oc")
    w.oracle_sha = "other"
    out = w.runner().stage_smoke()
    assert out["outcome"] == R.INVALID and any("oracle edit" in r for r in out["reasons"]) and "archive" not in out


def test_smoke_problem_list(w, monkeypatch):
    through(w, "oc")
    w.model.share_state = True                         # RN continues on R's trained brain
    out = w.runner().stage_smoke()
    assert out["outcome"] == R.INVALID and out["reasons"]


def _costs(w, monkeypatch, total):
    """Make design_cost_y return `total` h per option (C adds 1 h) without touching W's model."""
    def fake(costs, d, ys, w_spec, n_naive, k, with_c):
        t = total(d) + (1.0 if with_c else 0.0)
        return dict(parts_h=dict(oracle=0.0, naive=0.0, learn=t, band=0.0, c=1.0 if with_c else 0.0, noplast=0.0),
                    total_h=t)
    monkeypatch.setattr(YR.y_rules, "design_cost_y", fake)


def test_budget_gate_keeps_c_when_it_fits(w):
    through(w, "smoke")
    out = w.runner().stage_budget_gate()
    assert out["outcome"] == R.PASS and out["plan"]["with_c"] is True and out["n_reconfirmed"] == 1
    assert out["plan"]["design"] == doc()["oc"]["design"] and out["margin"] == 1.3


def test_budget_gate_drops_c_then_reconfirmed_alternative_then_stops(w, monkeypatch):
    through(w, "smoke")
    d = doc()
    base = d["oc"]["design"]
    # elapsed 1 h; budget 24 h: with margin 1.3 a total ≤ 17.69 h fits
    _costs(w, monkeypatch, lambda dsg: 17.0)                 # with C 18 h → over; without C 17 h → fits
    out = w.runner().stage_budget_gate()
    assert out["outcome"] == R.PASS and out["plan"]["with_c"] is False and out["options"][0]["in_budget"] is False
    assert out["plan"]["design"] == base
    Path(Y.summary).write_text(json.dumps(d))                # the block is rewritten only in this test
    YW.fake_reconfirm.fail = {1}
    _costs(w, monkeypatch, lambda dsg: 30.0 if dsg.get("rank", 0) in (0, 1) else 10.0)
    out = w.runner().stage_budget_gate()
    assert out["outcome"] == R.PASS and out["plan"]["design"]["rank"] == 2 and out["n_reconfirmed"] == 2
    assert [r["ok"] for r in out["reconfirm"]] == [True]                          # rank 1 skipped: over budget
    Path(Y.summary).write_text(json.dumps(d))
    _costs(w, monkeypatch, lambda dsg: 30.0)
    out = w.runner().stage_budget_gate()
    assert out["outcome"] == R.STOP_BUDGET and out["sentence"].endswith("(순서 7a).") and out["reconfirm"] == []


def test_budget_gate_reconfirmation_limit(w, monkeypatch):
    through(w, "smoke")
    YW.fake_reconfirm.fail = set(range(1, 30))
    _costs(w, monkeypatch, lambda dsg: 30.0 if dsg.get("rank", 0) == 0 else 10.0)
    out = w.runner().stage_budget_gate()
    assert out["outcome"] == R.STOP_BUDGET and out["n_reconfirmed"] == Y.reconfirm_max
    assert len(out["reconfirm"]) == Y.reconfirm_max - 1
    assert any(o.get("skipped") for o in out["options"])


def test_run_y_phase_b_stages():
    spec = importlib.util.spec_from_file_location("run_y", ROOT / "scripts/run_y.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.STAGES[6:] == YR.PHASE_B and set(m.POOL_STAGES) == {"oracle", "pilot", "smoke", "gates", "learn",
                                                                  "band", "records"}
    assert {"screened", "manifest", "ranking", "reconfirm"} <= set(m.QUIET)
    assert m.exit_code(dict(outcome="NOT_SEALED")) == 3 and m.exit_code(dict(outcome="FAIL")) == 3
