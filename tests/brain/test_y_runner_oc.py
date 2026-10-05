"""Y.7 6 (stage oc): the stage file sets, the resumable process-pool cells, the bootstrap → qualification →
selection (largest p_set, then cost; ×1.3 budget) → P2-8 reconfirmation (next design on a failure, at most five) →
records regardless of the outcome, and the STOPs (boot / reconfirm STOP_OC_UNREACHABLE, STOP_BUDGET, STOP_REUSE) and
refusals (precheck not reproducing, a different θ̂)."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import y_oc as O
from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain import y_store as S
from flymon.brain.config import Params
from flymon.brain.y_spec import SPEC as Y
from tests.brain import y_world_b as YW
from tests.brain.y_world_b import World, doc

REAL_BOOT = O.boot_draw
Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_store_phase_b_file_sets():
    assert S.ARCHIVE_ORDER[:6] == ("digest", "oracle", "stage0", "reuse", "pilot", "precheck")
    assert S.ARCHIVE_ORDER[6:] == YR.PHASE_B
    assert S.stage_files("precheck", Y) == [Y.oracle_detail, Y.precheck_detail]          # phase A unchanged
    assert S.stage_files("oc", Y) == [Y.oracle_detail, Y.oc_detail]
    assert S.stage_files("budget_gate", Y) == S.stage_files("estimate", Y) == [Y.oracle_detail]
    assert S.stage_files("judge", Y) == [Y.oracle_detail, Y.judge_detail]


def test_order_and_spec():
    assert YR.ORDER == ("digest", "oracle", "stage0", "reuse", "pilot", "precheck") + YR.PHASE_B
    yw = YR.y_w_spec(Y)
    assert yw.probe_seeds(3, 2, 2) == [62_000_000 + 3 * 4_000 + 200, 62_000_000 + 3 * 4_000 + 201]
    assert yw.train_base(3, 1) == 64_000_000 + 3 * 40_000 + 20 and yw.k_cap == 10
    sm = YR.y_w_spec(Y, True)
    assert sm.smoke_probe_seeds(0, 2) == [77_100_000, 77_100_001] and sm.smoke_train_base(0) == 77_110_000
    assert sm.oracle_seeds()["act"][0] == 77_150_000 and sm.oracle_seeds()["report"][-1] == 77_150_207


def test_pcells_pool_equals_inline_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    keys = list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs)[:3]
    r = O.r_v0(w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(4), n_pair=16, **kw)))
    th = O.fit_y(w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw), keys, r)
    ys = dataclasses.replace(Y, boot_reps=2, oc_chunk=2, workers=2)
    run = YR.Runner(dict(params=lambda: Params()), ys)
    jobs = [(f"boot_{bi}", dict(bi=bi), REAL_BOOT, (th, Z, ys, r, bi)) for bi in range(3)]
    par = run._pcells("p", jobs)
    run.oc_workers = 1
    inl = run._pcells("i", jobs)
    assert all(np.array_equal(a["arrays"]["hits"], b["arrays"]["hits"]) and a["meta"] == b["meta"]
               for a, b in zip(par, inl))
    calls = []

    def counted(*a):
        calls.append(a[-1])
        return REAL_BOOT(*a)
    Path(ys.progress_dir, "i", "boot_1.npz").unlink()
    again = run._pcells("i", [(t, k, counted, a) for t, k, _f, a in jobs])
    assert calls == [1] and np.array_equal(again[1]["arrays"]["hits"], inl[1]["arrays"]["hits"])
    assert json.loads(Path(ys.progress_dir, "i.json").read_text())["wall_s"] > 0


def test_oc_pass_selects_largest_p_set_then_reconfirms(w):
    out = w.runner().stage_oc()
    assert out["outcome"] == R.PASS and out == doc()["oc"]
    d = out["design"]
    assert (d["p_set"], d["q"], d["K"], d["F"], d["k_range"], d["rank"]) == (1.0, 0.75, 8, 8, [4, 8], 0)
    assert out["n_reconfirmed"] == 1 and out["reconfirm"][0]["ok"] and out["records"]["variants"]["Y"] is not None
    assert out["design_first"]["k"] == [4, 5, 6, 7, 8] and out["design_first"]["power_sim"] == 1.0
    assert out["records_target"]["p_set"] == 1.0 and out["n_len"] == w.n_len and out["env"]["python"]
    assert [a["src"] for a in out["archive"]] == [Y.oracle_detail, Y.oc_detail]
    assert out["ranking"][0]["cost_h"] <= out["ranking"][1]["cost_h"] or out["ranking"][1]["p_set"] < 1.0
    assert doc()["budget"]["ledger"][-1]["stage"] == "oc"


def test_oc_reconfirmation_takes_the_next_design_and_stops_after_five(w):
    YW.fake_reconfirm.fail = {0, 1}
    out = w.runner().stage_oc()
    assert out["outcome"] == R.PASS and out["design"]["rank"] == 2 and out["n_reconfirmed"] == 3
    assert [r["ok"] for r in out["reconfirm"]] == [False, False, True]


def test_oc_reconfirmation_stop(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    YW.fake_reconfirm.fail = set(range(10))
    out = w.runner().stage_oc()
    assert out["outcome"] == R.STOP_OC_UNREACHABLE and out["stop_kind"] == "reconfirm"
    assert out["n_reconfirmed"] == Y.reconfirm_max and "대체 설계 4개" in out["sentence"]
    assert out["design"] is None and out["records"]["target"]["p_set"] == 1.0      # records regardless
    with pytest.raises(SystemExit):
        w.runner()._require("smoke")                    # a STOP at order 6 ends Y's chain


def test_oc_boot_unreachable_keeps_records(w, monkeypatch):
    monkeypatch.setattr(YR.y_oc, "boot_draw", YW.fake_boot_fail)
    out = w.runner().stage_oc()
    assert out["outcome"] == R.STOP_OC_UNREACHABLE and out["stop_kind"] == "boot" and out["n_reconfirmed"] == 0
    assert out["sentence"].startswith("Y 파일럿 잡음에서 F ≤ 32로") and "보정 상태 개수" in out["sentence"]
    assert out["records_target"]["rule"] == "false_ok_max_power" and out["records"] is not None


def test_oc_stop_budget_at_selection(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, elapsed_s=24 * 3600.0)
    out = w.runner().stage_oc()
    assert out["outcome"] == R.STOP_BUDGET and out["sentence"].endswith("(순서 6 선택).")
    assert out["n_qualifying"] > 0 and out["n_in_budget"] == 0 and out["n_reconfirmed"] == 0


def test_oc_refusals_and_reuse_stop(w):
    d = doc()
    d["precheck"]["calibration"]["min"]["corners"][0]["a"] += 1.0
    Path(Y.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oc()
    assert e.value.code == 2
    d["precheck"]["calibration"]["min"]["corners"][0]["a"] -= 1.0
    d["pilot"]["theta"]["n_pairs"] = 99
    Path(Y.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        w.runner().stage_oc()
    d["pilot"]["theta"]["n_pairs"] = w_oc.summary(w.theta)["n_pairs"]
    Path(Y.summary).write_text(json.dumps(d))
    w.why = ["W 측정 키 x"]
    out = w.runner().stage_oc()
    assert out["outcome"] == R.STOP_REUSE and json.loads(Path(Y.oc_detail).read_text())["outcome"] == R.STOP_REUSE


def test_oc_resumes_the_bootstrap_cells(w, monkeypatch):
    seen, killed = [], []

    def flaky(theta, z, ys, r, bi, n_rep=None):
        seen.append(bi)
        if bi == 2 and not killed:
            killed.append(bi)
            raise RuntimeError("killed")
        return YW.fake_boot(theta, z, ys, r, bi, n_rep)
    monkeypatch.setattr(YR.y_oc, "boot_draw", flaky)
    with pytest.raises(RuntimeError):
        w.runner().stage_oc()
    assert "oc" not in doc()
    seen.clear()
    assert w.runner().stage_oc()["outcome"] == R.PASS and seen == [2, 3]
