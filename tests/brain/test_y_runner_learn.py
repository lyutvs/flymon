"""Y.7 9 · 10: learning (gate pairs × F × K × R · N · RN on Y's main-set seeds), the BAND 2K re-measure of every gate
pair (same training seeds, probes K … 2K − 1), the records (C if the plan keeps it, the plasticity-off control) —
manifests only, the main set used from learn on, the batch ledger's STOP_BUDGET (measured spend, no margin) and the
measurement-key refusal."""
import json
from pathlib import Path

import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.v_spec import SPEC as V
from flymon.brain.y_spec import SPEC as Y
from tests.brain.y_world_b import World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _inputs(m):
    return json.loads(Path(m["cache_file"]).read_text())["inputs"]


def test_learn_band_records_units_and_seeds(w):
    through(w, "estimate")
    learn = w.runner().stage_learn()
    assert learn["outcome"] == R.PASS and learn["n_units"] == 8 * 8 * 3 and learn["main_set_used"] is True
    assert "statistic" not in json.dumps(learn) and set(learn) >= {"manifest", "detail_sha256"}
    g0 = doc()["gates"]["gates"][0]
    ins = {(_inputs(m)["brain"], _inputs(m)["fly"]): _inputs(m) for m in learn["manifest"]
           if m["key"].startswith(g0["key"] + "|")}
    c = g0["c"]
    assert ins[("R", 2)]["probe_seeds"] == [62_000_000 + c * 4_000 + 200 + k for k in range(8)]
    assert ins[("R", 2)]["phases"] == [["PAM08", 20, 64_000_000 + c * 40_000], ["PPL105", 20, 64_000_000 + c * 40_000 + 20]]
    assert ins[("RN", 0)]["phases"][1][0] is None and ins[("N", 0)]["phases"][0][0] is None
    band = w.runner().stage_band()
    assert band["outcome"] == R.PASS and band["n_units"] == 8 * 8 * 3
    b = [_inputs(m) for m in band["manifest"] if m["key"] == f"{g0['key']}|2|R"][0]
    assert b["probe_seeds"] == [62_000_000 + c * 4_000 + 200 + k for k in range(8, 16)]
    assert b["phases"] == ins[("R", 2)]["phases"]
    rec = w.runner().stage_records()
    assert rec["outcome"] == R.PASS and rec["n_units"] == 8 * 8 * 3 + 2 * 2
    edits = {_inputs(m)["edit"] for m in rec["manifest"]}
    assert edits == {V.no_edit, V.lever_edit} and sum(not _inputs(m)["plastic"] for m in rec["manifest"]) == 4
    led = [e["stage"] for e in doc()["budget"]["ledger"]]
    assert led[-3:] == ["learn", "band", "records"]


def test_records_without_c(w):
    through(w, "gates")
    d = doc()
    w.runner().stage_estimate()
    d = doc()
    d["estimate"]["plan"]["with_c"] = False
    Path(Y.summary).write_text(json.dumps(d))
    through_rest = [w.runner().stage_learn(), w.runner().stage_band()]
    assert all(o["outcome"] == R.PASS for o in through_rest)
    assert w.runner().stage_records()["n_units"] == 4


def test_in_stage_budget_stop_uses_the_measured_ledger(w, monkeypatch):
    through(w, "estimate")
    d = doc()
    d["estimate"]["plan"]["parts_h"]["learn"] = 30.0                 # the remaining share alone exceeds 24 h
    Path(Y.summary).write_text(json.dumps(d))
    out = w.runner().stage_learn()
    assert out["outcome"] == R.STOP_BUDGET and out["main_set_used"] is True
    assert out["sentence"].startswith("남은 추정 비용 누적 ") and out["sentence"].endswith("(순서 9 학습 측정).")
    with pytest.raises(SystemExit):
        w.runner().stage_band()


def test_learn_refuses_on_a_changed_key(w):
    through(w, "estimate")
    w.ctx["keys"] = lambda: dict(w_measure_key=Y.w_measure_key, u_measure_key="u" * 64)
    with pytest.raises(SystemExit) as e:
        w.runner().stage_learn()
    assert e.value.code == YR.EXIT_KEY and "learn" not in doc()


def _resume_world(w, share_h):
    """Through estimate, the later parts zeroed and this stage's share set to share_h; then half of learn's units
    measured into the cache (a stage interrupted half-way). Elapsed ≈ 1 h."""
    through(w, "estimate")
    d = doc()
    d["estimate"]["plan"]["parts_h"].update(learn=share_h, band=0.0, c=0.0, noplast=0.0)
    Path(Y.summary).write_text(json.dumps(d))
    r = w.runner()
    wr, wm = w.w_runner(w.pool, False)
    units = wr.learn_units(r._view(doc()))
    wm.learn(units[:len(units) // 2], "learn")
    return len(units)


def test_resumed_stage_counts_only_the_remaining_units(w):
    """Half the units cached: the remaining share is share × 1/2 (what an uninterrupted run checks at the same point),
    not the full share. 1 h + 30 h / 2 = 16 h ≤ 24 → PASS; the old double count (1 h + 30 h) stopped."""
    n = _resume_world(w, 30.0)
    jobs = w.pool.jobs
    out = w.runner().stage_learn()
    assert out["outcome"] == R.PASS and out["n_units"] == n and w.pool.jobs - jobs == n - n // 2


def test_resumed_stage_still_stops_when_the_remaining_share_is_over(w):
    """The other side: 1 h + 50 h / 2 = 26 h > 24 → STOP_BUDGET with 남은 25.00 h (the remaining half, not 0)."""
    _resume_world(w, 50.0)
    out = w.runner().stage_learn()
    assert out["outcome"] == R.STOP_BUDGET and "남은 25.00 h" in out["reasons"][0]


def test_fresh_stage_takes_the_full_share(w):
    through(w, "estimate")
    d = doc()
    d["estimate"]["plan"]["parts_h"].update(learn=30.0, band=0.0, c=0.0, noplast=0.0)
    Path(Y.summary).write_text(json.dumps(d))
    out = w.runner().stage_learn()
    assert out["outcome"] == R.STOP_BUDGET and "남은 30.00 h" in out["reasons"][0]
