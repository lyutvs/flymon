"""Y.7 8 / 8a: the judged-seed naive final filter on the oracle lenient passes in declared order (one shared filter
function, stop at k_hi, STOP_FEW_PAIRS below the design's k_lo without changing the design, the main set still
unused) and the estimate on the real gate count (×1.3; C dropped first, then STOP_BUDGET; no design change)."""
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.r_pairs import row_key
from flymon.brain.y_spec import SPEC as Y
from tests.brain.w_world import MAIN
from tests.brain.y_world_b import World, doc, lenient, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_gates_take_lenient_pairs_in_declared_order_and_stop_at_k_hi(w):
    through(w, "budget_gate")
    out = w.runner().stage_gates()
    want = [row_key(r) for r in MAIN if lenient(r["c"])][:8]
    assert out["outcome"] == R.PASS and [g["key"] for g in out["gates"]] == want and out["n_screened"] == 8
    assert out["n_pre"] == w.n_len and out["stopped_at"] == 8
    seeds = {m["cache_file"] for m in out["manifest"]}
    assert len(seeds) == 8 * 8 and all(m["key"].endswith("|naive") for m in out["manifest"])
    raw = json.loads(Path(out["manifest"][0]["cache_file"]).read_text())["inputs"]
    c0 = out["gates"][0]["c"]
    assert raw["probe_seeds"] == [62_000_000 + c0 * 4_000 + k for k in range(8)] and raw["block"] == "naive"
    pre = np.stack([np.asarray([[[x["A"], y["A"]], [x["P"], y["P"]]] for x, y in zip(
        json.loads(Path(m["cache_file"]).read_text())["result"]["stages"][0]["x"],
        json.loads(Path(m["cache_file"]).read_text())["result"]["stages"][0]["y"])])
        for m in sorted(out["manifest"][:8], key=lambda m: int(m["key"].split("|")[-2]))])
    f = R.final_filter(pre, w.z, Y)
    assert out["screened"][0]["naive_d"] == pytest.approx(f["d"]) and out["screened"][0]["L_A"] == f["L_A"]


def test_gates_skip_unbalanced_and_low_pairs(w):
    through(w, "budget_gate")
    keys = [row_key(r) for r in MAIN if lenient(r["c"])]
    w.model.offset[keys[0]] = 40.0                       # MBON13(X) far above Y: unbalanced
    w.model.offset[keys[1]] = -15.0                      # MBON13(X) ≈ 15 < 20 (and unbalanced)
    out = w.runner().stage_gates()
    assert [s["gate"] for s in out["screened"][:2]] == [False, False] and len(out["gates"]) == 8
    assert out["n_screened"] == 10 and out["gates"][0]["key"] == keys[2]


def test_gates_stop_few_pairs_final_keeps_the_design(w):
    through(w, "budget_gate")
    for r in MAIN:
        w.model.offset[row_key(r)] = 40.0
    out = w.runner().stage_gates()
    assert out["outcome"] == R.STOP_FEW_PAIRS and out["stop_kind"] == "final" and out["gates"] == []
    assert out["n_screened"] == w.n_len and "설계의 최소 4에" in out["sentence"] and out["main_set"].startswith("미사용")
    assert out["design"] == doc()["budget_gate"]["plan"]["design"]
    with pytest.raises(SystemExit):
        w.runner().stage_estimate()


def test_gates_refuse_on_a_changed_oracle_detail(w):
    through(w, "budget_gate")
    p = Path(Y.oracle_detail)
    p.write_text(p.read_text().replace('"d_pre": 3.0', '"d_pre": 0.2', 1))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_gates()
    assert e.value.code == 2 and "gates" not in doc()


def test_gates_refuse_on_a_changed_measurement_key(w):
    through(w, "budget_gate")
    w.ctx["keys"] = lambda: dict(w_measure_key="x" * 64, u_measure_key=Y.u_measure_key)
    with pytest.raises(SystemExit) as e:
        w.runner().stage_gates()
    assert e.value.code == YR.EXIT_KEY


def test_estimate_uses_the_real_gate_count_then_drops_c_then_stops(w, monkeypatch):
    through(w, "gates")
    seen = []
    real = YR.y_rules.design_cost_y

    def spy(costs, d, ys, w_spec, n_naive, k, with_c):
        seen.append((n_naive, k, with_c))
        return real(costs, d, ys, w_spec, n_naive, k, with_c)
    monkeypatch.setattr(YR.y_rules, "design_cost_y", spy)
    out = w.runner().stage_estimate()
    assert out["outcome"] == R.PASS and out["plan"]["with_c"] is True and seen[0] == (0, 8, True)
    d = doc()
    d.pop("estimate")
    now = sum(e["wall_s"] for e in d["budget"]["ledger"])
    d["budget"]["ledger"].append(dict(stage="x", wall_s=24 * 3600.0 - now - 1.3 * out["options"][0]["total_h"] * 3600
                                      + 1))
    Path(Y.summary).write_text(json.dumps(d))
    out = w.runner().stage_estimate()
    assert out["outcome"] == R.PASS and out["plan"]["with_c"] is False
    d["budget"]["ledger"].append(dict(stage="x", wall_s=3600.0))
    Path(Y.summary).write_text(json.dumps(d))
    out = w.runner().stage_estimate()
    assert out["outcome"] == R.STOP_BUDGET and out["sentence"].endswith("(순서 8a).")
