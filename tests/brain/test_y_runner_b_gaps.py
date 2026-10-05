"""Phase-B review test gaps (Tasks 4–6), each pinned to a mutant that the earlier tests let through: the smoke checks
(seed block, z_V, an RN1-only fault, BAND probe seeds), the smoke unit costs (per-field max of pilot and smoke, the
gate's costs), 7a's `i > sel rank` filter, the gates' per-batch spend after a kill, the estimate on k_lo ≤ gates <
k_hi, the lenient-set sha and N_len checks alone, a K ≠ F design, and the batch-ledger arithmetic of learn / band."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y
from tests.brain import y_world_b as YW
from tests.brain.y_world_b import World, doc, through

H = 3600.0


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _put(d):
    Path(Y.summary).write_text(json.dumps(d))


# ================================================================ Task 4: smoke checks, unit costs, 7a filter
def test_smoke_seed_outside_the_block_is_named(w, monkeypatch):
    through(w, "oc")
    real = YR.y_w_spec

    def moved(ys, smoke=False):
        s = real(ys, smoke)
        return dataclasses.replace(s, smoke_probe_seed0=77_200_000) if smoke else s
    monkeypatch.setattr(YR, "y_w_spec", moved)
    out = w.runner().stage_smoke()
    assert out["outcome"] == R.INVALID
    assert "스모크 시드 77200000가 스모크 블록 밖" in out["reasons"]


def test_smoke_oracle_z_other_than_z_v(w):
    through(w, "oc")

    def other(pool):
        m = YW.SmokeOracle(w)
        m.z = {k: (v[0] + 1.0, v[1]) for k, v in W.z_v().items()}
        return m
    w.ctx["smoke_oracle"] = other
    out = w.runner().stage_smoke()
    assert out["outcome"] == R.INVALID and any(r.startswith("oracle z ") for r in out["reasons"])


def test_smoke_rn1_only_fault(w):
    """RN's first stage differs from R's and nothing else: the only reason is RN1 ≠ R1."""
    through(w, "oc")
    real = w.model.job

    def job(kw, pair):
        out = real(kw, pair)
        if [p[0] for p in kw["phases"]] == ["PAM08", None]:
            s1 = out["stages"][1]
            s1["x"][0] = dict(s1["x"][0], A=s1["x"][0]["A"] + 7)
        return out
    w.model.job = job
    out = w.runner().stage_smoke()
    assert out["outcome"] == R.INVALID and out["reasons"] == ["RN1 ≠ R1"]


def test_smoke_band_probe_seeds(w):
    through(w, "oc")
    w.runner().stage_smoke()
    man = json.loads(Path(Y.smoke_detail).read_text())["manifest"]
    band = [m for m in man if m["key"].endswith("|smoke_band")]
    assert band and all(json.loads(Path(m["cache_file"]).read_text())["inputs"]["probe_seeds"]
                        == list(range(77_100_008, 77_100_016)) for m in band)


def test_smoke_unit_costs_are_the_per_field_max_and_the_gate_uses_them(w):
    through(w, "oc")
    d = doc()
    d["pilot"]["costs"].update(trial_s=1e-12, presentation_s=1e-12)    # the smoke's measured costs are larger
    _put(d)
    sm = w.runner().stage_smoke()
    pilot = doc()["pilot"]["costs"]
    for f in Y.cost_fields:
        assert sm["costs_used"][f] == max(float(pilot[f]), float(sm["costs"].get(f, 0.0)))
    assert sm["costs_used"]["trial_s"] == sm["costs"]["trial_s"] > pilot["trial_s"]
    assert sm["costs_used"]["workers"] == pilot["workers"]
    gate = w.runner().stage_budget_gate()
    assert gate["costs"] == sm["costs_used"]


def test_budget_gate_tries_only_alternatives_ranked_after_the_selection(w, monkeypatch):
    """oc's rank 0 fails its reconfirmation → selection rank 1; 7a's alternatives are ranks 2, 3, … (never rank 0,
    never the selection again)."""
    YW.fake_reconfirm.fail = {0}
    through(w, "smoke")
    YW.fake_reconfirm.fail = set()
    sel = doc()["oc"]["design"]
    assert int(sel["rank"]) == 1
    n = len(doc()["oc"]["ranking"])

    def fake(costs, d, ys, w_spec, n_naive, k, with_c):
        return dict(parts_h=dict(oracle=0.0, naive=0.0, learn=30.0, band=0.0, c=0.0, noplast=0.0), total_h=30.0)
    monkeypatch.setattr(YR.y_rules, "design_cost_y", fake)
    out = w.runner().stage_budget_gate()
    ranks = [int(o["design"]["rank"]) for o in out["options"]]
    assert ranks == [1, 1] + list(range(2, n)) and n > 2


# ================================================================ Task 5: gates and estimate
def test_gates_keep_each_batchs_spend_after_a_kill(w, monkeypatch):
    through(w, "budget_gate")
    real, calls = w.pool.run_jobs, []
    per_batch = -(-8 // w.pool.n_workers)                  # one pair (F 8 units) per gates batch, n_w units per job

    def dying(fn, kws):                                    # the kill lands in the second gates batch
        calls.append(len(kws))
        if len(calls) > per_batch:
            raise KeyboardInterrupt("killed")
        import time as _t
        _t.sleep(0.05)
        return real(fn, kws)
    monkeypatch.setattr(w.pool, "run_jobs", dying)
    with pytest.raises(KeyboardInterrupt):
        w.runner().stage_gates()
    assert "gates" not in doc() and w.runner()._prog("gates") >= 0.05


def test_estimate_on_a_gate_count_between_k_lo_and_k_hi(w, monkeypatch):
    through(w, "gates")
    d = doc()
    k_lo, k_hi = (int(v) for v in d["budget_gate"]["plan"]["design"]["k_range"])
    k = k_lo
    assert k_lo <= k < k_hi
    d["gates"]["gates"] = d["gates"]["gates"][:k]
    _put(d)
    seen = []
    real = YR.y_rules.design_cost_y

    def spy(costs, dd, ys, w_spec, n_naive, kk, with_c):
        seen.append(kk)
        return real(costs, dd, ys, w_spec, n_naive, kk, with_c)
    monkeypatch.setattr(YR.y_rules, "design_cost_y", spy)
    out = w.runner().stage_estimate()
    assert out["n_gates"] == k and seen and set(seen) == {k}
    full = real(doc()["smoke"]["costs_used"], d["budget_gate"]["plan"]["design"], w.ys, W, 0, k_hi, True)
    assert out["options"][0]["total_h"] < full["total_h"]


def test_gates_refuse_on_the_oracle_sha_alone(w):
    """oracle.json rewritten with the same lenient count: only the sha256 check can refuse."""
    through(w, "budget_gate")
    p = Path(Y.oracle_detail)
    j = json.loads(p.read_text())
    j["pairs"][1]["L_A"] = 31.0                               # c 1: not lenient either way
    p.write_text(json.dumps(j))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_gates()
    assert e.value.code == 2 and "gates" not in doc()


def test_gates_refuse_on_n_len_alone(w):
    """oracle.json untouched (sha matches) but block oracle's N_len differs: only the N_len check can refuse."""
    through(w, "budget_gate")
    d = doc()
    d["oracle"]["derived"]["yield_rule"]["n_len"] += 1
    _put(d)
    with pytest.raises(SystemExit) as e:
        w.runner().stage_gates()
    assert e.value.code == 2 and "gates" not in doc()


def test_gates_with_k_other_than_f(w):
    """A design with F 10 ≠ K 8: each screened pair is F flies × K probe seeds."""
    through(w, "budget_gate")
    d = doc()
    d["budget_gate"]["plan"]["design"]["F"] = 10
    _put(d)
    out = w.runner().stage_gates()
    by = {}
    for m in out["manifest"]:
        ins = json.loads(Path(m["cache_file"]).read_text())["inputs"]
        by.setdefault(m["key"].rsplit("|", 2)[0], []).append(ins)
    assert out["n_screened"] >= 1 and all(len(v) == 10 for v in by.values())
    assert all(len(i["probe_seeds"]) == 8 for v in by.values() for i in v)
    assert sorted(i["fly"] for i in next(iter(by.values()))) == list(range(10))


# ================================================================ Task 6: the batch ledger of learn / band / records
def _ledger(w, parts, with_c=True, extra_s=0.0):
    through(w, "estimate")
    d = doc()
    d["estimate"]["plan"]["parts_h"].update(parts)
    d["estimate"]["plan"]["with_c"] = with_c
    if extra_s:
        d["budget"]["ledger"].append(dict(stage="x", wall_s=extra_s))
    _put(d)


def test_ledger_base_plus_small_shares_over_24_stops(w):
    """Ledger ≈ 21 h (1 h + 20 h) + learn 1.5 + band 1 + C 0.5 + no-plast 0.5 = 24.5 h > 24 → STOP; neither the
    ledger base nor the later parts alone would stop."""
    _ledger(w, dict(learn=1.5, band=1.0, c=0.5, noplast=0.5), extra_s=20 * H)
    out = w.runner().stage_learn()
    assert out["outcome"] == R.STOP_BUDGET and "남은 3.50 h" in out["reasons"][0]


def test_ledger_has_no_margin(w):
    """1 h + 15 + 5 = 21 h ≤ 24 at ×1.0 (×1.3 would be 27 h) → PASS (no margin at orders 9 · 10, Y.9.2 P2-9)."""
    _ledger(w, dict(learn=15.0, band=5.0, c=0.0, noplast=0.0))
    assert w.runner().stage_learn()["outcome"] == R.PASS


def test_ledger_later_band_counts_at_learn(w):
    _ledger(w, dict(learn=0.0, band=30.0, c=0.0, noplast=0.0))
    assert w.runner().stage_learn()["outcome"] == R.STOP_BUDGET


def test_ledger_c_counts_only_when_the_plan_keeps_it(w):
    _ledger(w, dict(learn=0.0, band=0.0, c=30.0, noplast=0.0), with_c=False)
    assert w.runner().stage_learn()["outcome"] == R.PASS


def test_ledger_progress_spend_stops_and_is_booked(w):
    """A large progress/learn.json (an earlier run's spend) → STOP; the block's ledger entry carries that spend."""
    _ledger(w, dict(learn=0.0, band=0.0, c=0.0, noplast=0.0))
    r = w.runner()
    r._prog_add("learn", 30 * H)
    out = w.runner().stage_learn()
    assert out["outcome"] == R.STOP_BUDGET
    last = doc()["budget"]["ledger"][-1]
    assert last["stage"] == "learn" and last["wall_s"] > 0 and last["wall_s"] >= 30 * H
