"""M0b budget extrapolation (spec C.6), throughput rows, exact-reproduction checks against the M0 result files, and the gate."""
from __future__ import annotations

import statistics
import time

from .pool_jobs import phase_timing_job

DECISIONS = 52_000            # spec 4.1: 42,000 training decisions + 10,000 exploration / secondary
EVAL_DECISIONS = 81_600       # spec 4.2: (42 flies x 4 evals + 12 REV x 1 + 6 WEAK-FLY x 4) x 20 battles x 20 fly decisions
WINDOWS = {"n_candidates": 4, "settle_decision_ms": 800.0, "read_ms": 600.0,
           "settle_reinforce_ms": 800.0, "pulse_max_ms": 600.0, "gap_ms": 200.0}


# ---- parent-side ---------------------------------------------------------------------------------
def throughput_row(pool, odor, strength: float, steps: int, warm: int, repeats: int = 3, idx=None,
                   settle_decision_ms: float = 800.0, read_ms: float = 600.0, settle_reinforce_ms: float = 800.0,
                   pulse_ms: float = 600.0, gap_ms: float = 200.0, reward_type: str = "PAM08") -> dict:
    """Two measurements per repeat. (1) In-worker step times of the two phases, all workers at once (the
    slowest worker bounds a batch, so the max over workers is kept). (2) End to end from the parent: one
    `decide_batch` (one fly per worker, four candidates) and one `reinforce_batch` at the real window
    lengths — this includes weight shipping, resets, presentation and result transfer, and is what the
    budget uses. The learning from (2) is undone afterwards."""
    W = pool.n_workers
    F = min(pool.n_flies, W)
    dec, rein, e2e_dec, e2e_rein = [], [], [], []
    for r in range(repeats):
        res = pool.run_jobs(phase_timing_job, [dict(odor=odor, strength=strength, steps=steps, warm=warm, reward_type=reward_type)] * W)
        dec.append(max(x["ms_decision"] for x in res))
        rein.append(max(x["ms_reinforce"] for x in res))
        t0 = time.perf_counter()
        pool.decide_batch([(f, [odor] * 4, 1000 + r) for f in range(F)], strength, settle_decision_ms, read_ms, idx)
        e2e_dec.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pool.reinforce_batch([(f, odor, reward_type, pulse_ms, 2000 + r) for f in range(F)], strength, settle_reinforce_ms, gap_ms)
        e2e_rein.append(time.perf_counter() - t0)
        for f in range(F):
            pool.w[f] = pool.w0[pool.flies[f].shuffle_seed].copy()
    return {"workers": W, "n_flies_batch": F, "repeats": repeats, "steps": steps, "warm": warm,
            "ms_decision_median": statistics.median(dec), "ms_decision_max": max(dec),
            "ms_reinforce_median": statistics.median(rein), "ms_reinforce_max": max(rein),
            "agg_slot_steps_per_ms": W / statistics.median(dec),
            "s_decide_batch_median": statistics.median(e2e_dec), "s_decide_batch_max": max(e2e_dec),
            "s_reinforce_batch_median": statistics.median(e2e_rein), "s_reinforce_batch_max": max(e2e_rein),
            "windows": {"settle_decision_ms": settle_decision_ms, "read_ms": read_ms, "settle_reinforce_ms": settle_reinforce_ms,
                        "pulse_ms": pulse_ms, "gap_ms": gap_ms}}


def budget_hours(ms_decision: float, ms_reinforce: float, workers: int, decisions: int = DECISIONS,
                 eval_decisions: int = EVAL_DECISIONS, n_candidates: int = 4, settle_decision_ms: float = 800.0,
                 read_ms: float = 600.0, settle_reinforce_ms: float = 800.0, pulse_max_ms: float = 600.0,
                 gap_ms: float = 200.0, dt_ms: float = 1.0) -> float:
    """Step-based estimate (diagnostic): training decisions cost n_candidates decision runs + one
    reinforcement each; evaluation decisions (spec 4.2, plasticity off) cost the decision runs only."""
    dec_steps = n_candidates * (settle_decision_ms + read_ms) / dt_ms
    rein_steps = (settle_reinforce_ms + pulse_max_ms + gap_ms) / dt_ms
    train_ms = decisions * (dec_steps * ms_decision + rein_steps * ms_reinforce)
    eval_ms = eval_decisions * dec_steps * ms_decision
    return (train_ms + eval_ms) / workers / 3.6e6


def budget_hours_e2e(s_decide_batch: float, s_reinforce_batch: float, n_flies_batch: int, decisions: int = DECISIONS,
                     eval_decisions: int = EVAL_DECISIONS) -> float:
    """Gate estimate from measured batch wall-clock: a batch serves n_flies_batch decisions at once."""
    return (decisions * (s_decide_batch + s_reinforce_batch) + eval_decisions * s_decide_batch) / n_flies_batch / 3600.0


def budget_table(rows: list, decisions: int = DECISIONS, eval_decisions: int = EVAL_DECISIONS,
                 limit_hours: float = 60.0) -> dict:
    """Hours for every worker count measured (end-to-end max over repeats, and the step-based estimate);
    the gate reads the best end-to-end configuration, not simply the largest one."""
    by_workers = []
    for r in rows:
        by_workers.append({"workers": r["workers"], "n_flies_batch": r["n_flies_batch"],
                           "hours_e2e": budget_hours_e2e(r["s_decide_batch_max"], r["s_reinforce_batch_max"], r["n_flies_batch"], decisions, eval_decisions),
                           "hours_e2e_training_only": budget_hours_e2e(r["s_decide_batch_max"], r["s_reinforce_batch_max"], r["n_flies_batch"], decisions, 0),
                           "hours_step_estimate": budget_hours(r["ms_decision_max"], r["ms_reinforce_max"], r["workers"], decisions, eval_decisions)})
    best = min(by_workers, key=lambda b: b["hours_e2e"])
    table = [
        {"assumption": f"baseline: end-to-end batches, {best['workers']} workers, training + evaluation", "hours": best["hours_e2e"]},
        {"assumption": "training decisions only", "hours": best["hours_e2e_training_only"]},
        {"assumption": "step-based estimate (diagnostic, excludes pool overhead)", "hours": best["hours_step_estimate"]},
    ]
    return {"workers": best["workers"], "n_flies_batch": best["n_flies_batch"], "decisions": decisions, "eval_decisions": eval_decisions,
            "windows": WINDOWS, "by_workers": by_workers, "rows": table, "limit_hours": limit_hours,
            "baseline_hours": table[0]["hours"], "gate_ok": table[0]["hours"] <= limit_hours}


FLOAT_KEYS = ("D_pre", "D_post", "dD", "D_pre_disc", "D_post_disc", "dD_disc", "weights_frac", "w_frac_a_core", "w_frac_p_core")


def exact_match(pool_per_seed: dict, m0_per_seed: dict) -> dict:
    """Bit-for-bit comparison of the pool's conditioning results with results/m0/conditioning.json:
    every seed and arm, the four raw probe count pairs and the float indices."""
    n, n_equal, worst = 0, 0, 0.0
    missing = []
    for seed, arms in m0_per_seed.items():
        for arm, ref in arms.items():
            n += 1
            got = pool_per_seed.get(str(seed), pool_per_seed.get(int(seed) if str(seed).isdigit() else seed, {})).get(arm)
            if got is None:
                missing.append(f"{seed}/{arm}")
                continue
            same = got["counts"] == ref["counts"] and all(got[k] == ref[k] for k in FLOAT_KEYS)
            worst = max(worst, max(abs(float(got[k]) - float(ref[k])) for k in FLOAT_KEYS))
            n_equal += int(same)
    return {"n_results": n, "n_equal": n_equal, "n_missing": len(missing), "missing": missing,
            "max_abs_diff": worst, "ok": n > 0 and n_equal == n}


def m0b_gate(budget: dict, conditioning_match: dict, sparsity_match: dict, decide_equal: bool) -> dict:
    return {"throughput_ok": bool(budget["gate_ok"]), "conditioning_exact_ok": bool(conditioning_match["ok"]),
            "sparsity_exact_ok": bool(sparsity_match["ok"]), "decide_equal_ok": bool(decide_equal),
            "passed": bool(budget["gate_ok"] and conditioning_match["ok"] and sparsity_match["ok"] and decide_equal)}
