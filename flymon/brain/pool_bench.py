"""M0b budget extrapolation (spec C.6), throughput rows, exact-reproduction checks against the M0 result files, and the gate."""
from __future__ import annotations

import os
import statistics
import sys
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


OLD_ENGINE_DIRS = ("results/m0/", "results/m0b/")
OLD_ENGINE_FILES = ("results/summary/m0.json", "results/summary/m0b.json", "results/summary/compartments.json")


def refuse_old_engine_output(out: str, kc_kc_scale: float) -> None:
    """The M0/M0b result files are the old engine's (kc_kc_scale 1.0) immutable, bit-exact references
    (spec D.5) and results/m0* is git-ignored, so an overwrite is unrecoverable. Any script that would write
    under those paths with another engine refuses (SystemExit 2); pass an explicit --out under results/m0c/."""
    if kc_kc_scale == 1.0 or not out:
        return
    rel = os.path.relpath(os.path.abspath(str(out)), os.getcwd()).replace(os.sep, "/")
    if rel.startswith(OLD_ENGINE_DIRS) or rel in OLD_ENGINE_FILES:
        print(f"refusing to write {out} with kc_kc_scale={kc_kc_scale}: that path holds the old engine's "
              f"(kc_kc_scale=1.0) immutable reference (spec D.5); use --out results/m0c/... or --kc-kc-scale 1.0",
              file=sys.stderr)
        raise SystemExit(2)


PRE_M0D_DIRS = ("results/m0/", "results/m0b/", "results/m0c/")
PRE_M0D_FILES = ("results/summary/m0.json", "results/summary/m0b.json", "results/summary/m0c.json",
                 "results/summary/compartments.json")


def refuse_modified_engine_output(out: str, params) -> None:
    """Spec H.2: an engine with any M0d mode on (graded APL, ORN depression, homeostatic thresholds, scaled APL input)
    never writes under the M0/M0b/M0c reference trees or summaries (SystemExit 2); its results go under results/m0d/."""
    modified = (params.apl_mode != "spiking" or params.orn_std or params.kc_thresh_mode != "pn_norm"
                or params.apl_input_scale != 1.0)
    if not modified or not out:
        return
    rel = os.path.relpath(os.path.abspath(str(out)), os.getcwd()).replace(os.sep, "/")
    if rel.startswith(PRE_M0D_DIRS) or rel in PRE_M0D_FILES:
        print(f"refusing to write {out} with M0d modes on (apl_mode={params.apl_mode}, orn_std={params.orn_std}, "
              f"kc_thresh_mode={params.kc_thresh_mode}, apl_input_scale={params.apl_input_scale}): that path holds "
              f"a pre-M0d engine's reference; use results/m0d/", file=sys.stderr)
        raise SystemExit(2)


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


SPARSITY_MATCH_KEYS = ("frac_active_A", "frac_active_B", "jaccard", "chance", "mbon_hz_A", "mbon_hz_B")


def match_sparsity_row(sparsity: dict, baseline: dict, reference: dict | None, p) -> dict:
    """Compare the pool's sparsity and baseline means with the reference sparsity grid row for these Params
    (rows without `kc_kc_scale` or `apl_input_scale` are the M0 engine, 1.0). Every term this run did not measure is reported as a
    note and left out of `diffs` instead of raising: `--sparsity-seeds 0` or `--rest-seeds 0` is a legitimate
    run shape, and a comparison crash here would throw away hours of pool work before the results are written."""
    if not reference:
        return {"ok": False, "note": "no reference file"}
    if not sparsity.get("per_seed"):
        return {"ok": False, "note": "no sparsity seeds in this run"}
    rows = [g for g in reference["grid"]
            if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac"), g.get("kc_kc_scale", 1.0),
                g.get("apl_input_scale", 1.0))
            == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac, p.kc_kc_scale, p.apl_input_scale)]
    if not rows:
        return {"ok": False, "note": f"no reference grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} "
                                     f"mbon_hold_frac={p.mbon_hold_frac} kc_kc_scale={p.kc_kc_scale} "
                                     f"apl_input_scale={p.apl_input_scale}"}
    row = rows[0]
    diffs = {k: abs(sparsity[k] - row[k]) for k in SPARSITY_MATCH_KEYS}
    out = {"diffs": diffs, "cpu_row": {k: row[k] for k in SPARSITY_MATCH_KEYS}}
    if baseline.get("mbon_hz_rest_trimmed") is None:
        out["note"] = "no rest seeds in this run: trimmed baseline not compared"
    else:
        diffs["mbon_hz_rest_trimmed"] = abs(baseline["mbon_hz_rest_trimmed"] - row["mbon_hz_rest_trimmed"])
        out["cpu_row"]["mbon_hz_rest_trimmed"] = row["mbon_hz_rest_trimmed"]
    out["max_abs_diff"] = max(diffs.values())
    out["ok"] = out["max_abs_diff"] == 0.0 and "note" not in out
    return out


def m0b_gate(budget: dict, conditioning_match: dict, sparsity_match: dict, decide_equal: bool) -> dict:
    return {"throughput_ok": bool(budget["gate_ok"]), "conditioning_exact_ok": bool(conditioning_match["ok"]),
            "sparsity_exact_ok": bool(sparsity_match["ok"]), "decide_equal_ok": bool(decide_equal),
            "passed": bool(budget["gate_ok"] and conditioning_match["ok"] and sparsity_match["ok"] and decide_equal)}


def m0c_gate(sparsity_row: dict, baseline: dict, runaway: dict, equivalence: dict, budget: dict,
             conditioning: dict, limit_hours: float = 60.0) -> dict:
    """The M0c gate (spec D.4), every term pre-registered. `sparsity_row` is the results/m0c sparsity grid row
    for Params(); `baseline` has "mbon_hz_rest_trimmed" over the 8 rest seeds; `runaway` has the per-seed KC
    counts at rest and under odour B; `equivalence` has the three checks against the old engine and the pool;
    `budget` is budget_table(); `conditioning` is the judged (seeds 8-15) summary. PASS = sparsity and baseline
    and runaway and equivalence and throughput; the conditioning criterion is recorded, not gated on. The runaway
    term includes its pre-registered sample sizes (8 rest seeds, 64 odour seeds): a shorter run cannot pass."""
    sparsity_ok = (0.03 <= sparsity_row["frac_active_A"] <= 0.07 and 0.03 <= sparsity_row["frac_active_B"] <= 0.07
                   and sparsity_row["jaccard"] <= sparsity_row["chance"])
    baseline_ok = 3.0 <= baseline["mbon_hz_rest_trimmed"] <= 4.0
    rest, odor = runaway["rest_n_kc_over_sat_per_seed"], runaway["odor_B_n_kc_over_sat_per_seed"]
    runaway_ok = (len(rest) >= 8 and len(odor) >= 64 and max(rest) == 0 and max(odor) == 0)   # sample sizes are part of the definition
    equivalence_ok = bool(equivalence["old_conditioning"]["ok"] and equivalence["old_sparsity"]["ok"]
                          and equivalence["arm_equal"]["ok"] and equivalence["decide_equal"])
    throughput_ok = bool(budget["gate_ok"]) and budget["limit_hours"] == limit_hours
    both, rev = conditioning["arms"]["both"]["mean_dD"], conditioning["arms"]["reversed"]["mean_dD"]
    flip_ok = (conditioning["n_flip"] == conditioning["n_seeds"] and abs(both) >= 0.3 and abs(rev) >= 0.3
               and (both > 0) != (rev > 0))
    return {"sparsity_ok": bool(sparsity_ok), "baseline_ok": bool(baseline_ok), "runaway_ok": bool(runaway_ok),
            "equivalence_ok": equivalence_ok, "throughput_ok": throughput_ok,
            "conditioning_index_flip_ok": bool(flip_ok), "channel_specific_seeds": int(conditioning["channel_specific_seeds"]),
            "passed": bool(sparsity_ok and baseline_ok and runaway_ok and equivalence_ok and throughput_ok)}
