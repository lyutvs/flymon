#!/usr/bin/env python3
"""Write results/summary/m0c.json: the M0c gate (spec appendix D.4) from the M0c result files.

Inputs (all written by the controller; none is overwritten here):
  results/m0c/sparsity.json           reproduce_flybrain_measurements.py sparsity --rest-seeds 8  (default Params row)
  results/m0c/reproduce_old.json      bench_pool.py reproduce --kc-kc-scale 1.0  vs results/m0  (old-engine equivalence)
  results/m0c/reproduce.json          bench_pool.py reproduce --seed-start 8 --rest-seeds 8 --odor-runaway-seeds 64 --arm-equal
  results/m0c/reproduce_seeds0-7.json bench_pool.py reproduce --seed-start 0 (reported alongside, not judged)
  results/m0c/throughput.json         bench_pool.py throughput
PASS = sparsity and baseline and runaway and equivalence and throughput. The conditioning criterion (pre-registered,
unchanged) is recorded as it comes out; it does not gate M0c. A gate that fails is written as failed, never refused —
but inputs of the wrong engine, the wrong seeds or a short run are refused with the shortfalls named (validate_inputs).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from flymon.brain.config import Params
from flymon.brain.pool_bench import budget_table, m0c_gate, refuse_old_engine_output


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing input {p}: run the M0c commands in README first")
    return json.loads(p.read_text())


def _git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"git_commit": head, "git_dirty": dirty}
    except Exception:   # not a git checkout: still write the summary
        return {"git_commit": "", "git_dirty": None}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(p: Params, sp: dict, old: dict, new: dict, reported: dict | None, th: dict) -> None:
    """The pre-registered identity and sample sizes (spec D.4): refuse to compose a gate from the wrong engine,
    a short run, or the wrong seeds, naming every shortfall."""
    problems = []
    row = pick_row(sp["grid"], p)
    if old["params"].get("kc_kc_scale", 1.0) != 1.0:
        problems.append(f"reproduce_old must be the old engine (kc_kc_scale=1.0), got {old['params'].get('kc_kc_scale')}")
    for name, d in (("reproduce", new), ("reproduce_reported", reported)):
        if d is not None and d["params"].get("kc_kc_scale", 1.0) != p.kc_kc_scale:
            problems.append(f"{name} must be the new engine (kc_kc_scale={p.kc_kc_scale}), got {d['params'].get('kc_kc_scale', 1.0)}")
    if th.get("kc_kc_scale", 1.0) != p.kc_kc_scale:
        problems.append(f"throughput must be the new engine (kc_kc_scale={p.kc_kc_scale}), got {th.get('kc_kc_scale', 1.0)}")
    if new["seeds"] != list(range(8, 16)):
        problems.append(f"judged conditioning seeds must be 8-15, got {new['seeds']}")
    if reported is not None and reported["seeds"] != list(range(8)):
        problems.append(f"reported conditioning seeds must be 0-7, got {reported['seeds']}")
    if old["conditioning_match"].get("n_results") != 40:
        problems.append(f"old-engine equivalence must cover 40 M0 results, got {old['conditioning_match'].get('n_results')}")
    if row.get("sparsity_seeds") != [100, 101, 102]:
        problems.append(f"sparsity seeds must be 100-102, got {row.get('sparsity_seeds')}")
    if list(row["rest_seeds"]) != list(range(100, 108)):
        problems.append(f"rest seeds must be 100-107, got {row['rest_seeds']}")
    if len(new["baseline"]["per_seed"]) != 8:
        problems.append(f"pool baseline must have 8 rest seeds, got {len(new['baseline']['per_seed'])}")
    odor_seeds = [r["seed"] for r in new["odor_runaway"]["per_seed"]]
    if odor_seeds != list(range(100, 164)):
        problems.append(f"odour-B runaway seeds must be 100-163, got {len(odor_seeds)} seeds")
    if new["odor_runaway"].get("sat_hz") != 150.0 or any(r["runaway"]["sat_hz"] != 100.0 for r in new["baseline"]["per_seed"]):
        problems.append("runaway thresholds must be 150 Hz (odour window) and 100 Hz (rest)")
    if not new["arm_equal"].get("pairs"):
        problems.append("reproduce must be run with --arm-equal")
    if not {r["workers"] for r in th["rows"]} >= {4, 8, 16}:
        problems.append(f"throughput must cover workers 4, 8, 16, got {sorted(r['workers'] for r in th['rows'])}")
    if problems:
        raise SystemExit("M0c summary refused (spec D.4 identity / sample sizes):\n  - " + "\n  - ".join(problems))


def pick_row(grid: list, p: Params) -> dict:
    """The sparsity grid row for these Params. Rows without `kc_kc_scale` are the M0 engine (1.0)."""
    rows = [g for g in grid if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac"), g.get("kc_kc_scale", 1.0))
            == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac, p.kc_kc_scale)]
    if not rows:
        raise SystemExit(f"no sparsity grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} "
                         f"mbon_hold_frac={p.mbon_hold_frac} kc_kc_scale={p.kc_kc_scale}")
    return rows[0]


def compose(p: Params, sp: dict, old: dict, new: dict, reported: dict | None, th: dict, limit_hours: float = 60.0,
            provenance: dict | None = None) -> dict:
    validate_inputs(p, sp, old, new, reported, th)
    row = pick_row(sp["grid"], p)
    baseline = {"mbon_hz_rest_trimmed": row["mbon_hz_rest_trimmed"], "mbon_hz_rest_trimmed_sd": row["mbon_hz_rest_trimmed_sd"],
                "per_seed": row.get("mbon_hz_rest_trimmed_per_seed"), "rest_seeds": row["rest_seeds"], "rest_ms": row["rest_ms"],
                "pool_diff": (new["sparsity_match"].get("diffs") or {}).get("mbon_hz_rest_trimmed")}
    runaway = {"rest_seeds": [r.get("seed", 100 + i) for i, r in enumerate(new["baseline"]["per_seed"])],
               "rest_n_kc_over_sat_per_seed": [r["runaway"]["n_kc_over_sat"] for r in new["baseline"]["per_seed"]],
               "rest_n_over_sat_per_seed": [r["runaway"]["n_over_sat"] for r in new["baseline"]["per_seed"]],
               "rest_spike_share_over_sat": new["baseline"]["runaway"]["spike_share_over_sat"],
               "odor_B_n_seeds": new["odor_runaway"]["n_seeds"], "odor_B_sat_hz": new["odor_runaway"]["sat_hz"],
               "odor_B_n_kc_over_sat_per_seed": [r["n_kc_over_sat"] for r in new["odor_runaway"]["per_seed"]],
               "odor_B_n_kc_over_100_per_seed": [r["n_kc_over_100"] for r in new["odor_runaway"]["per_seed"]],
               "odor_B_kc_hz_max": max(r["kc_hz_top5"][0] for r in new["odor_runaway"]["per_seed"]),
               "odor_B_seeds_with_kc_over_sat": new["odor_runaway"]["seeds_with_kc_over_sat"]}
    equivalence = {"old_conditioning": old["conditioning_match"], "old_sparsity": old["sparsity_match"],
                   "old_decide_equal": old["decide_equal"], "arm_equal": new["arm_equal"], "decide_equal": new["decide_equal"],
                   "new_sparsity_vs_in_process": new["sparsity_match"]}
    budget = budget_table(th["rows"], limit_hours=limit_hours)
    cond = new["conditioning"]
    judged = {k: cond[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")}
    judged["seeds"] = new["seeds"]
    out = {"params_frozen": dataclasses.asdict(p), **_git_state(), "inputs_sha256": provenance or {}, "kc_kc_scale_old_engine": 1.0,
           "seeds": {"sparsity": row["sparsity_seeds"], "rest": row["rest_seeds"], "odor_runaway": [r["seed"] for r in new["odor_runaway"]["per_seed"]],
                     "conditioning_judged": new["seeds"], "conditioning_reported": reported["seeds"] if reported else None},
           "sparsity": row, "baseline": baseline, "runaway": runaway, "conditioning": judged,
           "conditioning_reported_seeds": ({k: reported["conditioning"][k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")}
                                           if reported else None),
           "equivalence": equivalence, "throughput": th["rows"], "budget": budget, "memory": new.get("memory"),
           "reproduce_wall_clock_s": {"old": old.get("wall_clock_s"), "new": new.get("wall_clock_s")},
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    out["gate"] = m0c_gate(row, baseline, runaway, equivalence, budget, judged, limit_hours)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", default="results/m0c/sparsity.json")
    ap.add_argument("--reproduce-old", default="results/m0c/reproduce_old.json")
    ap.add_argument("--reproduce", default="results/m0c/reproduce.json")
    ap.add_argument("--reproduce-reported", default="results/m0c/reproduce_seeds0-7.json")
    ap.add_argument("--throughput", default="results/m0c/throughput.json")
    ap.add_argument("--out", default="results/summary/m0c.json")
    ap.add_argument("--limit-hours", type=float, default=60.0)
    a = ap.parse_args()
    refuse_old_engine_output(a.out, Params().kc_kc_scale)   # spec D.5: never over the old engine's summaries
    reported = json.loads(Path(a.reproduce_reported).read_text()) if Path(a.reproduce_reported).exists() else None
    paths = {"sparsity": a.sparsity, "reproduce_old": a.reproduce_old, "reproduce": a.reproduce, "throughput": a.throughput,
             **({"reproduce_reported": a.reproduce_reported} if reported is not None else {})}
    out = compose(Params(), _load(a.sparsity), _load(a.reproduce_old), _load(a.reproduce), reported, _load(a.throughput), a.limit_hours,
                  provenance={k: _sha256(v) for k, v in paths.items()})
    g = out["gate"]
    print(f"M0c gate {'PASS' if g['passed'] else 'FAIL'}: sparsity={g['sparsity_ok']} baseline={g['baseline_ok']} "
          f"({out['baseline']['mbon_hz_rest_trimmed']:.2f} Hz) runaway={g['runaway_ok']} equivalence={g['equivalence_ok']} "
          f"throughput={g['throughput_ok']} ({out['budget']['baseline_hours']:.1f} h); conditioning index flip "
          f"{'ok' if g['conditioning_index_flip_ok'] else 'FAIL (recorded)'}, channel-specific {g['channel_specific_seeds']}/{out['conditioning']['n_seeds']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
