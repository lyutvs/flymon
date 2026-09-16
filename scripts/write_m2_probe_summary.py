#!/usr/bin/env python3
"""Write results/summary/m2_probe.json: the M2 first-measurement record (spec appendix E).

This is NOT a gate. It records what the first M2 investigation measured, and it derives the spec D.6
ruling from the raw candidate map rather than restating it, so a wrong claim in prose cannot survive
here. Three things it is deliberate about:

  * D.6 (c) is "the M2 learning unit test fails". That test has not been run, so (c) is reported as
    "unmeasured" — never as passed. The 3% sparsity floor is a SEPARATE item of D.6's third bullet
    and is reported under its own name (`sparsity_floor_check`).
  * D.6 is an OR. If (a) or (b) is true the STD redesign condition has fired, and `std_condition_fired`
    says so regardless of what anyone decided to do about it.
  * The encoder vocabulary comparison (`vocab.json`) is recorded as exploratory only. Two independent
    red-team passes found it confounded and measured on the wrong axis for the pre-registered
    criterion, so it is not written as a recommendation.

Inputs (all produced by results/m2/scratch/*.py; none is overwritten here):
  results/m2/candidate_map.json       m2_probe.py    16 turns x 8 seeds, per-candidate MBON/KC record
  results/m2/candidate_map_meta.json  m2_probe.py    provisional encoder, turns, seeds, params
  results/m2/kc_code.json             m2_kc_code.py  within/between candidate KC-code Jaccard
  results/m2/all51_drive.json         m2_all51.py    51 glomeruli at matched total ORN drive
  results/m2/sep_grid.json            m2_sep_grid.py APL mode x kc_thresh separation grid
  results/m2/vocab.json               m2_vocab.py    encoder vocabulary comparison (exploratory)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SAT_HZ = 150.0
SPARSITY_FLOOR = 0.03          # spec D.6, third bullet
BAND = (0.05, 0.09)            # spec 5, M2: candidate KC activity 5-9%
RATIO_LIMIT = 2.0              # spec D.6 (b)

INPUTS = {
    "candidate_map": "results/m2/candidate_map.json",
    "candidate_map_meta": "results/m2/candidate_map_meta.json",
    "kc_code": "results/m2/kc_code.json",
    "all51_drive": "results/m2/all51_drive.json",
    "sep_grid": "results/m2/sep_grid.json",
    "vocab": "results/m2/vocab.json",
}
LEARNING_TEST = "results/m2/learning_unit_test.json"    # does not exist yet; that is the point


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing input {p}: run results/m2/scratch/*.py first")
    return json.loads(p.read_text())


def _git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"git_commit": head, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "", "git_dirty": None}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(meta: dict, cmap: dict) -> None:
    """Refuse to compose a record from a run whose identity does not match what the prose claims."""
    problems = []
    p = meta.get("params", {})
    if p.get("kc_kc_scale") != 0.0:
        problems.append(f"candidate map must be the M0c engine (kc_kc_scale=0.0), got {p.get('kc_kc_scale')}")
    if p.get("sub_ms") != 200.0:
        problems.append(f"D.6 (a) needs 200 ms sub-windows, got sub_ms={p.get('sub_ms')}")
    n_turns, n_seeds = len(meta.get("turns", [])), len(meta.get("seeds", []))
    if n_turns < 16:
        problems.append(f"expected >= 16 turns, got {n_turns}")
    if n_seeds < 8:
        problems.append(f"expected >= 8 paired-noise seeds, got {n_seeds}")
    if len(cmap.get("index", [])) != n_turns * n_seeds:
        problems.append(f"index has {len(cmap.get('index', []))} rows, expected {n_turns * n_seeds}")
    if problems:
        raise SystemExit("refusing to write the summary:\n  - " + "\n  - ".join(problems))


def d6_ruling(meta: dict, cmap: dict) -> dict:
    """Derive spec D.6 (a) and (b) and the separate 3% sparsity check from the raw candidate map."""
    turns = {t["turn"]: t for t in meta["turns"]}
    by_turn: dict = {}
    fracs, sub_max, over_a, n_pres = [], 0.0, 0, 0
    for (tn, sd), r in zip(cmap["index"], cmap["results"]):
        spikes = []
        for c in r["candidates"]:
            n_pres += 1
            fracs.append(c["kc_active_frac"])
            sub_max = max(sub_max, c["kc_max_hz_sub"])
            over_a += int(c["n_kc_sub_over_150"] > 0)
            spikes.append(c["kc_spikes"])
        by_turn.setdefault(tn, []).append(max(spikes) / max(1, min(spikes)))
    ratios = {tn: {"mean": float(np.mean(v)), "max": float(np.max(v))} for tn, v in by_turn.items()}
    turns_over = sorted(tn for tn, v in ratios.items() if v["max"] > RATIO_LIMIT)
    fr = np.array(fracs)
    return {
        "a_runaway": {
            "condition": "any KC above 150 Hz for a 200 ms window of the candidate presentation",
            "fired": bool(over_a > 0),
            "n_presentations": n_pres, "n_over": int(over_a),
            "max_kc_sub_window_hz": float(sub_max),
            "margin_hz": float(sub_max - SAT_HZ),
            "caveat": ("measured on 7 grid-aligned 200 ms tiles, not a sliding window, and the maximum "
                       "equals the threshold exactly (margin 0.0 Hz). D.6 does not fix the window "
                       "alignment; a sliding 200 ms window and a larger sample can flip this."),
        },
        "b_candidate_ratio": {
            "condition": f"within-turn candidate KC spike-count ratio > {RATIO_LIMIT}",
            "fired": bool(turns_over),
            "turns_over": turns_over,
            "n_turns_over": len(turns_over), "n_turns": len(ratios),
            "ratio_mean_over_turns": float(np.mean([v["mean"] for v in ratios.values()])),
            "ratio_max": float(max(v["max"] for v in ratios.values())),
            "per_turn": {str(tn): v for tn, v in sorted(ratios.items())},
        },
        "c_learning_unit_test": {
            "condition": "the M2 learning unit test fails",
            "status": "unmeasured",
            "fired": None,
            "expected_input": LEARNING_TEST,
            "present": Path(LEARNING_TEST).exists(),
            "note": ("the learning unit test (naive d' ~ 0, d' >= 1 after 20 rewards, drop after 20 "
                     "punishments) has not been run. The first write-up of this investigation "
                     "(2026-09-16) recorded the 3% sparsity check under this label and called it "
                     "passed; that was wrong and is corrected here."),
        },
        "sparsity_floor_check": {
            "note": "D.6 third bullet, a SEPARATE item from (c)",
            "floor": SPARSITY_FLOOR,
            "n_below_floor": int((fr < SPARSITY_FLOOR).sum()),
            "min": float(fr.min()), "median": float(np.median(fr)), "max": float(fr.max()),
            "spec5_band": list(BAND),
            "n_in_spec5_band": int(((fr >= BAND[0]) & (fr <= BAND[1])).sum()),
            "frac_in_spec5_band": float(((fr >= BAND[0]) & (fr <= BAND[1])).mean()),
            "n_presentations": n_pres,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/summary/m2_probe.json")
    a = ap.parse_args()

    data = {k: _load(v) for k, v in INPUTS.items()}
    validate(data["candidate_map_meta"], data["candidate_map"])
    ruling = d6_ruling(data["candidate_map_meta"], data["candidate_map"])
    fired = [k for k in ("a_runaway", "b_candidate_ratio") if ruling[k]["fired"]]

    kc51 = np.array([r[4] for r in data["all51_drive"]["rows"]])       # [receptor, nORN, orn, pn, kc, kc_on, sd]
    meta = data["candidate_map_meta"]
    out = {
        "milestone": "M2 first measurement (spec appendix E) — record, not a gate",
        "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **_git_state(),
        "inputs_sha256": {k: _sha256(v) for k, v in INPUTS.items()},
        "engine": meta["params"],
        "encoder": {
            "status": "provisional — spec 3.3 as written, implemented in results/m2/scratch/m2_probe.py",
            "n_channels": meta["n_channels"],
            "glomeruli_per_candidate": meta["glomeruli_per_candidate"],
            "channels": meta["channels"],
        },
        "sample": {"n_turns": len(meta["turns"]), "seeds": meta["seeds"],
                   "source": "flymon.battle.pool, deterministic turns; NOT real battle logs"},
        "d6_ruling": ruling,
        "std_condition_fired": bool(fired),
        "std_condition_fired_by": fired,
        "kc_code_separation": data["kc_code"]["mean"],
        "glomerulus_drive_51": {
            "protocol": "matched total ORN drive, 4 seeds",
            "min": float(kc51.min()), "median": float(np.median(kc51)), "max": float(kc51.max()),
            "max_over_min": float(kc51.max() / max(kc51.min(), 1e-9)),
            "cv": float(kc51.std() / kc51.mean()),
            "n_ge_200_spikes": int((kc51 >= 200).sum()),
        },
        "engine_axis_grid": {
            "note": ("APL mode x kc_thresh; mean KC-Jaccard separation moved 0.227-0.290 while candidate "
                     "KC activity moved 3.0%-21.5%. Scope: APL gain and the GLOBAL kc_thresh multiplier "
                     "only — the KC threshold normalisation RULE and its clip bounds (spec D.2, D.7) "
                     "were not varied."),
            "cells": data["sep_grid"],
        },
        "vocabulary_comparison": {
            "status": "EXPLORATORY — withdrawn as a basis for choosing an encoder",
            "why": ("two independent red-team passes (results/m2/red-team/) found it (i) confounded — "
                    "glomerulus selection rule, vocabulary structure, channel duplication and presentation "
                    "strength all changed together; (ii) measured on the within-turn axis while the "
                    "pre-registered primary criterion (spec 4.3 #1) judges the opponent-type axis, which "
                    "is absent from these turns; (iii) drawn from the same 16 turns used to pick the "
                    "glomeruli, leaving no holdout (spec 4.4)."),
            "cells": data["vocab"],
        },
        "red_team": {
            "runs": [{"external_model": "gpt-5.6-sol", "verdict": "RETHINK", "findings": "20 (P0 5, P1 9, P2 5, P3 1)"},
                     {"external_model": "gpt-6-astra", "verdict": "RETHINK", "findings": "19 (P0 3, P1 10, P2 5, P3 1)"}],
            "record": "results/m2/red-team/",
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    print(f"  D.6 (a) fired={ruling['a_runaway']['fired']} "
          f"(max sub-window {ruling['a_runaway']['max_kc_sub_window_hz']:.1f} Hz, "
          f"margin {ruling['a_runaway']['margin_hz']:+.1f} Hz)")
    print(f"  D.6 (b) fired={ruling['b_candidate_ratio']['fired']} "
          f"(turns over: {ruling['b_candidate_ratio']['turns_over']}, "
          f"max {ruling['b_candidate_ratio']['ratio_max']:.2f})")
    print(f"  D.6 (c) {ruling['c_learning_unit_test']['status']}")
    s = ruling["sparsity_floor_check"]
    print(f"  sparsity floor check: {s['n_below_floor']} below {SPARSITY_FLOOR:.0%}, min {s['min']:.2%}; "
          f"spec 5 band {s['frac_in_spec5_band']:.0%} of {s['n_presentations']}")
    print(f"  STD condition fired: {out['std_condition_fired']} by {out['std_condition_fired_by']}")


if __name__ == "__main__":
    main()
