#!/usr/bin/env python3
"""Spec G.8: run the D.6 (a) re-judgement and write its raw record (git-excluded; flymon/brain/d6a.py has the rule).

    uv run python scripts/run_m2_d6a.py --smoke     # turn 0 x seeds 400-401 -> results/m2/d6a/smoke.json
    uv run python scripts/run_m2_d6a.py             # 41 odours x seeds 400-463 -> results/m2/d6a/g8.json

The engine is Params() (the M0c engine, every H.2 mode off); nothing on the command line changes it. Before measuring,
a full run re-presents two of E.1's recorded cells (turn 12 seed 203 — E.2's 150.0 Hz maximum — and turn 0 seed 200)
and requires the read-window KC record to equal results/m2/candidate_map.json bit for bit and the sliding maximum to be
at least E.2's 200 ms tile maximum. The pool is called once per block of 8 seeds, far below FlyPool's 3600 s timeout.
Exit codes: 0 done, whatever D.6 (a) reads; 2 refused — not at the repository root, tracked files dirty on a full run,
the odours are not E.1's, the output exists (move it aside first), or the E.2 self-check is missing or fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool

NPZ = "data/malecns.npz"
OUT_DIR = Path("results/m2/d6a")
CODE = ("flymon/brain/d6a.py", "flymon/brain/h4_jobs.py", "flymon/brain/h4_pairs.py", "scripts/run_m2_d6a.py")
SEED_BLOCK = 8
E2_RAW = Path("results/m2/candidate_map.json")     # E.1's recorded candidate map (git-excluded)
E2_CELLS = ((12, 203), (0, 200))                   # E.2's 150.0 Hz maximum, and the first recorded cell


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    dirty = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": dirty}


def e2_check(pool, by_turn: dict) -> dict:
    """Re-present E2_CELLS: read-window KC activity and spike count equal E.1's record; sliding max >= its tile max."""
    rec = json.loads(E2_RAW.read_text())
    ref = {tuple(k): r for k, r in zip(rec["index"], rec["results"])}
    res = pool.run_jobs(d6a.d6a_job, [dict(odors=by_turn[t], seed=s) for t, s in E2_CELLS])
    cells, ok = [], True
    for (t, s), r in zip(E2_CELLS, res):
        for j, c in enumerate(ref[(t, s)]["candidates"]):
            same = r["kc_active_frac"][j] == c["kc_active_frac"] and r["kc_spikes"][j] == c["kc_spikes"]
            ge = r["max_win"][j] * 1000.0 / d6a.WINDOW_MS >= c["kc_max_hz_sub"]
            ok = ok and same and ge
            cells.append({"turn": t, "seed": s, "candidate": j, "read_equal": same, "max_win": r["max_win"][j],
                          "e2_tile_max_hz": c["kc_max_hz_sub"]})
    return {"ok": ok, "cells": cells}


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if not (Path("flymon/brain/d6a.py").is_file() and Path(NPZ).is_file()):
        return refuse("run from the repository root with data/malecns.npz present")
    out = OUT_DIR / ("smoke.json" if a.smoke else "g8.json")
    if out.exists():
        return refuse(f"{out} exists; move it aside first")
    git = git_state()
    if git["dirty"] and not a.smoke:
        return refuse(f"tracked files are dirty {git['dirty']}")
    if not a.smoke and not E2_RAW.is_file():
        return refuse(f"{E2_RAW} (E.1's record) is needed for the self-check")
    code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in CODE}      # hashed before measuring
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    odours = d6a.candidate_odours(pops)
    if len(odours) != d6a.N_ODOURS or d6a.odours_digest(odours) != d6a.ODOURS_DIGEST:
        return refuse("the candidate odours are not E.1's 41 (digest differs)")
    del conn
    by_turn = {}
    for o in odours:
        by_turn.setdefault(o["turn"], []).append(o["odor"])
    turns = [0] if a.smoke else sorted(by_turn)
    seeds = list(d6a.SEEDS[:2] if a.smoke else d6a.SEEDS)
    started, t0 = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), time.perf_counter()
    rows = []
    with FlyPool(NPZ, Params(), [{} for _ in range(a.workers)], workers=a.workers) as pool:
        print(f"pool up in {time.perf_counter() - t0:.0f}s; {len(turns)} turns x {len(seeds)} seeds", flush=True)
        check = None if a.smoke else e2_check(pool, by_turn)
        if check is not None:
            print(f"E.2 self-check {'ok' if check['ok'] else 'FAILED'}: {check['cells']}", flush=True)
            if not check["ok"]:
                return refuse("the E.2 self-check failed; nothing measured")
        for i in range(0, len(seeds), SEED_BLOCK):
            block = seeds[i:i + SEED_BLOCK]
            cells = [(t, s) for t in turns for s in block]
            res = pool.run_jobs(d6a.d6a_job, [dict(odors=by_turn[t], seed=s) for t, s in cells])
            rows += [{"turn": t, **r} for (t, _), r in zip(cells, res)]
            top = max(w for r in rows for w in r["max_win"])
            print(f"seeds {block[0]}-{block[-1]} done, {time.perf_counter() - t0:.0f}s, max window so far {top}",
                  flush=True)
    raw = {"what": "spec G.8: D.6 (a) re-judgement (E.7 #4) - raw record; the rule is flymon/brain/d6a.py judge()",
           "smoke": bool(a.smoke), "started_utc": started, "wall_s": round(time.perf_counter() - t0, 1), "git": git,
           "code_sha256": code,
           "params": d6a.engine_params(), "odours": odours, "odours_digest": d6a.ODOURS_DIGEST, "seeds": seeds,
           "strength": d6a.STRENGTH, "settle_ms": d6a.SETTLE_MS, "read_ms": d6a.READ_MS, "window_ms": d6a.WINDOW_MS,
           "over_spikes": d6a.OVER_SPIKES, "workers": a.workers, "e2_check": check, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=1) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out} ({len(rows)} rows, {raw['wall_s'] / 60:.1f} min)", flush=True)
    if not a.smoke:
        v = d6a.judge(raw)
        print(f"D.6 (a) {'MET' if v['fired'] else 'not met'}: {v['n_over']}/{v['n_presentations']} presentations over, "
              f"max window {v['max_win_spikes']} spikes ({v['max_win_hz']:.0f} Hz), margin {v['margin_spikes']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
