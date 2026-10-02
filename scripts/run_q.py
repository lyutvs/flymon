#!/usr/bin/env python3
"""Spec appendix Q (Q.6 wins; Q.6.9 wins over Q.6.4): the reward-readout (MBON05) bottleneck diagnosis — a
characterisation.

    uv run python scripts/run_q.py --stage repro                    # Q.6.8 reproduction gate (3 encoder caches, H.4 seeds)
    uv run python scripts/run_q.py --stage smoke --workers 4        # smoke seeds 24_008_xxx, mv and (c) KC probes, cost
    uv run python scripts/run_q.py --stage s_c                      # (c) lowered s from the smoke block (no pool)
    uv run python scripts/run_q.py --stage q0                       # Q0 from results/q/q0_cache (no pool)
    uv run python scripts/run_q.py --stage q1 --condition base      # then apl_mbon05 apl_nonkc s_up mv_lo mv_hi
    uv run python scripts/run_q.py --stage records                  # the five candidates' records (no pool)

Condition "s_up" is (c) = the LOWERED-s manipulation (Q.6.9: s 0.7, 0.8 when 0.7's KC median is outside the band; both
outside -> "(c) 조작 불가", q1:s_up written INVALID without running, ① 판단 불가) despite its name.

Writes results/summary/q_reward.json (one block per stage; commit each before the next) and raw files under
results/q/ (smoke entries under results/q/smoke/cache, real ones under results/q/cache). Exit 0 for every recorded
outcome (an INVALID condition is a record), 4 when the reproduction gate fails, 2 on a refusal (cwd, connectome sha256,
chain, uncommitted summary, dirty hashed file).

There is no --allow-dirty: a dirty hashed Q file or an uncommitted summary always refuses, and real runs are made only
from a clean tree. A failed reproduction block (exit 4) must be committed (as a record) or discarded (delete the
untracked summary, or `git checkout -- results/summary/q_reward.json`) before repro is rerun."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_REPRO_FAIL = 4
POOL_STAGES = ("repro", "smoke", "q1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("repro", "smoke", "s_c", "q0", "q1", "records"))
    ap.add_argument("--condition")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)

    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, code_key, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    if (a.stage == "q1") != (a.condition is not None):
        print("refusing: --condition goes with --stage q1 only", file=sys.stderr)
        return 2

    from flymon.brain import q_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.q_measure import Q_MEASURE_FILES, QMeasurer
    from flymon.brain.q_spec import SPEC
    from flymon.brain.q_store import QCache

    try:
        ctx = q_runner.build_ctx(SPEC, NPZ)
        code = code_key(NPZ, files=Q_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        root = SPEC.smoke_cache_dir if a.stage == "smoke" else SPEC.cache_dir
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            m = QMeasurer(pool, QCache(root, code), SPEC, ctx["params"], ctx["readout"], ctx["z"], ctx["types"],
                          ctx["n_kc"])
            r = q_runner.Runner(m, ctx, SPEC, code=code)
            out = r.stage_q1(a.condition) if a.stage == "q1" else getattr(r, f"stage_{a.stage}")()
        finally:
            if pool is not None:
                pool.close()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    if a.stage == "records":
        print(out["s_up_note"])
        print("\n".join(out["sentences"]))
    else:
        print(json.dumps({k: v for k, v in out.items() if k not in ("vals", "files", "manifest")}, ensure_ascii=False,
                         default=str)[:2000])
    return EXIT_REPRO_FAIL if a.stage == "repro" and not out.get("passed") else 0


if __name__ == "__main__":
    sys.exit(main())
