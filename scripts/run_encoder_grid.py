#!/usr/bin/env python3
"""The encoder-redesign track's stages (docs/superpowers/specs/2026-10-01-encoder-redesign-design.md 4.0):

    uv run python scripts/run_encoder_grid.py --stage set          # ⓪ judgement set list (no measurement)
    uv run python scripts/run_encoder_grid.py --stage drive        # ① single-glomerulus drive (C3, KC only)
    uv run python scripts/run_encoder_grid.py --stage codebook     # ② codebooks k = 3, 2
    uv run python scripts/run_encoder_grid.py --stage strength     # ③ strength calibration
    uv run python scripts/run_encoder_grid.py --stage oc           # ④ operating characteristics
    uv run python scripts/run_encoder_grid.py --stage even         # ⑤ even-turn oracle and selection
    uv run python scripts/run_encoder_grid.py --stage judge        # ⑥ judgement (once)
    ... --smoke [--workers 4]                                       # smoke seeds, results/encoder/smoke only

Writes results/summary/encoder_grid.json (smoke: results/encoder/smoke/summary.json) and the cache under
results/encoder/. Exit 0 for every recorded outcome (a STOP is a result, not an error) and prints its sentence;
exit 2 on a refusal (wrong cwd, connectome sha256, chain)."""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"


class Situations:
    """The runner's situation adapter over e_pairs (pops built once)."""

    def __init__(self, pops, spec):
        self.pops, self.spec = pops, spec

    @functools.lru_cache(maxsize=None)
    def even(self):
        from flymon.agent import e_pairs
        return e_pairs.even_situations(self.pops)

    @functools.lru_cache(maxsize=None)
    def _js(self):
        from flymon.agent import e_pairs
        return e_pairs.judgement_set(self.pops, self.spec)

    def judgement(self):
        return dict(self._js())

    @functools.lru_cache(maxsize=None)
    def used(self):
        from flymon.agent import e_pairs
        return e_pairs.used_situations(self.pops, self.spec)


def measure_files() -> tuple:
    from flymon.brain.h3_store import MEASURE_FILES
    return tuple(MEASURE_FILES) + ("flymon/brain/h4_jobs.py", "flymon/brain/k_jobs.py", "flymon/brain/h4_formula.py",
                                   "flymon/agent/e_measure.py", "flymon/agent/e_store.py")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("set", "drive", "codebook", "strength", "oc", "even", "judge"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)

    from flymon.brain.h3_store import ROOT, code_key, sha256_file
    from flymon.brain.h3_spec import SPEC as H3
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    if a.smoke and a.stage == "judge":
        print("refusing: smoke never judges", file=sys.stderr)
        return 2

    from flymon.agent import e_runner
    from flymon.agent.config import load_c3_config
    from flymon.agent.e_measure import EMeasurer
    from flymon.agent.e_spec import SPEC, smoke
    from flymon.agent.e_store import ECache
    from flymon.brain import h3_spec, odor_real
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlyPool

    spec = smoke(SPEC) if a.smoke else SPEC
    cfg = load_c3_config(SPEC.m0d_summary)
    m0d = json.loads(Path(SPEC.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    pops = Populations.from_connectome(Connectome.load(NPZ))
    gloms, c_norm = h3_spec.all51_glomeruli(pops)
    info = dict(receptor_counts={str(t): len(v) for t, v in pops.receptor_types.items()}, c_norm=c_norm,
                glomeruli=gloms, n_kc=len(pops.kc), max_rate_hz=float(cfg.params.max_rate_hz),
                cap_hz=float(odor_real.cap_hz(cfg.params)))
    oracle = dict(readout=dict(cfg.readout), z=dict(cfg.z), types=types)
    inputs = {NPZ: sha256_file(NPZ), SPEC.m0d_summary: sha256_file(SPEC.m0d_summary)}
    code = code_key(NPZ, files=measure_files())

    needs_pool = a.stage in ("drive", "strength", "even", "judge")
    pool = None
    if needs_pool:
        pool = FlyPool(NPZ, cfg.params, flies=[{}] * a.workers, workers=a.workers, punish_type="PPL105",
                       reward_type="PAM08", timeout_s=3600)
    try:
        m = EMeasurer(pool, ECache(f"{spec.raw_dir}/cache", code), cfg.params, info["n_kc"], spec)
        r = e_runner.Runner(m, info, spec, oracle=oracle, situations=Situations(pops, spec), smoke=a.smoke,
                            inputs=inputs)
        out = getattr(r, f"stage_{a.stage}")()
    finally:
        if pool is not None:
            pool.close()
    line = out.get("sentence") or json.dumps({k: v for k, v in out.items() if k != "reasons"}, ensure_ascii=False,
                                             default=str)
    print(f"{a.stage}: {out.get('outcome', out.get('band', out.get('status')))}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
