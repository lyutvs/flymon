#!/usr/bin/env python3
"""Write results/summary/m0.json from the M0 gate outputs.

Reads the data manifest and the two gate result files, picks the sparsity grid
row matching the current Params() defaults, and freezes those defaults into the
summary. Run after both reproduce_flybrain_measurements.py subcommands.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from flymon.brain.config import Params

MANIFEST = Path("data/malecns.manifest.json")
SPARSITY = Path("results/m0/sparsity.json")
CONDITIONING = Path("results/m0/conditioning.json")
OUT = Path("results/summary/m0.json")


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"missing input: {path} (run the M0 gate first)")
        raise SystemExit(2)
    return json.loads(path.read_text())


def main() -> None:
    man = _load(MANIFEST)
    sp = _load(SPARSITY)
    co = _load(CONDITIONING)
    p = Params()
    pick = [g for g in sp["grid"] if g["kc_thresh"] == p.kc_thresh and g["apl_scale"] == p.apl_scale]
    if not pick:
        print(f"no sparsity grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} in {SPARSITY}")
        raise SystemExit(2)
    out = {
        "data_manifest": {k: man[k] for k in ("n_neurons", "n_edges", "counts")},
        "inputs_sha256": {k: v["sha256"] for k, v in man["inputs"].items()},
        "params_frozen": dataclasses.asdict(p),
        "sparsity": pick[0],
        "mbon_hz_rest": sp["mbon_hz_rest"],
        "conditioning": {"n_flip": co["n_flip"], "noplast_max_abs_dD": co["noplast_max_abs_dD"], "arms": co["arms"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
