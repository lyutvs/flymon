#!/usr/bin/env python3
"""Write results/summary/m0.json from the M0 gate outputs.

Reads the data manifest and the two gate result files, picks the sparsity grid
row matching the current Params() defaults, checks that row against the M0 gate
(spec 5: KC sparsity 3-7%, overlap at or below chance, trimmed MBON baseline 3-4 Hz) and
freezes those defaults into the summary. A row that fails the gate is never
frozen. Run after both reproduce_flybrain_measurements.py subcommands.
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


def gate_ok(row: dict) -> bool:
    """Spec 5 M0 gate on one sparsity grid row."""
    return (0.03 <= row["frac_active_A"] <= 0.07 and 0.03 <= row["frac_active_B"] <= 0.07
            and row["jaccard"] <= row["chance"] and 3.0 <= row["mbon_hz_rest_trimmed"] <= 4.0)


def main() -> None:
    man = _load(MANIFEST)
    sp = _load(SPARSITY)
    co = _load(CONDITIONING)
    p = Params()
    pick = [g for g in sp["grid"] if g["kc_thresh"] == p.kc_thresh and g["apl_scale"] == p.apl_scale
            and g.get("mbon_hold_frac") == p.mbon_hold_frac]
    if not pick:
        print(f"no sparsity grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} "
              f"mbon_hold_frac={p.mbon_hold_frac} in {SPARSITY}")
        raise SystemExit(2)
    row = pick[0]
    if not gate_ok(row):
        print("sparsity grid row for the current Params() fails the M0 gate: "
              f"frac_active_A={row['frac_active_A']} frac_active_B={row['frac_active_B']} "
              f"jaccard={row['jaccard']} chance={row['chance']} "
              f"mbon_hz_rest_trimmed={row['mbon_hz_rest_trimmed']} (raw {row['mbon_hz_rest']}) "
              "(want 0.03-0.07, 0.03-0.07, jaccard <= chance, 3.0-4.0 Hz)")
        raise SystemExit(2)
    out = {
        "data_manifest": {k: man[k] for k in ("n_neurons", "n_edges", "counts")},
        "inputs_sha256": {k: v["sha256"] for k, v in man["inputs"].items()},
        "params_frozen": dataclasses.asdict(p),
        "sparsity": row,
        "mbon_hz_rest": sp["mbon_hz_rest"],
        "mbon_hz_rest_trimmed": sp["mbon_hz_rest_trimmed"],
        # record which dopamine channels the run used (older runs predate the flag: flybrain pair)
        "conditioning": {"n_flip": co["n_flip"], "noplast_max_abs_dD": co["noplast_max_abs_dD"],
                         "punish_type": co.get("params", {}).get("punish_type", "PPL105"),
                         "reward_type": co.get("params", {}).get("reward_type", "PAM08"),
                         "arms": co["arms"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
