#!/usr/bin/env python3
"""Write results/summary/m0.json from the M0 gate outputs.

Reads the data manifest and the two gate result files, picks the sparsity grid
row matching the current Params() defaults, checks that row against the M0 gate
(spec 5: KC sparsity 3-7%, overlap at or below chance, trimmed MBON baseline 3-4 Hz) and
freezes those defaults into the summary. A row that fails the gate is never
frozen. The conditioning block carries the graded gate statistics (`n_flip`, per-arm `mean_dD`) and,
when the run recorded it, the saturating index's `n_flip_disc` alongside.

The `gate` block records the M0 outcome as it actually came out: the pre-registered composite
criterion (graded index D flips sign between `both` and `reversed` on 8/8 seeds with
|mean dD| >= 0.3) FAILED, while sparsity and the per-channel odour-specific depression passed,
so the summary is written with `partial: true` rather than refused. Run after both
reproduce_flybrain_measurements.py subcommands.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from flymon.brain.conditioning import channel_specific_seeds
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


def index_flip_ok(co: dict) -> bool:
    """The pre-registered composite criterion. FAILS on the frozen M0 run (n_flip 0/8)."""
    both, rev = co["arms"]["both"]["mean_dD"], co["arms"]["reversed"]["mean_dD"]
    return bool(co["n_flip"] == co["n_seeds"] and abs(both) >= 0.3 and abs(rev) >= 0.3
                and (both > 0) != (rev > 0))


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
    sparsity_ok = gate_ok(row)
    if not sparsity_ok:
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
                         "settle_ms": co.get("params", {}).get("settle_ms"),
                         "da_baseline_ms": co.get("params", {}).get("da_baseline_ms"),
                         "arms": co["arms"]},
        "gate_runs": "results/m0/gate_runs.md",
    }
    n_spec = channel_specific_seeds(co["per_seed"])
    flip_ok = index_flip_ok(co)
    spec_ok = n_spec == co["n_seeds"]
    out["gate"] = {
        "sparsity_ok": sparsity_ok,
        "conditioning_index_flip_ok": flip_ok,
        "conditioning_channel_specific_ok": spec_ok,
        "channel_specific_seeds": n_spec,
        "passed": sparsity_ok and flip_ok,
        "partial": sparsity_ok and spec_ok and not flip_ok,
    }
    # the saturating index, when the run recorded it (older runs predate the graded index)
    if "n_flip_disc" in co:
        out["conditioning"]["n_flip_disc"] = co["n_flip_disc"]
    verdict = ("PASS" if out["gate"]["passed"] else "PARTIAL" if out["gate"]["partial"] else "FAIL")
    print(f"M0 gate {verdict}: sparsity_ok={sparsity_ok} index_flip_ok={flip_ok} "
          f"channel_specific={n_spec}/{co['n_seeds']} (pre-registered criterion is the index flip)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
