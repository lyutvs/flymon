#!/usr/bin/env python3
"""Spec AC.7 2a, last step: freeze the model manifest (AC.1 (i)) from the tau_rec-on-L_V summary.

    uv run python scripts/write_ac_model_manifest.py [--taurec results/summary/rescope_taurec_lv.json]

Writes results/summary/ac_model_manifest.json: status FROZEN (exit 0) or STOP_NO_RECOVERY (exit 2, AC stops). An
existing FROZEN manifest with other content is never overwritten (a new one needs a new user decision)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flymon.ac import store
from flymon.ac.config import ENCODER_GRID, TAUREC_LV, load_lv_config, lv_codebook
from flymon.ac.manifest import MODEL_MANIFEST, build_model_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taurec", default=TAUREC_LV)
    a = ap.parse_args(argv)
    p = Path(a.taurec)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (run scripts/run_rescope_taurec.py --lv first)")
    taurec = json.loads(p.read_text())
    ok = taurec.get("status") == "SELECTED"
    cfg = load_lv_config(float(taurec["recovery_per_pulse"]) if ok else 0.0)
    _, dg = lv_codebook()
    doc = build_model_manifest(taurec, dict(path=str(p), sha256=store.sha256_file(p)), cfg, dg,
                               store.sha256_file(ENCODER_GRID))
    out = Path(MODEL_MANIFEST)
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("status") == "FROZEN" and old != json.loads(json.dumps(doc, sort_keys=True, default=store._json_default)):
            raise SystemExit(f"refusing: {out} is frozen with other content (AC.1); a new manifest needs a user decision")
    store.write_json(out, doc)
    print(f"{doc['status']}: r = {doc.get('recovery_per_pulse')}; wrote {out}", flush=True)
    return 0 if doc["status"] == "FROZEN" else 2


if __name__ == "__main__":
    sys.exit(main())
