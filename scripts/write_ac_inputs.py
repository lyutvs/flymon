#!/usr/bin/env python3
"""Spec AC.7 3 (and the smoke / bench inputs of 2b): the confirmation set and the canonical schedules (learning
12 x 40 with seed 302; ONE 20-battle evaluation schedule with seed 303 that every fly plays), then the experiment-input
manifest (AC.1 (ii)).

    uv run python scripts/write_ac_inputs.py            # results/summary/ac_inputs.json + ac_inputs_manifest.json
    uv run python scripts/write_ac_inputs.py --smoke    # results/m4-smoke/inputs.json + inputs_manifest.json
    uv run python scripts/write_ac_inputs.py --bench    # results/m4-bench/...

Exit 0 (OK) or 2 (STOP_SET: fewer than n_pairs candidates). A run-mode document already on disk with other content is
never overwritten."""
from __future__ import annotations

import argparse
import json
import sys

from flymon.ac import schedules, store
from flymon.ac.spec import SPEC, bench, smoke


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--bench", action="store_true")
    a = ap.parse_args(argv)
    spec = smoke() if a.smoke else bench() if a.bench else SPEC
    doc_p, man_p = schedules.input_paths(spec)
    doc = schedules.inputs_doc(spec)
    if spec.mode == "run" and doc_p.exists():
        if json.loads(doc_p.read_text()) != json.loads(json.dumps(doc, sort_keys=True, default=float)):
            raise SystemExit(f"refusing: {doc_p} exists with other content (AC.1 freeze)")
    store.write_json(doc_p, doc)
    store.write_json(man_p, schedules.input_manifest(doc, doc_p))
    c = doc["confirm"]
    print(f"{doc['status']}: candidates {c['counts']}, drew {len(c['pairs'])} with seed {c['seed']}; "
          f"digests {({k: (v or '')[:12] for k, v in doc['digests'].items()})}; wrote {doc_p}, {man_p}; "
          f"{doc['label']}", flush=True)
    return 0 if doc["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
