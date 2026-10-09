#!/usr/bin/env python3
"""Spec AD.6 3 (and the smoke / bench inputs): the AD input document (reused confirmation set and evaluation schedule,
learning schedule 306 with its prefix digests, new 20 pairs 308, layout, tags) and the AD input manifest with the
stage-1 reuse fields (가) and the AD-fixed fields (나).

    uv run python scripts/write_ad_inputs.py            # results/summary/ad_inputs.json + ad_inputs_manifest.json
    uv run python scripts/write_ad_inputs.py --smoke    # results/m4-smoke/ad/inputs.json + inputs_manifest.json
    uv run python scripts/write_ad_inputs.py --bench 1  # results/m4-bench/ad-1/...

Exit 0 (OK) or 2 (STOP_REUSE: a reused stage-1 field differs; STOP_SET). A run-mode document already on disk with other
content is never overwritten."""
from __future__ import annotations

import argparse
import json
import sys

from flymon.ad import inputs, reuse, store
from flymon.ad.spec import SPEC, bench, smoke


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--bench", type=int, default=None, metavar="N")
    a = ap.parse_args(argv)
    spec = smoke() if a.smoke else bench(a.bench) if a.bench is not None else SPEC
    doc_p, man_p = inputs.input_paths(spec)
    doc = inputs.inputs_doc(spec)
    if spec.mode == "run" and doc_p.exists():
        if json.loads(doc_p.read_text()) != json.loads(json.dumps(doc, sort_keys=True, default=float)):
            raise SystemExit(f"refusing: {doc_p} exists with other content (AD.2 freeze)")
    store.write_json(doc_p, doc)
    fields, problems = None, []
    if spec.mode == "run":
        fields = reuse.reuse_fields(doc["digests"]["confirm"], doc["digests"]["eval"], spec)
        problems = reuse.reuse_problems(fields, spec)
    man = inputs.input_manifest(doc, doc_p, fields, problems)
    store.write_json(man_p, man)
    print(f"{man['status']}: digests {({k: (v or '')[:12] for k, v in doc['digests'].items()})}; new set from "
          f"{doc['new']['n_rest']} left; reuse problems {problems}; wrote {doc_p}, {man_p}; {doc['label']}", flush=True)
    return 0 if man["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
