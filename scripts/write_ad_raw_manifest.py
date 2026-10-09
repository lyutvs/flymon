#!/usr/bin/env python3
"""Spec AD.6 5: list every raw file of the AD run with its sha256 and archive them outside git, BEFORE the analysis.

    uv run python scripts/write_ad_raw_manifest.py --extra .superpowers/sdd/2026-10-10-ad-stage2/runs/ad_run_load.log

Writes results/summary/ad_raw_manifest.json and ~/flymon-archive/ad-raw-<YYYYMMDD>.tar.gz."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from flymon.ad import raw, store
from flymon.ad.spec import LABEL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/m4/ad")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--archive-dir", default="~/flymon-archive")
    a = ap.parse_args(argv)
    if not (Path(a.root) / "result.json").exists():
        raise SystemExit(f"refusing: {a.root}/result.json does not exist (the run is not finished or stopped)")
    entries = raw.listing(a.root, a.extra)
    day = dt.datetime.now().strftime("%Y%m%d")
    arc = raw.archive(entries, Path(a.archive_dir).expanduser() / f"ad-raw-{day}.tar.gz")
    doc = dict(files=entries, archive=arc, root=a.root, n_files=len(entries),
               restore="tar -xzf <archive> -C <repository root>; then flymon.ad.raw.verify(this list)", label=LABEL)
    store.write_json(raw.RAW, doc)
    print(f"listed {len(entries)} files ({sum(e['bytes'] for e in entries)} bytes); archive {arc['path']} "
          f"sha256 {arc['sha256'][:12]}; wrote {raw.RAW}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
