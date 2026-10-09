"""AD.6 5: the raw-data list. Before the analysis every file under results/m4/ad/ (logs, checkpoints, wall_clock.json,
result.json, futility.json, the books, batch_times.json) and the run load log are listed with their sha256 and size,
and packed into ~/flymon-archive/ad-raw-<date>.tar.gz (outside git; its sha256 is in the list). The list
(results/summary/ad_raw_manifest.json) goes into the result commit. Recovery: unpack the archive at the repository root
(paths are stored relative to it) and verify() the list."""
from __future__ import annotations

import tarfile
from pathlib import Path

from .store import sha256_file

RAW = "results/summary/ad_raw_manifest.json"


def listing(root, extra=()) -> list:
    files = [p for p in Path(root).rglob("*") if p.is_file()] + [Path(x) for x in extra]
    return sorted((dict(path=p.as_posix(), sha256=sha256_file(p), bytes=p.stat().st_size) for p in files),
                  key=lambda e: e["path"])


def archive(entries, dest) -> dict:
    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as t:
        for e in entries:
            t.add(e["path"], arcname=e["path"])
    return dict(path=str(dest), sha256=sha256_file(dest), bytes=dest.stat().st_size)


def verify(doc: dict) -> list:
    bad = []
    for e in doc["files"]:
        p = Path(e["path"])
        if not p.is_file() or sha256_file(p) != e["sha256"]:
            bad.append(e["path"])
    return bad
