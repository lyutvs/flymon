"""Guarded writes of spec appendix K (K.8.7): run reports and self-checks under results/m0d/k/ and the summary
results/summary/k_engine.json; nothing else. The cache and threshold files also live under results/m0d/k/ but go
through h3_store (whose guard allows results/m0d/). Every write is atomic (tmp + os.replace)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/m0d/k/"
ALLOWED_FILES = ("results/summary/k_engine.json",)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel in ALLOWED_FILES):
        print(f"refusing to write {path}: spec K writes only under {ALLOWED_DIR} and {ALLOWED_FILES}", file=sys.stderr)
        raise SystemExit(2)


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(obj) + "\n").encode(), params_list)


def write_summary_block(path, block: str, obj, params_list) -> Path:
    """Replace one top-level block of the summary, keeping the others (scan, judge)."""
    guard(path, params_list)
    doc = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    doc[block] = json.loads(canonical_pretty(obj))
    return write_json(path, doc, params_list)
