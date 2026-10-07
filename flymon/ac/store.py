"""Where AC may write: the git-ignored work trees results/m4/, results/m4-smoke/, results/m4-bench/ and the tracked
summaries results/summary/ac_*.json. JSON is written atomically with sorted keys; digest() is the content hash the
manifests use."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..rescope.store import _atomic_write, git_provenance  # noqa: F401  (git_provenance is re-exported)

ALLOWED_DIRS = ("results/m4/", "results/m4-smoke/", "results/m4-bench/")
SUMMARY_PREFIX = "results/summary/ac_"


def rel(path) -> str:
    return os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")


def guard(path) -> Path:
    r = rel(path)
    if not (r.startswith(ALLOWED_DIRS) or (r.startswith(SUMMARY_PREFIX) and r.endswith(".json"))):
        raise SystemExit(f"refusing to write {r}: AC outputs go under {ALLOWED_DIRS} or {SUMMARY_PREFIX}*.json")
    return Path(path)


def _json_default(o):
    """numpy scalars -> the matching Python scalar (int stays int, bool stays bool); anything else is an error."""
    item = getattr(o, "item", None)
    if callable(item) and getattr(o, "shape", None) == ():
        return item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(path, obj) -> Path:
    p = guard(path)
    _atomic_write(p, json.dumps(obj, sort_keys=True, indent=1, default=_json_default).encode())
    return p


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
