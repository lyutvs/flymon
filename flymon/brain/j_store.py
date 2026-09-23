"""Guarded writes of spec appendix J's summaries (J.10.8, J.11.6) and the C3 reference J starts from.

J's cache, threshold files and run reports live under results/m0d/j/ and go through h3_store (whose guard allows
results/m0d/). The two summaries go through `write_json` here, which calls both engine write guards for every Params,
allows only results/m0d/j/ and the two summaries, and refuses block "h3"'s file and C3's threshold copy by name: J reads
them and never writes them.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical_pretty, sha256_file, write_bytes as h3_write_bytes
from .j_params import params_from_json
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/m0d/j/"
ALLOWED_FILES = ("results/summary/std_scan.json", "results/summary/m2_engine.json")
READ_ONLY = ("results/summary/m0d.json", "results/summary/m0d_h3_c3_thresholds.npz")
C3_COPY = "results/summary/m0d_h3_c3_thresholds.npz"


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if rel in READ_ONLY or not (rel.startswith(ALLOWED_DIR) or rel in ALLOWED_FILES):
        print(f"refusing to write {path}: spec J writes only under {ALLOWED_DIR} and {ALLOWED_FILES}", file=sys.stderr)
        raise SystemExit(2)


def write_json(path, obj, params_list) -> Path:
    guard(path, params_list)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(canonical_pretty(obj) + "\n")
    os.replace(tmp, path)
    return path


def load_c3(summary: dict, block: str = "h3", name: str = "C3") -> tuple:
    """(C3's adopted Params, its guard {type: stats}, its adopted cell) from block "h3". A missing threshold file is
    restored from the committed copy when the sha256 matches (H.4's rule). ValueError when the block cannot be used."""
    try:
        c = summary[block]["combos"][name]
        if c["status"] != "adopted":
            raise ValueError(f"block {block}: {name} is {c['status']}, not adopted")
        params = params_from_json(c["adopted"]["params"])
        cell = next(x for x in c["cells"] if x.get("status") == "adopted")
        guard_stats = {t: st for k in ("A", "P") for t, st in cell["stage3"]["guard"][k]["types"].items()}
    except (KeyError, StopIteration, TypeError) as e:
        raise ValueError(f"block {block} is malformed: {e!r}") from None
    src = Path(params.kc_thresh_file)
    if not src.exists():
        if not Path(C3_COPY).exists() or sha256_file(C3_COPY) != params.kc_thresh_sha256:
            raise ValueError(f"{src} is missing and {C3_COPY} is missing or does not match {name}'s sha256")
        h3_write_bytes(src, Path(C3_COPY).read_bytes(), [params])
    if sha256_file(src) != params.kc_thresh_sha256:
        raise ValueError(f"{src}: sha256 differs from {name}'s Params")
    return params, guard_stats, cell
