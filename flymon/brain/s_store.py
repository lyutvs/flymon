"""S's only writer (S.5, S.8): raw files under results/s/, the one summary results/summary/s_lever.json, atomic writes
(temporary file + rename). r_store.py is part of the shared measurement key (r_measure.R_MEASURE_FILES), so it is never
edited (S.3 ①, S.9.5); S reuses it where it is path-free and wraps it where it is not (plan Reading 3):
- guard / write_bytes / write_json / read_summary / write_summary_block: r_store's code with S's allowed paths.
- SCache: r_store.RCache (ECache's key over the shared code key, RCache's smoke scope and its get) with only `put`
  routed through this module's writer and S's smoke seeds as the default scope — so an S entry has R's layout and
  key formula but can only land under results/s/.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for s_runner)
from .s_spec import SPEC as S_SPEC
from .s_spec import smoke

ALLOWED_DIR = "results/s/"
SUMMARY = "results/summary/s_lever.json"
SMOKE_SEEDS = frozenset(S_SPEC.smoke_seeds) | frozenset(smoke(S_SPEC).p.seeds) | frozenset(smoke(S_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"S writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(obj) + "\n").encode(), params_list)


def read_summary(path=SUMMARY) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def write_summary_block(path, block: str, obj, params_list) -> Path:
    doc = read_summary(path)
    doc[block] = obj
    return write_json(path, doc, params_list)


class SCache(RCache):
    """RCache with S's smoke seeds as the default scope and S's writer (a root outside results/s/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)
