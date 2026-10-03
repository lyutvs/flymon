"""T's only writer (T.5, T.8): raw files under results/t/, the one summary results/summary/t_lever.json, atomic writes
(temporary file + rename). r_store.py is part of the shared measurement key (r_measure.R_MEASURE_FILES), so it is never
edited (T.3 1, T.8); T reuses it where it is path-free and wraps it where it is not (plan Reading 3):
- guard / write_bytes / write_json / read_summary / write_summary_block: r_store's code with T's allowed paths.
- TCache: r_store.RCache (ECache's key over the shared code key, RCache's smoke scope and its get) with only `put`
  routed through this module's writer and T's smoke seeds as the default scope — a T entry has R's layout and key
  formula but can only land under results/t/.
- RReadCache: r_store.RCache over R's own root, read only — `put` refuses. Gate ③ reads R's even raw (C, and L as a
  record) through it by content key (T.9.2); a missing entry is a refusal, never a measurement into R's tree.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for t_runner)
from .t_spec import SPEC as T_SPEC
from .t_spec import smoke

ALLOWED_DIR = "results/t/"
SUMMARY = "results/summary/t_lever.json"
SMOKE_SEEDS = frozenset(T_SPEC.smoke_seeds) | frozenset(smoke(T_SPEC).p.seeds) | frozenset(smoke(T_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"T writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


class TCache(RCache):
    """RCache with T's smoke seeds as the default scope and T's writer (a root outside results/t/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


class RReadCache(RCache):
    """R's cache root, read only (T.9.2): get as RCache, put refuses."""

    def put(self, kind, inputs, result, params_list) -> None:
        _refuse(f"{self.root} is R's raw data; T reads it by content key and never writes it")
