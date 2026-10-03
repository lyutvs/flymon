"""U's only writer (U.5, U.8): raw files under results/u/, the one summary results/summary/u_lever.json, atomic writes
(temporary file + rename). r_store.py is in the shared measurement key and t_store.py hardcodes T's paths; U reuses
their path-free parts and wraps the rest:
- guard / write_bytes / write_json / read_summary / write_summary_block: t_store's code with U's allowed paths.
- UCache: r_store.RCache (ECache's key, RCache's smoke scope and its get) keyed by the U measurement key, with `put`
  routed through this module's writer and U's smoke seeds as the default scope — a U entry lands only under results/u/.
- RReadCache: T's (R's root, read only — `put` refuses). The endpoint gate and the even stage read R's even raw by
  content key through it.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for u_runner)
from .t_store import RReadCache  # noqa: F401  (re-exported for u_runner)
from .u_spec import SPEC as U_SPEC
from .u_spec import smoke

ALLOWED_DIR = "results/u/"
SUMMARY = "results/summary/u_lever.json"
SMOKE_SEEDS = frozenset(U_SPEC.smoke_seeds) | frozenset(smoke(U_SPEC).p.seeds) | frozenset(smoke(U_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"U writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


class UCache(RCache):
    """RCache with U's smoke seeds as the default scope and U's writer (a root outside results/u/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)
