"""V's only writer (V.5, V.8): raw files under results/v/, the one summary results/summary/v_lever.json, atomic writes
(temporary file + rename). u_store.py hardcodes U's paths; V reuses the path-free parts of r_store / t_store:
- guard / write_bytes / write_json / read_summary / write_summary_block: u_store's code with V's allowed paths.
- VCache: r_store.RCache (its key, smoke scope and get) keyed by the V measurement key (= U's, V.9.5 P1-3), with `put`
  routed through this module's writer and V's smoke seeds as the default scope — a V entry lands only under results/v/.
- RReadCache: T's (R's root, read only — `put` refuses). The path and even stages read R's even raw through it.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for v_runner)
from .t_store import RReadCache  # noqa: F401  (re-exported for v_runner)
from .v_spec import SPEC as V_SPEC
from .v_spec import smoke

ALLOWED_DIR = "results/v/"
SUMMARY = "results/summary/v_lever.json"
SMOKE_SEEDS = frozenset(V_SPEC.smoke_seeds) | frozenset(smoke(V_SPEC).p.seeds) | frozenset(smoke(V_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"V writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


class VCache(RCache):
    """RCache with V's smoke seeds as the default scope and V's writer (a root outside results/v/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)
