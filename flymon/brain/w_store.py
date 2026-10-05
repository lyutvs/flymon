"""W's only writer (W.8): raw files under results/w/, the one summary results/summary/w_learning.json, atomic writes
(temporary file + rename), nothing else (SystemExit 2).
- WCache: r_store.RCache (key over kind + inputs + the W measurement key; smoke and real entries never share a root)
  with `put` routed through this module's writer and W's smoke seeds as the smoke scope.
- VReadCache: V's cache root (results/v/cache) read only under V's measurement key (= U's) — the path gate reads V's
  gate ② arm rows by content key; put refuses.
- load_manifest / archive_copy: r_store's (paths and root are arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for w_runner)
from .w_spec import SPEC as W_SPEC

ALLOWED_DIR = "results/w/"
SUMMARY = "results/summary/w_learning.json"
SMOKE_SEEDS = W_SPEC.smoke_seed_set()


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"W writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


def write_summary_block(path, block: str, obj, params_list, ledger: dict | None = None) -> Path:
    """One block, plus (optionally) one entry appended to the running budget ledger (W.9.6 F)."""
    doc = read_summary(path)
    doc[block] = obj
    if ledger is not None:
        doc["ledger"] = list(doc.get("ledger", [])) + [ledger]
    return write_json(path, doc, params_list)


class WCache(RCache):
    """RCache with W's smoke seeds as the smoke scope and W's writer (a root outside results/w/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


class VReadCache(RCache):
    """V's raw data, read only (W.3 2): get as RCache; put refuses."""

    def __init__(self, root, code: dict):
        super().__init__(root, code, smoke_seeds=())

    def put(self, kind, inputs, result, params_list) -> None:
        _refuse(f"{self.root} is V's raw data; W reads it by content key and never writes it")
