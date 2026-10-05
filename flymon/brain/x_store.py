"""X's only writer (X.2 · X.8; 글쓴이 해석 (8)): results/x/ and results/summary/x_learning.json, atomic writes
(temporary file + rename), nothing else (SystemExit 2). W's tree is read only (w_store's writer refuses outside
results/w/, so X never uses it). Every object is made plain JSON first (`to_json`: numpy scalars — bools included —
and arrays, tuples, dataclasses, non-string keys), so a block or detail never fails to serialise. Phase B adds XCache
here."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from .h3_store import canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/x/"
SUMMARY = "results/summary/x_learning.json"


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _plain(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _plain(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):                    # np.bool_, np.integer, np.floating, …
        return o.item()
    return o


def to_json(obj):
    """The plain-JSON form of obj (what a reader of the written file gets back)."""
    return json.loads(json.dumps(_plain(obj), sort_keys=True))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"X writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(to_json(obj)) + "\n").encode(), params_list)


def read_summary(path=SUMMARY) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def write_summary_block(path, block: str, obj, params_list, ledger: dict | None = None) -> Path:
    """One block, plus (optionally) one entry appended to the running ledger (X.8's own budget)."""
    doc = read_summary(path)
    doc[block] = obj
    if ledger is not None:
        doc["ledger"] = list(doc.get("ledger", [])) + [ledger]
    return write_json(path, doc, params_list)
