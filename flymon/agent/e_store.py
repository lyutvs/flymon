"""The encoder track's only writer (spec 6): raw files under results/encoder/, the one summary
results/summary/encoder_grid.json, atomic writes, and a content-key cache (h3_store.MeasureCache's form with this
track's path guard — h3_store refuses paths outside results/m0d/)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from ..brain.h3_store import canonical, canonical_pretty
from ..brain.pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/encoder/"
SUMMARY = "results/summary/encoder_grid.json"


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: the encoder track writes only under {ALLOWED_DIR} and {SUMMARY}",
              file=sys.stderr)
        raise SystemExit(2)


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
    guard(path, params_list)
    doc = read_summary(path)
    doc[block] = obj
    return write_json(path, doc, params_list)


class ECache:
    def __init__(self, root: str, code: dict):
        self.root, self.code = str(root), dict(code)
        self.hits = self.misses = 0

    def key(self, kind: str, inputs) -> str:
        blob = json.dumps(canonical({"kind": kind, "inputs": inputs, "code": self.code}), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, kind, inputs) -> Path:
        return Path(self.root) / kind / f"{self.key(kind, inputs)[:24]}.json"

    def get(self, kind, inputs):
        p = self._path(kind, inputs)
        return json.loads(p.read_text())["result"] if p.exists() else None

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"kind": kind, "result": result}, params_list)

    def get_or_compute(self, kind, inputs, compute, params_list):
        got = self.get(kind, inputs)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        res = compute()
        self.put(kind, inputs, res, params_list)
        return json.loads(canonical_pretty(res))
