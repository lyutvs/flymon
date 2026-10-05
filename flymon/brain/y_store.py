"""Y's only writer (Y.2 code boundary): results/y/ and results/summary/y_learning.json, atomic writes (temporary file +
rename), nothing else (SystemExit 2). The running ledger lives in the summary's `budget` block (Y.7 예산). YCache =
r_store.RCache keyed by the W measurement key (w_measure.py unchanged, Y.2) with `put` through this writer. Phase A
adds here and changes none of these.
- archive_stage (Y red-team P3-13 as amended by Y.9.2: at every block commit and before any STOP sentence): one copy of
  the stage's fixed file set (stage_files: results/y/oracle.json from the oracle stage on + the stage's detail files)
  under <ys.archive_root>/<stage>/ via r_store.archive_copy (sha-checked). An archive is never overwritten: a repeat for
  the same stage is a no-op when the archive holds exactly that set byte-identically (so the controller may call it
  again after committing), and refuses (SystemExit 2) otherwise."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from .h3_store import canonical, canonical_pretty, sha256_file
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy

ALLOWED_DIR = "results/y/"
SUMMARY = "results/summary/y_learning.json"


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
    if isinstance(o, np.generic):
        return o.item()
    return o


def to_json(obj):
    return json.loads(json.dumps(_plain(obj), sort_keys=True))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"Y writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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
    doc = read_summary(path)
    doc[block] = obj
    if ledger is not None:
        b = dict(doc.get("budget") or {})
        b["ledger"] = list(b.get("ledger", [])) + [ledger]
        doc["budget"] = b
    return write_json(path, doc, params_list)


class YCache(RCache):
    """RCache under results/y/ (a root elsewhere refuses on put); no smoke seeds in 0p (phase A passes its own)."""

    def __init__(self, root, code: dict, smoke_seeds=()):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


ARCHIVE_ORDER = ("digest", "oracle")          # phase A appends its stages (and their detail files) here


def stage_files(stage: str, ys) -> list:
    """The fixed file set archived for a stage (Task 1 review M2): oracle.json from the oracle stage on (it exists
    exactly then), plus the stage's own detail files (none in 0p besides oracle.json)."""
    if stage not in ARCHIVE_ORDER:
        _refuse(f"no archive file set for stage {stage!r}")
    own = {"digest": [], "oracle": []}.get(stage, [])
    return ([ys.oracle_detail] if ARCHIVE_ORDER.index(stage) >= ARCHIVE_ORDER.index("oracle") else []) + own


def archive_stage(stage: str, ys) -> list:
    """Copy stage_files(stage) to <ys.archive_root>/<stage>/<parent>/<name>; returns the manifest (src, dst, sha256).
    Never overwrites: a repeat is a no-op returning the same manifest when the archive holds exactly that file set,
    byte-identical to the sources; anything else (a file missing, extra, or differing) refuses. A missing source
    refuses. An empty set archives nothing."""
    src = stage_files(stage, ys)
    missing = [f for f in src if not Path(f).exists()]
    if missing:
        _refuse(f"stage {stage}: archive source(s) missing: {missing}")
    if not src:
        return []
    root = Path(os.path.expanduser(ys.archive_root))
    dest = root / stage
    if not dest.exists():
        return archive_copy(src, dest, root)
    want = {dest / Path(f).parent.name / Path(f).name: f for f in src}
    have = {p for p in dest.rglob("*") if p.is_file()}
    if have != set(want):
        _refuse(f"archive {dest} holds {sorted(map(str, have))}, not stage {stage}'s fixed set; never overwritten")
    out = []
    for dst, f in want.items():
        sha = sha256_file(f)
        if sha256_file(dst) != sha:
            _refuse(f"archive {dest} exists and differs at {f}; an archive is never overwritten")
        out.append(dict(src=str(f), dst=str(dst), sha256=sha))
    return out
