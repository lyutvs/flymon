"""Z's only writer (Z.2 code boundary as amended by Z.9.2 P1-5 — reimplemented, y_store is never called to write):
results/z/, results/summary/z_learning.json and results/summary/z_oc_draws.json, atomic writes (temporary file +
rename), anything else SystemExit 2. The engine-output guards of every writer in this repository run first. Two
ledgers live in the summary's `budget` block: `ledger` (core, Z cap 24 h) and `records_ledger` (records, 6 h; Z.9.2
P2-10); an env mismatch (P2-11) is appended to `ledger` as an entry with `env_mismatch`.
- archive(files, dest): one copy of a fixed file set under <archive_root>/<dest>/<parent>/<name> (r_store.archive_copy,
  sha-checked). Never overwritten: a repeat is a no-op when the archive holds exactly that set byte-identically,
  anything else refuses (SystemExit 2) — y_store.archive_stage's rule, reimplemented."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from .h3_store import canonical_pretty, sha256_file
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import archive_copy

ALLOWED_DIR = "results/z/"
SUMMARY = "results/summary/z_learning.json"
DRAWS = "results/summary/z_oc_draws.json"


def refuse(msg: str, code: int = 2):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


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
    if not (rel.startswith(ALLOWED_DIR) or rel in (SUMMARY, DRAWS)):
        refuse(f"Z writes only under {ALLOWED_DIR}, {SUMMARY} and {DRAWS}, not {path}")


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


def append_ledger(path, entry: dict, params_list, which: str = "ledger") -> Path:
    doc = read_summary(path)
    b = dict(doc.get("budget") or {})
    b[which] = list(b.get(which, [])) + [entry]
    doc["budget"] = b
    return write_json(path, doc, params_list)


def write_summary_block(path, block: str, obj, params_list, ledger: dict | None = None,
                        records_ledger: dict | None = None) -> Path:
    doc = read_summary(path)
    doc[block] = obj
    b = dict(doc.get("budget") or {})
    if ledger is not None:
        b["ledger"] = list(b.get("ledger", [])) + [ledger]
    if records_ledger is not None:
        b["records_ledger"] = list(b.get("records_ledger", [])) + [records_ledger]
    if b:
        doc["budget"] = b
    return write_json(path, doc, params_list)


def archive(files: list, dest: str, zs) -> list:
    """Copy `files` to <zs.archive_root>/<dest>/; never overwrites (module docstring)."""
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        refuse(f"archive {dest}: source(s) missing: {missing[:3]}")
    if not files:
        return []
    root = Path(os.path.expanduser(zs.archive_root))
    d = root / dest
    if not d.exists():
        return archive_copy(list(files), d, root)
    want = {d / Path(f).parent.name / Path(f).name: f for f in files}
    have = {p for p in d.rglob("*") if p.is_file()}
    if have != set(want):
        refuse(f"archive {d} holds another file set; an archive is never overwritten")
    out = []
    for dst, f in want.items():
        sha = sha256_file(f)
        if sha256_file(dst) != sha:
            refuse(f"archive {d} differs at {f}; an archive is never overwritten")
        out.append(dict(src=str(f), dst=str(dst), sha256=sha))
    return out
