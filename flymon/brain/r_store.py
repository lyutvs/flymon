# flymon/brain/r_store.py
"""R's only writer (R.5, R.8, R.9.7): raw files under results/r/ (results/r/ref/ — the encoder ③ entries the
controller copies in — is read-only), the one summary results/summary/r_lever.json, atomic writes (temporary file +
rename), RCache (e_store.ECache's key; smoke and real entries never share a root; the key and inputs stored in every
entry; an unreadable or foreign entry is missing), load_manifest (every raw file's sha256 and key re-checked) and the
archive copy (only under the archive root, never over an existing directory, each copy's sha256 checked)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from ..agent.e_store import ECache
from .h3_store import canonical, canonical_pretty, sha256_file
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_spec import SPEC as R_SPEC
from .r_spec import smoke

ALLOWED_DIR = "results/r/"
READ_ONLY_DIR = "results/r/ref/"
SUMMARY = "results/summary/r_lever.json"
SMOKE_SEEDS = frozenset(R_SPEC.smoke_seeds) | frozenset(smoke(R_SPEC).p.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if rel.startswith(READ_ONLY_DIR) or not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"R writes only under {ALLOWED_DIR} (not {READ_ONLY_DIR}) and {SUMMARY}, not {path}")


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


def move_block(path, src: str, dst: str, params_list) -> Path:
    """Gate ②'s one rerun (R.5): block src is kept as dst."""
    doc = read_summary(path)
    doc[dst] = doc.pop(src)
    return write_json(path, doc, params_list)


def _seeds(inputs) -> list:
    out = []
    for k, v in dict(inputs).items():
        if k == "seed":
            out.append(int(v))
        elif k == "seeds" or k.endswith("_seeds"):
            out += [int(s) for s in v]
    return out


class RCache(ECache):
    """A root with a path component "smoke" takes only smoke seeds, any other root takes none (SystemExit 2)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code)
        self.smoke_seeds = frozenset(int(s) for s in smoke_seeds)

    def _scope(self, inputs) -> None:
        seeds = _seeds(inputs)
        is_smoke_root = "smoke" in Path(self.root).parts
        some = any(s in self.smoke_seeds for s in seeds)
        if seeds and (some != is_smoke_root or some != all(s in self.smoke_seeds for s in seeds)):
            _refuse(f"{self.root}: smoke seeds belong only under a smoke root, real seeds only outside it")

    def _path(self, kind, inputs):
        self._scope(inputs)
        return super()._path(kind, inputs)

    def get(self, kind, inputs):
        p = self._path(kind, inputs)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        if d.get("key") != self.key(kind, inputs) or d.get("kind") != kind:
            return None
        return d["result"]

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


def load_manifest(manifest: list) -> tuple:
    """([dict(key, result, cache_key, cache_file)], [reasons]) — a missing file, a sha256 other than the manifest's or
    a stored key other than the manifest's cache_key is a reason, and that entry is not returned."""
    got, bad = [], []
    for m in manifest:
        p = Path(m["cache_file"])
        if not p.exists():
            bad.append(f"{m['key']}: raw file {p} is missing")
            continue
        if sha256_file(p) != m["sha256"]:
            bad.append(f"{m['key']}: raw file {p} sha256 differs from the manifest")
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            bad.append(f"{m['key']}: raw file {p} is unreadable")
            continue
        if d.get("key") != m["cache_key"]:
            bad.append(f"{m['key']}: raw file {p} holds another key")
            continue
        got.append(dict(key=m["key"], result=d["result"], cache_key=m["cache_key"], cache_file=str(p)))
    return got, bad


def archive_copy(files: list, dest, root) -> list:
    """R.9.7: one copy of every raw file under dest/<kind>/<basename>; dest must be a new directory under root."""
    root_r = Path(os.path.expanduser(str(root))).resolve()
    dest_p = Path(os.path.expanduser(str(dest)))
    dest_r = dest_p.resolve()
    if dest_r == root_r or root_r not in dest_r.parents:
        _refuse(f"the archive copy goes only under {root_r}, not {dest_r}")
    if dest_p.exists():
        _refuse(f"archive directory {dest_p} exists; an archive is never overwritten")
    out = []
    for f in files:
        src = Path(f)
        dst = dest_p / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sha = sha256_file(src)
        if sha256_file(dst) != sha:
            _refuse(f"archive copy of {src} does not match its source")
        out.append(dict(src=str(src), dst=str(dst), sha256=sha))
    return out
