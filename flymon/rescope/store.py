"""Where re-scope outputs may go, and the primary test's per-pair checkpoint (same shape as b_store.Checkpoint)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..brain.pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIRS = ("results/rescope/", "results/rescope-smoke/")
ALLOWED_SUMMARY_PREFIX = "results/summary/rescope_"


def guard(path, params_list) -> Path:
    p = Path(path)
    rel = os.path.relpath(os.path.abspath(str(p)), os.getcwd()).replace(os.sep, "/")
    ok = rel.startswith(ALLOWED_DIRS) or (rel.startswith(ALLOWED_SUMMARY_PREFIX) and rel.endswith(".json"))
    if not ok:
        raise SystemExit(f"refusing to write {rel}: re-scope outputs go under {ALLOWED_DIRS} or {ALLOWED_SUMMARY_PREFIX}*.json")
    for params in params_list:
        refuse_old_engine_output(str(p), params.kc_kc_scale)
        refuse_modified_engine_output(str(p), params)
    return p


def _atomic_write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, p)


def write_json(path, obj, params_list) -> Path:
    p = guard(path, params_list)
    _atomic_write(p, json.dumps(obj, sort_keys=True, indent=1, default=float).encode())
    return p


def git_provenance(files=()) -> dict:
    # porcelain lines are "XY path"; X may be a space, so the output must not be stripped before slicing
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
    status = run("status", "--porcelain", "--", *files) if files else run("status", "--porcelain")
    dirty = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(dirty), "dirty_files": dirty}


class Checkpoint:
    def __init__(self, directory, key: str, params_list):
        self.path = guard(Path(directory) / "checkpoint.npz", params_list)
        self.key = key

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        with np.load(self.path, allow_pickle=False) as z:
            prog = json.loads(str(z["progress"]))
            if prog["key"] != self.key:
                return None
            flies = [dict(enabled=e, shuffle_seed=None, w=z[f"w{i}"].copy()) for i, e in enumerate(prog["enabled"])]
        return dict(records=prog["records"], x=prog["x"], done=prog["done"], provenance=prog["provenance"],
                    weights={"flies": flies})

    def save(self, st: dict) -> None:
        flies = st["weights"]["flies"]
        prog = dict(key=self.key, records=st["records"], x=st["x"], done=st["done"], provenance=st.get("provenance"),
                    enabled=[bool(f["enabled"]) for f in flies])
        arrays = {f"w{i}": np.asarray(f["w"], dtype=np.float32) for i, f in enumerate(flies)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".npz")
        os.close(fd)
        np.savez(tmp, progress=np.array(json.dumps(prog)), **arrays)
        os.replace(tmp, self.path)
