"""Guarded writes and the per-step checkpoint of spec J.12's B runs.

Raw records and checkpoints go under results/b/ (results/b-smoke/ for smoke runs, both git-excluded); the two summaries
are results/summary/b_calibration.json and results/summary/b_test.json. Nothing else is written. Every write calls both
engine write guards (the engine is C0 = Params(), for which neither fires on these paths) and is atomic.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from .h3_store import canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIRS = ("results/b/", "results/b-smoke/")
ALLOWED_FILES = ("results/summary/b_calibration.json", "results/summary/b_test.json")


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIRS) or rel in ALLOWED_FILES):
        print(f"refusing to write {path}: spec J.12 writes only under {ALLOWED_DIRS} and {ALLOWED_FILES}",
              file=sys.stderr)
        raise SystemExit(2)


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(obj) + "\n").encode(), params_list)


class Checkpoint:
    """<dir>/checkpoint.npz holding the progress (records, X, steps done, the code key; JSON) and the pool's weights, in
    ONE atomically replaced file: a step is either wholly recorded with its weights or not at all, so a resume never
    applies a training trial twice. A checkpoint written under another code key is ignored (the run starts over). The
    progress also keeps the provenance (git state, start time) of the run that measured its first step: a resumed or
    replayed run reports the run that measured, not itself."""

    def __init__(self, directory, key: str, params_list):
        self.path, self.key, self.params = Path(directory) / "checkpoint.npz", key, params_list

    def load(self):
        if not self.path.exists():
            return None
        with np.load(self.path) as z:
            d = json.loads(bytes(z["progress"]).decode())
            if d.get("key") != self.key:
                return None
            flies = [dict(enabled=bool(e), shuffle_seed=None, w=z[f"w{i}"].copy()) for i, e in enumerate(d["enabled"])]
        return dict(records=d["records"], x=d["x"], done=d["done"], provenance=d.get("provenance"),
                    weights={"flies": flies})

    def save(self, st: dict) -> None:
        import io
        flies = st["weights"]["flies"]
        prog = json.dumps(dict(key=self.key, records=st["records"], x=st["x"], done=st["done"],
                               provenance=st.get("provenance"), enabled=[bool(f["enabled"]) for f in flies])).encode()
        buf = io.BytesIO()
        np.savez(buf, progress=np.frombuffer(prog, np.uint8),
                 **{f"w{i}": np.asarray(f["w"], np.float32) for i, f in enumerate(flies)})
        write_bytes(self.path, buf.getvalue(), self.params)
