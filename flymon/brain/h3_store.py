"""Resume keys, guarded writes and the measurement cache of the M0d H.3 runner (spec appendix H.3a.11).

- **Resume key** (a measurement's cache key): sha256 over the content of the files a measurement's result depends on
  (`MEASURE_FILES`: the engine and measurement modules, the H.3 jobs and cache plumbing, uv.lock) and the connectome
  NPZ, the Python and NumPy versions, the measurement's kind and its canonical inputs (the full `Params`, which
  carries a C3 threshold file's sha256, and the odours or seeds). The procedure modules (rules, runner, C3, records,
  the CLI) decide what to measure, not what a measurement returns, so editing them keeps every cached measurement.
- **Dirty check and manifest**: `HASHED_FILES` = the measurement files plus the procedure modules and the CLI; a
  real run refuses when any of them is dirty. Documentation changes are recorded in the manifest and never block.
- **Guarded writes**: every output path goes through `guard`, which calls BOTH `refuse_old_engine_output` and
  `refuse_modified_engine_output` for every `Params` that produced the content (spec H.2) and additionally refuses
  anything outside results/m0d/ and results/summary/m0d.json — C0 is the default engine, for which neither
  older guard fires on results/m0c/. Writes go to a temporary file in the same directory and are renamed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ROOT = Path(__file__).resolve().parents[2]
MEASURE_FILES = ("uv.lock", "flymon/brain/circuits.py", "flymon/brain/config.py", "flymon/brain/connectome.py",
                 "flymon/brain/engine_cpu.py", "flymon/brain/fly_pool.py", "flymon/brain/measure.py",
                 "flymon/brain/stimuli.py", "flymon/brain/thresholds.py", "flymon/brain/h3_jobs.py",
                 "flymon/brain/h3_measure.py", "flymon/brain/h3_store.py")
HASHED_FILES = MEASURE_FILES + ("scripts/run_m0d_h3.py", "flymon/brain/pool_bench.py", "flymon/brain/h3_spec.py",
                                "flymon/brain/h3_rules.py", "flymon/brain/h3_runner.py", "flymon/brain/h3_c3.py",
                                "flymon/brain/h3_records.py")
ALLOWED_DIR = "results/m0d/"
ALLOWED_FILES = ("results/summary/m0d.json", "results/summary/m0d_h3_c3_thresholds.npz")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj) -> str:
    def conv(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return conv(dataclasses.asdict(o))
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    return json.dumps(conv(obj), sort_keys=True, separators=(",", ":"))


def code_key(npz, root: Path = ROOT, files=MEASURE_FILES) -> dict:
    hashed = {f: sha256_file(root / f) for f in files}
    hashed["npz:" + Path(npz).name] = sha256_file(npz)
    versions = dict(python=platform.python_version(), numpy=np.__version__)
    key = hashlib.sha256(canonical(dict(files=hashed, versions=versions)).encode()).hexdigest()
    return dict(key=key, files=hashed, versions=versions)


def git_state(root: Path = ROOT, files=HASHED_FILES) -> dict:
    """HEAD, the dirty hashed files (these block a real run) and every other dirty path (recorded only)."""
    def run(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout
    head = run("rev-parse", "HEAD").strip()
    dirty = [ln[3:] for ln in run("status", "--porcelain", "--", *files).splitlines() if ln.strip()]
    other = [ln[3:] for ln in run("status", "--porcelain").splitlines() if ln.strip() and ln[3:] not in dirty]
    return dict(commit=head, dirty_hashed=dirty, dirty_other=other)


def guard(path, params_list) -> None:
    """Both engine write guards for every Params, and the M0d output rule (SystemExit 2 on refusal)."""
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel in ALLOWED_FILES):
        print(f"refusing to write {path}: the H.3 runner writes only under {ALLOWED_DIR} and {ALLOWED_FILES}",
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


def canonical_pretty(obj) -> str:
    return json.dumps(json.loads(canonical(obj)), indent=1)


class MeasureCache:
    """Content-addressed results of measurements: <root>/<kind>/<key[:24]>.json holding {key, kind, inputs, result}.
    A file whose stored key differs from the computed one is never read (it is recomputed and replaced)."""

    def __init__(self, root, code: dict, run_id: str):
        self.root, self.code, self.run_id = Path(root), code, run_id
        self.hits = self.misses = 0
        self.used: dict = {}                 # path -> key of every entry this run read or wrote (the manifest)

    def key(self, kind: str, inputs: dict) -> str:
        return hashlib.sha256((self.code["key"] + "|" + kind + "|" + canonical(inputs)).encode()).hexdigest()

    def get_or_compute(self, kind: str, inputs: dict, compute, params_list):
        k = self.key(kind, inputs)
        path = self.root / kind / f"{k[:24]}.json"
        self.used[str(path)] = k
        if path.exists():
            try:
                d = json.loads(path.read_text())
                if d.get("key") == k:
                    self.hits += 1
                    return d["result"]
            except (OSError, ValueError):
                pass
        result = compute()
        write_json(path, dict(key=k, kind=kind, run_id=self.run_id, inputs=json.loads(canonical(inputs)),
                              result=result), params_list)
        self.misses += 1
        return json.loads(canonical(result))


def replace_summary_block(path, name: str, block: dict, params_list) -> Path:
    """results/summary/m0d.json holds one block per M0d step; replace this one, keep the others, write once."""
    path = Path(path)
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur[name] = block
    return write_json(path, cur, params_list)
