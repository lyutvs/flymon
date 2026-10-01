"""Storage and measurement of spec appendix P (P.4, P.6.5).
- The write guard: raw rows, caches and reports under results/p/ (git-ignored by `results/*`) and the summary
  results/summary/p_learning.json; nothing else (SystemExit 2). Every write is atomic (tmp + os.replace).
  o_measure's rule with P's paths: o_measure's guard reads its own module constants (results/o/), so P keeps its own
  copy (reading 1) — no O module is edited or patched.
- PCache: o_measure.OCache's entries (one file per item, the code key and the spec commit in each) through P's guard.
- PMeasurer: o_measure.OMeasurer's rounds of one item per worker (NMeasurer._items / _window unchanged) over
  o_jobs.arm_job, one item per (direction, arm, seed); each row also carries its direction, X, Y and operating point.
  A rerun reads every finished item without starting a job; an interrupted run loses at most one round."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from . import o_jobs, o_measure
from .h3_store import MeasureCache, canonical, canonical_pretty
from .o_measure import OMeasurer
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/p/"
SUMMARY = "results/summary/p_learning.json"
# built from O's lists: every O entry stays, plus the P measurement file; nothing re-listed by hand
MEASURE_FILES = tuple(dict.fromkeys(o_measure.MEASURE_FILES + ("flymon/brain/p_measure.py",)))
HASHED_FILES = tuple(dict.fromkeys(o_measure.HASHED_FILES + MEASURE_FILES + (
    "flymon/brain/p_spec.py", "flymon/brain/p_rules.py", "flymon/brain/p_cli.py", "scripts/run_p.py",
    "scripts/p_oc.py")))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: spec P writes only under {ALLOWED_DIR} and {SUMMARY}", file=sys.stderr)
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


def write_summary_block(path, block: str, obj, params_list) -> Path:
    guard(path, params_list)
    doc = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    doc[block] = json.loads(canonical_pretty(obj))
    return write_json(path, doc, params_list)


class PCache(MeasureCache):
    """MeasureCache with P's guarded write; every entry also holds the code key and the spec commit (O.7.5, P.6.5).
    Copied from o_measure:OCache (its write goes through o_measure's guard)."""

    def __init__(self, root, code: dict, run_id: str, spec_commit: str | None):
        super().__init__(root, code, run_id)
        self.spec_commit = spec_commit

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
        write_json(path, dict(key=k, kind=kind, run_id=self.run_id, code_key=self.code["key"],
                              spec_commit=self.spec_commit, inputs=json.loads(canonical(inputs)), result=result),
                   params_list)
        self.misses += 1
        return json.loads(canonical(result))


class PMeasurer(OMeasurer):
    """OMeasurer's item rounds over P's arms; `spec` is a PSpec (O's numbers through spec.o, N's through spec.o.n)."""

    def __init__(self, pool, spec, cache):
        super().__init__(pool, spec.o, cache)
        self.pspec = spec

    def p_arms(self, params, items, readout, punish_type) -> list:
        """items [{"direction", "x", "y", "edit", "arm", "punish", "plastic", "da_zero", "odor_x", "odor_y", "seed",
        "point"}] -> arm_job rows with direction / x / y / point; N2's training timings and seed rule (spec.o.n).
        The key is the unit (P, direction, arm, seed) plus everything the job reads; the point is part of it."""
        s, h4 = self.spec, self.spec.h4
        common = dict(params=params, readout=dict(readout), punish_type=punish_type, reward_type=s.h3.reward_type,
                      trials=int(h4.teach_trials), present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms,
                      train_settle_ms=h4.teach_window.settle_ms, seed_base=s.train_seed_base,
                      seed_stride=s.train_seed_stride, **self._window())
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     arm=i["arm"], punish=bool(i["punish"]), plastic=bool(i["plastic"]),
                     da_zero=bool(i["da_zero"])) for i in items]
        keys = [dict(j, stage="p", direction=i["direction"], x=i["x"], y=i["y"],
                     point=[float(v) for v in i["point"]]) for j, i in zip(jobs, items)]
        rows = self._items("p_arm", o_jobs.arm_job, params, jobs, keys)
        return [dict(r, direction=i["direction"], x=i["x"], y=i["y"], point=[float(v) for v in i["point"]])
                for r, i in zip(rows, items)]
