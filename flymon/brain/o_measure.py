"""Storage and measurement of spec appendix O (O.5, O.7.5).
- The write guard: raw rows, caches and reports under results/o/ (git-ignored by `results/*`) and the summary
  results/summary/o_states.json; nothing else (SystemExit 2). Every write is atomic (tmp + os.replace). n_store's rule
  with O's paths: n_store itself is hard-wired to results/n/.
- OCache: h3_store.MeasureCache's content-addressed entries, one file per item, written through O's guard, each also
  naming the code key and the spec commit (O.7.5).
- OMeasurer: n_measure.NMeasurer's rounds of one item per worker (its _items and _window, unchanged), over O's jobs:
  O1 one presentation per (condition, cell, seed), O2 one arm per (X, arm, seed). A rerun reads every finished item
  without starting a job; an interrupted run loses at most one round."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from . import n_measure, o_jobs
from .h3_store import MeasureCache, canonical, canonical_pretty
from .n_measure import NMeasurer
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/o/"
SUMMARY = "results/summary/o_states.json"
# built from N's lists (D2): every N entry stays, plus the O code files (O.5); nothing re-listed by hand
MEASURE_FILES = tuple(dict.fromkeys(n_measure.MEASURE_FILES + ("flymon/brain/o_jobs.py", "flymon/brain/o_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(n_measure.HASHED_FILES + MEASURE_FILES + (
    "flymon/brain/o_spec.py", "flymon/brain/o_rules.py", "flymon/brain/o_cli.py", "scripts/run_o1.py",
    "scripts/run_o2.py")))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: spec O writes only under {ALLOWED_DIR} and {SUMMARY}", file=sys.stderr)
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


class OCache(MeasureCache):
    """MeasureCache with O's guarded write; every entry also holds the code key and the spec commit (O.7.5)."""

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


class OMeasurer(NMeasurer):
    """NMeasurer's item rounds over O's jobs; `spec` is an OSpec (N's windows and timings through spec.n)."""

    def __init__(self, pool, spec, cache):
        super().__init__(pool, spec.n, cache)
        self.ospec = spec

    def o1_presentations(self, params, items, readout) -> list:
        """items [{"cond", "edit", "g", "stim", "seed", "odor"}] -> one presentation row per item, with cond / g / stim;
        the key is the unit (O1, condition, cell, seed) plus everything the job reads."""
        common = dict(params=params, readout=dict(readout), **self._window())
        jobs = [dict(common, edit=i["edit"], odor=dict(i["odor"]), seeds=(int(i["seed"]),)) for i in items]
        keys = [dict(j, stage="o1", cond=i["cond"], g=float(i["g"]), stim=i["stim"]) for j, i in zip(jobs, items)]
        out = self._items("o1_pres", o_jobs.presentation_job, params, jobs, keys)
        return [dict(rows[0], cond=i["cond"], g=float(i["g"]), stim=i["stim"]) for rows, i in zip(out, items)]

    def o2_arms(self, params, items, readout, punish_type) -> list:
        """items [{"x", "y", "edit", "arm", "punish", "plastic", "da_zero", "odor_x", "odor_y", "seed"}] -> arm_job rows
        with x / y; N2's training timings and seed rule (spec.n)."""
        s, h4 = self.spec, self.spec.h4
        common = dict(params=params, readout=dict(readout), punish_type=punish_type, reward_type=s.h3.reward_type,
                      trials=int(h4.teach_trials),
                      present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms,
                      train_settle_ms=h4.teach_window.settle_ms, seed_base=s.train_seed_base,
                      seed_stride=s.train_seed_stride, **self._window())
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     arm=i["arm"], punish=bool(i["punish"]), plastic=bool(i["plastic"]),
                     da_zero=bool(i["da_zero"])) for i in items]
        keys = [dict(j, stage="o2", x=i["x"], y=i["y"]) for j, i in zip(jobs, items)]
        rows = self._items("o2_arm", o_jobs.arm_job, params, jobs, keys)
        return [dict(r, x=i["x"], y=i["y"]) for r, i in zip(rows, items)]
