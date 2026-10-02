# flymon/brain/q_store.py
"""Q's only writer (Q.5, Q.6.8): raw files under results/q/ (results/q/q0_cache/ is read-only), the one summary
results/summary/q_reward.json, atomic writes (temporary file + rename), and QCache — e_store.ECache's key with this
guard, the key and inputs stored in every entry, and an unreadable or foreign entry treated as missing."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from ..agent.e_store import ECache
from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .q_spec import SPEC as Q_SPEC

ALLOWED_DIR = "results/q/"
READ_ONLY_DIR = "results/q/q0_cache/"
SUMMARY = "results/summary/q_reward.json"


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if rel.startswith(READ_ONLY_DIR) or not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: Q writes only under {ALLOWED_DIR} (not {READ_ONLY_DIR}) and {SUMMARY}",
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
    doc = read_summary(path)
    doc[block] = obj
    return write_json(path, doc, params_list)


SMOKE_SEEDS = frozenset(Q_SPEC.smoke_seeds)       # q_spec.SPEC.smoke_seeds (Q.6.8), read from q_spec, never restated


def _seeds(inputs) -> list:
    return [int(s) for k, v in dict(inputs).items() if k == "seeds" or k.endswith("_seeds") for s in v]


class QCache(ECache):
    """Smoke (seeds 24_008_xxx) and real entries never share a root: a root with a path component "smoke" takes only
    smoke seeds, any other root takes none (SystemExit 2), so a smoke entry cannot be read as a real one."""

    def _scope(self, inputs) -> None:
        seeds = _seeds(inputs)
        is_smoke_root = "smoke" in Path(self.root).parts
        if seeds and any(s in SMOKE_SEEDS for s in seeds) != is_smoke_root or any(
                s in SMOKE_SEEDS for s in seeds) != all(s in SMOKE_SEEDS for s in seeds):
            print(f"refusing {self.root}: smoke seeds belong only under a smoke root, real seeds only outside it",
                  file=sys.stderr)
            raise SystemExit(2)

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
