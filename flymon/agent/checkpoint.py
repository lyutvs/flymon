"""Per-battle checkpoints (spec 3.6): cohort weights (float32, exact), wiring/enabled flags, the schedule cursor and
the config hash, written to a temp file and renamed; two generations kept; manifest.json (schema version, and per
generation the state file, its sha256 and the completed battle ids) replaced atomically last. Resume takes the newest
generation whose checksum holds, else the one before, and the caller drops log records of uncommitted battles.
The engine RNG needs no checkpoint: every presentation reseeds from a seed derived from (fly, battle, turn).
Crash points: `fault` ("state" | "before_manifest") is a test hook; the store writes no logs, so the log-write crash point
is handled by filter_log on resume (test_resume_drops_uncommitted_log_records).
After a fallback load, the next commit builds on the generation actually loaded and drops the newer, unusable ones."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
FAULTS = (None, "state", "before_manifest")


class CrashForTest(Exception):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class CheckpointStore:
    def __init__(self, root, config_hash: str):
        self.root, self.config_hash = Path(root), config_hash
        self.root.mkdir(parents=True, exist_ok=True)
        self.fault: str | None = None   # test hook, one of FAULTS: raise CrashForTest at that point of commit()
        self._base: int | None = None   # generation returned by the last load() / commit(); commit() builds on it

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        m = json.loads(self.manifest_path.read_text())
        if m["schema"] != SCHEMA_VERSION:
            raise ValueError(f"manifest schema {m['schema']} != {SCHEMA_VERSION}")
        if m["config_hash"] != self.config_hash:
            raise ValueError(f"config hash {m['config_hash'][:12]} != this run's {self.config_hash[:12]}")
        return m

    def commit(self, battle_id: str, pool_state: dict, cursor: dict) -> int:
        if self.fault not in FAULTS:
            raise ValueError(f"unknown fault {self.fault!r}; expected one of {FAULTS}")
        m = self._manifest() or {"schema": SCHEMA_VERSION, "config_hash": self.config_hash, "generations": []}
        gen = max((g["generation"] for g in m["generations"]), default=0) + 1   # never reuse a file name
        if self._base is not None:   # drop generations newer than the one loaded (they failed their checksum)
            m["generations"] = [g for g in m["generations"] if g["generation"] <= self._base]
        prev = m["generations"][-1] if m["generations"] else {"generation": 0, "completed": []}
        buf = io.BytesIO()
        flies = pool_state["flies"]
        np.savez(buf, meta=np.array(json.dumps({"cursor": cursor, "flies": [
                     {"enabled": f["enabled"], "shuffle_seed": f["shuffle_seed"]} for f in flies]})),
                 **{f"w{i}": np.asarray(f["w"], np.float32) for i, f in enumerate(flies)})
        data = buf.getvalue()
        name = f"state_{gen:06d}.npz"
        if self.fault == "state":
            (self.root / f".{name}.tmp").write_bytes(data[: len(data) // 2])
            raise CrashForTest("state")
        _atomic_write(self.root / name, data)
        entry = {"generation": gen, "file": name, "sha256": _sha(data), "completed": prev["completed"] + [battle_id]}
        m["generations"] = (m["generations"] + [entry])[-2:]
        if self.fault == "before_manifest":
            raise CrashForTest("before_manifest")
        _atomic_write(self.manifest_path, json.dumps(m, indent=1).encode())
        keep = {g["file"] for g in m["generations"]}
        for p in self.root.glob("state_*.npz"):
            if p.name not in keep:
                p.unlink()
        self._base = gen
        return gen

    def load(self) -> dict | None:
        m = self._manifest()
        if m is None:
            return None
        for entry in reversed(m["generations"]):
            p = self.root / entry["file"]
            if not p.exists() or _sha(p.read_bytes()) != entry["sha256"]:
                continue
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            self._base = entry["generation"]
            flies = [dict(f, w=z[f"w{i}"]) for i, f in enumerate(meta["flies"])]
            return {"generation": entry["generation"], "pool_state": {"flies": flies}, "cursor": meta["cursor"],
                    "completed": list(entry["completed"])}
        raise ValueError("no checkpoint generation passes its checksum")


def filter_log(path, completed: set) -> int:
    """Keep only records of committed battles (and drop a torn last line); returns the number of lines dropped."""
    path = Path(path)
    if not path.exists():
        return 0
    keep, dropped = [], 0
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if rec.get("battle_id") in completed:
            keep.append(line)
        else:
            dropped += 1
    _atomic_write(path, ("".join(x + "\n" for x in keep)).encode())
    return dropped
