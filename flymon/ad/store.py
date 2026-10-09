"""Where AD may write: the git-ignored work trees results/m4/ad/, results/m4-smoke/ad/, results/m4-bench/ad-<n>/ and the
tracked summaries results/summary/ad_*.json (atomic, sorted keys; AC's helpers). code_changes_since() lists the code
files whose content differs from a commit, committed or not (the run manifest's code check, AD.4)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..ac.store import _json_default, digest, rel, sha256_file  # noqa: F401  (re-exported)
from ..rescope.store import _atomic_write, git_provenance  # noqa: F401  (git_provenance is re-exported)

ALLOWED_DIRS = ("results/m4/ad/", "results/m4-smoke/ad/", "results/m4-bench/ad-")
SUMMARY_PREFIX = "results/summary/ad_"


def guard(path) -> Path:
    r = rel(path)
    if not (r.startswith(ALLOWED_DIRS) or (r.startswith(SUMMARY_PREFIX) and r.endswith(".json"))):
        raise SystemExit(f"refusing to write {r}: AD outputs go under {ALLOWED_DIRS} or {SUMMARY_PREFIX}*.json")
    return Path(path)


def write_json(path, obj) -> Path:
    p = guard(path)
    _atomic_write(p, json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, default=_json_default).encode())
    return p


def _git(*args) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def head_commit() -> str:
    return _git("rev-parse", "HEAD").strip()


def code_changes_since(commit: str, paths) -> list:
    paths = [str(p) for p in paths]
    changed = set(_git("diff", "--name-only", str(commit), "--", *paths).split())
    untracked = set(_git("ls-files", "--others", "--exclude-standard", "--", *paths).split())
    return sorted(changed | untracked)
