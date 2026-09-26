"""The helpers every CLI of spec appendix L shares (plan ruling P16: one copy instead of five): refusals, the --out rule,
the run id, the dirty check, the code keys and the manifest check (a block produced under other code is refused — plan
reading 16, Review Focus 4), the commit check of the summary, the J.12.9 provenance record and the summary-block write.
The scripts import these names, so their tests patch them on the script module."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

from . import h3_store, l_store
from .h3_store import ROOT, code_key, sha256_file
from .l_measure import HASHED_FILES, MEASURE_FILES

POOL_TIMEOUT_S = 1800


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def out_allowed(out) -> bool:
    """--out must lie under results/m0d/l/ of the repository root (normalised like l_store.guard)."""
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(l_store.ALLOWED_DIR)


def run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES). HASHED_FILES names every L script up front; a file
    not yet written is left out of the manifest's file map (so its later arrival changes the key)."""
    have = tuple(f for f in HASHED_FILES if (ROOT / f).exists())
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=have)


def same_code(block: dict, key: str, manifest: str) -> bool:
    """A previous block is usable only under this measure key and this procedure manifest."""
    return block.get("measure_key") == key and (block.get("code") or {}).get("key") == manifest


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _rel(path) -> str | None:
    rel = os.path.relpath(os.path.abspath(str(path)), ROOT).replace(os.sep, "/")
    return None if rel.startswith("../") else rel


def check_committed(summary, blocks) -> str | None:
    """None when the summary is git-tracked and unchanged against HEAD (stage2_oc.check_tracked's test), else the
    refusal reason (it says "tracked" or "commit")."""
    rel = _rel(summary)
    if rel is None or _git("ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{summary} is not tracked by git: commit its blocks {list(blocks)} before this stage"
    if _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{summary} has uncommitted changes: commit its blocks {list(blocks)} before this stage"
    return None


def head_sha256(path) -> str | None:
    """sha256 of the file's committed blob at HEAD (None when untracked)."""
    rel = _rel(path)
    if rel is None:
        return None
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def provenance(files, args: dict, argv) -> dict:
    """J.12.9's record: the inputs' sha256, HEAD, the dirty ones among them, the arguments."""
    sha = {(_rel(f) or str(f)): sha256_file(f) for f in files}
    rels = [r for r in (_rel(f) for f in files) if r]
    dirty = [ln[3:] for ln in _git("status", "--porcelain", "--", *rels).stdout.splitlines() if ln.strip()]
    return dict(sha256=sha, commit=_git("rev-parse", "HEAD").stdout.strip(), dirty=dirty, args=args, argv=list(argv))


def guard_params(base, measurers) -> list:
    return [base] + list(dict.fromkeys(p for m in measurers for p in m.params_seen if p != base))


def write_block(summary, name: str, res: dict, report, params_list) -> Path:
    """The summary block = the run record + its report path and sha256, through l_store's guarded write."""
    return l_store.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                       params_list)
