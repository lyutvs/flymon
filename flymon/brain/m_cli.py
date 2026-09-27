"""The helpers every CLI of spec appendix M shares (M.10, plan reading 16): l_cli's refusals, run id, manifest check,
commit gate, run-id chain and provenance, re-exported; M's --out rule (results/m0d/m/), code keys over M's files, block
order (spec_check first, M.10.7) and guarded summary write."""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import h3_store, m_store
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import (check_committed, guard_params, load_c3_record, other_code, provenance, refuse, run_id,  # noqa: F401
                    run_id_chain, same_code, stage0_inputs)
from .m_measure import HASHED_FILES, MEASURE_FILES

POOL_TIMEOUT_S = 3600                   # an edit job runs about 2/3 of an oracle job; teach jobs run whole
ORDER = ("spec_check", "stage0", "oc", "stage1", "stage2", "list", "stage3")


def out_allowed(out) -> bool:
    """--out must lie under results/m0d/m/ of the repository root (normalised like m_store.guard)."""
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(m_store.ALLOWED_DIR)


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES). A missing hashed file is a refusal (exit 2, naming
    it; SystemExit), never a smaller manifest."""
    missing = [f for f in HASHED_FILES if not (ROOT / f).exists()]
    if missing:
        refuse(f"the hashed files {missing} do not exist (the manifest covers every M module and script)")
        raise SystemExit(2)
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=HASHED_FILES)


# M's own copy: l_cli.read_previous names results/m0d/l/ in its smoke rule and message (L only).
def read_previous(summary, need, smoke: bool, committed) -> tuple:
    """(doc, None) or (None, refusal): outside --smoke the summary is git-tracked and clean at HEAD (`committed`); then
    every block of `need` present. --smoke reads only a summary under results/m0d/m/."""
    if smoke:
        if not out_allowed(summary):
            return None, f"a --smoke run reads and writes only a smoke summary under {m_store.ALLOWED_DIR}, not {summary}"
    elif need:
        why = committed(summary, list(need))
        if why:
            return None, why
    try:
        doc = json.loads(Path(summary).read_text()) if need or Path(summary).exists() else {}
    except (OSError, ValueError) as e:
        return None, f"no usable summary at {summary}: {e}"
    missing = [b for b in need if not isinstance(doc.get(b) if isinstance(doc, dict) else None, dict)]
    if missing:
        return None, f"{summary} lacks the blocks {missing} (each earlier stage's block comes first)"
    return doc, None


def later_blocks(summary, name: str, doc: dict | None = None) -> str | None:
    """The refusal when the summary already holds a block of a stage after `name` in ORDER, or None."""
    if doc is None:
        try:
            doc = json.loads(Path(summary).read_text())
        except OSError:
            return None
        except ValueError as e:
            return f"no usable summary at {summary}: {e}"
    later = [b for b in ORDER[ORDER.index(name) + 1:] if isinstance(doc, dict) and b in doc]
    return None if not later else (f"{summary} already holds the later blocks {later}: block {name} is not rewritten "
                                   f"under them (start a new summary, or --smoke)")


# M's own copy: l_cli.write_block writes through l_store (results/m0d/l/, l_screen.json), which refuses M's paths.
def write_block(summary, name: str, res: dict, report, params_list) -> Path:
    return m_store.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                       params_list)
