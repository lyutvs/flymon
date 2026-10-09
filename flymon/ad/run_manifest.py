"""The run manifest (spec AD.4): committed after the re-bench and the extrapolation, before the run. It fixes T, the
point list, the new-set flag (and its points), the digest of the learning-schedule prefix actually used, the sha256 of
ad_budget.json and of the AD input manifest, and the code commit. Every run and resume verifies it (check(); a
mismatch is STOP_MANIFEST) and its sha256 is part of the checkpoint key, so a checkpoint written under another manifest
cannot be resumed (run_ad_arm raises STOP_MANIFEST). The code check lists every file under CODE_PATHS that differs from
the manifest's commit, committed later or not. STOP_FUTILE and a budget stop do not change the manifest; they go in
the result. The smoke writes its own manifest (no budget) under results/m4-smoke/ad/."""
from __future__ import annotations

import json
from pathlib import Path

from ..rescope import blocks
from . import inputs
from .spec import LABEL
from .store import code_changes_since, digest, sha256_file

RUN_MANIFEST = "results/summary/ad_run_manifest.json"
CODE_PATHS = ("flymon", "scripts/run_ad.py")


def manifest_path(spec) -> Path:
    return Path(RUN_MANIFEST) if spec.mode == "run" else Path(spec.out_root) / "run_manifest.json"


def _prefix_digest(spec, T: int) -> str:
    return blocks.schedule_digest(inputs.prefix(inputs.learn_canonical(spec), T))


def build(spec, *, T: int, new_set: bool, budget_path, inputs_manifest_path, commit: str) -> dict:
    return dict(status="OK", mode=spec.mode, T=int(T), new_set=bool(new_set), points=list(spec.points(T)),
                new_points=list(spec.new_points(T)) if new_set else [], futility_point=spec.futility_point,
                learn_prefix_digest=_prefix_digest(spec, T),
                budget_path=None if budget_path is None else str(budget_path),
                budget_sha256=None if budget_path is None else sha256_file(budget_path),
                inputs_manifest_path=str(inputs_manifest_path), inputs_manifest_sha256=sha256_file(inputs_manifest_path),
                code_commit=str(commit), code_paths=list(CODE_PATHS), label=LABEL)


def check(man: dict, spec, *, check_code: bool = True) -> list:
    bad = []
    T = int(man.get("T") or -1)
    if man.get("status") != "OK" or man.get("mode") != spec.mode:
        bad.append("status")
    if T not in (spec.learn_battles, spec.fallback_battles):
        bad.append("T")
    if man.get("points") != list(spec.points(T)):
        bad.append("points")
    if man.get("new_points") != (list(spec.new_points(T)) if man.get("new_set") else []):
        bad.append("new_points")
    if man.get("learn_prefix_digest") != _prefix_digest(spec, max(T, 0)):
        bad.append("learn_prefix_digest")
    if man.get("budget_path") is not None:
        bp = Path(man["budget_path"])
        if not bp.exists() or sha256_file(bp) != man.get("budget_sha256"):
            bad.append("budget_sha256")
        b = json.loads(bp.read_text()) if bp.exists() else {}
        if b.get("status") != "OK" or b.get("T") != man.get("T") or b.get("new_set") != man.get("new_set"):
            bad.append("budget_decision")
    ip = Path(man.get("inputs_manifest_path") or "")
    if not ip.is_file() or sha256_file(ip) != man.get("inputs_manifest_sha256"):
        bad.append("inputs_manifest_sha256")
    if check_code:
        changed = code_changes_since(man["code_commit"], man.get("code_paths") or CODE_PATHS)
        if changed:
            bad.append("code_changed: " + ", ".join(changed))
    return bad


def checkpoint_key(*, cfg_hash: str, layout, input_digests: dict, mode: str, lever_sha: str,
                   run_manifest_sha256: str) -> str:
    return digest(dict(cfg=cfg_hash, layout=[list(x) for x in layout], inputs=input_digests, mode=mode,
                       lever=lever_sha, run_manifest=run_manifest_sha256))
