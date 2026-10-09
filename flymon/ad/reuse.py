"""Spec AD.2 (가) / AD.5 6: what AD reuses from stage 1 must be stage 1's, else STOP_REUSE - the FROZEN model manifest
(its sha256 equal to the one stage 1's BRAIN result recorded), the confirmation-set digest 71a10cbc..., the
evaluation-schedule digest 03e209ea... (both recomputed and as stage 1 recorded them), and the six C-off point-0
situation records (global flies 12-17 of results/m4/stage1/BRAIN/logs/situations_init.jsonl: the file's sha256, the
records' content hash, frozen, AC.5 label). COMPARED fields are compared again at every run / resume and by the
analysis (a later change is STOP_REUSE)."""
from __future__ import annotations

import json
from pathlib import Path

from ..ac.manifest import MODEL_MANIFEST
from ..rescope.blocks import read_jsonl
from .spec import LABEL, SPEC
from .store import digest, sha256_file

CONFIRM_DIGEST = "71a10cbcabc67d551d9db9834c3436a10ce45e4172d44aba2c1b5c829bfec038"
EVAL_DIGEST = "03e209ea65d38d461ba58befe3b893a9f54d301edd58cf4d5abb6518a91a8dfa"
COMPARED = ("model_sha256", "confirm_digest", "eval_digest", "coff_file_sha256", "coff_content_hash")


def stage1_paths(spec=SPEC) -> dict:
    root = Path(spec.stage1_root) / "BRAIN"
    return dict(result=root / "result.json", init=root / "logs" / "situations_init.jsonl",
                points=root / "logs" / "situations.jsonl")


def coff_records(init_path, flies) -> list:
    flies = [int(f) for f in flies]
    return sorted((r for r in read_jsonl(init_path) if r.get("fly") in flies and r.get("point") == 0),
                  key=lambda r: r["fly"])


def stage1_fly_records(points_path, flies, point: int) -> list:
    flies = [int(f) for f in flies]
    return sorted((r for r in read_jsonl(points_path) if r.get("fly") in flies and r.get("point") == int(point)),
                  key=lambda r: r["fly"])


def reuse_fields(confirm_digest, eval_digest, spec=SPEC, model_path=MODEL_MANIFEST) -> dict:
    p = stage1_paths(spec)
    for k in ("result", "init"):
        if not p[k].exists():
            raise SystemExit(f"STOP_REUSE: {p[k]} does not exist (AD reuses stage 1's records, AD.1)")
    res = json.loads(p["result"].read_text())
    model = json.loads(Path(model_path).read_text())
    co = coff_records(p["init"], spec.coff_flies)
    return dict(model_path=str(model_path), model_status=model.get("status"), model_sha256=sha256_file(model_path),
                stage1_model_sha256=(res.get("model") or {}).get("sha256"),
                confirm_digest=confirm_digest, stage1_confirm_digest=res.get("confirm_digest"),
                eval_digest=eval_digest, stage1_eval_digest=res.get("eval_digest"),
                coff_path=str(p["init"]), coff_file_sha256=sha256_file(p["init"]), coff_content_hash=digest(co),
                coff_flies=[r["fly"] for r in co], coff_arm=sorted({str(r.get("arm")) for r in co}),
                coff_frozen=bool(co) and all(r.get("frozen") is True for r in co),
                coff_label=bool(co) and all(r.get("label") == LABEL for r in co))


def reuse_problems(f: dict, spec=SPEC) -> list:
    bad = []
    if f["model_status"] != "FROZEN":
        bad.append("model_status")
    if f["model_sha256"] != f["stage1_model_sha256"]:
        bad.append("model_sha256")
    if f["confirm_digest"] != CONFIRM_DIGEST or f["stage1_confirm_digest"] != CONFIRM_DIGEST:
        bad.append("confirm_digest")
    if f["eval_digest"] != EVAL_DIGEST or f["stage1_eval_digest"] != EVAL_DIGEST:
        bad.append("eval_digest")
    if f["coff_flies"] != list(spec.coff_flies) or f["coff_arm"] != ["COFF"]:
        bad.append("coff_records")
    if not f["coff_frozen"]:
        bad.append("coff_frozen")
    if not f["coff_label"]:
        bad.append("coff_label")
    return bad


def changed_since(old: dict, new: dict) -> list:
    return [k for k in COMPARED if (old or {}).get(k) != (new or {}).get(k)]
