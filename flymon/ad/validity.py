"""Point-wise validity, the futility check and the final reading point (spec AD.2, AD.4, AD.7).
- A fly enters the reading at point t iff its point-t situation record exists and passed the frozen check: the record
  is written only after learning battles 0 .. t-1 all finished (Task 7), so an INVALID fly keeps its earlier points.
- Futility check (AD.2, the only interim analysis): once every fly has its point-40 evaluation, FLY-L's right-direction
  share (per fly the share of pairs with diag.best_contrast > 0, averaged over the valid FLY-L flies at that point) at
  40 is compared with point 0; 40 <= 0 is STOP_FUTILE (learning stops at 40, every reading at 40). It reads only these
  two numbers. The decision is written once (futility.json); a resume recomputes it and must get the same.
- Final reading point: 40 under STOP_FUTILE; under STOP_BUDGET in the run, the last point every still-valid fly reached
  (the latest common point); else T ("complete" for 120, "fallback" for 80). A fly in a stopped arm (the learning
  book's `stopped`) is left out of the latest common point even when it carries no invalid mark: hooks mark it
  "arm_stopped" only when its skipped battle comes back, which a STOP_BUDGET session may never reach."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..ac import diag
from .spec import LABEL
from .store import write_json


def records(sits, *, point: int, arm=None, set_: str = "confirm", mode: str = "own") -> dict:
    out: dict = {}
    for r in sits:
        if r.get("set") != set_ or r.get("mode") != mode or r.get("point") != int(point) or r.get("frozen") is not True:
            continue
        if arm is not None and r.get("arm") != arm:
            continue
        if r["fly"] in out:
            raise ValueError(f"two {set_}/{mode} records of fly {r['fly']} at point {point}")
        out[r["fly"]] = r
    return out


def right_direction(rec: dict, pairs) -> float:
    return float(np.mean([d > 0 for d in diag.best_contrast(rec, pairs)]))


def share(recs: dict, pairs):
    return float(np.mean([right_direction(r, pairs) for r in recs.values()])) if recs else None


def futility(sits, pairs, spec) -> dict:
    fp = spec.futility_point
    r0, rc = records(sits, point=0, arm="FLYL"), records(sits, point=fp, arm="FLYL")
    b, a = share(r0, pairs), share(rc, pairs)
    status = "NO_FLYL" if a is None or b is None else ("STOP_FUTILE" if a <= b else "CONTINUE")
    return dict(status=status, point=fp, share_at_point=a, share_at_0=b, n_at_point=len(rc), n_at_0=len(r0),
                label=LABEL)


def futility_line(fut: dict) -> str:
    f = lambda x: "none" if x is None else f"{x:.3f}"
    return (f"futility check at {fut['point']}: FLY-L right-direction share {f(fut['share_at_point'])} vs point 0 "
            f"{f(fut['share_at_0'])} -> {fut['status']}")


def persist_futility(path, fut: dict) -> dict:
    p = Path(path)
    if p.exists():
        old = json.loads(p.read_text())
        keys = ("status", "share_at_point", "share_at_0", "n_at_point", "n_at_0")
        if any(old.get(k) != fut.get(k) for k in keys):
            raise SystemExit(f"STOP_INFRA: the futility check changed across a resume ({old} -> {fut})")
        return old
    write_json(p, fut)
    return fut


def reached(sits) -> dict:
    out: dict = {}
    for r in sits:
        if r.get("set") == "confirm" and r.get("mode") == "own" and r.get("frozen") is True:
            out[r["fly"]] = max(out.get(r["fly"], -1), int(r["point"]))
    return out


def latest_common_point(sits, lay, invalid: dict, stopped=None):
    rp, bad, halted = reached(sits), {int(k) for k in invalid}, set(stopped or ())
    alive = [g for i, (arm, _, g) in enumerate(lay) if i not in bad and arm not in halted]
    pts = [rp.get(g) for g in alive]
    if not pts or any(p is None for p in pts):
        return None
    return min(pts)


def final_point(spec, T: int, futility_status, run_status, sits, lay, invalid: dict, stopped=None) -> dict:
    if futility_status == "STOP_FUTILE":
        return dict(point=int(spec.futility_point), reason="STOP_FUTILE")
    if run_status == "STOP_BUDGET":
        return dict(point=latest_common_point(sits, lay, invalid, stopped), reason="STOP_BUDGET")
    return dict(point=int(T), reason="fallback" if int(T) == spec.fallback_battles else "complete")


def dropouts(invalid: dict, lay, sits) -> list:
    rp = reached(sits)
    out = []
    for i, reason in sorted((int(k), v) for k, v in invalid.items()):
        arm, _, g = lay[i]
        out.append(dict(fly=g, arm=arm, reason=reason, last_point=rp.get(g)))
    return out


def n_valid_by_point(sits, lay, points, set_: str = "confirm", mode: str = "own") -> dict:
    arms = list(dict.fromkeys(a for a, _, _ in lay))
    out = {}
    for p in points:
        recs = records(sits, point=p, set_=set_, mode=mode)
        out[int(p)] = {a: sum(1 for r in recs.values() if r.get("arm") == a) for a in arms}
    return out
