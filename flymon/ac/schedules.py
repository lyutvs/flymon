"""AC's battle schedules (spec AC.4, 4.4) and the experiment-input manifest (AC.1, the 4.4 freeze, part (ii)).
- Canonical learning schedule: make_schedule(learn_rows = 12 flies, 40 battles, seed 302); arm-local fly k plays
  canonical row k, so FLY k, C-off k and FLY-TB k share their learning team order.
- Canonical evaluation schedule (controller ruling, AC.4 "이 일정"): ONE schedule, make_schedule(eval_rows = 1,
  20 battles, seed 303), played by every fly of every arm (members (g, 0)); all flies share its team order. Every arm
  verifies the canonical schedules' digests.
- Ids <tag>-fGG-bBBB with the global fly g of the run (BRAIN: FLY 0-11, C-off 12-17, TB 18-23) and the canonical
  battle number; tags AL / AE (run), SL / SE (smoke), BL / BE (bench); derive_seed keys on the id, so no two flies or
  runs share a seed stream. Accounts fm-a<arm code><block>-fGG and opponents fm-h-<id>[-attempt] fit Showdown's 18
  characters.
- The input document holds the confirmation set and both canonical schedules; its manifest holds their digests and the
  document's sha256. check_inputs recomputes everything from the seeds. Both carry LABEL (OK and STOP_SET alike)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..battle.schedule import ScheduledBattle, make_schedule
from ..rescope import blocks
from . import confirm
from .spec import LABEL, SPEC
from .store import sha256_file

ID = re.compile(r"^(AL|AE|SL|SE|BL|BE)-f(\d{2})-b(\d{3})$")
CODE = {"FLY": "FL", "COFF": "CO", "TB": "TB", "RND": "RN", "MAX": "MX"}


def canonical(n_flies: int, n_battles: int, seed: int) -> list:
    return make_schedule(int(n_flies), int(n_battles), "heuristic", seed=int(seed))


def learn_canonical(spec=SPEC) -> list:
    return canonical(spec.learn_rows(), spec.learn_battles, spec.learn_seed)


def eval_canonical(spec=SPEC) -> list:
    return canonical(spec.eval_rows(), spec.eval_battles, spec.eval_seed)


def assign(canon, members, tag: str) -> list:
    by: dict = {}
    for s in canon:
        by.setdefault(int(s.fly_id), []).append(s)
    if not canon:  # bench: no evaluation block (eval_battles = 0)
        return []
    out = []
    for g, k in members:
        for s in by[int(k)]:
            b = int(s.battle_id.rsplit("-b", 1)[1])
            out.append(ScheduledBattle(f"{tag}-f{int(g):02d}-b{b:03d}", int(g), list(s.my_team), list(s.opp_team),
                                       s.opponent))
    return out


def battle_index(battle_id: str) -> int:
    m = ID.match(str(battle_id))
    if m is None:
        raise ValueError(f"battle id {battle_id!r} is not <AL|AE|SL|SE|BL|BE>-fNN-bNNN")
    return int(m.group(3))


def account(arm: str, block: str, fly: int) -> str:
    name = f"fm-a{CODE[arm]}{block}-f{int(fly):02d}"
    if len(name) > 18:
        raise ValueError(f"account {name!r} exceeds Showdown's 18 characters")
    return name


def opponent_name(battle_id: str, attempt: int) -> str:
    name = f"fm-h-{battle_id}" + (f"-{attempt}" if attempt else "")
    if len(name) > 18:
        raise ValueError(f"opponent name {name!r} exceeds Showdown's 18 characters")
    return name


def check_names(sched, retry_max: int) -> None:
    for sb in sched:
        opponent_name(sb.battle_id, retry_max)


def input_paths(spec=SPEC) -> tuple:
    if spec.mode == "run":
        return Path("results/summary/ac_inputs.json"), Path("results/summary/ac_inputs_manifest.json")
    return Path(spec.out_root) / "inputs.json", Path(spec.out_root) / "inputs_manifest.json"


def inputs_doc(spec=SPEC) -> dict:
    conf = confirm.confirmation_set(spec)
    learn, ev = learn_canonical(spec), eval_canonical(spec)
    blocks.assert_disjoint(learn, ev)
    return dict(status=conf["status"], mode=spec.mode, seeds=spec.seeds(), n_pairs=spec.n_pairs, confirm=conf,
                learn=blocks.schedule_rows(learn), eval=blocks.schedule_rows(ev),
                digests=dict(confirm=conf["digest"], learn=blocks.schedule_digest(learn),
                             eval=blocks.schedule_digest(ev)), label=LABEL)


def input_manifest(doc: dict, doc_path) -> dict:
    return dict(status=doc["status"], mode=doc["mode"], seeds=doc["seeds"], digests=doc["digests"],
                counts=doc["confirm"]["counts"], inputs_path=str(doc_path), inputs_sha256=sha256_file(doc_path),
                label=LABEL)


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=float))


def check_inputs(doc: dict, manifest: dict, spec, doc_path) -> list:
    """What does not match ([] = everything does): the document against one rebuilt from the seeds (per top-level
    key), the manifest's digests against the rebuilt ones, and the manifest's file sha256 against the file."""
    want = _norm(inputs_doc(spec))
    have = _norm(doc)
    bad = [k for k in sorted(set(want) | set(have)) if want.get(k) != have.get(k)]
    if manifest.get("digests") != want["digests"]:
        bad.append("manifest_digests")
    if manifest.get("inputs_sha256") != sha256_file(doc_path):
        bad.append("inputs_sha256")
    return bad
