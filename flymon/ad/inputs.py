"""AD's inputs (spec AD.1, AD.2): the confirmation set (AC.4, seed 301) and the evaluation schedule (seed 303, one row of
20) reused from stage 1; the new learning schedule make_schedule(12, 120, seed 306) - not stage 1's extended - with its
digest over all 120 battles and the 80- (fallback) and 40-battle (futility) prefix digests, so the fallback never redraws
it; the fresh 20-pair descriptive set (seed 308, AC.4's draw rule) from the candidates the confirmation draw left (82);
the layout (FLY-L 24-35, FLY-OS 36-47 on pool indices 0-23) and the ids DL / DE (smoke XL / XE, bench YL / YE) with the
GLOBAL fly in the id. FLY-L k and FLY-OS k play learning row k; every fly plays the one evaluation schedule.
check_inputs rebuilds the document from the seeds; a mismatch of the AD-fixed fields is STOP_MANIFEST (AD.2 (나))."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..ac import confirm
from ..ac import schedules as ac_sched
from ..ac.spec import SPEC as AC
from ..battle.schedule import ScheduledBattle
from ..rescope import blocks
from .spec import LABEL, SPEC
from .store import digest, sha256_file

ID = re.compile(r"^(DL|DE|XL|XE|YL|YE)-f(\d{2})-b(\d{3})$")
CODE = {"FLYL": "L", "FLYOS": "O"}


def battle_index(battle_id: str) -> int:
    m = ID.match(str(battle_id))
    if m is None:
        raise ValueError(f"battle id {battle_id!r} is not <DL|DE|XL|XE|YL|YE>-fNN-bNNN")
    return int(m.group(3))


def account(arm: str, block: str, gfly: int) -> str:
    name = f"fm-d{CODE[arm]}{block}-f{int(gfly):02d}"
    if len(name) > 18:
        raise ValueError(f"account {name!r} exceeds Showdown's 18 characters")
    return name


def learn_canonical(spec=SPEC) -> list:
    return ac_sched.canonical(spec.n_rows(), spec.learn_battles, spec.learn_seed)


def eval_canonical(spec=SPEC) -> list:
    return ac_sched.canonical(1, spec.eval_battles, spec.eval_seed)


def prefix(sched, n: int) -> list:
    return [s for s in sched if int(str(s.battle_id).rsplit("-b", 1)[1]) < int(n)]


def pair_key(p) -> str:
    return json.dumps(p["key"], sort_keys=True)


def new_set(spec, conf_pairs) -> dict:
    rows = confirm.candidate_pairs()["pairs"]
    used = {pair_key(p) for p in conf_pairs}
    rest = [r for r in rows if pair_key(r) not in used]
    out = dict(n_candidates=len(rows), n_rest=len(rest), n_pairs=spec.n_new_pairs, seed=spec.new_seed)
    if len(rest) < spec.n_new_pairs:
        return dict(out, status="STOP_SET", pairs=[], digest=None)
    pairs = confirm.draw(rest, spec.n_new_pairs, spec.new_seed)
    return dict(out, status="OK", pairs=pairs, digest=digest(pairs))


def assign(canon, members, tag: str) -> list:
    by: dict = {}
    for s in canon:
        by.setdefault(int(s.fly_id), []).append(s)
    out = []
    for i, g, k in members:
        for s in by.get(int(k), []):
            b = int(s.battle_id.rsplit("-b", 1)[1])
            out.append(ScheduledBattle(f"{tag}-f{int(g):02d}-b{b:03d}", int(i), list(s.my_team), list(s.opp_team),
                                       s.opponent))
    return out


def schedules_for(spec, doc, T: int) -> tuple:
    lay = spec.layout()
    learn = assign(prefix(blocks.schedule_from_rows(doc["learn"]), T),
                   [(i, g, k) for i, (_, k, g) in enumerate(lay)], spec.tags[0])
    ev = assign(blocks.schedule_from_rows(doc["eval"]), [(i, g, 0) for i, (_, _, g) in enumerate(lay)], spec.tags[1])
    return learn, ev


def input_paths(spec=SPEC) -> tuple:
    if spec.mode == "run":
        return Path("results/summary/ad_inputs.json"), Path("results/summary/ad_inputs_manifest.json")
    return Path(spec.out_root) / "inputs.json", Path(spec.out_root) / "inputs_manifest.json"


def _teams(sched) -> set:
    return {(tuple(s.my_team), tuple(s.opp_team)) for s in sched}


def inputs_doc(spec=SPEC) -> dict:
    conf = confirm.confirmation_set(spec)
    new = new_set(spec, conf["pairs"])
    learn, ev = learn_canonical(spec), eval_canonical(spec)
    blocks.assert_disjoint(learn, ev)                       # AD.2: the learning and evaluation teams never share
    overlap = len(_teams(learn) & _teams(ac_sched.learn_canonical(AC))) if spec.mode == "run" else None
    status = "OK" if conf["status"] == "OK" and new["status"] == "OK" else "STOP_SET"
    digests = dict(confirm=conf["digest"], eval=blocks.schedule_digest(ev), learn=blocks.schedule_digest(learn),
                   learn_fallback=blocks.schedule_digest(prefix(learn, spec.fallback_battles)),
                   learn_futility=blocks.schedule_digest(prefix(learn, spec.futility_point)), new=new["digest"])
    return dict(status=status, mode=spec.mode, seeds=spec.seeds(), n_pairs=spec.n_pairs, confirm=conf, new=new,
                learn=blocks.schedule_rows(learn), eval=blocks.schedule_rows(ev), digests=digests,
                layout=[list(x) for x in spec.layout()], tags=list(spec.tags), learn_overlap_302=overlap, label=LABEL)


def fixed_fields(doc: dict) -> dict:
    return dict(digests=doc["digests"], seeds=doc["seeds"], layout=doc["layout"], tags=doc["tags"])


def input_manifest(doc: dict, doc_path, reuse=None, problems=()) -> dict:
    problems = list(problems)
    status = doc["status"] if doc["status"] != "OK" else ("STOP_REUSE" if problems else "OK")
    return dict(status=status, mode=doc["mode"], fixed=fixed_fields(doc), reuse=reuse, reuse_problems=problems,
                counts=dict(confirm=doc["confirm"]["counts"], new_rest=doc["new"]["n_rest"]),
                inputs_path=str(doc_path), inputs_sha256=sha256_file(doc_path), label=LABEL)


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=float))


def check_inputs(doc: dict, manifest: dict, spec, doc_path) -> list:
    """What does not match ([] = everything does): the document against one rebuilt from the seeds (per top-level key),
    the manifest's AD-fixed fields against the rebuilt ones, and the manifest's file sha256 against the file."""
    want = _norm(inputs_doc(spec))
    have = _norm(doc)
    bad = [k for k in sorted(set(want) | set(have)) if want.get(k) != have.get(k)]
    if _norm(manifest.get("fixed")) != _norm(fixed_fields(want)):
        bad.append("manifest_fixed")
    if manifest.get("inputs_sha256") != sha256_file(doc_path):
        bad.append("inputs_sha256")
    return bad
