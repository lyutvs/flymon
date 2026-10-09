"""AD.5's smoke gates. STOP_SMOKE = a code defect: the full suite; the four safety-constraint tests; worker ==
in-process (KC counts, A, P, V, choice, the weight change after one pulse) and the workers' L_V sha; the stale-pulse
interleaving tests (a)-(d); the OS draw unit tests (key dependence, same-key reproduction, candidate sharing, re-draw
across turns, uniform / independent 16 x 16, global fly numbers, schedule id + battle.turn, byte-identical FLY-encoder
odour) and the global-seed tests; the run-manifest refusal test; the smoke run's log schema, FLY-OS shadow schema,
completion across a stop and a --resume, frozen situation evaluations, one evaluation digest; the OS total strength
|sum - k| <= 1e-9; the naive-L_V POOL odour response (tie / A = 0 / P = 0 shares <= 0.5) equal to stage 1's; the
stage-1 reproduction (flies 0 and 12, point 0: v / a / p / kc_active / pick / tie byte-identical). STOP_REUSE = AD.2
(가) fails. STOP_BALANCE = the committed KC calibration outside 1 +- max(0.15, 3 SD_R). Record only: the drawn species
per true opponent and the same-type share in the smoke battles, the KC > 2 turn share.
- One evaluation digest (ruling D1): what the flies PLAYED - every pool fly's evaluation battle ids in
  logs/eval/battles.jsonl (tag, global fly, battle number) mapped onto the document's evaluation rows - gives one
  schedule digest, equal to the document's and to result.json's; not the digest result.json copied from the document.
- AD.2 (가) before (나): the reuse fields are checked against the run-mode digests rebuilt from the seeds (and against
  the run's input manifest if it exists), so a stage-1 drift is STOP_REUSE; a rebuild that fails assert_disjoint (the
  learning and evaluation teams share) is a smoke gate with its own wording, not a traceback."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..rescope import blocks
from . import inputs, os_encoder, reuse

SMOKE = "results/summary/ad_smoke.json"
STALE_TESTS = ("tests/ad/test_ad_stale.py", "tests/battle/test_ad_player.py")
OS_TESTS = ("tests/ad/test_ad_os.py",)
SEED_TESTS = ("tests/ad/test_ad_seeds.py",
              "tests/battle/test_ad_player.py::test_pulse_request_carries_token_generation_and_global_seed",
              "tests/ad/test_ad_hooks.py::test_make_evaluate_uses_os_odours_and_global_seeds")
MANIFEST_TESTS = ("tests/ad/test_ad_run.py::test_resume_refuses_a_changed_run_manifest",
                  "tests/ad/test_ad_runner.py::test_changed_checkpoint_key_is_stop_manifest",
                  "tests/ad/test_ad_run_manifest.py::test_check_detects_changed_budget_code_and_points")
FIELDS = ("v", "a", "p", "kc_active", "pick", "tie")
OS_KEYS = {"true", "drawn", "same_types"}


def _records(paths):
    for p in paths:
        for line in Path(p).read_text().splitlines():
            yield json.loads(line)


def _fly_decisions(paths, os_flies=None):
    return [r for r in _records(paths) if r.get("kind") == "decision" and r.get("decider") == "fly"
            and (os_flies is None or r.get("fly") in os_flies)]


def os_shadow_errors(paths, os_flies, os_seed=None) -> list:
    """Fly decisions of FLY-OS pool flies whose `os` (written by _ADCommon._log: top-level, {true, drawn, same_types})
    is missing or malformed, and fly decisions of other flies that carry one. With os_seed, `drawn` must also be the
    seeded draw of the record (derive_seed(os_seed, "os", gfly, battle_id, turn))."""
    os_flies = {int(f) for f in os_flies}
    bad = []
    for r in _fly_decisions(paths):
        o = r.get("os")
        where = f"fly {r.get('fly')} {r.get('battle_id')} turn {r.get('turn')}"
        if r.get("fly") not in os_flies:
            if o is not None:
                bad.append(f"{where}: os on a non-FLY-OS fly")
            continue
        if not isinstance(o, dict) or set(o) != OS_KEYS or not isinstance(o["same_types"], bool):
            bad.append(f"{where}: os {o!r}")
        elif os_seed is not None:
            try:
                want = os_encoder.SPECIES[os_encoder.draw_index(
                    os_encoder.battle_seed(os_seed, r["gfly"], r["battle_id"], r["turn"]))]
            except (KeyError, TypeError, ValueError) as e:
                bad.append(f"{where}: no seed fields ({type(e).__name__}: {e})")
                continue
            if o["drawn"] != want:
                bad.append(f"{where}: drawn {o['drawn']!r} is not the seeded draw {want!r}")
    return bad


def os_draw_record(paths, os_flies) -> dict:
    by = defaultdict(lambda: defaultdict(int))
    same = []
    for r in _fly_decisions(paths, {int(f) for f in os_flies}):
        o = r.get("os") or {}
        if "drawn" in o:
            by[o["true"]][o["drawn"]] += 1
            same.append(bool(o.get("same_types")))
    return dict(n=len(same), same_types_share=(sum(same) / len(same)) if same else None,
                by_true={t: dict(sorted(d.items())) for t, d in sorted(by.items())})


def reproduction_diff(want: list, got: list) -> list:
    if not want:
        return ["no stage-1 records"]
    bad = []
    g = {r["fly"]: r for r in got}
    for w in want:
        h = g.get(w["fly"])
        if h is None:
            bad.append(f"fly {w['fly']}: no record")
            continue
        hs = {(s["pair"], s["side"]): s for s in h["situations"]}
        for s in w["situations"]:
            t = hs.get((s["pair"], s["side"]))
            for k in FIELDS:
                if t is None or json.dumps(s[k]) != json.dumps(t[k]):
                    bad.append(f"fly {w['fly']} pair {s['pair']} side {s['side']}: {k}")
    return bad


def same_response(ad: dict, ac: dict) -> bool:
    keys = ("n_situations", "n_candidates", "tie_ratio", "a0_ratio", "p0_ratio")
    return all(ad.get(k) == ac.get(k) for k in keys)


def eval_played(battles_path, eval_rows, lay, tag: str) -> dict:
    """Ruling D1: per pool fly, the evaluation schedule it played - its battle ids in battles.jsonl (each must carry the
    block's tag and the fly's global number) mapped by battle number onto the document's evaluation rows - and that
    schedule's digest. A fly missing, duplicating or adding a battle number is a problem."""
    rows = {int(r["battle_id"].rsplit("-b", 1)[1]): r for r in eval_rows}
    ids: dict = defaultdict(list)
    problems = []
    for rec in blocks.read_jsonl(battles_path):
        i, bid = int(rec["fly"]), str(rec["battle_id"])
        m = inputs.ID.match(bid)
        if i >= len(lay) or m is None or m.group(1) != tag or int(m.group(2)) != int(lay[i][2]):
            problems.append(f"pool fly {i}: battle id {bid!r} is not {tag}-f{lay[i][2] if i < len(lay) else '??'}-bNNN")
            continue
        ids[i].append(int(m.group(3)))
    digests = {}
    for i in range(len(lay)):
        got = sorted(ids.get(i, []))
        if got != sorted(rows):
            problems.append(f"pool fly {i} (fly {lay[i][2]}): played battles {got}, schedule {sorted(rows)}")
            continue
        digests[i] = blocks.schedule_digest(blocks.schedule_from_rows([rows[b] for b in got]))
    return dict(digests=digests, problems=problems)


def eval_shared(played: dict, doc_digest, result_digest, n_flies: int) -> bool:
    d = set(played["digests"].values())
    return (not played["problems"] and len(played["digests"]) == int(n_flies) and d == {doc_digest}
            and result_digest == doc_digest)


def run_inputs_rebuilt(spec) -> tuple:
    """(the run's input document rebuilt from the seeds, None) or (None, the reason): inputs_doc's assert_disjoint
    raises when a learning and an evaluation battle share teams and order (AD.2) - reported as a gate failure."""
    try:
        return inputs.inputs_doc(spec), None
    except ValueError as e:
        return None, f"learning and evaluation schedules share teams (AD.2, assert_disjoint): {e}"


def reuse_check(digests: dict, spec, run_manifest=None) -> tuple:
    """AD.2 (가) on the run's (rebuilt) confirm / eval digests: reuse_problems, plus - if the run's input manifest is
    given - its recorded reuse fields changed since (run_ad.py's _reuse_bad order: reuse before check_inputs)."""
    f = reuse.reuse_fields(digests["confirm"], digests["eval"], spec)
    bad = reuse.reuse_problems(f, spec)
    if run_manifest is not None and not f.get("missing_stage1"):
        bad = reuse.changed_since(run_manifest.get("reuse"), f) + bad
    return f, bad


def status(gates: dict, groups: dict) -> str:
    for code, names in (("STOP_SMOKE", groups["smoke"]), ("STOP_REUSE", groups["reuse"]),
                        ("STOP_BALANCE", groups["balance"])):
        if not all(gates[n] for n in names):
            return code
    return "OK"
