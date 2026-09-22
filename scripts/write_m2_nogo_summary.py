#!/usr/bin/env python3
"""Write results/summary/m2_nogo.json: the M2 no-go record (spec appendix I, "M2 no-go — 시험 불성립").

This is NOT a D.6 (c) FAIL. Spec 5's learning unit test never ran (F v3 withdrawn unrun, F v4 not written), and G.1 and
H.8 say no G or H data judges (c). The record derives from the committed summaries and two git-excluded recorded
measurements the chain the no-go rests on and D.6's status, and refuses to write when a link does not hold, so a claim
in prose cannot survive into the summary (the spec E rule, as in write_m2_probe_summary.py):

  G.10    oracle: testable pairs / all pairs                     results/summary/m2_oracle.json
  G.12    encoder comparison: best score < 0.5 (G.11 rule 4)     results/summary/m2_encoders.json
  H.4     selection STOP_LOW_T_B, no combination at the bar      results/summary/m0d.json block "h4"
  H.4a.8  specificity ceiling read by the H.4a.7 rule: B         results/m0d/diag/h4_specificity_ceiling{,_all}_summary.json
  D.6     (a) G.8 re-judgement, (b) E.2, (c) not measured        results/m2/d6a/g8.json, results/summary/m2_probe.json

The ceiling summaries are git-excluded, so each is bound to committed code: the commit it names must be an ancestor of
HEAD, and the ceiling script at that commit must hash to the sha256 it recorded.

Run from the repository root on a clean tree:  uv run python scripts/write_m2_nogo_summary.py
Exit 2: refused (dirty tree, a missing input, or a link that does not hold); nothing is written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from flymon.brain import d6a

OUT = Path("results/summary/m2_nogo.json")
INPUTS = {
    "m2_oracle": "results/summary/m2_oracle.json",
    "m2_encoders": "results/summary/m2_encoders.json",
    "m0d": "results/summary/m0d.json",
    "m2_probe": "results/summary/m2_probe.json",
    "ceiling_freq": "results/m0d/diag/h4_specificity_ceiling_summary.json",
    "ceiling_all": "results/m0d/diag/h4_specificity_ceiling_all_summary.json",
    "g8": "results/m2/d6a/g8.json",
}
CEILINGS = (("freq", "ceiling_freq"), ("all", "ceiling_all"))
CEILING_SCRIPT = "docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py"
LEARNING_SUMMARY = Path("results/summary/m2_learning.json")   # F.10 #4's summary; it exists only if the test ran
CODE = ("scripts/write_m2_nogo_summary.py", "flymon/brain/d6a.py")
BAR_T_B, BAR_F_A = 0.5, 2              # G.14.4 / H.4 absolute bar
ENCODER_STOP = 0.5                     # G.11 rule 4
CEIL_A, CEIL_B, N_B = 14, 10, 21       # H.4a.7, C3's (b) axis
NAME = "M2 no-go — 시험 불성립 (the learning unit test could not be built)"
DECISIONS = {                          # the user's, 2026-09-22 (spec appendix I) — recorded, not derived
    "route": "3번 (범위 축소 / M2 no-go 기록 + 주장 재설계), G.14.4 row 5 and H.3a.10 ②, under the H.4a.7 rule",
    "name": "M2 no-go — 시험 불성립, kept apart from F.7's FAIL (= D.6 (c) met)",
    "claims": "learning tests limited to the design pair (a mechanism control); 4.3 criterion 1 is not supported in "
              "its current form and is reported as untested; M3 and M4 are on hold, not discarded",
    "e3_std": "STD is not designed now; it is the first question of the next claim declaration",
    "g8": "D.6 (a) re-judged by G.8 for the record",
}


class Refused(Exception):
    """A link of the no-go chain does not hold; the message names it."""


def ceiling_reading(testable_b: int, n_b: int, f_a: int) -> str:
    """H.4a.7 on C3: "A" iff testable (b) >= 14/21 and F_a >= 2; "B" iff testable (b) <= 10/21; else "user"."""
    if n_b != N_B:
        raise Refused(f"a ceiling has {n_b} (b) pairs; H.4a.7 reads {N_B}")
    if testable_b >= CEIL_A and f_a >= BAR_F_A:
        return "A"
    return "B" if testable_b <= CEIL_B else "user"


def at_bar(agg: dict) -> bool:
    """G.14.4 / H.4's absolute bar, recomputed from T_b and F_a (never read from the stored flag)."""
    return agg["T_b"] >= BAR_T_B and agg["F_a"] >= BAR_F_A


def _agg(agg: dict) -> dict:
    return {k: agg[k] for k in ("testable_b", "n_b", "T_b", "F_a", "testable_a", "n_a")}


def ceiling_provenance(summary: dict) -> None:
    """The ceiling ran at a commit on HEAD's history, and the ceiling script there hashes to what it recorded."""
    commit = summary.get("commit", "")
    if summary.get("script") != CEILING_SCRIPT:
        raise Refused(f"ceiling script {summary.get('script')!r} is not {CEILING_SCRIPT}")
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    capture_output=True).returncode != 0:
        raise Refused(f"ceiling commit {commit[:12]!r} is not an ancestor of HEAD")
    blob = subprocess.run(["git", "show", f"{commit}:{CEILING_SCRIPT}"], capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest() != summary.get("script_sha256"):
        raise Refused(f"the ceiling script at {commit[:12]} does not hash to the recorded sha256")


def derive(src: dict, learning_ran: bool) -> dict:
    """The record body from the parsed inputs (keys of INPUTS). Raises Refused when a link of the chain does not hold."""
    if learning_ran:
        raise Refused(f"{LEARNING_SUMMARY} exists: the learning unit test ran, so the test was not 'not constructible'")
    oracle = src["m2_oracle"]
    g10 = {k: oracle["counts"][k] for k in ("testable", "pairs", "odd_gate_pool", "even_pilot_pool")}
    g10["design_pair_testable"] = oracle["design_pair_testable"]

    enc = src["m2_encoders"]["even"]
    scores = {e: s["score"] for e, s in sorted(enc["scores"].items())}
    best = max(scores.values())
    if enc["decision"]["winner"] is not None or best >= ENCODER_STOP:
        raise Refused(f"G.12 did not stop: winner {enc['decision']['winner']!r}, best score {best:.3f}")
    g12 = {"scores": scores, "best": best, "winner": None, "rule": "G.11 rule 4: best score < 0.5 -> stop and report"}

    blk = src["m0d"]["h4"]
    h4 = blk["h4"]
    combos = {c: _agg(v["oracle"]["aggregate"]) for c, v in sorted(h4["combos"].items())}
    reached = sorted(c for c, a in combos.items() if at_bar(a))
    if h4["outcome"] != "STOP_LOW_T_B" or reached:
        raise Refused(f"H.4 is not a STOP below the bar: outcome {h4['outcome']}, at the bar {reached}")
    h4_rec = {"run_id": blk["run_id"], "commit": blk["git"]["commit"], "outcome": h4["outcome"],
              "eligible": h4["selection"]["eligible"], "combos": combos,
              "bar": f"T_b >= {BAR_T_B} and F_a >= {BAR_F_A} (G.14.4, H.4)"}

    h4_run = f"results/m0d/h4/runs/{blk['run_id']}.json"
    families = {}
    for fam, key in CEILINGS:
        s = src[key]
        if s.get("smoke") is not False or s.get("h4_run") != h4_run or s.get("family", "freq") != fam:
            raise Refused(f"{INPUTS[key]} is not the full {fam}-family ceiling of H.4 run {blk['run_id']}")
        c3 = s["engines"]["C3"]["aggregate"]
        families[fam] = {"commit": s["commit"], "script_sha256": s["script_sha256"], "C3": _agg(c3),
                         "C0": _agg(s["engines"]["C0"]["aggregate"]),
                         "reading": ceiling_reading(c3["testable_b"], c3["n_b"], c3["F_a"])}
    readings = {f: v["reading"] for f, v in families.items()}
    if set(readings.values()) != {"B"}:
        raise Refused(f"the H.4a.7 rule does not read B on every ceiling family: {readings}")
    ceiling = {"rule": f"H.4a.7 on C3: A iff testable (b) >= {CEIL_A}/{N_B} and F_a >= {BAR_F_A}; "
                       f"B iff <= {CEIL_B}/{N_B}; otherwise the user's call",
               "families": families, "reading": "B"}

    try:
        a = d6a.judge(src["g8"])
    except ValueError as e:
        raise Refused(f"G.8 record: {e}") from None
    e2 = src["m2_probe"]["d6_ruling"]
    b = e2["b_candidate_ratio"]
    d6 = {
        "a": {"source": "G.8 re-judgement (E.7 #4), results/m2/d6a/g8.json", **a,
              "e2_record": {k: e2["a_runaway"][k] for k in ("fired", "n_presentations", "max_kc_sub_window_hz",
                                                             "margin_hz")}},
        "b": {"source": "E.2, results/summary/m2_probe.json", "fired": b["fired"], "n_turns_over": b["n_turns_over"],
              "n_turns": b["n_turns"], "ratio_max": b["ratio_max"]},
        "c": {"status": "not measured", "why": "spec 5's learning unit test never ran (F v3 withdrawn unrun, F v4 not "
                                               "written); G.1 and H.8: no G or H data judges (c)"},
    }
    return {"name": NAME, "not_a_d6c_fail": True, "learning_unit_test": {"ran": False, "summary": str(LEARNING_SUMMARY)},
            "chain": {"g10_oracle": g10, "g12_encoders": g12, "h4": h4_rec, "h4a8_ceiling": ceiling},
            "d6": d6, "std_condition_fired": bool(a["fired"] or b["fired"]), "decisions": DECISIONS}


def git_state(out: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    dirty = [ln[3:] for ln in run("status", "--porcelain").splitlines() if ln.strip() and ln[3:] != out.as_posix()]
    return {"git_commit": run("rev-parse", "HEAD").strip(), "dirty": dirty}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    try:
        git = git_state(a.out)
        if git["dirty"]:
            raise Refused(f"the tree is dirty: {git['dirty']}")
        missing = [p for p in INPUTS.values() if not Path(p).is_file()]
        if missing:
            raise Refused(f"missing inputs {missing}")
        src = {k: json.loads(Path(p).read_text()) for k, p in INPUTS.items()}
        for _, key in CEILINGS:
            ceiling_provenance(src[key])
        body = derive(src, LEARNING_SUMMARY.exists())
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    record = {"what": "spec appendix I: the M2 no-go record, derived by scripts/write_m2_nogo_summary.py",
              "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git["git_commit"], "git_dirty": False,
              "inputs_sha256": {p: _sha256(p) for p in INPUTS.values()}, "code_sha256": {p: _sha256(p) for p in CODE},
              **body}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    ch, d = body["chain"], body["d6"]
    print(f"wrote {a.out}: H.4 {ch['h4']['outcome']}, ceiling C3 "
          + ", ".join(f"{f} {v['C3']['testable_b']}/{v['C3']['n_b']}" for f, v in ch["h4a8_ceiling"]["families"].items())
          + f" -> B; D.6 (a) {'met' if d['a']['fired'] else 'not met'} (max window {d['a']['max_win_spikes']} spikes), "
          f"(b) {'met' if d['b']['fired'] else 'not met'}, (c) not measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
