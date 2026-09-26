#!/usr/bin/env python3
"""Spec L.11.2, stage 1: the naive-readout screen chosen on stage 0's 41 labelled (b) pairs behind the leave-one-turn-out
gate (l_screen.gate). Pure: no pool, seconds.

    uv run python scripts/run_l_stage1.py
    uv run python scripts/run_l_stage1.py --smoke --allow-dirty --summary results/m0d/l/smoke/l_screen.json

Run it from the repository root. Refused, in this order, unless: the hashed files are clean (or --allow-dirty); the
summary is git-tracked and unchanged against HEAD (outside --smoke; a --smoke run reads and writes only a smoke summary
under results/m0d/l/); blocks "stage0" and "oc" are present (L.11.4: the OC is recorded before stage 1); no later
block (stage2, stage3) is in the summary (outside --smoke); both were produced under this code's measure key and
procedure manifest; block oc's stage0_run_id is block stage0's run_id; stage 0's feature records are well formed.
Recorded:
- the gate (outcome, the rule chosen on all 41 pairs, the LOTO folds, AUCs, by-set precision / recall, the turn folds
  standing for the species folds, plan reading 6) and, when the outcome is not SCREEN_GO, its sentence (L.11.2);
- the stage-3 OC rows at q_b = the LOTO precision (plan reading 15; only when the precision exists), u = 0;
- the pairs whose select-seed G and report-seed G disagree (L.3's record): even pairs' report probes are counts.pre of
  the pinned ceiling file's C3 rows (H.4's oracle presentation; the file reproduces L.11.2's 13 / 14 G counts), odd
  pairs' are stage 0's oracle counts.pre (block stage0, odd[*].report_pre).
Report: <out>/runs/<run id>-stage1.{json,md}; block "stage1" of the summary. On SCREEN_GO the next step is printed:
commit the summary before stage 2 (L.3; stage 2 refuses an uncommitted one). Exit codes: 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from flymon.brain import l_oc, l_store
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT, sha256_file
from flymon.brain.l_cli import (check_committed, code_keys, git_state, head_sha256, later_blocks, other_code, out_allowed,
                                provenance, read_previous, refuse, rule_text, run_id, run_id_chain, same_code, write_block)
from flymon.brain.l_measure import HASHED_FILES, check_pinned
from flymon.brain.l_rules import sentence
from flymon.brain.l_screen import SCREEN_FEW, SCREEN_GO, SCREEN_IMPRECISE, SCREEN_NO_RULE, gate
from flymon.brain.l_spec import SPEC, LSpec, smoke

NEED = ("stage0", "oc")
PROVENANCE_FILES = ("flymon/brain/l_screen.py", "flymon/brain/l_oc.py", "flymon/brain/l_rules.py",
                    "flymon/brain/l_spec.py", "docs/superpowers/specs/j-diag/stage2_oc.py")


def g_of(pre: dict, spec) -> bool:
    """Guard G on one probe set: X and Y at least guard_min for every guard type on every seed (l_screen.features' G)."""
    return all(min(xy) >= spec.guard_min for t in spec.guard_types for xy in pre[t])


def report_pre(spec, s0: dict) -> dict:
    """{key: report-seed counts.pre} — even from the pinned ceiling file (C3, x_only, (b)), odd from block stage0."""
    d = check_pinned(spec.ceiling_path, spec.ceiling_sha256)
    out = {("b", int(r["turn"]), r["x"], r["y"]): r["counts"]["pre"] for r in d["rows"]["C3"]
           if r["mode"] == "x_only" and r["axis"] == "b"}
    out.update({("b", int(o["key"][1]), o["key"][2], o["key"][3]): o["report_pre"] for o in s0.get("odd") or []})
    return out


def g_disagree(feats: list, pre: dict, spec) -> list:
    """The feature records whose select-seed G (their feat) differs from their report-seed G; KeyError on a record with
    no report probes."""
    out = []
    for f in feats:
        k = ("b", int(f["key"][1]), f["key"][2], f["key"][3])
        rg = g_of(pre[k], spec)
        if bool(f["feat"]["G"]) != rg:
            out.append(dict(key=list(k), set=f["set"], select_G=bool(f["feat"]["G"]), report_G=rg))
    return out


def gate_sentence(g: dict, n: int, spec) -> str | None:
    lo = g["loto"]
    if g["outcome"] == SCREEN_GO:
        return None
    if g["outcome"] == SCREEN_FEW:
        return sentence(SCREEN_FEW, dict(n_pass=lo["n_pass"], n=n))
    if g["outcome"] == SCREEN_IMPRECISE:
        return sentence(SCREEN_IMPRECISE, dict(precision=lo["precision"], k=lo["n_hit"], n_pass=lo["n_pass"]))
    return sentence(SCREEN_NO_RULE, dict(cov=spec.cov_min_stage1))


def main(argv=None, spec: LSpec | None = None, summary_spec: LSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=l_store.SUMMARY)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/l/smoke" if a.smoke else "results/m0d/l/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {l_store.ALLOWED_DIR} of the repository root")
    rid = run_id()
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    doc, why = read_previous(a.summary, NEED, a.smoke, check_committed, out_allowed)
    if why:
        return refuse(why)
    why = None if a.smoke else later_blocks(a.summary, "stage1", doc)
    if why:
        return refuse(why)
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    code, manifest = code_keys(a.npz)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    why = other_code(doc, NEED, code["key"], manifest["key"], same_code)
    if why:
        return refuse(why)
    why = run_id_chain(doc, "oc", {"stage0_run_id": "stage0"})
    if why:
        return refuse(why)
    s0 = doc["stage0"]
    feats = s0.get("feats")
    if not feats or any("feat" not in p or "G" not in p["feat"] or "turn" not in p or "key" not in p for p in feats):
        return refuse("block stage0 has no feature records {key, turn, set, testable, feat}")
    try:
        pre = report_pre(spec, s0)
        dis = g_disagree(feats, pre, spec)
    except (OSError, ValueError) as e:
        return refuse(f"the pinned ceiling file is not usable (L.11.1): {e}")
    except (KeyError, TypeError) as e:
        return refuse(f"a feature record has no report-seed probes (even: ceiling counts.pre; odd: block stage0 "
                      f"odd[*].report_pre): {e!r}")
    t0 = time.time()
    g = gate(feats, spec)
    lo = g["loto"]
    oc_row = (None if lo["precision"] is None
              else l_oc.stage3_table((lo["precision"],), spec.oc_naive_a, spec.oc_ratio, spec))
    rule = None if g["final"] is None else g["final"]["rule"]
    files = [ROOT / f for f in PROVENANCE_FILES] + [spec.ceiling_path, a.summary]
    prov = provenance(files, vars(a), sys.argv[1:] if argv is None else argv)
    prov["summary_at_head"] = dict(path=a.summary, sha256=head_sha256(a.summary))
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, stage0_run_id=s0.get("run_id"), oc_run_id=doc["oc"].get("run_id"),
               n_feats=len(feats), gate=g, rule_text=rule_text(rule), sentence=gate_sentence(g, len(feats), spec),
               oc_row=oc_row, oc_row_note="stage-3 OC rows at q_b = the LOTO precision, u = 0 (plan reading 15); "
                                          "None when LOTO passed no pair",
               g_disagree=dis, g_disagree_note="select-seed G (the feature) vs report-seed G (counts.pre): even from "
                                               "the pinned ceiling file, odd from stage 0's oracle rows (L.3 record)",
               provenance=prov, wall_s=time.time() - t0)
    gp = [Params()]
    report = l_store.write_json(out / "runs" / f"{rid}-stage1.json", res, gp)
    lines = [f"# L.11.2 stage 1 {rid}\n\n", f"**{g['outcome']}**, rule {rule_text(rule)}, LOTO {lo['n_hit']}/"
             f"{lo['n_pass']} (precision {lo['precision']})\n\n",
             f"- select vs report G disagreements: {len(dis)} {[d['key'] for d in dis]}\n"]
    if res["sentence"]:
        lines.append(f"\n{res['sentence']}\n")
    l_store.write_bytes(out / "runs" / f"{rid}-stage1.md", "".join(lines).encode(), gp)
    print(f"wrote {report} (sha256 {sha256_file(report)[:12]}): {g['outcome']}, rule {rule_text(rule)}, "
          f"LOTO {lo['n_hit']}/{lo['n_pass']}", flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
        return 0
    write_block(a.summary, "stage1", res, report, gp)
    print(f"wrote {a.summary} (block stage1)", flush=True)
    if g["outcome"] == SCREEN_GO:
        print("next: commit results/summary/l_screen.json before stage 2" if not a.smoke
              else "next (smoke): scripts/run_l_stage2.py --smoke on this smoke summary", flush=True)
    else:
        print(f"stop: {g['outcome']} — {res['sentence']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
