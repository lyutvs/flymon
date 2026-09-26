#!/usr/bin/env python3
"""Spec L.11.4: the whole L procedure's operating characteristics, recorded after stage 0 and before stage 1 (they never
change a judgement). Pure: no pool.

    uv run python scripts/run_l_oc.py                                                        # minutes
    uv run python scripts/run_l_oc.py --smoke --allow-dirty --summary results/m0d/l/smoke/l_screen.json

Run it from the repository root. Refused unless block "stage0" of --summary exists under this code's measure key and
procedure manifest, and (outside --smoke) the summary is git-tracked and unchanged against HEAD; a --smoke run reads and
writes only a smoke summary under results/m0d/l/. Computed (flymon/brain/l_oc.py):
- the stage-1 gate on stage 0's 41 real feature vectors and turns, labels Bernoulli(q) for G-passing pairs and
  Bernoulli(q_fail) otherwise, q in SPEC.oc_q x q_fail in SPEC.oc_q_fail, SPEC.oc_draws draws from one
  numpy.random.default_rng(SPEC.oc_seed) taken in that order;
- P(COVERAGE_SHORT | c) for c in SPEC.oc_c;
- the exact stage-3 band table for q_b in SPEC.oc_q (plan ruling P13: it covers every gate q) x naive_a 2-8 x q_a/q_b in
  SPEC.oc_ratio at u = 0, plus the rows with C3's turn effects u (J.12.9's stage2_oc on H.4's recorded C3 oracle block);
- the whole-procedure products (l_oc.whole_table, stages independent, stated).
The report goes to <out>/runs/<run id>-oc.{json,md}; block "oc" carries the report's sha256 and the provenance of
J.12.9 (the inputs' sha256 incl. the summary at HEAD, commit, dirty, arguments). Exit codes: 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from flymon.brain import l_oc, l_store
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT, sha256_file
from flymon.brain.l_cli import (check_committed, code_keys, git_state, head_sha256, out_allowed, provenance, refuse,
                                run_id, same_code, write_block)
from flymon.brain.l_measure import HASHED_FILES
from flymon.brain.l_spec import SPEC, LSpec, smoke

PROVENANCE_FILES = ("flymon/brain/l_oc.py", "flymon/brain/l_screen.py", "flymon/brain/l_rules.py",
                    "flymon/brain/j_rules.py", "flymon/brain/l_spec.py", "docs/superpowers/specs/j-diag/stage2_oc.py")


def gate_records(gate_ps: dict) -> list:
    """{(q, q_fail): P} -> [{q, q_fail, P}] (JSON has no tuple keys)."""
    return [dict(q=q, q_fail=qf, P=dict(P)) for (q, qf), P in gate_ps.items()]


def uncovered_q(qs, rows: list) -> list:
    """The gate q's with no stage-3 row of exactly that q_b (whole_table matches floats exactly: refuse, never skip)."""
    have = {r["q_b"] for r in rows}
    return [q for q in qs if q not in have]


def main(argv=None, spec: LSpec | None = None, summary_spec: LSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=l_store.SUMMARY)
    ap.add_argument("--m0d", default=None)
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
    try:
        doc = json.loads(Path(a.summary).read_text())
    except (OSError, ValueError) as e:
        return refuse(f"no usable summary at {a.summary}: {e}")
    s0 = doc.get("stage0")
    if not isinstance(s0, dict):
        return refuse(f"{a.summary} has no block stage0 (run scripts/run_l_stage0.py and commit its summary first)")
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    code, manifest = code_keys(a.npz)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    if not same_code(s0, code["key"], manifest["key"]):
        return refuse(f"block stage0 was produced under other code (measure key {str(s0.get('measure_key'))[:12]}, "
                      f"manifest {str((s0.get('code') or {}).get('key'))[:12]}; this code's are {code['key'][:12]} / "
                      f"{manifest['key'][:12]})")
    if a.smoke:
        if not out_allowed(a.summary):
            return refuse(f"a --smoke run reads and writes only a smoke summary under {l_store.ALLOWED_DIR}, "
                          f"not {a.summary}")
    else:
        why = check_committed(a.summary, ["stage0"])
        if why:
            return refuse(why)
    feats = s0.get("feats")
    if not feats or any("feat" not in p or "G" not in p["feat"] or "turn" not in p for p in feats):
        return refuse("block stage0 has no feature records {turn, set, testable, feat}")
    m0d_path = a.m0d or spec.m0d_path
    try:
        rec = json.loads(Path(m0d_path).read_text())["h4"]["h4"]["combos"][spec.j.c3_name]["oracle"]
        ub, ua = l_oc.c3_offsets(rec)
    except (OSError, ValueError, KeyError, SystemExit) as e:
        return refuse(f"no usable C3 oracle record in block h4 of {m0d_path}: {e}")
    t0 = time.time()
    rows = (l_oc.stage3_table(spec.oc_q, spec.oc_naive_a, spec.oc_ratio, spec)
            + l_oc.stage3_table(spec.oc_q, (len(ua),), spec.oc_ratio, spec, u_offsets=(ub, ua)))
    missing = uncovered_q(spec.oc_q, rows)
    if missing:
        return refuse(f"the gate q's {missing} have no stage-3 row (q_b values {sorted({r['q_b'] for r in rows})})")
    rng = np.random.default_rng(spec.oc_seed)
    gate_ps = {}
    for q in spec.oc_q:
        for qf in spec.oc_q_fail:
            gate_ps[(q, qf)] = l_oc.gate_oc(feats, q, qf, spec, rng)
            print(f"gate q {q} q_fail {qf}: {gate_ps[(q, qf)]}", flush=True)
    short = {c: l_oc.coverage_short_prob(c, spec.n_pass, spec.max_screened) for c in spec.oc_c}
    whole = l_oc.whole_table(gate_ps, short, rows)
    files = [ROOT / f for f in PROVENANCE_FILES] + [m0d_path, a.summary]
    prov = provenance(files, vars(a), sys.argv[1:] if argv is None else argv)
    prov["summary_at_head"] = dict(path=a.summary, sha256=head_sha256(a.summary))
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, stage0_run_id=s0.get("run_id"), n_feats=len(feats),
               rng=f"one numpy.random.default_rng({spec.oc_seed}), cells in oc_q x oc_q_fail order",
               gate=gate_records(gate_ps), coverage_short=[dict(c=c, p=p) for c, p in short.items()],
               c3_offsets=dict(ub=list(ub), ua=list(ua)), stage3=rows, whole=whole, provenance=prov,
               wall_s=time.time() - t0)
    gp = [Params()]
    report = l_store.write_json(out / "runs" / f"{rid}-oc.json", res, gp)
    lines = [f"# L.11.4 operating characteristics {rid}\n\n", f"{l_oc.ASSUMPTION}\n\n",
             *(f"- gate q {g['q']} q_fail {g['q_fail']}: {g['P']}\n" for g in res["gate"]),
             *(f"- P(COVERAGE_SHORT | c {c}) = {p:.4f}\n" for c, p in short.items())]
    l_store.write_bytes(out / "runs" / f"{rid}-oc.md", "".join(lines).encode(), gp)
    print(f"wrote {report} (sha256 {sha256_file(report)[:12]}) in {res['wall_s']:.0f} s", flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "oc", res, report, gp)
        print(f"wrote {a.summary} (block oc)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
