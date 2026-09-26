#!/usr/bin/env python3
"""Spec L.11.3, stage 2: the stage-1 rule screens the new (b) pairs in declared order on naive measurements
(l_jobs.naive_job through l_measure.LMeasurer), stopping at the n_pass-th pass (plan reading 10), and draws the lift
sample from the screened failing pairs (decision 12).

    uv run python scripts/run_l_stage2.py                                   # ~35 min (resumable: rerun it)
    uv run python scripts/run_l_stage2.py --smoke --allow-dirty --workers 4 --summary results/m0d/l/smoke/l_screen.json

Run it from the repository root. Refused before the pool, in this order, unless: the hashed files are clean (or
--allow-dirty); the summary is git-tracked and unchanged against HEAD (outside --smoke; a --smoke run reads and writes
only a smoke summary under results/m0d/l/); blocks "stage0", "oc" and "stage1" are present; each was produced under
this code's measure key and procedure manifest; stage 1's outcome is SCREEN_GO with a rule; C3, its readout, z and
pools load from results/summary/m0d.json; the new pair lists' digests are SPEC.b_digest / SPEC.a_digest
(l_pairs.check_digests, before the screen order exists); H.4's even pair list is the declared one.
Then:
1. naive self-check: LMeasurer.naive on the first even (b) pair must equal the pinned ceiling file's row (fx / fy by
   np.array_equal, select-seed pre equal); a mismatch is recorded and exits 5 (stop and ask the user);
2. batches of --workers pairs in declared order, each followed by l_rules.screen_order, until it decides (the n_pass-th
   pass, or position min(max_screened, len(b)) reached); pairs measured after n* in the last batch stay in the cache and
   in `measured`, but are neither screened nor counted;
3. the lift draw from the screened failing pairs (SCREENED only);
4. the report <out>/runs/<run id>-stage2.{json,md} and block "stage2" (outside --smoke: clean, declared spec). On
   COVERAGE_SHORT the block carries L.11.3's sentence and no stage 3 follows.
Exit codes: 0 written, 2 refused, 5 naive self-check mismatch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from flymon.brain import j_store, l_store  # noqa: F401  (j_store: the C3 loader l_cli.load_c3_record uses)
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_pairs import even_pairs, pair_key, pairs_digest
from flymon.brain.j_runner import params_json
from flymon.brain.k_metrics import readout_weights
from flymon.brain.l_cli import (POOL_TIMEOUT_S, check_committed, code_keys, git_state, guard_params, head_sha256,
                                load_c3_record, new_set, other_code, out_allowed, provenance, read_previous, refuse,
                                rule_text, run_id, same_code, write_block)
from flymon.brain.l_measure import HASHED_FILES, LMeasurer, dense, load_even
from flymon.brain.l_rules import COVERAGE_SHORT, SCREENED, lift_draw, screen_order, sentence
from flymon.brain.l_screen import SCREEN_GO, features
from flymon.brain.l_spec import SPEC, LSpec, smoke

NEED = ("stage0", "oc", "stage1")
PROVENANCE_FILES = ("flymon/brain/l_pairs.py", "flymon/brain/l_screen.py", "flymon/brain/l_rules.py",
                    "flymon/brain/l_measure.py", "flymon/brain/l_jobs.py", "flymon/brain/l_spec.py")


def world(npz):
    """(connectome, populations) of the declared connectome."""
    conn = Connectome.load(npz)
    return conn, Populations.from_connectome(conn)


def first_even(pops, spec, c3_json) -> tuple:
    """(H.4's first even (b) pair, its row of the pinned ceiling file); ValueError on another even list or no row."""
    even = even_pairs(pops)
    d = pairs_digest(even)
    if d != spec.j.h4.pairs_digest:
        raise ValueError(f"the even pair list differs from H.4's ({d[:12]} != {spec.j.h4.pairs_digest[:12]})")
    first = next(p for p in even if p["axis"] == "b")
    row = next((e for e in load_even(spec, c3_json) if tuple(e["key"]) == tuple(pair_key(first))), None)
    if row is None:
        raise ValueError(f"the ceiling file has no row for the first even (b) pair {pair_key(first)}")
    return first, row


def naive_mismatch(got: dict, row: dict, n_kc: int) -> list:
    """The fields of naive_job's result that differ from the ceiling row: fx / fy bit for bit, pre exactly."""
    out = [f"f{s}" for s in ("x", "y")
           if not np.array_equal(dense(got[f"f{s}_idx"], got[f"f{s}_val"], n_kc), np.asarray(row[f"f{s}"], float))]
    return out + (["pre"] if got["pre"] != row["pre"] else [])


def main(argv=None, spec: LSpec | None = None, summary_spec: LSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
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
    doc, why = read_previous(a.summary, NEED, a.smoke, check_committed, out_allowed)
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
    g1 = doc["stage1"].get("gate") or {}
    if g1.get("outcome") != SCREEN_GO:
        return refuse(f"block stage1's outcome is {g1.get('outcome')}, not {SCREEN_GO}: stage 2 does not run")
    rule = (g1.get("final") or {}).get("rule")
    if not rule or "family" not in rule:
        return refuse("block stage1 has no final rule")
    m0d_path = a.m0d or spec.m0d_path
    try:
        c3, readout, z, pools_h4 = load_c3_record(m0d_path, spec)
    except ValueError as e:
        return refuse(str(e))
    c3_json = params_json(c3)
    types = [t for k in ("A", "P") for t in pools_h4[k]]
    conn, pops = world(a.npz)
    n_kc = len(pops.kc)
    try:
        s = new_set(pops, spec, summary_spec)                 # check_digests before the screen order exists
    except ValueError as e:
        return refuse(f"the new pair set is not the pinned one: {e}")
    try:
        first, row = first_even(pops, spec, c3_json)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"naive self-check input: {e}")
    ro = dict(spec.readout)
    w13 = readout_weights(conn, pops, ro["A"], spec.k.min_weight)
    w05 = readout_weights(conn, pops, ro["P"], spec.k.min_weight)
    b = s["b"]
    limit = min(spec.max_screened, len(b))
    t0 = time.time()
    print(f"run {rid}: rule {rule_text(rule)}, {len(b)} new (b) pairs, screening up to {limit} for {spec.n_pass} passes",
          flush=True)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, stage1_run_id=doc["stage1"].get("run_id"), c3=c3_json, rule=rule,
               rule_text=rule_text(rule), b_digest=summary_spec.b_digest, a_digest=summary_spec.a_digest,
               n_b=len(b), n_a=len(s["a"]), skipped=s["skipped"])
    feats, status, seen = [], "done", []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        lm = LMeasurer(pool, spec, types, cache)
        seen.append(lm)
        got = lm.naive(c3, [first])[0]
        bad = naive_mismatch(got, row, n_kc)
        res["selfcheck_naive"] = dict(ok=not bad, mismatched=bad, pair=list(pair_key(first)),
                                      note="l_jobs.naive_job on the first even (b) pair against the pinned ceiling "
                                           "row: fx / fy np.array_equal, select-seed pre equal (plan reading 11)")
        if bad:
            status = "mismatch_naive"
        else:
            n = max(1, pool.n_workers)
            while len(feats) < limit:
                batch = b[len(feats):min(len(feats) + n, limit)]
                for r in lm.naive(c3, batch):
                    feats.append(features(dense(r["fx_idx"], r["fx_val"], n_kc), dense(r["fy_idx"], r["fy_val"], n_kc),
                                          r["pre"], w13, w05, spec))
                sc = screen_order(feats, rule, spec.n_pass, spec.max_screened)
                print(f"measured {len(feats)}/{limit}: {len(sc['passed'])} passes ({(time.time() - t0) / 60:.1f} min)",
                      flush=True)
                if sc["outcome"] == SCREENED:
                    break
    gp = guard_params(Params(), seen)
    if status == "done":
        sc = screen_order(feats, rule, spec.n_pass, spec.max_screened)
        passed = set(sc["passed"])
        failed = [i for i in sc["screened"] if i not in passed]
        lift = lift_draw(failed, spec.n_lift, spec.lift_seed) if sc["outcome"] == SCREENED else []
        res.update(outcome=sc["outcome"], measured=len(feats), n_star=sc["n_star"], coverage=sc["coverage"],
                   screened=[dict(key=list(pair_key(b[i])), feat=feats[i], **{"pass": i in passed})
                             for i in sc["screened"]],
                   passed=[list(pair_key(b[i])) for i in sc["passed"]],
                   lift=[list(pair_key(b[i])) for i in lift],
                   lift_rng=f"numpy.random.default_rng({spec.lift_seed}) over the screened failing pairs",
                   measured_note="pairs measured after n* in the last batch are cached but neither screened nor "
                                 "counted (plan reading 10)",
                   sentence=None if sc["outcome"] == SCREENED else sentence(COVERAGE_SHORT, dict(
                       rule=rule_text(rule), n_pass=len(sc["passed"]), n_screened=len(sc["screened"]),
                       floor=spec.n_pass / spec.max_screened)))
    files = [ROOT / f for f in PROVENANCE_FILES] + [m0d_path, a.summary]
    res["provenance"] = provenance(files, vars(a), sys.argv[1:] if argv is None else argv)
    res["provenance"]["summary_at_head"] = dict(path=a.summary, sha256=head_sha256(a.summary))
    res["status"] = status
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = l_store.write_json(out / "runs" / f"{rid}-stage2.json", res, gp)
    lines = [f"# L.11.3 stage 2 {rid}\n\n", f"**status: {status}**\n\n",
             f"- naive self-check {res['selfcheck_naive']['pair']}: {res['selfcheck_naive']['ok']} "
             f"{res['selfcheck_naive']['mismatched']}\n"]
    if status == "done":
        lines.append(f"- {res['outcome']}: rule {res['rule_text']}, {len(res['passed'])} passes, n* {res['n_star']}, "
                     f"coverage {res['coverage']:.4g}, screened {len(res['screened'])}, measured {res['measured']}, "
                     f"lift {len(res['lift'])}\n")
        if res["sentence"]:
            lines.append(f"\n{res['sentence']}\n")
    l_store.write_bytes(out / "runs" / f"{rid}-stage2.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status} in {res['wall_s'] / 60:.1f} min", flush=True)
    if status != "done":
        print(f"naive self-check DIFFERS {res['selfcheck_naive']['mismatched']}: stop and ask the user", flush=True)
        return 5
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage2", res, report, gp)
        print(f"wrote {a.summary} (block stage2)", flush=True)
    if res["outcome"] == SCREENED:
        print(f"{SCREENED}: coverage {res['coverage']:.4g}; next: commit results/summary/l_screen.json before stage 3",
              flush=True)
    else:
        print(f"{COVERAGE_SHORT}: {res['sentence']} — no stage 3 follows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
