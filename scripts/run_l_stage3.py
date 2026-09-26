#!/usr/bin/env python3
"""Spec L.11.3 / L.5, stage 3: H.4's oracle (h4_jobs.oracle_job unchanged, through H4Measurer, on C3's recorded readout,
z and pools) on the n_pass screened (b) pairs, the fixed (a) pairs of the new set's first a_turns turns and the lift
sample; the judgement in J.12.9's bands.

    uv run python scripts/run_l_stage3.py                                   # ~50 min (resumable: rerun it)
    uv run python scripts/run_l_stage3.py --smoke --allow-dirty --workers 4 --summary results/m0d/l/smoke/l_screen.json

Run it from the repository root. Refused before the pool, in this order, unless: the hashed files are clean (or
--allow-dirty); the summary is git-tracked and unchanged against HEAD (outside --smoke; a --smoke run reads only a
smoke summary under results/m0d/l/); blocks "stage0", "oc", "stage1" and "stage2" are present; each was produced under
this code's measure key and procedure manifest; stage 2's outcome is SCREENED with stage 1's rule and the pinned pair
digests; C3, its readout, z and pools load from results/summary/m0d.json; the regenerated new pair lists carry the
pinned digests (l_pairs.check_digests); every recorded passed / lift pair is in them and there are n_pass passes.
Judged = the n_pass passes + the fixed (a) pairs (plan reading 12); the lift pairs are measured by the same oracle and
recorded, never judged. Then: l_rules.pair_rows_stats over the judged rows (any reason -> INVALID: report only, exit 3);
h4_formula.arm_aggregate(stats, naive_max, t_b_min, f_a_min); l_rules.stage3_reading (J.12.9's bands); the lift's
testable k / n; l_rules.diversity over the passes (my species and the original opponent's types from
l_pairs.new_turns); the closing sentence; the attempt history I-L (spec.attempts).
Report: <out>/runs/<run id>-stage3.{json,md}. Block "stage3" only for a complete, clean, non-smoke run on the declared
configuration. Exit codes: 0 read, 2 refused, 3 invalid oracle rows (recorded; no block).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from flymon.brain import j_store, l_store  # noqa: F401  (j_store: the C3 loader l_cli.load_c3_record uses)
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_formula import arm_aggregate
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import pair_key, pool_vocabulary
from flymon.brain.j_rules import INVALID
from flymon.brain.j_runner import params_json
from flymon.brain.l_cli import (POOL_TIMEOUT_S, check_committed, code_keys, git_state, guard_params, head_sha256,
                                load_c3_record, new_set, other_code, out_allowed, provenance, read_previous, refuse,
                                rule_text, run_id, same_code, write_block)
from flymon.brain.l_measure import HASHED_FILES
from flymon.brain.l_pairs import new_turns
from flymon.brain.l_rules import SCREENED, diversity, pair_rows_stats, sentence, stage3_reading
from flymon.brain.l_spec import SPEC, LSpec, smoke

NEED = ("stage0", "oc", "stage1", "stage2")
PROVENANCE_FILES = ("flymon/brain/l_pairs.py", "flymon/brain/l_rules.py", "flymon/brain/h4_formula.py",
                    "flymon/brain/j_rules.py", "flymon/brain/l_spec.py")


def world(npz):
    """(connectome, populations) of the declared connectome."""
    conn = Connectome.load(npz)
    return conn, Populations.from_connectome(conn)


def turn_info(declared) -> dict:
    """{turn: new_turns' record} of the declared new set (my species and the original opponent's types)."""
    species_types, move_info, _, _ = pool_vocabulary()
    return {t["turn"]: t for t in new_turns(species_types, move_info, declared.n_turns, declared.rng_seed)}


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
    s2 = doc["stage2"]
    if s2.get("outcome") != SCREENED:
        return refuse(f"block stage2's outcome is {s2.get('outcome')}, not {SCREENED}: stage 3 does not run")
    rule = ((doc["stage1"].get("gate") or {}).get("final") or {}).get("rule")
    if not rule or s2.get("rule") != rule:
        return refuse(f"block stage2's rule {s2.get('rule')} is not block stage1's final rule {rule}")
    if (s2.get("b_digest"), s2.get("a_digest")) != (summary_spec.b_digest, summary_spec.a_digest):
        return refuse(f"block stage2's pair digests ({str(s2.get('b_digest'))[:12]}, {str(s2.get('a_digest'))[:12]}) "
                      f"are not the pinned ones")
    m0d_path = a.m0d or spec.m0d_path
    try:
        c3, readout, z, pools_h4 = load_c3_record(m0d_path, spec)
    except ValueError as e:
        return refuse(str(e))
    c3_json = params_json(c3)
    conn, pops = world(a.npz)
    try:
        s = new_set(pops, spec, summary_spec)
    except ValueError as e:
        return refuse(f"the new pair set is not the pinned one: {e}")
    by = {tuple(pair_key(p)): p for p in s["b"]}
    passed = [("b", int(k[1]), k[2], k[3]) for k in s2.get("passed") or []]
    lift = [("b", int(k[1]), k[2], k[3]) for k in s2.get("lift") or []]
    if len(passed) != spec.n_pass:
        return refuse(f"block stage2 has {len(passed)} passed pairs, not n_pass {spec.n_pass}")
    unknown = [list(k) for k in passed + lift if k not in by]
    if unknown:
        return refuse(f"block stage2 names pairs that are not in the new (b) set: {unknown}")
    if len(lift) > spec.n_lift or set(lift) & set(passed):
        return refuse(f"block stage2's lift sample ({len(lift)} pairs) is not at most {spec.n_lift} failing pairs")
    judged = [by[k] for k in passed] + list(s["a"])
    lift_pairs = [by[k] for k in lift]
    t0 = time.time()
    print(f"run {rid}: oracle on {len(passed)} passed (b) + {len(s['a'])} (a) judged, {len(lift)} lift pairs", flush=True)
    seen = []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        m4 = H4Measurer(pool, spec.j.h4, judged + lift_pairs, pools_h4, cache, None)
        seen.append(m4)
        rows = m4.oracle(c3, readout, z)
    gp = guard_params(Params(), seen)
    jr, lr = rows[:len(judged)], rows[len(judged):]
    st = pair_rows_stats(jr, z, [pair_key(p) for p in judged], spec.j.h4)
    ls = pair_rows_stats(lr, z, lift, spec.j.h4)
    lk = sum(bool((ls["stats"].get(k) or {}).get("testable")) for k in lift)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, stage2_run_id=s2.get("run_id"), c3=c3_json, readout=readout, z=z,
               pools=pools_h4, rule=rule, rule_text=rule_text(rule), coverage=s2.get("coverage"),
               judged=dict(b=[list(k) for k in passed], a=[list(pair_key(p)) for p in s["a"]]),
               rows_stats=st["pairs"], reasons=st["reasons"],
               lift=dict(n=len(lift), k=lk, reasons=ls["reasons"], pairs=ls["pairs"],
                         note="the lift sample (decision 12): same oracle, recorded, never judged"),
               attempts=[dict(claim=c, verdict=v, commit=h) for c, v, h in spec.attempts])
    turns = turn_info(summary_spec)
    res["diversity"] = diversity([dict(x=by[k]["x"], me=turns[k[1]]["me"], opp_types=turns[k[1]]["opp_types"])
                                  for k in passed])
    if st["reasons"]:
        status = "invalid"
        res.update(aggregate=None, reading=dict(outcome=INVALID, band=None, reasons=st["reasons"]), sentence=None)
    else:
        status = "done"
        h4 = spec.j.h4
        agg = arm_aggregate(st["stats"], h4.naive_max, h4.t_b_min, h4.f_a_min)
        rd = stage3_reading(agg, spec)
        dv = res["diversity"]
        ctx = dict(rule=rule_text(rule), coverage=float(s2.get("coverage") or 0.0), n=agg["testable_b"], n_b=agg["n_b"],
                   k=agg["F_a"], m=lk, n_lift=len(lift), naive_a=rd["naive_a"], f_a_possible=rd["f_a_possible"],
                   p=dv["n_me"], q=dv["n_moves"], n_opp_types=dv["n_opp_types"], top_move_share=dv["top_move_share"],
                   spec=spec)
        res.update(aggregate=agg, reading=rd, sentence=None if rd["band"] is None else sentence(rd["band"], ctx))
    res["status"] = status
    files = [ROOT / f for f in PROVENANCE_FILES] + [m0d_path, a.summary]
    res["provenance"] = provenance(files, vars(a), sys.argv[1:] if argv is None else argv)
    res["provenance"]["summary_at_head"] = dict(path=a.summary, sha256=head_sha256(a.summary))
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = l_store.write_json(out / "runs" / f"{rid}-stage3.json", res, gp)
    rd = res["reading"]
    lines = [f"# L.11.3 stage 3 {rid}\n\n", f"**status: {status}; band {rd.get('band')}**\n\n",
             f"- rule {res['rule_text']}, coverage {res['coverage']}\n",
             f"- lift sample: {lk}/{len(lift)} testable\n", f"- diversity: {res['diversity']}\n"]
    if res["aggregate"]:
        ag = res["aggregate"]
        lines.append(f"- testable (b) {ag['testable_b']}/{ag['n_b']}, F_a {ag['F_a']} (naive (a) {ag['naive_a']}/"
                     f"{ag['n_a']})\n")
    if res["sentence"]:
        lines.append(f"\n{res['sentence']}\n")
    if st["reasons"]:
        lines.append(f"\nINVALID: {st['reasons']}\n")
    l_store.write_bytes(out / "runs" / f"{rid}-stage3.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status}, band {rd.get('band')} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec,
                    invalid=status != "done")
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage3", res, report, gp)
        print(f"wrote {a.summary} (block stage3): {res['sentence']}", flush=True)
    return 3 if status == "invalid" else 0


if __name__ == "__main__":
    sys.exit(main())
