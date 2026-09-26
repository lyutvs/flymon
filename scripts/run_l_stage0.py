#!/usr/bin/env python3
"""Spec L.11.1, stage 0: C3's oracle labels on the 20 odd-turn (b) pairs (h4_jobs.oracle_job unchanged, through
H4Measurer, on H.4's recorded C3 readout / z / pools), the 41 stage-1 feature records, three self-checks and D.6 on C3.

    uv run python scripts/run_l_stage0.py                                  # stage 0 (~55 min; resumable: rerun it)
    uv run python scripts/run_l_stage0.py --smoke --allow-dirty --workers 4  # minutes (2 odd pairs, 2 D.6 seeds)

Run it from the repository root. Refused first (outside --smoke) when --summary already holds a later block (oc, stage1,
...): block stage0 is never rewritten under them. Inputs, each refused before the pool starts: the declared connectome,
C3 from block "h3" of results/summary/m0d.json, H.4a.8's ceiling raw file and K's C3 activity cache by their pinned
sha256 (--ceiling / --k-act replace the paths, never the hashes), block "h4"'s C3 readout {A: MBON13, P: MBON05} with
its z, the odd pair list by SPEC.k.odd_pairs_digest. Self-checks: (i) both raw files load as C3's engine on the declared
seeds and pairs; (iii) the oracle re-run on the first even (b) pair reproduces H.4's recorded statistics bit for bit (a
recorded code-identity check, never a label); (ii) every odd pair's oracle KC fire fraction and spike count per act
seed equal the K cache's.
Labels: h4_formula.pair_stats on the report seeds (l_rules.pair_rows_stats' row checks); features:
l_screen.features on the select seeds (even from the ceiling file with H.4's labels, odd from the K cache with the
oracle's select.pre). D.6: j_runner.measure_d6 on C3 (d6a.SEEDS; a record, not a judgement).

Report: <out>/runs/<run id>-stage0.{json,md}. Block "stage0" of results/summary/l_screen.json is written only by a
complete run (no --smoke / --pairs / --no-d6, clean hashed files, the declared configuration, valid rows).
Exit codes: 0 done, 2 refused before measuring, 3 invalid oracle rows (recorded; no block), 5 self-check (ii) or (iii)
mismatch (recorded; stop and ask the user).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from flymon.brain import d6a, j_store, l_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_formula import pair_stats
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import pair_key, pairs_digest
from flymon.brain.j_measure import JMeasurer
from flymon.brain.j_rules import pair_mismatch
from flymon.brain.j_runner import measure_d6, params_json
from flymon.brain.j_setup import build
from flymon.brain.k_measure import KMeasurer
from flymon.brain.k_metrics import readout_weights
from flymon.brain.k_pairs import odd_pairs
from flymon.brain.l_cli import (POOL_TIMEOUT_S, code_keys, git_state, guard_params, later_blocks, out_allowed, refuse,
                                run_id, write_block)
from flymon.brain.l_measure import HASHED_FILES, check_pinned, load_even, load_odd_activity
from flymon.brain.l_rules import pair_rows_stats
from flymon.brain.l_screen import features
from flymon.brain.l_spec import SPEC, LSpec, smoke

SMOKE_ODD = 2          # --smoke's default --pairs: the first two odd pairs


def kc_mismatch(rows: list, k_rows: list, index: list, n_kc: int, seeds: list) -> list:
    """Self-check (ii): per odd pair and side, the oracle's kc.{x,y}.frac / spikes on each act seed against the K cache's
    len(kc) / n_kc and sum(n) for that odour and seed (index: KMeasurer.index, one (ix, iy) per row)."""
    per = {(int(r["i"]), int(r["seed"])): r for r in k_rows}
    out = []
    for row, ij in zip(rows, index):
        for side, i in zip(("x", "y"), ij):
            want = dict(frac=[len(per[(i, s)]["kc"]) / int(n_kc) for s in seeds],
                        spikes=[int(sum(per[(i, s)]["n"])) for s in seeds])
            for f in ("frac", "spikes"):
                if list(row["kc"][side][f]) != want[f]:
                    out.append(dict(key=list(pair_key(row)), side=side, field=f, got=list(row["kc"][side][f]),
                                    want=want[f]))
    return out


def _select_ok(pre: dict, spec) -> bool:
    n = len(spec.j.h4.select_seeds)
    return all(t in pre and len(pre[t]) == n and all(len(xy) == 2 for xy in pre[t]) for t in spec.guard_types)


def main(argv=None, spec: LSpec | None = None, summary_spec: LSpec = SPEC, require_root: bool = True,
         pools: dict | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=l_store.SUMMARY)
    ap.add_argument("--m0d", default=None)
    ap.add_argument("--ceiling", default=None)                 # another path to the pinned ceiling file (hash checked)
    ap.add_argument("--k-act", default=None)                   # another path to the pinned K activity cache (hash checked)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)         # verification only: the first N odd pairs
    ap.add_argument("--no-d6", action="store_true")            # verification only
    a = ap.parse_args(argv)
    if a.pairs is not None and a.pairs < 1:
        return refuse(f"--pairs must be at least 1, got {a.pairs}")
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    if a.ceiling:
        spec = dataclasses.replace(spec, ceiling_path=a.ceiling)
    if a.k_act:
        spec = dataclasses.replace(spec, k_act_path=a.k_act)
    n_pairs = a.pairs or (SMOKE_ODD if a.smoke else None)
    out = Path(a.out or ("results/m0d/l/smoke" if a.smoke else "results/m0d/l/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {l_store.ALLOWED_DIR} of the repository root")
    rid = run_id()
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    why = None if a.smoke else later_blocks(a.summary, "stage0")
    if why:
        return refuse(why)
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    code, manifest = code_keys(a.npz)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    m0d_path = a.m0d or spec.m0d_path
    try:
        m0d = json.loads(Path(m0d_path).read_text())
        c3, _, _ = j_store.load_c3(m0d, spec.j.h3_block, spec.j.c3_name)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable C3 in block {spec.j.h3_block} of {m0d_path}: {e}")
    c3_json = params_json(c3)
    try:
        k_raw = check_pinned(spec.k_act_path, spec.k_act_sha256)
        check_pinned(spec.ceiling_path, spec.ceiling_sha256)
    except (OSError, ValueError) as e:
        return refuse(f"a pinned input is not usable (L.11.1 pins it by sha256): {e}")
    try:
        rec = m0d["h4"]["h4"]["combos"][spec.j.c3_name]
        readout, z, pools_h4 = rec["readout"], rec.get("z"), m0d["h4"]["pools"]
    except (KeyError, TypeError) as e:
        return refuse(f"block h4 of {m0d_path} has no C3 record: {e!r}")
    if readout != dict(spec.readout):
        return refuse(f"block h4's C3 readout {readout} is not the declared {dict(spec.readout)}")
    if not z or any(k not in z or len(z[k]) != 2 for k in ("A", "P")):
        return refuse(f"block h4's C3 z is missing or malformed: {z}")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    odd = odd_pairs(pops)
    digest = pairs_digest(odd)
    if digest != spec.k.odd_pairs_digest:
        return refuse(f"the odd pair list differs from the declared one ({digest[:12]} != {spec.k.odd_pairs_digest[:12]})")
    try:                                                                            # self-check (i)
        even = load_even(spec, c3_json)
        odd_act = load_odd_activity(spec, odd, len(pops.kc), c3_json)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"self-check (i): {e}")
    labels = {tuple(pair_key(p)): bool(p["testable"]) for p in rec["oracle"]["pairs"] if p["axis"] == "b"}
    if sorted(e["key"] for e in even) != sorted(labels):
        return refuse(f"self-check (i): the ceiling file's {len(even)} even (b) pairs are not H.4's {len(labels)}")
    t0 = time.time()
    s = build(a.npz, spec.j, out, log=lambda x: print(x, flush=True), pools=pools)
    if pools is None and m0d[spec.j.h3_block].get("pools") != s["core"]:
        return refuse(f"the core pools {s['core']} differ from block {spec.j.h3_block}'s")
    if pools is None and s["pairs_digest"] != spec.j.h4.pairs_digest:
        return refuse(f"the even pair list differs from H.4's ({s['pairs_digest'][:12]})")
    pools_h4 = pools or pools_h4
    run_odd = odd[:n_pairs] if n_pairs else odd
    first = [p for p in s["pairs"] if p["axis"] == "b"][:1]
    print(f"run {rid}: L key {code['key'][:12]}, {len(run_odd)} odd (b) pairs", flush=True)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, c3=c3_json, readout=readout, z=z, pools=pools_h4,
               inputs=dict(ceiling=dict(path=spec.ceiling_path, sha256=spec.ceiling_sha256),
                           k_act=dict(path=spec.k_act_path, sha256=spec.k_act_sha256),
                           m0d=dict(path=m0d_path, sha256=sha256_file(m0d_path))),
               odd_pairs_digest=digest, n_odd=len(run_odd),
               checks=dict(i=dict(ok=True, even=len(even), odd=len(odd_act),
                                  note="ceiling and K cache load as C3's engine on the declared seeds, pairs and "
                                       "odour order (l_measure.load_even / load_odd_activity)")))
    status, seen = "done", []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        m4e = H4Measurer(pool, spec.j.h4, first, pools_h4, cache, None)             # self-check (iii)
        seen.append(m4e)
        row = m4e.oracle(c3, readout, z)[0]
        got = pair_stats(row["report"], z, spec.j.h4.testable_min)
        key = tuple(pair_key(first[0]))
        want = next(p for p in rec["oracle"]["pairs"] if tuple(pair_key(p)) == key)
        bad = pair_mismatch(got or {}, want)
        res["checks"]["iii"] = dict(ok=not bad, pair=list(key), got=got, want=want, mismatched=bad,
                                    note="h4_jobs.oracle_job unchanged, re-run on the first even (b) pair against "
                                         "H.4's record: a code-identity check, not used for labels (ruling S11)")
        if bad:
            status = "mismatch_iii"
        else:
            m4o = H4Measurer(pool, spec.j.h4, run_odd, pools_h4, cache, None)
            seen.append(m4o)
            rows = m4o.oracle(c3, readout, z)
            km = KMeasurer(None, None, odd, len(pops.kc), None)
            mism = kc_mismatch(rows, k_raw["result"], km.index[:len(rows)], len(pops.kc), spec.j.h4.act_seeds)
            res["checks"]["ii"] = dict(ok=not mism, n_pairs=len(rows), mismatched=mism)
            st = pair_rows_stats(rows, z, [pair_key(p) for p in run_odd], spec.j.h4)
            short = [list(pair_key(r)) for r in rows if not _select_ok(r["select"]["pre"], spec)]
            reasons = st["reasons"] + ([f"{len(short)} pairs whose select probes are not one [x, y] per declared "
                                        f"select seed for {list(spec.guard_types)}"] if short else [])
            res["reasons"] = reasons
            res["odd"] = [dict(key=list(pair_key(r)), testable=None if st["stats"].get(pair_key(r)) is None
                               else st["stats"][pair_key(r)]["testable"],
                               **{k: (st["stats"].get(pair_key(r)) or {}).get(k) for k in ("d_pre", "r", "p", "m")},
                               select_pre=r["select"]["pre"], report_pre=r["counts"]["pre"]) for r in rows]
            if mism:
                status = "mismatch_ii"
            elif reasons:
                status = "invalid"
            else:
                ro = dict(spec.readout)
                w13 = readout_weights(s["conn"], s["pops"], ro["A"], spec.k.min_weight)
                w05 = readout_weights(s["conn"], s["pops"], ro["P"], spec.k.min_weight)
                feats = [dict(key=list(e["key"]), turn=int(e["key"][1]), set="even", testable=labels[e["key"]],
                              feat=features(e["fx"], e["fy"], e["pre"], w13, w05, spec)) for e in even]
                feats += [dict(key=o["key"], turn=int(o["key"][1]), set="odd", testable=bool(o["testable"]),
                               feat=features(f["fx"], f["fy"], o["select_pre"], w13, w05, spec))
                          for o, f in zip(res["odd"], odd_act)]
                res["feats"] = feats
                if not a.no_d6:
                    jm = JMeasurer(pool, spec.j, s["all51"], cache)
                    seen.append(jm)
                    res["d6"] = measure_d6(jm, s["ctx"], c3, seeds=d6a.SEEDS[:2] if a.smoke else d6a.SEEDS,
                                           judged=not a.smoke)
    gp = guard_params(Params(), seen)
    res["status"] = status
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = l_store.write_json(out / "runs" / f"{rid}-stage0.json", res, gp)
    ch = res["checks"]
    lines = [f"# L.11.1 stage 0 {rid}\n\n", f"**status: {status}**\n\n",
             f"- self-check (i): {ch['i']['ok']}\n",
             f"- self-check (iii) first even (b) pair {ch['iii']['pair']}: {ch['iii']['ok']} {ch['iii']['mismatched']}\n"]
    if "ii" in ch:
        lines.append(f"- self-check (ii) {ch['ii']['n_pairs']} odd pairs: {ch['ii']['ok']} "
                     f"({len(ch['ii']['mismatched'])} mismatches)\n")
    if res.get("odd") is not None:
        lines.append(f"\nodd testable: {sum(bool(o['testable']) for o in res['odd'])}/{len(res['odd'])}; "
                     f"reasons: {res.get('reasons')}\n")
    if res.get("feats"):
        lines.append(f"\nfeature records: {len(res['feats'])}, G passes {sum(f['feat']['G'] for f in res['feats'])}\n")
    if res.get("d6"):
        lines.append(f"\nD.6 on C3 (a record): a {res['d6']['a']}, b {res['d6']['b']}\n")
    l_store.write_bytes(out / "runs" / f"{rid}-stage0.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status} in {res['wall_s'] / 60:.1f} min", flush=True)
    if status.startswith("mismatch"):
        print(f"self-check ({status.split('_')[1]}) DIFFERS: stop and ask the user", flush=True)
        return 5
    blockers = dict(smoke=a.smoke, pairs=a.pairs is not None, no_d6=a.no_d6, dirty=bool(git["dirty_hashed"]),
                    spec=spec != summary_spec, invalid=status != "done")
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage0", res, report, gp)
        print(f"wrote {a.summary} (block stage0)", flush=True)
    return 3 if status == "invalid" else 0


if __name__ == "__main__":
    sys.exit(main())
