#!/usr/bin/env python3
"""Spec M.10.3 (+ M.10.7, plan readings 2, 6, 10, 11 and 16), stage 1: the per-arm scan on the even (b) pairs.

- First job, a code-identity self-check: h4_jobs.oracle_job unchanged (H4Measurer) on the first even (b) pair with C3's
  single readout (MBON13 / MBON05, block h4's z) against block h4's recorded C3 pair (j_rules.pair_mismatch) — else
  exit 5, no block. (edit_job refuses a readout narrower than the edited core by design, so C3 cannot go through it.)
  Block h4's C3 counts r >= 2 and -p >= 2 on the scanned pairs are recorded as the baseline.
- pre_job once per pair (KC activity and unedited probes, shared by every candidate: M.10.3's reuse), then edit_job per
  pair for every guard-passing candidate of block stage0 against the fixed arm's incumbent POPULATION readout (reward
  candidate c: reward = c, punish = PPL105, readout {A: PPL105, P: c}; punish candidate c: reward = PAM08, punish = c,
  readout {A: c, P: PAM08}; z from block stage0), plus the incumbent combination itself (PAM08 x PPL105, readout
  {A: PPL105, P: PAM08}) once, the reference row of both arms (not ranked). A scanned candidate whose core cells overlap
  the other arm's incumbent core is skipped and recorded (reading 2). Each edit row's `edited` (edited cells == the
  readout group's cells) is logged per entry.
- Per entry: n (pairs with r >= 2, reward; -p >= 2, punish; undefined = not passing), median, per-pair r / p; ranking
  (n, median, the core's KC input from block spec_check), top 2 per arm (an arm with no scanned candidate: [incumbent]);
  the predicted joint counts over every reward x punish entry incl. the reference (a record, not a judgement).

    uv run python scripts/run_m_stage1.py                                        # hours (resumable: rerun it)
    uv run python scripts/run_m_stage1.py --smoke --allow-dirty --workers 4 --summary results/m0d/m/smoke/m_readout.json

Run it from the repository root. Refusals (exit 2) before the pool starts, in order: another directory, --out outside
results/m0d/m/, dirty hashed files (unless --allow-dirty), a summary that is not git-tracked and clean at HEAD (outside
--smoke; --smoke reads only a smoke summary under results/m0d/m/) or lacks any of the blocks spec_check, stage0, oc
(each listed), no connectome file, any of them produced under other code, a run-id chain break (stage0 on another
spec_check, oc on another stage0), a later block in the summary (outside --smoke), block stage0's status other than
"done", another connectome, an m0d.json other than the one stage 0 ran on (sha256) or a C3 record differing from block
stage0's, groups of block stage0 that are missing, have no z or whose cells differ from block spec_check's digests,
another even pair list (H.4's digest), other MBON cells than stage 0 measured.
Report: <out>/runs/<run id>-stage1.{json,md}. Block "stage1" (not committed before stage 2: stage 2 only needs it
present under the same code), outside --smoke only with clean hashed files and the declared spec; a --smoke run writes
it into its smoke summary (the L convention for chaining smoke stages). Exit codes: 0 done, 2 refused, 5 self-check
mismatch (recorded; stop and ask the user).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from flymon.brain import m_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_formula import pair_stats
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import even_pairs, pair_key, pairs_digest
from flymon.brain.j_rules import pair_mismatch
from flymon.brain.j_runner import params_json
from flymon.brain.m_cands import cells_digest, overlap
from flymon.brain.m_cli import (POOL_TIMEOUT_S, check_committed, code_keys, git_state, guard_params, later_blocks,
                                load_c3_record, other_code, out_allowed, read_previous, refuse, run_id, run_id_chain,
                                same_code, stage0_inputs, write_block)
from flymon.brain.m_measure import HASHED_FILES, MMeasurer
from flymon.brain.m_rules import arm_rows, predicted_joint, rank, top
from flymon.brain.m_spec import SPEC, MSpec, smoke

NEED = ("spec_check", "stage0", "oc")
DONE = "done"                   # block stage0's status stage 1 requires
MISMATCH = "mismatch"
OTHER = {"reward": "punish", "punish": "reward"}


def c3_baseline(pairs_h4: list, keys: list, tmin: float) -> dict:
    """Block h4's recorded C3 counts on the scanned pairs: r >= tmin, -p >= tmin (None = not passing), testable."""
    want = {tuple(k) for k in keys}
    got = [p for p in pairs_h4 if tuple(pair_key(p)) in want]
    ok = lambda v: v is not None and v >= tmin
    return dict(n_pairs=len(got), n_r=sum(ok(p.get("r")) for p in got),
                n_p=sum(ok(None if p.get("p") is None else -p["p"]) for p in got),
                testable=sum(bool(p.get("testable")) for p in got),
                note="block h4's C3 oracle (single readout MBON13 / MBON05) on these pairs: the baseline of the scan")


def edited_log(rows: list, groups: dict) -> dict:
    """The edit rows' `edited` ({arm: {group, cells}}), and whether every row edited exactly its readout group's cells."""
    first = rows[0]["edited"] if rows else {}
    ok = all(r["edited"] == first for r in rows) and all(
        sorted(int(c) for c in e["cells"]) == sorted(int(c) for c in groups[e["group"]]) for e in first.values())
    return dict(edited=first, edited_ok=bool(ok and rows))


def entry(name: str, arm: str, g: dict, rows: list, z: dict, groups: dict, kc_input: float, h4spec) -> dict:
    ar = arm_rows([r["report"] for r in rows], z, arm, h4spec)
    return dict(name=name, arm=arm, cells=list(g["cells"]), digest=g["digest"], kc_input=float(kc_input), **ar,
                **edited_log(rows, groups))


def main(argv=None, spec: MSpec | None = None, summary_spec: MSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=m_store.SUMMARY)
    ap.add_argument("--m0d", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/m/smoke" if a.smoke else "results/m0d/m/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {m_store.ALLOWED_DIR} of the repository root")
    rid = run_id()
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    doc, why = read_previous(a.summary, list(NEED), a.smoke, check_committed)
    if why:
        return refuse(why)
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    try:
        code, manifest = code_keys(a.npz)
    except SystemExit as e:                                  # code_keys named the missing hashed files
        return int(e.code or 2)
    why = (other_code(doc, list(NEED), code["key"], manifest["key"], same_code)
           or run_id_chain(doc, "stage0", {"spec_check_run_id": "spec_check"})
           or run_id_chain(doc, "oc", {"stage0_run_id": "stage0"})
           or (None if a.smoke else later_blocks(a.summary, "stage1", doc)))
    if why:
        return refuse(why)
    sc, s0 = doc["spec_check"], doc["stage0"]
    if s0.get("status") != DONE:
        return refuse(f"block stage0's status is {s0.get('status')}: stage 1 runs only after stage 0 is {DONE!r}")
    npz_sha = code["files"].get("npz:" + Path(a.npz).name)
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({str(npz_sha)[:12]})")
    m0d_path = a.m0d or spec.m0d_path
    why = stage0_inputs(s0, m0d_path)
    if why:
        return refuse(why)
    try:
        c3, readout, z_c3, pools_h4 = load_c3_record(m0d_path, spec.l)
        pairs_h4 = json.loads(Path(m0d_path).read_text())["h4"]["h4"]["combos"][spec.j.c3_name]["oracle"]["pairs"]
    except (ValueError, KeyError, TypeError) as e:
        return refuse(str(e) if isinstance(e, ValueError) else f"block h4 of {m0d_path} has no C3 oracle pairs: {e!r}")
    why = stage0_inputs(s0, m0d_path, c3=params_json(c3), readout=readout, z_c3=z_c3, pools=pools_h4)
    if why:
        return refuse(why)
    # ---- the groups of block stage0, pinned by block spec_check's digests (M.10.7) --------------------------------
    inc = {"reward": spec.incumbent_reward, "punish": spec.incumbent_punish}
    arms = s0.get("arms") or {}
    groups = {g.get("name"): g for g in s0.get("groups") or []}
    pinned = {c.get("name"): c for arm in ("reward", "punish") for c in (sc.get("candidates") or {}).get(arm, [])}
    names = list(inc.values()) + [n for arm in ("reward", "punish") for n in arms.get(arm, [])]
    bad = [n for n in names if n not in groups or groups[n].get("z") is None or n not in pinned
           or cells_digest(groups[n].get("cells") or []) != groups[n].get("digest")
           or groups[n].get("digest") != pinned[n].get("digest") or pinned[n].get("kc_input") is None]
    if bad:
        return refuse(f"block stage0's groups {bad} are missing, have no z, or their cells / digest / KC input differ "
                      f"from block spec_check's (M.10.7 pins them)")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    all_even = even_pairs(pops)
    if pairs_digest(all_even) != spec.j.h4.pairs_digest:
        return refuse(f"the even pair list differs from H.4's ({pairs_digest(all_even)[:12]})")
    even = [p for p in all_even if p["axis"] == "b"]
    even = even[:spec.n_even_b] if spec.n_even_b is not None else even
    cells = sorted(int(i) for i in pops.mbon)
    if [int(c) for c in s0.get("cells") or []] != cells:
        return refuse("the MBON cells differ from the ones block stage0 measured")
    want0 = next((p for p in pairs_h4 if tuple(pair_key(p)) == tuple(pair_key(even[0]))), None) if even else None
    if want0 is None:
        return refuse("block h4 has no C3 record of the first even (b) pair (the self-check's reference)")
    skipped = {arm: [dict(name=n, overlaps=inc[OTHER[arm]]) for n in arms.get(arm, [])
                     if overlap(groups[n], groups[inc[OTHER[arm]]])] for arm in ("reward", "punish")}
    scan = {arm: [n for n in arms.get(arm, []) if n not in {s["name"] for s in skipped[arm]}]
            for arm in ("reward", "punish")}
    z = {n: tuple(float(v) for v in groups[n]["z"]) for n in names}
    cg = {n: [int(i) for i in groups[n]["cells"]] for n in names}
    kc_input = {n: float(pinned[n]["kc_input"]) for n in names}
    keys = [list(pair_key(p)) for p in even]
    h4 = spec.j.h4

    t0 = time.time()
    print(f"run {rid}: M key {code['key'][:12]}, {len(even)} even (b) pairs, {len(cells)} MBON cells, scan {scan}, "
          f"skipped {skipped}", flush=True)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, spec_check_run_id=sc.get("run_id"), stage0_run_id=s0.get("run_id"),
               oc_run_id=doc["oc"].get("run_id"), c3=params_json(c3), readout=readout, z_c3=z_c3,
               inputs=dict(m0d=dict(path=m0d_path, sha256=sha256_file(m0d_path))), pairs=keys,
               pairs_digest=pairs_digest(even), n_pairs=len(even), cells=cells, incumbents=inc, scanned=scan,
               skipped=skipped, c3_baseline=c3_baseline(pairs_h4, keys, h4.testable_min))
    status, seen = DONE, []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=h4.h3.punish_type, reward_type=h4.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        m4 = H4Measurer(pool, h4, even[:1], pools_h4, cache, None)                   # the first job: code identity
        seen.append(m4)
        row = m4.oracle(c3, readout, z_c3)[0]
        got = pair_stats(row["report"], z_c3, h4.testable_min)
        mis = pair_mismatch(got or {}, want0)
        res["self_check"] = dict(ok=not mis, pair=keys[0], got=got, want=want0, mismatched=mis,
                                 note="h4_jobs.oracle_job unchanged (C3 single readout, block h4's z) re-run on the "
                                      "first even (b) pair against block h4's record: a code-identity check")
        if mis:
            status = MISMATCH
        else:
            m = MMeasurer(pool, spec, cache)
            seen.append(m)
            pres = m.pre(c3, even, cells)                                            # once per pair, shared
            ir, ip = inc["reward"], inc["punish"]
            ref_groups = {ir: cg[ir], ip: cg[ip]}
            ref_rows = m.edit(c3, even, pres, cells, ref_groups, {"A": ip, "P": ir}, {"A": z[ip], "P": z[ir]},
                              reward_type=ir, punish_type=ip)
            ref = {arm: entry(inc[arm], arm, groups[inc[arm]], ref_rows, {"A": z[ip], "P": z[ir]}, ref_groups,
                              kc_input[inc[arm]], h4) for arm in ("reward", "punish")}
            entries = {"reward": [], "punish": []}
            for c in scan["reward"]:
                print(f"reward candidate {c} (punish {ip})", flush=True)
                gr, zz = {c: cg[c], ip: cg[ip]}, {"A": z[ip], "P": z[c]}
                rows = m.edit(c3, even, pres, cells, gr, {"A": ip, "P": c}, zz, reward_type=c, punish_type=ip)
                entries["reward"].append(entry(c, "reward", groups[c], rows, zz, gr, kc_input[c], h4))
            for c in scan["punish"]:
                print(f"punish candidate {c} (reward {ir})", flush=True)
                gr, zz = {c: cg[c], ir: cg[ir]}, {"A": z[c], "P": z[ir]}
                rows = m.edit(c3, even, pres, cells, gr, {"A": c, "P": ir}, zz, reward_type=ir, punish_type=c)
                entries["punish"].append(entry(c, "punish", groups[c], rows, zz, gr, kc_input[c], h4))
            ranked = {arm: rank(entries[arm], arm) for arm in entries}
            tops = {arm: (top(ranked[arm], spec.top_k) if ranked[arm] else [ref[arm]]) for arm in entries}
            rr = {e["name"]: e["r"] for e in entries["reward"]} | {ir: ref["reward"]["r"]}
            pp = {e["name"]: e["p"] for e in entries["punish"]} | {ip: ref["punish"]["p"]}
            res.update(entries=entries, reference=dict(
                           name=f"{ir}|{ip}", n_r=ref["reward"]["n"], n_p=ref["punish"]["n"], r=ref["reward"]["r"],
                           p=ref["reward"]["p"], med_r=ref["reward"]["med"], med_p=ref["punish"]["med"],
                           defined_r=ref["reward"]["defined"], defined_p=ref["punish"]["defined"],
                           edited=ref["reward"]["edited"],
                           edited_ok=ref["reward"]["edited_ok"],
                           note="the incumbent combination (population readouts), both arms' reference row; not ranked"),
                       ranked={arm: [e["name"] for e in ranked[arm]] for arm in ranked},
                       top={arm: [dict(name=e["name"], cells=list(e["cells"]), digest=e["digest"], n=e["n"],
                                       med=e["med"], incumbent=e["name"] == inc[arm]) for e in tops[arm]]
                            for arm in tops},
                       predicted_joint=predicted_joint(rr, pp, h4.testable_min),
                       predicted_joint_note="#pairs with r_i >= 2 and -p_j >= 2 on the same pair, every reward x "
                                            "punish entry incl. the incumbents' reference row (a record, not a judgement)")
    gp = guard_params(Params(), seen)
    res["status"] = status
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = m_store.write_json(out / "runs" / f"{rid}-stage1.json", res, gp)
    ch = res["self_check"]
    lines = [f"# M.10.3 stage 1 {rid}\n\n", f"**status: {status}**\n\n",
             f"- self-check (oracle_job, C3 single readout, first even (b) pair vs block h4): {ch['ok']} "
             f"{ch['mismatched']}\n",
             f"- C3 baseline (block h4, {res['c3_baseline']['n_pairs']} pairs): r >= 2 {res['c3_baseline']['n_r']}, "
             f"-p >= 2 {res['c3_baseline']['n_p']}\n", f"- skipped (overlap with the other arm's incumbent): {skipped}\n"]
    if status == DONE:
        rf = res["reference"]
        lines.append(f"- reference {rf['name']}: n_r {rf['n_r']}, n_p {rf['n_p']}, edited_ok {rf['edited_ok']}\n")
        lines.append("\n| arm | candidate | n | median | defined | KC input | edited | edited ok |\n"
                     "|---|---|---|---|---|---|---|---|\n")
        for arm in ("reward", "punish"):
            for e in ranked[arm]:
                ed = {k: (v["group"], len(v["cells"])) for k, v in e["edited"].items()}
                lines.append(f"| {arm} | {e['name']} | {e['n']} | {e['med']:.4g} | {e['defined']} | "
                             f"{e['kc_input']:.6g} | {ed} | {e['edited_ok']} |\n")
        lines.append(f"\ntop: { {arm: [t['name'] for t in res['top'][arm]] for arm in res['top']} }\n\n"
                     f"predicted joint: {res['predicted_joint']}\n")
    m_store.write_bytes(out / "runs" / f"{rid}-stage1.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status} in {res['wall_s'] / 60:.1f} min", flush=True)
    if status == MISMATCH:
        print(f"self-check DIFFERS {ch['mismatched']}: stop and ask the user", flush=True)
        return 5
    print(f"top per arm: { {arm: [t['name'] for t in res['top'][arm]] for arm in res['top']} }", flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage1", res, report, gp)
        print(f"wrote {a.summary} (block stage1{', smoke' if a.smoke else ''})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
