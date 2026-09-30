#!/usr/bin/env python3
"""Spec M.10.5 (+ M.10.4, M.10.7, plan readings 6, 15 and 16), stage 3: the selected combination against the same-set C3
on the judgement list.

- The judgement list regenerated (l_pairs.new_pairs with a_turns = n_turns, m_rules.judgement_set, digests pinned) and
  required equal to block list's keys: (b) 21 + (a) 18.
- Selected: MMeasurer.pre on the list, then edit with block stage2's winner (groups, POPULATION readout, z, DAN names).
- C3: h4_jobs.oracle_job unchanged (H4Measurer) with C3's single readout (MBON13 / MBON05), block h4's z and pools as
  block stage0 recorded them, on the same pairs.
- l_rules.pair_rows_stats + h4_formula.arm_aggregate for both; m_rules.stage3_reading (n = selected testable_b, c = C3's,
  F_a the selected (a); M.10.7: n >= 11 with c >= n reads B_NO_CONCLUSION); no reading unless n_b = 21. The sentence
  (M.10.5) with the combination 〈PPLx·PAMy〉, the even selection k (block stage2's winner testable_b), the odd record h,
  the candidate count, the specificity-passing count, the core w_mbon medians, the DAN cell counts (PPL, PAM) and the
  list's move diversity; the attempts I-M and the even-data usage are recorded with it.

    uv run python scripts/run_m_stage3.py                                        # ~1 h (resumable: rerun it)
    uv run python scripts/run_m_stage3.py --smoke --allow-dirty --workers 4 --summary results/m0d/m/smoke/m_readout.json

Run it from the repository root. Refusals (exit 2) before the pool starts, in order: another directory, --out outside
results/m0d/m/, dirty hashed files (unless --allow-dirty), a summary that is not git-tracked and clean at HEAD (outside
--smoke; --smoke reads only a smoke summary under results/m0d/m/) or lacks block stage2 or list (each listed), no block
stage0 in it, no connectome file, any of stage0 / stage2 / list produced under other code, block list made on another
stage2 run, block stage2's gate other than STAGE2_GO, a later block (outside --smoke), another connectome, an m0d.json
other than the one stage 0 ran on (sha256) or a C3 record differing from block stage0's, block stage2 made on another
stage0 run, a winner whose cells / digests differ from block stage0's groups, a judgement list whose digests are not
the pinned ones or whose keys differ from block list's.
Report: <out>/runs/<run id>-stage3.{json,md}. Block "stage3" only when not --smoke, clean, the declared spec and a
reading (outcome not None); a --smoke run never writes it. Exit codes: 0 read, 2 refused, 3 invalid oracle rows
(recorded; no block).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from flymon.brain import m_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_formula import arm_aggregate
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import pair_key
from flymon.brain.j_runner import params_json
from flymon.brain.l_pairs import new_pairs
from flymon.brain.l_rules import pair_rows_stats
from flymon.brain.m_cands import cells_digest
from flymon.brain.m_cli import (POOL_TIMEOUT_S, check_committed, code_keys, git_state, guard_params, later_blocks,
                                load_c3_record, other_code, out_allowed, read_previous, refuse, run_id, run_id_chain,
                                same_code, stage0_inputs, write_block)
from flymon.brain.m_measure import HASHED_FILES, MMeasurer
from flymon.brain.m_rules import STAGE2_GO, judgement_set, sentence, stage3_reading
from flymon.brain.m_spec import SPEC, MSpec, smoke

NEED = ("stage2", "list")
EVEN_USAGE = "G·H·J·K·L·M"          # the stages whose selection used the even 21 pairs (M.10.5's SELECTED sentence)


def stats_agg(rows: list, pairs: list, z: dict, h4) -> dict:
    st = pair_rows_stats(rows, z, [pair_key(p) for p in pairs], h4)
    agg = None if st["reasons"] else arm_aggregate(st["stats"], h4.naive_max, h4.t_b_min, h4.f_a_min)
    return dict(reasons=st["reasons"], pairs=st["pairs"], agg=agg)


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
    if not isinstance(doc.get("stage0"), dict):
        return refuse(f"{a.summary} lacks block stage0 (stage 3 reads C3's pools and z and the groups from it)")
    try:
        code, manifest = code_keys(a.npz)
    except SystemExit as e:                                  # code_keys named the missing hashed files
        return int(e.code or 2)
    except OSError:
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    why = (other_code(doc, ["stage0", *NEED], code["key"], manifest["key"], same_code)
           or run_id_chain(doc, "list", {"stage2_run_id": "stage2"}))
    if why:
        return refuse(why)
    s0, s2, lst = doc["stage0"], doc["stage2"], doc["list"]
    if s2.get("gate") != STAGE2_GO:
        return refuse(f"block stage2's gate is {s2.get('gate')}, not {STAGE2_GO}: stage 3 does not run")
    why = None if a.smoke else later_blocks(a.summary, "stage3", doc)
    if why:
        return refuse(why)
    npz_sha = code["files"].get("npz:" + Path(a.npz).name)
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({str(npz_sha)[:12]})")
    m0d_path = a.m0d or spec.m0d_path
    why = stage0_inputs(s0, m0d_path)
    if why:
        return refuse(why)
    try:
        c3, readout_c3, z_c3, pools_h4 = load_c3_record(m0d_path, spec.l)
    except ValueError as e:
        return refuse(str(e))
    why = (stage0_inputs(s0, m0d_path, c3=params_json(c3), readout=readout_c3, z_c3=z_c3, pools=pools_h4)
           or run_id_chain(doc, "stage2", {"stage0_run_id": "stage0"}))
    if why:
        return refuse(why)
    # ---- block stage2's winner, tied to block stage0's groups --------------------------------------------------------
    w = s2.get("winner") or {}
    groups = {g.get("name"): g for g in s0.get("groups") or []}
    r_name, p_name = w.get("reward"), w.get("punish")
    wc, wd = w.get("cells") or {}, w.get("digests") or {}
    bad = [n for n in (r_name, p_name) if n not in groups or n not in wc
           or [int(c) for c in wc[n]] != [int(c) for c in groups[n].get("cells") or []]
           or cells_digest(wc[n]) != wd.get(n) or wd.get(n) != groups[n].get("digest")]
    if bad or not w.get("readout") or not w.get("z"):
        return refuse(f"block stage2's winner {w.get('name')} ({bad}) is missing or its cells / digests / readout / z "
                      f"differ from block stage0's groups")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    try:
        js = judgement_set(new_pairs(pops, dataclasses.replace(spec.l, a_turns=spec.l.n_turns)), spec)
    except ValueError as e:
        return refuse(str(e))
    keys = dict(b=[list(pair_key(p)) for p in js["b"]], a=[list(pair_key(p)) for p in js["a"]])
    if keys != dict(b=lst.get("b"), a=lst.get("a")) or (js["b_digest"], js["a_digest"]) != (lst.get("b_digest"),
                                                                                             lst.get("a_digest")):
        return refuse("the regenerated judgement list differs from block list's keys / digests")
    pairs = js["b"] + js["a"]
    cells = sorted(int(i) for i in pops.mbon)
    if [int(c) for c in s0.get("cells") or []] != cells:
        return refuse("the MBON cells differ from the ones block stage0 measured")
    gsel = {n: [int(c) for c in wc[n]] for n in (r_name, p_name)}
    zsel = {k: tuple(float(x) for x in v) for k, v in w["z"].items()}
    h4 = spec.j.h4
    combo = f"{p_name}·{r_name}"
    try:                                                     # the sentence's fixed context, checked before the pool
        dv = lst["diversity"]
        fixed = dict(combo=combo, k=int(w["testable_b"]), h=int(s2["odd"]["n_testable"]), n_cands=int(s2["n_cands"]),
                     m=int(s2["n_spec_passing"]),
                     w="·".join(f"{n} {float(groups[n]['w_mbon_stats']['median']):.4g}" for n in (p_name, r_name)),
                     dan=(int(groups[p_name]["n_dan_cells"]), int(groups[r_name]["n_dan_cells"])),
                     moves=int(dv["n_moves"]), top_move=str(dv["top_move_text"]))
    except (KeyError, TypeError, ValueError) as e:
        return refuse(f"blocks stage2 / list / stage0 lack a field of M.10.5's sentence context: {e!r}")

    t0 = time.time()
    print(f"run {rid}: {combo} vs C3 on (b) {len(js['b'])} + (a) {len(js['a'])}", flush=True)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, stage0_run_id=s0.get("run_id"), stage2_run_id=s2.get("run_id"),
               list_run_id=lst.get("run_id"), c3=params_json(c3), readout_c3=readout_c3, z_c3=z_c3, pools=pools_h4,
               inputs=dict(m0d=dict(path=m0d_path, sha256=sha256_file(m0d_path))), pairs=keys,
               b_digest=js["b_digest"], a_digest=js["a_digest"],
               selected=dict(name=w.get("name"), combo=combo, reward=r_name, punish=p_name, readout=w["readout"],
                             z={k: list(v) for k, v in zsel.items()}, cells=gsel, digests={n: wd[n] for n in gsel}),
               attempts=[dict(claim=c, verdict=v, commit=h) for c, v, h in spec.attempts], even_usage=EVEN_USAGE)
    seen = []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=h4.h3.punish_type, reward_type=h4.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        m = MMeasurer(pool, spec, cache)
        seen.append(m)
        pres = m.pre(c3, pairs, cells)
        rows_sel = m.edit(c3, pairs, pres, cells, gsel, w["readout"], zsel, reward_type=r_name, punish_type=p_name)
        m4 = H4Measurer(pool, h4, pairs, pools_h4, cache, None)
        seen.append(m4)
        rows_c3 = m4.oracle(c3, readout_c3, z_c3)
    gp = guard_params(Params(), seen)
    sel = stats_agg(rows_sel, pairs, zsel, h4)
    base = stats_agg(rows_c3, pairs, z_c3, h4)
    res.update(stats_selected=sel["pairs"], stats_c3=base["pairs"],
               reasons=dict(selected=sel["reasons"], c3=base["reasons"]), aggregate_selected=sel["agg"],
               aggregate_c3=base["agg"], edited=rows_sel[0]["edited"] if rows_sel else None)
    if sel["agg"] is None or base["agg"] is None:
        status, res["reading"], res["sentence"] = "invalid", None, None
    else:
        status = "done"
        rd = stage3_reading(sel["agg"], base["agg"], spec)
        res["reading"] = rd
        ctx = dict(fixed, n=rd["n"], c=rd["c"], fa=rd["F_a"], n_b=rd["n_b"], note=rd["note"], naive_a=rd["naive_a"],
                   f_a_possible=rd["f_a_possible"], attempts=res["attempts"], even_usage=EVEN_USAGE)
        res["context"] = ctx
        res["sentence"] = None if rd["outcome"] is None else sentence(rd["outcome"], ctx)
    res["status"] = status
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = m_store.write_json(out / "runs" / f"{rid}-stage3.json", res, gp)
    rd = res["reading"] or {}
    lines = [f"# M.10.5 stage 3 {rid}\n\n", f"**status: {status}; outcome {rd.get('outcome')}**\n\n",
             f"- selected {combo}: {res['aggregate_selected']}\n", f"- C3 (same set): {res['aggregate_c3']}\n",
             f"- reasons: {res['reasons']}\n", f"- reading: {rd}\n"]
    if res["sentence"]:
        lines.append(f"\n{res['sentence']}\n")
    m_store.write_bytes(out / "runs" / f"{rid}-stage3.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status}, outcome {rd.get('outcome')} ({rd.get('note')}) in {res['wall_s'] / 60:.1f} min",
          flush=True)
    blockers = dict(smoke=a.smoke, dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec,
                    invalid=status != "done", no_reading=rd.get("outcome") is None)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage3", res, report, gp)
        print(f"wrote {a.summary} (block stage3): {res['sentence']}", flush=True)
    return 3 if status == "invalid" else 0


if __name__ == "__main__":
    sys.exit(main())
