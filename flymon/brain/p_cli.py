"""The CLI layer of spec appendix P (P.4, P.6.4, P.6.5). One copy; scripts/p_oc.py and scripts/run_p.py import it.
- Refusals before any pool, in order: another directory; --out outside results/p/; a --smoke run outside
  results/p/smoke/ or a declared run inside it; dirty hashed files (unless --allow-dirty); an unreadable summary;
  outside --smoke, a block already in the real summary (no post-hoc rewrite; P.6.5's one rerun after a technical
  INVALID is the only exception, below); outside --smoke, a spec with no commit; no usable C3 record; the stage's
  `prepare` (c1 from block n1, the OC block before the judgement, O2's per-seed source for the OC).
- P.6.5's rerun: only for a stage that allows it (run_p), only when the real block's outcome is INVALID, only with
  --rerun-after-invalid, only when the code manifest differs from the INVALID run's, and only once: the INVALID block is
  kept as block "<name>_invalid" and the new run's block is final whatever it says. A block with a judged label is
  final: no rerun. A rerun with changed rule/CLI code reuses cached raw rows on purpose: the cache is keyed by the
  measurement code (p_measure.MEASURE_FILES), so only a measurement-code change refreshes entries (reading 11).
- Cache roots: a --smoke run under results/p/smoke/, a declared run under results/p/run/ (default); an --out or
  --summary crossing between the two is refused both ways. The cache key covers MEASURE_FILES, the manifest
  HASHED_FILES; a missing hashed file is a refusal.
- O2's per-seed source (P.6.4): block o2's run only; both O2 X rows (4:1, δ-DL) must carry identical seeds in the
  same order before the OC runs. The OC's null (p_rules.null_shift) is stated as OC_NULL in the record.
- The hooks a script imports and passes as `hooks(its own module)`, so its tests patch them on the script.
- O's and N's helpers are imported, never edited: the C3 record, the model ORN types, the Hallem stimuli at a point,
  the spec commit and block n1's oracle check (o_cli.n_oracle, on an OSpec holding only the c1 pair).
- The block, the report and the summary go through p_measure's guard; each carries the run id, the code keys, the git
  state and the spec commit.
Copied from o_cli (o_cli's main_stage is hard-wired to results/o/ and o_measure's guard; reading 1): out_allowed,
git_state, code_keys, make_measurer, spec_record, write_report, Ctx, parser and main_stage, with P's paths and
refusals."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import h3_store, n_cli, o_cli, o_measure, p_measure
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import check_committed, refuse, run_id
from .n_rules import INVALID, delta
from .o_rules import JUDGED, o2_declared, o2_edit_of, o2_key, validity
from .o_spec import SPEC as O_SPEC
from .odor_real import DataMismatch
from .p_measure import HASHED_FILES, MEASURE_FILES, PCache, PMeasurer
from .p_spec import SPEC, PSpec, smoke

NPZ = n_cli.NPZ
RUN_OUT = p_measure.ALLOWED_DIR + "run"
SMOKE_OUT = p_measure.ALLOWED_DIR + "smoke"
SMOKE_SUMMARY = f"{SMOKE_OUT}/{Path(p_measure.SUMMARY).name}"
O2_CACHE = o_cli.RUN_OUT + "/cache/o2_arm"            # O2's per-seed rows (git-ignored; P.6.4's source)
HOOK_NAMES = ("check_committed", "code_keys", "git_state", "load_c3", "m0d_sha", "make_measurer", "model_types",
              "spec_commit", "c1_source", "o2_source")
OC_NULL = "only s is shifted by mean(ℓ) − c₁; t fixed"   # p_rules.null_shift's construction, stated in the OC record


def hooks(module) -> dict:
    return {n: getattr(module, n) for n in HOOK_NAMES}


def in_dir(path, d: str) -> bool:
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    return (rel + "/").startswith(d.rstrip("/") + "/")


def out_allowed(out) -> bool:
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(p_measure.ALLOWED_DIR)


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES); a missing hashed file is a refusal (exit 2)."""
    missing = [f for f in HASHED_FILES if not (ROOT / f).exists()]
    if missing:
        refuse(f"the hashed files {missing} do not exist")
        raise SystemExit(2)
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=HASHED_FILES)


def load_c3(spec) -> tuple:
    return o_cli.load_c3(spec.o)


def m0d_sha(spec) -> str:
    return o_cli.m0d_sha(spec.o)


def model_types(npz) -> list:
    return o_cli.model_types(npz)


def spec_commit(path) -> str | None:
    return o_cli.spec_commit(path)


def c1_source(spec, smoke: bool, committed=check_committed, path=None) -> tuple:
    """P.3 / P.6.5: (record, None) or (None, refusal). Block n1's exact o of the c1 pair (outside --smoke the N summary
    must be committed) must agree with the declared o within oracle_o_tol (o_cli.n_oracle on an OSpec holding only the
    c1 pair); c1 = c1_frac |exact o|. --smoke without a readable block uses the declared o."""
    o = dataclasses.replace(spec.o, oracle_o=tuple(p for p in spec.o.oracle_o if p[0] == spec.c1_pair))
    rec, why = o_cli.n_oracle(o, smoke, committed=committed, path=path)
    if why:
        return None, why
    exact = None if rec["exact"] is None else rec["exact"][spec.c1_pair]
    used = rec["declared"][spec.c1_pair] if exact is None else exact
    return dict(pair=spec.c1_pair, declared=rec["declared"][spec.c1_pair], exact=exact, o=used,
                c1=spec.c1_frac * abs(used), tol=spec.o.oracle_o_tol), None


def o2_source(spec, smoke: bool, committed=check_committed, summary=None, cache=None) -> tuple:
    """P.6.4's input: (record, None) or (None, refusal). Block o2 of the committed O summary (a judged, declared run);
    the per-seed rows are the O2 cache entries written under that block's measure key; they must cover O2's declared
    (X, arm, seed) grid exactly (o_rules.validity), and D = mean(Δ(dV)[punish] - Δ(dV)[plastic]) recomputed from them
    must equal the block's D for every X within o2_check_tol. Both X rows must carry identical seeds in the same order
    (every arm P reads). The record's rows are in o2_key order; `null` states the OC's null construction (OC_NULL)."""
    summary = Path(summary) if summary is not None else ROOT / o_measure.SUMMARY
    cache = Path(cache) if cache is not None else ROOT / O2_CACHE
    why = committed(summary, ["o2"])
    if why:
        return None, why
    try:
        blk = json.loads(summary.read_text())["o2"]
        key, z, xs = blk["measure_key"], blk["c3"]["z"], blk["x"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, f"no usable block o2 in {summary}: {e!r}"
    if blk.get("outcome") != JUDGED or blk.get("smoke"):
        return None, f"block o2 in {summary} is not a judged declared run"
    rows = []
    for f in sorted(cache.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("kind") == "o2_arm" and d.get("code_key") == key:
            rows.append(dict(d["result"], x=d["inputs"]["x"], y=d["inputs"]["y"]))
    bad = validity(rows, o2_declared(O_SPEC), o2_key, O_SPEC.o2_seeds, o2_edit_of(O_SPEC))
    if bad:
        return None, f"O2's cache {cache} does not hold block o2's run: {'; '.join(bad)}"
    rows.sort(key=lambda r: (r["x"], r["arm"], int(r["seed"])))      # o2_key order, not the cache's file names
    order = {(x, a): [int(r["seed"]) for r in rows if r["x"] == x and r["arm"] == a]
             for x, _, _ in O_SPEC.o2_pairs for a in spec.arms}
    seeds = order[(O_SPEC.o2_pairs[0][0], spec.arms[0])]
    off = {f"{x}/{a}": s for (x, a), s in order.items() if s != seeds}
    if off:                                       # Task 4 review: both X rows pair seed by seed before the OC
        return None, (f"O2's X rows (4:1, δ-DL) do not carry identical seeds in the same order: {seeds[:4]}... vs "
                      f"{ {k: v[:4] for k, v in off.items()} }")
    one, three = spec.arms[0], spec.arms[2]
    D = {}
    for x, _, _ in O_SPEC.o2_pairs:
        by = {a: sorted((r for r in rows if r["x"] == x and r["arm"] == a), key=lambda r: int(r["seed"]))
              for a in (one, three)}
        got = float(sum(delta(r3, z) - delta(r1, z) for r1, r3 in zip(by[one], by[three])) / len(by[one]))
        D[x] = dict(recomputed=got, block=float(xs[x]["stats"]["D"]))
        if abs(got - D[x]["block"]) > spec.o2_check_tol:
            return None, f"O2's cache gives D {got} for X = {x}, block o2 says {D[x]['block']}"
    return dict(run_id=blk["run_id"], measure_key=key, z=z, rows=rows, D=D, summary=str(summary),
                cache=str(cache), n_rows=len(rows), seeds=seeds, null=OC_NULL), None


def make_measurer(ctx) -> tuple:
    """(PMeasurer, its FlyPool): the one pool construction of P's CLIs. The caller closes the pool."""
    from . import fly_pool
    h3 = ctx.spec.o.n.h3
    pool = fly_pool.FlyPool(ctx.args.npz, ctx.c3, flies=[{}] * ctx.args.workers, workers=ctx.args.workers,
                            punish_type=h3.punish_type, reward_type=h3.reward_type,
                            timeout_s=ctx.spec.o.pool_timeout_s)
    return PMeasurer(pool, ctx.spec, PCache(ctx.out / "cache", ctx.key, ctx.rid, ctx.spec_commit)), pool


def nview(ctx):
    """The context as n_cli's stimulus helpers read it (spec = N's)."""
    return SimpleNamespace(spec=ctx.spec.o.n, c3=ctx.c3, types=ctx.types)


def spec_record(spec) -> dict:
    return dict({f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "o"},
                o=o_cli.spec_record(spec.o))


def write_report(out, rid: str, name: str, res: dict, params_list) -> Path:
    base = Path(out) / "runs" / f"{rid}-{name}"
    path = p_measure.write_json(Path(f"{base}.json"), res, params_list)
    md = f"# P {name} — {rid}\n\n{res.get('sentence', '')}\n\noutcome: `{res.get('outcome')}`\n"
    p_measure.write_bytes(Path(f"{base}.md"), md.encode(), params_list)
    return path


@dataclass
class Ctx:
    args: argparse.Namespace
    spec: PSpec
    smoke: bool
    rid: str
    out: Path
    key: dict
    c3: object
    readout: dict
    z: dict
    types: list
    hooks: dict
    spec_commit: str | None
    extra: dict | None


def parser(doc_help: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--workers", type=int, default=SPEC.o.workers)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--rerun-after-invalid", action="store_true")
    return ap


def main_stage(name: str, argv, body, hooks: dict, *, spec=None, require_root: bool = True,
               doc_help: str | None = None, prepare=None, rerun_once: bool = False) -> int:
    """The refusals (module docstring), then body(ctx) -> (res, exit code); a DataMismatch inside the body is a
    refusal. Any other error propagates: nothing is written. The report goes under <out>/runs/ and block `name` into the
    summary (a --smoke run: its smoke summary). `prepare(spec, smoke, hooks, doc=, summary=, z=) -> (extra, refusal)`
    runs after C3 is loaded and before any pool; its extra is ctx.extra."""
    a = parser(doc_help).parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or (SMOKE_OUT if a.smoke else RUN_OUT))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {p_measure.ALLOWED_DIR} of the repository root")
    summary = Path(a.summary or (SMOKE_SUMMARY if a.smoke else p_measure.SUMMARY))
    if a.smoke and not (in_dir(out, SMOKE_OUT) and in_dir(summary, SMOKE_OUT)):
        return refuse(f"a --smoke run reads and writes only under {SMOKE_OUT}/ (cache and summary), not {out} / "
                      f"{summary}")
    if not a.smoke and (in_dir(out, SMOKE_OUT) or in_dir(summary, SMOKE_OUT)):
        return refuse(f"a declared run never reads or writes under the smoke root {SMOKE_OUT}/ ({out} / {summary})")
    git = hooks["git_state"](HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    try:
        doc = json.loads(summary.read_text()) if summary.exists() else {}
    except (OSError, ValueError) as e:
        return refuse(f"no usable summary at {summary}: {e}")
    prior = None
    if not a.smoke and name in doc:
        blk = doc[name]
        if not (rerun_once and isinstance(blk, dict) and blk.get("outcome") == INVALID):
            return refuse(f"{summary} already holds block {name}: each verdict goes to the user with no post-hoc "
                          f"change (P.4, P.6.5), so it is not rewritten")
        if f"{name}_invalid" in doc:
            return refuse(f"{summary} already holds block {name}_invalid: P.6.5 allows one rerun only")
        if not a.rerun_after_invalid:
            return refuse(f"block {name} is INVALID: after fixing the code, rerun once with --rerun-after-invalid "
                          f"(P.6.5)")
        prior = blk
    elif a.rerun_after_invalid:
        return refuse(f"--rerun-after-invalid needs an INVALID block {name} in the real summary {summary}")
    commit = hooks["spec_commit"](spec.o.spec_path)
    if commit is None and not a.smoke:
        return refuse(f"the spec {spec.o.spec_path} has no commit: commit appendix P before a real run")
    key, manifest = hooks["code_keys"](a.npz)
    if prior is not None and (prior.get("code") or {}).get("key") == manifest["key"]:
        return refuse(f"the code manifest equals the INVALID run's ({manifest['key'][:12]}): fix the code first "
                      f"(P.6.5)")
    try:
        c3, readout, z, _ = hooks["load_c3"](spec)
    except ValueError as e:
        return refuse(str(e))
    extra = None
    if prepare is not None:
        extra, why = prepare(spec, a.smoke, hooks, doc=doc, summary=summary, z=z)
        if why:
            return refuse(why)
    rid = run_id()
    ctx = Ctx(args=a, spec=spec, smoke=a.smoke, rid=rid, out=out, key=key, c3=c3, readout=dict(readout), z=z,
              types=hooks["model_types"](a.npz), hooks=hooks, spec_commit=commit, extra=extra)
    t0 = time.time()
    try:
        res, code = body(ctx)
    except DataMismatch as e:
        return refuse(f"the Hallem data do not match their pins: {e}")
    res = dict(res, name=name, run_id=rid, smoke=a.smoke, measure_key=key["key"], code=manifest, git=git,
               spec_commit=commit, inputs=dict(m0d_sha256=hooks["m0d_sha"](spec), data_sha256=spec.o.n.sha_pins(),
                                               state_p_min=spec.o.n.state_p_min),
               c3=dict(readout=readout, z=z), spec=spec_record(spec), argv=list(argv or []), wall_s=time.time() - t0,
               rerun_of=None if prior is None else prior.get("run_id"))
    report = write_report(out, rid, name, res, [c3])
    if prior is not None:
        p_measure.write_summary_block(summary, f"{name}_invalid", prior, [c3])
    p_measure.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                  [c3])
    print(res.get("sentence", ""))
    return code
