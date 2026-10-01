"""The CLI layer of spec appendix O (O.5, O.7.5, readings 13 and 18-19). One copy; run_o1.py / run_o2.py import it.
- Refusals before any pool, in order: another directory; --out outside results/o/; dirty hashed files (unless
  --allow-dirty); an unreadable summary; outside --smoke, a block already in the real summary (no post-hoc rewrite); the
  stage's `prepare` (O2: block n1's oracle o); outside --smoke, a spec with no commit; no usable C3 record.
  A --smoke run reads and writes only under results/o/smoke/ (its cache root and summary) and a declared run never
  there: a smoke run never reads or writes a real cache entry (Task 4 review).
- The hooks a script imports and passes as `hooks(its own module)`, so its tests patch them on the script.
- N's helpers are imported, never edited: the C3 record (n_cli.load_c3 on spec.n), the model ORN types, the Hallem
  stimuli at a point (n_cli.stimuli_at on `nview`) and the drive record.
- The block, the report and the summary go through o_measure's guard; each carries the run id, the code keys, the git
  state and the spec commit (O.7.5).
- O1's edit check (Task 2 review): the APL->non-KC block must zero at least one APL -> P readout (MBON05) edge on the
  real connectome, counted in this process on a rig built as o_jobs.rig builds it (nonkc_edit, before any pool).
O1 and O2 are independent (O.5): no block needs another.
Copied from n_cli (O.5 forbids editing N; plan ruling D4): out_allowed, git_state, code_keys, make_measurer,
spec_record, write_report, Ctx, parser and main_stage, with O's paths, O's spec and O's refusals."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import h3_store, n_cli, n_store, o_measure
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import check_committed, refuse, run_id
from .o_measure import HASHED_FILES, MEASURE_FILES, OCache, OMeasurer
from .o_spec import SPEC, OSpec, smoke
from .odor_real import DataMismatch

NPZ = n_cli.NPZ
RUN_OUT = o_measure.ALLOWED_DIR + "run"
SMOKE_OUT = o_measure.ALLOWED_DIR + "smoke"
SMOKE_SUMMARY = f"{SMOKE_OUT}/{Path(o_measure.SUMMARY).name}"
HOOK_NAMES = ("code_keys", "git_state", "load_c3", "m0d_sha", "make_measurer", "model_types", "spec_commit",
              "n_oracle", "nonkc_edit")


def hooks(module) -> dict:
    return {n: getattr(module, n) for n in HOOK_NAMES}


def in_dir(path, d: str) -> bool:
    """`path` (relative to the working directory) lies under `d`."""
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    return (rel + "/").startswith(d.rstrip("/") + "/")


def out_allowed(out) -> bool:
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(o_measure.ALLOWED_DIR)


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
    return n_cli.load_c3(spec.n)


def m0d_sha(spec) -> str:
    return n_cli.m0d_sha(spec.n)


def model_types(npz) -> list:
    return n_cli.model_types(npz)


def spec_commit(path) -> str | None:
    """The last commit touching the spec (O.7.5's 스펙 커밋), None when it has none."""
    r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(path)], capture_output=True,
                       text=True)
    return r.stdout.strip() or None


def n_oracle(spec, smoke: bool, committed=check_committed, path=None) -> tuple:
    """Reading 13: ({"declared", "exact"}, None) or (None, refusal). e uses O.7.4's declared o; outside --smoke N's
    summary must be committed and its block n1's exact o agree within oracle_o_tol. --smoke without a readable block
    records exact None."""
    declared = dict(spec.oracle_o)
    path = Path(path) if path is not None else ROOT / n_store.SUMMARY
    if not smoke:
        why = committed(path, ["n1"])
        if why:
            return None, why
    try:
        pairs = json.loads(path.read_text())["n1"]["pairs"]
        exact = {k: float(pairs[k]["o"]) for k in declared}
    except (OSError, ValueError, KeyError, TypeError) as e:
        if smoke:
            return dict(declared=declared, exact=None), None
        return None, f"no usable block n1 in {path}: {e!r}"
    off = {k: (exact[k], v) for k, v in declared.items() if abs(exact[k] - v) > spec.oracle_o_tol}
    if off:
        return None, f"block n1's oracle effects {off} disagree with O.7.4's declared values"
    return dict(declared=declared, exact=exact), None


def make_measurer(ctx) -> tuple:
    """(OMeasurer, its FlyPool): the one pool construction of O's CLIs. The caller closes the pool."""
    from . import fly_pool
    h3 = ctx.spec.n.h3
    pool = fly_pool.FlyPool(ctx.args.npz, ctx.c3, flies=[{}] * ctx.args.workers, workers=ctx.args.workers,
                            punish_type=h3.punish_type, reward_type=h3.reward_type, timeout_s=ctx.spec.pool_timeout_s)
    return OMeasurer(pool, ctx.spec, OCache(ctx.out / "cache", ctx.key, ctx.rid, ctx.spec_commit)), pool


def count_nonkc_edit(conn, pops, eng, readout: dict) -> dict:
    """Apply o_jobs' APL->non-KC edit to `eng` (a fresh engine, as o_jobs.rig makes one) and count the APL out-edges to
    the P readout type that were nonzero before and are zero after, and every APL -> non-KC edge zeroed."""
    import numpy as np
    from . import o_jobs
    from .h3_jobs import edge_sources
    from .n_jobs import readout_cells
    w0 = np.array(eng.csc.w, copy=True)
    o_jobs.apply_edit(eng, pops, o_jobs.NONKC)
    src, tgt = edge_sources(eng.csc), np.asarray(eng.csc.tgt, np.int64)
    apl = np.isin(src, np.asarray(pops.apl, np.int64))
    to_p = apl & np.isin(tgt, readout_cells(conn, readout)[1])
    zeroed = (w0 != 0) & (eng.csc.w == 0)
    nonkc = apl & ~np.isin(tgt, np.asarray(pops.kc, np.int64))
    return dict(edit=o_jobs.NONKC, readout_p=readout["P"], apl_to_p_edges=int(to_p.sum()),
                apl_to_p_zeroed=int((to_p & zeroed).sum()), apl_to_nonkc_zeroed=int((nonkc & zeroed).sum()))


def nonkc_edit(ctx) -> dict:
    """count_nonkc_edit on the real connectome (ctx.args.npz) at C3's Params: one engine, no pool (seconds)."""
    from .circuits import Populations
    from .connectome import Connectome
    from .engine_cpu import Engine
    conn = Connectome.load(str(ctx.args.npz))
    pops = Populations.from_connectome(conn)
    return count_nonkc_edit(conn, pops, Engine(conn, pops, ctx.c3, seed=0), ctx.readout)


def nview(ctx):
    """The context as n_cli's stimulus helpers read it (spec = N's)."""
    return SimpleNamespace(spec=ctx.spec.n, c3=ctx.c3, types=ctx.types)


def spec_record(spec) -> dict:
    return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "n"}


def write_report(out, rid: str, name: str, res: dict, params_list) -> Path:
    base = Path(out) / "runs" / f"{rid}-{name}"
    path = o_measure.write_json(Path(f"{base}.json"), res, params_list)
    md = f"# O {name} — {rid}\n\n{res.get('sentence', '')}\n\noutcome: `{res.get('outcome')}`\n"
    o_measure.write_bytes(Path(f"{base}.md"), md.encode(), params_list)
    return path


@dataclass
class Ctx:
    args: argparse.Namespace
    spec: OSpec
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
    ap.add_argument("--workers", type=int, default=SPEC.workers)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    return ap


def main_stage(name: str, argv, body, hooks: dict, *, spec=None, require_root: bool = True,
               doc_help: str | None = None, prepare=None) -> int:
    """The refusals (module docstring), then body(ctx) -> (res, exit code); a DataMismatch inside the body is a
    refusal. Any other error propagates: nothing is written. The report goes under <out>/runs/ and block `name` into
    the summary (a --smoke run: its smoke summary). `prepare(spec, smoke, hooks) -> (extra, refusal)` runs before the
    pool; its extra is ctx.extra."""
    a = parser(doc_help).parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or (SMOKE_OUT if a.smoke else RUN_OUT))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {o_measure.ALLOWED_DIR} of the repository root")
    summary = Path(a.summary or (SMOKE_SUMMARY if a.smoke else o_measure.SUMMARY))
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
    if not a.smoke and isinstance(doc, dict) and name in doc:
        return refuse(f"{summary} already holds block {name}: each verdict goes to the user with no post-hoc change "
                      f"(O.5), so it is not rewritten")
    extra = None
    if prepare is not None:
        extra, why = prepare(spec, a.smoke, hooks)
        if why:
            return refuse(why)
    commit = hooks["spec_commit"](spec.spec_path)
    if commit is None and not a.smoke:
        return refuse(f"the spec {spec.spec_path} has no commit: commit appendix O before a real run")
    key, manifest = hooks["code_keys"](a.npz)
    try:
        c3, readout, z, _ = hooks["load_c3"](spec)
    except ValueError as e:
        return refuse(str(e))
    rid = run_id()
    ctx = Ctx(args=a, spec=spec, smoke=a.smoke, rid=rid, out=out, key=key, c3=c3, readout=dict(readout), z=z,
              types=hooks["model_types"](a.npz), hooks=hooks, spec_commit=commit, extra=extra)
    t0 = time.time()
    try:
        res, code = body(ctx)
    except DataMismatch as e:
        return refuse(f"the Hallem data do not match their pins: {e}")
    res = dict(res, name=name, run_id=rid, smoke=a.smoke, measure_key=key["key"], code=manifest, git=git,
               spec_commit=commit, inputs=dict(m0d_sha256=hooks["m0d_sha"](spec), data_sha256=spec.n.sha_pins(),
                                               state_p_min=spec.n.state_p_min),
               c3=dict(readout=readout, z=z), spec=spec_record(spec), argv=list(argv or []), wall_s=time.time() - t0)
    report = write_report(out, rid, name, res, [c3])
    o_measure.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                  [c3])
    print(res.get("sentence", ""))
    return code
