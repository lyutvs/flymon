"""The helpers every CLI of spec appendix N shares (N.8.9, plan readings 12-14): one copy, the scripts import them.
- l_cli's refusal, run id, commit check, C3 record and manifest check, re-exported.
- N's --out rule (results/n/), code keys over N's files, the block order, the summary gate and the later-block refusal.
- The spec-note gate: an N.8a / N.8b paragraph committed in the spec at HEAD, citing the upstream block's run id.
- The hooks: a script imports the hook names and passes `hooks(its own module)`, so its tests patch them on the script.
- The pool and measurer (timeout and worker default from n_spec), the stimuli at a point and their drive record.
- The arm helpers of N2.0 / N2: the arm items of a condition list, the rows grouped per condition with every declared
  seed present (else ValueError), the firing-state shares (an aggregate: the per-presentation state stays in the raw
  per-(condition, seed) cache entries), and the KC vector blocks per stimulus.
- `main_stage`: every refusal before any pool, in order, then the stage body, the report and the summary block; with
  `after`, a second step that runs only once the block is written (the judge's post-verdict record).
Every number is n_spec's; an n_rules error in a body (missing or incomplete input) is never caught into an outcome."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import h3_store, n_store
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import check_committed, load_c3_record, other_code, refuse, run_id, same_code  # noqa: F401
from .n_jobs import EDITS
from .n_measure import HASHED_FILES, MEASURE_FILES, NCache, NMeasurer
from .n_rules import state
from .n_spec import SPEC, NSpec, smoke
from .odor_real import DataMismatch, cap_hz, glomerular, load_table, stimuli

POOL_TIMEOUT_S = SPEC.pool_timeout_s     # n_spec's (one item per worker per round); make_measurer reads ctx.spec's
ORDER = ("n0f", "n0", "n1", "n2_0", "n2")
NPZ = "data/malecns.npz"
RUN_OUT = n_store.ALLOWED_DIR + "run"
SMOKE_OUT = n_store.ALLOWED_DIR + "smoke"
SMOKE_SUMMARY = f"{SMOKE_OUT}/{Path(n_store.SUMMARY).name}"
HOOK_NAMES = ("check_committed", "code_keys", "git_state", "head_spec", "load_c3", "m0d_sha", "make_measurer",
              "model_types")
UNEDITED = EDITS[0]                      # "none": the APL-on engine, the only one N0, N1 and N2.0 ever run
DRIVE_KEYS = ("drive_hz", "clipped_hz", "clipped_total_hz", "capped", "weights")


def hooks(module) -> dict:
    """The hook functions as `module` (a script) holds them now: tests patch them on the script module."""
    return {n: getattr(module, n) for n in HOOK_NAMES}


def out_allowed(out) -> bool:
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(n_store.ALLOWED_DIR)


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES); a missing hashed file is a refusal (exit 2)."""
    missing = [f for f in HASHED_FILES if not (ROOT / f).exists()]
    if missing:
        refuse(f"the hashed files {missing} do not exist (the manifest covers every N module, script and data file)")
        raise SystemExit(2)
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=HASHED_FILES)


def read_previous(summary, need, smoke: bool, committed) -> tuple:
    """(doc, None) or (None, refusal): outside --smoke the summary is git-tracked and clean at HEAD (`committed`) when
    any block is needed; then every needed block is present. --smoke reads only a summary under results/n/."""
    if smoke:
        if not out_allowed(summary):
            return None, f"a --smoke run reads and writes only a smoke summary under {n_store.ALLOWED_DIR}, not {summary}"
    elif need:
        why = committed(summary, list(need))
        if why:
            return None, why
    try:
        doc = json.loads(Path(summary).read_text()) if need or Path(summary).exists() else {}
    except (OSError, ValueError) as e:
        return None, f"no usable summary at {summary}: {e}"
    missing = [b for b in need if not isinstance(doc.get(b) if isinstance(doc, dict) else None, dict)]
    if missing:
        return None, f"{summary} lacks the blocks {missing} (each earlier stage's block comes first)"
    return doc, None


def later_blocks(name: str, doc: dict) -> str | None:
    later = [b for b in ORDER[ORDER.index(name) + 1:] if isinstance(doc, dict) and b in doc]
    return None if not later else (f"the summary already holds the later blocks {later}: block {name} is not "
                                   f"rewritten under them (start a new summary, or --smoke)")


def head_spec(path) -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def spec_note(text: str | None, marker: str, rid) -> str | None:
    """Plan reading 12: a line starting "**N.8a" (or a heading "### N.8a") in the committed spec, and the upstream
    block's run id somewhere after it."""
    if text is None:
        return "the spec is not tracked at HEAD"
    m = re.search(rf"^(?:#+\s*|\*\*){re.escape(marker)}\b", text, re.M)
    if not m:
        return (f"the spec at HEAD has no {marker} paragraph: write it (scripts/write_n_notes.py) and commit it before "
                f"this stage (N.8.9)")
    if not rid or str(rid) not in text[m.start():]:
        return f"the spec's {marker} paragraph does not cite block run_id {rid}"
    return None


def load_c3(spec) -> tuple:
    return load_c3_record(spec.l.m0d_path, spec.l)


def m0d_sha(spec) -> str:
    return sha256_file(spec.l.m0d_path)


def model_types(npz) -> list:
    from .circuits import Populations
    from .connectome import Connectome
    return sorted(str(t) for t in Populations.from_connectome(Connectome.load(str(npz))).receptor_types)


def make_measurer(ctx) -> tuple:
    """(NMeasurer, its FlyPool): the one pool construction of N's CLIs. The caller closes the pool."""
    from . import fly_pool
    pool = fly_pool.FlyPool(ctx.args.npz, ctx.c3, flies=[{}] * ctx.args.workers, workers=ctx.args.workers,
                            punish_type=ctx.spec.h3.punish_type, reward_type=ctx.spec.h3.reward_type,
                            timeout_s=ctx.spec.pool_timeout_s)
    return NMeasurer(pool, ctx.spec, NCache(ctx.out / "cache", ctx.key, ctx.rid)), pool


def operating_point(doc: dict, spec, smoke: bool) -> tuple:
    sel = (doc.get("n0f") or {}).get("selected")
    if sel:
        return float(sel["g"]), float(sel["c_delta"])
    if smoke:
        return tuple(float(v) for v in spec.smoke_point)
    raise ValueError("block n0f selected no operating point")


def data_dir(spec) -> Path:
    """spec.data_dir, relative to the repository root unless absolute (a CLI never depends on its cwd for the data)."""
    p = Path(spec.data_dir)
    return p if p.is_absolute() else ROOT / p


def glomeruli(ctx) -> tuple:
    """(Table, per-glomerulus ΣΔ over the model's ORN types) from the pinned data; DataMismatch on any defect."""
    table = load_table(data_dir(ctx.spec), ctx.spec.sha_pins())
    return table, glomerular(table, ctx.types)


def stimuli_at(ctx, names, g: float, c_delta: float, glom: dict | None = None) -> dict:
    """odor_real.stimuli at one (g, c_δ) on C3's rate, H.3's strength and the refractory cap (`glom`: already loaded)."""
    glom = glomeruli(ctx)[1] if glom is None else glom
    return stimuli(glom, names, g, c_delta, ctx.spec.mixtures_dict(), ctx.c3.max_rate_hz, ctx.spec.h3.strength,
                   cap_hz(ctx.c3))


def drives(st: dict, names=None) -> dict:
    """The drive record of stimuli `st` (every one, or `names`): commanded Hz, clipped inhibition, capped channels."""
    return {s: {k: st[s][k] for k in DRIVE_KEYS} for s in (st if names is None else names)}


def arm_items(st: dict, spec, conds, seeds, plastic: bool) -> list:
    """NMeasurer.arms items: every (name, pair, edit) of `conds` on every seed, at the stimuli `st`."""
    ps = spec.pair_stimuli()
    return [dict(cond=name, edit=edit, odor_x=st[ps[pair][0]]["odor"], odor_y=st[ps[pair][1]]["odor"], seed=int(s),
                 plastic=bool(plastic)) for name, pair, edit in conds for s in seeds]


def by_condition(rows, names, seeds, what: str) -> dict:
    """{condition: its rows in seed order}. ValueError unless every name of `names` holds exactly `seeds`, each once,
    and no row belongs elsewhere: a short or empty measurement never reaches a rule."""
    names, want = list(names), sorted(int(s) for s in seeds)
    if not names or not want:
        raise ValueError(f"{what}: no condition or no seed was declared ({names}, {want})")
    by = {c: [] for c in names}
    for r in rows:
        if r.get("cond") not in by:
            raise ValueError(f"{what}: a row of the undeclared condition {r.get('cond')!r}")
        by[r["cond"]].append(r)
    for c in names:
        by[c].sort(key=lambda r: int(r["seed"]))
        got = [int(r["seed"]) for r in by[c]]
        if got != want:
            raise ValueError(f"{what}: rows are missing for condition {c}: {len(got)} of {len(want)} declared seeds "
                             f"(lacking {sorted(set(want) - set(got))[:8]}, extra or repeated "
                             f"{sorted((Counter(got) - Counter(want)).elements())[:8]})")
    return by


def firing_shares(by: dict, spec) -> dict:
    """N.8.8's record per condition: the share of its probes (pre / post x X / Y over the seeds) in the firing state."""
    out = {}
    for c, rows in by.items():
        f = [state(r[ph][k]["P"], spec) == "firing" for r in rows for ph in ("pre", "post") for k in ("x", "y")]
        if not f:
            raise ValueError(f"condition {c} has no probe to read a state from")
        out[c] = sum(f) / len(f)
    return out


def kc_blocks(rows, names, seeds) -> tuple:
    """({name: its KC-vector rows}, the KC count) of one kc_vectors call over `names`; ValueError unless every name has
    a row for every seed and the rows agree on the KC count."""
    names = list(names)
    n_kc = {int(r["n_kc"]) for rr in rows for r in rr}
    if len(rows) != len(names) or any(len(rr) != len(seeds) for rr in rows) or len(n_kc) != 1:
        raise ValueError(f"KC vectors are missing: {len(rows)} of {len(names)} stimuli, seeds per stimulus "
                         f"{sorted({len(rr) for rr in rows})} for {len(seeds)}, KC counts {sorted(n_kc)}")
    return dict(zip(names, rows)), n_kc.pop()


def spec_record(spec) -> dict:
    return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "l"}


def write_report(out, rid: str, name: str, res: dict, params_list) -> Path:
    base = Path(out) / "runs" / f"{rid}-{name}"
    path = n_store.write_json(Path(f"{base}.json"), res, params_list)
    md = f"# N {name} — {rid}\n\n{res.get('sentence', '')}\n\noutcome: `{res.get('outcome')}`\n"
    n_store.write_bytes(Path(f"{base}.md"), md.encode(), params_list)
    return path


@dataclass
class Ctx:
    args: argparse.Namespace
    spec: NSpec
    smoke: bool
    doc: dict
    rid: str
    out: Path
    key: dict
    c3: object
    readout: dict
    z: dict
    types: list
    hooks: dict


def parser(doc_help: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--workers", type=int, default=SPEC.workers)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    return ap


def main_stage(name: str, argv, body, hooks: dict, *, need=(), outcomes=None, note=None, spec=None,
               require_root: bool = True, doc_help: str | None = None, after=None) -> int:
    """Refusals (exit 2) before any pool, in order:
    1. another directory;
    2. --out outside results/n/;
    3. dirty hashed files (unless --allow-dirty);
    4. the summary gate (committed outside --smoke; blocks present);
    5. a later block (outside --smoke);
    6. an upstream outcome not in `outcomes` (outside --smoke; recorded in smoke_bypass under --smoke);
    7. the spec note `note` = (marker, block) (outside --smoke);
    8. an upstream block produced on another run of its own upstream;
    9. other code;
    10. another m0d.json than block n0f's;
    11. no usable C3 record.
    Then body(ctx) -> (res, exit code); a DataMismatch inside a later stage's body is a refusal. Any other error of a
    body (n_rules raises on missing or incomplete input) propagates: nothing is written and the exit is non-zero. The
    report goes under <out>/runs/ and block `name` into the summary (a --smoke run: its smoke summary).
    `after(ctx, res) -> dict` runs only once that block is on disk; its fields are added and the block is written again
    (N.8.6: the judge's block-condition KC record comes after its written verdict). If `after` raises, the block
    written before it stays as it is."""
    a = parser(doc_help).parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or (SMOKE_OUT if a.smoke else RUN_OUT))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {n_store.ALLOWED_DIR} of the repository root")
    summary = Path(a.summary or (SMOKE_SUMMARY if a.smoke else n_store.SUMMARY))
    git = hooks["git_state"](HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    doc, why = read_previous(summary, list(need), a.smoke, hooks["check_committed"])
    if why:
        return refuse(why)
    if not a.smoke:
        why = later_blocks(name, doc)
        if why:
            return refuse(why)
    bypass = []
    for b, allowed in (outcomes or {}).items():
        got = doc[b].get("outcome")
        if got not in allowed:
            if not a.smoke:
                return refuse(f"block {b} ended {got}, not {list(allowed)}: every STOP goes to the user (N.6)")
            bypass.append(f"{b}={got}")
    if note and not a.smoke:
        why = spec_note(hooks["head_spec"](spec.spec_path), note[0], doc[note[1]].get("run_id"))
        if why:
            return refuse(why)
    for b in need:
        for ub, urid in (doc[b].get("upstream") or {}).items():
            if (doc.get(ub) or {}).get("run_id") != urid:
                return refuse(f"block {b} was produced on another {ub} run ({urid}; the summary holds "
                              f"{(doc.get(ub) or {}).get('run_id')})")
    key, manifest = hooks["code_keys"](a.npz)
    if need:
        why = other_code(doc, list(need), key["key"], manifest["key"], same_code)
        if why:
            return refuse(why)
    m0d = hooks["m0d_sha"](spec)
    if need and ((doc["n0f"].get("inputs") or {}).get("m0d_sha256") != m0d):
        return refuse(f"results/summary/m0d.json (sha256 {m0d[:12]}) is not the one block n0f ran on")
    try:
        c3, readout, z, _ = hooks["load_c3"](spec)
    except ValueError as e:
        return refuse(str(e))
    rid = run_id()
    ctx = Ctx(args=a, spec=spec, smoke=a.smoke, doc=doc, rid=rid, out=out, key=key, c3=c3, readout=dict(readout),
              z=z, types=hooks["model_types"](a.npz), hooks=hooks)
    t0 = time.time()
    try:
        res, code = body(ctx)
    except DataMismatch as e:
        return refuse(f"the data no longer match the pins block n0f ran on: {e}")
    res = dict(res, name=name, run_id=rid, smoke=a.smoke, smoke_bypass=bypass, measure_key=key["key"], code=manifest,
               git=git, inputs=dict(m0d_sha256=m0d, data_sha256=spec.sha_pins()),
               upstream={b: doc[b].get("run_id") for b in need}, c3=dict(readout=readout, z=z),
               spec=spec_record(spec), argv=list(argv or []), wall_s=time.time() - t0)

    def write(res: dict) -> None:
        report = write_report(out, rid, name, res, [c3])
        n_store.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                    [c3])

    write(res)
    if after is not None:
        res = dict(res, **after(ctx, res))
        write(dict(res, wall_s=time.time() - t0))
    print(res.get("sentence", ""))
    return code
