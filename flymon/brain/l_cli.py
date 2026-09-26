"""The helpers every CLI of spec appendix L shares (plan ruling P16: one copy instead of five): refusals, the --out rule,
the run id, the dirty check, the code keys and the manifest check (a block produced under other code is refused — plan
reading 16, Review Focus 4), the commit and block gate of the summary, the new pair set cut to a spec, the J.12.9
provenance record and the summary-block write. The scripts import these names, so their tests patch them on the script
module (the gate helpers take the script's check_committed / same_code as arguments for that reason)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from . import h3_store, j_store, l_pairs, l_store
from .h3_store import ROOT, code_key, sha256_file
from .l_measure import HASHED_FILES, MEASURE_FILES

POOL_TIMEOUT_S = 1800


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def out_allowed(out) -> bool:
    """--out must lie under results/m0d/l/ of the repository root (normalised like l_store.guard)."""
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(l_store.ALLOWED_DIR)


def run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES). Every L file exists now: a missing hashed file is a
    refusal (exit 2, naming it), never a smaller manifest."""
    missing = [f for f in HASHED_FILES if not (ROOT / f).exists()]
    if missing:
        refuse(f"the hashed files {missing} do not exist (the manifest covers every L module and script)")
        raise SystemExit(2)
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=HASHED_FILES)


def same_code(block: dict, key: str, manifest: str) -> bool:
    """A previous block is usable only under this measure key and this procedure manifest."""
    return block.get("measure_key") == key and (block.get("code") or {}).get("key") == manifest


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _rel(path) -> str | None:
    rel = os.path.relpath(os.path.abspath(str(path)), ROOT).replace(os.sep, "/")
    return None if rel.startswith("../") else rel


def check_committed(summary, blocks) -> str | None:
    """None when the summary is git-tracked and unchanged against HEAD (stage2_oc.check_tracked's test), else the
    refusal reason (it says "tracked" or "commit")."""
    rel = _rel(summary)
    if rel is None or _git("ls-files", "--error-unmatch", rel).returncode != 0:
        return f"{summary} is not tracked by git: commit its blocks {list(blocks)} before this stage"
    if _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{summary} has uncommitted changes: commit its blocks {list(blocks)} before this stage"
    return None


def read_previous(summary, need, smoke: bool, committed, allowed=out_allowed) -> tuple:
    """(doc, None) or (None, the refusal): the commit gate of plan reading 16 in its order — (outside --smoke) the
    summary git-tracked and clean at HEAD (`committed`, the script's check_committed), then every block of `need` present
    (the refusal lists each missing one). A --smoke run skips the commit gate but reads only a summary under
    results/m0d/l/. The manifest check (other_code) comes after, once the code keys are known."""
    if smoke:
        if not allowed(summary):
            return None, f"a --smoke run reads and writes only a smoke summary under {l_store.ALLOWED_DIR}, not {summary}"
    else:
        why = committed(summary, list(need))
        if why:
            return None, why
    try:
        doc = json.loads(Path(summary).read_text())
    except (OSError, ValueError) as e:
        return None, f"no usable summary at {summary}: {e}"
    missing = [b for b in need if not isinstance(doc.get(b) if isinstance(doc, dict) else None, dict)]
    if missing:
        return None, f"{summary} lacks the blocks {missing} (each earlier stage's block comes first, committed)"
    return doc, None


def other_code(doc: dict, need, key: str, manifest: str, same) -> str | None:
    """The refusal naming every block of `need` produced under other code (`same`: the script's same_code), or None."""
    bad = [b for b in need if not same(doc[b], key, manifest)]
    if not bad:
        return None
    got = {b: (str(doc[b].get("measure_key"))[:12], str((doc[b].get("code") or {}).get("key"))[:12]) for b in bad}
    return (f"the blocks {bad} were produced under other code (measure key / manifest {got}; this code's are "
            f"{key[:12]} / {manifest[:12]})")


def rule_text(rule) -> str:
    """A stage-1 rule as the sentences print it."""
    if rule is None:
        return "none"
    return "G" if rule["family"] == "G" else f"G ∧ {rule['family']} {rule['op']} {float(rule['t']):.6g}"


def new_set(pops, spec, declared):
    """l_pairs.new_pairs on the declared spec, refused (ValueError) unless both digests are the pinned ones
    (l_pairs.check_digests), then cut to `spec`'s turns: (b) and skipped of turns < spec.n_turns, (a) of turns <
    spec.a_turns. Dedupe runs in declared order, so a smoke spec gets a prefix of the declared list."""
    s = l_pairs.check_digests(l_pairs.new_pairs(pops, declared), declared)
    return dict(b=[p for p in s["b"] if int(p["turn"]) < spec.n_turns],
                a=[p for p in s["a"] if int(p["turn"]) < spec.a_turns],
                skipped=[k for k in s["skipped"] if int(k["key"][1]) < spec.n_turns])


def load_c3_record(m0d_path, spec) -> tuple:
    """(C3's Params, H.4's C3 readout, z, pools) from results/summary/m0d.json (plan reading 1); ValueError naming what is
    unusable: no C3 in block h3, no C3 record in block h4, another readout than the declared one, a missing z."""
    try:
        m0d = json.loads(Path(m0d_path).read_text())
        c3, _, _ = j_store.load_c3(m0d, spec.j.h3_block, spec.j.c3_name)
        rec = m0d["h4"]["h4"]["combos"][spec.j.c3_name]
        readout, z, pools = rec["readout"], rec.get("z"), m0d["h4"]["pools"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"no usable C3 record in {m0d_path}: {e!r}") from None
    if readout != dict(spec.readout):
        raise ValueError(f"block h4's C3 readout {readout} is not the declared {dict(spec.readout)}")
    if not z or any(k not in z or len(z[k]) != 2 for k in ("A", "P")):
        raise ValueError(f"block h4's C3 z is missing or malformed: {z}")
    return c3, readout, z, pools


def head_sha256(path) -> str | None:
    """sha256 of the file's committed blob at HEAD (None when untracked)."""
    rel = _rel(path)
    if rel is None:
        return None
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def provenance(files, args: dict, argv) -> dict:
    """J.12.9's record: the inputs' sha256, HEAD, the dirty ones among them, the arguments."""
    sha = {(_rel(f) or str(f)): sha256_file(f) for f in files}
    rels = [r for r in (_rel(f) for f in files) if r]
    dirty = [ln[3:] for ln in _git("status", "--porcelain", "--", *rels).stdout.splitlines() if ln.strip()]
    return dict(sha256=sha, commit=_git("rev-parse", "HEAD").stdout.strip(), dirty=dirty, args=args, argv=list(argv))


def guard_params(base, measurers) -> list:
    return [base] + list(dict.fromkeys(p for m in measurers for p in m.params_seen if p != base))


def write_block(summary, name: str, res: dict, report, params_list) -> Path:
    """The summary block = the run record + its report path and sha256, through l_store's guarded write."""
    return l_store.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                       params_list)


# ---------------------------------------------------------------- the chain between the blocks (final review)
ORDER = ("stage0", "oc", "stage1", "stage2", "stage3")


def later_blocks(summary, name: str, doc: dict | None = None) -> str | None:
    """The refusal when the summary already holds a block of a stage after `name` (rewriting `name` would orphan it),
    or None. `doc`: the summary already read; otherwise it is read when it exists (a missing summary has none)."""
    if doc is None:
        try:
            doc = json.loads(Path(summary).read_text())
        except OSError:
            return None
        except ValueError as e:
            return f"no usable summary at {summary}: {e}"
    later = [b for b in ORDER[ORDER.index(name) + 1:] if isinstance(doc, dict) and b in doc]
    if not later:
        return None
    return (f"{summary} already holds the later blocks {later}: block {name} is not rewritten under them (start a new "
            f"summary, or --smoke)")


def run_id_chain(doc: dict, block: str, links: dict) -> str | None:
    """The refusal naming every upstream run id that block `block` recorded (links: {field: upstream block}) and that
    is missing or differs from that block's run_id, or None."""
    bad = {f: (doc[block].get(f), doc[u].get("run_id")) for f, u in links.items()
           if doc[block].get(f) is None or doc[block].get(f) != doc[u].get("run_id")}
    if not bad:
        return None
    what = ", ".join(f"{f} {got} != block {links[f]} run_id {want}" for f, (got, want) in bad.items())
    return f"block {block} was produced on other upstream runs ({what})"


def _norm(x):
    return json.loads(json.dumps(x))


def stage0_inputs(s0: dict, m0d_path, **record) -> str | None:
    """The refusal when the m0d summary (by sha256) or C3's record (record: c3 / readout / z / pools, each against the
    value block stage0 recorded) differs from what stage 0 ran on, or None."""
    want = ((s0.get("inputs") or {}).get("m0d") or {}).get("sha256")
    try:
        got = sha256_file(m0d_path)
    except OSError as e:
        return f"the m0d summary {m0d_path} is not readable: {e}"
    if not want or got != want:
        return (f"the m0d summary {m0d_path} (sha256 {got[:12]}) is not the one stage 0 ran on "
                f"(block stage0 inputs.m0d.sha256 {str(want)[:12]})")
    bad = [k for k, v in record.items() if _norm(v) != _norm(s0.get(k))]
    if bad:
        return f"C3's {bad} from {m0d_path} differ from what block stage0 recorded (stage 0 ran on another engine)"
    return None
