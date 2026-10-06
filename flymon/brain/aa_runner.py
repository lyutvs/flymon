"""Spec AA's stage chain (AA.7 6245–6264): stage0 (0a) → reuse (1) → smoke (2) → seal_code (3) → screen (4) → coverage
(4b) → learn (5) → estimate (6) → records (7), one block each in results/summary/aa_learning.json, written only through
aa_store. AA.2 6181–6184: the stage assembly is AA's own; Y's Runner is used only for its read methods (_doc, _z_b,
_theta_b) and Y's ctx only for pilot_back, through ctx; W's ctx (w_runner.build_ctx, read-only) gives main_rows
(= w_pairs.main_rows), params, readout and the V jm:L rows; W / Y / X pure functions are imported; nothing writes
through y_store / z_store / w_store and nothing assigns to a W / X / Y / Z module (plan Global Constraints).
- Chain: refusals (exit 2) as Z's; non-stage keys (budget, candidates, invalid, defect_<n>, reseal_<n>, <stage>_v<n>)
  are not stages. Stages after reuse also refuse when ctx's measurement keys differ from Y's (W.8 form).
- Env (AA.7 0a, Z.9.2 P2-11 form): env_now() at every start / resume vs every committed block and the stage's first
  start (progress/<stage>.env.json); a difference → core-ledger entry {stage, env_mismatch}, exit EXIT_ENV.
- Seal (AA.7 3): coverage / estimate / records compare seal_now() with block seal_code (or the latest reseal_<n>);
  a difference → exit EXIT_SEAL (plan Reading 15).
- Candidates (AA.3 6196, plan Reading 8): the tracked block is rewritten at every block write from
  progress/<stage>.candidates.json (states change at batch start inside a stage)."""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import aa_estimate, aa_rules, aa_store, w_records, w_verdict, x_oc, y_rules, z_split
from .r_store import load_manifest
from .aa_spec import SPEC as AA
from .h3_store import ROOT, canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .r_pairs import row_key
from .v_spec import SPEC as V_SPEC
from .w_measure import KIND
from .w_spec import BRAINS
from .w_spec import SPEC as W_SPEC
from .y_spec import SPEC as Y_SPEC

EXIT_ENV, EXIT_SEAL = 6, 7
YXW_GLOBS = ("y_*.py", "x_*.py", "w_*.py")
EXTRA_FILES = ("flymon/brain/z_split.py",)
WINDOWS = dict(strength=W_SPEC.strength, settle_ms=V_SPEC.settle_ms, read_ms=V_SPEC.read_ms,
               window_ms=V_SPEC.window_ms)
TIMING = dict(present_ms=W_SPEC.pulse_ms, gap_ms=W_SPEC.gap_ms, train_settle_ms=W_SPEC.train_settle_ms,
              seed_stride=W_SPEC.fly_train_stride)


def aa_files() -> list:
    out = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "flymon/brain").glob("aa_*.py"))
    return out + (["scripts/run_aa.py"] if (ROOT / "scripts/run_aa.py").exists() else [])


def yxw_files() -> list:
    return sorted(p.relative_to(ROOT).as_posix() for g in YXW_GLOBS for p in (ROOT / "flymon/brain").glob(g)) + \
        list(EXTRA_FILES)


def files_sha(files) -> dict:
    return {f: sha256_file(ROOT / f) for f in files}


def env_now() -> dict:
    import scipy
    return dict(uv_lock_sha256=sha256_file(ROOT / "uv.lock"), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, platform=platform.platform(), aa_files=files_sha(aa_files()),
                yxw_files=files_sha(yxw_files()))


def hashed_files() -> list:
    return aa_files() + yxw_files() + [Y_SPEC.summary, AA.z_summary]


def git_state() -> dict:
    return _h3_git_state(files=hashed_files())


def seal_now(s=AA) -> dict:
    files = files_sha(s.seal_files)
    nums = json.loads(canonical(s.seal_fields()))
    return dict(files=files, numbers=nums,
                key=hashlib.sha256(canonical(dict(files=files, numbers=nums)).encode()).hexdigest())


# ================================================================ units (W Runner.units' rule on Y's / AA's seeds)
def aa_w_spec(s=AA, ys=Y_SPEC):
    return dataclasses.replace(W_SPEC, probe_seed0=ys.probe_seed0, train_seed0=ys.train_seed0,
                               pilot_probe_seed0=ys.pilot_probe_seed0, pilot_train_seed0=ys.pilot_train_seed0,
                               smoke_probe_seed0=s.smoke_probe_seed0, smoke_train_seed0=s.smoke_train_seed0)


def smoke_seed_set(s=AA) -> frozenset:
    return frozenset(s.smoke_probe_seed0 + f * s.smoke_probe_span + k for f in range(s.smoke_fly_span)
                     for k in range(s.smoke_probe_span))


def unit(r, idx, fly, brain, edit, phases, seeds, plastic=True) -> dict:
    return dict(pair=row_key(r), idx=idx, fly=int(fly), brain=brain, edit=edit, odor_x=dict(r["odor_x"]),
                odor_y=dict(r["odor_y"]), phases=phases, probe_seeds=[int(x) for x in seeds], plastic=bool(plastic))


def units(rows, where, K, F, edit, brains=BRAINS, plastic=True, ws=None) -> list:
    """where = "main" (c = row["c"], Y's 62M / 64M), "pilot" (j = position, Y's 60M / 61M) or "smoke" (87_1xx)."""
    sp, out = ws or aa_w_spec(), []
    for j, r in enumerate(rows):
        for f in range(F):
            if where == "main":
                c = int(r["c"])
                seeds, b0, b1 = sp.probe_seeds(c, f, K), sp.train_base(c, 0), sp.train_base(c, 1)
            elif where == "pilot":
                c = j
                seeds, b0, b1 = sp.pilot_probe_seeds(j, f, K), sp.pilot_train_base(j, 0), sp.pilot_train_base(j, 1)
            else:
                c = "smoke"
                seeds, b0, b1 = sp.smoke_probe_seeds(f, K), sp.smoke_train_base(0), sp.smoke_train_base(1)
            for b in brains:
                ph = [] if b == "naive" else sp.phases(b if b in BRAINS else "R", b0, b1)
                out.append(unit(r, c, f, b, edit, ph, seeds, plastic))
    return out


def declared(edit: str, seeds_by_fly: dict) -> dict:
    lever = edit == V_SPEC.lever_edit
    return dict(edit=edit, csc_sha256=V_SPEC.sha_combined if lever else V_SPEC.sha_none,
                edit_edges=V_SPEC.lever_edges if lever else 0,
                block_edges=dict(V_SPEC.contrast_declared()["chain_entry"]) if lever else {}, probe_seeds=seeds_by_fly)


def pair_counts(rows: list, flies: list) -> dict:
    return w_records.pair_data(rows, flies)


def by_pair(rows: list) -> dict:
    out = {}
    for r in rows:
        out.setdefault(r["unit"]["pair"], []).append(r)
    return out


def manifest(rows: list) -> list:
    return [dict(key=f"{r['unit']['pair']}|{r['unit']['fly']}|{r['unit']['brain']}", cache_file=r["cache_file"],
                 cache_key=r["cache_key"], sha256=sha256_file(r["cache_file"])) for r in rows]


def _js(o):
    return json.loads(canonical(aa_store.to_json(o)))


def _same(a, b) -> bool:
    """Bit equality through the canonical JSON text (NaN / ±inf compare equal to themselves)."""
    return canonical(aa_store.to_json(a)) == canonical(aa_store.to_json(b))


# ================================================================ raw counts of the screen (AA.7 4 6258)
SETS = ("S1", "S31", "out")                   # AA.3: the primary set, all 31, the filter-dropped pairs


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def counts_f(stage: dict) -> np.ndarray:
    """w_records.counts' [K, 2, 2] layout as float, so a non-finite stored count is seen (machine check), not a crash."""
    return np.array([[[_num(x.get("A")), _num(y.get("A"))], [_num(x.get("P")), _num(y.get("P"))]]
                     for x, y in zip(stage["x"], stage["y"])], float).reshape(-1, 2, 2)


def raw_digest(results: list) -> str:
    """AA.7 4: sha256 of the count arrays of every probe stage of every unit, in unit order (x_oc.digest form: int64
    bytes with the shape prefixed); unequal shapes or non-finite counts (a machine-check case) hash per array."""
    arrs = [counts_f(st) for r in results for st in r["stages"]]
    if arrs and all(a.shape == arrs[0].shape for a in arrs) and all(np.isfinite(a).all() for a in arrs):
        return x_oc.digest(np.stack(arrs).astype(np.int64))
    return hashlib.sha256("".join(x_oc.digest(a) for a in arrs).encode()).hexdigest()


def _cov_job(sizes, method, set_tag, ci, s) -> dict:
    """One coverage cell for Runner._pcells (module level: the spawn pool pickles it by name)."""
    return dict(arrays={}, meta=dict(share=aa_estimate.coverage_cell(tuple(sizes), method, set_tag, ci, s)))


# ================================================================ the differential test (AA.7 0 6248)
def differential(ctx, s=AA, ys=Y_SPEC) -> dict:
    """AA's assembly of Y's four V pilot pairs (Y pilot seeds, block "pilot") against Y's: unit inputs = the stored
    inputs of Y's manifest entries (cache key equal), cache hits only (a miss fails; nothing is measured), the count
    arrays = Y's pilot_back on the same manifest, w_verdict.gate_stats equal (and = pilot.json record_all fly_stats),
    final_filter of pre = pilot.json admission."""
    det = ctx["y_pilot_detail"]()
    want = set(ys.pilot_v_pairs)
    mlist = [m for m in det["manifest"] if m["key"].rsplit("|", 2)[0] in want]
    man = {m["key"]: m for m in mlist}
    wm = ctx["y_reader"]()
    z = ctx["y_z"]()
    U = units(ctx["pilot_rows"](), "pilot", ys.pilot_probes, ys.pilot_flies, V_SPEC.lever_edit)
    why, got = [], []
    ukeys = [f"{u['pair']}|{u['fly']}|{u['brain']}" for u in U]
    for name, ks in (("AA 단위", ukeys), ("Y 매니페스트", [m["key"] for m in mlist])):
        dup = sorted({k for k in ks if ks.count(k) > 1})
        if dup:
            why.append(f"{name} 키 중복 {len(dup)}개: {dup[:3]}")
    if len(U) != len(man):
        why.append(f"단위 수 {len(U)} ≠ Y 매니페스트 {len(man)}")
    for u in U:
        key = f"{u['pair']}|{u['fly']}|{u['brain']}"
        m = man.get(key)
        ins = wm.inputs(u, "pilot")
        if m is None:
            why.append(f"{key}: Y 매니페스트에 없음")
            continue
        if wm.cache.key(KIND, ins) != m["cache_key"]:
            why.append(f"{key}: 캐시 키 ≠ Y")
        res = wm.cache.get(KIND, ins)
        if res is None:
            why.append(f"{key}: 캐시 적중 실패(새 측정 없음)")
            continue
        if not _same(ins, json.loads(Path(m["cache_file"]).read_text())["inputs"]):
            why.append(f"{key}: 단위 입력 ≠ Y")
        got.append(dict(unit=u, result=res))
    ypairs, bad = ctx["y_pilot_back"](list(man.values()))
    why += [f"Y 경로: {b}" for b in bad]
    flies = list(range(ys.pilot_flies))
    per = {}
    n_pair = len(flies) * len(BRAINS)
    for k, rows in by_pair(got).items():
        if len(rows) != n_pair:
            why.append(f"{k}: 적중 단위 {len(rows)} ≠ {n_pair}")
            continue
        mine, theirs = pair_counts(rows, flies), (ypairs or {}).get(k)
        ok_c = theirs is not None and all(np.array_equal(mine[st], theirs[st]) for st in w_verdict.STAGES)
        g1 = w_verdict.gate_stats(mine, z)
        ok_g = theirs is not None and np.array_equal(g1, w_verdict.gate_stats(theirs, z), equal_nan=True)
        rec = ((det.get("record_all") or {}).get("pairs") or {}).get(k, {}).get("fly_stats")
        ok_r = rec is not None and _same(rec, {g: g1[:, i].tolist() for i, g in enumerate(w_verdict.GATES)})
        adm = (det.get("admission") or {}).get(k)
        ok_a = adm is not None and _same(adm, y_rules.final_filter(mine["pre"], z, ys))
        per[k] = dict(counts_equal=bool(ok_c), gate_stats_equal=bool(ok_g), record_equal=bool(ok_r),
                      admission_equal=bool(ok_a))
        why += [f"{k}: {n} 불일치" for n, v in per[k].items() if not v]
    why += [f"{k}: 대조 안 됨" for k in ys.pilot_v_pairs if k not in per]
    return dict(ok=not why, reasons=why, n_units=len(U), n_hits=len(got), pairs=per)


def refuse(msg: str, code: int = 2):
    aa_store.refuse(msg, code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


# ================================================================ the engine context (plan Reading 3; read only)
def build_ctx(npz: str, s=AA) -> dict:
    """W's context (w_runner.build_ctx, built once, lazily) for main_rows / params / readout / the V jm:L rows of Y's four
    V pilot pairs; Y's ctx (y_runner.build_ctx, lazy) only for Y Runner read methods and pilot_back; nothing writes."""
    import contextlib
    import io
    import subprocess

    from . import w_runner, y_runner
    from .u_measure import u_measure_key
    from .w_measure import WMeasurer, w_measure_key
    cache = {}

    def wctx():
        if "w" not in cache:
            cache["w"] = w_runner.build_ctx(W_SPEC, npz)
        return cache["w"]

    def yctx():
        if "y" not in cache:
            cache["y"] = y_runner.build_ctx(npz)
        return cache["y"]

    def yrun():
        return y_runner.Runner(yctx(), Y_SPEC)

    def wkey():
        if "wk" not in cache:
            cache["wk"] = w_measure_key(npz)
        return cache["wk"]

    def keys():
        return dict(w_measure_key=wkey()["key"], u_measure_key=u_measure_key(npz)["key"])

    def pilot_rows():
        c = wctx()
        n = len(c["v_doc"]()["jm:L"]["manifest"])
        got = {row_key(r): r for r, _res in c["v_jm"]("L", n)}
        missing = [k for k in Y_SPEC.pilot_v_pairs if k not in got]
        if missing:
            raise ValueError(f"V 세트 후보 {missing}가 V jm:L에 없음")
        return [dict(got[k]) for k in Y_SPEC.pilot_v_pairs]

    def wm(pool, cache_obj):
        c = wctx()
        return WMeasurer(pool, cache_obj, c["params"], c["readout"], V_SPEC.p_type, W_SPEC.reward_dan,
                         W_SPEC.punish_dan, WINDOWS, TIMING)

    def measurer(pool, smoke=False):
        root = s.smoke_cache_dir if smoke else s.cache_dir
        return wm(pool, aa_store.AACache(root, wkey(), smoke_seeds=smoke_seed_set(s)))

    def y_reader():
        return wm(None, aa_store.ReadCache(Y_SPEC.cache_dir, wkey()))

    def y_theta(where: str) -> dict:
        r = yrun()
        doc = r._doc()
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                stop, th, tw, rv, _c, ks = r._theta_b(doc, where)
        except SystemExit:
            return dict(why=[f"Y θ̂ 재적합 거절: {err.getvalue().strip()}"])
        if stop:
            return dict(why=list(stop.get("reasons") or [stop.get("outcome")]))
        return dict(why=[], theta=th, theta_w=tw, r=rv, keys=ks, z=r._z_b(doc))

    def decl_sha(files: list, commit: str) -> dict:
        out = {}
        for f in files:
            r = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{f}"], capture_output=True)
            out[f] = hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
        return out

    def y_doc():
        return yrun()._doc()

    return dict(
        keys=keys, params=lambda: wctx()["params"], main_rows=lambda blk: wctx()["main_rows"](blk),
        pilot_rows=pilot_rows, measurer=measurer, y_reader=y_reader, y_doc=y_doc,
        y_z=lambda: yrun()._z_b(y_doc()), y_theta=y_theta,
        y_pilot_detail=lambda: json.loads(Path(Y_SPEC.pilot_detail).read_text()),
        y_pilot_back=lambda man: yctx()["pilot_back"](man),
        z_doc=lambda: aa_store.read_summary(s.z_summary), git_facts=w_runner.git_facts, decl_sha=decl_sha,
        lenient=lambda: z_split.lenient_items(Y_SPEC.oracle_detail, s.oracle_sha256, s.n_len))


# ================================================================ the runner
class Runner:
    def __init__(self, ctx: dict, s=AA, summary_path=None, pool=None):
        self.ctx, self.s, self.pool = ctx, s, pool
        self.summary_path = str(summary_path or s.summary)
        self.workers = None
        self._s = dict(core=0.0, records=0.0)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return aa_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        s = self.s
        if stage not in s.stages:
            refuse(f"{stage} is not an AA stage")
        self._clean(stage)
        doc = self._doc()
        order = list(s.stages)
        i = order.index(stage)
        missing = [b for b in order[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in order[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; AA never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; AA never rewrites a recorded block")
        stopped = [b for b in order[:i] if doc[b].get("outcome") != aa_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — AA stops there")
        if i > order.index("reuse"):
            k = self.ctx["keys"]()
            bad = [n for n in ("w_measure_key", "u_measure_key") if k.get(n) != getattr(Y_SPEC, n)]
            if bad:
                refuse(f"stage {stage}: measurement key(s) {bad} differ from Y's (W.8)")
        if stage in s.sealed_stages:
            self._seal_check(doc)
        self._env(stage, doc)
        return doc

    def _seal_check(self, doc: dict) -> None:
        """AA.7 3: seal_now() against block seal_code, or against the newest reseal_<n> (the defect procedure)."""
        rs = sorted((int(k.split("_", 1)[1]), k) for k in doc if re.fullmatch(r"reseal_\d+", k))
        blk = doc[rs[-1][1]] if rs else doc.get("seal_code")
        want = ((blk or {}).get("seal") or {}).get("key")
        if want is None or seal_now(self.s)["key"] != want:
            refuse(f"the sealed code differs from {rs[-1][1] if rs else 'seal_code'} (AA.7 3); defect procedure "
                   "first", EXIT_SEAL)

    def _env(self, stage: str, doc: dict) -> dict:
        """AA.7 0a (Z.9.2 P2-11 form) at the start and at every resume (module docstring)."""
        now = env_now()
        blocks = [(b, doc[b]["env"]) for b in self.s.stages if b in doc and "env" in doc[b]]
        p = Path(self.s.progress_dir) / f"{stage}.env.json"
        if p.exists():
            blocks.append((f"{stage} 첫 시작", json.loads(p.read_text())))
        why = aa_rules.env_reasons(blocks, now)
        if why:
            aa_store.append_ledger(self.summary_path, dict(stage=stage, env_mismatch=why, at=_now(), wall_s=0.0),
                                   self.plist)
            refuse(f"stage {stage}: environment differs (AA.7 0a): {why[:5]}", EXIT_ENV)
        if not p.exists():                  # recorded only once it agrees with every committed block
            aa_store.write_json(str(p), now, self.plist)
        return now

    def _detail(self, stage: str) -> str:
        return f"{self.s.raw_dir}/{stage}.json"

    def _files(self, stage: str) -> list:
        """The stage's detail file, plus the raw files of its manifest (screen, learn, records; AA.7 "블록마다 사본")."""
        p = Path(self._detail(stage))
        if not p.exists():
            return []
        out = [str(p)]
        if stage in ("screen", "learn", "records"):
            d = json.loads(p.read_text())
            out += [m["cache_file"] for m in d.get("manifest") or []]
        return list(dict.fromkeys(out))              # a duplicated unit (machine-check case) is one file

    # ---- candidates (AA.3 6196, plan Reading 8) ----------------------------------------------------------------------
    def _cands_path(self, stage: str) -> Path:
        return Path(self.s.progress_dir) / f"{stage}.candidates.json"

    def _cands(self, doc: dict, stage: str | None = None) -> list:
        if stage is not None and self._cands_path(stage).exists():
            return json.loads(self._cands_path(stage).read_text())
        if doc.get("candidates"):
            return [dict(x) for x in doc["candidates"]]
        keys = self.ctx["y_doc"]()["digest"]["set"]["keys"]
        return [dict(key=k, c=i, axis=k.split("|")[0], state=aa_rules.STATES[0]) for i, k in enumerate(keys)]

    def _cands_set(self, stage: str, keys: list, state: str) -> list:
        """States only move forward: untouched → screened → trained."""
        rank = {x: i for i, x in enumerate(aa_rules.STATES)}
        want = set(keys)
        cur = self._cands(self._doc(), stage)
        for x in cur:
            if x["key"] in want and rank[state] > rank[x["state"]]:
                x["state"] = state
        aa_store.write_json(str(self._cands_path(stage)), cur, self.plist)
        return cur

    # ---- blocks ------------------------------------------------------------------------------------------------------
    def _write(self, stage: str, body: dict, core_s=None, records_s=None) -> dict:
        s, extra = self.s, {}
        if body.get("outcome") != aa_rules.INVALID:
            extra["archive"] = aa_store.archive(self._files(stage), stage, s)
        k = self.ctx["keys"]()
        prog = self._prog(stage)
        block = aa_store.to_json(dict(body, **extra, stage=stage, order=s.stage_label(stage), env=env_now(),
                                      git=git_state(), written_at=_now(), w_measure_key=k.get("w_measure_key"),
                                      u_measure_key=k.get("u_measure_key")))
        core = float(prog["core"] if core_s is None else core_s)
        rec = float(prog["records"] if records_s is None else records_s)
        at = block["written_at"]
        if stage in s.records_stages:
            led, rled = None, dict(stage=stage, wall_s=rec, at=at)
        else:
            led, rled = dict(stage=stage, wall_s=core, at=at), (dict(stage=stage, wall_s=rec, at=at) if rec else None)
        doc = self._doc()
        aa_store.write_summary_block(self.summary_path, stage, block, self.plist, led, rled,
                                     candidates=self._cands(doc, stage))
        return block

    def archive(self, stage: str) -> list:
        blk = self._doc().get(stage)
        if blk is None:
            refuse(f"archive {stage}: no block {stage}")
        if blk.get("outcome") == aa_rules.INVALID:
            refuse(f"archive {stage}: block {stage} is INVALID; INVALID is never archived")
        return aa_store.archive(self._files(stage), stage, self.s)

    def run(self, stage: str) -> dict:
        if stage not in self.s.stages:
            refuse(f"{stage} is not an AA stage")
        fn = getattr(self, f"stage_{stage}", None)
        if fn is None:
            refuse(f"stage {stage} is not implemented yet")
        return fn()

    # ---- spend and cells -------------------------------------------------------------------------------------------
    def _prog_path(self, stage: str) -> Path:
        return Path(self.s.progress_dir) / f"{stage}.json"

    def _prog(self, stage: str) -> dict:
        p = self._prog_path(stage)
        d = json.loads(p.read_text()) if p.exists() else {}
        return dict(core=float(d.get("core", 0.0)), records=float(d.get("records", 0.0)))

    def _add(self, stage: str, sec: float, which: str = "core") -> None:
        d = self._prog(stage)
        d[which] += float(sec)
        aa_store.write_json(str(self._prog_path(stage)), dict(d, at=_now()), self.plist)

    def _finish(self, stage: str, t0: float) -> None:
        """The stage's own (non-cell) seconds go to its ledger."""
        which = "records" if stage in self.s.records_stages else "core"
        self._add(stage, (time.perf_counter() - t0) - self._s["core"] - self._s["records"], which)

    def _key(self, tag: str, key_obj) -> str:
        spec_sha = hashlib.sha256(canonical(self.s).encode()).hexdigest()
        return hashlib.sha256(canonical(dict(tag=tag, key=key_obj, s=spec_sha, aa=files_sha(aa_files()))).encode()
                              ).hexdigest()

    def _cell(self, stage: str, tag: str, key_obj, fn, which: str = "core"):
        key = self._key(tag, key_obj)
        p = Path(self.s.progress_dir) / stage / (_safe(tag) + ".json")
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except ValueError:
                d = {}
            if d.get("key") == key:
                return d["value"]
        t = time.perf_counter()
        v = aa_store.to_json(fn())
        aa_store.write_json(str(p), dict(key=key, value=v, at=_now()), self.plist)
        dt = time.perf_counter() - t
        self._add(stage, dt, which)
        self._s[which] += dt
        return v

    def _pcells(self, stage: str, jobs: list, which: str = "core", log=None) -> list:
        """jobs = [(tag, key_obj, fn, args)] with fn a module-level function returning {arrays, meta}; cells
        progress/<stage>/<tag>.npz; computed in a spawn pool of self.workers (or s.workers); results in job order."""
        import io
        root = Path(self.s.progress_dir) / stage

        def load(path, key):
            if not path.exists():
                return None
            try:
                with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as f:
                    if str(f["__key__"]) != key:
                        return None
                    return dict(arrays={k: f[k] for k in f.files if not k.startswith("__")},
                                meta=json.loads(str(f["__meta__"])))
            except Exception:
                return None

        def save(path, key, res):
            buf = io.BytesIO()
            np.savez(buf, __key__=np.array(key), __meta__=np.array(json.dumps(aa_store.to_json(res["meta"]))),
                     **res["arrays"])
            aa_store.write_bytes(str(path), buf.getvalue(), self.plist)
        out, todo = [None] * len(jobs), []
        for i, (tag, key_obj, fn, args) in enumerate(jobs):
            p = root / (_safe(tag) + ".npz")
            k = self._key(tag, key_obj)
            got = load(p, k)
            if got is None:
                todo.append((i, p, k, fn, args))
            else:
                out[i] = got
        last = time.perf_counter()

        def done(i, p, k, res):
            nonlocal last
            save(p, k, res)
            out[i] = dict(arrays=res["arrays"], meta=json.loads(json.dumps(aa_store.to_json(res["meta"]))))
            now = time.perf_counter()
            self._add(stage, now - last, which)
            self._s[which] += now - last
            last = now
            if log is not None:
                log(f"aa {stage}: {sum(o is not None for o in out)}/{len(out)} cells")
        workers = max(1, int(self.workers or self.s.workers))
        if workers == 1 or len(todo) <= 1:
            for i, p, k, fn, args in todo:
                done(i, p, k, fn(*args))
            return out
        for v in self.s.thread_env:
            os.environ.setdefault(v, "1")
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=mp.get_context("spawn")) as ex:
            try:
                fut = {ex.submit(fn, *args): (i, p, k) for i, p, k, fn, args in todo}
                for f in cf.as_completed(fut):
                    i, p, k = fut[f]
                    done(i, p, k, f.result())
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
        return out

    # ================================================================ 0a: stage0 (AA.7 0, 6246–6251)
    def stage_stage0(self) -> dict:
        self._require("stage0")
        s, t0 = self.s, time.perf_counter()
        bad = []
        log = Path(s.tests_log)
        lines = log.read_text().strip().splitlines() if log.exists() else []
        passed = [ln for ln in lines if re.search(r"\d+ passed", ln)]
        tests = dict(path=s.tests_log, log_sha256=sha256_file(log) if log.exists() else None,
                     last_line=lines[-1] if lines else None, passed_line=passed[-1] if passed else None)
        if tests["last_line"] != s.tests_ok_line:
            bad.append(f"시험 로그 {s.tests_log}의 마지막 줄 {tests['last_line']!r} ≠ {s.tests_ok_line!r}")
        if tests["passed_line"] is None:           # AA.7 0a records the passed-count line; -qq hides it → INVALID
            bad.append(f"시험 로그 {s.tests_log}에 통과 개수 줄(\"N passed\")이 없음")
        fails = [ln for ln in lines if re.search(r"\b\d+ (failed|errors?)\b", ln)]
        if fails:
            bad.append(f"시험 로그 {s.tests_log}에 실패 · 오류 요약 줄: {fails[-1]!r}")
        diff = differential(self.ctx, s)
        if not diff["ok"]:
            bad += [f"차등 시험: {x}" for x in diff["reasons"]]
        synth = self._cell("stage0", "synth", s.seal_fields(), lambda: aa_estimate.synth_validation(s))
        dec = dict(outcome=aa_rules.INVALID, reasons=bad) if bad else dict(outcome=aa_rules.PASS, reasons=[])
        numbers = aa_store.to_json(s)
        p = aa_store.write_json(self._detail("stage0"), dict(tests=tests, differential=diff, synth=synth,
                                                             numbers=numbers), self.plist)
        self._finish("stage0", t0)
        dsum = dict(ok=diff["ok"], n_units=diff["n_units"], n_hits=diff["n_hits"], pairs=diff["pairs"],
                    n_reasons=len(diff["reasons"]))
        return self._write("stage0", dict(
            dec, tests=tests, differential=dsum, synth=synth, decision_files=files_sha(aa_files() + list(s.seal_files)),
            yxw_files=files_sha(yxw_files()), seal=seal_now(s), numbers=numbers,
            numbers_sha256=hashlib.sha256(canonical(numbers).encode()).hexdigest(), detail_path=self._detail("stage0"),
            detail_sha256=sha256_file(p),
            note="AA.7 0a: 시험 로그(sha256), 합성 검증(기록), Y 파일럿 V 4쌍 차등 시험(캐시 적중만, 비트 단위), 결정 "
                 "파일 해시, 수치 표, 환경 해시(env — 이후 모든 단계 시작 · 재개에서 대조)."))

    # ================================================================ 1: reuse (AA.7 1, 6252)
    def stage_reuse(self) -> dict:
        doc = self._require("reuse")
        s, t0, ctx = self.s, time.perf_counter(), self.ctx
        facts = dict(y_doc=ctx["y_doc"](), z_doc=ctx["z_doc"](), keys=ctx["keys"](),
                     y_anc=(ctx["git_facts"](Y_SPEC.summary, [c for _, c in s.y_blocks]) or {}).get("ancestors"),
                     z_anc=(ctx["git_facts"](s.z_summary, [c for _, c in s.z_blocks]) or {}).get("ancestors"),
                     decl_sha=ctx["decl_sha"](yxw_files(), s.decl_commit), now_sha=files_sha(yxw_files()),
                     absent={p: not Path(p).exists() for p in s.y_absent_files}, lenient=None, axes={})
        try:
            items = ctx["lenient"]()
            axes = {}
            for _k, _c, a in items:
                axes[a] = axes.get(a, 0) + 1
            facts["axes"] = axes
            del items
        except z_split.LenientMismatch as e:
            facts["lenient"] = str(e)
        why = aa_rules.reuse_reasons(facts, s, Y_SPEC)
        counts = aa_rules.candidate_counts(self._cands(doc))
        body = aa_rules.reuse_stop(why, counts) if why else dict(outcome=aa_rules.PASS, reasons=[])
        body = dict(body, lenient_axes=facts["axes"], yxw_decl=s.decl_commit, n_yxw=len(facts["now_sha"]),
                    note="AA.7 1: Y 블록 일곱 · Z 블록 넷(결과 · 커밋), W · U 측정 키, 주 세트 digest, oracle.json "
                         "sha256 · N_len 31(축별 개수만), y/x/w · z_split 불변, Y gates · learn 블록과 진행 파일 없음.")
        aa_store.write_json(self._detail("reuse"), body, self.plist)
        self._finish("reuse", t0)
        return self._write("reuse", body)

    # ---- spend helpers for orders 2–4b ---------------------------------------------------------------------------
    def _workers(self) -> int:
        return max(1, int(self.workers or getattr(self.pool, "n_workers", 0) or self.s.workers))

    def _core_h(self, doc: dict) -> float:
        """The core ledger so far (AA.7 예산), hours."""
        return sum(float(e.get("wall_s") or 0.0) for e in (doc.get("budget") or {}).get("ledger", [])) / self.s.s_per_h

    def _ticker(self, stage: str):
        """Y's tick form: every call adds the seconds since the previous call to progress/<stage>.json (they survive a
        kill); the stage calls it once more in `finally` and before its block. No in-stage budget stop (AA.7 2 / 4)."""
        mark = [time.perf_counter()]

        def tick(*_a):
            now = time.perf_counter()
            self._add(stage, now - mark[0])
            mark[0] = now
        return tick

    def _costs(self, learn: list, naive: list) -> dict:
        lw = [float(g["result"]["wall_s"]) for g in learn]
        nw = [float(g["result"]["wall_s"]) for g in naive]
        return dict(w_records.unit_costs([g["result"] for g in learn], 2 * W_SPEC.trials, 0.0, self._workers()),
                    learn_job_s=float(np.median(lw)), learn_job_max_s=float(max(lw)), naive_job_s=float(np.median(nw)))

    def _estimates(self, costs: dict) -> dict:
        """plan Reading 10, hours: the screen and learn from the smoke job walls; coverage from one timed cell on a
        31-single-pair structure scaled to cells × sets × methods / workers; the estimate from one timed synthetic
        estimate of a 31-pair set."""
        s, w = self.s, self._workers()
        rounds = lambda n: -(-int(n) // w)                                  # noqa: E731
        t = time.perf_counter()
        aa_estimate.coverage_cell((1,) * s.n_len, "two_stage", "smoke", 0, s, reps=s.cov_timing_reps)
        cell_s = (time.perf_counter() - t) * s.cov_reps / s.cov_timing_reps
        r = aa_estimate.stream(s.synth_seed, "smoke")
        keys = [f"b|{i}|sx{i}|sy{i}" for i in range(s.n_len)]
        G = len(w_verdict.GATES)
        arrs = {k: dict(w=r.standard_normal((s.flies, G)), raw=r.standard_normal((s.flies, G)),
                        rc=r.standard_normal((s.flies, len(aa_estimate.RAW)))) for k in keys}
        t = time.perf_counter()
        aa_estimate.estimate_set(arrs, keys, "S1", s)
        est_s = time.perf_counter() - t
        n_cov = len(aa_estimate.cov_cells(s)) * len(SETS) * len(s.cov_methods)
        return {k: v / s.s_per_h for k, v in dict(
            screen=rounds(s.n_len * s.flies) * costs["naive_job_s"],
            coverage=cell_s * n_cov / w,
            learn=rounds(s.n_len * s.flies * len(BRAINS)) * costs["learn_job_max_s"],
            estimate=est_s).items()}

    # ================================================================ order 2: smoke (AA.7 2, 6253)
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        s = self.s
        tick = self._ticker("smoke")
        try:
            row = self.ctx["pilot_rows"]()[s.smoke_pair]
            wm = self.ctx["measurer"](self.pool, True)
            lu = units([row], "smoke", s.probes, s.smoke_flies, V_SPEC.lever_edit)
            nu = units([row], "smoke", s.probes, s.smoke_flies, V_SPEC.lever_edit, brains=("naive",))
            learn, naive = wm.learn(lu, "smoke"), wm.learn(nu, "smoke_naive")
        finally:
            tick()
        ws = aa_w_spec(s)
        flies = list(range(s.smoke_flies))
        seeds = {f: ws.smoke_probe_seeds(f, s.probes) for f in flies}
        nv = {g["unit"]["fly"]: w_records.counts(g["result"]["stages"][0]) for g in naive}
        problems = list(w_records.machine_reasons(learn, flies, declared(V_SPEC.lever_edit, seeds), naive=nv))
        for g in naive:
            if [int(x) for x in g["result"].get("probe_seeds") or []] != seeds[g["unit"]["fly"]]:
                problems.append(f"fly {g['unit']['fly']}: 순진 잡 프로브 시드가 선언과 다름")
        d = pair_counts(learn, flies)
        if bool(w_verdict.rn1_mismatch({k: np.asarray(v)[None] for k, v in d.items()})[0]):
            problems.append("RN1 ≠ R1")
        lo, hi = s.seed_blocks[1]
        used = [x for g in learn + naive for x in g["unit"]["probe_seeds"]] + \
            [ws.smoke_train_base(0), ws.smoke_train_base(1)]
        problems += [f"스모크 시드 {x}가 스모크 블록 밖" for x in used if not lo <= x < hi]
        zv = {k: tuple(v) for k, v in self.ctx["y_doc"]()["pilot"]["z_V"].items()}
        if {k: tuple(v) for k, v in self.ctx["y_z"]().items()} != zv:
            problems.append("z_V ≠ Y 블록 pilot의 z_V")
        costs = self._costs(learn, naive)
        est = self._estimates(costs)
        tick()
        elapsed = self._core_h(doc) + self._prog("smoke")["core"] / s.s_per_h
        rem = sum(est.values())
        ok = aa_rules.core_ok(elapsed, rem, s)
        cnt = aa_rules.candidate_counts(self._cands(doc, "smoke"))
        if problems:
            body = dict(outcome=aa_rules.INVALID, reasons=problems)      # a pipeline defect before the main set
        elif not ok:
            body = aa_rules.budget_stop(aa_rules.budget_text(elapsed, rem, s), s.stage_label("smoke"), "주 세트 미측정",
                                        cnt)
        else:
            body = dict(outcome=aa_rules.PASS, reasons=[])
        body.update(problems=problems, costs=costs, estimates_h=est, pair=row_key(row),
                    budget=dict(ok=ok, elapsed_h=elapsed, remaining_h=rem, margin=s.cost_margin, cap_h=s.core_cap_h),
                    seeds=dict(probe=seeds, train=[ws.smoke_train_base(0), ws.smoke_train_base(1)]),
                    manifest=manifest(learn + naive),
                    note="AA.7 2: 파일럿 b|17 · 마리 1 · AA 스모크 시드, R · N · RN + 순진 pre; 오라클 없음.")
        return self._write("smoke", body)

    # ================================================================ order 3: seal_code (AA.7 3, 6254)
    def stage_seal_code(self) -> dict:
        self._require("seal_code")
        t0 = time.perf_counter()
        seal = seal_now(self.s)
        return self._write("seal_code", dict(outcome=aa_rules.PASS, reasons=[], seal=seal,
                                             note="AA.7 3: 순진 거름 전에 봉인 — 주 세트 값은 아무것도 측정되지 않았다."),
                           core_s=time.perf_counter() - t0)

    # ================================================================ order 4: the naive screen (AA.7 4, 6257–6260)
    def stage_screen(self) -> dict:
        doc = self._require("screen")
        s = self.s
        tick = self._ticker("screen")
        try:
            items = self.ctx["lenient"]()                               # [(key, c, axis)] only
        except z_split.LenientMismatch as e:
            refuse(f"stage screen: {e}")
        try:
            rows = {int(r["c"]): r for r in self.ctx["main_rows"](self.ctx["y_doc"]()["digest"]["set"])}
        except ValueError as e:
            refuse(f"stage screen: the main set does not reproduce Y block digest: {e}")
        wrong = [k for k, c, _a in items if c not in rows or row_key(rows[c]) != k]
        if wrong:
            refuse(f"stage screen: lenient key(s) {wrong[:3]} are not main-set row c")
        z = self.ctx["y_z"]()
        keys = [k for k, _c, _a in items]
        prs = [rows[c] for _k, c, _a in items]
        U = units(prs, "main", s.probes, s.flies, V_SPEC.lever_edit, brains=("naive",))
        self._cands_set("screen", keys, aa_rules.STATES[1])                # before the first batch (plan Reading 8)
        wm = self.ctx["measurer"](self.pool, False)
        ins = [canonical(wm.inputs(u, "screen")) for u in U]
        first = list(dict.fromkeys(ins))                                    # identical inputs are measured once
        try:
            got1 = wm.learn([U[ins.index(x)] for x in first], "screen", check=tick)
        finally:
            tick()
        by_in = dict(zip(first, got1))
        got = [dict(by_in[x], unit=u) for u, x in zip(U, ins)]
        ws = aa_w_spec(s)
        want = {(row_key(r), f): ws.probe_seeds(int(r["c"]), f, s.probes) for r in prs for f in range(s.flies)}
        cnts = {(g["unit"]["pair"], g["unit"]["fly"]): counts_f(g["result"]["stages"][0]) for g in got}
        why = aa_rules.screen_machine_reasons([g["unit"] for g in got], cnts, want, len(prs) * s.flies, s.probes)
        man = manifest(got)
        digest = raw_digest([g["result"] for g in got])
        p = aa_store.write_json(self._detail("screen"), dict(
            manifest=man, seeds={f"{k}|{f}": v for (k, f), v in want.items()}, raw_digest=digest), self.plist)
        cnt = aa_rules.candidate_counts(self._cands(doc, "screen"))
        common = dict(raw_digest=digest, detail_path=self._detail("screen"), detail_sha256=sha256_file(p),
                      manifest_sha256=hashlib.sha256(canonical(man).encode()).hexdigest(), n_units=len(got),
                      seeds_ok=not any("시드" in x for x in why))
        if why:
            body = aa_rules.machine_stop("순진 거름", "; ".join(why[:5]), f"학습 측정 없음, {s.n_len}쌍 `screened`", cnt)
            tick()
            return self._write("screen", dict(body, **common, n_reasons=len(why)))
        bp = by_pair(got)
        pairs = []
        for k, c, ax in items:
            pre = np.stack([w_records.counts(g["result"]["stages"][0])
                            for g in sorted(bp[k], key=lambda g: g["unit"]["fly"])])
            f = y_rules.final_filter(pre, z, Y_SPEC)                       # the only main-set final_filter call
            pairs.append(dict(key=k, c=int(c), axis=ax, x_odour=y_rules.x_odour(k), naive_d=f["d"], L_A=f["L_A"],
                              L_P=f["L_P"], passed=f["passed"]))
        sel = dict(S1=[x["key"] for x in pairs if x["passed"]], S31=list(keys),
                   out=[x["key"] for x in pairs if not x["passed"]])
        groups = {st: dict(n=len(ks), sizes=sorted((len(g) for g in y_rules.merge_groups(ks)), reverse=True))
                  for st, ks in sel.items()}
        rc = aa_rules.reason_counts([dict(d=x["naive_d"], L_A=x["L_A"], L_P=x["L_P"]) for x in pairs], Y_SPEC)
        k = len(sel["S1"])
        body = aa_rules.no_pairs_stop(rc, cnt) if k == 0 else dict(outcome=aa_rules.PASS, reasons=[])
        tick()
        return self._write("screen", dict(
            body, **common, pairs=pairs, k=k, groups=groups, reasons_count=rc,
            note="AA.7 4: S1 매니페스트만(쌍별 키 · c · 축 · X 냄새 · 순진 d′ · L_A(X) · L_P(X) · 통과), k, 묶음 크기; "
                 "원자료 digest는 다음 단계(coverage) 시작에서 다시 읽어 대조(plan Reading 7)."))

    def _screen_raw(self, doc: dict) -> tuple:
        """The screen raw reloaded by its manifest (sha-checked): ({key: pre int [F, K, 2, 2]}, digest, reasons)."""
        blk = doc.get("screen") or {}
        p = Path(self._detail("screen"))
        if not p.exists():
            return {}, None, [f"{p} 없음"]
        bad = [] if sha256_file(p) == blk.get("detail_sha256") else [f"{p} sha256 ≠ screen 블록"]
        det = json.loads(p.read_text())
        got, b2 = load_manifest(det.get("manifest") or [])
        bad += b2
        res = [g["result"] for g in got]
        by = {}
        for g in got:
            pair, fly, _b = g["key"].rsplit("|", 2)
            by.setdefault(pair, []).append((int(fly), g["result"]))
        pre = {}
        if not bad:
            for k, v in by.items():
                pre[k] = np.stack([w_records.counts(r["stages"][0]) for _f, r in sorted(v, key=lambda x: x[0])])
        return pre, raw_digest(res), bad

    # ================================================================ order 4b: coverage (AA.7 4b 6261, AA.5 6227–6231)
    def stage_coverage(self) -> dict:
        doc = self._require("coverage")
        s, t0 = self.s, time.perf_counter()
        _pre, dig, bad = self._screen_raw(doc)
        del _pre
        cnt = aa_rules.candidate_counts(self._cands(doc, "coverage"))
        if bad or dig != doc["screen"]["raw_digest"]:
            self._finish("coverage", t0)
            why = f"원자료 digest 재확인 불일치{': ' + '; '.join(bad[:2]) if bad else ''}"
            return self._write("coverage", dict(aa_rules.machine_stop(
                "순진 거름", why, f"학습 측정 없음, {s.n_len}쌍 `screened`", cnt), raw_digest_now=dig))
        seal = seal_now(s)["key"]
        jobs, plan, nul = [], {}, {}
        for st in SETS:
            g = doc["screen"]["groups"][st]
            p = aa_estimate.ci_plan(int(g["n"]), len(g["sizes"]), s)
            if p["kind"] != "pooled":
                nul[st] = "k < 2 — 통합 CI 없음"
                continue
            plan[st] = [p["primary"]] + list(p["records"]) + [f"raw_{p['primary']}"]
            for m in plan[st]:
                for ci in range(len(aa_estimate.cov_cells(s))):
                    jobs.append((f"{st}_{m}_{ci}", dict(sizes=g["sizes"], m=m, st=st, ci=ci, seal=seal),
                                 _cov_job, (tuple(int(x) for x in g["sizes"]), m, st, ci, s)))
        vals = self._pcells("coverage", jobs, log=_log)
        it = iter(v["meta"]["share"] for v in vals)
        by_set = {}
        for st in SETS:
            if st not in plan:
                by_set[st] = None
                continue
            sizes = [int(x) for x in doc["screen"]["groups"][st]["sizes"]]
            by_set[st] = {m: dict(aa_estimate.coverage_summary([next(it) for _ in aa_estimate.cov_cells(s)], s),
                                  sizes=sizes, method=m) for m in plan[st]}
        self._finish("coverage", t0)
        return self._write("coverage", dict(
            outcome=aa_rules.PASS, reasons=[], by_set=by_set, null_reason=nul, seal_key=seal, raw_digest_now=dig,
            note="AA.5 · AA.7 4b: 묶음 크기 구조만 입력(결과값 없음); 집합별 1차 · 기록 방식 + 1차의 비표준화 판. ±∞ 제외판은 "
                 "같은 방식의 포함 확률을 쓴다(plan Reading 6); 바닥 민감도는 records."))

    # ================================================================ order 5: learn (AA.7 5 6262, AA.3 6191–6192)
    def _learn_units(self, doc: dict) -> tuple:
        """All 31 screened pairs (pass or fail) in the screen block's c order, R · N · RN on Y's judged seeds."""
        s = self.s
        try:
            rows = {int(r["c"]): r for r in self.ctx["main_rows"](self.ctx["y_doc"]()["digest"]["set"])}
        except ValueError as e:
            refuse(f"stage learn: the main set does not reproduce Y block digest: {e}")
        prs = [rows[int(p["c"])] for p in doc["screen"]["pairs"]]
        return prs, units(prs, "main", s.probes, s.flies, V_SPEC.lever_edit)

    def _pairs_complete(self, U: list, wm) -> int:
        """Pairs every unit of which has a cache entry for block "learn" (AA.8 "학습 〈n〉쌍 완료")."""
        by = {}
        for u in U:
            by.setdefault(u["pair"], []).append(wm.cache.get(KIND, wm.inputs(u, "learn")) is not None)
        return sum(all(v) for v in by.values())

    def _learn_reasons(self, doc: dict, prs: list, got: list) -> list:
        """AA.7 5: unit count / duplicate keys, RN1 = R1, learn pre = naive screen pre (bit for bit), every
        w_records.machine_reasons item, w_verdict.data_reasons, no NaN fly (the gate value itself is discarded)."""
        s, z = self.s, self.ctx["y_z"]()
        why = []
        want = len(prs) * s.flies * len(BRAINS)
        if len(got) != want:
            why.append(f"단위 수 {len(got)} ≠ {want}")
        keys = [(g["unit"]["pair"], g["unit"]["fly"], g["unit"]["brain"]) for g in got]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            why.append(f"단위 키 중복 {len(dup)}개: {dup[:3]}")
        pre, _dig, bad = self._screen_raw(doc)
        why += [f"순진 거름 원자료 재확인 실패: {b}" for b in bad[:3]]
        ws = aa_w_spec(s)
        flies = list(range(s.flies))
        bp = by_pair(got)
        for r in prs:
            k, c = row_key(r), int(r["c"])
            rows_k = bp.get(k) or []
            seeds = {f: ws.probe_seeds(c, f, s.probes) for f in flies}
            naive = {f: pre[k][f] for f in flies} if k in pre else None
            if naive is None and not bad:
                why.append(f"{k}: 순진 거름 pre 없음")
            try:
                why += [f"{k}: {x}" for x in w_records.machine_reasons(rows_k, flies, declared(V_SPEC.lever_edit, seeds),
                                                                         naive=naive)]
                d = pair_counts(rows_k, flies)
            except (KeyError, ValueError, TypeError) as e:
                why.append(f"{k}: 자료 결함 {type(e).__name__}: {e}")
                continue
            why += [f"{k}: {x}" for x in w_verdict.data_reasons(d, s.flies, s.probes)]
            if bool(w_verdict.rn1_mismatch({st: np.asarray(v)[None] for st, v in d.items()})[0]):
                why.append(f"{k}: RN1 ≠ R1")
            with np.errstate(all="ignore"):
                if np.isnan(np.asarray(w_verdict.gate_stats(d, z), float)).any():
                    why.append(f"{k}: NaN 마리")
        return why

    def stage_learn(self) -> dict:
        doc = self._require("learn")
        s = self.s
        if doc["screen"].get("outcome") != aa_rules.PASS or int(doc["screen"].get("k") or 0) == 0:
            refuse("stage learn: needs a PASS screen with k ≥ 1 (AA.3, STOP_NO_PAIRS)")
        tick = self._ticker("learn")
        prs, U = self._learn_units(doc)
        wm = self.ctx["measurer"](self.pool, False)
        ins = [canonical(wm.inputs(u, "learn")) for u in U]
        first = list(dict.fromkeys(ins))                                    # identical inputs are measured once
        UU = [U[ins.index(x)] for x in first]
        todo = [u for u in UU if wm.cache.get(KIND, wm.inputs(u, "learn")) is None]
        base = self._core_h(doc)
        est = doc["smoke"]["estimates_h"]
        cnt = lambda: aa_rules.candidate_counts(self._cands(self._doc(), "learn"))      # noqa: E731
        left = float(est["learn"]) * len(todo) / max(len(UU), 1) + float(est["estimate"])
        spent0 = base + self._prog("learn")["core"] / s.s_per_h
        if not aa_rules.core_ok(spent0, left, s):                         # the reservation (AA.7 5 "시작 직전")
            tick()
            return self._write("learn", aa_rules.budget_stop(
                aa_rules.budget_text(spent0, left, s), "5 전", "순진 거름만 측정(31쌍 `screened`, 학습 0)", cnt()))
        n_w = max(1, int(getattr(self.pool, "n_workers", 0) or 1))

        def check(a, _n):
            """Before each measurer round (plan Readings 9, 21): seconds to progress, the measured cap, then the
            round's pairs turn `trained` (states move forward only)."""
            tick()
            if not aa_rules.spent_ok(base + self._prog("learn")["core"] / s.s_per_h, s):
                raise _BudgetStop()
            self._cands_set("learn", sorted({u["pair"] for u in todo[a:a + n_w]}), aa_rules.STATES[2])
        try:
            got1 = wm.learn(UU, "learn", check=check)
        except _BudgetStop:
            tick()
            n = self._pairs_complete(U, wm)
            return self._write("learn", aa_rules.budget_stop(
                aa_rules.budget_text(base + self._prog("learn")["core"] / s.s_per_h, 0.0, s), s.stage_label("learn"),
                f"학습 {n}쌍 완료", cnt()))
        finally:
            tick()
        self._cands_set("learn", [row_key(r) for r in prs], aa_rules.STATES[2])   # every pair is measured by now
        by_in = dict(zip(first, got1))
        got = [dict(by_in[x], unit=u) for u, x in zip(U, ins)]
        why = self._learn_reasons(doc, prs, got)
        man = manifest(got)
        p = aa_store.write_json(self._detail("learn"), dict(manifest=man), self.plist)
        n_tr = cnt()["trained"]
        common = dict(manifest_sha256=hashlib.sha256(canonical(man).encode()).hexdigest(), n_units=len(got),
                      n_trained=n_tr, detail_path=self._detail("learn"), detail_sha256=sha256_file(p))
        if why:
            body = aa_rules.machine_stop("학습", "; ".join(why[:5]), f"학습 {n_tr}쌍 `trained`", cnt())
            tick()
            return self._write("learn", dict(body, **common, n_reasons=len(why)))
        tick()
        return self._write("learn", dict(
            outcome=aa_rules.PASS, reasons=[], **common,
            note="AA.7 5: 매니페스트만(통계 없음) — 31쌍 × 8마리 × R · N · RN, c 오름차순 배치; 학습 pre = 순진 거름 pre "
                 "비트 동일 · RN1 = R1 · 자료 결함 · NaN 마리 검사 통과. 결과는 estimate 블록에서 처음 읽는다."))

    def _learn_data(self, doc: dict) -> dict:
        """{key: {stage: [F, K, 2, 2]}} from the learn block's manifest (sha-checked; cache hits only — a missing or
        changed raw file refuses, nothing is measured)."""
        s = self.s
        blk = doc.get("learn") or {}
        p = Path(self._detail("learn"))
        if not p.exists() or sha256_file(p) != blk.get("detail_sha256"):
            refuse(f"learn raw: {p} is missing or its sha256 differs from block learn (캐시 적중 실패)")
        got, bad = load_manifest(json.loads(p.read_text()).get("manifest") or [])
        if bad:
            refuse(f"learn raw: 캐시 적중 실패 {len(bad)}건: {bad[:3]}")
        by = {}
        for g in got:
            pair, fly, brain = g["key"].rsplit("|", 2)
            by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
        flies = list(range(s.flies))
        return {k: pair_counts(v, flies) for k, v in by.items()}

    # ================================================================ order 6: estimate (AA.7 6 6263, AA.5, AA.8 6278)
    def _primary(self, doc: dict) -> dict:
        """S1 = the screen block's passers (never learning values); the small-k rule picks the primary CI. Also
        carries the DL τ̂ and the S1 φ medians the result sentence needs."""
        s = self.s
        z = self.ctx["y_z"]()
        data = self._learn_data(doc)
        S1 = [p for p in doc["screen"]["pairs"] if p["passed"]]
        keys = [p["key"] for p in S1]
        arrs = {k: aa_estimate.pair_arrays(data[k], z, s) for k in keys}
        est = aa_estimate.estimate_set(arrs, keys, "S1", s)
        pairs = []
        for p in S1:
            a = arrs[p["key"]]
            ci = aa_estimate.pair_ci(a, int(p["c"]), s)["ci"]
            pairs.append(dict(key=p["key"], c=int(p["c"]), naive_d=p["naive_d"], L_A=p["L_A"], L_P=p["L_P"],
                              **{g: dict(mean=float(a["w"][:, aa_estimate.GI[g]].mean()), ci=ci[g])
                                 for g in aa_estimate.PRIMARY}))
        fl = [aa_estimate.floors(data[k]) for k in keys]
        out = dict(set="S1", kind=est["plan"]["kind"], k=est["k"], n_groups=est["n_groups"], sizes=est["sizes"],
                   plan=est["plan"], flag=est["plan"]["flag"], pairs=pairs,
                   phi_median=dict(phi_R=float(np.median([f["phi_R"] for f in fl])),
                                   phi_P=float(np.median([f["phi_P"] for f in fl]))))
        if est["plan"]["kind"] == "pooled":
            m = est["plan"]["primary"]
            cov = ((doc["coverage"].get("by_set") or {}).get("S1") or {}).get(m) or {}
            out.update(pooled={g: dict(point=est["point"][g], ci=est["ci"][m][g], method=m)
                               for g in aa_estimate.PRIMARY},
                       range={g: est["range"][g] for g in aa_estimate.PRIMARY}, coverage=cov.get("min"),
                       under=cov.get("under"),
                       tau_dl={g: (est["dl"][g].get("tau") if est["dl"][g].get("available") else None)
                               for g in aa_estimate.PRIMARY})
        return aa_store.to_json(out)

    def _sentence_and_compare(self, a: dict) -> tuple:
        ra, pa = aa_estimate.PRIMARY
        if a["kind"] == "pair_study":
            p = a["pairs"][0]
            pt = {g: (p[g]["mean"], p[g]["ci"]) for g in (ra, pa)}
            r = dict(k=1, key=p["key"])
        else:
            pt = {g: (a["pooled"][g]["point"], a["pooled"][g]["ci"]) for g in (ra, pa)}
            rng = a["range"]
            r = dict(k=a["k"], g=a["n_groups"], few=a["flag"] == "적은 묶음", method=a["plan"]["primary"],
                     cov=a.get("coverage"), under=bool(a.get("under")), tau_ra=a["tau_dl"][ra], tau_pa=a["tau_dl"][pa],
                     # one range slot in AA.8's sentence: the envelope of the two associations' pair ranges
                     rng_min=min(rng[ra][0], rng[pa][0]), rng_max=max(rng[ra][1], rng[pa][1]))
        r.update(mu_ra=pt[ra][0], lo_ra=pt[ra][1][0], hi_ra=pt[ra][1][1], mu_pa=pt[pa][0], lo_pa=pt[pa][1][0],
                 hi_pa=pt[pa][1][1], phi_r=a["phi_median"]["phi_R"], phi_p=a["phi_median"]["phi_P"])
        cmp_ = {g: aa_estimate.compare_table(pt[g][0], pt[g][1][0], pt[g][1][1], self.s) for g in (ra, pa)}
        return aa_rules.result_sentence(r), cmp_

    def stage_estimate(self) -> dict:
        doc = self._require("estimate")                  # the seal check comes first (sealed stage)
        t0 = time.perf_counter()
        a = self._primary(doc)
        b = self._primary(doc)                           # AA.7 6: the same raw and code once more, bit for bit
        seal = seal_now(self.s)["key"]
        if canonical(a) != canonical(b):
            self._finish("estimate", t0)
            return self._write("estimate", dict(
                outcome=aa_rules.INVALID, reasons=["추정 재계산이 비트 단위로 같지 않음(AA.7 6)"], repeat_equal=False,
                seal_key=seal))
        sentence, cmp_ = self._sentence_and_compare(a)
        self._finish("estimate", t0)
        return self._write("estimate", dict(
            outcome=aa_rules.PASS, reasons=[], primary=a, sentence=sentence, compare=cmp_, repeat_equal=True,
            seal_key=seal,
            note="AA.7 6 · AA.5 · AA.8 · AA.9: 1차 = S1(순진 거름 통과, screen 블록 그대로) 두 연합 d′의 쌍별 · 통합 추정과 "
                 "작은-k 규칙의 1차 CI, 구조 맞춤 포함 확률(coverage 블록); 두 번 계산해 비트 같음. 비교 표는 기술 "
                 "전용(판정 아님). 학습 통계는 이 블록에서 처음 읽는다."))


class _BudgetStop(Exception):
    """Raised from the learn round check when the measured core ledger passes the cap (AA.7 5)."""
