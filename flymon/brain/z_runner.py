"""Spec Z's stage chain, order 0 (Z.7 0a–0d as amended by Z.9.2): stage0 (0a) → ydiag (0b) → reuse (0c) → sens (0c′)
→ split (0d), one block each in results/summary/z_learning.json, written only through z_store. Z.9.2 P1-5: the stage
assembly is Z's own; Y's Runner is used only for its read methods (_reuse_all, _theta_b, _z_b, _doc) through ctx,
and Y's pure functions are imported; nothing writes through y_store and nothing assigns to a Y / X / W module.
- Chain: ORDER = zs.stages (every Z stage, declared now). A stage refuses (SystemExit 2, nothing written) when an
  earlier block is missing, a later block or its own exists, an earlier block is not PASS, the summary has
  uncommitted changes, or a hashed file (z_* , y_* / x_* / w_*, Y's summary) is dirty. Stages after order 0 live in
  zs.later_modules (plan Reading 1): `run(stage)` calls `stage_<name>` here, else that module's STAGES[name](runner),
  else refuses.
- Environment (Z.9.2 P2-11, plan Reading 1): env_now() = uv.lock, Python, numpy, scipy, platform, the sha256 of every
  z_* file and scripts/run_z.py present, and of every flymon/brain/{y,x,w}_*.py. Every stage start and every resume
  compares env_now() with the env of every committed block and with the env the stage recorded at its first start
  (progress/<stage>.env.json); a difference appends {stage, env_mismatch} to the core ledger and exits EXIT_ENV
  (no block; 사용자 몫, not a STOP label). A file added after a block (a later plan's new module) is not a difference.
- Ledgers (Z.9.2 P2-10): each block adds its core seconds to budget.ledger and its records seconds (0b's record-only
  reproduction) to budget.records_ledger. The records part stops filling (null + reason) when records seconds would
  pass zs.records_budget_h; it never stops the stage.
- Cells: resumable JSON cells (progress/<stage>/<tag>.json) and array cells (.npz, computed in a spawn process pool,
  BLAS single-threaded), key = tag + inputs + ZSpec + z_* sha256; each cell's seconds go to progress/<stage>.json
  under its ledger, so a killed run keeps its spend.
- Archive (Y.9.2 P3-13 as Z.7 requires): each PASS / STOP block's own files are copied to ~/flymon-archive/z/<stage>/
  before the block is written; ydiag also copies Y's 200 cells to ~/flymon-archive/z/y_cells/. Never for INVALID."""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import importlib
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import z_oc, z_rules, z_split, z_store
from .h3_store import ROOT, canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .y_spec import SPEC as Y_SPEC

EXIT_ENV = 6
YXW_GLOBS = ("y_*.py", "x_*.py", "w_*.py")


def z_files() -> list:
    out = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "flymon/brain").glob("z_*.py"))
    return out + (["scripts/run_z.py"] if (ROOT / "scripts/run_z.py").exists() else [])


def yxw_files() -> list:
    return sorted(p.relative_to(ROOT).as_posix() for g in YXW_GLOBS for p in (ROOT / "flymon/brain").glob(g))


def files_sha(files) -> dict:
    return {f: sha256_file(ROOT / f) for f in files}


def env_now() -> dict:
    import scipy
    return dict(uv_lock_sha256=sha256_file(ROOT / "uv.lock"), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, platform=platform.platform(), z_files=files_sha(z_files()),
                yxw_files=files_sha(yxw_files()))


def hashed_files() -> list:
    return z_files() + yxw_files() + [Y_SPEC.summary]


def git_state() -> dict:
    return _h3_git_state(files=hashed_files())


def refuse(msg: str, code: int = 2):
    z_store.refuse(msg, code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def build_ctx(npz: str) -> dict:
    """Read-only access to Y (its ctx and Runner read methods), W's θ̂ through Y's reuse, the Y cells and details."""
    from . import w_runner, y_runner
    yctx = y_runner.build_ctx(npz)

    def yrun():
        return y_runner.Runner(yctx, Y_SPEC)

    def y_theta(where: str) -> dict:
        r = yrun()
        doc = r._doc()
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                stop, th, tw, rv, _c, keys = r._theta_b(doc, where)
        except SystemExit:
            return dict(why=[f"Y θ̂ 재적합 거절: {err.getvalue().strip()}"])
        if stop:
            return dict(why=list(stop.get("reasons") or [stop.get("outcome")]))
        return dict(why=[], theta=th, theta_w=tw, r=rv, keys=keys, z=r._z_b(doc))

    def y_reuse(where: str) -> list:
        stop, *_ = yrun()._reuse_all(where)
        return list(stop.get("reasons") or [stop.get("outcome")]) if stop else []

    def decl_sha(files: list, commit: str) -> dict:
        import subprocess
        out = {}
        for f in files:
            r = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{f}"], capture_output=True)
            out[f] = hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
        return out

    def oc_detail():
        p = Path(Y_SPEC.oc_detail)
        return (json.loads(p.read_text()), sha256_file(p)) if p.exists() else ({}, None)

    return dict(keys=yctx["keys"], params=yctx["params"], y_doc=lambda: yrun()._doc(), y_theta=y_theta,
                y_reuse=y_reuse, git_facts=w_runner.git_facts, decl_sha=decl_sha, oc_detail=oc_detail,
                y_pilot_back=yctx["pilot_back"])


class Runner:
    def __init__(self, ctx: dict, zs, summary_path=None):
        self.ctx, self.zs = ctx, zs
        self.summary_path = str(summary_path or zs.summary)
        self._s = dict(core=0.0, records=0.0)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return z_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        order = list(self.zs.stages)
        i = order.index(stage)
        missing = [b for b in order[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in order[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; Z never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; Z never rewrites a recorded block")
        stopped = [b for b in order[:i] if doc[b].get("outcome") != z_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — Z stops there")
        self._env(stage, doc)
        return doc

    def _env(self, stage: str, doc: dict) -> dict:
        """Z.9.2 P2-11 at the start and at every resume (module docstring)."""
        now = env_now()
        blocks = [(b, doc[b]["env"]) for b in self.zs.stages if b in doc and "env" in doc[b]]
        p = Path(self.zs.progress_dir) / f"{stage}.env.json"
        if p.exists():
            blocks.append((f"{stage} 첫 시작", json.loads(p.read_text())))
        why = z_rules.env_reasons(blocks, now)
        if why:
            z_store.append_ledger(self.summary_path, dict(stage=stage, env_mismatch=why, at=_now(), wall_s=0.0),
                                  self.plist)
            refuse(f"stage {stage}: environment differs (Z.9.2 P2-11): {why[:5]}", EXIT_ENV)
        if not p.exists():                  # recorded only once it agrees with every committed block
            z_store.write_json(str(p), now, self.plist)
        return now

    def _files(self, stage: str) -> list:
        zs = self.zs
        det = dict(stage0=zs.stage0_detail, ydiag=zs.ydiag_detail, sens=zs.sens_detail, split=zs.split_detail)
        return [det[stage]] if stage in det else []

    def _write(self, stage: str, body: dict) -> dict:
        extra = {}
        if body.get("outcome") != z_rules.INVALID:
            extra["archive"] = z_store.archive(self._files(stage), stage, self.zs)
        k = self.ctx["keys"]()
        prog = self._prog(stage)
        block = z_store.to_json(dict(body, **extra, stage=stage, order=dict(self.zs.stage_labels).get(stage),
                                     env=env_now(), git=git_state(), written_at=_now(),
                                     w_measure_key=k.get("w_measure_key"), u_measure_key=k.get("u_measure_key")))
        rec = dict(stage=stage, wall_s=prog["records"], at=block["written_at"]) if prog["records"] else None
        z_store.write_summary_block(self.summary_path, stage, block, self.plist,
                                    dict(stage=stage, wall_s=prog["core"], at=block["written_at"]), rec)
        return block

    def archive(self, stage: str) -> list:
        blk = self._doc().get(stage)
        if blk is None:
            refuse(f"archive {stage}: no block {stage}")
        if blk.get("outcome") == z_rules.INVALID:
            refuse(f"archive {stage}: block {stage} is INVALID; INVALID is never archived")
        return z_store.archive(self._files(stage), stage, self.zs)

    def run(self, stage: str) -> dict:
        fn = getattr(self, f"stage_{stage}", None)
        if fn is not None:
            return fn()
        for mod in self.zs.later_modules:
            try:
                m = importlib.import_module(mod)
            except ModuleNotFoundError:
                continue
            if stage in getattr(m, "STAGES", {}):
                return m.STAGES[stage](self)
        refuse(f"stage {stage} is not implemented yet (a later Z plan)")

    # ---- spend and cells -------------------------------------------------------------------------------------------
    def _prog_path(self, stage: str) -> Path:
        return Path(self.zs.progress_dir) / f"{stage}.json"

    def _prog(self, stage: str) -> dict:
        p = self._prog_path(stage)
        d = json.loads(p.read_text()) if p.exists() else {}
        return dict(core=float(d.get("core", 0.0)), records=float(d.get("records", 0.0)))

    def _add(self, stage: str, s: float, which: str = "core") -> None:
        d = self._prog(stage)
        d[which] += float(s)
        z_store.write_json(str(self._prog_path(stage)), dict(d, at=_now()), self.plist)

    def _key(self, tag: str, key_obj) -> str:
        spec_sha = hashlib.sha256(canonical(self.zs).encode()).hexdigest()
        return hashlib.sha256(canonical(dict(tag=tag, key=key_obj, zs=spec_sha, z=files_sha(z_files()))).encode()
                              ).hexdigest()

    def _cell(self, stage: str, tag: str, key_obj, fn, which: str = "core"):
        key = self._key(tag, key_obj)
        p = Path(self.zs.progress_dir) / stage / (re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".json")
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except ValueError:
                d = {}
            if d.get("key") == key:
                return d["value"]
        t = time.perf_counter()
        v = z_store.to_json(fn())
        z_store.write_json(str(p), dict(key=key, value=v, at=_now()), self.plist)
        dt = time.perf_counter() - t
        self._add(stage, dt, which)
        self._s[which] += dt
        return v

    def _pcells(self, stage: str, jobs: list, which: str = "core", log=None) -> list:
        """jobs = [(tag, key_obj, fn, args)] with fn a module-level z_oc function returning {arrays, meta}; cells
        progress/<stage>/<tag>.npz; computed in a spawn pool of zs.workers (or self.workers); results in job order."""
        root = Path(self.zs.progress_dir) / stage

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
            np.savez(buf, __key__=np.array(key), __meta__=np.array(json.dumps(z_store.to_json(res["meta"]))),
                     **res["arrays"])
            z_store.write_bytes(str(path), buf.getvalue(), self.plist)
        out, todo = [None] * len(jobs), []
        for i, (tag, key_obj, fn, args) in enumerate(jobs):
            p = root / (re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".npz")
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
            out[i] = dict(arrays=res["arrays"], meta=json.loads(json.dumps(z_store.to_json(res["meta"]))))
            now = time.perf_counter()
            self._add(stage, now - last, which)
            self._s[which] += now - last
            last = now
            if log is not None:
                log(f"z {stage}: {sum(o is not None for o in out)}/{len(out)} cells")
        workers = max(1, int(getattr(self, "workers", None) or self.zs.workers))
        if workers == 1 or len(todo) <= 1:
            for i, p, k, fn, args in todo:
                done(i, p, k, fn(*args))
            return out
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        for v in self.zs.thread_env:
            os.environ.setdefault(v, "1")
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=get_context("spawn")) as ex:
            try:
                fut = {ex.submit(fn, *args): (i, p, k) for i, p, k, fn, args in todo}
                for f in as_completed(fut):
                    i, p, k = fut[f]
                    done(i, p, k, f.result())
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
        return out

    def _finish(self, stage: str, t0: float) -> None:
        """The stage's own (non-cell) seconds go to the core ledger."""
        self._add(stage, (time.perf_counter() - t0) - self._s["core"] - self._s["records"], "core")

    def _records_left_s(self, stage: str, doc: dict) -> float:
        spent = sum(e.get("wall_s", 0.0) for e in (doc.get("budget") or {}).get("records_ledger", []))
        return self.zs.records_budget_h * self.zs.s_per_h - spent - self._prog(stage)["records"]

    # ================================================================ 0a: stage0 (Z.6.4, Z.9.2 P1-5 / P2-11)
    def stage_stage0(self) -> dict:
        self._require("stage0")
        zs, t0 = self.zs, time.perf_counter()
        bad = []
        log = Path(zs.tests_log)
        lines = log.read_text().strip().splitlines() if log.exists() else []
        tests = dict(path=zs.tests_log, sha256=sha256_file(log) if log.exists() else None,
                     last=lines[-1] if lines else None, summary=lines[-2] if len(lines) > 1 else None)
        if tests["last"] != zs.tests_ok_line:
            bad.append(f"시험 로그 {zs.tests_log}의 마지막 줄 {tests['last']!r} ≠ {zs.tests_ok_line!r}")
        fx = self._cell("stage0", "split_fixtures", {}, lambda: z_split.fixtures(zs))
        if not fx.get("ok"):
            bad.append(f"분할 픽스처 실패: {[k for k, v in fx.items() if v is False]}")
        z = {k: tuple(v) for k, v in self.ctx["y_doc"]()["pilot"]["z_V"].items()}

        def timing():
            from . import w_oc
            th = w_oc.fit(w_oc.synthetic_pilot(z_oc.y_oc.rng(zs.oc_seed, z_oc.y_oc.TAG_SYNTH), n_pair=zs.timing_pairs))
            return z_oc.unit_timing(th, z, zs)
        tm = self._cell("stage0", "unit_timing", dict(z=z), timing)
        dec = dict(outcome=z_rules.INVALID, reasons=bad) if bad else dict(outcome=z_rules.PASS, reasons=[])
        tests_files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests/brain").glob("test_z_*.py"))
        p = z_store.write_json(zs.stage0_detail, dict(tests=tests, split_fixtures=fx, unit_timing=tm,
                                                      numbers=z_store.to_json(zs)), self.plist)
        self._finish("stage0", t0)
        return self._write("stage0", dict(
            dec, tests=tests, split_fixtures=fx, unit_timing=tm, decision_files=files_sha(z_files()),
            yxw_files=files_sha(yxw_files()), test_files=files_sha(tests_files),
            numbers_sha256=hashlib.sha256(canonical(zs).encode()).hexdigest(), detail_path=zs.stage0_detail,
            detail_sha256=sha256_file(p),
            note="Z.7 0a: Z 코드와 검증(Z.6.4 + Z.9.2 P1-5 시험은 pytest — 로그 sha256), 결정 파일 해시, 수치 표"
                 "(상세), 환경 해시(env, Z.9.2 P2-11 — 이후 모든 단계 시작 · 재개에서 대조), R-pre 단위 비용(기록)."))

    # ================================================================ 0b: ydiag (Z.7 0b)
    def stage_ydiag(self) -> dict:
        doc = self._require("ydiag")
        zs, t0 = self.zs, time.perf_counter()
        ydoc = self.ctx["y_doc"]()
        cells = [f"{zs.y_cells_dir}/boot_{bi}.npz" for bi in range(zs.y_cells_n)]
        missing = [c for c in cells if not Path(c).exists()]
        why, gates = [], {}
        if missing:
            why.append(f"Y 순서 6 칸 {len(missing)}개 없음")
            body = z_rules.reuse_stop(why, "0b", zs)
            z_store.write_json(zs.ydiag_detail, dict(body, note="칸 없음 — 계산 없음"), self.plist)
            self._finish("ydiag", t0)
            return self._write("ydiag", body)
        draws, shas = z_oc.load_cells(zs.y_cells_dir, zs.y_cells_n)
        arch = z_store.archive(cells, zs.y_cells_archive, zs)
        det, det_sha = self.ctx["oc_detail"]()
        if det_sha != (ydoc.get("oc") or {}).get("detail_sha256"):
            why.append(f"{zs.y_oc_detail} sha256 {det_sha} ≠ Y oc 블록")
        gates["i"] = z_oc.gate_limits(draws, det, zs, z_store.to_json)
        if not gates["i"]["ok"]:
            why.append(f"(i) 칸 한계 ≠ Y oc 블록({ {k: v for k, v in gates['i'].items() if k != 'ok'} })")
        th = self.ctx["y_theta"]("1(Z 순서 0b에서 확인)")
        gates["iii"] = dict(ok=not th["why"], reasons=th["why"])
        if th["why"]:
            why += [f"(iii) {x}" for x in th["why"]]
        else:
            jobs = [(f"redraw_{bi}", dict(bi=bi, cell=shas[cells[bi]]), z_oc.redraw, (th["theta"], th["z"], th["r"], bi))
                    for bi in range(zs.redraw_n)]
            res = self._pcells("ydiag", jobs, "core", _log)
            eq = [z_oc.redraw_equal(x, draws[bi], z_store.to_json) for bi, x in enumerate(res)]
            gates["ii"] = dict(ok=all(e["ok"] for e in eq), n=len(eq), unequal=[i for i, e in enumerate(eq) if not e["ok"]])
            if not gates["ii"]["ok"]:
                why.append(f"(ii) 재모의 ≠ 칸: 추출 {gates['ii']['unequal'][:5]}")
        if why:
            body = z_rules.reuse_stop(why, "0b", zs)
            z_store.write_json(zs.ydiag_detail, dict(body, gates=gates, cells=shas), self.plist)
            self._finish("ydiag", t0)
            return self._write("ydiag", dict(body, gates=gates, cells_sha256=shas, y_cells_archive=arch))
        rec = self._ydiag_records(doc, draws, th, ydoc)
        cmp_ = z_rules.diag_compare(rec["computed"], zs)
        p = z_store.write_json(zs.ydiag_detail, dict(gates=gates, cells=shas, records=rec, compare=cmp_), self.plist)
        self._finish("ydiag", t0)
        return self._write("ydiag", dict(
            outcome=z_rules.PASS, reasons=[], gates=gates, cells_sha256=shas, y_cells_archive=arch,
            compare=dict(n=cmp_["n"], n_equal=cmp_["n_equal"], differ=cmp_["differ"]),
            records_null=rec["null"], detail_path=zs.ydiag_detail, detail_sha256=sha256_file(p),
            note="Z.7 0b: 코드 경로 재현(관문 (i)–(iii), 핵심 원장) + 기록 전용 재현(records 원장, 인쇄 자릿수 대조 — "
                 "다르면 Z.10에서 Z.0 정정, Z의 규칙은 바뀌지 않는다)."))

    def _ydiag_records(self, doc: dict, draws: list, th: dict, ydoc: dict) -> dict:
        """Z.7 0b 기록 전용, item by item under the records ledger's cap (null + reason past it)."""
        zs = self.zs
        theta, r, z, tw = th["theta"], th["r"], th["z"], th["theta_w"]
        pw = z_oc.draw_power(draws, zs)
        w = z_oc.per_draw_worst(pw)
        order = np.argsort(w)
        cor = [d["meta"]["cal"]["min"]["corners"] for d in draws]
        computed, null, detail = {}, {}, {}
        tkey = dict(theta=z_oc.w_oc.summary(theta), z=z)

        def item(name, fn, pooled=False):
            if self._records_left_s("ydiag", doc) <= 0:
                null[name] = "records 상한, Z.9.2 P2-10"
                return None
            v = fn() if pooled else self._cell("ydiag", f"rec_{name}", tkey, fn, "records")
            out = {k: x for k, x in v.items() if not k.startswith("_")}
            computed.update({k: x for k, x in out.items() if not isinstance(x, (dict, list))})
            detail[name] = v
            return v
        item("distribution", lambda: z_oc.rec_distribution(pw, zs))
        item("base_only", lambda: z_oc.rec_base_only(draws, zs))
        item("rho", lambda: z_oc.rec_rho(z_oc.components(theta, r, draws), w, zs))
        item("boot_drift", lambda: dict(y_drift_boot_sd=float(np.std(
            [c["drift"][0] for c in z_oc.components(theta, r, draws)], ddof=1))))
        item("near", lambda: z_oc.rec_near(z_oc.near_truedprimes(theta, r, z, draws), w, zs))
        item("reach", lambda: z_oc.rec_reach(z_oc.reach(theta, r, z, zs), w, zs))
        item("point", lambda: z_oc.point_near(theta, z, ydoc["precheck"]["calibration"]["min"], zs))
        item("thresholds", lambda: z_oc.draw_thresholds(theta, r, draws, w, zs))
        item("drift", lambda: z_oc.drift_facts(theta, tw))

        def selfbase():
            sel = [int(order[k]) for k in zs.selfbase_ranks]
            res = self._pcells("ydiag", [(f"self_{bi}", dict(tkey, bi=bi), z_oc.selfbase_job,
                                          (theta, r, z, bi, zs.cf_roots)) for bi in sel], "records", _log)
            return z_oc.rec_selfbase([x["meta"] for x in res])

        def counterfactual():
            cf = []
            for k in zs.cf_p_ranks:
                bi = int(order[k])
                cf += [("p", Y_SPEC.c_a, cp, bi) for cp in zs.cf_p] + [("base", Y_SPEC.c_a, Y_SPEC.c_p, bi)]
            for k in zs.cf_ranks:
                bi = int(order[k])
                cf += [("a", ca, Y_SPEC.c_p, bi) for ca in zs.cf_a] + [("ab", ca, cp, bi) for ca, cp in zs.cf_ab]
            res = self._pcells("ydiag", [(f"cf_{kd}_{ca:g}_{cp:g}_{bi}", dict(tkey, bi=bi), z_oc.cf_job,
                                          (theta, r, z, cor[bi], bi, kd, ca, cp, zs.cf_roots))
                                         for kd, ca, cp, bi in cf], "records", _log)
            return z_oc.rec_cf([c + (x["arrays"]["p"],) for c, x in zip(cf, res)], zs)

        def sub50():
            subs = [(ca, cp, bi) for ca, cp in zs.sub_thr for bi in range(zs.redraw_n)]
            res = self._pcells("ydiag", [(f"sub_{ca:g}_{cp:g}_{bi}", dict(tkey, bi=bi), z_oc.sub_job,
                                          (theta, r, z, cor[bi], bi, ca, cp)) for ca, cp, bi in subs], "records", _log)
            arr = {}
            for (ca, cp, _bi), x in zip(subs, res):
                arr.setdefault((ca, cp), []).append(x["arrays"]["p"])
            return z_oc.rec_sub({k: np.stack(v) for k, v in arr.items()}, zs)

        def psize():
            ps = [(n, bi) for n in zs.psize_ns for bi in range(zs.psize_draws)]
            res = self._pcells("ydiag", [(f"psize_{n}_{bi}", dict(tkey, n=n, bi=bi), z_oc.psize_job,
                                          (theta, r, z, n, bi, zs.psize_root, zs.sigma_share)) for n, bi in ps],
                               "records", _log)
            arr = {}
            for (n, _bi), x in zip(ps, res):
                arr.setdefault(n, []).append(x["arrays"]["p"])
            return z_oc.rec_psize({k: np.stack(v) for k, v in arr.items()}, zs)
        for name, fn in (("selfbase", selfbase), ("counterfactual", counterfactual), ("sub50", sub50),
                         ("psize", psize)):
            item(name, fn, True)
        return dict(computed=computed, null=null, detail=detail)

    # ================================================================ 0c: reuse (Z.7 0c)
    def stage_reuse(self) -> dict:
        self._require("reuse")
        zs, t0 = self.zs, time.perf_counter()
        why = []
        k = self.ctx["keys"]()
        if k.get("w_measure_key") != Y_SPEC.w_measure_key:
            why.append(f"W 측정 키 {k.get('w_measure_key')}")
        if k.get("u_measure_key") != Y_SPEC.u_measure_key:
            why.append(f"U 측정 키 {k.get('u_measure_key')}")
        why += self.ctx["y_reuse"]("1(Z 순서 0c에서 확인)")
        ydoc = self.ctx["y_doc"]()
        gf = self.ctx["git_facts"](Y_SPEC.summary, [c for _, c in zs.y_blocks])
        for (b, c), (_b, out) in zip(zs.y_blocks, zs.y_outcomes):
            if not (gf.get("ancestors") or {}).get(c):
                why.append(f"Y 블록 {b} 커밋 {c}가 HEAD 이력에 없음")
            if (ydoc.get(b) or {}).get("outcome") != out:
                why.append(f"Y 블록 {b}의 결과 {(ydoc.get(b) or {}).get('outcome')} ≠ {out}")
        st = ((ydoc.get("digest") or {}).get("set") or {})
        why += z_rules_digest(st)
        orc = ydoc.get("oracle") or {}
        if orc.get("detail_sha256") != zs.oracle_sha256:
            why.append(f"Y oracle 블록 detail_sha256 {orc.get('detail_sha256')} ≠ {zs.oracle_sha256}")
        if ((orc.get("derived") or {}).get("yield_rule") or {}).get("n_len") != zs.n_len:
            why.append(f"Y oracle 블록 N_len ≠ {zs.n_len}")
        try:
            items = z_split.lenient_items(Y_SPEC.oracle_detail, zs.oracle_sha256, zs.n_len)
            axes = {a: sum(x[2] == a for x in items) for a, _ in zs.n_len_axes}
            if axes != dict(zs.n_len_axes):
                why.append(f"관대 통과 축별 개수 {axes} ≠ {dict(zs.n_len_axes)}")
        except z_split.LenientMismatch as e:
            why.append(f"oracle.json: {e}")
        pil = ydoc.get("pilot") or {}
        pp = Path(Y_SPEC.pilot_detail)
        if not pp.exists() or sha256_file(pp) != pil.get("detail_sha256"):
            why.append(f"{Y_SPEC.pilot_detail} sha256 ≠ Y pilot 블록")
        else:
            _v, bad = self.ctx["y_pilot_back"](json.loads(pp.read_text())["manifest"])
            if bad:
                why.append(f"Y 파일럿 원자료 변경: {bad[:3]}")
        if tuple(pil.get("admitted") or ()) != zs.y_admitted or pil.get("n_sigma") != zs.y_n_sigma:
            why.append(f"Y 입장 쌍 {pil.get('admitted')} · n_Σ {pil.get('n_sigma')} ≠ 선언")
        decl = self.ctx["decl_sha"](yxw_files(), zs.decl_commit)
        now = files_sha(yxw_files())
        diff = sorted(f for f in now if decl.get(f) != now[f])
        if diff:
            why.append(f"y_* · x_* · w_* 가 {zs.decl_commit} 이후 바뀜: {diff[:5]}")
        body = z_rules.reuse_stop(why, "0c", zs) if why else dict(outcome=z_rules.PASS, reasons=[])
        self._finish("reuse", t0)
        return self._write("reuse", dict(body, yxw_decl=zs.decl_commit, n_yxw=len(now),
                                         note="Z.7 0c: Y.7 1 조건(Y _reuse_all) + Y 블록 일곱 + W · U 키 + 주 세트 digest + "
                                              "oracle.json sha256 · N_len 31 + Y 파일럿 매니페스트 · 입장 6 + y/x/w 불변."))

    # ================================================================ 0c′: sens (Z.9.2 P1-3, continue rule (가))
    def stage_sens(self) -> dict:
        self._require("sens")
        zs, t0 = self.zs, time.perf_counter()
        th = self.ctx["y_theta"]("1(Z 순서 0c′에서 확인)")
        if th["why"]:
            refuse(f"sens: Y θ̂ is not available: {th['why']}")
        tkey = dict(theta=z_oc.w_oc.summary(th["theta"]), z=th["z"])
        grid = [(n, di, d, bi) for n in zs.sens_ns for di, d in enumerate(zs.sens_deltas) for bi in range(zs.sens_draws)]
        jobs = [(f"sens_{n}_{di}_{bi}", dict(tkey, n=n, di=di, bi=bi), z_oc.sens_job,
                 (th["theta"], th["r"], th["z"], n, di, d, bi, zs.sens_seed, zs.sigma_share)) for n, di, d, bi in grid]
        res = self._pcells("sens", jobs, "core", _log)
        cells, L = {}, {}
        for n in zs.sens_ns:
            for di, d in enumerate(zs.sens_deltas):
                idx = [i for i, (n_, di_, _d, _b) in enumerate(grid) if (n_, di_) == (n, di)]
                a = z_oc.design_k(np.stack([res[i]["arrays"]["p"] for i in idx]), zs)
                s = z_oc.sens_summary(a, [res[i]["meta"] for i in idx], zs)
                cells[f"{n}|{di}"] = dict(n=n, delta=d, **s)
                L[(di, n)] = s["sim"]
        dec = z_rules.plan_rule_a(L, zs)
        table = [dict(n=n, delta_index=di, delta=d, limit=L[(di, n)]) for n in zs.sens_ns
                 for di, d in enumerate(zs.sens_deltas)]
        p = z_store.write_json(zs.sens_detail, dict(cells=cells), self.plist)
        summ = {k: {x: v[x] for x in ("n", "delta", "sim", "by_k", "base_only", "per_draw_worst_q", "n_below_bar",
                                      "n_cal_fail")} for k, v in cells.items()}
        self._finish("sens", t0)
        return self._write("sens", dict(dec, limits=table, cells=summ, detail_path=zs.sens_detail,
                                        detail_sha256=sha256_file(p),
                                        note="Z.9.2 P1-3 민감도 모의(분할 전 · 파일럿 전) + 계속 규칙 (가). 추출별 보정 a와 "
                                             "뽑힌 드리프트는 상세."))

    # ================================================================ 0d: split → key pre-check → continue rule (나)
    def stage_split(self) -> dict:
        doc = self._require("split")
        zs, t0 = self.zs, time.perf_counter()
        try:
            items = z_split.lenient_items(Y_SPEC.oracle_detail, zs.oracle_sha256, zs.n_len)
        except z_split.LenientMismatch as e:
            refuse(f"split: {e} (order 0c checked it — the file changed since)")
        forced = z_split.forced_odours(zs.y_admitted)
        sp = z_split.split(items, forced, zs.split_seed, zs.strata)
        rec = z_split.record(sp, forced, zs.strata)
        pre = z_rules.key_precheck(sp["P"], zs)
        p = z_store.write_json(zs.split_detail, dict(C=sp["C"], P=sp["P"], seed=zs.split_seed), self.plist)
        L = {(int(x["delta_index"]), int(x["n"])): float(x["limit"]) for x in doc["sens"]["limits"]}
        dec = (z_rules.pilot_few_0d(pre, len(sp["C"]), sp["forced_n"], zs)
               or z_rules.plan_rule_b(L, pre["j_max"], zs))
        self._finish("split", t0)
        return self._write("split", dict(dec, split=rec, precheck=pre, n_conf=len(sp["C"]), detail_path=zs.split_detail,
                                         detail_sha256=sha256_file(p),
                                         note="Z.3 + Z.9.2 P0-1: 키만 읽는 분할(X 냄새 묶음 · 축 층화 · 79_000_000) → 키 "
                                              "사전 점검(J_max, n_Σ^max) → 계속 규칙 (나). 키 목록은 split.json에만."))


def z_rules_digest(st: dict) -> list:
    """Y.7 0p-b's digest facts as Y's digest block recorded them (y_rules.digest_reasons, Y's numbers)."""
    from . import y_rules
    return y_rules.digest_reasons(dict(digest_keys=st.get("digest_keys"), n_b=st.get("n_b"), n_a=st.get("n_a"),
                                       last_turn=st.get("last_turn")), Y_SPEC)
