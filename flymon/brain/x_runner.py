"""Spec X's stage chain, phase A (X.5 0 · 0a; X.9.1.1): stage0 -> precheck, one block each in
results/summary/x_learning.json, written only through x_store, plus the running ledger (X.8's own 24 h). Phase B
appends its stages to ORDER after precheck (only after a precheck PASS) and never edits these two.
Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists, its own
block exists, an earlier gate did not PASS, the summary has uncommitted changes or a hashed X / W file is dirty. Both
stages re-check the W facts the precheck rests on (Reading 2): W measurement key, W pipeline key, w_measure.py's hash,
W's summary (tracked, clean, last commit), W blocks pilot / oc in HEAD's history on the W key, the pilot manifest's
sha256 values, results/w/oc.json's sha256, the pair order (W's own _pilot_back = block pilot's pairs), the balanced /
V1 pair facts and θ̂'s summary = W block oc's theta — any break → STOP_REUSE (records_unavailable, P2-10).
Long computations are resumable cells under results/x/progress/<stage>/ (one JSON file per cell, keyed by the cell's
inputs, the XSpec and the X files' sha256): a killed run, rerun with the same command, reuses every finished cell and
gets the same values (same seeds). Every block and detail is written as plain JSON (x_store.to_json) and the stage
returns exactly what was written."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import w_oc, w_runner, x_oc, x_rules, x_store, x_verdict
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .w_spec import SPEC as W_SPEC

ORDER = ("stage0", "precheck")
GATES = ("stage0", "precheck")
X_FILES = ("flymon/brain/x_spec.py", "flymon/brain/x_verdict.py", "flymon/brain/x_oc.py", "flymon/brain/x_rules.py",
           "flymon/brain/x_store.py", "flymon/brain/x_runner.py", "scripts/run_x.py")
W_FILES = tuple(dict.fromkeys(tuple(w_runner.W_PIPELINE_FILES) + ("flymon/brain/w_measure.py",)))
X_HASHED_FILES = X_FILES + W_FILES + ("results/summary/w_learning.json",)


def git_state() -> dict:
    return _h3_git_state(files=X_HASHED_FILES)


def files_sha(files) -> dict:
    return {f: sha256_file(_ROOT / f) for f in files}


def env_hashes() -> dict:
    """X.9.1.3 P2-11: uv.lock, Python, numpy, the X files and the W files X imports."""
    return dict(uv_lock_sha256=sha256_file(_ROOT / "uv.lock"), python=platform.python_version(),
                numpy=np.__version__, x_files=files_sha(X_FILES), w_files=files_sha(W_FILES))


def refuse(msg: str, code: int = 2):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def build_ctx(npz: str) -> dict:
    """The real context: W's summary and OC detail (read only), W's pilot through W's own _pilot_back on W's context
    (built once; the manifest's sha256 values checked first), the keys and git facts, the C3 params (writer guard)."""
    from . import w_store
    from .w_measure import w_measure_key
    cache = {}

    def wctx():
        if "c" not in cache:
            cache["c"] = w_runner.build_ctx(W_SPEC, npz)
        return cache["c"]

    def pilot(w_doc):
        _got, bad = w_store.load_manifest(w_doc["pilot"]["manifest"])
        if bad:
            return None, bad
        pairs, _ = w_runner.Runner(None, None, wctx(), W_SPEC)._pilot_back(w_doc)
        return pairs, []

    def oc_detail(path):
        p = Path(path)
        return (json.loads(p.read_text()), sha256_file(p)) if p.exists() else (None, None)
    return dict(w_doc=lambda: json.loads(Path(W_SPEC.summary).read_text()), oc_detail=oc_detail, pilot=pilot,
                keys=lambda: dict(w_measure_key=w_measure_key(npz)["key"], pipeline_key=w_runner.pipeline_key()["key"],
                                  w_measure_sha=sha256_file(_ROOT / "flymon/brain/w_measure.py")),
                facts=lambda xs: dict(w_runner.git_facts(xs.w_summary, [c for _, c in xs.w_commits]),
                                      git=summary_git(xs.w_summary)),
                params=lambda: wctx()["params"])


class Runner:
    def __init__(self, ctx: dict, xs, summary_path=None):
        self.ctx, self.xs = ctx, xs
        self.summary_path = str(summary_path or xs.summary)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return x_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed X / W files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; X never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; X never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != x_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — X stops there")
        return doc

    def _write(self, stage: str, body: dict, wall_s: float) -> dict:
        k = self.ctx["keys"]()
        block = dict(body, stage=stage, w_measure_key=k.get("w_measure_key"), pipeline_key=k.get("pipeline_key"),
                     git=git_state(), written_at=_now())
        block = x_store.to_json(block)
        x_store.write_summary_block(self.summary_path, stage, block, self.plist,
                                    dict(stage=stage, wall_s=float(wall_s), at=block["written_at"]))
        return block

    # ---- the W facts (Reading 2) -------------------------------------------------------------------------------------
    def _reuse(self) -> tuple:
        xs, c = self.xs, self.ctx
        w_doc = c["w_doc"]()
        why = x_rules.w_reuse_reasons(w_doc, c["keys"](), c["facts"](xs), c["oc_detail"](xs.w_oc_detail)[1], xs)
        pairs = theta = None
        if not why:
            pairs, bad = c["pilot"](w_doc)
            if bad:
                why.append("W 파일럿 원자료: " + "; ".join(bad[:3]))
            elif list(pairs) != list(w_doc["pilot"]["pairs"]):
                why.append("W _pilot_back 쌍 순서 ≠ W 파일럿 블록 pairs")
            else:
                theta = w_oc.fit(list(pairs.values()))
                if json.loads(canonical(w_oc.summary(theta))) != w_doc["oc"]["theta"]:
                    why.append("θ̂ 요약 ≠ W oc 블록 theta")
                why += x_rules.pair_fact_reasons(w_doc, list(pairs), xs)
        return why, w_doc, pairs, theta

    @staticmethod
    def _z(w_doc: dict) -> dict:
        return {k: (float(v[0]), float(v[1])) for k, v in w_doc["reuse"]["z_V"].items()}

    def _tables(self) -> dict:
        xs = self.xs
        return dict(designs=dict(p_set=list(xs.p_set_grid), q=list(xs.q_grid), K=list(xs.k_grid),
                                 F=[xs.f_min, xs.f_max], k=[xs.k_min, xs.k_cap],
                                 n=len(xs.p_set_grid) * len(xs.q_grid) * len(xs.k_grid) * (xs.f_max - xs.f_min + 1)),
                    set_rule=dict(min_gate_pairs=xs.min_gate_pairs, min_pass_pairs=xs.min_pass_pairs,
                                  digits=xs.round_digits),
                    m_needed=x_verdict.m_needed_table(xs),
                    seeds=dict(oc=xs.oc_seed, precheck=xs.precheck_seed, diag=xs.diag_seed, w_diagnosis=W_SPEC.oc_seed,
                               tags=dict(cal=x_oc.TAG_CAL, point=x_oc.TAG_POINT, record=x_oc.TAG_RECORD,
                                         synth=x_oc.TAG_SYNTH, p26=x_oc.TAG_P26, diag_grid=x_oc.TAG_DIAG_GRID,
                                         diag_pair=x_oc.TAG_DIAG_PAIR, diag_accept=x_oc.TAG_DIAG_ACCEPT)),
                    calibration=dict(rule="X.9.1.2 상태별", bracket_mult=xs.bracket_mult,
                                     w_bracket_mult=xs.w_bracket_mult, cal_iter=xs.cal_iter,
                                     cal_retry_iter=xs.cal_retry_iter, cal_tol=xs.cal_tol, cal_reps=xs.cal_reps,
                                     fill=dict(power=x_oc.fill_value("min"), false=x_oc.fill_value("max")),
                                     cal_floor_rule=xs.cal_floor_rule),
                    protocol="W 표 그대로(w_spec, W stage0 tables) — X.2")

    # ---- resumable cells (X.9.1.5: 2 h limit, resume) ---------------------------------------------------------------
    def _cells(self, stage: str):
        """The cell runner of one stage: progress_dir/<stage>/<tag>.json holding {key, value, at}. A file whose key
        differs (other inputs, another XSpec, changed X files) is recomputed and replaced. Returns the JSON form, on the
        first run as on a resumed one."""
        spec_sha = hashlib.sha256(canonical(self.xs).encode()).hexdigest()
        x_sha = files_sha(X_FILES)

        def cell(tag: str, key_obj, fn):
            key = hashlib.sha256(canonical(dict(tag=tag, key=key_obj, xs=spec_sha, x=x_sha)).encode()).hexdigest()
            p = Path(self.xs.progress_dir) / stage / (re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".json")
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                except ValueError:
                    d = {}
                if d.get("key") == key:
                    return d["value"]
            v = x_store.to_json(fn())
            x_store.write_json(str(p), dict(key=key, value=v, at=_now()), self.plist)
            return v
        return cell

    # ================================================================ 0: code checks + W diagnosis (X.5 0)
    def stage_stage0(self) -> dict:
        self._require("stage0")
        xs = self.xs
        t0 = time.perf_counter()
        why, w_doc, _pairs, theta = self._reuse()
        z = self._z(w_doc)
        cell = self._cells("stage0")
        diag = p26 = timing = None
        detail = {}
        if not why:
            oc, _sha = self.ctx["oc_detail"](xs.w_oc_detail)
            boot = oc["boot_calibration"]
            diag = cell("w_cal_diagnosis", dict(theta=w_oc.summary(theta), z=z,
                                                boot=hashlib.sha256(canonical(boot).encode()).hexdigest()),
                        lambda: x_oc.w_cal_diagnosis(theta, z, W_SPEC, xs, boot, log=_log))
            why += x_rules.diagnosis_reasons(diag, xs)
            p = x_store.write_json(xs.wcal_detail, diag, self.plist)
            detail = dict(wcal_detail_path=xs.wcal_detail, wcal_detail_sha256=sha256_file(p))
        syn = cell("synthetic", dict(z=z), lambda: x_oc.synthetic_validation(xs, z))
        if theta is not None and not why:
            idx = x_oc.rng(xs.oc_seed, x_oc.TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
            c = x_oc.calibrate_x(theta, xs.d_power, "min", idx, z, xs)
            a, b = (c["a"]["value"], c["b"]["value"]) if c["ok"] else (0.0, 0.0)
            p26 = cell("p26", dict(theta=w_oc.summary(theta), z=z, a=a, b=b),
                       lambda: x_oc.bit_identity(theta, z, xs, W_SPEC, xs.p26_reps, a, b))
            timing = x_oc.oc_timing(theta, z, xs)
        if why:
            dec = x_rules.reuse_stop(why)
        elif not syn["ok"] or not p26["equal"]:
            dec = dict(outcome=x_rules.INVALID, reasons=([] if syn["ok"] else ["X.4.6 합성 검증 실패"])
                       + ([] if p26["equal"] else ["P2-6 비트 동일 실패"]))
        else:
            dec = dict(outcome=x_rules.PASS, reasons=[])
        body = dict(dec, synthetic=syn, p26=p26, oc_timing=timing, tables=self._tables(),
                    w_cal_diagnosis=None if diag is None else dict(reproduced=diag["reproduced"],
                                                                   n_draws=diag["n_draws"], counts=diag["counts"],
                                                                   diffs=diag["diffs"][:5]),
                    decision_files=files_sha(X_FILES), w_files=files_sha(W_FILES), **detail,
                    note="X.5 순서 0(국면 A): 판정 코드·x_oc·합성 검증·P2-6·W 보정 진단. 진단은 규칙을 바꾸지 않는다"
                         "(X.9.1.2).")
        return self._write("stage0", body, time.perf_counter() - t0)

    # ================================================================ 0a: point-θ precheck + record-only diagnostics
    def stage_precheck(self) -> dict:
        self._require("precheck")
        xs = self.xs
        t0 = time.perf_counter()
        why, w_doc, pairs, theta = self._reuse()
        if why:
            return self._write("precheck", x_rules.reuse_stop(why), time.perf_counter() - t0)
        z = self._z(w_doc)
        abs_d = x_rules.abs_naive_d(w_doc, list(pairs))
        cell = self._cells("precheck")
        pc = x_oc.precheck(theta, z, xs, cell=cell, log=_log)
        _log(f"x precheck point done ({pc['timing_s']:.0f} s)")
        dec = x_rules.precheck_decision(pc)
        rec = None
        if dec["outcome"] != x_rules.PASS:
            rec = x_oc.point_records(theta, abs_d, pc["records_target"], z, xs, cell=cell, log=_log)
        diag = x_oc.diagnostics(theta, abs_d, z, xs, cell=cell, log=_log)
        resid = x_oc.residual_compare(list(pairs.values()), abs_d < xs.naive_max, xs)
        p1 = x_store.write_json(xs.precheck_detail, dict(precheck=pc, records=rec), self.plist)
        p2 = x_store.write_json(xs.diag_detail, diag, self.plist)
        body = dict(dec, theta=w_oc.summary(theta),
                    calibration={m: dict(ok=c["ok"], failure=c["failure"], first_failure=c["first_failure"],
                                         retried=c["retried"], a=(c["a"] or {}).get("value"),
                                         b=(c["b"] or {}).get("value"), true_dprime=c.get("true_dprime"))
                                 for m, c in pc["calibration"].items()},
                    best_power=pc["best_power"], records_target=pc["records_target"], at_f32=pc["at_f32"],
                    n_passing=len(pc["passing"]), m_needed=pc["m_needed"], records=rec,
                    records_note=None if rec is not None else "사전 점검 통과 — records는 국면 B의 OC(X.4.5)",
                    diag=x_oc.diag_summary(diag), residuals=resid, env=env_hashes(),
                    precheck_detail_path=xs.precheck_detail, precheck_detail_sha256=sha256_file(p1),
                    diag_detail_path=xs.diag_detail, diag_detail_sha256=sha256_file(p2),
                    timing=dict(precheck_s=pc["timing_s"], diag_s=diag["timing_s"]),
                    note="점 θ 사전 점검(X.9.1.1): 점 추정·V0·포락선 없음 — 통과해도 자격이 아니다. 기록 전용 진단"
                         "(X.9.1.4)은 관문에 쓰지 않는다.")
        return self._write("precheck", body, time.perf_counter() - t0)
