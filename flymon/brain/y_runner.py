"""Spec Y's stage chain, step 0p (Y.7 0p-b · 0p-c): digest -> oracle, one block each in results/summary/y_learning.json,
written only through y_store, plus the running ledger under `budget` (Y's own 24 h). Phase A appends its stages to
ORDER after oracle and never edits these two.
- digest: the U / W measurement keys, V's blocks through W's own reuse decision (w_runner.Runner._reuse_dec: V's
  z / kc_input / set / judge on V's keys, SELECTED, z_V as declared; R shared / T / U keys), then w_pairs.w_set
  regenerated: digest_keys 65dbf001… (0p plan Reading 1), (b) 167 · (a) 82, last turn 1967. Any break -> STOP_REUSE.
  The block records set_summary + keys (W never wrote a set block).
- oracle: W's stage_oracle path — main_rows checked against block digest's set, the measurement keys checked against
  block digest's (refusal otherwise), RMeasurer.oracle(rows, V_SPEC.cond("L"), "screen", W_SPEC.oracle_seeds()) over
  results/y/cache (W measurement key), pair_stats on z_V, the lever check per pair, L_A^or / L_P^or from report pre
  (Reading 3). Per-pair values go only to results/y/oracle.json (git-ignored, sha256 in the block), in exactly the
  record shape y_rules.count_table reads (axis, value, failure, testable, d_pre, L_A, L_P, plus key / c / turn / r / p);
  the block's count table is count_table of those very records (Y red-team P2-11: recomputable from oracle.json). The
  block holds the count table only (author's reading 6, Y.7 0p-c read rule). Resumable: the cache holds every finished
  pair; a killed run writes no block, and the wall time accumulates in results/y/progress (Reading 6).
- Failures (Reading 4): a lever mismatch counts in failures and makes the block INVALID (exit 5, not committed); an
  undefined d′ (pair_stats None) counts in no_value only.
- Archive (Y red-team P3-13 as amended by Y.9.2, y_store.archive_stage): the stage's fixed file set is copied just
  before every block that will be committed is written — PASS or a STOP, digest included (empty set in 0p) — so the
  copy precedes any STOP sentence; never for INVALID (a machine defect the controller discards). A refused archive
  leaves no block. Runner.archive(stage) (run_y.py --stage archive --name <stage>) repeats it after the commit;
  the repeat is idempotent.
- Y.9.2 P2-11: only oracle.json is bit-bound. When results/y/oracle.json exists for exactly the main rows, seeds and
  z_V and every record rebuilds bit-identically from its cache file, the oracle stage re-derives the block from it
  without measuring (detail_reused); otherwise it measures (cache-resumed) and records any replaced file's sha256.
- The block's derived section (y_rules.derive): count table, P1-7 yield rule (k ranges kept), P1-4 lenient-level
  quantiles; the early STOP_FEW_PAIRS now also fires when the yield rule drops both k ranges.
Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block or its own block
exists, an earlier gate did not PASS, the summary has uncommitted changes, or a hashed Y / W / V file is dirty.
Phase A (Y.7 orders 0 · 1 · 4 · 5; appended after oracle, digest / oracle untouched): stage0 (Y.6.6 fixtures,
synthetic validation, bit identity on W θ̂ with X's power a, b, threshold reproduction, fixture 6, Y.3.4 comparison,
OC timing, tables, decision-file and test hashes; INVALID on a failed check — never committed), reuse (X.5 1 through
X's own _reuse + Y.7 1's X facts), pilot (Y.4 on V's four candidates + W's three, 4a, θ, records), precheck (Y.6.1 on
the oracle block's kept k ranges, P2-12, P2-10, STOP records, env hashes). stage0 and pilot / precheck re-run the reuse
checks they rest on (STOP_REUSE, 〈where〉 "1(…)"). Long computations are resumable cells under
results/y/progress/<stage>/ (key = cell inputs + YSpec + Y files' sha256); the wall time of finished cells is kept in
results/y/progress/<stage>.json so a killed run's spend stays in the ledger.
Phase B (Y.7 orders 6–12; appended): ORDER / GATES grow by oc · smoke · budget_gate · gates · estimate · learn ·
band · records · seal · judge (read at call time by _require); build_ctx's dict gains _phase_b_ctx's entries; the
bootstrap and the reconfirmation run as resumable .npz cells in a spawn process pool (phase-B plan Reading 2); W's
Runner reads a view of Y's blocks (learn / band / records units, the seal's declared inputs, the judgement)."""
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
from . import w_oc, w_records, w_runner, w_verdict, x_runner, x_verdict, y_oc, y_rules, y_store
from .h3_store import ROOT, canonical
from .h3_store import git_state as _h3_git_state
from .h3_store import sha256_file
from .h4_formula import pair_stats
from .v_spec import SPEC as V_SPEC
from .w_pairs import set_summary
from .r_pairs import row_key
from .w_spec import SPEC as W_SPEC
from .x_spec import SPEC as X_SPEC

ORDER = ("digest", "oracle", "stage0", "reuse", "pilot", "precheck")
GATES = ORDER
Y_FILES = ("flymon/brain/y_spec.py", "flymon/brain/y_rules.py", "flymon/brain/y_store.py", "flymon/brain/y_runner.py",
           "scripts/run_y.py", "flymon/brain/y_oc.py")
W_FILES = tuple(dict.fromkeys(tuple(w_runner.W_PIPELINE_FILES) + ("flymon/brain/w_measure.py",)))
X_FILES = tuple(x_runner.X_FILES)
Y_HASHED_FILES = tuple(dict.fromkeys(Y_FILES + W_FILES + X_FILES + tuple(w_runner.V_HASHED_FILES)
                                     + ("results/summary/v_lever.json", "results/summary/w_learning.json",
                                        "results/summary/x_learning.json")))
Y_TEST_FILES = ("tests/brain/test_y_spec.py", "tests/brain/test_y_rules_a.py", "tests/brain/test_y_cal.py",
                "tests/brain/test_y_eval.py", "tests/brain/test_y_records.py", "tests/brain/test_y_precheck.py",
                "tests/brain/test_y_runner_a.py")


def git_state() -> dict:
    return _h3_git_state(files=Y_HASHED_FILES)


def refuse(msg: str, code: int = 2):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def env_hashes() -> dict:
    """X.9.1.3 P2-11 for Y (Y.7 5): uv.lock, Python, numpy, y_* and the x_* / w_* files Y imports."""
    return dict(uv_lock_sha256=sha256_file(ROOT / "uv.lock"), python=platform.python_version(), numpy=np.__version__,
                y_files=x_runner.files_sha(Y_FILES), x_files=x_runner.files_sha(X_FILES),
                w_files=x_runner.files_sha(W_FILES))


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def build_ctx(npz: str) -> dict:
    """W's context (built once, read only) and the keys; measurer(pool) = an RMeasurer over U's job copies behind
    YCache(results/y/cache, W measurement key) with z_V as declared."""
    from .h3_store import code_key
    from .r_measure import R_MEASURE_FILES, RMeasurer
    from .t_measure import t_measure_key
    from .u_measure import UPool, u_measure_key
    from .w_measure import w_measure_key
    from .y_spec import SPEC as YS
    cache = {}

    def wctx():
        if "c" not in cache:
            cache["c"] = w_runner.build_ctx(W_SPEC, npz)
        return cache["c"]

    def reuse():
        r = w_runner.Runner(None, None, wctx(), W_SPEC, code=code_key(npz, files=R_MEASURE_FILES),
                            tcode=t_measure_key(npz), ucode=u_measure_key(npz))
        return r._reuse_dec()

    def measurer(pool):
        c = wctx()
        return RMeasurer(UPool(pool), y_store.YCache(YS.cache_dir, w_measure_key(npz)), V_SPEC, c["params"],
                         c["readout"], W_SPEC.z_v(), c["types"], c["n_kc"])
    def xctx():
        if "x" not in cache:
            from . import x_runner as XR
            cache["x"] = XR.build_ctx(npz)
        return cache["x"]

    def x_reuse():
        return x_runner.Runner(xctx(), X_SPEC)._reuse()

    def x_facts(ys):
        p, dp = Path(ys.x_summary), Path(ys.x_precheck_diag)
        gf = w_runner.git_facts(ys.x_summary, [c for _, c in ys.x_commits])
        wd = json.loads(Path(W_SPEC.summary).read_text())
        wg = w_runner.git_facts(W_SPEC.summary, [c for _, c in ys.w_commits])
        return dict(x_doc=json.loads(p.read_text()) if p.exists() else {}, ancestors=gf.get("ancestors"),
                    last=gf.get("last"), git=summary_git(ys.x_summary),
                    diag_sha=sha256_file(dp) if dp.exists() else None, x_files=x_runner.files_sha(X_FILES),
                    w_ancestors=wg.get("ancestors"),
                    w_blocks={b: (wd.get(b) or {}).get("w_measure_key") for b, _ in ys.w_commits})

    def w_oc_detail():
        p = Path(W_SPEC.oc_detail)
        return json.loads(p.read_text()), sha256_file(p)

    def v_candidates(ys):
        c = wctx()
        n = len(c["v_doc"]()["jm:L"]["manifest"])
        got = {row_key(r): (r, res) for r, res in c["v_jm"]("L", n)}
        missing = [k for k in ys.pilot_v_pairs if k not in got]
        if missing:
            raise ValueError(f"V 세트 후보 {missing}가 V jm:L에 없음")
        return [got[k] for k in ys.pilot_v_pairs]

    def v_even_oracle(keys, z):
        from .r_measure import RMeasurer
        from .w_store import VReadCache
        c = wctx()
        want = set(keys)
        rows = [r for r in c["even_rows"] if row_key(r) in want]
        m = RMeasurer(None, VReadCache(V_SPEC.cache_dir, u_measure_key(npz)), V_SPEC, c["params"], c["readout"], z,
                      c["types"], c["n_kc"])
        return {g["key"]: g["result"] for g in m.oracle(rows, V_SPEC.cond("L"), "even", V_SPEC.h4_seeds())}

    def pilot_measure(rows, pool, check=None):
        import dataclasses
        from .w_measure import WMeasurer
        c = wctx()
        yw = dataclasses.replace(W_SPEC, pilot_probe_seed0=YS.pilot_probe_seed0,
                                 pilot_train_seed0=YS.pilot_train_seed0)
        windows = dict(strength=W_SPEC.strength, settle_ms=V_SPEC.settle_ms, read_ms=V_SPEC.read_ms,
                       window_ms=V_SPEC.window_ms)
        timing = dict(present_ms=W_SPEC.pulse_ms, gap_ms=W_SPEC.gap_ms, train_settle_ms=W_SPEC.train_settle_ms,
                      seed_stride=W_SPEC.fly_train_stride)
        wm = WMeasurer(pool, y_store.YCache(YS.cache_dir, w_measure_key(npz)), c["params"], c["readout"],
                       V_SPEC.p_type, W_SPEC.reward_dan, W_SPEC.punish_dan, windows, timing)
        r = w_runner.Runner(None, lambda smoke=False: wm, c, yw)
        units = r.units(rows, "pilot", YS.pilot_probes, YS.pilot_flies, V_SPEC.lever_edit)
        got = wm.learn(units, "pilot", check=check)
        pairs, mach = r._pilot_data(rows, got, YS.pilot_probes, YS.pilot_flies)
        return dict(pairs=pairs, machine=mach, manifest=r._manifest(got), jobs=[g["result"] for g in got],
                    n_units=len(units))

    def pilot_back(manifest):
        from . import w_store
        got, bad = w_store.load_manifest(manifest)
        if bad:
            return None, bad
        by = {}
        for g in got:
            pair, fly, brain = g["key"].rsplit("|", 2)
            by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
        return {k: w_records.pair_data(v, list(range(YS.pilot_flies))) for k, v in by.items()}, []
    return dict(keys=lambda: dict(w_measure_key=w_measure_key(npz)["key"], u_measure_key=u_measure_key(npz)["key"]),
                reuse=reuse, w_set=lambda: wctx()["w_set"](), main_rows=lambda blk: wctx()["main_rows"](blk),
                params=lambda: wctx()["params"], measurer=measurer, x_reuse=x_reuse, x_facts=x_facts,
                w_oc_detail=w_oc_detail, v_candidates=v_candidates, v_even_oracle=v_even_oracle,
                pilot_measure=pilot_measure, pilot_back=pilot_back, **_phase_b_ctx(wctx, npz))


class Runner:
    def __init__(self, ctx: dict, ys, measure=None, summary_path=None, pool=None):
        self.ctx, self.ys, self.measure, self.pool = ctx, ys, measure, pool
        self.summary_path = str(summary_path or ys.summary)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return y_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed Y / W / V files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; Y never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; Y never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != y_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — Y stops there")
        return doc

    def _write(self, stage: str, body: dict, wall_s: float) -> dict:
        """Archive the stage's fixed file set (PASS / STOP only; before the block, so the copy precedes any STOP
        sentence and a refused archive leaves no block), then the block."""
        k = self.ctx["keys"]()
        extra = {}
        if body.get("outcome") != y_rules.INVALID:
            extra["archive"] = y_store.archive_stage(stage, self.ys)
        block = y_store.to_json(dict(body, **extra, stage=stage, w_measure_key=k.get("w_measure_key"),
                                     u_measure_key=k.get("u_measure_key"), git=git_state(), written_at=_now()))
        y_store.write_summary_block(self.summary_path, stage, block, self.plist,
                                    dict(stage=stage, wall_s=float(wall_s), at=block["written_at"]))
        return block

    def archive(self, stage: str) -> list:
        """The controller's post-commit archive call (scripts/run_y.py --stage archive --name <stage>): the stage's
        block must exist and not be INVALID; idempotent (y_store.archive_stage)."""
        blk = self._doc().get(stage)
        if blk is None:
            refuse(f"archive {stage}: no block {stage} in {self.summary_path}")
        if blk.get("outcome") == y_rules.INVALID:
            refuse(f"archive {stage}: block {stage} is INVALID; INVALID is never archived")
        return y_store.archive_stage(stage, self.ys)

    def _prog_path(self, stage: str) -> str:
        return f"{self.ys.progress_dir}/{stage}.json"

    def _prog(self, stage: str) -> float:
        p = Path(self._prog_path(stage))
        return float(json.loads(p.read_text())["wall_s"]) if p.exists() else 0.0

    def _prog_add(self, stage: str, s: float) -> float:
        tot = self._prog(stage) + float(s)
        y_store.write_json(self._prog_path(stage), dict(wall_s=tot, at=_now()), self.plist)
        return tot

    # ================================================================ 0p-b: partial reuse + the main-set digest
    def stage_digest(self) -> dict:
        self._require("digest")
        ys, t0 = self.ys, time.perf_counter()
        keys = self.ctx["keys"]()
        why = y_rules.key_reasons(keys, ys)
        dec = self.ctx["reuse"]()
        if dec.get("outcome") != y_rules.PASS:
            why += list(dec.get("reasons") or [dec.get("outcome")])
        js = None
        try:
            js = self.ctx["w_set"]()
        except ValueError as e:
            why.append(f"주 세트 재생성 불가: {e}")
        if js is not None:
            why += y_rules.digest_reasons(js, ys)
        body = y_rules.reuse_stop(why, "0p-b") if why else dict(outcome=y_rules.PASS, reasons=[])
        body.update(set=None if js is None else dict(set_summary(js), keys=js["keys"]), keys=keys,
                    v_reuse=dict(outcome=dec.get("outcome"), reasons=list(dec.get("reasons") or [])),
                    note="Y.7 0p-b: U · W 측정 키, V 블록(w_rules.reuse), 주 세트 digest_keys · (b) · (a) · 마지막 턴.")
        return self._write("digest", body, time.perf_counter() - t0)

    # ================================================================ 0p-c: the main-set oracle (count table only)
    def stage_oracle(self) -> dict:
        doc = self._require("oracle")
        ys = self.ys
        keys = self.ctx["keys"]()
        got_k = {k: doc["digest"].get(k) for k in ("w_measure_key", "u_measure_key")}
        if got_k != {k: keys.get(k) for k in got_k}:
            refuse(f"the measurement keys {keys} are not block digest's {got_k}")
        try:
            rows = self.ctx["main_rows"](doc["digest"]["set"])
        except ValueError as e:
            refuse(f"the main set does not reproduce block digest: {e}")
        z = W_SPEC.z_v()
        seeds = W_SPEC.oracle_seeds()
        t0 = time.perf_counter()
        prev = Path(ys.oracle_detail)
        prev_sha = sha256_file(prev) if prev.exists() else None
        per = self._detail_from_cache(rows, seeds, z)
        reused = per is not None
        if reused:                  # Y.9.2 P2-11: oracle.json matches the cache -> derive only, no measuring
            wall = self._prog_add("oracle", time.perf_counter() - t0)
            p = prev
        else:
            m = self.measure()
            try:
                got = m.oracle(rows, V_SPEC.cond("L"), "screen", seeds)
            finally:
                wall = self._prog_add("oracle", time.perf_counter() - t0)
            per = [self._record(r, g["key"], g["result"], z) for r, g in zip(rows, got, strict=True)]
            man = [dict(key=g["key"], cache_file=g["cache_file"], cache_key=g["cache_key"],
                        sha256=sha256_file(g["cache_file"])) for g in got]
            p = y_store.write_json(ys.oracle_detail, dict(pairs=per, manifest=man, seeds=seeds,
                                                          z_V={k: list(v) for k, v in z.items()}), self.plist)
        # the derived section from the records as written (Y red-team P2-11: recomputable from oracle.json alone)
        derived = y_rules.derive(json.loads(Path(p).read_text())["pairs"], ys)
        dec = y_rules.oracle_decision(derived["counts"], ys)
        if "stage" in dec:                   # the STOP's stage ("early") would collide with the block's stage name
            dec["stop_stage"] = dec.pop("stage")
        sha = sha256_file(p)
        body = dict(dec, derived=derived, n=len(per), detail_path=ys.oracle_detail, detail_sha256=sha,
                    detail_reused=reused,
                    replaced_detail_sha256=prev_sha if prev_sha not in (None, sha) else None,
                    seeds=seeds, z_V={k: list(v) for k, v in z.items()}, condition="L", block="screen",
                    note="Y.7 0p-c: 개수 표만(쌍별 값은 git 제외 상세 파일, 글쓴이 해석 6) + Y.9.2 P1-7 수율 규칙 · "
                         "P1-4 관대 통과 수준 요약 분위수(derived; oracle.json에서 다시 계산, P2-11). 오라클 · 순진 "
                         "pre만으로는 주 세트를 사용한 것이 아니다(Y.0).")
        return self._write("oracle", body, wall)

    @staticmethod
    def _record(r: dict, key: str, res: dict, z) -> dict:
        """One pair's oracle.json record (the shape y_rules.count_table reads)."""
        q = res.get("q", {})
        want_q = (V_SPEC.lever_edit, V_SPEC.sha_combined, V_SPEC.lever_edges)
        st = pair_stats(res["report"], z, V_SPEC.testable_min)
        la, lp = y_rules.levels(res["report"])
        return dict(key=key, c=r["c"], axis=r["axis"], turn=r["turn"], value=st is not None,
                    failure=(q.get("edit"), q.get("csc_sha256"), q.get("edit_edges")) != want_q,
                    testable=bool(st and st["testable"]), d_pre=st and st["d_pre"], r=st and st["r"],
                    p=st and st["p"], L_A=la, L_P=lp)

    def _detail_from_cache(self, rows: list, seeds, z):
        """Y.9.2 P2-11: oracle.json's records when the file exists for exactly these rows, seeds and z_V and every
        record is rebuilt bit-identically from its cache file (sha256 as in the manifest); else None (measure)."""
        p = Path(self.ys.oracle_detail)
        if not p.exists():
            return None
        try:
            det = json.loads(p.read_text())
            man, pairs = det["manifest"], det["pairs"]
            if (det.get("seeds") != y_store.to_json(seeds)
                    or det.get("z_V") != y_store.to_json({k: list(v) for k, v in z.items()})
                    or len(man) != len(rows) or len(pairs) != len(rows)):
                return None
            rebuilt = []
            for r, e in zip(rows, man):
                f = Path(e["cache_file"])
                if not f.exists() or sha256_file(f) != e["sha256"]:
                    return None
                c = json.loads(f.read_text())
                if c.get("key") != e["cache_key"]:
                    return None
                rebuilt.append(self._record(r, e["key"], c["result"], z))
        except (KeyError, TypeError, ValueError):
            return None
        return pairs if y_store.to_json(rebuilt) == pairs else None

    # ================================================================ phase A helpers
    def _cells(self, stage: str):
        """Resumable cells: progress_dir/<stage>/<tag>.json = {key, value, at}; a different key (inputs, YSpec, Y
        files) is recomputed. Each computed cell's seconds are added to progress_dir/<stage>.json."""
        spec_sha = hashlib.sha256(canonical(self.ys).encode()).hexdigest()
        y_sha = x_runner.files_sha(Y_FILES)

        def cell(tag: str, key_obj, fn):
            key = hashlib.sha256(canonical(dict(tag=tag, key=key_obj, ys=spec_sha, y=y_sha)).encode()).hexdigest()
            p = Path(self.ys.progress_dir) / stage / (re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".json")
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                except ValueError:
                    d = {}
                if d.get("key") == key:
                    return d["value"]
            t = time.perf_counter()
            v = y_store.to_json(fn())
            y_store.write_json(str(p), dict(key=key, value=v, at=_now()), self.plist)
            dt = time.perf_counter() - t
            self._prog_add(stage, dt)
            self._cell_s += dt
            return v
        self._cell_s = 0.0
        return cell

    def _finish_wall(self, stage: str, t0: float) -> float:
        return self._prog_add(stage, (time.perf_counter() - t0) - getattr(self, "_cell_s", 0.0))

    def _tables(self) -> dict:
        ys = self.ys
        n = (len(ys.p_set_grid) * len(ys.q_grid) * len(ys.k_grid) * (ys.f_max - ys.f_min + 1) * len(ys.k_ranges))
        tags = dict(cal=y_oc.TAG_CAL, point=y_oc.TAG_POINT, record=y_oc.TAG_RECORD, synth=y_oc.TAG_SYNTH,
                    p26=y_oc.TAG_P26, boot=y_oc.TAG_BOOT, accept=y_oc.TAG_ACCEPT,
                    **{s: y_oc.tag(s) for s in ("mix", "near", "thr", "R-V", "R-pre")})
        return dict(designs=dict(p_set=list(ys.p_set_grid), q=list(ys.q_grid), K=list(ys.k_grid),
                                 F=[ys.f_min, ys.f_max], k_ranges=[list(k) for k in ys.k_ranges], n=n),
                    m_needed=x_verdict.m_needed_table(ys),
                    filter=dict(naive_max=ys.naive_max, c_a=ys.c_a, c_p=ys.c_p, lenient=[ys.lenient_d, ys.lenient_a,
                                                                                         ys.lenient_p]),
                    calibration=dict(grid_steps=ys.grid_steps, widen=list(ys.widen), refine_delta=ys.refine_delta,
                                     refine_parts=ys.refine_parts, knob_tol=ys.knob_tol, cal_tol=ys.cal_tol,
                                     cal_reps=ys.cal_reps, coarse_step_flag=ys.coarse_step_flag,
                                     fill=dict(power=y_oc.x_oc.fill_value("min"), false=y_oc.x_oc.fill_value("max"))),
                    generator=dict(tries=ys.tries, fill_max=ys.fill_max, scenarios=list(y_oc.SCENARIOS)),
                    seeds=dict(pilot_probe="60_000_000 + j·4_000 + f·100 + k",
                               pilot_train="61_000_000 + j·40_000 + f·1_000 + t", oc=ys.oc_seed,
                               precheck=ys.precheck_seed, compare=ys.compare_seed, small_boot=ys.small_boot_seed,
                               records=ys.records_seed, reconfirm=ys.reconfirm_seed, tags=tags),
                    pilot=dict(candidates=list(ys.pilot_w_pairs) + list(ys.pilot_v_pairs), flies=ys.pilot_flies,
                               probes=ys.pilot_probes, min_sigma=ys.pilot_min_sigma))

    def _reuse_all(self, where: str) -> tuple:
        """X's _reuse (W facts, θ̂ = W oc.theta) + W blocks reuse / path / pilot / oc + Y.7 1's X facts; (stop body or
        None, w_doc, pairs, θ_W, facts)."""
        why, w_doc, pairs, theta = self.ctx["x_reuse"]()
        facts = self.ctx["x_facts"](self.ys)
        why = list(why) + y_rules.w_block_reasons(facts, self.ys) + y_rules.x_reasons(facts, self.ys)
        return (y_rules.reuse_stop(why, where) if why else None), w_doc, pairs, theta, facts

    def _early(self, stage: str, body: dict, t0: float) -> dict:
        """An early STOP_REUSE still writes the stage's detail file, so the archive's fixed file set (P3-13,
        y_store.own_files) exists for every committed block."""
        det = dict(stage0=self.ys.stage0_detail, pilot=self.ys.pilot_detail, precheck=self.ys.precheck_detail)[stage]
        y_store.write_json(det, dict(body, note="재사용 조건 STOP — 계산 없음"), self.plist)
        return self._write(stage, body, time.perf_counter() - t0)

    # ================================================================ order 0: stage0 (Y.6.6, Y.3.4)
    def stage_stage0(self) -> dict:
        doc = self._require("stage0")
        ys, t0 = self.ys, time.perf_counter()
        stop, w_doc, _pairs, theta, facts = self._reuse_all("1(순서 0에서 확인)")
        if stop:
            return self._early("stage0", stop, t0)
        z = x_runner.Runner._z(w_doc)
        xm = facts["x_doc"]["precheck"]["calibration"]["min"]
        a, b = float(xm["a"]), float(xm["b"])
        cell = self._cells("stage0")
        syn = cell("synthetic", dict(z=z), lambda: y_oc.synthetic_validation(ys, z))
        fx = dict(calibration=cell("fixtures_cal", {}, lambda: y_oc.calibration_fixtures(ys)),
                  generator=cell("fixtures_gen", dict(z=z), lambda: y_oc.generator_fixtures(ys, z)),
                  theta=cell("fixtures_theta", {}, lambda: y_oc.theta_fixtures(ys)))
        th_key = w_oc.summary(theta)
        bit = cell("bit_identity", dict(theta=th_key, z=z, a=a, b=b),
                   lambda: y_oc.bit_identity(theta, z, ys, X_SPEC, ys.p26_reps, a, b))
        thr = y_oc.thresholds(theta, a, b, ys)
        oc_doc, _sha = self.ctx["w_oc_detail"]()
        boot = oc_doc["boot_calibration"]
        f6 = cell("fixture6", dict(theta=th_key, z=z, boot=hashlib.sha256(canonical(boot).encode()).hexdigest()),
                  lambda: y_oc.w_failed_recal(theta, boot, z, W_SPEC, ys, log=_log))
        vals = {k: (p["naive_d"], p["naive_median"]["MBON13_X"], p["naive_median"]["MBON05_X"])
                for k, p in w_doc["pilot"]["record"]["pairs"].items()}
        cmp_ = y_oc.compare_filters(theta, vals, z, ys, cell=cell, log=_log)
        timing = y_oc.oc_timing(theta, z, ys, a, b)
        bad = (([] if syn.get("ok") else ["합성 검증(W.9.9 P2-11) 실패"])
               + [f"픽스처 {k} 실패" for k, v in fx.items() if not v.get("ok")]
               + ([] if bit["equal"] else ["비트 동일 시험 실패"]))
        dec = dict(outcome=y_rules.INVALID, reasons=bad) if bad else dict(outcome=y_rules.PASS, reasons=[])
        p = y_store.write_json(ys.stage0_detail, dict(synthetic=syn, fixtures=fx, bit_identity=bit, thresholds=thr,
                                                      fixture6=f6, compare=cmp_), self.plist)
        body = dict(dec, synthetic={k: (v.get("ok") if isinstance(v, dict) else v) for k, v in syn.items()},
                    fixtures={g: {k: v.get("ok") for k, v in f.items() if isinstance(v, dict)} for g, f in fx.items()},
                    bit_identity=bit, thresholds=thr, fixture6=dict(n_failed=f6["n_failed"], counts=f6["counts"]),
                    compare=y_oc.compare_summary(cmp_), oc_timing=timing, tables=self._tables(),
                    decision_files=x_runner.files_sha(Y_FILES), x_files=x_runner.files_sha(X_FILES),
                    w_files=x_runner.files_sha(W_FILES), tests=x_runner.files_sha(Y_TEST_FILES),
                    k_ranges=doc["oracle"].get("k_ranges"), z_V={k: list(v) for k, v in z.items()},
                    detail_path=ys.stage0_detail, detail_sha256=sha256_file(p),
                    note="Y.7 0(국면 A): Y.6.6 전부 + 거름 후보 비교(Y.3.4, 기록 전용) + OC 시간 + 결정 파일 해시 + 수치 "
                         "표. 변이는 pytest(tests 해시). 문턱 재현은 기록 전용(20 · 43 불변).")
        return self._write("stage0", body, self._finish_wall("stage0", t0))

    # ================================================================ order 1: reuse (X.5 1 + X blocks)
    def stage_reuse(self) -> dict:
        self._require("reuse")
        t0 = time.perf_counter()
        stop, *_ = self._reuse_all("1")
        body = stop or dict(outcome=y_rules.PASS, reasons=[])
        body = dict(body, note="Y.7 1: X.5 1 조건(X의 _reuse: W 블록 · W 측정 키 · w_* 불변 · 매니페스트 · oc.json · θ̂) + "
                               "X 블록 868771a · 4b81035, precheck_diag.json 33f83895…, x_* 불변.")
        return self._write("reuse", body, time.perf_counter() - t0)

    # ================================================================ order 4: the pilot (Y.4, 4a)
    def _oracle_round_s(self, doc: dict) -> float:
        wall = sum(e["wall_s"] for e in (doc.get("budget") or {}).get("ledger", []) if e.get("stage") == "oracle")
        return float(wall) / max(1, -(-int(doc["oracle"].get("n", 0)) // max(1, int(self.ys.workers))))

    def _oracle_values(self, res: dict, z: dict) -> dict:
        st = pair_stats(res["report"], z, V_SPEC.testable_min)
        la, lp = y_rules.levels(res["report"])
        return dict(value=st is not None, testable=bool(st and st["testable"]), d_pre=st and st["d_pre"], L_A=la,
                    L_P=lp)

    def stage_pilot(self) -> dict:
        doc = self._require("pilot")
        ys, t0 = self.ys, time.perf_counter()
        stop, w_doc, wpairs, theta_w, _f = self._reuse_all("1(순서 4에서 확인)")
        vc = None
        if stop is None:
            try:
                vc = self.ctx["v_candidates"](ys)
            except ValueError as e:
                stop = y_rules.reuse_stop([f"V 세트 후보: {e}"], "1(순서 4에서 확인)")
        if stop:
            return self._early("pilot", stop, t0)
        z = x_runner.Runner._z(w_doc)
        rows = [r for r, _res in vc]
        def tick(done, todo):
            self._prog_add("pilot", time.perf_counter() - tick.t)
            tick.t = time.perf_counter()
        tick.t = t0                                       # wall time from stage entry (context builds included)
        m = self.ctx["pilot_measure"](rows, self.pool, tick)
        cands = {k: wpairs[k] for k in ys.pilot_w_pairs}
        cands.update({k: m["pairs"][k] for k in ys.pilot_v_pairs})
        adm = y_rules.admission({k: d["pre"] for k, d in cands.items()}, z, ys)
        keys = [k for k in cands if adm[k]["passed"]]
        jobs = m["jobs"]
        walls = dict(job_s_median=float(np.median([j["wall_s"] for j in jobs])),
                     train_s_median=float(np.median([j["train_s"] for j in jobs])),
                     probe_s_median=float(np.median([j["probe_s"] for j in jobs])))
        rec_all = w_records.pilot_record(cands, z, walls, W_SPEC)
        rec_adm = w_records.pilot_record({k: cands[k] for k in keys}, z, walls, W_SPEC) if keys else None
        dec = y_rules.pilot_gates(adm, rec_adm, m["machine"], ys, W_SPEC)
        extra = {}
        if dec["outcome"] == y_rules.PASS:
            th = y_oc.fit_y([cands[k] for k in keys], keys, y_oc.r_v0(theta_w))
            ex = w_verdict.judge({k: (cands[k], None) for k in keys}, [], z, W_SPEC.exploratory_q, ys.pilot_flies,
                                 ys.pilot_probes, W_SPEC)
            extra = dict(theta=json.loads(canonical(w_oc.summary(th))),
                         exploratory=dict(label="탐색", verdict=ex["verdict"],
                                          pairs={k: v["status"] for k, v in ex["pairs"].items()}))
        v_vals = {}
        for (r, res), k in zip(vc, ys.pilot_v_pairs):
            o = self._oracle_values(res, z)
            v_vals[k] = (o["d_pre"], o["L_A"], o["L_P"]) if o["value"] else None
        w_vals = {k: (p["naive_d"], p["naive_median"]["MBON13_X"], p["naive_median"]["MBON05_X"])
                  for k, p in w_doc["pilot"]["record"]["pairs"].items()}
        try:
            even = self.ctx["v_even_oracle"](list(w_vals), z)
            missing = [k for k in w_vals if k not in even]
            if missing:
                raise KeyError(f"V 짝수 블록 오라클에 {len(missing)}쌍 없음")
            flip = dict(y_rules.flip_record({k: self._oracle_values(even[k], z) for k in w_vals}, w_vals, ys),
                        available=True)
        except Exception as e:                            # record only: never crashes or steers the pilot
            flip = dict(available=False, reason=str(e), error=f"{type(e).__name__}: {e}")
        facts = dict(w_naive=y_rules.fact_flags(w_vals, dict(zip(ys.pilot_w_pairs, ys.pilot_w_naive)), ys),
                     v_oracle=y_rules.fact_flags({k: v for k, v in v_vals.items() if v},
                                                 dict(zip(ys.pilot_v_pairs, ys.pilot_v_oracle)), ys))
        costs = w_records.unit_costs(jobs, 2 * W_SPEC.trials, self._oracle_round_s(doc), W_SPEC.cost_workers())
        p = y_store.write_json(ys.pilot_detail, dict(admission=adm, record_all=rec_all, record_admitted=rec_adm,
                                                     v_oracle=v_vals, flip=flip, manifest=m["manifest"]), self.plist)
        body = dict(dec, **extra, admission=adm, candidates=list(cands), n_units=m["n_units"],
                    record={k: dict(naive_d=v["naive_d"], naive_median=v["naive_median"], median=v["median"],
                                    q_sat=v["q_sat"], mech=v["mech"]) for k, v in rec_all["pairs"].items()},
                    record_pooled={k: rec_all[k] for k in ("floor_both_share", "floor_taught_share",
                                                           "naive_floor_share", "walls")},
                    level_compare=y_rules.level_compare(doc["oracle"]["derived"]["lenient_levels"], adm, ys),
                    flip=flip, facts=facts, costs=costs,
                    precision=dict(pilot_probes_per_pair=ys.pilot_flies * ys.pilot_probes,
                                   main_probes_per_pair=[ys.f_min * min(ys.k_grid), ys.f_max * max(ys.k_grid)],
                                   note="Y.9.2 P3-15: 파일럿 입장은 쌍당 64 프로브, 주 세트 최종 거름은 F × K"),
                    z_V={k: list(v) for k, v in z.items()}, detail_path=ys.pilot_detail, detail_sha256=sha256_file(p),
                    note="Y.4: 파일럿은 주 세트 밖(W 균형 3 · V 세트 후보 4), 판정 코드는 탐색 라벨. θ는 거름 통과 쌍만, Σ는 "
                         "Earthquake 묶기 뒤 대각 + W V0 상관(Y.9.2 P1-3).")
        wall = self._prog_add("pilot", time.perf_counter() - tick.t)   # stop just before the block write
        return self._write("pilot", body, wall)

    # ================================================================ order 5: the point-θ precheck (Y.6.1)
    def stage_precheck(self) -> dict:
        doc = self._require("precheck")
        ys, t0 = self.ys, time.perf_counter()
        stop, w_doc, wpairs, theta_w, _f = self._reuse_all("1(순서 5에서 확인)")
        if stop:
            return self._early("precheck", stop, t0)
        z = x_runner.Runner._z(w_doc)
        pil = doc["pilot"]
        man = json.loads(Path(ys.pilot_detail).read_text())["manifest"]
        vpairs, bad = self.ctx["pilot_back"](man)
        if bad:
            refuse(f"pilot raw changed since block pilot: {bad[:3]}")
        cands = {k: wpairs[k] for k in ys.pilot_w_pairs}
        cands.update({k: vpairs[k] for k in ys.pilot_v_pairs})
        keys = list(pil["admitted"])
        r = y_oc.r_v0(theta_w)
        th = y_oc.fit_y([cands[k] for k in keys], keys, r)
        if json.loads(canonical(w_oc.summary(th))) != pil["theta"]:
            refuse("the refitted θ̂ is not block pilot's theta (Reading 21)")
        cell = self._cells("precheck")
        kr = [tuple(k) for k in doc["oracle"]["k_ranges"]]
        pc = y_oc.precheck_y(th, z, ys, kr, cell=cell, log=_log)
        dec = y_rules.precheck_decision(pc, len(keys), th["n_sigma"], ys)
        sb = cell("small_bootstrap", dict(theta=w_oc.summary(th), r=r.tolist()),
                  lambda: y_oc.small_bootstrap(th, z, ys, r, log=_log))
        thr = y_oc.threshold_recompute(th, pc["calibration"]["min"], ys)
        rec, notes = None, {}
        if dec["outcome"] != y_rules.PASS:
            thetas, notes = y_oc.record_thetas(th, theta_w, cands, keys, r, ys)
            rec = y_oc.point_records_y(thetas, pc["records_target"], z, ys, cell=cell, log=_log)
        p = y_store.write_json(ys.precheck_detail, dict(precheck=pc, records=rec, small_bootstrap=sb), self.plist)
        cal = {m: dict(ok=c["ok"], failure=c["failure"], corners=c["corners"], a=c["a"],
                       b=c["b"]) for m, c in pc["calibration"].items()}
        body = dict(dec, theta=pil["theta"], n_admitted=len(keys), n_sigma=th["n_sigma"], calibration=cal,
                    best_power=pc["best_power"], records_target=pc["records_target"], at_f32=pc["at_f32"],
                    n_passing=len(pc["passing"]), passing=pc["passing"][:ys.block_list_max],
                    fill_bad=pc["fill_bad"], stairs=pc["stairs"], m_needed=pc["m_needed"], k_ranges=pc["k_ranges"],
                    small_bootstrap=dict(counts=sb["counts"], n_draws=sb["n_draws"], seed=sb["seed"]),
                    thresholds=thr, records=rec, records_notes=notes,
                    records_note=None if rec is not None else "사전 점검 통과 — records는 국면 B의 OC(Y.6.5)",
                    env=env_hashes(), detail_path=ys.precheck_detail, detail_sha256=sha256_file(p),
                    timing=dict(precheck_s=pc["timing_s"]),
                    note="점 θ 사전 점검(Y.6.1): 점 추정 · 포락선 없음 — 통과해도 자격이 아니다. 시나리오 {기본, 근-문턱} × "
                         "군집 최악, 채움 > 1 % 제외(Y.9.2 P1-4 · P1-5), 0p 수율 규칙이 남긴 k 범위만(P1-7). 작은 "
                         "부트스트랩(P2-12)과 문턱 재계산(P2-10)은 기록 전용.")
        return self._write("precheck", body, self._finish_wall("precheck", t0))

    # ================================================================ phase B (Y.7 orders 6–12) — appended
    # ---- helpers ---------------------------------------------------------------------------------------------------
    def _keys_ok(self, stage: str) -> None:
        """Y.7 11 / W.8: the W and U measurement keys are the declared ones (before learning and before judging too);
        a change refuses with exit 7 and writes nothing."""
        k = self.ctx["keys"]()
        if (k.get("w_measure_key"), k.get("u_measure_key")) != (self.ys.w_measure_key, self.ys.u_measure_key):
            refuse(f"stage {stage}: the measurement keys {k} are not the declared ones (Y.2)", EXIT_KEY)

    def _ledger_h(self, doc: dict) -> float:
        return float(sum(e.get("wall_s", 0.0) for e in (doc.get("budget") or {}).get("ledger", []))) / self.ys.s_per_h

    def _z_b(self, doc: dict) -> dict:
        """Phase B's z: block pilot's z_V (W block reuse's full precision — the z of the pilot θ and the precheck,
        phase-A Reading 20)."""
        return {k: (float(v[0]), float(v[1])) for k, v in doc["pilot"]["z_V"].items()}

    def _early_b(self, stage: str, body: dict, t0: float, detail: str) -> dict:
        y_store.write_json(detail, dict(body, note="재사용 조건 STOP — 계산 없음"), self.plist)
        return self._write(stage, body, time.perf_counter() - t0)

    def _pcells(self, stage: str, jobs: list, log=None) -> list:
        """Resumable array cells computed in a spawn process pool (plan Reading 2): jobs = [(tag, key_obj, fn, args)]
        with fn a module-level y_oc function returning {arrays, meta}. A cell is progress_dir/<stage>/<tag>.npz holding
        its key; a different key recomputes. The parent writes each cell as it completes and adds the wall time since
        the previous completion to progress_dir/<stage>.json (so a killed run keeps its spend). Results do not depend on
        the worker count (every draw has its own streams). Returns [{arrays, meta}] in job order."""
        import io
        import os
        spec_sha = hashlib.sha256(canonical(self.ys).encode()).hexdigest()
        y_sha = x_runner.files_sha(Y_FILES)
        root = Path(self.ys.progress_dir) / stage

        def key_of(tag, key_obj):
            return hashlib.sha256(canonical(dict(tag=tag, key=key_obj, ys=spec_sha, y=y_sha)).encode()).hexdigest()

        def load(path, key):
            if not path.exists():
                return None
            try:
                with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as f:
                    if str(f["__key__"]) != key:
                        return None
                    meta = json.loads(str(f["__meta__"]))
                    return dict(arrays={k: f[k] for k in f.files if not k.startswith("__")}, meta=meta)
            except Exception:               # truncated / corrupt npz (BadZipFile, EOFError, …) → recompute the cell
                return None

        def save(path, key, res):
            buf = io.BytesIO()
            np.savez(buf, __key__=np.array(key), __meta__=np.array(json.dumps(y_store.to_json(res["meta"]))),
                     **res["arrays"])
            y_store.write_bytes(str(path), buf.getvalue(), self.plist)
        out, todo = [None] * len(jobs), []
        for i, (tag, key_obj, fn, args) in enumerate(jobs):
            p = root / (re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".npz")
            k = key_of(tag, key_obj)
            got = load(p, k)
            if got is None:
                todo.append((i, p, k, fn, args))
            else:
                out[i] = got
        if not hasattr(self, "_cell_s"):
            self._cell_s = 0.0
        workers = max(1, int(getattr(self, "oc_workers", None) or self.ys.workers))
        last = time.perf_counter()

        def done(i, p, k, res):
            nonlocal last
            save(p, k, res)
            out[i] = dict(arrays=res["arrays"], meta=json.loads(json.dumps(y_store.to_json(res["meta"]))))
            now = time.perf_counter()
            self._prog_add(stage, now - last)
            self._cell_s += now - last
            last = now
            if log is not None:
                log(f"y {stage}: {sum(o is not None for o in out)}/{len(out)} cells")
        if workers == 1 or len(todo) <= 1:
            for i, p, k, fn, args in todo:
                done(i, p, k, fn(*args))
            return out
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        for v in self.ys.thread_env:
            os.environ.setdefault(v, "1")
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=get_context("spawn")) as ex:
            try:
                fut = {ex.submit(fn, *args): (i, p, k) for i, p, k, fn, args in todo}
                for f in as_completed(fut):
                    i, p, k = fut[f]
                    done(i, p, k, f.result())
            except BaseException:           # an error or Ctrl-C does not run the queued draws
                ex.shutdown(wait=False, cancel_futures=True)
                raise
        return out

    def _theta_b(self, doc: dict, where: str):
        """Block pilot's θ̂ rebuilt from the raw (as the precheck, phase-A Reading 21) and the W pieces; (stop body or
        None, θ̂, θ_W, R_V0, candidates, admitted keys)."""
        stop, w_doc, wpairs, theta_w, _f = self._reuse_all(where)
        if stop:
            return stop, None, None, None, None, None
        man = json.loads(Path(self.ys.pilot_detail).read_text())["manifest"]
        vpairs, bad = self.ctx["pilot_back"](man)
        if bad:
            refuse(f"pilot raw changed since block pilot: {bad[:3]}")
        cands = {k: wpairs[k] for k in self.ys.pilot_w_pairs}
        cands.update({k: vpairs[k] for k in self.ys.pilot_v_pairs})
        keys = list(doc["pilot"]["admitted"])
        r = y_oc.r_v0(theta_w)
        th = y_oc.fit_y([cands[k] for k in keys], keys, r)
        if json.loads(canonical(w_oc.summary(th))) != doc["pilot"]["theta"]:
            refuse("the refitted θ̂ is not block pilot's theta (phase-A Reading 21)")
        return None, th, theta_w, r, cands, keys

    def _precheck_repro(self, doc: dict, th, z: dict) -> None:
        """Y.7 5: phase A's computation is unchanged — the precheck's point calibration on θ̂ (root precheck_seed)
        recomputed now equals block precheck's, bit for bit (as JSON), or the stage refuses."""
        ys = self.ys
        idx = y_oc.rng(ys.precheck_seed, y_oc.TAG_CAL).integers(0, len(th["resid"]), ys.cal_reps)
        now = {m: y_oc.calibrate_y(th, getattr(ys, f), m, idx, z, ys) for m, f in y_oc.MODES}
        want = doc["precheck"]["calibration"]
        got = {m: dict(ok=c["ok"], failure=c["failure"], corners=c["corners"], a=c["a"], b=c["b"])
               for m, c in now.items()}
        if y_store.to_json(got) != want:
            refuse("the precheck's calibration does not reproduce (Y.7 5: phase-A code changed?)")

    def _design_cost(self, doc: dict, d: dict, with_c: bool, costs: dict | None = None, k: int | None = None,
                     n_naive: int | None = None) -> dict:
        ys = self.ys
        c = costs or doc["pilot"]["costs"]
        n_len = int(doc["oracle"]["derived"]["yield_rule"]["n_len"])
        return y_rules.design_cost_y(c, d, ys, W_SPEC, n_len if n_naive is None else n_naive,
                                     int(d["k_range"][1]) if k is None else k, with_c)

    def _reconfirm(self, stage: str, th, z: dict, r, d: dict, rank: int, log=None) -> dict:
        jobs = [(f"rc{rank}_{bi}", dict(theta=w_oc.summary(th), design=d, rank=rank, bi=bi, z=z,
                                         reps=self.ys.reconfirm_reps),
                 y_oc.reconfirm_draw, (th, z, self.ys, r, d, rank, bi)) for bi in range(self.ys.boot_draws)]
        draws = self._pcells(stage, jobs, log)
        return y_oc.reconfirm_decide(draws, d, rank, self.ys)

    # ================================================================ order 6: bootstrap OC → qualify → select → P2-8
    def stage_oc(self) -> dict:
        doc = self._require("oc")
        ys, t0 = self.ys, time.perf_counter()
        stop, th, theta_w, r, cands, keys = self._theta_b(doc, "1(순서 6에서 확인)")
        if stop:
            return self._early_b("oc", stop, t0, ys.oc_detail)
        z = self._z_b(doc)
        self._precheck_repro(doc, th, z)
        cell = self._cells("oc")            # one clock for every cell of the stage (boot, reconfirm, records)
        kr = [tuple(int(v) for v in k) for k in doc["oracle"]["k_ranges"]]
        jobs = [(f"boot_{bi}", dict(theta=w_oc.summary(th), bi=bi, z=z, reps=ys.boot_reps), y_oc.boot_draw,
                 (th, z, ys, r, bi)) for bi in range(ys.boot_draws)]
        lim = y_oc.limits(self._pcells("oc", jobs, _log), ys, kr)
        qual = y_oc.qualify_boot(lim, ys, kr)

        def cost(dsg):
            return self._design_cost(doc, dsg, False)["total_h"]

        def select():
            # elapsed is read once, at the first selection; a resumed run reuses that selection (its progress already
            # holds spend made after it — reconfirmation, records), so ranking and rank tags stay the same
            el = self._ledger_h(doc) + self._prog("oc") / ys.s_per_h
            sl = y_oc.select_y(qual, cost, el, ys)
            if sl["outcome"] == y_rules.STOP_BUDGET:
                rows = [dict(p_set=ys.p_set_grid[i[0]], q=ys.q_grid[i[1]], K=ys.k_grid[i[2]], F=ys.f_min + i[3],
                             k_range=list(k)) for k, ok in qual.items() for i in np.ndindex(ok.shape) if ok[i]]
                best = min(cost(x) for x in rows)
                sl = dict(sl, budget_text=y_rules.budget_text(el, best, ys))
            return dict(elapsed=el, sel=sl)
        got = cell("selection", dict(theta=w_oc.summary(th), z=z, k_ranges=kr, draws=ys.boot_draws, reps=ys.boot_reps,
                                     ledger=(doc.get("budget") or {}).get("ledger", []), costs=doc["pilot"]["costs"],
                                     n_len=int(doc["oracle"]["derived"]["yield_rule"]["n_len"])), select)
        elapsed, sel = float(got["elapsed"]), got["sel"]
        at_f = y_oc.at_f_boot(lim, ys, kr, ys.f_max)
        dec = y_rules.oc_decision(sel, at_f, lim["counts"], ys)
        recs, design = [], None
        if dec["outcome"] == y_rules.PASS:
            for rank, d in enumerate(sel["ranking"][:ys.reconfirm_max]):
                rc = self._reconfirm("oc", th, z, r, d, rank, _log)
                recs.append(rc)
                if rc["ok"]:
                    design = dict(d, rank=rank)
                    break
            if design is None:
                dec = y_rules.reconfirm_stop(recs, ys)
        target = design or y_oc.records_target_b(lim, ys, kr, cost)
        thetas, notes = y_oc.record_thetas(th, theta_w, cands, keys, r, ys)
        rec = y_oc.records_b(thetas, {k: target[k] for k in ("p_set", "q", "K", "F")}, z, ys,
                             cell=cell, log=_log)
        first = _first_values(lim, design, ys) if design else None
        det = dict(limits={k: (_nan_none(v) if isinstance(v, np.ndarray) else v) for k, v in lim.items()
                           if k not in ("sim", "success_only")},
                   sim={f"[{a}, {b}]": {k: v.tolist() for k, v in s.items()} for (a, b), s in lim["sim"].items()},
                   success_only={m: (None if v is None else v.tolist()) for m, v in lim["success_only"].items()},
                   qualified={f"[{a}, {b}]": v.tolist() for (a, b), v in qual.items()}, selection=sel,
                   reconfirm=recs, records=rec, at_f32=at_f)
        p = y_store.write_json(ys.oc_detail, det, self.plist)
        body = dict(dec, design=design, design_first=first, ranking=sel["ranking"], n_qualifying=sel["n_qualifying"],
                    n_in_budget=sel["n_in_budget"], reconfirm=recs, n_reconfirmed=len(recs),
                    first_selected=sel.get("selected"), records_target=target, records=rec, records_notes=notes,
                    calibration_counts=lim["counts"], fill_bad=lim["fill_bad"].tolist(), at_f32=at_f,
                    elapsed_at_selection_h=elapsed, costs=doc["pilot"]["costs"], k_ranges=[list(k) for k in kr],
                    n_len=int(doc["oracle"]["derived"]["yield_rule"]["n_len"]), theta=doc["pilot"]["theta"],
                    z_V={k: list(v) for k, v in z.items()}, env=env_hashes(),
                    env_precheck_diff=_env_diff(doc["precheck"].get("env") or {}, env_hashes()),
                    detail_path=ys.oc_detail, detail_sha256=sha256_file(p),
                    boot=dict(draws=ys.boot_draws, reps=ys.boot_reps, root=ys.oc_seed),
                    note="Y.6.4 부트스트랩(200 × 400, 동시 단측 한계, 군집 · 시나리오 최악, 채움 > 1 % 제외) → Y.5 자격 · "
                         "선택(예산 ×1.3) → Y.9.2 P2-8 재확인(200 × 1600, 최대 5개) → Y.6.5 records(결과와 무관). "
                         "재확인 값이 공식 OC 값이고 처음 값은 함께 공개한다.")
        return self._write("oc", body, self._finish_wall("oc", t0))

    # ================================================================ order 7: smoke (Y.7 7)
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        self._keys_ok("smoke")
        ys, t0 = self.ys, time.perf_counter()
        d = doc["oc"]["design"]
        K = int(d["K"])
        row = self.ctx["v_candidates"](ys)[ys.smoke_pair][0]
        wr, wm = self.ctx["w_runner"](self.pool, True)
        lv = V_SPEC.lever_edit
        flies = list(range(ys.smoke_flies))
        learn = wm.learn(wr.units([row], "smoke", K, ys.smoke_flies, lv), "smoke")
        band = wm.learn(wr.units([row], "smoke", 2 * K, ys.smoke_flies, lv, k0=K), "smoke_band")
        naive = wm.learn(wr.units([row], "smoke", K, ys.smoke_flies, lv, brains=("naive",)), "smoke_naive")
        m = self.ctx["smoke_oracle"](self.pool)
        t1 = time.perf_counter()
        orc = m.oracle([row], V_SPEC.cond("L"), "smoke", wr.spec.oracle_seeds())
        oracle_s = time.perf_counter() - t1
        problems = smoke_problems(wr, learn, band, naive, orc, m, flies, K, ys)
        costs = w_records.unit_costs([g["result"] for g in learn], 2 * W_SPEC.trials, oracle_s, ys.workers)
        man = [dict(key=f"{g['unit']['pair']}|{g['unit']['fly']}|{g['unit']['brain']}|{blk}",
                    cache_file=g["cache_file"], cache_key=g["cache_key"], sha256=sha256_file(g["cache_file"]))
               for blk, got in (("smoke", learn), ("smoke_band", band), ("smoke_naive", naive)) for g in got]
        man += [dict(key=f"{g['key']}|oracle", cache_file=g["cache_file"], cache_key=g["cache_key"],
                     sha256=sha256_file(g["cache_file"])) for g in orc]
        p = y_store.write_json(ys.smoke_detail, dict(manifest=man, problems=problems), self.plist)
        body = dict(outcome=y_rules.INVALID if problems else y_rules.PASS, reasons=problems, design=d, costs=costs,
                    costs_used=y_rules.costs_max(doc["pilot"]["costs"], costs, ys), oracle_wall_s=oracle_s,
                    pair=row_key(row), seeds=dict(probe={f: wr.spec.smoke_probe_seeds(f, K) for f in flies},
                                                  train=[wr.spec.smoke_train_base(0), wr.spec.smoke_train_base(1)],
                                                  oracle=wr.spec.oracle_seeds()),
                    detail_path=ys.smoke_detail, detail_sha256=sha256_file(p),
                    note="Y.7 7: 스모크 시드(77_1xx_xxx), 파일럿 쌍 j 0, 마리 1 — 경로 · z_V · 시드 · sha 관계와 단위 비용.")
        return self._write("smoke", body, time.perf_counter() - t0)

    # ================================================================ order 7a: the worst-case budget gate
    def stage_budget_gate(self) -> dict:
        doc = self._require("budget_gate")
        ys, t0 = self.ys, time.perf_counter()
        self._cell_s = 0.0
        costs = doc["smoke"]["costs_used"]
        oc = doc["oc"]
        sel = oc["design"]
        failed = {json.dumps(r["design"], sort_keys=True) for r in oc["reconfirm"] if not r["ok"]}
        n_rc = int(oc["n_reconfirmed"])
        opts = [dict(design=sel, with_c=True), dict(design=sel, with_c=False)]
        opts += [dict(design=dict(a, rank=i), with_c=False) for i, a in enumerate(oc["ranking"])
                 if i > int(sel["rank"]) and json.dumps({k: a[k] for k in ("p_set", "q", "K", "F", "k_range")},
                                                        sort_keys=True) not in failed]
        th = z = r = None
        recs, tried, plan = [], [], None
        for o in opts:
            elapsed = self._ledger_h(doc) + self._prog("budget_gate") / ys.s_per_h
            c = self._design_cost(doc, o["design"], o["with_c"], costs)
            tried.append(dict(o, **c, elapsed_h=elapsed, in_budget=y_rules.in_budget(elapsed, c["total_h"], ys)))
            if not tried[-1]["in_budget"]:
                continue
            if o["design"] is not sel:
                if n_rc >= ys.reconfirm_max:
                    tried[-1]["skipped"] = "재확인 한도(Y.9.2 P2-8)"
                    continue
                if th is None:
                    stop, th, _tw, r, _c, _k = self._theta_b(doc, "1(순서 7a에서 확인)")
                    if stop:
                        return self._write("budget_gate", stop, self._finish_wall("budget_gate", t0))
                    z = self._z_b(doc)
                rc = self._reconfirm("budget_gate", th, z, r,
                                     {k: o["design"][k] for k in ("p_set", "q", "K", "F", "k_range")},
                                     int(o["design"]["rank"]), _log)
                n_rc += 1
                recs.append(rc)
                if not rc["ok"]:
                    tried[-1]["reconfirm"] = False
                    continue
                elapsed = self._ledger_h(doc) + self._prog("budget_gate") / ys.s_per_h
                if not y_rules.in_budget(elapsed, c["total_h"], ys):
                    tried[-1]["after_reconfirm"] = "예산 초과"
                    continue
            plan = dict(design=o["design"], with_c=o["with_c"], parts_h=c["parts_h"], total_h=c["total_h"])
            break
        if plan is None:
            dec = y_rules.budget_stop(self._ledger_h(doc) + self._prog("budget_gate") / ys.s_per_h,
                                      [dict(total_h=t["total_h"]) for t in tried], "순서 7a", ys)
        else:
            dec = dict(outcome=y_rules.PASS, reasons=[], plan=plan)
        body = dict(dec, options=tried, reconfirm=recs, n_reconfirmed=n_rc, costs=costs, margin=ys.cost_margin,
                    note="Y.7 7a: 누적 실측 + ×1.3 × (순진 거름 최악 + k_hi 학습 + BAND 2K + C + 가소성 끈 대조) ≤ 24 h — "
                         "C를 먼저 빼고, 그다음 대체 설계(재확인 통과 필요, 최대 5개), 그래도 넘으면 STOP_BUDGET.")
        return self._write("budget_gate", body, self._finish_wall("budget_gate", t0))

    # ================================================================ order 8: the judged-seed naive final filter
    def _lenient(self, doc: dict) -> list:
        """The oracle lenient pre-filter's passes (Y.3.3 1) from results/y/oracle.json — its sha256 must be block
        oracle's — in declared order (candidate number c: generator turn, (b) first)."""
        ys = self.ys
        p = Path(ys.oracle_detail)
        if not p.exists() or sha256_file(p) != doc["oracle"]["detail_sha256"]:
            refuse(f"{ys.oracle_detail} is missing or not block oracle's (sha256)")
        per = json.loads(p.read_text())["pairs"]
        out = sorted((x for x in per if y_rules.passes(x, ys)["y_lenient"]), key=lambda x: int(x["c"]))
        if len(out) != int(doc["oracle"]["derived"]["yield_rule"]["n_len"]):
            refuse("the lenient-pass count is not block oracle's N_len")
        return out

    def stage_gates(self) -> dict:
        doc = self._require("gates")
        self._keys_ok("gates")
        ys, t0 = self.ys, time.perf_counter()
        d = doc["budget_gate"]["plan"]["design"]
        K, F = int(d["K"]), int(d["F"])
        k_lo, k_hi = (int(v) for v in d["k_range"])
        cand = self._lenient(doc)
        try:
            rows = {r["c"]: r for r in self.ctx["main_rows"](doc["digest"]["set"])}
        except ValueError as e:
            refuse(f"the main set does not reproduce block digest: {e}")
        z = self._z_b(doc)
        wr, wm = self.ctx["w_runner"](self.pool, False)
        screened, gates, man = [], [], []
        i = 0
        n_w = max(1, int(getattr(self.pool, "n_workers", None) or ys.workers))
        last = time.perf_counter()
        while i < len(cand) and len(gates) < k_hi:
            batch = cand[i:i + max(1, min(k_hi - len(gates), n_w // F or 1))]
            units = [u for p in batch for u in wr.units([rows[p["c"]]], "main", K, F, V_SPEC.lever_edit,
                                                         brains=("naive",))]
            got = wm.learn(units, "naive")
            now = time.perf_counter()
            self._prog_add("gates", now - last)
            last = now
            by = wr._by_pair(got)
            for p in batch:
                g = sorted(by[p["key"]], key=lambda x: x["unit"]["fly"])
                pre = np.stack([w_records.counts(x["result"]["stages"][0]) for x in g])
                f = y_rules.final_filter(pre, z, ys)
                ok = bool(f["passed"]) and len(gates) < k_hi
                screened.append(dict(key=p["key"], c=p["c"], naive_d=f["d"], L_A=f["L_A"], L_P=f["L_P"], gate=ok))
                man += wr._manifest(g)
                if ok:
                    gates.append(dict(key=p["key"], c=p["c"]))
            i += len(batch)
        dec = y_rules.few_pairs_final(int(doc["oracle"]["n"]), len(cand), len(gates), k_lo, ys)
        p = y_store.write_json(ys.gates_detail, dict(screened=screened, gates=gates, manifest=man), self.plist)
        body = dict(dec, gates=gates, screened=screened, n_screened=len(screened), n_pre=len(cand), stopped_at=i,
                    design=d, manifest=man, z_V={k: list(v) for k, v in z.items()},
                    detail_path=ys.gates_detail, detail_sha256=sha256_file(p),
                    note="Y.3.3 2: 사전 거름 통과 쌍을 선언 순서로, 설계의 F × K pre(판정 프로브 시드)로 최종 거름 — k_hi에서 "
                         "멈춤. 순진 조건은 이 값으로 고정(BAND 2K로 다시 계산하지 않음). 주 세트는 아직 미사용(Y.0).")
        return self._write("gates", body, self._prog_add("gates", time.perf_counter() - last))

    # ================================================================ order 8a: the estimate on the real gate count
    def stage_estimate(self) -> dict:
        doc = self._require("estimate")
        ys = self.ys
        plan = doc["budget_gate"]["plan"]
        k = len(doc["gates"]["gates"])
        costs = doc["smoke"]["costs_used"]
        elapsed = self._ledger_h(doc)
        opts = ([True, False] if plan["with_c"] else [False])
        tried = []
        for wc in opts:
            c = self._design_cost(doc, plan["design"], wc, costs, k=k, n_naive=0)
            tried.append(dict(with_c=wc, **c, in_budget=y_rules.in_budget(elapsed, c["total_h"], ys)))
        ok = [t for t in tried if t["in_budget"]]
        if ok:
            dec = dict(outcome=y_rules.PASS, reasons=[], plan=dict(design=plan["design"], with_c=ok[0]["with_c"],
                                                                   parts_h=ok[0]["parts_h"], total_h=ok[0]["total_h"]))
        else:
            dec = y_rules.budget_stop(elapsed, tried, "순서 8a", ys)
        body = dict(dec, options=tried, n_gates=k, elapsed_h=elapsed, margin=ys.cost_margin,
                    note="Y.7 8a: 실제 관문 쌍 수로 갱신(×1.3) — C를 먼저 빼고, 그래도 넘으면 STOP_BUDGET(설계는 바꾸지 "
                         "않는다, Y.5).")
        return self._write("estimate", body, 0.0)

    # ================================================================ orders 9–10: learning, BAND 2K, records
    def _view(self, doc: dict) -> dict:
        """Y's blocks in the shape W's Runner reads (w_runner.Runner.learn_units / band_units / record_units / _want /
        _read): set, reuse z_V, budget / estimate plans, gates (design + pairs), naive (screened + manifest), the
        measuring blocks' manifests, oracle counts and oc's design."""
        d = doc["gates"]["design"]
        e = doc.get("estimate") or {}
        v = dict(set=dict(set=doc["digest"]["set"]), reuse=dict(z_V=doc["pilot"]["z_V"]),
                 budget=dict(plan=doc["budget_gate"]["plan"]), estimate=dict(plan=e.get("plan")),
                 gates=dict(design=dict(q=d["q"], K=d["K"], F=d["F"], k_cap=int(d["k_range"][1])),
                            gates=doc["gates"]["gates"]),
                 naive=dict(screened=doc["gates"]["screened"], manifest=doc["gates"]["manifest"]),
                 oracle=dict(n=doc["oracle"]["n"], n_testable=doc["oracle"]["derived"]["counts"]["all"]["testable"]),
                 oc=dict(selected=doc["oc"]["design"], records=None, drift_dprime=None))
        for s in ("learn", "band", "records"):
            if s in doc:
                v[s] = dict(manifest=doc[s]["manifest"])
        return v

    def _measure_b(self, stage: str, make_units, parts: tuple, after: tuple, detail: str) -> dict:
        """Y.7 9 · 10: the units through the W measurer behind YCache, with the batch ledger (measured spend, no
        margin — Y.9.2 P2-9): elapsed + spent + this stage's remaining share + the later parts > budget_h →
        STOP_BUDGET (the main set is used from learn on). Blocks hold manifests only — no statistic before the seal."""
        doc = self._require(stage)
        self._keys_ok(stage)
        ys = self.ys
        wr, wm = self.ctx["w_runner"](self.pool, False)
        view = self._view(doc)
        units = make_units(wr, view)
        est = doc["estimate"]["plan"]
        later = sum(est["parts_h"][p] for p in after if p != "c" or est["with_c"])
        share = sum(est["parts_h"][p] for p in parts if p != "c" or est["with_c"])
        base = self._ledger_h(doc)
        prev = self._prog(stage)
        t_start = time.perf_counter()

        def check(done, todo):
            now = time.perf_counter()
            self._prog_add(stage, now - check.t)
            check.t = now
            spent = (prev + now - t_start) / ys.s_per_h
            left = share * (1 - done / max(todo, 1)) if done else share
            if round(base + spent + left + later - ys.budget_h, ys.round_digits) > 0:
                raise w_runner.BudgetStop(w_rules.budget_h_text(base + spent, left + later))
        check.t = time.perf_counter()
        try:
            got = wm.learn(units, stage, check=check)
        except w_runner.BudgetStop as e:
            body = dict(outcome=y_rules.STOP_BUDGET, reasons=[str(e)], main_set_used=True,
                        sentence=y_rules.sentences_b(ys)[y_rules.STOP_BUDGET].format(h=str(e), where=WHERE[stage]))
            y_store.write_json(detail, dict(body, manifest=[]), self.plist)
            return self._write(stage, body, self._prog_add(stage, time.perf_counter() - check.t))
        man = wr._manifest(got)
        p = y_store.write_json(detail, dict(manifest=man), self.plist)
        body = dict(outcome=y_rules.PASS, reasons=[], manifest=man, n_units=len(units), main_set_used=True,
                    detail_path=detail, detail_sha256=sha256_file(p),
                    note="블록은 통계를 담지 않는다 — 판정은 봉인 뒤 1회(Y.7 12).")
        return self._write(stage, body, self._prog_add(stage, time.perf_counter() - check.t))

    def stage_learn(self) -> dict:
        return self._measure_b("learn", lambda wr, v: wr.learn_units(v), ("learn",), ("band", "c", "noplast"),
                               self.ys.learn_detail)

    def stage_band(self) -> dict:
        return self._measure_b("band", lambda wr, v: wr.band_units(v), ("band",), ("c", "noplast"),
                               self.ys.band_detail)

    def stage_records(self) -> dict:
        return self._measure_b("records", lambda wr, v: wr.record_units(v), ("c", "noplast"), (),
                               self.ys.records_detail)


# ================================================================ phase B module level (Y.7 orders 6–12) — appended
EXIT_KEY = 7
PHASE_B = ("oc", "smoke", "budget_gate", "gates", "estimate", "learn", "band", "records", "seal", "judge")
ORDER = ORDER + PHASE_B
GATES = ORDER


def y_w_spec(ys, smoke: bool = False):
    """W's spec with Y's seed roots (Y.2): main probe 62M / training 64M, pilot 60M / 61M, smoke 77_100_000 /
    77_110_000 / 77_150_000; k_cap = Y's largest k_hi; W's every other number (protocol, verdict, noplast)."""
    import dataclasses
    return dataclasses.replace(W_SPEC, probe_seed0=ys.probe_seed0, train_seed0=ys.train_seed0,
                               pilot_probe_seed0=ys.pilot_probe_seed0, pilot_train_seed0=ys.pilot_train_seed0,
                               smoke_probe_seed0=ys.smoke_probe_seed0, smoke_train_seed0=ys.smoke_train_seed0,
                               smoke_oracle_seed0=ys.smoke_oracle_seed0, k_cap=ys.k_cap, workers=ys.workers,
                               smoke=bool(smoke))


def _nan_none(a) -> list:
    """An array as JSON lists with NaN (a cell no draw simulated) as null."""
    a = np.asarray(a, float)
    return np.where(np.isnan(a), None, a).tolist()


def _first_values(lim: dict, d: dict, ys) -> dict:
    """Y.9.2 P2-8 ("처음 값은 함께 공개"): the design's bootstrap limits — k-wise and simultaneous — beside the
    reconfirmed ones."""
    i = (ys.p_set_grid.index(d["p_set"]), ys.q_grid.index(d["q"]), ys.k_grid.index(d["K"]), int(d["F"]) - ys.f_min)
    kr = tuple(int(v) for v in d["k_range"])
    ks = [k - ys.k_min for k in range(kr[0], kr[1] + 1)]
    return dict(k=list(range(kr[0], kr[1] + 1)), power_by_k=lim["power_lo_by_k"][i][ks].tolist(),
                false_by_k=lim["false_hi_by_k"][i][ks].tolist(), power_sim=float(lim["sim"][kr]["power"][i]),
                false_sim=float(lim["sim"][kr]["false"][i]))


def _env_diff(a: dict, b: dict) -> list:
    """The env fields (uv.lock, Python, numpy, and each file's sha256) that differ between two env_hashes records."""
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) or isinstance(vb, dict):
            va, vb = va or {}, vb or {}
            out += [f"{k}:{f}" for f in sorted(set(va) | set(vb)) if va.get(f) != vb.get(f)]
        elif va != vb:
            out.append(k)
    return out


def _phase_b_ctx(wctx, npz: str) -> dict:
    """build_ctx's phase-B entries: w_runner(pool, smoke) = (W's Runner over Y's seed spec on W's context, a
    WMeasurer behind YCache — results/y/cache, or results/y/smoke/cache with Y's smoke seed set); smoke_oracle(pool) =
    an RMeasurer over the smoke cache with 0p's z (W_SPEC.z_v())."""
    from .r_measure import RMeasurer
    from .u_measure import UPool
    from .w_measure import WMeasurer, w_measure_key
    from .y_spec import SPEC as YS

    def w_run(pool, smoke=False):
        c = wctx()
        yw = y_w_spec(YS, smoke)
        cache = (y_store.YCache(YS.smoke_cache_dir, w_measure_key(npz), smoke_seeds=yw.smoke_seed_set()) if smoke
                 else y_store.YCache(YS.cache_dir, w_measure_key(npz)))
        windows = dict(strength=W_SPEC.strength, settle_ms=V_SPEC.settle_ms, read_ms=V_SPEC.read_ms,
                       window_ms=V_SPEC.window_ms)
        timing = dict(present_ms=W_SPEC.pulse_ms, gap_ms=W_SPEC.gap_ms, train_settle_ms=W_SPEC.train_settle_ms,
                      seed_stride=W_SPEC.fly_train_stride)
        wm = WMeasurer(pool, cache, c["params"], c["readout"], V_SPEC.p_type, W_SPEC.reward_dan, W_SPEC.punish_dan,
                       windows, timing)
        return w_runner.Runner(None, lambda smoke_=False: wm, c, yw), wm

    def smoke_oracle(pool):
        c = wctx()
        yw = y_w_spec(YS, True)
        return RMeasurer(UPool(pool), y_store.YCache(YS.smoke_cache_dir, w_measure_key(npz),
                                                     smoke_seeds=yw.smoke_seed_set()),
                         V_SPEC, c["params"], c["readout"], W_SPEC.z_v(), c["types"], c["n_kc"])
    return dict(w_runner=w_run, smoke_oracle=smoke_oracle)


def smoke_problems(wr, learn, band, naive, orc, m, flies, K, ys) -> list:
    """Y.7 7's checks (W's smoke checks with Y's seeds): machine reasons (pre equal across brains and with the naive
    job, band weights), RN1 = R1, the oracle's lever (edit / CSC / edges), the oracle's z = W_SPEC.z_v() (0p's), and
    every probe / training / oracle seed inside the smoke block 77_100_000–77_199_999."""
    lv = V_SPEC.lever_edit
    seeds = {f: wr.spec.smoke_probe_seeds(f, K) for f in flies}
    nv = {g["unit"]["fly"]: w_records.counts(g["result"]["stages"][0]) for g in naive}
    out = list(w_records.machine_reasons(learn, flies, wr._declared(lv, seeds), naive=nv, band=band))
    d_k = w_records.pair_data(learn, flies)
    if bool(w_verdict.rn1_mismatch({s: np.asarray(v)[None] for s, v in d_k.items()})[0]):
        out.append("RN1 ≠ R1")
    q = orc[0]["result"].get("q", {})
    if (q.get("edit"), q.get("csc_sha256"), q.get("edit_edges")) != (lv, V_SPEC.sha_combined, V_SPEC.lever_edges):
        out.append(f"oracle edit {q.get('edit')} / CSC {q.get('csc_sha256')} / edges {q.get('edit_edges')}")
    if {k: tuple(v) for k, v in m.z.items()} != {k: tuple(v) for k, v in W_SPEC.z_v().items()}:
        out.append(f"oracle z {m.z} ≠ z_V {W_SPEC.z_v()}")
    lo, hi = ys.smoke_block
    used = [s for g in learn + band + naive for s in g["unit"]["probe_seeds"]]
    used += [wr.spec.smoke_train_base(0), wr.spec.smoke_train_base(1)]
    used += [s for v in wr.spec.oracle_seeds().values() for s in v]
    out += [f"스모크 시드 {s}가 스모크 블록 밖" for s in used if not lo <= s < hi]
    return out


WHERE = dict(learn="순서 9 학습 측정", band="순서 9 BAND 2K 재측정", records="순서 10 기록")

from . import w_rules  # noqa: E402 — phase B (the budget text of the batch ledger)
