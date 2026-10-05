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
- Archive (Y red-team P3-13, y_store.archive_stage): called just before a block that will be committed is written —
  PASS or a STOP — never for INVALID (a machine defect the controller discards; archiving it would block the fixed
  rerun, since an archive is never overwritten). A refused archive therefore leaves no block.
Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block or its own block
exists, an earlier gate did not PASS, the summary has uncommitted changes, or a hashed Y / W / V file is dirty."""
from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

from ..agent.e_runner import summary_git
from . import w_runner, y_rules, y_store
from .h3_store import git_state as _h3_git_state
from .h3_store import sha256_file
from .h4_formula import pair_stats
from .v_spec import SPEC as V_SPEC
from .w_pairs import set_summary
from .w_spec import SPEC as W_SPEC

ORDER = ("digest", "oracle")
GATES = ("digest", "oracle")
Y_FILES = ("flymon/brain/y_spec.py", "flymon/brain/y_rules.py", "flymon/brain/y_store.py", "flymon/brain/y_runner.py",
           "scripts/run_y.py")
W_FILES = tuple(dict.fromkeys(tuple(w_runner.W_PIPELINE_FILES) + ("flymon/brain/w_measure.py",)))
Y_HASHED_FILES = tuple(dict.fromkeys(Y_FILES + W_FILES + tuple(w_runner.V_HASHED_FILES)
                                     + ("results/summary/v_lever.json", "results/summary/w_learning.json")))


def git_state() -> dict:
    return _h3_git_state(files=Y_HASHED_FILES)


def refuse(msg: str, code: int = 2):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


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
    return dict(keys=lambda: dict(w_measure_key=w_measure_key(npz)["key"], u_measure_key=u_measure_key(npz)["key"]),
                reuse=reuse, w_set=lambda: wctx()["w_set"](), main_rows=lambda blk: wctx()["main_rows"](blk),
                params=lambda: wctx()["params"], measurer=measurer)


class Runner:
    def __init__(self, ctx: dict, ys, measure=None, summary_path=None):
        self.ctx, self.ys, self.measure = ctx, ys, measure
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

    def _write(self, stage: str, body: dict, wall_s: float, files=()) -> dict:
        """Archive (PASS / STOP only, before the block so a refused archive leaves no block), then the block."""
        k = self.ctx["keys"]()
        extra = {}
        if body.get("outcome") != y_rules.INVALID:
            extra["archive"] = y_store.archive_stage(stage, list(files), self.ys)
        block = y_store.to_json(dict(body, **extra, stage=stage, w_measure_key=k.get("w_measure_key"),
                                     u_measure_key=k.get("u_measure_key"), git=git_state(), written_at=_now()))
        y_store.write_summary_block(self.summary_path, stage, block, self.plist,
                                    dict(stage=stage, wall_s=float(wall_s), at=block["written_at"]))
        return block

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
        m = self.measure()
        t0 = time.perf_counter()
        try:
            got = m.oracle(rows, V_SPEC.cond("L"), "screen", seeds)
        finally:
            wall = self._prog_add("oracle", time.perf_counter() - t0)
        want_q = (V_SPEC.lever_edit, V_SPEC.sha_combined, V_SPEC.lever_edges)
        per, man = [], []
        for r, g in zip(rows, got, strict=True):
            res = g["result"]
            q = res.get("q", {})
            st = pair_stats(res["report"], z, V_SPEC.testable_min)
            la, lp = y_rules.levels(res["report"])
            per.append(dict(key=g["key"], c=r["c"], axis=r["axis"], turn=r["turn"], value=st is not None,
                            failure=(q.get("edit"), q.get("csc_sha256"), q.get("edit_edges")) != want_q,
                            testable=bool(st and st["testable"]), d_pre=st and st["d_pre"], r=st and st["r"],
                            p=st and st["p"], L_A=la, L_P=lp))
            man.append(dict(key=g["key"], cache_file=g["cache_file"], cache_key=g["cache_key"],
                            sha256=sha256_file(g["cache_file"])))
        p = y_store.write_json(ys.oracle_detail, dict(pairs=per, manifest=man, seeds=seeds,
                                                      z_V={k: list(v) for k, v in z.items()}), self.plist)
        # the count table from the records as written (Y red-team P2-11: recomputable from oracle.json alone)
        table = y_rules.count_table(json.loads(Path(p).read_text())["pairs"], ys)
        dec = y_rules.oracle_decision(table, ys)
        if "stage" in dec:                   # the STOP's stage ("early") would collide with the block's stage name
            dec["stop_stage"] = dec.pop("stage")
        body = dict(dec, counts=table, n=len(per), detail_path=ys.oracle_detail, detail_sha256=sha256_file(p),
                    seeds=seeds, z_V={k: list(v) for k, v in z.items()}, condition="L", block="screen",
                    note="Y.7 0p-c: 개수 표만(쌍별 값은 git 제외 상세 파일, 글쓴이 해석 6). 오라클 · 순진 pre만으로는 "
                         "주 세트를 사용한 것이 아니다(Y.0).")
        return self._write("oracle", body, wall)
