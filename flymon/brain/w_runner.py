"""Spec W's stage chain (W.9.9's order; plan Readings):
stage0 -> reuse -> path -> pilot -> oc -> smoke -> budget -> set -> oracle -> estimate -> naive -> gates -> learn ->
band -> records -> seal -> judge (and after judge only: recompute / invalid_run), one block each in
results/summary/w_learning.json, written only through w_store, plus a running budget ledger (W.9.6 F). Every stage
refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists, its own block exists,
an earlier gate did not pass, the summary has uncommitted changes or a hashed W file is dirty. Every stage after
`reuse` re-checks the reuse condition (V's blocks, R / T / U keys) and refuses with SystemExit 7 when it broke; learn,
seal and judge also re-check that every earlier block carries the current keys (W.8, exit 7).
The main set is used from block oracle on (W.9.9: 순서 7); every stage before it runs on V's raw data, the pilot pairs
or synthetic data only. learn / band / records hold raw manifests and no statistic; the verdict is read once at judge
(the BAND re-measure is measured for every gate pair before the seal, plan Reading 13)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import v_rules, w_oc, w_records, w_rules, w_store, w_verdict
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .h4_formula import pair_stats  # noqa: F401  (stage oracle, w_runner part 2)
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import p_items
from .t_measure import T_MEASURE_FILES
from .u_measure import U_MEASURE_FILES
from .v_runner import V_HASHED_FILES, kc_values
from .v_spec import SPEC as V_SPEC
from .w_measure import W_MEASURE_FILES
from .w_spec import BRAINS

ORDER = ("stage0", "reuse", "path", "pilot", "oc", "smoke", "budget", "set", "oracle", "estimate", "naive", "gates",
         "learn", "band", "records", "seal", "judge")
GATES = ("stage0", "reuse", "path", "pilot", "oc", "budget", "set", "oracle", "estimate", "naive", "gates", "learn",
         "band")
EXIT_REFUSE, EXIT_KEY = 2, 7
W_PIPELINE_FILES = ("flymon/brain/w_spec.py", "flymon/brain/w_pairs.py", "flymon/brain/w_store.py",
                    "flymon/brain/w_verdict.py", "flymon/brain/w_oc.py", "flymon/brain/w_records.py",
                    "flymon/brain/w_rules.py", "flymon/brain/w_runner.py", "scripts/run_w.py")
W_HASHED_FILES = tuple(dict.fromkeys(V_HASHED_FILES + W_MEASURE_FILES + W_PIPELINE_FILES
                                     + ("results/summary/v_lever.json",)))
DECISION_FILES = tuple(f for f in W_HASHED_FILES if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES
                       and f not in U_MEASURE_FILES and f not in W_MEASURE_FILES)
JUDGE_MARKER = "results/w/judge_read.json"
REREAD_MARKER = "results/w/judge_reread.json"
DONE_MARKER = "results/w/judge_done.json"
PINNED = ("oc", "budget", "gates")


def git_state() -> dict:
    return _h3_git_state(files=W_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(W_PIPELINE_FILES)


def decision_key() -> dict:
    return _files_key(DECISION_FILES)


def _sha(obj) -> str | None:
    return None if obj is None else hashlib.sha256(canonical(obj).encode()).hexdigest()


def decision_pins(doc: dict) -> dict:
    """W.3 10: the design (block oc's selection and block budget's plan), the gate pairs and z_V."""
    pins = {f"{b}_sha256": _sha(doc.get(b)) for b in PINNED}
    pins["z_V"] = (doc.get("reuse") or {}).get("z_V")
    return pins


def refuse(msg: str, code: int = EXIT_REFUSE):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _z(z: dict) -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in z.items()}


class BudgetStop(Exception):
    """Raised between rounds when the ledger's estimate passes the budget (W.9.8 H3)."""


def git_facts(path: str, commits) -> dict:
    """The last commit touching V's summary and which declared V commits are in HEAD's history."""
    def run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True)
    last = run("log", "-1", "--format=%H", "--", path).stdout.strip() or None
    anc = {c: run("merge-base", "--is-ancestor", c, "HEAD").returncode == 0 for c in commits}
    return dict(last=last, ancestors=anc)


def build_ctx(spec, npz: str) -> dict:
    """V's context (v_runner.build_ctx on V's spec: C3, block h4's z, the even rows, P's stimuli, R's / T's / U's
    summaries, …) plus W's: V's summary and git facts, V's KC values, the pilot rows, the main set, V's gate-② arm
    rows (read only, by content key) and V's judgement raw (by V's manifest)."""
    from ..agent.config import load_c3_config
    from . import w_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .r_measure import RMeasurer
    from .u_measure import u_measure_key
    from .v_runner import build_ctx as v_build_ctx
    from .v_store import load_manifest
    ctx = v_build_ctx(V_SPEC, npz)
    cfg = load_c3_config(V_SPEC.m0d_summary)
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    enc, params = ctx["enc"], cfg.params
    v_doc = lambda: json.loads(Path(spec.v_summary).read_text())  # noqa: E731
    vread = w_store.VReadCache(V_SPEC.cache_dir, u_measure_key(npz))
    v_m = RMeasurer(None, vread, V_SPEC, params, cfg.readout, cfg.z, ctx["types"], len(pops.kc))

    def v_gate2_rows(items):
        return v_m.arms(items, ctx["readout"], V_SPEC.p.o.n.h3.punish_type, "gate2", V_SPEC.p.o.n)

    def v_jm(name: str, n: int):
        d = v_doc()
        rows = ctx["judgement_rows"](kc_values(d), d["set"]["set"])[:n]
        got, bad = load_manifest(d[f"jm:{name}"]["manifest"])
        if bad:
            raise ValueError("; ".join(bad[:3]))
        res = {g["key"]: g["result"] for g in got}
        return [(r, res[row_key(r)]) for r in rows]

    ctx.update(
        v_doc=v_doc, v_git=lambda: summary_git(spec.v_summary),
        v_facts=lambda: git_facts(spec.v_summary, [c for _, c in spec.v_commits]),
        pilot_rows=lambda: w_pairs.pilot_rows(ctx["even_rows"], v_doc()["even"]["pairs"], spec),
        w_set=lambda: w_pairs.w_set(pops, enc, params, kc_values(v_doc()), v_doc()["set"]["set"], spec),
        main_rows=lambda blk: w_pairs.main_rows(pops, rc, enc, params, kc_values(v_doc()), v_doc()["set"]["set"],
                                                blk, spec),
        v_gate2_rows=v_gate2_rows, v_jm=v_jm, clusters=w_pairs.cluster_labels,
        c3_params=dict(learn_rate=params.learn_rate, kc_trace_ms=params.kc_trace_ms, da_trace_ms=params.da_trace_ms,
                       da_baseline_ms=params.da_baseline_ms, kc_trace_scale=params.kc_trace_scale,
                       da_trace_scale=params.da_trace_scale, min_weight_frac=params.min_weight_frac,
                       recovery_per_pulse=params.recovery_per_pulse, kc_kc_scale=params.kc_kc_scale,
                       dan_drive_mv=params.dan_drive_mv, core_frac=params.core_frac))
    return ctx


class Runner:
    def __init__(self, measure, learner, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, ucode: dict | None = None, wcode: dict | None = None,
                 pipeline: dict | None = None, archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) over U's job copies behind W's cache; learner -> a
        WMeasurer (or its interface). ucode = U's measurement key (= V's); wcode = the W measurement key."""
        self.measure, self.wm, self.ctx, self.spec = measure, learner, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.u_measure_key = (ucode or {}).get("key")
        self.w_measure_key = (wcode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]
        self._ms = {}

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return w_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed W files are dirty: {gs['dirty_hashed']}")

    def _reuse_dec(self) -> dict:
        c, sp = self.ctx, self.spec
        v_dec = v_rules.reuse(c["r_doc"](), c["r_git"](), c["t_doc"](), c["t_git"](), c["u_doc"](), c["u_git"](),
                              self.code_key, self.t_measure_key, self.u_measure_key, V_SPEC)
        f = c["v_facts"]()
        return w_rules.reuse(v_dec, c["v_doc"](), c["v_git"](), f["ancestors"], f["last"], self.u_measure_key,
                             self.code_key, self.t_measure_key, sp)

    def _require(self, stage: str, allow_own: bool = False) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; W never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; W never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != w_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — W stops there (W.3)")
        if i > ORDER.index("reuse"):
            r = self._reuse_dec()
            if r["outcome"] != w_rules.PASS:
                refuse(f"stage {stage}: the reuse condition broke ({'; '.join(r['reasons'])})", EXIT_KEY)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _keys_chain(self, doc: dict, stage: str) -> None:
        """W.8: the W measurement key (and R's, T's, U's) every earlier block carries is the current one (exit 7)."""
        if not (self.code_key and self.t_measure_key and self.u_measure_key and self.w_measure_key):
            refuse("a measurement key cannot be verified (W.8)", EXIT_KEY)
        for b in ORDER[:ORDER.index(stage)]:
            got = tuple(doc[b].get(k) for k in ("code_key", "t_measure_key", "u_measure_key", "w_measure_key"))
            if got != (self.code_key, self.t_measure_key, self.u_measure_key, self.w_measure_key):
                refuse(f"block {b}'s keys {got} are not the current ones (W.8)", EXIT_KEY)
            d = (doc[b].get("git") or {}).get("dirty_hashed")
            if d is None or d:
                refuse(f"block {b} was written with dirty hashed files (or no git record): {d}")

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, t_measure_key=self.t_measure_key,
                    u_measure_key=self.u_measure_key, w_measure_key=self.w_measure_key,
                    pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())

    def _write(self, stage: str, body: dict, wall_s: float = 0.0) -> dict:
        block = self._stamp(stage, body)
        ledger = dict(stage=stage, wall_s=float(wall_s), at=block["written_at"])
        w_store.write_summary_block(self.summary_path, stage, block, self.plist, ledger)
        return block

    def _elapsed_h(self, doc: dict) -> float:
        return w_records.elapsed_h(doc.get("ledger", []))

    # ---- progress (resumable wall clock of a pool stage) ------------------------------------------------------------
    def _prog_path(self, stage: str) -> str:
        return f"{self.spec.progress_dir}/{stage}.json"

    def _prog(self, stage: str) -> float:
        p = Path(self._prog_path(stage))
        return float(json.loads(p.read_text())["wall_s"]) if p.exists() else 0.0

    def _prog_add(self, stage: str, s: float) -> float:
        tot = self._prog(stage) + float(s)
        w_store.write_json(self._prog_path(stage), dict(wall_s=tot, at=_now()), self.plist)
        return tot

    # ---- measurers ---------------------------------------------------------------------------------------------------
    def _m(self, z: dict, smoke: bool = False):
        k = (tuple(sorted(_z(z).items())), smoke)
        if k not in self._ms:
            self._ms[k] = self.measure(_z(z), smoke)
        return self._ms[k]

    def _wm(self, smoke: bool = False):
        return self.wm(smoke)

    def _z_v(self, doc) -> dict:
        return _z(doc["reuse"]["z_V"])

    # ---- units -------------------------------------------------------------------------------------------------------
    def _unit(self, r, idx, fly, brain, edit, phases, seeds, plastic=True) -> dict:
        return dict(pair=row_key(r), idx=idx, fly=int(fly), brain=brain, edit=edit, odor_x=dict(r["odor_x"]),
                    odor_y=dict(r["odor_y"]), phases=phases, probe_seeds=[int(s) for s in seeds],
                    plastic=bool(plastic))

    def units(self, rows: list, where: str, K: int, F: int, edit: str, k0: int = 0, brains=BRAINS,
              plastic=True) -> list:
        """where = "main" (row["c"]), "pilot" (position j) or "smoke"; probes k0..K−1; F flies; brains."""
        sp, out = self.spec, []
        for j, r in enumerate(rows):
            for f in range(F):
                if where == "main":
                    c = r["c"]
                    seeds, b0, b1 = sp.probe_seeds(c, f, K, k0), sp.train_base(c, 0), sp.train_base(c, 1)
                elif where == "pilot":
                    c = j
                    seeds, b0, b1 = sp.pilot_probe_seeds(j, f, K, k0), sp.pilot_train_base(j, 0), \
                        sp.pilot_train_base(j, 1)
                else:
                    c = "smoke"
                    seeds, b0, b1 = sp.smoke_probe_seeds(f, K, k0), sp.smoke_train_base(0), sp.smoke_train_base(1)
                for b in brains:
                    ph = [] if b == "naive" else sp.phases(b if b in BRAINS else "R", b0, b1)
                    out.append(self._unit(r, c, f, b, edit, ph, seeds, plastic))
        return out

    def _declared(self, edit: str, seeds_by_fly: dict) -> dict:
        sp = V_SPEC
        lever = edit == sp.lever_edit
        return dict(edit=edit, csc_sha256=sp.sha_combined if lever else sp.sha_none,
                    edit_edges=sp.lever_edges if lever else 0,
                    block_edges=dict(sp.contrast_declared()["chain_entry"]) if lever else {},
                    probe_seeds=seeds_by_fly)

    @staticmethod
    def _by_pair(rows: list) -> dict:
        out = {}
        for r in rows:
            out.setdefault(r["unit"]["pair"], []).append(r)
        return out

    @staticmethod
    def _manifest(rows: list) -> list:
        return [dict(key=f"{r['unit']['pair']}|{r['unit']['fly']}|{r['unit']['brain']}", cache_file=r["cache_file"],
                     cache_key=r["cache_key"], sha256=sha256_file(r["cache_file"])) for r in rows]

    # ================================================================ 0: the verdict code, w_oc, tables (W.9.9 순서 0)
    def stage_stage0(self, n_boot_timing: int = 4) -> dict:
        """No pool, no data: the decision code's hashes, the protocol / seed / design tables, w_oc's synthetic
        validation (W.9.9 P2-11) and the OC's compute time (a synthetic pilot; the bootstrap timed on n_boot_timing
        draws and scaled to boot_draws) — PASS iff every synthetic check passed."""
        self._require("stage0")
        sp = self.spec
        z = sp.z_v()
        t0 = time.perf_counter()
        syn = w_oc.synthetic_validation(sp, z, n_rep=sp.synth_reps)
        rng = np.random.default_rng(np.random.SeedSequence([sp.oc_seed, w_oc.TAG_SYNTH, 1]))
        pil = w_oc.synthetic_pilot(rng, base=(35.0, 90.0), sd=6.0, corr=0.7, learn=(40.0, 15.0), drift_ax=3.0)
        doc = w_oc.run(pil, z, sp, lambda K, F: K * F, n_boot=n_boot_timing)
        tm = doc["timing"]
        est = tm["point_s"] + tm["records_s"] + tm["boot_s"] * sp.boot_draws / max(n_boot_timing, 1)
        files = {f: sha256_file(_ROOT / f) for f in ("flymon/brain/w_verdict.py", "flymon/brain/w_oc.py",
                                                    "flymon/brain/w_spec.py", "flymon/brain/w_measure.py")}
        tables = dict(protocol=dict(trials=sp.trials, pulse_ms=sp.pulse_ms, train_settle_ms=sp.train_settle_ms,
                                    gap_ms=sp.gap_ms, strength=sp.strength, reward=sp.reward_dan,
                                    punish=sp.punish_dan, probe=dict(settle_ms=V_SPEC.settle_ms,
                                                                     read_ms=V_SPEC.read_ms,
                                                                     window_ms=V_SPEC.window_ms),
                                    c3=self.ctx["c3_params"]),
                      designs=dict(q=list(sp.q_grid), K=list(sp.k_grid), F=[sp.f_min, sp.f_max], k_cap=sp.k_cap),
                      seeds=dict(probe="26_000_000 + c·4_000 + f·100 + k", train="28_000_000 + c·40_000 + f·1_000 + t",
                                 pilot_probe="40_000_000 + j·4_000 + f·100 + k",
                                 pilot_train="41_000_000 + j·40_000 + f·1_000 + t", smoke="42_100_000-42_199_999",
                                 oc=sp.oc_seed, oracle=sp.oracle_seeds()),
                      oc=dict(reps=sp.oc_reps, cal_reps=sp.cal_reps, cal_tol=sp.cal_tol, cal_iter=sp.cal_iter,
                              boot_draws=sp.boot_draws, boot_reps=sp.boot_reps, boot_level=sp.boot_level,
                              cluster_grid=list(sp.cluster_grid), noise="pilot residual vectors, resampled",
                              cal_floor_rule=sp.cal_floor_rule, synth_reps=sp.synth_reps))
        body = dict(outcome=w_rules.PASS if syn["ok"] else w_rules.INVALID,
                    reasons=[] if syn["ok"] else [k for k, v in syn.items() if isinstance(v, dict) and not v["ok"]],
                    synthetic=syn, oc_timing=tm, oc_compute_estimate_s=float(est), decision_files=files, tables=tables,
                    note="W.9.9 순서 0: 판정 코드·w_oc·보정·수치 표·합성 검증, 파일럿 전에 커밋.")
        return self._write("stage0", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 1: reuse (W.3 1)
    def stage_reuse(self) -> dict:
        self._require("reuse")
        dec = self._reuse_dec()
        v = self.ctx["v_doc"]()
        body = dict(dec, z_V=(v.get("z") or {}).get("z_V"), v_judge_band=(v.get("judge") or {}).get("band"),
                    u_measure_key_u=self.spec.u_measure_key_u, v_commits=dict(self.spec.v_commits))
        return self._write("reuse", body)

    # ================================================================ 2: the path gate (W.3 2, W.9.6 P2-13, W.9.9 P2-9)
    def stage_path(self) -> dict:
        doc = self._require("path")
        sp, ctx = self.spec, self.ctx
        t0 = time.perf_counter()
        wm = self._wm()
        checks, raw = [], {}
        # (i) V gate ②'s punish arm, L_V and C, both directions, the first repro_p_count seeds
        _c1, st = ctx["p_inputs"](V_SPEC.p, False)
        seeds = list(V_SPEC.p.seeds[:sp.repro_p_count])
        items = [i for i in p_items(V_SPEC.p, st, V_SPEC.lever_edit, seeds) + p_items(V_SPEC.p_c, st, V_SPEC.no_edit,
                                                                                      seeds)
                 if i["arm"] == sp.repro_p_arm]
        try:
            v_rows = ctx["v_gate2_rows"](items)
        except (RuntimeError, ValueError, AttributeError) as e:
            refuse(f"V's gate-② rows cannot be read: {e}")
        n = V_SPEC.p.o.n
        h4 = n.h4
        p_win = dict(strength=n.h3.strength, settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms,
                     window_ms=int(h4.kc_window_ms))
        p_tim = dict(present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms, train_settle_ms=h4.teach_window.settle_ms,
                     seed_stride=n.train_seed_stride)
        units = [dict(pair=f"P|{i['direction']}|{i['seed']}", idx="p", fly=i["seed"], brain="P", edit=i["edit"],
                      odor_x=i["odor_x"], odor_y=i["odor_y"], phases=[[n.h3.punish_type, int(h4.teach_trials),
                                                                       n.train_seed_base]],
                      probe_seeds=[i["seed"]], plastic=True) for i in items]
        got = wm.run(units, **p_win, **p_tim)
        raw["p"] = got
        for i, v, w in zip(items, v_rows, got):
            tag = f"{'L_V' if i['edit'] == V_SPEC.lever_edit else 'C'} {i['direction']} {i['seed']}"
            checks.append(dict(ref=f"V 관문 ② 처벌 팔 행({tag})", diffs=w_records.p_repro_diffs(w, v, tag),
                               invalid=self._job_invalid(w, i["edit"], tag)))
        # (ii) the oracle's naive probe: V's judgement raw report.pre, rows 0..n−1, L_V and C (report seeds)
        try:
            pairs = {nm: ctx["v_jm"](nm, sp.repro_naive_rows) for nm in ("L", "C")}
        except (KeyError, ValueError) as e:
            refuse(f"V's judgement raw cannot be read: {e}")
        rep = V_SPEC.judge_seeds()["report"]
        nunits, refs = [], []
        for nm, lst in pairs.items():
            edit = V_SPEC.cond(nm).edit
            for r, res in lst:
                nunits.append(dict(pair=row_key(r), idx="v", fly=0, brain="naive", edit=edit, odor_x=r["odor_x"],
                                   odor_y=r["odor_y"], phases=[], probe_seeds=rep, plastic=True))
                refs.append((nm, r, res))
        got2 = wm.run(nunits)
        raw["naive"] = got2
        for (nm, r, res), w in zip(refs, got2):
            tag = f"{nm} {row_key(r)}"
            checks.append(dict(ref=f"V 오라클 순진 카운트({tag})", diffs=w_records.naive_repro_diffs(w, res, tag),
                               invalid=self._job_invalid(w, V_SPEC.cond(nm).edit, tag)))
        # (iii) the reward path (W.9.9 P2-9): pilot pair 0, fly 0, R's reward phase only, K = pilot probes
        pr = ctx["pilot_rows"]()[0]
        u = self.units([pr], "pilot", sp.pilot_probes, 1, V_SPEC.lever_edit, brains=("R",))[0]
        u = dict(u, phases=u["phases"][:1], brain="reward_check")
        w3 = wm.run([u])[0]
        raw["reward"] = w3
        checks.append(dict(ref="보상 경로 양성 검사", diffs=w_records.reward_check(w3),
                           invalid=self._job_invalid(w3, V_SPEC.lever_edit, "reward")))
        dec = w_rules.path(checks)
        p = w_store.write_json(sp.path_detail, dict(w_measure_key=self.w_measure_key, rows=raw), self.plist)
        body = dict(dec, checks=checks, detail_path=sp.path_detail, detail_sha256=sha256_file(p),
                    n_p=len(units), n_naive=len(nunits), z_V=doc["reuse"]["z_V"])
        return self._write("path", body, wall_s=time.perf_counter() - t0)

    def _job_invalid(self, res: dict, edit: str, tag: str) -> list:
        d = self._declared(edit, {})
        return [f"{tag}: {k} {res.get(k)!r} ≠ {d[k]!r}" for k in ("edit", "csc_sha256", "edit_edges", "block_edges")
                if res.get(k) != d[k]]

    # ================================================================ 3 / 3a: the pilot (W.3 3, W.9.4, W.9.8 H6)
    def _pilot_data(self, rows: list, got: list, K: int, F: int) -> tuple:
        by = self._by_pair(got)
        pairs, mach = {}, []
        for j, r in enumerate(rows):
            k = row_key(r)
            seeds = {f: self.spec.pilot_probe_seeds(j, f, K) for f in range(F)}
            mach += [f"{k}: {m}" for m in w_records.machine_reasons(by[k], list(range(F)),
                                                                  self._declared(V_SPEC.lever_edit, seeds))]
            pairs[k] = w_records.pair_data(by[k], list(range(F)))
        return pairs, mach

    def stage_pilot(self) -> dict:
        doc = self._require("pilot")
        sp = self.spec
        rows = self.ctx["pilot_rows"]()
        K, F = sp.pilot_probes, sp.pilot_flies
        units = self.units(rows, "pilot", K, F, V_SPEC.lever_edit)
        t0 = time.perf_counter()
        prev = self._prog("pilot")
        wm = self._wm()

        def tick(done, todo):
            self._prog_add("pilot", time.perf_counter() - tick.t)
            tick.t = time.perf_counter()
        tick.t = time.perf_counter()
        got = wm.learn(units, "pilot", check=tick)
        wall = prev + (time.perf_counter() - t0)
        pairs, mach = self._pilot_data(rows, got, K, F)
        walls = dict(job_s_median=float(np.median([g["result"]["wall_s"] for g in got])),
                     train_s_median=float(np.median([g["result"]["train_s"] for g in got])),
                     probe_s_median=float(np.median([g["result"]["probe_s"] for g in got])))
        rec = w_records.pilot_record(pairs, self._z_v(doc), walls, sp)
        dec = w_rules.pilot(rec, mach, sp)
        exploratory = w_verdict.judge({k: (d, None) for k, d in pairs.items()}, mach, self._z_v(doc), 0.75, F, K, sp)
        body = dict(dec, record=rec, exploratory=dict(verdict=exploratory["verdict"], label="탐색",
                                                    pairs={k: v["status"] for k, v in exploratory["pairs"].items()}),
                    pairs=[row_key(r) for r in rows], n_units=len(units), manifest=self._manifest(got))
        return self._write("pilot", body, wall_s=wall)

    # ================================================================ 4: the OC and the design (W.9.3, H1, W.9.9)
    def _pilot_back(self, doc: dict) -> tuple:
        sp = self.spec
        rows = self.ctx["pilot_rows"]()
        got, bad = w_store.load_manifest(doc["pilot"]["manifest"])
        if bad:
            refuse(f"pilot raw changed since block pilot: {bad[:3]}")
        by = {}
        for g in got:
            pair, fly, brain = g["key"].rsplit("|", 2)
            by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
        pairs = {row_key(r): w_records.pair_data(by[row_key(r)], list(range(sp.pilot_flies))) for r in rows}
        return pairs, got

    def _costs(self, job_results: list, oracle_round_s: float) -> dict:
        return w_records.unit_costs(job_results, 2 * self.spec.trials, oracle_round_s, self.spec.workers)

    def _v_oracle_round_s(self) -> float:
        """V's jm:L (the same oracle job on L_V): its wall clock per worker round."""
        b = self.ctx["v_doc"]()["jm:L"]
        return float(b["wall_s"]) / max(1, -(-int(b["jobs"]) // int(V_SPEC.workers)))

    def stage_oc(self) -> dict:
        doc = self._require("oc")
        sp = self.spec
        pairs, got = self._pilot_back(doc)
        costs = self._costs([g["result"] for g in got], self._v_oracle_round_s())
        n_set = sp.n_b_expected + sp.n_a_expected

        def cost(K, F):
            return w_records.design_cost(costs, K, F, sp, n_set)["total_h"]
        t0 = time.perf_counter()
        oc = w_oc.run(list(pairs.values()), self._z_v(doc), sp, cost,
                      log=lambda m: print(m, file=sys.stderr))
        p = w_store.write_json(sp.oc_detail, oc, self.plist)
        dec = w_rules.oc(oc, sp)
        body = dict(dec, selected=oc["selected"], ranking=oc["ranking"], reachable=oc["reachable"],
                    calibration=oc["calibration"], theta=oc["theta"], records=oc.get("records"), costs=costs,
                    detail_path=sp.oc_detail, detail_sha256=sha256_file(p), timing=oc["timing"],
                    note="작동 특성은 파일럿 잡음 모형 조건부이며 Q → … → W 전체 절차의 오선택률이 아니다. 관문 쌍은 "
                         "오라클 순진·시험 가능으로 고른 조건부 표본이다. 파일럿은 순진 불균형 쌍 위주다.")
        return self._write("oc", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 5: smoke and the cost ledger (W.9.9 순서 5)
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp = self.spec
        if not sp.smoke:
            refuse("smoke needs the smoke spec (42_1xx_xxx)")
        d = doc["oc"]["selected"]
        K = int(d["K"])
        rows = self.ctx["pilot_rows"]()[:1]
        t0 = time.perf_counter()
        wm = self._wm(smoke=True)
        lv = V_SPEC.lever_edit
        learn = wm.learn(self.units(rows, "smoke", K, sp.smoke_flies, lv), "smoke")
        band = wm.learn(self.units(rows, "smoke", 2 * K, sp.smoke_flies, lv, k0=K), "smoke_band")
        naive = wm.learn(self.units(rows, "smoke", K, sp.smoke_flies, lv, brains=("naive",)), "smoke_naive")
        z_v = self._z_v(doc)
        m = self._m(z_v, smoke=True)
        t1 = time.perf_counter()
        orc = m.oracle(rows, V_SPEC.cond("L"), "smoke", sp.oracle_seeds())
        oracle_s = time.perf_counter() - t1
        flies = list(range(sp.smoke_flies))
        seeds = {f: sp.smoke_probe_seeds(f, K) for f in flies}
        nv = {g["unit"]["fly"]: w_records.counts(g["result"]["stages"][0]) for g in naive}
        problems = w_records.machine_reasons(learn, flies, self._declared(lv, seeds), naive=nv, band=band)
        d_k = w_records.pair_data(learn, flies)
        if bool(w_verdict.rn1_mismatch({s: np.asarray(v)[None] for s, v in d_k.items()})[0]):
            problems.append("RN1 ≠ R1")
        q = orc[0]["result"].get("q", {})
        if q.get("edit") != lv or q.get("csc_sha256") != V_SPEC.sha_combined or q.get("edit_edges") != 2:
            problems.append(f"oracle edit {q.get('edit')} / CSC {q.get('csc_sha256')} / edges {q.get('edit_edges')}")
        if list(m.z.items()) != list(z_v.items()):
            problems.append(f"oracle z {m.z} ≠ z_V {z_v}")
        costs = self._costs([g["result"] for g in learn], oracle_s)
        cost = w_records.design_cost(costs, K, int(d["F"]), sp, sp.n_b_expected + sp.n_a_expected)
        body = dict(problems=problems, design=d, costs=costs, cost=cost, oracle_wall_s=oracle_s,
                    seeds=dict(probe=seeds, oracle=sp.oracle_seeds()), pair=row_key(rows[0]))
        return self._write("smoke", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 5a: the worst-case budget gate (W.9.9 P1-5)
    def _options(self, doc: dict, n_set: int, n_naive: int | None, with_c_first: bool = True) -> list:
        sp, costs = self.spec, doc["smoke"]["costs"]
        sel = doc["oc"]["selected"]
        out = []
        if with_c_first:
            out.append(dict(design=sel, with_c=True, **w_records.design_cost(costs, sel["K"], sel["F"], sp, n_set,
                                                                             n_naive, True)))
        out.append(dict(design=sel, with_c=False, **w_records.design_cost(costs, sel["K"], sel["F"], sp, n_set,
                                                                          n_naive, False)))
        for alt in doc["oc"]["ranking"]:
            if (alt["q"], alt["K"], alt["F"]) != (sel["q"], sel["K"], sel["F"]):
                out.append(dict(design=alt, with_c=False, **w_records.design_cost(costs, alt["K"], alt["F"], sp, n_set,
                                                                                  n_naive, False)))
        return out

    def stage_budget(self) -> dict:
        doc = self._require("budget")
        sp = self.spec
        n_set = sp.n_b_expected + sp.n_a_expected
        dec = w_rules.budget(self._elapsed_h(doc), self._options(doc, n_set, None), sp, stage="5a")
        return self._write("budget", dict(dec, n_set=n_set, note="W.9.9 P1-5: 주 세트 digest 전 최악 비용 관문."))

    def _plan(self, doc: dict) -> dict:
        p = doc["budget"]["plan"]
        e = doc.get("estimate") or {}
        if e.get("plan"):
            p = dict(p, with_c=e["plan"]["with_c"])
        return p
