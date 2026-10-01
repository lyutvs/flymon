"""Spec 4.0-4.4, 5.1-5.6 and 6: the encoder track's stage runner. One summary block per stage (set, drive, codebook,
strength, oc, even, judge), written only through e_store; each stage refuses (SystemExit 2, nothing written) unless
every earlier block exists, and refuses to rewrite a block once a later one exists (spec 6: order violations are
refused). The measurer (EMeasurer, or tests' ScriptedMeasurer) and the situations (an adapter with .even(),
.judgement() and .used()) are injected, so the runner needs no engine and no pops.

Smoke (R5): the summary is results/encoder/smoke/summary.json (never results/summary/encoder_grid.json), stage_even
does not need the oc block, and stage_judge refuses. Recovery (5.6): a stage interrupted mid-measurement wrote no
block; rerunning it reuses the measurer's cached entries. stage_judge writes nothing unless every pair is back and the
counts are the declared ones (INCOMPLETE = not yet a judgement); once a judge block exists it never runs again (D8).

Final review: every block records the cache code key (`code_key`); every non-smoke stage refuses while
results/summary/encoder_grid.json has uncommitted changes (4.0: each stage's result is committed before the next);
stage_judge additionally refuses unless the summary is tracked and clean, no committed version of it ever held a judge
block, no block is a smoke block, every block's code key and inputs equal the current ones, the even block's sha256 of
every hashed file equals the current one (other provenance files that differ are recorded only), and the winner
codebook actually used has the even block's digest."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

from ..brain import d6a, l_cli
from ..brain.h3_store import MEASURE_FILES, ROOT, canonical, sha256_file
from ..brain.h3_store import git_state as _h3_git_state
from ..brain.h4_formula import arm_aggregate, pair_stats
from ..brain.h4_pairs import pool_vocabulary
from . import e_codebook, e_pairs, e_rules, e_store
from .encode_grid import Codebook, cap_ok, odour, reachable, unique_odours

ORDER = ("set", "drive", "codebook", "strength", "oc", "even", "judge")
SMOKE_SUMMARY = "summary.json"                 # under spec.raw_dir (results/encoder/smoke) in smoke mode
SCRIPT = "scripts/run_encoder_grid.py"
B_TB_CLOSE = " → 이 범위(사구체 서로소 결합 부호, 선언한 4설정)의 인코더 재설계를 닫는다."
KS = (3, 2)
STATS_176 = "not measured (spec 4.3 decision 95258dd)"
NPZ = "data/malecns.npz"
M0D = "results/summary/m0d.json"
INPUT_FILES = (NPZ, M0D)                       # the inputs every block records by sha256 (CLI)
# The cache code key (what a cached measurement's value depends on): the H.3 engine and measurement files, every
# brain module oracle_job and activity_job import (h4_measure / n_measure / b_files form), and the track's measurer
# and store.
MEASURE_FILES_E = tuple(dict.fromkeys(tuple(MEASURE_FILES) + (
    "flymon/brain/plasticity.py", "flymon/brain/presentation.py", "flymon/brain/conditioning.py",
    "flymon/brain/h4_jobs.py", "flymon/brain/k_jobs.py", "flymon/brain/h4_formula.py",
    "flymon/agent/e_measure.py", "flymon/agent/e_store.py")))
# Recorded in every block's provenance; a difference between the even block and the judge is recorded, never refused.
RECORD_ONLY_FILES = ("docs/superpowers/specs/2026-10-01-encoder-redesign-design.md", "tests/agent/e_scripted.py",
                     "tests/agent/test_e_runner.py")


def provenance_files() -> list:
    """The track's code (spec 6 provenance): flymon/agent/e_*.py, encode_grid.py and the CLI script."""
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "flymon/agent").glob("e_*.py"))
    return files + ["flymon/agent/encode_grid.py", SCRIPT]


def hashed_files() -> tuple:
    """Files whose dirty state blocks the judgement and whose sha256 must not change between the even block and the
    judge: the code key's files, every brain/agent/battle module the track imports (C3 loading, situations,
    formulas, guards), the track's own code, and the M0d summary (C3 thresholds, pools, readout)."""
    deps = ("flymon/agent/config.py", "flymon/battle/pool.py",
            "flymon/brain/h4_pairs.py", "flymon/brain/l_pairs.py", "flymon/brain/k_pairs.py",
            "flymon/brain/h3_spec.py", "flymon/brain/h4_spec.py", "flymon/brain/h3_c3.py",
            "flymon/brain/j_store.py", "flymon/brain/j_params.py", "flymon/brain/odor_real.py",
            "flymon/brain/pool_bench.py", "flymon/brain/d6a.py", M0D)
    return tuple(dict.fromkeys(MEASURE_FILES_E + deps + tuple(provenance_files())))


def git_state() -> dict:
    """h3_store.git_state over this track's hashed files (tests monkeypatch this)."""
    return _h3_git_state(files=hashed_files())


def provenance(args: dict) -> dict:
    """sha256 of every hashed file (refused on change before the judge) and the record-only files."""
    names = dict.fromkeys(hashed_files() + tuple(provenance_files()) + RECORD_ONLY_FILES)
    files = [ROOT / f for f in names if (ROOT / f).exists()]
    return l_cli.provenance(files, args, sys.argv)


def summary_git(path) -> dict:
    """Git facts about the summary file (tests monkeypatch this): tracked, dirty (uncommitted changes, untracked
    included) and the commits whose version of the file holds a judge block."""
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True)
    path = str(path)
    tracked = run("ls-files", "--error-unmatch", "--", path).returncode == 0
    st = run("status", "--porcelain", "--", path)
    dirty = st.returncode != 0 or bool(st.stdout.strip())
    judged = []
    log = run("log", "--format=%H", "--", path)
    for c in (log.stdout.split() if log.returncode == 0 else []):
        rel = run("ls-tree", "--name-only", "--full-name", c, "--", path).stdout.strip()
        if not rel:
            continue
        shown = run("show", f"{c}:{rel}")
        try:
            doc = json.loads(shown.stdout) if shown.returncode == 0 else {}
        except ValueError:
            doc = {}
        if isinstance(doc, dict) and "judge" in doc:
            judged.append(c)
    return dict(tracked=tracked, dirty=dirty, judge_commits=judged)


def refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _okey(m, opp) -> str:
    return f"{m}|{'+'.join(opp)}"


def _row_key(r) -> tuple:
    return (r["axis"], int(r["turn"]), r["x"], r["y"])


def _quantiles(xs) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda f: xs[min(len(xs) - 1, int(round(f * (len(xs) - 1))))]
    return dict(min=xs[0], q10=q(0.1), median=float(median(xs)), q90=q(0.9), max=xs[-1])


def _median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(median(xs)) if xs else None


class Runner:
    def __init__(self, measurer, pops_info: dict, spec, oracle: dict, situations, summary_path=e_store.SUMMARY,
                 smoke: bool = False, inputs: dict | None = None, code: dict | None = None):
        self.measurer, self.info, self.spec = measurer, dict(pops_info), spec
        self.oracle_cfg, self.sit, self.smoke = dict(oracle), situations, bool(smoke)
        self.inputs = dict(inputs or {})
        if code is None:
            code = getattr(getattr(measurer, "cache", None), "code", None)
        self.code_key = (code or {}).get("key")
        if self.smoke:
            if not str(spec.raw_dir).rstrip("/").startswith("results/encoder/smoke"):
                refuse(f"smoke raw_dir {spec.raw_dir} is not under results/encoder/smoke")
            self.summary_path = f"{spec.raw_dir.rstrip('/')}/{SMOKE_SUMMARY}"
        else:
            if str(summary_path).startswith("results/encoder/smoke"):
                refuse("a full run never writes the smoke summary")
            self.summary_path = str(summary_path)
        self.params_list = list(getattr(measurer, "params_list", []))
        self.species_types, _, self.mon_types, self.move_types = pool_vocabulary()
        self.cells = e_codebook.cells(self.move_types, self.mon_types)

    # ---- chain ---------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return e_store.read_summary(self.summary_path)

    def _summary_committed(self, stage: str):
        """Spec 4.0: each stage's result is committed before the next stage (full runs only)."""
        if self.smoke:
            return
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes (tracked={g['tracked']}, "
                       f"dirty={g['dirty']}); commit the previous stage's block first (spec 4.0)")

    def _require(self, stage: str, *blocks) -> dict:
        self._summary_committed(stage)
        doc = self._doc()
        missing = [b for b in blocks if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing} in {self.summary_path}")
        later = [b for b in ORDER[ORDER.index(stage) + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; rewriting {stage} would orphan them")
        if stage != "set" and doc["set"].get("status") != "OK":
            refuse(f"stage {stage}: block set has status {doc['set'].get('status')}")
        return doc

    def _write(self, stage: str, body: dict) -> dict:
        block = dict(body, stage=stage, smoke=self.smoke, inputs=self.inputs, code_key=self.code_key, git=git_state(),
                     provenance=provenance(dict(stage=stage, smoke=self.smoke)),
                     written_at=_dt.datetime.now(_dt.timezone.utc).isoformat())
        e_store.write_summary_block(self.summary_path, stage, block, self.params_list)
        return block

    def _judgement(self, doc: dict) -> dict:
        """The judgement set, checked against block set's digests (the list is fixed at stage set)."""
        js = self.sit.judgement()
        st = doc["set"]
        for k in ("digest_e0_b", "digest_e0_a", "digest_keys", "n_a"):
            if js[k] != st[k]:
                refuse(f"judgement set {k} {js[k]} differs from block set's {st[k]}")
        return js

    def _book(self, doc: dict, cfg: str) -> Codebook:
        k = str(self.spec.k_of(cfg))
        return Codebook(self.cells, doc["codebook"]["k"][k]["codebook"])

    def _odour_table(self, cb: Codebook, rule: str) -> tuple:
        rc = self.info["receptor_counts"]
        odours, single, dual = {}, [], []
        for m, o in reachable(self.species_types, self.move_types):
            oid = _okey(m, o)
            odours[oid] = odour(rc, cb, m, o, rule)
            (single if len(o) == 1 else dual).append(oid)
        return odours, single, dual

    # ---- ⓪ set (5.1) ---------------------------------------------------------------------------------------
    def stage_set(self) -> dict:
        self._require_set()
        js, even = self.sit.judgement(), self.sit.even()
        shared = e_pairs.shared_odour_count(js["b"], even)
        reasons = {}
        for s in js["skipped"]:
            reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1
        slim = lambda r: {k: r[k] for k in ("axis", "turn", "x", "y", "move_x", "opp_x", "move_y", "opp_y")}
        body = dict(status=js["status"], n_b=len(js["b"]), n_a=js["n_a"], last_turn=js["last_turn"],
                    digest_e0_b=js["digest_e0_b"], digest_e0_a=js["digest_e0_a"], digest_keys=js["digest_keys"],
                    skipped=js["skipped"], skipped_counts=reasons, shared_odour_count=shared,
                    rows_b=[slim(r) for r in js["b"]], rows_a=[slim(r) for r in js["a"]])
        out = dict(outcome="OK")
        if js["status"] == e_rules.STOP_SET_SHORT:
            out = dict(outcome=e_rules.STOP_SET_SHORT,
                       sentence=e_rules.sentence(e_rules.STOP_SET_SHORT, dict(m=len(js["b"]))))
        self._write("set", dict(body, **out))
        return out

    def _require_set(self):
        self._summary_committed("set")
        doc = self._doc()
        later = [b for b in ORDER[1:] if b in doc]
        if later:
            refuse(f"stage set: later block(s) {later} exist; the judgement set is fixed")

    # ---- ① drive (4.1) -------------------------------------------------------------------------------------
    def stage_drive(self) -> dict:
        self._require("drive", "set")
        gl = [g for g in self.info["glomeruli"] if g not in self.spec.exclude]
        res = self.measurer.drive(gl, self.info["c_norm"], self.info["receptor_counts"])
        body = dict(glomeruli=gl, c_norm=float(self.info["c_norm"]), strength=self.spec.drive_strength,
                    seeds=list(self.spec.drive_seeds), table=res,
                    mean_spikes={g: float(res[g]["mean_spikes"]) for g in gl})
        self._write("drive", body)
        return dict(outcome="OK")

    # ---- ② codebook (3.3) ----------------------------------------------------------------------------------
    def stage_codebook(self) -> dict:
        doc = self._require("codebook", "set", "drive")
        js = self._judgement(doc)
        used = self.sit.used()
        drive = {g: float(v) for g, v in doc["drive"]["mean_spikes"].items()}
        st, move, cells = self.species_types, self.move_types, self.cells
        ks = {}
        for k in KS:
            res = e_codebook.build(drive, k, self.spec,
                                   unique_check=lambda book: unique_odours(Codebook(cells, book), st, move))
            col = res.get("colouring")
            colouring = None if col is None else [[m, t, int(c)] for (m, t), c in zip(cells, col)]
            rec = dict(status=res["status"], k=k, omega=res["omega"], alphabet=res["alphabet"],
                       colour_seed=res["colour_seed"], C=res["C"], colouring=colouring,
                       colouring_digest=(None if colouring is None else hashlib.sha256(
                           canonical(colouring).encode()).hexdigest()),
                       cells=[list(c) for c in cells], codebook=res["codebook"], digest=res["digest"],
                       anneal=res["anneal"])
            if res["status"] == e_rules.UNDECIDED:
                rec["sentence"] = e_rules.sentence(e_rules.UNDECIDED, dict(k=k))
            ks[str(k)] = rec
        configs = {}
        rows = js["b"] + js["a"]
        for cfg in self.spec.configs:
            k, rule = self.spec.k_of(cfg), self.spec.dual_rule(cfg)
            rec = ks[str(k)]
            c = dict(k=k, rule=rule, digest=rec["digest"])
            if rec["status"] != "OK":
                configs[cfg] = dict(c, status=rec["status"], overlap=None)
                continue
            cb = Codebook(cells, rec["codebook"])
            ov = e_pairs.overlap_report(rows, used, self.info["receptor_counts"], cb, rule)
            configs[cfg] = dict(c, overlap=ov, status="OK" if ov["cross"] == 0 and ov["within"] == 0 else "NOT_UNIQUE")
        self._write("codebook", dict(k=ks, configs=configs))
        return dict(outcome="OK", configs={c: v["status"] for c, v in configs.items()})

    # ---- ③ strength (4.3) ----------------------------------------------------------------------------------
    def stage_strength(self) -> dict:
        doc = self._require("strength", "set", "drive", "codebook")
        sp, info = self.spec, self.info
        configs = {}
        for cfg in sp.configs:
            cbk = doc["codebook"]["configs"][cfg]
            if cbk["status"] != "OK":
                configs[cfg] = dict(status="CODEBOOK_" + cbk["status"], s=None, reasons={}, table={})
                continue
            rule = sp.dual_rule(cfg)
            cb = self._book(doc, cfg)
            odours, single, dual = self._odour_table(cb, rule)
            table, record = {}, {}
            for s in sp.s_grid:
                if not cap_ok(info["receptor_counts"], cb, self.species_types, self.move_types, rule, s,
                              info["max_rate_hz"], info["cap_hz"]):
                    table[s] = dict(cap_ok=False)
                    record[s] = dict(cap_ok=False, measured=False)
                    continue
                act = self.measurer.activity(odours, s, sp.strength_seeds)
                per = e_rules.odour_activity({oid: act[oid]["frac"] for oid in odours})
                sv, dv = [per[o] for o in single], [per[o] for o in dual]
                table[s] = dict(cap_ok=True, single=sv, dual=dv)
                st = e_rules.strength_ok(sv, dv, sp)
                wins = [int(w) for oid in odours for w in act[oid]["max_win"]]
                allv = sv + dv
                record[s] = dict(cap_ok=True, measured=True, **st,
                                 band_share=sum(sp.kc_band[0] <= x <= sp.kc_band[1] for x in allv) / len(allv),
                                 d6a_over=sum(w >= d6a.OVER_SPIKES for w in wins), d6a_presentations=len(wins),
                                 d6a_condition=d6a.CONDITION,
                                 single_dist=_quantiles(sv), dual_dist=_quantiles(dv), per_odour=per)
            ch = e_rules.choose_strength(table, sp)
            configs[cfg] = dict(status="OK" if ch["s"] is not None else "NO_ELIGIBLE_S", s=ch["s"],
                                reasons=ch["reasons"], table=record, n_odours=len(odours), n_single=len(single),
                                n_dual=len(dual), stats_176=STATS_176)
        self._write("strength", dict(configs=configs, seeds=list(sp.strength_seeds), s_grid=list(sp.s_grid),
                                     cap_hz=float(info["cap_hz"]), max_rate_hz=float(info["max_rate_hz"])))
        return dict(outcome="OK", s={c: v["s"] for c, v in configs.items()})

    # ---- ④ oc (5.5) ----------------------------------------------------------------------------------------
    def stage_oc(self) -> dict:
        doc = self._require("oc", "set", "drive", "codebook", "strength")
        n_a = int(doc["set"]["n_a"])
        oc = e_rules.operating_characteristics(n_a, self.spec)
        self._write("oc", dict(n_a=n_a, table=oc, sha256=hashlib.sha256(canonical(oc).encode()).hexdigest(),
                               shared_odour_count=doc["set"]["shared_odour_count"]))
        return dict(outcome="OK")

    # ---- oracle aggregation (4.4 / 5.2) --------------------------------------------------------------------
    def _measure(self, rows, strength, seeds, tag) -> tuple:
        """Oracle rows -> (reasons, aggregate, pairs, manifest). The h4_rules.combo_stats checks without its even-turn
        rule (the judgement set has odd turns): duplicates, the declared list, report seeds, undefined/NaN d'."""
        o = self.oracle_cfg
        got = self.measurer.oracle(rows, strength, o["readout"], o["z"], o["types"], seeds, tag)
        z, sp = o["z"], self.spec
        reasons = []
        keys = [_row_key(r) for r in got]
        if len(set(keys)) != len(keys):
            reasons.append("duplicate pair rows")
        if set(keys) != {_row_key(r) for r in rows}:
            reasons.append("pairs differ from the declared list")
        n = len(seeds["report"])
        stats, pairs = {}, []
        for r in got:
            if any(len(r["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P")):
                reasons.append(f"pair {_row_key(r)} report probes are not the {n} report seeds")
                continue
            s = pair_stats(r["report"], z, sp.testable_min)
            if s is None or any(math.isnan(s[f]) for f in ("d_pre", "r", "p", "m")):
                reasons.append(f"pair {_row_key(r)} has an undefined d'")
                continue
            stats[_row_key(r)] = s
            pairs.append(dict(axis=r["axis"], turn=int(r["turn"]), x=r["x"], y=r["y"], **s,
                              kc_jaccard=r.get("kc", {}).get("jaccard")))
        agg = None
        if not reasons and any(k[0] == "b" for k in stats) and any(k[0] == "a" for k in stats):
            agg = arm_aggregate(stats, sp.naive_max, sp.bar_b / sp.n_b, sp.f_a_min)
        manifest = []
        for r in rows:
            if hasattr(self.measurer, "oracle_key"):
                key, path = self.measurer.oracle_key(r, strength, o["readout"], o["z"], o["types"], seeds)
            else:
                key, path = None, None
            manifest.append(dict(tag=tag, axis=r["axis"], turn=int(r["turn"]), x=r["x"], y=r["y"], cache_key=key,
                                 cache_file=path,
                                 cache_sha256=(sha256_file(path) if path and Path(path).exists() else None)))
        return reasons, agg, pairs, manifest

    # ---- ⑤ even (4.4) --------------------------------------------------------------------------------------
    def stage_even(self) -> dict:
        need = ("set", "drive", "codebook", "strength") + (() if self.smoke else ("oc",))
        doc = self._require("even", *need)
        sp, rc = self.spec, self.info["receptor_counts"]
        rows = self.sit.even()
        if sum(r["axis"] == "b" for r in rows) != sp.n_b:
            refuse(f"even situations have {sum(r['axis'] == 'b' for r in rows)} (b) rows, not {sp.n_b}")
        seeds = dict(act=list(sp.even_act_seeds), select=list(sp.even_select_seeds), report=list(sp.even_report_seeds))
        results, why = {}, {}
        for cfg in sp.configs:
            cbs, sts = doc["codebook"]["configs"][cfg]["status"], doc["strength"]["configs"][cfg]
            eligible = cbs == "OK" and sts["s"] is not None
            if not eligible:
                why[cfg] = cbs if cbs != "OK" else "KC 자격 s 없음"
                results[cfg] = dict(eligible=False, reason=why[cfg], testable_b=None, F_a=None)
                continue
            cb = self._book(doc, cfg)
            orows = e_pairs.attach_odours(rows, rc, cb, sp.dual_rule(cfg))
            reasons, agg, pairs, manifest = self._measure(orows, sts["s"], seeds, f"even:{cfg}")
            if reasons:
                raise RuntimeError(f"even {cfg}: oracle rows cannot be judged: {reasons}")
            results[cfg] = dict(eligible=True, s=sts["s"], digest=doc["codebook"]["configs"][cfg]["digest"],
                                testable_b=agg["testable_b"], F_a=agg["F_a"], naive_a=agg["naive_a"],
                                aggregate=agg, pairs=pairs, manifest=manifest,
                                kc_jaccard_median=_median_or_none([p["kc_jaccard"] for p in pairs]))
        sel = e_rules.select_config(results, sp)
        out = dict(outcome=sel["outcome"], winner=sel["winner"], passing=sel["passing"], near=sel["near"])
        if sel["outcome"] == e_rules.STOP_NO_ELIGIBLE:
            out["sentence"] = e_rules.sentence(sel["outcome"], dict(
                reasons=", ".join(f"{c} {why[c]}" for c in sp.configs)))
        elif sel["outcome"] == e_rules.STOP_EVEN_LOW:
            parts = [f"{c} {r['testable_b']}/{sp.n_b}·F_a {r['F_a']}" if r["eligible"] else f"{c} 자격 없음({r['reason']})"
                     for c, r in ((c, results[c]) for c in sp.configs)]
            out["sentence"] = e_rules.sentence(sel["outcome"], dict(table=", ".join(parts)))
        else:
            w = results[sel["winner"]]
            out.update(winner_digest=w["digest"], winner_s=w["s"], winner_testable_b=w["testable_b"])
        self._write("even", dict(out, configs=results, seeds=seeds))
        return out

    # ---- ⑥ judge (5.2-5.4, 5.6) ----------------------------------------------------------------------------
    def stage_judge(self) -> dict:
        if self.smoke:
            refuse("smoke never judges (the judgement set is not used in smoke)")
        doc = self._doc()
        if "judge" in doc:
            refuse("block judge exists; the judgement set is never run again (D8)")
        g = summary_git(self.summary_path)
        if not Path(self.summary_path).exists() or not g["tracked"] or g["dirty"]:
            refuse(f"{self.summary_path} must be committed and clean before the judgement (tracked={g['tracked']}, "
                   f"dirty={g['dirty']})")
        if g["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block ({g['judge_commits']}); the "
                   f"judgement set is never run again (D8)")
        doc = self._require("judge", "set", "drive", "codebook", "strength", "oc", "even")
        prior = ORDER[:-1]
        smoke_blocks = [b for b in prior if doc[b].get("smoke", True) is not False]
        if smoke_blocks:
            refuse(f"block(s) {smoke_blocks} are smoke blocks or carry no smoke flag")
        if not self.code_key:
            refuse("no cache code key given; the judgement must run on the code the selection ran on")
        missing_inputs = [f for f in INPUT_FILES if f not in self.inputs]
        if missing_inputs:
            refuse(f"inputs {missing_inputs} are not hashed")
        for b in prior:
            if doc[b].get("code_key") != self.code_key:
                refuse(f"block {b}'s code key {doc[b].get('code_key')} is not the current {self.code_key}")
            if doc[b].get("inputs") != self.inputs:
                refuse(f"block {b}'s inputs {doc[b].get('inputs')} differ from the current {self.inputs}")
        prev = (doc["even"].get("provenance") or {}).get("sha256")
        if prev is None:
            refuse("block even has no provenance sha256")
        cur = provenance(dict(stage="judge", smoke=self.smoke)).get("sha256", {})
        hashed = set(hashed_files())
        changed = sorted(f for f in set(prev) | set(cur) if prev.get(f) != cur.get(f))
        if [f for f in changed if f in hashed]:
            refuse(f"hashed files differ from block even's provenance: {[f for f in changed if f in hashed]}")
        record_only_changed = [f for f in changed if f not in hashed]
        sp, ev = self.spec, doc["even"]
        if ev.get("outcome") != e_rules.SELECTED:
            refuse(f"block even's outcome is {ev.get('outcome')}, not SELECTED")
        win = ev["winner"]
        k = str(sp.k_of(win))
        if not (ev.get("winner_digest") == doc["codebook"]["configs"][win]["digest"] == doc["codebook"]["k"][k]["digest"]):
            refuse(f"winner {win}'s codebook digest differs between blocks even and codebook")
        book_used = doc["codebook"]["k"][k]["codebook"]
        if book_used is None or e_codebook.digest(book_used) != ev.get("winner_digest"):
            refuse(f"winner {win}'s codebook does not hash to block even's winner_digest")
        s = doc["strength"]["configs"][win].get("s")
        if s is None or s != ev.get("winner_s"):
            refuse(f"winner {win}'s strength is not recorded consistently ({s} vs {ev.get('winner_s')})")
        dirty = {b: doc[b].get("git", {}).get("dirty_hashed") for b in ORDER[:-1]}
        dirty = {b: d for b, d in dirty.items() if d is None or d}
        if dirty:
            refuse(f"blocks written with dirty hashed files (or no git record): {dirty}")
        now = git_state()
        if now["dirty_hashed"]:
            refuse(f"hashed files are dirty now: {now['dirty_hashed']}")
        js = self._judgement(doc)
        rows = js["b"] + js["a"]
        seeds = dict(act=list(sp.judge_act_seeds), select=list(sp.judge_select_seeds),
                     report=list(sp.judge_report_seeds))
        wrows = e_pairs.attach_odours(rows, self.info["receptor_counts"], self._book(doc, win), sp.dual_rule(win))
        erows = [dict(r, odor_x=r["odor_x_e0"], odor_y=r["odor_y_e0"]) for r in rows]
        rw, aw, pw, mw = self._measure(wrows, s, seeds, f"judge:{win}")
        re_, ae, pe, me = self._measure(erows, sp.e0_strength, seeds, "judge:E0")
        n_a_decl = int(doc["set"]["n_a"])
        if rw or re_ or aw is None or ae is None:
            return dict(status="INCOMPLETE", band=None, reasons=rw + re_)
        rb = e_rules.read_band(aw["testable_b"], ae["testable_b"], aw["F_a"], aw["n_b"], aw["n_a"], n_a_decl,
                               aw["naive_a"], sp)
        if rb["band"] == e_rules.NOT_READ or (ae["n_b"], ae["n_a"]) != (sp.n_b, n_a_decl):
            return dict(status="INCOMPLETE", band=None, reasons=["COUNTS"])
        fields = dict(config=win, k_even=ev["winner_testable_b"], T=doc["set"]["last_turn"], n=aw["testable_b"],
                      c=ae["testable_b"], f_a=aw["F_a"], n_a=n_a_decl, naive_a=aw["naive_a"], reason=rb["reason"])
        text = e_rules.sentence(rb["band"], fields)
        if rb["band"] == e_rules.B_TB:
            text += B_TB_CLOSE
        manifest = mw + me
        out = dict(status="READ", band=rb["band"], reason=rb["reason"], n=aw["testable_b"], c=ae["testable_b"],
                   F_a=aw["F_a"], naive_a=aw["naive_a"], n_a=n_a_decl, T=doc["set"]["last_turn"], sentence=text)
        if rb["band"] == e_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        self._write("judge", dict(out, winner=win, strength=s, e0_strength=sp.e0_strength, seeds=seeds,
                                  record_only_changed_since_even=record_only_changed,
                                  aggregate=dict(winner=aw, e0=ae), pairs=dict(winner=pw, e0=pe),
                                  kc_jaccard=dict(winner=[p["kc_jaccard"] for p in pw],
                                                  e0=[p["kc_jaccard"] for p in pe]),
                                  kc_jaccard_median=dict(winner=_median_or_none([p["kc_jaccard"] for p in pw]),
                                                         e0=_median_or_none([p["kc_jaccard"] for p in pe])),
                                  manifest=manifest))
        return out
