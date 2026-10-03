"""Spec S's stage chain (S.2-S.5, S.9; plan Readings 4-6, 9-12):
reuse -> set -> smoke -> oc -> gate2_oc -> gate2 -> jm:L -> jm:C -> jm:E0 -> seal -> judge
(and after judge only: recompute / invalid_run), one block each in results/summary/s_lever.json, written only through
s_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists,
its own block exists (gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or a hashed S file is
dirty. Every stage after `reuse` re-checks R's reuse condition (S.9.5: the shared measurement key equals R's and R's
reused blocks are intact) and refuses with SystemExit 7 when it broke — S has no re-measurement path (Reading 4). A
smoke with problems or a gate whose outcome is not PASS blocks every later stage, so the judgement set stays unused.
The judgement set is measured only through ctx["judgement_rows"] (s_pairs.judgement_rows: generated afresh and checked
against S.2's declared values on every call) from the judgement stages; `set` reads the list once (no measurement);
jm blocks carry no pair statistic. Blocks carry both keys: code_key = the shared measurement key (R_MEASURE_FILES,
which also keys the cache) and pipeline_key = S's own files (S.9.5)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent.e_runner import summary_git
from . import r_records, s_records, s_rules, s_store
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import R_HASHED_FILES, p_items
from .s_pairs import check_s_set
from .s_pairs import summary as set_summary
from .s_spec import smoke

ORDER = ("reuse", "set", "smoke", "oc", "gate2_oc", "gate2", "jm:L", "jm:C", "jm:E0", "seal", "judge")
GATES = ("reuse", "set", "gate2")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_REUSE = 2, 7
S_PIPELINE_FILES = ("flymon/brain/s_spec.py", "flymon/brain/s_pairs.py", "flymon/brain/s_store.py",
                    "flymon/brain/s_records.py", "flymon/brain/s_rules.py", "flymon/brain/s_runner.py",
                    "scripts/run_s.py")
S_HASHED_FILES = tuple(dict.fromkeys(R_HASHED_FILES + S_PIPELINE_FILES + ("results/summary/r_lever.json",)))
# The decision code: every hashed S file outside the shared measurement key. The seal pins its hash; the judge reads
# only under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in S_HASHED_FILES if f not in R_MEASURE_FILES)
# Written before the band is computed: the set was read once. REREAD: S.9.2's one re-generation after the mark.
# DONE: written right after the judge block, so a judge block written and then discarded never reopens the set.
JUDGE_MARKER = "results/s/judge_read.json"
REREAD_MARKER = "results/s/judge_reread.json"
DONE_MARKER = "results/s/judge_done.json"


def git_state() -> dict:
    return _h3_git_state(files=S_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(S_PIPELINE_FILES)


def decision_key() -> dict:
    return _files_key(DECISION_FILES)


def refuse(msg: str, code: int = EXIT_REFUSE):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _rounds(n: int, w: int) -> int:
    return -(-int(n) // max(1, int(w)))


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, z — block h4), the H.4 pools as types, the encoder summary, the even rows (smoke only),
    lazy callables for S's set (list only) and judgement rows (checked), P's committed entries (gate ②'s OC) and
    stimuli (c1 from block n1), and R's summary with its git state (the reuse condition)."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from . import n_cli, p_cli, q_pairs, r_pairs, s_pairs
    from .circuits import Populations
    from .connectome import Connectome
    cfg = load_c3_config(spec.m0d_summary)
    if (cfg.readout["A"], cfg.readout["P"]) != (spec.a_type, spec.p_type):
        refuse(f"C3's readout is {cfg.readout}, not A {spec.a_type} / P {spec.p_type}")
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    even = r_pairs.even_rows(pops, rc, enc, spec)
    p_doc = json.loads(Path(spec.p_summary).read_text())["p"]
    model_types = n_cli.model_types(npz)

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                even_rows=even, s_set=lambda: s_pairs.s_set(pops, enc, spec),
                judgement_rows=lambda: s_pairs.judgement_rows(pops, rc, enc, spec),
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, r_doc=lambda: json.loads(Path(spec.r_summary).read_text()),
                r_git=lambda: summary_git(spec.r_summary))


class Runner:
    def __init__(self, measurer, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 pipeline: dict | None = None, archive_root=None):
        self.m, self.ctx, self.spec = measurer, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return s_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed S files are dirty: {gs['dirty_hashed']}")

    def _reuse_now(self, stage: str) -> dict:
        """S.9.5: the reuse condition, re-checked at every stage after `reuse` (so before the judgement measurement,
        the seal and the judge too)."""
        r = s_rules.reuse(self.ctx["r_doc"](), self.ctx["r_git"](), self.code_key, self.spec)
        if r["outcome"] != s_rules.PASS:
            refuse(f"stage {stage}: R's reuse condition broke ({'; '.join(r['reasons'])}) — S has no re-measurement "
                   f"path; S stops here and the judgement set stays unused (S.9.5, plan Reading 4)", EXIT_REUSE)
        return r

    def _require(self, stage: str, allow_own: bool = False) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; S never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; S never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != s_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — S stops there and the "
                   f"judgement set stays unused (S.3)")
        if i > 0:
            self._reuse_now(stage)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, pipeline_key=self.pipeline_key, git=git_state(),
                    written_at=_now())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        s_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    # ---- reuse (S.3 ①, S.9.5) and the set (S.2) — no pool ------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        rd = self.ctx["r_doc"]()
        dec = s_rules.reuse(rd, self.ctx["r_git"](), self.code_key, sp)

        def g(*ks):
            return _dig(rd, ks)
        rec = dict(k_even=g("gate3", "testable_b"), c_even=g("gate3", "c_even"),
                   gate1_median=g("gate1", "record", "median"), repro_csc_sha256_none=g("repro", "csc_sha256_none"),
                   r_smoke_L_csc_sha256=g("smoke", "oracle", "L", "csc_sha256"),
                   r_gate2_ell={d: g("gate2", "p_judgement", "directions", d, "ell") for d in sp.p.directions},
                   r_gate2_label=g("gate2", "label"))
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, r_summary=sp.r_summary,
                    r_commits=dict(sp.r_commits), r_blocks={b: g(b, "written_at") for b in sp.r_reused}, records=rec)
        self._write("reuse", body)
        return body

    def stage_set(self) -> dict:
        """S.2, list only: STOP_SET_SHORT is recorded; any other mismatch with the declared T / n_a / digests refuses."""
        self._require("set")
        js = self.ctx["s_set"]()
        dec = s_rules.set_outcome(js)
        if dec["outcome"] == s_rules.PASS:
            bad = check_s_set(js, self.spec)
            if bad:
                refuse(f"the S set does not reproduce its declaration (S.2): {bad}")
        body = dict(dec, set=set_summary(js))
        self._write("set", body)
        return body

    # ---- smoke (S.3 ②) ---------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        c1, st = self.ctx["p_inputs"](sm.p, True)
        items = p_items(sm.p, st, sp.lever_edit) + p_items(sm.p_c, st, sp.no_edit)
        rows = self.m.arms(items, self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke", sm.p.o.n)
        p_wall = self.m.last_wall_s
        p = {}
        for n, pspec, edit in (("L", sm.p, sp.lever_edit), ("C", sm.p_c, sp.no_edit)):
            rs = [r for r in rows if r["edit"] == edit]
            res = p_judge(rs, self.ctx["z"], c1["c1"], pspec)
            p[n] = dict(outcome=res["outcome"], label=res["label"], reasons=list(res.get("reasons", [])),
                        edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rs}), n_rows=len(rs))
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            got = self.m.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self.ctx["z"], [row_key(r) for r in sel], sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  kc_median=s["kc_median"], saturation=s["saturation"], wall_s=self.m.last_wall_s,
                                  jobs=self.m.last_jobs)
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        s_store.write_json(sp.smoke_detail, detail, self.plist)
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
        body = dict(detail, problems=self._smoke_problems(p, orc, repro_sha), cost=self._cost(p_wall, orc, sm),
                    detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, repro_sha) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        bad = []
        if p["L"]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"P arms L: lever edges {p['L']['edit_edges']}")
        if p["C"]["edit_edges"] != [0]:
            bad.append(f"P arms C: edges {p['C']['edit_edges']}, declared none")
        for n in ("L", "C"):
            if p[n]["outcome"] == P_INVALID:
                bad.append(f"P arms {n}: INVALID {p[n]['reasons'][:2]}")
        if orc[nl]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"L: lever edges {orc[nl]['edit_edges']}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not R's repro {repro_sha}")
        for n in sp.cond_names:
            if orc[n]["reasons"]:
                bad.append(f"{n}: {orc[n]['reasons'][:2]}")
        return bad

    def _cost(self, p_wall, orc, sm) -> dict:
        """A rough estimate from smoke wall times: rounds of workers × the seed ratio (recorded, never a rule)."""
        sp = self.spec
        per = len(sp.p.directions) * len(sp.p.arms) * 2                      # L and C
        p_h = p_wall / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        ratio = sum(len(v) for v in sp.judge_seeds().values()) / sum(len(v) for v in sm.even_seeds().values())
        return dict(gate2_h=p_h, jm_per_condition_h=o_round * ratio * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (S.6, S.9.7) — before gate ② ---------------------------------------------
    def stage_oc(self) -> dict:
        self._require("oc")
        body = s_rules.oc(self.spec)
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7: STOP_PUNISH_WEAKENED's probability from P's committed block (192 entries, no lever), before gate ②."""
        self._require("gate2_oc")
        pairs, point = P_SPEC.pairs(), [float(v) for v in P_SPEC.o.o2_point]
        wanted = {(d, a, int(s)) for d in P_SPEC.directions for a in P_SPEC.arms for s in P_SPEC.seeds}
        ref = self.ctx["p_ref"](wanted)
        if set(ref) != wanted:
            refuse(f"P's committed cache {self.spec.p_cache_dir} lacks {sorted(wanted - set(ref))[:3]}")
        rows = [dict(ref[k], direction=k[0], x=pairs[k[0]][0], y=pairs[k[0]][1], point=point) for k in sorted(wanted)]
        body = s_records.gate2_oc(rows, self.ctx["z"], P_SPEC, self.spec)
        self._write("gate2_oc", body)
        return body

    # ---- gate ② (S.3 ④, S.9.3) ----------------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """rerun (S.3 ④, R.5): only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID
        run's (a fix in a shared measurement file changes the shared key and refuses through _reuse_now: S.9.5). The
        INVALID block and the new one are written together after the measurement."""
        prior = None
        if rerun:
            doc = self._doc()
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (S.3 ④, R.5)")
            if not blk or blk.get("outcome") != s_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (S.3 ④)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (S.3 ④)")
            self._require("gate2", allow_own=True)
            prior = blk
        else:
            doc = self._require("gate2")
        sp = self.spec
        c1, st = self.ctx["p_inputs"](sp.p, False)
        items = p_items(sp.p, st, sp.lever_edit) + p_items(sp.p_c, st, sp.no_edit)
        rows = self.m.arms(items, self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2", sp.p.o.n)
        rows_l = [r for r in rows if r["edit"] == sp.lever_edit]
        rows_c = [r for r in rows if r["edit"] == sp.no_edit]
        res_l = p_judge(rows_l, self.ctx["z"], c1["c1"], sp.p)
        res_c = p_judge(rows_c, self.ctx["z"], c1["c1"], sp.p_c)
        ratio, why = None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = s_records.ratio(rows_l, rows_c, self.ctx["z"], sp)
            except ValueError as e:
                why.append(str(e))
        dec = s_rules.gate2(res_l, res_c, ratio, sp) if not why else dict(outcome=s_rules.INVALID, reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [sp.lever_edges] or edges_c != [0]) and dec["outcome"] != s_rules.INVALID:
            dec = dict(outcome=s_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{sp.lever_edges} / 0"])
        r_ell = (self._doc().get("reuse") or {}).get("records", {}).get("r_gate2_ell", {})
        diff = ({d: ratio[d]["ell_L"] - r_ell[d] for d in ratio if r_ell.get(d) is not None} if ratio else None)
        body = dict(dec, ratio=ratio, ell_L_minus_r_gate2=diff, p_judgement_L=res_l, p_judgement_C=res_c,
                    c1_source=c1, edit_edges_L=edges_l, edit_edges_C=edges_c, seeds=list(sp.p.seeds),
                    rerun_of=(prior or {}).get("written_at"), wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        if prior is None:
            return self._write("gate2", body)
        return self._write_rerun(prior, body)

    def _write_rerun(self, prior: dict, body: dict) -> dict:
        doc = self._doc()
        if GATE2_INVALID in doc or doc.get("gate2") != prior:
            refuse("the summary changed during gate ②'s rerun; nothing written")
        block = self._stamp("gate2", body)
        doc[GATE2_INVALID], doc["gate2"] = prior, block
        s_store.write_json(self.summary_path, doc, self.plist)
        return block
