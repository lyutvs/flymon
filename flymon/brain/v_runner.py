"""Spec V's stage chain (V.3 as ordered by V.9.6; plan Readings):
reuse -> path -> kc_input -> set -> z -> kc_band -> even -> smoke -> oc -> gate2_oc -> gate2 -> measurement_started ->
jm:L -> jm:C -> jm:E0 -> seal -> judge (and after judge only: recompute / invalid_run), one block each in
results/summary/v_lever.json, written only through v_store. Every stage refuses (SystemExit 2, nothing written) when an
earlier block is missing, a later block exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary
has uncommitted changes or a hashed V file is dirty. Every stage after `reuse` re-checks the reuse condition (R's shared
key and repro, T's key and unedited z, U's key and blocks) and refuses with SystemExit 7 when it broke;
measurement_started, jm, seal and judge also re-check that every earlier block carries the current keys (V.8, exit 7).
A smoke with problems or a gate whose outcome is not PASS blocks every later stage, so the judgement set stays unused.
z (V.1): L_V's oracle (even, smoke's L, jm:L) runs on a measurer built with block z's z_V (α chosen on z_V); C's and
E0's, every P arm, the KC activities and the path oracle on block h4's z (`measure(z)`). Gate ② reads both P conditions
on h4 z, then V.9.2's z_V ratio.
The set (V.2) is regenerated from block kc_input's values and checked against block set on every call (ctx's lazy
callables); only kc_band (odours only), the judgement stages and `_cost` (seed counts) name it."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent import e_rules
from ..agent.e_runner import summary_git
from . import r_records, t_records, u_records, v_records, v_rules, v_store
from .h3_spec import SPEC as _H3
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import p_items
from .t_measure import T_MEASURE_FILES
from .u_measure import U_MEASURE_FILES, u_edit
from .u_runner import U_HASHED_FILES, r_raw_spec
from .v_spec import smoke

ORDER = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "smoke", "oc", "gate2_oc", "gate2",
         "measurement_started", "jm:L", "jm:C", "jm:E0", "seal", "judge")
GATES = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "gate2")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_KEY = 2, 7
V_PIPELINE_FILES = ("flymon/brain/v_spec.py", "flymon/brain/v_pairs.py", "flymon/brain/v_store.py",
                    "flymon/brain/v_records.py", "flymon/brain/v_rules.py", "flymon/brain/v_runner.py",
                    "scripts/run_v.py")
V_HASHED_FILES = tuple(dict.fromkeys(U_HASHED_FILES + V_PIPELINE_FILES + ("results/summary/u_lever.json",)))
# The decision code: every hashed V file outside the measurement files (R's, T's, U's). The seal pins its hash; the
# judge reads only under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in V_HASHED_FILES
                       if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES and f not in U_MEASURE_FILES)
JUDGE_MARKER = "results/v/judge_read.json"
REREAD_MARKER = "results/v/judge_reread.json"
DONE_MARKER = "results/v/judge_done.json"
ENGINES = (("none", "none"), ("apl", "apl"), ("lever", "lever"))


def git_state() -> dict:
    return _h3_git_state(files=V_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(V_PIPELINE_FILES)


def decision_key() -> dict:
    return _files_key(DECISION_FILES)


def _sha(obj) -> str | None:
    return None if obj is None else hashlib.sha256(canonical(obj).encode()).hexdigest()


PINNED = ("z", "set", "kc_input")


def decision_pins(doc: dict) -> dict:
    """V.3 12: block z (z_V), block set and block kc_input, hashed into the seal's decision record (canonical JSON)."""
    return {f"{b}_sha256": _sha(doc.get(b)) for b in PINNED}


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


def _ztuple(z: dict) -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in z.items()}


def n_presentations() -> int:
    """H.3's reference set: n odours × 2 probe seeds each (96), the count every side must have."""
    return 2 * int(_H3.reference.n)


def kc_values(doc: dict) -> dict:
    """Block kc_input's per-odour medians per engine — the set filter's only source (V.2)."""
    rec = doc["kc_input"]["record"]
    return {e: dict(rec[e]["per_odour"]) for e in ("none", "lever")}


def apl_spec(spec):
    """The APL->MBON05-only engine of V.3 2 (U's f = 0 edit), for the path stage's oracle against R's even L raw."""
    import dataclasses
    return dataclasses.replace(spec, lever_edit=u_edit(0.0))


def u_rows_reader(spec):
    """U's entry_f0 rows (results/u/scan.json, read only), refused unless the file's sha256 is the one U's scan block
    recorded and V declares (V.3 2's base for L_V)."""
    def u_rows() -> dict:
        p = Path(spec.u_scan_detail)
        if not p.exists() or sha256_file(p) != spec.u_scan_detail_sha256:
            raise ValueError(f"{spec.u_scan_detail} is missing or its sha256 is not {spec.u_scan_detail_sha256}")
        return json.loads(p.read_text())["rows"][spec.u_entry_point]
    return u_rows


def build_ctx(spec, npz: str) -> dict:
    """U's context (u_runner.build_ctx on V's spec: C3, block h4's z, the H.4 pools, the encoder summary, the even rows,
    the reference set, R's even raw read only, P's entries and stimuli, R's and T's summaries, T's z rows, the 112
    calibration odours, the ORN cap) without U's set callables, plus V's: the KC candidates, the set (from block
    kc_input's values, checked against block set), its odours, E0 odours and judgement rows, the KC record, the cluster
    labels, the static edit facts, U's summary and git state, U's per_odour_none and U's entry_f0 rows."""
    from ..agent.config import load_c3_config
    from . import v_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .u_runner import build_ctx as u_build_ctx
    ctx = u_build_ctx(spec, npz)
    for k in ("t_set", "set_odours", "judgement_rows", "clusters", "enc_per"):
        ctx.pop(k, None)
    cfg = load_c3_config(spec.m0d_summary)
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    enc, params = ctx["enc"], cfg.params
    ctx["enc_per"] = enc["strength"]["configs"][spec.config]["table"][str(spec.strength)]["per_odour"]
    u_doc = lambda: json.loads(Path(spec.u_summary).read_text())  # noqa: E731
    ctx.update(
        kc_candidates=lambda: v_pairs.candidate_odours(pops, enc, params),
        v_set=lambda kc: v_pairs.v_set(pops, enc, params, kc, spec),
        kc_record=lambda kc: v_pairs.kc_record(pops, enc, params, kc, spec),
        set_odours=lambda kc, blk: v_pairs.set_odours(pops, rc, enc, params, kc, blk, spec),
        set_e0_odours=lambda kc, blk: v_pairs.set_e0_odours(pops, enc, params, kc, blk, spec),
        judgement_rows=lambda kc, blk: v_pairs.judgement_rows(pops, rc, enc, params, kc, blk, spec),
        clusters=v_pairs.cluster_labels, csc_facts=lambda: v_records.csc_facts(conn, pops, params, spec),
        u_doc=u_doc, u_git=lambda: summary_git(spec.u_summary), u_none=lambda: spec.u_kc_none(u_doc()),
        u_rows=u_rows_reader(spec))
    return ctx


class Runner:
    def __init__(self, measure, zm, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, ucode: dict | None = None, pipeline: dict | None = None,
                 archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) on that z, one per z over one U pool and one VCache; zm: a
        u_measure.UZMeasurer (or its interface). ucode = the V measurement key (= U's measurement key)."""
        self.measure, self.zm, self.ctx, self.spec = measure, zm, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.u_measure_key = (ucode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]
        self._ms = {}

    # ---- measurers per z -----------------------------------------------------------------------------------------
    def _measurer(self, z: dict):
        k = tuple(sorted(z.items()))
        if k not in self._ms:
            self._ms[k] = self.measure(z)
        return self._ms[k]

    @property
    def m_h4(self):
        return self._measurer(_ztuple(self.ctx["z"]))

    def _z_v(self, doc) -> dict:
        return _ztuple(doc["z"]["z_V"])

    def _z_for(self, name: str, doc) -> dict:
        return self._z_v(doc) if name == self.spec.cond_names[0] else _ztuple(self.ctx["z"])

    def _m_for(self, name: str, doc):
        return self._measurer(self._z_for(name, doc))

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return v_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed V files are dirty: {gs['dirty_hashed']}")

    def _reuse_dec(self) -> dict:
        c = self.ctx
        return v_rules.reuse(c["r_doc"](), c["r_git"](), c["t_doc"](), c["t_git"](), c["u_doc"](), c["u_git"](),
                             self.code_key, self.t_measure_key, self.u_measure_key, self.spec)

    def _reuse_now(self, stage: str) -> dict:
        r = self._reuse_dec()
        if r["outcome"] != v_rules.PASS:
            refuse(f"stage {stage}: the reuse condition broke ({'; '.join(r['reasons'])}) — V has no re-measurement "
                   f"path; V stops here and the judgement set stays unused", EXIT_KEY)
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
            refuse(f"stage {stage}: later block(s) {later} exist; V never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; V never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != v_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — V stops there and the "
                   f"judgement set stays unused (V.3)")
        if i > 0:
            self._reuse_now(stage)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, t_measure_key=self.t_measure_key,
                    u_measure_key=self.u_measure_key, pipeline_key=self.pipeline_key, git=git_state(),
                    written_at=_now())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        v_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    def _side(self, edit: str, raw: dict, key: str) -> dict:
        ctx = self.ctx
        settle, steps = ctx["ref_window"]
        ref = self.zm.reference(edit, ctx["ref_odors"], ctx["ref_strength"], settle, steps)
        rest = self.zm.rest(edit, ctx["probe_seeds"], settle, steps)
        raw[key] = dict(edit=edit, ref=ref, rest=rest)
        return u_records.side(ref, rest, ctx["readout"], ctx["rec_types"], self.spec.kc_types, self.spec)

    def _r_even(self, rows, name):
        try:
            return self.ctx["r_even"](rows, name)
        except ValueError as e:
            refuse(f"V reads R's even raw by content key and it is not complete: {e}")

    def _set_call(self, what: str, doc: dict):
        try:
            return self.ctx[what](kc_values(doc), doc["set"]["set"])
        except ValueError as e:
            refuse(f"the V set does not reproduce block set (V.2): {e}")

    # ---- reuse (V.3 1) — no pool --------------------------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        dec = self._reuse_dec()
        rd, td, ud = self.ctx["r_doc"](), self.ctx["t_doc"](), self.ctx["u_doc"]()
        rec = dict(repro_csc_sha256_none=_dig(rd, ("repro", "csc_sha256_none")),
                   r_gate3=dict(testable_b=_dig(rd, ("gate3", "testable_b")), c_even=_dig(rd, ("gate3", "c_even"))),
                   t_z=dict(outcome=_dig(td, ("z", "outcome")), none_z=_dig(td, ("z", "none", "z")),
                            t_measure_key=_dig(td, ("z", "t_measure_key"))),
                   u=dict(path=_dig(ud, ("path", "outcome")), kc=_dig(ud, ("kc", "outcome")),
                          entry_f0_z=_dig(ud, ("scan", "contrast", sp.u_entry_point, "side", "z")),
                          entry_reading=_dig(ud, ("scan", "contrast", "readings", "entry", "reading")),
                          u_measure_key=_dig(ud, ("path", "u_measure_key"))))
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, t_measure_key_t=sp.t_measure_key_t,
                    u_measure_key_u=sp.u_measure_key_u, r_commits=dict(sp.r_commits), t_commits=dict(sp.t_commits),
                    u_commits=dict(sp.u_commits), records=rec)
        self._write("reuse", body)
        return body

    # ---- the path reproduction (V.3 2, V.9.5) -------------------------------------------------------------------
    def stage_path(self) -> dict:
        """The combined edit's static facts (V.1: INVALID unless 2 + 7 / 2 / 2 = 13 edges, CSC 2d359b8b…, MBON05->APL
        untouched); the reference set + same-seed rest, measured afresh with U's jobs on three engines — unedited (T's
        none rows, CSC 1aee8398…), APL->MBON05 only (T's lever rows, CSC 860cba4f…), L_V (U's entry_f0 rows, every field,
        CSC 2d359b8b…); the oracle on the first 3 even pairs (h4 z, H.4 seeds): APL->MBON05 only against R's even L raw
        and unedited against R's even C raw (labels aside). Any difference → STOP_V_PATH_REPRO; a defect → INVALID. The
        three sides are V's mechanism records (V.6). Raw rows go to results/v/path.json."""
        self._require("path")
        sp, ctx = self.spec, self.ctx
        try:
            t_rows, u_rows = ctx["t_rows"](), ctx["u_rows"]()
        except ValueError as e:
            refuse(f"T's z rows or U's entry_f0 rows cannot be read: {e}")
        facts = ctx["csc_facts"]()
        n, raw, sides, checks = n_presentations(), {}, {}, []
        chain = canonical(dict(sp.contrast_declared()["chain_entry"]))
        plan = (("none", sp.no_edit, 0, sp.sha_none, v_rules.NONE_ENGINE),
                ("apl", u_edit(0.0), sp.lever_edges, sp.sha_zero, v_rules.APL_ENGINE),
                ("lever", sp.lever_edit, sp.lever_edges, sp.sha_combined, v_rules.V_ENGINE))
        for key, edit, edges, sha, name in plan:
            s = self._side(edit, raw, key)
            sides[key] = s
            invalid = []
            if s["edit_edges"] != [edges]:
                invalid.append(f"{key}: edges {s['edit_edges']}, declared {edges}")
            if (s["n_ref"], s["n_rest"]) != (n, n):
                invalid.append(f"{key}: {s['n_ref']} reference / {s['n_rest']} rest, declared {n}")
            if key == "lever" and s["block_edges"] != [chain]:
                invalid.append(f"lever: chain entry edges {s['block_edges']}, declared {chain}")
            if key == "lever":
                ref = "U entry_f0 기준 집합 행"
                diffs = (v_records.full_row_diffs(raw[key]["ref"], u_rows["ref"], "기준 집합")
                         + v_records.full_row_diffs(raw[key]["rest"], u_rows["rest"], "휴지"))
            else:
                t_edit = sp.t_none_edit if key == "none" else sp.t_lever_edit
                ref = "T 편집 없는 엔진 기준 집합 행" if key == "none" else "T 지렛대 엔진 기준 집합 행"
                diffs = (u_records.row_diffs(raw[key]["ref"], t_rows[t_edit]["ref"], "기준 집합")
                         + u_records.row_diffs(raw[key]["rest"], t_rows[t_edit]["rest"], "휴지", rest=True))
            if s["csc_sha256"] != [sha]:
                diffs.append(f"CSC {s['csc_sha256']} ≠ {sha}")
            checks.append(dict(engine=name, ref=ref, diffs=diffs, invalid=invalid))
        rows = ctx["even_rows"][:sp.path_even_n]
        seeds, m = sp.h4_seeds(), self.m_h4
        nl, nc, _ = sp.cond_names
        for cond, rname, name in ((apl_spec(sp).cond(nl), nl, v_rules.APL_ENGINE), (sp.cond(nc), nc,
                                                                                     v_rules.NONE_ENGINE)):
            r_got = self._r_even(rows, rname)
            v_got = m.oracle(rows, cond, "path", seeds)
            checks.append(dict(engine=name, ref=f"R 짝수 {rname} 원자료 {len(rows)}쌍",
                               diffs=u_records.oracle_diffs(v_got, r_got, f"R 짝수 {rname}"), invalid=[]))
        dec = v_rules.path(checks, v_records.csc_reasons(facts, sp), sp)
        z_h4 = _ztuple(ctx["z"])
        records = {k: v_records.side_record(sides[k], sides["none"], z_h4, ctx["readout"]) for k in ("apl", "lever")}
        p = v_store.write_json(sp.path_detail, dict(u_measure_key=self.u_measure_key, readout=ctx["readout"],
                                                    rows=raw), self.plist)
        body = dict(dec, checks=checks, sides=sides, records=records, csc_facts=facts,
                    pairs=[row_key(r) for r in rows], n_presentations=n, detail_path=sp.path_detail,
                    detail_sha256=sha256_file(p), wall_s=m.last_wall_s)
        self._write("path", body)
        return body

    # ---- the KC input on both engines (V.3 3, V.9.1, V.9.5 P2-7 / P2-10) — no oracle ------------------------------
    def stage_kc_input(self) -> dict:
        self._require("kc_input")
        sp, ctx = self.spec, self.ctx
        try:
            cand = ctx["kc_candidates"]()
        except ValueError as e:
            refuse(f"the KC candidate odours cannot be listed: {e}")
        od, m, act, wall = cand["odours"], self.m_h4, {}, 0.0
        for e, edit in (("none", sp.no_edit), ("lever", sp.lever_edit)):
            act[e] = m.activity(od, edit, sp.strength, sp.kc_seeds(), "kc_input")
            wall += m.last_wall_s
        rec = v_records.kc_input_record(act, cand["cap"], cand["cap_hz"], sp)
        u_cmp = v_records.kc_u_diffs(rec["none"]["per_odour"], ctx["u_none"]())
        dec = v_rules.kc_input(rec, u_cmp, len(od), sp)
        p = v_store.write_json(sp.kc_input_detail, dict(u_measure_key=self.u_measure_key, activity=act), self.plist)
        body = dict(dec, record=rec, u_compare=u_cmp, n_candidates=len(od), n_candidate_rows=cand["n_rows"],
                    seeds=list(sp.kc_seeds()), detail_path=sp.kc_input_detail, detail_sha256=sha256_file(p),
                    wall_s=wall, note="V.5: KC 입력은 냄새 입력만 쓰고 오라클 결과를 보지 않는다.")
        self._write("kc_input", body)
        return body

    # ---- the set (V.2) — no pool --------------------------------------------------------------------------------
    def stage_set(self) -> dict:
        doc = self._require("set")
        kc = kc_values(doc)
        try:
            js = self.ctx["v_set"](kc)
            rec = self.ctx["kc_record"](kc)
        except ValueError as e:
            refuse(f"the V set cannot be generated: {e}")
        from .v_pairs import set_summary
        dec = v_rules.set_outcome(js)
        body = dict(dec, set=set_summary(js), odour_ids=js["odour_ids"], kc_record=rec,
                    note="V.9.5 P2-8: 판정 세트에 대한 어떤 오라클 측정보다 먼저 커밋한다.")
        self._write("set", body)
        return body

    # ---- z_V (V.3 5) — no pool ----------------------------------------------------------------------------------
    def stage_z(self) -> dict:
        doc = self._require("z")
        sp, ctx = self.spec, self.ctx
        side = doc["path"]["sides"]["lever"]
        dec = v_rules.z_v(side, n_presentations(), ctx["readout"], sp)
        z_h4 = _ztuple(ctx["z"])
        body = dict(dec, z_V=side["z"] if dec["outcome"] == v_rules.PASS else None, guard=side["guard"],
                    zero_sd=side["zero_sd"], record=doc["path"]["records"]["lever"],
                    z_h4={k: list(v) for k, v in z_h4.items()},
                    z_none_reused=bool(_ztuple(doc["reuse"]["records"]["t_z"]["none_z"]) == z_h4))
        self._write("z", body)
        return body

    # ---- the combined-engine KC band (V.3 6, V.9.1, V.9.5 P2-11) -------------------------------------------------
    def stage_kc_band(self) -> dict:
        doc = self._require("kc_band")
        sp, ctx = self.spec, self.ctx
        odours, single, dual = ctx["calib"]
        set_od = self._set_call("set_odours", doc)
        e0_od = self._set_call("set_e0_odours", doc)
        m = self.m_h4
        a112 = m.activity(odours, sp.lever_edit, sp.strength, sp.kc_seeds(), "kc_band")
        wall = m.last_wall_s
        aset = m.activity(set_od, sp.lever_edit, sp.strength, sp.kc_seeds(), "kc_band")
        wall += m.last_wall_s
        ae0 = m.activity(e0_od, sp.no_edit, sp.e0_strength, sp.kc_seeds(), "kc_band_e0")
        wall += m.last_wall_s
        rec112 = r_records.gate1_record(a112, single, dual, ctx["enc_per"], sp)
        per = e_rules.odour_activity({o: aset[o]["frac"] for o in aset})
        set_rec = dict(per_odour=per, n_odours=len(per), edit_edges=sorted({int(x) for v in aset.values()
                                                                           for x in v["edit_edges"]}),
                       csc_sha256=sorted({s for v in aset.values() for s in v["csc_sha256"]}))
        want = kc_values(doc)["lever"]
        bad = [o for o in sorted(per) if per[o] != want.get(o)]
        recheck = [f"{len(bad)} odour(s) differ: " + "; ".join(f"{o} {per[o]!r} ≠ {want.get(o)!r}" for o in bad[:3])
                   ] if bad else []
        e0 = e_rules.odour_activity({o: ae0[o]["frac"] for o in ae0})
        dec = v_rules.kc_band(rec112, ctx["cap_ok"], set_rec, recheck, sp)
        body = dict(dec, record_calib=rec112, record_set=set_rec, record_e0=dict(per_odour=e0, n_odours=len(e0),
                                                                                  strength=sp.e0_strength),
                    cap_ok=bool(ctx["cap_ok"]), seeds=list(sp.kc_seeds()), wall_s=wall,
                    note="V.5: 판정 세트의 냄새 입력만 쓰고 오라클 결과를 보지 않는다. E0 냄새 값은 기록이다(V.9.5 P2-11).")
        self._write("kc_band", body)
        return body

    # ---- the even pairs (V.3 7, V.9.3) ----------------------------------------------------------------------------
    def stage_even(self) -> dict:
        """C from R's even raw (h4 z) must reproduce c_even 7 first (STOP_EVEN_REPRO, L_V not measured); then L_V's 39
        even pairs on z_V (α on z_V), resumable, R's gate ③ validity (INVALID on a defect), ① the punishment filter, ②
        testable_b ≥ 11."""
        doc = self._require("even")
        sp = self.spec
        rows, seeds = self.ctx["even_rows"], sp.h4_seeds()
        keys = [row_key(r) for r in rows]
        nl, nc, _ = sp.cond_names
        r_c, r_l = self._r_even(rows, nc), self._r_even(rows, nl)
        z_h4 = _ztuple(self.ctx["z"])
        t_l = r_raw_spec(sp)
        C = r_records.cond_summary(r_c, sp.cond(nc), sp, z_h4, keys, seeds)
        L_h4 = r_records.cond_summary(r_l, t_l.cond(nl), t_l, z_h4, keys, seeds)
        l_h4 = (L_h4["aggregate"] or {}).get("testable_b")
        keep = r_records.KEEP
        base = dict(C={k: C.get(k) for k in keep}, L_h4_R=dict(testable_b=l_h4,
                                                                matches_r_gate3=bool(l_h4 == sp.r_even_L_h4)),
                    seeds=seeds, n_pairs=len(rows))
        stop = v_rules.even_repro(C, sp, l_h4)
        if stop:
            return self._write("even", dict(stop, **base))
        z_v = self._z_v(doc)
        m = self._measurer(z_v)
        got = m.oracle(rows, sp.cond(nl), "even", seeds)
        L = r_records.cond_summary(got, sp.cond(nl), sp, z_v, keys, seeds)
        v = v_rules.even_validity(L, C, sp, doc["reuse"]["records"]["repro_csc_sha256_none"])
        record = u_records.even_record(L, C, sp)
        c_even = (C["aggregate"] or {}).get("testable_b")
        dec = dict(outcome=v_rules.INVALID, reasons=v) if v else v_rules.even(record, c_even, sp)
        body = dict(dec, **base, c_even=c_even, record=record, L={k: L.get(k) for k in keep},
                    compare=r_records.compare(L, C, None, sp), pairs=L["pairs"],
                    z=[list(z_v[k]) for k in ("A", "P")], wall_s=m.last_wall_s,
                    note="V.9.3: 짝수 쌍은 POOL 안 상대라 판정 세트와 처벌 통과 분포가 다를 수 있다. 거름은 판정을 보증하지 "
                         "않는다.")
        self._write("even", body)
        return body

    # ---- smoke (V.3 8) -------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp = self.spec
        sm = smoke(sp)
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sm.p, True)
        items = p_items(sm.p, st, sp.lever_edit) + p_items(sm.p_c, st, sp.no_edit)
        rows = m.arms(items, self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke", sm.p.o.n)
        p_wall = m.last_wall_s
        p = {}
        for n, pspec, edit in (("L", sm.p, sp.lever_edit), ("C", sm.p_c, sp.no_edit)):
            rs = [r for r in rows if r["edit"] == edit]
            res = p_judge(rs, self.ctx["z"], c1["c1"], pspec)
            p[n] = dict(outcome=res["outcome"], label=res["label"], reasons=list(res.get("reasons", [])),
                        edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rs}),
                        csc_sha256=sorted({r["csc_sha256"] for r in rs}), n_rows=len(rs))
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            mc = self._m_for(cond.name, doc)
            got = mc.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self._z_for(cond.name, doc), [row_key(r) for r in sel],
                                       sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  edit=cond.edit, kc_median=s["kc_median"], saturation=s["saturation"],
                                  wall_s=mc.last_wall_s, jobs=mc.last_jobs, z={k: list(v) for k, v in mc.z.items()})
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        v_store.write_json(sp.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(p, orc, doc), cost=self._cost(p_wall, orc, sm),
                    detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, doc) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
        lever_sha = doc["path"]["sides"]["lever"]["csc_sha256"]
        bad = []
        if lever_sha != [sp.sha_combined]:
            bad.append(f"block path's L_V CSC {lever_sha} is not {sp.sha_combined}")
        if p["L"]["edit_edges"] != [sp.lever_edges] or p["L"]["csc_sha256"] != lever_sha:
            bad.append(f"P arms L: edges {p['L']['edit_edges']} on CSC {p['L']['csc_sha256']}, declared "
                       f"{sp.lever_edges} on {lever_sha}")
        if p["C"]["edit_edges"] != [0] or p["C"]["csc_sha256"] != [repro_sha]:
            bad.append(f"P arms C: edges {p['C']['edit_edges']} on CSC {p['C']['csc_sha256']}, declared none on R's "
                       f"repro")
        for n in ("L", "C"):
            if p[n]["outcome"] == P_INVALID:
                bad.append(f"P arms {n}: INVALID {p[n]['reasons'][:2]}")
        if orc[nl]["edit"] != sp.lever_edit or orc[nl]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"L: edit {orc[nl]['edit']} / edges {orc[nl]['edit_edges']}")
        if [orc[nl]["csc_sha256"]] != lever_sha:
            bad.append(f"L: CSC {orc[nl]['csc_sha256']} is not block path's L_V {lever_sha}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not R's repro {repro_sha}")
        if _ztuple(orc[nl]["z"]) != self._z_v(doc):
            bad.append(f"L's oracle ran on z {orc[nl]['z']}, not block z's z_V")
        for n in (nc, ne):
            if _ztuple(orc[n]["z"]) != _ztuple(self.ctx["z"]):
                bad.append(f"{n}'s oracle ran on z {orc[n]['z']}, not block h4's")
        for n in sp.cond_names:
            if orc[n]["reasons"]:
                bad.append(f"{n}: {orc[n]['reasons'][:2]}")
        return bad

    def _cost(self, p_wall, orc, sm) -> dict:
        """A rough estimate from smoke wall times: rounds of workers × the seed ratio (recorded, never a rule)."""
        sp = self.spec
        per = len(sp.p.directions) * len(sp.p.arms) * 2
        p_h = p_wall / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        n_sm = sum(len(v) for v in sm.even_seeds().values())
        jm = sum(len(v) for v in sp.judge_seeds().values()) / n_sm
        return dict(gate2_h=p_h, jm_per_condition_h=o_round * jm * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (V.6) — before gate ② ----------------------------------------------------
    def stage_oc(self) -> dict:
        """S.6's independent model (n_b 21 · n_a 43, V's notes) and T's cluster model on V's set clusters (block set),
        written to results/v/oc_cluster.json; the block records its file sha and table sha (V.9.5 P2-9)."""
        doc = self._require("oc")
        sp, s = self.spec, doc["set"]["set"]
        cl = v_rules.oc_cluster(s["clusters_b"], s["clusters_a"])
        p = v_store.write_json(sp.oc_cluster_out, cl, self.plist)
        body = dict(independent=v_rules.oc(sp), cluster={k: v for k, v in cl.items() if k != "rows"},
                    cluster_values=v_rules.cluster_values(cl), cluster_path=sp.oc_cluster_out,
                    cluster_file_sha256=sha256_file(p))
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7's gate ② OC on block h4's z (V.6), from P's committed block, before gate ②."""
        self._require("gate2_oc")
        pairs, point = P_SPEC.pairs(), [float(v) for v in P_SPEC.o.o2_point]
        wanted = {(d, a, int(s)) for d in P_SPEC.directions for a in P_SPEC.arms for s in P_SPEC.seeds}
        ref = self.ctx["p_ref"](wanted)
        if set(ref) != wanted:
            refuse(f"P's committed cache {self.spec.p_cache_dir} lacks {sorted(wanted - set(ref))[:3]}")
        rows = [dict(ref[k], direction=k[0], x=pairs[k[0]][0], y=pairs[k[0]][1], point=point) for k in sorted(wanted)]
        body = t_records.gate2_oc(rows, self.ctx["z"], P_SPEC, self.spec)
        self._write("gate2_oc", body)
        return body

    # ---- gate ② (V.3 10, V.9.2) ----------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """S's gate ② on 25_400_000+i with P_L(V) and P_C, both labelled on block h4's z; then V.9.2's z_V ratio.
        rerun: only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID run's."""
        prior = None
        if rerun:
            doc = self._require("gate2", allow_own=True)
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (V.3 10, R.5)")
            if not blk or blk.get("outcome") != v_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (V.3 10)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (V.3 10)")
            prior = blk
        else:
            doc = self._require("gate2")
        sp, z = self.spec, self.ctx["z"]
        z_v, z_h4 = self._z_v(doc), _ztuple(z)
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sp.p, False)
        items = p_items(sp.p, st, sp.lever_edit) + p_items(sp.p_c, st, sp.no_edit)
        rows = m.arms(items, self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2", sp.p.o.n)
        rows_l = [r for r in rows if r["edit"] == sp.lever_edit]
        rows_c = [r for r in rows if r["edit"] == sp.no_edit]
        res_l = p_judge(rows_l, z, c1["c1"], sp.p)
        res_c = p_judge(rows_c, z, c1["c1"], sp.p_c)
        ratio, ratio_zv, why = None, None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = t_records.ratio(rows_l, rows_c, z, sp)
                ratio_zv = v_records.ratio_two_z(rows_l, rows_c, z_v, z_h4, sp)
            except ValueError as e:
                why.append(str(e))
        dec = v_rules.gate2(res_l, res_c, ratio, ratio_zv, sp) if not why else dict(outcome=v_rules.INVALID,
                                                                                    reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [sp.lever_edges] or edges_c != [0]) and dec["outcome"] != v_rules.INVALID:
            dec = dict(outcome=v_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{sp.lever_edges} / 0"])
        zl = (None if dec["outcome"] == v_rules.INVALID
              else t_records.p_zlever(rows_l, z_v, c1["c1"], res_c, sp))
        body = dict(dec, ratio=ratio, ratio_z_V=ratio_zv, p_judgement_L=res_l, p_judgement_C=res_c, p_L_on_z_V=zl,
                    c1_source=c1, edit_edges_L=edges_l, edit_edges_C=edges_c, seeds=list(sp.p.seeds),
                    rerun_of=(prior or {}).get("written_at"), wall_s=m.last_wall_s, jobs=m.last_jobs)
        if prior is None:
            return self._write("gate2", body)
        return self._write_rerun(prior, body)

    def _write_rerun(self, prior: dict, body: dict) -> dict:
        doc = self._doc()
        if GATE2_INVALID in doc or doc.get("gate2") != prior:
            refuse("the summary changed during gate ②'s rerun; nothing written")
        block = self._stamp("gate2", body)
        doc[GATE2_INVALID], doc["gate2"] = prior, block
        v_store.write_json(self.summary_path, doc, self.plist)
        return block

    # ---- the judgement (V.3 11-13, V.5, V.9.5 P2-8) ----------------------------------------------------------------
    def _judge_chain(self, doc: dict, stage: str) -> None:
        """The judgement runs once, from clean trees, on the shared key every earlier block ran on and on the T and V
        measurement keys every earlier block carries (V.8: re-checked right before the judgement measurement and the
        judge — exit 7)."""
        if self.spec.smoke:
            refuse("smoke never measures the judgement set (V.3 8)")
        if not self.code_key:
            refuse("no code key given; the judgement must run on the code the gates ran on")
        if not self.u_measure_key or not self.t_measure_key:
            refuse(f"the V / T measurement key cannot be verified (V {self.u_measure_key}, T {self.t_measure_key}) "
                   f"(V.8)", EXIT_KEY)
        if self.u_measure_key != self.spec.u_measure_key_u:
            refuse(f"the V measurement key {self.u_measure_key} is not U's {self.spec.u_measure_key_u} (V.9.5)",
                   EXIT_KEY)
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("u_measure_key") != self.u_measure_key or doc[b].get("t_measure_key") != self.t_measure_key:
                refuse(f"block {b}'s V / T measurement key {doc[b].get('u_measure_key')} / {doc[b].get('t_measure_key')}"
                       f" is not the current {self.u_measure_key} / {self.t_measure_key} (V.8)", EXIT_KEY)
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the set is used once (V.5)")
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("code_key") != self.code_key:
                refuse(f"block {b}'s code key {doc[b].get('code_key')} is not the current {self.code_key}")
            d = (doc[b].get("git") or {}).get("dirty_hashed")
            if d is None or d:
                refuse(f"block {b} was written with dirty hashed files (or no git record): {d}")

    def _judgement_rows(self, doc: dict) -> list:
        return self._set_call("judgement_rows", doc)

    def stage_measurement_started(self) -> dict:
        """V.9.5 P2-8: the set-use marker, committed before the first judgement job (no pool): the set block's digests,
        the judgement seeds and z_V. From here on the V set is used (V.5)."""
        doc = self._require("measurement_started")
        self._judge_chain(doc, "measurement_started")
        sp, s = self.spec, doc["set"]["set"]
        body = dict(set={k: s[k] for k in ("digest_e0_b", "digest_e0_a", "digest_keys", "last_turn", "n_b", "n_a")},
                    seeds=sp.judge_seeds(), z_V=doc["z"]["z_V"], decision_pins=decision_pins(doc),
                    note="V.5 / V.9.5 P2-8: 판정 측정 첫 잡 전 표식 — 이 블록 뒤 V 세트는 사용된 것이다.")
        self._write("measurement_started", body)
        return body

    def stage_jm(self, name: str) -> dict:
        """One judgement condition on the 64 pairs (resumable), L_V on z_V and C / E0 on block h4's z; the block holds
        raw_check only."""
        sp = self.spec
        if name not in sp.cond_names:
            refuse(f"unknown condition {name}; V has {sp.cond_names}")
        stage = f"jm:{name}"
        doc = self._require(stage)
        self._judge_chain(doc, stage)
        rows = self._judgement_rows(doc)
        seeds, cond, m = sp.judge_seeds(), sp.cond(name), self._m_for(name, doc)
        got = m.oracle(rows, cond, r_records.JUDGE_BLOCK, seeds)
        rc = r_records.raw_check(got, cond, [row_key(r) for r in rows], seeds)
        man = [dict(x, sha256=sha256_file(x["cache_file"])) for x in rc["manifest"]]
        body = dict(rc, manifest=man, seeds=seeds, condition=name, edit=cond.edit,
                    z={k: list(v) for k, v in m.z.items()}, wall_s=m.last_wall_s, jobs=m.last_jobs)
        self._write(stage, body)
        return body

    def _raws(self, doc: dict) -> tuple:
        raws, bad = {}, []
        for n in self.spec.cond_names:
            got, b = v_store.load_manifest(doc[f"jm:{n}"]["manifest"])
            raws[n] = got
            bad += [f"{n}: {x}" for x in b]
        return raws, bad

    def stage_seal(self) -> dict:
        """R.9.7 as is (V.3 12): pre-read validity (every raw entry's stored inputs = its condition's measurer's, so
        L's carry L_V and z_V, C's / E0's "none" and h4 z), the manifest (192), the archive copy, the decision code hash
        with block z, block set and block kc_input pinned."""
        doc = self._require("seal")
        self._judge_chain(doc, "seal")
        sp = self.spec
        rows = self._judgement_rows(doc)
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        want = {n: r_records.judge_inputs(self._m_for(n, doc), rows, sp)[n] for n in sp.cond_names}
        v = r_records.preread_validity({n: doc[f"jm:{n}"] for n in sp.cond_names}, raws, sp,
                                       _ztuple(self.ctx["z"]), keys, self.code_key,
                                       doc["reuse"]["records"]["repro_csc_sha256_none"], want)
        nl = sp.cond_names[0]
        seeds = sp.judge_seeds()
        n_rep, n_act = len(seeds["report"]), len(seeds["act"])
        zv = self._z_v(doc)
        und = [g["key"] for g in raws[nl] if not r_records._short(g["result"], n_rep, n_act) and not r_records.stats_ok(
            r_records.pair_stats(g["result"]["report"], zv, sp.testable_min))]
        reasons = bad + v["reasons"] + ([f"{nl}: {len(und)} pair(s) with an undefined d′ on z_V"] if und else [])
        status = v_rules.INVALID if v["invalid"] else (v_rules.NOT_READ if reasons else v_rules.SEALED)
        manifest = [dict(x, condition=n) for n in sp.cond_names for x in doc[f"jm:{n}"]["manifest"]]
        s = doc["set"]["set"]
        body = dict(status=status, reasons=reasons, invalid=v["invalid"], checks=v["checks"], manifest=manifest,
                    n_files=len(manifest), archive=None, decision=dict(decision_key(), **decision_pins(doc)),
                    seeds=seeds, z=dict(z_V=doc["z"]["z_V"], z_h4={k: list(x) for k, x in _ztuple(self.ctx["z"]).items()}),
                    set={k: s[k] for k in ("digest_e0_b", "digest_e0_a", "digest_keys", "last_turn", "n_b", "n_a")})
        if status == v_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=v_store.archive_copy([x["cache_file"] for x in manifest], dest,
                                                                              self.archive_root))
        self._write("seal", body)
        return body

    def _read(self, doc: dict, mark=None) -> dict:
        """The bands and records from the sealed raw files (sha re-checked): L_V read on z_V, C and E0 on block h4's z
        (V.4); R.3's order with G_fail_S; V.7's sentence (V.9.4's cluster values in SELECTED); V.6's records."""
        sp = self.spec
        rows = self._judgement_rows(doc)
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        if bad:
            refuse(f"raw files changed since the seal: {bad[:3]}")
        if mark is not None:
            mark()                                       # the read starts here
        seeds = sp.judge_seeds()
        s = {n: r_records.cond_summary(raws[n], sp.cond(n), sp, self._z_for(n, doc), keys, seeds)
             for n in sp.cond_names}
        L, C, E0 = (s[n] for n in sp.cond_names)
        if L["aggregate"] is None or C["aggregate"] is None:
            refuse("no (b) pair to aggregate")
        aL, aC = L["aggregate"], C["aggregate"]
        gf = v_rules.g_fail_s(L["pairs"], C["pairs"], sp)
        rb = v_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                               aL["naive_a"], sp)
        if rb["band"] == v_rules.NOT_READ:
            return dict(status=v_rules.NOT_READ, band=rb["band"], reason=rb["reason"],
                        reasons=["the (b) / (a) pair counts differ from the declared counts (V.4 line 1)"])
        ax, ratio, ratio_zv = gf["axes"], doc["gate2"]["ratio"], doc["gate2"]["ratio_z_V"]
        names = list(sp.p.directions)
        k_even = doc["even"]["record"]["testable_b"]
        st = doc["set"]["set"]
        fields = dict(n=aL["testable_b"], c=aC["testable_b"], f_a=aL["F_a"], naive_a=aL["naive_a"], T=st["last_turn"],
                      k_even=k_even, k_cl=len(st["clusters_b"]), pb_L=ax["b"]["pun_L"], pb_C=ax["b"]["pun_C"],
                      pa_L=ax["a"]["pun_L"], pa_C=ax["a"]["pun_C"], d_b=ax["b"]["net_drop"], d_a=ax["a"]["net_drop"],
                      rho1=f"{ratio[names[0]]['ratio']:.3f}", rho2=f"{ratio[names[-1]]['ratio']:.3f}",
                      rho1_zv=f"{ratio_zv[names[0]]['ratio']:.3f}", rho2_zv=f"{ratio_zv[names[-1]]['ratio']:.3f}",
                      reason=rb["reason"], **doc["oc"]["cluster_values"])
        out = dict(band=rb["band"], reason=rb["reason"], n=aL["testable_b"], c=aC["testable_b"], F_a=aL["F_a"],
                   naive_a=aL["naive_a"], n_b=aL["n_b"], n_a=aL["n_a"], g_fail=gf)
        if rb["band"] == v_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        labels = self.ctx["clusters"](rows)
        records = dict(r_records.compare(L, C, E0, sp),
                       clusters=t_records.clusters({n: s[n]["pairs"] for n in sp.cond_names}, labels),
                       alpha_fixed=t_records.alpha_fixed(raws[sp.cond_names[0]], C, keys, seeds,
                                                         _ztuple(self.ctx["z"]), sp),
                       z=dict(z_V=doc["z"]["z_V"], record=doc["z"]["record"],
                              z_h4_reproduced=self._z_h4_reproduced(doc)),
                       path=dict(records=doc["path"]["records"],
                                 mech={k: v["mech"] for k, v in doc["path"]["sides"].items()}),
                       kc_input=dict(kc_record=doc["set"]["kc_record"], u_compare=doc["kc_input"]["u_compare"]),
                       gate2=dict(ratio_h4={d: ratio[d]["ratio"] for d in names},
                                  ratio_z_V={d: ratio_zv[d]["ratio"] for d in names}))
        return dict(out, status=v_rules.READ, sentence=v_rules.sentence(rb["band"], fields), records=records,
                    pairs={n: s[n]["pairs"] for n in sp.cond_names}, oc_sha256=doc["oc"]["independent"]["sha256"],
                    oc_cluster_sha256=doc["oc"]["cluster"]["sha256"],
                    p_labels=dict(L=doc["gate2"]["label_L"], C=doc["gate2"]["label_C"]),
                    p_ratio={d: ratio[d]["ratio"] for d in names}, k_even=k_even)

    def _z_h4_reproduced(self, doc: dict) -> bool:
        """From the reuse block: it passed and T's unedited z it recorded is block h4's z."""
        rb = doc.get("reuse") or {}
        nz = _dig(rb, ("records", "t_z", "none_z"))
        return bool(rb.get("outcome") == v_rules.PASS and isinstance(nz, dict)
                    and _ztuple(nz) == _ztuple(self.ctx["z"]))

    def _check_pins(self, doc: dict) -> None:
        """V.3 12: the live z, set and kc_input blocks are the ones the seal pinned; refuse otherwise."""
        sd = doc["seal"].get("decision") or {}
        moved = [k for k, v in decision_pins(doc).items() if sd.get(k) is None or sd.get(k) != v]
        if moved:
            refuse(f"the sealed decision record ({', '.join(moved)}) differs from the live blocks: the judgement reads "
                   f"only the z_V, set and KC input it was sealed with (V.3 12)")

    def stage_judge(self) -> dict:
        """Once (V.3 13). The marker is written before the band is computed. S.9.2 (V.5): with the marker and no judge
        block, judge is re-generated once — same sealed raw data (sha re-checked), the sealed decision code, no
        measurement; a second re-generation, a marker from another seal or decision code, or a judge block once written
        (DONE marker) refuses."""
        doc = self._require("judge")
        self._judge_chain(doc, "judge")
        if doc["seal"].get("status") != v_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: V reads only a sealed set (R.9.7)")
        sealed = (doc["seal"].get("decision") or {}).get("key")
        now = decision_key()["key"]
        if sealed != now:
            refuse(f"the decision code hash {now} is not the sealed {sealed}: the judgement reads only under the code "
                   f"it was sealed with (V.5)")
        self._check_pins(doc)
        if Path(DONE_MARKER).exists():
            refuse(f"{DONE_MARKER} exists: a judge block was written once; a discarded block does not reopen the set")
        resumed = None
        if Path(JUDGE_MARKER).exists():
            if Path(REREAD_MARKER).exists():
                refuse(f"{REREAD_MARKER} exists: judge was re-generated once after the mark already (V.5)")
            mk = json.loads(Path(JUDGE_MARKER).read_text())
            if mk.get("seal_written_at") != doc["seal"].get("written_at") or mk.get("decision_key") != sealed:
                refuse(f"{JUDGE_MARKER} belongs to another seal or decision code; no re-generation (V.5)")
            resumed = mk

        def mark():
            v_store.write_json(REREAD_MARKER if resumed else JUDGE_MARKER, dict(
                seal_written_at=doc["seal"].get("written_at"),
                seal_archive=(doc["seal"].get("archive") or {}).get("dir"),
                decision_key=now, code_key=self.code_key, t_measure_key=self.t_measure_key,
                u_measure_key=self.u_measure_key, pipeline_key=self.pipeline_key, read_at=_now()), self.plist)
        out = self._read(doc, mark)
        if out["status"] != v_rules.READ:
            return out                                   # NOT_READ: no block, the marker stays
        out = dict(out, resumed_after_mark=resumed is not None, mark_read_at=(resumed or {}).get("read_at"))
        block = self._write("judge", out)
        v_store.write_json(DONE_MARKER, dict(judge_written_at=block["written_at"]), self.plist)
        return block

    # ---- after reading (V.5 = U.5) ---------------------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (V.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: this set is closed (V.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """V.5 row 2: an analysis or summary defect after reading — the same sealed raw data recomputed."""
        if not note:
            refuse("--note is required (V.5: the correction is recorded)")
        doc = self._require_after_judge("recompute")
        self._check_pins(doc)                            # a decision-key change alone is recorded below, not refused
        out = self._read(doc)
        dk = decision_key()["key"]
        entry = dict(note=note, status=out["status"], band=out["band"], reason=out["reason"], n=out.get("n"),
                     c=out.get("c"), F_a=out.get("F_a"), naive_a=out.get("naive_a"), g_fail=out.get("g_fail"),
                     sentence=out.get("sentence"), decision_key=dk,
                     decision_changed_since_seal=bool(dk != (doc["seal"].get("decision") or {}).get("key")),
                     differs_from_judge=bool(out["band"] != doc["judge"]["band"]
                                             or out.get("sentence") != doc["judge"].get("sentence")),
                     code_key=self.code_key, t_measure_key=self.t_measure_key, u_measure_key=self.u_measure_key,
                     pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())
        v_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """V.5 row 3: a measurement defect after reading — INVALID_RUN; a replacement set is a new declaration."""
        if not note:
            refuse("--note is required (V.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=v_rules.INVALID_RUN, note=note, judge_band=doc["judge"]["band"],
                    rule="V.5: the same set is never run again; a replacement set needs a new declaration (user)")
        self._write("invalid_run", body)
        return body
