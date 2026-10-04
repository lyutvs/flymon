"""Spec U's stage chain (U.3 as ordered by U.9.5; plan Readings):
reuse -> path -> set -> scan -> kc -> even -> choose -> smoke -> oc -> gate2_oc -> gate2 -> jm:L -> jm:C -> jm:E0 ->
seal -> judge (and after judge only: recompute / invalid_run), one block each in results/summary/u_lever.json, written
only through u_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later
block exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or a
hashed U file is dirty. Every stage after `reuse` re-checks the reuse condition (R's shared key and repro block, T's
measurement key and unedited z) and refuses with SystemExit 7 when it broke; jm, seal and judge also re-check that every
earlier block carries the current U measurement key (U.8, exit 7). A smoke with problems or a gate whose outcome is not
PASS blocks every later stage, so the judgement set stays unused.
f and z (U.1, U.3): block scan holds z_f for every f passing the guard; block choose holds f*. L_f's oracle (the even
stage, smoke's L and jm:L) runs on a measurer built with z_f (α chosen on z_f), C's and E0's — and every P arm, KC
activity and the endpoint oracle, where U compares with R's raw on h4 z — on one built with block h4's z (`measure(z)`).
Every L-dependent call uses `u_spec.at(spec, f)`. Gate ② reads both P conditions on h4 z (T.9.3).
The judgement set is reached only through ctx["judgement_rows"] / ctx["set_odours"] (t_pairs: generated afresh and
checked against T.9.1's declared values on every call) from the KC band (odours only) and the judgement stages; `set`
reads the list once (no measurement). Blocks carry code_key (the shared measurement key), t_measure_key, u_measure_key
(which keys U's cache) and pipeline_key (U's own files)."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent.e_runner import summary_git
from . import r_records, t_records, t_rules, u_records, u_rules, u_store
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
from .t_pairs import check_t_set
from .t_pairs import summary as set_summary
from .t_runner import T_HASHED_FILES
from .t_spec import SPEC as T_SPEC
from .u_measure import U_MEASURE_FILES, u_edit
from .u_spec import at, smoke

ORDER = ("reuse", "path", "set", "scan", "kc", "even", "choose", "smoke", "oc", "gate2_oc", "gate2", "jm:L", "jm:C",
         "jm:E0", "seal", "judge")
GATES = ("reuse", "path", "set", "scan", "kc", "even", "choose", "gate2")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_KEY = 2, 7
U_PIPELINE_FILES = ("flymon/brain/u_spec.py", "flymon/brain/u_store.py", "flymon/brain/u_records.py",
                    "flymon/brain/u_rules.py", "flymon/brain/u_runner.py", "scripts/run_u.py")
U_HASHED_FILES = tuple(dict.fromkeys(T_HASHED_FILES + U_MEASURE_FILES + U_PIPELINE_FILES + (
    "results/summary/t_lever.json",)))
# The decision code: every hashed U file outside the U measurement key. The seal pins its hash; the judge reads only
# under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in U_HASHED_FILES
                       if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES and f not in U_MEASURE_FILES)
JUDGE_MARKER = "results/u/judge_read.json"
REREAD_MARKER = "results/u/judge_reread.json"
DONE_MARKER = "results/u/judge_done.json"


def git_state() -> dict:
    return _h3_git_state(files=U_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(U_PIPELINE_FILES)


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


def _ztuple(z: dict) -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in z.items()}


def fk(f) -> str:
    """The JSON key of an f (blocks key their per-f entries by it)."""
    return f"{float(f):.1f}"


def n_presentations() -> int:
    """H.3's reference set: n odours × 2 probe seeds each (96), the count every side must have."""
    return 2 * int(_H3.reference.n)


def r_raw_spec(spec):
    """The spec R's raw was measured under, for reading it by content key: U's numbers with T's (= R's) lever edit
    (U's own lever is unresolved until f is known, Reading 2)."""
    return dataclasses.replace(spec, lever_edit=spec.t_lever_edit)


def t_rows_reader(spec):
    """T's z rows (results/t/z.json, read only), refused unless the file's sha256 is the one T's z block recorded and U
    declares (U.9.1's comparison base)."""
    def t_rows() -> dict:
        p = Path(spec.t_z_detail)
        if not p.exists() or sha256_file(p) != spec.t_z_detail_sha256:
            raise ValueError(f"{spec.t_z_detail} is missing or its sha256 is not {spec.t_z_detail_sha256}")
        return json.loads(p.read_text())["rows"]
    return t_rows


def build_ctx(spec, npz: str) -> dict:
    """T's context (t_runner.build_ctx's: C3, block h4's z checked against z_h4, the H.4 pools as types, the encoder
    summary, the even rows, the reference set, T's set callables on T's own spec (Reading 2), R's even raw read only,
    P's entries and stimuli, R's summary and git state) plus R's 112 calibration odours, the ORN cap and encoder ③'s per-odour values (the KC band),
    the mechanism record types, T's summary and git state and T's z rows."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from ..agent.e_spec import SPEC as E
    from . import n_cli, odor_real, p_cli, q_pairs, r_pairs, t_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_spec import make_odors
    from .h3_store import code_key
    from .r_measure import RMeasurer
    from .t_runner import check_z_matches_spec, r_even_reader
    cfg = load_c3_config(spec.m0d_summary)
    if (cfg.readout["A"], cfg.readout["P"]) != (spec.a_type, spec.p_type):
        refuse(f"C3's readout is {cfg.readout}, not A {spec.a_type} / P {spec.p_type}")
    check_z_matches_spec(spec, cfg.z)
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    even = r_pairs.even_rows(pops, rc, enc, spec)
    p_doc = json.loads(Path(spec.p_summary).read_text())["p"]
    model_types = n_cli.model_types(npz)
    odors = make_odors(pops, _H3.reference)
    r_cache = u_store.RReadCache(spec.r_cache_dir, code_key(npz, files=R_MEASURE_FILES))
    r_m = RMeasurer(None, r_cache, spec, cfg.params, cfg.readout, cfg.z, types, len(pops.kc))

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                rec_types=list(dict.fromkeys(types + list(spec.chain_types))), even_rows=even, ref_odors=odors,
                probe_seeds=[int(s) for o in odors for s in o["seeds"]], ref_strength=float(_H3.strength),
                ref_window=(float(_H3.reference_window.settle_ms), int(_H3.reference_window.read_ms)),
                calib=r_pairs.calibration_odours(rc, cb, rule),
                cap_ok=r_pairs.cap_at(rc, cb, rule, spec.strength, cfg.params.max_rate_hz, odor_real.cap_hz(cfg.params)),
                enc_per=enc["strength"]["configs"][spec.config]["table"][str(spec.strength)]["per_odour"],
                t_set=lambda: t_pairs.t_set(pops, enc, T_SPEC, cfg.params),
                set_odours=lambda: t_pairs.set_odours(pops, rc, enc, T_SPEC, cfg.params),
                judgement_rows=lambda: t_pairs.judgement_rows(pops, rc, enc, T_SPEC, cfg.params),
                clusters=t_pairs.clusters, r_even=r_even_reader(r_cache, r_m, r_raw_spec(spec)),
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, r_doc=lambda: json.loads(Path(spec.r_summary).read_text()),
                r_git=lambda: summary_git(spec.r_summary), t_doc=lambda: json.loads(Path(spec.t_summary).read_text()),
                t_git=lambda: summary_git(spec.t_summary), t_rows=t_rows_reader(spec))


class Runner:
    def __init__(self, measure, zm, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, ucode: dict | None = None, pipeline: dict | None = None,
                 archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) on that z, one per z over one U pool and one UCache; zm: a
        u_measure.UZMeasurer (or its interface)."""
        self.measure, self.zm, self.ctx, self.spec = measure, zm, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.u_measure_key = (ucode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]
        self._ms = {}

    # ---- measurers per z, the spec per f -----------------------------------------------------------------------
    def _measurer(self, z: dict):
        k = tuple(sorted(z.items()))
        if k not in self._ms:
            self._ms[k] = self.measure(z)
        return self._ms[k]

    @property
    def m_h4(self):
        return self._measurer(_ztuple(self.ctx["z"]))

    def _z_f(self, doc, f) -> dict:
        return _ztuple(doc["scan"]["points"][fk(f)]["z"])

    def _f_star(self, doc) -> float:
        return float(doc["choose"]["f_star"])

    def _spf(self, doc):
        return at(self.spec, self._f_star(doc))

    def _z_for(self, name: str, doc) -> dict:
        return self._z_f(doc, self._f_star(doc)) if name == self.spec.cond_names[0] else _ztuple(self.ctx["z"])

    def _m_for(self, name: str, doc):
        return self._measurer(self._z_for(name, doc))

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return u_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed U files are dirty: {gs['dirty_hashed']}")

    def _reuse_dec(self) -> dict:
        c = self.ctx
        return u_rules.reuse(c["r_doc"](), c["r_git"](), c["t_doc"](), c["t_git"](), self.code_key,
                             self.t_measure_key, self.spec)

    def _reuse_now(self, stage: str) -> dict:
        r = self._reuse_dec()
        if r["outcome"] != u_rules.PASS:
            refuse(f"stage {stage}: the reuse condition broke ({'; '.join(r['reasons'])}) — U has no re-measurement "
                   f"path; U stops here and the judgement set stays unused", EXIT_KEY)
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
            refuse(f"stage {stage}: later block(s) {later} exist; U never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; U never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != u_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — U stops there and the "
                   f"judgement set stays unused (U.3)")
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
        u_store.write_summary_block(self.summary_path, stage, block, self.plist)
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
            refuse(f"U reads R's even raw by content key and it is not complete: {e}")

    # ---- reuse (U.3 1) — no pool ------------------------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        dec = self._reuse_dec()
        rd, td = self.ctx["r_doc"](), self.ctx["t_doc"]()
        rec = dict(repro_csc_sha256_none=_dig(rd, ("repro", "csc_sha256_none")),
                   r_smoke_L_csc_sha256=_dig(rd, ("smoke", "oracle", "L", "csc_sha256")),
                   r_gate3=dict(testable_b=_dig(rd, ("gate3", "testable_b")), c_even=_dig(rd, ("gate3", "c_even"))),
                   t_z=dict(outcome=_dig(td, ("z", "outcome")), none_z=_dig(td, ("z", "none", "z")),
                            lever_z=_dig(td, ("z", "lever", "z")), t_measure_key=_dig(td, ("z", "t_measure_key")),
                            detail_sha256=_dig(td, ("z", "detail_sha256"))))
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, t_measure_key_t=sp.t_measure_key_t,
                    r_summary=sp.r_summary, t_summary=sp.t_summary, r_commits=dict(sp.r_commits),
                    t_commits=dict(sp.t_commits), records=rec)
        self._write("reuse", body)
        return body

    # ---- the endpoint reproduction (U.9.1) ---------------------------------------------------------------------
    def stage_path(self) -> dict:
        """U's partial-edit path (U measurement key) at f = 1 and f = 0: the reference set and rest bit for bit equal to
        T's unedited / lever rows (CSC 1aee8398… / 860cba4f…), then the oracle on the first 3 even pairs (h4 z, H.4
        seeds) bit for bit equal to R's even C raw (f = 1) and L raw (f = 0), the edit labels aside. Any difference is
        STOP_U_PATH_REPRO; presentation or edge counts other than declared are INVALID. The f = 1 side is U's one
        unedited mechanism record (U.6, U.9.4 P2-7). Raw rows go to results/u/path.json."""
        self._require("path")
        sp, ctx = self.spec, self.ctx
        try:
            t_rows = ctx["t_rows"]()
        except ValueError as e:
            refuse(f"T's z rows cannot be read: {e}")
        n = n_presentations()
        raw, sides, checks = {}, {}, []
        for f, t_edit, sha in ((1.0, sp.t_none_edit, sp.sha_none), (0.0, sp.t_lever_edit, sp.sha_zero)):
            side = self._side(u_edit(f), raw, fk(f))
            sides[fk(f)] = side
            invalid = []
            if side["edit_edges"] != [sp.lever_edges]:
                invalid.append(f"f {fk(f)}: edges {side['edit_edges']}, declared {sp.lever_edges}")
            if (side["n_ref"], side["n_rest"]) != (n, n):
                invalid.append(f"f {fk(f)}: {side['n_ref']} reference / {side['n_rest']} rest, declared {n}")
            diffs = (u_records.row_diffs(raw[fk(f)]["ref"], t_rows[t_edit]["ref"], "기준 집합")
                     + u_records.row_diffs(raw[fk(f)]["rest"], t_rows[t_edit]["rest"], "휴지", rest=True))
            if side["csc_sha256"] != [sha]:
                diffs.append(f"CSC {side['csc_sha256']} ≠ {sha}")
            name = "T 편집 없는 엔진" if f == 1.0 else "T 지렛대 엔진"
            checks.append(dict(f=f, ref=f"{name} 기준 집합 행", diffs=diffs, invalid=invalid))
        rows = ctx["even_rows"][:sp.path_even_n]
        seeds, m = sp.h4_seeds(), self.m_h4
        nl, nc, _ = sp.cond_names
        for f, rname in ((0.0, nl), (1.0, nc)):
            r_got = self._r_even(rows, rname)
            u_got = m.oracle(rows, at(sp, f).cond(nl), "path", seeds)
            checks.append(dict(f=f, ref=f"R 짝수 {rname} 원자료 {len(rows)}쌍",
                               diffs=u_records.oracle_diffs(u_got, r_got, f"R 짝수 {rname}"), invalid=[]))
        dec = u_rules.path(checks, sp)
        p = u_store.write_json(sp.path_detail, dict(u_measure_key=self.u_measure_key, readout=ctx["readout"],
                                                    rows=raw), self.plist)
        body = dict(dec, checks=[dict(c, f=fk(c["f"])) for c in checks], sides=sides, pairs=[row_key(r) for r in rows],
                    n_presentations=n, detail_path=sp.path_detail, detail_sha256=sha256_file(p), wall_s=m.last_wall_s)
        self._write("path", body)
        return body

    # ---- the set (U.2 = T.9.1) — no pool ---------------------------------------------------------------------------
    def stage_set(self) -> dict:
        self._require("set")
        try:
            js = self.ctx["t_set"]()
        except ValueError as e:
            refuse(f"the T set cannot be generated: {e}")
        dec = u_rules.set_outcome(js)
        if dec["outcome"] == u_rules.PASS:
            bad = check_t_set(js, self.spec)
            if bad:
                refuse(f"the T set does not reproduce its declaration (U.2): {bad}")
        body = dict(dec, set=set_summary(js))
        self._write("set", body)
        return body

    # ---- the guard scan with mechanism records and the mechanism contrast (U.3 3 (a)(b), U.6, U.9.3) ------------
    def stage_scan(self) -> dict:
        """The 9 grid points: reference set + same-seed rest on L_f, T's z_lever decision per f (H.4's guard per readout
        type, SD > 0; edges 2 and 96 presentations, else INVALID), z_f, U.6's mechanism record and U.9.4 P2-6's ratios;
        then U.9.3's three contrast points (each INVALID — and left out of the readings — when its block edges are not
        the declared ones). Candidates = passing f; the smallest 3 are checked next (STOP_NO_QUALIFIED_F when none).
        Raw rows go to results/u/scan.json."""
        doc = self._require("scan")
        sp, ctx = self.spec, self.ctx
        ro, n, z_h4 = ctx["readout"], n_presentations(), _ztuple(self.ctx["z"])
        side1 = doc["path"]["sides"][fk(1.0)]
        side0 = doc["path"]["sides"][fk(0.0)]
        raw, points, decs = {}, {}, {}
        for f in sp.f_grid:
            spf = at(sp, f)
            s = self._side(spf.lever_edit, raw, fk(f))
            d = t_rules.z_lever(s, n, ro, spf)
            decs[f] = dict(d, guard=s["guard"])
            points[fk(f)] = dict(outcome=d["outcome"], reasons=d.get("reasons", []), failed=d.get("failed", []),
                                 z=s["z"] if d["outcome"] == u_rules.PASS else None, side=s,
                                 record=u_records.f_record(s, side1, z_h4, ro))
        contrast, declared = {}, sp.contrast_declared()
        for name, f, block in sp.contrast_points:
            s = self._side(u_edit(f, block), raw, name)
            want = canonical(declared[block])
            bad = []
            if s["block_edges"] != [want]:
                bad.append(f"{name}: block edges {s['block_edges']}, declared {want}")
            if s["edit_edges"] != [sp.lever_edges] or (s["n_ref"], s["n_rest"]) != (n, n):
                bad.append(f"{name}: edges {s['edit_edges']} / {s['n_ref']}·{s['n_rest']} presentations")
            contrast[name] = dict(f=f, block=block, invalid=bad, side=s)
        contrast["readings"] = self._contrast_readings(contrast, side1, side0)
        dec = u_rules.scan(decs, sp)
        p = u_store.write_json(sp.scan_detail, dict(u_measure_key=self.u_measure_key, readout=ro, rows=raw), self.plist)
        body = dict(dec, points=points, contrast=contrast, n_presentations=n, detail_path=sp.scan_detail,
                    detail_sha256=sha256_file(p))
        self._write("scan", body)
        return body

    def _contrast_readings(self, c: dict, side1: dict, side0: dict) -> dict:
        """U.9.3's fixed readings on MBON13's median Δ (record only; f = 0 / 1 alone are the endpoint sides)."""
        sp, a = self.spec, self.ctx["readout"]["A"]

        def dlt(s):
            return s["mech"]["types"][a]["median_delta"]
        d_none = dlt(side1) - dlt(side0)
        out = dict(d_none=d_none, delta_f0=dlt(side0), delta_f1=dlt(side1))
        if c["block_f1"]["invalid"] or c["block_f0"]["invalid"]:
            out["block"] = dict(reading=u_rules.INVALID)
        else:
            d_block = dlt(c["block_f1"]["side"]) - dlt(c["block_f0"]["side"])
            out["block"] = dict(d_block=d_block, reading=u_rules.contrast_block(d_block, d_none, sp))
        if c["entry_f0"]["invalid"]:
            out["entry"] = dict(reading=u_rules.INVALID)
        else:
            cut = c["entry_f0"]["side"]["mech"]["types"][a]
            out["entry"] = dict(median_delta=cut["median_delta"], zero_share=cut["zero_share"],
                                reading=u_rules.contrast_entry(cut, dlt(side0), sp))
        out["note"] = "U.9.3: 기록 전용 — 판정과 무관하다."
        return out

    # ---- the KC band per checked f (U.3 3 (c)) -----------------------------------------------------------------
    def stage_kc(self) -> dict:
        doc = self._require("kc")
        sp, ctx = self.spec, self.ctx
        odours, single, dual = ctx["calib"]
        try:
            set55 = ctx["set_odours"]()
        except ValueError as e:
            refuse(f"the T set does not reproduce its declaration: {e}")
        m = self.m_h4
        act_n = m.activity(set55, sp.no_edit, sp.strength, sp.kc_seeds(), "kc")
        wall = m.last_wall_s
        points, decs = {}, {}
        for f in [float(x) for x in doc["scan"]["checked"]]:
            spf = at(sp, f)
            a112 = m.activity(odours, spf.lever_edit, sp.strength, sp.kc_seeds(), "kc")
            wall += m.last_wall_s
            a55 = m.activity(set55, spf.lever_edit, sp.strength, sp.kc_seeds(), "kc")
            wall += m.last_wall_s
            r112 = r_records.gate1_record(a112, single, dual, ctx["enc_per"], spf)
            r55 = t_records.gate1s_record(a55, act_n, spf)
            d = u_rules.kc_point(r112, ctx["cap_ok"], r55, spf)
            decs[f] = d
            points[fk(f)] = dict(d, record_calib=r112, record_set=r55)
        dec = u_rules.kc(decs, doc["scan"], sp)
        body = dict(dec, points=points, cap_ok=bool(ctx["cap_ok"]), seeds=list(sp.kc_seeds()), wall_s=wall,
                    note="U.9.4 P2-8: KC 대역은 판정 세트의 냄새 입력만 쓰고 오라클 결과는 보지 않는다 — 판정 세트는 미사용으로 "
                         "남는다.")
        self._write("kc", body)
        return body

    # ---- the even pairs per qualified f (U.3 4) ----------------------------------------------------------------
    def stage_even(self) -> dict:
        """R's even raw for C (and R's L, a record) read by content key on block h4's z: C must reproduce c_even 7 first
        (STOP_EVEN_REPRO, no L_f measured); then each qualified f's 39 even pairs on z_f (α chosen on z_f), resumable,
        with R's gate ③ validity per f (INVALID on a defect)."""
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
        stop = u_rules.even_repro(C, sp, l_h4)
        if stop:
            return self._write("even", dict(stop, **base, per_f={}))
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
        per_f, bad, wall = {}, [], 0.0
        for f in [float(x) for x in doc["kc"]["qualified"]]:
            spf, z_f = at(sp, f), self._z_f(doc, f)
            m = self._measurer(z_f)
            got = m.oracle(rows, spf.cond(nl), "even", seeds)
            wall += m.last_wall_s
            L = r_records.cond_summary(got, spf.cond(nl), spf, z_f, keys, seeds)
            v = u_rules.even_validity(L, C, spf, repro_sha)
            bad += [f"f {fk(f)}: {x}" for x in v]
            per_f[fk(f)] = dict(record=u_records.even_record(L, C, spf), validity=v,
                                L={k: L.get(k) for k in keep}, compare=r_records.compare(L, C, None, spf),
                                pairs=L["pairs"], z=[list(z_f[k]) for k in ("A", "P")])
        dec = dict(outcome=u_rules.INVALID, reasons=bad) if bad else dict(outcome=u_rules.PASS, reasons=[])
        body = dict(dec, **base, c_even=(C["aggregate"] or {}).get("testable_b"), per_f=per_f, wall_s=wall)
        self._write("even", body)
        return body

    # ---- the f choice (U.3 5, U.9.2) — no pool -----------------------------------------------------------------
    def stage_choose(self) -> dict:
        doc = self._require("choose")
        ev = doc["even"]
        records = {float(k): v["record"] for k, v in ev["per_f"].items()}
        dec = u_rules.choose(records, ev["c_even"], self.spec)
        body = dict(dec, records={fk(f): r for f, r in sorted(records.items())},
                    z_f_star=None if dec.get("f_star") is None else doc["scan"]["points"][fk(dec["f_star"])]["z"])
        if dec.get("f_star") is not None:
            body["f_star"] = float(dec["f_star"])
        self._write("choose", body)
        return body

    # ---- smoke (U.3 6) ---------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        spf = self._spf(doc)
        sm = smoke(spf)
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sm.p, True)
        items = p_items(sm.p, st, spf.lever_edit) + p_items(sm.p_c, st, spf.no_edit)
        rows = m.arms(items, self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke", sm.p.o.n)
        p_wall = m.last_wall_s
        p = {}
        for n, pspec, edit in (("L", sm.p, spf.lever_edit), ("C", sm.p_c, spf.no_edit)):
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
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall, f_star=spf.f,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        u_store.write_json(spf.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(p, orc, doc, spf), cost=self._cost(p_wall, orc, sm, spf),
                    detail_path=spf.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, doc, spf) -> list:
        nl, nc, ne = spf.cond_names
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
        scan_sha = doc["scan"]["points"][fk(spf.f)]["side"]["csc_sha256"]
        bad = []
        if p["L"]["edit_edges"] != [spf.lever_edges]:
            bad.append(f"P arms L: lever edges {p['L']['edit_edges']}")
        if p["C"]["edit_edges"] != [0]:
            bad.append(f"P arms C: edges {p['C']['edit_edges']}, declared none")
        if p["L"]["csc_sha256"] != scan_sha:
            bad.append(f"P arms L: CSC {p['L']['csc_sha256']} is not the scan's L_f* {scan_sha}")
        for n in ("L", "C"):
            if p[n]["outcome"] == P_INVALID:
                bad.append(f"P arms {n}: INVALID {p[n]['reasons'][:2]}")
        if orc[nl]["edit"] != u_edit(spf.f) or orc[nl]["edit_edges"] != [spf.lever_edges]:
            bad.append(f"L: edit {orc[nl]['edit']} / edges {orc[nl]['edit_edges']}")
        if [orc[nl]["csc_sha256"]] != scan_sha:
            bad.append(f"L: CSC {orc[nl]['csc_sha256']} is not the scan's L_f* {scan_sha}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not R's repro {repro_sha}")
        if _ztuple(orc[nl]["z"]) != self._z_f(doc, spf.f):
            bad.append(f"L's oracle ran on z {orc[nl]['z']}, not block scan's z_f*")
        for n in (nc, ne):
            if _ztuple(orc[n]["z"]) != _ztuple(self.ctx["z"]):
                bad.append(f"{n}'s oracle ran on z {orc[n]['z']}, not block h4's")
        for n in spf.cond_names:
            if orc[n]["reasons"]:
                bad.append(f"{n}: {orc[n]['reasons'][:2]}")
        return bad

    def _cost(self, p_wall, orc, sm, spf) -> dict:
        """A rough estimate from smoke wall times: rounds of workers × the seed ratio (recorded, never a rule)."""
        per = len(spf.p.directions) * len(spf.p.arms) * 2
        p_h = p_wall / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(spf.p.seeds), spf.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        n_sm = sum(len(v) for v in sm.even_seeds().values())
        jm = sum(len(v) for v in spf.judge_seeds().values()) / n_sm
        return dict(gate2_h=p_h, jm_per_condition_h=o_round * jm * _rounds(spf.n_b + spf.n_a, spf.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (U.6) — before gate ② -------------------------------------------------
    def stage_oc(self) -> dict:
        """The independent model (S.6's, n_a 43, U's notes) and T's cluster model computed on T's own spec, equal to T's
        committed fixture and to its declared table sha 83a4d0bd… (U.6: recorded as is)."""
        self._require("oc")
        sp = self.spec
        cl = u_rules.oc_cluster(T_SPEC)
        fx = Path(_ROOT / sp.oc_cluster_fixture)
        if not fx.exists() or canonical(json.loads(fx.read_text())) != canonical(cl):
            refuse(f"the cluster OC differs from T's committed fixture {sp.oc_cluster_fixture} (U.6)")
        if cl["sha256"] != sp.oc_cluster_sha256:
            refuse(f"the cluster OC table sha {cl['sha256']} is not the declared {sp.oc_cluster_sha256} (U.6)")
        body = dict(independent=u_rules.oc(sp), cluster={k: v for k, v in cl.items() if k != "rows"},
                    cluster_fixture=sp.oc_cluster_fixture, cluster_fixture_sha256=sha256_file(fx))
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7's gate ② OC on block h4's z (U.6), from P's committed block, before gate ②."""
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

    # ---- gate ② (U.3 8 = T.9.3) ------------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """S's gate ② on 25_300_000+i with P_L(f*) and P_C, both read on block h4's z; P_L read on z_f* is a record.
        rerun: only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID run's."""
        prior = None
        if rerun:
            doc = self._require("gate2", allow_own=True)
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (U.3 8, R.5)")
            if not blk or blk.get("outcome") != u_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (U.3 8)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (U.3 8)")
            prior = blk
        else:
            doc = self._require("gate2")
        spf, z = self._spf(doc), self.ctx["z"]
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](spf.p, False)
        items = p_items(spf.p, st, spf.lever_edit) + p_items(spf.p_c, st, spf.no_edit)
        rows = m.arms(items, self.ctx["readout"], spf.p.o.n.h3.punish_type, "gate2", spf.p.o.n)
        rows_l = [r for r in rows if r["edit"] == spf.lever_edit]
        rows_c = [r for r in rows if r["edit"] == spf.no_edit]
        res_l = p_judge(rows_l, z, c1["c1"], spf.p)
        res_c = p_judge(rows_c, z, c1["c1"], spf.p_c)
        ratio, why = None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = t_records.ratio(rows_l, rows_c, z, spf)
            except ValueError as e:
                why.append(str(e))
        dec = u_rules.gate2(res_l, res_c, ratio, spf) if not why else dict(outcome=u_rules.INVALID, reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [spf.lever_edges] or edges_c != [0]) and dec["outcome"] != u_rules.INVALID:
            dec = dict(outcome=u_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{spf.lever_edges} / 0"])
        zl = (None if dec["outcome"] == u_rules.INVALID
              else t_records.p_zlever(rows_l, self._z_f(doc, spf.f), c1["c1"], res_c, spf))
        body = dict(dec, ratio=ratio, p_judgement_L=res_l, p_judgement_C=res_c, p_L_on_z_f=zl, f_star=spf.f,
                    c1_source=c1, edit_edges_L=edges_l, edit_edges_C=edges_c, seeds=list(spf.p.seeds),
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
        u_store.write_json(self.summary_path, doc, self.plist)
        return block

    # ---- the judgement (U.3 9-11, U.5) -------------------------------------------------------------------------
    def _judge_chain(self, doc: dict, stage: str) -> None:
        """The judgement runs once, from clean trees, on the shared key every earlier block ran on and on the T and U
        measurement keys every earlier block carries (U.8: re-checked right before the judgement measurement and the
        judge — exit 7)."""
        if self.spec.smoke:
            refuse("smoke never measures the judgement set (U.3 6)")
        if not self.code_key:
            refuse("no code key given; the judgement must run on the code the gates ran on")
        if not self.u_measure_key or not self.t_measure_key:
            refuse(f"the U / T measurement key cannot be verified (U {self.u_measure_key}, T {self.t_measure_key}) "
                   f"(U.8)", EXIT_KEY)
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("u_measure_key") != self.u_measure_key or doc[b].get("t_measure_key") != self.t_measure_key:
                refuse(f"block {b}'s U / T measurement key {doc[b].get('u_measure_key')} / {doc[b].get('t_measure_key')}"
                       f" is not the current {self.u_measure_key} / {self.t_measure_key} (U.8)", EXIT_KEY)
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the set is used once (U.5)")
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("code_key") != self.code_key:
                refuse(f"block {b}'s code key {doc[b].get('code_key')} is not the current {self.code_key}")
            d = (doc[b].get("git") or {}).get("dirty_hashed")
            if d is None or d:
                refuse(f"block {b} was written with dirty hashed files (or no git record): {d}")

    def _judgement_rows(self) -> list:
        try:
            return self.ctx["judgement_rows"]()
        except ValueError as e:
            refuse(f"the judgement set does not reproduce its declaration: {e}")

    def stage_jm(self, name: str) -> dict:
        """One judgement condition on the 64 pairs (resumable), L_f* on z_f* and C / E0 on block h4's z; the block holds
        raw_check only."""
        sp = self.spec
        if name not in sp.cond_names:
            refuse(f"unknown condition {name}; U has {sp.cond_names}")
        stage = f"jm:{name}"
        doc = self._require(stage)
        self._judge_chain(doc, stage)
        spf = self._spf(doc)
        rows = self._judgement_rows()
        seeds, cond, m = spf.judge_seeds(), spf.cond(name), self._m_for(name, doc)
        got = m.oracle(rows, cond, r_records.JUDGE_BLOCK, seeds)
        rc = r_records.raw_check(got, cond, [row_key(r) for r in rows], seeds)
        man = [dict(x, sha256=sha256_file(x["cache_file"])) for x in rc["manifest"]]
        body = dict(rc, manifest=man, seeds=seeds, condition=name, edit=cond.edit, f_star=spf.f,
                    z={k: list(v) for k, v in m.z.items()}, wall_s=m.last_wall_s, jobs=m.last_jobs)
        self._write(stage, body)
        return body

    def _raws(self, doc: dict) -> tuple:
        raws, bad = {}, []
        for n in self.spec.cond_names:
            got, b = u_store.load_manifest(doc[f"jm:{n}"]["manifest"])
            raws[n] = got
            bad += [f"{n}: {x}" for x in b]
        return raws, bad

    def stage_seal(self) -> dict:
        """R.9.7 as is (U.3 10): pre-read validity (every raw entry's stored inputs = its condition's measurer's, so
        L's carry u_edit(f*) and z_f*, C's / E0's "none" and h4 z), the manifest (192), the archive copy, the decision
        code hash pinned."""
        doc = self._require("seal")
        self._judge_chain(doc, "seal")
        spf = self._spf(doc)
        rows = self._judgement_rows()
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        want = {n: r_records.judge_inputs(self._m_for(n, doc), rows, spf)[n] for n in spf.cond_names}
        v = r_records.preread_validity({n: doc[f"jm:{n}"] for n in spf.cond_names}, raws, spf,
                                       _ztuple(self.ctx["z"]), keys, self.code_key,
                                       doc["reuse"]["records"]["repro_csc_sha256_none"], want)
        nl = spf.cond_names[0]
        seeds = spf.judge_seeds()
        n_rep, n_act = len(seeds["report"]), len(seeds["act"])
        zf = self._z_f(doc, spf.f)
        und = [g["key"] for g in raws[nl] if not r_records._short(g["result"], n_rep, n_act) and not r_records.stats_ok(
            r_records.pair_stats(g["result"]["report"], zf, spf.testable_min))]
        reasons = bad + v["reasons"] + ([f"{nl}: {len(und)} pair(s) with an undefined d′ on z_f*"] if und else [])
        status = u_rules.INVALID if v["invalid"] else (u_rules.NOT_READ if reasons else u_rules.SEALED)
        manifest = [dict(x, condition=n) for n in spf.cond_names for x in doc[f"jm:{n}"]["manifest"]]
        body = dict(status=status, reasons=reasons, invalid=v["invalid"], checks=v["checks"], manifest=manifest,
                    n_files=len(manifest), archive=None, decision=decision_key(), seeds=seeds, f_star=spf.f,
                    z=dict(z_f_star=doc["scan"]["points"][fk(spf.f)]["z"],
                           z_h4={k: list(x) for k, x in _ztuple(self.ctx["z"]).items()}),
                    set=dict(digest_e0_b=spf.digest_e0_b, digest_e0_a=spf.digest_e0_a, digest_keys=spf.digest_keys,
                             last_turn=spf.last_turn, n_b=spf.n_b, n_a=spf.n_a))
        if status == u_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=u_store.archive_copy([x["cache_file"] for x in manifest], dest,
                                                                              self.archive_root))
        self._write("seal", body)
        return body

    def _read(self, doc: dict, mark=None) -> dict:
        """The bands and records from the sealed raw files (sha re-checked): L_f* read on z_f*, C and E0 on block h4's z
        (U.4); R.3's order with G_fail_S; U.7's sentence; U.6's records."""
        spf = self._spf(doc)
        rows = self._judgement_rows()
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        if bad:
            refuse(f"raw files changed since the seal: {bad[:3]}")
        if mark is not None:
            mark()                                       # the read starts here
        seeds = spf.judge_seeds()
        s = {n: r_records.cond_summary(raws[n], spf.cond(n), spf, self._z_for(n, doc), keys, seeds)
             for n in spf.cond_names}
        L, C, E0 = (s[n] for n in spf.cond_names)
        if L["aggregate"] is None or C["aggregate"] is None:
            refuse("no (b) pair to aggregate")
        aL, aC = L["aggregate"], C["aggregate"]
        gf = u_rules.g_fail_s(L["pairs"], C["pairs"], spf)
        rb = u_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                               aL["naive_a"], spf)
        if rb["band"] == u_rules.NOT_READ:
            return dict(status=u_rules.NOT_READ, band=rb["band"], reason=rb["reason"],
                        reasons=["the (b) / (a) pair counts differ from the declared counts (U.4 line 1)"])
        ax, ratio = gf["axes"], doc["gate2"]["ratio"]
        names = list(spf.p.directions)
        k_even = doc["choose"]["testable_b"]
        fields = dict(n=aL["testable_b"], c=aC["testable_b"], f_a=aL["F_a"], naive_a=aL["naive_a"], T=spf.last_turn,
                      k_even=k_even, f=fk(spf.f), pb_L=ax["b"]["pun_L"], pb_C=ax["b"]["pun_C"], pa_L=ax["a"]["pun_L"],
                      pa_C=ax["a"]["pun_C"], d_b=ax["b"]["net_drop"], d_a=ax["a"]["net_drop"],
                      rho1=f"{ratio[names[0]]['ratio']:.3f}", rho2=f"{ratio[names[-1]]['ratio']:.3f}",
                      reason=rb["reason"])
        out = dict(band=rb["band"], reason=rb["reason"], n=aL["testable_b"], c=aC["testable_b"], F_a=aL["F_a"],
                   naive_a=aL["naive_a"], n_b=aL["n_b"], n_a=aL["n_a"], g_fail=gf, f_star=spf.f)
        if rb["band"] == u_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        labels = self.ctx["clusters"](rows)
        records = dict(r_records.compare(L, C, E0, spf),
                       clusters=t_records.clusters({n: s[n]["pairs"] for n in spf.cond_names}, labels),
                       alpha_fixed=t_records.alpha_fixed(raws[spf.cond_names[0]], C, keys, seeds,
                                                         _ztuple(self.ctx["z"]), spf),
                       z=dict(z_f_star=doc["scan"]["points"][fk(spf.f)]["z"],
                              f_record=doc["scan"]["points"][fk(spf.f)]["record"], z_h4_reproduced=True),
                       choose=dict(kept=doc["choose"]["kept"], checked=doc["choose"]["checked"]),
                       contrast=doc["scan"]["contrast"]["readings"])
        return dict(out, status=u_rules.READ, sentence=u_rules.sentence(rb["band"], fields), records=records,
                    pairs={n: s[n]["pairs"] for n in spf.cond_names}, oc_sha256=doc["oc"]["independent"]["sha256"],
                    oc_cluster_sha256=doc["oc"]["cluster"]["sha256"],
                    p_labels=dict(L=doc["gate2"]["label_L"], C=doc["gate2"]["label_C"]),
                    p_ratio={d: ratio[d]["ratio"] for d in names}, k_even=k_even)

    def stage_judge(self) -> dict:
        """Once (U.3 11). The marker is written before the band is computed. S.9.2 (U.5): with the marker and no judge
        block, judge is re-generated once — same sealed raw data (sha re-checked), the sealed decision code, no
        measurement; a second re-generation, a marker from another seal or decision code, or a judge block once written
        (DONE marker) refuses."""
        doc = self._require("judge")
        self._judge_chain(doc, "judge")
        if doc["seal"].get("status") != u_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: U reads only a sealed set (R.9.7)")
        sealed = (doc["seal"].get("decision") or {}).get("key")
        now = decision_key()["key"]
        if sealed != now:
            refuse(f"the decision code hash {now} is not the sealed {sealed}: the judgement reads only under the code "
                   f"it was sealed with (U.5)")
        if Path(DONE_MARKER).exists():
            refuse(f"{DONE_MARKER} exists: a judge block was written once; a discarded block does not reopen the set")
        resumed = None
        if Path(JUDGE_MARKER).exists():
            if Path(REREAD_MARKER).exists():
                refuse(f"{REREAD_MARKER} exists: judge was re-generated once after the mark already (U.5)")
            mk = json.loads(Path(JUDGE_MARKER).read_text())
            if mk.get("seal_written_at") != doc["seal"].get("written_at") or mk.get("decision_key") != sealed:
                refuse(f"{JUDGE_MARKER} belongs to another seal or decision code; no re-generation (U.5)")
            resumed = mk

        def mark():
            u_store.write_json(REREAD_MARKER if resumed else JUDGE_MARKER, dict(
                seal_written_at=doc["seal"].get("written_at"),
                seal_archive=(doc["seal"].get("archive") or {}).get("dir"),
                decision_key=now, code_key=self.code_key, t_measure_key=self.t_measure_key,
                u_measure_key=self.u_measure_key, pipeline_key=self.pipeline_key, read_at=_now()), self.plist)
        out = self._read(doc, mark)
        if out["status"] != u_rules.READ:
            return out                                   # NOT_READ: no block, the marker stays
        out = dict(out, resumed_after_mark=resumed is not None, mark_read_at=(resumed or {}).get("read_at"))
        block = self._write("judge", out)
        u_store.write_json(DONE_MARKER, dict(judge_written_at=block["written_at"]), self.plist)
        return block

    # ---- after reading (U.5 = T.5) ---------------------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (U.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: this set is closed (U.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """U.5 row 2: an analysis or summary defect after reading — the same sealed raw data recomputed."""
        if not note:
            refuse("--note is required (U.5: the correction is recorded)")
        doc = self._require_after_judge("recompute")
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
        u_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """U.5 row 3: a measurement defect after reading — INVALID_RUN; a replacement set is a new declaration."""
        if not note:
            refuse("--note is required (U.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=u_rules.INVALID_RUN, note=note, judge_band=doc["judge"]["band"],
                    rule="U.5: the same set is never run again; a replacement set needs a new declaration (user)")
        self._write("invalid_run", body)
        return body
