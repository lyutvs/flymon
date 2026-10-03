"""Spec T's stage chain (T.1-T.6, T.9; plan Readings 3-15):
reuse -> set -> z -> gate1s -> smoke -> oc -> gate2_oc -> gate2 -> gate3 -> jm:L -> jm:C -> jm:E0 -> seal -> judge
(and after judge only: recompute / invalid_run / rs_reread), one block each in results/summary/t_lever.json, written
only through t_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later
block exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or a
hashed T file is dirty. Every stage after `reuse` re-checks R's reuse condition (the shared measurement key equals
R's and R's repro / gate ① blocks are intact) and refuses with SystemExit 7 when it broke; jm, seal and judge also
re-check the T measurement key block z was measured under (T.9.6, exit 7). A smoke with problems or a gate whose
outcome is not PASS blocks every later stage, so the judgement set stays unused.
z (T.1, T.9.2-T.9.4): block z holds z_lever; L's oracle (gate ③'s even pairs and jm:L) runs on a measurer built with
z_lever, C's and E0's (and every P arm and KC activity, where z plays no part) on one built with block h4's z
(`measure(z)`, plan Reading 4). Gate ② reads both P conditions on h4 z (T.9.3).
The judgement set is reached only through ctx["judgement_rows"] / ctx["set_odours"] (t_pairs: generated afresh and
checked against T.9.1's declared values on every call) from the gate-① supplement and the judgement stages; `set`
reads the list once (no measurement); jm blocks carry no pair statistic. Blocks carry code_key (the shared measurement
key, which also keys the cache), t_measure_key (shared files + t_measure.py + h3_spec.py) and pipeline_key (T's own
files)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent.e_runner import summary_git
from . import r_records, t_records, t_rules, t_store
from .h3_store import ROOT as _ROOT
from .h3_spec import SPEC as _H3
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import p_items
from .s_runner import S_HASHED_FILES
from .t_measure import T_MEASURE_FILES
from .t_pairs import check_t_set
from .t_pairs import summary as set_summary
from .t_spec import smoke

ORDER = ("reuse", "set", "z", "gate1s", "smoke", "oc", "gate2_oc", "gate2", "gate3", "jm:L", "jm:C", "jm:E0", "seal",
         "judge")
GATES = ("reuse", "set", "z", "gate1s", "gate2", "gate3")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_KEY = 2, 7
T_PIPELINE_FILES = ("flymon/brain/t_spec.py", "flymon/brain/t_pairs.py", "flymon/brain/t_store.py",
                    "flymon/brain/t_records.py", "flymon/brain/t_rules.py", "flymon/brain/t_runner.py",
                    "scripts/run_t.py")
T_HASHED_FILES = tuple(dict.fromkeys(S_HASHED_FILES + T_MEASURE_FILES + T_PIPELINE_FILES + (
    "flymon/battle/pool.py", "results/summary/s_lever.json", "tests/brain/fixtures/t_oc_cluster.json")))
# The decision code: every hashed T file outside the T measurement key. The seal pins its hash; the judge reads only
# under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in T_HASHED_FILES if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES)
# Written before the band is computed: the set was read once. REREAD: S.9.2's one re-generation after the mark (T.5).
# DONE: written right after the judge block, so a judge block written and then discarded never reopens the set.
JUDGE_MARKER = "results/t/judge_read.json"
REREAD_MARKER = "results/t/judge_reread.json"
DONE_MARKER = "results/t/judge_done.json"


def git_state() -> dict:
    return _h3_git_state(files=T_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(T_PIPELINE_FILES)


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


def n_presentations() -> int:
    """H.3's reference set: n odours × 2 probe seeds each (96), the count every z side must have (T.9.6)."""
    return 2 * int(_H3.reference.n)


def r_even_reader(r_cache, r_m, spec):
    """R's even raw of one condition (block h4's z, H.4 seeds), read only: every entry is looked up with get() first
    and a missing one raises ValueError naming every missing pair — R's entries are never computed (r_m has no pool,
    and RReadCache.put refuses)."""
    def r_even(rows, name: str) -> list:
        cond, seeds = spec.cond(name), spec.h4_seeds()
        miss = [row_key(r) for r in rows if r_cache.get("r_oracle", r_m.inputs(r, cond, "even", seeds)) is None]
        if miss:
            raise ValueError(f"R's even raw for {name} lacks {len(miss)} pair(s) under the shared key: {miss}")
        return r_m.oracle(rows, cond, "even", seeds)
    return r_even


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, block h4's z), the H.4 pools as types, the encoder summary, the even rows, the H.3
    reference set and its probe seeds, lazy callables for T's set (list only), its odours (gate ①'s supplement), its
    judgement rows (checked) and cluster labels, R's even raw read only by content key (gate ③), P's committed entries
    (gate ②'s OC) and stimuli (c1 from block n1), R's summary with its git state (the reuse condition) and S's."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from . import n_cli, p_cli, q_pairs, r_pairs, t_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_spec import SPEC as H3
    from .h3_spec import make_odors
    from .h3_store import code_key
    from .r_measure import RMeasurer
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
    odors = make_odors(pops, H3.reference)
    r_cache = t_store.RReadCache(spec.r_cache_dir, code_key(npz, files=R_MEASURE_FILES))
    r_m = RMeasurer(None, r_cache, spec, cfg.params, cfg.readout, cfg.z, types, len(pops.kc))

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    r_even = r_even_reader(r_cache, r_m, spec)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                even_rows=even, ref_odors=odors, probe_seeds=[int(s) for o in odors for s in o["seeds"]],
                ref_strength=float(H3.strength), ref_window=(float(H3.reference_window.settle_ms),
                                                            int(H3.reference_window.read_ms)),
                t_set=lambda: t_pairs.t_set(pops, enc, spec, cfg.params),
                set_odours=lambda: t_pairs.set_odours(pops, rc, enc, spec, cfg.params),
                judgement_rows=lambda: t_pairs.judgement_rows(pops, rc, enc, spec, cfg.params),
                clusters=t_pairs.clusters, r_even=r_even,
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, r_doc=lambda: json.loads(Path(spec.r_summary).read_text()),
                r_git=lambda: summary_git(spec.r_summary), s_doc=lambda: json.loads(Path(spec.s_summary).read_text()))


class Runner:
    def __init__(self, measure, zm, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, pipeline: dict | None = None, archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) on that z, one per z over one pool and cache; zm: a
        t_measure.ZMeasurer (or its interface)."""
        self.measure, self.zm, self.ctx, self.spec = measure, zm, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]
        self._ms = {}

    def _measurer(self, z: dict):
        """One measurer per z for this runner's lifetime (plan Reading 4)."""
        k = tuple(sorted(z.items()))
        if k not in self._ms:
            self._ms[k] = self.measure(z)
        return self._ms[k]

    # ---- measurers per z (plan Reading 4) ----------------------------------------------------------------------------
    def _z_lever(self, doc=None) -> dict:
        doc = doc if doc is not None else self._doc()
        return _ztuple(doc["z"]["z_lever"])

    def _z_for(self, name: str, doc=None) -> dict:
        return self._z_lever(doc) if name == self.spec.cond_names[0] else _ztuple(self.ctx["z"])

    def _m_for(self, name: str, doc=None):
        return self._measurer(self._z_for(name, doc))

    @property
    def m_h4(self):
        return self._measurer(_ztuple(self.ctx["z"]))

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return t_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed T files are dirty: {gs['dirty_hashed']}")

    def _reuse_now(self, stage: str) -> dict:
        """T.3 1: the reuse condition, re-checked at every stage after `reuse`."""
        r = t_rules.reuse(self.ctx["r_doc"](), self.ctx["r_git"](), self.code_key, self.spec)
        if r["outcome"] != t_rules.PASS:
            refuse(f"stage {stage}: R's reuse condition broke ({'; '.join(r['reasons'])}) — T has no re-measurement "
                   f"path; T stops here and the judgement set stays unused (plan Reading 8)", EXIT_KEY)
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
            refuse(f"stage {stage}: later block(s) {later} exist; T never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; T never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != t_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — T stops there and the "
                   f"judgement set stays unused (T.3)")
        if i > 0:
            self._reuse_now(stage)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, t_measure_key=self.t_measure_key,
                    pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        t_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    # ---- reuse (T.3 1) and the set (T.2, T.9.1) — no pool ------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        rd = self.ctx["r_doc"]()
        dec = t_rules.reuse(rd, self.ctx["r_git"](), self.code_key, sp)

        def g(*ks):
            return _dig(rd, ks)
        rec = dict(gate1_median=g("gate1", "record", "median"), repro_csc_sha256_none=g("repro", "csc_sha256_none"),
                   r_smoke_L_csc_sha256=g("smoke", "oracle", "L", "csc_sha256"),
                   r_gate3=dict(testable_b=g("gate3", "testable_b"), c_even=g("gate3", "c_even")),
                   r_gate2_ell={d: g("gate2", "p_judgement", "directions", d, "ell") for d in sp.p.directions})
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, r_summary=sp.r_summary,
                    r_commits=dict(sp.r_commits), r_blocks={b: g(b, "written_at") for b in sp.r_reused}, records=rec)
        self._write("reuse", body)
        return body

    def stage_set(self) -> dict:
        """T.9.1, list only: STOP_SET_SHORT is recorded; any other mismatch with the declared values refuses."""
        self._require("set")
        try:
            js = self.ctx["t_set"]()
        except ValueError as e:
            refuse(f"the T set cannot be generated: {e}")
        dec = t_rules.set_outcome(js)
        if dec["outcome"] == t_rules.PASS:
            bad = check_t_set(js, self.spec)
            if bad:
                refuse(f"the T set does not reproduce its declaration (T.9.1): {bad}")
        body = dict(dec, set=set_summary(js))
        self._write("set", body)
        return body

    # ---- z (T.1, T.9.4, T.9.6, T.9.7) --------------------------------------------------------------------------------
    def stage_z(self) -> dict:
        """The unedited engine first, measured afresh (no cache): its z must equal block h4's bit for bit
        (STOP_Z_REPRO, the lever is then not measured); then the lever: H.4's readout guard per readout type and a
        nonzero SD (STOP_Z_DEGENERATE), else z_lever. Each side must hold H.3's 96 reference and 96 rest
        presentations (INVALID otherwise, the lever then unmeasured). Raw rows go to results/t/z.json."""
        doc = self._require("z")
        sp, ctx = self.spec, self.ctx
        ro, odors, seeds = ctx["readout"], ctx["ref_odors"], ctx["probe_seeds"]
        settle, steps = ctx["ref_window"]
        n = n_presentations()                       # 96 per engine, else INVALID (z_repro / z_lever)
        raw = {}

        def side(edit):
            ref = self.zm.reference(edit, odors, ctx["ref_strength"], settle, steps)
            rest = self.zm.rest(edit, seeds, settle, steps)
            raw[edit] = dict(ref=ref, rest=rest)
            return t_records.z_side(ref, rest, ro, sp)

        none = side(sp.no_edit)
        dec = t_rules.z_repro(none, n, doc["reuse"]["records"]["repro_csc_sha256_none"], sp)
        lever = None
        if dec["outcome"] == t_rules.PASS:
            lever = side(sp.lever_edit)
            dec = t_rules.z_lever(lever, n, ro, sp)
        p = t_store.write_json(sp.z_detail, dict(t_measure_key=self.t_measure_key, readout=ro, rows=raw), self.plist)
        body = dict(dec, none=none, lever=lever, z_h4={k: list(v) for k, v in sp.z_h4_dict().items()},
                    n_presentations=n, detail_path=sp.z_detail, detail_sha256=sha256_file(p))
        if dec["outcome"] == t_rules.PASS:
            body["z_lever"] = lever["z"]
            body["sd_ratio_lever_over_h4"] = t_records.z_ratios(lever["z"], sp.z_h4_dict())
        self._write("z", body)
        return body

    # ---- gate ① supplement (T.9.1) -----------------------------------------------------------------------------------
    def stage_gate1s(self) -> dict:
        self._require("gate1s")
        sp = self.spec
        try:
            odours = self.ctx["set_odours"]()
        except ValueError as e:
            refuse(f"the T set does not reproduce its declaration: {e}")
        m = self.m_h4
        act_l = m.activity(odours, sp.lever_edit, sp.strength, sp.kc_seeds(), "gate1s")
        wall = m.last_wall_s
        act_n = m.activity(odours, sp.no_edit, sp.strength, sp.kc_seeds(), "gate1s")
        rec = t_records.gate1s_record(act_l, act_n, sp)
        body = dict(t_rules.gate1s(rec, sp), record=rec, seeds=list(sp.kc_seeds()), wall_s=wall + m.last_wall_s)
        self._write("gate1s", body)
        return body

    # ---- smoke (T.3 4) ---------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
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
                        edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rs}), n_rows=len(rs))
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            mc = self._m_for(cond.name, doc)
            got = mc.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self._z_for(cond.name, doc), [row_key(r) for r in sel],
                                       sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  kc_median=s["kc_median"], saturation=s["saturation"], wall_s=mc.last_wall_s,
                                  jobs=mc.last_jobs, z={k: list(v) for k, v in mc.z.items()})
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        t_store.write_json(sp.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(p, orc, doc), cost=self._cost(p_wall, orc, sm),
                    detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, doc) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
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
        if _ztuple(orc[nl]["z"]) != self._z_lever(doc):
            bad.append(f"L's oracle ran on z {orc[nl]['z']}, not block z's z_lever")
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
        per = len(sp.p.directions) * len(sp.p.arms) * 2                      # L and C
        p_h = p_wall / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        n_sm = sum(len(v) for v in sm.even_seeds().values())
        jm = sum(len(v) for v in sp.judge_seeds().values()) / n_sm
        ev = sum(len(v) for v in sp.h4_seeds().values()) / n_sm
        return dict(gate2_h=p_h, gate3_L_h=o_round * ev * _rounds(len(self.ctx["even_rows"]), sp.workers) / 3600,
                    jm_per_condition_h=o_round * jm * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (T.6, T.9.6, T.9.7) — before gate ② -------------------------------------------
    def stage_oc(self) -> dict:
        """The independent model (S.6's, n_a 43) and the cluster-correlated model, the latter equal to its committed
        fixture (T.9.7: computed once, committed, re-checked here; the block records both sha256)."""
        self._require("oc")
        sp = self.spec
        cl = t_rules.oc_cluster(sp)
        fx = Path(_ROOT / sp.oc_cluster_fixture)
        if not fx.exists() or canonical(json.loads(fx.read_text())) != canonical(cl):
            refuse(f"the cluster OC differs from its committed fixture {sp.oc_cluster_fixture} (T.9.7)")
        body = dict(independent=t_rules.oc(sp), cluster={k: v for k, v in cl.items() if k != "rows"},
                    cluster_fixture=sp.oc_cluster_fixture, cluster_fixture_sha256=sha256_file(fx))
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7's gate ② OC on block h4's z (T.6, T.9.3), from P's committed block, before gate ②."""
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

    # ---- gate ② (T.3 6, T.9.3) ---------------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """S's gate ② on 25_200_000+i, both P conditions read on block h4's z (T.9.3); P_L read on z_lever is a record.
        rerun: only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID run's."""
        prior = None
        if rerun:
            doc = self._require("gate2", allow_own=True)
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (T.3 6, R.5)")
            if not blk or blk.get("outcome") != t_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (T.3 6)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (T.3 6)")
            prior = blk
        else:
            doc = self._require("gate2")
        sp, z = self.spec, self.ctx["z"]
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sp.p, False)
        items = p_items(sp.p, st, sp.lever_edit) + p_items(sp.p_c, st, sp.no_edit)
        rows = m.arms(items, self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2", sp.p.o.n)
        rows_l = [r for r in rows if r["edit"] == sp.lever_edit]
        rows_c = [r for r in rows if r["edit"] == sp.no_edit]
        res_l = p_judge(rows_l, z, c1["c1"], sp.p)
        res_c = p_judge(rows_c, z, c1["c1"], sp.p_c)
        ratio, why = None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = t_records.ratio(rows_l, rows_c, z, sp)
            except ValueError as e:
                why.append(str(e))
        dec = t_rules.gate2(res_l, res_c, ratio, sp) if not why else dict(outcome=t_rules.INVALID, reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [sp.lever_edges] or edges_c != [0]) and dec["outcome"] != t_rules.INVALID:
            dec = dict(outcome=t_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{sp.lever_edges} / 0"])
        zl = (None if dec["outcome"] == t_rules.INVALID
              else t_records.p_zlever(rows_l, self._z_lever(doc), c1["c1"], res_c, sp))
        body = dict(dec, ratio=ratio, p_judgement_L=res_l, p_judgement_C=res_c, p_L_on_z_lever=zl,
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
        t_store.write_json(self.summary_path, doc, self.plist)
        return block

    # ---- gate ③ (T.9.2) ----------------------------------------------------------------------------------------------
    def stage_gate3(self) -> dict:
        """R's even raw for C (and L, a record) read by content key on block h4's z: C must reproduce c_even 7 first
        (STOP_EVEN_REPRO, L is then not measured); then L's 39 even pairs re-measured on z_lever (α chosen on z_lever)
        and R's gate ③ rule (L testable_b ≥ 11, c_even 7)."""
        doc = self._require("gate3")
        sp = self.spec
        rows, seeds = self.ctx["even_rows"], sp.h4_seeds()
        keys = [row_key(r) for r in rows]
        nl, nc, _ = sp.cond_names
        try:
            r_c, r_l = self.ctx["r_even"](rows, nc), self.ctx["r_even"](rows, nl)
        except ValueError as e:
            refuse(f"gate ③ reads R's even raw by content key and it is not complete: {e}")
        z_h4 = _ztuple(self.ctx["z"])
        C = r_records.cond_summary(r_c, sp.cond(nc), sp, z_h4, keys, seeds)
        L_h4 = r_records.cond_summary(r_l, sp.cond(nl), sp, z_h4, keys, seeds)
        l_h4 = (L_h4["aggregate"] or {}).get("testable_b")
        L, m = None, None
        # the same condition as t_rules.gate3's STOP_EVEN_REPRO, so gate3 never sees L None after C reproduced
        if not C["reasons"] and (C["aggregate"] or {}).get("testable_b") == sp.c_even_expected:
            m = self._m_for(nl, doc)
            got = m.oracle(rows, sp.cond(nl), "even", seeds)
            L = r_records.cond_summary(got, sp.cond(nl), sp, self._z_lever(doc), keys, seeds)
        dec = t_rules.gate3(L, C, sp, doc["reuse"]["records"]["repro_csc_sha256_none"], l_h4)
        keep = r_records.KEEP
        body = dict(dec, L_h4_R=dict(testable_b=l_h4, matches_r_gate3=bool(l_h4 == sp.r_even_L_h4),
                                     **{k: L_h4.get(k) for k in ("aggregate", "counts")}),
                    C={k: C.get(k) for k in keep}, L=None if L is None else {k: L.get(k) for k in keep},
                    records=None if L is None else r_records.compare(L, C, None, sp), seeds=seeds,
                    n_pairs=len(rows), wall_s=None if m is None else m.last_wall_s,
                    jobs=None if m is None else m.last_jobs)
        self._write("gate3", body)
        return body
