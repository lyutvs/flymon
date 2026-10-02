# flymon/brain/r_runner.py
"""Spec R's stage chain (R.2-R.5, R.9; plan Readings 4, 6, 10, 14-18):
repro -> smoke -> oc -> gate1 -> gate2 -> even -> gate3 -> jm:L -> jm:C -> jm:E0 -> seal -> judge
(and after judge only: recompute / invalid_run), one block each in results/summary/r_lever.json, written only through
r_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists,
its own block exists (a failed repro and gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or
a hashed R file is dirty; a later stage after a failed repro refuses with SystemExit 4 (Reading 4). A smoke with
problems or a gate whose outcome is not PASS blocks every later stage, so the judgement set stays unused (R.2). The
judgement set is reached only through ctx["judgement_rows"] (r_pairs.judgement_rows) from the judgement stages; jm
blocks carry no pair statistic (Reading 10)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from statistics import median

import numpy as np

from ..agent.e_measure import MAX_ITEMS
from ..agent.e_runner import summary_git
from . import p_measure, r_records, r_rules, r_store
from .h3_store import canonical
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_spec import smoke

ORDER = ("repro", "smoke", "oc", "gate1", "gate2", "even", "gate3", "jm:L", "jm:C", "jm:E0", "seal", "judge")
GATES = ("gate1", "gate2", "gate3")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_REPRO = 2, 4
R_HASHED_FILES = tuple(dict.fromkeys(R_MEASURE_FILES + tuple(p_measure.HASHED_FILES) + (
    "flymon/brain/r_spec.py", "flymon/brain/r_pairs.py", "flymon/brain/r_records.py", "flymon/brain/r_rules.py",
    "flymon/brain/r_runner.py", "scripts/run_r.py", "flymon/agent/e_pairs.py", "flymon/agent/encode_grid.py",
    "flymon/agent/e_codebook.py", "flymon/agent/e_spec.py", "flymon/agent/e_rules.py", "flymon/agent/e_runner.py",
    "flymon/agent/config.py", "flymon/brain/h4_rules.py", "flymon/brain/h4_pairs.py", "flymon/brain/h4_spec.py",
    "flymon/brain/l_pairs.py", "flymon/brain/d6a.py", "flymon/brain/odor_real.py", "flymon/brain/q_pairs.py",
    "flymon/brain/q_spec.py", "results/summary/m0d.json")))


def git_state() -> dict:
    return _h3_git_state(files=R_HASHED_FILES)


def refuse(msg: str, code: int = EXIT_REFUSE):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def p_items(pspec, st: dict, edit: str, seeds=None) -> list:
    """scripts/run_p.py:items_p's items (test: equal for P's spec with edit "none") with the edit and seeds given."""
    point = tuple(float(v) for v in pspec.o.o2_point)
    flags = pspec.flags()
    seeds = pspec.seeds if seeds is None else seeds
    return [dict(direction=d, x=x, y=y, edit=edit, arm=a, punish=flags[a][0], plastic=flags[a][1],
                 da_zero=flags[a][2], odor_x=st[x]["odor"], odor_y=st[y]["odor"], seed=int(s), point=point)
            for d, (x, y) in pspec.pairs().items() for a in pspec.arms for s in seeds]


def _rounds(n: int, w: int) -> int:
    return -(-int(n) // max(1, int(w)))


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, z — block h4), the H.4 pools as types, the encoder summary, the calibration odours and the
    ORN cap at s, the even rows, and lazy callables for the judgement rows (digest-checked), the encoder ③ reference
    names, P's reference entries and stimuli (c1 from block n1), and Q0's files."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from ..agent.e_runner import MEASURE_FILES_E
    from ..agent.e_spec import SPEC as E
    from . import n_cli, odor_real, p_cli, q_pairs, r_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_store import code_key
    from .q_spec import SPEC as Q_SPEC
    cfg = load_c3_config(spec.m0d_summary)
    if (cfg.readout["A"], cfg.readout["P"]) != (spec.a_type, spec.p_type):
        refuse(f"C3's readout is {cfg.readout}, not A {spec.a_type} / P {spec.p_type}")
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    calib = r_pairs.calibration_odours(rc, cb, rule)
    cap = r_pairs.cap_at(rc, cb, rule, spec.strength, cfg.params.max_rate_hz, odor_real.cap_hz(cfg.params))
    even = r_pairs.even_rows(pops, rc, enc, spec)
    enc_code = code_key(npz, files=MEASURE_FILES_E)
    p_doc = json.loads(Path(spec.p_summary).read_text())["p"]
    model_types = n_cli.model_types(npz)

    def kc_ref(oid):
        if enc_code["key"] != enc["strength"]["code_key"]:
            return None
        return r_pairs.encoder_activity_ref(oid, calib[0][oid], cfg.params, len(pops.kc), enc_code, spec)

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                calib=calib, cap_ok=cap, even_rows=even,
                judgement_rows=lambda: r_pairs.judgement_rows(pops, rc, enc, spec), kc_ref=kc_ref, p_block=p_doc,
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, q0_ref=lambda row: q_pairs.q0_files([row], enc, Q_SPEC)[0])


class Runner:
    def __init__(self, measurer, ctx: dict, spec, summary_path=None, code: dict | None = None, archive_root=None):
        self.m, self.ctx, self.spec = measurer, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return r_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first "
                       f"(a failed repro block: commit it or discard it before rerunning)")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed R files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str, allow_own: bool = False) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; R never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; R never rewrites a recorded block")
        if i > 0 and not doc["repro"].get("passed"):
            refuse(f"stage {stage}: the no-edit reproduction gate did not pass (R.9.6)", EXIT_REPRO)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != r_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — R stops there and the "
                   f"judgement set stays unused (R.2)")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, git=git_state(),
                    written_at=_dt.datetime.now(_dt.timezone.utc).isoformat())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        r_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    # ---- repro (R.9.6, Reading 4) ----------------------------------------------------------------------------------
    def stage_repro(self) -> dict:
        doc = self._doc()
        self._require("repro", allow_own="repro" in doc and not doc["repro"].get("passed"))
        kc, p, orc = self._repro_kc(), self._repro_p(), self._repro_oracle()
        body = dict(passed=bool(kc["passed"] and p["passed"] and orc["passed"]), kc=kc, p=p, oracle=orc,
                    csc_sha256_none=orc["csc_sha256"])
        self._write("repro", body)
        return body

    def _repro_kc(self) -> dict:
        sp = self.spec
        odours = self.ctx["calib"][0]
        names = {oid: self.ctx["kc_ref"](oid) for oid in sp.kc_repro_odours}
        if any(v is None for v in names.values()):
            refuse("the encoder code key differs from block strength's: encoder ③'s activity entries cannot be named")
        missing = sorted(n for n in names.values() if not (Path(sp.ref_dir) / n).exists())
        if missing:
            refuse(f"encoder ③'s activity entries {missing} are not under {sp.ref_dir}/: copy them from the "
                   f"encoder-redesign worktree's results/encoder/cache/activity/ (plan Runs step 0; --list-refs)")
        got = self.m.activity({o: odours[o] for o in names}, sp.no_edit, sp.strength, sp.kc_seeds(), "repro")
        per = self.ctx["enc"]["strength"]["configs"][sp.config]["table"][str(sp.strength)]["per_odour"]
        rows = []
        for oid, name in names.items():
            ref = json.loads((Path(sp.ref_dir) / name).read_text())["result"]
            mine = dict(frac=got[oid]["frac"], max_win=got[oid]["max_win"])
            rows.append(dict(odour=oid, file=name, equal=canonical(mine) == canonical(ref),
                             median_matches_block=float(median(mine["frac"])) == float(per[oid]),
                             edit_edges=got[oid]["edit_edges"]))
        ok = all(r["equal"] and r["median_matches_block"] and r["edit_edges"] == [0] for r in rows)
        return dict(passed=bool(ok), odours=rows, seeds=list(sp.kc_seeds()), wall_s=self.m.last_wall_s)

    def _repro_p(self) -> dict:
        sp = self.spec
        _, st = self.ctx["p_inputs"](P_SPEC, False)
        items = p_items(P_SPEC, st, sp.no_edit, sp.p_repro_seeds())
        wanted = {(i["direction"], i["arm"], int(i["seed"])) for i in items}
        ref = self.ctx["p_ref"](wanted)
        if set(ref) != wanted:
            refuse(f"P's committed cache {sp.p_cache_dir} lacks {sorted(wanted - set(ref))[:3]} under block p's key")
        rows = self.m.arms(items, self.ctx["readout"], P_SPEC.o.n.h3.punish_type, "repro", P_SPEC.o.n)
        out = []
        for r in rows:
            k = (r["direction"], r["arm"], int(r["seed"]))
            out.append(dict(key=list(k), edit_edges=int(r["r"]["edit_edges"]),
                            equal=canonical(r_records.strip_job(r)) == canonical(r_records.strip_job(ref[k]))))
        ok = len(out) == len(wanted) and all(o["equal"] and o["edit_edges"] == 0 for o in out)
        return dict(passed=bool(ok), rows=out, seeds=list(sp.p_repro_seeds()), wall_s=self.m.last_wall_s)

    def _repro_oracle(self) -> dict:
        sp = self.spec
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        rows = [b[i] for i in sp.oracle_repro_idx]
        got = self.m.oracle(rows, sp.cond(sp.cond_names[1]), "repro", sp.h4_seeds())
        out = []
        for r, g in zip(rows, got):
            f = self.ctx["q0_ref"](r)
            ref = json.loads(Path(f["path"]).read_text())["result"] if f["ok"] else None
            res = g["result"]
            mine = r_records.strip_job(res, top=("q", "r"))
            out.append(dict(key=row_key(r), q0_sha_ok=bool(f["ok"]),
                            equal=bool(ref is not None and canonical(mine) == canonical(ref)),
                            cell_sums_ok=r_records.cell_sums_ok(res), edit_edges=int(res["q"]["edit_edges"]),
                            csc_sha256=res["q"]["csc_sha256"]))
        shas = sorted({o["csc_sha256"] for o in out})
        ok = len(shas) == 1 and all(o["q0_sha_ok"] and o["equal"] and o["cell_sums_ok"] and o["edit_edges"] == 0
                                    for o in out)
        return dict(passed=bool(ok), pairs=out, csc_sha256=shas[0] if ok else None, seeds=sp.h4_seeds(),
                    wall_s=self.m.last_wall_s)

    # ---- smoke (R.8) -----------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        odours = self.ctx["calib"][0]
        ids = sorted(odours)[:sm.smoke_odours]
        act = self.m.activity({o: odours[o] for o in ids}, sp.lever_edit, sp.strength, sm.kc_seeds(), "smoke")
        kc = dict(odours=ids, edit_edges=sorted({e for v in act.values() for e in v["edit_edges"]}),
                  csc_sha256=sorted({s for v in act.values() for s in v["csc_sha256"]}),
                  median=float(np.median([x for v in act.values() for x in v["frac"]])), wall_s=self.m.last_wall_s)
        c1, st = self.ctx["p_inputs"](sm.p, True)
        rows = self.m.arms(p_items(sm.p, st, sp.lever_edit), self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke",
                           sm.p.o.n)
        res = p_judge(rows, self.ctx["z"], c1["c1"], sm.p)
        p = dict(outcome=res["outcome"], label=res["label"], reasons=list(res.get("reasons", [])),
                 edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rows}), n_rows=len(rows),
                 wall_s=self.m.last_wall_s)
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            got = self.m.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self.ctx["z"], [row_key(r) for r in sel], sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  kc_median=s["kc_median"], saturation=s["saturation"], wall_s=self.m.last_wall_s,
                                  jobs=self.m.last_jobs)
        detail = dict(kc=kc, p=p, oracle=orc, pairs=[row_key(r) for r in sel],
                      seeds=dict(kc=list(sm.kc_seeds()), p=list(sm.p.seeds), oracle=sm.even_seeds()))
        r_store.write_json(sp.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(kc, p, orc, doc["repro"]["csc_sha256_none"]),
                    cost=self._cost(kc, p, orc, sm), detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, kc, p, orc, repro_sha) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        bad = []
        if kc["edit_edges"] != [sp.lever_edges] or len(kc["csc_sha256"]) != 1:
            bad.append(f"KC activity: lever edges {kc['edit_edges']} on CSC {kc['csc_sha256']}")
        if p["edit_edges"] != [sp.lever_edges]:
            bad.append(f"P arms: lever edges {p['edit_edges']}")
        if p["outcome"] == P_INVALID:
            bad.append(f"P arms: INVALID {p['reasons'][:2]}")
        if orc[nl]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"L: lever edges {orc[nl]['edit_edges']}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not the repro's {repro_sha}")
        for n in sp.cond_names:
            if orc[n]["reasons"]:
                bad.append(f"{n}: {orc[n]['reasons'][:2]}")
        return bad

    def _cost(self, kc, p, orc, sm) -> dict:
        """A rough estimate from smoke wall times: rounds of workers × the seed ratio (recorded, never a rule)."""
        sp = self.spec
        item_s = kc["wall_s"] / max(1, len(kc["odours"]) * len(sm.kc_seeds()))
        kc_h = item_s * MAX_ITEMS * _rounds(_rounds(sp.n_calib * len(sp.kc_seeds()), MAX_ITEMS), sp.workers) / 3600
        per = len(sp.p.directions) * len(sp.p.arms)
        p_h = p["wall_s"] / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        ratio = sum(len(v) for v in sp.h4_seeds().values()) / sum(len(v) for v in sm.even_seeds().values())
        return dict(gate1_h=kc_h, gate2_h=p_h,
                    even_h=o_round * ratio * _rounds(sp.n_b + sp.n_a_even, sp.workers) * len(sp.cond_names) / 3600,
                    jm_per_condition_h=o_round * ratio * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristic (R.6, R.9.5) — before any gate ---------------------------------------------
    def stage_oc(self) -> dict:
        self._require("oc")
        body = r_rules.oc(self.spec)
        self._write("oc", body)
        return body

    # ---- gate ① (R.9.2) ----------------------------------------------------------------------------------------------
    def stage_gate1(self) -> dict:
        self._require("gate1")
        sp = self.spec
        odours, single, dual = self.ctx["calib"]
        act = self.m.activity(odours, sp.lever_edit, sp.strength, sp.kc_seeds(), "gate1")
        per = self.ctx["enc"]["strength"]["configs"][sp.config]["table"][str(sp.strength)]["per_odour"]
        rec = r_records.gate1_record(act, single, dual, per, sp)
        body = dict(r_rules.gate1(rec, self.ctx["cap_ok"], sp), record=rec, cap_ok=bool(self.ctx["cap_ok"]),
                    seeds=list(sp.kc_seeds()), wall_s=self.m.last_wall_s)
        self._write("gate1", body)
        return body

    # ---- gate ② (R.2, R.5, Reading 3) -----------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """rerun (R.5, P.6.5): only an INVALID gate2 block, only once (no gate2_invalid block yet), only with a code
        key other than the INVALID run's. The INVALID block and the new one are written together in one summary write
        after the measurement, so an interrupted rerun leaves the summary as it was."""
        prior = None
        if rerun:
            doc = self._doc()
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (R.5, P.6.5)")
            if not blk or blk.get("outcome") != r_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (R.5)")
            if blk.get("code_key") == self.code_key:
                refuse("the code key equals the INVALID run's: fix the code first (R.5, P.6.5)")
            self._require("gate2", allow_own=True)
            prior = blk
        else:
            self._require("gate2")
        sp = self.spec
        c1, st = self.ctx["p_inputs"](sp.p, False)
        rows = self.m.arms(p_items(sp.p, st, sp.lever_edit), self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2",
                           sp.p.o.n)
        res = p_judge(rows, self.ctx["z"], c1["c1"], sp.p)
        dec = r_rules.gate2(res, sp)
        edges = sorted({int(r["r"]["edit_edges"]) for r in rows})
        if edges != [sp.lever_edges] and dec["outcome"] != r_rules.INVALID:
            dec = dict(outcome=r_rules.INVALID, label=dec.get("label"),
                       reasons=[f"the lever changed {edges} CSC edges, declared {sp.lever_edges}"])
        body = dict(dec, p_judgement=res, c1_source=c1, edit_edges=edges, seeds=list(sp.p.seeds),
                    rerun_of=(prior or {}).get("written_at"), wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        if prior is None:
            self._write("gate2", body)
            return body
        return self._write_rerun(prior, body)

    def _write_rerun(self, prior: dict, body: dict) -> dict:
        doc = self._doc()
        if GATE2_INVALID in doc or doc.get("gate2") != prior:
            refuse("the summary changed during gate ②'s rerun; nothing written")
        block = self._stamp("gate2", body)
        doc[GATE2_INVALID], doc["gate2"] = prior, block
        r_store.write_json(self.summary_path, doc, self.plist)
        return block

    # ---- gate ③ (R.9.3, R.9.6, Reading 6) -----------------------------------------------------------------------------
    def stage_even(self) -> dict:
        self._require("even")
        sp = self.spec
        rows = self.ctx["even_rows"]
        keys, seeds = [row_key(r) for r in rows], sp.even_seeds()
        out = {}
        for cond in sp.conditions():
            got = self.m.oracle(rows, cond, "even", seeds)
            out[cond.name] = dict(r_records.cond_summary(got, cond, sp, self.ctx["z"], keys, seeds),
                                  wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        body = dict(conditions=out, seeds=seeds, n_pairs=len(rows))
        self._write("even", body)
        return body

    def stage_gate3(self) -> dict:
        doc = self._require("gate3")
        sp = self.spec
        ev = doc["even"]["conditions"]
        L, C, E0 = (ev[n] for n in sp.cond_names)
        body = dict(r_rules.gate3(L, C, sp, doc["repro"]["csc_sha256_none"]),
                    records=r_records.compare(L, C, E0, sp))
        self._write("gate3", body)
        return body
