# flymon/brain/q_runner.py
"""Spec Q's stage chain (Q.5, Q.6.8, Q.6.9; plan Readings 3, 15, 19): repro -> smoke -> s_c -> q0 -> q1:<condition> x 6
-> records, one block each in results/summary/q_reward.json, written only through q_store. Every stage refuses
(SystemExit 2, nothing written) when an earlier block is missing, a later block exists, the summary has uncommitted
changes, or a hashed Q file is dirty. A failed reproduction gate blocks every later stage. A condition that fails its
gate is written as INVALID (a record) — Q has no STOP (Q.1).
(c) per Q.6.9: the condition still named "s_up" LOWERS s. Smoke probes the KC median (base Params) at s_down[0] (0.7);
inside the KC band -> 0.7, else s_down[1] (0.8); both outside -> s None, "(c) 조작 불가" (① 판단 불가), and q1:s_up is
written INVALID without running. q_pairs.cap_s (the cap-limited max s) is only a record."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git  # noqa: F401  (tests monkeypatch q_runner.summary_git)
from . import q_pairs, q_records, q_rules, q_store
from .h3_store import canonical
from .h3_store import git_state as _h3_git_state
from .q_measure import Q_MEASURE_FILES, cond_params
from .q_pairs import key_str
from .q_spec import smoke

ORDER = ("repro", "smoke", "s_c", "q0", "q1:base", "q1:apl_mbon05", "q1:apl_nonkc", "q1:s_up", "q1:mv_lo",
         "q1:mv_hi", "records")
Q_HASHED_FILES = tuple(dict.fromkeys(Q_MEASURE_FILES + (
    "flymon/brain/q_spec.py", "flymon/brain/q_pairs.py", "flymon/brain/q_records.py", "flymon/brain/q_rules.py",
    "flymon/brain/q_runner.py", "scripts/run_q.py", "flymon/agent/e_pairs.py", "flymon/agent/encode_grid.py",
    "flymon/agent/e_codebook.py", "flymon/agent/e_spec.py", "flymon/agent/config.py", "flymon/brain/h4_rules.py",
    "flymon/brain/h4_pairs.py", "flymon/brain/d6a.py", "flymon/brain/odor_real.py")))


def git_state() -> dict:
    return _h3_git_state(files=Q_HASHED_FILES)


def refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, z — block h4), the H.4 pools as types, the encoder summary, the 21 (b) rows with k2-norm
    odours, the ORN cap, and the encoder-key check when the encoder code key is unchanged."""
    from ..agent import e_pairs
    from ..agent.config import load_c3_config
    from ..agent.e_runner import MEASURE_FILES_E
    from . import odor_real
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_store import code_key
    cfg = load_c3_config(spec.m0d_summary)
    if cfg.readout["P"] != spec.p_type:
        refuse(f"C3's P readout is {cfg.readout['P']}, not {spec.p_type}")
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = q_pairs.b_rows(e_pairs.even_situations(pops), rc, enc, spec)
    enc_code = code_key(npz, files=MEASURE_FILES_E)
    ek = None
    if enc_code["key"] == enc["even"]["code_key"]:
        ek = lambda sel: q_pairs.encoder_key_match(sel, enc, spec, enc_code, cfg.params, cfg.readout, cfg.z, types,  # noqa: E731
                                                  len(pops.kc))
    return dict(rows=rows, enc=enc, readout=dict(cfg.readout), z=dict(cfg.z), types=types, params=cfg.params,
                n_kc=len(pops.kc), max_rate_hz=float(cfg.params.max_rate_hz), cap_hz=float(odor_real.cap_hz(cfg.params)),
                encoder_keys=ek)


def _r_diff(a, b):
    """Q0 r minus the encoder's stored r; 0.0 when equal (incl. equal infinities), None when not finite."""
    if a == b:
        return 0.0
    d = float(a) - float(b)
    return d if math.isfinite(d) else None


class Runner:
    def __init__(self, measurer, ctx: dict, spec, summary_path=None, code: dict | None = None):
        self.m, self.ctx, self.spec = measurer, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.rows = list(ctx["rows"])
        self.c_name = spec.cond_names[3]           # "s_up" = (c), the LOWERED-s manipulation (Q.6.9)

    # ---- chain -----------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return q_store.read_summary(self.summary_path)

    def _require(self, stage: str) -> dict:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first "
                       f"(a failed repro block: commit it or discard it before rerunning)")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed Q files are dirty: {gs['dirty_hashed']}")
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; Q never rewrites an earlier block")
        if i > 0 and not doc["repro"].get("passed"):
            refuse(f"stage {stage}: the reproduction gate did not pass")
        return doc

    def _write(self, stage: str, body: dict) -> dict:
        block = dict(body, stage=stage, code_key=self.code_key, git=git_state(),
                     written_at=_dt.datetime.now(_dt.timezone.utc).isoformat())
        q_store.write_summary_block(self.summary_path, stage, block, [self.ctx["params"]])
        return block

    def _subset(self, spec) -> list:
        return [self.rows[i] for i in spec.pairs_subset] if spec.pairs_subset else list(self.rows)

    def _conditions(self, spec, s, mv) -> list:
        """spec.conditions never receives None: with (c) 조작 불가 (s None) the (c) condition is left out and the others
        are built with the base strength as a placeholder that only the left-out (c) would have read."""
        cs = spec.conditions(spec.strength if s is None else s, *mv)
        return [c for c in cs if not (s is None and c.name == self.c_name)]

    # ---- repro (Q.6.8, Reading 3) ----------------------------------------------------------------------------
    def stage_repro(self) -> dict:
        self._require("repro")
        sp = self.spec
        files = q_pairs.q0_files(self.rows, self.ctx["enc"], sp)
        sel = [self.rows[i] for i in sp.repro_pairs]
        fsel = [files[i] for i in sp.repro_pairs]
        base = sp.conditions(sp.strength, 1.0, 1.0)[0]
        got = self.m.run(sel, base, "repro", sp.repro_seeds())
        ek = self.ctx.get("encoder_keys")
        km = ek(sel) if ek else [None] * len(sel)
        pairs = []
        for r, f, g, k in zip(sel, fsel, got, km):
            ref = json.loads(Path(f["path"]).read_text())["result"] if f["ok"] else None
            mine = {kk: v for kk, v in g["result"].items() if kk != "q"}
            pairs.append(dict(key=key_str(r), q0_sha_ok=f["ok"],
                              equal=bool(ref is not None and canonical(mine) == canonical(ref)), encoder_key_match=k))
        shas = sorted({g["result"]["q"]["csc_sha256"] for g in got})
        passed = (all(p["equal"] and p["q0_sha_ok"] and p["encoder_key_match"] is not False for p in pairs)
                  and len(shas) == 1)
        body = dict(passed=passed, pairs=pairs, encoder_key_checked=ek is not None,
                    csc_sha256=shas[0] if len(shas) == 1 else shas, seeds=sp.repro_seeds(), wall_s=self.m.last_wall_s)
        self._write("repro", body)
        return body

    # ---- smoke (Q.6.5, Q.6.8, Q.6.9, Reading 15) -------------------------------------------------------------
    def _kc_median(self, odours, params, strength, seeds) -> float:
        fr = self.m.kc_probe(odours, params, strength, seeds)
        return float(np.median([x for v in fr.values() for x in v]))

    def stage_smoke(self) -> dict:
        self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        lo, hi = sp.kc_band
        cap = q_pairs.cap_s(q_pairs.odours_of(self.rows), self.ctx["max_rate_hz"], self.ctx["cap_hz"], sp)
        odours = {f"{key_str(r)}|{side}": r[f"odor_{side}"] for r in self.rows for side in ("x", "y")}
        # (d): mv_scale probe at the base s; only the out-of-band side falls back (preflight ruling)
        kc = {}
        for scale in sp.mv_scales:
            c = sp.conditions(sp.strength, scale, scale)[4]
            kc[str(scale)] = self._kc_median(odours, cond_params(self.ctx["params"], c), sp.strength,
                                             sm.kc_probe_seeds)
        inside = [lo <= kc[str(s)] <= hi for s in sp.mv_scales]
        mv = [sp.mv_scales[j] if inside[j] else sp.mv_fallback[j] for j in range(2)]
        # (c) per Q.6.9: lowered s, s_down[0] first, s_down[1] when the first is out of band; both out -> 조작 불가
        kc_s, s = {}, None
        for cand in sp.s_down:
            kc_s[str(cand)] = self._kc_median(odours, self.ctx["params"], cand, sm.kc_probe_seeds)
            if lo <= kc_s[str(cand)] <= hi:
                s = cand
                break
        sel = self._subset(sm)
        per = {}
        for c in self._conditions(sm, s, mv):
            got = self.m.run(sel, c, "smoke", sm.q1_seeds())
            rec = q_records.condition_record(got, c, sm, self.ctx["z"], [key_str(r) for r in sel])
            per[c.name] = dict(status=rec["status"], reasons=rec["reasons"], kc_median=rec["kc_median"],
                               edit_edges=rec["edit_edges"], csc_sha256=rec["csc_sha256"], strength=c.strength,
                               mv_scale=c.mv_scale, wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
            if c.name == self.c_name:
                per[c.name]["note"] = q_rules.s_up_note(sp)
        note = q_rules.s_up_note(sp)
        s_choice = dict(s=s, candidates=list(sp.s_down), kc_probe_median=kc_s, unmanipulable=s is None,
                        reason=(f"(c) {sp.unmanip_note}: s {list(sp.s_down)} 모두 KC 대역 [{lo}, {hi}] 밖"
                                if s is None else None), direction="lowered", note=note)
        detail = dict(kc_probe_median=kc, mv=mv, s_choice=s_choice, s_cap_record=cap, conditions=per,
                      seeds=sm.q1_seeds(), kc_probe_seeds=list(sm.kc_probe_seeds), pairs=[key_str(r) for r in sel])
        q_store.write_json(sp.smoke_detail, detail, [self.ctx["params"]])
        body = dict(mv=mv, mv_fallback_used=[not x for x in inside], kc_probe_median=kc, s_choice=s_choice,
                    s_cap_record=cap, conditions=per, s_up_note=note,
                    edit_edges={n: v["edit_edges"] for n, v in per.items()}, detail=sp.smoke_detail)
        self._write("smoke", body)
        return body

    # ---- s_c (Q.6.4 as amended by Q.6.9) ---------------------------------------------------------------------
    def stage_s_c(self) -> dict:
        doc = self._require("s_c")
        sp = self.spec
        ch = doc["smoke"]["s_choice"]
        s = ch["s"]
        weak = bool(s is not None and abs(float(s) - sp.strength) < sp.s_weak)
        body = dict(s=s, weak=weak, weak_note=sp.weak_note if weak else None, direction="lowered",
                    unmanipulable=s is None, unmanip_note=sp.unmanip_note if s is None else None,
                    candidates=ch["candidates"], kc_probe_median=ch["kc_probe_median"], kc_band=list(sp.kc_band),
                    strength=sp.strength, s_weak=sp.s_weak, s_cap_record=doc["smoke"]["s_cap_record"],
                    note=q_rules.s_up_note(sp))
        self._write("s_c", body)
        return body

    # ---- q0 (Q.3, Q.6.1, Q.6.7, Q.6.8) ------------------------------------------------------------------------
    def _encoder_r_check(self, vals: dict) -> dict:
        """Q0 composite r per pair vs block even's stored pair r (encoder_grid.json, the k2-norm config) — a record;
        the controller stops on a mismatch at the real run."""
        pairs = self.ctx["enc"].get("even", {}).get("configs", {}).get(self.spec.config, {}).get("pairs")
        ref = {key_str(p): p["r"] for p in (pairs or []) if p.get("axis") == "b" and "r" in p}
        if not ref:
            return dict(available=False, encoder_grid_r_match=None,
                        why="encoder_grid.json block even holds no (b) pair r for this config")
        diff = {k: _r_diff(v["r"], ref[k]) for k, v in sorted(vals.items()) if k in ref}
        missing = sorted(set(vals) - set(ref))
        finite = [abs(d) for d in diff.values() if d is not None]
        match = bool(not missing and diff and all(v["r"] == ref[k] or math.isclose(v["r"], ref[k])
                                                  for k, v in vals.items()))
        return dict(available=True, diff=diff, max_abs_diff=max(finite) if finite else None, missing=missing,
                    n_compared=len(diff), encoder_grid_r_match=match)

    def stage_q0(self) -> dict:
        self._require("q0")
        sp = self.spec
        files = q_pairs.q0_files(self.rows, self.ctx["enc"], sp)
        results = [json.loads(Path(f["path"]).read_text())["result"] if f["ok"] else None for f in files]
        rec = q_records.q0_record(files, results, sp, self.ctx["z"])
        rec["encoder_r_check"] = self._encoder_r_check(rec["vals"])
        self._write("q0", rec)
        return rec

    # ---- q1 (Q.3, Q.6.2-Q.6.6, Q.6.9) -------------------------------------------------------------------------
    def _unmanipulable_block(self, doc) -> dict:
        sp = self.spec
        place = {c.name: c for c in sp.conditions(sp.strength, *doc["smoke"]["mv"])}[self.c_name]
        cond = dict(dataclasses.asdict(place), strength=None)
        return dict(condition=cond, status=q_records.INVALID,
                    reasons=[f"(c) {sp.unmanip_note} — s {list(sp.s_down)} 모두 KC 대역 밖 (Q.6.9), 실행하지 않음"],
                    vals={}, kc_median=None, d6a_over_share=None, apl_out_median=None, csc_sha256=None, edit_edges=[],
                    naive_px_median=None, naive_px_zero=0, jaccard_median=None, manifest=[], wall_s=0.0, jobs=0,
                    seeds=sp.q1_seeds(), unmanipulable=True, weak=False, note=q_rules.s_up_note(sp))

    def stage_q1(self, name: str) -> dict:
        if name not in self.spec.cond_names:
            refuse(f"unknown condition {name}; Q has {self.spec.cond_names}")
        doc = self._require(f"q1:{name}")
        sp = self.spec
        s = doc["s_c"]["s"]
        if name == self.c_name and s is None:
            rec = self._unmanipulable_block(doc)
            self._write(f"q1:{name}", rec)
            return rec
        conds = {c.name: c for c in self._conditions(sp, s, doc["smoke"]["mv"])}
        cond = conds[name]
        rows = self._subset(sp)
        got = self.m.run(rows, cond, "q1", sp.q1_seeds())
        repro_sha = doc["repro"]["csc_sha256"]
        same = repro_sha if cond.edit == sp.edit_none and cond.mv_scale == 1.0 else None
        other = repro_sha if cond.edit != sp.edit_none else None
        rec = q_records.condition_record(got, cond, sp, self.ctx["z"], [key_str(r) for r in rows], same_sha=same,
                                         other_sha=other)
        rec.update(wall_s=self.m.last_wall_s, jobs=self.m.last_jobs, seeds=sp.q1_seeds())
        if name == self.c_name:
            weak = bool(doc["s_c"]["weak"])
            rec.update(weak=weak, weak_note=sp.weak_note if weak else None, unmanipulable=False,
                       note=q_rules.s_up_note(sp))
        self._write(f"q1:{name}", rec)
        return rec

    # ---- records (Q.4, Q.6) ----------------------------------------------------------------------------------
    def stage_records(self) -> dict:
        doc = self._require("records")
        q1 = {n: doc[f"q1:{n}"] for n in self.spec.cond_names}
        out = q_rules.assemble(doc["q0"], q1, doc["s_c"], self.spec)
        self._write("records", out)
        return out
