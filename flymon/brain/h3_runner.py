"""M0d H.3 procedures (spec appendix H.3a.4-H.3a.5): the four stages, the candidate state machine, C0 and C1.

    candidate -> stage 2 (MBON hold) -> stage 3 (qualifications 1-3 + membrane recheck) -> rank -> stage 4 -> adopted
       |- qualification / membrane fails           -> qualification_failed, next candidate
       |- overlap CI holds 0 on 8 and on 16 seeds  -> indeterminate, next candidate
       |- the single passing guard type splits      -> indeterminate, next candidate
       '- stage 4 fails                             -> stage4_failed, next candidate
    candidates exhausted -> the combination is dropped (stop; the user decides)

Every candidate gets stages 2 and 3 (the ranking needs its final-Params margins); stage 4 then runs in rank order
until one passes, and lower-ranked candidates are recorded as not reached. The measurer is an interface
(h3_measure.PoolMeasurer in a run, a scripted stand-in in the tests); a deadline turns into ComputeAborted, which
the combination reports as "compute aborted", never as a drop.
"""
from __future__ import annotations

import dataclasses
import time

from .config import Params
from .h3_rules import (ADOPTED, COMBO_ABORTED, COMBO_ADOPTED, COMBO_DROPPED, GATE_FAILED, NOT_REACHED,
                       SEARCH_FAILED, baseline_ci_verdict, baseline_stats, bisect_log, bisect_mid, boot_mean_ci,
                       d4_stats, guard_pool, membrane_check, overlap_clause, qualification_verdict, rank_score,
                       reference_stats, runaway_verdict)


class ComputeAborted(Exception):
    pass


@dataclasses.dataclass
class Context:
    spec: object
    odors: list                      # the reference set, generator order
    pools: dict                      # {"A": [MBON types], "P": [MBON types]}
    n_kc: int = 0
    deadline: float | None = None    # time.time() after which no new measurement starts
    log: object = print
    extra: dict = dataclasses.field(default_factory=dict)   # C3: update mask, rule thresholds, threshold writer

    @property
    def odor_names(self) -> list:
        return [o["name"] for o in self.odors]

    @property
    def probe_seeds(self) -> list:
        return [int(s) for o in self.odors for s in o["seeds"]]


class Deadline:
    """Wraps a measurer: every measurement first checks the deadline."""

    def __init__(self, m, ctx: Context):
        self._m, self._ctx = m, ctx

    def __getattr__(self, name):
        fn = getattr(self._m, name)

        def call(*a, **kw):
            if self._ctx.deadline is not None and time.time() > self._ctx.deadline:
                raise ComputeAborted(f"deadline passed before {name}")
            return fn(*a, **kw)
        return call


def c1_params(spec, kc: float, scale: float, **kw) -> Params:
    return Params(apl_mode="graded", apl_r_max=spec.apl_r_max, kc_thresh=float(kc), apl_input_scale=float(scale), **kw)


# ================================================================ stage 1
def search_scale(m, ctx: Context, make_params) -> dict:
    """apl_input_scale bisection to the membrane target; make_params(s) -> Params."""
    spec = ctx.spec

    def ev(s):
        st = reference_stats(m.reference(make_params(s)), spec.release_quasi_linear)
        ctx.log(f"    s {s:.6g} -> {st['median_mv']:.3f} mV, KC {st['median_kc_pct']:.3f}%")
        return st
    lo, hi = spec.scale_bracket
    return bisect_log(ev, lo, hi, spec.scale_bisect_steps, spec.membrane_target_mv, spec.membrane_tol_mv)


# ================================================================ stage 2
def calibrate_hold(m, ctx: Context, params: Params) -> dict:
    spec = ctx.spec

    def ev(h):
        st = baseline_stats([r["trimmed"] for r in m.baseline(dataclasses.replace(params, mbon_hold_frac=h),
                                                               spec.baseline_cal_seeds)])
        ctx.log(f"    hold {h:.6f} -> {st['mean_hz']:.3f} Hz")
        return st
    lo, hi = spec.hold_bracket
    b = bisect_mid(ev, lo, hi, spec.hold_bisect_steps, spec.baseline_target_hz, key="mean_hz")
    b["calibration"] = ev(b["accepted"])
    return b


# ================================================================ stage 3
def qualify(m, ctx: Context, params: Params, check_membrane: bool) -> dict:
    spec = ctx.spec
    ref = m.reference(params)
    stats = reference_stats(ref, spec.release_quasi_linear)
    rest_by_seed = {r["seed"]: r for r in m.rest(params, ctx.probe_seeds)}
    guards = {k: guard_pool(ref, rest_by_seed, ctx.pools[k], ctx.odor_names, spec) for k in ("A", "P")}
    rows8 = m.design(params, spec.design_seeds)
    d8 = d4_stats(rows8, spec.d4_band)
    ci8 = boot_mean_ci(d8["overlap_margin_per_seed"], spec.boot_draws, spec.boot_seed)
    overlap = dict(clause8=overlap_clause(ci8["ci"]), ci8=ci8, d16=None, ci16=None)
    final = d8
    if overlap["clause8"] == "holds_zero":
        final = d4_stats(rows8 + m.design(params, spec.design_extra_seeds), spec.d4_band)
        overlap["d16"] = final
        overlap["ci16"] = boot_mean_ci(final["overlap_margin_per_seed"], spec.boot_draws, spec.boot_seed)
        c16 = overlap_clause(overlap["ci16"]["ci"])
        overlap["clause"] = "indeterminate" if c16 == "holds_zero" else c16
    else:
        overlap["clause"] = overlap["clause8"]
    membrane = membrane_check(stats, spec.membrane_target_mv, spec.membrane_tol_mv) if check_membrane else None
    sparsity_ok = bool(d8["margin_pp"] > 0)
    status, reasons = qualification_verdict(sparsity_ok, overlap["clause"],
                                            [("A", guards["A"]["verdict"]), ("P", guards["P"]["verdict"])],
                                            None if membrane is None else membrane["ok"])
    return dict(reference=stats, guard=guards, d4=d8, sparsity_ok=sparsity_ok, overlap=overlap, membrane=membrane,
                status=status, reasons=reasons,
                score=rank_score(d8["margin_pp"], final["overlap_margin"], spec.overlap_sd))


# ================================================================ stage 4
def stage4(m, ctx: Context, params: Params, judge_baseline: bool) -> dict:
    spec = ctx.spec
    base = baseline_stats([r["trimmed"] for r in m.baseline(params, spec.baseline_gate_seeds)])
    bv = baseline_ci_verdict(base, spec.baseline_ci_z, spec.baseline_band_hz)
    rest = m.baseline(params, spec.runaway_rest_seeds, sat_hz=spec.runaway_rest_sat_hz)
    odor = m.odor_runaway(params, spec.runaway_odor_seeds)
    run = runaway_verdict([r["n_kc_over_sat"] for r in rest], [r["n_kc_over_sat"] for r in odor],
                          len(spec.runaway_rest_seeds), len(spec.runaway_odor_seeds))
    run["odor_detail"] = [{k: r[k] for k in ("seed", "n_kc_over_record", "kc_hz_top5", "band")} for r in odor]
    run["odor_record_hz"] = spec.runaway_odor_record_hz
    ok = bool(run["ok"] and (bv["ok"] or not judge_baseline))
    return dict(baseline=base, baseline_ci=bv, baseline_judged=judge_baseline, runaway=run, ok=ok)


# ================================================================ the candidate pipeline
def run_candidates(m, ctx: Context, cands: list, calibrate: bool, check_membrane: bool, judge_baseline: bool):
    """cands: dicts with label, order, params (the stage-1 Params). Mutates them; returns the adopted one or None."""
    spec = ctx.spec
    for c in cands[:spec.max_candidates]:
        ctx.log(f"  {c['label']}: stage 2" if calibrate else f"  {c['label']}: stage 3")
        if calibrate:
            c["stage2"] = calibrate_hold(m, ctx, c["params"])
            c["final_params"] = dataclasses.replace(c["params"], mbon_hold_frac=c["stage2"]["accepted"])
        else:
            c["final_params"] = c["params"]
        c["stage3"] = qualify(m, ctx, c["final_params"], check_membrane)
        c["status"] = c["stage3"]["status"] or "qualified"
        ctx.log(f"  {c['label']}: stage 3 -> {c['status']} {c['stage3']['reasons']} score {c['stage3']['score']:+.4f}")
    qualified = sorted((c for c in cands[:spec.max_candidates] if c["status"] == "qualified"),
                       key=lambda c: (-c["stage3"]["score"], c["order"]))
    adopted = None
    for rank, c in enumerate(qualified, start=1):
        c["rank"] = rank
        if adopted is not None:
            c["status"] = NOT_REACHED
            continue
        c["stage4"] = stage4(m, ctx, c["final_params"], judge_baseline)
        c["status"] = ADOPTED if c["stage4"]["ok"] else GATE_FAILED
        ctx.log(f"  {c['label']}: stage 4 -> {c['status']}")
        if c["stage4"]["ok"]:
            adopted = c
    return adopted


def adopted_record(c) -> dict | None:
    """What later steps read from an adopted candidate: its final Params and its reference-set KC median (C3's A0)."""
    if c is None:
        return None
    return dict(label=c["label"], params=c["final_params"],
                reference_median_kc_pct=c["stage3"]["reference"]["median_kc_pct"])


def _combo(name: str, cells: list, adopted) -> dict:
    return dict(combo=name, status=COMBO_ADOPTED if adopted else COMBO_DROPPED, cells=cells,
                adopted=adopted_record(adopted))


def run_c0(m, ctx: Context) -> dict:
    """C0 = the current engine: no stage 1 or 2, no membrane recheck, baseline recorded but not judged (H.3a.5)."""
    m = Deadline(m, ctx)
    cell = dict(label="C0", order=0, kc=Params().kc_thresh, params=Params())
    try:
        adopted = run_candidates(m, ctx, [cell], calibrate=False, check_membrane=False, judge_baseline=False)
    except ComputeAborted as e:
        return dict(combo="C0", status=COMBO_ABORTED, cells=[cell], adopted=None, note=str(e))
    return _combo("C0", [cell], adopted)


def run_c1(m, ctx: Context) -> dict:
    spec = ctx.spec
    m = Deadline(m, ctx)
    cells = []
    try:
        for i, kc in enumerate(spec.kc_grid):
            ctx.log(f"C1 stage 1: kc_thresh {kc:g}")
            search = search_scale(m, ctx, lambda s, kc=kc: c1_params(spec, kc, s))
            cell = dict(label=f"C1 G({kc:g})", order=i, kc=float(kc), search=search)
            if search["converged"]:
                cell["params"] = c1_params(spec, kc, search["accepted"]["x"])
            else:
                cell["status"] = SEARCH_FAILED
            cells.append(cell)
        if ctx.extra.get("stop_after") == "stage1":
            return dict(combo="C1", status="stopped_after_stage1", cells=cells, adopted=None)
        adopted = run_candidates(m, ctx, [c for c in cells if "params" in c], calibrate=True, check_membrane=True,
                                 judge_baseline=True)
    except ComputeAborted as e:
        return dict(combo="C1", status=COMBO_ABORTED, cells=cells, adopted=None, note=str(e))
    return _combo("C1", cells, adopted)
