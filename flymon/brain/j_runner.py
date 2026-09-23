"""Spec appendix J's procedures (J.11.3-J.11.4): stage 1 scans the ORN->PN depression, stage 2 re-converges the best
setting by C3's rule and judges it by the M2 bar, with D.6 measured on the judged engine.

    stage 1: C3 (no depression) -> all51 + reference = the reference point
             every (f, tau): receptor_scale bisected until the median ALPN matches -> all51 metrics + reference KC median
             order = restorable settings inside the KC band, largest kc_on log10-variance reduction first
    stage 2: for the first max_settings settings of the order:
               C3's rule on that engine (J.10.4) -> dropped: the next setting | aborted: stop
               adopted -> all51 after re-convergence (record) -> H.4's readout reselection -> oracle -> reading
               SELECTED / B -> D.6 (a) and (b) on this engine
             every tried setting dropped -> STOP_NO_OPERATING_POINT

The measurers are interfaces (h3_measure.PoolMeasurer, h4_measure.H4Measurer, j_measure.JMeasurer in a run; scripted
stand-ins in the tests). A passed deadline raises h3_runner.ComputeAborted, reported as COMPUTE_ABORTED, never a verdict.
"""
from __future__ import annotations

import dataclasses
import json

from . import d6a
from .h3_c3 import c3_cell
from .h3_rules import COMBO_ABORTED, COMBO_ADOPTED, COMBO_DROPPED, bisect_log, reference_stats
from .h3_runner import ComputeAborted, Deadline, adopted_record, c1_params, run_candidates
from .h4_rules import DROPPED_NO_READOUT, STOP_MULTI_TYPE
from .h4_runner import Context as H4Context, reselect, run_oracle
from .j_params import with_std
from .j_rules import (B, COMPUTE_ABORTED, INVALID, NO_FEASIBLE_SETTING, SCAN_COMPLETE, SELECTED,
                      STOP_NO_OPERATING_POINT, d6b, judge_d6a_declared, scan_metrics, select_order, stage2_reading)


@dataclasses.dataclass
class JContext:
    spec: object                     # j_spec.JSpec
    h3: object                       # h3_runner.Context for C3's rule (spec with homeo_target = J's, odours, pools, extra)
    pools: dict                      # {"A": [MBON types], "P": [MBON types]}
    probe_seeds: list                # the reference set's seeds (the rest measurement's seeds)
    expected: list                   # [(axis, turn, x, y)] of the pair list
    odours: list                     # d6a.candidate_odours (E.1's 41 candidates) for D.6
    log: object = print

    @property
    def by_turn(self) -> dict:
        out = {}
        for o in self.odours:
            out.setdefault(int(o["turn"]), []).append(o["odor"])
        return out


def params_json(p) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(p)))


# ================================================================ stage 1 (J.11.3)
def stage1(jm, m3, ctx: JContext, base) -> dict:
    spec = ctx.spec
    jm, m3 = Deadline(jm, ctx.h3), Deadline(m3, ctx.h3)
    kc_pct = lambda p: reference_stats(m3.reference(p), spec.h4.h3.release_quasi_linear)["median_kc_pct"]
    out = dict(settings=[], order=[], outcome=None)
    try:
        base_m = scan_metrics(jm.all51_uni(base), spec.count_floor)
        out["base"] = dict(params=base, metrics=base_m, kc_pct=kc_pct(base))
        target = out["target_alpn"] = base_m["alpn_median"]
        for f in spec.std_f:
            for tau in spec.std_tau_ms:
                ctx.log(f"stage 1: f {f:g}, tau {tau:g} ms")
                seen = {}

                def ev(s, f=f, tau=tau):
                    seen[s] = scan_metrics(jm.all51_uni(with_std(base, f, tau, s)), spec.count_floor)
                    ctx.log(f"    receptor_scale {s:.6g} -> median ALPN {seen[s]['alpn_median']:.1f} (target {target:.1f})")
                    return dict(alpn_median=seen[s]["alpn_median"])
                lo, hi = spec.scale_bracket
                search = bisect_log(ev, lo, hi, spec.scale_bisect_steps, target, spec.alpn_match_tol * target,
                                    key="alpn_median")
                rec = dict(f=float(f), tau_ms=float(tau), search=search, restorable=bool(search["converged"]),
                           scale=None, feasible=False, reduction=None)
                if rec["restorable"]:
                    s = float(search["accepted"]["x"])
                    p = with_std(base, f, tau, s)
                    m = seen[s]
                    rec.update(scale=s, params=p, metrics=m, kc_pct=kc_pct(p),
                               reduction=1.0 - m["log10_var"]["kc_on"] / base_m["log10_var"]["kc_on"])
                    rec["feasible"] = bool(spec.kc_band_pct[0] <= rec["kc_pct"] <= spec.kc_band_pct[1])
                out["settings"].append(rec)
    except ComputeAborted as e:
        return dict(out, outcome=COMPUTE_ABORTED, note=str(e))
    out["order"] = select_order(out["settings"], spec)
    out["outcome"] = SCAN_COMPLETE if out["order"] else NO_FEASIBLE_SETTING
    return out


# ================================================================ stage 2 (J.11.4)
def reconverge(m3, ctx: JContext, make_base) -> dict:
    """C3's rule (H.3a.6 as J.10.4 fixes it) on the engine make_base(kc) builds: first_kc's cell first, adopted if its
    candidate passes stages 2-4; otherwise the rest of H.3's kc grid in order. make_base None is H.3's own C3."""
    h3ctx, first = ctx.h3, ctx.spec.first_kc
    rest = [k for k in h3ctx.spec.kc_grid if k != first]
    cells, adopted = [], None
    try:
        cell = c3_cell(m3, h3ctx, first, 0, make_base)
        cells.append(cell)
        if "params" in cell:
            adopted = run_candidates(m3, h3ctx, [cell], calibrate=True, check_membrane=True, judge_baseline=True)
        if adopted is None:
            own = []
            for kc in rest:
                own.append(c3_cell(m3, h3ctx, kc, len(cells), make_base))
                cells.append(own[-1])
            adopted = run_candidates(m3, h3ctx, [c for c in own if "params" in c], calibrate=True, check_membrane=True,
                                     judge_baseline=True)
    except ComputeAborted as e:
        return dict(status=COMBO_ABORTED, cells=cells, adopted=None, guard=None, note=str(e))
    guard = None if adopted is None else {t: st for k in ("A", "P")
                                          for t, st in adopted["stage3"]["guard"][k]["types"].items()}
    return dict(status=COMBO_ADOPTED if adopted else COMBO_DROPPED, cells=cells, adopted=adopted_record(adopted),
                guard=guard)


def judge(m4, ctx: JContext, name: str, params, guard: dict) -> dict:
    """H.4's readout reselection (its reactivity must equal the adopted candidate's stage-3 guard: the same
    measurement), the oracle on the declared pairs, and the M2 bar."""
    h4ctx = H4Context(spec=ctx.spec.h4, combos={name: params}, pools=ctx.pools, probe_seeds=ctx.probe_seeds,
                      expected=ctx.expected, h3_guard={name: guard}, log=ctx.log)
    r = reselect(m4, h4ctx, name, params)
    if r["status"] == STOP_MULTI_TYPE:
        return dict(outcome=STOP_MULTI_TYPE, reselect=r, reading=None)
    if r["status"] == DROPPED_NO_READOUT:
        return dict(outcome=B, reselect=r, reading=None, reason="no reactive and teachable readout in a pool")
    run_oracle(m4, h4ctx, name, params, r)
    if r["oracle"]["reasons"]:
        return dict(outcome=INVALID, reselect=r, reading=None)
    reading = stage2_reading(r["oracle"]["aggregate"], ctx.spec.h4)
    return dict(outcome=reading["outcome"], reselect=r, reading=reading)


def measure_d6(jm, ctx: JContext, params, seeds=d6a.SEEDS, judged: bool = True) -> dict:
    """G.8's presentation on the judged engine; (a) by the declared-engine entry, (b) by E.2's ratio rule."""
    raw = dict(smoke=not judged, seeds=[int(s) for s in seeds], params=params_json(params), odours=ctx.odours,
               strength=d6a.STRENGTH, settle_ms=d6a.SETTLE_MS, read_ms=d6a.READ_MS, window_ms=d6a.WINDOW_MS,
               over_spikes=d6a.OVER_SPIKES, rows=jm.d6(params, ctx.by_turn, seeds))
    b = d6b(raw, ctx.spec.d6b_ratio_limit)
    a = judge_d6a_declared(raw, params_json(params)) if judged else None
    block = ("G.8's block (d6a.SEEDS 400-463), not E.2's 8 seeds (spec J.11.4-5, plan reading 10)"
             if tuple(int(s) for s in seeds) == tuple(d6a.SEEDS) else "verification seeds (not judged)")
    return dict(a=a, b=b, n_rows=len(raw["rows"]), seeds=raw["seeds"], seed_block=block)


def stage2(m3, m4, jm, ctx: JContext, settings: list, order: list, with_d6: bool = True, d6_seeds=d6a.SEEDS) -> dict:
    """settings / order: stage 1's record (each setting with f, tau_ms, scale). Tries at most max_settings settings."""
    spec = ctx.spec
    m3, m4, jm = Deadline(m3, ctx.h3), Deadline(m4, ctx.h3), Deadline(jm, ctx.h3)
    out = dict(tried=[], outcome=None)
    try:
        for i in order[:spec.max_settings]:
            st = settings[i]
            name = f"STD f={st['f']:g} tau={st['tau_ms']:g} ms s={st['scale']:.6g}"
            ctx.log(f"stage 2: {name}")
            make = lambda kc, st=st: with_std(c1_params(ctx.h3.spec, kc, 1.0), st["f"], st["tau_ms"], st["scale"])
            rc = reconverge(m3, ctx, make)
            t = dict(setting=int(i), name=name, f=st["f"], tau_ms=st["tau_ms"], scale=st["scale"], reconverge=rc)
            out["tried"].append(t)
            if rc["status"] == COMBO_ABORTED:
                return dict(out, outcome=COMPUTE_ABORTED, note=rc.get("note"))
            if rc["status"] == COMBO_DROPPED:
                ctx.log(f"stage 2: {name} -> INVALID_ENGINE (no candidate adopted); next setting")
                continue
            p = rc["adopted"]["params"]
            t["all51_after"] = scan_metrics(jm.all51_uni(p), spec.count_floor)
            j = judge(m4, ctx, name, p, rc["guard"])
            out.update(judged=int(len(out["tried"]) - 1), judge=j, outcome=j["outcome"])
            ctx.log(f"stage 2: {name} -> {j['outcome']} {j.get('reading')}")
            if with_d6 and j["outcome"] in (SELECTED, B):
                out["d6"] = measure_d6(jm, ctx, p, d6_seeds, judged=tuple(d6_seeds) == tuple(d6a.SEEDS))
            return out
    except ComputeAborted as e:
        return dict(out, outcome=COMPUTE_ABORTED, note=str(e))
    return dict(out, outcome=STOP_NO_OPERATING_POINT)
