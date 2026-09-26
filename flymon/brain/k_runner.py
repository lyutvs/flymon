"""The runner of spec appendix K (K.8.2 / K.8.4).

stage1: for every g in the grid, re-converge C3's rule (j_runner.reconverge with k_make, J.10.4's state machine: a
setting whose candidates all fail is COMBO_DROPPED = INVALID_ENGINE); measure N and the records on the odd-turn (b)
pairs for ADOPTED engines only; rank (k_rules.select_order) against C3's recorded engine and gate (scan_outcome).
stage2: J.11.4 steps 2-3 on the chosen engine — j_runner.judge (H.4 readout reselection, z, oracle on the even pairs,
stage2_reading) and D.6 when the outcome is SELECTED or B. j_runner.stage2 is not used: it forces the depression on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import d6a
from .h3_rules import COMBO_ABORTED, COMBO_ADOPTED
from .h3_runner import ComputeAborted, Deadline
from .j_params import params_from_json
from .j_rules import B, COMPUTE_ABORTED, SELECTED, scan_metrics
from .j_runner import judge, measure_d6, params_json, reconverge
from .k_metrics import engine_metrics, kc_kc_edges, kc_kc_input, lobe_masks, readout_weights
from .k_params import k_make
from .k_rules import SCAN_GO, closing_state, scan_outcome, select_order


@dataclass
class KArrays:
    w13: np.ndarray
    w05: np.ndarray
    lobes: dict
    kk: tuple
    probe: np.ndarray          # global indices of the alpha'beta' KCs (raster membrane)
    mv: float


def arrays(conn, pops, spec, mv_per_synapse: float) -> KArrays:
    lobes = lobe_masks(conn, pops)
    return KArrays(w13=readout_weights(conn, pops, spec.target_type, spec.min_weight),
                   w05=readout_weights(conn, pops, spec.reward_type, spec.min_weight), lobes=lobes,
                   kk=kc_kc_edges(conn, pops, spec.min_weight),
                   probe=np.asarray(pops.kc, np.int64)[lobes["apbp"]], mv=float(mv_per_synapse))


def measure(km, params, arr: KArrays, g: float) -> dict:
    acts = km.activity(params)
    met = engine_metrics(acts, arr.w13, arr.w05, arr.lobes)
    cur = kc_kc_input(np.mean([a["cx"] for a in acts], axis=0), arr.kk, g, arr.mv)
    met["kc_kc_input"] = {k: float(cur[m].mean()) if m.any() else 0.0 for k, m in arr.lobes.items()}
    return met


def _all51(jm, params, floor: float):
    """K.8.3's all51 kc_on log10 variance record (J's all51 measurement, J's count floor), or None without a
    JMeasurer."""
    return None if jm is None else scan_metrics(jm.all51_uni(params), floor)["log10_var"]["kc_on"]


def stage1(m3, km, jctx, spec, c3, arr: KArrays, jm=None) -> dict:
    first_x = km.odours[km.index[0][0]] if km.index else None
    seed0 = int(spec.j.h4.act_seeds[0])
    m3, kmd = Deadline(m3, jctx.h3), Deadline(km, jctx.h3)
    jmd = None if jm is None else Deadline(jm, jctx.h3)
    base, settings = None, []
    try:
        base = dict(g=0.0, name="C3", params=params_json(c3), metrics=measure(kmd, c3, arr, 0.0))
        base["metrics"]["all51_kc_on_log10_var"] = _all51(jmd, c3, spec.j.count_floor)
        jctx.log(f"K stage 1: C3 N {base['metrics']['N']:.4g}")
        for g in spec.grid:
            jctx.log(f"K stage 1: g = {g:g}, C3 rule re-convergence")
            rc = reconverge(m3, jctx, k_make(jctx.h3.spec, g))
            st = dict(g=g, status=rc["status"], reconverge=rc, metrics=None)
            settings.append(st)
            if rc["status"] == COMBO_ABORTED:
                return dict(outcome=COMPUTE_ABORTED, base=base, settings=settings, order=[], ratios={},
                            note=rc.get("note"))
            if rc["status"] != COMBO_ADOPTED:
                jctx.log(f"K stage 1: g = {g:g} INVALID_ENGINE")
                continue
            p = rc["adopted"]["params"]
            st["metrics"] = measure(kmd, p, arr, g)
            st["metrics"]["all51_kc_on_log10_var"] = _all51(jmd, p, spec.j.count_floor)
            st["boundary_share"] = next((c.get("boundary_share") for c in rc["cells"]
                                         if c.get("label") == rc["adopted"]["label"]), None)
            if g == spec.raster_g and first_x is not None:
                st["raster"] = kmd.raster(p, first_x, seed0, arr.probe)
            jctx.log(f"K stage 1: g = {g:g} ADOPTED kc_thresh {p.kc_thresh:g}, N {st['metrics']['N']:.4g}")
    except ComputeAborted as e:
        return dict(outcome=COMPUTE_ABORTED, base=base, settings=settings, order=[], ratios={}, note=str(e))
    if not any(s["g"] == spec.raster_g and "raster" in s for s in settings):
        for s in settings:
            if s["g"] == spec.raster_g:
                s["raster"] = None                     # recorded: the raster setting was not ADOPTED
    n_c3 = base["metrics"]["N"]
    order = select_order(settings, n_c3, spec.tie_tol)
    ratios = {i: settings[i]["metrics"]["N"] / n_c3 for i in order}
    return dict(outcome=scan_outcome(settings, order, n_c3, spec.gate_ratio), base=base, settings=settings,
                order=order, ratios=ratios)


def stage2(m4, jm, jctx, setting: dict, with_d6: bool = True, d6_seeds=None) -> dict:
    p = params_from_json(setting["reconverge"]["adopted"]["params"])
    guard = setting["reconverge"]["guard"]
    name = f"KC-KC g={setting['g']:g} kc_thresh={p.kc_thresh:g}"
    m4, jm = Deadline(m4, jctx.h3), Deadline(jm, jctx.h3)
    try:
        j = judge(m4, jctx, name, p, guard)
        d6 = measure_d6(jm, jctx, p, d6a.SEEDS if d6_seeds is None else d6_seeds) \
            if with_d6 and j["outcome"] in (SELECTED, B) else None
    except ComputeAborted as e:
        return dict(name=name, g=setting["g"], params=params_json(p), judge=None, outcome=COMPUTE_ABORTED, d6=None,
                    state=COMPUTE_ABORTED, note=str(e))
    return dict(name=name, g=setting["g"], params=params_json(p), judge=j, outcome=j["outcome"], d6=d6,
                state=closing_state(j), scan=SCAN_GO)
