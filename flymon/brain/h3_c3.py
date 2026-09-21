"""M0d H.3 C3: C1 plus homeostatic KC thresholds (spec appendix H.3a.6).

A C3 cell is one `kc_thresh`. Each cycle bisects `apl_input_scale` at the current thresholds, then runs the
homeostasis loop at that scale; a cycle ends with the membrane of the loop's last measurement. Cycles repeat while
the membrane is outside 11 +- 1 mV, at most three, and stop as stalled when a cycle moved the membrane < 0.2 mV and
the KC activity < 0.2 pp. The first cycle starts from the PN-normalisation rule's thresholds and is expressed in
`pn_norm` mode, so its bisection is literally C1's stage 1 for that `kc_thresh` (same Params, same cache entries):
inheriting C1's operating point and re-deriving it are the same computation.

Order (H.3a.6): C1's adopted `kc_thresh` first; if that candidate fails anything, the remaining grid cells run on
C3's own (whatever C1's outcome); if C1 was dropped, every grid cell runs.

A0 (H.3a.6 as amended 2026-09-21, before any H.3 run): the homeostasis target is C1's adopted candidate's
reference-set KC active median at its final Params, as a fraction rounded to `homeo_target_digits` (0.062 for
G(1.6, 0.11863)); `SPEC.homeo_target` (0.062) only when C1 adopted nothing.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from .connectome import apply_sign_override
from .engine_cpu import Engine
from .h3_rules import (BOUNDARY_LIMITED, COMBO_ABORTED, COMBO_ADOPTED, COMBO_DROPPED, CYCLES_EXHAUSTED,
                       HOMEOSTASIS_UNCONVERGED, SEARCH_FAILED, STALLED, boundary_share, cycle_stalled,
                       firing_fraction, homeostasis_done, homeostasis_step, membrane_check, reference_stats,
                       theta_motion)
from .h3_runner import ComputeAborted, Context, Deadline, adopted_record, c1_params, run_candidates, search_scale
from .h3_store import sha256_file, write_bytes


def update_mask(conn, pops, params) -> np.ndarray:
    """KCs (in pops.kc order) with PN input that survives the engine's edge filter (min_weight, non-zero sign):
    the H.3a.6 update set (3,768 KCs on MaleCNS). It is a subset of `Engine.kc_pn_input > 0` (raw synapse sums),
    which the threshold loader requires to keep the rule value outside."""
    sign, _ = apply_sign_override(conn, params)
    is_pn = np.zeros(conn.N, bool); is_pn[np.asarray(pops.alpn, np.int64)] = True
    is_kc = np.zeros(conn.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
    m = is_pn[conn.pre] & is_kc[conn.post] & (conn.w >= params.min_weight) & (sign[conn.pre] != 0)
    kept = np.zeros(conn.N, np.float64)
    np.add.at(kept, conn.post[m], conn.w[m].astype(np.float64))
    return kept[np.asarray(pops.kc, np.int64)] > 0


def rule_thresholds(conn, pops, params) -> tuple[np.ndarray, np.ndarray]:
    """(the PN-normalisation rule's KC thresholds, Engine.kc_pn_input > 0) for this kc_thresh, from the engine."""
    p = dataclasses.replace(params, kc_thresh_mode="pn_norm", kc_thresh_file="", kc_thresh_sha256="")
    eng = Engine(conn, pops, p, seed=0)
    return eng.v_th[np.asarray(pops.kc, np.int64)].astype(np.float32).copy(), eng.kc_pn_input > 0


def npz_bytes(**arrays) -> bytes:
    """np.savez's layout with fixed zip timestamps: the bytes depend only on the arrays (the sha is in the cache key)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, arr in arrays.items():
            with z.open(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), "w") as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)
    return buf.getvalue()


class ThresholdFiles:
    """Content-addressed threshold files: the name is the hash of the data, so a resumed run reuses the same file and
    therefore the same sha256 in Params (and the same cache keys)."""

    def __init__(self, root, kc_body_ids):
        self.root = Path(root)
        self.ids = np.asarray(kc_body_ids, np.int64)

    def params(self, base, theta) -> object:
        th = np.asarray(theta, np.float32)
        name = hashlib.sha256(self.ids.tobytes() + th.tobytes()).hexdigest()[:24]
        path = self.root / f"theta-{name}.npz"
        p = dataclasses.replace(base, kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256="x")
        if not (path.exists() and self._holds(path, th)):          # a damaged or foreign file is rewritten
            write_bytes(path, npz_bytes(kc_body_ids=self.ids, v_th=th), [p])
        return dataclasses.replace(p, kc_thresh_sha256=sha256_file(path))

    def _holds(self, path, th) -> bool:
        try:
            with np.load(path) as d:
                return bool(np.array_equal(d["kc_body_ids"], self.ids) and np.array_equal(d["v_th"], th)
                            and d["v_th"].dtype == np.float32)
        except (OSError, ValueError, KeyError):
            return False


def homeostasis(m, ctx: Context, base, theta0, rule, update, files: ThresholdFiles) -> dict:
    """Iteration t measures at theta_t and stops when both conditions hold; otherwise theta_{t+1} is the update. At most
    homeo_max_iter measurements; the thresholds returned are always the last MEASURED ones (and their Params)."""
    spec = ctx.spec

    def params_for(theta):
        return base if np.array_equal(theta, rule) else files.params(base, theta)
    theta, prev, prev2, trace = np.asarray(theta0, np.float32), None, None, []
    for it in range(spec.homeo_max_iter):
        params = params_for(theta)
        rows = m.reference(params)
        stats = reference_stats(rows, spec.release_quasi_linear)
        a = firing_fraction(rows, ctx.n_kc)
        done = homeostasis_done(a, prev, theta, update, spec)
        trace.append(dict(iteration=it, median_mv=stats["median_mv"], median_kc_pct=stats["median_kc_pct"],
                          boundary_share=boundary_share(theta, rule, update, spec.homeo_clip),
                          **theta_motion(prev2, prev, theta, update, spec.homeo_dtheta_max), **done))
        ctx.log(f"    homeostasis {it}: a median {done['median_a']:.4f}, dtheta q {done['dtheta_q']}, "
                f"{stats['median_mv']:.3f} mV, KC {stats['median_kc_pct']:.3f}%")
        if done["done"]:
            return dict(converged=True, theta=theta, params=params, stats=stats, trace=trace)
        if it == spec.homeo_max_iter - 1:
            return dict(converged=False, theta=theta, params=params, stats=stats, trace=trace)
        prev2, prev, theta = prev, theta, homeostasis_step(theta, rule, a, update, spec.homeo_eta, spec.homeo_target,
                                                           spec.homeo_clip)


def run_cycles(spec, search, homeo, theta0, boundary) -> dict:
    """The C3 cycle loop. search(theta) -> a stage-1 bisection record; homeo(s, theta) -> the homeostasis record
    (converged, theta, params, stats, trace); boundary(theta) -> the share of updated KCs at a clip bound."""
    cycles, prev_end, theta = [], None, theta0
    for cyc in range(1, spec.c3_max_cycles + 1):
        s = search(theta)
        rec = dict(cycle=cyc, search=s)
        cycles.append(rec)
        if not s["converged"]:
            return dict(status=SEARCH_FAILED, cycles=cycles)
        h = homeo(s["accepted"]["x"], theta)
        rec["homeostasis"] = {k: h[k] for k in ("converged", "trace", "stats", "params")}
        if not h["converged"]:
            return dict(status=HOMEOSTASIS_UNCONVERGED, cycles=cycles)
        theta, end = h["theta"], h["stats"]
        rec["membrane"] = membrane_check(end, spec.membrane_target_mv, spec.membrane_tol_mv)
        if rec["membrane"]["ok"]:
            share = boundary(theta)
            if share > spec.homeo_boundary_share:
                return dict(status=BOUNDARY_LIMITED, boundary_share=share, cycles=cycles)
            return dict(status=None, params=h["params"], scale=float(s["accepted"]["x"]), boundary_share=share,
                        cycles=cycles)
        if prev_end is not None and cycle_stalled(prev_end, end, spec):
            return dict(status=STALLED, cycles=cycles)
        prev_end = end
    return dict(status=CYCLES_EXHAUSTED, cycles=cycles)


def c3_cell(m, ctx: Context, kc: float, order: int) -> dict:
    """One kc_thresh: cycles of (scale bisection at the current thresholds, homeostasis at that scale)."""
    spec, ex = ctx.spec, ctx.extra
    rule, has_pn = ex["rule_thresholds"](kc)
    update = ex["update_mask"]
    if (update & ~has_pn).any():
        raise ValueError("the update set must lie inside Engine.kc_pn_input > 0 (the loader keeps the rest at the rule)")
    files = ex["threshold_files"]
    base0 = c1_params(spec, kc, 1.0)

    def search(theta):
        ctx.log(f"C3 G({kc:g}): scale bisection")
        if np.array_equal(theta, rule):              # the rule's thresholds in pn_norm mode: C1's stage 1 exactly
            return search_scale(m, ctx, lambda s: dataclasses.replace(base0, apl_input_scale=s))
        return search_scale(m, ctx, lambda s: files.params(dataclasses.replace(base0, apl_input_scale=s), theta))

    def homeo(s, theta):
        return homeostasis(m, ctx, dataclasses.replace(base0, apl_input_scale=s), theta, rule, update, files)

    r = run_cycles(spec, search, homeo, rule.copy(), lambda th: boundary_share(th, rule, update, spec.homeo_clip))
    cell = dict(label=f"C3 G({kc:g})", order=order, kc=float(kc), cycles=r["cycles"])
    if r["status"] is None:
        cell.update(params=r["params"], scale=r["scale"], boundary_share=r["boundary_share"])
    else:
        cell["status"] = r["status"]
        if "boundary_share" in r:
            cell["boundary_share"] = r["boundary_share"]
    return cell


def c3_target(spec, c1: dict) -> tuple[float, str]:
    """A0 (H.3a.6 as amended): C1's adopted reference KC median as a fraction, rounded; the declared value otherwise."""
    if c1.get("status") == COMBO_ADOPTED:
        pct = float(c1["adopted"]["reference_median_kc_pct"])
        return round(pct / 100.0, spec.homeo_target_digits), f"C1 adopted {c1['adopted']['label']}: {pct:.6g}%"
    return float(spec.homeo_target), "declared (C1 adopted nothing)"


def run_c3(m, ctx: Context, c1: dict) -> dict:
    a0, source = c3_target(ctx.spec, c1)
    ctx = dataclasses.replace(ctx, spec=dataclasses.replace(ctx.spec, homeo_target=a0))
    ctx.log(f"C3: A0 = {a0} ({source})")
    spec = ctx.spec
    m = Deadline(m, ctx)
    inherited = None
    if c1.get("status") == COMBO_ADOPTED:
        inherited = float(c1["adopted"]["params"].kc_thresh)
    rest = [k for k in spec.kc_grid if k != inherited]
    cells, adopted = [], None
    out = dict(combo="C3", homeo_target=a0, homeo_target_source=source, cells=cells)
    try:
        if inherited is not None:
            cell = c3_cell(m, ctx, inherited, 0)
            cell["inherited_from_c1"] = True
            cell["inherited_scale_equal"] = bool(cell["cycles"][0]["search"]["accepted"]["x"]
                                                 == c1["adopted"]["params"].apl_input_scale)
            cells.append(cell)
            if "params" in cell:
                adopted = run_candidates(m, ctx, [cell], calibrate=True, check_membrane=True, judge_baseline=True)
        if adopted is None:
            own = []
            for kc in rest:
                own.append(c3_cell(m, ctx, kc, len(cells)))
                cells.append(own[-1])
            adopted = run_candidates(m, ctx, [c for c in own if "params" in c], calibrate=True, check_membrane=True,
                                     judge_baseline=True)
    except ComputeAborted as e:
        return dict(out, status=COMBO_ABORTED, adopted=None, note=str(e))
    return dict(out, status=COMBO_ADOPTED if adopted else COMBO_DROPPED, adopted=adopted_record(adopted))
