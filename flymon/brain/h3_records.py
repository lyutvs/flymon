"""M0d H.3 records (spec appendix H.3a.8): measured for each combination's adopted candidate, never judged.

- membrane: median and quartiles of the per-presentation APL mean, and the median over presentations of each
  instantaneous quantile (read steps x APL cells);
- release occupancy: share of presentations whose mean release fraction lies below / inside / above the sigmoid's
  quasi-linear range (graded APL only);
- a_i: firing-fraction distribution over the H.3a.6 update set;
- APL input by source class (KC / ALPN / MBON / other, spike-driven, per read step) — MBON -> APL is the readout
  feedback H.3a.2 asks to record — and the APL-derived input to the callout MBON types (MBON05, MBON13);
- half-split sampling error (24 + 24 odours) of membrane and KC activity, flagged above the 3-SD thresholds;
- guard: odour-cluster CIs of every pool type (the verdict used point estimates);
- all51: per-glomerulus KC drive, log10 variance and zero-drive count; for graded APL also against the clamped
  (fixed-release) engine with paired glomerulus-bootstrap CIs;
- gain control (graded APL): 384-odour clamped-release contrast of the odour-to-odour KC IQR, paired odour bootstrap;
- APL -> MBON edges zeroed, and (graded APL) the same scale applied to KC -> APL edges only: diagnostic CSC edits.
  Spec H.3a.11 keeps a CSC-editing diagnostic under results/m0d/diag/: their numbers are written only to
  `ctx.extra["diag_dir"]` and the record the run report and the summary carry is the edited weights' sha256 and
  that file's path and sha256.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from .h3_rules import (firing_fraction, guard_pool, half_split_error, iqr, log10_var, mbon_type_boot,
                       paired_boot_ci, per_odor_means, reference_stats, zero_count)
from .h3_store import canonical, sha256_file, write_json


def _median(rows, fn):
    return float(np.median([fn(p) for p in rows]))


def membrane_record(rows, spec) -> dict:
    st = reference_stats(rows, spec.release_quasi_linear)
    inst = np.array([p["apl_v_quantiles"] for p in rows], float)
    return dict(median_mv=st["median_mv"], q1_mv=st["q1_mv"], q3_mv=st["q3_mv"],
                instantaneous={f"p{q:g}": float(np.median(inst[:, i])) for i, q in enumerate(spec.apl_v_quantiles)},
                release={k: st[k] for k in ("median_release_frac", "rel_share_below", "rel_share_in",
                                            "rel_share_above") if k in st})


def ai_record(rows, n_kc: int, update, spec) -> dict:
    a = firing_fraction(rows, n_kc)[np.asarray(update, bool)]
    q = np.percentile(a, [25, 50, 75, 90, 99])
    return dict(n_kc=int(a.size), share_zero=float((a == 0).mean()), q1=float(q[0]), median=float(q[1]),
                q3=float(q[2]), p90=float(q[3]), p99=float(q[4]), max=float(a.max()),
                n_ge_mark=int((a >= spec.ai_clip_mark).sum()))


def apl_input_record(rows) -> dict:
    by = {c: _median(rows, lambda p, c=c: p["apl_input_by_class"][c]) for c in ("kc", "alpn", "mbon", "other")}
    tot = sum(by.values())
    return dict(median_mv_per_step=by, share={c: (v / tot if tot else None) for c, v in by.items()},
                apl_to_type=({n: _median(rows, lambda p, n=n: p["apl_to_type_input"][n])
                              for n in rows[0]["apl_to_type_input"]} if rows else {}))


def all51_drive(rows) -> tuple[list, np.ndarray]:
    gs = sorted({r["g"] for r in rows})
    return gs, np.array([np.mean([r["kc"] for r in rows if r["g"] == g]) for g in gs])


def clamped(params, r_bar: float, spec):
    """The fixed-release engine: the sigmoid made flat at 0.5 and apl_r_max chosen so the release matches R-bar."""
    return dataclasses.replace(params, apl_slope=spec.clamp_slope, apl_r_max=2.0 * r_bar * params.apl_r_max)


def gain_control(m, ctx, params) -> dict:
    spec = ctx.spec
    a_rows = m.reference(params, "extended")
    r_bar = float(np.mean([p["release_frac"] for p in a_rows]))
    release_a = float(np.mean([p["release_mean"] for p in a_rows]))
    b_params = clamped(params, r_bar, spec)
    b_rows = m.reference(b_params, "extended")
    release_b = float(np.mean([p["release_mean"] for p in b_rows]))
    rel = abs(release_b - release_a) / release_a if release_a else float("inf")
    names = sorted({p["odor"] for p in a_rows})
    kc_a = per_odor_means(a_rows, "kc_active_frac", names, 100.0)
    kc_b = per_odor_means(b_rows, "kc_active_frac", names, 100.0)
    out = dict(r_bar=r_bar, r_max_b=float(b_params.apl_r_max), release_a=release_a, release_b=release_b,
               release_rel_diff=float(rel), release_match_ok=bool(rel <= spec.release_match_tol), n_odors=len(names),
               iqr_a=iqr(kc_a), iqr_b=iqr(kc_b), b_params=b_params)
    if out["release_match_ok"]:
        out["iqr_diff"] = paired_boot_ci(kc_a, kc_b, iqr, spec.boot_draws, spec.boot_seed)
    return out


def all51_record(m, ctx, params, graded: bool, r_bar: float | None) -> dict:
    spec = ctx.spec
    gs, ka = all51_drive(m.all51(params))
    out = dict(glomeruli=gs, drive_a=ka.tolist(), log10_var_a=log10_var(ka), zero_count_a=zero_count(ka))
    if graded and r_bar is not None:
        _, kb = all51_drive(m.all51(clamped(params, r_bar, spec)))
        out.update(drive_b=kb.tolist(), log10_var_b=log10_var(kb), zero_count_b=zero_count(kb),
                   log10_var_diff=paired_boot_ci(ka, kb, log10_var, spec.boot_draws, spec.boot_seed),
                   zero_count_diff=paired_boot_ci(ka, kb, zero_count, spec.boot_draws, spec.boot_seed))
    return out


def edit_contrast(m, ctx, params, csc_edit) -> dict:
    """The contrast's numbers go to a file under the diagnostic directory; the returned record holds only hashes."""
    spec = ctx.spec
    rows = m.reference(params, csc_edit=csc_edit)
    rest = {r["seed"]: r for r in m.rest(params, ctx.probe_seeds, csc_edit=csc_edit)}
    st = reference_stats(rows, spec.release_quasi_linear)
    body = dict(params=params, csc_edit=list(csc_edit), csc_sha256=rows[0].get("csc_sha256"), median_mv=st["median_mv"],
                median_kc_pct=st["median_kc_pct"],
                guard={k: guard_pool(rows, rest, ctx.pools[k], ctx.odor_names, spec) for k in ("A", "P")})
    name = hashlib.sha256(canonical(dict(params=params, csc_edit=list(csc_edit))).encode()).hexdigest()[:24]
    path = write_json(Path(ctx.extra["diag_dir"]) / f"contrast-{name}.json", body, [params])
    return dict(csc_edit=list(csc_edit), csc_sha256=body["csc_sha256"], path=str(path), file_sha256=sha256_file(path))


def collect(m, ctx, params) -> dict:
    spec = ctx.spec
    graded = params.apl_mode == "graded"
    rows = m.reference(params)
    rest = {r["seed"]: r for r in m.rest(params, ctx.probe_seeds)}
    rec = dict(membrane=membrane_record(rows, spec), apl_input=apl_input_record(rows),
               half_split=half_split_error(rows, ctx.odor_names, spec.guard_half, spec.half_split_mv,
                                           spec.half_split_pp),
               guard_ci={n: mbon_type_boot(rows, rest, n, ctx.odor_names, spec.boot_draws, spec.boot_seed)
                         for k in ("A", "P") for n in ctx.pools[k]})
    if "update_mask" in ctx.extra:
        rec["a_i"] = ai_record(rows, ctx.n_kc, ctx.extra["update_mask"], spec)
    gc = gain_control(m, ctx, params) if graded else None
    rec["gain_control"] = gc
    rec["all51"] = all51_record(m, ctx, params, graded, gc["r_bar"] if gc else None)
    rec["apl_to_mbon_zero"] = edit_contrast(m, ctx, params, ("apl_to_mbon_zero",))
    if graded:
        rec["kc_to_apl_only"] = edit_contrast(m, ctx, dataclasses.replace(params, apl_input_scale=1.0),
                                              ("kc_to_apl_scale", float(params.apl_input_scale)))
    return rec
