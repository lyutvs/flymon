# flymon/brain/q_records.py
"""Q's per-pair values and gates (Readings 5, 16, 17). pair_values reads one oracle result (an encoder Q0 cache entry
or a Q1 q_oracle_job result) over its report seeds; q0_record is Q0's gate, values and F/S split (Q.3, Q.6.1, Q.6.7);
condition_record is one Q1 condition's gate (`INVALID` reasons) and records. Nothing here runs the engine.
Every condition's reach (W_X, f_X, by type) is recorded per pair, but ③ reads W_X / f_X from the BASE condition only
(Q.3, Q.6.2). Condition "s_up" is (c), which LOWERS s (0.7, fallback 0.8; Q.6.9) despite the name. naive_px_zero counts
the pairs whose naive P_X is 0 — "already at floor", outside ①'s rho (Q.6.9)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import d6a
from .h4_formula import dv, pair_stats
from .h4_rules import _only
from .q_pairs import key_str  # noqa: F401  (re-exported)

OK, INVALID = "OK", "INVALID"


def _col(probe: dict, k: str, j: int) -> np.ndarray:
    return np.asarray(probe[k], float)[:, j]


def _sd(x) -> float | None:
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if x.size >= 2 else None


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def pair_values(res: dict, z: dict, spec) -> dict | None:
    rep = res["report"]
    st = pair_stats(rep, z, spec.split_r)
    sp_ = pair_stats(_only(rep, "P"), z, spec.split_r)
    if st is None or sp_ is None:
        return None
    pre, r1 = rep["pre"], rep["R1"]
    dV = dv(r1, z) - dv(pre, z)
    px0, px1 = _col(pre, "P", 0), _col(r1, "P", 0)
    naive_px = float(px0.mean())
    kx, ky = res["kc"]["x"], res["kc"]["y"]
    out = dict(r=st["r"], d_pre=st["d_pre"], p=st["p"], m=st["m"], r_P=sp_["r"], naive_px=naive_px,
               naive_py=float(_col(pre, "P", 1).mean()), naive_ax=float(_col(pre, "A", 0).mean()),
               naive_ay=float(_col(pre, "A", 1).mean()), alpha_reward=float(res["alpha_reward"]),
               dV_mean=float(dV.mean()), dV_sd=_sd(dV), dpx_mean=float((px1 - px0).mean()), dpx_sd=_sd(px1 - px0),
               decomp={f"d{k}_{s}": float((_col(r1, k, j) - _col(pre, k, j)).mean())
                       for k in ("A", "P") for j, s in ((0, "X"), (1, "Y"))},
               stop_naive=float((px0 < spec.stop_p).mean()), stop_post=float((px1 < spec.stop_p).mean()),
               kc_frac=[float(v) for v in list(kx["frac"]) + list(ky["frac"])],
               kc_max_win=[int(v) for v in list(kx["max_win"]) + list(ky["max_win"])],
               jaccard=float(res["kc"]["jaccard"]), n_report=int(px0.size))
    q = res.get("q")
    if q is not None:
        out["dpx_fixed"] = {a: float((_col(f["report"], "P", 0) - px0).mean()) for a, f in q["fixed"].items()}
        out["px_after_fixed"] = {a: float(_col(f["report"], "P", 0).mean()) for a, f in q["fixed"].items()}
        a0 = str(float(spec.fixed_alphas[0]))
        out["ratio"] = None if naive_px == 0 or a0 not in out["dpx_fixed"] else abs(out["dpx_fixed"][a0]) / naive_px
        rc = q["reach"]
        out.update(W_X=float(rc["W_X"]), f_X=rc["f_X"], n_edges=int(rc["n_edges"]), n_active_kc=int(rc["n_active_kc"]),
                   reach_by_type=dict(rc["by_type"]), apl_out=[float(v) for v in q["apl_out"]["x"] + q["apl_out"]["y"]],
                   csc_sha256=q["csc_sha256"], edit_edges=int(q["edit_edges"]))
    return out


def _stat_ok(v) -> bool:
    return v is not None and all(_finite(v[f]) for f in ("r", "r_P", "dV_mean", "dV_sd", "dpx_mean", "naive_px"))


def _floor(vals: dict) -> int:
    """Q.6.9: pairs with naive P_X = 0 ("already at floor"), recorded apart from ①'s rho."""
    return sum(v["naive_px"] == 0 for v in vals.values())


def split(r_by_key: dict, spec) -> dict:
    return dict(F=sorted(k for k, r in r_by_key.items() if r < spec.split_r),
                S=sorted(k for k, r in r_by_key.items() if r >= spec.split_r))


def q0_record(files: list, results: list, spec, z: dict) -> dict:
    """Q0 (no simulation): files = q_pairs.q0_files rows, results = each file's "result" (None when unreadable)."""
    reasons = [f"Q0 cache digest mismatch: {f['key']}" for f in files if not f["ok"]]
    vals = {}
    n = len(spec.repro_report_seeds)
    for f, res in zip(files, results):
        if res is None or not f["ok"]:
            continue
        if any(len(res["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P")):
            reasons.append(f"{f['key']}: report probes are not the {n} H.4 report seeds")
            continue
        v = pair_values(res, z, spec)
        if not _stat_ok(v):
            reasons.append(f"{f['key']}: undefined or non-finite statistic")
            continue
        vals[f["key"]] = v
    if len(vals) != spec.n_b and not reasons:
        reasons.append(f"{len(vals)} Q0 pairs, not {spec.n_b}")
    sp = split({k: v["r"] for k, v in vals.items()}, spec) if not reasons else None
    if sp and (len(sp["F"]), len(sp["S"])) != (spec.n_f, spec.n_s):
        reasons.append(f"Q0 split F {len(sp['F'])} / S {len(sp['S'])}, declared {spec.n_f} / {spec.n_s}")
    lo, hi = spec.borderline
    border = [k for k in (sp["F"] if sp else []) if lo <= vals[k]["r"] <= hi]
    cancel = [k for k in (sp["F"] if sp else []) if vals[k]["r_P"] >= spec.split_r]
    return dict(status=INVALID if reasons else OK, reasons=reasons, files=files, vals=vals, split=sp,
                borderline=border, cancel=cancel, naive_px_zero=_floor(vals))


def condition_record(got: list, cond, spec, z: dict, expected: list, same_sha: str | None = None,
                     other_sha: str | None = None) -> dict:
    reasons = []
    keys = [g["key"] for g in got]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    exp = set(expected)
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    n = len(spec.report_seeds)
    want_fixed = {str(float(a)) for a in spec.fixed_alphas}
    vals = {}
    for g in got:
        res = g["result"]
        q = res.get("q")
        short = (any(len(res["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P")) or q is None
                 or set(q["fixed"]) != want_fixed or any(len(f["report"]["P"]) != n for f in q["fixed"].values()))
        if short:
            reasons.append(f"{g['key']}: probes are not the {n} report seeds or a fixed arm is missing")
            continue
        v = pair_values(res, z, spec)
        if not _stat_ok(v):
            reasons.append(f"{g['key']}: undefined or non-finite statistic")
            continue
        vals[g["key"]] = v
    shas = sorted({v["csc_sha256"] for v in vals.values()})
    if len(shas) > 1:
        reasons.append(f"CSC sha256 differs between rows: {shas}")
    if same_sha is not None and shas and shas != [same_sha]:
        reasons.append(f"CSC sha256 {shas} is not the reproduction gate's {same_sha}")
    if other_sha is not None and shas == [other_sha]:
        reasons.append("edit left the CSC sha256 equal to the unedited one")
    edges = sorted({v["edit_edges"] for v in vals.values()})
    if cond.edit == spec.edit_mbon05 and edges and edges != [spec.apl_mbon05_edges]:
        reasons.append(f"{cond.edit} changed {edges} edges, declared {spec.apl_mbon05_edges}")
    if cond.edit == spec.edit_nonkc and edges and (len(edges) != 1 or edges[0] <= 0):
        reasons.append(f"{cond.edit} changed {edges} edges")
    if cond.edit == spec.edit_none and edges and edges != [0]:
        reasons.append(f"edit none changed {edges} edges")
    fr = [x for v in vals.values() for x in v["kc_frac"]]
    kc_med = float(np.median(fr)) if fr else None
    lo, hi = spec.kc_band
    if cond.kc_gate and (kc_med is None or not lo <= kc_med <= hi):
        reasons.append(f"KC 활성 중앙값 {kc_med} 대역 [{lo}, {hi}] 밖")
    wins = [w for v in vals.values() for w in v["kc_max_win"]]
    apl = [a for v in vals.values() for a in v.get("apl_out", [])]
    return dict(condition=dataclasses.asdict(cond), status=INVALID if reasons else OK, reasons=reasons, vals=vals,
                kc_median=kc_med,
                d6a_over_share=(float(np.mean([w >= d6a.OVER_SPIKES for w in wins])) if wins else None),
                d6a_condition=d6a.CONDITION, apl_out_median=(float(np.median(apl)) if apl else None),
                csc_sha256=shas[0] if len(shas) == 1 else shas, edit_edges=edges,
                naive_px_median=(float(np.median([v["naive_px"] for v in vals.values()])) if vals else None),
                naive_px_zero=_floor(vals),
                jaccard_median=(float(np.median([v["jaccard"] for v in vals.values()])) if vals else None),
                manifest=[dict(key=g["key"], cache_key=g.get("cache_key"), cache_file=g.get("cache_file")) for g in got])
