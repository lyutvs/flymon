"""AD.5 3: FLY-OS's strength balance on the naive L_V brain (w0). For every confirmation-set candidate situation (20
pairs x 2 sides x candidates) and decision seed r in {0, 1, 2}, one request with the FLY odours (true opponent) and 16
requests with the OS odours of each POOL species (uniform weights = the draw distribution), all with the seed
derive_seed(305, "os-calib", r, pair, side) (paired noise). Measured: R_r = mean OS candidate KC-active count / mean
FLY count, R = the mean of the three, SD_R (ddof 1), the ratio per move type and per true-opponent single / dual type,
the KS distance between the FLY and the pooled OS counts, and their 10 / 50 / 90 percentiles. The measurement is
committed before band() judges it: R within 1 +- max(0.15, 3 SD_R), else STOP_BALANCE (a scientific balance failure,
not a code defect)."""
from __future__ import annotations

import numpy as np

from ..ac.situations import offline_odour
from ..agent.policy import derive_seed
from . import os_encoder
from .spec import LABEL

CALIB = "results/summary/ad_os_kc_calib.json"


def requests(grid, pairs, rep: int, tie_seed: int) -> tuple:
    meta, reqs = [], []
    for i, p in enumerate(pairs):
        for side, opp in ((0, p["o1"]), (1, p["o2"])):
            seed = derive_seed(int(tie_seed), "os-calib", int(rep), i, side)
            variants = [("fly", [offline_odour(grid, m, opp) for m in p["cands"]])]
            variants += [(sp, [os_encoder.odour_for_species(grid, grid.move_info[m][0], sp) for m in p["cands"]])
                         for sp in os_encoder.SPECIES]
            for name, ods in variants:
                meta.append(dict(rep=int(rep), pair=i, side=side, variant=name, true=opp, cands=list(p["cands"])))
                reqs.append((0, ods, seed))
    return meta, reqs


def kc_rows(meta, counts, n_skip: int) -> list:
    return [dict(m, kc=[int(x) for x in (np.asarray(c)[:, int(n_skip):] > 0).sum(1)]) for m, c in zip(meta, counts)]


def ks(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.union1d(a, b)
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _ratio(rows, keep) -> float | None:
    fly = [k for r in rows if r["variant"] == "fly" for k, c in zip(r["kc"], r["cands"]) if keep(r, c)]
    os_ = [k for r in rows if r["variant"] != "fly" for k, c in zip(r["kc"], r["cands"]) if keep(r, c)]
    return float(np.mean(os_) / np.mean(fly)) if fly and os_ and np.mean(fly) > 0 else None


def measure(rows, move_info, species_types) -> dict:
    reps = sorted({r["rep"] for r in rows})
    by_rep = [_ratio([r for r in rows if r["rep"] == k], lambda r, c: True) for k in reps]
    mtypes = sorted({move_info[c][0] for r in rows for c in r["cands"]})
    by_type = {t: _ratio(rows, lambda r, c, t=t: move_info[c][0] == t) for t in mtypes}
    dual = lambda r: "dual" if len(species_types[r["true"]]) == 2 else "single"
    by_dual = {d: _ratio(rows, lambda r, c, d=d: dual(r) == d) for d in ("single", "dual")}
    fly = [k for r in rows if r["variant"] == "fly" for k in r["kc"]]
    os_ = [k for r in rows if r["variant"] != "fly" for k in r["kc"]]
    pct = lambda x: {str(q): float(np.percentile(x, q)) for q in (10, 50, 90)}
    return dict(R_by_rep=by_rep, R=float(np.mean(by_rep)), sd_R=float(np.std(by_rep, ddof=1)) if len(by_rep) > 1 else 0.0,
                by_move_type=by_type, by_true_dual=by_dual, ks=ks(fly, os_), percentiles=dict(fly=pct(fly), os=pct(os_)),
                mean_kc=dict(fly=float(np.mean(fly)), os=float(np.mean(os_))), n_fly=len(fly), n_os=len(os_),
                reps=reps, label=LABEL)


def band(meas: dict, spec) -> dict:
    half = max(spec.kc_band_min, spec.kc_band_sd * float(meas["sd_R"]))
    R = float(meas["R"])
    return dict(status="OK" if abs(R - 1.0) <= half else "STOP_BALANCE", R=R, sd_R=float(meas["sd_R"]),
                half_width=half, lo=1.0 - half, hi=1.0 + half)
