"""W's operating characteristic (W.9.3 as amended by W.9.8 H1 / H4, W.9.9 P0-1 / P1-2 / P1-3 / P1-6 / P2-8 / P2-11
and W.9.10 2), committed with the verdict code before the pilot (W.9.9 순서 0). Nothing here runs the engine.

Model (probe level, fitted to the pilot; plan Readings 9-11):
- Slots SLOTS = pre, R1, R2, N1, N2, RN2 (RN1 is R1 copied, P1-2), counts per (cell A/P, odour X/Y).
- A pair's naive mean m = m0s + u + w: m0s = the pilot's grand naive mean with X and Y averaged per cell (a
  naive-balanced population pair), u ~ N(0, Σ_pair) the 2 × 2 pair random effect (cell × odour), w ~ N(0, g·Σ_pair)
  the cluster random effect shared by every pair of an experiment (g on the grid 0 / 0.5 / 1, the worst value used,
  P1-6). u + w is drawn conditioned on the pair being naive-balanced (|E ΔV_pre| / sd_pre < 0.5): gate pairs are the
  pairs that passed the naive screen and the OC counts no naive dropout (W.9.3).
- A fly adds v ~ N(0, Σ_fly) (same 2 × 2 structure) to every post-training slot; pre has no fly effect (one naive
  brain).
- Fixed stage effects from the pilot (P1-6): N1 = m + v + δ_N1, N2 = N1 + δ_N2, R1 = N1 + E1, RN2 = R1 + δ_RN2,
  R2 = RN2 + E2, with E1 = −a on MBON05(X) and −s_P·a on MBON05(Y), E2 = −b on MBON13(X) and −s_A·b on MBON13(Y)
  (s = the pilot's Y spillover, ratio of mean changes; s = 1 when the pilot's X change is not a depression).
- Probe noise: whole residual vectors (6 slots × 2 × 2, around each pilot fly's slot means, scaled by √(K/(K−1)))
  resampled from the pilot — the paired-noise correlation across slots, cells and odours is kept. Counts are
  max(0, rint(mean + residual)): the floor truncation.
- Calibration (W.9.9 P0-1): the population pair (u = w = v = 0) over cal_reps residual draws (common random numbers)
  gives each gate's true d′ (mean / sd over the draws). a is solved by bisection so that min(reward level, reward
  association) = 1.5 (power) or max(...) = 0.5 (false pass); then b, given a, for the punishment drop and
  association. Tolerance ±0.02, at most 40 steps; a target not bracketed on [0, the floor] or no convergence →
  unreachable. W.9.10 2 (cal_floor_rule "zero", the spec's value): the false-pass null is zero DAN injection — a
  handle (reward a, punishment b) whose gate pair maximum is already ≥ 0.5 at zero injection (the level gates do not
  subtract N / RN, so presentation drift can do this) is kept at zero injection (status "zero_floor"), not
  unreachable; the power calibration (min = 1.5) is unchanged. ("unreachable" = the literal reading, kept as a
  switch.) The drift-only gate d′ at zero injection is recorded per pair and per gate (drift_dprimes) for the OC
  block and W.10.
- Experiments: 8 pairs × 32 flies × 2·max(K) probes generated once per replicate and cut to every design (nested,
  common random numbers, P2-8); the W verdict code (w_verdict) runs on them: fly gates, fly classes, the pair code
  with q and F, the mechanism control, the 2K BAND resolution, RN1 = R1, the overall order. P(PASS)[q, K, F, k].
- Uncertainty (P1-3): a pair-level parametric bootstrap (boot_draws): J new pilot pairs drawn from the fitted model
  (pair means ~ N(m0, Σ_pair), per-pair drifts and spillover changes ~ their pilot normals, fly effects ~ N(0,
  Σ_fly) re-estimated, residual blocks resampled by pair) and refitted; each draw recalibrated and simulated with
  boot_reps experiments. Simultaneous one-sided limits over k: power = the 5th percentile over draws of min_k
  P(PASS), false pass = the 95th percentile of max_k P(PASS); the worst over the cluster grid.
- Selection (H1, W.9.9): a design (q, K, F) qualifies when its limits meet G.6 at every k (power ≥ 0.80, false pass
  ≤ 0.05) for F, F+1, F+2, F+3 (F ≥ 29: F alone); the cheapest qualifying design by the total wall-clock estimate
  wins (ties: larger q, then smaller K, then smaller F); every qualifying design in cost order is the alternative
  ranking (P1-4). None → STOP_OC_UNREACHABLE."""
from __future__ import annotations

import math
import time

import numpy as np

from . import w_verdict as WV

SLOTS = ("pre", "R1", "R2", "N1", "N2", "RN2")
SI = {s: i for i, s in enumerate(SLOTS)}
POST = (SI["R1"], SI["R2"], SI["N1"], SI["N2"], SI["RN2"])
# Every random stream is np.random.SeedSequence([oc_seed, tag, ...]) — deterministic, no Python hash.
TAG_CAL, TAG_POINT, TAG_BOOT, TAG_BOOT_SIM, TAG_RECORD, TAG_SYNTH = 1, 2, 3, 4, 5, 9
RECORD_TAGS = ("t0.5", "t1.0", "t1.5", "t2.0", "one", "half", "all1", "mixed")


def _rng(spec, *tags):
    return np.random.default_rng(np.random.SeedSequence([int(spec.oc_seed)] + [int(t) for t in tags]))


# ================================================================ fitting
def _stack(pilot: list) -> np.ndarray:
    """[J, F, K, 6, 2, 2] float from the pilot pairs' {stage: [F, K, 2, 2]} (RN1 is not a slot)."""
    return np.stack([np.stack([np.asarray(p[s], float) for s in SLOTS], axis=2) for p in pilot])


def _cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).reshape(len(x), -1)
    return np.cov(x, rowvar=False) if len(x) > 1 else np.zeros((x.shape[1], x.shape[1]))


def _psd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    return (v * np.clip(w, 0, None)) @ v.T


def _spill(x: float, y: float) -> float:
    return max(0.0, y / x) if x < 0 else 1.0


def assemble(pair_means, drift_pair, spill_pair, fly_cov, resid_blocks, n_fly: int, n_probe: int) -> dict:
    pm = np.asarray(pair_means, float)
    m0 = pm.mean(0)
    dp, sp = np.asarray(drift_pair, float), np.asarray(spill_pair, float)
    smean = sp.mean(0)
    return dict(pair_means=pm, m0=m0, m0s=np.repeat(m0.mean(-1, keepdims=True), 2, -1), pair_cov=_cov(pm),
                drift_pair=dp, drift=dp.mean(0), spill_pair=sp, spill=(_spill(smean[0, 0], smean[0, 1]),
                                                                       _spill(smean[1, 0], smean[1, 1])),
                fly_cov=_psd(np.asarray(fly_cov, float)), resid_blocks=[np.asarray(b, float) for b in resid_blocks],
                resid=np.concatenate(resid_blocks), n_fly=int(n_fly), n_probe=int(n_probe))


def fit(pilot: list) -> dict:
    """θ from the pilot pairs ({stage: int [F, K, 2, 2]}, K ≥ 2, F ≥ 2)."""
    a = _stack(pilot)
    J, F, K = a.shape[:3]
    pre, r1, r2, n1, n2, rn2 = (a[:, :, :, i] for i in range(6))
    pair_means = pre.mean((1, 2))
    drift_pair = np.stack([(n1 - pre).mean((1, 2)), (n2 - n1).mean((1, 2)), (rn2 - r1).mean((1, 2))], 1)
    rew, pun = (r1 - n1).mean((1, 2)), ((r2 - r1) - (rn2 - r1)).mean((1, 2))
    spill_pair = np.stack([rew[:, WV.P, :], pun[:, WV.A, :]], 1)              # [J, 2 (reward P, punish A), 2 (X, Y)]
    x = a[:, :, :, list(POST)].mean(3).reshape(J, F, K, 4)
    mf = x.mean(2)
    wcov = np.einsum("jfki,jfkl->il", x - mf[:, :, None], x - mf[:, :, None]) / (J * F * (K - 1))
    dm = mf - mf.mean(1, keepdims=True)
    bcov = np.einsum("jfi,jfl->il", dm, dm) / (J * (F - 1))
    resid = (a - a.mean(2, keepdims=True)) * math.sqrt(K / (K - 1))
    blocks = [resid[j].reshape(F * K, 6, 2, 2) for j in range(J)]
    return assemble(pair_means, drift_pair, spill_pair, bcov - wcov / K, blocks, F, K)


def boot(theta: dict, rng) -> dict:
    """One pair-level parametric bootstrap draw (module docstring)."""
    J, F = len(theta["pair_means"]), theta["n_fly"]
    pm = rng.multivariate_normal(theta["m0"].ravel(), theta["pair_cov"], J, method="eigh").reshape(J, 2, 2)
    dp = rng.multivariate_normal(theta["drift"].ravel(), _cov(theta["drift_pair"]), J, method="eigh")
    sp = rng.multivariate_normal(theta["spill_pair"].mean(0).ravel(), _cov(theta["spill_pair"]), J, method="eigh")
    v = rng.multivariate_normal(np.zeros(4), theta["fly_cov"], J * max(F - 1, 1), method="eigh")
    fly_cov = v.T @ v / len(v)
    pick = rng.integers(0, J, J)
    return assemble(pm, dp.reshape(J, 3, 2, 2), sp.reshape(J, 2, 2), fly_cov,
                    [theta["resid_blocks"][i] for i in pick], F, theta["n_probe"])


def summary(theta: dict) -> dict:
    return dict(m0=theta["m0"].tolist(), m0s=theta["m0s"].tolist(), pair_cov=theta["pair_cov"].tolist(),
                fly_cov=theta["fly_cov"].tolist(), drift=dict(zip(("N1-pre", "N2-N1", "RN2-RN1"),
                                                                 theta["drift"].tolist())),
                spill=dict(P=theta["spill"][0], A=theta["spill"][1]), n_pairs=len(theta["pair_means"]),
                n_resid=int(len(theta["resid"])), n_fly=theta["n_fly"], n_probe=theta["n_probe"])


# ================================================================ the generator
def slot_means(theta: dict, base, v, a, b) -> np.ndarray:
    """[..., 6, 2, 2] slot means from the naive base [..., 2, 2], the fly effect v [..., 2, 2] and the effects a, b."""
    base, v = np.asarray(base, float), np.asarray(v, float)
    a, b = np.asarray(a, float)[..., None, None], np.asarray(b, float)[..., None, None]
    sP, sA = theta["spill"]
    d = theta["drift"]
    e1 = np.zeros((2, 2)); e1[WV.P, WV.X], e1[WV.P, WV.Y] = -1.0, -sP
    e2 = np.zeros((2, 2)); e2[WV.A, WV.X], e2[WV.A, WV.Y] = -1.0, -sA
    n1 = base + v + d[0]
    r1 = n1 + a * e1
    rn2 = r1 + d[2]
    return np.stack([base + 0 * v, r1, rn2 + b * e2, n1, n1 + d[1], rn2], axis=-3)


def to_stages(counts: np.ndarray) -> dict:
    """counts [..., K, 6, 2, 2] -> {stage: [..., K, 2, 2]} with RN1 = R1 (P1-2)."""
    d = {s: counts[..., i, :, :] for s, i in SI.items()}
    d["RN1"] = d["R1"].copy()
    return d


def sd_pre(theta: dict, z: dict) -> float:
    pre = np.clip(np.rint(theta["m0s"] + theta["resid"][:, SI["pre"]]), 0, None)
    return float(np.std(WV.dv(pre, z), ddof=1))


def _pair_bases(theta, rng, n_rep, n_pair, g, z, naive_max, sdp):
    """[n_rep, n_pair, 2, 2] naive bases conditioned on balance; the cluster effect is shared within a replicate."""
    L = np.linalg.cholesky(theta["pair_cov"] + 1e-9 * np.eye(4))
    w = (rng.standard_normal((n_rep, 4)) @ L.T * math.sqrt(g)).reshape(n_rep, 1, 2, 2)
    out = np.empty((n_rep, n_pair, 2, 2))
    need = np.ones((n_rep, n_pair), bool)
    for _ in range(200):
        u = (rng.standard_normal((n_rep, n_pair, 4)) @ L.T).reshape(n_rep, n_pair, 2, 2)
        base = theta["m0s"] + w + u
        ok = np.abs(WV.dv(base, z)) / sdp < naive_max
        take = need & ok
        out[take] = base[take]
        need &= ~ok
        if not need.any():
            return out
    out[need] = (theta["m0s"] + w + 0 * out)[need]                   # unreachable balance: the population pair
    return out


def simulate(theta, rng, n_rep, n_pair, n_fly, n_probe, a_pair, b_pair, fly_a, fly_b, g, z, naive_max=0.5):
    """{stage: int [n_rep, n_pair, n_fly, n_probe, 2, 2]}; a_pair / b_pair [n_pair], fly_a / fly_b [n_fly]
    multipliers."""
    sdp = sd_pre(theta, z)
    base = _pair_bases(theta, rng, n_rep, n_pair, g, z, naive_max, sdp)
    Lf = np.linalg.cholesky(theta["fly_cov"] + 1e-9 * np.eye(4))
    v = (rng.standard_normal((n_rep, n_pair, n_fly, 4)) @ Lf.T).reshape(n_rep, n_pair, n_fly, 2, 2)
    a = np.asarray(a_pair, float)[None, :, None] * np.asarray(fly_a, float)[None, None, :]
    b = np.asarray(b_pair, float)[None, :, None] * np.asarray(fly_b, float)[None, None, :]
    mu = slot_means(theta, base[:, :, None], v, np.broadcast_to(a, v.shape[:3]), np.broadcast_to(b, v.shape[:3]))
    idx = rng.integers(0, len(theta["resid"]), (n_rep, n_pair, n_fly, n_probe))
    counts = np.clip(np.rint(mu[:, :, :, None] + theta["resid"][idx]), 0, None).astype(np.int32)
    return to_stages(counts)


# ================================================================ calibration (W.9.9 P0-1)
def true_dprimes(theta, a, b, idx, z) -> np.ndarray:
    """The four gates' directional true d′ (sign · d′) of the population pair over the residual draws idx."""
    mu = slot_means(theta, theta["m0s"], np.zeros((2, 2)), a, b)
    c = np.clip(np.rint(mu[None] + theta["resid"][idx]), 0, None)
    d = to_stages(c[None])                                       # [1 fly, n probes, 2, 2]
    return WV.SIGNS * WV.gate_stats(d, z)[0]


def _bisect(f, hi, tol, n_iter):
    lo = 0.0
    f_lo, f_hi = f(lo), f(hi)
    if abs(f_lo) <= tol:
        return dict(value=lo, status="ok", steps=0)
    if f_lo > 0:
        return dict(value=None, status="above_at_zero", f0=float(f_lo))
    if f_hi < 0:
        return dict(value=None, status="floor", f_hi=float(f_hi))
    for i in range(n_iter):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) <= tol:
            return dict(value=mid, status="ok", steps=i + 1)
        lo, hi = (mid, hi) if fm < 0 else (lo, mid)
    return dict(value=None, status="no_convergence")


def calibrate(theta, target: float, mode: str, idx, z, spec) -> dict:
    """a then b for "min" (power, record) or "max" (false pass) of the gate pairs at `target`."""
    agg = np.min if mode == "min" else np.max
    hi_a = float((theta["m0s"][WV.P, WV.X] + theta["drift"][0][WV.P, WV.X] + np.abs(theta["resid"]).max()) * 2 + 1)
    ra = _bisect(lambda x: agg(true_dprimes(theta, x, 0.0, idx, z)[[0, 2]]) - target, hi_a, spec.cal_tol,
                 spec.cal_iter)
    if ra["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        ra = dict(value=0.0, status="zero_floor", f0=ra["f0"])
    if ra["value"] is None:
        return dict(ok=False, a=ra, b=None)
    hi_b = float((theta["m0s"][WV.A, WV.X] + np.abs(theta["drift"]).sum() + np.abs(theta["resid"]).max()) * 2 + 1)
    rb = _bisect(lambda x: agg(true_dprimes(theta, ra["value"], x, idx, z)[[1, 3]]) - target, hi_b, spec.cal_tol,
                 spec.cal_iter)
    if rb["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        rb = dict(value=0.0, status="zero_floor", f0=rb["f0"])
    if rb["value"] is None:
        return dict(ok=False, a=ra, b=rb)
    td = true_dprimes(theta, ra["value"], rb["value"], idx, z)
    return dict(ok=True, a=ra, b=rb, true_dprime=dict(zip(WV.GATES, td.tolist())))


def drift_dprimes(theta, idx, z) -> dict:
    """W.9.10 2's disclosure: every gate's directional true d′ at zero injection (a = b = 0), i.e. what presentation
    drift alone gives — for the population pair (the calibration's) and for each pilot pair (its own drift on its own
    X / Y-averaged naive mean, the population's spillover and residuals). The level gates (reward level, punishment
    drop) are the ones drift can move; the association gates subtract N / RN."""
    pop = true_dprimes(theta, 0.0, 0.0, idx, z)
    pairs = []
    for pm, dp in zip(theta["pair_means"], theta["drift_pair"]):
        tj = dict(theta, m0s=np.repeat(np.asarray(pm, float).mean(-1, keepdims=True), 2, -1), drift=np.asarray(dp))
        pairs.append(dict(zip(WV.GATES, true_dprimes(tj, 0.0, 0.0, idx, z).tolist())))
    return dict(population=dict(zip(WV.GATES, pop.tolist())), pilot_pairs=pairs,
                level_gates=[WV.GATES[0], WV.GATES[1]])


# ================================================================ evaluation: the W verdict on simulated experiments
def evaluate(theta, rng, n_rep, a_pair, b_pair, fly_a, fly_b, g, z, spec) -> np.ndarray:
    """P(PASS) [len(q_grid), len(k_grid), F_max − F_min + 1, k_cap − k_min + 1]."""
    qs, ks = spec.q_grid, spec.k_grid
    fs = list(range(spec.f_min, spec.f_max + 1))
    kk = list(range(spec.k_min, spec.k_cap + 1))
    n_probe = 2 * max(ks)
    hits = np.zeros((len(qs), len(ks), len(fs), len(kk)))
    kw = dict(bar=spec.bar, band=spec.band_width, digits=spec.round_digits)
    done = 0
    while done < n_rep:
        n = min(spec.oc_chunk, n_rep - done)
        d = simulate(theta, rng, n, spec.k_cap, spec.f_max, n_probe, a_pair, b_pair, fly_a, fly_b, g, z,
                     spec.naive_max)
        mach = WV.rn1_mismatch(d)                                       # [n, pairs]
        for ki, K in enumerate(ks):
            dK = {s: v[..., :K, :, :] for s, v in d.items()}
            d2 = {s: v[..., :2 * K, :, :] for s, v in d.items()}
            cK, c2 = WV.fly_class(WV.gate_stats(dK, z), **kw), WV.fly_class(WV.gate_stats(d2, z), **kw)
            fK, f2 = WV.mech_fractions(dK), WV.mech_fractions(d2)
            for fi, F in enumerate(fs):
                mK = WV.mech_ok(fK[..., :F, :], spec.mech_min, spec.round_digits)
                m2 = WV.mech_ok(f2[..., :F, :], spec.mech_min, spec.round_digits)
                for qi, q in enumerate(qs):
                    fin = WV.pair_final(WV.pair_gate_code(cK[..., :F], q, F, spec.round_digits), mK,
                                        WV.pair_gate_code(c2[..., :F], q, F, spec.round_digits), m2)
                    for j, k in enumerate(kk):
                        ov = WV.overall_code(fin[:, :k], mach[:, :k].any(-1), spec.min_gate_pairs)
                        hits[qi, ki, fi, j] += (ov == WV.V_PASS).sum()
        done += n
    return hits / n_rep


def _uniform(spec, a, b):
    return [a] * spec.k_cap, [b] * spec.k_cap, [1.0] * spec.f_max, [1.0] * spec.f_max


def scenario_p(theta, cal, rng, n_rep, g, z, spec):
    if not cal["ok"]:
        return None
    return evaluate(theta, rng, n_rep, *_uniform(spec, cal["a"]["value"], cal["b"]["value"]), g, z, spec)


# ================================================================ the OC and the selection
def qualify(power_lo: np.ndarray, false_hi: np.ndarray, spec) -> np.ndarray:
    """[q, K, F] the envelope rule over F (module docstring)."""
    ok = (power_lo >= spec.p_power) & (false_hi <= spec.p_false)
    out = np.zeros_like(ok)
    nf = ok.shape[-1]
    for fi in range(nf):
        F = spec.f_min + fi
        hi = fi + 1 if F >= spec.envelope_solo_from else min(nf, fi + spec.envelope + 1)
        out[..., fi] = ok[..., fi:hi].all(-1)
    return out


def select(qual: np.ndarray, cost, spec) -> tuple:
    """(selected (q, K, F) or None, the qualifying designs in cost order with their costs)."""
    rows = []
    for qi, q in enumerate(spec.q_grid):
        for ki, K in enumerate(spec.k_grid):
            for fi in range(qual.shape[-1]):
                if qual[qi, ki, fi]:
                    F = spec.f_min + fi
                    rows.append(dict(q=q, K=K, F=F, cost_h=float(cost(K, F))))
    rows.sort(key=lambda r: (r["cost_h"], -r["q"], r["K"], r["F"]))
    return (rows[0] if rows else None), rows


def run(pilot: list, z: dict, spec, cost, n_boot=None, n_rep=None, n_boot_rep=None, log=None) -> dict:
    """The whole OC (module docstring) from the pilot pairs; cost(K, F) -> hours. Returns the table document."""
    t0 = time.perf_counter()
    n_boot = spec.boot_draws if n_boot is None else n_boot
    n_rep = spec.oc_reps if n_rep is None else n_rep
    n_boot_rep = spec.boot_reps if n_boot_rep is None else n_boot_rep
    theta = fit(pilot)
    idx = _rng(spec, TAG_CAL).integers(0, len(theta["resid"]), spec.cal_reps)
    shape = (len(spec.q_grid), len(spec.k_grid), spec.f_max - spec.f_min + 1, spec.k_cap - spec.k_min + 1)
    cal = {m: calibrate(theta, t, m, idx, z, spec) for m, t in (("min", spec.d_power), ("max", spec.d_false))}
    t_point = time.perf_counter()
    point = {}
    gs = list(spec.cluster_grid)
    for gi, g in enumerate(gs):
        for m in ("min", "max"):
            p = scenario_p(theta, cal[m], _rng(spec, TAG_POINT, gi, int(m == "max")), n_rep, g, z, spec)
            point[(g, m)] = p
    t_boot = time.perf_counter()
    boot_p = np.zeros((n_boot, len(gs), 2) + shape)
    boot_cal = []
    for bi in range(n_boot):
        rb = _rng(spec, TAG_BOOT, bi)
        tb = boot(theta, rb)
        ib = rb.integers(0, len(tb["resid"]), spec.cal_reps)
        cb = {m: calibrate(tb, t, m, ib, z, spec) for m, t in (("min", spec.d_power), ("max", spec.d_false))}
        boot_cal.append({m: dict(ok=c["ok"], a=(c["a"] or {}).get("value"), b=(c["b"] or {}).get("value"))
                         for m, c in cb.items()})
        for gi, g in enumerate(gs):
            for mi, m in enumerate(("min", "max")):
                p = scenario_p(tb, cb[m], _rng(spec, TAG_BOOT_SIM, bi, gi, mi), n_boot_rep, g, z, spec)
                boot_p[bi, gi, mi] = (0.0 if m == "min" else 1.0) if p is None else p
        if log is not None and (bi + 1) % 10 == 0:
            log(f"w oc bootstrap {bi + 1}/{n_boot} ({time.perf_counter() - t0:.0f} s)")
    t_sel = time.perf_counter()
    lo_pct, hi_pct = 100 * (1 - spec.boot_level), 100 * spec.boot_level
    if n_boot:
        power_lo = np.percentile(boot_p[:, :, 0].min(-1), lo_pct, axis=0).min(0)      # [q, K, F]
        false_hi = np.percentile(boot_p[:, :, 1].max(-1), hi_pct, axis=0).max(0)
    else:
        power_lo = np.zeros(shape[:3])
        false_hi = np.ones(shape[:3])
    reachable = cal["min"]["ok"] and cal["max"]["ok"]
    qual = qualify(power_lo, false_hi, spec) if reachable else np.zeros(shape[:3], bool)
    sel, ranking = select(qual, cost, spec)
    doc = dict(theta=summary(theta), calibration=cal, reachable=bool(reachable),
               drift_dprime=drift_dprimes(theta, idx, z), cal_floor_rule=spec.cal_floor_rule,
               point={f"g{g}|{m}": (None if p is None else p.tolist()) for (g, m), p in point.items()},
               power_lo=power_lo.tolist(), false_hi=false_hi.tolist(), qualified=qual.tolist(),
               selected=sel, ranking=ranking, boot_calibration=boot_cal,
               axes=dict(q=list(spec.q_grid), K=list(spec.k_grid), F=list(range(spec.f_min, spec.f_max + 1)),
                         k=list(range(spec.k_min, spec.k_cap + 1)), g=gs),
               n_boot=int(n_boot), n_rep=int(n_rep), n_boot_rep=int(n_boot_rep), seed=int(spec.oc_seed))
    t_rec = time.perf_counter()
    if sel is not None:
        doc["records"] = records(theta, sel, idx, z, spec, n_rep)
    t_end = time.perf_counter()
    doc["timing"] = dict(point_s=t_boot - t_point, boot_s=t_sel - t_boot, records_s=t_end - t_rec,
                         total_s=t_end - t0)
    return doc


def records(theta, sel, idx, z, spec, n_rep) -> dict:
    """W.6 / W.9.3 / W.9.9 P2-12 records at θ̂ for the selected design (q, K, F): P(PASS) by k at true d′ 0.5 · 1.0 ·
    1.5 · 2.0 (min-calibrated), the heterogeneous scenarios (one pair 0.5 + rest 1.5, every other pair 0.5, all 1.0)
    and the mixed-fly scenario (every other fly at 0) — each on every cluster level."""
    qi, ki = spec.q_grid.index(sel["q"]), spec.k_grid.index(sel["K"])
    fi = sel["F"] - spec.f_min
    cals = {t: calibrate(theta, t, "min", idx, z, spec) for t in sorted(set(spec.record_dprimes) | {1.0, 1.5})}
    out = dict(true_dprime={}, heterogeneous={}, mixed_flies={})

    def at(a_p, b_p, fa, fb, g, tag):
        r = _rng(spec, TAG_RECORD, spec.cluster_grid.index(g), RECORD_TAGS.index(tag))
        return evaluate(theta, r, n_rep, a_p, b_p, fa, fb, g, z, spec)[qi, ki, fi].tolist()

    def ab(t):
        c = cals[t]
        return (c["a"]["value"], c["b"]["value"]) if c["ok"] else (None, None)

    for g in spec.cluster_grid:
        out["true_dprime"][f"g{g}"] = {str(t): (at(*_uniform(spec, *ab(t)), g, f"t{t}") if cals[t]["ok"] else None)
                                       for t in spec.record_dprimes}
        a5, b5 = ab(0.5)
        a15, b15 = ab(1.5)
        a1, b1 = ab(1.0)
        het = {}
        if None not in (a5, a15):
            one = [a5] + [a15] * (spec.k_cap - 1), [b5] + [b15] * (spec.k_cap - 1)
            alt = [a5 if i % 2 else a15 for i in range(spec.k_cap)], [b5 if i % 2 else b15 for i in range(spec.k_cap)]
            het["one_pair_0.5"] = at(one[0], one[1], [1.0] * spec.f_max, [1.0] * spec.f_max, g, "one")
            het["half_pairs_0.5"] = at(alt[0], alt[1], [1.0] * spec.f_max, [1.0] * spec.f_max, g, "half")
        if a1 is not None:
            het["all_1.0"] = at(*_uniform(spec, a1, b1), g, "all1")
        out["heterogeneous"][f"g{g}"] = het
        if a15 is not None:
            half = [1.0 if i % 2 == 0 else 0.0 for i in range(spec.f_max)]
            out["mixed_flies"][f"g{g}"] = at([a15] * spec.k_cap, [b15] * spec.k_cap, half, half, g, "mixed")
    out["calibrations"] = {str(t): dict(ok=c["ok"], true_dprime=c.get("true_dprime")) for t, c in cals.items()}
    return out


# ================================================================ synthetic validation (W.9.3, W.9.9 P2-11)
def synthetic_pilot(rng, n_pair=16, n_fly=8, n_probe=8, base=(30.0, 60.0), sd=6.0, drift_ax=0.0, corr=0.0,
                    learn=(0.0, 0.0)) -> list:
    """Gaussian pilot pairs with known structure: naive means base (A, P) for X and Y, probe noise sd with an
    across-slot correlation corr, an X-only drift of MBON13 during the second phase (drift_ax per presentation
    block), and learning effects learn = (a, b) on R."""
    out = []
    cov = np.full((6, 6), corr) + (1 - corr) * np.eye(6)
    L = np.linalg.cholesky(cov)
    for _ in range(n_pair):
        mu = np.zeros((6, 2, 2))
        mu[:, WV.A, :], mu[:, WV.P, :] = base
        mu[SI["R1"], WV.P, WV.X] -= learn[0]
        mu[SI["R2"], WV.P, WV.X] -= learn[0]
        mu[SI["R2"], WV.A, WV.X] -= learn[1] + drift_ax
        mu[SI["RN2"], WV.P, WV.X] -= learn[0]
        mu[SI["RN2"], WV.A, WV.X] -= drift_ax
        mu[SI["N2"], WV.A, WV.X] -= drift_ax
        e = rng.standard_normal((n_fly, n_probe, 2, 2, 6)) @ L.T * sd
        c = np.clip(np.rint(mu[None, None] + np.moveaxis(e, -1, 2)), 0, None).astype(np.int64)
        d = to_stages(c)
        out.append({s: d[s] for s in WV.STAGES})
    return out


def simple_normal(spec, rng, d_true=1.5, n_rep=2000, q=0.75, k_probe=8) -> dict:
    """W.9.9 P0-1's simple normal model: per fly, four independent gates each estimated from K normal probes with true
    d′ d_true; P(PASS) of one pair at q by F (it must not rise from F 8 to 32)."""
    out = {}
    for F in (8, 16, 24, 32):
        x = rng.standard_normal((n_rep, F, 4, k_probe)) + d_true
        st = WV.dprime(x) * WV.SIGNS
        cls = WV.fly_class(st, spec.bar, spec.band_width, spec.round_digits)
        out[F] = float((WV.pair_gate_code(cls, q, F, spec.round_digits) == WV.P_PASS).mean())
    return out


def synthetic_validation(spec, z: dict, n_rep: int = 1000) -> dict:
    """The four known-answer fixtures and P0-1's check (W.9.9 P2-11 tolerances), evaluated by the OC machinery."""
    rng = _rng(spec, TAG_SYNTH)
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    th0 = fit(synthetic_pilot(rng, **kw))
    th_drift = fit(synthetic_pilot(rng, drift_ax=15.0, **kw))
    idx = rng.integers(0, len(th0["resid"]), spec.cal_reps)
    res = {}
    z0 = evaluate(th0, rng, n_rep, *_uniform(spec, 0.0, 0.0), 0.0, z, spec)
    res["zero_effect"] = dict(max_p=float(z0.max()), limit=0.02, ok=bool(z0.max() <= 0.02))
    big = calibrate(th0, 4.0, "min", idx, z, spec)
    pb = (evaluate(th0, rng, n_rep, *_uniform(spec, big["a"]["value"], big["b"]["value"]), 0.0, z, spec)
          if big["ok"] else np.zeros(1))
    res["big_effect"] = dict(min_p=float(pb.min()), limit=0.98, ok=bool(big["ok"] and pb.min() >= 0.98),
                             true_dprime=big.get("true_dprime"))
    one = evaluate(th_drift, rng, n_rep, *_uniform(spec, 0.0, 0.0), 0.0, z, spec)
    td = true_dprimes(th_drift, 0.0, 0.0, rng.integers(0, len(th_drift["resid"]), spec.cal_reps), z)
    res["one_gate"] = dict(max_p=float(one.max()), limit=0.02, ok=bool(one.max() <= 0.02 and td[1] >= 1.5),
                           true_dprime=dict(zip(WV.GATES, td.tolist())))
    if big["ok"]:
        alt = [1.0 if i % 2 == 0 else 0.0 for i in range(spec.f_max)]
        neg = evaluate(th0, rng, n_rep, [big["a"]["value"]] * spec.k_cap, [big["b"]["value"]] * spec.k_cap, alt,
                       [1.0 - x for x in alt], 0.0, z, spec)
        res["negative_correlation"] = dict(max_p=float(neg.max()), limit=0.02, ok=bool(neg.max() <= 0.02))
    else:
        res["negative_correlation"] = dict(max_p=None, ok=False)
    sn = simple_normal(spec, rng)
    res["simple_normal"] = dict(p_by_F={str(k): v for k, v in sn.items()}, ok=bool(sn[32] <= sn[8]))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res
