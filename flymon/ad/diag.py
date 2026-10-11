"""AD.11 record-only diagnosis helpers (no verdict, no rule change; every number carries AC.5's LABEL and is a
"기록 전용 추정"). Pure functions over AD's existing records (situation records, turn logs, checkpoint weights) and
over offline re-evaluations of EXISTING weights; plus edge_map_job, a module-level pool job (run_jobs) that returns
the plastic KC->MBON edge map of the L_V worker. AC.11's helpers (flymon/ac/diag.py) are reused, not copied:
best_contrast is the right-direction quantity (d > 0 = right direction), mult_class the effectiveness class."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..ac import diag as acd

CLASSES = ("super", "neutral", "not_very", "immune")


# ---------------------------------------------------------------- right direction, per pair and over points
def right_flags(rec, pairs) -> list:
    """Per pair 1 if AC.11's best_contrast d > 0 (the right direction), else 0."""
    return [int(d > 0) for d in acd.best_contrast(rec, pairs)]


def contrast_signs(ds) -> dict:
    """Share of contrasts above / below / at zero, mean, median (the symmetry check of FLY-OS's contrasts)."""
    d = np.asarray(list(ds), float)
    if d.size == 0:
        return dict(n=0)
    return dict(n=int(d.size), pos=float(np.mean(d > 0)), neg=float(np.mean(d < 0)), zero=float(np.mean(d == 0)),
                mean=float(d.mean()), median=float(np.median(d)))


def transitions(flags_by_point: dict) -> dict:
    """flags_by_point: {point: [0/1 per unit]} (same units at every point). Per consecutive pair of points the count
    of right->wrong and wrong->right changes and the units unchanged; totals over all steps."""
    pts = sorted(flags_by_point)
    steps, tot = {}, dict(r2w=0, w2r=0, same=0)
    for a, b in zip(pts, pts[1:]):
        x, y = np.asarray(flags_by_point[a], int), np.asarray(flags_by_point[b], int)
        if x.shape != y.shape:
            raise ValueError("the same units are needed at every point")
        s = dict(r2w=int(np.sum((x == 1) & (y == 0))), w2r=int(np.sum((x == 0) & (y == 1))), same=int(np.sum(x == y)))
        steps[f"{a}-{b}"] = s
        for k in tot:
            tot[k] += s[k]
    return dict(steps=steps, total=tot)


def majority_disagreement(flag_rows) -> float:
    """flag_rows: fly x pair 0/1 table of flies sharing weights and odours (only the decision seed differs). Mean over
    pairs of the share of flies whose flag differs from the pair's majority: the seed-noise flip level."""
    a = np.asarray(flag_rows, float)
    maj = (a.mean(0) >= 0.5).astype(float)
    return float(np.mean(a != maj))


# ---------------------------------------------------------------- readout variants (records only)
def readout_v(a, p, z: dict, kind: str) -> list:
    """V of one situation's candidates from the recorded MBON counts: "zv" = z_A - z_P (the run's V), "a_only" = z_A,
    "p_only" = -z_P, "raw" = a - p (counts, no z)."""
    a, p = np.asarray(a, float), np.asarray(p, float)
    za, zp = (a - z["A"][0]) / z["A"][1], (p - z["P"][0]) / z["P"][1]
    out = {"zv": za - zp, "a_only": za, "p_only": -zp, "raw": a - p}
    if kind not in out:
        raise ValueError(f"unknown readout variant {kind!r}")
    return [float(x) for x in out[kind]]


def with_v(rec, fn) -> dict:
    """A copy of a situation record whose every situation's v is fn(situation)."""
    return dict(rec, situations=[dict(s, v=list(fn(s))) for s in rec["situations"]])


def readout_variant(rec, z: dict, kind: str) -> dict:
    return with_v(rec, lambda s: readout_v(s["a"], s["p"], z, kind))


def delta_record(rec_t, rec_0) -> dict:
    """v(t) - v(0) per candidate, same fly / odours / decision seeds (AC.5: no point in the seed): the learned change
    with the naive w0 preference removed."""
    s0 = acd.sides(rec_0)
    return with_v(rec_t, lambda s: np.asarray(s["v"], float) - np.asarray(s0[(s["pair"], s["side"])]["v"], float))


def class_of(pair: dict, side: int, cand: int) -> str:
    return acd.mult_class((pair["mults1"] if side == 0 else pair["mults2"])[cand])


def class_means(recs, pairs) -> dict:
    """Mean centred V per effectiveness class over the given records (AC.11's centred V)."""
    acc = defaultdict(list)
    for r in recs:
        for s in r["situations"]:
            for c, v in enumerate(acd.centred(s["v"])):
                acc[class_of(pairs[s["pair"]], s["side"], c)].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def class_model_contrast(pairs, bias: dict) -> list:
    """Per pair the contrast d an additive class-only V (V = bias[class]) would give: what the class bias alone
    predicts for the right direction (missing classes count 0)."""
    out = []
    for p in pairs:
        b1, b2 = p["cands"].index(p["best1"]), p["cands"].index(p["best2"])
        v = lambda side, c: bias.get(class_of(p, side, c), 0.0)   # noqa: E731
        out.append((v(0, b1) - v(0, b2)) - (v(1, b1) - v(1, b2)))
    return out


def per_class_contrast(rec, pairs) -> dict:
    """The right-direction flags split by the effectiveness classes of (best1 at side 0, best2 at side 1)."""
    out = defaultdict(list)
    for p, d in zip(pairs, acd.best_contrast(rec, pairs)):
        b1, b2 = p["cands"].index(p["best1"]), p["cands"].index(p["best2"])
        out[f"{class_of(p, 0, b1)}|{class_of(p, 1, b2)}"].append(int(d > 0))
    return out


# ---------------------------------------------------------------- weights
def amplify(w, w0, k: float, floor: float = 0.2) -> np.ndarray:
    """w0 + k (w - w0), held at the plasticity floor floor * w0 and at >= 0: the learned change scaled by k."""
    w, w0 = np.asarray(w, np.float32), np.asarray(w0, np.float32)
    out = w0 + np.float32(k) * (w - w0)
    out = np.maximum(out, w0 * np.float32(floor))
    return np.maximum(out, 0).astype(np.float32)


def keep_only(w, w0, mask) -> np.ndarray:
    """The learned change kept where mask, w0 elsewhere."""
    w, w0, m = np.asarray(w, np.float32), np.asarray(w0, np.float32), np.asarray(mask, bool)
    return np.where(m, w, w0).astype(np.float32)


def floor_tol(floor: float = 0.2, r: float = 0.001) -> float:
    """The floor test's ratio bound. A checkpoint is written after recover_pulse, so an edge held at the floor reads
    floor + r (1 - floor) (0.2008 for r 0.001); the bound is that value, with a relative 1e-6 margin."""
    return (float(floor) + float(r) * (1.0 - float(floor))) * (1 + 1e-6)


def ratio_summary(w, w0, mask=None, floor: float = 0.2, r: float = 0.001) -> dict:
    """w / w0 over the selected edges: n, the floor count and share (<= floor_tol(floor, r): at the floor, recovered
    by one pulse), the unchanged share (== w0, the ceiling: the rule only depresses and recovers toward w0), the share
    above w0, and quantiles (min, q01, ...)."""
    w, w0 = np.asarray(w, float), np.asarray(w0, float)
    m = np.ones(w.shape, bool) if mask is None else np.asarray(mask, bool)
    x = w[m] / w0[m]
    if x.size == 0:
        return dict(n=0)
    q = np.percentile(x, [1, 10, 50, 90])
    at_floor = x <= floor_tol(floor, r)
    return dict(n=int(x.size), floor_n=int(at_floor.sum()), floor=float(np.mean(at_floor)),
                at_w0=float(np.mean(x == 1.0)), above_w0=float(np.mean(x > 1.0)), mean=float(x.mean()),
                min=float(x.min()), q01=float(q[0]), q10=float(q[1]), q50=float(q[2]), q90=float(q[3]),
                mean_deficit=float(np.mean(1.0 - x)))


def fit_tau(t, w, w_eq: float, w_start: float = 1.0, taus=None) -> dict:
    """Least-squares tau of the approach fraction (w_start - w) / (w_start - w_eq) = 1 - exp(-t / tau) over bin
    centres t (grid search, 0.05-battle steps over 0.5-200). Returns tau, the fitted and observed fractions, the SSE."""
    t, w = np.asarray(t, float), np.asarray(w, float)
    frac = (w_start - w) / (w_start - float(w_eq))
    grid = np.arange(0.5, 200.0, 0.05) if taus is None else np.asarray(taus, float)
    sse = np.array([np.sum((frac - (1 - np.exp(-t / g))) ** 2) for g in grid])
    k = int(np.argmin(sse))
    return dict(tau=float(grid[k]), sse=float(sse[k]), frac=[float(x) for x in frac],
                fitted=[float(x) for x in 1 - np.exp(-t / grid[k])], t=[float(x) for x in t], w_eq=float(w_eq))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p for k of n under p = 1/2 (doubled smaller tail, capped at 1)."""
    k = min(int(k), int(n) - int(k))
    tail = sum(math.comb(int(n), i) for i in range(k + 1)) / 2 ** int(n)
    return float(min(1.0, 2 * tail))


# ---------------------------------------------------------------- recovery
def retention(r: float, n) -> float:
    """Share of a depression left after n recovery pulses of rate r: (1 - r) ** n."""
    return float((1.0 - float(r)) ** float(n))


def no_recovery_factor(r: float, n: float) -> float:
    """Linear-regime ratio of the deficit after n pulses without recovery to the deficit with recovery r, for a
    constant depression per pulse: r n / (1 - exp(-r n)) (with recovery the deficit is (d / r)(1 - exp(-r n)))."""
    x = float(r) * float(n)
    return float(x / (1.0 - math.exp(-x))) if x > 0 else 1.0


def use_gaps(keys) -> list:
    """keys: one odour key per pulse, in order. For each pulse whose key appeared before, the number of pulses since
    that key's previous pulse (recovery steps between two uses of the same odour)."""
    last, out = {}, []
    for i, k in enumerate(keys):
        if k in last:
            out.append(i - last[k])
        last[k] = i
    return out


# ---------------------------------------------------------------- pulses by effectiveness class
def pulse_rows(turns, reward: str = "PAM08", punish: str = "PPL105") -> list:
    """turns: acd.join_turns output. One row per reinforced turn: the battle index, the outcome's effectiveness
    (the true opponent's), reward / punishment ms and counts."""
    from .inputs import battle_index
    rows = []
    for t in turns:
        if "reinforce" not in t:
            continue
        oc = (t.get("outcome") or {}).get("outcome") or {}
        pulses = t["reinforce"].get("pulses") or []
        r, p = acd.pulse_ms(pulses, reward, punish)
        rows.append(dict(b=battle_index(t["battle_id"]), eff=oc.get("effectiveness") or "none",
                         reward_ms=r, punish_ms=p, n_reward=sum(d == reward for d, _ in pulses),
                         n_punish=sum(d == punish for d, _ in pulses)))
    return rows


def pulse_class_table(rows, points) -> dict:
    """Per interval between points and effectiveness: turns, reward / punishment ms and counts, the reward share of
    the pulse ms (reward / (reward + punishment))."""
    pts = sorted(int(p) for p in points)
    out = {}
    for a, b in zip(pts, pts[1:]):
        by = defaultdict(list)
        for x in rows:
            if a <= x["b"] < b:
                by[x["eff"]].append(x)
        cell = {}
        for eff, xs in sorted(by.items()):
            rm, pm = sum(x["reward_ms"] for x in xs), sum(x["punish_ms"] for x in xs)
            cell[eff] = dict(turns=len(xs), reward_ms=rm, punish_ms=pm, n_reward=sum(x["n_reward"] for x in xs),
                             n_punish=sum(x["n_punish"] for x in xs),
                             reward_share=(rm / (rm + pm) if rm + pm > 0 else None),
                             net_ms_per_turn=(rm - pm) / len(xs))
        out[f"{a}-{b}"] = cell
    return out


# ---------------------------------------------------------------- KC patterns
def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def spearman_safe(x, y):
    return acd.spearman(list(x), list(y))


# ---------------------------------------------------------------- pool job (module level: spawn workers import it)
def edge_map_job(eng, pl, pops, comps, ro, types) -> dict:
    """The plastic KC->MBON edges of the worker's L_V engine: local KC index of each edge, target neuron id, and per DAN
    type the core-compartment mask (agent/jobs.edge_compartments_job's definition)."""
    tgt = eng.csc.tgt[pl.edges]
    return dict(pre_kc=np.asarray(pl.pre_kc), tgt=np.asarray(tgt), masks={t: np.isin(tgt, comps[t].core) for t in types})
