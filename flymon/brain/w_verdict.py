"""W's judgement code (W.4 as amended by W.9.1, W.9.2, W.9.8 H1 / H2 / H5 / H7, W.9.9 P1-2 / P2-9 / P2-10): pure
functions over raw counts, vectorised over leading axes, so that w_oc runs THIS code on its simulated experiments
(W.9.8 H4) and the judge runs it on the measured ones. The frozen F v3 verdict.py is not imported (W.9.6 P1-6); G.7's
defects are fixed here: a NaN fly statistic or fewer than 2 probes is INVALID (never a pass), the fly denominator is
the scheduled F, a missing or mis-shaped stage is INVALID, BAND ends in a coded re-measure (K -> 2K, then BAND is
FAIL), and the aggregation is per-fly joint satisfaction.

Data layout: a pair's counts are {stage: array [..., F, K, 2, 2]}, stage in STAGES, axis -2 the readout cell (0 = A =
MBON13, 1 = P = MBON05, both cells of a type summed), axis -1 the odour (0 = X, the taught odour; 1 = Y).
- ΔV = V(X) − V(Y), V = z_A − z_P (z = z_V for L_V, block h4's z for C).
- Fly gates (d′ over the fly's K probes, F.5's limits: sd 0 → 0 if the mean is 0, else ±∞): reward level d′(ΔV_R1)
  ≥ +1, punishment drop d′(ΔV_R2 − ΔV_R1) ≤ −1, reward association d′(ΔV_R1 − ΔV_N1) ≥ +1, punishment association
  d′((ΔV_R2 − ΔV_R1) − (ΔV_RN2 − ΔV_RN1)) ≤ −1. Margin = round(sign·d′ − 1, 9): ±∞ in the gate's direction passes,
  against it fails (H5).
- A fly is satisfied (every margin ≥ 0), BAND (every margin ≥ −0.2, one < 0) or failed; any NaN → the fly is invalid.
- Pair (W.9.1 with the design's q): q_sat = n_sat / F ≥ q → PASS; else q_sat + n_band / F ≥ q → BAND; else FAIL;
  an invalid fly → INVALID. BAND is resolved on the same flies with 2K probes (k 0..K−1 kept, K..2K−1 added):
  PASS → PASS, otherwise FAIL.
- Mechanism control (W.9.2 / H2), same flies and probes (after 2K for a BAND pair, P2-9): per fly the share of
  probes with MBON05(X) at R1 below N1, and with MBON13(X)'s change R2 − R1 below RN2 − RN1 (ties ½); both fly
  medians ≥ 0.75 or the pair is FAIL ("기계 대조 실패").
- Overall (H7): RN1 ≠ R1 bit for bit, or an INVALID pair, or another machine reason → STOP_MACHINE; a FAIL among the
  judgeable pairs → FAIL; a BAND left → UNDECIDED; fewer than 4 judgeable pairs → UNDECIDED; all PASS → PASS.
- Naive (W.9.5, W.9.9 P2-10): the pooled d′ of ΔV_pre over every fly's screening probes (F × K), fixed at the screen."""
from __future__ import annotations

import numpy as np

STAGES = ("pre", "R1", "R2", "N1", "N2", "RN1", "RN2")
GATES = ("reward_level", "punish_drop", "reward_assoc", "punish_assoc")
SIGNS = np.array([1.0, -1.0, 1.0, -1.0])
FLY_INVALID, FLY_FAIL, FLY_BAND, FLY_SAT = -1, 0, 1, 2
P_INVALID, P_FAIL, P_BAND, P_PASS = -1, 0, 1, 2
PAIR_LABEL = {P_INVALID: "INVALID", P_FAIL: "FAIL", P_BAND: "BAND", P_PASS: "PASS"}
V_STOP_MACHINE, V_FAIL, V_UNDECIDED, V_PASS = -1, 0, 1, 2
VERDICT_LABEL = {V_STOP_MACHINE: "STOP_MACHINE", V_FAIL: "FAIL", V_UNDECIDED: "UNDECIDED", V_PASS: "PASS"}
A, P, X, Y = 0, 1, 0, 1


def dprime(x, axis: int = -1):
    """F.5's d′ along `axis`: mean / sd (ddof 1); sd 0 → 0.0 when the mean is 0, else ±∞; fewer than 2 values or a
    NaN → NaN."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    if n < 2:
        return np.full(np.delete(np.array(x.shape), axis % x.ndim), np.nan) if x.ndim > 1 else np.float64(np.nan)
    m = x.mean(axis)
    sd = x.std(axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = m / sd
    lim = np.where(m == 0, 0.0, np.copysign(np.inf, m))
    return np.where(sd == 0, lim, d)


def dv(arr, z: dict):
    """ΔV per probe: arr [..., 2 cells, 2 odours] -> [...]."""
    a = np.asarray(arr, float)
    v = (a[..., A, :] - z["A"][0]) / z["A"][1] - (a[..., P, :] - z["P"][0]) / z["P"][1]
    return v[..., X] - v[..., Y]


def gate_stats(d: dict, z: dict):
    """[..., F, 4] fly d′ of the four gates (GATES order) over the probe axis."""
    r1, r2, n1 = dv(d["R1"], z), dv(d["R2"], z), dv(d["N1"], z)
    rn1, rn2 = dv(d["RN1"], z), dv(d["RN2"], z)
    return np.stack([dprime(r1), dprime(r2 - r1), dprime(r1 - n1), dprime((r2 - r1) - (rn2 - rn1))], axis=-1)


def margins(stats, bar: float = 1.0, digits: int = 9):
    with np.errstate(invalid="ignore"):
        return np.round(SIGNS * np.asarray(stats, float) - bar, digits)


def fly_class(stats, bar: float = 1.0, band: float = 0.2, digits: int = 9):
    """[..., F] FLY_SAT / FLY_BAND / FLY_FAIL / FLY_INVALID."""
    m = margins(stats, bar, digits)
    bad = np.isnan(m).any(-1)
    with np.errstate(invalid="ignore"):
        sat = (m >= 0).all(-1)
        fail = (m < -band).any(-1)
    return np.where(bad, FLY_INVALID, np.where(sat, FLY_SAT, np.where(fail, FLY_FAIL, FLY_BAND)))


def lower_fraction(a, b):
    """Share of probes (last axis) with a < b; ties count ½."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ((a < b).sum(-1) + 0.5 * (a == b).sum(-1)) / a.shape[-1]


def mech_fractions(d: dict):
    """[..., F, 2]: reward MBON05(X) R1 < N1; punishment MBON13(X) change R2 − R1 < RN2 − RN1."""
    px = lambda s: np.asarray(d[s], float)[..., P, X]      # noqa: E731
    ax = lambda s: np.asarray(d[s], float)[..., A, X]      # noqa: E731
    return np.stack([lower_fraction(px("R1"), px("N1")),
                     lower_fraction(ax("R2") - ax("R1"), ax("RN2") - ax("RN1"))], axis=-1)


def mech_ok(fr, mech_min: float = 0.75, digits: int = 9):
    """[...] both fly medians (axis -2) ≥ mech_min."""
    med = np.median(np.asarray(fr, float), axis=-2)
    return (np.round(med - mech_min, digits) >= 0).all(-1)


def pair_gate_code(cls, q: float, f_sched: int, digits: int = 9):
    """[...] P_PASS / P_BAND / P_FAIL / P_INVALID from fly classes [..., F] with the scheduled F as denominator."""
    cls = np.asarray(cls)
    n_sat, n_band = (cls == FLY_SAT).sum(-1), (cls == FLY_BAND).sum(-1)
    bad = (cls == FLY_INVALID).any(-1) | (cls.shape[-1] != f_sched)
    ok = np.round(n_sat / f_sched - q, digits) >= 0
    band = np.round((n_sat + n_band) / f_sched - q, digits) >= 0
    return np.where(bad, P_INVALID, np.where(ok, P_PASS, np.where(band, P_BAND, P_FAIL)))


def pair_final(code_k, mech_k, code_2k, mech_2k):
    """[...]: BAND → the 2K code (PASS stays only with the 2K mechanism control; BAND or FAIL → FAIL); PASS → PASS iff
    the K mechanism control passes; FAIL stays; INVALID anywhere → INVALID."""
    code_k, code_2k = np.asarray(code_k), np.asarray(code_2k)
    band_res = np.where(code_2k == P_INVALID, P_INVALID, np.where((code_2k == P_PASS) & mech_2k, P_PASS, P_FAIL))
    return np.where(code_k == P_INVALID, P_INVALID,
                    np.where(code_k == P_BAND, band_res,
                             np.where(code_k == P_PASS, np.where(mech_k, P_PASS, P_FAIL), P_FAIL)))


def rn1_mismatch(d: dict):
    """[...] over pairs: RN1 ≠ R1 anywhere (G.5's machine check, W.9.9 P1-2)."""
    r1, rn1 = np.asarray(d["R1"]), np.asarray(d["RN1"])
    return (r1 != rn1).reshape(r1.shape[:-4] + (-1,)).any(-1)


def overall_code(final, machine, min_pairs: int = 4):
    """[...]: H7's order over the pair axis (-1): machine or INVALID → STOP_MACHINE; FAIL → FAIL; BAND → UNDECIDED;
    judgeable < min_pairs → UNDECIDED; all PASS → PASS."""
    final = np.asarray(final)
    stop = np.asarray(machine) | (final == P_INVALID).any(-1)
    n = final.shape[-1]
    return np.where(stop, V_STOP_MACHINE, np.where((final == P_FAIL).any(-1), V_FAIL,
                    np.where((final == P_BAND).any(-1), V_UNDECIDED,
                             np.where(n < min_pairs, V_UNDECIDED, V_PASS))))


def naive_dprime(pre, z: dict) -> float:
    """The pooled naive d′ over every probe of every fly (pre [F, K, 2, 2])."""
    return float(dprime(dv(pre, z).reshape(-1)))


# ================================================================ the real-data judge (one pair, then the set)
def data_reasons(d: dict, f_sched: int, k: int) -> list:
    """G.7's missing-data refusal: every stage present, shape (F, K, 2, 2), non-negative integer counts."""
    out = []
    for s in STAGES:
        if s not in d:
            out.append(f"{s} 없음")
            continue
        a = np.asarray(d[s])
        if a.shape != (f_sched, k, 2, 2):
            out.append(f"{s} 모양 {a.shape} ≠ ({f_sched}, {k}, 2, 2)")
        elif not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
            out.append(f"{s} 카운트가 음이 아닌 정수가 아님")
    return out


def judge_pair(d_k: dict, d_2k: dict | None, z: dict, q: float, f_sched: int, k: int, spec) -> dict:
    """One gate pair: d_k = the K-probe counts, d_2k = the same flies with 2K probes (needed only when the K read is
    BAND). Records every fly statistic."""
    bad = data_reasons(d_k, f_sched, k)
    if bad:
        return dict(status="INVALID", reasons=bad)
    kw = dict(bar=spec.bar, band=spec.band_width, digits=spec.round_digits)
    st = gate_stats(d_k, z)
    cls = fly_class(st, **kw)
    code = int(pair_gate_code(cls, q, f_sched, spec.round_digits))
    fr = mech_fractions(d_k)
    mk = bool(mech_ok(fr, spec.mech_min, spec.round_digits))
    out = dict(stats=st.tolist(), classes=cls.tolist(), code_k=PAIR_LABEL[code], mech_k=fr.tolist(),
               mech_k_ok=mk, n_sat=int((cls == FLY_SAT).sum()), n_band=int((cls == FLY_BAND).sum()),
               q_sat=float((cls == FLY_SAT).sum() / f_sched), K=k)
    c2, m2 = P_INVALID, False
    if code == P_BAND:
        if d_2k is None:
            return dict(out, status="BAND", reasons=["2K 재측정 자료 없음"])
        bad2 = data_reasons(d_2k, f_sched, 2 * k)
        if bad2:
            return dict(out, status="INVALID", reasons=bad2)
        if any(not np.array_equal(np.asarray(d_2k[s])[:, :k], np.asarray(d_k[s])) for s in STAGES):
            return dict(out, status="INVALID", reasons=["2K 자료의 앞 K 프로브가 K 자료와 다름"])
        st2 = gate_stats(d_2k, z)
        cls2 = fly_class(st2, **kw)
        c2 = int(pair_gate_code(cls2, q, f_sched, spec.round_digits))
        fr2 = mech_fractions(d_2k)
        m2 = bool(mech_ok(fr2, spec.mech_min, spec.round_digits))
        n2 = int((cls2 == FLY_SAT).sum())
        out.update(stats_2k=st2.tolist(), classes_2k=cls2.tolist(), code_2k=PAIR_LABEL[c2], mech_2k=fr2.tolist(),
                   mech_2k_ok=m2, n_sat_2k=n2, q_sat_2k=float(n2 / f_sched))
    fin = int(pair_final(code, mk, c2, m2))
    reasons = []
    if fin == P_FAIL:
        used_cls, used_m = (cls2, m2) if code == P_BAND else (cls, mk)
        gate_code = c2 if code == P_BAND else code
        if gate_code != P_PASS:
            reasons.append("마리별 동시 충족 미달")
        if not used_m:
            reasons.append("기계 대조 실패")
        out["failing_gates"] = _failing_gates(st2 if code == P_BAND else st, used_cls, spec)
    return dict(out, status=PAIR_LABEL[fin], reasons=reasons, mech_fail=bool("기계 대조 실패" in reasons))


def _failing_gates(st, cls, spec) -> dict:
    m = margins(st, spec.bar, spec.round_digits)
    out = {}
    for j, g in enumerate(GATES):
        with np.errstate(invalid="ignore"):
            out[g] = int((m[:, j] < 0).sum())
    return out


def judge(pairs: dict, machine: list, z: dict, q: float, f_sched: int, k: int, spec) -> dict:
    """pairs = {pair key: (d_k, d_2k or None)} in declared order; machine = the runner's machine reasons (pre equal
    across brains and with the screen, band-job weights equal the main job's, …). H7's order."""
    res = {key: judge_pair(dk, d2, z, q, f_sched, k, spec) for key, (dk, d2) in pairs.items()}
    mm = [f"{key}: RN1 ≠ R1" for key, (dk, _) in pairs.items()
          if not data_reasons(dk, f_sched, k) and bool(rn1_mismatch({s: np.asarray(dk[s])[None] for s in STAGES})[0])]
    codes = np.array([{v: c for c, v in PAIR_LABEL.items()}[r["status"]] for r in res.values()])
    mach = list(machine) + mm
    code = int(overall_code(codes, bool(mach), spec.min_gate_pairs)) if len(codes) else V_UNDECIDED
    fails = [key for key, r in res.items() if r["status"] == "FAIL"]
    why = ""
    if code == V_UNDECIDED:
        why = "BAND 잔존" if any(r["status"] == "BAND" for r in res.values()) else \
            f"판정 가능 쌍 < {spec.min_gate_pairs}"
    return dict(verdict=VERDICT_LABEL[code], machine=mach, pairs=res, failing=fails,
                mech_fail=[key for key in fails if res[key].get("mech_fail")], undecided_cause=why,
                invalid=[key for key, r in res.items() if r["status"] == "INVALID"], n_pairs=len(res),
                n_judgeable=int(sum(r["status"] in ("PASS", "FAIL", "BAND") for r in res.values())))
