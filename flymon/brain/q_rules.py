"""Spec Q.4 as amended by Q.6 (Q.6.9 wins): the concordance rules for the five candidates as pure functions over Q0 / Q1
per-pair values (q_records.pair_values). Every label is a record — 일치 / 불일치 / 판단 불가 — never a verdict (Q.1).
Readings 7-13 of the plan fix the open choices: two-sided Wilcoxon (all-zero differences -> p = 1), Spearman None on
fewer than two points or a constant input (None meets no threshold), F/S always Q0's, the F/S-disagreement rule
downgrading only a 일치. Q.6.9: (c) is the LOWERED-s manipulation (s 0.7, fallback 0.8); ①'s 일치 needs
rho(base naive P_X, dr_P_c) >= +rho_min over pairs with naive P_X > 0 only, the P_X = 0 pairs recorded as "already at
floor". Part 1: helpers, F/S stability, noise floor, rules ⑤ ① ③ (② ④, sensitivity and assembly follow)."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import spearmanr, wilcoxon

MATCH, MISMATCH, UNDECIDED = "일치", "불일치", "판단 불가"


def med(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def auc_lower(f, s) -> float | None:
    """P(value_F < value_S), ties 1/2 (Q.6.3)."""
    f = [x for x in f if x is not None]
    s = [x for x in s if x is not None]
    if not f or not s:
        return None
    return sum(1 if a < b else 0.5 if a == b else 0 for a in f for b in s) / (len(f) * len(s))


def spearman(x, y) -> float | None:
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pts) < 2 or len({a for a, _ in pts}) < 2 or len({b for _, b in pts}) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr([a for a, _ in pts], [b for _, b in pts])[0])
    return None if math.isnan(rho) else rho


def wilcoxon_p(d) -> float:
    d = [float(x) for x in d if x is not None]
    if not any(x != 0 for x in d):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def fs_stability(F0, S0, base_r: dict, spec) -> dict:
    """Q.6.3: F/S re-split on the baseline's composite r, compared with Q0's (F/S stay Q0's for every label)."""
    Fb = sorted(k for k, r in base_r.items() if r < spec.split_r)
    Sb = sorted(k for k, r in base_r.items() if r >= spec.split_r)
    dis = len(set(F0) ^ set(Fb))
    n = len(F0) + len(S0)
    agree = 1 - dis / n
    return dict(disagree=dis, agree_rate=agree, unstable=bool(agree < spec.fs_agree_min),
                auc_blocked=bool(dis > spec.fs_disagree_max), F_base=Fb, S_base=Sb)


def noise_floor(V0: dict, B: dict) -> float:
    """Reading 13: median over all pairs of |r_P(Q0) - r_P(base)|."""
    return med([abs(V0[k]["r_P"] - B[k]["r_P"]) for k in sorted(V0) if k in B])


def _fs_block(lab, why, stab, spec):
    if lab == MATCH and stab["auc_blocked"]:
        return UNDECIDED, f"F/S 불일치 {stab['disagree']}쌍 > {spec.fs_disagree_max} — 일치를 판단 불가로 ({why})"
    return lab, why


# ================================================================ ⑤ 변동 (Q.6.1)
def _variation_one(V, F, S, spec) -> dict:
    mf = med([abs(V[k]["dV_mean"]) for k in F])
    ms = med([abs(V[k]["dV_mean"]) for k in S])
    sf = med([V[k]["dV_sd"] for k in F])
    ss = med([V[k]["dV_sd"] for k in S])
    if None in (mf, ms, sf, ss):
        lab = UNDECIDED
    elif mf >= spec.var_mean_match * ms and sf >= spec.var_sd_match * ss:
        lab = MATCH
    elif mf < spec.var_mean_mismatch * ms:
        lab = MISMATCH
    else:
        lab = UNDECIDED
    return dict(label=lab, F_abs_mean=mf, S_abs_mean=ms, F_sd=sf, S_sd=ss)


def rule_variation(V0, B, F, S, stab, spec) -> dict:
    q0, b = _variation_one(V0, F, S, spec), _variation_one(B, F, S, spec)
    if q0["label"] == b["label"] == MATCH:
        lab, why = MATCH, "Q0·기준선 모두 일치"
    elif q0["label"] == b["label"] == MISMATCH:
        lab, why = MISMATCH, "Q0·기준선 모두 불일치"
    else:
        lab, why = UNDECIDED, f"Q0 {q0['label']} / 기준선 {b['label']}"
    lab, why = _fs_block(lab, why, stab, spec)
    return dict(label=lab, why=why, q0=q0, base=b)


# ================================================================ ① 바닥 (Q.4, Q.6.1, Q.6.4, Q.6.9)
def rule_floor(V0, B, F, S, stab, C, s_c, spec) -> dict:
    """Q.6.9 ordering: (c) weak -> 판단 불가 (fixed); Q0 naive-P_X AUC <= auc_mismatch -> 불일치 (Q0 only, survives an
    INVALID (c)); (c) INVALID (C None) -> 판단 불가; AUC >= auc_match and rho(base naive P_X, dr_P_c) >= +rho_min over
    the pairs with base naive P_X > 0 -> 일치; else 판단 불가. (c) is the lowered-s manipulation (s_c block's s)."""
    a0, a1 = (str(float(a)) for a in spec.fixed_alphas)
    auc = auc_lower([V0[k]["naive_px"] for k in F], [V0[k]["naive_px"] for k in S])
    sat = med([(abs(B[k]["dpx_fixed"][a1]) - abs(B[k]["dpx_fixed"][a0])) / abs(B[k]["dpx_fixed"][a0])
               for k in F if B[k].get("dpx_fixed") and B[k]["dpx_fixed"][a0] != 0])
    keys = sorted(B)
    at_floor = sorted(k for k in keys if B[k]["naive_px"] == 0)
    rho_keys = [k for k in keys if B[k]["naive_px"] > 0] if spec.rho_positive_only else keys
    delta = rho = None
    if C is not None:
        delta = {k: C[k]["r_P"] - B[k]["r_P"] for k in keys}
        rho = spearman([B[k]["naive_px"] for k in rho_keys], [delta[k] for k in rho_keys])
    s = s_c["s"]
    weak = bool(s_c["weak"])                    # |s - 1.0| < s_weak, decided where the s_c block is built
    rec = dict(auc=auc, rho=rho, rho_n=len(rho_keys), delta_r_P=delta, saturation_F=sat, s=s, weak=weak,
               already_at_floor=len(at_floor), already_at_floor_keys=at_floor)
    cname = f"(c) s 하향 조작 (s {s})"
    if weak:
        return dict(rec, label=UNDECIDED, why=f"{cname} {spec.weak_note} — ①은 판단 불가로 고정")
    if auc is not None and auc <= spec.auc_mismatch:
        lab, why = MISMATCH, f"Q0 순진 P_X AUC {auc:.3f} ≤ {spec.auc_mismatch}"
    elif C is None:
        lab, why = UNDECIDED, f"{cname} INVALID — 결과를 쓰지 않음"
    elif auc is not None and auc >= spec.auc_match and rho is not None and rho >= spec.rho_min:
        lab, why = MATCH, (f"AUC {auc:.3f} ≥ {spec.auc_match}, {cname}의 ρ(순진 P_X, Δr_P) {rho:.3f} ≥ +{spec.rho_min}"
                           f" (P_X > 0인 {len(rho_keys)}쌍; 이미 바닥 {len(at_floor)}쌍)")
    else:
        lab, why = UNDECIDED, f"AUC {auc}, {cname}의 ρ {rho} (P_X > 0인 {len(rho_keys)}쌍)"
    lab, why = _fs_block(lab, why, stab, spec)
    return dict(rec, label=lab, why=why)


# ================================================================ ③ 편집 도달 (Q.6.2)
def rule_reach(B, F, S, stab, spec) -> dict:
    """Reading 6/8: everything (ratio, W_X, f_X, fixed-arm P_X) comes from the BASELINE table B only."""
    a0 = str(float(spec.fixed_alphas[0]))
    keys = sorted(B)
    rf = [B[k]["ratio"] for k in F]
    rs = [B[k]["ratio"] for k in S]
    auc = auc_lower(rf, rs)
    rho = spearman([B[k]["W_X"] for k in keys], [abs(B[k]["dpx_fixed"][a0]) for k in keys])
    px = med([B[k]["px_after_fixed"][a0] for k in F])
    rec = dict(auc=auc, rho=rho, px_after_F_median=px, n_ratio_undefined=sum(x is None for x in rf + rs),
               W_X={k: B[k]["W_X"] for k in keys}, f_X={k: B[k].get("f_X") for k in keys})
    if auc is None:
        lab, why = UNDECIDED, "비율이 정의되지 않음"
    elif auc <= spec.auc_mismatch:
        lab, why = MISMATCH, f"비율 AUC {auc:.3f} ≤ {spec.auc_mismatch}"
    elif auc >= spec.auc_match and rho is not None and rho >= spec.rho_min:
        if px is not None and px >= spec.stop_p:
            lab, why = MATCH, f"AUC {auc:.3f}, ρ(W_X) {rho:.3f}, F의 α {a0} 뒤 P_X 중앙값 {px:.2f} ≥ {spec.stop_p}"
        else:
            lab, why = UNDECIDED, f"F의 α {a0} 뒤 P_X 중앙값 {px} < {spec.stop_p} — ①과 겹침"
    else:
        lab, why = UNDECIDED, f"AUC {auc}, ρ {rho}"
    lab, why = _fs_block(lab, why, stab, spec)
    return dict(rec, label=lab, why=why)
