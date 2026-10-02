"""Spec Q.4 as amended by Q.6 (Q.6.9 wins): the concordance rules for the five candidates as pure functions over Q0 / Q1
per-pair values (q_records.pair_values). Every label is a record — 일치 / 불일치 / 판단 불가 — never a verdict (Q.1).
Readings 7-13 of the plan fix the open choices: two-sided Wilcoxon (all-zero differences -> p = 1), Spearman None on
fewer than two points or a constant input (None meets no threshold), F/S always Q0's, the F/S-disagreement rule
downgrading only a 일치. Q.6.9: (c) is the LOWERED-s manipulation (s 0.7, fallback 0.8); ①'s 일치 needs
rho(base naive P_X, dr_P_c) >= +rho_min over pairs with naive P_X > 0 only, the P_X = 0 pairs recorded as "already at
floor". Part 1: helpers, F/S stability, noise floor, rules ⑤ ① ③. Part 2: ② (increase only, apl_mbon05 only;
the floor guard of Q.6.3 kept), ④ (two-sided, direction recorded, ①-overlap rho on ①'s pair set), sensitivity
(borderline / A·Y cancel pairs dropped from F), records and assembly (weak recomputed from the s_c block)."""
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
    """Q.6.9 ordering: (c) 조작 불가 (s None) -> 판단 불가 (fixed); (c) weak -> 판단 불가 (fixed); Q0 naive-P_X AUC <= auc_mismatch -> 불일치 (Q0 only, survives an
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
    if s is None:                               # Q.6.9: both lowered s outside the KC band -> 조작 불가, ① 판단 불가
        return dict(rec, label=UNDECIDED, why=f"(c) s 하향 조작 {spec.unmanip_note} (s {list(spec.s_down)} 모두 KC 대역 "
                                              f"밖) — ①은 판단 불가로 고정 (Q.6.9)")
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


# ================================================================ ② APL 억제 and ④ 작동점 (Q.6.3, Q.6.5, Q.6.6, Q.6.9)
def paired_shift(Cv, B, F, noise, spec) -> dict:
    """Q.6.3: per-F-pair dr_P (condition - baseline), two-sided Wilcoxon, threshold max(delta_min, noise_mult x noise)."""
    d = {k: Cv[k]["r_P"] - B[k]["r_P"] for k in F}
    return dict(delta=d, median=med(list(d.values())), p=wilcoxon_p(list(d.values())),
                threshold=max(spec.delta_min, spec.noise_mult * noise))


def _cn(spec) -> dict:
    """Condition names by role, unpacked from spec.cond_names (base, (b), (b-record), (c), (d) lo, (d) hi)."""
    base, b, b_rec, c, lo, hi = spec.cond_names
    return dict(base=base, b=b, b_rec=b_rec, c=c, lo=lo, hi=hi)


def _rho_keys(B, spec) -> list:
    """The same restriction as ①'s rho (Q.6.9): pairs with BASELINE naive P_X > 0 when rho_positive_only."""
    keys = sorted(B)
    return [k for k in keys if B[k]["naive_px"] > 0] if spec.rho_positive_only else keys


def rule_apl(Cv, B, F, noise, spec, role: str = "primary") -> dict:
    """② (Q.6.3 + Q.6.6): increase only — median dr_P >= threshold and Wilcoxon p < wilcoxon_p, and the F median of
    the ratio |dP_X| / naive P_X rises over the baseline. Reads only the table it is given: assemble passes
    apl_mbon05 (primary); apl_nonkc is a record built by a separate call and never gates or labels ②."""
    if Cv is None:
        return dict(label=UNDECIDED, why="(b) INVALID — 결과를 쓰지 않음", role=role)
    sh = paired_shift(Cv, B, F, noise, spec)
    rd = med([Cv[k]["ratio"] - B[k]["ratio"] for k in F if Cv[k]["ratio"] is not None and B[k]["ratio"] is not None])
    nm = med([B[k]["naive_px"] for k in F])
    high = [k for k in F if B[k]["naive_px"] >= nm]
    # Q.6.3: "증가가 바닥 쌍에만 있으면 ①과 겹침" 가드는 유지한다 — the rise must reach >= floor_guard_pairs F pairs
    # at or above the F median naive P_X.
    guard = sum(sh["delta"][k] >= sh["threshold"] for k in high)
    rec = dict(sh, ratio_diff_median=rd, naive_px_F_median=nm, guard_pairs=guard, role=role)
    main = sh["p"] < spec.wilcoxon_p and sh["median"] is not None and sh["median"] >= sh["threshold"]
    if sh["median"] is not None and sh["median"] <= 0:
        lab, why = MISMATCH, f"중앙값 Δr_P {sh['median']:.3f} ≤ 0"
    elif main and rd is not None and rd > 0:
        if guard >= spec.floor_guard_pairs:
            lab, why = MATCH, (f"중앙값 Δr_P {sh['median']:.3f} ≥ {sh['threshold']:.3f}, p {sh['p']:.4f}, "
                               f"비율 차 {rd:.3f} > 0, 순진 P_X 중앙값 이상 F {guard}쌍")
        else:
            lab, why = UNDECIDED, f"증가가 순진 P_X 낮은 F 쌍에만({guard}쌍) — ①과 겹침"
    else:
        lab, why = UNDECIDED, f"중앙값 {sh['median']}, p {sh['p']}, 비율 차 {rd}"
    return dict(rec, label=lab, why=why)


def _direction(m) -> str | None:
    return None if m is None or m == 0 else ("증가" if m > 0 else "감소")


def rule_operating(L, H, B, F, noise, spec) -> dict:
    """④ (Q.6.5, Q.6.9): per scale, |median dr_P| >= threshold and two-sided Wilcoxon p < wilcoxon_p (direction
    recorded). ①-overlap rho(base naive P_X, dr_P) over ①'s pair set: mv_lo overlaps when rho >= +rho_min, mv_hi when
    rho <= -rho_min. 불일치 needs both scales valid with |median dr_P| < noise floor."""
    rk = _rho_keys(B, spec)
    cn = _cn(spec)
    per = {}
    for name, V, lo in ((cn["lo"], L, True), (cn["hi"], H, False)):
        if V is None:
            per[name] = dict(valid=False)
            continue
        sh = paired_shift(V, B, F, noise, spec)
        rho = spearman([B[k]["naive_px"] for k in rk], [V[k]["r_P"] - B[k]["r_P"] for k in rk])
        overlap = rho is not None and (rho >= spec.rho_min if lo else rho <= -spec.rho_min)
        m = sh["median"]
        size = None if m is None else (abs(m) if spec.change_two_sided else m)
        big = sh["p"] < spec.wilcoxon_p and size is not None and size >= sh["threshold"]
        small = m is not None and abs(m) < noise
        per[name] = dict(valid=True, median=m, direction=_direction(m), p=sh["p"], threshold=sh["threshold"],
                         rho=rho, rho_n=len(rk), overlap=overlap, big=big, small=small, delta=sh["delta"])
    valid = [n for n in per if per[n]["valid"]]
    if not valid:
        lab, why = UNDECIDED, "두 배율 모두 INVALID"
    elif any(per[n]["big"] and not per[n]["overlap"] for n in valid):
        lab, why = MATCH, ", ".join(f"{n} |중앙값 Δr_P| {abs(per[n]['median']):.3f} ({per[n]['direction']}), "
                                    f"p {per[n]['p']:.4f}" for n in valid if per[n]["big"] and not per[n]["overlap"])
    elif len(valid) == len(per) and all(per[n]["small"] for n in valid):
        lab, why = MISMATCH, f"유효한 두 배율 모두 |중앙값 Δr_P| < 잡음 바닥 {noise:.3f}"
    elif any(per[n]["big"] for n in valid):
        lab, why = UNDECIDED, "변화가 순진 P_X 방향으로 몰림 — ①과 겹침"
    else:
        lab, why = UNDECIDED, "변화가 기준에 못 미침" + ("" if len(valid) == len(per) else " (한 배율 INVALID)")
    return dict(label=lab, why=why, per=per, limitation=spec.mv_limitation)


# ================================================================ records and assembly
def transitions(B, Cv, spec) -> dict | None:
    if Cv is None:
        return None
    out = {"pass_to_pass": 0, "pass_to_fail": 0, "fail_to_pass": 0, "fail_to_fail": 0}
    for k in sorted(B):
        a = "pass" if B[k]["r_P"] >= spec.split_r else "fail"
        b = "pass" if Cv[k]["r_P"] >= spec.split_r else "fail"
        out[f"{a}_to_{b}"] += 1
    return out


def candidates(V0, B, F, S, stab, noise, q1_vals: dict, s_c, spec) -> dict:
    cn = _cn(spec)
    return dict(variation=rule_variation(V0, B, F, S, stab, spec),
                floor=rule_floor(V0, B, F, S, stab, q1_vals.get(cn["c"]), s_c, spec),
                apl=rule_apl(q1_vals.get(cn["b"]), B, F, noise, spec),          # (b-record) never enters ②
                reach=rule_reach(B, F, S, stab, spec),
                operating=rule_operating(q1_vals.get(cn["lo"]), q1_vals.get(cn["hi"]), B, F, noise, spec))


SYMBOL = dict(floor="① 바닥", apl="② APL 억제", reach="③ 편집 도달", operating="④ 작동점", variation="⑤ 변동")


def sentences(cands: dict, spec) -> list:
    out = []
    for k in ("floor", "apl", "reach", "operating", "variation"):
        c = cands[k]
        s = f"{SYMBOL[k]}: {c['label']} — {c['why']}"
        if k == "operating":
            s += f" (한계: {spec.mv_limitation})"
        out.append(s)
    return out


def _labels(cands: dict) -> dict:
    return {k: c["label"] for k, c in cands.items()}


def s_up_note(spec) -> str:
    """Q.6.9: the condition still named "s_up" is (c) = the LOWERED-s manipulation."""
    first, fallback = spec.s_down
    return f"{_cn(spec)['c']} = (c) s 하향 조작 (Q.6.9: s {first}, 대역 밖이면 {fallback})"


def _s_c_used(s_c: dict, spec) -> dict:
    """weak is recomputed here (|s - base s| < s_weak, base s = spec.strength = 1.0), never trusted from the input."""
    s = s_c.get("s")
    weak = s is not None and abs(float(s) - spec.strength) < spec.s_weak
    return dict(s_c, weak=bool(weak), weak_in=s_c.get("weak"), note=s_up_note(spec))


def assemble(q0: dict, q1: dict, s_c: dict, spec) -> dict:
    sc = _s_c_used(s_c, spec)
    cn = _cn(spec)
    base = q1.get(cn["base"])
    if q0["status"] != "OK" or base is None or base["status"] != "OK":
        why = "Q0 INVALID" if q0["status"] != "OK" else "기준선 INVALID"
        cands = {k: dict(label=UNDECIDED, why=why) for k in SYMBOL}
        return dict(candidates=cands, sentences=sentences(cands, spec), invalid=why, s_c=sc,
                    s_up_note=sc["note"])
    V0, B = q0["vals"], base["vals"]
    F, S = q0["split"]["F"], q0["split"]["S"]
    stab = fs_stability(F, S, {k: v["r"] for k, v in B.items()}, spec)
    noise = noise_floor(V0, B)
    vals = {n: (q1[n]["vals"] if q1.get(n) and q1[n]["status"] == "OK" else None) for n in spec.cond_names if n != cn["base"]}
    cands = candidates(V0, B, F, S, stab, noise, vals, sc, spec)
    border = [k for k in F if k in set(q0["borderline"])]
    F_nb = [k for k in F if k not in set(border)]
    F_nc = [k for k in F if k not in set(q0["cancel"])]
    sens_b = candidates(V0, B, F_nb, S, stab, noise, vals, sc, spec)
    sens_c = candidates(V0, B, F_nc, S, stab, noise, vals, sc, spec)
    cancel_base = [k for k in F if B[k]["r_P"] >= spec.split_r and B[k]["r"] < spec.split_r]
    decomp = {g: {ph: {f: med([V[k]["decomp"][f] for k in keys if "decomp" in V[k]])
                       for f in ("dA_X", "dA_Y", "dP_X", "dP_Y")} for ph, V in (("q0", V0), ("base", B))}
              for g, keys in (("F", F), ("S", S))}
    conds = {n: {f: q1[n].get(f) for f in ("status", "reasons", "kc_median", "d6a_over_share", "apl_out_median",
                                            "naive_px_median", "jaccard_median", "condition")}
             for n in spec.cond_names if q1.get(n)}
    records = dict(conditions=conds, noise_floor=noise, s_c=sc, s_up_note=sc["note"],
                   transitions={n: transitions(B, vals.get(n), spec) for n in spec.cond_names if n != cn["base"]},
                   decomposition=decomp,
                   apl_nonkc=rule_apl(vals.get(cn["b_rec"]), B, F, noise, spec, role="record"),
                   stop_share={k: dict(q0=V0[k].get("stop_naive"), base_naive=B[k].get("stop_naive"),
                                       base_post=B[k].get("stop_post")) for k in sorted(B)})
    return dict(candidates=cands, sentences=sentences(cands, spec), fs=stab, noise_floor=noise,
                borderline=border, sensitivity_borderline=_labels(sens_b),
                sensitivity_borderline_detail=sens_b, cancel=dict(q0=list(q0["cancel"]), base=cancel_base,
                                                                  note=spec.cancel_note),
                sensitivity_cancel=_labels(sens_c), sensitivity_cancel_detail=sens_c, records=records,
                s_c=sc, s_up_note=sc["note"], classification_unstable=stab["unstable"])
