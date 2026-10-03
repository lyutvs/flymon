"""T's decisions (T.1-T.4, T.6, T.7 as replaced by T.9). The judgement code is the authoritative source: the bands are
r_rules.read_band itself (T.4 = S.4 = R.3's order) read with T's numbers (n_a 43), G_fail_S is S's (s_rules.g_fail_s,
net drop ≥ 3 per axis) and gate ②'s order is S's (s_rules.gate2, read on block h4's z, T.9.3). New here: the reuse of
R's repro and gate ① (T.3 1), the set outcome (T.2 / T.9.1), the z gates (T.1, T.9.4, T.9.7: STOP_Z_REPRO, then
STOP_Z_DEGENERATE), the gate-① supplement on the T set's odours (T.9.1), gate ③ with R's C reproduction first (T.9.2),
the sentences (T.7 as replaced by T.9.5 / T.9.7, 〈…〉 → {field}) and the two operating characteristics (T.6, T.9.6,
T.9.7). Every number is a field of the TSpec passed in; no OC value ever changes a band."""
from __future__ import annotations

import hashlib

import numpy as np

from ..agent import e_rules
from . import r_rules, s_rules
from .h3_store import canonical

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (r_rules.PASS, r_rules.INVALID, r_rules.NOT_READ, r_rules.SEALED,
                                                       r_rules.READ, r_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = s_rules.STOP_REUSE, s_rules.STOP_SET_SHORT
STOP_Z_REPRO, STOP_Z_DEGENERATE = "STOP_Z_REPRO", "STOP_Z_DEGENERATE"
STOP_STRENGTH_LEVER = r_rules.STOP_STRENGTH_LEVER
STOP_EVEN_REPRO, STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH = ("STOP_EVEN_REPRO", r_rules.STOP_EVEN_LOW_LEVER,
                                                              r_rules.STOP_C_EVEN_MISMATCH)
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (s_rules.STOP_PUNISH_BROKEN, s_rules.STOP_P_REFERENCE,
                                                              s_rules.STOP_PUNISH_WEAKENED)
SELECTED, B_TB, B_FA, B_NC, B_PG = r_rules.SELECTED, r_rules.B_TB, r_rules.B_FA, r_rules.B_NC, r_rules.B_PG
BANDS = r_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = r_rules.C_ABOVE_BAR, r_rules.BELOW_BAR, r_rules.MARGIN
read_band = r_rules.read_band                      # T.4 = S.4: R.3's order (plan Reading 9)
g_fail_s = s_rules.g_fail_s                        # T.4: S's guard, each condition on its own z
gate2 = s_rules.gate2                              # T.3 6 / T.9.3: S's order, read on block h4's z


# ================================================================ gates before the set is used
def reuse(r_doc: dict, r_git: dict, shared_key: str, spec) -> dict:
    """T.3 1: R's repro and gate ① are reused iff the shared measurement key equals R's, R's summary is tracked and
    clean, and both blocks exist with R's key and passed. Else STOP_REUSE (plan Reading 8)."""
    why = []
    if shared_key != spec.r_shared_key:
        why.append(f"공유 측정 키 {shared_key} ≠ R {spec.r_shared_key}")
    if not r_git.get("tracked") or r_git.get("dirty"):
        why.append(f"{spec.r_summary} 미커밋")
    for b in spec.r_reused:
        blk = r_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"R 블록 {b} 없음")
            continue
        if blk.get("code_key") != spec.r_shared_key:
            why.append(f"R 블록 {b}의 코드 키 {blk.get('code_key')}")
        ok = blk.get("passed") if b == "repro" else blk.get("outcome") == PASS
        if not ok:
            why.append(f"R 블록 {b} 통과 아님")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


def set_outcome(js: dict) -> dict:
    """T.2: STOP_SET_SHORT when the turns ran out (the declared-value check is the runner's refusal)."""
    if js["status"] == STOP_SET_SHORT:
        return dict(outcome=STOP_SET_SHORT, sentence=sentence(STOP_SET_SHORT, dict(b=js["n_b"], a=js["n_a"])))
    return dict(outcome=PASS)


def _zfmt(z: dict) -> str:
    return ", ".join(f"{k} {v[0]!r} / {v[1]!r}" for k, v in sorted(z.items()))


def _side_reasons(side: dict, name: str, edges: int, n: int, sha=None) -> list:
    bad = []
    if side["edit_edges"] != [edges]:
        bad.append(f"{name}: edges {side['edit_edges']}, declared {edges}")
    if sha is not None and side["csc_sha256"] != [sha]:
        bad.append(f"{name}: CSC {side['csc_sha256']} is not the unedited engine's {sha}")
    if (side["n_ref"], side["n_rest"]) != (n, n):
        bad.append(f"{name}: {side['n_ref']} reference / {side['n_rest']} rest presentations, declared {n}")
    return bad


def z_repro(none: dict, n: int, repro_sha: str, spec) -> dict:
    """T.1 / T.9.6: the unedited engine's reference-set z must equal block h4's C3 z bit for bit (float equality of
    mean and SD), else STOP_Z_REPRO; wrong edges, CSC or presentation counts are INVALID."""
    bad = _side_reasons(none, "none", 0, n, repro_sha)
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    want = spec.z_h4_dict()
    got = None if none["z"] is None else {k: (float(v[0]), float(v[1])) for k, v in none["z"].items()}
    if got != want:
        val = none["why"] if got is None else _zfmt(got)
        return dict(outcome=STOP_Z_REPRO, reasons=[], sentence=sentence(STOP_Z_REPRO, dict(val=val)))
    return dict(outcome=PASS, reasons=[])


def z_lever(lever: dict, n: int, readout: dict, spec) -> dict:
    """T.9.4: under the lever every readout type must pass H.4's guard (median(read − same-seed rest) ≥ 5 and zero
    share ≤ 0.25) and have a nonzero SD (an SD of 0 is the same STOP), else STOP_Z_DEGENERATE naming every failing
    type; z_lever is then the lever's reference-set z."""
    bad = _side_reasons(lever, "lever", spec.lever_edges, n)
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    fails = [t for t in readout.values() if not lever["guard"][t]["passes"] or t in lever["zero_sd"]]
    if fails:
        g = lever["guard"]
        return dict(outcome=STOP_Z_DEGENERATE, reasons=[], failed=fails, sentence=sentence(STOP_Z_DEGENERATE, dict(
            type="·".join(fails), med="·".join(f"{g[t]['median_delta']:.1f}" for t in fails),
            zero="·".join(f"{g[t]['zero_share']:.3f}" for t in fails))))
    return dict(outcome=PASS, reasons=[])


def gate1s(rec: dict, spec) -> dict:
    """T.9.1: every T-set odour's KC activity median under the lever inside valid_band, else STOP_STRENGTH_LEVER;
    INVALID when the lever did not change exactly lever_edges edges, C was edited, or the odour / seed counts differ."""
    bad = []
    if rec["edit_edges"] != [spec.lever_edges] or rec["edit_edges_none"] != [0]:
        bad.append(f"edges lever {rec['edit_edges']} / none {rec['edit_edges_none']}, declared {spec.lever_edges} / 0")
    if rec["n_odours"] != spec.n_set_odours or rec["n_seeds"] != [len(spec.kc_seeds())]:
        bad.append(f"{rec['n_odours']} odours × {rec['n_seeds']} seeds, declared {spec.n_set_odours} × "
                   f"{len(spec.kc_seeds())}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    lo, hi = spec.valid_band
    failed = sorted(o for o, v in rec["per_odour"].items() if not lo <= v <= hi)
    if failed:
        cond = "T 세트 냄새 " + "; ".join(f"{o} {rec['per_odour'][o]:.4f}" for o in failed) + f" ∉ [{lo}, {hi}]"
        return dict(outcome=STOP_STRENGTH_LEVER, reasons=[], failed=failed,
                    sentence=sentence(STOP_STRENGTH_LEVER, dict(cond=cond)))
    return dict(outcome=PASS, reasons=[], failed=[])


def gate3(L, C: dict, spec, repro_sha: str, l_h4) -> dict:
    """T.9.2: C (R's even raw, h4 z) must reproduce c_even_expected first — else STOP_EVEN_REPRO and L is not measured
    (L may be None; l_h4 = R's L even raw read on h4 z, for the sentence); then R's gate ③ rule on L (re-measured,
    z_lever) and C: validity, STOP_C_EVEN_MISMATCH (defensive, unreachable after the reproduction, T.9.6),
    STOP_EVEN_LOW_LEVER when L's testable_b < bar_b."""
    c = (C.get("aggregate") or {}).get("testable_b")
    if C["reasons"] or c != spec.c_even_expected:
        return dict(outcome=STOP_EVEN_REPRO, reasons=[f"C: {m}" for m in C["reasons"]], c_even=c,
                    sentence=sentence(STOP_EVEN_REPRO, dict(l=l_h4, c=c)))
    return r_rules.gate3(L, C, spec, repro_sha)


# ================================================================ the closing sentences (T.7 → T.9.5 / T.9.7)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43)"
SENTENCES = {
    STOP_REUSE: "R 관문 재사용 조건(T.3 1)이 깨졌다({why}). T는 R 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 쓰지 않고 "
                "멈춘다 — 사용자 몫.",
    STOP_SET_SHORT: "넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다({b}·{a}쌍).",
    STOP_Z_REPRO: "편집 없는 엔진에서 기준 집합 z가 블록 h4 값을 재현하지 못했다({val}) — z 절차 결함.",
    STOP_Z_DEGENERATE: "APL→MBON05 제거 아래 기준 집합에서 판독 {type}이 반응성 가드를 넘지 못했다(Δ 중앙값 {med}, 0 비율 "
                       "{zero}) — z를 정할 수 없다.",
    STOP_STRENGTH_LEVER: r_rules.SENTENCES[STOP_STRENGTH_LEVER],
    STOP_EVEN_REPRO: "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L {l}, C {c}).",
    STOP_EVEN_LOW_LEVER: r_rules.SENTENCES[STOP_EVEN_LOW_LEVER],
    STOP_C_EVEN_MISMATCH: r_rules.SENTENCES[STOP_C_EVEN_MISMATCH],
    STOP_PUNISH_BROKEN: s_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: s_rules.SENTENCES[STOP_P_REFERENCE],
    STOP_PUNISH_WEAKENED: s_rules.SENTENCES[STOP_PUNISH_WEAKENED],
    B_TB: "APL→MBON05 제거가 넓힌 상대 풀 판정 세트(생성원 턴 0–{T})에서 {n}/21로 지렛대 없는 같은 세트 {c}/21보다 오르지 "
          "않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 "
          "결과는 R·S 그대로).",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: s_rules.SENTENCES[B_PG],
    B_FA: s_rules.SENTENCES[B_FA],
    SELECTED: "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 "
              "때, 판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–{T}, 기존 키·사구체 중복 제외, 모든 행이 POOL 밖 "
              "상대 타입 조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다"
              "({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 {rho1}·ℓ_r2 "
              "{rho2} ≥ 0.5(약화 정도 기록), 짝수 재판독 {k_even}/21, 판정 시드 24_400_xxx). 판정 세트의 상대는 1세대 기본 폼으로 "
              "넓힌 풀이며, 새 키는 POOL에 없는 상대 타입 조합(12개 중)에서 나온다 — 과제 풀 안의 시험 가능성은 R·S가 "
              "마지막이다. (b) 21쌍은 9개 상대 타입 조합 쌍에 몰려 있어 쌍끼리 독립이 아니다. 지렛대는 Q 결과를 보고 골랐고, "
              "z 규칙은 R·S 판정 결과를 본 뒤 정했다(T.0). 이 판정은 R(뒤에 가드 변경)·S(뒤에 z 규칙)에 이은 같은 지렛대의 세 "
              "번째 판정이다. 작동 특성은 T 세트 조건부 값이며 Q → R → S → T 전체 절차의 오선택률이 아니다. 실제 커넥톰 "
              "간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 "
              "섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristics (T.6, T.9.6, T.9.7)
OC_NOTES = (s_rules.OC_NOTES[0],
            "T.0 / T.6: 독립 가정은 T 세트에서 낙관적이다 — (b) 21쌍은 9개 상대 타입 조합 쌍, (a) 43쌍은 8개 상대 타입 조합에 "
            "몰려 있다. 군집 상관 행(ICC 0.3, 결과 전 고른 예시값)을 옆에 기록한다.",
            "작동 특성은 T 세트 조건부 값이며 Q → R → S → T 전체 절차의 오선택률이 아니다.")


def oc(spec) -> dict:
    """T.6: S.6's exact independent model (s_rules.oc: r_rules._cell, G_fail_S from net drop ≥ 3, R's guard beside it)
    on n_b 21 · n_a 43, with T's notes."""
    table = {k: v for k, v in s_rules.oc(spec).items() if k != "sha256"}
    table["notes"] = list(OC_NOTES)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())


def _g_axis_mc(rng, sizes, a_pf: float, a_fp: float, kappa: float, draws: int, drop: int) -> np.ndarray:
    """Per cluster (pass→fail, fail→pass, same) ~ Dirichlet(κ·(a_pf, a_fp, 1 − a_pf − a_fp)), its pairs multinomial
    (as two binomials); True where the axis' net drop Σ pf − Σ fp ≥ drop."""
    alpha = kappa * np.array([a_pf, a_fp, 1.0 - a_pf - a_fp])
    pf, fp = np.zeros(draws, np.int64), np.zeros(draws, np.int64)
    for n_k in sizes:
        p = rng.dirichlet(alpha, size=draws)
        k_pf = rng.binomial(int(n_k), p[:, 0])
        rest = 1.0 - p[:, 0]
        q_fp = np.clip(np.divide(p[:, 1], rest, out=np.zeros(draws), where=rest > 0), 0.0, 1.0)
        pf += k_pf
        fp += rng.binomial(int(n_k) - k_pf, q_fp)
    return (pf - fp) >= drop


def _cell(pn, pf, c: int, na: int, g: float, spec) -> dict:
    P = {b: 0.0 for b in BANDS}
    for i, wi in enumerate(pn):
        if not wi:
            continue
        for j, wj in enumerate(pf):
            if not wj:
                continue
            for gv, wg in ((False, 1 - g), (True, g)):
                if wg:
                    P[read_band(i, c, j, gv, spec.n_b, spec.n_a, na, spec)["band"]] += wi * wj * wg
    return P


def oc_cluster(spec) -> dict:
    """T.6 / T.9.6 / T.9.7: the cluster-correlated rows beside the independent model — Monte Carlo with one
    default_rng(oc_cluster_seed), consumed in this order: the g rows ((b) clusters then (a) clusters per row), then per
    q in oc_q the (b) testable draws (per cluster Beta(κq, κ(1 − q)), then Binomial) and the (a) draws (per cluster
    Beta, a uniformly random order of the 43 (a) pairs, Bernoulli per pair; F_a at naive_a = the first naive_a pairs).
    κ = (1 − ICC) / ICC; c fixed; G_fail_S independent of n and F_a with the cluster model's probability; each cell read
    through read_band. The clusters are T.9.1's table (spec.clusters_b / clusters_a)."""
    rng = np.random.default_rng(spec.oc_cluster_seed)
    kappa, N = (1.0 - spec.oc_cluster_icc) / spec.oc_cluster_icc, int(spec.oc_cluster_draws)
    sb = [int(n) for *_, n in spec.clusters_b]
    sa = [int(n) for _, n in spec.clusters_a]
    g_rows = []
    for kind, a_pf, a_fp in spec.oc_cluster_g:
        gb = _g_axis_mc(rng, sb, a_pf, a_fp, kappa, N, spec.g_fail_drop)
        ga = _g_axis_mc(rng, sa, a_pf, a_fp, kappa, N, spec.g_fail_drop)
        g_rows.append(dict(kind=kind, a_pf=a_pf, a_fp=a_fp, p=float(np.mean(gb | ga)),
                           p_independent=s_rules.g_prob(a_pf, a_fp, spec)))
    label_a = np.repeat(np.arange(len(sa)), sa)
    n_dist, fa_dist = {}, {}
    for q in spec.oc_q:
        n = np.zeros(N, np.int64)
        for n_k in sb:
            n += rng.binomial(n_k, rng.beta(kappa * q, kappa * (1 - q), size=N))
        n_dist[q] = (np.bincount(n, minlength=spec.n_b + 1) / N).tolist()
        qa = rng.beta(kappa * q, kappa * (1 - q), size=(N, len(sa)))
        order = rng.permuted(np.tile(label_a, (N, 1)), axis=1)
        hit = rng.random((N, label_a.size)) < np.take_along_axis(qa, order, axis=1)
        csum = np.concatenate([np.zeros((N, 1), np.int64), np.cumsum(hit, axis=1)], axis=1)
        fa_dist[q] = {na: (np.bincount(csum[:, na], minlength=na + 1) / N).tolist()
                      for na in range(min(spec.n_a, spec.oc_naive_max) + 1)}
    rows = []
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in range(min(spec.n_a, spec.oc_naive_max) + 1):
                for gr in [dict(kind="none", a_pf=0.0, a_fp=0.0, p=0.0)] + g_rows:
                    rows.append(dict(q=q, c=c, naive_a=na, g_kind=gr["kind"], a_pf=gr["a_pf"], a_fp=gr["a_fp"],
                                     g=gr["p"], P=_cell(n_dist[q], fa_dist[q][na], c, na, gr["p"], spec)))
    table = dict(model="clusters = T.9.1's table; per cluster a shared latent probability: guard (pf, fp, same) ~ "
                       "Dirichlet(κ·(a_pf, a_fp, 1 − a_pf − a_fp)), testable q_k ~ Beta(κq, κ(1 − q)); κ = (1 − ICC) / "
                       "ICC; c fixed; naive (a) pairs a uniformly random subset; G_fail_S independent of n and F_a",
                 icc=spec.oc_cluster_icc, kappa=kappa, draws=N, seed=spec.oc_cluster_seed, clusters_b=sb,
                 clusters_a=sa, g_fail=g_rows,
                 n_dist={str(q): dict(cluster=n_dist[q], independent=_binom(spec.n_b, q)) for q in spec.oc_q},
                 rows=rows, notes=list(OC_NOTES[1:]))
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())


def _binom(n: int, q: float) -> list:
    return [float(x) for x in e_rules._binom(n, q)]
