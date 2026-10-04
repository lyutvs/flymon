"""V's decisions (V.3, V.4, V.7 as amended by V.9). The judgement code is the authoritative source: the bands are
r_rules.read_band itself (V.4 = U.4 = T.4 = R.3's order, n_a 43), G_fail_S is S's (net drop ≥ 3 per axis) and gate ②'s
order is S's on block h4's z, with V.9.2's second scale (ℓ_L(z_V) / ℓ_C(h4 z) ≥ 0.5 both directions) after it. New
here: the reuse of R, T and U (V.3 1), the path reproduction (V.3 2, V.9.5), the KC input gate (V.3 3, V.9.1), the set
outcome (V.2), z_V (V.3 5 with SD in the sentence), the combined-engine KC band (V.3 6, V.9.1), the even gates (V.3 7,
V.9.3), the sentences (V.7 with V.9.3 / V.9.4 / V.9.7; 〈…〉 → {field}) and the OCs (V.6: the independent model with V's notes
and T's cluster method on V's set). Every number is a field of the VSpec passed in; no OC value changes a band."""
from __future__ import annotations

import dataclasses
import hashlib

from . import r_rules, s_rules, t_rules, u_rules
from .h3_store import canonical
from .t_spec import SPEC as T_SPEC

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (t_rules.PASS, t_rules.INVALID, t_rules.NOT_READ,
                                                       t_rules.SEALED, t_rules.READ, t_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = t_rules.STOP_REUSE, t_rules.STOP_SET_SHORT
STOP_V_PATH_REPRO, STOP_Z_DEGENERATE = "STOP_V_PATH_REPRO", t_rules.STOP_Z_DEGENERATE
STOP_STRENGTH_LEVER = t_rules.STOP_STRENGTH_LEVER
STOP_EVEN_REPRO, STOP_EVEN_PUNISH, STOP_EVEN_LOW_LEVER = (t_rules.STOP_EVEN_REPRO, u_rules.STOP_EVEN_PUNISH,
                                                          t_rules.STOP_EVEN_LOW_LEVER)
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (t_rules.STOP_PUNISH_BROKEN, t_rules.STOP_P_REFERENCE,
                                                              t_rules.STOP_PUNISH_WEAKENED)
SELECTED, B_TB, B_FA, B_NC, B_PG = t_rules.SELECTED, t_rules.B_TB, t_rules.B_FA, t_rules.B_NC, t_rules.B_PG
BANDS = t_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = t_rules.C_ABOVE_BAR, t_rules.BELOW_BAR, t_rules.MARGIN
NONE_ENGINE, APL_ENGINE, V_ENGINE = "편집 없는 엔진", "APL→MBON05 제거 단독 엔진", "조합 엔진"
POOL_COND = " (POOL 안 짝수 쌍 조건부)"               # V.9.3: appended to the two even STOP sentences
read_band = r_rules.read_band                         # V.4: R.3's order
g_fail_s = s_rules.g_fail_s                           # V.4: S's guard, each condition on its own z
even_repro = u_rules.even_repro                       # V.3 7: C (R's even raw, h4 z) reproduces c_even 7 first
even_validity = u_rules.even_validity                 # R's gate ③ validity list for L_V


# ================================================================ gates before the set is used
def reuse(r_doc, r_git, t_doc, t_git, u_doc, u_git, shared_key, t_key, u_key, spec) -> dict:
    """V.3 1: R's repro and T's unedited z as U.3 1 (u_rules.reuse); U's blocks iff the current V measurement key is the
    declared U key 8a4e0930…, U's summary is tracked and clean, and U's reuse / path / scan / kc blocks carry it (with
    R's shared key and T's key), path passed, the path and scan details' shas are the declared ones, the scan's entry_f0
    point is valid on CSC 2d359b8b…, and U's KC points agree on per_odour_none. Else STOP_REUSE."""
    r = u_rules.reuse(r_doc, r_git, t_doc, t_git, shared_key, t_key, spec)
    why = list(r.get("reasons") or [])
    if u_key != spec.u_measure_key_u:
        why.append(f"V 측정 키 {u_key} ≠ U 측정 키 {spec.u_measure_key_u}")
    if not u_git.get("tracked") or u_git.get("dirty"):
        why.append(f"{spec.u_summary} 미커밋")
    for b, _ in spec.u_commits:
        blk = u_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"U 블록 {b} 없음")
            continue
        if (blk.get("u_measure_key"), blk.get("code_key"), blk.get("t_measure_key")) != (
                spec.u_measure_key_u, spec.r_shared_key, spec.t_measure_key_t):
            why.append(f"U 블록 {b}의 키 {blk.get('u_measure_key')} / {blk.get('code_key')} / {blk.get('t_measure_key')}")
    path, scan = u_doc.get("path") or {}, u_doc.get("scan") or {}
    if path.get("outcome") != PASS or path.get("detail_sha256") != spec.u_path_detail_sha256:
        why.append(f"U 블록 path {path.get('outcome')} / 원자료 sha {path.get('detail_sha256')}")
    if scan.get("detail_sha256") != spec.u_scan_detail_sha256:
        why.append(f"U 블록 scan 원자료 sha {scan.get('detail_sha256')}")
    entry = ((scan.get("contrast") or {}).get(spec.u_entry_point) or {})
    if entry.get("invalid") != [] or (entry.get("side") or {}).get("csc_sha256") != [spec.sha_combined]:
        why.append(f"U {spec.u_entry_point} 행 무효 또는 CSC {(entry.get('side') or {}).get('csc_sha256')}")
    try:
        spec.u_kc_none(u_doc)
    except (KeyError, TypeError, ValueError) as e:
        why.append(f"U 블록 kc의 per_odour_none 읽기 실패({e})")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


def path(checks: list, csc_bad: list, spec) -> dict:
    """V.3 2: checks = [dict(engine, ref, diffs, invalid)] in measurement order; csc_bad = v_records.csc_reasons. A
    defect (presentations, edge labels, the static edit facts) is INVALID; any bit difference is STOP_V_PATH_REPRO, its
    sentence naming the first failing check, every failing check listed."""
    bad = list(csc_bad) + [m for c in checks for m in c.get("invalid", [])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    return _repro(checks)


def _repro(checks: list) -> dict:
    failed = [c for c in checks if c["diffs"]]
    if failed:
        c = failed[0]
        return dict(outcome=STOP_V_PATH_REPRO, reasons=[], failed=[dict(engine=x["engine"], ref=x["ref"],
                                                                         diffs=x["diffs"]) for x in failed],
                    sentence=sentence(STOP_V_PATH_REPRO, dict(engine=c["engine"], ref=c["ref"],
                                                              diff="; ".join(c["diffs"]))))
    return dict(outcome=PASS, reasons=[], failed=[])


def kc_input(rec: dict, u_cmp: dict, n_odours: int, spec) -> dict:
    """V.3 3 / V.9.1: INVALID unless both engines hold every candidate odour on all strength seeds, the unedited engine
    0 edges on the unedited CSC and L_V 2 edges on CSC 2d359b8b…; then the unedited medians must equal U's
    per_odour_none on the shared odours bit for bit (STOP_V_PATH_REPRO). The band is a filter here, not a gate."""
    bad = []
    for e, edges, sha in (("none", 0, spec.sha_none), ("lever", spec.lever_edges, spec.sha_combined)):
        r = rec[e]
        if r["n_odours"] != n_odours or r["n_seeds"] != [len(spec.kc_seeds())]:
            bad.append(f"{e}: {r['n_odours']} odours × {r['n_seeds']} seeds, declared {n_odours} × "
                       f"{len(spec.kc_seeds())}")
        if r["edit_edges"] != [edges] or r["csc_sha256"] != [sha]:
            bad.append(f"{e}: edges {r['edit_edges']} on CSC {r['csc_sha256']}, declared {edges} on {sha}")
    if not rec["cap"]["all_under"]:
        bad.append("a candidate odour above the ORN cap")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    return _repro([dict(engine=NONE_ENGINE, ref="U KC 블록(c0d09a7) per_odour_none", diffs=u_cmp["diffs"])])


def set_outcome(js: dict) -> dict:
    if js["status"] == STOP_SET_SHORT:
        return dict(outcome=STOP_SET_SHORT, sentence=sentence(STOP_SET_SHORT, dict(b=js["n_b"], a=js["n_a"])))
    return dict(outcome=PASS)


def z_v(side: dict, n: int, readout: dict, spec) -> dict:
    """V.3 5: T's guard decision (t_rules.z_lever: edges, 96 + 96, median Δ ≥ 5, zero share ≤ 0.25, SD > 0) on the
    combined engine's side from block path, the chain-entry edges as declared (else INVALID), and V's sentence (with
    the SD) on a failure."""
    want = canonical(dict(spec.contrast_declared()["chain_entry"]))
    d = t_rules.z_lever(side, n, readout, spec)
    bad = list(d.get("reasons") or []) if d["outcome"] == INVALID else []
    if side["block_edges"] != [want]:
        bad.append(f"chain entry edges {side['block_edges']}, declared {want}")
    if side["csc_sha256"] != [spec.sha_combined]:
        bad.append(f"CSC {side['csc_sha256']}, declared {spec.sha_combined}")
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    if d["outcome"] != PASS:
        g, z, key = side["guard"], side.get("z"), {t: k for k, t in readout.items()}
        sd = [f"{float(z[key[t]][1]):.3f}" if z else ("0" if t in side["zero_sd"] else "—") for t in d["failed"]]
        return dict(outcome=STOP_Z_DEGENERATE, reasons=d.get("reasons", []), failed=d["failed"],
                    sentence=sentence(STOP_Z_DEGENERATE, dict(
                        type="·".join(d["failed"]), med="·".join(f"{g[t]['median_delta']:.1f}" for t in d["failed"]),
                        zero="·".join(f"{g[t]['zero_share']:.3f}" for t in d["failed"]), sd="·".join(sd))))
    return dict(outcome=PASS, reasons=[])


def kc_band(rec112: dict, cap_ok: bool, set_rec: dict, recheck: list, spec) -> dict:
    """V.3 6 / V.9.1: R's gate ① rule on the 112 calibration odours under L_V (r_rules.gate1's INVALID checks, the
    median of per-odour medians in the band, the ORN cap); the V set's odours re-measured under L_V must equal block
    kc_input's L_V values bit for bit (else STOP_V_PATH_REPRO) and lie in the band; else STOP_STRENGTH_LEVER (V's
    sentence)."""
    g1 = r_rules.gate1(rec112, cap_ok, spec)
    bad = list(g1.get("reasons") or []) if g1["outcome"] == INVALID else []
    if set_rec["edit_edges"] != [spec.lever_edges] or set_rec["csc_sha256"] != [spec.sha_combined]:
        bad.append(f"V set odours: edges {set_rec['edit_edges']} on {set_rec['csc_sha256']}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[], calib=g1)
    rp = _repro([dict(engine=V_ENGINE, ref="KC 입력 블록의 V 세트 냄새 값", diffs=recheck)])
    if rp["outcome"] != PASS:
        return dict(rp, calib=g1)
    lo, hi = spec.valid_band
    failed = list(g1.get("failed", [])) + [f"V 세트 냄새 {o} {v:.4f} ∉ [{lo}, {hi}]"
                                          for o, v in sorted(set_rec["per_odour"].items()) if not lo <= v <= hi]
    if failed:
        return dict(outcome=STOP_STRENGTH_LEVER, reasons=[], failed=failed, calib=g1,
                    sentence=sentence(STOP_STRENGTH_LEVER, dict(cond="; ".join(failed))))
    return dict(outcome=PASS, reasons=[], failed=[], calib=g1)


def even(record: dict, c_even: int, spec) -> dict:
    """V.3 7 / V.9.3 on L_V's even record (u_records.even_record): ① (b) net drop < even_drop_b_lt and (a) net drop ≤
    even_drop_a_le (else STOP_EVEN_PUNISH); ② testable_b ≥ bar_b (else STOP_EVEN_LOW_LEVER); else PASS."""
    d = record["drops"]
    db, da = d["b"]["net_drop"], d["a"]["net_drop"]
    if not (db < spec.even_drop_b_lt and da <= spec.even_drop_a_le):
        return dict(outcome=STOP_EVEN_PUNISH, reasons=[], sentence=sentence(STOP_EVEN_PUNISH, dict(d_b=db, d_a=da)))
    if record["testable_b"] < spec.bar_b:
        return dict(outcome=STOP_EVEN_LOW_LEVER, reasons=[], sentence=sentence(STOP_EVEN_LOW_LEVER, dict(
            tb=record["testable_b"], c_even=c_even, k=record["F_a"])))
    return dict(outcome=PASS, reasons=[])


def gate2(res_l: dict, res_c: dict, ratio_h4: dict | None, ratio_zv: dict | None, spec) -> dict:
    """V.3 10 with V.9.2: S's order on block h4's z (INVALID → STOP_PUNISH_BROKEN → STOP_P_REFERENCE → h4 ratio <
    p_ratio_min → STOP_PUNISH_WEAKENED); then the z_V ratio ℓ_L(z_V) / ℓ_C(h4 z) < p_ratio_min in either direction →
    STOP_PUNISH_WEAKENED; else PASS. On STOP_PUNISH_WEAKENED the sentence is V.9.7 1's V-only sentence: both scales'
    ratios always, and every failing scale·direction named (`failed_scales`, h4 first). `scale` names the first
    failing scale in the order (h4, then z_V); `low` its failing directions."""
    d = s_rules.gate2(res_l, res_c, ratio_h4, spec)
    if d["outcome"] not in (PASS, STOP_PUNISH_WEAKENED):
        return dict(d, scale=None)
    names = list(spec.p.directions)
    failed = [[sc, x] for sc, r in (("h4", ratio_h4), ("z_V", ratio_zv)) for x in names
              if r[x]["ratio"] < spec.p_ratio_min]
    if not failed:
        return dict(d, scale=None)
    scale = failed[0][0]
    f = dict(h1=f"{ratio_h4[names[0]]['ratio']:.3f}", h2=f"{ratio_h4[names[-1]]['ratio']:.3f}",
             v1=f"{ratio_zv[names[0]]['ratio']:.3f}", v2=f"{ratio_zv[names[-1]]['ratio']:.3f}",
             miss=", ".join(f"{_SCALE_TEXT[sc]} ℓ_{x}" for sc, x in failed))
    return dict(d, outcome=STOP_PUNISH_WEAKENED, low=[x for sc, x in failed if sc == scale], scale=scale,
                failed_scales=failed, sentence=sentence(STOP_PUNISH_WEAKENED, f))


_SCALE_TEXT = {"h4": "h4 z", "z_V": "z_V"}


# ================================================================ the closing sentences (V.7, V.9.3, V.9.4, V.9.7)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43, 조합 지렛대)"
_LEVER = ("APL→MBON05 2간선 제거와 MBON05→MBON09/MBON11/MBON01 11간선 제거를 함께 한 모델 변형(커넥톰 간선 13개 제거)")
_FILTERED = "편집 없는 엔진과 조합 엔진 둘 다에서 KC 입력으로 거른"                 # V.9.7 3 (B_Tb, SELECTED)
SENTENCES = {
    STOP_REUSE: "R·T·U 재사용 조건(V.3 1)이 깨졌다({why}). V는 R·T·U 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 "
                "쓰지 않고 멈춘다 — 사용자 몫.",
    STOP_V_PATH_REPRO: "V 편집 경로가 {engine}에서 {ref}을 재현하지 못했다({diff}).",
    STOP_SET_SHORT: "KC 입력으로 거른 넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다({b}·{a}쌍).",
    STOP_Z_DEGENERATE: "조합 지렛대 아래 기준 집합에서 판독 {type}이 반응성 가드를 넘지 못했다(Δ 중앙값 {med}, 0 비율 "
                       "{zero}, SD {sd}) — z를 정할 수 없다.",
    STOP_STRENGTH_LEVER: "조합 지렛대 아래에서 E-grid k2-norm s 1.0이 KC 유효 대역을 잃었다({cond}).",
    STOP_EVEN_REPRO: t_rules.SENTENCES[STOP_EVEN_REPRO],
    STOP_EVEN_PUNISH: "조합 지렛대가 짝수 쌍에서 처벌 순감소 거름 기준((b) < 3, (a) ≤ 1)을 넘었다((b) {d_b}, (a) {d_a})."
                      + POOL_COND,
    STOP_EVEN_LOW_LEVER: r_rules.SENTENCES[STOP_EVEN_LOW_LEVER].replace("APL→MBON05 제거", "조합 지렛대", 1) + POOL_COND,
    STOP_PUNISH_BROKEN: s_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: s_rules.SENTENCES[STOP_P_REFERENCE],
    STOP_PUNISH_WEAKENED: "조합 지렛대 아래 같은 시드 P에서 처벌 학습량 비가 0.5에 못 미쳤다(h4 z 비 ℓ_r1 {h1}·ℓ_r2 {h2}, "
                          "z_V 비 ℓ_r1 {v1}·ℓ_r2 {v2}; 미달: {miss}).",                 # V.9.7 1
    B_TB: _LEVER + "이 " + _FILTERED + " 넓힌 상대 풀 판정 세트(생성원 턴 0–{T})에서 {n}/21로 지렛대 없는 같은 세트 "
                   "{c}/21보다 오르지 않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 이 조합 지렛대 주장을 "
                   "닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: s_rules.SENTENCES[B_PG] + " (조합 지렛대)",
    B_FA: s_rules.SENTENCES[B_FA] + " (조합 지렛대)",
    SELECTED: "C3에서 " + _LEVER + "과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 때, "
              + _FILTERED + " 판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–{T}, 기존 세트·키·"
              "사구체 중복 제외, 모든 행이 POOL 밖 상대 타입 조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 "
              "만족했고 지렛대 없는 같은 세트보다 높았다({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 "
              "≥ 3) 통과, 같은 시드 P 비 h4 z ℓ_r1 {rho1}·ℓ_r2 {rho2}, z_V ℓ_r1 {rho1_zv}·ℓ_r2 {rho2_zv} ≥ 0.5(약화 정도 "
              "기록), 짝수 {k_even}/21, 판정 시드 24_600_xxx). 판정 세트의 상대는 1세대 기본 폼으로 넓힌 풀이며 새 키는 POOL에 없는 상대 타입 조합에서 나온다 "
              "— 과제 풀 안의 시험 가능성은 R·S가 마지막이다. (b) 21쌍은 {k_cl}개 상대 타입 조합 쌍에 몰려 있어 쌍끼리 "
              "독립이 아니다. 지렛대는 U 기전 기록(사슬 진입부 차단에서 MBON13 회복)을 보고 골랐고 같은 측정의 출력 차단 "
              "읽기는 '사슬 비지지'였다. 이것은 Q 결과로 고른 지렛대 계열의 R·S·T·U에 이은 다섯 번째 시도다(V.0). V 세트 "
              "군집 상관 모형에서 G_fail_S null {cl_null}, 오선택(harm) {cl_harm1} / {cl_harm2}. 이 결과는 이 64쌍 조건부 "
              "기술 결과다. 작동 특성은 V 세트 조건부 값이며 Q → R → S → T → U → V 전체 절차의 오선택률이 아니다. 실제 "
              "커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 "
              "가능성이 섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristics (V.6, V.9.5 P2-9)
OC_NOTES = (s_rules.OC_NOTES[0],
            "V.6: 독립 가정은 V 세트에서 낙관적이다 — (b)·(a) 쌍이 상대 타입 조합 군집에 몰려 있다. V 세트의 군집으로 계산한 "
            "군집 상관 모형(T.9.6·T.9.7과 같은 방법, ICC 0.3)을 옆에 기록한다.",
            "V.6: 지렛대를 U 기전 기록을 보고 골랐으므로 짝수 관문은 낙관적일 수 있다.",
            "작동 특성은 V 세트 조건부 값이며 Q → R → S → T → U → V 전체 절차의 오선택률이 아니다.")


def _sealed(table: dict) -> dict:
    t = {k: v for k, v in table.items() if k != "sha256"}
    return dict(t, sha256=hashlib.sha256(canonical(t).encode()).hexdigest())


def oc(spec) -> dict:
    """V.6: S.6's exact independent model (s_rules.oc) on n_b 21 · n_a 43 with V's notes."""
    table = {k: v for k, v in s_rules.oc(spec).items() if k != "sha256"}
    table["notes"] = list(OC_NOTES)
    return _sealed(table)


def oc_cluster(clusters_b: list, clusters_a: list) -> dict:
    """V.6 / V.9.5 P2-9: T's cluster-correlated model (t_rules.oc_cluster: ICC 0.3, 10⁵ draws, T's seed, T's g rows and
    grids) on V's set clusters from block set, with V's notes; the sha is over the table."""
    sp = dataclasses.replace(T_SPEC, clusters_b=tuple(tuple(c) for c in clusters_b),
                             clusters_a=tuple(tuple(c) for c in clusters_a))
    table = t_rules.oc_cluster(sp)
    return _sealed(dict(table, notes=list(OC_NOTES[1:])))


def cluster_values(table: dict) -> dict:
    """V.9.4: G_fail_S null and the two harm rows of the cluster model, formatted for the SELECTED sentence."""
    g = table["g_fail"]
    null = [r["p"] for r in g if r["kind"] == "null"]
    harm = [r["p"] for r in g if r["kind"] == "harm"]
    return dict(cl_null=f"{null[0]:.3f}", cl_harm1=f"{harm[0]:.3f}", cl_harm2=f"{harm[1]:.3f}")
