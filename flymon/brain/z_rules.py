"""Z's order-0 decisions and sentences (Z.7 0a–0d, Z.8, Z.9.2 P1-3 · P0-1 · P2-11 · P3-14). Every number comes from
the ZSpec passed in (or from y_spec.SPEC for Y's filter numbers a sentence prints); every sentence is the spec's text
with 〈…〉 as {…}.
- reuse_stop: STOP_REUSE〈0b / 0c〉 with records_unavailable (Z.8) and the P3-14 row.
- plan_rule_a (0c′, continue rule (가)) and plan_rule_b (0d, (나)): L(−2.41, n) ≥ plan_bar passes — the comparison is
  round(L − bar, 9) ≥ 0 (Y's rounding), so 0.80 itself passes (Z.9.2 T8).
- key_precheck: J_max = |Y admitted| + h, n_Σ^max = |y_rules.merge_groups(Y admitted ∪ P keys)|; J_max < j_min →
  STOP_PILOT_FEW〈0d〉 (Z.9.2 P0-1), computed from keys only (signature: keys in, counts out).
- n_star: 24 if J_max ≥ 24 else 12 (Z.9.2 P1-3 (나); never interpolated).
- diag_compare: a computed value against the spec's printed one at the printed digits (Z.7 0b 기록 전용).
- env_diff / env_reasons: Z.9.2 P2-11 with plan Reading 1 (a file recorded by any earlier block must be unchanged)."""
from __future__ import annotations

import numpy as np

from . import y_rules
from .y_spec import SPEC as Y_SPEC

PASS, INVALID = y_rules.PASS, y_rules.INVALID
STOP_REUSE, STOP_PILOT_FEW = y_rules.STOP_REUSE, y_rules.STOP_PILOT_FEW
STOP_PLAN_UNREACHABLE = "STOP_PLAN_UNREACHABLE"
RECORDS_REASON = "OC 전 관문 STOP(Z.9.2 P3-14)"
UNUSED = "미사용"
# P3-14's rows for the order-0 STOPs: ①–⑨ (필 required · 불 unavailable · 부 partial) and the halves' status
P314 = {
    (STOP_REUSE, "0b"): ("부", "불", "불", "불", "불", "불", "불", "불"),
    (STOP_REUSE, "0c"): ("필", "불", "불", "불", "불", "불", "불", "불"),
    (STOP_PLAN_UNREACHABLE, "0c′"): ("필", "필", "불", "불", "불", "불", "불", "불"),
    (STOP_PILOT_FEW, "0d"): ("필", "필", "필", "불", "불", "불", "불", "불"),
    (STOP_PLAN_UNREACHABLE, "0d"): ("필", "필", "필", "불", "불", "불", "불", "불"),
}


def _g(x) -> str:
    return f"{float(x):g}"


def sentences(zs, ys=Y_SPEC) -> dict:
    p_set, q, K, F, (k_lo, k_hi) = zs.diag_design
    se = -zs.sens_deltas[zs.sens_cont_delta]
    sim = (f"Z 민감도 모의(Z.9.2 P1-3 — Y θ̂의 MBON13(X) 제시 드리프트 중심을 1 SE({_g(se)}) 나쁘게 옮기고 쌍 수준 배열을 "
           f"n쌍으로 타일링, 추출 {zs.sens_draws}회 × 실험 {zs.sens_reps}회, 대상 설계 p_set {_g(p_set)} · q {_g(q)} · "
           f"K {K} · F {F} · k 범위 [{k_lo}, {k_hi}], 군집 · 시나리오 최악)")
    strict = f"|d′| < {_g(ys.naive_max)} ∧ 순진 MBON13(X) ≥ {_g(ys.c_a)} ∧ 순진 MBON05(X) ≥ {_g(ys.c_p)}"
    return {
        STOP_REUSE: "Z 재사용 조건(Z.7 {where})이 깨졌다({why}). Z는 Y 코드 · Y 블록 · Y 파일럿 원자료 · 주 세트 오라클 "
                    "상세(`oracle.json`) · W · X · V 블록을 다시 재거나 고치는 경로를 갖지 않으므로 확인 반 학습 측정 "
                    "없이 멈춘다 — 사용자 몫.",
        (STOP_PLAN_UNREACHABLE, "0c′"): sim + f"에서 동시 검정력 하한이 n {zs.sens_ns[0]} {{l12}} · n {zs.sens_ns[1]} "
                                              f"{{l24}}로 {_g(zs.plan_bar)}에 못 미쳤다(2 SE {{two}}, 옮기지 않음 "
                                              f"{{zero}}) — 드리프트 중심이 Y 추정보다 1 SE 나쁘면 파일럿을 키워도 계획이 "
                                              f"G.6 한계에 닿지 않으므로, 분할 전 파일럿 측정 없이 멈춘다(주 세트 "
                                              f"{zs.n_len}쌍 모두 학습 미사용).",
        (STOP_PLAN_UNREACHABLE, "0d"): sim + f"에서 동시 검정력 하한이 키로 낸 최대 J {{j_max}} → n {{n_star}}에서 "
                                             f"{{value}}로 {_g(zs.plan_bar)}에 못 미쳤다(2 SE {{two}}, 옮기지 않음 "
                                             f"{{zero}}) — 드리프트 중심이 Y 추정보다 1 SE 나쁘면 파일럿을 키워도 계획이 "
                                             f"G.6 한계에 닿지 않으므로, 분할 뒤 파일럿 측정 없이 멈춘다(주 세트 "
                                             f"{zs.n_len}쌍 모두 학습 미사용).",
        (STOP_PILOT_FEW, "0d"): f"Z 분할(Z.3 · Z.9.2 P0-1)의 파일럿 반이 {{h}}쌍이라, 모두 입장해도 Z 파일럿 쌍 수 J가 "
                                f"Y 입장 {len(zs.y_admitted)} + {{h}} = {{j_max}}로 최소 {zs.j_min}에 못 미친다(묶은 뒤 "
                                f"최대 n_Σ {{n_sigma_max}}; 확인 반 {{n_conf}}쌍, 강제 묶음 {{n_forced}}쌍; 키만으로 "
                                f"계산했고 쌍별 값을 보지 않았다) — 파일럿 측정 없이 멈춘다(주 세트 {zs.n_len}쌍 모두 "
                                f"학습 미사용).",
        "strict": strict,
    }


def p314(outcome: str, where: str) -> dict:
    row = P314[(outcome, where)]
    cols = ("ydiag", "sens", "split", "pilot", "precheck_records", "oc_records", "gates", "learn_records")
    return dict(outputs=dict(zip(cols, row)), halves=dict(pilot=UNUSED, confirm=UNUSED))


def reuse_stop(why: list, where: str, zs) -> dict:
    return dict(outcome=STOP_REUSE, where=where, reasons=list(why),
                sentence=sentences(zs)[STOP_REUSE].format(where=where, why="; ".join(why)),
                records_unavailable=True, records_reason=RECORDS_REASON, p314=p314(STOP_REUSE, where))


def _ok(value: float, zs, ys=Y_SPEC) -> bool:
    return round(float(value) - zs.plan_bar, ys.round_digits) >= 0


def _f(x) -> str:
    return f"{float(x):.3f}"


def plan_rule_a(L: dict, zs) -> dict:
    """Continue rule (가) (0c′): L = {(δ index, n): limit}. Both n at δ −1 SE below the bar → STOP_PLAN_UNREACHABLE."""
    di = zs.sens_cont_delta
    n12, n24 = zs.sens_ns
    hit = not _ok(L[(di, n12)], zs) and not _ok(L[(di, n24)], zs)
    if not hit:
        return dict(outcome=PASS, reasons=[])
    two = " · ".join(_f(L[(2, n)]) for n in zs.sens_ns)
    zero = " · ".join(_f(L[(0, n)]) for n in zs.sens_ns)
    s = sentences(zs)[(STOP_PLAN_UNREACHABLE, "0c′")].format(l12=_f(L[(di, n12)]), l24=_f(L[(di, n24)]), two=two,
                                                             zero=zero)
    return dict(outcome=STOP_PLAN_UNREACHABLE, where="0c′", reasons=["계속 규칙 (가)"], sentence=s,
                records_unavailable=True, records_reason=RECORDS_REASON, p314=p314(STOP_PLAN_UNREACHABLE, "0c′"))


def n_star(j_max: int, zs) -> int:
    lo, hi = zs.n_star
    return hi if j_max >= hi else lo


def key_precheck(pilot_keys: list, zs) -> dict:
    """Z.9.2 P0-1's key pre-check from KEYS ONLY: h, J_max = |Y admitted| + h, n_Σ^max (Y admitted ∪ P merged)."""
    h = len(pilot_keys)
    j_max = len(zs.y_admitted) + h
    n_sig = len(y_rules.merge_groups(list(zs.y_admitted) + list(pilot_keys)))
    return dict(h=h, j_max=j_max, n_sigma_max=n_sig)


def pilot_few_0d(pre: dict, n_conf: int, n_forced: int, zs) -> dict | None:
    if pre["j_max"] >= zs.j_min:
        return None
    s = sentences(zs)[(STOP_PILOT_FEW, "0d")].format(h=pre["h"], j_max=pre["j_max"], n_sigma_max=pre["n_sigma_max"],
                                                     n_conf=n_conf, n_forced=n_forced)
    return dict(outcome=STOP_PILOT_FEW, where="0d", reasons=[f"J_max {pre['j_max']} < {zs.j_min}"], sentence=s,
                records_unavailable=True, records_reason=RECORDS_REASON, p314=p314(STOP_PILOT_FEW, "0d"))


def plan_rule_b(L: dict, j_max: int, zs) -> dict:
    """Continue rule (나) (0d, after the key pre-check): L(−1 SE, n*) below the bar → STOP_PLAN_UNREACHABLE〈0d〉."""
    di, ns = zs.sens_cont_delta, n_star(j_max, zs)
    v = L[(di, ns)]
    if _ok(v, zs):
        return dict(outcome=PASS, reasons=[], n_star=ns, value=v)
    two = " · ".join(_f(L[(2, n)]) for n in zs.sens_ns)
    zero = " · ".join(_f(L[(0, n)]) for n in zs.sens_ns)
    s = sentences(zs)[(STOP_PLAN_UNREACHABLE, "0d")].format(j_max=j_max, n_star=ns, value=_f(v), two=two, zero=zero)
    return dict(outcome=STOP_PLAN_UNREACHABLE, where="0d", reasons=["계속 규칙 (나)"], sentence=s, n_star=ns, value=v,
                records_unavailable=True, records_reason=RECORDS_REASON, p314=p314(STOP_PLAN_UNREACHABLE, "0d"))


def diag_compare(computed: dict, zs) -> dict:
    """Z.7 0b 기록 전용: each printed (name, value, digits) against the computed value rounded to the digits; a missing
    computed value (records cap, a failed item) is not equal and says so."""
    rows, diff = [], []
    for name, want, digits in zs.printed:
        got = computed.get(name)
        if got is None:
            rows.append(dict(name=name, printed=want, computed=None, equal=False, note="계산 없음"))
            diff.append(name)
            continue
        r = float(np.round(float(got), digits))
        eq = r == float(np.round(float(want), digits))
        rows.append(dict(name=name, printed=want, computed=float(got), rounded=r, equal=bool(eq)))
        if not eq:
            diff.append(name)
    return dict(rows=rows, differ=diff, n=len(rows), n_equal=len(rows) - len(diff))


def env_diff(old: dict, new: dict) -> list:
    """Fields of two env records that differ; for the per-file maps only files present in OLD are compared (a file
    added later is not a difference — plan Reading 1), a file of OLD missing in NEW is."""
    out = []
    for k in sorted(old):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict):
            b = b or {}
            out += [f"{k}:{f}" for f in sorted(a) if a[f] != b.get(f)]
        elif a != b:
            out.append(k)
    return out


def env_reasons(blocks: list, now: dict) -> list:
    """Z.9.2 P2-11: `now` against the env of every committed block in order (stage0 first)."""
    out = []
    for name, env in blocks:
        out += [f"{name} 대비 {x}" for x in env_diff(env or {}, now)]
    return sorted(set(out))
