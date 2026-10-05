"""0p's filters and count table (Y.3.2 · Y.3.3 · Y.7 0p-c), the digest / key checks and Y.8's sentences verbatim."""
import dataclasses
import math
import re
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import y_rules as R
from flymon.brain.y_spec import SPEC as Y


def _p(d, a, q, testable=True, axis="b", value=True, failure=False):
    return dict(axis=axis, value=value, failure=failure, testable=testable, d_pre=d, L_A=a, L_P=q)


def test_columns():
    assert R.COLUMNS == ("testable", "balanced", "y_strict", "y_lenient", "f2_0", "f2_25", "user31",
                         "user_onesided31")


def test_boundaries():
    assert R.passes(_p(0.5, 50, 200), Y)["balanced"] is False
    assert R.passes(_p(-0.4999, 50, 200), Y)["balanced"] is True
    s = R.passes(_p(0.1, 20.0, 43.0), Y)
    assert s["y_strict"] and s["y_lenient"]
    assert not R.passes(_p(0.1, 19.5, 43.0), Y)["y_strict"]
    assert not R.passes(_p(0.1, 20.0, 42.5), Y)["y_strict"]
    assert R.passes(_p(0.9, 16.0, 34.4), Y)["y_lenient"]
    assert not R.passes(_p(0.9999999999, 30, 100), Y)["y_lenient"]        # rounds to 1.0
    assert not R.passes(_p(0.9, 15.5, 100), Y)["y_lenient"]
    assert R.passes(_p(-0.99, 31.0, 0.0), Y)["user_onesided31"]
    assert not R.passes(_p(-1.0, 31.0, 0.0), Y)["user_onesided31"]
    assert R.passes(_p(5.0, 31.0, 0.0), Y)["user_onesided31"] and not R.passes(_p(5.0, 31.0, 0.0), Y)["user31"]
    assert R.passes(_p(0.0, 9.0, 52.0), Y)["f2_0"] and not R.passes(_p(0.0, 8.5, 52.0), Y)["f2_0"]
    assert R.passes(_p(0.0, 31.0, 94.5), Y)["f2_25"] and not R.passes(_p(0.0, 31.0, 94.0), Y)["f2_25"]


def test_not_testable_infinite_and_missing_count_nothing():
    assert not any(R.passes(_p(0.0, 50, 200, testable=False), Y).values())
    assert not any(R.passes(_p(None, None, None, testable=False, value=False), Y).values())
    s = R.passes(_p(math.inf, 50, 200), Y)
    assert s["testable"] and not s["balanced"] and not s["y_lenient"]
    assert not R.passes(_p(-math.inf, 50, 200), Y)["balanced"]


def test_count_table_rows_and_columns():
    per = [_p(0.1, 25, 50, axis="b"), _p(0.7, 17, 35, axis="a"), _p(0.1, 25, 50, testable=False, axis="a"),
           _p(None, None, None, testable=False, value=False, axis="b"), _p(0.0, 40, 100, axis="b", failure=True)]
    t = R.count_table(per, Y)
    assert set(t) == {"all", "b", "a"}
    assert t["all"]["n"] == 5 and t["b"]["n"] == 3 and t["a"]["n"] == 2
    assert t["all"]["no_value"] == 1 and t["all"]["failures"] == 1
    assert t["all"]["testable"] == 3 and t["all"]["balanced"] == 2 and t["all"]["y_strict"] == 2
    assert t["all"]["y_lenient"] == 3 and t["a"]["y_lenient"] == 1
    assert all(t["all"][c] == t["b"][c] + t["a"][c] for c in R.COLUMNS + ("n", "no_value", "failures"))


def _tab(lenient, failures=0):
    return {"all": dict({c: 0 for c in R.COLUMNS}, n=249, no_value=0, failures=failures, y_lenient=lenient)}


def test_oracle_decision_min_k_and_yield_rule():
    """Y.9.2 P1-7: [4, 8] needs N_len ≥ 6, [6, 10] ≥ 9; both dropped -> early STOP_FEW_PAIRS (Y.8's sentence below 4,
    P1-7's sentence for 4–5)."""
    d = R.oracle_decision(_tab(3), Y)
    assert d["outcome"] == R.STOP_FEW_PAIRS and d["stage"] == "early" and d["rule"] == "early"
    assert d["records_unavailable"] is True
    assert d["sentence"] == ("Y 주 세트 249쌍에서 오라클 사전 거름(시험 가능 ∧ |d_pre| < 1.0 ∧ 순진 MBON13(X) ≥ 16 ∧ "
                             "순진 MBON05(X) ≥ 34.4)을 통과한 쌍이 3개로 최소 관문 쌍 수 4에 못 미쳤다 — 주 세트 "
                             "학습 측정 없이 멈춘다.")
    for n_len in (4, 5):
        d = R.oracle_decision(_tab(n_len), Y)
        assert d["outcome"] == R.STOP_FEW_PAIRS and d["stage"] == "early" and d["rule"] == "yield"
        assert d["sentence"] == (f"Y 주 세트 249쌍에서 오라클 사전 거름(시험 가능 ∧ |d_pre| < 1.0 ∧ 순진 MBON13(X) ≥ 16 "
                                 f"∧ 순진 MBON05(X) ≥ 34.4)을 통과한 쌍이 {n_len}개로, 수율 규칙(Y.9.2 P1-7, c 1.5)이 "
                                 f"요구하는 k 범위 [4, 8]의 6개 · [6, 10]의 9개에 모두 못 미쳤다 — 주 세트 학습 측정 "
                                 f"없이 멈춘다.")
    for n_len, kept in ((6, [[4, 8]]), (8, [[4, 8]]), (9, [[4, 8], [6, 10]]), (249, [[4, 8], [6, 10]])):
        d = R.oracle_decision(_tab(n_len), Y)
        assert d["outcome"] == R.PASS and d["k_ranges"] == kept, n_len
    assert R.oracle_decision(_tab(0, failures=1), Y)["outcome"] == R.INVALID
    assert R.oracle_decision(_tab(9, failures=1), Y)["outcome"] == R.INVALID


def test_yield_rule_record():
    y = R.yield_rule(7, Y)
    assert y["n_len"] == 7 and y["c"] == 1.5 and y["kept"] == [[4, 8]] and y["dropped"] == [[6, 10]]
    assert y["required"] == [dict(k_range=[4, 8], min_n_len=6), dict(k_range=[6, 10], min_n_len=9)]
    assert y["c_basis"]["w_pilot_pairs"] == 16 and len(y["c_basis"]["oracle_lenient"]) == 3
    assert y["c_basis"]["pilot_strict"] == ["a|4 Rock Slide|Strength", "b|10 Surf vs Slowbro"]
    assert y["c_basis"]["rate"] == "2/3" and "거친 값" in y["note"]


def test_lenient_levels_quantiles():
    per = [_p(0.1, 20, 40), _p(0.2, 30, 50), _p(0.3, 40, 60), _p(0.4, 50, 70), _p(0.5, 60, 80),
           _p(0.2, 15, 300), _p(0.1, 99, 99, testable=False)]          # the last two fail the lenient filter
    lv = R.lenient_levels(per, Y)
    assert lv["n"] == 5 and lv["quantiles"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert lv["A"] == [20.0, 30.0, 40.0, 50.0, 60.0] and lv["P"] == [40.0, 50.0, 60.0, 70.0, 80.0]
    assert R.lenient_levels(per[5:], Y) == dict(n=0, quantiles=[0.0, 0.25, 0.5, 0.75, 1.0], A=None, P=None)
    d = R.derive(per, Y)
    assert set(d) == {"counts", "yield_rule", "lenient_levels"} and d["yield_rule"]["n_len"] == 5
    assert d["counts"] == R.count_table(per, Y) and d["lenient_levels"] == lv


def test_reuse_stop_sentence():
    d = R.reuse_stop(["W 측정 키 x ≠ y"], "0p-b")
    assert d["outcome"] == R.STOP_REUSE and d["records_unavailable"] is True
    assert d["sentence"] == ("Y 재사용 조건(Y.7 0p-b)이 깨졌다(W 측정 키 x ≠ y). Y는 W 파일럿 · 경로, X 블록, V 블록, "
                             "주 세트 생성원을 다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 없이 멈춘다 "
                             "— 사용자 몫.")


def test_key_and_digest_reasons():
    keys = dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key)
    assert R.key_reasons(keys, Y) == []
    assert len(R.key_reasons(dict(keys, w_measure_key="x"), Y)) == 1
    assert len(R.key_reasons(dict(keys, u_measure_key="x"), Y)) == 1
    js = dict(digest_keys=Y.main_digest_keys, n_b=167, n_a=82, last_turn=1967)
    assert R.digest_reasons(js, Y) == []
    for k, v in (("digest_keys", "0" * 64), ("n_b", 166), ("n_a", 83), ("last_turn", 1985)):
        assert len(R.digest_reasons(dict(js, **{k: v}), Y)) == 1, k


def test_levels_are_medians_of_odour_x():
    rep = {"pre": {"A": [[20, 99], [21, 0], [19, 0], [22, 0], [18, 0], [20, 0], [21, 0], [23, 0]],
                   "P": [[40, 0], [44, 0], [43, 0], [42, 0], [45, 0], [41, 0], [43, 0], [46, 0]]}}
    assert R.levels(rep) == (float(np.median([20, 21, 19, 22, 18, 20, 21, 23])),
                             float(np.median([40, 44, 43, 42, 45, 41, 43, 46])))


def test_count_table_is_a_pure_function_of_the_oracle_json_records(tmp_path, monkeypatch):
    """Y red-team P2-11: bit identity binds oracle.json only; the count table is recomputed from its records."""
    import copy
    import json

    from flymon.brain import y_store as YS
    from flymon.brain.config import Params
    per = [_p(0.1, 25, 50, axis="b"), _p(0.7, 17, 35, axis="a"), _p(-0.3, 40.5, 60.5, axis="b"),
           _p(None, None, None, testable=False, value=False, axis="a"), _p(math.inf, 30, 90, axis="b")]
    before = copy.deepcopy(per)
    t = R.count_table(per, Y)
    assert per == before                                                  # input not mutated
    assert R.count_table(per, Y) == t and R.count_table(per[::-1], Y) == t   # no state, order-free
    monkeypatch.chdir(tmp_path)
    YS.write_json(Y.oracle_detail, {"pairs": per}, [Params()])
    back = json.loads((tmp_path / Y.oracle_detail).read_text())["pairs"]
    assert R.count_table(back, Y) == t                                    # derived from oracle.json alone
    assert R.derive(back, Y) == R.derive(per, Y) and per == before


# ---- Y.8 / Y.9.2 sentences against the spec text ---------------------------------------------------------------------
SPEC_MD = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/2026-09-14-flymon-design.md"


def _y_section() -> str:
    t = SPEC_MD.read_text()
    return t[t.index("## 부록 Y."):]


def _y8(label: str) -> str:
    """The quoted sentence of Y.8's bullet starting with `label`."""
    sec = _y_section()
    sec = sec[sec.index("### Y.8 "):sec.index("### Y.9 ")]
    line = next(ln for ln in sec.splitlines() if ln.startswith(f"- **{label}"))
    return re.search(r'"([^"]*)"', line).group(1)


def _y92_yield() -> str:
    sec = _y_section()
    sec = sec[sec.index("#### Y.9.2 "):]
    line = next(ln for ln in sec.splitlines() if ln.startswith("- **둘 다 빠지면**"))
    return re.search(r'문장: "([^"]*)"', line).group(1)


FILLS = {
    (R.STOP_FEW_PAIRS, "early"): dict(n="〈249〉", k="〈k〉"),
    (R.STOP_FEW_PAIRS, "yield"): dict(n="〈249〉", k="〈N_len〉"),
    R.STOP_REUSE: dict(where="〈0p-b / 1〉", why="〈깨진 항목〉"),
    R.STOP_PILOT_FEW: dict(k="〈k〉", n_sigma="〈n_Σ〉", per_pair="〈쌍별 d′ · MBON13(X) · MBON05(X)〉"),
    R.PASS: dict(k="〈k〉", share="〈p_set < 1.0이면 'p_set 〈p〉 이상', 1.0이면 '전부'〉", n="〈n〉", m="〈m〉",
                 ratio="〈m/n〉", p="〈p〉", q="〈q〉", K="〈K〉", F="〈F〉", k_lo="〈k_lo〉", k_hi="〈k_hi〉",
                 fails="〈FAIL 쌍과 걸린 관문, 없으면 '없음'〉"),
    R.FAIL: dict(n="〈n〉", m="〈m〉", ratio="〈m/n〉",
                 why="〈'm/n 〈값〉 < p_set 〈p〉' / 'PASS 쌍 〈m〉 < 3' — 둘 다면 둘 다〉",
                 fails="〈FAIL 쌍과 걸린 관문〉", n_mech="〈n_mech〉"),
}
SPEC_TEXT = {
    (R.STOP_FEW_PAIRS, "early"): lambda: _y8("`STOP_FEW_PAIRS`(이른 멈춤, 0p-c)"),
    (R.STOP_FEW_PAIRS, "yield"): _y92_yield,
    R.STOP_REUSE: lambda: _y8("`STOP_REUSE`"),
    R.STOP_PILOT_FEW: lambda: _y8("`STOP_PILOT_FEW`"),
    R.PASS: lambda: _y8("`PASS`"),
    R.FAIL: lambda: _y8("`FAIL`"),
}


@pytest.mark.parametrize("key", list(SPEC_TEXT), ids=str)
def test_sentences_match_the_spec_verbatim(key):
    """Y.8 as amended in place by Y.9.2 (STOP_PILOT_FEW, PASS, FAIL) and Y.9.2 P1-7's yield STOP: the template filled
    with the spec's own 〈…〉 strings is the spec's sentence, character for character."""
    assert R.SENTENCES[key].format(**FILLS[key]) == SPEC_TEXT[key]()


def test_sentence_numbers_are_rebuilt_from_the_spec_object():
    """Task 1 review M1: every number in a sentence comes from YSpec — rebuilt from SPEC it is SENTENCES; changed
    numbers show up and the old ones vanish."""
    assert R.sentences(Y) == R.SENTENCES
    ys = dataclasses.replace(Y, lenient_d=1.25, lenient_a=17.5, lenient_p=36.0, naive_max=0.4, c_a=21.0, c_p=44.0,
                             yield_c=2.0, k_ranges=((3, 7), (5, 9)), pilot_w=2, pilot_v=6, pilot_min_sigma=4,
                             pool_turns=(300, 2000), lever_text="LEVER")
    s = R.sentences(ys)
    pre = "시험 가능 ∧ |d_pre| < 1.25 ∧ 순진 MBON13(X) ≥ 17.5 ∧ 순진 MBON05(X) ≥ 36"
    assert pre in s[(R.STOP_FEW_PAIRS, "early")] and "최소 관문 쌍 수 3에" in s[(R.STOP_FEW_PAIRS, "early")]
    assert "c 2)이 요구하는 k 범위 [3, 7]의 6개 · [5, 9]의 10개에" in s[(R.STOP_FEW_PAIRS, "yield")]
    assert "후보 8쌍(W 균형 2 · V 세트 후보 6)" in s[R.STOP_PILOT_FEW] and "최소 4에" in s[R.STOP_PILOT_FEW]
    assert "|d′| < 0.4 ∧ 순진 MBON13(X) ≥ 21 ∧ 순진 MBON05(X) ≥ 44" in s[R.STOP_PILOT_FEW]
    assert "모델(LEVER)" in s[R.PASS] and "생성원 턴 300–2000" in s[R.PASS] and "MBON05(X) ≥ 44)" in s[R.PASS]
    assert "|d_pre| < 1.25 ∧ MBON13(X) ≥ 17.5 ∧ MBON05(X) ≥ 36" in s[R.FAIL]
    for k in s:
        for old in ("34.4", "1985", "c 1.5", "MBON13(X) ≥ 16", "≥ 43"):
            assert old not in s[k], (k, old)
    d = R.oracle_decision(_tab(5), ys)                                     # needs 6 / 10 under the replaced spec
    assert d["outcome"] == R.STOP_FEW_PAIRS and d["rule"] == "yield" and "[3, 7]의 6개" in d["sentence"]
    assert R.oracle_decision(_tab(2), ys)["rule"] == "early" and R.oracle_decision(_tab(6), ys)["k_ranges"] == [[3, 7]]
