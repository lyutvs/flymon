"""0p's filters and count table (Y.3.2 · Y.3.3 · Y.7 0p-c), the digest / key checks and Y.8's sentences verbatim."""
import math

import numpy as np

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


def test_oracle_decision():
    def tab(lenient, failures=0):
        return {"all": dict({c: 0 for c in R.COLUMNS}, n=249, no_value=0, failures=failures, y_lenient=lenient)}
    assert R.oracle_decision(tab(4), Y)["outcome"] == R.PASS
    d = R.oracle_decision(tab(3), Y)
    assert d["outcome"] == R.STOP_FEW_PAIRS and d["stage"] == "early" and d["records_unavailable"] is True
    assert d["sentence"] == ("Y 주 세트 249쌍에서 오라클 사전 거름(시험 가능 ∧ |d_pre| < 1.0 ∧ 순진 MBON13(X) ≥ 16 ∧ "
                             "순진 MBON05(X) ≥ 34.4)을 통과한 쌍이 3개로 최소 관문 쌍 수 4에 못 미쳤다 — 주 세트 "
                             "학습 측정 없이 멈춘다.")
    assert R.oracle_decision(tab(0, failures=1), Y)["outcome"] == R.INVALID


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
