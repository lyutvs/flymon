import itertools
import pytest
from flymon.agent import e_rules as R
from flymon.agent.e_spec import SPEC


def test_band_order_and_full_cover():
    n_a = 32
    for n, c in itertools.product(range(22), range(22)):
        for fa in range(n_a + 1):
            b = R.read_band(n, c, fa, 21, n_a, n_a, naive_a=fa, spec=SPEC)
            assert b["band"] in (R.SELECTED, R.B_TB, R.B_FA, R.B_NC)
            if c >= 11:
                assert b == {"band": R.B_NC, "reason": R.E0_ABOVE_BAR}
            elif n <= c:
                assert b["band"] == R.B_TB
            elif n < 11:
                assert b == {"band": R.B_NC, "reason": R.BELOW_BAR}
            elif n - c < 2:
                assert b == {"band": R.B_NC, "reason": R.MARGIN}
            elif fa < 2:
                assert b["band"] == R.B_FA
            else:
                assert b["band"] == R.SELECTED


@pytest.mark.parametrize("n,c,fa,band", [(11, 10, 2, R.B_NC), (12, 10, 2, R.SELECTED), (11, 11, 5, R.B_NC),
                                         (9, 9, 5, R.B_TB), (10, 2, 5, R.B_NC), (11, 2, 1, R.B_FA)])
def test_band_edges(n, c, fa, band):
    assert R.read_band(n, c, fa, 21, 32, 32, fa, SPEC)["band"] == band


@pytest.mark.parametrize("n,c,fa,band,reason", [
    (15, 10, 5, R.SELECTED, None), (15, 11, 5, R.B_NC, R.E0_ABOVE_BAR),       # c = 10/11
    (5, 5, 5, R.B_TB, None), (6, 5, 5, R.B_NC, R.BELOW_BAR),                 # n = c / c + 1 below bar
    (12, 11, 5, R.B_NC, R.E0_ABOVE_BAR), (12, 10, 5, R.SELECTED, None),       # n = c + 2 at bar
    (11, 10, 5, R.B_NC, R.MARGIN), (10, 3, 5, R.B_NC, R.BELOW_BAR),           # n = c + 1 / n = 10
    (11, 3, 5, R.SELECTED, None), (11, 3, 1, R.B_FA, None), (11, 3, 2, R.SELECTED, None)])
def test_band_edges_spec(n, c, fa, band, reason):
    b = R.read_band(n, c, fa, 21, 32, 32, fa, SPEC)
    assert b["band"] == band
    if reason is not None:
        assert b["reason"] == reason


def test_b_fa_carries_naive():
    b = R.read_band(15, 2, 1, 21, 32, 32, 1, SPEC)
    assert b["band"] == R.B_FA and b["naive_a"] == 1 and b["f_a_possible"] is False
    assert R.read_band(15, 2, 0, 21, 32, 32, 6, SPEC)["f_a_possible"] is True


def test_not_read_on_counts():
    assert R.read_band(15, 2, 5, 20, 32, 32, 5, SPEC)["band"] == R.NOT_READ
    assert R.read_band(15, 2, 5, 21, 31, 32, 5, SPEC)["band"] == R.NOT_READ
    assert R.read_band(15, 2, 5, 22, 32, 32, 5, SPEC)["band"] == R.NOT_READ
    assert R.read_band(15, 2, 5, 21, 33, 32, 5, SPEC)["band"] == R.NOT_READ
    assert R.read_band(15, 12, 5, 20, 32, 32, 5, SPEC)["band"] == R.NOT_READ     # counts first


def test_select_pass_first():
    res = {"k3-full": dict(eligible=True, testable_b=10, F_a=3), "k3-norm": dict(eligible=True, testable_b=9, F_a=3),
           "k2-full": dict(eligible=False, testable_b=0, F_a=0), "k2-norm": dict(eligible=True, testable_b=12, F_a=2)}
    assert R.select_config(res, SPEC)["winner"] == "k2-norm"
    res["k3-norm"] = dict(eligible=True, testable_b=11, F_a=2)
    assert R.select_config(res, SPEC)["winner"] == "k3-norm"          # within 2 of 12, earlier in order
    for v in res.values(): v["testable_b"] = 10
    assert R.select_config(res, SPEC)["outcome"] == R.STOP_EVEN_LOW
    assert R.select_config({k: dict(eligible=False, testable_b=0, F_a=0) for k in SPEC.configs},
                           SPEC)["outcome"] == R.STOP_NO_ELIGIBLE


def test_select_ineligible_never_passes_and_fa_gate():
    res = {"k3-full": dict(eligible=False, testable_b=20, F_a=9), "k3-norm": dict(eligible=True, testable_b=15, F_a=1),
           "k2-full": dict(eligible=True, testable_b=12, F_a=2), "k2-norm": dict(eligible=True, testable_b=15, F_a=2)}
    out = R.select_config(res, SPEC)
    assert out["passing"] == ["k2-full", "k2-norm"] and out["near"] == ["k2-norm"] and out["winner"] == "k2-norm"
    res["k2-full"]["testable_b"] = 13                                   # 15 - 13 = 2 -> near, earlier
    assert R.select_config(res, SPEC)["winner"] == "k2-full"


def test_strength_eligible_first():
    good = dict(cap_ok=True, single=[0.06] * 32, dual=[0.06] * 80)
    near_bad = dict(cap_ok=True, single=[0.02] * 32, dual=[0.0554] * 80)
    capped = dict(cap_ok=False, single=None, dual=None)
    out = R.choose_strength({0.35: near_bad, 0.5: good, 2.0: capped}, SPEC)
    assert out["s"] == 0.5
    assert out["reasons"] == {0.35: "KC", 0.5: "OK", 2.0: "ORN_CAP"}
    assert R.choose_strength({0.35: near_bad}, SPEC)["s"] is None


def test_strength_tie_smaller_s():
    a = dict(cap_ok=True, single=[0.0504] * 32, dual=[0.0504] * 80)
    b = dict(cap_ok=True, single=[0.0604] * 32, dual=[0.0604] * 80)
    assert R.choose_strength({0.7: b, 0.5: a}, SPEC)["s"] == 0.5        # |0.0050| == |0.0050| -> smaller


def test_strength_tails_per_type():
    single = [0.06] * 28 + [0.02] * 4        # 12.5 % low -> fails
    assert not R.strength_ok(single, [0.06] * 80, SPEC)["ok"]
    single = [0.06] * 29 + [0.02] * 3        # 9.4 %
    assert R.strength_ok(single, [0.06] * 80, SPEC)["ok"]
    assert not R.strength_ok([0.06] * 32, [0.06] * 71 + [0.2] * 9, SPEC)["ok"]
    # pooled share 4/112 = 3.6 % would pass; the single-type share 12.5 % must still fail
    assert not R.strength_ok([0.06] * 28 + [0.2] * 4, [0.06] * 80, SPEC)["ok"]


def test_strength_band_inclusive():
    assert R.strength_ok([0.05] * 32, [0.05] * 80, SPEC)["ok"]
    assert R.strength_ok([0.09] * 32, [0.09] * 80, SPEC)["ok"]
    assert not R.strength_ok([0.0499] * 32, [0.0499] * 80, SPEC)["ok"]


def test_odour_activity_median():
    assert R.odour_activity({"x": [0.1, 0.3, 0.2], "y": [0.0, 1.0]}) == {"x": 0.2, "y": 0.5}


def test_sentences_fill():
    s = R.sentence(R.STOP_EVEN_LOW, dict(table="k3-full 9/21·F_a 2"))
    assert "k3-full 9/21" in s
    with pytest.raises(KeyError):
        R.sentence(R.SELECTED, {})


def test_sentences_all_outcomes():
    f = dict(reasons="r", table="t", m=19, k=3, config="k3-full", k_even=12, T=103, n=13, c=4, f_a=3, n_a=32,
             naive_a=6)
    for key in R.SENTENCES:
        assert "{" not in R.sentence(key, f)
    assert R.sentence(R.B_TB, f).startswith("선택된 설정 k3-full(4설정에서 짝수 턴으로 선택, 짝수 12/21)")
    assert "턴 64–103" in R.sentence(R.SELECTED, f) and "(13/21 대 4/21" in R.sentence(R.SELECTED, f)
    nc = R.sentence(R.B_NC, dict(f, reason=R.MARGIN))
    assert nc == R.sentence(R.B_NC + ":" + R.MARGIN, f) and "여유가 2 미만" in nc and "F_a 3/32" in nc
    with pytest.raises(KeyError):
        R.sentence("NOPE", f)


def test_oc_rows_sum_to_one():
    oc = R.operating_characteristics(32, SPEC)
    for row in oc["rows"]:
        assert abs(sum(row["P"].values()) - 1.0) < 1e-9
    assert {r["n_a"] for r in oc["rows"]} >= {18, 32}


def test_oc_row_layout():
    oc = R.operating_characteristics(32, SPEC)
    base = [r for r in oc["rows"] if r["tag"] == "base"]
    assert len(base) == 4 * 3 * 13
    delta = [r for r in oc["rows"] if r["tag"] == "c_delta"]
    assert {r["c"] for r in delta} == {0, 4, 2, 6, 5, 9}
    na = [r for r in oc["rows"] if r["tag"] == "n_a_row"]
    assert {(r["n_a"], r["naive_a"]) for r in na} == {(18, 4), (32, 7)}
    assert "4/18" in oc["assumption"] and all("4/18" in r["naive_a_rule"] for r in na)
    # naive_a = 0 -> F_a = 0 -> never SELECTED
    assert all(r["P"][R.SELECTED] == 0 for r in base if r["naive_a"] == 0)
    # c = 7, q = 0.5: P(SELECTED) equals P(n >= 11 ∧ n - c >= 2) * P(F_a >= 2), checked by hand formula
    from math import comb
    r = next(r for r in base if r["q"] == 0.5 and r["c"] == 7 and r["naive_a"] == 4)
    pn = sum(comb(21, i) for i in range(11, 22)) / 2 ** 21
    pf = 1 - (1 + 4) / 16
    assert abs(r["P"][R.SELECTED] - pn * pf) < 1e-12
