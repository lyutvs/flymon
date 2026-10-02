"""Q.4 / Q.6 rules as pure functions (Readings 7-13, Q.6.9): AUC = P(F < S) with ties 1/2, Spearman None on constants,
Wilcoxon p = 1 on all-zero differences; ⑤ needs Q0 and the baseline; F/S disagreement > 5 downgrades only a 일치; ①'s
weak rule wins, its Q0-only 불일치 survives an INVALID (c), its 일치 needs rho >= +0.4 over naive P_X > 0 pairs only
(Q.6.9, lowered s); ③ reads the baseline only and overlaps the floor."""
import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from flymon.brain import q_rules as R
from flymon.brain.q_spec import SPEC
from tests.brain.q_fixtures import KEYS

ROOT = Path(__file__).resolve().parents[2]
F, S = KEYS[:12], KEYS[12:]


def V(**o):
    d = dict(naive_px=20.0, r=1.0, r_P=1.0, dV_mean=0.5, dV_sd=0.5, ratio=0.3, dpx_fixed={"0.8": -6.0, "1.0": -6.5},
             px_after_fixed={"0.8": 14.0, "1.0": 13.5}, W_X=1.0)
    d.update(o)
    return d


def table(fn):
    return {k: fn(i, k) for i, k in enumerate(KEYS)}


STABLE = dict(disagree=0, agree_rate=1.0, unstable=False, auc_blocked=False)
BLOCKED = dict(disagree=6, agree_rate=15 / 21, unstable=False, auc_blocked=True)


def test_q_rules_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/q_rules.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 0.5, 1, 2}, nums


def test_helpers():
    assert R.auc_lower([1, 2], [2, 3]) == pytest.approx(0.875)
    assert R.auc_lower([], [1]) is None
    assert R.spearman([1, 2, 3], [5, 5, 5]) is None and R.spearman([1, 2, 3], [1, 2, 4]) == pytest.approx(1.0)
    assert R.spearman([1], [1]) is None
    assert R.wilcoxon_p([0.0] * 12) == 1.0 and R.wilcoxon_p([0.1 * (i + 1) for i in range(12)]) < 0.01
    assert R.med([None, 1.0, 3.0]) == 2.0 and R.med([]) is None


def test_fs_stability_counts_class_changes():
    b_r = {k: (3.0 if k in S else 1.0) for k in KEYS}
    for k in F[:6]:
        b_r[k] = 3.0
    st = R.fs_stability(F, S, b_r, SPEC)
    assert st["disagree"] == 6 and st["auc_blocked"] and not st["unstable"]           # agreement 15/21 = 0.714
    for k in F[6:7]:
        b_r[k] = 3.0
    assert R.fs_stability(F, S, b_r, SPEC)["unstable"]                                 # 14/21 < 0.7


def test_noise_floor_is_median_abs_rp_difference():
    V0 = table(lambda i, k: V(r_P=1.0))
    B = table(lambda i, k: V(r_P=1.0 + (0.3 if i % 2 else 0.1)))     # 11 x 0.1, 10 x 0.3 -> median 0.1
    assert R.noise_floor(V0, B) == pytest.approx(0.1)


def _var_tables(f_mean, f_sd):
    return table(lambda i, k: V(dV_mean=f_mean if k in F else 1.0, dV_sd=f_sd if k in F else 0.4))


def test_variation_needs_both_and_downgrade_only_match():
    match, mism, mid = _var_tables(0.9, 0.7), _var_tables(0.3, 0.7), _var_tables(0.6, 0.4)
    assert R.rule_variation(match, match, F, S, STABLE, SPEC)["label"] == R.MATCH
    assert R.rule_variation(match, mid, F, S, STABLE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_variation(mism, mism, F, S, STABLE, SPEC)["label"] == R.MISMATCH
    assert R.rule_variation(match, match, F, S, BLOCKED, SPEC)["label"] == R.UNDECIDED
    assert R.rule_variation(mism, mism, F, S, BLOCKED, SPEC)["label"] == R.MISMATCH


def _floor_tables(low_f=True, rho_sign=+1):
    V0 = table(lambda i, k: V(naive_px=(2.0 + i if k in F else 30.0 + i) if low_f else 20.0 + (i % 3)))
    B = table(lambda i, k: V(naive_px=V0[k]["naive_px"], r_P=1.0))
    C = table(lambda i, k: V(r_P=1.0 + rho_sign * 0.1 * V0[k]["naive_px"]))
    return V0, B, C


STRONG, WEAK = dict(s=0.7, weak=False), dict(s=0.9, weak=True)      # Q.6.9: (c) lowers s; |0.9 - 1| < 0.15 is weak


def test_floor_match_mismatch_and_ordering():
    V0, B, C = _floor_tables()
    got = R.rule_floor(V0, B, F, S, STABLE, C, STRONG, SPEC)
    assert got["label"] == R.MATCH and got["rho"] == pytest.approx(1.0)                  # rho >= +0.4 (Q.6.9)
    assert "하향" in got["why"]                                                           # names the lowered-s (c)
    assert R.rule_floor(V0, B, F, S, STABLE, C, WEAK, SPEC)["label"] == R.UNDECIDED           # weak: fixed
    assert "하향" in R.rule_floor(V0, B, F, S, STABLE, C, WEAK, SPEC)["why"]
    assert R.rule_floor(V0, B, F, S, STABLE, None, STRONG, SPEC)["label"] == R.UNDECIDED      # (c) INVALID
    assert R.rule_floor(V0, B, F, S, BLOCKED, C, STRONG, SPEC)["label"] == R.UNDECIDED
    V0f, Bf, Cf = _floor_tables(low_f=False)
    assert R.rule_floor(V0f, Bf, F, S, STABLE, None, STRONG, SPEC)["label"] == R.MISMATCH     # Q0-only AUC
    assert R.rule_floor(V0f, Bf, F, S, BLOCKED, Cf, STRONG, SPEC)["label"] == R.MISMATCH      # 불일치 not downgraded
    assert R.rule_floor(V0f, Bf, F, S, STABLE, Cf, WEAK, SPEC)["label"] == R.UNDECIDED        # weak checked first
    V0n, Bn, Cn = _floor_tables(rho_sign=-1)
    assert R.rule_floor(V0n, Bn, F, S, STABLE, Cn, STRONG, SPEC)["label"] == R.UNDECIDED      # rho < 0 is not 일치


def test_floor_rho_excludes_pairs_at_floor_and_counts_them():
    V0, B, C = _floor_tables()
    base = R.rule_floor(V0, B, F, S, STABLE, C, STRONG, SPEC)
    assert base["already_at_floor"] == 0 and base["rho_n"] == 21
    # two F pairs silent before any edit (naive P_X = 0) with arbitrary dr_P: rho is over the other 19 pairs only
    for k, junk in zip(F[:2], (7.0, -3.0)):
        V0[k] = dict(V0[k], naive_px=0.0)
        B[k] = dict(B[k], naive_px=0.0)
        C[k] = dict(C[k], r_P=B[k]["r_P"] + junk)
    got = R.rule_floor(V0, B, F, S, STABLE, C, STRONG, SPEC)
    assert got["already_at_floor"] == 2 and got["already_at_floor_keys"] == sorted(F[:2]) and got["rho_n"] == 19
    assert got["rho"] == pytest.approx(1.0) and got["label"] == R.MATCH
    rest = [k for k in KEYS if k not in F[:2]]
    assert got["rho"] == pytest.approx(R.spearman([B[k]["naive_px"] for k in rest],
                                                  [C[k]["r_P"] - B[k]["r_P"] for k in rest]))
    # the junk values do matter when the floor pairs are let in, so the exclusion is what keeps rho
    with_floor = R.rule_floor(V0, B, F, S, STABLE, C, STRONG, dataclasses.replace(SPEC, rho_positive_only=False))
    assert with_floor["rho"] != pytest.approx(got["rho"]) and with_floor["rho_n"] == 21
    # changing the floor pairs' dr_P to anything else leaves rho and label unchanged
    for k in F[:2]:
        C[k] = dict(C[k], r_P=B[k]["r_P"] - 50.0)
    again = R.rule_floor(V0, B, F, S, STABLE, C, STRONG, SPEC)
    assert again["rho"] == pytest.approx(got["rho"]) and again["label"] == got["label"]


def _reach_tables(px_after=14.0, rho=True, f_low=True):
    def one(i, k):
        if k in F:
            d08 = -(1.0 + 0.1 * i) if f_low else -(20.0 + 0.1 * i)          # F low (match) or F high (mismatch)
        else:
            d08 = -(8.0 + 0.1 * i)
        return V(naive_px=20.0, ratio=abs(d08) / 20.0, dpx_fixed={"0.8": d08, "1.0": d08 - 0.05},
                 px_after_fixed={"0.8": px_after, "1.0": px_after - 0.1}, W_X=(abs(d08) if rho else 1.0 + (i % 2)))
    return table(one)


def test_reach_rules():
    assert R.rule_reach(_reach_tables(), F, S, STABLE, SPEC)["label"] == R.MATCH
    assert R.rule_reach(_reach_tables(), F, S, BLOCKED, SPEC)["label"] == R.UNDECIDED
    assert R.rule_reach(_reach_tables(px_after=3.0), F, S, STABLE, SPEC)["label"] == R.UNDECIDED     # overlaps ①
    assert R.rule_reach(_reach_tables(rho=False), F, S, STABLE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_reach(_reach_tables(f_low=False), F, S, STABLE, SPEC)["label"] == R.MISMATCH
    assert R.rule_reach(_reach_tables(f_low=False), F, S, BLOCKED, SPEC)["label"] == R.MISMATCH


def test_reach_reads_the_baseline_table_only():
    # the signature admits one value table (the baseline); there is no Q0 / edited-condition table to read W_X from
    assert list(inspect.signature(R.rule_reach).parameters) == ["B", "F", "S", "stab", "spec"]
    B, other = _reach_tables(), _reach_tables(rho=False)        # identical except W_X
    assert all({**B[k], "W_X": 0} == {**other[k], "W_X": 0} for k in KEYS)
    got = R.rule_reach(B, F, S, STABLE, SPEC)
    assert got["label"] == R.MATCH and got["W_X"] == {k: B[k]["W_X"] for k in KEYS}
    assert got["f_X"] == {k: None for k in KEYS}
    # the W_X that decides is the passed (baseline) table's: the same rows with the other W_X change the label
    assert R.rule_reach(other, F, S, STABLE, SPEC)["label"] == R.UNDECIDED


def test_reach_drops_undefined_ratios_and_counts_them():
    B = _reach_tables()
    for k in F[:2]:
        B[k] = dict(B[k], ratio=None)
    rec = R.rule_reach(B, F, S, STABLE, SPEC)
    assert rec["n_ratio_undefined"] == 2 and rec["label"] == R.MATCH


# ================================================================ part 2: ② ④, sensitivity, assembly (Task 7)
NOISE = 0.1


def _apl(delta_fn, ratio_up=0.1):
    B = table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0, ratio=0.3))
    C = table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0 + delta_fn(i, k), ratio=0.3 + ratio_up))
    return C, B


def test_apl_match_guard_and_mismatch():
    C, B = _apl(lambda i, k: 1.0 + 0.1 * i)
    assert R.rule_apl(C, B, F, NOISE, SPEC)["label"] == R.MATCH
    C, B = _apl(lambda i, k: (2.0 + 0.1 * i) if i < 6 else 0.01 * (i + 1))         # rise only in low-naive F pairs
    rec = R.rule_apl(C, B, F, NOISE, SPEC)
    assert rec["label"] == R.UNDECIDED and rec["guard_pairs"] == 0
    C, B = _apl(lambda i, k: -0.2 - 0.01 * i)
    assert R.rule_apl(C, B, F, NOISE, SPEC)["label"] == R.MISMATCH                  # increase only: a fall is 불일치
    C, B = _apl(lambda i, k: 1.0 + 0.1 * i, ratio_up=-0.1)                         # ratio must also rise
    assert R.rule_apl(C, B, F, NOISE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_apl(None, B, F, NOISE, SPEC)["label"] == R.UNDECIDED


def test_threshold_uses_the_noise_floor():
    C, B = _apl(lambda i, k: 0.7)
    assert R.paired_shift(C, B, F, 0.4, SPEC)["threshold"] == pytest.approx(0.8)
    assert R.paired_shift(C, B, F, 0.1, SPEC)["threshold"] == pytest.approx(0.5)


def _mv(sign):
    B = table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0))
    return table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0 + sign * (1.0 + 0.1 * i))), B


def test_operating_directional_overlap():
    up, B = _mv(+1)                     # Δr_P rises with naive P_X: ρ = +1
    rec = R.rule_operating(up, None, B, F, NOISE, SPEC)
    assert rec["label"] == R.UNDECIDED and rec["per"]["mv_lo"]["overlap"]           # lo: ρ ≥ +0.4 overlaps
    rec = R.rule_operating(None, up, B, F, NOISE, SPEC)
    assert rec["label"] == R.MATCH and not rec["per"]["mv_hi"]["overlap"]           # hi overlaps only for ρ ≤ −0.4
    assert rec["per"]["mv_hi"]["direction"] == "증가"
    down, B = _mv(-1)                   # ρ = −1, median Δ negative: magnitude counts (two-sided, Q.6.9)
    assert R.rule_operating(None, down, B, F, NOISE, SPEC)["label"] == R.UNDECIDED
    rec = R.rule_operating(down, None, B, F, NOISE, SPEC)
    assert rec["label"] == R.MATCH and rec["per"]["mv_lo"]["direction"] == "감소"
    assert SPEC.mv_limitation in rec["limitation"]


def test_operating_rho_uses_floors_pair_set():
    up, B = _mv(+1)
    for k, junk in zip(F[:2], (-9.0, -7.0)):                     # two pairs already at floor with wild dr_P
        B[k] = dict(B[k], naive_px=0.0)
        up[k] = dict(up[k], r_P=B[k]["r_P"] + junk)
    per = R.rule_operating(up, None, B, F, NOISE, SPEC)["per"]["mv_lo"]
    assert per["rho_n"] == 19 and per["rho"] == pytest.approx(1.0) and per["overlap"]
    per_all = R.rule_operating(up, None, B, F, NOISE, dataclasses.replace(SPEC, rho_positive_only=False))["per"]
    assert per_all["mv_lo"]["rho_n"] == 21 and per_all["mv_lo"]["rho"] != pytest.approx(1.0)


def test_operating_mismatch_needs_both_valid_and_small():
    B = table(lambda i, k: V(r_P=1.0))
    same = table(lambda i, k: V(r_P=1.0))
    assert R.rule_operating(same, same, B, F, 0.2, SPEC)["label"] == R.MISMATCH
    assert R.rule_operating(same, None, B, F, 0.2, SPEC)["label"] == R.UNDECIDED
    assert R.rule_operating(None, None, B, F, 0.2, SPEC)["why"].startswith("두 배율 모두 INVALID")


def _block(vals, status="OK"):
    return dict(status=status, vals=vals, kc_median=0.05, d6a_over_share=0.0, apl_out_median=0.1,
                naive_px_median=20.0, jaccard_median=0.04, reasons=[], condition={})


def _q0_q1():
    V0 = table(lambda i, k: V(r=(1.0 if k in F else 3.0), r_P=(2.5 if k == F[0] else 1.0)))
    V0[F[1]] = dict(V0[F[1]], r=1.9)
    q0 = dict(status="OK", vals=V0, split=dict(F=F, S=S), borderline=[F[1]], cancel=[F[0]])
    q1 = {n: _block(table(lambda i, k: V(r=(1.0 if k in F else 3.0)))) for n in SPEC.cond_names}
    return q0, q1


def test_assemble_shapes_sensitivity_and_cancel():
    q0, q1 = _q0_q1()
    q1["mv_hi"] = _block({}, status="INVALID")
    out = R.assemble(q0, q1, dict(s=0.7, weak=False), SPEC)
    assert set(out["candidates"]) == {"variation", "floor", "apl", "reach", "operating"}
    assert all(c["label"] in (R.MATCH, R.MISMATCH, R.UNDECIDED) for c in out["candidates"].values())
    assert set(out["sensitivity_borderline"]) == set(out["candidates"]) == set(out["sensitivity_cancel"])
    assert out["cancel"]["q0"] == [F[0]] and out["records"]["apl_nonkc"]["role"] == "record"
    assert out["fs"]["disagree"] == 0 and len(out["sentences"]) == 5
    assert out["candidates"]["operating"]["per"]["mv_hi"]["valid"] is False
    assert SPEC.mv_limitation in out["sentences"][3]
    assert "하향" in out["records"]["s_up_note"] and out["records"]["s_c"]["s"] == 0.7


def test_assemble_recomputes_weak_from_s():
    q0, q1 = _q0_q1()
    out = R.assemble(q0, q1, dict(s=0.9, weak=False), SPEC)         # |0.9 - 1| < 0.15: weak whatever the flag says
    assert out["s_c"]["weak"] is True and out["s_c"]["weak_in"] is False
    assert out["candidates"]["floor"]["label"] == R.UNDECIDED and SPEC.weak_note in out["candidates"]["floor"]["why"]
    out = R.assemble(q0, q1, dict(s=0.7, weak=True), SPEC)
    assert out["s_c"]["weak"] is False and out["candidates"]["floor"]["weak"] is False


def test_apl_nonkc_never_labels_rule_two():
    q0, q1 = _q0_q1()
    q1["apl_mbon05"] = _block(table(lambda i, k: V(r=(1.0 if k in F else 3.0), r_P=2.0 + 0.1 * i, ratio=0.5)))
    ref = R.assemble(q0, q1, dict(s=0.7, weak=False), SPEC)["candidates"]["apl"]
    assert ref["label"] == R.MATCH
    for nonkc in (_block({}, status="INVALID"),
                  _block(table(lambda i, k: V(r_P=-40.0 - i, ratio=0.0)))):
        q1["apl_nonkc"] = nonkc
        got = R.assemble(q0, q1, dict(s=0.7, weak=False), SPEC)
        assert got["candidates"]["apl"]["label"] == ref["label"]
        assert got["candidates"]["apl"]["median"] == ref["median"]
    del q1["apl_nonkc"]
    assert R.assemble(q0, q1, dict(s=0.7, weak=False), SPEC)["candidates"]["apl"]["label"] == ref["label"]


def test_assemble_with_invalid_q0_or_base():
    q0 = dict(status="INVALID", vals={}, split=None, borderline=[], cancel=[])
    out = R.assemble(q0, {n: _block({}) for n in SPEC.cond_names}, dict(s=0.7, weak=False), SPEC)
    assert all(c["label"] == R.UNDECIDED for c in out["candidates"].values())
    q0, q1 = _q0_q1()
    q1["base"] = _block({}, status="INVALID")
    out = R.assemble(q0, q1, dict(s=0.7, weak=False), SPEC)
    assert out["invalid"] == "기준선 INVALID" and all(c["label"] == R.UNDECIDED for c in out["candidates"].values())
