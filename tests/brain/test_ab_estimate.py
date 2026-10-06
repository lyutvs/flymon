"""AB.4–AB.6 (judgement side): the three statistics, the two interval methods (TS, CG), the verdict and its fixtures
(AB.7 0 (i)–(xii)), the Hedges bar, the mirror property and the record helpers. Synthetic data only."""
import math

import numpy as np
import pytest
from scipy import stats

from flymon.brain import aa_estimate as AE
from flymon.brain import ab_estimate as E
from flymon.brain import w_verdict as WV
from flymon.brain.ab_spec import SPEC, small

S = small(SPEC, boot_b=2_000)
Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
LV = dict(D=[0.001, 0.0001], Dfin=[0.001], R=[0.001])
AL = dict(D=0.001, Dfin=0.001, R=0.001, F=0.0001)


# ---------------------------------------------------------------- synthetic sets
def world(k=20, gx_n=7, gt_n=9, eff=(3.0, -3.0, 3.0, -3.0), sd=1.0, seed=0, const=None, inf_flies=None):
    """k pairs; per gate probe contrasts x = eff + u_pair + N(0, sd) [F, K]; const = {gate: c} makes every probe c;
    inf_flies = (gate, n, value): n flies of every pair constant `value` (→ ±∞ d′)."""
    r = np.random.default_rng(seed)
    keys = [f"a|{i}|X{i % gx_n}|Y" for i in range(k)]
    tsets = [f"T{(i * 5) % gt_n}" for i in range(k)]
    per = {}
    for i, kk in enumerate(keys):
        x = np.empty((S.flies, S.probes, 4))
        for g in range(4):
            x[..., g] = eff[g] + r.normal(0, 0.3) + r.normal(0, sd, (S.flies, S.probes))
            if const and g in const:
                x[..., g] = const[g]
            if inf_flies and inf_flies[0] == g:
                x[:inf_flies[1], :, g] = inf_flies[2]
        dp = WV.dprime(np.ascontiguousarray(np.moveaxis(x, -1, 1)))   # [F, 4] (contiguous: exact constant → ±∞)
        per[kk] = dict(D=AE.winsorize(dp, S), Dfin=np.where(np.isinf(dp), np.nan, AE.winsorize(dp, S)),
                       R=x.mean(1), raw=dp)
    return per, keys, tsets


def judge(per, keys, tsets, alpha=AL, machine=(), mech=None):
    lim = E.set_limits(per, keys, tsets, E.levels_for(alpha), S)
    return E.verdict(lim, list(machine), mech or {}, alpha, S)


# ---------------------------------------------------------------- numbers
def test_constants_and_hedges_bar():
    assert round(AE.c_k(8), 4) == SPEC.c7_declared and round(AE.hedges_j(8), 4) == SPEC.hedges_j_declared
    assert round(E.delta(), 6) == SPEC.delta_declared
    assert round(AE.hedges_j(8) * AE.c_k(8), 4) == 1.0008
    assert math.isclose(SPEC.bar / AE.hedges_j(8), 27 / 24)
    # the bar is compared on the Hedges scale: an uncorrected 1.10 is below the bar, 1.13 above
    assert not E.beyond(1.10, 1.0, "D", S) and E.beyond(1.13, 1.0, "D", S)
    assert E.beyond(-1.13, -1.0, "D", S) and not E.beyond(-1.10, -1.0, "D", S)
    assert E.beyond(0.26, 1.0, "R", S) and not E.beyond(0.25, 1.0, "R", S)     # strict


def test_raw4_matches_aa_raw_contrasts_and_pair_stats():
    r = np.random.default_rng(3)
    d = {st: r.integers(0, 60, (8, 8, 2, 2)) for st in WV.STAGES}
    R = E.raw4(d, Z)
    assert np.array_equal(R[:, 2:], AE.raw_contrasts(d, Z))
    r1, r2 = WV.dv(d["R1"], Z), WV.dv(d["R2"], Z)
    assert np.array_equal(R[:, 0], r1.mean(-1)) and np.array_equal(R[:, 1], (r2 - r1).mean(-1))
    p = E.pair_stats(d, Z, S)
    assert np.array_equal(p["D"], np.clip(WV.gate_stats(d, Z), -10, 10))
    d2 = {st: np.full((8, 8, 2, 2), 5) for st in WV.STAGES}
    d2["R1"] = d2["R1"].copy()
    d2["R1"][..., WV.A, WV.X] = 7                                          # constant shift → ±∞ fly d′
    p2 = E.pair_stats(d2, Z, S)
    assert np.isinf(p2["raw"][:, 0]).all() and (p2["D"][:, 0] == 10).all() and np.isnan(p2["Dfin"][:, 0]).all()


def test_winsorize_both_ways_counterexamples_x():
    """AB.6 / fixture (x): clipping moves a pair estimate either way."""
    a = AE.winsorize(np.array([4, 4, 4, 4, 4, 4, 4, -100.0]), S).mean()
    b = AE.winsorize(np.array([100, 1, 1, 1, 1, 1, -5, -5.0]), S).mean()
    assert a == 2.25 and round(a * AE.hedges_j(8), 4) == 2.0
    assert b == 0.625 and round(b * AE.hedges_j(8), 4) == 0.5556


# ---------------------------------------------------------------- CG
def test_cg_hand_example_3x3():
    gx = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    gt = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    th = np.array([0.0, 2, 4, 3, 5, 7, 6, 8, 11])                       # additive X + T: V > max(V_X, V_T)
    e = th - th.mean()
    k = 9

    def V(lab):
        G = len(set(lab.tolist()))
        return G / (G - 1) * sum(e[lab == g].sum() ** 2 for g in set(lab.tolist())) / k ** 2
    vx, vt, vi = V(gx), V(gt), V(gx * 10 + gt)
    p = E.cg_parts(th, gx, gt)
    assert math.isclose(p["V_X"], vx) and math.isclose(p["V_T"], vt) and math.isclose(p["V_XT"], vi)
    assert p["df"] == 2 and math.isclose(p["V_star"], vx + vt - vi) and vx + vt - vi > max(vx, vt)
    lo, hi = E.cg_limits(th, gx, gt, [0.025])
    h = stats.t.ppf(0.975, 2) * math.sqrt(p["V_star"])
    assert math.isclose(lo[0], th.mean() - h) and math.isclose(hi[0], th.mean() + h)


def test_cg_psd_branch_and_nan_pairs():
    gx = np.array([0, 0, 1, 1, 2, 2])
    gt = np.array([0, 1, 0, 1, 0, 1])
    th = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])                     # X sums 0 → V_X 0, V − V_XT < 0
    p = E.cg_parts(th, gx, gt)
    assert p["V"] < max(p["V_X"], p["V_T"]) and p["psd_fix"] and p["V_star"] == max(p["V_X"], p["V_T"])
    th2 = th.copy()
    th2[0] = np.nan                                                      # a D_fin pair without a finite fly
    assert E.cg_parts(th2, gx, gt)["k"] == 5


def test_ts_is_aa_two_stage_and_percentile():
    per, keys, tsets = world()
    X = np.stack([per[k]["D"] for k in keys])
    from flymon.brain import y_rules
    reps = AE.boot_two_stage(X, y_rules.merge_groups(keys), AE.stream(SPEC.boot_seed, "pool", "S1"), 2_000, 1_000)
    lim = E.set_limits(per, keys, tsets, LV, S)
    assert np.allclose(lim["D"]["ts"]["lo"][0], np.percentile(reps, 0.1, axis=0))
    assert np.allclose(lim["D"]["ts"]["hi"][1], np.percentile(reps, 99.99, axis=0))


def test_stream_vectors_are_the_sealed_values():
    for (root, *tags), want in SPEC.stream_vectors:
        r = AE.stream(SPEC.root(root), *tags)
        assert tuple(int(v) for v in r.integers(0, 2 ** 63, size=3)) == want


def test_mirror_flips_and_swaps_limits():
    per, keys, tsets = world()
    neg = {k: dict(D=-v["D"], Dfin=-v["Dfin"], R=-v["R"], raw=-v["raw"]) for k, v in per.items()}
    a = E.set_limits(per, keys, tsets, LV, S)
    b = E.set_limits(neg, keys, tsets, LV, S)
    for n in E.STATS:
        assert np.allclose(np.array(a[n]["cg"]["lo"]), -np.array(b[n]["cg"]["hi"]))
        assert np.allclose(np.array(a[n]["ts"]["lo"]), -np.array(b[n]["ts"]["hi"]))


# ---------------------------------------------------------------- the verdict fixtures (AB.7 0)
def test_fixture_vi_specific_large_learning_passes():
    assert judge(*world())["label"] == E.PASS


def test_fixture_i_drift_without_association_not_pass():
    assert judge(*world(eff=(3.0, -3.0, 0.0, -3.0)))["label"] != E.PASS


def test_fixture_ii_punish_drop_without_association_not_pass():
    assert judge(*world(eff=(3.0, -3.0, 3.0, 0.0)))["label"] != E.PASS


def test_fixture_iii_no_learning_fail_or_undecided():
    assert judge(*world(eff=(0.0, 0.0, 0.0, 0.0)))["label"] in (E.FAIL, E.UNDECIDED)


def test_fixture_iv_small_constant_sd0_not_pass():
    v = judge(*world(const={0: 0.05, 1: -0.05, 2: 0.05, 3: -0.05}))
    assert v["label"] != E.PASS
    g = v["gates"]["reward_level"]
    assert not g["R"]["ok_TS"] and not g["Dfin"]["fin_ok"]


def test_fixture_iv_prime_floor_sd_near_zero_not_pass():
    v = judge(*world(eff=(0.0, 0.0, 0.0, 0.0), sd=1e-3))
    assert v["label"] != E.PASS


def test_fixture_v_reversed_direction_fails():
    v = judge(*world(eff=(-3.0, 3.0, -3.0, 3.0)))
    assert v["label"] == E.FAIL and v["fail_gates"] and all(v["gates"][g]["fail"]["past_zero"] for g in v["fail_gates"])


def test_fixture_vii_one_gate_straddles_undecided():
    v = judge(*world(eff=(3.0, -3.0, 3.0, -1.15), sd=1.0, k=12))
    assert v["label"] == E.UNDECIDED and any("punish_assoc" in c for c in v["causes"])


def test_fixture_viii_mechanism_and_machine():
    per, keys, tsets = world()
    mech = {keys[0]: np.array([[0.74, 0.9]] * 8)}
    assert judge(per, keys, tsets, mech=mech)["label"] == E.STOP_PROTOCOL
    assert judge(per, keys, tsets, machine=["noplast R1 ≠ pre (1 spike)"], mech=mech)["label"] == E.STOP_MACHINE
    ok = {keys[0]: np.array([[0.75, 0.75]] * 8)}
    assert judge(per, keys, tsets, mech=ok)["label"] == E.PASS


def test_fixture_ix_two_inf_flies_d_passes_dfin_does_not():
    per, keys, tsets = world(eff=(0.0, -3.0, 3.0, -3.0), sd=1.0, inf_flies=(0, 2, 0.4))
    for k in keys:                                          # the other six reward-level flies exactly 0
        per[k]["D"][2:, 0] = 0.0
        per[k]["Dfin"][2:, 0] = 0.0
        per[k]["raw"][2:, 0] = 0.0
        per[k]["R"][:, 0] = 5.0                              # a large R, so only D_fin can stop it
    p = per[keys[0]]
    assert p["D"][:, 0].mean() == 2.5 and round(2.5 * AE.hedges_j(8), 4) == 2.2222
    v = judge(per, keys, tsets)
    assert v["label"] != E.PASS and v["gates"]["reward_level"]["D"]["ok_TS"]
    assert not v["gates"]["reward_level"]["Dfin"]["ok_TS"]


def test_fixture_x_pair_estimates_from_counterexamples():
    for flies, want in (([4, 4, 4, 4, 4, 4, 4, -100.0], 2.25), ([100, 1, 1, 1, 1, 1, -5, -5.0], 0.625)):
        dp = np.array(flies)[:, None] * np.ones((1, 4))
        st = E.stats_from_dprime(dp, np.zeros((8, 4)), S)
        assert st["D"][:, 0].mean() == want


def test_fixture_xi_bonferroni_fail():
    """A gate whose upper end is below the bar at α 0.025 but straddles it at α_F is not FAIL (α_P = 0.025 here, so
    the "FAIL at α_P" mutant reads the 0.025 row and would FAIL)."""
    per, keys, tsets = world(eff=(3.0, -3.0, 0.80, -3.0), sd=1.0, k=16)
    al = dict(D=0.025, Dfin=0.025, R=0.025, F=0.0001)
    lim = E.set_limits(per, keys, tsets, E.levels_for(al), S)
    j = AE.hedges_j(8)
    hi_025 = lim["D"]["ts"]["hi"][0][2] * j, lim["D"]["cg"]["hi"][0][2] * j
    hi_f = lim["D"]["ts"]["hi"][1][2] * j, lim["D"]["cg"]["hi"][1][2] * j
    assert max(hi_025) < 1 < max(hi_f), (hi_025, hi_f)
    assert E.verdict(lim, [], {}, al, S)["label"] == E.UNDECIDED


def test_fail_needs_both_methods():
    per, keys, tsets = world(eff=(-3.0, 3.0, -3.0, 3.0))
    lim = E.set_limits(per, keys, tsets, E.levels_for(AL), S)
    for g in range(4):                                                  # CG never short of the bar
        lim["D"]["cg"]["hi"][1][g] = 5.0
        lim["D"]["cg"]["lo"][1][g] = -5.0
    assert E.verdict(lim, [], {}, AL, S)["label"] == E.UNDECIDED


def test_fixture_xii_fail_side_uncalibrated_is_undecided():
    v = judge(*world(eff=(-3.0, 3.0, -3.0, 3.0)), alpha=dict(AL, F=None))
    assert v["label"] == E.UNDECIDED and E.CAUSE_FAIL_UNCAL in v["causes"]


def test_dfin_k_and_group_floor():
    per, keys, tsets = world(k=9, gx_n=5, gt_n=5, inf_flies=(0, 8, 0.4))     # every reward-level fly ±∞
    v = judge(per, keys, tsets)
    assert v["label"] != E.PASS and not v["gates"]["reward_level"]["Dfin"]["fin_ok"]


@pytest.mark.parametrize("method", ["cg", "ts"])
def test_iut_needs_every_gate_stat_method(method):
    per, keys, tsets = world()
    lim = E.set_limits(per, keys, tsets, E.levels_for(AL), S)
    lim["R"][method]["lo"][0][2] = 0.1                                 # one method of one statistic of one gate
    v = E.verdict(lim, [], {}, AL, S)
    assert v["label"] == E.UNDECIDED and any("갈림" in c for c in v["causes"])


def test_r_condition_on_all_four_gates():
    per, keys, tsets = world()
    for k in keys:
        per[k]["R"][:, 0] = 0.1                                         # reward level: tiny unstandardised contrast
    assert judge(per, keys, tsets)["label"] == E.UNDECIDED


# ---------------------------------------------------------------- records
def test_fly_counts_and_floor_subset():
    per, keys, _t = world(inf_flies=(0, 2, 0.4))
    c = E.fly_counts(per, keys, S)
    assert c["reward_level"]["n_inf"] == 2 * len(keys) and c["reward_level"]["n_flies"] == 8 * len(keys)
    assert 0.0 <= c["punish_drop"]["share_ge1"] <= 1.0
    fl = {k: dict(phi_R=0.05 if i % 2 else 0.2, phi_P=0.0) for i, k in enumerate(keys)}
    assert E.floor_subset(fl, keys, S) == [k for i, k in enumerate(keys) if i % 2]


# ---------------------------------------------------------------- review minors (T5)
def test_fail_uncal_cause_only_in_the_fail_row():
    """AB.4 판정 3: with α_F not reached, the cause "FAIL 쪽 보정 미도달" is attached only when the result would sit
    in the FAIL row (some gate's D short of the bar under both methods at f_grid[0]), never to every UNDECIDED."""
    al = dict(AL, F=None)
    lv = E.levels_for(al, S, fail_probe=True)
    assert lv["D"] == [al["D"], S.f_grid[0]] and E.levels_for(al, S) == E.levels_for(al)
    per, keys, tsets = world(eff=(-3.0, 3.0, -3.0, 3.0))                      # reversed: the FAIL row
    v = E.verdict(E.set_limits(per, keys, tsets, lv, S), [], {}, al, S)
    assert v["label"] == E.UNDECIDED and E.CAUSE_FAIL_UNCAL in v["causes"]
    assert v["fail_uncal"]["alpha"] == S.f_grid[0] and v["fail_uncal"]["in_fail_row"]
    per, keys, tsets = world(eff=(3.0, -3.0, 3.0, -1.15), k=12)               # straddles: UNDECIDED, not the FAIL row
    v = E.verdict(E.set_limits(per, keys, tsets, lv, S), [], {}, al, S)
    assert v["label"] == E.UNDECIDED and v["causes"] and E.CAUSE_FAIL_UNCAL not in v["causes"]
    assert v["fail_uncal"] == dict(alpha=S.f_grid[0], gates=[], in_fail_row=False)


@pytest.mark.parametrize("field", ["gt_fin", "gx_fin"])
def test_dfin_group_floor_blocks_pass_with_enough_pairs(field):
    per, keys, tsets = world()
    lim = E.set_limits(per, keys, tsets, E.levels_for(AL), S)
    assert E.verdict(lim, [], {}, AL, S)["label"] == E.PASS
    assert lim["Dfin"]["k_fin"][1] >= S.k_min
    lim["Dfin"][field][1] = S.g_min - 1                                     # pairs ≥ 8 but one grouping < 5
    v = E.verdict(lim, [], {}, AL, S)
    assert v["label"] == E.UNDECIDED and not v["gates"]["punish_drop"]["Dfin"]["fin_ok"]
    assert any("하한 미달" in c for c in v["causes"])
    lim["Dfin"][field][1] = S.g_min
    assert E.verdict(lim, [], {}, AL, S)["label"] == E.PASS


def test_cg_crossed_count_is_non_empty_cells_only():
    gx = np.array([0, 0, 1, 1, 2, 2, 2])
    gt = np.array([0, 1, 0, 1, 0, 0, 0])                                   # cell (2, 1) empty: 5 of 3 × 2
    th = np.array([0.3, 2.0, -1.0, 4.0, 0.5, 1.5, -2.0])
    e = th - th.mean()
    k = th.size
    cells = gx * 10 + gt
    sums = [e[cells == c].sum() for c in sorted(set(cells.tolist()))]
    p = E.cg_parts(th, gx, gt)
    assert p["G_XT"] == 5 and p["G_X"] * p["G_T"] == 6
    assert math.isclose(p["V_XT"], 5 / 4 * sum(x ** 2 for x in sums) / k ** 2)
    assert not math.isclose(p["V_XT"], 6 / 5 * sum(x ** 2 for x in sums) / k ** 2)
