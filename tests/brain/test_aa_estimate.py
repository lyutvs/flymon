"""AA.4 / AA.5 / AA.6 / AA.9 / AA.9.1 inputs on synthetic raw counts (AA.7 0 6246 추정 픽스처) and the mutants of
AA.7 0 6249 that live here: median as primary, ±∞ dropped from the primary, a clip other than ±10, a flipped sign,
per-gate resample indices, probes resampled instead of flies, a pair-level first stage, two-stage primary below 5
groups, the small-k rule skipped, φ_R / φ_P swapped."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import aa_estimate as E
from flymon.brain import w_records, w_verdict as WV
from flymon.brain.aa_spec import SPEC
from flymon.brain.w_spec import SPEC as W

Z = {"A": (16.916666666666668, 12.483878492769072), "P": (80.16666666666667, 29.775432639827233)}
S = dataclasses.replace(SPEC, boot_b=400, boot_chunk=150)        # chunk boundary inside B on purpose


def pair(seed, F=8, K=8, a=20.0, b=10.0, base=(40.0, 90.0)):
    """{stage: int [F, K, 2, 2]}: PAM08 lowers MBON05(X) at R1 (and RN1), PPL105 lowers MBON13(X) at R2."""
    r = np.random.default_rng(seed)
    out = {}
    for s in WV.STAGES:
        x = np.empty((F, K, 2, 2))
        x[..., 0, :] = base[0] + r.normal(0, 4, (F, K, 2))
        x[..., 1, :] = base[1] + r.normal(0, 6, (F, K, 2))
        if s in ("R1", "R2", "RN1", "RN2"):
            x[..., 1, 0] -= a
        if s == "R2":
            x[..., 0, 0] -= b
        out[s] = np.clip(np.rint(x), 0, None).astype(np.int64)
    out["RN1"] = out["R1"].copy()
    return out


def test_fly_dprimes_are_gate_stats_bitwise():
    d = pair(1)
    assert np.array_equal(E.fly_dprimes(d, Z), WV.gate_stats(d, Z))


def test_winsorize_inf_and_large_counts_and_mean():
    d = pair(2)
    for s in ("R1", "N1"):                                          # fly 0: reward contrast constant → ±∞
        d[s][0] = d[s][0, :1]
    d["N1"][0, :, 1, 0] = d["R1"][0, :, 1, 0] + 7                   # constant positive shift → mean > 0, sd 0
    raw = E.fly_dprimes(d, Z)
    assert np.isinf(raw[0, E.GI["reward_assoc"]])
    rec = E.pair_record(d, Z, S)["gates"]["reward_assoc"]
    w = np.clip(raw[:, E.GI["reward_assoc"]], -10, 10)
    assert rec["mean"] == pytest.approx(float(w.mean()), abs=0) and rec["n_inf"] == 1 and rec["n_clipped"] >= 1
    assert rec["fly"][0] in ("+inf", "-inf")
    assert rec["drop_mean"] == pytest.approx(float(raw[1:, E.GI["reward_assoc"]].mean()))     # record only


def test_primary_is_mean_not_median_and_sign_kept():
    d = pair(3, a=-5.0, b=-30.0)                                   # punishment raises MBON13(X): positive punish assoc
    rec = E.pair_record(d, Z, S)["gates"]
    w = np.clip(E.fly_dprimes(d, Z), -10, 10)
    for i, g in enumerate(WV.GATES):
        assert rec[g]["mean"] == float(w[:, i].mean()) and rec[g]["median"] == float(np.median(w[:, i]))
    assert np.sign(rec["punish_assoc"]["mean"]) == np.sign(w[:, 3].mean())


def test_clip_is_ten():
    assert S.winsor == 10.0
    assert np.array_equal(E.winsorize(np.array([12.0, -np.inf, 3.0]), S), [10.0, -10.0, 3.0])


def test_fewer_than_two_flies_is_null():
    d = {s: v[:1] for s, v in pair(4).items()}
    rec = E.pair_record(d, Z, S)["gates"]["reward_assoc"]
    assert rec["mean"] is None and rec["t_ci"] is None


def test_nan_fly_raises():
    d = {s: v[:, :1] for s, v in pair(5).items()}                  # one probe → d′ NaN
    with pytest.raises(ValueError):
        E.pair_arrays(d, Z, S)


def test_floor_shares_match_pilot_record_taught():
    d = pair(6)
    d["R1"][..., 1, 0][:, :3] = 0                                   # only R1 MBON05(X) zeros
    f = E.floors(d)
    taught = w_records.pilot_record({"p": d}, Z, {}, W)["floor_taught_share"]
    assert f["phi_R"] == pytest.approx(taught) and f["phi_P"] == 0.0
    d = pair(6)
    d["R2"][..., 0, 0][:, :2] = 0                                   # only R2 MBON13(X) zeros
    f = E.floors(d)
    assert f["phi_P"] == pytest.approx(w_records.pilot_record({"p": d}, Z, {}, W)["floor_taught_share"])
    assert f["phi_R"] == 0.0 and set(f["stages"]) == set(WV.STAGES)


def test_raw_contrasts():
    d = pair(7)
    r1, n1 = WV.dv(d["R1"], Z), WV.dv(d["N1"], Z)
    assert np.allclose(E.raw_contrasts(d, Z)[:, 0], (r1 - n1).mean(-1), atol=0)


def test_pair_ci_reproducible_shared_indices_and_flies_not_probes():
    a = E.pair_arrays(pair(8), Z, S)
    x, y = E.pair_ci(a, 17, S), E.pair_ci(a, 17, S)
    assert np.array_equal(x["reps"], y["reps"]) and not np.array_equal(x["reps"], E.pair_ci(a, 18, S)["reps"])
    a2 = dict(a, w=np.tile(np.arange(8.0)[:, None], (1, 4)), rc=np.tile(np.arange(8.0)[:, None], (1, 2)))
    reps = E.pair_ci(a2, 17, S)["reps"]
    assert all(np.array_equal(reps[:, 0], reps[:, j]) for j in range(6))          # one index for every column
    assert np.allclose(reps * 8, np.rint(reps * 8))                                # means of resampled fly values


def _arrs(n_pairs, odours, seed0=10):
    keys = [f"a|{100 + i}|{odours[i % len(odours)]}|y{i}" for i in range(n_pairs)]
    return keys, {k: E.pair_arrays(pair(seed0 + i), Z, S) for i, k in enumerate(keys)}


def test_two_stage_draws_whole_groups():
    D = np.zeros((4, 8, 1))
    D[:3] = 1.0                                                     # group A = pairs 0-2 value 1, group B = pair 3 value 0
    reps = E.boot_two_stage(D, [[0, 1, 2], [3]], E.stream(1, "t"), 300, 100)[:, 0]
    assert set(np.round(reps, 12)) <= {0.0, 0.75, 1.0}
    pr = E.boot_pair(D, [[0, 1, 2], [3]], E.stream(1, "t"), 300, 100)[:, 0]
    assert set(np.round(pr, 12)) - {0.0, 0.75, 1.0}                 # the pair-level mutant is distinguishable


@pytest.mark.parametrize("k,g,kind,primary,records,flag", [
    (0, 0, "none", None, [], None), (1, 1, "pair_study", None, [], "쌍별 연구"),
    (2, 2, "pooled", "fly", ["two_stage", "pair"], "적은 묶음"), (4, 4, "pooled", "fly", ["two_stage", "pair"], "적은 묶음"),
    (4, 1, "pooled", "fly", ["pair"], "적은 묶음"), (6, 4, "pooled", "fly", ["two_stage", "pair"], "적은 묶음"),
    (5, 5, "pooled", "two_stage", ["fly", "pair"], None), (20, 9, "pooled", "two_stage", ["fly", "pair"], None)])
def test_ci_plan_table(k, g, kind, primary, records, flag):
    p = E.ci_plan(k, g, S)
    assert (p["kind"], p["primary"], p["records"], p["flag"]) == (kind, primary, records, flag)


def test_estimate_set_streams_and_paths():
    keys, arrs = _arrs(12, ["E1", "E2", "E3", "E4", "E5", "E6"])
    out = E.estimate_set(arrs, keys, "S1", S)
    assert out["plan"]["primary"] == "two_stage" and out["n_groups"] == 6 and out["sizes"] == [2] * 6
    W_ = np.stack([arrs[k]["w"] for k in keys])
    want = E.boot_two_stage(W_, [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
                            E.stream(S.boot_seed, "pool", "S1"), S.boot_b, S.boot_chunk)
    assert np.array_equal(out["_reps"], want)
    lo, hi = np.percentile(want[:, 3], S.pct)
    assert out["ci"]["two_stage"]["punish_assoc"] == [lo, hi]
    assert out["point"]["punish_assoc"] == float(W_.mean(1)[:, 3].mean())
    few = E.estimate_set(arrs, keys[:3], "S1", S)
    assert few["plan"]["primary"] == "fly" and few["plan"]["flag"] == "적은 묶음" and "range" in few
    assert np.array_equal(few["_reps"], E.boot_fly(W_[:3], [[0], [1], [2]], E.stream(S.boot_seed, "pool_fly", "S1"),
                                                   S.boot_b, S.boot_chunk))
    assert E.estimate_set(arrs, keys[:1], "S1", S)["plan"]["kind"] == "pair_study"
    o31 = E.estimate_set(arrs, keys, "S31", S)
    assert not np.array_equal(o31["_reps"], out["_reps"])            # flow "pool31"


def test_dl_hk_known_example():
    r = E.dl_hk(np.array([0.5, 1.0, 1.8, 1.2]), np.array([0.04, 0.09, 0.06, 0.05]), S)
    assert r["Q"] == pytest.approx(17.395674300254456, abs=1e-9)
    assert r["tau2"] == pytest.approx(0.27069377990430626, abs=1e-9)
    assert r["mu"] == pytest.approx(1.1181469052492783, abs=1e-9)
    assert r["ci"] == pytest.approx([0.5554688223116356, 1.680824988186921], abs=1e-9)
    assert r["I2"] == pytest.approx(0.8275433335771228, abs=1e-9)
    assert r["hk_ci"] == pytest.approx([0.25214414374888205, 1.9841496667496745], abs=1e-9)
    assert not E.dl_hk(np.array([1.0, 2.0]), np.array([0.0, 0.1]), S)["available"]
    assert not E.dl_hk(np.array([1.0]), np.array([0.1]), S)["available"]


def test_min_of_gates_signs_and_shared_reps():
    reps = np.array([[2.0, -3.0, 1.0, -0.5], [0.2, -3.0, 4.0, -4.0], [3.0, 0.5, 2.0, -2.0]])
    m = E.min_of_gates(np.array([2.0, -2.0, 1.5, -1.0]), reps, S)
    assert m["point"] == 1.0 and m["gate"] == "punish_assoc"
    assert m["lo"] == float(np.percentile([0.5, 0.2, -0.5], 2.5)) and m["hi"] == float(np.percentile([0.5, 0.2, -0.5], 97.5))


def test_compare_table_two_scales_and_bar_note():
    t = E.compare_table(1.2, 0.6, 1.8, S)
    row = {(r["ref"], r["scale"]): r for r in t}
    assert row[(1.0, "raw")]["point"] == "위" and row[(1.0, "raw")]["ci"] == "걸침"
    assert row[(1.5, "hedges")]["point"] == "아래" and "마리 관문 막대" in row[(1.0, "raw")]["note"]
    assert row[(0.5, "hedges")]["ci"] == "위" and row[(0.0, "raw")]["ci"] == "위"       # lo 0.6 × 0.8889 = 0.533 > 0.5


def test_g6_inputs_status_and_order():
    rows = E.g6_inputs(dict(point=0.9, lo=0.3, hi=1.6), S)
    assert [(r["scale"], r["row"]) for r in rows][:3] == [("hedges", "point"), ("hedges", "lo"), ("hedges", "hi")]
    st = {(r["scale"], r["row"]): r["status"] for r in rows}
    assert st[("hedges", "lo")] == st[("raw", "lo")] == "목표 ≤ 거짓 통과 효과" and st[("raw", "point")] == "compute"
    assert next(r for r in rows if r["scale"] == "hedges" and r["row"] == "point")["target"] == pytest.approx(0.9 * 8 / 9)


def test_constants():
    assert round(E.c_k(8), 4) == 1.1259 and round(E.hedges_j(8), 4) == 0.8889
