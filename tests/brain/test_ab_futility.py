"""AB.7 0f (AB.9.3) 가망 관문 시험: the AA source reproduces AA's records bit for bit (real data, cache files only),
the model's sources (DL / sd), the simulation's draw / fly steps, the judgement path (set_limits + verdict at the 0e
levels), the fixtures of AB.7 0, bit reproducibility and resume, the point decision P̂ < 0.5, the record grid
(even sizes, nearest-structure reselection, Pareto-minimal cells, the records cap)."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import aa_estimate as AE
from flymon.brain import ab_estimate as E
from flymon.brain.ab_spec import SPEC, small

ROOT = Path(__file__).resolve().parents[2]
S = small(SPEC, boot_b=200, boot_chunk=100, cal_chunk=3, fut_reps=6, fut_grid_reps=2)
G6 = E.rep_structure(SPEC.rep_sizes("g6"), S)                             # 18 pairs, 6 X groups, 9 type sets
LV = {"D": {"P": 0.001}, "Dfin": {"P": 0.001}, "R": {"P": 0.001}}
SG = np.array(E.SIGNS, float)


def _inp(mu_d, mu_r, tau=0.1, P=6, inf=False, seed=0, fly_sd=1.0, raw_sd=0.3, C=None):
    """A synthetic source of P pairs (finite fly d′ ~ N(0, fly_sd), contrasts ~ N(0, raw_sd)) and a model around it."""
    r = np.random.default_rng(seed)
    dp = r.normal(0.0, fly_sd, (P, S.flies, 4))
    if inf:
        dp = np.broadcast_to(SG * np.inf, dp.shape).copy()
    raw = r.normal(0.0, raw_sd, (P, S.flies, 4))
    C = np.eye(8) if C is None else np.asarray(C, float)
    return dict(dp=dp.tolist(), raw=raw.tolist(), mD=AE.winsorize(dp, S).mean(1).tolist(), mR=raw.mean(1).tolist(),
                mu=list(mu_d) + list(mu_r), tau=[tau] * 8, rho=[0.5] * 8, C=C.tolist(),
                chol=np.linalg.cholesky(C + S.fut_chol_eps * np.eye(8)).tolist())


def _p(inp, alpha=LV, n=6, st=G6, alloc="icc", stream=("fut", "t", "x")):
    return E.fut_run(st, "g6", alloc, inp, alpha, S, n=n, stream=stream)["passed"] / n


# ---------------------------------------------------------------- the AA source (real data, read only)
def _aa_source():
    doc = json.loads((ROOT / SPEC.aa_summary).read_text())
    rec = doc["records"]
    s1 = rec["s1_records"]
    man = json.loads((ROOT / SPEC.aa_learn_detail).read_text())["manifest"]
    keys = list(s1["keys"])
    by, n, bad = {}, 0, []
    for m in man:
        pair, fly, brain = m["key"].rsplit("|", 2)
        if pair not in keys:
            continue
        n += 1
        raw = (ROOT / m["cache_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != m["sha256"]:
            bad.append(m["key"])
        by.setdefault(pair, []).append(dict(unit=dict(fly=int(fly), brain=brain), result=json.loads(raw)["result"]))
    z = {k: (float(v[0]), float(v[1])) for k, v in
         json.loads((ROOT / "results/summary/y_learning.json").read_text())["pilot"]["z_V"].items()}
    return by, keys, z, s1, rec["pairs"], n, bad


def test_futility_inputs_reproduce_aa_records_bit_for_bit():
    by, keys, z, s1, pairs, n, bad = _aa_source()
    assert (len(keys), n, bad) == (SPEC.fut_src_pairs, SPEC.fut_src_units, [])
    arr = E.futility_inputs(by, keys, z, SPEC)
    assert E.fut_check(arr, pairs, s1, SPEC) == []
    m = E.fut_model(arr, s1, SPEC)
    assert tuple(m["tau_src"][:4]) == SPEC.fut_decl_tau_src and tuple(round(t, 3) for t in m["tau"][:4]) == \
        SPEC.fut_decl_tau
    assert tuple(round(v, 3) for v in m["mu"][:4]) == SPEC.fut_decl_point
    assert tuple(round(v, 3) for v in m["mu"][6:8]) == SPEC.fut_decl_raw_assoc
    assert tuple(m["x_sizes"]) == SPEC.fut_x_sizes and all(0.0 <= v <= 1.0 for v in m["rho"])
    C = np.asarray(m["C"])
    assert C.shape == (8, 8) and np.allclose(np.diag(C), 1.0) and np.allclose(C, C.T)
    bent = json.loads(json.dumps(pairs))                                    # one fly changed → caught
    g = "reward_assoc"
    bent[keys[0]]["raw"][g]["fly"][0] += 1e-12
    assert E.fut_check(arr, bent, s1, SPEC)


# ---------------------------------------------------------------- signatures and the judgement path
def test_signatures_take_no_measured_values():
    assert list(inspect.signature(E.futility).parameters) == ["inp", "alpha_by_tag", "s", "cell"]
    assert list(inspect.signature(E.futility_grid).parameters) == ["inp", "sel_by_tag", "s", "cell", "stop"]
    assert list(inspect.signature(E.fut_run).parameters)[:6] == ["st", "tag", "alloc", "inp", "alpha", "s"]
    with pytest.raises(TypeError):
        E.fut_run([(0, 0)], "g6", "icc", _inp([8] * 4, [8] * 4), LV, S, n=1)
    src = inspect.getsource(E.fut_rep)
    assert "set_limits(" in src and "verdict(" in src and "levels_for(" in src


def test_judgement_is_the_verdict_at_the_given_levels(monkeypatch):
    seen = []
    real = E.verdict

    def spy(lim, machine, mech, alpha, s):
        seen.append((dict(alpha), machine, mech, {n: lim[n]["levels"] for n in E.STATS}))
        return real(lim, machine, mech, alpha, s)
    monkeypatch.setattr(E, "verdict", spy)
    E.fut_run(G6, "g6", "icc", _inp(SG[:4] * 8, SG * 8), LV, S, n=2, stream=("fut", "spy"))
    assert len(seen) == 2
    al, machine, mech, lv = seen[0]
    assert al == dict(D=0.001, Dfin=0.001, R=0.001, F=None) and machine == [] and mech == {}
    assert lv == dict(D=[0.001], Dfin=[0.001], R=[0.001])


# ---------------------------------------------------------------- the draw and fly steps
def test_draw_uses_c_and_the_allocation():
    C = np.eye(8)
    C[0, 4] = C[4, 0] = 0.9
    inp = _inp([0] * 4, [0] * 4, tau=1.0, C=C)
    r = AE.stream(S.synth_seed, "fut", "test", "draw")
    th = np.concatenate([E.fut_draw(G6, inp, "pair", r, S)[0] for _ in range(300)])
    assert np.corrcoef(th[:, 0], th[:, 4])[0, 1] > 0.8 and abs(np.corrcoef(th[:, 0], th[:, 1])[0, 1]) < 0.2
    gx = np.asarray(G6)[:, 0]
    thg = E.fut_draw(G6, inp, "group", r, S)[0]
    assert all(np.ptp(thg[gx == j], axis=0).max() == 0 for j in set(gx.tolist()))      # ρ = 1: one value per group
    thp = E.fut_draw(G6, inp, "pair", r, S)[0]
    assert np.ptp(thp[gx == 0], axis=0).min() > 0                                       # ρ = 0: pairs differ


def test_flies_shift_source_residuals_and_keep_infinities():
    inp = _inp([0] * 4, [0] * 4)
    dp = np.asarray(inp["dp"])
    dp[1, 3, 2] = np.inf
    inp["dp"] = dp.tolist()
    inp["mD"] = AE.winsorize(dp, S).mean(1).tolist()
    th = np.full((2, 8), 2.0)
    q = np.array([1, 0])
    f = np.tile(np.arange(S.flies), (2, 1))
    D, Dfin, R = E.fut_flies(th, q, f, inp, S)
    mD = np.asarray(inp["mD"])
    want = 2.0 + (np.clip(dp[1], -10, 10) - mD[1])
    assert D[0, 3, 2] == 10.0 and np.isnan(Dfin[0, 3, 2])                                 # +∞ kept, not shifted
    keep = np.isfinite(dp[1])
    assert np.array_equal(D[0][keep], np.clip(want, -10, 10)[keep]) and np.array_equal(Dfin[0][keep], D[0][keep])
    raw, mR = np.asarray(inp["raw"]), np.asarray(inp["mR"])
    assert np.array_equal(R[1], 2.0 + (raw[0] - mR[0]))


# ---------------------------------------------------------------- the AB.7 0 fixtures
def test_fixture_large_effects_pass():
    assert _p(_inp(SG[:4] * 8, SG * 8)) > 0.95


def test_fixture_d_at_the_bar_fails():
    j = E.hedges(S)
    assert _p(_inp(SG[:4] / j, SG * 8)) < 0.05


def test_fixture_level_gate_r_small_never_passes():
    mu_r = SG * 8
    mu_r[0] = 0.1
    assert _p(_inp(SG[:4] * 8, mu_r)) == 0.0


def test_fixture_all_source_flies_infinite_never_passes():
    assert _p(_inp(SG[:4] * 8, SG * 8, inf=True)) == 0.0                          # D at ±10, D_fin empty


def test_uses_the_given_levels():
    inp = _inp(SG[:4] * 2.5, SG * 8, tau=0.6)                             # D-limited (validated: 4 vs 12 of 12)
    a = E.fut_run(G6, "g6", "icc", inp, LV, S, n=12, stream=("fut", "lv"))
    b = E.fut_run(G6, "g6", "icc", inp, {"D": {"P": 0.025}, "Dfin": {"P": 0.025}, "R": {"P": 0.025}}, S, n=12,
                  stream=("fut", "lv"))
    assert b["passed"] > a["passed"]


def test_pass_needs_both_methods():
    p = E.fut_run(G6, "g6", "icc", _inp(SG[:4] * 2.5, SG * 8, tau=0.6), LV, S, n=12, stream=("fut", "lv"))
    b = p["blocked"]
    assert all(b[g][n]["TS"] == 0 for g in b for n in E.STATS) and any(b[g]["D"]["CG"] > 0 for g in b)
    assert p["passed"] < p["n"]                                     # CG blocks replicates TS alone would pass


# ---------------------------------------------------------------- reproducibility and resume
def test_reproducible_and_resumable():
    inp = _inp(SG[:4] * 2.6, SG * 0.9, tau=1.0)
    a = E.fut_run(G6, "g6", "icc", inp, LV, S, n=7)
    assert a == E.fut_run(G6, "g6", "icc", inp, LV, S, n=7)
    seen = []
    E.fut_run(G6, "g6", "icc", inp, LV, S, n=7, on_chunk=seen.append)
    assert seen[0]["done"] == S.cal_chunk
    assert E.fut_run(G6, "g6", "icc", inp, LV, S, n=7, resume=seen[0]) == a
    assert a["state"] != E.fut_run(G6, "g6", "pair", inp, LV, S, n=7)["state"]
    assert a["state"] != E.fut_run(G6, "g5", "icc", inp, LV, S, n=7)["state"]
    r = AE.stream(SPEC.synth_seed, "fut", "g6", "icc")
    assert tuple(int(v) for v in r.integers(0, 2 ** 63, size=3)) == SPEC.stream_vectors[-1][1]


# ---------------------------------------------------------------- the decision (point P̂, strict)
def _fake(passed_by_row):
    jobs = []

    def run(kind, args):
        st, tag, alloc, _inp_, alpha, s = args[:6]
        jobs.append((kind, tag, alloc, st, alpha))
        x = passed_by_row.get(f"{tag}|{alloc}", 0)
        return dict(done=s.fut_reps, n=s.fut_reps, passed=x, blocked=E._blocked_zero(), state=None)
    return run, jobs


def test_decision_point_estimate_strict_and_on_g6_icc():
    s = small(SPEC, fut_reps=2_000)
    al = {t: {"D": {"P": 0.001 * (i + 1), "F": None}, "Dfin": {"P": 0.0005}, "R": {"P": 0.00025}}
          for i, t in enumerate(s.fut_tags)}
    run, jobs = _fake({"g6|icc": 1_000})
    out = E.futility({}, al, s, run)
    assert out["judge"]["row"] == "g6|icc" and out["judge"]["p_hat"] == 0.5 and out["judge"]["futile"] is False
    assert len(out["rows"]) == 9 and [(t, a) for _, t, a, _, _ in jobs] == [(t, a) for t in s.fut_tags
                                                                           for a in s.fut_allocs]
    assert all(st == E.rep_structure(s.rep_sizes(t), s) and alpha is al[t] for _, t, _, st, alpha in jobs)
    run, _ = _fake({"g6|icc": 999, "g5|icc": 2_000, "g7|icc": 2_000, "g6|pair": 2_000})
    o2 = E.futility({}, al, s, run)
    assert o2["judge"]["futile"] is True and o2["rows"]["g6|icc"]["cp95"][1] > 0.5     # CP not used for the decision
    assert o2["rows"]["g6|icc"]["alpha"] == dict(D=0.002, Dfin=0.0005, R=0.00025, F=None)


def test_small_scale_end_to_end():
    s = small(S, fut_reps=3)
    al = {t: LV for t in s.fut_tags}
    out = E.futility(_inp(SG[:4] * 8, SG * 8), al, s)
    assert out["judge"]["futile"] is False and all(r["n"] == 3 for r in out["rows"].values())
    w = out["rows"]["g6|icc"]["worst"]
    assert set(w) == {"gate", "stat", "method", "share"}


# ---------------------------------------------------------------- the record grid
def test_grid_helpers():
    assert E.even_sizes(10, 4) == (3, 3, 2, 2) and E.even_sizes(8, 8) == (1,) * 8 and sum(E.even_sizes(40, 12)) == 40
    cells = E.grid_cells(SPEC)
    assert len(cells) == 68 and (9, 8) not in cells and (8, 8) in cells
    assert [E.nearest_tag(g, SPEC) for g in (5, 6, 7, 12)] == ["g5", "g6", "g7", "g7"]
    assert [len(E.scenario_cells(t, SPEC)) for t, _ in SPEC.fut_scenarios] == [24, 20, 20, 11]
    assert E.scenario_cells("sd1", SPEC) == [1, 2, 3, 6, 7, 10, 11, 14, 22, 23, 24]


def test_reselect_matches_hand_choice():
    def cnt(x):                                                            # x misses at every P level
        return {"D": {"P": list(x), "F": [0] * 9}, "Dfin": {"P": [0] * 8}, "R": {"P": [0] * 8}}
    sel = {str(ci): cnt([0] * 8) for ci in range(1, 25)}
    sel["19"] = cnt([300, 200, 150, 104, 103, 50, 10, 0])                  # x ≤ 103 of 5 000 passes the CP rule
    assert E.reselect(sel, E.scenario_cells("all", SPEC), SPEC)["D"]["P"] == 0.001
    assert E.reselect(sel, E.scenario_cells("noskew", SPEC), SPEC)["D"]["P"] == 0.025
    sel["1"] = cnt([999] * 8)
    assert E.reselect(sel, E.scenario_cells("all", SPEC), SPEC) is None


def test_pareto_min_hand_example():
    pts = [(5, 40), (6, 24), (7, 24), (8, 12), (9, 12), (12, 8), (6, 28), (12, 40)]
    assert E.pareto_min(pts) == [(5, 40), (6, 24), (8, 12), (12, 8)]
    assert E.pareto_min([]) == []


def test_grid_runs_reselects_and_stops_at_the_records_cap():
    sel = {t: {str(ci): {"D": {"P": [0] * 8, "F": [0] * 9}, "Dfin": {"P": [0] * 8}, "R": {"P": [0] * 8}}
               for ci in range(1, 25)} for t in SPEC.fut_tags}
    seen = []

    def run(kind, args):
        st, tag, alloc, _i, alpha, s, n, _r, _c, _b, stream = args
        seen.append((tag, alloc, n, tuple(stream)))
        g, k = stream[3], stream[4]
        return dict(done=n, n=n, passed=n if (g >= 7 and k >= 24) or (g >= 10) else 0, blocked=E._blocked_zero(),
                    state=None)
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 2
    out = E.futility_grid({}, sel, S, run, stop)
    assert list(out) == ["all", "noskew", "sd2", "sd1"]
    assert out["all"]["pareto"] == [[7, 24], [10, 12]] and out["noskew"]["pareto"] == [[7, 24], [10, 12]]
    assert out["sd2"]["pareto"] is None and all(c["reason"] == E.RECORDS_REASON for c in out["sd1"]["cells"])
    assert len(seen) == 2 * 68 and all(a == S.fut_grid_alloc and n == S.fut_grid_reps for _, a, n, _ in seen)
    assert seen[0][3] == ("fut", "grid", "all", 5, 8) and seen[0][0] == "g5"
    assert {t for t, _, _, st in seen if st[3] >= 7} == {"g7"}
