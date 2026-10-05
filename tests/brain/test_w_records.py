"""W's records and plumbing (W.3 2, W.6, W.9.4, W.9.6 P2-10, W.9.8 H6, W.9.9 P1-4 / P2-9): job rows -> [K, 2, 2] and
{stage: [F, K, 2, 2]}; the machine reasons (pre across brains and the screen, RN's reward-phase weights = R's — an RN
that skipped its own reward phase is caught, the declared edit / CSC / edges / probe seeds, the band jobs' weights);
the 2K extension; the path comparisons; the reward check; the pilot record; the cost model."""
import copy

import numpy as np

from flymon.brain import w_records as WR
from flymon.brain import w_verdict as WV
from flymon.brain.w_spec import SPEC


def pres(a, p, seed=0):
    return dict(seed=seed, A=a, P=p, kc_frac=0.05, kc_spikes=1, kc_max_win_hz=1.0, apl_out_per_step=0.1, wall_s=0.3,
                steps=1400)


def stage(name, k, ax=30, px=60, sha="w0"):
    return dict(stage=name, x=[pres(ax + i, px - i, i) for i in range(k)], y=[pres(30, 60, i) for i in range(k)],
                w_sha256=sha, weights_frac=1.0, weights_frac_A=1.0, weights_frac_P=0.9, da_integral=None)


def job(brain, fly, k=4, edit="E", seeds=None):
    sh = {"R": ("r1", "r2"), "N": ("n1", "n2"), "RN": ("r1", "rn2")}[brain]
    return dict(unit=dict(pair="p", fly=fly, brain=brain), result=dict(
        edit=edit, csc_sha256="S", edit_edges=2, block_edges={"a": 1}, w0_sha256="w0", fly=fly, plastic=True,
        probe_seeds=seeds or [fly * 100 + i for i in range(k)],
        stages=[stage("pre", k), stage("S1", k, px=40, sha=sh[0]), stage("S2", k, ax=20, px=40, sha=sh[1])],
        wall_s=1.0, train_s=0.8, probe_s=0.2))


DECL = dict(edit="E", csc_sha256="S", edit_edges=2, block_edges={"a": 1},
            probe_seeds={f: [f * 100 + i for i in range(4)] for f in range(2)})


def rows2():
    return [job(b, f) for f in range(2) for b in ("R", "N", "RN")]


def test_counts_and_pair_data():
    c = WR.counts(stage("pre", 3))
    assert c.shape == (3, 2, 2) and c[1].tolist() == [[31, 30], [59, 60]]
    d = WR.pair_data(rows2(), [0, 1])
    assert set(d) == set(WV.STAGES) and d["R1"].shape == (2, 4, 2, 2)
    assert np.array_equal(d["R1"], d["RN1"]) and d["R2"][0, 0, WV.A, WV.X] == 20


def test_machine_reasons():
    assert WR.machine_reasons(rows2(), [0, 1], DECL) == []
    r = rows2()
    r[1]["result"]["stages"][0]["x"][0]["A"] += 1                                  # N's pre differs
    assert WR.machine_reasons(r, [0, 1], DECL) == ["fly 0: N pre ≠ R pre"]
    r = rows2()
    r[2]["result"]["stages"][1]["w_sha256"] = "w0"                                 # RN skipped its reward phase
    assert WR.machine_reasons(r, [0, 1], DECL) == ["fly 0: RN 보상 뒤 가중치 ≠ R"]
    r = rows2()
    r[0]["result"]["edit_edges"] = 0
    assert WR.machine_reasons(r, [0, 1], DECL) == ["fly 0 R: edit_edges 0 ≠ 2"]
    assert WR.machine_reasons(rows2()[:5], [0, 1], DECL) == ["fly 1: ['RN'] 없음"]
    naive = {0: WR.counts(stage("pre", 4)), 1: WR.counts(stage("pre", 4)) + 1}
    assert WR.machine_reasons(rows2(), [0, 1], DECL, naive=naive) == ["fly 1: 학습 pre ≠ 순진 거름 프로브"]
    band = [copy.deepcopy(x) for x in rows2()]
    assert WR.machine_reasons(rows2(), [0, 1], DECL, band=band) == []
    band[3]["result"]["stages"][2]["w_sha256"] = "zz"
    assert WR.machine_reasons(rows2(), [0, 1], DECL, band=band) == ["band fly 1 R: 재훈련 가중치 ≠ 주 측정"]
    seeds = dict(DECL, probe_seeds={0: [1, 2, 3, 4], 1: DECL["probe_seeds"][1]})
    assert WR.machine_reasons(rows2(), [0, 1], seeds)[0] == "fly 0 R: 프로브 시드가 선언과 다름"


def test_extend():
    d = WR.pair_data(rows2(), [0, 1])
    e = WR.extend(d, rows2(), [0, 1])
    assert e["R1"].shape == (2, 8, 2, 2) and np.array_equal(e["R1"][:, :4], d["R1"])


def test_p_and_naive_repro_and_reward_check():
    w = job("R", 0, k=1)["result"]
    w["stages"] = w["stages"][:2]
    st = {s["stage"]: s for s in w["stages"]}
    v = dict(pre=dict(x=st["pre"]["x"][0], y=st["pre"]["y"][0]), post=dict(x=st["S1"]["x"][0], y=st["S1"]["y"][0]),
             w0_sha256="w0", w_post_sha256="r1", weights_frac=1.0, weights_frac_A=1.0, weights_frac_P=0.9,
             da_integral=None, csc_sha256="S")
    assert WR.p_repro_diffs(w, v, "t") == []
    v2 = copy.deepcopy(v)
    v2["post"]["x"]["wall_s"] = 9.0                                                # wall time is not compared
    assert WR.p_repro_diffs(w, v2, "t") == []
    v2["post"]["x"]["A"] += 1
    v2["w_post_sha256"] = "q"
    assert WR.p_repro_diffs(w, v2, "t") == ["t post x", "t w_post_sha256"]
    res = {"report": {"pre": {"A": [[30, 30]], "P": [[60, 60]]}}}
    assert WR.naive_repro_diffs(w, res, "n") == []
    res["report"]["pre"]["P"][0][1] = 61
    assert WR.naive_repro_diffs(w, res, "n") == ["n 순진 카운트"]
    assert WR.reward_check(w) == []
    w["stages"][1]["w_sha256"] = "w0"
    w["stages"][1]["weights_frac_P"] = 1.0
    w["stages"][1]["x"][0]["P"] = 60
    assert len(WR.reward_check(w)) == 3


def test_pilot_record():
    d = WR.pair_data(rows2(), [0, 1])
    rec = WR.pilot_record({"p": d}, {"A": (0.0, 1.0), "P": (0.0, 1.0)}, dict(job_s_median=1.0), SPEC)
    p = rec["pairs"]["p"]
    assert set(p["median"]) == set(WV.GATES) and rec["label"] == "탐색"
    assert p["spill"]["reward"] == dict(x=0.0, y=0.0) and p["suppression"]["N2_minus_pre"] == -10.0
    assert rec["floor_both_share"] == 0.0 and rec["walls"] == dict(job_s_median=1.0)


def test_costs():
    c = WR.unit_costs([job("R", 0)["result"]], 40, 120.0, 16)
    assert c == dict(trial_s=0.02, presentation_s=0.2 / 24, oracle_round_s=120.0, workers=16)
    dc = WR.design_cost(dict(trial_s=1.0, presentation_s=1.0, oracle_round_s=100.0, workers=16), 8, 8, SPEC, 249)
    p = dc["parts_h"]
    assert p["oracle"] == 16 * 100 / 3600 and p["naive"] == -(-249 * 8 // 16) * 16 / 3600
    assert p["learn"] == p["band"] == p["c"] == 12 * (40 + 48) / 3600 and p["noplast"] == 88 / 3600
    no_c = WR.design_cost(dict(trial_s=1.0, presentation_s=1.0, oracle_round_s=100.0, workers=16), 8, 8, SPEC, 0, 30,
                          False)
    assert no_c["parts_h"]["c"] == 0.0 and no_c["parts_h"]["oracle"] == 0.0
    assert WR.elapsed_h([dict(wall_s=1800.0), dict(wall_s=1800.0)]) == 1.0


def test_docstring_names_exist():
    """Every function the module docstring lists (the "- a / b / c:" lines) exists in w_records."""
    import re

    from flymon.brain import w_records
    names = [n.split(" (")[0].strip() for line in w_records.__doc__.splitlines() if line.startswith("- ")
             for n in line[2:].split(":")[0].split("/")]
    assert names and all(callable(getattr(w_records, n, None)) for n in names), names
    assert re.search(r"\bworst_cost\b", w_records.__doc__) is None
