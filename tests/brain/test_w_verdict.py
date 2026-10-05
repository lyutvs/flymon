"""W's judgement code (W.4, W.9.1, W.9.2, W.9.8 H2 / H5 / H7, W.9.9 P1-2 / P2-9 / P2-10 / P2-11; G.7's defects; F.6's
fixtures moved to RN, z_V-style z and per-fly joint satisfaction). Known answers: specific learning PASS; no
learning, X-only presentation drift (F.6 path 1), punishment-phase drift with a powerless punishment (F.6 path 2),
full generalisation, a broken mechanism control and reward-only / punishment-only flies (joint vs marginal) FAIL; the
BAND boundaries 0.80 / 0.79 / −0.80; ±∞ by direction; NaN and K < 2 INVALID → STOP_MACHINE; a missing stage, a
mis-shaped stage and a short fly list INVALID; the scheduled-F denominator; BAND → 2K (PASS / FAIL / pending); RN1 one
bit off → STOP_MACHINE; H7's order. Mutations each flip a fixture: no RN1 check, the valid-fly denominator, a deleted
gate, the ±∞ direction ignored, the mechanism control ignored."""
import numpy as np
import pytest

from flymon.brain import w_verdict as WV
from flymon.brain.w_spec import SPEC

Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}           # V = A − P, ΔV = (A_X − P_X) − (A_Y − P_Y)
F, K = 8, 8


def pair(seed=0, a=0.0, b=0.0, drift1=0.0, drift2=0.0, gen=0.0, sd=1.0, f=F, k=K, ypx=0.0):
    """Counts around 50 with probe noise sd shared by every slot of a probe (paired noise) plus a small independent
    part; a = reward drop of MBON05(X) on R / RN, b = punishment drop of MBON13(X) on R2, drift1 / drift2 = X-only
    MBON13 drops after the first / second presentation block (every brain), gen = the share of every change Y gets
    too, ypx = a rise of MBON05(Y) at R1 / R2 / RN (a gate effect that is not on the taught cell)."""
    r = np.random.default_rng(seed)
    base = np.full((f, k, 2, 2), 50.0) + r.normal(0, sd, (f, k, 1, 1))
    d = {}
    for s in WV.STAGES:
        m = base + r.normal(0, sd * 0.5, (f, k, 2, 2))
        dx = np.zeros((2, 2))
        if s != "pre":
            dx[WV.A, WV.X] -= drift1
        if s in ("R2", "N2", "RN2"):
            dx[WV.A, WV.X] -= drift2
        if s in ("R1", "R2", "RN1", "RN2"):
            dx[WV.P, WV.X] -= a
            dx[WV.P, WV.Y] += ypx
        if s == "R2":
            dx[WV.A, WV.X] -= b
        dx[:, WV.Y] += gen * dx[:, WV.X]
        d[s] = np.clip(np.rint(m + dx), 0, None).astype(np.int64)
    d["RN1"] = d["R1"].copy()
    return d


def jp(d, d2=None, q=0.75, f=F, k=K):
    return WV.judge_pair(d, d2, Z, q, f, k, SPEC)


def test_dprime_limits_and_nan():
    assert float(WV.dprime([1.0, 1.0])) == np.inf and float(WV.dprime([-2.0, -2.0])) == -np.inf
    assert float(WV.dprime([0.0, 0.0])) == 0.0
    assert np.isnan(WV.dprime([1.0])) and np.isnan(WV.dprime([1.0, np.nan]))
    assert float(WV.dprime([1.0, 3.0])) == pytest.approx(2.0 / np.sqrt(2.0))
    x = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert WV.dprime(x).tolist() == [np.inf, 1.0 / np.sqrt(2.0)]


def test_fly_class_boundaries_and_infinities():
    c = lambda s: int(WV.fly_class(np.array(s, float)))            # noqa: E731
    assert c([1.0, -1.0, 1.0, -1.0]) == WV.FLY_SAT
    assert c([0.80, -1.0, 1.0, -1.0]) == WV.FLY_BAND                # 0.80 is BAND (F.5)
    assert c([0.79, -1.0, 1.0, -1.0]) == WV.FLY_FAIL                # 0.79 is FAIL
    assert c([1.0, -0.80, 1.0, -1.0]) == WV.FLY_BAND
    assert c([1.0, -0.79, 1.0, -1.0]) == WV.FLY_FAIL
    assert c([np.inf, -np.inf, np.inf, -np.inf]) == WV.FLY_SAT      # H5: the gate's direction passes
    assert c([-np.inf, -1.0, 1.0, -1.0]) == WV.FLY_FAIL             # against it fails
    assert c([1.0, np.inf, 1.0, -1.0]) == WV.FLY_FAIL
    assert c([1.0, -1.0, np.nan, -1.0]) == WV.FLY_INVALID
    assert c([0.9, -0.9, 0.85, -0.95]) == WV.FLY_BAND


def test_pair_code_scheduled_denominator():
    S, B, Fl = WV.FLY_SAT, WV.FLY_BAND, WV.FLY_FAIL
    assert int(WV.pair_gate_code(np.array([S] * 6 + [Fl] * 2), 0.75, 8)) == WV.P_PASS      # 6/8 = 0.75
    assert int(WV.pair_gate_code(np.array([S] * 5 + [Fl] * 3), 0.75, 8)) == WV.P_FAIL
    assert int(WV.pair_gate_code(np.array([S] * 5 + [B] + [Fl] * 2), 0.75, 8)) == WV.P_BAND
    assert int(WV.pair_gate_code(np.array([S] * 5 + [B] + [Fl] * 2), 0.625, 8)) == WV.P_PASS
    assert int(WV.pair_gate_code(np.array([S] * 6), 0.75, 8)) == WV.P_INVALID           # 6 of 8 flies present
    assert int(WV.pair_gate_code(np.array([S] * 7 + [WV.FLY_INVALID]), 0.5, 8)) == WV.P_INVALID


def test_known_answers():
    assert jp(pair(a=8, b=8))["status"] == "PASS"
    assert jp(pair())["status"] == "FAIL"
    one = jp(pair(drift1=6))                                       # F.6 path 1: X-only drift, no DAN effect
    assert one["status"] == "FAIL" and one["failing_gates"]["reward_assoc"] >= 6
    two = jp(pair(a=8, drift2=6))                                  # F.6 path 2: punishment powerless, drift
    assert two["status"] == "FAIL" and two["failing_gates"]["punish_assoc"] >= 6
    assert jp(pair(a=8, b=8, gen=1.0))["status"] == "FAIL"        # full generalisation
    mech = jp(pair(b=8, ypx=8))                                    # gates pass through Y, MBON05(X) untouched
    assert mech["status"] == "FAIL" and mech["reasons"] == ["기계 대조 실패"] and mech["mech_fail"]


def test_joint_not_marginal():
    """Reward-only flies and punishment-only flies: every gate passes in half the flies (a per-gate median at q 0.5
    would pass), no fly passes all four — FAIL at every q (G.7: joint satisfaction)."""
    rw, pu = pair(a=8, seed=1), pair(b=8, seed=2)
    d = {s: np.concatenate([rw[s][:4], pu[s][:4]]) for s in WV.STAGES}
    st = WV.gate_stats(d, Z)
    med = np.median(st, axis=0) * WV.SIGNS                         # F v3's per-gate medians: each ≥ 0.8
    assert (med >= 0.8).all()
    for q in SPEC.q_grid:
        r = jp(d, q=q)
        assert r["status"] == "FAIL" and r["n_sat"] == 0


def test_band_resolution():
    for seed in range(400):                                        # find a BAND pair at K = 8 with these effects
        d2 = pair(a=2.2, b=2.2, seed=seed, k=2 * K)
        d = {s: v[:, :K] for s, v in d2.items()}
        if jp(d)["code_k"] == "BAND":
            break
    else:
        pytest.fail("no BAND fixture found")
    r = jp(d, d2)
    assert r["status"] in ("PASS", "FAIL") and r["code_2k"] in ("PASS", "BAND", "FAIL")
    assert r["status"] == ("PASS" if r["code_2k"] == "PASS" and r["mech_2k_ok"] else "FAIL")
    assert jp(d)["status"] == "BAND"                               # no 2K data: pending
    bad = {s: v.copy() for s, v in d2.items()}
    bad["R1"][0, 0, 0, 0] += 1
    assert jp(d, bad)["status"] == "INVALID"                       # the 2K data's first K probes must be K's


def test_invalid_data():
    d = pair(a=8, b=8)
    assert jp({k: v for k, v in d.items() if k != "N2"})["status"] == "INVALID"
    assert jp({s: v[:7] for s, v in d.items()})["status"] == "INVALID"          # 7 of 8 flies
    assert jp({s: v[:, :1] for s, v in d.items()}, k=1)["status"] == "INVALID"  # K < 2: NaN d′
    neg = {s: v.copy() for s, v in d.items()}
    neg["R2"][0, 0, 0, 0] = -1
    assert jp(neg)["status"] == "INVALID"


def _set(n=4, **kw):
    return {f"p{i}": (pair(seed=10 + i, **kw), None) for i in range(n)}


def test_overall_order():
    j = lambda pairs, mach=(): WV.judge(pairs, list(mach), Z, 0.75, F, K, SPEC)   # noqa: E731
    assert j(_set(a=8, b=8))["verdict"] == "PASS"
    three = j(_set(3, a=8, b=8))
    assert three["verdict"] == "UNDECIDED" and three["undecided_cause"] == "판정 가능 쌍 < 4"
    mixed = dict(_set(4, a=8, b=8), bad=(pair(), None))
    assert j(mixed)["verdict"] == "FAIL" and j(mixed)["failing"] == ["bad"]
    assert j(_set(a=8, b=8), ["fly 0: N pre ≠ R pre"])["verdict"] == "STOP_MACHINE"
    rn = _set(a=8, b=8)
    rn["p0"][0]["RN1"][0, 0, 1, 0] += 1                             # RN1 one bit off (W.9.9 P1-2)
    assert j(rn)["verdict"] == "STOP_MACHINE"
    inv = dict(_set(a=8, b=8), bad=({s: v[:, :1] for s, v in pair().items()}, None))
    assert j(inv)["verdict"] == "STOP_MACHINE" and j(inv)["invalid"] == ["bad"]
    for s in range(400):                                            # a BAND left beats nothing but FAIL / machine
        d = pair(a=2.2, b=2.2, seed=s)
        if jp(d)["code_k"] == "BAND":
            break
    band = dict(_set(a=8, b=8), band=(d, None))
    assert j(band)["verdict"] == "UNDECIDED" and j(band)["undecided_cause"] == "BAND 잔존"
    assert j(dict(band, bad=(pair(), None)))["verdict"] == "FAIL"


def test_naive_pooled():
    d = pair()
    assert abs(WV.naive_dprime(d["pre"], Z)) < 0.5
    pre = d["pre"].copy()
    pre[..., WV.A, WV.X] += 30
    assert WV.naive_dprime(pre, Z) > 0.5


# ================================================================ mutations (W.9.9 P2-11)
def test_mutation_no_rn1_check(monkeypatch):
    rn = _set(a=8, b=8)
    rn["p0"][0]["RN1"][0, 0, 1, 0] += 1
    assert WV.judge(rn, [], Z, 0.75, F, K, SPEC)["verdict"] == "STOP_MACHINE"
    monkeypatch.setattr(WV, "rn1_mismatch", lambda d: np.zeros(np.asarray(d["R1"]).shape[:-4], bool))
    assert WV.judge(rn, [], Z, 0.75, F, K, SPEC)["verdict"] != "STOP_MACHINE"


def test_mutation_valid_fly_denominator(monkeypatch):
    six = np.array([WV.FLY_SAT] * 6)
    assert int(WV.pair_gate_code(six, 0.75, 8)) == WV.P_INVALID
    real = WV.pair_gate_code
    monkeypatch.setattr(WV, "pair_gate_code", lambda cls, q, f, digits=9: real(cls, q, np.asarray(cls).shape[-1],
                                                                               digits))
    assert int(WV.pair_gate_code(six, 0.75, 8)) == WV.P_PASS


def test_mutation_deleted_gate(monkeypatch):
    """F.6 path 2 with the mechanism control switched off: the punishment association gate alone catches it."""
    d = pair(a=8, drift2=6)
    monkeypatch.setattr(WV, "mech_ok", lambda fr, mech_min=0.75, digits=9: np.ones(np.asarray(fr).shape[:-2], bool))
    assert jp(d)["status"] == "FAIL"
    real = WV.gate_stats
    monkeypatch.setattr(WV, "gate_stats", lambda dd, z: np.concatenate(
        [real(dd, z)[..., :3], np.full(real(dd, z)[..., 3:].shape, -np.inf)], -1))
    assert jp(d)["status"] == "PASS"


def test_mutation_direction_ignored(monkeypatch):
    st = np.array([-np.inf, -1.0, 1.0, -1.0])
    assert int(WV.fly_class(st)) == WV.FLY_FAIL
    monkeypatch.setattr(WV, "SIGNS", np.array([1.0, 1.0, 1.0, 1.0]))
    monkeypatch.setattr(WV, "margins", lambda s, bar=1.0, digits=9: np.round(np.abs(np.asarray(s, float)) - bar,
                                                                               digits))
    assert int(WV.fly_class(st)) == WV.FLY_SAT


def test_mutation_mechanism_ignored(monkeypatch):
    d = pair(b=8, ypx=8)
    assert jp(d)["status"] == "FAIL"
    monkeypatch.setattr(WV, "mech_ok", lambda fr, mech_min=0.75, digits=9: np.ones(np.asarray(fr).shape[:-2], bool))
    assert jp(d)["status"] == "PASS"
