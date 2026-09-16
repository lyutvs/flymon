"""Exercise verdict.py on synthetic fixtures with known answers, mutations that must change a verdict,
and the M0c raw data that sank v2's protocol check. Run: uv run python validate.py"""
from __future__ import annotations

import copy, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import verdict as V   # noqa: E402

REPO = Path(__file__).resolve().parents[4]   # repo root: docs/superpowers/specs/m2-learning-f-v3/validate.py
N_SEEDS, N_FLIES = 8, 8
FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'XX'}] {name:<58} got {got!s:<16} want {want}")
    if not ok:
        FAILS.append(name)


# ---- synthetic generator -----------------------------------------------------------------------
def brain(rng, base, reward=(0, 0), punish=(0, 0), drift_P=0.0, drift_A=0.0, stage2=True, bimodal_Px=False, drift_x_only=False, drift_stages=(1, 2)):
    """Counts for one brain through two stages. base = naive means {A: [x, y], P: [x, y]}.
    reward = (fraction P_X lost, fraction P_Y lost) associatively; punish likewise for A.
    drift_* = non-associative fractional loss applied to BOTH candidates every stage (also in no-DAN brains)."""
    def draw(mA, mP):
        A = rng.poisson(np.maximum(mA, 0), size=(N_SEEDS, 2)).astype(float)
        mPs = np.tile(np.maximum(mP, 0), (N_SEEDS, 1)).astype(float)
        if bimodal_Px:
            mPs[: N_SEEDS // 3, 0] = 1.0              # like design odour b: MBON05 near-silent on some seeds
        P = rng.poisson(mPs).astype(float)
        return {"A": A.tolist(), "P": P.tolist()}
    a0, p0 = np.array(base["A"], float), np.array(base["P"], float)
    pre = draw(a0, p0)
    dP = np.array([drift_P, 0.0 if drift_x_only else drift_P]); dA = np.array([drift_A, 0.0 if drift_x_only else drift_A])
    k1, k2 = (1.0 if 1 in drift_stages else 0.0), (1.0 if 2 in drift_stages else 0.0)
    p1 = p0 * (1 - k1 * dP) * (1 - np.array(reward)); a1 = a0 * (1 - k1 * dA)
    s1 = draw(a1, p1)
    a2 = a1 * (1 - k2 * dA) * (1 - np.array(punish)); p2 = p1 * (1 - k2 * dP)
    s2 = draw(a2, p2)
    return pre, s1, s2


def make_pair(seed, base, reward, punish, drift_P=0.25, drift_A=0.15, bimodal_Px=False, same_probe_noise=True, drift_x_only=False, drift_stages=(1, 2)):
    """N_FLIES flies; each has a DAN brain (R) and a paired no-DAN brain (N) sharing its probe seeds."""
    flies, pres = [], []
    for f in range(N_FLIES):
        rng_r = np.random.default_rng(seed * 1000 + f)
        rng_n = np.random.default_rng(seed * 1000 + f) if same_probe_noise else np.random.default_rng(seed * 7919 + f)
        pre, R1, R2 = brain(rng_r, base, reward, punish, drift_P, drift_A, bimodal_Px=bimodal_Px, drift_x_only=drift_x_only, drift_stages=drift_stages)
        _pre_n, N1, N2 = brain(rng_n, base, (0, 0), (0, 0), drift_P, drift_A, bimodal_Px=bimodal_Px, drift_x_only=drift_x_only, drift_stages=drift_stages)
        flies.append(V.fly_stats(pre, R1, R2, N1, N2)); pres.append(pre)
    return pres, flies


BASE = {"A": [22, 22], "P": [40, 40]}          # naive-equal pair: dV ~ 0

print("\n== synthetic fixtures ==")
pres, flies = make_pair(1, BASE, reward=(0.95, 0.10), punish=(0.70, 0.10))
check("working, specific learning -> PASS", V.pair_verdict(pres, flies)["state"], "PASS")

pres, flies = make_pair(2, BASE, reward=(0, 0), punish=(0, 0))
check("no learning, only non-associative drift -> FAIL", V.pair_verdict(pres, flies)["state"], "FAIL")

pres, flies = make_pair(3, BASE, reward=(0.95, 0.95), punish=(0.70, 0.70))
check("full generalisation to Y (non-specific) -> FAIL", V.pair_verdict(pres, flies)["state"], "FAIL")

pres, flies = make_pair(4, BASE, reward=(0, 0), punish=(0, 0), drift_P=0.6, drift_A=0.4)
check("big non-associative drift, no associative part -> FAIL", V.pair_verdict(pres, flies)["state"], "FAIL")

pres, flies = make_pair(5, {"A": [22, 22], "P": [15, 55]}, reward=(0.95, 0.10), punish=(0.70, 0.10))
check("naive preference far from 0 -> NAIVE_REGRESSED", V.pair_verdict(pres, flies)["state"], "NAIVE_REGRESSED")

def make_control(seed, base, reward, punish, bimodal_Px=True, drift_P=0.25, drift_A=0.15):
    out = []
    for f in range(N_FLIES):
        pre, R1, R2 = brain(np.random.default_rng(seed * 1000 + f), base, reward, punish, drift_P, drift_A, bimodal_Px=bimodal_Px)
        _p, N1, N2 = brain(np.random.default_rng(seed * 1000 + f), base, (0, 0), (0, 0), drift_P, drift_A, bimodal_Px=bimodal_Px)
        out.append(V.control_fly(pre, R1, R2, N1, N2))
    return V.control_verdict(out)


CTRL = {"A": [30, 0], "P": [45, 42]}          # design pair: MBON13 answers X (odour b) only
check("protocol check: bimodal P(X), working -> PASS", make_control(6, CTRL, (0.95, 0.30), (0.70, 0.0))["state"], "PASS")
check("protocol check: bimodal P(X), no learning -> FAIL", make_control(7, CTRL, (0, 0), (0, 0))["state"], "FAIL")
check("protocol check: reward works, punish does nothing -> FAIL", make_control(10, CTRL, (0.95, 0.30), (0, 0))["state"], "FAIL")

pres, flies = make_pair(12, BASE, reward=(0, 0), punish=(0, 0), drift_P=0.55, drift_A=0.0, drift_x_only=True)
pv12 = V.pair_verdict(pres, flies)
print(f"       (X-only presentation drift, no DAN: reward_level median {pv12['medians']['reward_level']:.2f}, "
      f"reward_assoc median {pv12['medians']['reward_assoc']:.2f})")
check("X-only presentation drift, NO DAN effect -> FAIL (level alone would pass)", pv12["state"], "FAIL")
check("  ...and the literal spec-5 level gate alone WOULD have passed it", pv12["medians"]["reward_level"] >= V.BAR, True)

pres, flies = make_pair(13, BASE, reward=(0.95, 0.10), punish=(0.70, 0.10), drift_P=0.55, drift_A=0.0, drift_x_only=True)
check("X-only presentation drift PLUS specific learning -> PASS", V.pair_verdict(pres, flies)["state"], "PASS")

PUNISH_MIMIC = dict(drift_P=0.0, drift_A=0.6, drift_x_only=True, drift_stages=(2,))
pres, flies = make_pair(14, BASE, reward=(0.95, 0.10), punish=(0, 0), **PUNISH_MIMIC)
pv14 = V.pair_verdict(pres, flies)
m = pv14["medians"]
print(f"       (reward works, punish inert, stage-2 presentation erodes MBON13 of X: reward_level {m['reward_level']:.2f}, "
      f"punish_drop {m['punish_drop']:.2f}, reward_assoc {m['reward_assoc']:.2f}, punish_assoc {m['punish_assoc']:.2f})")
check("reward works, punish inert, presentation mimics punishment -> FAIL", pv14["state"], "FAIL")
check("  ...and both literal spec-5 level gates alone WOULD have passed it",
      m["reward_level"] >= V.BAR and m["punish_drop"] <= -V.BAR, True)

pres, flies = make_pair(11, BASE, reward=(0, 0), punish=(0, 0), same_probe_noise=False)
check("no learning, R and no-DAN brains NOT bit-identical -> FAIL", V.pair_verdict(pres, flies)["state"], "FAIL")

z = {"A": [[0, 0]] * N_SEEDS, "P": [[0, 0]] * N_SEEDS}
live = {"A": [[22, 22]] * N_SEEDS, "P": [[40, 40]] * N_SEEDS}
check("every readout cell dead after training -> FAIL (not UNDEFINED)",
      V.pair_verdict([live] * N_FLIES, [V.fly_stats(live, z, z, live, live) for _ in range(N_FLIES)])["state"], "FAIL")

dead = [V.FlyStats(None, None, None, None, None, None) for _ in range(N_FLIES)]
pres, _ = make_pair(8, BASE, reward=(0, 0), punish=(0, 0))
check("statistics undefined in most flies -> UNDEFINED", V.pair_verdict(pres, dead)["state"], "UNDEFINED")

pres, flies = make_pair(9, BASE, reward=(0.95, 0.10), punish=(0.70, 0.10))
near = [V.FlyStats(0.9, -1.5, 1.4, -1.3, 0, 0) for _ in range(N_FLIES)]
check("one gate 0.1 short of the bar -> BAND", V.pair_verdict(pres, near)["state"], "BAND")
check("one gate exactly 0.2 short (0.80) -> BAND (inclusive)",
      V.pair_verdict(pres, [V.FlyStats(0.80, -1.5, 1.4, -1.3, 0, 0) for _ in range(N_FLIES)])["state"], "BAND")
check("one gate 0.21 short (0.79) -> FAIL",
      V.pair_verdict(pres, [V.FlyStats(0.79, -1.5, 1.4, -1.3, 0, 0) for _ in range(N_FLIES)])["state"], "FAIL")
check("punish gate exactly 0.2 short (-0.80) -> BAND",
      V.pair_verdict(pres, [V.FlyStats(1.5, -0.80, 1.4, -1.3, 0, 0) for _ in range(N_FLIES)])["state"], "BAND")

pre = {"A": [[20, 20]] * N_SEEDS, "P": [[40, 40]] * N_SEEDS}
bad = copy.deepcopy(pre); bad["P"][3] = [40, 39]
check("noplast identical -> ok", V.noplast_ok(pre, copy.deepcopy(pre)), True)
check("noplast off by one spike -> not ok", V.noplast_ok(pre, bad), False)

pv = {"state": "PASS"}; fv = {"state": "FAIL"}; nr = {"state": "NAIVE_REGRESSED"}; bd = {"state": "BAND"}
check("overall: noplast broken -> STOP_MACHINE", V.overall({"t1": False}, pv, {"a": pv, "b": pv})["verdict"], "STOP_MACHINE")
check("overall: control fails -> STOP_PROTOCOL", V.overall({"t1": True}, fv, {"a": pv, "b": pv})["verdict"], "STOP_PROTOCOL")
check("overall: one gate FAIL -> FAIL", V.overall({"t1": True}, pv, {"a": pv, "b": fv, "c": pv})["verdict"], "FAIL")
check("overall: all evaluable PASS (>=2) -> PASS", V.overall({"t1": True}, pv, {"a": pv, "b": nr, "c": pv})["verdict"], "PASS")
check("overall: only one evaluable -> UNDECIDED", V.overall({"t1": True}, pv, {"a": pv, "b": nr, "c": nr})["verdict"], "UNDECIDED")
check("overall: a BAND pair -> UNDECIDED", V.overall({"t1": True}, pv, {"a": pv, "b": bd, "c": pv})["verdict"], "UNDECIDED")

# ---- mutations: each must change at least one verdict ------------------------------------------
print("\n== mutations (a surviving mutation means the rule is not tested) ==")


def verdicts():
    out = []
    for seed, base, rw, pu, kw in ((1, BASE, (0.95, 0.10), (0.70, 0.10), {}),
                                   (4, BASE, (0, 0), (0, 0), {"drift_P": 0.6, "drift_A": 0.4}),
                                   (3, BASE, (0.95, 0.95), (0.70, 0.70), {}),
                                   (12, BASE, (0, 0), (0, 0), {"drift_P": 0.55, "drift_A": 0.0, "drift_x_only": True}),
                                   (14, BASE, (0.95, 0.10), (0, 0), dict(drift_P=0.0, drift_A=0.6, drift_x_only=True, drift_stages=(2,)))):
        pres, flies = make_pair(seed, base, rw, pu, **kw)
        out.append(V.pair_verdict(pres, flies)["state"])
    return out


baseline = verdicts()
orig_dv, orig_fly = V.dv, V.fly_stats

V.dv = lambda c: -orig_dv(c)                                            # 1 sign alignment flipped
check("M1 flip dV sign changes a verdict", verdicts() != baseline, True); V.dv = orig_dv


def no_contrast(pre, R1, R2, N1, N2):                                    # 2 drop the no-DAN subtraction
    s = orig_fly(pre, R1, R2, N1, N2)
    p, r1, r2 = V.dv(pre), V.dv(R1), V.dv(R2)
    s.reward_assoc = V.dprime(r1 - p); s.punish_assoc = V.dprime(r2 - r1)
    return s


V.fly_stats = no_contrast
check("M2 remove no-DAN contrast changes a verdict", verdicts() != baseline, True); V.fly_stats = orig_fly

orig_gates = V.GATES; V.GATES = (("reward_level", +1), ("punish_drop", -1))   # 3 level gates only
check("M3 drop associative gates changes a verdict", verdicts() != baseline, True); V.GATES = orig_gates

orig_sign = V.SIGN_MIN; V.SIGN_MIN = 0.0                                  # 5 protocol check neutered
check("M5 neuter the protocol check lets 'no learning' through",
      make_control(7, CTRL, (0, 0), (0, 0))["state"] == "PASS", True); V.SIGN_MIN = orig_sign

orig_dp = V.dprime                                                        # 6 revert the sd = 0 limit
V.dprime = lambda x: (None if np.asarray(x).size < 2 or float(np.asarray(x, float).std(ddof=1)) == 0.0
                      else float(np.asarray(x, float).mean() / np.asarray(x, float).std(ddof=1)))
check("M6 revert sd=0 limit turns dead-readout FAIL into something else",
      V.pair_verdict([live] * N_FLIES, [V.fly_stats(live, z, z, live, live) for _ in range(N_FLIES)])["state"] != "FAIL", True)
V.dprime = orig_dp

orig_np = V.noplast_ok; V.noplast_ok = lambda a, b: True                 # 4 noplast check disabled
check("M4 disable noplast check lets a broken machine through",
      V.overall({"t1": V.noplast_ok(pre, bad)}, pv, {"a": pv, "b": pv})["verdict"] != "STOP_MACHINE", True)
V.noplast_ok = orig_np

# ---- M0c raw data: the path that sank v2 ---------------------------------------------------------
M0C_SIGN = []
print("\n== M0c raw data (protocol-check pair: X = odour b, Y = odour a; seeds stand in for probe seeds) ==")
print("   M0c has no paired no-DAN brain; two stand-ins bound it: noplast (zero drift) and punish_only")
print("   (b got no DAN while PPL105 drove a - the largest non-associative loss on record).")
for fname in ("conditioning_seeds0-7.json", "conditioning.json"):
    d = json.load(open(REPO / "results/m0c" / fname)); ps = d["per_seed"]
    seeds = sorted(ps, key=int)

    def arm_counts(arm, which):
        c = [ps[s][arm]["counts"] for s in seeds]
        x, y = f"{which}_minus", f"{which}_plus"          # b = cs_minus = X, a = cs_plus = Y
        return {"A": [[k[x]["A"], k[y]["A"]] for k in c], "P": [[k[x]["P"], k[y]["P"]] for k in c]}

    pre = arm_counts("reward_only", "pre"); R1 = arm_counts("reward_only", "post")
    Rp = arm_counts("reversed", "post")                    # reversed: b (cs_minus) got PPL105
    for label, N in (("noplast", arm_counts("noplast", "post")), ("punish_only", arm_counts("punish_only", "post"))):
        p, r1, n1 = V.dv(pre), V.dv(R1), V.dv(N)
        rew = V.lower_fraction(np.asarray(R1["P"])[:, 0], np.asarray(N["P"])[:, 0])
        pun = V.lower_fraction(np.asarray(Rp["A"])[:, 0], np.asarray(N["A"])[:, 0])
        print(f"   {fname:<28} N={label:<12} v2 d'_delta {V.dprime(r1 - p):5.2f} | v3 effect-size d'(dV_R1-dV_N) {V.dprime(r1 - n1):5.2f}"
              f" | v3 SIGN reward {rew:.3f} punish {pun:.3f} -> {'PASS' if min(rew, pun) >= V.SIGN_MIN else 'FAIL'}")
        M0C_SIGN.append(min(rew, pun) >= V.SIGN_MIN)

check("M0c: protocol check PASSES on all 4 stand-in combinations (real working protocol)", all(M0C_SIGN), True)
print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
