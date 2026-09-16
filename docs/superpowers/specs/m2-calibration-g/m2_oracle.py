"""Spec appendix G.4 — oracle upper bound. No training: weights are set directly.

For each within-turn candidate pair (34) and the M0 design pair: estimate how often each Kenyon cell fires
for the taught odour X (seeds 500-507), then depress the KC->MBON edges of the reward compartment (PAM08
core) in proportion to that frequency, probe; then also depress the punishment compartment (PPL105 core),
probe again. Two families swept over depth alpha in {0.2, 0.5, 0.8} (G.4a): freq w <- w0 (1 - alpha f),
all w <- w0 (1 - alpha) wherever f > 0. Maximal depression is not maximal separation - MBON05 is a threshold
unit, so flooring the shared input silences it for both candidates - hence the sweep: the best reward depth
is kept, then the best punish depth on top of it.
Probes: X and Y at seeds 600-607, readout V = z(MBON13) - z(MBON05) with the frozen F.3 constants.

Classification rule (G.4, unchanged by G.4a): testable iff under either family the best reward level
d'(dV_R1) >= +2 and the best punish drop d'(dV_R2 - dV_R1) <= -2. Oracle magnitudes never rank testable pairs.
This is calibration: no D.6 (c) verdict comes from it (G.1).

Run from the repo root:  PYTHONPATH=results/m2/scratch uv run python results/m2/scratch/m2_oracle.py
"""
from __future__ import annotations

import itertools, json, time
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.presentation import decide
from flymon.brain.stimuli import design_odor_pair, present
from m2_probe import NPZ, READ_MS, SETTLE_MS, assign_channels, build_turns, pool_vocabulary, turn_odors

Z = {"A": (21.8293, 18.1032), "P": (40.6433, 24.0891)}
ACT_SEEDS = list(range(500, 508))
PROBE_SEEDS = list(range(600, 608))
STRENGTH = 0.35
ALPHAS = [0.2, 0.5, 0.8]   # G.4a
OUT = Path("results/m2/calibration/oracle.json")


def oracle_job(eng, pl, pops, comps, ro, odor_x: dict, odor_y: dict) -> dict:
    """One pair on one worker. Leaves the worker's weights reset (pool_jobs contract)."""
    t = eng.conn.type
    cells13, cells05 = np.flatnonzero(t == "MBON13"), np.flatnonzero(t == "MBON05")
    idx = np.concatenate([cells13, cells05]); n13 = len(cells13)
    kc = pops.kc
    try:
        pl.reset_weights(); pl.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(kc), float)
            for s in ACT_SEEDS:
                eng.reset(s); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
                present(eng, pops, odor, STRENGTH)
                eng.run(SETTLE_MS)
                fired += eng.run(READ_MS)[kc] > 0
            return fired / len(ACT_SEEDS)

        def probe():
            A, P = [], []
            for s in PROBE_SEEDS:
                c = decide(eng, pl, pops, [odor_x, odor_y], STRENGTH, s, SETTLE_MS, READ_MS, idx=idx)
                A.append(c[:, :n13].sum(1).tolist()); P.append(c[:, n13:].sum(1).tolist())
            return {"A": A, "P": P}

        fx, fy = activity(odor_x), activity(odor_y)
        w = eng.csc.w
        mb = pl.mb_local
        rew = np.isin(pl.post_mb, mb[comps["PAM08"].core])
        pun = np.isin(pl.post_mb, mb[comps["PPL105"].core])
        out = {"pre": probe(), "variants": {},
               "kc": {"x_active": int((fx > 0).sum()), "y_active": int((fy > 0).sum()),
                      "both_active": int(((fx > 0) & (fy > 0)).sum()),
                      "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))},
               "n_edges": {"reward": int(rew.sum()), "punish": int(pun.sum())}}
        pre = out["pre"]
        families = {"freq": lambda a: 1.0 - a * fx, "all": lambda a: np.where(fx > 0, 1.0 - a, 1.0)}
        for name, factor in families.items():
            fam = {"reward": {}, "punish": {}}
            for a_r in ALPHAS:                                   # G.4a: sweep reward depth, keep the best level
                wv = pl.w0.copy(); wv[rew] = pl.w0[rew] * factor(a_r)[pl.pre_kc[rew]]
                w[pl.edges] = wv
                R1 = probe()
                fam["reward"][str(a_r)] = {"R1": R1, "reward_level": dprime(dv(R1)),
                                           "w_frac": float((wv[rew] / pl.w0[rew]).mean())}
            a_star = max(ALPHAS, key=lambda a: (fam["reward"][str(a)]["reward_level"], -a))
            R1s = fam["reward"][str(a_star)]["R1"]
            for a_p in ALPHAS:                                   # then sweep punish depth on top of it
                wv = pl.w0.copy()
                wv[rew] = pl.w0[rew] * factor(a_star)[pl.pre_kc[rew]]
                wv[pun] = pl.w0[pun] * factor(a_p)[pl.pre_kc[pun]]
                w[pl.edges] = wv
                R2 = probe()
                fam["punish"][str(a_p)] = {"R2": R2, "punish_drop": dprime(dv(R2) - dv(R1s)),
                                           "w_frac": float((wv[pun] / pl.w0[pun]).mean())}
            a_pun = min(ALPHAS, key=lambda a: (fam["punish"][str(a)]["punish_drop"], a))
            fam["alpha_reward"], fam["alpha_punish"] = a_star, a_pun
            fam["best_reward_level"] = fam["reward"][str(a_star)]["reward_level"]
            fam["best_punish_drop"] = fam["punish"][str(a_pun)]["punish_drop"]
            fam["spill_P"] = spill(pre, R1s, "P")
            fam["spill_A"] = spill(R1s, fam["punish"][str(a_pun)]["R2"], "A")
            out["variants"][name] = fam
            w[pl.edges] = pl.w0
        return out
    finally:
        pl.reset_weights(); pl.set_enabled(True)


def dv(c):
    A = np.asarray(c["A"], float); P = np.asarray(c["P"], float)
    V = (A - Z["A"][0]) / Z["A"][1] - (P - Z["P"][0]) / Z["P"][1]
    return V[:, 0] - V[:, 1]


def dprime(x):
    x = np.asarray(x, float); sd = float(x.std(ddof=1)); m = float(x.mean())
    return (0.0 if m == 0 else float(np.copysign(np.inf, m))) if sd == 0 else m / sd


def spill(pre, post, cell):
    """(drop of Y) / (drop of X) in mean spikes for one readout cell; None when X did not drop."""
    a0, a1 = np.asarray(pre[cell], float).mean(0), np.asarray(post[cell], float).mean(0)
    dx, dy = a0[0] - a1[0], a0[1] - a1[1]
    return None if dx <= 0 else float(dy / dx)


def main() -> None:
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    st, mi, mon_types, move_types = pool_vocabulary()
    groups = [("my", mon_types), ("opp", mon_types), ("move", move_types),
              ("pow", ["lt60", "60to89", "ge90"]), ("myhp", ["low", "mid", "high"]), ("opphp", ["low", "mid", "high"])]
    channels = assign_channels(pops, groups)
    turns = build_turns(st, mi, 16)
    pairs, jobs = [], []
    for t in turns:
        od = turn_odors(pops, channels, t)
        for i, j in itertools.combinations(range(len(t["candidates"])), 2):
            pairs.append({"turn": t["turn"], "x": t["candidates"][i]["move"], "y": t["candidates"][j]["move"],
                          "split": "gate-candidate (odd)" if t["turn"] % 2 else "pilot-only (even)"})
            jobs.append(dict(odor_x=od[i], odor_y=od[j]))
    a, b = design_odor_pair(pops, k=8, seed=0)
    pairs.append({"turn": "design", "x": "odour b", "y": "odour a", "split": "protocol check"})
    jobs.append(dict(odor_x=b, odor_y=a))
    del conn
    print(f"{len(jobs)} oracle jobs")
    t0 = time.perf_counter()
    with FlyPool(NPZ, Params(), [{} for _ in range(16)], workers=16, timeout_s=7200.0) as pool:
        res = pool.run_jobs(oracle_job, jobs)
    print(f"oracle done in {time.perf_counter() - t0:.0f}s")

    rows = []
    for meta, r in zip(pairs, res):
        row = dict(meta, kc=r["kc"], d_pre=dprime(dv(r["pre"])), variants={})
        for name, v in r["variants"].items():
            row["variants"][name] = {
                "reward_level": v["best_reward_level"], "punish_drop": v["best_punish_drop"],
                "alpha_reward": v["alpha_reward"], "alpha_punish": v["alpha_punish"],
                "spill_P": v["spill_P"], "spill_A": v["spill_A"],
                "reward_curve": {a: v["reward"][a]["reward_level"] for a in v["reward"]},
                "punish_curve": {a: v["punish"][a]["punish_drop"] for a in v["punish"]}}
        row["testable"] = any(v["reward_level"] >= 2 and v["punish_drop"] <= -2 for v in row["variants"].values())
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rule": "G.4a: testable iff either family (best over alpha): reward_level >= 2 and punish_drop <= -2", "alphas": ALPHAS,
                               "act_seeds": ACT_SEEDS, "probe_seeds": PROBE_SEEDS, "raw": res, "rows": rows}, indent=1))

    def f(x):
        return "  None" if x is None else (f"{x:6.2f}" if np.isfinite(x) else f"{'+inf' if x > 0 else '-inf':>6}")
    print(f"\n{'turn':>6} {'X':<12} {'Y':<12} {'split':<20} {'KC J':>5} {'d_pre':>6} | "
          f"{'freq rew':>8} {'freq pun':>8} {'spillP':>6} | {'all rew':>7} {'all pun':>7} {'spillP':>6} | testable  (best over alpha)")
    for r in rows:
        fr, al = r["variants"]["freq"], r["variants"]["all"]
        print(f"{str(r['turn']):>6} {r['x']:<12} {r['y']:<12} {r['split']:<20} {r['kc']['jaccard']:5.2f} {f(r['d_pre'])} | "
              f"{f(fr['reward_level']):>8} {f(fr['punish_drop']):>8} {f(fr['spill_P'])} | "
              f"{f(al['reward_level']):>7} {f(al['punish_drop']):>7} {f(al['spill_P'])} | {r['testable']}")
    for split in ("gate-candidate (odd)", "pilot-only (even)"):
        sel = [r for r in rows if r["split"] == split]
        print(f"\n{split}: testable {sum(r['testable'] for r in sel)} / {len(sel)}")
    print(f"protocol check (design pair) testable: {rows[-1]['testable']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
