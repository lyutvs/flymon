# post-hoc exploration: re-aggregate G.12 even-turn raw data (no new simulation)
import json, numpy as np
Z = {"A": (21.8293, 18.1032), "P": (40.6433, 24.0891)}
def dv(c):
    A = np.asarray(c["A"], float); P = np.asarray(c["P"], float)
    V = (A - Z["A"][0]) / Z["A"][1] - (P - Z["P"][0]) / Z["P"][1]
    return V[:, 0] - V[:, 1]
def dprime(x):
    x = np.asarray(x, float); sd = float(x.std(ddof=1)); m = float(x.mean())
    return (0.0 if m == 0 else float(np.copysign(np.inf, m))) if sd == 0 else m / sd
pool = []
print("enc ax  n | test lvl>=2 chg>=2 | test&|pre|<.5 lvl&|pre|<.5 chg&|pre|<.5 | |pre|<.5 | turns(test)")
for e in ["E0", "E1", "E2", "E3"]:
    rows = json.load(open(f"results/m2/calibration/encoders/{e}_even.json"))["rows"]
    for ax in "ab":
        R = [r for r in rows if r["axis"] == ax]
        chk = sum(abs(dprime(dv(r["pre"])) - r["d_pre"]) < 1e-9 for r in R)
        for r in R:
            r["chg"] = max(dprime(dv(v["R1"]) - dv(r["pre"])) for v in r["reward"].values())
            r["lvl2"] = max(dprime(dv(v["R1"])) for v in r["reward"].values())
            assert abs(r["lvl2"] - r["best_reward_level"]) < 1e-9
            pool.append((e, ax, r["d_pre"], r["best_reward_level"] >= 2))
        nz = [r for r in R if abs(r["d_pre"]) < 0.5]
        print(f"{e}  {ax} {len(R):2d} | {sum(r['testable'] for r in R):4d} {sum(r['best_reward_level']>=2 for r in R):6d} {sum(r['chg']>=2 for r in R):6d} |"
              f" {sum(r['testable'] for r in nz):13d} {sum(r['best_reward_level']>=2 for r in nz):11d} {sum(r['chg']>=2 for r in nz):11d} | {len(nz):8d} | "
              f"{sorted({r['turn'] for r in R if r['testable']})} dpre-check {chk}/{len(R)}")
hi = [p for p in pool if p[2] > 0.5]; lo = [p for p in pool if p[2] < -0.5]
print("reward-level pass | d_pre>+0.5:", sum(p[3] for p in hi), "/", len(hi), " | d_pre<-0.5:", sum(p[3] for p in lo), "/", len(lo))
