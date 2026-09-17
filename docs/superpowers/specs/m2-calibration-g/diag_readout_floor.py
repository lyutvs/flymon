"""Scratch diagnostic (not a verdict): naive MBON13/MBON05 counts and KC activity, graded vs spiking."""
import json, numpy as np
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.presentation import decide
from m2_engine_probe import configure, present_kc
from m2_encoder_compare import build_encoder, pairs_for
from m2_probe import NPZ, READ_MS, SETTLE_MS, build_turns, pool_vocabulary

SEEDS = [500, 501, 502, 503]

def diag_job(eng, pl, pops, comps, ro, odor, mode, kc_thresh):
    configure(eng, pops, mode, kc_thresh)
    t = eng.conn.type
    idx13, idx05 = np.flatnonzero(t == "MBON13"), np.flatnonzero(t == "MBON05")
    try:
        pl.reset_weights(); pl.set_enabled(False)
        out = {"kc": [], "m13": [], "m05": []}
        for s in SEEDS:
            o = present_kc(eng, pl, pops, odor, s); out["kc"].append(float((o["read"] > 0).mean()))
            c = decide(eng, pl, pops, [odor], 0.35, s, SETTLE_MS, READ_MS, idx=np.concatenate([idx13, idx05]))[0]
            out["m13"].append(int(c[:len(idx13)].sum())); out["m05"].append(int(c[len(idx13):].sum()))
        return out
    finally:
        pl.reset_weights(); pl.set_enabled(True)

if __name__ == "__main__":
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    st, mi, mt, vt = pool_vocabulary()
    turns = [t for t in build_turns(st, mi, 16) if t["turn"] % 2 == 0]
    pairs = pairs_for("E0", pops, build_encoder("E0", pops, mt, vt), turns, st)
    odors = list({json.dumps(sorted(p["odor_x"].items())): p["odor_x"] for p in pairs}.values())[::4][:6]
    res = {}
    for name, mode, k in [("G", "graded", 1.5), ("S-1.5", "spiking", 1.5), ("S-2.25", "spiking", 2.25)]:
        with FlyPool(NPZ, Params(), [{} for _ in range(6)], workers=6, timeout_s=3600.0) as pool:
            r = pool.run_jobs(diag_job, [dict(odor=o, mode=mode, kc_thresh=k) for o in odors])
        kc = np.array([x for o in r for x in o["kc"]]); m13 = np.array([x for o in r for x in o["m13"]]); m05 = np.array([x for o in r for x in o["m05"]])
        res[name] = {"kc_median_pct": round(100 * float(np.median(kc)), 2), "mbon13_mean": float(m13.mean()), "mbon13_zero": float((m13 == 0).mean()),
                     "mbon05_mean": float(m05.mean()), "mbon05_zero": float((m05 == 0).mean())}
        print(name, res[name], flush=True)
    json.dump(res, open("results/m2/calibration/engine_probe/diag_readout_floor.json", "w"), indent=1)
