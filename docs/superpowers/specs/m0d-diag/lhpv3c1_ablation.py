"""Phase 3 minimal test: does removing LHPV3c1's out-edges collapse the all51 KC-drive disparity? Diagnostic only."""
import json, sys
import numpy as np
sys.path.insert(0, "docs/superpowers/specs/m2-calibration-g")
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import present
from m2_probe import EXCLUDE, NPZ, READ_MS, SETTLE_MS

TARGET = "LHPV3c1"

def job(eng, pl, pops, comps, ro, rtype, seed, c_norm, ablate):
    if ablate and not getattr(eng, "_ablated", False):
        idx = np.flatnonzero(np.asarray(eng.conn.type).astype(str) == TARGET)
        for i in idx:
            eng.csc.w[eng.csc.ptr[i]:eng.csc.ptr[i + 1]] = 0.0
        eng._ablated = True
    if not ablate and getattr(eng, "_ablated", False):
        raise RuntimeError("baseline job on an ablated worker")
    pl.reset_weights(); pl.set_enabled(False)
    eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
    present(eng, pops, {rtype: c_norm / len(pops.receptor_types[rtype])}, 0.35)
    eng.run(SETTLE_MS); c = eng.run(READ_MS); pl.set_enabled(True)
    lh = np.flatnonzero(np.asarray(eng.conn.type).astype(str) == TARGET)
    return {"g": rtype, "seed": seed, "kc": int(c[pops.kc].sum()), "kc_on": int((c[pops.kc] > 0).sum()),
            "pn": int(c[pops.alpn].sum()), "lhpv3c1": int(c[lh].sum()), "apl": int(c[pops.apl].sum())}

def stats(rows, label):
    gs = sorted({r["g"] for r in rows})
    K = np.array([np.mean([r["kc"] for r in rows if r["g"] == g]) for g in gs])
    L = np.array([np.mean([r["lhpv3c1"] for r in rows if r["g"] == g]) for g in gs])
    lo = max(K.min(), 1)
    band = max(sum((K >= t) & (K <= 8 * t)) for t in (100, 200, 400, 600))
    print(f"{label}: KC min {K.min():.1f} median {np.median(K):.0f} max {K.max():.0f}  max/min {K.max()/lo:.0f}x  "
          f"CV {K.std()/K.mean():.2f}  log10 var {np.var(np.log10(np.maximum(K, 1))):.3f}  best 8x band {band}/51  "
          f"KC>=200: {int((K >= 200).sum())}  LHPV3c1 spikes mean {L.mean():.1f}  corr(logKC, log(LHPV3c1+1)) {np.corrcoef(np.log10(np.maximum(K,1)), np.log10(L+1))[0,1]:+.2f}")
    return dict(zip(gs, K.tolist()))

if __name__ == "__main__":
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    rts = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts]))
    del conn
    jobs = lambda ab: [dict(rtype=r, seed=200 + i, c_norm=c_norm, ablate=ab) for r in rts for i in range(2)]
    with FlyPool(NPZ, Params(), [{} for _ in range(16)], workers=16) as pool:
        base = pool.run_jobs(job, jobs(False))
    with FlyPool(NPZ, Params(), [{} for _ in range(16)], workers=16) as pool:
        abl = pool.run_jobs(job, jobs(True))
    kb = stats(base, "baseline "); ka = stats(abl, "ablated  ")
    ref = {r[0]: r[4] for r in json.load(open("results/m2/all51_drive.json"))["rows"]}
    print("baseline vs stored all51 (4 seeds) corr log:", round(float(np.corrcoef(np.log10(np.maximum([kb[g] for g in kb], 1)), np.log10(np.maximum([ref[g] for g in kb], 1)))[0, 1]), 3))
    json.dump({"baseline": base, "ablated": abl}, open(sys.argv[1], "w"), indent=1)
