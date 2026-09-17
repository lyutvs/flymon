"""Phase 1 evidence at the PN(g) and KC boundaries for each glomerulus, all51 protocol (seed 200). No verdict."""
import json, sys
import numpy as np
sys.path.insert(0, "docs/superpowers/specs/m2-calibration-g")
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import present
from m2_probe import EXCLUDE, NPZ, READ_MS, SETTLE_MS

def job(eng, pl, pops, comps, ro, rtype, seed, c_norm):
    pl.reset_weights(); pl.set_enabled(False)
    eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
    present(eng, pops, {rtype: c_norm / len(pops.receptor_types[rtype])}, 0.35)
    eng.run(SETTLE_MS); c = eng.run(READ_MS); pl.set_enabled(True)
    return {"pn": c[pops.alpn].tolist(), "kc": c[pops.kc].tolist(), "apl": int(c[pops.apl].sum())}

if __name__ == "__main__":
    p = Params(); conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    rts = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts]))
    with FlyPool(NPZ, Params(), [{} for _ in range(16)], workers=16) as pool:
        res = pool.run_jobs(job, [dict(rtype=r, seed=200, c_norm=c_norm) for r in rts])
    N = conn.N; pn_ids = np.asarray(pops.alpn); kc_ids = np.asarray(pops.kc)
    is_pn = np.zeros(N, bool); is_pn[pn_ids] = True; is_kc = np.zeros(N, bool); is_kc[kc_ids] = True
    orn_of = np.full(N, -1); allr = sorted(pops.receptor_types)
    for i, t in enumerate(allr): orn_of[pops.receptor_types[t]] = i
    m = (orn_of[conn.pre] >= 0) & is_pn[conn.post]; pos = np.full(N, -1); pos[pn_ids] = np.arange(len(pn_ids))
    M = np.zeros((len(pn_ids), len(allr))); np.add.at(M, (pos[conn.post[m]], orn_of[conn.pre[m]]), conn.w[m])
    tot = M.sum(1); best = M.argmax(1); uni = (M.max(1) / np.maximum(tot, 1) >= 0.8) & (tot >= 20)
    kpos = np.full(N, -1); kpos[kc_ids] = np.arange(len(kc_ids))
    keep = conn.w >= p.min_weight
    mk = is_pn[conn.pre] & is_kc[conn.post]
    pn_in = np.zeros(N); np.add.at(pn_in, conn.post[mk], conn.w[mk])
    med = np.median(pn_in[kc_ids]); vth_rel = p.kc_thresh * np.clip(pn_in[kc_ids] / med, *p.kc_norm_clip)   # threshold / v_thresh
    rows = []
    for r, t in zip(res, rts):
        gi = allr.index(t)
        own = uni & (best == gi)
        pn_c = np.asarray(r["pn"]); kc_c = np.asarray(r["kc"])
        own_rate = pn_c[own].mean() / (READ_MS / 1000) if own.any() else 0.0
        other_pn = pn_c[~own].sum()
        e = mk & keep & np.isin(conn.pre, pn_ids[own])
        syn_g = np.bincount(kpos[conn.post[e]], weights=conn.w[e], minlength=len(kc_ids))
        u = syn_g / vth_rel                                   # synapses from g per unit of relative threshold
        firing = kc_c > 0
        rows.append({"g": t, "kc_spikes": int(kc_c.sum()), "kc_on": int(firing.sum()), "own_pn_n": int(own.sum()),
                     "own_pn_hz": float(own_rate), "other_pn_spikes": int(other_pn), "apl": r["apl"],
                     "u_max": float(u.max()), "u_top5": float(np.sort(u)[-5:].mean()), "n_u_ge_30": int((u >= 30).sum()),
                     "firing_with_syn_g": int((firing & (syn_g > 0)).sum()), "firing_without_syn_g": int((firing & (syn_g == 0)).sum()),
                     "median_u_of_firing": float(np.median(u[firing])) if firing.any() else None})
    def lg(x): return np.log10(np.maximum(np.asarray(x, float), 1.0))
    k = lg([x["kc_spikes"] for x in rows])
    for f in ("own_pn_hz", "own_pn_n", "other_pn_spikes", "u_max", "u_top5", "n_u_ge_30"):
        v = np.asarray([x[f] for x in rows], float)
        print(f"corr(log KC, log {f:<16}) {np.corrcoef(k, lg(v + (1 if f.startswith('n_') else 0)))[0,1]:+.2f}")
    drive = lg([x["own_pn_hz"] * x["u_max"] for x in rows])
    print(f"corr(log KC, log own_pn_hz*u_max) {np.corrcoef(k, drive)[0,1]:+.2f}")
    print(f"KC spikes from KCs without synapses from g's uniPNs: {sum(x['firing_without_syn_g'] for x in rows)} cells over all glomeruli; with: {sum(x['firing_with_syn_g'] for x in rows)}")
    cols = ["g", "kc_spikes", "kc_on", "own_pn_n", "own_pn_hz", "other_pn_spikes", "u_max", "n_u_ge_30", "firing_with_syn_g", "firing_without_syn_g", "apl"]
    rows.sort(key=lambda x: x["kc_spikes"])
    print(" | ".join(cols))
    for x in rows[:8] + rows[-8:]:
        print(" | ".join(f"{x[c]:.1f}" if isinstance(x[c], float) else str(x[c]) for c in cols))
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
