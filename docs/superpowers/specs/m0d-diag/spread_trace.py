"""Phase 1: trace AL spread for a few glomeruli (all51 protocol, seed 200). Descriptive only."""
import json, sys, collections
import numpy as np
sys.path.insert(0, "docs/superpowers/specs/m2-calibration-g")
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, apply_sign_override
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import present
from m2_probe import EXCLUDE, NPZ, READ_MS, SETTLE_MS

GLOMS = ["ORN_DP1m", "ORN_DC1", "ORN_DL2d", "ORN_DA2", "ORN_VM5v", "ORN_VA1d", "ORN_DA3"]

def job(eng, pl, pops, comps, ro, rtype, seed, c_norm):
    pl.reset_weights(); pl.set_enabled(False)
    eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
    present(eng, pops, {rtype: c_norm / len(pops.receptor_types[rtype])}, 0.35)
    eng.run(SETTLE_MS); c = eng.run(READ_MS); pl.set_enabled(True)
    return c.astype(np.int32)

if __name__ == "__main__":
    p = Params(); conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    rts = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts]))
    with FlyPool(NPZ, Params(), [{} for _ in range(7)], workers=7) as pool:
        res = pool.run_jobs(job, [dict(rtype=g, seed=200, c_norm=c_norm) for g in GLOMS])
    N = conn.N; t = np.asarray(conn.type).astype(str); cls = np.asarray(conn.cls if hasattr(conn, "cls") else conn.type).astype(str)
    sign, _ = apply_sign_override(conn, p)
    pn_ids = np.asarray(pops.alpn); kc_ids = np.asarray(pops.kc)
    is_pn = np.zeros(N, bool); is_pn[pn_ids] = True; is_kc = np.zeros(N, bool); is_kc[kc_ids] = True
    orn_of = np.full(N, -1); allr = sorted(pops.receptor_types)
    for i, r in enumerate(allr): orn_of[pops.receptor_types[r]] = i
    m = (orn_of[conn.pre] >= 0) & is_pn[conn.post]; pos = np.full(N, -1); pos[pn_ids] = np.arange(len(pn_ids))
    M = np.zeros((len(pn_ids), len(allr))); np.add.at(M, (pos[conn.post[m]], orn_of[conn.pre[m]]), conn.w[m])
    tot = M.sum(1); best = M.argmax(1); uni = (M.max(1) / np.maximum(tot, 1) >= 0.8) & (tot >= 20)
    keep = (conn.w >= p.min_weight) & (sign[conn.pre] != 0)
    def group(i):
        if orn_of[i] >= 0: return "ORN"
        if is_pn[i]: return "PN"
        if is_kc[i]: return "KC"
        if "LN" in t[i]: return "LN(" + ("+" if sign[i] > 0 else "-") + ")"
        return "other(" + ("+" if sign[i] > 0 else "-") + ")"
    grp = np.array([group(i) for i in range(N)])
    for g, c in zip(GLOMS, res):
        c = np.asarray(c); gi = allr.index(g)
        own = uni & (best == gi); multi_g = (~uni) & (M[:, gi] > 0); other_uni = uni & (best != gi)
        pnc = c[pn_ids]
        cat = {"uni_own": int(pnc[own].sum()), "uni_other": int(pnc[other_uni].sum()), "multi_with_g": int(pnc[multi_g].sum()),
               "multi_without_g": int(pnc[(~uni) & (M[:, gi] == 0)].sum())}
        # input to recruited other-glomerulus uniPNs, by presynaptic group: sum spikes * w * sign
        rec = pn_ids[other_uni & (pnc > 0)]
        e = keep & np.isin(conn.post, rec)
        contrib = collections.Counter()
        for gr, s in zip(grp[conn.pre[e]], (c[conn.pre[e]] * conn.w[e] * sign[conn.pre[e]]).astype(float)):
            contrib[gr] += s
        # input to firing KCs by PN category
        fk = kc_ids[c[kc_ids] > 0]
        ek = keep & np.isin(conn.post, fk) & is_pn[conn.pre]
        pre = conn.pre[ek]; val = c[pre] * conn.w[ek]
        pcat = np.where(np.isin(pre, pn_ids[own]), "uni_own", np.where(np.isin(pre, pn_ids[other_uni]), "uni_other",
                        np.where(np.isin(pre, pn_ids[multi_g]), "multi_with_g", "multi_without_g")))
        kin = {k: float(val[pcat == k].sum()) for k in ("uni_own", "uni_other", "multi_with_g", "multi_without_g")}
        top_rec = collections.Counter({t[pn_ids[i]]: int(pnc[i]) for i in np.flatnonzero(other_uni & (pnc > 0))}).most_common(5)
        print(f"\n{g}: KC spikes {int(c[kc_ids].sum())}, firing KCs {len(fk)}")
        print(f"  PN spikes by category {cat}; recruited other-glom uniPNs {len(rec)}; top {top_rec}")
        print(f"  input to recruited uniPNs by group (spikes*syn*sign): { {k: round(v) for k, v in contrib.most_common()} }")
        print(f"  PN input to firing KCs by category (spikes*syn): { {k: round(v) for k, v in kin.items()} }")
