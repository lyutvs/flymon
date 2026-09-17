"""Phase 1 evidence: where does the across-glomerulus KC-drive disparity (all51) arise? No simulation."""
import json, sys
import numpy as np
sys.path.insert(0, "docs/superpowers/specs/m2-calibration-g")
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.circuits import Populations
from m2_probe import NPZ

p = Params()
conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
rows = {r[0]: r for r in json.load(open("results/m2/all51_drive.json"))["rows"]}   # rtype, nORN, orn, pn, kc, kc_on, sd
N = conn.N
is_pn = np.zeros(N, bool); is_pn[pops.alpn] = True
is_kc = np.zeros(N, bool); is_kc[pops.kc] = True
orn_of = np.full(N, -1); rtypes = sorted(pops.receptor_types)
for i, t in enumerate(rtypes): orn_of[pops.receptor_types[t]] = i

# PN glomerulus by ORN input (all synapse counts, before pruning)
m = (orn_of[conn.pre] >= 0) & is_pn[conn.post]
pn_ids = np.asarray(pops.alpn); pos = np.full(N, -1); pos[pn_ids] = np.arange(len(pn_ids))
M = np.zeros((len(pn_ids), len(rtypes)))
np.add.at(M, (pos[conn.post[m]], orn_of[conn.pre[m]]), conn.w[m])
tot = M.sum(1); best = M.argmax(1); frac = np.where(tot > 0, M.max(1) / np.maximum(tot, 1), 0)
uni = (frac >= 0.8) & (tot >= 20)
print(f"ALPN {len(pn_ids)}: with ORN input {int((tot>0).sum())}, uniglomerular (>=80%, >=20 syn) {int(uni.sum())}")

# KC thresholds as the engine builds them
pn_in = np.zeros(N); mk = is_pn[conn.pre] & is_kc[conn.post]; np.add.at(pn_in, conn.post[mk], conn.w[mk])
med = np.median(pn_in[pops.kc]); ratio = pn_in / med; lo, hi = p.kc_norm_clip

out = []
for gi, t in enumerate(rtypes):
    if t not in rows: continue
    r = rows[t]
    pns = pn_ids[uni & (best == gi)]
    orn_pn_syn = float(M[uni & (best == gi), gi].sum())
    e = np.isin(conn.pre, pns) & is_kc[conn.post]
    w_all = conn.w[e]; kept = w_all >= p.min_weight
    posts = conn.post[e]
    kc_conn = np.bincount(posts[kept], minlength=N)[pops.kc]            # distinct PN connections (claw proxy) from this glomerulus
    kc_syn = np.bincount(posts[kept], weights=w_all[kept], minlength=N)[pops.kc]
    recv = kc_conn > 0
    rkc = np.asarray(pops.kc)[recv]
    out.append({"g": t, "nORN": r[1], "orn": r[2], "pn": r[3], "kc": r[4], "kc_on": r[5],
                "n_pn": int(len(pns)), "orn_pn_syn": orn_pn_syn,
                "pnkc_syn_kept": float(w_all[kept].sum()), "pnkc_syn_pruned": float(w_all[~kept].sum()),
                "kc_recv": int(recv.sum()), "kc_ge2": int((kc_conn >= 2).sum()), "kc_ge3": int((kc_conn >= 3).sum()),
                "kc_syn_per_recv": float(kc_syn[recv].mean()) if recv.any() else 0.0,
                "max_syn_frac_of_pn_in": float((kc_syn[recv] / pn_in[rkc]).max()) if recv.any() else 0.0,
                "clip_lo_frac": float((ratio[rkc] <= lo).mean()) if recv.any() else 0.0,
                "clip_hi_frac": float((ratio[rkc] >= hi).mean()) if recv.any() else 0.0})

def lg(x): return np.log10(np.maximum(np.asarray(x, float), 1.0))
kc, pn, orn = lg([o["kc"] for o in out]), lg([o["pn"] for o in out]), lg([o["orn"] for o in out])
print(f"glomeruli {len(out)}; KC max/min {10**(kc.max()-kc.min()):.0f}x")
print(f"var log10: ORN {orn.var():.3f}  PN {pn.var():.3f}  KC {kc.var():.3f}  PN->KC gain {np.var(kc-pn):.3f}  cov(PN, gain) {np.cov(pn, kc-pn)[0,1]:.3f}")
feat = {k: lg([o[k] for o in out]) for k in ("n_pn", "orn_pn_syn", "pnkc_syn_kept", "kc_recv", "kc_ge2", "kc_ge3", "kc_syn_per_recv")}
for k, v in feat.items():
    print(f"  corr(log KC, log {k:<16}) = {np.corrcoef(kc, v)[0,1]:+.2f}   corr(log PN, log {k}) = {np.corrcoef(pn, v)[0,1]:+.2f}")
print("  corr(log KC, max_syn_frac_of_pn_in) = %+.2f" % np.corrcoef(kc, [o["max_syn_frac_of_pn_in"] for o in out])[0,1])
print("  corr(log KC, clip_lo_frac) = %+.2f, clip_hi_frac = %+.2f" % (np.corrcoef(kc, [o["clip_lo_frac"] for o in out])[0,1], np.corrcoef(kc, [o["clip_hi_frac"] for o in out])[0,1]))
print(f"pruned PN->KC synapse share (w<{p.min_weight}): {sum(o['pnkc_syn_pruned'] for o in out)/sum(o['pnkc_syn_pruned']+o['pnkc_syn_kept'] for o in out):.3f}")
print(f"KC thresholds: clipped low {(ratio[pops.kc]<=lo).mean():.3f}, high {(ratio[pops.kc]>=hi).mean():.3f}")
o_sorted = sorted(out, key=lambda o: o["kc"])
cols = ["g", "nORN", "pn", "kc", "n_pn", "orn_pn_syn", "pnkc_syn_kept", "kc_recv", "kc_ge2", "kc_ge3", "kc_syn_per_recv", "max_syn_frac_of_pn_in"]
print(" | ".join(cols))
for o in o_sorted[:8] + o_sorted[-8:]:
    print(" | ".join(str(round(o[c], 2)) if isinstance(o[c], float) else str(o[c]) for c in cols))
json.dump(out, open(sys.argv[1], "w"), indent=1)
