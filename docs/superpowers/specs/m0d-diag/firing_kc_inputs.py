"""Phase 1: what drives the KCs that fire under single-glomerulus stimulation? Descriptive only."""
import sys, collections
import numpy as np
sys.path.insert(0, "docs/superpowers/specs/m2-calibration-g"); sys.path.insert(0, sys.argv[1])
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, apply_sign_override
from flymon.brain.fly_pool import FlyPool
from m2_probe import EXCLUDE, NPZ
from spread_trace import job

GLOMS = ["ORN_DA2", "ORN_VM5v", "ORN_DP1m", "ORN_DA3"]

if __name__ == "__main__":
    p = Params(); conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    rts = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts]))
    with FlyPool(NPZ, Params(), [{} for _ in range(4)], workers=4) as pool:
        res = pool.run_jobs(job, [dict(rtype=g, seed=200, c_norm=c_norm) for g in GLOMS])
        rest = pool.run_jobs(job, [dict(rtype="ORN_DA3", seed=200, c_norm=0.0)])[0]      # no odour at all
    N = conn.N; t = np.asarray(conn.type).astype(str); sign, _ = apply_sign_override(conn, p)
    kc_ids = np.asarray(pops.kc); pn_ids = np.asarray(pops.alpn)
    is_pn = np.zeros(N, bool); is_pn[pn_ids] = True
    mk = is_pn[conn.pre] & np.isin(conn.post, kc_ids)
    pn_in = np.zeros(N); np.add.at(pn_in, conn.post[mk], conn.w[mk])
    keep = (conn.w >= p.min_weight) & (sign[conn.pre] != 0)
    print(f"rest (no odour): KC spikes {int(np.asarray(rest)[kc_ids].sum())}, firing KCs {int((np.asarray(rest)[kc_ids] > 0).sum())}")
    for g, c in zip(GLOMS, res):
        c = np.asarray(c); fk = kc_ids[c[kc_ids] > 0]
        if not len(fk):
            print(f"\n{g}: no firing KCs"); continue
        zero_pn = fk[pn_in[fk] == 0]
        e = keep & np.isin(conn.post, fk) & (c[conn.pre] > 0)
        by = collections.Counter()
        for pre_t, s in zip(t[conn.pre[e]], (c[conn.pre[e]] * conn.w[e] * sign[conn.pre[e]]).astype(float)):
            key = "PN" if pre_t.endswith("PN") or "PN" in pre_t else pre_t
            by[key] += s
        print(f"\n{g}: firing KCs {len(fk)} (spikes {int(c[fk].sum())}); with zero PN input {len(zero_pn)}; "
              f"KC types {collections.Counter(t[fk]).most_common(4)}")
        print(f"  spikes from zero-PN-input KCs: {int(c[zero_pn].sum())}; median spikes per firing KC {np.median(c[fk]):.0f}")
        print(f"  active input to firing KCs by presynaptic type (spikes*syn*sign): {[(k, round(v)) for k, v in by.most_common(6)]}")
        print(f"  same KCs at rest: firing {int((np.asarray(rest)[fk] > 0).sum())}, spikes {int(np.asarray(rest)[fk].sum())}")
