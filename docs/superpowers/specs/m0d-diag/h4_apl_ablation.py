"""M0d H.4a.2: readout reactivity (the H.3 guard statistic: reference set + same-seed rest) on the three adopted
engines with APL's edges zeroed onto the readout pools' MBON types ("pools": MBON13, MBON18, MBON05, MBON21) or onto
the two readout types only ("readout": MBON13, MBON05), next to the unedited engine ("none", which must equal the H.3
guard). H.3 recorded the full APL -> MBON ablation (block "h3", records "apl_to_mbon_zero"). Diagnostic only: no oracle.

Why this exists. H.3a.9 (2) asks for the APL -> MBON ablated V next to the main V. Every ablation that includes
APL -> MBON05 releases MBON05 and silences MBON13 through the network, so the ablated V has no working A term.

    uv run python docs/superpowers/specs/m0d-diag/h4_apl_ablation.py [out.json]   # default results/m0d/diag/h4_apl_ablation.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import flymon.brain.h3_jobs as J
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import chunks
from flymon.brain.h3_rules import mbon_type_stats
from flymon.brain.h3_spec import SPEC, make_odors

_orig = J.apply_csc_edit


def _patched(eng, pops, edit):
    if edit[0] != "apl_to_types_zero":
        return _orig(eng, pops, edit)
    t = np.asarray(eng.conn.type).astype(str)
    src, tgt = J.edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
    is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
    is_t = np.isin(t, list(edit[1]))
    eng.csc.w[is_apl[src] & is_t[tgt]] = np.float32(0.0)
    import hashlib
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


J.apply_csc_edit = _patched      # module level: the spawned workers re-run this line when they import the script

TYPES = ("MBON13", "MBON18", "MBON05", "MBON21")
EDITS = {"none": None, "pools": ("apl_to_types_zero", TYPES), "readout": ("apl_to_types_zero", ("MBON13", "MBON05"))}


def params_from(d: dict) -> Params:
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "results/m0d/diag/h4_apl_ablation.json"
    h3 = json.load(open("results/summary/m0d.json"))["h3"]
    conn = Connectome.load("data/malecns.npz"); pops = Populations.from_connectome(conn)
    odors = make_odors(pops, SPEC.reference)
    seeds = sorted({int(s) for o in odors for s in o["seeds"]})
    res = {}
    with FlyPool("data/malecns.npz", Params(), [{} for _ in range(16)], workers=16, timeout_s=7200.0) as pool:
        for name in ("C0", "C1", "C3"):
            p = params_from(h3["combos"][name]["adopted"]["params"])
            for label, edit in EDITS.items():
                t0 = time.time()
                common = dict(params=p, strength=SPEC.strength, settle_ms=SPEC.reference_window.settle_ms,
                              read_steps=int(SPEC.reference_window.read_ms), quantiles=tuple(SPEC.apl_v_quantiles),
                              callout=tuple(SPEC.callout_types), csc_edit=edit)
                ref = [r for part in pool.run_jobs(J.reference_job, [dict(common, odors=c) for c in chunks(odors, 16)]) for r in part]
                rc = dict(params=p, settle_ms=SPEC.reference_window.settle_ms, read_steps=int(SPEC.reference_window.read_ms),
                          csc_edit=edit)
                rest = [r for part in pool.run_jobs(J.rest_job, [dict(rc, seeds=c) for c in chunks(seeds, 16)]) for r in part]
                rb = {r["seed"]: r for r in rest}
                stats = {t: mbon_type_stats(ref, rb, t, 5.0, 0.25) for t in TYPES}
                res[f"{name}/{label}"] = dict(stats=stats, median_kc_pct=float(np.median([r["kc_active_frac"] for r in ref]) * 100),
                                              median_mv=float(np.median([r["apl_v_mean"] for r in ref])))
                print(f"{name}/{label} {time.time() - t0:.0f}s " + "  ".join(
                    f"{t} ({s['median_delta']:.1f}, {s['zero_share']:.3f}, {'P' if s['passes'] else '-'})" for t, s in stats.items()),
                      flush=True)
                json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
