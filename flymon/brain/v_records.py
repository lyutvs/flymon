"""V's numbers that no R / S / T / U module returns — records and the values V's gates read.
- csc_facts (V.1, V.3 2, V.9.5 P1-3): the combined edit's static facts on a fresh engine (no simulation): u_measure's
  apply_u_edit with L_V on C3's CSC — its CSC sha, the APL->MBON05 mask count, the chain-entry pair counts, the edges
  whose bits changed (by source / target type) and the MBON05->APL edges before and after (count, bytes, weights).
- full_row_diffs (V.3 2): V's combined reference / rest rows against U's entry_f0 rows, every field bit for bit (the
  same job on the same edit: odour, seed, every type count, KC fractions, the mechanism reads, CSC, edge labels).
- kc_input_record (V.3 3, V.9.5 P2-7): per engine the per-odour KC median over the strength seeds (e_rules'
  odour_activity, as R's gate ① and T.9.1), the per-odour seed range, edges, CSC, counts; kc_u_diffs: V's unedited
  medians against U's per_odour_none on the shared odours, bit for bit.
- side_record (V.6, V.9.2): MBON05 mean ratio (L_V / unedited) and σ_V / σ_h4 per readout.
- ratio_two_z (V.9.2): s_records.ratio with L read on z_L and C on z_C (ℓ_L(z_V) / ℓ_C(h4 z)), same seeds, same
  paired bootstrap."""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from ..agent import e_rules
from .h3_store import canonical
from .n_rules import _ci
from .o_rules import boot_weights
from .s_records import vectors


def csc_facts(conn, pops, params, spec) -> dict:
    from .engine_cpu import Engine
    from .h3_jobs import edge_sources
    from .u_measure import apply_u_edit
    eng = Engine(conn, pops, params, seed=0)
    w0 = eng.csc.w.copy()
    src, tgt = edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
    t = np.asarray(eng.conn.type).astype(str)
    is_apl = np.zeros(eng.N, bool)
    is_apl[np.asarray(pops.apl, np.int64)] = True
    m5a = (t[src] == spec.p_type) & is_apl[tgt]
    sha0 = hashlib.sha256(w0.tobytes()).hexdigest()
    sha, n_apl, blocks = apply_u_edit(eng, pops, spec.lever_edit, spec.p_type)
    changed = np.flatnonzero(eng.csc.w.view(np.uint32) != w0.view(np.uint32))
    pairs = Counter(f"{'APL' if is_apl[src[i]] else t[src[i]]}->{t[tgt[i]]}" for i in changed)
    return dict(sha_none=sha0, csc_sha256=sha, apl_edges=int(n_apl), block_edges=dict(sorted(blocks.items())),
                changed=int(changed.size), changed_pairs=dict(sorted(pairs.items())),
                mbon05_apl=dict(n=int(m5a.sum()), before=[float(x) for x in w0[m5a]],
                                after=[float(x) for x in eng.csc.w[m5a]],
                                same_bits=bool(np.array_equal(eng.csc.w[m5a].view(np.uint32),
                                                              w0[m5a].view(np.uint32)))))


def csc_reasons(facts: dict, spec) -> list:
    """V.1: INVALID unless 2 APL->MBON05 edges, the chain entry 7 / 2 / 2, 13 changed edges, CSC 2d359b8b…, and the
    MBON05->APL edges untouched (2 edges, bytes equal before and after, weights within 0.001 of V.0's −7.865 /
    −12.929)."""
    bad = []
    if facts["apl_edges"] != spec.lever_edges:
        bad.append(f"APL->MBON05 edges {facts['apl_edges']}, declared {spec.lever_edges}")
    if facts["block_edges"] != dict(sorted(spec.contrast_declared()["chain_entry"].items())):
        bad.append(f"chain entry edges {facts['block_edges']}, declared {spec.contrast_declared()['chain_entry']}")
    if facts["changed"] != spec.n_lever_edges_total:
        bad.append(f"{facts['changed']} edges changed, declared {spec.n_lever_edges_total}")
    if facts["csc_sha256"] != spec.sha_combined:
        bad.append(f"CSC {facts['csc_sha256']}, declared {spec.sha_combined}")
    if facts["sha_none"] != spec.sha_none:
        bad.append(f"unedited CSC {facts['sha_none']}, declared {spec.sha_none}")
    m = facts["mbon05_apl"]
    near = len(m["after"]) == len(spec.mbon05_apl_weights) and all(
        abs(x - y) <= spec.mbon05_apl_weight_tol for x, y in zip(sorted(m["after"]), sorted(spec.mbon05_apl_weights)))
    if m["n"] != spec.mbon05_apl_edges or not m["same_bits"] or not near:
        bad.append(f"MBON05->APL {m}, declared {spec.mbon05_apl_edges} edges {spec.mbon05_apl_weights} untouched")
    return bad


def full_row_diffs(v_rows: list, u_rows: list, label: str) -> list:
    if len(v_rows) != len(u_rows):
        return [f"{label}: {len(v_rows)} rows, U {len(u_rows)}"]
    bad = [f"{u.get('odor', 'rest')}/{u['seed']}" for v, u in zip(v_rows, u_rows) if canonical(v) != canonical(u)]
    return [f"{label}: {len(bad)} row(s) differ: {bad[:3]}"] if bad else []


def _seed_range(act: dict) -> dict:
    return {o: [float(min(v["frac"])), float(max(v["frac"]))] for o, v in act.items()}


def kc_input_record(act: dict, cap: dict, cap_hz: float, spec) -> dict:
    """act = {"none": activity, "lever": activity} over the same odours (RMeasurer.activity's values)."""
    lo, hi = spec.valid_band
    out = {}
    for e, a in act.items():
        per = e_rules.odour_activity({o: a[o]["frac"] for o in a})
        out[e] = dict(per_odour=per, seed_range=_seed_range(a), n_odours=len(per),
                      n_seeds=sorted({len(v["frac"]) for v in a.values()}),
                      outside=sorted(o for o, v in per.items() if not lo <= v <= hi),
                      min=min(per.values()) if per else None, max=max(per.values()) if per else None,
                      edit_edges=sorted({int(x) for v in a.values() for x in v["edit_edges"]}),
                      csc_sha256=sorted({s for v in a.values() for s in v["csc_sha256"]}))
    return dict(out, cap=dict(per_odour_hz=dict(sorted(cap.items())), cap_hz=float(cap_hz),
                              all_under=all(v <= cap_hz for v in cap.values())), band=list(spec.valid_band))


def kc_u_diffs(per_none: dict, u_none: dict) -> dict:
    shared = sorted(set(per_none) & set(u_none))
    bad = [o for o in shared if per_none[o] != u_none[o]]
    return dict(n_shared=len(shared), shared=shared, differ=bad,
                diffs=[f"{o} {per_none[o]!r} ≠ U {u_none[o]!r}" for o in bad[:5]])


def side_record(side_v: dict, side_none: dict, z_h4: dict, readout: dict) -> dict:
    p = readout["P"]
    m0 = side_none["mech"]["types"][p]["mean_stim"]
    z = side_v.get("z")
    return dict(p_mean_ratio=(side_v["mech"]["types"][p]["mean_stim"] / m0) if m0 else None,
                sd_ratio=None if z is None else {k: float(z[k][1]) / float(z_h4[k][1]) for k in z_h4})


def ratio_two_z(rows_l: list, rows_c: list, z_l: dict, z_c: dict, spec) -> dict:
    """s_records.ratio with L read on z_l and C on z_c; ValueError when L and C do not hold the same seeds."""
    vl, vc = vectors(rows_l, z_l, spec.p), vectors(rows_c, z_c, spec.p_c)
    o = spec.p.o
    out = {}
    for d in spec.p.directions:
        if vl[d]["seeds"] != vc[d]["seeds"]:
            raise ValueError(f"direction {d}: L and C must hold the same seeds in the same order")
        dl_l, dl_c = np.asarray(vl[d]["dl"], float), np.asarray(vc[d]["dl"], float)
        W = boot_weights(dl_l.size, o.boot_draws, o.boot_seed)
        bl, bc = W @ dl_l, W @ dl_c
        out[d] = dict(ell_L=float(dl_l.mean()), ell_C=float(dl_c.mean()), ratio=float(dl_l.mean() / dl_c.mean()),
                      ratio_ci=_ci(bl / bc, o.ci_level), n=int(dl_l.size), seeds=list(vl[d]["seeds"]))
    return out
