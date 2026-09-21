"""FlyPool worker jobs for the M0d H.3 runner (spec appendix H.3a).

Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them.
Every job builds (or reuses) its own Engine from the `Params` it is given; the worker's default engine only lends
its connectome. Plasticity is never touched. The presentation protocols are the committed diagnostics' call
sequences (reset / clear_drive / present / run(settle) / step x read), so the numbers they recorded reproduce:
reference set = `apl_input_scale_sweep.job` + `readout_floor_guard.job_stim`, rest = `readout_floor_guard.job_rest`,
design pair = `d4_margin_protocol.job_design`, baseline = `guard_after_baseline.job_baseline` + the M0c runaway
count, odour runaway = `pool_jobs.odor_runaway_job`, all51 = `apl_cap_disparity.job_all51`. Reading the membrane
and the release inside the read loop does not change the engine state.

`csc_edit` builds a diagnostic variant whose CSC is changed after construction (spec H.3a.8 records, written only
under results/m0d/diag/): ("apl_to_mbon_zero",) zeroes every APL -> MBON edge; ("kc_to_apl_scale", s) multiplies
every KC -> APL edge by s. The job then also returns the sha256 of the edited weight vector.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .engine_cpu import Engine
from .measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline
from .stimuli import design_odor_pair, present

_CACHE: dict = {}          # (Params, csc_edit) -> (Engine, derived arrays); one entry per worker


def mbon_type_index(conn, pops) -> dict:
    """{MBON type: indices of all its cells, both hemispheres} (readout_floor_guard.mbon_type_index)."""
    t = np.asarray(conn.type).astype(str)
    out = {}
    for i in pops.mbon:
        out.setdefault(t[int(i)], []).append(int(i))
    return {n: np.array(sorted(v), np.int64) for n, v in sorted(out.items())}


def edge_sources(csc) -> np.ndarray:
    return np.repeat(np.arange(csc.ptr.size - 1, dtype=np.int64), np.diff(csc.ptr))


def apply_csc_edit(eng: Engine, pops, edit) -> str:
    """Edit the finished CSC in place (the graded-APL out-edge views follow) and return sha256 of the weights."""
    kind = edit[0]
    src, tgt = edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
    is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
    if kind == "apl_to_mbon_zero":
        is_mbon = np.zeros(eng.N, bool); is_mbon[np.asarray(pops.mbon, np.int64)] = True
        eng.csc.w[is_apl[src] & is_mbon[tgt]] = np.float32(0.0)
    elif kind == "kc_to_apl_scale":
        is_kc = np.zeros(eng.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
        eng.csc.w[is_kc[src] & is_apl[tgt]] *= np.float32(edit[1])
    else:
        raise ValueError(f"unknown csc_edit {edit!r}")
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


def engine_for(conn, pops, params, csc_edit=None):
    """The worker's engine for (params, csc_edit), built once and reused while jobs keep asking for it; also the
    per-source weight into APL (the APL-input record)."""
    key = (params, csc_edit)
    if key not in _CACHE:
        _CACHE.clear()
        eng = Engine(conn, pops, params, seed=0)
        edit_sha = apply_csc_edit(eng, pops, csc_edit) if csc_edit is not None else None
        src, tgt = edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
        is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
        into = is_apl[tgt]
        w_to_apl = np.bincount(src[into], weights=eng.csc.w[into].astype(np.float64), minlength=eng.N)
        _CACHE[key] = (eng, dict(w_to_apl=w_to_apl, types=mbon_type_index(conn, pops), edit_sha=edit_sha, apl_to={}))
    return _CACHE[key]


def apl_to_type(eng, pops, derived, name) -> np.ndarray:
    """Each APL cell's summed weight onto the cells of one MBON type (memoised in the engine's derived record)."""
    if name not in derived["apl_to"]:
        src, tgt = edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
        is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
        is_t = np.zeros(eng.N, bool); is_t[derived["types"][name]] = True
        m = is_apl[src] & is_t[tgt]
        per = np.bincount(src[m], weights=eng.csc.w[m].astype(np.float64), minlength=eng.N)
        derived["apl_to"][name] = per[np.asarray(pops.apl, np.int64)]
    return derived["apl_to"][name]


def _classes(pops, n):
    cls = np.full(n, "other", object)
    cls[np.asarray(pops.alpn, np.int64)] = "alpn"
    cls[np.asarray(pops.mbon, np.int64)] = "mbon"
    cls[np.asarray(pops.kc, np.int64)] = "kc"
    return cls


def reference_job(eng, pl, pops, comps, ro, params, odors, strength: float, settle_ms: float, read_steps: int,
                  quantiles, callout, csc_edit=None) -> list:
    """One block of reference presentations: APL membrane (mean, per-cell mean, instantaneous quantiles), release
    (graded APL), KC activity and the fired-KC positions, per-type MBON counts, and the APL input by source class."""
    e, d = engine_for(eng.conn, pops, params, csc_edit)
    apl_to = {n: apl_to_type(e, pops, d, n) for n in callout if n in d["types"]}
    graded = e.p.apl_mode == "graded"
    apl = np.asarray(pops.apl, np.int64)
    n_apl = apl.size
    cls = _classes(pops, e.N)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], strength); e.run(settle_ms)
            counts = np.zeros(e.N, np.int32)
            apl_v = np.zeros((read_steps, n_apl), np.float64)
            rel, rel_cell = 0.0, np.zeros(n_apl, np.float64)
            for i in range(read_steps):
                counts[e.step()] += 1
                va = e.v[apl]
                apl_v[i] = va
                if graded:
                    r = e.apl_release(va).astype(np.float64)
                    rel += float(r.sum())
                    rel_cell += r
            kc = counts[pops.kc]
            per_step = (rel_cell if graded else counts[apl].astype(np.float64)) / read_steps
            drive = counts.astype(np.float64) * d["w_to_apl"] / read_steps
            row = dict(odor=o["name"], seed=int(seed), kc_active_frac=float((kc > 0).mean()), kc_spikes=int(kc.sum()),
                       fired=np.flatnonzero(kc > 0).astype(np.int64).tolist(),
                       apl_v_mean=float(apl_v.mean()), apl_v_mean_per_cell=apl_v.mean(axis=0).tolist(),
                       apl_v_quantiles=[float(x) for x in np.percentile(apl_v, list(quantiles))],
                       release_mean=(rel / (read_steps * n_apl)) if graded else None,
                       release_frac=(rel / (read_steps * n_apl) / e.p.apl_r_max) if graded else None,
                       apl_output_per_step=per_step.tolist(),
                       apl_input_by_class={c: float(drive[cls == c].sum()) for c in ("kc", "alpn", "mbon", "other")},
                       apl_to_type_input={n: float(per_step @ w) for n, w in apl_to.items()},
                       types={n: int(counts[idx].sum()) for n, idx in d["types"].items()})
            if d["edit_sha"] is not None:
                row["csc_sha256"] = d["edit_sha"]
            out.append(row)
    return out


def rest_job(eng, pl, pops, comps, ro, params, seeds, settle_ms: float, read_steps: int, csc_edit=None) -> list:
    """The same window with no odour: the guard's same-seed resting counts per MBON type."""
    e, d = engine_for(eng.conn, pops, params, csc_edit)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); e.run(settle_ms)
        counts = np.zeros(e.N, np.int32)
        for _ in range(read_steps):
            counts[e.step()] += 1
        out.append(dict(seed=int(seed), kc_spikes=int(counts[pops.kc].sum()),
                        types={n: int(counts[idx].sum()) for n, idx in d["types"].items()}))
    return out


def design_job(eng, pl, pops, comps, ro, params, seeds, k: int, odor_seed: int, strength: float,
               settle_ms: float, read_ms: float) -> list:
    """D.4 on the design pair for these seeds (kc_sparsity for both odours per seed)."""
    e, _ = engine_for(eng.conn, pops, params)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    out = []
    for seed in seeds:
        ra = kc_sparsity(e, pops, a, strength, seed=int(seed), settle_ms=settle_ms, read_ms=read_ms)
        rb = kc_sparsity(e, pops, b, strength, seed=int(seed), settle_ms=settle_ms, read_ms=read_ms)
        out.append(dict(seed=int(seed), frac_active_A=ra["frac_active"], frac_active_B=rb["frac_active"],
                        jaccard=jaccard(ra["active"], rb["active"]),
                        chance=chance_jaccard(ra["frac_active"], rb["frac_active"]),
                        kc_hz_A=ra["kc_hz"], kc_hz_B=rb["kc_hz"], mbon_hz_A=ra["mbon_hz"], mbon_hz_B=rb["mbon_hz"]))
    return out


def baseline_job(eng, pl, pops, comps, ro, params, seeds, ms: float, sat_hz: float) -> list:
    """The M0c resting MBON baseline per seed (trimmed mean over cells <= sat_hz) and the KCs above sat_hz."""
    e, _ = engine_for(eng.conn, pops, params)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive()
        counts = e.run(ms)
        r = mbon_baseline(e, pops, int(seed), ms, sat_hz, counts=counts)
        over = (counts / (ms / 1000.0)) > sat_hz
        out.append(dict(seed=int(seed), trimmed=float(r["mbon_hz_trimmed"]), raw=float(r["mbon_hz"]),
                        n_saturated=int(r["n_saturated"]), n_types_active=int(r["n_types_active"]),
                        n_kc_over_sat=int(over[pops.kc].sum()), n_over_sat=int(over.sum())))
    return out


def odor_runaway_job(eng, pl, pops, comps, ro, params, seeds, k: int, odor_seed: int, strength: float,
                     settle_ms: float, read_ms: float, sat_hz: float, record_hz: float) -> list:
    """D.4 runaway under odour B of the design pair: KCs whose read-window rate exceeds sat_hz (judged), and the KCs
    between record_hz and sat_hz with their body ids and rates (recorded, as D.4 asks)."""
    e, _ = engine_for(eng.conn, pops, params)
    _, b = design_odor_pair(pops, k=k, seed=odor_seed)
    kc = np.asarray(pops.kc, np.int64)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); present(e, pops, b, strength); e.run(settle_ms)
        counts = e.run(read_ms)
        kc_hz = counts[kc] / (read_ms / 1000.0)
        band = np.flatnonzero((kc_hz > record_hz) & (kc_hz <= sat_hz))
        out.append(dict(seed=int(seed), n_kc_over_sat=int((kc_hz > sat_hz).sum()),
                        n_kc_over_record=int((kc_hz > record_hz).sum()),
                        kc_hz_top5=[float(x) for x in np.sort(kc_hz)[::-1][:5]],
                        band=[dict(body_id=int(eng.conn.bodyId[kc[i]]), hz=float(kc_hz[i])) for i in band]))
    return out


def all51_job(eng, pl, pops, comps, ro, params, items, c_norm: float, strength: float, settle_ms: float,
              read_steps: int) -> list:
    """One glomerulus alone per presentation at strength c_norm / its receptor count."""
    e, _ = engine_for(eng.conn, pops, params)
    out = []
    for rtype, seed in items:
        e.reset(int(seed)); e.clear_drive()
        present(e, pops, {rtype: c_norm / len(pops.receptor_types[rtype])}, strength)
        e.run(settle_ms)
        counts = np.zeros(e.N, np.int32)
        for _ in range(read_steps):
            counts[e.step()] += 1
        kc = counts[pops.kc]
        out.append(dict(g=rtype, seed=int(seed), kc=int(kc.sum()), kc_on=int((kc > 0).sum()),
                        pn=int(counts[pops.alpn].sum())))
    return out
