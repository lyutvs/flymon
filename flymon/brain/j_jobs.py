"""FlyPool worker jobs of spec appendix J: the all51 record the depression scan needs (J.11.3) and D.6 on a declared
engine (J.11.4-5).

Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them. Every
job builds (or reuses) its engine from the `Params` it is given (h3_jobs.engine_for / h4_jobs.rig_for), so a
`StdParams` job carries its depression and receptor scale; the worker's default engine only lends its connectome.
"""
from __future__ import annotations

import numpy as np

from . import d6a
from .h3_jobs import engine_for
from .h4_jobs import rig_for
from .stimuli import present

_UNI: dict = {}


def uni_pns(conn, pops, frac: float, min_syn: int) -> dict:
    """{glomerulus: its uniglomerular ALPN indices}: >= frac of a PN's ORN input synapses (all counts, before the
    CSC's min_weight) come from one glomerulus and it has >= min_syn of them (m0d-diag/disparity_stages.py's rule)."""
    key = (conn.N, int(conn.pre.size), float(frac), int(min_syn))
    if key not in _UNI:
        rtypes = sorted(pops.receptor_types)
        orn_of = np.full(conn.N, -1, np.int64)
        for i, t in enumerate(rtypes):
            orn_of[np.asarray(pops.receptor_types[t], np.int64)] = i
        pn = np.asarray(pops.alpn, np.int64)
        pos = np.full(conn.N, -1, np.int64)
        pos[pn] = np.arange(pn.size)
        m = (orn_of[conn.pre] >= 0) & (pos[conn.post] >= 0)
        M = np.zeros((pn.size, len(rtypes)))
        np.add.at(M, (pos[conn.post[m]], orn_of[conn.pre[m]]), conn.w[m])
        tot, best = M.sum(1), M.argmax(1)
        share = np.where(tot > 0, M.max(1) / np.maximum(tot, 1), 0.0)
        uni = (share >= frac) & (tot >= min_syn)
        _UNI.clear()
        _UNI[key] = {t: pn[uni & (best == i)] for i, t in enumerate(rtypes)}
    return _UNI[key]


def all51_uni_job(eng, pl, pops, comps, ro, params, items, c_norm: float, strength: float, settle_ms: float,
                  read_steps: int, uni_frac: float, uni_min_syn: int) -> list:
    """h3_jobs.all51_job's presentation (one glomerulus alone at strength c_norm / its receptor count) with the scan's
    measures: KC spikes and active KCs, ALPN spikes, the glomerulus's own uniPN mean rate, its receptors' realised rate
    and their mean depression resource at the end of the read (None without depression)."""
    e, _ = engine_for(eng.conn, pops, params)
    uni = uni_pns(eng.conn, pops, uni_frac, uni_min_syn)
    sec = read_steps * e.p.dt / 1000.0
    out = []
    for rtype, seed in items:
        e.reset(int(seed)); e.clear_drive()
        present(e, pops, {rtype: c_norm / len(pops.receptor_types[rtype])}, strength)
        e.run(settle_ms)
        counts = np.zeros(e.N, np.int32)
        for _ in range(read_steps):
            counts[e.step()] += 1
        kc, rec, own = counts[pops.kc], np.asarray(pops.receptor_types[rtype], np.int64), uni.get(rtype)
        out.append(dict(g=rtype, seed=int(seed), kc=int(kc.sum()), kc_on=int((kc > 0).sum()),
                        pn=int(counts[pops.alpn].sum()), uni_n=int(0 if own is None else own.size),
                        uni_hz=(None if own is None or own.size == 0 else float(counts[own].mean() / sec)),
                        orn_hz=float(counts[rec].mean() / sec),
                        std_r=(float(e._std_r[rec].mean()) if e.p.orn_std else None)))
    return out


def d6_job(eng, pl, pops, comps, ro, params, odors: list, seed: int) -> dict:
    """d6a.d6a_job (G.8's presentation, E.1's paired noise) on the rig built from `params` instead of the worker's."""
    e, p, c = rig_for(eng.conn, pops, params)
    return d6a.d6a_job(e, p, pops, c, ro, odors, int(seed))
