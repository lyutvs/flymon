"""Spec 10.2's primary protocol: independent arms Rr / N / N' (brain name N2) / noplast from naive, pre -> X -> 20 trials
-> S1, on b_runner.probe / b_runner.train (used unmodified). No punishment arm.

b_runner.train gives each brain its DAN by name (Rr and noplast the reward DAN, N and N2 none) and N2 its own training
seeds (train_seed(..., second_null=True)); the layout below simply has no Rp. X is rules.choose_x on the pre records
(the re-scope rule, no fixed X). A checkpoint (records, X, steps done, the pool's weights) is written after every step, so
an interrupted run resumes at the next step with the same weights; it keeps the provenance of the run that measured the
first step, and `replayed` says every step was already done at the start.
"""
from __future__ import annotations

import numpy as np

from ..brain.b_runner import probe, steps, train
from .rules import choose_x


def check_recovery(params) -> float:
    """Spec 10.2: qualification and the primary run the C3 engine at recovery_per_pulse = 0; any other value refuses
    (SystemExit). Returns the value (0.0) for the summaries."""
    r = float(params.recovery_per_pulse)
    if r != 0.0:
        raise SystemExit(f"refusing: C3 recovery_per_pulse is {r}, not 0 (spec 10.2 fixes recovery 0 for the primary)")
    return r


def layout(spec) -> list:
    """[(brain, fly)] in pool order: Rr x n, N x n, N2 x n, noplast 0."""
    n = range(spec.n_flies)
    return [("Rr", f) for f in n] + [("N", f) for f in n] + [("N2", f) for f in n] + [("noplast", 0)]


def run_pair(pool, spec, pair, odors, cells, checkpoint=None, log=print, provenance=None) -> dict:
    lay = layout(spec)
    st = (checkpoint.load() if checkpoint else None) or dict(records=[], x=None, done=[])
    if not st["done"]:
        st["provenance"] = provenance
    replayed = all(s in st["done"] for s in steps(spec))
    if st["done"] and "weights" in st:
        pool.load_state(st["weights"])
    for step in steps(spec):
        if step in st["done"]:
            continue
        if step in ("pre", "S1"):
            st["records"] += probe(pool, spec, pair, lay, odors, cells, step)
            if step == "pre":
                st["x"] = choose_x(st["records"], spec)
                log(f"{pair}: X = odour {st['x']}")
        else:
            train(pool, spec, pair, lay, odors[st["x"]], int(step.split(":")[1]))
        st["done"].append(step)
        if checkpoint:
            checkpoint.save(dict(st, weights=pool.state()))
        log(f"{pair}: {step} done")
    return dict(pair=pair, x=st["x"], layout=lay, records=st["records"], provenance=st.get("provenance"),
                replayed=replayed)


# ---- spec 10.4 item 6 (recorded only): X-core MBON05 edges' w/w0 ----------------------------------------------
def xcore_job(eng, pl, pops, comps, ro, cells) -> dict:
    """Worker job: for every plastic edge (pl.edges order) whether its post cell is in `cells` (the MBON05 cells) and
    its presynaptic KC's local index (into pops.kc); and pops.kc itself."""
    return dict(post_in=np.isin(eng.csc.tgt[pl.edges], np.asarray(cells)), pre_kc=np.asarray(pl.pre_kc).copy(),
                kc=np.asarray(pops.kc).copy())


def xcore_masks(pool, spec, pair, odors, cells) -> dict:
    """On the naive pool (call before run_pair): {"a": mask, "b": mask, "n_active": {...}, "seed"} - per odour the
    plastic edges whose post cell is an MBON05 (spec.p_type) cell and whose presynaptic KC spikes at least once in a
    naive read of that odour (fly 0, its first probe seed, probe settle / read). run_pair picks X later, so both
    odours' masks are built here."""
    job = pool.run_jobs(xcore_job, [dict(cells=np.asarray(cells[spec.p_type]))])[0]
    seed = int(spec.probe_seeds(pair, 0)[0])
    counts = np.asarray(pool.decide_batch([(0, [odors["a"], odors["b"]], seed)], spec.strength,
                                          settle_ms=spec.probe_settle_ms, read_ms=spec.probe_read_ms,
                                          idx=job["kc"])[0])
    post_in, pre_kc = np.asarray(job["post_in"], bool), np.asarray(job["pre_kc"])
    out = dict(seed=seed, n_active={})
    for j, o in enumerate(("a", "b")):
        active = np.flatnonzero(counts[j] > 0)
        out[o] = post_in & np.isin(pre_kc, active)
        out["n_active"][o] = int(active.size)
    return out


def _w0(pool, i):
    f = pool.flies[i]
    ss = f.get("shuffle_seed") if isinstance(f, dict) else f.shuffle_seed
    return pool.w0[ss]


def xcore_weights(pool, lay, mask) -> dict:
    """{"Rr": [...], "N": [...]}: per fly (layout order) w/w0 on the mask's edges, read from the pool now (after
    run_pair, before the pool is reset)."""
    out = {"Rr": [], "N": []}
    for i, (b, _) in enumerate(lay):
        if b in out:
            r = np.asarray(pool.w[i], float) / np.asarray(_w0(pool, i), float)
            out[b].append(r[np.asarray(mask, bool)])
    return out
