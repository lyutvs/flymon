"""Spec J.12.2's protocol as amended by J.12.7 (independent arms from naive, F.1 reading b) on a FlyPool (F.2's fly
API): one pair's probes and training block, resumable per step.

    brains: Rr_0..Rr_{n-1}, Rp_0.., N_0.., [N2_0.. — the calibration pilot only], noplast_0
    steps:  pre -> choose X -> train:0..T-1 -> S1
    one block for every brain at once: Rr gets the reward DAN, Rp the punishment DAN, noplast the reward DAN with
    plasticity off, N and N2 none; N2 uses its own training seeds.

The pool is an interface (flymon.brain.fly_pool.FlyPool in a run, a fake in the tests): decide_batch, reinforce_batch,
state, load_state. A checkpoint (records, X, the steps done, the pool's weights) is written after every step, so an
interrupted run resumes at the next step with the same weights. The checkpoint also keeps the provenance of the run
that measured the first step; run_pair returns it (not the caller's, when resuming) and whether every step was already
done at the start (a replay: nothing measured now).
"""
from __future__ import annotations

import numpy as np

from .b_rules import choose_x, naive_x


def layout(spec, with_n2: bool) -> list:
    """[(brain, fly)] in pool order."""
    n = range(spec.n_flies)
    return ([("Rr", f) for f in n] + [("Rp", f) for f in n] + [("N", f) for f in n]
            + ([("N2", f) for f in n] if with_n2 else []) + [("noplast", 0)])


def fly_specs(lay: list) -> list:
    return [dict(enabled=b != "noplast") for b, _ in lay]


def steps(spec) -> list:
    return ["pre"] + [f"train:{t}" for t in range(spec.trials)] + ["S1"]


def dan_of(spec, brain: str):
    """The DAN each brain's block drives (None: presentation only)."""
    return {"Rr": spec.reward_dan, "Rp": spec.punish_dan, "noplast": spec.reward_dan}.get(brain)


def probe(pool, spec, pair: str, lay: list, odors: dict, cells: dict, stage: str) -> list:
    """Every brain at each of its probe seeds, one decide_batch per probe index (calls stay short)."""
    idx = np.concatenate([cells[spec.a_type], cells[spec.p_type]])
    n_a = len(cells[spec.a_type])
    out = []
    for k in range(spec.n_probe):
        reqs = [(i, [odors["a"], odors["b"]], spec.probe_seeds(pair, f)[k]) for i, (b, f) in enumerate(lay)]
        res = pool.decide_batch(reqs, spec.strength, settle_ms=spec.probe_settle_ms, read_ms=spec.probe_read_ms, idx=idx)
        for (i, _, seed), counts in zip(reqs, res):
            b, f = lay[i]
            c = np.asarray(counts)
            out.append(dict(brain=b, fly=int(f), stage=stage, seed=int(seed),
                            counts={o: {spec.a_type: int(c[j, :n_a].sum()), spec.p_type: int(c[j, n_a:].sum())}
                                    for j, o in enumerate(("a", "b"))}))
    return out


def train(pool, spec, pair: str, lay: list, x_odor: dict, trial: int) -> None:
    """One trial of the block for every brain at once (one reinforcement per fly per batch)."""
    reqs = [(i, x_odor, dan_of(spec, b), spec.pulse_ms,
             spec.train_seed(pair, f, trial, second_null=b == "N2")) for i, (b, f) in enumerate(lay)]
    pool.reinforce_batch(reqs, spec.strength, settle_ms=spec.train_settle_ms, gap_ms=spec.train_gap_ms)


def run_pair(pool, spec, pair: str, odors: dict, cells: dict, with_n2: bool, fixed_x: str | None,
             checkpoint=None, log=print, provenance: dict | None = None) -> dict:
    """checkpoint: an object with load() -> dict | None and save(dict) (b_store.Checkpoint in a run). provenance: this
    run's (e.g. git state and start time); it is kept only when this run measures the first step, else the stored one
    is returned."""
    lay = layout(spec, with_n2)
    st = checkpoint.load() if checkpoint else None
    st = st or dict(records=[], x=None, done=[])
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
                st["x"] = choose_x(st["records"], spec, fixed_x)
                log(f"{pair}: X = odour {st['x']}")
        else:
            train(pool, spec, pair, lay, odors[st["x"]], int(step.split(":")[1]))
        st["done"].append(step)
        if checkpoint:
            checkpoint.save(dict(st, weights=pool.state()))
        log(f"{pair}: {step} done")
    return dict(pair=pair, x=st["x"], layout=lay, records=st["records"], provenance=st.get("provenance"),
                replayed=replayed)


def screen_calibration(spec, measure) -> dict:
    """J.12.7's calibration-pair rule on naive probes only: measure(seed) -> the reward-arm brains' pre records for
    design_odor_pair(k, seed); the first candidate whose rule-chosen X has a naive A-type median >= calibration_min_a is
    the calibration pair, else the first candidate (recorded as such)."""
    rows = []
    for seed in spec.calibration_candidates:
        recs = measure(seed)
        x = choose_x(recs, spec, None)
        a = naive_x(recs, x, spec)[spec.a_type]
        rows.append(dict(seed=int(seed), x=x, naive_a=a))
        if a >= spec.calibration_min_a:
            return dict(selected=int(seed), met=True, rows=rows)
    return dict(selected=int(spec.calibration_candidates[0]), met=False, rows=rows)
