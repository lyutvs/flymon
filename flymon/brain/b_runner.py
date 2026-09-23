"""Spec J.12.2's protocol on a FlyPool (F.2's fly API): one pair's probes and training blocks, resumable per step.

    brains: R_0..R_{n-1}, N_0..N_{n-1}, [N2_0..N2_{n-1} — the calibration pilot only], noplast_0
    steps:  pre -> choose X -> reward:0..T-1 -> S1 -> punish:0..T-1 -> S2
    reward / punish: R and noplast get the DAN (noplast has plasticity off), N and N2 get none; N2 uses its own seeds.

The pool is an interface (flymon.brain.fly_pool.FlyPool in a run, a fake in the tests): decide_batch, reinforce_batch,
state, load_state. A checkpoint (records, X, the steps done, the pool's weights) is written after every step, so an
interrupted run resumes at the next step with the same weights.
"""
from __future__ import annotations

import numpy as np

from .b_rules import choose_x


def layout(spec, with_n2: bool) -> list:
    """[(brain, fly)] in pool order."""
    n = range(spec.n_flies)
    return ([("R", f) for f in n] + [("N", f) for f in n] + ([("N2", f) for f in n] if with_n2 else [])
            + [("noplast", 0)])


def fly_specs(lay: list) -> list:
    return [dict(enabled=b != "noplast") for b, _ in lay]


def steps(spec) -> list:
    return (["pre"] + [f"reward:{t}" for t in range(spec.trials)] + ["S1"]
            + [f"punish:{t}" for t in range(spec.trials)] + ["S2"])


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


def train(pool, spec, pair: str, lay: list, x_odor: dict, dan: str, trial: int) -> None:
    """One trial of a block for every brain at once (one reinforcement per fly per batch)."""
    reqs = [(i, x_odor, dan if b in ("R", "noplast") else None, spec.pulse_ms,
             spec.train_seed(pair, f, trial, second_null=b == "N2")) for i, (b, f) in enumerate(lay)]
    pool.reinforce_batch(reqs, spec.strength, settle_ms=spec.train_settle_ms, gap_ms=spec.train_gap_ms)


def run_pair(pool, spec, pair: str, odors: dict, cells: dict, with_n2: bool, fixed_x: str | None,
             checkpoint=None, log=print) -> dict:
    """checkpoint: an object with load() -> dict | None and save(dict) (b_store.Checkpoint in a run)."""
    lay = layout(spec, with_n2)
    st = checkpoint.load() if checkpoint else None
    st = st or dict(records=[], x=None, done=[])
    if st["done"] and "weights" in st:
        pool.load_state(st["weights"])
    for step in steps(spec):
        if step in st["done"]:
            continue
        if step in ("pre", "S1", "S2"):
            st["records"] += probe(pool, spec, pair, lay, odors, cells, step)
            if step == "pre":
                st["x"] = choose_x(st["records"], spec, fixed_x)
                log(f"{pair}: X = odour {st['x']}")
        else:
            block, t = step.split(":")
            dan = spec.reward_dan if block == "reward" else spec.punish_dan
            train(pool, spec, pair, lay, odors[st["x"]], dan, int(t) + (spec.trials if block == "punish" else 0))
        st["done"].append(step)
        if checkpoint:
            checkpoint.save(dict(st, weights=pool.state()))
        log(f"{pair}: {step} done")
    return dict(pair=pair, x=st["x"], layout=lay, records=st["records"])
