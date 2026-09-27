"""Spec 10.2's primary protocol: independent arms Rr / N / N' (brain name N2) / noplast from naive, pre -> X -> 20 trials
-> S1, on b_runner.probe / b_runner.train (used unmodified). No punishment arm.

b_runner.train gives each brain its DAN by name (Rr and noplast the reward DAN, N and N2 none) and N2 its own training
seeds (train_seed(..., second_null=True)); the layout below simply has no Rp. X is rules.choose_x on the pre records
(the re-scope rule, no fixed X). A checkpoint (records, X, steps done, the pool's weights) is written after every step, so
an interrupted run resumes at the next step with the same weights; it keeps the provenance of the run that measured the
first step, and `replayed` says every step was already done at the start.
"""
from __future__ import annotations

from ..brain.b_runner import probe, steps, train
from .rules import choose_x


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
