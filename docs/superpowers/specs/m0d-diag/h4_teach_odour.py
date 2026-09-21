"""M0d H.4a.1: per-MBON-type teachability on the M0c single-channel arms, in both odour orders, on the three adopted
engines (block "h3" of results/summary/m0d.json). Diagnostic only: no oracle, no M2 turns.

Why this exists. H.4 step 1 asks whether each readout type's count on "the taught odour" falls in >= 6 of 8 seeds
(M0c `punish_only` / `reward_only`, seeds 8-15). M0c's arms teach odour a on the punishment channel and odour b on the
reward channel, and MBON13 (PPL105 core) hardly answers odour a (spec A.4). This runs every arm in both orders
(order "ab" = M0c: CS+ = odour a; "ba" exchanges the odours) and keeps every pool type's count, with the A / P core
sums so C0's order "ab" can be compared with results/m0c/conditioning.json (it matches bit for bit).

    uv run python docs/superpowers/specs/m0d-diag/h4_teach_odour.py [out.json]   # default results/m0d/diag/h4_teach_odour.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from flymon.brain.conditioning import arms, train_block
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import design_odor_pair, present

TYPES = ("MBON13", "MBON18", "MBON05", "MBON21")
SEEDS = list(range(8, 16))


def params_from(d: dict) -> Params:
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def teach_job(eng, pl, pops, comps, ro, seed: int, arm: str, order: str) -> dict:
    t = np.asarray(eng.conn.type).astype(str)
    idx = {n: np.flatnonzero(t == n) for n in TYPES}
    a, b = design_odor_pair(pops, k=8, seed=0)
    cs_plus, cs_minus = (a, b) if order == "ab" else (b, a)

    def probe(odor):
        was = pl.enabled
        pl.set_enabled(False)
        eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
        present(eng, pops, odor, 0.35)
        eng.run(800.0)
        c = eng.run(600.0)
        pl.set_enabled(was)
        return {"A": int(c[ro.a_core].sum()), "P": int(c[ro.p_core].sum()), **{n: int(c[i].sum()) for n, i in idx.items()}}

    punish, reward, plastic = arms()[arm]
    pl.reset_weights()
    pre = {"plus": probe(cs_plus), "minus": probe(cs_minus)}
    pl.set_enabled(plastic)
    train_block(eng, pl, pops, cs_plus, cs_minus, 0.35, seed, punish, reward, 12, 800.0, 200.0, 800.0)
    pl.set_enabled(True)
    post = {"plus": probe(cs_plus), "minus": probe(cs_minus)}
    pl.reset_weights()
    return {"seed": seed, "arm": arm, "order": order, "pre": pre, "post": post}


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "results/m0d/diag/h4_teach_odour.json"
    h3 = json.load(open("results/summary/m0d.json"))["h3"]
    res = {}
    for name in ("C0", "C1", "C3"):
        p = params_from(h3["combos"][name]["adopted"]["params"])
        t0 = time.time()
        jobs = [dict(seed=s, arm=arm, order=o) for o in ("ab", "ba") for arm in ("punish_only", "reward_only") for s in SEEDS]
        with FlyPool("data/malecns.npz", p, [{} for _ in range(16)], workers=16, timeout_s=7200.0) as pool:
            res[name] = pool.run_jobs(teach_job, jobs)
        print(f"{name} done {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
