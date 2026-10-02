# flymon/brain/q_measure.py
"""Q's measurer (Q.3, Q.6.8): q_jobs.q_oracle_job per (pair, condition, seed block) behind QCache — one atomic file
per entry, written as soon as its worker round returns, so an interrupted condition resumes with only the missing
pairs — and the smoke KC-band probe through k_jobs.activity_job (Q.6.5). A condition's Params are C3's with
mv_per_synapse x mv_scale (cond_params); the C3 threshold file is untouched."""
from __future__ import annotations

import dataclasses
import sys
import time

from ..agent.e_measure import MAX_ITEMS, EMeasurer
from ..agent.e_runner import MEASURE_FILES_E
from . import q_jobs
from .q_pairs import key_str

# The cache code key: everything oracle_job's result depends on (the encoder track's list) plus Q's job, O's and N's
# edit modules it calls, and Q's measurer and store.
Q_MEASURE_FILES = tuple(dict.fromkeys(MEASURE_FILES_E + (
    "flymon/brain/q_jobs.py", "flymon/brain/o_jobs.py", "flymon/brain/n_jobs.py", "flymon/brain/n_spec.py",
    "flymon/brain/q_measure.py", "flymon/brain/q_store.py")))


def cond_params(params, cond):
    if cond.mv_scale == 1.0:
        return params
    return dataclasses.replace(params, mv_per_synapse=params.mv_per_synapse * cond.mv_scale)


def _odour(o: dict) -> dict:
    return {str(g): float(v) for g, v in o.items()}


class QMeasurer:
    def __init__(self, pool, cache, spec, params, readout: dict, z: dict, types, n_kc: int):
        self.pool, self.cache, self.spec, self.params = pool, cache, spec, params
        self.readout, self.z, self.types, self.n_kc = dict(readout), dict(z), list(types), int(n_kc)
        self.last_wall_s, self.last_jobs = 0.0, 0

    def _windows(self) -> dict:
        sp = self.spec
        return dict(settle_ms=float(sp.settle_ms), read_ms=float(sp.read_ms), window_ms=int(sp.window_ms))

    def kwargs(self, row: dict, cond, seeds: dict) -> dict:
        sp = self.spec
        return dict(params=cond_params(self.params, cond), odor_x=_odour(row["odor_x"]), odor_y=_odour(row["odor_y"]),
                    readout=dict(self.readout), z={k: [float(v) for v in self.z[k]] for k in self.z},
                    types=[str(t) for t in self.types], act_seeds=[int(s) for s in seeds["act"]],
                    select_seeds=[int(s) for s in seeds["select"]], report_seeds=[int(s) for s in seeds["report"]],
                    alphas=[float(a) for a in sp.alphas], strength=float(cond.strength), **self._windows(),
                    punish_type=sp.punish_type, reward_type=sp.reward_type, edit=cond.edit,
                    fixed_alphas=[float(a) for a in sp.fixed_alphas], active_fx=float(sp.active_fx))

    def inputs(self, row: dict, cond, block: str, seeds: dict) -> dict:
        return dict(self.kwargs(row, cond, seeds), pair=key_str(row), condition=cond.name, block=block)

    def run(self, rows: list, cond, block: str, seeds: dict) -> list:
        ins = [self.inputs(r, cond, block, seeds) for r in rows]
        todo = [i for i, x in enumerate(ins) if self.cache.get("oracle", x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        plist = [cond_params(self.params, cond)]
        t0 = time.perf_counter()
        if todo:
            print(f"q oracle {cond.name}/{block}: {len(todo)}/{len(rows)} pairs to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            res = self.pool.run_jobs(q_jobs.q_oracle_job, [self.kwargs(rows[i], cond, seeds) for i in batch])
            for i, r in zip(batch, res):
                self.cache.put("oracle", ins[i], r, plist)
            print(f"q oracle {cond.name}/{block}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for r, x in zip(rows, ins):
            got = self.cache.get("oracle", x)
            if got is None:
                raise RuntimeError(f"q oracle {cond.name}/{block}: pair {key_str(r)} missing after the run")
            out.append(dict(key=key_str(r), result=got, cache_key=self.cache.key("oracle", x),
                            cache_file=str(self.cache._path("oracle", x))))
        return out

    def kc_probe(self, odours: dict, params, strength: float, seeds) -> dict:
        """{odour id: [KC active fraction per seed]} (read window). Delegates to e_measure.EMeasurer.activity (same
        k_jobs.activity_job batching of MAX_ITEMS, same per-odour "activity" cache kind, Params in the key); QCache
        is an ECache, so only the Params and the windows of this measurer are passed in."""
        em = EMeasurer(self.pool, self.cache, params, self.n_kc, self.spec)
        got = em.activity({str(k): _odour(o) for k, o in odours.items()}, float(strength), [int(x) for x in seeds])
        return {oid: got[str(oid)]["frac"] for oid in odours}
