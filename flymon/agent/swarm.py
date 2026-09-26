"""FlyPool batches as BatchBarrier run_batch callables (spec 2 steps 3 and 5; M0b's swarm, now carrying M3's V,
softmax and reinforcement). FlyPool is synchronous and serialized by its lock, so every call runs in one executor
thread and the event loop never blocks."""
from __future__ import annotations

import asyncio
import concurrent.futures

import numpy as np

from . import policy
from .jobs import edge_compartments_job


class SafetyStop(Exception):
    """A spec 4.3 safety constraint failed: stop the run and report."""


class BrainSwarm:
    def __init__(self, pool, cfg, cells: dict, kc, mode: str = "learn"):
        self.pool, self.cfg, self.mode = pool, cfg, mode
        self.a_idx, self.p_idx = np.asarray(cells[cfg.readout["A"]]), np.asarray(cells[cfg.readout["P"]])
        self.kc = np.asarray(kc)
        self.idx = np.concatenate([self.a_idx, self.p_idx, self.kc])
        self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._edge_masks = pool.run_jobs(edge_compartments_job, [dict(types=[cfg.reward_type, cfg.punish_type])])[0]

    async def _run(self, fn, *args, **kw):
        return await asyncio.get_running_loop().run_in_executor(self._exec, lambda: fn(*args, **kw))

    async def decide_run_batch(self, reqs) -> list:
        seeds, jobs = [], []
        for r in reqs:
            c = r.context
            s = policy.derive_seed("decide", c["fly"], c["battle_id"], c["turn"], c["k"])
            seeds.append(s)
            jobs.append((c["fly"], c["odours"], s))
        counts = await self._run(self.pool.decide_batch, jobs, self.cfg.strength, self.cfg.settle_ms,
                                 self.cfg.read_ms, self.idx)
        na, npp = len(self.a_idx), len(self.p_idx)
        out = []
        for r, s, cnt in zip(reqs, seeds, counts):
            c = r.context
            a, p = cnt[:, :na].sum(1), cnt[:, na:na + npp].sum(1)
            v = policy.values(a, p, self.cfg.z)
            t = policy.tau(c["battle_index"], self.cfg)
            pick = policy.choose(v, t, policy.derive_seed("choose", s), self.mode)
            c["detail"] = {"v": [float(x) for x in v], "a": [int(x) for x in a], "p": [int(x) for x in p],
                           "kc_active": [int(x) for x in (cnt[:, na + npp:] > 0).sum(1)], "tau": float(t), "seed": int(s)}
            out.append(pick)
        return out

    async def reinforce_run_batch(self, reqs) -> list:
        jobs = [(r.context["fly"], r.context["odour"], r.context["dan"], r.context["ms"], r.context["seed"]) for r in reqs]
        await self._run(self.pool.reinforce_batch, jobs, self.cfg.strength, self.cfg.settle_ms)
        for fly in {j[0] for j in jobs}:
            self.check_median(fly)
        return [0] * len(reqs)

    def median_ratio(self, fly: int) -> float:
        w0 = self.pool.w0[self.pool.flies[fly].shuffle_seed]
        return float(np.median(self.pool.w[fly] / w0))

    def check_median(self, fly: int) -> None:
        m = self.median_ratio(fly)
        if m < self.cfg.median_floor:
            raise SafetyStop(f"fly {fly}: plastic weight median {m:.3f} of w0 < {self.cfg.median_floor} (spec 4.3)")

    def compartment_fracs(self, fly: int) -> dict:
        w0 = self.pool.w0[self.pool.flies[fly].shuffle_seed]
        return {t: float(np.mean(self.pool.w[fly][m] / w0[m])) for t, m in self._edge_masks.items()}
