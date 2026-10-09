"""ADSwarm: LVSwarm whose reinforcement batch refuses stale requests (spec AD.5 2). A request is delivered only if its
attempt token is the fly's current one and the fly's weight generation equals the one the request was created under;
LeverFlyPool.reinforce_batch repeats both checks at write-back, under its lock. Refused requests are listed in
self.stale (and appended to stale_path if given) with at = "submit" | "apply"."""
from __future__ import annotations

from ..ac.swarm import LVSwarm
from ..rescope.blocks import append_jsonl


class ADSwarm(LVSwarm):
    def __init__(self, pool, cfg, cells: dict, kc, *, attempts, stale_path=None, **kw):
        super().__init__(pool, cfg, cells, kc, **kw)
        self.attempts, self.stale_path = attempts, stale_path
        self.stale: list = []

    def gen_of(self, fly: int) -> int:
        return int(getattr(self.pool.w, "gen", {}).get(int(fly), 0))

    def _current(self, c: dict) -> bool:
        return self.attempts.is_current(c["fly"], c.get("attempt"))

    def _refuse(self, c: dict, at: str) -> None:
        rec = dict(fly=int(c["fly"]), attempt=c.get("attempt"), gen=c.get("gen"), at=at)
        self.stale.append(rec)
        if self.stale_path is not None:
            append_jsonl(self.stale_path, rec)

    # copied from flymon/agent/swarm.py:BrainSwarm.reinforce_run_batch (+ AD.5 2's submission and write-back checks)
    async def reinforce_run_batch(self, reqs) -> list:
        live = []
        for r in reqs:
            c = r.context
            if self._current(c) and self.gen_of(c["fly"]) == int(c["gen"]):
                live.append(c)
            else:
                self._refuse(c, "submit")
        # FlyPool refuses a fly twice in one batch: the n-th request of each fly goes to round n (request order kept)
        rounds: list = []
        seen: dict = {}
        for c in live:
            n = seen.get(c["fly"], 0); seen[c["fly"]] = n + 1
            if n == len(rounds):
                rounds.append([])
            rounds[n].append(c)
        for rnd in rounds:
            jobs = [(c["fly"], c["odour"], c["dan"], c["ms"], c["seed"]) for c in rnd]
            applied = await self._run(self.pool.reinforce_batch, jobs, self.cfg.strength, self.cfg.settle_ms,
                                      expect=[int(c["gen"]) for c in rnd],
                                      accept=lambda k, rnd=rnd: self._current(rnd[k]))
            for c, ok in zip(rnd, applied):
                if not ok:
                    self._refuse(c, "apply")
        for fly in {c["fly"] for c in live}:
            self.check_median(fly)
        return [0] * len(reqs)
