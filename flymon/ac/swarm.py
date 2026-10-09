"""LVSwarm: BrainSwarm's decision batch with spec AC's tie rule and AC.3's shadow record. In evaluation (argmax) a tie
in V is broken from derive_seed(tie_seed, "tie", fly, battle_id, turn, k) and flagged; in learning the softmax is M3's
and only the flag is recorded. Every fly decision also records whether the turn's candidate KC counts differ by more
than kc_ratio (D.6 (a), AC.0 6) and each candidate's E-grid cell [move type, opponent types].
seed_fly (AD.1) maps the pool index to the global fly for the decision and tie seeds."""
from __future__ import annotations

import concurrent.futures

import numpy as np

from ..agent import policy
from ..agent.swarm import BrainSwarm
from .spec import SPEC


def kc_ratio_exceeds(kc_active, ratio: float) -> bool:
    kc = [int(x) for x in kc_active]
    if len(kc) < 2 or max(kc) == 0:
        return False
    if min(kc) == 0:
        return True
    return max(kc) / min(kc) > ratio


def egrid_fn(enc):
    def cell(battle, move) -> list:
        mtype = enc.move_info[enc.move_by_id[move.id]][0]
        opp = enc._species(battle.opponent_active_pokemon)
        return [mtype, sorted(enc.species_types[opp])]
    return cell


class LVSwarm(BrainSwarm):
    def __init__(self, pool, cfg, cells: dict, kc, mode: str = "learn", tie_seed: int = SPEC.tie_seed,
                 kc_ratio: float = SPEC.kc_ratio, egrid=None, threads: int = 64, seed_fly: dict | None = None):
        super().__init__(pool, cfg, cells, kc, mode)
        self.tie_seed, self.kc_ratio, self.egrid = int(tie_seed), float(kc_ratio), egrid
        # AD.1: pool index -> global fly for every derive_seed key (None = the pool index, AC's stage 1)
        self.seed_fly = None if seed_fly is None else {int(k): int(v) for k, v in dict(seed_fly).items()}
        # speedup brief fix 3: a pool whose calls may overlap (LeverFlyPool overlap=True) gets one executor thread per
        # in-flight batch, so batches no longer queue behind each other; any other pool keeps BrainSwarm's one thread
        self.overlapping = bool(getattr(pool, "overlap", False))
        if self.overlapping:
            self._exec.shutdown(wait=False)
            self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

    def _sf(self, fly) -> int:
        return int(fly) if self.seed_fly is None else self.seed_fly[int(fly)]

    async def decide_run_batch(self, reqs) -> list:
        # copied from flymon/agent/swarm.py:BrainSwarm.decide_run_batch (tie rule and shadow fields added)
        seeds, jobs = [], []
        for r in reqs:
            c = r.context
            s = policy.derive_seed("decide", self._sf(c["fly"]), c["battle_id"], c["turn"], c["k"])
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
            if self.mode == "eval":
                pick, tied = policy.argmax_tiebreak(
                    v, policy.derive_seed(self.tie_seed, "tie", self._sf(c["fly"]), c["battle_id"], c["turn"], c["k"]))
            else:
                pick = policy.choose(v, t, policy.derive_seed("choose", s), self.mode)
                tied = bool(np.sum(v == v.max()) > 1)
            kc = [int(x) for x in (cnt[:, na + npp:] > 0).sum(1)]
            battle = getattr(r, "battle", None)
            egrid = [self.egrid(battle, m) for m in r.candidates] if (self.egrid and battle is not None) else None
            c["detail"] = {"v": [float(x) for x in v], "a": [int(x) for x in a], "p": [int(x) for x in p],
                           "kc_active": kc, "tau": float(t), "seed": int(s), "tie": bool(tied),
                           "kc_ratio_gt2": kc_ratio_exceeds(kc, self.kc_ratio), "egrid": egrid}
            out.append(int(pick))
        return out
