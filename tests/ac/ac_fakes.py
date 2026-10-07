"""Shared fakes for tests/ac: a stub E-grid codebook on 49 synthetic glomeruli (tests/agent/test_encode_grid.py's),
stub battles and moves; Task 12 adds a fake pool and fake battles."""
from types import SimpleNamespace

from flymon.ac.tb import TBEncoder
from flymon.agent import e_codebook as ecb
from flymon.agent import encode_grid as eg
from flymon.agent.e_spec import SPEC as E_SPEC
from flymon.agent.e_spec import smoke as e_smoke
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, MON, MOVE = pool_vocabulary()
GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
RC = {g: 1 + (i % 5) for i, g in enumerate(GLOMS)}
DRIVE = {g: float(10 + 50 * i) for i, g in enumerate(GLOMS)}
CB = eg.Codebook(ecb.cells(MOVE, MON), ecb.build(DRIVE, 2, e_smoke(E_SPEC))["codebook"])


class StubPops:
    receptor_types = {g: list(range(RC[g])) for g in GLOMS}


def Mon(species, hp=1.0):
    return SimpleNamespace(species=species, current_hp_fraction=hp)


def Mv(move_id):
    return SimpleNamespace(id=move_id)


def battle(me, opp):
    return SimpleNamespace(active_pokemon=Mon(me), opponent_active_pokemon=Mon(opp))


def stub_grid():
    return eg.GridEncoder(StubPops(), CB, "norm")


def stub_tb():
    return TBEncoder(StubPops(), CB, "norm")


# ---- a fake pool and fake battles (rescope's test_run_battles pattern) -------------------------------------------
import asyncio
import hashlib
from pathlib import Path

import numpy as np

from flymon.rescope import blocks


class FakePool:
    def __init__(self, n, enabled=True):
        self.n_flies = n
        self.flies = [SimpleNamespace(shuffle_seed=None) for _ in range(n)]
        self.w0 = {None: np.linspace(1.0, 2.0, 5, dtype=np.float32)}
        self.w = {i: self.w0[None].copy() for i in range(n)}
        self.en = list(enabled) if isinstance(enabled, (list, tuple)) else [enabled] * n

    def set_enabled(self, f, on):
        self.en[f] = bool(on)

    def state(self):
        return {"flies": [dict(enabled=self.en[i], shuffle_seed=None, w=self.w[i].copy()) for i in range(self.n_flies)]}

    def load_state(self, d):
        for i, e in enumerate(d["flies"]):
            self.w[i] = np.asarray(e["w"], np.float32).copy()
            self.en[i] = bool(e["enabled"])


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:6], 16)


def fake_attempt_for(out, pool, fail=()):
    """A deterministic fake battle: in L an enabled fly's weights move by an amount from the battle id (a disabled fly,
    C-off, stays at w0); `fail` holds (battle_id, attempt) pairs that end unfinished."""
    def attempt_for(block):
        async def attempt(sb, n):
            f = sb.fly_id
            logs = Path(out) / ("logs" if block == "L" else "logs/eval")
            blocks.append_jsonl(logs / f"fly{f:02d}.jsonl", {"battle_id": sb.battle_id, "fly": f, "kind": "decision"})
            if block == "L" and (pool is None or pool.en[f]):
                if pool is not None:
                    pool.w[f] = pool.w[f] * np.float32(0.99) + np.float32(_h(sb.battle_id) % 7) * np.float32(0.01)
            await asyncio.sleep(0)
            if (sb.battle_id, n) in fail:
                return {"finished": False, "won": None}
            return {"finished": True, "won": _h(sb.battle_id, "won") % 2 == 0, "fly_turns": 3, "coach_turns": 1}
        return attempt
    return attempt_for


def run_fake(out, *, learn, eval_, n, pool=None, resume=False, stop_after=None, fail=(), after_battle=None,
             should_stop=None, wrap=None, retry_max=3):
    pool = pool if pool is not None else FakePool(n)
    swarm = SimpleNamespace(mode="learn")
    af = fake_attempt_for(out, pool, fail)
    if wrap is not None:
        af = wrap(af)
    run = asyncio.run(blocks.run_arm(out, eval_=eval_, attempt_for=af, cfg_hash="h" * 64, retry_max=retry_max,
                                     resume=resume, n_flies=n, learn=learn, pool=pool, swarm=swarm,
                                     reset_player=lambda b, f: None, stop_after=stop_after, after_battle=after_battle,
                                     should_stop=should_stop))
    return run, pool


def snapshot(out):
    """Committed battle records (sorted by id) and the per-fly logs. The raw battles.jsonl order differs after a resume
    and is not an analysis input, so only logs/**/fly*.jsonl are compared verbatim (rescope's snapshot)."""
    out = Path(out)
    recs = {p: sorted(blocks.read_jsonl(out / p), key=lambda r: r["battle_id"])
            for p in ("logs/battles.jsonl", "logs/eval/battles.jsonl")}
    logs = {p.relative_to(out).as_posix(): p.read_text() for p in sorted(out.glob("logs/**/fly*.jsonl"))
            if "retries" not in p.parts}
    return recs, logs
