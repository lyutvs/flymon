"""Showdown plumbing for AC's M4 arms: scripts/run_rescope_battles.py's Battles with AC's account names and ids, one
encoder per fly (FLY-TB flies get the TB encoder) and the arm read from the fly's global id. Learning players are
AgentPlayers, evaluation players rescope's EvalPlayer (no pulse, multipliers logged), RND / MAX rescope's
NoBrainPlayer (RandomProvider reseeded per battle from derive_seed("rnd", phase, fly, battle_id))."""
from __future__ import annotations

from pathlib import Path

from ..rescope import blocks
from . import schedules


async def on_player_loop(player, coro):
    from poke_env.concurrency import handle_threaded_coroutines
    return await handle_threaded_coroutines(coro, player.ps_client.loop)


def team(species) -> str:
    from ..battle.pool import by_species, team_export
    return team_export([by_species[s] for s in species])


class ACBattles:  # copied from scripts/run_rescope_battles.py:Battles (AC ids, per-fly encoder, arm per fly)
    def __init__(self, arm_of: dict, out, srv_cfg, swarm=None, encoders=None, phase: str = "ac"):
        self.arm_of, self.out, self.srv_cfg = dict(arm_of), Path(out), srv_cfg
        self.swarm, self.encoders, self.phase = swarm, dict(encoders or {}), phase
        self.players = {"L": {}, "E": {}}
        self.bars = {}
        if swarm is not None:
            from ..battle.barrier import BatchBarrier
            from .barrier import OverlapBarrier
            # speedup brief fix 3: batches of an overlapping swarm (LVSwarm on LeverFlyPool) may run side by side
            Bar = OverlapBarrier if getattr(swarm, "overlapping", False) else BatchBarrier
            self.bars["L"] = (Bar(swarm.decide_run_batch), Bar(swarm.reinforce_run_batch))
            self.bars["E"] = (Bar(swarm.decide_run_batch), Bar(blocks.refuse_pulses))

    def player(self, block: str, sb):
        p = self.players[block].get(sb.fly_id)
        if p is not None:
            return p
        from poke_env.ps_client import AccountConfiguration
        from ..agent.player import AgentPlayer
        from ..agent.screen import ScreenTable
        from ..battle.coach import Coach
        from ..battle.providers import MaxDamageProvider, RandomProvider
        f = int(sb.fly_id)
        arm = self.arm_of[f]
        logs = self.out / "logs" if block == "L" else self.out / "logs" / "eval"
        common = dict(log_path=logs / f"fly{f:02d}.jsonl", coach=Coach(),
                      account_configuration=AccountConfiguration(schedules.account(arm, block, f), None),
                      battle_format="gen1ou", server_configuration=self.srv_cfg, team=team(sb.my_team),
                      max_concurrent_battles=1)
        if arm in ("RND", "MAX"):
            prov = RandomProvider(seed=f) if arm == "RND" else MaxDamageProvider()
            p = blocks.NoBrainPlayer(f, phase=self.phase, provider=prov, barrier=None, **common)
        else:
            dbar, rbar = self.bars[block]
            brain = dict(fly=f, encoder=self.encoders[f], table=ScreenTable.allow_all(), rbarrier=rbar, barrier=dbar,
                         **common)
            p = blocks.EvalPlayer(**brain) if block == "E" else AgentPlayer(**brain)
        self.players[block][f] = p
        return p

    def reset_player(self, block: str, fly: int) -> None:
        p = self.players[block].get(fly)
        if p is not None and hasattr(p, "_queue"):
            p._queue.clear()
            p._choice.clear()

    def attempt_for(self, block: str):
        async def attempt(sb, n):
            from poke_env.ps_client import AccountConfiguration
            from ..battle.opponents import KINDS
            p = self.player(block, sb)
            opp = KINDS[sb.opponent](account_configuration=AccountConfiguration(
                schedules.opponent_name(sb.battle_id, n), None), battle_format="gen1ou",
                server_configuration=self.srv_cfg, team=team(sb.opp_team), max_concurrent_battles=1)
            p.update_team(team(sb.my_team))
            p.start_battle(sb.battle_id, schedules.battle_index(sb.battle_id))
            before = set(p.battles)
            try:
                await p.battle_against(opp, n_battles=1)
                if getattr(p, "fatal", None) is not None:
                    raise p.fatal
                new = sorted(set(p.battles) - before)
                if len(new) != 1:
                    return {"finished": False, "won": None, "battle_tag": None, "new_battles": new}
                tag = new[0]
                st = p.battle_stats(tag)
                res = dict(st, battle_tag=tag, turns=int(p.battles[tag].turn))
                if not st["finished"]:
                    return res
                if self.swarm is not None:
                    await on_player_loop(p, p.drain())
                    if p.fatal is not None:
                        raise p.fatal
                    p.write_battle_summary(tag, self.swarm.compartment_fracs(sb.fly_id))
                return res
            finally:
                await opp.ps_client.stop_listening()
        return attempt

    async def close_block(self, block: str) -> None:
        ps = self.players[block]
        while ps:
            _, p = ps.popitem()
            await p.ps_client.stop_listening()

    async def close(self) -> None:
        for ps in self.players.values():
            for p in ps.values():
                await p.ps_client.stop_listening()
