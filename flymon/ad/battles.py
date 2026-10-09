"""Showdown plumbing for AD (spec AD.1, AD.5 2): AC's ACBattles with AD's players (ADAgentPlayer in learning,
ADEvalPlayer in evaluation; both carry the global fly, the attempt token and the generation reader), AD's account names,
one encoder per fly (FLY-L the FLY encoder, FLY-OS its own OSEncoder). Every attempt begins its token (and the
encoder's schedule id) before it plays; the rollback of an unfinished attempt retires the token before the weights are
restored (reset_player is called first in run_ad_arm's rollback)."""
from __future__ import annotations

from ..ac.battles import ACBattles, on_player_loop, team
from . import inputs


class ADBattles(ACBattles):
    def __init__(self, lay, out, srv_cfg, swarm, encoders, attempts):
        super().__init__({i: arm for i, (arm, _, _) in enumerate(lay)}, out, srv_cfg, swarm=swarm, encoders=encoders,
                         phase="ad")
        self.lay, self.attempts = list(lay), attempts

    # copied from flymon/ac/battles.py:ACBattles.player (AD players and account names; brain arms only)
    def player(self, block: str, sb):
        p = self.players[block].get(sb.fly_id)
        if p is not None:
            return p
        from poke_env.ps_client import AccountConfiguration
        from ..agent.screen import ScreenTable
        from ..battle.coach import Coach
        from .player import ADAgentPlayer, ADEvalPlayer
        i = int(sb.fly_id)
        arm, _, g = self.lay[i]
        logs = self.out / "logs" if block == "L" else self.out / "logs" / "eval"
        dbar, rbar = self.bars[block]
        kw = dict(fly=i, encoder=self.encoders[i], table=ScreenTable.allow_all(), rbarrier=rbar, barrier=dbar, gfly=g,
                  attempts=self.attempts, gen_of=self.swarm.gen_of, log_path=logs / f"fly{i:02d}.jsonl", coach=Coach(),
                  account_configuration=AccountConfiguration(inputs.account(arm, block, g), None),
                  battle_format="gen1ou", server_configuration=self.srv_cfg, team=team(sb.my_team),
                  max_concurrent_battles=1)
        p = ADEvalPlayer(**kw) if block == "E" else ADAgentPlayer(**kw)
        self.players[block][i] = p
        return p

    def reset_player(self, block: str, fly: int) -> None:
        self.attempts.retire(fly)
        super().reset_player(block, fly)

    # copied from flymon/ac/battles.py:ACBattles.attempt_for (+ start_attempt: the attempt token, the encoder's id)
    def attempt_for(self, block: str):
        async def attempt(sb, n):
            from poke_env.ps_client import AccountConfiguration
            from ..ac import schedules as ac_sched
            from ..battle.opponents import KINDS
            p = self.player(block, sb)
            opp = KINDS[sb.opponent](account_configuration=AccountConfiguration(
                ac_sched.opponent_name(sb.battle_id, n), None), battle_format="gen1ou",
                server_configuration=self.srv_cfg, team=team(sb.opp_team), max_concurrent_battles=1)
            p.update_team(team(sb.my_team))
            p.start_attempt(sb.battle_id, inputs.battle_index(sb.battle_id), n)
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
                await on_player_loop(p, p.drain())
                if p.fatal is not None:
                    raise p.fatal
                p.write_battle_summary(tag, self.swarm.compartment_fracs(sb.fly_id))
                return res
            finally:
                await opp.ps_client.stop_listening()
        return attempt
