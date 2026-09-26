import asyncio
import importlib.util
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_m3_smoke", Path("scripts/run_m3_smoke.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_out_must_be_under_results_m3():
    with pytest.raises(SystemExit):
        m.parse_args(["--out", "results/m0d/x"])
    assert m.parse_args(["--out", "results/m3/smoke/x"]).flies == 2


def test_opponent_name_is_per_battle_and_fits_showdown():
    names = {m.opponent_name(f"f{f:02d}-b{b:03d}") for f in range(3) for b in range(3)}
    assert len(names) == 9 and all(len(n) <= 18 for n in names)
    assert m.opponent_name("f01-b002") == "fm-h-f01-b002"
    with pytest.raises(ValueError):
        m.opponent_name("x" * 20)


def test_battle_opponent_uses_the_kind_class_with_the_per_battle_account(monkeypatch):
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(kw)
    monkeypatch.setitem(m.KINDS, "heuristic", Fake)
    opp = m.make_battle_opponent("heuristic", "f00-b001", "srv", "team")
    assert isinstance(opp, Fake)
    kw = made[0]
    assert kw["account_configuration"].username == "fm-h-f00-b001"
    assert (kw["battle_format"], kw["server_configuration"], kw["team"], kw["max_concurrent_battles"]) == \
        ("gen1ou", "srv", "team", 1)


class _LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.t.start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(10)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop); self.t.join(10); self.loop.close()


async def _contend(lock):
    """Force the lock to wait once, which binds it to the running loop (as concurrent in-battle flushes do)."""
    await lock.acquire()
    waiter = asyncio.ensure_future(lock.acquire())
    await asyncio.sleep(0)
    lock.release(); await waiter; lock.release()


def test_drain_runs_on_the_player_loop_where_the_barrier_lock_is_bound():
    from flymon.battle.barrier import BatchBarrier
    from flymon.agent.swarm import SafetyStop
    poke = _LoopThread()
    try:
        async def run_batch(reqs):
            return [0] * len(reqs)
        bar = BatchBarrier(run_batch, deadline_ms=1)
        poke.run(_contend(bar._lock))                         # bound to the "POKE_LOOP"
        seen = []

        async def drain():
            seen.append(asyncio.get_running_loop())
            await _contend(bar._lock)                          # fine on the bound loop
            bar.register("p"); await bar.submit("p", None, ["PAM08"], {}); bar.unregister("p")
        player = SimpleNamespace(ps_client=SimpleNamespace(loop=poke.loop))

        async def main():
            await m.on_player_loop(player, drain())
            with pytest.raises(RuntimeError, match="different event loop"):
                await _contend(bar._lock)                      # the old path: same lock from the main loop

            async def boom():
                raise SafetyStop("stop")
            with pytest.raises(SafetyStop):
                await m.on_player_loop(player, boom())         # fatal semantics survive the hop
        asyncio.run(main())
        assert seen == [poke.loop]
    finally:
        poke.close()
