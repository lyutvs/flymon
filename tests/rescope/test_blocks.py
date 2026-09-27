from flymon.rescope.blocks import block_schedule, schedule_digest, assert_disjoint, battle_index


def test_block_ids_and_index():
    s = block_schedule(2, 3, 201, "PL")
    assert [b.battle_id for b in s[:3]] == ["PL-f00-b000", "PL-f00-b001", "PL-f00-b002"]
    assert battle_index("PE-f01-b017") == 17


def test_eval_schedule_same_for_all_arms():
    assert schedule_digest(block_schedule(4, 5, 202, "PE")) == schedule_digest(block_schedule(4, 5, 202, "PE"))


def test_disjoint_detects_repeat():
    import pytest
    learn = block_schedule(2, 3, 201, "PL")
    assert_disjoint(learn, block_schedule(2, 3, 202, "PE"))
    with pytest.raises(ValueError):
        assert_disjoint(learn, [learn[0]])


def test_spec_schedules_disjoint_and_ids_fit_showdown():
    """The pilot and judge schedules at their largest planned size: learn/eval disjoint, opponent names <= 18."""
    from flymon.rescope.spec import SPEC
    for _, ls, es in SPEC.schedule_seeds:
        learn, ev = block_schedule(32, SPEC.learn_battles, ls, "PL"), block_schedule(32, 300, es, "PE")
        assert_disjoint(learn, ev)
        assert max(len(f"fm-h-{s.battle_id}-3") for s in learn + ev) <= 18


def test_digest_changes_with_schedule():
    assert schedule_digest(block_schedule(2, 3, 202, "PE")) != schedule_digest(block_schedule(2, 3, 102, "PE"))
    assert schedule_digest(block_schedule(2, 3, 202, "PE")) != schedule_digest(block_schedule(2, 4, 202, "PE"))


# ---- phase-tagged ids and the per-battle RND seed (final-review finding 5) -------------------------------
def test_block_tags_carry_the_phase():
    import pytest
    from flymon.rescope.blocks import BLOCKS, block_tag
    assert BLOCKS == ("PL", "PE", "JL", "JE")
    assert [block_tag(p, b) for p in ("pilot", "judge") for b in ("L", "E")] == ["PL", "PE", "JL", "JE"]
    for bad in (("pilot", "X"), ("other", "L")):
        with pytest.raises(ValueError):
            block_tag(*bad)
    with pytest.raises(ValueError):
        block_schedule(1, 1, 1, "L")                             # the old untagged block is refused


def test_pilot_and_judge_ids_differ_so_derive_seed_differs():
    from flymon.agent.policy import derive_seed
    pl, jl = block_schedule(2, 3, 101, "PL"), block_schedule(2, 3, 101, "JL")
    assert [s.my_team for s in pl] == [s.my_team for s in jl]    # same seed, same teams ...
    assert not {s.battle_id for s in pl} & {s.battle_id for s in jl}   # ... but never the same id
    assert all(derive_seed("reinforce", 0, a.battle_id, 3, 0) != derive_seed("reinforce", 0, b.battle_id, 3, 0)
               for a, b in zip(pl, jl))
    assert battle_index("JL-f01-b002") == 2 and battle_index("JE-f00-b000") == 0


def test_battle_index_refuses_foreign_ids():
    import pytest
    for bad in ("L-f00-b001", "f00-b001", "PL-f00-b1000", "XX-f00-b001"):
        with pytest.raises(ValueError):
            battle_index(bad)


def test_longest_opponent_name_is_18():
    import importlib.util
    from pathlib import Path
    import pytest
    from flymon.rescope.spec import SPEC
    p = Path(__file__).resolve().parents[2] / "scripts" / "run_rescope_battles.py"
    spec = importlib.util.spec_from_file_location("rb_names", p)
    rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
    name = rb.opponent_name("PL-f00-b000", SPEC.retry_max)
    assert name == "fm-h-PL-f00-b000-3" and len(name) == 18
    rb.check_names(block_schedule(32, 300, 202, "JE"), SPEC.retry_max)
    with pytest.raises(ValueError):                               # 1000 battles: a 4-digit index would not fit
        rb.check_names(block_schedule(1, 1001, 202, "JE"), SPEC.retry_max)


def test_rnd_reseeded_per_battle_so_resume_replays(tmp_path):
    """A resumed RND fly (fresh player, starting at battle 2) draws what the uninterrupted fly drew at battle 2."""
    import asyncio
    from poke_env.ps_client import AccountConfiguration
    from flymon.battle.coach import Coach
    from flymon.battle.providers import RandomProvider
    from flymon.rescope.blocks import NoBrainPlayer, rnd_seed

    def player(name):
        return NoBrainPlayer(3, phase="judge", provider=RandomProvider(seed=3), coach=Coach(), barrier=None,
                             log_path=tmp_path / f"{name}.jsonl", account_configuration=AccountConfiguration(name, None),
                             battle_format="gen1ou", start_listening=False)

    def draws(p, bid):
        p.start_battle(bid, battle_index(bid))
        return [asyncio.run(p.provider.decide(None, [0, 1, 2, 3], {})) for _ in range(20)]

    ids = [s.battle_id for s in block_schedule(4, 3, 202, "JE") if s.fly_id == 3]
    whole = player("fm-rnd-a")
    seq = {b: draws(whole, b) for b in ids}
    resumed = player("fm-rnd-b")
    assert draws(resumed, ids[2]) == seq[ids[2]]
    assert seq[ids[0]] != seq[ids[1]]
    assert rnd_seed("pilot", 3, "PE-f03-b000") != rnd_seed("judge", 3, "PE-f03-b000")
