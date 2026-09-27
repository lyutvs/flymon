from flymon.rescope.blocks import block_schedule, schedule_digest, assert_disjoint, battle_index


def test_block_ids_and_index():
    s = block_schedule(2, 3, 201, "L")
    assert [b.battle_id for b in s[:3]] == ["L-f00-b000", "L-f00-b001", "L-f00-b002"]
    assert battle_index("E-f01-b017") == 17


def test_eval_schedule_same_for_all_arms():
    assert schedule_digest(block_schedule(4, 5, 202, "E")) == schedule_digest(block_schedule(4, 5, 202, "E"))


def test_disjoint_detects_repeat():
    import pytest
    learn = block_schedule(2, 3, 201, "L")
    assert_disjoint(learn, block_schedule(2, 3, 202, "E"))
    with pytest.raises(ValueError):
        assert_disjoint(learn, [learn[0]])


def test_spec_schedules_disjoint_and_ids_fit_showdown():
    """The pilot and judge schedules at their largest planned size: learn/eval disjoint, opponent names <= 18."""
    from flymon.rescope.spec import SPEC
    for _, ls, es in SPEC.schedule_seeds:
        learn, ev = block_schedule(32, SPEC.learn_battles, ls, "L"), block_schedule(32, 300, es, "E")
        assert_disjoint(learn, ev)
        assert max(len(f"fm-h-{s.battle_id}-3") for s in learn + ev) <= 18


def test_digest_changes_with_schedule():
    assert schedule_digest(block_schedule(2, 3, 202, "E")) != schedule_digest(block_schedule(2, 3, 102, "E"))
    assert schedule_digest(block_schedule(2, 3, 202, "E")) != schedule_digest(block_schedule(2, 4, 202, "E"))
