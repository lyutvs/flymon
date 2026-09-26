import pytest

from flymon.agent.pulses import pulses_for
from flymon.battle.attribution import Outcome


@pytest.mark.parametrize("o, want", [
    (Outcome(dealt_frac=0.5, effectiveness="neutral"), [("PAM08", 200.0)]),
    (Outcome(dealt_frac=0.25, effectiveness="super"), [("PAM08", 100.0)]),
    (Outcome(dealt_frac=0.0, effectiveness="resisted"), [("PPL105", 200.0)]),
    (Outcome(dealt_frac=0.0, effectiveness="immune"), [("PPL105", 400.0)]),
    (Outcome(dealt_frac=0.3, effectiveness="neutral", target_fainted_by_me=True), [("PAM08", 320.0)]),
    (Outcome(dealt_frac=0.0, effectiveness="neutral"), []),
    (Outcome(dealt_frac=0.5, missed=True), []),
    (Outcome(dealt_frac=0.5, no_action=True), []),
    (Outcome(dealt_frac=0.5, effectiveness="resisted", uncertain=True), []),
])
def test_table_rows(o, want):
    assert [(d, pytest.approx(ms)) for d, ms in pulses_for(o)] == want


def test_resisted_ko_gives_reward_then_punish():
    o = Outcome(dealt_frac=0.1, effectiveness="resisted", target_fainted_by_me=True)
    assert pulses_for(o) == [("PAM08", pytest.approx(240.0)), ("PPL105", 200.0)]
