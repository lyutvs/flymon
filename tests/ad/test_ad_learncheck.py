"""AD.3's learning-check records (record only): right-direction share, centred V by multiplier class, the switch curve,
the self-noise null, reward / punishment totals per interval, cumulative pulses, floor contact and its pre-registered
reading, the power-vs-multiplier conflict share, the changed-weight fraction."""
import pytest

from flymon.ad import learncheck as lc
from flymon.ad.spec import SPEC

PAIRS = [dict(cands=["m0", "m1"], best1="m0", best2="m1", mults1=[2.0, 0.5], mults2=[0.5, 2.0]) for _ in range(2)]
MI = {"m0": ("WATER", 90), "m1": ("FIRE", 40)}


def rec(g, point, pos, floor=None, frac=0.0):
    sits = []
    for i, p in enumerate(pos):
        sits.append(dict(pair=i, side=0, v=[1.0, 0.0] if p else [0.0, 1.0], pick=0 if p else 1))
        sits.append(dict(pair=i, side=1, v=[0.0, 1.0] if p else [1.0, 0.0], pick=1 if p else 0))
    return dict(fly=g, point=point, arm="FLYL" if g < 36 else "FLYOS", set="confirm", mode="own", frozen=True,
                situations=sits, switched=[int(p) for p in pos], rate=sum(pos) / len(pos), floor_contact=floor,
                weights=dict(frac_changed=frac))


SITS = [rec(24, 0, [0, 0]), rec(24, 20, [1, 0], 0.2, 0.1), rec(36, 0, [0, 0]), rec(36, 20, [0, 0], 0.7, 0.3)]


def test_curves():
    assert lc.right_share(SITS, PAIRS, (0, 20))["FLYL"] == {0: dict(share=0.0, n=1), 20: dict(share=0.5, n=1)}
    sc = lc.switch_curve(SITS, (0, 20))
    assert sc["FLYL"][20] == dict(mean=0.5, per_fly={24: 0.5}) and sc["FLYOS"][20]["mean"] == 0.0
    assert lc.null_curve(SITS, PAIRS, (20,))["FLYOS"][20] == 0.0
    assert lc.floor_curve(SITS, (20,)) == {"FLYL": {20: 0.2}, "FLYOS": {20: 0.7}}
    assert lc.frac_changed(SITS, (20,))["FLYOS"][20] == pytest.approx(0.3)
    cv = lc.centred_v(SITS, PAIRS, (20,))["FLYL"][20]
    assert cv["super"]["n"] == 4 and cv["not_very"]["n"] == 4
    assert cv["super"]["mean"] == pytest.approx(0.0) and cv["not_very"]["mean"] == pytest.approx(0.0)   # one pair each way
    cs = lc.conflict_share(SITS, PAIRS, MI, (20,))["FLYL"][20]
    assert cs["n"] == 2 and cs["mult_share"] == pytest.approx(0.5)      # power picks m0; side 1's multiplier picks m1


def test_floor_reading():
    r = lc.floor_reading({"FLYL": {0: None, 20: 0.2, 40: 0.6}, "FLYOS": {0: None, 20: 0.4, 40: 0.3}}, 0.5)
    assert r == dict(values={0: None, 20: 0.4, 40: 0.6}, first_over=40, last_ok=20, read=0.5)
    assert lc.floor_reading({"FLYL": {20: 0.1}, "FLYOS": {20: 0.2}}, 0.5)["first_over"] is None


def test_pulse_tables():
    lay = SPEC.layout()
    logs = {0: [dict(kind="reinforce", battle_id="DL-f24-b003", pulses=[["PAM08", 400.0], ["PPL105", 200.0]]),
                dict(kind="reinforce", battle_id="DL-f24-b025", pulses=[["PAM08", 100.0]]),
                dict(kind="decision", battle_id="DL-f24-b025")],
            12: [dict(kind="reinforce", battle_id="DL-f36-b030", pulses=[])]}
    t = lc.pulse_tables(logs, lay, (0, 20, 40))
    assert t["by_fly"][24]["0-20"] == dict(n_reward=1, ms_reward=400.0, n_punish=1, ms_punish=200.0)
    assert t["by_fly"][24]["20-40"] == dict(n_reward=1, ms_reward=100.0, n_punish=0, ms_punish=0.0)
    assert t["by_arm"]["FLYL"]["0-20"]["ms_reward"] == 400.0 and t["by_arm"]["FLYOS"]["20-40"]["n_reward"] == 0
    assert t["cumulative"]["FLYL"][40] == 3.0 and t["cumulative"]["FLYL"][20] == 2.0 and t["cumulative"]["FLYOS"][40] == 0.0
