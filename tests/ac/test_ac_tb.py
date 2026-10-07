"""FLY-TB's odour (AC.2): the uniform mean over the 16 POOL species' type sets of the E-grid odour of the move type,
rescaled to the mean total strength of those FLY odours. The four tests named test_tb_gate_* are AC.7 2b's FLY-TB
micro-smoke gate (scripts/check_ac_smoke.py runs this file); each runs on the stub codebook and, when the data are
present, on the real k2-norm codebook."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from poke_env.data.normalize import to_id_str

from flymon.agent import encode_grid as eg
from flymon.battle.pool import POOL
from tests.ac.ac_fakes import MOVE, Mv, battle, stub_tb

NPZ = Path("data/malecns.npz")


def _real():
    from flymon.ac.config import lv_codebook
    from flymon.ac.tb import TBEncoder
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    cb, _ = lv_codebook()
    return TBEncoder(Populations.from_connectome(Connectome.load(str(NPZ))), cb, "norm")


@pytest.fixture(scope="module", params=["stub", "real"])
def tb(request):
    if request.param == "real":
        if not NPZ.exists():
            pytest.skip("data/malecns.npz not built")
        return _real()
    return stub_tb()


def _fly_mean_total(tb, mtype):
    return float(np.mean([sum(eg.odour(tb.rc, tb.cb, mtype, tuple(sorted(tb.species_types[m.species])), tb.rule).values())
                          for m in POOL]))


def test_tb_gate_opponent_change_is_byte_identical(tb):
    for me in ("Blastoise", "Chansey", "Charizard"):
        for mv in [a for m in POOL if m.species == me for a in m.attacks]:
            mid = to_id_str(mv)
            ref = json.dumps(tb.odour(battle(me, "Snorlax"), Mv(mid)), sort_keys=True)
            for opp in (m.species for m in POOL):
                assert json.dumps(tb.odour(battle(me, opp), Mv(mid)), sort_keys=True) == ref


def test_tb_gate_move_type_change_differs(tb):
    for m1, m2 in itertools.combinations(MOVE, 2):
        assert tb.tb_odour(m1) != tb.tb_odour(m2)


def test_tb_gate_has_glomeruli(tb):
    for m in MOVE:
        o = tb.tb_odour(m)
        assert len(o) > 0 and all(np.isfinite(v) and v > 0 for v in o.values())


def test_tb_gate_total_within_1pct_of_fly_mean(tb):
    for m in MOVE:
        want = _fly_mean_total(tb, m)
        assert abs(sum(tb.tb_odour(m).values()) - want) <= 0.01 * want
        assert tb.fly_mean_total(m) == pytest.approx(want)


def test_tb_odour_is_the_uniform_mean_rescaled():
    tb = stub_tb()
    fly = [eg.odour(tb.rc, tb.cb, "WATER", tuple(sorted(tb.species_types[m.species])), "norm") for m in POOL]
    acc = {}
    for o in fly:
        for g, s in o.items():
            acc[g] = acc.get(g, 0.0) + s / len(fly)
    k = np.mean([sum(o.values()) for o in fly]) / sum(acc.values())
    got = tb.tb_odour("WATER")
    assert set(got) == set(acc) and all(got[g] == pytest.approx(acc[g] * k) for g in acc)
