import math
import pytest
from flymon.rescope import rules as R
from flymon.rescope.spec import SPEC
from tests.rescope.rescope_fixtures import records

CASES = [records(dx=12, noise=2), records(dx=1, dy=-10, noise=0.5), records(dx=-0.5, dy=-10, noise=0.5),
         records(dx=6, null=30, noise=2), records(p_naive=30, dx=12, noise=2)]
# case 4: make Y's naive MBON05 low so only the Y floor catches it (noplast's S1 too, so its S1 still equals its pre)
for r in CASES[4]:
    if r["stage"] == "pre" or r["brain"] == "noplast":
        r["counts"]["b"][SPEC.p_type] = 3

MUTANTS = {
    "no_inf_on_nonpositive_ex": ("_spill", lambda ex, ey: 0.0 if ex <= 0 else abs(ey) / ex),
    "clip_instead_of_abs": ("_spill", lambda ex, ey: math.inf if ex <= 0 else max(0.0, ey / ex)),
    "no_null_compare": ("_beats_null", lambda med, null_max: True),
    "no_y_floor": ("_floor_odours", lambda x, y: (x,)),
}

def verdicts():
    return [R.pair_verdict(c, "a", SPEC, "p1000")["status"] for c in CASES]

@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name):
    base = verdicts()
    attr, fn = MUTANTS[name]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, attr, fn)
        assert verdicts() != base, f"mutant {name} survived"

def test_threshold_mutant_killed():
    assert [math.ceil(2 * m / 3) for m in (4, 6)] != [R.threshold_t(m, SPEC) for m in (4, 6)]
