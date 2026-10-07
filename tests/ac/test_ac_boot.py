"""AC.5's unpaired two-stage percentile bootstrap: flies resampled independently per arm, units resampled inside each
drawn fly; unequal arm sizes allowed; relabelling fly ids changes nothing."""
import numpy as np
import pytest

from flymon.ac import boot


def _arm(n, p, k, seed):
    rng = np.random.default_rng(seed)
    return {f"fly{i}": list((rng.random(k) < p).astype(float)) for i in range(n)}


def test_unequal_arm_sizes():
    for na, nb in ((12, 6), (12, 16), (6, 12)):
        r = boot.two_stage_diff(_arm(na, 0.7, 20, 1), _arm(nb, 0.3, 20, 2), 2000, 304)
        assert r["n_a"] == na and r["n_b"] == nb and r["lo"] <= r["diff"] <= r["hi"] and r["diff"] > 0


def test_fly_id_relabelling_is_invariant():
    a, b = _arm(12, 0.6, 20, 3), _arm(6, 0.4, 20, 4)
    r1 = boot.two_stage_diff(a, b, 3000, 304)
    keys = list(a)
    a2 = {f"x{i}": a[k] for i, k in enumerate(reversed(keys))}
    b2 = dict(reversed(list({f"y{i}": v for i, v in enumerate(b.values())}.items())))
    assert boot.two_stage_diff(a2, b2, 3000, 304) == r1
    assert boot.two_stage_diff(list(a.values())[::-1], list(b.values()), 3000, 304) == r1


def test_seeded_and_point_estimate():
    a, b = _arm(9, 0.5, 20, 5), _arm(4, 0.5, 20, 6)
    assert boot.two_stage_diff(a, b, 1000, 304) == boot.two_stage_diff(a, b, 1000, 304)
    assert boot.two_stage_diff(a, b, 1000, 304)["diff"] == pytest.approx(boot.arm_mean(a) - boot.arm_mean(b))
    assert boot.arm_mean({"f": [1, 0], "g": [1, 1, 1, 1]}) == pytest.approx(0.75)


def test_degenerate_arms():
    r = boot.two_stage_diff({"a": [1.0] * 5, "b": [1.0] * 3}, {"c": [0.0] * 4}, 500, 304)
    assert r["diff"] == r["lo"] == r["hi"] == 1.0
    with pytest.raises(ValueError, match="no units"):
        boot.two_stage_diff({"a": []}, {"c": [0.0]}, 10, 304)


def test_meets():
    assert boot.meets({"diff": 0.2, "lo": 0.01}, 0.15)
    assert not boot.meets({"diff": 0.14, "lo": 0.05}, 0.15)
    assert not boot.meets({"diff": 0.3, "lo": 0.0}, 0.15)
    assert boot.meets({"diff": 0.01, "lo": 0.001}, 0.0)
