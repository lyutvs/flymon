import numpy as np
import pytest

from flymon.agent import policy
from flymon.agent.config import AgentConfig
from flymon.brain.config import Params

CFG = AgentConfig(params=Params(), z={"A": (10.0, 5.0), "P": (20.0, 10.0)}, readout={"A": "MBON13", "P": "MBON05"})


def test_values_are_z_a_minus_z_p():
    v = policy.values(np.array([10.0, 20.0]), np.array([20.0, 10.0]), CFG.z)
    np.testing.assert_allclose(v, [0.0, 3.0])


def test_tau_is_linear_over_the_first_20_battles_then_flat():
    assert policy.tau(0, CFG) == pytest.approx(1.0)
    assert policy.tau(19, CFG) == pytest.approx(0.2)
    assert policy.tau(10, CFG) == pytest.approx(1.0 - 0.8 * 10 / 19)
    assert policy.tau(50, CFG) == pytest.approx(0.2)


def test_eval_is_argmax_and_learn_is_softmax_sample():
    v = np.array([0.0, 5.0, 1.0])
    assert policy.choose(v, 1.0, seed=7, mode="eval") == 1
    picks = [policy.choose(v, 1.0, seed=s, mode="learn") for s in range(400)]
    assert 0.9 < picks.count(1) / 400 <= 1.0 and set(picks) <= {0, 1, 2}


def test_rule_is_stateless():
    """Safety constraint 2: the same inputs give the same pick whatever was called before, and the module holds no
    mutable state."""
    v = np.array([0.3, 0.1, 0.2])
    first = policy.choose(v, 0.5, seed=11, mode="learn")
    for s in range(100):
        policy.choose(np.random.default_rng(s).normal(size=3), 0.7, seed=s, mode="learn")
    assert policy.choose(v, 0.5, seed=11, mode="learn") == first
    mutable = [n for n, o in vars(policy).items() if not n.startswith("__") and isinstance(o, (list, dict, set, np.ndarray))]
    assert mutable == []


def test_derive_seed_is_deterministic_and_part_sensitive():
    assert policy.derive_seed(0, "f00-b000", 3, 0) == policy.derive_seed(0, "f00-b000", 3, 0)
    assert policy.derive_seed(0, "f00-b000", 3, 0) != policy.derive_seed(0, "f00-b000", 3, 1)
    assert 0 <= policy.derive_seed("x") < 2 ** 31


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        policy.choose(np.zeros(2), 1.0, seed=0, mode="greedy")
