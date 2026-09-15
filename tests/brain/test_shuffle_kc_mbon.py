import dataclasses

import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import build_csc, shuffle_kc_mbon


def test_shuffle_permutes_kc_side_of_kc_mbon_edges_only(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    s = shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3)
    assert s.E == c.E and s.N == c.N
    np.testing.assert_array_equal(s.post, c.post)
    np.testing.assert_array_equal(s.w, c.w)
    m = np.isin(c.pre, pops.kc) & np.isin(c.post, pops.mbon)
    np.testing.assert_array_equal(s.pre[~m], c.pre[~m])            # every other edge untouched
    assert (s.pre[m] != c.pre[m]).any() and np.isin(s.pre[m], pops.kc).all()
    out_before = np.bincount(c.pre[m], weights=c.w[m], minlength=c.N)[pops.kc]
    out_after = np.bincount(s.pre[m], weights=s.w[m], minlength=c.N)[pops.kc]
    np.testing.assert_array_equal(np.sort(out_before), np.sort(out_after))   # KC out-weights are permuted, not changed
    np.testing.assert_array_equal(shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3).pre, s.pre)
    assert (shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=4).pre != s.pre).any()


def test_shuffle_rejects_connectome_without_kc_mbon_edges(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    with pytest.raises(ValueError, match="no KC->MBON edges"):
        shuffle_kc_mbon(c, pops.kc, pops.mbon[:0], seed=1)


def _kc_input_per_mbon(csc, pops, N):
    """Total signed KC->* weight arriving at every MBON, at the CSC level (what the engine actually runs)."""
    pre = csc.pre_of_edge()
    sel = np.isin(pre, pops.kc)
    return np.bincount(csc.tgt[sel], weights=csc.w[sel], minlength=N)[pops.mbon]


def test_shuffle_preserves_kc_input_per_mbon_at_the_csc_level(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    p = Params(min_weight=1, balance_hemispheres=False)
    csc0 = build_csc(c, p, pops.apl)
    csc1 = build_csc(shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3), p, pops.apl)
    np.testing.assert_allclose(_kc_input_per_mbon(csc0, pops, c.N), _kc_input_per_mbon(csc1, pops, c.N), rtol=1e-6)


def test_shuffle_rejects_a_kc_population_with_a_mixed_or_zero_sign(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    c = dataclasses.replace(c, sign=c.sign.copy())
    c.sign[pops.kc[0]] = 0
    with pytest.raises(ValueError, match="sign-homogeneous"):
        shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3)
