import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.connectome import shuffle_kc_mbon


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
