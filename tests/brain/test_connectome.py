import numpy as np
import pytest

from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, apply_sign_override, build_csc, hemisphere_scale


def test_roundtrip_npz(tmp_path, synthetic_connectome):
    c = synthetic_connectome()
    path = tmp_path / "c.npz"
    c.save(path)
    d = Connectome.load(path)
    assert d.N == c.N and d.E == c.E
    np.testing.assert_array_equal(d.pre, c.pre)
    assert d.type.dtype.kind == "U"


def test_load_rejects_bad_schema(tmp_path):
    np.savez(tmp_path / "bad.npz", bodyId=np.arange(3))
    with pytest.raises(ValueError, match="missing"):
        Connectome.load(tmp_path / "bad.npz")


def test_sign_override_counts_prefix_matches(synthetic_connectome):
    c = synthetic_connectome()
    # synthetic builder labels two cells 'lLN1_a' and one 'lLN2P_a' as acetylcholine (+1)
    sign, n = apply_sign_override(c, Params())
    assert n == 3
    assert (sign[np.char.startswith(c.type, "lLN")] == -1).all()
    # untouched elsewhere
    other = ~np.char.startswith(c.type, "lLN")
    np.testing.assert_array_equal(sign[other], c.sign[other])


def test_hemisphere_scale_is_inL_over_inR(synthetic_connectome):
    c = synthetic_connectome()
    inL = c.w[c.side[c.post] == "L"].sum()
    inR = c.w[c.side[c.post] == "R"].sum()
    assert hemisphere_scale(c) == pytest.approx(inL / inR)


def test_csc_matches_dense(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(min_weight=1, balance_hemispheres=False)
    apl = np.flatnonzero(c.type == "APL")
    csc = build_csc(c, p, apl)
    # dense reference: M[post, pre] = sign[pre] * w * mv_per_synapse, APL rows scaled
    M = np.zeros((c.N, c.N), np.float64)
    sign, _ = apply_sign_override(c, p)
    for a, b, w in zip(c.pre, c.post, c.w):
        s = sign[a]
        if s == 0 or w < p.min_weight:
            continue
        scale = p.apl_scale if a in set(apl.tolist()) else 1.0
        M[b, a] += s * w * p.mv_per_synapse * scale
    for i in range(c.N):
        col = np.zeros(c.N)
        np.add.at(col, csc.tgt[csc.ptr[i]:csc.ptr[i + 1]], csc.w[csc.ptr[i]:csc.ptr[i + 1]])
        np.testing.assert_allclose(col, M[:, i], atol=1e-6)


def test_csc_drops_below_threshold_and_zero_sign(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(min_weight=5)
    csc = build_csc(c, p, np.flatnonzero(c.type == "APL"))
    pre = csc.pre_of_edge()
    sign, _ = apply_sign_override(c, p)
    assert (sign[pre] != 0).all()
    assert (np.abs(csc.w) >= 5 * p.mv_per_synapse * min(1.0, p.apl_scale) - 1e-9).all()
