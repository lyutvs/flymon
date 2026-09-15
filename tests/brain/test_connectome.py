import dataclasses
from pathlib import Path

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


def _dense_reference(c, p, apl, kc):
    """M[post, pre] = sign[pre] * w * mv_per_synapse, APL rows scaled by apl_scale, KC->KC entries
    scaled by kc_kc_scale (0 drops them), edges below min_weight or from sign-0 cells skipped."""
    M = np.zeros((c.N, c.N), np.float64)
    sign, _ = apply_sign_override(c, p)
    apl_set, kc_set = set(apl.tolist()), set(kc.tolist())
    for a, b, w in zip(c.pre, c.post, c.w):
        s = sign[a]
        if s == 0 or w < p.min_weight:
            continue
        scale = p.apl_scale if a in apl_set else 1.0
        if a in kc_set and b in kc_set:
            scale *= p.kc_kc_scale
        M[b, a] += s * w * p.mv_per_synapse * scale
    return M


def _with_kc_kc_edges(c, n_edges=30, seed=0):
    """The synthetic fixture has no KC->KC edges; append `n_edges` of them (weights 1..9) so the
    kc_kc_scale rule has something to act on."""
    rng = np.random.default_rng(seed)
    kc = np.flatnonzero(c.cls == "Kenyon_Cell")
    a, b = rng.choice(kc, n_edges), rng.choice(kc, n_edges)
    keep = a != b
    return dataclasses.replace(c, pre=np.concatenate([c.pre, a[keep]]).astype(c.pre.dtype),
                               post=np.concatenate([c.post, b[keep]]).astype(c.post.dtype),
                               w=np.concatenate([c.w, rng.integers(1, 10, keep.sum())]).astype(c.w.dtype))


def test_csc_matches_dense(synthetic_connectome):
    c = _with_kc_kc_edges(synthetic_connectome())
    apl, kc = np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell")
    for scale in (0.0, 0.5, 1.0):
        p = Params(min_weight=1, balance_hemispheres=False, kc_kc_scale=scale)
        csc = build_csc(c, p, apl, kc)
        M = _dense_reference(c, p, apl, kc)
        for i in range(c.N):
            col = np.zeros(c.N)
            np.add.at(col, csc.tgt[csc.ptr[i]:csc.ptr[i + 1]], csc.w[csc.ptr[i]:csc.ptr[i + 1]])
            np.testing.assert_allclose(col, M[:, i], atol=1e-6)


def test_kc_kc_scale_zero_drops_exactly_the_kc_kc_edges(synthetic_connectome):
    """kc_kc_scale=0 removes the KC->KC edges from the CSC (not merely zeroes them): the edge count
    falls by exactly the number of KC->KC edges that pass min_weight, and nothing else moves."""
    c = _with_kc_kc_edges(synthetic_connectome())
    apl, kc = np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell")
    p1, p0 = Params(min_weight=3), Params(min_weight=3, kc_kc_scale=0.0)
    assert p1.kc_kc_scale == 0.0 and Params(kc_kc_scale=1.0).kc_kc_scale == 1.0   # default is off
    csc1, csc0 = build_csc(c, Params(min_weight=3, kc_kc_scale=1.0), apl, kc), build_csc(c, p0, apl, kc)
    is_kc = np.zeros(c.N, bool); is_kc[kc] = True
    sign, _ = apply_sign_override(c, p1)
    n_kc_kc = int((is_kc[c.pre] & is_kc[c.post] & (c.w >= 3) & (sign[c.pre] != 0)).sum())
    assert n_kc_kc > 0
    assert len(csc0.w) == len(csc1.w) - n_kc_kc
    pre0 = csc0.pre_of_edge()
    assert not (is_kc[pre0] & is_kc[csc0.tgt]).any()
    # every non-KC->KC edge is untouched: same (pre, tgt, w) multiset
    pre1 = csc1.pre_of_edge()
    keep = ~(is_kc[pre1] & is_kc[csc1.tgt])
    np.testing.assert_array_equal(pre1[keep], pre0)
    np.testing.assert_array_equal(csc1.tgt[keep], csc0.tgt)
    np.testing.assert_array_equal(csc1.w[keep], csc0.w)


def _build_csc_m0(conn, params, apl_idx):
    """The M0/M0b build_csc body, verbatim (no KC->KC rule): the reference for kc_kc_scale=1.0."""
    sign, _ = apply_sign_override(conn, params)
    keep = (conn.w >= params.min_weight) & (sign[conn.pre] != 0)
    pre, post = conn.pre[keep], conn.post[keep]
    mv = conn.w[keep].astype(np.float32)
    mv *= sign[pre].astype(np.float32)
    mv *= np.float32(params.mv_per_synapse)
    if params.balance_hemispheres:
        m = conn.side[post] == "R"
        mv[m] *= np.float32(hemisphere_scale(conn))
    is_apl = np.zeros(conn.N, bool)
    is_apl[apl_idx] = True
    m = is_apl[pre]
    mv[m] *= np.float32(params.apl_scale)
    order = np.argsort(pre, kind="stable")
    pre, post, mv = pre[order], post[order], mv[order]
    counts = np.bincount(pre, minlength=conn.N)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ptr, post.astype(np.int32), mv


def test_kc_kc_scale_one_is_the_m0_csc_bit_for_bit(synthetic_connectome):
    """kc_kc_scale=1.0 must reproduce the M0/M0b CSC exactly (same float32 operation order, no extra
    multiply), so the old result files stay bit-exact reproduction references."""
    c = _with_kc_kc_edges(synthetic_connectome())
    apl, kc = np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell")
    p = Params(min_weight=1, kc_kc_scale=1.0)
    csc = build_csc(c, p, apl, kc)
    ptr, tgt, w = _build_csc_m0(c, p, apl)
    assert np.array_equal(csc.ptr, ptr) and np.array_equal(csc.tgt, tgt) and np.array_equal(csc.w, w)


def test_csc_drops_below_threshold_and_zero_sign(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(min_weight=5)
    csc = build_csc(c, p, np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell"))
    pre = csc.pre_of_edge()
    sign, _ = apply_sign_override(c, p)
    assert (sign[pre] != 0).all()
    assert (np.abs(csc.w) >= 5 * p.mv_per_synapse * min(1.0, p.apl_scale) - 1e-9).all()


def test_load_rejects_out_of_range_edge_index(tmp_path, synthetic_connectome):
    c = synthetic_connectome()
    c.pre[0] = -1
    path = tmp_path / "neg.npz"
    c.save(path)
    with pytest.raises(ValueError, match="out of range"):
        Connectome.load(path)


def test_sign_override_counts_only_actual_changes(synthetic_connectome):
    c = synthetic_connectome()
    lln = np.char.startswith(c.type, "lLN")
    c.sign[lln] = -1  # already inhibitory: the override has nothing to change
    sign, n = apply_sign_override(c, Params())
    assert n == 0
    np.testing.assert_array_equal(sign, c.sign)


@pytest.mark.skipif(not Path("data/malecns.npz").exists(), reason="MaleCNS connectome not built")
def test_real_kc_kc_scale_edge_counts():
    """On MaleCNS the M0 engine (kc_kc_scale=1.0) has 6,005,611 CSC edges and the M0c engine (0.0) exactly 33,247 fewer:
    the KC->KC edges at min_weight 5 (spec D.1). Fixed numbers so a later change to the rule cannot pass unnoticed."""
    from flymon.brain.circuits import Populations
    c = Connectome.load("data/malecns.npz")
    pops = Populations.from_connectome(c)
    n_old = len(build_csc(c, Params(kc_kc_scale=1.0), pops.apl, pops.kc).w)
    n_new = len(build_csc(c, Params(), pops.apl, pops.kc).w)
    assert (n_old, n_new) == (6_005_611, 5_972_364)
