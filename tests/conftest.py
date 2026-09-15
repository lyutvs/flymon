"""Synthetic connectome builder shared by brain tests.

Layout (indices are contiguous blocks, in this order):
  ORN receptors (class 'olfactory', types ORN_<k>), ALPN, KC, MBON, DAN (PAM/PPL1 types),
  APL (one), lLN1/lLN2 (three, labelled acetylcholine on purpose), DN, MN.
Edges: ORN->ALPN, ALPN->KC (random 3 glomeruli each), KC->MBON (all pairs), DAN->MBON,
  APL->KC, KC->APL, lLN->ALPN, MBON->DN, DN->MN. Weights are synapse counts >= 1.

With disjoint_kc=True the ALPN->KC draw is restricted so that odour-specific learning is
expressible: KC i draws only from the ALPNs of glomerulus group i % 2 (group 0 = ORN_DM1 and
ORN_DA1, group 1 = ORN_VA2 and ORN_DM6; ORN_VC1's ALPNs stay unused), so the two groups drive
disjoint Kenyon-cell sets.
"""
from __future__ import annotations

import numpy as np
import pytest

from flymon.brain.connectome import Connectome


def _build(n_orn=20, n_alpn=10, n_kc=40, n_mbon=4, n_dan=4, n_dn=6, n_mn=4, seed=0, disjoint_kc=False):
    rng = np.random.default_rng(seed)
    types, cls, sc, nt, side = [], [], [], [], []

    def add(n, t, c, s, n_t, sd=None):  # t and n_t may be a value or a callable of the block index
        for i in range(n):
            types.append(t(i) if callable(t) else t)
            cls.append(c); sc.append(s); nt.append(n_t(i) if callable(n_t) else n_t)
            side.append(sd if sd else ("L" if i % 2 == 0 else "R"))

    add(n_orn, lambda i: f"ORN_{'DM1 DA1 VA2 DM6 VC1'.split()[i % 5]}", "olfactory", "cb_sensory", "acetylcholine", "M")
    add(n_alpn, lambda i: f"ALPN{i}", "ALPN", "cb_intrinsic", "acetylcholine")
    add(n_kc, lambda i: "KCab-m", "Kenyon_Cell", "cb_intrinsic", "acetylcholine")
    add(n_mbon, lambda i: f"MBON{i + 1:02d}", "MBON", "cb_intrinsic", lambda i: "acetylcholine" if i % 2 else "glutamate")
    add(n_dan, lambda i: ["PAM08", "PAM08", "PPL105", "PPL105"][i], "DAN", "cb_intrinsic", "dopamine")
    add(1, "APL", "", "cb_intrinsic", "gaba", "L")
    add(3, lambda i: ["lLN1_a", "lLN1_a", "lLN2P_a"][i], "ALLN", "cb_intrinsic", "acetylcholine")
    add(n_dn, lambda i: f"DNg{i:03d}", "", "descending_neuron", "acetylcholine")
    add(n_mn, lambda i: f"MNleg{i}", "", "vnc_motor", "acetylcholine")

    types = np.array(types); cls = np.array(cls); sc = np.array(sc); nt = np.array(nt); side = np.array(side)
    N = len(types)
    SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
    sign = np.array([SIGN.get(x, 0) for x in nt], np.int8)

    def idx(mask):
        return np.flatnonzero(mask)

    orn, alpn, kc, mbon = idx(cls == "olfactory"), idx(cls == "ALPN"), idx(cls == "Kenyon_Cell"), idx(cls == "MBON")
    dan, apl, lln = idx(cls == "DAN"), idx(types == "APL"), idx(np.char.startswith(types, "lLN"))
    dn, mn = idx(sc == "descending_neuron"), idx(sc == "vnc_motor")

    pre, post, w = [], [], []

    def edge(a, b, weight):
        pre.append(a); post.append(b); w.append(int(weight))

    glom = {t: i for i, t in enumerate(sorted(set(types[orn])))}
    for o in orn:                                  # each glomerulus -> two ALPNs
        g = glom[types[o]]
        for b in (alpn[(2 * g) % n_alpn], alpn[(2 * g + 1) % n_alpn]):
            edge(o, b, 12)
    groups = [["ORN_DM1", "ORN_DA1"], ["ORN_VA2", "ORN_DM6"]]
    group_alpn = [np.array(sorted({alpn[(2 * glom[t] + j) % n_alpn] for t in g for j in (0, 1)}))
                  for g in groups]
    for n, k in enumerate(kc):                     # 3 ALPN inputs, 6-20 synapses
        pool = group_alpn[n % len(group_alpn)] if disjoint_kc else alpn
        for a in rng.choice(pool, 3, replace=False):
            edge(a, k, rng.integers(6, 21))
        edge(k, apl[0], 4); edge(apl[0], k, 30)    # APL loop
    for k in kc:
        for m in mbon:
            edge(k, m, rng.integers(5, 15))
    for d in dan:                                  # PAM08 -> MBON01/02 strongly, PPL105 -> MBON03/04
        targets = mbon[:2] if types[d].startswith("PAM") else mbon[2:]
        for m in targets:
            edge(d, m, 60)
        edge(d, mbon[(d % 2) + 2] if types[d].startswith("PAM") else mbon[d % 2], 3)  # stray, below core
    for l in lln:
        for a in alpn:
            edge(l, a, 8)
    for m in mbon:
        for d_ in dn[:3]:
            edge(m, d_, 9)
    for d_ in dn:
        edge(d_, mn[d_ % n_mn], 7)

    return Connectome(
        bodyId=np.arange(10_000, 10_000 + N, dtype=np.int64), type=types, cls=cls, sc=sc, nt=nt,
        sign=sign, side=side, pre=np.array(pre, np.int32), post=np.array(post, np.int32), w=np.array(w, np.int32),
    )


@pytest.fixture
def synthetic_connectome():
    return _build


@pytest.fixture(scope="module")
def synthetic_npz(tmp_path_factory):
    """The disjoint-KC synthetic connectome saved as an npz, for pool workers that load from disk."""
    path = tmp_path_factory.mktemp("conn") / "synthetic.npz"
    _build(disjoint_kc=True).save(path)
    return path
