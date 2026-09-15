"""Connectome arrays, our npz schema, sign/hemisphere corrections, CSC out-edge build."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import Params

SCHEMA = {
    "bodyId": np.int64, "type": None, "cls": None, "sc": None, "nt": None,
    "sign": np.int8, "side": None, "pre": np.int32, "post": np.int32, "w": np.int32,
}
SIGN_OF_NT = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}


@dataclass
class Connectome:
    bodyId: np.ndarray
    type: np.ndarray
    cls: np.ndarray
    sc: np.ndarray
    nt: np.ndarray
    sign: np.ndarray
    side: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    w: np.ndarray

    @property
    def N(self) -> int:
        return int(self.bodyId.shape[0])

    @property
    def E(self) -> int:
        return int(self.pre.shape[0])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in SCHEMA})

    @classmethod
    def load(cls, path: str | Path) -> "Connectome":
        d = np.load(path, allow_pickle=False)
        missing = [k for k in SCHEMA if k not in d.files]
        if missing:
            raise ValueError(f"npz missing arrays: {missing}")
        arrs = {}
        for k, dt in SCHEMA.items():
            a = d[k]
            arrs[k] = a.astype(dt) if dt is not None else a.astype(str)
        c = cls(**arrs)
        if (
            c.pre.min(initial=0) < 0
            or c.post.min(initial=0) < 0
            or c.pre.max(initial=-1) >= c.N
            or c.post.max(initial=-1) >= c.N
        ):
            raise ValueError("edge index out of range")
        return c


def apply_sign_override(conn: Connectome, params: Params) -> tuple[np.ndarray, int]:
    """Re-sign cell types by prefix (spec 3.1: lLN1/lLN2 are inhibitory). Returns (sign, n_changed).

    n_changed counts cells whose sign the override actually altered, each cell once
    even if several prefixes match it.
    """
    sign = conn.sign.astype(np.int8).copy()
    changed = np.zeros(sign.shape, bool)
    for prefix, s in params.sign_override:
        m = np.char.startswith(conn.type, prefix)
        changed |= m & (sign != s)
        sign[m] = s
    return sign, int(changed.sum())


def hemisphere_scale(conn: Connectome) -> float:
    """inL / inR: total synapse count onto left vs right postsynaptic cells. Applied to R inputs."""
    in_l = float(conn.w[conn.side[conn.post] == "L"].sum())
    in_r = float(conn.w[conn.side[conn.post] == "R"].sum())
    return in_l / in_r if in_l > 0 and in_r > 0 else 1.0


@dataclass
class CSC:
    """Out-edges grouped by presynaptic neuron: targets tgt[ptr[i]:ptr[i+1]] with signed mV weights."""
    ptr: np.ndarray
    tgt: np.ndarray
    w: np.ndarray

    @property
    def N(self) -> int:
        return int(self.ptr.shape[0] - 1)

    def pre_of_edge(self) -> np.ndarray:
        return np.repeat(np.arange(self.N, dtype=np.int32), np.diff(self.ptr))


def build_csc(conn: Connectome, params: Params, apl_idx: np.ndarray, kc_idx: np.ndarray) -> CSC:
    """Signed mV out-edges. Edges below `min_weight` synapses or from sign-0 cells are dropped; APL
    out-edges are scaled by `apl_scale`; KC->KC edges are scaled by `kc_kc_scale` and, when that is 0,
    dropped from the CSC altogether (spec appendix D). Right-hemisphere inputs get the hemisphere factor."""
    if not math.isfinite(params.kc_kc_scale):
        raise ValueError(f"kc_kc_scale must be finite, got {params.kc_kc_scale!r}")
    sign, _ = apply_sign_override(conn, params)
    keep = (conn.w >= params.min_weight) & (sign[conn.pre] != 0)
    kc_kc = None
    if params.kc_kc_scale != 1.0:      # 1.0 is the M0/M0b engine: no mask, no multiply, bit-identical CSC
        is_kc = np.zeros(conn.N, bool)
        is_kc[np.asarray(kc_idx, dtype=np.int64)] = True
        kc_kc = is_kc[conn.pre] & is_kc[conn.post]
        if params.kc_kc_scale == 0.0:
            keep &= ~kc_kc
    pre, post = conn.pre[keep], conn.post[keep]
    # float32 end to end and in-place masked multiplies: at 100M+ edges each float64 temporary
    # costs ~1 GB, and np.where would allocate a second full-length array per correction
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
    if kc_kc is not None and params.kc_kc_scale != 0.0:
        m = kc_kc[keep]
        mv[m] *= np.float32(params.kc_kc_scale)
    order = np.argsort(pre, kind="stable")
    pre, post, mv = pre[order], post[order], mv[order]
    counts = np.bincount(pre, minlength=conn.N)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return CSC(ptr=ptr, tgt=post.astype(np.int32), w=mv)


def shuffle_kc_mbon(conn: Connectome, kc: np.ndarray, mbon: np.ndarray, seed: int) -> Connectome:
    """C-shuf control (spec 4.1): permute the presynaptic Kenyon cell of every KC->MBON edge with one
    permutation over the KC population, so the odour code reaches the MBONs through scrambled wiring.
    Edge count, the weight multiset and every MBON's total KC input are preserved; every other edge is
    untouched. Returns a new Connectome (arrays other than `pre` are shared)."""
    is_kc = np.zeros(conn.N, bool); is_kc[kc] = True
    is_mb = np.zeros(conn.N, bool); is_mb[mbon] = True
    m = is_kc[conn.pre] & is_mb[conn.post]
    if not m.any():
        raise ValueError("no KC->MBON edges to shuffle")
    signs = np.unique(conn.sign[kc])
    if signs.size != 1 or signs[0] == 0:
        raise ValueError("shuffle_kc_mbon needs a sign-homogeneous, non-zero Kenyon cell population "
                         "(build_csc keeps and signs edges by their presynaptic cell); "
                         f"got signs {signs.tolist()}")
    perm = np.random.default_rng(int(seed)).permutation(len(kc))
    mapping = np.arange(conn.N, dtype=conn.pre.dtype)
    mapping[kc] = kc[perm]
    pre = conn.pre.copy()
    pre[m] = mapping[pre[m]]
    return replace(conn, pre=pre)
