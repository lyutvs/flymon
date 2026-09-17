"""Per-KC threshold files for the homeostatic threshold mode (spec appendix H.2, H.3).

A file is an npz with `kc_body_ids` (int64, the connectome's KC order) and `v_th` (float32, mV above rest).
The engine only accepts a file whose sha256 matches `Params.kc_thresh_sha256`, whose body ids match its KC
population in order, and which leaves every KC without PN input at the PN-normalisation rule's value.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def save_kc_thresholds(path, kc_body_ids, v_th) -> tuple[Path, str]:
    """Write the file and return (path, sha256 of its bytes)."""
    path = Path(path)
    ids = np.asarray(kc_body_ids, np.int64)
    th = np.asarray(v_th, np.float32)
    if ids.shape != th.shape:
        raise ValueError(f"kc_body_ids {ids.shape} and v_th {th.shape} differ in shape")
    with open(path, "wb") as f:                     # a file handle: np.savez would append .npz to a str path
        np.savez(f, kc_body_ids=ids, v_th=th)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_kc_thresholds(path: str, sha256: str, kc_body_ids, rule_v_th, has_pn_input) -> np.ndarray:
    """Validated thresholds in KC order (float32). `rule_v_th` is the PN-normalisation rule's value per KC and
    `has_pn_input` marks KCs with any PN input; KCs without PN input must keep the rule's value."""
    if not path:
        raise ValueError("kc_thresh_mode='homeostatic' needs kc_thresh_file")
    if not sha256:
        raise ValueError("kc_thresh_mode='homeostatic' needs kc_thresh_sha256")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"kc_thresh_file sha256 {digest} != kc_thresh_sha256 {sha256}")
    with np.load(path) as d:
        ids, th = d["kc_body_ids"].astype(np.int64), d["v_th"].astype(np.float32)
    want = np.asarray(kc_body_ids, np.int64)
    if ids.shape != want.shape:
        raise ValueError(f"threshold file holds {ids.size} KCs, the connectome has {want.size}")
    if not np.array_equal(ids, want):
        raise ValueError("threshold file body ids differ from the connectome's KC order")
    if not np.isfinite(th).all() or (th <= 0).any():
        raise ValueError("thresholds must be finite and positive")
    fixed = ~np.asarray(has_pn_input, bool)
    if not np.allclose(th[fixed], np.asarray(rule_v_th, np.float32)[fixed], rtol=0, atol=1e-6):
        raise ValueError("thresholds of KCs with no PN input must keep the PN-normalisation rule's value")
    return th
