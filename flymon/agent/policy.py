"""The fly's decision rule (spec 3.5): V = z_A - z_P from the readout counts; softmax(V / tau) while learning, with
tau linear 1.0 -> 0.2 over the first 20 battles; argmax in evaluation. No learning state: every random draw comes from
a seed derived from (fly, battle, turn, ...), never from a generator kept between calls (safety constraint 2)."""
from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2 ** 31)


def values(a_counts, p_counts, z: dict) -> np.ndarray:
    a, p = np.asarray(a_counts, float), np.asarray(p_counts, float)
    return (a - z["A"][0]) / z["A"][1] - (p - z["P"][0]) / z["P"][1]


def tau(battle_index: int, cfg) -> float:
    k = min(max(int(battle_index), 0), cfg.tau_battles - 1)
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * k / (cfg.tau_battles - 1)


def choose(v, tau_value: float, seed: int, mode: str) -> int:
    v = np.asarray(v, float)
    if mode == "eval":
        return int(np.argmax(v))
    if mode != "learn":
        raise ValueError(f"mode must be 'learn' or 'eval', got {mode!r}")
    x = (v - v.max()) / float(tau_value)
    p = np.exp(x) / np.exp(x).sum()
    return int(np.random.default_rng(int(seed)).choice(len(v), p=p))


def argmax_tiebreak(v, seed: int) -> tuple:
    """Spec AC.2: argmax, with a tie (two or more candidates at the maximum V) broken uniformly at random from `seed`;
    returns (index, tied). No state is kept between calls."""
    v = np.asarray(v, float)
    top = np.flatnonzero(v == v.max())
    if top.size == 1:
        return int(top[0]), False
    return int(np.random.default_rng(int(seed)).choice(top)), True
