# tests/brain/q_fixtures.py
"""Fabricated oracle results and rows for the Q record / rule / runner tests (no engine)."""
import numpy as np

TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
READOUT = {"A": "MBON13", "P": "MBON05"}
Z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}


def fake_rows(n=21):
    return [dict(axis="b", turn=2 * (i // 3), x=f"x{i}", y=f"y{i}", move_x="WATER", opp_x=["FIRE"], move_y="WATER",
                 opp_y=["GRASS"], odor_x={f"ORN_A{i}": 1.25, f"ORN_B{i}": 0.75}, odor_y={f"ORN_C{i}": 1.0})
            for i in range(n)]


KEYS = [f"b|{2 * (i // 3)}|x{i}|y{i}" for i in range(21)]


def fake_result(n_rep, n_act, px, dpx, sd, seed, q=True, sha="sha-base", edges=0, kc=0.05, w_x=1.0,
                fixed=(0.8, 1.0)):
    """An oracle_job-shaped dict: A constant across phases, Y's P constant, X's P = px (+0..2) pre and pre + dpx
    (+ sd noise) after the reward edit; R2 = R1 (p = 0). With q: Q's records, the fixed arm at alpha scaled
    by alpha / 0.8."""
    rng = np.random.default_rng(seed)
    pre_p = [[int(px) + int(rng.integers(0, 3)), 10 + int(rng.integers(0, 3))] for _ in range(n_rep)]

    def edited(scale):
        return [[max(0, a + int(round(scale * dpx + sd * rng.standard_normal()))), b] for a, b in pre_p]

    r1 = edited(1.0)
    a_ = [[30 + int(rng.integers(0, 3)), 30 + int(rng.integers(0, 3))] for _ in range(n_rep)]
    zero = [[0, 0] for _ in range(n_rep)]

    def cnt(P):
        return {"MBON13": a_, "MBON18": zero, "MBON05": P, "MBON21": zero}

    side = {"frac": [kc] * n_act, "spikes": [100] * n_act, "max_win": [5] * n_act}
    res = {"select": {"pre": {}, "reward": {}, "punish": {}}, "alpha_reward": 0.5, "alpha_punish": 0.8,
           "counts": {"pre": cnt(pre_p), "R1": cnt(r1), "R2": cnt(r1)},
           "report": {"pre": {"A": a_, "P": pre_p}, "R1": {"A": a_, "P": r1}, "R2": {"A": a_, "P": r1}},
           "kc": {"x": dict(side), "y": dict(side), "jaccard": 0.04}}
    if q:
        fx = {}
        for a in fixed:
            pa = edited(a / 0.8)
            fx[str(float(a))] = {"counts": cnt(pa), "report": {"A": a_, "P": pa}}
        res["q"] = {"edit": "none", "csc_sha256": sha, "edit_edges": edges, "fixed": fx,
                    "apl_out": {"x": [0.1] * n_act, "y": [0.1] * n_act}, "fx": {"idx": [], "val": []},
                    "reach": {"W_X": w_x, "n_edges": 3, "n_active_kc": 5, "f_X": 0.4,
                              "by_type": {"MBON05": w_x, "MBON21": 0.1}, "total": w_x + 0.1}}
    return res
