# tests/brain/r_fixtures.py
"""Fabricated oracle results for R's record / rule / runner tests (no engine). fake_oracle builds an r_oracle_job-shaped
result whose pair_stats under Z are fixed by three flags:
- r_ok: R1 lowers X's P by 10 (± 0/1) -> r ~ 18 (else P_X ± 3, mean 0 -> r ~ 0);
- p_ok: R2 lowers X's A by 10 (± 0/1) -> p ~ -18, -p >= 2 (else A_X ± 3 -> p ~ 0);
- bal: X's naive A = Y's (± noise, mean 0) -> d_pre = 0 (else X's A is 20 higher -> d_pre ~ 12).
testable <=> r_ok and p_ok; naive <=> bal; punish pass <=> p_ok; reward pass <=> r_ok. X's and Y's naive P move
together by H8 (so the P type count has a nonzero SD for z renormalisation and cancels in V_X - V_Y). The per-cell P
counts split the type count over 2 cells, so they sum to it.
fake_measurer / write_raws: a real RMeasurer (no pool, no cache) for the judgement inputs, and raw entries written
under a tmp root in RCache's layout ({key, kind, inputs, result}) so preread_validity can re-read the stored inputs."""
import json
from pathlib import Path
E8, F8, G8 = [1, -1, 2, -2, 1, -1, 2, -2], [0, 1, 0, 1, 0, 1, 0, 1], [3, -3, 3, -3, 3, -3, 3, -3]
H8 = [0, 2, 0, 2, 0, 2, 0, 2]
TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
READOUT = {"A": "MBON13", "P": "MBON05"}
Z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}


def _cyc(v, n):
    return [v[i % len(v)] for i in range(n)]


def fake_oracle(r_ok=False, p_ok=True, bal=False, sha="sha-C", edges=0, edit="none", n_rep=8, n_act=8, kc=0.05,
                px=26):
    e, f, g, h = _cyc(E8, n_rep), _cyc(F8, n_rep), _cyc(G8, n_rep), _cyc(H8, n_rep)
    ax = [(20 if bal else 40) + e[i] for i in range(n_rep)]
    pre = {"A": [[ax[i], 20] for i in range(n_rep)], "P": [[px + h[i], px + h[i]] for i in range(n_rep)]}
    px1 = [(px + h[i] - 10 + f[i]) if r_ok else (px + h[i] + g[i]) for i in range(n_rep)]
    r1 = {"A": [list(a) for a in pre["A"]], "P": [[px1[i], px + h[i]] for i in range(n_rep)]}
    ax2 = [(ax[i] - 10 + f[i]) if p_ok else (ax[i] + g[i]) for i in range(n_rep)]
    r2 = {"A": [[ax2[i], 20] for i in range(n_rep)], "P": [list(p) for p in r1["P"]]}
    side = {"frac": [kc] * n_act, "spikes": [100] * n_act, "max_win": [5] * n_act}

    def cells(j):
        return [[row[j] // 2, row[j] - row[j] // 2] for row in pre["P"]]

    return {"select": {}, "alpha_reward": 0.5, "alpha_punish": 0.8, "counts": {},
            "report": {"pre": pre, "R1": r1, "R2": r2},
            "kc": {"x": dict(side), "y": dict(side), "jaccard": 0.04},
            "q": {"edit": edit, "csc_sha256": sha, "edit_edges": edges, "fixed": {},
                  "apl_out": {"x": [0.2] * n_act, "y": [0.2] * n_act}, "fx": {"idx": [], "val": []},
                  "reach": {"W_X": 1.0, "n_edges": 3, "n_active_kc": 5, "f_X": 0.4, "by_type": {}, "total": 1.0}},
            "r": {"p_type": "MBON05", "n_cells": 2, "p_cells": {"x": cells(0), "y": cells(1)}, "read_ms": 600.0,
                  "dt": 1.0, "refrac_steps": 2}}


def fake_rows(n_b=21, n_a=32, turn0=64):
    rows = []
    for i in range(n_b):
        rows.append(dict(axis="b", turn=turn0 + i, x=f"bx{i}", y=f"by{i}", odor_x={"G": 1.0}, odor_y={"H": 1.0},
                         odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}))
    for i in range(n_a):
        rows.append(dict(axis="a", turn=turn0 + i, x=f"ax{i}", y=f"ay{i}", odor_x={"G": 1.0}, odor_y={"H": 1.0},
                         odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}))
    return rows


def key_of(r):
    return f"{r['axis']}|{int(r['turn'])}|{r['x']}|{r['y']}"


def got_for(rows, plan=None, default=(False, True, False), **kw):
    """[dict(key, result)] with fake_oracle(*plan.get(key, default), **kw)."""
    plan = plan or {}
    return [dict(key=key_of(r), result=fake_oracle(*plan.get(key_of(r), default), **kw), cache_key=f"ck{i}",
                 cache_file=f"results/r/cache/r_oracle/{i}.json") for i, r in enumerate(rows)]


def fake_measurer(spec):
    from flymon.brain.r_measure import RMeasurer
    return RMeasurer(None, None, spec, {"fake_params": 1.0}, READOUT, Z, TYPES, 10)


def write_raws(raws, rows, want, root):
    """raws {cond: got_for(rows, ...)} -> the same rows with cache_file pointing at a written entry whose `inputs` are
    want[cond][pair key]."""
    out = {}
    for n, got in raws.items():
        d = Path(root) / n
        d.mkdir(parents=True, exist_ok=True)
        out[n] = []
        for i, (g, r) in enumerate(zip(got, rows)):
            p = d / f"{i}.json"
            p.write_text(json.dumps(dict(key=g["cache_key"], kind="r_oracle", inputs=want[n][key_of(r)],
                                         result=g["result"])))
            out[n].append(dict(g, cache_file=str(p)))
    return out
