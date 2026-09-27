"""The measurement layer of spec appendix M (M.10.1-M.10.3). Task 3 adds `compose` only; the rest follows."""
from __future__ import annotations

from .m_jobs import group_counts


def compose(pre: dict, ed: dict, cells, groups: dict) -> dict:
    """pre_job + edit_job -> oracle_job's row shape, counts summed per group (groups = {name: cells})."""
    gc = lambda counts: group_counts(counts, cells, groups)
    ro = ed["readout"]
    ap = lambda g: {"A": g[ro["A"]], "P": g[ro["P"]]}
    pre_rep, r1, r2 = gc(pre["pre_rep"]), gc(ed["R1_rep"]), gc(ed["R2_rep"])
    return {"select": {"pre": gc(pre["pre_sel"]),
                       "reward": {a: {"R1": gc(v["R"]), "change": v["change"]} for a, v in ed["reward"].items()},
                       "punish": {a: {"R2": gc(v["R"]), "change": v["change"]} for a, v in ed["punish"].items()}},
            "alpha_reward": ed["alpha_reward"], "alpha_punish": ed["alpha_punish"],
            "counts": {"pre": pre_rep, "R1": r1, "R2": r2},
            "report": {"pre": ap(pre_rep), "R1": ap(r1), "R2": ap(r2)}, "kc": pre["kc"]}
