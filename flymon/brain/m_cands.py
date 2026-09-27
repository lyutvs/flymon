"""Spec M.10.1-M.10.2: the candidates are the distinct CORE CELL sets of the PAM (reward) and PPL1 (punish) DAN types
(circuits.compartments' core: MBON cells with w_mbon >= core_frac) — two DANs with the same cells are one candidate,
named by the DAN with most cells; the offline specificity metric S_c is K.8.3's S with the KC weights onto the
candidate's core cells; a candidate goes on only when its median S over the even (b) 21 pairs is strictly above its arm
incumbent's (an arm with none is fixed to its incumbent; STOP_NO_SPECIFICITY only when both arms are empty)."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .k_metrics import _kc_pos, pair_metrics
from .m_spec import SPEC as DECLARED

SPEC_GO = "SPEC_GO"
STOP_NO_SPECIFICITY = "STOP_NO_SPECIFICITY"


def cells_digest(cells) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in cells)).encode()).hexdigest()


def core_groups(cell_types, comps: dict, family: str) -> list:
    """One entry per distinct non-empty core cell set of `family`'s DAN types, sorted by representative name."""
    t = np.asarray(cell_types).astype(str)
    by: dict = {}
    for name in sorted(comps):
        cp = comps[name]
        if cp.family != family or len(cp.core) == 0:
            continue
        by.setdefault(tuple(sorted(int(i) for i in cp.core)), []).append(name)
    out = []
    for cells, names in by.items():
        rep = max(names, key=lambda n: (len(comps[n].cells), -names.index(n)))    # most cells, then first name
        out.append(dict(name=rep, dans=list(names), family=family, cells=list(cells),
                        types=sorted(set(t[list(cells)].tolist())), n_dan_cells=int(len(comps[rep].cells)),
                        digest=cells_digest(cells), w_mbon=[float(comps[rep].w_mbon[i]) for i in cells]))
    return sorted(out, key=lambda d: d["name"])


def candidates(cell_types, comps: dict, spec) -> dict:
    """{"reward": PAM groups, "punish": PPL1 groups}. ValueError unless the representatives are the DECLARED lists
    (m_spec.SPEC, M.10.1: 13 PAM / 8 PPL1) in order; only then kept to `spec`'s lists (a smoke spec filters), which
    must name declared candidates."""
    out = {"reward": core_groups(cell_types, comps, DECLARED.reward_family),
           "punish": core_groups(cell_types, comps, DECLARED.punish_family)}
    for arm, want in (("reward", DECLARED.reward_candidates), ("punish", DECLARED.punish_candidates)):
        got = tuple(c["name"] for c in out[arm])
        if got != tuple(want):
            raise ValueError(f"{arm} candidates {got} are not the declared {tuple(want)} ({DECLARED.reward_family} / "
                             f"{DECLARED.punish_family} core cell sets, M.10.1)")
    kept = {}
    for arm, keep in (("reward", spec.reward_candidates), ("punish", spec.punish_candidates)):
        extra = [n for n in keep if n not in {c["name"] for c in out[arm]}]
        if extra:
            raise ValueError(f"{arm} spec names undeclared candidates {extra} (M.10.1)")
        kept[arm] = [c for c in out[arm] if c["name"] in set(keep)]           # declared order
    return kept


def overlap(a: dict, b: dict) -> bool:
    return bool(set(a["cells"]) & set(b["cells"]))


def cell_weights(conn, pops, cells, min_weight: int) -> np.ndarray:
    """k_metrics.readout_weights on a cell list: per-KC synapse count onto `cells` over edges >= min_weight."""
    pos = _kc_pos(conn, pops)
    sel = (pos[conn.pre] >= 0) & np.isin(conn.post, np.asarray(cells, np.int64)) & (conn.w >= min_weight)
    return np.bincount(pos[conn.pre[sel]], weights=conn.w[sel].astype(np.float64), minlength=len(pops.kc))


def lobe_shares(w, lobes: dict) -> dict:
    """{lobe: w[mask].sum() / w.sum()} (0 without weight), lobes from k_metrics.lobe_masks."""
    tot = float(np.sum(w))
    return {k: (float(np.sum(w[m])) / tot if tot > 0 else 0.0) for k, m in lobes.items()}


def pair_s(fx, fy, w) -> float:
    return pair_metrics(fx, fy, w, w)["S"]


def spec_check(values: dict, spec) -> dict:
    """values: {candidate name: [S per even (b) pair]} for every candidate of `spec` (incumbents included; KeyError on
    a missing one). A candidate passes iff its median is strictly above its arm incumbent's; STOP_NO_SPECIFICITY iff
    none passes in either arm (an arm with none is fixed to its incumbent downstream)."""
    med = {k: float(np.median(v)) for k, v in values.items()}
    q = {k: [float(x) for x in np.percentile(v, [25, 75])] for k, v in values.items()}
    inc = {"reward": med[spec.incumbent_reward], "punish": med[spec.incumbent_punish]}
    arms = {"reward": (spec.reward_candidates, spec.incumbent_reward),
            "punish": (spec.punish_candidates, spec.incumbent_punish)}
    passing = {a: [n for n in names if n != i and med[n] > inc[a]] for a, (names, i) in arms.items()}
    outcome = SPEC_GO if passing["reward"] or passing["punish"] else STOP_NO_SPECIFICITY
    return dict(outcome=outcome, medians=med, quartiles=q, incumbent_median=inc, passing=passing)
