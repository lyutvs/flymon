"""The pure rules of spec appendix O as amended by O.7 (O.7 wins over O.2-O.5). No engine, no pool.
- The validity gate both stages pass before any judgement (O.7.5).
- The seed-cluster bootstrap (O.7.2 / O.7.4): whole seeds resampled with replacement, every cell, condition and arm of
  a seed together; spec.boot_draws draws through n_rules.boot_index at spec.boot_seed; percentile CIs (n_rules._ci).
- O1 (O.7.2): mixed cells and STOP_NO_SILENT_STATE, the nature of the two states, the pathway 2x2, per-stimulus drive
  dependence, and the records (co-silence phi, per-cell P / APL / KC distributions, the APL->KC block's numbers).
- O2 (O.7.4 as amended by O.7.8): depression, the associative component, the state-flip guard, plumbing, records
  (dopamine dependence is the record DA_DEPENDENT_BY_CONSTRUCTION, not a judgement).
The presentation state is n_rules.state and the per-cell numbers n_rules.cell_stats: never redefined here (O.7.5).
Every threshold is a field of the OSpec passed in."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from .n_rules import INVALID, _ci, boot_index, cell_stats, delta, state  # noqa: F401  (INVALID, delta: re-exported)

JUDGED = "JUDGED"
STOP_NO_SILENT_STATE = "STOP_NO_SILENT_STATE"
BISTABLE_NETWORK = "BISTABLE_NETWORK"
BIMODAL_P_READOUT = "BIMODAL_P_READOUT"
GRADED = "GRADED"
READOUT_PATH = "READOUT_PATH"
KC_NETWORK_PATH = "KC_NETWORK_PATH"
BOTH_PATHS = "BOTH_PATHS"
NEITHER_PATH = "NEITHER_PATH"
FLAT = "FLAT"
INCREASING = "INCREASING"
DECREASING = "DECREASING"
NONMONOTONIC_OR_UNCLEAR = "NONMONOTONIC_OR_UNCLEAR"
DEPRESSION_PRESENT = "DEPRESSION_PRESENT"
NO_DEPRESSION = "NO_DEPRESSION"
INCONCLUSIVE = "INCONCLUSIVE"
SEPARABLE = "SEPARABLE"
NOT_SEPARABLE = "NOT_SEPARABLE"
FLIP_SUFFIX = "_STATE_FLIP_SENSITIVE"
PATH_LABELS = {(False, True): READOUT_PATH, (True, False): KC_NETWORK_PATH, (True, True): BOTH_PATHS,
               (False, False): NEITHER_PATH}                  # (APL->KC block removes, APL->non-KC block removes)


# ================================================================ the gate and the bootstrap (O.7.5)
def finite(o) -> bool:
    """Every number inside a row (nested dicts / lists) is finite; strings and booleans pass, None fails."""
    if isinstance(o, dict):
        return all(finite(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return all(finite(v) for v in o)
    if isinstance(o, (bool, np.bool_, str)):
        return True
    if isinstance(o, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(o))
    return False


def validity(rows, declared, key_of, seeds, edit_of=None) -> list:
    """O.7.5's gate: the declared Cartesian product complete, keys unique, every value finite, no undeclared seed, one
    CSC sha256 per edit and the edits' vectors mutually distinct. The reasons, [] when valid. An empty declaration is a
    defect (nothing to judge is never a pass). edit_of(row) -> the edit the row's condition declares (None: unchecked);
    a row that ran another edit is a defect, so "one sha per edit" is also one sha per condition."""
    bad = []
    if not declared:
        bad.append("the declared grid is empty: no row to judge")
    if edit_of is not None:
        wrong = sorted((key_of(r) for r in rows if r["edit"] != edit_of(r)), key=str)
        if wrong:
            bad.append(f"{len(wrong)} row(s) ran an edit their condition does not declare, e.g. {wrong[:3]}")
    cnt = Counter(key_of(r) for r in rows)
    dup = sorted((k for k, v in cnt.items() if v > 1), key=str)
    if dup:
        bad.append(f"{len(dup)} key(s) appear more than once, e.g. {dup[:3]}")
    missing = sorted(set(declared) - set(cnt), key=str)
    if missing:
        bad.append(f"{len(missing)} declared key(s) have no row, e.g. {missing[:3]}")
    extra = sorted(set(cnt) - set(declared), key=str)
    if extra:
        bad.append(f"{len(extra)} row key(s) were not declared, e.g. {extra[:3]}")
    off = sorted({int(r["seed"]) for r in rows} - {int(s) for s in seeds})
    if off:
        bad.append(f"undeclared seed(s) {off[:8]}")
    nonfinite = [key_of(r) for r in rows if not finite(r)]
    if nonfinite:
        bad.append(f"{len(nonfinite)} row(s) hold a non-finite value, e.g. {nonfinite[:3]}")
    sha = defaultdict(set)
    for r in rows:
        sha[r["edit"]].add(r["csc_sha256"])
    many = {e: len(s) for e, s in sorted(sha.items()) if len(s) != 1}
    if many:
        bad.append(f"edit(s) ran on several CSC weight vectors: {many}")
    one = [next(iter(s)) for _, s in sorted(sha.items()) if len(s) == 1]
    if len(one) != len(set(one)):
        bad.append("two edits share one CSC weight vector (an edit changed nothing)")
    return bad


def boot_weights(n: int, draws: int, seed: int) -> np.ndarray:
    """draws x n resampling weights (each row sums to 1) from n_rules.boot_index: a weighted mean over seeds is one
    seed-cluster bootstrap replicate."""
    idx = boot_index(n, draws, seed)
    W = np.zeros((int(draws), int(n)))
    np.add.at(W, (np.arange(int(draws))[:, None], idx), 1.0)
    return W / int(n)


def auc(pos, neg) -> float | None:
    """P(pos > neg) + 1/2 P(tie) (Mann-Whitney); None when a group is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return None
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


# ================================================================ O1 (O.7.2)
def cell_key(g, stim) -> str:
    return f"{float(g):g}|{stim}"


def o1_cells(spec) -> list:
    """(g, stimulus) cells; a repeated g or stimulus would collapse two declared cells into one key (ValueError)."""
    gs, st = [float(g) for g in spec.o1_g_grid], list(spec.o1_stimuli)
    if len(set(gs)) != len(gs) or len(set(st)) != len(st):
        raise ValueError(f"o1_g_grid {spec.o1_g_grid} / o1_stimuli {spec.o1_stimuli} repeat a value")
    return [(g, s) for g in gs for s in st]


def o1_edit_of(spec):
    """row -> the edit its condition declares (None for an undeclared condition: a defect in validity)."""
    edits = dict(spec.o1_conditions)
    return lambda r: edits.get(r["cond"])


def o1_key(r) -> tuple:
    return (r["cond"], float(r["g"]), r["stim"], int(r["seed"]))


def o1_declared(spec) -> set:
    return {(c, g, s, int(seed)) for c, _ in spec.o1_conditions for g, s in o1_cells(spec) for seed in spec.o1_seeds}


def silent(rows, spec) -> np.ndarray:
    return np.array([state(r["P"], spec.n) == "silent" for r in rows], bool)


def mixed_cells(S, spec) -> list:
    """Indices of the cells (rows of S, cells x seeds, 1 = silent) whose silent and firing shares are both
    >= mixed_min (reading 3)."""
    sil, fire = S.mean(1), (1.0 - S).mean(1)
    return [i for i in range(S.shape[0]) if sil[i] >= spec.mixed_min and fire[i] >= spec.mixed_min]


def cell_nature(rows, spec) -> dict:
    """(a) P mid-band share <= mid_max; (b) / (c) separation AUC (reading 5) of APL output / KC spikes, silent vs
    firing, >= auc_min."""
    s = silent(rows, spec)
    P = np.array([r["P"] for r in rows], float)
    lo, hi = spec.mid_band
    mid = float(((P >= lo) & (P < hi)).mean())
    out = dict(mid_share=mid, a=bool(mid <= spec.mid_max))
    for name, field in (("apl", "apl_out_per_step"), ("kc", "kc_spikes")):
        v = np.array([r[field] for r in rows], float)
        d = auc(v[s], v[~s])
        out[f"auc_{name}_silent_above"] = d
        out[f"auc_{name}"] = None if d is None else max(d, 1.0 - d)
    out["b"] = bool(out["auc_apl"] is not None and out["auc_apl"] >= spec.auc_min)
    out["c"] = bool(out["auc_kc"] is not None and out["auc_kc"] >= spec.auc_min)
    out["abc"] = bool(out["a"] and out["b"] and out["c"])
    return out


def nature_label(nat: list, spec) -> str:
    """BISTABLE_NETWORK if (a)(b)(c) hold in >= nature_share of the mixed cells, else BIMODAL_P_READOUT if (a) does,
    else GRADED (reading 6; integer comparison)."""
    n = len(nat)
    if n == 0:
        raise ValueError("no mixed cell to judge")
    num, den = spec.nature_share
    if den * sum(c["abc"] for c in nat) >= num * n:
        return BISTABLE_NETWORK
    if den * sum(c["a"] for c in nat) >= num * n:
        return BIMODAL_P_READOUT
    return GRADED


def pathway(S: dict, mixed: list, W, spec) -> dict:
    """O.7.2-3 over the mixed cells: pooled silent shares, and for each partial block c "removes" iff
    q_c <= path_ratio q_on and the CI of q_c - q_on lies below 0 (reading 7). The all-output block is a record."""
    on, kc, nonkc, all_ = (c for c, _ in spec.o1_conditions)
    q = {c: float(S[c][mixed].mean()) for c in (on, kc, nonkc, all_)}
    qb_on = (W @ S[on][mixed].T).mean(1)
    arms = {}
    for c in (kc, nonkc):
        ci = _ci((W @ S[c][mixed].T).mean(1) - qb_on, spec.ci_level)
        arms[c] = dict(q=q[c], diff=q[c] - q[on], diff_ci95=ci,
                       removes=bool(q[c] <= spec.path_ratio * q[on] and ci[1] < 0))
    return dict(label=PATH_LABELS[(arms[kc]["removes"], arms[nonkc]["removes"])], q=q, arms=arms, record_all=q[all_])


def drive(S, cells: list, W, spec) -> dict:
    """O.7.2-4 per stimulus: OLS slope of the cell silent share on the g rank (0 = smallest g, whatever the order of
    o1_g_grid), its slope_ci_level bootstrap CI and the range of the point estimates (reading 8)."""
    gs = sorted(float(g) for g in spec.o1_g_grid)
    if len(gs) < 2:
        raise ValueError("drive dependence needs at least two g")
    x = np.arange(len(gs), dtype=float)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    out = {}
    for stim in spec.o1_stimuli:
        M = S[[cells.index((g, stim)) for g in gs]]
        y = M.mean(1)
        slope = float(xc @ y / sxx)
        ci = _ci((W @ M.T) @ xc / sxx, spec.slope_ci_level)
        rng = float(y.max() - y.min())
        if rng < spec.flat_range:
            lab = FLAT
        elif ci[0] > 0:
            lab = INCREASING
        elif ci[1] < 0:
            lab = DECREASING
        else:
            lab = NONMONOTONIC_OR_UNCLEAR
        out[stim] = dict(label=lab, silent_share=y.tolist(), slope=slope, slope_ci=ci, range=rng)
    return out


def phi_pairs(S, cells: list, spec) -> list:
    """O.2's record: the seed-wise phi of co-silence for every pair of cells whose silent share is in phi_band."""
    lo, hi = spec.phi_band
    sh = S.mean(1)
    keep = [i for i in range(len(cells)) if lo <= sh[i] <= hi]
    return [dict(a=cell_key(*cells[i]), b=cell_key(*cells[j]), phi=float(np.corrcoef(S[i], S[j])[0, 1]),
                 same_stimulus=cells[i][1] == cells[j][1]) for i, j in combinations(keep, 2)]


def cell_record(rows, spec) -> dict:
    """n_rules.cell_stats (its per-presentation kc_frac list dropped) + the silent share, the sorted P values and the
    APL / KC quantiles by state."""
    s = silent(rows, spec)

    def qs(v):
        return [float(x) for x in np.quantile(v, spec.record_quantiles)] if v.size else None

    apl = np.array([r["apl_out_per_step"] for r in rows], float)
    kc = np.array([r["kc_spikes"] for r in rows], float)
    rec = {k: v for k, v in cell_stats(rows, spec.n).items() if k != "kc_frac"}
    return dict(rec, silent_share=float(s.mean()), P=sorted(int(r["P"]) for r in rows),
                apl_out={"silent": qs(apl[s]), "firing": qs(apl[~s])},
                kc_spikes={"silent": qs(kc[s]), "firing": qs(kc[~s])})


def _condition_view(S_c, by, cond, cells, W, spec) -> dict:
    m = mixed_cells(S_c, spec)
    nat = [dict(cell=cell_key(*cells[i]), **cell_nature(by[(cond,) + cells[i]], spec)) for i in m]
    return dict(mixed=m, mixed_cells=[cell_key(*cells[i]) for i in m],
                nature=dict(label=nature_label(nat, spec) if len(m) >= spec.mixed_cells_min else None, cells=nat),
                drive=drive(S_c, cells, W, spec))


def o1_judge(rows: list, spec) -> dict:
    """The gate (INVALID), then the records, then STOP_NO_SILENT_STATE (judgements 2-4 not made: reading 4) or the
    three judgements on the on condition."""
    bad = validity(rows, o1_declared(spec), o1_key, spec.o1_seeds, o1_edit_of(spec))
    if bad:
        return dict(outcome=INVALID, reasons=bad, n_rows=len(rows))
    cells = o1_cells(spec)
    conds = [c for c, _ in spec.o1_conditions]
    on, kc = conds[0], conds[1]
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], float(r["g"]), r["stim"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: int(r["seed"]))
    S = {c: np.array([silent(by[(c,) + cell], spec) for cell in cells], float) for c in conds}
    W = boot_weights(len(spec.o1_seeds), spec.boot_draws, spec.boot_seed)
    view_on = _condition_view(S[on], by, on, cells, W, spec)
    view_kc = _condition_view(S[kc], by, kc, cells, W, spec)
    rec = dict(reasons=[], mixed_cells=view_on["mixed_cells"],
               cells={c: {cell_key(*cell): cell_record(by[(c,) + cell], spec) for cell in cells} for c in conds},
               phi_on=phi_pairs(S[on], cells, spec),
               kc_block_record={k: view_kc[k] for k in ("mixed_cells", "nature", "drive")},
               sha={c: by[(c,) + cells[0]][0]["csc_sha256"] for c in conds})
    if len(view_on["mixed"]) < spec.mixed_cells_min:
        return dict(outcome=STOP_NO_SILENT_STATE, nature=None, pathway=None, drive=None, **rec)
    return dict(outcome=JUDGED, nature=view_on["nature"], pathway=pathway(S, view_on["mixed"], W, spec),
                drive=view_on["drive"], **rec)
