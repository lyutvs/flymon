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
DA_DEPENDENT_BY_CONSTRUCTION = "DA_DEPENDENT_BY_CONSTRUCTION"   # O.7.8: a record, never a judged label
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


# ================================================================ O2 (O.7.4 as amended by O.7.8)
def o2_key(r) -> tuple:
    return (r["x"], r["arm"], int(r["seed"]))


def o2_declared(spec) -> set:
    return {(x, a[0], int(s)) for x, _, _ in spec.o2_pairs for a in spec.o2_arms for s in spec.o2_seeds}


def o2_edit_of(spec):
    """row -> the edit O2 declares for it: every O2 arm runs the on rig (O.3: 켬 조건, 편집 없음)."""
    return lambda r: spec.on_edit


def _diff(rows, side: str, field: str) -> np.ndarray:
    return np.array([r["post"][side][field] - r["pre"][side][field] for r in rows], float)


def _inside(ci, b: float) -> bool:
    return bool(-b <= ci[0] and ci[1] <= b)


def o2_stats(by: dict, z: dict, o: float, spec, keep=None) -> dict | None:
    """by {arm: rows in seed order}, arms named as spec.o2_arms (1 plastic, 2 frozen, 3 punish, 4 da_zero). On the seeds
    in `keep` (all when None); None with fewer than 2. One seed resample (boot_weights) for every statistic.
    Judged labels: depression (O.7.4-1) and separable (O.7.4-3). m4 (O.7.4-2) is computed and recorded only: O.7.8 made
    dopamine dependence a consequence of the rule (plasticity opens only on phasic dopamine), not a judgement."""
    one, two, three, four = (a[0] for a in spec.o2_arms)
    if keep is not None:
        by = {a: [r for r in rows if int(r["seed"]) in keep] for a, rows in by.items()}
    n = len(by[one])
    if n < 2:
        return None
    W = boot_weights(n, spec.boot_draws, spec.boot_seed)

    def stat(v):
        return float(np.mean(v)), _ci(W @ v, spec.ci_level)

    dA = {a: _diff(rows, "x", "A") for a, rows in by.items()}
    b = spec.depression_frac * float(np.mean([r["pre"]["x"]["A"] for r in by[one]]))
    m, m_ci = stat(dA[one] - dA[two])
    if m_ci[1] < 0 and m <= -b:
        dep = DEPRESSION_PRESENT
    elif _inside(m_ci, b):
        dep = NO_DEPRESSION
    else:
        dep = INCONCLUSIVE
    m4, m4_ci = stat(dA[four] - dA[two])                                   # record (O.7.8)
    ddv = {a: np.array([delta(r, z) for r in rows]) for a, rows in by.items()}
    e = spec.sep_frac * abs(float(o))
    D, D_ci = stat(ddv[three] - ddv[one])
    if (D_ci[0] > 0 or D_ci[1] < 0) and abs(D) >= e:
        sep = SEPARABLE
    elif _inside(D_ci, e):
        sep = NOT_SEPARABLE
    else:
        sep = INCONCLUSIVE
    same = {f"{side}_{f}": stat(_diff(by[one], side, f) - _diff(by[two], side, f))
            for side, f in (("x", "P"), ("y", "A"), ("y", "P"))}
    return dict(n=n, seeds=[int(r["seed"]) for r in by[one]], b=b, e=e, m=m, m_ci=m_ci, m4=m4, m4_ci=m4_ci, D=D,
                D_ci=D_ci, D_sign=int(np.sign(D)), labels=dict(depression=dep, separable=sep),
                mean_dA={a: float(v.mean()) for a, v in dA.items()},
                same_differences={k: dict(mean=v[0], ci=v[1]) for k, v in same.items()})


def plumbing(by: dict, spec) -> list:
    """O.7.4-5 + O.7.8: arm 2 (plasticity off) and arm 4 (rule dopamine held at 0) must end with the plastic weights
    bit-identical to w0 (sha256), every seed. The reasons, [] when clean."""
    _, two, _, four = (a[0] for a in spec.o2_arms)
    return [f"plumbing: arm {a} seed {r['seed']} moved its weights (sha256 {r['w_post_sha256'][:12]} != w0 "
            f"{r['w0_sha256'][:12]})" for a in (two, four) for r in by[a] if r["w_post_sha256"] != r["w0_sha256"]]


def o2_x(by: dict, z: dict, o: float, spec) -> dict:
    """One X: the plumbing check (arms 2 and 4, else INVALID), the judgements, the state-flip guard (O.7.4-4: judged
    labels only; the dopamine record is not compared) and the records."""
    one, two, three, four = (a[0] for a in spec.o2_arms)
    bad = plumbing(by, spec)
    if bad:
        return dict(outcome=INVALID, reasons=bad, labels=None)
    st = o2_stats(by, z, o, spec)
    if st is None:
        return dict(outcome=INVALID, reasons=[f"O2 needs at least 2 seeds per arm, got {len(by[one])}"], labels=None)
    flips = {a: [int(r["seed"]) for r in rows if state(r["pre"]["x"]["P"], spec.n) != state(r["post"]["x"]["P"],
                                                                                              spec.n)]
             for a, rows in by.items()}
    limit = spec.flip_max_frac * st["n"]
    labels, guard = dict(st["labels"]), None
    if any(len(v) > limit for v in flips.values()):
        drop = sorted(set().union(*(set(v) for v in flips.values())))
        alt = o2_stats(by, z, o, spec, keep=set(st["seeds"]) - set(drop))
        alt_labels = alt["labels"] if alt else {k: None for k in labels}
        for k, v in labels.items():
            if v is not None and alt_labels[k] != v:
                labels[k] = v + FLIP_SUFFIX
        guard = dict(dropped=drop, alt=alt)
    firing = set(st["seeds"])
    for rows in by.values():
        for r in rows:
            if any(state(r[ph][k]["P"], spec.n) != "firing" for ph in ("pre", "post") for k in ("x", "y")):
                firing.discard(int(r["seed"]))
    record = dict(
        da=dict(label=DA_DEPENDENT_BY_CONSTRUCTION if st["labels"]["depression"] == DEPRESSION_PRESENT else None,
                m4=st["m4"], m4_ci=st["m4_ci"],
                note="O.7.8: 가소성은 위상 도파민으로만 열린다 — ④는 구조상 ②와 같다. 판정이 아니라 규칙의 귀결(기록)"),
        frozen_dA_all_zero=bool(all(r["post"]["x"]["A"] == r["pre"]["x"]["A"] for r in by[two])),
        frozen_probes_identical=bool(all(r["pre"][k][f] == r["post"][k][f] for r in by[two] for k in ("x", "y")
                                         for f in ("A", "P", "kc_spikes"))),
        da_zero_weights_unmoved=bool(all(r["w_post_sha256"] == r["w0_sha256"] for r in by[four])),
        state_conditional=dict(n=len(firing), stats=o2_stats(by, z, o, spec, keep=firing)),
        da_integral={a: {k: float(np.mean([r["da_integral"][k] for r in rows])) for k in rows[0]["da_integral"]}
                     for a, rows in by.items()},
        weights_frac={a: {k: float(np.mean([r[f] for r in rows]))
                          for k, f in (("all", "weights_frac"), ("A_punish_core", "weights_frac_A"),
                                       ("P_reward_core", "weights_frac_P"))}
                      for a, rows in by.items()},
        weights_frac_note="A·P = 처벌·보상 구획(가르친 구획)의 core 가중치 비율 — 판독 세포(MBON13·MBON05)가 아니다",
        a_x_course=None, a_x_course_note="훈련 중 프로브가 rig에 없다(N2 훈련 루프) — 제시 횟수에 따른 A_X 경과는 기록하지 않는다")
    return dict(outcome=JUDGED, reasons=[], labels=labels, stats=st, flips=flips, flip_limit=limit, guard=guard,
                record=record)


def o2_judge(rows: list, z: dict, o_by_pair: dict, spec) -> dict:
    """The gate (the stage INVALID; O2's edit mapping included, and every pair's oracle o present and finite), then
    every X of spec.o2_pairs with its pair's o (reading 15)."""
    bad = validity(rows, o2_declared(spec), o2_key, spec.o2_seeds, o2_edit_of(spec))
    for _, _, pair in spec.o2_pairs:
        if pair not in o_by_pair or not finite(o_by_pair[pair]):
            bad.append(f"oracle o for pair {pair!r} missing or non-finite: {o_by_pair.get(pair)}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, x=None)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["x"]][r["arm"]].append(r)
    xs = {}
    for x, y, pair in spec.o2_pairs:
        arms = {a[0]: sorted(by[x][a[0]], key=lambda r: int(r["seed"])) for a in spec.o2_arms}
        xs[x] = dict(y=y, pair=pair, o=float(o_by_pair[pair]), **o2_x(arms, z, o_by_pair[pair], spec))
    return dict(outcome=JUDGED, reasons=[], x=xs)


# ================================================================ sentences
def sentence(name: str, res: dict) -> str:
    o = res.get("outcome")
    if name not in ("o1", "o2"):
        raise ValueError(f"unknown stage {name!r}")
    if o == INVALID:
        return f"{name.upper()} INVALID: {'; '.join(res['reasons'])}"
    if name == "o1":
        n = len(res["mixed_cells"])
        if o == STOP_NO_SILENT_STATE:
            return (f"O1: 켬 조건의 혼합 칸 {n}개 — STOP_NO_SILENT_STATE. 실제 냄새 rig에서는 특성화할 두 상태가 없다. "
                    "사용자에게 보고한다.")
        drive_txt = ", ".join(f"{s} {v['label']}" for s, v in res["drive"].items())
        return (f"O1: 혼합 칸 {n}개 — 상태 {res['nature']['label']}, 경로 {res['pathway']['label']}, "
                f"구동 의존성 {drive_txt}. 이 커넥톰 모델(C3)의 성질이다.")
    parts = []
    for x, v in res["x"].items():
        if v["outcome"] == INVALID:
            parts.append(f"X = {x}: INVALID ({'; '.join(v['reasons'][:2])})")
            continue
        lab, da = v["labels"], v["record"]["da"]["label"]
        parts.append(f"X = {x}: {lab['depression']}, {lab['separable']}"
                     + (f" (기록: {da} — O.7.8, 판정 아님)" if da else ""))
    return "O2: " + " / ".join(parts) + ". 학습 주장이 아니다."
