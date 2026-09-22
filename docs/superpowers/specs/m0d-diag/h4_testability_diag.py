"""M0d H.4a.5 diagnostic: which clause of G.14.4's testability rule stops the H.4 oracle, from the recorded run only.

Why this exists. H.4 (spec appendix H.4 step 3; result H.4a.5) ended `STOP_LOW_T_B`: the only eligible combination, C3,
had 7 of 21 (b)-axis pairs testable (T_b 0.333 < 0.5); C0 had 4/21 (F_a 1), C1 2/21 (F_a 0). Before the user chooses
between taking C3 to H.5, a new declaration (a readout redesign, or an engine change H.8 excludes) and vocabulary
expansion / an M2 no-go, this reads the recorded per-pair oracle rows (G.14.3's `oracle_job` output, H.4's measurement
cache) and asks which bottleneck they point to:
  1. which clause of `pair_stats`'s `testable` fails (reward r, punishment -p, both; wrong signs);
  2. whether the oracle's edit saturates (the chosen alphas, depletion of the readout on X, X-alone vs X-minus-Y d');
  3. the naive readout floor on each pair (H.4a.4) and which MBON type moves under each edit (per-type counts);
  4. KC overlap of X and Y (the recorded Jaccard) and KC activity;
  5. seed noise (margins, all 4/4 splits of the report seeds, leave-one-seed-out, a seed bootstrap, turn bootstrap);
  6. C0 / C1 / C3 on the same pairs;
plus a scan of the A/P weighting of V (V_w = z_A - w z_P) as an optimistic, post hoc bound on readout re-weighting.
Everything is descriptive and post hoc on data the selection already used; nothing here is a rule or a verdict.

No engine: reads results/m0d/h4/runs/<run>.json and the oracle cache entries it lists (sha256-checked against the
report's artifact list), recomputes `h4_rules.combo_stats` / `combo_records` from the cached rows and stops unless they
equal the report. Imports only the pure-numpy formula and rules modules (h4_formula, h4_rules); thresholds come from
the report's recorded spec block. Writes one JSON (git-ignored drafts) and prints the tables.

    uv run python docs/superpowers/specs/m0d-diag/h4_testability_diag.py [out.json]
        # default .superpowers/drafts/2026-09-22-h4-testability-diag.json
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flymon.brain.h4_formula import arm_aggregate, dprime, pair_stats
from flymon.brain.h4_rules import combo_records, combo_stats

RUN = Path("results/m0d/h4/runs/20260921T174430Z-7986f4.json")
OUT = Path(".superpowers/drafts/2026-09-22-h4-testability-diag.json")
TYPES = ("MBON13", "MBON18", "MBON05", "MBON21")
PHASES = ("pre", "R1", "R2")
BOOT_N, BOOT_SEED = 2000, 20260922
TURN_BOOT_N, TURN_BOOT_SEED = 10000, 20260917            # G.14.4's turn bootstrap size and seed
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, math.inf)   # V_w = z_A - w z_P; w = 1 is H.4's V
F4_FLOOR = 5                                               # F.4's floor guard: >= 5 spikes in every seed
H5_T_B = 0.4                                               # H.5's confirmation bar (T_b >= 0.4 and F_a >= 2)


# ================================================================ small statistics
def rank(x) -> np.ndarray:
    x = np.asarray(x, float)
    r = np.empty(len(x)); r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = rank(a), rank(b)
    return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def auc(pos, neg):
    """P(a draw from pos > a draw from neg), ties 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def pct(x, qs=(2.5, 50, 97.5)):
    return [float(v) for v in np.percentile(np.asarray(x, float), qs)]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(o):
    """JSON-safe: tuples -> lists, numpy -> python, non-finite floats -> strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else str(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ================================================================ loading and the equality check
def load():
    run = json.loads(RUN.read_text())
    spec = SimpleNamespace(**run["spec"])
    spec.report_seeds = tuple(spec.report_seeds)
    combos = list(run["spec"]["combos"])
    blk = run["h4"]["combos"]
    z = {c: {k: tuple(blk[c]["z"][k]) for k in ("A", "P")} for c in combos}
    got = {c: {} for c in combos}
    for path, digest in run["artifacts"].items():
        if "/h4/cache/oracle/" not in path:
            continue
        if sha256(path) != digest:
            raise SystemExit(f"{path}: sha256 differs from the run report's artifact list")
        e = json.loads(Path(path).read_text())
        ez = {k: tuple(e["inputs"]["z"][k]) for k in ("A", "P")}
        owner = [c for c in combos if z[c] == ez]
        if len(owner) != 1 or e["inputs"]["readout"] != blk[owner[0]]["readout"]:
            raise SystemExit(f"{path}: cannot attribute the entry to one combination")
        ax, turn, x, y = e["inputs"]["pair"]
        key = (ax, int(turn), x, y)
        if key in got[owner[0]]:
            raise SystemExit(f"{path}: duplicate pair {key} in {owner[0]}")
        got[owner[0]][key] = dict(e["result"], axis=ax, turn=int(turn), x=x, y=y)
    rows = {}
    for c in combos:
        order = [(p["axis"], int(p["turn"]), p["x"], p["y"]) for p in blk[c]["oracle"]["pairs"]]
        if set(order) != set(got[c]) or len(order) != len(got[c]):
            raise SystemExit(f"{c}: cached pairs differ from the report's pairs")
        rows[c] = [got[c][k] for k in order]
    readouts = {c: blk[c]["readout"] for c in combos}
    if any(r != {"A": "MBON13", "P": "MBON05"} for r in readouts.values()):
        raise SystemExit(f"readouts are not MBON13 / MBON05 everywhere: {readouts}")
    return run, spec, combos, z, rows


def canon(o) -> str:
    return json.dumps(json.loads(json.dumps(clean(o))), sort_keys=True)


def check_against_report(run, spec, combos, z, rows) -> None:
    blk = run["h4"]["combos"]
    for c in combos:
        expected = [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows[c]]
        if canon(combo_stats(rows[c], z[c], expected, spec)) != canon(blk[c]["oracle"]):
            raise SystemExit(f"{c}: combo_stats recomputed from the cache differs from the report")
        if canon(combo_records(rows[c], z[c], spec)) != canon(blk[c]["records"]):
            raise SystemExit(f"{c}: combo_records recomputed from the cache differs from the report")


# ================================================================ per-pair features
def only(report: dict, keep: str) -> dict:
    """h4_rules._only: the other readout type's counts set to 0, so its term cancels in V(X) - V(Y)."""
    drop = "P" if keep == "A" else "A"
    return {ph: {keep: pr[keep], drop: [[0, 0] for _ in pr[drop]]} for ph, pr in report.items()}


def sub(report: dict, idx) -> dict:
    return {ph: {k: [pr[k][i] for i in idx] for k in ("A", "P")} for ph, pr in report.items()}


def weighted(report: dict, z: dict, w: float):
    if w == 0:
        return only(report, "A"), z
    if math.isinf(w):
        return only(report, "P"), z
    return report, {"A": z["A"], "P": (z["P"][0], z["P"][1] / w)}


def ratio(a, b):
    return float(a / b) if abs(b) > 1e-12 else None


def features(row: dict, z: dict, tmin: float) -> dict:
    rep = row["report"]
    st = pair_stats(rep, z, tmin)
    A = {ph: np.asarray(rep[ph]["A"], float) for ph in PHASES}
    P = {ph: np.asarray(rep[ph]["P"], float) for ph in PHASES}
    V = {ph: (A[ph] - z["A"][0]) / z["A"][1] - (P[ph] - z["P"][0]) / z["P"][1] for ph in PHASES}   # seeds x [X, Y]
    dV = {ph: V[ph][:, 0] - V[ph][:, 1] for ph in PHASES}
    rew, pun = dV["R1"] - dV["pre"], dV["R2"] - dV["R1"]
    rx, ry = V["R1"] - V["pre"], V["R2"] - V["R1"]            # per-candidate V changes under reward / punishment
    r_ok, p_ok = st["r"] >= tmin, -st["p"] >= tmin
    clause = ("testable" if st["testable"] else "reward" if (not r_ok and p_ok) else
              "punishment" if (r_ok and not p_ok) else "both")
    cnt = {ph: {t: np.asarray(row["counts"][ph][t], float) for t in TYPES} for ph in PHASES}
    floor = {t: dict(x=float(cnt["pre"][t][:, 0].mean()), y=float(cnt["pre"][t][:, 1].mean()),
                     x_zero=float((cnt["pre"][t][:, 0] == 0).mean()), y_zero=float((cnt["pre"][t][:, 1] == 0).mean()))
             for t in TYPES}
    change = {t: dict(reward_x=float((cnt["R1"][t] - cnt["pre"][t])[:, 0].mean()),
                      reward_y=float((cnt["R1"][t] - cnt["pre"][t])[:, 1].mean()),
                      punish_x=float((cnt["R2"][t] - cnt["R1"][t])[:, 0].mean()),
                      punish_y=float((cnt["R2"][t] - cnt["R1"][t])[:, 1].mean())) for t in TYPES}
    single = {k: pair_stats(only(rep, k), z, tmin) for k in ("A", "P")}
    sel = row["select"]
    kc = row["kc"]
    # F.4's floor guard on the report seeds (pre): both candidates, MBON13 >= 5 and MBON05 >= 5 in every seed
    floor_guard = bool(all((cnt["pre"][t] >= F4_FLOOR).all() for t in ("MBON13", "MBON05")))
    # select seeds: MBON05 change on X and Y at every alpha (R1(alpha) - pre), and its Y / X ratio
    pre05 = np.asarray(sel["pre"]["MBON05"], float)
    alpha_series = {}
    for a, v in sel["reward"].items():
        d05 = np.asarray(v["R1"]["MBON05"], float) - pre05
        alpha_series[a] = dict(MBON05_x=float(d05[:, 0].mean()), MBON05_y=float(d05[:, 1].mean()),
                               ratio=ratio(d05[:, 1].mean(), d05[:, 0].mean()),
                               MBON05_x_left=ratio(np.asarray(v["R1"]["MBON05"], float)[:, 0].mean(), pre05[:, 0].mean()))
    return dict(
        axis=row["axis"], turn=row["turn"], x=row["x"], y=row["y"], **st, margin=st["m"] - tmin, clause=clause,
        r_ok=bool(r_ok), p_ok=bool(p_ok), r_wrong_sign=bool(st["r"] <= 0), p_wrong_sign=bool(st["p"] >= 0),
        reward=dict(mean=float(rew.mean()), sd=float(rew.std(ddof=1)), d_x=dprime(rx[:, 0]), d_y=dprime(rx[:, 1]),
                    gen=ratio(rx[:, 1].mean(), rx[:, 0].mean())),
        punish=dict(mean=float(pun.mean()), sd=float(pun.std(ddof=1)), d_x=dprime(ry[:, 0]), d_y=dprime(ry[:, 1]),
                    gen=ratio(ry[:, 1].mean(), ry[:, 0].mean())),
        alpha_reward=float(row["alpha_reward"]), alpha_punish=float(row["alpha_punish"]),
        select_reward={a: float(v["change"]) for a, v in sel["reward"].items()},
        select_punish={a: float(v["change"]) for a, v in sel["punish"].items()},
        floor=floor, change=change, floor_guard=floor_guard, alpha_series=alpha_series,
        MBON13_R1=dict(x=float(cnt["R1"]["MBON13"][:, 0].mean()), y=float(cnt["R1"]["MBON13"][:, 1].mean())),
        depletion=dict(P_x_R1_over_pre=ratio(P["R1"][:, 0].mean(), P["pre"][:, 0].mean()),
                       A_x_R2_over_R1=ratio(A["R2"][:, 0].mean(), A["R1"][:, 0].mean()),
                       P_x_R1=float(P["R1"][:, 0].mean()), A_x_R2=float(A["R2"][:, 0].mean())),
        single={k: dict(r=s["r"], p=s["p"], m=s["m"], testable=s["testable"]) for k, s in single.items()},
        kc=dict(jaccard=float(kc["jaccard"]), x_frac=float(np.mean(kc["x"]["frac"])),
                y_frac=float(np.mean(kc["y"]["frac"])), x_spikes=float(np.mean(kc["x"]["spikes"])),
                y_spikes=float(np.mean(kc["y"]["spikes"]))))


# ================================================================ the six questions
def q1_clauses(F: list, rows: list) -> dict:
    n = len(F)
    sel_r = [max(f["select_reward"].values()) for f in F]
    return dict(
        n=n, testable=sum(f["testable"] for f in F),
        fail_reward_only=sum(f["clause"] == "reward" for f in F),
        fail_punishment_only=sum(f["clause"] == "punishment" for f in F),
        fail_both=sum(f["clause"] == "both" for f in F),
        r_ge_2=sum(f["r_ok"] for f in F), negp_ge_2=sum(f["p_ok"] for f in F), r_lt_1=sum(f["r"] < 1 for f in F),
        r_wrong_sign=sum(f["r_wrong_sign"] for f in F), p_wrong_sign=sum(f["p_wrong_sign"] for f in F),
        select_reward_change_ge_2=sum(v >= 2 for v in sel_r),
        jaccard_median=med([f["kc"]["jaccard"] for f in F]),
        r_median=med([f["r"] for f in F]), negp_median=med([-f["p"] for f in F]))


def q2_learning(F: list) -> dict:
    fail_r = [f for f in F if not f["r_ok"]]
    fail_p = [f for f in F if not f["p_ok"]]
    pass_r = [f for f in F if f["r_ok"]]
    pass_p = [f for f in F if f["p_ok"]]
    ar = Counter(f["alpha_reward"] for f in F); ap = Counter(f["alpha_punish"] for f in F)
    interior = [f for f in F if f["select_reward"]["0.8"] < max(f["select_reward"].values())]
    punish_mono = [f for f in F if f["select_punish"]["0.2"] >= f["select_punish"]["0.5"] >= f["select_punish"]["0.8"]]
    unsat = [f for f in fail_r if f["alpha_reward"] == 0.8 and (f["depletion"]["P_x_R1_over_pre"] or 0) > 0.5]
    unsat_p = [f for f in fail_p if f["alpha_punish"] == 0.8 and (f["depletion"]["A_x_R2_over_R1"] or 0) > 0.5]
    series = {grp: {a: dict(ratio=med([f["alpha_series"][a]["ratio"] for f in fs]),
                            MBON05_x_left=med([f["alpha_series"][a]["MBON05_x_left"] for f in fs]))
                    for a in ("0.2", "0.5", "0.8")} for grp, fs in (("pass", pass_r), ("fail", fail_r))}
    return dict(
        alpha_series_select_MBON05=series,
        alpha_reward=dict(sorted(ar.items())), alpha_punish=dict(sorted(ap.items())),
        alpha_reward_fail_r=dict(sorted(Counter(f["alpha_reward"] for f in fail_r).items())),
        alpha_punish_fail_p=dict(sorted(Counter(f["alpha_punish"] for f in fail_p).items())),
        reward_peak_interior=len(interior), punish_monotone=len(punish_mono),
        P_x_R1_over_pre_median=med([f["depletion"]["P_x_R1_over_pre"] for f in F]),
        P_x_R1_over_pre_lt_0p2=sum(f["depletion"]["P_x_R1_over_pre"] is not None and f["depletion"]["P_x_R1_over_pre"] < 0.2 for f in F),
        A_x_R2_over_R1_median=med([f["depletion"]["A_x_R2_over_R1"] for f in F]),
        A_x_R2_over_R1_lt_0p2=sum(f["depletion"]["A_x_R2_over_R1"] is not None and f["depletion"]["A_x_R2_over_R1"] < 0.2 for f in F),
        fail_r_unsaturated_at_0p8=len(unsat), fail_p_unsaturated_at_0p8=len(unsat_p),
        n_fail_r=len(fail_r), n_fail_p=len(fail_p),
        fail_r_with_x_alone_ge_2=sum((f["reward"]["d_x"] or 0) >= 2 for f in fail_r),
        fail_p_with_x_alone_ge_2=sum(-(f["punish"]["d_x"] or 0) >= 2 for f in fail_p),
        reward_gen_median=med([f["reward"]["gen"] for f in F]), punish_gen_median=med([f["punish"]["gen"] for f in F]),
        reward_gen_median_fail=med([f["reward"]["gen"] for f in fail_r]),
        reward_gen_median_pass=med([f["reward"]["gen"] for f in pass_r]),
        punish_gen_median_fail=med([f["punish"]["gen"] for f in fail_p]),
        punish_gen_median_pass=med([f["punish"]["gen"] for f in pass_p]),
        reward_mean_pass=med([f["reward"]["mean"] for f in pass_r]), reward_mean_fail=med([f["reward"]["mean"] for f in fail_r]),
        reward_sd_pass=med([f["reward"]["sd"] for f in pass_r]), reward_sd_fail=med([f["reward"]["sd"] for f in fail_r]),
        punish_mean_pass=med([f["punish"]["mean"] for f in pass_p]), punish_mean_fail=med([f["punish"]["mean"] for f in fail_p]),
        punish_sd_pass=med([f["punish"]["sd"] for f in pass_p]), punish_sd_fail=med([f["punish"]["sd"] for f in fail_p]),
        select_minus_report_reward=med([max(f["select_reward"].values()) - f["r"] for f in F]),
        select_minus_report_punish=med([min(f["select_punish"].values()) - f["p"] for f in F]))


def q3_floor(F: list) -> dict:
    t = [f for f in F if f["testable"]]; nt = [f for f in F if not f["testable"]]
    fl = lambda fs, ty, side: [f["floor"][ty][side] for f in fs]
    out = dict(
        MBON13_x_median_testable=med(fl(t, "MBON13", "x")), MBON13_x_median_not=med(fl(nt, "MBON13", "x")),
        MBON05_x_median_testable=med(fl(t, "MBON05", "x")), MBON05_x_median_not=med(fl(nt, "MBON05", "x")),
        MBON13_x_zero_median=med(fl(F, "MBON13", "x_zero")), MBON13_y_zero_median=med(fl(F, "MBON13", "y_zero")),
        MBON13_x_mean=float(np.mean(fl(F, "MBON13", "x"))), MBON13_y_mean=float(np.mean(fl(F, "MBON13", "y"))),
        MBON05_x_mean=float(np.mean(fl(F, "MBON05", "x"))), MBON05_y_mean=float(np.mean(fl(F, "MBON05", "y"))),
        auc_testable_MBON13_x=auc(fl(t, "MBON13", "x"), fl(nt, "MBON13", "x")),
        auc_testable_MBON05_x=auc(fl(t, "MBON05", "x"), fl(nt, "MBON05", "x")),
        auc_ppass_MBON13_x=auc([f["floor"]["MBON13"]["x"] for f in F if f["p_ok"]],
                               [f["floor"]["MBON13"]["x"] for f in F if not f["p_ok"]]),
        auc_rpass_MBON05_x=auc([f["floor"]["MBON05"]["x"] for f in F if f["r_ok"]],
                               [f["floor"]["MBON05"]["x"] for f in F if not f["r_ok"]]),
        auc_rpass_MBON13_x=auc([f["floor"]["MBON13"]["x"] for f in F if f["r_ok"]],
                               [f["floor"]["MBON13"]["x"] for f in F if not f["r_ok"]]),
        spearman_negp_MBON13_x=spearman([-f["p"] for f in F], fl(F, "MBON13", "x")),
        spearman_r_MBON05_x=spearman([f["r"] for f in F], fl(F, "MBON05", "x")),
        spearman_m_MBON13_x=spearman([f["m"] for f in F], fl(F, "MBON13", "x")),
        change={ty: {k: float(np.mean([f["change"][ty][k] for f in F])) for k in ("reward_x", "reward_y", "punish_x", "punish_y")}
                for ty in TYPES},
        single={k: dict(r_ge_2=sum(f["single"][k]["r"] >= 2 for f in F), negp_ge_2=sum(-f["single"][k]["p"] >= 2 for f in F),
                        testable=sum(f["single"][k]["testable"] for f in F), r_median=med([f["single"][k]["r"] for f in F]),
                        negp_median=med([-f["single"][k]["p"] for f in F])) for k in ("A", "P")},
        full_r_gt_A_only_r=sum(f["r"] > f["single"]["A"]["r"] for f in F),
        spearman_rA_rP=spearman([f["single"]["A"]["r"] for f in F], [f["single"]["P"]["r"] for f in F]),
        MBON13_x_R1_mean=float(np.mean([f["MBON13_R1"]["x"] for f in F])),
        auc_ppass_MBON13_x_R1=auc([f["MBON13_R1"]["x"] for f in F if f["p_ok"]],
                                  [f["MBON13_R1"]["x"] for f in F if not f["p_ok"]]),
        auc_testable_MBON13_x_R1=auc([f["MBON13_R1"]["x"] for f in t], [f["MBON13_R1"]["x"] for f in nt]),
        floor_guard={k: dict(n=len(fs), testable=sum(f["testable"] for f in fs), r_ok=sum(f["r_ok"] for f in fs),
                             p_ok=sum(f["p_ok"] for f in fs))
                     for k, fs in (("pass", [f for f in F if f["floor_guard"]]), ("fail", [f for f in F if not f["floor_guard"]]))})
    out["lifted_by_P"] = [dict(pair=f"t{f['turn']} {f['x']} / {f['y']}", r=f["r"], p=f["p"], m=f["m"],
                               A_only=f["single"]["A"], P_only=f["single"]["P"])
                          for f in F if f["testable"] and not f["single"]["A"]["testable"]]
    out["lost_by_P"] = [dict(pair=f"t{f['turn']} {f['x']} / {f['y']}", r=f["r"], p=f["p"], m=f["m"],
                             A_only=f["single"]["A"], P_only=f["single"]["P"])
                        for f in F if not f["testable"] and f["single"]["A"]["testable"]]
    return out


def q4_kc(F: list) -> dict:
    t = [f for f in F if f["testable"]]; nt = [f for f in F if not f["testable"]]
    j = lambda fs: [f["kc"]["jaccard"] for f in fs]
    return dict(
        jaccard_median_testable=med(j(t)), jaccard_median_not=med(j(nt)),
        jaccard_min=float(min(j(F))), jaccard_max=float(max(j(F))),
        auc_testable_low_jaccard=auc([-v for v in j(t)], [-v for v in j(nt)]),
        auc_rpass_low_jaccard=auc([-f["kc"]["jaccard"] for f in F if f["r_ok"]], [-f["kc"]["jaccard"] for f in F if not f["r_ok"]]),
        spearman_r_jaccard=spearman([f["r"] for f in F], j(F)),
        spearman_negp_jaccard=spearman([-f["p"] for f in F], j(F)),
        spearman_reward_gen_jaccard=spearman([f["reward"]["gen"] if f["reward"]["gen"] is not None else np.nan for f in F], j(F)),
        spearman_punish_gen_jaccard=spearman([f["punish"]["gen"] if f["punish"]["gen"] is not None else np.nan for f in F], j(F)),
        x_frac_median=med([f["kc"]["x_frac"] for f in F]), y_frac_median=med([f["kc"]["y_frac"] for f in F]),
        x_frac_median_testable=med([f["kc"]["x_frac"] for f in t]), x_frac_median_not=med([f["kc"]["x_frac"] for f in nt]),
        abs_dpre_median_testable=med([abs(f["d_pre"]) for f in t]), abs_dpre_median_not=med([abs(f["d_pre"]) for f in nt]),
        auc_testable_abs_dpre=auc([abs(f["d_pre"]) for f in t], [abs(f["d_pre"]) for f in nt]))


def count_b(rows_b: list, z: dict, tmin: float, idx) -> tuple:
    flags = [bool((s := pair_stats(sub(r["report"], idx), z, tmin)) and s["testable"]) for r in rows_b]
    return sum(flags), flags


def count_fa(rows_a: list, z: dict, tmin: float, naive_max: float, idx) -> int:
    return sum(bool((s := pair_stats(sub(r["report"], idx), z, tmin)) and s["testable"] and abs(s["d_pre"]) < naive_max)
               for r in rows_a)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) >= k)."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def q5_noise(combos, z, rows, F, tmin, naive_max) -> dict:
    n_seeds = len(rows[combos[0]][0]["report"]["pre"]["A"])
    halves = list(itertools.combinations(range(n_seeds), n_seeds // 2))
    rows_b = {c: [r for r in rows[c] if r["axis"] == "b"] for c in combos}
    Fb = {c: [f for f in F[c] if f["axis"] == "b"] for c in combos}
    full = {c: [f["testable"] for f in Fb[c]] for c in combos}
    out = {}
    half_counts = {c: [] for c in combos}
    half_flags = {c: [] for c in combos}
    for h in halves:
        for c in combos:
            k, fl = count_b(rows_b[c], z[c], tmin, h)
            half_counts[c].append(k); half_flags[c].append(fl)
    comp = [(i, halves.index(tuple(sorted(set(range(n_seeds)) - set(h))))) for i, h in enumerate(halves) if i < len(halves) // 2]
    for c in combos:
        F_ = Fb[c]
        nt = [f for f in F_ if not f["testable"]]; t = [f for f in F_ if f["testable"]]
        rec = [half_flags[c][0], half_flags[c][halves.index(tuple(range(n_seeds // 2, n_seeds)))]]
        share = np.mean(np.asarray(half_flags[c], float), axis=0)
        loo = [count_b(rows_b[c], z[c], tmin, [i for i in range(n_seeds) if i != j]) for j in range(n_seeds)]
        loo_flip = sum(any(fl[i] != full[c][i] for _, fl in loo) for i in range(len(F_)))
        out[c] = dict(
            margin_not=dict(ge_m0p5=sum(-0.5 <= f["margin"] < 0 for f in nt), m1_to_m0p5=sum(-1 <= f["margin"] < -0.5 for f in nt),
                            m2_to_m1=sum(-2 <= f["margin"] < -1 for f in nt), lt_m2=sum(f["margin"] < -2 for f in nt),
                            median=med([f["margin"] for f in nt])),
            margin_testable=dict(lt_0p5=sum(f["margin"] < 0.5 for f in t), lt_1=sum(f["margin"] < 1 for f in t),
                                 values=sorted(float(f["margin"]) for f in t)),
            recorded_halves=[sum(h) for h in rec],
            recorded_halves_flips=[dict(lost=sum(full[c][i] and not h[i] for i in range(len(F_))),
                                        gained=sum((not full[c][i]) and h[i] for i in range(len(F_)))) for h in rec],
            all_halves=dict(n=len(halves), mean=float(np.mean(half_counts[c])), sd=float(np.std(half_counts[c])),
                            min=int(min(half_counts[c])), max=int(max(half_counts[c])),
                            abs_diff_complementary=dict(mean=float(np.mean([abs(half_counts[c][a] - half_counts[c][b]) for a, b in comp])),
                                                        max=int(max(abs(half_counts[c][a] - half_counts[c][b]) for a, b in comp)))),
            pairs_testable_in_ge_half_of_halves=int((share >= 0.5).sum()),
            pairs_testable_in_any_half=int((share > 0).sum()),
            loo_counts=[k for k, _ in loo], loo_pairs_flipping=int(loo_flip))
    # C3 - C0 (and C3 - C1) on the same 4/4 halves
    for a, b in (("C3", "C0"), ("C3", "C1"), ("C0", "C1")):
        d = [x - y for x, y in zip(half_counts[a], half_counts[b])]
        out[f"halves_{a}_minus_{b}"] = dict(mean=float(np.mean(d)), min=int(min(d)), max=int(max(d)),
                                             share_le_0=float(np.mean([v <= 0 for v in d])))
    # seed bootstrap (same resampled seed indices for every combination)
    rng = np.random.default_rng(BOOT_SEED)
    boot = {c: [] for c in combos}
    for _ in range(BOOT_N):
        idx = rng.integers(0, n_seeds, n_seeds)
        for c in combos:
            boot[c].append(count_b(rows_b[c], z[c], tmin, idx)[0])
    out["seed_bootstrap"] = dict(n=BOOT_N, seed=BOOT_SEED, **{c: pct(boot[c]) for c in combos},
                                 C3_minus_C0=pct(np.subtract(boot["C3"], boot["C0"])),
                                 C3_minus_C0_share_le_0=float(np.mean(np.subtract(boot["C3"], boot["C0"]) <= 0)),
                                 C3_share_ge_11=float(np.mean(np.asarray(boot["C3"]) >= 11)))
    # turn bootstrap (G.14.4's resampling unit) on the recorded testable flags
    turns = sorted({f["turn"] for f in Fb["C3"]})
    rng = np.random.default_rng(TURN_BOOT_SEED)
    diffs, c3 = [], []
    for _ in range(TURN_BOOT_N):
        pick = rng.choice(turns, len(turns), replace=True)
        sel = lambda c: [fl for tt in pick for f, fl in zip(Fb[c], full[c]) if f["turn"] == tt]
        s3, s0 = sel("C3"), sel("C0")
        diffs.append(np.mean(s3) - np.mean(s0)); c3.append(np.mean(s3))
    out["turn_bootstrap"] = dict(n=TURN_BOOT_N, seed=TURN_BOOT_SEED, T_b_C3=pct(c3), T_b_C3_minus_C0=pct(diffs),
                                 share_le_0=float(np.mean(np.asarray(diffs) <= 0)),
                                 T_b_C3_share_ge_h5=float(np.mean(np.asarray(c3) >= H5_T_B - 1e-12)),
                                 T_b_C3_share_ge_0p5=float(np.mean(np.asarray(c3) >= 0.5 - 1e-12)))
    n_b = len(Fb["C3"]); k3 = sum(full["C3"])
    out["h5_binomial"] = dict(n=n_b, p=k3 / n_b, need=math.ceil(H5_T_B * n_b - 1e-9),
                              P_T_b_ge_h5=binom_sf(math.ceil(H5_T_B * n_b - 1e-9), n_b, k3 / n_b),
                              P_T_b_ge_0p5=binom_sf(math.ceil(0.5 * n_b - 1e-9), n_b, k3 / n_b))
    # F_a's seed fragility (the other half of the bar), on the same halves / leave-one-out / bootstrap draws
    rows_a = {c: [r for r in rows[c] if r["axis"] == "a"] for c in combos}
    rng = np.random.default_rng(BOOT_SEED)
    fa_boot = {c: [] for c in combos}
    for _ in range(BOOT_N):
        idx = rng.integers(0, n_seeds, n_seeds)
        for c in combos:
            fa_boot[c].append(count_fa(rows_a[c], z[c], tmin, naive_max, idx))
    out["F_a_fragility"] = {c: dict(
        full=count_fa(rows_a[c], z[c], tmin, naive_max, range(n_seeds)),
        halves=dict(Counter(count_fa(rows_a[c], z[c], tmin, naive_max, h) for h in halves)),
        loo=[count_fa(rows_a[c], z[c], tmin, naive_max, [i for i in range(n_seeds) if i != j]) for j in range(n_seeds)],
        bootstrap_share_ge_2=float(np.mean(np.asarray(fa_boot[c]) >= 2))) for c in combos}
    b01 = sum(x and not y for x, y in zip(full["C3"], full["C0"])); b10 = sum(y and not x for x, y in zip(full["C3"], full["C0"]))
    out["mcnemar_C3_vs_C0"] = dict(C3_only=b01, C0_only=b10, p_exact=mcnemar_exact(b01, b10))
    return out


def q6_engine(combos, F) -> dict:
    Fb = {c: [f for f in F[c] if f["axis"] == "b"] for c in combos}
    table = []
    for i, f0 in enumerate(Fb["C0"]):
        table.append(dict(pair=f"t{f0['turn']} {f0['x']} / {f0['y']}",
                          **{c: dict(testable=Fb[c][i]["testable"], clause=Fb[c][i]["clause"], r=Fb[c][i]["r"],
                                     p=Fb[c][i]["p"], m=Fb[c][i]["m"],
                                     MBON13_x=Fb[c][i]["floor"]["MBON13"]["x"], MBON13_y=Fb[c][i]["floor"]["MBON13"]["y"],
                                     MBON05_x=Fb[c][i]["floor"]["MBON05"]["x"], jaccard=Fb[c][i]["kc"]["jaccard"],
                                     x_frac=Fb[c][i]["kc"]["x_frac"], A_x_R2_over_R1=Fb[c][i]["depletion"]["A_x_R2_over_R1"])
                             for c in combos}))
    d = lambda a, b, g: [g(Fb[a][i]) - g(Fb[b][i]) for i in range(len(Fb[a]))]
    out = dict(table=table)
    for a, b in (("C3", "C0"), ("C3", "C1"), ("C1", "C0")):
        dfl = d(a, b, lambda f: f["floor"]["MBON13"]["x"])
        dnegp = d(a, b, lambda f: -np.clip(f["p"], -10, 10))
        dr = d(a, b, lambda f: np.clip(f["r"], -10, 10))
        dm = d(a, b, lambda f: np.clip(f["m"], -10, 10))
        out[f"{a}_minus_{b}"] = dict(
            median_r=med(dr), median_negp=med(dnegp), median_m=med(dm), median_MBON13_x=med(dfl),
            median_MBON05_x=med(d(a, b, lambda f: f["floor"]["MBON05"]["x"])),
            median_jaccard=med(d(a, b, lambda f: f["kc"]["jaccard"])),
            median_x_frac=med(d(a, b, lambda f: f["kc"]["x_frac"])),
            spearman_dMBON13x_dnegp=spearman(dfl, dnegp), spearman_dMBON13x_dm=spearman(dfl, dm),
            spearman_dMBON13x_dr=spearman(dfl, dr),
            gained=[t["pair"] + f" ({t[b]['clause']} in {b})" for t in table if t[a]["testable"] and not t[b]["testable"]],
            lost=[t["pair"] + f" ({t[a]['clause']} in {a})" for t in table if t[b]["testable"] and not t[a]["testable"]])
    # pooled (b) rows: punishment-clause pass rate by MBON13-on-X floor band, per combination
    allfl = np.array([f["floor"]["MBON13"]["x"] for c in combos for f in Fb[c]])
    edges = np.percentile(allfl, [100 / 3, 200 / 3])
    band = lambda v: int(np.searchsorted(edges, v, side="right"))
    out["floor_bands"] = dict(edges=[float(e) for e in edges], **{
        c: [dict(n=sum(band(f["floor"]["MBON13"]["x"]) == k for f in Fb[c]),
                 p_ok=sum(f["p_ok"] for f in Fb[c] if band(f["floor"]["MBON13"]["x"]) == k),
                 r_ok=sum(f["r_ok"] for f in Fb[c] if band(f["floor"]["MBON13"]["x"]) == k),
                 testable=sum(f["testable"] for f in Fb[c] if band(f["floor"]["MBON13"]["x"]) == k)) for k in range(3)]
        for c in combos})
    out["pooled_auc_testable_MBON13_x"] = auc([f["floor"]["MBON13"]["x"] for c in combos for f in Fb[c] if f["testable"]],
                                              [f["floor"]["MBON13"]["x"] for c in combos for f in Fb[c] if not f["testable"]])
    out["pooled_auc_ppass_MBON13_x"] = auc([f["floor"]["MBON13"]["x"] for c in combos for f in Fb[c] if f["p_ok"]],
                                           [f["floor"]["MBON13"]["x"] for c in combos for f in Fb[c] if not f["p_ok"]])
    out["punish_depletion_median"] = {c: med([f["depletion"]["A_x_R2_over_R1"] for f in Fb[c]]) for c in combos}
    return out


def weight_scan(combos, z, rows, spec) -> dict:
    out = {}
    for c in combos:
        out[c] = {}
        for w in WEIGHTS:
            st = {}
            for r in rows[c]:
                rep, zz = weighted(r["report"], z[c], w)
                st[(r["axis"], r["turn"], r["x"], r["y"])] = pair_stats(rep, zz, spec.testable_min)
            agg = arm_aggregate(st, spec.naive_max, spec.t_b_min, spec.f_a_min)
            b = [s for k, s in st.items() if k[0] == "b"]
            out[c][str(w)] = dict(testable_b=agg["testable_b"], testable_a=agg["testable_a"], F_a=agg["F_a"],
                                  r_ge_2_b=sum(s["r"] >= 2 for s in b), negp_ge_2_b=sum(-s["p"] >= 2 for s in b))
    return out


# ================================================================ printing
def show(title: str, header: list, body: list) -> None:
    print(f"\n## {title}")
    w = [max(len(str(h)), *(len(str(r[i])) for r in body)) for i, h in enumerate(header)]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    for r in body:
        print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


def f2(v):
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    run, spec, combos, z, rows = load()
    check_against_report(run, spec, combos, z, rows)
    tmin = spec.testable_min
    F = {c: [features(r, z[c], tmin) for r in rows[c]] for c in combos}
    ax = lambda c, a: [f for f in F[c] if f["axis"] == a]
    res = dict(run=run["run_id"], commit=run["git"]["commit"], report=str(RUN), check="recomputed combo_stats and "
               "combo_records equal the report for every combination", z=z,
               q1={c: {a: q1_clauses(ax(c, a), rows[c]) for a in ("a", "b")} for c in combos},
               q2={c: {a: q2_learning(ax(c, a)) for a in ("a", "b")} for c in combos},
               q3={c: {a: q3_floor(ax(c, a)) for a in ("a", "b")} for c in combos},
               q4={c: {a: q4_kc(ax(c, a)) for a in ("a", "b")} for c in combos},
               q5=q5_noise(combos, z, rows, F, tmin, spec.naive_max), q6=q6_engine(combos, F),
               weight_scan=weight_scan(combos, z, rows, spec), pairs=F)
    print(f"run {res['run']} (commit {res['commit'][:7]}): {res['check']}")

    show("Q1 which clause fails (per combination and axis)",
         ["combo", "axis", "n", "testable", "fail r only", "fail -p only", "fail both", "r>=2", "-p>=2", "r<1",
          "r<=0", "p>=0", "sel. reward chg>=2", "med r", "med -p", "med Jaccard"],
         [[c, a, q["n"], q["testable"], q["fail_reward_only"], q["fail_punishment_only"], q["fail_both"], q["r_ge_2"],
           q["negp_ge_2"], q["r_lt_1"], q["r_wrong_sign"], q["p_wrong_sign"], q["select_reward_change_ge_2"],
           f2(q["r_median"]), f2(q["negp_median"]), f2(q["jaccard_median"])]
          for c in combos for a in ("b", "a") for q in [res["q1"][c][a]]])
    show("Q2 alpha choice and saturation",
         ["combo", "axis", "alpha_r counts", "alpha_p counts", "alpha_r | r fails", "reward peak < 0.8", "punish monotone",
          "MBON05(X) R1/pre med", "<0.2", "MBON13(X) R2/R1 med", "<0.2", "r-fail unsat@0.8", "p-fail unsat@0.8"],
         [[c, a, q["alpha_reward"], q["alpha_punish"], q["alpha_reward_fail_r"], q["reward_peak_interior"],
           q["punish_monotone"], f2(q["P_x_R1_over_pre_median"]), q["P_x_R1_over_pre_lt_0p2"],
           f2(q["A_x_R2_over_R1_median"]), q["A_x_R2_over_R1_lt_0p2"], f"{q['fail_r_unsaturated_at_0p8']}/{q['n_fail_r']}",
           f"{q['fail_p_unsaturated_at_0p8']}/{q['n_fail_p']}"] for c in combos for a in ("b", "a") for q in [res["q2"][c][a]]])
    show("Q2 effect vs noise, X alone vs X - Y (medians)",
         ["combo", "axis", "reward mean pass/fail", "reward sd pass/fail", "punish mean pass/fail", "punish sd pass/fail",
          "r-fails with d'(dV_X)>=2", "p-fails with -d'(dV_X)>=2", "reward Y/X gen pass/fail", "punish Y/X gen pass/fail",
          "select - report r", "select - report p"],
         [[c, a, f"{f2(q['reward_mean_pass'])}/{f2(q['reward_mean_fail'])}", f"{f2(q['reward_sd_pass'])}/{f2(q['reward_sd_fail'])}",
           f"{f2(q['punish_mean_pass'])}/{f2(q['punish_mean_fail'])}", f"{f2(q['punish_sd_pass'])}/{f2(q['punish_sd_fail'])}",
           f"{q['fail_r_with_x_alone_ge_2']}/{q['n_fail_r']}", f"{q['fail_p_with_x_alone_ge_2']}/{q['n_fail_p']}",
           f"{f2(q['reward_gen_median_pass'])}/{f2(q['reward_gen_median_fail'])}",
           f"{f2(q['punish_gen_median_pass'])}/{f2(q['punish_gen_median_fail'])}",
           f2(q["select_minus_report_reward"]), f2(q["select_minus_report_punish"])]
          for c in combos for a in ("b", "a") for q in [res["q2"][c][a]]])
    show("Q2 select seeds: MBON05 Y/X change ratio and MBON05 left on X (R1/pre) by alpha, medians, reward pass | fail",
         ["combo", "axis", *(f"a={a} ratio pass|fail" for a in ("0.2", "0.5", "0.8")),
          *(f"a={a} X left pass|fail" for a in ("0.2", "0.5", "0.8"))],
         [[c, a, *(f"{f2(s['pass'][al]['ratio'])}|{f2(s['fail'][al]['ratio'])}" for al in ("0.2", "0.5", "0.8")),
           *(f"{f2(s['pass'][al]['MBON05_x_left'])}|{f2(s['fail'][al]['MBON05_x_left'])}" for al in ("0.2", "0.5", "0.8"))]
          for c in combos for a in ("b", "a") for s in [res["q2"][c][a]["alpha_series_select_MBON05"]]])
    show("Q3 naive floor (report seeds, pre) and its relation to testability",
         ["combo", "axis", "MBON13 X/Y mean", "MBON05 X/Y mean", "MBON13 X zero med", "MBON13 X med test/not",
          "AUC test~MBON13X", "AUC -p pass~MBON13X", "AUC r pass~MBON05X", "AUC r pass~MBON13X", "rho(-p,MBON13X)",
          "rho(r,MBON05X)"],
         [[c, a, f"{q['MBON13_x_mean']:.1f}/{q['MBON13_y_mean']:.1f}", f"{q['MBON05_x_mean']:.1f}/{q['MBON05_y_mean']:.1f}",
           f2(q["MBON13_x_zero_median"]), f"{f2(q['MBON13_x_median_testable'])}/{f2(q['MBON13_x_median_not'])}",
           f2(q["auc_testable_MBON13_x"]), f2(q["auc_ppass_MBON13_x"]), f2(q["auc_rpass_MBON05_x"]),
           f2(q["auc_rpass_MBON13_x"]), f2(q["spearman_negp_MBON13_x"]), f2(q["spearman_r_MBON05_x"])]
          for c in combos for a in ("b", "a") for q in [res["q3"][c][a]]])
    show("Q3 F.4 floor guard (both candidates, MBON13 and MBON05 >= 5 in every report seed) and MBON13 on X after reward",
         ["combo", "axis", "guard pass: n / testable / r ok / -p ok", "guard fail: n / testable / r ok / -p ok",
          "MBON13 X pre / R1 mean", "AUC -p pass~MBON13X R1", "AUC test~MBON13X R1"],
         [[c, a, "{n} / {testable} / {r_ok} / {p_ok}".format(**q["floor_guard"]["pass"]),
           "{n} / {testable} / {r_ok} / {p_ok}".format(**q["floor_guard"]["fail"]),
           f"{q['MBON13_x_mean']:.1f} / {q['MBON13_x_R1_mean']:.1f}", f2(q["auc_ppass_MBON13_x_R1"]),
           f2(q["auc_testable_MBON13_x_R1"])] for c in combos for a in ("b", "a") for q in [res["q3"][c][a]]])
    show("Q3 per-type mean count change (reward = R1 - pre, punishment = R2 - R1), (b) pairs",
         ["combo", "type", "reward X", "reward Y", "punish X", "punish Y"],
         [[c, t, *(f"{res['q3'][c]['b']['change'][t][k]:+.1f}" for k in ("reward_x", "reward_y", "punish_x", "punish_y"))]
          for c in combos for t in TYPES])
    show("Q3 single-type statistics (other term constant)",
         ["combo", "axis", "A-only r>=2", "A-only -p>=2", "A-only testable", "P-only r>=2", "P-only -p>=2", "P-only testable",
          "med P-only r", "med P-only -p", "full r > A-only r", "rho(rA, rP)"],
         [[c, a, q["single"]["A"]["r_ge_2"], q["single"]["A"]["negp_ge_2"], q["single"]["A"]["testable"],
           q["single"]["P"]["r_ge_2"], q["single"]["P"]["negp_ge_2"], q["single"]["P"]["testable"],
           f2(q["single"]["P"]["r_median"]), f2(q["single"]["P"]["negp_median"]), q["full_r_gt_A_only_r"],
           f2(q["spearman_rA_rP"])] for c in combos for a in ("b", "a") for q in [res["q3"][c][a]]])
    for c in combos:
        for item in res["q3"][c]["b"]["lifted_by_P"]:
            print(f"  {c} (b) lifted by P: {item['pair']}: full r {item['r']:.2f} p {item['p']:.2f}; A-only r "
                  f"{item['A_only']['r']:.2f} p {item['A_only']['p']:.2f}; P-only r {item['P_only']['r']:.2f} p {item['P_only']['p']:.2f}")
        for item in res["q3"][c]["b"]["lost_by_P"]:
            print(f"  {c} (b) lost by P: {item['pair']}: full r {item['r']:.2f} p {item['p']:.2f}; A-only r "
                  f"{item['A_only']['r']:.2f} p {item['A_only']['p']:.2f}; P-only r {item['P_only']['r']:.2f} p {item['P_only']['p']:.2f}")
    show("Q4 KC overlap and activity",
         ["combo", "axis", "Jaccard med test/not", "Jaccard range", "AUC test~low J", "AUC r pass~low J", "rho(r,J)",
          "rho(-p,J)", "rho(reward gen,J)", "KC frac X/Y med", "KC frac X test/not", "|d_pre| med test/not", "AUC test~|d_pre|"],
         [[c, a, f"{f2(q['jaccard_median_testable'])}/{f2(q['jaccard_median_not'])}", f"{q['jaccard_min']:.2f}-{q['jaccard_max']:.2f}",
           f2(q["auc_testable_low_jaccard"]), f2(q["auc_rpass_low_jaccard"]), f2(q["spearman_r_jaccard"]),
           f2(q["spearman_negp_jaccard"]), f2(q["spearman_reward_gen_jaccard"]),
           f"{q['x_frac_median']:.3f}/{q['y_frac_median']:.3f}",
           f"{f2(q['x_frac_median_testable'])}/{f2(q['x_frac_median_not'])}",
           f"{f2(q['abs_dpre_median_testable'])}/{f2(q['abs_dpre_median_not'])}", f2(q["auc_testable_abs_dpre"])]
          for c in combos for a in ("b", "a") for q in [res["q4"][c][a]]])
    q5 = res["q5"]
    show("Q5 seed noise, (b) axis",
         ["combo", "non-testable margin [-0.5,0) / [-1,-0.5) / [-2,-1) / <-2", "testable margins", "recorded halves",
          "half flips lost/gained", "all 70 halves mean (min-max)", "|half-half| mean (max)", "pairs testable in >=50% halves",
          "LOO counts", "LOO flipping pairs"],
         [[c, "{ge_m0p5} / {m1_to_m0p5} / {m2_to_m1} / {lt_m2}".format(**q["margin_not"]),
           " ".join(f"{v:.2f}" for v in q["margin_testable"]["values"]), q["recorded_halves"],
           " ; ".join(f"{h['lost']}/{h['gained']}" for h in q["recorded_halves_flips"]),
           f"{q['all_halves']['mean']:.1f} ({q['all_halves']['min']}-{q['all_halves']['max']})",
           f"{q['all_halves']['abs_diff_complementary']['mean']:.1f} ({q['all_halves']['abs_diff_complementary']['max']})",
           q["pairs_testable_in_ge_half_of_halves"], q["loo_counts"], q["loo_pairs_flipping"]] for c in combos for q in [q5[c]]])
    print(f"  halves C3-C0: {q5['halves_C3_minus_C0']}; C3-C1: {q5['halves_C3_minus_C1']}; C0-C1: {q5['halves_C0_minus_C1']}")
    print(f"  seed bootstrap (n {BOOT_N}) 2.5/50/97.5%: " + ", ".join(f"{c} {q5['seed_bootstrap'][c]}" for c in combos)
          + f"; C3-C0 {q5['seed_bootstrap']['C3_minus_C0']}, P(C3-C0<=0) {q5['seed_bootstrap']['C3_minus_C0_share_le_0']:.3f}, "
            f"P(C3>=11) {q5['seed_bootstrap']['C3_share_ge_11']:.3f}")
    print(f"  turn bootstrap (n {TURN_BOOT_N}): T_b C3 {[round(v, 3) for v in q5['turn_bootstrap']['T_b_C3']]}, C3-C0 "
          f"{[round(v, 3) for v in q5['turn_bootstrap']['T_b_C3_minus_C0']]}, P(<=0) {q5['turn_bootstrap']['share_le_0']:.3f}")
    print(f"  McNemar C3 vs C0 (b): {q5['mcnemar_C3_vs_C0']}")
    tb, hb = q5["turn_bootstrap"], q5["h5_binomial"]
    print(f"  H.5 odds for C3 (T_b >= {H5_T_B}): turn bootstrap share {tb['T_b_C3_share_ge_h5']:.3f} (>= 0.5: "
          f"{tb['T_b_C3_share_ge_0p5']:.3f}); binomial n {hb['n']} p {hb['p']:.3f}: P(>= {hb['need']}) "
          f"{hb['P_T_b_ge_h5']:.3f}, P(T_b >= 0.5) {hb['P_T_b_ge_0p5']:.3f}")
    for c in combos:
        fa = q5["F_a_fragility"][c]
        print(f"  F_a {c}: full {fa['full']}, 70 halves {dict(sorted(fa['halves'].items()))}, LOO {fa['loo']}, "
              f"bootstrap P(F_a >= 2) {fa['bootstrap_share_ge_2']:.3f}")
    q6 = res["q6"]
    show("Q6 (b) pairs across combinations (T = testable; clause that fails otherwise; MBON13 on X, pre mean)",
         ["pair", *(f"{c} clause" for c in combos), *(f"{c} m" for c in combos), *(f"{c} MBON13 X" for c in combos),
          *(f"{c} J" for c in combos)],
         [[t["pair"], *("T" if t[c]["testable"] else t[c]["clause"] for c in combos), *(f2(float(t[c]["m"])) for c in combos),
           *(f"{t[c]['MBON13_x']:.1f}" for c in combos), *(f"{t[c]['jaccard']:.2f}" for c in combos)] for t in q6["table"]])
    for k in ("C3_minus_C0", "C3_minus_C1", "C1_minus_C0"):
        v = q6[k]
        print(f"  {k}: median d r {f2(v['median_r'])}, d -p {f2(v['median_negp'])}, d m {f2(v['median_m'])}, "
              f"d MBON13X {f2(v['median_MBON13_x'])}, d MBON05X {f2(v['median_MBON05_x'])}, d J {f2(v['median_jaccard'])}, "
              f"d KCfracX {v['median_x_frac']:+.4f}; rho(dMBON13X, d -p) {f2(v['spearman_dMBON13x_dnegp'])}, "
              f"rho(dMBON13X, d m) {f2(v['spearman_dMBON13x_dm'])}, rho(dMBON13X, d r) {f2(v['spearman_dMBON13x_dr'])}")
        print(f"    gained: {v['gained']}\n    lost: {v['lost']}")
    fb = q6["floor_bands"]
    print(f"  pooled (b) MBON13-on-X floor bands, edges {[round(e, 1) for e in fb['edges']]} (low / mid / high):")
    for c in combos:
        print(f"    {c}: " + " | ".join(f"n {b['n']} -p ok {b['p_ok']} r ok {b['r_ok']} T {b['testable']}" for b in fb[c]))
    print(f"  pooled AUC testable~MBON13X {f2(q6['pooled_auc_testable_MBON13_x'])}, -p pass~MBON13X "
          f"{f2(q6['pooled_auc_ppass_MBON13_x'])}; MBON13(X) R2/R1 median {q6['punish_depletion_median']}")
    ws = res["weight_scan"]
    show("Readout weighting scan V_w = z_A - w z_P (post hoc upper bound; w = 1 is H.4's V)",
         ["combo", *(f"w={w:g}" for w in WEIGHTS)],
         [[c, *(f"{ws[c][str(w)]['testable_b']} (r{ws[c][str(w)]['r_ge_2_b']}/p{ws[c][str(w)]['negp_ge_2_b']}) Fa{ws[c][str(w)]['F_a']}"
                for w in WEIGHTS)] for c in combos])
    print("  cell = testable (b) (r>=2 count / -p>=2 count on (b)) and F_a")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean(res), indent=1) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
