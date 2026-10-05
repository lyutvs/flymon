"""W's records and plumbing between raw job rows and the verdict code (W.3, W.6, W.9.2, W.9.4, W.9.6, W.9.8 H6,
W.9.9 P1-4 / P2-9): no engine, no writes.
- counts / pair_data: a w_learn_job row's probes -> [K, 2 cells, 2 odours]; one pair's R / N / RN rows per fly ->
  {stage: int [F, K, 2, 2]} (pre and R1 / R2 from R, N1 / N2 from N, RN1 / RN2 from RN).
- machine_reasons: pre equal across a fly's brains and equal to the naive screen's probes, RN's weights after the
  reward phase equal R's (the state behind RN1 = R1), the declared edit / CSC / edge counts / probe seeds, the band
  jobs' weights equal the main jobs' at every stage.
- p_repro_diffs / naive_repro_diffs / reward_check: the path gate's comparisons (W.3 2 (i), (ii), W.9.9 P2-9).
- pilot_record: W.6 / W.9.4 / W.9.8 H6's pilot statistics (exploratory).
- unit_costs / design_cost / worst_cost: the wall-clock model behind the OC's ranking and the budget gate."""
from __future__ import annotations

import math
from statistics import median

import numpy as np

from . import w_verdict as WV

STAGE_OF = {("R", "pre"): "pre", ("R", "S1"): "R1", ("R", "S2"): "R2", ("N", "S1"): "N1", ("N", "S2"): "N2",
            ("RN", "S1"): "RN1", ("RN", "S2"): "RN2"}
ROW_SKIP = ("wall_s",)


def counts(stage: dict) -> np.ndarray:
    """[K, 2, 2] int: [k, A/P, X/Y] from a job stage's x / y presentation rows."""
    return np.array([[[x["A"], y["A"]], [x["P"], y["P"]]] for x, y in zip(stage["x"], stage["y"])], dtype=np.int64)


def _stages(res: dict) -> dict:
    return {s["stage"]: s for s in res["stages"]}


def pair_data(rows: list, flies: list) -> dict:
    """rows = [dict(unit, result)] of one pair (any order); flies = the scheduled fly numbers in order."""
    by = {(r["unit"]["fly"], r["unit"]["brain"]): r["result"] for r in rows}
    out = {}
    for (brain, st), name in STAGE_OF.items():
        out[name] = np.stack([counts(_stages(by[(f, brain)])[st]) for f in flies])
    return out


def machine_reasons(rows: list, flies: list, declared: dict, naive: dict | None = None,
                    band: list | None = None) -> list:
    """declared = {edit, csc_sha256, edit_edges, block_edges, probe_seeds: {fly: [...]}}; naive = {fly: [K, 2, 2]}
    of the screen (None: no screen); band = the pair's band rows (None: none)."""
    by = {(r["unit"]["fly"], r["unit"]["brain"]): r["result"] for r in rows}
    out = []
    for f in flies:
        got = {b: by.get((f, b)) for b in ("R", "N", "RN")}
        miss = [b for b, v in got.items() if v is None]
        if miss:
            out.append(f"fly {f}: {miss} 없음")
            continue
        st = {b: _stages(v) for b, v in got.items()}
        pre = counts(st["R"]["pre"])
        for b in ("N", "RN"):
            if not np.array_equal(counts(st[b]["pre"]), pre):
                out.append(f"fly {f}: {b} pre ≠ R pre")
        if st["RN"]["S1"]["w_sha256"] != st["R"]["S1"]["w_sha256"]:
            out.append(f"fly {f}: RN 보상 뒤 가중치 ≠ R")
        if naive is not None and not np.array_equal(np.asarray(naive[f]), pre[: len(naive[f])]):
            out.append(f"fly {f}: 학습 pre ≠ 순진 거름 프로브")
        for b, v in got.items():
            for k in ("edit", "csc_sha256", "edit_edges", "block_edges"):
                if v.get(k) != declared[k]:
                    out.append(f"fly {f} {b}: {k} {v.get(k)!r} ≠ {declared[k]!r}")
            if v.get("probe_seeds") != declared["probe_seeds"][f]:
                out.append(f"fly {f} {b}: 프로브 시드가 선언과 다름")
    for r in band or []:
        u, res = r["unit"], r["result"]
        main = by.get((u["fly"], u["brain"]))
        if main is None:
            out.append(f"band fly {u['fly']} {u['brain']}: 주 측정 없음")
            continue
        a, b = _stages(main), _stages(res)
        if [a[s]["w_sha256"] for s in ("pre", "S1", "S2")] != [b[s]["w_sha256"] for s in ("pre", "S1", "S2")]:
            out.append(f"band fly {u['fly']} {u['brain']}: 재훈련 가중치 ≠ 주 측정")
    return out


def extend(d_k: dict, band_rows: list, flies: list) -> dict:
    """The 2K data: the K probes of d_k followed by the band jobs' probes K..2K−1."""
    ext = pair_data(band_rows, flies)
    return {s: np.concatenate([np.asarray(d_k[s]), ext[s]], axis=1) for s in WV.STAGES}


# ================================================================ the path gate (W.3 2, W.9.9 P2-9)
def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ROW_SKIP}


def p_repro_diffs(w_res: dict, v_row: dict, tag: str) -> list:
    """W's job with P's punish-arm settings vs V's gate-② arm row: pre / post presentations (wall aside), the weight
    shas, the weight fractions, the dopamine integral and the CSC."""
    st = _stages(w_res)
    out = []
    for name, w_stage, v_key in (("pre", st["pre"], "pre"), ("post", st["S1"], "post")):
        for s in ("x", "y"):
            if _strip(w_stage[s][0]) != _strip(v_row[v_key][s]):
                out.append(f"{tag} {name} {s}")
    pairs = (("w0_sha256", w_res["w0_sha256"]), ("w_post_sha256", st["S1"]["w_sha256"]),
             ("weights_frac", st["S1"]["weights_frac"]), ("weights_frac_A", st["S1"]["weights_frac_A"]),
             ("weights_frac_P", st["S1"]["weights_frac_P"]), ("da_integral", st["S1"]["da_integral"]),
             ("csc_sha256", w_res["csc_sha256"]))
    out += [f"{tag} {k}" for k, w in pairs if w != v_row[k]]
    return out


def naive_repro_diffs(w_res: dict, v_result: dict, tag: str) -> list:
    """W's naive probes on V's report seeds vs V's oracle raw report.pre (MBON13 / MBON05 × X, Y, every seed)."""
    c = counts(_stages(w_res)["pre"])
    v = np.stack([np.asarray(v_result["report"]["pre"]["A"]), np.asarray(v_result["report"]["pre"]["P"])], axis=1)
    return [] if np.array_equal(c, v) else [f"{tag} 순진 카운트"]


def reward_check(w_res: dict) -> list:
    """W.9.9 P2-9: after the reward phase the plastic weights moved, the reward core's weights fell, and MBON05(X)'s
    mean over the probes fell."""
    st = _stages(w_res)
    pre, s1 = counts(st["pre"]), counts(st["S1"])
    out = []
    if st["S1"]["w_sha256"] == w_res["w0_sha256"]:
        out.append("가르친 가중치 불변")
    if not st["S1"]["weights_frac_P"] < 1.0:
        out.append(f"보상 구획 가중치 비 {st['S1']['weights_frac_P']}")
    if not s1[:, WV.P, WV.X].mean() < pre[:, WV.P, WV.X].mean():
        out.append(f"MBON05(X) {pre[:, WV.P, WV.X].mean()} → {s1[:, WV.P, WV.X].mean()}")
    return out


# ================================================================ the pilot (W.6, W.9.2, W.9.4, W.9.8 H6)
def _corr(st) -> list:
    """The fly-level correlation of the four gates (±∞ clipped to ±10, a constant gate → 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.corrcoef(np.nan_to_num(np.asarray(st, float), posinf=10, neginf=-10).T)).tolist()


def _pcorr(d: dict, s1: str, s2: str, z: dict) -> float | None:
    """W.6: the probe-noise correlation of ΔV between two slots (same probe seeds), fly means removed, pooled."""
    a, b = WV.dv(d[s1], z), WV.dv(d[s2], z)
    a, b = (a - a.mean(-1, keepdims=True)).ravel(), (b - b.mean(-1, keepdims=True)).ravel()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _sd(x) -> float | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size >= 2 else None


def pilot_record(pairs: dict, z: dict, walls: dict, spec) -> dict:
    """pairs = {key: {stage: [F, K, 2, 2]}} (exploratory). Per pair: fly gate d′, their medians, q_sat at 0.75, the
    mechanism fractions, the naive d′; pooled: floors, Y spillover, presentation-evoked MBON13(X) suppression, the
    naive-floor share, the per-unit wall clock."""
    per, fl_any, fl_both, n_probe = {}, 0, 0, 0
    for key, d in pairs.items():
        st = WV.gate_stats(d, z)
        cls = WV.fly_class(st, spec.bar, spec.band_width, spec.round_digits)
        fr = WV.mech_fractions(d)
        a = {s: np.asarray(d[s], float) for s in WV.STAGES}
        r1, r2 = a["R1"], a["R2"]
        both = ((r1[..., WV.P, WV.X] == 0) & (r1[..., WV.P, WV.Y] == 0)) | \
               ((r2[..., WV.A, WV.X] == 0) & (r2[..., WV.A, WV.Y] == 0))
        taught = (r1[..., WV.P, WV.X] == 0) | (r2[..., WV.A, WV.X] == 0)
        fl_both += int(both.sum())
        fl_any += int(taught.sum())
        n_probe += int(both.size)
        rew = (r1 - a["N1"]).mean((0, 1))
        pun = ((r2 - r1) - (a["RN2"] - a["RN1"])).mean((0, 1))
        per[key] = dict(
            fly_stats={g: st[:, i].tolist() for i, g in enumerate(WV.GATES)},
            median={g: float(np.median(st[:, i])) for i, g in enumerate(WV.GATES)},
            q_sat=float((cls == WV.FLY_SAT).mean()), mech=dict(reward=float(np.median(fr[:, 0])),
                                                              punish=float(np.median(fr[:, 1]))),
            naive_d=WV.naive_dprime(d["pre"], z),
            naive_median=dict(MBON13_X=float(np.median(a["pre"][..., WV.A, WV.X])),
                              MBON05_X=float(np.median(a["pre"][..., WV.P, WV.X]))),
            spill=dict(reward=dict(x=float(rew[WV.P, WV.X]), y=float(rew[WV.P, WV.Y])),
                       punish=dict(x=float(pun[WV.A, WV.X]), y=float(pun[WV.A, WV.Y]))),
            suppression=dict(N2_minus_pre=float((a["N2"] - a["pre"])[..., WV.A, WV.X].mean()),
                             RN2_minus_RN1=float((a["RN2"] - a["RN1"])[..., WV.A, WV.X].mean())),
            between_fly_sd={g: _sd(st[:, i]) for i, g in enumerate(WV.GATES)},
            brain_noise_corr=dict(R1_N1=_pcorr(d, "R1", "N1", z), R2_RN2=_pcorr(d, "R2", "RN2", z),
                                  R1_R2=_pcorr(d, "R1", "R2", z)),
            gate_corr=_corr(st))
    naive_floor = [k for k, p in per.items() if min(p["naive_median"].values()) < spec.naive_floor_spikes]
    bins = {"<1": [], "1-5": [], ">=5": []}
    for k, p in per.items():
        ad = abs(p["naive_d"])
        if p["between_fly_sd"]["punish_assoc"] is not None:
            bins["<1" if ad < 1 else ("1-5" if ad < 5 else ">=5")].append(p["between_fly_sd"]["punish_assoc"])
    return dict(pairs=per, floor_both_share=fl_both / max(n_probe, 1), floor_taught_share=fl_any / max(n_probe, 1),
                naive_floor_pairs=naive_floor, naive_floor_share=len(naive_floor) / max(len(per), 1),
                noise_by_naive_bin={b: (float(median(v)) if v else None) for b, v in bins.items()},
                walls=walls, label="탐색")


# ================================================================ costs (W.9.6 F, W.9.9 P1-4 / P1-5)
def unit_costs(job_results: list, trials: int, oracle_round_s: float, workers: int) -> dict:
    """Seconds per trial (train_s / trials of a learning job) and per presentation (probe_s / (2 odours × probes ×
    stages)), medians over the jobs; the oracle's seconds per worker round; the worker count."""
    tr = [r["train_s"] / trials for r in job_results if r.get("train_s")]
    pr = [r["probe_s"] / (2 * len(r["probe_seeds"]) * len(r["stages"])) for r in job_results]
    return dict(trial_s=float(np.median(tr)) if tr else 0.0, presentation_s=float(np.median(pr)),
                oracle_round_s=float(oracle_round_s), workers=int(workers))


def _rounds(n: int, w: int) -> int:
    return -(-int(n) // max(1, int(w)))


def job_s(c: dict, k: int, trials: int = 40, stages: int = 3) -> float:
    return trials * c["trial_s"] + stages * 2 * k * c["presentation_s"]


def design_cost(c: dict, K: int, F: int, spec, n_set: int, n_naive: int | None = None, with_c: bool = True) -> dict:
    """Hours of the remaining W work for design (K, F): the oracle screen of n_set pairs, the naive screen of
    n_naive pairs (default n_set: the worst case), the learning measurement, the band re-measure upper bound (every
    gate pair), C (if kept) and the plasticity-off control."""
    w, kc = c["workers"], spec.k_cap
    n_naive = n_set if n_naive is None else n_naive
    parts = dict(oracle=_rounds(n_set, w) * c["oracle_round_s"],
                 naive=_rounds(n_naive * F, w) * 2 * K * c["presentation_s"],
                 learn=_rounds(kc * F * 3, w) * job_s(c, K),
                 band=_rounds(kc * F * 3, w) * job_s(c, K),
                 c=(_rounds(kc * F * 3, w) * job_s(c, K)) if with_c else 0.0,
                 noplast=_rounds(spec.noplast_pairs * spec.noplast_flies, w) * job_s(c, K))
    return dict(parts_h={k: v / 3600 for k, v in parts.items()}, total_h=sum(parts.values()) / 3600)


def elapsed_h(ledger: list) -> float:
    return float(sum(e.get("wall_s", 0.0) for e in ledger)) / 3600


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
