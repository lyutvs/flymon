"""The pure rules of spec appendix N as amended by N.8. No engine, no pool.
- The presentation state (N.8.8).
- N0f's per-cell statistics and operating-point rule (N.8.3), with the state-share flags.
- N0's similarity gate (N.8.4).
- N1's per-pair oracle reading and gate (N.3, N.8.5).
- N2's raw-unit statistics, validity checks and verdict (N.8.6).
- N2.0's thresholds, budget and outcome.
- The design view N2.0 may read from block n0f (blinding), and the sentences."""
from __future__ import annotations

import numpy as np

from .h4_formula import dprime, dv

OPERATING_POINT = "OPERATING_POINT"
STOP_DATA_MISMATCH = "STOP_DATA_MISMATCH"
STOP_NO_OPERATING_POINT = "STOP_NO_OPERATING_POINT"
SIMILARITY_GO = "SIMILARITY_GO"
STOP_SIMILARITY_ORDER = "STOP_SIMILARITY_ORDER"
N1_GO = "N1_GO"
STOP_UNTESTABLE = "STOP_UNTESTABLE"
N2_0_GO = "N2_0_GO"
STOP_POWER = "STOP_POWER"
STOP_BUDGET = "STOP_BUDGET"
SUPPORTED = "SUPPORTED"
NOT_REPLICATED = "NOT_REPLICATED"
NO_LEARNING = "NO_LEARNING"
INVALID = "INVALID"


def state(P: int, spec) -> str:
    return "firing" if int(P) >= spec.state_p_min else "silent"


# ================================================================ N0f (N.8.3)
def cell_stats(rows: list, spec) -> dict:
    """One (odour, condition) block of presentation rows -> N.8.3's numbers."""
    kc = np.array([r["kc_frac"] for r in rows], float)
    hz = np.array([r["kc_max_win_hz"] for r in rows], float)
    A = np.array([r["A"] for r in rows], float); P = np.array([r["P"] for r in rows], float)
    return dict(n=len(rows), kc_frac_median=float(np.median(kc)), kc_frac=kc.tolist(),
                runaway_share=float((hz > spec.runaway_hz).mean()), max_win_hz_max=float(hz.max()),
                A_median=float(np.median(A)), P_median=float(np.median(P)),
                A_zero_share=float((A == 0).mean()), P_zero_share=float((P == 0).mean()),
                firing_share=float((P >= spec.state_p_min).mean()),
                apl_out_mean=float(np.mean([r["apl_out_per_step"] for r in rows])),
                wall_s_median=float(np.median([r["wall_s"] for r in rows])))


def wall_per_step(rows: list) -> float:
    return float(np.median([r["wall_s"] / r["steps"] for r in rows]))


def point_key(g: float, c_delta: float) -> str:
    return f"{g:g}|{c_delta:g}"


def point_checks(cells: dict, spec) -> dict:
    """cells {stimulus: {"on" | "block" | "all": cell_stats}} at one (g, c_δ). Returns N.8.3's clauses ① ② ③ on the
    judged stimuli (IA / EB alone are records) and the distance of the on medians' mean from the target."""
    lo, hi = spec.kc_band
    on = [cells[s]["on"] for s in spec.judged_stimuli]
    both = on + [cells[s]["block"] for s in spec.judged_stimuli]
    band = all(lo <= c["kc_frac_median"] <= hi for c in on)
    runaway = all(c["kc_frac_median"] <= spec.kc_block_max and c["runaway_share"] <= spec.runaway_share_max
                  for c in both)
    floor = all(c["A_zero_share"] <= spec.zero_share_max and c["P_zero_share"] <= spec.zero_share_max for c in both)
    mean_on = float(np.mean([c["kc_frac_median"] for c in on]))
    return dict(band=bool(band), runaway=bool(runaway), floor=bool(floor), ok=bool(band and runaway and floor),
                mean_on=mean_on, dist=abs(mean_on - spec.kc_target))


def select_point(grid: dict, spec) -> dict:
    """grid {(g, c_δ): cells} -> the passing point nearest the target; ties -> smaller g, then smaller c_δ."""
    checks = {k: point_checks(v, spec) for k, v in grid.items()}
    table = {point_key(*k): c for k, c in checks.items()}
    ok = [k for k, c in checks.items() if c["ok"]]
    if not ok:
        return dict(outcome=STOP_NO_OPERATING_POINT, selected=None, checks=table)
    g, c = min(ok, key=lambda k: (checks[k]["dist"], k[0], k[1]))
    return dict(outcome=OPERATING_POINT, selected=dict(g=float(g), c_delta=float(c)), checks=table)


def state_flags(cells: dict, spec) -> list:
    """Every (stimulus, block condition) whose firing share differs from the on share by more than 0.25 (a record)."""
    out = []
    for s, by in cells.items():
        for cond in ("block", "all"):
            if abs(by[cond]["firing_share"] - by["on"]["firing_share"]) > spec.state_diff_flag:
                out.append(dict(stimulus=s, condition=cond, on=by["on"]["firing_share"],
                                share=by[cond]["firing_share"]))
    return out


# ================================================================ N0 (N.8.4)
def _pearson_rows(a, b) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True); b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), np.nan)


def similarity(fired: dict, n_kc: int, spec, draws: int | None = None, seed: int | None = None) -> dict:
    """fired {stimulus: [fired KC positions per seed]} (the same seeds in the same order) -> r(4:1, 1:4),
    r(4:1, δ-DL) of the KC firing-probability vectors, Δr, and the common-seed bootstrap 95% CI. SIMILARITY_GO iff
    the lower bound is finite and > 0; an undefined r (a silent vector) fails."""
    x, y_sim = spec.pair_stimuli()["sim"]
    y_dis = spec.pair_stimuli()["dis"][1]
    n = len(fired[x])
    if n < 2 or any(len(fired[s]) != n for s in (y_sim, y_dis)):
        raise ValueError("similarity needs the same >= 2 seeds for every stimulus")
    M = {}
    for s in (x, y_sim, y_dis):
        m = np.zeros((n, int(n_kc)))
        for i, f in enumerate(fired[s]):
            m[i, np.asarray(f, np.int64)] = 1.0
        M[s] = m
    f = {s: M[s].mean(0) for s in M}
    r_sim, r_dis = float(_pearson_rows(f[x], f[y_sim])), float(_pearson_rows(f[x], f[y_dis]))
    draws = spec.boot_draws if draws is None else int(draws)
    idx = np.random.default_rng(spec.boot_seed if seed is None else seed).integers(0, n, (draws, n))
    wt = np.stack([np.bincount(row, minlength=n) for row in idx]) / n                  # draws x seeds weights
    d = np.empty(draws)
    for a in range(0, draws, 500):
        fb = {s: wt[a:a + 500] @ M[s] for s in M}
        d[a:a + 500] = _pearson_rows(fb[x], fb[y_sim]) - _pearson_rows(fb[x], fb[y_dis])
    if np.isfinite(d).all():
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    else:
        lo = hi = float("nan")
    ok = bool(np.isfinite(lo) and lo > 0)
    return dict(outcome=SIMILARITY_GO if ok else STOP_SIMILARITY_ORDER, r_sim=r_sim, r_dis=r_dis,
                delta_r=r_sim - r_dis, ci95=[lo, hi], draws=draws, n_seeds=n)


# ================================================================ N1 (N.3, N.8.5)
def n1_pair(row: dict, z: dict, spec) -> dict:
    """p0 = d'(dV_P - dV_pre) on the report seeds; testable iff p0 <= -testable_min and sd(dV_P - dV_pre) > 0;
    o = mean(dV_P - dV_pre) in z units (N.8.6's c1)."""
    pre, post = dv(row["report"]["pre"], z), dv(row["report"]["P"], z)
    d = post - pre
    p0 = dprime(d)
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    A = np.asarray(row["report"]["pre"]["A"], float); P = np.asarray(row["report"]["pre"]["P"], float)
    return dict(p0=p0, sd=sd, o=float(d.mean()), d_pre=dprime(pre),
                testable=bool(p0 is not None and p0 <= -spec.testable_min and sd > 0), alpha=row["alpha_punish"],
                changes={a: v["change"] for a, v in row["select"]["punish"].items()}, jaccard=row["kc"]["jaccard"],
                floor=dict(A_zero_share=float((A == 0).mean()), P_zero_share=float((P == 0).mean())))


def n1_outcome(pairs: dict) -> dict:
    """N1_GO iff both pairs are testable. The two pairs are named, so a missing one raises (no empty all())."""
    t = {k: bool(pairs[k]["testable"]) for k in ("sim", "dis")}
    if t["sim"] and t["dis"]:
        return dict(outcome=N1_GO, note=None)
    note = ("다른 쌍만 시험 가능: APL이 있어도 비슷한 쌍의 변별이 사전 지정 편집 프로토콜에서 서지 않는다"
            if t["dis"] and not t["sim"] else None)
    return dict(outcome=STOP_UNTESTABLE, note=note)


# ================================================================ N2 (N.8.6)
N2_ORDER = ("sim_on", "sim_off", "dis_on", "dis_off")


def _block(side: dict) -> dict:
    return {"A": [[side["x"]["A"], side["y"]["A"]]], "P": [[side["x"]["P"], side["y"]["P"]]]}


def delta(row: dict, z: dict) -> float:
    """Δ = dV_post - dV_pre of one (condition, seed) arm; dV = h4_formula.dv (V = z_A - z_P, X - Y), z units."""
    return float(dv(_block(row["post"]), z)[0] - dv(_block(row["pre"]), z)[0])


def deltas(by: dict, conds, z: dict) -> dict:
    """{cond: Δ per seed, in seed order}; ValueError unless every condition has the same seeds (paired by seed), each
    once, and at least one; a condition absent from `by` raises KeyError."""
    conds = list(conds)
    if not conds:
        raise ValueError("no condition to read")
    seeds, out = None, {}
    for c in conds:
        rows = sorted(by[c], key=lambda r: r["seed"])
        s = [int(r["seed"]) for r in rows]
        if not s:
            raise ValueError(f"condition {c} has no rows")
        if len(set(s)) != len(s):
            raise ValueError(f"condition {c} repeats a seed: {s}")
        if seeds is None:
            seeds = s
        elif s != seeds:
            raise ValueError(f"condition {c} has seeds {s}, not {seeds} (N2 pairs the conditions by seed)")
        out[c] = np.array([delta(r, z) for r in rows])
    return out


def boot_index(n: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, int(n), (int(draws), int(n)))


def _ci(v, level: float) -> list:
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(np.percentile(v, tail)), float(np.percentile(v, 100.0 - tail))]


def n2_stats(delta: dict, spec, draws: int | None = None, seed: int | None = None) -> dict:
    """ℓ_k = -mean Δ_k; D_sim = ℓ_sim,on - ℓ_sim,off, D_dis likewise; CIs from one common-seed resample per draw
    (spec.ci_level for ℓ and D_sim, spec.equiv_ci_level for D_dis's TOST). The four conditions are named (a missing
    one raises KeyError); they need the same >= 2 seeds."""
    arr = [np.asarray(delta[k], float) for k in N2_ORDER]
    if len({a.shape for a in arr}) != 1 or arr[0].ndim != 1 or arr[0].size < 2:
        raise ValueError(f"N2 needs the same >= 2 seeds in every condition, got {[a.shape for a in arr]}")
    X = np.stack(arr)
    ell = -X.mean(1)
    idx = boot_index(X.shape[1], spec.boot_draws if draws is None else draws, spec.boot_seed if seed is None else seed)
    B = -X[:, idx].mean(2)
    return dict(n=int(X.shape[1]), ell={k: float(ell[i]) for i, k in enumerate(N2_ORDER)},
                ell_ci95={k: _ci(B[i], spec.ci_level) for i, k in enumerate(N2_ORDER)},
                D_sim=float(ell[0] - ell[1]), D_sim_ci95=_ci(B[0] - B[1], spec.ci_level),
                D_dis=float(ell[2] - ell[3]), D_dis_ci90=_ci(B[2] - B[3], spec.equiv_ci_level))


def verdict(st: dict, c1: dict, delta_min: float, eps: float, invalid=()) -> dict:
    """INVALID first. Then:
    - ① ℓ_sim,on >= c1_sim and ℓ_dis,on >= c1_dis, each 95% CI lower > 0; ¬① -> NO_LEARNING;
    - ② D_sim 95% CI lower >= δ_min;
    - ③ D_dis 90% CI inside [-ε, ε];
    - ④ ℓ_dis,off >= c1_dis.
    SUPPORTED iff ② ③ ④ all hold, else NOT_REPLICATED. A threshold that is not a positive number is refused
    (ValueError): with c1 / δ_min <= 0 the clause it guards would hold for nothing. An undefined statistic (nan) fails
    its clause; a missing one raises KeyError."""
    if invalid:
        return dict(verdict=INVALID, clauses=None, reasons=list(invalid))
    th = dict(c1_sim=c1["sim"], c1_dis=c1["dis"], delta_min=delta_min, eps=eps)
    unset = [k for k, v in th.items() if not float(v) > 0]
    if unset:
        raise ValueError(f"threshold(s) {unset} not positive ({th}): N2 cannot be judged")
    e, ci = st["ell"], st["ell_ci95"]
    one = bool(e["sim_on"] >= c1["sim"] and e["dis_on"] >= c1["dis"] and ci["sim_on"][0] > 0 and ci["dis_on"][0] > 0)
    two = bool(st["D_sim_ci95"][0] >= delta_min)
    three = bool(-eps <= st["D_dis_ci90"][0] and st["D_dis_ci90"][1] <= eps)
    four = bool(e["dis_off"] >= c1["dis"])
    clauses = dict(one=one, two=two, three=three, four=four)
    v = NO_LEARNING if not one else (SUPPORTED if two and three and four else NOT_REPLICATED)
    return dict(verdict=v, clauses=clauses, reasons=[])


def plumbing(rows: list) -> list:
    """N.8.7's plumbing check on the plasticity-off reruns: probes unchanged and weights exactly w0. No rerun at all, or
    a row that ran with plasticity on, fails (the check did not happen)."""
    if not rows:
        return ["plumbing check failed: no plasticity-off rerun was given"]
    bad = []
    for r in rows:
        if r["plastic"] is not False:
            bad.append(f"plumbing check failed: {r['cond']} seed {r['seed']} ran with plasticity on")
            continue
        same = all(r["pre"][k][f] == r["post"][k][f] for k in ("x", "y") for f in ("A", "P", "kc_spikes"))
        if not same or r["weights_frac"] != 1.0:
            bad.append(f"plumbing check failed: {r['cond']} seed {r['seed']} (probes changed or weights moved)")
    return bad


def block_validity(by: dict, spec) -> list:
    """N.8.3 ② / ③ on the block conditions' naive (pre-training) probes (plan reading 10). A stimulus with no block
    probe is a failure, not a pass."""
    x, y_sim = spec.pair_stimuli()["sim"]
    y_dis = spec.pair_stimuli()["dis"][1]
    stim = {x: [r["pre"]["x"] for r in by["sim_off"]], y_sim: [r["pre"]["y"] for r in by["sim_off"]],
            y_dis: [r["pre"]["y"] for r in by["dis_off"]]}
    bad = []
    for s, rows in stim.items():
        if not rows:
            bad.append(f"{s} block: no block-condition naive probe")
            continue
        c = cell_stats(rows, spec)
        if not c["kc_frac_median"] <= spec.kc_block_max:
            bad.append(f"{s} block: KC median {c['kc_frac_median']:.3f} > {spec.kc_block_max}")
        if not c["runaway_share"] <= spec.runaway_share_max:
            bad.append(f"{s} block: sub-window > {spec.runaway_hz:g} Hz in {c['runaway_share']:.3f} of presentations")
        for k in ("A", "P"):
            if not c[f"{k}_zero_share"] <= spec.zero_share_max:
                bad.append(f"{s} block: readout {k} zero share {c[f'{k}_zero_share']:.3f} > {spec.zero_share_max}")
    return bad


def csc_checks(by: dict) -> list:
    """Plan reading 10: one CSC weight vector per condition, and each judged block arm's differs from its on arm's.
    Every judged arm (N2_ORDER) must be there."""
    bad = [f"condition {c} is missing: no CSC weight vector to check" for c in N2_ORDER if c not in by]
    sha = {c: sorted({r["csc_sha256"] for r in rows}) for c, rows in by.items()}
    bad += [f"condition {c} ran on {len(s)} CSC weight vectors" for c, s in sha.items() if len(s) != 1]
    for p in ("sim", "dis"):
        if f"{p}_on" in sha and f"{p}_off" in sha and sha[f"{p}_on"] == sha[f"{p}_off"]:
            bad.append(f"{p}: the APL->KC edit changed no weight")
    return bad


def dprime_record(delta: dict) -> dict:
    L = {}
    for k in N2_ORDER:
        d = dprime(delta[k])
        L[k] = None if d is None else -d
    I = None if any(v is None for v in L.values()) else (L["sim_on"] - L["sim_off"]) - (L["dis_on"] - L["dis_off"])
    return dict(L=L, I=I, note="d′ 기반 L·I는 기록만 (N.8.6)")


def state_conditional(by: dict, z: dict, spec) -> dict:
    """N.8.8's record: ℓ over the seeds whose four probes (pre / post x X / Y) are all in the firing state."""
    out = {}
    for c, rows in by.items():
        keep = [r for r in rows if all(r[ph][k]["P"] >= spec.state_p_min for ph in ("pre", "post") for k in ("x", "y"))]
        out[c] = dict(n=len(keep), n_all=len(rows), ell=(-float(np.mean([delta(r, z) for r in keep])) if keep else None))
    return out


# ================================================================ N2.0 (N.8.6, decision ⑥)
def thresholds(o: dict, pilot: dict, spec) -> dict:
    """c1 = c1_frac |o_pair| (N1); δ_min = delta_min_frac ℓ̂_sim,on, ε = eps_frac ℓ̂_dis,on, alternative D_sim =
    alt_frac ℓ̂_sim,on (pilot). Calibratable iff both ℓ̂ and both c1 are positive numbers (plan reading 8)."""
    ell_hat = {k: -float(np.mean(pilot[k])) for k in ("sim", "dis")}
    c1 = {k: spec.c1_frac * abs(float(o[k])) for k in ("sim", "dis")}
    return dict(ell_hat=ell_hat, pilot_sd={k: float(np.std(pilot[k], ddof=1)) for k in ("sim", "dis")}, c1=c1,
                delta_min=spec.delta_min_frac * ell_hat["sim"], eps=spec.eps_frac * ell_hat["dis"],
                alt_D_sim=spec.alt_frac * ell_hat["sim"],
                calibratable=bool(all(ell_hat[k] > 0 and c1[k] > 0 for k in ("sim", "dis"))))


def steps_per_arm(spec, dt: float) -> int:
    h4 = spec.h4
    probe = int(round((h4.oracle_window.settle_ms + h4.oracle_window.read_ms) / dt))
    trial = int(round((h4.teach_window.settle_ms + h4.teach_present_ms + h4.teach_gap_ms) / dt))
    return 4 * probe + int(h4.teach_trials) * trial


def budget_hours(n: int, wall_s_per_step: float, spec, dt: float = 1.0) -> float:
    """Plan reading 11: every judged and record arm, plus the plumbing reruns, over budget_workers workers."""
    conds = len(spec.n2_conditions) + len(spec.n2_record_conditions)
    items = int(n) * conds + int(spec.plumbing_seeds) * conds
    return items * steps_per_arm(spec, dt) * float(wall_s_per_step) / spec.budget_workers / 3600.0


DESIGN_KEYS = ("outcome", "run_id", "selected", "wall_s_per_step", "state_shares", "validity", "state_flags")


def design_view(n0f: dict) -> dict:
    """What N2.0 may read from block n0f (N.8.6: state shares and validity numbers only; no block KC correlation)."""
    return {k: n0f.get(k) for k in DESIGN_KEYS}


def scenario_key(pairing, mult: float) -> str:
    """One OC scenario: how the block arm is paired with the on arm x its deviation multiplier."""
    return f"{pairing}|{float(mult):g}x"


def n2_0_outcome(th: dict, oc: dict | None, budget_h: float, spec) -> dict:
    """STOP_POWER (uncalibratable pilot, then no n) before STOP_BUDGET. N2_0_GO needs the OC's per-scenario minimal n
    for every spec.oc_pairings x spec.sd_mults scenario, none above n (else ValueError: the table is not the declared
    one), n on the grid, and a budget that is a number <= budget_h (nan stops)."""
    if not th["calibratable"]:
        return dict(outcome=STOP_POWER, reason=f"pilot ℓ̂ ≤ 0 or c₁ = 0 ({th['ell_hat']}): δ_min, ε, c₁ and the "
                                               "alternative cannot be set")
    if oc is None or oc.get("n") is None:
        return dict(outcome=STOP_POWER, reason=(f"no n in {tuple(spec.n_grid)} with null ≤ {spec.null_max} and power "
                                                f"≥ {spec.power_min} under every spread {tuple(spec.sd_mults)} x "
                                                f"pairing {tuple(spec.oc_pairings)}"))
    n, per = oc["n"], oc.get("scenario_n") or {}
    want = [scenario_key(p, m) for p in spec.oc_pairings for m in spec.sd_mults]
    short = [k for k in want if per.get(k) is None or per[k] > n]
    if short:
        raise ValueError(f"OC scenario(s) {short} not met at n = {n} (scenario_n = {per}): refusing N2_0_GO")
    if n not in tuple(spec.n_grid):
        raise ValueError(f"n = {n} is not on the grid {tuple(spec.n_grid)}")
    if not budget_h <= spec.budget_h:
        return dict(outcome=STOP_BUDGET, reason=f"judgement block {budget_h:.1f} h > {spec.budget_h:g} h")
    return dict(outcome=N2_0_GO, reason=None)


# ================================================================ sentences
def sentence(name: str, res: dict) -> str:
    o = res.get("outcome")
    if name == "n0f":
        if o == OPERATING_POINT:
            s = res["selected"]
            return f"N0f: 작동점 g = {s['g']:g}, c_δ = {s['c_delta']:g}. N.8a를 커밋한 뒤 N0로 간다."
        if o == STOP_DATA_MISMATCH:
            return f"N0: 데이터 불일치 — {res.get('reason')} (STOP_DATA_MISMATCH). 사용자 판단으로 넘긴다."
        return ("N0f: 켬 대역·차단 폭주·판독 바닥 세 조건을 모두 만족하는 (g, c_δ)가 없다 (STOP_NO_OPERATING_POINT). "
                "사용자 판단으로 넘긴다.")
    if name == "n0":
        s = res["similarity"]
        head = (f"N0: Δr = r(4:1, 1:4) − r(4:1, δ-DL) = {s['delta_r']:.3f}, 95% CI "
                f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
        return head + (" — 유사도 순서 통과, N1로 간다." if o == SIMILARITY_GO
                       else " — STOP_SIMILARITY_ORDER. 사용자 판단으로 넘긴다.")
    if name == "n1":
        p = res["pairs"]
        head = f"N1: p0 비슷한 쌍 {p['sim']['p0']}, 다른 쌍 {p['dis']['p0']}"
        if o == N1_GO:
            return head + " — 두 쌍 모두 시험 가능, N2.0으로 간다."
        return (head + " — STOP_UNTESTABLE: 사전 지정 편집 프로토콜(편집식 하나, α 3점)에서 시험이 성립하지 않는다."
                + (f" {res['note']}" if res.get("note") else ""))
    if name == "n2_0":
        if o == N2_0_GO:
            return f"N2.0: n = {res['n']}, 판정 블록 예상 {res['budget_h']:.1f} h. N.8b를 커밋한 뒤 N2 판정으로 간다."
        return f"N2.0: {o} — {res.get('reason')}. 사용자 판단으로 넘긴다."
    if name == "n2":
        v = res["verdict"]
        if o == SUPPORTED:
            return ("N2 SUPPORTED: C3 엔진에서 Hallem 2006 실제 냄새로, APL→KC 선택 차단이 비슷한 쌍의 학습된 변별만 "
                    "떨어뜨린다(Lin 2014 유사 패턴).")
        if o == NOT_REPLICATED:
            return f"N2 NOT_REPLICATED: 학습은 있으나 조항 {[k for k, x in v['clauses'].items() if not x]} 불성립."
        if o == NO_LEARNING:
            return "N2 NO_LEARNING: 학습량이 사전 지정 편집 프로토콜 효과의 1/4에 못 미친다(D.6 (c) 성격의 기록, 판정 아님)."
        if o == INVALID:
            return f"N2 INVALID: {'; '.join(v['reasons'])}"
        raise ValueError(f"unknown N2 outcome {o!r}")
    raise ValueError(f"unknown stage {name!r}")
