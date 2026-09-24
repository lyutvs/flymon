"""Spec J.12.9 (replacing J.11.4-6): the operating characteristic of stage 2's reading, computed before stage 2 runs
(a record shown to the user; it does not change the reading — J.12.9 decision 2's bands are how it is used).

Fixed pair list: H.4's C3 record (block "h4" of results/summary/m0d.json, committed) — its even-turn (b) pairs
(21, per turn 3 2 3 3 3 3 2 2) and its naive-balanced (a) pairs (turn 0: 1, turn 4: 3), both found by applying
h4_formula.arm_aggregate to the record; the script stops unless that equals the recorded aggregate (testable_b 7,
naive_a 4, F_a 2) and n_b is the declared j_spec.SPEC.stage2_n_b. Turns are not resampled; only each pair's testable
outcome is random.

Model (model-based, not an empirical bootstrap): pair probability logistic(c + u_turn), u_turn the turn's effect in
the C3 record (add-0.5 smoothing, as J.12.7): u_turn = logit((b_testable + 0.5)/(b + 1)) - logit(overall). The
intercept c is solved per axis so the mean pair probability over the fixed (b) pairs is q_b and over the fixed naive
(a) pairs is q_a. The baseline u = 0 is the plain binomial. testable_b and F_a are Poisson-binomial sums over
independent pairs, computed exactly by convolution (no sampling); every (testable_b, F_a) cell is read by
j_rules.stage2_reading itself and weighted by its probability.
    uv run python docs/superpowers/specs/j-diag/stage2_oc.py [--summary results/summary/m0d.json]
                                                             [--out results/j/diag/stage2_oc.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from flymon.brain.h4_formula import arm_aggregate
from flymon.brain.j_rules import B, B_FA, B_NO_CONCLUSION, B_TB, SELECTED, stage2_reading
from flymon.brain.j_spec import SPEC

ROOT = Path(__file__).resolve().parents[4]
QS = (0.33, 0.5, 0.6, 0.7)
RATIOS = (0.5, 1.0)                         # q_a / q_b in the sensitivity rows
NAIVE_RANGE = tuple(range(2, 7))            # naive (a) pair counts in the sensitivity rows
BANDS = (SELECTED, B_TB, B_NO_CONCLUSION, B_FA)
C3_AGGREGATE = dict(testable_b=7, naive_a=4, F_a=2)    # the recorded C3 values J.12.9 names
PROVENANCE_FILES = ("flymon/brain/j_rules.py", "flymon/brain/j_spec.py", "flymon/brain/h4_spec.py",
                    "flymon/brain/h4_formula.py", "docs/superpowers/specs/j-diag/stage2_oc.py")
ASSUMPTIONS = (
    "C3's turn structure (the 21 even-turn (b) pairs per turn and their turn effects) and its naive (a) composition "
    "(4 naive-balanced (a) pairs: turn 0 one, turn 4 three) carry over to the STD engine; only each pair's testable "
    "outcome is random, pairs independent given the turn effect.",
    "The confirmation step (H.5, the M0d confirmation turn set) is not modelled: P(SELECTED) is before confirmation; "
    "P(SELECTED and confirmed) is not computed.",
    "Model-based (a logistic turn-effect model calibrated to the declared q per axis, and a u = 0 binomial baseline), "
    "not an empirical bootstrap; the distributions are exact (Poisson-binomial by convolution), no sampling error.",
    "If stage 2's naive_a differs from C3's 4, the B sentence cites the matching sensitivity row (J.12.9).",
)


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


# ================================================================ the fixed pair list
def _key(p: dict) -> tuple:
    return (p["axis"], int(p["turn"]), p["x"], p["y"])


def structure(rec: dict) -> dict:
    """The C3 record's fixed pair list: per turn the (b) pair count and testable count, and the naive (a) pairs —
    naive-ness and the aggregate taken from h4_formula.arm_aggregate. Stops unless it equals the recorded aggregate."""
    stats = {_key(p): dict(d_pre=p["d_pre"], testable=bool(p["testable"])) for p in rec["pairs"]}
    h4 = SPEC.h4
    agg = arm_aggregate(stats, h4.naive_max, h4.t_b_min, h4.f_a_min)
    want = rec["aggregate"]
    for k in ("n_b", "testable_b", "naive_a", "F_a"):
        if agg[k] != want[k]:
            raise SystemExit(f"stage2_oc: arm_aggregate's {k} = {agg[k]} on the record's pairs, the recorded "
                             f"aggregate says {want[k]}: refusing")
    for k, v in C3_AGGREGATE.items():
        if want[k] != v:
            raise SystemExit(f"stage2_oc: the recorded aggregate's {k} = {want[k]}, not C3's {v} (J.12.9): refusing")
    if agg["n_b"] != SPEC.stage2_n_b:
        raise SystemExit(f"stage2_oc: n_b = {agg['n_b']} is not the declared stage2_n_b {SPEC.stage2_n_b}: refusing")
    b_only = {k: s for k, s in stats.items() if k[0] == "b"}
    turns = {}
    for k, s in stats.items():
        t = turns.setdefault(k[1], dict(b=0, b_testable=0, a=0, a_naive=0))
        if k[0] == "b":
            t["b"] += 1
            t["b_testable"] += s["testable"]
        else:
            t["a"] += 1
            # the naive rule is arm_aggregate's: one (a) pair with the (b) pairs (T_b needs them), read naive_a
            t["a_naive"] += arm_aggregate({**b_only, k: s}, h4.naive_max, h4.t_b_min, h4.f_a_min)["naive_a"]
    if sum(t["a_naive"] for t in turns.values()) != agg["naive_a"]:
        raise SystemExit("stage2_oc: per-pair naive count differs from arm_aggregate's naive_a: refusing")
    return dict(turns=dict(sorted(turns.items())), aggregate={k: agg[k] for k in ("n_b", "testable_b", "naive_a", "F_a",
                                                                                 "n_a", "testable_a")})


def turn_effects(turns: dict) -> dict:
    nb = sum(t["b"] for t in turns.values())
    overall = (sum(t["b_testable"] for t in turns.values()) + 0.5) / (nb + 1.0)
    return {k: logit((t["b_testable"] + 0.5) / (t["b"] + 1.0)) - logit(overall) for k, t in turns.items()}


def pair_offsets(turns: dict, u: dict | None) -> tuple:
    """(b offsets, naive (a) offsets): one u_turn per fixed pair (0 everywhere when u is None)."""
    off = (lambda k: 0.0) if u is None else (lambda k: u[k])
    ub = [off(k) for k, t in turns.items() for _ in range(t["b"])]
    ua = [off(k) for k, t in turns.items() for _ in range(t["a_naive"])]
    return ub, ua


# ================================================================ calibration and exact distributions
def calibrate(offsets: list, q: float) -> tuple:
    """Intercept c with mean(logistic(c + u)) == q over the fixed pairs (bisection; the mean is increasing in c)."""
    mean = lambda c: sum(logistic(c + u) for u in offsets) / len(offsets)
    lo, hi = -60.0, 60.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean(mid) < q else (lo, mid)
    c = 0.5 * (lo + hi)
    probs = [logistic(c + u) for u in offsets]
    return c, probs


def poisson_binomial(probs: list) -> np.ndarray:
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1.0 - p, p])
    return dist


def read_cells(pb: np.ndarray, pa: np.ndarray, n_b: int, naive_a: int) -> dict:
    """Weight every (testable_b, F_a) cell by its probability and read it with stage2_reading itself."""
    out = {k: 0.0 for k in (SELECTED, B) + BANDS[1:]}
    for tb, wb in enumerate(pb):
        for fa, wa in enumerate(pa):
            r = stage2_reading(dict(T_b=tb / n_b, testable_b=tb, n_b=n_b, F_a=fa, naive_a=naive_a), SPEC)
            w = float(wb * wa)
            out[r["outcome"]] += w
            if r["band"] != SELECTED:
                out[r["band"]] += w
    return out


def oc_row(ub: list, ua: list, q_b: float, q_a: float) -> dict:
    c_b, prob_b = calibrate(ub, q_b)
    c_a, prob_a = calibrate(ua, q_a)
    mean_b, mean_a = float(np.mean(prob_b)), float(np.mean(prob_a))
    assert abs(mean_b - q_b) < 1e-9 and abs(mean_a - q_a) < 1e-9, (mean_b, q_b, mean_a, q_a)
    pb, pa = poisson_binomial(prob_b), poisson_binomial(prob_a)
    cells = read_cells(pb, pa, len(ub), len(ua))
    sel, close, top = SPEC.stage2_select_testable_b, SPEC.stage2_close_max_testable_b, SPEC.stage2_select_testable_b - 1
    return dict(q_b=q_b, q_a=q_a, n_b=len(ub), naive_a=len(ua), intercept_b=c_b, intercept_a=c_a,
                mean_pair_p_b=mean_b, mean_pair_p_a=mean_a,
                p_selected=cells[SELECTED], p_b=cells[B], p_b_tb=cells[B_TB], p_b_no_conclusion=cells[B_NO_CONCLUSION],
                p_b_fa=cells[B_FA], p_testable_b_ge_select=float(pb[sel:].sum()),
                p_f_a_ge_min=float(pa[SPEC.h4.f_a_min:].sum()), p_testable_b_le_close=float(pb[:close + 1].sum()),
                p_testable_b_le_below_select=float(pb[:top + 1].sum()),
                testable_b_dist=[float(x) for x in pb], f_a_dist=[float(x) for x in pa])


# ================================================================ the record
def check_tracked(summary: Path) -> str:
    rel = str(summary.resolve().relative_to(ROOT))
    if git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"stage2_oc: {rel} is not tracked by git: refusing")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"stage2_oc: {rel} has uncommitted changes: refusing")
    return rel


def provenance(summary_rel: str, args: dict) -> dict:
    files = {summary_rel: sha256(ROOT / summary_rel), **{f: sha256(ROOT / f) for f in PROVENANCE_FILES}}
    dirty = git("status", "--porcelain", "--", *files).stdout.splitlines()
    return dict(sha256=files, commit=git("rev-parse", "HEAD").stdout.strip(), dirty=dirty, args=args,
                argv=sys.argv[1:])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/summary/m0d.json")
    ap.add_argument("--out", default="results/j/diag/stage2_oc.json")
    a = ap.parse_args(argv)
    summary = Path(a.summary) if Path(a.summary).is_absolute() else (Path.cwd() / a.summary)
    rel = check_tracked(summary)
    rec = json.loads(summary.read_text())["h4"]["h4"]["combos"]["C3"]["oracle"]
    st = structure(rec)
    turns, u = st["turns"], turn_effects(st["turns"])
    models = dict(turn_effects=pair_offsets(turns, u), independent=pair_offsets(turns, None))
    rows = [dict(model=m, **oc_row(ub, ua, q, q)) for q in QS for m, (ub, ua) in models.items()]
    sens = []
    ub0 = models["independent"][0]
    for na in NAIVE_RANGE:
        for q in QS:
            for ratio in RATIOS:
                r = oc_row(ub0, [0.0] * na, q, ratio * q)
                sens.append(dict(model="independent", ratio_q_a_over_q_b=ratio,
                                 **{k: v for k, v in r.items() if k not in ("testable_b_dist", "f_a_dist")}))
    res = dict(what="spec J.12.9 operating characteristic of stage 2's reading (replaces J.11.4-6 and J.12.7's numbers)",
               assumptions=list(ASSUMPTIONS),
               c3_record=dict(aggregate=st["aggregate"],
                              turns={str(k): dict(t, u_turn=u[k]) for k, t in turns.items()}),
               bar=dict(t_b_min=SPEC.h4.t_b_min, f_a_min=SPEC.h4.f_a_min, n_b=SPEC.stage2_n_b,
                        select_testable_b=SPEC.stage2_select_testable_b,
                        close_max_testable_b=SPEC.stage2_close_max_testable_b),
               rows=rows, sensitivity=sens,
               provenance=provenance(rel, dict(summary=a.summary, out=a.out)))
    print(f"{'model':13} {'q':>5} {'SELECTED':>9} {'B_Tb':>7} {'B_noC':>7} {'B_Fa':>7} {'Tb>=11':>7} {'Fa>=2':>7} "
          f"{'Tb<=7':>7} {'Tb<=10':>7}")
    for r in rows:
        print(f"{r['model']:13} {r['q_b']:5.2f} {r['p_selected']:9.4f} {r['p_b_tb']:7.4f} {r['p_b_no_conclusion']:7.4f} "
              f"{r['p_b_fa']:7.4f} {r['p_testable_b_ge_select']:7.4f} {r['p_f_a_ge_min']:7.4f} "
              f"{r['p_testable_b_le_close']:7.4f} {r['p_testable_b_le_below_select']:7.4f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {out}  sha256 {sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
