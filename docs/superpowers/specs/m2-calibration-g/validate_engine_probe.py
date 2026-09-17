"""Validation of engine_probe_verdict.py (spec G.14): fixtures near the thresholds, limits, missing data,
every outcome under both contrast tests, and mutations that must change the verdict.

Run:  python validate_engine_probe.py   -> prints one line per check, exits 1 on any failure.
The fixture helpers re-derive m and testable independently of the module on purpose.
"""
from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("engine_probe_verdict", HERE / "engine_probe_verdict.py")
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)

# even-turn structure of the real pair lists (G.12 raw data): turn -> pair count
A_TURNS = {0: 3, 2: 1, 4: 3, 6: 3, 8: 3, 10: 3, 12: 1, 14: 1}      # 18
B_TURNS = {0: 3, 2: 2, 4: 3, 6: 3, 8: 3, 10: 3, 12: 2, 14: 2}      # 21
KEYS_A = [("a", t, f"x{t}_{j}", f"y{t}_{j}") for t, n in A_TURNS.items() for j in range(n)]
KEYS_B = [("b", t, f"x{t}_{j}", f"y{t}_{j}") for t, n in B_TURNS.items() for j in range(n)]
EXPECTED = KEYS_A + KEYS_B
MATCH_OK = {"diff_pp": 0.2}
TESTS = V.TESTS

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))


def S(d_pre: float, r: float, p: float) -> dict:
    m = min(r, -p)
    return {"d_pre": d_pre, "r": r, "p": p, "m": m, "testable": bool(m >= 2.0)}


def arm(a_fn, b_fn) -> dict:
    """a_fn/b_fn: (index within axis, key) -> stats."""
    out = {k: a_fn(i, k) for i, k in enumerate(KEYS_A)}
    out.update({k: b_fn(i, k) for i, k in enumerate(KEYS_B)})
    return out


def run(stats_by_arm, test, expected=None, match=None):
    return V.verdict_from_stats(stats_by_arm, EXPECTED if expected is None else expected,
                                MATCH_OK if match is None else match, test)


@contextlib.contextmanager
def patched(**attrs):
    old = {k: getattr(V, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(V, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(V, k, v)


GOOD = lambda i, k: S(0.1, 3.0, -3.0)
WEAK = lambda i, k: S(0.1, 1.0, -1.0)


def fx_go():
    return {"G": arm(GOOD, GOOD), "S-match": arm(WEAK, WEAK), "S-ref": arm(WEAK, WEAK)}


def fx_tb(n_testable_b: int):
    """G beats S-match on every (b) pair; exactly n_testable_b (b) pairs are testable in G; F_a = 18."""
    g_b = lambda i, k: S(0.1, 3.0, -3.0) if i < n_testable_b else S(0.1, 1.5, -1.5)
    s_b = lambda i, k: S(0.1, 0.5, -0.5)
    return {"G": arm(GOOD, g_b), "S-match": arm(WEAK, s_b), "S-ref": arm(WEAK, s_b)}


def fx_fa(n_naive_testable_a: int):
    """G bar on (b); on (a) every pair is testable but only n have |d_pre| < 0.5."""
    g_a = lambda i, k: S(0.1 if i < n_naive_testable_a else 0.8, 3.0, -3.0)
    return {"G": arm(g_a, GOOD), "S-match": arm(WEAK, WEAK), "S-ref": arm(WEAK, WEAK)}


def fx_sparsity_identical():
    return {"G": arm(GOOD, GOOD), "S-match": arm(GOOD, GOOD), "S-ref": arm(WEAK, WEAK)}


def fx_sparsity_contrast_only():
    return {"G": arm(WEAK, WEAK), "S-match": arm(GOOD, GOOD), "S-ref": arm(WEAK, WEAK)}


def fx_no_effect():
    return {"G": arm(WEAK, WEAK), "S-match": arm(WEAK, WEAK), "S-ref": arm(WEAK, WEAK)}


def fx_inf_every_turn():
    """First (b) pair of every turn: G m = +inf vs S-match 9.9; the rest: G 0 vs S-match 1.
    Clipped, every turn sums negative (no improvement); unclipped, every turn is +inf."""
    first = {}
    for k in KEYS_B:
        first.setdefault(k[1], k)
    g_b = lambda i, k: S(0.1, np.inf, -np.inf) if first[k[1]] == k else S(0.1, 0.0, 0.0)
    s_b = lambda i, k: S(0.1, 9.9, -9.9) if first[k[1]] == k else S(0.1, 1.0, -1.0)
    return {"G": arm(WEAK, g_b), "S-match": arm(WEAK, s_b), "S-ref": arm(WEAK, WEAK)}


def probe(dv_vals, a0=30.0, p0=40.0):
    """Raw probe whose dv() equals dv_vals exactly: MBON13 carries the X-Y difference, MBON05 is equal."""
    sa = V.Z["A"][1]
    return {"A": [[a0 + d * sa, a0] for d in dv_vals], "P": [[p0, p0] for _ in dv_vals]}


def main() -> int:
    # ---- dprime limits ------------------------------------------------------------------------------------
    check("dprime: 1 seed is undefined", V.dprime([1.0]) is None)
    check("dprime: sd 0, mean 0 -> 0.0", V.dprime([0.0, 0.0, 0.0]) == 0.0)
    check("dprime: sd 0, mean > 0 -> +inf", V.dprime([2.0, 2.0]) == math.inf)
    check("dprime: sd 0, mean < 0 -> -inf", V.dprime([-2.0, -2.0]) == -math.inf)
    check("dprime: ordinary value", abs(V.dprime([1.0, 3.0]) - 2.0 / math.sqrt(2.0)) < 1e-12)

    # ---- pair_stats from raw probes ------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 8)
    pre = 3.0 + noise                                   # large naive offset
    r1 = pre + 0.05 + rng.normal(0, 0.5, 8)             # tiny reward change on top of it
    r2 = r1 - 3.0 + rng.normal(0, 0.1, 8)               # strong punishment drop
    rep = {"pre": probe(pre), "R1": probe(r1), "R2": probe(r2)}
    ps = V.pair_stats(rep)
    check("dv round-trips the fixture probe", np.allclose(V.dv(probe(pre)), pre))
    check("pair_stats: change, not level (offset pair is not testable)", ps is not None and not ps["testable"],
          f"r={ps and ps['r']:.3f}")
    check("pair_stats: level would pass on the same pair", V.dprime(r1) >= 2.0)
    same = {"pre": probe([1.0] * 8), "R1": probe([4.0] * 8), "R2": probe([1.0] * 8)}
    s2 = V.pair_stats(same)
    check("pair_stats: sd-0 change gives +inf / -inf, m = +inf, testable", s2 is not None
          and s2["r"] == math.inf and s2["p"] == -math.inf and s2["m"] == math.inf and s2["testable"])
    flat = {"pre": probe([1.0] * 8), "R1": probe([1.0] * 8), "R2": probe([1.0] * 8)}
    s3 = V.pair_stats(flat)
    check("pair_stats: no change anywhere -> r = p = 0, not testable", s3 is not None and s3["r"] == 0.0
          and s3["p"] == 0.0 and not s3["testable"])
    both_up = {"pre": probe([1.0] * 8), "R1": probe([4.0] * 8), "R2": probe([7.0] * 8)}
    s4 = V.pair_stats(both_up)
    check("pair_stats: punishment moving the wrong way -> m = -inf", s4 is not None and s4["m"] == -math.inf
          and not s4["testable"])
    one = {"pre": probe([1.0]), "R1": probe([2.0]), "R2": probe([0.0])}
    check("pair_stats: 1 seed -> None", V.pair_stats(one) is None)
    try:
        V.pair_stats({"pre": probe([1.0] * 8), "R1": probe([1.0] * 7), "R2": probe([1.0] * 8)})
        check("pair_stats: mismatched seed counts raise", False)
    except ValueError:
        check("pair_stats: mismatched seed counts raise", True)
    # m exactly at the bar, through the module (d' arithmetic from raw floats cannot land on 2.0 exactly)
    bar_rep = {"pre": probe([0.0] * 2), "R1": probe([1.0, 3.0]), "R2": probe([-9.0, -7.0])}   # p = -inf
    at_bar = V.pair_stats(bar_rep)
    check("pair_stats: m = r when -p is larger", at_bar is not None and at_bar["m"] == at_bar["r"],
          str(at_bar))
    with patched(TESTABLE=at_bar["r"]):
        check("pair_stats: m equal to the bar is testable (>=)", V.pair_stats(bar_rep)["testable"])
    with patched(TESTABLE=np.nextafter(at_bar["r"], np.inf)):
        check("pair_stats: m just below the bar is not testable", not V.pair_stats(bar_rep)["testable"])

    # ---- outcomes under both tests -------------------------------------------------------------------------
    for t in TESTS:
        check(f"[{t}] clear improvement + bar -> GO_M0D", run(fx_go(), t)["outcome"] == "GO_M0D")
        check(f"[{t}] T_b 11/21 (0.524) -> GO_M0D", run(fx_tb(11), t)["outcome"] == "GO_M0D")
        check(f"[{t}] T_b 10/21 (0.476) -> PROMISING", run(fx_tb(10), t)["outcome"] == "PROMISING")
        check(f"[{t}] F_a 2 -> GO_M0D", run(fx_fa(2), t)["outcome"] == "GO_M0D")
        check(f"[{t}] F_a 1 -> PROMISING", run(fx_fa(1), t)["outcome"] == "PROMISING")
        r = run(fx_sparsity_identical(), t)
        check(f"[{t}] identical arms with bar -> SPARSITY (no improvement at delta 0)", r["outcome"] == "SPARSITY",
              str(r.get("contrast")))
        check(f"[{t}] only S-match has the bar -> SPARSITY", run(fx_sparsity_contrast_only(), t)["outcome"] == "SPARSITY")
        check(f"[{t}] nothing -> NO_EFFECT", run(fx_no_effect(), t)["outcome"] == "NO_EFFECT")
        r = run(fx_inf_every_turn(), t)
        check(f"[{t}] +inf clipped: no improvement", r["outcome"] == "NO_EFFECT", str(r.get("contrast")))

    b = run(fx_sparsity_identical(), "bootstrap")["contrast"]
    check("bootstrap: CI lower exactly 0 is not an improvement (strict >)", b["lo"] == 0.0 and not b["improve"])
    tiny = V.turn_bootstrap([k[1] for k in KEYS_B], [1e-9] * len(KEYS_B))
    check("bootstrap: any uniformly positive delta improves (documented property)", tiny["improve"])
    check("bootstrap: deterministic", V.turn_bootstrap([0, 0, 2, 2], [1.0, -0.5, 0.3, 0.2]) ==
          V.turn_bootstrap([0, 0, 2, 2], [1.0, -0.5, 0.3, 0.2]))
    sf = V.turn_signflip([k[1] for k in KEYS_B], [1.0] * len(KEYS_B))
    check("signflip: all 8 turns positive -> p = 1/256", abs(sf["p"] - 1 / 256) < 1e-12 and sf["improve"])
    one_turn = [1.0 if k[1] == 0 else 0.0 for k in KEYS_B]
    sf1 = V.turn_signflip([k[1] for k in KEYS_B], one_turn)
    check("signflip: one positive turn, rest 0 -> p = 0.5", abs(sf1["p"] - 0.5) < 1e-12 and not sf1["improve"])
    turns_b = [k[1] for k in KEYS_B]
    seven = [(-1.0 if k[1] == 14 else 1.0) for k in KEYS_B]
    sf7 = V.turn_signflip(turns_b, seven)
    check("signflip: one negative turn among 8 still improves", sf7["improve"], f"p={sf7['p']:.4f}")

    # ---- INVALID ---------------------------------------------------------------------------------------------
    for t in TESTS:
        fx = fx_go(); del fx["S-ref"]
        check(f"[{t}] missing arm -> INVALID", run(fx, t)["outcome"] == "INVALID")
        fx = fx_go(); fx["G"].pop(KEYS_B[4])
        check(f"[{t}] missing pair -> INVALID", run(fx, t)["outcome"] == "INVALID")
        fx = fx_go(); fx["S-match"][("b", 0, "extra", "pair")] = S(0, 1, -1)
        check(f"[{t}] extra pair -> INVALID", run(fx, t)["outcome"] == "INVALID")
        fx = fx_go(); fx["G"][KEYS_A[0]] = None
        check(f"[{t}] undefined d' -> INVALID", run(fx, t)["outcome"] == "INVALID")
        fx = fx_go(); odd = ("b", 3, "xo", "yo")
        for a_ in fx.values():
            a_[odd] = S(0.1, 3.0, -3.0)
        check(f"[{t}] odd turn in pairs -> INVALID", run(fx, t, expected=EXPECTED + [odd])["outcome"] == "INVALID")
        check(f"[{t}] activity match 0.6 pp -> INVALID", run(fx_go(), t, match={"diff_pp": 0.6})["outcome"] == "INVALID")
        check(f"[{t}] activity match -0.5 pp (boundary) -> valid", run(fx_go(), t, match={"diff_pp": -0.5})["outcome"] == "GO_M0D")
        check(f"[{t}] activity match missing -> INVALID", run(fx_go(), t, match={})["outcome"] == "INVALID")
        check(f"[{t}] activity match NaN -> INVALID", run(fx_go(), t, match={"diff_pp": float('nan')})["outcome"] == "INVALID")
        fx = fx_go(); fx["G"][KEYS_B[0]] = S(0.1, float("nan"), -3.0)
        check(f"[{t}] NaN pair statistic -> INVALID", run(fx, t)["outcome"] == "INVALID")
    try:
        run(fx_go(), "ttest")
        check("unknown test name raises", False)
    except ValueError:
        check("unknown test name raises", True)
    row = {"axis": "a", "turn": 0, "x": "x", "y": "y", "report": flat}
    dup = V.verdict({"G": [row, dict(row)], "S-match": [row], "S-ref": [row]}, [("a", 0, "x", "y")], MATCH_OK, "bootstrap")
    check("raw: duplicate pair rows -> INVALID", dup["outcome"] == "INVALID")

    # ---- end to end from raw probes --------------------------------------------------------------------------
    def raw_rows(testable: bool):
        rows = []
        for (ax, t, x, y) in EXPECTED:
            base = np.linspace(-0.1, 0.1, 8)
            up = 4.0 if testable else 0.5
            rows.append({"axis": ax, "turn": t, "x": x, "y": y,
                         "report": {"pre": probe(base), "R1": probe(base + up + np.linspace(0, 0.4, 8)),
                                    "R2": probe(base + np.linspace(0, 0.4, 8) - 0.2 * np.arange(8) / 8)}})
        return rows
    raw_good, raw_weak = raw_rows(True), raw_rows(False)
    e2e = V.verdict({"G": raw_good, "S-match": raw_weak, "S-ref": raw_weak}, EXPECTED, MATCH_OK, "bootstrap")
    check("raw end-to-end: G testable everywhere, S-match not -> GO_M0D", e2e["outcome"] == "GO_M0D",
          str(e2e.get("arms", {}).get("G")))

    # ---- mutations: each must change at least one verdict ---------------------------------------------------
    def outcome_of(fx, t="bootstrap", **kw):
        return run(fx, t, **kw)["outcome"]

    with patched(TEST_ARM="S-match", CONTRAST_ARM="G"):
        check("mutation: swapped arms changes GO_M0D", outcome_of(fx_go()) != "GO_M0D")
    with patched(CLIP=np.inf):
        check("mutation: no clipping turns +inf into an improvement (bootstrap)",
              outcome_of(fx_inf_every_turn()) != "NO_EFFECT")
        check("mutation: no clipping turns +inf into an improvement (signflip)",
              outcome_of(fx_inf_every_turn(), "signflip") != "NO_EFFECT")
    with patched(reward_stat=lambda pre_, r1_: V.dprime(r1_)):
        m_ps = V.pair_stats(rep)
        check("mutation: level instead of change makes the offset pair testable", m_ps is not None and m_ps["testable"])
    with patched(naive_ok=lambda d: True):
        check("mutation: dropping the naive filter changes F_a 1 -> GO_M0D", outcome_of(fx_fa(1)) == "GO_M0D")
    with patched(turn_ok=lambda t: True):
        fx = fx_go(); odd = ("b", 3, "xo", "yo")
        for a_ in fx.values():
            a_[odd] = S(0.1, 3.0, -3.0)
        check("mutation: accepting odd turns lets the odd-turn fixture through",
              outcome_of(fx, expected=EXPECTED + [odd]) != "INVALID")
    with patched(MATCH_TOL_PP=np.inf):
        check("mutation: no match tolerance accepts a 0.6 pp mismatch",
              outcome_of(fx_go(), match={"diff_pp": 0.6}) != "INVALID")
    with patched(BAR_F_A=0):
        check("mutation: F_a bar removed changes F_a 1", outcome_of(fx_fa(0)) == "GO_M0D")

    width = max(len(n) for n, *_ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail if not ok else ''}".rstrip())
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
