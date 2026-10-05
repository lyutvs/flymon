"""X's set verdict (X.3 as amended by X.9.1.3; Q1 + Q2) over W's pair verdict (w_verdict, unchanged: fly gates, q,
2K BAND, mechanism control, RN1 = R1). n = judgeable gate pairs (final PASS or FAIL), m = PASS pairs, b = BAND left
(no 2K data); a machine reason or an INVALID pair → STOP_MACHINE first, so n + b = k.
- PASS: n ≥ 4 ∧ m ≥ 3 ∧ m / n ≥ p_set ∧ m / (n + b) ≥ p_set.
- FAIL: n ≥ 4 ∧ [(m + b) / (n + b) < p_set ∨ (m + b) < 3].
- otherwise UNDECIDED — "판정 가능 쌍 < 4" when n < 4, else "BAND 잔존".
Comparisons round(x − p_set, 9) (digits from the spec). With b = 0 and n ≥ 4 PASS and FAIL are complements; p_set 1.0
equals W's overall_code wherever n ≥ 4. Vectorised over leading axes; p_set broadcasts against them. The OC simulates
this code itself (x_oc.set_hits)."""
from __future__ import annotations

import numpy as np

from . import w_verdict as WV

S_STOP_MACHINE, S_FAIL, S_UNDECIDED, S_PASS = WV.V_STOP_MACHINE, WV.V_FAIL, WV.V_UNDECIDED, WV.V_PASS
SET_LABEL = dict(WV.VERDICT_LABEL)


def counts(final):
    """(n, m, b, invalid) over the pair axis (-1) of final pair codes."""
    f = np.asarray(final)
    m = (f == WV.P_PASS).sum(-1)
    return m + (f == WV.P_FAIL).sum(-1), m, (f == WV.P_BAND).sum(-1), (f == WV.P_INVALID).any(-1)


def _ratio(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def set_code_counts(n, m, b, stop, p_set, xs):
    n, m, b = (np.asarray(v, float) for v in (n, m, b))
    p, d = np.asarray(p_set, float), xs.round_digits
    nb = n + b
    enough = n >= xs.min_gate_pairs
    ok = (enough & (m >= xs.min_pass_pairs) & (np.round(_ratio(m, n) - p, d) >= 0)
          & (np.round(_ratio(m, nb) - p, d) >= 0))
    bad = enough & ((np.round(_ratio(m + b, nb) - p, d) < 0) | (m + b < xs.min_pass_pairs))
    return np.where(np.asarray(stop, bool), S_STOP_MACHINE, np.where(ok, S_PASS, np.where(bad, S_FAIL, S_UNDECIDED)))


def set_code(final, machine, p_set, xs):
    n, m, b, inv = counts(final)
    return set_code_counts(n, m, b, np.asarray(machine, bool) | inv, p_set, xs)


def undecided_cause(n, xs) -> str:
    return f"판정 가능 쌍 < {xs.min_gate_pairs}" if n < xs.min_gate_pairs else "BAND 잔존"


def m_needed(k: int, p_set: float, xs):
    """The smallest m that PASSes with n = k, b = 0 (None: none)."""
    for m in range(k + 1):
        if int(set_code_counts(k, m, 0, False, p_set, xs)) == S_PASS:
            return m
    return None


def m_needed_table(xs) -> dict:
    """X.9.1.3 P1-3: m needed per (p_set, k) and the per-k duplicate cells (records only — selection unchanged)."""
    ks = list(range(xs.k_min, xs.k_cap + 1))
    table = {str(p): [m_needed(k, p, xs) for k in ks] for p in xs.p_set_grid}
    dup = {}
    for j, k in enumerate(ks):
        groups = {}
        for p in xs.p_set_grid:
            groups.setdefault(table[str(p)][j], []).append(p)
        dup[str(k)] = [g for g in groups.values() if len(g) > 1]
    return dict(k=ks, m=table, duplicates=dup, note="중복(기록용) — 선택 규칙은 바꾸지 않는다(X.9.1.3 P1-3)")
