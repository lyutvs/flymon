"""Spec 3.1-3.3: the 96-cell glomerulus-disjoint conjunctive codebook. Pure (no engine, no pops)."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter

import numpy as np


def cells(move_types, mon_types) -> list:
    return [(m, t) for m in move_types for t in mon_types]


def dual_set(species_types) -> set:
    return {frozenset(v) for v in species_types.values() if len(v) == 2}


def hard_conflict(a, b, duals) -> bool:
    (m1, t1), (m2, t2) = a, b
    return m1 == m2 or t1 == t2 or frozenset((t1, t2)) in duals


def conflict_graph(cs, duals) -> list:
    adj = [set() for _ in cs]
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if hard_conflict(cs[i], cs[j], duals):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def max_clique_size(adj) -> int:
    """Bron-Kerbosch with pivoting (96 vertices: fast enough)."""
    best = 0

    def bk(r, p, x):
        nonlocal best
        if not p and not x:
            best = max(best, r)
            return
        if r + len(p) <= best:
            return
        u = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[u]):
            bk(r + 1, p & adj[v], x & adj[v])
            p = p - {v}
            x = x | {v}

    bk(0, set(range(len(adj))), set())
    return best


def dsatur(adj, rng) -> list:
    """DSATUR: max saturation, then max degree, then the earliest position in a random permutation (spec 3.3)."""
    n = len(adj)
    col = [-1] * n
    tie = rng.permutation(n)
    rank = {int(v): p for p, v in enumerate(tie)}
    for _ in range(n):
        cand = [v for v in range(n) if col[v] < 0]
        sat = {v: len({col[u] for u in adj[v] if col[u] >= 0}) for v in cand}
        v = max(cand, key=lambda v: (sat[v], len(adj[v]), -rank[v]))
        used = {col[u] for u in adj[v] if col[u] >= 0}
        col[v] = next(c for c in range(n) if c not in used)
    return col


def find_colouring(adj, target: int, seeds):
    """First tie seed t (in the given order) whose DSATUR colouring uses <= target colours."""
    for s in seeds:
        col = dsatur(adj, np.random.default_rng(int(s)))
        if max(col) + 1 <= target:
            return int(s), col
    return None, None


def _lg(x) -> float:
    return math.log10(float(x) + 1.0)


def alphabet(drive: dict, k: int, spec) -> list:
    gl = [g for g in drive if g not in spec.exclude]

    def order(gs):
        return sorted(gs, key=lambda g: (-drive[g], g))

    if k == 3:
        return order(gl)
    target = _lg(math.sqrt(spec.band_lo * spec.band_hi))
    near = sorted(gl, key=lambda g: (abs(_lg(drive[g]) - target), g))[:spec.k2_alphabet]
    return order(near)


def construct(colours, alpha, k) -> list:
    return [tuple(sorted(alpha[c * k + j] for j in range(k))) for c in colours]


def hard_violations(book, adj) -> int:
    return sum(len(set(book[i]) & set(book[j])) for i in range(len(book)) for j in adj[i] if j > i)


def soft_violations(book, adj) -> int:
    n = len(book)
    return sum(max(0, len(set(book[i]) & set(book[j])) - 1) for i in range(n) for j in range(i + 1, n)
               if j not in adj[i])


def dup_pairs(book) -> int:
    return sum(c * (c - 1) // 2 for c in Counter(tuple(sorted(w)) for w in book).values())


def logvar(book, drive) -> float:
    return float(np.var([_lg(sum(drive[g] for g in w)) for w in book]))


def objective(book, adj, drive, spec) -> tuple:
    d, s, v = dup_pairs(book), soft_violations(book, adj), logvar(book, drive)
    return spec.w_dup * d + spec.w_soft * s + v, d, s, v


def exact_search(adj, alpha, k, budget):
    """Backtracking over cells in degree order, each cell picking a k-subset disjoint from its conflicting
    neighbours' glomeruli. Returns a codebook, "INFEASIBLE" (search exhausted) or None (node budget spent)."""
    n = len(adj)
    book = [None] * n
    nodes = 0
    order = sorted(range(n), key=lambda v: -len(adj[v]))

    def rec(i):
        nonlocal nodes
        if i == n:
            return True
        v = order[i]
        banned = set().union(*(set(book[u]) for u in adj[v] if book[u] is not None)) if adj[v] else set()
        for combo in itertools.combinations([g for g in alpha if g not in banned], k):
            nodes += 1
            if nodes > budget:
                raise TimeoutError
            book[v] = tuple(sorted(combo))
            if rec(i + 1):
                return True
            book[v] = None
        return False

    try:
        return list(book) if rec(0) else "INFEASIBLE"
    except TimeoutError:
        return None


class _AnnealState:
    """Incremental objective: dup via a codeword Counter, soft via a pairwise overlap table touched only on the
    moved cell's row, logvar via np.var over a per-cell log-drive array (bit-identical to `objective`)."""

    def __init__(self, start, adj, alpha, drive, spec):
        self.spec, self.drive = spec, drive
        n = len(start)
        self.book = [sorted(w) for w in start]
        self.sets = [set(w) for w in self.book]
        self.nonadj = [set(range(n)) - adj[i] - {i} for i in range(n)]
        self.holders = {g: set() for g in alpha}
        for i, w in enumerate(self.book):
            for g in w:
                self.holders.setdefault(g, set()).add(i)
        self.ov = [[len(self.sets[i] & self.sets[j]) for j in range(n)] for i in range(n)]
        self.count = Counter(tuple(w) for w in self.book)
        self.lv = np.array([_lg(sum(drive[g] for g in w)) for w in self.book], dtype=float)
        self.d = sum(c * (c - 1) // 2 for c in self.count.values())
        self.s = sum(max(0, self.ov[i][j] - 1) for i in range(n) for j in self.nonadj[i] if j > i)
        self.v = float(np.var(self.lv))
        self.J = spec.w_dup * self.d + spec.w_soft * self.s + self.v
        self._pending = None

    def propose(self, i, out_g, in_g) -> float:
        old = self.book[i]
        new = sorted([g for g in old if g != out_g] + [in_g])
        to, tn = tuple(old), tuple(new)
        d2 = self.d - (self.count[to] - 1) + self.count[tn]
        ds, changes = 0, []
        for j in (self.holders.get(out_g, set()) | self.holders.get(in_g, set())) & self.nonadj[i]:
            o = self.ov[i][j]
            o2 = o - (out_g in self.sets[j]) + (in_g in self.sets[j])
            if o2 != o:
                ds += max(0, o2 - 1) - max(0, o - 1)
                changes.append((j, o2))
        s2 = self.s + ds
        lv_i = _lg(sum(self.drive[g] for g in new))
        saved = self.lv[i]
        self.lv[i] = lv_i
        v2 = float(np.var(self.lv))
        self.lv[i] = saved
        J2 = self.spec.w_dup * d2 + self.spec.w_soft * s2 + v2
        self._pending = (i, out_g, in_g, new, to, tn, changes, lv_i, d2, s2, v2, J2)
        return J2

    def accept(self):
        i, out_g, in_g, new, to, tn, changes, lv_i, d2, s2, v2, J2 = self._pending
        self.book[i] = new
        self.sets[i] = set(new)
        self.holders[out_g].discard(i)
        self.holders.setdefault(in_g, set()).add(i)
        for j, o2 in changes:
            self.ov[i][j] = self.ov[j][i] = o2
        self.count[to] -= 1
        if not self.count[to]:
            del self.count[to]
        self.count[tn] += 1
        self.lv[i] = lv_i
        self.d, self.s, self.v, self.J = d2, s2, v2, J2
        self._pending = None


def anneal(start, adj, alpha, drive, k, spec) -> dict:
    """Moves swap one glomerulus of one cell for an alphabet glomerulus not in it; moves creating a hard violation
    are rejected. Same accept rule and RNG call order as a full recompute; the objective is updated incrementally."""
    best = None
    for r in range(spec.anneal_restarts):
        rng = np.random.default_rng(spec.anneal_seed0 + 1000 * k + r)
        st = _AnnealState(start, adj, alpha, drive, spec)
        n = len(st.book)
        for it in range(spec.anneal_iters):
            T = spec.t_start + (spec.t_end - spec.t_start) * it / max(1, spec.anneal_iters - 1)
            i = int(rng.integers(n))
            old = st.book[i]
            out_g = old[int(rng.integers(k))]
            choices = [g for g in alpha if g not in old]
            in_g = choices[int(rng.integers(len(choices)))]
            if any(in_g in st.sets[j] for j in adj[i]):
                continue
            J = st.J
            J2 = st.propose(i, out_g, in_g)
            if J2 <= J or rng.random() < math.exp(-(J2 - J) / T):
                st.accept()
        book = [tuple(w) for w in st.book]
        J, d, s, v = objective(book, adj, drive, spec)
        if best is None or J < best["J"]:
            best = dict(codebook=book, J=J, restart=r, dup=d, soft=s, logvar=v)
    return best


def digest(book) -> str:
    return hashlib.sha256(json.dumps([list(w) for w in book], separators=(",", ":")).encode()).hexdigest()


def build(drive: dict, k: int, spec, unique_check=None) -> dict:
    """Spec 3.3 end to end. `unique_check(codebook) -> bool` is injected (Task 3: 112-odour uniqueness); the default
    checks only that the 96 codewords are distinct."""
    from ..brain.h4_pairs import pool_vocabulary
    st, _, mon, move = pool_vocabulary()
    cs = cells(move, mon)
    adj = conflict_graph(cs, dual_set(st))
    omega = max_clique_size(adj)
    alpha = alphabet(drive, k, spec)
    out = dict(k=k, omega=omega, alphabet=alpha, colour_seed=None, colouring=None, C=None)
    if omega * k > len(alpha):
        return dict(out, status="INFEASIBLE", codebook=None, anneal=None, digest=None)
    seed, col = find_colouring(adj, omega, range(spec.dsatur_tie_seeds))      # spec 3.3: first C = omega
    if col is not None:
        start = construct(col, alpha, k)
        out.update(colour_seed=seed, colouring=col, C=max(col) + 1)
    else:
        start = exact_search(adj, alpha, k, spec.exact_node_budget)
        if start == "INFEASIBLE":
            return dict(out, status="INFEASIBLE", codebook=None, anneal=None, digest=None)
        if start is None:
            return dict(out, status="UNDECIDED", codebook=None, anneal=None, digest=None)
    an = anneal(start, adj, alpha, drive, k, spec)
    book = an["codebook"]
    assert hard_violations(book, adj) == 0
    uniq = (unique_check or (lambda b: dup_pairs(b) == 0))(book)
    status = "OK" if (dup_pairs(book) == 0 and uniq) else "NOT_UNIQUE"
    return dict(out, status=status, codebook=[list(w) for w in book],
                anneal={kk: an[kk] for kk in ("J", "restart", "dup", "soft", "logvar")}, digest=digest(book))
