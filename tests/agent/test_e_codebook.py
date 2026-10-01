import itertools
import math
import numpy as np
import pytest
from flymon.agent import e_codebook as cb
from flymon.agent.e_spec import SPEC, smoke
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, MON, MOVE = pool_vocabulary()
CELLS = cb.cells(MOVE, MON)
DUALS = cb.dual_set(ST)
ADJ = cb.conflict_graph(CELLS, DUALS)
GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
DRIVE = {g: float(10 + 50 * i) for i, g in enumerate(GLOMS)}

def test_cells_and_duals():
    assert len(CELLS) == 96 and CELLS[0] == (MOVE[0], MON[0])
    assert len(DUALS) == 10 and frozenset({"GRASS", "POISON"}) in DUALS

def test_hard_conflict_rules():
    m1, m2 = MOVE[0], MOVE[1]
    assert cb.hard_conflict((m1, "FIRE"), (m1, "ICE"), DUALS)            # same row
    assert cb.hard_conflict((m1, "FIRE"), (m2, "FIRE"), DUALS)           # same column
    assert cb.hard_conflict((m1, "GRASS"), (m2, "POISON"), DUALS)        # dual cross term
    assert not cb.hard_conflict((m1, "FIRE"), (m2, "ICE"), DUALS)

def test_clique_is_16():
    assert cb.max_clique_size(ADJ) == 16

def test_dsatur_valid_and_colouring_found():
    seed, col = cb.find_colouring(ADJ, 16, range(SPEC.dsatur_tie_seeds))
    assert seed is not None and max(col) + 1 == 16
    assert all(col[i] != col[j] for i in range(96) for j in ADJ[i])

def test_construct_has_no_hard_violation():
    _, col = cb.find_colouring(ADJ, 16, range(1000))
    alpha = cb.alphabet(DRIVE, 3, SPEC)
    book = cb.construct(col, alpha, 3)
    assert cb.hard_violations(book, ADJ) == 0 and all(len(w) == 3 for w in book)

def test_alphabet_k2_band_and_order():
    drive = {**DRIVE, "ORN_DA1": 1131.0, "ORN_V": 1131.0}
    a2 = cb.alphabet(drive, 2, SPEC)
    assert len(a2) == 40 and "ORN_DA1" not in a2 and "ORN_V" not in a2
    target = np.log10(np.sqrt(400 * 3200) + 1)
    far = max(abs(np.log10(drive[g] + 1) - target) for g in a2)
    out = [g for g in GLOMS if g not in a2]
    assert all(abs(np.log10(drive[g] + 1) - target) >= far for g in out)
    a3 = cb.alphabet(drive, 3, SPEC)
    assert len(a3) == 49 and a3 == sorted(a3, key=lambda g: (-drive[g], g))

def test_anneal_keeps_hard_zero_and_removes_duplicates():
    sp = smoke(SPEC)
    res = cb.build(DRIVE, 2, sp)
    assert res["status"] == "OK"
    book = [tuple(w) for w in res["codebook"]]
    assert cb.hard_violations(book, ADJ) == 0
    assert cb.dup_pairs(book) == 0 and len(set(book)) == 96

def test_build_is_deterministic():
    sp = smoke(SPEC)
    assert cb.build(DRIVE, 2, sp)["digest"] == cb.build(DRIVE, 2, sp)["digest"]

def test_infeasible_when_alphabet_too_small():
    small = {g: DRIVE[g] for g in GLOMS[:40]}
    res = cb.build(small, 3, smoke(SPEC))         # omega*k = 48 > 40
    assert res["status"] == "INFEASIBLE"

def test_not_unique_status():
    res = cb.build(DRIVE, 2, smoke(SPEC), unique_check=lambda book: False)
    assert res["status"] == "NOT_UNIQUE"


# --- incremental objective (brief Task 2 performance clause) -------------------------------------------------------

def _start(k):
    _, col = cb.find_colouring(ADJ, 16, range(SPEC.dsatur_tie_seeds))
    alpha = cb.alphabet(DRIVE, k, SPEC)
    return cb.construct(col, alpha, k), alpha


@pytest.mark.parametrize("k", [2, 3])
def test_incremental_objective_equals_full_after_1000_moves(k):
    start, alpha = _start(k)
    st = cb._AnnealState(start, ADJ, alpha, DRIVE, SPEC)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        i = int(rng.integers(96)); w = st.book[i]
        out_g = w[int(rng.integers(k))]
        choices = [g for g in alpha if g not in w]
        in_g = choices[int(rng.integers(len(choices)))]
        J2 = st.propose(i, out_g, in_g)        # moves are applied regardless of hard violations here
        st.accept()
        full = cb.objective([tuple(x) for x in st.book], ADJ, DRIVE, SPEC)
        assert (st.J, st.d, st.s, st.v) == full and J2 == full[0]


def _anneal_full_reference(start, adj, alpha, drive, k, spec):
    """The brief's full-recompute anneal, kept verbatim as the reference."""
    best = None
    for r in range(spec.anneal_restarts):
        rng = np.random.default_rng(spec.anneal_seed0 + 1000 * k + r)
        book = [list(w) for w in start]
        J, *_ = cb.objective([tuple(w) for w in book], adj, drive, spec)
        n = len(book)
        for it in range(spec.anneal_iters):
            T = spec.t_start + (spec.t_end - spec.t_start) * it / max(1, spec.anneal_iters - 1)
            i = int(rng.integers(n)); old = book[i][:]
            out_g = old[int(rng.integers(k))]
            choices = [g for g in alpha if g not in old]
            in_g = choices[int(rng.integers(len(choices)))]
            new = sorted([g for g in old if g != out_g] + [in_g])
            if any(in_g in book[j] for j in adj[i]):
                continue
            book[i] = new
            J2, *_ = cb.objective([tuple(w) for w in book], adj, drive, spec)
            if J2 <= J or rng.random() < math.exp(-(J2 - J) / T):
                J = J2
            else:
                book[i] = old
        J, d, s, v = cb.objective([tuple(w) for w in book], adj, drive, spec)
        if best is None or J < best["J"]:
            best = dict(codebook=[tuple(w) for w in book], J=J, restart=r, dup=d, soft=s, logvar=v)
    return best


@pytest.mark.parametrize("k", [2, 3])
def test_incremental_anneal_matches_full_reference(k):
    import dataclasses
    sp = dataclasses.replace(SPEC, anneal_iters=400, anneal_restarts=2)
    start, alpha = _start(k)
    assert cb.anneal(start, ADJ, alpha, DRIVE, k, sp) == _anneal_full_reference(start, ADJ, alpha, DRIVE, k, sp)


def test_alphabet_rejects_other_k():
    with pytest.raises(ValueError):
        cb.alphabet(DRIVE, 4, SPEC)
