# tests/brain/test_o_spec.py
"""Spec O.2 / O.3 / O.5 / O.7: O's numbers live in o_spec, N's are read through spec.n (never restated); the seed blocks
22_000_000 + i (i < 64), 22_001_000 + i (i < 32) and smoke 22_009_xxx are disjoint from each other and from N's; O2's
training seeds (n_spec.train_seed) collide with no declared block and no N training seed; smoke changes scale only."""
from dataclasses import fields

from flymon.brain import n_spec
from flymon.brain.o_spec import SPEC, OSpec, smoke

SM = smoke(SPEC)
N = SPEC.n


def test_seed_blocks_are_the_declared_ones():
    assert SPEC.o1_seeds == tuple(range(22_000_000, 22_000_064))
    assert SPEC.o2_seeds == tuple(range(22_001_000, 22_001_032))
    for s in SM.o1_seeds + SM.o2_seeds:
        assert 22_009_000 <= s < 22_010_000
    blocks = [set(SPEC.o1_seeds), set(SPEC.o2_seeds), set(SM.o1_seeds), set(SM.o2_seeds)]
    assert sum(len(b) for b in blocks) == len(set().union(*blocks))


def _n_declared_blocks():
    nmax = max(N.n_grid)
    return (set(N.n0f_seeds) | set(N.act_seeds) | set(N.select_seeds) | set(N.report_seeds) | set(N.pilot_seeds)
            | set(N.judge_seeds(nmax)))


def test_o_blocks_avoid_n_blocks():
    o = set(SPEC.o1_seeds) | set(SPEC.o2_seeds) | set(SM.o1_seeds) | set(SM.o2_seeds)
    assert not o & _n_declared_blocks()


def test_training_seeds_collide_with_no_declared_block():
    probes = (set(SPEC.o1_seeds) | set(SPEC.o2_seeds) | set(SM.o1_seeds) | set(SM.o2_seeds)
              | _n_declared_blocks())
    n_train = {N.train_seed(s, t) for s in set(N.pilot_seeds) | set(N.judge_seeds(max(N.n_grid)))
               for t in range(int(N.h4.teach_trials))}
    for sp in (SPEC, SM):
        tr = sp.o2_train_seeds()
        assert len(tr) == len(set(tr)) == len(sp.o2_seeds) * int(N.h4.teach_trials)
        assert not set(tr) & probes and not set(tr) & n_train
        assert tr[0] == n_spec.train_seed(sp.o2_seeds[0], 0, N.train_seed_base, N.train_seed_stride)
    assert SPEC.o2_train_seeds()[0] == 1_000_000 + 22_001_000 * 1000          # train_block's rule, exact int
    assert not set(SM.o2_train_seeds()) & set(SPEC.o2_train_seeds())


def test_counts_match_o7():
    assert SPEC.n_o1_presentations() == 7680                                    # O.7.1: 4 x 6 x 5 x 64
    assert SPEC.n_o2_arms() == 256                                              # O.7.7: 2 x 4 x 32


def test_n_numbers_are_read_not_restated():
    assert SPEC.o1_g_grid is N.g_grid and list(SPEC.o1_g_grid) == sorted(SPEC.o1_g_grid)
    assert SPEC.o1_stimuli is N.n0f_stimuli
    assert SPEC.mid_band[0] == N.state_p_min
    assert (SPEC.boot_draws, SPEC.boot_seed) == (N.boot_draws, N.boot_seed)
    assert SPEC.spec_path == N.spec_path and SPEC.workers == N.workers


def test_conditions_arms_and_pairs():
    assert SPEC.o1_conditions == (("on", "none"), ("kc", "apl_to_kc_zero"), ("nonkc", "apl_to_nonkc_zero"),
                                  ("all", "apl_all_zero"))
    assert SPEC.on_edit == "none"
    assert SPEC.o2_arms == (("plastic", False, True, False), ("frozen", False, False, False),
                            ("punish", True, True, False), ("da_zero", False, True, True))
    assert SPEC.o2_pairs == (("4:1", "1:4", "sim"), ("dDL", "4:1", "dis"))
    assert dict(SPEC.oracle_o) == {"sim": -2.348, "dis": -2.433}
    assert SPEC.o2_point == (0.25, 8.0) and SPEC.o1_c_delta == 8.0
    assert set(dict(SPEC.oracle_o)) == {p for _, _, p in SPEC.o2_pairs}


def test_smoke_changes_scale_only():
    scale = {"o1_g_grid", "o1_seed0", "o1_n_seeds", "o2_seed0", "o2_n_seeds", "boot_draws"}
    for f in fields(OSpec):
        if f.name not in scale:
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert len(SM.o1_g_grid) == 2 and set(SM.o1_g_grid) <= set(SPEC.o1_g_grid)
    assert SM.boot_draws == SPEC.smoke_boot_draws


def test_ci_level_is_read_from_n_and_not_restated():
    from pathlib import Path
    assert SPEC.ci_level == N.ci_level
    src = (Path(__file__).resolve().parents[2] / "flymon" / "brain" / "o_spec.py").read_text()
    assert "0.95" not in src
