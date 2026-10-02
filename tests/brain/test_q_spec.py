# tests/brain/test_q_spec.py
"""Spec Q.3 / Q.6: every Q number in q_spec, encoder numbers read from e_spec (not restated); Q's seed blocks
24_000_000+i / 24_000_100+i / 24_000_200+i (i < 8) and smoke 24_008_000-24_008_099 collide with no declared seed and no
declared seed's training seeds (P.6.6's form, test_p_spec's collector, e_spec.py included); no Q code path names the L
judgement set."""
import ast
import dataclasses
import importlib
from pathlib import Path

from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.q_spec import SPEC, Condition, QSpec, smoke
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_q() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/q_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


Q_SEEDS = set(SPEC.act_seeds) | set(SPEC.select_seeds) | set(SPEC.report_seeds) | set(SPEC.smoke_seeds)


def test_q_blocks_are_the_declared_ones():
    assert SPEC.act_seeds == tuple(range(24_000_000, 24_000_008))
    assert SPEC.select_seeds == tuple(range(24_000_100, 24_000_108))
    assert SPEC.report_seeds == tuple(range(24_000_200, 24_000_208))
    assert SPEC.smoke_seeds == tuple(range(24_008_000, 24_008_100))
    sm = set(SM.act_seeds) | set(SM.select_seeds) | set(SM.report_seeds) | set(SM.kc_probe_seeds)
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 11
    assert SPEC.repro_seeds() == dict(act=list(E.even_act_seeds), select=list(E.even_select_seeds),
                                      report=list(E.even_report_seeds))


def test_e_spec_is_collected_and_q_spec_enumerated():
    assert "flymon/agent/e_spec.py" in MODULES and "flymon/brain/q_spec.py" in MODULES


def test_q_seeds_collide_with_nothing_declared():
    declared = _declared_without_q()
    for s in (500, 615, 24_001_000, 24_002_000, 24_009_000, 24_100_000, 24_100_207, 23_000_000):
        assert s in declared, s
    assert not Q_SEEDS & declared
    assert not {q for q in Q_SEEDS if (q - BASE) // STRIDE in declared}     # no declared s trains on a Q seed
    assert 23_000 not in declared and 23_008 not in declared


def test_encoder_numbers_are_read_not_restated():
    assert (SPEC.settle_ms, SPEC.read_ms, SPEC.window_ms) == (E.settle_ms, E.read_ms, E.window_ms)
    assert SPEC.alphas == E.alphas == (0.2, 0.5, 0.8) and SPEC.n_b == E.n_b == 21
    assert SPEC.split_r == E.testable_min == 2.0 and SPEC.m0d_summary == E.m0d_summary
    assert (SPEC.punish_type, SPEC.reward_type) == ("PPL105", "PAM08")


def test_conditions_in_order_and_flags():
    cs = SPEC.conditions(1.3, 0.8, 1.25)
    assert tuple(c.name for c in cs) == SPEC.cond_names == ("base", "apl_mbon05", "apl_nonkc", "s_up", "mv_lo", "mv_hi")
    d = {c.name: c for c in cs}
    assert d["base"] == Condition("base", "none", 1.0, 1.0, "baseline", False)
    assert d["apl_mbon05"].edit == "apl_to_mbon05_zero" and d["apl_mbon05"].role == "primary"
    assert d["apl_nonkc"].edit == "apl_to_nonkc_zero" and d["apl_nonkc"].role == "record"
    assert d["s_up"].strength == 1.3 and d["s_up"].edit == "none"
    assert (d["mv_lo"].mv_scale, d["mv_hi"].mv_scale) == (0.8, 1.25)
    assert all(c.kc_gate for n, c in d.items() if n != "base")


def test_smoke_changes_scale_only():
    keep = {"act_seeds", "select_seeds", "report_seeds", "kc_probe_seeds", "pairs_subset", "smoke", "workers"}
    for f in dataclasses.fields(QSpec):
        if f.name not in keep:
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.pairs_subset == SPEC.smoke_pairs and len(SM.report_seeds) >= 3


ALLOWED_Q_SPEC = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 20, 0.03, 0.05, 0.15, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0,
                  1.1, 1.25, 1.5, 1.89, 1.97, 2.0, 5.0, 3600.0, 24_000_000, 24_000_008, 24_000_100, 24_000_108,
                  24_000_200, 24_000_208, 24_008_000, 24_008_100}


def test_q_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/q_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_Q_SPEC, nums - ALLOWED_Q_SPEC


FORBIDDEN = {"judgement_set", "used_situations", "new_turns", "alternate_from", "l_pairs", "judge_act_seeds",
             "judge_select_seeds", "judge_report_seeds", "judge_first_turn", "judge_last_turn", "l_rng_seed",
             "l_total_turns"}


def test_no_q_code_path_names_the_l_judgement_set():
    files = sorted((ROOT / "flymon/brain").glob("q_*.py")) + sorted((ROOT / "scripts").glob("run_q*.py"))
    assert files
    for p in files:
        tree = ast.parse(p.read_text())
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names |= {a.name.split(".")[-1] for a in n.names}
                if isinstance(n, ast.ImportFrom) and n.module:
                    names |= set(n.module.split("."))
        assert not names & FORBIDDEN, (p.name, names & FORBIDDEN)
        assert "24_100" not in p.read_text() and "24100" not in p.read_text(), p.name


def test_q69_amendments_encoded():
    assert SPEC.s_down == (0.7, 0.8) and SPEC.s_weak == 0.15 and SPEC.rho_min == 0.4
    assert SPEC.rho_positive_only and SPEC.change_two_sided
    assert SPEC.conditions(0.7, 0.8, 1.25)[3].strength == 0.7
