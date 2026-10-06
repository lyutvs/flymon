# tests/brain/test_p_spec.py
"""Spec P.2 / P.6.6 / P.6.7: p_spec derives from OSpec by dataclasses.replace (only the pairs, the arms and the seed
blocks differ) and restates no O or N number; the two directions of N's dissimilar pair; arms 1-3 with O's flags; seed
block 23_000_000 + i (i < 32), smoke 23_009_xxx. Every seed any *_spec.py in flymon/ declares, and every training seed
1_000_000 + s·1000 + t of those, is disjoint from P's blocks and P's training seeds — in particular no declared s is
22_000 (its training seeds are 23_000_000 + t) or 22_009."""
import ast
import dataclasses
import importlib
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.o_spec import SPEC as O_SPEC, OSpec
from flymon.brain.p_spec import SPEC, PSpec, smoke

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
N = O_SPEC.n

# every spec module of the repository but p_spec itself; a new *_spec.py must be added here (test below)
MODULES = {"flymon/brain/b_spec.py": "flymon.brain.b_spec", "flymon/brain/h3_spec.py": "flymon.brain.h3_spec",
           "flymon/brain/h4_spec.py": "flymon.brain.h4_spec", "flymon/brain/j_spec.py": "flymon.brain.j_spec",
           "flymon/brain/k_spec.py": "flymon.brain.k_spec", "flymon/brain/l_spec.py": "flymon.brain.l_spec",
           "flymon/brain/m_spec.py": "flymon.brain.m_spec", "flymon/brain/n_spec.py": "flymon.brain.n_spec",
           "flymon/brain/o_spec.py": "flymon.brain.o_spec", "flymon/rescope/spec.py": "flymon.rescope.spec",
           "flymon/agent/e_spec.py": "flymon.agent.e_spec", "flymon/brain/q_spec.py": "flymon.brain.q_spec",
           "flymon/brain/r_spec.py": "flymon.brain.r_spec", "flymon/brain/s_spec.py": "flymon.brain.s_spec",
           "flymon/brain/t_spec.py": "flymon.brain.t_spec", "flymon/brain/u_spec.py": "flymon.brain.u_spec",
           "flymon/brain/v_spec.py": "flymon.brain.v_spec", "flymon/brain/w_spec.py": "flymon.brain.w_spec",
           "flymon/brain/x_spec.py": "flymon.brain.x_spec", "flymon/brain/y_spec.py": "flymon.brain.y_spec",
           "flymon/brain/z_spec.py": "flymon.brain.z_spec", "flymon/brain/aa_spec.py": "flymon.brain.aa_spec",
           "flymon/brain/ab_spec.py": "flymon.brain.ab_spec"}


def _spec_files() -> set:
    out = set()
    for p in (ROOT / "flymon").rglob("*.py"):
        if "node_modules" in p.parts or "__pycache__" in p.parts:
            continue
        if p.name == "spec.py" or p.name.endswith("_spec.py"):
            out.add(p.relative_to(ROOT).as_posix())
    return out


def _ints(v) -> set:
    if isinstance(v, bool):
        return set()
    if isinstance(v, int):
        return {v}
    if isinstance(v, (tuple, list)):
        return set().union(set(), *(_ints(x) for x in v))
    return set()


def _expand(obj) -> set:
    """The seed blocks a spec declares through a method or a (start, count) pair rather than a literal field."""
    name, out = type(obj).__name__, set()
    if name == "OdorSet":
        out |= set(range(obj.probe0, obj.probe0 + 2 * obj.n))
    elif name == "NSpec":
        out |= set(obj.judge_seeds(max(obj.n_grid)))
    elif name == "OSpec":
        out |= set(obj.o1_seeds) | set(obj.o2_seeds)
    elif name in ("BSpec", "RSpec"):
        names = [n for n, _ in obj.pair_seeds] if name == "BSpec" else list(obj.pair_names())
        for nm in names:
            for f in range(obj.n_flies):
                out |= set(obj.probe_seeds(nm, f))
                out |= {obj.train_seed(nm, f, t, sn) for t in range(obj.trials) for sn in (False, True)}
            if name == "RSpec":
                out |= set().union(*(set(v) for v in obj.qual_seeds(nm).values()))
        if name == "RSpec":
            out |= set(range(obj.taurec_seed_base - 1, obj.taurec_seed_base + obj.taurec_pulses))
    return out


def _collect(obj, out: set, seen: set) -> None:
    """Every int under a field whose name holds "seed" (or probe0), recursively through nested specs, plus _expand."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            _collect(v, out, seen)
        elif "seed" in f.name or f.name == "probe0":
            out |= _ints(v)
    out |= _expand(obj)


def _declared() -> set:
    out, seen = set(d6a.SEEDS), set()                    # d6a's D.6 (a) block: declared outside a spec module
    for mod in MODULES.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


DECLARED = _declared()
P_SEEDS = set(SPEC.seeds) | set(SM.seeds)
P_TRAIN = set(SPEC.train_seeds()) | set(SM.train_seeds())


def test_every_spec_module_is_enumerated():
    assert _spec_files() == set(MODULES) | {"flymon/brain/p_spec.py"}


def test_the_collector_sees_the_known_blocks():
    for s in (21_000_000, 21_002_000, 22_000_000, 22_001_000, 980_000, 986_207, 400_000, 810_100, 500, 9_000_000,
              991_999, 400):
        assert s in DECLARED, s


    declared = float(dict(O_SPEC.oracle_o)[SPEC.c1_pair])
    assert declared == -2.433 and round(SPEC.c1_frac * abs(declared), 3) == 0.608   # P.3
    assert SPEC.seeds == tuple(range(23_000_000, 23_000_032))
    assert len(SM.seeds) == SM.o.o2_n_seeds and all(23_009_000 <= s < 23_010_000 for s in SM.seeds)
    assert not set(SPEC.seeds) & set(SM.seeds)


def test_p_blocks_and_training_seeds_collide_with_nothing_declared():
    base, stride = N.train_seed_base, N.train_seed_stride
    assert (base, stride) == (1_000_000, 1000)                          # P.6.6's formula, read from n_spec
    assert not P_SEEDS & DECLARED
    assert not {p for p in P_SEEDS if (p - base) // stride in DECLARED}  # no declared s trains on a P seed (any t)
    assert not P_TRAIN & DECLARED
    assert not {p for p in P_TRAIN if (p - base) // stride in DECLARED}
    assert 22_000 not in DECLARED and 22_009 not in DECLARED             # P.6.6: s = 22_000 trains on 23_000_000 + t
    assert len(P_TRAIN) == len(P_SEEDS) * int(N.h4.teach_trials)
    assert SPEC.train_seeds()[0] == base + 23_000_000 * stride


def test_derived_from_ospec_by_replace_only():
    diff = {f.name for f in dataclasses.fields(OSpec) if getattr(SPEC.o, f.name) != getattr(O_SPEC, f.name)}
    assert diff == {"o2_pairs", "o2_arms", "o2_seed0", "smoke_seed0"}
    assert SPEC.o.o2_point == O_SPEC.o2_point and SPEC.o.o2_n_seeds == O_SPEC.o2_n_seeds == 32


def test_directions_arms_and_c1():
    assert SPEC.pairs() == {"r1": ("4:1", "dDL"), "r2": ("dDL", "4:1")}
    assert SPEC.flags() == {"plastic": (False, True, False), "frozen": (False, False, False),
                            "punish": (True, True, False)}
    assert tuple(a[0] for a in SPEC.o.o2_arms) == SPEC.arms == ("plastic", "frozen", "punish")
    assert {p for _, _, p in SPEC.o.o2_pairs} == {SPEC.c1_pair}
    declared = float(dict(O_SPEC.oracle_o)[SPEC.c1_pair])
    assert declared == -2.433 and round(SPEC.c1_frac * abs(declared), 3) == 0.608   # P.3
    assert SPEC.n_arms() == 192 and SM.n_arms() == 2 * 3 * SM.o.o2_n_seeds


def test_smoke_changes_scale_only():
    assert SM.o.boot_draws == O_SPEC.smoke_boot_draws and SM.oc_draws == SPEC.smoke_oc_draws
    for f in dataclasses.fields(PSpec):
        if f.name not in ("o", "oc_draws"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert (SPEC.oc_draws, SPEC.oc_seed) == (N.oc_draws, N.oc_seed)


def test_p_spec_restates_no_o_or_n_number():
    tree = ast.parse((ROOT / "flymon/brain/p_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 23_000_000, 23_009_000, 0.5, 1e-9}, nums             # 0: a tuple index
