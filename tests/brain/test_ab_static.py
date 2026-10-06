"""AB.7 0 6800–6801 (+ 6809@248cd25), Stage I part (plan Task 11a): the static check (no monkeypatching, no judgement
calls, no BAND 2K), the synthetic chain through 0f leaving AA / Y / Z / V untouched (PASS and STOP_FUTILE with the
record grid), the 0f block holding no Gen-2 value, and the mutant catalogue — every AB.7 0 mutant names the tests
that kill it, and every row of a built stage has ≥ 1 concrete edit in tests/brain/ab_mutants.py (run on a scratch
export, never on the worktree). Task 11b adds the Stage II tests and flips BUILT to both stages."""
import ast
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain import ab_rules
from flymon.brain.ab_spec import SPEC
from flymon.brain.h3_store import sha256_file
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y
from flymon.brain.z_spec import SPEC as Z
from tests.brain.ab_mutants import CATALOGUE, LATER, MUTANTS, STAGES, stage_of
from tests.brain.ab_world import World, doc

ROOT = Path(__file__).resolve().parents[2]
AB_FILES = [*sorted((ROOT / "flymon/brain").glob("ab_*.py")), ROOT / "scripts/run_ab.py"]
BANNED_CALLS = {"judge_pair", "judge", "overall_code", "pair_final", "pair_gate_code", "fly_class", "mech_ok"}
FOREIGN = ("aa_", "y_", "x_", "w_", "v_", "t_", "z_", "AA_", "Y_", "X_", "W_", "V_", "T_", "Z_")
BUILT = ("I",)                                   # Task 11b: ("I", "II")
STAGE1 = ("stage0", "reuse", "generate", "seal_code", "cal_gate", "futility")    # 0a–0f
OK = dict(n_sel=10, n_ver=10, workers=1, p_target=0.9, f_target=0.9)   # test scale: every structure reaches an α


def _imported(tree) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
    return out


def _spine(t):
    """The nodes an assignment target writes through: tuple / list / starred elements, then each Subscript's and
    Attribute's value down to the base name (an index expression — cnt[mod.f(x)] = … — is read, not written)."""
    if isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _spine(e)
        return
    while True:
        yield t
        if isinstance(t, ast.Starred):
            t = t.value
        elif isinstance(t, (ast.Subscript, ast.Attribute)):
            t = t.value
        else:
            return


@pytest.mark.parametrize("p", AB_FILES, ids=lambda p: p.name)
def test_no_monkeypatching(p):
    """AB.7 0 다른 모듈 속성 대입 · 판정 코드 · BAND 2K: no setattr / mock / reload / monkeypatch text; no
    assignment to an attribute (or vars(…)[…] / __dict__[…]) of a foreign-prefixed or an imported module name; no
    judgement call; no k0= keyword."""
    src = p.read_text()
    for bad in ("setattr(", "__setattr__", "mock", "reload(", "monkeypatch"):
        assert bad not in src, (p.name, bad)
    tree = ast.parse(src)
    mods = _imported(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for a in _spine(t):
                    if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name):
                        assert not a.value.id.startswith(FOREIGN), (p.name, n.lineno, a.value.id)
                        assert a.value.id not in mods, (p.name, n.lineno, "imported module attribute", a.value.id)
                    if isinstance(a, ast.Subscript):
                        v = a.value
                        assert not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "vars"), \
                            (p.name, n.lineno, "vars(...)[...] =")
                        assert not (isinstance(v, ast.Attribute) and v.attr == "__dict__"), (p.name, n.lineno)
        if isinstance(n, ast.Attribute):
            assert n.attr not in BANNED_CALLS, (p.name, n.lineno, n.attr)
        if isinstance(n, ast.Name):
            assert n.id not in BANNED_CALLS, (p.name, n.lineno, n.id)
        if isinstance(n, ast.alias):
            assert n.name not in BANNED_CALLS and (n.asname or "") not in BANNED_CALLS, (p.name, n.lineno, n.name)
        if isinstance(n, ast.keyword):
            assert n.arg != "k0", (p.name, n.lineno, "BAND 2K probes (k0)")


# ================================================================ the synthetic chain through 0f (AB.7 0 6801)
OTHERS = ("results/aa", "results/y", "results/z")
SUMMARIES = (SPEC.aa_summary, Y.summary, Z.summary, W.v_summary)


def _hashes():
    out = {}
    for d in OTHERS:
        out[d] = {str(p): sha256_file(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}
    for f in SUMMARIES:
        out[f] = sha256_file(f) if Path(f).exists() else None
    return out


def _world(tmp_path, monkeypatch, sub, **kw):
    d = tmp_path / sub
    d.mkdir()
    w = World(d, monkeypatch)
    w.s = dataclasses.replace(w.s, **dict(OK, **kw))
    for f in ("results/y/oracle.json", "results/z/split.json", Z.summary):      # stand-ins: Y / Z files exist too
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        Path(f).write_text(json.dumps(dict(world=f)))
    return w


def _gen2_keys():
    return list(doc()["generate"]["keys"])


def test_stage1_chain_leaves_others_untouched(tmp_path, monkeypatch):
    """World.chain through futility twice — fut_threshold 0.0 (PASS) and 1.01 (STOP_FUTILE + the record grid): the
    sha256 of every file under results/{aa,y,z} and of the aa / y / z / v summaries is unchanged, every 0a–0f block
    exists, results/ab/cache/ holds nothing."""
    for sub, thr, outcome in (("pass", 0.0, ab_rules.PASS), ("futile", 1.01, ab_rules.STOP_FUTILE)):
        w = _world(tmp_path, monkeypatch, sub, fut_threshold=thr)
        before = _hashes()
        assert before["results/aa"] and before["results/y"] and before["results/z"], sub
        assert all(before[f] for f in SUMMARIES), sub
        w.chain("cal_gate")
        out = w.runner().run("futility")
        assert out["outcome"] == outcome, (sub, out.get("sentence"))
        assert (out["grid"] is not None) == (outcome == ab_rules.STOP_FUTILE), sub
        assert _hashes() == before, sub
        d = doc()
        assert [b for b in STAGE1 if b in d] == list(STAGE1), sub
        assert [b for b in SPEC.stages if b in d] == list(STAGE1), sub
        cache = Path(SPEC.cache_dir)
        assert not cache.exists() or not any(p.is_file() for p in cache.rglob("*")), sub


def test_futility_block_has_no_gen2_value(tmp_path, monkeypatch):
    """AB.7 0f 측정 없음: the futility block and results/ab/futility.json name no Gen-2 set key (the world's generate
    block keys), PASS and STOP_FUTILE alike."""
    for sub, thr in (("pass", 0.0), ("futile", 1.01)):
        w = _world(tmp_path, monkeypatch, sub, fut_threshold=thr)
        w.chain("cal_gate")
        w.runner().run("futility")
        keys = _gen2_keys()
        assert len(keys) == len(w.gen["rows"]) > 0
        blob = json.dumps(doc()["futility"]) + Path(SPEC.fut_detail).read_text()
        hit = [k for k in keys if k in blob]
        assert hit == [], (sub, hit[:3])


# ================================================================ the catalogue (AB.7 0 6800 + 6809@248cd25)
def _exists(ref):
    f, name = ref.split("::")
    p = ROOT / f
    return p.exists() and f"def {name}(" in p.read_text()


def test_mutant_catalogue_points_at_existing_tests():
    assert len(CATALOGUE) == 47
    assert [r[0] for r in CATALOGUE] == list(range(1, 48))
    assert len({r[1] for r in CATALOGUE}) == 47
    for n, name, stage, tests in CATALOGUE:
        assert stage in STAGES and tests, (n, name)
        if stage in BUILT:
            for t in tests:
                assert _exists(t), (n, name, t)
    assert {n for n, _m, st, _t in CATALOGUE if st == "II"} == {3, 22, 23, 24, 26, 35, 36}
    assert set(LATER) == {27, 34}                         # rows split between Stage I and Stage II


def test_source_mutants_cover_the_catalogue_and_still_apply():
    """Every catalogue row of a built stage has ≥ 1 concrete edit; labels unique; each edit's anchor occurs exactly
    once in its file (so the catalogue follows the source) and its killing tests exist."""
    rows = {n: st for n, _m, st, _t in CATALOGUE}
    built = [m for m in MUTANTS if stage_of(m) in BUILT]
    covered = {n for m in built for n in m[0]}
    assert covered == {n for n, st in rows.items() if st in BUILT}
    assert all(set(m[0]) <= set(rows) for m in MUTANTS)
    labels = [m[1] for m in MUTANTS]
    assert len(labels) == len(set(labels))
    for nums, label, f, old, new, tests in built:
        assert old != new and tests, label
        assert (ROOT / f).read_text().count(old) == 1, (label, f)
        for t in tests:
            assert _exists(t), (label, t)
