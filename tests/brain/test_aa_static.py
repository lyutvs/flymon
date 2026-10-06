"""AA.7 0 6249–6250: the static check (no monkeypatching, no judgement / BAND code, no writes outside results/aa), the
full synthetic chain leaving Y / Z untouched, and the mutant catalogue — each mutant of 6249 names the test that kills
it (the catalogue test fails if a named test disappears). The concrete source mutants live in tests/brain/aa_mutants.py
(run on a scratch export, never on the worktree); the tests below whose names appear there close the gaps the
per-module tests leave (estimate_set / the estimate block / the learn units)."""
import ast
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import aa_estimate as E
from flymon.brain import w_verdict as WV
from flymon.brain import y_spec, y_store
from flymon.brain.aa_spec import SPEC
from flymon.brain.h3_store import sha256_file
from flymon.brain.r_pairs import row_key
from tests.brain.aa_mutants import MUTANTS
from tests.brain.aa_world import LENIENT_C, MAIN, World, doc

ROOT = Path(__file__).resolve().parents[2]
AA_FILES = [*sorted((ROOT / "flymon/brain").glob("aa_*.py")), ROOT / "scripts/run_aa.py"]
BANNED_CALLS = {"judge_pair", "judge", "overall_code", "pair_final", "pair_gate_code", "fly_class", "mech_ok",
                "band_units", "extend"}
FOREIGN = ("y_", "x_", "w_", "z_", "Y_", "X_", "W_", "Z_", "V_")
SEED_ARGS = {"probe_seeds": 3, "pilot_probe_seeds": 3, "smoke_probe_seeds": 2}     # (c | j, f, K) / (f, K): no k0
Z = {"A": (16.916666666666668, 12.483878492769072), "P": (80.16666666666667, 29.775432639827233)}
S = dataclasses.replace(SPEC, boot_b=300, boot_chunk=120)
C_OF = {row_key(r): r["c"] for r in MAIN}


def _mentions_probes(n) -> bool:
    return any((isinstance(a, ast.Attribute) and a.attr == "probes") or (isinstance(a, ast.Name) and a.id == "K")
               for a in ast.walk(n))


@pytest.mark.parametrize("p", AA_FILES, ids=lambda p: p.name)
def test_no_monkeypatching_no_judgement_no_band(p):
    src = p.read_text()
    for bad in ("setattr(", "__setattr__", "mock", "reload(", "monkeypatch"):
        assert bad not in src, (p.name, bad)
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for a in ast.walk(t):
                    if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name):
                        assert not a.value.id.startswith(FOREIGN), (p.name, n.lineno, a.value.id)
                    if isinstance(a, ast.Subscript):          # vars(mod)[…] = / obj.__dict__[…] =
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
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SEED_ARGS:
            assert len(n.args) <= SEED_ARGS[n.func.attr], (p.name, n.lineno, "BAND 2K probes (positional k0)")
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            two = [x for x in (n.left, n.right) if isinstance(x, ast.Constant) and x.value == 2]
            assert not (two and _mentions_probes(n)), (p.name, n.lineno, "BAND 2K probes (2 × K)")


def test_final_filter_only_in_screen():
    tree = ast.parse((ROOT / "flymon/brain/aa_runner.py").read_text())
    users = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
             for n in ast.walk(f) if isinstance(n, ast.Attribute) and n.attr == "final_filter"}
    assert users == {"stage_screen", "differential"}
    for p in AA_FILES:
        if p.name != "aa_runner.py":
            assert "final_filter" not in p.read_text(), p.name


def _hashes():
    out = dict(allowed=y_store.ALLOWED_DIR, spec=y_spec.SPEC == y_spec.YSpec())
    for d in ("results/y", "results/z"):
        out[d] = {str(p): sha256_file(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}
    for f in ("results/summary/y_learning.json", "results/summary/z_learning.json"):
        out[f] = sha256_file(f) if Path(f).exists() else None
    return out


def test_full_chain_leaves_y_and_z_untouched(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    before = _hashes()
    assert before["results/y"] and before["results/summary/y_learning.json"]        # the world's Y files are there
    w.chain("records")
    assert _hashes() == before and before["allowed"] == "results/y/" and before["spec"]
    d = doc()
    assert [b for b in SPEC.stages if b in d] == list(SPEC.stages)


def test_learn_block_has_no_statistic(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("learn")
    blob = str(doc()["learn"])
    for word in ("naive_d", "'mean'", "gate_stats", "'d':", "L_A"):
        assert word not in blob


# ================================================================ gap tests named by aa_mutants (synthetic only)
def _arrs(P, seed=3, F=8, inf_at=None):
    r = np.random.default_rng(seed)
    out = {}
    for j in range(P):
        raw = r.gamma(1.2, 1.5, (F, 4)) * np.where(np.arange(4) % 2, -1.0, 1.0) + r.normal(0, 0.3, (F, 4))
        raw[0, :] *= 4.0                                                # skewed: mean ≠ median, some |d′| > 10
        if inf_at is not None and j in inf_at:
            raw[1, :] = np.where(np.arange(4) % 2, -np.inf, np.inf)
        out[j] = dict(raw=raw, w=np.clip(raw, -10.0, 10.0), rc=r.normal(0, 1, (F, 2)))
    return out


def _keys(xs):
    return [f"a|{300 + i}|{x}|y{i}" for i, x in enumerate(xs)]


def test_estimate_set_point_is_mean_of_winsorized_fly_means():
    keys = _keys(["X0", "X1", "X2", "X3", "X4", "X5", "X6"])
    a = _arrs(len(keys), inf_at={2, 5})
    arrs = {k: a[j] for j, k in enumerate(keys)}
    est = E.estimate_set(arrs, keys, "S1", S)
    pm = np.stack([np.clip(arrs[k]["raw"], -10.0, 10.0).mean(0) for k in keys])      # pair means, ±∞ → ±10 kept
    for i, g in enumerate(WV.GATES):
        assert est["point"][g] == pytest.approx(float(pm[:, i].mean()), rel=1e-12, abs=1e-12)
        assert est["range"][g] == pytest.approx([float(pm[:, i].min()), float(pm[:, i].max())], rel=1e-12)
    med = np.stack([np.median(arrs[k]["w"], 0) for k in keys]).mean(0)
    assert not np.allclose(med, pm.mean(0))                                          # the fixture separates them


def test_estimate_set_groups_by_x_odour():
    keys = _keys(["X0", "X0", "X0", "X1", "X2", "X2", "X3", "X4"])
    a = _arrs(len(keys))
    est = E.estimate_set({k: a[j] for j, k in enumerate(keys)}, keys, "S1", S)
    assert est["n_groups"] == 5 and est["sizes"] == [3, 2, 1, 1, 1]
    assert est["plan"]["primary"] == "two_stage"


def test_estimate_set_follows_small_k_rule():
    for xs, kind, primary, flag in ((["X0"], "pair_study", None, "쌍별 연구"),
                                    (["X0", "X1", "X2"], "pooled", "fly", "적은 묶음"),
                                    (["X0", "X0", "X1", "X1", "X2", "X3"], "pooled", "fly", "적은 묶음"),
                                    (["X0", "X1", "X2", "X3", "X4"], "pooled", "two_stage", None)):
        keys = _keys(xs)
        a = _arrs(len(keys))
        est = E.estimate_set({k: a[j] for j, k in enumerate(keys)}, keys, "S1", S)
        assert (est["plan"]["kind"], est["plan"]["primary"], est["plan"]["flag"]) == (kind, primary, flag), xs
        assert (kind == "pooled") == ("point" in est)
        if kind == "pooled":
            assert set(est["ci"]) == {primary, *est["plan"]["records"]}


def test_estimate_set_gates_share_one_resample():
    keys = _keys(["X0", "X0", "X1", "X2", "X3", "X4", "X5"])
    a = _arrs(len(keys))
    arrs = {k: dict(a[j], w=np.repeat(a[j]["w"][:, :1], 4, 1)) for j, k in enumerate(keys)}    # 4 identical gates
    for xs in (keys, keys[:4]):                                         # two-stage primary, then fly-level primary
        est = E.estimate_set({k: arrs[k] for k in xs}, xs, "S1", S)
        assert all(np.array_equal(est["_reps"][:, 0], est["_reps"][:, j]) for j in range(4))
        for m, ci in est["ci"].items():
            assert all(ci[g] == ci[WV.GATES[0]] for g in WV.GATES), m


def test_pair_arrays_use_every_probe_once():
    r = np.random.default_rng(11)
    d = {s: np.clip(np.rint(r.normal(50, 15, (8, 8, 2, 2))), 0, None).astype(np.int64) for s in WV.STAGES}
    d["RN1"] = d["R1"].copy()
    a = E.pair_arrays(d, Z, S)
    g = np.asarray(WV.gate_stats(d, Z), float)
    assert np.array_equal(a["raw"], g) and np.array_equal(a["w"], np.clip(g, -10.0, 10.0))


def _world(tmp_path, monkeypatch, fail=()):
    w = World(tmp_path, monkeypatch)
    w.s = dataclasses.replace(w.s, workers=1)
    for c in fail:
        w.model.offset[row_key(MAIN[c])] = 40.0
    return w


def test_estimate_block_matches_estimators_on_learn_raw(tmp_path, monkeypatch):
    """The estimate block's pair means and pooled points, recomputed here from the learn raw with w_verdict.gate_stats
    and a plain ±10 clip and mean (independent of aa_estimate)."""
    w = _world(tmp_path, monkeypatch, fail=LENIENT_C[:2])
    real = w.model.probe
    code = {None: 1, "PAM08": 2, "PPL105": 3}

    def probe(pair, seed, edit, applied, plastic):       # brain-history noise: fly d′ finite and spread (mean ≠ median);
        x, y = real(pair, seed, edit, applied, plastic)  # pre (nothing applied) and RN1 = R1 stay bit-identical
        if applied:
            n = np.random.default_rng([int(seed)] + [code[a] for a in applied]).normal(0, 4, 2)
            x = dict(x, A=int(max(0, x["A"] + round(n[0]))), P=int(max(0, x["P"] + round(n[1]))))
        return x, y
    w.model.probe = probe
    for i, c in enumerate(LENIENT_C):
        w.model.effects[row_key(MAIN[c])] = (3.0 + i % 5, 2.0 + i % 3)
    out = w.chain("estimate")
    d = doc()
    data = w.runner()._learn_data(d)
    pr = out["primary"]
    s1 = [p["key"] for p in d["screen"]["pairs"] if p["passed"]]
    assert [p["key"] for p in pr["pairs"]] == s1 and pr["plan"]["primary"] == "two_stage"
    fly = {k: np.clip(np.asarray(WV.gate_stats(data[k], w.z), float), -10.0, 10.0) for k in s1}
    assert any(not np.allclose(np.median(v, 0)[[E.GI[g] for g in E.PRIMARY]], v.mean(0)[[E.GI[g] for g in E.PRIMARY]])
               for v in fly.values())                    # the fixture separates mean and median
    pm = {k: v.mean(0) for k, v in fly.items()}
    for p in pr["pairs"]:
        for g in E.PRIMARY:
            assert p[g]["mean"] == pytest.approx(float(pm[p["key"]][E.GI[g]]), rel=1e-12, abs=1e-12), (p["key"], g)
    for g in E.PRIMARY:
        v = np.array([pm[k][E.GI[g]] for k in s1])
        assert pr["pooled"][g]["point"] == pytest.approx(float(v.mean()), rel=1e-12, abs=1e-12)


def test_learn_units_use_main_seeds_of_c(tmp_path, monkeypatch):
    """Learn (R · N · RN): probe seeds = Y's 62M + c × 4_000 + f × 100 + k (k < K, no BAND 2K), training base = Y's
    64M + c × 40_000, for every lenient pair and fly."""
    w = _world(tmp_path, monkeypatch)
    w.chain("learn")
    det = json.loads(Path(w.s.raw_dir, "learn.json").read_text())
    seen = set()
    for m in det["manifest"]:
        ins = json.loads(Path(m["cache_file"]).read_text())["inputs"]
        c, f = C_OF[ins["pair"]], ins["fly"]
        assert ins["block"] == "learn" and ins["brain"] in ("R", "N", "RN")
        assert ins["probe_seeds"] == [62_000_000 + c * 4_000 + f * 100 + k for k in range(w.s.probes)]
        assert ins["phases"][0][2] == 64_000_000 + c * 40_000 and ins["phases"][1][2] == 64_000_000 + c * 40_000 + 20
        seen.add((ins["pair"], f, ins["brain"]))
    assert seen == {(row_key(MAIN[c]), f, b) for c in LENIENT_C for f in range(w.s.flies) for b in ("R", "N", "RN")}


# ================================================================ the catalogue
CATALOGUE = {   # AA.7 0 6249 mutant -> killing test (file::name)
    "1차 집합을 학습 값으로 고름": "test_aa_runner_c.py::test_s1_never_changes_with_learning_values",
    "학습 뒤 순진 값을 다시 계산": "test_aa_runner_c.py::test_s1_never_changes_with_learning_values",
    "1단계를 쌍 단위로(묶음 무시)": "test_aa_estimate.py::test_two_stage_draws_whole_groups",
    "마리 대신 프로브를 재표집": "test_aa_estimate.py::test_pair_ci_reproducible_shared_indices_and_flies_not_probes",
    "평균 대신 중앙값을 1차로": "test_aa_estimate.py::test_primary_is_mean_not_median_and_sign_kept",
    "±∞ 마리를 평균에서 뺌(1차에서)": "test_aa_estimate.py::test_winsorize_inf_and_large_counts_and_mean",
    "잘라냄 값을 ±10 아닌 값으로": "test_aa_estimate.py::test_clip_is_ten",
    "부호를 뒤집음": "test_aa_estimate.py::test_primary_is_mean_not_median_and_sign_kept",
    "관문마다 다른 재표집 색인": "test_aa_estimate.py::test_pair_ci_reproducible_shared_indices_and_flies_not_probes",
    "coverage에 결과값을 넘김": "test_aa_coverage.py::test_signature_takes_structure_only",
    "묶음 < 5인데 두 단계 CI를 1차로": "test_aa_estimate.py::test_ci_plan_table",
    "k = 0인데 학습으로 진행": "test_aa_runner_b.py::test_screen_k0_stop_no_pairs",
    "차등 시험에서 시드 공식 · 단계 순서 · 뇌 순서를 바꿈": "test_aa_runner_a.py::test_differential_catches_mutants",
    "바닥 몫에서 R1/R2 또는 X/Y를 바꿈": "test_aa_estimate.py::test_floor_shares_match_pilot_record_taught",
    "작은-k 규칙 생략": "test_aa_estimate.py::test_ci_plan_table",
    "판정 코드(judge_pair 등) 호출": "test_aa_static.py::test_no_monkeypatching_no_judgement_no_band",
    "BAND 2K 측정": "test_aa_static.py::test_no_monkeypatching_no_judgement_no_band",
    "results/y/ · results/z/에 씀": "test_aa_static.py::test_full_chain_leaves_y_and_z_untouched",
    "Y · Z 모듈 속성 대입(몽키패치)": "test_aa_static.py::test_no_monkeypatching_no_judgement_no_band",
    "주 세트 시드 대신 다른 블록 사용": "test_aa_runner_b.py::test_screen_seeds_use_c_and_main_block",
    "c 대신 31쌍 안의 번호로 시드 계산": "test_aa_runner_a.py::test_units_main_seed_formula_uses_c",
}


def _exists(ref):
    f, name = ref.split("::")
    return f"def {name}(" in (ROOT / "tests/brain" / Path(f).name).read_text()


def test_mutant_catalogue_points_at_existing_tests():
    assert len(CATALOGUE) == 21
    for m, ref in CATALOGUE.items():
        assert _exists(ref), (m, ref)


def test_source_mutants_cover_the_catalogue_and_still_apply():
    """Every AA.7 0 mutant has ≥ 1 concrete source edit in aa_mutants; each edit's anchor occurs exactly once in its
    file (so the catalogue follows the source) and its killing tests exist."""
    assert {m[0] for m in MUTANTS} == set(CATALOGUE)
    labels = [m[1] for m in MUTANTS]
    assert len(labels) == len(set(labels))
    for cat, label, f, old, new, tests in MUTANTS:
        assert old != new and tests, label
        assert (ROOT / f).read_text().count(old) == 1, (label, f)
        for t in tests:
            assert _exists(t), (label, t)
