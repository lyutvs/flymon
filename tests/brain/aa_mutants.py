"""AA.7 0 6249 mutant catalogue, as concrete source edits. Each entry = (catalogue name, label, file, old, new, killing
tests). `old` occurs exactly once in `file` (test_aa_static checks it, so the catalogue follows the source). The
mutants are never applied to the worktree: `python -m tests.brain.aa_mutants <scratch>` exports HEAD (plus the
working-tree tests/brain/aa_mutants.py and test_aa_static.py) into <scratch>, applies one mutant at a time there and
runs its killing tests with PYTHONDONTWRITEBYTECODE=1; a mutant is killed iff the run fails (pytest exit 1).
Options: --only <comma-separated label substrings>, --python <interpreter> (default: the worktree's .venv)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

R, E, S, ST = "flymon/brain/aa_runner.py", "flymon/brain/aa_estimate.py", "flymon/brain/aa_spec.py", \
    "flymon/brain/aa_store.py"
T_A, T_B, T_C, T_E, T_COV, T_X = ("tests/brain/test_aa_runner_a.py", "tests/brain/test_aa_runner_b.py",
                                  "tests/brain/test_aa_runner_c.py", "tests/brain/test_aa_estimate.py",
                                  "tests/brain/test_aa_coverage.py", "tests/brain/test_aa_static.py")
S1_LINE = 'S1 = [p for p in doc["screen"]["pairs"] if p["passed"]]'
PAIR_MEAN = '**{g: dict(mean=float(a["w"][:, aa_estimate.GI[g]].mean()), ci=ci[g])'
REC_MEAN = "mean=float(x.mean()) if ok else None"
PAIR_DRAW = "reps[o:o + n] = V[r.integers(0, F, size=(n, F))].mean(1)"
SET_REPS = "reps = BOOT[m](W_, groups, stream(s.boot_seed, *_tags(m, set_tag, s)), B, s.boot_chunk)"
SCREEN_U = 'U = units(prs, "main", s.probes, s.flies, V_SPEC.lever_edit, brains=("naive",))'
LEARN_U = 'return prs, units(prs, "main", s.probes, s.flies, V_SPEC.lever_edit)'
SEAL_W = 'return self._write("seal_code", dict(outcome=aa_rules.PASS'


def _n(test_file, name):
    return f"{test_file}::{name}"


MUTANTS = [
    # 1
    ("1차 집합을 학습 값으로 고름", "s1_by_learning", R, S1_LINE,
     'S1 = [p for p in doc["screen"]["pairs"] if p["passed"] or '
     'float(np.mean(aa_estimate.pair_arrays(data[p["key"]], z, s)["w"])) > -1e9]',
     [_n(T_C, "test_s1_never_changes_with_learning_values")]),
    # 2
    ("학습 뒤 순진 값을 다시 계산", "naive_recomputed_select", R, S1_LINE,
     'S1 = [p for p in doc["screen"]["pairs"] if y_rules.final_filter(data[p["key"]]["pre"], z, Y_SPEC)["passed"]]',
     [_n(T_C, "test_s1_never_changes_with_learning_values")]),
    ("학습 뒤 순진 값을 다시 계산", "naive_recomputed_value", R, 'c=int(p["c"]), naive_d=p["naive_d"], L_A=p["L_A"]',
     'c=int(p["c"]), naive_d=y_rules.final_filter(data[p["key"]]["pre"], z, Y_SPEC)["d"], L_A=p["L_A"]',
     [_n(T_C, "test_s1_never_changes_with_learning_values")]),
    # 3
    ("1단계를 쌍 단위로(묶음 무시)", "two_stage_singletons", E, "    G, M = len(groups), max(len(g) for g in groups)",
     "    groups = [[p] for g in groups for p in g]\n    G, M = len(groups), max(len(g) for g in groups)",
     [_n(T_E, "test_two_stage_draws_whole_groups")]),
    ("1단계를 쌍 단위로(묶음 무시)", "set_groups_ignored", E, "groups = y_rules.merge_groups(list(keys))",
     "groups = [[j] for j in range(len(keys))]",
     [_n(T_X, "test_estimate_set_groups_by_x_odour")]),
    # 4
    ("마리 대신 프로브를 재표집", "pair_ci_probe_noise", E, PAIR_DRAW,
     "reps[o:o + n] = V.mean(0) + V.std(0) * r.standard_normal((n, V.shape[1])) / math.sqrt(s.probes)",
     [_n(T_E, "test_pair_ci_reproducible_shared_indices_and_flies_not_probes")]),
    ("마리 대신 프로브를 재표집", "pair_arrays_probe_resample", E, "    raw = fly_dprimes(d, z)\n",
     "    _rp = np.random.default_rng(0).integers(0, s.probes, s.probes)\n"
     "    raw = fly_dprimes({k_: np.asarray(v_)[:, _rp] for k_, v_ in d.items()}, z)\n",
     [_n(T_X, "test_pair_arrays_use_every_probe_once")]),
    # 5
    ("평균 대신 중앙값을 1차로", "pair_record_median", E, REC_MEAN, "mean=float(np.median(x)) if ok else None",
     [_n(T_E, "test_primary_is_mean_not_median_and_sign_kept")]),
    ("평균 대신 중앙값을 1차로", "set_theta_median", E, "    theta = W_.mean(1)\n", "    theta = np.median(W_, 1)\n",
     [_n(T_X, "test_estimate_set_point_is_mean_of_winsorized_fly_means")]),
    ("평균 대신 중앙값을 1차로", "primary_pair_median", R, PAIR_MEAN,
     '**{g: dict(mean=float(np.median(a["w"][:, aa_estimate.GI[g]])), ci=ci[g])',
     [_n(T_X, "test_estimate_block_matches_estimators_on_learn_raw")]),
    # 6
    ("±∞ 마리를 평균에서 뺌(1차에서)", "pair_record_drop_inf", E, REC_MEAN,
     "mean=float(x[np.isfinite(xr)].mean()) if ok else None",
     [_n(T_E, "test_winsorize_inf_and_large_counts_and_mean")]),
    ("±∞ 마리를 평균에서 뺌(1차에서)", "set_theta_drop_inf", E, "    theta = W_.mean(1)\n",
     "    theta = _nanmean(np.where(np.isinf(R_), np.nan, W_), 1)\n",
     [_n(T_X, "test_estimate_set_point_is_mean_of_winsorized_fly_means")]),
    # 7
    ("잘라냄 값을 ±10 아닌 값으로", "spec_winsor_8", S, "    winsor: float = 10.0", "    winsor: float = 8.0",
     [_n(T_E, "test_clip_is_ten")]),
    ("잘라냄 값을 ±10 아닌 값으로", "winsorize_half", E, "np.clip(np.asarray(x, float), -s.winsor, s.winsor)",
     "np.clip(np.asarray(x, float), -s.winsor / 2, s.winsor / 2)", [_n(T_E, "test_clip_is_ten")]),
    # 8
    ("부호를 뒤집음", "fly_dprime_negated", E, "return np.asarray(WV.gate_stats(d, z), float)",
     "return -np.asarray(WV.gate_stats(d, z), float)", [_n(T_E, "test_fly_dprimes_are_gate_stats_bitwise")]),
    ("부호를 뒤집음", "pair_record_negated", E, REC_MEAN, "mean=float(-x.mean()) if ok else None",
     [_n(T_E, "test_primary_is_mean_not_median_and_sign_kept")]),
    ("부호를 뒤집음", "set_point_negated", E, 'out["point"] = {g: float(theta[:, i].mean())',
     'out["point"] = {g: float(-theta[:, i].mean())', [_n(T_X, "test_estimate_set_point_is_mean_of_winsorized_fly_means")]),
    ("부호를 뒤집음", "primary_pair_negated", R, PAIR_MEAN,
     '**{g: dict(mean=float(-a["w"][:, aa_estimate.GI[g]].mean()), ci=ci[g])',
     [_n(T_X, "test_estimate_block_matches_estimators_on_learn_raw")]),
    # 9
    ("관문마다 다른 재표집 색인", "pair_ci_per_column", E, PAIR_DRAW,
     "reps[o:o + n] = np.stack([V[r.integers(0, F, size=(n, F)), j].mean(1) for j in range(V.shape[1])], 1)",
     [_n(T_E, "test_pair_ci_reproducible_shared_indices_and_flies_not_probes")]),
    ("관문마다 다른 재표집 색인", "set_reps_per_gate", E, SET_REPS,
     "reps = (lambda r_: np.concatenate([BOOT[m](W_[..., i:i + 1], groups, r_, B, s.boot_chunk) "
     "for i in range(W_.shape[2])], 1))(stream(s.boot_seed, *_tags(m, set_tag, s)))",
     [_n(T_X, "test_estimate_set_gates_share_one_resample")]),
    # 10
    ("coverage에 결과값을 넘김", "coverage_values_param", E,
     "def coverage(sizes, method, set_tag, s=AA, reps=None, B=None) -> dict:",
     "def coverage(sizes, method, set_tag, s=AA, reps=None, B=None, values=None) -> dict:",
     [_n(T_COV, "test_signature_takes_structure_only")]),
    ("coverage에 결과값을 넘김", "coverage_cell_values_param", E,
     "def coverage_cell(sizes, method, set_tag, ci, s=AA, reps=None, B=None) -> float:",
     "def coverage_cell(sizes, method, set_tag, ci, s=AA, reps=None, B=None, point=None) -> float:",
     [_n(T_COV, "test_signature_takes_structure_only")]),
    # 11
    ("묶음 < 5인데 두 단계 CI를 1차로", "ci_plan_groups_2", E, "    if n_groups >= s.min_groups:",
     "    if n_groups >= 2:", [_n(T_E, "test_ci_plan_table")]),
    ("묶음 < 5인데 두 단계 CI를 1차로", "spec_min_groups_2", S, "    min_groups: int = 5", "    min_groups: int = 2",
     [_n(T_E, "test_ci_plan_table")]),
    # 12
    ("k = 0인데 학습으로 진행", "screen_k0_pass", R, "body = aa_rules.no_pairs_stop(rc, cnt) if k == 0 else",
     "body = aa_rules.no_pairs_stop(rc, cnt) if k < 0 else", [_n(T_B, "test_screen_k0_stop_no_pairs")]),
    ("k = 0인데 학습으로 진행", "ci_plan_k0_pooled", E,
     '        return dict(kind="none", primary=None, records=[], flag=None)\n', "        pass\n",
     [_n(T_E, "test_ci_plan_table")]),
    # 13
    ("차등 시험에서 시드 공식 · 단계 순서 · 뇌 순서를 바꿈", "pilot_seed_j_plus_1", R,
     "sp.pilot_probe_seeds(j, f, K), sp.pilot_train_base(j, 0)",
     "sp.pilot_probe_seeds(j + 1, f, K), sp.pilot_train_base(j, 0)",
     [_n(T_A, "test_differential_bit_equal_on_y_path")]),
    ("차등 시험에서 시드 공식 · 단계 순서 · 뇌 순서를 바꿈", "phase_order_swapped", R,
     'sp.phases(b if b in BRAINS else "R", b0, b1)', 'sp.phases(b if b in BRAINS else "R", b1, b0)',
     [_n(T_A, "test_differential_bit_equal_on_y_path")]),
    ("차등 시험에서 시드 공식 · 단계 순서 · 뇌 순서를 바꿈", "brain_phases_swapped", R,
     'sp.phases(b if b in BRAINS else "R", b0, b1)',
     'sp.phases({"R": "RN", "RN": "R"}.get(b, b) if b in BRAINS else "R", b0, b1)',
     [_n(T_A, "test_differential_bit_equal_on_y_path")]),
    ("차등 시험에서 시드 공식 · 단계 순서 · 뇌 순서를 바꿈", "count_stages_swapped", R,
     "    return w_records.pair_data(rows, flies)\n",
     "    d = w_records.pair_data(rows, flies)\n    return dict(d, R1=d[\"R2\"], R2=d[\"R1\"])\n",
     [_n(T_A, "test_differential_bit_equal_on_y_path")]),
    # 14
    ("바닥 몫에서 R1/R2 또는 X/Y를 바꿈", "phi_R_from_R2", E, "phi_R=float((r1[..., WV.P, WV.X] == 0).mean())",
     "phi_R=float((r2[..., WV.P, WV.X] == 0).mean())", [_n(T_E, "test_floor_shares_match_pilot_record_taught")]),
    ("바닥 몫에서 R1/R2 또는 X/Y를 바꿈", "phi_R_Y", E, "phi_R=float((r1[..., WV.P, WV.X] == 0).mean())",
     "phi_R=float((r1[..., WV.P, WV.Y] == 0).mean())", [_n(T_E, "test_floor_shares_match_pilot_record_taught")]),
    ("바닥 몫에서 R1/R2 또는 X/Y를 바꿈", "phi_P_from_R1", E, "phi_P=float((r2[..., WV.A, WV.X] == 0).mean())",
     "phi_P=float((r1[..., WV.A, WV.X] == 0).mean())", [_n(T_E, "test_floor_shares_match_pilot_record_taught")]),
    ("바닥 몫에서 R1/R2 또는 X/Y를 바꿈", "phi_P_Y", E, "phi_P=float((r2[..., WV.A, WV.X] == 0).mean())",
     "phi_P=float((r2[..., WV.A, WV.Y] == 0).mean())", [_n(T_E, "test_floor_shares_match_pilot_record_taught")]),
    # 15
    ("작은-k 규칙 생략", "set_plan_fixed", E, "plan = ci_plan(len(keys), len(groups), s)",
     'plan = dict(kind="pooled", primary="two_stage", records=["fly", "pair"], flag=None)',
     [_n(T_X, "test_estimate_set_follows_small_k_rule")]),
    ("작은-k 규칙 생략", "ci_plan_few_groups_two_stage", E,
     '    return dict(kind="pooled", primary="fly", records=rec, flag="적은 묶음")',
     '    return dict(kind="pooled", primary="two_stage", records=["fly", "pair"], flag=None)',
     [_n(T_E, "test_ci_plan_table")]),
    # 16
    ("판정 코드(judge_pair 등) 호출", "judge_attr", R, "class _BudgetStop(Exception):",
     "_J = w_verdict.judge_pair\n\n\nclass _BudgetStop(Exception):",
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("판정 코드(judge_pair 등) 호출", "judge_from_import", R, "from .r_store import load_manifest\n",
     "from .r_store import load_manifest\nfrom .w_verdict import overall_code as _oc  # noqa: F401\n",
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    # 17
    ("BAND 2K 측정", "k0_keyword", R, "seeds, b0, b1 = sp.probe_seeds(c, f, K), sp.train_base(c, 0)",
     "seeds, b0, b1 = sp.probe_seeds(c, f, 2 * K, k0=K), sp.train_base(c, 0)",
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("BAND 2K 측정", "k0_positional", R, "seeds, b0, b1 = sp.probe_seeds(c, f, K), sp.train_base(c, 0)",
     "seeds, b0, b1 = sp.probe_seeds(c, f, 2 * K, K), sp.train_base(c, 0)",
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("BAND 2K 측정", "learn_2K", R, LEARN_U, LEARN_U.replace("s.probes", "2 * s.probes"),
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("BAND 2K 측정", "screen_2K", R, SCREEN_U, SCREEN_U.replace("s.probes", "2 * s.probes"),
     [_n(T_B, "test_screen_seeds_use_c_and_main_block")]),
    # 18
    ("results/y/ · results/z/에 씀", "write_results_y", R, SEAL_W,
     'Path("results/y/aa_touch.json").write_text("{}")\n        ' + SEAL_W,
     [_n(T_X, "test_full_chain_leaves_y_and_z_untouched")]),
    ("results/y/ · results/z/에 씀", "write_results_z", R, SEAL_W,
     'Path("results/z").mkdir(exist_ok=True)\n        Path("results/z/aa_touch.json").write_text("{}")\n        '
     + SEAL_W, [_n(T_X, "test_full_chain_leaves_y_and_z_untouched")]),
    ("results/y/ · results/z/에 씀", "guard_widened_to_results", ST, 'ALLOWED_DIR = "results/aa/"',
     'ALLOWED_DIR = "results/"', [_n("tests/brain/test_aa_store.py", "test_guard_refuses_outside")]),
    # 19
    ("Y · Z 모듈 속성 대입(몽키패치)", "assign_y_attr", R, "class _BudgetStop(Exception):",
     "y_rules.final_filter = y_rules.final_filter\n\n\nclass _BudgetStop(Exception):",
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("Y · Z 모듈 속성 대입(몽키패치)", "setattr_z", R, "class _BudgetStop(Exception):",
     'setattr(z_split, "lenient_items", z_split.lenient_items)\n\n\nclass _BudgetStop(Exception):',
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("Y · Z 모듈 속성 대입(몽키패치)", "vars_y_spec", R, "class _BudgetStop(Exception):",
     'vars(y_oc)["tag"] = y_oc.tag\n\n\nclass _BudgetStop(Exception):',
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    ("Y · Z 모듈 속성 대입(몽키패치)", "spec_dict_assign", R, "class _BudgetStop(Exception):",
     'Y_SPEC.__dict__["d_false"] = Y_SPEC.d_false\n\n\nclass _BudgetStop(Exception):',
     [_n(T_X, "test_no_monkeypatching_no_judgement_no_band")]),
    # 20
    ("주 세트 시드 대신 다른 블록 사용", "screen_pilot_block", R, SCREEN_U, SCREEN_U.replace('"main"', '"pilot"'),
     [_n(T_B, "test_screen_seeds_use_c_and_main_block")]),
    ("주 세트 시드 대신 다른 블록 사용", "learn_pilot_block", R, LEARN_U, LEARN_U.replace('"main"', '"pilot"'),
     [_n(T_X, "test_learn_units_use_main_seeds_of_c")]),
    ("주 세트 시드 대신 다른 블록 사용", "w_spec_pilot_probe_seed", R, "probe_seed0=ys.probe_seed0,",
     "probe_seed0=ys.pilot_probe_seed0,", [_n(T_B, "test_screen_seeds_use_c_and_main_block")]),
    ("주 세트 시드 대신 다른 블록 사용", "w_spec_pilot_train_seed", R, "train_seed0=ys.train_seed0,",
     "train_seed0=ys.pilot_train_seed0,", [_n(T_X, "test_learn_units_use_main_seeds_of_c")]),
    # 21
    ("c 대신 31쌍 안의 번호로 시드 계산", "main_c_is_position", R, '                c = int(r["c"])\n',
     "                c = j\n", [_n(T_A, "test_units_main_seed_formula_uses_c")]),
]


def export(dest: Path, root: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(root), "archive", "HEAD", "flymon", "tests", "scripts", "data",
                          "results/summary", "pyproject.toml", "uv.lock"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc, check=True)
    for f in ("tests/brain/aa_mutants.py", "tests/brain/test_aa_static.py"):
        if (root / f).exists():
            shutil.copy2(root / f, dest / f)


def run(dest: Path, py: str, only: str | None, log) -> list:
    out = []
    env = dict(os.environ, PYTHONPATH=str(dest), PYTHONDONTWRITEBYTECODE="1")
    for cat, label, f, old, new, tests in MUTANTS:
        if only and not any(o in label for o in only.split(",")):
            continue
        p = dest / f
        src = p.read_text()
        assert src.count(old) == 1, (label, src.count(old))
        p.write_text(src.replace(old, new))
        t0 = time.time()
        try:
            r = subprocess.run([py, "-m", "pytest", *tests, "-o", "addopts=", "-q", "-x", "-p", "no:cacheprovider"],
                               cwd=dest, env=env, capture_output=True, text=True)
        finally:
            p.write_text(src)
        killed = r.returncode == 1
        lines = r.stdout.strip().splitlines() or ["?"]
        why = next((x for x in lines if x.startswith("E ")), "")[:160]
        line = (f"{'KILLED ' if killed else 'SURVIVED'} rc={r.returncode} {time.time() - t0:5.1f}s {label} [{cat}] :: "
                f"{lines[-1]} :: {why}")
        print(line, file=log, flush=True)
        out.append((label, killed, r.returncode))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--only")
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--python", default=str(root / ".venv/bin/python"))
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    export(dest, root)
    res = run(dest, a.python, a.only, sys.stdout)
    n = sum(k for _l, k, _r in res)
    print(f"mutants killed {n} / {len(res)}; survivors: {[l_ for l_, k, _r in res if not k]}", flush=True)
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
