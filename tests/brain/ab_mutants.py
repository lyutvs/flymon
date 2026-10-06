"""AB.7 0 6800 + 6809@248cd25 mutant catalogue (47 rows), as concrete source edits.

CATALOGUE = [(row, AB.7 0 mutant, stage, killing tests)] — stage "I" (orders 0, 0a–0f: the code built and run first)
or "II" (orders 3–10, plan "Staged build"). Rows 27 and 34 are split: their Stage I edits (dfin_floor_dropped,
other_block_seed) are here, their Stage II parts (yield2_dropped, the runner's screen spec) in LATER (Task 11b).
MUTANTS = [(rows, label, file, old, new, killing tests)]: `old` occurs exactly once in `file` (test_ab_static checks
it, so the catalogue follows the source); one edit may serve two rows (dfin_floor_dropped: rows 9 and 27).

The mutants are never applied to the worktree: `python -m tests.brain.ab_mutants <scratch>` exports HEAD's flymon /
tests / scripts / docs / pyproject.toml / uv.lock (plus the working-tree ab_mutants.py and test_ab_static.py) into
<scratch>, symlinks the worktree's data/ and results/ (read only: no killing test writes outside tmp_path), applies one
mutant at a time there and runs its killing tests from cwd <scratch> (so <scratch>/flymon is imported; no PYTHONPATH)
with PYTHONDONTWRITEBYTECODE=1; a mutant is killed iff the run fails (pytest exit 1).
Options: --only <comma-separated label substrings>, --stage I|II|all (default I), --python <interpreter>."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

E, P, SP, ST, R = ("flymon/brain/ab_estimate.py", "flymon/brain/ab_pairs.py", "flymon/brain/ab_spec.py",
                   "flymon/brain/ab_store.py", "flymon/brain/ab_runner.py")
TE, TC, TP, TS, TF, TB, TST, TX = ("tests/brain/test_ab_estimate.py", "tests/brain/test_ab_calibrate.py",
                                   "tests/brain/test_ab_pairs.py", "tests/brain/test_ab_spec.py",
                                   "tests/brain/test_ab_futility.py", "tests/brain/test_ab_runner_b.py",
                                   "tests/brain/test_ab_store.py", "tests/brain/test_ab_static.py")
TRC, TRD = "tests/brain/test_ab_runner_c.py", "tests/brain/test_ab_runner_d.py"      # Stage II (Task 9 / Task 10)
STAGES = ("I", "II")
II_ROWS = (3, 22, 23, 24, 26, 35, 36)


def _t(f, name):
    return f"{f}::{name}"


def _row(n):
    return "II" if n in II_ROWS else "I"


CATALOGUE = [   # (row, AB.7 0 mutant, stage, killing tests) — plan Task 11 table
    (1, "막대 척도 뒤집기", "I", [_t(TE, "test_constants_and_hedges_bar")]),
    (2, "IUT를 OR로", "I", [_t(TE, "test_iut_needs_every_gate_stat_method"),
                          _t(TE, "test_fixture_i_drift_without_association_not_pass")]),
    (3, "보정 수준 대신 95% 고정", "II", [_t(TRD, "test_verdict_reads_7b_levels")]),
    (4, "묶음 무시(쌍 단위 1단계를 1차로)", "I", [_t(TE, "test_ts_is_aa_two_stage_and_percentile")]),
    (5, "CG 상대 타입 집합 방향 삭제", "I", [_t(TE, "test_cg_hand_example_3x3")]),
    (6, "CG 양정치 보정 삭제", "I", [_t(TE, "test_cg_psd_branch_and_nan_pairs")]),
    (7, "CG df를 k − 1로", "I", [_t(TE, "test_cg_hand_example_3x3")]),
    (8, "TS 삭제", "I", [_t(TE, "test_iut_needs_every_gate_stat_method")]),
    (9, "D_fin 조건 삭제", "I", [_t(TE, "test_fixture_ix_two_inf_flies_d_passes_dfin_does_not"),
                             _t(TE, "test_dfin_k_and_group_floor")]),
    (10, "R 최소 효과를 0으로", "I", [_t(TE, "test_constants_and_hedges_bar"),
                                 _t(TE, "test_fixture_iv_small_constant_sd0_not_pass")]),
    (11, "R 조건을 두 연합만으로", "I", [_t(TE, "test_raw4_matches_aa_raw_contrasts_and_pair_stats"),
                                  _t(TE, "test_r_condition_on_all_four_gates")]),
    (12, "FAIL을 α_P에서", "I", [_t(TE, "test_fixture_xi_bonferroni_fail")]),
    (13, "FAIL을 한 방식만으로", "I", [_t(TE, "test_fail_needs_both_methods")]),
    (14, "보정을 방식마다 따로", "I", [_t(TC, "test_joint_miss_is_the_and_of_both_methods")]),
    (15, "선택 흐름으로 검증", "I", [_t(TC, "test_calibrate_phases_and_verdict_of_the_gate")]),
    (16, "CP 상한 대신 점 추정", "I", [_t(TC, "test_cp_bounds_match_scipy_and_spec_counts")]),
    (17, "참값을 δ × c(7)로", "I", [_t(TC, "test_truth_is_the_winsorized_expectation")]),
    (18, "보정 B를 2 000으로", "I", [_t(TC, "test_cal_run_uses_boot_b_and_its_own_stream")]),
    (19, "거울 칸 삭제", "I", [_t(TC, "test_rep_structures_and_cells")]),
    (20, "0e를 기록 전용으로", "I", [_t(TB, "test_cal_gate_stop_calibration")]),
    (21, "기계 대조 삭제", "I", [_t(TE, "test_fixture_viii_mechanism_and_machine")]),
    (22, "noplast 삭제", "II", [_t(TRD, "test_learn_noplast_mismatch_stop_machine")]),
    (23, "S1을 학습 값으로 고름", "II", [_t(TRD, "test_s1_never_changes_with_learning_values")]),
    (24, "학습 뒤 순진 값 재계산", "II", [_t(TRD, "test_s1_never_changes_with_learning_values")]),
    (25, "calibrate에 값 전달", "I", [_t(TC, "test_structure_only")]),
    (26, "관대 탈락쌍 학습", "II", [_t(TRD, "test_learn_s1_only_c_order_states_training_then_trained")]),
    (27, "k_min · g_min 생략", "I", [_t(TE, "test_dfin_k_and_group_floor")]),           # + LATER[27]
    (28, "lv_odour 생략 / AA 31쌍만", "I", [_t(TP, "test_generate_matches_every_declared_value"),
                                       _t(TP, "test_lv_sources")]),
    (29, "셔플 시드 · 기본 종 규칙 변경", "I", [_t(TS, "test_reuse_and_generator_facts"),
                                        _t(TP, "test_gen2_opponents_rule")]),
    (30, "대체 상대를 합친 목록에서", "I", [_t(TP, "test_generate_matches_every_declared_value")]),
    (31, "FAIL을 \"PASS 아님\"으로", "I", [_t(TE, "test_fixture_vii_one_gate_straddles_undecided")]),
    (32, "results/aa · AA 캐시 · results/y · results/z에 씀", "I",
     [_t(TST, "test_guard_refuses_outside"), _t(TX, "test_stage1_chain_leaves_others_untouched")]),
    (33, "몽키패치", "I", [_t(TX, "test_no_monkeypatching")]),
    (34, "다른 블록 시드", "I", [_t(TS, "test_ab_seed_blocks_collide_with_nothing_declared")]),  # + LATER[34]
    (35, "c 대신 S1 안의 번호", "II", [_t(TRD, "test_learn_s1_only_c_order_states_training_then_trained")]),
    (36, "라운드 시작에 trained", "II", [_t(TRD, "test_learn_s1_only_c_order_states_training_then_trained")]),
    (37, "문자열을 SeedSequence에", "I", [_t(TC, "test_cal_run_reproducible_resumable_and_monotone")]),
    (38, "P̂ 대신 CP 상한 · 하한으로 판단", "I", [_t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    (39, "판단 문턱을 0.5가 아닌 값으로", "I", [_t(TS, "test_futility_numbers"),
                                        _t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    (40, "0f를 기록 전용으로(STOP 없음)", "I", [_t(TB, "test_futility_stop_futile")]),
    (41, "0f를 순서 3 뒤로", "I", [_t(TS, "test_chain")]),
    (42, "판단 구조를 g 6이 아닌 구조로", "I", [_t(TS, "test_futility_numbers"),
                                        _t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    (43, "모의 PASS에서 D_fin · R · CG 가운데 하나를 뺌", "I",
     [_t(TF, "test_fixture_all_source_flies_infinite_never_passes"),
      _t(TF, "test_fixture_level_gate_r_small_never_passes"), _t(TF, "test_pass_needs_both_methods")]),
    (44, "관문을 서로 독립으로 생성(C 무시)", "I", [_t(TF, "test_draw_uses_c_and_the_allocation")]),
    (45, "원천 마리를 뽑지 않고 정규 잡음으로 대체", "I", [_t(TF, "test_flies_shift_source_residuals_and_keep_infinities")]),
    (46, "0e 수준 대신 고정 α", "I", [_t(TF, "test_decision_point_estimate_strict_and_on_g6_icc"),
                                 _t(TB, "test_futility_pass_block")]),
    (47, "격자 결과로 AB 수치 · 순서를 바꿈", "I", [_t(TB, "test_futility_pass_block")]),
]
LATER = {   # the Stage II parts of rows 27 and 34 (Task 11b writes their edits)
    27: ("yield2_dropped", [_t(TRC, "test_screen_yield_two_stop")]),
    34: ("screen ab_w_spec(s) → aa_runner.aa_w_spec()", [_t(TRC, "test_screen_s1_manifest_candidates_and_seeds_use_c")]),
}

_IUT = 'ok = all(r[n]["ok_TS"] and r[n]["ok_CG"] and r[n]["fin_ok"] for r in gates.values() for n in STATS)'
_FUT_DEC = 'futile = _r(j["p_hat"] - s.fut_threshold, s) < 0'
_FUT_PASS = '    return v["label"] == PASS, v["gates"]'
_CAL_PASS = '        if not why:\n            return self._write("cal_gate", dict(common, outcome=ab_rules.PASS'
_FUT_STOP = '        if not j["futile"]:\n            return self._write("futility", dict(common, outcome=ab_rules.PASS'
_FUT_ALPHA = '        alpha = {t: cal[t]["alpha"] for t in s.fut_tags}'
_GRID_IF = '        if j["futile"]:\n            tick()                                            # the core part'
_BSTOP = "class _BudgetStop(Exception):"

MUTANTS = [   # (rows, label, file, old, new, killing tests) — the plan's validated pure-module edits + Stage I runner
    ((1,), "hedges_flip", E, 'return 1.0 if name == "R" else hedges(s)', 'return 1.0',
     [_t(TE, "test_constants_and_hedges_bar")]),
    ((2,), "iut_or", E, _IUT, 'ok = any(r[n]["ok_TS"] or r[n]["ok_CG"] for r in gates.values() for n in STATS)',
     [_t(TE, "test_iut_needs_every_gate_stat_method"), _t(TE, "test_fixture_i_drift_without_association_not_pass")]),
    ((4,), "groups_ignored", E, '    groups = y_rules.merge_groups(list(keys))\n    out = {}',
     '    groups = [[i] for i in range(len(keys))]\n    out = {}', [_t(TE, "test_ts_is_aa_two_stage_and_percentile")]),
    ((5,), "cg_one_way", E, 'vs = max(v, vx, vt)', 'vs = vx', [_t(TE, "test_cg_hand_example_3x3")]),
    ((6,), "cg_no_psd", E, 'vs = max(v, vx, vt)', 'vs = v', [_t(TE, "test_cg_psd_branch_and_nan_pairs")]),
    ((7,), "cg_df_k", E, 'df=min(Gx, Gt) - 1, k=k', 'df=k - 1, k=k', [_t(TE, "test_cg_hand_example_3x3")]),
    ((8,), "ts_dropped", E, 'ok_ts, ok_cg = beyond(ts_v, sign, name, s), beyond(cg_v, sign, name, s)',
     'ok_ts = ok_cg = beyond(cg_v, sign, name, s)', [_t(TE, "test_iut_needs_every_gate_stat_method")]),
    ((9,), "dfin_dropped", E, _IUT,
     'ok = all(r[n]["ok_TS"] and r[n]["ok_CG"] for r in gates.values() for n in ("D", "R"))',
     [_t(TE, "test_fixture_ix_two_inf_flies_d_passes_dfin_does_not")]),
    ((10,), "raw_min_zero", E, 'return s.raw_min if name == "R" else s.bar', 'return 0.0 if name == "R" else s.bar',
     [_t(TE, "test_constants_and_hedges_bar"), _t(TE, "test_fixture_iv_small_constant_sd0_not_pass")]),
    ((11,), "raw_assoc_only", E,
     '    return np.stack([r1.mean(-1), (r2 - r1).mean(-1), (r1 - n1).mean(-1), ((r2 - r1) - (rn2 - rn1)).mean(-1)], -1)',
     '    big = np.full(r1.shape[:-1], 1e9)\n'
     '    return np.stack([big, -big, (r1 - n1).mean(-1), ((r2 - r1) - (rn2 - rn1)).mean(-1)], -1)',
     [_t(TE, "test_raw4_matches_aa_raw_contrasts_and_pair_stats")]),
    ((12,), "fail_at_alpha_p", E, 'j = L["levels"].index(alpha["F"])', 'j = L["levels"].index(alpha["D"])',
     [_t(TE, "test_fixture_xi_bonferroni_fail")]),
    ((13,), "fail_one_method", E, 'both=short_of(f_ts, sign, s) and short_of(f_cg, sign, s)',
     'both=short_of(f_ts, sign, s) or short_of(f_cg, sign, s)', [_t(TE, "test_fail_needs_both_methods")]),
    ((14,), "cal_per_method", E, 'row = dict(P=(up_ts & up_cg)[:n_p]', 'row = dict(P=up_ts[:n_p]',
     [_t(TC, "test_joint_miss_is_the_and_of_both_methods")]),
    ((15,), "ver_on_sel_stream", E, '("ver", (items[i][0], items[i][1], "ver", ci,',
     '("ver", (items[i][0], items[i][1], "sel", ci,', [_t(TC, "test_calibrate_phases_and_verdict_of_the_gate")]),
    ((16,), "cp_point", E, 'return 1.0 if x >= n else float(stats.beta.ppf(s.cp_level, x + 1, n - x))',
     'return x / n', [_t(TC, "test_cp_bounds_match_scipy_and_spec_counts")]),
    ((17,), "truth_unclipped", E, '    return {k: acc[k] / cnt[k] for k in STATS}',
     '    return dict(D=delta(s) * AE.c_k(s.probes), Dfin=delta(s) * AE.c_k(s.probes), R=delta(s))',
     [_t(TC, "test_truth_is_the_winsorized_expectation")]),
    ((18,), "cal_b_2000", E, '= limits(st3[name][..., None], groups, gx, gt, lv, r, s, B)',     # rep_misses only
     '= limits(st3[name][..., None], groups, gx, gt, lv, r, s, 2_000)',
     [_t(TC, "test_cal_run_uses_boot_b_and_its_own_stream")]),
    ((19,), "mirror_cells", SP, '("skX", 1.0), ("skX", -1.0),', '("skX", 1.0), ("skX", 1.0),',
     [_t(TC, "test_rep_structures_and_cells")]),
    ((20,), "cal_gate_record_only", R, _CAL_PASS, _CAL_PASS.replace("if not why:", "if True:"),
     [_t(TB, "test_cal_gate_stop_calibration")]),
    ((21,), "mech_dropped", E, '    bad = mech_bad(mech, s)\n', '    bad = []\n',
     [_t(TE, "test_fixture_viii_mechanism_and_machine")]),
    ((25,), "cal_values_arg", E, 'def calibrate(st, tag: str, s=AB, cell=None) -> dict:',
     'def calibrate(st, tag: str, s=AB, cell=None, values=None) -> dict:', [_t(TC, "test_structure_only")]),
    ((25,), "cal_many_values_arg", E, 'def calibrate_many(items: list, s=AB, cell=None) -> list:',
     'def calibrate_many(items: list, s=AB, cell=None, values=None) -> list:', [_t(TC, "test_structure_only")]),
    ((9, 27), "dfin_floor_dropped", E,
     'fin_ok = (L["k_fin"][gi] >= s.k_min and L["gx_fin"][gi] >= s.g_min and L["gt_fin"][gi] >= s.g_min)',
     'fin_ok = True', [_t(TE, "test_dfin_k_and_group_floor")]),
    ((28,), "lv_skip_off", P, 'if why is None and lv is not None and any(okey(m, o) in lv for m, o in v_pairs.sides(r)):',
     'if False:', [_t(TP, "test_generate_matches_every_declared_value")]),
    ((28,), "lv_aa31_only", P, '    rows = ([wk[k] for k in trained if k in wk] + [ek[k] for k in w_pilot if k in ek]',
     '    rows = ([wk[k] for k in trained if k in wk] + 0 * [ek[k] for k in w_pilot if k in ek]',
     [_t(TP, "test_lv_sources")]),
    ((29,), "shuffle_seed", SP, 'shuffle_seed: int = 20261007', 'shuffle_seed: int = 20261006',
     [_t(TS, "test_reuse_and_generator_facts"), _t(TP, "test_generate_matches_every_declared_value")]),
    ((29,), "base_species_rule", P, '        if "baseSpecies" in v and to_id_str(v["baseSpecies"]) != k:\n'
     '            continue\n', '', [_t(TP, "test_gen2_opponents_rule")]),
    ((30,), "alt_union", P, '        self.alts = list(self.names)', '        self.alts = list(base.names) + list(self.names)',
     [_t(TP, "test_generate_matches_every_declared_value")]),
    ((31,), "fail_as_not_pass", E, '    return dict(label=UNDECIDED, causes=causes, gates=gates, fail_uncal=probe)',
     '    return dict(label=FAIL, causes=causes, gates=gates, fail_uncal=probe)', [_t(TE, "test_fixture_vii_one_gate_straddles_undecided")]),
    ((32,), "store_allows_all", ST, 'ALLOWED_DIR = "results/ab/"', 'ALLOWED_DIR = "results/"',
     [_t(TST, "test_guard_refuses_outside")]),
    ((33,), "monkeypatch_inserted", R, _BSTOP,
     'setattr(w_verdict, "GATES", w_verdict.GATES)\n\n\n' + _BSTOP, [_t(TX, "test_no_monkeypatching")]),
    ((34,), "other_block_seed", SP, 'probe_seed0: int = 88_100_000', 'probe_seed0: int = 62_000_000',
     [_t(TS, "test_ab_seed_blocks_collide_with_nothing_declared")]),
    ((37,), "string_seedsequence", E, '    r = AE.stream(s.cal_seed, "cal", tag, phase, int(ci))',
     '    r = np.random.default_rng(np.random.SeedSequence([s.cal_seed, "cal", tag, phase, int(ci)]))',
     [_t(TC, "test_cal_run_reproducible_resumable_and_monotone")]),
    # ---- AB.9.3 (rows 38–47)
    ((38,), "fut_cp_upper", E, _FUT_DEC, 'futile = _r(j["cp95"][1] - s.fut_threshold, s) < 0',
     [_t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    ((38,), "fut_cp_lower", E, _FUT_DEC, 'futile = _r(j["cp95"][0] - s.fut_threshold, s) < 0',
     [_t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    ((39,), "fut_le", E, _FUT_DEC, 'futile = _r(j["p_hat"] - s.fut_threshold, s) <= 0',
     [_t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    ((39,), "fut_threshold", SP, 'fut_threshold: float = 0.5 ', 'fut_threshold: float = 0.4 ',
     [_t(TS, "test_futility_numbers")]),
    ((40,), "fut_record_only", R, _FUT_STOP, _FUT_STOP.replace('if not j["futile"]:', "if True:"),
     [_t(TB, "test_futility_stop_futile")]),
    ((41,), "fut_after_kc", SP, '"cal_gate", "futility", "kc_input", "set",', '"cal_gate", "kc_input", "futility", "set",',
     [_t(TS, "test_chain")]),
    ((42,), "fut_judge_g7", SP, 'fut_judge: tuple = ("g6", "icc")', 'fut_judge: tuple = ("g7", "icc")',
     [_t(TS, "test_futility_numbers"), _t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    ((43,), "fut_ts_only", E, _FUT_PASS,
     '    return all(r[n]["ok_TS"] and r[n]["fin_ok"] for r in v["gates"].values() for n in STATS), v["gates"]',
     [_t(TF, "test_pass_needs_both_methods")]),
    ((43,), "fut_no_dfin", E, _FUT_PASS,
     '    return all(r[n]["ok_TS"] and r[n]["ok_CG"] for r in v["gates"].values() for n in ("D", "R")), v["gates"]',
     [_t(TF, "test_fixture_all_source_flies_infinite_never_passes")]),
    ((43,), "fut_no_r", E, _FUT_PASS,
     '    return all(r[n]["ok_TS"] and r[n]["ok_CG"] and r[n]["fin_ok"] for r in v["gates"].values() '
     'for n in ("D", "Dfin")), v["gates"]', [_t(TF, "test_fixture_level_gate_r_small_never_passes")]),
    ((44,), "fut_no_corr", E, '    L = np.asarray(inp["chol"], float)\n    tau, rho',
     '    L = np.eye(len(inp["chol"]))\n    tau, rho', [_t(TF, "test_draw_uses_c_and_the_allocation")]),
    ((45,), "fut_normal_flies", E,
     '    new = np.where(inf, src, th[:, None, :4] + (AE.winsorize(src, s) - mD[q][:, None, :]))',
     '    new = np.where(inf, src, th[:, None, :4] + np.random.default_rng(0).standard_normal(src.shape))',
     [_t(TF, "test_flies_shift_source_residuals_and_keep_infinities")]),
    ((46,), "fut_fixed_alpha", E, 't, a, inp, alpha_by_tag[t], s)) for t, a in rows]',
     't, a, inp, {"D": {"P": 0.025}, "Dfin": {"P": 0.025}, "R": {"P": 0.025}}, s)) for t, a in rows]',
     [_t(TF, "test_decision_point_estimate_strict_and_on_g6_icc")]),
    ((46,), "fut_runner_fixed_alpha", R, _FUT_ALPHA,
     '        alpha = {t: {"D": {"P": 0.025, "F": None}, "Dfin": {"P": 0.025}, "R": {"P": 0.025}} '
     'for t in s.fut_tags}', [_t(TB, "test_futility_pass_block")]),
    ((47,), "grid_on_pass", R, _GRID_IF, _GRID_IF.replace('if j["futile"]:', "if True:"),
     [_t(TB, "test_futility_pass_block")]),
]


def stage_of(m) -> str:
    """An edit is Stage II iff every row it serves is a Stage II row (Stage II parts of rows 27 / 34 come in 11b)."""
    return "II" if all(_row(n) == "II" for n in m[0]) else "I"


def export(dest: Path, root: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    arc = subprocess.run(["git", "-C", str(root), "archive", "HEAD", "flymon", "tests", "scripts", "docs",
                          "pyproject.toml", "uv.lock"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc, check=True)
    for f in ("tests/brain/ab_mutants.py", "tests/brain/test_ab_static.py"):
        if (root / f).exists():
            shutil.copy2(root / f, dest / f)
    for d in ("data", "results"):                     # read only: the killing tests write under tmp_path
        (dest / d).symlink_to(root / d)


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return dict(env, PYTHONDONTWRITEBYTECODE="1")


def check_import(dest: Path, py: str) -> str:
    """The scratch copy's flymon is the one imported from cwd <dest>."""
    r = subprocess.run([py, "-c", "import flymon.brain.ab_runner as m; print(m.__file__)"], cwd=dest, env=_env(),
                       capture_output=True, text=True, check=True)
    got = Path(r.stdout.strip()).resolve()
    assert got == (dest / R).resolve(), (got, dest)
    return str(got)


def run(dest: Path, py: str, only: str | None, stage: str, log) -> list:
    out = []
    env = _env()
    for rows, label, f, old, new, tests in MUTANTS:
        if stage != "all" and stage_of((rows,)) != stage:
            continue
        if only and not any(o in label for o in only.split(",")):
            continue
        p = dest / f
        src = p.read_text()
        assert src.count(old) == 1, (label, src.count(old))
        p.write_text(src.replace(old, new))
        t0 = time.time()
        try:
            r = subprocess.run([py, "-m", "pytest", *tests, "-o", "addopts=", "-q", "-x", "-p", "no:cacheprovider",
                                "--color=no"],
                               cwd=dest, env=env, capture_output=True, text=True)
        finally:
            p.write_text(src)
        killed = r.returncode == 1
        lines = r.stdout.strip().splitlines() or ["?"]
        why = next((x for x in lines if x.startswith("E ")), "")[:160]           # the first assertion line
        line = (f"{'KILLED ' if killed else 'SURVIVED'} rc={r.returncode} {time.time() - t0:5.1f}s {label} rows "
                f"{list(rows)} :: {lines[-1]} :: {why}")
        print(line, file=log, flush=True)
        out.append((label, killed, r.returncode))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--only")
    ap.add_argument("--stage", default="I", choices=("I", "II", "all"))
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--python", default=str(root / ".venv/bin/python"))
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    export(dest, root)
    print(f"scratch import: {check_import(dest, a.python)}", flush=True)
    res = run(dest, a.python, a.only, a.stage, sys.stdout)
    n = sum(k for _l, k, _r in res)
    print(f"mutants killed {n} / {len(res)}; survivors: {[l_ for l_, k, _r in res if not k]}", flush=True)
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
