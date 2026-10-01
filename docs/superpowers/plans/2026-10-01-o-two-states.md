# Spec O — C3's Two States (O1) and Presentation-Evoked Depression (O2) on the Real-Odour Rig: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix O (as overridden by O.7) needs, and nothing else:
- O1: the two-state map of C3 on N's real-odour rig — 4 conditions × 6 g × 5 stimuli × 64 seeds = 7 680 presentations — with the mixed-cell existence rule, the nature of the two states (P mid band, APL / KC AUCs), the pathway 2×2 (APL→KC vs APL→non-KC), per-stimulus drive dependence, and the records;
- O2: presentation-evoked depression at N.8a's operating point — 2 X × 4 arms × 32 seeds = 256 arms — with the INCONCLUSIVE bands, the dopamine-zero arm, the associative component, the state-flip guard and the sha256 plumbing check;
- both behind a validity gate, per-item checkpoint files, and CLIs with `--smoke`. The controller runs the real stages afterwards (section "Runs (controller)").

This is a characterisation, **not a learning claim** (O.1). No verdict here changes N.9, the re-scope closure or the Pokémon claim.

**Architecture:**
- New files only. N's modules are imported, never edited (O.5).
  - `o_spec` holds every O number; N's numbers are read through `spec.n` (an `NSpec`).
  - `o_jobs` holds O's own rig cache (N's three edits + the new `apl_to_nonkc_zero`), the O1 presentation job, and the O2 training loop / arm job (punish-injection flag, dopamine-zero arm, phasic-dopamine meter).
  - `o_measure` is O's write guard (`results/o/`, `results/summary/o_states.json`), its content-addressed cache (one file per item, with code key and spec commit) and `OMeasurer` (a subclass of `n_measure.NMeasurer`, reusing its round-per-worker `_items`).
  - `o_rules` holds the pure rules: validity gate, seed-cluster bootstrap, O1 and O2 judgements, records, sentences.
  - `o_cli` holds the refusals and the stage skeleton the two scripts share.
- Two stage scripts: `scripts/run_o1.py`, `scripts/run_o2.py`. O1 and O2 are independent (O.5): neither needs the other's block.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`; `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **O** (O.1–O.6) and its red-team amendment **O.7** (O.7.1–O.7.7), which wins wherever they differ:

| O.7 section | Replaces / adds |
|---|---|
| O.7.1 | fourth O1 condition `apl_to_nonkc_zero`; partition test; 7 680 presentations |
| O.7.2 | O1 judgements 1–4 replaced: mixed cells (`STOP_NO_SILENT_STATE` if < 2), state nature (`BISTABLE_NETWORK` / `BIMODAL_P_READOUT` / `GRADED`), pathway 2×2, per-stimulus drive labels with 99% slope CIs; all CIs seed-cluster bootstrap |
| O.7.3 | one O training loop with a punish flag (flag on ≡ `n_jobs.absolute_arm_job`, bit-identical); arms ①–④ incl. the dopamine-zero arm; phasic dopamine integral and `weights_frac_by_mbon_set` records |
| O.7.4 | O2 judgements with `INCONCLUSIVE` bands, `DA_DEPENDENT` / `DA_INDEPENDENT`, e from N.9's o, state-flip guard, sha256 plumbing |
| O.7.5 | validity gate, per-item atomic files with code key + spec commit, `n_rules.state` / `cell_stats` reused |
| O.7.6–O.7.7 | disclosure; cost (O1 ≈ 25 min, O2 ≈ 15 min at 16 workers) |

## Global Constraints

- **Commits carry no trailers of any kind.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No existing file changes (O.5).** Every existing module, script, test and `.gitignore` stays byte-identical — in particular `n_jobs.py` (its `EDITS`, `apply_edit`, `rig`, `_RIG`), `n_spec.py`, `n_rules.py`, `n_measure.py`, `n_cli.py`, `n_store.py`, `odor_real.py`, `plasticity.py`, `conditioning.py`, `fly_pool.py`. O only adds the files in "File Structure".
- **One configuration object.** Every O number is a field of `flymon.brain.o_spec.SPEC` (`OSpec`). N's numbers (g grid, stimuli and mixtures, Hallem pins, probe windows, training timings, train-seed rule, bootstrap draws and seed, the state threshold P < 5) are read through `SPEC.n` or assigned from `N_SPEC` in the field default — **never restated as a literal**. `o_rules`, `o_jobs`, `o_measure`, `o_cli` and the scripts contain no threshold literal.
- **Numbers (verbatim from O / O.7):**
  - **State** (O.1 = N.8.3): P(MBON05) < 5 silent, ≥ 5 firing — `n_rules.state(P, spec.n)`, never redefined (O.7.5).
  - **O1 grid:** g ∈ N0f's grid {0.125, 0.25, 0.5, 1, 2, 4} × stimuli {4:1, 1:4, δ-DL, IA, EB} (c_δ = 8 on δ-DL only) × conditions {on `none`, APL→KC `apl_to_kc_zero`, APL→non-KC `apl_to_nonkc_zero`, all-output `apl_all_zero`}. Seeds **22_000_000 + i, i < 64**. 7 680 presentations; N0f's `presentation_job` procedure and windows, scalars only.
  - **O1 judgements (O.7.2):** all CIs seed-cluster bootstrap (64 seeds resampled with replacement, every cell and condition of a seed together), 10 000 draws, percentile, seed `n_spec.boot_seed`.
    1. Mixed cell: on-condition (g, stimulus) cell with silent share **and** firing share both ≥ 0.10 (≥ 7 of 64). Fewer than **2** → `STOP_NO_SILENT_STATE`.
    2. Per mixed cell: (a) share of presentations with P ∈ [5, 20) ≤ 0.05; (b) AUC of `apl_out_per_step` silent vs firing ≥ 0.9; (c) AUC of KC spike count ≥ 0.9. (a)(b)(c) in ≥ 2/3 of mixed cells → `BISTABLE_NETWORK`; else (a) in ≥ 2/3 → `BIMODAL_P_READOUT`; else `GRADED`.
    3. Pooled silent share over mixed cells q_on, q_kc, q_nonkc (same seeds, same cells). Partial block c removes silence ⇔ q_c ≤ 0.25 · q_on **and** 95% CI upper of (q_c − q_on) < 0. Non-KC only → `READOUT_PATH`; KC only → `KC_NETWORK_PATH`; both → `BOTH_PATHS`; neither → `NEITHER_PATH`. All-output block recorded, not judged.
    4. Per stimulus (all five), on condition: least-squares slope of cell silent share on g rank (0–5) and range (max − min). Range < 0.15 → `FLAT`; else 99% slope CI above 0 → `INCREASING`, below → `DECREASING`, containing 0 → `NONMONOTONIC_OR_UNCLEAR`. Labels per stimulus, not pooled.
  - **O1 records:** APL→KC block's same numbers; seed-wise φ of co-silence over on-cell pairs with silent share in [0.1, 0.9]; per-cell P / APL / KC distributions by state; the mixed-cell list.
  - **O2 setup:** N.8a point g = 0.25, c_δ = 8, edit `none`; X ∈ {4:1 (Y = 1:4), δ-DL (Y = 4:1)}. Seeds **22_001_000 + i, i < 32**. N2's absolute arm: `train_plus_only` timings (`h4.teach_trials`, `teach_present_ms`, `teach_gap_ms`, `teach_window.settle_ms`), PPL105 punishment, train seed `n_spec.train_seed(seed, trial, 1_000_000, 1000)`.
  - **O2 arms (O.7.3):** ① `plastic` — no punish injection, plasticity on; ② `frozen` — no punish injection, plasticity off (plumbing); ③ `punish` — punishment, plasticity on (≡ N2's arm); ④ `da_zero` — no punish injection, plasticity on, the rule's dopamine trace held at 0 every step, DAN firing untouched.
  - **O2 judgements (O.7.4)**, per X, seed bootstrap 95% (10 000, percentile, `boot_seed`):
    1. m = mean_seed(ΔA_X[①] − ΔA_X[②]), b = 0.25 · mean(① pre A_X). CI upper < 0 and m ≤ −b → `DEPRESSION_PRESENT`; CI ⊂ [−b, b] → `NO_DEPRESSION`; else `INCONCLUSIVE`.
    2. Only if 1 is `DEPRESSION_PRESENT`: m₄ = mean(ΔA_X[④] − ΔA_X[②]). CI ⊂ [−b, b] → `DA_DEPENDENT`; m₄ ≤ −b and CI upper < 0 → `DA_INDEPENDENT`; else `INCONCLUSIVE`.
    3. D = mean(Δ(dV)[③] − Δ(dV)[①]), e = 0.25 · |o|, o = N.9's oracle raw effect (X = 4:1 → sim −2.348; X = δ-DL → dis −2.433). CI excludes 0 and |D| ≥ e → `SEPARABLE`; CI ⊂ [−e, e] → `NOT_SEPARABLE`; else `INCONCLUSIVE`. Sign recorded.
    4. State-flip guard: per arm, seeds whose X state differs pre vs post. Any arm > 1/8 of seeds (4 of 32) → the same judgement without the flipped seeds beside it; a label that changes gets `_STATE_FLIP_SENSITIVE`.
    5. Plumbing: ② post-training weight sha256 == w0's, bit-identical, every seed; else that X is `INVALID`.
  - **Validity gate (O.7.5)** before any judgement: declared Cartesian product complete, keys unique, all values finite, no undeclared seed, one CSC sha256 per edit and edits mutually distinct. Any failure → that stage `INVALID` with reasons, no judgement.
  - **Smoke** seed block 22_009_xxx (O.5); never a declared stage's.
- **Every output path is guarded.** Raw rows and caches under `results/o/` (git-ignored by the existing `.gitignore` rule `results/*`, which re-includes only `results/summary/` — no `.gitignore` edit is needed); the summary is `results/summary/o_states.json` (smoke: `results/o/smoke/o_states.json`). Everything is written through `o_measure`'s guard; anything else is refused (SystemExit 2).
- **Worker jobs and pools run only from script files with `if __name__ == "__main__":` or from pytest**, never from a heredoc (spawn re-imports the parent — a fork bomb).
- **Subagents never write under `results/` and never start a real run or a smoke run on the connectome.** Tests write only under `tmp_path`. The controller runs every real step (section "Runs (controller)").
- **Tests locate the repository via `Path(__file__)`** (`Path(__file__).resolve().parents[1]` in `tests/`, `parents[2]` in `tests/brain/`), never via the cwd.
- **Full suite before each commit:** `uv run pytest -q -rfE -o addopts=""`.
  - Baseline at `be0ca3d`: **1640 passed / 1 skipped / 3 xfailed**.
  - Known flake: `tests/test_run_m0d_h3.py::test_a_complete_run_writes_everything_through_both_guards_and_resumes_from_the_cache`. Rerun that file once.
  - Never run two suites at once.
- **Style:** match `n_*`. Module docstring citing the spec section, dense one-line comments, `from __future__ import annotations`.

## Readings of the spec (decided here — the controller's rulings)

1. **File set.** O.5 lists `o_spec`, `o_jobs`, `o_rules`, `run_o1`, `run_o2`. The write guard, cache and measurer go in `o_measure.py`, and the stage skeleton goes in `o_cli.py`. There is no `o_store.py`; the guard lives in `o_measure`. N's `n_store` / `n_cli.main_stage` cannot be reused: they are hard-wired to `results/n/` and N's block order.
2. **The new edit and its rig.** `n_jobs.apply_edit` rejects any edit outside N's `EDITS`, and `n_jobs.rig` calls it. So O has its own `o_jobs.EDITS = n_jobs.EDITS + ("apl_to_nonkc_zero",)`, its own `apply_edit` and its own `_RIG` cache under `(Params, edit)`.
   - `o_jobs.apply_edit` delegates N's three edits to `n_jobs.apply_edit` unchanged.
   - It implements `apl_to_nonkc_zero` as "every CSC edge whose source is APL and whose target is not a KC set to 0", which includes APL→MBON05.
   - The graded-APL out-edge views follow, because they are views into `csc.w`.
   - O1's presentation job is `n_jobs._present` driven on O's rig, so its rows are N0f's rows. A test pins row equality with `n_jobs.presentation_job` for N's edits.
3. **O1 mixed-cell threshold.** "둘 다 ≥ 0.10 (64시드 중 7 이상)" is coded as share ≥ `mixed_min` on both sides (7/64 = 0.109 passes, 6/64 = 0.094 fails).
4. **`STOP_NO_SILENT_STATE` scope.** O.2.1 says judgements 2–4 are not made, and O.7.2 does not revoke that. On STOP the block carries every record (per-cell distributions, φ pairs, mixed-cell list, the APL→KC block record) but `nature`, `pathway` and `drive` are `None`.
   - Judgement 4 is per stimulus over all six g of the on condition. It is not restricted to mixed cells, because a slope needs every g. "다음 판정은 혼합 칸에서만" is read as applying to 2 and 3.
5. **AUC direction.** O.7.2 gives no direction. (b) and (c) use the separation AUC max(U, 1 − U), where U = P(silent value > firing value) + ½ P(tie). The directed U is recorded as `auc_*_silent_above`.
6. **"(a)만 2/3 이상"** is read as "(a) holds in ≥ 2/3 of mixed cells". It is checked only after `BISTABLE_NETWORK` has failed. The comparison is integer: `3·k ≥ 2·n`.
7. **Pathway statistic.** The pooled share q_c is the mean over (mixed cell, seed). Each bootstrap draw uses one seed resample for every condition and cell (seed-cluster), and the CI is taken on q_c,b − q_on,b.
8. **Slope bootstrap.**
   - The rank is the position in ascending `o1_g_grid` (a test asserts the grid is ascending).
   - The range is taken on the point estimates, and the 99% CI is a percentile CI of the bootstrap slopes.
9. **Arms without punishment.** The arm skips both `drive_dan` and `recover_pulse`. This is `conditioning.train_block`'s rule for `dan is None`. With punishment, the loop is `n_jobs.train_plus_only` call for call, which the regression test pins bit-for-bit.
10. **Dopamine-zero arm (④).** During training, every DAN type's MBON weight vector in the plasticity rule (`Plasticity.types[name][1]`) is replaced by zeros. That keeps `p.da` (and `p.da_base`) exactly 0 at every step. DAN cells, their drive and their spikes are untouched, because `drive_dan` / `quiet_dan` read only the cells. The vectors are restored afterwards (test).
    - **Consequence the controller should know:** `Plasticity.on_step` changes weights only through phasic dopamine, and recovery runs only after a pulse. So arm ④ cannot move any weight, and is expected to equal ② bit-for-bit: m₄ = 0 and CI = [0, 0].
    - Hence whenever judgement 1 is `DEPRESSION_PRESENT`, judgement 2 is `DA_DEPENDENT` **by construction of the rule**. The arm runs as declared, and the block records `da_zero_weights_unmoved`.
11. **Phasic dopamine integral.** It is ∫ max(da − da_base, 0) dt, accumulated over the present + gap window of every training trial (the steps where the rule can act; settle excluded). It is reported per DAN-type compartment as the mean over that compartment's core MBONs.
12. **`weights_frac_by_mbon_set` (A·P core)** is read as the readout's A cells (MBON13) and P cells (MBON05). It is measured after training, next to `weights_frac`.
13. **o for e.**
    - O2 uses O.7.4's declared o (`o_spec.oracle_o`).
    - Outside `--smoke`, `run_o2` first requires `results/summary/n_real_odour.json` to be committed. Its block `n1` exact `o` must agree within 5·10⁻⁴, else the run is refused (exit 2). Both values are recorded.
    - `--smoke` uses the declared values when the block is unreadable.
14. **State-flip guard.**
    - The limit is `count > flip_max_frac · n` (n = 32 → more than 4).
    - The dropped seeds are the union over the four arms, removed from every arm (the statistics are paired).
    - Each of the three labels is compared separately.
    - If fewer than 2 seeds remain, the alternative is `None` and every non-`None` label is suffixed.
15. **INVALID scope.** The validity gate failing makes the whole stage `INVALID` (no X judged). The plumbing check failing makes only that X `INVALID`. Exit 0 needs a judged stage with no X `INVALID`; otherwise exit 5.
16. **"훈련 중 프로브"** is not in the rig: there is no probe inside N2's training loop. The A_X course over presentations is recorded as absent, with that note.
17. **State-conditional record (O2).** It covers the seeds whose every probe (pre / post × X / Y × all four arms) is firing, and repeats the statistics on them as a record (labels included, but not judged).
18. **No post-hoc rewrite.** Outside `--smoke`, a stage refuses (exit 2) when the real summary already holds its block. The block is written only once the stage ends, so an interrupted run resumes from the cache files.
19. **Exit codes.** 0 = judged; 5 = `STOP_NO_SILENT_STATE` or `INVALID` (block written; the controller reports to the user); 2 = refusal (nothing written).
20. **Smoke.** `o_spec.smoke` changes only the scale, never a threshold:
    - O1: 8 seeds from 22_009_000 on two g (`o1_g_grid[1:3]`);
    - O2: 4 seeds from 22_009_100;
    - 200 bootstrap draws.
21. **Per-item files.** One content-addressed cache file per (O1, condition, cell, seed) and per (O2, X, arm, seed). The key covers the Params, edit, odour, seed, windows and flags. Each file holds `code_key` and `spec_commit` (the last commit touching the spec), written atomically.

## Review Focus

1. **The da-zero arm leaking into the next arm on the same worker.** If `Plasticity.types` or `Engine.on_step` stayed swapped in the cached rig, every later arm on that worker would silently learn nothing, or keep integrating. Expect a punish arm run after a da-zero arm in one worker to equal a fresh-rig punish arm, and `e.on_step == p.on_step` after every job. Test in Task 3.
2. **Mixed N and O rigs in one worker.** O1 alternates N's edits and the new one on the same workers. Expect on → non-KC block → on to give identical first and third rows. A non-KC block job must equal a fresh rig's, and O's cache must never touch `n_jobs._RIG`. Test in Task 2.
3. **Degenerate inputs at the statistics edges.** Examples: an O2 pre-A_X mean of 0 (b = 0), a flip guard that leaves < 2 seeds, a mixed cell exactly at 0.10, a stimulus whose g-shares are non-monotone with zero slope. Expect a label (`NO_DEPRESSION`, suffixed labels, mixed, `NONMONOTONIC_OR_UNCLEAR`), never a crash or a false pass. Test in Tasks 5 and 6.
4. **Interrupted runs and resume.** A run killed mid-stage leaves some item files. Expect a rerun to start jobs only for the missing items, and a resumed-from-cache run to start no pool job at all. Once the block is written, the real stage refuses to rewrite it. Test in Tasks 4 and 7.
5. **Non-finite or duplicated rows reaching a rule.** Examples: a NaN APL value, a cache file read twice into the row list, a seed outside the block, two CSC vectors under one edit, an edit that changed nothing. Expect `INVALID` with a reason naming each, before any judgement. Test in Tasks 5 and 6.

---

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/o_spec.py` | `OSpec` (every O number, seed blocks, arms, smoke scale) + `smoke()` |
| `flymon/brain/o_jobs.py` | `NONKC`, `EDITS`, `_RIG`, `apply_edit`, `rig`, `presentation_job`, `DaMeter`, `dopamine_zero`, `train_x`, `weights_sha256`, `arm_job` |
| `flymon/brain/o_measure.py` | `ALLOWED_DIR`, `SUMMARY`, `MEASURE_FILES`, `HASHED_FILES`, `guard`, `write_bytes`, `write_json`, `write_summary_block`, `OCache`, `OMeasurer` |
| `flymon/brain/o_rules.py` | outcomes and labels, `finite`, `validity`, `boot_weights`, `auc`, O1 (`o1_cells`, `cell_key`, `o1_key`, `o1_declared`, `silent`, `mixed_cells`, `cell_nature`, `nature_label`, `pathway`, `drive`, `phi_pairs`, `cell_record`, `o1_judge`), O2 (`o2_key`, `o2_declared`, `o2_stats`, `o2_x`, `o2_judge`), `sentence` |
| `flymon/brain/o_cli.py` | hooks (`code_keys`, `git_state`, `load_c3`, `m0d_sha`, `make_measurer`, `model_types`, `spec_commit`, `n_oracle`), `out_allowed`, `nview`, `write_report`, `Ctx`, `main_stage` |
| `scripts/run_o1.py`, `scripts/run_o2.py` | the two stages |
| `tests/brain/test_o_spec.py`, `tests/brain/test_o_jobs.py`, `tests/brain/test_o_jobs_arms.py`, `tests/brain/test_o_measure.py`, `tests/brain/test_o_rules_o1.py`, `tests/brain/test_o_rules_o2.py`, `tests/test_run_o.py` | tests |

Task order: 1 spec → 2–3 jobs → 4 measurer → 5–6 rules → 7 CLIs. Later tasks consume only names listed in earlier tasks' **Produces**.

N names O calls (all verified at `be0ca3d`):
- `n_spec.SPEC` / `NSpec` / `train_seed(seed, trial, base, stride)`;
- `n_jobs.EDITS`, `apply_edit(eng, pops, edit) -> sha`, `readout_cells(conn, readout) -> (a_cells, p_cells)`, `_present(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells) -> (row, read)`, `PRESENTATION_KEYS`, `presentation_job(...)`, `train_plus_only(...)`, `absolute_arm_job(...)`;
- `n_rules.state(P, spec)`, `cell_stats(rows, spec)`, `boot_index(n, draws, seed)`, `_ci(v, level)`, `delta(row, z)`, `wall_per_step(rows)`, `INVALID`;
- `n_measure.MEASURE_FILES`, `NMeasurer` (`_items(kind, fn, params, items, keys)`, `_window()`);
- `n_cli.NPZ`, `load_c3(nspec)`, `m0d_sha(nspec)`, `model_types(npz)`, `glomeruli(ctx)`, `stimuli_at(ctx, names, g, c_delta, glom=None)`, `drives(st)`;
- `n_store.SUMMARY`;
- `l_cli.refuse`, `run_id`, `check_committed`;
- `h3_store.ROOT`, `MeasureCache`, `canonical`, `canonical_pretty`, `code_key`, `git_state`, `sha256_file`;
- `pool_bench.refuse_old_engine_output`, `refuse_modified_engine_output`;
- `Plasticity.types`, `.da`, `.da_base`, `.mb_local`, `.edges`, `.w0`, `.on_step`, `.weights_frac()`, `.weights_frac_by_mbon_set(mbon_idx)`, `.drive_dan`, `.quiet_dan`, `.recover_pulse`, `.reset_traces`, `.reset_weights`, `.set_enabled`;
- `Engine.on_step`, `.reset`, `.run`, `.step`, `.clear_drive`, `.csc.w`.

---

### Task 1: `o_spec` — every O number, the seed blocks and the training-seed check

**Files:**
- Create: `flymon/brain/o_spec.py`
- Test: `tests/brain/test_o_spec.py`

**Interfaces:**
- Consumes: `n_spec.SPEC`, `NSpec`, `train_seed`.
- Produces:
  - `OSpec` (frozen dataclass) with the fields below.
  - Properties `o1_seeds`, `o2_seeds` (tuples of int) and `on_edit` (str).
  - Methods `o2_train_seeds() -> tuple`, `n_o1_presentations() -> int`, `n_o2_arms() -> int`.
  - `SPEC = OSpec()`, `smoke(spec) -> OSpec`.
  - The field names later tasks use:
    - O1: `n`, `o1_conditions`, `o1_g_grid`, `o1_stimuli`, `o1_c_delta`, `o1_seed0`, `o1_n_seeds`, `mixed_min`, `mixed_cells_min`, `mid_band`, `mid_max`, `auc_min`, `nature_share`, `path_ratio`, `ci_level`, `slope_ci_level`, `flat_range`, `phi_band`, `record_quantiles`;
    - O2: `o2_point`, `o2_pairs`, `o2_arms`, `o2_seed0`, `o2_n_seeds`, `depression_frac`, `sep_frac`, `oracle_o`, `oracle_o_tol`, `flip_max_frac`;
    - runs and smoke: `boot_draws`, `boot_seed`, `smoke_seed0`, `smoke_o1_n`, `smoke_o2_offset`, `smoke_o2_n`, `smoke_boot_draws`, `workers`, `pool_timeout_s`, `spec_path`.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert SPEC.o1_g_grid == N.g_grid and list(SPEC.o1_g_grid) == sorted(SPEC.o1_g_grid)
    assert SPEC.o1_stimuli == N.n0f_stimuli
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_o_spec.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.o_spec`).

- [ ] **Step 3: Write `o_spec.py`**

```python
"""The configuration of spec appendix O as amended by O.7 (O.7 wins over O.2-O.5): C3's two-state map (O1) and
presentation-evoked depression (O2) on N's real-odour rig. A characterisation, no learning claim (O.1).
Every O number is a field here. N's numbers (Hallem pins and mixtures, the g grid and stimuli, probe windows, training
timings and the train-seed rule, the bootstrap draws and seed, the state threshold P < 5) are read through `n` or
assigned from N_SPEC in a default: none is restated."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .n_spec import SPEC as N_SPEC, NSpec, train_seed


@dataclass(frozen=True)
class OSpec:
    n: NSpec = N_SPEC
    # ---- O1 (O.2, O.7.1, O.7.2) ----------------------------------------------------------------------------------------
    o1_conditions: tuple = (("on", "none"), ("kc", "apl_to_kc_zero"), ("nonkc", "apl_to_nonkc_zero"),
                            ("all", "apl_all_zero"))                 # O.7.1: the first is the APL-on rig
    o1_g_grid: tuple = N_SPEC.g_grid               # "N0f와 같음"
    o1_stimuli: tuple = N_SPEC.n0f_stimuli
    o1_c_delta: float = 8.0                        # N.8a's value, δ-DL only
    o1_seed0: int = 22_000_000
    o1_n_seeds: int = 64
    mixed_min: float = 0.10                        # O.7.2-1: silent and firing share both >= 0.10 (7 of 64)
    mixed_cells_min: int = 2                       # fewer -> STOP_NO_SILENT_STATE
    mid_band: tuple = (5, 20)                      # (a): P in [5, 20)
    mid_max: float = 0.05
    auc_min: float = 0.9                           # (b) APL, (c) KC
    nature_share: tuple = (2, 3)                   # >= 2/3 of mixed cells
    path_ratio: float = 0.25                       # O.7.2-3: q_c <= 0.25 q_on
    ci_level: float = 0.95
    slope_ci_level: float = 0.99                   # O.7.2-4: Bonferroni over five stimuli
    flat_range: float = 0.15
    phi_band: tuple = (0.1, 0.9)                   # O.2 record: cell pairs with silent share in [0.1, 0.9]
    record_quantiles: tuple = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)   # per-cell APL / KC distributions (record)
    # ---- O2 (O.3, O.7.3, O.7.4) ----------------------------------------------------------------------------------------
    o2_point: tuple = (0.25, 8.0)                  # (g, c_δ): N.8a's operating point
    o2_pairs: tuple = (("4:1", "1:4", "sim"), ("dDL", "4:1", "dis"))   # (X, Y, the N.9 pair whose oracle o sets e)
    o2_arms: tuple = (("plastic", False, True, False), ("frozen", False, False, False),
                      ("punish", True, True, False), ("da_zero", False, True, True))   # (name, punish, plastic, da_zero)
    o2_seed0: int = 22_001_000
    o2_n_seeds: int = 32
    depression_frac: float = 0.25                  # b = 0.25 x mean pre A_X of arm 1
    sep_frac: float = 0.25                         # e = 0.25 x |o|
    oracle_o: tuple = (("sim", -2.348), ("dis", -2.433))   # O.7.4: N.9's oracle raw effects
    oracle_o_tol: float = 0.0005                   # reading 13: block n1's exact o must round to these
    flip_max_frac: float = 0.125                   # O.7.4-4: more than 1/8 of seeds (4 of 32)
    # ---- bootstrap (O.7.2, O.7.4) ---------------------------------------------------------------------------------------
    boot_draws: int = N_SPEC.boot_draws
    boot_seed: int = N_SPEC.boot_seed
    # ---- smoke (O.5: 22_009_xxx) and runs -------------------------------------------------------------------------------
    smoke_seed0: int = 22_009_000
    smoke_o1_n: int = 8
    smoke_o2_offset: int = 100
    smoke_o2_n: int = 4
    smoke_boot_draws: int = 200
    workers: int = N_SPEC.workers
    pool_timeout_s: float = N_SPEC.pool_timeout_s
    spec_path: str = N_SPEC.spec_path

    @property
    def o1_seeds(self) -> tuple:
        return tuple(range(self.o1_seed0, self.o1_seed0 + self.o1_n_seeds))

    @property
    def o2_seeds(self) -> tuple:
        return tuple(range(self.o2_seed0, self.o2_seed0 + self.o2_n_seeds))

    @property
    def on_edit(self) -> str:
        return self.o1_conditions[0][1]

    def o2_train_seeds(self) -> tuple:
        n = self.n
        return tuple(train_seed(s, t, n.train_seed_base, n.train_seed_stride) for s in self.o2_seeds
                     for t in range(int(n.h4.teach_trials)))

    def n_o1_presentations(self) -> int:
        return len(self.o1_conditions) * len(self.o1_g_grid) * len(self.o1_stimuli) * self.o1_n_seeds

    def n_o2_arms(self) -> int:
        return len(self.o2_pairs) * len(self.o2_arms) * self.o2_n_seeds


SPEC = OSpec()


def smoke(spec: OSpec) -> OSpec:
    """Scale only (reading 20): O1 on smoke_o1_n seeds from smoke_seed0 at two g of the grid (a slope needs two), O2 on
    smoke_o2_n seeds from smoke_seed0 + smoke_o2_offset, smoke_boot_draws draws. Every threshold stays."""
    return dataclasses.replace(spec, o1_g_grid=spec.o1_g_grid[1:3], o1_seed0=spec.smoke_seed0,
                               o1_n_seeds=spec.smoke_o1_n, o2_seed0=spec.smoke_seed0 + spec.smoke_o2_offset,
                               o2_n_seeds=spec.smoke_o2_n, boot_draws=spec.smoke_boot_draws)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_o_spec.py -q -o addopts=""`
Expected: PASS (7 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_spec.py tests/brain/test_o_spec.py
git commit -m "feat(o): o_spec — O1/O2 numbers, seed blocks 22_000/22_001/22_009, N numbers read through spec.n, training-seed collision test"
```

---

### Task 2: `o_jobs` (1) — the `apl_to_nonkc_zero` edit, O's rig cache, O1 presentations

**Files:**
- Create: `flymon/brain/o_jobs.py`
- Test: `tests/brain/test_o_jobs.py`

**Interfaces:**
- Consumes: `n_jobs.EDITS`, `apply_edit`, `readout_cells`, `_present`, `PRESENTATION_KEYS`; `h3_jobs.edge_sources`; `circuits.compartments`; `engine_cpu.Engine`; `plasticity.Plasticity`.
- Produces:
  - `NONKC = "apl_to_nonkc_zero"`, `EDITS` (N's three + `NONKC`), `_RIG`.
  - `apply_edit(eng, pops, edit) -> str` (sha256 of `csc.w`).
  - `rig(conn, pops, params, edit) -> (Engine, Plasticity, comps, sha)`.
  - `presentation_job(eng, pl, pops, comps, ro, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms) -> list[PRESENTATION_KEYS row + edit + csc_sha256]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_o_jobs.py
"""Spec O.7.1: "apl_to_nonkc_zero" zeroes exactly the APL -> non-KC edges (APL -> MBON included); with "apl_to_kc_zero" it
partitions "apl_all_zero"'s edge set; graded-APL views follow; O's rig cache is its own (n_jobs._RIG never touched) and
delegates N's edits to n_jobs.apply_edit; an O1 presentation row is N0f's row (n_jobs._present on O's rig)."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain import o_jobs as O
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_jobs import edge_sources

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
PG = replace(P, apl_mode="graded")
RO = {"A": "MBON03", "P": "MBON01"}
X = {"ORN_DM1": 1.0, "ORN_DA1": 0.5}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    t = np.asarray(c.type).astype(str)
    apl, mbon = int(np.flatnonzero(t == "APL")[0]), int(np.flatnonzero(t == "MBON01")[0])
    c = replace(c, pre=np.append(c.pre, np.int32(apl)), post=np.append(c.post, np.int32(mbon)),
                w=np.append(c.w, np.int32(20)))                 # one APL -> MBON edge: the non-KC edit has a target
    O._RIG.clear(); N._RIG.clear(); H4._RIG.clear()
    return c, Populations.from_connectome(c)


def _zeroed(c, pops, params, edit) -> set:
    e0, e1 = Engine(c, pops, params, seed=0), Engine(c, pops, params, seed=0)
    O.apply_edit(e1, pops, edit)
    return set(np.flatnonzero(e0.csc.w != e1.csc.w).tolist())


@pytest.mark.parametrize("params", [P, PG])
def test_the_two_partial_edits_partition_the_all_output_edit(conn_pops, params):
    c, pops = conn_pops
    kc, nonkc, all_ = (_zeroed(c, pops, params, e) for e in ("apl_to_kc_zero", O.NONKC, "apl_all_zero"))
    assert kc and nonkc and not (kc & nonkc) and (kc | nonkc) == all_


def test_all_output_is_every_apl_out_edge(conn_pops):
    c, pops = conn_pops
    e = Engine(c, pops, P, seed=0)
    assert _zeroed(c, pops, P, "apl_all_zero") == set(np.flatnonzero(np.isin(edge_sources(e.csc), pops.apl)).tolist())


def test_nonkc_zeroes_exactly_the_apl_to_non_kc_edges(conn_pops):
    c, pops = conn_pops
    e0, e1 = Engine(c, pops, P, seed=0), Engine(c, pops, P, seed=0)
    sha = O.apply_edit(e1, pops, O.NONKC)
    src, tgt = edge_sources(e1.csc), e1.csc.tgt.astype(np.int64)
    is_apl, to_kc = np.isin(src, pops.apl), np.isin(tgt, pops.kc)
    to_other = is_apl & ~to_kc
    t = np.asarray(c.type).astype(str)
    assert to_other.any() and np.isin(tgt[to_other], np.flatnonzero(t == "MBON01")).any()
    assert np.all(e1.csc.w[to_other] == 0) and np.all(e0.csc.w[to_other] != 0)
    assert np.array_equal(e1.csc.w[~to_other], e0.csc.w[~to_other])
    assert sha not in (O.apply_edit(e0, pops, "none"),)


def test_graded_views_follow_the_nonkc_edit(conn_pops):
    c, pops = conn_pops
    e, _, _, _ = O.rig(c, pops, PG, O.NONKC)
    is_kc = np.zeros(e.N, bool); is_kc[pops.kc] = True
    for tgt, w in e._apl_edges:
        assert np.all(w[~is_kc[tgt]] == 0) and np.any(w[is_kc[tgt]] != 0)


def test_ns_edits_are_delegated_and_unknown_edits_refused(conn_pops):
    c, pops = conn_pops
    for edit in N.EDITS:
        assert O.apply_edit(Engine(c, pops, P, seed=0), pops, edit) == N.apply_edit(Engine(c, pops, P, seed=0), pops,
                                                                                     edit)
    assert O.EDITS == N.EDITS + (O.NONKC,)
    with pytest.raises(ValueError, match="unknown edit"):
        O.apply_edit(Engine(c, pops, P, seed=0), pops, "apl_half")


def test_rig_is_keyed_by_params_and_edit_and_never_touches_ns_cache(conn_pops):
    c, pops = conn_pops
    a = O.rig(c, pops, P, "none")
    b = O.rig(c, pops, P, O.NONKC)
    assert a[0] is not b[0] and a[3] != b[3] and len(O._RIG) == 1
    assert O.rig(c, pops, P, O.NONKC)[0] is b[0] and len(N._RIG) == 0


@pytest.mark.parametrize("params", [P, PG])
def test_o1_rows_are_n0f_rows_for_ns_edits(conn_pops, params):
    c, pops = conn_pops
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    for edit in N.EDITS:
        kw = dict(params=params, edit=edit, odor=X, seeds=[3, 4], readout=RO, **W)
        assert strip(O.presentation_job(_Stub(c), None, pops, None, None, **kw)) == \
            strip(N.presentation_job(_Stub(c), None, pops, None, None, **kw))


@pytest.mark.parametrize("params", [P, PG])
def test_on_nonkc_on_in_one_worker(conn_pops, params):
    c, pops = conn_pops
    kw = dict(params=params, odor=X, seeds=[3, 4], readout=RO, **W)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    on1 = O.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    blk = O.presentation_job(_Stub(c), None, pops, None, None, edit=O.NONKC, **kw)
    on2 = O.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    assert strip(on1) == strip(on2)
    O._RIG.clear()
    assert strip(blk) == strip(O.presentation_job(_Stub(c), None, pops, None, None, edit=O.NONKC, **kw))
    assert {r["csc_sha256"] for r in on1} != {r["csc_sha256"] for r in blk}
    assert [r["edit"] for r in blk] == [O.NONKC] * 2


def test_an_o1_row_is_scalars_only(conn_pops):
    c, pops = conn_pops
    rows = O.presentation_job(_Stub(c), None, pops, None, None, params=P, edit=O.NONKC, odor=X, seeds=[3],
                              readout=RO, **W)
    assert set(rows[0]) == set(N.PRESENTATION_KEYS) | {"edit", "csc_sha256"}
    assert all(np.isscalar(v) for v in rows[0].values())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_o_jobs.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.o_jobs`).

- [ ] **Step 3: Write `o_jobs.py` (part 1)**

```python
"""FlyPool worker jobs of spec appendix O (O.7.1, O.7.3). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle them; the worker's default engine only lends its connectome.

O's rig is cached per worker under (Params, edit), edit in EDITS = n_jobs.EDITS + ("apl_to_nonkc_zero",):
- N's three edits are made by n_jobs.apply_edit, unchanged;
- "apl_to_nonkc_zero" (O.7.1): every APL out-edge whose target is not a KC set to 0 (APL -> MBON05 included). With
  "apl_to_kc_zero" it partitions "apl_all_zero"'s edge set (test).
The edit is made in place on a freshly built engine's CSC (the graded-APL out-edge views follow). The cache is O's own:
n_jobs._RIG is never touched, and no N module changes (O.5).

presentation_job: O1's presentations, N0f's procedure (n_jobs._present: reset, clear_drive, present, settle, read,
plasticity off), scalars only."""
from __future__ import annotations

import hashlib

import numpy as np

from . import n_jobs
from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .plasticity import Plasticity

NONKC = "apl_to_nonkc_zero"
EDITS = n_jobs.EDITS + (NONKC,)
_RIG: dict = {}          # (Params, edit) -> (Engine, Plasticity, comps, csc sha256); one entry per worker


def apply_edit(eng, pops, edit: str) -> str:
    """N's edits through n_jobs.apply_edit; NONKC zeroes every APL -> non-KC edge in place. The sha256 of csc.w."""
    if edit != NONKC:
        return n_jobs.apply_edit(eng, pops, edit)
    is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
    is_kc = np.zeros(eng.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
    m = is_apl[edge_sources(eng.csc)] & ~is_kc[eng.csc.tgt.astype(np.int64)]
    eng.csc.w[m] = np.float32(0.0)
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


def rig(conn, pops, params, edit: str):
    """The worker's rig for (params, edit), built once and reused while jobs keep asking for it (one entry: a switch
    rebuilds, it never edits a cached engine)."""
    key = (params, edit)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha = apply_edit(eng, pops, edit)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha)
    return _RIG[key]


def presentation_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                     settle_ms: float, read_ms: float, window_ms: int) -> list:
    """O1's presentations on O's rig: n_jobs._present per seed, plasticity off, weights at w0; scalars only."""
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = n_jobs.readout_cells(e.conn, readout)
    p.reset_weights()
    was = p.enabled
    p.set_enabled(False)
    try:
        return [dict(n_jobs._present(e, p, pops, odor, s, strength, settle_ms, read_ms, window_ms, a_cells,
                                     p_cells)[0], edit=edit, csc_sha256=sha) for s in seeds]
    finally:
        p.set_enabled(was)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_o_jobs.py -q -o addopts=""`
Expected: PASS (12 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_jobs.py tests/brain/test_o_jobs.py
git commit -m "feat(o): o_jobs rig with apl_to_nonkc_zero (partitions apl_all_zero), O's own cache, O1 presentations = N0f rows"
```

---

### Task 3: `o_jobs` (2) — the O training loop, dopamine-zero arm, phasic-dopamine meter, O2 arm job

**Files:**
- Modify: `flymon/brain/o_jobs.py` (append; the file is new in this plan)
- Test: `tests/brain/test_o_jobs_arms.py`

**Interfaces:**
- Consumes: Task 2's `rig`, `n_jobs.readout_cells`, `n_jobs._present`; `n_spec.train_seed`; `stimuli.present`.
- Produces:
  - `DaMeter(e, p)`: a context manager that installs itself as `e.on_step`. It has `.on`, `.acc` and `.by_compartment(comps) -> {str: float}`.
  - `dopamine_zero(p)`: a context manager.
  - `train_x(e, p, pops, cs, strength, seed, punish, trials, present_ms, gap_ms, settle_ms, seed_base, seed_stride, meter=None) -> None`. `punish` is a DAN type name or `None`.
  - `weights_sha256(w) -> str`.
  - `arm_job(eng, pl, pops, comps, ro, params, edit, odor_x, odor_y, seed, arm, punish, plastic, da_zero, readout, punish_type, strength, settle_ms, read_ms, window_ms, trials, present_ms, gap_ms, train_settle_ms, seed_base, seed_stride) -> dict`. The dict has keys `seed`, `edit`, `arm`, `punish`, `plastic`, `da_zero`, `csc_sha256`, `pre: {"x", "y"}`, `post: {"x", "y"}`, `weights_frac`, `weights_frac_A`, `weights_frac_P`, `w0_sha256`, `w_post_sha256`, `da_integral: {compartment: float}` and `wall_s`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_o_jobs_arms.py
"""Spec O.7.3: one O training loop with a punish flag. Flag on is n_jobs.absolute_arm_job bit for bit (regression); flag
off skips drive_dan and recover_pulse (train_block's dan-None rule). The dopamine-zero arm holds the rule's dopamine
trace at exactly 0 every step and leaves DAN spikes untouched, and its swap never outlives the job. The phasic-dopamine
integral is recorded per compartment, weights_frac_by_mbon_set for the readout's A and P cells; the plumbing arm's
post-training weights hash to w0's."""
import contextlib
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain import o_jobs as O
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.o_spec import SPEC

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}                  # synthetic: PPL105 core = MBON03/04, PAM08 core = MBON01/02
X, Y = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)
ARM = dict(readout=RO, punish_type="PPL105", trials=2, present_ms=300.0, gap_ms=50.0, train_settle_ms=100.0,
           seed_base=1_000_000, seed_stride=1000, **W)
FLAGS = {name: dict(punish=pu, plastic=pl, da_zero=dz) for name, pu, pl, dz in SPEC.o2_arms}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    t = np.asarray(c.type).astype(str)
    apl, mbon = int(np.flatnonzero(t == "APL")[0]), int(np.flatnonzero(t == "MBON01")[0])
    c = replace(c, pre=np.append(c.pre, np.int32(apl)), post=np.append(c.post, np.int32(mbon)),
                w=np.append(c.w, np.int32(20)))
    O._RIG.clear(); N._RIG.clear(); H4._RIG.clear()
    return c, Populations.from_connectome(c)


def _o(c, pops, arm, edit="none", seed=9):
    return O.arm_job(_Stub(c), None, pops, None, None, params=P, edit=edit, odor_x=X, odor_y=Y, seed=seed, arm=arm,
                     **FLAGS[arm], **ARM)


def _rows(d):
    return {k: {f: v for f, v in r.items() if f != "wall_s"} for k, r in d.items()}


@pytest.mark.parametrize("edit", ["none", "apl_to_kc_zero"])
def test_punish_arm_is_bit_identical_to_absolute_arm_job(conn_pops, edit):
    c, pops = conn_pops
    o = _o(c, pops, "punish", edit=edit)
    n = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit=edit, odor_x=X, odor_y=Y, seed=9,
                           plastic=True, **ARM)
    for k in ("seed", "edit", "plastic", "csc_sha256", "weights_frac"):
        assert o[k] == n[k], k
    assert _rows(o["pre"]) == _rows(n["pre"]) and _rows(o["post"]) == _rows(n["post"])
    assert o["weights_frac"] < 1.0 and o["w_post_sha256"] != o["w0_sha256"]      # not vacuous: weights moved


def test_train_x_with_punish_equals_train_plus_only(conn_pops):
    c, pops = conn_pops
    e, p, _, _ = O.rig(c, pops, P, "none")
    p.reset_weights(); p.set_enabled(True)
    N.train_plus_only(e, p, pops, X, 3.0, 7, "PPL105", 2, 300.0, 50.0, 100.0, 1_000_000, 1000)
    ref = e.csc.w[p.edges].copy()
    p.reset_weights(); p.set_enabled(True)
    O.train_x(e, p, pops, X, 3.0, 7, "PPL105", 2, 300.0, 50.0, 100.0, 1_000_000, 1000)
    assert not np.array_equal(ref, p.w0) and np.array_equal(e.csc.w[p.edges], ref)
    p.reset_weights()


def test_train_x_without_punish_never_drives_or_recovers(conn_pops, monkeypatch):
    c, pops = conn_pops
    e, p, _, _ = O.rig(c, pops, P, "none")
    calls = []
    monkeypatch.setattr(p, "drive_dan", lambda *a: calls.append("drive"))
    monkeypatch.setattr(p, "recover_pulse", lambda: calls.append("recover"))
    seen, orig = [], e.reset
    monkeypatch.setattr(e, "reset", lambda seed=None: (seen.append(seed), orig(seed))[1])
    O.train_x(e, p, pops, X, 3.0, 22_001_005, None, 3, 100.0, 50.0, 100.0, 1_000_000, 1000)
    assert calls == [] and seen == [1_000_000 + 22_001_005 * 1000 + t for t in range(3)]


def test_dopamine_zero_holds_the_trace_at_zero_and_leaves_dan_spikes(conn_pops):
    c, pops = conn_pops
    e, p, _, _ = O.rig(c, pops, P, "none")
    dan = p.types["PPL105"][0]

    def run(zero):
        e.reset(5); p.reset_traces(); e.clear_drive(); p.quiet_dan(); p.set_enabled(False)
        p.drive_dan("PPL105", e.p.dan_drive_mv)
        n_dan, da_max = 0, 0.0
        with (O.dopamine_zero(p) if zero else contextlib.nullcontext()):
            for _ in range(200):
                n_dan += int(np.isin(e.step(), dan).sum())
                da_max = max(da_max, float(np.abs(p.da).max()), float(np.abs(p.da_base).max()))
        p.quiet_dan(); p.set_enabled(True)
        return n_dan, da_max

    n_plain, da_plain = run(False)
    n_zero, da_zero = run(True)
    assert n_plain > 0 and n_zero == n_plain                  # DAN firing untouched
    assert da_plain > 0 and da_zero == 0.0                    # the rule's dopamine trace held at exactly 0
    assert p.types["PPL105"][1].any()                         # restored


def test_da_zero_arm_moves_no_weight_and_integrates_no_dopamine(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "da_zero")
    assert r["w_post_sha256"] == r["w0_sha256"] and r["weights_frac"] == 1.0
    assert all(v == 0.0 for v in r["da_integral"].values())


def test_frozen_arm_is_the_plumbing_check(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "frozen")
    assert r["w_post_sha256"] == r["w0_sha256"] and r["weights_frac"] == 1.0
    assert _rows(r["pre"]) == _rows(r["post"])


def test_punish_arm_records_dopamine_and_readout_weights(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "punish")
    _, _, comps, _ = O.rig(c, pops, P, "none")
    assert set(r["da_integral"]) == {k for k, cp in comps.items() if cp.core.size}
    assert r["da_integral"]["PPL105"] > 0.0
    assert r["weights_frac_A"] < 1.0                          # PPL105's core is the A readout here
    assert (r["arm"], r["punish"], r["plastic"], r["da_zero"]) == ("punish", True, True, False)


def test_a_da_zero_arm_does_not_leak_into_the_next_arm(conn_pops):
    c, pops = conn_pops
    _o(c, pops, "da_zero")
    after = _o(c, pops, "punish")
    e, p, _, _ = O.rig(c, pops, P, "none")
    assert e.on_step == p.on_step and all(w.any() for _, w in p.types.values())
    O._RIG.clear()
    fresh = _o(c, pops, "punish")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s", "pre", "post")}
    assert strip(after) == strip(fresh) and _rows(after["post"]) == _rows(fresh["post"])


def test_arms_on_block_on_in_one_worker(conn_pops):
    c, pops = conn_pops
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s", "pre", "post")}
    a, b, a2 = _o(c, pops, "punish"), _o(c, pops, "punish", edit=O.NONKC), _o(c, pops, "punish")
    assert strip(a) == strip(a2) and a["csc_sha256"] != b["csc_sha256"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_o_jobs_arms.py -q -o addopts=""`
Expected: FAIL (`AttributeError: module 'flymon.brain.o_jobs' has no attribute 'arm_job'`).

- [ ] **Step 3: Append to `o_jobs.py`**

Add to the imports at the top of the file: `import contextlib`, `import time`, `from .n_spec import train_seed`, `from .stimuli import present`. Then append:

```python
# ================================================================ O2: the training loop and the arms (O.7.3)
class DaMeter:
    """Installed as the engine's step hook while a training runs: the plasticity rule runs unchanged
    (Plasticity.on_step), then, while `on`, the phasic dopamine max(da - da_base, 0) is integrated per MBON (x dt).
    Reading the traces changes nothing; leaving the context restores e.on_step = p.on_step."""

    def __init__(self, e, p):
        self.e, self.p, self.on = e, p, False
        self.acc = np.zeros(p.da.shape, np.float64)

    def __call__(self, eng, fired) -> None:
        self.p.on_step(eng, fired)
        if self.on:
            self.acc += np.maximum(self.p.da - self.p.da_base, 0.0) * eng.p.dt

    def __enter__(self):
        self.e.on_step = self
        return self

    def __exit__(self, *exc) -> None:
        self.e.on_step = self.p.on_step

    def by_compartment(self, comps) -> dict:
        """Reading 11: per DAN type, the mean over its core MBONs of the integral."""
        return {name: float(self.acc[self.p.mb_local[cp.core]].mean()) for name, cp in sorted(comps.items())
                if cp.core.size}


@contextlib.contextmanager
def dopamine_zero(p):
    """Arm 4 (O.7.3, reading 10): every DAN type's MBON weight vector in the rule set to 0, so p.da and p.da_base stay
    exactly 0 every step. The DAN cells, their drive and their spikes are untouched (drive_dan / quiet_dan read only
    the cells). Restored on exit."""
    saved = p.types
    p.types = {k: (cells, np.zeros_like(w)) for k, (cells, w) in saved.items()}
    p.da[:] = 0; p.da_base[:] = 0
    try:
        yield
    finally:
        p.types = saved


def train_x(e, p, pops, cs, strength: float, seed: int, punish, trials: int, present_ms: float, gap_ms: float,
            settle_ms: float, seed_base: int, seed_stride: int, meter=None) -> None:
    """X alone, n_jobs.train_plus_only's loop with a punish flag (reading 9). Per trial:
    - reset to train_seed(seed, trial);
    - settle with the weights frozen;
    - the punishment DAN for present_ms if `punish` is a DAN type (None: no injection);
    - the gap, then one recovery step only after a pulse (train_block's dan-None rule).
    The meter integrates over present + gap."""
    for t in range(int(trials)):
        e.reset(train_seed(seed, t, seed_base, seed_stride))
        p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, cs, strength)
        was = p.enabled
        p.set_enabled(False); e.run(settle_ms); p.set_enabled(was)
        if punish is not None:
            p.drive_dan(punish, e.p.dan_drive_mv)
        if meter is not None:
            meter.on = True
        e.run(present_ms)
        p.quiet_dan(); e.clear_drive(); e.run(gap_ms)
        if meter is not None:
            meter.on = False
        if punish is not None:
            p.recover_pulse()


def weights_sha256(w) -> str:
    return hashlib.sha256(np.ascontiguousarray(w, np.float32).tobytes()).hexdigest()


def arm_job(eng, pl, pops, comps, ro, params, edit: str, odor_x: dict, odor_y: dict, seed: int, arm: str,
            punish: bool, plastic: bool, da_zero: bool, readout: dict, punish_type: str, strength: float,
            settle_ms: float, read_ms: float, window_ms: int, trials: int, present_ms: float, gap_ms: float,
            train_settle_ms: float, seed_base: int, seed_stride: int) -> dict:
    """One (X, arm, seed) of O2 in the edit's rig (O.7.3):
    - probes of X and Y at `seed`, plasticity off (n_jobs._present rows, as absolute_arm_job);
    - training of X alone (train_x): punishment iff `punish`, weights frozen unless `plastic`, the rule's dopamine
      held at 0 iff `da_zero`; the phasic dopamine integrated per compartment;
    - the probes again; weights_frac, weights_frac_by_mbon_set of the readout's A / P cells, and the sha256 of the
      plastic weights before (w0) and after training (the plumbing check, O.7.4-5).
    With punish and plastic on this is absolute_arm_job's result (regression test). Weights, the step hook and the
    rule's dopamine weights are restored."""
    e, p, c, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = n_jobs.readout_cells(e.conn, readout)
    t0 = time.perf_counter()

    def probes():
        was = p.enabled
        p.set_enabled(False)
        try:
            return {k: n_jobs._present(e, p, pops, o, seed, strength, settle_ms, read_ms, window_ms, a_cells,
                                       p_cells)[0] for k, o in (("x", odor_x), ("y", odor_y))}
        finally:
            p.set_enabled(was)

    meter = DaMeter(e, p)
    try:
        p.reset_weights()
        w0_sha = weights_sha256(p.w0)
        pre = probes()
        p.set_enabled(bool(plastic))
        with meter, (dopamine_zero(p) if da_zero else contextlib.nullcontext()):
            train_x(e, p, pops, odor_x, strength, int(seed), punish_type if punish else None, trials, present_ms,
                    gap_ms, train_settle_ms, seed_base, seed_stride, meter)
        p.set_enabled(True)
        post = probes()
        w_post = weights_sha256(e.csc.w[p.edges])
        wf = float(p.weights_frac())
        wa, wp = float(p.weights_frac_by_mbon_set(a_cells)), float(p.weights_frac_by_mbon_set(p_cells))
        da = meter.by_compartment(c)
    finally:
        e.on_step = p.on_step
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), edit=edit, arm=arm, punish=bool(punish), plastic=bool(plastic),
                da_zero=bool(da_zero), csc_sha256=sha, pre=pre, post=post, weights_frac=wf, weights_frac_A=wa,
                weights_frac_P=wp, w0_sha256=w0_sha, w_post_sha256=w_post, da_integral=da,
                wall_s=time.perf_counter() - t0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_o_jobs.py tests/brain/test_o_jobs_arms.py -q -o addopts=""`
Expected: PASS (22 tests).

If `test_punish_arm_is_bit_identical_to_absolute_arm_job` fails **only** on the "not vacuous" assert, the synthetic pulse moved no weight. Lengthen `present_ms` in `ARM` (e.g. 600.0) until the weights move. Never delete the assert. Any other mismatch is a real divergence from `train_plus_only`: fix `train_x`, never the test.

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_jobs.py tests/brain/test_o_jobs_arms.py
git commit -m "feat(o): O2 training loop with punish flag (bit-identical to absolute_arm_job), dopamine-zero arm, phasic-dopamine meter, w sha256"
```

---

### Task 4: `o_measure` — the write guard, per-item cache files, `OMeasurer`

**Files:**
- Create: `flymon/brain/o_measure.py`
- Test: `tests/brain/test_o_measure.py`

**Interfaces:**
- Consumes:
  - from earlier tasks: `o_jobs.presentation_job`, `o_jobs.arm_job`; `OSpec` (`n`);
  - from N and shared code: `n_measure.MEASURE_FILES`, `NMeasurer` (its `_items`, `_window`); `h3_store.MeasureCache`, `canonical`, `canonical_pretty`; `pool_bench.refuse_old_engine_output`, `refuse_modified_engine_output`.
- Produces:
  - constants and writers: `ALLOWED_DIR = "results/o/"`, `SUMMARY = "results/summary/o_states.json"`, `MEASURE_FILES`, `HASHED_FILES`, `guard(path, params_list)`, `write_bytes`, `write_json`, `write_summary_block(path, block, obj, params_list)`;
  - the cache: `OCache(root, code, run_id, spec_commit)`;
  - the measurer: `OMeasurer(pool, spec: OSpec, cache)` with
    - `.o1_presentations(params, items, readout) -> list[row + cond/g/stim]`, where items are `[{"cond", "edit", "g", "stim", "seed", "odor"}]`;
    - `.o2_arms(params, items, readout, punish_type) -> list[arm row + x/y]`, where items are `[{"x", "y", "edit", "arm", "punish", "plastic", "da_zero", "odor_x", "odor_y", "seed"}]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_o_measure.py
"""Spec O.7.5: one atomic cache file per (O1, condition, cell, seed) and per (O2, X, arm, seed), each carrying the code
key and the spec commit; a rerun resumes without a pool and an interrupted one computes only the missing items; O writes
only under results/o/ and results/summary/o_states.json; rounds of one item per worker."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flymon.brain import n_measure, o_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_measure import (HASHED_FILES, MEASURE_FILES, OCache, OMeasurer, write_json,
                                    write_summary_block)
from flymon.brain.o_spec import SPEC

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}
X, Y = {"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}
CODE = {"key": "k" * 64}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


class InProcessPool:
    n_workers = 3

    def __init__(self, conn, pops):
        self.conn, self.pops, self.rounds, self.jobs = conn, pops, [], 0

    def run_jobs(self, fn, jobs):
        self.rounds.append(len(jobs)); self.jobs += len(jobs)
        return [fn(_Stub(self.conn), None, self.pops, None, None, **j) for j in jobs]


class NoPool:
    n_workers = 3

    def run_jobs(self, fn, jobs):
        raise AssertionError("a resumed run must not touch the pool")


def _spec():
    h4 = replace(N_SPEC.h4, teach_trials=1, teach_present_ms=100.0, teach_gap_ms=20.0,
                 teach_window=replace(N_SPEC.h4.teach_window, settle_ms=50.0),
                 oracle_window=replace(N_SPEC.h4.oracle_window, settle_ms=30.0, read_ms=50.0))
    return replace(SPEC, n=replace(N_SPEC, l=replace(N_SPEC.l, j=replace(N_SPEC.l.j, h4=h4))))


@pytest.fixture
def world(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome(disjoint_kc=True)
    o_jobs._RIG.clear()
    return c, Populations.from_connectome(c)


def _o1_items():
    return [dict(cond=cond, edit=edit, g=1.0, stim="X", seed=s, odor=X)
            for cond, edit in (("on", "none"), ("kc", "apl_to_kc_zero")) for s in (5, 6)]


def _o2_items():
    return [dict(x="X", y="Y", edit="none", arm=a, punish=pu, plastic=pl, da_zero=dz, odor_x=X, odor_y=Y, seed=5)
            for a, pu, pl, dz in SPEC.o2_arms]


def test_o1_one_file_per_item_with_code_key_and_spec_commit_and_resume(world):
    c, pops = world
    pool = InProcessPool(c, pops)
    rows = OMeasurer(pool, _spec(), OCache("results/o/cache", CODE, "run1", "c0ffee")).o1_presentations(
        P, _o1_items(), RO)
    assert [(r["cond"], r["seed"]) for r in rows] == [("on", 5), ("on", 6), ("kc", 5), ("kc", 6)]
    assert pool.rounds == [3, 1]
    files = sorted(Path("results/o/cache/o1_pres").glob("*.json"))
    assert len(files) == 4
    d = json.loads(files[0].read_text())
    assert d["spec_commit"] == "c0ffee" and d["code_key"] == "k" * 64 and d["inputs"]["stage"] == "o1"
    again = OMeasurer(NoPool(), _spec(), OCache("results/o/cache", CODE, "run2", "c0ffee")).o1_presentations(
        P, _o1_items(), RO)
    assert json.dumps(again, sort_keys=True) == json.dumps(json.loads(json.dumps(rows)), sort_keys=True)


def test_an_interrupted_run_computes_only_the_missing_items(world):
    c, pops = world
    OMeasurer(InProcessPool(c, pops), _spec(), OCache("results/o/cache", CODE, "run1", "x")).o1_presentations(
        P, _o1_items()[:2], RO)
    pool = InProcessPool(c, pops)
    OMeasurer(pool, _spec(), OCache("results/o/cache", CODE, "run2", "x")).o1_presentations(P, _o1_items(), RO)
    assert pool.jobs == 2


def test_o2_one_file_per_arm_and_resume(world):
    c, pops = world
    rows = OMeasurer(InProcessPool(c, pops), _spec(), OCache("results/o/cache", CODE, "r", "x")).o2_arms(
        P, _o2_items(), RO, "PPL105")
    assert [r["arm"] for r in rows] == [a for a, *_ in SPEC.o2_arms] and {r["x"] for r in rows} == {"X"}
    assert len(list(Path("results/o/cache/o2_arm").glob("*.json"))) == 4        # the arms never share an entry
    again = OMeasurer(NoPool(), _spec(), OCache("results/o/cache", CODE, "r2", "x")).o2_arms(
        P, _o2_items(), RO, "PPL105")
    assert [r["w_post_sha256"] for r in again] == [r["w_post_sha256"] for r in rows]


def test_writes_only_under_results_o(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(Path("results/o/x/a.json"), {"a": 1}, [P])
    write_summary_block(Path("results/summary/o_states.json"), "o1", {"outcome": "JUDGED"}, [P])
    assert json.loads(Path("results/summary/o_states.json").read_text()) == {"o1": {"outcome": "JUDGED"}}
    for bad in ("results/n/x.json", "results/summary/n_real_odour.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            write_json(Path(bad), {}, [P])
        assert e.value.code == 2


def test_file_lists_extend_ns():
    assert set(n_measure.MEASURE_FILES) <= set(MEASURE_FILES)
    assert {"flymon/brain/o_jobs.py", "flymon/brain/o_measure.py"} <= set(MEASURE_FILES)
    assert set(MEASURE_FILES) <= set(HASHED_FILES)
    assert {"flymon/brain/o_spec.py", "flymon/brain/o_rules.py", "flymon/brain/o_cli.py", "scripts/run_o1.py",
            "scripts/run_o2.py", "flymon/brain/n_rules.py"} <= set(HASHED_FILES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_o_measure.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.o_measure`).

- [ ] **Step 3: Write `o_measure.py`**

```python
"""Storage and measurement of spec appendix O (O.5, O.7.5).
- The write guard: raw rows, caches and reports under results/o/ (git-ignored by `results/*`) and the summary
  results/summary/o_states.json; nothing else (SystemExit 2). Every write is atomic (tmp + os.replace). n_store's rule
  with O's paths: n_store itself is hard-wired to results/n/.
- OCache: h3_store.MeasureCache's content-addressed entries, one file per item, written through O's guard, each also
  naming the code key and the spec commit (O.7.5).
- OMeasurer: n_measure.NMeasurer's rounds of one item per worker (its _items and _window, unchanged), over O's jobs:
  O1 one presentation per (condition, cell, seed), O2 one arm per (X, arm, seed). A rerun reads every finished item
  without starting a job; an interrupted run loses at most one round."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from . import n_measure, o_jobs
from .h3_store import MeasureCache, canonical, canonical_pretty
from .n_measure import NMeasurer
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/o/"
SUMMARY = "results/summary/o_states.json"
MEASURE_FILES = tuple(dict.fromkeys(n_measure.MEASURE_FILES + ("flymon/brain/o_jobs.py", "flymon/brain/o_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + (
    "flymon/brain/n_spec.py", "flymon/brain/n_store.py", "flymon/brain/n_rules.py", "flymon/brain/n_cli.py",
    "scripts/fetch_door_hallem.py", "flymon/brain/o_spec.py", "flymon/brain/o_rules.py", "flymon/brain/o_cli.py",
    "scripts/run_o1.py", "scripts/run_o2.py")))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: spec O writes only under {ALLOWED_DIR} and {SUMMARY}", file=sys.stderr)
        raise SystemExit(2)


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(obj) + "\n").encode(), params_list)


def write_summary_block(path, block: str, obj, params_list) -> Path:
    guard(path, params_list)
    doc = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    doc[block] = json.loads(canonical_pretty(obj))
    return write_json(path, doc, params_list)


class OCache(MeasureCache):
    """MeasureCache with O's guarded write; every entry also holds the code key and the spec commit (O.7.5)."""

    def __init__(self, root, code: dict, run_id: str, spec_commit: str | None):
        super().__init__(root, code, run_id)
        self.spec_commit = spec_commit

    def get_or_compute(self, kind: str, inputs: dict, compute, params_list):
        k = self.key(kind, inputs)
        path = self.root / kind / f"{k[:24]}.json"
        self.used[str(path)] = k
        if path.exists():
            try:
                d = json.loads(path.read_text())
                if d.get("key") == k:
                    self.hits += 1
                    return d["result"]
            except (OSError, ValueError):
                pass
        result = compute()
        write_json(path, dict(key=k, kind=kind, run_id=self.run_id, code_key=self.code["key"],
                              spec_commit=self.spec_commit, inputs=json.loads(canonical(inputs)), result=result),
                   params_list)
        self.misses += 1
        return json.loads(canonical(result))


class OMeasurer(NMeasurer):
    """NMeasurer's item rounds over O's jobs; `spec` is an OSpec (N's windows and timings through spec.n)."""

    def __init__(self, pool, spec, cache):
        super().__init__(pool, spec.n, cache)
        self.ospec = spec

    def o1_presentations(self, params, items, readout) -> list:
        """items [{"cond", "edit", "g", "stim", "seed", "odor"}] -> one presentation row per item, with cond / g / stim;
        the key is the unit (O1, condition, cell, seed) plus everything the job reads."""
        common = dict(params=params, readout=dict(readout), **self._window())
        jobs = [dict(common, edit=i["edit"], odor=dict(i["odor"]), seeds=(int(i["seed"]),)) for i in items]
        keys = [dict(j, stage="o1", cond=i["cond"], g=float(i["g"]), stim=i["stim"]) for j, i in zip(jobs, items)]
        out = self._items("o1_pres", o_jobs.presentation_job, params, jobs, keys)
        return [dict(rows[0], cond=i["cond"], g=float(i["g"]), stim=i["stim"]) for rows, i in zip(out, items)]

    def o2_arms(self, params, items, readout, punish_type) -> list:
        """items [{"x", "y", "edit", "arm", "punish", "plastic", "da_zero", "odor_x", "odor_y", "seed"}] -> arm_job rows
        with x / y; N2's training timings and seed rule (spec.n)."""
        s, h4 = self.spec, self.spec.h4
        common = dict(params=params, readout=dict(readout), punish_type=punish_type, trials=int(h4.teach_trials),
                      present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms,
                      train_settle_ms=h4.teach_window.settle_ms, seed_base=s.train_seed_base,
                      seed_stride=s.train_seed_stride, **self._window())
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     arm=i["arm"], punish=bool(i["punish"]), plastic=bool(i["plastic"]),
                     da_zero=bool(i["da_zero"])) for i in items]
        keys = [dict(j, stage="o2", x=i["x"], y=i["y"]) for j, i in zip(jobs, items)]
        rows = self._items("o2_arm", o_jobs.arm_job, params, jobs, keys)
        return [dict(r, x=i["x"], y=i["y"]) for r, i in zip(rows, items)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_o_measure.py -q -o addopts=""`
Expected: PASS (5 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_measure.py tests/brain/test_o_measure.py
git commit -m "feat(o): o_measure — results/o/ guard, per-item cache files with code key + spec commit, OMeasurer over NMeasurer rounds"
```

---

### Task 5: `o_rules` (1) — validity gate, seed-cluster bootstrap, O1 judgements and records

**Files:**
- Create: `flymon/brain/o_rules.py`
- Test: `tests/brain/test_o_rules_o1.py`

**Interfaces:**
- Consumes: `n_rules.INVALID`, `state`, `cell_stats`, `boot_index`, `_ci`, `delta`; `OSpec` fields from Task 1.
- Produces:
  - outcome and label constants: `JUDGED`, `STOP_NO_SILENT_STATE`, `BISTABLE_NETWORK`, `BIMODAL_P_READOUT`, `GRADED`, `READOUT_PATH`, `KC_NETWORK_PATH`, `BOTH_PATHS`, `NEITHER_PATH`, `FLAT`, `INCREASING`, `DECREASING`, `NONMONOTONIC_OR_UNCLEAR`, `DEPRESSION_PRESENT`, `NO_DEPRESSION`, `INCONCLUSIVE`, `DA_DEPENDENT`, `DA_INDEPENDENT`, `SEPARABLE`, `NOT_SEPARABLE`, `FLIP_SUFFIX`, plus `INVALID` re-exported;
  - shared helpers: `finite(obj) -> bool`, `validity(rows, declared, key_of, seeds) -> list[str]`, `boot_weights(n, draws, seed) -> ndarray (draws × n)`, `auc(pos, neg) -> float | None`;
  - O1 cell helpers: `cell_key(g, stim) -> str`, `o1_cells(spec) -> list[(g, stim)]`, `o1_key(row)`, `o1_declared(spec) -> set`, `silent(rows, spec) -> ndarray[bool]`;
  - O1 rules: `mixed_cells(S, spec) -> list[int]`, `cell_nature(rows, spec) -> dict`, `nature_label(nat, spec) -> str`, `pathway(S, mixed, W, spec) -> dict`, `drive(S, cells, W, spec) -> dict`;
  - O1 records and the judge: `phi_pairs(S, cells, spec) -> list`, `cell_record(rows, spec) -> dict`, `o1_judge(rows, spec) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_o_rules_o1.py
"""Spec O.7.2 / O.7.5: the validity gate before any judgement; mixed cells (both shares >= 0.10, fewer than 2 ->
STOP_NO_SILENT_STATE); the state nature from the P mid band and the APL / KC AUCs; the pathway 2x2 with its 0.25 ratio
and CI clause; per-stimulus drive labels with 99% seed-cluster slope CIs; the records. The state is n_rules.state."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import n_rules
from flymon.brain import o_rules as R
from flymon.brain.o_spec import SPEC

S0 = 100


def _spec(g=(0.25, 1.0), stim=("4:1", "dDL"), n=40, draws=500):
    return replace(SPEC, o1_g_grid=g, o1_stimuli=stim, o1_seed0=S0, o1_n_seeds=n, boot_draws=draws)


def _row(cond, edit, g, stim, seed, P, apl, kc):
    return dict(seed=seed, A=10, P=P, kc_frac=0.05, kc_spikes=kc, kc_max_win_hz=50.0, apl_out_per_step=apl,
                wall_s=0.01, steps=1400, edit=edit, csc_sha256="sha-" + edit, cond=cond, g=float(g), stim=stim)


def _rows(spec, sil, fire_P=40, same_net=False):
    """sil(cond, g, stim, i) -> silent? Silent: P 0, APL 1.0, KC 10; firing: P fire_P, APL 0.1, KC 100 (same_net: APL /
    KC equal in both states)."""
    out = []
    for cond, edit in spec.o1_conditions:
        for g in spec.o1_g_grid:
            for s in spec.o1_stimuli:
                for i, seed in enumerate(spec.o1_seeds):
                    q = sil(cond, g, s, i)
                    apl, kc = (0.5, 50) if same_net else ((1.0, 10) if q else (0.1, 100))
                    out.append(_row(cond, edit, g, s, seed, 0 if q else fire_P, apl, kc))
    return out


def test_state_and_cell_stats_are_ns():
    assert R.state is n_rules.state and R.cell_stats is n_rules.cell_stats
    sp = _spec()
    rows = [dict(P=p) for p in (0, 4, 5, 40)]
    assert R.silent(rows, sp).tolist() == [True, True, False, False]


def test_auc_and_boot_weights():
    assert R.auc([2, 3], [1, 1]) == 1.0 and R.auc([1], [1]) == 0.5 and R.auc([], [1]) is None
    W = R.boot_weights(5, 7, 11)
    idx = n_rules.boot_index(5, 7, 11)
    assert np.allclose(W.sum(1), 1.0) and np.allclose(W[3] * 5, np.bincount(idx[3], minlength=5))


def test_the_gate_names_every_defect_and_judges_nothing():
    sp = _spec()
    rows = _rows(sp, lambda c, g, s, i: c == "on" and i < 12)
    assert R.o1_judge(rows, sp)["outcome"] == R.JUDGED
    cases = {
        "no row": rows[1:],
        "more than once": rows + [rows[0]],
        "undeclared seed": [dict(rows[0], seed=999)] + rows[1:],
        "non-finite": [dict(rows[0], apl_out_per_step=float("nan"))] + rows[1:],
        "several CSC": [dict(rows[0], csc_sha256="other")] + rows[1:],
        "share one CSC": [dict(r, csc_sha256="sha-none") if r["edit"] == "apl_to_nonkc_zero" else r for r in rows],
    }
    for why, bad in cases.items():
        res = R.o1_judge(bad, sp)
        assert res["outcome"] == R.INVALID and any(why in x for x in res["reasons"]), why
        assert "nature" not in res and "pathway" not in res


def test_fewer_than_two_mixed_cells_stops():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda *a: False), sp)
    assert res["outcome"] == R.STOP_NO_SILENT_STATE and res["mixed_cells"] == []
    assert res["nature"] is None and res["pathway"] is None and res["drive"] is None and res["cells"]["on"]
    one = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and g == 0.25 and s == "4:1" and i < 8), sp)
    assert one["outcome"] == R.STOP_NO_SILENT_STATE and one["mixed_cells"] == ["0.25|4:1"]


@pytest.mark.parametrize("k, mixed", [(4, True), (3, False)])
def test_the_mixed_threshold_is_ten_percent(k, mixed):
    sp = _spec()                                                    # 40 seeds: 4 -> 0.10 (mixed), 3 -> 0.075
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < k), sp)
    assert (res["outcome"] == R.JUDGED) is mixed and len(res["mixed_cells"]) == (4 if mixed else 0)


def test_bistable_network_and_readout_path():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c in ("on", "kc") and i < 12), sp)
    assert res["outcome"] == R.JUDGED and len(res["mixed_cells"]) == 4
    assert res["nature"]["label"] == R.BISTABLE_NETWORK
    assert all(c["auc_apl"] == 1.0 and c["auc_apl_silent_above"] == 1.0 for c in res["nature"]["cells"])
    assert res["pathway"]["label"] == R.READOUT_PATH and res["pathway"]["q"]["on"] == pytest.approx(0.3)
    assert {s: v["label"] for s, v in res["drive"].items()} == {"4:1": R.FLAT, "dDL": R.FLAT}
    assert res["phi_on"] and all(p["phi"] == pytest.approx(1.0) for p in res["phi_on"])
    assert res["cells"]["on"]["0.25|4:1"]["P"][:12] == [0] * 12


def test_bimodal_p_readout_when_the_network_does_not_split():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < 12, same_net=True), sp)
    assert res["nature"]["label"] == R.BIMODAL_P_READOUT
    assert all(c["auc_apl"] == 0.5 and c["a"] and not c["b"] for c in res["nature"]["cells"])


def test_graded_when_p_sits_in_the_mid_band():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < 12, fire_P=10), sp)
    assert res["nature"]["label"] == R.GRADED and all(c["mid_share"] == pytest.approx(0.7)
                                                      for c in res["nature"]["cells"])


@pytest.mark.parametrize("silent_conds, label", [
    (("on", "kc"), R.READOUT_PATH), (("on", "nonkc"), R.KC_NETWORK_PATH), (("on",), R.BOTH_PATHS),
    (("on", "kc", "nonkc"), R.NEITHER_PATH)])
def test_pathway_labels(silent_conds, label):
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c in silent_conds and i < 12), sp)
    assert res["pathway"]["label"] == label and res["pathway"]["record_all"] == 0.0


def test_a_partial_drop_above_a_quarter_does_not_remove():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: (c == "on" and i < 12) or (c == "kc" and i < 6)), sp)
    kc = res["pathway"]["arms"]["kc"]
    assert kc["diff_ci95"][1] < 0 and not kc["removes"]           # 0.15 > 0.25 x 0.3: the ratio clause fails
    assert res["pathway"]["label"] == R.READOUT_PATH


def test_drive_labels_per_stimulus():
    sp = _spec()
    up = {(0.25, "4:1"): 12, (1.0, "4:1"): 24, (0.25, "dDL"): 24, (1.0, "dDL"): 12}
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < up[(g, s)]), sp)
    d = res["drive"]
    assert d["4:1"]["label"] == R.INCREASING and d["4:1"]["slope_ci"][0] > 0
    assert d["dDL"]["label"] == R.DECREASING and d["dDL"]["slope"] == pytest.approx(-0.3)
    flat = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < (12 if g == 0.25 else 16)), sp)
    assert {v["label"] for v in flat["drive"].values()} == {R.FLAT}


def test_a_non_monotone_stimulus_is_unclear():
    sp = _spec(g=(0.25, 1.0, 4.0))
    k = {0.25: 8, 1.0: 24, 4.0: 8}
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < k[g]), sp)
    assert {v["label"] for v in res["drive"].values()} == {R.NONMONOTONIC_OR_UNCLEAR}
    assert all(v["slope"] == 0.0 and v["range"] == pytest.approx(0.4) for v in res["drive"].values())


def test_records_and_determinism():
    sp = _spec()
    rows = _rows(sp, lambda c, g, s, i: c in ("on", "kc") and i < 12)
    a, b = R.o1_judge(rows, sp), R.o1_judge(list(reversed(rows)), sp)
    assert a == b                                                   # row order and the bootstrap never matter
    assert a["kc_block_record"]["nature"]["label"] == R.BISTABLE_NETWORK
    assert a["sha"] == {c: "sha-" + e for c, e in sp.o1_conditions}
    rec = a["cells"]["on"]["0.25|4:1"]
    assert rec["silent_share"] == pytest.approx(0.3) and rec["apl_out"]["silent"][0] == 1.0
    assert "kc_frac" not in rec and rec["firing_share"] == pytest.approx(0.7)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_o_rules_o1.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.o_rules`).

- [ ] **Step 3: Write `o_rules.py` (part 1)**

```python
"""The pure rules of spec appendix O as amended by O.7 (O.7 wins over O.2-O.5). No engine, no pool.
- The validity gate both stages pass before any judgement (O.7.5).
- The seed-cluster bootstrap (O.7.2 / O.7.4): whole seeds resampled with replacement, every cell, condition and arm of
  a seed together; spec.boot_draws draws through n_rules.boot_index at spec.boot_seed; percentile CIs (n_rules._ci).
- O1 (O.7.2): mixed cells and STOP_NO_SILENT_STATE, the nature of the two states, the pathway 2x2, per-stimulus drive
  dependence, and the records (co-silence phi, per-cell P / APL / KC distributions, the APL->KC block's numbers).
- O2 (O.7.4): depression, dopamine dependence, the associative component, the state-flip guard, plumbing, records.
The presentation state is n_rules.state and the per-cell numbers n_rules.cell_stats: never redefined here (O.7.5).
Every threshold is a field of the OSpec passed in."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from .n_rules import INVALID, _ci, boot_index, cell_stats, delta, state  # noqa: F401  (INVALID, delta: re-exported)

JUDGED = "JUDGED"
STOP_NO_SILENT_STATE = "STOP_NO_SILENT_STATE"
BISTABLE_NETWORK = "BISTABLE_NETWORK"
BIMODAL_P_READOUT = "BIMODAL_P_READOUT"
GRADED = "GRADED"
READOUT_PATH = "READOUT_PATH"
KC_NETWORK_PATH = "KC_NETWORK_PATH"
BOTH_PATHS = "BOTH_PATHS"
NEITHER_PATH = "NEITHER_PATH"
FLAT = "FLAT"
INCREASING = "INCREASING"
DECREASING = "DECREASING"
NONMONOTONIC_OR_UNCLEAR = "NONMONOTONIC_OR_UNCLEAR"
DEPRESSION_PRESENT = "DEPRESSION_PRESENT"
NO_DEPRESSION = "NO_DEPRESSION"
INCONCLUSIVE = "INCONCLUSIVE"
DA_DEPENDENT = "DA_DEPENDENT"
DA_INDEPENDENT = "DA_INDEPENDENT"
SEPARABLE = "SEPARABLE"
NOT_SEPARABLE = "NOT_SEPARABLE"
FLIP_SUFFIX = "_STATE_FLIP_SENSITIVE"
PATH_LABELS = {(False, True): READOUT_PATH, (True, False): KC_NETWORK_PATH, (True, True): BOTH_PATHS,
               (False, False): NEITHER_PATH}                  # (APL->KC block removes, APL->non-KC block removes)


# ================================================================ the gate and the bootstrap (O.7.5)
def finite(o) -> bool:
    """Every number inside a row (nested dicts / lists) is finite; strings and booleans pass, None fails."""
    if isinstance(o, dict):
        return all(finite(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return all(finite(v) for v in o)
    if isinstance(o, (bool, np.bool_, str)):
        return True
    if isinstance(o, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(o))
    return False


def validity(rows, declared, key_of, seeds) -> list:
    """O.7.5's gate: the declared Cartesian product complete, keys unique, every value finite, no undeclared seed, one
    CSC sha256 per edit and the edits' vectors mutually distinct. The reasons, [] when valid."""
    bad = []
    cnt = Counter(key_of(r) for r in rows)
    dup = sorted((k for k, v in cnt.items() if v > 1), key=str)
    if dup:
        bad.append(f"{len(dup)} key(s) appear more than once, e.g. {dup[:3]}")
    missing = sorted(set(declared) - set(cnt), key=str)
    if missing:
        bad.append(f"{len(missing)} declared key(s) have no row, e.g. {missing[:3]}")
    extra = sorted(set(cnt) - set(declared), key=str)
    if extra:
        bad.append(f"{len(extra)} row key(s) were not declared, e.g. {extra[:3]}")
    off = sorted({int(r["seed"]) for r in rows} - {int(s) for s in seeds})
    if off:
        bad.append(f"undeclared seed(s) {off[:8]}")
    nonfinite = [key_of(r) for r in rows if not finite(r)]
    if nonfinite:
        bad.append(f"{len(nonfinite)} row(s) hold a non-finite value, e.g. {nonfinite[:3]}")
    sha = defaultdict(set)
    for r in rows:
        sha[r["edit"]].add(r["csc_sha256"])
    many = {e: len(s) for e, s in sorted(sha.items()) if len(s) != 1}
    if many:
        bad.append(f"edit(s) ran on several CSC weight vectors: {many}")
    one = [next(iter(s)) for _, s in sorted(sha.items()) if len(s) == 1]
    if len(one) != len(set(one)):
        bad.append("two edits share one CSC weight vector (an edit changed nothing)")
    return bad


def boot_weights(n: int, draws: int, seed: int) -> np.ndarray:
    """draws x n resampling weights (each row sums to 1) from n_rules.boot_index: a weighted mean over seeds is one
    seed-cluster bootstrap replicate."""
    idx = boot_index(n, draws, seed)
    W = np.zeros((int(draws), int(n)))
    np.add.at(W, (np.arange(int(draws))[:, None], idx), 1.0)
    return W / int(n)


def auc(pos, neg) -> float | None:
    """P(pos > neg) + 1/2 P(tie) (Mann-Whitney); None when a group is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return None
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


# ================================================================ O1 (O.7.2)
def cell_key(g, stim) -> str:
    return f"{float(g):g}|{stim}"


def o1_cells(spec) -> list:
    return [(float(g), s) for g in spec.o1_g_grid for s in spec.o1_stimuli]


def o1_key(r) -> tuple:
    return (r["cond"], float(r["g"]), r["stim"], int(r["seed"]))


def o1_declared(spec) -> set:
    return {(c, g, s, int(seed)) for c, _ in spec.o1_conditions for g, s in o1_cells(spec) for seed in spec.o1_seeds}


def silent(rows, spec) -> np.ndarray:
    return np.array([state(r["P"], spec.n) == "silent" for r in rows], bool)


def mixed_cells(S, spec) -> list:
    """Indices of the cells (rows of S, cells x seeds, 1 = silent) whose silent and firing shares are both
    >= mixed_min (reading 3)."""
    sil, fire = S.mean(1), (1.0 - S).mean(1)
    return [i for i in range(S.shape[0]) if sil[i] >= spec.mixed_min and fire[i] >= spec.mixed_min]


def cell_nature(rows, spec) -> dict:
    """(a) P mid-band share <= mid_max; (b) / (c) separation AUC (reading 5) of APL output / KC spikes, silent vs
    firing, >= auc_min."""
    s = silent(rows, spec)
    P = np.array([r["P"] for r in rows], float)
    lo, hi = spec.mid_band
    mid = float(((P >= lo) & (P < hi)).mean())
    out = dict(mid_share=mid, a=bool(mid <= spec.mid_max))
    for name, field in (("apl", "apl_out_per_step"), ("kc", "kc_spikes")):
        v = np.array([r[field] for r in rows], float)
        d = auc(v[s], v[~s])
        out[f"auc_{name}_silent_above"] = d
        out[f"auc_{name}"] = None if d is None else max(d, 1.0 - d)
    out["b"] = bool(out["auc_apl"] is not None and out["auc_apl"] >= spec.auc_min)
    out["c"] = bool(out["auc_kc"] is not None and out["auc_kc"] >= spec.auc_min)
    out["abc"] = bool(out["a"] and out["b"] and out["c"])
    return out


def nature_label(nat: list, spec) -> str:
    """BISTABLE_NETWORK if (a)(b)(c) hold in >= nature_share of the mixed cells, else BIMODAL_P_READOUT if (a) does,
    else GRADED (reading 6; integer comparison)."""
    n = len(nat)
    if n == 0:
        raise ValueError("no mixed cell to judge")
    num, den = spec.nature_share
    if den * sum(c["abc"] for c in nat) >= num * n:
        return BISTABLE_NETWORK
    if den * sum(c["a"] for c in nat) >= num * n:
        return BIMODAL_P_READOUT
    return GRADED


def pathway(S: dict, mixed: list, W, spec) -> dict:
    """O.7.2-3 over the mixed cells: pooled silent shares, and for each partial block c "removes" iff
    q_c <= path_ratio q_on and the CI of q_c - q_on lies below 0 (reading 7). The all-output block is a record."""
    on, kc, nonkc, all_ = (c for c, _ in spec.o1_conditions)
    q = {c: float(S[c][mixed].mean()) for c in (on, kc, nonkc, all_)}
    qb_on = (W @ S[on][mixed].T).mean(1)
    arms = {}
    for c in (kc, nonkc):
        ci = _ci((W @ S[c][mixed].T).mean(1) - qb_on, spec.ci_level)
        arms[c] = dict(q=q[c], diff=q[c] - q[on], diff_ci95=ci,
                       removes=bool(q[c] <= spec.path_ratio * q[on] and ci[1] < 0))
    return dict(label=PATH_LABELS[(arms[kc]["removes"], arms[nonkc]["removes"])], q=q, arms=arms, record_all=q[all_])


def drive(S, cells: list, W, spec) -> dict:
    """O.7.2-4 per stimulus: OLS slope of the cell silent share on the g rank (ascending o1_g_grid), its
    slope_ci_level bootstrap CI and the range of the point estimates (reading 8)."""
    gs = [float(g) for g in spec.o1_g_grid]
    if len(gs) < 2:
        raise ValueError("drive dependence needs at least two g")
    x = np.arange(len(gs), dtype=float)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    out = {}
    for stim in spec.o1_stimuli:
        M = S[[cells.index((g, stim)) for g in gs]]
        y = M.mean(1)
        slope = float(xc @ y / sxx)
        ci = _ci((W @ M.T) @ xc / sxx, spec.slope_ci_level)
        rng = float(y.max() - y.min())
        if rng < spec.flat_range:
            lab = FLAT
        elif ci[0] > 0:
            lab = INCREASING
        elif ci[1] < 0:
            lab = DECREASING
        else:
            lab = NONMONOTONIC_OR_UNCLEAR
        out[stim] = dict(label=lab, silent_share=y.tolist(), slope=slope, slope_ci=ci, range=rng)
    return out


def phi_pairs(S, cells: list, spec) -> list:
    """O.2's record: the seed-wise phi of co-silence for every pair of cells whose silent share is in phi_band."""
    lo, hi = spec.phi_band
    sh = S.mean(1)
    keep = [i for i in range(len(cells)) if lo <= sh[i] <= hi]
    return [dict(a=cell_key(*cells[i]), b=cell_key(*cells[j]), phi=float(np.corrcoef(S[i], S[j])[0, 1]),
                 same_stimulus=cells[i][1] == cells[j][1]) for i, j in combinations(keep, 2)]


def cell_record(rows, spec) -> dict:
    """n_rules.cell_stats (its per-presentation kc_frac list dropped) + the silent share, the sorted P values and the
    APL / KC quantiles by state."""
    s = silent(rows, spec)

    def qs(v):
        return [float(x) for x in np.quantile(v, spec.record_quantiles)] if v.size else None

    apl = np.array([r["apl_out_per_step"] for r in rows], float)
    kc = np.array([r["kc_spikes"] for r in rows], float)
    rec = {k: v for k, v in cell_stats(rows, spec.n).items() if k != "kc_frac"}
    return dict(rec, silent_share=float(s.mean()), P=sorted(int(r["P"]) for r in rows),
                apl_out={"silent": qs(apl[s]), "firing": qs(apl[~s])},
                kc_spikes={"silent": qs(kc[s]), "firing": qs(kc[~s])})


def _condition_view(S_c, by, cond, cells, W, spec) -> dict:
    m = mixed_cells(S_c, spec)
    nat = [dict(cell=cell_key(*cells[i]), **cell_nature(by[(cond,) + cells[i]], spec)) for i in m]
    return dict(mixed=m, mixed_cells=[cell_key(*cells[i]) for i in m],
                nature=dict(label=nature_label(nat, spec) if len(m) >= spec.mixed_cells_min else None, cells=nat),
                drive=drive(S_c, cells, W, spec))


def o1_judge(rows: list, spec) -> dict:
    """The gate (INVALID), then the records, then STOP_NO_SILENT_STATE (judgements 2-4 not made: reading 4) or the
    three judgements on the on condition."""
    bad = validity(rows, o1_declared(spec), o1_key, spec.o1_seeds)
    if bad:
        return dict(outcome=INVALID, reasons=bad, n_rows=len(rows))
    cells = o1_cells(spec)
    conds = [c for c, _ in spec.o1_conditions]
    on, kc = conds[0], conds[1]
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], float(r["g"]), r["stim"])].append(r)
    for k in by:
        by[k].sort(key=lambda r: int(r["seed"]))
    S = {c: np.array([silent(by[(c,) + cell], spec) for cell in cells], float) for c in conds}
    W = boot_weights(len(spec.o1_seeds), spec.boot_draws, spec.boot_seed)
    view_on = _condition_view(S[on], by, on, cells, W, spec)
    view_kc = _condition_view(S[kc], by, kc, cells, W, spec)
    rec = dict(reasons=[], mixed_cells=view_on["mixed_cells"],
               cells={c: {cell_key(*cell): cell_record(by[(c,) + cell], spec) for cell in cells} for c in conds},
               phi_on=phi_pairs(S[on], cells, spec),
               kc_block_record={k: view_kc[k] for k in ("mixed_cells", "nature", "drive")},
               sha={c: by[(c,) + cells[0]][0]["csc_sha256"] for c in conds})
    if len(view_on["mixed"]) < spec.mixed_cells_min:
        return dict(outcome=STOP_NO_SILENT_STATE, nature=None, pathway=None, drive=None, **rec)
    return dict(outcome=JUDGED, nature=view_on["nature"], pathway=pathway(S, view_on["mixed"], W, spec),
                drive=view_on["drive"], **rec)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_o_rules_o1.py -q -o addopts=""`
Expected: PASS (17 tests).

If `test_drive_labels_per_stimulus` fails on the INCREASING label, print the 99% CI before changing anything. With 40 seeds, the expected lower bound is ≈ 0.11. Never loosen `slope_ci_level`.

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_rules.py tests/brain/test_o_rules_o1.py
git commit -m "feat(o): o_rules O1 — validity gate, seed-cluster bootstrap, mixed cells, state nature (AUCs), pathway 2x2, per-stimulus drive labels"
```

---

### Task 6: `o_rules` (2) — O2 judgements, state-flip guard, plumbing, sentences

**Files:**
- Modify: `flymon/brain/o_rules.py` (append)
- Test: `tests/brain/test_o_rules_o2.py`

**Interfaces:**
- Consumes: Task 5's `validity`, `boot_weights`, labels; `n_rules.state`, `delta`, `_ci`.
- Produces:
  - keys: `o2_key(row)`, `o2_declared(spec) -> set`;
  - statistics: `o2_stats(by, z, o, spec, keep=None) -> dict | None`, where `by` is `{arm: rows in seed order}`;
  - the X judgement: `o2_x(by, z, o, spec) -> dict` with `outcome`, `reasons`, `labels`, `stats`, `flips`, `flip_limit`, `guard` and `record`;
  - the stage: `o2_judge(rows, z, o_by_pair, spec) -> dict` with `outcome`, `reasons` and `x` (a dict per X name);
  - `sentence(name, res) -> str` for `"o1"` / `"o2"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_o_rules_o2.py
"""Spec O.7.4: depression with its b band and INCONCLUSIVE; dopamine dependence only after DEPRESSION_PRESENT; the
associative component with e = 0.25 |o|; the state-flip guard (> 1/8 of seeds -> the same judgement without them,
changed labels suffixed); plumbing by sha256 (that X INVALID); the gate before anything (the stage INVALID)."""
from dataclasses import replace

import pytest

from flymon.brain import o_rules as R
from flymon.brain.o_spec import SPEC

S0 = 500
Z = {"A": [0.0, 1.0], "P": [0.0, 1.0]}                     # dV = (A_x - P_x) - (A_y - P_y)
O = dict(SPEC.oracle_o)


def _spec(n=16, draws=500):
    return replace(SPEC, o2_seed0=S0, o2_n_seeds=n, boot_draws=draws)


def _p(A, P=40):
    return dict(seed=0, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1, wall_s=0.01,
                steps=1400)


def _rows(spec, post_A, pre_A=20.0, post_P=lambda x, a, i: 40, moved=lambda x, a, i: a in ("plastic", "punish")):
    out = []
    for x, y, _ in spec.o2_pairs:
        for a, pu, pl, dz in spec.o2_arms:
            for i, s in enumerate(spec.o2_seeds):
                out.append(dict(seed=s, edit="none", arm=a, punish=pu, plastic=pl, da_zero=dz, csc_sha256="sha-none",
                                pre={"x": _p(pre_A), "y": _p(20.0)},
                                post={"x": _p(post_A(x, a, i), post_P(x, a, i)), "y": _p(20.0)},
                                weights_frac=0.9, weights_frac_A=0.8, weights_frac_P=1.0, w0_sha256="w0",
                                w_post_sha256="w1" if moved(x, a, i) else "w0",
                                da_integral={"PPL105": 1.0, "PAM08": 0.1}, wall_s=1.0, x=x, y=y))
    return out


def _A(plastic, punish=None, da_zero=0.0):
    """post A_X: 20 minus the arm's drop; plastic and punish carry a 0.5 jitter on odd seeds."""
    drop = {"plastic": plastic, "frozen": 0.0, "punish": plastic if punish is None else punish, "da_zero": da_zero}
    return lambda x, a, i: 20.0 - drop[a] - (0.5 * (i % 2) if a in ("plastic", "punish") else 0.0)


def _x(res, x="4:1"):
    return res["x"][x]


def test_depression_present_and_dopamine_dependent():
    sp = _spec()
    res = R.o2_judge(_rows(sp, _A(10.0)), Z, O, sp)
    v = _x(res)
    assert res["outcome"] == R.JUDGED and v["outcome"] == R.JUDGED
    assert v["labels"] == dict(depression=R.DEPRESSION_PRESENT, da=R.DA_DEPENDENT, separable=R.NOT_SEPARABLE)
    assert v["stats"]["b"] == 5.0 and v["stats"]["m"] == pytest.approx(-10.25) and v["stats"]["m4_ci"] == [0.0, 0.0]
    assert v["record"]["frozen_dA_all_zero"] and v["record"]["da_zero_weights_unmoved"]
    assert v["guard"] is None and v["o"] == -2.348 and res["x"]["dDL"]["o"] == -2.433


def test_dopamine_independent():
    v = _x(R.o2_judge(_rows(_spec(), _A(10.0, da_zero=10.0)), Z, O, _spec()))
    assert v["labels"]["da"] == R.DA_INDEPENDENT


def test_no_depression_and_no_da_judgement():
    v = _x(R.o2_judge(_rows(_spec(), _A(0.0)), Z, O, _spec()))
    assert v["labels"]["depression"] == R.NO_DEPRESSION and v["labels"]["da"] is None


def test_inconclusive_when_the_ci_straddles_b():
    post = lambda x, a, i: 20.0 - (9.0 if a in ("plastic", "punish") and i % 2 == 0 else 0.0)
    v = _x(R.o2_judge(_rows(_spec(), post), Z, O, _spec()))
    assert v["stats"]["m"] == pytest.approx(-4.5) and v["labels"]["depression"] == R.INCONCLUSIVE


def test_separable_and_its_inconclusive_band():
    sp = _spec()
    v = _x(R.o2_judge(_rows(sp, _A(10.0, punish=20.0)), Z, O, sp))
    assert v["labels"]["separable"] == R.SEPARABLE and v["stats"]["D"] == pytest.approx(-10.0)
    assert v["stats"]["D_sign"] == -1 and v["stats"]["e"] == pytest.approx(0.25 * 2.348)
    post = lambda x, a, i: 20.0 - 10.0 - (1.0 if a == "punish" and i % 2 == 0 else 0.0) if a in ("plastic", "punish") \
        else 20.0
    w = _x(R.o2_judge(_rows(sp, post), Z, O, sp))
    assert w["stats"]["D"] == pytest.approx(-0.5) and w["labels"]["separable"] == R.INCONCLUSIVE


def test_plumbing_failure_invalidates_only_that_x():
    sp = _spec()
    moved = lambda x, a, i: a in ("plastic", "punish") or (x == "4:1" and a == "frozen" and i == 3)
    res = R.o2_judge(_rows(sp, _A(10.0), moved=moved), Z, O, sp)
    assert res["outcome"] == R.JUDGED
    assert _x(res)["outcome"] == R.INVALID and _x(res)["labels"] is None and "plumbing" in _x(res)["reasons"][0]
    assert _x(res, "dDL")["outcome"] == R.JUDGED


def test_flip_guard_suffixes_the_labels_that_change():
    sp = _spec()                                                          # 16 seeds: limit 2
    post_A = lambda x, a, i: 0.0 if a in ("plastic", "punish") and i < 4 else 20.0
    post_P = lambda x, a, i: 0 if a in ("plastic", "punish") and i < 4 else 40
    v = _x(R.o2_judge(_rows(sp, post_A, post_P=post_P), Z, O, sp))
    assert v["stats"]["m"] == -5.0 and v["flip_limit"] == 2.0 and len(v["flips"]["plastic"]) == 4
    assert v["labels"]["depression"] == R.DEPRESSION_PRESENT + R.FLIP_SUFFIX
    assert v["labels"]["da"] == R.DA_DEPENDENT + R.FLIP_SUFFIX
    assert v["labels"]["separable"] == R.NOT_SEPARABLE                     # unchanged without the flipped seeds
    assert v["guard"]["dropped"] == [S0, S0 + 1, S0 + 2, S0 + 3]
    assert v["guard"]["alt"]["labels"]["depression"] == R.NO_DEPRESSION


def test_flips_at_the_limit_do_not_trigger_the_guard():
    sp = _spec()
    post_P = lambda x, a, i: 0 if a == "plastic" and i < 2 else 40
    v = _x(R.o2_judge(_rows(sp, _A(10.0), post_P=post_P), Z, O, sp))
    assert v["guard"] is None and v["labels"]["depression"] == R.DEPRESSION_PRESENT


def test_a_guard_leaving_fewer_than_two_seeds_marks_every_label():
    sp = _spec(n=4)
    post_P = lambda x, a, i: 0 if a == "plastic" and i < 3 else 40
    v = _x(R.o2_judge(_rows(sp, _A(10.0), post_P=post_P), Z, O, sp))
    assert v["guard"]["alt"] is None
    assert all(l.endswith(R.FLIP_SUFFIX) for l in v["labels"].values() if l is not None)


def test_zero_baseline_is_judged_not_crashed():
    sp = _spec()
    v = _x(R.o2_judge(_rows(sp, lambda x, a, i: 0.0, pre_A=0.0), Z, O, sp))
    assert v["stats"]["b"] == 0.0 and v["labels"]["depression"] == R.NO_DEPRESSION


def test_state_conditional_record_keeps_all_firing_seeds():
    sp = _spec()
    post_P = lambda x, a, i: 0 if a == "punish" and i == 0 else 40
    v = _x(R.o2_judge(_rows(sp, _A(10.0), post_P=post_P), Z, O, sp))
    assert v["record"]["state_conditional"]["n"] == 15
    assert v["record"]["da_integral"]["punish"] == {"PPL105": 1.0, "PAM08": 0.1}


def test_the_gate_judges_nothing():
    sp = _spec()
    rows = _rows(sp, _A(10.0))
    for bad, why in ((rows[1:], "no row"), ([dict(rows[0], seed=1)] + rows[1:], "undeclared seed"),
                     ([dict(rows[0], weights_frac=float("inf"))] + rows[1:], "non-finite")):
        res = R.o2_judge(bad, Z, O, sp)
        assert res["outcome"] == R.INVALID and res["x"] is None and any(why in r for r in res["reasons"])


def test_sentences():
    sp = _spec()
    s = R.sentence("o2", R.o2_judge(_rows(sp, _A(10.0)), Z, O, sp))
    assert s.startswith("O2: X = 4:1: DEPRESSION_PRESENT, DA_DEPENDENT") and "학습 주장이 아니다" in s
    assert "INVALID" in R.sentence("o1", dict(outcome=R.INVALID, reasons=["x"]))
    with pytest.raises(ValueError):
        R.sentence("o3", dict(outcome=R.JUDGED))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_o_rules_o2.py -q -o addopts=""`
Expected: FAIL (`AttributeError: module 'flymon.brain.o_rules' has no attribute 'o2_judge'`).

- [ ] **Step 3: Append to `o_rules.py`**

```python
# ================================================================ O2 (O.7.4)
def o2_key(r) -> tuple:
    return (r["x"], r["arm"], int(r["seed"]))


def o2_declared(spec) -> set:
    return {(x, a[0], int(s)) for x, _, _ in spec.o2_pairs for a in spec.o2_arms for s in spec.o2_seeds}


def _diff(rows, side: str, field: str) -> np.ndarray:
    return np.array([r["post"][side][field] - r["pre"][side][field] for r in rows], float)


def _inside(ci, b: float) -> bool:
    return bool(-b <= ci[0] and ci[1] <= b)


def o2_stats(by: dict, z: dict, o: float, spec, keep=None) -> dict | None:
    """by {arm: rows in seed order}, arms named as spec.o2_arms (1 plastic, 2 frozen, 3 punish, 4 da_zero). On the seeds
    in `keep` (all when None); None with fewer than 2. One seed resample (boot_weights) for every statistic."""
    one, two, three, four = (a[0] for a in spec.o2_arms)
    if keep is not None:
        by = {a: [r for r in rows if int(r["seed"]) in keep] for a, rows in by.items()}
    n = len(by[one])
    if n < 2:
        return None
    W = boot_weights(n, spec.boot_draws, spec.boot_seed)

    def stat(v):
        return float(np.mean(v)), _ci(W @ v, spec.ci_level)

    dA = {a: _diff(rows, "x", "A") for a, rows in by.items()}
    b = spec.depression_frac * float(np.mean([r["pre"]["x"]["A"] for r in by[one]]))
    m, m_ci = stat(dA[one] - dA[two])
    if m_ci[1] < 0 and m <= -b:
        dep = DEPRESSION_PRESENT
    elif _inside(m_ci, b):
        dep = NO_DEPRESSION
    else:
        dep = INCONCLUSIVE
    m4, m4_ci = stat(dA[four] - dA[two])
    da = None
    if dep == DEPRESSION_PRESENT:
        da = (DA_DEPENDENT if _inside(m4_ci, b) else
              DA_INDEPENDENT if (m4 <= -b and m4_ci[1] < 0) else INCONCLUSIVE)
    ddv = {a: np.array([delta(r, z) for r in rows]) for a, rows in by.items()}
    e = spec.sep_frac * abs(float(o))
    D, D_ci = stat(ddv[three] - ddv[one])
    if (D_ci[0] > 0 or D_ci[1] < 0) and abs(D) >= e:
        sep = SEPARABLE
    elif _inside(D_ci, e):
        sep = NOT_SEPARABLE
    else:
        sep = INCONCLUSIVE
    same = {f"{side}_{f}": stat(_diff(by[one], side, f) - _diff(by[two], side, f))
            for side, f in (("x", "P"), ("y", "A"), ("y", "P"))}
    return dict(n=n, seeds=[int(r["seed"]) for r in by[one]], b=b, e=e, m=m, m_ci=m_ci, m4=m4, m4_ci=m4_ci, D=D,
                D_ci=D_ci, D_sign=int(np.sign(D)), labels=dict(depression=dep, da=da, separable=sep),
                mean_dA={a: float(v.mean()) for a, v in dA.items()},
                same_differences={k: dict(mean=v[0], ci=v[1]) for k, v in same.items()})


def o2_x(by: dict, z: dict, o: float, spec) -> dict:
    """One X: the plumbing check (arm 2's post-training sha256 == w0's, every seed; else INVALID), the judgements, the
    state-flip guard (reading 14) and the records."""
    one, two, three, four = (a[0] for a in spec.o2_arms)
    bad = [f"plumbing: arm {two} seed {r['seed']} moved its weights (sha256 {r['w_post_sha256'][:12]} != w0 "
           f"{r['w0_sha256'][:12]})" for r in by[two] if r["w_post_sha256"] != r["w0_sha256"]]
    if bad:
        return dict(outcome=INVALID, reasons=bad, labels=None)
    st = o2_stats(by, z, o, spec)
    if st is None:
        raise ValueError("O2 needs at least 2 seeds per arm")
    flips = {a: [int(r["seed"]) for r in rows if state(r["pre"]["x"]["P"], spec.n) != state(r["post"]["x"]["P"],
                                                                                              spec.n)]
             for a, rows in by.items()}
    limit = spec.flip_max_frac * st["n"]
    labels, guard = dict(st["labels"]), None
    if any(len(v) > limit for v in flips.values()):
        drop = sorted(set().union(*(set(v) for v in flips.values())))
        alt = o2_stats(by, z, o, spec, keep=set(st["seeds"]) - set(drop))
        alt_labels = alt["labels"] if alt else {k: None for k in labels}
        for k, v in labels.items():
            if v is not None and alt_labels[k] != v:
                labels[k] = v + FLIP_SUFFIX
        guard = dict(dropped=drop, alt=alt)
    firing = set(st["seeds"])
    for rows in by.values():
        for r in rows:
            if any(state(r[ph][k]["P"], spec.n) != "firing" for ph in ("pre", "post") for k in ("x", "y")):
                firing.discard(int(r["seed"]))
    record = dict(
        frozen_dA_all_zero=bool(all(r["post"]["x"]["A"] == r["pre"]["x"]["A"] for r in by[two])),
        frozen_probes_identical=bool(all(r["pre"][k][f] == r["post"][k][f] for r in by[two] for k in ("x", "y")
                                         for f in ("A", "P", "kc_spikes"))),
        da_zero_weights_unmoved=bool(all(r["w_post_sha256"] == r["w0_sha256"] for r in by[four])),
        state_conditional=dict(n=len(firing), stats=o2_stats(by, z, o, spec, keep=firing)),
        da_integral={a: {k: float(np.mean([r["da_integral"][k] for r in rows])) for k in rows[0]["da_integral"]}
                     for a, rows in by.items()},
        weights_frac={a: {k: float(np.mean([r[f] for r in rows]))
                          for k, f in (("all", "weights_frac"), ("A", "weights_frac_A"), ("P", "weights_frac_P"))}
                      for a, rows in by.items()},
        a_x_course=None, a_x_course_note="훈련 중 프로브가 rig에 없다(N2 훈련 루프) — 제시 횟수에 따른 A_X 경과는 기록하지 않는다")
    return dict(outcome=JUDGED, reasons=[], labels=labels, stats=st, flips=flips, flip_limit=limit, guard=guard,
                record=record)


def o2_judge(rows: list, z: dict, o_by_pair: dict, spec) -> dict:
    """The gate (the stage INVALID), then every X of spec.o2_pairs with its pair's o (reading 15)."""
    bad = validity(rows, o2_declared(spec), o2_key, spec.o2_seeds)
    if bad:
        return dict(outcome=INVALID, reasons=bad, x=None)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["x"]][r["arm"]].append(r)
    xs = {}
    for x, y, pair in spec.o2_pairs:
        arms = {a[0]: sorted(by[x][a[0]], key=lambda r: int(r["seed"])) for a in spec.o2_arms}
        xs[x] = dict(y=y, pair=pair, o=float(o_by_pair[pair]), **o2_x(arms, z, o_by_pair[pair], spec))
    return dict(outcome=JUDGED, reasons=[], x=xs)


# ================================================================ sentences
def sentence(name: str, res: dict) -> str:
    o = res.get("outcome")
    if name not in ("o1", "o2"):
        raise ValueError(f"unknown stage {name!r}")
    if o == INVALID:
        return f"{name.upper()} INVALID: {'; '.join(res['reasons'])}"
    if name == "o1":
        n = len(res["mixed_cells"])
        if o == STOP_NO_SILENT_STATE:
            return (f"O1: 켬 조건의 혼합 칸 {n}개 — STOP_NO_SILENT_STATE. 실제 냄새 rig에서는 특성화할 두 상태가 없다. "
                    "사용자에게 보고한다.")
        drive_txt = ", ".join(f"{s} {v['label']}" for s, v in res["drive"].items())
        return (f"O1: 혼합 칸 {n}개 — 상태 {res['nature']['label']}, 경로 {res['pathway']['label']}, "
                f"구동 의존성 {drive_txt}. 이 커넥톰 모델(C3)의 성질이다.")
    parts = []
    for x, v in res["x"].items():
        if v["outcome"] == INVALID:
            parts.append(f"X = {x}: INVALID ({'; '.join(v['reasons'][:2])})")
            continue
        lab = v["labels"]
        parts.append(f"X = {x}: {lab['depression']}" + (f", {lab['da']}" if lab["da"] else "") + f", {lab['separable']}")
    return "O2: " + " / ".join(parts) + ". 학습 주장이 아니다."
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_o_rules_o1.py tests/brain/test_o_rules_o2.py -q -o addopts=""`
Expected: PASS (30 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_rules.py tests/brain/test_o_rules_o2.py
git commit -m "feat(o): o_rules O2 — depression / DA dependence / separability with INCONCLUSIVE bands, state-flip guard, sha256 plumbing, sentences"
```

---

### Task 7: `o_cli`, `scripts/run_o1.py`, `scripts/run_o2.py`

**Files:**
- Create: `flymon/brain/o_cli.py`, `scripts/run_o1.py`, `scripts/run_o2.py`
- Test: `tests/test_run_o.py`

**Interfaces:**
- Consumes:
  - from earlier tasks: `o_measure` (guard, `OCache`, `OMeasurer`, `HASHED_FILES`, `MEASURE_FILES`, `SUMMARY`, `ALLOWED_DIR`); `o_rules.o1_judge`, `o2_judge`, `sentence`, `JUDGED`, `INVALID`; `o_spec.SPEC`, `smoke`;
  - from N and shared code: `n_cli.NPZ`, `load_c3`, `m0d_sha`, `model_types`, `glomeruli`, `stimuli_at`, `drives`; `n_rules.wall_per_step`; `n_store.SUMMARY`; `l_cli.refuse`, `run_id`, `check_committed`; `h3_store.ROOT`, `code_key`, `git_state`, `sha256_file`; `fly_pool.FlyPool`.
- Produces:
  - `o_cli` constants and hooks: `HOOK_NAMES`, `hooks(module)`, `out_allowed`, `git_state`, `code_keys`, `load_c3`, `m0d_sha`, `model_types`, `spec_commit(path)`, `n_oracle(spec, smoke, committed=check_committed, path=None) -> (extra | None, refusal | None)`, `make_measurer(ctx)`;
  - `o_cli` stage machinery: `nview(ctx)`, `write_report`, `Ctx`, `main_stage(name, argv, body, hooks, *, spec=None, require_root=True, doc_help=None, prepare=None) -> int`;
  - the scripts: `run_o1.main(argv=None, spec=None, require_root=True)`, `run_o1.items_o1(spec, st)`, `run_o2.main(...)` and `run_o2.items_o2(spec, st)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_o.py
"""Spec O.5 / O.7.5 and readings 13, 15, 18, 19: the O CLIs refuse before any pool on another root, an --out outside
results/o/, dirty hashed files, a block already in the real summary (no post-hoc rewrite), an uncommitted spec, and (O2)
a block n1 whose oracle o disagrees with O.7.4's; O1 asks for every (condition, cell, seed) once with the four edits and
writes its block (JUDGED exit 0, STOP_NO_SILENT_STATE exit 5); O2 asks for every (X, arm, seed) with the declared flags
and writes its block. Every O hashed file exists."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flymon.brain import o_cli
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_measure import HASHED_FILES
from flymon.brain.o_rules import BISTABLE_NETWORK, DEPRESSION_PRESENT, JUDGED, READOUT_PATH, STOP_NO_SILENT_STATE
from flymon.brain.o_spec import SPEC, smoke
from flymon.brain.odor_real import load_table

ROOT = Path(__file__).resolve().parents[1]
SM = smoke(SPEC)
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])
KEY = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}
TYPES = sorted({"ORN_" + p for v in load_table(ROOT / N_SPEC.data_dir, N_SPEC.sha_pins()).glomeruli.values() for p in v})
ORACLE = dict(declared=dict(SPEC.oracle_o), exact={"sim": -2.3482, "dis": -2.4325})
STAGES = ["run_o1", "run_o2"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def _prow(seed, A=20.0, P=40, apl=0.1, kc=100):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=kc, kc_max_win_hz=50.0, apl_out_per_step=apl,
                wall_s=0.01, steps=1400)


class FakePool:
    def close(self):
        pass


class FakeMeasurer:
    """O1: on / kc silent on every second seed (P 0, APL 1.0, KC 10), else firing; O2: arm drops plastic 10, punish 15."""

    def __init__(self, silent=lambda i: i % 2 == 0):
        self.silent, self.calls = silent, []

    def o1_presentations(self, params, items, readout):
        self.calls.append(("o1", sorted({i["edit"] for i in items}), len(items)))
        out = []
        for it in items:
            q = self.silent(it["seed"] - SM.o1_seed0) and it["cond"] in ("on", "kc")
            out.append(dict(_prow(it["seed"], P=0 if q else 40, apl=1.0 if q else 0.1, kc=10 if q else 100),
                            edit=it["edit"], csc_sha256="sha-" + it["edit"], cond=it["cond"], g=it["g"],
                            stim=it["stim"]))
        return out

    def o2_arms(self, params, items, readout, punish_type):
        self.calls.append(("o2", sorted({(i["arm"], i["punish"], i["plastic"], i["da_zero"]) for i in items}),
                           len(items)))
        out = []
        for it in items:
            i = it["seed"] - SM.o2_seed0
            drop = {"plastic": 10.0, "frozen": 0.0, "punish": 15.0, "da_zero": 0.0}[it["arm"]]
            jit = 0.5 * (i % 2) if it["arm"] in ("plastic", "punish") else 0.0
            moved = it["arm"] in ("plastic", "punish")
            out.append(dict(seed=it["seed"], edit=it["edit"], arm=it["arm"], punish=it["punish"],
                            plastic=it["plastic"], da_zero=it["da_zero"], csc_sha256="sha-none",
                            pre={"x": _prow(it["seed"]), "y": _prow(it["seed"])},
                            post={"x": _prow(it["seed"], A=20.0 - drop - jit), "y": _prow(it["seed"])},
                            weights_frac=0.9 if moved else 1.0, weights_frac_A=0.8 if moved else 1.0,
                            weights_frac_P=1.0, w0_sha256="w0", w_post_sha256="w1" if moved else "w0",
                            da_integral={"PPL105": 1.0}, wall_s=1.0, x=it["x"], y=it["y"]))
        return out


def _patch(mod, monkeypatch, measurer=None, commit="c0ffee", oracle=(ORACLE, None)):
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "code_keys", lambda npz: KEY)
    monkeypatch.setattr(mod, "m0d_sha", lambda spec: "s" * 64)
    monkeypatch.setattr(mod, "load_c3", lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, ZU, None))
    monkeypatch.setattr(mod, "model_types", lambda npz: TYPES)
    monkeypatch.setattr(mod, "spec_commit", lambda path: commit)
    monkeypatch.setattr(mod, "n_oracle", lambda spec, smoke: oracle)
    monkeypatch.setattr(mod, "make_measurer",
                        lambda ctx: (measurer, FakePool()) if measurer else pytest.fail("no pool may start here"))
    monkeypatch.setattr(o_cli, "out_allowed", lambda out: True)


def test_every_hashed_file_exists():
    assert [f for f in HASHED_FILES if not (ROOT / f).exists()] == []


@pytest.mark.parametrize("name", STAGES)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_out_outside_results_o(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/n/x"]) == 2 and "results/o/" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_dirty_hashed_files(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/o_jobs.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


@pytest.mark.parametrize("name, block", [("run_o1", "o1"), ("run_o2", "o2")])
def test_never_rewrites_a_block_of_the_real_summary(name, block, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    p = tmp_path / "results/summary/o_states.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({block: {"outcome": JUDGED}}))
    assert mod.main([], require_root=False) == 2 and "post-hoc" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_uncommitted_spec(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, commit=None)
    assert mod.main([], require_root=False) == 2 and "spec" in capsys.readouterr().err


def test_o2_refuses_a_disagreeing_oracle(tmp_path, monkeypatch, capsys):
    mod = _script("run_o2")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, oracle=(None, "block n1's oracle effects disagree"))
    assert mod.main(["--smoke"], require_root=False) == 2 and "disagree" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_n_oracle_reads_block_n1(tmp_path):
    s = tmp_path / "n.json"
    s.write_text(json.dumps({"n1": {"pairs": {"sim": {"o": -2.3482251}, "dis": {"o": -2.4325499}}}}))
    ok, why = o_cli.n_oracle(SPEC, False, committed=lambda *a: None, path=s)
    assert why is None and ok["declared"] == dict(SPEC.oracle_o) and ok["exact"]["sim"] == -2.3482251
    s.write_text(json.dumps({"n1": {"pairs": {"sim": {"o": -2.36}, "dis": {"o": -2.4325}}}}))
    assert "disagree" in o_cli.n_oracle(SPEC, False, committed=lambda *a: None, path=s)[1]
    assert o_cli.n_oracle(SPEC, False, committed=lambda *a: "not tracked", path=s)[1] == "not tracked"
    ok, why = o_cli.n_oracle(SPEC, True, committed=lambda *a: "never asked", path=tmp_path / "none.json")
    assert why is None and ok["exact"] is None


def test_o1_smoke_judges_and_asks_for_every_unit_once(tmp_path, monkeypatch, capsys):
    mod = _script("run_o1")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    assert fm.calls == [("o1", sorted(e for _, e in SPEC.o1_conditions), SM.n_o1_presentations())]
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o1"]
    assert doc["outcome"] == JUDGED and doc["nature"]["label"] == BISTABLE_NETWORK
    assert doc["pathway"]["label"] == READOUT_PATH and doc["spec_commit"] == "c0ffee" and doc["smoke"] is True
    assert Path(doc["report"]).exists() and "O1: 혼합 칸" in capsys.readouterr().out


def test_o1_without_silence_stops_with_exit_5(tmp_path, monkeypatch):
    mod = _script("run_o1")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(silent=lambda i: False))
    assert mod.main(["--smoke"], require_root=False) == 5
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o1"]
    assert doc["outcome"] == STOP_NO_SILENT_STATE and doc["nature"] is None


def test_o2_smoke_judges_both_x(tmp_path, monkeypatch, capsys):
    mod = _script("run_o2")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    assert fm.calls == [("o2", sorted(SPEC.o2_arms), SM.n_o2_arms())]
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o2"]
    assert doc["outcome"] == JUDGED and set(doc["x"]) == {x for x, _, _ in SPEC.o2_pairs}
    assert doc["x"]["4:1"]["labels"]["depression"] == DEPRESSION_PRESENT
    assert doc["oracle"] == json.loads(json.dumps(ORACLE)) and doc["point"] == {"g": 0.25, "c_delta": 8.0}
    assert "O2: X = 4:1" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_run_o.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.o_cli`).

- [ ] **Step 3: Write `o_cli.py`**

```python
"""The CLI layer of spec appendix O (O.5, O.7.5, readings 13 and 18-19). One copy; run_o1.py / run_o2.py import it.
- Refusals before any pool, in order: another directory; --out outside results/o/; dirty hashed files (unless
  --allow-dirty); an unreadable summary; outside --smoke, a block already in the real summary (no post-hoc rewrite); the
  stage's `prepare` (O2: block n1's oracle o); outside --smoke, a spec with no commit; no usable C3 record.
- The hooks a script imports and passes as `hooks(its own module)`, so its tests patch them on the script.
- N's helpers are imported, never edited: the C3 record (n_cli.load_c3 on spec.n), the model ORN types, the Hallem
  stimuli at a point (n_cli.stimuli_at on `nview`) and the drive record.
- The block, the report and the summary go through o_measure's guard; each carries the run id, the code keys, the git
  state and the spec commit (O.7.5).
O1 and O2 are independent (O.5): no block needs another."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import h3_store, n_cli, n_store, o_measure
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import check_committed, refuse, run_id
from .o_measure import HASHED_FILES, MEASURE_FILES, OCache, OMeasurer
from .o_spec import SPEC, OSpec, smoke
from .odor_real import DataMismatch

NPZ = n_cli.NPZ
RUN_OUT = o_measure.ALLOWED_DIR + "run"
SMOKE_OUT = o_measure.ALLOWED_DIR + "smoke"
SMOKE_SUMMARY = f"{SMOKE_OUT}/{Path(o_measure.SUMMARY).name}"
HOOK_NAMES = ("code_keys", "git_state", "load_c3", "m0d_sha", "make_measurer", "model_types", "spec_commit",
              "n_oracle")


def hooks(module) -> dict:
    return {n: getattr(module, n) for n in HOOK_NAMES}


def out_allowed(out) -> bool:
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(o_measure.ALLOWED_DIR)


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES); a missing hashed file is a refusal (exit 2)."""
    missing = [f for f in HASHED_FILES if not (ROOT / f).exists()]
    if missing:
        refuse(f"the hashed files {missing} do not exist")
        raise SystemExit(2)
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=HASHED_FILES)


def load_c3(spec) -> tuple:
    return n_cli.load_c3(spec.n)


def m0d_sha(spec) -> str:
    return n_cli.m0d_sha(spec.n)


def model_types(npz) -> list:
    return n_cli.model_types(npz)


def spec_commit(path) -> str | None:
    """The last commit touching the spec (O.7.5's 스펙 커밋), None when it has none."""
    r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(path)], capture_output=True,
                       text=True)
    return r.stdout.strip() or None


def n_oracle(spec, smoke: bool, committed=check_committed, path=None) -> tuple:
    """Reading 13: ({"declared", "exact"}, None) or (None, refusal). e uses O.7.4's declared o; outside --smoke N's
    summary must be committed and its block n1's exact o agree within oracle_o_tol. --smoke without a readable block
    records exact None."""
    declared = dict(spec.oracle_o)
    path = Path(path) if path is not None else ROOT / n_store.SUMMARY
    if not smoke:
        why = committed(path, ["n1"])
        if why:
            return None, why
    try:
        pairs = json.loads(path.read_text())["n1"]["pairs"]
        exact = {k: float(pairs[k]["o"]) for k in declared}
    except (OSError, ValueError, KeyError, TypeError) as e:
        if smoke:
            return dict(declared=declared, exact=None), None
        return None, f"no usable block n1 in {path}: {e!r}"
    off = {k: (exact[k], v) for k, v in declared.items() if abs(exact[k] - v) > spec.oracle_o_tol}
    if off:
        return None, f"block n1's oracle effects {off} disagree with O.7.4's declared values"
    return dict(declared=declared, exact=exact), None


def make_measurer(ctx) -> tuple:
    """(OMeasurer, its FlyPool): the one pool construction of O's CLIs. The caller closes the pool."""
    from . import fly_pool
    h3 = ctx.spec.n.h3
    pool = fly_pool.FlyPool(ctx.args.npz, ctx.c3, flies=[{}] * ctx.args.workers, workers=ctx.args.workers,
                            punish_type=h3.punish_type, reward_type=h3.reward_type, timeout_s=ctx.spec.pool_timeout_s)
    return OMeasurer(pool, ctx.spec, OCache(ctx.out / "cache", ctx.key, ctx.rid, ctx.spec_commit)), pool


def nview(ctx):
    """The context as n_cli's stimulus helpers read it (spec = N's)."""
    return SimpleNamespace(spec=ctx.spec.n, c3=ctx.c3, types=ctx.types)


def spec_record(spec) -> dict:
    return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "n"}


def write_report(out, rid: str, name: str, res: dict, params_list) -> Path:
    base = Path(out) / "runs" / f"{rid}-{name}"
    path = o_measure.write_json(Path(f"{base}.json"), res, params_list)
    md = f"# O {name} — {rid}\n\n{res.get('sentence', '')}\n\noutcome: `{res.get('outcome')}`\n"
    o_measure.write_bytes(Path(f"{base}.md"), md.encode(), params_list)
    return path


@dataclass
class Ctx:
    args: argparse.Namespace
    spec: OSpec
    smoke: bool
    rid: str
    out: Path
    key: dict
    c3: object
    readout: dict
    z: dict
    types: list
    hooks: dict
    spec_commit: str | None
    extra: dict | None


def parser(doc_help: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--workers", type=int, default=SPEC.workers)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    return ap


def main_stage(name: str, argv, body, hooks: dict, *, spec=None, require_root: bool = True,
               doc_help: str | None = None, prepare=None) -> int:
    """The refusals (module docstring), then body(ctx) -> (res, exit code); a DataMismatch inside the body is a
    refusal. Any other error propagates: nothing is written. The report goes under <out>/runs/ and block `name` into
    the summary (a --smoke run: its smoke summary). `prepare(spec, smoke, hooks) -> (extra, refusal)` runs before the
    pool; its extra is ctx.extra."""
    a = parser(doc_help).parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or (SMOKE_OUT if a.smoke else RUN_OUT))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {o_measure.ALLOWED_DIR} of the repository root")
    summary = Path(a.summary or (SMOKE_SUMMARY if a.smoke else o_measure.SUMMARY))
    if a.smoke and not out_allowed(summary):
        return refuse(f"a --smoke run writes only a smoke summary under {o_measure.ALLOWED_DIR}, not {summary}")
    git = hooks["git_state"](HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    try:
        doc = json.loads(summary.read_text()) if summary.exists() else {}
    except (OSError, ValueError) as e:
        return refuse(f"no usable summary at {summary}: {e}")
    if not a.smoke and isinstance(doc, dict) and name in doc:
        return refuse(f"{summary} already holds block {name}: each verdict goes to the user with no post-hoc change "
                      f"(O.5), so it is not rewritten")
    extra = None
    if prepare is not None:
        extra, why = prepare(spec, a.smoke, hooks)
        if why:
            return refuse(why)
    commit = hooks["spec_commit"](spec.spec_path)
    if commit is None and not a.smoke:
        return refuse(f"the spec {spec.spec_path} has no commit: commit appendix O before a real run")
    key, manifest = hooks["code_keys"](a.npz)
    try:
        c3, readout, z, _ = hooks["load_c3"](spec)
    except ValueError as e:
        return refuse(str(e))
    rid = run_id()
    ctx = Ctx(args=a, spec=spec, smoke=a.smoke, rid=rid, out=out, key=key, c3=c3, readout=dict(readout), z=z,
              types=hooks["model_types"](a.npz), hooks=hooks, spec_commit=commit, extra=extra)
    t0 = time.time()
    try:
        res, code = body(ctx)
    except DataMismatch as e:
        return refuse(f"the Hallem data do not match their pins: {e}")
    res = dict(res, name=name, run_id=rid, smoke=a.smoke, measure_key=key["key"], code=manifest, git=git,
               spec_commit=commit, inputs=dict(m0d_sha256=hooks["m0d_sha"](spec), data_sha256=spec.n.sha_pins(),
                                               state_p_min=spec.n.state_p_min),
               c3=dict(readout=readout, z=z), spec=spec_record(spec), argv=list(argv or []), wall_s=time.time() - t0)
    report = write_report(out, rid, name, res, [c3])
    o_measure.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                  [c3])
    print(res.get("sentence", ""))
    return code
```

`out_allowed` is `n_cli.out_allowed` with O's directory.

- [ ] **Step 4: Write `scripts/run_o1.py`**

```python
#!/usr/bin/env python3
"""Spec O.2 / O.7.1 / O.7.2 (O1: C3's two-state map on the real-odour rig; a characterisation, no learning claim).

Steps:
1. Load the pinned Hallem data; a mismatch is a refusal (exit 2, nothing written).
2. Present every O1 stimulus at every g of o_spec.o1_g_grid (c_δ = o1_c_delta on δ-DL only) on the O1 seeds under the
   four conditions (on, APL->KC block, APL->non-KC block, all-output block). One presentation per job and one cache
   file per (O1, condition, cell, seed) under results/o/run/cache/o1_pres/: rerunning the command resumes.
3. o_rules.o1_judge: the validity gate (INVALID), then STOP_NO_SILENT_STATE or the three judgements, with the records.

    uv run python scripts/run_o1.py                                        # the controller only (R1; resumable)
    uv run python scripts/run_o1.py --smoke --allow-dirty --workers 4      # results/o/smoke/

Exit 0 for JUDGED, 5 for STOP_NO_SILENT_STATE or INVALID (block written; reported to the user), 2 for a refusal."""
from __future__ import annotations

import sys

from flymon.brain import n_cli
from flymon.brain.n_rules import wall_per_step
from flymon.brain.o_cli import (code_keys, git_state, hooks, load_c3, m0d_sha, main_stage,  # noqa: F401
                                make_measurer, model_types, n_oracle, nview, spec_commit)
from flymon.brain.o_rules import JUDGED, o1_judge, sentence


def items_o1(spec, st: dict) -> list:
    """Every declared (condition, g, stimulus, seed), grouped by condition (a worker rebuilds its rig only on a
    switch)."""
    return [dict(cond=cond, edit=edit, g=float(g), stim=s, seed=int(seed), odor=st[g][s]["odor"])
            for cond, edit in spec.o1_conditions for g in spec.o1_g_grid for s in spec.o1_stimuli
            for seed in spec.o1_seeds]


def body(ctx) -> tuple:
    spec, nv = ctx.spec, nview(ctx)
    _, glom = n_cli.glomeruli(nv)
    st = {g: n_cli.stimuli_at(nv, spec.o1_stimuli, g, spec.o1_c_delta, glom) for g in spec.o1_g_grid}
    items = items_o1(spec, st)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.o1_presentations(ctx.c3, items, ctx.readout)
    finally:
        pool.close()
    res = o1_judge(rows, spec)
    res.update(drives={f"{float(g):g}": n_cli.drives(st[g]) for g in spec.o1_g_grid}, n_presentations=len(rows),
               wall_s_per_step=wall_per_step(rows) if rows else None)
    return dict(res, sentence=sentence("o1", res)), (0 if res["outcome"] == JUDGED else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("o1", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `scripts/run_o2.py`**

```python
#!/usr/bin/env python3
"""Spec O.3 / O.7.3 / O.7.4 (O2: presentation-evoked depression at N.8a's point; a characterisation, no learning
claim — its result is only grounds for declaring a later test, O.3).

Steps:
1. Refuse unless block n1 of the committed N summary agrees with O.7.4's declared oracle o (reading 13).
2. At o_spec.o2_point, for each X of o2_pairs and each arm of o2_arms (1 plastic, 2 frozen, 3 punish, 4 da_zero) on the
   O2 seeds: o_jobs.arm_job (N2's absolute-arm procedure with a punish flag). One cache file per (O2, X, arm, seed)
   under results/o/run/cache/o2_arm/: rerunning the command resumes.
3. o_rules.o2_judge: the validity gate (INVALID), then per X the plumbing check, the three judgements, the state-flip
   guard and the records.

    uv run python scripts/run_o2.py                                        # the controller only (R2; resumable)
    uv run python scripts/run_o2.py --smoke --allow-dirty --workers 4

Exit 0 for a judged stage with no X INVALID, 5 otherwise (block written; reported to the user), 2 for a refusal."""
from __future__ import annotations

import sys

from flymon.brain import n_cli
from flymon.brain.o_cli import (code_keys, git_state, hooks, load_c3, m0d_sha, main_stage,  # noqa: F401
                                make_measurer, model_types, n_oracle, nview, spec_commit)
from flymon.brain.o_rules import INVALID, JUDGED, o2_judge, sentence


def items_o2(spec, st: dict) -> list:
    """Every declared (X, arm, seed) on the APL-on rig, grouped by X then arm."""
    return [dict(x=x, y=y, edit=spec.on_edit, arm=a, punish=pu, plastic=pl, da_zero=dz, odor_x=st[x]["odor"],
                 odor_y=st[y]["odor"], seed=int(s))
            for x, y, _ in spec.o2_pairs for a, pu, pl, dz in spec.o2_arms for s in spec.o2_seeds]


def body(ctx) -> tuple:
    spec, nv = ctx.spec, nview(ctx)
    g, c = spec.o2_point
    names = list(dict.fromkeys(s for x, y, _ in spec.o2_pairs for s in (x, y)))
    st = n_cli.stimuli_at(nv, names, g, c)
    items = items_o2(spec, st)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.o2_arms(ctx.c3, items, ctx.readout, spec.n.h3.punish_type)
    finally:
        pool.close()
    res = o2_judge(rows, ctx.z, ctx.extra["declared"], spec)
    res.update(oracle=ctx.extra, drives=n_cli.drives(st), point=dict(g=float(g), c_delta=float(c)), n_arms=len(rows))
    ok = res["outcome"] == JUDGED and all(v["outcome"] != INVALID for v in res["x"].values())
    return dict(res, sentence=sentence("o2", res)), (0 if ok else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("o2", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__, prepare=lambda sp, smoke, hk: hk["n_oracle"](sp, smoke))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_o.py -q -o addopts=""`
Expected: PASS (16 tests).

- [ ] **Step 7: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/o_cli.py scripts/run_o1.py scripts/run_o2.py tests/test_run_o.py
git commit -m "feat(o): o_cli + run_o1 / run_o2 — refusals before any pool, no post-hoc block rewrite, n1 oracle check, blocks with spec commit"
```

Expected suite after Task 7: 1640 + 7 + 12 + 10 + 5 + 17 + 13 + 16 = **1720 passed / 1 skipped / 3 xfailed**.

---

## Self-review (done while writing)

- **Spec coverage.**
  - O.1 conditions and the state definition are in Task 1 (`o1_conditions`, `n.state_p_min` via `n_rules.state`).
  - O.7.1's edit and partition test are in Task 2; 7 680 presentations in Tasks 1 and 7.
  - O.7.2 judgements 1–4 and the records are in Task 5.
  - O.7.3: the single loop with a punish flag, bit-identity, the da-zero arm, the dopamine integral and `weights_frac_by_mbon_set` are in Task 3.
  - O.7.4 judgements 1–5 are in Task 6, and the o values in Tasks 1 and 7.
  - O.7.5: the gate is in Tasks 5 and 6, the per-item files with code key and spec commit in Task 4, the reuse of `state` / `cell_stats` in Task 5 (identity test).
  - O.3's training-seed test is in Task 1. O.5's file set is reading 1, and the summary and raw paths are in Tasks 4 and 7.
  - O.5's no post-hoc rewrite is reading 18 (Task 7). O.6 needs no code. O.7.7's cost is in the Runs section.
- **Placeholders:** none. Every step has its code or command.
- **Names across tasks:**
  - `o_jobs.presentation_job`, `arm_job` are consumed by `OMeasurer`.
  - `OMeasurer.o1_presentations`, `o2_arms` are consumed by the scripts.
  - `o_rules.o1_judge(rows, spec)`, `o2_judge(rows, z, o_by_pair, spec)` and `sentence` are consumed by the scripts.
  - `o_cli.SMOKE_SUMMARY` is used by the tests.
  - Row keys `cond`, `g`, `stim`, `x`, `y`, `arm` are set by `OMeasurer` and read by `o1_key` / `o2_key`.

## Runs (controller)

Every run happens from the repository root, in the background (`run_in_background`), after the full suite passes at the final commit. Subagents never run these. **Report each stage's verdict to the user before starting the next.** No post-hoc change of any rule (O.5).

- **R0 smoke** (seed block 22_009_xxx; writes only `results/o/smoke/`):
  1. `uv run python scripts/run_o1.py --smoke --allow-dirty --workers 4` runs 4 conditions × 2 g × 5 stimuli × 8 seeds = 320 presentations.
  2. `uv run python scripts/run_o2.py --smoke --allow-dirty --workers 4` runs 2 X × 4 arms × 4 seeds = 32 arms.
  3. Check in `results/o/smoke/o_states.json`:
     - `o1.sha` holds four distinct sha256s;
     - the non-KC block's sha differs from `none`'s (the real connectome has APL → non-KC edges);
     - O2's `record.frozen_probes_identical` and `record.da_zero_weights_unmoved` are true for both X;
     - no `INVALID`;
     - the `spec_commit` field is present.
  4. Record wall clock per presentation and per arm. If the arm time is far from O.7.7's ~50 s, re-estimate R2 before starting it.
  5. Rerun both smoke commands to confirm the resume: every item is a cache hit and the pool starts no job (one round of pool start-up only).
- **R1 O1:**
  1. `uv run python scripts/run_o1.py` runs 7 680 presentations (~25 min at 16 workers).
  2. Rerun the same command after an interruption; it resumes from `results/o/run/cache/o1_pres/`.
  3. Exit 0 (`JUDGED`) or 5 (`STOP_NO_SILENT_STATE` / `INVALID`).
  4. Commit `results/summary/o_states.json` (block `o1`; `results/summary/` is tracked, and the raw files under `results/o/` stay ignored): `git commit -m "results(o): O1 ..."`.
  5. Write the O1 result paragraph into the spec after O.7, citing the run id. It includes the labels, the mixed-cell list, q_on / q_kc / q_nonkc / q_all, and the per-stimulus slopes. Commit it.
  6. Report to the user, and include the note that `BISTABLE_NETWORK` / `DRIVE_*` are properties of C3, not of the fly (O.6).
- **R2 O2** (independent of R1's outcome, O.5; run after R1 so the two pools never overlap and the summary file has one writer):
  1. `uv run python scripts/run_o2.py` runs 256 arms (~15 min).
  2. It refuses unless `results/summary/n_real_odour.json` is committed with block `n1` agreeing with O.7.4's o.
  3. Commit block `o2`.
  4. Write the O2 result paragraph (labels per X with m, b, m₄, D, e and CIs, the flip guard, the plumbing records, the per-compartment dopamine integrals). Commit it, and report to the user.
  5. State reading 10 next to the result: `DA_DEPENDENT` follows by construction whenever `DEPRESSION_PRESENT`.
  6. O2's result is only grounds for declaring a later "other pair only" learning test (O.3). No such test is declared here.
- **Ledger:** after R2, add the README ledger lines (ko/en) for appendix O, in the existing format.
