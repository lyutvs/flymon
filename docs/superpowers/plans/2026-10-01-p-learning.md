# Spec P — Punishment Learning on the Real-Odour Dissimilar Pair (First Learning Judgement): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix P (as overridden by P.6) needs, and nothing else:
- the judgement run: 2 directions (r1 X = 4:1 / Y = δ-DL, r2 X = δ-DL / Y = 4:1) × 3 arms (① no-punish-plastic, ② no-punish-frozen, ③ punish-plastic) × 32 seeds 23_000_000 + i = 192 arms, judged per direction by P.6.2's (i)–(iii) and labelled `INVALID` → `LEARNS_CONFIRMATORY` → `DIRECTION_DEPENDENT` → `NO_LEARNING` → `INCONCLUSIVE`, with the records of P.3 / P.6.2 / P.6.3;
- the operating-characteristic record (P.6.4) from O2's committed run, written before the judgement run;
- the seed-collision test of P.6.6 over every `*_spec.py` in `flymon/`;
- both stages behind a validity gate, per-item checkpoint files, refusals and `--smoke`. The controller runs the real stages afterwards (section "Runs (controller)").

This is a **confirmatory** test declared after O2 (P.0): a pass is a replication, not a discovery. Whatever the label, the conclusion stops at "this connectome model (C3), this pair, this procedure" (P.1) and is not a verdict on the Pokémon claim (P.5).

**Architecture:**
- New files only. O's and N's modules are imported, never edited (P.4, P.6.7).
  - `p_spec` holds `PSpec`: its field `o` is `o_spec.SPEC` derived with `dataclasses.replace` (only the pairs, the arms and the seed blocks differ); P-only numbers are `PSpec` fields; N's and O's numbers are read through `spec.o` / `spec.o.n`.
  - `p_rules` holds the pure rules: the gate, the per-direction rule and labels, the records, the operating characteristic, the sentences.
  - `p_measure` is P's write guard (`results/p/`, `results/summary/p_learning.json`), its content-addressed cache (`PCache`) and `PMeasurer` (a subclass of `o_measure.OMeasurer` that runs `o_jobs.arm_job` per (direction, arm, seed)).
  - `p_cli` holds the refusals, the c1 source (block n1), the O2 source (block o2 + O2's cache) and the stage skeleton.
- Two scripts: `scripts/p_oc.py` (block `oc`, no pool) and `scripts/run_p.py` (block `p`, the pool). `run_p` refuses a declared run until block `oc` is committed.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`; `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **P** (P.0–P.5) and its red-team amendment **P.6** (P.6.1–P.6.7), which wins wherever they differ:

| P.6 section | Replaces / adds |
|---|---|
| P.6.1 | the claim narrowed: punishment lowers X's readout (relative dV and X's own V) more than the same presentation without punishment, concentrated on X; "associative" is a consequence of the rule, no unpaired arm |
| P.6.2 | P.3's judgement replaced: per direction (i) CI lower of ℓ_r > 0 and ℓ_r ≥ c₁, (ii) CI upper of s_r < 0, (iii) \|t_r\| ≤ 0.5·\|s_r\|; labels `INVALID` → `LEARNS_CONFIRMATORY` → `DIRECTION_DEPENDENT` → `NO_LEARNING` → `INCONCLUSIVE`; averaged ℓ a record |
| P.6.3 | flip re-analysis without suffix (record); state analysis stratified by pre-state (record) |
| P.6.4 | operating characteristic from O2's per-seed values, recorded before the run; r1 approximated by O2's 4:1 row; n = 32 fixed |
| P.6.5 | the full O.7.5 gate + expected pair / point / arm flags; c₁ from block n1 within 5·10⁻⁴; one rerun after a technical `INVALID` |
| P.6.6 | collision test over every `*_spec.py` seed and its training seeds; no s = 22 000 |
| P.6.7 | `p_spec` from `OSpec` by `dataclasses.replace`; mutation tests of the judgement and the gate |

## Global Constraints

- **Commits carry no trailers of any kind.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No existing file changes (P.4).** Every existing module, script, test and `.gitignore` stays byte-identical — in particular `o_spec.py`, `o_jobs.py`, `o_measure.py`, `o_rules.py`, `o_cli.py`, `run_o1.py`, `run_o2.py`, every `n_*.py`, `odor_real.py`, `plasticity.py`, `fly_pool.py`. P only adds the files in "File Structure". No O module global is monkeypatched at runtime either (reading 1).
- **Another session owns the encoder redesign.** Never read-modify-write `flymon/agent/encode.py` or `docs/superpowers/specs/2026-10-01-encoder-redesign*.md` (P preamble).
- **One configuration object.** Every P number is a field of `flymon.brain.p_spec.SPEC` (`PSpec`). Its `o` is `o_spec.SPEC` with `dataclasses.replace` of `o2_pairs`, `o2_arms`, `o2_seed0`, `smoke_seed0` only (P.6.7, test). Nothing `o_spec` or `n_spec` holds is restated: N's numbers via `spec.o.n`, O's via `spec.o`. A test parses `p_spec.py` and allows only the literals {0, 23_000_000, 23_009_000, 0.5, 1e-9}; another allows `p_rules.py` only {0, 1, 2, 3, 12} (indices, "at least 2 seeds", excerpt lengths).
- **Numbers (verbatim from P / P.6, with where they live):**
  - **Directions (P.2):** r1 X = 4:1 (punished), Y = δ-DL; r2 X = δ-DL (punished), Y = 4:1 — derived from `n_spec.pairs`' `dis` pair, `PSpec.pairs()`.
  - **Arms (P.2):** ③ `punish` (punish on, plastic on), ① `plastic` (no punish injection, plastic on), ② `frozen` (no punish injection, plastic off) — O's `o2_arms` rows filtered by name, flags unchanged; `da_zero` is not run. Procedure, time constants, punishment type, training-seed rule = O2's `o_jobs.arm_job` (≡ N2's absolute conditioning, bit-identical with punish on).
  - **Operating point:** O's `o2_point` (g 0.25, c_δ 8, N.8a), edit `none` (`spec.o.on_edit`).
  - **Seeds (P.2):** **23_000_000 + i, i < 32** (`o2_seed0` replaced, `o2_n_seeds` O's 32); smoke **23_009_xxx** (`smoke_seed0` 23_009_000; O's smoke offset 100 and size 4 → 23_009_100–23_009_103). Training seeds `n_spec.train_seed(s, t, 1_000_000, 1000)`.
  - **Readout (P.2):** V = z_A − z_P with C3's frozen z; dV = V_X − V_Y; Δ = post − pre (`n_rules.delta`).
  - **Judgement (P.6.2)**, per direction r, seed bootstrap 95% (`spec.o.boot_draws` = 10 000, percentile, `spec.o.boot_seed` = `n_spec.boot_seed`, both directions resampled with the same seed draws):
    - ℓ_r = −mean_seed(Δ(dV)[③] − Δ(dV)[①]); s_r = mean_seed(ΔV_X[③] − ΔV_X[①]); t_r = mean_seed(ΔV_Y[③] − ΔV_Y[①]) (so ℓ_r = t_r − s_r).
    - (i) CI lower of ℓ_r > 0 **and** ℓ_r ≥ c₁; (ii) CI upper of s_r < 0; (iii) |t_r| ≤ **0.5**·|s_r| (point estimates; `concentration_frac`).
    - (i)–(iii) all → **학습 확인** (`LEARNING_CONFIRMED`); else CI upper of ℓ_r < c₁ → **학습 배제** (`LEARNING_EXCLUDED`); else **미정** (`UNDECIDED`).
    - Labels in this order: `INVALID` → both confirmed `LEARNS_CONFIRMATORY` → one confirmed and the other excluded `DIRECTION_DEPENDENT` → both excluded `NO_LEARNING` → else `INCONCLUSIVE`.
    - The averaged ℓ = (ℓ_r1 + ℓ_r2)/2 and its CI: a record.
  - **c₁ (P.3, P.6.5):** c₁ = `n_spec.c1_frac` (0.25) · |o_dis|, o_dis read from block `n1` of `results/summary/n_real_odour.json`; it must agree with the declared −2.433 (`o_spec.oracle_o`'s `dis`) within **5·10⁻⁴** (`oracle_o_tol`), else refusal. Declared c₁ = 0.608.
  - **Validity (P.3, P.6.5)** before any judgement: O.7.5's gate (`o_rules.validity`: complete Cartesian product, unique keys, finite values, declared seeds, one CSC sha256, the declared edit), plus every row's direction pair, operating point and arm flags, plus a positive finite c₁; then ②'s post-training weight sha256 == w0's for every seed (a direction failing → that direction `INVALID` → the whole `INVALID`).
  - **Flip (P.6.3):** O.7.4-4's trigger (an arm with more than **1/8** of its seeds flipped, `flip_max_frac`); the re-analysis without the flipped seeds is a record **without a suffix**; the state analysis is a record stratified by the pre-state.
  - **OC (P.6.4):** `n_spec.oc_draws` (2 000) experiments, `n_spec.oc_seed`, n = 32; smoke `n_spec.smoke`'s `oc_draws` (20).
  - **Rerun (P.6.5):** one rerun with fixed code after a technical `INVALID`; a run that produced a label is final.
- **Every output path is guarded.** Raw rows and caches under `results/p/` (git-ignored by the existing `.gitignore` rule `results/*`, which re-includes only `results/summary/` — checked: no `.gitignore` edit is needed); the summary is `results/summary/p_learning.json` (blocks `oc`, `p`, and `p_invalid` only after a P.6.5 rerun); smoke `results/p/smoke/p_learning.json`. Everything is written through `p_measure`'s guard; anything else is refused (SystemExit 2).
- **Worker jobs and pools run only from script files with `if __name__ == "__main__":` or from pytest**, never from a heredoc (spawn re-imports the parent — a fork bomb). Both P scripts carry the guard.
- **Subagents never write under `results/` and never start a real or smoke stage** — including `p_oc.py`, which reads O2's real cache. Tests write only under `tmp_path`. The controller runs every real step (section "Runs (controller)").
- **Tests locate the repository via `Path(__file__)`** (`Path(__file__).resolve().parents[1]` in `tests/`, `parents[2]` in `tests/brain/`), never via the cwd.
- **Full suite before each commit:** `uv run pytest -q -rfE -o addopts=""`.
  - Baseline at `0cc9e5a`: **1742 passed / 1 skipped / 3 xfailed**.
  - Known flake: `tests/test_run_m0d_h3.py::test_a_complete_run_writes_everything_through_both_guards_and_resumes_from_the_cache`. Rerun that file once.
  - Never run two suites at once.
- **Style:** match `o_*`. Module docstring citing the spec section, dense one-line comments, `from __future__ import annotations`.

## Readings of the spec (decided here — the controller's rulings)

1. **File set: `p_measure` and `p_cli` are unavoidable.** P.4 lists `p_spec`, `p_rules`, `run_p` (and `p_cli` if needed). O's guard cannot write `results/p/`: `o_measure.guard` reads its module constants `ALLOWED_DIR` / `SUMMARY`, `OCache.get_or_compute` writes through `o_measure.write_json`, and `o_cli.main_stage` is wired to `o_cli.RUN_OUT` / `SMOKE_OUT`, `o_measure.SUMMARY` and `o_measure.write_summary_block`. Editing O is forbidden and monkeypatching O's globals at runtime would move O's own guard in-process, so P copies the guard / cache / stage skeleton into `p_measure` and `p_cli` (as O copied N's). `p_oc.py` is the P.6.4 script.
2. **`PSpec` wraps a derived `OSpec`.** `dataclasses.replace` can only set `OSpec` fields, so `PSpec.o = replace(o_spec.SPEC, o2_pairs, o2_arms, o2_seed0, smoke_seed0)` (a test asserts exactly these four differ) and the P-only numbers — `concentration_frac` 0.5, `o2_check_tol` 1e-9, the OC draw counts read from N — are `PSpec` fields. O1 fields inside `o` are inert.
3. **Directions** come from `n_spec.pairs`' `dis` pair: r1 = (4:1, dDL), r2 = (dDL, 4:1); both directions use the `dis` o for c₁ (P.6.2 states the shared c₁ as an assumption).
4. **Arms** are O's `o2_arms` rows named `plastic` (①), `frozen` (②), `punish` (③), flags unchanged; arm ④ (`da_zero`) is not run. `arm_job`'s `reward_type` stays N's `h3.reward_type` (O's measurer passes it).
5. **c₁** is computed from block n1's **exact** o (c₁ = 0.25·|−2.43255…| = 0.6081…), not from the rounded 0.608; the 5·10⁻⁴ check runs through `o_cli.n_oracle` on an `OSpec` holding only the `dis` pair (so the `sim` value is not P's gate). `--smoke` without a readable block uses the declared −2.433.
6. **Per-seed vectors.** dl = −(Δ(dV)[③] − Δ(dV)[①]), ds = ΔV_X[③] − ΔV_X[①], dt = ΔV_Y[③] − ΔV_Y[①] (dl = dt − ds, test). V of one probe is computed as `h4_formula.dv`'s V (test pins V_X − V_Y = dv).
7. **Outcome order per direction:** "confirmed" is tested before "excluded". (Both can only coincide when a percentile CI upper lies below the point estimate; then confirmation wins, as P.6.2 lists it first.)
8. **`DIRECTION_DEPENDENT`** = exactly {confirmed, excluded} across the two directions. Any undecided direction → `INCONCLUSIVE` unless both are excluded.
9. **Joint bootstrap.** One `boot_weights(32, 10 000, boot_seed)` matrix W is applied to both directions' vectors (seeds in the same order; a mismatch is a `ValueError`, never judged). Strata and state-conditional records use their own W on their subset (they are per direction).
10. **INVALID scope.** The gate, a non-positive or non-finite c₁, or any direction's ② plumbing failure makes the whole judgement `INVALID` (P.3: "그 방향 INVALID → 전체 INVALID"). Exit 0 for any judged label (including `NO_LEARNING` / `INCONCLUSIVE`), 5 for `INVALID`, 2 for a refusal.
11. **P.6.5 rerun.** Only `run_p`, only when the real block `p` has outcome `INVALID`, only with `--rerun-after-invalid`, only when the code manifest differs from the `INVALID` block's (`code.key`), and only once: the first block moves to `p_invalid`, the new block is final whatever it says. A judged block refuses any rerun. The new code key gives fresh cache entries (the key includes it).
12. **Flip re-analysis (P.6.3 overrides P.3's suffix).** Flips are counted per (direction, arm) with `n_rules.state` on X's pre vs post P. If any count > `flip_max_frac`·32 (= 4), the union of flipped seeds over both directions and all arms is dropped and `judge_vectors` runs on the rest (< 2 left → `None`). Recorded as `flip.reanalysis` with `note`; the judged label never gets `_STATE_FLIP_SENSITIVE`.
13. **Pre-state strata (P.6.3).** Per direction, by X's pre-probe state in arm ③ (arm ①'s pre probe is the same probe; `pre_identical` records the A / P / kc_spikes equality); `direction_stats` per stratum with its own bootstrap, `None` below 2 seeds. A record.
14. **State-conditional (P.3 record).** Per direction, the seeds whose every probe (pre / post × X / Y × all three arms) fires.
15. **Other P.3 records.** Per direction: ΔA_X, ΔP_X, ΔA_Y, ΔP_Y per arm and ③ − ① (raw units, CI); ①'s non-associative change ① − ② of Δ(dV) and ΔV_X; per-arm mean phasic-dopamine integral per compartment; per-arm `weights_frac` (all, punish core, reward core); ②'s probes unchanged.
16. **OC source (P.6.4).** The committed `results/summary/o_states.json` block `o2` holds only aggregates; the per-seed rows live in O2's git-ignored cache `results/o/run/cache/o2_arm/` (256 entries of run `20261001T072746Z-8f685c`, present in this worktree). `p_cli.o2_source` reads the entries whose `code_key` equals the block's `measure_key`, requires O2's declared (X, arm, seed) grid exactly (`o_rules.validity`), and recomputes D = mean(Δ(dV)[punish] − Δ(dV)[plastic]) per X to within 1e-9 of the block's D (checked here: −2.0750… and −2.3977… reproduce). C3's z must equal block o2's z.
17. **OC mapping and nulls.** r1 ← O2 X = 4:1 (its Y was 1:4 — the stated limitation, `y_matches` False), r2 ← O2 X = δ-DL (Y = 4:1, same pair). One experiment = 32 seed indices drawn with replacement (the same indices for both directions), judged by `judge_vectors` with the real 10 000-draw W — P.6.2's whole rule. Scenarios: `observed`; `null_both` (each direction's ds shifted by mean(dl) − c₁ on every seed, Y untouched, so mean ℓ = c₁ exactly); `null_r1`, `null_r2` (one direction at c₁, the other observed). The false-pass probability is reported per null scenario and as their maximum. **Expect it near 0.5 for a single direction at c₁**, because (i) compares the point estimate ℓ_r with c₁ (a prototype on the O2 rows gave 0.44–0.54 at 50 draws). This is a record; n stays 32 (P.6.4).
18. **OC before R1.** Outside `--smoke`, `run_p` refuses unless the real summary holds block `oc` and is committed (`l_cli.check_committed`).
19. **P.6.6's list.** No `*_spec.py` holds a 12 000 000+ block today (grep); the test enumerates what exists programmatically: every int under a field named `*seed*` (or `probe0`) through nested specs, each module's `smoke(SPEC)`, `OdorSet` probe ranges, N's judge seeds at the largest n, O's O1/O2 blocks, B's and the re-scope's probe / train / qualification / taurec seeds via their methods, and `d6a.SEEDS`. A new spec file that is not enumerated fails the test. A training-seed image is checked for every t < stride (1000), not only t < 12.
20. **Point on rows.** `PMeasurer` adds `direction`, `x`, `y` and `point` to each row (the point is also in the cache key); the gate compares the point with `o2_point`.
21. **Smoke.** `p_spec.smoke` = O's smoke on `o` (4 seeds from 23_009_100, 200 bootstrap draws) and N's smoke `oc_draws` (20). A smoke run reads / writes only `results/p/smoke/`; `p_oc --smoke` still reads the committed O2 source (read-only).
22. **No post-hoc rewrite.** Outside `--smoke`, a stage refuses (exit 2) when the real summary already holds its block (P.6.5's rerun is the only exception). The block is written once the stage ends, so an interrupted `run_p` resumes from the cache files.

## Review Focus

1. **Directions or arms paired on the wrong seeds.** If rows arrive in another order (a resumed run, a shuffled pool), a pairing by position would silently compare different seeds. Expect the result to be identical for shuffled rows and a `ValueError` (never a label) when the directions hold different seed lists. Test in Task 2.
2. **Degenerate statistics at the rule's edges.** Zero-variance vectors (CI [x, x]), s_r = 0, an effect exactly at c₁, c₁ NaN / 0 / missing, fewer than 2 seeds in a stratum or after a flip drop. Expect a label or a `None` record, never a crash or a false pass. Tests in Tasks 2 and 3.
3. **The OC fed from the wrong O2 data.** A cache holding another run's entries, a missing entry, a recomputed D off the block's, C3's z differing from O2's. Expect a refusal naming the defect, never a recorded OC. Tests in Tasks 6 and 7.
4. **The rerun rule abused.** A rerun of a judged block, a second rerun, a rerun with unchanged code, the flag without an `INVALID` block. Expect exit 2 with the reason and the summary untouched. Test in Task 7.
5. **P writing where O or N live.** P's guard must refuse `results/o/`, `o_states.json`, `n_real_odour.json`; a smoke run must never read or write the real cache / summary and a declared run never the smoke ones. Tests in Tasks 5 and 7.

---

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/p_spec.py` | `ARMS`, `DIRECTIONS`, `C1_PAIR`, `P_O` (the derived `OSpec`), `PSpec` (+ `seeds`, `c1_frac`, `train_seeds()`, `pairs()`, `flags()`, `declared_o()`, `n_arms()`), `SPEC`, `smoke()` |
| `flymon/brain/p_rules.py` | labels and outcomes; gate (`p_key`, `p_declared`, `expectations`, `group`, `plumbing`); judgement (`v_of`, `arm_vectors`, `direction_stats`, `label`, `judge_vectors`); records (`raw_records`, `flips`, `flip_reanalysis`, `strata`, `state_conditional`, `mechanism_records`); `p_judge`; `sentence`; OC (`o2_vectors`, `null_shift`, `oc_run`, `oc_sentence`) |
| `flymon/brain/p_measure.py` | `ALLOWED_DIR`, `SUMMARY`, `MEASURE_FILES`, `HASHED_FILES`, `guard`, `write_bytes`, `write_json`, `write_summary_block`, `PCache`, `PMeasurer.p_arms` |
| `flymon/brain/p_cli.py` | hooks (`check_committed`, `code_keys`, `git_state`, `load_c3`, `m0d_sha`, `make_measurer`, `model_types`, `spec_commit`, `c1_source`, `o2_source`), `out_allowed`, `nview`, `spec_record`, `write_report`, `Ctx`, `parser`, `main_stage` |
| `scripts/p_oc.py`, `scripts/run_p.py` | the two stages |
| `tests/brain/test_p_spec.py`, `tests/brain/test_p_rules.py`, `tests/brain/test_p_measure.py`, `tests/brain/test_p_cli.py`, `tests/test_run_p.py` | tests |

Task order: 1 spec → 2–4 rules → 5 measurer → 6 CLI layer → 7 scripts. Later tasks consume only names listed in earlier tasks' **Produces**.

O / N names P calls (all verified at `0cc9e5a`):
- `o_spec.SPEC` / `OSpec` / `smoke(spec)`; fields `o2_pairs`, `o2_arms`, `o2_seed0`, `o2_n_seeds`, `o2_point`, `on_edit`, `oracle_o`, `oracle_o_tol`, `flip_max_frac`, `ci_level`, `boot_draws`, `boot_seed`, `smoke_seed0`, `smoke_boot_draws`, `workers`, `pool_timeout_s`, `spec_path`, `n`; `o2_seeds`, `o2_train_seeds()`, `n_o2_arms()`;
- `n_spec.smoke(spec)`; `NSpec.pairs`, `c1_frac`, `oc_draws`, `oc_seed`, `train_seed_base`, `train_seed_stride`, `h4.teach_trials`, `h3.punish_type`, `h3.reward_type`, `state_p_min`, `sha_pins()`;
- `o_jobs.arm_job(eng, pl, pops, comps, ro, params, edit, odor_x, odor_y, seed, arm, punish, plastic, da_zero, readout, punish_type, reward_type, strength, settle_ms, read_ms, window_ms, trials, present_ms, gap_ms, train_settle_ms, seed_base, seed_stride) -> dict` (keys `seed`, `edit`, `arm`, `punish`, `plastic`, `da_zero`, `csc_sha256`, `pre`, `post`, `weights_frac`, `weights_frac_A`, `weights_frac_P`, `w0_sha256`, `w_post_sha256`, `da_integral`, `wall_s`), `o_jobs._RIG`;
- `o_measure.MEASURE_FILES`, `HASHED_FILES`, `SUMMARY`, `OMeasurer(pool, ospec, cache)` (`_items(kind, fn, params, jobs, keys)`, `_window()`, `.spec` = N's spec);
- `o_rules.validity(rows, declared, key_of, seeds, edit_of=None) -> list`, `boot_weights(n, draws, seed)`, `finite(o)`, `o2_declared(spec)`, `o2_key(r)`, `o2_edit_of(spec)`, `JUDGED`, `INCONCLUSIVE`;
- `o_cli.load_c3(ospec)`, `m0d_sha(ospec)`, `model_types(npz)`, `spec_commit(path)`, `n_oracle(ospec, smoke, committed=, path=) -> ({"declared", "exact"}, None) | (None, why)`, `spec_record(ospec)`, `RUN_OUT`;
- `n_rules.INVALID`, `NO_LEARNING`, `_ci(v, level)`, `delta(row, z)`, `state(P, nspec)`; `h4_formula.dv(probe, z)` (test only);
- `n_cli.NPZ`, `stimuli_at(ctx, names, g, c_delta)`, `drives(st)`; `l_cli.check_committed(summary, blocks)`, `refuse(why)`, `run_id()`;
- `h3_store.ROOT`, `MeasureCache`, `canonical`, `canonical_pretty`, `code_key`, `git_state`, `sha256_file`; `pool_bench.refuse_old_engine_output`, `refuse_modified_engine_output`; `d6a.SEEDS` (test only).

---

### Task 1: `p_spec` — the derived OSpec, the directions and arms, and the P.6.6 collision test

**Files:**
- Create: `flymon/brain/p_spec.py`
- Test: `tests/brain/test_p_spec.py`

**Interfaces:**
- Consumes: `o_spec.SPEC`, `OSpec`, `o_spec.smoke`; `n_spec.smoke`.
- Produces:
  - `ARMS = ("plastic", "frozen", "punish")`, `DIRECTIONS = ("r1", "r2")`, `C1_PAIR = "dis"`, `P_O` (an `OSpec`).
  - `PSpec` (frozen dataclass): fields `o`, `directions`, `arms`, `c1_pair`, `concentration_frac`, `oc_draws`, `oc_seed`, `smoke_oc_draws`, `o2_check_tol`; properties `seeds -> tuple`, `c1_frac -> float`; methods `train_seeds() -> tuple`, `pairs() -> {direction: (x, y)}`, `flags() -> {arm: (punish, plastic, da_zero)}`, `declared_o() -> float`, `n_arms() -> int`.
  - `SPEC = PSpec()`, `smoke(spec) -> PSpec`.

- [ ] **Step 1: Write the failing tests**

```python
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
           "flymon/brain/o_spec.py": "flymon.brain.o_spec", "flymon/rescope/spec.py": "flymon.rescope.spec"}


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


def test_p_blocks_are_the_declared_ones():
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
    assert SPEC.declared_o() == -2.433 and round(SPEC.c1_frac * abs(SPEC.declared_o()), 3) == 0.608   # P.3
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_p_spec.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.p_spec`).

- [ ] **Step 3: Write `p_spec.py`**

```python
"""The configuration of spec appendix P as amended by P.6 (P.6 wins over P.1-P.4): the first learning judgement of the
project — punishment learning on N's real-odour dissimilar pair (IA:EB 4:1 vs δ-DL), both directions, on new seeds. A
confirmatory test declared after O2 (P.0).
The O-shaped part `o` is o_spec.SPEC derived with dataclasses.replace (P.6.7): only the pairs (the two directions), the
arms (①②③ of O's four, with O's flags) and the seed blocks differ. N's and O's numbers are read through `o` / `o.n`,
never restated; the P-only numbers are this dataclass's own fields."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .n_spec import smoke as n_smoke
from .o_spec import SPEC as O_SPEC, OSpec, smoke as o_smoke

ARMS = ("plastic", "frozen", "punish")            # P.2: ① no punish, plastic; ② no punish, frozen; ③ punish, plastic
DIRECTIONS = ("r1", "r2")                         # P.2: r1 X = 4:1, Y = δ-DL; r2 X = δ-DL, Y = 4:1
C1_PAIR = "dis"                                   # P.3: c1 from N1's dissimilar-pair oracle (N.8.6's c1 rule)
_X, _Y = dict((p, (x, y)) for p, x, y in O_SPEC.n.pairs)[C1_PAIR]

P_O = dataclasses.replace(
    O_SPEC,
    o2_pairs=((_X, _Y, C1_PAIR), (_Y, _X, C1_PAIR)),                 # (X, Y, the N1 pair whose o sets c1), r1 then r2
    o2_arms=tuple(a for a in O_SPEC.o2_arms if a[0] in ARMS),       # O's (name, punish, plastic, da_zero), arm 4 dropped
    o2_seed0=23_000_000,                                             # P.2: 23_000_000 + i, i < o2_n_seeds (32, O's)
    smoke_seed0=23_009_000)                                          # P.2: smoke 23_009_xxx (O's offset 100 -> 23_009_100+)


@dataclass(frozen=True)
class PSpec:
    o: OSpec = P_O
    directions: tuple = DIRECTIONS
    arms: tuple = ARMS
    c1_pair: str = C1_PAIR
    concentration_frac: float = 0.5               # P.6.2 (iii): |mean dt| <= 0.5 |s_r|
    oc_draws: int = O_SPEC.n.oc_draws             # P.6.4: simulated experiments (N's OC count)
    oc_seed: int = O_SPEC.n.oc_seed
    smoke_oc_draws: int = n_smoke(O_SPEC.n).oc_draws
    o2_check_tol: float = 1e-9                    # P.6.4: recomputed O2 D vs the committed block's D

    @property
    def seeds(self) -> tuple:
        return self.o.o2_seeds

    @property
    def c1_frac(self) -> float:
        return self.o.n.c1_frac

    def train_seeds(self) -> tuple:
        return self.o.o2_train_seeds()

    def pairs(self) -> dict:
        return {d: (x, y) for d, (x, y, _) in zip(self.directions, self.o.o2_pairs)}

    def flags(self) -> dict:
        return {a: (pu, pl, dz) for a, pu, pl, dz in self.o.o2_arms}

    def declared_o(self) -> float:
        return float(dict(self.o.oracle_o)[self.c1_pair])

    def n_arms(self) -> int:
        return len(self.directions) * len(self.arms) * self.o.o2_n_seeds


SPEC = PSpec()


def smoke(spec: PSpec) -> PSpec:
    """Scale only: O's smoke (4 seeds from 23_009_100, 200 bootstrap draws) and N's smoke OC count."""
    return dataclasses.replace(spec, o=o_smoke(spec.o), oc_draws=spec.smoke_oc_draws)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_p_spec.py -q -o addopts=""`
Expected: PASS (8 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/p_spec.py tests/brain/test_p_spec.py
git commit -m "feat(p): p_spec — OSpec derived by replace (pairs, arms, seeds 23_000_000/23_009_xxx), directions r1/r2, P.6.6 seed-collision test over every *_spec.py"
```

Expected suite: **1750 passed / 1 skipped / 3 xfailed**.

---

### Task 2: `p_rules` (1) — the gate, the per-direction rule, the labels, mutation tests

**Files:**
- Create: `flymon/brain/p_rules.py`
- Test: `tests/brain/test_p_rules.py`

**Interfaces:**
- Consumes: `p_spec.SPEC` (`directions`, `arms`, `seeds`, `pairs()`, `flags()`, `concentration_frac`, `o.ci_level`, `o.boot_draws`, `o.boot_seed`, `o.o2_point`, `o.on_edit`); `o_rules.validity`, `boot_weights`, `finite`, `JUDGED`, `INCONCLUSIVE`; `n_rules.INVALID`, `NO_LEARNING`, `_ci`, `delta`, `state`.
- Produces (row keys: `direction`, `x`, `y`, `arm`, `seed`, `punish`, `plastic`, `da_zero`, `edit`, `csc_sha256`, `point`, `pre`, `post`, `w0_sha256`, `w_post_sha256`, and `arm_job`'s other keys):
  - constants `LEARNS_CONFIRMATORY`, `DIRECTION_DEPENDENT`, `LABELS`, `CONFIRMED`, `EXCLUDED`, `UNDECIDED`, `OC_RECORDED`, `FLIP_NOTE` (re-exports `INVALID`, `NO_LEARNING`, `INCONCLUSIVE`, `JUDGED`);
  - `p_key(r) -> tuple`, `p_declared(spec) -> set`, `expectations(rows, spec) -> list`, `group(rows, spec) -> {d: {arm: rows}}`, `plumbing(by, spec) -> {d: [reasons]}`;
  - `v_of(probe, z) -> float`, `arm_vectors(by_one_direction, z, spec) -> {"seeds", "dl", "ds", "dt"}`, `direction_stats(dl, ds, dt, W, c1, spec) -> dict` (keys `n`, `ell`, `ell_ci`, `s`, `s_ci`, `t`, `t_ci`, `i`, `ii`, `iii`, `outcome`), `label(outcomes) -> str`, `judge_vectors(vec, c1, spec, W=None) -> {"label", "directions", "mean_ell"}`;
  - `p_judge(rows, z, c1, spec) -> dict` (keys `outcome`, `label`, `reasons`, and when judged `c1`, `pairs`, `directions`, `mean_ell`, `seeds`; when `INVALID` by plumbing `invalid_directions`). Task 3 adds `flip` and `records`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_p_rules.py
"""Spec P.3 / P.6.2 / P.6.5: the gate (O.7.5's validity + the declared pair, point and arm flags + a positive c1), arm 2's
sha256 plumbing, the per-direction rule (i)-(iii) and the labels in P.6.2's order, one seed bootstrap for both
directions; mutation tests (one direction zeroed, one reversed, arm 2's weights moved)."""
import json
import random
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain.h4_formula import dv
from flymon.brain.n_rules import INVALID, NO_LEARNING, _ci
from flymon.brain.o_rules import INCONCLUSIVE, JUDGED, boot_weights
from flymon.brain.p_rules import (CONFIRMED, DIRECTION_DEPENDENT, EXCLUDED, LEARNS_CONFIRMATORY, UNDECIDED,
                                  arm_vectors, group, judge_vectors, label, p_judge, v_of)
from flymon.brain.p_spec import SPEC

T = replace(SPEC, o=replace(SPEC.o, o2_n_seeds=16, boot_draws=2000))
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}
C1 = 0.608


def _probe(seed, A=20.0, P=10):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1,
                wall_s=0.01, steps=1400)


def make_rows(effect=None, y_effect=None, spec=T, nonassoc=2.0, jitter=0.2, flip=(), silent_pre=()):
    """With ZU (V = A - P): the plastic arm lowers A_X by `nonassoc`; the punish arm by nonassoc + effect[d] + j_i;
    y_effect[d] lowers the punish arm's A_Y by y_effect[d] + j_i. So ds = -(effect + j), dt = -(y_effect + j) and
    ℓ_d = effect - y_effect. `flip`: seed indices whose plastic-arm X post P is 0; `silent_pre`: seed indices whose
    probes are all silent (P 0)."""
    effect = effect or {"r1": 2.4, "r2": 2.4}
    y_effect = y_effect or {}
    out = []
    for d, (x, y) in spec.pairs().items():
        for a in spec.arms:
            pu, pl, dz = spec.flags()[a]
            for i, s in enumerate(spec.seeds):
                j = jitter * ((i % 4) - 1.5)
                ax = {"plastic": nonassoc, "frozen": 0.0, "punish": nonassoc + effect[d] + j}[a]
                ay = y_effect[d] + j if (a == "punish" and d in y_effect) else 0.0
                p0 = 0 if i in silent_pre else 10
                px = 0 if (a == "plastic" and i in flip) else p0
                out.append(dict(direction=d, x=x, y=y, seed=s, arm=a, punish=pu, plastic=pl, da_zero=dz, edit="none",
                                csc_sha256="sha-none", point=list(spec.o.o2_point),
                                pre={"x": _probe(s, P=p0), "y": _probe(s, P=p0)},
                                post={"x": _probe(s, A=20.0 - ax, P=px), "y": _probe(s, A=20.0 - ay, P=p0)},
                                w0_sha256="w0", w_post_sha256="w0" if a == "frozen" else "w1",
                                weights_frac=1.0 if a == "frozen" else 0.9, weights_frac_A=0.8, weights_frac_P=1.0,
                                da_integral={"PPL105": 1.0 if pu else 0.0}, wall_s=1.0))
    return out


def test_v_of_is_h4_formulas_v():
    z = {"A": [10.0, 4.0], "P": [20.0, 5.0]}
    x, y = _probe(1, A=30.0, P=12), _probe(1, A=7.0, P=40)
    assert v_of(x, z) - v_of(y, z) == pytest.approx(dv({"A": [[30.0, 7.0]], "P": [[12, 40]]}, z)[0])


def test_vectors_are_the_declared_differences():
    by = group(make_rows(), T)
    v = arm_vectors(by["r1"], ZU, T)
    j = 0.2 * (np.arange(16) % 4 - 1.5)
    assert v["seeds"] == list(T.seeds)
    assert np.allclose(v["ds"], -(2.4 + j)) and np.allclose(v["dt"], 0.0)
    assert np.allclose(v["dl"], v["dt"] - v["ds"])                          # ℓ = t - s per seed


def test_both_directions_learn():
    res = p_judge(make_rows(), ZU, C1, T)
    assert res["outcome"] == JUDGED and res["label"] == LEARNS_CONFIRMATORY
    for d in ("r1", "r2"):
        s = res["directions"][d]
        assert s["outcome"] == CONFIRMED and s["i"] and s["ii"] and s["iii"]
        assert s["ell"] == pytest.approx(2.4) and s["s"] == pytest.approx(-2.4) and s["t"] == pytest.approx(0.0)
    assert res["mean_ell"]["ell"] == pytest.approx(2.4) and res["mean_ell"]["record"] is True


def test_mutation_one_direction_zeroed_is_direction_dependent():
    res = p_judge(make_rows(effect={"r1": 2.4, "r2": 0.0}), ZU, C1, T)
    assert res["directions"]["r2"]["outcome"] == EXCLUDED and res["label"] == DIRECTION_DEPENDENT


def test_mutation_one_direction_reversed_never_passes():
    res = p_judge(make_rows(effect={"r1": 2.4, "r2": -2.4}), ZU, C1, T)
    assert res["directions"]["r2"]["outcome"] == EXCLUDED and res["label"] == DIRECTION_DEPENDENT
    assert res["mean_ell"]["ell"] == pytest.approx(0.0, abs=1e-9)          # the average alone hides the reversal


def test_no_effect_anywhere_is_no_learning():
    assert p_judge(make_rows(effect={"r1": 0.0, "r2": 0.0}), ZU, C1, T)["label"] == NO_LEARNING


def test_a_small_noisy_effect_is_inconclusive():
    res = p_judge(make_rows(effect={"r1": 0.5, "r2": 0.5}, jitter=1.0), ZU, C1, T)
    assert {s["outcome"] for s in res["directions"].values()} == {UNDECIDED} and res["label"] == INCONCLUSIVE


def test_a_change_spread_onto_y_fails_concentration():
    res = p_judge(make_rows(effect={"r1": 3.0, "r2": 2.4}, y_effect={"r1": 2.0}), ZU, C1, T)
    s = res["directions"]["r1"]
    assert s["ell"] == pytest.approx(1.0) and s["i"] and s["ii"] and not s["iii"]
    assert s["outcome"] == UNDECIDED and res["label"] == INCONCLUSIVE


def test_x_not_lowered_itself_fails_criterion_ii():
    res = p_judge(make_rows(effect={"r1": 0.0, "r2": 2.4}, y_effect={"r1": -2.0}), ZU, C1, T)
    s = res["directions"]["r1"]
    assert s["ell"] == pytest.approx(2.0) and s["i"] and not s["ii"] and s["outcome"] == UNDECIDED
    assert res["label"] == INCONCLUSIVE


def test_mutation_frozen_weights_moved_is_invalid():
    rows = make_rows()
    for r in rows:
        if r["direction"] == "r1" and r["arm"] == "frozen" and r["seed"] == T.seeds[3]:
            r["w_post_sha256"] = "moved"
    res = p_judge(rows, ZU, C1, T)
    assert res["outcome"] == res["label"] == INVALID and res["invalid_directions"] == ["r1"]
    assert "plumbing" in res["reasons"][0] and str(T.seeds[3]) in res["reasons"][0]


@pytest.mark.parametrize("mutate, word", [
    (lambda rows: rows.pop(), "no row"),
    (lambda rows: rows.append(dict(rows[0])), "more than once"),
    (lambda rows: rows[0]["post"]["x"].update(A=float("nan")), "non-finite"),
    (lambda rows: rows[0].update(seed=99), "undeclared seed"),
    (lambda rows: rows[0].update(x="1:4"), "(X, Y)"),
    (lambda rows: rows[0].update(punish=not rows[0]["punish"]), "flags"),
    (lambda rows: rows[0].update(point=[1.0, 1.0]), "operating point"),
    (lambda rows: rows[0].update(edit="apl_to_kc_zero"), "edit"),
])
def test_the_gate_names_each_defect(mutate, word):
    rows = make_rows()
    mutate(rows)
    res = p_judge(rows, ZU, C1, T)
    assert res["outcome"] == INVALID and word in " ".join(res["reasons"])


@pytest.mark.parametrize("c1", [float("nan"), 0.0, -0.6, None])
def test_a_missing_or_nonpositive_c1_is_invalid(c1):
    res = p_judge(make_rows(), ZU, c1, T)
    assert res["outcome"] == INVALID and "c1" in " ".join(res["reasons"])


def test_row_order_does_not_change_the_result():
    rows = make_rows(effect={"r1": 2.4, "r2": 0.9}, jitter=0.8)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    a, b = p_judge(rows, ZU, C1, T), p_judge(shuffled, ZU, C1, T)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_label_order():
    assert label({"r1": CONFIRMED, "r2": CONFIRMED}) == LEARNS_CONFIRMATORY
    assert label({"r1": CONFIRMED, "r2": EXCLUDED}) == label({"r1": EXCLUDED, "r2": CONFIRMED}) == DIRECTION_DEPENDENT
    assert label({"r1": EXCLUDED, "r2": EXCLUDED}) == NO_LEARNING
    for pair in ((CONFIRMED, UNDECIDED), (UNDECIDED, EXCLUDED), (UNDECIDED, UNDECIDED)):
        assert label(dict(zip(("r1", "r2"), pair))) == INCONCLUSIVE


def test_one_bootstrap_for_both_directions():
    rows = make_rows(effect={"r1": 2.4, "r2": 1.5}, jitter=0.9)
    res = p_judge(rows, ZU, C1, T)
    by = group(rows, T)
    vec = {d: arm_vectors(by[d], ZU, T) for d in T.directions}
    W = boot_weights(16, T.o.boot_draws, T.o.boot_seed)
    for d in T.directions:
        assert res["directions"][d]["ell_ci"] == _ci(W @ vec[d]["dl"], T.o.ci_level)
    assert res["mean_ell"]["ci"] == _ci(W @ np.mean([vec["r1"]["dl"], vec["r2"]["dl"]], axis=0), T.o.ci_level)


def test_directions_must_share_their_seeds():
    v = dict(seeds=[1, 2], dl=np.ones(2), ds=-np.ones(2), dt=np.zeros(2))
    with pytest.raises(ValueError):
        judge_vectors({"r1": v, "r2": dict(v, seeds=[1, 3])}, C1, T)


def test_zero_variance_effects_are_judged_not_crashed():
    assert p_judge(make_rows(jitter=0.0), ZU, C1, T)["label"] == LEARNS_CONFIRMATORY
    res = p_judge(make_rows(effect={"r1": 0.0, "r2": 0.0}, jitter=0.0), ZU, C1, T)
    assert res["label"] == NO_LEARNING and res["directions"]["r1"]["ell_ci"] == [0.0, 0.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_p_rules.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.p_rules`).

- [ ] **Step 3: Write `p_rules.py`**

```python
"""The pure rules of spec appendix P as amended by P.6 (P.6 wins over P.1-P.4). No engine, no pool.
- The gate (P.3, P.6.5): o_rules.validity (O.7.5: complete, unique keys, finite, declared seeds, one CSC sha) plus every
  row's declared expectations (direction pair, operating point, arm flags) and a positive finite c1; then arm ②'s
  bit-identity (sha256 of the plastic weights == w0's). Any failure: the whole judgement INVALID.
- The judgement (P.6.2): per direction r, per seed, over the punish (③) and plastic (①) arms
    dl = -(Δ(dV)[③] - Δ(dV)[①]),  ds = ΔV_X[③] - ΔV_X[①],  dt = ΔV_Y[③] - ΔV_Y[①]   (dl = dt - ds),
  one seed bootstrap shared by both directions (the same seed numbers resampled together, P.3), the per-direction
  outcome (confirmed / excluded / undecided) and the label in P.6.2's order.
- The records (P.3, P.6.2, P.6.3): the averaged ℓ, raw-unit changes, the non-associative change, the flip re-analysis
  (no suffix), the pre-state strata, the state-conditional numbers, dopamine integrals and core weight fractions.
- The operating characteristic (P.6.4) from O2's per-seed values.
The presentation state is n_rules.state, Δ(dV) is n_rules.delta; every threshold is a field of the PSpec passed in."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .n_rules import INVALID, NO_LEARNING, _ci, delta, state
from .o_rules import INCONCLUSIVE, JUDGED, boot_weights, finite, validity

LEARNS_CONFIRMATORY = "LEARNS_CONFIRMATORY"
DIRECTION_DEPENDENT = "DIRECTION_DEPENDENT"
LABELS = (INVALID, LEARNS_CONFIRMATORY, DIRECTION_DEPENDENT, NO_LEARNING, INCONCLUSIVE)   # P.6.2's evaluation order
CONFIRMED = "LEARNING_CONFIRMED"                  # per direction: (i)-(iii) all hold
EXCLUDED = "LEARNING_EXCLUDED"                    # per direction: CI upper of ℓ_r < c1
UNDECIDED = "UNDECIDED"
OC_RECORDED = "RECORDED"
FLIP_NOTE = ("P.6.3: 판정은 무조건부 분석만으로 한다 — 뒤집힌 시드를 뺀 재분석은 접미사 없는 기록이다(①·③은 사전 프로브가 같아 "
             "뒤집힘은 처치 뒤 변수다)")


# ================================================================ the gate (P.3, P.6.5)
def p_key(r) -> tuple:
    return (r["direction"], r["arm"], int(r["seed"]))


def p_declared(spec) -> set:
    return {(d, a, int(s)) for d in spec.directions for a in spec.arms for s in spec.seeds}


def expectations(rows, spec) -> list:
    """P.6.5: every row ran its direction's (X, Y), its arm's (punish, plastic, da_zero) flags and the declared point.
    The reasons, [] when every row matches."""
    pairs, flags, point = spec.pairs(), spec.flags(), [float(v) for v in spec.o.o2_point]
    wrong = defaultdict(list)
    for r in rows:
        k = p_key(r)
        if r["direction"] in pairs and (r.get("x"), r.get("y")) != pairs[r["direction"]]:
            wrong["ran another (X, Y) than its direction declares"].append(k)
        got = tuple(bool(r.get(f)) for f in ("punish", "plastic", "da_zero"))
        if r["arm"] in flags and got != tuple(flags[r["arm"]]):
            wrong["ran other arm flags than its arm declares"].append(k)
        if [float(v) for v in r.get("point", ())] != point:
            wrong[f"ran another operating point than {point}"].append(k)
    return [f"{len(ks)} row(s) {why}, e.g. {sorted(ks, key=str)[:3]}" for why, ks in sorted(wrong.items())]


def group(rows, spec) -> dict:
    """{direction: {arm: rows in seed order}}."""
    by = {d: {a: [] for a in spec.arms} for d in spec.directions}
    for r in rows:
        by[r["direction"]][r["arm"]].append(r)
    for d in by:
        for a in by[d]:
            by[d][a].sort(key=lambda r: int(r["seed"]))
    return by


def plumbing(by: dict, spec) -> dict:
    """P.3: arm ② (plasticity off) ends with the plastic weights bit-identical to w0 (sha256), every seed. {direction:
    reasons}, only directions with a reason."""
    two = spec.arms[1]
    out = {}
    for d in spec.directions:
        bad = [f"plumbing: direction {d} arm {two} seed {r['seed']} moved its weights (sha256 "
               f"{r['w_post_sha256'][:12]} != w0 {r['w0_sha256'][:12]})" for r in by[d][two]
               if r["w_post_sha256"] != r["w0_sha256"]]
        if bad:
            out[d] = bad
    return out


# ================================================================ the judgement (P.6.2)
def v_of(probe: dict, z: dict) -> float:
    """V = z_A - z_P of one probe, C3's frozen z (h4_formula.dv is V_X - V_Y of the same: test)."""
    return float((probe["A"] - z["A"][0]) / z["A"][1] - (probe["P"] - z["P"][0]) / z["P"][1])


def _dv(rows, side: str, z: dict) -> np.ndarray:
    return np.array([v_of(r["post"][side], z) - v_of(r["pre"][side], z) for r in rows], float)


def arm_vectors(by: dict, z: dict, spec) -> dict:
    """One direction's per-seed vectors over arms ① and ③ (module docstring). Both arms must hold the same seeds in
    the same order (ValueError otherwise: a pairing defect is never judged)."""
    one, _, three = spec.arms
    seeds = [int(r["seed"]) for r in by[one]]
    if seeds != [int(r["seed"]) for r in by[three]]:
        raise ValueError("arms 1 and 3 must hold the same seeds in the same order")
    ds = _dv(by[three], "x", z) - _dv(by[one], "x", z)
    dt = _dv(by[three], "y", z) - _dv(by[one], "y", z)
    dl = -(np.array([delta(r, z) for r in by[three]], float) - np.array([delta(r, z) for r in by[one]], float))
    return dict(seeds=seeds, dl=dl, ds=ds, dt=dt)


def direction_stats(dl, ds, dt, W, c1: float, spec) -> dict:
    """P.6.2 for one direction: (i) CI lower of ℓ_r > 0 and ℓ_r >= c1; (ii) CI upper of s_r < 0; (iii) |t_r| <=
    concentration_frac |s_r| (point estimates). Confirmed iff (i)-(iii), else excluded iff CI upper of ℓ_r < c1, else
    undecided (reading 7: confirmed is tested first)."""
    lv = spec.o.ci_level
    dl, ds, dt = (np.asarray(v, float) for v in (dl, ds, dt))
    ell, s, t = float(dl.mean()), float(ds.mean()), float(dt.mean())
    ell_ci, s_ci, t_ci = _ci(W @ dl, lv), _ci(W @ ds, lv), _ci(W @ dt, lv)
    i = bool(ell_ci[0] > 0 and ell >= c1)
    ii = bool(s_ci[1] < 0)
    iii = bool(abs(t) <= spec.concentration_frac * abs(s))
    if i and ii and iii:
        out = CONFIRMED
    elif ell_ci[1] < c1:
        out = EXCLUDED
    else:
        out = UNDECIDED
    return dict(n=int(dl.size), ell=ell, ell_ci=ell_ci, s=s, s_ci=s_ci, t=t, t_ci=t_ci, i=i, ii=ii, iii=iii,
                outcome=out)


def label(outcomes: dict) -> str:
    """P.6.2's order after INVALID: every direction confirmed; one confirmed and the rest excluded; every direction
    excluded; anything else."""
    v = set(outcomes.values())
    if v == {CONFIRMED}:
        return LEARNS_CONFIRMATORY
    if v == {CONFIRMED, EXCLUDED}:
        return DIRECTION_DEPENDENT
    if v == {EXCLUDED}:
        return NO_LEARNING
    return INCONCLUSIVE


def judge_vectors(vec: dict, c1: float, spec, W=None) -> dict:
    """The label from per-direction vectors {direction: {seeds, dl, ds, dt}}. One resampling matrix W for every
    direction (P.3: the same seed numbers resampled together), so the directions must hold the same seeds (ValueError).
    The averaged ℓ and its CI are a record (P.6.2)."""
    if len({tuple(v["seeds"]) for v in vec.values()}) != 1:
        raise ValueError("the directions must hold the same seed numbers in the same order")
    n = len(next(iter(vec.values()))["seeds"])
    if W is None:
        W = boot_weights(n, spec.o.boot_draws, spec.o.boot_seed)
    dirs = {d: direction_stats(v["dl"], v["ds"], v["dt"], W, c1, spec) for d, v in vec.items()}
    mean_dl = np.mean([np.asarray(vec[d]["dl"], float) for d in vec], axis=0)
    return dict(label=label({d: s["outcome"] for d, s in dirs.items()}), directions=dirs,
                mean_ell=dict(ell=float(mean_dl.mean()), ci=_ci(W @ mean_dl, spec.o.ci_level), record=True))


# ================================================================ the stage (P.6.2, P.6.5)
def p_judge(rows: list, z: dict, c1, spec) -> dict:
    """The gate (INVALID), arm ②'s plumbing (any direction: INVALID), then the judgement (Task 3 adds the records)."""
    bad = validity(rows, p_declared(spec), p_key, spec.seeds, lambda r: spec.o.on_edit) + expectations(rows, spec)
    if not (finite(c1) and float(c1) > 0):
        bad.append(f"c1 {c1!r} is not a positive finite number")
    if bad:
        return dict(outcome=INVALID, label=INVALID, reasons=bad, n_rows=len(rows))
    by = group(rows, spec)
    pl = plumbing(by, spec)
    if pl:
        return dict(outcome=INVALID, label=INVALID, reasons=[m for d in pl for m in pl[d]],
                    invalid_directions=sorted(pl), n_rows=len(rows))
    vec = {d: arm_vectors(by[d], z, spec) for d in spec.directions}
    main = judge_vectors(vec, float(c1), spec)
    return dict(outcome=JUDGED, label=main["label"], reasons=[], c1=float(c1), pairs=spec.pairs(),
                directions=main["directions"], mean_ell=main["mean_ell"], seeds={d: v["seeds"] for d, v in vec.items()})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_p_rules.py -q -o addopts=""`
Expected: PASS (27 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/p_rules.py tests/brain/test_p_rules.py
git commit -m "feat(p): p_rules — gate (O.7.5 + pair/point/flags + c1), arm 2 plumbing, per-direction (i)-(iii), labels in P.6.2 order, joint bootstrap, mutation tests"
```

Expected suite: **1777 passed / 1 skipped / 3 xfailed**.

---

### Task 3: `p_rules` (2) — the records, the flip re-analysis, the pre-state strata, the sentence

**Files:**
- Modify: `flymon/brain/p_rules.py` (insert the records section before `# ===... the stage`, replace `p_judge`, append the sentences section)
- Test: `tests/brain/test_p_rules.py` (append)

**Interfaces:**
- Consumes: Task 2's names.
- Produces: `RAW`, `raw_records(by, z, spec)`, `flips(by, spec)`, `flip_reanalysis(by_dir, vec, c1, spec)`, `strata(by, v, c1, spec)`, `state_conditional(by, v, c1, spec)`, `mechanism_records(by, spec)`, `sentence(res) -> str`; `p_judge`'s judged result gains `flip` (keys `flips`, `limit`, `triggered`, `dropped`, `reanalysis`, `note`) and `records` (per direction: `raw`, `strata`, `state_conditional`, `da_integral`, `weights_frac`, `frozen_probes_identical`).

- [ ] **Step 1: Append the failing tests to `tests/brain/test_p_rules.py`**

```python
# ---- Task 3: the records (P.3, P.6.2, P.6.3) and the sentence
from flymon.brain.p_rules import FLIP_NOTE, sentence  # noqa: E402


def test_raw_and_nonassociative_records():
    rec = p_judge(make_rows(), ZU, C1, T)["records"]["r1"]
    raw = rec["raw"]
    assert raw["arms"]["punish"]["dA_x"]["mean"] == pytest.approx(-4.4)
    assert raw["arms"]["frozen"]["dA_x"]["mean"] == 0.0 and raw["arms"]["plastic"]["dA_y"]["mean"] == 0.0
    assert raw["punish_minus_plastic"]["dA_x"]["mean"] == pytest.approx(-2.4)
    assert raw["nonassociative"]["d_dV"]["mean"] == pytest.approx(-2.0)
    assert raw["nonassociative"]["dV_X"]["mean"] == pytest.approx(-2.0)
    assert rec["da_integral"]["punish"] == {"PPL105": 1.0} and rec["da_integral"]["plastic"] == {"PPL105": 0.0}
    assert rec["weights_frac"]["frozen"]["all"] == 1.0 and rec["frozen_probes_identical"] is True


def test_no_flip_no_reanalysis():
    f = p_judge(make_rows(), ZU, C1, T)["flip"]
    assert f["triggered"] is False and f["reanalysis"] is None and f["limit"] == T.o.flip_max_frac * 16


def test_flips_give_an_unsuffixed_reanalysis_and_leave_the_label_alone():
    res = p_judge(make_rows(flip={0, 1, 2, 3, 5}), ZU, C1, T)
    f = res["flip"]
    assert f["triggered"] is True and f["flips"]["r1"]["plastic"] == [T.seeds[i] for i in (0, 1, 2, 3, 5)]
    assert f["dropped"] == [T.seeds[i] for i in (0, 1, 2, 3, 5)]
    assert f["reanalysis"]["label"] == LEARNS_CONFIRMATORY and f["reanalysis"]["directions"]["r1"]["n"] == 11
    assert res["label"] == LEARNS_CONFIRMATORY and "SENSITIVE" not in json.dumps(res) and f["note"] == FLIP_NOTE


def test_flips_leaving_fewer_than_two_seeds_give_no_reanalysis():
    f = p_judge(make_rows(flip=set(range(15))), ZU, C1, T)["flip"]
    assert f["triggered"] is True and f["reanalysis"] is None


def test_pre_state_strata_and_state_conditional():
    rec = p_judge(make_rows(silent_pre={0, 1, 2, 3}), ZU, C1, T)["records"]["r2"]
    st = rec["strata"]
    assert st["pre_identical"] is True and st["silent"]["n"] == 4 and st["firing"]["n"] == 12
    assert st["firing"]["stats"]["outcome"] == CONFIRMED and st["silent"]["stats"] is not None
    assert rec["state_conditional"]["n"] == 12
    one = p_judge(make_rows(silent_pre={0}), ZU, C1, T)["records"]["r2"]["strata"]["silent"]
    assert one["n"] == 1 and one["stats"] is None


def test_sentences():
    ok = sentence(p_judge(make_rows(), ZU, C1, T))
    assert ok.startswith("P: LEARNS_CONFIRMATORY") and "확인 시험" in ok and "C3" in ok and "r2(X = dDL)" in ok
    rows = make_rows()
    rows.pop()
    assert sentence(p_judge(rows, ZU, C1, T)).startswith("P INVALID:")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_p_rules.py -q -o addopts=""`
Expected: FAIL (`ImportError: cannot import name 'sentence'`).

- [ ] **Step 3: Insert the records section into `p_rules.py`, directly above the line `# ================================================================ the stage (P.6.2, P.6.5)`**

```python
# ================================================================ the records (P.3, P.6.2, P.6.3)
def _subset(v: dict, keep) -> dict:
    idx = [i for i, s in enumerate(v["seeds"]) if s in keep]
    return dict(seeds=[v["seeds"][i] for i in idx], **{k: np.asarray(v[k], float)[idx] for k in ("dl", "ds", "dt")})


def _stat(v, W, spec) -> dict:
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), ci=_ci(W @ v, spec.o.ci_level))


def _raw(rows, side: str, field: str) -> np.ndarray:
    return np.array([r["post"][side][field] - r["pre"][side][field] for r in rows], float)


RAW = (("x", "A"), ("x", "P"), ("y", "A"), ("y", "P"))


def raw_records(by: dict, z: dict, spec) -> dict:
    """P.3's records for one direction: ΔA_X, ΔP_X, ΔA_Y, ΔP_Y per arm and ③ - ① (raw units), and ①'s
    non-associative change ① - ② of Δ(dV) and ΔV_X; seed-bootstrap CIs."""
    one, two, three = spec.arms
    W = boot_weights(len(by[one]), spec.o.boot_draws, spec.o.boot_seed)
    arms = {a: {f"d{f}_{s}": _stat(_raw(by[a], s, f), W, spec) for s, f in RAW} for a in spec.arms}
    p_minus_1 = {f"d{f}_{s}": _stat(_raw(by[three], s, f) - _raw(by[one], s, f), W, spec) for s, f in RAW}
    ddv = {a: np.array([delta(r, z) for r in by[a]], float) for a in (one, two)}
    nonassoc = dict(d_dV=_stat(ddv[one] - ddv[two], W, spec),
                    dV_X=_stat(_dv(by[one], "x", z) - _dv(by[two], "x", z), W, spec))
    return dict(arms=arms, punish_minus_plastic=p_minus_1, nonassociative=nonassoc)


def flips(by: dict, spec) -> dict:
    """Per arm, the seeds whose X probe changed state (n_rules.state) from pre to post."""
    return {a: [int(r["seed"]) for r in rows
                if state(r["pre"]["x"]["P"], spec.o.n) != state(r["post"]["x"]["P"], spec.o.n)]
            for a, rows in by.items()}


def flip_reanalysis(by_dir: dict, vec: dict, c1: float, spec) -> dict:
    """P.6.3: O.7.4-4's trigger (an arm of a direction with more than flip_max_frac of its seeds flipped); then the
    judgement without the union of flipped seeds (every arm, both directions), a record with no suffix (None below 2)."""
    fl = {d: flips(by_dir[d], spec) for d in spec.directions}
    n = len(next(iter(vec.values()))["seeds"])
    limit = spec.o.flip_max_frac * n
    triggered = any(len(s) > limit for d in fl for s in fl[d].values())
    out = dict(flips=fl, limit=limit, triggered=triggered, dropped=None, reanalysis=None, note=FLIP_NOTE)
    if triggered:
        drop = sorted(set().union(*(set(s) for d in fl for s in fl[d].values())))
        keep = set(next(iter(vec.values()))["seeds"]) - set(drop)
        out["dropped"] = drop
        if len(keep) >= 2:
            out["reanalysis"] = judge_vectors({d: _subset(v, keep) for d, v in vec.items()}, c1, spec)
    return out


def _probe_same(a: dict, b: dict) -> bool:
    return all(a[k][f] == b[k][f] for k in ("x", "y") for f in ("A", "P", "kc_spikes"))


def strata(by: dict, v: dict, c1: float, spec) -> dict:
    """P.6.3: one direction's statistics within each pre-probe state of X (arm ③'s pre probe; arm ①'s is the same
    probe — `pre_identical`), each stratum with its own bootstrap; None below 2 seeds. A record."""
    one, _, three = spec.arms
    st = {int(r["seed"]): state(r["pre"]["x"]["P"], spec.o.n) for r in by[three]}
    out = dict(pre_identical=bool(all(_probe_same(a["pre"], b["pre"]) for a, b in zip(by[one], by[three]))))
    for s in ("firing", "silent"):
        keep = {k for k, x in st.items() if x == s}
        sub = _subset(v, keep)
        out[s] = dict(n=len(keep), stats=(direction_stats(sub["dl"], sub["ds"], sub["dt"],
                                                          boot_weights(len(keep), spec.o.boot_draws, spec.o.boot_seed),
                                                          c1, spec) if len(keep) >= 2 else None))
    return out


def state_conditional(by: dict, v: dict, c1: float, spec) -> dict:
    """P.3's record: the seeds whose every probe (pre / post x X / Y, every arm of the direction) fires."""
    keep = {int(r["seed"]) for r in by[spec.arms[0]]}
    for rows in by.values():
        for r in rows:
            if any(state(r[ph][k]["P"], spec.o.n) != "firing" for ph in ("pre", "post") for k in ("x", "y")):
                keep.discard(int(r["seed"]))
    sub = _subset(v, keep)
    return dict(n=len(keep), stats=(direction_stats(sub["dl"], sub["ds"], sub["dt"],
                                                    boot_weights(len(keep), spec.o.boot_draws, spec.o.boot_seed), c1,
                                                    spec) if len(keep) >= 2 else None))


def mechanism_records(by: dict, spec) -> dict:
    """P.3's records: per arm the mean phasic-dopamine integral per compartment and the core weight fractions."""
    return dict(
        da_integral={a: {k: float(np.mean([r["da_integral"][k] for r in rows])) for k in rows[0]["da_integral"]}
                     for a, rows in by.items()},
        weights_frac={a: {k: float(np.mean([r[f] for r in rows]))
                          for k, f in (("all", "weights_frac"), ("A_punish_core", "weights_frac_A"),
                                       ("P_reward_core", "weights_frac_P"))}
                      for a, rows in by.items()},
        frozen_probes_identical=bool(all(_probe_same(r["pre"], r["post"]) for r in by[spec.arms[1]])))
```

- [ ] **Step 4: Replace the whole `the stage` section (`p_judge`) with**

```python
# ================================================================ the stage (P.6.2, P.6.5)
def p_judge(rows: list, z: dict, c1, spec) -> dict:
    """The gate (INVALID), arm ②'s plumbing (any direction: INVALID), then the judgement and the records."""
    bad = validity(rows, p_declared(spec), p_key, spec.seeds, lambda r: spec.o.on_edit) + expectations(rows, spec)
    if not (finite(c1) and float(c1) > 0):
        bad.append(f"c1 {c1!r} is not a positive finite number")
    if bad:
        return dict(outcome=INVALID, label=INVALID, reasons=bad, n_rows=len(rows))
    by = group(rows, spec)
    pl = plumbing(by, spec)
    if pl:
        return dict(outcome=INVALID, label=INVALID, reasons=[m for d in pl for m in pl[d]],
                    invalid_directions=sorted(pl), n_rows=len(rows))
    vec = {d: arm_vectors(by[d], z, spec) for d in spec.directions}
    main = judge_vectors(vec, float(c1), spec)
    records = {d: dict(raw=raw_records(by[d], z, spec), strata=strata(by[d], vec[d], float(c1), spec),
                       state_conditional=state_conditional(by[d], vec[d], float(c1), spec),
                       **mechanism_records(by[d], spec)) for d in spec.directions}
    return dict(outcome=JUDGED, label=main["label"], reasons=[], c1=float(c1), pairs=spec.pairs(),
                directions=main["directions"], mean_ell=main["mean_ell"],
                flip=flip_reanalysis(by, vec, float(c1), spec), records=records,
                seeds={d: v["seeds"] for d, v in vec.items()})
```

- [ ] **Step 5: Append the sentences section at the end of `p_rules.py`**

```python
# ================================================================ sentences
def sentence(res: dict) -> str:
    if res["outcome"] == INVALID:
        return f"P INVALID: {'; '.join(res['reasons'][:3])}"
    c1 = res["c1"]
    parts = [f"{d}(X = {res['pairs'][d][0]}) {s['outcome']}, ℓ {s['ell']:.3f} [{s['ell_ci'][0]:.3f}, "
             f"{s['ell_ci'][1]:.3f}], s {s['s']:.3f}, t {s['t']:.3f}" for d, s in res["directions"].items()]
    return (f"P: {res['label']} — " + " / ".join(parts) + f"; c₁ {c1:.3f}. 새 시드에서의 확인 시험(O2를 본 뒤의 재현, "
            "P.0)이다. 연합성은 가소성 규칙(KC 흔적 × 위상 도파민)의 귀결이지 측정이 아니다(P.6.1). 결론은 이 커넥톰 모델(C3)·"
            "이 쌍·이 절차까지이며 포켓몬 1차 주장의 판정이 아니다(P.1, P.5).")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_p_rules.py -q -o addopts=""`
Expected: PASS (33 tests).

- [ ] **Step 7: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/p_rules.py tests/brain/test_p_rules.py
git commit -m "feat(p): p_rules records — raw and non-associative changes, unsuffixed flip re-analysis, pre-state strata, state-conditional, dopamine and core weights, sentence"
```

Expected suite: **1783 passed / 1 skipped / 3 xfailed**.

---

### Task 4: `p_rules` (3) — the operating characteristic (P.6.4) and the literal guard

**Files:**
- Modify: `flymon/brain/p_rules.py` (append)
- Test: `tests/brain/test_p_rules.py` (append)

**Interfaces:**
- Consumes: Task 2's `arm_vectors`, `judge_vectors`, `boot_weights`; `o_spec.SPEC` (O2's pairs, passed in as `o_spec`).
- Produces: `o2_vectors(rows, z, spec, o_spec) -> {d: {"seeds", "dl", "ds", "dt", "x", "y", "o2_y", "y_matches"}}`, `null_shift(v, c1) -> dict`, `oc_run(vec, c1, spec) -> dict` (keys `outcome` = `OC_RECORDED`, `n`, `source_n`, `draws`, `seed`, `c1`, `scenarios` {`observed`, `null_both`, `null_r1`, `null_r2`} → {`p_learns`, `labels`}, `false_pass_max`, `o2_point`), `oc_sentence(res) -> str`.

- [ ] **Step 1: Append the failing tests to `tests/brain/test_p_rules.py`**

```python
# ---- Task 4: the operating characteristic (P.6.4) and the literal guard
import ast  # noqa: E402
from pathlib import Path  # noqa: E402

from flymon.brain.o_spec import SPEC as O_SPEC  # noqa: E402
from flymon.brain.p_rules import OC_RECORDED, null_shift, o2_vectors, oc_run, oc_sentence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TO = replace(T, oc_draws=40)


def make_o2_rows(effect=None, n=16, jitter=0.4):
    """O2-shaped rows (O's pairs, all four arms, seeds 22_001_000 + i): the punish arm lowers A_X by 2 + effect[x] + j."""
    effect = effect or {"4:1": 2.0, "dDL": 2.4}
    out = []
    for x, y, _ in O_SPEC.o2_pairs:
        for a, pu, pl, dz in O_SPEC.o2_arms:
            for i in range(n):
                s = O_SPEC.o2_seed0 + i
                j = jitter * ((i % 4) - 1.5)
                ax = {"plastic": 2.0, "frozen": 0.0, "punish": 2.0 + effect[x] + j, "da_zero": 0.0}[a]
                out.append(dict(x=x, y=y, arm=a, seed=s, punish=pu, plastic=pl, da_zero=dz,
                                pre={"x": _probe(s), "y": _probe(s)},
                                post={"x": _probe(s, A=20.0 - ax), "y": _probe(s)}))
    return out


def test_o2_vectors_map_each_direction_to_the_o2_x():
    vec = o2_vectors(make_o2_rows(), ZU, T, O_SPEC)
    assert vec["r1"]["x"] == "4:1" and vec["r1"]["o2_y"] == "1:4" and vec["r1"]["y_matches"] is False
    assert vec["r2"]["x"] == "dDL" and vec["r2"]["o2_y"] == "4:1" and vec["r2"]["y_matches"] is True
    assert np.mean(vec["r1"]["dl"]) == pytest.approx(2.0) and np.mean(vec["r2"]["dl"]) == pytest.approx(2.4)
    assert vec["r1"]["seeds"] == vec["r2"]["seeds"] == [O_SPEC.o2_seed0 + i for i in range(16)]


def test_null_shift_puts_the_mean_at_c1_and_keeps_y():
    v = o2_vectors(make_o2_rows(), ZU, T, O_SPEC)["r2"]
    w = null_shift(v, C1)
    assert np.mean(w["dl"]) == pytest.approx(C1) and np.array_equal(w["dt"], v["dt"])
    assert np.allclose(w["dl"], w["dt"] - w["ds"]) and np.std(w["dl"]) == pytest.approx(np.std(v["dl"]))


def test_oc_run_records_every_scenario_deterministically():
    vec = o2_vectors(make_o2_rows(), ZU, T, O_SPEC)
    a, b = oc_run(vec, C1, TO), oc_run(vec, C1, TO)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["outcome"] == OC_RECORDED and a["n"] == 16 and a["source_n"] == 16 and a["draws"] == 40
    assert set(a["scenarios"]) == {"observed", "null_both", "null_r1", "null_r2"}
    for sc in a["scenarios"].values():
        assert sum(sc["labels"].values()) == pytest.approx(1.0)
        assert sc["p_learns"] == sc["labels"][LEARNS_CONFIRMATORY]
    assert a["scenarios"]["observed"]["p_learns"] == 1.0
    assert a["false_pass_max"] == max(a["scenarios"][k]["p_learns"] for k in ("null_both", "null_r1", "null_r2"))
    assert a["false_pass_max"] < 1.0 and a["o2_point"]["r1"]["y_matches"] is False
    assert "r1은 O2의 X = 4:1 행" in oc_sentence(a)


def test_p_rules_holds_no_threshold_literal():
    tree = ast.parse((ROOT / "flymon/brain/p_rules.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1, 2, 3, 12}, nums           # indices, "at least 2 seeds", reason / sha excerpts
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_p_rules.py -q -o addopts=""`
Expected: FAIL (`ImportError: cannot import name 'null_shift'`).

- [ ] **Step 3: Append the OC section at the end of `p_rules.py`**

```python
# ================================================================ the operating characteristic (P.6.4)
def o2_vectors(rows: list, z: dict, spec, o_spec) -> dict:
    """O2's per-seed vectors for each P direction: the O2 X equal to the direction's X, arms ① and ③. r1's Y differs
    in O2 (X = 4:1 ran with Y = 1:4): recorded as y_matches False (P.6.4's stated limitation)."""
    o2_y = {x: y for x, y, _ in o_spec.o2_pairs}
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["x"]][r["arm"]].append(r)
    out = {}
    for d, (x, y) in spec.pairs().items():
        if x not in o2_y:
            raise ValueError(f"O2 ran no X = {x}")
        arms = {a: sorted(by[x][a], key=lambda r: int(r["seed"])) for a in spec.arms}
        out[d] = dict(arm_vectors(arms, z, spec), x=x, y=y, o2_y=o2_y[x], y_matches=o2_y[x] == y)
    return out


def null_shift(v: dict, c1: float) -> dict:
    """The null at c1: X's own change ds moved by mean(dl) - c1 on every seed, Y untouched, so mean(dl) = c1 exactly."""
    ds = np.asarray(v["ds"], float) + (float(np.mean(v["dl"])) - float(c1))
    return dict(v, ds=ds, dl=np.asarray(v["dt"], float) - ds)


def oc_run(vec: dict, c1: float, spec) -> dict:
    """P.6.4: oc_draws simulated experiments of n = o2_n_seeds seeds, each resampled with replacement from O2's seeds
    (one index vector for both directions), judged by judge_vectors with the real bootstrap. Scenarios: observed;
    null_both (both directions at c1); null_<d> (direction d at c1, the other as observed). Label shares per scenario."""
    n = spec.o.o2_n_seeds
    src_n = len(next(iter(vec.values()))["dl"])
    scen = {"observed": vec, "null_both": {d: null_shift(v, c1) for d, v in vec.items()}}
    for d in spec.directions:
        scen[f"null_{d}"] = {e: (null_shift(v, c1) if e == d else v) for e, v in vec.items()}
    W = boot_weights(n, spec.o.boot_draws, spec.o.boot_seed)
    rng = np.random.default_rng(spec.oc_seed)
    counts = {k: Counter() for k in scen}
    for _ in range(int(spec.oc_draws)):
        idx = rng.integers(0, src_n, n)
        for k, vv in scen.items():
            draw = {d: dict(seeds=list(range(n)), **{f: np.asarray(v[f], float)[idx] for f in ("dl", "ds", "dt")})
                    for d, v in vv.items()}
            counts[k][judge_vectors(draw, c1, spec, W)["label"]] += 1
    m = int(spec.oc_draws)
    return dict(outcome=OC_RECORDED, n=n, source_n=src_n, draws=m, seed=spec.oc_seed, c1=float(c1),
                scenarios={k: dict(p_learns=c[LEARNS_CONFIRMATORY] / m, labels={lab: c[lab] / m for lab in LABELS[1:]})
                           for k, c in counts.items()},
                false_pass_max=max(counts[k][LEARNS_CONFIRMATORY] / m for k in scen if k != "observed"),
                o2_point={d: dict(ell=float(np.mean(v["dl"])), s=float(np.mean(v["ds"])), t=float(np.mean(v["dt"])),
                                  x=v.get("x"), y=v.get("y"), o2_y=v.get("o2_y"), y_matches=v.get("y_matches"))
                          for d, v in vec.items()})


def oc_sentence(res: dict) -> str:
    sc = res["scenarios"]
    nulls = ", ".join(f"{k} {v['p_learns']:.3f}" for k, v in sc.items() if k != "observed")
    return (f"P OC: n = {res['n']}에서 LEARNS_CONFIRMATORY 확률 — 관측 {sc['observed']['p_learns']:.3f}, c₁ 귀무 {nulls} "
            f"(실험 {res['draws']}회). r1은 O2의 X = 4:1 행(Y = 1:4)으로 대신했다(P.6.4). n은 바꾸지 않는다.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_p_rules.py -q -o addopts=""`
Expected: PASS (37 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/p_rules.py tests/brain/test_p_rules.py
git commit -m "feat(p): p_rules OC — O2 vectors per direction (r1 via O2 4:1), c1 nulls, whole-rule simulation with the real bootstrap, no threshold literal"
```

Expected suite: **1787 passed / 1 skipped / 3 xfailed**.

---

### Task 5: `p_measure` — P's write guard, per-item cache files, `PMeasurer`

**Files:**
- Create: `flymon/brain/p_measure.py`
- Test: `tests/brain/test_p_measure.py`

**Interfaces:**
- Consumes: `o_measure.MEASURE_FILES`, `HASHED_FILES`, `OMeasurer`; `o_jobs.arm_job`; `h3_store.MeasureCache`, `canonical`, `canonical_pretty`; `pool_bench` refusals.
- Produces: `ALLOWED_DIR = "results/p/"`, `SUMMARY = "results/summary/p_learning.json"`, `MEASURE_FILES`, `HASHED_FILES`, `guard(path, params_list)`, `write_bytes`, `write_json`, `write_summary_block(path, block, obj, params_list)`, `PCache(root, code, run_id, spec_commit)`, `PMeasurer(pool, pspec, cache)` with `.pspec` and `p_arms(params, items, readout, punish_type) -> rows` (items as in the docstring; rows = `arm_job` rows + `direction`, `x`, `y`, `point`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_p_measure.py
"""Spec P.4 / P.6.5: one atomic cache file per (P, direction, arm, seed) with the code key and the spec commit; rows carry
direction, X, Y and point; a rerun resumes without a pool and an interrupted run computes only the missing items; P
writes only under results/p/ and results/summary/p_learning.json (never O's or N's paths)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from flymon.brain import o_jobs, o_measure
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.p_measure import (HASHED_FILES, MEASURE_FILES, PCache, PMeasurer, write_json,
                                    write_summary_block)
from flymon.brain.p_spec import SPEC

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
    n = SPEC.o.n
    h4 = replace(n.h4, teach_trials=1, teach_present_ms=100.0, teach_gap_ms=20.0,
                 teach_window=replace(n.h4.teach_window, settle_ms=50.0),
                 oracle_window=replace(n.h4.oracle_window, settle_ms=30.0, read_ms=50.0))
    return replace(SPEC, o=replace(SPEC.o, n=replace(n, l=replace(n.l, j=replace(n.l.j, h4=h4)))))


@pytest.fixture
def world(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome(disjoint_kc=True)
    o_jobs._RIG.clear()
    return c, Populations.from_connectome(c)


def _items(seeds=(5,)):
    return [dict(direction=d, x="X" if d == "r1" else "Y", y="Y" if d == "r1" else "X", edit="none", arm=a, punish=pu,
                 plastic=pl, da_zero=dz, odor_x=X if d == "r1" else Y, odor_y=Y if d == "r1" else X, seed=s,
                 point=(0.25, 8.0))
            for d in SPEC.directions for a, pu, pl, dz in SPEC.o.o2_arms for s in seeds]


def test_one_file_per_arm_with_code_key_spec_commit_and_resume(world):
    c, pops = world
    pool = InProcessPool(c, pops)
    rows = PMeasurer(pool, _spec(), PCache("results/p/cache", CODE, "r1", "c0ffee")).p_arms(P, _items(), RO, "PPL105")
    assert [(r["direction"], r["arm"]) for r in rows] == [(d, a) for d in SPEC.directions for a in SPEC.arms]
    assert all(r["point"] == [0.25, 8.0] for r in rows) and pool.rounds == [3, 3]
    files = sorted(Path("results/p/cache/p_arm").glob("*.json"))
    assert len(files) == 6
    d = json.loads(files[0].read_text())
    assert d["spec_commit"] == "c0ffee" and d["code_key"] == "k" * 64 and d["inputs"]["stage"] == "p"
    assert d["inputs"]["reward_type"] == N_SPEC.h3.reward_type and "point" not in d["result"]
    frozen = [r for r in rows if r["arm"] == "frozen"]
    assert all(r["w_post_sha256"] == r["w0_sha256"] for r in frozen)
    again = PMeasurer(NoPool(), _spec(), PCache("results/p/cache", CODE, "r2", "c0ffee")).p_arms(
        P, _items(), RO, "PPL105")
    assert [r["w_post_sha256"] for r in again] == [r["w_post_sha256"] for r in rows]


def test_an_interrupted_run_computes_only_the_missing_items(world):
    c, pops = world
    PMeasurer(InProcessPool(c, pops), _spec(), PCache("results/p/cache", CODE, "r1", "x")).p_arms(
        P, _items()[:2], RO, "PPL105")
    pool = InProcessPool(c, pops)
    PMeasurer(pool, _spec(), PCache("results/p/cache", CODE, "r2", "x")).p_arms(P, _items(), RO, "PPL105")
    assert pool.jobs == 4


def test_writes_only_under_results_p(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(Path("results/p/x/a.json"), {"a": 1}, [P])
    write_summary_block(Path("results/summary/p_learning.json"), "p", {"outcome": "JUDGED"}, [P])
    assert json.loads(Path("results/summary/p_learning.json").read_text()) == {"p": {"outcome": "JUDGED"}}
    for bad in ("results/o/x.json", o_measure.SUMMARY, "results/summary/n_real_odour.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            write_json(Path(bad), {}, [P])
        assert e.value.code == 2


def test_file_lists_extend_os():
    assert set(o_measure.MEASURE_FILES) <= set(MEASURE_FILES) and "flymon/brain/p_measure.py" in MEASURE_FILES
    assert set(MEASURE_FILES) <= set(HASHED_FILES) and set(o_measure.HASHED_FILES) <= set(HASHED_FILES)
    assert {"flymon/brain/p_spec.py", "flymon/brain/p_rules.py", "flymon/brain/p_cli.py", "scripts/run_p.py",
            "scripts/p_oc.py", "flymon/brain/o_rules.py", "flymon/brain/n_rules.py"} <= set(HASHED_FILES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_p_measure.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.p_measure`).

- [ ] **Step 3: Write `p_measure.py`**

```python
"""Storage and measurement of spec appendix P (P.4, P.6.5).
- The write guard: raw rows, caches and reports under results/p/ (git-ignored by `results/*`) and the summary
  results/summary/p_learning.json; nothing else (SystemExit 2). Every write is atomic (tmp + os.replace).
  o_measure's rule with P's paths: o_measure's guard reads its own module constants (results/o/), so P keeps its own
  copy (reading 1) — no O module is edited or patched.
- PCache: o_measure.OCache's entries (one file per item, the code key and the spec commit in each) through P's guard.
- PMeasurer: o_measure.OMeasurer's rounds of one item per worker (NMeasurer._items / _window unchanged) over
  o_jobs.arm_job, one item per (direction, arm, seed); each row also carries its direction, X, Y and operating point.
  A rerun reads every finished item without starting a job; an interrupted run loses at most one round."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from . import o_jobs, o_measure
from .h3_store import MeasureCache, canonical, canonical_pretty
from .o_measure import OMeasurer
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/p/"
SUMMARY = "results/summary/p_learning.json"
# built from O's lists: every O entry stays, plus the P measurement file; nothing re-listed by hand
MEASURE_FILES = tuple(dict.fromkeys(o_measure.MEASURE_FILES + ("flymon/brain/p_measure.py",)))
HASHED_FILES = tuple(dict.fromkeys(o_measure.HASHED_FILES + MEASURE_FILES + (
    "flymon/brain/p_spec.py", "flymon/brain/p_rules.py", "flymon/brain/p_cli.py", "scripts/run_p.py",
    "scripts/p_oc.py")))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: spec P writes only under {ALLOWED_DIR} and {SUMMARY}", file=sys.stderr)
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


class PCache(MeasureCache):
    """MeasureCache with P's guarded write; every entry also holds the code key and the spec commit (O.7.5, P.6.5).
    Copied from o_measure:OCache (its write goes through o_measure's guard)."""

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


class PMeasurer(OMeasurer):
    """OMeasurer's item rounds over P's arms; `spec` is a PSpec (O's numbers through spec.o, N's through spec.o.n)."""

    def __init__(self, pool, spec, cache):
        super().__init__(pool, spec.o, cache)
        self.pspec = spec

    def p_arms(self, params, items, readout, punish_type) -> list:
        """items [{"direction", "x", "y", "edit", "arm", "punish", "plastic", "da_zero", "odor_x", "odor_y", "seed",
        "point"}] -> arm_job rows with direction / x / y / point; N2's training timings and seed rule (spec.o.n).
        The key is the unit (P, direction, arm, seed) plus everything the job reads; the point is part of it."""
        s, h4 = self.spec, self.spec.h4
        common = dict(params=params, readout=dict(readout), punish_type=punish_type, reward_type=s.h3.reward_type,
                      trials=int(h4.teach_trials), present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms,
                      train_settle_ms=h4.teach_window.settle_ms, seed_base=s.train_seed_base,
                      seed_stride=s.train_seed_stride, **self._window())
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     arm=i["arm"], punish=bool(i["punish"]), plastic=bool(i["plastic"]),
                     da_zero=bool(i["da_zero"])) for i in items]
        keys = [dict(j, stage="p", direction=i["direction"], x=i["x"], y=i["y"],
                     point=[float(v) for v in i["point"]]) for j, i in zip(jobs, items)]
        rows = self._items("p_arm", o_jobs.arm_job, params, jobs, keys)
        return [dict(r, direction=i["direction"], x=i["x"], y=i["y"], point=[float(v) for v in i["point"]])
                for r, i in zip(rows, items)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_p_measure.py -q -o addopts=""`
Expected: PASS (4 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/p_measure.py tests/brain/test_p_measure.py
git commit -m "feat(p): p_measure — results/p/ guard, PCache with code key and spec commit, PMeasurer.p_arms over o_jobs.arm_job (direction, x, y, point on rows)"
```

Expected suite: **1791 passed / 1 skipped / 3 xfailed**.

---

### Task 6: `p_cli` — refusals, c₁ from block n1, the O2 source, the stage skeleton with the one rerun

**Files:**
- Create: `flymon/brain/p_cli.py`
- Test: `tests/brain/test_p_cli.py`

**Interfaces:**
- Consumes: Task 1's `SPEC`, `PSpec`, `smoke`; Task 5's `HASHED_FILES`, `MEASURE_FILES`, `PCache`, `PMeasurer`, `ALLOWED_DIR`, `SUMMARY`, `write_json`, `write_bytes`, `write_summary_block`; `o_cli.n_oracle`, `load_c3`, `m0d_sha`, `model_types`, `spec_commit`, `spec_record`, `RUN_OUT`; `o_rules.validity`, `o2_declared`, `o2_key`, `o2_edit_of`, `JUDGED`; `n_rules.INVALID`, `delta`; `l_cli.check_committed`, `refuse`, `run_id`.
- Produces: `NPZ`, `RUN_OUT = "results/p/run"`, `SMOKE_OUT = "results/p/smoke"`, `SMOKE_SUMMARY`, `O2_CACHE`, `HOOK_NAMES`, `hooks(module)`, `in_dir`, `out_allowed`, `git_state`, `code_keys(npz)`, `load_c3(pspec)`, `m0d_sha(pspec)`, `model_types(npz)`, `spec_commit(path)`, `check_committed` (re-export), `c1_source(spec, smoke, committed=, path=) -> ({"pair", "declared", "exact", "o", "c1", "tol"}, None) | (None, why)`, `o2_source(spec, smoke, committed=, summary=, cache=) -> ({"run_id", "measure_key", "z", "rows", "D", "summary", "cache", "n_rows"}, None) | (None, why)`, `make_measurer(ctx)`, `nview(ctx)`, `spec_record(spec)`, `write_report`, `Ctx`, `parser` (adds `--rerun-after-invalid`), `main_stage(name, argv, body, hooks, *, spec=None, require_root=True, doc_help=None, prepare=None, rerun_once=False) -> int` with `prepare(spec, smoke, hooks, doc=, summary=, z=) -> (extra, why)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_p_cli.py
"""Spec P.3 / P.6.4 / P.6.5: c1 comes from block n1's dissimilar-pair o (within 5e-4 of the declared -2.433; the other
pair is not P's business), O2's per-seed source is exactly block o2's run (its measure key, complete, D reproduced),
and the pool is built from the spec."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flymon.brain import p_cli
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.p_spec import SPEC, smoke

OK = lambda *a: None          # noqa: E731  (a committed summary)
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}


def _n1(tmp_path, dis=-2.4325499, sim=-9.0):
    s = tmp_path / "n.json"
    s.write_text(json.dumps({"n1": {"pairs": {"sim": {"o": sim}, "dis": {"o": dis}}}}))
    return s


def test_c1_reads_the_dissimilar_pair_of_block_n1(tmp_path):
    rec, why = p_cli.c1_source(SPEC, False, committed=OK, path=_n1(tmp_path))
    assert why is None and rec["pair"] == "dis" and rec["exact"] == -2.4325499 and rec["declared"] == -2.433
    assert rec["c1"] == pytest.approx(0.25 * 2.4325499) and round(rec["c1"], 3) == 0.608 and rec["tol"] == 0.0005


def test_c1_refuses_a_disagreeing_or_uncommitted_block(tmp_path):
    assert "disagree" in p_cli.c1_source(SPEC, False, committed=OK, path=_n1(tmp_path, dis=-2.4400))[1]
    assert p_cli.c1_source(SPEC, False, committed=lambda *a: "not tracked", path=_n1(tmp_path))[1] == "not tracked"
    assert "no usable block n1" in p_cli.c1_source(SPEC, False, committed=OK, path=tmp_path / "none.json")[1]


def test_c1_in_smoke_without_a_block_uses_the_declared_o(tmp_path):
    rec, why = p_cli.c1_source(smoke(SPEC), True, committed=lambda *a: "never asked", path=tmp_path / "none.json")
    assert why is None and rec["exact"] is None and rec["c1"] == pytest.approx(0.25 * 2.433)


def _probe(seed, A=20.0, P=10):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1,
                wall_s=0.01, steps=1400)


def _o2_world(tmp_path, key="m" * 64, D_shift=0.0, drop=0, other_key_extra=True):
    """A committed-looking block o2 and its cache: punish lowers A_X by 2 + 2.4, plastic by 2 (D = -2.4 with ZU)."""
    cache = tmp_path / "o2_arm"
    cache.mkdir()
    n = 0
    for x, y, _ in O_SPEC.o2_pairs:
        for a, pu, pl, dz in O_SPEC.o2_arms:
            for s in O_SPEC.o2_seeds:
                ax = {"plastic": 2.0, "frozen": 0.0, "punish": 4.4, "da_zero": 0.0}[a]
                res = dict(seed=s, edit="none", arm=a, punish=pu, plastic=pl, da_zero=dz, csc_sha256="sha",
                           pre={"x": _probe(s), "y": _probe(s)}, post={"x": _probe(s, A=20.0 - ax), "y": _probe(s)},
                           w0_sha256="w0", w_post_sha256="w0")
                n += 1
                if n <= drop:
                    continue
                (cache / f"{n:04d}.json").write_text(json.dumps(dict(
                    key=str(n), kind="o2_arm", run_id="run-o2", code_key=key, inputs=dict(x=x, y=y, arm=a, seed=s),
                    result=res)))
    if other_key_extra:                                    # an entry of another run: never read
        (cache / "zzzz.json").write_text(json.dumps(dict(key="z", kind="o2_arm", run_id="old", code_key="o" * 64,
                                                         inputs=dict(x="4:1", y="1:4", arm="punish", seed=1),
                                                         result=dict(seed=1))))
    summ = tmp_path / "o_states.json"
    summ.write_text(json.dumps({"o2": dict(outcome="JUDGED", smoke=False, run_id="run-o2", measure_key="m" * 64,
                                           c3=dict(z=ZU), x={x: dict(stats=dict(D=-2.4 + D_shift))
                                                             for x, _, _ in O_SPEC.o2_pairs})}))
    return summ, cache


def test_o2_source_reads_exactly_block_o2s_run(tmp_path):
    summ, cache = _o2_world(tmp_path)
    rec, why = p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)
    assert why is None and rec["n_rows"] == O_SPEC.n_o2_arms() and rec["run_id"] == "run-o2" and rec["z"] == ZU
    assert rec["D"]["dDL"]["recomputed"] == pytest.approx(-2.4)


def test_o2_source_refuses_a_cache_that_is_not_the_blocks_run(tmp_path):
    summ, cache = _o2_world(tmp_path, D_shift=0.01)
    assert "block o2 says" in p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]


def test_o2_source_refuses_a_missing_entry(tmp_path):
    summ, cache = _o2_world(tmp_path, drop=1)
    assert "no row" in p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]


def test_o2_source_refuses_another_measure_key_or_an_uncommitted_summary(tmp_path):
    summ, cache = _o2_world(tmp_path, key="q" * 64)
    assert "no row" in p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]
    assert p_cli.o2_source(SPEC, False, committed=lambda *a: "dirty", summary=summ, cache=cache)[1] == "dirty"


def test_pool_settings_come_from_the_spec(monkeypatch):
    from flymon.brain import fly_pool
    got = {}
    monkeypatch.setattr(fly_pool, "FlyPool", lambda *a, **k: got.update(k) or SimpleNamespace(close=lambda: None))
    ctx = SimpleNamespace(args=SimpleNamespace(npz="x.npz", workers=3), c3=Params(), spec=smoke(SPEC),
                          out=Path("results/p/x"), key={"key": "k" * 64}, rid="r", spec_commit="c")
    m, pool = p_cli.make_measurer(ctx)
    assert got["timeout_s"] == N_SPEC.pool_timeout_s and got["workers"] == 3
    assert got["punish_type"] == N_SPEC.h3.punish_type and got["reward_type"] == N_SPEC.h3.reward_type
    assert m.cache.root == Path("results/p/x/cache") and m.cache.spec_commit == "c" and m.pspec == smoke(SPEC)
    assert p_cli.parser(None).parse_args([]).workers == N_SPEC.workers


def test_p_cli_refuses_when_a_hashed_file_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(p_cli, "HASHED_FILES", p_cli.HASHED_FILES + ("flymon/brain/no_such_file.py",))
    with pytest.raises(SystemExit) as e:
        p_cli.code_keys(p_cli.NPZ)
    assert e.value.code == 2 and "no_such_file" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_p_cli.py -q -o addopts=""`
Expected: FAIL (`ImportError: cannot import name 'p_cli'`).

- [ ] **Step 3: Write `p_cli.py`**

```python
"""The CLI layer of spec appendix P (P.4, P.6.4, P.6.5). One copy; scripts/p_oc.py and scripts/run_p.py import it.
- Refusals before any pool, in order: another directory; --out outside results/p/; a --smoke run outside
  results/p/smoke/ or a declared run inside it; dirty hashed files (unless --allow-dirty); an unreadable summary;
  outside --smoke, a block already in the real summary (no post-hoc rewrite; P.6.5's one rerun after a technical
  INVALID is the only exception, below); outside --smoke, a spec with no commit; no usable C3 record; the stage's
  `prepare` (c1 from block n1, the OC block before the judgement, O2's per-seed source for the OC).
- P.6.5's rerun: only for a stage that allows it (run_p), only when the real block's outcome is INVALID, only with
  --rerun-after-invalid, only when the code manifest differs from the INVALID run's, and only once: the INVALID block is
  kept as block "<name>_invalid" and the new run's block is final whatever it says.
- The hooks a script imports and passes as `hooks(its own module)`, so its tests patch them on the script.
- O's and N's helpers are imported, never edited: the C3 record, the model ORN types, the Hallem stimuli at a point,
  the spec commit and block n1's oracle check (o_cli.n_oracle, on an OSpec holding only the c1 pair).
- The block, the report and the summary go through p_measure's guard; each carries the run id, the code keys, the git
  state and the spec commit.
Copied from o_cli (o_cli's main_stage is hard-wired to results/o/ and o_measure's guard; reading 1): out_allowed,
git_state, code_keys, make_measurer, spec_record, write_report, Ctx, parser and main_stage, with P's paths and refusals."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import h3_store, n_cli, o_cli, o_measure, p_measure
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import check_committed, refuse, run_id
from .n_rules import INVALID, delta
from .o_rules import JUDGED, o2_declared, o2_edit_of, o2_key, validity
from .o_spec import SPEC as O_SPEC
from .odor_real import DataMismatch
from .p_measure import HASHED_FILES, MEASURE_FILES, PCache, PMeasurer
from .p_spec import SPEC, PSpec, smoke

NPZ = n_cli.NPZ
RUN_OUT = p_measure.ALLOWED_DIR + "run"
SMOKE_OUT = p_measure.ALLOWED_DIR + "smoke"
SMOKE_SUMMARY = f"{SMOKE_OUT}/{Path(p_measure.SUMMARY).name}"
O2_CACHE = o_cli.RUN_OUT + "/cache/o2_arm"            # O2's per-seed rows (git-ignored; P.6.4's source)
HOOK_NAMES = ("check_committed", "code_keys", "git_state", "load_c3", "m0d_sha", "make_measurer", "model_types",
              "spec_commit", "c1_source", "o2_source")


def hooks(module) -> dict:
    return {n: getattr(module, n) for n in HOOK_NAMES}


def in_dir(path, d: str) -> bool:
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    return (rel + "/").startswith(d.rstrip("/") + "/")


def out_allowed(out) -> bool:
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(p_measure.ALLOWED_DIR)


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
    return o_cli.load_c3(spec.o)


def m0d_sha(spec) -> str:
    return o_cli.m0d_sha(spec.o)


def model_types(npz) -> list:
    return o_cli.model_types(npz)


def spec_commit(path) -> str | None:
    return o_cli.spec_commit(path)


def c1_source(spec, smoke: bool, committed=check_committed, path=None) -> tuple:
    """P.3 / P.6.5: (record, None) or (None, refusal). Block n1's exact o of the c1 pair (outside --smoke the N summary
    must be committed) must agree with the declared o within oracle_o_tol (o_cli.n_oracle on an OSpec holding only the
    c1 pair); c1 = c1_frac |exact o|. --smoke without a readable block uses the declared o."""
    o = dataclasses.replace(spec.o, oracle_o=tuple(p for p in spec.o.oracle_o if p[0] == spec.c1_pair))
    rec, why = o_cli.n_oracle(o, smoke, committed=committed, path=path)
    if why:
        return None, why
    exact = None if rec["exact"] is None else rec["exact"][spec.c1_pair]
    used = rec["declared"][spec.c1_pair] if exact is None else exact
    return dict(pair=spec.c1_pair, declared=rec["declared"][spec.c1_pair], exact=exact, o=used,
                c1=spec.c1_frac * abs(used), tol=spec.o.oracle_o_tol), None


def o2_source(spec, smoke: bool, committed=check_committed, summary=None, cache=None) -> tuple:
    """P.6.4's input: (record, None) or (None, refusal). Block o2 of the committed O summary (a judged, declared run);
    the per-seed rows are the O2 cache entries written under that block's measure key; they must cover O2's declared
    (X, arm, seed) grid exactly (o_rules.validity), and D = mean(Δ(dV)[punish] - Δ(dV)[plastic]) recomputed from them
    must equal the block's D for every X within o2_check_tol."""
    summary = Path(summary) if summary is not None else ROOT / o_measure.SUMMARY
    cache = Path(cache) if cache is not None else ROOT / O2_CACHE
    why = committed(summary, ["o2"])
    if why:
        return None, why
    try:
        blk = json.loads(summary.read_text())["o2"]
        key, z, xs = blk["measure_key"], blk["c3"]["z"], blk["x"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, f"no usable block o2 in {summary}: {e!r}"
    if blk.get("outcome") != JUDGED or blk.get("smoke"):
        return None, f"block o2 in {summary} is not a judged declared run"
    rows = []
    for f in sorted(cache.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("kind") == "o2_arm" and d.get("code_key") == key:
            rows.append(dict(d["result"], x=d["inputs"]["x"], y=d["inputs"]["y"]))
    bad = validity(rows, o2_declared(O_SPEC), o2_key, O_SPEC.o2_seeds, o2_edit_of(O_SPEC))
    if bad:
        return None, f"O2's cache {cache} does not hold block o2's run: {'; '.join(bad)}"
    one, three = spec.arms[0], spec.arms[2]
    D = {}
    for x, _, _ in O_SPEC.o2_pairs:
        by = {a: sorted((r for r in rows if r["x"] == x and r["arm"] == a), key=lambda r: int(r["seed"]))
              for a in (one, three)}
        got = float(sum(delta(r3, z) - delta(r1, z) for r1, r3 in zip(by[one], by[three])) / len(by[one]))
        D[x] = dict(recomputed=got, block=float(xs[x]["stats"]["D"]))
        if abs(got - D[x]["block"]) > spec.o2_check_tol:
            return None, f"O2's cache gives D {got} for X = {x}, block o2 says {D[x]['block']}"
    return dict(run_id=blk["run_id"], measure_key=key, z=z, rows=rows, D=D, summary=str(summary),
                cache=str(cache), n_rows=len(rows)), None


def make_measurer(ctx) -> tuple:
    """(PMeasurer, its FlyPool): the one pool construction of P's CLIs. The caller closes the pool."""
    from . import fly_pool
    h3 = ctx.spec.o.n.h3
    pool = fly_pool.FlyPool(ctx.args.npz, ctx.c3, flies=[{}] * ctx.args.workers, workers=ctx.args.workers,
                            punish_type=h3.punish_type, reward_type=h3.reward_type,
                            timeout_s=ctx.spec.o.pool_timeout_s)
    return PMeasurer(pool, ctx.spec, PCache(ctx.out / "cache", ctx.key, ctx.rid, ctx.spec_commit)), pool


def nview(ctx):
    """The context as n_cli's stimulus helpers read it (spec = N's)."""
    return SimpleNamespace(spec=ctx.spec.o.n, c3=ctx.c3, types=ctx.types)


def spec_record(spec) -> dict:
    return dict({f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "o"},
                o=o_cli.spec_record(spec.o))


def write_report(out, rid: str, name: str, res: dict, params_list) -> Path:
    base = Path(out) / "runs" / f"{rid}-{name}"
    path = p_measure.write_json(Path(f"{base}.json"), res, params_list)
    md = f"# P {name} — {rid}\n\n{res.get('sentence', '')}\n\noutcome: `{res.get('outcome')}`\n"
    p_measure.write_bytes(Path(f"{base}.md"), md.encode(), params_list)
    return path


@dataclass
class Ctx:
    args: argparse.Namespace
    spec: PSpec
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
    ap.add_argument("--workers", type=int, default=SPEC.o.workers)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--rerun-after-invalid", action="store_true")
    return ap


def main_stage(name: str, argv, body, hooks: dict, *, spec=None, require_root: bool = True,
               doc_help: str | None = None, prepare=None, rerun_once: bool = False) -> int:
    """The refusals (module docstring), then body(ctx) -> (res, exit code); a DataMismatch inside the body is a
    refusal. Any other error propagates: nothing is written. The report goes under <out>/runs/ and block `name` into the
    summary (a --smoke run: its smoke summary). `prepare(spec, smoke, hooks, doc=, summary=, z=) -> (extra, refusal)`
    runs after C3 is loaded and before any pool; its extra is ctx.extra."""
    a = parser(doc_help).parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or (SMOKE_OUT if a.smoke else RUN_OUT))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {p_measure.ALLOWED_DIR} of the repository root")
    summary = Path(a.summary or (SMOKE_SUMMARY if a.smoke else p_measure.SUMMARY))
    if a.smoke and not (in_dir(out, SMOKE_OUT) and in_dir(summary, SMOKE_OUT)):
        return refuse(f"a --smoke run reads and writes only under {SMOKE_OUT}/ (cache and summary), not {out} / "
                      f"{summary}")
    if not a.smoke and (in_dir(out, SMOKE_OUT) or in_dir(summary, SMOKE_OUT)):
        return refuse(f"a declared run never reads or writes under the smoke root {SMOKE_OUT}/ ({out} / {summary})")
    git = hooks["git_state"](HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    try:
        doc = json.loads(summary.read_text()) if summary.exists() else {}
    except (OSError, ValueError) as e:
        return refuse(f"no usable summary at {summary}: {e}")
    prior = None
    if not a.smoke and name in doc:
        blk = doc[name]
        if not (rerun_once and isinstance(blk, dict) and blk.get("outcome") == INVALID):
            return refuse(f"{summary} already holds block {name}: each verdict goes to the user with no post-hoc "
                          f"change (P.4, P.6.5), so it is not rewritten")
        if f"{name}_invalid" in doc:
            return refuse(f"{summary} already holds block {name}_invalid: P.6.5 allows one rerun only")
        if not a.rerun_after_invalid:
            return refuse(f"block {name} is INVALID: after fixing the code, rerun once with --rerun-after-invalid "
                          f"(P.6.5)")
        prior = blk
    elif a.rerun_after_invalid:
        return refuse(f"--rerun-after-invalid needs an INVALID block {name} in the real summary {summary}")
    commit = hooks["spec_commit"](spec.o.spec_path)
    if commit is None and not a.smoke:
        return refuse(f"the spec {spec.o.spec_path} has no commit: commit appendix P before a real run")
    key, manifest = hooks["code_keys"](a.npz)
    if prior is not None and (prior.get("code") or {}).get("key") == manifest["key"]:
        return refuse(f"the code manifest equals the INVALID run's ({manifest['key'][:12]}): fix the code first (P.6.5)")
    try:
        c3, readout, z, _ = hooks["load_c3"](spec)
    except ValueError as e:
        return refuse(str(e))
    extra = None
    if prepare is not None:
        extra, why = prepare(spec, a.smoke, hooks, doc=doc, summary=summary, z=z)
        if why:
            return refuse(why)
    rid = run_id()
    ctx = Ctx(args=a, spec=spec, smoke=a.smoke, rid=rid, out=out, key=key, c3=c3, readout=dict(readout), z=z,
              types=hooks["model_types"](a.npz), hooks=hooks, spec_commit=commit, extra=extra)
    t0 = time.time()
    try:
        res, code = body(ctx)
    except DataMismatch as e:
        return refuse(f"the Hallem data do not match their pins: {e}")
    res = dict(res, name=name, run_id=rid, smoke=a.smoke, measure_key=key["key"], code=manifest, git=git,
               spec_commit=commit, inputs=dict(m0d_sha256=hooks["m0d_sha"](spec), data_sha256=spec.o.n.sha_pins(),
                                               state_p_min=spec.o.n.state_p_min),
               c3=dict(readout=readout, z=z), spec=spec_record(spec), argv=list(argv or []), wall_s=time.time() - t0,
               rerun_of=None if prior is None else prior.get("run_id"))
    report = write_report(out, rid, name, res, [c3])
    if prior is not None:
        p_measure.write_summary_block(summary, f"{name}_invalid", prior, [c3])
    p_measure.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)),
                                  [c3])
    print(res.get("sentence", ""))
    return code
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_p_cli.py -q -o addopts=""`
Expected: PASS (9 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/p_cli.py tests/brain/test_p_cli.py
git commit -m "feat(p): p_cli — refusals, c1 from block n1's dis o (5e-4), O2 source from block o2 + its cache (D reproduced), stage skeleton with P.6.5's single rerun"
```

Expected suite: **1800 passed / 1 skipped / 3 xfailed**.

---

### Task 7: `scripts/run_p.py`, `scripts/p_oc.py`

**Files:**
- Create: `scripts/run_p.py`, `scripts/p_oc.py`
- Test: `tests/test_run_p.py`

**Interfaces:**
- Consumes: Task 6's `main_stage`, `hooks`, the hook functions, `nview`; Task 2–4's `p_judge`, `sentence`, `JUDGED`, `o2_vectors`, `oc_run`, `oc_sentence`; `n_cli.stimuli_at`, `drives`; `o_spec.SPEC`.
- Produces: `run_p.items_p(spec, st)`, `run_p.prepare`, `run_p.body`, `run_p.main(argv=None, spec=None, require_root=True) -> int`; `p_oc.LIMITATION`, `p_oc.prepare`, `p_oc.body`, `p_oc.main(...)`. Blocks `p` (and `p_invalid` after a rerun) and `oc` in `results/summary/p_learning.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_p.py
"""Spec P.4 / P.6.4 / P.6.5: the P CLIs refuse before any pool on another root, an --out outside results/p/, dirty hashed
files, a block already in the real summary, an uncommitted spec, a disagreeing block n1; run_p refuses without a
committed block oc, asks for every (direction, arm, seed) once with the declared pair, flags and point, and writes its
block (judged exit 0, INVALID exit 5); P.6.5's single rerun after a technical INVALID; p_oc records block oc from O2's
rows without a pool. Every P hashed file exists."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flymon.brain import p_cli, p_measure
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.odor_real import load_table
from flymon.brain.p_measure import HASHED_FILES
from flymon.brain.p_rules import DIRECTION_DEPENDENT, LEARNS_CONFIRMATORY, OC_RECORDED
from flymon.brain.p_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[1]
SM = smoke(SPEC)
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])      # noqa: E731
KEY = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}
TYPES = sorted({"ORN_" + p for v in load_table(ROOT / N_SPEC.data_dir, N_SPEC.sha_pins()).glomeruli.values() for p in v})
C1 = dict(pair="dis", declared=-2.433, exact=-2.4325, o=-2.4325, c1=0.25 * 2.4325, tol=0.0005)
STAGES = ["run_p", "p_oc"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def _probe(seed, A=20.0, P=10):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1,
                wall_s=0.01, steps=1400)


class FakePool:
    def close(self):
        pass


class FakeMeasurer:
    """The punish arm lowers A_X by 2 + effect[direction] + jitter, the plastic arm by 2, the frozen arm not at all."""

    def __init__(self, effect=None, frozen_moves=False):
        self.effect = effect or {"r1": 2.4, "r2": 2.4}
        self.frozen_moves, self.calls = frozen_moves, []

    def p_arms(self, params, items, readout, punish_type):
        self.calls.append((sorted({(i["direction"], i["x"], i["y"]) for i in items}),
                           sorted({(i["arm"], i["punish"], i["plastic"], i["da_zero"]) for i in items}),
                           sorted({tuple(i["point"]) for i in items}), len(items)))
        out = []
        for it in items:
            i = it["seed"] - SM.seeds[0]
            ax = {"plastic": 2.0, "frozen": 0.0, "punish": 2.0 + self.effect[it["direction"]] + 0.5 * (i % 2)}[it["arm"]]
            moved = it["arm"] != "frozen" or self.frozen_moves
            out.append(dict(seed=it["seed"], edit=it["edit"], arm=it["arm"], punish=it["punish"],
                            plastic=it["plastic"], da_zero=it["da_zero"], csc_sha256="sha-none",
                            pre={"x": _probe(it["seed"]), "y": _probe(it["seed"])},
                            post={"x": _probe(it["seed"], A=20.0 - ax), "y": _probe(it["seed"])},
                            weights_frac=0.9, weights_frac_A=0.8, weights_frac_P=1.0, w0_sha256="w0",
                            w_post_sha256="w1" if moved else "w0", da_integral={"PPL105": 1.0}, wall_s=1.0,
                            direction=it["direction"], x=it["x"], y=it["y"], point=list(it["point"])))
        return out


def _o2_source(spec, smoke):
    """Two O2-like X with 32 seeds: punish lowers A_X by 2 + 2.4 + jitter, plastic by 2."""
    rows = []
    for x, y, _ in O_SPEC.o2_pairs:
        for a in ("plastic", "punish"):
            for i, s in enumerate(O_SPEC.o2_seeds):
                ax = 2.0 if a == "plastic" else 4.4 + 0.3 * (i % 3)
                rows.append(dict(x=x, y=y, arm=a, seed=s, pre={"x": _probe(s), "y": _probe(s)},
                                 post={"x": _probe(s, A=20.0 - ax), "y": _probe(s)}))
    return dict(run_id="run-o2", measure_key="q" * 64, z=ZU, rows=rows, D={}, summary="s", cache="c",
                n_rows=len(rows)), None


def _patch(mod, monkeypatch, measurer=None, commit="c0ffee", c1=(C1, None), committed=None, key=KEY):
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "code_keys", lambda npz: key)
    monkeypatch.setattr(mod, "m0d_sha", lambda spec: "s" * 64)
    monkeypatch.setattr(mod, "load_c3", lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, ZU, None))
    monkeypatch.setattr(mod, "model_types", lambda npz: TYPES)
    monkeypatch.setattr(mod, "spec_commit", lambda path: commit)
    monkeypatch.setattr(mod, "c1_source", lambda spec, smoke: c1)
    monkeypatch.setattr(mod, "o2_source", _o2_source)
    monkeypatch.setattr(mod, "check_committed", committed or (lambda summary, blocks: None))
    monkeypatch.setattr(mod, "make_measurer",
                        lambda ctx: (measurer, FakePool()) if measurer else pytest.fail("no pool may start here"))
    monkeypatch.setattr(p_cli, "out_allowed", lambda out: True)


def _real_summary(tmp_path, doc):
    p = tmp_path / p_measure.SUMMARY
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def test_every_hashed_file_exists():
    assert [f for f in HASHED_FILES if not (ROOT / f).exists()] == []


@pytest.mark.parametrize("name", STAGES)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_out_outside_results_p(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/o/x"]) == 2 and "results/p/" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_dirty_hashed_files(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/p_rules.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


@pytest.mark.parametrize("name, block", [("run_p", "p"), ("p_oc", "oc")])
def test_never_rewrites_a_judged_block(name, block, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}, block: {"outcome": "JUDGED"}})
    assert mod.main([], require_root=False) == 2 and "post-hoc" in capsys.readouterr().err
    assert mod.main(["--rerun-after-invalid"], require_root=False) == 2


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_uncommitted_spec(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, commit=None)
    if name == "run_p":
        _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}})
    assert mod.main([], require_root=False) == 2 and "spec" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_a_disagreeing_block_n1(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, c1=(None, "block n1's oracle effects disagree"))
    assert mod.main(["--smoke"], require_root=False) == 2 and "disagree" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_run_p_refuses_without_a_committed_oc_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    assert mod.main([], require_root=False) == 2 and "block oc" in capsys.readouterr().err
    _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED, "run_id": "oc1"}})
    _patch(mod, monkeypatch, committed=lambda summary, blocks: "has uncommitted changes")
    assert mod.main([], require_root=False) == 2 and "uncommitted" in capsys.readouterr().err


def test_run_p_smoke_asks_for_every_unit_once_and_judges(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    assert fm.calls == [(sorted((d, x, y) for d, (x, y) in SPEC.pairs().items()),
                         sorted(SPEC.o.o2_arms), [(0.25, 8.0)], SM.n_arms())]
    doc = json.loads(Path(p_cli.SMOKE_SUMMARY).read_text())["p"]
    assert doc["outcome"] == "JUDGED" and doc["label"] == LEARNS_CONFIRMATORY and doc["smoke"] is True
    assert doc["c1"] == pytest.approx(C1["c1"]) and doc["c1_source"] == C1 and doc["spec_commit"] == "c0ffee"
    assert doc["point"] == {"g": 0.25, "c_delta": 8.0} and Path(doc["report"]).exists()
    assert "P: LEARNS_CONFIRMATORY" in capsys.readouterr().out


def test_run_p_real_run_needs_oc_first_and_writes_block_p(tmp_path, monkeypatch):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer(effect={"r1": 2.4, "r2": 0.0})
    _patch(mod, monkeypatch, fm)
    summ = _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED, "run_id": "oc1"}})
    assert mod.main([], require_root=False, spec=SM) == 0          # a declared run at the fake measurer's scale
    doc = json.loads(summ.read_text())
    assert doc["p"]["label"] == DIRECTION_DEPENDENT and doc["p"]["oc_run_id"] == "oc1" and doc["oc"]["run_id"] == "oc1"


def test_an_invalid_block_is_rerun_once_with_fixed_code(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(frozen_moves=True))
    summ = _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED, "run_id": "oc1"}})
    assert mod.main([], require_root=False, spec=SM) == 5
    first = json.loads(summ.read_text())["p"]
    assert first["outcome"] == "INVALID" and "plumbing" in first["reasons"][0]
    assert mod.main([], require_root=False, spec=SM) == 2 and "--rerun-after-invalid" in capsys.readouterr().err
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 2       # same code manifest
    assert "fix the code" in capsys.readouterr().err
    fixed = ({"key": "k" * 64, "files": {}}, {"key": "n" * 64, "files": {}})
    _patch(mod, monkeypatch, FakeMeasurer(), key=fixed)
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 0
    doc = json.loads(summ.read_text())
    assert doc["p_invalid"]["run_id"] == first["run_id"] and doc["p"]["rerun_of"] == first["run_id"]
    assert doc["p"]["label"] == LEARNS_CONFIRMATORY
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 2       # final now


def test_a_second_invalid_is_final(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(frozen_moves=True))
    summ = _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}, "p": {"outcome": "INVALID", "run_id": "a",
                                                                            "code": {"key": "old"}},
                                    "p_invalid": {"outcome": "INVALID", "run_id": "b"}})
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 2
    assert "one rerun only" in capsys.readouterr().err and json.loads(summ.read_text())["p"]["run_id"] == "a"


def test_rerun_flag_without_an_invalid_block_is_refused(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}})
    assert mod.main(["--rerun-after-invalid"], require_root=False) == 2 and "needs an INVALID" in capsys.readouterr().err


def test_p_oc_smoke_records_block_oc_without_a_pool(tmp_path, monkeypatch, capsys):
    mod = _script("p_oc")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)                                   # no measurer: a pool start fails the test
    assert mod.main(["--smoke"], require_root=False) == 0
    doc = json.loads(Path(p_cli.SMOKE_SUMMARY).read_text())["oc"]
    assert doc["outcome"] == OC_RECORDED and doc["draws"] == SM.oc_draws and doc["n"] == SM.o.o2_n_seeds
    assert set(doc["scenarios"]) == {"observed", "null_both", "null_r1", "null_r2"}
    assert doc["o2_source"]["run_id"] == "run-o2" and "Y = 1:4" in doc["limitation"]
    assert "P OC:" in capsys.readouterr().out


def test_p_oc_refuses_when_c3_z_differs_from_o2s(tmp_path, monkeypatch, capsys):
    mod = _script("p_oc")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "o2_source", lambda spec, smoke: (dict(_o2_source(spec, smoke)[0],
                                                                    z={"A": [1.0, 1.0], "P": [0.0, 1.0]}), None))
    assert mod.main(["--smoke"], require_root=False) == 2 and "differs from O2's" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_smoke_and_real_runs_use_separate_roots(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    assert mod.main(["--smoke", "--out", p_cli.RUN_OUT], require_root=False) == 2
    assert mod.main(["--out", p_cli.SMOKE_OUT], require_root=False) == 2
    assert mod.main(["--smoke", "--summary", p_measure.SUMMARY], require_root=False) == 2
    assert mod.main(["--summary", p_cli.SMOKE_SUMMARY], require_root=False) == 2
    assert capsys.readouterr().err.count("smoke") >= 4 and not (tmp_path / "results").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_run_p.py -q -o addopts=""`
Expected: FAIL (`FileNotFoundError` for `scripts/run_p.py`).

- [ ] **Step 3: Write `scripts/run_p.py`**

```python
#!/usr/bin/env python3
"""Spec P.2 / P.6.2 / P.6.5 (the first learning judgement: punishment learning on the real-odour dissimilar pair, both
directions, new seeds 23_000_000 + i; a confirmatory test declared after O2, P.0).

Steps:
1. Refuse (outside --smoke) unless the committed P summary holds block oc (P.6.4: the operating characteristic is
   recorded before the judgement run), and unless block n1's dissimilar-pair o agrees with -2.433 within 5e-4 (c1).
2. At the operating point, for each direction (r1 X = 4:1 / Y = δ-DL, r2 X = δ-DL / Y = 4:1) and each arm (plastic ①,
   frozen ②, punish ③) on the P seeds: o_jobs.arm_job. One cache file per (P, direction, arm, seed) under
   results/p/run/cache/p_arm/: rerunning the command resumes.
3. p_rules.p_judge: the gate and arm ②'s plumbing (INVALID), the per-direction rule and the label, the records.

    uv run python scripts/run_p.py                                        # the controller only (R1; resumable)
    uv run python scripts/run_p.py --rerun-after-invalid                  # P.6.5: once, after a technical INVALID
    uv run python scripts/run_p.py --smoke --allow-dirty --workers 4

Exit 0 for a judged stage (any label), 5 for INVALID (block written; reported to the user), 2 for a refusal."""
from __future__ import annotations

import sys

from flymon.brain import n_cli
from flymon.brain.p_cli import (c1_source, check_committed, code_keys, git_state, hooks, load_c3,  # noqa: F401
                                m0d_sha, main_stage, make_measurer, model_types, nview, o2_source, spec_commit)
from flymon.brain.p_rules import JUDGED, p_judge, sentence


def items_p(spec, st: dict) -> list:
    """Every declared (direction, arm, seed) on the APL-on rig at the operating point, grouped by direction then arm."""
    point = tuple(float(v) for v in spec.o.o2_point)
    flags = spec.flags()
    return [dict(direction=d, x=x, y=y, edit=spec.o.on_edit, arm=a, punish=flags[a][0], plastic=flags[a][1],
                 da_zero=flags[a][2], odor_x=st[x]["odor"], odor_y=st[y]["odor"], seed=int(s), point=point)
            for d, (x, y) in spec.pairs().items() for a in spec.arms for s in spec.seeds]


def prepare(spec, smoke: bool, hk: dict, doc: dict, summary, z) -> tuple:
    """P.6.4 before P.6.2 (outside --smoke: block oc present and committed), then c1 from block n1 (P.6.5)."""
    if not smoke:
        if "oc" not in doc:
            return None, f"{summary} holds no block oc: record the operating characteristic first (P.6.4)"
        why = hk["check_committed"](summary, ["oc"])
        if why:
            return None, why
    c1, why = hk["c1_source"](spec, smoke)
    if why:
        return None, why
    return dict(c1=c1, oc_run_id=(doc.get("oc") or {}).get("run_id")), None


def body(ctx) -> tuple:
    spec, nv = ctx.spec, nview(ctx)
    g, c = spec.o.o2_point
    names = list(dict.fromkeys(s for x, y in spec.pairs().values() for s in (x, y)))
    st = n_cli.stimuli_at(nv, names, g, c)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.p_arms(ctx.c3, items_p(spec, st), ctx.readout, spec.o.n.h3.punish_type)
    finally:
        pool.close()
    res = p_judge(rows, ctx.z, ctx.extra["c1"]["c1"], spec)        # z: C3's frozen z, as O2's
    res.update(c1_source=ctx.extra["c1"], oc_run_id=ctx.extra["oc_run_id"], drives=n_cli.drives(st),
               point=dict(g=float(g), c_delta=float(c)), n_arms=len(rows))
    return dict(res, sentence=sentence(res)), (0 if res["outcome"] == JUDGED else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("p", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__, prepare=prepare, rerun_once=True)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/p_oc.py`**

```python
#!/usr/bin/env python3
"""Spec P.6.4 (the operating characteristic of P.6.2's whole rule, recorded before the judgement run; n stays 32
whatever it says).

Steps:
1. Refuse unless block o2 of the committed O summary is a judged declared run whose per-seed rows are in O2's cache
   (results/o/run/cache/o2_arm/, written under the block's measure key; D reproduced for both X), unless C3's z equals
   O2's, and unless block n1's dissimilar-pair o agrees with -2.433 within 5e-4 (c1).
2. Map the directions onto O2's rows: r1 (X = 4:1) <- O2's X = 4:1 (its Y was 1:4: the stated limitation), r2 (X = δ-DL)
   <- O2's X = δ-DL (Y = 4:1, the same pair).
3. p_rules.oc_run: oc_draws experiments of 32 seeds resampled from O2's, each judged by P.6.2's rule with the real
   bootstrap, under the observed effects and under c1 nulls (both directions; each direction alone). No pool is built.

    uv run python scripts/p_oc.py                                          # the controller only (before R1)
    uv run python scripts/p_oc.py --smoke --allow-dirty

Exit 0 when recorded, 2 for a refusal."""
from __future__ import annotations

import json
import sys

from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.p_cli import (c1_source, check_committed, code_keys, git_state, hooks, load_c3,  # noqa: F401
                                m0d_sha, main_stage, make_measurer, model_types, o2_source, spec_commit)
from flymon.brain.p_rules import o2_vectors, oc_run, oc_sentence

LIMITATION = ("P.6.4: r1(X = 4:1, Y = δ-DL)은 처벌 조건화로 돈 적이 없어 O2의 X = 4:1 행(Y = 1:4)으로 대신했다 — "
              "r1의 운영 특성은 근사다. n은 32로 고정하며 이 결과로 바꾸지 않는다.")


def prepare(spec, smoke: bool, hk: dict, doc: dict, summary, z) -> tuple:
    c1, why = hk["c1_source"](spec, smoke)
    if why:
        return None, why
    src, why = hk["o2_source"](spec, smoke)
    if why:
        return None, why
    if json.loads(json.dumps(z)) != json.loads(json.dumps(src["z"])):
        return None, f"C3's z {z} differs from O2's {src['z']}: O2's rows are not in P's units"
    return dict(c1=c1, o2=src), None


def body(ctx) -> tuple:
    spec, ex = ctx.spec, ctx.extra
    vec = o2_vectors(ex["o2"]["rows"], ex["o2"]["z"], spec, O_SPEC)
    res = oc_run(vec, ex["c1"]["c1"], spec)
    res.update(c1_source=ex["c1"], limitation=LIMITATION,
               o2_source={k: ex["o2"][k] for k in ("run_id", "measure_key", "D", "summary", "cache", "n_rows")})
    return dict(res, sentence=oc_sentence(res)), 0


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("oc", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__, prepare=prepare)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_p.py -q -o addopts=""`
Expected: PASS (23 tests).

- [ ] **Step 6: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add scripts/run_p.py scripts/p_oc.py tests/test_run_p.py
git commit -m "feat(p): run_p + p_oc — OC block before the judgement, every (direction, arm, seed) once, block p with c1 source, single rerun after a technical INVALID"
```

Expected suite after Task 7: 1742 + 8 + 27 + 6 + 4 + 4 + 9 + 23 = **1823 passed / 1 skipped / 3 xfailed**.

---

## Self-review (done while writing)

- **Spec coverage.**
  - P.0 (confirmatory disclosure) is in `sentence` and the result paragraph (Runs). P.1 / P.6.1 (narrowed claim, associativity a consequence of the rule, no unpaired arm) are in `sentence`; no unpaired arm is built.
  - P.2: directions, arms and the shared `arm_job` procedure are Task 1 (`pairs()`, `flags()`, `P_O`) and Task 5 (`p_arms`); seeds and the smoke block are Task 1; the readout unit is Task 2 (`v_of`, `n_rules.delta`).
  - P.3 / P.6.2: c₁ in Task 6 (`c1_source`); the rule, labels and joint bootstrap in Task 2; the averaged ℓ record in Task 2; P.3's records in Task 3.
  - P.6.3: Task 3 (`flip_reanalysis` without suffix, `strata`).
  - P.6.4: Task 4 (`oc_run`), Task 6 (`o2_source`), Task 7 (`p_oc.py`, `run_p` refusing without block `oc`), Runs (OC before R1).
  - P.6.5: gate in Task 2; input pins (pair, point, flags, c₁ within 5·10⁻⁴) in Tasks 2 and 6; atomic per-item files with code key and spec commit in Task 5; the single rerun in Tasks 6 and 7.
  - P.6.6: Task 1 (every `*_spec.py`, training images for any t, s ∉ {22 000, 22 009}).
  - P.6.7: Task 1 (`replace`, four fields), Task 2 (mutation tests: one direction zeroed, one reversed, ②'s weights moved).
  - P.4: file set (reading 1), summary / raw paths (Task 5), order (Runs). P.5 needs no code.
- **Placeholders:** none. Every step has its code or command. All code in this plan was run in a scratch worktree at `0cc9e5a` before it was written here: the task-by-task test counts above are the counts observed, and `p_oc.py --smoke` ran end to end on the real O2 source (it reproduced both committed D values and c₁ = 0.6081…).
- **Names across tasks:**
  - `PMeasurer.p_arms` (Task 5) is consumed by `run_p.body` (Task 7); its row keys `direction`, `x`, `y`, `point` are read by `p_key` / `expectations` (Task 2).
  - `p_judge(rows, z, c1, spec)` and `sentence(res)` (Tasks 2–3) are consumed by `run_p.body`; `o2_vectors(rows, z, spec, o_spec)`, `oc_run(vec, c1, spec)`, `oc_sentence(res)` (Task 4) by `p_oc.body`.
  - `c1_source` / `o2_source` (Task 6) return the dicts `run_p.prepare` / `p_oc.prepare` read (`["c1"]`, `["rows"]`, `["z"]`, `["run_id"]`, …).
  - `p_cli.SMOKE_SUMMARY`, `RUN_OUT`, `SMOKE_OUT` and `p_measure.SUMMARY` are used by the tests.

## Runs (controller)

Every run happens from the repository root, in the background (`run_in_background`), after the full suite passes at the final commit. Subagents never run these. No rule or number changes after any run (P.4, P.6.4: n stays 32).

- **R0 smoke** (seed block 23_009_xxx; writes only `results/p/smoke/`):
  1. `uv run python scripts/p_oc.py --smoke --allow-dirty` — seconds, no pool; reads the committed block `o2` and `results/o/run/cache/o2_arm/` read-only. Check `o2_source.D` reproduces both X (−2.0750…, −2.3977…), `c1_source.exact` ≈ −2.43255 and `c1` ≈ 0.6081.
  2. `uv run python scripts/run_p.py --smoke --allow-dirty --workers 4` — 2 directions × 3 arms × 4 seeds = 24 arms. Check in `results/p/smoke/p_learning.json`: no `INVALID`; `records.<d>.frozen_probes_identical` true for both directions; `records.<d>.strata.pre_identical` true; every row's pair / point as declared (the gate passed); `spec_commit` present.
  3. Record the wall clock per arm (O2: 256 arms ≈ 15 min at 16 workers → expect R1 ≈ 11 min). Rerun the smoke command to confirm the resume: every item a cache hit, no job.
- **OC record (P.6.4), before R1:**
  1. `uv run python scripts/p_oc.py` — `oc_draws` 2 000 experiments × 4 scenarios with the 10 000-draw bootstrap (≈ 10–20 s, no pool).
  2. Commit block `oc`: `git add results/summary/p_learning.json && git commit -m "results(p): OC record — LEARNS_CONFIRMATORY probability at n = 32 (observed / c1 nulls), r1 via O2 4:1 (run <rid>)"`.
  3. Do not change n or any rule whatever it says (P.6.4). Expect a single-direction c₁ null near 0.5 (reading 17); report that number with the verdict, not as a reason to stop.
- **R1 judgement run (one run):**
  1. `uv run python scripts/run_p.py` — 192 arms (~11 min at 16 workers). It refuses unless block `oc` is committed and block n1's dis o agrees with −2.433 within 5·10⁻⁴.
  2. After an interruption, rerun the same command; it resumes from `results/p/run/cache/p_arm/`.
  3. Exit 0 (a label) or 5 (`INVALID`). On `INVALID` from a code or infrastructure defect (P.6.5): keep block `p` as it is (commit it), fix the code in a new reviewed task, commit, then `uv run python scripts/run_p.py --rerun-after-invalid` exactly once; that block is final whatever it says.
  4. Commit `results/summary/p_learning.json` (`results/summary/` is tracked; `results/p/` stays ignored): `git commit -m "results(p): P judgement — <LABEL> (r1 <outcome>, r2 <outcome>; run <rid>)"`.
  5. Write the result section into the spec after P.6 (`### P.7 결과`), citing the run id and the OC run id: the label; per direction ℓ_r [CI], s_r [CI], t_r, (i)/(ii)/(iii); c₁ (exact); the averaged ℓ [CI] (record); the flip re-analysis and pre-state strata (records); ΔA_X / ΔP_X / ΔA_Y / ΔP_Y and ①'s non-associative change; the OC numbers; the P.0 sentence ("새 시드에서의 확인 시험 — 통과는 발견이 아니라 재현"); P.6.1's note (associativity is the rule's consequence); P.1's scope (C3, this pair, this procedure; not the Pokémon claim). Commit it: `git commit -m "docs(spec): P.7 — P result: <LABEL> ..."`.
  6. Add the README ledger lines for appendix P, Korean (next to the O entry, around line 173) and English (around line 254), in the existing format: what was judged, the label, the two directions' ℓ with CIs, c₁, that it is a confirmatory replication on C3. Commit: `git commit -m "docs(readme): appendix P ledger (ko/en) — <LABEL>"`.
  7. Report to the user with no post-hoc change (P.4). P.5's next step depends on the label (`LEARNS_CONFIRMATORY` → join the encoder-redesign track, declared together with that track's spec; otherwise the user decides).
