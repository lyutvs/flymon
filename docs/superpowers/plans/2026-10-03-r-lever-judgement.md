# Spec R — Lever APL→MBON05 Removal, One M2 Judgement: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix R needs (as overridden by R.9), and nothing else:
- a **no-edit reproduction gate** (R.9.6) before smoke. R's new edit-capable jobs, run without the edit, must reproduce three things bit for bit:
  - encoder ③'s KC activity entries (k2-norm, s 1.0) for 3 odours × the 8 strength seeds;
  - P's committed `p_arm` entries for 3 seeds × 2 directions × 3 arms;
  - one Q0 oracle file through R's oracle wrapper.
- **smoke** on new seeds 25_008_xxx / 25_009_xxx, even (b) pairs only. The judgement set is never used here.
- the **operating characteristic** (R.6, R.9.5), recorded before any gate.
- **gate ①** validity (R.9.2): 112 calibration odours × 24_002_000+i under the lever, s 1.0. Pass needs KC median ∈ [0.03, 0.15] ∧ ORN cap. The original encoder qualification is a record.
- **gate ②** P replication with the lever: seeds 25_000_000+i (i < 32), c₁ read from block n1. `LEARNS_CONFIRMATORY` is required.
- **gate ③** even (R.9.3, R.9.6): L and C in the same run (E0 as a record), H.4 seeds. Checks are c_even = 7 first, then testable_b ≥ 11.
- the **judgement** on the encoder's judgement set: L, C and E0, one condition per run, seeds 24_100_xxx, digests re-checked.
- **seal** (R.9.7): pre-read validity, the raw manifest committed, an archive copy in `~/flymon-archive/r/`.
- **judge**: R.3's bands in fixed order with R.9.4 / R.9.5 (G_fail before F_a). Closing sentences are fixed in code before any measurement. Records follow R.6 as amended.

R is **one M2 judgement**. Its STOP labels (`STOP_STRENGTH_LEVER`, `STOP_PUNISH_BROKEN`, `STOP_EVEN_LOW_LEVER`) and its bands go to the user unchanged. The judgement set is used once (R.5).

**Architecture:**
- New files only: `flymon/brain/r_*.py`, `scripts/run_r.py`, and tests. The one exception is a single `MODULES` line in `tests/brain/test_p_spec.py` (Reading 1).
- Everything else is imported and never edited: the encoder track (`flymon/agent/e_*.py`, `encode_grid.py`), Q (`q_*`), P (`p_*`), O, N, `h4_*`, `k_jobs`, `fly_pool`, `engine_cpu`.
- Modules:
  - `r_spec`: every R number (class `LeverSpec`).
  - `r_pairs`: calibration odours, even rows, judgement rows (digest-checked), the reference files of the reproduction gate.
  - `r_jobs`: three worker jobs on Q's edit-capable rig `q_jobs.q_rig`:
    - `kc_activity_job` (k_jobs.activity_job + edit);
    - `r_arm_job` (o_jobs.arm_job, copied, + edit);
    - `r_oracle_job` (`q_jobs.q_oracle_job` called unchanged, plus a per-cell MBON05 naive probe).
  - `r_store`: the guard, atomic writes, `RCache`, manifest loading, the archive copy.
  - `r_measure`: the measurer, one cache entry per unit, with resume.
  - `r_records`: per-pair values, raw checks, R.6 records, pre-read validity. No decisions.
  - `r_rules`: gates, G_fail, bands, sentences, operating characteristic.
  - `r_runner`: the stage chain.
- `scripts/run_r.py` is the CLI.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`, `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **R** (R.0–R.8) and its red-team amendment **R.9**, which wins wherever the two differ. Reused rules:
- `docs/superpowers/specs/2026-10-01-encoder-redesign-design.md`:
  - 4.3: the strength rules;
  - 4.4: the even oracle;
  - 4.5: the seeds;
  - 5.1: the set;
  - 5.3: the bands;
  - 5.5: the OC;
  - 5.6: recovery;
  - 13: the result.
- Appendix P, P.6.2: the judgement procedure (`p_rules.p_judge`).
- Appendix Q: `q_jobs` with its `apl_to_mbon05_zero` edit, and Q's reproduction gate `02650a3`.

| R.9 section | Overrides / adds |
|---|---|
| R.9.1 | Disclosure only (no code). The gate-① forecast ≈ 0.048 is why gate ① became a validity gate. |
| R.9.2 | Gate ① = KC median ∈ **[0.03, 0.15]** ∧ ORN cap. The original qualification ([0.05, 0.09], single/dual tails ≤ 10%) is recorded beside encoder ③'s values on the same seeds. |
| R.9.3 | Gate ③ = testable_b ≥ 11 only. F_a / naive_a are recorded on even. The `STOP_EVEN_LOW_LEVER` sentence and the B_Fa sentence are replaced. |
| R.9.4 | Band 2 stays `c ≥ 11`. C's F_a / naive_a are recorded. Fixture for the counterexample: c 11 · F_a_C 0 · n 13 · F_a 2 → B_결론없음. |
| R.9.5 | G_fail ⇔ on (b) or (a): [pun_L ≤ pun_C − 2] ∨ [#(C pass ∧ L fail) ≥ 3]. Per-pair −p is recorded. G_fail's OC (OR false alarm) is recorded before the judgement. Fixtures pin pass→fail 2/3. |
| R.9.6 | The no-edit reproduction gate comes before smoke. The oracle uses the `q_jobs` path. Gate ③ runs L and C in the same code and run; c_even must be 7, else record and stop. |
| R.9.7 | Pre-read validity, then raw manifest commit and archive. z-renormalised values are records only. MBON05 saturation is per cell. E0 is kept. Mutation tests. |

## Global Constraints

- **Commits carry no trailers.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **Numbers live only in `r_spec`.**
  - Every R number is a field or method of `flymon.brain.r_spec.SPEC` (`LeverSpec`), or of `smoke(SPEC)` / `dataclasses.replace` in tests.
  - Encoder numbers are read from `e_spec.SPEC` (`E`):
    - windows 800 / 600 / 200 and α {0.2, 0.5, 0.8};
    - H.4 seeds 500–615, strength seeds 24_002_000–007, judgement seeds 24_100_xxx;
    - bar 11, margin 2, f_a_min 2, n_b 21, naive_max 0.5, testable_min 2.0;
    - OC grids, e0_strength 0.35.
  - The KC band [0.03, 0.15], `active_fx`, the encoder summary path and Q0's cache path come from `q_spec.SPEC`. P's numbers come from `p_spec.SPEC`.
  - Literal guards: `r_spec.py` may hold only the literal set in Task 1; `r_rules.py` only {0, 1}.
- **Never edit** `flymon/agent/e_*.py`, `encode_grid.py`, `encode.py`, `q_*`, `p_*`, `o_*`, `n_*`, `h4_*`, `k_jobs`, `h3_*`, `d6a`, `plasticity`, `fly_pool`, `engine_cpu`, `scripts/run_q.py`, `scripts/run_p.py`.
  - The one exception is Reading 1: one line in `tests/brain/test_p_spec.py`'s `MODULES`.
  - No module global of another track is monkeypatched at runtime: no write to `q_jobs._RIG`, `o_jobs._RIG` and the like.
- **The judgement set (L generator turns 64–103) is reached only through `r_pairs.judgement_rows`.**
  - Inside R, only `r_runner.build_ctx` (which only builds a lazy callable) and the runner's `_judgement_rows`, `stage_jm`, `stage_seal`, `stage_judge` and `_read` name it (AST test, Task 8).
  - `r_pairs.judgement_rows` refuses a smoke spec. Smoke uses even (b) pairs 0 and 20 only.
  - No R file holds an integer literal in 24_100_000–24_100_999 or 24_002_000–24_002_999 (AST test, Task 1).
- **Raw data of other tracks is read-only for R:**
  - `results/q/q0_cache/` (Q0 files);
  - `results/p/run/cache/p_arm/` (P's entries);
  - `results/r/ref/activity/` (encoder ③ entries copied by the controller, Runs step 0).
  - `results/encoder/` does not exist in this worktree and is never opened.
- **Writes are guarded.**
  - Raw files go under `results/r/` (git-ignored by `results/*`); the summary is `results/summary/r_lever.json` (tracked).
  - `results/r/ref/` is read-only, and anything else is refused with SystemExit 2.
  - The archive copy goes only under `~/flymon-archive/r/<seal id>/`, never over an existing directory.
  - Every write is atomic (temporary file + rename).
- **Scripts:**
  - Tests find the repository with `ROOT = Path(__file__).resolve().parents[2]`.
  - `scripts/run_r.py` has `if __name__ == "__main__":`. Pools are never started from a heredoc or `python -c`.
- **Subagents never run a real or smoke stage and never write under `results/`.** Tests write only under `tmp_path`. The controller runs every stage (section "Runs (controller)").
- **Numbers from R / R.9, verbatim:**
  - **Model:** C3 (block `h4` Params) + `apl_to_mbon05_zero` (exactly **2** CSC edges). Readout A = MBON13, P = MBON05; z = block `h4` in **every** condition.
  - **Conditions:** L = lever + E-grid k2-norm s **1.0**; C = no edit + E-grid k2-norm s 1.0; E0 = no edit + `odor_x_e0` / `odor_y_e0` at s **0.35** (record only).
  - **Gate ①:** 112 odours × 24_002_000+i (i < 8) under the lever. KC median ∈ **[0.03, 0.15]** ∧ ORN cap → PASS, else `STOP_STRENGTH_LEVER`.
  - **Gate ②:** P.6.2 on R's P spec, seeds **25_000_000+i** (i < 32), smoke **25_009_100–103**. c₁ comes from block n1 (0.608). `LEARNS_CONFIRMATORY` → PASS, other label → `STOP_PUNISH_BROKEN`, `INVALID` → one rerun after a fix.
  - **Gate ③:** even (a) 18 + (b) 21, H.4 seeds 500–615, L / C / E0 in one stage. c_even ≠ **7** → `STOP_C_EVEN_MISMATCH` (Reading 6); else testable_b < **11** → `STOP_EVEN_LOW_LEVER`.
  - **Judgement:**
    - (b) 21 · (a) 32; seeds 24_100_000+i / 24_100_100+i / 24_100_200+i (i < 8);
    - digests: E0 (b) `4b9bccc9…`, (a) `78509547…`, keys `33be39a1…` (full values in `r_spec`), last turn 103.
  - **Bands (first match):**
    1. counts ≠ 21 / 32 → `NOT_READ`;
    2. c ≥ 11 → B_결론없음;
    3. n ≤ c → B_Tb;
    4. n < 11 → B_결론없음;
    5. n − c < 2 → B_결론없음;
    6. G_fail → B_처벌가드;
    7. F_a < 2 → B_Fa;
    8. otherwise `SELECTED`.
  - **G_fail:** on (b) or (a): pun_C − pun_L ≥ **2** ∨ #(C pass ∧ L fail) ≥ **3**; punish pass ⇔ −p ≥ 2.
  - **Saturation:** per cell, rate = count / 0.6 s. Ceiling = 1000 / (refrac_steps × dt) Hz (C3: 1000 / (2 × 1.0) = 500 Hz). Shares at ≥ **0.5×** and ≥ **0.8×**; median, 95th percentile, max.
  - **Smoke seeds:** **25_008_000–25_008_099** (KC 2, even act 2 / select 2 / report 3), P smoke 25_009_100–103.

## Readings of the spec (decided here — the controller's rulings)

1. **`test_p_spec.py` gets one line.** Its `test_every_spec_module_is_enumerated` globs every `*_spec.py` and fails as soon as `flymon/brain/r_spec.py` exists. Task 1 adds `"flymon/brain/r_spec.py": "flymon.brain.r_spec"` to its `MODULES`. This is the same pattern Q needed (`efc0e24`, Q plan Reading 1). It is the only edit to a non-R file, and the reviewer flags it for the user.
2. **Class `LeverSpec`, and the reused seed blocks live in methods.**
   - `tests/brain/test_p_spec.py:_expand` treats a class *named* `RSpec` as `flymon.rescope`'s, so R's class must not be called `RSpec`.
   - `tests/agent/test_e_spec_store.py` collects every `seed`-named field default of every `flymon/brain/*_spec.py` and requires the encoder's blocks (24_002_xxx, 24_100_xxx, …) to be absent. R reuses those blocks on purpose (R.8), so `kc_seeds()`, `h4_seeds()`, `even_seeds()`, `judge_seeds()` and `p_repro_seeds()` are methods reading `E` / `P` at call time.
   - Only R's new blocks are fields: `smoke_seeds` and the nested `p` spec's `o2_seed0` / `smoke_seed0`. The collision tests see those.
3. **Gate ② reuses `p_rules.p_judge` unchanged.**
   - `p_judge`'s gate requires every row's edit to equal `spec.o.on_edit` (= `o1_conditions[0][1]`). R's P spec (`r_spec.R_P`) is therefore `p_spec.SPEC` with only `o2_seed0` (25_000_000), `smoke_seed0` (25_009_000) and `o1_conditions[0]` = (`"lever"`, `apl_to_mbon05_zero`) replaced. No P code changes.
   - c₁ comes from `p_cli.c1_source(R_P, smoke)`, i.e. block n1, as P read it.
   - The stimuli come from `n_cli.stimuli_at` at P's point (g 0.25, c_δ 8).
   - P.6.4's OC is not rerun: R.2 asks for P.6.2's judgement procedure only.
   - An `INVALID` gate ② can be rerun once with `--rerun-after-invalid`, only with a changed code key. The INVALID block is kept as `gate2_invalid` (R.5 / P.6.5).
4. **The reproduction gate (R.9.6)** has three parts. All must pass, or later stages refuse (exit 4).
   - **(a) KC.** Three odours, `ELECTRIC|FIGHTING` (single), `GRASS|FIRE+FLYING` and `WATER|PSYCHIC+WATER` (dual), × strength seeds 24_002_000–007, no edit, s 1.0, run through `kc_activity_job`.
     - `{frac, max_win}` must be canonical-JSON equal to the encoder's activity entry.
     - The entry's file name is the encoder cache key, computed by `EMeasurer._act_inputs` + `ECache._path` while the encoder code key equals block `strength`'s (true at HEAD). The controller copies the 3 files from the encoder-redesign worktree into `results/r/ref/activity/` (Runs step 0; `--list-refs` prints the names).
     - The odour's median must also equal block `strength`'s `per_odour` value, a second check that the file is the right one.
   - **(b) P.** Seeds `P.seeds[i]` for i ∈ (0, 15, 31) × 2 directions × 3 arms = 18 `r_arm_job` runs, no edit. Each must be canonical-JSON equal to P's committed `p_arm` entry, the one whose `code_key` is block `p`'s `measure_key`.
     - Both sides are compared after removing `wall_s` at any depth (timing).
     - On R's side the measurer's decorations (`direction`, `x`, `y`, `point`) and R's `r` key are removed too.
   - **(c) Oracle.** Even (b) pair 0 with no edit on H.4 seeds through `r_oracle_job`. The result minus `q` and `r` must equal the Q0 file's `result` (Q0 sha checked by `q_pairs.q0_files`). Per-cell sums must equal the type counts. This pins R's wrapper (Q's own repro already pinned `q_oracle_job`). It also records the unedited CSC sha (`csc_sha256_none`), which C and E0 must carry later.
5. **The oracle is `q_jobs.q_oracle_job` with `fixed_alphas = ()`**, so Q's fixed arms do not run. The result minus `q` equals `h4_jobs.oracle_job`'s (test).
   - `r_oracle_job` then probes the P readout type's cells on the report seeds with the weights at w0: `presentation.decide` with `idx` = the MBON05 cells, the same seeds and step sequence as the oracle's `pre` probe.
   - The per-cell counts must sum to `report.pre.P` (raw check). Saturation is computed from them.
6. **Gate ③ is one stage `even`.** It measures L, C and E0 on the 39 even pairs in one pool (~70–90 min; resumable), then `gate3` reads the even block (no pool).
   - E0 is a record (R.6's saturation "L·C·E0, 짝수와 판정 세트").
   - Order of checks: any validity reason → `INVALID`; C's testable_b ≠ 7 → **`STOP_C_EVEN_MISMATCH`**, a name chosen here for R.9.6's "다르면 기록하고 멈춘다", with the fixed sentence in `r_rules.SENTENCES`; then L testable_b < 11 → `STOP_EVEN_LOW_LEVER`; else PASS.
7. **G_fail (R.9.5), per axis over the pairs present in both L and C:**
   - pun_X = #(−p ≥ 2) under X;
   - drop ⇔ pun_C − pun_L ≥ 2;
   - flips ⇔ #(C pass ∧ L fail) ≥ 3.

   G_fail ⇔ drop ∨ flips on (b) or on (a).
8. **B_결론없음 sentences** are R.3's sentence plus `" ({n}/21 대 {c}/21, F_a {f_a}/32)"`, the encoder's `_NUMS` form ("R.3의 해당 문장 + 수치"). B_Tb appends `" → 이 지렛대를 닫는다."`.
9. **`STOP_STRENGTH_LEVER`'s 〈걸린 조건과 값〉** is the failing condition(s) of R.9.2, e.g. `"냄새별 KC 활성 중앙값 0.0281 ∉ [0.03, 0.15]"` and/or `"ORN 상한(s 1.0)"`, joined by `"; "`. R.7's wording "KC 활성 자격" is kept verbatim.
10. **Pre-read validity (R.9.7) has three outcomes in `seal`:**
    - `INVALID`: L changed a number of CSC edges other than exactly 2 (R.6's "아니면 `INVALID`").
    - `NOT_READ`: any other failure:
      - completeness, uniqueness, probe lengths, finite values;
      - per-cell sums, code key, seeds;
      - C / E0 edges ≠ 0 or an edit name that differs from the condition's;
      - L sha = C sha; C sha ≠ E0 sha or ≠ the repro's unedited sha;
      - a raw file whose sha256 differs from the jm manifest.
    - `SEALED`: nothing failed.

    `judge` reads only after `SEALED`. `NOT_READ` and `INVALID` are stop points for the user. The judgement blocks `jm:*` hold **no pair statistic**: only `raw_check`'s completeness, edges, sha and manifest, so nothing band-relevant is visible before the seal.
11. **Saturation (R.9.7):**
    - per-cell rate = count × 1000 / read_ms over every cell × report seed × X/Y of the naive probe;
    - ceiling = 1000 / (refrac_steps × dt) from the job's own Params (C3: refractory 2.2 ms, dt 1.0 → 2 steps → 500 Hz);
    - recorded: shares of cell-presentations at ≥ 0.5× / 0.8× the ceiling, plus median, 95th percentile and max.
12. **z renormalisation (R.9.7, record only)** uses the reference set = the condition's own naive report-seed probes (pairs × seeds × X/Y, type sums), `h4_rules.z_constants` with H.4's `z_ddof` (0). It gives testable_b, F_a, naive_a and median |d_pre| under that z, beside median |d_pre| under block `h4`'s z. It never feeds a band.
13. **OC (R.6, R.9.5)** is exact, assuming independent pairs.
    - n ~ Bin(21, q) and F_a ~ Bin(naive_a, q) (encoder 5.5's rows q × c × naive_a ∈ 0..12).
    - G_fail ~ Bernoulli(g), with g from a per-pair discordance model per axis: P(C pass ∧ L fail) = a_pf and P(C fail ∧ L pass) = a_fp, multinomial over the axis' pairs, OR over (b) 21 and (a) 32.
    - g rows: none (0); the null false alarm a_pf = a_fp ∈ {0.02, 0.05, 0.1, 0.15}; harm rows (a_pf, a_fp) ∈ {(0.1, 0.02), (0.15, 0.02), (0.2, 0.05)}.
    - Each cell is weighted through `read_band` itself. The table sha256 goes into the summary.
14. **Exit codes of `scripts/run_r.py`:**
    - 0: PASS / SEALED / READ / a record stage;
    - 2: a refusal;
    - 3: a gate STOP (recorded, R stops);
    - 4: repro failed;
    - 5: gate INVALID;
    - 6: seal not SEALED, or judge NOT_READ.
15. **One R code key** (`R_MEASURE_FILES`) for every stage.
    - `jm:*`, `seal` and `judge` refuse unless every earlier block carries the current code key and was written with no dirty hashed file.
    - They also refuse when the summary's git history already holds a `judge` block (`e_runner.summary_git`'s `judge_commits`: the set is used once).
    - Smoke must be problem-free (edges, sha relations, validity) before `gate1` and every later stage.
16. **Every stage refuses to rewrite its own block.** The exceptions are a failed `repro` (fix, commit or discard, rerun: R.9.6) and gate ②'s one INVALID rerun. A STOP or INVALID gate blocks every later stage.
17. **Recovery after reading (R.5)** has two stages, both needing `--note`:
    - `recompute` reruns the band arithmetic from the sealed raw files (sha re-checked) and appends to block `recompute`;
    - `invalid_run` writes block `invalid_run` once.

    Neither re-measures.
18. **The archive** goes to `~/flymon-archive/r/<UTC stamp>-<HEAD[:12]>/<kind>/<basename>`, with each copy's sha256 checked against the source. The list is recorded in block `seal`.

## Review Focus

1. **A judgement measurement killed at the 2 h limit (or by hand) mid-condition.** Expect no `jm:*` block. A rerun recomputes only the missing pairs and writes the block once all 53 are back. Tests: Task 4 (`test_oracle_resumes_only_missing_pairs`) and Task 8 (`test_jm_writes_nothing_until_complete`).
2. **A raw file changed or deleted between `seal` and `judge`.** Expect `judge` to refuse (exit 2) and never compute a band. Test: Task 8 (`test_judge_refuses_a_raw_file_changed_after_the_seal`).
3. **The encoder reference files missing or wrong.** Expect `repro` to refuse with the basenames to copy when files are missing. When a copied file differs, or its median disagrees with block `strength`, expect `passed: false` and exit 4. Tests: Task 7 (`test_repro_refuses_without_refs`, `test_repro_fails_on_a_wrong_reference`).
4. **Writing where another track lives.** Expect SystemExit 2 for `results/q/…`, `results/p/…`, `results/encoder/…`, `results/r/ref/…`, `results/summary/q_reward.json`, and for an archive destination outside `~/flymon-archive/r/` or already existing. Test: Task 4.
5. **The judgement set reached early** by smoke, a gate, or `even`. Expect `ctx["judgement_rows"]` never to be called before `jm:L`, and `r_pairs.judgement_rows` to refuse a smoke spec. Tests: Task 2 and Task 8 (`test_judgement_set_is_touched_only_from_jm_on`).

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/r_spec.py` (create) | `LeverSpec`, `Cond`, `R_P`, `SPEC`, `smoke()`: every R number; reused blocks as methods |
| `flymon/brain/r_pairs.py` (create) | `row_key`, `okey`, `calibration_odours`, `cap_at`, `even_rows`, `check_set`, `judgement_rows`, `cond_odours`, `encoder_activity_ref`, `p_reference` |
| `flymon/brain/r_jobs.py` (create) | `kc_activity_job`, `r_arm_job`, `r_oracle_job` (worker jobs on `q_jobs.q_rig`) |
| `flymon/brain/r_store.py` (create) | guard, atomic writes, summary blocks, `move_block`, `RCache`, `load_manifest`, `archive_copy` |
| `flymon/brain/r_measure.py` (create) | `R_MEASURE_FILES`, `RMeasurer` (activity, oracle, arms; per-unit cache with resume) |
| `flymon/brain/r_records.py` (create) | `strip_job`, `cell_sums_ok`, `pair_row`, `stats_ok`, `aggregate`, `counts`, `raw_check`, `saturation`, `z_renorm`, `cond_summary`, `transitions`, `compare`, `gate1_record`, `preread_validity` |
| `flymon/brain/r_rules.py` (create) | outcome / band constants, `gate1`, `gate2`, `gate3`, `g_fail`, `read_band`, `SENTENCES`, `sentence`, `g_axis`, `g_prob`, `oc` |
| `flymon/brain/r_runner.py` (create) | `ORDER`, `R_HASHED_FILES`, `p_items`, `build_ctx`, `Runner` (all stages) |
| `scripts/run_r.py` (create) | the CLI, `exit_code`, `__main__` guard |
| `tests/brain/test_r_spec.py`, `test_r_pairs.py`, `test_r_jobs.py`, `test_r_store_measure.py`, `test_r_records.py`, `test_r_rules.py`, `test_r_runner.py`, `test_r_judge.py`, `r_fixtures.py`, `r_world.py` (create) | tests and fixtures |
| `tests/brain/test_p_spec.py` (modify: one `MODULES` line) | Reading 1 |

---

### Task 1: `r_spec` — numbers, conditions, the gate-② P spec, seed-collision and literal guards

**Files:**
- Create: `flymon/brain/r_spec.py`
- Modify: `tests/brain/test_p_spec.py` (one `MODULES` entry)
- Test: `tests/brain/test_r_spec.py`

**Interfaces:**
- Consumes: `e_spec.SPEC`, `q_spec.SPEC`, `p_spec.SPEC` / `PSpec` / `smoke`, `q_jobs.MBON05` / `q_jobs.NONE`, `e_measure.PUNISH_TYPE` / `REWARD_TYPE`.
- Produces:
  - `Cond(name, edit, odour, strength)`, with odour ∈ {"egrid", "e0"};
  - `LeverSpec` with the fields below;
  - `LeverSpec.conditions() -> (L, C, E0)`, `cond(name) -> Cond`;
  - `kc_seeds() -> tuple`, `h4_seeds() -> dict`, `even_seeds() -> dict`, `judge_seeds() -> dict` (ValueError when smoke), `p_repro_seeds() -> tuple`;
  - `R_P` (PSpec), `SPEC`, `smoke(spec) -> LeverSpec`;
  - field names used later: `config, strength, lever_edit, no_edit, lever_edges, a_type, p_type, punish_type, reward_type, e0_strength, cond_names, settle_ms, read_ms, window_ms, alphas, fixed_alphas, active_fx, testable_min, naive_max, bar_b, f_a_min, margin, n_b, n_a, n_a_even, last_turn, digest_e0_b, digest_e0_a, digest_keys, valid_band, n_calib, p, c_even_expected, g_fail_drop, g_fail_flips, sat_fracs, sat_quantile, ms_per_s, oc_q, oc_c, oc_naive_max, oc_discord, oc_harm, kc_repro_odours, p_repro_idx, oracle_repro_idx, smoke_seeds, smoke_pairs, smoke_odours, smoke, encoder_summary, m0d_summary, q0_cache_dir, p_summary, p_cache_dir, ref_dir, cache_dir, smoke_cache_dir, smoke_detail, summary, archive_root, workers, pool_timeout_s`.

- [ ] **Step 1: Add R to the spec-module list (Reading 1)**

In `tests/brain/test_p_spec.py`, the `MODULES` dict's last line is:

```python
           "flymon/agent/e_spec.py": "flymon.agent.e_spec", "flymon/brain/q_spec.py": "flymon.brain.q_spec"}
```

Replace it with (one entry added; nothing else in the file changes):

```python
           "flymon/agent/e_spec.py": "flymon.agent.e_spec", "flymon/brain/q_spec.py": "flymon.brain.q_spec",
           "flymon/brain/r_spec.py": "flymon.brain.r_spec"}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/brain/test_r_spec.py
"""Spec R.2 / R.8 / R.9: every R number in r_spec; R's new blocks (P replication 25_000_000+i, P smoke 25_009_xxx, R
smoke 25_008_xxx) collide with no declared seed and no declared seed's training seeds (test_p_spec's collector); the
reused blocks (encoder ③ 24_002_xxx, H.4 500-615, judgement 24_100_xxx) are exactly their owners' and live in methods,
so the encoder track's collision test (field defaults of every brain *_spec module) never sees them; R's P spec differs
from P's only in its seeds and the edit p_judge's gate expects; no R file holds a reused block as a literal."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.p_spec import SPEC as P
from flymon.brain.q_spec import SPEC as Q
from flymon.brain.r_spec import R_P, SPEC, Cond, LeverSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_r() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/r_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


R_SEEDS = set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.smoke_seeds)
R_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds())


def test_r_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/r_spec.py") == "flymon.brain.r_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.p.seeds == tuple(range(25_000_000, 25_000_032))
    assert len(SM.p.seeds) == 4 and all(25_009_000 <= s < 25_010_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(25_008_000, 25_008_100))
    sm = set(SM.kc_seeds()) | {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 9


def test_new_blocks_collide_with_nothing_declared():
    declared = _declared_without_r()
    for s in (500, 615, 24_002_000, 24_100_000, 24_100_207, 23_000_000, 24_000_000, 24_008_000, 22_000_000):
        assert s in declared, s
    assert not R_SEEDS & declared
    assert not {s for s in R_SEEDS if (s - BASE) // STRIDE in declared}     # no declared s trains on an R seed
    assert not R_TRAIN & declared
    assert not {s for s in R_TRAIN if (s - BASE) // STRIDE in declared}
    assert not R_TRAIN & R_SEEDS


def test_reused_blocks_are_read_from_their_owners():
    assert SPEC.kc_seeds() == E.strength_seeds == tuple(range(24_002_000, 24_002_008))
    h4 = dict(act=list(range(500, 508)), select=list(range(600, 608)), report=list(range(608, 616)))
    assert SPEC.h4_seeds() == SPEC.even_seeds() == h4 == SM.h4_seeds()
    assert SPEC.judge_seeds() == dict(act=list(E.judge_act_seeds), select=list(E.judge_select_seeds),
                                      report=list(E.judge_report_seeds))
    assert SPEC.p_repro_seeds() == (23_000_000, 23_000_015, 23_000_031) and set(SPEC.p_repro_seeds()) <= set(P.seeds)
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_no_encoder_seed_sits_in_an_r_field():
    assert not _module_seeds("flymon.brain.r_spec") & e_spec.track_seeds(E)


def test_gate2_spec_differs_from_p_only_in_seeds_and_the_edit():
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(R_P.o, f.name) != getattr(P.o, f.name)}
    assert diff == {"o2_seed0", "smoke_seed0", "o1_conditions"}
    assert R_P.o.on_edit == SPEC.lever_edit == "apl_to_mbon05_zero"
    assert R_P.o.o1_conditions[1:] == P.o.o1_conditions[1:]
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(R_P, f.name) == getattr(P, f.name), f.name
    assert SPEC.p is R_P and R_P.pairs() == P.pairs() and R_P.flags() == P.flags()


def test_conditions_and_numbers():
    L, C, E0 = SPEC.conditions()
    assert L == Cond("L", "apl_to_mbon05_zero", "egrid", 1.0)
    assert C == Cond("C", "none", "egrid", 1.0)
    assert E0 == Cond("E0", "none", "e0", E.e0_strength) and E.e0_strength == 0.35
    assert SPEC.cond("C") == C and SPEC.cond_names == ("L", "C", "E0")
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.n_b, SPEC.n_a, SPEC.n_a_even) == (11, 2, 2, 21, 32, 18)
    assert SPEC.valid_band == Q.kc_band == (0.03, 0.15)
    assert (SPEC.g_fail_drop, SPEC.g_fail_flips, SPEC.c_even_expected, SPEC.lever_edges) == (2, 3, 7, 2)
    assert (SPEC.settle_ms, SPEC.read_ms, SPEC.window_ms, SPEC.alphas) == (800.0, 600.0, 200, (0.2, 0.5, 0.8))
    assert SPEC.fixed_alphas == () and SPEC.last_turn == 103 and SPEC.n_calib == 112
    assert SPEC.digest_e0_b.startswith("4b9bccc9") and SPEC.digest_e0_a.startswith("78509547")
    assert SPEC.digest_keys.startswith("33be39a1")


def test_paths_match_their_owners():
    from flymon.brain import p_cli, p_measure
    assert SPEC.p_summary == p_measure.SUMMARY
    assert SPEC.p_cache_dir == p_cli.RUN_OUT + "/cache/p_arm"
    assert SPEC.q0_cache_dir == Q.q0_cache_dir and SPEC.encoder_summary == Q.encoder_summary
    assert SPEC.m0d_summary == E.m0d_summary


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(LeverSpec):
        if f.name not in ("p", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.p.o.on_edit == SPEC.lever_edit and SM.workers == 4


ALLOWED_R_SPEC = {0, 1, 2, 3, 4, 6, 7, 9, 15, 16, 18, 20, 31, 32, 103, 112, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0,
                  95.0, 1000.0, 3600.0, 25_000_000, 25_009_000, 25_008_000, 25_008_100}


def test_r_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/r_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_R_SPEC, nums - ALLOWED_R_SPEC


def test_no_r_file_holds_a_reused_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("r_*.py")) + sorted((ROOT / "scripts").glob("run_r*.py"))
    assert files
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_002_000 <= i < 24_003_000}
        assert not bad, (p.name, bad)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_spec.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_spec'`.

- [ ] **Step 4: Write the implementation**

```python
# flymon/brain/r_spec.py
"""Every number of spec appendix R as amended by R.9 (R.9 wins over R.0-R.8): the lever APL->MBON05 removal and its
one M2 judgement on the encoder track's unused judgement set (L generator turns 64-103).
- Numbers the encoder track, Q and P hold are read (E = e_spec.SPEC, Q = q_spec.SPEC, P = p_spec.SPEC), never restated.
- The reused seed blocks — gate ① = encoder ③'s strength seeds (on purpose, R.8), gate ③ = H.4's 500-615, the
  judgement oracle block (encoder 4.5) — are returned by methods and never stored in a field: the encoder track's
  collision test reads every "seed" field of every flymon/brain/*_spec.py and would flag them (plan Reading 2).
- The class is LeverSpec, not RSpec: tests/brain/test_p_spec.py's collector expands a class named RSpec as
  flymon.rescope's (plan Reading 2).
- Gate ②'s P spec R_P is p_spec.SPEC with only the seed blocks (25_000_000+i, smoke 25_009_xxx) and O's first
  condition — the edit p_rules.p_judge's gate expects through spec.o.on_edit — changed (plan Reading 3)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.e_measure import PUNISH_TYPE, REWARD_TYPE
from ..agent.e_spec import SPEC as E
from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .q_jobs import MBON05 as LEVER_EDIT
from .q_jobs import NONE as NO_EDIT
from .q_spec import SPEC as Q

LEVER = "lever"
R_P = dataclasses.replace(P_SPEC, o=dataclasses.replace(
    P_SPEC.o, o2_seed0=25_000_000, smoke_seed0=25_009_000,
    o1_conditions=((LEVER, LEVER_EDIT),) + P_SPEC.o.o1_conditions[1:]))


@dataclass(frozen=True)
class Cond:
    name: str
    edit: str               # q_jobs edit: the lever (apl_to_mbon05_zero) or none
    odour: str              # "egrid": the row's E-grid k2-norm odor_x / odor_y; "e0": its odor_x_e0 / odor_y_e0
    strength: float


@dataclass(frozen=True)
class LeverSpec:
    # ---- the model variant and the encoder (R.1) ---------------------------------------------------------------
    config: str = "k2-norm"
    strength: float = 1.0
    lever_edit: str = LEVER_EDIT
    no_edit: str = NO_EDIT
    lever_edges: int = 2                        # R.6: exactly 2 CSC edges, else INVALID
    a_type: str = "MBON13"
    p_type: str = "MBON05"
    punish_type: str = PUNISH_TYPE
    reward_type: str = REWARD_TYPE
    e0_strength: float = E.e0_strength
    cond_names: tuple = ("L", "C", "E0")
    # ---- the oracle (H.4 / encoder 4.4 as is) ---------------------------------------------------------------------
    settle_ms: float = E.settle_ms
    read_ms: float = E.read_ms
    window_ms: int = E.window_ms
    alphas: tuple = E.alphas
    fixed_alphas: tuple = ()                    # q_oracle_job's fixed reward-only arms are not run (Reading 5)
    active_fx: float = Q.active_fx              # q_oracle_job's reach record threshold, unchanged
    testable_min: float = E.testable_min
    naive_max: float = E.naive_max
    bar_b: int = E.bar_b
    f_a_min: int = E.f_a_min
    margin: int = E.margin
    n_b: int = E.n_b
    # ---- the judgement set (encoder 5.1; ⓪ 2c6a292) -----------------------------------------------------------------
    n_a: int = 32
    n_a_even: int = 18
    last_turn: int = 103
    digest_e0_b: str = "4b9bccc9a640d8affb36754f862f8ed67ea04c0b07e64f3d1529c57c6fe1ea3f"
    digest_e0_a: str = "78509547f571ec8ddbd0b2b575d989f761a547b2b52a48cc89bd0e8fe7c3b5c1"
    digest_keys: str = "33be39a1c5b086a7034e941ad20f4fbe055ce0708bc84ea6e138ba8fe44e028e"
    # ---- gate ① (R.9.2) -----------------------------------------------------------------------------------------------
    valid_band: tuple = Q.kc_band               # [0.03, 0.15], "Q와 같은 대역"
    n_calib: int = 112
    # ---- gate ② (R.2, Reading 3) ----------------------------------------------------------------------------------------
    p: PSpec = R_P
    # ---- gate ③ (R.9.3, R.9.6) ------------------------------------------------------------------------------------------
    c_even_expected: int = 7                    # encoder 13: k2-norm 7/21 on the same even pairs and seeds
    # ---- bands (R.3, R.9.4, R.9.5) -----------------------------------------------------------------------------------
    g_fail_drop: int = 2
    g_fail_flips: int = 3
    # ---- records (R.6, R.9.7) -------------------------------------------------------------------------------------------
    sat_fracs: tuple = (0.5, 0.8)
    sat_quantile: float = 95.0
    ms_per_s: float = 1000.0
    # ---- the operating characteristic (R.6, R.9.5, Reading 13) ------------------------------------------------------
    oc_q: tuple = E.oc_q
    oc_c: tuple = E.oc_c
    oc_naive_max: int = E.oc_naive_max
    oc_discord: tuple = (0.02, 0.05, 0.1, 0.15)
    oc_harm: tuple = ((0.1, 0.02), (0.15, 0.02), (0.2, 0.05))
    # ---- the no-edit reproduction gate (R.9.6, Reading 4) -----------------------------------------------------------
    kc_repro_odours: tuple = ("ELECTRIC|FIGHTING", "GRASS|FIRE+FLYING", "WATER|PSYCHIC+WATER")
    p_repro_idx: tuple = (0, 15, 31)
    oracle_repro_idx: tuple = (0,)
    # ---- smoke (R.8) ----------------------------------------------------------------------------------------------------
    smoke_seeds: tuple = tuple(range(25_008_000, 25_008_100))
    smoke_pairs: tuple = (0, 20)                # even (b) indices; the judgement set is never in smoke
    smoke_odours: int = 4
    smoke: bool = False
    # ---- paths and the pool ---------------------------------------------------------------------------------------------
    encoder_summary: str = Q.encoder_summary
    m0d_summary: str = E.m0d_summary
    q0_cache_dir: str = Q.q0_cache_dir
    p_summary: str = "results/summary/p_learning.json"
    p_cache_dir: str = "results/p/run/cache/p_arm"
    ref_dir: str = "results/r/ref/activity"
    cache_dir: str = "results/r/cache"
    smoke_cache_dir: str = "results/r/smoke/cache"
    smoke_detail: str = "results/r/smoke.json"
    summary: str = "results/summary/r_lever.json"
    archive_root: str = "~/flymon-archive/r"
    workers: int = 16
    pool_timeout_s: float = 3600.0

    def conditions(self) -> tuple:
        names = self.cond_names
        return (Cond(names[0], self.lever_edit, "egrid", self.strength),
                Cond(names[1], self.no_edit, "egrid", self.strength),
                Cond(names[2], self.no_edit, "e0", self.e0_strength))

    def cond(self, name: str) -> Cond:
        return {c.name: c for c in self.conditions()}[name]

    def kc_seeds(self) -> tuple:
        """Gate ① (and the KC repro): encoder ③'s strength seeds; smoke: two R smoke seeds."""
        return tuple(self.smoke_seeds[0:2]) if self.smoke else tuple(E.strength_seeds)

    def h4_seeds(self) -> dict:
        return dict(act=list(E.even_act_seeds), select=list(E.even_select_seeds), report=list(E.even_report_seeds))

    def even_seeds(self) -> dict:
        if self.smoke:
            s = self.smoke_seeds
            return dict(act=list(s[2:4]), select=list(s[4:6]), report=list(s[6:9]))
        return self.h4_seeds()

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (R.8)")
        return dict(act=list(E.judge_act_seeds), select=list(E.judge_select_seeds), report=list(E.judge_report_seeds))

    def p_repro_seeds(self) -> tuple:
        return tuple(P_SPEC.seeds[i] for i in self.p_repro_idx)


SPEC = LeverSpec()


def smoke(spec: LeverSpec = SPEC) -> LeverSpec:
    """Scale only: P's smoke derivation on R's P spec (4 seeds 25_009_100+), R's smoke seeds through the methods, 4
    workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), smoke=True, workers=4)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_r_spec.py tests/brain/test_p_spec.py tests/brain/test_q_spec.py tests/agent/test_e_spec_store.py -q`
Expected: PASS (all four files; the three existing collision tests stay green with R added).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/r_spec.py tests/brain/test_r_spec.py tests/brain/test_p_spec.py
git commit -m "feat(r): r_spec — R numbers, gate-2 P spec, reused seed blocks as methods, collision and literal guards"
```

---

### Task 2: `r_pairs` — calibration odours, even rows, judgement rows with digests, reproduction references

**Files:**
- Create: `flymon/brain/r_pairs.py`
- Test: `tests/brain/test_r_pairs.py`

**Interfaces:**
- Consumes: Task 1 `SPEC`, `smoke`, `Cond`; `e_pairs.even_situations` / `judgement_set` / `attach_odours`; `q_pairs.codebook` / `key_str` / `b_rows`; `encode_grid.reachable` / `odour` / `cap_ok`; `h4_pairs.pool_vocabulary`; `EMeasurer._act_inputs`, `ECache._path`.
- Produces:
  - `row_key(r) -> str`, the format `"axis|turn|x|y"` (= `q_pairs.key_str`);
  - `okey(m, opp) -> str`;
  - `calibration_odours(rc, cb, rule) -> (dict, list, list)`;
  - `cap_at(rc, cb, rule, s, max_rate_hz, cap_hz) -> bool`;
  - `even_rows(pops, rc, enc, spec) -> list`;
  - `check_set(js, enc, spec) -> list[str]`;
  - `judgement_rows(pops, rc, enc, spec) -> list` (ValueError on a digest mismatch or a smoke spec);
  - `cond_odours(row, cond) -> (dict, dict)`;
  - `encoder_activity_ref(oid, odour, params, n_kc, enc_code, spec) -> str` (a basename);
  - `p_reference(cache_dir, measure_key, wanted: set) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_r_pairs.py
"""R.3 / R.9.6 / encoder 4.3, 5.1: the 112 calibration odours are encoder ③'s; the even rows are H.4's 8 even turns
((a) 18, (b) 21 — the (b) rows equal Q's); the judgement rows are the encoder's set with the three digests checked
against both R's declared values and block set, refused for a smoke spec; the encoder activity reference is the
encoder's own cache name; P's reference entries are block p's (kind and measure key)."""
import copy
import json
from pathlib import Path

import pytest

from flymon.agent.config import load_c3_config
from flymon.brain import odor_real, q_pairs
from flymon.brain import r_pairs as RP
from flymon.brain.q_spec import SPEC as Q
from flymon.brain.r_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
ENC = json.loads((ROOT / "results/summary/encoder_grid.json").read_text())
NPZ = ROOT / "data/malecns.npz"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="no connectome")


@pytest.fixture(scope="module")
def world():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    return pops, {str(t): len(v) for t, v in pops.receptor_types.items()}


@needs_npz
def test_calibration_odours_are_encoder_3s_112(world):
    _, rc = world
    odours, single, dual = RP.calibration_odours(rc, q_pairs.codebook(ENC, SPEC), "norm")
    per = ENC["strength"]["configs"]["k2-norm"]["table"]["1.0"]["per_odour"]
    assert set(odours) == set(per) and len(odours) == SPEC.n_calib
    assert (len(single), len(dual)) == (32, 80)
    assert SPEC.kc_repro_odours[0] in single and set(SPEC.kc_repro_odours[1:]) <= set(dual)


@needs_npz
def test_cap_holds_at_s_1_and_not_at_1_4(world):
    _, rc = world
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    cb = q_pairs.codebook(ENC, SPEC)
    args = (cfg.params.max_rate_hz, odor_real.cap_hz(cfg.params))
    assert RP.cap_at(rc, cb, "norm", 1.0, *args) is True
    assert RP.cap_at(rc, cb, "norm", 1.4, *args) is False             # encoder ③: ORN_CAP at 1.4


@needs_npz
def test_even_rows_are_h4s_and_their_b_rows_are_qs(world):
    pops, rc = world
    from flymon.agent import e_pairs
    rows = RP.even_rows(pops, rc, ENC, SPEC)
    assert [r["axis"] for r in rows].count("b") == 21 and [r["axis"] for r in rows].count("a") == 18
    assert all(0 <= r["turn"] < 16 and r["turn"] % 2 == 0 for r in rows)
    qb = q_pairs.b_rows(e_pairs.even_situations(pops), rc, ENC, Q)
    mine = [r for r in rows if r["axis"] == "b"]
    assert [RP.row_key(r) for r in mine] == [q_pairs.key_str(r) for r in qb]
    assert all(a["odor_x"] == b["odor_x"] and a["odor_y"] == b["odor_y"] for a, b in zip(mine, qb))


@needs_npz
def test_judgement_rows_check_the_digests_and_refuse_smoke(world):
    pops, rc = world
    rows = RP.judgement_rows(pops, rc, ENC, SPEC)
    assert sum(r["axis"] == "b" for r in rows) == 21 and sum(r["axis"] == "a" for r in rows) == 32
    assert all(64 <= r["turn"] <= 103 for r in rows)
    assert all("odor_x_e0" in r and "odor_x" in r for r in rows)
    bad = copy.deepcopy(ENC)
    bad["set"]["digest_keys"] = "0" * 64
    with pytest.raises(ValueError, match="digest_keys"):
        RP.judgement_rows(pops, rc, bad, SPEC)
    with pytest.raises(ValueError, match="smoke"):
        RP.judgement_rows(pops, rc, ENC, smoke(SPEC))


def _js(**kw):
    js = dict(digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, n_a=32,
              last_turn=103, status="OK", b=[0] * 21, a=[0] * 32)
    return dict(js, **kw)


def test_check_set_compares_generated_block_and_declared():
    assert RP.check_set(_js(), ENC, SPEC) == []
    assert any("digest_e0_b" in m for m in RP.check_set(_js(digest_e0_b="x"), ENC, SPEC))
    enc = copy.deepcopy(ENC)
    enc["set"]["n_a"] = 31
    assert any("n_a" in m for m in RP.check_set(_js(), enc, SPEC))
    assert RP.check_set(_js(b=[0] * 20), ENC, SPEC)
    assert RP.check_set(_js(status="STOP_SET_SHORT"), ENC, SPEC)


def test_cond_odours():
    row = dict(odor_x={"A": 1.0}, odor_y={"B": 1.0}, odor_x_e0={"C": 1.0}, odor_y_e0={"D": 1.0})
    L, C, E0 = SPEC.conditions()
    assert RP.cond_odours(row, L) == ({"A": 1.0}, {"B": 1.0}) == RP.cond_odours(row, C)
    assert RP.cond_odours(row, E0) == ({"C": 1.0}, {"D": 1.0})


def test_encoder_activity_ref_is_the_encoder_cache_name():
    from flymon.agent.e_measure import EMeasurer
    from flymon.agent.e_spec import SPEC as E
    from flymon.agent.e_store import ECache
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    code = {"key": "k", "files": {}, "versions": {}}
    a = RP.encoder_activity_ref("X|Y", {"ORN_A": 1.0}, cfg.params, 100, code, SPEC)
    b = RP.encoder_activity_ref("X|Z", {"ORN_A": 1.0}, cfg.params, 100, code, SPEC)
    assert a != b and a.endswith(".json") and len(a) == 24 + len(".json")
    m = EMeasurer(None, ECache(f"{E.raw_dir}/cache", code), cfg.params, 100, E, guard_params=False)
    assert a == m.cache._path("activity", m._act_inputs("X|Y", {"ORN_A": 1.0}, 1.0, E.strength_seeds)).name


def test_p_reference_reads_only_block_ps_entries(tmp_path):
    d = tmp_path / "p_arm"
    d.mkdir()

    def put(name, kind, key, direction, arm, seed, res):
        (d / name).write_text(json.dumps(dict(kind=kind, code_key=key, inputs=dict(direction=direction, arm=arm,
                                                                                   seed=seed), result=res)))
    put("a.json", "p_arm", "K", "r1", "punish", 23_000_000, {"v": 1})
    put("b.json", "p_arm", "OTHER", "r1", "plastic", 23_000_000, {"v": 2})
    put("c.json", "o2_arm", "K", "r2", "punish", 23_000_000, {"v": 3})
    (d / "d.json").write_text("{broken")
    got = RP.p_reference(d, "K", {("r1", "punish", 23_000_000), ("r1", "plastic", 23_000_000)})
    assert got == {("r1", "punish", 23_000_000): {"v": 1}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_pairs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_pairs'`.

- [ ] **Step 3: Write the implementation**

```python
# flymon/brain/r_pairs.py
"""R's rows and references (R.1-R.3, R.9.6). Nothing here runs the engine.
- calibration_odours: encoder 4.3's 112 reachable situation odours (e_runner.Runner._odour_table's construction).
- even_rows: H.4's 8 even turns, (a) 18 · (b) 21, E-grid k2-norm odours (e_pairs.even_situations checks the E0
  digest; q_pairs.codebook checks the committed codebook digest). E0 odours stay on each row (odor_x_e0 / odor_y_e0).
- judgement_rows: the encoder's judgement set (L generator turns 64-103) with the three digests, n_a and the last turn
  checked against R's declared values AND block set (R.3); refused for a smoke spec (R.8). Only R's judgement stages
  call it (plan Global Constraints).
- the reproduction gate's references (Reading 4): the encoder ③ activity entry's file name (EMeasurer's key, computed
  only), and P's committed p_arm entries (kind p_arm, block p's measure key)."""
from __future__ import annotations

import json
from pathlib import Path

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import cap_ok, odour, reachable
from . import q_pairs
from .h4_pairs import pool_vocabulary


def row_key(r) -> str:
    return f"{r['axis']}|{int(r['turn'])}|{r['x']}|{r['y']}"


def okey(m, opp) -> str:
    """e_runner._okey's odour id ("ELECTRIC|DRAGON+FLYING")."""
    return f"{m}|{'+'.join(opp)}"


def calibration_odours(rc: dict, cb, rule: str) -> tuple:
    """({id: odour}, single-type ids, dual-type ids) over every reachable (move type, opponent types)."""
    st, _, _, move = pool_vocabulary()
    odours, single, dual = {}, [], []
    for m, o in reachable(st, move):
        oid = okey(m, o)
        odours[oid] = odour(rc, cb, m, o, rule)
        (single if len(o) == 1 else dual).append(oid)
    return odours, single, dual


def cap_at(rc: dict, cb, rule: str, s: float, max_rate_hz: float, cap_hz: float) -> bool:
    """Encoder 4.3 condition 4 (computed, lever-independent): no glomerulus of any reachable odour above the cap."""
    st, _, _, move = pool_vocabulary()
    return bool(cap_ok(rc, cb, st, move, rule, s, max_rate_hz, cap_hz))


def even_rows(pops, rc: dict, enc: dict, spec) -> list:
    rows = e_pairs.attach_odours(e_pairs.even_situations(pops), rc, q_pairs.codebook(enc, spec),
                                 E.dual_rule(spec.config))
    nb, na = sum(r["axis"] == "b" for r in rows), sum(r["axis"] == "a" for r in rows)
    if (nb, na) != (spec.n_b, spec.n_a_even):
        raise ValueError(f"even rows (b) {nb} · (a) {na}, declared {spec.n_b} · {spec.n_a_even}")
    return rows


def check_set(js: dict, enc: dict, spec) -> list:
    """R.3: the generated set's digests, n_a and last turn equal R's declared values and block set's; [] when so."""
    st = enc["set"]
    bad = []
    for k, want in (("digest_e0_b", spec.digest_e0_b), ("digest_e0_a", spec.digest_e0_a),
                    ("digest_keys", spec.digest_keys), ("n_a", spec.n_a), ("last_turn", spec.last_turn)):
        if js[k] != want or st[k] != want:
            bad.append(f"judgement set {k}: generated {js[k]!r}, block set {st[k]!r}, declared {want!r}")
    if js["status"] != "OK" or len(js["b"]) != spec.n_b or len(js["a"]) != spec.n_a:
        bad.append(f"judgement set status {js['status']}, (b) {len(js['b'])}, (a) {len(js['a'])}")
    return bad


def judgement_rows(pops, rc: dict, enc: dict, spec) -> list:
    """The judgement set's (b) 21 then (a) 32 rows with E-grid k2-norm odours; ValueError on any mismatch."""
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (R.8)")
    js = e_pairs.judgement_set(pops, E)
    bad = check_set(js, enc, spec)
    if bad:
        raise ValueError("; ".join(bad))
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))


def cond_odours(row: dict, cond) -> tuple:
    if cond.odour == "egrid":
        return row["odor_x"], row["odor_y"]
    if cond.odour == "e0":
        return row["odor_x_e0"], row["odor_y_e0"]
    raise ValueError(f"unknown odour kind {cond.odour!r}")


def encoder_activity_ref(oid: str, odour_: dict, params, n_kc: int, enc_code: dict, spec) -> str:
    """The basename of encoder ③'s activity entry for this odour (k2-norm, s = spec.strength, the strength seeds):
    EMeasurer's own key over its own inputs — computed, nothing is read or written."""
    from ..agent.e_measure import EMeasurer
    from ..agent.e_store import ECache
    m = EMeasurer(None, ECache(f"{E.raw_dir}/cache", enc_code), params, n_kc, E, guard_params=False)
    return m.cache._path("activity", m._act_inputs(oid, odour_, spec.strength, E.strength_seeds)).name


def p_reference(cache_dir, measure_key: str, wanted: set) -> dict:
    """{(direction, arm, seed): result} of P's committed run: kind p_arm and code_key = block p's measure key; an
    unreadable file is skipped."""
    out = {}
    for f in sorted(Path(cache_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("kind") != "p_arm" or d.get("code_key") != measure_key:
            continue
        i = d["inputs"]
        k = (i["direction"], i["arm"], int(i["seed"]))
        if k in wanted:
            out[k] = d["result"]
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_r_pairs.py tests/brain/test_r_spec.py -q`
Expected: PASS (the `needs_npz` tests run: `data/malecns.npz` exists in this worktree).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/r_pairs.py tests/brain/test_r_pairs.py
git commit -m "feat(r): r_pairs — calibration odours, even rows, digest-checked judgement rows, repro references"
```

---
### Task 3: `r_jobs` — edit-capable KC activity, P arm and oracle jobs on Q's rig

**Files:**
- Create: `flymon/brain/r_jobs.py`
- Test: `tests/brain/test_r_jobs.py`

**Interfaces:**
- Consumes: `q_jobs.q_rig(conn, pops, params, edit, p_type) -> (Engine, Plasticity, comps, sha, n_edges)`, `q_jobs.q_oracle_job`, `h4_jobs._present_kc` / `type_cells`, `n_jobs._present` / `readout_cells`, `o_jobs.DaMeter` / `dopamine_zero` / `train_x` / `weights_sha256`, `presentation.decide`.
- Produces (all module-level, FlyPool signature `fn(engine, plasticity, pops, comps, readout, **kwargs)`):
  - `kc_activity_job(..., params, edit, p_type, items, strength, settle_ms, read_ms, window_ms) -> list[dict(i, seed, kc, n, max_win, csc_sha256, edit_edges)]`;
  - `r_arm_job(..., params, edit, p_type, odor_x, odor_y, seed, arm, punish, plastic, da_zero, readout, punish_type, reward_type, strength, settle_ms, read_ms, window_ms, trials, present_ms, gap_ms, train_settle_ms, seed_base, seed_stride) -> dict` (o_jobs.arm_job's keys + `r = {edit_edges, p_type}`);
  - `r_oracle_job(..., **q_oracle_job kwargs) -> dict` (q_oracle_job's result + `r = {p_type, n_cells, p_cells: {x, y}: [[per-cell count] per report seed], read_ms, dt, refrac_steps}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_r_jobs.py
"""R's worker jobs on q_jobs.q_rig (R.9.6, Reading 4/5): with edit "none" kc_activity_job is k_jobs.activity_job,
r_arm_job is o_jobs.arm_job (minus wall_s) and r_oracle_job minus "r" is q_oracle_job (whose result minus "q" is
h4_jobs.oracle_job with no fixed arms); under apl_to_mbon05_zero every job records the edited edge count and a new CSC
sha; r_oracle_job's per-cell naive probe sums to the oracle's pre counts of the P type."""
import json
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import k_jobs as K
from flymon.brain import n_jobs as N
from flymon.brain import o_jobs as O
from flymon.brain import q_jobs as Q
from flymon.brain import r_jobs as R
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.stimuli import design_odor_pair

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")            # synthetic PPL105 core, then PAM08 core
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), punish_type="PPL105", reward_type="PAM08", **W)
ARM = dict(readout=READOUT, punish_type="PPL105", reward_type="PAM08", trials=2, present_ms=300.0, gap_ms=50.0,
           train_settle_ms=100.0, seed_base=1_000_000, seed_stride=1000, **W)
FLAGS = {name: dict(punish=pu, plastic=pl, da_zero=dz) for name, pu, pl, dz in O_SPEC.o2_arms}
X, Y = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    t = np.asarray(c.type).astype(str)
    apl = int(np.flatnonzero(t == "APL")[0])
    m1 = np.flatnonzero(t == "MBON01")
    c = replace(c, pre=np.append(c.pre, np.full(len(m1), apl, np.int32)), post=np.append(c.post, m1.astype(np.int32)),
                w=np.append(c.w, np.full(len(m1), 20, np.int32)))      # APL -> every MBON01 cell
    Q._RIG.clear(); H4._RIG.clear(); O._RIG.clear(); N._RIG.clear()
    return c, Populations.from_connectome(c)


def _canon(x):
    return json.dumps(x, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "wall_s"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _lever_edges(c, pops) -> int:
    return Q.apply_q_edit(Engine(c, pops, P, seed=0), pops, Q.MBON05, "MBON01")[1]


def test_kc_activity_none_is_k_jobs_activity(conn_pops):
    c, pops = conn_pops
    items = [(0, X, 500), (1, Y, 501), (2, X, 502)]
    ref = K.activity_job(_Stub(c), None, pops, None, None, params=P, items=items, **W)
    got = R.kc_activity_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", items=items, **W)
    assert [{k: v for k, v in g.items() if k not in ("csc_sha256", "edit_edges")} for g in got] == ref
    assert {g["edit_edges"] for g in got} == {0} and len({g["csc_sha256"] for g in got}) == 1
    assert any(g["kc"] for g in got), "the synthetic regime must drive KCs"


def test_kc_activity_under_the_lever_records_the_edit(conn_pops):
    c, pops = conn_pops
    n = _lever_edges(c, pops)
    items = [(0, X, 500)]
    none = R.kc_activity_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", items=items, **W)
    lev = R.kc_activity_job(_Stub(c), None, pops, None, None, params=P, edit=Q.MBON05, p_type="MBON01", items=items, **W)
    assert n > 0 and lev[0]["edit_edges"] == n and lev[0]["csc_sha256"] != none[0]["csc_sha256"]


@pytest.mark.parametrize("arm", ["plastic", "frozen", "punish"])
def test_r_arm_none_is_o_arm_job(conn_pops, arm):
    c, pops = conn_pops
    ref = O.arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=X, odor_y=Y, seed=9, arm=arm,
                    **FLAGS[arm], **ARM)
    got = R.r_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", odor_x=X, odor_y=Y,
                      seed=9, arm=arm, **FLAGS[arm], **ARM)
    assert got["r"] == {"edit_edges": 0, "p_type": "MBON01"}
    assert _canon(_strip({k: v for k, v in got.items() if k != "r"})) == _canon(_strip(ref))


def test_r_arm_under_the_lever(conn_pops):
    c, pops = conn_pops
    n = _lever_edges(c, pops)
    none = R.r_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", odor_x=X, odor_y=Y,
                       seed=9, arm="punish", **FLAGS["punish"], **ARM)
    got = R.r_arm_job(_Stub(c), None, pops, None, None, params=P, edit=Q.MBON05, p_type="MBON01", odor_x=X, odor_y=Y,
                      seed=9, arm="punish", **FLAGS["punish"], **ARM)
    assert got["edit"] == Q.MBON05 and got["r"]["edit_edges"] == n and got["csc_sha256"] != none["csc_sha256"]


def test_q_oracle_without_fixed_arms_is_oracle_job(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    ref = H4.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, fixed_alphas=(),
                         active_fx=0.5, **ORACLE)
    assert _canon({k: v for k, v in got.items() if k != "q"}) == _canon(ref) and got["q"]["fixed"] == {}


@pytest.mark.parametrize("edit", ["none", Q.MBON05])
def test_r_oracle_is_q_oracle_plus_a_per_cell_naive_probe(conn_pops, edit):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    kw = dict(params=P, edit=edit, odor_x=a, odor_y=b, fixed_alphas=(), active_fx=0.5, **ORACLE)
    ref = Q.q_oracle_job(_Stub(c), None, pops, None, None, **kw)
    got = R.r_oracle_job(_Stub(c), None, pops, None, None, **kw)
    assert _canon({k: v for k, v in got.items() if k != "r"}) == _canon(ref)
    r = got["r"]
    t = np.asarray(c.type).astype(str)
    assert r["p_type"] == "MBON01" and r["n_cells"] == int((t == "MBON01").sum())
    assert r["read_ms"] == 100.0 and r["dt"] == P.dt and r["refrac_steps"] == P.refrac_steps()
    pre = got["report"]["pre"]["P"]
    for j, side in enumerate(("x", "y")):
        assert len(r["p_cells"][side]) == 3
        assert [sum(cells) for cells in r["p_cells"][side]] == [row[j] for row in pre]
    assert any(any(row) for row in pre), "the synthetic regime must drive the readout"
    if edit == Q.MBON05:
        assert got["q"]["edit_edges"] == _lever_edges(c, pops)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_jobs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_jobs'`.

- [ ] **Step 3: Write the implementation**

```python
# flymon/brain/r_jobs.py
"""FlyPool worker jobs of spec appendix R (R.9.6, R.9.7). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle them; the worker's engine only lends its connectome. Every job
takes its rig from q_jobs.q_rig(conn, pops, params, edit, p_type) — Q's edit-capable rig (built as h4_jobs.rig_for /
o_jobs.rig build theirs, then the edit in place; Q's reproduction gate 02650a3 pinned it), cached per worker under
(Params, edit, P type). q_jobs, k_jobs, o_jobs, n_jobs and h4_jobs are imported, never edited.
- kc_activity_job: k_jobs.activity_job's body (h4_jobs._present_kc per item, plasticity off, weights reset) on Q's
  rig; each row also carries the CSC sha and the edited edge count (gate ①; with edit "none" it is activity_job's
  rows — test — and reproduces encoder ③'s entries, R.9.6).
- r_arm_job: o_jobs.arm_job copied onto Q's rig (O's rig knows no apl_to_mbon05_zero); with edit "none" it is
  arm_job's result (minus wall_s) plus "r" (test; the reproduction gate checks it against P's committed entries).
- r_oracle_job: q_jobs.q_oracle_job called unchanged (R.9.6: Q's verified oracle path), then the P readout type's
  per-cell naive probe on the report seeds — presentation.decide with idx = the type's cells, the weights at w0, the
  oracle's own pre-probe sequence, so the per-cell counts sum to report.pre.P (R.9.7 per-cell saturation)."""
from __future__ import annotations

import contextlib
import time

import numpy as np

from . import o_jobs, q_jobs
from .h4_jobs import _present_kc, type_cells
from .n_jobs import _present, readout_cells
from .presentation import decide


def kc_activity_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, items, strength: float,
                    settle_ms: float, read_ms: float, window_ms: int) -> list:
    e, p, _, sha, n_edit = q_jobs.q_rig(eng.conn, pops, params, edit, p_type)
    out = []
    try:
        p.reset_weights()
        p.set_enabled(False)
        for i, odor, seed in items:
            o = _present_kc(e, p, pops, odor, int(seed), strength, settle_ms, read_ms, int(window_ms))
            nz = np.flatnonzero(o["read"])
            out.append(dict(i=int(i), seed=int(seed), kc=nz.tolist(), n=o["read"][nz].astype(int).tolist(),
                            max_win=int(o["max_win"]), csc_sha256=sha, edit_edges=int(n_edit)))
    finally:
        p.reset_weights()
        p.set_enabled(True)
    return out


def r_arm_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, odor_x: dict, odor_y: dict, seed: int,
              arm: str, punish: bool, plastic: bool, da_zero: bool, readout: dict, punish_type: str, reward_type: str,
              strength: float, settle_ms: float, read_ms: float, window_ms: int, trials: int, present_ms: float,
              gap_ms: float, train_settle_ms: float, seed_base: int, seed_stride: int) -> dict:
    """o_jobs:arm_job (copied; O2's one (X, arm, seed)) on Q's rig. Probes of X and Y at `seed` (plasticity off),
    training of X alone (punishment iff `punish`, weights frozen unless `plastic`, the rule's dopamine at 0 iff
    `da_zero`), the probes again, the weight fractions, the phasic dopamine per compartment, and the plastic-weight
    sha256 before and after. Weights, the step hook and the rule's dopamine weights are restored."""
    e, p, c, sha, n_edit = q_jobs.q_rig(eng.conn, pops, params, edit, p_type)
    a_cells, p_cells = readout_cells(e.conn, readout)
    a_core, p_core = c[punish_type].core, c[reward_type].core
    t0 = time.perf_counter()

    def probes():
        was = p.enabled
        p.set_enabled(False)
        try:
            return {k: _present(e, p, pops, o, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells)[0]
                    for k, o in (("x", odor_x), ("y", odor_y))}
        finally:
            p.set_enabled(was)

    meter = o_jobs.DaMeter(e, p)
    try:
        p.reset_weights()
        w0_sha = o_jobs.weights_sha256(p.w0)
        pre = probes()
        p.set_enabled(bool(plastic))
        with meter, (o_jobs.dopamine_zero(p) if da_zero else contextlib.nullcontext()):
            o_jobs.train_x(e, p, pops, odor_x, strength, int(seed), punish_type if punish else None, trials,
                           present_ms, gap_ms, train_settle_ms, seed_base, seed_stride, meter)
        p.set_enabled(True)
        post = probes()
        w_post = o_jobs.weights_sha256(e.csc.w[p.edges])
        wf = float(p.weights_frac())
        wa, wp = float(p.weights_frac_by_mbon_set(a_core)), float(p.weights_frac_by_mbon_set(p_core))
        da = meter.by_compartment(c)
    finally:
        e.on_step = p.on_step
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), edit=edit, arm=arm, punish=bool(punish), plastic=bool(plastic),
                da_zero=bool(da_zero), csc_sha256=sha, pre=pre, post=post, weights_frac=wf, weights_frac_A=wa,
                weights_frac_P=wp, w0_sha256=w0_sha, w_post_sha256=w_post, da_integral=da,
                wall_s=time.perf_counter() - t0, r=dict(edit_edges=int(n_edit), p_type=p_type))


def r_oracle_job(eng, pl, pops, comps, ro, **kw) -> dict:
    """q_jobs.q_oracle_job(**kw) unchanged, then "r": the P readout type's per-cell read-window counts of X and Y on
    every report seed with the weights at w0 (q_oracle_job leaves them there), plus what the saturation record needs
    (read window, dt, refractory steps of the job's Params)."""
    res = q_jobs.q_oracle_job(eng, pl, pops, comps, ro, **kw)
    p_type = kw["readout"]["P"]
    e, p, _, _, _ = q_jobs.q_rig(eng.conn, pops, kw["params"], kw["edit"], p_type)
    cells = type_cells(e.conn, [p_type])[p_type]
    xs, ys = [], []
    try:
        p.reset_weights()
        for s in kw["report_seeds"]:
            cnt = decide(e, p, pops, [kw["odor_x"], kw["odor_y"]], kw["strength"], int(s), kw["settle_ms"],
                         kw["read_ms"], idx=cells)
            xs.append(np.asarray(cnt[0]).astype(int).tolist())
            ys.append(np.asarray(cnt[1]).astype(int).tolist())
    finally:
        p.reset_weights()
        p.set_enabled(True)
    res["r"] = dict(p_type=p_type, n_cells=int(cells.size), p_cells={"x": xs, "y": ys}, read_ms=float(kw["read_ms"]),
                    dt=float(e.p.dt), refrac_steps=int(e.p.refrac_steps()))
    return res
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_r_jobs.py tests/brain/test_q_jobs.py tests/brain/test_o_jobs_arms.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/r_jobs.py tests/brain/test_r_jobs.py
git commit -m "feat(r): r_jobs — edit-capable KC activity, P arm and oracle (+ per-cell MBON05 probe) jobs on q_rig"
```

---

### Task 4: `r_store` and `r_measure` — guard, cache, manifest, archive; the measurer with resume

**Files:**
- Create: `flymon/brain/r_store.py`, `flymon/brain/r_measure.py`
- Test: `tests/brain/test_r_store_measure.py`

**Interfaces:**
- Consumes: Task 1 `SPEC` / `smoke`; Task 2 `row_key`, `cond_odours`; Task 3 jobs; `e_store.ECache`; `h3_store.canonical` / `canonical_pretty` / `sha256_file`; `pool_bench` guards; `q_measure.Q_MEASURE_FILES`; `p_measure.MEASURE_FILES`; `e_measure.MAX_ITEMS`.
- Produces:
  - `r_store`:
    - `ALLOWED_DIR`, `READ_ONLY_DIR`, `SUMMARY`, `SMOKE_SEEDS`;
    - `guard(path, params_list)`, `write_bytes`, `write_json`, `read_summary`, `write_summary_block(path, block, obj, params_list)`, `move_block(path, src, dst, params_list)`;
    - `RCache(root, code, smoke_seeds=SMOKE_SEEDS)` with `key`, `_path`, `get`, `put`;
    - `load_manifest(manifest) -> (got, bad)`;
    - `archive_copy(files, dest, root) -> list[dict(src, dst, sha256)]`.
  - `r_measure`:
    - `R_MEASURE_FILES`;
    - `RMeasurer(pool, cache, spec, params, readout, z, types, n_kc)` with:
      - `activity(odours, edit, s, seeds, block) -> {oid: {frac, max_win, csc_sha256: [..], edit_edges: [..]}}`;
      - `kwargs(row, cond, seeds) -> dict`;
      - `inputs(row, cond, block, seeds) -> dict`;
      - `oracle(rows, cond, block, seeds) -> [dict(key, result, cache_key, cache_file)]`;
      - `arm_common(nspec, readout, punish_type) -> dict`;
      - `arms(items, readout, punish_type, block, nspec) -> rows` (with direction / x / y / point);
      - `last_wall_s`, `last_jobs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_r_store_measure.py
"""R's writer and measurer (R.5, R.8, R.9.7): writes only under results/r/ (results/r/ref/ read-only) and
results/summary/r_lever.json, atomically; RCache keeps smoke and real entries apart and treats a truncated or foreign
entry as missing; load_manifest re-checks every raw file's sha256 and key; the archive copy goes only under its root,
never over an existing directory. The measurer writes one entry per unit and resumes with only the missing units; its
oracle inputs are the encoder's (plus edit / no fixed arm / active_fx) and its arm inputs are P's (plus p_type)."""
import hashlib
import json
from pathlib import Path

import pytest

from flymon.agent.e_measure import EMeasurer
from flymon.agent.e_spec import SPEC as E
from flymon.agent.e_store import ECache
from flymon.brain import p_measure, r_jobs
from flymon.brain import r_store as S
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.r_spec import SPEC

READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]


def test_guard_and_summary_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/q/x.json", "results/p/x.json", "results/encoder/x.json", "results/r/ref/activity/x.json",
                "results/summary/q_reward.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            S.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(S.write_json("results/r/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    S.write_summary_block(S.SUMMARY, "repro", {"n": 1}, [])
    S.write_summary_block(S.SUMMARY, "smoke", {"m": 2}, [])
    S.move_block(S.SUMMARY, "smoke", "smoke_old", [])
    assert S.read_summary() == {"repro": {"n": 1}, "smoke_old": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))


def test_cache_scope_and_bad_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = S.RCache("results/r/cache", {"key": "k"}), S.RCache("results/r/smoke/cache", {"key": "k"})
    real.put("r_oracle", {"act_seeds": [500]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [500]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_009_100})                   # a smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [500]})                  # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [25_008_000, 500]})          # mixed
    sm.put("r_arm", {"seed": 25_009_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_009_100}) == {"v": 2}
    p = real._path("r_oracle", {"act_seeds": [500]})
    p.write_text("{trunc")
    assert real.get("r_oracle", {"act_seeds": [500]}) is None
    p.write_text(json.dumps({"key": "other", "kind": "r_oracle", "result": {}}))
    assert real.get("r_oracle", {"act_seeds": [500]}) is None


def test_load_manifest_rechecks_sha_and_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = Path("results/r/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    man = [dict(key="b|0|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))]
    got, bad = S.load_manifest(man)
    assert bad == [] and got == [dict(key="b|0|x|y", result={"v": 1}, cache_key="K", cache_file=str(f))]
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 2}}))
    assert S.load_manifest(man)[1] and S.load_manifest([dict(man[0], cache_file="results/r/none.json")])[1]


def test_archive_copy_only_under_its_root_and_never_over(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = Path("results/r/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text("x")
    root = tmp_path / "arch"
    out = S.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == "x" and out[0]["sha256"] == sha256_file(f)
    assert Path(out[0]["dst"]).parent.name == "r_oracle"
    with pytest.raises(SystemExit):
        S.archive_copy([str(f)], root / "seal1", root)            # exists
    with pytest.raises(SystemExit):
        S.archive_copy([str(f)], tmp_path / "elsewhere", root)    # outside the root


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is r_jobs.kc_activity_job:
            return [[dict(i=i, seed=s, kc=[0, 1], n=[1, 1], max_win=3, csc_sha256="sha-L", edit_edges=2)
                     for i, _, s in kw["items"]] for kw in kws]
        if fn is r_jobs.r_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


def _m(tmp_path, monkeypatch, n=2):
    monkeypatch.chdir(tmp_path)
    pool = FakePool(n)
    return RMeasurer(pool, S.RCache("results/r/cache", {"key": "k"}), SPEC, Params(), READOUT, Z, TYPES, 100), pool


ROWS = [dict(axis="b", turn=0, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_activity_writes_one_entry_per_odour_and_resumes(tmp_path, monkeypatch):
    m, pool = _m(tmp_path, monkeypatch)
    od = {f"o{i}": {"G": 1.0} for i in range(3)}
    got = m.activity(od, SPEC.lever_edit, 1.0, (500, 501, 502, 503), "gate1")
    assert set(got) == set(od) and got["o0"]["frac"] == [0.02] * 4 and got["o0"]["edit_edges"] == [2]
    assert got["o0"]["csc_sha256"] == ["sha-L"] and got["o0"]["max_win"] == [3] * 4
    n = len(pool.calls)
    assert m.activity(od, SPEC.lever_edit, 1.0, (500, 501, 502, 503), "gate1") == got and len(pool.calls) == n


def test_oracle_resumes_only_missing_pairs(tmp_path, monkeypatch):
    m, pool = _m(tmp_path, monkeypatch)
    L = SPEC.cond("L")
    got = m.oracle(ROWS, L, "jm", SPEC.h4_seeds())
    assert [g["key"] for g in got] == ["b|0|x0|y0", "b|0|x1|y1", "b|0|x2|y2"]
    assert all(Path(g["cache_file"]).exists() for g in got) and got[0]["result"]["edit"] == SPEC.lever_edit
    Path(got[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = m.oracle(ROWS, L, "jm", SPEC.h4_seeds())
    assert pool.calls == [("r_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in got]
    e0 = m.oracle(ROWS[:1], SPEC.cond("E0"), "jm", SPEC.h4_seeds())
    assert e0[0]["result"]["echo"] == {"E": 1.0}


def test_oracle_inputs_are_the_encoders_plus_edit(tmp_path, monkeypatch):
    m, _ = _m(tmp_path, monkeypatch)
    em = EMeasurer(None, ECache("results/encoder/cache", {"key": "k"}), Params(), 100, E)
    ref = em._oracle_kw(ROWS[0], 1.0, READOUT, Z, TYPES, SPEC.h4_seeds())
    mine = m.kwargs(ROWS[0], SPEC.cond("C"), SPEC.h4_seeds())
    assert {k: v for k, v in mine.items() if k not in ("edit", "fixed_alphas", "active_fx")} == ref
    assert mine["edit"] == "none" and mine["fixed_alphas"] == [] and mine["active_fx"] == SPEC.active_fx


class _PassCache:
    def get_or_compute(self, kind, inputs, compute, params_list):
        return compute()


def test_arm_inputs_are_ps_plus_p_type(tmp_path, monkeypatch):
    m, _ = _m(tmp_path, monkeypatch)
    seen = []

    class Capture:
        n_workers = 1

        def run_jobs(self, fn, kws):
            seen.extend(kws)
            return [{} for _ in kws]

    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=23_000_000, point=(0.25, 8.0))
    p_measure.PMeasurer(Capture(), P_SPEC, _PassCache()).p_arms(Params(), [item], READOUT, "PPL105")
    common = m.arm_common(P_SPEC.o.n, READOUT, "PPL105")
    assert {k: v for k, v in common.items() if k != "p_type"} == {k: v for k, v in seen[0].items()
                                                                  if k in common}
    assert set(seen[0]) - set(common) == {"edit", "odor_x", "odor_y", "seed", "arm", "punish", "plastic", "da_zero"}
    rows = m.arms([item], READOUT, "PPL105", "repro", P_SPEC.o.n)
    assert rows[0]["direction"] == "r1" and rows[0]["point"] == [0.25, 8.0] and rows[0]["r"] == {"edit_edges": 0}


def test_measure_files_cover_every_job_module():
    for f in ("flymon/brain/q_jobs.py", "flymon/brain/o_jobs.py", "flymon/brain/n_jobs.py", "flymon/brain/h4_jobs.py",
              "flymon/brain/r_jobs.py", "flymon/brain/r_measure.py", "flymon/brain/r_store.py"):
        assert f in R_MEASURE_FILES, f
    assert set(p_measure.MEASURE_FILES) <= set(R_MEASURE_FILES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_store_measure.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_store'`.

- [ ] **Step 3: Write `r_store`**

```python
# flymon/brain/r_store.py
"""R's only writer (R.5, R.8, R.9.7): raw files under results/r/ (results/r/ref/ — the encoder ③ entries the
controller copies in — is read-only), the one summary results/summary/r_lever.json, atomic writes (temporary file +
rename), RCache (e_store.ECache's key; smoke and real entries never share a root; the key and inputs stored in every
entry; an unreadable or foreign entry is missing), load_manifest (every raw file's sha256 and key re-checked) and the
archive copy (only under the archive root, never over an existing directory, each copy's sha256 checked)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from ..agent.e_store import ECache
from .h3_store import canonical, canonical_pretty, sha256_file
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_spec import SPEC as R_SPEC
from .r_spec import smoke

ALLOWED_DIR = "results/r/"
READ_ONLY_DIR = "results/r/ref/"
SUMMARY = "results/summary/r_lever.json"
SMOKE_SEEDS = frozenset(R_SPEC.smoke_seeds) | frozenset(smoke(R_SPEC).p.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if rel.startswith(READ_ONLY_DIR) or not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"R writes only under {ALLOWED_DIR} (not {READ_ONLY_DIR}) and {SUMMARY}, not {path}")


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(obj) + "\n").encode(), params_list)


def read_summary(path=SUMMARY) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def write_summary_block(path, block: str, obj, params_list) -> Path:
    doc = read_summary(path)
    doc[block] = obj
    return write_json(path, doc, params_list)


def move_block(path, src: str, dst: str, params_list) -> Path:
    """Gate ②'s one rerun (R.5): block src is kept as dst."""
    doc = read_summary(path)
    doc[dst] = doc.pop(src)
    return write_json(path, doc, params_list)


def _seeds(inputs) -> list:
    out = []
    for k, v in dict(inputs).items():
        if k == "seed":
            out.append(int(v))
        elif k == "seeds" or k.endswith("_seeds"):
            out += [int(s) for s in v]
    return out


class RCache(ECache):
    """A root with a path component "smoke" takes only smoke seeds, any other root takes none (SystemExit 2)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code)
        self.smoke_seeds = frozenset(int(s) for s in smoke_seeds)

    def _scope(self, inputs) -> None:
        seeds = _seeds(inputs)
        is_smoke_root = "smoke" in Path(self.root).parts
        some = any(s in self.smoke_seeds for s in seeds)
        if seeds and (some != is_smoke_root or some != all(s in self.smoke_seeds for s in seeds)):
            _refuse(f"{self.root}: smoke seeds belong only under a smoke root, real seeds only outside it")

    def _path(self, kind, inputs):
        self._scope(inputs)
        return super()._path(kind, inputs)

    def get(self, kind, inputs):
        p = self._path(kind, inputs)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        if d.get("key") != self.key(kind, inputs) or d.get("kind") != kind:
            return None
        return d["result"]

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


def load_manifest(manifest: list) -> tuple:
    """([dict(key, result, cache_key, cache_file)], [reasons]) — a missing file, a sha256 other than the manifest's or
    a stored key other than the manifest's cache_key is a reason, and that entry is not returned."""
    got, bad = [], []
    for m in manifest:
        p = Path(m["cache_file"])
        if not p.exists():
            bad.append(f"{m['key']}: raw file {p} is missing")
            continue
        if sha256_file(p) != m["sha256"]:
            bad.append(f"{m['key']}: raw file {p} sha256 differs from the manifest")
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            bad.append(f"{m['key']}: raw file {p} is unreadable")
            continue
        if d.get("key") != m["cache_key"]:
            bad.append(f"{m['key']}: raw file {p} holds another key")
            continue
        got.append(dict(key=m["key"], result=d["result"], cache_key=m["cache_key"], cache_file=str(p)))
    return got, bad


def archive_copy(files: list, dest, root) -> list:
    """R.9.7: one copy of every raw file under dest/<kind>/<basename>; dest must be a new directory under root."""
    root_r = Path(os.path.expanduser(str(root))).resolve()
    dest_p = Path(os.path.expanduser(str(dest)))
    dest_r = dest_p.resolve()
    if dest_r == root_r or root_r not in dest_r.parents:
        _refuse(f"the archive copy goes only under {root_r}, not {dest_r}")
    if dest_p.exists():
        _refuse(f"archive directory {dest_p} exists; an archive is never overwritten")
    out = []
    for f in files:
        src = Path(f)
        dst = dest_p / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sha = sha256_file(src)
        if sha256_file(dst) != sha:
            _refuse(f"archive copy of {src} does not match its source")
        out.append(dict(src=str(src), dst=str(dst), sha256=sha))
    return out
```

- [ ] **Step 4: Write `r_measure`**

```python
# flymon/brain/r_measure.py
"""R's measurer (R.2, R.3, R.5, R.9.6): the three R jobs behind RCache, one atomic entry per unit, written as soon as
its worker round returns, so an interrupted stage resumes with only the missing units and never returns a partial
list.
- activity: kc_activity_job over (odour, seed) items in jobs of e_measure.MAX_ITEMS, one "r_act" entry per odour
  (EMeasurer.activity's batching and frac = active KCs / n_kc, with the edit, its sha and edge count added).
- oracle: r_oracle_job, one "r_oracle" entry per (pair, condition, block); the kwargs are the encoder's oracle inputs
  (EMeasurer._oracle_kw, test) plus edit, fixed_alphas () and active_fx; the condition picks the odours (E-grid / E0)
  and the strength.
- arms: r_arm_job, one "r_arm" entry per (direction, arm, seed); the kwargs are PMeasurer.p_arms' (test) plus p_type."""
from __future__ import annotations

import sys
import time

from ..agent.e_measure import MAX_ITEMS
from . import p_measure, r_jobs
from .q_measure import Q_MEASURE_FILES
from .r_pairs import cond_odours, row_key

# The cache code key: everything the jobs' results depend on — Q's list (the encoder's engine files, q_jobs, o_jobs,
# n_jobs, n_spec), P's (o_jobs' training, n_measure's windows), and R's jobs, measurer and store.
R_MEASURE_FILES = tuple(dict.fromkeys(Q_MEASURE_FILES + tuple(p_measure.MEASURE_FILES) + (
    "flymon/brain/r_jobs.py", "flymon/brain/r_measure.py", "flymon/brain/r_store.py")))


def _odour(o: dict) -> dict:
    return {str(g): float(v) for g, v in o.items()}


class RMeasurer:
    def __init__(self, pool, cache, spec, params, readout: dict, z: dict, types, n_kc: int):
        self.pool, self.cache, self.spec, self.params = pool, cache, spec, params
        self.readout, self.z, self.types, self.n_kc = dict(readout), dict(z), list(types), int(n_kc)
        self.last_wall_s, self.last_jobs = 0.0, 0

    def _windows(self) -> dict:
        sp = self.spec
        return dict(settle_ms=float(sp.settle_ms), read_ms=float(sp.read_ms), window_ms=int(sp.window_ms))

    # ---- KC activity (gate ①, repro, smoke) -------------------------------------------------------------------------
    def _act_inputs(self, oid, odour, edit, s, seeds, block) -> dict:
        return dict(params=self.params, edit=edit, p_type=self.spec.p_type, odour_id=str(oid), odour=_odour(odour),
                    strength=float(s), seeds=[int(x) for x in seeds], block=block, **self._windows())

    def activity(self, odours: dict, edit: str, s: float, seeds, block: str) -> dict:
        seeds = [int(x) for x in seeds]
        ins = {oid: self._act_inputs(oid, o, edit, s, seeds, block) for oid, o in odours.items()}
        todo = [oid for oid in odours if self.cache.get("r_act", ins[oid]) is None]
        t0 = time.perf_counter()
        if todo:
            print(f"r activity {block}: {len(todo)}/{len(odours)} odours to measure", file=sys.stderr)
            entries = [(oid, odours[oid]) for oid in todo]
            items = [(k * len(seeds) + j, _odour(o), sd) for k, (_, o) in enumerate(entries) for j, sd in enumerate(seeds)]
            jobs = [items[a:a + MAX_ITEMS] for a in range(0, len(items), MAX_ITEMS)]
            common = dict(params=self.params, edit=edit, p_type=self.spec.p_type, strength=float(s), **self._windows())
            got, done = {}, set()
            n_w = max(1, int(self.pool.n_workers))
            for a in range(0, len(jobs), n_w):
                for part in self.pool.run_jobs(r_jobs.kc_activity_job, [dict(common, items=c) for c in jobs[a:a + n_w]]):
                    for r in part:
                        got[int(r["i"])] = r
                for k, (oid, _) in enumerate(entries):
                    idx = range(k * len(seeds), (k + 1) * len(seeds))
                    if k not in done and all(i in got for i in idx):
                        done.add(k)
                        rs = [got[i] for i in idx]
                        self.cache.put("r_act", ins[oid], dict(
                            frac=[len(r["kc"]) / self.n_kc for r in rs], max_win=[int(r["max_win"]) for r in rs],
                            csc_sha256=sorted({r["csc_sha256"] for r in rs}),
                            edit_edges=sorted({int(r["edit_edges"]) for r in rs})), [self.params])
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = {}
        for oid in odours:
            v = self.cache.get("r_act", ins[oid])
            if v is None:
                raise RuntimeError(f"r activity {block}: odour {oid} missing after the run")
            out[oid] = v
        return out

    # ---- the oracle (repro, smoke, even, judgement) --------------------------------------------------------------
    def kwargs(self, row: dict, cond, seeds: dict) -> dict:
        sp = self.spec
        ox, oy = cond_odours(row, cond)
        return dict(params=self.params, odor_x=_odour(ox), odor_y=_odour(oy), readout=dict(self.readout),
                    z={k: [float(v) for v in self.z[k]] for k in self.z}, types=[str(t) for t in self.types],
                    act_seeds=[int(s) for s in seeds["act"]], select_seeds=[int(s) for s in seeds["select"]],
                    report_seeds=[int(s) for s in seeds["report"]], alphas=[float(a) for a in sp.alphas],
                    strength=float(cond.strength), **self._windows(), punish_type=sp.punish_type,
                    reward_type=sp.reward_type, edit=cond.edit, fixed_alphas=[float(a) for a in sp.fixed_alphas],
                    active_fx=float(sp.active_fx))

    def inputs(self, row: dict, cond, block: str, seeds: dict) -> dict:
        return dict(self.kwargs(row, cond, seeds), pair=row_key(row), condition=cond.name, block=block)

    def oracle(self, rows: list, cond, block: str, seeds: dict) -> list:
        ins = [self.inputs(r, cond, block, seeds) for r in rows]
        todo = [i for i, x in enumerate(ins) if self.cache.get("r_oracle", x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        t0 = time.perf_counter()
        if todo:
            print(f"r oracle {cond.name}/{block}: {len(todo)}/{len(rows)} pairs to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            res = self.pool.run_jobs(r_jobs.r_oracle_job, [self.kwargs(rows[i], cond, seeds) for i in batch])
            for i, r in zip(batch, res):
                self.cache.put("r_oracle", ins[i], r, [self.params])
            print(f"r oracle {cond.name}/{block}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for r, x in zip(rows, ins):
            got = self.cache.get("r_oracle", x)
            if got is None:
                raise RuntimeError(f"r oracle {cond.name}/{block}: pair {row_key(r)} missing after the run")
            out.append(dict(key=row_key(r), result=got, cache_key=self.cache.key("r_oracle", x),
                            cache_file=str(self.cache._path("r_oracle", x))))
        return out

    # ---- P arms (repro, smoke, gate ②) ---------------------------------------------------------------------------
    def arm_common(self, nspec, readout: dict, punish_type: str) -> dict:
        """PMeasurer.p_arms' common kwargs (N2's training timings and seed rule, NMeasurer._window) plus p_type."""
        h4 = nspec.h4
        return dict(params=self.params, readout=dict(readout), punish_type=punish_type,
                    reward_type=nspec.h3.reward_type, trials=int(h4.teach_trials), present_ms=h4.teach_present_ms,
                    gap_ms=h4.teach_gap_ms, train_settle_ms=h4.teach_window.settle_ms,
                    seed_base=nspec.train_seed_base, seed_stride=nspec.train_seed_stride,
                    strength=nspec.h3.strength, settle_ms=h4.oracle_window.settle_ms,
                    read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms), p_type=self.spec.p_type)

    def arms(self, items: list, readout: dict, punish_type: str, block: str, nspec) -> list:
        common = self.arm_common(nspec, readout, punish_type)
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     arm=i["arm"], punish=bool(i["punish"]), plastic=bool(i["plastic"]), da_zero=bool(i["da_zero"]))
                for i in items]
        ins = [dict(j, block=block, direction=i["direction"], x=i["x"], y=i["y"],
                    point=[float(v) for v in i["point"]]) for j, i in zip(jobs, items)]
        todo = [k for k, x in enumerate(ins) if self.cache.get("r_arm", x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        t0 = time.perf_counter()
        if todo:
            print(f"r arms {block}: {len(todo)}/{len(items)} to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            for k, r in zip(batch, self.pool.run_jobs(r_jobs.r_arm_job, [jobs[k] for k in batch])):
                self.cache.put("r_arm", ins[k], r, [self.params])
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for x, i in zip(ins, items):
            got = self.cache.get("r_arm", x)
            if got is None:
                raise RuntimeError(f"r arms {block}: {i['direction']}/{i['arm']}/{i['seed']} missing after the run")
            out.append(dict(got, direction=i["direction"], x=i["x"], y=i["y"], point=[float(v) for v in i["point"]]))
        return out
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_r_store_measure.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/r_store.py flymon/brain/r_measure.py tests/brain/test_r_store_measure.py
git commit -m "feat(r): r_store and r_measure — guarded writes, RCache, manifest re-check, archive copy, resumable measurer"
```

---

### Task 5: `r_records` — per-pair values, raw checks, R.6 records, pre-read validity

**Files:**
- Create: `flymon/brain/r_records.py`, `tests/brain/r_fixtures.py`
- Test: `tests/brain/test_r_records.py`

**Interfaces:**
- Consumes: Task 1 `SPEC` / `Cond`; `h4_formula.pair_stats` / `arm_aggregate`; `h4_rules.z_constants`; `h4_spec.SPEC.z_ddof`; `e_rules.odour_activity` / `strength_ok`; `d6a.OVER_SPIKES` / `CONDITION`; `o_rules.finite`.
- Produces:
  - `strip_job(x, top=R_KEYS) -> dict`, `cell_sums_ok(res) -> bool`, `stats_ok(st) -> bool`;
  - `pair_row(key, res, st, spec) -> dict(key, axis, d_pre, r, p, m, testable, reward_pass, punish_pass, naive, alpha_reward, alpha_punish)`;
  - `aggregate(pairs, spec) -> arm_aggregate dict | None`, `counts(pairs) -> {b, a: {n, testable, reward_pass, punish_pass, naive}}`;
  - `raw_check(got, cond, expected, seeds) -> dict(reasons, edit_edges, csc_sha256, n_pairs, manifest)`;
  - `saturation(got, spec) -> dict | None`, `z_renorm(got, spec, z) -> dict`;
  - `cond_summary(got, cond, spec, z, expected, seeds) -> dict` (raw_check's fields + `pairs`, `aggregate`, `counts`, `kc_median`, `d6a_over_share`, `d6a_condition`, `apl_out_median`, `naive_A_median`, `alpha_reward`, `alpha_punish`, `saturation`, `z_renorm`, `condition`);
  - `transitions(to, frm) -> dict`, `compare(L, C, E0, spec) -> dict`;
  - `gate1_record(act, single, dual, enc_per, spec) -> dict`;
  - `preread_validity(blocks, raws, spec, z, expected, code_key, repro_sha) -> dict(reasons, invalid, checks)`.
  - Test fixtures (`tests/brain/r_fixtures.py`): `fake_oracle(r_ok, p_ok, bal, sha, edges, edit, n_rep, n_act, kc)`, `fake_rows(n_b, n_a, turn0)`, `got_for(rows, plan, ...)`.

- [ ] **Step 1: Write the fixtures**

```python
# tests/brain/r_fixtures.py
"""Fabricated oracle results for R's record / rule / runner tests (no engine). fake_oracle builds an r_oracle_job-shaped
result whose pair_stats under Z are fixed by three flags:
- r_ok: R1 lowers X's P by 10 (± 0/1) -> r ~ 18 (else P_X ± 3, mean 0 -> r ~ 0);
- p_ok: R2 lowers X's A by 10 (± 0/1) -> p ~ -18, -p >= 2 (else A_X ± 3 -> p ~ 0);
- bal: X's naive A = Y's (± noise, mean 0) -> d_pre = 0 (else X's A is 20 higher -> d_pre ~ 12).
testable <=> r_ok and p_ok; naive <=> bal; punish pass <=> p_ok; reward pass <=> r_ok. X's and Y's naive P move
together by H8 (so the P type count has a nonzero SD for z renormalisation and cancels in V_X - V_Y). The per-cell P
counts split the type count over 2 cells, so they sum to it."""
E8, F8, G8 = [1, -1, 2, -2, 1, -1, 2, -2], [0, 1, 0, 1, 0, 1, 0, 1], [3, -3, 3, -3, 3, -3, 3, -3]
H8 = [0, 2, 0, 2, 0, 2, 0, 2]
TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
READOUT = {"A": "MBON13", "P": "MBON05"}
Z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}


def _cyc(v, n):
    return [v[i % len(v)] for i in range(n)]


def fake_oracle(r_ok=False, p_ok=True, bal=False, sha="sha-C", edges=0, edit="none", n_rep=8, n_act=8, kc=0.05,
                px=26):
    e, f, g, h = _cyc(E8, n_rep), _cyc(F8, n_rep), _cyc(G8, n_rep), _cyc(H8, n_rep)
    ax = [(20 if bal else 40) + e[i] for i in range(n_rep)]
    pre = {"A": [[ax[i], 20] for i in range(n_rep)], "P": [[px + h[i], px + h[i]] for i in range(n_rep)]}
    px1 = [(px + h[i] - 10 + f[i]) if r_ok else (px + h[i] + g[i]) for i in range(n_rep)]
    r1 = {"A": [list(a) for a in pre["A"]], "P": [[px1[i], px + h[i]] for i in range(n_rep)]}
    ax2 = [(ax[i] - 10 + f[i]) if p_ok else (ax[i] + g[i]) for i in range(n_rep)]
    r2 = {"A": [[ax2[i], 20] for i in range(n_rep)], "P": [list(p) for p in r1["P"]]}
    side = {"frac": [kc] * n_act, "spikes": [100] * n_act, "max_win": [5] * n_act}

    def cells(j):
        return [[row[j] // 2, row[j] - row[j] // 2] for row in pre["P"]]

    return {"select": {}, "alpha_reward": 0.5, "alpha_punish": 0.8, "counts": {},
            "report": {"pre": pre, "R1": r1, "R2": r2},
            "kc": {"x": dict(side), "y": dict(side), "jaccard": 0.04},
            "q": {"edit": edit, "csc_sha256": sha, "edit_edges": edges, "fixed": {},
                  "apl_out": {"x": [0.2] * n_act, "y": [0.2] * n_act}, "fx": {"idx": [], "val": []},
                  "reach": {"W_X": 1.0, "n_edges": 3, "n_active_kc": 5, "f_X": 0.4, "by_type": {}, "total": 1.0}},
            "r": {"p_type": "MBON05", "n_cells": 2, "p_cells": {"x": cells(0), "y": cells(1)}, "read_ms": 600.0,
                  "dt": 1.0, "refrac_steps": 2}}


def fake_rows(n_b=21, n_a=32, turn0=64):
    rows = []
    for i in range(n_b):
        rows.append(dict(axis="b", turn=turn0 + i, x=f"bx{i}", y=f"by{i}", odor_x={"G": 1.0}, odor_y={"H": 1.0},
                         odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}))
    for i in range(n_a):
        rows.append(dict(axis="a", turn=turn0 + i, x=f"ax{i}", y=f"ay{i}", odor_x={"G": 1.0}, odor_y={"H": 1.0},
                         odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}))
    return rows


def key_of(r):
    return f"{r['axis']}|{int(r['turn'])}|{r['x']}|{r['y']}"


def got_for(rows, plan=None, default=(False, True, False), **kw):
    """[dict(key, result)] with fake_oracle(*plan.get(key, default), **kw)."""
    plan = plan or {}
    return [dict(key=key_of(r), result=fake_oracle(*plan.get(key_of(r), default), **kw), cache_key=f"ck{i}",
                 cache_file=f"results/r/cache/r_oracle/{i}.json") for i, r in enumerate(rows)]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/brain/test_r_records.py
"""R's records (R.4, R.6, R.9.5, R.9.7): pair rows from h4_formula.pair_stats; raw_check finds duplicates, missing
pairs, short probes, a wrong edit, broken per-cell sums and several CSC shas without any pair statistic; saturation is
per cell against 1000 / (refrac_steps × dt); z renormalisation is a record; transitions count C -> L per criterion;
gate ①'s record sits beside encoder ③'s values; pre-read validity flags L edges != 2 as INVALID and every other
defect as a reason."""
import copy

import pytest

from flymon.brain import r_records as RR
from flymon.brain.r_spec import SPEC
from tests.brain.r_fixtures import Z, fake_oracle, fake_rows, got_for, key_of

L, C, E0 = SPEC.conditions()
JS = SPEC.judge_seeds()
ROWS = fake_rows()
KEYS = [key_of(r) for r in ROWS]


def test_fixture_flags_fix_the_pair_statistics():
    from flymon.brain.h4_formula import pair_stats
    for r_ok in (False, True):
        for p_ok in (False, True):
            for bal in (False, True):
                st = pair_stats(fake_oracle(r_ok, p_ok, bal)["report"], Z, 2.0)
                row = RR.pair_row("b|64|x|y", fake_oracle(r_ok, p_ok, bal), st, SPEC)
                assert (row["reward_pass"], row["punish_pass"], row["naive"]) == (r_ok, p_ok, bal)
                assert row["testable"] == (r_ok and p_ok) and row["axis"] == "b"


def test_strip_job_removes_timing_and_r_keys():
    x = dict(seed=1, wall_s=2.0, pre={"x": {"A": 1, "wall_s": 3.0}}, r={"edit_edges": 0}, direction="r1", point=[1])
    assert RR.strip_job(x) == dict(seed=1, pre={"x": {"A": 1}})
    assert RR.strip_job(dict(q=1, r=2, report=3), top=("q", "r")) == dict(report=3)


def test_raw_check_clean_and_its_defects():
    got = got_for(ROWS, n_rep=8, n_act=8)
    rc = RR.raw_check(got, C, KEYS, JS)
    assert rc["reasons"] == [] and rc["edit_edges"] == [0] and rc["csc_sha256"] == "sha-C" and rc["n_pairs"] == 53
    assert "pairs" not in rc and "aggregate" not in rc                       # no pair statistic before the seal
    dup = got + [got[0]]
    assert any("duplicate" in m for m in RR.raw_check(dup, C, KEYS, JS)["reasons"])
    assert any("missing 1" in m for m in RR.raw_check(got[1:], C, KEYS, JS)["reasons"])
    short = copy.deepcopy(got)
    short[0]["result"]["report"]["pre"]["A"].pop()
    assert any("probes" in m for m in RR.raw_check(short, C, KEYS, JS)["reasons"])
    cells = copy.deepcopy(got)
    cells[0]["result"]["r"]["p_cells"]["x"][0][0] += 1
    assert any("per-cell" in m for m in RR.raw_check(cells, C, KEYS, JS)["reasons"])
    edited = copy.deepcopy(got)
    edited[0]["result"]["q"]["edit"] = SPEC.lever_edit
    assert any("ran edit" in m for m in RR.raw_check(edited, C, KEYS, JS)["reasons"])
    two = copy.deepcopy(got)
    two[0]["result"]["q"]["csc_sha256"] = "other"
    assert any("differs between rows" in m for m in RR.raw_check(two, C, KEYS, JS)["reasons"])


def test_cond_summary_counts_and_aggregate():
    plan = {k: (True, True, False) for k in KEYS[:12]}                          # 12 (b) testable
    plan.update({k: (True, True, True) for k in KEYS[21:24]})                   # 3 (a) testable and naive
    s = RR.cond_summary(got_for(ROWS, plan), C, SPEC, Z, KEYS, JS)
    a = s["aggregate"]
    assert (a["testable_b"], a["F_a"], a["naive_a"], a["n_b"], a["n_a"]) == (12, 3, 3, 21, 32)
    assert s["counts"]["b"] == dict(n=21, testable=12, reward_pass=12, punish_pass=21, naive=0)
    assert s["kc_median"] == 0.05 and s["d6a_over_share"] == 0.0 and s["apl_out_median"] == 0.2
    assert s["alpha_punish"] == {"0.8": 53} and s["reasons"] == []


def test_saturation_is_per_cell_against_the_single_cell_ceiling():
    got = got_for(ROWS[:1], n_rep=8)            # P 26 / 28 per type -> cells 13 / 13 and 14 / 14 per 0.6 s
    s = RR.saturation(got, SPEC)
    assert s["cap_hz"] == 500.0 and s["n"] == 2 * 2 * 8
    assert s["median"] == pytest.approx(13.5 / 0.6) and s["max"] == pytest.approx(14 / 0.6)
    assert s["share"] == {"0.5": 0.0, "0.8": 0.0}
    hot = copy.deepcopy(got)
    hot[0]["result"]["r"]["p_cells"]["x"] = [[300, 0]] * 8                # 500 Hz on one cell, 8 presentations
    s2 = RR.saturation(hot, SPEC)
    assert s2["max"] == pytest.approx(500.0) and s2["share"]["0.8"] == pytest.approx(8 / 32)


def test_z_renorm_is_a_record_beside_block_h4s_z():
    s = RR.z_renorm(got_for(ROWS, {k: (True, True, True) for k in KEYS}), SPEC, Z)
    assert set(s) >= {"z", "testable_b", "F_a", "naive_a", "abs_d_pre_median", "abs_d_pre_median_h4"}
    assert s["abs_d_pre_median_h4"] == 0.0


def test_transitions_count_c_to_l():
    cp = [dict(key=k, axis="b", testable=False, reward_pass=False, punish_pass=True, p=-3.0) for k in "abc"]
    lp = [dict(cp[0], punish_pass=False, p=-1.0), dict(cp[1], reward_pass=True), dict(cp[2])]
    t = RR.transitions(lp, cp)["b"]
    assert t["punish_pass"] == {"pp": 2, "pf": 1, "fp": 0, "ff": 0}
    assert t["reward_pass"] == {"pp": 0, "pf": 0, "fp": 1, "ff": 2}
    assert t["minus_p"]["a"] == [3.0, 1.0]


def test_gate1_record_beside_encoder_3():
    act = {"s1": dict(frac=[0.04] * 8, max_win=[5] * 8, csc_sha256=["sha-L"], edit_edges=[2]),
           "d1": dict(frac=[0.06] * 8, max_win=[40] * 8, csc_sha256=["sha-L"], edit_edges=[2])}
    rec = RR.gate1_record(act, ["s1"], ["d1"], {"s1": 0.05, "d1": 0.05}, SPEC)
    assert rec["median"] == pytest.approx(0.05) and rec["n_odours"] == 2 and rec["n_seeds"] == [8]
    assert rec["decreased"] == 1 and rec["d6a_over"] == 8 and rec["d6a_presentations"] == 16
    assert rec["edit_edges"] == [2] and rec["csc_sha256"] == ["sha-L"] and "ok" in rec["original"]


def _blocks(code="K"):
    return {n: dict(code_key=code, seeds=JS) for n in SPEC.cond_names}


def _raws(l_edges=2, c_edit="none", c_edges=0, c_sha="sha-C"):
    return {"L": got_for(ROWS, sha="sha-L", edges=l_edges, edit=SPEC.lever_edit),
            "C": got_for(ROWS, sha=c_sha, edges=c_edges, edit=c_edit),
            "E0": got_for(ROWS, sha="sha-C", edges=0, edit="none")}


def test_preread_validity_clean():
    v = RR.preread_validity(_blocks(), _raws(), SPEC, Z, KEYS, "K", "sha-C")
    assert v["reasons"] == [] and v["invalid"] == []


@pytest.mark.parametrize("edges", [1, 3])
def test_preread_validity_l_edges_other_than_2_is_invalid(edges):
    v = RR.preread_validity(_blocks(), _raws(l_edges=edges), SPEC, Z, KEYS, "K", "sha-C")
    assert v["invalid"] and "L changed" in v["invalid"][0]


def test_preread_validity_an_edit_in_c_is_a_reason():
    v = RR.preread_validity(_blocks(), _raws(c_edit=SPEC.lever_edit, c_edges=2, c_sha="sha-L"), SPEC, Z, KEYS, "K",
                            "sha-C")
    assert v["invalid"] == [] and any("C" in m for m in v["reasons"])


def test_preread_validity_code_key_and_sha_relations():
    assert RR.preread_validity(_blocks(code="old"), _raws(), SPEC, Z, KEYS, "K", "sha-C")["reasons"]
    assert RR.preread_validity(_blocks(), _raws(), SPEC, Z, KEYS, "K", "sha-other")["reasons"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_records.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_records'`.

- [ ] **Step 4: Write the implementation**

```python
# flymon/brain/r_records.py
"""R's per-pair values and records (R.3, R.4, R.6, R.9.5, R.9.7). Nothing here runs the engine or decides a band.
- strip_job: a job result without timing (`wall_s`, any depth) and R's top-level keys — what the reproduction gate
  compares bit for bit with the reference entries (Reading 4).
- raw_check: completeness / uniqueness / probe lengths / edit / per-cell sums / finite values / one CSC sha of one
  condition's oracle results, WITHOUT any pair statistic: the only content of a judgement measurement block (jm:*), so
  nothing band-relevant is visible before the seal (R.9.7, Reading 10).
- cond_summary: raw_check plus h4_formula.pair_stats per pair (m = min(r, −p), testable ⇔ m ≥ 2, naive ⇔ |d_pre| <
  0.5; an undefined or NaN d′ is a reason, ±inf is kept as the encoder's _measure keeps it), arm_aggregate, per-axis
  pass counts and R.6's records.
- transitions / compare: C → L (and E0 → L) tables and the records gate ③ and the judgement write.
- gate1_record: gate ①'s per-odour activity under the lever beside encoder ③'s values (same seeds).
- preread_validity: R.9.7's checks over the three judgement conditions (INVALID: L edges != 2; reasons otherwise)."""
from __future__ import annotations

import math
from collections import Counter
from statistics import median

import numpy as np

from ..agent import e_rules
from ..agent.e_spec import SPEC as E
from . import d6a
from .h4_formula import arm_aggregate, pair_stats
from .h4_rules import z_constants
from .h4_spec import SPEC as H4
from .o_rules import finite

STRIP = ("wall_s",)
R_KEYS = ("r", "direction", "x", "y", "point")
CRITERIA = ("testable", "reward_pass", "punish_pass")


def strip_job(x: dict, top=R_KEYS) -> dict:
    def rec(v):
        if isinstance(v, dict):
            return {k: rec(w) for k, w in v.items() if k not in STRIP}
        if isinstance(v, list):
            return [rec(w) for w in v]
        return v
    return {k: rec(v) for k, v in x.items() if k not in top and k not in STRIP}


def cell_sums_ok(res: dict) -> bool:
    r = res.get("r")
    if r is None:
        return False
    pre = res["report"]["pre"]["P"]
    for j, side in enumerate(("x", "y")):
        cells = r["p_cells"][side]
        if len(cells) != len(pre) or any(int(sum(c)) != int(row[j]) for c, row in zip(cells, pre)):
            return False
    return True


def stats_ok(st) -> bool:
    return st is not None and all(isinstance(st[f], (int, float)) and not math.isnan(st[f])
                                  for f in ("d_pre", "r", "p", "m"))


def pair_row(key: str, res: dict, st: dict, spec) -> dict:
    tm = spec.testable_min
    return dict(key=key, axis=key.split("|", 1)[0], d_pre=float(st["d_pre"]), r=float(st["r"]), p=float(st["p"]),
                m=float(st["m"]), testable=bool(st["testable"]), reward_pass=bool(st["r"] >= tm),
                punish_pass=bool(-st["p"] >= tm), naive=bool(abs(st["d_pre"]) < spec.naive_max),
                alpha_reward=float(res["alpha_reward"]), alpha_punish=float(res["alpha_punish"]))


def aggregate(pairs: list, spec) -> dict | None:
    """h4_formula.arm_aggregate over the pair rows; None without a (b) pair (T_b would divide by 0)."""
    if not any(p["axis"] == "b" for p in pairs):
        return None
    return arm_aggregate({(p["axis"], p["key"]): p for p in pairs}, spec.naive_max, spec.bar_b / spec.n_b,
                         spec.f_a_min)


def counts(pairs: list) -> dict:
    out = {}
    for ax in ("b", "a"):
        ps = [p for p in pairs if p["axis"] == ax]
        out[ax] = dict(n=len(ps), **{c: sum(p[c] for p in ps) for c in CRITERIA}, naive=sum(p["naive"] for p in ps))
    return out


def _short(res: dict, n_rep: int, n_act: int) -> bool:
    q, r = res.get("q"), res.get("r")
    try:
        return (q is None or r is None
                or any(len(res["report"][ph][k]) != n_rep for ph in ("pre", "R1", "R2") for k in ("A", "P"))
                or any(len(r["p_cells"][s]) != n_rep for s in ("x", "y"))
                or any(len(res["kc"][s]["frac"]) != n_act or len(res["kc"][s]["max_win"]) != n_act
                       for s in ("x", "y")))
    except (KeyError, TypeError):
        return True


def raw_check(got: list, cond, expected: list, seeds: dict) -> dict:
    n_rep, n_act = len(seeds["report"]), len(seeds["act"])
    reasons = []
    keys = [g["key"] for g in got]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    exp = set(expected)
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    for g in got:
        res = g["result"]
        if _short(res, n_rep, n_act):
            reasons.append(f"{g['key']}: probes are not the {n_rep} report / {n_act} activity seeds, or a record "
                           f"is missing")
            continue
        q, r = res["q"], res["r"]
        if q.get("edit") != cond.edit:
            reasons.append(f"{g['key']}: ran edit {q.get('edit')}, condition {cond.name} declares {cond.edit}")
        if not cell_sums_ok(res):
            reasons.append(f"{g['key']}: per-cell {r.get('p_type')} counts do not sum to the type count")
        if not (finite(res["report"]) and finite(res["kc"]) and finite(r["p_cells"]) and finite(q["apl_out"])):
            reasons.append(f"{g['key']}: a non-finite value")
    qs = [g["result"]["q"] for g in got if isinstance(g["result"].get("q"), dict)]
    edges = sorted({int(q["edit_edges"]) for q in qs})
    shas = sorted({q["csc_sha256"] for q in qs})
    if len(shas) > 1:
        reasons.append(f"CSC sha256 differs between rows: {shas}")
    return dict(reasons=reasons, edit_edges=edges, csc_sha256=shas[0] if len(shas) == 1 else shas, n_pairs=len(got),
                manifest=[dict(key=g["key"], cache_key=g.get("cache_key"), cache_file=g.get("cache_file"))
                          for g in got])


def saturation(got: list, spec) -> dict | None:
    """R.9.7: every cell × report seed × X/Y of the naive per-cell probe (w0); rate = count × 1000 / read window;
    ceiling = 1000 / (refrac_steps × dt) of one cell; shares at ≥ each sat_frac × ceiling (Reading 11)."""
    rs = [g["result"]["r"] for g in got if isinstance(g["result"].get("r"), dict)]
    if not rs:
        return None
    rates = [c * spec.ms_per_s / r["read_ms"] for r in rs for s in ("x", "y") for seed in r["p_cells"][s] for c in seed]
    caps = sorted({spec.ms_per_s / (r["refrac_steps"] * r["dt"]) for r in rs})
    if len(caps) != 1 or not rates:
        return dict(cap_hz=caps, n=len(rates), why="mixed ceilings or no cell")
    x, cap = np.asarray(rates, float), caps[0]
    return dict(cap_hz=cap, n=int(x.size), median=float(np.median(x)),
                q95=float(np.percentile(x, spec.sat_quantile)), max=float(x.max()),
                share={str(f): float((x >= f * cap).mean()) for f in spec.sat_fracs})


def _rows_under(got: list, z: dict, spec) -> list:
    out = []
    for g in got:
        st = pair_stats(g["result"]["report"], z, spec.testable_min)
        if stats_ok(st):
            out.append(pair_row(g["key"], g["result"], st, spec))
    return out


def z_renorm(got: list, spec, z_h4: dict) -> dict:
    """R.9.7, record only (Reading 12): z re-estimated on this condition's own naive report-seed probes (pairs × seeds
    × X/Y, type sums; ddof = H.4's z_ddof) and what it would give, beside median |d_pre| under block h4's z."""
    ref = [{"types": {"A": float(a), "P": float(p)}} for g in got
           for ra, rp in zip(g["result"]["report"]["pre"]["A"], g["result"]["report"]["pre"]["P"])
           for a, p in zip(ra, rp)]
    try:
        zr = z_constants(ref, {"A": "A", "P": "P"}, H4.z_ddof)
    except ValueError as e:
        return dict(z=None, why=str(e))
    pr, ph = _rows_under(got, zr, spec), _rows_under(got, z_h4, spec)
    agg = aggregate(pr, spec) or {}

    def med(ps):
        return float(median(abs(p["d_pre"]) for p in ps)) if ps else None

    return dict(z={k: [float(v) for v in zr[k]] for k in zr}, testable_b=agg.get("testable_b"), F_a=agg.get("F_a"),
                naive_a=agg.get("naive_a"), abs_d_pre_median=med(pr), abs_d_pre_median_h4=med(ph))


def cond_summary(got: list, cond, spec, z: dict, expected: list, seeds: dict) -> dict:
    rc = raw_check(got, cond, expected, seeds)
    n_rep, n_act = len(seeds["report"]), len(seeds["act"])
    ok = [g for g in got if not _short(g["result"], n_rep, n_act)]
    pairs, und = [], []
    for g in ok:
        st = pair_stats(g["result"]["report"], z, spec.testable_min)
        if not stats_ok(st):
            und.append(g["key"])
            continue
        pairs.append(pair_row(g["key"], g["result"], st, spec))
    reasons = list(rc["reasons"]) + ([f"{len(und)} pair(s) with an undefined d′: {und[:3]}"] if und else [])
    fr = [x for g in ok for s in ("x", "y") for x in g["result"]["kc"][s]["frac"]]
    wins = [w for g in ok for s in ("x", "y") for w in g["result"]["kc"][s]["max_win"]]
    apl = [a for g in ok for s in ("x", "y") for a in g["result"]["q"]["apl_out"][s]]
    a_naive = [c for g in ok for row in g["result"]["report"]["pre"]["A"] for c in row]
    return dict(rc, reasons=reasons, pairs=pairs, aggregate=aggregate(pairs, spec), counts=counts(pairs),
                kc_median=float(np.median(fr)) if fr else None,
                d6a_over_share=float(np.mean([w >= d6a.OVER_SPIKES for w in wins])) if wins else None,
                d6a_condition=d6a.CONDITION, apl_out_median=float(np.median(apl)) if apl else None,
                naive_A_median=float(np.median(a_naive)) if a_naive else None,
                alpha_reward=dict(Counter(str(p["alpha_reward"]) for p in pairs)),
                alpha_punish=dict(Counter(str(p["alpha_punish"]) for p in pairs)),
                saturation=saturation(ok, spec), z_renorm=z_renorm(ok, spec, z),
                condition=dict(name=cond.name, edit=cond.edit, odour=cond.odour, strength=cond.strength))


def transitions(to: list, frm: list) -> dict:
    """{axis: {criterion: {"pp","pf","fp","ff"}, "minus_p": {key: [−p from, −p to]}}}: first letter = `frm` (C or E0)
    passes, second = `to` (L) passes (R.4, R.6, R.9.5)."""
    tm, fm = {p["key"]: p for p in to}, {p["key"]: p for p in frm}
    out = {}
    for ax in ("b", "a"):
        ks = sorted(k for k in fm if k in tm and fm[k]["axis"] == ax)
        t = {}
        for c in CRITERIA:
            cell = {"pp": 0, "pf": 0, "fp": 0, "ff": 0}
            for k in ks:
                cell[("p" if fm[k][c] else "f") + ("p" if tm[k][c] else "f")] += 1
            t[c] = cell
        t["minus_p"] = {k: [-fm[k]["p"], -tm[k]["p"]] for k in ks}
        out[ax] = t
    return out


KEEP = ("aggregate", "counts", "kc_median", "d6a_over_share", "apl_out_median", "naive_A_median", "alpha_reward",
        "alpha_punish", "saturation", "z_renorm", "edit_edges", "csc_sha256", "reasons")


def compare(L: dict, C: dict, E0: dict | None, spec) -> dict:
    names = spec.cond_names
    conds = {names[0]: L, names[1]: C, **({names[2]: E0} if E0 else {})}
    return dict(transitions_C_to_L=transitions(L["pairs"], C["pairs"]),
                transitions_E0_to_L=transitions(L["pairs"], E0["pairs"]) if E0 else None,
                C_F_a=(C["aggregate"] or {}).get("F_a"), C_naive_a=(C["aggregate"] or {}).get("naive_a"),
                conditions={n: {k: s.get(k) for k in KEEP} for n, s in conds.items()})


def gate1_record(act: dict, single: list, dual: list, enc_per: dict, spec) -> dict:
    """R.9.2: the per-odour activity (median over seeds, e_rules.odour_activity) under the lever, the 112-odour median,
    the ORIGINAL qualification (encoder 4.3 conditions 1-3, e_rules.strength_ok on E's numbers) as a record, and the
    paired comparison with encoder ③'s per-odour values on the same seeds."""
    per = e_rules.odour_activity({o: act[o]["frac"] for o in act})
    sv, dv_ = [per[o] for o in single], [per[o] for o in dual]
    wins = [int(w) for v in act.values() for w in v["max_win"]]
    enc = {o: float(enc_per[o]) for o in per if o in enc_per}
    return dict(median=float(median(sv + dv_)), n_odours=len(per),
                n_seeds=sorted({len(v["frac"]) for v in act.values()}), original=e_rules.strength_ok(sv, dv_, E),
                encoder_median=float(median(enc.values())) if enc else None, per_odour=per, encoder_per_odour=enc,
                missing_in_encoder=sorted(set(per) - set(enc)), decreased=sum(per[o] < enc[o] for o in enc),
                ratio_median=(float(median(per[o] / enc[o] for o in enc if enc[o] > 0)) if enc else None),
                d6a_over=sum(w >= d6a.OVER_SPIKES for w in wins), d6a_presentations=len(wins),
                d6a_condition=d6a.CONDITION, edit_edges=sorted({int(e) for v in act.values() for e in v["edit_edges"]}),
                csc_sha256=sorted({s for v in act.values() for s in v["csc_sha256"]}))


def preread_validity(blocks: dict, raws: dict, spec, z: dict, expected: list, code_key: str, repro_sha: str) -> dict:
    """R.9.7 (Reading 10) over the judgement conditions: block code key and seeds, raw_check, defined d′ per pair, L
    exactly lever_edges CSC edges (else `invalid`), C and E0 none, L ≠ C sha, C = E0 = the repro's unedited sha."""
    reasons, invalid, checks = [], [], {}
    seeds = spec.judge_seeds()
    n_rep, n_act = len(seeds["report"]), len(seeds["act"])
    for n in spec.cond_names:
        cond, b = spec.cond(n), blocks[n]
        if b.get("code_key") != code_key:
            reasons.append(f"{n}: block code key {b.get('code_key')} is not the current {code_key}")
        if b.get("seeds") != seeds:
            reasons.append(f"{n}: block seeds are not the judgement oracle seeds")
        rc = raw_check(raws[n], cond, expected, seeds)
        reasons += [f"{n}: {x}" for x in rc["reasons"]]
        und = [g["key"] for g in raws[n] if not _short(g["result"], n_rep, n_act)
               and not stats_ok(pair_stats(g["result"]["report"], z, spec.testable_min))]
        if und:
            reasons.append(f"{n}: {len(und)} pair(s) with an undefined d′")
        checks[n] = dict(edit_edges=rc["edit_edges"], csc_sha256=rc["csc_sha256"], n_pairs=rc["n_pairs"])
    nl, nc, ne = spec.cond_names
    if checks[nl]["edit_edges"] != [spec.lever_edges]:
        invalid.append(f"L changed {checks[nl]['edit_edges']} CSC edges, declared exactly {spec.lever_edges} (R.6)")
    for n in (nc, ne):
        if checks[n]["edit_edges"] != [0]:
            reasons.append(f"{n} changed {checks[n]['edit_edges']} CSC edges, declared none")
    if checks[nl]["csc_sha256"] == checks[nc]["csc_sha256"]:
        reasons.append("L and C ran on the same CSC weights")
    if not (checks[nc]["csc_sha256"] == checks[ne]["csc_sha256"] == repro_sha):
        reasons.append(f"C / E0 CSC sha256 {checks[nc]['csc_sha256']} / {checks[ne]['csc_sha256']} is not the "
                       f"unedited engine's {repro_sha}")
    return dict(reasons=reasons, invalid=invalid, checks=checks)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_r_records.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/r_records.py tests/brain/r_fixtures.py tests/brain/test_r_records.py
git commit -m "feat(r): r_records — pair rows, raw checks without statistics, saturation, z renorm, transitions, pre-read validity"
```

---
### Task 6: `r_rules` — gates, G_fail, the bands, the closing sentences, the operating characteristic

**Files:**
- Create: `flymon/brain/r_rules.py`
- Test: `tests/brain/test_r_rules.py`

**Interfaces:**
- Consumes: Task 1 `SPEC`; `e_rules` band names and `_binom`; `p_rules.LEARNS_CONFIRMATORY`; `n_rules.INVALID`; `h3_store.canonical`.
- Produces:
  - constants `PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN, STOP_STRENGTH_LEVER, STOP_PUNISH_BROKEN, STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH, SELECTED, B_TB, B_FA, B_NC, B_PG, BANDS, C_ABOVE_BAR, BELOW_BAR, MARGIN`;
  - `gate1(rec, cap_ok, spec) -> dict(outcome, reasons, failed[, sentence])`;
  - `gate2(res, spec) -> dict(outcome, reasons, label[, sentence])`;
  - `gate3(L, C, spec, repro_sha) -> dict(outcome, reasons[, testable_b, c_even, F_a, naive_a, C_F_a, C_naive_a, sentence])`;
  - `g_fail(lp, cp, spec) -> dict(g_fail, axes={b, a: {pun_L, pun_C, pass_to_fail, fail_to_pass, drop, flips, fail}})`;
  - `read_band(n, c, f_a, g, n_b, n_a, naive_a, spec) -> dict(band, reason[, naive_a, f_a_possible])`;
  - `SENTENCES`, `sentence(outcome, fields) -> str`;
  - `g_axis(N, a_pf, a_fp, spec) -> float`, `g_prob(a_pf, a_fp, spec) -> float`;
  - `oc(spec) -> dict(assumption, g_fail, rows, sha256)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_r_rules.py
"""R.3 / R.7 / R.9.2-R.9.5: the judgement code is the authoritative source. Every cell n, c ∈ 0..21 × F_a ∈ 0..32 ×
G_fail reads exactly one band, equal to an independent transcription of R.3's ordered lines; the boundary fixtures (c
10/11, n = c / c+1 / c+2, n 10/11, F_a 1/2, G_fail before F_a, counts 20/22 and 31/33, the R.9.4 counterexample);
G_fail's drop 1/2 and pass->fail 2/3 boundaries on either axis; the gates; the sentences verbatim; the OC exact."""
import itertools

import pytest

from flymon.brain import r_rules as R
from flymon.brain.r_spec import SPEC


def _ref_band(n, c, f, g, nb=21, na=32):
    lines = [(nb != 21 or na != 32, R.NOT_READ), (c >= 11, R.B_NC), (n <= c, R.B_TB), (n < 11, R.B_NC),
             (n - c < 2, R.B_NC), (g, R.B_PG), (f < 2, R.B_FA), (True, R.SELECTED)]
    return next(b for cond, b in lines if cond)


def test_every_cell_reads_exactly_the_transcribed_band():
    seen = set()
    for n, c, f, g in itertools.product(range(22), range(22), range(33), (False, True)):
        b = R.read_band(n, c, f, g, 21, 32, f, SPEC)["band"]
        assert b == _ref_band(n, c, f, g), (n, c, f, g)
        seen.add(b)
    assert seen == set(R.BANDS)


@pytest.mark.parametrize("n,c,f,g,band,reason", [
    (13, 10, 2, False, R.SELECTED, "PASS"),
    (13, 11, 2, False, R.B_NC, R.C_ABOVE_BAR),            # c 10 / 11
    (13, 11, 2, True, R.B_NC, R.C_ABOVE_BAR),             # R.9.4: c 11 · F_a_C 0 · n 13 · F_a 2 -> B_결론없음
    (8, 8, 5, False, R.B_TB, "NO_GAIN"),                   # n = c
    (9, 8, 5, False, R.B_NC, R.BELOW_BAR),                 # n = c + 1, below 11
    (10, 8, 5, False, R.B_NC, R.BELOW_BAR),                # n = 10
    (11, 8, 5, False, R.SELECTED, "PASS"),                 # n = 11
    (11, 10, 5, False, R.B_NC, R.MARGIN),                  # n = c + 1 at the bar
    (12, 10, 5, False, R.SELECTED, "PASS"),                # n = c + 2
    (13, 8, 1, False, R.B_FA, "F_A"),                      # F_a 1
    (13, 8, 2, False, R.SELECTED, "PASS"),                 # F_a 2
    (13, 8, 1, True, R.B_PG, "PUNISH_GUARD"),              # G_fail before F_a (R.3 line 6, user addition ii)
    (13, 8, 5, True, R.B_PG, "PUNISH_GUARD"),
    (11, 10, 5, True, R.B_NC, R.MARGIN),                   # margin before G_fail
])
def test_boundaries(n, c, f, g, band, reason):
    out = R.read_band(n, c, f, g, 21, 32, max(f, 4), SPEC)
    assert (out["band"], out["reason"]) == (band, reason)


@pytest.mark.parametrize("nb,na", [(20, 32), (22, 32), (21, 31), (21, 33)])
def test_counts_other_than_21_32_are_not_read(nb, na):
    assert R.read_band(13, 8, 5, False, nb, na, 5, SPEC)["band"] == R.NOT_READ


def test_b_fa_carries_naive_a_and_possibility():
    out = R.read_band(13, 8, 1, False, 21, 32, 1, SPEC)
    assert out["naive_a"] == 1 and out["f_a_possible"] is False
    assert R.read_band(13, 8, 1, False, 21, 32, 4, SPEC)["f_a_possible"] is True


def _pp(ax, c_pass, l_pass, n=10):
    cp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in c_pass) for i in range(n)]
    lp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in l_pass) for i in range(n)]
    return lp, cp


@pytest.mark.parametrize("ax", ["b", "a"])
def test_g_fail_drop_1_vs_2(ax):
    lp, cp = _pp(ax, set(range(5)), set(range(4)))           # 5 -> 4: drop 1, pass->fail 1
    assert R.g_fail(lp, cp, SPEC)["g_fail"] is False
    lp, cp = _pp(ax, set(range(5)), set(range(3)))           # 5 -> 3: drop 2
    out = R.g_fail(lp, cp, SPEC)
    assert out["g_fail"] is True and out["axes"][ax]["drop"] and not out["axes"][ax]["flips"]


@pytest.mark.parametrize("ax", ["b", "a"])
def test_g_fail_pass_to_fail_2_vs_3_with_no_drop(ax):
    lp, cp = _pp(ax, set(range(5)), {2, 3, 4, 5, 6})         # pass->fail 2, fail->pass 2, drop 0
    out = R.g_fail(lp, cp, SPEC)
    assert out["g_fail"] is False and out["axes"][ax]["pass_to_fail"] == 2
    lp, cp = _pp(ax, set(range(5)), {3, 4, 5, 6, 7})         # pass->fail 3, fail->pass 3, drop 0
    out = R.g_fail(lp, cp, SPEC)
    assert out["g_fail"] is True and out["axes"][ax]["flips"] and not out["axes"][ax]["drop"]
    assert (out["axes"][ax]["pun_L"], out["axes"][ax]["pun_C"]) == (5, 5)


def _rec(**kw):
    return dict(dict(median=0.045, n_odours=112, n_seeds=[8], edit_edges=[2], csc_sha256=["s"]), **kw)


def test_gate1():
    assert R.gate1(_rec(), True, SPEC)["outcome"] == R.PASS
    assert R.gate1(_rec(median=0.03), True, SPEC)["outcome"] == R.PASS            # band inclusive
    assert R.gate1(_rec(median=0.15), True, SPEC)["outcome"] == R.PASS
    out = R.gate1(_rec(median=0.0281), True, SPEC)
    assert out["outcome"] == R.STOP_STRENGTH_LEVER and "0.0281" in out["sentence"]
    assert R.gate1(_rec(median=0.1501), True, SPEC)["outcome"] == R.STOP_STRENGTH_LEVER
    out = R.gate1(_rec(), False, SPEC)
    assert out["outcome"] == R.STOP_STRENGTH_LEVER and "ORN 상한" in out["sentence"]
    assert R.gate1(_rec(edit_edges=[1]), True, SPEC)["outcome"] == R.INVALID
    assert R.gate1(_rec(csc_sha256=["a", "b"]), True, SPEC)["outcome"] == R.INVALID
    assert R.gate1(_rec(n_odours=111), True, SPEC)["outcome"] == R.INVALID


def test_gate2():
    ok = dict(outcome="JUDGED", label="LEARNS_CONFIRMATORY", directions={"r1": {"ell": 1.8}, "r2": {"ell": 2.2}},
              reasons=[])
    assert R.gate2(ok, SPEC)["outcome"] == R.PASS
    out = R.gate2(dict(ok, label="DIRECTION_DEPENDENT"), SPEC)
    assert out["outcome"] == R.STOP_PUNISH_BROKEN
    assert out["sentence"] == ("APL→MBON05 제거 아래에서 P의 처벌 학습 확인이 재현되지 않았다"
                               "(DIRECTION_DEPENDENT, ℓ_r1 1.800, ℓ_r2 2.200).")
    assert R.gate2(dict(outcome="INVALID", label="INVALID", reasons=["x"]), SPEC)["outcome"] == R.INVALID


def _sum(tb, fa=0, edges=(2,), sha="sha-L", reasons=()):
    return dict(reasons=list(reasons), edit_edges=list(edges), csc_sha256=sha,
                aggregate=dict(testable_b=tb, F_a=fa, naive_a=2, n_b=21, n_a=18))


def test_gate3():
    C = _sum(7, edges=(0,), sha="sha-C")
    out = R.gate3(_sum(11), C, SPEC, "sha-C")
    assert out["outcome"] == R.PASS and (out["testable_b"], out["c_even"]) == (11, 7)
    out = R.gate3(_sum(10), C, SPEC, "sha-C")
    assert out["outcome"] == R.STOP_EVEN_LOW_LEVER
    assert out["sentence"] == ("APL→MBON05 제거 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다"
                               "(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 0/18).")
    out = R.gate3(_sum(16), _sum(8, edges=(0,), sha="sha-C"), SPEC, "sha-C")
    assert out["outcome"] == R.STOP_C_EVEN_MISMATCH and "8/21" in out["sentence"]   # checked before testable_b
    assert R.gate3(_sum(16, edges=(1,)), C, SPEC, "sha-C")["outcome"] == R.INVALID
    assert R.gate3(_sum(16), C, SPEC, "sha-other")["outcome"] == R.INVALID
    assert R.gate3(_sum(16, sha="sha-C"), C, SPEC, "sha-C")["outcome"] == R.INVALID
    assert R.gate3(_sum(16, reasons=("dup",)), C, SPEC, "sha-C")["outcome"] == R.INVALID


F = dict(n=14, c=8, f_a=3, naive_a=4, T=103, k_even=12, pb_L=15, pb_C=17, pa_L=20, pa_C=22, cond="x",
         label="NO_LEARNING", l1="0.100", l2="0.200", tb=9, c_even=7, k=0)


def test_sentences_verbatim():
    for o in (R.STOP_STRENGTH_LEVER, R.STOP_PUNISH_BROKEN, R.STOP_EVEN_LOW_LEVER, R.STOP_C_EVEN_MISMATCH, R.B_TB,
              R.B_PG, R.B_FA, R.SELECTED):
        assert "{" not in R.sentence(o, F), o
    assert R.sentence(R.B_TB, F) == ("APL→MBON05 제거가 판정 세트(L 생성기 턴 64–103)에서 14/21로 지렛대 없는 같은 세트 "
                                     "8/21보다 오르지 않았다. → 이 지렛대를 닫는다.")
    assert R.sentence(R.B_NC, dict(F, reason=R.C_ABOVE_BAR)) == (
        "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다. (14/21 대 8/21, F_a 3/32)")
    assert R.sentence(R.B_NC, dict(F, reason=R.BELOW_BAR)).startswith("지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다.")
    assert R.sentence(R.B_NC, dict(F, reason=R.MARGIN)).startswith("기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다")
    assert R.sentence(R.B_PG, F) == ("M2 (b) 기준과 여유는 넘었지만 지렛대가 처벌 통과를 줄였다(처벌 통과 (b) 15 대 17, "
                                     "(a) 20 대 22).")
    assert R.sentence(R.B_FA, F) == ("M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다(14/21 대 8/21, F_a 3/32, "
                                     "naive_a 4). 다음 병목은 F_a(순진 균형 (a) 쌍)다.")
    sel = R.sentence(R.SELECTED, F)
    assert sel.startswith("C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 판정 세트(L 생성기 턴 64–103")
    assert "(14/21 대 8/21, 여유 ≥ 2, F_a 3/32, 처벌 가드 통과, 짝수 12/21, P 재현 `LEARNS_CONFIRMATORY`, 판정 시드 24_100_xxx)" in sel
    assert sel.endswith("실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다.")
    assert R.sentence(R.STOP_STRENGTH_LEVER, F) == ("APL→MBON05 제거 아래에서 E-grid k2-norm s 1.0이 KC 활성 자격(x)을 "
                                                    "잃었다.")
    with pytest.raises(KeyError):
        R.sentence(R.B_FA, {})


def test_g_axis_matches_brute_force_on_a_small_axis():
    a_pf, a_fp, tot = 0.2, 0.1, 0.0
    for states in itertools.product(("pf", "fp", "same"), repeat=4):
        w = 1.0
        for s in states:
            w *= {"pf": a_pf, "fp": a_fp, "same": 1 - a_pf - a_fp}[s]
        k, j = states.count("pf"), states.count("fp")
        if k - j >= 2 or k >= 3:
            tot += w
    assert R.g_axis(4, a_pf, a_fp, SPEC) == pytest.approx(tot)


def test_g_prob_null_and_harm():
    assert R.g_prob(0.0, 0.0, SPEC) == 0.0
    vals = [R.g_prob(a, a, SPEC) for a in SPEC.oc_discord]
    assert all(0 < v < 1 for v in vals) and vals == sorted(vals)
    assert R.g_prob(0.2, 0.05, SPEC) > R.g_prob(0.05, 0.05, SPEC)


def test_oc_rows_are_distributions_and_carry_the_g_rows():
    oc = R.oc(SPEC)
    assert len(oc["g_fail"]) == len(SPEC.oc_discord) + len(SPEC.oc_harm)
    n_na = min(SPEC.n_a, SPEC.oc_naive_max) + 1
    assert len(oc["rows"]) == len(SPEC.oc_q) * len(SPEC.oc_c) * n_na * (1 + len(oc["g_fail"]))
    for row in oc["rows"][::37]:
        assert sum(row["P"].values()) == pytest.approx(1.0)
    assert all(r["P"][R.B_PG] == 0 for r in oc["rows"] if r["g_kind"] == "none")
    assert len(oc["sha256"]) == 64 and oc["sha256"] == R.oc(SPEC)["sha256"]


def test_r_rules_literals_are_only_0_and_1():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "flymon/brain/r_rules.py").read_text()
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1}, nums
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_rules.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_rules'`.

- [ ] **Step 3: Write the implementation**

```python
# flymon/brain/r_rules.py
"""R's decisions (R.2, R.3, R.7, R.9.2-R.9.5). The judgement code is the authoritative source (R.3): read_band is R.3's
ordered list with R.9.4 / R.9.5; every number is a field of the LeverSpec passed in. SENTENCES are R.7 as amended by
R.9.3, verbatim, 〈…〉 replaced by {field}, fixed before any measurement (plan Readings 6, 8, 9). The operating
characteristic is exact (Reading 13) and never changes a band."""
from __future__ import annotations

import hashlib
from math import comb

from ..agent import e_rules
from .h3_store import canonical
from .n_rules import INVALID as P_INVALID
from .p_rules import LEARNS_CONFIRMATORY

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = "PASS", "INVALID", "NOT_READ", "SEALED", "READ", "INVALID_RUN"
STOP_STRENGTH_LEVER, STOP_PUNISH_BROKEN = "STOP_STRENGTH_LEVER", "STOP_PUNISH_BROKEN"
STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH = "STOP_EVEN_LOW_LEVER", "STOP_C_EVEN_MISMATCH"
SELECTED, B_TB, B_FA, B_NC = e_rules.SELECTED, e_rules.B_TB, e_rules.B_FA, e_rules.B_NC
B_PG = "B_처벌가드"
BANDS = (SELECTED, B_TB, B_FA, B_NC, B_PG)
C_ABOVE_BAR, BELOW_BAR, MARGIN = "C_ABOVE_BAR", "BELOW_BAR", "MARGIN"


# ================================================================ gates (R.2, R.9.2, R.9.3, R.9.6)
def gate1(rec: dict, cap_ok: bool, spec) -> dict:
    """R.9.2: INVALID when the lever did not change exactly lever_edges edges on one CSC or the odours / seeds are not
    the declared ones; else STOP_STRENGTH_LEVER when the 112-odour median is outside valid_band or the ORN cap fails."""
    bad = []
    if rec["edit_edges"] != [spec.lever_edges] or len(rec["csc_sha256"]) != 1:
        bad.append(f"lever edges {rec['edit_edges']} on CSC {rec['csc_sha256']}, declared {spec.lever_edges} on one")
    if rec["n_odours"] != spec.n_calib or rec["n_seeds"] != [len(spec.kc_seeds())]:
        bad.append(f"{rec['n_odours']} odours × {rec['n_seeds']} seeds, declared {spec.n_calib} × "
                   f"{len(spec.kc_seeds())}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    lo, hi = spec.valid_band
    failed = []
    if not lo <= rec["median"] <= hi:
        failed.append(f"냄새별 KC 활성 중앙값 {rec['median']:.4f} ∉ [{lo}, {hi}]")
    if not cap_ok:
        failed.append(f"ORN 상한(s {spec.strength})")
    if failed:
        return dict(outcome=STOP_STRENGTH_LEVER, reasons=[], failed=failed,
                    sentence=sentence(STOP_STRENGTH_LEVER, dict(cond="; ".join(failed))))
    return dict(outcome=PASS, reasons=[], failed=[])


def gate2(res: dict, spec) -> dict:
    """R.2: p_rules.p_judge's result on R's P spec. INVALID stays INVALID (one rerun, R.5); LEARNS_CONFIRMATORY passes;
    any other label is STOP_PUNISH_BROKEN (no looser rule)."""
    if res.get("outcome") == P_INVALID:
        return dict(outcome=INVALID, reasons=list(res.get("reasons", [])), label=P_INVALID)
    if res["label"] == LEARNS_CONFIRMATORY:
        return dict(outcome=PASS, reasons=[], label=res["label"])
    d, names = res["directions"], list(spec.p.directions)
    return dict(outcome=STOP_PUNISH_BROKEN, reasons=[], label=res["label"],
                sentence=sentence(STOP_PUNISH_BROKEN, dict(label=res["label"], l1=f"{d[names[0]]['ell']:.3f}",
                                                           l2=f"{d[names[-1]]['ell']:.3f}")))


def gate3(L: dict, C: dict, spec, repro_sha: str) -> dict:
    """R.9.3 / R.9.6 on the even block's L and C summaries (Reading 6): validity first (reasons, L exactly
    lever_edges edges, C none, C on the unedited engine of the reproduction gate, L ≠ C); then C's testable_b must be
    c_even_expected (else STOP_C_EVEN_MISMATCH); then L's testable_b ≥ bar_b (else STOP_EVEN_LOW_LEVER)."""
    reasons = [f"L: {m}" for m in L["reasons"]] + [f"C: {m}" for m in C["reasons"]]
    if L["edit_edges"] != [spec.lever_edges]:
        reasons.append(f"L changed {L['edit_edges']} CSC edges, declared {spec.lever_edges}")
    if C["edit_edges"] != [0]:
        reasons.append(f"C changed {C['edit_edges']} CSC edges, declared none")
    if C["csc_sha256"] != repro_sha:
        reasons.append(f"C ran on CSC {C['csc_sha256']}, not the reproduction gate's {repro_sha}")
    if L["csc_sha256"] == C["csc_sha256"]:
        reasons.append("L and C ran on the same CSC weights")
    if L.get("aggregate") is None or C.get("aggregate") is None:
        reasons.append("no (b) aggregate")
    if reasons:
        return dict(outcome=INVALID, reasons=reasons)
    aL, aC = L["aggregate"], C["aggregate"]
    base = dict(reasons=[], testable_b=aL["testable_b"], c_even=aC["testable_b"], F_a=aL["F_a"],
                naive_a=aL["naive_a"], C_F_a=aC["F_a"], C_naive_a=aC["naive_a"])
    fields = dict(tb=aL["testable_b"], c_even=aC["testable_b"], k=aL["F_a"])
    if aC["testable_b"] != spec.c_even_expected:
        return dict(base, outcome=STOP_C_EVEN_MISMATCH, sentence=sentence(STOP_C_EVEN_MISMATCH, fields))
    if aL["testable_b"] < spec.bar_b:
        return dict(base, outcome=STOP_EVEN_LOW_LEVER, sentence=sentence(STOP_EVEN_LOW_LEVER, fields))
    return dict(base, outcome=PASS)


# ================================================================ the punishment guard (R.4, R.9.5, Reading 7)
def g_fail(lp: list, cp: list, spec) -> dict:
    """Per axis over the pairs in both: pun = #(−p ≥ 2); drop ⇔ pun_C − pun_L ≥ g_fail_drop; flips ⇔ #(C pass ∧ L
    fail) ≥ g_fail_flips. G_fail ⇔ drop or flips on (b) or (a)."""
    lm, cm = {p["key"]: p for p in lp}, {p["key"]: p for p in cp}
    axes = {}
    for ax in ("b", "a"):
        ks = sorted(k for k, p in cm.items() if p["axis"] == ax and k in lm)
        pun_c = sum(bool(cm[k]["punish_pass"]) for k in ks)
        pun_l = sum(bool(lm[k]["punish_pass"]) for k in ks)
        pf = sum(bool(cm[k]["punish_pass"]) and not lm[k]["punish_pass"] for k in ks)
        fp = sum(bool(lm[k]["punish_pass"]) and not cm[k]["punish_pass"] for k in ks)
        drop, flips = pun_c - pun_l >= spec.g_fail_drop, pf >= spec.g_fail_flips
        axes[ax] = dict(n=len(ks), pun_L=pun_l, pun_C=pun_c, pass_to_fail=pf, fail_to_pass=fp, drop=bool(drop),
                        flips=bool(flips), fail=bool(drop or flips))
    return dict(g_fail=any(a["fail"] for a in axes.values()), axes=axes)


# ================================================================ the bands (R.3, R.9.4, R.9.5)
def read_band(n, c, f_a, g, n_b, n_a, naive_a, spec) -> dict:
    """The first matching line of R.3, in its fixed order."""
    if n_b != spec.n_b or n_a != spec.n_a:
        return dict(band=NOT_READ, reason="COUNTS")
    if c >= spec.bar_b:
        return dict(band=B_NC, reason=C_ABOVE_BAR)
    if n <= c:
        return dict(band=B_TB, reason="NO_GAIN")
    if n < spec.bar_b:
        return dict(band=B_NC, reason=BELOW_BAR)
    if n - c < spec.margin:
        return dict(band=B_NC, reason=MARGIN)
    if g:
        return dict(band=B_PG, reason="PUNISH_GUARD")
    if f_a < spec.f_a_min:
        return dict(band=B_FA, reason="F_A", naive_a=naive_a, f_a_possible=bool(naive_a >= spec.f_a_min))
    return dict(band=SELECTED, reason="PASS")


# ================================================================ the closing sentences (R.7, R.9.3; Readings 6, 8, 9)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/32)"
SENTENCES = {
    STOP_STRENGTH_LEVER: "APL→MBON05 제거 아래에서 E-grid k2-norm s 1.0이 KC 활성 자격({cond})을 잃었다.",
    STOP_PUNISH_BROKEN: "APL→MBON05 제거 아래에서 P의 처벌 학습 확인이 재현되지 않았다({label}, ℓ_r1 {l1}, ℓ_r2 {l2}).",
    STOP_EVEN_LOW_LEVER: "APL→MBON05 제거 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다"
                         "({tb}/21, 지렛대 없는 같은 실행 {c_even}/21, F_a 기록 {k}/18).",
    STOP_C_EVEN_MISMATCH: "지렛대 없는 같은 실행의 짝수 (b) testable_b가 인코더 13절의 7/21과 달랐다({c_even}/21) — "
                          "측정 경로 결함 신호로 기록하고 멈춘다(R.9.6).",
    B_TB: "APL→MBON05 제거가 판정 세트(L 생성기 턴 64–{T})에서 {n}/21로 지렛대 없는 같은 세트 {c}/21보다 오르지 않았다."
          " → 이 지렛대를 닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: "M2 (b) 기준과 여유는 넘었지만 지렛대가 처벌 통과를 줄였다(처벌 통과 (b) {pb_L} 대 {pb_C}, (a) {pa_L} 대 {pa_C}).",
    B_FA: "M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다({n}/21 대 {c}/21, F_a {f_a}/32, naive_a {naive_a})."
          " 다음 병목은 F_a(순진 균형 (a) 쌍)다.",
    SELECTED: "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 판정 세트(L 생성기 턴 64–{T}, E-grid 키 "
              "중복 제외) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다({n}/21 대 "
              "{c}/21, 여유 ≥ 2, F_a {f_a}/32, 처벌 가드 통과, 짝수 {k_even}/21, P 재현 `LEARNS_CONFIRMATORY`, 판정 시드 "
              "24_100_xxx). 지렛대는 Q 결과를 보고 골랐고 Q의 사전 ② 일치 조건은 넘지 못했다(R.0). 실제 커넥톰 간선을 지운 "
              "모델이며 오라클 시험 가능성이지 학습 시험이 아니다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristic (R.6, R.9.5, Reading 13)
def g_axis(N: int, a_pf: float, a_fp: float, spec) -> float:
    """P(drop ∨ flips) on an axis of N independent pairs, each C pass ∧ L fail with probability a_pf and C fail ∧ L
    pass with a_fp (multinomial; drop ⇔ k_pf − k_fp ≥ g_fail_drop, flips ⇔ k_pf ≥ g_fail_flips)."""
    rest = 1 - a_pf - a_fp
    tot = 0.0
    for k in range(N + 1):
        for j in range(N - k + 1):
            if k - j >= spec.g_fail_drop or k >= spec.g_fail_flips:
                tot += comb(N, k) * comb(N - k, j) * a_pf ** k * a_fp ** j * rest ** (N - k - j)
    return tot


def g_prob(a_pf: float, a_fp: float, spec) -> float:
    """G_fail on (b) n_b or (a) n_a pairs, the axes independent."""
    return 1 - (1 - g_axis(spec.n_b, a_pf, a_fp, spec)) * (1 - g_axis(spec.n_a, a_pf, a_fp, spec))


def _cell(q: float, c: int, naive_a: int, g: float, spec) -> dict:
    pn, pf = e_rules._binom(spec.n_b, q), e_rules._binom(naive_a, q)
    P = {b: 0.0 for b in BANDS}
    for i, wi in enumerate(pn):
        for j, wj in enumerate(pf):
            for gv, wg in ((False, 1 - g), (True, g)):
                if wg:
                    P[read_band(i, c, j, gv, spec.n_b, spec.n_a, naive_a, spec)["band"]] += wi * wj * wg
    return P


def oc(spec) -> dict:
    """Exact conditional OC: n ~ Bin(n_b, q), F_a ~ Bin(naive_a, q), G_fail ~ Bernoulli(g), c fixed, each cell weighted
    through read_band. g rows: none, the null false alarm (a_pf = a_fp ∈ oc_discord) and harm rows (oc_harm)."""
    g_rows = ([dict(kind="null", a_pf=a, a_fp=a, p=g_prob(a, a, spec)) for a in spec.oc_discord]
              + [dict(kind="harm", a_pf=x, a_fp=y, p=g_prob(x, y, spec)) for x, y in spec.oc_harm])
    rows = []
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in range(min(spec.n_a, spec.oc_naive_max) + 1):
                for gr in [dict(kind="none", a_pf=0.0, a_fp=0.0, p=0.0)] + g_rows:
                    rows.append(dict(q=q, c=c, naive_a=na, g_kind=gr["kind"], a_pf=gr["a_pf"], a_fp=gr["a_fp"],
                                     g=gr["p"], P=_cell(q, c, na, gr["p"], spec)))
    table = dict(assumption="pairs independent; q_b = q_a; c fixed; G_fail independent of n and F_a with probability "
                            "g from a per-pair discordance model per axis, OR over (b) and (a)",
                 g_fail=g_rows, rows=rows)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_r_rules.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/r_rules.py tests/brain/test_r_rules.py
git commit -m "feat(r): r_rules — gates, G_fail, R.3 bands (exhaustive + boundary fixtures), closing sentences, exact OC"
```

---
### Task 7: `r_runner` (1) — context, chain and refusals, repro, smoke, OC, gates ①–③

**Files:**
- Create: `flymon/brain/r_runner.py`, `tests/brain/r_world.py`
- Test: `tests/brain/test_r_runner.py`

**Interfaces:**
- Consumes:
  - Tasks 1–6: `SPEC`, `smoke`, `r_pairs.*`, `RMeasurer`'s interface (`activity`, `oracle`, `arms`, `last_wall_s`, `last_jobs`), `r_store.*`, `r_records.*`, `r_rules.*`;
  - `p_rules.p_judge` (as the module global `r_runner.p_judge`, so tests can monkeypatch it), `p_cli.c1_source` / `nview`, `n_cli.stimuli_at` / `model_types`;
  - `e_runner.summary_git`, `h3_store.git_state`.
- Produces:
  - `ORDER`, `GATES`, `R_HASHED_FILES`, `git_state()`, `refuse(msg)`;
  - `p_items(pspec, st, edit, seeds=None) -> list`;
  - `build_ctx(spec, npz) -> dict` with keys `params, readout, z, types, n_kc, enc, calib, cap_ok, even_rows, judgement_rows (callable), kc_ref (callable oid -> basename | None), p_block, p_ref (callable wanted -> dict), p_inputs (callable (pspec, smoke) -> (c1 record, stimuli)), q0_ref (callable row -> {path, ok})`;
  - `Runner(measurer, ctx, spec, summary_path=None, code=None, archive_root=None)` with `stage_repro()`, `stage_smoke()`, `stage_oc()`, `stage_gate1()`, `stage_gate2(rerun=False)`, `stage_even()`, `stage_gate3()`;
  - block fields read later: `repro.passed`, `repro.csc_sha256_none`, `smoke.problems`, `oc.sha256`, `gate1/2/3.outcome`, `gate2.label`, `gate3.testable_b`, `even.conditions`.

- [ ] **Step 1: Write the test world**

```python
# tests/brain/r_world.py
"""A scripted R world for the runner tests (no engine, no pool): a fake ctx, a Scripted measurer that writes real
cache-like files (the seal and the judge re-read them), the reproduction gate's reference files, and a P judgement
stub (r_runner.p_judge is monkeypatched)."""
import hashlib
import json
from pathlib import Path

from flymon.brain import r_runner as RR
from flymon.brain.config import Params
from flymon.brain.r_spec import SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, key_of

SHA = {"L": "sha-L", "C": "sha-C", "E0": "sha-C"}


def calib():
    single = [SPEC.kc_repro_odours[0]] + [f"S{i}|T" for i in range(31)]
    dual = list(SPEC.kc_repro_odours[1:]) + [f"D{i}|T+U" for i in range(78)]
    return {o: {f"ORN_{i}": 1.0} for i, o in enumerate(single + dual)}, single, dual


def ref_name(oid):
    return oid.replace("|", "_").replace("+", "_") + ".json"


def arm_result(direction, arm, seed, edit="none"):
    probe = {"A": 10, "P": 11, "kc_frac": 0.05, "wall_s": 0.1}
    return dict(seed=int(seed), edit=edit, arm=arm, punish=arm == "punish", plastic=arm != "frozen", da_zero=False,
                csc_sha256="sha-L" if edit != "none" else "sha-C", pre={"x": dict(probe), "y": dict(probe)},
                post={"x": dict(probe), "y": dict(probe)}, weights_frac=0.0, weights_frac_A=0.0, weights_frac_P=0.0,
                w0_sha256="w", w_post_sha256="w", da_integral={}, wall_s=1.0)


class Scripted:
    """RMeasurer's interface. plan[(block, cond)][pair key] = (r_ok, p_ok, bal) (default: punish passes only);
    override[(block, cond)] = dict(edges=, edit=, sha=) changes one block's condition only; fail_once makes the next
    oracle call raise like an interrupted measurement."""

    def __init__(self, plan=None, kc=0.045, override=None):
        self.plan, self.kc, self.override = plan or {}, kc, override or {}
        self.kc_override, self.fail_once = {}, False
        self.calls, self.last_wall_s, self.last_jobs = [], 2.0, 1

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, len(odours)))
        lever = edit == SPEC.lever_edit
        v = self.kc if lever else 0.05
        return {o: dict(frac=list(self.kc_override.get(o, [v] * len(seeds))), max_win=[5] * len(seeds),
                        csc_sha256=["sha-L" if lever else "sha-C"], edit_edges=[SPEC.lever_edges if lever else 0])
                for o in odours}

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [dict(arm_result(i["direction"], i["arm"], i["seed"], i["edit"]),
                     r=dict(edit_edges=SPEC.lever_edges if i["edit"] == SPEC.lever_edit else 0, p_type="MBON05"),
                     direction=i["direction"], x=i["x"], y=i["y"], point=[float(v) for v in i["point"]])
                for i in items]

    def oracle(self, rows, cond, block, seeds):
        self.calls.append(("oracle", block, cond.name, len(rows)))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("worker died")
        o = self.override.get((block, cond.name), {})
        out = []
        for r in rows:
            k = key_of(r)
            flags = self.plan.get((block, cond.name), {}).get(k, (False, True, False))
            res = fake_oracle(*flags, sha=o.get("sha", SHA[cond.name]),
                              edges=o.get("edges", SPEC.lever_edges if cond.name == "L" else 0),
                              edit=o.get("edit", cond.edit), n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ck = hashlib.sha256(f"{block}|{cond.name}|{k}".encode()).hexdigest()
            f = Path("results/r/cache/r_oracle") / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": {}, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch, p_label="LEARNS_CONFIRMATORY"):
        monkeypatch.chdir(tmp_path)
        self.judge_commits, self.p_label, self.judgement_calls = [], p_label, 0
        monkeypatch.setattr(RR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(RR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(RR, "p_judge", self._p_judge)
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 32, turn0=64)
        odours, single, dual = calib()
        self.enc = {"strength": {"configs": {SPEC.config: {"table": {str(SPEC.strength): {
            "per_odour": {o: 0.05 for o in odours}}}}}}}
        d = tmp_path / SPEC.ref_dir
        d.mkdir(parents=True)
        for o in SPEC.kc_repro_odours:
            (d / ref_name(o)).write_text(json.dumps({"kind": "activity",
                                                     "result": {"frac": [0.05] * 8, "max_win": [5] * 8}}))
        q0 = tmp_path / "q0" / "0.json"
        q0.parent.mkdir()
        ref = fake_oracle(n_rep=8, n_act=8)
        q0.write_text(json.dumps({"kind": "oracle", "result": {k: v for k, v in ref.items() if k not in ("q", "r")}}))
        self.archive = tmp_path / "archive"
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, enc=self.enc,
                        calib=(odours, single, dual), cap_ok=True, even_rows=self.even,
                        judgement_rows=self._judgement_rows, kc_ref=ref_name, p_block={"measure_key": "K"},
                        p_ref=lambda wanted: {k: arm_result(*k) for k in wanted}, p_inputs=self._p_inputs,
                        q0_ref=lambda row: dict(path=str(q0), ok=True))

    def _judgement_rows(self):
        self.judgement_calls += 1
        return self.judge

    def _p_inputs(self, pspec, smoke_):
        names = {s for xy in pspec.pairs().values() for s in xy}
        return dict(c1=0.608), {n: {"odor": {f"G_{n}": 1.0}} for n in names}

    def _p_judge(self, rows, z, c1, spec):
        if self.p_label == "INVALID":
            return dict(outcome="INVALID", label="INVALID", reasons=["broken"], n_rows=len(rows))
        return dict(outcome="JUDGED", label=self.p_label, reasons=[], n_rows=len(rows),
                    directions={"r1": {"ell": 1.79}, "r2": {"ell": 2.16}})

    def runner(self, m, code="r" * 64):
        return RR.Runner(m, self.ctx, SPEC, code={"key": code}, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def pass_plan(self, n=14, c=8, f_a=3, even_l=12, even_c=7):
        """even: L even_l / C even_c (b) testable; judgement: L n / C c (b) testable and L f_a (a) testable and naive;
        punishment passes everywhere (no G_fail)."""
        eb, jb, ja = self.keys(self.even, "b"), self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("even", "L"): {k: (True, True, False) for k in eb[:even_l]},
                ("even", "C"): {k: (True, True, False) for k in eb[:even_c]},
                ("judge", "L"): {**{k: (True, True, False) for k in jb[:n]}, **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "C"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through_gate3(w, m, r=None):
    r = r or w.runner(m)
    assert r.stage_repro()["passed"]
    assert r.stage_smoke()["problems"] == []
    r.stage_oc()
    assert r.stage_gate1()["outcome"] == "PASS"
    assert r.stage_gate2()["outcome"] == "PASS"
    r.stage_even()
    assert r.stage_gate3()["outcome"] == "PASS"
    return r
```

- [ ] **Step 2: Write the failing test**

```python
# tests/brain/test_r_runner.py
"""The R stage chain up to gate ③ (R.2, R.5, R.9.6; Readings 3, 4, 6, 14-16): repro -> smoke -> oc -> gate1 -> gate2
-> even -> gate3; each stage refuses (exit 2, nothing written) when an earlier block is missing, a later one exists,
its own block exists (a failed repro and gate ②'s one INVALID rerun excepted), the summary is uncommitted or a hashed
file is dirty; a failed repro, a smoke problem or a gate STOP / INVALID blocks every later stage; the judgement set is
never touched before the judgement stages."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import r_runner as RR
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_spec import SPEC
from tests.brain.r_world import Scripted, World, doc, ref_name, through_gate3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate3_never_touches_the_judgement_set(w):
    m = Scripted(plan=w.pass_plan())
    through_gate3(w, m)
    d = doc()
    assert set(d) == set(RR.ORDER[:RR.ORDER.index("jm:L")])
    assert d["repro"]["kc"]["passed"] and d["repro"]["p"]["passed"] and d["repro"]["oracle"]["passed"]
    assert d["repro"]["csc_sha256_none"] == "sha-C" and len(d["repro"]["p"]["rows"]) == 18
    assert d["smoke"]["oracle"]["L"]["edit_edges"] == [2] and d["smoke"]["p"]["edit_edges"] == [2]
    assert d["gate1"]["record"]["median"] == pytest.approx(0.045) and d["gate1"]["record"]["original"]["ok"] is False
    assert d["gate2"]["label"] == "LEARNS_CONFIRMATORY" and d["gate2"]["seeds"] == list(SPEC.p.seeds)
    assert (d["gate3"]["testable_b"], d["gate3"]["c_even"]) == (12, 7)
    assert d["gate3"]["records"]["transitions_C_to_L"]["b"]["testable"]["fp"] == 5
    assert w.judgement_calls == 0
    assert not any(c[1] == "judge" for c in m.calls if c[0] == "oracle")
    assert all(c[3] == 2 for c in m.calls if c[0] == "oracle" and c[1] == "smoke")


def test_repro_refuses_without_refs(w):
    (Path(SPEC.ref_dir) / ref_name(SPEC.kc_repro_odours[1])).unlink()
    with pytest.raises(SystemExit) as e:
        w.runner(Scripted()).stage_repro()
    assert e.value.code == 2 and not Path(SPEC.summary).exists()


def test_repro_fails_on_a_wrong_reference_and_can_be_rerun(w):
    m = Scripted()
    m.kc_override[SPEC.kc_repro_odours[0]] = [0.05] * 7 + [0.06]
    r = w.runner(m)
    out = r.stage_repro()
    assert out["passed"] is False and not out["kc"]["odours"][0]["equal"]
    with pytest.raises(SystemExit):
        r.stage_smoke()
    m.kc_override.clear()
    assert r.stage_repro()["passed"] is True                  # a failed repro block may be rewritten (R.9.6)
    with pytest.raises(SystemExit):
        r.stage_repro()                                       # a passed one may not


def test_order_and_rewrite_refusals(w):
    r = w.runner(Scripted(plan=w.pass_plan()))
    with pytest.raises(SystemExit):
        r.stage_gate1()                                       # repro, smoke, oc missing
    assert not Path(SPEC.summary).exists()
    r.stage_repro()
    r.stage_smoke()
    with pytest.raises(SystemExit):
        r.stage_smoke()                                       # own block exists
    with pytest.raises(SystemExit):
        r.stage_even()                                        # oc, gate1, gate2 missing


def test_dirty_summary_or_hashed_file_refuses(w, monkeypatch):
    r = w.runner(Scripted())
    r.stage_repro()
    monkeypatch.setattr(RR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_smoke()
    monkeypatch.setattr(RR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(RR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/r_jobs.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_smoke()


def test_smoke_problems_block_later_stages(w):
    m = Scripted(override={("smoke", "C"): dict(edges=2, edit=SPEC.lever_edit, sha="sha-L")})
    r = w.runner(m)
    r.stage_repro()
    sm = r.stage_smoke()
    assert sm["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_gate1_stop_is_recorded_and_blocks_gate2(w):
    r = w.runner(Scripted(kc=0.02))
    r.stage_repro(); r.stage_smoke(); r.stage_oc()
    g1 = r.stage_gate1()
    assert g1["outcome"] == "STOP_STRENGTH_LEVER" and "0.0200" in g1["sentence"]
    with pytest.raises(SystemExit):
        r.stage_gate2()


def test_gate2_stop_and_the_one_invalid_rerun(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, p_label="NO_LEARNING")
    r = w.runner(Scripted())
    r.stage_repro(); r.stage_smoke(); r.stage_oc(); r.stage_gate1()
    g2 = r.stage_gate2()
    assert g2["outcome"] == "STOP_PUNISH_BROKEN" and "NO_LEARNING" in g2["sentence"]
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                             # only an INVALID block reruns
    with pytest.raises(SystemExit):
        r.stage_even()


def test_gate2_invalid_reruns_once_with_new_code_and_the_judgement_chain_accepts_it(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, p_label="INVALID")
    m = Scripted(plan=w.pass_plan())
    r = w.runner(m)
    r.stage_repro(); r.stage_smoke(); r.stage_oc(); r.stage_gate1()
    assert r.stage_gate2()["outcome"] == "INVALID"
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                             # same code key
    w.p_label = "LEARNS_CONFIRMATORY"
    r2 = w.runner(m, code="s" * 64)
    assert r2.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["outcome"] == "PASS"
    with pytest.raises(SystemExit):
        r2.stage_gate2(rerun=True)                            # once only
    r2.stage_even()
    assert r2.stage_gate3()["outcome"] == "PASS"
    assert r2.stage_jm("L")["n_pairs"] == 53                  # pre-gate2 blocks may carry the INVALID run's key


@pytest.mark.parametrize("even_l,even_c,outcome", [(16, 8, "STOP_C_EVEN_MISMATCH"), (10, 7, "STOP_EVEN_LOW_LEVER")])
def test_gate3_stops(w, even_l, even_c, outcome):
    r = w.runner(Scripted(plan=w.pass_plan(even_l=even_l, even_c=even_c)))
    r.stage_repro(); r.stage_smoke(); r.stage_oc(); r.stage_gate1(); r.stage_gate2(); r.stage_even()
    g3 = r.stage_gate3()
    assert g3["outcome"] == outcome and g3["sentence"]
    with pytest.raises(SystemExit):
        r.stage_jm("L")
    assert w.judgement_calls == 0


def test_p_items_are_run_p_items_with_the_edit():
    spec = importlib.util.spec_from_file_location("run_p_for_test", ROOT / "scripts/run_p.py")
    run_p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_p)
    st = {n: {"odor": {f"G_{n}": 1.0}} for xy in P_SPEC.pairs().values() for n in xy}
    assert RR.p_items(P_SPEC, st, "none") == run_p.items_p(P_SPEC, st)
    lev = RR.p_items(SPEC.p, st, SPEC.lever_edit)
    assert len(lev) == 192 and {i["edit"] for i in lev} == {SPEC.lever_edit}
    assert {i["seed"] for i in lev} == set(SPEC.p.seeds)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_build_ctx_on_the_real_data(monkeypatch):
    monkeypatch.chdir(ROOT)
    ctx = RR.build_ctx(SPEC, str(NPZ))
    assert len(ctx["even_rows"]) == 39 and len(ctx["calib"][0]) == 112 and ctx["cap_ok"] is True
    assert ctx["readout"] == {"A": "MBON13", "P": "MBON05"} and ctx["p_block"]["label"] == "LEARNS_CONFIRMATORY"
    names = [ctx["kc_ref"](o) for o in SPEC.kc_repro_odours]
    assert all(n and n.endswith(".json") for n in names) and len(set(names)) == 3
    f = ctx["q0_ref"]([r for r in ctx["even_rows"] if r["axis"] == "b"][0])
    assert f["path"].startswith(SPEC.q0_cache_dir)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.r_runner'` (`stage_jm`, used by two tests, arrives in Task 8; until then those two fail with `AttributeError` — expected, Task 8 turns them green).

- [ ] **Step 4: Write the implementation**

```python
# flymon/brain/r_runner.py
"""Spec R's stage chain (R.2-R.5, R.9; plan Readings 4, 6, 10, 14-18):
repro -> smoke -> oc -> gate1 -> gate2 -> even -> gate3 -> jm:L -> jm:C -> jm:E0 -> seal -> judge
(and after judge only: recompute / invalid_run), one block each in results/summary/r_lever.json, written only through
r_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists,
its own block exists (a failed repro and gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or
a hashed R file is dirty. A failed repro, a smoke with problems or a gate whose outcome is not PASS blocks every later
stage, so the judgement set stays unused (R.2). The judgement set is reached only through ctx["judgement_rows"]
(r_pairs.judgement_rows) from the judgement stages; jm blocks carry no pair statistic (Reading 10)."""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from statistics import median

import numpy as np

from ..agent.e_measure import MAX_ITEMS
from ..agent.e_runner import summary_git  # noqa: F401  (tests monkeypatch r_runner.summary_git)
from . import p_measure, r_records, r_rules, r_store
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge  # noqa: F401  (tests monkeypatch r_runner.p_judge)
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_spec import smoke

ORDER = ("repro", "smoke", "oc", "gate1", "gate2", "even", "gate3", "jm:L", "jm:C", "jm:E0", "seal", "judge")
GATES = ("gate1", "gate2", "gate3")
R_HASHED_FILES = tuple(dict.fromkeys(R_MEASURE_FILES + tuple(p_measure.HASHED_FILES) + (
    "flymon/brain/r_spec.py", "flymon/brain/r_pairs.py", "flymon/brain/r_records.py", "flymon/brain/r_rules.py",
    "flymon/brain/r_runner.py", "scripts/run_r.py", "flymon/agent/e_pairs.py", "flymon/agent/encode_grid.py",
    "flymon/agent/e_codebook.py", "flymon/agent/e_spec.py", "flymon/agent/e_rules.py", "flymon/agent/e_runner.py",
    "flymon/agent/config.py", "flymon/brain/h4_rules.py", "flymon/brain/h4_pairs.py", "flymon/brain/h4_spec.py",
    "flymon/brain/l_pairs.py", "flymon/brain/d6a.py", "flymon/brain/odor_real.py", "flymon/brain/q_pairs.py",
    "flymon/brain/q_spec.py", "results/summary/m0d.json")))


def git_state() -> dict:
    return _h3_git_state(files=R_HASHED_FILES)


def refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def p_items(pspec, st: dict, edit: str, seeds=None) -> list:
    """scripts/run_p.py:items_p's items (test: equal for P's spec with edit "none") with the edit and seeds given."""
    point = tuple(float(v) for v in pspec.o.o2_point)
    flags = pspec.flags()
    seeds = pspec.seeds if seeds is None else seeds
    return [dict(direction=d, x=x, y=y, edit=edit, arm=a, punish=flags[a][0], plastic=flags[a][1],
                 da_zero=flags[a][2], odor_x=st[x]["odor"], odor_y=st[y]["odor"], seed=int(s), point=point)
            for d, (x, y) in pspec.pairs().items() for a in pspec.arms for s in seeds]


def _rounds(n: int, w: int) -> int:
    return -(-int(n) // max(1, int(w)))


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, z — block h4), the H.4 pools as types, the encoder summary, the calibration odours and the
    ORN cap at s, the even rows, and lazy callables for the judgement rows (digest-checked), the encoder ③ reference
    names, P's reference entries and stimuli (c1 from block n1), and Q0's files."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from ..agent.e_runner import MEASURE_FILES_E
    from ..agent.e_spec import SPEC as E
    from . import n_cli, odor_real, p_cli, q_pairs, r_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_store import code_key
    from .q_spec import SPEC as Q_SPEC
    cfg = load_c3_config(spec.m0d_summary)
    if (cfg.readout["A"], cfg.readout["P"]) != (spec.a_type, spec.p_type):
        refuse(f"C3's readout is {cfg.readout}, not A {spec.a_type} / P {spec.p_type}")
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    calib = r_pairs.calibration_odours(rc, cb, rule)
    cap = r_pairs.cap_at(rc, cb, rule, spec.strength, cfg.params.max_rate_hz, odor_real.cap_hz(cfg.params))
    even = r_pairs.even_rows(pops, rc, enc, spec)
    enc_code = code_key(npz, files=MEASURE_FILES_E)
    p_doc = json.loads(Path(spec.p_summary).read_text())["p"]
    model_types = n_cli.model_types(npz)

    def kc_ref(oid):
        if enc_code["key"] != enc["strength"]["code_key"]:
            return None
        return r_pairs.encoder_activity_ref(oid, calib[0][oid], cfg.params, len(pops.kc), enc_code, spec)

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                calib=calib, cap_ok=cap, even_rows=even,
                judgement_rows=lambda: r_pairs.judgement_rows(pops, rc, enc, spec), kc_ref=kc_ref, p_block=p_doc,
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, q0_ref=lambda row: q_pairs.q0_files([row], enc, Q_SPEC)[0])


class Runner:
    def __init__(self, measurer, ctx: dict, spec, summary_path=None, code: dict | None = None, archive_root=None):
        self.m, self.ctx, self.spec = measurer, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return r_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first "
                       f"(a failed repro block: commit it or discard it before rerunning)")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed R files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str, allow_own: bool = False) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; R never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; R never rewrites a recorded block")
        if i > 0 and not doc["repro"].get("passed"):
            refuse(f"stage {stage}: the no-edit reproduction gate did not pass (R.9.6)")
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != r_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — R stops there and the "
                   f"judgement set stays unused (R.2)")
        return doc

    def _write(self, stage: str, body: dict) -> dict:
        block = dict(body, stage=stage, code_key=self.code_key, git=git_state(),
                     written_at=_dt.datetime.now(_dt.timezone.utc).isoformat())
        r_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    # ---- repro (R.9.6, Reading 4) ----------------------------------------------------------------------------------
    def stage_repro(self) -> dict:
        doc = self._doc()
        self._require("repro", allow_own="repro" in doc and not doc["repro"].get("passed"))
        kc, p, orc = self._repro_kc(), self._repro_p(), self._repro_oracle()
        body = dict(passed=bool(kc["passed"] and p["passed"] and orc["passed"]), kc=kc, p=p, oracle=orc,
                    csc_sha256_none=orc["csc_sha256"])
        self._write("repro", body)
        return body

    def _repro_kc(self) -> dict:
        sp = self.spec
        odours = self.ctx["calib"][0]
        names = {oid: self.ctx["kc_ref"](oid) for oid in sp.kc_repro_odours}
        if any(v is None for v in names.values()):
            refuse("the encoder code key differs from block strength's: encoder ③'s activity entries cannot be named")
        missing = sorted(n for n in names.values() if not (Path(sp.ref_dir) / n).exists())
        if missing:
            refuse(f"encoder ③'s activity entries {missing} are not under {sp.ref_dir}/: copy them from the "
                   f"encoder-redesign worktree's results/encoder/cache/activity/ (plan Runs step 0; --list-refs)")
        got = self.m.activity({o: odours[o] for o in names}, sp.no_edit, sp.strength, sp.kc_seeds(), "repro")
        per = self.ctx["enc"]["strength"]["configs"][sp.config]["table"][str(sp.strength)]["per_odour"]
        rows = []
        for oid, name in names.items():
            ref = json.loads((Path(sp.ref_dir) / name).read_text())["result"]
            mine = dict(frac=got[oid]["frac"], max_win=got[oid]["max_win"])
            rows.append(dict(odour=oid, file=name, equal=canonical(mine) == canonical(ref),
                             median_matches_block=float(median(mine["frac"])) == float(per[oid]),
                             edit_edges=got[oid]["edit_edges"]))
        ok = all(r["equal"] and r["median_matches_block"] and r["edit_edges"] == [0] for r in rows)
        return dict(passed=bool(ok), odours=rows, seeds=list(sp.kc_seeds()), wall_s=self.m.last_wall_s)

    def _repro_p(self) -> dict:
        sp = self.spec
        _, st = self.ctx["p_inputs"](P_SPEC, False)
        items = p_items(P_SPEC, st, sp.no_edit, sp.p_repro_seeds())
        wanted = {(i["direction"], i["arm"], int(i["seed"])) for i in items}
        ref = self.ctx["p_ref"](wanted)
        if set(ref) != wanted:
            refuse(f"P's committed cache {sp.p_cache_dir} lacks {sorted(wanted - set(ref))[:3]} under block p's key")
        rows = self.m.arms(items, self.ctx["readout"], P_SPEC.o.n.h3.punish_type, "repro", P_SPEC.o.n)
        out = []
        for r in rows:
            k = (r["direction"], r["arm"], int(r["seed"]))
            out.append(dict(key=list(k), edit_edges=int(r["r"]["edit_edges"]),
                            equal=canonical(r_records.strip_job(r)) == canonical(r_records.strip_job(ref[k]))))
        ok = len(out) == len(wanted) and all(o["equal"] and o["edit_edges"] == 0 for o in out)
        return dict(passed=bool(ok), rows=out, seeds=list(sp.p_repro_seeds()), wall_s=self.m.last_wall_s)

    def _repro_oracle(self) -> dict:
        sp = self.spec
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        rows = [b[i] for i in sp.oracle_repro_idx]
        got = self.m.oracle(rows, sp.cond(sp.cond_names[1]), "repro", sp.h4_seeds())
        out = []
        for r, g in zip(rows, got):
            f = self.ctx["q0_ref"](r)
            ref = json.loads(Path(f["path"]).read_text())["result"] if f["ok"] else None
            res = g["result"]
            mine = r_records.strip_job(res, top=("q", "r"))
            out.append(dict(key=row_key(r), q0_sha_ok=bool(f["ok"]),
                            equal=bool(ref is not None and canonical(mine) == canonical(ref)),
                            cell_sums_ok=r_records.cell_sums_ok(res), edit_edges=int(res["q"]["edit_edges"]),
                            csc_sha256=res["q"]["csc_sha256"]))
        shas = sorted({o["csc_sha256"] for o in out})
        ok = len(shas) == 1 and all(o["q0_sha_ok"] and o["equal"] and o["cell_sums_ok"] and o["edit_edges"] == 0
                                    for o in out)
        return dict(passed=bool(ok), pairs=out, csc_sha256=shas[0] if ok else None, seeds=sp.h4_seeds(),
                    wall_s=self.m.last_wall_s)

    # ---- smoke (R.8) -----------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        odours = self.ctx["calib"][0]
        ids = sorted(odours)[:sm.smoke_odours]
        act = self.m.activity({o: odours[o] for o in ids}, sp.lever_edit, sp.strength, sm.kc_seeds(), "smoke")
        kc = dict(odours=ids, edit_edges=sorted({e for v in act.values() for e in v["edit_edges"]}),
                  csc_sha256=sorted({s for v in act.values() for s in v["csc_sha256"]}),
                  median=float(np.median([x for v in act.values() for x in v["frac"]])), wall_s=self.m.last_wall_s)
        c1, st = self.ctx["p_inputs"](sm.p, True)
        rows = self.m.arms(p_items(sm.p, st, sp.lever_edit), self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke",
                           sm.p.o.n)
        res = p_judge(rows, self.ctx["z"], c1["c1"], sm.p)
        p = dict(outcome=res["outcome"], label=res["label"], reasons=list(res.get("reasons", [])),
                 edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rows}), n_rows=len(rows),
                 wall_s=self.m.last_wall_s)
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            got = self.m.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self.ctx["z"], [row_key(r) for r in sel], sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  kc_median=s["kc_median"], saturation=s["saturation"], wall_s=self.m.last_wall_s,
                                  jobs=self.m.last_jobs)
        detail = dict(kc=kc, p=p, oracle=orc, pairs=[row_key(r) for r in sel],
                      seeds=dict(kc=list(sm.kc_seeds()), p=list(sm.p.seeds), oracle=sm.even_seeds()))
        r_store.write_json(sp.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(kc, p, orc, doc["repro"]["csc_sha256_none"]),
                    cost=self._cost(kc, p, orc, sm), detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, kc, p, orc, repro_sha) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        bad = []
        if kc["edit_edges"] != [sp.lever_edges] or len(kc["csc_sha256"]) != 1:
            bad.append(f"KC activity: lever edges {kc['edit_edges']} on CSC {kc['csc_sha256']}")
        if p["edit_edges"] != [sp.lever_edges]:
            bad.append(f"P arms: lever edges {p['edit_edges']}")
        if p["outcome"] == P_INVALID:
            bad.append(f"P arms: INVALID {p['reasons'][:2]}")
        if orc[nl]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"L: lever edges {orc[nl]['edit_edges']}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not the repro's {repro_sha}")
        for n in sp.cond_names:
            if orc[n]["reasons"]:
                bad.append(f"{n}: {orc[n]['reasons'][:2]}")
        return bad

    def _cost(self, kc, p, orc, sm) -> dict:
        """A rough estimate from smoke wall times: rounds of workers × the seed ratio (recorded, never a rule)."""
        sp = self.spec
        item_s = kc["wall_s"] / max(1, len(kc["odours"]) * len(sm.kc_seeds()))
        kc_h = item_s * MAX_ITEMS * _rounds(_rounds(sp.n_calib * len(sp.kc_seeds()), MAX_ITEMS), sp.workers) / 3600
        per = len(sp.p.directions) * len(sp.p.arms)
        p_h = p["wall_s"] / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        ratio = sum(len(v) for v in sp.h4_seeds().values()) / sum(len(v) for v in sm.even_seeds().values())
        return dict(gate1_h=kc_h, gate2_h=p_h,
                    even_h=o_round * ratio * _rounds(sp.n_b + sp.n_a_even, sp.workers) * len(sp.cond_names) / 3600,
                    jm_per_condition_h=o_round * ratio * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristic (R.6, R.9.5) — before any gate ---------------------------------------------
    def stage_oc(self) -> dict:
        self._require("oc")
        body = r_rules.oc(self.spec)
        self._write("oc", body)
        return body

    # ---- gate ① (R.9.2) ----------------------------------------------------------------------------------------------
    def stage_gate1(self) -> dict:
        self._require("gate1")
        sp = self.spec
        odours, single, dual = self.ctx["calib"]
        act = self.m.activity(odours, sp.lever_edit, sp.strength, sp.kc_seeds(), "gate1")
        per = self.ctx["enc"]["strength"]["configs"][sp.config]["table"][str(sp.strength)]["per_odour"]
        rec = r_records.gate1_record(act, single, dual, per, sp)
        body = dict(r_rules.gate1(rec, self.ctx["cap_ok"], sp), record=rec, cap_ok=bool(self.ctx["cap_ok"]),
                    seeds=list(sp.kc_seeds()), wall_s=self.m.last_wall_s)
        self._write("gate1", body)
        return body

    # ---- gate ② (R.2, R.5, Reading 3) -----------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        doc, prior = self._doc(), None
        if rerun:
            blk = doc.get("gate2")
            if not blk or blk.get("outcome") != r_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (R.5)")
            if "gate2_invalid" in doc:
                refuse("gate2 was rerun once already (R.5, P.6.5)")
            if blk.get("code_key") == self.code_key:
                refuse("the code key equals the INVALID run's: fix the code first (R.5, P.6.5)")
            self._require("gate2", allow_own=True)
            prior = blk
            r_store.move_block(self.summary_path, "gate2", "gate2_invalid", self.plist)
        else:
            self._require("gate2")
        sp = self.spec
        c1, st = self.ctx["p_inputs"](sp.p, False)
        rows = self.m.arms(p_items(sp.p, st, sp.lever_edit), self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2",
                           sp.p.o.n)
        res = p_judge(rows, self.ctx["z"], c1["c1"], sp.p)
        dec = r_rules.gate2(res, sp)
        edges = sorted({int(r["r"]["edit_edges"]) for r in rows})
        if edges != [sp.lever_edges] and dec["outcome"] != r_rules.INVALID:
            dec = dict(outcome=r_rules.INVALID, label=dec.get("label"),
                       reasons=[f"the lever changed {edges} CSC edges, declared {sp.lever_edges}"])
        body = dict(dec, p_judgement=res, c1_source=c1, edit_edges=edges, seeds=list(sp.p.seeds),
                    rerun_of=(prior or {}).get("written_at"), wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        self._write("gate2", body)
        return body

    # ---- gate ③ (R.9.3, R.9.6, Reading 6) -----------------------------------------------------------------------------
    def stage_even(self) -> dict:
        self._require("even")
        sp = self.spec
        rows = self.ctx["even_rows"]
        keys, seeds = [row_key(r) for r in rows], sp.even_seeds()
        out = {}
        for cond in sp.conditions():
            got = self.m.oracle(rows, cond, "even", seeds)
            out[cond.name] = dict(r_records.cond_summary(got, cond, sp, self.ctx["z"], keys, seeds),
                                  wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        body = dict(conditions=out, seeds=seeds, n_pairs=len(rows))
        self._write("even", body)
        return body

    def stage_gate3(self) -> dict:
        doc = self._require("gate3")
        sp = self.spec
        ev = doc["even"]["conditions"]
        L, C, E0 = (ev[n] for n in sp.cond_names)
        body = dict(r_rules.gate3(L, C, sp, doc["repro"]["csc_sha256_none"]),
                    records=r_records.compare(L, C, E0, sp))
        self._write("gate3", body)
        return body
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_r_runner.py -q -k "not jm and not gate3_stops and not gate2_invalid"`
Expected: PASS (the three deselected tests need `stage_jm` from Task 8).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/r_runner.py tests/brain/r_world.py tests/brain/test_r_runner.py
git commit -m "feat(r): r_runner (1) — context, stage chain and refusals, no-edit repro, smoke, OC, gates 1-3"
```

---

### Task 8: `r_runner` (2) and `scripts/run_r.py` — judgement measurement, seal, judge, recovery; the CLI; mutation tests

**Files:**
- Modify: `flymon/brain/r_runner.py` (add methods to `Runner`, after `stage_gate3`)
- Create: `scripts/run_r.py`
- Test: `tests/brain/test_r_judge.py`

**Interfaces:**
- Consumes: Task 7's `Runner`, `ORDER`, `summary_git`, `git_state`, `refuse`; `r_records.raw_check` / `cond_summary` / `compare` / `preread_validity`; `r_store.load_manifest` / `archive_copy` / `write_summary_block`; `r_rules.g_fail` / `read_band` / `sentence` and the constants.
- Produces:
  - `Runner` methods: `_judge_chain(doc, stage)`, `_judgement_rows()`, `_raws(doc) -> (raws, bad)`, `_read(doc) -> dict`, `stage_jm(name)`, `stage_seal()`, `stage_judge()`, `_require_after_judge(stage)`, `stage_recompute(note)`, `stage_invalid_run(note)`;
  - block `jm:<n>`: `{reasons, edit_edges, csc_sha256, n_pairs, manifest[{key, cache_key, cache_file, sha256}], seeds, condition}`;
  - block `seal`: `{status, reasons, invalid, checks, manifest, n_files, set, archive}`;
  - block `judge`: `{status, band, reason, n, c, F_a, naive_a, n_b, n_a, g_fail, sentence, records, pairs, oc_sha256, p_label, k_even[, f_a_possible]}`;
  - `scripts/run_r.py`: `main(argv)`, `check_args(a)`, `exit_code(stage, out)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_r_judge.py
"""The judgement end of the R chain (R.3, R.5, R.9.7; Readings 10, 14, 15, 17, 18) and its mutation tests: jm blocks
carry no pair statistic and are written only when every pair is back; the seal re-checks every raw file, runs the
pre-read validity and archives the raw files; the judge reads only a sealed set, re-checks the raw files and writes the
band, the sentence and the records; a gate block deleted, L edges 1 / 3, an edit in C, a tampered digest, a changed
raw file or an earlier committed judge all refuse or mark the set; the judgement set is touched only from jm on."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import r_runner as RR
from flymon.brain import r_store as S
from flymon.brain.r_spec import SPEC
from tests.brain.r_world import Scripted, World, doc, through_gate3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through_gate3(w, m)
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def test_full_chain_reads_selected(w):
    m = Scripted(plan=w.pass_plan(n=14, c=8, f_a=3))
    r = _to_seal(w, m)
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 53 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 53
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm       # nothing to read before the seal
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 159
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive))
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    assert "(14/21 대 8/21, 여유 ≥ 2, F_a 3/32, 처벌 가드 통과, 짝수 12/21" in out["sentence"]
    assert out["g_fail"]["g_fail"] is False and out["records"]["transitions_C_to_L"]["b"]["testable"]["fp"] == 6
    assert out["oc_sha256"] == doc()["oc"]["sha256"] and set(doc()) == set(RR.ORDER)
    with pytest.raises(SystemExit):
        r.stage_judge()                                     # once


def test_punish_guard_band_through_the_runner(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    jb = w.keys(w.judge, "b")
    plan[("judge", "L")].update({k: (False, False, False) for k in jb[18:21]})       # 3 pass->fail on (b)
    r = _to_seal(w, Scripted(plan=plan))
    r.stage_seal()
    out = r.stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["b"]["pass_to_fail"] == 3
    assert "처벌 통과 (b) 18 대 21" in out["sentence"]


def test_judgement_set_is_touched_only_from_jm_on(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = Scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    m.fail_once = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 53


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))
    d = doc()
    del d["gate2"]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and w.judgement_calls == 0


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    m = Scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)})
    r = _to_seal(w, m)
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_an_edit_in_c_fails_the_preread_validity(w):
    m = Scripted(plan=w.pass_plan(), override={("judge", "C"): dict(edges=2, edit=SPEC.lever_edit, sha="sha-L")})
    r = _to_seal(w, m)
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any("C" in x for x in seal["reasons"])
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_a_tampered_digest_refuses(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("judgement set digest_keys: generated '00', block set '33be', declared '33be'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_judge_refuses_a_raw_file_changed_after_the_seal(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    assert r.stage_seal()["status"] == "SEALED"
    f = Path(doc()["jm:C"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"sha-C"', '"sha-X"'))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in doc()


def test_an_earlier_committed_judge_refuses_the_judgement(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))
    w.judge_commits.append("abc")
    with pytest.raises(SystemExit):
        r.stage_jm("L")


def test_a_changed_code_key_refuses_the_judgement(w):
    m = Scripted(plan=w.pass_plan())
    through_gate3(w, m)
    with pytest.raises(SystemExit):
        w.runner(m, code="z" * 64).stage_jm("L")


def test_recovery_after_reading(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    r.stage_seal()
    r.stage_judge()
    with pytest.raises(SystemExit):
        r.stage_recompute("")
    e = r.stage_recompute("records code fix (test)")
    assert e["band"] == "SELECTED" and e["differs_from_judge"] is False
    r.stage_recompute("second")
    assert len(doc()["recompute"]) == 2
    assert r.stage_invalid_run("measurement defect (test)")["status"] == "INVALID_RUN"
    with pytest.raises(SystemExit):
        r.stage_invalid_run("again")
    with pytest.raises(SystemExit):
        r.stage_recompute("after invalid")


WATCH = {"judgement_set", "judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds"}
ALLOWED = {"r_spec.py": {"judge_seeds"}, "r_pairs.py": {"judgement_rows"}, "r_records.py": {"preread_validity"},
           "r_runner.py": {"build_ctx", "_judgement_rows", "stage_jm", "stage_seal", "_read"}}


def _owners(tree) -> set:
    out = set()

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            if (isinstance(ch, ast.Name) and ch.id in WATCH) or (isinstance(ch, ast.Attribute) and ch.attr in WATCH):
                out.add(fn)
            visit(ch, ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn)
    visit(tree, None)
    return out


def test_only_the_judgement_stages_name_the_judgement_set():
    files = sorted((ROOT / "flymon/brain").glob("r_*.py")) + [ROOT / "scripts/run_r.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def _cli():
    spec = importlib.util.spec_from_file_location("run_r_for_test", ROOT / "scripts/run_r.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_argument_refusals_and_exit_codes(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    for argv in (["--stage", "gate1", "--condition", "L"], ["--stage", "jm"], ["--stage", "recompute"],
                 ["--stage", "smoke", "--note", "x"], ["--stage", "gate1", "--rerun-after-invalid"], [],
                 ["--list-refs", "--stage", "repro"], ["--stage", "repro"]):     # the last: wrong cwd
        assert cli.main(argv) == 2, argv
    assert cli.exit_code("repro", {"passed": True}) == 0 and cli.exit_code("repro", {"passed": False}) == 4
    assert cli.exit_code("gate1", {"outcome": "PASS"}) == 0
    assert cli.exit_code("gate1", {"outcome": "STOP_STRENGTH_LEVER"}) == 3
    assert cli.exit_code("gate2", {"outcome": "INVALID"}) == 5
    assert cli.exit_code("gate3", {"outcome": "STOP_C_EVEN_MISMATCH"}) == 3
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "NOT_READ"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_r.py").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/brain/test_r_judge.py -q`
Expected: FAIL (`AttributeError: 'Runner' object has no attribute 'stage_jm'`, and the CLI file is missing).

- [ ] **Step 3: Add the judgement methods to `Runner`** (in `flymon/brain/r_runner.py`, inside `class Runner`, after `stage_gate3`)

```python
    # ---- the judgement (R.3, R.5, R.9.7; Readings 10, 15) --------------------------------------------------------
    def _judge_chain(self, doc: dict, stage: str) -> None:
        """Reading 15: the judgement runs once, from clean trees, on the code every earlier block ran on (the blocks
        before gate2 may carry the INVALID gate2 run's key when gate ② was rerun after a fix)."""
        if self.spec.smoke:
            refuse("smoke never measures the judgement set (R.8)")
        if not self.code_key:
            refuse("no code key given; the judgement must run on the code the gates ran on")
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the judgement set is used once "
                   f"(R.5)")
        old = (doc.get("gate2_invalid") or {}).get("code_key")
        pre = set(ORDER[:ORDER.index("gate2")])
        for b in ORDER[:ORDER.index(stage)]:
            keys = {self.code_key} | ({old} if old and b in pre else set())
            if doc[b].get("code_key") not in keys:
                refuse(f"block {b}'s code key {doc[b].get('code_key')} is not the current {self.code_key}")
            d = (doc[b].get("git") or {}).get("dirty_hashed")
            if d is None or d:
                refuse(f"block {b} was written with dirty hashed files (or no git record): {d}")

    def _judgement_rows(self) -> list:
        try:
            return self.ctx["judgement_rows"]()
        except ValueError as e:
            refuse(f"the judgement set does not reproduce its declaration: {e}")

    def stage_jm(self, name: str) -> dict:
        """One judgement condition on the 53 pairs (resumable); the block holds raw_check only (Reading 10)."""
        sp = self.spec
        if name not in sp.cond_names:
            refuse(f"unknown condition {name}; R has {sp.cond_names}")
        stage = f"jm:{name}"
        doc = self._require(stage)
        self._judge_chain(doc, stage)
        rows = self._judgement_rows()
        seeds, cond = sp.judge_seeds(), sp.cond(name)
        got = self.m.oracle(rows, cond, "judge", seeds)
        rc = r_records.raw_check(got, cond, [row_key(r) for r in rows], seeds)
        man = [dict(m, sha256=sha256_file(m["cache_file"])) for m in rc["manifest"]]
        body = dict(rc, manifest=man, seeds=seeds, condition=name, wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        self._write(stage, body)
        return body

    def _raws(self, doc: dict) -> tuple:
        raws, bad = {}, []
        for n in self.spec.cond_names:
            got, b = r_store.load_manifest(doc[f"jm:{n}"]["manifest"])
            raws[n] = got
            bad += [f"{n}: {x}" for x in b]
        return raws, bad

    def stage_seal(self) -> dict:
        """R.9.7: pre-read validity; SEALED -> the archive copy. The block (manifest included) is committed before the
        judge reads anything."""
        doc = self._require("seal")
        self._judge_chain(doc, "seal")
        sp = self.spec
        keys = [row_key(r) for r in self._judgement_rows()]
        raws, bad = self._raws(doc)
        v = r_records.preread_validity({n: doc[f"jm:{n}"] for n in sp.cond_names}, raws, sp, self.ctx["z"], keys,
                                       self.code_key, doc["repro"]["csc_sha256_none"])
        reasons = bad + v["reasons"]
        status = r_rules.INVALID if v["invalid"] else (r_rules.NOT_READ if reasons else r_rules.SEALED)
        manifest = [dict(m, condition=n) for n in sp.cond_names for m in doc[f"jm:{n}"]["manifest"]]
        body = dict(status=status, reasons=reasons, invalid=v["invalid"], checks=v["checks"], manifest=manifest,
                    n_files=len(manifest), archive=None,
                    set=dict(digest_e0_b=sp.digest_e0_b, digest_e0_a=sp.digest_e0_a, digest_keys=sp.digest_keys,
                             last_turn=sp.last_turn, n_b=sp.n_b, n_a=sp.n_a))
        if status == r_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=r_store.archive_copy([m["cache_file"] for m in manifest], dest,
                                                                              self.archive_root))
        self._write("seal", body)
        return body

    def _read(self, doc: dict) -> dict:
        """The bands and records from the sealed raw files (sha re-checked): R.3's order with R.9.5's G_fail, R.7's
        sentence, R.6's records."""
        sp = self.spec
        keys = [row_key(r) for r in self._judgement_rows()]
        raws, bad = self._raws(doc)
        if bad:
            refuse(f"raw files changed since the seal: {bad[:3]}")
        seeds = sp.judge_seeds()
        s = {n: r_records.cond_summary(raws[n], sp.cond(n), sp, self.ctx["z"], keys, seeds) for n in sp.cond_names}
        L, C, E0 = (s[n] for n in sp.cond_names)
        if L["aggregate"] is None or C["aggregate"] is None:
            refuse("no (b) pair to aggregate")
        aL, aC = L["aggregate"], C["aggregate"]
        gf = r_rules.g_fail(L["pairs"], C["pairs"], sp)
        rb = r_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                               aL["naive_a"], sp)
        out = dict(band=rb["band"], reason=rb["reason"], n=aL["testable_b"], c=aC["testable_b"], F_a=aL["F_a"],
                   naive_a=aL["naive_a"], n_b=aL["n_b"], n_a=aL["n_a"], g_fail=gf)
        if rb["band"] == r_rules.NOT_READ:
            return dict(out, status=r_rules.NOT_READ, sentence=None)
        ax = gf["axes"]
        fields = dict(n=aL["testable_b"], c=aC["testable_b"], f_a=aL["F_a"], naive_a=aL["naive_a"], T=sp.last_turn,
                      k_even=doc["gate3"]["testable_b"], pb_L=ax["b"]["pun_L"], pb_C=ax["b"]["pun_C"],
                      pa_L=ax["a"]["pun_L"], pa_C=ax["a"]["pun_C"], reason=rb["reason"])
        if rb["band"] == r_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        return dict(out, status=r_rules.READ, sentence=r_rules.sentence(rb["band"], fields),
                    records=r_records.compare(L, C, E0, sp), pairs={n: s[n]["pairs"] for n in sp.cond_names},
                    oc_sha256=doc["oc"]["sha256"], p_label=doc["gate2"]["label"], k_even=doc["gate3"]["testable_b"])

    def stage_judge(self) -> dict:
        doc = self._require("judge")
        self._judge_chain(doc, "judge")
        if doc["seal"].get("status") != r_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: R reads only a sealed set (R.9.7)")
        out = self._read(doc)
        if out["status"] != r_rules.READ:
            return out                                   # NOT_READ: nothing written (R.5's first row)
        self._write("judge", out)
        return out

    # ---- recovery after reading (R.5, Reading 17) ------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (R.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: this set is closed (R.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """R.5 row 2: an analysis or summary defect after reading — the same sealed raw data recomputed, no
        measurement; appended to block recompute with the note."""
        if not note:
            refuse("--note is required (R.5: the correction is recorded)")
        doc = self._require_after_judge("recompute")
        out = self._read(doc)
        entry = dict(note=note, status=out["status"], band=out["band"], reason=out["reason"], n=out["n"], c=out["c"],
                     F_a=out["F_a"], naive_a=out["naive_a"], g_fail=out["g_fail"], sentence=out.get("sentence"),
                     differs_from_judge=bool(out["band"] != doc["judge"]["band"]
                                             or out.get("sentence") != doc["judge"].get("sentence")),
                     code_key=self.code_key, git=git_state(),
                     written_at=_dt.datetime.now(_dt.timezone.utc).isoformat())
        r_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """R.5 row 3: a measurement defect after reading — INVALID_RUN, the same set never runs again."""
        if not note:
            refuse("--note is required (R.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=r_rules.INVALID_RUN, note=note, judge_band=doc["judge"]["band"],
                    rule="R.5: the same set is never run again; a replacement set needs a new declaration (user)")
        self._write("invalid_run", body)
        return body
```

- [ ] **Step 4: Write the CLI**

```python
#!/usr/bin/env python3
"""Spec appendix R (R.9 wins over R.0-R.8): the lever APL->MBON05 removal and its one M2 judgement on the encoder
track's unused judgement set. The controller runs every stage; commit each block before the next.

    uv run python scripts/run_r.py --list-refs                       # encoder ③ entries to copy (no pool, no block)
    uv run python scripts/run_r.py --stage repro                     # R.9.6 no-edit reproduction gate
    uv run python scripts/run_r.py --stage smoke --workers 4         # 25_008_xxx / 25_009_xxx, even (b) 0 and 20 only
    uv run python scripts/run_r.py --stage oc                        # R.6 / R.9.5 operating characteristic (no pool)
    uv run python scripts/run_r.py --stage gate1                     # ① validity under the lever (R.9.2)
    uv run python scripts/run_r.py --stage gate2                     # ② P replication with the lever
    uv run python scripts/run_r.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (R.5)
    uv run python scripts/run_r.py --stage even                      # ③ L, C, E0 on H.4 seeds (one pool, resumable)
    uv run python scripts/run_r.py --stage gate3                     # ③ read (no pool)
    uv run python scripts/run_r.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_r.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_r.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_r.py --stage recompute --note "..."    # after judge: analysis defect (R.5)
    uv run python scripts/run_r.py --stage invalid_run --note "..."  # after judge: measurement defect (R.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (recorded; R stops), 4 repro failed, 5 a gate
INVALID, 6 seal not SEALED or judge NOT_READ, 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted
summary, dirty hashed file, data pins)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_REPRO_FAIL, EXIT_INVALID, EXIT_NOT_READ = 3, 4, 5, 6
STAGES = ("repro", "smoke", "oc", "gate1", "gate2", "even", "gate3", "jm", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("repro", "smoke", "gate1", "gate2", "even", "jm")
QUIET = ("pairs", "manifest", "records", "p_judgement", "conditions", "rows", "odours", "archive", "checks", "record",
         "g_fail", "rows_detail")


def check_args(a) -> str | None:
    if a.list_refs:
        return "--list-refs runs alone" if a.stage else None
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (R.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import r_rules as R
    if stage == "repro":
        return 0 if out.get("passed") else EXIT_REPRO_FAIL
    if stage in ("gate1", "gate2", "gate3"):
        o = out.get("outcome")
        return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)
    if stage == "seal":
        return 0 if out.get("status") == R.SEALED else EXIT_NOT_READ
    if stage == "judge":
        return 0 if out.get("status") == R.READ else EXIT_NOT_READ
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--condition", choices=("L", "C", "E0"))
    ap.add_argument("--workers", type=int)
    ap.add_argument("--note")
    ap.add_argument("--rerun-after-invalid", action="store_true")
    ap.add_argument("--list-refs", action="store_true")
    a = ap.parse_args(argv)
    why = check_args(a)
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2

    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, code_key, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2

    from flymon.brain import r_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
    from flymon.brain.r_spec import SPEC
    from flymon.brain.r_store import RCache

    try:
        ctx = r_runner.build_ctx(SPEC, NPZ)
        if a.list_refs:
            for oid in SPEC.kc_repro_odours:
                print(f"{oid}\t{ctx['kc_ref'](oid)}")
            return 0
        code = code_key(NPZ, files=R_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        root = SPEC.smoke_cache_dir if a.stage == "smoke" else SPEC.cache_dir
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            m = RMeasurer(pool, RCache(root, code), SPEC, ctx["params"], ctx["readout"], ctx["z"], ctx["types"],
                          ctx["n_kc"])
            r = r_runner.Runner(m, ctx, SPEC, code=code)
            if a.stage == "jm":
                out = r.stage_jm(a.condition)
            elif a.stage == "gate2":
                out = r.stage_gate2(rerun=a.rerun_after_invalid)
            elif a.stage in ("recompute", "invalid_run"):
                out = getattr(r, f"stage_{a.stage}")(a.note)
            else:
                out = getattr(r, f"stage_{a.stage}")()
        finally:
            if pool is not None:
                pool.close()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    except DataMismatch as e:
        print(f"refusing: the Hallem data do not match their pins: {e}", file=sys.stderr)
        return 2
    if out.get("sentence"):
        print(out["sentence"])
    print(json.dumps({k: v for k, v in out.items() if k not in QUIET}, ensure_ascii=False, default=str)[:2000])
    return exit_code(a.stage, out)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_r_judge.py tests/brain/test_r_runner.py -q`
Expected: PASS (including the three runner tests deselected in Task 7).

- [ ] **Step 6: Run the whole suite**

Run (controller: in the background, `run_in_background: true`; read only the exit code and the failure lines): `uv run pytest -q -x`
Expected: PASS, with no change to any existing test's result.

- [ ] **Step 7: Commit**

```bash
git add flymon/brain/r_runner.py scripts/run_r.py tests/brain/test_r_judge.py
git commit -m "feat(r): judgement measurement, seal (pre-read validity, manifest, archive), judge, recovery; run_r CLI; mutation tests"
```

---
## Self-review (done while writing)

- **Spec coverage:**

  | Spec item | Where it lives |
  |---|---|
  | R.0 / R.9.1 disclosures | no code; restated in the result section the user asks for after the run |
  | R.1 model, encoder, readout, z fixed to block `h4` | Task 1 (`Cond`, `LeverSpec`), Task 4 (`kwargs`: z from C3 for every condition), Task 7 (`build_ctx`) |
  | R.2 order, refusals, "commit before next" | Task 7 `_require` / `_clean`, Task 8 `_judge_chain`; Runs |
  | R.2 / R.9.2 gate ① (validity band, ORN cap, original qualification as a record) | Task 5 `gate1_record`, Task 6 `gate1`, Task 7 `stage_gate1` |
  | R.2 gate ② (P.6.2, seeds 25_000_000+i, c₁ from n1, `STOP_PUNISH_BROKEN`) | Task 1 `R_P`, Task 3 `r_arm_job`, Task 6 `gate2`, Task 7 `stage_gate2` (Reading 3) |
  | R.2 / R.9.3 / R.9.6 gate ③ (testable_b only, L and C in one run, c_even = 7) | Task 6 `gate3`, Task 7 `stage_even` / `stage_gate3` (Reading 6) |
  | R.3 set, digests, seeds, L / C / E0, n / c / F_a / naive_a | Task 2 `judgement_rows` / `check_set`, Task 8 `stage_jm` / `_read` |
  | R.3 bands, exhaustive and boundary fixtures; R.9.4 counterexample | Task 6 tests |
  | R.4 / R.9.5 punishment guard: G_fail, per-pair −p, pass→fail 2/3 | Task 6 `g_fail`, Task 5 `transitions`, Task 8 B_처벌가드 test |
  | R.5 recovery table, raw manifest, gate ② one rerun | Task 4 resume, Task 8 `stage_jm` / `stage_seal` / `stage_recompute` / `stage_invalid_run`, Task 7 rerun |
  | R.6 / R.9.7 records: KC, D.6 (a), APL, edges, CSC, per-cell MBON05 saturation, naive P/A, pass counts, transitions, E0 comparison, z renorm | Task 3 per-cell probe, Task 5 `cond_summary` / `compare` |
  | R.6 / R.9.5 OC including the G_fail false alarm, before the judgement | Task 6 `oc`, Task 7 `stage_oc` (before gate1) |
  | R.7 / R.9.3 closing sentences fixed in code | Task 6 `SENTENCES` (verbatim tests) |
  | R.8 R-only files, seeds, collision test, smoke never on the judgement set, cost, 2 h split | File Structure, Task 1, Task 8 AST test, Runs |
  | R.9.6 no-edit reproduction gate (KC, P, oracle) before smoke | Task 3, Task 7 `stage_repro` (Reading 4) |
  | R.9.7 pre-read validity, raw seal and archive, mutation tests | Task 5 `preread_validity`, Task 4 `archive_copy`, Task 8 seal and mutation tests |

- **Placeholders:** none. Every code step has its full code.
- **Type consistency:**
  - `row_key` is defined in `r_pairs` (the `q_pairs.key_str` format) and used by `r_measure` and `r_runner`. The fixtures' `key_of` matches it.
  - `cond_summary` / `raw_check` fields (`reasons`, `edit_edges`, `csc_sha256`, `n_pairs`, `manifest`, `pairs`, `aggregate`, `counts`) are what `gate3`, `compare`, `preread_validity`, `stage_jm` and `_read` read.
  - `RMeasurer.arms(items, readout, punish_type, block, nspec)` and `oracle(rows, cond, block, seeds)` match the Scripted measurer and every runner call.
  - Block fields read later are all written by the stage that owns them: `repro.csc_sha256_none`, `smoke.problems`, `oc.sha256`, `gate2.label`, `gate3.testable_b`, `jm:*.manifest[].sha256`, `seal.status`.
- **Review Focus:** each item has a test in its owning task:
  1. Task 4 resume and Task 8 `test_jm_writes_nothing_until_complete`;
  2. Task 8 `test_judge_refuses_a_raw_file_changed_after_the_seal`;
  3. Task 7 `test_repro_refuses_without_refs` / `test_repro_fails_on_a_wrong_reference_and_can_be_rerun`;
  4. Task 4 guard / archive tests;
  5. Task 2 smoke refusal and Task 8 `test_judgement_set_is_touched_only_from_jm_on` plus the AST test.

## Runs (controller)

The controller does every run, from `/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot` (branch `open-fly-brain-connectome`), with Bash `run_in_background: true` and `timeout: 7200000`, after the full suite passes at the final implementation commit.
- Subagents never run these steps.
- When a run finishes, read only its exit code and the last lines of its output.
- No rule or number changes after any run.
- Each block is committed before the next stage; the stage refuses otherwise.
- `results/summary/r_lever.json` is tracked; `results/r/` is git-ignored.
- Commit messages carry no trailers.

**The 2 h background limit:**
- The longest single runs are `even` (~70–90 min: L, C, E0 × 39 pairs) and each `jm` (~60–80 min: 53 pairs), which is why the judgement runs one condition per invocation.
- If a run is stopped at the limit, or dies, rerun the same command. It resumes from `results/r/cache/` with only the missing units, and no block is written until a stage is complete.
- Never run two pool stages at once.

**Stop points:** any STOP outcome (exit 3), a gate `INVALID` that cannot be rerun (gate ① / ③), seal `NOT_READ` / `INVALID` (exit 6), a failed repro after one diagnosed fix attempt, and the judgement result itself. At a stop point:
1. Commit the block.
2. Overwrite `/private/tmp/claude-503/p-latest.md` with the full text:
   - the stage and its label or band;
   - the closing sentence verbatim;
   - the numbers behind it;
   - the commits;
   - what was not done (spec result section, README ledger);
   - the decision R.7 leaves to the user.
3. Stop. Do not continue to another stage, and do not write a result section until the user decides.

0. **Preconditions (no pool):**
   1. `git status` must be clean and HEAD must be the final implementation commit.
   2. Check the references P and Q left: `ls results/p/run/cache/p_arm | wc -l` → 192 and `ls results/q/q0_cache | wc -l` → 21.
   3. Run `uv run python scripts/run_r.py --list-refs`. It prints three lines `odour<TAB>basename`, and no basename may be `None`.
   4. Copy the three encoder ③ entries (Reading 4):
      ```bash
      mkdir -p results/r/ref/activity ~/flymon-archive/r
      for f in <the three basenames>; do
        cp /Users/jeonsehyeon/orca/workspaces/fruit-fly/encoder-redesign/results/encoder/cache/activity/$f results/r/ref/activity/
      done
      ```
1. **Reproduction gate (R.9.6):**
   1. Run `uv run python scripts/run_r.py --stage repro`. It takes ~10–15 min at 16 workers.
   2. Expect exit 0 and `passed: true`:
      - `kc`: 3/3 `equal` and `median_matches_block`;
      - `p`: 18/18 `equal`;
      - `oracle`: 1/1 `equal`, `cell_sums_ok`;
      - `csc_sha256_none` equal to `results/summary/q_reward.json` → `repro.csc_sha256` (`1aee839811b8…`).
   3. On exit 4:
      1. Stop and diagnose with `superpowers:systematic-debugging`.
      2. Have a reviewed fix task.
      3. Commit the failed block as a record, or discard it (`git checkout -- results/summary/r_lever.json`, or delete it if untracked).
      4. Rerun.
   4. Commit: `git add results/summary/r_lever.json && git commit -m "results(r): no-edit reproduction gate — encoder ③ 3/3 odours, P 18/18 arms, Q0 oracle 1/1 bit for bit"`.
2. **Smoke (R.8):**
   1. Run `uv run python scripts/run_r.py --stage smoke --workers 4` (25_008_xxx / 25_009_1xx, even (b) pairs 0 and 20).
   2. Expect exit 0 and `problems: []`:
      - KC, P and L `edit_edges` are `[2]`; C and E0 are `[0]`;
      - the L CSC equals `q_reward.json` → `q1:apl_mbon05.csc_sha256` (`860cba4f9eea…`);
      - the C and E0 CSC equals the repro's.
   3. Report the `cost` estimate in one line.
   4. Commit: `git commit -m "results(r): smoke — lever edges 2 (KC, P, oracle), sha relations OK, cost estimate <even h> / <jm h per condition>"`.
3. **Operating characteristic (R.6, R.9.5):**
   1. Run `uv run python scripts/run_r.py --stage oc`. It takes seconds and needs no pool.
   2. Commit: `git commit -m "results(r): operating characteristic — bands × G_fail rows recorded before any gate"`.
4. **Gate ① (R.9.2):**
   1. Run `uv run python scripts/run_r.py --stage gate1`. It takes ~10–20 min.
   2. On exit 0 (PASS), commit: `git commit -m "results(r): gate 1 PASS — KC median <m> under the lever (original qualification <ok>, encoder ③ <m3>)"`.
   3. On exit 3 (`STOP_STRENGTH_LEVER`) or 5 (INVALID): commit `results(r): gate 1 <label> — <failed condition>`, then go to the **stop point**.
5. **Gate ② (R.2):**
   1. Run `uv run python scripts/run_r.py --stage gate2`. It takes ~15–20 min.
   2. On exit 0, commit: `git commit -m "results(r): gate 2 PASS — P replication with the lever LEARNS_CONFIRMATORY (ℓ_r1 <·>, ℓ_r2 <·>)"`.
   3. On exit 3 (`STOP_PUNISH_BROKEN`): commit, then go to the **stop point**.
   4. On exit 5 (INVALID):
      1. Commit the block.
      2. Fix the code with a reviewed task; the code key must change.
      3. Run `uv run python scripts/run_r.py --stage gate2 --rerun-after-invalid` once. Its label is final.
6. **Gate ③ measurement:**
   1. Run `uv run python scripts/run_r.py --stage even`. It takes ~70–90 min and resumes after a stop.
   2. Commit: `git commit -m "results(r): even oracle — L, C, E0 on H.4 seeds (39 pairs each)"`.
7. **Gate ③ read:**
   1. Run `uv run python scripts/run_r.py --stage gate3`. It needs no pool.
   2. On exit 0, commit: `git commit -m "results(r): gate 3 PASS — even testable_b <tb>/21 (C <c_even>/21)"`.
   3. On exit 3 (`STOP_EVEN_LOW_LEVER` / `STOP_C_EVEN_MISMATCH`) or 5: commit, then go to the **stop point**.
8. **Judgement measurement (one condition per run, in this order: L, C, E0):**
   1. Run `uv run python scripts/run_r.py --stage jm --condition <L|C|E0>`. Each takes ~60–80 min and resumes after a stop.
   2. The blocks hold no statistic. Do not compute anything from `results/r/cache/` before the seal.
   3. Commit after each: `git commit -m "results(r): judgement measurement <cond> — 53 pairs complete (no band read)"`.
9. **Seal (R.9.7):**
   1. Run `uv run python scripts/run_r.py --stage seal`. It needs no pool.
   2. On exit 0 (`SEALED`), check `find <seal.archive.dir> -type f | wc -l` → 159, then commit: `git commit -m "results(r): seal — pre-read validity OK, raw manifest (159 entries), archive copy ~/flymon-archive/r/<dir>"`.
   3. On exit 6 (`NOT_READ` / `INVALID`): commit, then go to the **stop point**. The judgement is not read.
10. **Judge:**
    1. Run `uv run python scripts/run_r.py --stage judge`. It needs no pool.
    2. Commit: `git commit -m "results(r): judgement — <band> (<n>/21 vs <c>/21, F_a <f>/32, G_fail <true|false>)"`.
    3. Go to the **stop point**:
       - The result goes to the user with the closing sentence verbatim and R.7's consequence:
         - B_Tb closes the lever;
         - `SELECTED`: F v4 learning test declaration is the user's;
         - the other bands are recorded for the user to judge.
       - The spec result section (`### R.10 결과`) and the README ledger lines (ko/en) are written only after the user's decision.
       - Pushing follows the standing FlyMon push rule (`gh auth switch --user lyutvs`). The verdict itself is reported, not acted on.
