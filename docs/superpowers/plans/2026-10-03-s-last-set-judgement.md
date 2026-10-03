# Spec S — The Same Lever on the Last Unused Set, One M2 Judgement: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix S needs (as overridden by S.9), and nothing else:
- a **reuse gate** (S.3 ①, S.9.5): R's no-edit reproduction (`22c934b`), gate ① (`6ad2201`) and gate ③ (`a83977f`) are reused only while the shared measurement key over `r_measure.R_MEASURE_FILES` equals R's `3c2699c7…` and R's summary holds those blocks, committed and passed. The condition is re-checked at every later stage, so also right before the judgement measurement, the seal and the judge.
- the **S set** (S.2), list only: L generator turns 104–209 minus `used_situations` and R's set keys; it must reproduce the declared T 177, (b) 21, (a) 43 and three digests, or the stage refuses; `STOP_SET_SHORT` when (b) runs out.
- **smoke** on new seeds 24_309_xxx (oracle, even (b) pairs 0 and 20) and 25_109_xxx (P_L and P_C). The judgement set is never measured here.
- the **operating characteristic** (S.6): G_fail_S and the bands on n_b 21 · n_a 43, beside R's guard on the same counts; then **gate ②'s OC** (S.9.7) from P's committed block. Both are recorded before gate ②.
- **gate ②** (S.3 ④, S.9.3): P_L (lever) and P_C (no edit) on the same new seeds 25_100_000+i (i < 32); L `LEARNS_CONFIRMATORY`, C `LEARNS_CONFIRMATORY`, ℓ_L / ℓ_C ≥ 0.5 in both directions, in that order.
- the **judgement** on the S set: L, C and E0, one condition per run, seeds 24_300_xxx, 64 pairs each.
- **seal** (R.9.7 as is): pre-read validity, the raw manifest (192 entries) committed, the decision-code hash pinned, an archive copy in `~/flymon-archive/s/`.
- **judge**: R.3's bands in fixed order (`r_rules.read_band` itself, n_a 43) with G_fail_S (net drop ≥ 3 on either axis). It reads once, with S.9.2's single re-generation after an interruption between the read mark and the judge block.

S is **one M2 judgement on the last set**. Its STOP labels and bands go to the user unchanged. There is no replacement set (S.5).

**Architecture:**
- New files only: `flymon/brain/s_*.py`, `scripts/run_s.py`, and tests. The one exception is a single `MODULES` line in `tests/brain/test_p_spec.py` (Reading 1).
- **No file of R is edited** — in particular none of `r_measure.R_MEASURE_FILES` (the shared key). S imports R's jobs, measurer, cache, records and band function and runs them unchanged.
- Modules:
  - `s_spec`: every S number (class `LastSetSpec`, a subclass of R's `LeverSpec`), the two gate-② P specs `P_L` / `P_C`.
  - `s_pairs`: the S set (list only), its check against S.2's declared values, the judgement rows.
  - `s_store`: S's guard and writer, `SCache` (R's `RCache` with S's writer and smoke seeds).
  - `s_rules`: the reuse / set / gate-② decisions, G_fail_S, the band function (R's), the sentences, the OC.
  - `s_records`: gate ②'s ratio with its paired bootstrap and gate ②'s OC.
  - `s_runner`: the stage chain.
- `scripts/run_s.py` is the CLI.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`, `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **S** (S.0–S.8) and its red-team amendment **S.9**, which wins wherever the two differ. S reuses appendix **R** (R.3 bands, R.5 recovery, R.6 / R.9.7 records, seal and validity, R.9.5's G_fail OC model) and R's implementation at `909c193` (judge-read marker and decision-code seal). Reused rules elsewhere: encoder spec 5.1 (the set rule), P.6.2 (`p_rules.p_judge`).

| S.9 section | Overrides / adds |
|---|---|
| S.9.1 | G_fail_S stays net drop ≥ 3 (disclosures only). SELECTED reads "처벌 순감소 가드(축마다 ≥ 3) 통과". |
| S.9.2 | New S.5 row: marker present, judge block absent → judge re-generated **once** (same sealed raw data, same decision code, no measurement), `resumed_after_mark: true` and the mark's time; a second re-generation refuses. Fault-injection tests. |
| S.9.3 | Gate ② measures **both** L and C on 25_100_000+i. |
| S.9.4 | F_a ≥ 2 kept (no code beyond n_a 43). |
| S.9.5 | Two keys per block: the shared measurement key (`R_MEASURE_FILES`) and the S pipeline key. Reuse ⇔ shared key = `3c2699c7…` ∧ R's repro / ① / ③ blocks committed. Re-checked right before the judgement measurement and the judge. |
| S.9.6 | P_L / P_C differ by one edit (fixture). A global seed-collision test over 24_300 / 24_309 / 25_100 / 25_109 and their training seeds. |
| S.9.7 | OC notes ((b) guard coupled with n; conditional on S's set). Gate ②'s OC (true ratio 0.4 / 0.5 / 0.61 / 0.83 / 1.0, normal approximation) before gate ②. SELECTED's ratio wording "(약화 정도 기록)". |
| S.9.8 | Implementation choice made here (Readings 2–4), and the task count and review cost below. |

**Task count and review cost (S.9.8):** 6 tasks. Each task is one implementer run and one reviewer run (the `sdd-implementer` / `sdd-reviewer` agents), then one final whole-branch review: **13 subagent runs**. Expected size: ~1,000 lines of new code and ~1,100 lines of tests (R was ~1,900 / ~1,900 over 8 tasks), all of it drafted and run in a detached scratch worktree while this plan was written: the S tests passed (100), the rest of the suite showed only the failures caused by the worktree's missing git-ignored `results/` caches, and the real CLI ran `reuse` → `set` → `smoke` (4 workers) → `oc` → `gate2_oc` on the real connectome and R's summary (PASS; set reproduced; smoke problem-free, L CSC `860cba4f…` = R's, C / E0 `1aee8398…`; cost estimate gate ② 0.29 h, jm 0.46 h per condition; both OC tables as declared). Gate ② and the judgement were not run there. No task needs a pool; the only slow tests are Task 2's (they load the connectome, ~10 s). Tasks 1–4 are independent of each other except for imports of `s_spec`; Task 5 needs 1–4; Task 6 needs 5.

## Global Constraints

- **Commits carry no trailers.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No R file changes; none of the shared measurement files changes.** `r_measure.R_MEASURE_FILES` (Q's and P's measure files, `r_jobs.py`, `r_measure.py`, `r_store.py`) keys the cache and R's gates are reused only while that key equals R's `3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc` (S.3 ①, S.9.5; test in Task 1). `r_spec`, `r_pairs`, `r_records`, `r_rules`, `r_runner`, `run_r.py` and R's tests are not edited either.
- **Never edit** `flymon/agent/e_*.py`, `encode_grid.py`, `q_*`, `p_*`, `o_*`, `n_*`, `h4_*`, `k_jobs`, `h3_*`, `d6a`, `plasticity`, `fly_pool`, `engine_cpu`, `scripts/run_q.py`, `scripts/run_p.py`, `scripts/run_r.py`. The one exception is Reading 1: one line in `tests/brain/test_p_spec.py`'s `MODULES`. No module global of another track is monkeypatched at runtime.
- **S never runs a copy of shared measurement code.** No S file defines a job, a measurer or a cache class of its own (only `SCache(RCache)` overriding `put`), and no S file loads code by path (`importlib`, `runpy`, `exec`) — test in Task 6.
- **Numbers live only in `s_spec`.** Every S number is a field or method of `flymon.brain.s_spec.SPEC` (`LastSetSpec`) or of `smoke(SPEC)`. R's are inherited from `LeverSpec`; encoder, Q and P numbers are read from their specs. Literal guards: `s_spec.py` may hold only Task 1's literal set; `s_rules.py` only {0, 1}; no S file holds an integer in 24_100_xxx, 24_002_xxx, 25_000_000–25_009_999 or 23_000_000–23_009_999.
- **The judgement set (L generator turns 104–177) is measured only through `s_pairs.judgement_rows`.** Inside S, only `s_runner.build_ctx` (lazy callables), `stage_set` (the list, no measurement), `_judgement_rows`, `stage_jm`, `stage_seal`, `_read` and `_cost` (seed counts only) name it or its seeds (AST test, Task 6). `judgement_rows` refuses a smoke spec.
- **Raw data of other tracks is read-only for S:** `results/summary/r_lever.json` (R's blocks), `results/p/run/cache/p_arm/` (P's entries, gate ②'s OC), `results/summary/encoder_grid.json`, `results/summary/p_learning.json`, `results/summary/m0d.json`. S never reads `results/r/`.
- **Writes are guarded.** Raw files go under `results/s/` (git-ignored by `results/*`); the summary is `results/summary/s_lever.json` (tracked). Anything else — `results/r/`, `results/p/`, `results/q/`, `results/encoder/`, R's or any other summary — is refused with SystemExit 2. The archive copy goes only under `~/flymon-archive/s/<seal id>/`, never over an existing directory. Every write is atomic.
- **Scripts:** tests find the repository with `ROOT = Path(__file__).resolve().parents[2]`; `scripts/run_s.py` has `if __name__ == "__main__":`; pools are never started from a heredoc or `python -c`.
- **Subagents never run a real or smoke stage and never write under `results/`.** Tests write only under `tmp_path`. The controller runs every stage (section "Runs (controller)").
- **Numbers from S / S.9, verbatim:**
  - **Model:** C3 (block `h4` Params) + `apl_to_mbon05_zero` (exactly **2** CSC edges). Readout A = MBON13, P = MBON05; z = block `h4` in every condition (z renormalisation is a record).
  - **Conditions:** L = lever + E-grid k2-norm s **1.0**; C = no edit + E-grid k2-norm s 1.0; E0 = no edit + E0 odours at s **0.35** (record only).
  - **Set:** L turns **104–209**, minus `e_pairs.used_situations` and R's set keys; (b) **21** (rest of the 21st (b)'s turn dropped), (a) **43**, T **177**; digests E0 (b) `2b92f40a448fe5552598ac8ab265fb5cdc50424e8de945a8a5be587c44553456`, E0 (a) `7b53467569821e3c3ffdbe2eb317843b22de033041717b9ec8d48fbc40cf8cc7`, keys `884d49cc5d7c29a9a37e17166a1220c52ccbc1675b79f14f7e5abe2e0ca91231`.
  - **Gate ②:** P_L and P_C on **25_100_000+i** (i < 32), smoke **25_109_100–103**, c₁ from block n1 (0.608). Order: either `INVALID` → `INVALID` (one rerun after a fix); L ≠ `LEARNS_CONFIRMATORY` → `STOP_PUNISH_BROKEN`; C ≠ → `STOP_P_REFERENCE`; ℓ_L / ℓ_C < **0.5** in r1 or r2 → `STOP_PUNISH_WEAKENED`; else PASS.
  - **Judgement seeds:** act **24_300_000+i**, select **24_300_100+i**, report **24_300_200+i** (i < 8). Smoke oracle seeds **24_309_000–24_309_099**.
  - **Bands (first match, R.3):** 1 (b) ≠ 21 or (a) ≠ 43 → `NOT_READ`; 2 c ≥ 11 → B_결론없음; 3 n ≤ c → B_Tb; 4 n < 11 → B_결론없음; 5 n − c < 2 → B_결론없음; 6 G_fail_S → B_처벌가드; 7 F_a < 2 → B_Fa; 8 `SELECTED`.
  - **G_fail_S:** on (b) or (a), over pairs in both L and C: pun_C − pun_L ≥ **3**, pun = #(−p ≥ 2). pass→fail / fail→pass / per-pair −p are records.
  - **OC (S.6):** G_fail_S null 0.02 / 0.05 / 0.10 / 0.15 → **0.034 / 0.145 / 0.281 / 0.362**; harm 0.10/0.02, 0.15/0.02, 0.20/0.05, 0.10/0.05 → **0.752 / 0.949 / 0.961 / 0.549**; R's guard on the same rows **0.163 / 0.489 / 0.888 / 0.988; 0.938 / 0.994 / 0.999 / 0.906**.
  - **Reuse:** shared key **`3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc`**; R blocks repro `22c934b`, gate ① `6ad2201`, gate ③ `a83977f` (k_even 16/21, c_even 7/21).

## Readings of the spec (decided here — the controller's rulings)

1. **`test_p_spec.py` gets one line.** Its `test_every_spec_module_is_enumerated` fails as soon as `flymon/brain/s_spec.py` exists. Task 1 adds `"flymon/brain/s_spec.py": "flymon.brain.s_spec"` to `MODULES` (the pattern Q and R used). It is the only edit to a non-S file; the reviewer flags it for the user.
2. **`LastSetSpec(LeverSpec)`.** S subclasses R's spec, so R's records (`r_records`), R's band function and R's measurer read S's numbers through the same field names. Only what S changes is restated:
   - the set (first_turn 104, set_last_turn = E's 209, n_a 43, last_turn 177, the three digests);
   - the judgement seeds and S's smoke seeds as **fields** (so test_p_spec's collector and the encoder's collision test see them — S.9.6); `judge_seeds()` reads them;
   - `p = P_L`, `p_c = P_C` (P's spec with `o2_seed0` 25_100_000 and `smoke_seed0` 25_109_000; P_L also replaces `o1_conditions[0]` with the lever, the edit `p_judge`'s gate expects);
   - `g_fail_drop = 3` (S's net drop) and `g_fail_flips = None` (S has no pass→fail branch, so R's `g_fail` / `g_prob` fail loudly on an S spec);
   - `oc_harm` with S.6's fourth row (0.10 / 0.05), `p_ratio_min 0.5`, `gate2_oc_rhos`, the reuse fields and S's paths.
   R's other inherited fields (`kc_repro_odours`, `valid_band`, `n_calib`, `c_even_expected`, `ref_dir`, …) stay but no S code reads them. The class is not named `RSpec` (test_p_spec's `_expand` special-cases that name).
3. **The store (S.8, S.9.5).** `r_store.py` is in the shared key and hardcodes `results/r/` in its module-level `guard`, which `RCache.put` calls through `write_json`; it cannot be edited. Decision: **`s_store.SCache` subclasses `RCache` and overrides only `put`** (routed through `s_store.write_json`, whose guard allows `results/s/` and `s_lever.json` only) and takes S's smoke seeds (24_309_xxx, 25_109_1xx) as the default scope. Key, path layout, `get` and the smoke/real scope are inherited unchanged: an S entry has R's layout and R's key formula over the same shared code key, but can only land under `results/s/cache/`, and the unchanged `RMeasurer` writes through it. `load_manifest` and `archive_copy` are path-free and re-exported as they are. Rejected: a wrapper object (`RMeasurer` calls `get / put / key / _path`, a subclass keeps them in one place) and a copy of `r_store` (two writers would drift).
4. **A different shared key is a STOP for the user, not a re-measurement.** S.9.5 says S would then re-run R's chain (repro → smoke → OC → ① → ② → even → ③). That would mean re-implementing R's repro, gate ①, even and gate ③ stages for S's paths (R's runner writes through `r_store` to `results/r/` and cannot be pointed elsewhere without edits). S edits no shared file, so a mismatch can only come from outside — another track editing a Q / P measure file, a Python / NumPy upgrade, another connectome — and that is a user decision. So: stage `reuse` records **`STOP_REUSE`** (exit 3) with the reasons; every later stage re-runs the same check and refuses with **exit 7** when it breaks (this is also the check "right before the judgement measurement and the judge"). The controller stops; re-measurement would need its own plan. The reuse condition also requires R's summary to be tracked and clean and its `repro` (passed), `gate1` and `gate3` (PASS) blocks to carry R's key.
5. **Stage `set` (S.2)** runs first after `reuse`, list only (no measurement, ~1 s). It generates the set exactly as encoder 5.1 / `e_pairs.judgement_set` does over turns 104–209, excluding `used_situations` and R's set keys — R's set is regenerated and its digests checked against R's declared values and the encoder block (`r_pairs.check_set`) before its keys are used. `STOP_SET_SHORT` is recorded (exit 3); any other mismatch with T / n_a / digests refuses with no block. `judgement_rows` regenerates and re-checks the set on every call (jm, seal, judge, recompute). The STOP_SET_SHORT sentence is the encoder's with S's turn range ("L 생성기 턴 104–209에서 …").
6. **Smoke (S.3 ②)** checks what S runs that R's smoke did not: P_L and P_C (P's smoke derivation, 25_109_100–103, edges 2 / 0, neither INVALID) and the oracle on even (b) pairs 0 and 20 for L / C / E0 with 24_309 seeds (L edges 2, C / E0 0, L ≠ C sha, C = E0 = R's unedited `repro.csc_sha256_none`). No KC activity smoke (gate ① is reused). It records a cost estimate (gate ② L+C, jm per condition).
7. **Bands are R's function.** `s_rules.read_band is r_rules.read_band`: S.4 keeps R.3's order and only n_a changes (a field). G_fail_S is S's own (`g_fail_s`). The exhaustive fixture covers n, c ∈ 0..21 × F_a ∈ 0..43 × G_fail_S, plus the boundaries S.4 lists. The OC uses R's model (`r_rules._cell`) with S's g, and records R's guard on the same pair counts (`r_rules.g_prob` on R's spec with n_a 43) beside each g row.
8. **Sentences.** S.7 verbatim with 〈…〉 → `{field}`; B_결론없음 = R.3's line + `" ({n}/21 대 {c}/21, F_a {f_a}/43)"`; B_Tb's range is "104–{T}"; SELECTED uses S.9.1's "처벌 순감소 가드(축마다 ≥ 3) 통과" and S.9.7's "같은 시드 P 비 ℓ_r1 {rho1}·ℓ_r2 {rho2} ≥ 0.5(약화 정도 기록)", and S.9.7's "작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 오선택률이 아니다." is placed before the last sentence. `STOP_REUSE` (Reading 4) is new. `STOP_PUNISH_BROKEN` is R's sentence.
9. **Gate ② (S.3 ④).** One pool call measures P_L's and P_C's 2 × 3 × 32 arms (block `gate2`, 384 entries, cache inputs differ by edit). `p_rules.p_judge` judges each on its own spec. The ratio ℓ_L / ℓ_C per direction is the point estimate the gate reads; ℓ is `p_judge`'s ℓ (mean of per-seed dl). A paired seed bootstrap (one `o_rules.boot_weights` matrix for both conditions, P's draws, seed and CI level) gives its CI as a record, beside ℓ_L(S) − ℓ_L(R gate ②, 25_000_xxx). Edges must be L [2] / C [0] (else INVALID). An INVALID gate ② reruns once (`--rerun-after-invalid`) only with a changed **pipeline** key; a fix in a shared file changes the shared key and refuses with exit 7 (S.9.5).
10. **Gate ②'s OC (S.9.7)** is stage `gate2_oc`, before gate ②, from P's committed block (192 entries, seeds 23_000_xxx, no lever): per direction ℓ_P and its bootstrap SE; for a true ratio ρ, SE(ρ̂) ≈ SE_P √(1 + ρ²) / ℓ_P (ℓ_C ≈ ℓ_P, ℓ_L = ρ ℓ_P, equal independent SEs) and P(STOP_PUNISH_WEAKENED) = 1 − Π_d Φ((ρ − 0.5) / SE_d). Computed from P's block while planning: **0.998 / 0.750 / 0.080 / 0.000 / 0.000** at ρ 0.4 / 0.5 / 0.61 / 0.83 / 1.0 (ℓ_P 1.787 / 2.158, SE 0.096 / 0.117). It never changes the gate.
11. **Keys.** Every block carries `code_key` = the shared measurement key (`h3_store.code_key(NPZ, files=R_MEASURE_FILES)`, which also keys the cache) and `pipeline_key` = a hash of S's files (`s_*.py`, `run_s.py`). `jm:*`, `seal` and `judge` refuse unless every earlier block carries the current shared key and was written with no dirty hashed file, and when the summary's git history already holds a `judge` block. The **decision code** = every S hashed file outside the shared key (S's files, R's decision files, the encoder / H.4 / L / Q pair and rule files, `m0d.json`, `r_lever.json`); `seal` pins its hash and `judge` reads only under it (R `909c193`).
12. **Read once, re-generate once (S.9.2).** `judge` writes `results/s/judge_read.json` before computing a band. After the judge block it writes `results/s/judge_done.json`. A later `judge` with the read marker and no judge block (an interruption between the two) re-generates once: the marker must name this seal's `written_at` and the sealed decision key, the decision key must still equal the seal's, and the raw files are re-checked against the manifest; it writes `results/s/judge_reread.json` before reading and the block carries `resumed_after_mark: true` and the mark's `read_at`. Refused: a re-read marker already present (second re-generation), the done marker present (a written block that was discarded does not reopen the set), a marker of another seal or decision code. `NOT_READ` writes no block and returns no number.
13. **Exit codes of `scripts/run_s.py`:** 0 PASS / SEALED / READ / a record stage; 2 a refusal; 3 a STOP (`STOP_REUSE`, `STOP_SET_SHORT`, gate ②'s three); 5 gate ② INVALID; 6 seal not SEALED, judge NOT_READ, or smoke with problems; 7 R's reuse condition broke after `reuse`.
14. **Records at judge (S.6):** `r_records.compare` (C → L and E0 → L transition tables of testable / reward / punish with per-pair −p, per-condition KC, D.6 (a), APL, edges, CSC, per-cell MBON05 saturation, naive A / A_X / P_X, α choices, z renormalisation), G_fail_S per axis (pun_L, pun_C, net drop, pass→fail, fail→pass), the gate-② labels and ratios, `k_even` from R's gate ③ and the OC sha.
15. **Recovery (S.5 = R.5):** `recompute --note` reruns the arithmetic from the sealed files (appends to block `recompute`); `invalid_run --note` writes `INVALID_RUN` once. There is no replacement set.

## Review Focus

1. **The shared key drifts between stages** (another track edits a Q / P measure file, or the venv is upgraded) after `reuse` passed. Expect every later stage, including `jm`, `seal` and `judge`, to refuse with exit 7 and write nothing. Tests: Task 5 (`test_reuse_broken_after_reuse_refuses_with_7`), Task 6 (`test_reuse_is_rechecked_before_measurement_seal_and_judge`).
2. **The judge killed between the read mark and the judge block.** Expect exactly one re-generation with the same band and sentence and `resumed_after_mark: true`; a second interruption ends the set. Tests: Task 6 (`test_resume_after_mark_once`, `test_a_second_regeneration_refuses`).
3. **A judgement measurement killed at the 2 h limit mid-condition.** Expect no `jm:*` block; a rerun measures only the missing pairs and writes the block when all 64 are back. Tests: Task 3 (`test_the_unchanged_measurer_writes_s_entries_and_resumes`), Task 6 (`test_jm_writes_nothing_until_complete`).
4. **Writing where R or another track lives** (`results/r/…`, `r_lever.json`, `results/p/…`, an archive outside `~/flymon-archive/s/`). Expect SystemExit 2. Test: Task 3.
5. **The S set generated differently** (a changed exclusion, R's set not reproducing, turns running out). Expect `set` to refuse (or record `STOP_SET_SHORT`) and `jm` to refuse on a tampered set. Tests: Task 2, Task 5 (`test_set_mismatch_refuses_without_a_block`), Task 6 (`test_mutation_a_tampered_set_refuses`).

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/s_spec.py` (create) | `LastSetSpec(LeverSpec)`, `P_L`, `P_C`, `SPEC`, `smoke()` |
| `flymon/brain/s_pairs.py` (create) | `STOP_SET_SHORT`, `OK`, `r_set_keys`, `s_set`, `check_s_set`, `summary`, `judgement_rows` |
| `flymon/brain/s_store.py` (create) | `ALLOWED_DIR`, `SUMMARY`, `SMOKE_SEEDS`, `guard`, `write_bytes`, `write_json`, `read_summary`, `write_summary_block`, `SCache`; re-exports `load_manifest`, `archive_copy` |
| `flymon/brain/s_rules.py` (create) | outcome / band constants, `reuse`, `set_outcome`, `gate2`, `g_fail_s`, `read_band` (= R's), `SENTENCES`, `sentence`, `g_axis`, `g_prob`, `OC_NOTES`, `oc` |
| `flymon/brain/s_records.py` (create) | `vectors`, `ratio`, `gate2_oc` |
| `flymon/brain/s_runner.py` (create) | `ORDER`, `S_PIPELINE_FILES`, `S_HASHED_FILES`, `DECISION_FILES`, markers, `pipeline_key`, `decision_key`, `build_ctx`, `Runner` |
| `scripts/run_s.py` (create) | the CLI, `exit_code`, `__main__` guard |
| `tests/brain/test_s_spec.py`, `test_s_pairs.py`, `test_s_store.py`, `test_s_rules.py`, `test_s_records.py`, `test_s_runner.py`, `test_s_judge.py`, `s_fixtures.py`, `s_world.py` (create) | tests and fixtures (`r_fixtures.py` is reused as is) |
| `tests/brain/test_p_spec.py` (modify: one `MODULES` entry) | Reading 1 |

---

### Task 1: `s_spec` — numbers, the two gate-② P specs, seed collisions, literal guards, the shared key

**Files:**
- Create: `flymon/brain/s_spec.py`
- Modify: `tests/brain/test_p_spec.py` (one `MODULES` entry)
- Test: `tests/brain/test_s_spec.py`

**Interfaces:**
- Consumes: `r_spec.LeverSpec` / `LEVER` / `SPEC`, `p_spec.SPEC` / `PSpec` / `smoke`, `q_jobs.MBON05`, `e_spec.SPEC`.
- Produces: `P_L`, `P_C` (PSpec); `LastSetSpec` with the new or overridden fields `first_turn, set_last_turn, n_a, last_turn, digest_e0_b, digest_e0_a, digest_keys, judge_act_seeds, judge_select_seeds, judge_report_seeds, smoke_seeds, p, p_c, p_ratio_min, gate2_oc_rhos, g_fail_drop, g_fail_flips, oc_harm, r_shared_key, r_summary, r_reused, r_commits, cache_dir, smoke_cache_dir, smoke_detail, summary, archive_root` and every `LeverSpec` field / method (`conditions()`, `cond()`, `even_seeds()`, …); `judge_seeds() -> dict` (ValueError when smoke); `SPEC`; `smoke(spec) -> LastSetSpec` (p, p_c → P smoke, `smoke=True`, workers 4).

- [ ] **Step 1: Add S to the spec-module list (Reading 1)**

In `tests/brain/test_p_spec.py`, the `MODULES` dict's last line is:

```python
           "flymon/brain/r_spec.py": "flymon.brain.r_spec"}
```

Replace it with (one entry added; nothing else in the file changes):

```python
           "flymon/brain/r_spec.py": "flymon.brain.r_spec", "flymon/brain/s_spec.py": "flymon.brain.s_spec"}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/brain/test_s_spec.py
"""Spec S.2 / S.3 / S.8 / S.9.5 / S.9.6: every S number in s_spec; S's new blocks (judgement 24_300_xxx, smoke
24_309_xxx, gate ② 25_100_000+i and its smoke 25_109_xxx, and their training seeds) collide with no declared seed of
any other spec module (test_p_spec's collector, R included); the two gate-② P specs differ from each other in the one
edit only and from P in their seed blocks; R's reused blocks are not S fields; no S file holds R's judgement, gate-①
or gate-② block as a literal; the shared measurement key over R_MEASURE_FILES is still R's 3c2699c7…."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.p_spec import SPEC as P
from flymon.brain.r_spec import SPEC as R
from flymon.brain.r_spec import LeverSpec
from flymon.brain.s_spec import P_C, P_L, SPEC, LastSetSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_s() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/s_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


S_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds))
S_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_s_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/s_spec.py") == "flymon.brain.s_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_300_000, 24_300_008)),
                                      select=list(range(24_300_100, 24_300_108)),
                                      report=list(range(24_300_200, 24_300_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_100_000, 25_100_032))
    assert len(SPEC.p.seeds) == P.o.o2_n_seeds
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_109_000 <= s < 25_110_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_309_000, 24_309_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """S.9.6: S's blocks and their derived training seeds against every other declared block (R, P, Q, the encoder,
    H.4 ... d6a) — no shared seed, no declared seed training on an S seed, no S training seed declared."""
    declared = _declared_without_s()
    for s in (500, 615, 24_002_000, 24_100_000, 24_100_207, 25_000_000, 25_008_000, 25_009_000, 23_000_000,
              24_000_000, 22_000_000):
        assert s in declared, s
    assert not S_SEEDS & declared
    assert not {s for s in S_SEEDS if (s - BASE) // STRIDE in declared}
    assert not S_TRAIN & declared
    assert not {s for s in S_TRAIN if (s - BASE) // STRIDE in declared}
    assert not S_TRAIN & S_SEEDS


def test_no_encoder_or_r_reused_seed_sits_in_an_s_field():
    fields = _module_seeds("flymon.brain.s_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & set(R.p.seeds) and not fields & set(range(500, 616))
    assert SPEC.judge_seeds() != R.judge_seeds()


def test_gate2_specs_differ_in_the_one_edit_only():
    """S.9.6: P_L and P_C are the same P spec but for o1_conditions[0] (the edit p_judge's gate expects)."""
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_L.o, f.name) != getattr(P_C.o, f.name)}
    assert diff == {"o1_conditions"}
    assert P_L.o.o1_conditions[1:] == P_C.o.o1_conditions[1:] == P.o.o1_conditions[1:]
    assert (P_L.o.on_edit, P_C.o.on_edit) == ("apl_to_mbon05_zero", "none") == (SPEC.lever_edit, SPEC.no_edit)
    diff_p = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_C.o, f.name) != getattr(P.o, f.name)}
    assert diff_p == {"o2_seed0", "smoke_seed0"}
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(P_L, f.name) == getattr(P_C, f.name) == getattr(P, f.name), f.name
    assert SPEC.p is P_L and SPEC.p_c is P_C


def test_numbers():
    assert isinstance(SPEC, LeverSpec) and isinstance(SPEC, LastSetSpec)
    assert (SPEC.first_turn, SPEC.set_last_turn, SPEC.last_turn, SPEC.n_b, SPEC.n_a) == (104, 209, 177, 21, 43)
    assert SPEC.digest_e0_b == "2b92f40a448fe5552598ac8ab265fb5cdc50424e8de945a8a5be587c44553456"
    assert SPEC.digest_e0_a == "7b53467569821e3c3ffdbe2eb317843b22de033041717b9ec8d48fbc40cf8cc7"
    assert SPEC.digest_keys == "884d49cc5d7c29a9a37e17166a1220c52ccbc1675b79f14f7e5abe2e0ca91231"
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.testable_min, SPEC.naive_max) == (11, 2, 2, 2.0, 0.5)
    assert (SPEC.g_fail_drop, SPEC.g_fail_flips, SPEC.p_ratio_min) == (3, None, 0.5)
    assert SPEC.gate2_oc_rhos == (0.4, 0.5, 0.61, 0.83, 1.0)
    assert SPEC.oc_discord == (0.02, 0.05, 0.1, 0.15)
    assert SPEC.oc_harm == ((0.1, 0.02), (0.15, 0.02), (0.2, 0.05), (0.1, 0.05))
    assert SPEC.r_shared_key == "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    assert SPEC.r_summary == R.summary == "results/summary/r_lever.json"
    assert SPEC.r_reused == ("repro", "gate1", "gate3")
    assert dict(SPEC.r_commits) == dict(repro="22c934b", gate1="6ad2201", gate2="78f514a", gate3="a83977f")
    assert (SPEC.summary, SPEC.cache_dir, SPEC.archive_root) == ("results/summary/s_lever.json", "results/s/cache",
                                                                 "~/flymon-archive/s")
    assert SPEC.conditions() == R.conditions() and SPEC.cond_names == ("L", "C", "E0")
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "lever_edges",
              "config", "strength", "e0_strength", "oc_q", "oc_c", "oc_naive_max", "sat_fracs", "smoke_pairs"):
        assert getattr(SPEC, f) == getattr(R, f), f


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(LastSetSpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4 and SM.p.o.on_edit == SPEC.lever_edit and SM.p_c.o.on_edit == SPEC.no_edit


ALLOWED_S_SPEC = {0.4, 0.5, 0.61, 0.83, 1.0, 0.1, 0.02, 0.15, 0.2, 0.05, 3, 43, 104, 177, 24_300_000, 24_300_008,
                  24_300_100, 24_300_108, 24_300_200, 24_300_208, 24_309_000, 24_309_100, 25_100_000, 25_109_000, 4}


def test_s_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/s_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_S_SPEC, nums - ALLOWED_S_SPEC


def test_no_s_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("s_*.py")) + sorted((ROOT / "scripts").glob("run_s*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_002_000 <= i < 24_003_000
               or 25_000_000 <= i < 25_010_000 or 23_000_000 <= i < 23_010_000}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_measurement_key_is_still_rs():
    """S.3 ① / S.9.5: the key over r_measure.R_MEASURE_FILES (+ the connectome, Python and NumPy versions) equals the
    key every R block carries; S edits none of those files."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_s_spec.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'flymon.brain.s_spec'` (and `test_p_spec.py::test_every_spec_module_is_enumerated` fails until the module exists).

- [ ] **Step 4: Write the implementation**

```python
"""Every number of spec appendix S as amended by S.9 (S.9 wins over S.0-S.8): the same lever (APL->MBON05 removal) on
the last unused judgement set (L generator turns 104-209), one M2 judgement.
- LastSetSpec subclasses R's LeverSpec, so R's records and bands (r_records, r_rules.read_band) read S's numbers
  through the same field names; only what S changes is restated here (plan Reading 2).
- S's new seed blocks are FIELDS (judgement 24_300_xxx, smoke 24_309_xxx, gate ② 25_100_000+i and smoke 25_109_xxx in
  the nested P specs), so every collision collector sees them (S.9.6). R's reused blocks stay in LeverSpec's methods.
- Gate ② (S.3 ④, S.9.3, S.9.6) runs two P specs on the same new block: P_L (the lever, the edit p_judge's gate expects
  through spec.o.on_edit) and P_C (no edit, P's own first condition). They differ in that one edit only.
- g_fail_drop is S's net-drop threshold (G_fail_S ⇔ pun_C − pun_L ≥ 3 on an axis); g_fail_flips is None because S has
  no pass->fail branch (r_rules.g_fail / g_prob, R's rule, would fail loudly on an S spec)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.e_spec import SPEC as E
from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .q_jobs import MBON05 as LEVER_EDIT
from .r_spec import LEVER, LeverSpec
from .r_spec import SPEC as R_SPEC

_S_O = dict(o2_seed0=25_100_000, smoke_seed0=25_109_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_S_O))
P_L = dataclasses.replace(P_SPEC, o=dataclasses.replace(
    P_SPEC.o, **_S_O, o1_conditions=((LEVER, LEVER_EDIT),) + P_SPEC.o.o1_conditions[1:]))


@dataclass(frozen=True)
class LastSetSpec(LeverSpec):
    # ---- the set (S.2) ------------------------------------------------------------------------------------------
    first_turn: int = 104
    set_last_turn: int = E.judge_last_turn          # 209: the L generator's last turn (encoder 5.1)
    n_a: int = 43
    last_turn: int = 177                            # T, declared before any measurement
    digest_e0_b: str = "2b92f40a448fe5552598ac8ab265fb5cdc50424e8de945a8a5be587c44553456"
    digest_e0_a: str = "7b53467569821e3c3ffdbe2eb317843b22de033041717b9ec8d48fbc40cf8cc7"
    digest_keys: str = "884d49cc5d7c29a9a37e17166a1220c52ccbc1675b79f14f7e5abe2e0ca91231"
    # ---- the judgement seeds (S.3 ⑤) and S's smoke (S.3 ②) ------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_300_000, 24_300_008))
    judge_select_seeds: tuple = tuple(range(24_300_100, 24_300_108))
    judge_report_seeds: tuple = tuple(range(24_300_200, 24_300_208))
    smoke_seeds: tuple = tuple(range(24_309_000, 24_309_100))
    # ---- gate ② (S.3 ④, S.9.3, S.9.6, S.9.7) ----------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    p_ratio_min: float = 0.5
    gate2_oc_rhos: tuple = (0.4, 0.5, 0.61, 0.83, 1.0)
    # ---- the guard (S.4, S.9.1) --------------------------------------------------------------------------------------
    g_fail_drop: int = 3
    g_fail_flips: object = None
    # ---- the operating characteristic (S.6) ---------------------------------------------------------------------------
    oc_harm: tuple = ((0.1, 0.02), (0.15, 0.02), (0.2, 0.05), (0.1, 0.05))
    # ---- R's reused gates (S.3 ①, S.9.5) -----------------------------------------------------------------------------
    r_shared_key: str = "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    r_summary: str = R_SPEC.summary
    r_reused: tuple = ("repro", "gate1", "gate3")
    r_commits: tuple = (("repro", "22c934b"), ("gate1", "6ad2201"), ("gate2", "78f514a"), ("gate3", "a83977f"))
    # ---- paths ------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/s/cache"
    smoke_cache_dir: str = "results/s/smoke/cache"
    smoke_detail: str = "results/s/smoke.json"
    summary: str = "results/summary/s_lever.json"
    archive_root: str = "~/flymon-archive/s"

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (S.3 ②)")
        return dict(act=list(self.judge_act_seeds), select=list(self.judge_select_seeds),
                    report=list(self.judge_report_seeds))


SPEC = LastSetSpec()


def smoke(spec: LastSetSpec = SPEC) -> LastSetSpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_109_100+), S's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_s_spec.py tests/brain/test_p_spec.py tests/brain/test_r_spec.py tests/agent/test_e_spec_store.py -q`
Expected: all pass (`test_shared_measurement_key_is_still_rs` needs `data/malecns.npz`; it passed at `e49c9df`).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/s_spec.py tests/brain/test_s_spec.py tests/brain/test_p_spec.py
git commit -m "feat(s): s_spec — S numbers on R's LeverSpec, P_L / P_C gate-2 specs, seed collisions, shared key test"
```

---

### Task 2: `s_pairs` — the S set (list only), its declaration check, the judgement rows

**Files:**
- Create: `flymon/brain/s_pairs.py`
- Test: `tests/brain/test_s_pairs.py`

**Interfaces:**
- Consumes: `e_pairs` (`judgement_set`, `used_situations`, `egrid_key`, `_rows_for_turn`, `_e0_digest`, `attach_odours`), `h4_pairs.pool_vocabulary` / `e0_channels`, `l_pairs.new_turns` / `alternate_from`, `q_pairs.codebook`, `r_pairs.check_set`, `r_spec.SPEC`, Task 1's spec fields.
- Produces: `STOP_SET_SHORT`, `OK`; `r_set_keys(pops, enc) -> set` (ValueError if R's set does not reproduce); `s_set(pops, enc, spec) -> dict(b, a, n_b, n_a, last_turn, skipped, status, digest_e0_b, digest_e0_a, digest_keys)`; `check_s_set(js, spec) -> list[str]`; `summary(js) -> dict`; `judgement_rows(pops, rc, enc, spec) -> list` ((b) 21 then (a) 43 rows with odours; ValueError on mismatch or smoke).

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_s_pairs.py
"""S.2: the S set generated from L turns 104-209 equals the declared set (T 177, (b) 21, (a) 43, three digests); its
keys meet none of the even / odd H.4 turns, L turns 0-63 or R's set; no key repeats inside it; every turn is in
104..T; R's set is digest-checked before its keys are used; STOP_SET_SHORT when the turns run out; a mismatch or a
smoke spec refuses judgement_rows."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.agent import e_pairs
from flymon.agent.e_spec import SPEC as E
from flymon.brain import s_pairs as SP
from flymon.brain.s_spec import SPEC, smoke

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


@pytest.fixture(scope="module")
def js(world):
    return SP.s_set(world[0], ENC, SPEC)


@needs_npz
def test_the_set_equals_the_declaration(js):
    assert SP.check_s_set(js, SPEC) == []
    assert (js["status"], js["n_b"], js["n_a"], js["last_turn"]) == ("OK", 21, 43, 177)
    assert js["digest_e0_b"].startswith("2b92f40a") and js["digest_keys"].startswith("884d49cc")
    assert set(SP.summary(js)) == {"status", "n_b", "n_a", "last_turn", "skipped", "digest_e0_b", "digest_e0_a",
                                   "digest_keys"}


@needs_npz
def test_keys_meet_no_used_situation_and_no_r_key(world, js):
    pops, _ = world
    keys = [e_pairs.egrid_key(r) for r in js["b"] + js["a"]]
    assert len(set(keys)) == len(keys)                                     # no duplicate inside the set
    used = {e_pairs.egrid_key(r) for r in e_pairs.used_situations(pops, E)}  # H.4 even + odd turns, L turns 0-63
    assert not set(keys) & used
    assert not set(keys) & SP.r_set_keys(pops, ENC)                        # R's set (turns 64-103)
    turns = [r["turn"] for r in js["b"] + js["a"]]
    assert SPEC.first_turn <= min(turns) and max(turns) <= js["last_turn"] <= SPEC.set_last_turn
    assert max(r["turn"] for r in js["b"]) == js["last_turn"]
    assert js["skipped"]["r_set"] > 0 and js["skipped"]["used"] > 0


@needs_npz
def test_r_keys_need_rs_digests(world, monkeypatch):
    pops, _ = world
    bad = json.loads(json.dumps(ENC))
    bad["set"]["digest_keys"] = "0" * 64
    with pytest.raises(ValueError, match="R's judgement set"):
        SP.r_set_keys(pops, bad)


@needs_npz
def test_stop_set_short_when_the_turns_run_out(world):
    short = dataclasses.replace(SPEC, set_last_turn=150)
    out = SP.s_set(world[0], ENC, short)
    assert out["status"] == SP.STOP_SET_SHORT and out["n_b"] < 21 and out["last_turn"] == 150
    assert SP.check_s_set(out, short)


@needs_npz
def test_judgement_rows_check_the_declaration_and_refuse_smoke(world):
    pops, rc = world
    rows = SP.judgement_rows(pops, rc, ENC, SPEC)
    assert [r["axis"] for r in rows] == ["b"] * 21 + ["a"] * 43
    assert all(set(r) >= {"odor_x", "odor_y", "odor_x_e0", "odor_y_e0"} for r in rows)
    with pytest.raises(ValueError, match="digest_keys"):
        SP.judgement_rows(pops, rc, ENC, dataclasses.replace(SPEC, digest_keys="0" * 64))
    with pytest.raises(ValueError, match="n_a"):
        SP.judgement_rows(pops, rc, ENC, dataclasses.replace(SPEC, n_a=42))
    with pytest.raises(ValueError, match="smoke"):
        SP.judgement_rows(pops, rc, ENC, smoke(SPEC))


def test_check_s_set_names_each_mismatch():
    js = dict(status="OK", n_b=21, n_a=43, last_turn=177, digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a,
              digest_keys=SPEC.digest_keys)
    assert SP.check_s_set(js, SPEC) == []
    for k, v in (("last_turn", 176), ("n_a", 44), ("digest_e0_a", "x")):
        assert any(k in m for m in SP.check_s_set(dict(js, **{k: v}), SPEC)), k
    assert SP.check_s_set(dict(js, status=SP.STOP_SET_SHORT, n_b=20), SPEC)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_s_pairs.py -q`
Expected: collection error `ImportError: cannot import name 's_pairs'`.

- [ ] **Step 3: Write the implementation**

```python
"""S's judgement set (S.2), list only — nothing here runs the engine.
- s_set: L generator turns first_turn..set_last_turn in declared order; a row whose E-grid key is in
  e_pairs.used_situations (H.4's 16 turns and L turns 0-63), in R's judgement set (turns 64-103, (b) 21 · (a) 32 — its
  three digests re-checked against R's declared values and the encoder block set first, r_pairs.check_set) or already
  in the set is skipped; (b) rows are kept until n_b and the rest of that turn is dropped (encoder 5.1,
  e_pairs.judgement_set's loop); (a) = every remaining (a) row of turns first_turn..T. Status STOP_SET_SHORT when the
  (b) rows run out before set_last_turn.
- check_s_set: the generated set against S's declared T, n_a and digests (S.2: a mismatch refuses).
- judgement_rows: the checked set's (b) 21 then (a) 43 rows with E-grid k2-norm odours; refused for a smoke spec. Only
  S's judgement stages call it (plan Global Constraints)."""
from __future__ import annotations

import hashlib
import json

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from . import h4_pairs, l_pairs, q_pairs, r_pairs
from .r_spec import SPEC as R_SPEC

STOP_SET_SHORT = "STOP_SET_SHORT"
OK = "OK"


def r_set_keys(pops, enc: dict) -> set:
    """The E-grid keys of R's judgement set, after its digests, n_a and last turn match R's declared values and the
    encoder block set (ValueError otherwise)."""
    js = e_pairs.judgement_set(pops, E)
    bad = r_pairs.check_set(js, enc, R_SPEC)
    if bad:
        raise ValueError("R's judgement set does not reproduce: " + "; ".join(bad))
    return {e_pairs.egrid_key(r) for r in js["b"] + js["a"]}


def s_set(pops, enc: dict, spec) -> dict:
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    used = {e_pairs.egrid_key(r) for r in e_pairs.used_situations(pops, E)}
    r_keys = r_set_keys(pops, enc)
    taken, b, a, skipped, last_turn = set(), [], [], {"used": 0, "r_set": 0, "in_set": 0, "after_21st": 0}, None
    turns = l_pairs.new_turns(st, mi, E.l_total_turns, E.l_rng_seed)
    for t in turns[spec.first_turn:spec.set_last_turn + 1]:
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        rows = e_pairs._rows_for_turn(pops, chan, t, alt, st, with_a=True)
        for ix, r in enumerate(rows):
            k = e_pairs.egrid_key(r)
            if k in used:
                skipped["used"] += 1
                continue
            if k in r_keys:
                skipped["r_set"] += 1
                continue
            if k in taken:
                skipped["in_set"] += 1
                continue
            taken.add(k)
            (b if r["axis"] == "b" else a).append(r)
            if len(b) == spec.n_b:
                skipped["after_21st"] += len(rows) - ix - 1
                break
        last_turn = t["turn"]
        if len(b) == spec.n_b:
            break
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in b + a]
    return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=last_turn, skipped=skipped,
                status=OK if len(b) == spec.n_b else STOP_SET_SHORT,
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest())


def check_s_set(js: dict, spec) -> list:
    """S.2: [] when the generated set equals the declared one (status OK, (b) n_b, T, n_a, three digests)."""
    bad = []
    if js["status"] != OK or js["n_b"] != spec.n_b:
        bad.append(f"S set status {js['status']}, (b) {js['n_b']}, declared {spec.n_b}")
    for k, want in (("last_turn", spec.last_turn), ("n_a", spec.n_a), ("digest_e0_b", spec.digest_e0_b),
                    ("digest_e0_a", spec.digest_e0_a), ("digest_keys", spec.digest_keys)):
        if js[k] != want:
            bad.append(f"S set {k}: generated {js[k]!r}, declared {want!r}")
    return bad


def summary(js: dict) -> dict:
    """What block `set` records: counts, T, skips, digests (the declared values, already public in S.2)."""
    return {k: js[k] for k in ("status", "n_b", "n_a", "last_turn", "skipped", "digest_e0_b", "digest_e0_a",
                               "digest_keys")}


def judgement_rows(pops, rc: dict, enc: dict, spec) -> list:
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (S.3 ②)")
    js = s_set(pops, enc, spec)
    bad = check_s_set(js, spec)
    if bad:
        raise ValueError("; ".join(bad))
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_s_pairs.py -q`
Expected: 6 passed (~10 s: the connectome loads once per module). The generated set gives `status OK, n_b 21, n_a 43, last_turn 177`, skips `used 231, r_set 32, in_set 16, after_21st 2`, and the three declared digests.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/s_pairs.py tests/brain/test_s_pairs.py
git commit -m "feat(s): s_pairs — S set from L turns 104-209 minus used and R keys, declared T/n_a/digest check, judgement rows"
```

---

### Task 3: `s_store` — S's guard and writer, `SCache` on R's `RCache`

**Files:**
- Create: `flymon/brain/s_store.py`
- Test: `tests/brain/test_s_store.py`

**Interfaces:**
- Consumes: `r_store.RCache` / `load_manifest` / `archive_copy` (unchanged), `h3_store.canonical` / `canonical_pretty`, `pool_bench` guards, Task 1's `SPEC` / `smoke`.
- Produces: `ALLOWED_DIR = "results/s/"`, `SUMMARY = "results/summary/s_lever.json"`, `SMOKE_SEEDS`; `guard(path, params_list)`, `write_bytes`, `write_json(path, obj, params_list) -> Path`, `read_summary(path=SUMMARY) -> dict`, `write_summary_block(path, block, obj, params_list)`; `SCache(root, code, smoke_seeds=SMOKE_SEEDS)` with RCache's `key / _path / get` and S's `put`; `load_manifest`, `archive_copy` (r_store's objects).

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_s_store.py
"""S's writer (S.5, S.8; plan Reading 3): writes only under results/s/ and results/summary/s_lever.json, atomically —
every other track's path (results/r/, results/p/, results/q/, results/encoder/, R's summary) refuses; SCache is
RCache with S's writer and S's smoke seeds (smoke and real entries never share a root; a truncated or foreign entry is
missing); r_store's load_manifest and archive_copy are reused as they are (archive only under S's root); and the
unchanged RMeasurer writes S entries under results/s/ and resumes with only the missing units."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_store
from flymon.brain import s_store as S
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.s_spec import SPEC, smoke

READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/r/x.json", "results/r/cache/r_oracle/a.json", "results/p/x.json", "results/q/x.json",
                "results/encoder/x.json", "results/summary/r_lever.json", "results/summary/q_reward.json",
                "results/summary/p_learning.json", "results/sx/a.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            S.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(S.write_json("results/s/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    S.write_summary_block(S.SUMMARY, "reuse", {"n": 1}, [])
    S.write_summary_block(S.SUMMARY, "set", {"m": 2}, [])
    assert S.read_summary() == {"reuse": {"n": 1}, "set": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert S.SUMMARY == SPEC.summary


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/s").mkdir(parents=True)
    outside.mkdir()
    (root / "results/s/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        S.write_json("results/s/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_scache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = S.SCache(SPEC.cache_dir, {"key": "k"}), S.SCache(SPEC.smoke_cache_dir, {"key": "k"})
    assert isinstance(real, r_store.RCache)
    assert S.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_300_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_300_000]}) == {"v": 1}
    d = json.loads(real._path("r_oracle", {"act_seeds": [24_300_000]}).read_text())
    assert d["key"] == real.key("r_oracle", {"act_seeds": [24_300_000]}) and d["inputs"] == {"act_seeds": [24_300_000]}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_109_100})                   # S's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_300_000]})           # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [24_309_000, 24_300_000]})   # mixed
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [25_008_000]})               # R's smoke seed is not S's
    sm.put("r_arm", {"seed": 25_109_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_109_100}) == {"v": 2}
    p = real._path("r_oracle", {"act_seeds": [24_300_000]})
    p.write_text("{trunc")
    assert real.get("r_oracle", {"act_seeds": [24_300_000]}) is None
    p.write_text(json.dumps({"key": "other", "kind": "r_oracle", "result": {}}))
    assert real.get("r_oracle", {"act_seeds": [24_300_000]}) is None
    with pytest.raises(SystemExit):
        S.SCache("results/r/cache", {"key": "k"}).put("r_oracle", {"act_seeds": [24_300_000]}, {"v": 1}, [])


def test_scache_keys_equal_rcaches():
    """Same key formula and layout as R (ECache over the shared code key) — only the root and the writer differ."""
    a, b = S.SCache("results/s/cache", {"key": "k"}), r_store.RCache("results/r/cache", {"key": "k"})
    ins = {"act_seeds": [500]}
    assert a.key("r_oracle", ins) == b.key("r_oracle", ins)
    assert a._path("r_oracle", ins).name == b._path("r_oracle", ins).name


def test_manifest_and_archive_are_r_stores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert S.load_manifest is r_store.load_manifest and S.archive_copy is r_store.archive_copy
    f = Path("results/s/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    man = [dict(key="b|104|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))]
    got, bad = S.load_manifest(man)
    assert bad == [] and got[0]["result"] == {"v": 1}
    root = tmp_path / "arch-s"
    out = S.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == f.read_text()
    with pytest.raises(SystemExit):
        S.archive_copy([str(f)], tmp_path / "arch-r" / "x", root)


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is r_jobs.r_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


ROWS = [dict(axis="b", turn=104 + i, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_the_unchanged_measurer_writes_s_entries_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = RMeasurer(pool, S.SCache(SPEC.cache_dir, {"key": "k"}), SPEC, Params(), READOUT, Z, TYPES, 100)
    got = m.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    assert all(g["cache_file"].startswith("results/s/cache/r_oracle/") for g in got)
    stored = json.loads(Path(got[0]["cache_file"]).read_text())["inputs"]
    assert stored["act_seeds"] == SPEC.judge_seeds()["act"] and stored["edit"] == SPEC.lever_edit
    Path(got[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = m.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    assert pool.calls == [("r_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in got]
    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=25_100_000, point=(0.25, 8.0))
    rows = m.arms([item, dict(item, edit=SPEC.lever_edit)], READOUT, "PPL105", "gate2", P_SPEC.o.n)
    assert [r["edit"] for r in rows] == ["none", SPEC.lever_edit] and len(list(Path("results/s/cache/r_arm").glob(
        "*.json"))) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_s_store.py -q`
Expected: collection error `ImportError: cannot import name 's_store'`.

- [ ] **Step 3: Write the implementation**

```python
"""S's only writer (S.5, S.8): raw files under results/s/, the one summary results/summary/s_lever.json, atomic writes
(temporary file + rename). r_store.py is part of the shared measurement key (r_measure.R_MEASURE_FILES), so it is never
edited (S.3 ①, S.9.5); S reuses it where it is path-free and wraps it where it is not (plan Reading 3):
- guard / write_bytes / write_json / read_summary / write_summary_block: r_store's code with S's allowed paths.
- SCache: r_store.RCache (ECache's key over the shared code key, RCache's smoke scope and its get) with only `put`
  routed through this module's writer and S's smoke seeds as the default scope — so an S entry has R's layout and
  key formula but can only land under results/s/.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for s_runner)
from .s_spec import SPEC as S_SPEC
from .s_spec import smoke

ALLOWED_DIR = "results/s/"
SUMMARY = "results/summary/s_lever.json"
SMOKE_SEEDS = frozenset(S_SPEC.smoke_seeds) | frozenset(smoke(S_SPEC).p.seeds) | frozenset(smoke(S_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"S writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


class SCache(RCache):
    """RCache with S's smoke seeds as the default scope and S's writer (a root outside results/s/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_s_store.py tests/brain/test_r_store_measure.py -q`
Expected: all pass (R's store tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/s_store.py tests/brain/test_s_store.py
git commit -m "feat(s): s_store — writes only under results/s/ and s_lever.json; SCache = RCache with S's writer and smoke seeds"
```

---

### Task 4: `s_rules` and `s_records` — gates, G_fail_S, bands, sentences, both OCs, gate ②'s ratio

**Files:**
- Create: `flymon/brain/s_rules.py`, `flymon/brain/s_records.py`
- Test: `tests/brain/s_fixtures.py`, `tests/brain/test_s_rules.py`, `tests/brain/test_s_records.py`

**Interfaces:**
- Consumes: `r_rules` (constants, `read_band`, `SENTENCES`, `_cell`, `g_prob`), `e_rules.STOP_SET_SHORT`, `p_rules` (`LEARNS_CONFIRMATORY`, `group`, `arm_vectors`, `p_judge` in tests), `n_rules` (`INVALID`, `_ci`), `o_rules.boot_weights`, `r_spec.SPEC`, `r_runner.p_items` (fixtures), Task 1's spec.
- Produces:
  - `s_rules`: `PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN, STOP_REUSE, STOP_SET_SHORT, STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED, SELECTED, B_TB, B_FA, B_NC, B_PG, BANDS, C_ABOVE_BAR, BELOW_BAR, MARGIN`; `read_band` (R's); `reuse(r_doc, r_git, shared_key, spec) -> dict(outcome, reasons[, sentence])`; `set_outcome(js) -> dict`; `gate2(res_l, res_c, ratio, spec) -> dict(outcome, reasons, label_L, label_C[, low, sentence])`; `g_fail_s(lp, cp, spec) -> dict(g_fail, axes{b,a: n, pun_L, pun_C, net_drop, pass_to_fail, fail_to_pass, fail})`; `SENTENCES`, `sentence(outcome, fields)`; `g_axis`, `g_prob`, `OC_NOTES`, `oc(spec) -> dict(assumption, notes, g_fail[{kind, a_pf, a_fp, p, r_guard}], rows, sha256)`.
  - `s_records`: `vectors(rows, z, pspec)`, `ratio(rows_l, rows_c, z, spec) -> {direction: ell_L, ell_C, ratio, ratio_ci, n, seeds}`, `gate2_oc(p_rows, z, pspec, spec) -> dict(model, source, directions, rows[{rho, se_ratio, p_stop_weakened}])`.
  - `tests/brain/s_fixtures.py`: `Z`, `C1`, `arm_row(item, drop, sha, edges)`, `stimuli(pspec)`, `p_rows(pspec, edit, drop, sha="sha-C", edges=0)`.

- [ ] **Step 1: Write the fixtures and the failing tests**

```python
"""Fabricated P arm rows for S's gate-② tests (no engine): arm_row gives r_arm_job + RMeasurer.arms' shape for one
item; only the punish arm (③) lowers X's A after training, by drop + seed % 3, so per seed dl = (drop + seed % 3) /
z_A's SD and ℓ ≈ (drop + 1) / 9 under Z — p_rules.p_judge confirms a direction when ℓ ≥ c₁ (0.608, drop ≥ 5) and its
s / t conditions hold (X falls, Y does not move). p_rows builds a whole P grid (two directions × three arms × seeds)."""
from flymon.brain.r_runner import p_items

Z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}
C1 = 0.608


def arm_row(item: dict, drop: float, sha: str, edges: int) -> dict:
    k = float(drop + int(item["seed"]) % 3) if item["punish"] else 0.0
    probe = {"A": 30.0, "P": 26.0, "kc_frac": 0.05, "kc_spikes": 100}
    post_x = dict(probe, A=30.0 - k)
    return dict(seed=int(item["seed"]), edit=item["edit"], arm=item["arm"], punish=bool(item["punish"]),
                plastic=bool(item["plastic"]), da_zero=bool(item["da_zero"]), csc_sha256=sha,
                pre={"x": dict(probe), "y": dict(probe)}, post={"x": post_x, "y": dict(probe)}, weights_frac=0.0,
                weights_frac_A=0.0, weights_frac_P=0.0, w0_sha256="w0",
                w_post_sha256="w0" if not item["plastic"] else "w1", da_integral={}, wall_s=1.0,
                r=dict(edit_edges=int(edges), p_type="MBON05"), direction=item["direction"], x=item["x"], y=item["y"],
                point=[float(v) for v in item["point"]])


def stimuli(pspec) -> dict:
    names = {s for xy in pspec.pairs().values() for s in xy}
    return {n: {"odor": {f"G_{n}": 1.0}} for n in names}


def p_rows(pspec, edit: str, drop: float, sha: str = "sha-C", edges: int = 0) -> list:
    return [arm_row(i, drop, sha, edges) for i in p_items(pspec, stimuli(pspec), edit)]
```

```python
# tests/brain/test_s_rules.py
"""S.2 / S.3 / S.4 / S.6 / S.7 / S.9: every cell n, c ∈ 0..21 × F_a ∈ 0..43 × G_fail_S reads exactly one band, equal
to an independent transcription of S.4's ordered lines; the boundary fixtures (c 10/11, n = c / c+1 / c+2, n 10/11,
F_a 1/2, net drop 2/3 on each axis, a symmetric swap 3/3 -> no G_fail, counts 20/22 and 42/44); the reuse gate, the
set outcome, gate ②'s order (INVALID, L, C, ratio 0.5 per direction); the sentences verbatim; the OC exact and equal
to S.6's declared values beside R's guard on the same pair counts."""
import dataclasses
import itertools

import pytest

from flymon.brain import r_rules
from flymon.brain import s_rules as R
from flymon.brain.s_spec import SPEC


def _ref_band(n, c, f, g, nb=21, na=43):
    lines = [(nb != 21 or na != 43, R.NOT_READ), (c >= 11, R.B_NC), (n <= c, R.B_TB), (n < 11, R.B_NC),
             (n - c < 2, R.B_NC), (g, R.B_PG), (f < 2, R.B_FA), (True, R.SELECTED)]
    return next(b for cond, b in lines if cond)


def test_every_cell_reads_exactly_the_transcribed_band():
    seen = set()
    for n, c, f, g in itertools.product(range(22), range(22), range(44), (False, True)):
        b = R.read_band(n, c, f, g, 21, 43, f, SPEC)["band"]
        assert b == _ref_band(n, c, f, g), (n, c, f, g)
        seen.add(b)
    assert seen == set(R.BANDS) and R.read_band is r_rules.read_band


@pytest.mark.parametrize("n,c,f,g,band,reason", [
    (13, 10, 2, False, R.SELECTED, "PASS"),
    (13, 11, 2, False, R.B_NC, R.C_ABOVE_BAR),            # c 10 / 11
    (8, 8, 5, False, R.B_TB, "NO_GAIN"),                   # n = c
    (9, 8, 5, False, R.B_NC, R.BELOW_BAR),                 # n = c + 1, below 11
    (10, 8, 5, False, R.B_NC, R.BELOW_BAR),                # n = 10
    (11, 8, 5, False, R.SELECTED, "PASS"),                 # n = 11
    (11, 10, 5, False, R.B_NC, R.MARGIN),                  # n = c + 1 at the bar
    (12, 10, 5, False, R.SELECTED, "PASS"),                # n = c + 2
    (13, 8, 1, False, R.B_FA, "F_A"),                      # F_a 1
    (13, 8, 2, False, R.SELECTED, "PASS"),                 # F_a 2
    (13, 8, 1, True, R.B_PG, "PUNISH_GUARD"),              # G_fail_S before F_a
    (11, 10, 5, True, R.B_NC, R.MARGIN),                   # margin before G_fail_S
])
def test_boundaries(n, c, f, g, band, reason):
    out = R.read_band(n, c, f, g, 21, 43, max(f, 4), SPEC)
    assert (out["band"], out["reason"]) == (band, reason)


@pytest.mark.parametrize("nb,na", [(20, 43), (22, 43), (21, 42), (21, 44), (21, 32)])
def test_counts_other_than_21_43_are_not_read(nb, na):
    assert R.read_band(13, 8, 5, False, nb, na, 5, SPEC)["band"] == R.NOT_READ


def _pp(ax, c_pass, l_pass, n=12):
    cp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in c_pass) for i in range(n)]
    lp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in l_pass) for i in range(n)]
    return lp, cp


@pytest.mark.parametrize("ax", ["b", "a"])
def test_net_drop_2_vs_3(ax):
    lp, cp = _pp(ax, set(range(6)), set(range(4)))           # 6 -> 4: net drop 2
    out = R.g_fail_s(lp, cp, SPEC)
    assert out["g_fail"] is False and out["axes"][ax]["net_drop"] == 2
    lp, cp = _pp(ax, set(range(6)), set(range(3)))           # 6 -> 3: net drop 3
    out = R.g_fail_s(lp, cp, SPEC)
    assert out["g_fail"] is True and out["axes"][ax]["fail"] and out["axes"][ax]["net_drop"] == 3
    other = "a" if ax == "b" else "b"
    assert out["axes"][other]["n"] == 0 and out["axes"][other]["fail"] is False


@pytest.mark.parametrize("ax", ["b", "a"])
def test_a_symmetric_swap_never_fails(ax):
    lp, cp = _pp(ax, set(range(5)), {3, 4, 5, 6, 7})         # pass->fail 3, fail->pass 3 (R's (a) at R.10)
    out = R.g_fail_s(lp, cp, SPEC)["axes"][ax]
    assert (out["pass_to_fail"], out["fail_to_pass"], out["net_drop"], out["fail"]) == (3, 3, 0, False)
    lp, cp = _pp(ax, set(range(10)), set(range(2, 10)) | {10, 11}, n=20)   # 2 -> fail, 2 -> pass: big swap, drop 0
    assert R.g_fail_s(lp, cp, SPEC)["g_fail"] is False


def test_only_pairs_in_both_conditions_count():
    lp, cp = _pp("a", set(range(6)), set())
    out = R.g_fail_s(lp[:2], cp, SPEC)["axes"]["a"]
    assert (out["n"], out["pun_C"], out["net_drop"]) == (2, 2, 2)


def _blk(**kw):
    return dict(dict(code_key=SPEC.r_shared_key), **kw)


R_DOC = dict(repro=_blk(passed=True), gate1=_blk(outcome="PASS"), gate3=_blk(outcome="PASS"))
CLEAN = dict(tracked=True, dirty=False)


def test_reuse():
    assert R.reuse(R_DOC, CLEAN, SPEC.r_shared_key, SPEC)["outcome"] == R.PASS
    out = R.reuse(R_DOC, CLEAN, "f" * 64, SPEC)
    assert out["outcome"] == R.STOP_REUSE and "공유 측정 키" in out["sentence"]
    assert R.reuse(R_DOC, dict(tracked=True, dirty=True), SPEC.r_shared_key, SPEC)["outcome"] == R.STOP_REUSE
    assert R.reuse(R_DOC, dict(tracked=False, dirty=False), SPEC.r_shared_key, SPEC)["outcome"] == R.STOP_REUSE
    for b, bad in (("repro", _blk(passed=False)), ("gate1", _blk(outcome="STOP_STRENGTH_LEVER")),
                   ("gate3", _blk(outcome="PASS", code_key="x")), ("gate1", None)):
        doc = dict(R_DOC, **{b: bad}) if bad is not None else {k: v for k, v in R_DOC.items() if k != b}
        out = R.reuse(doc, CLEAN, SPEC.r_shared_key, SPEC)
        assert out["outcome"] == R.STOP_REUSE and any(b in w for w in out["reasons"]), b


def test_set_outcome():
    assert R.set_outcome(dict(status="OK", n_b=21))["outcome"] == R.PASS
    out = R.set_outcome(dict(status=R.STOP_SET_SHORT, n_b=19))
    assert out["outcome"] == R.STOP_SET_SHORT
    assert out["sentence"] == "L 생성기 턴 104–209에서 E-grid 키 중복을 뺀 (b) 쌍이 21개에 못 미쳤다(19쌍)."


def _res(label="LEARNS_CONFIRMATORY", l1=1.8, l2=2.2):
    return dict(outcome="JUDGED", label=label, reasons=[], directions={"r1": {"ell": l1}, "r2": {"ell": l2}})


def _ratio(r1, r2):
    return {"r1": dict(ratio=r1), "r2": dict(ratio=r2)}


def test_gate2_order_and_ratio_boundary():
    assert R.gate2(_res(), _res(), _ratio(0.5, 0.5), SPEC)["outcome"] == R.PASS          # 0.5 passes
    out = R.gate2(_res(l1=0.89), _res(l1=1.8), _ratio(0.4999, 0.9), SPEC)
    assert out["outcome"] == R.STOP_PUNISH_WEAKENED and out["low"] == ["r1"]
    assert out["sentence"] == ("APL→MBON05 제거 아래 같은 시드의 처벌 학습량이 지렛대 없는 쪽의 절반에 못 미쳤다"
                               "(ℓ_r1 0.890 대 1.800, ℓ_r2 2.200 대 2.200).")
    assert R.gate2(_res(), _res(), _ratio(0.9, 0.49), SPEC)["low"] == ["r2"]
    out = R.gate2(_res("NO_LEARNING", 0.1, 0.2), _res(), _ratio(0.1, 0.1), SPEC)
    assert out["outcome"] == R.STOP_PUNISH_BROKEN                                        # L before C and ratio
    assert out["sentence"] == ("APL→MBON05 제거 아래에서 P의 처벌 학습 확인이 재현되지 않았다"
                               "(NO_LEARNING, ℓ_r1 0.100, ℓ_r2 0.200).")
    out = R.gate2(_res(), _res("DIRECTION_DEPENDENT"), _ratio(0.1, 0.1), SPEC)
    assert out["outcome"] == R.STOP_P_REFERENCE                                          # C before ratio
    assert out["sentence"] == ("지렛대 없는 P가 새 시드 블록에서 처벌 학습 확인을 재현하지 못했다(DIRECTION_DEPENDENT) — "
                               "비교 기준이 없다.")
    inv = dict(outcome="INVALID", label="INVALID", reasons=["broken"])
    assert R.gate2(inv, _res(), None, SPEC)["outcome"] == R.INVALID                      # INVALID first
    assert R.gate2(_res("NO_LEARNING"), inv, None, SPEC)["reasons"] == ["C: broken"]


F = dict(n=14, c=8, f_a=3, naive_a=4, T=177, k_even=16, pb_L=15, pb_C=18, pa_L=20, pa_C=22, d_b=3, d_a=2,
         rho1="0.700", rho2="0.900", why="x", m=19, label="NO_LEARNING", l1="0.100", l2="0.200", l1L="1", l1C="2",
         l2L="3", l2C="4")


def test_sentences_verbatim():
    for o in (R.STOP_REUSE, R.STOP_SET_SHORT, R.STOP_PUNISH_BROKEN, R.STOP_P_REFERENCE, R.STOP_PUNISH_WEAKENED,
              R.B_TB, R.B_PG, R.B_FA, R.SELECTED):
        assert "{" not in R.sentence(o, F), o
    assert R.sentence(R.B_TB, F) == ("APL→MBON05 제거가 마지막 판정 세트(L 생성기 턴 104–177)에서 14/21로 지렛대 없는 같은 "
                                     "세트 8/21보다 오르지 않았다. → 이 지렛대를 닫는다.")
    assert R.sentence(R.B_NC, dict(F, reason=R.C_ABOVE_BAR)) == (
        "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다. (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_NC, dict(F, reason=R.BELOW_BAR)) == (
        "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다. (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_NC, dict(F, reason=R.MARGIN)) == (
        "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다 (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_PG, F) == ("M2 (b) 기준과 여유는 넘었지만 지렛대 아래 처벌 통과가 순감소했다(처벌 통과 (b) 15 대 18, "
                                     "(a) 20 대 22; 순감소 (b) 3, (a) 2).")
    assert R.sentence(R.B_FA, F) == ("M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다(14/21 대 8/21, F_a 3/43, "
                                     "naive_a 4). 다음 병목은 F_a(순진 균형 (a) 쌍)다.")
    assert R.sentence(R.SELECTED, F) == (
        "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 마지막 판정 세트(L 생성기 턴 104–177, "
        "E-grid 키 중복 제외) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다"
        "(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 0.700·ℓ_r2 0.900 ≥ "
        "0.5(약화 정도 기록), 짝수 16/21, 판정 시드 24_300_xxx). 지렛대는 Q 결과를 보고 골랐고, 이 판정은 R 판정 결과를 본 뒤 "
        "가드를 바꾼 두 번째 판정이다(S.0). 작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 오선택률이 아니다. "
        "실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다.")
    with pytest.raises(KeyError):
        R.sentence(R.B_FA, {})


def test_g_axis_matches_brute_force_on_a_small_axis():
    a_pf, a_fp, tot = 0.2, 0.1, 0.0
    for states in itertools.product(("pf", "fp", "same"), repeat=5):
        w = 1.0
        for s in states:
            w *= {"pf": a_pf, "fp": a_fp, "same": 1 - a_pf - a_fp}[s]
        if states.count("pf") - states.count("fp") >= 3:
            tot += w
    assert R.g_axis(5, a_pf, a_fp, SPEC) == pytest.approx(tot)


DECLARED_S = {("null", 0.02, 0.02): 0.034, ("null", 0.05, 0.05): 0.145, ("null", 0.1, 0.1): 0.281,
              ("null", 0.15, 0.15): 0.362, ("harm", 0.1, 0.02): 0.752, ("harm", 0.15, 0.02): 0.949,
              ("harm", 0.2, 0.05): 0.961, ("harm", 0.1, 0.05): 0.549}
DECLARED_R = {("null", 0.02, 0.02): 0.163, ("null", 0.05, 0.05): 0.489, ("null", 0.1, 0.1): 0.888,
              ("null", 0.15, 0.15): 0.988, ("harm", 0.1, 0.02): 0.938, ("harm", 0.15, 0.02): 0.994,
              ("harm", 0.2, 0.05): 0.999, ("harm", 0.1, 0.05): 0.906}


def test_oc_reproduces_s6s_declared_values_and_rs_guard_row():
    oc = R.oc(SPEC)
    got = {(g["kind"], g["a_pf"], g["a_fp"]): g for g in oc["g_fail"]}
    assert set(got) == set(DECLARED_S)
    for k, v in DECLARED_S.items():
        assert round(got[k]["p"], 3) == v, k
        assert round(got[k]["r_guard"], 3) == DECLARED_R[k], k
    assert round(R.g_prob(0.1, 0.1, dataclasses.replace(SPEC, g_fail_drop=2)), 3) == 0.461    # S.9.1: ≥ 2 would be


def test_oc_rows_are_distributions():
    oc = R.oc(SPEC)
    n_na = min(SPEC.n_a, SPEC.oc_naive_max) + 1
    assert len(oc["rows"]) == len(SPEC.oc_q) * len(SPEC.oc_c) * n_na * (1 + len(oc["g_fail"]))
    for row in oc["rows"][::41]:
        assert sum(row["P"].values()) == pytest.approx(1.0) and set(row["P"]) == set(R.BANDS)
    assert all(r["P"][R.B_PG] == 0 for r in oc["rows"] if r["g_kind"] == "none")
    assert len(oc["notes"]) == 2 and "pun_C(b) − n ≥ 3" in oc["notes"][0] and "오선택률" in oc["notes"][1]
    assert len(oc["sha256"]) == 64 and oc["sha256"] == R.oc(SPEC)["sha256"]


def test_s_rules_literals_are_only_0_and_1():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "flymon/brain/s_rules.py").read_text()
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1}, nums
```

```python
# tests/brain/test_s_records.py
"""S.3 ④ / S.9.7: the gate-② ratio is ℓ_L / ℓ_C per direction on the same seeds (ℓ = p_rules' ℓ), with a paired seed
bootstrap CI; unequal seeds refuse; gate ②'s OC from P's block gives P(STOP_PUNISH_WEAKENED) = 1 − 0.5² at a true
ratio of exactly 0.5 and falls as the true ratio grows."""
import numpy as np
import pytest

from flymon.brain import s_records as SR
from flymon.brain.p_rules import p_judge
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.s_spec import SPEC
from tests.brain.s_fixtures import C1, Z, p_rows


def test_fixture_rows_pass_p_judge():
    res = p_judge(p_rows(SPEC.p_c, "none", 16), Z, C1, SPEC.p_c)
    assert res["label"] == "LEARNS_CONFIRMATORY", res.get("reasons")
    res = p_judge(p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2), Z, C1, SPEC.p)
    assert res["label"] == "LEARNS_CONFIRMATORY", res.get("reasons")


def test_ratio_is_ell_l_over_ell_c_on_the_same_seeds():
    rl = p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2)
    rc = p_rows(SPEC.p_c, "none", 16)
    out = SR.ratio(rl, rc, Z, SPEC)
    k = np.array([s % 3 for s in SPEC.p.seeds], float)
    want = float(np.mean(8 + k) / np.mean(16 + k))
    jl, jc = p_judge(rl, Z, C1, SPEC.p), p_judge(rc, Z, C1, SPEC.p_c)
    for d in SPEC.p.directions:
        assert out[d]["ratio"] == pytest.approx(want)
        assert out[d]["ell_L"] == pytest.approx(jl["directions"][d]["ell"])
        assert out[d]["ell_C"] == pytest.approx(jc["directions"][d]["ell"])
        lo, hi = out[d]["ratio_ci"]
        assert lo <= out[d]["ratio"] <= hi and out[d]["n"] == 32 and out[d]["seeds"] == list(SPEC.p.seeds)


def test_ratio_refuses_unequal_seeds():
    rl = p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2)
    rc = [dict(r, seed=r["seed"] + 1) if r["seed"] == SPEC.p.seeds[0] else r for r in p_rows(SPEC.p_c, "none", 16)]
    rc = [r for r in rc if not (r["seed"] == SPEC.p.seeds[1])]
    with pytest.raises(ValueError):
        SR.ratio(rl, rc, Z, SPEC)


def test_gate2_oc_from_ps_block():
    oc = SR.gate2_oc(p_rows(P_SPEC, "none", 16), Z, P_SPEC, SPEC)
    assert oc["source"]["seeds"] == list(P_SPEC.seeds) and oc["source"]["n_rows"] == 192
    assert [r["rho"] for r in oc["rows"]] == list(SPEC.gate2_oc_rhos)
    at = {r["rho"]: r["p_stop_weakened"] for r in oc["rows"]}
    assert at[0.5] == pytest.approx(0.75)                       # Φ(0) in each of two directions
    ps = [at[r] for r in SPEC.gate2_oc_rhos]
    assert ps == sorted(ps, reverse=True) and at[0.4] > 0.75 and at[1.0] < 0.05
    for d, s in oc["directions"].items():
        assert s["n"] == 32 and s["se"] > 0 and s["ell"] == pytest.approx((16 + 1) / 9, rel=0.05)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_s_rules.py tests/brain/test_s_records.py -q`
Expected: collection errors `ImportError: cannot import name 's_rules'` / `'s_records'`.

- [ ] **Step 3: Write `s_rules`**

```python
"""S's decisions (S.2-S.4, S.7, S.9). The judgement code is the authoritative source: the bands are r_rules.read_band
itself (R.3's order and R.9.4, S.4 "R.3 구간 순서 그대로") read with S's numbers (n_a 43); only G_fail is S's (S.4,
S.9.1: net drop pun_C − pun_L ≥ g_fail_drop on either axis, no pass->fail branch). Every number is a field of the
LastSetSpec passed in. SENTENCES are S.7 as amended by S.9.1 / S.9.7, 〈…〉 replaced by {field}, fixed before any
measurement (plan Reading 8). The operating characteristic is exact and never changes a band (S.6)."""
from __future__ import annotations

import dataclasses
import hashlib
from math import comb

from ..agent import e_rules
from . import r_rules
from .h3_store import canonical
from .n_rules import INVALID as P_INVALID
from .p_rules import LEARNS_CONFIRMATORY
from .r_spec import SPEC as R_SPEC

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (r_rules.PASS, r_rules.INVALID, r_rules.NOT_READ, r_rules.SEALED,
                                                       r_rules.READ, r_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = "STOP_REUSE", e_rules.STOP_SET_SHORT
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (r_rules.STOP_PUNISH_BROKEN, "STOP_P_REFERENCE",
                                                              "STOP_PUNISH_WEAKENED")
SELECTED, B_TB, B_FA, B_NC, B_PG = r_rules.SELECTED, r_rules.B_TB, r_rules.B_FA, r_rules.B_NC, r_rules.B_PG
BANDS = r_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = r_rules.C_ABOVE_BAR, r_rules.BELOW_BAR, r_rules.MARGIN
read_band = r_rules.read_band                      # the same function: S.4 keeps R.3's order (Reading 7)


# ================================================================ gates (S.2, S.3 ①, S.3 ④, S.9.5)
def reuse(r_doc: dict, r_git: dict, shared_key: str, spec) -> dict:
    """S.3 ① / S.9.5: R's gates are reused iff the shared measurement key equals R's, R's summary is tracked and clean,
    and its reused blocks (repro, gate ①, gate ③) exist with R's key and passed. Else STOP_REUSE (Reading 4: the
    re-measurement path is not built; S stops for the user)."""
    why = []
    if shared_key != spec.r_shared_key:
        why.append(f"공유 측정 키 {shared_key} ≠ R {spec.r_shared_key}")
    if not r_git.get("tracked") or r_git.get("dirty"):
        why.append(f"{spec.r_summary} 미커밋")
    for b in spec.r_reused:
        blk = r_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"R 블록 {b} 없음")
            continue
        if blk.get("code_key") != spec.r_shared_key:
            why.append(f"R 블록 {b}의 코드 키 {blk.get('code_key')}")
        ok = blk.get("passed") if b == "repro" else blk.get("outcome") == PASS
        if not ok:
            why.append(f"R 블록 {b} 통과 아님")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


def set_outcome(js: dict) -> dict:
    """S.2: STOP_SET_SHORT when (b) ran out before the last turn (the declared-value check is the runner's refusal)."""
    if js["status"] == STOP_SET_SHORT:
        return dict(outcome=STOP_SET_SHORT, sentence=sentence(STOP_SET_SHORT, dict(m=js["n_b"])))
    return dict(outcome=PASS)


def gate2(res_l: dict, res_c: dict, ratio: dict | None, spec) -> dict:
    """S.3 ④ in its order: an INVALID P judgement (either condition) stays INVALID (one rerun, S.3 ④ / R.5); L not
    LEARNS_CONFIRMATORY -> STOP_PUNISH_BROKEN; C not -> STOP_P_REFERENCE; ℓ_L / ℓ_C < p_ratio_min in either direction
    -> STOP_PUNISH_WEAKENED; else PASS."""
    bad = [f"{n}: {m}" for n, res in (("L", res_l), ("C", res_c)) if res.get("outcome") == P_INVALID
           for m in (res.get("reasons") or ["INVALID"])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, label_L=res_l.get("label"), label_C=res_c.get("label"))
    names = list(spec.p.directions)
    dl, dc = res_l["directions"], res_c["directions"]
    base = dict(reasons=[], label_L=res_l["label"], label_C=res_c["label"])
    if res_l["label"] != LEARNS_CONFIRMATORY:
        return dict(base, outcome=STOP_PUNISH_BROKEN, sentence=sentence(STOP_PUNISH_BROKEN, dict(
            label=res_l["label"], l1=f"{dl[names[0]]['ell']:.3f}", l2=f"{dl[names[-1]]['ell']:.3f}")))
    if res_c["label"] != LEARNS_CONFIRMATORY:
        return dict(base, outcome=STOP_P_REFERENCE, sentence=sentence(STOP_P_REFERENCE, dict(label=res_c["label"])))
    low = [d for d in names if ratio[d]["ratio"] < spec.p_ratio_min]
    fields = dict(l1L=f"{dl[names[0]]['ell']:.3f}", l1C=f"{dc[names[0]]['ell']:.3f}",
                  l2L=f"{dl[names[-1]]['ell']:.3f}", l2C=f"{dc[names[-1]]['ell']:.3f}")
    if low:
        return dict(base, outcome=STOP_PUNISH_WEAKENED, low=low, sentence=sentence(STOP_PUNISH_WEAKENED, fields))
    return dict(base, outcome=PASS)


# ================================================================ the punishment guard (S.4, S.9.1)
def g_fail_s(lp: list, cp: list, spec) -> dict:
    """Per axis over the pairs in both: pun = #(−p ≥ 2); net_drop = pun_C − pun_L; fail ⇔ net_drop ≥ g_fail_drop.
    pass->fail / fail->pass are records (S.4). G_fail_S ⇔ fail on (b) or (a)."""
    lm, cm = {p["key"]: p for p in lp}, {p["key"]: p for p in cp}
    axes = {}
    for ax in ("b", "a"):
        ks = sorted(k for k, p in cm.items() if p["axis"] == ax and k in lm)
        pun_c = sum(bool(cm[k]["punish_pass"]) for k in ks)
        pun_l = sum(bool(lm[k]["punish_pass"]) for k in ks)
        pf = sum(bool(cm[k]["punish_pass"]) and not lm[k]["punish_pass"] for k in ks)
        fp = sum(bool(lm[k]["punish_pass"]) and not cm[k]["punish_pass"] for k in ks)
        axes[ax] = dict(n=len(ks), pun_L=pun_l, pun_C=pun_c, net_drop=pun_c - pun_l, pass_to_fail=pf,
                        fail_to_pass=fp, fail=bool(pun_c - pun_l >= spec.g_fail_drop))
    return dict(g_fail=any(a["fail"] for a in axes.values()), axes=axes)


# ================================================================ the closing sentences (S.7, S.9.1, S.9.7; Reading 8)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43)"
SENTENCES = {
    STOP_REUSE: "R 관문 재사용 조건(S.9.5)이 깨졌다({why}). S는 R 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 쓰지 않고 "
                "멈춘다 — 사용자 몫.",
    STOP_SET_SHORT: "L 생성기 턴 104–209에서 E-grid 키 중복을 뺀 (b) 쌍이 21개에 못 미쳤다({m}쌍).",
    STOP_PUNISH_BROKEN: r_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: "지렛대 없는 P가 새 시드 블록에서 처벌 학습 확인을 재현하지 못했다({label}) — 비교 기준이 없다.",
    STOP_PUNISH_WEAKENED: "APL→MBON05 제거 아래 같은 시드의 처벌 학습량이 지렛대 없는 쪽의 절반에 못 미쳤다"
                          "(ℓ_r1 {l1L} 대 {l1C}, ℓ_r2 {l2L} 대 {l2C}).",
    B_TB: "APL→MBON05 제거가 마지막 판정 세트(L 생성기 턴 104–{T})에서 {n}/21로 지렛대 없는 같은 세트 {c}/21보다 오르지 "
          "않았다. → 이 지렛대를 닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: "M2 (b) 기준과 여유는 넘었지만 지렛대 아래 처벌 통과가 순감소했다(처벌 통과 (b) {pb_L} 대 {pb_C}, (a) {pa_L} 대 "
          "{pa_C}; 순감소 (b) {d_b}, (a) {d_a}).",
    B_FA: "M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다({n}/21 대 {c}/21, F_a {f_a}/43, naive_a {naive_a})."
          " 다음 병목은 F_a(순진 균형 (a) 쌍)다.",
    SELECTED: "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 마지막 판정 세트(L 생성기 턴 104–{T}, "
              "E-grid 키 중복 제외) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다"
              "({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 {rho1}·ℓ_r2 "
              "{rho2} ≥ 0.5(약화 정도 기록), 짝수 {k_even}/21, 판정 시드 24_300_xxx). 지렛대는 Q 결과를 보고 골랐고, 이 판정은 "
              "R 판정 결과를 본 뒤 가드를 바꾼 두 번째 판정이다(S.0). 작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 "
              "오선택률이 아니다. 실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristic (S.6, S.9.1, S.9.7)
def g_axis(N: int, a_pf: float, a_fp: float, spec) -> float:
    """P(k_pf − k_fp ≥ g_fail_drop) on an axis of N independent pairs (multinomial: C pass ∧ L fail with a_pf, C fail ∧
    L pass with a_fp)."""
    rest = 1 - a_pf - a_fp
    tot = 0.0
    for k in range(N + 1):
        for j in range(N - k + 1):
            if k - j >= spec.g_fail_drop:
                tot += comb(N, k) * comb(N - k, j) * a_pf ** k * a_fp ** j * rest ** (N - k - j)
    return tot


def g_prob(a_pf: float, a_fp: float, spec) -> float:
    """G_fail_S on (b) n_b or (a) n_a pairs, the axes independent (OR)."""
    return 1 - (1 - g_axis(spec.n_b, a_pf, a_fp, spec)) * (1 - g_axis(spec.n_a, a_pf, a_fp, spec))


OC_NOTES = ("S.9.7: 지렛대 아래 보상 (b)가 포화하면(R 21/21) testable_b = pun_L(b)이므로 (b) 가드는 pun_C(b) − n ≥ 3으로 n과 "
            "결합한다. 'G_fail은 n과 독립' 가정이 (b) 축에서 깨지고, 실질 처벌 가드는 (a) 축이 맡는다.",
            "작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 오선택률이 아니다.")


def oc(spec) -> dict:
    """Exact conditional OC with R's model (r_rules._cell: n ~ Bin(n_b, q), F_a ~ Bin(naive_a, q), G_fail ~
    Bernoulli(g), c fixed, each cell read through read_band) on S's n_b 21 · n_a 43, g from G_fail_S; beside each g row
    R's guard on the same pair counts (r_guard: drop ≥ 2 ∨ pass->fail ≥ 3, S.6)."""
    rg = dataclasses.replace(R_SPEC, n_b=spec.n_b, n_a=spec.n_a)

    def grow(kind, x, y):
        return dict(kind=kind, a_pf=x, a_fp=y, p=g_prob(x, y, spec), r_guard=r_rules.g_prob(x, y, rg))

    g_rows = ([grow("null", a, a) for a in spec.oc_discord] + [grow("harm", x, y) for x, y in spec.oc_harm])
    rows = []
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in range(min(spec.n_a, spec.oc_naive_max) + 1):
                for gr in [dict(kind="none", a_pf=0.0, a_fp=0.0, p=0.0)] + g_rows:
                    rows.append(dict(q=q, c=c, naive_a=na, g_kind=gr["kind"], a_pf=gr["a_pf"], a_fp=gr["a_fp"],
                                     g=gr["p"], P=r_rules._cell(q, c, na, gr["p"], spec)))
    table = dict(assumption="pairs independent; q_b = q_a; c fixed; G_fail_S independent of n and F_a with probability "
                            "g from a per-pair discordance model per axis (net drop ≥ g_fail_drop), OR over (b) and (a)",
                 notes=list(OC_NOTES), g_fail=g_rows, rows=rows)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())
```

- [ ] **Step 4: Write `s_records`**

```python
"""S's gate-② numbers (S.3 ④, S.9.7) that p_rules does not return — records and the ratio the gate reads; no engine.
- vectors: p_rules.group + p_rules.arm_vectors per direction (P.6.2's per-seed dl, ds, dt over arms ① and ③).
- ratio: per direction ℓ_L / ℓ_C on the same seeds (ℓ = mean dl = p_rules' ℓ), with a paired seed bootstrap (one
  o_rules.boot_weights matrix for both conditions, P's draws and seed; the CI at P's level) — the gate reads the
  point estimate (S.3 ④), the CI is a record.
- gate2_oc (S.9.7, recorded before gate ②, never changes it): from P's committed block (no lever, seeds 23_000_xxx),
  per direction ℓ_P and its bootstrap SE; for a true ratio ρ, ℓ_C ≈ ℓ_P and ℓ_L = ρ ℓ_P with both SEs equal to P's
  and independent, so SE(ρ̂) ≈ SE_P √(1 + ρ²) / ℓ_P; P(STOP_PUNISH_WEAKENED) = 1 − Π_d Φ((ρ − p_ratio_min) / SE_d)
  (normal approximation, directions independent; pairing by seed would shrink the SE — the approximation is
  conservative toward STOP)."""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from .n_rules import _ci
from .o_rules import boot_weights
from .p_rules import arm_vectors, group


def vectors(rows: list, z: dict, pspec) -> dict:
    by = group(rows, pspec)
    return {d: arm_vectors(by[d], z, pspec) for d in pspec.directions}


def ratio(rows_l: list, rows_c: list, z: dict, spec) -> dict:
    """{direction: ell_L, ell_C, ratio, ratio_ci, n}; ValueError when L and C do not hold the same seeds."""
    vl, vc = vectors(rows_l, z, spec.p), vectors(rows_c, z, spec.p_c)
    o = spec.p.o
    out = {}
    for d in spec.p.directions:
        if vl[d]["seeds"] != vc[d]["seeds"]:
            raise ValueError(f"direction {d}: L and C must hold the same seeds in the same order")
        dl_l, dl_c = np.asarray(vl[d]["dl"], float), np.asarray(vc[d]["dl"], float)
        W = boot_weights(dl_l.size, o.boot_draws, o.boot_seed)
        bl, bc = W @ dl_l, W @ dl_c
        out[d] = dict(ell_L=float(dl_l.mean()), ell_C=float(dl_c.mean()), ratio=float(dl_l.mean() / dl_c.mean()),
                      ratio_ci=_ci(bl / bc, o.ci_level), n=int(dl_l.size), seeds=list(vl[d]["seeds"]))
    return out


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def gate2_oc(p_rows: list, z: dict, pspec, spec) -> dict:
    """S.9.7: STOP_PUNISH_WEAKENED's probability at each true ratio in spec.gate2_oc_rhos from P's block (module
    docstring for the model)."""
    v = vectors(p_rows, z, pspec)
    o = pspec.o
    dirs = {}
    for d in pspec.directions:
        dl = np.asarray(v[d]["dl"], float)
        W = boot_weights(dl.size, o.boot_draws, o.boot_seed)
        dirs[d] = dict(ell=float(dl.mean()), se=float(np.std(W @ dl)), n=int(dl.size))
    rows = []
    for rho in spec.gate2_oc_rhos:
        se = {d: s["se"] * sqrt(1.0 + rho * rho) / s["ell"] for d, s in dirs.items()}
        keep = float(np.prod([_phi((rho - spec.p_ratio_min) / se[d]) for d in dirs]))
        rows.append(dict(rho=float(rho), se_ratio=se, p_stop_weakened=1.0 - keep))
    return dict(model="SE(ρ̂) ≈ SE_P √(1 + ρ²) / ℓ_P per direction (ℓ_C ≈ ℓ_P, ℓ_L = ρ ℓ_P, equal independent SEs); "
                      "P(STOP) = 1 − Π_d Φ((ρ − p_ratio_min) / SE_d); normal approximation, directions independent",
                source=dict(seeds=[int(s) for s in pspec.seeds], n_rows=len(p_rows)), directions=dirs, rows=rows)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_s_rules.py tests/brain/test_s_records.py tests/brain/test_r_rules.py -q`
Expected: all pass; the OC test reproduces S.6's eight G_fail_S values and R's eight guard values to three decimals.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/s_rules.py flymon/brain/s_records.py tests/brain/s_fixtures.py tests/brain/test_s_rules.py tests/brain/test_s_records.py
git commit -m "feat(s): s_rules / s_records — reuse, set, gate 2 order and ratio, G_fail_S net drop >= 3, R's bands, S.7 sentences, OCs"
```

---

### Task 5: `s_runner` (1) — context, chain and refusals, reuse, set, smoke, OC, gate ②'s OC, gate ②

**Files:**
- Create: `flymon/brain/s_runner.py` (first part — Task 6 appends the judgement methods to class `Runner`)
- Test: `tests/brain/s_world.py`, `tests/brain/test_s_runner.py`

**Interfaces:**
- Consumes: Tasks 1–4; `r_records` (`cond_summary`, `raw_check`, `JUDGE_BLOCK`), `r_runner.R_HASHED_FILES` / `p_items`, `r_pairs.row_key` / `even_rows` / `p_reference`, `p_rules.p_judge`, `e_runner.summary_git`, `h3_store` (`ROOT`, `canonical`, `sha256_file`, `git_state`), `r_measure.R_MEASURE_FILES`; tests reuse `tests/brain/r_fixtures.py`.
- Produces: `ORDER`, `GATES`, `GATE2_INVALID`, `EXIT_REFUSE = 2`, `EXIT_REUSE = 7`, `S_PIPELINE_FILES`, `S_HASHED_FILES`, `DECISION_FILES`, `JUDGE_MARKER`, `REREAD_MARKER`, `DONE_MARKER`, `git_state()`, `pipeline_key()`, `decision_key()`, `refuse(msg, code)`, `build_ctx(spec, npz) -> dict` (keys `params, readout, z, types, n_kc, enc, even_rows, s_set, judgement_rows, p_ref, p_inputs, r_doc, r_git`), `Runner(measurer, ctx, spec, summary_path=None, code=None, pipeline=None, archive_root=None)` with `stage_reuse`, `stage_set`, `stage_smoke`, `stage_oc`, `stage_gate2_oc`, `stage_gate2(rerun=False)`; block fields read later: `reuse.records.{k_even, repro_csc_sha256_none, r_gate2_ell}`, `smoke.problems`, `oc.sha256`, `gate2.{label_L, label_C, ratio}`.
- `tests/brain/s_world.py`: `CODE`, `PIPE`, `r_doc()`, `declared_set(**kw)`, `Scripted`, `World`, `doc()`, `through_gate2(w, m)`.

- [ ] **Step 1: Write the world and the failing test**

```python
"""A scripted S world for the runner tests (no engine, no pool): a fake ctx (R's summary as the reuse source, S's set
as a declared-value dict, P's block as fixture rows), a Scripted measurer whose P arms come from s_fixtures (real
p_rules.p_judge reads them: ℓ ≈ (drop + 1) / 9) and whose oracle writes real cache-like files under results/s/cache
(the seal and the judge re-read them), and helpers to run the chain."""
import hashlib
import json
from pathlib import Path

from flymon.brain import s_runner as SR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.s_spec import SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli

SHA = {"L": "sha-L", "C": "sha-C", "E0": "sha-C"}
CODE = {"key": SPEC.r_shared_key}
PIPE = {"key": "p" * 64}


def r_doc() -> dict:
    """R's summary as S reads it (S.3 ①): the three reused blocks plus what S records from R."""
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L"}}),
                gate1=dict(outcome="PASS", code_key=k, record={"median": 0.0457}, written_at="r-g1"),
                gate2=dict(outcome="PASS", code_key=k, label="LEARNS_CONFIRMATORY",
                           p_judgement={"directions": {"r1": {"ell": 1.087}, "r2": {"ell": 1.792}}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def declared_set(**kw) -> dict:
    js = dict(status="OK", n_b=SPEC.n_b, n_a=SPEC.n_a, last_turn=SPEC.last_turn, skipped={"used": 1},
              digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, b=[], a=[])
    return dict(js, **kw)


class Scripted:
    """RMeasurer's interface. plan[(block, cond)][pair key] = (r_ok, p_ok, bal) (default: punish passes only);
    override[(block, cond)] = dict(edges=, edit=, sha=) changes one block's condition only; drop = {edit: P drop}
    sets ℓ per condition; arm_edges = {edit: edges}; fail_once makes the next oracle call raise like an
    interrupted measurement."""

    def __init__(self, plan=None, override=None, drop=None, arm_edges=None):
        self.plan, self.override = plan or {}, override or {}
        self.drop = dict({SPEC.lever_edit: 12, SPEC.no_edit: 16}, **(drop or {}))
        self.arm_edges = dict({SPEC.lever_edit: SPEC.lever_edges, SPEC.no_edit: 0}, **(arm_edges or {}))
        self.fail_once, self.calls, self.last_wall_s, self.last_jobs = False, [], 2.0, 1
        self._rm = RMeasurer(None, None, SPEC, Params(), READOUT, Z, TYPES, 100)

    def inputs(self, row, cond, block, seeds):
        return self._rm.inputs(row, cond, block, seeds)

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self.drop[i["edit"]], "sha-L" if i["edit"] == SPEC.lever_edit else "sha-C",
                        self.arm_edges[i["edit"]]) for i in items]

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
            f = Path(SPEC.cache_dir) / "r_oracle" / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.judge_commits, self.judgement_calls, self.set_calls = [], 0, 0
        self.r, self.r_git, self.js = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"]), declared_set()
        monkeypatch.setattr(SR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(SR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=104)
        p_ref = {(r["direction"], r["arm"], r["seed"]): {k: v for k, v in r.items()
                                                         if k not in ("direction", "x", "y", "point", "r")}
                 for r in p_rows(P_SPEC, "none", 16)}
        self.archive = tmp_path / "archive"
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, even_rows=self.even,
                        s_set=self._s_set, judgement_rows=self._judgement_rows,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git))

    def _s_set(self):
        self.set_calls += 1
        return dict(self.js)

    def _judgement_rows(self):
        self.judgement_calls += 1
        return self.judge

    def runner(self, m, code=None, pipeline=None):
        return SR.Runner(m, self.ctx, SPEC, code=code or CODE, pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def pass_plan(self, n=14, c=8, f_a=3):
        """judgement: L n / C c (b) testable and L f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", "L"): {**{k: (True, True, False) for k in jb[:n]}, **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "C"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through_gate2(w, m, r=None):
    r = r or w.runner(m)
    assert r.stage_reuse()["outcome"] == "PASS"
    assert r.stage_set()["outcome"] == "PASS"
    assert r.stage_smoke()["problems"] == []
    r.stage_oc()
    r.stage_gate2_oc()
    assert r.stage_gate2()["outcome"] == "PASS"
    return r
```

```python
# tests/brain/test_s_runner.py
"""The S stage chain up to gate ② (S.2, S.3 ①-④, S.9.3, S.9.5, S.9.7; Readings 4-6, 9): reuse -> set -> smoke -> oc
-> gate2_oc -> gate2; each stage refuses (exit 2, nothing written) when an earlier block is missing, a later one
exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted or a hashed file is
dirty; a STOP_REUSE, STOP_SET_SHORT, smoke problem or gate ② STOP / INVALID blocks every later stage; R's reuse
condition broken after `reuse` refuses every later stage with exit 7; the judgement set is never measured before the
judgement stages."""
import json
from pathlib import Path

import pytest

from flymon.brain import s_rules
from flymon.brain import s_runner as SR
from flymon.brain.s_spec import SPEC
from tests.brain.s_world import Scripted, World, doc, through_gate2


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate2_never_touches_the_judgement_set(w):
    m = Scripted()
    through_gate2(w, m)
    d = doc()
    assert set(d) == set(SR.ORDER[:SR.ORDER.index("jm:L")])
    assert d["reuse"]["records"]["k_even"] == 16 and d["reuse"]["records"]["repro_csc_sha256_none"] == "sha-C"
    assert d["reuse"]["shared_key"] == SPEC.r_shared_key and d["reuse"]["code_key"] == SPEC.r_shared_key
    assert all(d[b]["pipeline_key"] == "p" * 64 for b in d)
    assert d["set"]["set"]["last_turn"] == 177 and w.set_calls == 1
    assert d["smoke"]["p"]["L"]["edit_edges"] == [2] and d["smoke"]["p"]["C"]["edit_edges"] == [0]
    assert d["smoke"]["p"]["L"]["label"] == d["smoke"]["p"]["C"]["label"] == "LEARNS_CONFIRMATORY"
    assert Path(SPEC.smoke_detail).exists()
    assert d["gate2_oc"]["rows"][1]["rho"] == 0.5 and d["gate2_oc"]["rows"][1]["p_stop_weakened"] == pytest.approx(0.75)
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"]) == ("LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY")
    assert g2["seeds"] == list(SPEC.p.seeds) and g2["edit_edges_L"] == [2] and g2["edit_edges_C"] == [0]
    assert 0.5 <= g2["ratio"]["r1"]["ratio"] < 1 and set(g2["ell_L_minus_r_gate2"]) == {"r1", "r2"}
    assert g2["ell_L_minus_r_gate2"]["r1"] == pytest.approx(g2["ratio"]["r1"]["ell_L"] - 1.087)
    assert w.judgement_calls == 0
    assert not any(c[1] == "judge" for c in m.calls if c[0] == "oracle")
    assert all(c[3] == 2 for c in m.calls if c[0] == "oracle" and c[1] == "smoke")
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]


def test_reuse_stop_when_the_shared_key_differs(w):
    out = w.runner(Scripted(), code={"key": "f" * 64}).stage_reuse()
    assert out["outcome"] == s_rules.STOP_REUSE and "공유 측정 키" in out["sentence"]
    assert doc()["reuse"]["outcome"] == s_rules.STOP_REUSE
    with pytest.raises(SystemExit) as e:
        w.runner(Scripted()).stage_set()
    assert e.value.code == 2 and w.set_calls == 0


@pytest.mark.parametrize("mutate", ["gate1", "dirty", "key"])
def test_reuse_broken_after_reuse_refuses_with_7(w, mutate):
    r = w.runner(Scripted())
    r.stage_reuse()
    if mutate == "gate1":
        w.r["gate1"]["outcome"] = "STOP_STRENGTH_LEVER"
    elif mutate == "dirty":
        w.r_git["dirty"] = True
    else:
        r = w.runner(Scripted(), code={"key": "f" * 64})
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == SR.EXIT_REUSE and "set" not in doc() and w.set_calls == 0


def test_set_stop_set_short_blocks_later_stages(w):
    w.js = dict(w.js, status="STOP_SET_SHORT", n_b=19, last_turn=209)
    r = w.runner(Scripted())
    r.stage_reuse()
    out = r.stage_set()
    assert out["outcome"] == s_rules.STOP_SET_SHORT and "(19쌍)" in out["sentence"]
    with pytest.raises(SystemExit) as e:
        r.stage_smoke()
    assert e.value.code == 2


def test_set_mismatch_refuses_without_a_block(w):
    w.js = dict(w.js, digest_keys="0" * 64)
    r = w.runner(Scripted())
    r.stage_reuse()
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == 2 and "set" not in doc()


def _to_gate2(w, m):
    r = w.runner(m)
    r.stage_reuse()
    r.stage_set()
    r.stage_smoke()
    r.stage_oc()
    r.stage_gate2_oc()
    return r


@pytest.mark.parametrize("drop,outcome", [
    ({SPEC.lever_edit: 6}, s_rules.STOP_PUNISH_WEAKENED),          # ℓ_L ≈ 0.78 ≥ c1, ratio ≈ 7/17 < 0.5
    ({SPEC.lever_edit: 3}, s_rules.STOP_PUNISH_BROKEN),            # ℓ_L ≈ 0.44 < c1
    ({SPEC.no_edit: 3}, s_rules.STOP_P_REFERENCE),                 # ℓ_C ≈ 0.44 < c1, L fine
])
def test_gate2_stops_block_the_judgement(w, drop, outcome):
    r = _to_gate2(w, Scripted(drop=drop))
    out = r.stage_gate2()
    assert out["outcome"] == outcome and out["sentence"]
    with pytest.raises(SystemExit) as e:
        r._require("jm:L")                                         # every judgement stage starts here
    assert e.value.code == 2 and w.judgement_calls == 0


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    m = Scripted()
    r = _to_gate2(w, m)
    m.arm_edges[SPEC.lever_edit] = 1                               # the lever changed 1 edge in gate ② only
    assert r.stage_gate2()["outcome"] == s_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                                  # same pipeline key: fix first
    m.arm_edges[SPEC.lever_edit] = SPEC.lever_edges
    fixed = w.runner(m, pipeline={"key": "q" * 64})
    assert fixed.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["rerun_of"] == d["gate2_invalid"]["written_at"]
    with pytest.raises(SystemExit):
        fixed.stage_gate2(rerun=True)                              # once


def test_gate2_invalid_when_the_p_judgement_is_invalid(w):
    m = Scripted()
    r = _to_gate2(w, m)
    orig = m.arms
    m.arms = lambda items, *a: [dict(x, csc_sha256="sha-other") if x["seed"] == SPEC.p.seeds[0] else x
                                for x in orig(items, *a)]
    out = r.stage_gate2()
    assert out["outcome"] == s_rules.INVALID and out["ratio"] is None


def test_smoke_problem_blocks_later_stages(w):
    m = Scripted(override={("smoke", "L"): dict(edges=1)})
    r = w.runner(m)
    r.stage_reuse()
    r.stage_set()
    assert r.stage_smoke()["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    r = w.runner(Scripted())
    with pytest.raises(SystemExit):
        r.stage_set()                                              # reuse missing
    r.stage_reuse()
    with pytest.raises(SystemExit):
        r.stage_reuse()                                            # own block
    monkeypatch.setattr(SR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_set()                                              # uncommitted summary
    monkeypatch.setattr(SR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(SR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/s_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_set()                                              # dirty hashed file
    assert set(doc()) == {"reuse"}


def test_gate2_oc_refuses_without_ps_block(w):
    r = w.runner(Scripted())
    r.stage_reuse()
    r.stage_set()
    r.stage_smoke()
    r.stage_oc()
    w.ctx["p_ref"] = lambda wanted: {}
    with pytest.raises(SystemExit):
        r.stage_gate2_oc()
    assert "gate2_oc" not in doc()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_s_runner.py -q`
Expected: collection error `ImportError: cannot import name 's_runner'`.

- [ ] **Step 3: Write the first part of `s_runner`**

```python
"""Spec S's stage chain (S.2-S.5, S.9; plan Readings 4-6, 9-12):
reuse -> set -> smoke -> oc -> gate2_oc -> gate2 -> jm:L -> jm:C -> jm:E0 -> seal -> judge
(and after judge only: recompute / invalid_run), one block each in results/summary/s_lever.json, written only through
s_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists,
its own block exists (gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or a hashed S file is
dirty. Every stage after `reuse` re-checks R's reuse condition (S.9.5: the shared measurement key equals R's and R's
reused blocks are intact) and refuses with SystemExit 7 when it broke — S has no re-measurement path (Reading 4). A
smoke with problems or a gate whose outcome is not PASS blocks every later stage, so the judgement set stays unused.
The judgement set is measured only through ctx["judgement_rows"] (s_pairs.judgement_rows: generated afresh and checked
against S.2's declared values on every call) from the judgement stages; `set` reads the list once (no measurement);
jm blocks carry no pair statistic. Blocks carry both keys: code_key = the shared measurement key (R_MEASURE_FILES,
which also keys the cache) and pipeline_key = S's own files (S.9.5)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent.e_runner import summary_git
from . import r_records, s_records, s_rules, s_store
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import R_HASHED_FILES, p_items
from .s_pairs import check_s_set
from .s_pairs import summary as set_summary
from .s_spec import smoke

ORDER = ("reuse", "set", "smoke", "oc", "gate2_oc", "gate2", "jm:L", "jm:C", "jm:E0", "seal", "judge")
GATES = ("reuse", "set", "gate2")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_REUSE = 2, 7
S_PIPELINE_FILES = ("flymon/brain/s_spec.py", "flymon/brain/s_pairs.py", "flymon/brain/s_store.py",
                    "flymon/brain/s_records.py", "flymon/brain/s_rules.py", "flymon/brain/s_runner.py",
                    "scripts/run_s.py")
S_HASHED_FILES = tuple(dict.fromkeys(R_HASHED_FILES + S_PIPELINE_FILES + ("results/summary/r_lever.json",)))
# The decision code: every hashed S file outside the shared measurement key. The seal pins its hash; the judge reads
# only under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in S_HASHED_FILES if f not in R_MEASURE_FILES)
# Written before the band is computed: the set was read once. REREAD: S.9.2's one re-generation after the mark.
# DONE: written right after the judge block, so a judge block written and then discarded never reopens the set.
JUDGE_MARKER = "results/s/judge_read.json"
REREAD_MARKER = "results/s/judge_reread.json"
DONE_MARKER = "results/s/judge_done.json"


def git_state() -> dict:
    return _h3_git_state(files=S_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(S_PIPELINE_FILES)


def decision_key() -> dict:
    return _files_key(DECISION_FILES)


def refuse(msg: str, code: int = EXIT_REFUSE):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _rounds(n: int, w: int) -> int:
    return -(-int(n) // max(1, int(w)))


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, z — block h4), the H.4 pools as types, the encoder summary, the even rows (smoke only),
    lazy callables for S's set (list only) and judgement rows (checked), P's committed entries (gate ②'s OC) and
    stimuli (c1 from block n1), and R's summary with its git state (the reuse condition)."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from . import n_cli, p_cli, q_pairs, r_pairs, s_pairs
    from .circuits import Populations
    from .connectome import Connectome
    cfg = load_c3_config(spec.m0d_summary)
    if (cfg.readout["A"], cfg.readout["P"]) != (spec.a_type, spec.p_type):
        refuse(f"C3's readout is {cfg.readout}, not A {spec.a_type} / P {spec.p_type}")
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    even = r_pairs.even_rows(pops, rc, enc, spec)
    p_doc = json.loads(Path(spec.p_summary).read_text())["p"]
    model_types = n_cli.model_types(npz)

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                even_rows=even, s_set=lambda: s_pairs.s_set(pops, enc, spec),
                judgement_rows=lambda: s_pairs.judgement_rows(pops, rc, enc, spec),
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, r_doc=lambda: json.loads(Path(spec.r_summary).read_text()),
                r_git=lambda: summary_git(spec.r_summary))


class Runner:
    def __init__(self, measurer, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 pipeline: dict | None = None, archive_root=None):
        self.m, self.ctx, self.spec = measurer, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return s_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed S files are dirty: {gs['dirty_hashed']}")

    def _reuse_now(self, stage: str) -> dict:
        """S.9.5: the reuse condition, re-checked at every stage after `reuse` (so before the judgement measurement,
        the seal and the judge too)."""
        r = s_rules.reuse(self.ctx["r_doc"](), self.ctx["r_git"](), self.code_key, self.spec)
        if r["outcome"] != s_rules.PASS:
            refuse(f"stage {stage}: R's reuse condition broke ({'; '.join(r['reasons'])}) — S has no re-measurement "
                   f"path; S stops here and the judgement set stays unused (S.9.5, plan Reading 4)", EXIT_REUSE)
        return r

    def _require(self, stage: str, allow_own: bool = False) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; S never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; S never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != s_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — S stops there and the "
                   f"judgement set stays unused (S.3)")
        if i > 0:
            self._reuse_now(stage)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, pipeline_key=self.pipeline_key, git=git_state(),
                    written_at=_now())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        s_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    # ---- reuse (S.3 ①, S.9.5) and the set (S.2) — no pool ------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        rd = self.ctx["r_doc"]()
        dec = s_rules.reuse(rd, self.ctx["r_git"](), self.code_key, sp)

        def g(*ks):
            return _dig(rd, ks)
        rec = dict(k_even=g("gate3", "testable_b"), c_even=g("gate3", "c_even"),
                   gate1_median=g("gate1", "record", "median"), repro_csc_sha256_none=g("repro", "csc_sha256_none"),
                   r_smoke_L_csc_sha256=g("smoke", "oracle", "L", "csc_sha256"),
                   r_gate2_ell={d: g("gate2", "p_judgement", "directions", d, "ell") for d in sp.p.directions},
                   r_gate2_label=g("gate2", "label"))
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, r_summary=sp.r_summary,
                    r_commits=dict(sp.r_commits), r_blocks={b: g(b, "written_at") for b in sp.r_reused}, records=rec)
        self._write("reuse", body)
        return body

    def stage_set(self) -> dict:
        """S.2, list only: STOP_SET_SHORT is recorded; any other mismatch with the declared T / n_a / digests refuses."""
        self._require("set")
        js = self.ctx["s_set"]()
        dec = s_rules.set_outcome(js)
        if dec["outcome"] == s_rules.PASS:
            bad = check_s_set(js, self.spec)
            if bad:
                refuse(f"the S set does not reproduce its declaration (S.2): {bad}")
        body = dict(dec, set=set_summary(js))
        self._write("set", body)
        return body

    # ---- smoke (S.3 ②) ---------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        c1, st = self.ctx["p_inputs"](sm.p, True)
        items = p_items(sm.p, st, sp.lever_edit) + p_items(sm.p_c, st, sp.no_edit)
        rows = self.m.arms(items, self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke", sm.p.o.n)
        p_wall = self.m.last_wall_s
        p = {}
        for n, pspec, edit in (("L", sm.p, sp.lever_edit), ("C", sm.p_c, sp.no_edit)):
            rs = [r for r in rows if r["edit"] == edit]
            res = p_judge(rs, self.ctx["z"], c1["c1"], pspec)
            p[n] = dict(outcome=res["outcome"], label=res["label"], reasons=list(res.get("reasons", [])),
                        edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rs}), n_rows=len(rs))
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            got = self.m.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self.ctx["z"], [row_key(r) for r in sel], sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  kc_median=s["kc_median"], saturation=s["saturation"], wall_s=self.m.last_wall_s,
                                  jobs=self.m.last_jobs)
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        s_store.write_json(sp.smoke_detail, detail, self.plist)
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
        body = dict(detail, problems=self._smoke_problems(p, orc, repro_sha), cost=self._cost(p_wall, orc, sm),
                    detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, repro_sha) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        bad = []
        if p["L"]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"P arms L: lever edges {p['L']['edit_edges']}")
        if p["C"]["edit_edges"] != [0]:
            bad.append(f"P arms C: edges {p['C']['edit_edges']}, declared none")
        for n in ("L", "C"):
            if p[n]["outcome"] == P_INVALID:
                bad.append(f"P arms {n}: INVALID {p[n]['reasons'][:2]}")
        if orc[nl]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"L: lever edges {orc[nl]['edit_edges']}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not R's repro {repro_sha}")
        for n in sp.cond_names:
            if orc[n]["reasons"]:
                bad.append(f"{n}: {orc[n]['reasons'][:2]}")
        return bad

    def _cost(self, p_wall, orc, sm) -> dict:
        """A rough estimate from smoke wall times: rounds of workers × the seed ratio (recorded, never a rule)."""
        sp = self.spec
        per = len(sp.p.directions) * len(sp.p.arms) * 2                      # L and C
        p_h = p_wall / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        ratio = sum(len(v) for v in sp.judge_seeds().values()) / sum(len(v) for v in sm.even_seeds().values())
        return dict(gate2_h=p_h, jm_per_condition_h=o_round * ratio * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (S.6, S.9.7) — before gate ② ---------------------------------------------
    def stage_oc(self) -> dict:
        self._require("oc")
        body = s_rules.oc(self.spec)
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7: STOP_PUNISH_WEAKENED's probability from P's committed block (192 entries, no lever), before gate ②."""
        self._require("gate2_oc")
        pairs, point = P_SPEC.pairs(), [float(v) for v in P_SPEC.o.o2_point]
        wanted = {(d, a, int(s)) for d in P_SPEC.directions for a in P_SPEC.arms for s in P_SPEC.seeds}
        ref = self.ctx["p_ref"](wanted)
        if set(ref) != wanted:
            refuse(f"P's committed cache {self.spec.p_cache_dir} lacks {sorted(wanted - set(ref))[:3]}")
        rows = [dict(ref[k], direction=k[0], x=pairs[k[0]][0], y=pairs[k[0]][1], point=point) for k in sorted(wanted)]
        body = s_records.gate2_oc(rows, self.ctx["z"], P_SPEC, self.spec)
        self._write("gate2_oc", body)
        return body

    # ---- gate ② (S.3 ④, S.9.3) ----------------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """rerun (S.3 ④, R.5): only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID
        run's (a fix in a shared measurement file changes the shared key and refuses through _reuse_now: S.9.5). The
        INVALID block and the new one are written together after the measurement."""
        prior = None
        if rerun:
            doc = self._doc()
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (S.3 ④, R.5)")
            if not blk or blk.get("outcome") != s_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (S.3 ④)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (S.3 ④)")
            self._require("gate2", allow_own=True)
            prior = blk
        else:
            doc = self._require("gate2")
        sp = self.spec
        c1, st = self.ctx["p_inputs"](sp.p, False)
        items = p_items(sp.p, st, sp.lever_edit) + p_items(sp.p_c, st, sp.no_edit)
        rows = self.m.arms(items, self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2", sp.p.o.n)
        rows_l = [r for r in rows if r["edit"] == sp.lever_edit]
        rows_c = [r for r in rows if r["edit"] == sp.no_edit]
        res_l = p_judge(rows_l, self.ctx["z"], c1["c1"], sp.p)
        res_c = p_judge(rows_c, self.ctx["z"], c1["c1"], sp.p_c)
        ratio, why = None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = s_records.ratio(rows_l, rows_c, self.ctx["z"], sp)
            except ValueError as e:
                why.append(str(e))
        dec = s_rules.gate2(res_l, res_c, ratio, sp) if not why else dict(outcome=s_rules.INVALID, reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [sp.lever_edges] or edges_c != [0]) and dec["outcome"] != s_rules.INVALID:
            dec = dict(outcome=s_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{sp.lever_edges} / 0"])
        r_ell = (self._doc().get("reuse") or {}).get("records", {}).get("r_gate2_ell", {})
        diff = ({d: ratio[d]["ell_L"] - r_ell[d] for d in ratio if r_ell.get(d) is not None} if ratio else None)
        body = dict(dec, ratio=ratio, ell_L_minus_r_gate2=diff, p_judgement_L=res_l, p_judgement_C=res_c,
                    c1_source=c1, edit_edges_L=edges_l, edit_edges_C=edges_c, seeds=list(sp.p.seeds),
                    rerun_of=(prior or {}).get("written_at"), wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        if prior is None:
            return self._write("gate2", body)
        return self._write_rerun(prior, body)

    def _write_rerun(self, prior: dict, body: dict) -> dict:
        doc = self._doc()
        if GATE2_INVALID in doc or doc.get("gate2") != prior:
            refuse("the summary changed during gate ②'s rerun; nothing written")
        block = self._stamp("gate2", body)
        doc[GATE2_INVALID], doc["gate2"] = prior, block
        s_store.write_json(self.summary_path, doc, self.plist)
        return block
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_s_runner.py -q`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/s_runner.py tests/brain/s_world.py tests/brain/test_s_runner.py
git commit -m "feat(s): s_runner (1) — chain with reuse re-check (exit 7), set, smoke, OC, gate-2 OC, gate 2 L+C with one INVALID rerun"
```

---

### Task 6: `s_runner` (2) and `scripts/run_s.py` — judgement measurement, seal, judge read once with S.9.2's re-generation, recovery; the CLI; mutation, fault-injection and import tests

**Files:**
- Modify: `flymon/brain/s_runner.py` (append to class `Runner`, which ends the file)
- Create: `scripts/run_s.py`
- Test: `tests/brain/test_s_judge.py`

**Interfaces:**
- Consumes: Task 5's `Runner` and module names; `r_records.preread_validity` / `judge_inputs` / `compare` / `cond_summary`; `s_store.load_manifest` / `archive_copy`; `s_rules.g_fail_s` / `read_band` / `sentence`.
- Produces: `Runner.stage_jm(name)`, `_raws`, `stage_seal()`, `_read(doc, mark=None)`, `stage_judge()`, `_require_after_judge`, `stage_recompute(note)`, `stage_invalid_run(note)`; `scripts/run_s.py` with `STAGES`, `POOL_STAGES`, `check_args`, `exit_code(stage, out)`, `main(argv=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_s_judge.py
"""The judgement end of the S chain (S.3 ⑤-⑦, S.4, S.5, S.9.2, S.9.5; Readings 4, 10-12) and its mutation tests: jm
blocks carry no pair statistic and are written only when all 64 pairs are back; the reuse condition is re-checked
before the judgement measurement, the seal and the judge (exit 7); the seal re-checks every raw file, runs the pre-read
validity, archives 192 raw files and pins the decision code; the judge reads only a sealed set under the sealed
decision code, once — with S.9.2's one re-generation after an interruption between the mark and the block (fault
injection); G_fail_S fires on a net drop of 3 and never on a symmetric swap; the CLI's refusals and exit codes; S never
imports a modified copy of a shared measurement file; the judgement set is touched only from jm on."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_measure, r_store, s_store
from flymon.brain import s_runner as SR
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.s_spec import SPEC
from flymon.brain.s_store import SCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.s_world import CODE, Scripted, World, doc, through_gate2

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through_gate2(w, m)
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def _sealed(w, plan=None):
    r = _to_seal(w, Scripted(plan=plan or w.pass_plan(n=14, c=8, f_a=3)))
    assert r.stage_seal()["status"] == "SEALED"
    return r


def test_full_chain_reads_selected(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan(n=14, c=8, f_a=3)))
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 64 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 64
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 192 and seal["set"]["n_a"] == 43
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive)) and seal["decision"]["key"] == SR.decision_key()["key"]
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    rho = doc()["gate2"]["ratio"]
    assert (f"(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 "
            f"{rho['r1']['ratio']:.3f}·ℓ_r2 {rho['r2']['ratio']:.3f} ≥ 0.5(약화 정도 기록), 짝수 16/21") in out["sentence"]
    assert out["resumed_after_mark"] is False and out["mark_read_at"] is None
    assert out["g_fail"]["g_fail"] is False and out["records"]["transitions_C_to_L"]["b"]["testable"]["fp"] == 6
    assert out["oc_sha256"] == doc()["oc"]["sha256"] and set(doc()) == set(SR.ORDER)
    assert Path(SR.JUDGE_MARKER).exists() and Path(SR.DONE_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()                                     # once


def test_net_drop_3_on_a_reads_the_punish_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", "L")].update({k: (False, False, False) for k in ja[30:33]})       # 3 (a) pass->fail, none back
    out = _sealed(w, plan).stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["a"]["net_drop"] == 3
    assert out["sentence"] == ("M2 (b) 기준과 여유는 넘었지만 지렛대 아래 처벌 통과가 순감소했다(처벌 통과 (b) 21 대 21, "
                               "(a) 40 대 43; 순감소 (b) 0, (a) 3).")


def test_a_symmetric_swap_of_3_does_not_fire_the_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", "L")].update({k: (False, False, False) for k in ja[30:33]})       # 3 pass->fail
    plan[("judge", "C")] = dict(plan[("judge", "C")], **{k: (False, False, False) for k in ja[33:36]})  # 3 fail->pass
    out = _sealed(w, plan).stage_judge()
    ax = out["g_fail"]["axes"]["a"]
    assert (ax["pass_to_fail"], ax["fail_to_pass"], ax["net_drop"]) == (3, 3, 0) and out["band"] == "SELECTED"


def test_judgement_set_is_touched_only_from_jm_on(w):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = Scripted(plan=w.pass_plan())
    r = through_gate2(w, m)
    m.fail_once = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 64


@pytest.mark.parametrize("where", ["jm", "seal", "judge"])
def test_reuse_is_rechecked_before_measurement_seal_and_judge(w, where):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    w.r["gate3"]["code_key"] = "x" * 64                      # R's block rewritten under another key
    with pytest.raises(SystemExit) as e:
        {"jm": lambda: r.stage_jm("L"), "seal": r.stage_seal, "judge": r.stage_judge}[where]()
    assert e.value.code == SR.EXIT_REUSE
    assert not Path(SR.JUDGE_MARKER).exists()


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))
    d = doc()
    del d["gate2"]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and w.judgement_calls == 0


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    r = _to_seal(w, Scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)}))
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_an_edit_in_c_fails_the_preread_validity(w):
    m = Scripted(plan=w.pass_plan(), override={("judge", "C"): dict(edges=2, edit=SPEC.lever_edit, sha="sha-L")})
    seal = _to_seal(w, m).stage_seal()
    assert seal["status"] == "NOT_READ" and any("C" in x for x in seal["reasons"])


def test_mutation_a_tampered_set_refuses(w):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("S set digest_keys: generated '00', declared '884d'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_mutation_a_manifest_pointing_c_at_l_files_fails_the_preread_validity(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    d = doc()
    d["jm:C"]["manifest"] = d["jm:L"]["manifest"]
    Path(SPEC.summary).write_text(json.dumps(d))
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any(x.startswith("C: ") and "stored inputs" in x for x in seal["reasons"])


def test_judge_refuses_a_raw_file_changed_after_the_seal(w):
    r = _sealed(w)
    f = Path(doc()["jm:C"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"sha-C"', '"sha-X"'))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in doc() and not Path(SR.JUDGE_MARKER).exists()


def test_an_earlier_committed_judge_or_another_code_key_refuses(w):
    m = Scripted(plan=w.pass_plan())
    r = through_gate2(w, m)
    w.judge_commits.append("abc")
    with pytest.raises(SystemExit):
        r.stage_jm("L")
    w.judge_commits.clear()
    with pytest.raises(SystemExit):
        w.runner(m, code={"key": "z" * 64}).stage_jm("L")


def test_judge_twice_with_the_first_block_discarded_refuses(w):
    r = _sealed(w)
    assert r.stage_judge()["status"] == "READ"
    marker = json.loads(Path(SR.JUDGE_MARKER).read_text())
    assert marker["seal_written_at"] == doc()["seal"]["written_at"] and marker["read_at"]
    d = doc()
    del d["judge"]                                          # the judge block discarded (git checkout)
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()                                     # DONE marker: no re-generation for a written block
    assert e.value.code == 2 and "judge" not in doc()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(SR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(SR.JUDGE_MARKER).exists()


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(SR.s_rules, "read_band", lambda *a, **k: dict(band=SR.s_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == SR.s_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(SR.JUDGE_MARKER).exists()


class _Kill(Exception):
    pass


def _kill_judge_write(monkeypatch, seen):
    """Fault injection (S.9.2): the process dies after the mark, while the judge block is being written."""
    orig = s_store.write_summary_block

    def boom(path, block, obj, plist):
        if block == "judge":
            seen.append(obj)
            raise _Kill()
        return orig(path, block, obj, plist)
    monkeypatch.setattr(SR.s_store, "write_summary_block", boom)
    return orig


def test_resume_after_mark_once(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    mark = json.loads(Path(SR.JUDGE_MARKER).read_text())
    assert "judge" not in doc() and not Path(SR.DONE_MARKER).exists() and not Path(SR.REREAD_MARKER).exists()
    monkeypatch.setattr(SR.s_store, "write_summary_block", orig)
    out = r.stage_judge()                                   # the one re-generation
    assert out["status"] == "READ" and out["resumed_after_mark"] is True and out["mark_read_at"] == mark["read_at"]
    assert (out["band"], out["sentence"]) == (seen[0]["band"], seen[0]["sentence"])
    assert doc()["judge"]["resumed_after_mark"] is True and Path(SR.REREAD_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_a_second_regeneration_refuses(w, monkeypatch):
    r = _sealed(w)
    seen = []
    _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    with pytest.raises(_Kill):
        r.stage_judge()                                     # the re-generation dies too
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and len(seen) == 2 and "judge" not in doc()


def test_regeneration_refuses_another_decision_code_or_seal(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    monkeypatch.setattr(SR.s_store, "write_summary_block", orig)
    real = SR.decision_key
    monkeypatch.setattr(SR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit):
        r.stage_judge()                                     # another decision code
    monkeypatch.setattr(SR, "decision_key", real)
    m = json.loads(Path(SR.JUDGE_MARKER).read_text())
    Path(SR.JUDGE_MARKER).write_text(json.dumps(dict(m, seal_written_at="another seal")))
    with pytest.raises(SystemExit):                         # a marker from another seal
        r.stage_judge()
    assert "judge" not in doc() and not Path(SR.REREAD_MARKER).exists()


def test_recovery_after_reading(w):
    r = _sealed(w)
    r.stage_judge()
    with pytest.raises(SystemExit):
        r.stage_recompute("")
    e = r.stage_recompute("records code fix (test)")
    assert e["band"] == "SELECTED" and e["differs_from_judge"] is False and e["decision_changed_since_seal"] is False
    assert r.stage_invalid_run("measurement defect (test)")["status"] == "INVALID_RUN"
    with pytest.raises(SystemExit):
        r.stage_invalid_run("again")
    with pytest.raises(SystemExit):
        r.stage_recompute("after invalid")


class OraclePool:
    """r_oracle_job-shaped outputs for a real RMeasurer + SCache; the pair key travels in the odour name ("G|<key>"
    E-grid, "E|<key>" E0)."""

    def __init__(self, plan):
        self.n_workers, self.plan, self.jobs = 4, plan, 0

    def run_jobs(self, fn, kws):
        assert fn is r_jobs.r_oracle_job
        out = []
        for kw in kws:
            self.jobs += 1
            lever = kw["edit"] == SPEC.lever_edit
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            flags = self.plan.get(("judge", cond), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha="sha-L" if lever else "sha-C", edges=SPEC.lever_edges if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_real_cache_round_trip_seals_and_reads(w):
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through_gate2(w, Scripted(plan=plan))
    pool = OraclePool(plan)
    real = w.runner(RMeasurer(pool, SCache(SPEC.cache_dir, CODE), SPEC, w.ctx["params"], READOUT, Z, TYPES, 100))
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 64
    assert pool.jobs == 192
    f = doc()["jm:L"]["manifest"][0]["cache_file"]
    stored = json.loads(Path(f).read_text())["inputs"]
    assert f.startswith("results/s/cache/r_oracle/") and stored["act_seeds"] == SPEC.judge_seeds()["act"]
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds", "s_set"}
# s_spec's None: the judgement seed fields at class level
ALLOWED = {"s_spec.py": {None, "judge_seeds"}, "s_pairs.py": {"judgement_rows"},
           "s_runner.py": {"build_ctx", "_judgement_rows", "stage_set", "stage_jm", "stage_seal", "_read", "_cost"}}


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
    files = sorted((ROOT / "flymon/brain").glob("s_*.py")) + [ROOT / "scripts/run_s.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def test_s_never_imports_a_modified_copy_of_a_shared_measurement_file():
    """S.3 ① / S.9.5: S runs R's jobs, measurer and cache themselves — the objects S uses are the ones defined in the
    R_MEASURE_FILES modules, no S file defines a job, a measurer or a cache of its own (SCache only subclasses
    RCache's writer), and no S file loads code by path."""
    assert s_store.RCache is r_store.RCache and issubclass(SCache, r_store.RCache)
    assert {k for k in vars(SCache) if not k.startswith("__")} == {"put"}
    for obj in (r_jobs.kc_activity_job, r_jobs.r_arm_job, r_jobs.r_oracle_job, RMeasurer, r_store.RCache):
        src = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
        assert src in R_MEASURE_FILES, src
    shared = {"kc_activity_job", "r_arm_job", "r_oracle_job", "RMeasurer", "RCache", "ECache", "q_oracle_job",
              "arm_job", "q_rig"}
    for p in sorted((ROOT / "flymon/brain").glob("s_*.py")) + [ROOT / "scripts/run_s.py"]:
        tree = ast.parse(p.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not defs & shared, (p.name, defs & shared)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not names & {"importlib", "exec", "spec_from_file_location", "runpy"}, p.name


def _cli():
    spec = importlib.util.spec_from_file_location("run_s_for_test", ROOT / "scripts/run_s.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_argument_refusals_and_exit_codes(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    for argv in (["--stage", "reuse", "--condition", "L"], ["--stage", "jm"], ["--stage", "recompute"],
                 ["--stage", "smoke", "--note", "x"], ["--stage", "set", "--rerun-after-invalid"], [],
                 ["--stage", "reuse"]):                                   # the last: wrong cwd
        assert cli.main(argv) == 2, argv
    assert cli.exit_code("reuse", {"outcome": "PASS"}) == 0 and cli.exit_code("reuse", {"outcome": "STOP_REUSE"}) == 3
    assert cli.exit_code("set", {"outcome": "STOP_SET_SHORT"}) == 3
    assert cli.exit_code("gate2", {"outcome": "STOP_PUNISH_WEAKENED"}) == 3
    assert cli.exit_code("gate2", {"outcome": "STOP_P_REFERENCE"}) == 3
    assert cli.exit_code("gate2", {"outcome": "INVALID"}) == 5
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    assert cli.exit_code("oc", {}) == 0 and cli.exit_code("gate2_oc", {}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "INVALID"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_s.py").read_text()


def test_hashed_pipeline_and_decision_files():
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.r_runner import R_HASHED_FILES
    assert set(SR.S_PIPELINE_FILES) == {f"flymon/brain/s_{n}.py" for n in
                                        ("spec", "pairs", "store", "records", "rules", "runner")} | {"scripts/run_s.py"}
    assert not set(SR.S_PIPELINE_FILES) & set(R_MEASURE_FILES)
    assert set(R_HASHED_FILES) | set(SR.S_PIPELINE_FILES) <= set(SR.S_HASHED_FILES)
    assert "results/summary/r_lever.json" in SR.S_HASHED_FILES
    assert set(SR.DECISION_FILES) == set(SR.S_HASHED_FILES) - set(R_MEASURE_FILES)
    assert len(SR.pipeline_key()["key"]) == 64 and set(SR.pipeline_key()["files"]) == set(SR.S_PIPELINE_FILES)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_s_judge.py -q`
Expected: failures with `AttributeError: 'Runner' object has no attribute 'stage_jm'` and `FileNotFoundError` for `scripts/run_s.py`.

- [ ] **Step 3: Append the judgement methods to class `Runner`** (at the end of `flymon/brain/s_runner.py`, indented as methods)

```python
    # ---- the judgement (S.3 ⑤–⑦, S.5, S.9.2) ------------------------------------------------------------------------
    def _judge_chain(self, doc: dict, stage: str) -> None:
        """The judgement runs once, from clean trees, on the shared key every earlier block ran on."""
        if self.spec.smoke:
            refuse("smoke never measures the judgement set (S.3 ②)")
        if not self.code_key:
            refuse("no code key given; the judgement must run on the code the gates ran on")
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the set is used once (S.5)")
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("code_key") != self.code_key:
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
        """One judgement condition on the 64 pairs (resumable); the block holds raw_check only."""
        sp = self.spec
        if name not in sp.cond_names:
            refuse(f"unknown condition {name}; S has {sp.cond_names}")
        stage = f"jm:{name}"
        doc = self._require(stage)
        self._judge_chain(doc, stage)
        rows = self._judgement_rows()
        seeds, cond = sp.judge_seeds(), sp.cond(name)
        got = self.m.oracle(rows, cond, r_records.JUDGE_BLOCK, seeds)
        rc = r_records.raw_check(got, cond, [row_key(r) for r in rows], seeds)
        man = [dict(m, sha256=sha256_file(m["cache_file"])) for m in rc["manifest"]]
        body = dict(rc, manifest=man, seeds=seeds, condition=name, wall_s=self.m.last_wall_s, jobs=self.m.last_jobs)
        self._write(stage, body)
        return body

    def _raws(self, doc: dict) -> tuple:
        raws, bad = {}, []
        for n in self.spec.cond_names:
            got, b = s_store.load_manifest(doc[f"jm:{n}"]["manifest"])
            raws[n] = got
            bad += [f"{n}: {x}" for x in b]
        return raws, bad

    def stage_seal(self) -> dict:
        """R.9.7 as is (S.3 ⑥): pre-read validity; SEALED -> the archive copy; the decision code hash pinned."""
        doc = self._require("seal")
        self._judge_chain(doc, "seal")
        sp = self.spec
        rows = self._judgement_rows()
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        v = r_records.preread_validity({n: doc[f"jm:{n}"] for n in sp.cond_names}, raws, sp, self.ctx["z"], keys,
                                       self.code_key, doc["reuse"]["records"]["repro_csc_sha256_none"],
                                       r_records.judge_inputs(self.m, rows, sp))
        reasons = bad + v["reasons"]
        status = s_rules.INVALID if v["invalid"] else (s_rules.NOT_READ if reasons else s_rules.SEALED)
        manifest = [dict(m, condition=n) for n in sp.cond_names for m in doc[f"jm:{n}"]["manifest"]]
        body = dict(status=status, reasons=reasons, invalid=v["invalid"], checks=v["checks"], manifest=manifest,
                    n_files=len(manifest), archive=None, decision=decision_key(),
                    set=dict(digest_e0_b=sp.digest_e0_b, digest_e0_a=sp.digest_e0_a, digest_keys=sp.digest_keys,
                             last_turn=sp.last_turn, n_b=sp.n_b, n_a=sp.n_a))
        if status == s_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=s_store.archive_copy([m["cache_file"] for m in manifest], dest,
                                                                              self.archive_root))
        self._write("seal", body)
        return body

    def _read(self, doc: dict, mark=None) -> dict:
        """The bands and records from the sealed raw files (sha re-checked): R.3's order (r_rules.read_band) with
        G_fail_S, S.7's sentence, R.6 / S.6's records."""
        sp = self.spec
        keys = [row_key(r) for r in self._judgement_rows()]
        raws, bad = self._raws(doc)
        if bad:
            refuse(f"raw files changed since the seal: {bad[:3]}")
        if mark is not None:
            mark()                                       # the read starts here
        seeds = sp.judge_seeds()
        s = {n: r_records.cond_summary(raws[n], sp.cond(n), sp, self.ctx["z"], keys, seeds) for n in sp.cond_names}
        L, C, E0 = (s[n] for n in sp.cond_names)
        if L["aggregate"] is None or C["aggregate"] is None:
            refuse("no (b) pair to aggregate")
        aL, aC = L["aggregate"], C["aggregate"]
        gf = s_rules.g_fail_s(L["pairs"], C["pairs"], sp)
        rb = s_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                               aL["naive_a"], sp)
        if rb["band"] == s_rules.NOT_READ:               # no number of the set leaves a NOT_READ read
            return dict(status=s_rules.NOT_READ, band=rb["band"], reason=rb["reason"],
                        reasons=["the (b) / (a) pair counts differ from the declared counts (S.4 line 1)"])
        ax, ratio = gf["axes"], doc["gate2"]["ratio"]
        names = list(sp.p.directions)
        fields = dict(n=aL["testable_b"], c=aC["testable_b"], f_a=aL["F_a"], naive_a=aL["naive_a"], T=sp.last_turn,
                      k_even=doc["reuse"]["records"]["k_even"], pb_L=ax["b"]["pun_L"], pb_C=ax["b"]["pun_C"],
                      pa_L=ax["a"]["pun_L"], pa_C=ax["a"]["pun_C"], d_b=ax["b"]["net_drop"], d_a=ax["a"]["net_drop"],
                      rho1=f"{ratio[names[0]]['ratio']:.3f}", rho2=f"{ratio[names[-1]]['ratio']:.3f}",
                      reason=rb["reason"])
        out = dict(band=rb["band"], reason=rb["reason"], n=aL["testable_b"], c=aC["testable_b"], F_a=aL["F_a"],
                   naive_a=aL["naive_a"], n_b=aL["n_b"], n_a=aL["n_a"], g_fail=gf)
        if rb["band"] == s_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        return dict(out, status=s_rules.READ, sentence=s_rules.sentence(rb["band"], fields),
                    records=r_records.compare(L, C, E0, sp), pairs={n: s[n]["pairs"] for n in sp.cond_names},
                    oc_sha256=doc["oc"]["sha256"], p_labels=dict(L=doc["gate2"]["label_L"], C=doc["gate2"]["label_C"]),
                    p_ratio={d: ratio[d]["ratio"] for d in names}, k_even=doc["reuse"]["records"]["k_even"])

    def stage_judge(self) -> dict:
        """Once (S.3 ⑦). The marker is written before the band is computed. S.9.2: with the marker and no judge block
        (an interruption between the mark and the block), judge is re-generated once — same sealed raw data (sha
        re-checked), the sealed decision code, no measurement; the block says resumed_after_mark and the mark's time; a
        second re-generation, a marker from another seal or decision code, or a judge block once written (DONE marker)
        refuses."""
        doc = self._require("judge")
        self._judge_chain(doc, "judge")
        if doc["seal"].get("status") != s_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: S reads only a sealed set (R.9.7)")
        sealed = (doc["seal"].get("decision") or {}).get("key")
        now = decision_key()["key"]
        if sealed != now:
            refuse(f"the decision code hash {now} is not the sealed {sealed}: the judgement reads only under the code "
                   f"it was sealed with (S.5)")
        if Path(DONE_MARKER).exists():
            refuse(f"{DONE_MARKER} exists: a judge block was written once; a discarded block does not reopen the set")
        resumed = None
        if Path(JUDGE_MARKER).exists():
            if Path(REREAD_MARKER).exists():
                refuse(f"{REREAD_MARKER} exists: judge was re-generated once after the mark already (S.9.2)")
            mk = json.loads(Path(JUDGE_MARKER).read_text())
            if mk.get("seal_written_at") != doc["seal"].get("written_at") or mk.get("decision_key") != sealed:
                refuse(f"{JUDGE_MARKER} belongs to another seal or decision code; no re-generation (S.9.2)")
            resumed = mk

        def mark():
            s_store.write_json(REREAD_MARKER if resumed else JUDGE_MARKER, dict(
                seal_written_at=doc["seal"].get("written_at"), seal_archive=(doc["seal"].get("archive") or {}).get("dir"),
                decision_key=now, code_key=self.code_key, pipeline_key=self.pipeline_key, read_at=_now()), self.plist)
        out = self._read(doc, mark)
        if out["status"] != s_rules.READ:
            return out                                   # NOT_READ: no block, the marker stays
        out = dict(out, resumed_after_mark=resumed is not None,
                   mark_read_at=(resumed or {}).get("read_at"))
        block = self._write("judge", out)
        s_store.write_json(DONE_MARKER, dict(judge_written_at=block["written_at"]), self.plist)
        return block

    # ---- recovery after reading (S.5 = R.5) --------------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (S.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: this set is closed (S.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """S.5 row 2: an analysis or summary defect after reading — the same sealed raw data recomputed."""
        if not note:
            refuse("--note is required (S.5: the correction is recorded)")
        doc = self._require_after_judge("recompute")
        out = self._read(doc)
        dk = decision_key()["key"]
        entry = dict(note=note, status=out["status"], band=out["band"], reason=out["reason"], n=out.get("n"),
                     c=out.get("c"), F_a=out.get("F_a"), naive_a=out.get("naive_a"), g_fail=out.get("g_fail"),
                     sentence=out.get("sentence"), decision_key=dk,
                     decision_changed_since_seal=bool(dk != (doc["seal"].get("decision") or {}).get("key")),
                     differs_from_judge=bool(out["band"] != doc["judge"]["band"]
                                             or out.get("sentence") != doc["judge"].get("sentence")),
                     code_key=self.code_key, pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())
        s_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """S.5 row 3: a measurement defect after reading — INVALID_RUN; there is no replacement set (S.5)."""
        if not note:
            refuse("--note is required (S.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=s_rules.INVALID_RUN, note=note, judge_band=doc["judge"]["band"],
                    rule="S.5: the same set is never run again and no replacement set exists; the lever's M2 judgement "
                         "ends without a conclusion (user)")
        self._write("invalid_run", body)
        return body
```

- [ ] **Step 4: Write the CLI**

```python
#!/usr/bin/env python3
"""Spec appendix S (S.9 wins over S.0-S.8): the same lever (APL->MBON05 removal) on the last unused judgement set (L
generator turns 104-209), one M2 judgement. The controller runs every stage; commit each block before the next.

    uv run python scripts/run_s.py --stage reuse                     # S.3 ① / S.9.5 R's gates reused (no pool)
    uv run python scripts/run_s.py --stage set                       # S.2 the set, list only (no pool)
    uv run python scripts/run_s.py --stage smoke --workers 4         # 24_309_xxx / 25_109_xxx, even (b) 0 and 20 only
    uv run python scripts/run_s.py --stage oc                        # S.6 operating characteristic (no pool)
    uv run python scripts/run_s.py --stage gate2_oc                  # S.9.7 gate ② operating characteristic (no pool)
    uv run python scripts/run_s.py --stage gate2                     # ② P_L and P_C on 25_100_000+i
    uv run python scripts/run_s.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (S.3 ④)
    uv run python scripts/run_s.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_s.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_s.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_s.py --stage recompute --note "..."    # after judge: analysis defect (S.5)
    uv run python scripts/run_s.py --stage invalid_run --note "..."  # after judge: measurement defect (S.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_SET_SHORT, gate ②'s three
STOPs; recorded, S stops), 5 gate ② INVALID, 6 seal not SEALED, judge NOT_READ or a smoke with problems, 7 R's reuse
condition broke after `reuse` (S stops; no block), 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted
summary, dirty hashed file, data pins)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "set", "smoke", "oc", "gate2_oc", "gate2", "jm", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("smoke", "gate2", "jm")
QUIET = ("pairs", "manifest", "records", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive", "checks",
         "g_fail", "set")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (S.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import s_rules as R
    if stage in ("reuse", "set", "gate2"):
        o = out.get("outcome")
        return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)
    if stage == "smoke":
        return EXIT_NOT_READ if out.get("problems") else 0
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

    from flymon.brain import s_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
    from flymon.brain.s_spec import SPEC
    from flymon.brain.s_store import SCache

    try:
        ctx = s_runner.build_ctx(SPEC, NPZ)
        code = code_key(NPZ, files=R_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        root = SPEC.smoke_cache_dir if a.stage == "smoke" else SPEC.cache_dir
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            m = RMeasurer(pool, SCache(root, code), SPEC, ctx["params"], ctx["readout"], ctx["z"], ctx["types"],
                          ctx["n_kc"])
            r = s_runner.Runner(m, ctx, SPEC, code=code, pipeline=s_runner.pipeline_key())
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

- [ ] **Step 5: Run the S tests and the full suite**

Run: `uv run pytest tests/brain/test_s_judge.py -q`
Expected: 28 passed.

Run: `uv run pytest -q`
Expected: the whole suite passes (in the scratch worktree used to draft this plan it passed with only the skips caused by the missing git-ignored `results/` caches).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/s_runner.py scripts/run_s.py tests/brain/test_s_judge.py
git commit -m "feat(s): s_runner (2) and run_s.py — jm, seal, judge once with one re-generation after the mark, recovery; mutation tests"
```

---

## Self-review (done while writing)

- **Spec coverage:**

  | Spec item | Where it lives |
  |---|---|
  | S.0 / S.9.1 / S.9.4 disclosures | no code; the OC notes (Task 4), the result section S.10 |
  | S.1 model, encoder, readout, z | inherited from R's spec (Task 1), R's measurer (Task 3), `build_ctx` (Task 5) |
  | S.2 set, declared values, STOP_SET_SHORT, exclusion tests | Task 2; stage `set` (Task 5); re-check per jm / seal / judge (Task 6) |
  | S.3 ① / S.9.5 reuse, shared key, two keys, re-checks | Task 1 key test, Task 4 `reuse`, Task 5 `stage_reuse` / `_reuse_now`, Task 6 re-check tests and import test (Reading 4) |
  | S.3 ② smoke | Task 5 `stage_smoke` (Reading 6) |
  | S.3 ③ / S.6 OC; S.9.7 notes | Task 4 `oc` (declared values reproduced), Task 5 `stage_oc` |
  | S.3 ④ / S.9.3 / S.9.6 gate ②; S.9.7 gate ②'s OC | Task 1 `P_L` / `P_C` and one-edit fixture, Task 4 `gate2` / `ratio` / `gate2_oc`, Task 5 `stage_gate2_oc` / `stage_gate2` |
  | S.3 ⑤–⑦ judgement measurement, seal, judge marker | Task 6 |
  | S.4 bands, G_fail_S, exhaustive and boundary fixtures | Task 4 |
  | S.5 / S.9.2 recovery and re-generation after the mark | Task 6 (fault-injection tests) |
  | S.6 records | `r_records.compare` / `cond_summary` via Task 6 `_read` (Reading 14) |
  | S.7 sentences | Task 4 (verbatim tests) |
  | S.8 / S.9.6 files, writes, seeds, global collision | Tasks 1, 3; Global Constraints |
  | S.9.8 implementation choice, task count, review cost | Readings 2–4; header |

- **Placeholders:** none. Every code step has its full code; the code is the code that ran in the scratch worktree.
- **Type consistency:** `Runner` methods used by `through_gate2` / `_to_seal` exist in Tasks 5 / 6; block fields read later (`reuse.records.repro_csc_sha256_none`, `reuse.records.k_even`, `gate2.ratio`, `gate2.label_L`, `oc.sha256`, `seal.decision.key`, `seal.written_at`) are written by the stages that own them; `Scripted.arms` / `oracle` match `RMeasurer.arms` / `oracle`.
- **Review Focus:** each item has its test in its owning task (listed in the section).

## Runs (controller)

The controller does every run, from `/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot` (branch `open-fly-brain-connectome`), with Bash `run_in_background: true` and `timeout: 7200000`, after the full suite passes at the final implementation commit.
- Subagents never run these steps.
- When a run finishes, read only its exit code and the last lines of its output.
- No rule or number changes after any run.
- Each block is committed before the next stage; the stage refuses otherwise.
- `results/summary/s_lever.json` is tracked; `results/s/` is git-ignored. `results/summary/r_lever.json` is never touched.
- Commit messages carry no trailers.

**The 2 h background limit:**
- The longest single runs are `gate2` (~0.3 h by the scratch smoke's estimate: 384 arms) and each `jm` (~0.5 h: 64 pairs), which is why the judgement runs one condition per invocation.
- If a run is stopped at the limit, or dies, rerun the same command. It resumes from `results/s/cache/` with only the missing units, and no block is written until a stage is complete.
- Never run two pool stages at once.

**Stop points:** `STOP_REUSE` (exit 3) or any exit 7; `STOP_SET_SHORT` (exit 3) or a `set` refusal (exit 2: the set does not reproduce); a smoke with problems (exit 6) after one diagnosed fix attempt; any gate ② STOP (exit 3); a second gate ② `INVALID`; seal `NOT_READ` / `INVALID` (exit 6); judge `NOT_READ` (exit 6); and the judgement result itself. At a stop point:
1. Commit the block (a smoke block with problems is not committed; see step 3).
2. Overwrite `/private/tmp/claude-503/p-latest.md` with the full text:
   - the stage and its label or band;
   - the closing sentence verbatim;
   - the numbers behind it;
   - the commits;
   - what was or was not written (S.10, README ledger);
   - the decision S.7 leaves to the user.
3. Stop. Do not continue to another stage.

0. **Preconditions (no pool):**
   1. `git status` must be clean and HEAD must be the final implementation commit; `uv run pytest -q` passes.
   2. `ls results/p/run/cache/p_arm | wc -l` → 192 (gate ②'s OC reads P's block).
   3. R's reused blocks are in history: `git merge-base --is-ancestor 22c934b HEAD && git merge-base --is-ancestor 6ad2201 HEAD && git merge-base --is-ancestor a83977f HEAD`.
   4. `mkdir -p ~/flymon-archive/s`.
1. **Reuse (S.3 ①, S.9.5):**
   1. Run `uv run python scripts/run_s.py --stage reuse` (seconds, no pool).
   2. On exit 0, commit: `git add results/summary/s_lever.json && git commit -m "results(s): reuse — shared key 3c2699c7 = R's; R repro 22c934b, gate 1 6ad2201, gate 3 a83977f reused (k_even 16/21)"`.
   3. On exit 3 (`STOP_REUSE`): commit `results(s): reuse STOP_REUSE — <reasons>`, then go to the **stop point** (Reading 4: re-measurement is the user's call and needs its own plan).
2. **Set (S.2):**
   1. Run `uv run python scripts/run_s.py --stage set` (no pool).
   2. On exit 0, commit: `git commit -m "results(s): set — L turns 104-177, (b) 21 · (a) 43, digests 2b92f40a / 7b534675 / 884d49cc reproduced"`.
   3. On exit 3 (`STOP_SET_SHORT`): commit, then **stop point**. On exit 2: **stop point** (nothing written; the set does not reproduce its declaration).
3. **Smoke (S.3 ②):**
   1. Run `uv run python scripts/run_s.py --stage smoke --workers 4`.
   2. Expect exit 0 and `problems: []`: P L edges `[2]`, C `[0]`, both labelled (no INVALID); oracle L `[2]`, C / E0 `[0]`; L CSC = R smoke's L (`860cba4f…`, recorded in `reuse.records`); C / E0 CSC = R's repro (`1aee8398…`).
   3. Report the `cost` estimate in one line, then commit: `git commit -m "results(s): smoke — lever edges 2 (P_L, oracle), P_C 0, sha relations OK, cost estimate <gate2 h> / <jm h per condition>"`.
   4. On exit 6: do not commit the block. Diagnose with `superpowers:systematic-debugging`, have a reviewed fix task (it must not touch a shared measurement file — that ends in exit 7), discard the block (`git checkout -- results/summary/s_lever.json`), rerun once. A second failure is a **stop point**.
4. **Operating characteristic (S.6):**
   1. Run `uv run python scripts/run_s.py --stage oc` (seconds).
   2. Commit: `git commit -m "results(s): operating characteristic — G_fail_S null 0.034/0.145/0.281/0.362, harm 0.752/0.949/0.961/0.549 beside R's guard"`.
5. **Gate ②'s OC (S.9.7):**
   1. Run `uv run python scripts/run_s.py --stage gate2_oc` (seconds; reads P's 192 entries).
   2. Expect `p_stop_weakened` ≈ 0.998 / 0.750 / 0.080 / 0.000 / 0.000 at ρ 0.4 / 0.5 / 0.61 / 0.83 / 1.0 (Reading 10).
   3. Commit: `git commit -m "results(s): gate 2 operating characteristic — P(STOP_PUNISH_WEAKENED) at true ratio 0.4/0.5/0.61/0.83/1.0 recorded before gate 2"`.
6. **Gate ② (S.3 ④):**
   1. Run `uv run python scripts/run_s.py --stage gate2` (~0.3 h; resumes after a stop).
   2. On exit 0, commit: `git commit -m "results(s): gate 2 PASS — P_L and P_C LEARNS_CONFIRMATORY on 25_100_xxx, ratio r1 <·> r2 <·> (>= 0.5)"`.
   3. On exit 3 (`STOP_PUNISH_BROKEN` / `STOP_P_REFERENCE` / `STOP_PUNISH_WEAKENED`): commit, then **stop point**.
   4. On exit 5 (INVALID): commit the block; fix the S code with a reviewed task (the pipeline key must change); run `uv run python scripts/run_s.py --stage gate2 --rerun-after-invalid` once. Its outcome is final. If the fix needs a shared measurement file, stop instead (exit 7 path).
7. **Judgement measurement (one condition per run, in this order: L, C, E0):**
   1. Run `uv run python scripts/run_s.py --stage jm --condition <L|C|E0>` (~0.5 h each; resumes after a stop).
   2. The blocks hold no statistic. Do not compute anything from `results/s/cache/` before the seal.
   3. Commit after each: `git commit -m "results(s): judgement measurement <cond> — 64 pairs complete (no band read)"`.
8. **Seal (R.9.7):**
   1. Run `uv run python scripts/run_s.py --stage seal` (no pool).
   2. On exit 0 (`SEALED`), check `find <seal.archive.dir> -type f | wc -l` → 192, then commit: `git commit -m "results(s): seal — pre-read validity OK, raw manifest (192 entries), decision code pinned, archive ~/flymon-archive/s/<dir>"`.
   3. On exit 6 (`NOT_READ` / `INVALID`): commit, then **stop point**. The judgement is not read.
9. **Judge (once):**
   1. Run `uv run python scripts/run_s.py --stage judge` (no pool).
   2. If it died after writing `results/s/judge_read.json` and before the block (no `judge` block, no `results/s/judge_done.json`), rerun the same command once: S.9.2's re-generation (`resumed_after_mark: true`). Any further failure is a **stop point**.
   3. Commit: `git commit -m "results(s): judgement — <band> (<n>/21 vs <c>/21, F_a <f>/43, G_fail_S <true|false>)"`.
   4. Write the result before stopping:
      - the spec section `### S.10 결과 (…, 판정 1회 — **<band>**)` at the end of appendix S, in R.10's form: implementation range and block commits; reuse / set / smoke / OC / gate ②'s OC / gate ② (labels, ℓ, ratios with CIs, difference to R's gate ②) / seal (manifest count, archive dir); the judgement (n, c, F_a, naive_a, G_fail_S per axis with net drop and pass→fail / fail→pass); the closing sentence verbatim; the records table (testable / reward / punish / naive per axis, F_a, KC, APL, naive P_X / A, MBON05 per-cell saturation, transitions C → L, z renormalisation); "다음: S.7대로 기록하고 사용자가 판단한다 — 이 지렛대의 M2 판정 세트는 남지 않는다".
      - the README ledger lines, Korean (after R's line ~190) and English (after R's line ~305), in R's form: one bullet each with the band, the numbers and "next is the user's decision".
      - Commit: `git commit -m "docs(s): S.10 result — <band> (<n>/21 vs <c>/21, F_a <f>/43, G_fail_S <…>); README ledger ko/en"`.
   5. Go to the **stop point** with the result: the closing sentence verbatim and S.7's consequence (B_Tb closes the lever; `SELECTED`: an F v4 learning-test declaration is the user's; every other band is recorded for the user). Pushing follows the standing FlyMon push rule (`gh auth switch --user lyutvs`); the verdict itself is reported, not acted on.
