# Spec T — Each Engine's Own Reference-Set z, the Same Lever on a Widened-Pool Set, One M2 Judgement: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix T needs (as replaced by T.9, T.9.7 listing the replacements), and nothing else:
- a **reuse gate** (T.3 1): R's no-edit reproduction (`22c934b`) and gate ① (`6ad2201`) are reused only while the shared measurement key over `r_measure.R_MEASURE_FILES` equals R's `3c2699c7…` and R's summary holds those blocks, committed and passed; re-checked at every later stage.
- the **T set** (T.2 as replaced by T.9.1), list only: the widened Gen-1 opponent pool (127), the permutation by `default_rng(20261004)`, the declared exclusion order (ORN cap → glomerulus collision → used key → glomerulus-level duplicate → in-set duplicate → both type sets in POOL); it must reproduce (b) 21 last turn 17, (a) 43 last turn 103, three digests, the skip counts and the cluster table, or the stage refuses; `STOP_SET_SHORT` when the turns run out.
- **z** (T.1, T.9.4, T.9.6, T.9.7): a T-only measurement file runs the H.3 reference set (96 presentations) and the same-seed rest on an edited engine. The unedited engine, measured afresh, must give block h4's C3 z bit for bit (`STOP_Z_REPRO`); then the lever engine must pass H.4's readout guard per readout type with a nonzero SD (`STOP_Z_DEGENERATE`); else **z_lever**.
- **gate ① supplement** (T.9.1): the KC band [0.03, 0.15] under the lever for each of the T set's 55 odours on the strength seeds (`STOP_STRENGTH_LEVER`).
- **smoke** on new seeds 24_409_xxx (oracle, even (b) pairs 0 and 20, L on z_lever, C / E0 on h4 z) and 25_209_xxx (P_L and P_C). The judgement set is never measured here.
- the **operating characteristics** (T.6, T.9.6, T.9.7): S.6's independent model on n_b 21 · n_a 43, the cluster-correlated model (ICC 0.3, 10⁵ draws, seed 20261005) equal to its committed fixture `tests/brain/fixtures/t_oc_cluster.json`, and gate ②'s OC on h4 z — all recorded before gate ②.
- **gate ②** (T.3 6, T.9.3): P_L and P_C on 25_200_000+i (i < 32), both read on block h4's z, S's order and ratio ≥ 0.5; P_L read on z_lever is a record.
- **gate ③** (T.9.2): C from R's committed even raw by content key (h4 z) must reproduce c_even 7 (`STOP_EVEN_REPRO`); L's 39 even pairs re-measured on z_lever; pass iff L testable_b ≥ 11 and c_even 7.
- the **judgement** on the T set: L (z_lever), C and E0 (h4 z), one condition per run, seeds 24_400_xxx, 64 pairs each.
- **seal** (R.9.7 as is): pre-read validity, the raw manifest (192 entries), the decision-code hash pinned, an archive copy in `~/flymon-archive/t/`.
- **judge**: R.3's bands in fixed order (`r_rules.read_band` itself, n_a 43) with S's G_fail_S (net drop ≥ 3 on either axis), T.7's sentences as replaced by T.9.5 / T.9.7. It reads once, with S.9.2's single re-generation after an interruption between the read mark and the judge block.
- **rs_reread** (T.9.6): R's and S's judgement raw re-read under T's z rule — after judge only, a record.

T is **one M2 judgement on a widened-pool set**. Its STOP labels and bands go to the user unchanged.

**Architecture:**
- New files only: `flymon/brain/t_*.py`, `scripts/run_t.py`, tests and one fixture. The one exception is a single `MODULES` entry in `tests/brain/test_p_spec.py` (Reading 1).
- **No file of R or S is edited** — in particular none of `r_measure.R_MEASURE_FILES` (the shared key). T imports R's jobs, measurer, cache, records and band function and S's guard, gate-② and OC code, and runs them unchanged.
- Modules:
  - `t_spec`: every T number (class `TSpec`, a subclass of S's `LastSetSpec`), the two gate-② P specs `P_L` / `P_C`.
  - `t_pairs`: the widened opponent pool, the T set (list only), its check against T.9.1's declared values, the judgement rows, the set's odours, the cluster labels.
  - `t_measure` (**T's measurement file**, in the T measurement key): the reference-set and rest jobs on an edited engine and `ZMeasurer`.
  - `t_store`: T's guard and writer, `TCache` (R's `RCache` with T's writer), `RReadCache` (R's root, read only).
  - `t_rules`: reuse, set, z, gate-① supplement and gate-③ decisions, the sentences, both OCs.
  - `t_records`: z sides, gate-① supplement record, P_L on z_lever, cluster table, α-fixed sensitivity.
  - `t_runner`: the stage chain.
- `scripts/run_t.py` is the CLI.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`, `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **T** (T.0–T.8) and its red-team amendments **T.9** (T.9.1–T.9.6) and **T.9.7** (re-red-team; it lists the replaced sentences — every replaced sentence is void). T reuses appendix **S** (S.4 guard, S.5 / S.9.2 recovery, S.6 / S.9.7 OCs and records, S's implementation at `541b865`) and **R** (R.3 bands, R.9.7 records, seal and validity, R's implementation), and H.3 / H.4's z procedure (H.3a.3's reference set, H.4 1 and H.4a.3-3: readout guard and z constants).

| T.9 section | Overrides / adds |
|---|---|
| T.9.1 | The ORN cap as the first exclusion; the re-declared set ((b) last turn 17, (a) last turn 103, three digests, skip counts, 55 odours, no E1 clash); the cluster table (9 (b) / 8 (a) clusters; "8개 조합 쌍" reads "9개"); gate ①'s KC band on the T set's odours (`STOP_STRENGTH_LEVER`). Unit tests: ORN cap, E1. |
| T.9.2 | L's oracle (even and judgement, α selection and report) on **z_lever**, C / E0 on h4 z. Gate ③ = L's 39 even pairs re-measured on z_lever; C reused from R's even raw, which must reproduce c_even 7 first (`STOP_EVEN_REPRO`). The "h4 z reading" record is renamed **α-fixed sensitivity**. |
| T.9.3 | Gate ② (labels, c₁ 0.608, ℓ, ratio) reads **both** P_L and P_C on h4 z; P_L on z_lever is a record. Gate ②'s OC on h4 z. |
| T.9.4 | The readout guard on the lever's reference set (median Δ ≥ 5 and zero share ≤ 25% per readout type; SD 0 the same STOP) → `STOP_Z_DEGENERATE`; z_lever's means / SDs and σ_lever / σ_h4 recorded. |
| T.9.5 | Sentences: B_Tb closes T's claim only (not the lever); SELECTED's consequence narrowed; "third judgement" and "9개 상대 타입 조합 쌍" added. |
| T.9.6 | The cluster model (Dirichlet guard rows, Beta testable rows, κ = (1 − ICC)/ICC, c fixed, 10⁵ draws, seed 20261005); R · S raw re-read under T's z only after judge; the T measurement key re-checked before the judgement measurement and the judge, and the unedited z measured with the new job without a cache; `STOP_C_EVEN_MISMATCH` defensive; global seed-collision test with 24_400 / 24_409 / 25_200 / 25_209 and 20261004 / 20261005. |
| T.9.7 | The replacement list (gate ② on h4 z only; z_lever for the L oracle only; T.3 7 → T.9.2; T.7's B_Tb / SELECTED consequences → T.9.5; T.2's declared values → T.9.1); `STOP_Z_REPRO` then `STOP_Z_DEGENERATE` at the z stage; the cluster OC's input rows (null 0.10/0.10, harm 0.10/0.02 and 0.10/0.05, the full oc_q × oc_c × naive_a grid) and its fixture. |

**Task count and review cost:** 6 tasks. Each task is one implementer run and one reviewer run (the `sdd-implementer` / `sdd-reviewer` agents), then one final whole-branch review: **13 subagent runs**. Expected size: ~1,800 lines of new code and ~1,850 lines of tests plus one ~210 KB fixture (S was ~1,000 / ~1,100 over 6 tasks), all of it drafted and run in a detached scratch worktree while this plan was written: the T tests passed (116), and the rest of the suite showed only the 12 `tests/test_run_l.py` failures caused by the scratch worktree's missing git-ignored `results/m0d/` files. On the real connectome and R's summary the real CLI ran `reuse` (PASS, shared key `3c2699c7…`) and `set` (PASS: the generated set reproduced every T.9.1 value — (b) 21 last turn 17, (a) 43 last turn 103, digests `8c9729bf…` / `e31b5526…` / `37dde1ca…`, skips cap 9 · collision 0 · used 164 · glom_dup 0 · in_set 48 · pool_only 57, 55 odours, no E1 clash). Outside the CLI, the scratch also ran `ZMeasurer` on a 16-worker pool for the **unedited** engine over the whole reference set: its z equals block h4's C3 z bit for bit (A 10.78125 / 9.412096743243064, P 26.25 / 19.30889259728101) and its guard statistics equal H.3's (MBON13 Δ 6.0 / zero 0.146, MBON05 22.0 / 0.115) in 37 s; the lever job changed 2 edges (CSC `860cba4f…`, R's L) on one odour. **The lever's z, the gate-① supplement, smoke, gate ②, gate ③ and the judgement were not run there** (their outcomes are gates of the declared run). Tasks 1–4 are independent of each other except for imports of `t_spec` (and Task 4's tests of `t_records`); Task 5 needs 1–4; Task 6 needs 5. The slow tests are Task 2's and Task 3's (they load the connectome; Task 3's runs the engine for one reference odour, ~15 s).

## Global Constraints

- **Commits carry no trailers.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No R or S file changes; none of the shared measurement files changes.** `r_measure.R_MEASURE_FILES` keys the cache, and R's gates are reused only while that key equals R's `3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc` (T.3 1, T.8; test in Task 1). No `r_*`, `s_*` module, `run_r.py`, `run_s.py`, R's or S's tests or fixtures are edited.
- **Never edit** `flymon/agent/e_*.py`, `encode_grid.py`, `q_*`, `p_*`, `o_*`, `n_*`, `h3_*`, `h4_*`, `k_jobs`, `d6a`, `plasticity`, `fly_pool`, `engine_cpu`, `flymon/battle/pool.py`, any `scripts/run_*.py` other than `run_t.py`. The one exception is Reading 1: one entry in `tests/brain/test_p_spec.py`'s `MODULES`. No module global of another track is monkeypatched at runtime.
- **T's only measurement code of its own is `flymon/brain/t_measure.py`** (the reference-set and rest jobs, Reading 5). The **T measurement key** = `R_MEASURE_FILES` + `t_measure.py` + `h3_spec.py`; block z records it and jm / seal / judge re-check it (exit 7). No T file defines a copy of a shared job, measurer or cache (only `TCache` / `RReadCache`, overriding `put`), and no T file loads code by path (`importlib`, `runpy`, `exec`) — test in Task 6.
- **Numbers live only in `t_spec`.** Every T number is a field or method of `flymon.brain.t_spec.SPEC` (`TSpec`) or of `smoke(SPEC)`; S's and R's are inherited; H.3's reference set, H.4's guard thresholds and ddof, the encoder's strength seeds and Q / P numbers are read from their specs. Literal guards: `t_spec.py` may hold only Task 1's literal set; `t_rules.py` only {0, 1}; no T file holds an integer in 24_100_xxx, 24_300_xxx–24_309_xxx, 24_002_xxx, 25_000_000–25_009_999, 25_100_000–25_109_999, 23_000_000–23_009_999 or 800_000–800_199.
- **The judgement set is reached only through `t_pairs.judgement_rows` / `t_pairs.set_odours`.** Inside T, only `t_runner.build_ctx` (lazy callables), `stage_set` (the list, no measurement), `stage_gate1s` (the 55 odours, no pair), `_judgement_rows`, `stage_jm`, `stage_seal`, `_read` and `_cost` (seed counts only) name it or its seeds (AST test, Task 6). Both refuse a smoke spec.
- **Raw data of other tracks is read-only for T:** `results/summary/r_lever.json`, `results/summary/s_lever.json`, `results/r/cache/r_oracle/` (gate ③'s C and L even raw by content key; R's and S's judgement raw by their manifests after judge only), `results/p/run/cache/p_arm/` (gate ②'s OC), `results/summary/encoder_grid.json`, `results/summary/p_learning.json`, `results/summary/m0d.json`.
- **Writes are guarded.** Raw files go under `results/t/` (git-ignored by `results/*`); the summary is `results/summary/t_lever.json` (tracked). Anything else — `results/r/`, `results/s/`, `results/p/`, R's, S's or any other summary — is refused with SystemExit 2. The archive copy goes only under `~/flymon-archive/t/<seal id>/`, never over an existing directory. Every write is atomic. The fixture `tests/brain/fixtures/t_oc_cluster.json` is written once by the Task 4 implementer, never by a stage.
- **Scripts:** tests find the repository with `ROOT = Path(__file__).resolve().parents[2]`; `scripts/run_t.py` has `if __name__ == "__main__":`; pools are never started from a heredoc or `python -c`.
- **Subagents never run a real or smoke stage and never write under `results/`.** Tests write only under `tmp_path`. The controller runs every stage (section "Runs (controller)").
- **Numbers from T / T.9, verbatim:**
  - **Model:** C3 (block `h4` Params) + `apl_to_mbon05_zero` (exactly **2** CSC edges). Readout A = MBON13, P = MBON05.
  - **z rule (oracle only):** L's oracle on **z_lever** (the lever engine's reference-set z); C and E0 on block h4's z **A 10.78125 / 9.412096743243064, P 26.25 / 19.30889259728101**, which the unedited engine must reproduce bit for bit (`STOP_Z_REPRO`). Reference set = H.3's 48 odours (`default_rng(800_000)`), strength 0.35, seeds 800_100 + 2j / 800_101 + 2j, 96 presentations, settle 800 ms · read 600 steps, type counts summed over both hemispheres, mean and population SD (ddof 0). Lever guard per readout type: median(read − same-seed rest) **≥ 5** and zero share **≤ 0.25**, SD > 0 (`STOP_Z_DEGENERATE`).
  - **Conditions:** L = lever + E-grid k2-norm s **1.0**; C = no edit + E-grid k2-norm s 1.0; E0 = no edit + E0 odours at s **0.35** (record only).
  - **Set:** opponents = Gen-1 dex 1–151, no `forme`, types within the 12 codebook types (**127**); me = POOL; combos (POOL order × widened order, `excluded_combos` removed, **1,986**) permuted by `default_rng(20261004)`; HP = `l_pairs.HPS[j mod 6]`; alternate = G.11 over the widened order; rows = `e_pairs._rows_for_turn`; exclusions in order: ORN cap (`encode_grid.odour` values × max_rate_hz 200 × s 1.0 ≤ `odor_real.cap_hz` 333.33) → glomerulus collision → used keys (`used_situations` + L turns 0–209) → glomerulus-level duplicates of any used row → in-set duplicates → both opponent type sets in POOL; (b) until **21**, (a) until **43**; declared (b) last turn **17**, (a) last turn **103**, digests E0 (b) `8c9729bfba529a7bea8d783a629c1c1f54380af79ffc8032e46f657d669a0c71`, E0 (a) `e31b552636bc237f1d9fc85034432849b8a922e0d8b5fd17b1f87274b725e989`, keys `37dde1ca0c9d0bf33772b42359e45257726b46d552837ffb4cde75e8beb03c97`; skips cap **9**, collision **0**, used **164**, glom_dup **0**, in_set **48**, pool_only **57**; **55** set odours, E1 clashes with the 112 odours **0**.
  - **Gate ① supplement:** the 55 set odours × strength seeds **24_002_000+i** (i < 8), lever, per-odour KC median in **[0.03, 0.15]**; unedited values recorded.
  - **Gate ②:** P_L and P_C on **25_200_000+i** (i < 32), smoke **25_209_100–103**, c₁ from block n1 (0.608), both read on h4 z. Order: either `INVALID` → `INVALID` (one rerun after a fix); L ≠ `LEARNS_CONFIRMATORY` → `STOP_PUNISH_BROKEN`; C ≠ → `STOP_P_REFERENCE`; ℓ_L / ℓ_C < **0.5** in r1 or r2 → `STOP_PUNISH_WEAKENED`; else PASS.
  - **Gate ③:** H.4 seeds (act 500–507, select 600–607, report 608–615), 39 even pairs; C from R's even raw must give c_even **7** (`STOP_EVEN_REPRO`); L re-measured on z_lever; pass iff L testable_b **≥ 11** and c_even 7.
  - **Judgement seeds:** act **24_400_000+i**, select **24_400_100+i**, report **24_400_200+i** (i < 8). Smoke oracle seeds **24_409_000–24_409_099**.
  - **Bands (first match, R.3 = S.4):** 1 (b) ≠ 21 or (a) ≠ 43 → `NOT_READ`; 2 c ≥ 11 → B_결론없음; 3 n ≤ c → B_Tb; 4 n < 11 → B_결론없음; 5 n − c < 2 → B_결론없음; 6 G_fail_S → B_처벌가드; 7 F_a < 2 → B_Fa; 8 `SELECTED`. n and F_a on z_lever, c on h4 z.
  - **G_fail_S:** on (b) or (a), over pairs in both L and C: pun_C − pun_L ≥ **3**, pun = #(−p ≥ 2), each condition on its own z.
  - **OC:** independent (S.6) G_fail_S null **0.034 / 0.145 / 0.281 / 0.362**, harm **0.752 / 0.949 / 0.961 / 0.549**; cluster model ICC **0.3**, **10⁵** draws, seed **20261005**, rows null (0.10, 0.10), harm (0.10, 0.02) and (0.10, 0.05), the full oc_q × oc_c × naive_a grid; gate ②'s OC at true ratio 0.4 / 0.5 / 0.61 / 0.83 / 1.0 on h4 z.
  - **Reuse:** shared key **`3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc`**; R blocks repro `22c934b`, gate ① `6ad2201` (R's gate ③ `a83977f` is a record: k_even 16, c_even 7).

## Readings of the spec (decided here — the controller's rulings)

1. **`test_p_spec.py` gets one entry.** Its `test_every_spec_module_is_enumerated` fails as soon as `flymon/brain/t_spec.py` exists. Task 1 adds `"flymon/brain/t_spec.py": "flymon.brain.t_spec"` to `MODULES` (the pattern Q, R and S used). It is the only edit to a non-T file; the reviewer flags it for the user.
2. **`TSpec(LastSetSpec)`.** T subclasses S's spec, so R's records (`r_records`), R's band function, S's guard (`s_rules.g_fail_s`, `g_fail_drop` 3), S's gate ② (`s_rules.gate2`, `p_ratio_min` 0.5) and S's OC code read T's numbers through the same field names. Only what T changes is restated: the set (generator fields, declared values, skip counts, cluster table), z (`z_h4` as a literal pair checked against block h4; the guard thresholds and ddof **read** from `h4_spec`), the judgement and smoke seeds as fields, `p = P_L` / `p_c = P_C` (P's spec with `o2_seed0` 25_200_000 and `smoke_seed0` 25_209_000; P_L also replaces `o1_conditions[0]` with the lever), the cluster OC fields, the reuse fields (`r_reused` = repro and gate ① only), `r_cache_dir` (R's) and `s_summary` (S's), and T's paths. **Every seed-bearing field S declared is overridden** (test), so no S block is reused by name. The class is not named `RSpec` (test_p_spec's `_expand` special-cases that name).
3. **The store (T.8).** `r_store.py` is in the shared key and hardcodes `results/r/`; it cannot be edited. Decision (S's Reading 3): **`t_store.TCache` subclasses `RCache` and overrides only `put`** (through `t_store.write_json`, whose guard allows `results/t/` and `t_lever.json` only) and takes T's smoke seeds as the default scope. Gate ③ reads R's even raw through **`t_store.RReadCache`**, an `RCache` over R's root whose `put` refuses: the entries are found by R's own key formula over the same shared code key (test on the real cache, Task 3), and nothing is ever written into R's tree.
4. **One `RMeasurer` per z.** `RMeasurer` holds one z and puts it into every oracle job's kwargs and cache inputs, and `r_measure.py` cannot change. So the runner takes a factory `measure(z)` (the CLI builds one `RMeasurer` per z over one pool and one `TCache`): L's oracle — gate ③'s even pairs, `jm:L`, smoke's L — runs on `measure(z_lever)` (z_lever read from block z), C's and E0's on `measure(h4 z)`; P arms and KC activity, where z plays no part, use the h4 measurer. The cache inputs then differ by z (as they should: the α selection depends on it), and the seal's stored-inputs check compares each condition's raw entries with **its own** measurer's inputs, so a `jm:L` measured on the wrong z is `NOT_READ` (mutation test). Rejected: one measurer with a mutable z (an interrupted run could mix z inside one condition) and a T copy of `RMeasurer` (a second oracle path).
5. **T's measurement file.** No shared job measures the reference set on the lever: `h3_jobs.reference_job` / `rest_job` know only H.3's edits, and `q_jobs.q_rig` builds Plasticity. `t_measure.py` therefore copies `reference_job` / `rest_job`'s call sequence (reset / clear_drive / present / run(settle) / step × read) onto an engine built as `h3_jobs.engine_for` builds it (`Engine(conn, pops, params, seed=0)`, no plasticity) and edited in place by `q_jobs.apply_q_edit` (the lever's own edit, unchanged). With edit "none" its counts are `reference_job`'s (test on the real engine, Task 3), and in the scratch run over the whole reference set they gave block h4's z bit for bit. The z stage **does not cache** (T.9.6 "캐시 없이"; 2 × 192 presentations take about a minute on 16 workers): it measures the unedited engine first, decides `STOP_Z_REPRO` (bit for bit = Python float equality of mean and SD with both block h4's z and `t_spec.z_h4`), then measures the lever only after a pass and decides `STOP_Z_DEGENERATE`; the rows go once to `results/t/z.json` with the T measurement key. Wrong edges (none 0 / lever 2), a CSC other than R's unedited sha for "none", or presentation counts other than 96 are `INVALID` (exit 5) — a defect signal, not in the spec's STOP list.
6. **The set (T.2 / T.9.1).** `t_pairs.t_set` first regenerates S's set (and through it R's) and checks their declared digests (T.2's "같은 코드로 R 세트·S 세트 digest가 먼저 재현될 것"), then follows the declared order row by row; a row failing a check is counted under the first failing reason, and rows of a full axis are dropped uncounted. The ORN cap needs the odour, and an odour with a glomerulus collision cannot be built, so **an unencodable odour is not a cap failure** — it falls to the collision check (no difference: collision 0). Skip counts therefore include rows of turns after the (b) axis is full while (a) still fills — that is how the declared counts arise. **T (〈T_a〉 in the sentences) = 103**, the last (a) turn; the (b) axis ends at turn 17. `STOP_SET_SHORT` uses T.2's sentence with 〈b〉·〈a〉.
7. **Gate ① supplement (T.9.1)** is stage `gate1s`, after `z` and before `smoke`: R's `kc_activity_job` through `RMeasurer.activity` on the T set's 55 distinct E-grid odours × `kc_seeds()` (encoder ③'s 24_002_000+i), under the lever (judged) and unedited (recorded). Each odour's median over seeds must lie in `valid_band`. The spec's sentence is elided ("…T 세트 냄새 〈걸린 냄새와 값〉…"); T uses R's `STOP_STRENGTH_LEVER` sentence with the condition "T 세트 냄새 〈odour value; …〉 ∉ [0.03, 0.15]". Wrong edges or odour / seed counts are `INVALID`. This stage reads the set's odours (no pair, no oracle) — the one pre-judgement use of the set, required by T.9.1.
8. **A different shared key is a STOP for the user, not a re-measurement** (S's Reading 4). Stage `reuse` records `STOP_REUSE` (exit 3); every later stage re-runs the check and refuses with **exit 7**. T reuses only R's repro and gate ① (T.3 1); R's gate ③ values (16 / 7) are records — T re-measures gate ③ (T.9.2).
9. **Bands are R's function, the guard and gate ② are S's.** `t_rules.read_band is r_rules.read_band`, `t_rules.g_fail_s is s_rules.g_fail_s`, `t_rules.gate2 is s_rules.gate2` (its sentences are S.3's, the ones T.7 names). The exhaustive fixture covers n, c ∈ 0..21 × F_a ∈ 0..43 × G_fail_S.
10. **Smoke (T.3 4)** checks what T runs that S's smoke did not: P_L / P_C on 25_209_1xx, the oracle on even (b) pairs 0 and 20 with L on z_lever and C / E0 on h4 z (a problem when a condition's measurer carries another z), edges 2 / 0, L ≠ C sha, C = E0 = R's unedited sha. The widened-pool rows' odour computation (E-grid and E0) is checked at list level (stage `set` and Task 2's tests: every judgement row carries non-empty E-grid and E0 odours) and **not measured**: every widened-pool row is either a set row or a candidate for a future declaration. The cost estimate adds gate ③'s L.
11. **The operating characteristics (T.6, T.9.6, T.9.7).** Stage `oc` writes the independent model (`s_rules.oc` on T's spec — S.6's values on n_b 21 · n_a 43 — with T's notes) and the cluster model. Cluster-model choices the spec leaves open: the guard's per-cluster multinomial is drawn as two binomials (pass→fail, then fail→pass among the rest); the testable model draws n from the 9 (b) clusters and F_a from naive_a pairs taken as a uniformly random subset of the 43 (a) pairs (each in its cluster); G_fail_S stays independent of n and F_a with the cluster model's probability (as in the independent model); c is fixed; one `default_rng(20261005)` is consumed in a fixed order (guard rows, then per q the (b) and (a) draws). The q grid is `oc_q` (0.33 / 0.5 / 0.6 / 0.7: T.9.7's "독립 모형의 testable q 그리드(oc_q)" wins over T.6's "0.33·0.55"). The Task 4 implementer writes the fixture once; the stage recomputes the table, refuses unless it equals the fixture (canonical JSON), and records the fixture's sha256 and the table's; the rows live in the fixture, the block holds the g rows and the n distributions. Gate ②'s OC is S's (`s_records.gate2_oc`) on h4 z (T.9.3). Values computed while planning: cluster G_fail_S null 0.401 (independent 0.281), harm 0.685 (0.752) and 0.571 (0.549); P(n ≥ 11) at q 0.33 / 0.5 / 0.6 / 0.7 = 0.097 / 0.498 / 0.776 / 0.939 (independent 0.052 / 0.500 / 0.826 / 0.974).
12. **Gate ② (T.3 6, T.9.3)** is S's stage on 25_200_xxx with both P conditions read on h4 z; the block adds `p_L_on_z_lever` (P_L's `p_judge` on z_lever and ℓ_L(z_lever) / ℓ_C(h4 z) per direction) as a record. An INVALID gate ② reruns once (`--rerun-after-invalid`) only with a changed pipeline key.
13. **Gate ③ (T.9.2)** is one stage: R's even raw for C **and** L is read through `RReadCache` with h4 z (a missing entry refuses, exit 2 — the replaced T.3 7's "재계산" path is not built, since T.9.2 says reuse); C must give c_even 7 with no reasons, else `STOP_EVEN_REPRO` (sentence "(L 〈R's L raw on h4 z〉, C 〈·〉)") and **L is not measured**; then L's 39 even pairs on `measure(z_lever)` (cached in `results/t/cache`, resumable) and `r_rules.gate3` (validity, `STOP_C_EVEN_MISMATCH` — unreachable after the reproduction, kept defensive — and `STOP_EVEN_LOW_LEVER` with R.9.3's sentence). `k_even` in SELECTED = this L testable_b on z_lever.
14. **Keys.** Every block carries `code_key` = the shared measurement key (which keys the cache), `t_measure_key` and `pipeline_key` = a hash of T's files (`t_spec`, `t_pairs`, `t_store`, `t_records`, `t_rules`, `t_runner`, `run_t.py`). `jm:*`, `seal` and `judge` refuse unless every earlier block carries the current shared key and was written with no dirty hashed file, when block z's T measurement key differs from the current one (exit 7, T.9.6), and when the summary's git history already holds a `judge` block. The **decision code** = every T hashed file outside the T measurement key (T's files, S's and R's decision files, the encoder / H.4 / L / Q pair and rule files, `battle/pool.py`, the cluster fixture, `m0d.json`, `r_lever.json`, `s_lever.json`); `seal` pins its hash and `judge` reads only under it.
15. **Read once, re-generate once (T.5 = S.9.2)** with `results/t/judge_read.json`, `judge_reread.json`, `judge_done.json`, exactly as S.
16. **Sentences (T.7 as replaced by T.9.5 / T.9.7).** 〈…〉 → `{field}`; B_결론없음 = R.3's line + `" ({n}/21 대 {c}/21, F_a {f_a}/43)"`; B_Tb's consequence is T.9.5's ("→ 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 결과는 R·S 그대로)."); B_처벌가드 and B_Fa are S.7's (n_a 43). SELECTED is T.7's text with "8개" read as "9개" (T.9.1), T.9.5's sentence "이 판정은 R(뒤에 가드 변경)·S(뒤에 z 규칙)에 이은 같은 지렛대의 세 번째 판정이다." placed after "(T.0)." and T.9.5's consequence appended last ("귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다."). `STOP_Z_REPRO`, `STOP_Z_DEGENERATE`, `STOP_EVEN_REPRO` are T.1's, T.9.4's and T.3 7's sentences (with several failing types joined by "·"); `STOP_REUSE` is new (Reading 8).
17. **Records at judge (T.6):** `r_records.compare` (C → L and E0 → L transitions, per-condition KC, APL, edges, CSC, saturation, naive values, α choices, z renormalisation), G_fail_S per axis, the cluster table (testable / reward / punish passes per T.9.1 cluster and condition), the **α-fixed sensitivity** (L's raw read on h4 z: n, F_a, naive_a, G_fail_S, the band it would give — "판정 아님"), z_lever and its SD ratios, gate ②'s labels and ratios, both OC shas, k_even.
18. **R · S re-read (T.9.6)** is stage `rs_reread`, allowed only after `judge` (and refused once written): R's and S's `jm:L` / `jm:C` raw through their manifests (sha checked), L on z_lever (α as chosen on h4 z when they were measured), C on h4 z, S's guard; R's and S's bands stand. A record.
19. **Exit codes of `scripts/run_t.py`:** 0 PASS / SEALED / READ / a record stage; 2 a refusal; 3 a STOP (`STOP_REUSE`, `STOP_SET_SHORT`, `STOP_Z_REPRO`, `STOP_Z_DEGENERATE`, `STOP_STRENGTH_LEVER`, gate ②'s three, `STOP_EVEN_REPRO`, `STOP_EVEN_LOW_LEVER`, `STOP_C_EVEN_MISMATCH`); 5 a gate `INVALID` (z, gate1s, gate2, gate3); 6 seal not SEALED, judge NOT_READ, or smoke with problems; 7 R's reuse condition or the T measurement key broke.
20. **Recovery (T.5 = S.5):** `recompute --note` reruns the arithmetic from the sealed files (appends to block `recompute`); `invalid_run --note` writes `INVALID_RUN` once. A replacement set is a new declaration (T.5).

## Review Focus

1. **The z stage stopping or passing for the wrong reason** — the unedited engine's z one ulp away from block h4's, or the lever's MBON13 at median Δ 4 / zero share 25/96 / SD 0. Expect `STOP_Z_REPRO` before any lever measurement, `STOP_Z_DEGENERATE` naming the failing type(s) at exactly H.4's boundaries, and no `z_lever` in either block. Tests: Task 4 (`test_z_repro_needs_every_bit_of_h4s_z`, `test_z_lever_guard_boundaries`), Task 5 (`test_z_repro_stop_leaves_the_lever_unmeasured`, `test_z_degenerate_stops`).
2. **A condition measured on the other condition's z** (L's raw on h4 z, or C's on z_lever), by a code path or an interrupted run. Expect the seal's stored-inputs check to read `NOT_READ`. Tests: Task 3 (`test_one_measurer_per_z_over_one_cache`), Task 6 (`test_mutation_l_raw_on_h4_z_fails_the_preread_validity`, `test_real_cache_round_trip_seals_and_reads`).
3. **R's even raw not reproducing or not present** (another R cache, a pruned `results/r/`). Expect `STOP_EVEN_REPRO` with L unmeasured, or a refusal (exit 2) with nothing written — never a measurement into R's tree. Tests: Task 3 (`test_rreadcache_reads_rs_root_and_never_writes`, `test_rs_even_raw_is_found_by_content_key`), Task 5 (`test_gate3_even_repro_stop_leaves_l_unmeasured`, `test_gate3_refuses_when_rs_even_raw_is_missing`).
4. **The shared key or the T measurement key drifts between stages** (another track edits a shared file; `t_measure.py` or `h3_spec.py` changes after block z). Expect every later stage, and jm / seal / judge for the T key, to refuse with exit 7 and write nothing. Tests: Task 5 (`test_reuse_broken_after_reuse`), Task 6 (`test_keys_are_rechecked_before_measurement_seal_and_judge`).
5. **The T set generated differently** (a changed exclusion order, the ORN cap missing, R's or S's set not reproducing, turns running out). Expect `set` to refuse (or record `STOP_SET_SHORT`) and `jm` to refuse on a tampered set. Tests: Task 2 (`test_a_lower_cap_drops_the_rows_that_exceed_it`, `test_s_and_r_sets_must_reproduce_first`, `test_stop_set_short_when_the_turns_run_out`), Task 5 (`test_set_short_and_mismatch`), Task 6 (`test_mutation_a_tampered_set_refuses`).

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/t_spec.py` (create) | `TSpec(LastSetSpec)`, `P_L`, `P_C`, `SPEC`, `smoke()` |
| `flymon/brain/t_pairs.py` (create) | `STOP_SET_SHORT`, `OK`, `REASONS`, `opponents`, `cluster_of`, `pool_type_sets`, `t_set`, `check_t_set`, `summary`, `judgement_rows`, `set_odours`, `clusters` |
| `flymon/brain/t_measure.py` (create — T's measurement file) | `T_MEASURE_FILES`, `t_measure_key`, `z_engine`, `t_ref_job`, `t_rest_job`, `ZMeasurer` |
| `flymon/brain/t_store.py` (create) | `ALLOWED_DIR`, `SUMMARY`, `SMOKE_SEEDS`, `guard`, `write_bytes`, `write_json`, `read_summary`, `write_summary_block`, `TCache`, `RReadCache`; re-exports `load_manifest`, `archive_copy` |
| `flymon/brain/t_rules.py` (create) | outcome / band constants, `read_band` (R's), `g_fail_s` / `gate2` (S's), `reuse`, `set_outcome`, `z_repro`, `z_lever`, `gate1s`, `gate3`, `SENTENCES`, `sentence`, `OC_NOTES`, `oc`, `oc_cluster` |
| `flymon/brain/t_records.py` (create) | `z_side`, `z_ratios`, `gate1s_record`, `p_zlever`, `clusters`, `alpha_fixed`; `ratio` / `gate2_oc` (S's) |
| `flymon/brain/t_runner.py` (create) | `ORDER`, `GATES`, `T_PIPELINE_FILES`, `T_HASHED_FILES`, `DECISION_FILES`, markers, `pipeline_key`, `decision_key`, `build_ctx`, `Runner` |
| `scripts/run_t.py` (create) | the CLI, `exit_code`, `__main__` guard |
| `tests/brain/test_t_spec.py`, `test_t_pairs.py`, `test_t_measure.py`, `test_t_store.py`, `test_t_rules.py`, `test_t_records.py`, `test_t_runner.py`, `test_t_judge.py`, `t_fixtures.py`, `t_world.py` (create) | tests and fixtures (`r_fixtures.py`, `s_fixtures.py` are reused as they are) |
| `tests/brain/fixtures/t_oc_cluster.json` (create, generated in Task 4) | the cluster OC table (T.9.7) |
| `tests/brain/test_p_spec.py` (modify: one `MODULES` entry) | Reading 1 |

---

### Task 1: `t_spec` — numbers, the two gate-② P specs, seed collisions, literal guards, the shared key

**Files:**
- Create: `flymon/brain/t_spec.py`
- Modify: `tests/brain/test_p_spec.py` (one `MODULES` entry)
- Test: `tests/brain/test_t_spec.py`

**Interfaces:**
- Consumes: `s_spec.LastSetSpec` / `SPEC`, `r_spec.LEVER` / `SPEC`, `p_spec.SPEC` / `PSpec` / `smoke`, `q_jobs.MBON05`, `h4_spec.SPEC`.
- Produces: `P_L`, `P_C` (PSpec); `TSpec` with the new or overridden fields `opp_num_max, n_opp, n_combos, set_rng_seed, first_turn, set_last_turn, n_a, last_turn_b, last_turn, digest_e0_b, digest_e0_a, digest_keys, skipped_declared, n_set_odours, clusters_b, clusters_a, z_h4, z_guard_med_min, z_guard_zero_max, z_ddof, judge_act_seeds, judge_select_seeds, judge_report_seeds, smoke_seeds, p, p_c, r_even_L_h4, oc_cluster_seed, oc_cluster_icc, oc_cluster_draws, oc_cluster_g, oc_cluster_fixture, r_reused, r_commits, r_cache_dir, s_summary, cache_dir, smoke_cache_dir, smoke_detail, z_detail, summary, archive_root` and every inherited field / method (`conditions()`, `cond()`, `kc_seeds()`, `h4_seeds()`, `even_seeds()`, …); `z_h4_dict() -> {"A": (mean, sd), "P": (mean, sd)}`; `judge_seeds() -> dict` (ValueError when smoke); `SPEC`; `smoke(spec) -> TSpec` (p, p_c → P smoke, `smoke=True`, workers 4).

- [ ] **Step 1: Add T to the spec-module list (Reading 1)**

In `tests/brain/test_p_spec.py`, the `MODULES` dict's last line is:

```python
           "flymon/brain/r_spec.py": "flymon.brain.r_spec", "flymon/brain/s_spec.py": "flymon.brain.s_spec"}
```

Replace it with (one entry added; nothing else in the file changes):

```python
           "flymon/brain/r_spec.py": "flymon.brain.r_spec", "flymon/brain/s_spec.py": "flymon.brain.s_spec",
           "flymon/brain/t_spec.py": "flymon.brain.t_spec"}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/brain/test_t_spec.py
"""Spec T.1 / T.2 (T.9.1) / T.3 / T.8 / T.9.6: every T number in t_spec; T's new blocks (judgement 24_400_xxx, smoke
24_409_xxx, gate ② 25_200_000+i and its smoke 25_209_xxx, their training seeds, the set generator's 20261004 and the
cluster OC's 20261005) collide with no declared seed of any other spec module (test_p_spec's collector, R and S
included); the two gate-② P specs differ from each other in the one edit only and from P in their seed blocks; every
seed field S declared is overridden; z_h4 is block h4's C3 z and the guard thresholds are H.4's; no T file holds
another track's block as a literal; the shared measurement key over R_MEASURE_FILES is still R's 3c2699c7…."""
import ast
import dataclasses
import importlib
import json
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.h4_spec import SPEC as H4
from flymon.brain.p_spec import SPEC as P
from flymon.brain.r_spec import SPEC as R
from flymon.brain.s_spec import SPEC as S
from flymon.brain.s_spec import LastSetSpec
from flymon.brain.t_spec import P_C, P_L, SPEC, TSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_t() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/t_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


T_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds)
           | {SPEC.set_rng_seed, SPEC.oc_cluster_seed})
T_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_t_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/t_spec.py") == "flymon.brain.t_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_400_000, 24_400_008)),
                                      select=list(range(24_400_100, 24_400_108)),
                                      report=list(range(24_400_200, 24_400_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_200_000, 25_200_032))
    assert len(SPEC.p.seeds) == P.o.o2_n_seeds
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_209_000 <= s < 25_210_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_409_000, 24_409_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    assert (SPEC.set_rng_seed, SPEC.oc_cluster_seed) == (20261004, 20261005)
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """T.9.6 (P3-11): T's blocks, the generator and OC seeds and the derived training seeds against every other
    declared block (S, R, P, Q, the encoder, H.4 ... d6a)."""
    declared = _declared_without_t()
    for s in (500, 615, 24_002_000, 24_100_000, 24_300_000, 24_309_000, 25_000_000, 25_100_000, 25_109_000, 23_000_000,
              24_000_000, 22_000_000):
        assert s in declared, s
    assert not T_SEEDS & declared
    assert not {s for s in T_SEEDS if (s - BASE) // STRIDE in declared}
    assert not T_TRAIN & declared
    assert not {s for s in T_TRAIN if (s - BASE) // STRIDE in declared}
    assert not T_TRAIN & T_SEEDS


def test_every_s_seed_field_is_overridden():
    s_seed_fields = {f.name for f in dataclasses.fields(LastSetSpec) if "seed" in f.name} | {"p", "p_c"}
    for name in s_seed_fields:
        assert getattr(SPEC, name) != getattr(S, name), name
    fields = _module_seeds("flymon.brain.t_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & set(R.p.seeds) and not fields & set(range(500, 616))
    assert not fields & (set(S.judge_act_seeds) | set(S.smoke_seeds) | set(S.p.seeds))


def test_gate2_specs_differ_in_the_one_edit_only():
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
    assert isinstance(SPEC, LastSetSpec) and isinstance(SPEC, TSpec)
    assert (SPEC.opp_num_max, SPEC.n_opp, SPEC.n_combos, SPEC.first_turn, SPEC.set_last_turn) == (151, 127, 1986, 0,
                                                                                                    1985)
    assert (SPEC.n_b, SPEC.n_a, SPEC.last_turn_b, SPEC.last_turn, SPEC.n_set_odours) == (21, 43, 17, 103, 55)
    assert SPEC.digest_e0_b == "8c9729bfba529a7bea8d783a629c1c1f54380af79ffc8032e46f657d669a0c71"
    assert SPEC.digest_e0_a == "e31b552636bc237f1d9fc85034432849b8a922e0d8b5fd17b1f87274b725e989"
    assert SPEC.digest_keys == "37dde1ca0c9d0bf33772b42359e45257726b46d552837ffb4cde75e8beb03c97"
    assert dict(SPEC.skipped_declared) == dict(cap=9, collision=0, used=164, glom_dup=0, in_set=48, pool_only=57)
    assert sum(n for *_, n in SPEC.clusters_b) == 21 and len(SPEC.clusters_b) == 9
    assert sum(n for _, n in SPEC.clusters_a) == 43 and len(SPEC.clusters_a) == 8
    assert SPEC.z_h4_dict() == {"A": (10.78125, 9.412096743243064), "P": (26.25, 19.30889259728101)}
    assert (SPEC.z_guard_med_min, SPEC.z_guard_zero_max, SPEC.z_ddof) == (5.0, 0.25, 0) == (
        H4.react_med_delta_min, H4.react_zero_share_max, H4.z_ddof)
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.testable_min, SPEC.naive_max) == (11, 2, 2, 2.0, 0.5)
    assert (SPEC.g_fail_drop, SPEC.g_fail_flips, SPEC.p_ratio_min, SPEC.c_even_expected, SPEC.r_even_L_h4) == (
        3, None, 0.5, 7, 16)
    assert SPEC.valid_band == (0.03, 0.15) and SPEC.kc_seeds() == tuple(E.strength_seeds)
    assert SPEC.gate2_oc_rhos == (0.4, 0.5, 0.61, 0.83, 1.0) and SPEC.oc_harm == S.oc_harm
    assert (SPEC.oc_cluster_icc, SPEC.oc_cluster_draws) == (0.3, 100_000)
    assert SPEC.oc_cluster_g == (("null", 0.1, 0.1), ("harm", 0.1, 0.02), ("harm", 0.1, 0.05))
    assert SPEC.r_shared_key == "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    assert (SPEC.r_summary, SPEC.r_cache_dir, SPEC.s_summary) == ("results/summary/r_lever.json", "results/r/cache",
                                                                  "results/summary/s_lever.json")
    assert SPEC.r_reused == ("repro", "gate1")
    assert (SPEC.summary, SPEC.cache_dir, SPEC.archive_root, SPEC.z_detail) == (
        "results/summary/t_lever.json", "results/t/cache", "~/flymon-archive/t", "results/t/z.json")
    assert SPEC.conditions() == R.conditions() and SPEC.cond_names == ("L", "C", "E0")
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "lever_edges",
              "config", "strength", "e0_strength", "oc_q", "oc_c", "oc_naive_max", "sat_fracs", "smoke_pairs"):
        assert getattr(SPEC, f) == getattr(R, f), f


def test_z_h4_is_block_h4s_c3_z():
    from flymon.agent.config import load_c3_config
    assert load_c3_config(SPEC.m0d_summary).z == SPEC.z_h4_dict()
    h4 = json.loads((ROOT / SPEC.m0d_summary).read_text())["h4"]["h4"]["combos"]["C3"]["z"]
    assert {k: tuple(v) for k, v in h4.items()} == SPEC.z_h4_dict()


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(TSpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4 and SM.p.o.on_edit == SPEC.lever_edit and SM.p_c.o.on_edit == SPEC.no_edit


ALLOWED_T_SPEC = {151, 127, 1986, 20261004, 0, 1985, 43, 17, 103, 9, 164, 48, 57, 55, 1, 3, 4, 7, 8, 2, 5,
                  10.78125, 9.412096743243064, 26.25, 19.30889259728101, 24_400_000, 24_400_008, 24_400_100,
                  24_400_108, 24_400_200, 24_400_208, 24_409_000, 24_409_100, 16, 20261005, 0.3, 100_000, 0.1, 0.02,
                  0.05, 25_200_000, 25_209_000}


def test_t_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/t_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_T_SPEC, nums - ALLOWED_T_SPEC


def test_no_t_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("t_*.py")) + sorted((ROOT / "scripts").glob("run_t*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_300_000 <= i < 24_310_000
               or 24_002_000 <= i < 24_003_000 or 25_000_000 <= i < 25_010_000 or 25_100_000 <= i < 25_110_000
               or 23_000_000 <= i < 23_010_000 or 800_000 <= i < 800_200}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_measurement_key_is_still_rs():
    """T.3 1 / T.8: the key over r_measure.R_MEASURE_FILES equals the key every R block carries; T edits none."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_t_spec.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'flymon.brain.t_spec'` (and `test_p_spec.py::test_every_spec_module_is_enumerated` fails until the module exists).

- [ ] **Step 4: Write the implementation**

```python
"""Every number of spec appendix T as amended by T.9 (T.9 wins over T.0-T.8, T.9.7 lists the replacements): the same
lever (APL->MBON05 removal), each engine variant's own reference-set z for the oracle, one M2 judgement on a new set
drawn from the widened Gen-1 opponent pool.
- TSpec subclasses S's LastSetSpec (itself R's LeverSpec), so R's records and bands (r_records, r_rules.read_band) and
  S's guard, gate ② and OC code (s_rules, s_records) read T's numbers through the same field names; only what T
  changes is restated here (plan Reading 2). Every seed-bearing field S declared is overridden, so no S block is
  reused by name.
- T's new seed blocks are FIELDS (judgement 24_400_xxx, smoke 24_409_xxx, gate ② 25_200_000+i and smoke 25_209_xxx in
  the nested P specs, the set generator's 20261004 and the cluster OC's 20261005), so every collision collector sees
  them (T.9.6). Gate ① supplement's strength seeds stay R's kc_seeds() method (encoder ③'s block, T.9.1).
- z (T.1, T.9.2-T.9.4): z_h4 is block h4's C3 z, which the unedited engine must reproduce bit for bit; the readout
  guard's thresholds and z's ddof are H.4's (read from h4_spec, never restated); the reference set is H.3's."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .h4_spec import SPEC as H4
from .q_jobs import MBON05 as LEVER_EDIT
from .r_spec import LEVER
from .r_spec import SPEC as R_SPEC
from .s_spec import SPEC as S_SPEC
from .s_spec import LastSetSpec

_T_O = dict(o2_seed0=25_200_000, smoke_seed0=25_209_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_T_O))
P_L = dataclasses.replace(P_SPEC, o=dataclasses.replace(
    P_SPEC.o, **_T_O, o1_conditions=((LEVER, LEVER_EDIT),) + P_SPEC.o.o1_conditions[1:]))


@dataclass(frozen=True)
class TSpec(LastSetSpec):
    # ---- the set (T.2 as replaced by T.9.1) -------------------------------------------------------------------------
    opp_num_max: int = 151                          # Gen-1 dex numbers 1..151, base forms (no `forme`)
    n_opp: int = 127                                # opponents whose types all lie in the codebook's 12 types
    n_combos: int = 1986
    set_rng_seed: int = 20261004
    first_turn: int = 0
    set_last_turn: int = 1985                       # the permutation's end (n_combos - 1)
    n_a: int = 43
    last_turn_b: int = 17                           # the 21st (b) row's turn
    last_turn: int = 103                            # T: the 43rd (a) row's turn (T_a in the sentences)
    digest_e0_b: str = "8c9729bfba529a7bea8d783a629c1c1f54380af79ffc8032e46f657d669a0c71"
    digest_e0_a: str = "e31b552636bc237f1d9fc85034432849b8a922e0d8b5fd17b1f87274b725e989"
    digest_keys: str = "37dde1ca0c9d0bf33772b42359e45257726b46d552837ffb4cde75e8beb03c97"
    skipped_declared: tuple = (("cap", 9), ("collision", 0), ("used", 164), ("glom_dup", 0), ("in_set", 48),
                               ("pool_only", 57))
    n_set_odours: int = 55
    clusters_b: tuple = (("FIRE*", "WATER", 1), ("GRASS+POISON", "FIRE*", 3), ("GRASS+POISON", "GROUND*", 3),
                         ("GROUND*", "NORMAL", 3), ("GROUND*", "POISON*", 3), ("NORMAL", "ROCK+WATER*", 1),
                         ("POISON*", "NORMAL", 1), ("ROCK+WATER*", "NORMAL", 2), ("WATER", "FLYING+NORMAL*", 4))
    clusters_a: tuple = (("FIRE*", 7), ("FLYING+NORMAL*", 8), ("FLYING+POISON*", 2), ("FLYING+WATER*", 1),
                         ("GRASS*", 5), ("GROUND*", 9), ("POISON*", 9), ("ROCK+WATER*", 2))
    # ---- z (T.1, T.9.2-T.9.4) ----------------------------------------------------------------------------------------
    z_h4: tuple = (("A", (10.78125, 9.412096743243064)), ("P", (26.25, 19.30889259728101)))
    z_guard_med_min: float = H4.react_med_delta_min        # 5: median(read − same-seed rest) per readout type
    z_guard_zero_max: float = H4.react_zero_share_max      # 0.25: share of zero read counts
    z_ddof: int = H4.z_ddof                                # 0: population SD
    # ---- the judgement seeds (T.3 8) and T's smoke (T.3 4) -----------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_400_000, 24_400_008))
    judge_select_seeds: tuple = tuple(range(24_400_100, 24_400_108))
    judge_report_seeds: tuple = tuple(range(24_400_200, 24_400_208))
    smoke_seeds: tuple = tuple(range(24_409_000, 24_409_100))
    # ---- gate ② (T.3 6, T.9.3) ---------------------------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    # ---- gate ③ (T.9.2) ----------------------------------------------------------------------------------------------
    r_even_L_h4: int = 16                         # R's gate ③ L testable_b on h4 z (a record beside STOP_EVEN_REPRO)
    # ---- the cluster-correlated OC (T.6, T.9.6, T.9.7) ---------------------------------------------------------------
    oc_cluster_seed: int = 20261005
    oc_cluster_icc: float = 0.3
    oc_cluster_draws: int = 100_000
    oc_cluster_g: tuple = (("null", 0.1, 0.1), ("harm", 0.1, 0.02), ("harm", 0.1, 0.05))
    oc_cluster_fixture: str = "tests/brain/fixtures/t_oc_cluster.json"
    # ---- R's reused gates (T.3 1) ------------------------------------------------------------------------------------
    r_reused: tuple = ("repro", "gate1")
    r_commits: tuple = (("repro", "22c934b"), ("gate1", "6ad2201"), ("gate3", "a83977f"))
    r_cache_dir: str = R_SPEC.cache_dir           # R's even raw (gate ③'s C), read only, by content key
    s_summary: str = S_SPEC.summary               # S's judgement raw, read only after judge (T.9.6)
    # ---- paths -------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/t/cache"
    smoke_cache_dir: str = "results/t/smoke/cache"
    smoke_detail: str = "results/t/smoke.json"
    z_detail: str = "results/t/z.json"
    summary: str = "results/summary/t_lever.json"
    archive_root: str = "~/flymon-archive/t"

    def z_h4_dict(self) -> dict:
        return {k: (float(v[0]), float(v[1])) for k, v in self.z_h4}

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (T.3 4)")
        return dict(act=list(self.judge_act_seeds), select=list(self.judge_select_seeds),
                    report=list(self.judge_report_seeds))


SPEC = TSpec()


def smoke(spec: TSpec = SPEC) -> TSpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_209_100+), T's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_t_spec.py tests/brain/test_p_spec.py tests/brain/test_s_spec.py tests/brain/test_r_spec.py tests/agent/test_e_spec_store.py -q`
Expected: all pass (11 in `test_t_spec.py`; `test_shared_measurement_key_is_still_rs` needs `data/malecns.npz`).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/t_spec.py tests/brain/test_t_spec.py tests/brain/test_p_spec.py
git commit -m "feat(t): t_spec — T numbers on S's LastSetSpec, P_L / P_C gate-2 specs on 25_200_xxx, seed collisions, shared key test"
```

---

### Task 2: `t_pairs` — the widened-pool set (list only), its declaration check, rows, odours, clusters

**Files:**
- Create: `flymon/brain/t_pairs.py`
- Test: `tests/brain/test_t_pairs.py`

**Interfaces:**
- Consumes: `e_pairs` (`used_situations`, `egrid_key`, `_rows_for_turn`, `_e0_digest`, `attach_odours`), `encode_grid` (`glomeruli`, `odour`, `reachable`), `h4_pairs.pool_vocabulary` / `e0_channels`, `l_pairs.new_turns` / `alternate_from` / `excluded_combos` / `HPS`, `q_pairs.codebook`, `odor_real.cap_hz`, `s_pairs.s_set` / `check_s_set`, `r_pairs.okey` / `row_key`, `battle.pool.POOL`, poke_env's Gen-1 pokedex, Task 1's spec.
- Produces: `STOP_SET_SHORT`, `OK`, `REASONS`; `opponents(mon_types, spec) -> [(name, types)]`; `pool_type_sets() -> set`; `cluster_of(row, pool_sets) -> tuple`; `t_set(pops, enc, spec, params) -> dict(b, a, n_b, n_a, last_turn, last_turn_b, last_turn_a, status, skipped, n_opp, n_combos, n_odours, e1_clashes, cap_fails, all_off_pool, clusters_b, clusters_a, digest_e0_b, digest_e0_a, digest_keys)` (ValueError if S's set does not reproduce); `check_t_set(js, spec) -> list[str]`; `summary(js) -> dict`; `judgement_rows(pops, rc, enc, spec, params) -> list` ((b) 21 then (a) 43 rows with odours; ValueError on mismatch or smoke); `set_odours(pops, rc, enc, spec, params) -> {odour id: odour}` (55; same refusals); `clusters(rows) -> {pair key: label}` ("X 대 Y" on (b), the type set on (a), `*` = off POOL).

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_t_pairs.py
"""T.2 / T.9.1: the T set generated from the widened Gen-1 opponent pool equals the declared set ((b) 21 last turn 17,
(a) 43 last turn 103, three digests, skip counts cap 9 · collision 0 · used 164 · glom_dup 0 · in_set 48 · pool_only
57, the cluster table, 55 odours); its keys meet no used key (H.4 turns, L turns 0-209 — R's and S's sets included) and
its glomerulus-set pairs no used row's; no key repeats inside it; every row holds an off-POOL opponent type set; no set
odour clashes with the 112 reachable odours (E1) or exceeds the ORN cap; a row whose odour exceeds the cap is dropped;
S's (and R's) set must reproduce first; STOP_SET_SHORT when the turns run out; a mismatch or a smoke spec refuses
judgement_rows / set_odours."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.agent import e_pairs, encode_grid
from flymon.agent.e_spec import SPEC as E
from flymon.brain import l_pairs, odor_real, q_pairs
from flymon.brain import t_pairs as TP
from flymon.brain.t_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
ENC = json.loads((ROOT / "results/summary/encoder_grid.json").read_text())
NPZ = ROOT / "data/malecns.npz"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="no connectome")


@pytest.fixture(scope="module")
def world():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    return pops, {str(t): len(v) for t, v in pops.receptor_types.items()}, load_c3_config(SPEC.m0d_summary).params


@pytest.fixture(scope="module")
def js(world):
    return TP.t_set(world[0], ENC, SPEC, world[2])


def _used_rows(pops):
    from flymon.brain import h4_pairs
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    rows = list(e_pairs.used_situations(pops, E))
    for t in l_pairs.new_turns(st, mi, 210, E.l_rng_seed):
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        rows += e_pairs._rows_for_turn(pops, chan, t, alt, st, with_a=True)
    return rows


@needs_npz
def test_the_set_equals_the_declaration(js):
    assert TP.check_t_set(js, SPEC) == []
    assert (js["status"], js["n_b"], js["n_a"], js["last_turn_b"], js["last_turn"]) == ("OK", 21, 43, 17, 103)
    assert js["skipped"] == dict(cap=9, collision=0, used=164, glom_dup=0, in_set=48, pool_only=57)
    assert (js["n_opp"], js["n_combos"], js["n_odours"]) == (127, 1986, 55)
    assert js["digest_e0_b"].startswith("8c9729bf") and js["digest_keys"].startswith("37dde1ca")
    assert set(TP.summary(js)) == {"status", "n_b", "n_a", "last_turn", "last_turn_b", "last_turn_a", "skipped",
                                   "n_opp", "n_combos", "n_odours", "e1_clashes", "cap_fails", "all_off_pool",
                                   "clusters_b", "clusters_a", "digest_e0_b", "digest_e0_a", "digest_keys"}


@needs_npz
def test_keys_and_glomerulus_pairs_meet_nothing_used(world, js):
    pops, rc, _ = world
    cb = q_pairs.codebook(ENC, SPEC)
    rows = js["b"] + js["a"]
    keys = [e_pairs.egrid_key(r) for r in rows]
    assert len(set(keys)) == len(keys)                                     # no duplicate inside the set
    used = _used_rows(pops)
    assert not set(keys) & {e_pairs.egrid_key(r) for r in used}            # H.4 turns, L turns 0-209 (R, S sets)

    def gk(r):
        return frozenset({frozenset(encode_grid.glomeruli(cb, r["move_x"], tuple(r["opp_x"]))),
                          frozenset(encode_grid.glomeruli(cb, r["move_y"], tuple(r["opp_y"])))})
    assert not {gk(r) for r in rows} & {gk(r) for r in used}
    ps = TP.pool_type_sets()
    assert all(not (tuple(r["opp_x"]) in ps and tuple(r["opp_y"]) in ps) for r in rows)
    assert max(r["turn"] for r in js["b"]) == 17 and max(r["turn"] for r in js["a"]) == 103


@needs_npz
def test_e1_and_the_orn_cap_on_the_set_odours(world, js):
    pops, rc, params = world
    cb, rule = q_pairs.codebook(ENC, SPEC), E.dual_rule(SPEC.config)
    assert js["e1_clashes"] == [] and js["cap_fails"] == []
    cap = odor_real.cap_hz(params)
    for r in js["b"] + js["a"]:
        for m, o in ((r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"]))):
            assert all(params.max_rate_hz * SPEC.strength * v <= cap for v in
                       encode_grid.odour(rc, cb, m, o, rule).values())
    od = encode_grid.odour(rc, cb, "ELECTRIC", ("FIRE",), rule)            # T.9.1: the old set's odour over the cap
    assert params.max_rate_hz * SPEC.strength * max(od.values()) > cap
    assert TP._cap_fails(rc, cb, "ELECTRIC", ("FIRE",), rule, SPEC.strength, params.max_rate_hz, cap)


@needs_npz
def test_a_lower_cap_drops_the_rows_that_exceed_it(world, js):
    pops, rc, params = world
    hot = dataclasses.replace(params, max_rate_hz=params.max_rate_hz * 1.25)   # 250 Hz: more odours over the cap
    out = TP.t_set(pops, ENC, SPEC, hot)
    assert out["skipped"]["cap"] > js["skipped"]["cap"] and out["cap_fails"] == []
    assert TP.check_t_set(out, SPEC)


@needs_npz
def test_s_and_r_sets_must_reproduce_first(world, monkeypatch):
    monkeypatch.setattr(TP, "S_SPEC", dataclasses.replace(TP.S_SPEC, digest_keys="0" * 64))
    with pytest.raises(ValueError, match="S's judgement set"):
        TP.t_set(world[0], ENC, SPEC, world[2])


@needs_npz
def test_stop_set_short_when_the_turns_run_out(world):
    short = dataclasses.replace(SPEC, set_last_turn=60)
    out = TP.t_set(world[0], ENC, short, world[2])
    assert out["status"] == TP.STOP_SET_SHORT and out["last_turn"] == 60 and out["n_a"] < 43
    assert TP.check_t_set(out, short)


@needs_npz
def test_judgement_rows_set_odours_and_clusters(world):
    pops, rc, params = world
    rows = TP.judgement_rows(pops, rc, ENC, SPEC, params)
    assert [r["axis"] for r in rows] == ["b"] * 21 + ["a"] * 43
    assert all(set(r) >= {"odor_x", "odor_y", "odor_x_e0", "odor_y_e0"} and r["odor_x_e0"] and r["odor_y_e0"]
               for r in rows)
    od = TP.set_odours(pops, rc, ENC, SPEC, params)
    assert len(od) == 55 and all(isinstance(v, dict) and v for v in od.values())
    cl = TP.clusters(rows)
    assert len(cl) == 64 and cl[f"b|{rows[0]['turn']}|{rows[0]['x']}|{rows[0]['y']}"].count(" 대 ") == 1
    from collections import Counter
    assert Counter(v for k, v in cl.items() if k.startswith("a|")) == {f"{c}": n for c, n in SPEC.clusters_a}
    assert Counter(v for k, v in cl.items() if k.startswith("b|")) == {f"{x} 대 {y}": n for x, y, n in SPEC.clusters_b}
    with pytest.raises(ValueError, match="digest_keys"):
        TP.judgement_rows(pops, rc, ENC, dataclasses.replace(SPEC, digest_keys="0" * 64), params)
    with pytest.raises(ValueError, match="n_a"):
        TP.set_odours(pops, rc, ENC, dataclasses.replace(SPEC, n_a=42), params)
    with pytest.raises(ValueError, match="smoke"):
        TP.judgement_rows(pops, rc, ENC, smoke(SPEC), params)


def test_check_t_set_names_each_mismatch():
    js = dict(status="OK", n_b=21, n_a=43, last_turn=103, last_turn_b=17, last_turn_a=103, n_opp=127, n_combos=1986,
              n_odours=55, digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys,
              skipped=dict(SPEC.skipped_declared), clusters_b=sorted(list(c) for c in SPEC.clusters_b),
              clusters_a=sorted(list(c) for c in SPEC.clusters_a), e1_clashes=[], cap_fails=[], all_off_pool=True)
    assert TP.check_t_set(js, SPEC) == []
    for k, v in (("last_turn", 102), ("last_turn_b", 16), ("n_a", 44), ("digest_e0_a", "x"),
                 ("skipped", dict(js["skipped"], cap=0)), ("e1_clashes", ["FIRE|GROUND"]), ("all_off_pool", False),
                 ("cap_fails", ["ELECTRIC|FIRE"]), ("n_odours", 54)):
        assert any(k in m for m in TP.check_t_set(dict(js, **{k: v}), SPEC)), k
    assert TP.check_t_set(dict(js, status=TP.STOP_SET_SHORT, n_b=20), SPEC)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_t_pairs.py -q`
Expected: collection error `ImportError: cannot import name 't_pairs'`.

- [ ] **Step 3: Write the implementation**

```python
"""T's judgement set (T.2 as replaced by T.9.1), list only — nothing here runs the engine.
- opponents: Gen-1 dex numbers 1..opp_num_max without `forme`, every type in the codebook's 12 types, dex order (127).
- t_set: combos (me ∈ POOL, opp ∈ the widened pool, opp ≠ me, not in l_pairs.excluded_combos) in POOL order × the
  widened order, permuted once by default_rng(set_rng_seed); turn j's HP = l_pairs.HPS[j mod 6]; my moves = POOL's first
  four attacks (l_pairs.new_turns' turn shape); the alternate = G.11 over the widened order (first k ≥ 1, not my
  species, type-disjoint with the opponent); rows = e_pairs._rows_for_turn (L's and the encoder's row order and E0
  odours). Each row is checked in the declared order — ORN cap → glomerulus collision → used key
  (e_pairs.used_situations and L turns 0-209, so R's and S's sets too) → glomerulus-level duplicate of any used row →
  key already in the set → both opponent type sets in POOL — and a row failing one is counted under that reason; then
  (b) rows are taken until n_b and (a) rows until n_a (a full axis ignores its rows). Before any of it, S's set (and through it R's) must
  reproduce its declared digests with the same code (s_pairs.s_set / check_s_set).
- ORN cap (T.9.1): every glomerulus value v of the row's E-grid k2-norm odour (encode_grid.odour, the config's dual
  rule) satisfies max_rate_hz × s × v ≤ cap_hz (encode_grid.cap_ok's inequality); an odour that cannot be encoded is
  left to the collision check (plan Reading 6).
- check_t_set: the generated set against T.9.1's declared values (turns, counts, digests, skip counts, the cluster
  table, 55 odours, no E1 clash with the 112 reachable odours, every row off-POOL, every odour under the cap).
- judgement_rows: the checked set's (b) 21 then (a) 43 rows with E-grid odours; refused for a smoke spec. set_odours:
  its 55 distinct E-grid odours (gate ①'s supplement). Only T's judgement stages and the gate-① supplement call them
  (plan Global Constraints)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import glomeruli, odour, reachable
from ..battle.pool import POOL
from . import h4_pairs, l_pairs, odor_real, q_pairs, s_pairs
from .r_pairs import okey, row_key
from .s_spec import SPEC as S_SPEC

STOP_SET_SHORT = "STOP_SET_SHORT"
OK = "OK"
REASONS = ("cap", "collision", "used", "glom_dup", "in_set", "pool_only")


def opponents(mon_types, spec) -> list:
    """[(name, types)] of the widened pool, dex order."""
    dex = GenData.from_gen(1).pokedex
    out = []
    for _, v in sorted(dex.items(), key=lambda kv: kv[1].get("num", 0)):
        if not (1 <= v.get("num", 0) <= spec.opp_num_max) or v.get("forme"):
            continue
        ts = tuple(t.upper() for t in v["types"])
        if set(ts) <= set(mon_types):
            out.append((v["name"], ts))
    return out


def _glom(cb, m, o):
    try:
        return frozenset(glomeruli(cb, m, o))
    except ValueError:
        return None


def _gkey(cb, r) -> frozenset:
    return frozenset({_glom(cb, r["move_x"], tuple(r["opp_x"])), _glom(cb, r["move_y"], tuple(r["opp_y"]))})


def _cap_fails(rc, cb, m, o, rule, s, max_rate_hz, cap_hz) -> bool:
    try:
        od = odour(rc, cb, m, o, rule)
    except ValueError:
        return False                                       # not encodable: the collision check takes it
    return not all(max_rate_hz * s * v <= cap_hz for v in od.values())


def _tag(types, pool_sets) -> str:
    return "+".join(types) + ("" if tuple(types) in pool_sets else "*")


def cluster_of(r, pool_sets) -> tuple:
    """T.9.1's cluster: (b) (X's opponent type set, Y's); (a) the opponent type set (X's = Y's). * = off POOL."""
    x = _tag(tuple(r["opp_x"]), pool_sets)
    return (x, _tag(tuple(r["opp_y"]), pool_sets)) if r["axis"] == "b" else (x,)


def pool_type_sets() -> set:
    st, _, _, _ = h4_pairs.pool_vocabulary()
    return {tuple(sorted(v)) for v in st.values()}


def t_set(pops, enc: dict, spec, params) -> dict:
    """The set and its records (module docstring). ValueError when S's (or R's) set does not reproduce first."""
    js_s = s_pairs.s_set(pops, enc, S_SPEC)
    bad = s_pairs.check_s_set(js_s, S_SPEC)
    if bad:
        raise ValueError("S's judgement set does not reproduce: " + "; ".join(bad))
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    max_rate, cap = float(params.max_rate_hz), float(odor_real.cap_hz(params))
    opp = opponents(mon, spec)
    species = dict(st)
    species.update({n: ts for n, ts in opp})
    names = [n for n, _ in opp]
    pool_sets = {tuple(sorted(v)) for v in st.values()}
    ex = l_pairs.excluded_combos()
    combos = [(m, n) for m in POOL for n in names if to_id_str(n) != to_id_str(m.species) and (m.species, n) not in ex]
    used_rows = list(e_pairs.used_situations(pops, E))
    for t in l_pairs.new_turns(st, mi, E.judge_last_turn + 1, E.l_rng_seed):
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        used_rows += e_pairs._rows_for_turn(pops, chan, t, alt, st, with_a=True)
    used = {e_pairs.egrid_key(r) for r in used_rows}
    used_g = {_gkey(cb, r) for r in used_rows}
    order = np.random.default_rng(spec.set_rng_seed).permutation(len(combos))
    b, a, taken, skipped, last = [], [], set(), Counter({k: 0 for k in REASONS}), None
    for j, ix in enumerate(order[:spec.set_last_turn + 1]):
        me, on = combos[int(ix)]
        my_hp, op_hp = l_pairs.HPS[j % len(l_pairs.HPS)]
        cands = [{"move": x, "type": mi[x][0], "bp": mi[x][1]} for x in me.attacks][:4]
        t = {"turn": j, "pos": j, "me": me.species, "opp": on, "my_types": list(st[me.species]),
             "opp_types": list(species[on]), "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands}
        s0, alt = names.index(on), None
        for k in range(1, len(names)):
            c = names[(s0 + k) % len(names)]
            if to_id_str(c) != to_id_str(me.species) and not (set(species[c]) & set(species[on])):
                alt = c
                break
        if alt is None:
            raise ValueError(f"turn {j}: no type-disjoint alternate opponent in the widened pool")
        for r in e_pairs._rows_for_turn(pops, chan, t, alt, species, with_a=True):
            sides = ((r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"])))
            k = e_pairs.egrid_key(r)
            if any(_cap_fails(rc, cb, m, o, rule, spec.strength, max_rate, cap) for m, o in sides):
                skipped["cap"] += 1
            elif any(_glom(cb, m, o) is None for m, o in sides):
                skipped["collision"] += 1
            elif k in used:
                skipped["used"] += 1
            elif _gkey(cb, r) in used_g:
                skipped["glom_dup"] += 1
            elif k in taken:
                skipped["in_set"] += 1
            elif tuple(r["opp_x"]) in pool_sets and tuple(r["opp_y"]) in pool_sets:
                skipped["pool_only"] += 1
            elif r["axis"] == "b" and len(b) < spec.n_b:
                taken.add(k)
                b.append(r)
            elif r["axis"] == "a" and len(a) < spec.n_a:
                taken.add(k)
                a.append(r)
        last = j
        if len(b) == spec.n_b and len(a) == spec.n_a:
            break
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in b + a]
    reach = {frozenset(glomeruli(cb, m, o)): (m, o) for m, o in reachable(st, move)}
    odours = sorted({side for r in b + a for side in ((r["move_x"], tuple(r["opp_x"])),
                                                       (r["move_y"], tuple(r["opp_y"])))})
    e1 = [okey(m, o) for m, o in odours if reach.get(frozenset(glomeruli(cb, m, o)), (m, o)) != (m, o)]
    cap_bad = [okey(m, o) for m, o in odours if _cap_fails(rc, cb, m, o, rule, spec.strength, max_rate, cap)]
    clusters = Counter(cluster_of(r, pool_sets) for r in b + a)
    return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=last,
                last_turn_b=max((r["turn"] for r in b), default=None),
                last_turn_a=max((r["turn"] for r in a), default=None),
                status=OK if (len(b), len(a)) == (spec.n_b, spec.n_a) else STOP_SET_SHORT, skipped=dict(skipped),
                n_opp=len(names), n_combos=len(combos), n_odours=len(odours), e1_clashes=e1, cap_fails=cap_bad,
                all_off_pool=all(not (tuple(r["opp_x"]) in pool_sets and tuple(r["opp_y"]) in pool_sets)
                                 for r in b + a),
                clusters_b=sorted([*c, n] for c, n in clusters.items() if len(c) == 2),
                clusters_a=sorted([*c, n] for c, n in clusters.items() if len(c) == 1),
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest())


def check_t_set(js: dict, spec) -> list:
    """T.9.1: [] when the generated set equals the declared one."""
    bad = []
    if js["status"] != OK or (js["n_b"], js["n_a"]) != (spec.n_b, spec.n_a):
        bad.append(f"T set status {js['status']}, n_b {js['n_b']} · n_a {js['n_a']}, declared {spec.n_b} · {spec.n_a}")
    for k, want in (("last_turn_b", spec.last_turn_b), ("last_turn", spec.last_turn), ("last_turn_a", spec.last_turn),
                    ("n_opp", spec.n_opp), ("n_combos", spec.n_combos), ("n_odours", spec.n_set_odours),
                    ("digest_e0_b", spec.digest_e0_b), ("digest_e0_a", spec.digest_e0_a),
                    ("digest_keys", spec.digest_keys), ("skipped", dict(spec.skipped_declared)),
                    ("clusters_b", sorted(list(c) for c in spec.clusters_b)),
                    ("clusters_a", sorted(list(c) for c in spec.clusters_a)),
                    ("e1_clashes", []), ("cap_fails", []), ("all_off_pool", True)):
        if js[k] != want:
            bad.append(f"T set {k}: generated {js[k]!r}, declared {want!r}")
    return bad


def summary(js: dict) -> dict:
    """What block `set` records (the declared values, already public in T.9.1)."""
    return {k: js[k] for k in ("status", "n_b", "n_a", "last_turn", "last_turn_b", "last_turn_a", "skipped", "n_opp",
                               "n_combos", "n_odours", "e1_clashes", "cap_fails", "all_off_pool", "clusters_b",
                               "clusters_a", "digest_e0_b", "digest_e0_a", "digest_keys")}


def _checked(pops, enc: dict, spec, params) -> dict:
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (T.3 4)")
    js = t_set(pops, enc, spec, params)
    bad = check_t_set(js, spec)
    if bad:
        raise ValueError("; ".join(bad))
    return js


def judgement_rows(pops, rc: dict, enc: dict, spec, params) -> list:
    js = _checked(pops, enc, spec, params)
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))


def set_odours(pops, rc: dict, enc: dict, spec, params) -> dict:
    """{odour id: E-grid odour} of the checked set's distinct (move type, opponent types) — gate ①'s supplement."""
    js = _checked(pops, enc, spec, params)
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    sides = sorted({side for r in js["b"] + js["a"] for side in ((r["move_x"], tuple(r["opp_x"])),
                                                                  (r["move_y"], tuple(r["opp_y"])))})
    return {okey(m, o): odour(rc, cb, m, o, rule) for m, o in sides}


def clusters(rows: list) -> dict:
    """{pair key: cluster label} (T.9.1's table: "X 대 Y" on (b), the type set on (a))."""
    ps = pool_type_sets()
    return {row_key(r): " 대 ".join(cluster_of(r, ps)) for r in rows}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_t_pairs.py -q`
Expected: 8 passed (a few seconds: the connectome loads once per module; one `t_set` call takes ~0.1 s). The generated set gives `status OK, n_b 21, n_a 43, last_turn_b 17, last_turn 103`, skips `cap 9, collision 0, used 164, glom_dup 0, in_set 48, pool_only 57`, 55 odours and the three declared digests.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/t_pairs.py tests/brain/test_t_pairs.py
git commit -m "feat(t): t_pairs — widened Gen-1 opponent set with ORN cap first, declared T.9.1 values, rows, 55 odours, clusters"
```

---

### Task 3: `t_measure` and `t_store` — T's measurement file (reference set on an edited engine) and T's writer and caches

**Files:**
- Create: `flymon/brain/t_measure.py`, `flymon/brain/t_store.py`
- Test: `tests/brain/test_t_measure.py`, `tests/brain/test_t_store.py`

**Interfaces:**
- Consumes: `engine_cpu.Engine`, `h3_jobs.mbon_type_index` / `reference_job` / `rest_job` (tests), `h3_measure.chunks`, `h3_store.code_key` / `canonical` / `canonical_pretty`, `q_jobs.apply_q_edit`, `stimuli.present`, `r_measure.R_MEASURE_FILES` / `RMeasurer`, `r_store.RCache` / `load_manifest` / `archive_copy` (unchanged), `pool_bench` guards, Task 1's `SPEC` / `smoke`.
- Produces:
  - `t_measure`: `T_MEASURE_FILES = ("flymon/brain/t_measure.py", "flymon/brain/h3_spec.py")`; `t_measure_key(npz) -> dict(key, files)`; `t_ref_job(eng, pl, pops, comps, ro, params, edit, p_type, types, odors, strength, settle_ms, read_steps) -> [dict(odor, seed, types{type: count}, kc_active_frac, csc_sha256, edit_edges)]`; `t_rest_job(…, seeds, settle_ms, read_steps) -> [dict(seed, types, csc_sha256, edit_edges)]`; `ZMeasurer(pool, params, p_type, types)` with `reference(edit, odors, strength, settle_ms, read_steps) -> list` and `rest(edit, seeds, settle_ms, read_steps) -> list`.
  - `t_store`: `ALLOWED_DIR = "results/t/"`, `SUMMARY = "results/summary/t_lever.json"`, `SMOKE_SEEDS`; `guard`, `write_bytes`, `write_json(path, obj, params_list) -> Path`, `read_summary(path=SUMMARY)`, `write_summary_block(path, block, obj, params_list)`; `TCache(root, code, smoke_seeds=SMOKE_SEEDS)` (RCache with T's `put`); `RReadCache(root, code)` (RCache whose `put` refuses); `load_manifest`, `archive_copy` (r_store's objects).

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_t_measure.py
"""T's measurement file (T.1, T.8, T.9.4, T.9.6): the T measurement key is the shared key's files plus t_measure.py
and h3_spec.py and differs from the shared key; ZMeasurer splits the reference set and the rest seeds over the pool
as h3_measure.chunks does; and on the real connectome t_ref_job / t_rest_job with edit "none" give exactly
h3_jobs.reference_job's / rest_job's counts (the unedited z can reproduce block h4's z bit for bit), while the lever
changes 2 CSC edges (R's L CSC 860cba4f…, the unedited 1aee8398…)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from flymon.brain import h3_jobs
from flymon.brain import t_measure as TM
from flymon.brain.h3_spec import SPEC as H3
from flymon.brain.h3_spec import make_odors
from flymon.brain.r_measure import R_MEASURE_FILES
from flymon.brain.t_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


def test_t_measure_files_and_key():
    assert TM.T_MEASURE_FILES == ("flymon/brain/t_measure.py", "flymon/brain/h3_spec.py")
    assert not set(TM.T_MEASURE_FILES) & set(R_MEASURE_FILES)
    if NPZ.exists():
        k = TM.t_measure_key(str(NPZ))
        assert set(k["files"]) - {"npz:malecns.npz"} == set(R_MEASURE_FILES) | set(TM.T_MEASURE_FILES)
        assert k["key"] != SPEC.r_shared_key and len(k["key"]) == 64


class FakePool:
    n_workers = 3

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, [len(kw.get("odors", kw.get("seeds", []))) for kw in kws], kws[0]))
        if fn is TM.t_ref_job:
            return [[dict(odor=o["name"], seed=s) for o in kw["odors"] for s in o["seeds"]] for kw in kws]
        return [[dict(seed=s) for s in kw["seeds"]] for kw in kws]


def test_zmeasurer_chunks_and_arguments():
    pool = FakePool()
    zm = TM.ZMeasurer(pool, {"p": 1}, "MBON05", ["MBON13", "MBON05"])
    odors = [dict(name=f"R{j:02d}", seeds=[2 * j, 2 * j + 1], strengths={}) for j in range(7)]
    ref = zm.reference(SPEC.lever_edit, odors, 0.35, 800.0, 600)
    assert [(r["odor"], r["seed"]) for r in ref] == [(o["name"], s) for o in odors for s in o["seeds"]]
    name, sizes, kw = pool.calls[0]
    assert name == "t_ref_job" and sizes == [3, 3, 1]
    assert {k: kw[k] for k in ("edit", "p_type", "types", "strength", "settle_ms", "read_steps")} == dict(
        edit=SPEC.lever_edit, p_type="MBON05", types=["MBON13", "MBON05"], strength=0.35, settle_ms=800.0,
        read_steps=600)
    rest = zm.rest("none", list(range(14)), 800.0, 600)
    assert [r["seed"] for r in rest] == list(range(14)) and pool.calls[1][0] == "t_rest_job"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_unedited_jobs_equal_h3s_reference_and_rest_jobs():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    conn = Connectome.load(str(NPZ))
    pops = Populations.from_connectome(conn)
    params = load_c3_config(SPEC.m0d_summary).params
    od, w, eng = make_odors(pops, H3.reference)[:1], H3.reference_window, SimpleNamespace(conn=conn)
    types = ["MBON13", "MBON18", "MBON05", "MBON21"]
    mine = TM.t_ref_job(eng, None, pops, None, None, params, "none", "MBON05", types, od, H3.strength,
                        w.settle_ms, int(w.read_ms))
    ref = h3_jobs.reference_job(eng, None, pops, None, None, params, od, H3.strength, w.settle_ms, int(w.read_ms),
                                tuple(H3.apl_v_quantiles), tuple(H3.callout_types))
    assert [m["types"] for m in mine] == [{t: r["types"][t] for t in types} for r in ref]
    assert [m["kc_active_frac"] for m in mine] == [r["kc_active_frac"] for r in ref]
    assert {m["edit_edges"] for m in mine} == {0} and mine[0]["csc_sha256"].startswith("1aee8398")
    seed = od[0]["seeds"][0]
    r1 = TM.t_rest_job(eng, None, pops, None, None, params, "none", "MBON05", types, [seed], w.settle_ms,
                       int(w.read_ms))
    r2 = h3_jobs.rest_job(eng, None, pops, None, None, params, [seed], w.settle_ms, int(w.read_ms))
    assert r1[0]["types"] == {t: r2[0]["types"][t] for t in types}
    lv = TM.t_rest_job(eng, None, pops, None, None, params, SPEC.lever_edit, "MBON05", types, [seed], w.settle_ms,
                       int(w.read_ms))
    assert lv[0]["edit_edges"] == SPEC.lever_edges and lv[0]["csc_sha256"].startswith("860cba4f")
```

```python
# tests/brain/test_t_store.py
"""T's writer and caches (T.5, T.8, T.9.2; plan Readings 3, 5): writes only under results/t/ and
results/summary/t_lever.json, atomically — every other track's path (results/r/, results/s/, results/p/, R's and S's
summaries) refuses; TCache is RCache with T's writer and T's smoke seeds; RReadCache reads R's root by content key and
never writes; r_store's load_manifest and archive_copy are reused as they are; the unchanged RMeasurer, one per z,
writes T entries under results/t/ whose inputs differ only by z; and R's committed even raw for C (and L) is found by
content key under results/r/cache with block h4's z (gate ③'s reuse)."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_store
from flymon.brain import t_store as T
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.t_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
READOUT, Z, TYPES = {"A": "MBON13", "P": "MBON05"}, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, ["MBON13", "MBON05"]
ZL = {"A": (7.0, 6.0), "P": (80.0, 30.0)}


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/r/x.json", "results/r/cache/r_oracle/a.json", "results/s/x.json", "results/s/cache/a.json",
                "results/p/x.json", "results/q/x.json", "results/summary/r_lever.json",
                "results/summary/s_lever.json", "results/summary/p_learning.json", "results/tx/a.json",
                "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            T.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(T.write_json("results/t/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    T.write_summary_block(T.SUMMARY, "reuse", {"n": 1}, [])
    T.write_summary_block(T.SUMMARY, "set", {"m": 2}, [])
    assert T.read_summary() == {"reuse": {"n": 1}, "set": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert T.SUMMARY == SPEC.summary


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/t").mkdir(parents=True)
    outside.mkdir()
    (root / "results/t/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        T.write_json("results/t/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_tcache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = T.TCache(SPEC.cache_dir, {"key": "k"}), T.TCache(SPEC.smoke_cache_dir, {"key": "k"})
    assert isinstance(real, r_store.RCache)
    assert T.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_400_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_400_000]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_209_100})                   # T's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_400_000]})           # a real seed under the smoke root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"seeds": [24_309_000]})               # S's smoke seed is not T's
    sm.put("r_arm", {"seed": 25_209_100}, {"v": 2}, [])
    assert sm.get("r_arm", {"seed": 25_209_100}) == {"v": 2}
    p = real._path("r_oracle", {"act_seeds": [24_400_000]})
    p.write_text("{trunc")
    assert real.get("r_oracle", {"act_seeds": [24_400_000]}) is None
    with pytest.raises(SystemExit):
        T.TCache("results/r/cache", {"key": "k"}).put("r_oracle", {"act_seeds": [24_400_000]}, {"v": 1}, [])


def test_rreadcache_reads_rs_root_and_never_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = r_store.RCache("results/r/cache", {"key": "k"})
    w.put("r_oracle", {"act_seeds": [500]}, {"v": 3}, [])
    ro = T.RReadCache("results/r/cache", {"key": "k"})
    assert ro.get("r_oracle", {"act_seeds": [500]}) == {"v": 3}
    assert ro.key("r_oracle", {"act_seeds": [500]}) == w.key("r_oracle", {"act_seeds": [500]})
    before = sorted(Path("results/r").rglob("*"))
    with pytest.raises(SystemExit):
        ro.put("r_oracle", {"act_seeds": [501]}, {"v": 4}, [])
    assert sorted(Path("results/r").rglob("*")) == before
    assert {k for k in vars(T.RReadCache) if not k.startswith("__")} == {"put"}
    assert {k for k in vars(T.TCache) if not k.startswith("__")} == {"put"}


def test_manifest_and_archive_are_r_stores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert T.load_manifest is r_store.load_manifest and T.archive_copy is r_store.archive_copy
    f = Path("results/t/cache/r_oracle/a.json")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"key": "K", "kind": "r_oracle", "result": {"v": 1}}))
    man = [dict(key="b|0|x|y", cache_key="K", cache_file=str(f), sha256=sha256_file(f))]
    got, bad = T.load_manifest(man)
    assert bad == [] and got[0]["result"] == {"v": 1}
    root = tmp_path / "arch-t"
    out = T.archive_copy([str(f)], root / "seal1", root)
    assert Path(out[0]["dst"]).read_text() == f.read_text()
    with pytest.raises(SystemExit):
        T.archive_copy([str(f)], tmp_path / "arch-s" / "x", root)


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn is r_jobs.r_oracle_job:
            return [dict(echo=kw["odor_x"], edit=kw["edit"], z=kw["z"]) for kw in kws]
        return [dict(seed=kw["seed"], arm=kw["arm"], edit=kw["edit"], r=dict(edit_edges=0)) for kw in kws]


ROWS = [dict(axis="b", turn=i, x=f"x{i}", y=f"y{i}", odor_x={f"G{i}": 1.0}, odor_y={"H": 1.0},
             odor_x_e0={"E": 1.0}, odor_y_e0={"F": 1.0}) for i in range(3)]


def test_one_measurer_per_z_over_one_cache(tmp_path, monkeypatch):
    """Plan Reading 4: L's measurer carries z_lever, C's h4 z; same cache, entries differ only by z and condition."""
    monkeypatch.chdir(tmp_path)
    pool, cache = FakePool(), T.TCache(SPEC.cache_dir, {"key": "k"})
    ml = RMeasurer(pool, cache, SPEC, Params(), READOUT, ZL, TYPES, 100)
    mc = RMeasurer(pool, cache, SPEC, Params(), READOUT, Z, TYPES, 100)
    gl = ml.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    gc = mc.oracle(ROWS, SPEC.cond("C"), "judge", SPEC.judge_seeds())
    il, ic = (json.loads(Path(g[0]["cache_file"]).read_text())["inputs"] for g in (gl, gc))
    assert il["z"] == {k: list(v) for k, v in ZL.items()} and ic["z"] == {k: list(v) for k, v in Z.items()}
    assert {k for k in il if il[k] != ic[k]} == {"z", "edit", "condition"}
    assert all(g["cache_file"].startswith("results/t/cache/r_oracle/") for g in gl + gc)
    Path(gl[1]["cache_file"]).unlink()
    pool.calls.clear()
    again = ml.oracle(ROWS, SPEC.cond("L"), "judge", SPEC.judge_seeds())
    assert pool.calls == [("r_oracle_job", 1)] and [g["result"] for g in again] == [g["result"] for g in gl]
    item = dict(direction="r1", x="4:1", y="dDL", edit="none", arm="punish", punish=True, plastic=True,
                da_zero=False, odor_x={"G": 1.0}, odor_y={"H": 1.0}, seed=25_200_000, point=(0.25, 8.0))
    rows = mc.arms([item, dict(item, edit=SPEC.lever_edit)], READOUT, "PPL105", "gate2", P_SPEC.o.n)
    assert [r["edit"] for r in rows] == ["none", SPEC.lever_edit]


NPZ = ROOT / "data/malecns.npz"
R_CACHE = ROOT / SPEC.r_cache_dir / "r_oracle"


@pytest.mark.skipif(not (NPZ.exists() and R_CACHE.exists()), reason="no connectome or no R raw cache")
def test_rs_even_raw_is_found_by_content_key():
    """T.9.2: with block h4's z, T's spec builds the same oracle inputs R's even stage stored, so every L and C even
    entry is read from results/r/cache with no pool and nothing written."""
    from flymon.agent.config import load_c3_config
    from flymon.brain import q_pairs, r_pairs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    cfg = load_c3_config(SPEC.m0d_summary)
    enc = json.loads((ROOT / SPEC.encoder_summary).read_text())
    q_pairs.check_strength(enc, SPEC)
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = r_pairs.even_rows(pops, rc, enc, SPEC)
    m0d = json.loads((ROOT / SPEC.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    cache = T.RReadCache(str(ROOT / SPEC.r_cache_dir), code_key(str(NPZ), files=R_MEASURE_FILES))
    m = RMeasurer(None, cache, SPEC, cfg.params, cfg.readout, cfg.z, types, len(pops.kc))
    for n in ("C", "L"):
        ins = [m.inputs(r, SPEC.cond(n), "even", SPEC.h4_seeds()) for r in rows]
        assert all(cache.get("r_oracle", x) is not None for x in ins), n
    got = m.oracle(rows, SPEC.cond("C"), "even", SPEC.h4_seeds())
    assert len(got) == 39 and m.last_jobs == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_t_measure.py tests/brain/test_t_store.py -q`
Expected: collection errors `ImportError: cannot import name 't_measure'` / `'t_store'`.

- [ ] **Step 3: Write `t_measure`**

```python
"""T's only measurement file (T.1, T.8, T.9.4, T.9.6): the H.3 reference set and its same-seed rest on an edited
engine, for each engine variant's own z and the readout guard. The shared measurement files (r_measure.R_MEASURE_FILES)
are not touched; this file and h3_spec (the reference set's generator) form the T measurement key with them.
- t_ref_job: h3_jobs.reference_job's call sequence (reset / clear_drive / present / run(settle) / step x read) on an
  engine built as h3_jobs.engine_for builds it (Engine(conn, pops, params, seed=0), no plasticity) and then edited in
  place by q_jobs.apply_q_edit (the lever's 2 edges; "none" changes nothing), returning per presentation the summed
  read-window count of every listed MBON type (both hemispheres, h3_jobs.mbon_type_index), the KC activity, the CSC
  sha and the edited edge count. With edit "none" its counts are reference_job's (test), so the unedited z reproduces
  block h4's z bit for bit (T.1).
- t_rest_job: h3_jobs.rest_job's sequence on the same engine (no odour): the guard's same-seed resting counts.
- ZMeasurer: both jobs over a pool in h3_measure.chunks, no cache — the unedited reproduction runs without a cache
  (T.9.6) and the lever's z is minutes of work; the rows are written once under results/t/ by the runner."""
from __future__ import annotations

import numpy as np

from .engine_cpu import Engine
from .h3_jobs import mbon_type_index
from .h3_measure import chunks
from .h3_store import code_key
from .q_jobs import apply_q_edit
from .r_measure import R_MEASURE_FILES
from .stimuli import present

T_MEASURE_FILES = ("flymon/brain/t_measure.py", "flymon/brain/h3_spec.py")
_RIG: dict = {}          # (Params, edit, p_type) -> (Engine, type index, csc sha256, edges changed); one per worker


def t_measure_key(npz: str) -> dict:
    """T.8 / T.9.6: the shared measurement key's files plus T's measurement files."""
    return code_key(npz, files=tuple(dict.fromkeys(R_MEASURE_FILES + T_MEASURE_FILES)))


def z_engine(conn, pops, params, edit: str, p_type: str):
    key = (params, edit, p_type)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha, n = apply_q_edit(eng, pops, edit, p_type)
        _RIG[key] = (eng, mbon_type_index(conn, pops), sha, n)
    return _RIG[key]


def _read(e, steps: int) -> np.ndarray:
    counts = np.zeros(e.N, np.int32)
    for _ in range(int(steps)):
        counts[e.step()] += 1
    return counts


def t_ref_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, types, odors, strength: float,
              settle_ms: float, read_steps: int) -> list:
    e, idx, sha, n = z_engine(eng.conn, pops, params, edit, p_type)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], strength); e.run(settle_ms)
            counts = _read(e, read_steps)
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), types={t: int(counts[idx[t]].sum()) for t in types},
                            kc_active_frac=float((kc > 0).mean()), csc_sha256=sha, edit_edges=int(n)))
    return out


def t_rest_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, types, seeds, settle_ms: float,
               read_steps: int) -> list:
    e, idx, sha, n = z_engine(eng.conn, pops, params, edit, p_type)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); e.run(settle_ms)
        counts = _read(e, read_steps)
        out.append(dict(seed=int(seed), types={t: int(counts[idx[t]].sum()) for t in types}, csc_sha256=sha,
                        edit_edges=int(n)))
    return out


class ZMeasurer:
    def __init__(self, pool, params, p_type: str, types):
        self.pool, self.params, self.p_type, self.types = pool, params, p_type, [str(t) for t in types]

    def _common(self, edit: str, settle_ms: float, read_steps: int) -> dict:
        return dict(params=self.params, edit=edit, p_type=self.p_type, types=list(self.types),
                    settle_ms=float(settle_ms), read_steps=int(read_steps))

    def reference(self, edit: str, odors: list, strength: float, settle_ms: float, read_steps: int) -> list:
        common = dict(self._common(edit, settle_ms, read_steps), strength=float(strength))
        parts = self.pool.run_jobs(t_ref_job, [dict(common, odors=c) for c in chunks(odors, self.pool.n_workers)])
        return [r for part in parts for r in part]

    def rest(self, edit: str, seeds: list, settle_ms: float, read_steps: int) -> list:
        common = self._common(edit, settle_ms, read_steps)
        seeds = [int(s) for s in seeds]
        parts = self.pool.run_jobs(t_rest_job, [dict(common, seeds=c) for c in chunks(seeds, self.pool.n_workers)])
        return [r for part in parts for r in part]
```

- [ ] **Step 4: Write `t_store`**

```python
"""T's only writer (T.5, T.8): raw files under results/t/, the one summary results/summary/t_lever.json, atomic writes
(temporary file + rename). r_store.py is part of the shared measurement key (r_measure.R_MEASURE_FILES), so it is never
edited (T.3 1, T.8); T reuses it where it is path-free and wraps it where it is not (plan Reading 3):
- guard / write_bytes / write_json / read_summary / write_summary_block: r_store's code with T's allowed paths.
- TCache: r_store.RCache (ECache's key over the shared code key, RCache's smoke scope and its get) with only `put`
  routed through this module's writer and T's smoke seeds as the default scope — a T entry has R's layout and key
  formula but can only land under results/t/.
- RReadCache: r_store.RCache over R's own root, read only — `put` refuses. Gate ③ reads R's even raw (C, and L as a
  record) through it by content key (T.9.2); a missing entry is a refusal, never a measurement into R's tree.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for t_runner)
from .t_spec import SPEC as T_SPEC
from .t_spec import smoke

ALLOWED_DIR = "results/t/"
SUMMARY = "results/summary/t_lever.json"
SMOKE_SEEDS = frozenset(T_SPEC.smoke_seeds) | frozenset(smoke(T_SPEC).p.seeds) | frozenset(smoke(T_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"T writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


class TCache(RCache):
    """RCache with T's smoke seeds as the default scope and T's writer (a root outside results/t/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


class RReadCache(RCache):
    """R's cache root, read only (T.9.2): get as RCache, put refuses."""

    def put(self, kind, inputs, result, params_list) -> None:
        _refuse(f"{self.root} is R's raw data; T reads it by content key and never writes it")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_t_measure.py tests/brain/test_t_store.py tests/brain/test_s_store.py tests/brain/test_r_store_measure.py -q`
Expected: all pass (3 + 7 T tests). `test_unedited_jobs_equal_h3s_reference_and_rest_jobs` runs the engine for one reference odour on two seeds (~15 s) and compares every type count with `h3_jobs.reference_job` / `rest_job`; `test_rs_even_raw_is_found_by_content_key` needs `results/r/cache/r_oracle/` (present in this worktree; skipped elsewhere).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/t_measure.py flymon/brain/t_store.py tests/brain/test_t_measure.py tests/brain/test_t_store.py
git commit -m "feat(t): t_measure (reference set + rest on an edited engine, T measurement key) and t_store (results/t/ writer, TCache, read-only RReadCache)"
```

---

### Task 4: `t_rules` and `t_records` — gates, z decisions, sentences, both OCs, the cluster fixture, records

**Files:**
- Create: `flymon/brain/t_rules.py`, `flymon/brain/t_records.py`, `tests/brain/fixtures/t_oc_cluster.json` (generated in Step 4)
- Test: `tests/brain/t_fixtures.py`, `tests/brain/test_t_rules.py`, `tests/brain/test_t_records.py`

**Interfaces:**
- Consumes: `r_rules` (constants, `read_band`, `gate3`, `SENTENCES`), `s_rules` (`g_fail_s`, `gate2`, `g_prob`, `oc`, `OC_NOTES`, `SENTENCES`, `sentence`), `e_rules` (`odour_activity`, `_binom`), `r_records.cond_summary`, `s_records.ratio` / `gate2_oc`, `h3_rules.mbon_type_stats`, `h4_rules.z_constants`, `p_rules.p_judge`, `h3_store.canonical`, Task 1's spec; tests reuse `r_fixtures.py` and `s_fixtures.py`.
- Produces:
  - `t_rules`: `PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN, STOP_REUSE, STOP_SET_SHORT, STOP_Z_REPRO, STOP_Z_DEGENERATE, STOP_STRENGTH_LEVER, STOP_EVEN_REPRO, STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH, STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED, SELECTED, B_TB, B_FA, B_NC, B_PG, BANDS, C_ABOVE_BAR, BELOW_BAR, MARGIN`; `read_band` (R's), `g_fail_s` / `gate2` (S's); `reuse(r_doc, r_git, shared_key, spec)`; `set_outcome(js)`; `z_repro(none_side, n, repro_sha, spec)`; `z_lever(lever_side, n, readout, spec)` (`failed` on STOP); `gate1s(rec, spec)` (`failed`); `gate3(L | None, C, spec, repro_sha, l_h4)`; `SENTENCES`, `sentence(outcome, fields)`; `OC_NOTES`; `oc(spec) -> dict(... notes, g_fail, rows, sha256)`; `oc_cluster(spec) -> dict(model, icc, kappa, draws, seed, clusters_b, clusters_a, g_fail[{kind, a_pf, a_fp, p, p_independent}], n_dist{q: {cluster, independent}}, rows, notes, sha256)`.
  - `t_records`: `z_side(ref, rest, readout, spec) -> dict(z | None, why, guard{type: stats}, zero_sd, n_ref, n_rest, edit_edges, csc_sha256)`; `z_ratios(z_lever, z_h4)`; `gate1s_record(act_lever, act_none, spec)`; `p_zlever(rows_l, z_lever, c1, res_c, spec)`; `clusters(pairs_by_condition, labels)`; `alpha_fixed(raws_l, C_summary, keys, seeds, z_h4, spec)`; `ratio`, `gate2_oc` (S's).
  - `tests/brain/t_fixtures.py`: `SHA_NONE`, `SHA_L`, `ref_rows(a, p, edges, sha, seed0)`, `rest_rows(n, rest, edges, sha, seed0)`, `counts(n, base, zeros)`.

- [ ] **Step 1: Write the fixtures and the failing tests**

```python
"""Fabricated reference-set rows for T's z tests (no engine): ref_rows / rest_rows give t_ref_job / t_rest_job's
shape. With base counts a / p per presentation and a rest of `rest` spikes, the readout guard reads median(read − rest)
and the zero share of the read counts; z is the counts' mean and population SD."""

SHA_NONE, SHA_L = "sha-C", "sha-L"


def ref_rows(a: list, p: list, edges: int = 0, sha: str = SHA_NONE, seed0: int = 1000) -> list:
    return [dict(odor=f"R{i // 2:02d}", seed=seed0 + i, types={"MBON13": int(x), "MBON05": int(y)},
                 kc_active_frac=0.05, csc_sha256=sha, edit_edges=edges) for i, (x, y) in enumerate(zip(a, p))]


def rest_rows(n: int, rest: int = 0, edges: int = 0, sha: str = SHA_NONE, seed0: int = 1000) -> list:
    return [dict(seed=seed0 + i, types={"MBON13": rest, "MBON05": rest}, csc_sha256=sha, edit_edges=edges)
            for i in range(n)]


def counts(n: int, base: int, zeros: int = 0) -> list:
    """n counts: `zeros` zeros first, the rest base + (i % 5)."""
    return [0] * zeros + [base + (i % 5) for i in range(n - zeros)]
```

```python
# tests/brain/test_t_rules.py
"""T.1-T.4 / T.6 / T.7 as replaced by T.9: every cell n, c ∈ 0..21 × F_a ∈ 0..43 × G_fail_S reads exactly one band
(R.3's order, r_rules.read_band itself) and G_fail_S / gate ② are S's functions; the reuse gate (R's repro and
gate ①), the set outcome, the z gates in their order (STOP_Z_REPRO on any bit of difference from block h4's z, then
STOP_Z_DEGENERATE on H.4's guard at its boundaries 5 / 0.25 or a zero SD), the gate-① supplement on the T set's
odours, gate ③ with R's C reproduction first; the sentences verbatim; the independent OC equal to S.6's values with
T's notes; the cluster-correlated OC equal to its committed fixture and to the values recorded here."""
import dataclasses
import itertools
import json
from pathlib import Path

import pytest

from flymon.brain import r_rules, s_rules
from flymon.brain import t_records as TR
from flymon.brain import t_rules as R
from flymon.brain.h3_store import canonical
from flymon.brain.t_spec import SPEC
from tests.brain.t_fixtures import SHA_L, SHA_NONE, counts, ref_rows, rest_rows

ROOT = Path(__file__).resolve().parents[2]
READOUT = {"A": "MBON13", "P": "MBON05"}


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
    assert seen == set(R.BANDS)
    assert R.read_band is r_rules.read_band and R.g_fail_s is s_rules.g_fail_s and R.gate2 is s_rules.gate2


@pytest.mark.parametrize("n,c,f,g,band", [
    (13, 10, 2, False, R.SELECTED), (13, 11, 2, False, R.B_NC), (8, 8, 5, False, R.B_TB), (10, 8, 5, False, R.B_NC),
    (11, 8, 5, False, R.SELECTED), (11, 10, 5, False, R.B_NC), (12, 10, 5, False, R.SELECTED),
    (13, 8, 1, False, R.B_FA), (13, 8, 2, False, R.SELECTED), (13, 8, 1, True, R.B_PG), (11, 10, 5, True, R.B_NC)])
def test_boundaries(n, c, f, g, band):
    assert R.read_band(n, c, f, g, 21, 43, max(f, 4), SPEC)["band"] == band


@pytest.mark.parametrize("nb,na", [(20, 43), (22, 43), (21, 42), (21, 44), (21, 32)])
def test_counts_other_than_21_43_are_not_read(nb, na):
    assert R.read_band(13, 8, 5, False, nb, na, 5, SPEC)["band"] == R.NOT_READ


def test_net_drop_2_vs_3_on_t():
    cp = [dict(key=f"a|{i}", axis="a", punish_pass=i < 6) for i in range(12)]
    lp2 = [dict(key=f"a|{i}", axis="a", punish_pass=i < 4) for i in range(12)]
    lp3 = [dict(key=f"a|{i}", axis="a", punish_pass=i < 3) for i in range(12)]
    assert R.g_fail_s(lp2, cp, SPEC)["g_fail"] is False and R.g_fail_s(lp3, cp, SPEC)["g_fail"] is True


def _blk(**kw):
    return dict(dict(code_key=SPEC.r_shared_key), **kw)


R_DOC = dict(repro=_blk(passed=True), gate1=_blk(outcome="PASS"))
CLEAN = dict(tracked=True, dirty=False)


def test_reuse_needs_rs_repro_and_gate1_only():
    assert R.reuse(R_DOC, CLEAN, SPEC.r_shared_key, SPEC)["outcome"] == R.PASS
    out = R.reuse(R_DOC, CLEAN, "f" * 64, SPEC)
    assert out["outcome"] == R.STOP_REUSE and "공유 측정 키" in out["sentence"] and "T는 R 관문을" in out["sentence"]
    assert R.reuse(R_DOC, dict(tracked=True, dirty=True), SPEC.r_shared_key, SPEC)["outcome"] == R.STOP_REUSE
    for b, bad in (("repro", _blk(passed=False)), ("gate1", _blk(outcome="STOP_STRENGTH_LEVER")),
                   ("gate1", _blk(outcome="PASS", code_key="x")), ("repro", None)):
        doc = dict(R_DOC, **{b: bad}) if bad is not None else {k: v for k, v in R_DOC.items() if k != b}
        out = R.reuse(doc, CLEAN, SPEC.r_shared_key, SPEC)
        assert out["outcome"] == R.STOP_REUSE and any(b in w for w in out["reasons"]), b


def test_set_outcome():
    assert R.set_outcome(dict(status="OK", n_b=21, n_a=43))["outcome"] == R.PASS
    out = R.set_outcome(dict(status=R.STOP_SET_SHORT, n_b=21, n_a=40))
    assert out["outcome"] == R.STOP_SET_SHORT
    assert out["sentence"] == "넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다(21·40쌍)."


def _side(a, p, rest=0, edges=0, sha=SHA_NONE):
    return TR.z_side(ref_rows(a, p, edges, sha), rest_rows(len(a), rest, edges, sha), READOUT, SPEC)


def test_z_repro_needs_every_bit_of_h4s_z():
    side = _side(counts(96, 8), counts(96, 20))
    h4 = SPEC.z_h4_dict()
    exact = dict(side, z={k: [v[0], v[1]] for k, v in h4.items()})
    assert R.z_repro(exact, 96, SHA_NONE, SPEC)["outcome"] == R.PASS
    off = dict(side, z=dict(exact["z"], A=[h4["A"][0], h4["A"][1] + 1e-15]))
    out = R.z_repro(off, 96, SHA_NONE, SPEC)
    assert out["outcome"] == R.STOP_Z_REPRO
    assert out["sentence"].startswith("편집 없는 엔진에서 기준 집합 z가 블록 h4 값을 재현하지 못했다(A 10.78125 / ")
    assert out["sentence"].endswith(") — z 절차 결함.")
    none_z = dict(exact, z=None, why="readout MBON13: zero SD")
    assert R.z_repro(none_z, 96, SHA_NONE, SPEC)["outcome"] == R.STOP_Z_REPRO
    assert R.z_repro(exact, 96, "sha-other", SPEC)["outcome"] == R.INVALID
    assert R.z_repro(dict(exact, edit_edges=[2]), 96, SHA_NONE, SPEC)["outcome"] == R.INVALID
    assert R.z_repro(dict(exact, n_rest=95), 96, SHA_NONE, SPEC)["outcome"] == R.INVALID


@pytest.mark.parametrize("a,p,outcome,failed", [
    (counts(96, 5), counts(96, 30), R.PASS, []),                          # MBON13 median Δ 7 (5..9)
    (counts(96, 3), counts(96, 30), R.PASS, []),                          # median Δ exactly 5
    (counts(96, 2), counts(96, 30), R.STOP_Z_DEGENERATE, ["MBON13"]),     # median Δ 4
    (counts(96, 10, zeros=24), counts(96, 30), R.PASS, []),               # zero share exactly 0.25
    (counts(96, 10, zeros=25), counts(96, 30), R.STOP_Z_DEGENERATE, ["MBON13"]),
    (counts(96, 10), [0] * 96, R.STOP_Z_DEGENERATE, ["MBON05"]),          # silent P type: guard and zero SD
])
def test_z_lever_guard_boundaries(a, p, outcome, failed):
    side = _side(a, p, edges=2, sha=SHA_L)
    out = R.z_lever(side, 96, READOUT, SPEC)
    assert out["outcome"] == outcome and out.get("failed", []) == failed


def test_z_lever_zero_sd_and_sentence():
    side = _side([0] * 96, counts(96, 30), edges=2, sha=SHA_L)
    assert side["z"] is None and "zero SD" in side["why"]
    out = R.z_lever(side, 96, READOUT, SPEC)
    assert out["outcome"] == R.STOP_Z_DEGENERATE
    assert out["sentence"] == ("APL→MBON05 제거 아래 기준 집합에서 판독 MBON13이 반응성 가드를 넘지 못했다(Δ 중앙값 0.0, "
                               "0 비율 1.000) — z를 정할 수 없다.")
    assert R.z_lever(dict(side, edit_edges=[0]), 96, READOUT, SPEC)["outcome"] == R.INVALID


def _act(per: dict, edges: int, sha: str, n_seeds: int = 8) -> dict:
    return {o: dict(frac=[v] * n_seeds, max_win=[3] * n_seeds, edit_edges=[edges], csc_sha256=[sha])
            for o, v in per.items()}


def test_gate1s_band_and_sentence():
    ods = {f"M{i}|T": 0.05 for i in range(55)}
    rec = TR.gate1s_record(_act(ods, 2, SHA_L), _act(ods, 0, SHA_NONE), SPEC)
    assert R.gate1s(rec, SPEC)["outcome"] == R.PASS
    edge = dict(ods, **{"M0|T": 0.03, "M1|T": 0.15})
    assert R.gate1s(TR.gate1s_record(_act(edge, 2, SHA_L), _act(ods, 0, SHA_NONE), SPEC), SPEC)["outcome"] == R.PASS
    bad = dict(ods, **{"M3|T": 0.029, "M4|T": 0.2})
    out = R.gate1s(TR.gate1s_record(_act(bad, 2, SHA_L), _act(ods, 0, SHA_NONE), SPEC), SPEC)
    assert out["outcome"] == R.STOP_STRENGTH_LEVER and out["failed"] == ["M3|T", "M4|T"]
    assert out["sentence"] == ("APL→MBON05 제거 아래에서 E-grid k2-norm s 1.0이 KC 활성 자격(T 세트 냄새 M3|T 0.0290; "
                               "M4|T 0.2000 ∉ [0.03, 0.15])을 잃었다.")
    assert R.gate1s(TR.gate1s_record(_act(ods, 1, SHA_L), _act(ods, 0, SHA_NONE), SPEC), SPEC)["outcome"] == R.INVALID
    few = {o: v for o, v in list(ods.items())[:54]}
    assert R.gate1s(TR.gate1s_record(_act(few, 2, SHA_L), _act(few, 0, SHA_NONE), SPEC), SPEC)["outcome"] == R.INVALID
    assert R.gate1s(TR.gate1s_record(_act(ods, 2, SHA_L, 7), _act(ods, 0, SHA_NONE, 7), SPEC),
                    SPEC)["outcome"] == R.INVALID


def _summ(tb, edges, sha, reasons=()):
    return dict(reasons=list(reasons), edit_edges=[edges], csc_sha256=sha,
                aggregate=dict(testable_b=tb, F_a=1, naive_a=2, n_b=21, n_a=18))


def test_gate3_order():
    L, C = _summ(12, 2, SHA_L), _summ(7, 0, SHA_NONE)
    assert R.gate3(L, C, SPEC, SHA_NONE, 16)["outcome"] == R.PASS
    out = R.gate3(None, _summ(6, 0, SHA_NONE), SPEC, SHA_NONE, 16)
    assert out["outcome"] == R.STOP_EVEN_REPRO and out["sentence"] == ("R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 "
                                                                     "못했다(L 16, C 6).")
    assert R.gate3(L, _summ(7, 0, SHA_NONE, ["x"]), SPEC, SHA_NONE, 16)["outcome"] == R.STOP_EVEN_REPRO
    out = R.gate3(_summ(10, 2, SHA_L), C, SPEC, SHA_NONE, 16)
    assert out["outcome"] == R.STOP_EVEN_LOW_LEVER and "(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 1/18)" in out["sentence"]
    assert R.gate3(_summ(11, 2, SHA_L), C, SPEC, SHA_NONE, 16)["outcome"] == R.PASS
    assert R.gate3(_summ(12, 1, SHA_L), C, SPEC, SHA_NONE, 16)["outcome"] == R.INVALID


F = dict(n=14, c=8, f_a=3, naive_a=4, T=103, k_even=12, pb_L=15, pb_C=18, pa_L=20, pa_C=22, d_b=3, d_a=2,
         rho1="0.700", rho2="0.900", why="x", b=21, a=40, val="A 1 / 2", type="MBON13", med="3.0", zero="0.250",
         cond="x", l=16, tb=10, c_even=7, k=1, label="NO_LEARNING", l1="0.100", l2="0.200", l1L="1", l1C="2", l2L="3",
         l2C="4")


def test_sentences_verbatim():
    for o in (R.STOP_REUSE, R.STOP_SET_SHORT, R.STOP_Z_REPRO, R.STOP_Z_DEGENERATE, R.STOP_STRENGTH_LEVER,
              R.STOP_EVEN_REPRO, R.STOP_EVEN_LOW_LEVER, R.STOP_C_EVEN_MISMATCH, R.STOP_PUNISH_BROKEN,
              R.STOP_P_REFERENCE, R.STOP_PUNISH_WEAKENED, R.B_TB, R.B_PG, R.B_FA, R.SELECTED):
        assert "{" not in R.sentence(o, F), o
    assert R.sentence(R.B_TB, F) == (
        "APL→MBON05 제거가 넓힌 상대 풀 판정 세트(생성원 턴 0–103)에서 14/21로 지렛대 없는 같은 세트 8/21보다 오르지 않았다"
        "(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 결과는 "
        "R·S 그대로).")
    assert R.sentence(R.B_NC, dict(F, reason=R.MARGIN)) == (
        "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다 (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_NC, dict(F, reason=R.C_ABOVE_BAR)).endswith(" (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_PG, F) == s_rules.sentence(s_rules.B_PG, F)
    assert R.sentence(R.B_FA, F) == ("M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다(14/21 대 8/21, F_a 3/43, "
                                     "naive_a 4). 다음 병목은 F_a(순진 균형 (a) 쌍)다.")
    assert R.sentence(R.SELECTED, F) == (
        "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 때, "
        "판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–103, 기존 키·사구체 중복 제외, 모든 행이 POOL 밖 상대 타입 "
        "조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다(14/21 대 8/21, "
        "여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 0.700·ℓ_r2 0.900 ≥ 0.5(약화 정도 기록), "
        "짝수 재판독 12/21, 판정 시드 24_400_xxx). 판정 세트의 상대는 1세대 기본 폼으로 넓힌 풀이며, 새 키는 POOL에 없는 "
        "상대 타입 조합(12개 중)에서 나온다 — 과제 풀 안의 시험 가능성은 R·S가 마지막이다. (b) 21쌍은 9개 상대 타입 조합 "
        "쌍에 몰려 있어 쌍끼리 독립이 아니다. 지렛대는 Q 결과를 보고 골랐고, z 규칙은 R·S 판정 결과를 본 뒤 정했다(T.0). "
        "이 판정은 R(뒤에 가드 변경)·S(뒤에 z 규칙)에 이은 같은 지렛대의 세 번째 판정이다. 작동 특성은 T 세트 조건부 값이며 "
        "Q → R → S → T 전체 절차의 오선택률이 아니다. 실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 "
        "아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 "
        "이것만으로 정당화하지 않는다.")
    with pytest.raises(KeyError):
        R.sentence(R.B_FA, {})


DECLARED_S6 = {("null", 0.02, 0.02): 0.034, ("null", 0.05, 0.05): 0.145, ("null", 0.1, 0.1): 0.281,
               ("null", 0.15, 0.15): 0.362, ("harm", 0.1, 0.02): 0.752, ("harm", 0.15, 0.02): 0.949,
               ("harm", 0.2, 0.05): 0.961, ("harm", 0.1, 0.05): 0.549}


def test_independent_oc_is_s6s_with_ts_notes():
    oc = R.oc(SPEC)
    got = {(g["kind"], g["a_pf"], g["a_fp"]): round(g["p"], 3) for g in oc["g_fail"]}
    assert got == DECLARED_S6
    assert len(oc["notes"]) == 3 and "9개 상대 타입 조합 쌍" in oc["notes"][1] and "Q → R → S → T" in oc["notes"][2]
    assert oc["sha256"] == R.oc(SPEC)["sha256"] and len(oc["rows"]) == 4 * 3 * 13 * 9


# Recorded while planning (T.9.7: the fixture is the declaration; these lines name its headline values).
CLUSTER_G = {("null", 0.1, 0.1): 0.401, ("harm", 0.1, 0.02): 0.685, ("harm", 0.1, 0.05): 0.571}
CLUSTER_P_N_GE_11 = {"0.33": 0.097, "0.5": 0.498, "0.6": 0.776, "0.7": 0.939}


@pytest.fixture(scope="module")
def cluster():
    return R.oc_cluster(SPEC)


def test_cluster_oc_equals_its_fixture(cluster):
    fx = json.loads((ROOT / SPEC.oc_cluster_fixture).read_text())
    assert canonical(fx) == canonical(cluster) and fx["sha256"] == cluster["sha256"]


def test_cluster_oc_headline_values(cluster):
    assert (cluster["seed"], cluster["draws"], cluster["icc"]) == (20261005, 100_000, 0.3)
    assert cluster["clusters_b"] == [1, 3, 3, 3, 3, 1, 1, 2, 4] and cluster["clusters_a"] == [7, 8, 2, 1, 5, 9, 9, 2]
    got = {(g["kind"], g["a_pf"], g["a_fp"]): round(g["p"], 3) for g in cluster["g_fail"]}
    assert got == CLUSTER_G
    assert {(g["kind"], g["a_pf"], g["a_fp"]): round(g["p_independent"], 3) for g in cluster["g_fail"]} == {
        k: DECLARED_S6[k] for k in CLUSTER_G}
    for q, d in cluster["n_dist"].items():
        assert sum(d["cluster"]) == pytest.approx(1.0) and round(sum(d["cluster"][11:]), 3) == CLUSTER_P_N_GE_11[q]
    assert len(cluster["rows"]) == 4 * 3 * 13 * 4
    for row in cluster["rows"][::37]:
        assert sum(row["P"].values()) == pytest.approx(1.0) and set(row["P"]) == set(R.BANDS)
    assert all(r["P"][R.B_PG] == 0 for r in cluster["rows"] if r["g_kind"] == "none")


def test_cluster_oc_depends_on_icc_and_seed():
    small = dataclasses.replace(SPEC, oc_cluster_draws=2000)
    a, b = R.oc_cluster(small), R.oc_cluster(dataclasses.replace(small, oc_cluster_seed=1))
    assert a["sha256"] != b["sha256"] and a["sha256"] == R.oc_cluster(small)["sha256"]
    lo = R.oc_cluster(dataclasses.replace(small, oc_cluster_icc=0.01))
    assert abs(lo["g_fail"][0]["p"] - lo["g_fail"][0]["p_independent"]) < 0.05


def test_t_rules_literals_are_only_0_and_1():
    import ast
    src = (ROOT / "flymon/brain/t_rules.py").read_text()
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1}, nums
```

```python
# tests/brain/test_t_records.py
"""T's records: z_side is h4_rules.z_constants (mean, population SD) and H.4's guard statistic on the readout types,
with the zero-SD types named; z_ratios are σ_lever / σ_h4; P_L read on z_lever is a record beside the gate (ℓ_L under
z_lever over ℓ_C under h4 z); the cluster table counts passes per T.9.1 cluster; the α-fixed sensitivity reads L's raw
on h4 z; gate ②'s ratio and OC are S's functions."""
import numpy as np
import pytest

from flymon.brain import s_records
from flymon.brain import t_records as TR
from flymon.brain.p_rules import p_judge
from flymon.brain.t_spec import SPEC
from tests.brain.r_fixtures import Z, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, p_rows
from tests.brain.t_fixtures import counts, ref_rows, rest_rows

READOUT = {"A": "MBON13", "P": "MBON05"}


def test_z_side_is_population_mean_sd_and_the_guard():
    a, p = counts(96, 8), counts(96, 20, zeros=10)
    side = TR.z_side(ref_rows(a, p), rest_rows(96, rest=2), READOUT, SPEC)
    assert side["z"] == {"A": [float(np.mean(a)), float(np.std(a))], "P": [float(np.mean(p)), float(np.std(p))]}
    g = side["guard"]
    assert g["MBON13"]["median_delta"] == float(np.median(np.array(a) - 2)) and g["MBON13"]["zero_share"] == 0.0
    assert g["MBON05"]["zero_share"] == pytest.approx(10 / 96) and g["MBON05"]["passes"]
    assert (side["n_ref"], side["n_rest"], side["edit_edges"], side["csc_sha256"], side["zero_sd"]) == (
        96, 96, [0], ["sha-C"], [])
    flat = TR.z_side(ref_rows([5] * 96, p), rest_rows(96), READOUT, SPEC)
    assert flat["z"] is None and flat["zero_sd"] == ["MBON13"] and "zero SD" in flat["why"]
    assert TR.z_ratios({"A": [6.0, 4.0], "P": [80.0, 40.0]}, {"A": (10.0, 8.0), "P": (26.0, 20.0)}) == {"A": 0.5,
                                                                                                         "P": 2.0}


def test_p_zlever_is_a_record_beside_the_h4_gate():
    rl = p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2)
    rc = p_rows(SPEC.p_c, "none", 16)
    res_c = p_judge(rc, Z, C1, SPEC.p_c)
    zl = {"A": (10.0, 4.5), "P": (26.0, 19.0)}                    # σA halved: ℓ_L doubles on z_lever
    rec = TR.p_zlever(rl, zl, C1, res_c, SPEC)
    res_l = p_judge(rl, Z, C1, SPEC.p)
    for d in SPEC.p.directions:
        assert rec["ell"][d] == pytest.approx(2 * res_l["directions"][d]["ell"])
        assert rec["ratio_to_C_h4"][d] == pytest.approx(rec["ell"][d] / res_c["directions"][d]["ell"])
    assert rec["label"] == "LEARNS_CONFIRMATORY"
    assert TR.ratio is s_records.ratio and TR.gate2_oc is s_records.gate2_oc


def test_clusters_count_per_cluster_and_condition():
    pairs = {"L": [dict(key="b|1|x|y", testable=True, reward_pass=True, punish_pass=False),
                   dict(key="a|2|x|y", testable=False, reward_pass=True, punish_pass=True)],
             "C": [dict(key="b|1|x|y", testable=False, reward_pass=False, punish_pass=True)]}
    labels = {"b|1|x|y": "GROUND* 대 NORMAL", "a|2|x|y": "FIRE*"}
    assert TR.clusters(pairs, labels) == {
        "GROUND* 대 NORMAL": {"L": dict(n=1, testable=1, reward_pass=1, punish_pass=0),
                              "C": dict(n=1, testable=0, reward_pass=0, punish_pass=1)},
        "FIRE*": {"L": dict(n=1, testable=0, reward_pass=1, punish_pass=1)}}


def test_alpha_fixed_reads_l_on_h4_z():
    from flymon.brain.r_records import cond_summary
    rows = fake_rows(21, 43, turn0=0)
    keys = [key_of(r) for r in rows]
    seeds = SPEC.judge_seeds()
    plan = {k: (True, True, False) for k in keys[:14]}
    gl = got_for(rows, plan, edges=2, edit=SPEC.lever_edit, sha="sha-L")
    gc = got_for(rows, {k: (True, True, False) for k in keys[:8]})
    C = cond_summary(gc, SPEC.cond("C"), SPEC, Z, keys, seeds)
    out = TR.alpha_fixed(gl, C, keys, seeds, Z, SPEC)
    assert (out["n"], out["c"], out["band"]) == (14, 8, "B_Fa") and out["g_fail"]["g_fail"] is False
    assert "판정 아님" in out["note"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_t_rules.py tests/brain/test_t_records.py -q`
Expected: collection errors `ImportError: cannot import name 't_records'` / `'t_rules'`.

- [ ] **Step 3: Write `t_rules` and `t_records`**

```python
"""T's decisions (T.1-T.4, T.6, T.7 as replaced by T.9). The judgement code is the authoritative source: the bands are
r_rules.read_band itself (T.4 = S.4 = R.3's order) read with T's numbers (n_a 43), G_fail_S is S's (s_rules.g_fail_s,
net drop ≥ 3 per axis) and gate ②'s order is S's (s_rules.gate2, read on block h4's z, T.9.3). New here: the reuse of
R's repro and gate ① (T.3 1), the set outcome (T.2 / T.9.1), the z gates (T.1, T.9.4, T.9.7: STOP_Z_REPRO, then
STOP_Z_DEGENERATE), the gate-① supplement on the T set's odours (T.9.1), gate ③ with R's C reproduction first (T.9.2),
the sentences (T.7 as replaced by T.9.5 / T.9.7, 〈…〉 → {field}) and the two operating characteristics (T.6, T.9.6,
T.9.7). Every number is a field of the TSpec passed in; no OC value ever changes a band."""
from __future__ import annotations

import hashlib

import numpy as np

from ..agent import e_rules
from . import r_rules, s_rules
from .h3_store import canonical

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (r_rules.PASS, r_rules.INVALID, r_rules.NOT_READ, r_rules.SEALED,
                                                       r_rules.READ, r_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = s_rules.STOP_REUSE, s_rules.STOP_SET_SHORT
STOP_Z_REPRO, STOP_Z_DEGENERATE = "STOP_Z_REPRO", "STOP_Z_DEGENERATE"
STOP_STRENGTH_LEVER = r_rules.STOP_STRENGTH_LEVER
STOP_EVEN_REPRO, STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH = ("STOP_EVEN_REPRO", r_rules.STOP_EVEN_LOW_LEVER,
                                                              r_rules.STOP_C_EVEN_MISMATCH)
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (s_rules.STOP_PUNISH_BROKEN, s_rules.STOP_P_REFERENCE,
                                                              s_rules.STOP_PUNISH_WEAKENED)
SELECTED, B_TB, B_FA, B_NC, B_PG = r_rules.SELECTED, r_rules.B_TB, r_rules.B_FA, r_rules.B_NC, r_rules.B_PG
BANDS = r_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = r_rules.C_ABOVE_BAR, r_rules.BELOW_BAR, r_rules.MARGIN
read_band = r_rules.read_band                      # T.4 = S.4: R.3's order (plan Reading 9)
g_fail_s = s_rules.g_fail_s                        # T.4: S's guard, each condition on its own z
gate2 = s_rules.gate2                              # T.3 6 / T.9.3: S's order, read on block h4's z


# ================================================================ gates before the set is used
def reuse(r_doc: dict, r_git: dict, shared_key: str, spec) -> dict:
    """T.3 1: R's repro and gate ① are reused iff the shared measurement key equals R's, R's summary is tracked and
    clean, and both blocks exist with R's key and passed. Else STOP_REUSE (plan Reading 8)."""
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
    """T.2: STOP_SET_SHORT when the turns ran out (the declared-value check is the runner's refusal)."""
    if js["status"] == STOP_SET_SHORT:
        return dict(outcome=STOP_SET_SHORT, sentence=sentence(STOP_SET_SHORT, dict(b=js["n_b"], a=js["n_a"])))
    return dict(outcome=PASS)


def _zfmt(z: dict) -> str:
    return ", ".join(f"{k} {v[0]!r} / {v[1]!r}" for k, v in sorted(z.items()))


def _side_reasons(side: dict, name: str, edges: int, n: int, sha=None) -> list:
    bad = []
    if side["edit_edges"] != [edges]:
        bad.append(f"{name}: edges {side['edit_edges']}, declared {edges}")
    if sha is not None and side["csc_sha256"] != [sha]:
        bad.append(f"{name}: CSC {side['csc_sha256']} is not the unedited engine's {sha}")
    if (side["n_ref"], side["n_rest"]) != (n, n):
        bad.append(f"{name}: {side['n_ref']} reference / {side['n_rest']} rest presentations, declared {n}")
    return bad


def z_repro(none: dict, n: int, repro_sha: str, spec) -> dict:
    """T.1 / T.9.6: the unedited engine's reference-set z must equal block h4's C3 z bit for bit (float equality of
    mean and SD), else STOP_Z_REPRO; wrong edges, CSC or presentation counts are INVALID."""
    bad = _side_reasons(none, "none", 0, n, repro_sha)
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    want = spec.z_h4_dict()
    got = None if none["z"] is None else {k: (float(v[0]), float(v[1])) for k, v in none["z"].items()}
    if got != want:
        val = none["why"] if got is None else _zfmt(got)
        return dict(outcome=STOP_Z_REPRO, reasons=[], sentence=sentence(STOP_Z_REPRO, dict(val=val)))
    return dict(outcome=PASS, reasons=[])


def z_lever(lever: dict, n: int, readout: dict, spec) -> dict:
    """T.9.4: under the lever every readout type must pass H.4's guard (median(read − same-seed rest) ≥ 5 and zero
    share ≤ 0.25) and have a nonzero SD (an SD of 0 is the same STOP), else STOP_Z_DEGENERATE naming every failing
    type; z_lever is then the lever's reference-set z."""
    bad = _side_reasons(lever, "lever", spec.lever_edges, n)
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    fails = [t for t in readout.values() if not lever["guard"][t]["passes"] or t in lever["zero_sd"]]
    if fails:
        g = lever["guard"]
        return dict(outcome=STOP_Z_DEGENERATE, reasons=[], failed=fails, sentence=sentence(STOP_Z_DEGENERATE, dict(
            type="·".join(fails), med="·".join(f"{g[t]['median_delta']:.1f}" for t in fails),
            zero="·".join(f"{g[t]['zero_share']:.3f}" for t in fails))))
    return dict(outcome=PASS, reasons=[])


def gate1s(rec: dict, spec) -> dict:
    """T.9.1: every T-set odour's KC activity median under the lever inside valid_band, else STOP_STRENGTH_LEVER;
    INVALID when the lever did not change exactly lever_edges edges, C was edited, or the odour / seed counts differ."""
    bad = []
    if rec["edit_edges"] != [spec.lever_edges] or rec["edit_edges_none"] != [0]:
        bad.append(f"edges lever {rec['edit_edges']} / none {rec['edit_edges_none']}, declared {spec.lever_edges} / 0")
    if rec["n_odours"] != spec.n_set_odours or rec["n_seeds"] != [len(spec.kc_seeds())]:
        bad.append(f"{rec['n_odours']} odours × {rec['n_seeds']} seeds, declared {spec.n_set_odours} × "
                   f"{len(spec.kc_seeds())}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    lo, hi = spec.valid_band
    failed = sorted(o for o, v in rec["per_odour"].items() if not lo <= v <= hi)
    if failed:
        cond = "T 세트 냄새 " + "; ".join(f"{o} {rec['per_odour'][o]:.4f}" for o in failed) + f" ∉ [{lo}, {hi}]"
        return dict(outcome=STOP_STRENGTH_LEVER, reasons=[], failed=failed,
                    sentence=sentence(STOP_STRENGTH_LEVER, dict(cond=cond)))
    return dict(outcome=PASS, reasons=[], failed=[])


def gate3(L, C: dict, spec, repro_sha: str, l_h4) -> dict:
    """T.9.2: C (R's even raw, h4 z) must reproduce c_even_expected first — else STOP_EVEN_REPRO and L is not measured
    (L may be None; l_h4 = R's L even raw read on h4 z, for the sentence); then R's gate ③ rule on L (re-measured,
    z_lever) and C: validity, STOP_C_EVEN_MISMATCH (defensive, unreachable after the reproduction, T.9.6),
    STOP_EVEN_LOW_LEVER when L's testable_b < bar_b."""
    c = (C.get("aggregate") or {}).get("testable_b")
    if C["reasons"] or c != spec.c_even_expected:
        return dict(outcome=STOP_EVEN_REPRO, reasons=[f"C: {m}" for m in C["reasons"]], c_even=c,
                    sentence=sentence(STOP_EVEN_REPRO, dict(l=l_h4, c=c)))
    return r_rules.gate3(L, C, spec, repro_sha)


# ================================================================ the closing sentences (T.7 → T.9.5 / T.9.7)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43)"
SENTENCES = {
    STOP_REUSE: "R 관문 재사용 조건(T.3 1)이 깨졌다({why}). T는 R 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 쓰지 않고 "
                "멈춘다 — 사용자 몫.",
    STOP_SET_SHORT: "넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다({b}·{a}쌍).",
    STOP_Z_REPRO: "편집 없는 엔진에서 기준 집합 z가 블록 h4 값을 재현하지 못했다({val}) — z 절차 결함.",
    STOP_Z_DEGENERATE: "APL→MBON05 제거 아래 기준 집합에서 판독 {type}이 반응성 가드를 넘지 못했다(Δ 중앙값 {med}, 0 비율 "
                       "{zero}) — z를 정할 수 없다.",
    STOP_STRENGTH_LEVER: r_rules.SENTENCES[STOP_STRENGTH_LEVER],
    STOP_EVEN_REPRO: "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L {l}, C {c}).",
    STOP_EVEN_LOW_LEVER: r_rules.SENTENCES[STOP_EVEN_LOW_LEVER],
    STOP_C_EVEN_MISMATCH: r_rules.SENTENCES[STOP_C_EVEN_MISMATCH],
    STOP_PUNISH_BROKEN: s_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: s_rules.SENTENCES[STOP_P_REFERENCE],
    STOP_PUNISH_WEAKENED: s_rules.SENTENCES[STOP_PUNISH_WEAKENED],
    B_TB: "APL→MBON05 제거가 넓힌 상대 풀 판정 세트(생성원 턴 0–{T})에서 {n}/21로 지렛대 없는 같은 세트 {c}/21보다 오르지 "
          "않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 "
          "결과는 R·S 그대로).",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: s_rules.SENTENCES[B_PG],
    B_FA: s_rules.SENTENCES[B_FA],
    SELECTED: "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 "
              "때, 판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–{T}, 기존 키·사구체 중복 제외, 모든 행이 POOL 밖 "
              "상대 타입 조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다"
              "({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 {rho1}·ℓ_r2 "
              "{rho2} ≥ 0.5(약화 정도 기록), 짝수 재판독 {k_even}/21, 판정 시드 24_400_xxx). 판정 세트의 상대는 1세대 기본 폼으로 "
              "넓힌 풀이며, 새 키는 POOL에 없는 상대 타입 조합(12개 중)에서 나온다 — 과제 풀 안의 시험 가능성은 R·S가 "
              "마지막이다. (b) 21쌍은 9개 상대 타입 조합 쌍에 몰려 있어 쌍끼리 독립이 아니다. 지렛대는 Q 결과를 보고 골랐고, "
              "z 규칙은 R·S 판정 결과를 본 뒤 정했다(T.0). 이 판정은 R(뒤에 가드 변경)·S(뒤에 z 규칙)에 이은 같은 지렛대의 세 "
              "번째 판정이다. 작동 특성은 T 세트 조건부 값이며 Q → R → S → T 전체 절차의 오선택률이 아니다. 실제 커넥톰 "
              "간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 "
              "섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristics (T.6, T.9.6, T.9.7)
OC_NOTES = (s_rules.OC_NOTES[0],
            "T.0 / T.6: 독립 가정은 T 세트에서 낙관적이다 — (b) 21쌍은 9개 상대 타입 조합 쌍, (a) 43쌍은 8개 상대 타입 조합에 "
            "몰려 있다. 군집 상관 행(ICC 0.3, 결과 전 고른 예시값)을 옆에 기록한다.",
            "작동 특성은 T 세트 조건부 값이며 Q → R → S → T 전체 절차의 오선택률이 아니다.")


def oc(spec) -> dict:
    """T.6: S.6's exact independent model (s_rules.oc: r_rules._cell, G_fail_S from net drop ≥ 3, R's guard beside it)
    on n_b 21 · n_a 43, with T's notes."""
    table = {k: v for k, v in s_rules.oc(spec).items() if k != "sha256"}
    table["notes"] = list(OC_NOTES)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())


def _g_axis_mc(rng, sizes, a_pf: float, a_fp: float, kappa: float, draws: int, drop: int) -> np.ndarray:
    """Per cluster (pass→fail, fail→pass, same) ~ Dirichlet(κ·(a_pf, a_fp, 1 − a_pf − a_fp)), its pairs multinomial
    (as two binomials); True where the axis' net drop Σ pf − Σ fp ≥ drop."""
    alpha = kappa * np.array([a_pf, a_fp, 1.0 - a_pf - a_fp])
    pf, fp = np.zeros(draws, np.int64), np.zeros(draws, np.int64)
    for n_k in sizes:
        p = rng.dirichlet(alpha, size=draws)
        k_pf = rng.binomial(int(n_k), p[:, 0])
        rest = 1.0 - p[:, 0]
        q_fp = np.clip(np.divide(p[:, 1], rest, out=np.zeros(draws), where=rest > 0), 0.0, 1.0)
        pf += k_pf
        fp += rng.binomial(int(n_k) - k_pf, q_fp)
    return (pf - fp) >= drop


def _cell(pn, pf, c: int, na: int, g: float, spec) -> dict:
    P = {b: 0.0 for b in BANDS}
    for i, wi in enumerate(pn):
        if not wi:
            continue
        for j, wj in enumerate(pf):
            if not wj:
                continue
            for gv, wg in ((False, 1 - g), (True, g)):
                if wg:
                    P[read_band(i, c, j, gv, spec.n_b, spec.n_a, na, spec)["band"]] += wi * wj * wg
    return P


def oc_cluster(spec) -> dict:
    """T.6 / T.9.6 / T.9.7: the cluster-correlated rows beside the independent model — Monte Carlo with one
    default_rng(oc_cluster_seed), consumed in this order: the g rows ((b) clusters then (a) clusters per row), then per
    q in oc_q the (b) testable draws (per cluster Beta(κq, κ(1 − q)), then Binomial) and the (a) draws (per cluster
    Beta, a uniformly random order of the 43 (a) pairs, Bernoulli per pair; F_a at naive_a = the first naive_a pairs).
    κ = (1 − ICC) / ICC; c fixed; G_fail_S independent of n and F_a with the cluster model's probability; each cell read
    through read_band. The clusters are T.9.1's table (spec.clusters_b / clusters_a)."""
    rng = np.random.default_rng(spec.oc_cluster_seed)
    kappa, N = (1.0 - spec.oc_cluster_icc) / spec.oc_cluster_icc, int(spec.oc_cluster_draws)
    sb = [int(n) for *_, n in spec.clusters_b]
    sa = [int(n) for _, n in spec.clusters_a]
    g_rows = []
    for kind, a_pf, a_fp in spec.oc_cluster_g:
        gb = _g_axis_mc(rng, sb, a_pf, a_fp, kappa, N, spec.g_fail_drop)
        ga = _g_axis_mc(rng, sa, a_pf, a_fp, kappa, N, spec.g_fail_drop)
        g_rows.append(dict(kind=kind, a_pf=a_pf, a_fp=a_fp, p=float(np.mean(gb | ga)),
                           p_independent=s_rules.g_prob(a_pf, a_fp, spec)))
    label_a = np.repeat(np.arange(len(sa)), sa)
    n_dist, fa_dist = {}, {}
    for q in spec.oc_q:
        n = np.zeros(N, np.int64)
        for n_k in sb:
            n += rng.binomial(n_k, rng.beta(kappa * q, kappa * (1 - q), size=N))
        n_dist[q] = (np.bincount(n, minlength=spec.n_b + 1) / N).tolist()
        qa = rng.beta(kappa * q, kappa * (1 - q), size=(N, len(sa)))
        order = rng.permuted(np.tile(label_a, (N, 1)), axis=1)
        hit = rng.random((N, label_a.size)) < np.take_along_axis(qa, order, axis=1)
        csum = np.concatenate([np.zeros((N, 1), np.int64), np.cumsum(hit, axis=1)], axis=1)
        fa_dist[q] = {na: (np.bincount(csum[:, na], minlength=na + 1) / N).tolist()
                      for na in range(min(spec.n_a, spec.oc_naive_max) + 1)}
    rows = []
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in range(min(spec.n_a, spec.oc_naive_max) + 1):
                for gr in [dict(kind="none", a_pf=0.0, a_fp=0.0, p=0.0)] + g_rows:
                    rows.append(dict(q=q, c=c, naive_a=na, g_kind=gr["kind"], a_pf=gr["a_pf"], a_fp=gr["a_fp"],
                                     g=gr["p"], P=_cell(n_dist[q], fa_dist[q][na], c, na, gr["p"], spec)))
    table = dict(model="clusters = T.9.1's table; per cluster a shared latent probability: guard (pf, fp, same) ~ "
                       "Dirichlet(κ·(a_pf, a_fp, 1 − a_pf − a_fp)), testable q_k ~ Beta(κq, κ(1 − q)); κ = (1 − ICC) / "
                       "ICC; c fixed; naive (a) pairs a uniformly random subset; G_fail_S independent of n and F_a",
                 icc=spec.oc_cluster_icc, kappa=kappa, draws=N, seed=spec.oc_cluster_seed, clusters_b=sb,
                 clusters_a=sa, g_fail=g_rows,
                 n_dist={str(q): dict(cluster=n_dist[q], independent=_binom(spec.n_b, q)) for q in spec.oc_q},
                 rows=rows, notes=list(OC_NOTES[1:]))
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())


def _binom(n: int, q: float) -> list:
    return [float(x) for x in e_rules._binom(n, q)]
```

```python
"""T's numbers that no R / S module returns — records and the values the z gates read; no engine.
- z_side (T.1, T.9.4): one engine variant's reference-set z (h4_rules.z_constants over the readout types' summed
  counts, ddof 0; None with the reason when an SD is 0) and H.4's readout guard per readout type
  (h3_rules.mbon_type_stats: median(read − same-seed rest), zero share) with the edges, CSC and counts.
- z_ratios: σ_lever / σ_h4 per readout (T.9.4's record).
- gate1s_record (T.9.1): per-odour KC activity medians (e_rules.odour_activity) under the lever and unedited, on the
  T set's odours, with edges, CSC and seed counts.
- p_zlever (T.9.3, record only): P_L's p_judge read on z_lever and ℓ_L(z_lever) / ℓ_C(h4 z) per direction.
- clusters (T.6): testable / reward / punish passes per T.9.1 cluster and condition.
- alpha_fixed (T.9.2, record only — "α 고정 민감도"): L's judgement raw (α chosen under z_lever) read on h4 z: n, F_a,
  naive_a, G_fail_S against C and the band that reading would give.
- ratio / gate2_oc: s_records' (gate ② on h4 z, T.9.3)."""
from __future__ import annotations

from ..agent import e_rules
from . import r_records, s_records, s_rules
from .h3_rules import mbon_type_stats
from .h4_rules import z_constants
from .p_rules import p_judge

ratio = s_records.ratio
gate2_oc = s_records.gate2_oc


def z_side(ref: list, rest: list, readout: dict, spec) -> dict:
    rb = {int(r["seed"]): r for r in rest}
    guard = {t: mbon_type_stats(ref, rb, t, spec.z_guard_med_min, spec.z_guard_zero_max) for t in readout.values()}
    try:
        z, why = z_constants(ref, readout, spec.z_ddof), None
    except ValueError as e:
        z, why = None, str(e)
    zero_sd = [t for t in readout.values() if len({r["types"][t] for r in ref}) < 2]
    return dict(z=None if z is None else {k: [float(v[0]), float(v[1])] for k, v in z.items()}, why=why, guard=guard,
                zero_sd=zero_sd, n_ref=len(ref), n_rest=len(rest),
                edit_edges=sorted({int(r["edit_edges"]) for r in ref + rest}),
                csc_sha256=sorted({r["csc_sha256"] for r in ref + rest}))


def z_ratios(z_lever: dict, z_h4: dict) -> dict:
    return {k: float(z_lever[k][1]) / float(z_h4[k][1]) for k in z_h4}


def gate1s_record(act_l: dict, act_n: dict, spec) -> dict:
    per = e_rules.odour_activity({o: act_l[o]["frac"] for o in act_l})
    per_n = e_rules.odour_activity({o: act_n[o]["frac"] for o in act_n})
    lo, hi = spec.valid_band
    return dict(per_odour=per, per_odour_none=per_n, n_odours=len(per),
                n_seeds=sorted({len(v["frac"]) for v in act_l.values()}),
                outside=sorted(o for o, v in per.items() if not lo <= v <= hi),
                outside_none=sorted(o for o, v in per_n.items() if not lo <= v <= hi),
                min=min(per.values()), max=max(per.values()),
                edit_edges=sorted({int(e) for v in act_l.values() for e in v["edit_edges"]}),
                edit_edges_none=sorted({int(e) for v in act_n.values() for e in v["edit_edges"]}),
                csc_sha256=sorted({s for v in act_l.values() for s in v["csc_sha256"]}),
                csc_sha256_none=sorted({s for v in act_n.values() for s in v["csc_sha256"]}))


def p_zlever(rows_l: list, z_lever: dict, c1: float, res_c: dict, spec) -> dict:
    res = p_judge(rows_l, z_lever, c1, spec.p)
    d = res.get("directions") or {}
    dc = res_c.get("directions") or {}
    return dict(label=res.get("label"), outcome=res.get("outcome"),
                ell={k: v.get("ell") for k, v in d.items()},
                ratio_to_C_h4={k: (d[k]["ell"] / dc[k]["ell"] if k in dc and dc[k].get("ell") else None) for k in d})


def clusters(pairs: dict, labels: dict) -> dict:
    """{cluster: {condition: {n, testable, reward_pass, punish_pass}}} over the judgement pairs (T.6)."""
    out = {}
    for cond, ps in pairs.items():
        for p in ps:
            cl = labels[p["key"]]
            c = out.setdefault(cl, {}).setdefault(cond, dict(n=0, testable=0, reward_pass=0, punish_pass=0))
            c["n"] += 1
            for k in ("testable", "reward_pass", "punish_pass"):
                c[k] += int(bool(p[k]))
    return out


def alpha_fixed(raws_l: list, C: dict, keys: list, seeds: dict, z_h4: dict, spec) -> dict:
    """L's raw (α chosen under z_lever) on h4 z, against C (h4 z) — a record, never the judgement (T.9.2)."""
    L = r_records.cond_summary(raws_l, spec.cond(spec.cond_names[0]), spec, z_h4, keys, seeds)
    aL, aC = L["aggregate"] or {}, C["aggregate"] or {}
    gf = s_rules.g_fail_s(L["pairs"], C["pairs"], spec)
    band = None
    if L["aggregate"] and C["aggregate"]:
        band = s_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                                 aL["naive_a"], spec)["band"]
    return dict(n=aL.get("testable_b"), c=aC.get("testable_b"), F_a=aL.get("F_a"), naive_a=aL.get("naive_a"),
                g_fail=gf, band=band, note="α 고정 민감도: z_lever가 고른 α의 원자료를 h4 z로 읽은 값 — 판정 아님 (T.9.2)")
```

- [ ] **Step 4: Generate the cluster OC fixture once (T.9.7)**

Run (no pool; about a second):

```bash
uv run python -c "from flymon.brain import t_rules; from flymon.brain.h3_store import canonical_pretty; from flymon.brain.t_spec import SPEC; open(SPEC.oc_cluster_fixture, 'w').write(canonical_pretty(t_rules.oc_cluster(SPEC)) + '\n')"
```

Expected: `tests/brain/fixtures/t_oc_cluster.json` (~210 KB) with `"sha256": "83a4d0bdce784536908e3b0e2e00efe8cb6610dec43ee6fca2be4060c7ecffb0"`. The file is written once here and never again (a stage only compares against it). If the sha differs, the NumPy version differs from the planning run (`uv.lock`) — stop and report; do not edit the values in the test.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_t_rules.py tests/brain/test_t_records.py tests/brain/test_s_rules.py tests/brain/test_r_rules.py -q`
Expected: all pass (36 + 4 T tests); the independent OC reproduces S.6's eight values, the cluster OC equals the fixture and reads null 0.401, harm 0.685 / 0.571, P(n ≥ 11) 0.097 / 0.498 / 0.776 / 0.939.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/t_rules.py flymon/brain/t_records.py tests/brain/t_fixtures.py tests/brain/test_t_rules.py tests/brain/test_t_records.py tests/brain/fixtures/t_oc_cluster.json
git commit -m "feat(t): t_rules / t_records — reuse, set, z repro and guard, gate-1 supplement, gate 3 with C repro, T.9.5 sentences, both OCs and the cluster fixture"
```

---

### Task 5: `t_runner` (1) — context, chain and refusals, reuse, set, z, gate ① supplement, smoke, OCs, gate ②, gate ③

**Files:**
- Create: `flymon/brain/t_runner.py` (first part — Task 6 appends the judgement methods to class `Runner`)
- Test: `tests/brain/t_world.py`, `tests/brain/test_t_runner.py`

**Interfaces:**
- Consumes: Tasks 1–4; `r_records` (`cond_summary`, `compare`, `KEEP`), `r_runner.p_items`, `r_pairs` (`row_key`, `even_rows`, `p_reference`), `s_runner.S_HASHED_FILES`, `p_rules.p_judge`, `e_runner.summary_git`, `h3_store` (`ROOT`, `canonical`, `sha256_file`, `git_state`, `code_key`), `h3_spec` (`SPEC.reference`, `make_odors`), `r_measure.R_MEASURE_FILES` / `RMeasurer`; tests reuse `r_fixtures.py` and `s_fixtures.py`.
- Produces: `ORDER`, `GATES`, `GATE2_INVALID`, `EXIT_REFUSE = 2`, `EXIT_KEY = 7`, `T_PIPELINE_FILES`, `T_HASHED_FILES`, `DECISION_FILES`, `JUDGE_MARKER`, `REREAD_MARKER`, `DONE_MARKER`, `git_state()`, `pipeline_key()`, `decision_key()`, `refuse(msg, code)`, `build_ctx(spec, npz) -> dict` (keys `params, readout, z, types, n_kc, enc, even_rows, ref_odors, probe_seeds, ref_strength, ref_window, t_set, set_odours, judgement_rows, clusters, r_even, p_ref, p_inputs, r_doc, r_git, s_doc`), `Runner(measure, zm, ctx, spec, summary_path=None, code=None, tcode=None, pipeline=None, archive_root=None)` with `_z_lever`, `_z_for`, `_m_for`, `m_h4`, `stage_reuse`, `stage_set`, `stage_z`, `stage_gate1s`, `stage_smoke`, `stage_oc`, `stage_gate2_oc`, `stage_gate2(rerun=False)`, `stage_gate3`; block fields read later: `reuse.records.repro_csc_sha256_none`, `z.z_lever`, `z.t_measure_key`, `z.sd_ratio_lever_over_h4`, `smoke.problems`, `oc.independent.sha256`, `oc.cluster.sha256`, `gate2.{label_L, label_C, ratio}`, `gate3.testable_b`.
- `tests/brain/t_world.py`: `SPEC` (T's spec with this world's block-h4 z), `CODE`, `TCODE`, `PIPE`, `ZL`, `COUNTS`, `r_doc()`, `declared_set(**kw)`, `ZScripted`, `Scripted` (with `at(z)`), `World`, `doc()`, `through_gate3(w, m)`.

- [ ] **Step 1: Write the world and the failing test**

```python
"""A scripted T world for the runner tests (no engine, no pool): a fake ctx (R's summary as the reuse source, T's set
as a declared-value dict, the reference set as 48 fake odours, R's even raw as fabricated entries, P's block as
fixture rows), a ZScripted reference / rest measurer whose unedited counts give exactly block h4's z of this world
(A 10 / 9, P 26 / 19 — the world's spec carries that z_h4) and whose lever counts give z_lever A 6 / 3, P 80 / 20, and
a Scripted measurer that is built per z (`at(z)`, as the runner's measure(z) is), whose P arms come from s_fixtures
(real p_rules.p_judge reads them) and whose oracle writes real cache-like files under results/t/cache (the seal and the
judge re-read them; the stored inputs carry the view's z). Pair statistics of r_fixtures' fake oracle do not depend on
z (each phase moves one readout type only), so the same flags read the same under both z."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

from flymon.brain import t_rules
from flymon.brain import t_runner as TR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.t_spec import SPEC as T_SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli
from tests.brain.t_fixtures import ref_rows, rest_rows

SPEC = dataclasses.replace(T_SPEC, z_h4=(("A", Z["A"]), ("P", Z["P"])))
SHA = {"L": "sha-L", "C": "sha-C", "E0": "sha-C"}
CODE = {"key": SPEC.r_shared_key}
TCODE = {"key": "t" * 64}
PIPE = {"key": "p" * 64}
N_REF = 96
ZL = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
COUNTS = {SPEC.no_edit: ([1, 19] * 48, [7, 45] * 48),          # mean 10 / SD 9, mean 26 / SD 19: block h4's z here
          SPEC.lever_edit: ([3, 9] * 48, [60, 100] * 48)}      # z_lever A 6 / 3, P 80 / 20; guard Δ 6 / 80
_CLUSTER = {}


def cluster_oc(spec):
    """t_rules.oc_cluster once per test session (10⁵ draws take about a second)."""
    if "v" not in _CLUSTER:
        _CLUSTER["v"] = REAL_OC_CLUSTER(spec)
    return _CLUSTER["v"]


REAL_OC_CLUSTER = t_rules.oc_cluster


def r_doc() -> dict:
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L"}}),
                gate1=dict(outcome="PASS", code_key=k, record={"median": 0.0457}, written_at="r-g1"),
                gate2=dict(outcome="PASS", code_key=k, label="LEARNS_CONFIRMATORY",
                           p_judgement={"directions": {"r1": {"ell": 1.087}, "r2": {"ell": 1.792}}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def declared_set(**kw) -> dict:
    js = dict(status="OK", n_b=21, n_a=43, last_turn=103, last_turn_b=17, last_turn_a=103, n_opp=127, n_combos=1986,
              n_odours=55, skipped=dict(SPEC.skipped_declared), e1_clashes=[], cap_fails=[], all_off_pool=True,
              clusters_b=sorted(list(c) for c in SPEC.clusters_b), clusters_a=sorted(list(c) for c in SPEC.clusters_a),
              digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, b=[], a=[])
    return dict(js, **kw)


class ZScripted:
    """t_measure.ZMeasurer's interface; counts[edit] = (MBON13 counts, MBON05 counts) over the 96 presentations."""

    def __init__(self, counts=None, edges=None):
        self.counts = dict(COUNTS, **(counts or {}))
        self.edges = dict({SPEC.no_edit: 0, SPEC.lever_edit: SPEC.lever_edges}, **(edges or {}))
        self.calls = []

    def _sha(self, edit):
        return "sha-C" if edit == SPEC.no_edit else "sha-L"

    def reference(self, edit, odors, strength, settle_ms, read_steps):
        self.calls.append(("reference", edit, len(odors)))
        a, p = self.counts[edit]
        return ref_rows(a, p, self.edges[edit], self._sha(edit))

    def rest(self, edit, seeds, settle_ms, read_steps):
        self.calls.append(("rest", edit, len(seeds)))
        return rest_rows(len(seeds), 0, self.edges[edit], self._sha(edit))


class Scripted:
    """RMeasurer's interface, built per z through at(z). plan[(block, cond)][pair key] = (r_ok, p_ok, bal) (default:
    punish passes only); override[(block, cond)] = dict(edges=, edit=, sha=) changes one block's condition only;
    drop = {edit: P drop} sets ℓ per condition; arm_edges = {edit: edges}; kc = {edit: {odour: frac}} sets KC
    activity; st["fail_once"] makes the next oracle call raise like an interrupted measurement."""

    def __init__(self, plan=None, override=None, drop=None, arm_edges=None, kc=None):
        self.plan, self.override = plan or {}, override or {}
        self.drop = dict({SPEC.lever_edit: 12, SPEC.no_edit: 16}, **(drop or {}))
        self.arm_edges = dict({SPEC.lever_edit: SPEC.lever_edges, SPEC.no_edit: 0}, **(arm_edges or {}))
        self.kc = kc or {}
        self.st, self.calls, self.last_wall_s, self.last_jobs = {"fail_once": False}, [], 2.0, 1
        self.z = dict(Z)
        self._rm = RMeasurer(None, None, SPEC, Params(), READOUT, self.z, TYPES, 100)

    def at(self, z):
        v = copy.copy(self)
        v.z = {k: tuple(x) for k, x in z.items()}
        v._rm = RMeasurer(None, None, SPEC, Params(), READOUT, v.z, TYPES, 100)
        return v

    def inputs(self, row, cond, block, seeds):
        return self._rm.inputs(row, cond, block, seeds)

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self.drop[i["edit"]], "sha-L" if i["edit"] == SPEC.lever_edit else "sha-C",
                        self.arm_edges[i["edit"]]) for i in items]

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, edit, len(odours)))
        lever = edit == SPEC.lever_edit
        fr = self.kc.get(edit, {})
        return {o: dict(frac=[fr.get(o, 0.05)] * len(seeds), max_win=[3] * len(seeds),
                        edit_edges=[self.arm_edges[edit]], csc_sha256=["sha-L" if lever else "sha-C"]) for o in odours}

    def oracle(self, rows, cond, block, seeds):
        self.calls.append(("oracle", block, cond.name, len(rows), tuple(self.z["A"])))
        if self.st["fail_once"]:
            self.st["fail_once"] = False
            raise RuntimeError("worker died")
        o = self.override.get((block, cond.name), {})
        out = []
        for r in rows:
            k = key_of(r)
            flags = self.plan.get((block, cond.name), {}).get(k, (False, True, False))
            res = fake_oracle(*flags, sha=o.get("sha", SHA[cond.name]),
                              edges=o.get("edges", SPEC.lever_edges if cond.name == "L" else 0),
                              edit=o.get("edit", cond.edit), n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ck = hashlib.sha256(f"{block}|{cond.name}|{k}|{self.z}".encode()).hexdigest()
            f = Path(SPEC.cache_dir) / "r_oracle" / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(TR.t_rules, "oc_cluster", cluster_oc)
        self.judge_commits, self.judgement_calls, self.set_calls, self.odour_calls = [], 0, 0, 0
        self.r, self.r_git, self.js = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"]), declared_set()
        self.s = {}
        monkeypatch.setattr(TR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(TR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=104)
        eb = self.keys(self.even, "b")
        self.r_even_plan = {"C": {k: (True, True, False) for k in eb[:7]},
                            "L": {k: (True, True, False) for k in eb[:16]}}
        self.even_plan = {("even", "L"): {k: (True, True, False) for k in eb[:12]}}
        p_ref = {(r["direction"], r["arm"], r["seed"]): {k: v for k, v in r.items()
                                                         if k not in ("direction", "x", "y", "point", "r")}
                 for r in p_rows(P_SPEC, "none", 16)}
        self.archive = tmp_path / "archive"
        odors = [dict(name=f"R{j:02d}", seeds=[1000 + 2 * j, 1001 + 2 * j], strengths={}) for j in range(N_REF // 2)]
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, even_rows=self.even,
                        ref_odors=odors, probe_seeds=[s for o in odors for s in o["seeds"]], ref_strength=0.35,
                        ref_window=(800.0, 600), t_set=self._t_set, set_odours=self._set_odours,
                        judgement_rows=self._judgement_rows, clusters=self._clusters, r_even=self._r_even,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git),
                        s_doc=lambda: json.loads(json.dumps(self.s)))

    def _t_set(self):
        self.set_calls += 1
        return dict(self.js)

    def _set_odours(self):
        self.odour_calls += 1
        return {f"M{i}|T": {"G": 1.0} for i in range(55)}

    def _judgement_rows(self):
        self.judgement_calls += 1
        return self.judge

    def _clusters(self, rows):
        return {key_of(r): ("GROUND* 대 NORMAL" if r["axis"] == "b" else "FIRE*") for r in rows}

    def _r_even(self, rows, name):
        lever = name == "L"
        return got_for(rows, self.r_even_plan[name], sha="sha-L" if lever else "sha-C",
                       edges=SPEC.lever_edges if lever else 0, edit=SPEC.lever_edit if lever else SPEC.no_edit)

    def runner(self, m, zm=None, code=None, tcode=None, pipeline=None):
        return TR.Runner(m.at, zm or ZScripted(), self.ctx, SPEC, code=code or CODE, tcode=tcode or TCODE,
                         pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def scripted(self, **kw):
        m = Scripted(**kw)
        m.plan = {**self.even_plan, **m.plan}
        return m

    def pass_plan(self, n=14, c=8, f_a=3):
        """judgement: L n / C c (b) testable and L f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", "L"): {**{k: (True, True, False) for k in jb[:n]},
                                 **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "C"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through_gate3(w, m, r=None, zm=None):
    r = r or w.runner(m, zm)
    assert r.stage_reuse()["outcome"] == "PASS"
    assert r.stage_set()["outcome"] == "PASS"
    assert r.stage_z()["outcome"] == "PASS"
    assert r.stage_gate1s()["outcome"] == "PASS"
    assert r.stage_smoke()["problems"] == []
    r.stage_oc()
    r.stage_gate2_oc()
    assert r.stage_gate2()["outcome"] == "PASS"
    assert r.stage_gate3()["outcome"] == "PASS"
    return r
```

```python
# tests/brain/test_t_runner.py
"""The T stage chain up to gate ③ (T.1-T.3, T.6, T.9.1-T.9.4, T.9.6, T.9.7; plan Readings 3-12): reuse -> set -> z ->
gate1s -> smoke -> oc -> gate2_oc -> gate2 -> gate3; each stage refuses (exit 2, nothing written) when an earlier block
is missing, a later one exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted
or a hashed file is dirty; STOP_REUSE, STOP_SET_SHORT, STOP_Z_REPRO (the lever then unmeasured), STOP_Z_DEGENERATE,
STOP_STRENGTH_LEVER, a smoke problem, gate ② STOPs / INVALID, STOP_EVEN_REPRO (L then unmeasured) and
STOP_EVEN_LOW_LEVER block every later stage; R's reuse condition broken after `reuse` refuses with exit 7; L's oracle
runs on z_lever and C / E0's on block h4's z; gate ② reads both P conditions on h4 z; the judgement set is touched only
through the set's odours (gate ①'s supplement) before the judgement stages."""
import json
from pathlib import Path

import pytest

from flymon.brain import t_rules
from flymon.brain import t_runner as TR
from tests.brain.t_world import SPEC, ZL, ZScripted, World, doc, through_gate3


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate3(w):
    m = w.scripted()
    through_gate3(w, m)
    d = doc()
    assert set(d) == set(TR.ORDER[:TR.ORDER.index("jm:L")])
    assert all(d[b]["pipeline_key"] == "p" * 64 and d[b]["t_measure_key"] == "t" * 64 for b in d)
    assert d["reuse"]["records"]["r_gate3"] == dict(testable_b=16, c_even=7)
    assert d["set"]["set"]["last_turn"] == 103 and w.set_calls == 1
    z = d["z"]
    assert z["z_lever"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and z["none"]["z"] == {"A": [10.0, 9.0],
                                                                                      "P": [26.0, 19.0]}
    assert z["sd_ratio_lever_over_h4"] == {"A": 3.0 / 9.0, "P": 20.0 / 19.0}
    assert z["lever"]["guard"]["MBON13"]["median_delta"] == 6.0 and Path(SPEC.z_detail).exists()
    assert d["gate1s"]["record"]["n_odours"] == 55 and w.odour_calls == 1
    orc = d["smoke"]["oracle"]
    assert orc["L"]["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert orc["C"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} == orc["E0"]["z"]
    assert set(d["smoke"]["cost"]) == {"gate2_h", "gate3_L_h", "jm_per_condition_h", "note"}
    assert d["oc"]["cluster_fixture_sha256"] and round(d["oc"]["cluster"]["g_fail"][0]["p"], 3) == 0.401
    assert d["gate2_oc"]["rows"][1]["p_stop_weakened"] == pytest.approx(0.75)
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"], g2["seeds"]) == ("LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY",
                                                            list(SPEC.p.seeds))
    assert g2["p_L_on_z_lever"]["ell"]["r1"] == pytest.approx(3 * g2["ratio"]["r1"]["ell_L"])  # σA 9 -> 3
    g3 = d["gate3"]
    assert (g3["testable_b"], g3["c_even"], g3["L_h4_R"]["testable_b"], g3["L_h4_R"]["matches_r_gate3"]) == (
        12, 7, 16, True)
    assert w.judgement_calls == 0
    ev = [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    assert ev == [("oracle", "even", "L", 39, ZL["A"])]                      # L only, on z_lever; C from R's raw
    sm = [c for c in m.calls if c[0] == "oracle" and c[1] == "smoke"]
    assert [(c[2], c[4]) for c in sm] == [("L", ZL["A"]), ("C", (10.0, 9.0)), ("E0", (10.0, 9.0))]
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]


def test_reuse_stop_and_broken_reuse_refuses_with_7(w):
    out = w.runner(w.scripted(), code={"key": "f" * 64}).stage_reuse()
    assert out["outcome"] == t_rules.STOP_REUSE and "공유 측정 키" in out["sentence"]
    with pytest.raises(SystemExit) as e:
        w.runner(w.scripted()).stage_set()
    assert e.value.code == 2 and w.set_calls == 0


def test_reuse_broken_after_reuse(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.r["gate1"]["outcome"] = "STOP_STRENGTH_LEVER"
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == TR.EXIT_KEY and "set" not in doc() and w.set_calls == 0


def test_set_short_and_mismatch(w):
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.js = dict(w.js, digest_keys="0" * 64)
    with pytest.raises(SystemExit) as e:
        r.stage_set()
    assert e.value.code == 2 and "set" not in doc()
    w.js = dict(w.js, status="STOP_SET_SHORT", n_a=40, last_turn=1985)
    out = r.stage_set()
    assert out["outcome"] == t_rules.STOP_SET_SHORT and "(21·40쌍)" in out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_z()


def _to_z(w, m, zm):
    r = w.runner(m, zm)
    r.stage_reuse()
    r.stage_set()
    return r


def test_z_repro_stop_leaves_the_lever_unmeasured(w):
    zm = ZScripted(counts={SPEC.no_edit: ([1, 19] * 47 + [1, 20], [7, 45] * 48)})
    r = _to_z(w, w.scripted(), zm)
    out = r.stage_z()
    assert out["outcome"] == t_rules.STOP_Z_REPRO and out["lever"] is None and "z_lever" not in out
    assert "z 절차 결함" in out["sentence"]
    assert {c[1] for c in zm.calls} == {SPEC.no_edit}
    with pytest.raises(SystemExit):
        r.stage_gate1s()


@pytest.mark.parametrize("counts,failed", [
    (([0, 6] * 48, [60, 100] * 48), ["MBON13"]),                 # MBON13 median Δ 3 < 5
    (([3, 9] * 48, [0] * 96), ["MBON05"]),                       # silent MBON05: guard and zero SD
])
def test_z_degenerate_stops(w, counts, failed):
    zm = ZScripted(counts={SPEC.lever_edit: counts})
    r = _to_z(w, w.scripted(), zm)
    out = r.stage_z()
    assert out["outcome"] == t_rules.STOP_Z_DEGENERATE and out["failed"] == failed and "z_lever" not in out
    with pytest.raises(SystemExit):
        r.stage_gate1s()


def test_z_invalid_when_the_lever_edits_other_than_2_edges(w):
    r = _to_z(w, w.scripted(), ZScripted(edges={SPEC.lever_edit: 1}))
    assert r.stage_z()["outcome"] == t_rules.INVALID


def test_gate1s_stop_names_the_odours(w):
    m = w.scripted(kc={SPEC.lever_edit: {"M3|T": 0.02}})
    r = _to_z(w, m, ZScripted())
    r.stage_z()
    out = r.stage_gate1s()
    assert out["outcome"] == t_rules.STOP_STRENGTH_LEVER and out["failed"] == ["M3|T"]
    assert "T 세트 냄새 M3|T 0.0200" in out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_smoke()


def _to_gate2(w, m):
    r = w.runner(m)
    for s in ("reuse", "set", "z", "gate1s", "smoke", "oc", "gate2_oc"):
        getattr(r, f"stage_{s}")()
    return r


@pytest.mark.parametrize("drop,outcome", [
    ({SPEC.lever_edit: 6}, t_rules.STOP_PUNISH_WEAKENED), ({SPEC.lever_edit: 3}, t_rules.STOP_PUNISH_BROKEN),
    ({SPEC.no_edit: 3}, t_rules.STOP_P_REFERENCE)])
def test_gate2_stops_block_the_judgement(w, drop, outcome):
    r = _to_gate2(w, w.scripted(drop=drop))
    out = r.stage_gate2()
    assert out["outcome"] == outcome and out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_gate3()
    assert w.judgement_calls == 0


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    m = w.scripted()
    r = _to_gate2(w, m)
    m.arm_edges[SPEC.lever_edit] = 1
    assert r.stage_gate2()["outcome"] == t_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)
    m.arm_edges[SPEC.lever_edit] = SPEC.lever_edges
    fixed = w.runner(m, pipeline={"key": "q" * 64})
    assert fixed.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["rerun_of"] == d["gate2_invalid"]["written_at"]
    with pytest.raises(SystemExit):
        fixed.stage_gate2(rerun=True)


def test_gate3_even_repro_stop_leaves_l_unmeasured(w):
    m = w.scripted()
    r = _to_gate2(w, m)
    r.stage_gate2()
    w.r_even_plan["C"] = dict(list(w.r_even_plan["C"].items())[:6])          # R's C raw reads 6, not 7
    out = r.stage_gate3()
    assert out["outcome"] == t_rules.STOP_EVEN_REPRO and out["L"] is None
    assert out["sentence"] == "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L 16, C 6)."
    assert not [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    with pytest.raises(SystemExit):
        r._require("jm:L")


def test_gate3_low_lever_stop(w):
    m = w.scripted()
    m.plan[("even", "L")] = {}
    r = _to_gate2(w, m)
    r.stage_gate2()
    out = r.stage_gate3()
    assert out["outcome"] == t_rules.STOP_EVEN_LOW_LEVER and out["testable_b"] == 0
    with pytest.raises(SystemExit):
        r._require("jm:L")


def test_gate3_refuses_when_rs_even_raw_is_missing(w):
    r = _to_gate2(w, w.scripted())
    r.stage_gate2()

    def missing(rows, name):
        raise ValueError("R's even raw for C lacks 1 pair(s)")
    w.ctx["r_even"] = missing
    with pytest.raises(SystemExit) as e:
        r.stage_gate3()
    assert e.value.code == 2 and "gate3" not in doc()


def test_smoke_problem_blocks_later_stages(w):
    m = w.scripted(override={("smoke", "L"): dict(edges=1)})
    r = _to_z(w, m, ZScripted())
    r.stage_z()
    r.stage_gate1s()
    assert r.stage_smoke()["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_oc_refuses_a_cluster_table_other_than_the_fixture(w, monkeypatch):
    r = _to_z(w, w.scripted(), ZScripted())
    for s in ("z", "gate1s", "smoke"):
        getattr(r, f"stage_{s}")()
    real = TR.t_rules.oc_cluster
    monkeypatch.setattr(TR.t_rules, "oc_cluster", lambda spec: dict(real(spec), sha256="0" * 64))
    with pytest.raises(SystemExit):
        r.stage_oc()
    assert "oc" not in doc()


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    r = w.runner(w.scripted())
    with pytest.raises(SystemExit):
        r.stage_set()
    r.stage_reuse()
    with pytest.raises(SystemExit):
        r.stage_reuse()
    monkeypatch.setattr(TR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_set()
    monkeypatch.setattr(TR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(TR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/t_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_set()
    assert set(doc()) == {"reuse"}
    assert json.loads(Path(SPEC.summary).read_text())["reuse"]["outcome"] == "PASS"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_t_runner.py -q`
Expected: collection error `ImportError: cannot import name 't_runner'`.

- [ ] **Step 3: Write the first part of `t_runner`**

```python
"""Spec T's stage chain (T.1-T.6, T.9; plan Readings 3-15):
reuse -> set -> z -> gate1s -> smoke -> oc -> gate2_oc -> gate2 -> gate3 -> jm:L -> jm:C -> jm:E0 -> seal -> judge
(and after judge only: recompute / invalid_run / rs_reread), one block each in results/summary/t_lever.json, written
only through t_store. Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later
block exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary has uncommitted changes or a
hashed T file is dirty. Every stage after `reuse` re-checks R's reuse condition (the shared measurement key equals
R's and R's repro / gate ① blocks are intact) and refuses with SystemExit 7 when it broke; jm, seal and judge also
re-check the T measurement key block z was measured under (T.9.6, exit 7). A smoke with problems or a gate whose
outcome is not PASS blocks every later stage, so the judgement set stays unused.
z (T.1, T.9.2-T.9.4): block z holds z_lever; L's oracle (gate ③'s even pairs and jm:L) runs on a measurer built with
z_lever, C's and E0's (and every P arm and KC activity, where z plays no part) on one built with block h4's z
(`measure(z)`, plan Reading 4). Gate ② reads both P conditions on h4 z (T.9.3).
The judgement set is reached only through ctx["judgement_rows"] / ctx["set_odours"] (t_pairs: generated afresh and
checked against T.9.1's declared values on every call) from the gate-① supplement and the judgement stages; `set`
reads the list once (no measurement); jm blocks carry no pair statistic. Blocks carry code_key (the shared measurement
key, which also keys the cache), t_measure_key (shared files + t_measure.py + h3_spec.py) and pipeline_key (T's own
files)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent.e_runner import summary_git
from . import r_records, t_records, t_rules, t_store
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import p_items
from .s_runner import S_HASHED_FILES
from .t_measure import T_MEASURE_FILES
from .t_pairs import check_t_set
from .t_pairs import summary as set_summary
from .t_spec import smoke

ORDER = ("reuse", "set", "z", "gate1s", "smoke", "oc", "gate2_oc", "gate2", "gate3", "jm:L", "jm:C", "jm:E0", "seal",
         "judge")
GATES = ("reuse", "set", "z", "gate1s", "gate2", "gate3")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_KEY = 2, 7
T_PIPELINE_FILES = ("flymon/brain/t_spec.py", "flymon/brain/t_pairs.py", "flymon/brain/t_store.py",
                    "flymon/brain/t_records.py", "flymon/brain/t_rules.py", "flymon/brain/t_runner.py",
                    "scripts/run_t.py")
T_HASHED_FILES = tuple(dict.fromkeys(S_HASHED_FILES + T_MEASURE_FILES + T_PIPELINE_FILES + (
    "flymon/battle/pool.py", "results/summary/s_lever.json", "tests/brain/fixtures/t_oc_cluster.json")))
# The decision code: every hashed T file outside the T measurement key. The seal pins its hash; the judge reads only
# under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in T_HASHED_FILES if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES)
# Written before the band is computed: the set was read once. REREAD: S.9.2's one re-generation after the mark (T.5).
# DONE: written right after the judge block, so a judge block written and then discarded never reopens the set.
JUDGE_MARKER = "results/t/judge_read.json"
REREAD_MARKER = "results/t/judge_reread.json"
DONE_MARKER = "results/t/judge_done.json"


def git_state() -> dict:
    return _h3_git_state(files=T_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(T_PIPELINE_FILES)


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


def _ztuple(z: dict) -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in z.items()}


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, block h4's z), the H.4 pools as types, the encoder summary, the even rows, the H.3
    reference set and its probe seeds, lazy callables for T's set (list only), its odours (gate ①'s supplement), its
    judgement rows (checked) and cluster labels, R's even raw read only by content key (gate ③), P's committed entries
    (gate ②'s OC) and stimuli (c1 from block n1), R's summary with its git state (the reuse condition) and S's."""
    from types import SimpleNamespace

    from ..agent.config import load_c3_config
    from . import n_cli, p_cli, q_pairs, r_pairs, t_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_spec import SPEC as H3
    from .h3_spec import make_odors
    from .h3_store import code_key
    from .r_measure import RMeasurer
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
    odors = make_odors(pops, H3.reference)
    r_cache = t_store.RReadCache(spec.r_cache_dir, code_key(npz, files=R_MEASURE_FILES))
    r_m = RMeasurer(None, r_cache, spec, cfg.params, cfg.readout, cfg.z, types, len(pops.kc))

    def p_inputs(pspec, smoke_: bool):
        c1, why = p_cli.c1_source(pspec, smoke_)
        if why:
            refuse(why)
        g, c = pspec.o.o2_point
        names = list(dict.fromkeys(s for x, y in pspec.pairs().values() for s in (x, y)))
        nv = p_cli.nview(SimpleNamespace(spec=pspec, c3=cfg.params, types=model_types))
        return c1, n_cli.stimuli_at(nv, names, g, c)

    def r_even(rows, name: str) -> list:
        """R's even raw of one condition (block h4's z, H.4 seeds), read only; ValueError when an entry is missing."""
        cond, seeds = spec.cond(name), spec.h4_seeds()
        miss = [row_key(r) for r in rows if r_cache.get("r_oracle", r_m.inputs(r, cond, "even", seeds)) is None]
        if miss:
            raise ValueError(f"R's even raw for {name} lacks {len(miss)} pair(s) under the shared key: {miss[:3]}")
        return r_m.oracle(rows, cond, "even", seeds)

    return dict(params=cfg.params, readout=dict(cfg.readout), z=dict(cfg.z), types=types, n_kc=len(pops.kc), enc=enc,
                even_rows=even, ref_odors=odors, probe_seeds=[int(s) for o in odors for s in o["seeds"]],
                ref_strength=float(H3.strength), ref_window=(float(H3.reference_window.settle_ms),
                                                            int(H3.reference_window.read_ms)),
                t_set=lambda: t_pairs.t_set(pops, enc, spec, cfg.params),
                set_odours=lambda: t_pairs.set_odours(pops, rc, enc, spec, cfg.params),
                judgement_rows=lambda: t_pairs.judgement_rows(pops, rc, enc, spec, cfg.params),
                clusters=t_pairs.clusters, r_even=r_even,
                p_ref=lambda wanted: r_pairs.p_reference(spec.p_cache_dir, p_doc["measure_key"], wanted),
                p_inputs=p_inputs, r_doc=lambda: json.loads(Path(spec.r_summary).read_text()),
                r_git=lambda: summary_git(spec.r_summary), s_doc=lambda: json.loads(Path(spec.s_summary).read_text()))


class Runner:
    def __init__(self, measure, zm, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, pipeline: dict | None = None, archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) on that z, one per z over one pool and cache; zm: a
        t_measure.ZMeasurer (or its interface)."""
        self.measure, self.zm, self.ctx, self.spec = measure, zm, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]

    # ---- measurers per z (plan Reading 4) ----------------------------------------------------------------------------
    def _z_lever(self, doc=None) -> dict:
        doc = doc if doc is not None else self._doc()
        return _ztuple(doc["z"]["z_lever"])

    def _z_for(self, name: str, doc=None) -> dict:
        return self._z_lever(doc) if name == self.spec.cond_names[0] else _ztuple(self.ctx["z"])

    def _m_for(self, name: str, doc=None):
        return self.measure(self._z_for(name, doc))

    @property
    def m_h4(self):
        return self.measure(_ztuple(self.ctx["z"]))

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return t_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed T files are dirty: {gs['dirty_hashed']}")

    def _reuse_now(self, stage: str) -> dict:
        """T.3 1: the reuse condition, re-checked at every stage after `reuse`."""
        r = t_rules.reuse(self.ctx["r_doc"](), self.ctx["r_git"](), self.code_key, self.spec)
        if r["outcome"] != t_rules.PASS:
            refuse(f"stage {stage}: R's reuse condition broke ({'; '.join(r['reasons'])}) — T has no re-measurement "
                   f"path; T stops here and the judgement set stays unused (plan Reading 8)", EXIT_KEY)
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
            refuse(f"stage {stage}: later block(s) {later} exist; T never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; T never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != t_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — T stops there and the "
                   f"judgement set stays unused (T.3)")
        if i > 0:
            self._reuse_now(stage)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, t_measure_key=self.t_measure_key,
                    pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        t_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    # ---- reuse (T.3 1) and the set (T.2, T.9.1) — no pool ------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        rd = self.ctx["r_doc"]()
        dec = t_rules.reuse(rd, self.ctx["r_git"](), self.code_key, sp)

        def g(*ks):
            return _dig(rd, ks)
        rec = dict(gate1_median=g("gate1", "record", "median"), repro_csc_sha256_none=g("repro", "csc_sha256_none"),
                   r_smoke_L_csc_sha256=g("smoke", "oracle", "L", "csc_sha256"),
                   r_gate3=dict(testable_b=g("gate3", "testable_b"), c_even=g("gate3", "c_even")),
                   r_gate2_ell={d: g("gate2", "p_judgement", "directions", d, "ell") for d in sp.p.directions})
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, r_summary=sp.r_summary,
                    r_commits=dict(sp.r_commits), r_blocks={b: g(b, "written_at") for b in sp.r_reused}, records=rec)
        self._write("reuse", body)
        return body

    def stage_set(self) -> dict:
        """T.9.1, list only: STOP_SET_SHORT is recorded; any other mismatch with the declared values refuses."""
        self._require("set")
        try:
            js = self.ctx["t_set"]()
        except ValueError as e:
            refuse(f"the T set cannot be generated: {e}")
        dec = t_rules.set_outcome(js)
        if dec["outcome"] == t_rules.PASS:
            bad = check_t_set(js, self.spec)
            if bad:
                refuse(f"the T set does not reproduce its declaration (T.9.1): {bad}")
        body = dict(dec, set=set_summary(js))
        self._write("set", body)
        return body

    # ---- z (T.1, T.9.4, T.9.6, T.9.7) --------------------------------------------------------------------------------
    def stage_z(self) -> dict:
        """The unedited engine first, measured afresh (no cache): its z must equal block h4's bit for bit
        (STOP_Z_REPRO, the lever is then not measured); then the lever: H.4's readout guard per readout type and a
        nonzero SD (STOP_Z_DEGENERATE), else z_lever. Raw rows go to results/t/z.json."""
        doc = self._require("z")
        sp, ctx = self.spec, self.ctx
        ro, odors, seeds = ctx["readout"], ctx["ref_odors"], ctx["probe_seeds"]
        settle, steps = ctx["ref_window"]
        n = len(seeds)
        raw = {}

        def side(edit):
            ref = self.zm.reference(edit, odors, ctx["ref_strength"], settle, steps)
            rest = self.zm.rest(edit, seeds, settle, steps)
            raw[edit] = dict(ref=ref, rest=rest)
            return t_records.z_side(ref, rest, ro, sp)

        none = side(sp.no_edit)
        dec = t_rules.z_repro(none, n, doc["reuse"]["records"]["repro_csc_sha256_none"], sp)
        lever = None
        if dec["outcome"] == t_rules.PASS:
            lever = side(sp.lever_edit)
            dec = t_rules.z_lever(lever, n, ro, sp)
        p = t_store.write_json(sp.z_detail, dict(t_measure_key=self.t_measure_key, readout=ro, rows=raw), self.plist)
        body = dict(dec, none=none, lever=lever, z_h4={k: list(v) for k, v in sp.z_h4_dict().items()},
                    n_presentations=n, detail_path=sp.z_detail, detail_sha256=sha256_file(p))
        if dec["outcome"] == t_rules.PASS:
            body["z_lever"] = lever["z"]
            body["sd_ratio_lever_over_h4"] = t_records.z_ratios(lever["z"], sp.z_h4_dict())
        self._write("z", body)
        return body

    # ---- gate ① supplement (T.9.1) -----------------------------------------------------------------------------------
    def stage_gate1s(self) -> dict:
        self._require("gate1s")
        sp = self.spec
        try:
            odours = self.ctx["set_odours"]()
        except ValueError as e:
            refuse(f"the T set does not reproduce its declaration: {e}")
        m = self.m_h4
        act_l = m.activity(odours, sp.lever_edit, sp.strength, sp.kc_seeds(), "gate1s")
        wall = m.last_wall_s
        act_n = m.activity(odours, sp.no_edit, sp.strength, sp.kc_seeds(), "gate1s")
        rec = t_records.gate1s_record(act_l, act_n, sp)
        body = dict(t_rules.gate1s(rec, sp), record=rec, seeds=list(sp.kc_seeds()), wall_s=wall + m.last_wall_s)
        self._write("gate1s", body)
        return body

    # ---- smoke (T.3 4) ---------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sm.p, True)
        items = p_items(sm.p, st, sp.lever_edit) + p_items(sm.p_c, st, sp.no_edit)
        rows = m.arms(items, self.ctx["readout"], sm.p.o.n.h3.punish_type, "smoke", sm.p.o.n)
        p_wall = m.last_wall_s
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
            mc = self._m_for(cond.name, doc)
            got = mc.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self._z_for(cond.name, doc), [row_key(r) for r in sel],
                                       sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  kc_median=s["kc_median"], saturation=s["saturation"], wall_s=mc.last_wall_s,
                                  jobs=mc.last_jobs, z={k: list(v) for k, v in mc.z.items()})
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        t_store.write_json(sp.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(p, orc, doc), cost=self._cost(p_wall, orc, sm),
                    detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, doc) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
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
        if _ztuple(orc[nl]["z"]) != self._z_lever(doc):
            bad.append(f"L's oracle ran on z {orc[nl]['z']}, not block z's z_lever")
        for n in (nc, ne):
            if _ztuple(orc[n]["z"]) != _ztuple(self.ctx["z"]):
                bad.append(f"{n}'s oracle ran on z {orc[n]['z']}, not block h4's")
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
        n_sm = sum(len(v) for v in sm.even_seeds().values())
        jm = sum(len(v) for v in sp.judge_seeds().values()) / n_sm
        ev = sum(len(v) for v in sp.h4_seeds().values()) / n_sm
        return dict(gate2_h=p_h, gate3_L_h=o_round * ev * _rounds(len(self.ctx["even_rows"]), sp.workers) / 3600,
                    jm_per_condition_h=o_round * jm * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (T.6, T.9.6, T.9.7) — before gate ② -------------------------------------------
    def stage_oc(self) -> dict:
        """The independent model (S.6's, n_a 43) and the cluster-correlated model, the latter equal to its committed
        fixture (T.9.7: computed once, committed, re-checked here; the block records both sha256)."""
        self._require("oc")
        sp = self.spec
        cl = t_rules.oc_cluster(sp)
        fx = Path(_ROOT / sp.oc_cluster_fixture)
        if not fx.exists() or canonical(json.loads(fx.read_text())) != canonical(cl):
            refuse(f"the cluster OC differs from its committed fixture {sp.oc_cluster_fixture} (T.9.7)")
        body = dict(independent=t_rules.oc(sp), cluster={k: v for k, v in cl.items() if k != "rows"},
                    cluster_fixture=sp.oc_cluster_fixture, cluster_fixture_sha256=sha256_file(fx))
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7's gate ② OC on block h4's z (T.6, T.9.3), from P's committed block, before gate ②."""
        self._require("gate2_oc")
        pairs, point = P_SPEC.pairs(), [float(v) for v in P_SPEC.o.o2_point]
        wanted = {(d, a, int(s)) for d in P_SPEC.directions for a in P_SPEC.arms for s in P_SPEC.seeds}
        ref = self.ctx["p_ref"](wanted)
        if set(ref) != wanted:
            refuse(f"P's committed cache {self.spec.p_cache_dir} lacks {sorted(wanted - set(ref))[:3]}")
        rows = [dict(ref[k], direction=k[0], x=pairs[k[0]][0], y=pairs[k[0]][1], point=point) for k in sorted(wanted)]
        body = t_records.gate2_oc(rows, self.ctx["z"], P_SPEC, self.spec)
        self._write("gate2_oc", body)
        return body

    # ---- gate ② (T.3 6, T.9.3) ---------------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """S's gate ② on 25_200_000+i, both P conditions read on block h4's z (T.9.3); P_L read on z_lever is a record.
        rerun: only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID run's."""
        prior = None
        if rerun:
            doc = self._require("gate2", allow_own=True)
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (T.3 6, R.5)")
            if not blk or blk.get("outcome") != t_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (T.3 6)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (T.3 6)")
            prior = blk
        else:
            doc = self._require("gate2")
        sp, z = self.spec, self.ctx["z"]
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sp.p, False)
        items = p_items(sp.p, st, sp.lever_edit) + p_items(sp.p_c, st, sp.no_edit)
        rows = m.arms(items, self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2", sp.p.o.n)
        rows_l = [r for r in rows if r["edit"] == sp.lever_edit]
        rows_c = [r for r in rows if r["edit"] == sp.no_edit]
        res_l = p_judge(rows_l, z, c1["c1"], sp.p)
        res_c = p_judge(rows_c, z, c1["c1"], sp.p_c)
        ratio, why = None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = t_records.ratio(rows_l, rows_c, z, sp)
            except ValueError as e:
                why.append(str(e))
        dec = t_rules.gate2(res_l, res_c, ratio, sp) if not why else dict(outcome=t_rules.INVALID, reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [sp.lever_edges] or edges_c != [0]) and dec["outcome"] != t_rules.INVALID:
            dec = dict(outcome=t_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{sp.lever_edges} / 0"])
        zl = (None if dec["outcome"] == t_rules.INVALID
              else t_records.p_zlever(rows_l, self._z_lever(doc), c1["c1"], res_c, sp))
        body = dict(dec, ratio=ratio, p_judgement_L=res_l, p_judgement_C=res_c, p_L_on_z_lever=zl,
                    c1_source=c1, edit_edges_L=edges_l, edit_edges_C=edges_c, seeds=list(sp.p.seeds),
                    rerun_of=(prior or {}).get("written_at"), wall_s=m.last_wall_s, jobs=m.last_jobs)
        if prior is None:
            return self._write("gate2", body)
        return self._write_rerun(prior, body)

    def _write_rerun(self, prior: dict, body: dict) -> dict:
        doc = self._doc()
        if GATE2_INVALID in doc or doc.get("gate2") != prior:
            refuse("the summary changed during gate ②'s rerun; nothing written")
        block = self._stamp("gate2", body)
        doc[GATE2_INVALID], doc["gate2"] = prior, block
        t_store.write_json(self.summary_path, doc, self.plist)
        return block

    # ---- gate ③ (T.9.2) ----------------------------------------------------------------------------------------------
    def stage_gate3(self) -> dict:
        """R's even raw for C (and L, a record) read by content key on block h4's z: C must reproduce c_even 7 first
        (STOP_EVEN_REPRO, L is then not measured); then L's 39 even pairs re-measured on z_lever (α chosen on z_lever)
        and R's gate ③ rule (L testable_b ≥ 11, c_even 7)."""
        doc = self._require("gate3")
        sp = self.spec
        rows, seeds = self.ctx["even_rows"], sp.h4_seeds()
        keys = [row_key(r) for r in rows]
        nl, nc, _ = sp.cond_names
        try:
            r_c, r_l = self.ctx["r_even"](rows, nc), self.ctx["r_even"](rows, nl)
        except ValueError as e:
            refuse(f"gate ③ reads R's even raw by content key and it is not complete: {e}")
        z_h4 = _ztuple(self.ctx["z"])
        C = r_records.cond_summary(r_c, sp.cond(nc), sp, z_h4, keys, seeds)
        L_h4 = r_records.cond_summary(r_l, sp.cond(nl), sp, z_h4, keys, seeds)
        l_h4 = (L_h4["aggregate"] or {}).get("testable_b")
        L, m = None, None
        if not C["reasons"] and (C["aggregate"] or {}).get("testable_b") == sp.c_even_expected:
            m = self._m_for(nl, doc)
            got = m.oracle(rows, sp.cond(nl), "even", seeds)
            L = r_records.cond_summary(got, sp.cond(nl), sp, self._z_lever(doc), keys, seeds)
        dec = t_rules.gate3(L, C, sp, doc["reuse"]["records"]["repro_csc_sha256_none"], l_h4)
        keep = r_records.KEEP
        body = dict(dec, L_h4_R=dict(testable_b=l_h4, matches_r_gate3=bool(l_h4 == sp.r_even_L_h4),
                                     **{k: L_h4.get(k) for k in ("aggregate", "counts")}),
                    C={k: C.get(k) for k in keep}, L=None if L is None else {k: L.get(k) for k in keep},
                    records=None if L is None else r_records.compare(L, C, None, sp), seeds=seeds,
                    n_pairs=len(rows), wall_s=None if m is None else m.last_wall_s,
                    jobs=None if m is None else m.last_jobs)
        self._write("gate3", body)
        return body
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_t_runner.py -q`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/t_runner.py tests/brain/t_world.py tests/brain/test_t_runner.py
git commit -m "feat(t): t_runner (1) — chain with reuse re-check (exit 7), set, z (repro then guard), gate-1 supplement, smoke, OCs, gate 2 on h4 z, gate 3 with R's C raw"
```

---

### Task 6: `t_runner` (2) and `scripts/run_t.py` — judgement measurement per z, seal, judge read once with one re-generation, recovery, R · S re-read; the CLI; mutation, fault-injection and import tests

**Files:**
- Modify: `flymon/brain/t_runner.py` (append to class `Runner`, which ends the file)
- Create: `scripts/run_t.py`
- Test: `tests/brain/test_t_judge.py`

**Interfaces:**
- Consumes: Task 5's `Runner` and module names; `r_records.preread_validity` / `judge_inputs` / `compare` / `cond_summary` / `raw_check` / `stats_ok` / `pair_stats` / `_short` / `JUDGE_BLOCK`; `t_store.load_manifest` / `archive_copy`; `t_rules.g_fail_s` / `read_band` / `sentence`; `t_records.clusters` / `alpha_fixed`; `t_measure.ZMeasurer` / `t_measure_key` (CLI).
- Produces: `Runner._judge_chain`, `_judgement_rows`, `stage_jm(name)`, `_raws`, `stage_seal()`, `_read(doc, mark=None)`, `stage_judge()`, `_require_after_judge`, `stage_recompute(note)`, `stage_invalid_run(note)`, `stage_rs_reread()`; `scripts/run_t.py` with `STAGES`, `POOL_STAGES`, `GATE_STAGES`, `check_args`, `exit_code(stage, out)`, `main(argv=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/brain/test_t_judge.py
"""The judgement end of the T chain (T.3 8-10, T.4, T.5, T.6, T.9.2, T.9.6; Readings 4, 8, 10-15) and its mutation
tests: jm:L runs on z_lever and jm:C / jm:E0 on block h4's z, and every block carries no pair statistic and is written
only when all 64 pairs are back; R's reuse condition and the T measurement key are re-checked before the judgement
measurement, the seal and the judge (exit 7); the seal re-checks every raw file and its stored inputs (L's carry
z_lever), archives 192 raw files and pins the decision code; the judge reads L on z_lever and C / E0 on h4 z, once —
with the one re-generation after an interruption between the mark and the block; the records hold the clusters, the
α-fixed sensitivity and z; rs_reread runs only after judge; the CLI's refusals and exit codes; T never defines a copy
of a shared measurement job, measurer or cache; the judgement set is touched only from gate1s (its odours) and jm on."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_measure, r_store, t_store
from flymon.brain import t_runner as TR
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.t_measure import T_MEASURE_FILES
from flymon.brain.t_store import RReadCache, TCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.t_world import CODE, SPEC, TCODE, ZL, World, doc, through_gate3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through_gate3(w, m)
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def _sealed(w, plan=None):
    m = w.scripted(plan=plan or w.pass_plan(n=14, c=8, f_a=3))
    r = _to_seal(w, m)
    assert r.stage_seal()["status"] == "SEALED"
    return r


def test_full_chain_reads_selected(w):
    r = _to_seal(w, w.scripted(plan=w.pass_plan(n=14, c=8, f_a=3)))
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 64 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 64
    assert jm["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and doc()["jm:C"]["z"] == {"A": [10.0, 9.0],
                                                                                     "P": [26.0, 19.0]}
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm
    stored = json.loads(Path(jm["manifest"][0]["cache_file"]).read_text())["inputs"]
    assert stored["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and stored["act_seeds"] == SPEC.judge_seeds()["act"]
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 192 and seal["set"]["n_a"] == 43
    assert seal["z"]["z_lever"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive)) and seal["decision"]["key"] == TR.decision_key()["key"]
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    rho = doc()["gate2"]["ratio"]
    assert (f"(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 "
            f"{rho['r1']['ratio']:.3f}·ℓ_r2 {rho['r2']['ratio']:.3f} ≥ 0.5(약화 정도 기록), 짝수 재판독 12/21") in out["sentence"]
    assert "생성원 턴 0–103" in out["sentence"] and "세 번째 판정" in out["sentence"]
    rec = out["records"]
    assert rec["clusters"]["GROUND* 대 NORMAL"]["L"] == dict(n=21, testable=14, reward_pass=14, punish_pass=21)
    assert rec["alpha_fixed"]["n"] == 14 and "판정 아님" in rec["alpha_fixed"]["note"]
    assert rec["z"]["z_lever"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert out["resumed_after_mark"] is False and out["k_even"] == 12
    assert out["oc_sha256"] == doc()["oc"]["independent"]["sha256"] and set(doc()) == set(TR.ORDER)
    assert Path(TR.JUDGE_MARKER).exists() and Path(TR.DONE_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_net_drop_3_on_a_reads_the_punish_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", "L")].update({k: (False, False, False) for k in ja[30:33]})
    out = _sealed(w, plan).stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["a"]["net_drop"] == 3


def test_b_tb_closes_the_t_claim_only(w):
    out = _sealed(w, w.pass_plan(n=8, c=8, f_a=3)).stage_judge()
    assert out["band"] == "B_Tb" and out["sentence"].endswith(
        "→ 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 결과는 R·S 그대로).")


def test_judgement_set_is_touched_only_from_gate1s_and_jm_on(w):
    r = through_gate3(w, w.scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0 and w.odour_calls == 1
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    m.st["fail_once"] = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 64


@pytest.mark.parametrize("where", ["jm", "seal", "judge"])
@pytest.mark.parametrize("what", ["reuse", "tkey"])
def test_keys_are_rechecked_before_measurement_seal_and_judge(w, where, what):
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    if what == "reuse":
        w.r["gate1"]["code_key"] = "x" * 64
    else:
        r = w.runner(m, tcode={"key": "u" * 64})
    with pytest.raises(SystemExit) as e:
        {"jm": lambda: r.stage_jm("L"), "seal": r.stage_seal, "judge": r.stage_judge}[where]()
    assert e.value.code == TR.EXIT_KEY
    assert not Path(TR.JUDGE_MARKER).exists()


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    r = _to_seal(w, w.scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)}))
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_l_raw_on_h4_z_fails_the_preread_validity(w):
    """A jm:L measured with the wrong z (block h4's) is caught by the stored inputs at the seal."""
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    real_m_for = r._m_for
    r._m_for = lambda name, doc=None: m.at(Z) if name == "L" else real_m_for(name, doc)
    r.stage_jm("L")
    r._m_for = real_m_for
    r.stage_jm("C")
    r.stage_jm("E0")
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any(x.startswith("L: ") and "stored inputs" in x for x in seal["reasons"])


def test_mutation_a_tampered_set_refuses(w):
    r = through_gate3(w, w.scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("T set digest_keys: generated '00', declared '37dd'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through_gate3(w, w.scripted(plan=w.pass_plan()))
    d = doc()
    del d["gate3"]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and w.judgement_calls == 0


def test_judge_refuses_a_raw_file_changed_after_the_seal(w):
    r = _sealed(w)
    f = Path(doc()["jm:C"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"sha-C"', '"sha-X"'))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in doc() and not Path(TR.JUDGE_MARKER).exists()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(TR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(TR.JUDGE_MARKER).exists()


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(TR.t_rules, "read_band", lambda *a, **k: dict(band=TR.t_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == TR.t_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(TR.JUDGE_MARKER).exists()


class _Kill(Exception):
    pass


def _kill_judge_write(monkeypatch, seen):
    orig = t_store.write_summary_block

    def boom(path, block, obj, plist):
        if block == "judge":
            seen.append(obj)
            raise _Kill()
        return orig(path, block, obj, plist)
    monkeypatch.setattr(TR.t_store, "write_summary_block", boom)
    return orig


def test_resume_after_mark_once(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    mark = json.loads(Path(TR.JUDGE_MARKER).read_text())
    assert "judge" not in doc() and not Path(TR.DONE_MARKER).exists() and mark["t_measure_key"] == "t" * 64
    monkeypatch.setattr(TR.t_store, "write_summary_block", orig)
    out = r.stage_judge()
    assert out["status"] == "READ" and out["resumed_after_mark"] is True and out["mark_read_at"] == mark["read_at"]
    assert (out["band"], out["sentence"]) == (seen[0]["band"], seen[0]["sentence"])
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_a_second_regeneration_refuses(w, monkeypatch):
    r = _sealed(w)
    seen = []
    _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    with pytest.raises(_Kill):
        r.stage_judge()
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and len(seen) == 2 and "judge" not in doc()


def test_recovery_after_reading(w):
    r = _sealed(w)
    with pytest.raises(SystemExit):
        r.stage_rs_reread()                                 # only after judge
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


def _other_track(rows, plan, root: Path) -> dict:
    """An R- or S-shaped summary: jm:L / jm:C blocks with manifests over written raw files, and a judge block."""
    out = {}
    for n in ("L", "C"):
        man = []
        for r in rows:
            k = key_of(r)
            res = fake_oracle(*plan.get((n, k), (False, True, False)), sha="sha-L" if n == "L" else "sha-C",
                              edges=2 if n == "L" else 0, edit=SPEC.lever_edit if n == "L" else SPEC.no_edit)
            ck = hashlib.sha256(f"{root}|{n}|{k}".encode()).hexdigest()
            f = root / n / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "result": res}))
            man.append(dict(key=k, cache_key=ck, cache_file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        out[f"jm:{n}"] = dict(manifest=man, seeds=dict(act=list(range(8)), select=list(range(8)),
                                                        report=list(range(8))))
    out["judge"] = dict(band="B_Fa")
    return out


def test_rs_reread_records_rs_and_s_raw_under_ts_z(w, tmp_path):
    from tests.brain.r_fixtures import fake_rows
    rows = fake_rows(21, 32, turn0=64)
    plan = {("L", key_of(r)): (True, True, False) for r in rows[:12]}
    plan.update({("C", key_of(r)): (True, True, False) for r in rows[:6]})
    w.r.update(_other_track(rows, plan, tmp_path / "rraw"))
    w.s = _other_track(fake_rows(21, 43, turn0=104), {}, tmp_path / "sraw")
    r = _sealed(w)
    r.stage_judge()
    out = r.stage_rs_reread()
    assert (out["rows"]["R"]["n"], out["rows"]["R"]["c"], out["rows"]["R"]["n_a"]) == (12, 6, 32)
    assert out["rows"]["S"]["n_a"] == 43 and out["rows"]["R"]["band_judged"] == "B_Fa" and "기록 전용" in out["note"]
    with pytest.raises(SystemExit):
        r.stage_rs_reread()


class OraclePool:
    """r_oracle_job-shaped outputs for real RMeasurers + TCache; the pair key travels in the odour name ("G|<key>"
    E-grid, "E|<key>" E0); the z each job received is recorded."""

    def __init__(self, plan):
        self.n_workers, self.plan, self.jobs, self.z = 4, plan, 0, {}

    def run_jobs(self, fn, kws):
        assert fn is r_jobs.r_oracle_job
        out = []
        for kw in kws:
            self.jobs += 1
            lever = kw["edit"] == SPEC.lever_edit
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            self.z[cond] = kw["z"]
            flags = self.plan.get(("judge", cond), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha="sha-L" if lever else "sha-C", edges=SPEC.lever_edges if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_real_cache_round_trip_seals_and_reads(w):
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through_gate3(w, w.scripted(plan=plan))
    pool, cache, ms = OraclePool(plan), TCache(SPEC.cache_dir, CODE), {}

    def measure(z):
        k = json.dumps({a: list(b) for a, b in sorted(z.items())})
        return ms.setdefault(k, RMeasurer(pool, cache, SPEC, w.ctx["params"], READOUT, z, TYPES, 100))
    real = TR.Runner(measure, None, w.ctx, SPEC, code=CODE, tcode=TCODE, pipeline={"key": "p" * 64},
                     archive_root=w.archive)
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 64
    assert pool.jobs == 192
    assert pool.z["L"] == {k: list(v) for k, v in ZL.items()} and pool.z["C"] == pool.z["E0"] == {
        k: list(v) for k, v in Z.items()}
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds", "t_set", "set_odours", "_checked"}
ALLOWED = {"t_spec.py": {None, "judge_seeds"}, "t_pairs.py": {"_checked", "judgement_rows", "set_odours"},
           "t_runner.py": {"build_ctx", "_judgement_rows", "stage_set", "stage_gate1s", "stage_jm", "stage_seal",
                           "_read", "_cost"}}


def _owners(tree) -> set:
    out = set()

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            if ((isinstance(ch, ast.Name) and ch.id in WATCH) or (isinstance(ch, ast.Attribute) and ch.attr in WATCH)
                    or (isinstance(ch, ast.Constant) and ch.value in WATCH)):
                out.add(fn)
            visit(ch, ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn)
    visit(tree, None)
    return out


def test_only_the_judgement_stages_name_the_judgement_set():
    files = sorted((ROOT / "flymon/brain").glob("t_*.py")) + [ROOT / "scripts/run_t.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def test_t_never_defines_a_copy_of_shared_measurement_code():
    """T.3 1 / T.8: T runs R's jobs, measurer and cache themselves; its caches only override put (TCache: T's
    writer; RReadCache: refuse); its own measurement file defines only the reference-set jobs; no T file loads code by
    path."""
    assert t_store.RCache is r_store.RCache and issubclass(TCache, r_store.RCache) and issubclass(RReadCache,
                                                                                                r_store.RCache)
    for cls in (TCache, RReadCache):
        assert {k for k in vars(cls) if not k.startswith("__")} == {"put"}
    for obj in (r_jobs.kc_activity_job, r_jobs.r_arm_job, r_jobs.r_oracle_job, RMeasurer, r_store.RCache):
        src = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
        assert src in R_MEASURE_FILES, src
    shared = {"kc_activity_job", "r_arm_job", "r_oracle_job", "RMeasurer", "RCache", "ECache", "q_oracle_job",
              "arm_job", "q_rig", "reference_job", "rest_job", "engine_for", "apply_q_edit", "apply_csc_edit"}
    for p in sorted((ROOT / "flymon/brain").glob("t_*.py")) + [ROOT / "scripts/run_t.py"]:
        tree = ast.parse(p.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not defs & shared, (p.name, defs & shared)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not names & {"importlib", "exec", "spec_from_file_location", "runpy"}, p.name
    assert not set(T_MEASURE_FILES) & set(R_MEASURE_FILES) and r_measure.R_MEASURE_FILES == R_MEASURE_FILES


def _cli():
    spec = importlib.util.spec_from_file_location("run_t_for_test", ROOT / "scripts/run_t.py")
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
    for st in ("reuse", "set", "z", "gate1s", "gate2", "gate3"):
        assert cli.exit_code(st, {"outcome": "PASS"}) == 0 and cli.exit_code(st, {"outcome": "INVALID"}) == 5
    for st, o in (("reuse", "STOP_REUSE"), ("set", "STOP_SET_SHORT"), ("z", "STOP_Z_REPRO"),
                  ("z", "STOP_Z_DEGENERATE"), ("gate1s", "STOP_STRENGTH_LEVER"), ("gate2", "STOP_PUNISH_WEAKENED"),
                  ("gate3", "STOP_EVEN_REPRO"), ("gate3", "STOP_EVEN_LOW_LEVER")):
        assert cli.exit_code(st, {"outcome": o}) == 3, (st, o)
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    assert cli.exit_code("oc", {}) == 0 and cli.exit_code("rs_reread", {}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "INVALID"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert set(cli.POOL_STAGES) == {"z", "gate1s", "smoke", "gate2", "gate3", "jm"}
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_t.py").read_text()


def test_hashed_pipeline_and_decision_files():
    from flymon.brain.s_runner import S_HASHED_FILES
    assert set(TR.T_PIPELINE_FILES) == {f"flymon/brain/t_{n}.py" for n in
                                        ("spec", "pairs", "store", "records", "rules", "runner")} | {"scripts/run_t.py"}
    assert not set(TR.T_PIPELINE_FILES) & (set(R_MEASURE_FILES) | set(T_MEASURE_FILES))
    assert set(S_HASHED_FILES) | set(TR.T_PIPELINE_FILES) | set(T_MEASURE_FILES) <= set(TR.T_HASHED_FILES)
    assert {"results/summary/s_lever.json", "results/summary/r_lever.json",
            "tests/brain/fixtures/t_oc_cluster.json"} <= set(TR.T_HASHED_FILES)
    assert set(TR.DECISION_FILES) == set(TR.T_HASHED_FILES) - set(R_MEASURE_FILES) - set(T_MEASURE_FILES)
    assert [f for f in TR.T_HASHED_FILES if not (ROOT / f).exists()] == []
    assert len(TR.pipeline_key()["key"]) == 64 and set(TR.pipeline_key()["files"]) == set(TR.T_PIPELINE_FILES)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_t_judge.py -q`
Expected: failures with `AttributeError: 'Runner' object has no attribute 'stage_jm'` and `FileNotFoundError` for `scripts/run_t.py`.

- [ ] **Step 3: Append the judgement methods to class `Runner`** (at the end of `flymon/brain/t_runner.py`, indented as methods)

```python
    # ---- the judgement (T.3 8-10, T.5) -------------------------------------------------------------------------------
    def _judge_chain(self, doc: dict, stage: str) -> None:
        """The judgement runs once, from clean trees, on the shared key every earlier block ran on and on the T
        measurement key block z was measured under (T.9.6: re-checked right before the judgement measurement and the
        judge — exit 7)."""
        if self.spec.smoke:
            refuse("smoke never measures the judgement set (T.3 4)")
        if not self.code_key or not self.t_measure_key:
            refuse("no code key or T measurement key given; the judgement must run on the code the gates ran on")
        if doc["z"].get("t_measure_key") != self.t_measure_key:
            refuse(f"the T measurement key {self.t_measure_key} is not block z's {doc['z'].get('t_measure_key')} "
                   f"(T.9.6)", EXIT_KEY)
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the set is used once (T.5)")
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
        """One judgement condition on the 64 pairs (resumable), L on z_lever and C / E0 on block h4's z; the block holds
        raw_check only."""
        sp = self.spec
        if name not in sp.cond_names:
            refuse(f"unknown condition {name}; T has {sp.cond_names}")
        stage = f"jm:{name}"
        doc = self._require(stage)
        self._judge_chain(doc, stage)
        rows = self._judgement_rows()
        seeds, cond, m = sp.judge_seeds(), sp.cond(name), self._m_for(name, doc)
        got = m.oracle(rows, cond, r_records.JUDGE_BLOCK, seeds)
        rc = r_records.raw_check(got, cond, [row_key(r) for r in rows], seeds)
        man = [dict(x, sha256=sha256_file(x["cache_file"])) for x in rc["manifest"]]
        body = dict(rc, manifest=man, seeds=seeds, condition=name, z={k: list(v) for k, v in m.z.items()},
                    wall_s=m.last_wall_s, jobs=m.last_jobs)
        self._write(stage, body)
        return body

    def _raws(self, doc: dict) -> tuple:
        raws, bad = {}, []
        for n in self.spec.cond_names:
            got, b = t_store.load_manifest(doc[f"jm:{n}"]["manifest"])
            raws[n] = got
            bad += [f"{n}: {x}" for x in b]
        return raws, bad

    def stage_seal(self) -> dict:
        """R.9.7 as is (T.3 9): pre-read validity (every raw entry's stored inputs = its condition's measurer's, so L's
        carry z_lever and C's / E0's h4 z), the manifest (192), the archive copy, the decision code hash pinned."""
        doc = self._require("seal")
        self._judge_chain(doc, "seal")
        sp = self.spec
        rows = self._judgement_rows()
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        want = {n: r_records.judge_inputs(self._m_for(n, doc), rows, sp)[n] for n in sp.cond_names}
        v = r_records.preread_validity({n: doc[f"jm:{n}"] for n in sp.cond_names}, raws, sp, _ztuple(self.ctx["z"]),
                                       keys, self.code_key, doc["reuse"]["records"]["repro_csc_sha256_none"], want)
        nl = sp.cond_names[0]
        seeds = sp.judge_seeds()
        n_rep, n_act = len(seeds["report"]), len(seeds["act"])
        und = [g["key"] for g in raws[nl] if not r_records._short(g["result"], n_rep, n_act) and not r_records.stats_ok(
            r_records.pair_stats(g["result"]["report"], self._z_lever(doc), sp.testable_min))]
        reasons = bad + v["reasons"] + ([f"{nl}: {len(und)} pair(s) with an undefined d′ on z_lever"] if und else [])
        status = t_rules.INVALID if v["invalid"] else (t_rules.NOT_READ if reasons else t_rules.SEALED)
        manifest = [dict(x, condition=n) for n in sp.cond_names for x in doc[f"jm:{n}"]["manifest"]]
        body = dict(status=status, reasons=reasons, invalid=v["invalid"], checks=v["checks"], manifest=manifest,
                    n_files=len(manifest), archive=None, decision=decision_key(), seeds=seeds,
                    z=dict(z_lever=doc["z"]["z_lever"], z_h4={k: list(x) for k, x in _ztuple(self.ctx["z"]).items()}),
                    set=dict(digest_e0_b=sp.digest_e0_b, digest_e0_a=sp.digest_e0_a, digest_keys=sp.digest_keys,
                             last_turn=sp.last_turn, n_b=sp.n_b, n_a=sp.n_a))
        if status == t_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=t_store.archive_copy([x["cache_file"] for x in manifest], dest,
                                                                              self.archive_root))
        self._write("seal", body)
        return body

    def _read(self, doc: dict, mark=None) -> dict:
        """The bands and records from the sealed raw files (sha re-checked): L read on z_lever, C and E0 on block h4's
        z (T.4); R.3's order with G_fail_S; T.7's sentence (T.9.5 / T.9.7); T.6's records."""
        sp = self.spec
        rows = self._judgement_rows()
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        if bad:
            refuse(f"raw files changed since the seal: {bad[:3]}")
        if mark is not None:
            mark()                                       # the read starts here
        seeds = sp.judge_seeds()
        s = {n: r_records.cond_summary(raws[n], sp.cond(n), sp, self._z_for(n, doc), keys, seeds)
             for n in sp.cond_names}
        L, C, E0 = (s[n] for n in sp.cond_names)
        if L["aggregate"] is None or C["aggregate"] is None:
            refuse("no (b) pair to aggregate")
        aL, aC = L["aggregate"], C["aggregate"]
        gf = t_rules.g_fail_s(L["pairs"], C["pairs"], sp)
        rb = t_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                               aL["naive_a"], sp)
        if rb["band"] == t_rules.NOT_READ:               # no number of the set leaves a NOT_READ read
            return dict(status=t_rules.NOT_READ, band=rb["band"], reason=rb["reason"],
                        reasons=["the (b) / (a) pair counts differ from the declared counts (T.4 line 1)"])
        ax, ratio = gf["axes"], doc["gate2"]["ratio"]
        names = list(sp.p.directions)
        k_even = doc["gate3"]["testable_b"]
        fields = dict(n=aL["testable_b"], c=aC["testable_b"], f_a=aL["F_a"], naive_a=aL["naive_a"], T=sp.last_turn,
                      k_even=k_even, pb_L=ax["b"]["pun_L"], pb_C=ax["b"]["pun_C"], pa_L=ax["a"]["pun_L"],
                      pa_C=ax["a"]["pun_C"], d_b=ax["b"]["net_drop"], d_a=ax["a"]["net_drop"],
                      rho1=f"{ratio[names[0]]['ratio']:.3f}", rho2=f"{ratio[names[-1]]['ratio']:.3f}",
                      reason=rb["reason"])
        out = dict(band=rb["band"], reason=rb["reason"], n=aL["testable_b"], c=aC["testable_b"], F_a=aL["F_a"],
                   naive_a=aL["naive_a"], n_b=aL["n_b"], n_a=aL["n_a"], g_fail=gf)
        if rb["band"] == t_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        labels = self.ctx["clusters"](rows)
        records = dict(r_records.compare(L, C, E0, sp),
                       clusters=t_records.clusters({n: s[n]["pairs"] for n in sp.cond_names}, labels),
                       alpha_fixed=t_records.alpha_fixed(raws[sp.cond_names[0]], C, keys, seeds,
                                                         _ztuple(self.ctx["z"]), sp),
                       z=dict(z_lever=doc["z"]["z_lever"], sd_ratio=doc["z"].get("sd_ratio_lever_over_h4"),
                              z_h4_reproduced=True))
        return dict(out, status=t_rules.READ, sentence=t_rules.sentence(rb["band"], fields), records=records,
                    pairs={n: s[n]["pairs"] for n in sp.cond_names}, oc_sha256=doc["oc"]["independent"]["sha256"],
                    oc_cluster_sha256=doc["oc"]["cluster"]["sha256"],
                    p_labels=dict(L=doc["gate2"]["label_L"], C=doc["gate2"]["label_C"]),
                    p_ratio={d: ratio[d]["ratio"] for d in names}, k_even=k_even)

    def stage_judge(self) -> dict:
        """Once (T.3 10). The marker is written before the band is computed. S.9.2 (T.5): with the marker and no judge
        block, judge is re-generated once — same sealed raw data (sha re-checked), the sealed decision code, no
        measurement; the block says resumed_after_mark and the mark's time; a second re-generation, a marker from
        another seal or decision code, or a judge block once written (DONE marker) refuses."""
        doc = self._require("judge")
        self._judge_chain(doc, "judge")
        if doc["seal"].get("status") != t_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: T reads only a sealed set (R.9.7)")
        sealed = (doc["seal"].get("decision") or {}).get("key")
        now = decision_key()["key"]
        if sealed != now:
            refuse(f"the decision code hash {now} is not the sealed {sealed}: the judgement reads only under the code "
                   f"it was sealed with (T.5)")
        if Path(DONE_MARKER).exists():
            refuse(f"{DONE_MARKER} exists: a judge block was written once; a discarded block does not reopen the set")
        resumed = None
        if Path(JUDGE_MARKER).exists():
            if Path(REREAD_MARKER).exists():
                refuse(f"{REREAD_MARKER} exists: judge was re-generated once after the mark already (T.5)")
            mk = json.loads(Path(JUDGE_MARKER).read_text())
            if mk.get("seal_written_at") != doc["seal"].get("written_at") or mk.get("decision_key") != sealed:
                refuse(f"{JUDGE_MARKER} belongs to another seal or decision code; no re-generation (T.5)")
            resumed = mk

        def mark():
            t_store.write_json(REREAD_MARKER if resumed else JUDGE_MARKER, dict(
                seal_written_at=doc["seal"].get("written_at"),
                seal_archive=(doc["seal"].get("archive") or {}).get("dir"),
                decision_key=now, code_key=self.code_key, t_measure_key=self.t_measure_key,
                pipeline_key=self.pipeline_key, read_at=_now()), self.plist)
        out = self._read(doc, mark)
        if out["status"] != t_rules.READ:
            return out                                   # NOT_READ: no block, the marker stays
        out = dict(out, resumed_after_mark=resumed is not None, mark_read_at=(resumed or {}).get("read_at"))
        block = self._write("judge", out)
        t_store.write_json(DONE_MARKER, dict(judge_written_at=block["written_at"]), self.plist)
        return block

    # ---- after reading (T.5 = S.5 = R.5; T.9.6) ----------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (T.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: this set is closed (T.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """T.5 row 2: an analysis or summary defect after reading — the same sealed raw data recomputed."""
        if not note:
            refuse("--note is required (T.5: the correction is recorded)")
        doc = self._require_after_judge("recompute")
        out = self._read(doc)
        dk = decision_key()["key"]
        entry = dict(note=note, status=out["status"], band=out["band"], reason=out["reason"], n=out.get("n"),
                     c=out.get("c"), F_a=out.get("F_a"), naive_a=out.get("naive_a"), g_fail=out.get("g_fail"),
                     sentence=out.get("sentence"), decision_key=dk,
                     decision_changed_since_seal=bool(dk != (doc["seal"].get("decision") or {}).get("key")),
                     differs_from_judge=bool(out["band"] != doc["judge"]["band"]
                                             or out.get("sentence") != doc["judge"].get("sentence")),
                     code_key=self.code_key, t_measure_key=self.t_measure_key, pipeline_key=self.pipeline_key,
                     git=git_state(), written_at=_now())
        t_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """T.5 row 3: a measurement defect after reading — INVALID_RUN; a replacement set is a new declaration (T.5)."""
        if not note:
            refuse("--note is required (T.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=t_rules.INVALID_RUN, note=note, judge_band=doc["judge"]["band"],
                    rule="T.5: the same set is never run again; a replacement set needs a new declaration (user)")
        self._write("invalid_run", body)
        return body

    def stage_rs_reread(self) -> dict:
        """T.9.6, record only and only after judge: R's and S's judgement raw (read only, sha-checked through their
        manifests) re-read under T's z rule — L on z_lever (α as chosen on h4 z when they were measured), C on h4 z —
        with S's guard and R.3's bands; their own bands stand."""
        doc = self._require_after_judge("rs_reread")
        if "rs_reread" in doc:
            refuse("block rs_reread exists; T never rewrites a recorded block")
        sp = self.spec
        nl, nc, _ = sp.cond_names
        z_l, z_h4 = self._z_lever(doc), _ztuple(self.ctx["z"])
        out = {}
        for name, other in (("R", self.ctx["r_doc"]()), ("S", self.ctx["s_doc"]())):
            raws, bad = {}, []
            for n in (nl, nc):
                got, b = t_store.load_manifest(other[f"jm:{n}"]["manifest"])
                raws[n], bad = got, bad + [f"{n}: {x}" for x in b]
            if bad:
                out[name] = dict(reasons=bad[:5])
                continue
            keys, seeds = [x["key"] for x in other[f"jm:{nl}"]["manifest"]], other[f"jm:{nl}"]["seeds"]
            L = r_records.cond_summary(raws[nl], sp.cond(nl), sp, z_l, keys, seeds)
            C = r_records.cond_summary(raws[nc], sp.cond(nc), sp, z_h4, keys, seeds)
            aL, aC = L["aggregate"] or {}, C["aggregate"] or {}
            out[name] = dict(n=aL.get("testable_b"), c=aC.get("testable_b"), F_a=aL.get("F_a"),
                             naive_a=aL.get("naive_a"), n_b=aL.get("n_b"), n_a=aL.get("n_a"),
                             g_fail=t_rules.g_fail_s(L["pairs"], C["pairs"], sp),
                             band_judged=(other.get("judge") or {}).get("band"), reasons=L["reasons"] + C["reasons"])
        body = dict(rows=out, note="T.9.6: 기록 전용 — R·S 판정 원자료를 T의 z 규칙으로 다시 읽은 값이며 R·S의 판정은 그대로다. "
                                  "L의 α는 측정 당시 h4 z로 고른 것이다.")
        self._write("rs_reread", body)
        return body
```

- [ ] **Step 4: Write the CLI**

```python
#!/usr/bin/env python3
"""Spec appendix T (T.9 wins over T.0-T.8): each engine variant's own reference-set z for the oracle, the same lever
(APL->MBON05 removal), one M2 judgement on a new set from the widened Gen-1 opponent pool. The controller runs every
stage; commit each block before the next.

    uv run python scripts/run_t.py --stage reuse                     # T.3 1 R's repro and gate ① reused (no pool)
    uv run python scripts/run_t.py --stage set                       # T.9.1 the set, list only (no pool)
    uv run python scripts/run_t.py --stage z                         # T.1 / T.9.4 reference-set z, unedited then lever
    uv run python scripts/run_t.py --stage gate1s                    # T.9.1 KC band on the T set's 55 odours
    uv run python scripts/run_t.py --stage smoke --workers 4         # 24_409_xxx / 25_209_xxx, even (b) 0 and 20 only
    uv run python scripts/run_t.py --stage oc                        # T.6 both operating characteristics (no pool)
    uv run python scripts/run_t.py --stage gate2_oc                  # T.6 gate ② operating characteristic (no pool)
    uv run python scripts/run_t.py --stage gate2                     # ② P_L and P_C on 25_200_000+i, h4 z
    uv run python scripts/run_t.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (T.3 6)
    uv run python scripts/run_t.py --stage gate3                     # T.9.2 C from R's even raw, L on z_lever
    uv run python scripts/run_t.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_t.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_t.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_t.py --stage recompute --note "..."    # after judge: analysis defect (T.5)
    uv run python scripts/run_t.py --stage invalid_run --note "..."  # after judge: measurement defect (T.5)
    uv run python scripts/run_t.py --stage rs_reread                 # after judge: R / S raw under T's z (record)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_SET_SHORT, STOP_Z_REPRO,
STOP_Z_DEGENERATE, STOP_STRENGTH_LEVER, gate ②'s three, STOP_EVEN_REPRO, STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH;
recorded, T stops), 5 a gate INVALID (z, gate1s, gate2, gate3), 6 seal not SEALED, judge NOT_READ or a smoke with
problems, 7 R's reuse condition or the T measurement key broke (T stops; no block), 2 a refusal (arguments, cwd,
connectome sha256, chain, uncommitted summary, dirty hashed file, data pins, R's even raw missing)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "set", "z", "gate1s", "smoke", "oc", "gate2_oc", "gate2", "gate3", "jm", "seal", "judge",
          "recompute", "invalid_run", "rs_reread")
POOL_STAGES = ("z", "gate1s", "smoke", "gate2", "gate3", "jm")
GATE_STAGES = ("reuse", "set", "z", "gate1s", "gate2", "gate3")
QUIET = ("pairs", "manifest", "records", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive", "checks",
         "g_fail", "set", "none", "lever", "record", "independent", "cluster", "C", "L", "L_h4_R")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (T.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import t_rules as R
    if stage in GATE_STAGES:
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
    from flymon.brain.h3_store import ROOT, canonical, code_key, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2

    from flymon.brain import t_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
    from flymon.brain.t_measure import ZMeasurer, t_measure_key
    from flymon.brain.t_spec import SPEC
    from flymon.brain.t_store import TCache

    try:
        ctx = t_runner.build_ctx(SPEC, NPZ)
        code = code_key(NPZ, files=R_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        cache = TCache(SPEC.smoke_cache_dir if a.stage == "smoke" else SPEC.cache_dir, code)
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            measurers = {}

            def measure(z):
                k = canonical(z)
                if k not in measurers:
                    measurers[k] = RMeasurer(pool, cache, SPEC, ctx["params"], ctx["readout"], z, ctx["types"],
                                             ctx["n_kc"])
                return measurers[k]
            zm = ZMeasurer(pool, ctx["params"], SPEC.p_type, ctx["types"])
            r = t_runner.Runner(measure, zm, ctx, SPEC, code=code, tcode=t_measure_key(NPZ),
                                pipeline=t_runner.pipeline_key())
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

- [ ] **Step 5: Run the T tests and the full suite**

Run: `uv run pytest tests/brain/test_t_judge.py -q`
Expected: 28 passed.

Run: `uv run pytest -q`
Expected: the whole suite passes (in the scratch worktree used to draft this plan it passed except the 12 `tests/test_run_l.py` tests that need the git-ignored `results/m0d/` files the worktree lacked).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/t_runner.py scripts/run_t.py tests/brain/test_t_judge.py
git commit -m "feat(t): t_runner (2) and run_t.py — jm per z, seal, judge once with one re-generation, recovery, rs_reread after judge; mutation tests"
```

---

## Self-review (done while writing)

- **Spec coverage:**

  | Spec item | Where it lives |
  |---|---|
  | T.0 disclosures (incl. T.9.4's possible stop) | no code; the OC notes (Task 4), the result section T.10 |
  | T.1 model, z rule, `STOP_Z_REPRO`; T.9.2 / T.9.3 / T.9.7 z only for the L oracle | Task 1 (`z_h4`), Task 3 (`t_measure`), Task 4 (`z_repro`), Task 5 (`stage_z`, `measure(z)`), Task 6 (jm per z, seal inputs) |
  | T.2 / T.9.1 set, declared values, cap, E1, clusters, `STOP_SET_SHORT`, unit tests | Task 2; stage `set` (Task 5); re-check per gate1s / jm / seal / judge (Tasks 5–6) |
  | T.9.1 gate ① supplement | Task 4 (`gate1s`, record), Task 5 (`stage_gate1s`) |
  | T.3 1 reuse | Task 1 key test, Task 4 `reuse`, Task 5 `stage_reuse` / `_reuse_now` |
  | T.3 4 smoke | Task 5 `stage_smoke` (Reading 10) |
  | T.6 / T.9.6 / T.9.7 OCs and fixture | Task 4 (`oc`, `oc_cluster`, fixture), Task 5 (`stage_oc`, `stage_gate2_oc`) |
  | T.3 6 / T.9.3 gate ② | Task 1 (`P_L` / `P_C`), Task 5 `stage_gate2` (h4 z; P_L on z_lever recorded) |
  | T.9.2 gate ③ | Task 3 (`RReadCache`, content-key test), Task 4 (`gate3`), Task 5 (`stage_gate3`) |
  | T.9.4 readout guard | Task 4 (`z_side`, `z_lever`), Task 5 (`stage_z`) |
  | T.3 8–10 judgement measurement, seal, judge once | Task 6 |
  | T.4 bands, G_fail_S, fixtures | Task 4 (exhaustive and boundary tests) |
  | T.5 / S.9.2 recovery and re-generation after the mark | Task 6 (fault-injection tests) |
  | T.6 records (z, α-fixed sensitivity, clusters) | Task 4 (`t_records`), Task 6 `_read` |
  | T.7 / T.9.5 / T.9.7 sentences | Task 4 (verbatim tests), Task 6 (B_Tb, SELECTED in the chain) |
  | T.8 / T.9.6 files, writes, keys, seeds, global collision, R · S re-read after judge | Tasks 1, 3, 5, 6; Global Constraints |

- **Placeholders:** none. Every code step has its full code; the code is the code that ran in the scratch worktree.
- **Type consistency:** `Runner` methods used by `through_gate3` / `_to_seal` exist in Tasks 5 / 6; block fields read later (`reuse.records.repro_csc_sha256_none`, `z.z_lever`, `z.t_measure_key`, `gate2.ratio`, `gate2.label_L`, `gate3.testable_b`, `oc.independent.sha256`, `oc.cluster.sha256`, `seal.decision.key`, `seal.written_at`) are written by the stages that own them; `Scripted.arms` / `oracle` / `activity` / `inputs` and `ZScripted.reference` / `rest` match `RMeasurer` and `ZMeasurer`.
- **Review Focus:** each item has its test in its owning task (listed in the section).

## Runs (controller)

The controller does every run, from `/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot` (branch `open-fly-brain-connectome`), with Bash `run_in_background: true` and `timeout: 7200000`, after the full suite passes at the final implementation commit.
- Subagents never run these steps.
- When a run finishes, read only its exit code and the last lines of its output.
- No rule or number changes after any run.
- Each block is committed before the next stage; the stage refuses otherwise.
- `results/summary/t_lever.json` is tracked; `results/t/` is git-ignored. `results/summary/r_lever.json`, `results/summary/s_lever.json`, `results/r/` and `results/s/` are never touched.
- Commit messages carry no trailers.

**The 2 h background limit:**
- The longest single runs are `gate2` (~0.3 h: 384 arms), `gate3` (~0.5 h: 39 L pairs on the H.4 seeds) and each `jm` (~0.5 h: 64 pairs); the judgement runs one condition per invocation. `z` (~1–2 min) and `gate1s` (a few minutes: 55 odours × 8 seeds × 2) are short; `z` is not cached and simply reruns from the start after a kill (it writes nothing until it finishes).
- If a run is stopped at the limit, or dies, rerun the same command. It resumes from `results/t/cache/` with only the missing units, and no block is written until a stage is complete.
- Never run two pool stages at once.

**Stop points:** `STOP_REUSE` (exit 3) or any exit 7; `STOP_SET_SHORT` (exit 3) or a `set` refusal (exit 2: the set does not reproduce); `STOP_Z_REPRO` or `STOP_Z_DEGENERATE` (exit 3) or a z `INVALID` (exit 5); `STOP_STRENGTH_LEVER` (exit 3) or a gate1s `INVALID`; a smoke with problems (exit 6) after one diagnosed fix attempt; any gate ② STOP (exit 3); a second gate ② `INVALID`; `STOP_EVEN_REPRO`, `STOP_EVEN_LOW_LEVER` or `STOP_C_EVEN_MISMATCH` (exit 3), a gate ③ `INVALID` or a gate ③ refusal (R's even raw missing); seal `NOT_READ` / `INVALID` (exit 6); judge `NOT_READ` (exit 6); and the judgement result itself. At a stop point:
1. Commit the block (a smoke block with problems is not committed; see step 5).
2. For the judgement result only: write the spec section T.10 and the README ledger first (step 12).
3. Overwrite `/private/tmp/claude-503/p-latest.md` with the full text:
   - the stage and its label or band;
   - the closing sentence verbatim;
   - the numbers behind it;
   - the commits;
   - what was or was not written (T.10, README ledger);
   - the decision T.7 / T.9.5 leaves to the user.
4. Stop. Do not continue to another stage.

0. **Preconditions (no pool):**
   1. `git status` must be clean and HEAD must be the final implementation commit; `uv run pytest -q` passes.
   2. `ls results/p/run/cache/p_arm | wc -l` → 192 (gate ②'s OC reads P's block); `ls results/r/cache/r_oracle | wc -l` → 277 (gate ③ reads R's even raw; `uv run pytest tests/brain/test_t_store.py -q` must not skip `test_rs_even_raw_is_found_by_content_key`).
   3. R's reused blocks are in history: `git merge-base --is-ancestor 22c934b HEAD && git merge-base --is-ancestor 6ad2201 HEAD`.
   4. `mkdir -p ~/flymon-archive/t`.
1. **Reuse (T.3 1):**
   1. Run `uv run python scripts/run_t.py --stage reuse` (seconds, no pool).
   2. On exit 0, commit: `git add results/summary/t_lever.json && git commit -m "results(t): reuse — shared key 3c2699c7 = R's; R repro 22c934b and gate 1 6ad2201 reused"`.
   3. On exit 3 (`STOP_REUSE`): commit `results(t): reuse STOP_REUSE — <reasons>`, then **stop point**.
2. **Set (T.2 / T.9.1):**
   1. Run `uv run python scripts/run_t.py --stage set` (no pool).
   2. On exit 0, commit: `git commit -m "results(t): set — widened pool, (b) 21 last turn 17 · (a) 43 last turn 103, digests 8c9729bf / e31b5526 / 37dde1ca reproduced"`.
   3. On exit 3 (`STOP_SET_SHORT`): commit, then **stop point**. On exit 2: **stop point** (nothing written).
3. **z (T.1, T.9.4, T.9.7):**
   1. Run `uv run python scripts/run_t.py --stage z` (~1–2 min).
   2. On exit 0, report z_lever (A, P means and SDs) and the SD ratios in one line, then commit: `git commit -m "results(t): z — unedited z = block h4 bit for bit; lever guard passed; z_lever A <m>/<sd>, P <m>/<sd>"`.
   3. On exit 3 (`STOP_Z_REPRO` or `STOP_Z_DEGENERATE`) or exit 5 (`INVALID`): commit `results(t): z <label> — <sentence or reasons>`, then **stop point** (T.9.4 disclosed that the guard may stop here).
4. **Gate ① supplement (T.9.1):**
   1. Run `uv run python scripts/run_t.py --stage gate1s` (a few minutes).
   2. On exit 0, commit: `git commit -m "results(t): gate 1 supplement — 55 T-set odours inside [0.03, 0.15] under the lever (min <·>, max <·>)"`.
   3. On exit 3 / 5: commit, then **stop point**.
5. **Smoke (T.3 4):**
   1. Run `uv run python scripts/run_t.py --stage smoke --workers 4`.
   2. Expect exit 0 and `problems: []`: P L edges `[2]`, C `[0]`, both labelled; oracle L `[2]` on z_lever, C / E0 `[0]` on h4 z; L CSC = R smoke's L (`860cba4f…`); C / E0 CSC = R's repro (`1aee8398…`).
   3. Report the `cost` estimate in one line, then commit: `git commit -m "results(t): smoke — lever edges 2, L on z_lever, C/E0 on h4 z, sha relations OK, cost <gate2 h> / <gate3 h> / <jm h per condition>"`.
   4. On exit 6: do not commit the block. Diagnose with `superpowers:systematic-debugging`, have a reviewed fix task (it must not touch a shared measurement file — that ends in exit 7 — nor `t_measure.py`, which would change the T measurement key after block z: then stop instead), discard the block (`git checkout -- results/summary/t_lever.json`), rerun once. A second failure is a **stop point**.
6. **Operating characteristics (T.6):**
   1. Run `uv run python scripts/run_t.py --stage oc` (seconds).
   2. Commit: `git commit -m "results(t): operating characteristics — independent G_fail_S 0.034/0.145/0.281/0.362, 0.752/0.949/0.961/0.549; cluster (ICC 0.3) null 0.401, harm 0.685/0.571 (fixture sha <8>)"`.
7. **Gate ②'s OC:**
   1. Run `uv run python scripts/run_t.py --stage gate2_oc` (seconds; reads P's 192 entries).
   2. Expect `p_stop_weakened` ≈ 0.998 / 0.750 / 0.080 / 0.000 / 0.000 at ρ 0.4 / 0.5 / 0.61 / 0.83 / 1.0 (S's values: same block, same h4 z).
   3. Commit: `git commit -m "results(t): gate 2 operating characteristic — P(STOP_PUNISH_WEAKENED) at true ratio 0.4/0.5/0.61/0.83/1.0 on h4 z, before gate 2"`.
8. **Gate ② (T.3 6, T.9.3):**
   1. Run `uv run python scripts/run_t.py --stage gate2` (~0.3 h; resumes after a stop).
   2. On exit 0, commit: `git commit -m "results(t): gate 2 PASS — P_L and P_C LEARNS_CONFIRMATORY on 25_200_xxx (h4 z), ratio r1 <·> r2 <·> (>= 0.5)"`.
   3. On exit 3: commit, then **stop point**.
   4. On exit 5 (INVALID): commit the block; fix the T code with a reviewed task (the pipeline key must change; a shared measurement file or `t_measure.py` must not change); run `uv run python scripts/run_t.py --stage gate2 --rerun-after-invalid` once. Its outcome is final.
9. **Gate ③ (T.9.2):**
   1. Run `uv run python scripts/run_t.py --stage gate3` (~0.5 h; resumes after a stop; reads R's C raw first and measures nothing when it does not reproduce).
   2. On exit 0, commit: `git commit -m "results(t): gate 3 PASS — C from R's even raw c_even 7 (h4 z), L re-measured on z_lever testable_b <k>/21"`.
   3. On exit 3 / 5 or a refusal (exit 2, R's raw missing): commit any block, then **stop point**.
10. **Judgement measurement (one condition per run, in this order: L, C, E0):**
    1. Run `uv run python scripts/run_t.py --stage jm --condition <L|C|E0>` (~0.5 h each; resumes after a stop).
    2. The blocks hold no statistic. Do not compute anything from `results/t/cache/` before the seal.
    3. Commit after each: `git commit -m "results(t): judgement measurement <cond> — 64 pairs complete (<z_lever|h4 z>, no band read)"`.
11. **Seal (R.9.7):**
    1. Run `uv run python scripts/run_t.py --stage seal` (no pool).
    2. On exit 0 (`SEALED`), check `find <seal.archive.dir> -type f | wc -l` → 192, then commit: `git commit -m "results(t): seal — pre-read validity OK, raw manifest (192 entries), decision code pinned, archive ~/flymon-archive/t/<dir>"`.
    3. On exit 6 (`NOT_READ` / `INVALID`): commit, then **stop point**. The judgement is not read.
12. **Judge (once):**
    1. Run `uv run python scripts/run_t.py --stage judge` (no pool).
    2. If it died after writing `results/t/judge_read.json` and before the block (no `judge` block, no `results/t/judge_done.json`), rerun the same command once (`resumed_after_mark: true`). Any further failure is a **stop point**.
    3. Commit: `git commit -m "results(t): judgement — <band> (<n>/21 vs <c>/21, F_a <f>/43, G_fail_S <true|false>)"`.
    4. Run `uv run python scripts/run_t.py --stage rs_reread` (record only, no pool) and commit: `git commit -m "results(t): R and S judgement raw re-read under T's z (record)"`.
    5. Write the result before stopping:
       - the spec section `### T.10 결과 (…, 판정 1회 — **<band>**)` at the end of appendix T, in S.10's form: implementation range and block commits; reuse / set / z (z_lever, SD ratios, guard values of both engines) / gate ① supplement (min / max, unedited values) / smoke / both OCs / gate ②'s OC / gate ② (labels, ℓ, ratios with CIs, P_L on z_lever) / gate ③ (c_even from R's raw, L's R raw on h4 z, L on z_lever) / seal (manifest count, archive dir); the judgement (n, c, F_a, naive_a, G_fail_S per axis with net drop and pass→fail / fail→pass); the closing sentence verbatim; the records table (testable / reward / punish / naive per axis, F_a, KC, APL, naive P_X / A, MBON05 per-cell saturation, transitions C → L), the cluster table, the α-fixed sensitivity, z renormalisation and the R · S re-read; "다음: T.7(T.9.5)대로 기록하고 사용자가 판단한다".
       - the README ledger lines, Korean (after S's bullet, ~line 196) and English (after S's bullet, ~line 322), in S's form: one bullet each with the band, the numbers and "next is the user's decision".
       - Commit: `git commit -m "docs(t): T.10 result — <band> (<n>/21 vs <c>/21, F_a <f>/43, G_fail_S <…>); README ledger ko/en"`.
    6. Go to the **stop point** with the result: the closing sentence verbatim and T.9.5's consequence (B_Tb closes T's claim only, not the lever; `SELECTED` stops at "M2 testability on the widened pool with per-engine z" and does not by itself justify a POOL F v4 learning test; every other band is recorded for the user). Pushing follows the standing FlyMon push rule (`gh auth switch --user lyutvs`); the verdict itself is reported, not acted on.
