# Spec V — Combined Lever (APL->MBON05 Removed + MBON05 Chain-Entry Cut), Each Engine's Own Reference-Set z, One M2 Judgement on a New KC-Input-Filtered Widened-Pool Set: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix V needs (as amended by V.9, which wins over V.0–V.8), and nothing else:
- a **reuse gate** (V.3 1, V.9.5 P3-12): R's no-edit reproduction (`22c934b`) while the shared measurement key is R's `3c2699c7…`; T's unedited reference-set z (T z block `0eb642e`) while the T measurement key is `7255f872…`; U's blocks (`d00fdf2` reuse, `3b83824` path, `d732b20` scan, `c0d09a7` kc) while the **V measurement key — U's measurement key, unchanged, literal `8a4e0930…`** — is the one they carry; re-checked at every later stage (exit 7).
- the **path reproduction** (V.3 2, V.9.5 P1-3, `STOP_V_PATH_REPRO`): the combined edit's static facts (2 APL->MBON05 edges + MBON05->MBON09 / 11 / 01 7 / 2 / 2 = 13 edges, CSC `2d359b8b…`, MBON05->APL untouched — else `INVALID`); the H.3 reference set + same-seed rest **measured afresh** on three engines with U's jobs — unedited (= T `none` rows, CSC `1aee8398…`), APL->MBON05 only (= T lever rows, `860cba4f…`), combined L_V (= U's `entry_f0` rows, every field, `2d359b8b…`); the oracle on the first 3 even pairs: APL-only = R's even L raw, unedited = R's even C raw.
- the **KC input** (V.3 3, V.9.1, V.9.5 P2-7 / P2-10, no oracle): the 173 E-grid odours of the full widened-pool permutation's rows that pass the filling-independent skips, each on both engines (unedited and L_V) at strength seeds 24_002_000+i (i < 8): per-odour medians, per-odour seed ranges, the ORN-cap arithmetic; the unedited medians equal U's `per_odour_none` bit for bit on the 53 shared odours (else `STOP_V_PATH_REPRO`).
- the **set** (V.2, V.9.1, V.9.5 P2-10 / P3-12), list only: T's generator from turn 0, T's used rows + T's 64-row set as used (T's set reproduced first), T's skips plus `kc_input` last (any odour of the row outside [0.03, 0.15] on either engine, values from block `kc_input` only), (b) 21 · (a) 43; digests, last turns, skip counts and the cluster table fixed in block `set` before any oracle on the set (`STOP_SET_SHORT`).
- **z_V** (V.3 5): T's readout guard on the path stage's L_V side (median Δ ≥ 5, zero share ≤ 0.25, SD > 0), `STOP_Z_DEGENERATE` with the SD in the sentence; the unedited z is the reused T `none` = block h4.
- the **combined-engine KC band** (V.3 6, V.9.1, V.9.5 P2-11): R's gate ① rule on the 112 calibration odours under L_V and the ORN cap (`STOP_STRENGTH_LEVER`, V's sentence); the V set's odours re-measured under L_V equal to block `kc_input`'s L_V values bit for bit (`STOP_V_PATH_REPRO`); the set's E0 odours on the unedited engine recorded.
- the **even gate** (V.3 7, V.9.3): C from R's even raw (h4 z) reproduces c_even 7 first (`STOP_EVEN_REPRO`); L_V's 39 even pairs on z_V; ① (b) net drop < 3 and (a) net drop ≤ 1 (`STOP_EVEN_PUNISH`); ② testable_b ≥ 11 (`STOP_EVEN_LOW_LEVER`); both sentences end with " (POOL 안 짝수 쌍 조건부)".
- **smoke** on 24_609_xxx (oracle, even (b) pairs 0 and 20, L_V on z_V, C / E0 on h4 z) and 25_409_xxx (P_L(V), P_C).
- the **operating characteristics** (V.6, V.9.5 P2-9): S.6's independent model (n_b 21 · n_a 43) with V's notes; T's cluster-correlated model recomputed on block `set`'s clusters, written to `results/v/oc_cluster.json` and its sha256 recorded; gate ②'s OC on h4 z — all before gate ②.
- **gate ②** (V.3 10, V.9.2): P_L(V) and P_C on 25_400_000+i (i < 32), both labelled on h4 z; S's order with the h4 ratio, then ℓ_L(z_V) / ℓ_C(h4 z) ≥ 0.5 in both directions (`STOP_PUNISH_WEAKENED`).
- the **`measurement_started`** marker (V.9.5 P2-8), then the **judgement** on V's set: L_V (z_V), C and E0 (h4 z), one condition per run, seeds 24_600_xxx, 64 pairs each; **seal** (R.9.7 as is, block z / set / kc_input and the decision code pinned, archive `~/flymon-archive/v/`); **judge** once (U.4's bands, V.7's sentences with V.9.4's cluster values), with S.9.2's single re-generation.

V is **one M2 judgement of a combined lever on a new set**. Its STOP labels and bands go to the user unchanged.

**Architecture:**
- New files only: `flymon/brain/v_*.py`, `scripts/run_v.py`, tests and test helpers. The one exception is a single `MODULES` entry in `tests/brain/test_p_spec.py` (Reading 1).
- **No measurement file is added or edited** (V.9.5 P1-3). V runs U's measurement file `flymon/brain/u_measure.py` unchanged: the lever is the string `u_edit(0.0, "chain_entry")` (= U's `entry_f0`), jobs go through `u_measure.UPool` and `UZMeasurer`, and R's `RMeasurer` runs unchanged on them. The **V measurement key** is `u_measure.u_measure_key(npz)` — U's key, declared as the literal `8a4e0930…` and checked at reuse, the marker, jm, seal and judge. No R / S / T / U file is edited.
- Modules:
  - `v_spec`: every V number (class `VSpec`, a subclass of U's `USpec`), the two gate-② P specs, `smoke()`.
  - `v_pairs`: the set generator (T's generator from turn 0, T's set as used, the `kc_input` skip), the KC candidate odours, the set's odours / E0 odours / judgement rows (checked against block `set`), the KC-filter record.
  - `v_store`: V's guard and writer, `VCache` (R's `RCache` with V's writer); T's `RReadCache` re-exported.
  - `v_records`: the combined edit's static facts and INVALID reasons, full-row comparison with U's rows, the KC input record and U comparison, the side record, the two-scale gate-② ratio.
  - `v_rules`: reuse, path, kc_input, set outcome, z_V, kc_band, even, gate ②, sentences, OCs.
  - `v_runner`: the stage chain.
- `scripts/run_v.py` is the CLI.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`, `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **V** (V.0–V.8) and its red-team amendments **V.9** (V.9.1–V.9.6; V.9 wins). V reuses appendix **U** (U.4 bands, U.5 recovery, U.9.1 path method, U.9.2 even filter numbers, U.9.3 chain-entry edge counts; U's implementation `1b50e15..ca71c89` and blocks `d00fdf2`, `3b83824`, `d732b20`, `c0d09a7`), **T** (T.1 z rule, T.2 / T.9.1 generator and skips, T.9.3 gate ② on h4 z, T.9.4 guard, T.9.5 consequence, T.9.6 / T.9.7 cluster method), **S** (S.4 guard, S.6 / S.9.7 OCs, S.9.2 re-generation) and **R** (R.3 bands, R.9.2 gate ① rule, R.9.3 sentence, R.9.7 seal and validity), and H.3 / H.4's reference set and guard.

| V.9 section | Overrides / adds |
|---|---|
| V.9.1 | KC input on **both** engines; the filter drops a row when any odour is outside [0.03, 0.15] on **either** engine; V.3 6's set-odour check becomes a bit-for-bit re-check (`STOP_V_PATH_REPRO`), the 112-odour check and ORN cap stay the gate. |
| V.9.2 | Gate ② ratio on h4 z **and** ℓ_L(z_V) / ℓ_C(h4 z), both directions, each ≥ 0.5; labels on h4 z. |
| V.9.3 | Even gates ① and ② kept; both STOP sentences end with " (POOL 안 짝수 쌍 조건부)". |
| V.9.4 | SELECTED carries "V 세트 군집 상관 모형에서 G_fail_S null 〈·〉, 오선택(harm) 〈·〉 / 〈·〉. 이 결과는 이 64쌍 조건부 기술 결과다." before "작동 특성은 V 세트 조건부 값이며 …". |
| V.9.5 | `u_measure` reused unchanged, V measurement key = U's literal `8a4e0930…`; path measured without cache; MBON05->APL check; σP disclosure; KC seed ranges; "before any oracle on the judgement set" and the `measurement_started` block; the cluster model as a runtime output with its sha; the KC candidate set definition; E-grid only (E0 recorded); `kc_input` last; reuse keys as literals; `STOP_Z_DEGENERATE` with SD; resume points. |
| V.9.6 | Order: reuse → path → kc_input → set → z → kc_band → even → smoke → oc (+ gate2_oc) → gate2 → measurement_started → jm → seal → judge → V.10 and README ledger. |

**Task count and review cost:** 6 tasks. Each task is one implementer run and one reviewer run (the `sdd-implementer` / `sdd-reviewer` agents), then one final whole-branch review: **13 subagent runs**. Expected size: ~1,940 lines of new code and ~1,900 lines of tests (U was ~2,030 / ~1,990; V has no measurement file but a set generator). All of it was drafted and run in a detached scratch worktree while this plan was written (results listed under "Validation done while planning"). Tasks 2 and 3 import Task 1's `v_spec` (Task 3's `v_fixtures` and tests need nothing else); Task 4 needs 1–3; Task 5 needs 4; Task 6 needs 5. The slow tests are the real-connectome tests of Tasks 1–2 (~1 min) and the runner worlds (each test walks the chain; ~5–10 min per file).

**Validation done while planning (scratch worktree, removed afterwards):** every V test passed (128 tests over the eight V test files), each task's tests also passed with only that task's part of `v_runner.py` in place (Task 4 alone, Tasks 4–5), and the spec-collector suites (`test_e_spec_store`, `test_p_spec`, `test_q_spec` … `test_u_spec`, `test_run_m`; 164 tests) passed with V's `MODULES` entry. On the real data: the shared, T and V measurement keys are R's `3c2699c7…`, T's `7255f872…` and U's `8a4e0930…`; `csc_facts` found exactly 13 changed edges (APL->MBON05 2, MBON05->MBON09 7, ->MBON11 2, ->MBON01 2), CSC `2d359b8b…`, MBON05->APL 2 edges with unchanged bytes (−7.8651862, −12.928195); `candidate_odours` listed 173 odours over 3,324 rows (53 of U's 55 T-set odours among them; 0 collision, 0 E1 and 0 glomerulus-duplicate rows); with U's unedited values for the 53 and an in-band placeholder for the rest, the generator filled (b) 21 by turn 30 and (a) 43 by turn 296 (V.0's estimate) — a list-only dry run, not V's set (V's set depends on the measured KC input). The real CLI ran `reuse` from a scratch commit: PASS (exit 0). **`path` and every later stage were not run** (they are the declared run). The full repository suite was not run in the scratch tree (only the suites above); Task 6's Step 5 runs it.

## Global Constraints

- **Commits carry no trailers.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No R, S, T or U file changes; no measurement file is added or changed.** `r_measure.R_MEASURE_FILES` must stay R's `3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc`; T's measurement key `7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374`; `u_measure.u_measure_key` (= the V measurement key) `8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523` (test in Task 1). No `r_*`, `s_*`, `t_*`, `u_*` module, `run_r.py`, `run_s.py`, `run_t.py`, `run_u.py`, their tests, fixtures or helpers are edited; `t_measure.py`, `h3_spec.py`, `u_measure.py` are never touched.
- **Never edit** `flymon/agent/e_*.py`, `encode_grid.py`, `q_*`, `p_*`, `o_*`, `n_*`, `h3_*`, `h4_*`, `k_jobs`, `d6a`, `plasticity`, `fly_pool`, `engine_cpu`, `flymon/battle/pool.py`, any `scripts/run_*.py` other than `run_v.py`. The one exception is Reading 1: one entry in `tests/brain/test_p_spec.py`'s `MODULES`. No module global of another track is monkeypatched at runtime (tests may monkeypatch).
- **V defines no job, rig, measurer or cache copy** (only `VCache`, overriding `put`), adds no `v_measure*.py`, and no V file loads code by path (`importlib`, `runpy`, `exec`) — test in Task 6. `v_records.csc_facts` builds an `Engine` and calls `u_measure.apply_u_edit` to read static weights only (no simulation).
- **Numbers live only in `v_spec`.** Every V number is a field or method of `flymon.brain.v_spec.SPEC` (`VSpec`) or of `smoke(SPEC)`; U's, T's, S's and R's are inherited; H.3's reference set, H.4's guard thresholds and ddof, the encoder's strength seeds and P numbers are read from their specs. Literal guards: `v_spec.py` may hold only Task 1's literal set; no V file holds an integer in 24_100_xxx, 24_300_xxx–24_309_xxx, 24_400_xxx–24_409_xxx, 24_500_xxx–24_509_xxx, 24_002_xxx, 25_000_000–25_009_999, 25_100_000–25_109_999, 25_200_000–25_209_999, 25_300_000–25_309_999, 23_000_000–23_009_999, 800_000–800_199, or 20261004 / 20261005. T's generator seed and T's cluster-OC seed are read from `t_spec.SPEC` (Reading 2).
- **The set is reached only through ctx's callables, from block `kc_input`'s values and checked against block `set`.** Inside V, only `v_pairs` (`v_set`, `_checked`, `judgement_rows`, `set_odours`, `set_e0_odours`), `v_runner.build_ctx`, `_set_call`, `_judgement_rows`, `stage_set` (the list, no measurement), `stage_kc_band` (odours only, no pair, no oracle — V.5), `stage_measurement_started`, `stage_jm`, `stage_seal`, `_read` and `_cost` (seed counts only) name it or its seeds (AST test, Task 6). `v_pairs` refuses a smoke spec.
- **Raw data of other tracks is read-only for V:** `results/summary/{r,t,u}_lever.json`, `results/t/z.json` (sha256 `fc8bb380…`), `results/u/scan.json` (sha256 `a4cddfd7…`, U's `entry_f0` rows), `results/r/cache/r_oracle/` (R's even raw, by content key through `RReadCache`), `results/p/run/cache/p_arm/` (gate ②'s OC), `results/summary/encoder_grid.json`, `results/summary/p_learning.json`, `results/summary/m0d.json`.
- **Writes are guarded.** Raw files go under `results/v/` (git-ignored by `results/*`); the summary is `results/summary/v_lever.json` (tracked). Anything else — `results/u/`, `results/t/`, `results/r/`, `results/s/`, `results/p/`, any other summary — is refused with SystemExit 2. The archive copy goes only under `~/flymon-archive/v/<seal id>/`, never over an existing directory. Every write is atomic.
- **Scripts:** tests find the repository with `ROOT = Path(__file__).resolve().parents[2]`; `scripts/run_v.py` has `if __name__ == "__main__":`; pools are never started from a heredoc or `python -c`.
- **Subagents never run a real or smoke stage and never write under `results/`.** Tests write only under `tmp_path`. The controller runs every stage (section "Runs (controller)").
- **Numbers from V / V.9, verbatim:**
  - **Model:** C3 (block `h4` Params). **L_V** = `u_edit(0.0, "chain_entry")` = `"u_apl_mbon05_x0.0+chain_entry"`: APL->MBON05 2 edges × 0 (+0.0), MBON05->MBON09 **7**, ->MBON11 **2**, ->MBON01 **2** set to 0 — **13** edges, CSC **`2d359b8b6947d542348b658db977e1edbe0db5919e7f4b1187598cb10ef1253a`**; MBON05->APL **2** edges untouched (V.0: −7.865, −12.929). The jobs report `edit_edges` 2 (the APL->MBON05 mask, U's convention); the 7 / 2 / 2 are the reference rows' `block_edges`. Readout A = MBON13, P = MBON05.
  - **z rule (oracle only):** L_V's oracle (even, judgement) on **z_V** (its own reference-set z, α chosen on it); C and E0 on block h4's z **A 10.78125 / 9.412096743243064, P 26.25 / 19.30889259728101**. Reference set = H.3's 48 odours, strength 0.35, 96 presentations + 96 same-seed rests, type counts summed over both hemispheres, mean and population SD (ddof 0). Guard per readout type: median(read − rest) **≥ 5**, zero share **≤ 0.25**, SD > 0.
  - **Path gate:** unedited rows ≡ T `none` (CSC **`1aee839811b8aa662588fbfab3050cee00962ddc8ecd4d2fd1361fd0c72692d5`**), APL-only rows ≡ T lever (CSC **`860cba4f9eead9d7a85b632f2c93c51790c5082fe80460975cb1300239c73e46`**), L_V rows ≡ U `entry_f0` (every field); T's rows from `results/t/z.json` sha256 **`fc8bb380eb4050c286624bf697422d6b57cb109ba55b13f91bf7d5f6c127762c`**, U's from `results/u/scan.json` sha256 **`a4cddfd7fbdec1dd8a95e93088aeecff2c1e9eca389454bcd8d81d4f45c0ab7e`** (U's path detail `57ae0c4ed11396421e21a91ae0bd5f88a20c2e8215e612a1225d6599b94be329` is checked at reuse); oracle on the first **3** even pairs (declared order), H.4 seeds, h4 z: APL-only ≡ R's even L raw, unedited ≡ R's even C raw.
  - **KC input / band:** strength seeds **24_002_000+i** (i < 8), s 1.0; band **[0.03, 0.15]** (closed); 112 calibration odours: median of per-odour medians in the band and the ORN cap (R.9.2); 173 candidate odours on both engines (V.9.5 P2-10); U's `per_odour_none` (block `c0d09a7`, equal at its KC points 0.6 / 0.7 / 0.8) bit for bit on the shared odours.
  - **Set:** T's generator (127 opponents, 1,986 combos, T's permutation seed), turns **0–1985**, (b) **21**, (a) **43**; skips in order cap → collision → E1 → used → glomerulus duplicate → in-set → POOL-only → **kc_input**.
  - **Even:** H.4 seeds (act 500–507, select 600–607, report 608–615), 39 even pairs; C from R's even raw must give c_even **7** (`STOP_EVEN_REPRO`); ① (b) net drop **< 3** and (a) net drop **≤ 1** (pun = #(−p ≥ 2), C on h4 z, L_V on z_V); ② testable_b **≥ 11**.
  - **Gate ②:** P_L(V) and P_C on **25_400_000+i** (i < 32), smoke **25_409_100–103**, c₁ from block n1 (0.608), labels on h4 z; order INVALID (one rerun) → L ≠ `LEARNS_CONFIRMATORY` `STOP_PUNISH_BROKEN` → C ≠ `STOP_P_REFERENCE` → h4 ratio ℓ_L / ℓ_C < **0.5** in either direction `STOP_PUNISH_WEAKENED` → ℓ_L(z_V) / ℓ_C(h4 z) < **0.5** in either direction `STOP_PUNISH_WEAKENED` → PASS.
  - **Judgement seeds:** act **24_600_000+i**, select **24_600_100+i**, report **24_600_200+i** (i < 8). Smoke oracle seeds **24_609_000–24_609_099**.
  - **Bands (first match, V.4 = U.4 = R.3):** 1 (b) ≠ 21 or (a) ≠ 43 → `NOT_READ`; 2 c ≥ 11 → B_결론없음; 3 n ≤ c → B_Tb; 4 n < 11 → B_결론없음; 5 n − c < 2 → B_결론없음; 6 G_fail_S → B_처벌가드; 7 F_a < 2 → B_Fa; 8 `SELECTED`. n and F_a on z_V, c on h4 z. G_fail_S: pun_C − pun_L ≥ **3** on (b) or (a).
  - **OC:** independent G_fail_S null **0.034 / 0.145 / 0.281 / 0.362**, harm **0.752 / 0.949 / 0.961 / 0.549** (n_b 21 · n_a 43); cluster model = T.9.6 / T.9.7's method (ICC **0.3**, 10⁵ draws, T's seed) on block `set`'s clusters; gate ②'s OC at true ratio 0.4 / 0.5 / 0.61 / 0.83 / 1.0 on h4 z.
  - **Reuse:** shared key **`3c2699c7…`** (R repro `22c934b`); T measurement key **`7255f872…`** (T z block `0eb642e`); V = U measurement key **`8a4e0930…`** (U blocks `d00fdf2` / `3b83824` / `d732b20` / `c0d09a7`).

## Readings of the spec (decided here — the controller's rulings)

None of these changes a judgement rule, a gate threshold or a STOP sentence; each fixes plumbing the spec leaves to the implementation. Ambiguities that would change one are under "OPEN — needs user".

1. **`test_p_spec.py` gets one entry.** Its `test_every_spec_module_is_enumerated` fails as soon as `flymon/brain/v_spec.py` exists. Task 1 adds `"flymon/brain/v_spec.py": "flymon.brain.v_spec"` to `MODULES` (the pattern Q, R, S, T and U used). It is the only edit to a non-V file; the reviewer flags it for the user.
2. **`VSpec(USpec)` and T's seeds.** V subclasses U's spec so R's records and bands, S's guard / gate ② / OC code, T's z and OC code and U's records read V's numbers through the same field names. Every seed-bearing field U declared is overridden (test). The generator's permutation seed and the cluster-OC seed are T's (`t_spec.SPEC.set_rng_seed` / `oc_cluster_seed`): `v_pairs` and `v_rules.oc_cluster` read them from `t_spec.SPEC`, and V's spec keeps U's `()` in those two fields (T's global collision test forbids another spec module declaring 20261004 / 20261005). V.6 says "T.9.6·T.9.7과 같은 방법"; T.6 fixes the method with seed 20261005, ICC 0.3 and 10⁵ draws, so V uses the same three (test: on T's own clusters V's function reproduces T's committed fixture's rows). T's declared set values, which `USpec` inherits, are cleared (`""`, `None`, `()`), so nothing reads them as V's.
3. **The lever, its edge count and the static check (V.1, V.9.5 P1-3).** The lever is the fixed string `u_edit(0.0, "chain_entry")`; U's `apply_u_edit` reports `edit_edges` = the APL->MBON05 mask size (2) on every job, and the chain-entry counts in the reference rows' `block_edges`. "편집 간선 수가 (i) 2 · (ii) 7/2/2" is therefore checked as: jobs' `edit_edges` [2] and the reference rows' `block_edges` = {MBON05->MBON01: 2, MBON05->MBON09: 7, MBON05->MBON11: 2}, plus the path stage's static facts (`v_records.csc_facts`: a fresh C3 engine, `apply_u_edit`, the edges whose bytes changed — 13: APL->MBON05 2 and the 7 / 2 / 2 — the CSC sha, the MBON05->APL edges' bytes before and after). V.1's MBON05->APL rule is "편집 전과 같아야 한다": bytes equal and 2 edges. V.0 prints the weights as −7.865 and −12.929; the float32 weights are −7.8651862 and −12.928195 (measured while planning), so the declared values are matched within V.0's printed precision (0.001), not by 3-decimal rounding (−12.928195 rounds to −12.928).
4. **The path gate (V.3 2, V.9.5 P1-3).** Stage `path` measures the reference set + rest with `UZMeasurer` (no cache) on three edits: `"none"` (the unedited engine — compared with T's `none` rows by U's `row_diffs` on T's fields, plus CSC), `u_edit(0.0)` (APL-only — with T's lever rows) and L_V (with U's `entry_f0` rows from `results/u/scan.json`, every field bit for bit, `v_records.full_row_diffs` — the same job on the same edit). Then the oracle on `even_rows[:3]` with h4 z and the H.4 seeds through V's cache (block `path`, whose entries exist only from V's own path runs): APL-only (`cond L` with `u_edit(0.0)`) against R's even L raw read through `RReadCache` under `u_runner.r_raw_spec` (T's = R's lever edit), labels stripped (U's `oracle_diffs`); unedited (`cond C`) against R's even C raw. A static-fact defect, wrong presentation counts, wrong edge labels or wrong chain-entry edges are `INVALID`; any bit difference is `STOP_V_PATH_REPRO`, the sentence naming the first failing check (`〈엔진〉` = 편집 없는 엔진 / APL→MBON05 제거 단독 엔진 / 조합 엔진, `〈기준〉` e.g. "U entry_f0 기준 집합 행", `〈다른 항목〉` the differing rows / pairs). The three sides are V.6's mechanism records (MBON05 ratio L_V / unedited, σ_V / σ_h4).
5. **The KC input (V.3 3, V.9.1, V.9.5 P2-7 / P2-10).** The candidate odours are `v_pairs.candidate_odours`: every row of T's permutation (turns 0–1985) passing cap, collision, E1, used (incl. T's set) and glomerulus-duplicate checks and not POOL-only (173 odours over 3,324 rows on the real data, 53 of them in U's 55 — measured while planning). Both engines are measured with `RMeasurer.activity` under block name `"kc_input"` (resumable per odour; the block name is part of the cache key, so these are fresh measurements, not U's cache). The record holds per engine the per-odour median (e_rules' `odour_activity`, as R's gate ①), the per-odour seed range, edges, CSC, and the ORN-cap arithmetic; the block keeps the per-odour medians (the set's only source, V.2). `INVALID` on wrong odour / seed counts, edges (none 0 / L_V 2), CSC (none `1aee8398…` / L_V `2d359b8b…`) or a candidate above the cap; then the unedited medians vs U's `per_odour_none` on the shared odours → `STOP_V_PATH_REPRO` ("편집 없는 엔진", "U KC 블록(c0d09a7) per_odour_none"). The band is a filter here, never a gate.
6. **The E1 skip (V.2, V.9.5 P2-10).** T's code checks E1 clashes on the finished set; V.9.5 lists "E1 충돌" among the skips, so `v_pairs` skips a row whose side's glomerulus set equals a different reachable odour's, right after the collision check, and also records `e1_clashes` on the finished set. On the real data no candidate row has an E1 clash (measured while planning: 0 rows), so the set is the same under either reading.
7. **The set (V.2).** `v_pairs.v_set` builds T's generator state (`t_pairs` on T's own spec), reproduces T's set first (which reproduces S's and R's), adds T's 64 rows to the used keys and glomerulus keys, walks turns 0–1985 with the skips in V.9.5 P3-12's order and fills (b) to 21, (a) to 43, stopping when both are full. `STOP_SET_SHORT` uses V.2's sentence. Block `set` holds T's summary fields (digests, last turns, counts, skip counts, cluster table) and the set's odour ids; later stages regenerate the set from block `kc_input`'s values and refuse (exit 2) unless every summary field equals block `set`'s (`check_v_set`). Block `set` also holds V.6's KC-input record (`kc_record`: odours outside the band per engine, candidate rows using one, R's / S's / T's set rows using one — records only). V.3 4's "판정 세트에 대한 어떤 오라클보다 먼저" is satisfied by the order (V.9.5 P2-8).
8. **z_V (V.3 5).** Stage `z` (no pool) reads the path stage's L_V side: T's `t_rules.z_lever` decision (edges 2, 96 + 96, guard per readout type, SD > 0) plus the chain-entry edges and CSC (`INVALID` otherwise); a failure uses V's sentence with "SD 〈·〉" = the failing type's population SD to 3 decimals (`0` when the type's counts are constant, from `zero_sd`). The unedited z is the reused T `none` side (`z_none_reused` records that it equals block h4).
9. **The combined-engine KC band (V.3 6, V.9.1, V.9.5 P2-11).** Stage `kc_band` measures under L_V with block name `"kc_band"` (fresh entries): the 112 calibration odours (R's `gate1_record` and `gate1`: median of per-odour medians in the band; the ORN cap from ctx) and the V set's odours (re-measured, compared bit for bit with block `kc_input`'s L_V values, and each checked in the band — implied by the filter, kept as a defensive check). Order: `INVALID` (R's gate ① validity, set-odour edges / CSC) → `STOP_V_PATH_REPRO` ("조합 엔진", "KC 입력 블록의 V 세트 냄새 값") → `STOP_STRENGTH_LEVER` (V.3 6's sentence, 〈걸린 조건·냄새와 값〉 = R's condition texts and "V 세트 냄새 〈id〉 〈v〉 ∉ [0.03, 0.15]"). V.9.5 P2-11's "E0 냄새 값은 기록한다" is read as: the set's distinct E0 odours measured on the unedited engine at the E0 strength (the E0 condition's engine), block name `"kc_band_e0"`, recorded only.
10. **The even gate (V.3 7, V.9.3).** U's `even_repro` (T's sentence, L = R's even L raw on h4 z) first — `STOP_EVEN_REPRO` and no L_V measured; then L_V's 39 pairs on `measure(z_V)`; R's gate ③ validity (L edges 2, C none on R's repro CSC, L ≠ C) → `INVALID`; then ① and ② (`v_rules.even`). `STOP_EVEN_LOW_LEVER` is R.9.3's sentence with "APL→MBON05 제거" replaced by "조합 지렛대" (V.3 7) and " (POOL 안 짝수 쌍 조건부)" appended after its final period, as U appended its own suffix; `STOP_EVEN_PUNISH` is V.3 7's sentence with the same suffix.
11. **Smoke, OC, gate ②, judgement, seal, judge, recovery** follow U's stages with L_V fixed. Smoke checks L_V's CSC (P arms and oracle) against block `path`'s L_V side and `2d359b8b…`. The OC stage computes T's cluster method on block `set`'s clusters, writes `results/v/oc_cluster.json` and records its file sha and table sha plus the three values V.9.4 quotes (`cluster_values`: the `null` row and the two `harm` rows of `g_fail`, 3 decimals); V.9.5 P2-9's "tests fixture는 그 계산의 단위 테스트용만" is Task 3's test (T's clusters reproduce T's fixture). Gate ② records P_L read on z_V (`t_records.p_zlever`). The seal pins block `z`, `set` and `kc_input` (canonical-JSON sha256) beside the decision code hash; judge and recompute refuse when they moved.
12. **Keys and caches (V.8, V.9.5 P1-3).** V's cache (`VCache` under `results/v/cache`) is keyed by the V measurement key (= U's). Every block carries `code_key` (the shared key), `t_measure_key`, `u_measure_key` (the V measurement key) and `pipeline_key` (V's own files: `v_spec`, `v_pairs`, `v_store`, `v_records`, `v_rules`, `v_runner`, `run_v.py`). The reuse check refuses any V measurement key other than U's literal at every stage (exit 7); `measurement_started`, jm, seal and judge also refuse unless every earlier block carries the current keys (V.8's "판정 측정 직전과 판정 직전"). V's hashed files = U's hashed files + V's pipeline files + `results/summary/u_lever.json`; the decision files are those minus R's, T's and U's measurement files.
13. **`measurement_started` (V.9.5 P2-8)** is its own stage between gate ② and `jm:L` (no pool): it re-checks the chain and keys and records the set digests, the judgement seeds, z_V and the three pinned block shas. Once committed, the V set is used (V.5).
14. **Mechanism records in the judgement (V.6).** The judge's records add R's `compare` (KC, saturation, naive values, transitions), T's cluster counts and α-fixed sensitivity (on V's cluster labels, `t_pairs.clusters`), z_V and its side record, the path stage's three mechanism records, the KC-input record and both gate-② ratios. No new contrast measurement (V.6).
15. **Sentences.** V.7 verbatim with V.9.3 / V.9.4: B_Tb with T.9.5's consequence in the sentence (as U), R.3's B_결론없음 lines + " (〈n〉/21 대 〈c〉/21, F_a 〈f_a〉/43, 조합 지렛대)", S.7's B_처벌가드 / B_Fa + " (조합 지렛대)", SELECTED with V.9.4's insert and T.9.5's consequence appended (as U). `〈마지막 턴〉` = block `set`'s `last_turn` (the turn where both axes filled); `〈k_cl〉` = the number of (b) cluster rows in block `set`. `STOP_V_PATH_REPRO` keeps V.3 2's template (one particle "을" for every 〈기준〉, as U). `STOP_REUSE` is V's own ("R·T·U 재사용 조건(V.3 1)이 깨졌다(…)"); the spec names the label only. The reused STOP labels (`STOP_EVEN_REPRO`, gate ②'s three) keep their sentences verbatim ("해당 문장" — see OPEN 1 and 2).
16. **Exit codes of `scripts/run_v.py`:** 0 PASS / SEALED / READ / a record stage; 2 a refusal; 3 a STOP; 5 a gate `INVALID` (path, kc_input, set, z, kc_band, even, gate2); 6 seal not SEALED, judge NOT_READ, or smoke with problems; 7 the reuse condition or a measurement key broke.

## OPEN — needs user

These change a STOP or band sentence's content; none changes a band, gate threshold or order. The code below implements the stated default, so implementation can proceed; **the controller asks the user before the stage that first reads each default (gate ② for 1, judge for 2 and 3) and does not run that stage until answered.**

1. **Gate ②'s sentences under V.9.2's second scale.** V.9.2 adds "z_V로 읽은 비 < 0.5 → `STOP_PUNISH_WEAKENED`" without a new sentence; S's sentence prints "ℓ_r1 〈L〉 대 〈C〉, ℓ_r2 …" and begins "APL→MBON05 제거 아래 …" (V.7 says "해당 문장"). Default in the code: when only the z_V ratio fails, the sentence is S's verbatim with ℓ_L(z_V) and ℓ_C(h4 z) filled in, and the block records `scale: "z_V"`; the wording "APL→MBON05 제거" is kept (as U kept it for its partial lever). Alternative: a V sentence naming "조합 지렛대" and "z_V 척도".
2. **Which ratio SELECTED quotes.** V.7's SELECTED says "같은 시드 P 비 ℓ_r1 〈rho1〉·ℓ_r2 〈rho2〉 ≥ 0.5"; after V.9.2 there are two ratios. Default: the h4-z ratios (T's and U's convention; both scales are in the judge records). Alternative: quote both.
3. **"편집 없는 엔진의 KC 입력 유효성으로 거른" in B_Tb and SELECTED.** V.7 wrote these before V.9.1 made the filter read both engines. Default: V.7's words verbatim (V.9 did not amend them). Alternative: "두 엔진의 KC 입력 유효성으로 거른".

## Review Focus

1. **The set changing between block `set` and the judgement** (a hand-edited `kc_input` value, a regenerated set with one different row, T's set no longer reproducing). Expect exit 2 from kc_band, jm, seal and judge and nothing written. Tests: Task 1 (`test_regenerated_set_must_equal_block_set_and_smoke_never_reaches_it`), Task 4 (`test_a_set_that_no_longer_matches_block_set_refuses_kc_band`), Task 6 (`test_mutation_a_changed_kc_input_block_refuses_the_judgement_measurement`, `test_judge_refuses_a_pinned_block_changed_after_the_seal`).
2. **A combined edit that is not the declared one reaching a measurement** (the chain entry not applied, other edge counts, the MBON05->APL edges touched, the APL-only edit in jm:L). Expect `INVALID` or `STOP_V_PATH_REPRO` at path, a smoke problem, or the seal's stored-inputs check reading `NOT_READ`. Tests: Task 2 (`test_mutation_an_edit_without_the_chain_entry_fails`, `test_csc_reasons_name_each_break`), Task 4 (`test_path_mutation_the_chain_entry_never_reached_the_engine`, `test_path_invalid_on_the_static_edit_facts`, `test_path_invalid_on_other_chain_entry_edges`), Task 6 (`test_mutation_l_raw_of_the_apl_only_edit_fails_the_preread_validity`).
3. **The filter at its boundaries** (an odour exactly 0.03 or 0.15, out of band on one engine only, a candidate without a value). Expect the closed band, either engine dropping the row, a refusal on a missing value. Tests: Task 1 (`test_kc_input_is_the_last_skip_and_reads_both_engines`, `test_a_missing_value_refuses`, `test_set_short_when_the_filter_empties_the_pool`), Task 4 (`test_set_filters_by_both_engines`).
4. **Gate ② passing on h4 z while z_V's ratio is below 0.5** (V.9.2's reason). Expect `STOP_PUNISH_WEAKENED` with `scale: "z_V"`, and no marker afterwards. Tests: Task 2 (`test_ratio_two_z_equals_ss_ratio_on_one_z_and_scales_with_sigma_a`), Task 3 (`test_gate2_both_scales_in_order`), Task 5 (`test_gate2_stop_on_the_z_v_scale_alone`).
5. **A key or a reused block drifting between stages** (another track edits `u_measure.py`; U's summary uncommitted; the V measurement key ≠ U's literal). Expect exit 7 and nothing written, at every stage for the reuse condition and before the marker, jm, seal and judge for the keys. Tests: Task 4 (`test_reuse_broken_after_reuse_refuses_with_7`, `test_reuse_stop_names_u_and_later_stages_refuse`), Task 6 (`test_keys_are_rechecked_before_the_marker_measurement_seal_and_judge`).

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/v_spec.py` (create) | `LEVER_V`, `P_C`, `P_L`, `VSpec(USpec)` (with `u_kc_none`), `SPEC`, `smoke` |
| `flymon/brain/v_pairs.py` (create) | `REASONS`, `SUMMARY_KEYS`, `sides`, `_Gen`, `candidate_odours`, `v_set`, `set_summary`, `check_v_set`, `judgement_rows`, `set_odours`, `set_e0_odours`, `kc_record`, `cluster_labels` |
| `flymon/brain/v_store.py` (create) | `ALLOWED_DIR`, `SUMMARY`, `SMOKE_SEEDS`, `guard`, `write_bytes`, `write_json`, `read_summary`, `write_summary_block`, `VCache`; re-exports `RReadCache`, `load_manifest`, `archive_copy` |
| `flymon/brain/v_records.py` (create) | `csc_facts`, `csc_reasons`, `full_row_diffs`, `kc_input_record`, `kc_u_diffs`, `side_record`, `ratio_two_z` |
| `flymon/brain/v_rules.py` (create) | outcome / band constants, `read_band` / `g_fail_s` / `even_repro` / `even_validity` (R's, S's, U's), `reuse`, `path`, `kc_input`, `set_outcome`, `z_v`, `kc_band`, `even`, `gate2`, `SENTENCES`, `sentence`, `OC_NOTES`, `oc`, `oc_cluster`, `cluster_values` |
| `flymon/brain/v_runner.py` (create in Task 4, extended in Tasks 5–6) | `ORDER`, `GATES`, `V_PIPELINE_FILES`, `V_HASHED_FILES`, `DECISION_FILES`, markers, `pipeline_key`, `decision_key`, `decision_pins`, `kc_values`, `apl_spec`, `u_rows_reader`, `build_ctx`, `Runner` |
| `scripts/run_v.py` (create) | the CLI, `make_measure`, `exit_code`, `__main__` guard |
| `tests/brain/test_v_spec.py`, `test_v_pairs.py`, `test_v_store.py`, `test_v_records.py`, `test_v_rules.py`, `test_v_runner.py`, `test_v_gate2.py`, `test_v_judge.py`, `v_fixtures.py`, `v_world.py` (create) | tests and helpers (`r_fixtures.py`, `s_fixtures.py`, `u_fixtures.py` are reused as they are) |
| `tests/brain/test_p_spec.py` (modify: one `MODULES` entry) | Reading 1 |

---

### Task 1: `v_spec` and `v_pairs` — V's numbers, the lever, the seed blocks, and the KC-filtered set generator

**Files:**
- Create: `flymon/brain/v_spec.py`, `flymon/brain/v_pairs.py`
- Modify: `tests/brain/test_p_spec.py` (one `MODULES` entry)
- Test: `tests/brain/test_v_spec.py`, `tests/brain/test_v_pairs.py`

**Interfaces:**
- Consumes: `u_spec.USpec` / `SPEC` / `p_lever`, `u_measure.u_edit`, `p_spec.SPEC` / `smoke`; `t_pairs` (`t_set`, `check_t_set`, `opponents`, `_glom`, `_gkey`, `_cap_fails`, `cluster_of`, `clusters`), `t_spec.SPEC`, `e_pairs`, `h4_pairs`, `l_pairs`, `q_pairs`, `odor_real`, `r_pairs` (`okey`, `row_key`, `check_set`), `s_pairs.s_set`.
- Produces: `LEVER_V: str`; `VSpec` (fields listed in the code; `u_kc_none(u_doc) -> {odour id: float}`), `SPEC`, `smoke(spec) -> VSpec`; `v_pairs.candidate_odours(pops, enc, params) -> dict(odours, cap, cap_hz, n_rows, n_turns)`; `v_set(pops, enc, params, kc, spec) -> dict` (T's summary fields + `b`, `a`, `odour_ids`); `set_summary(js) -> dict`; `check_v_set(js, block) -> list[str]`; `judgement_rows(pops, rc, enc, params, kc, block, spec) -> list`; `set_odours(pops, rc, enc, params, kc, block, spec) -> {id: odour}`; `set_e0_odours(pops, enc, params, kc, block, spec) -> {key: odour}`; `kc_record(pops, enc, params, kc, spec) -> dict`; `cluster_labels(rows) -> {pair key: label}`. `kc = {"none": {id: median}, "lever": {id: median}}`.

- [ ] **Step 1: Add V to the spec-module list (Reading 1)**

In `tests/brain/test_p_spec.py`, extend `MODULES`'s last line:

```python
           "flymon/brain/t_spec.py": "flymon.brain.t_spec", "flymon/brain/u_spec.py": "flymon.brain.u_spec",
           "flymon/brain/v_spec.py": "flymon.brain.v_spec"}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/brain/test_v_spec.py
"""Spec V.1-V.3 / V.6 / V.9: every V number in v_spec; V's new blocks (judgement 24_600_xxx, smoke 24_609_xxx, gate ②
25_400_000+i and its smoke 25_409_xxx, their training seeds) collide with no declared seed of any other spec module (U
and T included); every seed field U declared is overridden while T's generator / cluster seeds stay T's; the lever is
the fixed string u_edit(0.0, "chain_entry") on L and on the lever P spec, and the edit string is in every cache key;
T's declared set values are cleared; the two gate-② P specs differ in the one edit only; no V file holds another
track's block as a literal; the shared, T and V measurement keys are R's, T's and U's."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.agent import e_spec
from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.config import Params
from flymon.brain.p_spec import SPEC as P
from flymon.brain.r_measure import RMeasurer
from flymon.brain.t_spec import SPEC as T
from flymon.brain.u_measure import u_edit
from flymon.brain.u_spec import SPEC as U
from flymon.brain.u_spec import USpec
from flymon.brain.v_spec import LEVER_V, P_C, P_L, SPEC, VSpec, smoke
from tests.agent.test_e_spec_store import _module_seeds
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000


def _declared_without_v() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/v_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


V_SEEDS = (set(SPEC.judge_act_seeds) | set(SPEC.judge_select_seeds) | set(SPEC.judge_report_seeds)
           | set(SPEC.smoke_seeds) | set(SPEC.p.seeds) | set(SM.p.seeds) | set(SPEC.p_c.seeds) | set(SM.p_c.seeds))
V_TRAIN = set(SPEC.p.train_seeds()) | set(SM.p.train_seeds()) | set(SPEC.p_c.train_seeds()) | set(SM.p_c.train_seeds())


def test_v_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/v_spec.py") == "flymon.brain.v_spec"


def test_new_blocks_are_the_declared_ones():
    assert SPEC.judge_seeds() == dict(act=list(range(24_600_000, 24_600_008)),
                                      select=list(range(24_600_100, 24_600_108)),
                                      report=list(range(24_600_200, 24_600_208)))
    assert SPEC.p.seeds == SPEC.p_c.seeds == tuple(range(25_400_000, 25_400_032))
    assert SM.p.seeds == SM.p_c.seeds and len(SM.p.seeds) == 4 and all(25_409_000 <= s < 25_410_000 for s in SM.p.seeds)
    assert SPEC.smoke_seeds == tuple(range(24_609_000, 24_609_100))
    sm = {s for v in SM.even_seeds().values() for s in v}
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 7
    with pytest.raises(ValueError, match="smoke"):
        SM.judge_seeds()


def test_global_seed_collision():
    """V's blocks and their derived training seeds against every other declared block (U, T, S, R, P, Q, the encoder,
    H.4 ... d6a)."""
    declared = _declared_without_v()
    for s in (500, 615, 24_002_000, 24_400_000, 24_500_000, 24_509_000, 25_200_000, 25_300_000, 25_309_000,
              23_000_000, 24_000_000, 22_000_000):
        assert s in declared, s
    assert not V_SEEDS & declared
    assert not {s for s in V_SEEDS if (s - BASE) // STRIDE in declared}
    assert not V_TRAIN & declared
    assert not {s for s in V_TRAIN if (s - BASE) // STRIDE in declared}
    assert not V_TRAIN & V_SEEDS


def test_every_u_seed_field_is_overridden_and_ts_generator_seeds_stay_ts():
    u_seed_fields = {f.name for f in dataclasses.fields(USpec) if "seed" in f.name} - {"set_rng_seed",
                                                                                         "oc_cluster_seed"}
    for name in u_seed_fields | {"p", "p_c"}:
        assert getattr(SPEC, name) != getattr(U, name), name
    assert SPEC.set_rng_seed == () == SPEC.oc_cluster_seed           # V generates with T's spec (v_pairs / v_rules)
    assert (T.set_rng_seed, T.oc_cluster_seed) == (20261004, 20261005)
    fields = _module_seeds("flymon.brain.v_spec")
    assert not fields & e_spec.track_seeds(E)
    assert not fields & (set(U.judge_act_seeds) | set(U.smoke_seeds) | set(U.p.seeds))


def test_ts_declared_set_values_are_cleared():
    assert (SPEC.digest_e0_b, SPEC.digest_e0_a, SPEC.digest_keys) == ("", "", "")
    assert (SPEC.last_turn, SPEC.last_turn_b, SPEC.n_set_odours) == (None, None, None)
    assert SPEC.skipped_declared == SPEC.clusters_b == SPEC.clusters_a == ()
    assert (SPEC.first_turn, SPEC.n_b, SPEC.n_a) == (0, 21, 43)


def test_the_lever_is_the_combined_edit_everywhere():
    assert LEVER_V == u_edit(0.0, "chain_entry") == "u_apl_mbon05_x0.0+chain_entry"
    assert SPEC.lever_edit == LEVER_V and SPEC.cond("L").edit == LEVER_V and SPEC.p.o.on_edit == LEVER_V
    assert SPEC.p is P_L and SPEC.p_c is P_C and P_C.o.on_edit == "none"
    assert SPEC.cond("C").edit == SPEC.cond("E0").edit == "none" and SPEC.lever_edges == 2
    assert smoke(SPEC).p.o.on_edit == LEVER_V


def test_the_edit_is_in_every_cache_key():
    """V.3 2 unit test: the lever string is an input of every KC-activity and oracle cache entry."""
    m = RMeasurer(None, None, SPEC, Params(), {"A": "MBON13", "P": "MBON05"}, {"A": (1.0, 1.0), "P": (1.0, 1.0)},
                  ["MBON13", "MBON05"], 100)
    a = m._act_inputs("X|Y", {"G": 1.0}, SPEC.lever_edit, 1.0, [1], "kc_input")
    n = m._act_inputs("X|Y", {"G": 1.0}, SPEC.no_edit, 1.0, [1], "kc_input")
    assert a["edit"] == LEVER_V and n["edit"] == "none" and a != n
    row = dict(axis="b", turn=0, x="m1", y="m2", odor_x={"G": 1.0}, odor_y={"G": 2.0})
    assert m.inputs(row, SPEC.cond("L"), "even", SPEC.h4_seeds())["edit"] == LEVER_V


def test_gate2_specs_differ_in_the_one_edit_only():
    diff = {f.name for f in dataclasses.fields(type(P.o)) if getattr(SPEC.p.o, f.name) != getattr(P_C.o, f.name)}
    assert diff == {"o1_conditions"}
    diff_p = {f.name for f in dataclasses.fields(type(P.o)) if getattr(P_C.o, f.name) != getattr(P.o, f.name)}
    assert diff_p == {"o2_seed0", "smoke_seed0"}
    for f in dataclasses.fields(type(P)):
        if f.name != "o":
            assert getattr(SPEC.p, f.name) == getattr(P_C, f.name) == getattr(P, f.name), f.name


def test_numbers():
    assert isinstance(SPEC, USpec) and isinstance(SPEC, VSpec)
    assert SPEC.sha_combined == "2d359b8b6947d542348b658db977e1edbe0db5919e7f4b1187598cb10ef1253a"
    assert SPEC.sha_none == "1aee839811b8aa662588fbfab3050cee00962ddc8ecd4d2fd1361fd0c72692d5"
    assert SPEC.sha_zero == "860cba4f9eead9d7a85b632f2c93c51790c5082fe80460975cb1300239c73e46"
    assert (SPEC.n_lever_edges_total, SPEC.mbon05_apl_edges, SPEC.mbon05_apl_weights, SPEC.mbon05_apl_weight_tol) == (
        13, 2, (-7.865, -12.929), 0.001)
    assert SPEC.contrast_declared()["chain_entry"] == {"MBON05->MBON09": 7, "MBON05->MBON11": 2, "MBON05->MBON01": 2}
    assert SPEC.u_measure_key_u == "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    assert dict(SPEC.u_commits) == {"reuse": "d00fdf2", "path": "3b83824", "scan": "d732b20", "kc": "c0d09a7"}
    assert (SPEC.u_summary, SPEC.u_path_detail, SPEC.u_scan_detail, SPEC.u_entry_point) == (
        "results/summary/u_lever.json", "results/u/path.json", "results/u/scan.json", "entry_f0")
    assert SPEC.u_path_detail_sha256 == "57ae0c4ed11396421e21a91ae0bd5f88a20c2e8215e612a1225d6599b94be329"
    assert SPEC.u_scan_detail_sha256 == "a4cddfd7fbdec1dd8a95e93088aeecff2c1e9eca389454bcd8d81d4f45c0ab7e"
    assert SPEC.t_measure_key_t == "7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374"
    assert SPEC.r_shared_key == "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    assert (SPEC.even_drop_b_lt, SPEC.even_drop_a_le, SPEC.path_even_n) == (3, 1, 3)
    assert (SPEC.summary, SPEC.cache_dir, SPEC.smoke_cache_dir, SPEC.archive_root) == (
        "results/summary/v_lever.json", "results/v/cache", "results/v/smoke/cache", "~/flymon-archive/v")
    assert (SPEC.path_detail, SPEC.kc_input_detail, SPEC.smoke_detail, SPEC.oc_cluster_out) == (
        "results/v/path.json", "results/v/kc_input.json", "results/v/smoke.json", "results/v/oc_cluster.json")
    assert (SPEC.bar_b, SPEC.margin, SPEC.f_a_min, SPEC.g_fail_drop, SPEC.p_ratio_min, SPEC.c_even_expected) == (
        11, 2, 2, 3, 0.5, 7)
    assert (SPEC.z_guard_med_min, SPEC.z_guard_zero_max, SPEC.z_ddof) == (5.0, 0.25, 0)
    assert SPEC.valid_band == (0.03, 0.15) and SPEC.kc_seeds() == tuple(E.strength_seeds) and SPEC.n_calib == 112
    assert SPEC.kc_seeds() == tuple(range(24_002_000, 24_002_008))
    for f in ("settle_ms", "read_ms", "window_ms", "alphas", "fixed_alphas", "active_fx", "config", "strength",
              "e0_strength", "oc_q", "oc_c", "oc_naive_max", "smoke_pairs", "r_cache_dir", "r_summary",
              "gate2_oc_rhos", "oc_harm", "r_even_L_h4", "z_h4", "oc_cluster_icc", "oc_cluster_draws",
              "oc_cluster_g"):
        assert getattr(SPEC, f) == getattr(U, f) == getattr(T, f), f
    for f in ("chain_types", "kc_types", "t_z_detail_sha256", "t_summary", "t_z_detail", "t_none_edit",
              "t_lever_edit", "contrast_edges"):
        assert getattr(SPEC, f) == getattr(U, f), f


def test_u_kc_none_reads_one_agreeing_map():
    pts = {k: {"record_set": {"per_odour_none": {"A|B": 0.05}}} for k in SPEC.u_kc_points}
    assert SPEC.u_kc_none({"kc": {"points": pts}}) == {"A|B": 0.05}
    pts["0.7"]["record_set"]["per_odour_none"] = {"A|B": 0.06}
    with pytest.raises(ValueError):
        SPEC.u_kc_none({"kc": {"points": pts}})


def test_smoke_changes_scale_only():
    for f in dataclasses.fields(VSpec):
        if f.name not in ("p", "p_c", "smoke", "workers"):
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.workers == 4


ALLOWED_V_SPEC = {25_400_000, 25_409_000, 0.0, 13, 2, 4, -7.865, -12.929, 0.001, 0, 43, 3, 1, 24_600_000, 24_600_008,
                  24_600_100, 24_600_108, 24_600_200, 24_600_208, 24_609_000, 24_609_100}


def test_v_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/v_spec.py").read_text())
    nums = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant):
            nums.add(-n.operand.value)
        elif isinstance(n, ast.Constant) and type(n.value) in (int, float):
            nums.add(n.value)
    assert nums - {7.865, 12.929} <= ALLOWED_V_SPEC, nums - ALLOWED_V_SPEC


def test_no_v_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("v_*.py")) + sorted((ROOT / "scripts").glob("run_v*.py"))
    assert len(files) >= 1
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_100_000 <= i < 24_101_000 or 24_300_000 <= i < 24_310_000
               or 24_400_000 <= i < 24_410_000 or 24_500_000 <= i < 24_510_000 or 24_002_000 <= i < 24_003_000
               or 25_000_000 <= i < 25_010_000 or 25_100_000 <= i < 25_110_000 or 25_200_000 <= i < 25_210_000
               or 25_300_000 <= i < 25_310_000 or 23_000_000 <= i < 23_010_000 or 800_000 <= i < 800_200
               or i in (20261004, 20261005)}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_t_and_v_measurement_keys_are_rs_ts_and_us():
    """V.3 1 / V.9.5 P1-3: the key over R_MEASURE_FILES is R's, the T measurement key is T's z block's and the V
    measurement key (u_measure.u_measure_key, unchanged) is U's literal; V edits none of their files."""
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import u_measure_key
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
    assert t_measure_key(str(NPZ))["key"] == SPEC.t_measure_key_t
    assert u_measure_key(str(NPZ))["key"] == SPEC.u_measure_key_u
```

```python
"""V's set generator (V.2, V.9.1, V.9.5 P2-10 / P3-12) on the real connectome and encoder summary, list only: T's set
reproduces first and joins used; the candidate odours are the full permutation's rows past the filling-independent
skips (173 odours, 53 of them U's 55); kc_input is the last skip and reads both engines; (b) 21 · (a) 43 from turn 0;
no V key or glomerulus key repeats a used one; STOP_SET_SHORT when the filter empties the pool; a missing value
refuses; the regenerated set must equal block set; smoke never reaches the set; the filter record."""
import json
from pathlib import Path

import pytest

from flymon.agent import e_pairs
from flymon.brain import t_pairs, v_pairs
from flymon.brain.r_pairs import okey
from flymon.brain.t_spec import SPEC as T
from flymon.brain.v_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="no connectome")


@pytest.fixture(scope="module")
def real():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    enc = json.loads((ROOT / SPEC.encoder_summary).read_text())
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    cand = v_pairs.candidate_odours(pops, enc, cfg.params)
    return dict(pops=pops, enc=enc, params=cfg.params, cand=cand,
                rc={str(t): len(v) for t, v in pops.receptor_types.items()})


def _kc(real, none=None, lever=None):
    ids = list(real["cand"]["odours"])
    return {"none": {o: (none or {}).get(o, 0.05) for o in ids}, "lever": {o: (lever or {}).get(o, 0.05) for o in ids}}


def test_candidates_are_the_filling_independent_rows(real):
    c = real["cand"]
    assert len(c["odours"]) == 173 and c["n_turns"] == 1986 and c["n_rows"] == 3324
    assert c["cap_hz"] == pytest.approx(1000 / 3) and all(v <= c["cap_hz"] for v in c["cap"].values())
    u55 = json.loads((ROOT / SPEC.u_summary).read_text())["kc"]["points"]["0.6"]["record_set"]["per_odour_none"]
    assert len(set(c["odours"]) & set(u55)) == 53


def test_set_from_turn_0_with_ts_set_used(real):
    js = v_pairs.v_set(real["pops"], real["enc"], real["params"], _kc(real), SPEC)
    assert js["status"] == "OK" and (js["n_b"], js["n_a"]) == (21, 43)
    assert min(r["turn"] for r in js["b"] + js["a"]) < T.last_turn         # walked from turn 0, not after T's turns
    assert js["skipped"]["collision"] == js["skipped"]["e1"] == js["skipped"]["glom_dup"] == 0
    assert js["skipped"]["kc_input"] == 0 and js["e1_clashes"] == [] and js["cap_fails"] == [] and js["all_off_pool"]
    t = t_pairs.t_set(real["pops"], real["enc"], T, real["params"])
    t_keys = {e_pairs.egrid_key(r) for r in t["b"] + t["a"]}
    assert not t_keys & {e_pairs.egrid_key(r) for r in js["b"] + js["a"]}
    assert len({e_pairs.egrid_key(r) for r in js["b"] + js["a"]}) == 64
    assert set(js["odour_ids"]) <= set(real["cand"]["odours"])


def test_kc_input_is_the_last_skip_and_reads_both_engines(real):
    base = v_pairs.v_set(real["pops"], real["enc"], real["params"], _kc(real), SPEC)
    target = base["odour_ids"][0]
    for side in ("none", "lever"):
        bad = {target: 0.0299} if side == "none" else {target: 0.1501}
        js = v_pairs.v_set(real["pops"], real["enc"], real["params"],
                           _kc(real, **{side: bad}), SPEC)
        assert target not in js["odour_ids"] and js["skipped"]["kc_input"] > 0
    edge = v_pairs.v_set(real["pops"], real["enc"], real["params"], _kc(real, none={target: 0.03}, lever={target: 0.15}),
                         SPEC)
    assert target in edge["odour_ids"]                     # [0.03, 0.15] is closed


def test_set_short_when_the_filter_empties_the_pool(real):
    js = v_pairs.v_set(real["pops"], real["enc"], real["params"],
                       _kc(real, none={o: 0.01 for o in real["cand"]["odours"]}), SPEC)
    assert js["status"] == "STOP_SET_SHORT" and (js["n_b"], js["n_a"]) == (0, 0) and js["last_turn"] == 1985


def test_a_missing_value_refuses(real):
    kc = _kc(real)
    del kc["lever"][next(iter(kc["lever"]))]
    with pytest.raises(ValueError, match="no lever value"):
        v_pairs.v_set(real["pops"], real["enc"], real["params"], kc, SPEC)


def test_regenerated_set_must_equal_block_set_and_smoke_never_reaches_it(real):
    kc = _kc(real)
    js = v_pairs.v_set(real["pops"], real["enc"], real["params"], kc, SPEC)
    block = json.loads(json.dumps(v_pairs.set_summary(js)))
    assert v_pairs.check_v_set(js, block) == []
    rows = v_pairs.judgement_rows(real["pops"], real["rc"], real["enc"], real["params"], kc, block, SPEC)
    assert len(rows) == 64 and all("odor_x" in r and "odor_x_e0" in r for r in rows)
    od = v_pairs.set_odours(real["pops"], real["rc"], real["enc"], real["params"], kc, block, SPEC)
    assert sorted(od) == sorted(js["odour_ids"])
    e0 = v_pairs.set_e0_odours(real["pops"], real["enc"], real["params"], kc, block, SPEC)
    assert 0 < len(e0) <= 128
    tampered = dict(block, digest_keys="0" * 64)
    with pytest.raises(ValueError, match="digest_keys"):
        v_pairs.judgement_rows(real["pops"], real["rc"], real["enc"], real["params"], kc, tampered, SPEC)
    with pytest.raises(ValueError, match="smoke"):
        v_pairs.judgement_rows(real["pops"], real["rc"], real["enc"], real["params"], kc, block, smoke(SPEC))


def test_kc_record_counts_dropped_odours(real):
    u55 = json.loads((ROOT / SPEC.u_summary).read_text())["kc"]["points"]["0.6"]["record_set"]["per_odour_none"]
    kc = _kc(real, none={o: v for o, v in u55.items()})
    rec = v_pairs.kc_record(real["pops"], real["enc"], real["params"], kc, SPEC)
    low = sorted(o for o, v in u55.items() if v < 0.03 and o in real["cand"]["odours"])
    assert rec["outside"]["none"] == low and rec["outside"]["lever"] == [] and rec["dropped"] == low
    assert rec["set_rows"]["R"]["n"] == 53 and rec["set_rows"]["S"]["n"] == 64 and rec["set_rows"]["T"]["n"] == 64
    assert rec["candidate_rows_dropped"] > 0


def test_cluster_labels_are_ts():
    rows = [dict(axis="b", turn=0, x="m1", y="m2", opp_x=["FIRE"], opp_y=["WATER"]),
            dict(axis="a", turn=1, x="m1", y="m2", opp_x=["GROUND"], opp_y=["GROUND"])]
    assert v_pairs.cluster_labels(rows) == t_pairs.clusters(rows)
    assert okey("FIRE", ("GRASS",)) == "FIRE|GRASS"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_v_spec.py tests/brain/test_v_pairs.py -q`
Expected: FAIL at collection (`ModuleNotFoundError: No module named 'flymon.brain.v_spec'`).

- [ ] **Step 4: Write `v_spec`**

```python
"""Every number of spec appendix V as amended by V.9 (V.9 wins over V.0-V.8): the combined lever L_V (APL->MBON05 removed
and the MBON05->MBON09 / MBON11 / MBON01 chain entry cut — u_measure.u_edit(0.0, "chain_entry"), 13 CSC edges), each
engine variant's own reference-set z for the oracle (z_V for L_V), a new set from T's widened-pool generator filtered by
the per-odour KC input of both engines, and one M2 judgement on it.
- VSpec subclasses U's USpec (and through it T's TSpec, S's and R's), so R's records and bands, S's guard / gate ② / OC
  code, T's z / OC code and U's sides and records read V's numbers through the same field names; only what V changes is
  restated. Every seed-bearing field U declared is overridden (judgement 24_600_xxx, smoke 24_609_xxx, gate ②
  25_400_000+i and its smoke 25_409_xxx in the nested P specs).
- V measures with U's measurement file unchanged (V.9.5 P1-3): the lever is the fixed string u_edit(0.0, "chain_entry");
  its APL->MBON05 mask count is 2 (the jobs' edit_edges) and the chain entry is 7 / 2 / 2 (the reference rows'
  block_edges) — 13 edges, CSC 2d359b8b…. The V measurement key is U's, declared as a literal.
- V's set has no declared digests (V.2: the set block fixes them before any oracle on the set); T's set's declared
  values, which USpec inherits, are cleared here so nothing reads them as V's. The generator's permutation seed and T's
  cluster-OC seed stay T's (t_spec.SPEC), as in U.
- Paths: results/v/ (git-ignored), results/summary/v_lever.json (tracked), ~/flymon-archive/v."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .u_measure import u_edit
from .u_spec import SPEC as U_SPEC
from .u_spec import USpec, p_lever

LEVER_V = u_edit(0.0, "chain_entry")                # "u_apl_mbon05_x0.0+chain_entry" (= U's entry_f0)
_V_O = dict(o2_seed0=25_400_000, smoke_seed0=25_409_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_V_O))
P_L = p_lever(P_C, LEVER_V)


@dataclass(frozen=True)
class VSpec(USpec):
    # ---- the lever (V.1, V.9.5) ---------------------------------------------------------------------------------------
    lever_edit: str = LEVER_V
    f: object = 0.0                                  # U's f of the APL->MBON05 part (removed)
    sha_combined: str = "2d359b8b6947d542348b658db977e1edbe0db5919e7f4b1187598cb10ef1253a"
    n_lever_edges_total: int = 13                    # 2 APL->MBON05 + 7 / 2 / 2 chain entry
    mbon05_apl_edges: int = 2                        # V.1: untouched, count and weights
    mbon05_apl_weights: tuple = (-7.865, -12.929)    # as V.0 prints them (float32 −7.8651862, −12.928195)
    mbon05_apl_weight_tol: float = 0.001             # V.0's printed precision; the rule itself is "bytes unchanged"
    # ---- the reuse (V.3 1, V.9.5 P3-12) -------------------------------------------------------------------------------
    u_measure_key_u: str = "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    u_summary: str = U_SPEC.summary
    u_commits: tuple = (("reuse", "d00fdf2"), ("path", "3b83824"), ("scan", "d732b20"), ("kc", "c0d09a7"))
    u_path_detail: str = U_SPEC.path_detail
    u_scan_detail: str = U_SPEC.scan_detail
    u_path_detail_sha256: str = "57ae0c4ed11396421e21a91ae0bd5f88a20c2e8215e612a1225d6599b94be329"
    u_scan_detail_sha256: str = "a4cddfd7fbdec1dd8a95e93088aeecff2c1e9eca389454bcd8d81d4f45c0ab7e"
    u_entry_point: str = "entry_f0"                  # U's scan row of the same edit (V.3 2's base for L_V)
    u_kc_points: tuple = ("0.6", "0.7", "0.8")       # every U KC point holds the same per_odour_none
    # ---- the set (V.2, V.9.1, V.9.5 P2-10 / P3-12) --------------------------------------------------------------------
    first_turn: int = 0
    n_a: int = 43
    kc_reasons_last: str = "kc_input"
    digest_e0_b: str = ""                            # T's declared set values are not V's: V's are in block set
    digest_e0_a: str = ""
    digest_keys: str = ""
    last_turn: object = None
    last_turn_b: object = None
    skipped_declared: tuple = ()
    n_set_odours: object = None
    clusters_b: tuple = ()
    clusters_a: tuple = ()
    # ---- the even gate (V.3 7, V.9.3) ---------------------------------------------------------------------------------
    even_drop_b_lt: int = 3
    even_drop_a_le: int = 1
    # ---- the judgement seeds (V.3 11) and V's smoke (V.3 8) -----------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_600_000, 24_600_008))
    judge_select_seeds: tuple = tuple(range(24_600_100, 24_600_108))
    judge_report_seeds: tuple = tuple(range(24_600_200, 24_600_208))
    smoke_seeds: tuple = tuple(range(24_609_000, 24_609_100))
    # ---- gate ② (V.3 10, V.9.2) ---------------------------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    # ---- the cluster OC (V.6, V.9.5 P2-9): T's method on V's clusters, a runtime output -------------------------------
    oc_cluster_out: str = "results/v/oc_cluster.json"
    oc_cluster_sha256: str = ""                      # V's table is computed after block set; no declared sha
    # ---- paths ----------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/v/cache"
    smoke_cache_dir: str = "results/v/smoke/cache"
    smoke_detail: str = "results/v/smoke.json"
    path_detail: str = "results/v/path.json"
    kc_input_detail: str = "results/v/kc_input.json"
    summary: str = "results/summary/v_lever.json"
    archive_root: str = "~/flymon-archive/v"

    def u_kc_none(self, u_doc: dict) -> dict:
        """U's per_odour_none (the unedited engine's 55 T-set odours, U block kc), equal at every U KC point."""
        pts = [u_doc["kc"]["points"][k]["record_set"]["per_odour_none"] for k in self.u_kc_points]
        if any(p != pts[0] for p in pts):
            raise ValueError("U's KC points disagree on per_odour_none")
        return dict(pts[0])


SPEC = VSpec()


def smoke(spec: VSpec = SPEC) -> VSpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_409_100+), V's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
```

- [ ] **Step 5: Write `v_pairs`**

```python
"""V's judgement set (V.2 as amended by V.9.1 / V.9.5), list only — nothing here runs the engine.
- The generator is T's (T.2 / T.9.1, t_pairs, T's own spec: the widened Gen-1 pool, the 1,986 combos permuted once by
  T's set_rng_seed, turn j's HP, my moves, the alternate, e_pairs._rows_for_turn's rows), walked from turn 0.
- used = T's used rows (e_pairs.used_situations, L turns 0-209 — R's and S's sets too) + T's set (64 rows), whose keys
  and glomerulus keys join used. T's set is reproduced first (t_pairs.t_set / check_t_set, which reproduces S's and R's
  first); a mismatch refuses (ValueError).
- Skips in the declared order (V.9.5 P2-10 / P3-12): ORN cap → glomerulus collision → E1 clash (a side's glomerulus set
  equals a different reachable odour's) → used key → glomerulus-level duplicate → key already in the set → both opponent
  type sets in POOL → kc_input (last): a row is skipped when any of its two E-grid odours has a per-odour KC median
  outside valid_band on either engine (the values of block kc_input only). Then (b) rows until n_b, (a) until n_a.
- candidate_odours (V.3 3, V.9.5 P2-10): the E-grid odours of every row of the whole permutation that passes the skips
  independent of the filling (cap, collision, E1, used, glomerulus duplicate, POOL-only), with each odour's ORN-cap
  arithmetic (max_rate_hz × s × max v against cap_hz).
- set_summary: what block set fixes before any oracle on the set (digests, last turns, odour count, skip counts, the
  cluster table); later stages regenerate the set from block kc_input's values and refuse unless it equals block set.
- judgement_rows / set_odours / set_e0_odours: the checked set's rows with E-grid odours, its distinct E-grid odours
  (the KC band's re-check) and its distinct E0 odours (a record, V.9.5 P2-11); refused for a smoke spec.
- kc_record: the filter's record (V.6): dropped odours per engine, candidate rows using them, and R's, S's and T's set
  rows using them."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from poke_env.data.normalize import to_id_str

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import glomeruli, odour, reachable
from ..battle.pool import POOL
from . import h4_pairs, l_pairs, odor_real, q_pairs, r_pairs, s_pairs, t_pairs
from .r_pairs import okey, row_key
from .r_spec import SPEC as R_SPEC
from .s_spec import SPEC as S_SPEC
from .t_spec import SPEC as T_SPEC

STOP_SET_SHORT = "STOP_SET_SHORT"
OK = "OK"
REASONS = ("cap", "collision", "e1", "used", "glom_dup", "in_set", "pool_only", "kc_input")
ENGINES = ("none", "lever")
SUMMARY_KEYS = ("status", "n_b", "n_a", "last_turn", "last_turn_b", "last_turn_a", "skipped", "n_opp", "n_combos",
                "n_odours", "e1_clashes", "cap_fails", "all_off_pool", "clusters_b", "clusters_a", "digest_e0_b",
                "digest_e0_a", "digest_keys")


def sides(r) -> tuple:
    return (r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"]))


class _Gen:
    """T's generator state (t_pairs.t_set's construction on T's own spec) plus T's set as used."""

    def __init__(self, pops, enc: dict, params):
        js_t = t_pairs.t_set(pops, enc, T_SPEC, params)
        bad = t_pairs.check_t_set(js_t, T_SPEC)
        if bad:
            raise ValueError("T's judgement set does not reproduce: " + "; ".join(bad))
        self.js_t = js_t
        st, mi, mon, move = h4_pairs.pool_vocabulary()
        self.st, self.mi = st, mi
        self.chan = h4_pairs.e0_channels(pops, mon, move)
        self.pops = pops
        self.rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
        self.cb, self.rule = q_pairs.codebook(enc, T_SPEC), E.dual_rule(T_SPEC.config)
        self.max_rate, self.cap = float(params.max_rate_hz), float(odor_real.cap_hz(params))
        opp = t_pairs.opponents(mon, T_SPEC)
        self.species = dict(st)
        self.species.update({n: ts for n, ts in opp})
        self.names = [n for n, _ in opp]
        self.pool_sets = {tuple(sorted(v)) for v in st.values()}
        ex = l_pairs.excluded_combos()
        self.combos = [(m, n) for m in POOL for n in self.names
                       if to_id_str(n) != to_id_str(m.species) and (m.species, n) not in ex]
        used_rows = list(e_pairs.used_situations(pops, E))
        for t in l_pairs.new_turns(st, mi, E.judge_last_turn + 1, E.l_rng_seed):
            alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
            used_rows += e_pairs._rows_for_turn(pops, self.chan, t, alt, st, with_a=True)
        used_rows += js_t["b"] + js_t["a"]
        self.used = {e_pairs.egrid_key(r) for r in used_rows}
        self.used_g = {t_pairs._gkey(self.cb, r) for r in used_rows}
        self.reach = {frozenset(glomeruli(self.cb, m, o)): (m, o) for m, o in reachable(st, move)}
        self.order = np.random.default_rng(T_SPEC.set_rng_seed).permutation(len(self.combos))

    def rows(self, j: int) -> list:
        me, on = self.combos[int(self.order[j])]
        st, mi, names, species = self.st, self.mi, self.names, self.species
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
        return e_pairs._rows_for_turn(self.pops, self.chan, t, alt, species, with_a=True)

    def cap_fails(self, m, o) -> bool:
        return t_pairs._cap_fails(self.rc, self.cb, m, o, self.rule, T_SPEC.strength, self.max_rate, self.cap)

    def e1(self, m, o) -> bool:
        g = t_pairs._glom(self.cb, m, o)
        return g is not None and self.reach.get(g, (m, o)) != (m, o)

    def fixed_reason(self, r) -> str | None:
        """The first skip independent of the filling (cap, collision, e1, used, glom_dup), else None."""
        sd = sides(r)
        if any(self.cap_fails(m, o) for m, o in sd):
            return "cap"
        if any(t_pairs._glom(self.cb, m, o) is None for m, o in sd):
            return "collision"
        if any(self.e1(m, o) for m, o in sd):
            return "e1"
        if e_pairs.egrid_key(r) in self.used:
            return "used"
        if t_pairs._gkey(self.cb, r) in self.used_g:
            return "glom_dup"
        return None

    def pool_only(self, r) -> bool:
        return tuple(r["opp_x"]) in self.pool_sets and tuple(r["opp_y"]) in self.pool_sets

    def odour(self, m, o) -> dict:
        return odour(self.rc, self.cb, m, o, self.rule)


def candidate_odours(pops, enc: dict, params) -> dict:
    """V.3 3 / V.9.5 P2-10: {odours: {id: E-grid odour}, cap: {id: max Hz}, cap_hz, n_rows, turns}."""
    g = _Gen(pops, enc, params)
    ids, n_rows = set(), 0
    for j in range(len(g.order)):
        for r in g.rows(j):
            if g.fixed_reason(r) is None and not g.pool_only(r):
                n_rows += 1
                ids.update(sides(r))
    od = {okey(m, o): g.odour(m, o) for m, o in sorted(ids)}
    return dict(odours=od, cap={k: g.max_rate * float(T_SPEC.strength) * max(v.values()) for k, v in od.items()},
                cap_hz=g.cap, n_rows=n_rows, n_turns=len(g.order))


def _kc_out(kc: dict, oid: str, band) -> bool:
    lo, hi = band
    vals = []
    for e in ENGINES:
        if oid not in kc[e]:
            raise ValueError(f"odour {oid} has no {e} value in block kc_input")
        vals.append(kc[e][oid])
    return any(not lo <= v <= hi for v in vals)


def v_set(pops, enc: dict, params, kc: dict, spec) -> dict:
    """The set and its records (module docstring). kc = {"none": {id: median}, "lever": {id: median}} from block
    kc_input. ValueError when T's set does not reproduce or a candidate odour has no value."""
    g = _Gen(pops, enc, params)
    b, a, taken, skipped, last = [], [], set(), Counter({k: 0 for k in REASONS}), None
    for j in range(spec.first_turn, len(g.order)):
        for r in g.rows(j):
            k = e_pairs.egrid_key(r)
            why = g.fixed_reason(r)
            if why is None and k in taken:
                why = "in_set"
            if why is None and g.pool_only(r):
                why = "pool_only"
            if why is None and any(_kc_out(kc, okey(m, o), spec.valid_band) for m, o in sides(r)):
                why = "kc_input"
            if why is not None:
                skipped[why] += 1
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
    odours = sorted({s for r in b + a for s in sides(r)})
    clusters = Counter(t_pairs.cluster_of(r, g.pool_sets) for r in b + a)
    return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=last,
                last_turn_b=max((r["turn"] for r in b), default=None),
                last_turn_a=max((r["turn"] for r in a), default=None),
                status=OK if (len(b), len(a)) == (spec.n_b, spec.n_a) else STOP_SET_SHORT, skipped=dict(skipped),
                n_opp=len(g.names), n_combos=len(g.combos), n_odours=len(odours),
                e1_clashes=[okey(m, o) for m, o in odours if g.e1(m, o)],
                cap_fails=[okey(m, o) for m, o in odours if g.cap_fails(m, o)],
                all_off_pool=all(not g.pool_only(r) for r in b + a),
                clusters_b=sorted([*c, n] for c, n in clusters.items() if len(c) == 2),
                clusters_a=sorted([*c, n] for c, n in clusters.items() if len(c) == 1),
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
                odour_ids=[okey(m, o) for m, o in odours])


def set_summary(js: dict) -> dict:
    return {k: js[k] for k in SUMMARY_KEYS}


def check_v_set(js: dict, block: dict) -> list:
    """[] when the regenerated set equals block set's record (every SUMMARY_KEYS value)."""
    return [f"V set {k}: regenerated {js[k]!r}, block set {block.get(k)!r}" for k in SUMMARY_KEYS
            if json.loads(json.dumps(js[k])) != block.get(k)]


def _checked(pops, enc: dict, params, kc: dict, block: dict, spec) -> dict:
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (V.3 8)")
    js = v_set(pops, enc, params, kc, spec)
    bad = check_v_set(js, block)
    if js["status"] != OK or bad:
        raise ValueError("; ".join(bad) or f"V set status {js['status']}")
    return js


def judgement_rows(pops, rc: dict, enc: dict, params, kc: dict, block: dict, spec) -> list:
    js = _checked(pops, enc, params, kc, block, spec)
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))


def set_odours(pops, rc: dict, enc: dict, params, kc: dict, block: dict, spec) -> dict:
    """{odour id: E-grid odour} of the checked set (V.3 6's re-check)."""
    js = _checked(pops, enc, params, kc, block, spec)
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    return {okey(m, o): odour(rc, cb, m, o, rule) for m, o in sorted({s for r in js["b"] + js["a"] for s in sides(r)})}


def set_e0_odours(pops, enc: dict, params, kc: dict, block: dict, spec) -> dict:
    """{"<pair key>|x" / "|y": E0 odour} of the checked set (V.9.5 P2-11's record), distinct by content."""
    js = _checked(pops, enc, params, kc, block, spec)
    out, seen = {}, set()
    for r in js["b"] + js["a"]:
        for s, o in (("x", r["odor_x_e0"]), ("y", r["odor_y_e0"])):
            c = json.dumps(o, sort_keys=True)
            if c not in seen:
                seen.add(c)
                out[f"{row_key(r)}|{s}"] = o
    return out


def kc_record(pops, enc: dict, params, kc: dict, spec) -> dict:
    """V.6's KC input record: odours outside valid_band per engine, candidate rows using a dropped odour, and R's, S's
    and T's judgement-set rows using one (records only)."""
    lo, hi = spec.valid_band
    out = {e: sorted(o for o, v in kc[e].items() if not lo <= v <= hi) for e in ENGINES}
    dropped = set(out["none"]) | set(out["lever"])
    g = _Gen(pops, enc, params)
    n_cand = 0
    for j in range(len(g.order)):
        for r in g.rows(j):
            if g.fixed_reason(r) is None and not g.pool_only(r) and any(okey(m, o) in dropped for m, o in sides(r)):
                n_cand += 1
    js_r = e_pairs.judgement_set(pops, E)
    bad = r_pairs.check_set(js_r, enc, R_SPEC)
    if bad:
        raise ValueError("R's judgement set does not reproduce: " + "; ".join(bad))
    js_s = s_pairs.s_set(pops, enc, S_SPEC)
    sets = dict(R=js_r["b"] + js_r["a"], S=js_s["b"] + js_s["a"], T=g.js_t["b"] + g.js_t["a"])

    def using(rows):
        return sum(any(okey(m, o) in dropped for m, o in sides(r)) for r in rows)
    return dict(outside=out, dropped=sorted(dropped), candidate_rows_dropped=n_cand,
                set_rows={k: dict(n=len(v), using_dropped=using(v)) for k, v in sets.items()})


def cluster_labels(rows: list) -> dict:
    """{pair key: cluster label} (T.9.1's labels: "X 대 Y" on (b), the type set on (a))."""
    return t_pairs.clusters(rows)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/brain/test_v_spec.py tests/brain/test_v_pairs.py tests/brain/test_p_spec.py tests/brain/test_u_spec.py tests/brain/test_t_spec.py tests/agent/test_e_spec_store.py -q`
Expected: all pass (the real-data tests need `data/malecns.npz`; ~1 min).

- [ ] **Step 7: Commit**

```bash
git add flymon/brain/v_spec.py flymon/brain/v_pairs.py tests/brain/test_v_spec.py tests/brain/test_v_pairs.py tests/brain/test_p_spec.py
git commit -m "feat(v): v_spec (combined lever u_edit(0.0, chain_entry), seeds 24_600/24_609/25_400/25_409, U key literal) and v_pairs (T's generator from turn 0, T's set as used, kc_input skip last on both engines, candidate odours)"
```

---

### Task 2: `v_store` and `v_records` — V's writer and cache, the combined edit's static facts, full-row and KC-input comparisons, the two-scale ratio

**Files:**
- Create: `flymon/brain/v_store.py`, `flymon/brain/v_records.py`
- Test: `tests/brain/test_v_store.py`, `tests/brain/test_v_records.py`

**Interfaces:**
- Consumes: `v_spec.SPEC` / `smoke`; `r_store` (`RCache`, `archive_copy`, `load_manifest`), `t_store.RReadCache`, `h3_store` (`canonical`, `canonical_pretty`), `pool_bench` guards; `engine_cpu.Engine`, `h3_jobs.edge_sources`, `u_measure.apply_u_edit`, `e_rules.odour_activity`, `s_records.vectors`, `o_rules.boot_weights`, `n_rules._ci`.
- Produces: `v_store.ALLOWED_DIR = "results/v/"`, `SUMMARY`, `SMOKE_SEEDS`, `guard`, `write_bytes`, `write_json(path, obj, params_list) -> Path`, `read_summary(path) -> dict`, `write_summary_block(path, block, obj, params_list)`, `VCache(root, code)`; `v_records.csc_facts(conn, pops, params, spec) -> dict(sha_none, csc_sha256, apl_edges, block_edges, changed, changed_pairs, mbon05_apl)`, `csc_reasons(facts, spec) -> list[str]`, `full_row_diffs(v_rows, u_rows, label) -> list[str]`, `kc_input_record(act, cap, cap_hz, spec) -> dict(none, lever, cap, band)`, `kc_u_diffs(per_none, u_none) -> dict(n_shared, shared, differ, diffs)`, `side_record(side_v, side_none, z_h4, readout) -> dict(p_mean_ratio, sd_ratio)`, `ratio_two_z(rows_l, rows_c, z_l, z_c, spec) -> {direction: dict(ell_L, ell_C, ratio, ratio_ci, n, seeds)}`.

- [ ] **Step 1: Write the failing tests**

```python
"""V's writer and caches (V.5, V.8): writes only under results/v/ and results/summary/v_lever.json, atomically — every
other track's path (results/u/, results/t/, results/r/, results/s/, results/p/, their summaries) refuses; VCache is
RCache with V's writer and V's smoke seeds; RReadCache is T's (R's root, read only); r_store's load_manifest and
archive_copy are reused."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_store, t_store
from flymon.brain import v_store as V
from flymon.brain.v_spec import SPEC, smoke


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/u/x.json", "results/u/cache/r_act/a.json", "results/t/z.json", "results/r/cache/a.json",
                "results/s/x.json", "results/p/x.json", "results/summary/u_lever.json", "results/summary/t_lever.json",
                "results/summary/r_lever.json", "results/vx/a.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            V.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(V.write_json("results/v/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    V.write_summary_block(V.SUMMARY, "reuse", {"n": 1}, [])
    V.write_summary_block(V.SUMMARY, "path", {"m": 2}, [])
    assert V.read_summary() == {"reuse": {"n": 1}, "path": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert V.SUMMARY == SPEC.summary and V.ALLOWED_DIR == "results/v/"


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/v").mkdir(parents=True)
    outside.mkdir()
    (root / "results/v/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        V.write_json("results/v/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_vcache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = V.VCache(SPEC.cache_dir, {"key": "u"}), V.VCache(SPEC.smoke_cache_dir, {"key": "u"})
    assert isinstance(real, r_store.RCache) and {k for k in vars(V.VCache) if not k.startswith("__")} == {"put"}
    assert V.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_600_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_600_000]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_409_100})                   # V's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_600_000]})           # a real seed under the smoke root
    bad = V.VCache("results/u/cache", {"key": "u"})
    with pytest.raises(SystemExit):
        bad.put("r_act", {"edit": "x"}, {"v": 1}, [])
    assert V.RReadCache is t_store.RReadCache and V.load_manifest is r_store.load_manifest
    assert V.archive_copy is r_store.archive_copy
```

```python
"""V's records (V.1, V.3 2-3, V.6, V.9.2, V.9.5): the combined edit's static facts on the real connectome (13 edges
changed by bits — APL->MBON05 2 and the chain entry 7 / 2 / 2 — CSC 2d359b8b…, MBON05->APL untouched) and its
INVALID reasons; a mutation whose chain entry is not applied fails them; full-row comparison with U's rows; the KC input
record (per-odour medians, seed ranges, ORN cap) and the bit-for-bit comparison with U's per_odour_none; the side
record; the two-scale gate-② ratio (L on z_L, C on z_C)."""
from pathlib import Path

import pytest

from flymon.brain import s_records, v_records
from flymon.brain.v_spec import SPEC
from tests.brain.s_fixtures import p_rows
from tests.brain.u_fixtures import ref_rows, rest_rows

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


@pytest.fixture(scope="module")
def real():
    if not NPZ.exists():
        pytest.skip("no connectome")
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    conn = Connectome.load(str(NPZ))
    return dict(conn=conn, pops=Populations.from_connectome(conn),
                params=load_c3_config(str(ROOT / SPEC.m0d_summary)).params)


def test_combined_edit_facts_on_the_real_connectome(real):
    f = v_records.csc_facts(real["conn"], real["pops"], real["params"], SPEC)
    assert f["sha_none"] == SPEC.sha_none and f["csc_sha256"] == SPEC.sha_combined
    assert (f["apl_edges"], f["changed"]) == (2, 13)
    assert f["block_edges"] == {"MBON05->MBON01": 2, "MBON05->MBON09": 7, "MBON05->MBON11": 2}
    assert f["changed_pairs"] == {"APL->MBON05": 2, "MBON05->MBON01": 2, "MBON05->MBON09": 7, "MBON05->MBON11": 2}
    assert f["mbon05_apl"]["n"] == 2 and f["mbon05_apl"]["same_bits"]
    assert sorted(f["mbon05_apl"]["after"]) == pytest.approx([-12.928195, -7.8651862], abs=1e-6)
    assert v_records.csc_reasons(f, SPEC) == []


def test_mutation_an_edit_without_the_chain_entry_fails(real, monkeypatch):
    from flymon.brain import u_measure
    from flymon.brain.u_measure import u_edit
    real_apply = u_measure.apply_u_edit
    monkeypatch.setattr(u_measure, "apply_u_edit",
                        lambda eng, pops, edit, p: real_apply(eng, pops, u_edit(0.0), p))
    f = v_records.csc_facts(real["conn"], real["pops"], real["params"], SPEC)
    bad = v_records.csc_reasons(f, SPEC)
    assert f["changed"] == 2 and any("chain entry" in b for b in bad) and any("CSC" in b for b in bad)


def test_csc_reasons_name_each_break():
    ok = dict(sha_none=SPEC.sha_none, csc_sha256=SPEC.sha_combined, apl_edges=2,
              block_edges={"MBON05->MBON01": 2, "MBON05->MBON09": 7, "MBON05->MBON11": 2}, changed=13,
              mbon05_apl=dict(n=2, before=[-7.8651862, -12.928195], after=[-7.8651862, -12.928195], same_bits=True))
    assert v_records.csc_reasons(ok, SPEC) == []
    for k, v in (("apl_edges", 1), ("changed", 14), ("csc_sha256", "x"), ("sha_none", "y"),
                 ("mbon05_apl", dict(ok["mbon05_apl"], same_bits=False)),
                 ("mbon05_apl", dict(ok["mbon05_apl"], after=[-7.8651862, -12.9302])),
                 ("block_edges", {"MBON05->MBON09": 7})):
        assert len(v_records.csc_reasons(dict(ok, **{k: v}), SPEC)) == 1, k


def test_full_row_diffs_compare_every_field():
    a = ref_rows([1, 2], [3, 4], 2, "sha-V", blocks={"MBON05->MBON09": 7})
    assert v_records.full_row_diffs(a, [dict(r) for r in a], "기준 집합") == []
    b = [dict(r) for r in a]
    b[1] = dict(b[1], mech=dict(b[1]["mech"], apl_out_per_step=0.31))
    assert v_records.full_row_diffs(a, b, "기준 집합") == ["기준 집합: 1 row(s) differ: ['R00/1001']"]
    assert v_records.full_row_diffs(a, b[:1], "x") == ["x: 2 rows, U 1"]
    r = rest_rows(2, 0, 2, "sha-V")
    assert v_records.full_row_diffs(r, [dict(x) for x in r], "휴지") == []


def _act(vals: dict, edges: int, sha: str) -> dict:
    return {o: dict(frac=list(v), max_win=[3] * len(v), edit_edges=[edges], csc_sha256=[sha]) for o, v in vals.items()}


def test_kc_input_record_and_the_u_comparison():
    act = {"none": _act({"A|B": [0.04, 0.05, 0.06], "C|D": [0.02, 0.02, 0.03]}, 0, "sha-C"),
           "lever": _act({"A|B": [0.05, 0.05, 0.05], "C|D": [0.16, 0.2, 0.2]}, 2, "sha-V")}
    rec = v_records.kc_input_record(act, {"A|B": 300.0, "C|D": 200.0}, 1000 / 3, SPEC)
    assert rec["none"]["per_odour"] == {"A|B": 0.05, "C|D": 0.02} and rec["none"]["outside"] == ["C|D"]
    assert rec["lever"]["outside"] == ["C|D"] and rec["none"]["seed_range"]["A|B"] == [0.04, 0.06]
    assert rec["none"]["n_seeds"] == [3] and rec["lever"]["edit_edges"] == [2] and rec["cap"]["all_under"]
    assert rec["band"] == [0.03, 0.15]
    cmp = v_records.kc_u_diffs(rec["none"]["per_odour"], {"A|B": 0.05, "X|Y": 0.1})
    assert (cmp["n_shared"], cmp["differ"], cmp["diffs"]) == (1, [], [])
    cmp = v_records.kc_u_diffs(rec["none"]["per_odour"], {"A|B": 0.0500001})
    assert cmp["differ"] == ["A|B"] and cmp["diffs"] == ["A|B 0.05 ≠ U 0.0500001"]


def test_side_record_ratios():
    s_none = dict(mech=dict(types={"MBON05": dict(mean_stim=26.0)}), z={"A": [10.0, 9.0], "P": [26.0, 19.0]})
    s_v = dict(mech=dict(types={"MBON05": dict(mean_stim=80.0)}), z={"A": [17.0, 12.0], "P": [80.0, 29.0]})
    rec = v_records.side_record(s_v, s_none, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, {"A": "MBON13", "P": "MBON05"})
    assert rec == dict(p_mean_ratio=80.0 / 26.0, sd_ratio={"A": 12.0 / 9.0, "P": 29.0 / 19.0})


def test_ratio_two_z_equals_ss_ratio_on_one_z_and_scales_with_sigma_a():
    rows_l = p_rows(SPEC.p, SPEC.lever_edit, 12, "sha-V", 2)
    rows_c = p_rows(SPEC.p_c, "none", 16)
    z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}
    one = s_records.ratio(rows_l, rows_c, z, SPEC)
    two = v_records.ratio_two_z(rows_l, rows_c, z, z, SPEC)
    assert {d: v["ratio"] for d, v in one.items()} == {d: v["ratio"] for d, v in two.items()}
    zv = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
    two_v = v_records.ratio_two_z(rows_l, rows_c, zv, z, SPEC)
    for d in two_v:
        assert two_v[d]["ratio"] == pytest.approx(3 * one[d]["ratio"]) and two_v[d]["ell_C"] == one[d]["ell_C"]
    with pytest.raises(ValueError):
        v_records.ratio_two_z(rows_l[:-1], rows_c, zv, z, SPEC)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_v_store.py tests/brain/test_v_records.py -q`
Expected: FAIL at collection (`ImportError: cannot import name 'v_store'`).

- [ ] **Step 3: Write `v_store`**

```python
"""V's only writer (V.5, V.8): raw files under results/v/, the one summary results/summary/v_lever.json, atomic writes
(temporary file + rename). u_store.py hardcodes U's paths; V reuses the path-free parts of r_store / t_store:
- guard / write_bytes / write_json / read_summary / write_summary_block: u_store's code with V's allowed paths.
- VCache: r_store.RCache (its key, smoke scope and get) keyed by the V measurement key (= U's, V.9.5 P1-3), with `put`
  routed through this module's writer and V's smoke seeds as the default scope — a V entry lands only under results/v/.
- RReadCache: T's (R's root, read only — `put` refuses). The path and even stages read R's even raw through it.
- load_manifest / archive_copy: r_store's, unchanged (they take their paths and root as arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for v_runner)
from .t_store import RReadCache  # noqa: F401  (re-exported for v_runner)
from .v_spec import SPEC as V_SPEC
from .v_spec import smoke

ALLOWED_DIR = "results/v/"
SUMMARY = "results/summary/v_lever.json"
SMOKE_SEEDS = frozenset(V_SPEC.smoke_seeds) | frozenset(smoke(V_SPEC).p.seeds) | frozenset(smoke(V_SPEC).p_c.seeds)


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"V writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


class VCache(RCache):
    """RCache with V's smoke seeds as the default scope and V's writer (a root outside results/v/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)
```

- [ ] **Step 4: Write `v_records`**

```python
"""V's numbers that no R / S / T / U module returns — records and the values V's gates read.
- csc_facts (V.1, V.3 2, V.9.5 P1-3): the combined edit's static facts on a fresh engine (no simulation): u_measure's
  apply_u_edit with L_V on C3's CSC — its CSC sha, the APL->MBON05 mask count, the chain-entry pair counts, the edges
  whose bits changed (by source / target type) and the MBON05->APL edges before and after (count, bytes, weights).
- full_row_diffs (V.3 2): V's combined reference / rest rows against U's entry_f0 rows, every field bit for bit (the
  same job on the same edit: odour, seed, every type count, KC fractions, the mechanism reads, CSC, edge labels).
- kc_input_record (V.3 3, V.9.5 P2-7): per engine the per-odour KC median over the strength seeds (e_rules'
  odour_activity, as R's gate ① and T.9.1), the per-odour seed range, edges, CSC, counts; kc_u_diffs: V's unedited
  medians against U's per_odour_none on the shared odours, bit for bit.
- side_record (V.6, V.9.2): MBON05 mean ratio (L_V / unedited) and σ_V / σ_h4 per readout.
- ratio_two_z (V.9.2): s_records.ratio with L read on z_L and C on z_C (ℓ_L(z_V) / ℓ_C(h4 z)), same seeds, same
  paired bootstrap."""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from ..agent import e_rules
from .h3_store import canonical
from .n_rules import _ci
from .o_rules import boot_weights
from .s_records import vectors


def csc_facts(conn, pops, params, spec) -> dict:
    from .engine_cpu import Engine
    from .h3_jobs import edge_sources
    from .u_measure import apply_u_edit
    eng = Engine(conn, pops, params, seed=0)
    w0 = eng.csc.w.copy()
    src, tgt = edge_sources(eng.csc), eng.csc.tgt.astype(np.int64)
    t = np.asarray(eng.conn.type).astype(str)
    is_apl = np.zeros(eng.N, bool)
    is_apl[np.asarray(pops.apl, np.int64)] = True
    m5a = (t[src] == spec.p_type) & is_apl[tgt]
    sha0 = hashlib.sha256(w0.tobytes()).hexdigest()
    sha, n_apl, blocks = apply_u_edit(eng, pops, spec.lever_edit, spec.p_type)
    changed = np.flatnonzero(eng.csc.w.view(np.uint32) != w0.view(np.uint32))
    pairs = Counter(f"{'APL' if is_apl[src[i]] else t[src[i]]}->{t[tgt[i]]}" for i in changed)
    return dict(sha_none=sha0, csc_sha256=sha, apl_edges=int(n_apl), block_edges=dict(sorted(blocks.items())),
                changed=int(changed.size), changed_pairs=dict(sorted(pairs.items())),
                mbon05_apl=dict(n=int(m5a.sum()), before=[float(x) for x in w0[m5a]],
                                after=[float(x) for x in eng.csc.w[m5a]],
                                same_bits=bool(np.array_equal(eng.csc.w[m5a].view(np.uint32),
                                                              w0[m5a].view(np.uint32)))))


def csc_reasons(facts: dict, spec) -> list:
    """V.1: INVALID unless 2 APL->MBON05 edges, the chain entry 7 / 2 / 2, 13 changed edges, CSC 2d359b8b…, and the
    MBON05->APL edges untouched (2 edges, bytes equal before and after, weights within 0.001 of V.0's −7.865 /
    −12.929)."""
    bad = []
    if facts["apl_edges"] != spec.lever_edges:
        bad.append(f"APL->MBON05 edges {facts['apl_edges']}, declared {spec.lever_edges}")
    if facts["block_edges"] != dict(sorted(spec.contrast_declared()["chain_entry"].items())):
        bad.append(f"chain entry edges {facts['block_edges']}, declared {spec.contrast_declared()['chain_entry']}")
    if facts["changed"] != spec.n_lever_edges_total:
        bad.append(f"{facts['changed']} edges changed, declared {spec.n_lever_edges_total}")
    if facts["csc_sha256"] != spec.sha_combined:
        bad.append(f"CSC {facts['csc_sha256']}, declared {spec.sha_combined}")
    if facts["sha_none"] != spec.sha_none:
        bad.append(f"unedited CSC {facts['sha_none']}, declared {spec.sha_none}")
    m = facts["mbon05_apl"]
    near = len(m["after"]) == len(spec.mbon05_apl_weights) and all(
        abs(x - y) <= spec.mbon05_apl_weight_tol for x, y in zip(sorted(m["after"]), sorted(spec.mbon05_apl_weights)))
    if m["n"] != spec.mbon05_apl_edges or not m["same_bits"] or not near:
        bad.append(f"MBON05->APL {m}, declared {spec.mbon05_apl_edges} edges {spec.mbon05_apl_weights} untouched")
    return bad


def full_row_diffs(v_rows: list, u_rows: list, label: str) -> list:
    if len(v_rows) != len(u_rows):
        return [f"{label}: {len(v_rows)} rows, U {len(u_rows)}"]
    bad = [f"{u.get('odor', 'rest')}/{u['seed']}" for v, u in zip(v_rows, u_rows) if canonical(v) != canonical(u)]
    return [f"{label}: {len(bad)} row(s) differ: {bad[:3]}"] if bad else []


def _seed_range(act: dict) -> dict:
    return {o: [float(min(v["frac"])), float(max(v["frac"]))] for o, v in act.items()}


def kc_input_record(act: dict, cap: dict, cap_hz: float, spec) -> dict:
    """act = {"none": activity, "lever": activity} over the same odours (RMeasurer.activity's values)."""
    lo, hi = spec.valid_band
    out = {}
    for e, a in act.items():
        per = e_rules.odour_activity({o: a[o]["frac"] for o in a})
        out[e] = dict(per_odour=per, seed_range=_seed_range(a), n_odours=len(per),
                      n_seeds=sorted({len(v["frac"]) for v in a.values()}),
                      outside=sorted(o for o, v in per.items() if not lo <= v <= hi),
                      min=min(per.values()) if per else None, max=max(per.values()) if per else None,
                      edit_edges=sorted({int(x) for v in a.values() for x in v["edit_edges"]}),
                      csc_sha256=sorted({s for v in a.values() for s in v["csc_sha256"]}))
    return dict(out, cap=dict(per_odour_hz=dict(sorted(cap.items())), cap_hz=float(cap_hz),
                              all_under=all(v <= cap_hz for v in cap.values())), band=list(spec.valid_band))


def kc_u_diffs(per_none: dict, u_none: dict) -> dict:
    shared = sorted(set(per_none) & set(u_none))
    bad = [o for o in shared if per_none[o] != u_none[o]]
    return dict(n_shared=len(shared), shared=shared, differ=bad,
                diffs=[f"{o} {per_none[o]!r} ≠ U {u_none[o]!r}" for o in bad[:5]])


def side_record(side_v: dict, side_none: dict, z_h4: dict, readout: dict) -> dict:
    p = readout["P"]
    m0 = side_none["mech"]["types"][p]["mean_stim"]
    z = side_v.get("z")
    return dict(p_mean_ratio=(side_v["mech"]["types"][p]["mean_stim"] / m0) if m0 else None,
                sd_ratio=None if z is None else {k: float(z[k][1]) / float(z_h4[k][1]) for k in z_h4})


def ratio_two_z(rows_l: list, rows_c: list, z_l: dict, z_c: dict, spec) -> dict:
    """s_records.ratio with L read on z_l and C on z_c; ValueError when L and C do not hold the same seeds."""
    vl, vc = vectors(rows_l, z_l, spec.p), vectors(rows_c, z_c, spec.p_c)
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_v_store.py tests/brain/test_v_records.py -q`
Expected: all pass (the two real-connectome tests take ~2 s each).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/v_store.py flymon/brain/v_records.py tests/brain/test_v_store.py tests/brain/test_v_records.py
git commit -m "feat(v): v_store (results/v/ guard, VCache) and v_records (combined edit static facts: 13 edges, CSC 2d359b8b, MBON05->APL untouched; full-row and KC-input comparisons; ratio on two z)"
```

---

### Task 3: `v_rules` — reuse, path, KC input, set, z_V, KC band, even, gate ②, sentences, OCs

**Files:**
- Create: `flymon/brain/v_rules.py`, `tests/brain/v_fixtures.py`
- Test: `tests/brain/test_v_rules.py`

**Interfaces:**
- Consumes: `r_rules` (`read_band`, `gate1`, `SENTENCES`), `s_rules` (`g_fail_s`, `gate2`, `oc`, `OC_NOTES`, `SENTENCES`), `t_rules` (`z_lever`, `oc_cluster`, constants, `SENTENCES`), `u_rules` (`reuse`, `even_repro`, `even_validity`, `STOP_EVEN_PUNISH`), `t_spec.SPEC`, `h3_store.canonical`.
- Produces: constants `PASS`, `INVALID`, `NOT_READ`, `SEALED`, `READ`, `INVALID_RUN`, every STOP label and band, `NONE_ENGINE`, `APL_ENGINE`, `V_ENGINE`, `POOL_COND`; `reuse(r_doc, r_git, t_doc, t_git, u_doc, u_git, shared_key, t_key, u_key, spec) -> dict(outcome, reasons[, sentence])`; `path(checks, csc_bad, spec)`; `kc_input(rec, u_cmp, n_odours, spec)`; `set_outcome(js)`; `z_v(side, n, readout, spec)`; `kc_band(rec112, cap_ok, set_rec, recheck, spec)`; `even(record, c_even, spec)`; `gate2(res_l, res_c, ratio_h4, ratio_zv, spec) -> dict(outcome, …, scale)`; `SENTENCES`, `sentence(outcome, fields) -> str`; `OC_NOTES`, `oc(spec) -> dict`, `oc_cluster(clusters_b, clusters_a) -> dict`, `cluster_values(table) -> dict(cl_null, cl_harm1, cl_harm2)`. `v_fixtures`: `U_IDS`, `CANDS`, `r_doc()`, `t_doc()`, `u_doc()`.

- [ ] **Step 1: Write the fixtures and the failing test**

```python
"""Fabricated R, T and U summaries for V's reuse tests (no engine): R's repro and gate ③, T's z block (its unedited side
on block h4's z of this world, A 10 / 9, P 26 / 19), U's reuse / path / scan / kc blocks under U's measurement key ("u"
× 64) with the detail shas "e" × 64 / "f" × 64, the entry_f0 point on CSC "sha-V" and per_odour_none on U_IDS + one
odour V never lists; CANDS are the world's 20 KC candidate odours (5 of them U's)."""
from flymon.brain.v_spec import SPEC

U_IDS = [f"U{i}|T" for i in range(5)]
CANDS = U_IDS + [f"K{i}|T" for i in range(15)]


def r_doc() -> dict:
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L0"}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def t_doc() -> dict:
    g = dict(passes=True, median_delta=10.0, zero_share=0.0)
    return dict(z=dict(outcome="STOP_Z_DEGENERATE", t_measure_key="t" * 64, code_key=SPEC.r_shared_key,
                       detail_sha256="d" * 64, lever=dict(z={"A": [0.5, 1.0], "P": [80.0, 20.0]}),
                       none=dict(z={"A": [10.0, 9.0], "P": [26.0, 19.0]}, csc_sha256=["sha-C"], edit_edges=[0],
                                 guard={"MBON13": g, "MBON05": g})))


def u_doc() -> dict:
    keys = dict(u_measure_key="u" * 64, code_key=SPEC.r_shared_key, t_measure_key="t" * 64)
    pts = {k: dict(record_set=dict(per_odour_none={o: 0.05 for o in U_IDS + ["Z|T"]})) for k in SPEC.u_kc_points}
    return dict(reuse=dict(outcome="PASS", **keys), path=dict(outcome="PASS", detail_sha256="e" * 64, **keys),
                scan=dict(outcome="PASS", detail_sha256="f" * 64, contrast=dict(
                    entry_f0=dict(invalid=[], side=dict(csc_sha256=["sha-V"], z={"A": [16.9, 12.5]})),
                    readings=dict(entry=dict(reading="사슬 지지"))), **keys),
                kc=dict(outcome="STOP_NO_QUALIFIED_F", points=pts, **keys))
```

```python
"""V's decisions (V.3, V.4, V.7, V.9): the bands, G_fail_S and gate ②'s order are R's / S's functions; reuse names
every break of R, T and U; path / kc_input / kc_band reproduction at their boundaries; z_V with the SD in its sentence;
the even gate's ① filter ((b) < 3, (a) ≤ 1) before ② testable_b ≥ 11; gate ②'s second scale; every sentence verbatim
with V.9.3's POOL condition and V.9.4's cluster values; the OCs (independent with V's notes, T's cluster method on V's
clusters)."""
import json

import pytest

from flymon.brain import r_rules, s_rules, t_rules, u_rules, v_rules
from flymon.brain.v_spec import SPEC
from tests.brain.v_fixtures import CANDS, r_doc, t_doc, u_doc


def test_the_bands_guard_and_gate2_are_rs_and_ss_functions():
    assert v_rules.read_band is r_rules.read_band and v_rules.g_fail_s is s_rules.g_fail_s
    assert v_rules.even_repro is u_rules.even_repro and v_rules.even_validity is u_rules.even_validity


@pytest.mark.parametrize("n,c,f_a,g,band", [(14, 11, 3, False, "B_결론없음"), (8, 8, 3, False, "B_Tb"),
                                            (10, 8, 3, False, "B_결론없음"), (12, 11, 3, False, "B_결론없음"),
                                            (14, 8, 3, True, "B_처벌가드"), (14, 8, 1, False, "B_Fa"),
                                            (14, 8, 2, False, "SELECTED"), (13, 11, 2, False, "B_결론없음")])
def test_band_order_on_vs_numbers(n, c, f_a, g, band):
    assert v_rules.read_band(n, c, f_a, g, 21, 43, 5, SPEC)["band"] == band
    assert v_rules.read_band(n, c, f_a, g, 21, 42, 5, SPEC)["band"] == "NOT_READ"


def _ok():
    return dict(tracked=True, dirty=False, judge_commits=[])


def _reuse(u=None, u_git=None, u_key=None, **kw):
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_none="sha-C", sha_combined="sha-V", t_measure_key_t="t" * 64,
                             t_z_detail_sha256="d" * 64, u_measure_key_u="u" * 64, u_path_detail_sha256="e" * 64,
                             u_scan_detail_sha256="f" * 64, z_h4=(("A", (10.0, 9.0)), ("P", (26.0, 19.0))))
    return v_rules.reuse(r_doc(), _ok(), t_doc(), _ok(), u if u is not None else u_doc(), u_git or _ok(),
                         sp.r_shared_key, "t" * 64, u_key or "u" * 64, sp)


def test_reuse_passes_and_names_every_u_break():
    assert _reuse()["outcome"] == "PASS"
    out = _reuse(u_key="v" * 64)
    assert out["outcome"] == "STOP_REUSE" and "V 측정 키" in out["reasons"][0]
    assert "미커밋" in _reuse(u_git=dict(tracked=True, dirty=True))["reasons"][0]
    for mut, want in ((lambda d: d.pop("kc"), "U 블록 kc 없음"),
                      (lambda d: d["path"].__setitem__("u_measure_key", "x"), "U 블록 path의 키"),
                      (lambda d: d["path"].__setitem__("detail_sha256", "x"), "U 블록 path PASS / 원자료 sha x"),
                      (lambda d: d["scan"].__setitem__("detail_sha256", "x"), "U 블록 scan 원자료 sha x"),
                      (lambda d: d["scan"]["contrast"]["entry_f0"].__setitem__("invalid", ["bad"]), "entry_f0 행 무효"),
                      (lambda d: d["kc"]["points"]["0.7"]["record_set"].__setitem__("per_odour_none", {}),
                       "per_odour_none 읽기 실패")):
        d = u_doc()
        mut(d)
        out = _reuse(u=d)
        assert out["outcome"] == "STOP_REUSE" and any(want in x for x in out["reasons"]), (want, out["reasons"])


def test_path_invalid_stop_and_pass():
    ok = [dict(engine="편집 없는 엔진", ref="T 편집 없는 엔진 기준 집합 행", diffs=[], invalid=[])]
    assert v_rules.path(ok, [], SPEC)["outcome"] == "PASS"
    assert v_rules.path(ok, ["13 edges"], SPEC)["outcome"] == "INVALID"
    bad = ok + [dict(engine="조합 엔진", ref="U entry_f0 기준 집합 행", diffs=["기준 집합: 1 row(s) differ"], invalid=[])]
    out = v_rules.path(bad, [], SPEC)
    assert out["outcome"] == "STOP_V_PATH_REPRO" and out["sentence"] == (
        "V 편집 경로가 조합 엔진에서 U entry_f0 기준 집합 행을 재현하지 못했다(기준 집합: 1 row(s) differ).")


def _rec(n=20, edges=(0, 2), sha=("sha-C", "sha-V"), seeds=8, under=True):
    return dict(none=dict(n_odours=n, n_seeds=[seeds], edit_edges=[edges[0]], csc_sha256=[sha[0]]),
                lever=dict(n_odours=n, n_seeds=[seeds], edit_edges=[edges[1]], csc_sha256=[sha[1]]),
                cap=dict(all_under=under))


def test_kc_input_invalid_stop_and_pass():
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_none="sha-C", sha_combined="sha-V")
    cmp_ok = dict(diffs=[])
    assert v_rules.kc_input(_rec(), cmp_ok, 20, sp)["outcome"] == "PASS"
    for r in (_rec(n=19), _rec(edges=(0, 3)), _rec(sha=("sha-C", "sha-L0")), _rec(seeds=7), _rec(under=False)):
        assert v_rules.kc_input(r, cmp_ok, 20, sp)["outcome"] == "INVALID"
    out = v_rules.kc_input(_rec(), dict(diffs=["A|B 0.05 ≠ U 0.0501"]), 20, sp)
    assert out["outcome"] == "STOP_V_PATH_REPRO" and out["sentence"] == (
        "V 편집 경로가 편집 없는 엔진에서 U KC 블록(c0d09a7) per_odour_none을 재현하지 못했다(A|B 0.05 ≠ U 0.0501).")


def test_set_outcome_sentence():
    assert v_rules.set_outcome(dict(status="OK"))["outcome"] == "PASS"
    assert v_rules.set_outcome(dict(status="STOP_SET_SHORT", n_b=21, n_a=40))["sentence"] == (
        "KC 입력으로 거른 넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다(21·40쌍).")


def _side(a_med=6.0, a_zero=0.1, p_med=40.0, z=True, block=None, sha="sha-V"):
    g = {"MBON13": dict(passes=a_med >= 5 and a_zero <= 0.25, median_delta=a_med, zero_share=a_zero),
         "MBON05": dict(passes=True, median_delta=p_med, zero_share=0.0)}
    return dict(guard=g, z={"A": [16.9, 12.48], "P": [80.2, 29.8]} if z else None, why=None if z else "zero SD",
                zero_sd=[] if z else ["MBON13"], n_ref=96, n_rest=96, edit_edges=[2], csc_sha256=[sha],
                block_edges=[block or json.dumps(dict(SPEC.contrast_declared()["chain_entry"]), sort_keys=True,
                                                 separators=(",", ":"))])


def test_z_v_pass_stop_with_sd_and_invalid():
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_combined="sha-V")
    ro = {"A": "MBON13", "P": "MBON05"}
    assert v_rules.z_v(_side(), 96, ro, sp)["outcome"] == "PASS"
    out = v_rules.z_v(_side(a_med=4.0), 96, ro, sp)
    assert out["sentence"] == ("조합 지렛대 아래 기준 집합에서 판독 MBON13이 반응성 가드를 넘지 못했다(Δ 중앙값 4.0, 0 비율 0.100, "
                               "SD 12.480) — z를 정할 수 없다.")
    out = v_rules.z_v(_side(z=False), 96, ro, sp)
    assert out["outcome"] == "STOP_Z_DEGENERATE" and "SD 0)" in out["sentence"]
    assert v_rules.z_v(_side(block='{"MBON05->MBON09":7}'), 96, ro, sp)["outcome"] == "INVALID"
    assert v_rules.z_v(_side(sha="sha-L0"), 96, ro, sp)["outcome"] == "INVALID"
    assert v_rules.z_v(_side(), 95, ro, sp)["outcome"] == "INVALID"


def _g1(median=0.05, edges=2, n=112):
    return dict(median=median, edit_edges=[edges], csc_sha256=["sha-V"], n_odours=n, n_seeds=[8])


def test_kc_band_order():
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_combined="sha-V")
    srec = dict(per_odour={o: 0.05 for o in CANDS[:5]}, edit_edges=[2], csc_sha256=["sha-V"])
    assert v_rules.kc_band(_g1(), True, srec, [], sp)["outcome"] == "PASS"
    assert v_rules.kc_band(_g1(edges=3), True, srec, [], sp)["outcome"] == "INVALID"
    assert v_rules.kc_band(_g1(), True, dict(srec, csc_sha256=["sha-C"]), [], sp)["outcome"] == "INVALID"
    out = v_rules.kc_band(_g1(median=0.2), True, srec, ["1 odour(s) differ: X"], sp)
    assert out["outcome"] == "STOP_V_PATH_REPRO"                         # the re-check comes before the band
    out = v_rules.kc_band(_g1(median=0.2), False, srec, [], sp)
    assert out["outcome"] == "STOP_STRENGTH_LEVER" and out["sentence"] == (
        "조합 지렛대 아래에서 E-grid k2-norm s 1.0이 KC 유효 대역을 잃었다(냄새별 KC 활성 중앙값 0.2000 ∉ [0.03, 0.15]; "
        "ORN 상한(s 1.0)).")
    out = v_rules.kc_band(_g1(), True, dict(srec, per_odour={"A|B": 0.0299}), [], sp)
    assert out["outcome"] == "STOP_STRENGTH_LEVER" and "V 세트 냄새 A|B 0.0299 ∉ [0.03, 0.15]" in out["sentence"]


def _even(db, da, tb, fa=0):
    return dict(drops={"b": dict(net_drop=db), "a": dict(net_drop=da)}, testable_b=tb, F_a=fa)


def test_even_gate_boundaries():
    assert v_rules.even(_even(2, 1, 11), 7, SPEC)["outcome"] == "PASS"
    assert v_rules.even(_even(3, 0, 15), 7, SPEC)["outcome"] == "STOP_EVEN_PUNISH"
    assert v_rules.even(_even(0, 2, 15), 7, SPEC)["outcome"] == "STOP_EVEN_PUNISH"
    out = v_rules.even(_even(3, 2, 5), 7, SPEC)                           # ① before ②
    assert out["sentence"] == ("조합 지렛대가 짝수 쌍에서 처벌 순감소 거름 기준((b) < 3, (a) ≤ 1)을 넘었다((b) 3, (a) 2). "
                               "(POOL 안 짝수 쌍 조건부)")
    out = v_rules.even(_even(-1, -2, 10, 4), 7, SPEC)
    assert out["outcome"] == "STOP_EVEN_LOW_LEVER" and out["sentence"] == (
        "조합 지렛대 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 "
        "4/18). (POOL 안 짝수 쌍 조건부)")


def _res(label="LEARNS_CONFIRMATORY", l1=1.0, l2=1.0):
    return dict(outcome="PASS", label=label, directions={"r1": dict(ell=l1), "r2": dict(ell=l2)})


def _ratio(r1, r2, ell_l=0.5, ell_c=1.0):
    return {d: dict(ratio=v, ell_L=ell_l, ell_C=ell_c) for d, v in (("r1", r1), ("r2", r2))}


def test_gate2_both_scales_in_order():
    assert v_rules.gate2(_res(), _res(), _ratio(0.6, 0.7), _ratio(0.5, 0.9), SPEC)["outcome"] == "PASS"
    out = v_rules.gate2(_res(), _res(), _ratio(0.49, 0.7), _ratio(0.9, 0.9), SPEC)
    assert (out["outcome"], out["scale"]) == ("STOP_PUNISH_WEAKENED", "h4")
    out = v_rules.gate2(_res(), _res(), _ratio(0.6, 0.7), _ratio(0.9, 0.499, 0.4, 0.81), SPEC)
    assert (out["outcome"], out["scale"], out["low"]) == ("STOP_PUNISH_WEAKENED", "z_V", ["r2"])
    assert out["sentence"] == ("APL→MBON05 제거 아래 같은 시드의 처벌 학습량이 지렛대 없는 쪽의 절반에 못 미쳤다(ℓ_r1 0.400 대 "
                               "0.810, ℓ_r2 0.400 대 0.810).")
    assert v_rules.gate2(_res("NOT_LEARNING"), _res(), None, None, SPEC)["outcome"] == "STOP_PUNISH_BROKEN"
    assert v_rules.gate2(_res(), _res("NOT_LEARNING"), None, None, SPEC)["outcome"] == "STOP_P_REFERENCE"
    assert v_rules.gate2(dict(outcome="INVALID", reasons=["x"]), _res(), None, None, SPEC)["outcome"] == "INVALID"


F = dict(n=14, c=8, f_a=3, naive_a=4, T=296, k_even=13, k_cl=11, pb_L=20, pb_C=21, pa_L=40, pa_C=41, d_b=1, d_a=1,
         rho1="0.700", rho2="0.800", cl_null="0.401", cl_harm1="0.685", cl_harm2="0.571")


def test_every_sentence_fills():
    for o in ("B_Tb", "B_처벌가드", "B_Fa", "SELECTED"):
        assert v_rules.sentence(o, dict(F, reason="PASS"))
    for reason in ("C_ABOVE_BAR", "BELOW_BAR", "MARGIN"):
        assert v_rules.sentence("B_결론없음", dict(F, reason=reason)).endswith("(14/21 대 8/21, F_a 3/43, 조합 지렛대)")
    assert v_rules.sentence("B_Tb", F) == (
        "APL→MBON05 2간선 제거와 MBON05→MBON09/MBON11/MBON01 11간선 제거를 함께 한 모델 변형(커넥톰 간선 13개 제거)이 편집 "
        "없는 엔진의 KC 입력 유효성으로 거른 넓힌 상대 풀 판정 세트(생성원 턴 0–296)에서 14/21로 지렛대 없는 같은 세트 8/21보다 "
        "오르지 않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 이 조합 지렛대 주장을 닫는다.")
    assert v_rules.sentence("B_처벌가드", F).endswith(" (조합 지렛대)") and v_rules.sentence("B_Fa", F).endswith(" (조합 지렛대)")
    s = v_rules.sentence("SELECTED", F)
    assert "(b) 21쌍은 11개 상대 타입 조합 쌍에 몰려" in s and "판정 시드 24_600_xxx" in s and "생성원 턴 0–296" in s
    assert ("V 세트 군집 상관 모형에서 G_fail_S null 0.401, 오선택(harm) 0.685 / 0.571. 이 결과는 이 64쌍 조건부 기술 결과다. "
            "작동 특성은 V 세트 조건부 값이며") in s
    assert s.endswith("POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.")
    assert v_rules.sentence("STOP_REUSE", dict(why="x")).startswith("R·T·U 재사용 조건(V.3 1)이 깨졌다(x).")
    assert v_rules.SENTENCES["STOP_EVEN_REPRO"] == t_rules.SENTENCES["STOP_EVEN_REPRO"]
    for o in ("STOP_PUNISH_BROKEN", "STOP_P_REFERENCE", "STOP_PUNISH_WEAKENED"):
        assert v_rules.SENTENCES[o] == s_rules.SENTENCES[o]


def test_oc_notes_and_cluster_values():
    oc = v_rules.oc(SPEC)
    assert oc["notes"] == list(v_rules.OC_NOTES) and len(oc["sha256"]) == 64
    assert "Q → R → S → T → U → V" in oc["notes"][-1]
    table = dict(g_fail=[dict(kind="null", p=0.40069), dict(kind="harm", p=0.68484), dict(kind="harm", p=0.57129)])
    assert v_rules.cluster_values(table) == dict(cl_null="0.401", cl_harm1="0.685", cl_harm2="0.571")


def test_cluster_model_on_ts_clusters_reproduces_ts_fixture():
    """V.6: T's method — on T's own clusters V's oc_cluster reproduces T's committed fixture's g rows and n_dist (only
    the notes and therefore the sha differ)."""
    from pathlib import Path

    from flymon.brain.t_spec import SPEC as T
    root = Path(__file__).resolve().parents[2]
    fx = json.loads((root / T.oc_cluster_fixture).read_text())
    got = v_rules.oc_cluster([list(c) for c in T.clusters_b], [list(c) for c in T.clusters_a])
    assert got["g_fail"] == fx["g_fail"] and got["n_dist"] == fx["n_dist"] and got["rows"] == fx["rows"]
    assert got["notes"] == list(v_rules.OC_NOTES[1:]) and got["seed"] == 20261005 and got["icc"] == 0.3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_v_rules.py -q`
Expected: FAIL at collection (`ImportError: cannot import name 'v_rules'`).

- [ ] **Step 3: Write the implementation**

```python
"""V's decisions (V.3, V.4, V.7 as amended by V.9). The judgement code is the authoritative source: the bands are
r_rules.read_band itself (V.4 = U.4 = T.4 = R.3's order, n_a 43), G_fail_S is S's (net drop ≥ 3 per axis) and gate ②'s
order is S's on block h4's z, with V.9.2's second scale (ℓ_L(z_V) / ℓ_C(h4 z) ≥ 0.5 both directions) after it. New
here: the reuse of R, T and U (V.3 1), the path reproduction (V.3 2, V.9.5), the KC input gate (V.3 3, V.9.1), the set
outcome (V.2), z_V (V.3 5 with SD in the sentence), the combined-engine KC band (V.3 6, V.9.1), the even gates (V.3 7,
V.9.3), the sentences (V.7 with V.9.3 / V.9.4; 〈…〉 → {field}) and the OCs (V.6: the independent model with V's notes
and T's cluster method on V's set). Every number is a field of the VSpec passed in; no OC value changes a band."""
from __future__ import annotations

import dataclasses
import hashlib

from . import r_rules, s_rules, t_rules, u_rules
from .h3_store import canonical
from .t_spec import SPEC as T_SPEC

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (t_rules.PASS, t_rules.INVALID, t_rules.NOT_READ,
                                                       t_rules.SEALED, t_rules.READ, t_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = t_rules.STOP_REUSE, t_rules.STOP_SET_SHORT
STOP_V_PATH_REPRO, STOP_Z_DEGENERATE = "STOP_V_PATH_REPRO", t_rules.STOP_Z_DEGENERATE
STOP_STRENGTH_LEVER = t_rules.STOP_STRENGTH_LEVER
STOP_EVEN_REPRO, STOP_EVEN_PUNISH, STOP_EVEN_LOW_LEVER = (t_rules.STOP_EVEN_REPRO, u_rules.STOP_EVEN_PUNISH,
                                                          t_rules.STOP_EVEN_LOW_LEVER)
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (t_rules.STOP_PUNISH_BROKEN, t_rules.STOP_P_REFERENCE,
                                                              t_rules.STOP_PUNISH_WEAKENED)
SELECTED, B_TB, B_FA, B_NC, B_PG = t_rules.SELECTED, t_rules.B_TB, t_rules.B_FA, t_rules.B_NC, t_rules.B_PG
BANDS = t_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = t_rules.C_ABOVE_BAR, t_rules.BELOW_BAR, t_rules.MARGIN
NONE_ENGINE, APL_ENGINE, V_ENGINE = "편집 없는 엔진", "APL→MBON05 제거 단독 엔진", "조합 엔진"
POOL_COND = " (POOL 안 짝수 쌍 조건부)"               # V.9.3: appended to the two even STOP sentences
read_band = r_rules.read_band                         # V.4: R.3's order
g_fail_s = s_rules.g_fail_s                           # V.4: S's guard, each condition on its own z
even_repro = u_rules.even_repro                       # V.3 7: C (R's even raw, h4 z) reproduces c_even 7 first
even_validity = u_rules.even_validity                 # R's gate ③ validity list for L_V


# ================================================================ gates before the set is used
def reuse(r_doc, r_git, t_doc, t_git, u_doc, u_git, shared_key, t_key, u_key, spec) -> dict:
    """V.3 1: R's repro and T's unedited z as U.3 1 (u_rules.reuse); U's blocks iff the current V measurement key is the
    declared U key 8a4e0930…, U's summary is tracked and clean, and U's reuse / path / scan / kc blocks carry it (with
    R's shared key and T's key), path passed, the path and scan details' shas are the declared ones, the scan's entry_f0
    point is valid on CSC 2d359b8b…, and U's KC points agree on per_odour_none. Else STOP_REUSE."""
    r = u_rules.reuse(r_doc, r_git, t_doc, t_git, shared_key, t_key, spec)
    why = list(r.get("reasons") or [])
    if u_key != spec.u_measure_key_u:
        why.append(f"V 측정 키 {u_key} ≠ U 측정 키 {spec.u_measure_key_u}")
    if not u_git.get("tracked") or u_git.get("dirty"):
        why.append(f"{spec.u_summary} 미커밋")
    for b, _ in spec.u_commits:
        blk = u_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"U 블록 {b} 없음")
            continue
        if (blk.get("u_measure_key"), blk.get("code_key"), blk.get("t_measure_key")) != (
                spec.u_measure_key_u, spec.r_shared_key, spec.t_measure_key_t):
            why.append(f"U 블록 {b}의 키 {blk.get('u_measure_key')} / {blk.get('code_key')} / {blk.get('t_measure_key')}")
    path, scan = u_doc.get("path") or {}, u_doc.get("scan") or {}
    if path.get("outcome") != PASS or path.get("detail_sha256") != spec.u_path_detail_sha256:
        why.append(f"U 블록 path {path.get('outcome')} / 원자료 sha {path.get('detail_sha256')}")
    if scan.get("detail_sha256") != spec.u_scan_detail_sha256:
        why.append(f"U 블록 scan 원자료 sha {scan.get('detail_sha256')}")
    entry = ((scan.get("contrast") or {}).get(spec.u_entry_point) or {})
    if entry.get("invalid") != [] or (entry.get("side") or {}).get("csc_sha256") != [spec.sha_combined]:
        why.append(f"U {spec.u_entry_point} 행 무효 또는 CSC {(entry.get('side') or {}).get('csc_sha256')}")
    try:
        spec.u_kc_none(u_doc)
    except (KeyError, TypeError, ValueError) as e:
        why.append(f"U 블록 kc의 per_odour_none 읽기 실패({e})")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


def path(checks: list, csc_bad: list, spec) -> dict:
    """V.3 2: checks = [dict(engine, ref, diffs, invalid)] in measurement order; csc_bad = v_records.csc_reasons. A
    defect (presentations, edge labels, the static edit facts) is INVALID; any bit difference is STOP_V_PATH_REPRO, its
    sentence naming the first failing check, every failing check listed."""
    bad = list(csc_bad) + [m for c in checks for m in c.get("invalid", [])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    return _repro(checks)


def _repro(checks: list) -> dict:
    failed = [c for c in checks if c["diffs"]]
    if failed:
        c = failed[0]
        return dict(outcome=STOP_V_PATH_REPRO, reasons=[], failed=[dict(engine=x["engine"], ref=x["ref"],
                                                                         diffs=x["diffs"]) for x in failed],
                    sentence=sentence(STOP_V_PATH_REPRO, dict(engine=c["engine"], ref=c["ref"],
                                                              diff="; ".join(c["diffs"]))))
    return dict(outcome=PASS, reasons=[], failed=[])


def kc_input(rec: dict, u_cmp: dict, n_odours: int, spec) -> dict:
    """V.3 3 / V.9.1: INVALID unless both engines hold every candidate odour on all strength seeds, the unedited engine
    0 edges on the unedited CSC and L_V 2 edges on CSC 2d359b8b…; then the unedited medians must equal U's
    per_odour_none on the shared odours bit for bit (STOP_V_PATH_REPRO). The band is a filter here, not a gate."""
    bad = []
    for e, edges, sha in (("none", 0, spec.sha_none), ("lever", spec.lever_edges, spec.sha_combined)):
        r = rec[e]
        if r["n_odours"] != n_odours or r["n_seeds"] != [len(spec.kc_seeds())]:
            bad.append(f"{e}: {r['n_odours']} odours × {r['n_seeds']} seeds, declared {n_odours} × "
                       f"{len(spec.kc_seeds())}")
        if r["edit_edges"] != [edges] or r["csc_sha256"] != [sha]:
            bad.append(f"{e}: edges {r['edit_edges']} on CSC {r['csc_sha256']}, declared {edges} on {sha}")
    if not rec["cap"]["all_under"]:
        bad.append("a candidate odour above the ORN cap")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    return _repro([dict(engine=NONE_ENGINE, ref="U KC 블록(c0d09a7) per_odour_none", diffs=u_cmp["diffs"])])


def set_outcome(js: dict) -> dict:
    if js["status"] == STOP_SET_SHORT:
        return dict(outcome=STOP_SET_SHORT, sentence=sentence(STOP_SET_SHORT, dict(b=js["n_b"], a=js["n_a"])))
    return dict(outcome=PASS)


def z_v(side: dict, n: int, readout: dict, spec) -> dict:
    """V.3 5: T's guard decision (t_rules.z_lever: edges, 96 + 96, median Δ ≥ 5, zero share ≤ 0.25, SD > 0) on the
    combined engine's side from block path, the chain-entry edges as declared (else INVALID), and V's sentence (with
    the SD) on a failure."""
    want = canonical(dict(spec.contrast_declared()["chain_entry"]))
    d = t_rules.z_lever(side, n, readout, spec)
    bad = list(d.get("reasons") or []) if d["outcome"] == INVALID else []
    if side["block_edges"] != [want]:
        bad.append(f"chain entry edges {side['block_edges']}, declared {want}")
    if side["csc_sha256"] != [spec.sha_combined]:
        bad.append(f"CSC {side['csc_sha256']}, declared {spec.sha_combined}")
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    if d["outcome"] != PASS:
        g, z, key = side["guard"], side.get("z"), {t: k for k, t in readout.items()}
        sd = [f"{float(z[key[t]][1]):.3f}" if z else ("0" if t in side["zero_sd"] else "—") for t in d["failed"]]
        return dict(outcome=STOP_Z_DEGENERATE, reasons=d.get("reasons", []), failed=d["failed"],
                    sentence=sentence(STOP_Z_DEGENERATE, dict(
                        type="·".join(d["failed"]), med="·".join(f"{g[t]['median_delta']:.1f}" for t in d["failed"]),
                        zero="·".join(f"{g[t]['zero_share']:.3f}" for t in d["failed"]), sd="·".join(sd))))
    return dict(outcome=PASS, reasons=[])


def kc_band(rec112: dict, cap_ok: bool, set_rec: dict, recheck: list, spec) -> dict:
    """V.3 6 / V.9.1: R's gate ① rule on the 112 calibration odours under L_V (r_rules.gate1's INVALID checks, the
    median of per-odour medians in the band, the ORN cap); the V set's odours re-measured under L_V must equal block
    kc_input's L_V values bit for bit (else STOP_V_PATH_REPRO) and lie in the band; else STOP_STRENGTH_LEVER (V's
    sentence)."""
    g1 = r_rules.gate1(rec112, cap_ok, spec)
    bad = list(g1.get("reasons") or []) if g1["outcome"] == INVALID else []
    if set_rec["edit_edges"] != [spec.lever_edges] or set_rec["csc_sha256"] != [spec.sha_combined]:
        bad.append(f"V set odours: edges {set_rec['edit_edges']} on {set_rec['csc_sha256']}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[], calib=g1)
    rp = _repro([dict(engine=V_ENGINE, ref="KC 입력 블록의 V 세트 냄새 값", diffs=recheck)])
    if rp["outcome"] != PASS:
        return dict(rp, calib=g1)
    lo, hi = spec.valid_band
    failed = list(g1.get("failed", [])) + [f"V 세트 냄새 {o} {v:.4f} ∉ [{lo}, {hi}]"
                                          for o, v in sorted(set_rec["per_odour"].items()) if not lo <= v <= hi]
    if failed:
        return dict(outcome=STOP_STRENGTH_LEVER, reasons=[], failed=failed, calib=g1,
                    sentence=sentence(STOP_STRENGTH_LEVER, dict(cond="; ".join(failed))))
    return dict(outcome=PASS, reasons=[], failed=[], calib=g1)


def even(record: dict, c_even: int, spec) -> dict:
    """V.3 7 / V.9.3 on L_V's even record (u_records.even_record): ① (b) net drop < even_drop_b_lt and (a) net drop ≤
    even_drop_a_le (else STOP_EVEN_PUNISH); ② testable_b ≥ bar_b (else STOP_EVEN_LOW_LEVER); else PASS."""
    d = record["drops"]
    db, da = d["b"]["net_drop"], d["a"]["net_drop"]
    if not (db < spec.even_drop_b_lt and da <= spec.even_drop_a_le):
        return dict(outcome=STOP_EVEN_PUNISH, reasons=[], sentence=sentence(STOP_EVEN_PUNISH, dict(d_b=db, d_a=da)))
    if record["testable_b"] < spec.bar_b:
        return dict(outcome=STOP_EVEN_LOW_LEVER, reasons=[], sentence=sentence(STOP_EVEN_LOW_LEVER, dict(
            tb=record["testable_b"], c_even=c_even, k=record["F_a"])))
    return dict(outcome=PASS, reasons=[])


def gate2(res_l: dict, res_c: dict, ratio_h4: dict | None, ratio_zv: dict | None, spec) -> dict:
    """V.3 10 with V.9.2: S's order on block h4's z (INVALID → STOP_PUNISH_BROKEN → STOP_P_REFERENCE → h4 ratio <
    p_ratio_min → STOP_PUNISH_WEAKENED); then the z_V ratio ℓ_L(z_V) / ℓ_C(h4 z) < p_ratio_min in either direction →
    STOP_PUNISH_WEAKENED (S's sentence filled with the z_V-scale ℓ — plan OPEN 1); else PASS. `scale` names the failing
    ratio."""
    d = s_rules.gate2(res_l, res_c, ratio_h4, spec)
    if d["outcome"] != PASS:
        return dict(d, scale="h4" if d["outcome"] == STOP_PUNISH_WEAKENED else None)
    names = list(spec.p.directions)
    low = [x for x in names if ratio_zv[x]["ratio"] < spec.p_ratio_min]
    if low:
        f = dict(l1L=f"{ratio_zv[names[0]]['ell_L']:.3f}", l1C=f"{ratio_zv[names[0]]['ell_C']:.3f}",
                 l2L=f"{ratio_zv[names[-1]]['ell_L']:.3f}", l2C=f"{ratio_zv[names[-1]]['ell_C']:.3f}")
        return dict(d, outcome=STOP_PUNISH_WEAKENED, low=low, scale="z_V",
                    sentence=sentence(STOP_PUNISH_WEAKENED, f))
    return dict(d, scale=None)


# ================================================================ the closing sentences (V.7, V.9.3, V.9.4)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43, 조합 지렛대)"
_LEVER = ("APL→MBON05 2간선 제거와 MBON05→MBON09/MBON11/MBON01 11간선 제거를 함께 한 모델 변형(커넥톰 간선 13개 제거)")
SENTENCES = {
    STOP_REUSE: "R·T·U 재사용 조건(V.3 1)이 깨졌다({why}). V는 R·T·U 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 "
                "쓰지 않고 멈춘다 — 사용자 몫.",
    STOP_V_PATH_REPRO: "V 편집 경로가 {engine}에서 {ref}을 재현하지 못했다({diff}).",
    STOP_SET_SHORT: "KC 입력으로 거른 넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다({b}·{a}쌍).",
    STOP_Z_DEGENERATE: "조합 지렛대 아래 기준 집합에서 판독 {type}이 반응성 가드를 넘지 못했다(Δ 중앙값 {med}, 0 비율 "
                       "{zero}, SD {sd}) — z를 정할 수 없다.",
    STOP_STRENGTH_LEVER: "조합 지렛대 아래에서 E-grid k2-norm s 1.0이 KC 유효 대역을 잃었다({cond}).",
    STOP_EVEN_REPRO: t_rules.SENTENCES[STOP_EVEN_REPRO],
    STOP_EVEN_PUNISH: "조합 지렛대가 짝수 쌍에서 처벌 순감소 거름 기준((b) < 3, (a) ≤ 1)을 넘었다((b) {d_b}, (a) {d_a})."
                      + POOL_COND,
    STOP_EVEN_LOW_LEVER: r_rules.SENTENCES[STOP_EVEN_LOW_LEVER].replace("APL→MBON05 제거", "조합 지렛대", 1) + POOL_COND,
    STOP_PUNISH_BROKEN: s_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: s_rules.SENTENCES[STOP_P_REFERENCE],
    STOP_PUNISH_WEAKENED: s_rules.SENTENCES[STOP_PUNISH_WEAKENED],
    B_TB: _LEVER + "이 편집 없는 엔진의 KC 입력 유효성으로 거른 넓힌 상대 풀 판정 세트(생성원 턴 0–{T})에서 {n}/21로 지렛대 "
                   "없는 같은 세트 {c}/21보다 오르지 않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 이 조합 "
                   "지렛대 주장을 닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: s_rules.SENTENCES[B_PG] + " (조합 지렛대)",
    B_FA: s_rules.SENTENCES[B_FA] + " (조합 지렛대)",
    SELECTED: "C3에서 " + _LEVER + "과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 때, 편집 없는 "
              "엔진의 KC 입력 유효성으로 거른 판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–{T}, 기존 세트·키·"
              "사구체 중복 제외, 모든 행이 POOL 밖 상대 타입 조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 "
              "만족했고 지렛대 없는 같은 세트보다 높았다({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 "
              "≥ 3) 통과, 같은 시드 P 비 ℓ_r1 {rho1}·ℓ_r2 {rho2} ≥ 0.5(약화 정도 기록), 짝수 {k_even}/21, 판정 시드 "
              "24_600_xxx). 판정 세트의 상대는 1세대 기본 폼으로 넓힌 풀이며 새 키는 POOL에 없는 상대 타입 조합에서 나온다 "
              "— 과제 풀 안의 시험 가능성은 R·S가 마지막이다. (b) 21쌍은 {k_cl}개 상대 타입 조합 쌍에 몰려 있어 쌍끼리 "
              "독립이 아니다. 지렛대는 U 기전 기록(사슬 진입부 차단에서 MBON13 회복)을 보고 골랐고 같은 측정의 출력 차단 "
              "읽기는 '사슬 비지지'였다. 이것은 Q 결과로 고른 지렛대 계열의 R·S·T·U에 이은 다섯 번째 시도다(V.0). V 세트 "
              "군집 상관 모형에서 G_fail_S null {cl_null}, 오선택(harm) {cl_harm1} / {cl_harm2}. 이 결과는 이 64쌍 조건부 "
              "기술 결과다. 작동 특성은 V 세트 조건부 값이며 Q → R → S → T → U → V 전체 절차의 오선택률이 아니다. 실제 "
              "커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 "
              "가능성이 섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristics (V.6, V.9.5 P2-9)
OC_NOTES = (s_rules.OC_NOTES[0],
            "V.6: 독립 가정은 V 세트에서 낙관적이다 — (b)·(a) 쌍이 상대 타입 조합 군집에 몰려 있다. V 세트의 군집으로 계산한 "
            "군집 상관 모형(T.9.6·T.9.7과 같은 방법, ICC 0.3)을 옆에 기록한다.",
            "V.6: 지렛대를 U 기전 기록을 보고 골랐으므로 짝수 관문은 낙관적일 수 있다.",
            "작동 특성은 V 세트 조건부 값이며 Q → R → S → T → U → V 전체 절차의 오선택률이 아니다.")


def _sealed(table: dict) -> dict:
    t = {k: v for k, v in table.items() if k != "sha256"}
    return dict(t, sha256=hashlib.sha256(canonical(t).encode()).hexdigest())


def oc(spec) -> dict:
    """V.6: S.6's exact independent model (s_rules.oc) on n_b 21 · n_a 43 with V's notes."""
    table = {k: v for k, v in s_rules.oc(spec).items() if k != "sha256"}
    table["notes"] = list(OC_NOTES)
    return _sealed(table)


def oc_cluster(clusters_b: list, clusters_a: list) -> dict:
    """V.6 / V.9.5 P2-9: T's cluster-correlated model (t_rules.oc_cluster: ICC 0.3, 10⁵ draws, T's seed, T's g rows and
    grids) on V's set clusters from block set, with V's notes; the sha is over the table."""
    sp = dataclasses.replace(T_SPEC, clusters_b=tuple(tuple(c) for c in clusters_b),
                             clusters_a=tuple(tuple(c) for c in clusters_a))
    table = t_rules.oc_cluster(sp)
    return _sealed(dict(table, notes=list(OC_NOTES[1:])))


def cluster_values(table: dict) -> dict:
    """V.9.4: G_fail_S null and the two harm rows of the cluster model, formatted for the SELECTED sentence."""
    g = table["g_fail"]
    null = [r["p"] for r in g if r["kind"] == "null"]
    harm = [r["p"] for r in g if r["kind"] == "harm"]
    return dict(cl_null=f"{null[0]:.3f}", cl_harm1=f"{harm[0]:.3f}", cl_harm2=f"{harm[1]:.3f}")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_v_rules.py -q`
Expected: all pass (~5 s; the cluster-model test runs T's Monte Carlo once).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/v_rules.py tests/brain/v_fixtures.py tests/brain/test_v_rules.py
git commit -m "feat(v): v_rules — reuse R/T/U, path and KC-input reproduction, set outcome, z_V with SD, combined-engine KC band, even filter then testable_b, gate 2 on h4 then z_V, V.7/V.9 sentences, OCs"
```

---

### Task 4: `v_runner` (1) — context, chain and refusals, reuse, path, KC input, set, z_V, KC band, even

**Files:**
- Create: `flymon/brain/v_runner.py`, `tests/brain/v_world.py`
- Test: `tests/brain/test_v_runner.py`

**Interfaces:**
- Consumes: Tasks 1–3; `u_runner.build_ctx` / `U_HASHED_FILES` / `r_raw_spec`, `u_records` (`side`, `row_diffs`, `oracle_diffs`, `even_record`), `r_records` (`cond_summary`, `gate1_record`, `compare`, `KEEP`), `e_runner.summary_git`, `h3_store` (`git_state`, `sha256_file`, `canonical`, `ROOT`), `h3_spec.SPEC`.
- Produces: `ORDER`, `GATES`, `EXIT_REFUSE = 2`, `EXIT_KEY = 7`, `V_PIPELINE_FILES`, `V_HASHED_FILES`, `DECISION_FILES`, `JUDGE_MARKER`, `REREAD_MARKER`, `DONE_MARKER`, `PINNED`, `git_state()`, `pipeline_key()`, `decision_key()`, `decision_pins(doc) -> {z_sha256, set_sha256, kc_input_sha256}`, `n_presentations()`, `kc_values(doc)`, `apl_spec(spec)`, `u_rows_reader(spec)`, `build_ctx(spec, npz) -> dict` (ctx keys: U's minus `t_set` / `set_odours` / `judgement_rows` / `clusters`, plus `kc_candidates()`, `v_set(kc)`, `kc_record(kc)`, `set_odours(kc, blk)`, `set_e0_odours(kc, blk)`, `judgement_rows(kc, blk)`, `clusters(rows)`, `csc_facts()`, `u_doc()`, `u_git()`, `u_none()`, `u_rows()`), `Runner(measure, zm, ctx, spec, summary_path=None, code=None, tcode=None, ucode=None, pipeline=None, archive_root=None)` with `stage_reuse`, `stage_path`, `stage_kc_input`, `stage_set`, `stage_z`, `stage_kc_band`, `stage_even` and the helpers `_require`, `_write`, `_stamp`, `_side`, `_r_even`, `_set_call`, `_measurer`, `m_h4`, `_z_v`, `_z_for`, `_m_for`. `v_world`: `SPEC`, `CODE`, `TCODE`, `UCODE`, `PIPE`, `APL`, `ZV`, `COUNTS`, `ZScripted`, `Scripted`, `World`, `doc()`, `through(w, m, last, r, zm)`.

- [ ] **Step 1: Write the world and the failing test**

```python
"""A scripted V world for the runner tests (no engine, no pool): a fake ctx (R's, T's and U's summaries as the reuse
source, T's z rows and U's entry_f0 rows for the path gate, the static edit facts, 20 fake KC candidate odours of which
5 are U's, a set callable that honours block kc_input's values and block set, the reference set as 48 fake odours, 112
fake calibration odours, R's even raw as fabricated entries, P's block as fixture rows), a ZScripted reference / rest
measurer whose counts depend on the edit (none: block h4's z of this world, A 10 / 9, P 26 / 19; APL->MBON05 only: T's
lever shape, MBON13 Δ 1 — fails the guard; L_V: z_V A 6 / 3, P 80 / 20, Δ 6 — passes), and a Scripted measurer built
per z (`at(z)`) whose oracle writes real cache-like files under results/v/cache keyed by (block, edit) plans, whose P
arms come from s_fixtures and whose KC activity is scripted per (block, edit). Default even plan: L_V testable_b 13,
punishment passes everywhere (net drops 0)."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

from flymon.brain import v_rules
from flymon.brain import v_runner as VR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.u_measure import is_u_edit, u_edit
from flymon.brain.v_spec import LEVER_V
from flymon.brain.v_spec import SPEC as V_SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli
from tests.brain.u_fixtures import ref_rows, rest_rows
from tests.brain.v_fixtures import CANDS, U_IDS, r_doc, t_doc, u_doc  # noqa: F401  (re-exported)

SPEC = dataclasses.replace(V_SPEC, z_h4=(("A", Z["A"]), ("P", Z["P"])), sha_none="sha-C", sha_zero="sha-L0",
                           sha_combined="sha-V", t_measure_key_t="t" * 64, t_z_detail_sha256="d" * 64,
                           u_measure_key_u="u" * 64, u_path_detail_sha256="e" * 64, u_scan_detail_sha256="f" * 64)
CODE = {"key": SPEC.r_shared_key}
TCODE = {"key": "t" * 64}
UCODE = {"key": "u" * 64}
PIPE = {"key": "p" * 64}
N_REF = 96
APL = u_edit(0.0)
ZV = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
REC_TYPES = list(dict.fromkeys(TYPES + list(SPEC.chain_types)))
EXTRA = {t: 1 for t in REC_TYPES if t not in ("MBON13", "MBON05")}
CHAIN = dict(SPEC.contrast_declared()["chain_entry"])
COUNTS = {"none": ([1, 19] * 48, [7, 45] * 48),                       # block h4's z of this world; Δ 10
          APL: ([0, 2] * 48, [60, 100] * 48),                          # MBON13 Δ 1, zero share 0.5 (T's lever shape)
          LEVER_V: ([3, 9] * 48, [60, 100] * 48)}                      # z_V A 6 / 3, P 80 / 20; MBON13 Δ 6
_OC = {}


def sha_of(edit: str) -> str:
    return {"none": "sha-C", APL: "sha-L0", LEVER_V: "sha-V"}.get(edit, f"sha-{edit}")


def edges_of(edit: str) -> int:
    return SPEC.lever_edges if is_u_edit(edit) else 0


def cached_oc_cluster(cb, ca):
    k = json.dumps([cb, ca])
    if k not in _OC:
        _OC[k] = REAL_OC_CLUSTER(cb, ca)
    return _OC[k]


REAL_OC_CLUSTER = v_rules.oc_cluster


def t_rows() -> dict:
    """T's z rows of this world: "none" = the unedited counts, T's lever = the APL->MBON05-only counts."""
    out = {}
    for t_edit, edit, e, sha in ((SPEC.t_none_edit, "none", 0, "sha-C"), (SPEC.t_lever_edit, APL, 2, "sha-L0")):
        a, p = COUNTS[edit]
        out[t_edit] = dict(ref=[dict(r, types={"MBON13": r["types"]["MBON13"], "MBON05": r["types"]["MBON05"]})
                                for r in ref_rows(a, p, e, sha)], rest=rest_rows(N_REF, 0, e, sha))
    return out


def csc_facts(**kw) -> dict:
    f = dict(sha_none="sha-C", csc_sha256="sha-V", apl_edges=2, block_edges=dict(sorted(CHAIN.items())), changed=13,
             changed_pairs={}, mbon05_apl=dict(n=2, before=[-7.8651862, -12.928195], after=[-7.8651862, -12.928195],
                                               same_bits=True))
    return dict(f, **kw)


class ZScripted:
    """u_measure.UZMeasurer's interface; counts[edit] = (MBON13 counts, MBON05 counts); edges / blocks override the
    edge labels per edit."""

    def __init__(self, counts=None, edges=None, blocks=None, n=None):
        self.counts = {**COUNTS, **(counts or {})}
        self.edges, self.blocks, self.n = dict(edges or {}), dict(blocks or {}), n
        self.calls = []

    def _rows(self, edit):
        a, p = self.counts[edit]
        bl = self.blocks.get(edit, CHAIN if edit == LEVER_V else {})
        return a, p, self.edges.get(edit, edges_of(edit)), bl

    def reference(self, edit, odors, strength, settle_ms, read_steps):
        self.calls.append(("reference", edit, len(odors)))
        a, p, e, bl = self._rows(edit)
        rows = ref_rows(a, p, e, sha_of(edit), extra=EXTRA, blocks=bl)
        return rows[:self.n] if self.n else rows

    def rest(self, edit, seeds, settle_ms, read_steps):
        self.calls.append(("rest", edit, len(seeds)))
        _, _, e, bl = self._rows(edit)
        return rest_rows(len(seeds), 0, e, sha_of(edit), extra=EXTRA, blocks=bl)


def u_rows() -> dict:
    zm = ZScripted()
    return dict(edit=LEVER_V, ref=zm.reference(LEVER_V, [None] * 48, 0.35, 800.0, 600),
                rest=zm.rest(LEVER_V, list(range(N_REF)), 800.0, 600))


class Scripted:
    """RMeasurer's interface, built per z through at(z). plan[(block, edit)][pair key] = (r_ok, p_ok, bal) (default:
    punish passes only); override[(block, cond name)] = dict(edges=, edit=, sha=) changes one block's condition only;
    drop = {edit: P drop}; arm_edges = {edit: edges}; kc = {edit or (block, edit): {odour: frac}};
    st["fail_once"] makes the next oracle call raise like an interrupted measurement."""

    def __init__(self, plan=None, override=None, drop=None, arm_edges=None, kc=None):
        self.plan, self.override = plan or {}, override or {}
        self.drop, self.arm_edges, self.kc = dict(drop or {}), dict(arm_edges or {}), kc or {}
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

    def _drop(self, edit):
        return self.drop.get(edit, 12 if is_u_edit(edit) else 16)

    def _edges(self, edit):
        return self.arm_edges.get(edit, edges_of(edit))

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self._drop(i["edit"]), sha_of(i["edit"]), self._edges(i["edit"])) for i in items]

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, edit, len(odours)))
        fr = self.kc.get((block, edit), self.kc.get(edit, {}))
        return {o: dict(frac=[fr.get(o, 0.05)] * len(seeds), max_win=[3] * len(seeds),
                        edit_edges=[self._edges(edit)], csc_sha256=[sha_of(edit)]) for o in odours}

    def oracle(self, rows, cond, block, seeds):
        self.calls.append(("oracle", block, cond.name, cond.edit, len(rows), tuple(self.z["A"])))
        if self.st["fail_once"]:
            self.st["fail_once"] = False
            raise RuntimeError("worker died")
        o = self.override.get((block, cond.name), {})
        out = []
        for r in rows:
            k = key_of(r)
            flags = self.plan.get((block, cond.edit), {}).get(k, (False, True, False))
            res = fake_oracle(*flags, sha=o.get("sha", sha_of(cond.edit)),
                              edges=o.get("edges", self._edges(cond.edit)), edit=o.get("edit", cond.edit),
                              n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ck = hashlib.sha256(f"{block}|{cond.name}|{cond.edit}|{k}|{self.z}".encode()).hexdigest()
            f = Path(SPEC.cache_dir) / "r_oracle" / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(VR.v_rules, "oc_cluster", cached_oc_cluster)
        self.judge_commits, self.judgement_calls, self.set_calls, self.odour_calls = [], 0, 0, 0
        self.r, self.r_git = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"])
        self.t, self.t_git = t_doc(), dict(tracked=True, dirty=False, judge_commits=[])
        self.u, self.u_git = u_doc(), dict(tracked=True, dirty=False, judge_commits=[])
        self.t_rows, self.u_rows, self.facts, self.short = t_rows(), u_rows(), csc_facts(), False
        monkeypatch.setattr(VR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(VR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=300)
        eb = self.keys(self.even, "b")
        self.r_even_plan = {"C": {k: (True, True, False) for k in eb[:7]},
                            "L": {k: (True, True, False) for k in eb[:16]}}
        self.even_plan = {("even", LEVER_V): {k: (True, True, False) for k in eb[:13]},
                          ("path", APL): dict(self.r_even_plan["L"]),
                          ("path", "none"): dict(self.r_even_plan["C"])}
        p_ref = {(r["direction"], r["arm"], r["seed"]): {k: v for k, v in r.items()
                                                         if k not in ("direction", "x", "y", "point", "r")}
                 for r in p_rows(P_SPEC, "none", 16)}
        self.archive = tmp_path / "archive"
        odors = [dict(name=f"R{j:02d}", seeds=[1000 + 2 * j, 1001 + 2 * j], strengths={}) for j in range(N_REF // 2)]
        calib = {f"C{i}|T": {"G": 1.0} for i in range(112)}
        ids = list(calib)
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, rec_types=REC_TYPES,
                        even_rows=self.even, ref_odors=odors, probe_seeds=[s for o in odors for s in o["seeds"]],
                        ref_strength=0.35, ref_window=(800.0, 600), calib=(calib, ids[:56], ids[56:]), cap_ok=True,
                        enc_per={o: 0.05 for o in ids}, r_even=self._r_even,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git),
                        t_doc=lambda: json.loads(json.dumps(self.t)), t_git=lambda: dict(self.t_git),
                        t_rows=lambda: json.loads(json.dumps(self.t_rows)),
                        u_doc=lambda: json.loads(json.dumps(self.u)), u_git=lambda: dict(self.u_git),
                        u_none=lambda: SPEC.u_kc_none(self.u), u_rows=lambda: json.loads(json.dumps(self.u_rows)),
                        csc_facts=lambda: dict(self.facts), kc_candidates=self._cands, v_set=self._v_set,
                        kc_record=lambda kc: dict(outside={"none": [], "lever": []}, dropped=[],
                                                  candidate_rows_dropped=0, set_rows={}),
                        set_odours=self._set_odours, set_e0_odours=self._set_e0_odours,
                        judgement_rows=self._judgement_rows, clusters=self._clusters)

    # ---- the set callables (V.2): honour block kc_input's values and block set --------------------------------------
    def _cands(self):
        return dict(odours={o: {"G": 1.0} for o in CANDS}, cap={o: 200.0 for o in CANDS}, cap_hz=1000 / 3,
                    n_rows=40, n_turns=1986)

    def _js(self, kc):
        lo, hi = SPEC.valid_band
        used = [o for o in CANDS if all(lo <= kc[e][o] <= hi for e in ("none", "lever"))][:10]
        b, a = (self.judge[:21], self.judge[21:]) if not self.short else ([], [])
        return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=296, last_turn_b=30, last_turn_a=296,
                    status="STOP_SET_SHORT" if self.short else "OK",
                    skipped=dict(cap=16, collision=0, e1=0, used=722, glom_dup=0, in_set=87, pool_only=171,
                                 kc_input=sum(not all(lo <= kc[e][o] <= hi for e in ("none", "lever"))
                                              for o in CANDS)),
                    n_opp=127, n_combos=1986, n_odours=len(used), e1_clashes=[], cap_fails=[], all_off_pool=True,
                    clusters_b=[["GROUND*", "NORMAL", 21]], clusters_a=[["FIRE*", 43]], digest_e0_b="b" * 64,
                    digest_e0_a="a" * 64, digest_keys="k" * 64, odour_ids=used)

    def _v_set(self, kc):
        self.set_calls += 1
        return self._js(kc)

    def _checked(self, kc, blk):
        from flymon.brain.v_pairs import check_v_set
        js = self._js(kc)
        bad = check_v_set(js, blk)
        if bad:
            raise ValueError("; ".join(bad))
        return js

    def _set_odours(self, kc, blk):
        self.odour_calls += 1
        return {o: {"G": 1.0} for o in self._checked(kc, blk)["odour_ids"]}

    def _set_e0_odours(self, kc, blk):
        return {f"{key_of(r)}|x": {"E": 1.0} for r in self._checked(kc, blk)["b"][:3]}

    def _judgement_rows(self, kc, blk):
        self._checked(kc, blk)
        self.judgement_calls += 1
        return self.judge

    def _clusters(self, rows):
        return {key_of(r): ("GROUND* 대 NORMAL" if r["axis"] == "b" else "FIRE*") for r in rows}

    def _r_even(self, rows, name):
        lever = name == "L"
        return got_for(rows, self.r_even_plan[name], sha="sha-L0" if lever else "sha-C", edges=2 if lever else 0,
                       edit=SPEC.t_lever_edit if lever else SPEC.no_edit)

    def runner(self, m, zm=None, code=None, tcode=None, ucode=None, pipeline=None):
        return VR.Runner(m.at, zm or ZScripted(), self.ctx, SPEC, code=code or CODE, tcode=tcode or TCODE,
                         ucode=ucode or UCODE, pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def scripted(self, **kw):
        m = Scripted(**kw)
        m.plan = {**self.even_plan, **m.plan}
        return m

    def pass_plan(self, n=14, c=8, f_a=3):
        """judgement: L_V n / C c (b) testable and L_V f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", LEVER_V): {**{k: (True, True, False) for k in jb[:n]},
                                     **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "none"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through(w, m, last="gate2", r=None, zm=None):
    r = r or w.runner(m, zm)
    for s in VR.ORDER[:VR.ORDER.index(last) + 1]:
        out = getattr(r, f"stage_{s}")()
        if s in VR.GATES:
            assert out["outcome"] == "PASS", (s, out)
        if s == "smoke":
            assert out["problems"] == [], out["problems"]
    return r
```

```python
"""The V stage chain up to gate ② (V.3 as ordered by V.9.6): reuse -> path -> kc_input -> set -> z -> kc_band -> even ->
smoke -> oc -> gate2_oc -> gate2; each stage refuses (exit 2, nothing written) when an earlier block is missing, a
later one exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted or a hashed
file is dirty; STOP_REUSE, STOP_V_PATH_REPRO (path, kc_input, kc_band), STOP_SET_SHORT, STOP_Z_DEGENERATE (SD in the
sentence), STOP_STRENGTH_LEVER, STOP_EVEN_REPRO (L_V then unmeasured), STOP_EVEN_PUNISH / STOP_EVEN_LOW_LEVER (POOL
condition appended), a smoke problem and gate ② STOPs (both scales) block every later stage; the reuse condition broken
after `reuse` refuses with exit 7; L_V's oracle runs on z_V and C / E0's on block h4's z; the set is regenerated from
block kc_input and checked against block set; the judgement set is touched only through its odours before jm."""
import json
from pathlib import Path

import pytest

from flymon.brain import v_rules
from flymon.brain import v_runner as VR
from flymon.brain.v_spec import LEVER_V
from tests.brain.v_world import APL, CANDS, SPEC, U_IDS, ZV, World, ZScripted, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to(w, m, zm, last):
    r = w.runner(m, zm)
    for s in VR.ORDER[:VR.ORDER.index(last)]:
        getattr(r, f"stage_{s}")()
    return r


def test_chain_to_even(w):
    m, zm = w.scripted(), ZScripted()
    through(w, m, last="even", zm=zm)
    d = doc()
    assert set(d) == set(VR.ORDER[:VR.ORDER.index("smoke")])
    assert all(d[b]["pipeline_key"] == "p" * 64 and d[b]["u_measure_key"] == "u" * 64 for b in d)
    assert d["reuse"]["records"]["u"]["entry_reading"] == "사슬 지지"
    pa = d["path"]
    assert [(c["engine"], c["diffs"]) for c in pa["checks"]] == [
        ("편집 없는 엔진", []), ("APL→MBON05 제거 단독 엔진", []), ("조합 엔진", []), ("APL→MBON05 제거 단독 엔진", []),
        ("편집 없는 엔진", [])]
    assert [c[1] for c in zm.calls if c[0] == "reference"] == ["none", APL, LEVER_V]
    assert pa["sides"]["none"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} and Path(SPEC.path_detail).exists()
    assert pa["sides"]["lever"]["block_edges"] == [json.dumps(dict(SPEC.contrast_declared()["chain_entry"]),
                                                              separators=(",", ":"), sort_keys=True)]
    assert pa["records"]["lever"] == dict(p_mean_ratio=80.0 / 26.0, sd_ratio={"A": 3.0 / 9.0, "P": 20.0 / 19.0})
    assert set(pa["sides"]["lever"]["mech"]["types"]) == set(w.ctx["rec_types"])
    pc = [c for c in m.calls if c[0] == "oracle" and c[1] == "path"]
    assert pc == [("oracle", "path", "L", APL, 3, (10.0, 9.0)), ("oracle", "path", "C", "none", 3, (10.0, 9.0))]
    ki = d["kc_input"]
    assert ki["n_candidates"] == 20 and ki["u_compare"]["n_shared"] == 5 and ki["u_compare"]["differ"] == []
    assert ki["record"]["none"]["seed_range"][CANDS[0]] == [0.05, 0.05] and ki["record"]["cap"]["all_under"]
    acts = [c for c in m.calls if c[0] == "activity"]
    assert acts[:2] == [("activity", "kc_input", "none", 20), ("activity", "kc_input", LEVER_V, 20)]
    st = d["set"]
    assert st["set"]["n_b"] == 21 and st["set"]["n_a"] == 43 and st["set"]["digest_keys"] == "k" * 64
    assert "odour_ids" in st and "kc_record" in st
    z = d["z"]
    assert z["z_V"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and z["z_none_reused"] is True
    kb = d["kc_band"]
    assert kb["calib"]["outcome"] == "PASS" and kb["record_set"]["n_odours"] == 10 and kb["record_e0"]["n_odours"] == 3
    assert [c[1:3] for c in acts[2:]] == [("kc_band", LEVER_V), ("kc_band", LEVER_V), ("kc_band_e0", "none")]
    ev = d["even"]
    assert ev["c_even"] == 7 and ev["L_h4_R"] == dict(testable_b=16, matches_r_gate3=True)
    assert ev["record"]["testable_b"] == 13 and ev["record"]["drops"]["b"]["net_drop"] == 0
    evc = [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    assert evc == [("oracle", "even", "L", LEVER_V, 39, ZV["A"])]
    assert w.judgement_calls == 0 and w.odour_calls == 1


def test_reuse_stop_names_u_and_later_stages_refuse(w):
    w.u["scan"]["detail_sha256"] = "x" * 64
    out = w.runner(w.scripted(), ucode={"key": "v" * 64}).stage_reuse()
    assert out["outcome"] == v_rules.STOP_REUSE and "V 측정 키" in out["sentence"] and "scan 원자료" in out["sentence"]
    assert out["sentence"].startswith("R·T·U 재사용 조건(V.3 1)이 깨졌다(")
    with pytest.raises(SystemExit) as e:
        w.runner(w.scripted()).stage_path()
    assert e.value.code == 2


def test_reuse_broken_after_reuse_refuses_with_7(w):
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.u_git["dirty"] = True
    with pytest.raises(SystemExit) as e:
        r.stage_path()
    assert e.value.code == VR.EXIT_KEY and "path" not in doc()


def test_path_stop_on_the_unedited_rows(w):
    a, p = [1, 19] * 47 + [1, 20], [7, 45] * 48
    out = _to(w, w.scripted(), ZScripted(counts={"none": (a, p)}), "path").stage_path()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["failed"][0]["engine"] == "편집 없는 엔진"
    assert out["sentence"].startswith("V 편집 경로가 편집 없는 엔진에서 T 편집 없는 엔진 기준 집합 행을 재현하지 못했다(기준 "
                                      "집합: 1 row(s) differ")


def test_path_mutation_the_chain_entry_never_reached_the_engine(w):
    """V.3 2 unit test: an L_V whose chain-entry cut is not applied (its rows are the APL-only rows) fails against U."""
    zm = ZScripted(counts={LEVER_V: ([0, 2] * 48, [60, 100] * 48)})
    out = _to(w, w.scripted(), zm, "path").stage_path()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["failed"][0]["engine"] == "조합 엔진"
    assert "U entry_f0 기준 집합 행" in out["sentence"]


def test_path_mutation_an_oracle_on_the_unedited_csc_fails(w):
    m = w.scripted(override={("path", "L"): dict(sha="sha-C")})
    out = _to(w, m, ZScripted(), "path").stage_path()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["failed"][0]["ref"] == "R 짝수 L 원자료 3쌍"


@pytest.mark.parametrize("bad", [dict(changed=12), dict(csc_sha256="sha-x"),
                                 dict(mbon05_apl=dict(n=2, before=[-7.86, -12.93], after=[0.0, -12.93],
                                                      same_bits=False)),
                                 dict(block_edges={"MBON05->MBON01": 2, "MBON05->MBON09": 6, "MBON05->MBON11": 2})])
def test_path_invalid_on_the_static_edit_facts(w, bad):
    w.facts.update(bad)
    out = _to(w, w.scripted(), ZScripted(), "path").stage_path()
    assert out["outcome"] == v_rules.INVALID and out["reasons"]
    with pytest.raises(SystemExit):
        w.runner(w.scripted()).stage_kc_input()


def test_path_invalid_without_96_presentations(w):
    out = _to(w, w.scripted(), ZScripted(n=95), "path").stage_path()
    assert out["outcome"] == v_rules.INVALID and any("95 reference" in x for x in out["reasons"])


def test_path_invalid_on_other_chain_entry_edges(w):
    zm = ZScripted(blocks={LEVER_V: {"MBON05->MBON09": 6}})
    out = _to(w, w.scripted(), zm, "path").stage_path()
    assert out["outcome"] == v_rules.INVALID and any("chain entry edges" in x for x in out["reasons"])


def test_kc_input_stop_when_us_none_value_differs(w):
    w.u["kc"]["points"] = {k: dict(record_set=dict(per_odour_none={U_IDS[0]: 0.0501}))
                           for k in SPEC.u_kc_points}
    out = _to(w, w.scripted(), ZScripted(), "kc_input").stage_kc_input()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO
    assert out["sentence"].startswith("V 편집 경로가 편집 없는 엔진에서 U KC 블록(c0d09a7) per_odour_none을 재현하지 "
                                      f"못했다({U_IDS[0]} 0.05 ≠ U 0.0501)")


def test_kc_input_invalid_on_edges(w):
    m = w.scripted(arm_edges={LEVER_V: 3})
    out = _to(w, m, ZScripted(), "kc_input").stage_kc_input()
    assert out["outcome"] == v_rules.INVALID and any("lever: edges [3]" in x for x in out["reasons"])


def test_set_filters_by_both_engines(w):
    m = w.scripted(kc={("kc_input", LEVER_V): {CANDS[0]: 0.2}})
    r = _to(w, m, ZScripted(), "set")
    st = r.stage_set()
    assert st["outcome"] == "PASS" and CANDS[0] not in st["odour_ids"] and st["set"]["skipped"]["kc_input"] == 1


def test_set_short_sentence(w):
    w.short = True
    out = _to(w, w.scripted(), ZScripted(), "set").stage_set()
    assert out["outcome"] == v_rules.STOP_SET_SHORT and out["sentence"] == (
        "KC 입력으로 거른 넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다(0·0쌍).")
    with pytest.raises(SystemExit):
        w.runner(w.scripted()).stage_z()


def test_z_stop_names_the_sd(w):
    zm = ZScripted(counts={LEVER_V: ([0, 2] * 48, [60, 100] * 48)})
    w.u_rows = dict(w.u_rows, ref=zm.reference(LEVER_V, [None] * 48, 0.35, 800.0, 600))
    out = _to(w, w.scripted(), zm, "z").stage_z()
    assert out["outcome"] == v_rules.STOP_Z_DEGENERATE and out["sentence"] == (
        "조합 지렛대 아래 기준 집합에서 판독 MBON13이 반응성 가드를 넘지 못했다(Δ 중앙값 1.0, 0 비율 0.500, SD 1.000) — z를 "
        "정할 수 없다.")


def test_kc_band_recheck_and_strength(w):
    m = w.scripted(kc={("kc_band", LEVER_V): {f"K{i}|T": 0.0499 for i in range(15)}})
    out = _to(w, m, ZScripted(), "kc_band").stage_kc_band()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["sentence"].startswith(
        "V 편집 경로가 조합 엔진에서 KC 입력 블록의 V 세트 냄새 값을 재현하지 못했다(")


def test_kc_band_strength_stop_on_the_calibration_median(w):
    calib = list(w.ctx["calib"][0])
    m = w.scripted(kc={("kc_band", LEVER_V): {o: 0.2 for o in calib}})
    out = _to(w, m, ZScripted(), "kc_band").stage_kc_band()
    assert out["outcome"] == v_rules.STOP_STRENGTH_LEVER and out["sentence"].startswith(
        "조합 지렛대 아래에서 E-grid k2-norm s 1.0이 KC 유효 대역을 잃었다(냄새별 KC 활성 중앙값 0.2000 ∉ [0.03, 0.15])")


def test_even_repro_stop_leaves_l_unmeasured(w):
    w.r_even_plan["C"] = {k: v for k, v in list(w.r_even_plan["C"].items())[:6]}
    m = w.scripted()
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == v_rules.STOP_EVEN_REPRO and out["sentence"] == (
        "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L 16, C 6).")
    assert not [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]


def test_even_punish_filter_boundaries(w):
    ea = w.keys(w.even, "a")
    m = w.scripted(plan={("even", LEVER_V): {**{k: (True, True, False) for k in w.keys(w.even, "b")[:13]},
                                            **{k: (False, False, False) for k in ea[:2]}}})
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == v_rules.STOP_EVEN_PUNISH and out["sentence"] == (
        "조합 지렛대가 짝수 쌍에서 처벌 순감소 거름 기준((b) < 3, (a) ≤ 1)을 넘었다((b) 0, (a) 2). (POOL 안 짝수 쌍 조건부)")
    with pytest.raises(SystemExit):
        w.runner(w.scripted())._require("smoke")                         # STOP blocks every later stage


def test_even_punish_filter_passes_at_b_2_and_a_1(w):
    eb, ea = w.keys(w.even, "b"), w.keys(w.even, "a")
    m = w.scripted(plan={("even", LEVER_V): {**{k: (True, True, False) for k in eb[:13]},
                                            **{k: (True, False, False) for k in eb[19:21]},
                                            **{k: (False, False, False) for k in ea[:1]}}})
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == "PASS" and (out["record"]["drops"]["b"]["net_drop"],
                                         out["record"]["drops"]["a"]["net_drop"]) == (2, 1)


def test_even_low_lever_sentence(w):
    m = w.scripted(plan={("even", LEVER_V): {k: (True, True, False) for k in w.keys(w.even, "b")[:10]}})
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == v_rules.STOP_EVEN_LOW_LEVER and out["sentence"] == (
        "조합 지렛대 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 "
        "0/18). (POOL 안 짝수 쌍 조건부)")


def test_even_invalid_on_wrong_edges(w):
    m = w.scripted(override={("even", "L"): dict(edges=3)})
    assert _to(w, m, ZScripted(), "even").stage_even()["outcome"] == v_rules.INVALID


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    m = w.scripted()
    r = _to(w, m, ZScripted(), "kc_input")
    with pytest.raises(SystemExit):
        r.stage_path()                                                   # own block exists
    with pytest.raises(SystemExit):
        r.stage_set()                                                    # kc_input missing
    monkeypatch.setattr(VR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_kc_input()
    monkeypatch.setattr(VR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(VR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=["flymon/brain/v_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_kc_input()


def test_a_set_that_no_longer_matches_block_set_refuses_kc_band(w):
    r = _to(w, w.scripted(), ZScripted(), "kc_band")
    d = doc()
    d["set"]["set"]["digest_keys"] = "0" * 64
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_kc_band()
    assert e.value.code == 2 and "kc_band" not in doc()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_v_runner.py -q`
Expected: FAIL at collection (`ImportError: cannot import name 'v_runner'`).

- [ ] **Step 3: Write the first part of `v_runner`**

```python
"""Spec V's stage chain (V.3 as ordered by V.9.6; plan Readings):
reuse -> path -> kc_input -> set -> z -> kc_band -> even -> smoke -> oc -> gate2_oc -> gate2 -> measurement_started ->
jm:L -> jm:C -> jm:E0 -> seal -> judge (and after judge only: recompute / invalid_run), one block each in
results/summary/v_lever.json, written only through v_store. Every stage refuses (SystemExit 2, nothing written) when an
earlier block is missing, a later block exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary
has uncommitted changes or a hashed V file is dirty. Every stage after `reuse` re-checks the reuse condition (R's shared
key and repro, T's key and unedited z, U's key and blocks) and refuses with SystemExit 7 when it broke;
measurement_started, jm, seal and judge also re-check that every earlier block carries the current keys (V.8, exit 7).
A smoke with problems or a gate whose outcome is not PASS blocks every later stage, so the judgement set stays unused.
z (V.1): L_V's oracle (even, smoke's L, jm:L) runs on a measurer built with block z's z_V (α chosen on z_V); C's and
E0's, every P arm, the KC activities and the path oracle on block h4's z (`measure(z)`). Gate ② reads both P conditions
on h4 z, then V.9.2's z_V ratio.
The set (V.2) is regenerated from block kc_input's values and checked against block set on every call (ctx's lazy
callables); only kc_band (odours only), the judgement stages and `_cost` (seed counts) name it."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

from ..agent import e_rules
from ..agent.e_runner import summary_git
from . import r_records, t_records, u_records, v_records, v_rules, v_store
from .h3_spec import SPEC as _H3
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .n_rules import INVALID as P_INVALID
from .p_rules import p_judge
from .p_spec import SPEC as P_SPEC
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import p_items
from .t_measure import T_MEASURE_FILES
from .u_measure import U_MEASURE_FILES, u_edit
from .u_runner import U_HASHED_FILES, r_raw_spec
from .v_spec import smoke

ORDER = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "smoke", "oc", "gate2_oc", "gate2",
         "measurement_started", "jm:L", "jm:C", "jm:E0", "seal", "judge")
GATES = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "gate2")
GATE2_INVALID = "gate2_invalid"
EXIT_REFUSE, EXIT_KEY = 2, 7
V_PIPELINE_FILES = ("flymon/brain/v_spec.py", "flymon/brain/v_pairs.py", "flymon/brain/v_store.py",
                    "flymon/brain/v_records.py", "flymon/brain/v_rules.py", "flymon/brain/v_runner.py",
                    "scripts/run_v.py")
V_HASHED_FILES = tuple(dict.fromkeys(U_HASHED_FILES + V_PIPELINE_FILES + ("results/summary/u_lever.json",)))
# The decision code: every hashed V file outside the measurement files (R's, T's, U's). The seal pins its hash; the
# judge reads only under the sealed decision code (R 909c193's rule).
DECISION_FILES = tuple(f for f in V_HASHED_FILES
                       if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES and f not in U_MEASURE_FILES)
JUDGE_MARKER = "results/v/judge_read.json"
REREAD_MARKER = "results/v/judge_reread.json"
DONE_MARKER = "results/v/judge_done.json"
ENGINES = (("none", "none"), ("apl", "apl"), ("lever", "lever"))


def git_state() -> dict:
    return _h3_git_state(files=V_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(V_PIPELINE_FILES)


def decision_key() -> dict:
    return _files_key(DECISION_FILES)


def _sha(obj) -> str | None:
    return None if obj is None else hashlib.sha256(canonical(obj).encode()).hexdigest()


PINNED = ("z", "set", "kc_input")


def decision_pins(doc: dict) -> dict:
    """V.3 12: block z (z_V), block set and block kc_input, hashed into the seal's decision record (canonical JSON)."""
    return {f"{b}_sha256": _sha(doc.get(b)) for b in PINNED}


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


def n_presentations() -> int:
    """H.3's reference set: n odours × 2 probe seeds each (96), the count every side must have."""
    return 2 * int(_H3.reference.n)


def kc_values(doc: dict) -> dict:
    """Block kc_input's per-odour medians per engine — the set filter's only source (V.2)."""
    rec = doc["kc_input"]["record"]
    return {e: dict(rec[e]["per_odour"]) for e in ("none", "lever")}


def apl_spec(spec):
    """The APL->MBON05-only engine of V.3 2 (U's f = 0 edit), for the path stage's oracle against R's even L raw."""
    import dataclasses
    return dataclasses.replace(spec, lever_edit=u_edit(0.0))


def u_rows_reader(spec):
    """U's entry_f0 rows (results/u/scan.json, read only), refused unless the file's sha256 is the one U's scan block
    recorded and V declares (V.3 2's base for L_V)."""
    def u_rows() -> dict:
        p = Path(spec.u_scan_detail)
        if not p.exists() or sha256_file(p) != spec.u_scan_detail_sha256:
            raise ValueError(f"{spec.u_scan_detail} is missing or its sha256 is not {spec.u_scan_detail_sha256}")
        return json.loads(p.read_text())["rows"][spec.u_entry_point]
    return u_rows


def build_ctx(spec, npz: str) -> dict:
    """U's context (u_runner.build_ctx on V's spec: C3, block h4's z, the H.4 pools, the encoder summary, the even rows,
    the reference set, R's even raw read only, P's entries and stimuli, R's and T's summaries, T's z rows, the 112
    calibration odours, the ORN cap) without U's set callables, plus V's: the KC candidates, the set (from block
    kc_input's values, checked against block set), its odours, E0 odours and judgement rows, the KC record, the cluster
    labels, the static edit facts, U's summary and git state, U's per_odour_none and U's entry_f0 rows."""
    from ..agent.config import load_c3_config
    from . import v_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .u_runner import build_ctx as u_build_ctx
    ctx = u_build_ctx(spec, npz)
    for k in ("t_set", "set_odours", "judgement_rows", "clusters", "enc_per"):
        ctx.pop(k, None)
    cfg = load_c3_config(spec.m0d_summary)
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    enc, params = ctx["enc"], cfg.params
    ctx["enc_per"] = enc["strength"]["configs"][spec.config]["table"][str(spec.strength)]["per_odour"]
    u_doc = lambda: json.loads(Path(spec.u_summary).read_text())  # noqa: E731
    ctx.update(
        kc_candidates=lambda: v_pairs.candidate_odours(pops, enc, params),
        v_set=lambda kc: v_pairs.v_set(pops, enc, params, kc, spec),
        kc_record=lambda kc: v_pairs.kc_record(pops, enc, params, kc, spec),
        set_odours=lambda kc, blk: v_pairs.set_odours(pops, rc, enc, params, kc, blk, spec),
        set_e0_odours=lambda kc, blk: v_pairs.set_e0_odours(pops, enc, params, kc, blk, spec),
        judgement_rows=lambda kc, blk: v_pairs.judgement_rows(pops, rc, enc, params, kc, blk, spec),
        clusters=v_pairs.cluster_labels, csc_facts=lambda: v_records.csc_facts(conn, pops, params, spec),
        u_doc=u_doc, u_git=lambda: summary_git(spec.u_summary), u_none=lambda: spec.u_kc_none(u_doc()),
        u_rows=u_rows_reader(spec))
    return ctx


class Runner:
    def __init__(self, measure, zm, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, ucode: dict | None = None, pipeline: dict | None = None,
                 archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) on that z, one per z over one U pool and one VCache; zm: a
        u_measure.UZMeasurer (or its interface). ucode = the V measurement key (= U's measurement key)."""
        self.measure, self.zm, self.ctx, self.spec = measure, zm, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.u_measure_key = (ucode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]
        self._ms = {}

    # ---- measurers per z -----------------------------------------------------------------------------------------
    def _measurer(self, z: dict):
        k = tuple(sorted(z.items()))
        if k not in self._ms:
            self._ms[k] = self.measure(z)
        return self._ms[k]

    @property
    def m_h4(self):
        return self._measurer(_ztuple(self.ctx["z"]))

    def _z_v(self, doc) -> dict:
        return _ztuple(doc["z"]["z_V"])

    def _z_for(self, name: str, doc) -> dict:
        return self._z_v(doc) if name == self.spec.cond_names[0] else _ztuple(self.ctx["z"])

    def _m_for(self, name: str, doc):
        return self._measurer(self._z_for(name, doc))

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return v_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed V files are dirty: {gs['dirty_hashed']}")

    def _reuse_dec(self) -> dict:
        c = self.ctx
        return v_rules.reuse(c["r_doc"](), c["r_git"](), c["t_doc"](), c["t_git"](), c["u_doc"](), c["u_git"](),
                             self.code_key, self.t_measure_key, self.u_measure_key, self.spec)

    def _reuse_now(self, stage: str) -> dict:
        r = self._reuse_dec()
        if r["outcome"] != v_rules.PASS:
            refuse(f"stage {stage}: the reuse condition broke ({'; '.join(r['reasons'])}) — V has no re-measurement "
                   f"path; V stops here and the judgement set stays unused", EXIT_KEY)
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
            refuse(f"stage {stage}: later block(s) {later} exist; V never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; V never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != v_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — V stops there and the "
                   f"judgement set stays unused (V.3)")
        if i > 0:
            self._reuse_now(stage)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, t_measure_key=self.t_measure_key,
                    u_measure_key=self.u_measure_key, pipeline_key=self.pipeline_key, git=git_state(),
                    written_at=_now())

    def _write(self, stage: str, body: dict) -> dict:
        block = self._stamp(stage, body)
        v_store.write_summary_block(self.summary_path, stage, block, self.plist)
        return block

    def _side(self, edit: str, raw: dict, key: str) -> dict:
        ctx = self.ctx
        settle, steps = ctx["ref_window"]
        ref = self.zm.reference(edit, ctx["ref_odors"], ctx["ref_strength"], settle, steps)
        rest = self.zm.rest(edit, ctx["probe_seeds"], settle, steps)
        raw[key] = dict(edit=edit, ref=ref, rest=rest)
        return u_records.side(ref, rest, ctx["readout"], ctx["rec_types"], self.spec.kc_types, self.spec)

    def _r_even(self, rows, name):
        try:
            return self.ctx["r_even"](rows, name)
        except ValueError as e:
            refuse(f"V reads R's even raw by content key and it is not complete: {e}")

    def _set_call(self, what: str, doc: dict):
        try:
            return self.ctx[what](kc_values(doc), doc["set"]["set"])
        except ValueError as e:
            refuse(f"the V set does not reproduce block set (V.2): {e}")

    # ---- reuse (V.3 1) — no pool --------------------------------------------------------------------------------
    def stage_reuse(self) -> dict:
        self._require("reuse")
        sp = self.spec
        dec = self._reuse_dec()
        rd, td, ud = self.ctx["r_doc"](), self.ctx["t_doc"](), self.ctx["u_doc"]()
        rec = dict(repro_csc_sha256_none=_dig(rd, ("repro", "csc_sha256_none")),
                   r_gate3=dict(testable_b=_dig(rd, ("gate3", "testable_b")), c_even=_dig(rd, ("gate3", "c_even"))),
                   t_z=dict(outcome=_dig(td, ("z", "outcome")), none_z=_dig(td, ("z", "none", "z")),
                            t_measure_key=_dig(td, ("z", "t_measure_key"))),
                   u=dict(path=_dig(ud, ("path", "outcome")), kc=_dig(ud, ("kc", "outcome")),
                          entry_f0_z=_dig(ud, ("scan", "contrast", sp.u_entry_point, "side", "z")),
                          entry_reading=_dig(ud, ("scan", "contrast", "readings", "entry", "reading")),
                          u_measure_key=_dig(ud, ("path", "u_measure_key"))))
        body = dict(dec, shared_key=self.code_key, r_shared_key=sp.r_shared_key, t_measure_key_t=sp.t_measure_key_t,
                    u_measure_key_u=sp.u_measure_key_u, r_commits=dict(sp.r_commits), t_commits=dict(sp.t_commits),
                    u_commits=dict(sp.u_commits), records=rec)
        self._write("reuse", body)
        return body

    # ---- the path reproduction (V.3 2, V.9.5) -------------------------------------------------------------------
    def stage_path(self) -> dict:
        """The combined edit's static facts (V.1: INVALID unless 2 + 7 / 2 / 2 = 13 edges, CSC 2d359b8b…, MBON05->APL
        untouched); the reference set + same-seed rest, measured afresh with U's jobs on three engines — unedited (T's
        none rows, CSC 1aee8398…), APL->MBON05 only (T's lever rows, CSC 860cba4f…), L_V (U's entry_f0 rows, every field,
        CSC 2d359b8b…); the oracle on the first 3 even pairs (h4 z, H.4 seeds): APL->MBON05 only against R's even L raw
        and unedited against R's even C raw (labels aside). Any difference → STOP_V_PATH_REPRO; a defect → INVALID. The
        three sides are V's mechanism records (V.6). Raw rows go to results/v/path.json."""
        self._require("path")
        sp, ctx = self.spec, self.ctx
        try:
            t_rows, u_rows = ctx["t_rows"](), ctx["u_rows"]()
        except ValueError as e:
            refuse(f"T's z rows or U's entry_f0 rows cannot be read: {e}")
        facts = ctx["csc_facts"]()
        n, raw, sides, checks = n_presentations(), {}, {}, []
        chain = canonical(dict(sp.contrast_declared()["chain_entry"]))
        plan = (("none", sp.no_edit, 0, sp.sha_none, v_rules.NONE_ENGINE),
                ("apl", u_edit(0.0), sp.lever_edges, sp.sha_zero, v_rules.APL_ENGINE),
                ("lever", sp.lever_edit, sp.lever_edges, sp.sha_combined, v_rules.V_ENGINE))
        for key, edit, edges, sha, name in plan:
            s = self._side(edit, raw, key)
            sides[key] = s
            invalid = []
            if s["edit_edges"] != [edges]:
                invalid.append(f"{key}: edges {s['edit_edges']}, declared {edges}")
            if (s["n_ref"], s["n_rest"]) != (n, n):
                invalid.append(f"{key}: {s['n_ref']} reference / {s['n_rest']} rest, declared {n}")
            if key == "lever" and s["block_edges"] != [chain]:
                invalid.append(f"lever: chain entry edges {s['block_edges']}, declared {chain}")
            if key == "lever":
                ref = "U entry_f0 기준 집합 행"
                diffs = (v_records.full_row_diffs(raw[key]["ref"], u_rows["ref"], "기준 집합")
                         + v_records.full_row_diffs(raw[key]["rest"], u_rows["rest"], "휴지"))
            else:
                t_edit = sp.t_none_edit if key == "none" else sp.t_lever_edit
                ref = "T 편집 없는 엔진 기준 집합 행" if key == "none" else "T 지렛대 엔진 기준 집합 행"
                diffs = (u_records.row_diffs(raw[key]["ref"], t_rows[t_edit]["ref"], "기준 집합")
                         + u_records.row_diffs(raw[key]["rest"], t_rows[t_edit]["rest"], "휴지", rest=True))
            if s["csc_sha256"] != [sha]:
                diffs.append(f"CSC {s['csc_sha256']} ≠ {sha}")
            checks.append(dict(engine=name, ref=ref, diffs=diffs, invalid=invalid))
        rows = ctx["even_rows"][:sp.path_even_n]
        seeds, m = sp.h4_seeds(), self.m_h4
        nl, nc, _ = sp.cond_names
        for cond, rname, name in ((apl_spec(sp).cond(nl), nl, v_rules.APL_ENGINE), (sp.cond(nc), nc,
                                                                                     v_rules.NONE_ENGINE)):
            r_got = self._r_even(rows, rname)
            v_got = m.oracle(rows, cond, "path", seeds)
            checks.append(dict(engine=name, ref=f"R 짝수 {rname} 원자료 {len(rows)}쌍",
                               diffs=u_records.oracle_diffs(v_got, r_got, f"R 짝수 {rname}"), invalid=[]))
        dec = v_rules.path(checks, v_records.csc_reasons(facts, sp), sp)
        z_h4 = _ztuple(ctx["z"])
        records = {k: v_records.side_record(sides[k], sides["none"], z_h4, ctx["readout"]) for k in ("apl", "lever")}
        p = v_store.write_json(sp.path_detail, dict(u_measure_key=self.u_measure_key, readout=ctx["readout"],
                                                    rows=raw), self.plist)
        body = dict(dec, checks=checks, sides=sides, records=records, csc_facts=facts,
                    pairs=[row_key(r) for r in rows], n_presentations=n, detail_path=sp.path_detail,
                    detail_sha256=sha256_file(p), wall_s=m.last_wall_s)
        self._write("path", body)
        return body

    # ---- the KC input on both engines (V.3 3, V.9.1, V.9.5 P2-7 / P2-10) — no oracle ------------------------------
    def stage_kc_input(self) -> dict:
        self._require("kc_input")
        sp, ctx = self.spec, self.ctx
        try:
            cand = ctx["kc_candidates"]()
        except ValueError as e:
            refuse(f"the KC candidate odours cannot be listed: {e}")
        od, m, act, wall = cand["odours"], self.m_h4, {}, 0.0
        for e, edit in (("none", sp.no_edit), ("lever", sp.lever_edit)):
            act[e] = m.activity(od, edit, sp.strength, sp.kc_seeds(), "kc_input")
            wall += m.last_wall_s
        rec = v_records.kc_input_record(act, cand["cap"], cand["cap_hz"], sp)
        u_cmp = v_records.kc_u_diffs(rec["none"]["per_odour"], ctx["u_none"]())
        dec = v_rules.kc_input(rec, u_cmp, len(od), sp)
        p = v_store.write_json(sp.kc_input_detail, dict(u_measure_key=self.u_measure_key, activity=act), self.plist)
        body = dict(dec, record=rec, u_compare=u_cmp, n_candidates=len(od), n_candidate_rows=cand["n_rows"],
                    seeds=list(sp.kc_seeds()), detail_path=sp.kc_input_detail, detail_sha256=sha256_file(p),
                    wall_s=wall, note="V.5: KC 입력은 냄새 입력만 쓰고 오라클 결과를 보지 않는다.")
        self._write("kc_input", body)
        return body

    # ---- the set (V.2) — no pool --------------------------------------------------------------------------------
    def stage_set(self) -> dict:
        doc = self._require("set")
        kc = kc_values(doc)
        try:
            js = self.ctx["v_set"](kc)
            rec = self.ctx["kc_record"](kc)
        except ValueError as e:
            refuse(f"the V set cannot be generated: {e}")
        from .v_pairs import set_summary
        dec = v_rules.set_outcome(js)
        body = dict(dec, set=set_summary(js), odour_ids=js["odour_ids"], kc_record=rec,
                    note="V.9.5 P2-8: 판정 세트에 대한 어떤 오라클 측정보다 먼저 커밋한다.")
        self._write("set", body)
        return body

    # ---- z_V (V.3 5) — no pool ----------------------------------------------------------------------------------
    def stage_z(self) -> dict:
        doc = self._require("z")
        sp, ctx = self.spec, self.ctx
        side = doc["path"]["sides"]["lever"]
        dec = v_rules.z_v(side, n_presentations(), ctx["readout"], sp)
        z_h4 = _ztuple(ctx["z"])
        body = dict(dec, z_V=side["z"] if dec["outcome"] == v_rules.PASS else None, guard=side["guard"],
                    zero_sd=side["zero_sd"], record=doc["path"]["records"]["lever"],
                    z_h4={k: list(v) for k, v in z_h4.items()},
                    z_none_reused=bool(_ztuple(doc["reuse"]["records"]["t_z"]["none_z"]) == z_h4))
        self._write("z", body)
        return body

    # ---- the combined-engine KC band (V.3 6, V.9.1, V.9.5 P2-11) -------------------------------------------------
    def stage_kc_band(self) -> dict:
        doc = self._require("kc_band")
        sp, ctx = self.spec, self.ctx
        odours, single, dual = ctx["calib"]
        set_od = self._set_call("set_odours", doc)
        e0_od = self._set_call("set_e0_odours", doc)
        m = self.m_h4
        a112 = m.activity(odours, sp.lever_edit, sp.strength, sp.kc_seeds(), "kc_band")
        wall = m.last_wall_s
        aset = m.activity(set_od, sp.lever_edit, sp.strength, sp.kc_seeds(), "kc_band")
        wall += m.last_wall_s
        ae0 = m.activity(e0_od, sp.no_edit, sp.e0_strength, sp.kc_seeds(), "kc_band_e0")
        wall += m.last_wall_s
        rec112 = r_records.gate1_record(a112, single, dual, ctx["enc_per"], sp)
        per = e_rules.odour_activity({o: aset[o]["frac"] for o in aset})
        set_rec = dict(per_odour=per, n_odours=len(per), edit_edges=sorted({int(x) for v in aset.values()
                                                                           for x in v["edit_edges"]}),
                       csc_sha256=sorted({s for v in aset.values() for s in v["csc_sha256"]}))
        want = kc_values(doc)["lever"]
        bad = [o for o in sorted(per) if per[o] != want.get(o)]
        recheck = [f"{len(bad)} odour(s) differ: " + "; ".join(f"{o} {per[o]!r} ≠ {want.get(o)!r}" for o in bad[:3])
                   ] if bad else []
        e0 = e_rules.odour_activity({o: ae0[o]["frac"] for o in ae0})
        dec = v_rules.kc_band(rec112, ctx["cap_ok"], set_rec, recheck, sp)
        body = dict(dec, record_calib=rec112, record_set=set_rec, record_e0=dict(per_odour=e0, n_odours=len(e0),
                                                                                  strength=sp.e0_strength),
                    cap_ok=bool(ctx["cap_ok"]), seeds=list(sp.kc_seeds()), wall_s=wall,
                    note="V.5: 판정 세트의 냄새 입력만 쓰고 오라클 결과를 보지 않는다. E0 냄새 값은 기록이다(V.9.5 P2-11).")
        self._write("kc_band", body)
        return body

    # ---- the even pairs (V.3 7, V.9.3) ----------------------------------------------------------------------------
    def stage_even(self) -> dict:
        """C from R's even raw (h4 z) must reproduce c_even 7 first (STOP_EVEN_REPRO, L_V not measured); then L_V's 39
        even pairs on z_V (α on z_V), resumable, R's gate ③ validity (INVALID on a defect), ① the punishment filter, ②
        testable_b ≥ 11."""
        doc = self._require("even")
        sp = self.spec
        rows, seeds = self.ctx["even_rows"], sp.h4_seeds()
        keys = [row_key(r) for r in rows]
        nl, nc, _ = sp.cond_names
        r_c, r_l = self._r_even(rows, nc), self._r_even(rows, nl)
        z_h4 = _ztuple(self.ctx["z"])
        t_l = r_raw_spec(sp)
        C = r_records.cond_summary(r_c, sp.cond(nc), sp, z_h4, keys, seeds)
        L_h4 = r_records.cond_summary(r_l, t_l.cond(nl), t_l, z_h4, keys, seeds)
        l_h4 = (L_h4["aggregate"] or {}).get("testable_b")
        keep = r_records.KEEP
        base = dict(C={k: C.get(k) for k in keep}, L_h4_R=dict(testable_b=l_h4,
                                                                matches_r_gate3=bool(l_h4 == sp.r_even_L_h4)),
                    seeds=seeds, n_pairs=len(rows))
        stop = v_rules.even_repro(C, sp, l_h4)
        if stop:
            return self._write("even", dict(stop, **base))
        z_v = self._z_v(doc)
        m = self._measurer(z_v)
        got = m.oracle(rows, sp.cond(nl), "even", seeds)
        L = r_records.cond_summary(got, sp.cond(nl), sp, z_v, keys, seeds)
        v = v_rules.even_validity(L, C, sp, doc["reuse"]["records"]["repro_csc_sha256_none"])
        record = u_records.even_record(L, C, sp)
        c_even = (C["aggregate"] or {}).get("testable_b")
        dec = dict(outcome=v_rules.INVALID, reasons=v) if v else v_rules.even(record, c_even, sp)
        body = dict(dec, **base, c_even=c_even, record=record, L={k: L.get(k) for k in keep},
                    compare=r_records.compare(L, C, None, sp), pairs=L["pairs"],
                    z=[list(z_v[k]) for k in ("A", "P")], wall_s=m.last_wall_s,
                    note="V.9.3: 짝수 쌍은 POOL 안 상대라 판정 세트와 처벌 통과 분포가 다를 수 있다. 거름은 판정을 보증하지 "
                         "않는다.")
        self._write("even", body)
        return body
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_v_runner.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/v_runner.py tests/brain/v_world.py tests/brain/test_v_runner.py
git commit -m "feat(v): v_runner (1) — chain, reuse R/T/U, path (three engines afresh + 3+3 even pairs), KC input on both engines, set from block kc_input, z_V, combined-engine KC band, even filter"
```

---

### Task 5: `v_runner` (2) — smoke, the operating characteristics, gate ② on two scales

**Files:**
- Modify: `flymon/brain/v_runner.py` (append methods to class `Runner`)
- Test: `tests/brain/test_v_gate2.py`

**Interfaces:**
- Consumes: Task 4's `Runner` helpers; `r_runner.p_items`, `p_rules.p_judge`, `p_spec.SPEC`, `t_records` (`ratio`, `p_zlever`, `gate2_oc`), `v_records.ratio_two_z`, `v_rules` (`gate2`, `oc`, `oc_cluster`, `cluster_values`), `v_spec.smoke`, `n_rules.INVALID`.
- Produces: `Runner.stage_smoke`, `_smoke_problems`, `_cost`, `stage_oc` (block `oc`: `independent`, `cluster`, `cluster_values`, `cluster_path`, `cluster_file_sha256`), `stage_gate2_oc`, `stage_gate2(rerun=False)` (block `gate2`: `outcome`, `label_L`, `label_C`, `scale`, `ratio`, `ratio_z_V`, `p_L_on_z_V`, …), `_write_rerun`.

- [ ] **Step 1: Write the failing test**

```python
"""The V chain from smoke to gate ② (V.3 8-10, V.6, V.9.2): smoke on 24_609_xxx / 25_409_xxx checks L_V's edges, CSC
(block path's) and z_V and C / E0's h4 z and R's repro CSC; the OCs (independent with V's notes, T's cluster method on
block set's clusters written to results/v/oc_cluster.json with its sha); gate ② labels both P conditions on h4 z, reads
the h4 ratio first and then ℓ_L(z_V) / ℓ_C(h4 z), each ≥ 0.5 in both directions; a smoke problem or any gate ② STOP
blocks the marker and the judgement; gate ②'s one INVALID rerun; one measurer per z."""
from pathlib import Path

import pytest

from flymon.brain import v_rules
from flymon.brain import v_runner as VR
from flymon.brain.v_spec import LEVER_V
from tests.brain.v_world import SPEC, World, ZScripted, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to(w, m, zm, last):
    r = w.runner(m, zm)
    for s in VR.ORDER[:VR.ORDER.index(last)]:
        getattr(r, f"stage_{s}")()
    return r


def test_chain_to_gate2(w):
    m, zm = w.scripted(), ZScripted()
    through(w, m, zm=zm)
    d = doc()
    assert set(d) == set(VR.ORDER[:VR.ORDER.index("measurement_started")])
    orc = d["smoke"]["oracle"]
    assert orc["L"]["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and orc["L"]["edit"] == LEVER_V
    assert orc["C"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} == orc["E0"]["z"]
    oc = d["oc"]
    assert Path(SPEC.oc_cluster_out).exists() and oc["cluster_path"] == SPEC.oc_cluster_out
    assert oc["cluster"]["clusters_b"] == [21] and oc["cluster"]["clusters_a"] == [43]
    assert set(oc["cluster_values"]) == {"cl_null", "cl_harm1", "cl_harm2"}
    assert any("U 기전 기록" in n for n in oc["independent"]["notes"])
    g2 = d["gate2"]
    assert (g2["label_L"], g2["label_C"], g2["seeds"], g2["scale"]) == (
        "LEARNS_CONFIRMATORY", "LEARNS_CONFIRMATORY", list(SPEC.p.seeds), None)
    assert g2["edit_edges_L"] == [2] and g2["ratio_z_V"]["r1"]["ratio"] == pytest.approx(3 * g2["ratio"]["r1"]["ratio"])
    assert g2["p_L_on_z_V"]["ell"]["r1"] == pytest.approx(g2["ratio_z_V"]["r1"]["ell_L"])
    assert w.judgement_calls == 0 and w.odour_calls == 1
    assert [c for c in m.calls if c[0] == "arms"] == [("arms", "smoke", 48), ("arms", "gate2", 384)]



def test_smoke_problem_blocks_later_stages(w):
    m = w.scripted(override={("smoke", "L"): dict(sha="sha-C")})
    out = _to(w, m, ZScripted(), "smoke").stage_smoke()
    assert out["problems"] and any("block path's L_V" in p for p in out["problems"])
    with pytest.raises(SystemExit):
        w.runner(m).stage_oc()


@pytest.mark.parametrize("drop,outcome", [({LEVER_V: 2}, "STOP_PUNISH_BROKEN"), ({"none": 2}, "STOP_P_REFERENCE"),
                                          ({LEVER_V: 6}, "STOP_PUNISH_WEAKENED")])
def test_gate2_stops_on_h4_block_the_judgement(w, drop, outcome):
    m = w.scripted(drop=drop)
    out = _to(w, m, ZScripted(), "gate2").stage_gate2()
    assert out["outcome"] == outcome
    if outcome == "STOP_PUNISH_WEAKENED":
        assert out["scale"] == "h4"
    with pytest.raises(SystemExit):
        w.runner(m)._require("measurement_started")                     # STOP blocks the marker


def test_gate2_stop_on_the_z_v_scale_alone(w):
    """V.9.2: σA_V = 30 makes ℓ_L(z_V) / ℓ_C(h4 z) = 13/30 / (17/9) ≈ 0.23 while the h4 ratio 13/17 passes."""
    lever = ([5, 65] * 48, [60, 100] * 48)
    zm = ZScripted(counts={LEVER_V: lever})
    w.u_rows = dict(edit=LEVER_V, ref=zm.reference(LEVER_V, [None] * 48, 0.35, 800.0, 600),
                    rest=zm.rest(LEVER_V, list(range(96)), 800.0, 600))
    m = w.scripted()
    out = _to(w, m, zm, "gate2").stage_gate2()
    assert out["outcome"] == "STOP_PUNISH_WEAKENED" and out["scale"] == "z_V" and set(out["low"]) == {"r1", "r2"}
    assert all(out["ratio"][d]["ratio"] >= 0.5 for d in ("r1", "r2"))
    assert out["sentence"].startswith("APL→MBON05 제거 아래 같은 시드의 처벌 학습량이 지렛대 없는 쪽의 절반에 못 미쳤다(ℓ_r1 "
                                      f"{out['ratio_z_V']['r1']['ell_L']:.3f} 대 ")


def test_gate2_invalid_rerun_once_with_a_changed_pipeline_key(w):
    _to(w, w.scripted(), ZScripted(), "gate2")
    r = w.runner(w.scripted(arm_edges={LEVER_V: 3}))                    # only gate ②'s P arms carry 3 edges
    assert r.stage_gate2()["outcome"] == v_rules.INVALID
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                                        # same pipeline key
    m2 = w.scripted()
    r2 = w.runner(m2, pipeline={"key": "q" * 64})
    out = r2.stage_gate2(rerun=True)
    assert out["outcome"] == "PASS" and out["rerun_of"] and "gate2_invalid" in doc()
    with pytest.raises(SystemExit):
        r2.stage_gate2(rerun=True)


def test_one_measurer_per_z(w):
    m = w.scripted()
    made = []

    def measure(z):
        made.append(dict(z))
        return m.at(z)
    r = VR.Runner(measure, ZScripted(), w.ctx, SPEC, code={"key": SPEC.r_shared_key}, tcode={"key": "t" * 64},
                  ucode={"key": "u" * 64}, pipeline={"key": "p" * 64}, archive_root=w.archive)
    through(w, m, r=r)
    assert sorted(tuple(z["A"]) for z in made) == [(6.0, 3.0), (10.0, 9.0)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_v_gate2.py -q`
Expected: FAIL (`AttributeError: 'Runner' object has no attribute 'stage_smoke'`).

- [ ] **Step 3: Append the methods to class `Runner`** (at the end of `flymon/brain/v_runner.py`, indented as methods)

```python
    # ---- smoke (V.3 8) -------------------------------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp = self.spec
        sm = smoke(sp)
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
                        edit_edges=sorted({int(r["r"]["edit_edges"]) for r in rs}),
                        csc_sha256=sorted({r["csc_sha256"] for r in rs}), n_rows=len(rs))
        b = [r for r in self.ctx["even_rows"] if r["axis"] == "b"]
        sel = [b[i] for i in sm.smoke_pairs]
        orc = {}
        for cond in sm.conditions():
            mc = self._m_for(cond.name, doc)
            got = mc.oracle(sel, cond, "smoke", sm.even_seeds())
            s = r_records.cond_summary(got, cond, sm, self._z_for(cond.name, doc), [row_key(r) for r in sel],
                                       sm.even_seeds())
            orc[cond.name] = dict(reasons=s["reasons"], edit_edges=s["edit_edges"], csc_sha256=s["csc_sha256"],
                                  edit=cond.edit, kc_median=s["kc_median"], saturation=s["saturation"],
                                  wall_s=mc.last_wall_s, jobs=mc.last_jobs, z={k: list(v) for k, v in mc.z.items()})
        detail = dict(p=p, oracle=orc, pairs=[row_key(r) for r in sel], p_wall_s=p_wall,
                      seeds=dict(p=list(sm.p.seeds), oracle=sm.even_seeds()))
        v_store.write_json(sp.smoke_detail, detail, self.plist)
        body = dict(detail, problems=self._smoke_problems(p, orc, doc), cost=self._cost(p_wall, orc, sm),
                    detail_path=sp.smoke_detail)
        self._write("smoke", body)
        return body

    def _smoke_problems(self, p, orc, doc) -> list:
        sp = self.spec
        nl, nc, ne = sp.cond_names
        repro_sha = doc["reuse"]["records"]["repro_csc_sha256_none"]
        lever_sha = doc["path"]["sides"]["lever"]["csc_sha256"]
        bad = []
        if lever_sha != [sp.sha_combined]:
            bad.append(f"block path's L_V CSC {lever_sha} is not {sp.sha_combined}")
        if p["L"]["edit_edges"] != [sp.lever_edges] or p["L"]["csc_sha256"] != lever_sha:
            bad.append(f"P arms L: edges {p['L']['edit_edges']} on CSC {p['L']['csc_sha256']}, declared "
                       f"{sp.lever_edges} on {lever_sha}")
        if p["C"]["edit_edges"] != [0] or p["C"]["csc_sha256"] != [repro_sha]:
            bad.append(f"P arms C: edges {p['C']['edit_edges']} on CSC {p['C']['csc_sha256']}, declared none on R's "
                       f"repro")
        for n in ("L", "C"):
            if p[n]["outcome"] == P_INVALID:
                bad.append(f"P arms {n}: INVALID {p[n]['reasons'][:2]}")
        if orc[nl]["edit"] != sp.lever_edit or orc[nl]["edit_edges"] != [sp.lever_edges]:
            bad.append(f"L: edit {orc[nl]['edit']} / edges {orc[nl]['edit_edges']}")
        if [orc[nl]["csc_sha256"]] != lever_sha:
            bad.append(f"L: CSC {orc[nl]['csc_sha256']} is not block path's L_V {lever_sha}")
        for n in (nc, ne):
            if orc[n]["edit_edges"] != [0]:
                bad.append(f"{n}: edges {orc[n]['edit_edges']}, declared none")
        if orc[nl]["csc_sha256"] == orc[nc]["csc_sha256"]:
            bad.append("L and C ran on the same CSC weights")
        if not (orc[nc]["csc_sha256"] == orc[ne]["csc_sha256"] == repro_sha):
            bad.append(f"C / E0 CSC {orc[nc]['csc_sha256']} / {orc[ne]['csc_sha256']} is not R's repro {repro_sha}")
        if _ztuple(orc[nl]["z"]) != self._z_v(doc):
            bad.append(f"L's oracle ran on z {orc[nl]['z']}, not block z's z_V")
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
        per = len(sp.p.directions) * len(sp.p.arms) * 2
        p_h = p_wall / _rounds(per * len(sm.p.seeds), sm.workers) * _rounds(per * len(sp.p.seeds), sp.workers) / 3600
        o_round = max(v["wall_s"] for v in orc.values()) / _rounds(len(sm.smoke_pairs), sm.workers)
        n_sm = sum(len(v) for v in sm.even_seeds().values())
        jm = sum(len(v) for v in sp.judge_seeds().values()) / n_sm
        return dict(gate2_h=p_h, jm_per_condition_h=o_round * jm * _rounds(sp.n_b + sp.n_a, sp.workers) / 3600,
                    note="rough: smoke wall time x worker rounds x seed ratio")

    # ---- the operating characteristics (V.6) — before gate ② ----------------------------------------------------
    def stage_oc(self) -> dict:
        """S.6's independent model (n_b 21 · n_a 43, V's notes) and T's cluster model on V's set clusters (block set),
        written to results/v/oc_cluster.json; the block records its file sha and table sha (V.9.5 P2-9)."""
        doc = self._require("oc")
        sp, s = self.spec, doc["set"]["set"]
        cl = v_rules.oc_cluster(s["clusters_b"], s["clusters_a"])
        p = v_store.write_json(sp.oc_cluster_out, cl, self.plist)
        body = dict(independent=v_rules.oc(sp), cluster={k: v for k, v in cl.items() if k != "rows"},
                    cluster_values=v_rules.cluster_values(cl), cluster_path=sp.oc_cluster_out,
                    cluster_file_sha256=sha256_file(p))
        self._write("oc", body)
        return body

    def stage_gate2_oc(self) -> dict:
        """S.9.7's gate ② OC on block h4's z (V.6), from P's committed block, before gate ②."""
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

    # ---- gate ② (V.3 10, V.9.2) ----------------------------------------------------------------------------------
    def stage_gate2(self, rerun: bool = False) -> dict:
        """S's gate ② on 25_400_000+i with P_L(V) and P_C, both labelled on block h4's z; then V.9.2's z_V ratio.
        rerun: only an INVALID gate2 block, only once, only with a pipeline key other than the INVALID run's."""
        prior = None
        if rerun:
            doc = self._require("gate2", allow_own=True)
            blk = doc.get("gate2")
            if GATE2_INVALID in doc:
                refuse("gate2 was rerun once already (V.3 10, R.5)")
            if not blk or blk.get("outcome") != v_rules.INVALID:
                refuse("--rerun-after-invalid needs an INVALID gate2 block (V.3 10)")
            if blk.get("pipeline_key") == self.pipeline_key:
                refuse("the pipeline key equals the INVALID run's: fix the code first (V.3 10)")
            prior = blk
        else:
            doc = self._require("gate2")
        sp, z = self.spec, self.ctx["z"]
        z_v, z_h4 = self._z_v(doc), _ztuple(z)
        m = self.m_h4
        c1, st = self.ctx["p_inputs"](sp.p, False)
        items = p_items(sp.p, st, sp.lever_edit) + p_items(sp.p_c, st, sp.no_edit)
        rows = m.arms(items, self.ctx["readout"], sp.p.o.n.h3.punish_type, "gate2", sp.p.o.n)
        rows_l = [r for r in rows if r["edit"] == sp.lever_edit]
        rows_c = [r for r in rows if r["edit"] == sp.no_edit]
        res_l = p_judge(rows_l, z, c1["c1"], sp.p)
        res_c = p_judge(rows_c, z, c1["c1"], sp.p_c)
        ratio, ratio_zv, why = None, None, []
        if res_l.get("outcome") != P_INVALID and res_c.get("outcome") != P_INVALID:
            try:
                ratio = t_records.ratio(rows_l, rows_c, z, sp)
                ratio_zv = v_records.ratio_two_z(rows_l, rows_c, z_v, z_h4, sp)
            except ValueError as e:
                why.append(str(e))
        dec = v_rules.gate2(res_l, res_c, ratio, ratio_zv, sp) if not why else dict(outcome=v_rules.INVALID,
                                                                                    reasons=why)
        edges_l = sorted({int(r["r"]["edit_edges"]) for r in rows_l})
        edges_c = sorted({int(r["r"]["edit_edges"]) for r in rows_c})
        if (edges_l != [sp.lever_edges] or edges_c != [0]) and dec["outcome"] != v_rules.INVALID:
            dec = dict(outcome=v_rules.INVALID, reasons=[f"edges L {edges_l} / C {edges_c}, declared "
                                                         f"{sp.lever_edges} / 0"])
        zl = (None if dec["outcome"] == v_rules.INVALID
              else t_records.p_zlever(rows_l, z_v, c1["c1"], res_c, sp))
        body = dict(dec, ratio=ratio, ratio_z_V=ratio_zv, p_judgement_L=res_l, p_judgement_C=res_c, p_L_on_z_V=zl,
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
        v_store.write_json(self.summary_path, doc, self.plist)
        return block
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_v_runner.py tests/brain/test_v_gate2.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/v_runner.py tests/brain/test_v_gate2.py
git commit -m "feat(v): v_runner (2) — smoke (L_V on z_V, CSC of block path), OCs (cluster model on block set's clusters -> results/v/oc_cluster.json), gate 2 on h4 z then z_V ratio"
```

---

### Task 6: `v_runner` (3) and `scripts/run_v.py` — the marker, jm, seal, judge once with one re-generation, recovery; the CLI; mutation, key and AST tests

**Files:**
- Modify: `flymon/brain/v_runner.py` (append methods to class `Runner`)
- Create: `scripts/run_v.py`
- Test: `tests/brain/test_v_judge.py`

**Interfaces:**
- Consumes: Tasks 4–5; `r_records` (`raw_check`, `judge_inputs`, `preread_validity`, `pair_stats`, `stats_ok`, `_short`, `compare`, `JUDGE_BLOCK`), `t_records` (`clusters`, `alpha_fixed`), `v_store` (`load_manifest`, `archive_copy`), `fly_pool.FlyPool`, `u_measure` (`UPool`, `UZMeasurer`, `u_measure_key`), `t_measure.t_measure_key`, `r_measure` (`RMeasurer`, `R_MEASURE_FILES`).
- Produces: `Runner._judge_chain`, `_judgement_rows`, `stage_measurement_started`, `stage_jm(name)`, `_raws`, `stage_seal`, `_read`, `_z_h4_reproduced`, `_check_pins`, `stage_judge`, `_require_after_judge`, `stage_recompute(note)`, `stage_invalid_run(note)`; `scripts/run_v.py` with `STAGES`, `POOL_STAGES`, `GATE_STAGES`, `QUIET`, `check_args`, `make_measure(stage, ucode, pool, ctx)`, `exit_code(stage, out)`, `main(argv)`.

- [ ] **Step 1: Write the failing test**

```python
"""The judgement end of the V chain (V.3 11-13, V.4, V.5, V.6, V.8, V.9.4, V.9.5 P2-8) and its mutation tests: the
measurement_started marker comes before any judgement job; jm:L runs L_V on z_V and jm:C / jm:E0 on block h4's z, every
block carries no pair statistic and is written only when all 64 pairs are back; the reuse condition and the T / V
measurement keys are re-checked before the marker, the judgement measurement, the seal and the judge (exit 7); the seal
re-checks every raw file and its stored inputs, archives 192 raw files and pins the decision code with block z, set and
kc_input; the judge reads once — with the one re-generation after an interruption between the mark and the block — and
fills V.7's sentence with V.9.4's cluster values; the CLI's refusals and exit codes; V defines no job and loads no code
by path; the judgement set is named only by the KC band and the judgement stages."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import u_measure as UM
from flymon.brain import v_store
from flymon.brain import v_runner as VR
from flymon.brain.r_measure import RMeasurer
from flymon.brain.v_spec import LEVER_V
from flymon.brain.v_store import VCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.v_world import CODE, SPEC, TCODE, UCODE, ZV, World, ZScripted, doc, through

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_jm(w, m):
    r = through(w, m, zm=ZScripted())
    r.stage_measurement_started()
    return r


def _to_seal(w, m):
    r = _to_jm(w, m)
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def _sealed(w, plan=None):
    r = _to_seal(w, w.scripted(plan=plan or w.pass_plan(n=14, c=8, f_a=3)))
    assert r.stage_seal()["status"] == "SEALED"
    return r


def test_full_chain_reads_selected(w):
    m = w.scripted(plan=w.pass_plan(n=14, c=8, f_a=3))
    r = through(w, m, zm=ZScripted())
    assert w.judgement_calls == 0
    ms = r.stage_measurement_started()
    assert ms["set"]["digest_keys"] == "k" * 64 and ms["z_V"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert ms["seeds"] == SPEC.judge_seeds() and set(ms["decision_pins"]) == {"z_sha256", "set_sha256",
                                                                               "kc_input_sha256"}
    assert not [c for c in m.calls if c[0] == "oracle" and c[1] == "judge"]
    for n in SPEC.cond_names:
        r.stage_jm(n)
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 64 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 64 and jm["edit"] == LEVER_V
    assert jm["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and doc()["jm:C"]["z"] == {"A": [10.0, 9.0],
                                                                                     "P": [26.0, 19.0]}
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm
    stored = json.loads(Path(jm["manifest"][0]["cache_file"]).read_text())["inputs"]
    assert stored["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and stored["edit"] == LEVER_V
    assert stored["act_seeds"] == SPEC.judge_seeds()["act"] == list(range(24_600_000, 24_600_008))
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 192 and seal["set"]["n_a"] == 43
    assert seal["z"]["z_V"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert set(seal["decision"]) >= {"key", "z_sha256", "set_sha256", "kc_input_sha256"}
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive)) and seal["decision"]["key"] == VR.decision_key()["key"]
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    rho, cv = doc()["gate2"]["ratio"], doc()["oc"]["cluster_values"]
    assert (f"(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 "
            f"{rho['r1']['ratio']:.3f}·ℓ_r2 {rho['r2']['ratio']:.3f} ≥ 0.5(약화 정도 기록), 짝수 13/21, 판정 시드 "
            f"24_600_xxx)") in out["sentence"]
    assert (f"V 세트 군집 상관 모형에서 G_fail_S null {cv['cl_null']}, 오선택(harm) {cv['cl_harm1']} / "
            f"{cv['cl_harm2']}.") in out["sentence"]
    assert "생성원 턴 0–296" in out["sentence"] and "(b) 21쌍은 1개 상대 타입 조합 쌍에" in out["sentence"]
    rec = out["records"]
    assert rec["clusters"]["GROUND* 대 NORMAL"]["L"] == dict(n=21, testable=14, reward_pass=14, punish_pass=21)
    assert rec["alpha_fixed"]["n"] == 14 and rec["z"]["z_V"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert rec["z"]["z_h4_reproduced"] is True and set(rec["path"]["mech"]) == {"none", "apl", "lever"}
    assert set(rec["gate2"]) == {"ratio_h4", "ratio_z_V"} and "kc_record" in rec["kc_input"]
    assert out["resumed_after_mark"] is False and out["k_even"] == 13
    assert out["oc_cluster_sha256"] == doc()["oc"]["cluster"]["sha256"] and set(doc()) == set(VR.ORDER)
    assert Path(VR.JUDGE_MARKER).exists() and Path(VR.DONE_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_net_drop_3_on_a_reads_the_punish_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", LEVER_V)].update({k: (False, False, False) for k in ja[30:33]})
    out = _sealed(w, plan).stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["a"]["net_drop"] == 3
    assert out["sentence"].endswith(" (조합 지렛대)")


def test_b_tb_closes_the_combined_lever_claim_only(w):
    out = _sealed(w, w.pass_plan(n=8, c=8, f_a=3)).stage_judge()
    assert out["band"] == "B_Tb" and out["sentence"].endswith(
        "→ 넓힌 풀·엔진별 z에서의 이 조합 지렛대 주장을 닫는다.") and "생성원 턴 0–296" in out["sentence"]


def test_b_nc_names_the_combined_lever(w):
    out = _sealed(w, w.pass_plan(n=10, c=8, f_a=3)).stage_judge()
    assert out["band"] == "B_결론없음" and out["sentence"].endswith("(10/21 대 8/21, F_a 3/43, 조합 지렛대)")


def test_jm_needs_the_marker_and_writes_nothing_until_complete(w):
    m = w.scripted(plan=w.pass_plan())
    r = through(w, m)
    with pytest.raises(SystemExit):
        r.stage_jm("L")                                                  # no measurement_started block
    r.stage_measurement_started()
    m.st["fail_once"] = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 64


@pytest.mark.parametrize("where", ["measurement_started", "jm", "seal", "judge"])
@pytest.mark.parametrize("what", ["reuse", "tkey", "ukey", "no_ukey"])
def test_keys_are_rechecked_before_the_marker_measurement_seal_and_judge(w, where, what):
    m = w.scripted(plan=w.pass_plan())
    r = through(w, m)
    if where != "measurement_started":
        r.stage_measurement_started()
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    if what == "reuse":
        w.u["path"]["detail_sha256"] = "x" * 64
    elif what == "tkey":
        r = w.runner(m, tcode={"key": "x" * 64})
    elif what == "ukey":
        r = w.runner(m, ucode={"key": "v" * 64})
    else:
        r = VR.Runner(m.at, None, w.ctx, SPEC, code=CODE, tcode=TCODE, ucode=None, pipeline={"key": "p" * 64},
                      archive_root=w.archive)
    before = doc()
    with pytest.raises(SystemExit) as e:
        {"measurement_started": r.stage_measurement_started, "jm": lambda: r.stage_jm("L"), "seal": r.stage_seal,
         "judge": r.stage_judge}[where]()
    assert e.value.code == VR.EXIT_KEY and doc() == before
    assert not Path(VR.JUDGE_MARKER).exists()


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    r = _to_seal(w, w.scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)}))
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_l_raw_on_h4_z_fails_the_preread_validity(w):
    m = w.scripted(plan=w.pass_plan())
    r = _to_jm(w, m)
    real_m_for = r._m_for
    r._m_for = lambda name, doc=None: m.at(Z) if name == "L" else real_m_for(name, doc)
    r.stage_jm("L")
    r._m_for = real_m_for
    r.stage_jm("C")
    r.stage_jm("E0")
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any(x.startswith("L: ") and "stored inputs" in x for x in seal["reasons"])


def test_mutation_l_raw_of_the_apl_only_edit_fails_the_preread_validity(w):
    m = w.scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edit=UM.u_edit(0.0))})
    seal = _to_seal(w, m).stage_seal()
    assert seal["status"] == "NOT_READ" and any("ran edit u_apl_mbon05_x0.0" in x for x in seal["reasons"])


def test_mutation_a_changed_kc_input_block_refuses_the_judgement_measurement(w):
    r = _to_jm(w, w.scripted(plan=w.pass_plan()))
    d = doc()
    d["kc_input"]["record"]["lever"]["per_odour"]["K0|T"] = 0.2           # block set no longer regenerates
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = _to_jm(w, w.scripted(plan=w.pass_plan()))
    d = doc()
    del d["even"]
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
    assert "judge" not in doc() and not Path(VR.JUDGE_MARKER).exists()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(VR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(VR.JUDGE_MARKER).exists()


def _edit_summary(fn):
    d = doc()
    fn(d)
    Path(SPEC.summary).write_text(json.dumps(d))


@pytest.mark.parametrize("block", ["z", "set", "kc_input"])
def test_judge_refuses_a_pinned_block_changed_after_the_seal(w, block):
    """V.3 12: the seal's decision hash covers block z, set and kc_input; judge and recompute refuse on change."""
    r = _sealed(w)
    _edit_summary(lambda d: d[block].__setitem__("note_added_later", 1))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(VR.JUDGE_MARKER).exists()


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(VR.v_rules, "read_band", lambda *a, **k: dict(band=VR.v_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == VR.v_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(VR.JUDGE_MARKER).exists()


class _Kill(Exception):
    pass


def _kill_judge_write(monkeypatch, seen):
    orig = v_store.write_summary_block

    def boom(path, block, obj, plist):
        if block == "judge":
            seen.append(obj)
            raise _Kill()
        return orig(path, block, obj, plist)
    monkeypatch.setattr(VR.v_store, "write_summary_block", boom)
    return orig


def test_resume_after_mark_once(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    mark = json.loads(Path(VR.JUDGE_MARKER).read_text())
    assert "judge" not in doc() and not Path(VR.DONE_MARKER).exists() and mark["u_measure_key"] == "u" * 64
    monkeypatch.setattr(VR.v_store, "write_summary_block", orig)
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


def test_recovery_after_reading(w, monkeypatch):
    r = _sealed(w)
    with pytest.raises(SystemExit):
        r.stage_recompute("before judge")
    r.stage_judge()
    with pytest.raises(SystemExit):
        r.stage_recompute("")
    e = r.stage_recompute("records code fix (test)")
    assert e["band"] == "SELECTED" and e["differs_from_judge"] is False and e["decision_changed_since_seal"] is False
    monkeypatch.setattr(VR, "decision_key", lambda: dict(key="d" * 64, files={}))
    assert r.stage_recompute("decision code fix (test)")["decision_changed_since_seal"] is True
    assert r.stage_invalid_run("measurement defect (test)")["status"] == "INVALID_RUN"
    with pytest.raises(SystemExit):
        r.stage_invalid_run("again")
    with pytest.raises(SystemExit):
        r.stage_recompute("after invalid")


class OraclePool:
    """u_oracle_job-shaped outputs for real RMeasurers over UPool + VCache; the pair key travels in the odour name
    ("G|<key>" E-grid, "E|<key>" E0); the z and edit each job received are recorded."""

    def __init__(self, plan):
        self.n_workers, self.plan, self.jobs, self.z, self.edits = 4, plan, 0, {}, set()

    def run_jobs(self, fn, kws):
        assert fn is UM.u_oracle_job
        out = []
        for kw in kws:
            self.jobs += 1
            lever = UM.is_u_edit(kw["edit"])
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            self.z[cond] = kw["z"]
            self.edits.add(kw["edit"])
            flags = self.plan.get(("judge", kw["edit"]), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha="sha-V" if lever else "sha-C", edges=2 if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_real_cache_round_trip_seals_and_reads(w):
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through(w, w.scripted(plan=plan)).stage_measurement_started()
    pool, cache, ms = OraclePool(plan), VCache(SPEC.cache_dir, UCODE), {}

    def measure(z):
        k = json.dumps({a: list(b) for a, b in sorted(z.items())})
        return ms.setdefault(k, RMeasurer(UM.UPool(pool), cache, SPEC, w.ctx["params"], READOUT, z, TYPES, 100))
    real = VR.Runner(measure, None, w.ctx, SPEC, code=CODE, tcode=TCODE, ucode=UCODE, pipeline={"key": "p" * 64},
                     archive_root=w.archive)
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 64
    assert pool.jobs == 192 and pool.edits == {LEVER_V, "none"}
    assert pool.z["L"] == {k: list(v) for k, v in ZV.items()} and pool.z["C"] == pool.z["E0"] == {
        k: list(v) for k, v in Z.items()}
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds", "v_set", "set_odours", "set_e0_odours", "_checked", "_set_call"}
ALLOWED = {"v_spec.py": {None, "judge_seeds"},
           "v_pairs.py": {"_checked", "judgement_rows", "set_odours", "set_e0_odours", "v_set"},
           "v_runner.py": {"build_ctx", "_set_call", "_judgement_rows", "stage_set", "stage_kc_band",
                           "stage_measurement_started", "stage_jm", "stage_seal", "_read", "_cost"}}


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
    files = sorted((ROOT / "flymon/brain").glob("v_*.py")) + [ROOT / "scripts/run_v.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def test_v_defines_no_job_or_measurer_and_loads_no_code_by_path():
    """V.9.5 P1-3: V measures only through U's measurement file (unchanged); no V file defines a job, rig, measurer or
    cache copy (only VCache, overriding put) and none loads code by path."""
    copies = {"u_kc_activity_job", "u_arm_job", "u_oracle_job", "u_ref_job", "u_rest_job", "u_rig", "u_engine",
              "apply_u_edit", "UPool", "UZMeasurer", "kc_activity_job", "r_arm_job", "r_oracle_job", "RMeasurer",
              "RCache", "ECache", "q_oracle_job", "arm_job", "q_rig", "reference_job", "rest_job", "engine_for",
              "apply_q_edit", "apply_csc_edit", "t_ref_job", "t_rest_job", "z_engine", "ZMeasurer"}
    for p in sorted((ROOT / "flymon/brain").glob("v_*.py")) + [ROOT / "scripts/run_v.py"]:
        tree = ast.parse(p.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not defs & copies, (p.name, defs & copies)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not names & {"importlib", "exec", "spec_from_file_location", "runpy"}, p.name
    assert not list((ROOT / "flymon/brain").glob("v_measure*.py"))


def _cli():
    spec = importlib.util.spec_from_file_location("run_v_for_test", ROOT / "scripts/run_v.py")
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
    for st in ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "gate2"):
        assert cli.exit_code(st, {"outcome": "PASS"}) == 0 and cli.exit_code(st, {"outcome": "INVALID"}) == 5
    for st, o in (("reuse", "STOP_REUSE"), ("path", "STOP_V_PATH_REPRO"), ("kc_input", "STOP_V_PATH_REPRO"),
                  ("set", "STOP_SET_SHORT"), ("z", "STOP_Z_DEGENERATE"), ("kc_band", "STOP_STRENGTH_LEVER"),
                  ("kc_band", "STOP_V_PATH_REPRO"), ("even", "STOP_EVEN_REPRO"), ("even", "STOP_EVEN_PUNISH"),
                  ("even", "STOP_EVEN_LOW_LEVER"), ("gate2", "STOP_PUNISH_WEAKENED")):
        assert cli.exit_code(st, {"outcome": o}) == 3, (st, o)
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    for st in ("oc", "gate2_oc", "measurement_started"):
        assert cli.exit_code(st, {}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "INVALID"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert set(cli.POOL_STAGES) == {"path", "kc_input", "kc_band", "even", "smoke", "gate2", "jm"}
    assert set(cli.STAGES) == {s.split(":")[0] for s in VR.ORDER} | {"recompute", "invalid_run"}
    assert set(cli.GATE_STAGES) == set(VR.GATES)
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_v.py").read_text()


def test_cli_measure_factory_builds_rmeasurers_over_upool_and_vcache_only(tmp_path, monkeypatch):
    from flymon.brain.config import Params
    from flymon.brain.v_spec import SPEC as V_SPEC
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    ctx = dict(params=Params(), readout=READOUT, types=TYPES, n_kc=100)
    measure = cli.make_measure("even", UCODE, None, ctx)
    m_h4, m_v = measure(Z), measure(ZV)
    assert measure({k: list(v) for k, v in Z.items()}) is m_h4 and m_v is not m_h4 and m_v.cache is m_h4.cache
    for m in (m_h4, m_v):
        assert isinstance(m, RMeasurer) and type(m.cache) is VCache and m.spec is V_SPEC
        assert isinstance(m.pool, UM.UPool) and m.cache.root == V_SPEC.cache_dir and m.cache.code == UCODE
    m_v.cache.put("r_oracle", {"act_seeds": [500], "edit": LEVER_V}, {"x": 1}, [Params()])
    written = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.startswith(v_store.ALLOWED_DIR + "cache/r_oracle/") for p in written), written
    sm = cli.make_measure("smoke", UCODE, None, ctx)(Z)
    assert sm.cache.root == V_SPEC.smoke_cache_dir and sm.cache.root.startswith(v_store.ALLOWED_DIR)


def test_hashed_and_decision_files():
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import T_MEASURE_FILES
    from flymon.brain.u_measure import U_MEASURE_FILES
    from flymon.brain.u_runner import U_HASHED_FILES
    assert set(VR.V_PIPELINE_FILES) == {f"flymon/brain/v_{n}.py" for n in
                                        ("spec", "pairs", "store", "records", "rules", "runner")} | {"scripts/run_v.py"}
    assert set(U_HASHED_FILES) | set(VR.V_PIPELINE_FILES) <= set(VR.V_HASHED_FILES)
    assert "results/summary/u_lever.json" in VR.V_HASHED_FILES
    assert set(VR.DECISION_FILES) == (set(VR.V_HASHED_FILES) - set(R_MEASURE_FILES) - set(T_MEASURE_FILES)
                                      - set(U_MEASURE_FILES))
    assert [f for f in VR.V_HASHED_FILES if not (ROOT / f).exists()] == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/brain/test_v_judge.py -q`
Expected: FAIL (`AttributeError: 'Runner' object has no attribute 'stage_measurement_started'`).

- [ ] **Step 3: Append the judgement methods to class `Runner`** (at the end of `flymon/brain/v_runner.py`, indented as methods)

```python
    # ---- the judgement (V.3 11-13, V.5, V.9.5 P2-8) ----------------------------------------------------------------
    def _judge_chain(self, doc: dict, stage: str) -> None:
        """The judgement runs once, from clean trees, on the shared key every earlier block ran on and on the T and V
        measurement keys every earlier block carries (V.8: re-checked right before the judgement measurement and the
        judge — exit 7)."""
        if self.spec.smoke:
            refuse("smoke never measures the judgement set (V.3 8)")
        if not self.code_key:
            refuse("no code key given; the judgement must run on the code the gates ran on")
        if not self.u_measure_key or not self.t_measure_key:
            refuse(f"the V / T measurement key cannot be verified (V {self.u_measure_key}, T {self.t_measure_key}) "
                   f"(V.8)", EXIT_KEY)
        if self.u_measure_key != self.spec.u_measure_key_u:
            refuse(f"the V measurement key {self.u_measure_key} is not U's {self.spec.u_measure_key_u} (V.9.5)",
                   EXIT_KEY)
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("u_measure_key") != self.u_measure_key or doc[b].get("t_measure_key") != self.t_measure_key:
                refuse(f"block {b}'s V / T measurement key {doc[b].get('u_measure_key')} / {doc[b].get('t_measure_key')}"
                       f" is not the current {self.u_measure_key} / {self.t_measure_key} (V.8)", EXIT_KEY)
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the set is used once (V.5)")
        for b in ORDER[:ORDER.index(stage)]:
            if doc[b].get("code_key") != self.code_key:
                refuse(f"block {b}'s code key {doc[b].get('code_key')} is not the current {self.code_key}")
            d = (doc[b].get("git") or {}).get("dirty_hashed")
            if d is None or d:
                refuse(f"block {b} was written with dirty hashed files (or no git record): {d}")

    def _judgement_rows(self, doc: dict) -> list:
        return self._set_call("judgement_rows", doc)

    def stage_measurement_started(self) -> dict:
        """V.9.5 P2-8: the set-use marker, committed before the first judgement job (no pool): the set block's digests,
        the judgement seeds and z_V. From here on the V set is used (V.5)."""
        doc = self._require("measurement_started")
        self._judge_chain(doc, "measurement_started")
        sp, s = self.spec, doc["set"]["set"]
        body = dict(set={k: s[k] for k in ("digest_e0_b", "digest_e0_a", "digest_keys", "last_turn", "n_b", "n_a")},
                    seeds=sp.judge_seeds(), z_V=doc["z"]["z_V"], decision_pins=decision_pins(doc),
                    note="V.5 / V.9.5 P2-8: 판정 측정 첫 잡 전 표식 — 이 블록 뒤 V 세트는 사용된 것이다.")
        self._write("measurement_started", body)
        return body

    def stage_jm(self, name: str) -> dict:
        """One judgement condition on the 64 pairs (resumable), L_V on z_V and C / E0 on block h4's z; the block holds
        raw_check only."""
        sp = self.spec
        if name not in sp.cond_names:
            refuse(f"unknown condition {name}; V has {sp.cond_names}")
        stage = f"jm:{name}"
        doc = self._require(stage)
        self._judge_chain(doc, stage)
        rows = self._judgement_rows(doc)
        seeds, cond, m = sp.judge_seeds(), sp.cond(name), self._m_for(name, doc)
        got = m.oracle(rows, cond, r_records.JUDGE_BLOCK, seeds)
        rc = r_records.raw_check(got, cond, [row_key(r) for r in rows], seeds)
        man = [dict(x, sha256=sha256_file(x["cache_file"])) for x in rc["manifest"]]
        body = dict(rc, manifest=man, seeds=seeds, condition=name, edit=cond.edit,
                    z={k: list(v) for k, v in m.z.items()}, wall_s=m.last_wall_s, jobs=m.last_jobs)
        self._write(stage, body)
        return body

    def _raws(self, doc: dict) -> tuple:
        raws, bad = {}, []
        for n in self.spec.cond_names:
            got, b = v_store.load_manifest(doc[f"jm:{n}"]["manifest"])
            raws[n] = got
            bad += [f"{n}: {x}" for x in b]
        return raws, bad

    def stage_seal(self) -> dict:
        """R.9.7 as is (V.3 12): pre-read validity (every raw entry's stored inputs = its condition's measurer's, so
        L's carry L_V and z_V, C's / E0's "none" and h4 z), the manifest (192), the archive copy, the decision code hash
        with block z, block set and block kc_input pinned."""
        doc = self._require("seal")
        self._judge_chain(doc, "seal")
        sp = self.spec
        rows = self._judgement_rows(doc)
        keys = [row_key(r) for r in rows]
        raws, bad = self._raws(doc)
        want = {n: r_records.judge_inputs(self._m_for(n, doc), rows, sp)[n] for n in sp.cond_names}
        v = r_records.preread_validity({n: doc[f"jm:{n}"] for n in sp.cond_names}, raws, sp,
                                       _ztuple(self.ctx["z"]), keys, self.code_key,
                                       doc["reuse"]["records"]["repro_csc_sha256_none"], want)
        nl = sp.cond_names[0]
        seeds = sp.judge_seeds()
        n_rep, n_act = len(seeds["report"]), len(seeds["act"])
        zv = self._z_v(doc)
        und = [g["key"] for g in raws[nl] if not r_records._short(g["result"], n_rep, n_act) and not r_records.stats_ok(
            r_records.pair_stats(g["result"]["report"], zv, sp.testable_min))]
        reasons = bad + v["reasons"] + ([f"{nl}: {len(und)} pair(s) with an undefined d′ on z_V"] if und else [])
        status = v_rules.INVALID if v["invalid"] else (v_rules.NOT_READ if reasons else v_rules.SEALED)
        manifest = [dict(x, condition=n) for n in sp.cond_names for x in doc[f"jm:{n}"]["manifest"]]
        s = doc["set"]["set"]
        body = dict(status=status, reasons=reasons, invalid=v["invalid"], checks=v["checks"], manifest=manifest,
                    n_files=len(manifest), archive=None, decision=dict(decision_key(), **decision_pins(doc)),
                    seeds=seeds, z=dict(z_V=doc["z"]["z_V"], z_h4={k: list(x) for k, x in _ztuple(self.ctx["z"]).items()}),
                    set={k: s[k] for k in ("digest_e0_b", "digest_e0_a", "digest_keys", "last_turn", "n_b", "n_a")})
        if status == v_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=v_store.archive_copy([x["cache_file"] for x in manifest], dest,
                                                                              self.archive_root))
        self._write("seal", body)
        return body

    def _read(self, doc: dict, mark=None) -> dict:
        """The bands and records from the sealed raw files (sha re-checked): L_V read on z_V, C and E0 on block h4's z
        (V.4); R.3's order with G_fail_S; V.7's sentence (V.9.4's cluster values in SELECTED); V.6's records."""
        sp = self.spec
        rows = self._judgement_rows(doc)
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
        gf = v_rules.g_fail_s(L["pairs"], C["pairs"], sp)
        rb = v_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                               aL["naive_a"], sp)
        if rb["band"] == v_rules.NOT_READ:
            return dict(status=v_rules.NOT_READ, band=rb["band"], reason=rb["reason"],
                        reasons=["the (b) / (a) pair counts differ from the declared counts (V.4 line 1)"])
        ax, ratio = gf["axes"], doc["gate2"]["ratio"]
        names = list(sp.p.directions)
        k_even = doc["even"]["record"]["testable_b"]
        st = doc["set"]["set"]
        fields = dict(n=aL["testable_b"], c=aC["testable_b"], f_a=aL["F_a"], naive_a=aL["naive_a"], T=st["last_turn"],
                      k_even=k_even, k_cl=len(st["clusters_b"]), pb_L=ax["b"]["pun_L"], pb_C=ax["b"]["pun_C"],
                      pa_L=ax["a"]["pun_L"], pa_C=ax["a"]["pun_C"], d_b=ax["b"]["net_drop"], d_a=ax["a"]["net_drop"],
                      rho1=f"{ratio[names[0]]['ratio']:.3f}", rho2=f"{ratio[names[-1]]['ratio']:.3f}",
                      reason=rb["reason"], **doc["oc"]["cluster_values"])
        out = dict(band=rb["band"], reason=rb["reason"], n=aL["testable_b"], c=aC["testable_b"], F_a=aL["F_a"],
                   naive_a=aL["naive_a"], n_b=aL["n_b"], n_a=aL["n_a"], g_fail=gf)
        if rb["band"] == v_rules.B_FA:
            out["f_a_possible"] = rb["f_a_possible"]
        labels = self.ctx["clusters"](rows)
        records = dict(r_records.compare(L, C, E0, sp),
                       clusters=t_records.clusters({n: s[n]["pairs"] for n in sp.cond_names}, labels),
                       alpha_fixed=t_records.alpha_fixed(raws[sp.cond_names[0]], C, keys, seeds,
                                                         _ztuple(self.ctx["z"]), sp),
                       z=dict(z_V=doc["z"]["z_V"], record=doc["z"]["record"],
                              z_h4_reproduced=self._z_h4_reproduced(doc)),
                       path=dict(records=doc["path"]["records"],
                                 mech={k: v["mech"] for k, v in doc["path"]["sides"].items()}),
                       kc_input=dict(kc_record=doc["set"]["kc_record"], u_compare=doc["kc_input"]["u_compare"]),
                       gate2=dict(ratio_h4={d: ratio[d]["ratio"] for d in names},
                                  ratio_z_V={d: doc["gate2"]["ratio_z_V"][d]["ratio"] for d in names}))
        return dict(out, status=v_rules.READ, sentence=v_rules.sentence(rb["band"], fields), records=records,
                    pairs={n: s[n]["pairs"] for n in sp.cond_names}, oc_sha256=doc["oc"]["independent"]["sha256"],
                    oc_cluster_sha256=doc["oc"]["cluster"]["sha256"],
                    p_labels=dict(L=doc["gate2"]["label_L"], C=doc["gate2"]["label_C"]),
                    p_ratio={d: ratio[d]["ratio"] for d in names}, k_even=k_even)

    def _z_h4_reproduced(self, doc: dict) -> bool:
        """From the reuse block: it passed and T's unedited z it recorded is block h4's z."""
        rb = doc.get("reuse") or {}
        nz = _dig(rb, ("records", "t_z", "none_z"))
        return bool(rb.get("outcome") == v_rules.PASS and isinstance(nz, dict)
                    and _ztuple(nz) == _ztuple(self.ctx["z"]))

    def _check_pins(self, doc: dict) -> None:
        """V.3 12: the live z, set and kc_input blocks are the ones the seal pinned; refuse otherwise."""
        sd = doc["seal"].get("decision") or {}
        moved = [k for k, v in decision_pins(doc).items() if sd.get(k) is None or sd.get(k) != v]
        if moved:
            refuse(f"the sealed decision record ({', '.join(moved)}) differs from the live blocks: the judgement reads "
                   f"only the z_V, set and KC input it was sealed with (V.3 12)")

    def stage_judge(self) -> dict:
        """Once (V.3 13). The marker is written before the band is computed. S.9.2 (V.5): with the marker and no judge
        block, judge is re-generated once — same sealed raw data (sha re-checked), the sealed decision code, no
        measurement; a second re-generation, a marker from another seal or decision code, or a judge block once written
        (DONE marker) refuses."""
        doc = self._require("judge")
        self._judge_chain(doc, "judge")
        if doc["seal"].get("status") != v_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: V reads only a sealed set (R.9.7)")
        sealed = (doc["seal"].get("decision") or {}).get("key")
        now = decision_key()["key"]
        if sealed != now:
            refuse(f"the decision code hash {now} is not the sealed {sealed}: the judgement reads only under the code "
                   f"it was sealed with (V.5)")
        self._check_pins(doc)
        if Path(DONE_MARKER).exists():
            refuse(f"{DONE_MARKER} exists: a judge block was written once; a discarded block does not reopen the set")
        resumed = None
        if Path(JUDGE_MARKER).exists():
            if Path(REREAD_MARKER).exists():
                refuse(f"{REREAD_MARKER} exists: judge was re-generated once after the mark already (V.5)")
            mk = json.loads(Path(JUDGE_MARKER).read_text())
            if mk.get("seal_written_at") != doc["seal"].get("written_at") or mk.get("decision_key") != sealed:
                refuse(f"{JUDGE_MARKER} belongs to another seal or decision code; no re-generation (V.5)")
            resumed = mk

        def mark():
            v_store.write_json(REREAD_MARKER if resumed else JUDGE_MARKER, dict(
                seal_written_at=doc["seal"].get("written_at"),
                seal_archive=(doc["seal"].get("archive") or {}).get("dir"),
                decision_key=now, code_key=self.code_key, t_measure_key=self.t_measure_key,
                u_measure_key=self.u_measure_key, pipeline_key=self.pipeline_key, read_at=_now()), self.plist)
        out = self._read(doc, mark)
        if out["status"] != v_rules.READ:
            return out                                   # NOT_READ: no block, the marker stays
        out = dict(out, resumed_after_mark=resumed is not None, mark_read_at=(resumed or {}).get("read_at"))
        block = self._write("judge", out)
        v_store.write_json(DONE_MARKER, dict(judge_written_at=block["written_at"]), self.plist)
        return block

    # ---- after reading (V.5 = U.5) ---------------------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (V.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: this set is closed (V.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """V.5 row 2: an analysis or summary defect after reading — the same sealed raw data recomputed."""
        if not note:
            refuse("--note is required (V.5: the correction is recorded)")
        doc = self._require_after_judge("recompute")
        self._check_pins(doc)                            # a decision-key change alone is recorded below, not refused
        out = self._read(doc)
        dk = decision_key()["key"]
        entry = dict(note=note, status=out["status"], band=out["band"], reason=out["reason"], n=out.get("n"),
                     c=out.get("c"), F_a=out.get("F_a"), naive_a=out.get("naive_a"), g_fail=out.get("g_fail"),
                     sentence=out.get("sentence"), decision_key=dk,
                     decision_changed_since_seal=bool(dk != (doc["seal"].get("decision") or {}).get("key")),
                     differs_from_judge=bool(out["band"] != doc["judge"]["band"]
                                             or out.get("sentence") != doc["judge"].get("sentence")),
                     code_key=self.code_key, t_measure_key=self.t_measure_key, u_measure_key=self.u_measure_key,
                     pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())
        v_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """V.5 row 3: a measurement defect after reading — INVALID_RUN; a replacement set is a new declaration."""
        if not note:
            refuse("--note is required (V.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=v_rules.INVALID_RUN, note=note, judge_band=doc["judge"]["band"],
                    rule="V.5: the same set is never run again; a replacement set needs a new declaration (user)")
        self._write("invalid_run", body)
        return body
```

- [ ] **Step 4: Write the CLI**

```python
#!/usr/bin/env python3
"""Spec appendix V (V.9 wins over V.0-V.8): the combined lever L_V (APL->MBON05 removed + MBON05->MBON09 / MBON11 /
MBON01 chain entry cut, 13 CSC edges), each engine variant's own reference-set z for the oracle, a new set from T's
widened-pool generator filtered by both engines' per-odour KC input, one M2 judgement. The controller runs every stage;
commit each block before the next.

    uv run python scripts/run_v.py --stage reuse                     # V.3 1 R / T / U reused (no pool)
    uv run python scripts/run_v.py --stage path                      # V.3 2 three engines + 3 + 3 even pairs, no cache
    uv run python scripts/run_v.py --stage kc_input                  # V.3 3 candidate odours on both engines
    uv run python scripts/run_v.py --stage set                       # V.2 the set, list only (no pool)
    uv run python scripts/run_v.py --stage z                         # V.3 5 z_V from block path (no pool)
    uv run python scripts/run_v.py --stage kc_band                   # V.3 6 112 calibration odours + set re-check
    uv run python scripts/run_v.py --stage even                      # V.3 7 C from R's even raw, L_V on z_V
    uv run python scripts/run_v.py --stage smoke --workers 4         # 24_609_xxx / 25_409_xxx, even (b) 0 and 20 only
    uv run python scripts/run_v.py --stage oc                        # V.6 both operating characteristics (no pool)
    uv run python scripts/run_v.py --stage gate2_oc                  # V.6 gate ② operating characteristic (no pool)
    uv run python scripts/run_v.py --stage gate2                     # ② P_L(V) and P_C on 25_400_000+i (h4 z, z_V ratio)
    uv run python scripts/run_v.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (V.3 10)
    uv run python scripts/run_v.py --stage measurement_started       # V.9.5 P2-8 the set-use marker (no pool)
    uv run python scripts/run_v.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_v.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_v.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_v.py --stage recompute --note "..."    # after judge: analysis defect (V.5)
    uv run python scripts/run_v.py --stage invalid_run --note "..."  # after judge: measurement defect (V.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_V_PATH_REPRO, STOP_SET_SHORT,
STOP_Z_DEGENERATE, STOP_STRENGTH_LEVER, STOP_EVEN_REPRO, STOP_EVEN_PUNISH, STOP_EVEN_LOW_LEVER, gate ②'s three;
recorded, V stops), 5 a gate INVALID (path, kc_input, z, kc_band, even, gate2), 6 seal not SEALED, judge NOT_READ or a
smoke with problems, 7 the reuse condition or a measurement key broke (V stops; no block), 2 a refusal (arguments, cwd,
connectome sha256, chain, uncommitted summary, dirty hashed file, data pins, R's even raw, T's z rows or U's rows
missing, the set not reproducing block set)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "smoke", "oc", "gate2_oc", "gate2",
          "measurement_started", "jm", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("path", "kc_input", "kc_band", "even", "smoke", "gate2", "jm")
GATE_STAGES = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "gate2")
QUIET = ("pairs", "manifest", "records", "record", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive",
         "checks", "g_fail", "set", "sides", "C", "L", "independent", "cluster", "record_calib", "record_set",
         "record_e0", "compare", "kc_record", "odour_ids", "u_compare", "csc_facts", "ratio", "ratio_z_V", "guard")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (V.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def make_measure(stage: str, ucode: dict, pool, ctx: dict):
    """measure(z): one RMeasurer per z, every one with V's spec over one UPool (U's job copies, unchanged) and one
    VCache keyed by the V measurement key (= U's) and rooted at V's cache (results/v/cache, results/v/smoke/cache for
    smoke), so no V entry lands in R's, T's or U's tree (VCache.put writes only through v_store's guard)."""
    from flymon.brain.h3_store import canonical
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.u_measure import UPool
    from flymon.brain.v_spec import SPEC
    from flymon.brain.v_store import VCache
    cache = VCache(SPEC.smoke_cache_dir if stage == "smoke" else SPEC.cache_dir, ucode)
    upool = UPool(pool)
    measurers = {}

    def measure(z):
        k = canonical({t: list(v) for t, v in z.items()})
        if k not in measurers:
            measurers[k] = RMeasurer(upool, cache, SPEC, ctx["params"], ctx["readout"], z, ctx["types"], ctx["n_kc"])
        return measurers[k]
    return measure


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import v_rules as R
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
    from flymon.brain.h3_store import ROOT, code_key, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2

    from flymon.brain import v_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import UZMeasurer, u_measure_key
    from flymon.brain.v_spec import SPEC

    try:
        ctx = v_runner.build_ctx(SPEC, NPZ)
        code, ucode = code_key(NPZ, files=R_MEASURE_FILES), u_measure_key(NPZ)
        workers = a.workers or SPEC.workers
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            measure = make_measure(a.stage, ucode, pool, ctx)
            zm = UZMeasurer(pool, ctx["params"], SPEC.p_type, ctx["rec_types"], SPEC.kc_types)
            r = v_runner.Runner(measure, zm, ctx, SPEC, code=code, tcode=t_measure_key(NPZ), ucode=ucode,
                                pipeline=v_runner.pipeline_key())
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

- [ ] **Step 5: Run the V tests and the full suite**

Run: `uv run pytest tests/brain/test_v_judge.py -q`
Expected: all pass (~10 minutes: each test walks the chain).

Run: `uv run pytest tests/brain/ -q -k "test_v_ or test_p_spec or test_u_ or test_t_"`
Expected: all pass.

Run: `uv run pytest -q`
Expected: the whole suite passes.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/v_runner.py scripts/run_v.py tests/brain/test_v_judge.py
git commit -m "feat(v): v_runner (3) and run_v.py — measurement_started marker, jm (L_V on z_V), seal pinning z/set/kc_input, judge once with one re-generation, recovery; mutation, key and AST tests"
```

---

## Self-review (done while writing)

- **Spec coverage:**

  | Spec item | Where it lives |
  |---|---|
  | V.0 disclosures (lever chosen from U's records, 5th attempt, T-set odours below the band, widened pool, set feasibility, MBON05 / MBON13 higher, σA / σP ratios) | no code; the SELECTED sentence (Task 3), the OC notes (Task 3), the path records (Task 4), the result section V.10 |
  | V.1 L_V, 2 + 7 / 2 / 2 = 13 edges, CSC `2d359b8b…`, MBON05->APL untouched, else INVALID; z rule | Task 1 (`LEVER_V`), Task 2 (`csc_facts`, `csc_reasons`), Task 3 (`path`, `z_v`), Task 4 (`stage_path`, `stage_z`), Task 6 (seal: L edges 2 and stored edit) |
  | V.2 / V.9.1 / V.9.5 P2-10 / P3-12 set, generator, used, filter on both engines, kc_input last, STOP_SET_SHORT, digests before any oracle on the set | Task 1 (`v_pairs`), Task 3 (`set_outcome`), Task 4 (`stage_set`, `_set_call`), Task 6 (re-checks at jm / seal / judge, pin) |
  | V.3 1 reuse (R, T, U with literal keys) | Task 1 (key tests), Task 3 (`reuse`), Task 4 (`stage_reuse`, `_reuse_now`) |
  | V.3 2 / V.9.5 P1-3 path (no cache, three engines, 3 + 3 even pairs, unit tests: weights, edit in the cache key, not-applied mutation) | Task 1 (`test_the_edit_is_in_every_cache_key`), Task 2 (real-connectome facts and mutation), Task 3 (`path`), Task 4 (`stage_path` and its mutation tests) |
  | V.3 3 / V.9.1 / V.9.5 P2-7 KC input on both engines, U bit-for-bit, seed ranges, ORN cap | Task 1 (`candidate_odours`), Task 2 (`kc_input_record`, `kc_u_diffs`), Task 3 (`kc_input`), Task 4 (`stage_kc_input`) |
  | V.3 5 / V.9.5 P3-13 z_V with SD | Task 3 (`z_v`), Task 4 (`stage_z`) |
  | V.3 6 / V.9.1 / V.9.5 P2-11 KC band, re-check, E0 record | Task 3 (`kc_band`), Task 4 (`stage_kc_band`) |
  | V.3 7 / V.9.3 even (c_even 7 first, ① filter, ② testable_b, POOL suffix) | Task 3 (`even`, sentences), Task 4 (`stage_even`) |
  | V.3 8 smoke | Task 5 `stage_smoke` |
  | V.3 9 / V.6 / V.9.4 / V.9.5 P2-9 OCs (independent, V cluster model to `results/v/oc_cluster.json` + sha, notes) | Task 3 (`oc`, `oc_cluster`, `cluster_values`), Task 5 (`stage_oc`, `stage_gate2_oc`) |
  | V.3 10 / V.9.2 gate ② on two scales | Task 2 (`ratio_two_z`), Task 3 (`gate2`), Task 5 (`stage_gate2`) |
  | V.9.5 P2-8 `measurement_started` | Task 6 |
  | V.3 11–13 judgement measurement, seal (pins z / set / kc_input), judge once, recovery (V.5) | Task 6 |
  | V.4 bands, G_fail_S | Task 3 (R's / S's functions, boundary tests), Task 6 (bands in the chain) |
  | V.7 sentences (with V.9.3 / V.9.4) | Task 3 (verbatim tests), Task 6 (B_Tb, B_결론없음, B_처벌가드, SELECTED in the chain) |
  | V.8 files, writes, keys, archive | Tasks 2, 4, 6; Global Constraints |
  | V.9.6 order | Task 4 `ORDER` |

- **Placeholders:** none. Every code step has its full code; the code is the code that ran in the scratch worktree.
- **Type consistency:** the blocks later stages read (`reuse.records.repro_csc_sha256_none` / `t_z.none_z`, `path.sides.{none,apl,lever}` / `records`, `kc_input.record.{none,lever}.per_odour` / `u_compare`, `set.set` / `kc_record`, `z.z_V` / `record`, `even.record.testable_b`, `oc.cluster_values` / `independent.sha256` / `cluster.sha256`, `gate2.ratio` / `ratio_z_V` / `label_L` / `label_C`, `seal.decision.{key,z_sha256,set_sha256,kc_input_sha256}`, `seal.written_at`) are written by the stages that precede them; `Scripted.arms` / `oracle` / `activity` / `inputs` and `ZScripted.reference` / `rest` match `RMeasurer` and `UZMeasurer`.
- **Review Focus:** each item has its test in its owning task (listed in the section).

## Runs (controller)

The controller does every run, from `/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot` (branch `open-fly-brain-connectome`), with Bash `run_in_background: true` and `timeout: 7200000`, after the full suite passes at the final implementation commit.
- Subagents never run these steps.
- When a run finishes, read only its exit code and the last lines of its output.
- No rule or number changes after any run.
- Each block is committed before the next stage; the stage refuses otherwise.
- `results/summary/v_lever.json` is tracked; `results/v/` is git-ignored. `results/summary/{r,s,t,u}_lever.json`, `results/r/`, `results/s/`, `results/t/` and `results/u/` are never touched.
- Commit messages carry no trailers.
- **OPEN items** (section above): ask the user for OPEN 1 before step 10 (gate ②) and for OPEN 2–3 before step 13 (judge), writing the question into `/private/tmp/claude-503/p-latest.md` and stopping; do not run those stages until answered. If an answer changes code, it is a reviewed fix task before the affected stage (the decision code is pinned only at the seal).

**The 2 h background limit:**
- Pool stages and rough costs (V.8): `path` (~10 min: 3 × 192 presentations + 6 oracle pairs, not cached — a killed run restarts and writes nothing until it finishes), `kc_input` (~10–20 min: 173 odours × 8 seeds × 2 engines; resumes per odour), `kc_band` (~10 min: 112 + the set's odours under L_V, the E0 odours unedited), `even` (~0.5 h; resumes per pair), `smoke`, `gate2` (~0.3 h; resumes per seed), each `jm` (~0.5 h; resumes per pair).
- If a run is stopped at the limit, or dies, rerun the same command. It resumes from `results/v/cache/` with only the missing units, and no block is written until a stage is complete.
- Never run two pool stages at once.

**Stop points:** `STOP_REUSE` (exit 3) or any exit 7; `STOP_V_PATH_REPRO` at path, kc_input or kc_band (exit 3) or their `INVALID` (exit 5) or refusal (T's z rows, U's rows or R's even raw missing); `STOP_SET_SHORT` or a set refusal; `STOP_Z_DEGENERATE` or a z `INVALID`; `STOP_STRENGTH_LEVER`; `STOP_EVEN_REPRO`, `STOP_EVEN_PUNISH`, `STOP_EVEN_LOW_LEVER` (exit 3), an even `INVALID` or refusal; a smoke with problems (exit 6) after one diagnosed fix attempt; any gate ② STOP (exit 3); a second gate ② `INVALID`; seal `NOT_READ` / `INVALID` (exit 6); judge `NOT_READ` (exit 6); and the judgement result itself. At a stop point:
1. Commit the block (a smoke block with problems is not committed; see step 8).
2. Write the spec section V.10 and the README ledger (step 14) — V.8: "관문 STOP이나 판정 결과에서 V.10 결과 절과 README 원장(ko/en)을 쓴 뒤 멈춘다". The path stage's mechanism records go into V.10 whenever path ran.
3. Overwrite `/private/tmp/claude-503/p-latest.md` with the full text:
   - the stage and its label or band;
   - the closing sentence verbatim;
   - the numbers behind it;
   - the mechanism records (three engines) and, when they ran, the KC-input record (dropped odours, rows) and the set's facts;
   - the commits;
   - what was or was not written (V.10, README ledger);
   - the decision V.7 leaves to the user.
4. Stop. Do not continue to another stage.

0. **Preconditions (no pool):**
   1. `git status` must be clean and HEAD must be the final implementation commit; `uv run pytest -q` passes.
   2. `ls results/p/run/cache/p_arm | wc -l` → 192; `ls results/r/cache/r_oracle | wc -l` → 277; `shasum -a 256 results/t/z.json results/u/scan.json results/u/path.json` → `fc8bb380…`, `a4cddfd7…`, `57ae0c4e…`.
   3. The reused blocks are in history: `git merge-base --is-ancestor 22c934b HEAD && git merge-base --is-ancestor 0eb642e HEAD && git merge-base --is-ancestor c0d09a7 HEAD`.
   4. `mkdir -p ~/flymon-archive/v`.
1. **Reuse (V.3 1):**
   1. Run `uv run python scripts/run_v.py --stage reuse` (seconds, no pool).
   2. On exit 0, commit: `git add results/summary/v_lever.json && git commit -m "results(v): reuse — shared key 3c2699c7 = R's (repro 22c934b); T key 7255f872 = T z block 0eb642e; V key = U key 8a4e0930 (U blocks d00fdf2/3b83824/d732b20/c0d09a7)"`.
   3. On exit 3: commit `results(v): reuse STOP_REUSE — <reasons>`, then **stop point**.
2. **Path (V.3 2):**
   1. Run `uv run python scripts/run_v.py --stage path` (~10 min).
   2. On exit 0, commit: `git commit -am "results(v): path — unedited = T none rows, APL-only = T lever rows, L_V = U entry_f0 rows bit for bit (CSC 1aee8398 / 860cba4f / 2d359b8b, 13 edges, MBON05->APL untouched); oracle 3+3 = R even L / C raw"`.
   3. On exit 3 / 5: commit `results(v): path <label> — <sentence or reasons>`, then **stop point**.
3. **KC input (V.3 3):**
   1. Run `uv run python scripts/run_v.py --stage kc_input` (~10–20 min; resumes).
   2. Report in one line: candidates, outside-band odours per engine, U comparison (shared / differ).
   3. On exit 0, commit: `git commit -am "results(v): kc_input — 173 candidate odours on both engines; outside [0.03, 0.15]: none <k> / L_V <k>; U per_odour_none reproduced on <n> shared odours"`.
   4. On exit 3 / 5: commit, then **stop point**.
4. **Set (V.2):**
   1. Run `uv run python scripts/run_v.py --stage set` (no pool).
   2. On exit 0, commit: `git commit -am "results(v): set — (b) 21 last turn <t_b> · (a) 43 last turn <t_a>; digests <e0_b[:8]> / <e0_a[:8]> / <keys[:8]>; kc_input skips <k>; <k_cl> (b) clusters"`.
   3. On exit 3 (`STOP_SET_SHORT`) or a refusal: commit any block, then **stop point**.
5. **z_V (V.3 5):**
   1. Run `uv run python scripts/run_v.py --stage z` (no pool).
   2. On exit 0, commit: `git commit -am "results(v): z — z_V A <m> / <sd>, P <m> / <sd>; MBON13 Δ <·> zero <·>; σA_V/σA_h4 <·>, σP_V/σP_h4 <·>"`.
   3. On exit 3 / 5: commit, then **stop point**.
6. **Combined-engine KC band (V.3 6):**
   1. Run `uv run python scripts/run_v.py --stage kc_band` (~10 min).
   2. On exit 0, commit: `git commit -am "results(v): kc_band — 112-odour median <·> under L_V, ORN cap OK, V set odours re-measured = kc_input bit for bit (min/max <·>/<·>)"`.
   3. On exit 3 / 5: commit, then **stop point**.
7. **Even (V.3 7):**
   1. Run `uv run python scripts/run_v.py --stage even` (~0.5 h; resumes).
   2. On exit 0, commit: `git commit -am "results(v): even — C from R's even raw c_even 7; L_V testable_b <k>/21 on z_V; net drops (b) <d> / (a) <d>"`.
   3. On exit 3 / 5 or a refusal: commit any block, then **stop point**.
8. **Smoke (V.3 8):**
   1. Run `uv run python scripts/run_v.py --stage smoke --workers 4`.
   2. Expect exit 0 and `problems: []`: P L edges `[2]` on block path's L_V CSC (`2d359b8b…`), C `[0]` on R's repro; oracle L edit `u_apl_mbon05_x0.0+chain_entry` edges `[2]` on z_V, C / E0 `[0]` on h4 z with R's repro CSC (`1aee8398…`).
   3. Report the `cost` estimate in one line, then commit: `git commit -am "results(v): smoke — L_V (CSC 2d359b8b) on z_V, C/E0 on h4 z, sha relations OK, cost <gate2 h> / <jm h per condition>"`.
   4. On exit 6: do not commit the block. Diagnose with `superpowers:systematic-debugging`, have a reviewed fix task (it must not touch any measurement file — R's, T's or `u_measure.py`; that ends in exit 7), discard the block (`git checkout -- results/summary/v_lever.json`), rerun once. A second failure is a **stop point**.
9. **Operating characteristics (V.6):**
   1. Run `uv run python scripts/run_v.py --stage oc` (seconds), then `uv run python scripts/run_v.py --stage gate2_oc` (seconds).
   2. Commit each: `git commit -am "results(v): operating characteristics — independent 0.034/0.145/0.281/0.362, 0.752/0.949/0.961/0.549; V cluster model (results/v/oc_cluster.json <sha[:8]>) null <·>, harm <·>/<·>"` and `git commit -am "results(v): gate 2 operating characteristic — P(STOP_PUNISH_WEAKENED) at true ratio 0.4/0.5/0.61/0.83/1.0 on h4 z, before gate 2"`.
10. **Gate ② (V.3 10, V.9.2)** — after OPEN 1 is answered:
    1. Run `uv run python scripts/run_v.py --stage gate2` (~0.3 h; resumes).
    2. On exit 0, commit: `git commit -am "results(v): gate 2 PASS — P_L(V) and P_C LEARNS_CONFIRMATORY on 25_400_xxx (h4 z); ratio h4 r1 <·> r2 <·>, z_V r1 <·> r2 <·> (>= 0.5)"`.
    3. On exit 3: commit, then **stop point** (name the failing scale).
    4. On exit 5 (INVALID): commit the block; fix the V code with a reviewed task (the pipeline key must change; no measurement file may change); run `uv run python scripts/run_v.py --stage gate2 --rerun-after-invalid` once. Its outcome is final.
11. **The set-use marker (V.9.5 P2-8):**
    1. Run `uv run python scripts/run_v.py --stage measurement_started` (no pool).
    2. Commit: `git commit -am "results(v): measurement_started — V set used from here (digests <keys[:8]>), seeds 24_600_xxx, z_V pinned"`.
12. **Judgement measurement (one condition per run, in this order: L, C, E0):**
    1. Run `uv run python scripts/run_v.py --stage jm --condition <L|C|E0>` (~0.5 h each; resumes).
    2. The blocks hold no statistic. Do not compute anything from `results/v/cache/` before the seal.
    3. Commit after each: `git commit -am "results(v): judgement measurement <cond> — 64 pairs complete (<L_V on z_V|h4 z>, no band read)"`.
13. **Seal (R.9.7)** — after OPEN 2–3 are answered:
    1. Run `uv run python scripts/run_v.py --stage seal` (no pool).
    2. On exit 0 (`SEALED`), check `find <seal.archive.dir> -type f | wc -l` → 192, then commit: `git commit -am "results(v): seal — pre-read validity OK, raw manifest (192 entries), decision code + z/set/kc_input pinned, archive ~/flymon-archive/v/<dir>"`.
    3. On exit 6: commit, then **stop point**. The judgement is not read.
14. **Judge (once) and the result:**
    1. Run `uv run python scripts/run_v.py --stage judge` (no pool).
    2. If it died after writing `results/v/judge_read.json` and before the block (no `judge` block, no `results/v/judge_done.json`), rerun the same command once (`resumed_after_mark: true`). Any further failure is a **stop point**.
    3. Commit: `git commit -am "results(v): judgement — <band> (<n>/21 vs <c>/21, F_a <f_a>/43, G_fail_S <true|false>)"`.
    4. Write the result (also at every gate STOP above, with what was reached):
       - the spec section `### V.10 결과 (…, <판정 1회 — **<band>**> | <관문 STOP — **<label>**, 판정 없음>)` at the end of appendix V, in U.10's form: implementation range and block commits; reuse; the path gate (three engines, static facts, 3 + 3 oracle pairs) and the mechanism records per engine (MBON05 / 13 / 09 / 11 / 01 / 03, CRE055, MBON30, LHMB1, α′β′ KC subtypes, all KCs, APL release, MBON05 ratio, σA / σP ratios); the KC input (candidates, per-engine outside-band odours, U comparison, seed-range extremes, rows dropped, R / S / T rows using them); the set (turns, digests, skips, cluster table); z_V; the KC band; the even values and gates; smoke; both OCs (with the cluster model's sha); gate ② (labels, ℓ, both ratios, P_L on z_V); the marker; seal; the judgement (n, c, F_a, naive_a, G_fail_S per axis); the closing sentence verbatim; the records (transitions, clusters, α-fixed sensitivity, saturation, naive values); "다음: V.7대로 기록하고 사용자가 판단한다".
       - the README ledger lines, Korean (after U's bullet, which ends at ~line 204) and English (after U's bullet, which ends at ~line 345), in U's form: one bullet each with the label or band, the numbers and "next is the user's decision".
       - Commit: `git commit -m "docs(v): V.10 result — <band or STOP label> (<numbers>); README ledger ko/en"`.
    5. Go to the **stop point** with the result: the closing sentence verbatim and V.7's consequence (B_Tb closes the combined-lever claim on the widened pool with per-engine z only; `SELECTED` stops at "M2 testability on the widened pool with per-engine z" and does not by itself justify a POOL F v4 learning test; every other band is recorded for the user). Pushing follows the standing FlyMon push rule (`gh auth switch --user lyutvs`); the verdict itself is reported, not acted on.
