# Spec Q — Reward-Readout (MBON05) Bottleneck Diagnosis (Characterisation): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix Q needs (as overridden by Q.6), and nothing else:
- **Q0** (no simulation): per-pair records pulled from the 21 encoder k2-norm even (b) oracle caches already copied to `results/q/q0_cache/`, with a digest gate. These records fix the F/S split: F = composite r < 2 (12 pairs), S = r ≥ 2 (9 pairs).
- a **reproduction gate**: the unmanipulated Q oracle, run on H.4 seeds, must reproduce 3 of those caches bit for bit before anything else runs;
- **Q1** (simulation, new seeds 24_000_000+i / 24_000_100+i / 24_000_200+i, i < 8). Six conditions run one at a time, all with the 21 (b) pairs:
  - the baseline, which also carries (a), the fixed α 0.8 / 1.0 arms;
  - (b) `apl_to_mbon05_zero` (primary) and `apl_to_nonkc_zero` (record only);
  - (c) s = s_c;
  - (d) `mv_scale` lo and hi.
- the **records**: for each of the five candidates, 일치 / 불일치 / 판단 불가, plus `INVALID` where data is defective. Also the sensitivity values, the noise floor and the decomposition records.

Q is a **characterisation**. It has no verdict label and no STOP (Q.1). An agreement (일치) is evidence for choosing a lever in appendix R. It does not prove a cause (Q.5).

**Architecture:**
- New files only: `flymon/brain/q_*.py`, `scripts/run_q.py`, and tests. The one exception is a single line added to `tests/brain/test_p_spec.py` (Reading 1).
- Everything else is imported and never edited: N, O and P modules; the encoder track (`flymon/agent/e_*.py`, `encode_grid.py`, `encode.py`); and `h4_*`.
- Modules:
  - `q_spec` holds every Q number.
  - `q_pairs` rebuilds the 21 (b) rows with k2-norm odours and maps them to the Q0 cache files. It also computes s_c.
  - `q_jobs` is the worker job. It runs H.4's oracle, copied, on a rig built for (Params, edit), and adds Q's records: the fixed arms, reach / W_X / f_X and the APL output.
  - `q_store` holds the guarded writes and the content-key cache.
  - `q_measure` holds the measurer: one cache entry per (pair, condition, seed block), with resume.
  - `q_records` holds the per-pair values and the per-condition gate.
  - `q_rules` holds the concordance rules and the assembly of the records.
  - `q_runner` holds the stage chain.
- `scripts/run_q.py` is the CLI.

**Tech Stack:** Python 3.13, NumPy, SciPy 1.18 (`spearmanr`, `wilcoxon`), pytest, `uv`, and `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **Q** (Q.0–Q.5) and its red-team amendment **Q.6**, which wins wherever the two differ. Context:
- `docs/superpowers/specs/2026-10-01-encoder-redesign-design.md`: §4.3 for s and the ORN cap, §4.4 for the even oracle, §4.5 for the seed blocks, §13 for the k2-norm result.
- The committed `results/summary/encoder_grid.json`: blocks `codebook`, `strength`, `even`.

| Q.6 section | Overrides / adds |
|---|---|
| Q.6.1 | F/S stay on composite r. Rules ①–④ use the MBON05-only r_P, i.e. `h4_rules.combo_records`' `single_type["P"]`. Rule ⑤ uses the per-seed Δ(V_X − V_Y) mean and SD, and must hold on **both** Q0 and the Q1 baseline. The A/P decomposition is a record. An F pair with r_P ≥ 2 is recorded as "A·Y항 상쇄". |
| Q.6.2 | The edit is w0·(1 − α·fx). Fixed arms α 0.8 / 1.0 run on the baseline report seeds (the selected α_r baseline stays). ③ is defined through W_X = Σ w0·fx and: AUC of \|ΔP_X(0.8)\|/naive P_X ≥ 0.75 ∧ ρ(W_X, \|ΔP_X(0.8)\|) ≥ 0.4 ∧ F median P_X after α 0.8 ≥ 5. The 0.8 → 1.0 saturation moves to ①'s auxiliary record. |
| Q.6.3 | ② and ④ use the F pairs' paired Δr_P: Wilcoxon p < 0.05 ∧ median ≥ max(0.5, 2 × noise floor). Noise floor = median over pairs of \|r_P(Q0) − r_P(Q1 baseline)\|. ② 불일치 ⇔ median ≤ 0. ④ 불일치 ⇔ \|median\| < noise floor for both valid scales. If F/S disagreement > 5, the AUC rules (①③⑤) become 판단 불가. AUC = P(F < S), ties ½. |
| Q.6.4 | (c) s = the largest s on the 0.05 grid that keeps all 21 pairs' odours under the ORN cap, computed and committed before the run. If s − 1.0 < 0.15, (c) is "조작 약함" and ① is fixed at 판단 불가. The KC band [3%, 15%] gates (b), (c) and (d). |
| Q.6.5 | `mv_scale` (renamed from g) × 0.8 / × 1.25, with directional overlap guards: × 0.8 ρ ≥ +0.4 and × 1.25 ρ ≤ −0.4. If smoke finds the KC median out of band, the fallback is × 0.9 / × 1.1. The ④ limitation sentence goes with the result. |
| Q.6.6 | The primary (b) is `apl_to_mbon05_zero` (APL → MBON05 only, 2 edges); `apl_to_nonkc_zero` is a record. ② also needs the F median of (ratio_b − ratio_base) > 0, and keeps the floor-overlap guard. |
| Q.6.7 | A sensitivity value with the 3 borderline F pairs (r 1.89–1.97) removed. The labels themselves are not changed. |
| Q.6.8 | Smoke seeds 24_008_000–24_008_099; the collision collector covers `e_spec.py`; the Q0 cache copy; content-key storage with atomic writes per (pair × condition) and resume; gates for completeness, uniqueness and finite values; the reproduction gate before smoke; z constants from block `h4` in every condition. |

## Global Constraints

- **Commits carry no trailers.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **Numbers live only in `q_spec`.**
  - Every Q number is a field of `flymon.brain.q_spec.SPEC` (`QSpec`), or of `smoke(SPEC)` / a `dataclasses.replace` for tests.
  - Numbers the encoder track already holds are read from `e_spec.SPEC` (`E`) in field defaults and never restated: windows 800 / 600 / 200, α {0.2, 0.5, 0.8}, the H.4 oracle seeds 500–615, `n_b` 21, `testable_min` 2.0 and the M0d summary path.
  - The types `PPL105` / `PAM08` are read from `e_measure`.
  - Literal-guard tests: `q_spec.py` may contain only the literal set listed in Task 1, and `q_rules.py` only {0, 0.5, 1, 2}.
- **Never edit N/O/P files, `flymon/agent/e_*.py`, `encode_grid.py` or `encode.py` (import only).** The same goes for `h4_*`, `k_jobs`, `h3_*`, `d6a`, `plasticity`, `fly_pool` and `engine_cpu`.
  - The one exception is Reading 1: a single-line addition to `tests/brain/test_p_spec.py`'s `MODULES`, which the encoder track also needed and the user approved in `efc0e24`. The reviewer flags it for the user.
  - No module global of another track is monkeypatched at runtime.
- **The L judgement set (L generator turns 64–103) is never touched.**
  - Q uses only `e_pairs.even_situations`, i.e. H.4's `build_turns(…, 16)` even turns 0–14.
  - `q_pairs.check_turns` refuses any other turn.
  - An AST test (Task 1) forbids, in every `flymon/brain/q_*.py` and `scripts/run_q*.py`, the names `judgement_set`, `used_situations`, `new_turns`, `alternate_from`, `l_pairs`, `judge_act_seeds`, `judge_select_seeds`, `judge_report_seeds`, `judge_first_turn`, `judge_last_turn`, `l_rng_seed` and `l_total_turns`.
  - The judgement seed blocks 24_100_xxx appear in no Q code path.
- **Q0 reads only `results/q/q0_cache/`.** It finds each file by the basename of block `even`'s manifest `cache_file` and checks it against the manifest's `cache_sha256`. `results/encoder/` is never opened; the encoder cache path is used only to compute a key string (Task 2).
- **Writes are guarded.**
  - Raw files go under `results/q/` (git-ignored by `results/*`).
  - The summary is `results/summary/q_reward.json` (tracked).
  - `results/q/q0_cache/` is read-only for Q's guard, and anything else is refused with SystemExit 2.
  - Writes go to a temporary file and are renamed into place.
- **Scripts:**
  - Tests locate the repository through `Path(__file__)` (`ROOT = Path(__file__).resolve().parents[2]`).
  - Any script that builds a `FlyPool` has `if __name__ == "__main__":`. Pools are never started from a heredoc or `python -c` (spawn re-imports the parent, which forks endlessly).
- **Subagents never run a real or smoke stage and never write under `results/`.** Tests write only under `tmp_path`. The controller runs every stage (section "Runs (controller)").
- **Numbers from Q / Q.6, verbatim:**
  - **Pairs:** E-grid **k2-norm, s 1.0**, even (b) **21** pairs in `h4_pairs.even_pairs` order. The E0 digest is checked inside `e_pairs.even_situations`.
  - **Split:** F = Q0 composite r **< 2** (**12** pairs), S = r **≥ 2** (**9**). The split is fixed on Q0 and never redrawn on Q1. Borderline F pairs have r in **[1.89, 1.97]** (3 pairs).
  - **Oracle:** H.4 as is.
    - C3's adopted Params, with readout A = MBON13 and P = MBON05, as returned by `load_c3_config`.
    - z constants from block `h4`, in every condition.
    - α_r chosen over **{0.2, 0.5, 0.8}**.
    - Windows 800 + 600 ms, KC window 200 ms.
    - Activity, selection and report seeds as listed under "Seeds" below.
    - Fixed reward-only arms **α 0.8 and 1.0** on the report seeds.
  - **Seeds:**
    - Q1: activity **24_000_000+i**, selection **24_000_100+i**, report **24_000_200+i**, i < **8**.
    - Smoke: **24_008_000–24_008_099**.
    - Reproduction gate: H.4's 500–507 / 600–607 / 608–615.
  - **Conditions (one at a time, never mixed):**
    - base: edit none, mv 1.0, s 1.0;
    - (b) `apl_to_mbon05_zero` (**2** CSC edges; primary);
    - (b-record) `apl_to_nonkc_zero`;
    - (c) s = s_c, the largest s on the **0.05** grid with max_rate·s·v ≤ cap for every glomerulus value of the 42 odours. If s_c − 1.0 < **0.15** the condition is "조작 약함";
    - (d) `mv_per_synapse` × **0.8** / × **1.25**, falling back to × **0.9** / × **1.1** per side if the smoke KC median is out of band. The C3 threshold file is left as is.
  - **KC band gate:** median KC active fraction in **[0.03, 0.15]** for (b), (c) and (d); outside it, the condition is `INVALID`.
  - **Stop threshold:** P < **5**.
  - **f_X:** the active-KC threshold is fx ≥ **0.5**.
  - **Rules:**
    - AUC: ≥ **0.75** 일치, ≤ **0.6** 불일치.
    - ρ thresholds: **±0.4**.
    - Wilcoxon p < **0.05**.
    - Δ ≥ max(**0.5**, **2** × noise floor).
    - ⑤: F \|mean\| ≥ **0.75** × S ∧ F SD ≥ **1.5** × S for 일치; F \|mean\| < **0.5** × S for 불일치.
    - ② guard: ≥ **2** F pairs at or above the F median naive P_X.
    - Agreement < **0.7** → "분류 불안정" record; disagreement > **5** → AUC rules 판단 불가.

## Readings of the spec (decided here — the controller's rulings)

1. **`test_p_spec.py` gets one line.** Its `test_every_spec_module_is_enumerated` fails as soon as `flymon/brain/q_spec.py` exists, because it globs `*_spec.py`. Task 1 adds `"flymon/brain/q_spec.py": "flymon.brain.q_spec"` to its `MODULES`. Q's own collision test reuses that file's `_collect` and `MODULES`. `e_spec.py` is already in `MODULES` (`efc0e24`), so the "extend to `e_spec.py`" in Q.6.8 is already satisfied, and the Q test asserts it. **This is the only edit to a P file, and it needs the user's OK at review.** Without it, P's test goes red.
2. **The oracle is copied, not wrapped.** `h4_jobs.oracle_job` builds its rig with `h4_jobs.rig_for(params)`, which has no edit. `q_jobs.q_oracle_job` therefore copies oracle_job's call sequence onto `q_rig(params, edit)`, appends the fixed arms after R2, and reads the APL output during the activity presentations (reading `e.v` does not change the engine). Two checks pin it:
   - a test: with edit `none`, the result minus key `"q"` equals `oracle_job`'s;
   - the reproduction gate: the same comparison against the encoder cache.
3. **The reproduction gate** uses 3 pairs, (b) indices `(0, 10, 20)`; the spec asks for "2–4". It has three parts:
   - the Q oracle's result minus `"q"` must be canonical-JSON equal to the Q0 file's `result`;
   - the Q0 file's sha must match;
   - when the current encoder code key equals block `even`'s `code_key` (it does at HEAD 4d851d7), the rebuilt kwargs must give the manifest's `cache_key`, via `EMeasurer.oracle_key`, computed only and never written. This proves the odours, Params, z, types, seeds, α and windows are the encoder's.

   On failure the stage exits 4 and every later stage refuses.
4. **(a)'s fixed arms run in every condition**, after R2, on the report seeds. ②'s "ratio" (ΔP_X/naive P_X) needs them in (b) as well. Every ratio uses the fixed **α 0.8** arm: ratio = \|mean ΔP_X(0.8)\| / naive P_X, and it is undefined (None) when naive P_X = 0. ΔP_X(α) = mean over report seeds of (P_X after the reward-only α edit − P_X pre).
5. **Per-pair quantities** (`q_records.pair_values`), all over the report seeds:
   - naive P_X = mean pre P_X;
   - ΔP_X = R1 − pre (R1 = selected α_r);
   - Δ(V_X − V_Y) = `dv(R1) − dv(pre)`, giving its mean and SD (ddof 1), so r = mean/SD;
   - r_P = `pair_stats(h4_rules._only(report, "P"))["r"]`; a test checks it equals `combo_records`' `single_type["P"]`;
   - stop share = share of seeds with P_X < 5, recorded both naive and after R1.
6. **f_X and W_X** are computed in the worker from X's KC firing probabilities on the condition's activity seeds; only the baseline's values feed ③. The plastic edges are restricted to those in PAM08's core:
   - W_X = Σ w0·fx over the edges onto cells of type MBON05;
   - f_X = share of KCs with fx ≥ 0.5 that have at least one such edge;
   - also recorded: the edge count, the active-KC count, and the Σ w0·fx share of each core type (MBON21's share among them).
7. **Wilcoxon** is `scipy.stats.wilcoxon(…, zero_method="wilcox", alternative="two-sided")`. If every difference is zero, p = 1.0. **Spearman** is `scipy.stats.spearmanr`; a constant input or fewer than 2 points gives None, and a None never satisfies a threshold.
8. **AUC rules.**
   - **AUC inputs:** ①'s AUC is over Q0 naive P_X. ③'s AUC is over the baseline ratio, with pairs whose ratio is undefined dropped and counted. F/S always come from Q0.
   - **ρ inputs:** every ρ (①, ③, ④'s guards) is over all 21 pairs. ①'s ρ uses the baseline naive P_X and Δr_P(c) = r_P(c) − r_P(base).
   - **The > 5 rule:** Q.6.3 says the F/S-conditioned **일치** of ①③⑤ becomes 판단 불가. Read literally, it downgrades only a would-be 일치; a 불일치 stands.
9. **⑤ combination:** 일치 iff both Q0 and the baseline give 일치; 불일치 iff both give 불일치; anything else is 판단 불가.
10. **① ordering:**
    - (c) weak → 판단 불가, fixed (Q.6.4), checked first;
    - otherwise Q0 AUC ≤ 0.6 → 불일치. Since this uses Q0 only, it holds even when (c) is `INVALID`;
    - otherwise, if (c) is `INVALID` → 판단 불가;
    - otherwise AUC ≥ 0.75 ∧ ρ ≤ −0.4 → 일치;
    - otherwise 판단 불가.
11. **② ordering** (on F pairs):
    - median Δr_P ≤ 0 → 불일치;
    - otherwise Wilcoxon p < 0.05 ∧ median ≥ threshold ∧ the median of (ratio_b − ratio_base) > 0, where the threshold is max(0.5, 2 × noise floor):
      - if ≥ 2 of the F pairs with baseline naive P_X ≥ the F median have Δr_P ≥ threshold → 일치;
      - else → 판단 불가 (overlaps ①);
    - otherwise 판단 불가.

    `apl_to_nonkc_zero` gets the same computation, stored as a record only.
12. **④ ordering:** "변화" is the magnitude \|median Δr_P\|, in either direction.
    - A scale is "big" if p < 0.05 ∧ \|median\| ≥ threshold.
    - A scale "overlaps" by its own direction: lo (0.8 or 0.9) overlaps if ρ(naive P_X, Δr_P) ≥ +0.4; hi (1.25 or 1.1) overlaps if ρ ≤ −0.4.
    - Any big scale that does not overlap → 일치.
    - Otherwise, both scales valid with \|median\| < noise floor → 불일치.
    - Both `INVALID` → 판단 불가; anything else → 판단 불가.

    The Q.6.5 limitation sentence is always attached.
13. **The noise floor** is the median over all 21 pairs.
14. **s_c** follows Q.6.4 literally: there is no 1.4 ceiling (Q.3's 1.4 is replaced), and the cap is checked only on the 42 odours of the 21 pairs, using `encode_grid.cap_ok`'s inequality with `odor_real.cap_hz(C3)` and C3's `max_rate_hz`. Weak ⇔ round(s_c − 1.0, 10) < 0.15.
15. **The mv fallback is decided per side.**
    - Smoke measures the KC median of the 42 odours × 4 smoke seeds at × 0.8 and × 1.25, using `k_jobs.activity_job` with the scaled Params.
    - A side that is out of band switches to its fallback (0.8 → 0.9, 1.25 → 1.1).
    - The fallback is not measured again in smoke; Q1's KC gate decides it.
16. **The KC gate metric** is the median, over the condition's 21 pairs × 2 odours × 8 activity seeds, of the read-window KC active fraction. The base is recorded but not gated (Q.6.4 lists (b), (c) and (d)).
17. **INVALID per condition (Q.4 / Q.6.8).** Any of the following makes a condition `INVALID`; candidates that need it become 판단 불가 (except ①'s Q0-only 불일치):
    - incomplete or duplicate pairs;
    - probe counts that are not the report seeds, or a missing fixed arm;
    - an undefined or non-finite r, r_P or ΔV statistic;
    - more than one CSC sha in a condition;
    - base or s_up with a sha different from the repro sha;
    - an edit condition whose sha equals the repro sha;
    - `apl_to_mbon05_zero` touching a number of edges other than 2;
    - `apl_to_nonkc_zero` touching 0 edges;
    - edit none touching any edge;
    - for (b), (c) and (d), a KC median outside the band.

    Q0 itself is `INVALID` if any file's sha or completeness check fails, or if the split is not 12/9.
18. **"A·Y항 상쇄"** = F pairs with r_P ≥ 2, listed for both Q0 and the baseline. They stay in F for the labels. A second sensitivity value with them removed is recorded next to Q.6.7's borderline sensitivity.
19. **One condition per invocation.** `--stage q1 --condition <name>` runs in this order: base, apl_mbon05, apl_nonkc, s_up, mv_lo, mv_hi. Each opens one pool, and the summary is committed before the next. Every stage refuses if:
    - an earlier block is missing;
    - a later block exists;
    - the summary has uncommitted changes;
    - a hashed Q file is dirty.
20. **Storage keys.** The cache key is sha256 over the Q code key (the files `oracle_job` depends on, plus `q_jobs`, `o_jobs`, `n_jobs`, `n_spec`, `q_measure`, `q_store`) and the canonical inputs. The inputs are the full job kwargs (Params, edit, odours, readout, z, types, seeds, α, fixed α, windows) plus `pair`, `condition` and `block`. That makes one atomic file per (pair, condition, seed block): rerunning a stage resumes, and a corrupt or mismatched file is recomputed.

## Review Focus

1. **naive P_X = 0 on a pair (an X silent before any edit).** The ratio is undefined. Expect it to be None and dropped from ③'s AUC with a count, never a ZeroDivisionError and never 0. Tests: Task 5 (`pair_values`) and Task 6 (`rule_reach`).
2. **Degenerate statistics.** SD 0 gives r = ±inf; constant inputs give Spearman NaN; all-zero differences make Wilcoxon raise. Expect: an infinite or undefined r / r_P makes the condition `INVALID` (a data defect, not a label); ρ = None never satisfies a threshold; Wilcoxon on all zeros gives p = 1.0. Tests: Task 5 and Task 6.
3. **An interrupted or corrupted run.** A truncated cache file, a file whose stored key differs, or a stage killed mid-condition. Expect recomputation of exactly the missing or bad entries and nothing else, and no block written for a partial condition. Tests: Task 4 and Task 8.
4. **Writing where another track lives.** Q's guard must refuse `results/q/q0_cache/…`, `results/encoder/…`, `results/summary/encoder_grid.json` and anything outside `results/q/` plus Q's summary. Tests: Task 4.
5. **The L judgement set reached indirectly.** A situation list that contains turn ≥ 16, an odd turn, or turns 64–103, or Q code that imports L's generator. Expect a `ValueError` before any odour is built, and the AST guard failing in CI. Tests: Task 1 and Task 2.

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/q_spec.py` (create) | `QSpec`, `Condition`, `SPEC`, `smoke()`: every Q number |
| `flymon/brain/q_pairs.py` (create) | the k2-norm codebook check, the 21 (b) rows with odours, the turn guard, the manifest → Q0 file map, the encoder-key check, `cap_s` |
| `flymon/brain/q_jobs.py` (create) | `apply_q_edit`, `q_rig`, `_present_kc_apl`, `reach`, `q_oracle_job` (the worker job) |
| `flymon/brain/q_store.py` (create) | the guard (`results/q/`, `q_reward.json`; q0_cache read-only), atomic writes, summary blocks, `QCache` |
| `flymon/brain/q_measure.py` (create) | `Q_MEASURE_FILES`, `cond_params`, `QMeasurer` (oracle rounds per condition, KC probe) |
| `flymon/brain/q_records.py` (create) | `key_str`, `pair_values`, `q0_record`, `condition_record` (the gates and per-condition records) |
| `flymon/brain/q_rules.py` (create) | statistics helpers, split / stability / noise floor, the five rules, `assemble` |
| `flymon/brain/q_runner.py` (create) | `build_ctx`, `Runner` (stages repro, smoke, s_c, q0, q1, records) |
| `scripts/run_q.py` (create) | the CLI with the `__main__` guard |
| `tests/brain/test_q_spec.py`, `test_q_pairs.py`, `test_q_jobs.py`, `test_q_store_measure.py`, `test_q_records.py`, `test_q_rules.py`, `test_q_runner.py`, `q_fixtures.py` (create) | tests |
| `tests/brain/test_p_spec.py` (modify: one `MODULES` line) | Reading 1 |

---

### Task 1: `q_spec` — numbers, conditions, the seed-collision test, the judgement-set guard

**Files:**
- Create: `flymon/brain/q_spec.py`
- Create: `tests/brain/test_q_spec.py`
- Modify: `tests/brain/test_p_spec.py` (one line in `MODULES`, Reading 1)

**Interfaces:**
- Produces:
  - `Condition(name, edit, mv_scale, strength, role, kc_gate)`, frozen;
  - `QSpec`, whose fields are listed below, with methods `q1_seeds() -> dict`, `repro_seeds() -> dict` and `conditions(s_c: float, mv_lo: float, mv_hi: float) -> tuple[Condition, ...]` in the order of `cond_names`;
  - `SPEC`;
  - `smoke(spec) -> QSpec`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_q_spec.py
"""Spec Q.3 / Q.6: every Q number in q_spec, encoder numbers read from e_spec (not restated); Q's seed blocks
24_000_000+i / 24_000_100+i / 24_000_200+i (i < 8) and smoke 24_008_000-24_008_099 collide with no declared seed and no
declared seed's training seeds (P.6.6's form, test_p_spec's collector, e_spec.py included); no Q code path names the L
judgement set."""
import ast
import dataclasses
import importlib
from pathlib import Path

from flymon.agent.e_spec import SPEC as E
from flymon.brain import d6a
from flymon.brain.q_spec import SPEC, Condition, QSpec, smoke
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
SM = smoke(SPEC)
BASE, STRIDE = 1_000_000, 1000          # conditioning.train_block's rule (n_spec.train_seed_base / stride)


def _declared_without_q() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/q_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


Q_SEEDS = set(SPEC.act_seeds) | set(SPEC.select_seeds) | set(SPEC.report_seeds) | set(SPEC.smoke_seeds)


def test_q_blocks_are_the_declared_ones():
    assert SPEC.act_seeds == tuple(range(24_000_000, 24_000_008))
    assert SPEC.select_seeds == tuple(range(24_000_100, 24_000_108))
    assert SPEC.report_seeds == tuple(range(24_000_200, 24_000_208))
    assert SPEC.smoke_seeds == tuple(range(24_008_000, 24_008_100))
    sm = set(SM.act_seeds) | set(SM.select_seeds) | set(SM.report_seeds) | set(SM.kc_probe_seeds)
    assert sm <= set(SPEC.smoke_seeds) and len(sm) == 11
    assert SPEC.repro_seeds() == dict(act=list(E.even_act_seeds), select=list(E.even_select_seeds),
                                      report=list(E.even_report_seeds))


def test_e_spec_is_collected_and_q_spec_enumerated():
    assert "flymon/agent/e_spec.py" in MODULES and "flymon/brain/q_spec.py" in MODULES


def test_q_seeds_collide_with_nothing_declared():
    declared = _declared_without_q()
    for s in (500, 615, 24_001_000, 24_002_000, 24_009_000, 24_100_000, 24_100_207, 23_000_000):
        assert s in declared, s
    assert not Q_SEEDS & declared
    assert not {q for q in Q_SEEDS if (q - BASE) // STRIDE in declared}     # no declared s trains on a Q seed
    assert 23_000 not in declared and 23_008 not in declared


def test_encoder_numbers_are_read_not_restated():
    assert (SPEC.settle_ms, SPEC.read_ms, SPEC.window_ms) == (E.settle_ms, E.read_ms, E.window_ms)
    assert SPEC.alphas == E.alphas == (0.2, 0.5, 0.8) and SPEC.n_b == E.n_b == 21
    assert SPEC.split_r == E.testable_min == 2.0 and SPEC.m0d_summary == E.m0d_summary
    assert (SPEC.punish_type, SPEC.reward_type) == ("PPL105", "PAM08")


def test_conditions_in_order_and_flags():
    cs = SPEC.conditions(1.3, 0.8, 1.25)
    assert tuple(c.name for c in cs) == SPEC.cond_names == ("base", "apl_mbon05", "apl_nonkc", "s_up", "mv_lo", "mv_hi")
    d = {c.name: c for c in cs}
    assert d["base"] == Condition("base", "none", 1.0, 1.0, "baseline", False)
    assert d["apl_mbon05"].edit == "apl_to_mbon05_zero" and d["apl_mbon05"].role == "primary"
    assert d["apl_nonkc"].edit == "apl_to_nonkc_zero" and d["apl_nonkc"].role == "record"
    assert d["s_up"].strength == 1.3 and d["s_up"].edit == "none"
    assert (d["mv_lo"].mv_scale, d["mv_hi"].mv_scale) == (0.8, 1.25)
    assert all(c.kc_gate for n, c in d.items() if n != "base")


def test_smoke_changes_scale_only():
    keep = {"act_seeds", "select_seeds", "report_seeds", "kc_probe_seeds", "pairs_subset", "smoke", "workers"}
    for f in dataclasses.fields(QSpec):
        if f.name not in keep:
            assert getattr(SM, f.name) == getattr(SPEC, f.name), f.name
    assert SM.smoke and SM.pairs_subset == SPEC.smoke_pairs and len(SM.report_seeds) >= 3


ALLOWED_Q_SPEC = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 20, 0.03, 0.05, 0.15, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0,
                  1.1, 1.25, 1.5, 1.89, 1.97, 2.0, 5.0, 3600.0, 24_000_000, 24_000_008, 24_000_100, 24_000_108,
                  24_000_200, 24_000_208, 24_008_000, 24_008_100}


def test_q_spec_literals_are_the_declared_ones():
    tree = ast.parse((ROOT / "flymon/brain/q_spec.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= ALLOWED_Q_SPEC, nums - ALLOWED_Q_SPEC


FORBIDDEN = {"judgement_set", "used_situations", "new_turns", "alternate_from", "l_pairs", "judge_act_seeds",
             "judge_select_seeds", "judge_report_seeds", "judge_first_turn", "judge_last_turn", "l_rng_seed",
             "l_total_turns"}


def test_no_q_code_path_names_the_l_judgement_set():
    files = sorted((ROOT / "flymon/brain").glob("q_*.py")) + sorted((ROOT / "scripts").glob("run_q*.py"))
    assert files
    for p in files:
        tree = ast.parse(p.read_text())
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names |= {a.name.split(".")[-1] for a in n.names}
                if isinstance(n, ast.ImportFrom) and n.module:
                    names |= set(n.module.split("."))
        assert not names & FORBIDDEN, (p.name, names & FORBIDDEN)
        assert "24_100" not in p.read_text() and "24100" not in p.read_text(), p.name
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_spec.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.brain.q_spec'`.

- [ ] **Step 3: Write `q_spec.py`**

```python
# flymon/brain/q_spec.py
"""Every number of spec appendix Q as amended by Q.6 (Q.6 wins over Q.2-Q.5): the reward-readout (MBON05) bottleneck
diagnosis on the E-grid k2-norm even (b) pairs — a characterisation: per candidate 일치 / 불일치 / 판단 불가, INVALID
for defective data, no verdict label and no STOP (Q.1).
The encoder track's numbers (windows, alphas, the H.4 oracle seeds, n_b, the r split, the M0d path) and the DAN types
are read from e_spec / e_measure, never restated. Smoke runs and tests use smoke(SPEC) or dataclasses.replace."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.e_measure import PUNISH_TYPE, REWARD_TYPE
from ..agent.e_spec import SPEC as E


@dataclass(frozen=True)
class Condition:
    name: str
    edit: str               # "none" | "apl_to_mbon05_zero" | "apl_to_nonkc_zero"
    mv_scale: float         # x mv_per_synapse (Q.6.5 mv_scale)
    strength: float         # presentation strength s
    role: str               # "baseline" | "primary" | "record"
    kc_gate: bool           # Q.6.4: the KC band gates (b), (c), (d)


@dataclass(frozen=True)
class QSpec:
    # ---- pairs and the split (Q.3, Q.6.1, Q.6.7) -----------------------------------------------------------------
    config: str = "k2-norm"
    strength: float = 1.0                       # E-grid k2-norm s 1.0 (encoder block strength)
    n_b: int = E.n_b
    n_f: int = 12                               # Q.3: r < 2 on Q0
    n_s: int = 9
    split_r: float = E.testable_min             # F: r < 2, S: r >= 2
    borderline: tuple = (1.89, 1.97)            # Q.6.7: F pairs with r in [1.89, 1.97]
    even_turns: int = 16                        # h4_pairs.build_turns(..., 16) as e_pairs.even_situations builds them
    pairs_subset: tuple = ()                    # () = all 21 (b) pairs; smoke: smoke_pairs
    smoke_pairs: tuple = (0, 20)
    # ---- the oracle (H.4 as is) and Q's records (Q.6.2, Q.3) --------------------------------------------------------
    alphas: tuple = E.alphas
    fixed_alphas: tuple = (0.8, 1.0)            # Q.6.2: fixed reward-only arms on the report seeds
    active_fx: float = 0.5                      # Q.3: f_X counts KCs with naive firing probability >= 0.5
    settle_ms: float = E.settle_ms
    read_ms: float = E.read_ms
    window_ms: int = E.window_ms
    punish_type: str = PUNISH_TYPE
    reward_type: str = REWARD_TYPE
    p_type: str = "MBON05"                      # the P readout; apl_to_mbon05_zero's target type
    # ---- the reproduction gate (Q.6.8): H.4 seeds, 3 of the 21 (b) pairs ---------------------------------------
    repro_act_seeds: tuple = E.even_act_seeds
    repro_select_seeds: tuple = E.even_select_seeds
    repro_report_seeds: tuple = E.even_report_seeds
    repro_pairs: tuple = (0, 10, 20)
    # ---- Q1 seeds (Q.3) and smoke (Q.6.8) ----------------------------------------------------------------------
    act_seeds: tuple = tuple(range(24_000_000, 24_000_008))
    select_seeds: tuple = tuple(range(24_000_100, 24_000_108))
    report_seeds: tuple = tuple(range(24_000_200, 24_000_208))
    smoke_seeds: tuple = tuple(range(24_008_000, 24_008_100))
    kc_probe_seeds: tuple = ()                  # smoke only: the mv_scale KC-band probe
    # ---- conditions (Q.3, Q.6.4-Q.6.6) ------------------------------------------------------------------------
    cond_names: tuple = ("base", "apl_mbon05", "apl_nonkc", "s_up", "mv_lo", "mv_hi")
    edit_none: str = "none"
    edit_mbon05: str = "apl_to_mbon05_zero"
    edit_nonkc: str = "apl_to_nonkc_zero"
    apl_mbon05_edges: int = 2                   # Q.6.6: APL -> MBON05 on the real connectome
    mv_scales: tuple = (0.8, 1.25)
    mv_fallback: tuple = (0.9, 1.1)
    s_step: float = 0.05
    s_weak: float = 0.15                        # s_c - 1.0 < 0.15 -> "조작 약함", ① 판단 불가
    kc_band: tuple = (0.03, 0.15)
    stop_p: float = 5.0                         # P < 5 = silent (정지)
    # ---- rules (Q.4, Q.6.1-Q.6.6) ------------------------------------------------------------------------------
    auc_match: float = 0.75
    auc_mismatch: float = 0.6
    rho_min: float = 0.4
    wilcoxon_p: float = 0.05
    delta_min: float = 0.5
    noise_mult: float = 2.0
    var_mean_match: float = 0.75
    var_sd_match: float = 1.5
    var_mean_mismatch: float = 0.5
    floor_guard_pairs: int = 2
    fs_agree_min: float = 0.7
    fs_disagree_max: int = 5
    mv_limitation: str = "mv_scale은 입력 구동도 바꾸므로 ④의 일치는 ①과 완전히 갈린 것이 아니다"
    weak_note: str = "조작 약함"
    cancel_note: str = "A·Y항 상쇄"
    # ---- paths and the pool -------------------------------------------------------------------------------------
    encoder_summary: str = "results/summary/encoder_grid.json"
    m0d_summary: str = E.m0d_summary
    q0_cache_dir: str = "results/q/q0_cache"
    cache_dir: str = "results/q/cache"
    smoke_detail: str = "results/q/smoke.json"
    summary: str = "results/summary/q_reward.json"
    smoke: bool = False
    workers: int = 16
    pool_timeout_s: float = 3600.0

    def q1_seeds(self) -> dict:
        return dict(act=list(self.act_seeds), select=list(self.select_seeds), report=list(self.report_seeds))

    def repro_seeds(self) -> dict:
        return dict(act=list(self.repro_act_seeds), select=list(self.repro_select_seeds),
                    report=list(self.repro_report_seeds))

    def conditions(self, s_c: float, mv_lo: float, mv_hi: float) -> tuple:
        s = self.strength
        return (Condition(self.cond_names[0], self.edit_none, 1.0, s, "baseline", False),
                Condition(self.cond_names[1], self.edit_mbon05, 1.0, s, "primary", True),
                Condition(self.cond_names[2], self.edit_nonkc, 1.0, s, "record", True),
                Condition(self.cond_names[3], self.edit_none, 1.0, float(s_c), "primary", True),
                Condition(self.cond_names[4], self.edit_none, float(mv_lo), s, "primary", True),
                Condition(self.cond_names[5], self.edit_none, float(mv_hi), s, "primary", True))


SPEC = QSpec()


def smoke(spec: QSpec = SPEC) -> QSpec:
    """Scale only: 2 activity, 2 selection, 3 report and 4 KC-probe seeds from the smoke block, the two smoke pairs,
    4 workers. Every threshold stays."""
    s = spec.smoke_seeds
    return dataclasses.replace(spec, act_seeds=s[0:2], select_seeds=s[2:4], report_seeds=s[4:7],
                               kc_probe_seeds=s[7:11], pairs_subset=spec.smoke_pairs, smoke=True, workers=4)
```

- [ ] **Step 4: Add Q's spec module to P's enumeration (Reading 1)**

In `tests/brain/test_p_spec.py`, the `MODULES` dict ends with `"flymon/agent/e_spec.py": "flymon.agent.e_spec"}`. Change that last entry to:

```python
           "flymon/agent/e_spec.py": "flymon.agent.e_spec", "flymon/brain/q_spec.py": "flymon.brain.q_spec"}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_spec.py tests/brain/test_p_spec.py -q`
Expected: PASS. The AST guard sees only `q_spec.py` for now and covers the later Q files automatically through the glob.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/q_spec.py tests/brain/test_q_spec.py tests/brain/test_p_spec.py
git commit -m "feat(q): q_spec — Q numbers, conditions, seed blocks, collision and judgement-set guards"
```

---

### Task 2: `q_pairs` — the 21 (b) rows, the turn guard, the Q0 file map, the encoder-key check, s_c

**Files:**
- Create: `flymon/brain/q_pairs.py`
- Test: `tests/brain/test_q_pairs.py`

**Interfaces:**
- Consumes: `QSpec` (Task 1). `e_pairs.attach_odours` / `even_situations`, `e_codebook.cells` / `digest`, `encode_grid.Codebook`, `e_spec.SPEC.k_of` / `dual_rule`, `h4_pairs.pool_vocabulary`, `h3_store.sha256_file`, `e_measure.EMeasurer`, `e_store.ECache`.
- Produces:
  - `key_str(row) -> str` in the form `"axis|turn|x|y"`;
  - `check_turns(rows, spec)`, which raises ValueError;
  - `codebook(enc, spec) -> Codebook`;
  - `check_strength(enc, spec)`;
  - `b_rows(situations, receptor_counts, enc, spec) -> list[dict]`;
  - `manifest(enc, spec) -> dict[str, dict]`;
  - `q0_files(rows, enc, spec) -> list[dict]` with the keys `key, path, cache_key, sha256_declared, sha256, ok`;
  - `encoder_key_match(rows, enc, spec, code, params, readout, z, types, n_kc) -> list[bool]`;
  - `odours_of(rows) -> list[dict]`;
  - `cap_s(odours, max_rate_hz, cap_hz, spec) -> dict` with the keys `s, weak, vmax, step, strength, max_rate_hz, cap_hz`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_q_pairs.py
"""Q.3 / Q.6.4 / Q.6.8: the k2-norm codebook is the committed one (digest), only H.4's even turns 0-14 are accepted (the
L judgement set never), the Q0 file of each (b) pair is the manifest's basename under results/q/q0_cache with its
sha256 checked, and s_c is the largest 0.05-grid s keeping every odour under the ORN cap."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from flymon.brain import q_pairs as Q
from flymon.brain.q_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
ENC = json.loads((ROOT / "results/summary/encoder_grid.json").read_text())
NPZ = ROOT / "data/malecns.npz"


def test_codebook_is_the_committed_k2_norm_digest():
    cb = Q.codebook(ENC, SPEC)
    assert cb.k == 2 and len(cb.words) == 96
    bad = copy.deepcopy(ENC)
    bad["codebook"]["k"]["2"]["codebook"][0] = list(reversed(bad["codebook"]["k"]["2"]["codebook"][1]))
    with pytest.raises(ValueError, match="digest"):
        Q.codebook(bad, SPEC)


def test_strength_is_s_1():
    Q.check_strength(ENC, SPEC)
    bad = copy.deepcopy(ENC)
    bad["strength"]["configs"]["k2-norm"]["s"] = 1.4
    with pytest.raises(ValueError):
        Q.check_strength(bad, SPEC)


@pytest.mark.parametrize("turn", [64, 103, 16, 3])
def test_turn_guard_refuses_anything_but_h4_even_turns(turn):
    rows = [dict(axis="b", turn=0, x="a", y="b"), dict(axis="b", turn=turn, x="c", y="d")]
    with pytest.raises(ValueError, match="judgement set"):
        Q.check_turns(rows, SPEC)


def test_manifest_has_the_21_b_pairs():
    m = Q.manifest(ENC, SPEC)
    assert len(m) == 21 and all(k.startswith("b|") for k in m)
    assert all(int(k.split("|")[1]) % 2 == 0 for k in m)


def test_q0_files_check_the_declared_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = copy.deepcopy(ENC)
    m = [e for e in enc["even"]["configs"]["k2-norm"]["manifest"] if e["axis"] == "b"]
    d = tmp_path / SPEC.q0_cache_dir
    d.mkdir(parents=True)
    for e in m:
        (d / Path(e["cache_file"]).name).write_text('{"kind": "oracle", "result": {}}')
    good = m[0]
    blob = (d / Path(good["cache_file"]).name).read_bytes()
    good["cache_sha256"] = hashlib.sha256(blob).hexdigest()
    rows = [dict(axis=e["axis"], turn=e["turn"], x=e["x"], y=e["y"]) for e in m]
    files = Q.q0_files(rows, enc, SPEC)
    assert files[0]["ok"] and not any(f["ok"] for f in files[1:])
    assert all(f["path"].startswith(SPEC.q0_cache_dir + "/") for f in files)
    assert "results/encoder" not in json.dumps(files)


def test_cap_s_grid_and_weak_rule():
    odours = [{"g1": 1.25, "g2": 0.75}, {"g3": 0.5}]
    got = Q.cap_s(odours, 200.0, 1000.0 / 3.0, SPEC)          # s * 1.25 <= 1.6667 -> 1.3
    assert got["s"] == 1.3 and got["weak"] is False and got["vmax"] == 1.25
    got = Q.cap_s([{"g": 1.6}], 200.0, 1000.0 / 3.0, SPEC)     # s <= 1.0417 -> 1.0, weak
    assert got["s"] == 1.0 and got["weak"] is True
    got = Q.cap_s([{"g": 1.5}], 200.0, 1000.0 / 3.0, SPEC)     # s <= 1.1111 -> 1.1, weak (0.1 < 0.15)
    assert got["s"] == 1.1 and got["weak"] is True
    with pytest.raises(ValueError, match="cap"):
        Q.cap_s([{"g": 2.0}], 200.0, 1000.0 / 3.0, SPEC)        # s 1.0 itself is over the cap


@pytest.mark.skipif(not NPZ.exists(), reason="needs data/malecns.npz")
def test_real_rows_are_the_encoder_odours():
    from flymon.agent import e_pairs
    from flymon.agent.config import load_c3_config
    from flymon.agent.e_runner import MEASURE_FILES_E
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_store import code_key
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = Q.b_rows(e_pairs.even_situations(pops), rc, ENC, SPEC)
    assert len(rows) == 21 and all(r["axis"] == "b" and r["turn"] % 2 == 0 and r["turn"] < 16 for r in rows)
    assert [Q.key_str(r) for r in rows] == list(Q.manifest(ENC, SPEC))
    code = code_key(str(NPZ), files=MEASURE_FILES_E)
    if code["key"] != ENC["even"]["code_key"]:
        pytest.skip("encoder code key moved; the reproduction gate covers it")
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    m0d = json.loads((ROOT / SPEC.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    assert all(Q.encoder_key_match(rows, ENC, SPEC, code, cfg.params, cfg.readout, cfg.z, types, len(pops.kc)))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_pairs.py -q`
Expected: FAIL with `ImportError: cannot import name 'q_pairs'`.

- [ ] **Step 3: Write `q_pairs.py`**

```python
# flymon/brain/q_pairs.py
"""Q.3's pairs: the E-grid k2-norm (s 1.0) even (b) 21 pairs — e_pairs.even_situations (H.4's situations and order;
the E0 digest is checked there) re-encoded with the encoder's committed k = 2 codebook (digest checked against block
codebook) — the Q0 file of each pair (block even's manifest basename under results/q/q0_cache, sha256 checked; Q0 never
opens results/encoder), the encoder-key check of the rebuilt oracle inputs, and Q.6.4's (c) strength."""
from __future__ import annotations

from pathlib import Path

from ..agent import e_codebook, e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import Codebook
from .h3_store import sha256_file
from .h4_pairs import pool_vocabulary


def key_str(r) -> str:
    return f"{r['axis']}|{int(r['turn'])}|{r['x']}|{r['y']}"


def check_turns(rows, spec) -> None:
    """Only H.4's even turns 0..even_turns-1; anything else (an odd turn, the L judgement set's 64-103) is refused."""
    bad = sorted({int(r["turn"]) for r in rows if not (0 <= int(r["turn"]) < spec.even_turns and int(r["turn"]) % 2 == 0)})
    if bad:
        raise ValueError(f"Q uses only H.4's even turns 0..{spec.even_turns - 1}; got turns {bad} "
                         f"(the L judgement set is never used)")


def codebook(enc: dict, spec) -> Codebook:
    k = str(E.k_of(spec.config))
    cfg, rec = enc["codebook"]["configs"][spec.config], enc["codebook"]["k"][k]
    book = rec["codebook"]
    if cfg["status"] != "OK" or book is None or e_codebook.digest(book) != cfg["digest"] or rec["digest"] != cfg["digest"]:
        raise ValueError(f"{spec.config}: codebook digest does not match block codebook's {cfg['digest']}")
    _, _, mon, move = pool_vocabulary()
    cells = e_codebook.cells(move, mon)
    if [list(c) for c in cells] != [list(c) for c in rec["cells"]]:
        raise ValueError(f"{spec.config}: codebook cells differ from block codebook's")
    return Codebook(cells, book)


def check_strength(enc: dict, spec) -> None:
    s1 = enc["strength"]["configs"][spec.config]["s"]
    s2 = enc["even"]["configs"][spec.config]["s"]
    if not (s1 == s2 == spec.strength):
        raise ValueError(f"{spec.config}: encoder strength {s1}/{s2} is not Q's s {spec.strength}")


def b_rows(situations: list, receptor_counts: dict, enc: dict, spec) -> list:
    check_turns(situations, spec)
    cb = codebook(enc, spec)
    rows = [r for r in e_pairs.attach_odours(situations, receptor_counts, cb, E.dual_rule(spec.config))
            if r["axis"] == "b"]
    if len(rows) != spec.n_b:
        raise ValueError(f"{len(rows)} (b) rows, not {spec.n_b}")
    return rows


def manifest(enc: dict, spec) -> dict:
    out = {key_str(e): e for e in enc["even"]["configs"][spec.config]["manifest"] if e["axis"] == "b"}
    if len(out) != spec.n_b:
        raise ValueError(f"block even's manifest holds {len(out)} (b) entries, not {spec.n_b}")
    return out


def q0_files(rows: list, enc: dict, spec) -> list:
    man = manifest(enc, spec)
    out = []
    for r in rows:
        e = man[key_str(r)]
        p = Path(spec.q0_cache_dir) / Path(e["cache_file"]).name
        got = sha256_file(p) if p.exists() else None
        out.append(dict(key=key_str(r), path=str(p), cache_key=e["cache_key"], sha256_declared=e["cache_sha256"],
                        sha256=got, ok=bool(got is not None and got == e["cache_sha256"])))
    return out


def encoder_key_match(rows, enc, spec, code: dict, params, readout, z, types, n_kc) -> list:
    """The encoder's oracle cache key of each row's rebuilt inputs (EMeasurer.oracle_key — computed, nothing is read or
    written) equals block even's manifest cache_key: same odours, Params, readout, z, types, H.4 seeds, alphas, windows."""
    from ..agent.e_measure import EMeasurer
    from ..agent.e_store import ECache
    m = EMeasurer(None, ECache(f"{E.raw_dir}/cache", code), params, n_kc, E, guard_params=False)
    man = manifest(enc, spec)
    return [m.oracle_key(r, spec.strength, readout, z, types, spec.repro_seeds())[0] == man[key_str(r)]["cache_key"]
            for r in rows]


def odours_of(rows) -> list:
    return [o for r in rows for o in (r["odor_x"], r["odor_y"])]


def cap_s(odours, max_rate_hz: float, cap_hz: float, spec) -> dict:
    """Q.6.4: the largest s on the s_step grid with max_rate_hz * s * v <= cap_hz for every glomerulus value v of every
    odour (encode_grid.cap_ok's inequality); weak <=> s - strength < s_weak."""
    vals = [float(v) for o in odours for v in o.values()]

    def ok(s):
        return all(max_rate_hz * s * v <= cap_hz for v in vals)

    if not ok(spec.strength):
        raise ValueError(f"s {spec.strength} is already over the ORN cap")
    n = int(round(spec.strength / spec.s_step))
    while ok(round((n + 1) * spec.s_step, 10)):
        n += 1
    s = round(n * spec.s_step, 10)
    return dict(s=s, weak=bool(round(s - spec.strength, 10) < spec.s_weak), vmax=max(vals), step=spec.s_step,
                strength=spec.strength, max_rate_hz=float(max_rate_hz), cap_hz=float(cap_hz), n_odours=len(odours))
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_pairs.py tests/brain/test_q_spec.py -q`
Expected: PASS. The real-NPZ test runs in this worktree.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/q_pairs.py tests/brain/test_q_pairs.py
git commit -m "feat(q): q_pairs — k2-norm (b) rows, turn guard, Q0 file map, encoder-key check, s_c"
```

---

### Task 3: `q_jobs` — Q's rig and edits, the copied oracle with fixed arms, reach and the APL record

**Files:**
- Create: `flymon/brain/q_jobs.py`
- Test: `tests/brain/test_q_jobs.py`

**Interfaces:**
- Consumes: `h4_jobs.type_cells`, `h4_formula.dprime` / `dv`, `o_jobs.apply_edit` / `NONKC`, `h3_jobs.edge_sources`, `presentation.decide`, `stimuli.present`.
- Produces:
  - `NONE = "none"`, `MBON05 = "apl_to_mbon05_zero"`, `NONKC`, `EDITS`;
  - `apply_q_edit(eng, pops, edit, p_type) -> (sha256, n_edges)`;
  - `q_rig(conn, pops, params, edit, p_type) -> (Engine, Plasticity, comps, sha, n_edges)` and `_RIG`;
  - `reach(e, p, c, fx, reward_type, p_type, active_fx) -> dict`;
  - `q_oracle_job(eng, pl, pops, comps, ro, params, edit, odor_x, odor_y, readout, z, types, act_seeds, select_seeds, report_seeds, alphas, fixed_alphas, active_fx, strength, settle_ms, read_ms, window_ms, punish_type, reward_type) -> dict`. It returns oracle_job's dict plus `"q": {edit, csc_sha256, edit_edges, fixed: {str(α): {counts, report}}, apl_out: {x, y}, fx: {idx, val}, reach}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_q_jobs.py
"""Q's worker job: with edit "none" the result minus "q" is h4_jobs.oracle_job's (Reading 2); apl_to_mbon05_zero zeroes
exactly the APL out-edges onto the P readout type; apl_to_nonkc_zero is O's edit; the fixed arms are the reward-only
edit at alpha; reach is w0*fx over the reward core's P-type edges; the rig is cached per (Params, edit, type)."""
import json
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import o_jobs as O
from flymon.brain import q_jobs as Q
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_jobs import edge_sources
from flymon.brain.stimuli import design_odor_pair

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")            # synthetic PPL105 core, then PAM08 core
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), strength=3.0, settle_ms=50.0, read_ms=100.0,
              window_ms=20, punish_type="PPL105", reward_type="PAM08")
QX = dict(fixed_alphas=(0.8, 1.0), active_fx=0.5)


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
    Q._RIG.clear(); H4._RIG.clear(); O._RIG.clear()
    return c, Populations.from_connectome(c)


def _canon(x):
    return json.dumps(x, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def test_none_edit_is_oracle_job(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    ref = H4.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    assert _canon({k: v for k, v in got.items() if k != "q"}) == _canon(ref)
    q = got["q"]
    assert q["edit"] == "none" and q["edit_edges"] == 0 and set(q["fixed"]) == {"0.8", "1.0"}
    assert len(q["apl_out"]["x"]) == 2 and all(np.isfinite(q["apl_out"]["x"]))
    assert any(x for row in ref["report"]["pre"]["P"] for x in row), "the synthetic regime must drive the readout"


def test_fixed_arm_at_the_selected_alpha_is_r1(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    kw = dict(ORACLE, alphas=(0.8,))
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **kw, **QX)
    assert got["alpha_reward"] == 0.8 and got["q"]["fixed"]["0.8"]["report"] == got["report"]["R1"]


def _zeroed(c, pops, edit, p_type="MBON01"):
    e0, e1 = Engine(c, pops, P, seed=0), Engine(c, pops, P, seed=0)
    _, n = Q.apply_q_edit(e1, pops, edit, p_type)
    return set(np.flatnonzero(e0.csc.w != e1.csc.w).tolist()), n, e0


def test_mbon05_edit_zeroes_exactly_apl_to_p_type(conn_pops):
    c, pops = conn_pops
    got, n, e0 = _zeroed(c, pops, Q.MBON05)
    t = np.asarray(c.type).astype(str)
    want = set(np.flatnonzero(np.isin(edge_sources(e0.csc), pops.apl)
                              & (t[e0.csc.tgt.astype(np.int64)] == "MBON01")).tolist())
    assert got == want and n == len(want) > 0


def test_nonkc_edit_is_o_jobs_edit(conn_pops):
    c, pops = conn_pops
    got, n, _ = _zeroed(c, pops, Q.NONKC)
    e0, e1 = Engine(c, pops, P, seed=0), Engine(c, pops, P, seed=0)
    O.apply_edit(e1, pops, O.NONKC)
    assert got == set(np.flatnonzero(e0.csc.w != e1.csc.w).tolist()) and n == len(got) > 0
    with pytest.raises(ValueError):
        Q.apply_q_edit(Engine(c, pops, P, seed=0), pops, "apl_all_zero", "MBON01")


def test_reach_is_w0_fx_over_the_reward_core_p_edges(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    e, p, comps, _, _ = Q._RIG[(P, "none", "MBON01")]
    fx = np.zeros(len(pops.kc))
    fx[got["q"]["fx"]["idx"]] = got["q"]["fx"]["val"]
    t = np.asarray(c.type).astype(str)
    core = set(comps["PAM08"].core.tolist())
    W, kcs = 0.0, set()
    for i, edge in enumerate(p.edges):
        tgt = int(e.csc.tgt[edge])
        if tgt in core and t[tgt] == "MBON01":
            W += float(p.w0[i]) * fx[p.pre_kc[i]]
            kcs.add(int(p.pre_kc[i]))
    r = got["q"]["reach"]
    assert r["W_X"] == pytest.approx(W) and W > 0
    active = set(np.flatnonzero(fx >= 0.5).tolist())
    assert r["n_active_kc"] == len(active)
    assert r["f_X"] == (pytest.approx(len(active & kcs) / len(active)) if active else None)
    assert set(r["by_type"]) <= {"MBON01", "MBON02"}


def test_mbon05_edit_changes_counts_and_rig_cache(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    n0 = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    m5 = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit=Q.MBON05, odor_x=a, odor_y=b, **ORACLE, **QX)
    assert list(Q._RIG) == [(P, Q.MBON05, "MBON01")]
    assert m5["q"]["csc_sha256"] != n0["q"]["csc_sha256"] and m5["q"]["edit_edges"] > 0
    assert Q._RIG[(P, Q.MBON05, "MBON01")][1].weights_frac() == 1.0


def test_pool_equals_in_process(conn_pops, synthetic_npz, synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)                  # the npz's connectome (no extra APL edge)
    pops = Populations.from_connectome(c)
    a, b = design_odor_pair(pops, k=2, seed=0)
    Q._RIG.clear()
    here = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    with FlyPool(synthetic_npz, Params(), [{}], workers=1) as pool:
        there = pool.run_jobs(Q.q_oracle_job, [dict(params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)])[0]
    assert _canon(here) == _canon(there)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_jobs.py -q`
Expected: FAIL with `ImportError: cannot import name 'q_jobs'`.

- [ ] **Step 3: Write `q_jobs.py`**

```python
# flymon/brain/q_jobs.py
"""FlyPool worker job of spec appendix Q (Q.3, Q.6.2, Q.6.6). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle it; the worker's default engine only lends its connectome.

q_oracle_job is h4_jobs.oracle_job's call sequence (G.14.3) copied onto Q's rig (h4_jobs is imported, never edited):
activity of X and Y on act_seeds, alpha_r chosen on select_seeds (reward change max, then punishment change min, ties to
the smaller alpha), pre / R1 / R2 on report_seeds. After R2 it adds Q's records under key "q":
- the fixed reward-only arms (Q.6.2): w0 * (1 - alpha * fx) on the reward core at each fixed alpha, probed on the
  report seeds;
- X's KC firing probabilities fx (sparse), and reach (Q.6.2 W_X = sum w0 * fx over the reward core's edges onto the P
  readout type; Q.3 f_X = share of KCs with fx >= active_fx having such an edge; per-type sums);
- the APL output of every activity presentation (read window; graded: summed release per step; spiking: spikes),
  read as n_jobs._present reads it (reading the membrane does not change the engine).
With edit "none" the result minus "q" equals oracle_job's (test; the reproduction gate checks it against the encoder
cache, Q.6.8).

Edits (Q.6.6), applied in place on a freshly built engine's CSC (graded-APL views follow), one rig per worker cached
under (Params, edit, P type): "none"; "apl_to_mbon05_zero" — every APL out-edge onto a cell of the P readout type
(MBON05: 2 CSC edges on the real connectome) set to 0; "apl_to_nonkc_zero" — o_jobs.apply_edit, unchanged."""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from . import o_jobs
from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .h4_formula import dprime, dv
from .h4_jobs import type_cells
from .plasticity import Plasticity
from .presentation import decide
from .stimuli import present

NONE = "none"
MBON05 = "apl_to_mbon05_zero"
NONKC = o_jobs.NONKC
EDITS = (NONE, MBON05, NONKC)
_RIG: dict = {}          # (Params, edit, p_type) -> (Engine, Plasticity, comps, csc sha256, edges changed)


def apply_q_edit(eng, pops, edit: str, p_type: str) -> tuple:
    if edit not in EDITS:
        raise ValueError(f"unknown Q edit {edit!r}; Q has {EDITS}")
    before = eng.csc.w.copy()
    if edit == MBON05:
        is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
        t = np.asarray(eng.conn.type).astype(str)
        m = is_apl[edge_sources(eng.csc)] & (t[eng.csc.tgt.astype(np.int64)] == p_type)
        eng.csc.w[m] = np.float32(0.0)
    elif edit == NONKC:
        o_jobs.apply_edit(eng, pops, NONKC)
    n = int((before != eng.csc.w).sum())
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest(), n


def q_rig(conn, pops, params, edit: str, p_type: str):
    key = (params, edit, p_type)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha, n = apply_q_edit(eng, pops, edit, p_type)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha, n)
    return _RIG[key]


def _present_kc_apl(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms: int) -> dict:
    """h4_jobs._present_kc (copied) with the read-window APL output added as n_jobs._present reads it."""
    kc = pops.kc
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    apl = np.asarray(pops.apl, np.int64)
    graded = e.p.apl_mode == "graded"
    e.reset(seed); p.reset_traces(); e.clear_drive(); p.quiet_dan()
    present(e, pops, odor, strength)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32)
    apl_out = 0.0
    n_settle, n_total = int(round(settle_ms / e.p.dt)), int(round((settle_ms + read_ms) / e.p.dt))
    for step in range(n_total):
        fired = e.step()
        f = pos[fired]; f = f[f >= 0]
        win[f] += 1; hist.append(f)
        if len(hist) > window_ms:
            win[hist.popleft()] -= 1
        if f.size:
            max_win = max(max_win, int(win.max()))
        if step >= n_settle:
            read[f] += 1
            if graded:
                apl_out += float(e.apl_release(e.v[apl]).sum())
            else:
                apl_out += float(np.isin(fired, apl).sum())
    return {"read": read, "max_win": max_win,
            "apl_out_per_step": apl_out / ((n_total - n_settle) * max(apl.size, 1))}


def reach(e, p, c, fx, reward_type: str, p_type: str, active_fx: float) -> dict:
    t = np.asarray(e.conn.type).astype(str)
    post_t = t[e.csc.tgt[p.edges].astype(np.int64)]
    rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core])
    contrib = p.w0.astype(np.float64) * fx[p.pre_kc]
    on_p = rew & (post_t == p_type)
    by_type = {str(k): float(contrib[rew & (post_t == k)].sum()) for k in sorted(set(post_t[rew].tolist()))}
    total = float(contrib[rew].sum())
    active = fx >= active_fx
    has = np.zeros(len(fx), bool); has[p.pre_kc[on_p]] = True
    n_active = int(active.sum())
    return dict(W_X=float(contrib[on_p].sum()), n_edges=int((on_p & (fx[p.pre_kc] > 0)).sum()),
                n_active_kc=n_active, f_X=(float((active & has).sum() / n_active) if n_active else None),
                by_type=by_type, total=total)


def q_oracle_job(eng, pl, pops, comps, ro, params, edit: str, odor_x: dict, odor_y: dict, readout: dict, z: dict,
                 types, act_seeds, select_seeds, report_seeds, alphas, fixed_alphas, active_fx: float,
                 strength: float, settle_ms: float, read_ms: float, window_ms: int, punish_type: str,
                 reward_type: str) -> dict:
    e, p, c, sha, n_edit = q_rig(eng.conn, pops, params, edit, readout["P"])
    cells = type_cells(e.conn, types)
    idx = np.concatenate([cells[n] for n in types])
    bounds = np.cumsum([0] + [len(cells[n]) for n in types])
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win, apl = [], [], [], []
            for s in act_seeds:
                o = _present_kc_apl(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"]); apl.append(o["apl_out_per_step"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}, apl

        def probe(seeds):
            out = {n: [] for n in types}
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                for j, n in enumerate(types):
                    out[n].append(cnt[:, bounds[j]:bounds[j + 1]].sum(1).tolist())
            return out

        def ap(pr):
            return {"A": pr[readout["A"]], "P": pr[readout["P"]]}

        fx, kc_x, apl_x = activity(odor_x)
        fy, kc_y, apl_y = activity(odor_y)
        rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core]); pun = np.isin(p.post_mb, p.mb_local[c[punish_type].core])
        w = e.csc.w

        def set_w(a_r, a_p=None):
            wv = p.w0.copy()
            if a_r is not None:
                wv[rew] = p.w0[rew] * (1.0 - a_r * fx)[p.pre_kc[rew]]
            if a_p is not None:
                wv[pun] = p.w0[pun] * (1.0 - a_p * fx)[p.pre_kc[pun]]
            w[p.edges] = wv

        set_w(None); pre_sel = probe(select_seeds)
        reward = {}
        for a in alphas:
            set_w(a); r1 = probe(select_seeds)
            reward[str(a)] = {"R1": r1, "change": dprime(dv(ap(r1), z) - dv(ap(pre_sel), z))}
        a_r = max(alphas, key=lambda a: (reward[str(a)]["change"], -a))
        punish = {}
        for a in alphas:
            set_w(a_r, a); r2 = probe(select_seeds)
            punish[str(a)] = {"R2": r2, "change": dprime(dv(ap(r2), z) - dv(ap(reward[str(a_r)]["R1"]), z))}
        a_p = min(alphas, key=lambda a: (punish[str(a)]["change"], a))
        set_w(None); pre = probe(report_seeds)
        set_w(a_r); R1 = probe(report_seeds)
        set_w(a_r, a_p); R2 = probe(report_seeds)
        fixed = {}
        for a in fixed_alphas:
            set_w(float(a)); fa = probe(report_seeds)
            fixed[str(float(a))] = {"counts": fa, "report": ap(fa)}
        w[p.edges] = p.w0
        nz = np.flatnonzero(fx)
        return {"select": {"pre": pre_sel, "reward": reward, "punish": punish},
                "alpha_reward": a_r, "alpha_punish": a_p,
                "counts": {"pre": pre, "R1": R1, "R2": R2},
                "report": {"pre": ap(pre), "R1": ap(R1), "R2": ap(R2)},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))},
                "q": {"edit": edit, "csc_sha256": sha, "edit_edges": n_edit, "fixed": fixed,
                      "apl_out": {"x": apl_x, "y": apl_y},
                      "fx": {"idx": nz.tolist(), "val": fx[nz].tolist()},
                      "reach": reach(e, p, c, fx, reward_type, readout["P"], active_fx)}}
    finally:
        p.reset_weights(); p.set_enabled(True)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_jobs.py tests/brain/test_h4_jobs.py tests/brain/test_o_jobs.py -q`
Expected: PASS. The h4 and o suites stay green because nothing in them changed.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/q_jobs.py tests/brain/test_q_jobs.py
git commit -m "feat(q): q_jobs — Q rig with apl_to_mbon05_zero, copied H.4 oracle with fixed arms, reach and APL record"
```

---

### Task 4: `q_store` and `q_measure` — guarded atomic storage, content-key cache, the measurer with resume

**Files:**
- Create: `flymon/brain/q_store.py`
- Create: `flymon/brain/q_measure.py`
- Test: `tests/brain/test_q_store_measure.py`

**Interfaces:**
- Consumes: `e_store.ECache`, `h3_store.canonical` / `canonical_pretty`, `pool_bench` guards, `e_runner.MEASURE_FILES_E`, `q_jobs.q_oracle_job`, `k_jobs.activity_job`, `q_pairs.key_str`, `QSpec`, `Condition`.
- Produces:
  - `q_store`: `ALLOWED_DIR`, `READ_ONLY_DIR`, `SUMMARY`, `guard(path, params_list)`, `write_json(path, obj, params_list)`, `read_summary(path)`, `write_summary_block(path, block, obj, params_list)`, and `QCache(root, code)` with `get`, `put`, `key` and `_path`.
  - `q_measure`:
    - `Q_MEASURE_FILES`;
    - `cond_params(params, cond) -> Params`;
    - `QMeasurer(pool, cache, spec, params, readout, z, types, n_kc)`, with:
      - `kwargs(row, cond, seeds) -> dict`;
      - `inputs(row, cond, block, seeds) -> dict`;
      - `run(rows, cond, block, seeds) -> list[{key, result, cache_key, cache_file}]`;
      - `kc_probe(odours: dict, params, strength, seeds) -> dict[oid, list[float]]`;
      - the attributes `last_wall_s` and `last_jobs`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_q_store_measure.py
"""Q.6.8 storage: one atomic content-key file per (pair, condition, seed block) under results/q/cache, resume after an
interruption, a corrupt or foreign file recomputed, writes refused outside results/q/ + q_reward.json and inside
results/q/q0_cache; the job inputs are the encoder's oracle inputs plus edit / fixed arms / active_fx (Reading 3)."""
import json
from pathlib import Path

import pytest

from flymon.agent.e_measure import EMeasurer
from flymon.agent.e_spec import SPEC as E
from flymon.agent.e_store import ECache
from flymon.brain import q_store
from flymon.brain.config import Params
from flymon.brain.q_measure import Q_MEASURE_FILES, QMeasurer, cond_params
from flymon.brain.q_spec import SPEC

READOUT = {"A": "MBON13", "P": "MBON05"}
Z = {"A": (10.78125, 9.41), "P": (26.25, 19.31)}
TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
ROW = dict(axis="b", turn=0, x="Surf", y="Ice Beam", odor_x={"ORN_DM1": 1.2, "ORN_VA2": 0.8},
           odor_y={"ORN_DM6": 1.0, "ORN_VC1": 1.0})
CODE = {"key": "k" * 64}


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn.__name__ == "activity_job":
            return [[dict(i=i, seed=s, kc=[0, 1, 2], n=[1, 1, 1], max_win=3) for i, _, s in kw["items"]] for kw in kws]
        return [{"echo": kw["odor_x"], "edit": kw["edit"], "strength": kw["strength"]} for kw in kws]


def _m(pool, root="results/q/cache"):
    return QMeasurer(pool, q_store.QCache(root, CODE), SPEC, Params(), READOUT, Z, TYPES, n_kc=100)


@pytest.mark.parametrize("path", ["results/encoder/cache/oracle/x.json", "results/summary/encoder_grid.json",
                                  "results/q/q0_cache/x.json", "results/o/x.json", "elsewhere.json"])
def test_guard_refuses(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        q_store.write_json(path, {}, [Params()])
    assert e.value.code == 2 and not (tmp_path / path).exists()


def test_guard_allows_q_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    q_store.write_json("results/q/cache/oracle/a.json", {"a": 1}, [Params()])
    q_store.write_summary_block(q_store.SUMMARY, "s_c", {"s": 1.0}, [Params()])
    assert json.loads((tmp_path / q_store.SUMMARY).read_text()) == {"s_c": {"s": 1.0}}
    assert not list((tmp_path / "results/q/cache/oracle").glob(".*.tmp"))


def test_kwargs_are_the_encoder_oracle_inputs_plus_q_fields():
    seeds = SPEC.repro_seeds()
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    mine = _m(None).kwargs(ROW, base, seeds)
    enc = EMeasurer(None, ECache("unused", CODE), Params(), 100, E, guard_params=False)._oracle_kw(
        ROW, 1.0, READOUT, Z, TYPES, seeds)
    extra = {"edit", "fixed_alphas", "active_fx"}
    assert {k: v for k, v in mine.items() if k not in extra} == enc
    assert mine["edit"] == "none" and mine["fixed_alphas"] == [0.8, 1.0] and mine["active_fx"] == 0.5


def test_cond_params_scales_mv_only():
    p = Params()
    lo = SPEC.conditions(1.0, 0.8, 1.25)[4]
    q = cond_params(p, lo)
    assert q.mv_per_synapse == p.mv_per_synapse * 0.8 and q.kc_thresh_file == p.kc_thresh_file
    assert cond_params(p, SPEC.conditions(1.0, 0.8, 1.25)[0]) is p


def test_one_file_per_pair_condition_block_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    rows = [dict(ROW, x=f"x{i}") for i in range(5)]
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    out = _m(pool).run(rows, base, "q1", SPEC.q1_seeds())
    assert [o["key"] for o in out] == [f"b|0|x{i}|Ice Beam" for i in range(5)]
    assert pool.calls == [("q_oracle_job", 2), ("q_oracle_job", 2), ("q_oracle_job", 1)]
    files = sorted((tmp_path / "results/q/cache/oracle").glob("*.json"))
    assert len(files) == 5
    pool.calls.clear()
    files[0].write_text("{trunc")                                   # a corrupt entry
    again = _m(pool).run(rows, base, "q1", SPEC.q1_seeds())
    assert pool.calls == [("q_oracle_job", 1)] and [o["result"] for o in again] == [o["result"] for o in out]
    pool.calls.clear()
    _m(pool).run(rows, base, "repro", SPEC.repro_seeds())            # another seed block: new entries
    assert sum(n for _, n in pool.calls) == 5


def test_foreign_key_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    base = SPEC.conditions(1.0, 1.0, 1.0)[0]
    _m(pool).run([ROW], base, "q1", SPEC.q1_seeds())
    f = next((tmp_path / "results/q/cache/oracle").glob("*.json"))
    d = json.loads(f.read_text()); d["key"] = "other"; f.write_text(json.dumps(d))
    pool.calls.clear()
    _m(pool).run([ROW], base, "q1", SPEC.q1_seeds())
    assert pool.calls == [("q_oracle_job", 1)]


def test_kc_probe_fractions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = _m(FakePool()).kc_probe({"a": ROW["odor_x"], "b": ROW["odor_y"]}, Params(), 1.0, (24_008_007, 24_008_008))
    assert got == {"a": [0.03, 0.03], "b": [0.03, 0.03]}


def test_measure_files_cover_the_job_imports():
    for f in ("flymon/brain/q_jobs.py", "flymon/brain/o_jobs.py", "flymon/brain/n_jobs.py", "flymon/brain/h4_jobs.py",
              "flymon/brain/q_measure.py", "flymon/brain/q_store.py", "flymon/brain/k_jobs.py"):
        assert f in Q_MEASURE_FILES
    root = Path(__file__).resolve().parents[2]
    assert all((root / f).exists() for f in Q_MEASURE_FILES)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_store_measure.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `q_store.py`**

```python
# flymon/brain/q_store.py
"""Q's only writer (Q.5, Q.6.8): raw files under results/q/ (results/q/q0_cache/ is read-only), the one summary
results/summary/q_reward.json, atomic writes (temporary file + rename), and QCache — e_store.ECache's key with this
guard, the key and inputs stored in every entry, and an unreadable or foreign entry treated as missing."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from ..agent.e_store import ECache
from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/q/"
READ_ONLY_DIR = "results/q/q0_cache/"
SUMMARY = "results/summary/q_reward.json"


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if rel.startswith(READ_ONLY_DIR) or not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: Q writes only under {ALLOWED_DIR} (not {READ_ONLY_DIR}) and {SUMMARY}",
              file=sys.stderr)
        raise SystemExit(2)


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


class QCache(ECache):
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
```

- [ ] **Step 4: Write `q_measure.py`**

```python
# flymon/brain/q_measure.py
"""Q's measurer (Q.3, Q.6.8): q_jobs.q_oracle_job per (pair, condition, seed block) behind QCache — one atomic file
per entry, written as soon as its worker round returns, so an interrupted condition resumes with only the missing
pairs — and the smoke KC-band probe through k_jobs.activity_job (Q.6.5). A condition's Params are C3's with
mv_per_synapse x mv_scale (cond_params); the C3 threshold file is untouched."""
from __future__ import annotations

import dataclasses
import sys
import time

from ..agent.e_runner import MEASURE_FILES_E
from . import k_jobs, q_jobs
from .q_pairs import key_str

MAX_ITEMS = 64
# The cache code key: everything oracle_job's result depends on (the encoder track's list) plus Q's job, O's and N's
# edit modules it calls, and Q's measurer and store.
Q_MEASURE_FILES = tuple(dict.fromkeys(MEASURE_FILES_E + (
    "flymon/brain/q_jobs.py", "flymon/brain/o_jobs.py", "flymon/brain/n_jobs.py", "flymon/brain/n_spec.py",
    "flymon/brain/q_measure.py", "flymon/brain/q_store.py")))


def cond_params(params, cond):
    if cond.mv_scale == 1.0:
        return params
    return dataclasses.replace(params, mv_per_synapse=params.mv_per_synapse * cond.mv_scale)


def _odour(o: dict) -> dict:
    return {str(g): float(v) for g, v in o.items()}


class QMeasurer:
    def __init__(self, pool, cache, spec, params, readout: dict, z: dict, types, n_kc: int):
        self.pool, self.cache, self.spec, self.params = pool, cache, spec, params
        self.readout, self.z, self.types, self.n_kc = dict(readout), dict(z), list(types), int(n_kc)
        self.last_wall_s, self.last_jobs = 0.0, 0

    def _windows(self) -> dict:
        sp = self.spec
        return dict(settle_ms=float(sp.settle_ms), read_ms=float(sp.read_ms), window_ms=int(sp.window_ms))

    def kwargs(self, row: dict, cond, seeds: dict) -> dict:
        sp = self.spec
        return dict(params=cond_params(self.params, cond), odor_x=_odour(row["odor_x"]), odor_y=_odour(row["odor_y"]),
                    readout=dict(self.readout), z={k: [float(v) for v in self.z[k]] for k in self.z},
                    types=[str(t) for t in self.types], act_seeds=[int(s) for s in seeds["act"]],
                    select_seeds=[int(s) for s in seeds["select"]], report_seeds=[int(s) for s in seeds["report"]],
                    alphas=[float(a) for a in sp.alphas], strength=float(cond.strength), **self._windows(),
                    punish_type=sp.punish_type, reward_type=sp.reward_type, edit=cond.edit,
                    fixed_alphas=[float(a) for a in sp.fixed_alphas], active_fx=float(sp.active_fx))

    def inputs(self, row: dict, cond, block: str, seeds: dict) -> dict:
        return dict(self.kwargs(row, cond, seeds), pair=key_str(row), condition=cond.name, block=block)

    def run(self, rows: list, cond, block: str, seeds: dict) -> list:
        ins = [self.inputs(r, cond, block, seeds) for r in rows]
        todo = [i for i, x in enumerate(ins) if self.cache.get("oracle", x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        plist = [cond_params(self.params, cond)]
        t0 = time.perf_counter()
        if todo:
            print(f"q oracle {cond.name}/{block}: {len(todo)}/{len(rows)} pairs to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            res = self.pool.run_jobs(q_jobs.q_oracle_job, [self.kwargs(rows[i], cond, seeds) for i in batch])
            for i, r in zip(batch, res):
                self.cache.put("oracle", ins[i], r, plist)
            print(f"q oracle {cond.name}/{block}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for r, x in zip(rows, ins):
            got = self.cache.get("oracle", x)
            if got is None:
                raise RuntimeError(f"q oracle {cond.name}/{block}: pair {key_str(r)} missing after the run")
            out.append(dict(key=key_str(r), result=got, cache_key=self.cache.key("oracle", x),
                            cache_file=str(self.cache._path("oracle", x))))
        return out

    def kc_probe(self, odours: dict, params, strength: float, seeds) -> dict:
        """{odour id: [KC active fraction per seed]} (read window, k_jobs.activity_job), cached per odour."""
        seeds = [int(s) for s in seeds]

        def inp(oid, o):
            return dict(params=params, odour_id=str(oid), odour=_odour(o), strength=float(strength), seeds=seeds,
                        **self._windows())

        todo = [(oid, o) for oid, o in odours.items() if self.cache.get("kc_probe", inp(oid, o)) is None]
        if todo:
            items = [(k * len(seeds) + j, _odour(o), s) for k, (_, o) in enumerate(todo) for j, s in enumerate(seeds)]
            jobs = [items[a:a + MAX_ITEMS] for a in range(0, len(items), MAX_ITEMS)]
            common = dict(params=params, strength=float(strength), **self._windows())
            got = {}
            n_w = max(1, int(self.pool.n_workers))
            for a in range(0, len(jobs), n_w):
                for part in self.pool.run_jobs(k_jobs.activity_job, [dict(common, items=c) for c in jobs[a:a + n_w]]):
                    for r in part:
                        got[int(r["i"])] = r
            for k, (oid, o) in enumerate(todo):
                rows = [got[k * len(seeds) + j] for j in range(len(seeds))]
                self.cache.put("kc_probe", inp(oid, o), dict(frac=[len(r["kc"]) / self.n_kc for r in rows]), [params])
        return {oid: self.cache.get("kc_probe", inp(oid, o))["frac"] for oid, o in odours.items()}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_store_measure.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/q_store.py flymon/brain/q_measure.py tests/brain/test_q_store_measure.py
git commit -m "feat(q): q_store and q_measure — guarded atomic per-pair cache with resume, QMeasurer, KC probe"
```

---

### Task 5: `q_records` — per-pair values, the Q0 record, the per-condition gate (INVALID)

**Files:**
- Create: `flymon/brain/q_records.py`
- Create: `tests/brain/q_fixtures.py`
- Test: `tests/brain/test_q_records.py`

**Interfaces:**
- Consumes: `h4_formula.pair_stats` / `dv`, `h4_rules._only` / `combo_records`, `d6a.OVER_SPIKES` / `CONDITION`, `q_pairs.key_str`, `QSpec`, `Condition`.
- Produces:
  - `INVALID = "INVALID"`, `OK = "OK"`;
  - `key_str`, re-exported;
  - `pair_values(res, z, spec) -> dict | None`, with the fields of Reading 5 plus `dpx_fixed`, `px_after_fixed`, `ratio`, `W_X`, `f_X`, `n_edges`, `n_active_kc`, `reach_by_type`, `apl_out`, `csc_sha256`, `edit_edges` when `"q"` is present;
  - `q0_record(files, results, spec, z) -> dict` with the keys `status, reasons, files, vals, split, borderline, cancel`;
  - `condition_record(got, cond, spec, z, expected, same_sha=None, other_sha=None) -> dict` with the keys `condition, status, reasons, vals, kc_median, d6a_over_share, d6a_condition, apl_out_median, csc_sha256, edit_edges, naive_px_median, jaccard_median, manifest`.
- Fixtures (`q_fixtures.py`): `TYPES`, `READOUT`, `Z`, `KEYS`, `fake_rows(n)`, and `fake_result(n_rep, n_act, px, dpx, sd, seed, q=True, sha=..., edges=0, kc=0.05, w_x=1.0)`.

- [ ] **Step 1: Write the fixtures**

```python
# tests/brain/q_fixtures.py
"""Fabricated oracle results and rows for the Q record / rule / runner tests (no engine)."""
import numpy as np

TYPES = ["MBON13", "MBON18", "MBON05", "MBON21"]
READOUT = {"A": "MBON13", "P": "MBON05"}
Z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}


def fake_rows(n=21):
    return [dict(axis="b", turn=2 * (i // 3), x=f"x{i}", y=f"y{i}", move_x="WATER", opp_x=["FIRE"], move_y="WATER",
                 opp_y=["GRASS"], odor_x={f"ORN_A{i}": 1.25, f"ORN_B{i}": 0.75}, odor_y={f"ORN_C{i}": 1.0})
            for i in range(n)]


KEYS = [f"b|{2 * (i // 3)}|x{i}|y{i}" for i in range(21)]


def fake_result(n_rep, n_act, px, dpx, sd, seed, q=True, sha="sha-base", edges=0, kc=0.05, w_x=1.0,
                fixed=(0.8, 1.0)):
    """An oracle_job-shaped dict: A constant across phases, Y's P constant, X's P = px (+0..2) pre and pre + dpx
    (+ sd noise) after the reward edit; R2 = R1 (p = 0). With q: Q's records, the fixed arm at alpha scaled
    by alpha / 0.8."""
    rng = np.random.default_rng(seed)
    pre_p = [[int(px) + int(rng.integers(0, 3)), 10 + int(rng.integers(0, 3))] for _ in range(n_rep)]

    def edited(scale):
        return [[max(0, a + int(round(scale * dpx + sd * rng.standard_normal()))), b] for a, b in pre_p]

    r1 = edited(1.0)
    a_ = [[30 + int(rng.integers(0, 3)), 30 + int(rng.integers(0, 3))] for _ in range(n_rep)]
    zero = [[0, 0] for _ in range(n_rep)]

    def cnt(P):
        return {"MBON13": a_, "MBON18": zero, "MBON05": P, "MBON21": zero}

    side = {"frac": [kc] * n_act, "spikes": [100] * n_act, "max_win": [5] * n_act}
    res = {"select": {"pre": {}, "reward": {}, "punish": {}}, "alpha_reward": 0.5, "alpha_punish": 0.8,
           "counts": {"pre": cnt(pre_p), "R1": cnt(r1), "R2": cnt(r1)},
           "report": {"pre": {"A": a_, "P": pre_p}, "R1": {"A": a_, "P": r1}, "R2": {"A": a_, "P": r1}},
           "kc": {"x": dict(side), "y": dict(side), "jaccard": 0.04}}
    if q:
        fx = {}
        for a in fixed:
            pa = edited(a / 0.8)
            fx[str(float(a))] = {"counts": cnt(pa), "report": {"A": a_, "P": pa}}
        res["q"] = {"edit": "none", "csc_sha256": sha, "edit_edges": edges, "fixed": fx,
                    "apl_out": {"x": [0.1] * n_act, "y": [0.1] * n_act}, "fx": {"idx": [], "val": []},
                    "reach": {"W_X": w_x, "n_edges": 3, "n_active_kc": 5, "f_X": 0.4,
                              "by_type": {"MBON05": w_x, "MBON21": 0.1}, "total": w_x + 0.1}}
    return res
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/brain/test_q_records.py
"""Readings 5, 16, 17: per-pair values (r_P = combo_records' single_type P; r = dV mean / SD; ratio None when naive P_X
is 0), the Q0 record (sha gate, 12/9 split, borderline, A·Y cancel) and the per-condition gate (completeness,
uniqueness, finite values, one CSC sha, edit edge counts, the KC band for gated conditions)."""
import math

import numpy as np
import pytest

from flymon.brain import q_records as R
from flymon.brain.h4_rules import combo_records
from flymon.brain.h4_spec import SPEC as H4
from flymon.brain.q_spec import SPEC
from tests.brain.q_fixtures import KEYS, Z, fake_result, fake_rows

COND = {c.name: c for c in SPEC.conditions(1.3, 0.8, 1.25)}


def test_r_p_is_combo_records_single_type_and_r_is_mean_over_sd():
    res = fake_result(8, 8, px=20, dpx=-6, sd=2.0, seed=1)
    v = R.pair_values(res, Z, SPEC)
    rows = [dict(axis="b", turn=0, x="a", y="b", report=res["report"])]
    single = combo_records(rows, Z, H4)["single_type"]["P"]["pairs"][0]
    assert v["r_P"] == pytest.approx(single["r"])
    assert v["r"] == pytest.approx(v["dV_mean"] / v["dV_sd"])
    assert v["naive_px"] == pytest.approx(np.mean([p[0] for p in res["report"]["pre"]["P"]]))
    assert set(v["dpx_fixed"]) == {"0.8", "1.0"} and v["ratio"] == pytest.approx(abs(v["dpx_fixed"]["0.8"]) / v["naive_px"])
    assert 0.0 <= v["stop_naive"] <= 1.0 and v["W_X"] == 1.0


def test_ratio_is_none_when_x_is_silent():
    res = fake_result(8, 8, px=0, dpx=0, sd=0.0, seed=2)
    for ph in ("pre", "R1", "R2"):
        res["report"][ph]["P"] = [[0, 10 + i % 3] for i in range(8)]
    res["q"]["fixed"]["0.8"]["report"]["P"] = [[0, 10] for _ in range(8)]
    v = R.pair_values(res, Z, SPEC)
    assert v is None or v["ratio"] is None                      # sd 0 of dV may also make r undefined


def test_q0_record_without_q_fields_and_split():
    rows = fake_rows()
    res = [fake_result(8, 8, px=20, dpx=(-12 if i >= 12 else -0.5), sd=(1.0 if i >= 12 else 3.0), seed=i, q=False)
           for i in range(21)]
    files = [dict(key=k, ok=True, path=f"p{i}") for i, k in enumerate(KEYS)]
    rec = R.q0_record(files, res, SPEC, Z)
    assert rec["status"] == R.OK, rec["reasons"]
    assert len(rec["split"]["F"]) == 12 and len(rec["split"]["S"]) == 9
    assert "dpx_fixed" not in next(iter(rec["vals"].values()))
    files[3]["ok"] = False
    bad = R.q0_record(files, res, SPEC, Z)
    assert bad["status"] == R.INVALID and any("digest" in r for r in bad["reasons"])
    assert rows[0]["axis"] == "b"


def _got(cond, n=21, **kw):
    return [dict(key=KEYS[i], result=fake_result(8, 8, px=20, dpx=-6, sd=2.0, seed=i, **kw), cache_key="c",
                 cache_file="f") for i in range(n)]


def test_condition_ok_and_records():
    rec = R.condition_record(_got(COND["base"]), COND["base"], SPEC, Z, KEYS, same_sha="sha-base")
    assert rec["status"] == R.OK and rec["kc_median"] == 0.05 and rec["edit_edges"] == [0]
    assert rec["d6a_over_share"] == 0.0 and rec["apl_out_median"] == 0.1


@pytest.mark.parametrize("mutate,needle", [
    (lambda g: g[:-1], "missing"),
    (lambda g: g + g[:1], "duplicate"),
    (lambda g: [dict(x, result=dict(x["result"], q={**x["result"]["q"], "csc_sha256": f"s{i}"})) for i, x in enumerate(g)],
     "sha"),
])
def test_condition_invalid_on_defects(mutate, needle):
    rec = R.condition_record(mutate(_got(COND["base"])), COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.INVALID and any(needle in r for r in rec["reasons"]), rec["reasons"]


def test_condition_invalid_on_short_probes_and_missing_fixed_arm():
    g = _got(COND["base"])
    g[0]["result"]["report"]["pre"]["P"] = g[0]["result"]["report"]["pre"]["P"][:7]
    del g[1]["result"]["q"]["fixed"]["1.0"]
    rec = R.condition_record(g, COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.INVALID and sum(KEYS[0] in r or KEYS[1] in r for r in rec["reasons"]) == 2


def test_non_finite_r_is_invalid_not_a_crash():
    g = _got(COND["base"])
    P = g[2]["result"]["report"]["pre"]["P"]
    g[2]["result"]["report"]["R1"]["P"] = [[a - 3, b] for a, b in P]      # constant change: SD 0 -> r = inf
    rec = R.condition_record(g, COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.INVALID and any(KEYS[2] in r for r in rec["reasons"])


def test_edit_edge_counts_and_kc_band():
    ok = R.condition_record(_got(COND["apl_mbon05"], sha="sha-m05", edges=2), COND["apl_mbon05"], SPEC, Z, KEYS,
                            other_sha="sha-base")
    assert ok["status"] == R.OK
    three = R.condition_record(_got(COND["apl_mbon05"], sha="sha-m05", edges=3), COND["apl_mbon05"], SPEC, Z, KEYS)
    assert three["status"] == R.INVALID
    same = R.condition_record(_got(COND["apl_mbon05"], edges=2), COND["apl_mbon05"], SPEC, Z, KEYS,
                              other_sha="sha-base")
    assert same["status"] == R.INVALID
    hot = R.condition_record(_got(COND["s_up"], kc=0.2), COND["s_up"], SPEC, Z, KEYS, same_sha="sha-base")
    assert hot["status"] == R.INVALID and any("KC" in r for r in hot["reasons"])
    base_hot = R.condition_record(_got(COND["base"], kc=0.2), COND["base"], SPEC, Z, KEYS)
    assert base_hot["status"] == R.OK                                   # the base is recorded, not gated
    assert math.isclose(hot["kc_median"], 0.2)
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_records.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 4: Write `q_records.py`**

```python
# flymon/brain/q_records.py
"""Q's per-pair values and gates (Readings 5, 16, 17). pair_values reads one oracle result (an encoder Q0 cache entry
or a Q1 q_oracle_job result) over its report seeds; q0_record is Q0's gate, values and F/S split (Q.3, Q.6.1, Q.6.7);
condition_record is one Q1 condition's gate (`INVALID` reasons) and records. Nothing here runs the engine."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import d6a
from .h4_formula import dv, pair_stats
from .h4_rules import _only
from .q_pairs import key_str  # noqa: F401  (re-exported)

OK, INVALID = "OK", "INVALID"


def _col(probe: dict, k: str, j: int) -> np.ndarray:
    return np.asarray(probe[k], float)[:, j]


def _sd(x) -> float | None:
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if x.size >= 2 else None


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def pair_values(res: dict, z: dict, spec) -> dict | None:
    rep = res["report"]
    st = pair_stats(rep, z, spec.split_r)
    sp_ = pair_stats(_only(rep, "P"), z, spec.split_r)
    if st is None or sp_ is None:
        return None
    pre, r1 = rep["pre"], rep["R1"]
    dV = dv(r1, z) - dv(pre, z)
    px0, px1 = _col(pre, "P", 0), _col(r1, "P", 0)
    naive_px = float(px0.mean())
    kx, ky = res["kc"]["x"], res["kc"]["y"]
    out = dict(r=st["r"], d_pre=st["d_pre"], p=st["p"], m=st["m"], r_P=sp_["r"], naive_px=naive_px,
               naive_py=float(_col(pre, "P", 1).mean()), naive_ax=float(_col(pre, "A", 0).mean()),
               naive_ay=float(_col(pre, "A", 1).mean()), alpha_reward=float(res["alpha_reward"]),
               dV_mean=float(dV.mean()), dV_sd=_sd(dV), dpx_mean=float((px1 - px0).mean()), dpx_sd=_sd(px1 - px0),
               decomp={f"d{k}_{s}": float((_col(r1, k, j) - _col(pre, k, j)).mean())
                       for k in ("A", "P") for j, s in ((0, "X"), (1, "Y"))},
               stop_naive=float((px0 < spec.stop_p).mean()), stop_post=float((px1 < spec.stop_p).mean()),
               kc_frac=[float(v) for v in list(kx["frac"]) + list(ky["frac"])],
               kc_max_win=[int(v) for v in list(kx["max_win"]) + list(ky["max_win"])],
               jaccard=float(res["kc"]["jaccard"]), n_report=int(px0.size))
    q = res.get("q")
    if q is not None:
        out["dpx_fixed"] = {a: float((_col(f["report"], "P", 0) - px0).mean()) for a, f in q["fixed"].items()}
        out["px_after_fixed"] = {a: float(_col(f["report"], "P", 0).mean()) for a, f in q["fixed"].items()}
        a0 = str(float(spec.fixed_alphas[0]))
        out["ratio"] = None if naive_px == 0 or a0 not in out["dpx_fixed"] else abs(out["dpx_fixed"][a0]) / naive_px
        rc = q["reach"]
        out.update(W_X=float(rc["W_X"]), f_X=rc["f_X"], n_edges=int(rc["n_edges"]), n_active_kc=int(rc["n_active_kc"]),
                   reach_by_type=dict(rc["by_type"]), apl_out=[float(v) for v in q["apl_out"]["x"] + q["apl_out"]["y"]],
                   csc_sha256=q["csc_sha256"], edit_edges=int(q["edit_edges"]))
    return out


def _stat_ok(v) -> bool:
    return v is not None and all(_finite(v[f]) for f in ("r", "r_P", "dV_mean", "dV_sd", "dpx_mean", "naive_px"))


def split(r_by_key: dict, spec) -> dict:
    return dict(F=sorted(k for k, r in r_by_key.items() if r < spec.split_r),
                S=sorted(k for k, r in r_by_key.items() if r >= spec.split_r))


def q0_record(files: list, results: list, spec, z: dict) -> dict:
    """Q0 (no simulation): files = q_pairs.q0_files rows, results = each file's "result" (None when unreadable)."""
    reasons = [f"Q0 cache digest mismatch: {f['key']}" for f in files if not f["ok"]]
    vals = {}
    n = len(spec.repro_report_seeds)
    for f, res in zip(files, results):
        if res is None or not f["ok"]:
            continue
        if any(len(res["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P")):
            reasons.append(f"{f['key']}: report probes are not the {n} H.4 report seeds")
            continue
        v = pair_values(res, z, spec)
        if not _stat_ok(v):
            reasons.append(f"{f['key']}: undefined or non-finite statistic")
            continue
        vals[f["key"]] = v
    if len(vals) != spec.n_b and not reasons:
        reasons.append(f"{len(vals)} Q0 pairs, not {spec.n_b}")
    sp = split({k: v["r"] for k, v in vals.items()}, spec) if not reasons else None
    if sp and (len(sp["F"]), len(sp["S"])) != (spec.n_f, spec.n_s):
        reasons.append(f"Q0 split F {len(sp['F'])} / S {len(sp['S'])}, declared {spec.n_f} / {spec.n_s}")
    lo, hi = spec.borderline
    border = [k for k in (sp["F"] if sp else []) if lo <= vals[k]["r"] <= hi]
    cancel = [k for k in (sp["F"] if sp else []) if vals[k]["r_P"] >= spec.split_r]
    return dict(status=INVALID if reasons else OK, reasons=reasons, files=files, vals=vals, split=sp,
                borderline=border, cancel=cancel)


def condition_record(got: list, cond, spec, z: dict, expected: list, same_sha: str | None = None,
                     other_sha: str | None = None) -> dict:
    reasons = []
    keys = [g["key"] for g in got]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    exp = set(expected)
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    n = len(spec.report_seeds)
    want_fixed = {str(float(a)) for a in spec.fixed_alphas}
    vals = {}
    for g in got:
        res = g["result"]
        q = res.get("q")
        short = (any(len(res["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P")) or q is None
                 or set(q["fixed"]) != want_fixed or any(len(f["report"]["P"]) != n for f in q["fixed"].values()))
        if short:
            reasons.append(f"{g['key']}: probes are not the {n} report seeds or a fixed arm is missing")
            continue
        v = pair_values(res, z, spec)
        if not _stat_ok(v):
            reasons.append(f"{g['key']}: undefined or non-finite statistic")
            continue
        vals[g["key"]] = v
    shas = sorted({v["csc_sha256"] for v in vals.values()})
    if len(shas) > 1:
        reasons.append(f"CSC sha256 differs between rows: {shas}")
    if same_sha is not None and shas and shas != [same_sha]:
        reasons.append(f"CSC sha256 {shas} is not the reproduction gate's {same_sha}")
    if other_sha is not None and shas == [other_sha]:
        reasons.append("edit left the CSC sha256 equal to the unedited one")
    edges = sorted({v["edit_edges"] for v in vals.values()})
    if cond.edit == spec.edit_mbon05 and edges and edges != [spec.apl_mbon05_edges]:
        reasons.append(f"{cond.edit} changed {edges} edges, declared {spec.apl_mbon05_edges}")
    if cond.edit == spec.edit_nonkc and edges and (len(edges) != 1 or edges[0] <= 0):
        reasons.append(f"{cond.edit} changed {edges} edges")
    if cond.edit == spec.edit_none and edges and edges != [0]:
        reasons.append(f"edit none changed {edges} edges")
    fr = [x for v in vals.values() for x in v["kc_frac"]]
    kc_med = float(np.median(fr)) if fr else None
    lo, hi = spec.kc_band
    if cond.kc_gate and (kc_med is None or not lo <= kc_med <= hi):
        reasons.append(f"KC 활성 중앙값 {kc_med} 대역 [{lo}, {hi}] 밖")
    wins = [w for v in vals.values() for w in v["kc_max_win"]]
    apl = [a for v in vals.values() for a in v.get("apl_out", [])]
    return dict(condition=dataclasses.asdict(cond), status=INVALID if reasons else OK, reasons=reasons, vals=vals,
                kc_median=kc_med,
                d6a_over_share=(float(np.mean([w >= d6a.OVER_SPIKES for w in wins])) if wins else None),
                d6a_condition=d6a.CONDITION, apl_out_median=(float(np.median(apl)) if apl else None),
                csc_sha256=shas[0] if len(shas) == 1 else shas, edit_edges=edges,
                naive_px_median=(float(np.median([v["naive_px"] for v in vals.values()])) if vals else None),
                jaccard_median=(float(np.median([v["jaccard"] for v in vals.values()])) if vals else None),
                manifest=[dict(key=g["key"], cache_key=g.get("cache_key"), cache_file=g.get("cache_file")) for g in got])
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_records.py -q`
Expected: PASS. If the fabricated Q0 split is not exactly 12/9 for these seeds, raise the S `dpx` or lower its `sd` in the test. Do not change the code.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/q_records.py tests/brain/q_fixtures.py tests/brain/test_q_records.py
git commit -m "feat(q): q_records — per-pair values (r, r_P, dV, fixed arms, reach), Q0 record, condition gate"
```

---

### Task 6: `q_rules` (1) — statistics, split stability, noise floor, rules ⑤ ① ③

**Files:**
- Create: `flymon/brain/q_rules.py`
- Test: `tests/brain/test_q_rules.py`

**Interfaces:**
- Consumes: `QSpec`.
- Produces:
  - `MATCH = "일치"`, `MISMATCH = "불일치"`, `UNDECIDED = "판단 불가"`;
  - `med(xs) -> float | None`;
  - `auc_lower(f, s) -> float | None`;
  - `spearman(x, y) -> float | None`;
  - `wilcoxon_p(d) -> float`;
  - `fs_stability(F0, S0, B_r, spec) -> dict` with the keys `disagree, agree_rate, unstable, auc_blocked, F_base, S_base`;
  - `noise_floor(V0, B) -> float`;
  - `rule_variation(V0, B, F, S, stab, spec)`, `rule_floor(V0, B, F, S, stab, C, s_c, spec)` and `rule_reach(V0, B, F, S, stab, spec)`, each returning a dict that has at least `label` and `why`.
- Value dicts `V*[key]` are the `pair_values` outputs of Task 5. `C` is the s_up condition's vals, or None when that condition is `INVALID`. `s_c` is block `s_c`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_q_rules.py
"""Q.4 / Q.6 rules as pure functions (Readings 7-13): AUC = P(F < S) with ties 1/2, Spearman None on constants,
Wilcoxon p = 1 on all-zero differences; ⑤ needs Q0 and the baseline; F/S disagreement > 5 downgrades only a 일치; ①'s
weak rule wins and its Q0-only 불일치 survives an INVALID (c); ③'s floor overlap; ② and ④ (Task 7)."""
import pytest

from flymon.brain import q_rules as R
from flymon.brain.q_spec import SPEC
from tests.brain.q_fixtures import KEYS

F, S = KEYS[:12], KEYS[12:]


def V(**o):
    d = dict(naive_px=20.0, r=1.0, r_P=1.0, dV_mean=0.5, dV_sd=0.5, ratio=0.3, dpx_fixed={"0.8": -6.0, "1.0": -6.5},
             px_after_fixed={"0.8": 14.0, "1.0": 13.5}, W_X=1.0)
    d.update(o)
    return d


def table(fn):
    return {k: fn(i, k) for i, k in enumerate(KEYS)}


STABLE = dict(disagree=0, agree_rate=1.0, unstable=False, auc_blocked=False)
BLOCKED = dict(disagree=6, agree_rate=15 / 21, unstable=False, auc_blocked=True)


def test_helpers():
    assert R.auc_lower([1, 2], [2, 3]) == pytest.approx(0.875)
    assert R.auc_lower([], [1]) is None
    assert R.spearman([1, 2, 3], [5, 5, 5]) is None and R.spearman([1, 2, 3], [1, 2, 4]) == pytest.approx(1.0)
    assert R.wilcoxon_p([0.0] * 12) == 1.0 and R.wilcoxon_p([0.1 * (i + 1) for i in range(12)]) < 0.01
    assert R.med([None, 1.0, 3.0]) == 2.0 and R.med([]) is None


def test_fs_stability_counts_class_changes():
    b_r = {k: (3.0 if k in S else 1.0) for k in KEYS}
    for k in F[:6]:
        b_r[k] = 3.0
    st = R.fs_stability(F, S, b_r, SPEC)
    assert st["disagree"] == 6 and st["auc_blocked"] and not st["unstable"]           # agreement 15/21 = 0.714
    for k in F[6:7]:
        b_r[k] = 3.0
    assert R.fs_stability(F, S, b_r, SPEC)["unstable"]                                 # 14/21 < 0.7


def test_noise_floor_is_median_abs_rp_difference():
    V0 = table(lambda i, k: V(r_P=1.0))
    B = table(lambda i, k: V(r_P=1.0 + (0.3 if i % 2 else 0.1)))     # 11 x 0.1, 10 x 0.3 -> median 0.1
    assert R.noise_floor(V0, B) == pytest.approx(0.1)


def _var_tables(f_mean, f_sd):
    return table(lambda i, k: V(dV_mean=f_mean if k in F else 1.0, dV_sd=f_sd if k in F else 0.4))


def test_variation_needs_both_and_downgrade_only_match():
    match, mism, mid = _var_tables(0.9, 0.7), _var_tables(0.3, 0.7), _var_tables(0.6, 0.4)
    assert R.rule_variation(match, match, F, S, STABLE, SPEC)["label"] == R.MATCH
    assert R.rule_variation(match, mid, F, S, STABLE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_variation(mism, mism, F, S, STABLE, SPEC)["label"] == R.MISMATCH
    assert R.rule_variation(match, match, F, S, BLOCKED, SPEC)["label"] == R.UNDECIDED
    assert R.rule_variation(mism, mism, F, S, BLOCKED, SPEC)["label"] == R.MISMATCH


def _floor_tables(low_f=True, rho_sign=-1):
    V0 = table(lambda i, k: V(naive_px=(2.0 + i if k in F else 30.0 + i) if low_f else 20.0 + (i % 3)))
    B = table(lambda i, k: V(naive_px=V0[k]["naive_px"], r_P=1.0))
    C = table(lambda i, k: V(r_P=1.0 + rho_sign * 0.1 * V0[k]["naive_px"]))
    return V0, B, C


STRONG, WEAK = dict(s=1.3, weak=False), dict(s=1.1, weak=True)


def test_floor_match_mismatch_and_ordering():
    V0, B, C = _floor_tables()
    assert R.rule_floor(V0, B, F, S, STABLE, C, STRONG, SPEC)["label"] == R.MATCH
    assert R.rule_floor(V0, B, F, S, STABLE, C, WEAK, SPEC)["label"] == R.UNDECIDED           # weak: fixed
    assert R.rule_floor(V0, B, F, S, STABLE, None, STRONG, SPEC)["label"] == R.UNDECIDED      # (c) INVALID
    assert R.rule_floor(V0, B, F, S, BLOCKED, C, STRONG, SPEC)["label"] == R.UNDECIDED
    V0f, Bf, Cf = _floor_tables(low_f=False)
    assert R.rule_floor(V0f, Bf, F, S, STABLE, None, STRONG, SPEC)["label"] == R.MISMATCH     # Q0-only AUC
    V0p, Bp, Cp = _floor_tables(rho_sign=+1)
    assert R.rule_floor(V0p, Bp, F, S, STABLE, Cp, STRONG, SPEC)["label"] == R.UNDECIDED      # rho > 0


def _reach_tables(px_after=14.0, rho=True, f_low=True):
    def one(i, k):
        if k in F:
            d08 = -(1.0 + 0.1 * i) if f_low else -(20.0 + 0.1 * i)          # F low (match) or F high (mismatch)
        else:
            d08 = -(8.0 + 0.1 * i)
        return V(naive_px=20.0, ratio=abs(d08) / 20.0, dpx_fixed={"0.8": d08, "1.0": d08 - 0.05},
                 px_after_fixed={"0.8": px_after, "1.0": px_after - 0.1}, W_X=(abs(d08) if rho else 1.0 + (i % 2)))
    return table(one)


def test_reach_rules():
    B = _reach_tables()
    assert R.rule_reach(B, B, F, S, STABLE, SPEC)["label"] == R.MATCH
    assert R.rule_reach(_reach_tables(px_after=3.0), _reach_tables(px_after=3.0), F, S, STABLE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_reach(_reach_tables(rho=False), _reach_tables(rho=False), F, S, STABLE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_reach(_reach_tables(f_low=False), _reach_tables(f_low=False), F, S, STABLE, SPEC)["label"] == R.MISMATCH


def test_reach_drops_undefined_ratios_and_counts_them():
    B = _reach_tables()
    for k in F[:2]:
        B[k] = dict(B[k], ratio=None)
    rec = R.rule_reach(B, B, F, S, STABLE, SPEC)
    assert rec["n_ratio_undefined"] == 2 and rec["label"] == R.MATCH
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_rules.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `q_rules.py` (part 1)**

```python
# flymon/brain/q_rules.py
"""Spec Q.4 as amended by Q.6: the concordance rules for the five candidates as pure functions over Q0 / Q1 per-pair
values (q_records.pair_values). Every label is a record — 일치 / 불일치 / 판단 불가 — never a verdict (Q.1).
Readings 7-13 of the plan fix the open choices: two-sided Wilcoxon (all-zero differences -> p = 1), Spearman None on
fewer than two points or a constant input (None meets no threshold), ρ over all 21 pairs, F/S always Q0's, the
F/S-disagreement rule downgrading only a 일치."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import spearmanr, wilcoxon

MATCH, MISMATCH, UNDECIDED = "일치", "불일치", "판단 불가"


def med(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def auc_lower(f, s) -> float | None:
    """P(value_F < value_S), ties 1/2 (Q.6.3)."""
    f = [x for x in f if x is not None]; s = [x for x in s if x is not None]
    if not f or not s:
        return None
    return sum(1.0 if a < b else 0.5 if a == b else 0.0 for a in f for b in s) / (len(f) * len(s))


def spearman(x, y) -> float | None:
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pts) < 2 or len({a for a, _ in pts}) < 2 or len({b for _, b in pts}) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr([a for a, _ in pts], [b for _, b in pts])[0])
    return None if math.isnan(rho) else rho


def wilcoxon_p(d) -> float:
    d = [float(x) for x in d if x is not None]
    if not any(x != 0 for x in d):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def fs_stability(F0, S0, base_r: dict, spec) -> dict:
    Fb = sorted(k for k, r in base_r.items() if r < spec.split_r)
    Sb = sorted(k for k, r in base_r.items() if r >= spec.split_r)
    dis = len(set(F0) ^ set(Fb))
    n = len(F0) + len(S0)
    agree = 1.0 - dis / n
    return dict(disagree=dis, agree_rate=agree, unstable=bool(agree < spec.fs_agree_min),
                auc_blocked=bool(dis > spec.fs_disagree_max), F_base=Fb, S_base=Sb)


def noise_floor(V0: dict, B: dict) -> float:
    return med([abs(V0[k]["r_P"] - B[k]["r_P"]) for k in sorted(V0) if k in B])


def _fs_block(lab, why, stab, spec):
    if lab == MATCH and stab["auc_blocked"]:
        return UNDECIDED, f"F/S 불일치 {stab['disagree']}쌍 > {spec.fs_disagree_max} — 일치를 판단 불가로 ({why})"
    return lab, why


# ================================================================ ⑤ 변동 (Q.6.1)
def _variation_one(V, F, S, spec) -> dict:
    mf = med([abs(V[k]["dV_mean"]) for k in F]); ms = med([abs(V[k]["dV_mean"]) for k in S])
    sf = med([V[k]["dV_sd"] for k in F]); ss = med([V[k]["dV_sd"] for k in S])
    if None in (mf, ms, sf, ss):
        lab = UNDECIDED
    elif mf >= spec.var_mean_match * ms and sf >= spec.var_sd_match * ss:
        lab = MATCH
    elif mf < spec.var_mean_mismatch * ms:
        lab = MISMATCH
    else:
        lab = UNDECIDED
    return dict(label=lab, F_abs_mean=mf, S_abs_mean=ms, F_sd=sf, S_sd=ss)


def rule_variation(V0, B, F, S, stab, spec) -> dict:
    q0, b = _variation_one(V0, F, S, spec), _variation_one(B, F, S, spec)
    if q0["label"] == b["label"] == MATCH:
        lab, why = MATCH, "Q0·기준선 모두 일치"
    elif q0["label"] == b["label"] == MISMATCH:
        lab, why = MISMATCH, "Q0·기준선 모두 불일치"
    else:
        lab, why = UNDECIDED, f"Q0 {q0['label']} / 기준선 {b['label']}"
    lab, why = _fs_block(lab, why, stab, spec)
    return dict(label=lab, why=why, q0=q0, base=b)


# ================================================================ ① 바닥 (Q.4, Q.6.1, Q.6.4)
def rule_floor(V0, B, F, S, stab, C, s_c, spec) -> dict:
    a0, a1 = (str(float(a)) for a in spec.fixed_alphas)
    auc = auc_lower([V0[k]["naive_px"] for k in F], [V0[k]["naive_px"] for k in S])
    sat = med([(abs(B[k]["dpx_fixed"][a1]) - abs(B[k]["dpx_fixed"][a0])) / abs(B[k]["dpx_fixed"][a0])
               for k in F if B[k].get("dpx_fixed") and B[k]["dpx_fixed"][a0] != 0])
    keys = sorted(B)
    delta = rho = None
    if C is not None:
        delta = {k: C[k]["r_P"] - B[k]["r_P"] for k in keys}
        rho = spearman([B[k]["naive_px"] for k in keys], [delta[k] for k in keys])
    rec = dict(auc=auc, rho=rho, delta_r_P=delta, saturation_F=sat, s=s_c["s"], weak=s_c["weak"])
    if s_c["weak"]:
        return dict(rec, label=UNDECIDED, why=f"(c) {spec.weak_note} (s {s_c['s']}) — ①은 판단 불가로 고정")
    if auc is not None and auc <= spec.auc_mismatch:
        lab, why = MISMATCH, f"순진 P_X AUC {auc:.3f} ≤ {spec.auc_mismatch}"
    elif C is None:
        lab, why = UNDECIDED, "(c) INVALID — 결과를 쓰지 않음"
    elif auc is not None and auc >= spec.auc_match and rho is not None and rho <= -spec.rho_min:
        lab, why = MATCH, f"AUC {auc:.3f} ≥ {spec.auc_match}, ρ {rho:.3f} ≤ −{spec.rho_min}"
    else:
        lab, why = UNDECIDED, f"AUC {auc}, ρ {rho}"
    lab, why = _fs_block(lab, why, stab, spec)
    return dict(rec, label=lab, why=why)


# ================================================================ ③ 편집 도달 (Q.6.2)
def rule_reach(V0, B, F, S, stab, spec) -> dict:
    a0 = str(float(spec.fixed_alphas[0]))
    keys = sorted(B)
    rf = [B[k]["ratio"] for k in F]; rs = [B[k]["ratio"] for k in S]
    auc = auc_lower(rf, rs)
    rho = spearman([B[k]["W_X"] for k in keys], [abs(B[k]["dpx_fixed"][a0]) for k in keys])
    px = med([B[k]["px_after_fixed"][a0] for k in F])
    rec = dict(auc=auc, rho=rho, px_after_F_median=px, n_ratio_undefined=sum(x is None for x in rf + rs),
               W_X={k: B[k]["W_X"] for k in keys}, f_X={k: B[k].get("f_X") for k in keys})
    if auc is None:
        lab, why = UNDECIDED, "비율이 정의되지 않음"
    elif auc <= spec.auc_mismatch:
        lab, why = MISMATCH, f"비율 AUC {auc:.3f} ≤ {spec.auc_mismatch}"
    elif auc >= spec.auc_match and rho is not None and rho >= spec.rho_min:
        if px is not None and px >= spec.stop_p:
            lab, why = MATCH, f"AUC {auc:.3f}, ρ(W_X) {rho:.3f}, F의 α 0.8 뒤 P_X 중앙값 {px:.2f} ≥ {spec.stop_p}"
        else:
            lab, why = UNDECIDED, f"F의 α 0.8 뒤 P_X 중앙값 {px} < {spec.stop_p} — ①과 겹침"
    else:
        lab, why = UNDECIDED, f"AUC {auc}, ρ {rho}"
    lab, why = _fs_block(lab, why, stab, spec)
    return dict(rec, label=lab, why=why)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_rules.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/q_rules.py tests/brain/test_q_rules.py
git commit -m "feat(q): q_rules (1) — AUC/Spearman/Wilcoxon, F/S stability, noise floor, rules ⑤ ① ③"
```

---

### Task 7: `q_rules` (2) — rules ② and ④, sensitivity, A·Y cancel, assembly, literal guard

**Files:**
- Modify: `flymon/brain/q_rules.py` (append)
- Test: `tests/brain/test_q_rules.py` (append)

**Interfaces:**
- Consumes: Task 6's helpers, and the `q_records.q0_record` / `condition_record` block shapes.
- Produces:
  - `paired_shift(Cv, B, F, noise, spec) -> dict` with the keys `delta, median, p, threshold`;
  - `rule_apl(Cv, B, F, noise, spec) -> dict`;
  - `rule_operating(L, H, B, F, noise, spec) -> dict`;
  - `transitions(B, Cv, spec) -> dict`;
  - `candidates(V0, B, F, S, stab, noise, q1_vals: dict, s_c, spec) -> dict`, keyed `variation, floor, apl, reach, operating`;
  - `assemble(q0: dict, q1: dict[name, block], s_c: dict, spec) -> dict`;
  - `sentences(cands, spec) -> list[str]`.

- [ ] **Step 1: Append the failing tests**

```python
# appended to tests/brain/test_q_rules.py
import ast
from pathlib import Path

NOISE = 0.1


def _apl(delta_fn, ratio_up=0.1):
    B = table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0, ratio=0.3))
    C = table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0 + delta_fn(i, k), ratio=0.3 + ratio_up))
    return C, B


def test_apl_match_guard_and_mismatch():
    C, B = _apl(lambda i, k: 1.0 + 0.1 * i)
    assert R.rule_apl(C, B, F, NOISE, SPEC)["label"] == R.MATCH
    C, B = _apl(lambda i, k: (2.0 + 0.1 * i) if i < 6 else 0.01 * (i + 1))         # rise only in low-naive F pairs
    rec = R.rule_apl(C, B, F, NOISE, SPEC)
    assert rec["label"] == R.UNDECIDED and rec["guard_pairs"] == 0
    C, B = _apl(lambda i, k: -0.2 - 0.01 * i)
    assert R.rule_apl(C, B, F, NOISE, SPEC)["label"] == R.MISMATCH
    C, B = _apl(lambda i, k: 1.0 + 0.1 * i, ratio_up=-0.1)                         # ratio must also rise
    assert R.rule_apl(C, B, F, NOISE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_apl(None, B, F, NOISE, SPEC)["label"] == R.UNDECIDED


def test_threshold_uses_the_noise_floor():
    C, B = _apl(lambda i, k: 0.7)
    assert R.paired_shift(C, B, F, 0.4, SPEC)["threshold"] == pytest.approx(0.8)
    assert R.paired_shift(C, B, F, 0.1, SPEC)["threshold"] == pytest.approx(0.5)


def _mv(sign):
    B = table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0))
    return table(lambda i, k: V(naive_px=2.0 + i, r_P=1.0 + sign * (1.0 + 0.1 * i))), B


def test_operating_directional_overlap():
    up, B = _mv(+1)                     # Δr_P rises with naive P_X: ρ = +1
    rec = R.rule_operating(up, None, B, F, NOISE, SPEC)
    assert rec["label"] == R.UNDECIDED and rec["per"]["mv_lo"]["overlap"]           # lo: ρ ≥ +0.4 overlaps
    rec = R.rule_operating(None, up, B, F, NOISE, SPEC)
    assert rec["label"] == R.MATCH and not rec["per"]["mv_hi"]["overlap"]           # hi overlaps only for ρ ≤ −0.4
    down, B = _mv(-1)                   # ρ = −1, median Δ negative: magnitude counts
    assert R.rule_operating(None, down, B, F, NOISE, SPEC)["label"] == R.UNDECIDED
    assert R.rule_operating(down, None, B, F, NOISE, SPEC)["label"] == R.MATCH
    assert SPEC.mv_limitation in R.rule_operating(down, None, B, F, NOISE, SPEC)["limitation"]


def test_operating_mismatch_needs_both_valid_and_small():
    B = table(lambda i, k: V(r_P=1.0))
    same = table(lambda i, k: V(r_P=1.0))
    assert R.rule_operating(same, same, B, F, 0.2, SPEC)["label"] == R.MISMATCH
    assert R.rule_operating(same, None, B, F, 0.2, SPEC)["label"] == R.UNDECIDED
    assert R.rule_operating(None, None, B, F, 0.2, SPEC)["why"].startswith("두 배율 모두 INVALID")


def _block(vals, status="OK"):
    return dict(status=status, vals=vals, kc_median=0.05, d6a_over_share=0.0, apl_out_median=0.1,
                naive_px_median=20.0, jaccard_median=0.04, reasons=[], condition={})


def test_assemble_shapes_sensitivity_and_cancel():
    V0 = table(lambda i, k: V(r=(1.0 if k in F else 3.0), r_P=(2.5 if k == F[0] else 1.0)))
    V0[F[1]] = dict(V0[F[1]], r=1.9)
    q0 = dict(status="OK", vals=V0, split=dict(F=F, S=S), borderline=[F[1]], cancel=[F[0]])
    q1 = {n: _block(table(lambda i, k: V(r=(1.0 if k in F else 3.0)))) for n in SPEC.cond_names}
    q1["mv_hi"] = _block({}, status="INVALID")
    out = R.assemble(q0, q1, dict(s=1.3, weak=False), SPEC)
    assert set(out["candidates"]) == {"variation", "floor", "apl", "reach", "operating"}
    assert all(c["label"] in (R.MATCH, R.MISMATCH, R.UNDECIDED) for c in out["candidates"].values())
    assert set(out["sensitivity_borderline"]) == set(out["candidates"]) == set(out["sensitivity_cancel"])
    assert out["cancel"]["q0"] == [F[0]] and out["records"]["apl_nonkc"]["role"] == "record"
    assert out["fs"]["disagree"] == 0 and len(out["sentences"]) == 5
    assert out["candidates"]["operating"]["per"]["mv_hi"]["valid"] is False


def test_assemble_with_invalid_q0_or_base():
    q0 = dict(status="INVALID", vals={}, split=None, borderline=[], cancel=[])
    out = R.assemble(q0, {n: _block({}) for n in SPEC.cond_names}, dict(s=1.3, weak=False), SPEC)
    assert all(c["label"] == R.UNDECIDED for c in out["candidates"].values())


def test_q_rules_literals():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "flymon/brain/q_rules.py").read_text())
    nums = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 0.5, 1, 2, 0.0, 1.0}, nums
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_rules.py -q`
Expected: FAIL with `AttributeError: module 'flymon.brain.q_rules' has no attribute 'rule_apl'`.

- [ ] **Step 3: Append to `q_rules.py`**

```python
# ================================================================ ② APL 억제 and ④ 작동점 (Q.6.3, Q.6.5, Q.6.6)
def paired_shift(Cv, B, F, noise, spec) -> dict:
    d = {k: Cv[k]["r_P"] - B[k]["r_P"] for k in F}
    return dict(delta=d, median=med(list(d.values())), p=wilcoxon_p(list(d.values())),
                threshold=max(spec.delta_min, spec.noise_mult * noise))


def rule_apl(Cv, B, F, noise, spec, role: str = "primary") -> dict:
    if Cv is None:
        return dict(label=UNDECIDED, why="(b) INVALID — 결과를 쓰지 않음", role=role)
    sh = paired_shift(Cv, B, F, noise, spec)
    rd = med([Cv[k]["ratio"] - B[k]["ratio"] for k in F if Cv[k]["ratio"] is not None and B[k]["ratio"] is not None])
    nm = med([B[k]["naive_px"] for k in F])
    high = [k for k in F if B[k]["naive_px"] >= nm]
    guard = sum(sh["delta"][k] >= sh["threshold"] for k in high)
    rec = dict(sh, ratio_diff_median=rd, naive_px_F_median=nm, guard_pairs=guard, role=role)
    main = sh["p"] < spec.wilcoxon_p and sh["median"] is not None and sh["median"] >= sh["threshold"]
    if sh["median"] is not None and sh["median"] <= 0:
        lab, why = MISMATCH, f"중앙값 Δr_P {sh['median']:.3f} ≤ 0"
    elif main and rd is not None and rd > 0:
        if guard >= spec.floor_guard_pairs:
            lab, why = MATCH, (f"중앙값 Δr_P {sh['median']:.3f} ≥ {sh['threshold']:.3f}, p {sh['p']:.4f}, "
                               f"비율 차 {rd:.3f} > 0, 순진 P_X 중앙값 이상 F {guard}쌍")
        else:
            lab, why = UNDECIDED, f"증가가 순진 P_X 낮은 F 쌍에만({guard}쌍) — ①과 겹침"
    else:
        lab, why = UNDECIDED, f"중앙값 {sh['median']}, p {sh['p']}, 비율 차 {rd}"
    return dict(rec, label=lab, why=why)


def rule_operating(L, H, B, F, noise, spec) -> dict:
    keys = sorted(B)
    per = {}
    for name, V, up in (("mv_lo", L, True), ("mv_hi", H, False)):
        if V is None:
            per[name] = dict(valid=False)
            continue
        sh = paired_shift(V, B, F, noise, spec)
        rho = spearman([B[k]["naive_px"] for k in keys], [V[k]["r_P"] - B[k]["r_P"] for k in keys])
        overlap = rho is not None and (rho >= spec.rho_min if up else rho <= -spec.rho_min)
        m = sh["median"]
        big = sh["p"] < spec.wilcoxon_p and m is not None and abs(m) >= sh["threshold"]
        small = m is not None and abs(m) < noise
        per[name] = dict(valid=True, median=m, p=sh["p"], threshold=sh["threshold"], rho=rho, overlap=overlap,
                         big=big, small=small, delta=sh["delta"])
    valid = [n for n in per if per[n]["valid"]]
    if not valid:
        lab, why = UNDECIDED, "두 배율 모두 INVALID"
    elif any(per[n]["big"] and not per[n]["overlap"] for n in valid):
        lab, why = MATCH, ", ".join(f"{n} 중앙값 Δr_P {per[n]['median']:.3f}" for n in valid if per[n]["big"])
    elif len(valid) == 2 and all(per[n]["small"] for n in valid):
        lab, why = MISMATCH, f"유효한 두 배율 모두 |중앙값 Δr_P| < 잡음 바닥 {noise:.3f}"
    elif any(per[n]["big"] for n in valid):
        lab, why = UNDECIDED, "변화가 순진 P_X 방향으로 몰림 — ①과 겹침"
    else:
        lab, why = UNDECIDED, "변화가 기준에 못 미침" + ("" if len(valid) == 2 else " (한 배율 INVALID)")
    return dict(label=lab, why=why, per=per, limitation=spec.mv_limitation)


# ================================================================ records and assembly
def transitions(B, Cv, spec) -> dict | None:
    if Cv is None:
        return None
    out = {"pass_to_pass": 0, "pass_to_fail": 0, "fail_to_pass": 0, "fail_to_fail": 0}
    for k in sorted(B):
        a = "pass" if B[k]["r_P"] >= spec.split_r else "fail"
        b = "pass" if Cv[k]["r_P"] >= spec.split_r else "fail"
        out[f"{a}_to_{b}"] += 1
    return out


def candidates(V0, B, F, S, stab, noise, q1_vals: dict, s_c, spec) -> dict:
    return dict(variation=rule_variation(V0, B, F, S, stab, spec),
                floor=rule_floor(V0, B, F, S, stab, q1_vals.get("s_up"), s_c, spec),
                apl=rule_apl(q1_vals.get("apl_mbon05"), B, F, noise, spec),
                reach=rule_reach(V0, B, F, S, stab, spec),
                operating=rule_operating(q1_vals.get("mv_lo"), q1_vals.get("mv_hi"), B, F, noise, spec))


SYMBOL = dict(floor="① 바닥", apl="② APL 억제", reach="③ 편집 도달", operating="④ 작동점", variation="⑤ 변동")


def sentences(cands: dict, spec) -> list:
    out = []
    for k in ("floor", "apl", "reach", "operating", "variation"):
        c = cands[k]
        s = f"{SYMBOL[k]}: {c['label']} — {c['why']}"
        if k == "operating":
            s += f" (한계: {spec.mv_limitation})"
        out.append(s)
    return out


def _labels(cands: dict) -> dict:
    return {k: c["label"] for k, c in cands.items()}


def assemble(q0: dict, q1: dict, s_c: dict, spec) -> dict:
    base = q1.get(spec.cond_names[0])
    if q0["status"] != "OK" or base is None or base["status"] != "OK":
        why = "Q0 INVALID" if q0["status"] != "OK" else "기준선 INVALID"
        cands = {k: dict(label=UNDECIDED, why=why) for k in SYMBOL}
        return dict(candidates=cands, sentences=sentences(cands, spec), invalid=why)
    V0, B = q0["vals"], base["vals"]
    F, S = q0["split"]["F"], q0["split"]["S"]
    stab = fs_stability(F, S, {k: v["r"] for k, v in B.items()}, spec)
    noise = noise_floor(V0, B)
    vals = {n: (q1[n]["vals"] if q1.get(n) and q1[n]["status"] == "OK" else None) for n in spec.cond_names[1:]}
    cands = candidates(V0, B, F, S, stab, noise, vals, s_c, spec)
    border = [k for k in F if k in set(q0["borderline"])]
    F_nb = [k for k in F if k not in set(border)]
    F_nc = [k for k in F if k not in set(q0["cancel"])]
    sens_b = candidates(V0, B, F_nb, S, stab, noise, vals, s_c, spec)
    sens_c = candidates(V0, B, F_nc, S, stab, noise, vals, s_c, spec)
    cancel_base = [k for k in F if B[k]["r_P"] >= spec.split_r and B[k]["r"] < spec.split_r]
    decomp = {g: {ph: {f: med([V[k]["decomp"][f] for k in keys if "decomp" in V[k]])
                       for f in ("dA_X", "dA_Y", "dP_X", "dP_Y")} for ph, V in (("q0", V0), ("base", B))}
              for g, keys in (("F", F), ("S", S))}
    conds = {n: {f: q1[n].get(f) for f in ("status", "reasons", "kc_median", "d6a_over_share", "apl_out_median",
                                            "naive_px_median", "jaccard_median", "condition")}
             for n in spec.cond_names if q1.get(n)}
    records = dict(conditions=conds, noise_floor=noise,
                   transitions={n: transitions(B, vals.get(n), spec) for n in spec.cond_names[1:]},
                   decomposition=decomp,
                   apl_nonkc=rule_apl(vals.get("apl_nonkc"), B, F, noise, spec, role="record"),
                   stop_share={k: dict(q0=V0[k].get("stop_naive"), base_naive=B[k].get("stop_naive"),
                                       base_post=B[k].get("stop_post")) for k in sorted(B)})
    return dict(candidates=cands, sentences=sentences(cands, spec), fs=stab, noise_floor=noise,
                borderline=border, sensitivity_borderline=_labels(sens_b),
                sensitivity_borderline_detail=sens_b, cancel=dict(q0=list(q0["cancel"]), base=cancel_base,
                                                                  note=spec.cancel_note),
                sensitivity_cancel=_labels(sens_c), records=records,
                classification_unstable=stab["unstable"])
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_rules.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/q_rules.py tests/brain/test_q_rules.py
git commit -m "feat(q): q_rules (2) — rules ② ④ with noise-floor threshold and directional guards, sensitivity, assembly"
```

---

### Task 8: `q_runner` and `scripts/run_q.py` — the stage chain, refusals, the reproduction gate, smoke

**Files:**
- Create: `flymon/brain/q_runner.py`
- Create: `scripts/run_q.py`
- Test: `tests/brain/test_q_runner.py`

**Interfaces:**
- Consumes: everything above; `e_runner.summary_git` (import only); `h3_store.git_state`, `code_key` and `canonical`; `load_c3_config`; `odor_real.cap_hz`; `e_pairs.even_situations`; `FlyPool`.
- Produces:
  - `ORDER`, `Q_HASHED_FILES`, `refuse(msg)`, `build_ctx(spec, npz) -> dict`;
  - `Runner(measurer, ctx, spec, summary_path=None, code=None)` with the stages `stage_repro()`, `stage_smoke()`, `stage_s_c()`, `stage_q0()`, `stage_q1(name)` and `stage_records()`;
  - `scripts/run_q.py main(argv) -> int`, which exits 0 for a recorded outcome, 4 when the reproduction gate fails, and 2 on a refusal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_q_runner.py
"""The Q stage chain (Readings 3, 15, 19): repro -> smoke -> s_c -> q0 -> q1 (six conditions, one per call) ->
records; each stage refuses (exit 2, nothing written) when an earlier block is missing, a later one exists, the summary
is uncommitted or a hashed file is dirty; a failed reproduction gate blocks smoke; the mv fallback per side; a gated
condition outside the KC band is INVALID and recorded, not refused."""
import hashlib
import json
from pathlib import Path

import pytest

from flymon.brain import q_runner as QR
from flymon.brain.config import Params
from flymon.brain.q_spec import SPEC
from tests.brain.q_fixtures import KEYS, READOUT, TYPES, Z, fake_result, fake_rows

SHA = {"base": "sha-base", "s_up": "sha-base", "apl_mbon05": "sha-m05", "apl_nonkc": "sha-nonkc", "mv_lo": "sha-lo",
       "mv_hi": "sha-hi"}
EDGES = {"apl_mbon05": 2, "apl_nonkc": 394}


class Scripted:
    def __init__(self, q0, kc_for=lambda p: 0.05, kc_cond=None, alter_repro=False):
        self.q0, self.kc_for, self.kc_cond, self.alter = q0, kc_for, kc_cond or {}, alter_repro
        self.calls, self.last_wall_s, self.last_jobs = [], 1.5, 0

    def run(self, rows, cond, block, seeds):
        self.calls.append((cond.name, block, len(rows)))
        out = []
        for r in rows:
            k = QR.key_str(r)
            i = KEYS.index(k)
            n, n_act = len(seeds["report"]), len(seeds["act"])
            if block == "repro":
                res = json.loads(json.dumps(self.q0[k]))
                if self.alter:
                    res["alpha_reward"] = 0.2 if res["alpha_reward"] != 0.2 else 0.5
                res["q"] = fake_result(n, n_act, 20, -6, 2.0, i)["q"]
            else:
                res = fake_result(n, n_act, px=4 + 3 * (i % 7), dpx=-(1.0 + i % 5), sd=2.0, seed=1000 + i,
                                  sha=SHA[cond.name], edges=EDGES.get(cond.name, 0),
                                  kc=self.kc_cond.get(cond.name, 0.05))
            out.append(dict(key=k, result=res, cache_key=f"ck{i}", cache_file=f"results/q/cache/oracle/{i}.json"))
        return out

    def kc_probe(self, odours, params, strength, seeds):
        return {oid: [self.kc_for(params)] * len(seeds) for oid in odours}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(QR, "git_state", lambda: dict(commit="c", dirty_hashed=[], dirty_other=[]))
    rows = fake_rows()
    q0, man = {}, []
    d = tmp_path / SPEC.q0_cache_dir
    d.mkdir(parents=True)
    for i, r in enumerate(rows):
        k = QR.key_str(r)
        res = fake_result(8, 8, px=20, dpx=(-12 if i >= 12 else -0.5), sd=(1.0 if i >= 12 else 3.0), seed=i, q=False)
        q0[k] = res
        f = d / f"{i:024d}.json"
        f.write_text(json.dumps({"kind": "oracle", "result": res}))
        man.append(dict(axis="b", turn=r["turn"], x=r["x"], y=r["y"], cache_key=f"key{i}",
                        cache_file=f"results/encoder/cache/oracle/{i:024d}.json",
                        cache_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    enc = {"even": {"code_key": "c", "configs": {"k2-norm": {"s": 1.0, "manifest": man}}}}
    ctx = dict(rows=rows, enc=enc, readout=READOUT, z=Z, types=TYPES, params=Params(), n_kc=100,
               max_rate_hz=200.0, cap_hz=1000.0 / 3.0, encoder_keys=lambda sel: [True] * len(sel))
    return ctx, q0


def _runner(ctx, m):
    return QR.Runner(m, ctx, SPEC, code={"key": "q" * 64})


def _doc():
    return json.loads(Path(SPEC.summary).read_text())


def test_full_chain(world):
    ctx, q0 = world
    m = Scripted(q0)
    r = _runner(ctx, m)
    assert r.stage_repro()["passed"]
    sm = r.stage_smoke()
    assert sm["mv"] == [0.8, 1.25]
    assert r.stage_s_c()["s"] == 1.3                     # vmax 1.25 in the fixture rows
    assert r.stage_q0()["status"] == "OK"
    for n in SPEC.cond_names:
        assert r.stage_q1(n)["status"] == "OK", n
    out = r.stage_records()
    doc = _doc()
    assert set(doc) == set(QR.ORDER)                     # canonical JSON sorts keys; the chain order is _require's
    assert set(out["candidates"]) == {"variation", "floor", "apl", "reach", "operating"}
    assert doc["records"]["fs"]["disagree"] >= 0 and len(doc["records"]["sentences"]) == 5
    assert ("base", "q1", 21) in m.calls and ("base", "repro", 3) in m.calls
    assert all(c[2] == 2 for c in m.calls if c[1] == "smoke")


def test_order_and_rewrite_refusals(world):
    ctx, q0 = world
    r = _runner(ctx, Scripted(q0))
    with pytest.raises(SystemExit) as e:
        r.stage_q0()
    assert e.value.code == 2 and not Path(SPEC.summary).exists()
    r.stage_repro(); r.stage_smoke()
    with pytest.raises(SystemExit):
        r.stage_repro()                                  # a later block exists
    with pytest.raises(SystemExit):
        r.stage_q1("base")                               # s_c and q0 missing


def test_repro_failure_blocks_smoke(world):
    ctx, q0 = world
    r = _runner(ctx, Scripted(q0, alter_repro=True))
    out = r.stage_repro()
    assert out["passed"] is False and not all(p["equal"] for p in _doc()["repro"]["pairs"])
    with pytest.raises(SystemExit):
        r.stage_smoke()


def test_repro_fails_on_encoder_key_mismatch(world):
    ctx, q0 = world
    ctx = dict(ctx, encoder_keys=lambda sel: [True, False, True])
    assert _runner(ctx, Scripted(q0)).stage_repro()["passed"] is False


def test_mv_fallback_per_side(world):
    ctx, q0 = world
    base_mv = Params().mv_per_synapse
    kc = lambda p: 0.02 if p.mv_per_synapse < base_mv else 0.05
    r = _runner(ctx, Scripted(q0, kc_for=kc))
    r.stage_repro()
    assert r.stage_smoke()["mv"] == [0.9, 1.25]


def test_kc_band_invalidates_a_gated_condition_and_is_recorded(world):
    ctx, q0 = world
    r = _runner(ctx, Scripted(q0, kc_cond={"s_up": 0.2, "base": 0.2}))
    r.stage_repro(); r.stage_smoke(); r.stage_s_c(); r.stage_q0()
    assert r.stage_q1("base")["status"] == "OK"
    for n in SPEC.cond_names[1:]:
        r.stage_q1(n)
    assert _doc()["q1:s_up"]["status"] == "INVALID"
    out = r.stage_records()
    assert out["candidates"]["floor"]["label"] in ("불일치", "판단 불가")


def test_dirty_summary_or_hashed_files_refuse(world, monkeypatch):
    ctx, q0 = world
    monkeypatch.setattr(QR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    r = _runner(ctx, Scripted(q0))
    Path(SPEC.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(SPEC.summary).write_text("{}")
    with pytest.raises(SystemExit):
        r.stage_repro()
    monkeypatch.setattr(QR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(QR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/q_jobs.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_repro()


def test_script_has_main_guard_and_refuses_wrong_cwd(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_q.py").read_text()
    assert 'if __name__ == "__main__":' in text
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_q", root / "scripts/run_q.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    assert mod.main(["--stage", "s_c"]) == 2
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/brain/test_q_runner.py -q`
Expected: FAIL with `ImportError: cannot import name 'q_runner'`.

- [ ] **Step 3: Write `q_runner.py`**

```python
# flymon/brain/q_runner.py
"""Spec Q's stage chain (Q.5, Q.6.8; plan Readings 3, 15, 19): repro -> smoke -> s_c -> q0 -> q1:<condition> x 6 ->
records, one block each in results/summary/q_reward.json, written only through q_store. Every stage refuses (SystemExit
2, nothing written) when an earlier block is missing, a later block exists, the summary has uncommitted changes, or a
hashed Q file is dirty. A failed reproduction gate blocks every later stage. A condition that fails its gate is written
as INVALID (a record) — Q has no STOP (Q.1)."""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git  # noqa: F401  (tests monkeypatch q_runner.summary_git)
from . import q_pairs, q_records, q_rules, q_store
from .h3_store import canonical
from .h3_store import git_state as _h3_git_state
from .q_measure import Q_MEASURE_FILES, cond_params
from .q_pairs import key_str
from .q_spec import smoke

ORDER = ("repro", "smoke", "s_c", "q0", "q1:base", "q1:apl_mbon05", "q1:apl_nonkc", "q1:s_up", "q1:mv_lo",
         "q1:mv_hi", "records")
Q_HASHED_FILES = tuple(dict.fromkeys(Q_MEASURE_FILES + (
    "flymon/brain/q_spec.py", "flymon/brain/q_pairs.py", "flymon/brain/q_records.py", "flymon/brain/q_rules.py",
    "flymon/brain/q_runner.py", "scripts/run_q.py", "flymon/agent/e_pairs.py", "flymon/agent/encode_grid.py",
    "flymon/agent/e_codebook.py", "flymon/agent/e_spec.py", "flymon/agent/config.py", "flymon/brain/h4_rules.py",
    "flymon/brain/h4_pairs.py", "flymon/brain/d6a.py", "flymon/brain/odor_real.py")))


def git_state() -> dict:
    return _h3_git_state(files=Q_HASHED_FILES)


def refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def build_ctx(spec, npz: str) -> dict:
    """C3 (Params, readout, z — block h4), the H.4 pools as types, the encoder summary, the 21 (b) rows with k2-norm
    odours, the ORN cap, and the encoder-key check when the encoder code key is unchanged."""
    from ..agent import e_pairs
    from ..agent.config import load_c3_config
    from ..agent.e_runner import MEASURE_FILES_E
    from . import odor_real
    from .circuits import Populations
    from .connectome import Connectome
    from .h3_store import code_key
    cfg = load_c3_config(spec.m0d_summary)
    if cfg.readout["P"] != spec.p_type:
        refuse(f"C3's P readout is {cfg.readout['P']}, not {spec.p_type}")
    m0d = json.loads(Path(spec.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    enc = json.loads(Path(spec.encoder_summary).read_text())
    q_pairs.check_strength(enc, spec)
    pops = Populations.from_connectome(Connectome.load(npz))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = q_pairs.b_rows(e_pairs.even_situations(pops), rc, enc, spec)
    enc_code = code_key(npz, files=MEASURE_FILES_E)
    ek = None
    if enc_code["key"] == enc["even"]["code_key"]:
        ek = lambda sel: q_pairs.encoder_key_match(sel, enc, spec, enc_code, cfg.params, cfg.readout, cfg.z, types,
                                                  len(pops.kc))
    return dict(rows=rows, enc=enc, readout=dict(cfg.readout), z=dict(cfg.z), types=types, params=cfg.params,
                n_kc=len(pops.kc), max_rate_hz=float(cfg.params.max_rate_hz), cap_hz=float(odor_real.cap_hz(cfg.params)),
                encoder_keys=ek)


class Runner:
    def __init__(self, measurer, ctx: dict, spec, summary_path=None, code: dict | None = None):
        self.m, self.ctx, self.spec = measurer, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.rows = list(ctx["rows"])

    # ---- chain -----------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return q_store.read_summary(self.summary_path)

    def _require(self, stage: str) -> dict:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed Q files are dirty: {gs['dirty_hashed']}")
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; Q never rewrites an earlier block")
        if i > 0 and not doc["repro"].get("passed"):
            refuse(f"stage {stage}: the reproduction gate did not pass")
        return doc

    def _write(self, stage: str, body: dict) -> dict:
        block = dict(body, stage=stage, code_key=self.code_key, git=git_state(),
                     written_at=_dt.datetime.now(_dt.timezone.utc).isoformat())
        q_store.write_summary_block(self.summary_path, stage, block, [self.ctx["params"]])
        return block

    def _subset(self, spec) -> list:
        return [self.rows[i] for i in spec.pairs_subset] if spec.pairs_subset else list(self.rows)

    # ---- repro (Q.6.8, Reading 3) ----------------------------------------------------------------------------
    def stage_repro(self) -> dict:
        self._require("repro")
        sp = self.spec
        files = q_pairs.q0_files(self.rows, self.ctx["enc"], sp)
        sel = [self.rows[i] for i in sp.repro_pairs]
        fsel = [files[i] for i in sp.repro_pairs]
        base = sp.conditions(sp.strength, 1.0, 1.0)[0]
        got = self.m.run(sel, base, "repro", sp.repro_seeds())
        ek = self.ctx.get("encoder_keys")
        km = ek(sel) if ek else [None] * len(sel)
        pairs = []
        for r, f, g, k in zip(sel, fsel, got, km):
            ref = json.loads(Path(f["path"]).read_text())["result"] if f["ok"] else None
            mine = {kk: v for kk, v in g["result"].items() if kk != "q"}
            pairs.append(dict(key=key_str(r), q0_sha_ok=f["ok"], equal=bool(ref is not None and canonical(mine) == canonical(ref)),
                              encoder_key_match=k))
        shas = sorted({g["result"]["q"]["csc_sha256"] for g in got})
        passed = (all(p["equal"] and p["q0_sha_ok"] and p["encoder_key_match"] is not False for p in pairs)
                  and len(shas) == 1)
        body = dict(passed=passed, pairs=pairs, encoder_key_checked=ek is not None,
                    csc_sha256=shas[0] if len(shas) == 1 else shas, seeds=sp.repro_seeds(), wall_s=self.m.last_wall_s)
        self._write("repro", body)
        return body

    # ---- smoke (Q.6.5, Q.6.8, Reading 15) --------------------------------------------------------------------
    def stage_smoke(self) -> dict:
        self._require("smoke")
        sp, sm = self.spec, smoke(self.spec)
        lo, hi = sp.kc_band
        s_info = q_pairs.cap_s(q_pairs.odours_of(self.rows), self.ctx["max_rate_hz"], self.ctx["cap_hz"], sp)
        odours = {f"{key_str(r)}|{side}": r[f"odor_{side}"] for r in self.rows for side in ("x", "y")}
        kc = {}
        for scale in sp.mv_scales:
            c = sp.conditions(1.0, scale, scale)[4]
            fr = self.m.kc_probe(odours, cond_params(self.ctx["params"], c), sp.strength, sm.kc_probe_seeds)
            kc[str(scale)] = float(np.median([x for v in fr.values() for x in v]))
        inside = [lo <= kc[str(s)] <= hi for s in sp.mv_scales]
        mv = [sp.mv_scales[j] if inside[j] else sp.mv_fallback[j] for j in range(2)]
        sel = self._subset(sm)
        per = {}
        for c in sm.conditions(s_info["s"], *mv):
            got = self.m.run(sel, c, "smoke", sm.q1_seeds())
            rec = q_records.condition_record(got, c, sm, self.ctx["z"], [key_str(r) for r in sel])
            per[c.name] = dict(status=rec["status"], reasons=rec["reasons"], kc_median=rec["kc_median"],
                               edit_edges=rec["edit_edges"], csc_sha256=rec["csc_sha256"], wall_s=self.m.last_wall_s,
                               jobs=self.m.last_jobs)
        detail = dict(kc_probe_median=kc, mv=mv, s_c_preview=s_info, conditions=per, seeds=sm.q1_seeds(),
                      kc_probe_seeds=list(sm.kc_probe_seeds), pairs=[key_str(r) for r in sel])
        q_store.write_json(sp.smoke_detail, detail, [self.ctx["params"]])
        body = dict(mv=mv, mv_fallback_used=[not x for x in inside], kc_probe_median=kc, conditions=per,
                    s_c_preview=s_info["s"], detail=sp.smoke_detail)
        self._write("smoke", body)
        return body

    # ---- s_c (Q.6.4) -----------------------------------------------------------------------------------------
    def stage_s_c(self) -> dict:
        self._require("s_c")
        info = q_pairs.cap_s(q_pairs.odours_of(self.rows), self.ctx["max_rate_hz"], self.ctx["cap_hz"], self.spec)
        self._write("s_c", info)
        return info

    # ---- q0 (Q.3, Q.6.1, Q.6.7, Q.6.8) ------------------------------------------------------------------------
    def stage_q0(self) -> dict:
        self._require("q0")
        sp = self.spec
        files = q_pairs.q0_files(self.rows, self.ctx["enc"], sp)
        results = [json.loads(Path(f["path"]).read_text())["result"] if f["ok"] else None for f in files]
        rec = q_records.q0_record(files, results, sp, self.ctx["z"])
        self._write("q0", rec)
        return rec

    # ---- q1 (Q.3, Q.6.2-Q.6.6) --------------------------------------------------------------------------------
    def stage_q1(self, name: str) -> dict:
        if name not in self.spec.cond_names:
            refuse(f"unknown condition {name}; Q has {self.spec.cond_names}")
        doc = self._require(f"q1:{name}")
        sp = self.spec
        conds = {c.name: c for c in sp.conditions(doc["s_c"]["s"], *doc["smoke"]["mv"])}
        cond = conds[name]
        rows = self._subset(sp)
        got = self.m.run(rows, cond, "q1", sp.q1_seeds())
        repro_sha = doc["repro"]["csc_sha256"]
        same = repro_sha if cond.edit == sp.edit_none and cond.mv_scale == 1.0 else None
        other = repro_sha if cond.edit != sp.edit_none else None
        rec = q_records.condition_record(got, cond, sp, self.ctx["z"], [key_str(r) for r in rows], same_sha=same,
                                         other_sha=other)
        rec.update(wall_s=self.m.last_wall_s, jobs=self.m.last_jobs, seeds=sp.q1_seeds())
        if name == "s_up":
            rec.update(weak=doc["s_c"]["weak"], note=sp.weak_note if doc["s_c"]["weak"] else None)
        self._write(f"q1:{name}", rec)
        return rec

    # ---- records (Q.4, Q.6) ----------------------------------------------------------------------------------
    def stage_records(self) -> dict:
        doc = self._require("records")
        q1 = {n: doc[f"q1:{n}"] for n in self.spec.cond_names}
        out = q_rules.assemble(doc["q0"], q1, doc["s_c"], self.spec)
        self._write("records", out)
        return out
```

- [ ] **Step 4: Write `scripts/run_q.py`**

```python
#!/usr/bin/env python3
"""Spec appendix Q (Q.6 wins): the reward-readout (MBON05) bottleneck diagnosis — a characterisation.

    uv run python scripts/run_q.py --stage repro                    # Q.6.8 reproduction gate (3 encoder caches, H.4 seeds)
    uv run python scripts/run_q.py --stage smoke --workers 4        # smoke seeds 24_008_xxx, mv KC probe, cost
    uv run python scripts/run_q.py --stage s_c                      # Q.6.4 (c) strength (no pool)
    uv run python scripts/run_q.py --stage q0                       # Q0 from results/q/q0_cache (no pool)
    uv run python scripts/run_q.py --stage q1 --condition base      # then apl_mbon05 apl_nonkc s_up mv_lo mv_hi
    uv run python scripts/run_q.py --stage records                  # the five candidates' records (no pool)

Writes results/summary/q_reward.json (one block per stage; commit each before the next) and raw files under
results/q/. Exit 0 for every recorded outcome (an INVALID condition is a record), 4 when the reproduction gate fails,
2 on a refusal (cwd, connectome sha256, chain)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_REPRO_FAIL = 4
POOL_STAGES = ("repro", "smoke", "q1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("repro", "smoke", "s_c", "q0", "q1", "records"))
    ap.add_argument("--condition")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)

    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, code_key, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    if (a.stage == "q1") != (a.condition is not None):
        print("refusing: --condition goes with --stage q1 only", file=sys.stderr)
        return 2

    from flymon.brain import q_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.q_measure import Q_MEASURE_FILES, QMeasurer
    from flymon.brain.q_spec import SPEC
    from flymon.brain.q_store import QCache

    ctx = q_runner.build_ctx(SPEC, NPZ)
    code = code_key(NPZ, files=Q_MEASURE_FILES)
    workers = a.workers or SPEC.workers
    pool = None
    if a.stage in POOL_STAGES:
        pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                       reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
    try:
        m = QMeasurer(pool, QCache(SPEC.cache_dir, code), SPEC, ctx["params"], ctx["readout"], ctx["z"], ctx["types"],
                      ctx["n_kc"])
        r = q_runner.Runner(m, ctx, SPEC, code=code)
        out = r.stage_q1(a.condition) if a.stage == "q1" else getattr(r, f"stage_{a.stage}")()
    finally:
        if pool is not None:
            pool.close()
    if a.stage == "records":
        print("\n".join(out["sentences"]))
    else:
        print(json.dumps({k: v for k, v in out.items() if k not in ("vals", "files", "manifest")}, ensure_ascii=False,
                         default=str)[:2000])
    return EXIT_REPRO_FAIL if a.stage == "repro" and not out.get("passed") else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/brain/test_q_runner.py -q`
Expected: PASS. The wrong-cwd test returns 2 before importing anything heavy.

- [ ] **Step 6: Run the full suite (in the background)**

Run: `uv run pytest -q -x` with `run_in_background: true`. When it finishes, read only the exit code and the failure lines.
Expected: PASS, including `test_q_spec.py`'s AST guard, which now sees every `q_*.py` and `scripts/run_q.py`.

- [ ] **Step 7: Commit**

```bash
git add flymon/brain/q_runner.py scripts/run_q.py tests/brain/test_q_runner.py
git commit -m "feat(q): q_runner and run_q.py — stage chain with refusals, reproduction gate, smoke with mv fallback, records"
```

---

## Self-review (done while writing)

- **Spec coverage:**

  | Spec item | Where it lives |
  |---|---|
  | Q.1 (no verdict; judgement set unused) | Global Constraints, Task 1 AST guard, Task 2 turn guard |
  | Q.2 five candidates | Tasks 6–7 |
  | Q.3 pairs and F/S | Tasks 2 and 5 |
  | Q.3 Q0 | Task 5 `q0_record` |
  | f_X | Task 3 `reach` |
  | Q1 conditions | Task 1 `conditions`, Task 3 edits, Task 4 `cond_params` |
  | (d) KC median, D.6 (a) and APL records | Task 5 |
  | Seeds | Task 1 |
  | Q.4 / Q.6 rules | Tasks 6–7 |
  | `INVALID` | Task 5, Reading 17 |
  | "분류 불안정" | `fs_stability` |
  | Q.5 file set | File Structure |
  | Q.6.1 | r_P, ⑤ on dV, decomposition, A·Y cancel |
  | Q.6.2 | fixed arms, W_X, ③, saturation in ① |
  | Q.6.3 | paired Δr_P, noise floor, > 5 rule, AUC definition |
  | Q.6.4 | `cap_s`, weak rule, KC gate |
  | Q.6.5 | mv_scale, fallback, directional guards, limitation sentence |
  | Q.6.6 | `apl_to_mbon05_zero` primary, nonkc as record, ratio, guard |
  | Q.6.7 | borderline sensitivity |
  | Q.6.8 | smoke seeds, collector (Task 1), Q0 cache, storage and resume, repro gate, z from block h4 |

- **Placeholders:** none. Every code step has the full code.
- **Type consistency:**
  - `key_str` is defined in `q_pairs` and used by `q_records`, `q_measure` and `q_runner`.
  - The `vals` fields used in `q_rules` (`naive_px`, `r`, `r_P`, `dV_mean`, `dV_sd`, `ratio`, `dpx_fixed`, `px_after_fixed`, `W_X`, `decomp`, `stop_naive`, `stop_post`, `f_X`) are all produced by `pair_values`.
  - `QSpec.cond_names` matches `ORDER`'s `q1:*` blocks.
  - `QMeasurer.run`'s return value has the shape that `condition_record` and `Runner` read.
- **Review Focus:** each of the five items has a test in its owning task: Task 5 (ratio None, non-finite r), Task 6 (Spearman / Wilcoxon degenerate, ratio counts), Task 4 (corrupt or foreign entries, resume), Task 4 (guard), Tasks 1–2 (judgement set).

## Runs (controller)

All runs are done by the controller, from the repository root (`/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot`), with Bash `run_in_background: true`, after the full suite passes at the final implementation commit. Subagents never run these steps. When a run finishes, read only its exit code and the last lines of its output. No rule or number changes after any run. Each block is committed before the next stage, and the stage refuses otherwise. `results/summary/` is tracked; `results/q/` is git-ignored.

1. **Reproduction gate (Q.6.8):**
   1. Run `uv run python scripts/run_q.py --stage repro`. It runs 3 pairs × the base condition on H.4 seeds 500–615 and takes about 10–20 min at 3 busy workers.
   2. Expect exit 0, `passed: true`, every pair `equal` / `q0_sha_ok` / `encoder_key_match` true, and one `csc_sha256`.
   3. On exit 4, stop. Diagnose with `superpowers:systematic-debugging`, have a reviewed fix task, and rerun. The block can be rewritten because no later block exists yet.
   4. Commit: `git add results/summary/q_reward.json && git commit -m "results(q): reproduction gate — 3/3 encoder caches reproduced bit for bit (H.4 seeds)"`.
2. **Smoke (Q.6.8, Q.6.5):**
   1. Run `uv run python scripts/run_q.py --stage smoke --workers 4`. It does three things:
      - the KC probe: 42 odours × 4 seeds at × 0.8 / × 1.25;
      - all 6 conditions × 2 pairs on smoke seeds 24_008_000–24_008_010;
      - it writes `results/q/smoke.json`.
   2. Check:
      - `apl_mbon05` has `edit_edges == [2]`;
      - `apl_nonkc` has edges > 0 (about 394);
      - the base and s_up rows carry the repro sha;
      - `mv` and `mv_fallback_used`.
   3. Re-estimate the Q1 cost from `wall_s`, scaling for 8/8/8 seeds and 21 pairs. The encoder baseline estimate is about 2–3 h for Q1 in total. Report the estimate in one line.
   4. Commit: `git commit -m "results(q): smoke — mv <lo>/<hi> (fallback <…>), edit edges 2 / <n>, cost estimate <h>"`.
3. **s_c (Q.6.4), before any Q1 run:**
   1. Run `uv run python scripts/run_q.py --stage s_c`. It takes seconds and needs no pool.
   2. Record s_c, vmax and `weak`. If `weak` is true, ① is already fixed at 판단 불가; s_up still runs as a record.
   3. Commit: `git commit -m "results(q): (c) strength s_c = <s> (ORN cap over the 21 pairs' odours; weak <bool>)"`.
4. **Q0:**
   1. Run `uv run python scripts/run_q.py --stage q0`. It needs no pool and reads only `results/q/q0_cache/`.
   2. Expect status OK, F 12 / S 9 and 3 borderline pairs.
   3. Commit: `git commit -m "results(q): Q0 — F 12 / S 9 from the encoder k2-norm caches (digest OK)"`.
5. **Q1, one condition per run, one pool at a time, in this order:** base, apl_mbon05, apl_nonkc, s_up, mv_lo, mv_hi.
   1. For each condition, run `uv run python scripts/run_q.py --stage q1 --condition <name>`. Each takes about 20–30 min at 16 workers.
   2. If a run is interrupted, rerun the same command. It resumes from `results/q/cache/oracle/`.
   3. Commit after each: `git commit -m "results(q): Q1 <name> — <OK|INVALID (reason)>, KC median <x>"`.
   4. An `INVALID` condition is recorded, never rerun with changed rules.
6. **Records:**
   1. Run `uv run python scripts/run_q.py --stage records`. It needs no pool.
   2. Commit: `git commit -m "results(q): Q records — ① <label> ② <label> ③ <label> ④ <label> ⑤ <label>"`.
7. **Spec result section:**
   1. Write `### Q.7 결과 (2026-10-xx, 특성화 — 판정 아님)` after Q.6 in `docs/superpowers/specs/2026-09-14-flymon-design.md`. It contains:
      - the run commits;
      - the repro result;
      - s_c and weak;
      - mv as used;
      - each condition's status, KC median, D.6 (a) share and APL;
      - for each candidate, its label with the numbers behind it (AUC, ρ, median Δr_P, p, threshold, noise floor);
      - the borderline and A·Y-cancel sensitivities;
      - the F/S stability, and "분류 불안정" if it applies;
      - the transition tables;
      - the decomposition;
      - (b-record) `apl_to_nonkc_zero`;
      - the ④ limitation sentence verbatim;
      - Q.5's interpretation limits: 21 pairs, so a 2–3 pair difference is not a shape; an agreement is evidence for choosing a lever in R, not proof of a cause; the scope is C3 · E-grid k2-norm · even (b).
   2. Commit: `git commit -m "docs(spec): Q.7 — Q records: ① … ⑤ … (characterisation)"`.
8. **README ledger:**
   1. Add one Korean line next to the encoder-redesign entry (around line 180) and one English line in the "In English" section, in the existing format.
   2. Commit: `git commit -m "docs(readme): appendix Q ledger (ko/en) — reward-readout diagnosis records"`.
   3. Report to the user. Do not push unless the user asks; the FlyMon push rule needs `gh auth switch --user lyutvs`. Appendix R (lever choice) is the user's decision.
