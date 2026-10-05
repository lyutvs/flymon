# Spec W — F v4 Learning Test on the Combined Lever (F.2 R · N · RN on L_V, z_V, per-fly joint satisfaction, a design chosen by W's own operating characteristic): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what spec appendix W needs (precedence **W.9.9 > W.9.8 > W.9.1–W.9.7 > W.0–W.8**), and nothing else — the M2 learning unit (spec 5) judged with the real learning rule on V's combined lever:
- **Stage 0 (W.9.9 순서 0)** — the W verdict code (`w_verdict`, never the frozen F v3 `verdict.py`, G.7's defects fixed: NaN / probes < 2 → INVALID → `STOP_MACHINE`, the scheduled F as denominator, missing data refused, BAND → 2K coded, per-fly joint satisfaction with q), `w_oc` (a probe-level simulator that runs the W verdict code itself; gate-vector calibration by bisection; pair-level parametric bootstrap; simultaneous one-sided limits; the design rule of H1 / W.9.9), the numeric tables (protocol, seeds, designs, OC), the synthetic validation of W.9.9 P2-11 and the OC's compute time — committed before any measurement.
- **Reuse (W.3 1)** — V's blocks (z `928eaad`, KC input `7dc199d`, set `cf0b3b2`, judgement `a279a56` = SELECTED) on V's keys, V's own reuse condition (R `3c2699c7…`, T `7255f872…`, U = V `8a4e0930…`), z_V as declared — else `STOP_REUSE`; re-checked at every later stage (exit 7).
- **Path (W.3 2, W.9.6 P2-13, W.9.9 P2-9)** — W's sequential job (a W-only measurement file `w_measure.py` on `u_measure.u_rig`) with P's punish-arm settings reproduces V's gate-② arm rows bit for bit (L_V and C, both directions, the first 2 seeds); its naive probes reproduce V's oracle raw `report.pre` bit for bit (L_V and C, V's judgement rows 0–2, V's report seeds); the reward path moves the taught weights and lowers MBON05(X) — else `STOP_W_PATH_REPRO`.
- **Pilot (W.3 3, W.9.4, H6)** — the POOL even pilot (V's L_V-testable (b) even pairs + a|4 Rock Slide|Strength = 16 pairs; 8 flies × 8 probes; R · N · RN; seeds 40_000_000 / 41_000_000 blocks), records only ("탐색"), then `STOP_PILOT_NO_EFFECT` (H6).
- **OC (W.9.3, H1, H4, W.9.9)** — one design (q ∈ {0.5, 0.625, 0.75}, K ∈ {8, 16}, F 8–32; k cap 8) and the alternative ranking, or `STOP_OC_UNREACHABLE`.
- **Smoke and cost ledger (순서 5)**, the **worst-case budget gate (5a, 24 h; C dropped first, then the alternative designs, else `STOP_BUDGET`)**, the **main set** (V's generator, turns 306–1985, V's set as used, V's KC filter, no axis cap: (b) 167 · (a) 82) digested before any oracle on it, the **oracle screen** (249 pairs, 24_700_xxx; the set is used from here), the **estimate update**, the **naive screen** (judgement probe seeds, oracle-testable pairs in declared order until 8 gate pairs), the **gate pairs** (`STOP_FEW_PAIRS` < 4), the **learning measurement** (gate pairs × F × R / N / RN, K probes, seeds 26_000_000 + c·4_000 + f·100 + k / 28_000_000 + c·40_000 + f·1_000 + t), the **BAND probes** (the same flies re-trained, probes K..2K−1), the **records** (C on h4 z, the plasticity-off control), the **seal** (`~/flymon-archive/w/`) and **one judgement** (H7's order, W.7 / W.9.7 / W.9.8 sentences).

W is **one F v4 learning judgement on the combined lever**. Its STOP labels and verdict go to the user unchanged.

**Architecture:**
- New files only: `flymon/brain/w_*.py`, `scripts/run_w.py`, `tests/brain/test_w_*.py` and `tests/brain/w_world.py`. The one exception is one `MODULES` entry in `tests/brain/test_p_spec.py` (Reading 1).
- **One new measurement file, W's own**: `flymon/brain/w_measure.py` (the sequential job `w_learn_job` on `u_measure.u_rig`, unchanged U code; `WMeasurer` with one cache entry per pair · fly · brain). The **W measurement key** = U's measurement key's files + `w_measure.py` (`w_measure.w_measure_key`). The oracle screen runs U's unchanged oracle copy through `u_measure.UPool` and R's unchanged `RMeasurer`, behind W's cache.
- Modules:
  - `w_spec`: every W number (`WSpec`, a plain frozen dataclass — no V seed in any field), seed formulas, phases, designs, `smoke()`.
  - `w_pairs`: the pilot rows, the main set (V's `_Gen` + V's set as used), its summary / check, the checked rows with candidate numbers.
  - `w_verdict`: the judgement code, vectorised (`w_oc` runs it on simulated experiments; the judge on the measured ones).
  - `w_oc`: fit, generator, calibration, evaluation, bootstrap, selection, records, synthetic validation.
  - `w_records`: job rows → verdict arrays, machine reasons, path comparisons, the pilot record, the cost model.
  - `w_rules`: gate decisions and every sentence.
  - `w_store`: the writer, `WCache`, `VReadCache`.
  - `w_measure`: the measurement file (above).
  - `w_runner`: the stage chain.
- `scripts/run_w.py` is the CLI.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`, `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **W** (W.0–W.8) with W.9 (W.9.1–W.9.7), **W.9.8** (H1–H9) and **W.9.9** (P0-1, P1/P2/P3, 순서). W reuses spec **5** (the M2 learning unit), appendix **F** (F.1 sequential reading, F.2 protocol, F.5 gates / BAND / mechanism control / ±∞ limits, F.6 fixtures, F.7 consequences), **G** (G.2 even pilot only, G.5 RN, G.6 targets and knobs, G.7 defects), **J.12** (the sequential F.2 precedent), and **V** (V.3, V.9, V.10: the lever, z_V, the KC filter, the set generator, gate ②'s rows, the judgement raw).

| W.9 section | Overrides / adds |
|---|---|
| W.9.1 | per-fly joint satisfaction (q), BAND re-measure (same flies, 2K), NaN → INVALID → STOP_MACHINE |
| W.9.2 | the mechanism control on every gate pair, same flies and probes (RN for punishment) |
| W.9.3 | `w_oc` committed before the pilot (4000 reps, seed 42_000_000), 3-level model, synthetic validation, F(k) |
| W.9.4 | J.12 disclosure; `STOP_PILOT_NO_EFFECT` (replaced by H6); pilot records |
| W.9.5 | naive screen with the judgement probe seeds (F × K), reused as pre (bit-identical); gate pairs in turn order, (b) first; minimum 4 judgeable |
| W.9.6 | 24 h budget and ledger; new verdict code; job unit pair · fly · brain; RN from pre; candidate-numbered seeds; C3 timings (W.1 table); P2-13 bitwise naive repro |
| W.9.7 | sentences (q in PASS; "(마리별 동시 충족 집계)" in FAIL; new STOPs) |
| W.9.8 | H1 designs chosen by the OC; H2 mechanism failure = pair FAIL, no STOP_PROTOCOL; H3 order; H4 probe-level OC running the verdict code; H5 ±∞ by direction; H6 pilot stop; H7 order; H8 INVALID_RUN; H9 cache key, BAND as its own artefact |
| W.9.9 | P0-1 gate-vector targets (min = 1.5 power / max = 0.5 false pass) by bisection; P1-2 RN1 copied in the simulation; P1-3 bootstrap limits; P1-4 wall-clock cost, alternative ranking; P1-5 budget gate before the set digest; P1-6 drift / spillover / low-dim REs / cluster grid; P2-7 k cap 8; P2-8 nested generation; P2-9 reward-path check, BAND mechanism after 2K; P2-10 naive fixed at the screen; P2-11 synthetic pass criteria; P2-12 disclosures; P3-13 pairs beyond the cap unmeasured; 순서 0–14 |

**Task count and review cost:** 8 tasks. Each task is one implementer run and one reviewer run (`sdd-implementer` / `sdd-reviewer`), then one final whole-branch review: **17 subagent runs**. Expected size: ~2,820 lines of new code and ~1,690 lines of tests (85 tests). Dependencies: Tasks 2–5 need Task 1's `w_spec`; Task 3 needs Task 2; Task 4 needs Task 2; Task 5 needs Task 4 (its tests use `w_records`); Task 6 needs 1–5; Task 7 (the CLI) needs Task 6; Task 8 (the second half of `w_runner`) needs 6–7 (`decision_key` hashes `scripts/run_w.py`). The slow tests are `test_w_measure.py` (real connectome, ~2.5 min) and the runner worlds (~1 min each).

**Validation done while planning (scratch worktree, removed afterwards):** every W test passed (85 tests over the twelve W test files), the spec-collector suites (`test_e_spec_store`, `test_p_spec` with W's `MODULES` entry, `test_q_spec` … `test_v_spec`, `test_w_spec`, `test_run_m`; 249 tests with the W files) passed, Task 6's tests passed with only Task 6's part of `w_runner.py` in place, and the full repository suite ran in the scratch tree: everything passed except 8 tests of `tests/test_run_l.py` / `tests/test_run_m0d_h3.py` that read git-ignored `results/m0d/` files absent from a fresh worktree (FileNotFoundError on `results/m0d/k/run/cache/…`, `results/m0d/diag/…`; no W file involved) — Task 8's Step 5 runs the suite in the real tree. On the real data: the shared, T and U measurement keys are R's `3c2699c7…`, T's `7255f872…` and U's `8a4e0930…`, and the W measurement key was `761274e0…` at the draft; W's job with P's punish-arm settings equals `u_arm_job` (L_V) and `r_arm_job` ("none") field for field (2 trials), W's naive probe equals `presentation.decide`'s per-type sums on V's report seeds, RN1 = R1 bit for bit (counts and weights) while N and RN2 differ, and a rig whose combined edit is not applied measures other counts (Task 5's tests); the main set reproduced W.0's facts — **(b) 167 · (a) 82 = 249 candidates** on turns 306–1967 (skips: used 4,516, in-set 1,405, POOL-only 1,099, kc_input 484, cap 137), V's set reproducing block `cf0b3b2` first; the pilot is 16 pairs (15 (b) + a|4). The real CLI ran **`stage0` (PASS: every synthetic check within W.9.9 P2-11's limits — zero effect 0.000, d′ 4 effect 1.000, one gate 0.000, gates anti-correlated across flies 0.000, P0-1's simple normal model 0.351 → 0.0945 from F 8 to 32; OC compute estimate 1,472 s ≈ 25 min) and `reuse` (PASS)** from a scratch commit. **`path` and every later stage were not run** (they are the declared run). **Timing probe (planning estimate, not a result):** one in-process W job fragment on POOL even pair a|0|Surf|Earthquake under L_V (4 presentations, 3 PAM08 trials): 1.98 s per presentation and 2.09 s per trial (≈ 1.4–1.5 ms per engine step single-process) — a full K = 8 job (40 trials + 3 stages × 8 probes × X, Y ≈ 123k steps) ≈ 3 min single-process. V's own pool runs put the 16-worker step cost at ≈ 2.7 ms (gate ②: 384 arm jobs in 1,771 s), i.e. ≈ 5.5 min per K = 8 job in the pool: the pilot (384 jobs) ≈ 2.2 h; learning, the BAND probes and C each ≈ 2.2 h at F = 16, K = 8; V's `jm:L` oracle ran 64 pairs in 2,817 s (704 s per 16-worker round), so the 249-pair screen ≈ 3.1 h; a naive-screen job (K = 8) ≈ 1 min, so the worst-case screen at F = 16 ≈ 4.2 h — and the worst case is plausible, since V's set had only 3 of 64 pairs both oracle-testable and naive-balanced (W.0). These are only for sizing; the pilot's and the smoke's measured costs replace them (Readings 11–12), and the 24 h budget gate (5a) is expected to bind for large F.

## Global Constraints

- **Commits carry no trailers.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No R, S, T, U or V file changes; one new measurement file only.** `r_measure.R_MEASURE_FILES` stays R's `3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc`; T's measurement key `7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374`; `u_measure.u_measure_key` `8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523` (test in Task 1). `t_measure.py`, `h3_spec.py`, `u_measure.py`, `verdict.py` (the frozen F v3 file) and every `r_*`, `s_*`, `t_*`, `u_*`, `v_*` module, script, test, fixture or helper are never edited. The only edit outside `w_*` / `run_w.py` / `w_world.py` is Reading 1's `MODULES` entry.
- **Never edit** `flymon/agent/*`, `q_*`, `p_*`, `o_*`, `n_*`, `h3_*`, `h4_*`, `k_jobs`, `d6a`, `plasticity`, `presentation`, `fly_pool`, `engine_cpu`, `flymon/battle/pool.py`, any `scripts/run_*.py` other than `run_w.py`. No module global of another track is monkeypatched at runtime (tests may monkeypatch).
- **W's only job is `w_measure.w_learn_job`** (test in Task 7); no W file loads code by path (`importlib.util`, `runpy`, `exec`) or names the frozen `m2-learning-f-v3` directory (test in Task 7). The oracle screen uses `u_measure.UPool` and `r_measure.RMeasurer` unchanged.
- **Numbers live only in `w_spec`** (W's) or are read from `v_spec.SPEC` / the C3 config / V's summary (V's). No W file holds an integer of another track's seed block (24_000_000–24_699_999, 24_710_000–25_999_999, 800_000–800_199, 20261004 / 20261005) — test in Task 1. `WSpec` carries no V seed in any field (test), so every repository seed collector sees W's blocks only.
- **The main set is reached only through ctx's callables, regenerated and checked against block `set`** (`w_pairs.main_rows`); no stage before `oracle` measures it (the pilot, smoke and path use the POOL even pairs, V's raw and V's rows only).
- **Raw data of other tracks is read-only for W:** `results/summary/{r,t,u,v}_lever.json`, `results/v/cache/r_arm/` (V's gate-② rows, by content key through `VReadCache` under U's key), `results/v/cache/r_oracle/` (V's judgement raw, through V's `jm:L` / `jm:C` manifests, sha-checked), `results/summary/p_learning.json`, `results/summary/encoder_grid.json`, `results/summary/m0d.json`.
- **Writes are guarded.** Raw files under `results/w/` (git-ignored by `results/*`), the summary `results/summary/w_learning.json` (tracked). Anything else is refused with SystemExit 2. The archive copy goes only under `~/flymon-archive/w/<seal id>/`, never over an existing directory. Every write is atomic.
- **Scripts:** tests find the repository with `ROOT = Path(__file__).resolve().parents[2]`; `scripts/run_w.py` ends with `if __name__ == "__main__":`; pools are never started from a heredoc or `python -c`.
- **Subagents never run a real or smoke stage and never write under `results/`.** Tests write only under `tmp_path`. The controller runs every stage (section "Runs (controller)").
- **Never change rules or numbers after results.** No W number, rule or sentence changes after a stage has run on data; a defect found after a stage is a reviewed fix task before the next stage (the decision code is pinned at the seal).
- **Numbers from W / W.9, verbatim:**
  - **Engine and readout (W.1):** C3 + L_V = `u_edit(0.0, "chain_entry")` (CSC `2d359b8b…`, `edit_edges` 2, chain entry 7 / 2 / 2); readout V = z_A − z_P with **z_V A 16.917 / 12.484, P 80.167 / 29.775** (V block z `928eaad`, compared at 3 decimals; the block's full floats are used); C (records) on block h4's z.
  - **Protocol (F.2, W.9.6 P2-12; Reading 3):** R = pre → PAM08 × 20 → R1 → PPL105 × 20 → R2; N = no DAN × 20 → N1 → × 20 → N2; RN = PAM08 × 20 → RN1 → no DAN × 20 → RN2 (from its own pre); pulse 400 ms, train settle 800 ms, gap 200 ms (`o_jobs.train_x`), strength 1.0; probes X then Y per seed, settle 800 · read 600 · window 200 (V's oracle windows), plasticity off; C3 Params unchanged.
  - **Verdict (W.4, W.9.1, H2, H5, H7):** fly gates reward level d′(ΔV_R1) ≥ +1, punishment drop d′(ΔV_R2 − ΔV_R1) ≤ −1, reward association d′(ΔV_R1 − ΔV_N1) ≥ +1, punishment association d′((ΔV_R2 − ΔV_R1) − (ΔV_RN2 − ΔV_RN1)) ≤ −1; margin rounded to 9 digits; BAND width **0.2** inclusive (0.80 BAND, 0.79 FAIL); ±∞ by direction; NaN or probes < 2 → INVALID; q_sat = satisfied / scheduled F; PASS q_sat ≥ q; BAND q_sat + band / F ≥ q → 2K (PASS stays, else FAIL); mechanism control fly medians ≥ **0.75** both (reward MBON05(X) R1 < N1; punishment MBON13(X) R2 − R1 < RN2 − RN1; ties ½), after 2K for a BAND pair, failure = pair FAIL; overall: machine / INVALID → STOP_MACHINE → FAIL → BAND left → UNDECIDED → judgeable < **4** → UNDECIDED → all PASS → PASS; naive pooled |d′| < **0.5**.
  - **Designs (H1, W.9.9 P2-7):** q ∈ **{0.5, 0.625, 0.75}**, K ∈ **{8, 16}** (BAND 2K), F **8–32**, k cap **8**, k **4–8**; G.6 targets power ≥ **0.80** at min gate d′ **1.5**, false pass ≤ **0.05** at max gate d′ **0.5**; bootstrap **200** draws, simultaneous one-sided **95%** limits, worst over cluster grid **0 / 0.5 / 1**; envelope F..F+3, F **29–32** alone; cost = total wall clock, ties larger q, then smaller K.
  - **w_oc (W.9.3, H4, W.9.9):** **4000** reps, seed **42_000_000**, calibration **2000** draws, tolerance **±0.02**, at most **40** bisection steps.
  - **Seeds (W.9.6 P2-11):** main probe **26_000_000 + c·4_000 + f·100 + k**, training **28_000_000 + c·40_000 + f·1_000 + t** (c < 300, f < 32, t < 40); pilot **40_000_000 + j·4_000 + f·100 + k**, **41_000_000 + j·40_000 + f·1_000 + t** (j < 25); smoke **42_100_000 block**; OC **42_000_000**; oracle screen **24_700_000+i / 24_700_100+i / 24_700_200+i** (i < 8).
  - **Pilot (W.3 3, H6):** V-even L_V-testable (b) + a|4 Rock Slide|Strength, **8** flies × **8** probes; stop if not (reward assoc ≥ 0.5 share ≥ 0.5 and punish assoc ≤ −0.5 share ≥ 0.5), or > half the pairs reversed, or X·Y-floor probe share > **0.5**.
  - **Main set (W.2, W.0):** V's generator, turns **306–1985**, V's set rows used, V's KC filter [0.03, 0.15] on both engines, no axis cap, **(b) 167 · (a) 82**.
  - **Budget (W.9.6 F, P1-4, P1-5):** **24 h** from stage 0, C dropped first, then the alternative designs in cost order, else `STOP_BUDGET`.
  - **Reuse (W.3 1):** shared **`3c2699c7…`**, T **`7255f872…`**, U = V **`8a4e0930…`**; V blocks z `928eaad`, KC input `7dc199d`, set `cf0b3b2`, judgement `a279a56` (SELECTED).

## Readings of the spec (decided here — the controller's rulings)

None of these changes a judgement rule, a gate threshold or a STOP sentence; each fixes plumbing the spec leaves to the implementation. Ambiguities that would change one are under "OPEN — needs user".

1. **`test_p_spec.py` gets one entry.** Its `test_every_spec_module_is_enumerated` fails as soon as `flymon/brain/w_spec.py` exists. Task 1 adds `"flymon/brain/w_spec.py": "flymon.brain.w_spec"` to `MODULES` (the pattern Q … V used). It is the only edit to a non-W file; the reviewer flags it for the user.
2. **`WSpec` is a plain dataclass, not a `VSpec`.** Every repository seed collector (`test_p_spec._collect`, `test_e_spec_store._module_seeds`, each track's "declared without me" test) walks every field whose name holds "seed", nested dataclasses included. A `VSpec` subclass (or a nested P spec) would re-declare V's 24_600_xxx / 25_400_xxx blocks and break V's own collision test. So `WSpec` holds W's numbers only (its seed fields carry W's blocks: 26_000_000, 28_000_000, 40_000_000, 41_000_000, 42_000_000, 42_100_000, 42_110_000, 42_150_000, 24_700_xxx — test), and V's numbers (the lever string, the oracle windows, alphas, `testable_min`, gate ②'s P spec, `judge_seeds`, the CSC shas) are read from `v_spec.SPEC` where they are used.
3. **The protocol table (W.9.6 P2-12, "W.1 부록 표"; committed in block `stage0`).** C3's `Params` unchanged (read from the C3 config at run time and listed in the block: learn rate 3e-4, traces 200 / 100 / 200 ms, scales 40 / 20, min weight 0.2, recovery 0, kc_kc 0, DAN drive 70 mV, core 0.2). Training = `o_jobs.train_x` unchanged (reset to the trial's seed, settle 800 ms with the weights frozen, the DAN for 400 ms — F.2's pulse — then the gap 200 ms, recovery after a pulse), 20 trials per phase, X alone. Probes = `n_jobs._present` (reset, traces, drive, DAN quiet, present, settle 800 + read 600, window 200), X then Y per probe seed, plasticity off — the presentation sequence of V's oracle `decide` and of P's arm job, which is what makes W.3 2's bit-for-bit reproductions possible. **Strength 1.0** for probes and training (the E-grid k2-norm operating strength of every V / W odour; P / O used one strength for both too). PAM08 / PPL105 are H.4's readout compartments (W.0).
4. **The job (W.9.6 P2-10, H9).** One unit = pair × fly × brain. Every brain runs from the naive rig and measures its own pre (so pre(R) = pre(N) = pre(RN) = the naive screen's probes is a machine check, and RN never shares R's state — RN re-runs the reward phase from pre with R's training seeds, so RN1 = R1 bit for bit by determinism, and RN's weights after the reward phase must equal R's). Phase i trains with `train_x(seed = f, seed_base = base_i, seed_stride = 1000)`, i.e. train seed = base_i + f·1000 + t′; base_1 = base_0 + 20 makes phase 2's seeds t = 20..39 of W.9.6's formula. After each phase: the plastic weights' sha256, the weight fractions (all, the punishment core, the reward core) and the phasic-dopamine integral (`o_jobs.DaMeter`).
5. **The path gate (W.3 2, W.9.6 P2-13, W.9.9 P2-9).** Measured afresh (no cache), three parts, raw to `results/w/path.json`:
   - (i) "보상 0회·처벌 12 × 800 ms·P 프로브 설정": W's job with one phase [PPL105, 12, P's train seed base], probe seed = the arm seed, P's strength 0.35 and windows / timings (`h4.oracle_window`, `teach_present_ms` 800, `teach_gap_ms`, `teach_window.settle_ms`) — compared with V's gate-② **punish-arm** rows (the arm that trains; P's arms are plastic / frozen / punish), L_V and C, both directions, the first **2** seeds (25_400_000, 25_400_001) = 8 rows, read from V's cache by content key (`VReadCache` under U's measurement key, through an `RMeasurer` that never runs a job). Compared: pre and post X / Y presentations (wall time aside), w0 / w_post sha, the three weight fractions, the dopamine integral, the CSC.
   - (ii) W's naive probes (no phase) on V's judgement rows 0–2 (declared order) under L_V and C, V's report seeds 24_600_200+i, against V's oracle raw `report.pre` (MBON13 / MBON05 × X, Y, every seed) read through V's `jm:L` / `jm:C` manifests (sha-checked) — mandatory, no substitute (P2-13).
   - (iii) the reward path: pilot pair 0, fly 0, R's reward phase only, the pilot's seeds — the plastic weights moved, the reward core's weight fraction < 1, MBON05(X)'s probe mean fell.
   - A job whose edit / CSC / edge count / chain-entry block is not the declared one is `INVALID`; any difference is `STOP_W_PATH_REPRO` with the first failing check in the sentence. The unit tests (Task 5) pin the same equalities in miniature and the "edit not applied" mutation.
6. **Seeds (W.9.6 P2-11).** The candidate number c is the main set's declared position (Reading 15); j the pilot position. BAND's added probes are k = K..2K−1 (≤ 31 when K = 16) — W.9.6's "k < 16" was written before H1 allowed K = 16 with a 2K re-measure; the formula is unchanged and k < 100 stays inside the fly stride (layout test). Smoke: probes 42_100_000 + f·100 + k, training 42_110_000 + 20·phase + f·1_000 + t′, oracle 42_150_000+i / +100 / +200 — all inside the declared 42_100_000 block and disjoint (test). `w_oc`'s random streams are `SeedSequence([42_000_000, tag, …])` (no Python `hash`).
7. **Pilot and H6 (W.3 3, W.9.4, H6).** Pilot pairs = V block even's L_V-testable (b) pairs + `a|4|Rock Slide|Strength`, in the even rows' declared order (16 on the real data). H6 is read on each pilot pair's fly medians: (i) the share of pairs with reward association ≥ 0.5 and the share with punishment association ≤ −0.5, both ≥ 0.5 (inclusive); (ii) "부호가 반대" = reward association median < 0 or punishment association median > 0, "과반" = a share > 0.5; (iii) the probe share (every R-brain probe of every pilot pair) with MBON05(X) and MBON05(Y) both 0 at R1, or MBON13(X) and MBON13(Y) both 0 at R2, > 0.5. 3a lives in block `pilot`'s outcome (a machine mismatch is `INVALID`). The exploratory verdict (q 0.75, F 8, K 8) is recorded with the label "탐색" and never gates.
8. **Stage 0 (W.9.9 순서 0).** Block `stage0` holds the decision files' shas (`w_verdict`, `w_oc`, `w_spec`, `w_measure`), the tables (Reading 3's protocol with C3's values, the seed formulas, the designs, the OC settings), the synthetic validation (W.9.9 P2-11: zero effect ≤ 0.02, every gate d′ 4 ≥ 0.98, one gate only ≤ 0.02, gates anti-correlated across flies ≤ 0.02, P0-1's simple normal model not rising from F 8 to 32) and the OC's compute time on a synthetic pilot (point estimates and records at full size, the bootstrap timed on 4 draws and scaled to 200). PASS iff every synthetic check passed. The mutations W.9.9 P2-11 requires (RN1 one bit, the fly denominator, a deleted gate, the ±∞ direction) are pytest mutation tests (Task 2) committed with the code.
9. **The OC's model (W.9.3, H4, W.9.9 P1-6; committed at stage 0).** Probe noise = **resampling of the pilot's residual vectors** (each pilot fly's probe deviation from its slot means across the 6 slots × 2 cells × 2 odours, scaled √(K/(K−1))): one of the two allowed choices, and the one that keeps the paired-noise correlation between slots of the same probe seed. Pair random effect = 2 × 2 intercepts ~ N(0, Σ_pair) **conditioned on naive balance** (|E ΔV_pre| / sd_pre < 0.5 — gate pairs are pairs that passed the naive screen, and the OC counts no naive dropout); the population pair m0 averages X and Y per cell. The cluster random effect (variance g·Σ_pair, g on the grid) is **shared by every gate pair of an experiment** (the gate pairs' cluster assignment is unknown before the set digest; sharing is the extreme), worst g taken per scenario. The fly random effect (same 2 × 2 structure) applies to the post-training slots only (one naive brain); its covariance is the between-fly covariance of post-slot fly means minus the probe-noise share, made PSD. Fixed stage effects: N1 − pre, N2 − N1, RN2 − RN1 (presentation drift, pilot means). Y spillover s = (mean Y change) / (mean X change) of the reward (MBON05, R1 − N1) and punishment (MBON13, (R2 − R1) − (RN2 − RN1)) effects, 0 below, **1 when the pilot's X change is not a depression**. Counts = max(0, rint(mean + residual)).
10. **Calibration (W.9.9 P0-1).** The "참 d′" of a gate = mean / sd over 2000 residual draws of the population pair with every random effect 0 (common random numbers within a calibration). a is bisected on [0, a_floor] (a_floor puts every R1 MBON05(X) at 0), then b on [0, b_floor] with a fixed; "포화" = the target not reached at the floor (status `floor`), "수렴하지 않으면" = 40 steps without |f| ≤ 0.02 (`no_convergence`); a target already exceeded at zero effect (`above_at_zero`) is unreachable under the literal default (OPEN 2).
11. **Bootstrap and selection (W.9.9 P1-3 / P1-4, H1).** "쌍 단위 파라메트릭 부트스트랩" = J new pilot pairs drawn from the fitted model (pair means ~ N(m0, Σ_pair), per-pair drifts and spillover changes ~ their between-pair normals, J·(F−1) fly effects ~ N(0, Σ_fly) re-estimated, residual blocks resampled by pair), refitted; each draw recalibrated and simulated with **400** experiments per scenario and cluster level (the point estimates use 4000; a draw's Monte-Carlo noise widens the limits, the conservative side). Power limit = 5th percentile over draws of min_k P(PASS); false-pass limit = 95th percentile of max_k (simultaneous over k = 4–8); worst over the cluster grid; a draw whose calibration is unreachable counts as power 0 / false pass 1. Qualification: limits met at F, F+1, F+2, F+3 (F ≥ 29 alone). Cost = `w_records.design_cost` total hours: the 249-pair oracle (V's `jm:L` wall per worker round), the worst naive screen (249 pairs × F jobs of K probes), learning and the BAND upper bound (8 pairs × F × 3 jobs each), C, the plasticity-off control — unit costs from the pilot's jobs (OPEN 1). Ties: larger q, then smaller K, then smaller F. The qualifying designs in cost order are the alternative ranking.
12. **Budget ledger (W.9.6 F, W.9.8 H3, W.9.9 P1-5).** Every block write appends `{stage, wall_s}` to a `ledger` list in `w_learning.json` (resumable pool stages keep their cumulative wall in `results/w/progress/<stage>.json`); elapsed = Σ ledger wall from stage 0. Gate 5a (block `budget`): elapsed + the worst remaining (smoke's unit costs) with C, else without C, else each alternative design without C — the first that fits 24 h is the plan, else `STOP_BUDGET`. Block `estimate` (순서 8) re-costs the plan's design with the real testable count (C dropped first). Inside `naive`, `learn` and `band`, before every worker round, elapsed + this stage's remaining share + the later stages' estimate > 24 h ends the stage with a `STOP_BUDGET` block. The sentence's 〈h〉 is filled "누적 〈e〉 h + 남은 〈r〉 h = 〈t〉 h" (the quantity compared with 24 h).
13. **The BAND re-measure is measured for every gate pair before the seal** (stage `band`): the same flies re-trained with the same training seeds, probes k = K..2K−1 only, its own cache entries (H9's "별도 산출물"). The verdict reads the 2K data only for pairs that are BAND at K, so no statistic is read before the seal and the judge reads once (V's single-read protocol). The retrained weights must equal the main jobs' at every stage (machine check). The budget already carries this cost (P1-4's BAND upper bound).
14. **The naive screen (W.9.5, H3 9, P1-4, P2-10, P3-13).** Oracle-testable pairs in declared order; per pair F jobs of the design's K probes on the judgement probe seeds; pooled |d′| < 0.5 → gate pair. Batches hold at most min(needed, workers // F) pairs, so no pair after the 8th gate pair is ever measured. The naive condition is fixed here; the learning jobs' pre must equal these probes bit for bit (machine).
15. **The main set (W.2).** `w_pairs.w_set`: V's generator state (`v_pairs._Gen`, which reproduces T's set first), V's set regenerated from V's KC values and checked against V's block set, its rows' keys and glomerulus keys added to used; turns 306–1985 with V's skip order; every passing row kept (no cap); declared order = turn, then (b) before (a), then the generator's order; c = position. Block `set` is `INVALID` unless (b) 167 · (a) 82 (W.0's fact — another count is another generator). Later stages regenerate the set and refuse unless every summary field and the key list equal block `set`'s.
16. **Records (W.3 9, W.6).** C (edit "none", block h4's z) on the same gate pairs, flies, seeds and brains with K probes, when the plan keeps C; the plasticity-off control on the first 2 gate pairs × flies 0–1 (R's phases, plasticity off, L_V): every stage's counts = pre and every weight sha = w0. Records only.
17. **Seal and judge (W.3 10–11, W.5, H8).** The seal re-derives every expected unit (naive / learn / band / records) and refuses a raw entry whose stored inputs differ (`INVALID`) or whose file changed / is missing (`NOT_READ`); it pins the decision code (W's hashed files minus the measurement files R / T / U / W), block `oc`, block `budget`, block `gates` and z_V; archive `~/flymon-archive/w/`. Judge = V's once-only protocol (marker, one re-generation after the mark, never after a judge block); `recompute` (analysis defect) and `invalid_run` (measurement defect; H8: never the same gate pairs again) only after judge.
18. **Keys, chain and exit codes.** Every block carries `code_key` (shared), `t_measure_key`, `u_measure_key` (= V's, U's literal), `w_measure_key`, `pipeline_key` and a ledger entry. Every stage after `reuse` re-checks the reuse condition (exit 7). `oracle`, `naive`, `learn`, `band`, `records`, `seal`, `judge` also require every earlier block to carry the current four measurement keys and a clean git record (W.8: "학습 측정 직전과 판정 직전", exit 7). `scripts/run_w.py`: 0 PASS / SEALED / a judgement read / a record stage; 2 a refusal; 3 a STOP; 5 a gate INVALID; 6 seal not SEALED or a smoke with problems; 7 a reuse or key break.
19. **Sentences.** W.7 / W.9.3 / W.9.4 / W.9.7 / W.9.8 verbatim with 〈…〉 filled; `STOP_REUSE` is W's own ("W 재사용 조건(W.3 1)이 깨졌다(…)"; the spec names the label only); `PASS` carries W.9.8's parenthesis and T.9.5's consequence; the defaults of OPEN 3–7 fill the rest.

## OPEN — needs user

These change a STOP sentence's content or a gate's input (the design the OC picks); none changes a gate threshold or the verdict's order. The code below implements the stated default, so implementation can proceed. **The controller asks all seven together before stage 0** (OPEN 1–2 change code that stage 0 commits; asking once is simpler than stopping again later), writing them into `/private/tmp/claude-503/p-latest.md` and stopping; it does not run stage 0 until answered. An answer that changes code is a reviewed fix task before stage 0.

1. **Which costs rank the designs (W.9.9 P1-4 "스모크로 추정한 총 벽시계").** The order puts the OC (4) before the smoke (5), so the smoke's costs do not exist when the design is chosen. Default: the OC ranks with the pilot's measured unit costs (the same job on the same machine: seconds per trial and per presentation, medians over the 384 pilot jobs) and V's `jm:L` oracle wall per worker round; the smoke re-costs the chosen and alternative designs for the budget gate (5a). Alternative: rank only after the smoke (the OC would then commit the qualifying set and the smoke stage would pick by cost).
2. **The false-pass calibration when the zero effect already exceeds 0.5.** With presentation-evoked depression (J.12.8's MBON13(X) 9 → 1 without DAN), the punishment drop gate's true d′ can exceed 0.5 at b = 0, so "max of the punishment gates = 0.5" has no root on b ≥ 0. While planning, a synthetic pilot with a 4-spike MBON13(X) drift (base 40, probe sd 4) already gave this. Default (literal: "수렴하지 않으면 그 설계는 '불가'"): unreachable → every design 불가 → `STOP_OC_UNREACHABLE` (`cal_floor_rule = "unreachable"`). Alternative (`cal_floor_rule = "zero"`): use the zero effect for that gate pair (the weakest association the model allows) and record "zero_floor".
3. **`STOP_OC_UNREACHABLE`'s 〈k별 P(PASS)〉 under H1's designs.** Default: for each (q, K) at F = 32, the simultaneous bootstrap limits "검정력 하한 〈·〉, 거짓 통과 상한 〈·〉" (worst over k and cluster level); when the calibration itself was unreachable, "보정 불가 — min: a 〈status〉, b 〈status〉; max: …". Alternative: the point P(PASS) by k of one named design.
4. **`STOP_FEW_PAIRS`'s wording after W.9.5.** W.2's sentence says "오라클 순진 균형·시험 가능 쌍", but W.9.5 moved the naive screen to the judgement probes. Default: W.2's sentence verbatim. Alternative: "순진 프로브 균형·오라클 시험 가능 쌍".
5. **`UNDECIDED`'s cause.** W.7 quotes F.7 "(BAND 잔존 / 판정 가능 쌍 < 2)" while H7 uses 4. Default: "원인(〈BAND 잔존 | 판정 가능 쌍 < 4〉)을 기록하고 사용자 판단." (unreachable in practice: gate pairs ≥ 4 and BAND resolved before the seal).
6. **`STOP_MACHINE`'s sentence.** W.7 says "F.7 그대로(… 같은 등록으로 재실행)", and H8 says W.5 wins over F.7's rerun. Default: "기계 검사 불일치(〈항목〉) — F.7의 STOP_MACHINE. W.9.8 H8에 따라 같은 관문 쌍으로 다시 돌리지 않는다(INVALID_RUN 여부와 엔진·풀 결함 처리는 사용자 몫)."
7. **`FAIL`'s parenthesis.** W.9.7 appends "(마리별 동시 충족 집계)" and W.9.8 puts the mechanism-control failures in the parenthesis. Default: "F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부(〈쌍: 사유(걸린 관문 마리 수)〉; …; 기계 대조 실패 〈n〉쌍) (마리별 동시 충족 집계)." with W.7's STD consequence as a separate field.

## Review Focus

1. **A naive screen that measures past the 8th gate pair or takes pairs out of order** (P3-13, P3-15). Expect batches of at most the number still needed, unbalanced pairs skipped, turn order with (b) first. Tests: Task 8 (`test_naive_stops_at_eight_gate_pairs_and_unbalanced_pairs_skip`), Task 1 (`test_real_pilot_and_main_set`).
2. **RN sharing R's state, or RN1 ≠ R1 / a pre mismatch passing silently** (G.5, W.9.6 P2-10, W.9.9 P1-2). Expect a machine reason → `STOP_MACHINE`. Tests: Task 5 (`test_rn1_equals_r1_bitwise_and_n_differs`, real), Task 4 (`test_machine_reasons`: an RN that skipped its reward phase), Task 2 (`test_overall_order`: RN1 one bit; `test_mutation_no_rn1_check`), Task 3 (`test_rn1_bit_in_the_simulator_stops_the_machine`), Task 8 (`test_rn_units_run_both_phases_from_pre`).
3. **A NaN or wrong-direction ±∞ fly statistic counting as a pass** (G.7, H5). Expect INVALID → `STOP_MACHINE`, or a failing fly. Tests: Task 2 (`test_fly_class_boundaries_and_infinities`, `test_invalid_data`, `test_mutation_direction_ignored`).
4. **An OC that does not run the verdict code, or a design picked off the envelope** (H4, W.9.9 P1-3). Expect P(PASS) 0 when the verdict always fails, the envelope / solo rule, cost-then-q-then-K. Tests: Task 3 (`test_evaluation_runs_the_verdict_code`, `test_qualify_envelope_and_solo`, `test_select_cost_then_q_then_k`).
5. **A raw entry measured with other inputs, or changed after its block, reaching the judgement** (W.3 10, W.5). Expect the seal `INVALID` / `NOT_READ` and judge refusing; moved pins or decision code refused. Tests: Task 8 (`test_seal_invalid_on_an_entry_measured_with_other_inputs`, `test_seal_not_read_on_a_changed_file`, `test_judge_refuses_moved_pins_and_decision_code`).

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/w_spec.py` (create) | `BRAINS`, `WSpec` (numbers, seed formulas, phases, designs, `z_v`, `smoke_seed_set`), `SPEC`, `smoke` |
| `flymon/brain/w_pairs.py` (create) | `pilot_rows`, `w_set`, `set_summary`, `check_w_set`, `main_rows`, `cluster_labels` |
| `flymon/brain/w_verdict.py` (create) | `dprime`, `dv`, `gate_stats`, `margins`, `fly_class`, `mech_fractions`, `mech_ok`, `pair_gate_code`, `pair_final`, `rn1_mismatch`, `overall_code`, `naive_dprime`, `data_reasons`, `judge_pair`, `judge` |
| `flymon/brain/w_oc.py` (create) | `fit`, `boot`, `summary`, `slot_means`, `to_stages`, `simulate`, `true_dprimes`, `calibrate`, `evaluate`, `qualify`, `select`, `run`, `records`, `synthetic_pilot`, `simple_normal`, `synthetic_validation` |
| `flymon/brain/w_records.py` (create) | `counts`, `pair_data`, `machine_reasons`, `extend`, `p_repro_diffs`, `naive_repro_diffs`, `reward_check`, `pilot_record`, `unit_costs`, `job_s`, `design_cost`, `elapsed_h` |
| `flymon/brain/w_rules.py` (create) | outcome constants, `SENTENCES`, `CONSEQUENCE`, `sentence`, `reuse`, `path`, `pilot`, `oc`, `oc_unreachable_text`, `budget`, `gate_pairs`, `verdict_sentence` |
| `flymon/brain/w_measure.py` (create) | `W_MEASURE_FILES`, `KIND`, `w_measure_key`, `w_learn_job`, `WMeasurer` |
| `flymon/brain/w_store.py` (create) | `ALLOWED_DIR`, `SUMMARY`, `guard`, `write_bytes`, `write_json`, `read_summary`, `write_summary_block` (+ ledger), `WCache`, `VReadCache`; re-exports `load_manifest`, `archive_copy` |
| `flymon/brain/w_runner.py` (create in Task 6, extended in Task 8) | `ORDER`, `GATES`, file lists, markers, `pipeline_key`, `decision_key`, `decision_pins`, `git_facts`, `build_ctx`, `Runner` |
| `scripts/run_w.py` (create, Task 7) | the CLI, `make_measurers`, `exit_code`, `__main__` guard |
| `tests/brain/test_w_spec.py`, `test_w_pairs.py`, `test_w_verdict.py`, `test_w_oc.py`, `test_w_records.py`, `test_w_rules.py`, `test_w_measure.py`, `test_w_store.py`, `test_w_runner.py`, `test_w_cli.py`, `test_w_runner_main.py`, `test_w_judge.py`, `w_world.py` (create) | tests and the runner world (`r_fixtures.py`, `s_fixtures.py` are reused as they are) |
| `tests/brain/test_p_spec.py` (modify: one `MODULES` entry) | Reading 1 |

---

### Task 1: `w_spec` and `w_pairs` — W's numbers, the seed blocks, the pilot pairs and the main set

**Files:**
- Create: `flymon/brain/w_spec.py`, `flymon/brain/w_pairs.py`
- Modify: `tests/brain/test_p_spec.py` (one `MODULES` entry)
- Test: `tests/brain/test_w_spec.py`, `tests/brain/test_w_pairs.py`

**Interfaces:**
- Consumes: `v_pairs` (`_Gen`, `_kc_out`, `sides`, `REASONS`, `v_set`, `check_v_set`), `v_spec.SPEC`, `e_pairs`, `q_pairs.codebook`, `t_pairs` (`_gkey`, `cluster_of`, `clusters`), `r_pairs` (`okey`, `row_key`), `tests/brain/test_p_spec` (`MODULES`, `_collect`).
- Produces: `BRAINS = ("R", "N", "RN")`; `WSpec` with the fields in the code and `probe_seeds(c, f, k_n, k0=0) -> list`, `pilot_probe_seeds(j, f, k_n, k0=0)`, `smoke_probe_seeds(f, k_n, k0=0)`, `train_base(c, phase) -> int`, `pilot_train_base(j, phase)`, `smoke_train_base(phase)`, `oracle_seeds() -> {act, select, report}`, `smoke_seed_set() -> frozenset`, `phases(brain, base0, base1) -> [[dan | None, trials, base], …]`, `designs() -> [(q, K, F)]`, `z_v() -> {"A": (m, sd), "P": …}`; `SPEC`; `smoke(spec) -> WSpec`. `w_pairs.pilot_rows(even_rows, v_even_pairs, spec) -> list`; `w_set(pops, enc, params, kc, v_block, spec) -> dict` (`rows`, `keys` + `SUMMARY_KEYS`); `set_summary(js) -> dict`; `check_w_set(js, block) -> list[str]`; `main_rows(pops, rc, enc, params, kc, v_block, block, spec) -> list` (rows with `odor_x`, `odor_y`, `c`); `cluster_labels(rows) -> dict`.

- [ ] **Step 1: Add W to the spec-module list (Reading 1)**

In `tests/brain/test_p_spec.py`, extend `MODULES`'s last line:

```python
           "flymon/brain/v_spec.py": "flymon.brain.v_spec", "flymon/brain/w_spec.py": "flymon.brain.w_spec"}
```

- [ ] **Step 2: Write the failing tests**

```python
"""Spec W.1-W.3 / W.9: every W number in w_spec; W's blocks (main 26_000_000 / 28_000_000 by candidate, pilot
40_000_000 / 41_000_000, smoke 42_1xx_xxx, OC 42_000_000, oracle 24_700_xxx) as formulas, inside their layout and
colliding with no declared seed of any other spec module (nor with their P-style training seeds); W declares none of
V's seeds (WSpec is not a VSpec); the phases of R / N / RN; the design grid; the shared, T, U and W measurement keys."""
import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

from flymon.brain import d6a
from flymon.brain.w_spec import BRAINS, SPEC, WSpec, smoke
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
BASE, STRIDE = 1_000_000, 1000


def _declared_without_w() -> set:
    out, seen = set(d6a.SEEDS), set()
    mods = {k: v for k, v in MODULES.items() if k != "flymon/brain/w_spec.py"}
    mods["flymon/brain/p_spec.py"] = "flymon.brain.p_spec"
    for mod in mods.values():
        m = importlib.import_module(mod)
        _collect(m.SPEC, out, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), out, seen)
    return out


def w_seeds(n_cand=SPEC.n_b_expected + SPEC.n_a_expected, n_pilot=16, k_max=32) -> tuple:
    probes, trains = set(), set()
    for c in range(n_cand):
        for f in range(SPEC.f_max):
            probes |= set(SPEC.probe_seeds(c, f, k_max))
            trains |= {SPEC.train_base(c, p) + f * SPEC.fly_train_stride + t for p in (0, 1)
                       for t in range(SPEC.trials)}
    for j in range(n_pilot):
        for f in range(SPEC.pilot_flies):
            probes |= set(SPEC.pilot_probe_seeds(j, f, SPEC.pilot_probes))
            trains |= {SPEC.pilot_train_base(j, p) + f * SPEC.fly_train_stride + t for p in (0, 1)
                       for t in range(SPEC.trials)}
    probes |= set(SPEC.smoke_seed_set()) | {s for v in SPEC.oracle_seeds().values() for s in v}
    trains |= {SPEC.smoke_train_base(p) + f * SPEC.fly_train_stride + t for p in (0, 1) for f in range(SPEC.f_max)
               for t in range(SPEC.trials)}
    return probes, trains


def test_w_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/w_spec.py") == "flymon.brain.w_spec"


def test_seed_formulas():
    assert SPEC.probe_seeds(5, 3, 8) == [26_000_000 + 5 * 4_000 + 3 * 100 + k for k in range(8)]
    assert SPEC.probe_seeds(5, 3, 16, k0=8) == [26_000_000 + 20_000 + 300 + k for k in range(8, 16)]
    assert SPEC.train_base(5, 0) == 28_000_000 + 5 * 40_000 and SPEC.train_base(5, 1) == 28_000_000 + 200_020
    assert SPEC.pilot_probe_seeds(2, 1, 8) == [40_000_000 + 8_000 + 100 + k for k in range(8)]
    assert SPEC.pilot_train_base(2, 1) == 41_000_000 + 80_000 + 20
    assert SPEC.oracle_seeds() == dict(act=list(range(24_700_000, 24_700_008)),
                                       select=list(range(24_700_100, 24_700_108)),
                                       report=list(range(24_700_200, 24_700_208)))
    sm = smoke(SPEC).oracle_seeds()
    assert all(42_100_000 <= s < 42_200_000 for v in sm.values() for s in v)
    assert SPEC.oc_seed == 42_000_000
    for bad in (lambda: SPEC.probe_seeds(300, 0, 8), lambda: SPEC.probe_seeds(0, 32, 8),
                lambda: SPEC.probe_seeds(0, 0, 101), lambda: SPEC.pilot_probe_seeds(25, 0, 8)):
        with pytest.raises(ValueError):
            bad()


def test_layout_has_no_internal_overlap():
    probes, trains = w_seeds()
    assert not probes & trains
    main = {s for s in probes | trains if s < 42_000_000}
    smoke_ = (probes | trains) - main
    assert min(probes) >= 24_700_000 and max(main) < 42_000_000
    assert all(42_100_000 <= s < 42_200_000 for s in smoke_)
    assert all(not 27_200_000 <= s < 28_000_000 for s in main)


def test_global_seed_collision():
    declared = _declared_without_w()
    for s in (500, 24_002_000, 24_600_000, 25_400_000, 23_000_000, 22_000_000):
        assert s in declared, s
    probes, trains = w_seeds()
    mine = probes | trains
    assert not mine & declared
    assert not {s for s in mine if (s - BASE) // STRIDE in declared}       # no declared s trains P-style on W's


def test_wspec_carries_no_v_seed():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {26_000_000, 28_000_000, 40_000_000, 41_000_000, 42_100_000, 42_110_000, 42_150_000, 42_000_000} \
        | set(range(24_700_000, 24_700_008)) | set(range(24_700_100, 24_700_108)) | set(range(24_700_200, 24_700_208))
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(WSpec))


def test_phases_and_designs():
    assert BRAINS == ("R", "N", "RN")
    assert SPEC.phases("R", 10, 30) == [["PAM08", 20, 10], ["PPL105", 20, 30]]
    assert SPEC.phases("N", 10, 30) == [[None, 20, 10], [None, 20, 30]]
    assert SPEC.phases("RN", 10, 30) == [["PAM08", 20, 10], [None, 20, 30]]
    d = SPEC.designs()
    assert len(d) == 3 * 2 * 25 and d[0] == (0.5, 8, 8) and d[-1] == (0.75, 16, 32)


def test_numbers():
    assert (SPEC.bar, SPEC.band_width, SPEC.naive_max, SPEC.mech_min, SPEC.round_digits) == (1.0, 0.2, 0.5, 0.75, 9)
    assert (SPEC.d_power, SPEC.p_power, SPEC.d_false, SPEC.p_false) == (1.5, 0.80, 0.5, 0.05)
    assert (SPEC.oc_reps, SPEC.cal_reps, SPEC.cal_tol, SPEC.cal_iter, SPEC.boot_draws) == (4000, 2000, 0.02, 40, 200)
    assert SPEC.cluster_grid == (0.0, 0.5, 1.0) and SPEC.boot_level == 0.95 and SPEC.boot_reps == 400
    assert (SPEC.k_cap, SPEC.k_min, SPEC.envelope, SPEC.envelope_solo_from) == (8, 4, 3, 29)
    assert (SPEC.trials, SPEC.pulse_ms, SPEC.train_settle_ms, SPEC.gap_ms, SPEC.strength) == (20, 400.0, 800.0,
                                                                                              200.0, 1.0)
    assert (SPEC.first_turn, SPEC.last_turn, SPEC.n_b_expected, SPEC.n_a_expected) == (306, 1985, 167, 82)
    assert SPEC.budget_h == 24.0 and SPEC.min_gate_pairs == 4 and SPEC.cal_floor_rule == "unreachable"
    assert SPEC.z_v() == {"A": (16.917, 12.484), "P": (80.167, 29.775)}
    assert dict(SPEC.v_commits) == {"z": "928eaad", "kc_input": "7dc199d", "set": "cf0b3b2", "judge": "a279a56"}
    assert (SPEC.summary, SPEC.cache_dir, SPEC.archive_root) == (
        "results/summary/w_learning.json", "results/w/cache", "~/flymon-archive/w")


def test_smoke_changes_scale_only():
    sm = smoke(SPEC)
    for f in dataclasses.fields(WSpec):
        if f.name not in ("smoke", "workers"):
            assert getattr(sm, f.name) == getattr(SPEC, f.name), f.name
    assert sm.smoke and sm.workers == 4


def test_no_w_file_holds_another_tracks_block_as_a_literal():
    files = sorted((ROOT / "flymon/brain").glob("w_*.py")) + [ROOT / "scripts/run_w.py"]
    for p in files:
        ints = {n.value for n in ast.walk(ast.parse(p.read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is int}
        bad = {i for i in ints if 24_000_000 <= i < 24_700_000 or 24_710_000 <= i < 26_000_000
               or i in (20261004, 20261005) or 800_000 <= i < 800_200}
        assert not bad, (p.name, bad)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_shared_t_and_u_keys_are_the_reused_ones():
    from flymon.brain.h3_store import code_key
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import u_measure_key
    assert code_key(str(NPZ), files=R_MEASURE_FILES)["key"] == SPEC.r_shared_key
    assert t_measure_key(str(NPZ))["key"] == SPEC.t_measure_key_t
    assert u_measure_key(str(NPZ))["key"] == SPEC.u_measure_key_u
```

```python
"""W's pairs (W.2, W.3 3, W.9.6 P2-11 / P3-15): the pilot = V-even L_V-testable (b) pairs + a|4 Rock Slide|Strength in
the even rows' order; the main set from V's generator with V's set as used, turns 306-1985, no axis cap, V's skips,
(b) 167 · (a) 82 (W.0's fact), declared order turn then (b) first, candidate numbers c; block-set checking."""
import json
from pathlib import Path

import pytest

from flymon.brain import w_pairs
from flymon.brain.w_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


def test_pilot_rows_order_and_refusals():
    even = [dict(axis=a, turn=t, x=x, y="y") for a, t, x in (("b", 0, "s"), ("a", 4, "Rock Slide"), ("b", 2, "q"),
                                                              ("b", 4, "r"))]
    for r in even:
        if r["x"] == "Rock Slide":
            r["y"] = "Strength"
    v_even = [dict(key="b|0|s|y", axis="b", testable=True), dict(key="b|2|q|y", axis="b", testable=False),
              dict(key="b|4|r|y", axis="b", testable=True), dict(key="a|4|Rock Slide|Strength", axis="a",
                                                                testable=True)]
    got = w_pairs.pilot_rows(even, v_even, SPEC)
    assert [(r["axis"], r["turn"]) for r in got] == [("b", 0), ("a", 4), ("b", 4)]
    with pytest.raises(ValueError):
        w_pairs.pilot_rows(even[:1], v_even, SPEC)


def test_check_w_set():
    js = dict(n_b=1, n_a=1, n=2, first_turn=306, last_turn=307, skipped={}, clusters_b=[], clusters_a=[],
              digest_keys="k", digest_e0_b="b", digest_e0_a="a", n_odours=2, all_off_pool=True, keys=["x", "y"])
    blk = dict(w_pairs.set_summary(js), keys=["x", "y"])
    assert w_pairs.check_w_set(js, blk) == []
    assert w_pairs.check_w_set(dict(js, keys=["y", "x"]), blk) == ["W set keys differ from block set"]
    assert w_pairs.check_w_set(dict(js, n_b=2), blk)[0].startswith("W set n_b")


@pytest.fixture(scope="module")
def real():
    if not NPZ.exists() or not (ROOT / SPEC.v_summary).exists():
        pytest.skip("no connectome or V summary")
    from flymon.agent.config import load_c3_config
    from flymon.brain import r_pairs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.v_runner import kc_values
    from flymon.brain.v_spec import SPEC as V
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    enc = json.loads((ROOT / V.encoder_summary).read_text())
    v = json.loads((ROOT / SPEC.v_summary).read_text())
    params = load_c3_config(V.m0d_summary).params
    rc = {str(t): len(x) for t, x in pops.receptor_types.items()}
    return dict(pops=pops, enc=enc, v=v, params=params, rc=rc, kc=kc_values(v),
                even=r_pairs.even_rows(pops, rc, enc, V))


def test_real_pilot_and_main_set(real):
    pil = w_pairs.pilot_rows(real["even"], real["v"]["even"]["pairs"], SPEC)
    assert len(pil) == 16 and sum(r["axis"] == "b" for r in pil) == 15
    js = w_pairs.w_set(real["pops"], real["enc"], real["params"], real["kc"], real["v"]["set"]["set"], SPEC)
    assert (js["n_b"], js["n_a"], js["n"]) == (167, 82, 249) and js["all_off_pool"]
    assert js["first_turn"] == 306 and js["last_turn"] <= 1985
    t = [(r["turn"], r["axis"] != "b") for r in js["rows"]]
    assert t == sorted(t) and min(r["turn"] for r in js["rows"]) >= 306
    blk = dict(w_pairs.set_summary(js), keys=js["keys"])
    rows = w_pairs.main_rows(real["pops"], real["rc"], real["enc"], real["params"], real["kc"],
                             real["v"]["set"]["set"], blk, SPEC)
    assert [r["c"] for r in rows] == list(range(249)) and "odor_x" in rows[0]
    with pytest.raises(ValueError):
        w_pairs.main_rows(real["pops"], real["rc"], real["enc"], real["params"], real["kc"],
                          real["v"]["set"]["set"], dict(blk, digest_keys="x"), SPEC)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_spec.py tests/brain/test_w_pairs.py -q`
Expected: FAIL (ImportError: `flymon.brain.w_spec`).

- [ ] **Step 4: Write `w_spec`**

```python
"""Every number of spec appendix W as amended by W.9 (W.9.9 > W.9.8 > W.9.1-W.9.7 > W.0-W.8): the F v4 learning test
(spec 5's M2 learning unit, F.2's sequential R / N brains plus G.5's state-matched RN) on V's combined lever L_V,
read with V's z_V, judged by per-fly joint satisfaction with a design (q, K, F; k cap 8) chosen by W's own operating
characteristic (w_oc) from a POOL even pilot.
- WSpec is a plain frozen dataclass, not a VSpec: every field whose name holds "seed" carries a W block only, so the
  repository's seed collectors (tests/brain/test_p_spec.py MODULES) see W's blocks and nothing of V's. V's numbers
  (the lever string, the oracle's windows, alphas, seeds and z rule, the readout, the gate-② P spec) are read from
  v_spec.SPEC where they are used (w_runner, w_pairs), never restated here.
- Seeds (W.9.6 P2-11, W.9.8 H9): candidate c of the main set (declared order), fly f, probe k, trial t —
  probe 26_000_000 + c·4_000 + f·100 + k, training 28_000_000 + c·40_000 + f·1_000 + t (t < 40: reward / first
  phase t < 20, punishment / second phase 20 ≤ t < 40); pilot pair j — 40_000_000 + j·4_000 + f·100 + k and
  41_000_000 + j·40_000 + f·1_000 + t; smoke inside 42_100_000-42_199_999; w_oc's generator root 42_000_000; the
  oracle screen 24_700_000+i / 24_700_100+i / 24_700_200+i (i < 8).
- The protocol table (W.9.6 P2-12, "W.1 부록 표"): C3's Params unchanged (block h4: kc_kc_scale 0, recovery 0, learn
  rate 3e-4 …, read from the C3 config at run time), o_jobs.train_x's settle 800 ms and gap 200 ms, F.2's pulse 400 ms
  × 20 per phase, PAM08 (reward) / PPL105 (punishment), the probe windows of V's oracle (settle 800 · read 600 ·
  window 200 at the E-grid strength 1.0 — the same probe path as V's oracle, W.3 2 (ii)); training at the same
  strength 1.0.
- Paths: results/w/ (git-ignored), results/summary/w_learning.json (tracked), ~/flymon-archive/w."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

BRAINS = ("R", "N", "RN")                          # F.2 R and N, G.5's RN (W.9.6 P2-10: RN runs from pre)


@dataclass(frozen=True)
class WSpec:
    # ---- engine and readout (W.1) ------------------------------------------------------------------------------------
    z_v_declared: tuple = (("A", (16.917, 12.484)), ("P", (80.167, 29.775)))   # W.1, V block z 928eaad (3 decimals)
    z_v_digits: int = 3
    # ---- the protocol table (F.2, W.9.6 P2-12) -----------------------------------------------------------------------
    trials: int = 20                               # per phase
    pulse_ms: float = 400.0
    train_settle_ms: float = 800.0                 # o_jobs.train_x's settle (h4 teach_window)
    gap_ms: float = 200.0                          # o_jobs.train_x's gap (h4 teach_gap_ms)
    strength: float = 1.0                          # E-grid k2-norm s 1.0: probes and training
    reward_dan: str = "PAM08"
    punish_dan: str = "PPL105"
    # ---- the verdict (F.5, W.9.1, W.9.2, W.9.8 H2 / H5 / H7) --------------------------------------------------------
    bar: float = 1.0
    band_width: float = 0.2
    naive_max: float = 0.5
    mech_min: float = 0.75
    round_digits: int = 9
    min_gate_pairs: int = 4
    # ---- the designs (W.9.8 H1, W.9.9 P2-7) --------------------------------------------------------------------------
    q_grid: tuple = (0.5, 0.625, 0.75)
    k_grid: tuple = (8, 16)                        # probes K; BAND re-measure 2K
    f_min: int = 8
    f_max: int = 32
    k_min: int = 4                                 # gate pairs: the OC's k runs k_min .. k_cap
    k_cap: int = 8
    envelope: int = 3                              # F, F+1, F+2, F+3
    envelope_solo_from: int = 29                   # F 29-32 alone (W.9.9 P1-3)
    # ---- G.6's targets and w_oc (W.9.3, W.9.8 H4, W.9.9) -------------------------------------------------------------
    d_power: float = 1.5
    p_power: float = 0.80
    d_false: float = 0.5
    p_false: float = 0.05
    oc_reps: int = 4000
    oc_seed: int = 42_000_000
    cal_reps: int = 2000
    cal_tol: float = 0.02
    cal_iter: int = 40
    boot_draws: int = 200
    boot_reps: int = 400                           # experiments per bootstrap draw (Reading 11)
    boot_level: float = 0.95
    cluster_grid: tuple = (0.0, 0.5, 1.0)
    record_dprimes: tuple = (0.5, 1.0, 1.5, 2.0)
    oc_chunk: int = 50
    cal_floor_rule: str = "unreachable"            # OPEN 2: "unreachable" (literal) or "zero"
    # ---- the pilot (W.3 3, W.9.4, W.9.8 H6) --------------------------------------------------------------------------
    pilot_flies: int = 8
    pilot_probes: int = 8
    pilot_extra: tuple = ("a|4|Rock Slide|Strength",)    # + every L_V-testable (b) even pair (V block even)
    no_effect_d: float = 0.5
    no_effect_share: float = 0.5
    floor_share_max: float = 0.5
    naive_floor_spikes: int = 2                    # W.9.2's pilot pre-check record (< 2 spikes)
    # ---- the main set (W.2) ------------------------------------------------------------------------------------------
    first_turn: int = 306
    last_turn: int = 1985
    n_b_expected: int = 167                        # W.0 fact check
    n_a_expected: int = 82
    n_cand_max: int = 300
    # ---- seeds (W.9.6 P2-11; Reading 6) ------------------------------------------------------------------------------
    probe_seed0: int = 26_000_000
    train_seed0: int = 28_000_000
    pilot_probe_seed0: int = 40_000_000
    pilot_train_seed0: int = 41_000_000
    smoke_probe_seed0: int = 42_100_000
    smoke_train_seed0: int = 42_110_000
    smoke_oracle_seed0: int = 42_150_000
    cand_probe_stride: int = 4_000
    cand_train_stride: int = 40_000
    fly_probe_stride: int = 100
    fly_train_stride: int = 1_000
    phase_trial_offset: int = 20                   # the second phase's t starts at 20
    n_pilot_max: int = 25
    oracle_act_seeds: tuple = tuple(range(24_700_000, 24_700_008))
    oracle_select_seeds: tuple = tuple(range(24_700_100, 24_700_108))
    oracle_report_seeds: tuple = tuple(range(24_700_200, 24_700_208))
    # ---- the path gate (W.3 2, W.9.9 P2-9) ---------------------------------------------------------------------------
    repro_p_count: int = 2                         # V gate ②'s first 2 seeds per direction
    repro_p_arm: str = "punish"
    repro_naive_rows: int = 3                      # V's judgement rows 0-2, both conditions
    # ---- the plasticity-off control and C (W.3 9) --------------------------------------------------------------------
    noplast_pairs: int = 2
    noplast_flies: int = 2
    # ---- the budget (W.9.6 F, W.9.9 P1-4 / P1-5) ---------------------------------------------------------------------
    budget_h: float = 24.0
    # ---- V's reuse (W.3 1) -------------------------------------------------------------------------------------------
    r_shared_key: str = "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    t_measure_key_t: str = "7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374"
    u_measure_key_u: str = "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    v_commits: tuple = (("z", "928eaad"), ("kc_input", "7dc199d"), ("set", "cf0b3b2"), ("judge", "a279a56"))
    v_band: str = "SELECTED"
    v_summary: str = "results/summary/v_lever.json"
    # ---- smoke and the pool ------------------------------------------------------------------------------------------
    smoke: bool = False
    smoke_flies: int = 1
    workers: int = 16
    pool_timeout_s: float = 3600.0
    # ---- paths -------------------------------------------------------------------------------------------------------
    summary: str = "results/summary/w_learning.json"
    raw_dir: str = "results/w"
    cache_dir: str = "results/w/cache"
    smoke_cache_dir: str = "results/w/smoke/cache"
    progress_dir: str = "results/w/progress"
    oc_detail: str = "results/w/oc.json"
    path_detail: str = "results/w/path.json"
    archive_root: str = "~/flymon-archive/w"

    # ---- seeds -------------------------------------------------------------------------------------------------------
    def _probe(self, root: int, stride: int, n: int, f: int, k_n: int) -> list:
        if not 0 <= f < self.f_max or not 0 < k_n <= self.fly_probe_stride:
            raise ValueError(f"fly {f} / {k_n} probes outside the block layout")
        return [root + n * stride + f * self.fly_probe_stride + k for k in range(k_n)]

    def probe_seeds(self, c: int, f: int, k_n: int, k0: int = 0) -> list:
        """Main-set candidate c, fly f: probes k0 .. k_n - 1 (W.9.6 P2-11)."""
        if not 0 <= c < self.n_cand_max:
            raise ValueError(f"candidate {c} outside 0..{self.n_cand_max - 1}")
        return self._probe(self.probe_seed0, self.cand_probe_stride, c, f, k_n)[k0:]

    def pilot_probe_seeds(self, j: int, f: int, k_n: int, k0: int = 0) -> list:
        if not 0 <= j < self.n_pilot_max:
            raise ValueError(f"pilot pair {j} outside 0..{self.n_pilot_max - 1}")
        return self._probe(self.pilot_probe_seed0, self.cand_probe_stride, j, f, k_n)[k0:]

    def smoke_probe_seeds(self, f: int, k_n: int, k0: int = 0) -> list:
        return self._probe(self.smoke_probe_seed0, 0, 0, f, k_n)[k0:]

    def train_base(self, c: int, phase: int) -> int:
        """train_x's seed_base of phase 0 / 1 for candidate c (train seed = base + f·1_000 + t′)."""
        if not 0 <= c < self.n_cand_max:
            raise ValueError(f"candidate {c} outside 0..{self.n_cand_max - 1}")
        return self.train_seed0 + c * self.cand_train_stride + phase * self.phase_trial_offset

    def pilot_train_base(self, j: int, phase: int) -> int:
        if not 0 <= j < self.n_pilot_max:
            raise ValueError(f"pilot pair {j} outside 0..{self.n_pilot_max - 1}")
        return self.pilot_train_seed0 + j * self.cand_train_stride + phase * self.phase_trial_offset

    def smoke_train_base(self, phase: int) -> int:
        return self.smoke_train_seed0 + phase * self.phase_trial_offset

    def oracle_seeds(self) -> dict:
        if self.smoke:
            s = self.smoke_oracle_seed0
            return dict(act=list(range(s, s + 8)), select=list(range(s + 100, s + 108)),
                        report=list(range(s + 200, s + 208)))
        return dict(act=list(self.oracle_act_seeds), select=list(self.oracle_select_seeds),
                    report=list(self.oracle_report_seeds))

    def smoke_seed_set(self) -> frozenset:
        """Every seed a smoke entry may carry (WCache's smoke scope)."""
        s = set()
        for f in range(self.f_max):
            s |= set(self._probe(self.smoke_probe_seed0, 0, 0, f, self.fly_probe_stride))
        o = self.smoke_oracle_seed0
        return frozenset(s | set(range(o, o + 8)) | set(range(o + 100, o + 108)) | set(range(o + 200, o + 208)))

    # ---- phases (F.2, G.5) -------------------------------------------------------------------------------------------
    def phases(self, brain: str, base0: int, base1: int) -> list:
        """[[dan or None, trials, seed_base]] of R (PAM08 then PPL105), N (none, none) and RN (PAM08, then none)."""
        dans = {"R": (self.reward_dan, self.punish_dan), "N": (None, None), "RN": (self.reward_dan, None)}[brain]
        return [[dans[0], int(self.trials), int(base0)], [dans[1], int(self.trials), int(base1)]]

    def designs(self) -> list:
        return [(q, k, f) for q in self.q_grid for k in self.k_grid for f in range(self.f_min, self.f_max + 1)]

    def z_v(self) -> dict:
        return {k: tuple(v) for k, v in self.z_v_declared}


SPEC = WSpec()


def smoke(spec: WSpec = SPEC) -> WSpec:
    """Scale only: the smoke flag (oracle seeds 42_150_xxx, inside the smoke block) and 4 workers. Every threshold
    stays."""
    return dataclasses.replace(spec, smoke=True, workers=4)
```

- [ ] **Step 5: Write `w_pairs`**

```python
"""W's pairs, list only — nothing here runs the engine.
- Pilot (W.3 3, W.9.4): the POOL even pairs (H.4's even declaration, r_pairs.even_rows) that V's block even found
  L_V-testable on axis (b), plus a|4 Rock Slide|Strength, in the even rows' declared order; pilot number j = the
  position in that list. The pilot is records-only and never gates (G.2, G.5).
- Main set (W.2): V's generator (v_pairs._Gen: T's permutation, T's used rows and T's set as used) with V's set rows
  (regenerated from block kc_input's values of V's summary and checked against V's block set) added to the used keys
  and glomerulus keys; turns 306-1985, every (b) and (a) row that passes V's skips in V's order (cap → collision → E1 →
  used → glomerulus duplicate → in-set → POOL-only → kc_input on either engine), no axis cap. Declared order (W.9.6
  P3-15): generator turn, then (b) before (a) within a turn, then the generator's own order; candidate number c = the
  position (W.9.6 P2-11: c < 300). The block `set` fixes its digests, counts, last turn, skip counts and clusters
  before any oracle on it; later stages regenerate it and refuse unless it equals the block."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from . import q_pairs, t_pairs, v_pairs
from .r_pairs import okey, row_key
from .v_pairs import REASONS, _Gen, _kc_out, sides
from .v_spec import SPEC as V_SPEC

OK = "OK"
SUMMARY_KEYS = ("n_b", "n_a", "n", "first_turn", "last_turn", "skipped", "clusters_b", "clusters_a", "digest_keys",
                "digest_e0_b", "digest_e0_a", "n_odours", "all_off_pool")


def pilot_rows(even_rows: list, v_even_pairs: list, spec) -> list:
    """The pilot pairs in the even rows' declared order: V-even (b) testable on L_V, plus spec.pilot_extra."""
    testable_b = {p["key"] for p in v_even_pairs if p["axis"] == "b" and p["testable"]}
    keep = testable_b | set(spec.pilot_extra)
    rows = [r for r in even_rows if row_key(r) in keep]
    missing = keep - {row_key(r) for r in rows}
    if missing:
        raise ValueError(f"pilot pairs not among the even rows: {sorted(missing)}")
    if len(rows) > spec.n_pilot_max:
        raise ValueError(f"{len(rows)} pilot pairs exceed the seed layout's {spec.n_pilot_max}")
    return rows


def _order(rows: list) -> list:
    return sorted(rows, key=lambda r: (int(r["turn"]), 0 if r["axis"] == "b" else 1))


def w_set(pops, enc: dict, params, kc: dict, v_block: dict, spec) -> dict:
    """The main set and its record (module docstring). kc = V's block kc_input values {"none", "lever"}; v_block = V's
    block set's "set" record. ValueError when V's set does not reproduce or an odour has no KC value."""
    js_v = v_pairs.v_set(pops, enc, params, kc, V_SPEC)
    bad = v_pairs.check_v_set(js_v, v_block)
    if bad:
        raise ValueError("V's set does not reproduce V's block set: " + "; ".join(bad[:3]))
    g = _Gen(pops, enc, params)
    v_rows = js_v["b"] + js_v["a"]
    g.used |= {e_pairs.egrid_key(r) for r in v_rows}
    g.used_g |= {t_pairs._gkey(g.cb, r) for r in v_rows}
    taken, rows, skipped = set(), [], Counter({k: 0 for k in REASONS})
    for j in range(spec.first_turn, spec.last_turn + 1):
        for r in g.rows(j):
            k = e_pairs.egrid_key(r)
            why = g.fixed_reason(r)
            if why is None and k in taken:
                why = "in_set"
            if why is None and g.pool_only(r):
                why = "pool_only"
            if why is None and any(_kc_out(kc, okey(m, o), V_SPEC.valid_band) for m, o in sides(r)):
                why = "kc_input"
            if why is not None:
                skipped[why] += 1
            else:
                taken.add(k)
                rows.append(r)
    rows = _order(rows)
    b, a = [r for r in rows if r["axis"] == "b"], [r for r in rows if r["axis"] == "a"]
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in rows]
    clusters = Counter(t_pairs.cluster_of(r, g.pool_sets) for r in rows)
    odours = sorted({s for r in rows for s in sides(r)})
    return dict(rows=rows, n_b=len(b), n_a=len(a), n=len(rows), first_turn=spec.first_turn,
                last_turn=max((r["turn"] for r in rows), default=None), skipped=dict(skipped),
                clusters_b=sorted([*c, n] for c, n in clusters.items() if len(c) == 2),
                clusters_a=sorted([*c, n] for c, n in clusters.items() if len(c) == 1),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a), n_odours=len(odours),
                all_off_pool=all(not g.pool_only(r) for r in rows), keys=[row_key(r) for r in rows])


def set_summary(js: dict) -> dict:
    return {k: js[k] for k in SUMMARY_KEYS}


def check_w_set(js: dict, block: dict) -> list:
    """[] when the regenerated set equals block set's record (every SUMMARY_KEYS value and the key list)."""
    bad = [f"W set {k}: regenerated {js[k]!r}, block set {block.get(k)!r}" for k in SUMMARY_KEYS
           if json.loads(json.dumps(js[k])) != block.get(k)]
    if js["keys"] != block.get("keys"):
        bad.append("W set keys differ from block set")
    return bad


def main_rows(pops, rc: dict, enc: dict, params, kc: dict, v_block: dict, block: dict, spec) -> list:
    """The checked main set's rows with E-grid odours and candidate numbers c (declared order)."""
    if spec.smoke:
        raise ValueError("smoke never uses the main set (W.9.5)")
    js = w_set(pops, enc, params, kc, v_block, spec)
    bad = check_w_set(js, block)
    if bad:
        raise ValueError("; ".join(bad[:3]))
    rows = e_pairs.attach_odours(js["rows"], rc, q_pairs.codebook(enc, V_SPEC), E.dual_rule(V_SPEC.config))
    return [dict(r, c=i) for i, r in enumerate(rows)]


def cluster_labels(rows: list) -> dict:
    return t_pairs.clusters(rows)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/brain/test_w_spec.py tests/brain/test_w_pairs.py tests/brain/test_p_spec.py tests/brain/test_v_spec.py tests/brain/test_u_spec.py tests/brain/test_t_spec.py tests/agent/test_e_spec_store.py -q`
Expected: all pass (the real-data tests need `data/malecns.npz` and V's summary; ~1 min). The real main set must be (b) 167 · (a) 82.

- [ ] **Step 7: Commit**

```bash
git add flymon/brain/w_spec.py flymon/brain/w_pairs.py tests/brain/test_w_spec.py tests/brain/test_w_pairs.py tests/brain/test_p_spec.py
git commit -m "feat(w): w_spec (seed blocks 26/28/40/41/42 by candidate, oracle 24_700_xxx, designs, protocol table) and w_pairs (pilot = V-even testable (b) + a|4, main set turns 306-1985 with V's set as used, 167/82)"
```

---

### Task 2: `w_verdict` — the W judgement code (per-fly joint satisfaction, BAND → 2K, mechanism control, H7)

**Files:**
- Create: `flymon/brain/w_verdict.py`
- Test: `tests/brain/test_w_verdict.py`

**Interfaces:**
- Consumes: `w_spec.SPEC` (bar, band width, digits, mechanism minimum, minimum gate pairs).
- Produces (arrays `[..., F, K, 2 cells (A, P), 2 odours (X, Y)]` per stage in `STAGES = ("pre", "R1", "R2", "N1", "N2", "RN1", "RN2")`): `dprime(x, axis=-1)`, `dv(arr, z)`, `gate_stats(d, z) -> [..., F, 4]`, `margins`, `fly_class(stats, bar, band, digits) -> [..., F]` (`FLY_INVALID / FAIL / BAND / SAT`), `mech_fractions(d) -> [..., F, 2]`, `mech_ok(fr, mech_min, digits) -> [...]`, `pair_gate_code(cls, q, f_sched, digits)`, `pair_final(code_k, mech_k, code_2k, mech_2k)` (`P_INVALID / FAIL / BAND / PASS`), `rn1_mismatch(d) -> [...]`, `overall_code(final, machine, min_pairs)` (`V_STOP_MACHINE / FAIL / UNDECIDED / PASS`), `naive_dprime(pre, z) -> float`, `data_reasons(d, f, k) -> list`, `judge_pair(d_k, d_2k, z, q, f, k, spec) -> dict` (`status`, `reasons`, `mech_fail`, fly records), `judge(pairs, machine, z, q, f, k, spec) -> dict` (`verdict`, `machine`, `pairs`, `failing`, `mech_fail`, `undecided_cause`, `invalid`, `n_pairs`, `n_judgeable`); labels `GATES`, `SIGNS`, `PAIR_LABEL`, `VERDICT_LABEL`, indices `A, P, X, Y`.

- [ ] **Step 1: Write the failing tests**

```python
"""W's judgement code (W.4, W.9.1, W.9.2, W.9.8 H2 / H5 / H7, W.9.9 P1-2 / P2-9 / P2-10 / P2-11; G.7's defects; F.6's
fixtures moved to RN, z_V-style z and per-fly joint satisfaction). Known answers: specific learning PASS; no
learning, X-only presentation drift (F.6 path 1), punishment-phase drift with a powerless punishment (F.6 path 2),
full generalisation, a broken mechanism control and reward-only / punishment-only flies (joint vs marginal) FAIL; the
BAND boundaries 0.80 / 0.79 / −0.80; ±∞ by direction; NaN and K < 2 INVALID → STOP_MACHINE; a missing stage, a
mis-shaped stage and a short fly list INVALID; the scheduled-F denominator; BAND → 2K (PASS / FAIL / pending); RN1 one
bit off → STOP_MACHINE; H7's order. Mutations each flip a fixture: no RN1 check, the valid-fly denominator, a deleted
gate, the ±∞ direction ignored, the mechanism control ignored."""
import numpy as np
import pytest

from flymon.brain import w_verdict as WV
from flymon.brain.w_spec import SPEC

Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}           # V = A − P, ΔV = (A_X − P_X) − (A_Y − P_Y)
F, K = 8, 8


def pair(seed=0, a=0.0, b=0.0, drift1=0.0, drift2=0.0, gen=0.0, sd=1.0, f=F, k=K, ypx=0.0):
    """Counts around 50 with probe noise sd shared by every slot of a probe (paired noise) plus a small independent
    part; a = reward drop of MBON05(X) on R / RN, b = punishment drop of MBON13(X) on R2, drift1 / drift2 = X-only
    MBON13 drops after the first / second presentation block (every brain), gen = the share of every change Y gets
    too, ypx = a rise of MBON05(Y) at R1 / R2 / RN (a gate effect that is not on the taught cell)."""
    r = np.random.default_rng(seed)
    base = np.full((f, k, 2, 2), 50.0) + r.normal(0, sd, (f, k, 1, 1))
    d = {}
    for s in WV.STAGES:
        m = base + r.normal(0, sd * 0.5, (f, k, 2, 2))
        dx = np.zeros((2, 2))
        if s != "pre":
            dx[WV.A, WV.X] -= drift1
        if s in ("R2", "N2", "RN2"):
            dx[WV.A, WV.X] -= drift2
        if s in ("R1", "R2", "RN1", "RN2"):
            dx[WV.P, WV.X] -= a
            dx[WV.P, WV.Y] += ypx
        if s == "R2":
            dx[WV.A, WV.X] -= b
        dx[:, WV.Y] += gen * dx[:, WV.X]
        d[s] = np.clip(np.rint(m + dx), 0, None).astype(np.int64)
    d["RN1"] = d["R1"].copy()
    return d


def jp(d, d2=None, q=0.75, f=F, k=K):
    return WV.judge_pair(d, d2, Z, q, f, k, SPEC)


def test_dprime_limits_and_nan():
    assert float(WV.dprime([1.0, 1.0])) == np.inf and float(WV.dprime([-2.0, -2.0])) == -np.inf
    assert float(WV.dprime([0.0, 0.0])) == 0.0
    assert np.isnan(WV.dprime([1.0])) and np.isnan(WV.dprime([1.0, np.nan]))
    assert float(WV.dprime([1.0, 3.0])) == pytest.approx(2.0 / np.sqrt(2.0))
    x = np.array([[1.0, 1.0], [0.0, 2.0]])
    assert WV.dprime(x).tolist() == [np.inf, 1.0 / np.sqrt(2.0)]


def test_fly_class_boundaries_and_infinities():
    c = lambda s: int(WV.fly_class(np.array(s, float)))            # noqa: E731
    assert c([1.0, -1.0, 1.0, -1.0]) == WV.FLY_SAT
    assert c([0.80, -1.0, 1.0, -1.0]) == WV.FLY_BAND                # 0.80 is BAND (F.5)
    assert c([0.79, -1.0, 1.0, -1.0]) == WV.FLY_FAIL                # 0.79 is FAIL
    assert c([1.0, -0.80, 1.0, -1.0]) == WV.FLY_BAND
    assert c([1.0, -0.79, 1.0, -1.0]) == WV.FLY_FAIL
    assert c([np.inf, -np.inf, np.inf, -np.inf]) == WV.FLY_SAT      # H5: the gate's direction passes
    assert c([-np.inf, -1.0, 1.0, -1.0]) == WV.FLY_FAIL             # against it fails
    assert c([1.0, np.inf, 1.0, -1.0]) == WV.FLY_FAIL
    assert c([1.0, -1.0, np.nan, -1.0]) == WV.FLY_INVALID
    assert c([0.9, -0.9, 0.85, -0.95]) == WV.FLY_BAND


def test_pair_code_scheduled_denominator():
    S, B, Fl = WV.FLY_SAT, WV.FLY_BAND, WV.FLY_FAIL
    assert int(WV.pair_gate_code(np.array([S] * 6 + [Fl] * 2), 0.75, 8)) == WV.P_PASS      # 6/8 = 0.75
    assert int(WV.pair_gate_code(np.array([S] * 5 + [Fl] * 3), 0.75, 8)) == WV.P_FAIL
    assert int(WV.pair_gate_code(np.array([S] * 5 + [B] + [Fl] * 2), 0.75, 8)) == WV.P_BAND
    assert int(WV.pair_gate_code(np.array([S] * 5 + [B] + [Fl] * 2), 0.625, 8)) == WV.P_PASS
    assert int(WV.pair_gate_code(np.array([S] * 6), 0.75, 8)) == WV.P_INVALID           # 6 of 8 flies present
    assert int(WV.pair_gate_code(np.array([S] * 7 + [WV.FLY_INVALID]), 0.5, 8)) == WV.P_INVALID


def test_known_answers():
    assert jp(pair(a=8, b=8))["status"] == "PASS"
    assert jp(pair())["status"] == "FAIL"
    one = jp(pair(drift1=6))                                       # F.6 path 1: X-only drift, no DAN effect
    assert one["status"] == "FAIL" and one["failing_gates"]["reward_assoc"] >= 6
    two = jp(pair(a=8, drift2=6))                                  # F.6 path 2: punishment powerless, drift
    assert two["status"] == "FAIL" and two["failing_gates"]["punish_assoc"] >= 6
    assert jp(pair(a=8, b=8, gen=1.0))["status"] == "FAIL"        # full generalisation
    mech = jp(pair(b=8, ypx=8))                                    # gates pass through Y, MBON05(X) untouched
    assert mech["status"] == "FAIL" and mech["reasons"] == ["기계 대조 실패"] and mech["mech_fail"]


def test_joint_not_marginal():
    """Reward-only flies and punishment-only flies: every gate passes in half the flies (a per-gate median at q 0.5
    would pass), no fly passes all four — FAIL at every q (G.7: joint satisfaction)."""
    rw, pu = pair(a=8, seed=1), pair(b=8, seed=2)
    d = {s: np.concatenate([rw[s][:4], pu[s][:4]]) for s in WV.STAGES}
    st = WV.gate_stats(d, Z)
    med = np.median(st, axis=0) * WV.SIGNS                         # F v3's per-gate medians: each ≥ 0.8
    assert (med >= 0.8).all()
    for q in SPEC.q_grid:
        r = jp(d, q=q)
        assert r["status"] == "FAIL" and r["n_sat"] == 0


def test_band_resolution():
    for seed in range(400):                                        # find a BAND pair at K = 8 with these effects
        d2 = pair(a=2.2, b=2.2, seed=seed, k=2 * K)
        d = {s: v[:, :K] for s, v in d2.items()}
        if jp(d)["code_k"] == "BAND":
            break
    else:
        pytest.fail("no BAND fixture found")
    r = jp(d, d2)
    assert r["status"] in ("PASS", "FAIL") and r["code_2k"] in ("PASS", "BAND", "FAIL")
    assert r["status"] == ("PASS" if r["code_2k"] == "PASS" and r["mech_2k_ok"] else "FAIL")
    assert jp(d)["status"] == "BAND"                               # no 2K data: pending
    bad = {s: v.copy() for s, v in d2.items()}
    bad["R1"][0, 0, 0, 0] += 1
    assert jp(d, bad)["status"] == "INVALID"                       # the 2K data's first K probes must be K's


def test_invalid_data():
    d = pair(a=8, b=8)
    assert jp({k: v for k, v in d.items() if k != "N2"})["status"] == "INVALID"
    assert jp({s: v[:7] for s, v in d.items()})["status"] == "INVALID"          # 7 of 8 flies
    assert jp({s: v[:, :1] for s, v in d.items()}, k=1)["status"] == "INVALID"  # K < 2: NaN d′
    neg = {s: v.copy() for s, v in d.items()}
    neg["R2"][0, 0, 0, 0] = -1
    assert jp(neg)["status"] == "INVALID"


def _set(n=4, **kw):
    return {f"p{i}": (pair(seed=10 + i, **kw), None) for i in range(n)}


def test_overall_order():
    j = lambda pairs, mach=(): WV.judge(pairs, list(mach), Z, 0.75, F, K, SPEC)   # noqa: E731
    assert j(_set(a=8, b=8))["verdict"] == "PASS"
    three = j(_set(3, a=8, b=8))
    assert three["verdict"] == "UNDECIDED" and three["undecided_cause"] == "판정 가능 쌍 < 4"
    mixed = dict(_set(4, a=8, b=8), bad=(pair(), None))
    assert j(mixed)["verdict"] == "FAIL" and j(mixed)["failing"] == ["bad"]
    assert j(_set(a=8, b=8), ["fly 0: N pre ≠ R pre"])["verdict"] == "STOP_MACHINE"
    rn = _set(a=8, b=8)
    rn["p0"][0]["RN1"][0, 0, 1, 0] += 1                             # RN1 one bit off (W.9.9 P1-2)
    assert j(rn)["verdict"] == "STOP_MACHINE"
    inv = dict(_set(a=8, b=8), bad=({s: v[:, :1] for s, v in pair().items()}, None))
    assert j(inv)["verdict"] == "STOP_MACHINE" and j(inv)["invalid"] == ["bad"]
    for s in range(400):                                            # a BAND left beats nothing but FAIL / machine
        d = pair(a=2.2, b=2.2, seed=s)
        if jp(d)["code_k"] == "BAND":
            break
    band = dict(_set(a=8, b=8), band=(d, None))
    assert j(band)["verdict"] == "UNDECIDED" and j(band)["undecided_cause"] == "BAND 잔존"
    assert j(dict(band, bad=(pair(), None)))["verdict"] == "FAIL"


def test_naive_pooled():
    d = pair()
    assert abs(WV.naive_dprime(d["pre"], Z)) < 0.5
    pre = d["pre"].copy()
    pre[..., WV.A, WV.X] += 30
    assert WV.naive_dprime(pre, Z) > 0.5


# ================================================================ mutations (W.9.9 P2-11)
def test_mutation_no_rn1_check(monkeypatch):
    rn = _set(a=8, b=8)
    rn["p0"][0]["RN1"][0, 0, 1, 0] += 1
    assert WV.judge(rn, [], Z, 0.75, F, K, SPEC)["verdict"] == "STOP_MACHINE"
    monkeypatch.setattr(WV, "rn1_mismatch", lambda d: np.zeros(np.asarray(d["R1"]).shape[:-4], bool))
    assert WV.judge(rn, [], Z, 0.75, F, K, SPEC)["verdict"] != "STOP_MACHINE"


def test_mutation_valid_fly_denominator(monkeypatch):
    six = np.array([WV.FLY_SAT] * 6)
    assert int(WV.pair_gate_code(six, 0.75, 8)) == WV.P_INVALID
    real = WV.pair_gate_code
    monkeypatch.setattr(WV, "pair_gate_code", lambda cls, q, f, digits=9: real(cls, q, np.asarray(cls).shape[-1],
                                                                               digits))
    assert int(WV.pair_gate_code(six, 0.75, 8)) == WV.P_PASS


def test_mutation_deleted_gate(monkeypatch):
    """F.6 path 2 with the mechanism control switched off: the punishment association gate alone catches it."""
    d = pair(a=8, drift2=6)
    monkeypatch.setattr(WV, "mech_ok", lambda fr, mech_min=0.75, digits=9: np.ones(np.asarray(fr).shape[:-2], bool))
    assert jp(d)["status"] == "FAIL"
    real = WV.gate_stats
    monkeypatch.setattr(WV, "gate_stats", lambda dd, z: np.concatenate(
        [real(dd, z)[..., :3], np.full(real(dd, z)[..., 3:].shape, -np.inf)], -1))
    assert jp(d)["status"] == "PASS"


def test_mutation_direction_ignored(monkeypatch):
    st = np.array([-np.inf, -1.0, 1.0, -1.0])
    assert int(WV.fly_class(st)) == WV.FLY_FAIL
    monkeypatch.setattr(WV, "SIGNS", np.array([1.0, 1.0, 1.0, 1.0]))
    monkeypatch.setattr(WV, "margins", lambda s, bar=1.0, digits=9: np.round(np.abs(np.asarray(s, float)) - bar,
                                                                               digits))
    assert int(WV.fly_class(st)) == WV.FLY_SAT


def test_mutation_mechanism_ignored(monkeypatch):
    d = pair(b=8, ypx=8)
    assert jp(d)["status"] == "FAIL"
    monkeypatch.setattr(WV, "mech_ok", lambda fr, mech_min=0.75, digits=9: np.ones(np.asarray(fr).shape[:-2], bool))
    assert jp(d)["status"] == "PASS"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_verdict.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write `w_verdict`**

```python
"""W's judgement code (W.4 as amended by W.9.1, W.9.2, W.9.8 H1 / H2 / H5 / H7, W.9.9 P1-2 / P2-9 / P2-10): pure
functions over raw counts, vectorised over leading axes, so that w_oc runs THIS code on its simulated experiments
(W.9.8 H4) and the judge runs it on the measured ones. The frozen F v3 verdict.py is not imported (W.9.6 P1-6); G.7's
defects are fixed here: a NaN fly statistic or fewer than 2 probes is INVALID (never a pass), the fly denominator is
the scheduled F, a missing or mis-shaped stage is INVALID, BAND ends in a coded re-measure (K -> 2K, then BAND is
FAIL), and the aggregation is per-fly joint satisfaction.

Data layout: a pair's counts are {stage: array [..., F, K, 2, 2]}, stage in STAGES, axis -2 the readout cell (0 = A =
MBON13, 1 = P = MBON05, both cells of a type summed), axis -1 the odour (0 = X, the taught odour; 1 = Y).
- ΔV = V(X) − V(Y), V = z_A − z_P (z = z_V for L_V, block h4's z for C).
- Fly gates (d′ over the fly's K probes, F.5's limits: sd 0 → 0 if the mean is 0, else ±∞): reward level d′(ΔV_R1)
  ≥ +1, punishment drop d′(ΔV_R2 − ΔV_R1) ≤ −1, reward association d′(ΔV_R1 − ΔV_N1) ≥ +1, punishment association
  d′((ΔV_R2 − ΔV_R1) − (ΔV_RN2 − ΔV_RN1)) ≤ −1. Margin = round(sign·d′ − 1, 9): ±∞ in the gate's direction passes,
  against it fails (H5).
- A fly is satisfied (every margin ≥ 0), BAND (every margin ≥ −0.2, one < 0) or failed; any NaN → the fly is invalid.
- Pair (W.9.1 with the design's q): q_sat = n_sat / F ≥ q → PASS; else q_sat + n_band / F ≥ q → BAND; else FAIL;
  an invalid fly → INVALID. BAND is resolved on the same flies with 2K probes (k 0..K−1 kept, K..2K−1 added):
  PASS → PASS, otherwise FAIL.
- Mechanism control (W.9.2 / H2), same flies and probes (after 2K for a BAND pair, P2-9): per fly the share of
  probes with MBON05(X) at R1 below N1, and with MBON13(X)'s change R2 − R1 below RN2 − RN1 (ties ½); both fly
  medians ≥ 0.75 or the pair is FAIL ("기계 대조 실패").
- Overall (H7): RN1 ≠ R1 bit for bit, or an INVALID pair, or another machine reason → STOP_MACHINE; a FAIL among the
  judgeable pairs → FAIL; a BAND left → UNDECIDED; fewer than 4 judgeable pairs → UNDECIDED; all PASS → PASS.
- Naive (W.9.5, W.9.9 P2-10): the pooled d′ of ΔV_pre over every fly's screening probes (F × K), fixed at the screen."""
from __future__ import annotations

import numpy as np

STAGES = ("pre", "R1", "R2", "N1", "N2", "RN1", "RN2")
GATES = ("reward_level", "punish_drop", "reward_assoc", "punish_assoc")
SIGNS = np.array([1.0, -1.0, 1.0, -1.0])
FLY_INVALID, FLY_FAIL, FLY_BAND, FLY_SAT = -1, 0, 1, 2
P_INVALID, P_FAIL, P_BAND, P_PASS = -1, 0, 1, 2
PAIR_LABEL = {P_INVALID: "INVALID", P_FAIL: "FAIL", P_BAND: "BAND", P_PASS: "PASS"}
V_STOP_MACHINE, V_FAIL, V_UNDECIDED, V_PASS = -1, 0, 1, 2
VERDICT_LABEL = {V_STOP_MACHINE: "STOP_MACHINE", V_FAIL: "FAIL", V_UNDECIDED: "UNDECIDED", V_PASS: "PASS"}
A, P, X, Y = 0, 1, 0, 1


def dprime(x, axis: int = -1):
    """F.5's d′ along `axis`: mean / sd (ddof 1); sd 0 → 0.0 when the mean is 0, else ±∞; fewer than 2 values or a
    NaN → NaN."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    if n < 2:
        return np.full(np.delete(np.array(x.shape), axis % x.ndim), np.nan) if x.ndim > 1 else np.float64(np.nan)
    m = x.mean(axis)
    sd = x.std(axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = m / sd
    lim = np.where(m == 0, 0.0, np.copysign(np.inf, m))
    return np.where(sd == 0, lim, d)


def dv(arr, z: dict):
    """ΔV per probe: arr [..., 2 cells, 2 odours] -> [...]."""
    a = np.asarray(arr, float)
    v = (a[..., A, :] - z["A"][0]) / z["A"][1] - (a[..., P, :] - z["P"][0]) / z["P"][1]
    return v[..., X] - v[..., Y]


def gate_stats(d: dict, z: dict):
    """[..., F, 4] fly d′ of the four gates (GATES order) over the probe axis."""
    r1, r2, n1 = dv(d["R1"], z), dv(d["R2"], z), dv(d["N1"], z)
    rn1, rn2 = dv(d["RN1"], z), dv(d["RN2"], z)
    return np.stack([dprime(r1), dprime(r2 - r1), dprime(r1 - n1), dprime((r2 - r1) - (rn2 - rn1))], axis=-1)


def margins(stats, bar: float = 1.0, digits: int = 9):
    with np.errstate(invalid="ignore"):
        return np.round(SIGNS * np.asarray(stats, float) - bar, digits)


def fly_class(stats, bar: float = 1.0, band: float = 0.2, digits: int = 9):
    """[..., F] FLY_SAT / FLY_BAND / FLY_FAIL / FLY_INVALID."""
    m = margins(stats, bar, digits)
    bad = np.isnan(m).any(-1)
    with np.errstate(invalid="ignore"):
        sat = (m >= 0).all(-1)
        fail = (m < -band).any(-1)
    return np.where(bad, FLY_INVALID, np.where(sat, FLY_SAT, np.where(fail, FLY_FAIL, FLY_BAND)))


def lower_fraction(a, b):
    """Share of probes (last axis) with a < b; ties count ½."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ((a < b).sum(-1) + 0.5 * (a == b).sum(-1)) / a.shape[-1]


def mech_fractions(d: dict):
    """[..., F, 2]: reward MBON05(X) R1 < N1; punishment MBON13(X) change R2 − R1 < RN2 − RN1."""
    px = lambda s: np.asarray(d[s], float)[..., P, X]      # noqa: E731
    ax = lambda s: np.asarray(d[s], float)[..., A, X]      # noqa: E731
    return np.stack([lower_fraction(px("R1"), px("N1")),
                     lower_fraction(ax("R2") - ax("R1"), ax("RN2") - ax("RN1"))], axis=-1)


def mech_ok(fr, mech_min: float = 0.75, digits: int = 9):
    """[...] both fly medians (axis -2) ≥ mech_min."""
    med = np.median(np.asarray(fr, float), axis=-2)
    return (np.round(med - mech_min, digits) >= 0).all(-1)


def pair_gate_code(cls, q: float, f_sched: int, digits: int = 9):
    """[...] P_PASS / P_BAND / P_FAIL / P_INVALID from fly classes [..., F] with the scheduled F as denominator."""
    cls = np.asarray(cls)
    n_sat, n_band = (cls == FLY_SAT).sum(-1), (cls == FLY_BAND).sum(-1)
    bad = (cls == FLY_INVALID).any(-1) | (cls.shape[-1] != f_sched)
    ok = np.round(n_sat / f_sched - q, digits) >= 0
    band = np.round((n_sat + n_band) / f_sched - q, digits) >= 0
    return np.where(bad, P_INVALID, np.where(ok, P_PASS, np.where(band, P_BAND, P_FAIL)))


def pair_final(code_k, mech_k, code_2k, mech_2k):
    """[...]: BAND → the 2K code (PASS stays only with the 2K mechanism control; BAND or FAIL → FAIL); PASS → PASS iff
    the K mechanism control passes; FAIL stays; INVALID anywhere → INVALID."""
    code_k, code_2k = np.asarray(code_k), np.asarray(code_2k)
    band_res = np.where(code_2k == P_INVALID, P_INVALID, np.where((code_2k == P_PASS) & mech_2k, P_PASS, P_FAIL))
    return np.where(code_k == P_INVALID, P_INVALID,
                    np.where(code_k == P_BAND, band_res,
                             np.where(code_k == P_PASS, np.where(mech_k, P_PASS, P_FAIL), P_FAIL)))


def rn1_mismatch(d: dict):
    """[...] over pairs: RN1 ≠ R1 anywhere (G.5's machine check, W.9.9 P1-2)."""
    r1, rn1 = np.asarray(d["R1"]), np.asarray(d["RN1"])
    return (r1 != rn1).reshape(r1.shape[:-4] + (-1,)).any(-1)


def overall_code(final, machine, min_pairs: int = 4):
    """[...]: H7's order over the pair axis (-1): machine or INVALID → STOP_MACHINE; FAIL → FAIL; BAND → UNDECIDED;
    judgeable < min_pairs → UNDECIDED; all PASS → PASS."""
    final = np.asarray(final)
    stop = np.asarray(machine) | (final == P_INVALID).any(-1)
    n = final.shape[-1]
    return np.where(stop, V_STOP_MACHINE, np.where((final == P_FAIL).any(-1), V_FAIL,
                    np.where((final == P_BAND).any(-1), V_UNDECIDED,
                             np.where(n < min_pairs, V_UNDECIDED, V_PASS))))


def naive_dprime(pre, z: dict) -> float:
    """The pooled naive d′ over every probe of every fly (pre [F, K, 2, 2])."""
    return float(dprime(dv(pre, z).reshape(-1)))


# ================================================================ the real-data judge (one pair, then the set)
def data_reasons(d: dict, f_sched: int, k: int) -> list:
    """G.7's missing-data refusal: every stage present, shape (F, K, 2, 2), non-negative integer counts."""
    out = []
    for s in STAGES:
        if s not in d:
            out.append(f"{s} 없음")
            continue
        a = np.asarray(d[s])
        if a.shape != (f_sched, k, 2, 2):
            out.append(f"{s} 모양 {a.shape} ≠ ({f_sched}, {k}, 2, 2)")
        elif not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
            out.append(f"{s} 카운트가 음이 아닌 정수가 아님")
    return out


def judge_pair(d_k: dict, d_2k: dict | None, z: dict, q: float, f_sched: int, k: int, spec) -> dict:
    """One gate pair: d_k = the K-probe counts, d_2k = the same flies with 2K probes (needed only when the K read is
    BAND). Records every fly statistic."""
    bad = data_reasons(d_k, f_sched, k)
    if bad:
        return dict(status="INVALID", reasons=bad)
    kw = dict(bar=spec.bar, band=spec.band_width, digits=spec.round_digits)
    st = gate_stats(d_k, z)
    cls = fly_class(st, **kw)
    code = int(pair_gate_code(cls, q, f_sched, spec.round_digits))
    fr = mech_fractions(d_k)
    mk = bool(mech_ok(fr, spec.mech_min, spec.round_digits))
    out = dict(stats=st.tolist(), classes=cls.tolist(), code_k=PAIR_LABEL[code], mech_k=fr.tolist(),
               mech_k_ok=mk, n_sat=int((cls == FLY_SAT).sum()), n_band=int((cls == FLY_BAND).sum()),
               q_sat=float((cls == FLY_SAT).sum() / f_sched), K=k)
    c2, m2 = P_INVALID, False
    if code == P_BAND:
        if d_2k is None:
            return dict(out, status="BAND", reasons=["2K 재측정 자료 없음"])
        bad2 = data_reasons(d_2k, f_sched, 2 * k)
        if bad2:
            return dict(out, status="INVALID", reasons=bad2)
        if any(not np.array_equal(np.asarray(d_2k[s])[:, :k], np.asarray(d_k[s])) for s in STAGES):
            return dict(out, status="INVALID", reasons=["2K 자료의 앞 K 프로브가 K 자료와 다름"])
        st2 = gate_stats(d_2k, z)
        cls2 = fly_class(st2, **kw)
        c2 = int(pair_gate_code(cls2, q, f_sched, spec.round_digits))
        fr2 = mech_fractions(d_2k)
        m2 = bool(mech_ok(fr2, spec.mech_min, spec.round_digits))
        n2 = int((cls2 == FLY_SAT).sum())
        out.update(stats_2k=st2.tolist(), classes_2k=cls2.tolist(), code_2k=PAIR_LABEL[c2], mech_2k=fr2.tolist(),
                   mech_2k_ok=m2, n_sat_2k=n2, q_sat_2k=float(n2 / f_sched))
    fin = int(pair_final(code, mk, c2, m2))
    reasons = []
    if fin == P_FAIL:
        used_cls, used_m = (cls2, m2) if code == P_BAND else (cls, mk)
        gate_code = c2 if code == P_BAND else code
        if gate_code != P_PASS:
            reasons.append("마리별 동시 충족 미달")
        if not used_m:
            reasons.append("기계 대조 실패")
        out["failing_gates"] = _failing_gates(st2 if code == P_BAND else st, used_cls, spec)
    return dict(out, status=PAIR_LABEL[fin], reasons=reasons, mech_fail=bool("기계 대조 실패" in reasons))


def _failing_gates(st, cls, spec) -> dict:
    m = margins(st, spec.bar, spec.round_digits)
    out = {}
    for j, g in enumerate(GATES):
        with np.errstate(invalid="ignore"):
            out[g] = int((m[:, j] < 0).sum())
    return out


def judge(pairs: dict, machine: list, z: dict, q: float, f_sched: int, k: int, spec) -> dict:
    """pairs = {pair key: (d_k, d_2k or None)} in declared order; machine = the runner's machine reasons (pre equal
    across brains and with the screen, band-job weights equal the main job's, …). H7's order."""
    res = {key: judge_pair(dk, d2, z, q, f_sched, k, spec) for key, (dk, d2) in pairs.items()}
    mm = [f"{key}: RN1 ≠ R1" for key, (dk, _) in pairs.items()
          if not data_reasons(dk, f_sched, k) and bool(rn1_mismatch({s: np.asarray(dk[s])[None] for s in STAGES})[0])]
    codes = np.array([{v: c for c, v in PAIR_LABEL.items()}[r["status"]] for r in res.values()])
    mach = list(machine) + mm
    code = int(overall_code(codes, bool(mach), spec.min_gate_pairs)) if len(codes) else V_UNDECIDED
    fails = [key for key, r in res.items() if r["status"] == "FAIL"]
    why = ""
    if code == V_UNDECIDED:
        why = "BAND 잔존" if any(r["status"] == "BAND" for r in res.values()) else \
            f"판정 가능 쌍 < {spec.min_gate_pairs}"
    return dict(verdict=VERDICT_LABEL[code], machine=mach, pairs=res, failing=fails,
                mech_fail=[key for key in fails if res[key].get("mech_fail")], undecided_cause=why,
                invalid=[key for key, r in res.items() if r["status"] == "INVALID"], n_pairs=len(res),
                n_judgeable=int(sum(r["status"] in ("PASS", "FAIL", "BAND") for r in res.values())))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_w_verdict.py -q`
Expected: 14 passed (seconds).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/w_verdict.py tests/brain/test_w_verdict.py
git commit -m "feat(w): w_verdict — per-fly joint satisfaction with q and the scheduled F, BAND -> 2K, mechanism control as pair FAIL, ±inf by direction, NaN/K<2 INVALID, H7 order; F.6 fixtures on RN and mutations"
```

---

### Task 3: `w_oc` — the probe-level operating characteristic that runs the W verdict code

**Files:**
- Create: `flymon/brain/w_oc.py`
- Test: `tests/brain/test_w_oc.py`

**Interfaces:**
- Consumes: `w_verdict` (Task 2), `w_spec.SPEC`.
- Produces: `fit(pilot: [{stage: [F, K, 2, 2]}]) -> θ dict`; `boot(θ, rng) -> θ`; `summary(θ) -> dict`; `slot_means`, `to_stages(counts) -> {stage: …}` (RN1 = R1 copied); `simulate(θ, rng, n_rep, n_pair, n_fly, n_probe, a_pair, b_pair, fly_a, fly_b, g, z, naive_max)`; `true_dprimes(θ, a, b, idx, z) -> [4]`; `calibrate(θ, target, "min" | "max", idx, z, spec) -> {ok, a, b, true_dprime}`; `evaluate(θ, rng, n_rep, a_pair, b_pair, fly_a, fly_b, g, z, spec) -> P(PASS) [q, K, F, k]`; `qualify(power_lo, false_hi, spec) -> [q, K, F] bool`; `select(qual, cost, spec) -> (design | None, ranking)`; `run(pilot, z, spec, cost, n_boot=None, n_rep=None, n_boot_rep=None, log=None) -> doc` (`theta`, `calibration`, `reachable`, `point`, `power_lo`, `false_hi`, `qualified`, `selected`, `ranking`, `records`, `timing`, …); `records(...)`; `synthetic_pilot(rng, …) -> list`; `simple_normal(spec, rng, …) -> {F: p}`; `synthetic_validation(spec, z, n_rep=1000) -> dict` (`ok`); `TAG_SYNTH`.

- [ ] **Step 1: Write the failing tests**

```python
"""W's operating characteristic (W.9.3, W.9.8 H1 / H4, W.9.9 P0-1 / P1-2 / P1-3 / P1-6 / P2-8 / P2-11): the fit
recovers a known synthetic pilot; the calibration meets the gate-vector targets within ±0.02 (min = 1.5 power, max =
0.5 false pass) and reports floor / above-at-zero; the evaluation runs the W verdict code itself (a verdict that
always fails gives P(PASS) 0) on nested experiments (P(PASS) never rises with k); an RN1 bit flipped in the simulator
stops the machine; the run is deterministic; the envelope / solo rule and the cost-then-q-then-K order; the synthetic
validation's tolerances (W.9.9 P2-11)."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain.w_spec import SPEC

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
SP = dataclasses.replace(SPEC, oc_chunk=50)


@pytest.fixture(scope="module")
def theta():
    rng = np.random.default_rng(3)
    return w_oc.fit(w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0)))


def test_fit_recovers_the_synthetic_structure():
    rng = np.random.default_rng(3)
    theta = w_oc.fit(w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0), drift_ax=4.0))
    assert theta["m0s"].shape == (2, 2) and np.allclose(theta["m0s"][0], 40.0, atol=1.0)
    assert np.allclose(theta["m0s"][1], 90.0, atol=1.0) and theta["m0s"][0, 0] == theta["m0s"][0, 1]
    n1, n2, rn2 = theta["drift"]
    assert abs(n2[WV.A, WV.X] + 4.0) < 0.5 and abs(rn2[WV.A, WV.X] + 4.0) < 0.5 and abs(n1[WV.A, WV.X]) < 0.5
    assert theta["spill"] == (0.0, 0.0) or max(theta["spill"]) < 0.05
    assert len(theta["resid"]) == 16 * 8 * 8 and theta["resid"].shape[1:] == (6, 2, 2)
    assert np.all(np.linalg.eigvalsh(theta["fly_cov"]) >= -1e-9)
    tb = w_oc.boot(theta, np.random.default_rng(0))
    assert tb["pair_cov"].shape == (4, 4) and len(tb["resid_blocks"]) == 16 and not np.allclose(tb["m0"], theta["m0"])


def test_calibration_meets_the_gate_vector_targets(theta):
    idx = np.random.default_rng(1).integers(0, len(theta["resid"]), SP.cal_reps)
    c = w_oc.calibrate(theta, 1.5, "min", idx, Z, SP)
    td = np.array(list(c["true_dprime"].values()))
    assert c["ok"] and abs(min(td[0], td[2]) - 1.5) <= SP.cal_tol and abs(min(td[1], td[3]) - 1.5) <= SP.cal_tol
    f = w_oc.calibrate(theta, 0.5, "max", idx, Z, SP)
    td = np.array(list(f["true_dprime"].values()))
    assert f["ok"] and abs(max(td[0], td[2]) - 0.5) <= SP.cal_tol and abs(max(td[1], td[3]) - 0.5) <= SP.cal_tol
    hi = w_oc.calibrate(theta, 40.0, "min", idx, Z, SP)
    assert not hi["ok"] and hi["a"]["status"] in ("floor", "no_convergence")


def test_above_at_zero_is_unreachable_unless_the_zero_rule(theta):
    """OPEN 2: a drift that alone puts the punishment drop above 0.5 at b = 0."""
    rng = np.random.default_rng(4)
    th = w_oc.fit(w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0), drift_ax=15.0))
    idx = rng.integers(0, len(th["resid"]), SP.cal_reps)
    f = w_oc.calibrate(th, 0.5, "max", idx, Z, SP)
    assert not f["ok"] and f["b"]["status"] == "above_at_zero"
    z = w_oc.calibrate(th, 0.5, "max", idx, Z, dataclasses.replace(SP, cal_floor_rule="zero"))
    assert z["ok"] and z["b"]["status"] == "zero_floor" and z["b"]["value"] == 0.0


def test_evaluation_runs_the_verdict_code(theta, monkeypatch):
    rng = np.random.default_rng(2)
    args = w_oc._uniform(SP, 30.0, 15.0)
    p = w_oc.evaluate(theta, rng, 100, *args, 0.0, Z, SP)
    assert p.shape == (3, 2, 25, 5) and p.max() > 0.5
    assert np.all(np.diff(p, axis=-1) <= 1e-12)                    # nested pairs: more pairs never pass more
    monkeypatch.setattr(WV, "pair_final", lambda *a: np.full(np.asarray(a[0]).shape, WV.P_FAIL))
    assert w_oc.evaluate(theta, np.random.default_rng(2), 100, *args, 0.0, Z, SP).max() == 0.0


def test_rn1_bit_in_the_simulator_stops_the_machine(theta, monkeypatch):
    real = w_oc.to_stages

    def flipped(c):
        d = real(c)
        d["RN1"] = d["RN1"].copy()
        d["RN1"][..., 0, 0, 0, 0] += 1
        return d
    args = w_oc._uniform(SP, 30.0, 15.0)
    assert w_oc.evaluate(theta, np.random.default_rng(2), 50, *args, 0.0, Z, SP).max() > 0.5
    monkeypatch.setattr(w_oc, "to_stages", flipped)
    assert w_oc.evaluate(theta, np.random.default_rng(2), 50, *args, 0.0, Z, SP).max() == 0.0


def test_determinism(theta):
    args = w_oc._uniform(SP, 25.0, 12.0)
    a = w_oc.evaluate(theta, np.random.default_rng(9), 60, *args, 0.5, Z, SP)
    b = w_oc.evaluate(theta, np.random.default_rng(9), 60, *args, 0.5, Z, SP)
    assert np.array_equal(a, b)


def test_qualify_envelope_and_solo():
    nf = SP.f_max - SP.f_min + 1
    lo = np.zeros((3, 2, nf))
    hi = np.ones((3, 2, nf))
    ok_f = [10, 11, 12, 13, 15, 29]                                 # F 14 fails: 10 needs 10-13 → qualifies
    for F in ok_f:
        lo[2, 0, F - SP.f_min], hi[2, 0, F - SP.f_min] = 0.9, 0.0
    q = w_oc.qualify(lo, hi, SP)
    got = [SP.f_min + i for i in np.flatnonzero(q[2, 0])]
    assert got == [10, 29]                                          # 11: 11-14 has 14; 29 alone (solo rule)
    lo[2, 0, 30 - SP.f_min], hi[2, 0, 30 - SP.f_min] = 0.9, 0.06    # false pass above 0.05
    assert not w_oc.qualify(lo, hi, SP)[2, 0, 30 - SP.f_min]


def test_select_cost_then_q_then_k():
    nf = SP.f_max - SP.f_min + 1
    qual = np.zeros((3, 2, nf), bool)
    qual[0, 0, 2] = qual[2, 0, 2] = qual[2, 1, 0] = True
    sel, rank = w_oc.select(qual, lambda K, F: K * F / 64, SP)
    assert sel == dict(q=0.75, K=8, F=10, cost_h=1.25) and [(r["q"], r["K"], r["F"]) for r in rank] == [
        (0.75, 8, 10), (0.5, 8, 10), (0.75, 16, 8)]
    assert w_oc.select(np.zeros_like(qual), lambda K, F: 1.0, SP) == (None, [])


def test_run_small_is_deterministic(theta):
    rng = np.random.default_rng(3)
    pilot = w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    a = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=2, n_rep=60, n_boot_rep=30)
    b = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=2, n_rep=60, n_boot_rep=30)
    for k in ("power_lo", "false_hi", "qualified", "selected", "ranking", "point"):
        assert a[k] == b[k], k
    assert set(a["timing"]) == {"point_s", "boot_s", "records_s", "total_s"} and a["reachable"]


def test_synthetic_validation_meets_p2_11():
    r = w_oc.synthetic_validation(SP, Z, n_rep=300)
    assert r["ok"], r
    assert r["zero_effect"]["max_p"] <= 0.02 and r["big_effect"]["min_p"] >= 0.98
    assert r["one_gate"]["max_p"] <= 0.02 and r["negative_correlation"]["max_p"] <= 0.02
    sn = r["simple_normal"]["p_by_F"]
    assert sn["32"] <= sn["8"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_oc.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write `w_oc`**

```python
"""W's operating characteristic (W.9.3 as amended by W.9.8 H1 / H4 and W.9.9 P0-1 / P1-2 / P1-3 / P1-6 / P2-8 /
P2-11), committed with the verdict code before the pilot (W.9.9 순서 0). Nothing here runs the engine.

Model (probe level, fitted to the pilot; plan Readings 9-11):
- Slots SLOTS = pre, R1, R2, N1, N2, RN2 (RN1 is R1 copied, P1-2), counts per (cell A/P, odour X/Y).
- A pair's naive mean m = m0s + u + w: m0s = the pilot's grand naive mean with X and Y averaged per cell (a
  naive-balanced population pair), u ~ N(0, Σ_pair) the 2 × 2 pair random effect (cell × odour), w ~ N(0, g·Σ_pair)
  the cluster random effect shared by every pair of an experiment (g on the grid 0 / 0.5 / 1, the worst value used,
  P1-6). u + w is drawn conditioned on the pair being naive-balanced (|E ΔV_pre| / sd_pre < 0.5): gate pairs are the
  pairs that passed the naive screen and the OC counts no naive dropout (W.9.3).
- A fly adds v ~ N(0, Σ_fly) (same 2 × 2 structure) to every post-training slot; pre has no fly effect (one naive
  brain).
- Fixed stage effects from the pilot (P1-6): N1 = m + v + δ_N1, N2 = N1 + δ_N2, R1 = N1 + E1, RN2 = R1 + δ_RN2,
  R2 = RN2 + E2, with E1 = −a on MBON05(X) and −s_P·a on MBON05(Y), E2 = −b on MBON13(X) and −s_A·b on MBON13(Y)
  (s = the pilot's Y spillover, ratio of mean changes; s = 1 when the pilot's X change is not a depression).
- Probe noise: whole residual vectors (6 slots × 2 × 2, around each pilot fly's slot means, scaled by √(K/(K−1)))
  resampled from the pilot — the paired-noise correlation across slots, cells and odours is kept. Counts are
  max(0, rint(mean + residual)): the floor truncation.
- Calibration (W.9.9 P0-1): the population pair (u = w = v = 0) over cal_reps residual draws (common random numbers)
  gives each gate's true d′ (mean / sd over the draws). a is solved by bisection so that min(reward level, reward
  association) = 1.5 (power) or max(...) = 0.5 (false pass); then b, given a, for the punishment drop and
  association. Tolerance ±0.02, at most 40 steps; a target not bracketed on [0, the floor] or no convergence →
  unreachable (cal_floor_rule "unreachable"; "zero" = use the zero effect when zero already exceeds the false-pass
  target — the alternative of OPEN 2).
- Experiments: 8 pairs × 32 flies × 2·max(K) probes generated once per replicate and cut to every design (nested,
  common random numbers, P2-8); the W verdict code (w_verdict) runs on them: fly gates, fly classes, the pair code
  with q and F, the mechanism control, the 2K BAND resolution, RN1 = R1, the overall order. P(PASS)[q, K, F, k].
- Uncertainty (P1-3): a pair-level parametric bootstrap (boot_draws): J new pilot pairs drawn from the fitted model
  (pair means ~ N(m0, Σ_pair), per-pair drifts and spillover changes ~ their pilot normals, fly effects ~ N(0,
  Σ_fly) re-estimated, residual blocks resampled by pair) and refitted; each draw recalibrated and simulated with
  boot_reps experiments. Simultaneous one-sided limits over k: power = the 5th percentile over draws of min_k
  P(PASS), false pass = the 95th percentile of max_k P(PASS); the worst over the cluster grid.
- Selection (H1, W.9.9): a design (q, K, F) qualifies when its limits meet G.6 at every k (power ≥ 0.80, false pass
  ≤ 0.05) for F, F+1, F+2, F+3 (F ≥ 29: F alone); the cheapest qualifying design by the total wall-clock estimate
  wins (ties: larger q, then smaller K, then smaller F); every qualifying design in cost order is the alternative
  ranking (P1-4). None → STOP_OC_UNREACHABLE."""
from __future__ import annotations

import math
import time

import numpy as np

from . import w_verdict as WV

SLOTS = ("pre", "R1", "R2", "N1", "N2", "RN2")
SI = {s: i for i, s in enumerate(SLOTS)}
POST = (SI["R1"], SI["R2"], SI["N1"], SI["N2"], SI["RN2"])
# Every random stream is np.random.SeedSequence([oc_seed, tag, ...]) — deterministic, no Python hash.
TAG_CAL, TAG_POINT, TAG_BOOT, TAG_BOOT_SIM, TAG_RECORD, TAG_SYNTH = 1, 2, 3, 4, 5, 9
RECORD_TAGS = ("t0.5", "t1.0", "t1.5", "t2.0", "one", "half", "all1", "mixed")


def _rng(spec, *tags):
    return np.random.default_rng(np.random.SeedSequence([int(spec.oc_seed)] + [int(t) for t in tags]))


# ================================================================ fitting
def _stack(pilot: list) -> np.ndarray:
    """[J, F, K, 6, 2, 2] float from the pilot pairs' {stage: [F, K, 2, 2]} (RN1 is not a slot)."""
    return np.stack([np.stack([np.asarray(p[s], float) for s in SLOTS], axis=2) for p in pilot])


def _cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).reshape(len(x), -1)
    return np.cov(x, rowvar=False) if len(x) > 1 else np.zeros((x.shape[1], x.shape[1]))


def _psd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    return (v * np.clip(w, 0, None)) @ v.T


def _spill(x: float, y: float) -> float:
    return max(0.0, y / x) if x < 0 else 1.0


def assemble(pair_means, drift_pair, spill_pair, fly_cov, resid_blocks, n_fly: int, n_probe: int) -> dict:
    pm = np.asarray(pair_means, float)
    m0 = pm.mean(0)
    dp, sp = np.asarray(drift_pair, float), np.asarray(spill_pair, float)
    smean = sp.mean(0)
    return dict(pair_means=pm, m0=m0, m0s=np.repeat(m0.mean(-1, keepdims=True), 2, -1), pair_cov=_cov(pm),
                drift_pair=dp, drift=dp.mean(0), spill_pair=sp, spill=(_spill(smean[0, 0], smean[0, 1]),
                                                                       _spill(smean[1, 0], smean[1, 1])),
                fly_cov=_psd(np.asarray(fly_cov, float)), resid_blocks=[np.asarray(b, float) for b in resid_blocks],
                resid=np.concatenate(resid_blocks), n_fly=int(n_fly), n_probe=int(n_probe))


def fit(pilot: list) -> dict:
    """θ from the pilot pairs ({stage: int [F, K, 2, 2]}, K ≥ 2, F ≥ 2)."""
    a = _stack(pilot)
    J, F, K = a.shape[:3]
    pre, r1, r2, n1, n2, rn2 = (a[:, :, :, i] for i in range(6))
    pair_means = pre.mean((1, 2))
    drift_pair = np.stack([(n1 - pre).mean((1, 2)), (n2 - n1).mean((1, 2)), (rn2 - r1).mean((1, 2))], 1)
    rew, pun = (r1 - n1).mean((1, 2)), ((r2 - r1) - (rn2 - r1)).mean((1, 2))
    spill_pair = np.stack([rew[:, WV.P, :], pun[:, WV.A, :]], 1)              # [J, 2 (reward P, punish A), 2 (X, Y)]
    x = a[:, :, :, list(POST)].mean(3).reshape(J, F, K, 4)
    mf = x.mean(2)
    wcov = np.einsum("jfki,jfkl->il", x - mf[:, :, None], x - mf[:, :, None]) / (J * F * (K - 1))
    dm = mf - mf.mean(1, keepdims=True)
    bcov = np.einsum("jfi,jfl->il", dm, dm) / (J * (F - 1))
    resid = (a - a.mean(2, keepdims=True)) * math.sqrt(K / (K - 1))
    blocks = [resid[j].reshape(F * K, 6, 2, 2) for j in range(J)]
    return assemble(pair_means, drift_pair, spill_pair, bcov - wcov / K, blocks, F, K)


def boot(theta: dict, rng) -> dict:
    """One pair-level parametric bootstrap draw (module docstring)."""
    J, F = len(theta["pair_means"]), theta["n_fly"]
    pm = rng.multivariate_normal(theta["m0"].ravel(), theta["pair_cov"], J, method="eigh").reshape(J, 2, 2)
    dp = rng.multivariate_normal(theta["drift"].ravel(), _cov(theta["drift_pair"]), J, method="eigh")
    sp = rng.multivariate_normal(theta["spill_pair"].mean(0).ravel(), _cov(theta["spill_pair"]), J, method="eigh")
    v = rng.multivariate_normal(np.zeros(4), theta["fly_cov"], J * max(F - 1, 1), method="eigh")
    fly_cov = v.T @ v / len(v)
    pick = rng.integers(0, J, J)
    return assemble(pm, dp.reshape(J, 3, 2, 2), sp.reshape(J, 2, 2), fly_cov,
                    [theta["resid_blocks"][i] for i in pick], F, theta["n_probe"])


def summary(theta: dict) -> dict:
    return dict(m0=theta["m0"].tolist(), m0s=theta["m0s"].tolist(), pair_cov=theta["pair_cov"].tolist(),
                fly_cov=theta["fly_cov"].tolist(), drift=dict(zip(("N1-pre", "N2-N1", "RN2-RN1"),
                                                                 theta["drift"].tolist())),
                spill=dict(P=theta["spill"][0], A=theta["spill"][1]), n_pairs=len(theta["pair_means"]),
                n_resid=int(len(theta["resid"])), n_fly=theta["n_fly"], n_probe=theta["n_probe"])


# ================================================================ the generator
def slot_means(theta: dict, base, v, a, b) -> np.ndarray:
    """[..., 6, 2, 2] slot means from the naive base [..., 2, 2], the fly effect v [..., 2, 2] and the effects a, b."""
    base, v = np.asarray(base, float), np.asarray(v, float)
    a, b = np.asarray(a, float)[..., None, None], np.asarray(b, float)[..., None, None]
    sP, sA = theta["spill"]
    d = theta["drift"]
    e1 = np.zeros((2, 2)); e1[WV.P, WV.X], e1[WV.P, WV.Y] = -1.0, -sP
    e2 = np.zeros((2, 2)); e2[WV.A, WV.X], e2[WV.A, WV.Y] = -1.0, -sA
    n1 = base + v + d[0]
    r1 = n1 + a * e1
    rn2 = r1 + d[2]
    return np.stack([base + 0 * v, r1, rn2 + b * e2, n1, n1 + d[1], rn2], axis=-3)


def to_stages(counts: np.ndarray) -> dict:
    """counts [..., K, 6, 2, 2] -> {stage: [..., K, 2, 2]} with RN1 = R1 (P1-2)."""
    d = {s: counts[..., i, :, :] for s, i in SI.items()}
    d["RN1"] = d["R1"].copy()
    return d


def sd_pre(theta: dict, z: dict) -> float:
    pre = np.clip(np.rint(theta["m0s"] + theta["resid"][:, SI["pre"]]), 0, None)
    return float(np.std(WV.dv(pre, z), ddof=1))


def _pair_bases(theta, rng, n_rep, n_pair, g, z, naive_max, sdp):
    """[n_rep, n_pair, 2, 2] naive bases conditioned on balance; the cluster effect is shared within a replicate."""
    L = np.linalg.cholesky(theta["pair_cov"] + 1e-9 * np.eye(4))
    w = (rng.standard_normal((n_rep, 4)) @ L.T * math.sqrt(g)).reshape(n_rep, 1, 2, 2)
    out = np.empty((n_rep, n_pair, 2, 2))
    need = np.ones((n_rep, n_pair), bool)
    for _ in range(200):
        u = (rng.standard_normal((n_rep, n_pair, 4)) @ L.T).reshape(n_rep, n_pair, 2, 2)
        base = theta["m0s"] + w + u
        ok = np.abs(WV.dv(base, z)) / sdp < naive_max
        take = need & ok
        out[take] = base[take]
        need &= ~ok
        if not need.any():
            return out
    out[need] = (theta["m0s"] + w + 0 * out)[need]                   # unreachable balance: the population pair
    return out


def simulate(theta, rng, n_rep, n_pair, n_fly, n_probe, a_pair, b_pair, fly_a, fly_b, g, z, naive_max=0.5):
    """{stage: int [n_rep, n_pair, n_fly, n_probe, 2, 2]}; a_pair / b_pair [n_pair], fly_a / fly_b [n_fly]
    multipliers."""
    sdp = sd_pre(theta, z)
    base = _pair_bases(theta, rng, n_rep, n_pair, g, z, naive_max, sdp)
    Lf = np.linalg.cholesky(theta["fly_cov"] + 1e-9 * np.eye(4))
    v = (rng.standard_normal((n_rep, n_pair, n_fly, 4)) @ Lf.T).reshape(n_rep, n_pair, n_fly, 2, 2)
    a = np.asarray(a_pair, float)[None, :, None] * np.asarray(fly_a, float)[None, None, :]
    b = np.asarray(b_pair, float)[None, :, None] * np.asarray(fly_b, float)[None, None, :]
    mu = slot_means(theta, base[:, :, None], v, np.broadcast_to(a, v.shape[:3]), np.broadcast_to(b, v.shape[:3]))
    idx = rng.integers(0, len(theta["resid"]), (n_rep, n_pair, n_fly, n_probe))
    counts = np.clip(np.rint(mu[:, :, :, None] + theta["resid"][idx]), 0, None).astype(np.int32)
    return to_stages(counts)


# ================================================================ calibration (W.9.9 P0-1)
def true_dprimes(theta, a, b, idx, z) -> np.ndarray:
    """The four gates' directional true d′ (sign · d′) of the population pair over the residual draws idx."""
    mu = slot_means(theta, theta["m0s"], np.zeros((2, 2)), a, b)
    c = np.clip(np.rint(mu[None] + theta["resid"][idx]), 0, None)
    d = to_stages(c[None])                                       # [1 fly, n probes, 2, 2]
    return WV.SIGNS * WV.gate_stats(d, z)[0]


def _bisect(f, hi, tol, n_iter):
    lo = 0.0
    f_lo, f_hi = f(lo), f(hi)
    if abs(f_lo) <= tol:
        return dict(value=lo, status="ok", steps=0)
    if f_lo > 0:
        return dict(value=None, status="above_at_zero", f0=float(f_lo))
    if f_hi < 0:
        return dict(value=None, status="floor", f_hi=float(f_hi))
    for i in range(n_iter):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) <= tol:
            return dict(value=mid, status="ok", steps=i + 1)
        lo, hi = (mid, hi) if fm < 0 else (lo, mid)
    return dict(value=None, status="no_convergence")


def calibrate(theta, target: float, mode: str, idx, z, spec) -> dict:
    """a then b for "min" (power, record) or "max" (false pass) of the gate pairs at `target`."""
    agg = np.min if mode == "min" else np.max
    hi_a = float((theta["m0s"][WV.P, WV.X] + theta["drift"][0][WV.P, WV.X] + np.abs(theta["resid"]).max()) * 2 + 1)
    ra = _bisect(lambda x: agg(true_dprimes(theta, x, 0.0, idx, z)[[0, 2]]) - target, hi_a, spec.cal_tol,
                 spec.cal_iter)
    if ra["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        ra = dict(value=0.0, status="zero_floor", f0=ra["f0"])
    if ra["value"] is None:
        return dict(ok=False, a=ra, b=None)
    hi_b = float((theta["m0s"][WV.A, WV.X] + np.abs(theta["drift"]).sum() + np.abs(theta["resid"]).max()) * 2 + 1)
    rb = _bisect(lambda x: agg(true_dprimes(theta, ra["value"], x, idx, z)[[1, 3]]) - target, hi_b, spec.cal_tol,
                 spec.cal_iter)
    if rb["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        rb = dict(value=0.0, status="zero_floor", f0=rb["f0"])
    if rb["value"] is None:
        return dict(ok=False, a=ra, b=rb)
    td = true_dprimes(theta, ra["value"], rb["value"], idx, z)
    return dict(ok=True, a=ra, b=rb, true_dprime=dict(zip(WV.GATES, td.tolist())))


# ================================================================ evaluation: the W verdict on simulated experiments
def evaluate(theta, rng, n_rep, a_pair, b_pair, fly_a, fly_b, g, z, spec) -> np.ndarray:
    """P(PASS) [len(q_grid), len(k_grid), F_max − F_min + 1, k_cap − k_min + 1]."""
    qs, ks = spec.q_grid, spec.k_grid
    fs = list(range(spec.f_min, spec.f_max + 1))
    kk = list(range(spec.k_min, spec.k_cap + 1))
    n_probe = 2 * max(ks)
    hits = np.zeros((len(qs), len(ks), len(fs), len(kk)))
    kw = dict(bar=spec.bar, band=spec.band_width, digits=spec.round_digits)
    done = 0
    while done < n_rep:
        n = min(spec.oc_chunk, n_rep - done)
        d = simulate(theta, rng, n, spec.k_cap, spec.f_max, n_probe, a_pair, b_pair, fly_a, fly_b, g, z,
                     spec.naive_max)
        mach = WV.rn1_mismatch(d)                                       # [n, pairs]
        for ki, K in enumerate(ks):
            dK = {s: v[..., :K, :, :] for s, v in d.items()}
            d2 = {s: v[..., :2 * K, :, :] for s, v in d.items()}
            cK, c2 = WV.fly_class(WV.gate_stats(dK, z), **kw), WV.fly_class(WV.gate_stats(d2, z), **kw)
            fK, f2 = WV.mech_fractions(dK), WV.mech_fractions(d2)
            for fi, F in enumerate(fs):
                mK = WV.mech_ok(fK[..., :F, :], spec.mech_min, spec.round_digits)
                m2 = WV.mech_ok(f2[..., :F, :], spec.mech_min, spec.round_digits)
                for qi, q in enumerate(qs):
                    fin = WV.pair_final(WV.pair_gate_code(cK[..., :F], q, F, spec.round_digits), mK,
                                        WV.pair_gate_code(c2[..., :F], q, F, spec.round_digits), m2)
                    for j, k in enumerate(kk):
                        ov = WV.overall_code(fin[:, :k], mach[:, :k].any(-1), spec.min_gate_pairs)
                        hits[qi, ki, fi, j] += (ov == WV.V_PASS).sum()
        done += n
    return hits / n_rep


def _uniform(spec, a, b):
    return [a] * spec.k_cap, [b] * spec.k_cap, [1.0] * spec.f_max, [1.0] * spec.f_max


def scenario_p(theta, cal, rng, n_rep, g, z, spec):
    if not cal["ok"]:
        return None
    return evaluate(theta, rng, n_rep, *_uniform(spec, cal["a"]["value"], cal["b"]["value"]), g, z, spec)


# ================================================================ the OC and the selection
def qualify(power_lo: np.ndarray, false_hi: np.ndarray, spec) -> np.ndarray:
    """[q, K, F] the envelope rule over F (module docstring)."""
    ok = (power_lo >= spec.p_power) & (false_hi <= spec.p_false)
    out = np.zeros_like(ok)
    nf = ok.shape[-1]
    for fi in range(nf):
        F = spec.f_min + fi
        hi = fi + 1 if F >= spec.envelope_solo_from else min(nf, fi + spec.envelope + 1)
        out[..., fi] = ok[..., fi:hi].all(-1)
    return out


def select(qual: np.ndarray, cost, spec) -> tuple:
    """(selected (q, K, F) or None, the qualifying designs in cost order with their costs)."""
    rows = []
    for qi, q in enumerate(spec.q_grid):
        for ki, K in enumerate(spec.k_grid):
            for fi in range(qual.shape[-1]):
                if qual[qi, ki, fi]:
                    F = spec.f_min + fi
                    rows.append(dict(q=q, K=K, F=F, cost_h=float(cost(K, F))))
    rows.sort(key=lambda r: (r["cost_h"], -r["q"], r["K"], r["F"]))
    return (rows[0] if rows else None), rows


def run(pilot: list, z: dict, spec, cost, n_boot=None, n_rep=None, n_boot_rep=None, log=None) -> dict:
    """The whole OC (module docstring) from the pilot pairs; cost(K, F) -> hours. Returns the table document."""
    t0 = time.perf_counter()
    n_boot = spec.boot_draws if n_boot is None else n_boot
    n_rep = spec.oc_reps if n_rep is None else n_rep
    n_boot_rep = spec.boot_reps if n_boot_rep is None else n_boot_rep
    theta = fit(pilot)
    idx = _rng(spec, TAG_CAL).integers(0, len(theta["resid"]), spec.cal_reps)
    shape = (len(spec.q_grid), len(spec.k_grid), spec.f_max - spec.f_min + 1, spec.k_cap - spec.k_min + 1)
    cal = {m: calibrate(theta, t, m, idx, z, spec) for m, t in (("min", spec.d_power), ("max", spec.d_false))}
    t_point = time.perf_counter()
    point = {}
    gs = list(spec.cluster_grid)
    for gi, g in enumerate(gs):
        for m in ("min", "max"):
            p = scenario_p(theta, cal[m], _rng(spec, TAG_POINT, gi, int(m == "max")), n_rep, g, z, spec)
            point[(g, m)] = p
    t_boot = time.perf_counter()
    boot_p = np.zeros((n_boot, len(gs), 2) + shape)
    boot_cal = []
    for bi in range(n_boot):
        rb = _rng(spec, TAG_BOOT, bi)
        tb = boot(theta, rb)
        ib = rb.integers(0, len(tb["resid"]), spec.cal_reps)
        cb = {m: calibrate(tb, t, m, ib, z, spec) for m, t in (("min", spec.d_power), ("max", spec.d_false))}
        boot_cal.append({m: dict(ok=c["ok"], a=(c["a"] or {}).get("value"), b=(c["b"] or {}).get("value"))
                         for m, c in cb.items()})
        for gi, g in enumerate(gs):
            for mi, m in enumerate(("min", "max")):
                p = scenario_p(tb, cb[m], _rng(spec, TAG_BOOT_SIM, bi, gi, mi), n_boot_rep, g, z, spec)
                boot_p[bi, gi, mi] = (0.0 if m == "min" else 1.0) if p is None else p
        if log is not None and (bi + 1) % 10 == 0:
            log(f"w oc bootstrap {bi + 1}/{n_boot} ({time.perf_counter() - t0:.0f} s)")
    t_sel = time.perf_counter()
    lo_pct, hi_pct = 100 * (1 - spec.boot_level), 100 * spec.boot_level
    if n_boot:
        power_lo = np.percentile(boot_p[:, :, 0].min(-1), lo_pct, axis=0).min(0)      # [q, K, F]
        false_hi = np.percentile(boot_p[:, :, 1].max(-1), hi_pct, axis=0).max(0)
    else:
        power_lo = np.zeros(shape[:3])
        false_hi = np.ones(shape[:3])
    reachable = cal["min"]["ok"] and cal["max"]["ok"]
    qual = qualify(power_lo, false_hi, spec) if reachable else np.zeros(shape[:3], bool)
    sel, ranking = select(qual, cost, spec)
    doc = dict(theta=summary(theta), calibration=cal, reachable=bool(reachable),
               point={f"g{g}|{m}": (None if p is None else p.tolist()) for (g, m), p in point.items()},
               power_lo=power_lo.tolist(), false_hi=false_hi.tolist(), qualified=qual.tolist(),
               selected=sel, ranking=ranking, boot_calibration=boot_cal,
               axes=dict(q=list(spec.q_grid), K=list(spec.k_grid), F=list(range(spec.f_min, spec.f_max + 1)),
                         k=list(range(spec.k_min, spec.k_cap + 1)), g=gs),
               n_boot=int(n_boot), n_rep=int(n_rep), n_boot_rep=int(n_boot_rep), seed=int(spec.oc_seed))
    t_rec = time.perf_counter()
    if sel is not None:
        doc["records"] = records(theta, sel, idx, z, spec, n_rep)
    t_end = time.perf_counter()
    doc["timing"] = dict(point_s=t_boot - t_point, boot_s=t_sel - t_boot, records_s=t_end - t_rec,
                         total_s=t_end - t0)
    return doc


def records(theta, sel, idx, z, spec, n_rep) -> dict:
    """W.6 / W.9.3 / W.9.9 P2-12 records at θ̂ for the selected design (q, K, F): P(PASS) by k at true d′ 0.5 · 1.0 ·
    1.5 · 2.0 (min-calibrated), the heterogeneous scenarios (one pair 0.5 + rest 1.5, every other pair 0.5, all 1.0)
    and the mixed-fly scenario (every other fly at 0) — each on every cluster level."""
    qi, ki = spec.q_grid.index(sel["q"]), spec.k_grid.index(sel["K"])
    fi = sel["F"] - spec.f_min
    cals = {t: calibrate(theta, t, "min", idx, z, spec) for t in sorted(set(spec.record_dprimes) | {1.0, 1.5})}
    out = dict(true_dprime={}, heterogeneous={}, mixed_flies={})

    def at(a_p, b_p, fa, fb, g, tag):
        r = _rng(spec, TAG_RECORD, spec.cluster_grid.index(g), RECORD_TAGS.index(tag))
        return evaluate(theta, r, n_rep, a_p, b_p, fa, fb, g, z, spec)[qi, ki, fi].tolist()

    def ab(t):
        c = cals[t]
        return (c["a"]["value"], c["b"]["value"]) if c["ok"] else (None, None)

    for g in spec.cluster_grid:
        out["true_dprime"][f"g{g}"] = {str(t): (at(*_uniform(spec, *ab(t)), g, f"t{t}") if cals[t]["ok"] else None)
                                       for t in spec.record_dprimes}
        a5, b5 = ab(0.5)
        a15, b15 = ab(1.5)
        a1, b1 = ab(1.0)
        het = {}
        if None not in (a5, a15):
            one = [a5] + [a15] * (spec.k_cap - 1), [b5] + [b15] * (spec.k_cap - 1)
            alt = [a5 if i % 2 else a15 for i in range(spec.k_cap)], [b5 if i % 2 else b15 for i in range(spec.k_cap)]
            het["one_pair_0.5"] = at(one[0], one[1], [1.0] * spec.f_max, [1.0] * spec.f_max, g, "one")
            het["half_pairs_0.5"] = at(alt[0], alt[1], [1.0] * spec.f_max, [1.0] * spec.f_max, g, "half")
        if a1 is not None:
            het["all_1.0"] = at(*_uniform(spec, a1, b1), g, "all1")
        out["heterogeneous"][f"g{g}"] = het
        if a15 is not None:
            half = [1.0 if i % 2 == 0 else 0.0 for i in range(spec.f_max)]
            out["mixed_flies"][f"g{g}"] = at([a15] * spec.k_cap, [b15] * spec.k_cap, half, half, g, "mixed")
    out["calibrations"] = {str(t): dict(ok=c["ok"], true_dprime=c.get("true_dprime")) for t, c in cals.items()}
    return out


# ================================================================ synthetic validation (W.9.3, W.9.9 P2-11)
def synthetic_pilot(rng, n_pair=16, n_fly=8, n_probe=8, base=(30.0, 60.0), sd=6.0, drift_ax=0.0, corr=0.0,
                    learn=(0.0, 0.0)) -> list:
    """Gaussian pilot pairs with known structure: naive means base (A, P) for X and Y, probe noise sd with an
    across-slot correlation corr, an X-only drift of MBON13 during the second phase (drift_ax per presentation
    block), and learning effects learn = (a, b) on R."""
    out = []
    cov = np.full((6, 6), corr) + (1 - corr) * np.eye(6)
    L = np.linalg.cholesky(cov)
    for _ in range(n_pair):
        mu = np.zeros((6, 2, 2))
        mu[:, WV.A, :], mu[:, WV.P, :] = base
        mu[SI["R1"], WV.P, WV.X] -= learn[0]
        mu[SI["R2"], WV.P, WV.X] -= learn[0]
        mu[SI["R2"], WV.A, WV.X] -= learn[1] + drift_ax
        mu[SI["RN2"], WV.P, WV.X] -= learn[0]
        mu[SI["RN2"], WV.A, WV.X] -= drift_ax
        mu[SI["N2"], WV.A, WV.X] -= drift_ax
        e = rng.standard_normal((n_fly, n_probe, 2, 2, 6)) @ L.T * sd
        c = np.clip(np.rint(mu[None, None] + np.moveaxis(e, -1, 2)), 0, None).astype(np.int64)
        d = to_stages(c)
        out.append({s: d[s] for s in WV.STAGES})
    return out


def simple_normal(spec, rng, d_true=1.5, n_rep=2000, q=0.75, k_probe=8) -> dict:
    """W.9.9 P0-1's simple normal model: per fly, four independent gates each estimated from K normal probes with true
    d′ d_true; P(PASS) of one pair at q by F (it must not rise from F 8 to 32)."""
    out = {}
    for F in (8, 16, 24, 32):
        x = rng.standard_normal((n_rep, F, 4, k_probe)) + d_true
        st = WV.dprime(x) * WV.SIGNS
        cls = WV.fly_class(st, spec.bar, spec.band_width, spec.round_digits)
        out[F] = float((WV.pair_gate_code(cls, q, F, spec.round_digits) == WV.P_PASS).mean())
    return out


def synthetic_validation(spec, z: dict, n_rep: int = 1000) -> dict:
    """The four known-answer fixtures and P0-1's check (W.9.9 P2-11 tolerances), evaluated by the OC machinery."""
    rng = _rng(spec, TAG_SYNTH)
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    th0 = fit(synthetic_pilot(rng, **kw))
    th_drift = fit(synthetic_pilot(rng, drift_ax=15.0, **kw))
    idx = rng.integers(0, len(th0["resid"]), spec.cal_reps)
    res = {}
    z0 = evaluate(th0, rng, n_rep, *_uniform(spec, 0.0, 0.0), 0.0, z, spec)
    res["zero_effect"] = dict(max_p=float(z0.max()), limit=0.02, ok=bool(z0.max() <= 0.02))
    big = calibrate(th0, 4.0, "min", idx, z, spec)
    pb = (evaluate(th0, rng, n_rep, *_uniform(spec, big["a"]["value"], big["b"]["value"]), 0.0, z, spec)
          if big["ok"] else np.zeros(1))
    res["big_effect"] = dict(min_p=float(pb.min()), limit=0.98, ok=bool(big["ok"] and pb.min() >= 0.98),
                             true_dprime=big.get("true_dprime"))
    one = evaluate(th_drift, rng, n_rep, *_uniform(spec, 0.0, 0.0), 0.0, z, spec)
    td = true_dprimes(th_drift, 0.0, 0.0, rng.integers(0, len(th_drift["resid"]), spec.cal_reps), z)
    res["one_gate"] = dict(max_p=float(one.max()), limit=0.02, ok=bool(one.max() <= 0.02 and td[1] >= 1.5),
                           true_dprime=dict(zip(WV.GATES, td.tolist())))
    if big["ok"]:
        alt = [1.0 if i % 2 == 0 else 0.0 for i in range(spec.f_max)]
        neg = evaluate(th0, rng, n_rep, [big["a"]["value"]] * spec.k_cap, [big["b"]["value"]] * spec.k_cap, alt,
                       [1.0 - x for x in alt], 0.0, z, spec)
        res["negative_correlation"] = dict(max_p=float(neg.max()), limit=0.02, ok=bool(neg.max() <= 0.02))
    else:
        res["negative_correlation"] = dict(max_p=None, ok=False)
    sn = simple_normal(spec, rng)
    res["simple_normal"] = dict(p_by_F={str(k): v for k, v in sn.items()}, ok=bool(sn[32] <= sn[8]))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_w_oc.py tests/brain/test_w_verdict.py -q`
Expected: 24 passed (~20 s).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/w_oc.py tests/brain/test_w_oc.py
git commit -m "feat(w): w_oc — pilot residual-vector model (naive-balanced pair REs, fly REs, cluster grid, drift, spillover, floor), gate-vector bisection calibration, nested experiments through w_verdict, pair-level parametric bootstrap limits, envelope selection by cost, synthetic validation"
```

---

### Task 4: `w_records` and `w_rules` — job rows to verdict arrays, machine checks, costs; gate decisions and sentences

**Files:**
- Create: `flymon/brain/w_records.py`, `flymon/brain/w_rules.py`
- Test: `tests/brain/test_w_records.py`, `tests/brain/test_w_rules.py`

**Interfaces:**
- Consumes: `w_verdict` (Task 2), `w_spec.SPEC`.
- Produces: `w_records.counts(stage) -> [K, 2, 2]`; `pair_data(rows, flies) -> {stage: [F, K, 2, 2]}` (rows = `[dict(unit=dict(pair, fly, brain), result=w_learn_job result)]`); `machine_reasons(rows, flies, declared, naive=None, band=None) -> list`; `extend(d_k, band_rows, flies) -> dict`; `p_repro_diffs(w_res, v_row, tag)`, `naive_repro_diffs(w_res, v_result, tag)`, `reward_check(w_res)` (lists); `pilot_record(pairs, z, walls, spec) -> dict`; `unit_costs(job_results, trials, oracle_round_s, workers) -> dict`; `job_s(c, k)`; `design_cost(c, K, F, spec, n_set, n_naive=None, with_c=True) -> {parts_h, total_h}`; `elapsed_h(ledger) -> float`. `w_rules`: `PASS`, `INVALID`, `NOT_READ`, `SEALED`, `READ`, `INVALID_RUN`, the STOP labels, `SENTENCES`, `CONSEQUENCE`, `sentence(outcome, fields)`, `reuse(v_dec, v_doc, v_git, ancestors, last_v_commit, u_key, code_key, t_key, spec)`, `path(checks)`, `pilot(rec, machine, spec)`, `oc(doc, spec)`, `oc_unreachable_text(doc, spec)`, `budget(elapsed_h, options, spec)`, `gate_pairs(n_set, k, spec)`, `verdict_sentence(v, design, spec) -> {sentence, consequence}`.

- [ ] **Step 1: Write the failing tests**

```python
"""W's records and plumbing (W.3 2, W.6, W.9.4, W.9.6 P2-10, W.9.8 H6, W.9.9 P1-4 / P2-9): job rows -> [K, 2, 2] and
{stage: [F, K, 2, 2]}; the machine reasons (pre across brains and the screen, RN's reward-phase weights = R's — an RN
that skipped its own reward phase is caught, the declared edit / CSC / edges / probe seeds, the band jobs' weights);
the 2K extension; the path comparisons; the reward check; the pilot record; the cost model."""
import copy

import numpy as np

from flymon.brain import w_records as WR
from flymon.brain import w_verdict as WV
from flymon.brain.w_spec import SPEC


def pres(a, p, seed=0):
    return dict(seed=seed, A=a, P=p, kc_frac=0.05, kc_spikes=1, kc_max_win_hz=1.0, apl_out_per_step=0.1, wall_s=0.3,
                steps=1400)


def stage(name, k, ax=30, px=60, sha="w0"):
    return dict(stage=name, x=[pres(ax + i, px - i, i) for i in range(k)], y=[pres(30, 60, i) for i in range(k)],
                w_sha256=sha, weights_frac=1.0, weights_frac_A=1.0, weights_frac_P=0.9, da_integral=None)


def job(brain, fly, k=4, edit="E", seeds=None):
    sh = {"R": ("r1", "r2"), "N": ("n1", "n2"), "RN": ("r1", "rn2")}[brain]
    return dict(unit=dict(pair="p", fly=fly, brain=brain), result=dict(
        edit=edit, csc_sha256="S", edit_edges=2, block_edges={"a": 1}, w0_sha256="w0", fly=fly, plastic=True,
        probe_seeds=seeds or [fly * 100 + i for i in range(k)],
        stages=[stage("pre", k), stage("S1", k, px=40, sha=sh[0]), stage("S2", k, ax=20, px=40, sha=sh[1])],
        wall_s=1.0, train_s=0.8, probe_s=0.2))


DECL = dict(edit="E", csc_sha256="S", edit_edges=2, block_edges={"a": 1},
            probe_seeds={f: [f * 100 + i for i in range(4)] for f in range(2)})


def rows2():
    return [job(b, f) for f in range(2) for b in ("R", "N", "RN")]


def test_counts_and_pair_data():
    c = WR.counts(stage("pre", 3))
    assert c.shape == (3, 2, 2) and c[1].tolist() == [[31, 30], [59, 60]]
    d = WR.pair_data(rows2(), [0, 1])
    assert set(d) == set(WV.STAGES) and d["R1"].shape == (2, 4, 2, 2)
    assert np.array_equal(d["R1"], d["RN1"]) and d["R2"][0, 0, WV.A, WV.X] == 20


def test_machine_reasons():
    assert WR.machine_reasons(rows2(), [0, 1], DECL) == []
    r = rows2()
    r[1]["result"]["stages"][0]["x"][0]["A"] += 1                                  # N's pre differs
    assert WR.machine_reasons(r, [0, 1], DECL) == ["fly 0: N pre ≠ R pre"]
    r = rows2()
    r[2]["result"]["stages"][1]["w_sha256"] = "w0"                                 # RN skipped its reward phase
    assert WR.machine_reasons(r, [0, 1], DECL) == ["fly 0: RN 보상 뒤 가중치 ≠ R"]
    r = rows2()
    r[0]["result"]["edit_edges"] = 0
    assert WR.machine_reasons(r, [0, 1], DECL) == ["fly 0 R: edit_edges 0 ≠ 2"]
    assert WR.machine_reasons(rows2()[:5], [0, 1], DECL) == ["fly 1: ['RN'] 없음"]
    naive = {0: WR.counts(stage("pre", 4)), 1: WR.counts(stage("pre", 4)) + 1}
    assert WR.machine_reasons(rows2(), [0, 1], DECL, naive=naive) == ["fly 1: 학습 pre ≠ 순진 거름 프로브"]
    band = [copy.deepcopy(x) for x in rows2()]
    assert WR.machine_reasons(rows2(), [0, 1], DECL, band=band) == []
    band[3]["result"]["stages"][2]["w_sha256"] = "zz"
    assert WR.machine_reasons(rows2(), [0, 1], DECL, band=band) == ["band fly 1 R: 재훈련 가중치 ≠ 주 측정"]
    seeds = dict(DECL, probe_seeds={0: [1, 2, 3, 4], 1: DECL["probe_seeds"][1]})
    assert WR.machine_reasons(rows2(), [0, 1], seeds)[0] == "fly 0 R: 프로브 시드가 선언과 다름"


def test_extend():
    d = WR.pair_data(rows2(), [0, 1])
    e = WR.extend(d, rows2(), [0, 1])
    assert e["R1"].shape == (2, 8, 2, 2) and np.array_equal(e["R1"][:, :4], d["R1"])


def test_p_and_naive_repro_and_reward_check():
    w = job("R", 0, k=1)["result"]
    w["stages"] = w["stages"][:2]
    st = {s["stage"]: s for s in w["stages"]}
    v = dict(pre=dict(x=st["pre"]["x"][0], y=st["pre"]["y"][0]), post=dict(x=st["S1"]["x"][0], y=st["S1"]["y"][0]),
             w0_sha256="w0", w_post_sha256="r1", weights_frac=1.0, weights_frac_A=1.0, weights_frac_P=0.9,
             da_integral=None, csc_sha256="S")
    assert WR.p_repro_diffs(w, v, "t") == []
    v2 = copy.deepcopy(v)
    v2["post"]["x"]["wall_s"] = 9.0                                                # wall time is not compared
    assert WR.p_repro_diffs(w, v2, "t") == []
    v2["post"]["x"]["A"] += 1
    v2["w_post_sha256"] = "q"
    assert WR.p_repro_diffs(w, v2, "t") == ["t post x", "t w_post_sha256"]
    res = {"report": {"pre": {"A": [[30, 30]], "P": [[60, 60]]}}}
    assert WR.naive_repro_diffs(w, res, "n") == []
    res["report"]["pre"]["P"][0][1] = 61
    assert WR.naive_repro_diffs(w, res, "n") == ["n 순진 카운트"]
    assert WR.reward_check(w) == []
    w["stages"][1]["w_sha256"] = "w0"
    w["stages"][1]["weights_frac_P"] = 1.0
    w["stages"][1]["x"][0]["P"] = 60
    assert len(WR.reward_check(w)) == 3


def test_pilot_record():
    d = WR.pair_data(rows2(), [0, 1])
    rec = WR.pilot_record({"p": d}, {"A": (0.0, 1.0), "P": (0.0, 1.0)}, dict(job_s_median=1.0), SPEC)
    p = rec["pairs"]["p"]
    assert set(p["median"]) == set(WV.GATES) and rec["label"] == "탐색"
    assert p["spill"]["reward"] == dict(x=0.0, y=0.0) and p["suppression"]["N2_minus_pre"] == -10.0
    assert rec["floor_both_share"] == 0.0 and rec["walls"] == dict(job_s_median=1.0)


def test_costs():
    c = WR.unit_costs([job("R", 0)["result"]], 40, 120.0, 16)
    assert c == dict(trial_s=0.02, presentation_s=0.2 / 24, oracle_round_s=120.0, workers=16)
    dc = WR.design_cost(dict(trial_s=1.0, presentation_s=1.0, oracle_round_s=100.0, workers=16), 8, 8, SPEC, 249)
    p = dc["parts_h"]
    assert p["oracle"] == 16 * 100 / 3600 and p["naive"] == -(-249 * 8 // 16) * 16 / 3600
    assert p["learn"] == p["band"] == p["c"] == 12 * (40 + 48) / 3600 and p["noplast"] == 88 / 3600
    no_c = WR.design_cost(dict(trial_s=1.0, presentation_s=1.0, oracle_round_s=100.0, workers=16), 8, 8, SPEC, 0, 30,
                          False)
    assert no_c["parts_h"]["c"] == 0.0 and no_c["parts_h"]["oracle"] == 0.0
    assert WR.elapsed_h([dict(wall_s=1800.0), dict(wall_s=1800.0)]) == 1.0
```

```python
"""W's gate decisions and sentences (W.3, W.7, W.9.3-W.9.9): reuse (V's blocks, keys, commits, band, z_V), the path
gate's INVALID / STOP order, H6's three conditions, the OC's outcome (OPEN 3's default text), the budget order (C
first, then the alternatives), STOP_FEW_PAIRS, and every closing sentence verbatim."""
import dataclasses

from flymon.brain import w_rules as R
from flymon.brain.w_spec import SPEC

KEYS = dict(u_measure_key=SPEC.u_measure_key_u, code_key=SPEC.r_shared_key, t_measure_key=SPEC.t_measure_key_t)
ZV = {"A": [16.916666666666668, 12.483878492769072], "P": [80.16666666666667, 29.775432639827233]}


def v_doc():
    return dict(z=dict(KEYS, z_V=ZV), kc_input=dict(KEYS), set=dict(KEYS), judge=dict(KEYS, band="SELECTED"))


def reuse(**kw):
    a = dict(v_dec=dict(outcome="PASS", reasons=[]), v_doc=v_doc(), v_git=dict(tracked=True, dirty=False),
             ancestors={c: True for _, c in SPEC.v_commits}, last_v_commit="a279a56" + "f" * 33,
             u_key=SPEC.u_measure_key_u, code_key=SPEC.r_shared_key, t_key=SPEC.t_measure_key_t, spec=SPEC)
    a.update(kw)
    return R.reuse(**a)


def test_reuse():
    assert reuse()["outcome"] == "PASS"
    d = v_doc()
    d["z"]["z_V"] = {"A": [16.9, 12.484], "P": ZV["P"]}
    for kw, word in ((dict(v_dec=dict(outcome="STOP_REUSE", reasons=["R 재현"])), "R 재현"),
                     (dict(u_key="x"), "U 측정 키 x"), (dict(v_git=dict(tracked=True, dirty=True)), "미커밋"),
                     (dict(last_v_commit="b" * 40), "마지막 커밋"),
                     (dict(ancestors={c: c != "cf0b3b2" for _, c in SPEC.v_commits}), "cf0b3b2(set)"),
                     (dict(v_doc=d), "V z_V")):
        out = reuse(**kw)
        assert out["outcome"] == R.STOP_REUSE and word in out["sentence"], (kw, out)
    d = v_doc()
    d["judge"]["band"] = "B_Tb"
    assert "V 판정 B_Tb" in reuse(v_doc=d)["sentence"]
    assert reuse(v_doc=d)["sentence"].startswith("W 재사용 조건(W.3 1)이 깨졌다(")


def test_path_order():
    ok = dict(ref="a", diffs=[], invalid=[])
    assert R.path([ok, ok])["outcome"] == "PASS"
    out = R.path([ok, dict(ref="V 관문 ② 처벌 팔 행(L_V r1 1)", diffs=["x pre"], invalid=[]),
                  dict(ref="b", diffs=["y"], invalid=[])])
    assert out["outcome"] == R.STOP_W_PATH_REPRO and len(out["failed"]) == 2
    assert out["sentence"] == "W 학습 경로가 V 관문 ② 처벌 팔 행(L_V r1 1)을 재현하지 못했다(x pre)."
    assert R.path([dict(ref="a", diffs=["d"], invalid=["edges 3"])])["outcome"] == R.INVALID


def rec(pairs, floor=0.0):
    return dict(pairs={f"p{i}": dict(median=dict(reward_assoc=r, punish_assoc=p)) for i, (r, p) in enumerate(pairs)},
                floor_both_share=floor)


def test_pilot_h6():
    assert R.pilot(rec([(1.0, -1.0)] * 4), [], SPEC)["outcome"] == "PASS"
    assert R.pilot(rec([(1.0, -1.0)] * 4), ["fly 0: N pre ≠ R pre"], SPEC)["outcome"] == R.INVALID
    out = R.pilot(rec([(1.0, -0.4)] * 3 + [(1.0, -1.0)]), [], SPEC)                  # punish share 0.25
    assert out["outcome"] == R.STOP_PILOT_NO_EFFECT and out["reasons"][0].startswith("(i)")
    assert R.pilot(rec([(1.0, -1.0)] * 2 + [(0.5, -0.5)] * 2), [], SPEC)["outcome"] == "PASS"   # 0.5 / −0.5 count
    out = R.pilot(rec([(1.0, -1.0)] * 2 + [(-0.1, -1.0)] * 3), [], SPEC)            # reversed 0.6 > 0.5
    assert any(r.startswith("(ii)") for r in out["reasons"])
    out = R.pilot(rec([(1.0, -1.0)] * 4, floor=0.51), [], SPEC)
    assert out["reasons"] == ["(iii) X·Y 함께 바닥인 프로브 비율 0.510(> 0.5)"]
    assert out["sentence"] == ("파일럿에서 조합 지렛대의 F.2 학습 효과가 (iii) X·Y 함께 바닥인 프로브 비율 0.510(> 0.5) — 주 "
                               "세트를 쓰지 않고 멈춘다.")
    assert R.pilot(rec([(1.0, -1.0)] * 4, floor=0.5), [], SPEC)["outcome"] == "PASS"


def test_oc_outcome_and_open3_text():
    assert R.oc(dict(selected=dict(q=0.75, K=8, F=10)), SPEC)["outcome"] == "PASS"
    nf = SPEC.f_max - SPEC.f_min + 1
    doc = dict(selected=None, reachable=True, power_lo=[[[0.5] * nf] * 2] * 3, false_hi=[[[0.01] * nf] * 2] * 3)
    out = R.oc(doc, SPEC)
    assert out["outcome"] == R.STOP_OC_UNREACHABLE
    assert out["sentence"].startswith("파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다(q 0.5·K 8·F 32: 검정력 하한 "
                                      "0.500, 거짓 통과 상한 0.010; ")
    un = R.oc(dict(selected=None, reachable=False, calibration=dict(min=dict(ok=False, a=dict(status="floor"), b=None),
                                                                  max=dict(ok=True, a=dict(status="ok"),
                                                                           b=dict(status="ok")))), SPEC)
    assert "보정 불가 — min: a floor, b None; max: a ok, b ok" in un["sentence"]


def test_budget_order():
    opts = [dict(design="s", with_c=True, total_h=10.0), dict(design="s", with_c=False, total_h=7.0),
            dict(design="alt", with_c=False, total_h=6.0)]
    assert R.budget(10.0, opts, SPEC)["plan"]["with_c"] is True
    assert R.budget(15.0, opts, SPEC)["plan"] == opts[1]
    assert R.budget(17.5, opts, SPEC)["plan"] == opts[2]
    out = R.budget(19.0, opts, SPEC)
    assert out["outcome"] == R.STOP_BUDGET and out["sentence"] == (
        "남은 추정 비용 누적 19.00 h + 남은 10.00 h = 29.00 h가 W 상한 24 h를 넘는다.")
    assert R.budget(14.0, opts, dataclasses.replace(SPEC, budget_h=24.0))["plan"]["with_c"] is True


def test_few_pairs():
    assert R.gate_pairs(249, 4, SPEC)["outcome"] == "PASS"
    out = R.gate_pairs(249, 3, SPEC)
    assert out["sentence"] == "W 주 세트 249쌍에서 오라클 순진 균형·시험 가능 쌍이 3개로 최소 4에 못 미쳤다."


def test_verdict_sentences():
    pairs = {"b|1|x|y": dict(status="FAIL", reasons=["마리별 동시 충족 미달"], failing_gates=dict(punish_assoc=5)),
             "a|2|x|y": dict(status="FAIL", reasons=["기계 대조 실패"], failing_gates={})}
    v = dict(verdict="FAIL", pairs=pairs, failing=list(pairs), mech_fail=["a|2|x|y"], machine=[], invalid=[],
             n_pairs=8, n_judgeable=8, undecided_cause="")
    s = R.verdict_sentence(v, dict(q=0.75, K=8, F=12), SPEC)
    assert s["sentence"] == ("F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부(b|1|x|y: 마리별 동시 "
                             "충족 미달(punish_assoc 5마리); a|2|x|y: 기계 대조 실패; 기계 대조 실패 1쌍) (마리별 동시 충족 집계).")
    assert s["consequence"].startswith("STD 재설계 여부는 사용자 몫")
    p = R.verdict_sentence(dict(v, verdict="PASS", failing=[], mech_fail=[]), dict(q=0.625, K=16, F=12), SPEC)
    assert p["sentence"] == (
        "조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, "
        "엔진별 z)에서 실제 학습 규칙으로 M2 학습 단위가 섰다 — 넓힌 풀(생성원 턴 306–1985), 오라클 순진·시험 가능 쌍 "
        "조건부(관문 쌍 8개, 판정 가능 8개, 설계 q 0.625 · K 16 · F 12 · k 상한 8, 마리별 동시 충족).")
    assert p["consequence"] == "POOL 배틀 과제(M3)는 별도 선언이 필요하다(T.9.5 좁힘 유지)."
    u = R.verdict_sentence(dict(v, verdict="UNDECIDED", undecided_cause="BAND 잔존"), {}, SPEC)
    assert u["sentence"] == "원인(BAND 잔존)을 기록하고 사용자 판단."
    m = R.verdict_sentence(dict(v, verdict="STOP_MACHINE", machine=["p: RN1 ≠ R1"], invalid=["q"]), {}, SPEC)
    assert m["sentence"].startswith("기계 검사 불일치(p: RN1 ≠ R1; INVALID 쌍 q) — F.7의 STOP_MACHINE.")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_records.py tests/brain/test_w_rules.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write `w_records`**

```python
"""W's records and plumbing between raw job rows and the verdict code (W.3, W.6, W.9.2, W.9.4, W.9.6, W.9.8 H6,
W.9.9 P1-4 / P2-9): no engine, no writes.
- counts / pair_data: a w_learn_job row's probes -> [K, 2 cells, 2 odours]; one pair's R / N / RN rows per fly ->
  {stage: int [F, K, 2, 2]} (pre and R1 / R2 from R, N1 / N2 from N, RN1 / RN2 from RN).
- machine_reasons: pre equal across a fly's brains and equal to the naive screen's probes, RN's weights after the
  reward phase equal R's (the state behind RN1 = R1), the declared edit / CSC / edge counts / probe seeds, the band
  jobs' weights equal the main jobs' at every stage.
- p_repro_diffs / naive_repro_diffs / reward_check: the path gate's comparisons (W.3 2 (i), (ii), W.9.9 P2-9).
- pilot_record: W.6 / W.9.4 / W.9.8 H6's pilot statistics (exploratory).
- unit_costs / design_cost / worst_cost: the wall-clock model behind the OC's ranking and the budget gate."""
from __future__ import annotations

import math
from statistics import median

import numpy as np

from . import w_verdict as WV

STAGE_OF = {("R", "pre"): "pre", ("R", "S1"): "R1", ("R", "S2"): "R2", ("N", "S1"): "N1", ("N", "S2"): "N2",
            ("RN", "S1"): "RN1", ("RN", "S2"): "RN2"}
ROW_SKIP = ("wall_s",)


def counts(stage: dict) -> np.ndarray:
    """[K, 2, 2] int: [k, A/P, X/Y] from a job stage's x / y presentation rows."""
    return np.array([[[x["A"], y["A"]], [x["P"], y["P"]]] for x, y in zip(stage["x"], stage["y"])], dtype=np.int64)


def _stages(res: dict) -> dict:
    return {s["stage"]: s for s in res["stages"]}


def pair_data(rows: list, flies: list) -> dict:
    """rows = [dict(unit, result)] of one pair (any order); flies = the scheduled fly numbers in order."""
    by = {(r["unit"]["fly"], r["unit"]["brain"]): r["result"] for r in rows}
    out = {}
    for (brain, st), name in STAGE_OF.items():
        out[name] = np.stack([counts(_stages(by[(f, brain)])[st]) for f in flies])
    return out


def machine_reasons(rows: list, flies: list, declared: dict, naive: dict | None = None,
                    band: list | None = None) -> list:
    """declared = {edit, csc_sha256, edit_edges, block_edges, probe_seeds: {fly: [...]}}; naive = {fly: [K, 2, 2]}
    of the screen (None: no screen); band = the pair's band rows (None: none)."""
    by = {(r["unit"]["fly"], r["unit"]["brain"]): r["result"] for r in rows}
    out = []
    for f in flies:
        got = {b: by.get((f, b)) for b in ("R", "N", "RN")}
        miss = [b for b, v in got.items() if v is None]
        if miss:
            out.append(f"fly {f}: {miss} 없음")
            continue
        st = {b: _stages(v) for b, v in got.items()}
        pre = counts(st["R"]["pre"])
        for b in ("N", "RN"):
            if not np.array_equal(counts(st[b]["pre"]), pre):
                out.append(f"fly {f}: {b} pre ≠ R pre")
        if st["RN"]["S1"]["w_sha256"] != st["R"]["S1"]["w_sha256"]:
            out.append(f"fly {f}: RN 보상 뒤 가중치 ≠ R")
        if naive is not None and not np.array_equal(np.asarray(naive[f]), pre[: len(naive[f])]):
            out.append(f"fly {f}: 학습 pre ≠ 순진 거름 프로브")
        for b, v in got.items():
            for k in ("edit", "csc_sha256", "edit_edges", "block_edges"):
                if v.get(k) != declared[k]:
                    out.append(f"fly {f} {b}: {k} {v.get(k)!r} ≠ {declared[k]!r}")
            if v.get("probe_seeds") != declared["probe_seeds"][f]:
                out.append(f"fly {f} {b}: 프로브 시드가 선언과 다름")
    for r in band or []:
        u, res = r["unit"], r["result"]
        main = by.get((u["fly"], u["brain"]))
        if main is None:
            out.append(f"band fly {u['fly']} {u['brain']}: 주 측정 없음")
            continue
        a, b = _stages(main), _stages(res)
        if [a[s]["w_sha256"] for s in ("pre", "S1", "S2")] != [b[s]["w_sha256"] for s in ("pre", "S1", "S2")]:
            out.append(f"band fly {u['fly']} {u['brain']}: 재훈련 가중치 ≠ 주 측정")
    return out


def extend(d_k: dict, band_rows: list, flies: list) -> dict:
    """The 2K data: the K probes of d_k followed by the band jobs' probes K..2K−1."""
    ext = pair_data(band_rows, flies)
    return {s: np.concatenate([np.asarray(d_k[s]), ext[s]], axis=1) for s in WV.STAGES}


# ================================================================ the path gate (W.3 2, W.9.9 P2-9)
def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in ROW_SKIP}


def p_repro_diffs(w_res: dict, v_row: dict, tag: str) -> list:
    """W's job with P's punish-arm settings vs V's gate-② arm row: pre / post presentations (wall aside), the weight
    shas, the weight fractions, the dopamine integral and the CSC."""
    st = _stages(w_res)
    out = []
    for name, w_stage, v_key in (("pre", st["pre"], "pre"), ("post", st["S1"], "post")):
        for s in ("x", "y"):
            if _strip(w_stage[s][0]) != _strip(v_row[v_key][s]):
                out.append(f"{tag} {name} {s}")
    pairs = (("w0_sha256", w_res["w0_sha256"]), ("w_post_sha256", st["S1"]["w_sha256"]),
             ("weights_frac", st["S1"]["weights_frac"]), ("weights_frac_A", st["S1"]["weights_frac_A"]),
             ("weights_frac_P", st["S1"]["weights_frac_P"]), ("da_integral", st["S1"]["da_integral"]),
             ("csc_sha256", w_res["csc_sha256"]))
    out += [f"{tag} {k}" for k, w in pairs if w != v_row[k]]
    return out


def naive_repro_diffs(w_res: dict, v_result: dict, tag: str) -> list:
    """W's naive probes on V's report seeds vs V's oracle raw report.pre (MBON13 / MBON05 × X, Y, every seed)."""
    c = counts(_stages(w_res)["pre"])
    v = np.stack([np.asarray(v_result["report"]["pre"]["A"]), np.asarray(v_result["report"]["pre"]["P"])], axis=1)
    return [] if np.array_equal(c, v) else [f"{tag} 순진 카운트"]


def reward_check(w_res: dict) -> list:
    """W.9.9 P2-9: after the reward phase the plastic weights moved, the reward core's weights fell, and MBON05(X)'s
    mean over the probes fell."""
    st = _stages(w_res)
    pre, s1 = counts(st["pre"]), counts(st["S1"])
    out = []
    if st["S1"]["w_sha256"] == w_res["w0_sha256"]:
        out.append("가르친 가중치 불변")
    if not st["S1"]["weights_frac_P"] < 1.0:
        out.append(f"보상 구획 가중치 비 {st['S1']['weights_frac_P']}")
    if not s1[:, WV.P, WV.X].mean() < pre[:, WV.P, WV.X].mean():
        out.append(f"MBON05(X) {pre[:, WV.P, WV.X].mean()} → {s1[:, WV.P, WV.X].mean()}")
    return out


# ================================================================ the pilot (W.6, W.9.2, W.9.4, W.9.8 H6)
def _corr(st) -> list:
    """The fly-level correlation of the four gates (±∞ clipped to ±10, a constant gate → 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.corrcoef(np.nan_to_num(np.asarray(st, float), posinf=10, neginf=-10).T)).tolist()


def _pcorr(d: dict, s1: str, s2: str, z: dict) -> float | None:
    """W.6: the probe-noise correlation of ΔV between two slots (same probe seeds), fly means removed, pooled."""
    a, b = WV.dv(d[s1], z), WV.dv(d[s2], z)
    a, b = (a - a.mean(-1, keepdims=True)).ravel(), (b - b.mean(-1, keepdims=True)).ravel()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _sd(x) -> float | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size >= 2 else None


def pilot_record(pairs: dict, z: dict, walls: dict, spec) -> dict:
    """pairs = {key: {stage: [F, K, 2, 2]}} (exploratory). Per pair: fly gate d′, their medians, q_sat at 0.75, the
    mechanism fractions, the naive d′; pooled: floors, Y spillover, presentation-evoked MBON13(X) suppression, the
    naive-floor share, the per-unit wall clock."""
    per, fl_any, fl_both, n_probe = {}, 0, 0, 0
    for key, d in pairs.items():
        st = WV.gate_stats(d, z)
        cls = WV.fly_class(st, spec.bar, spec.band_width, spec.round_digits)
        fr = WV.mech_fractions(d)
        a = {s: np.asarray(d[s], float) for s in WV.STAGES}
        r1, r2 = a["R1"], a["R2"]
        both = ((r1[..., WV.P, WV.X] == 0) & (r1[..., WV.P, WV.Y] == 0)) | \
               ((r2[..., WV.A, WV.X] == 0) & (r2[..., WV.A, WV.Y] == 0))
        taught = (r1[..., WV.P, WV.X] == 0) | (r2[..., WV.A, WV.X] == 0)
        fl_both += int(both.sum())
        fl_any += int(taught.sum())
        n_probe += int(both.size)
        rew = (r1 - a["N1"]).mean((0, 1))
        pun = ((r2 - r1) - (a["RN2"] - a["RN1"])).mean((0, 1))
        per[key] = dict(
            fly_stats={g: st[:, i].tolist() for i, g in enumerate(WV.GATES)},
            median={g: float(np.median(st[:, i])) for i, g in enumerate(WV.GATES)},
            q_sat=float((cls == WV.FLY_SAT).mean()), mech=dict(reward=float(np.median(fr[:, 0])),
                                                              punish=float(np.median(fr[:, 1]))),
            naive_d=WV.naive_dprime(d["pre"], z),
            naive_median=dict(MBON13_X=float(np.median(a["pre"][..., WV.A, WV.X])),
                              MBON05_X=float(np.median(a["pre"][..., WV.P, WV.X]))),
            spill=dict(reward=dict(x=float(rew[WV.P, WV.X]), y=float(rew[WV.P, WV.Y])),
                       punish=dict(x=float(pun[WV.A, WV.X]), y=float(pun[WV.A, WV.Y]))),
            suppression=dict(N2_minus_pre=float((a["N2"] - a["pre"])[..., WV.A, WV.X].mean()),
                             RN2_minus_RN1=float((a["RN2"] - a["RN1"])[..., WV.A, WV.X].mean())),
            between_fly_sd={g: _sd(st[:, i]) for i, g in enumerate(WV.GATES)},
            brain_noise_corr=dict(R1_N1=_pcorr(d, "R1", "N1", z), R2_RN2=_pcorr(d, "R2", "RN2", z),
                                  R1_R2=_pcorr(d, "R1", "R2", z)),
            gate_corr=_corr(st))
    naive_floor = [k for k, p in per.items() if min(p["naive_median"].values()) < spec.naive_floor_spikes]
    bins = {"<1": [], "1-5": [], ">=5": []}
    for k, p in per.items():
        ad = abs(p["naive_d"])
        if p["between_fly_sd"]["punish_assoc"] is not None:
            bins["<1" if ad < 1 else ("1-5" if ad < 5 else ">=5")].append(p["between_fly_sd"]["punish_assoc"])
    return dict(pairs=per, floor_both_share=fl_both / max(n_probe, 1), floor_taught_share=fl_any / max(n_probe, 1),
                naive_floor_pairs=naive_floor, naive_floor_share=len(naive_floor) / max(len(per), 1),
                noise_by_naive_bin={b: (float(median(v)) if v else None) for b, v in bins.items()},
                walls=walls, label="탐색")


# ================================================================ costs (W.9.6 F, W.9.9 P1-4 / P1-5)
def unit_costs(job_results: list, trials: int, oracle_round_s: float, workers: int) -> dict:
    """Seconds per trial (train_s / trials of a learning job) and per presentation (probe_s / (2 odours × probes ×
    stages)), medians over the jobs; the oracle's seconds per worker round; the worker count."""
    tr = [r["train_s"] / trials for r in job_results if r.get("train_s")]
    pr = [r["probe_s"] / (2 * len(r["probe_seeds"]) * len(r["stages"])) for r in job_results]
    return dict(trial_s=float(np.median(tr)) if tr else 0.0, presentation_s=float(np.median(pr)),
                oracle_round_s=float(oracle_round_s), workers=int(workers))


def _rounds(n: int, w: int) -> int:
    return -(-int(n) // max(1, int(w)))


def job_s(c: dict, k: int, trials: int = 40, stages: int = 3) -> float:
    return trials * c["trial_s"] + stages * 2 * k * c["presentation_s"]


def design_cost(c: dict, K: int, F: int, spec, n_set: int, n_naive: int | None = None, with_c: bool = True) -> dict:
    """Hours of the remaining W work for design (K, F): the oracle screen of n_set pairs, the naive screen of
    n_naive pairs (default n_set: the worst case), the learning measurement, the band re-measure upper bound (every
    gate pair), C (if kept) and the plasticity-off control."""
    w, kc = c["workers"], spec.k_cap
    n_naive = n_set if n_naive is None else n_naive
    parts = dict(oracle=_rounds(n_set, w) * c["oracle_round_s"],
                 naive=_rounds(n_naive * F, w) * 2 * K * c["presentation_s"],
                 learn=_rounds(kc * F * 3, w) * job_s(c, K),
                 band=_rounds(kc * F * 3, w) * job_s(c, K),
                 c=(_rounds(kc * F * 3, w) * job_s(c, K)) if with_c else 0.0,
                 noplast=_rounds(spec.noplast_pairs * spec.noplast_flies, w) * job_s(c, K))
    return dict(parts_h={k: v / 3600 for k, v in parts.items()}, total_h=sum(parts.values()) / 3600)


def elapsed_h(ledger: list) -> float:
    return float(sum(e.get("wall_s", 0.0) for e in ledger)) / 3600


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
```

- [ ] **Step 4: Write `w_rules`**

```python
"""W's gate decisions and sentences (W.3, W.7 as amended by W.9.3 / W.9.4 / W.9.6 / W.9.7 / W.9.8 / W.9.9). The pair
and overall verdict live in w_verdict (the authoritative judgement code); this module decides the gates around it and
fills the closing sentences (〈…〉 → {field}). Every number comes from the WSpec passed in."""
from __future__ import annotations

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = "PASS", "INVALID", "NOT_READ", "SEALED", "READ", "INVALID_RUN"
STOP_REUSE, STOP_W_PATH_REPRO = "STOP_REUSE", "STOP_W_PATH_REPRO"
STOP_PILOT_NO_EFFECT, STOP_OC_UNREACHABLE = "STOP_PILOT_NO_EFFECT", "STOP_OC_UNREACHABLE"
STOP_BUDGET, STOP_FEW_PAIRS = "STOP_BUDGET", "STOP_FEW_PAIRS"
V_PASS, V_FAIL, V_UNDECIDED, V_STOP_MACHINE = "PASS", "FAIL", "UNDECIDED", "STOP_MACHINE"
LEVER_TXT = "C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, 엔진별 z"

SENTENCES = {
    STOP_REUSE: "W 재사용 조건(W.3 1)이 깨졌다({why}). W는 V의 블록을 다시 재는 경로를 갖지 않으므로 주 세트를 쓰지 "
                "않고 멈춘다 — 사용자 몫.",
    STOP_W_PATH_REPRO: "W 학습 경로가 {ref}을 재현하지 못했다({diff}).",
    STOP_PILOT_NO_EFFECT: "파일럿에서 조합 지렛대의 F.2 학습 효과가 {cond} — 주 세트를 쓰지 않고 멈춘다.",
    STOP_OC_UNREACHABLE: "파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다({p_by_k}).",
    STOP_BUDGET: "남은 추정 비용 {h}가 W 상한 24 h를 넘는다.",
    STOP_FEW_PAIRS: "W 주 세트 {n}쌍에서 오라클 순진 균형·시험 가능 쌍이 {k}개로 최소 4에 못 미쳤다.",
    V_STOP_MACHINE: "기계 검사 불일치({why}) — F.7의 STOP_MACHINE. W.9.8 H8에 따라 같은 관문 쌍으로 다시 돌리지 "
                    "않는다(INVALID_RUN 여부와 엔진·풀 결함 처리는 사용자 몫).",
    V_UNDECIDED: "원인({cause})을 기록하고 사용자 판단.",
    V_PASS: "조합 지렛대 모델(" + LEVER_TXT + ")에서 실제 학습 규칙으로 M2 학습 단위가 섰다 — 넓힌 풀(생성원 턴 "
            "306–1985), 오라클 순진·시험 가능 쌍 조건부(관문 쌍 {k}개, 판정 가능 {m}개, 설계 q {q} · K {K} · F {F} · "
            "k 상한 {k_max}, 마리별 동시 충족).",
    V_FAIL: "F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부({fails}; 기계 대조 실패 "
            "{n_mech}쌍) (마리별 동시 충족 집계).",
}
CONSEQUENCE = {
    V_PASS: "POOL 배틀 과제(M3)는 별도 선언이 필요하다(T.9.5 좁힘 유지).",
    V_FAIL: "STD 재설계 여부는 사용자 몫(F.7은 D.6 (c) 충족을 말하지만, 이 판정이 지렛대 엔진 조건부임을 함께 적는다).",
    V_UNDECIDED: "사용자 판단(F.7).",
    V_STOP_MACHINE: "사용자 몫(W.7, W.9.8 H8).",
}


def sentence(outcome: str, fields: dict) -> str:
    return SENTENCES[outcome].format(**fields)


# ================================================================ W.3 1: reuse
def reuse(v_dec: dict, v_doc: dict, v_git: dict, ancestors: dict, last_v_commit: str | None, u_key: str,
          code_key: str, t_key: str, spec) -> dict:
    """V's own reuse condition (R, T, U through v_rules.reuse on V's spec) plus V's blocks: V's summary tracked and
    clean and last changed by V's judge commit, the four V commits in HEAD's history, V's blocks z / kc_input / set /
    judge on V's keys (U's measurement key literal, R's shared key, T's key), V's judgement SELECTED, z_V as
    declared."""
    why = list(v_dec.get("reasons") or [])
    if u_key != spec.u_measure_key_u:
        why.append(f"U 측정 키 {u_key} ≠ {spec.u_measure_key_u}")
    if code_key != spec.r_shared_key or t_key != spec.t_measure_key_t:
        why.append(f"공유 키 {code_key} / T 키 {t_key}")
    if not v_git.get("tracked") or v_git.get("dirty"):
        why.append(f"{spec.v_summary} 미커밋")
    judge_commit = dict(spec.v_commits)["judge"]
    if not (last_v_commit or "").startswith(judge_commit):
        why.append(f"{spec.v_summary}의 마지막 커밋 {last_v_commit} ≠ V 판정 {judge_commit}")
    for b, c in spec.v_commits:
        if not ancestors.get(c):
            why.append(f"V 커밋 {c}({b})가 HEAD 이력에 없음")
        blk = v_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"V 블록 {b} 없음")
            continue
        if (blk.get("u_measure_key"), blk.get("code_key"), blk.get("t_measure_key")) != (
                spec.u_measure_key_u, spec.r_shared_key, spec.t_measure_key_t):
            why.append(f"V 블록 {b}의 키")
    if (v_doc.get("judge") or {}).get("band") != spec.v_band:
        why.append(f"V 판정 {(v_doc.get('judge') or {}).get('band')}")
    zv = (v_doc.get("z") or {}).get("z_V") or {}
    if {k: tuple(round(float(x), spec.z_v_digits) for x in v) for k, v in zv.items()} != spec.z_v():
        why.append(f"V z_V {zv}")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


# ================================================================ W.3 2: the path gate
def path(checks: list) -> dict:
    """checks = [dict(ref, diffs, invalid)] in order; a defect is INVALID, a difference STOP_W_PATH_REPRO (the first
    failing check in the sentence, every one listed)."""
    bad = [m for c in checks for m in c.get("invalid", [])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    failed = [c for c in checks if c["diffs"]]
    if failed:
        c = failed[0]
        return dict(outcome=STOP_W_PATH_REPRO, reasons=[],
                    failed=[dict(ref=x["ref"], diffs=x["diffs"]) for x in failed],
                    sentence=sentence(STOP_W_PATH_REPRO, dict(ref=c["ref"], diff="; ".join(c["diffs"][:6]))))
    return dict(outcome=PASS, reasons=[], failed=[])


# ================================================================ W.9.8 H6: the pilot's stop
def pilot(rec: dict, machine: list, spec) -> dict:
    """INVALID on a machine reason; then H6 (i)-(iii) on the pilot pairs' fly medians; else PASS."""
    if machine:
        return dict(outcome=INVALID, reasons=machine)
    meds = [p["median"] for p in rec["pairs"].values()]
    n = len(meds)
    r_ok = sum(m["reward_assoc"] >= spec.no_effect_d for m in meds) / n
    p_ok = sum(m["punish_assoc"] <= -spec.no_effect_d for m in meds) / n
    rev = sum(m["reward_assoc"] < 0 or m["punish_assoc"] > 0 for m in meds) / n
    fl = rec["floor_both_share"]
    hit = []
    if not (r_ok >= spec.no_effect_share and p_ok >= spec.no_effect_share):
        hit.append(f"(i) 보상 연합 d′ ≥ 0.5 쌍 비율 {r_ok:.3f}·처벌 연합 d′ ≤ −0.5 쌍 비율 {p_ok:.3f}(둘 다 ≥ 0.5 아님)")
    if rev > 0.5:
        hit.append(f"(ii) 보상 또는 처벌 연합 부호가 반대인 쌍 비율 {rev:.3f}(> 0.5)")
    if fl > spec.floor_share_max:
        hit.append(f"(iii) X·Y 함께 바닥인 프로브 비율 {fl:.3f}(> 0.5)")
    vals = dict(reward_share=r_ok, punish_share=p_ok, reversed_share=rev, floor_both_share=fl)
    if hit:
        return dict(outcome=STOP_PILOT_NO_EFFECT, reasons=hit, values=vals,
                    sentence=sentence(STOP_PILOT_NO_EFFECT, dict(cond="; ".join(hit))))
    return dict(outcome=PASS, reasons=[], values=vals)


# ================================================================ W.9.3 / H1: the OC's outcome
def oc(doc: dict, spec) -> dict:
    if doc.get("selected"):
        return dict(outcome=PASS, reasons=[], design=doc["selected"])
    p_by_k = oc_unreachable_text(doc, spec)
    return dict(outcome=STOP_OC_UNREACHABLE, reasons=["선택 규칙을 만족하는 설계 없음"],
                sentence=sentence(STOP_OC_UNREACHABLE, dict(p_by_k=p_by_k)))


def oc_unreachable_text(doc: dict, spec) -> str:
    """OPEN 3's default: for every (q, K) at F = 32, the k-wise bootstrap limits "power lower / false-pass upper"
    (simultaneous over k, the worst cluster level); or the calibration status when the targets were unreachable."""
    if not doc.get("reachable"):
        cal = doc.get("calibration") or {}

        def st(m, ab):
            return ((cal.get(m) or {}).get(ab) or {}).get("status")
        return "보정 불가 — " + "; ".join(f"{m}: a {st(m, 'a')}, b {st(m, 'b')}" for m in ("min", "max"))
    out = []
    lo, hi = doc["power_lo"], doc["false_hi"]
    for qi, q in enumerate(spec.q_grid):
        for ki, K in enumerate(spec.k_grid):
            out.append(f"q {q}·K {K}·F 32: 검정력 하한 {lo[qi][ki][-1]:.3f}, 거짓 통과 상한 {hi[qi][ki][-1]:.3f}")
    return "; ".join(out)


# ================================================================ W.9.6 F / W.9.9 P1-4 / P1-5: the budget
def budget(elapsed_h: float, options: list, spec) -> dict:
    """options = [dict(design, with_c, total_h)] in the order to try (selected with C, selected without C, then the
    alternative designs without C). The first whose elapsed + total ≤ 24 h is the plan; none → STOP_BUDGET."""
    for o in options:
        if elapsed_h + o["total_h"] <= spec.budget_h:
            return dict(outcome=PASS, reasons=[], plan=o, elapsed_h=elapsed_h)
    worst = options[0]
    h = f"누적 {elapsed_h:.2f} h + 남은 {worst['total_h']:.2f} h = {elapsed_h + worst['total_h']:.2f} h"
    return dict(outcome=STOP_BUDGET, reasons=[h], plan=None, elapsed_h=elapsed_h,
                sentence=sentence(STOP_BUDGET, dict(h=h)))


# ================================================================ W.2 / W.9.5 / H7: the gate pairs
def gate_pairs(n_set: int, k: int, spec) -> dict:
    if k < spec.min_gate_pairs:
        return dict(outcome=STOP_FEW_PAIRS, reasons=[f"관문 쌍 {k} < {spec.min_gate_pairs}"],
                    sentence=sentence(STOP_FEW_PAIRS, dict(n=n_set, k=k)))
    return dict(outcome=PASS, reasons=[])


# ================================================================ W.7 / W.9.7 / W.9.8: the verdict's sentence
def verdict_sentence(v: dict, design: dict, spec) -> dict:
    """v = w_verdict.judge's output; design = (q, K, F) of the plan."""
    out = v["verdict"]
    if out == V_PASS:
        f = dict(k=v["n_pairs"], m=v["n_judgeable"], q=design["q"], K=design["K"], F=design["F"], k_max=spec.k_cap)
    elif out == V_FAIL:
        f = dict(fails="; ".join(f"{k}: " + "·".join(v["pairs"][k]["reasons"]) + _gates(v["pairs"][k])
                                 for k in v["failing"]), n_mech=len(v["mech_fail"]))
    elif out == V_UNDECIDED:
        f = dict(cause=v["undecided_cause"])
    else:
        f = dict(why="; ".join((v["machine"] + [f"INVALID 쌍 {k}" for k in v["invalid"]])[:6]))
    return dict(sentence=sentence(out, f), consequence=CONSEQUENCE[out])


def _gates(p: dict) -> str:
    g = p.get("failing_gates") or {}
    hit = [f"{k} {n}마리" for k, n in g.items() if n]
    return f"({', '.join(hit)})" if hit else ""
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_w_records.py tests/brain/test_w_rules.py -q`
Expected: 13 passed.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/w_records.py flymon/brain/w_rules.py tests/brain/test_w_records.py tests/brain/test_w_rules.py
git commit -m "feat(w): w_records (job rows -> verdict arrays, machine checks incl. RN's own reward phase, path comparisons, pilot record, cost model) and w_rules (reuse, path, H6, OC, budget, few pairs, every sentence)"
```

---

### Task 5: `w_measure` and `w_store` — W's only measurement file, the writer and the caches

**Files:**
- Create: `flymon/brain/w_measure.py`, `flymon/brain/w_store.py`
- Test: `tests/brain/test_w_measure.py`, `tests/brain/test_w_store.py`

**Interfaces:**
- Consumes: `u_measure.u_rig` / `apply_u_edit` / `U_MEASURE_FILES` / `u_arm_job` (tests), `o_jobs.train_x` / `DaMeter` / `weights_sha256`, `n_jobs._present` / `readout_cells`, `r_store.RCache`, `w_records` (tests; Task 4), `w_spec.SPEC`.
- Produces: `W_MEASURE_FILES = ("flymon/brain/w_measure.py",)`, `KIND = "w_learn"`, `w_measure_key(npz) -> dict`, `w_learn_job(eng, pl, pops, comps, ro, params, edit, p_type, odor_x, odor_y, readout, reward_type, punish_type, phases, probe_seeds, strength, settle_ms, read_ms, window_ms, present_ms, gap_ms, train_settle_ms, fly, seed_stride, plastic) -> dict` (`edit`, `csc_sha256`, `edit_edges`, `block_edges`, `w0_sha256`, `fly`, `plastic`, `probe_seeds`, `stages` = [{`stage` pre / S1 / S2, `x`, `y`, `w_sha256`, `weights_frac`, `weights_frac_A`, `weights_frac_P`, `da_integral`}], `wall_s`, `probe_s`, `train_s`); `WMeasurer(pool, cache, params, readout, p_type, reward_type, punish_type, windows, timing)` with `kwargs(u, override=None)`, `inputs(u, block)`, `run(units, **override) -> list`, `learn(units, block, check=None) -> [dict(unit, result, cache_key, cache_file)]`. `w_store`: `ALLOWED_DIR`, `SUMMARY`, `SMOKE_SEEDS`, `guard`, `write_bytes`, `write_json`, `read_summary`, `write_summary_block(path, block, obj, params_list, ledger=None)`, `WCache(root, code)`, `VReadCache(root, code)`, `load_manifest`, `archive_copy`.

- [ ] **Step 1: Write the failing tests**

```python
"""W's measurement file (W.1, W.3 2, W.9.6 P2-10, W.9.8 H9): the W measurement key (U's files + w_measure.py), the
measurer's cache (one entry per pair · fly · brain, every input in the key, resume with the missing units only, the
budget check between rounds), and on the real connectome: W's job with P's punish-arm settings equals u_arm_job (L_V)
and r_arm_job ("none") field for field; its naive probes equal presentation.decide's counts (the oracle's probe) on
the same seeds; RN1 = R1 bit for bit (counts and weights) while N differs and R2 ≠ RN2; a rig whose edit is not applied
fails the reproduction (measured). RN's independence from R (W.9.6 P2-10) is pinned in test_w_records (the machine
check catches an RN that skipped its own reward phase) and test_w_runner (every RN unit carries both phases)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.brain import u_measure as UM
from flymon.brain import w_measure as WM
from flymon.brain.config import Params
from flymon.brain.r_measure import R_MEASURE_FILES
from flymon.brain.t_measure import T_MEASURE_FILES
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.w_store import WCache

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
READOUT = {"A": "MBON13", "P": "MBON05"}


def test_w_measure_key_is_u_files_plus_w_measure():
    assert WM.W_MEASURE_FILES == ("flymon/brain/w_measure.py",)
    files = set(R_MEASURE_FILES) | set(T_MEASURE_FILES) | set(UM.U_MEASURE_FILES)
    assert not set(WM.W_MEASURE_FILES) & files
    if NPZ.exists():
        k, u = WM.w_measure_key(str(NPZ)), UM.u_measure_key(str(NPZ))
        assert set(k["files"]) == set(u["files"]) | set(WM.W_MEASURE_FILES)
        assert k["key"] != u["key"] and len(k["key"]) == 64


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        assert fn is WM.w_learn_job
        self.calls.append(len(kws))
        return [dict(fly=kw["fly"], probe_seeds=kw["probe_seeds"], stages=[], n=len(self.calls)) for kw in kws]


def _unit(fly, brain="R", k=2, trials=20, block_dan="PAM08"):
    return dict(pair="b|306|x|y", idx=0, fly=fly, brain=brain, edit=V.lever_edit, odor_x={"G1": 1.0},
                odor_y={"G2": 1.0}, phases=[[block_dan, trials, 28_000_000], [None, trials, 28_000_020]],
                probe_seeds=W.probe_seeds(0, fly, k), plastic=True)


def _measurer(pool, tmp_path):
    cache = WCache(tmp_path / "results/w/cache", {"key": "w" * 64})
    return WM.WMeasurer(pool, cache, Params(), READOUT, "MBON05", "PAM08", "PPL105",
                        dict(strength=1.0, settle_ms=800.0, read_ms=600.0, window_ms=200),
                        dict(present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, seed_stride=1000))


def test_cache_unit_resume_and_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/w").mkdir(parents=True)
    pool = FakePool(2)
    m = _measurer(pool, tmp_path)
    units = [_unit(f, b) for f in range(2) for b in ("R", "N", "RN")]
    got = m.learn(units[:3], "pilot")
    assert pool.calls == [2, 1] and [g["unit"]["brain"] for g in got] == ["R", "N", "RN"]
    got = m.learn(units, "pilot")                                      # resume: only the 3 missing units
    assert pool.calls == [2, 1, 2, 1] and len(got) == 6 and m.last_jobs == 3
    assert len({g["cache_key"] for g in got}) == 6
    base = m.inputs(units[0], "pilot")
    for change in (dict(probe_seeds=W.probe_seeds(0, 0, 3)), dict(fly=1), dict(brain="N"), dict(edit="none"),
                   dict(phases=[["PAM08", 19, 28_000_000], [None, 20, 28_000_020]]), dict(pair="a|1|x|y")):
        u = dict(units[0], **change)
        assert m.cache.key(WM.KIND, m.inputs(u, "pilot")) != m.cache.key(WM.KIND, base), change
    assert m.cache.key(WM.KIND, m.inputs(units[0], "learn")) != m.cache.key(WM.KIND, base)


def test_check_runs_before_each_round_and_can_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/w").mkdir(parents=True)
    pool = FakePool(2)
    m = _measurer(pool, tmp_path)
    seen = []

    class Stop(Exception):
        pass

    def check(done, todo):
        seen.append((done, todo))
        if done >= 2:
            raise Stop
    with pytest.raises(Stop):
        m.learn([_unit(f) for f in range(5)], "learn", check=check)
    assert seen == [(0, 5), (2, 5)] and pool.calls == [2]
    assert len(m.learn([_unit(f) for f in range(2)], "learn")) == 2 and pool.calls == [2]


# ================================================================ the real connectome
@pytest.fixture(scope="module")
def real():
    if not NPZ.exists():
        pytest.skip("no connectome")
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_spec import make_odors
    conn = Connectome.load(str(NPZ))
    pops = Populations.from_connectome(conn)
    o = make_odors(pops, H3.reference)[0]["strengths"]
    ks = sorted(o)
    return SimpleNamespace(conn=conn, pops=pops, params=load_c3_config(V.m0d_summary).params,
                           eng=SimpleNamespace(conn=conn), ox={k: o[k] for k in ks[: len(ks) // 2]},
                           oy={k: o[k] for k in ks[len(ks) // 2:]})


def _p_kw(real):
    n = V.p.o.n
    h4 = n.h4
    return dict(params=real.params, p_type="MBON05", odor_x=real.ox, odor_y=real.oy, readout=READOUT,
                punish_type="PPL105", reward_type="PAM08", strength=n.h3.strength,
                settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms),
                present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms, train_settle_ms=h4.teach_window.settle_ms)


@pytest.mark.parametrize("edit", [V.lever_edit, "none"])
def test_p_punish_arm_reproduced(real, edit):
    """W.3 2 (i) in miniature (2 trials): phases [[PPL105, 2, base]], probe seed = the arm seed, P's windows —
    every field of u_arm_job's punish arm (r_arm_job's for "none")."""
    from flymon.brain import w_records as WR
    kw, n, seed = _p_kw(real), V.p.o.n, 25_400_000
    ref = UM.u_arm_job(real.eng, None, real.pops, None, None, edit=edit, seed=seed, arm="punish", punish=True,
                       plastic=True, da_zero=False, trials=2, seed_base=n.train_seed_base,
                       seed_stride=n.train_seed_stride, **kw)
    mine = WM.w_learn_job(real.eng, None, real.pops, None, None, edit=edit, phases=[["PPL105", 2, n.train_seed_base]],
                          probe_seeds=[seed], fly=seed, seed_stride=n.train_seed_stride, plastic=True, **kw)
    assert WR.p_repro_diffs(mine, ref, "t") == []
    assert mine["block_edges"] == ({} if edit == "none" else dict(V.contrast_declared()["chain_entry"]))


def test_naive_probe_is_the_oracles_decide(real):
    """W.3 2 (ii) in miniature: W's naive probes (phases []) = presentation.decide's per-type sums on the same seeds."""
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.presentation import decide
    from flymon.brain import w_records as WR
    seeds = [24_600_200, 24_600_201]
    mine = WM.w_learn_job(real.eng, None, real.pops, None, None, params=real.params, edit=V.lever_edit,
                          p_type="MBON05", odor_x=real.ox, odor_y=real.oy, readout=READOUT, reward_type="PAM08",
                          punish_type="PPL105", phases=[], probe_seeds=seeds, strength=1.0, settle_ms=V.settle_ms,
                          read_ms=V.read_ms, window_ms=V.window_ms, present_ms=400.0, gap_ms=200.0,
                          train_settle_ms=800.0, fly=0, seed_stride=1000, plastic=True)
    e, p, *_ = UM.u_rig(real.conn, real.pops, real.params, V.lever_edit, "MBON05")
    cells = type_cells(e.conn, ["MBON13", "MBON05"])
    idx = np.concatenate([cells["MBON13"], cells["MBON05"]])
    n13 = len(cells["MBON13"])
    p.reset_weights()
    ref = []
    for s in seeds:
        c = decide(e, p, real.pops, [real.ox, real.oy], 1.0, s, V.settle_ms, V.read_ms, idx=idx)
        ref.append([[c[0, :n13].sum(), c[1, :n13].sum()], [c[0, n13:].sum(), c[1, n13:].sum()]])
    assert np.array_equal(WR.counts(mine["stages"][0]), np.asarray(ref)) and len(mine["stages"]) == 1


def _learn(real, dans, edit=V.lever_edit, trials=2):
    return WM.w_learn_job(real.eng, None, real.pops, None, None, params=real.params, edit=edit, p_type="MBON05",
                          odor_x=real.ox, odor_y=real.oy, readout=READOUT, reward_type="PAM08", punish_type="PPL105",
                          phases=[[dans[0], trials, 41_000_000], [dans[1], trials, 41_000_020]],
                          probe_seeds=[40_000_000], strength=1.0, settle_ms=V.settle_ms, read_ms=V.read_ms,
                          window_ms=V.window_ms, present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, fly=0,
                          seed_stride=1000, plastic=True)


def test_rn1_equals_r1_bitwise_and_n_differs(real):
    """G.5 / W.9.6 P2-10: RN runs from pre on its own; after the shared reward phase RN1 = R1 (counts and weights);
    N (no DAN) differs; the punishment phase separates R2 from RN2."""
    from flymon.brain import w_records as WR
    r, rn, n = _learn(real, ("PAM08", "PPL105")), _learn(real, ("PAM08", None)), _learn(real, (None, None))
    st = {k: {s["stage"]: s for s in v["stages"]} for k, v in (("R", r), ("RN", rn), ("N", n))}
    assert np.array_equal(WR.counts(st["R"]["S1"]), WR.counts(st["RN"]["S1"]))
    assert st["R"]["S1"]["w_sha256"] == st["RN"]["S1"]["w_sha256"] != st["N"]["S1"]["w_sha256"]
    assert st["R"]["S2"]["w_sha256"] != st["RN"]["S2"]["w_sha256"]
    assert np.array_equal(WR.counts(st["R"]["pre"]), WR.counts(st["N"]["pre"]))


def test_mutation_unapplied_edit_fails_the_reproduction(real, monkeypatch):
    """A rig whose combined edit never reaches the engine measures other counts than the real L_V rig (U.9.1's
    mutation, measured): the path gate would stop."""
    from flymon.brain import w_records as WR
    good = WM.w_learn_job(real.eng, None, real.pops, None, None, **dict(
        _p_kw(real), edit=V.lever_edit, phases=[], probe_seeds=[25_400_000], fly=0, seed_stride=1000, plastic=True))
    real_apply = UM.apply_u_edit
    monkeypatch.setattr(UM, "apply_u_edit", lambda eng, pops, edit, p_type: real_apply(eng, pops, "none", p_type))
    UM._RIG.clear()
    bad = WM.w_learn_job(real.eng, None, real.pops, None, None, **dict(
        _p_kw(real), edit=V.lever_edit, phases=[], probe_seeds=[25_400_000], fly=0, seed_stride=1000, plastic=True))
    UM._RIG.clear()
    assert not np.array_equal(WR.counts(good["stages"][0]), WR.counts(bad["stages"][0]))
```

```python
"""W's writer (W.8): writes only under results/w/ and results/summary/w_learning.json (SystemExit 2 otherwise, R / S /
T / U / V trees included), atomically; the budget ledger is appended with a block; WCache keeps smoke and real seeds
apart; V's cache is read only."""
import json

import pytest

from flymon.brain import w_store as WS
from flymon.brain.config import Params
from flymon.brain.w_spec import SPEC


def test_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    WS.write_json("results/w/x.json", {"a": 1}, [Params()])
    WS.write_summary_block(WS.SUMMARY, "reuse", {"b": 2}, [Params()], ledger=dict(stage="reuse", wall_s=1.0))
    WS.write_summary_block(WS.SUMMARY, "path", {"c": 3}, [Params()], ledger=dict(stage="path", wall_s=2.0))
    d = json.loads((tmp_path / WS.SUMMARY).read_text())
    assert d["reuse"] == {"b": 2} and [e["stage"] for e in d["ledger"]] == ["reuse", "path"]
    for bad in ("results/v/x.json", "results/summary/v_lever.json", "results/u/x.json", "results/r/cache/x.json",
                "x.json", "results/wx/a.json"):
        with pytest.raises(SystemExit) as e:
            WS.write_json(bad, {}, [Params()])
        assert e.value.code == 2
    assert not list((tmp_path / "results/w").glob(".*.tmp"))


def test_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = WS.WCache("results/w/cache", {"key": "k"})
    s = WS.WCache("results/w/smoke/cache", {"key": "k"})
    real = dict(probe_seeds=SPEC.probe_seeds(0, 0, 2))
    sm = dict(probe_seeds=SPEC.smoke_probe_seeds(0, 2))
    c.put("w_learn", real, {"r": 1}, [Params()])
    assert c.get("w_learn", real) == {"r": 1}
    s.put("w_learn", sm, {"r": 2}, [Params()])
    for cache, ins in ((c, sm), (s, real)):
        with pytest.raises(SystemExit):
            cache.put("w_learn", ins, {}, [Params()])
    v = WS.VReadCache("results/v/cache", {"key": "u"})
    assert v.get("r_arm", dict(seed=25_400_000)) is None
    with pytest.raises(SystemExit):
        v.put("r_arm", dict(seed=25_400_000), {}, [Params()])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_measure.py tests/brain/test_w_store.py -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write `w_measure`**

```python
"""W's only measurement file (W.0, W.1, W.8, W.9.6 P2-10 / P2-12, W.9.8 H9): the sequential F.2 learning job on U's rig
(u_measure.u_rig, the combined lever L_V or "none"), and the cache plumbing around it. The shared measurement files
(r_measure.R_MEASURE_FILES), T's (t_measure.T_MEASURE_FILES) and U's (u_measure.U_MEASURE_FILES) are imported, never
edited; this file joins them in the W measurement key (= U's measurement key 8a4e0930… + this file), which keys W's
cache and every W block.
- w_learn_job: one (pair, fly, brain) unit. The rig's weights are reset; the probes (X then Y per probe seed,
  n_jobs._present — the presentation sequence of P's arm job and of the oracle's decide: reset, traces, drive, DAN
  quiet, present, settle + read; plasticity off) run at "pre", then after every phase. A phase is o_jobs.train_x
  unchanged — X alone, `trials` trials, train seed = seed_base + fly·seed_stride + t, settle with the weights frozen,
  the phase's DAN (None: no injection) for present_ms, the gap, recovery after a pulse — with plasticity on iff
  `plastic`, the phasic dopamine integrated per compartment (o_jobs.DaMeter). After each phase: the plastic weights'
  sha256 and the weight fractions (all, the punishment / reward cores). The rig is restored on exit.
  F.2's R = [PAM08, PPL105], N = [None, None], G.5's RN = [PAM08, None] (w_spec.phases); each brain is its own job
  from the naive rig (RN never shares R's state, W.9.6 P2-10). With phases [[PPL105, 12, base]], probe seed = seed and
  P's windows / timings, it is u_arm_job's punish arm (W.3 2 (i)); with phases [] it is the naive probe (W.3 2 (ii)).
- WMeasurer: one cache entry ("w_learn") per unit, keyed by the W measurement key + every job input + pair, brain and
  block (H9: pair · fly · brain · stage list · K · probe seeds · training arguments · engine); a round of n_workers
  units is written as soon as it returns, so a stage resumes with the missing units only. `run` skips the cache (the
  path gate measures afresh)."""
from __future__ import annotations

import sys
import time

from . import o_jobs
from .h3_store import code_key
from .n_jobs import _present, readout_cells
from .r_measure import R_MEASURE_FILES
from .t_measure import T_MEASURE_FILES
from .u_measure import U_MEASURE_FILES, u_rig

W_MEASURE_FILES = ("flymon/brain/w_measure.py",)
KIND = "w_learn"


def w_measure_key(npz: str) -> dict:
    """W.1: the U measurement key's files plus W's measurement file."""
    return code_key(npz, files=tuple(dict.fromkeys(R_MEASURE_FILES + T_MEASURE_FILES + U_MEASURE_FILES
                                                   + W_MEASURE_FILES)))


def w_learn_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, odor_x: dict, odor_y: dict, readout: dict,
                reward_type: str, punish_type: str, phases, probe_seeds, strength: float, settle_ms: float,
                read_ms: float, window_ms: int, present_ms: float, gap_ms: float, train_settle_ms: float, fly: int,
                seed_stride: int, plastic: bool) -> dict:
    e, p, c, sha, n_edit, blocks = u_rig(eng.conn, pops, params, edit, p_type)
    a_cells, p_cells = readout_cells(e.conn, readout)
    a_core, p_core = c[punish_type].core, c[reward_type].core
    t0 = time.perf_counter()
    walls = dict(probe_s=0.0, train_s=0.0)

    def probes() -> dict:
        t = time.perf_counter()
        was = p.enabled
        p.set_enabled(False)
        try:
            out = {"x": [], "y": []}
            for s in probe_seeds:
                for k, o in (("x", odor_x), ("y", odor_y)):
                    out[k].append(_present(e, p, pops, o, int(s), strength, settle_ms, read_ms, window_ms, a_cells,
                                           p_cells)[0])
            return out
        finally:
            p.set_enabled(was)
            walls["probe_s"] += time.perf_counter() - t

    def snap(name: str, pr: dict, da) -> dict:
        return dict(stage=name, x=pr["x"], y=pr["y"], w_sha256=o_jobs.weights_sha256(e.csc.w[p.edges]),
                    weights_frac=float(p.weights_frac()), weights_frac_A=float(p.weights_frac_by_mbon_set(a_core)),
                    weights_frac_P=float(p.weights_frac_by_mbon_set(p_core)), da_integral=da)

    meter = o_jobs.DaMeter(e, p)
    try:
        p.reset_weights()
        w0 = o_jobs.weights_sha256(p.w0)
        stages = [snap("pre", probes(), None)]
        for i, (dan, trials, base) in enumerate(phases):
            t = time.perf_counter()
            meter.acc[:] = 0
            p.set_enabled(bool(plastic))
            with meter:
                o_jobs.train_x(e, p, pops, odor_x, strength, int(fly), dan, int(trials), present_ms, gap_ms,
                               train_settle_ms, int(base), int(seed_stride), meter)
            p.set_enabled(True)
            walls["train_s"] += time.perf_counter() - t
            stages.append(snap(f"S{i + 1}", probes(), meter.by_compartment(c)))
    finally:
        e.on_step = p.on_step
        p.reset_weights()
        p.set_enabled(True)
    return dict(edit=edit, csc_sha256=sha, edit_edges=int(n_edit), block_edges=dict(blocks), w0_sha256=w0,
                fly=int(fly), plastic=bool(plastic), probe_seeds=[int(s) for s in probe_seeds], stages=stages,
                wall_s=time.perf_counter() - t0, **walls)


class WMeasurer:
    """w_learn_job over a pool (FlyPool.run_jobs) behind a cache (w_store.WCache)."""

    def __init__(self, pool, cache, params, readout: dict, p_type: str, reward_type: str, punish_type: str,
                 windows: dict, timing: dict):
        """windows = {strength, settle_ms, read_ms, window_ms}; timing = {present_ms, gap_ms, train_settle_ms,
        seed_stride}."""
        self.pool, self.cache, self.params = pool, cache, params
        self.common = dict(params=params, p_type=p_type, readout=dict(readout), reward_type=reward_type,
                           punish_type=punish_type, **{k: windows[k] for k in ("strength", "settle_ms", "read_ms")},
                           window_ms=int(windows["window_ms"]), **timing)
        self.last_wall_s, self.last_jobs = 0.0, 0

    def kwargs(self, u: dict, override: dict | None = None) -> dict:
        return dict(self.common, **(override or {}), edit=u["edit"], odor_x=dict(u["odor_x"]), odor_y=dict(u["odor_y"]),
                    phases=[list(ph) for ph in u["phases"]], probe_seeds=[int(s) for s in u["probe_seeds"]],
                    fly=int(u["fly"]), plastic=bool(u["plastic"]))

    def inputs(self, u: dict, block: str) -> dict:
        return dict(self.kwargs(u), pair=u["pair"], brain=u["brain"], block=block)

    def run(self, units: list, **override) -> list:
        """No cache (the path gate): rows in order; `override` replaces common arguments (P's windows and timings for
        W.3 2 (i))."""
        n_w = max(1, int(self.pool.n_workers))
        out, t0 = [], time.perf_counter()
        for a in range(0, len(units), n_w):
            out += self.pool.run_jobs(w_learn_job, [self.kwargs(u, override) for u in units[a:a + n_w]])
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(units)
        return out

    def learn(self, units: list, block: str, check=None) -> list:
        """Every unit through the cache; check(done, todo) after each round may raise to stop the stage (the budget
        ledger). Rows in order, each with its cache key and file."""
        ins = [self.inputs(u, block) for u in units]
        todo = [i for i, x in enumerate(ins) if self.cache.get(KIND, x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        t0 = time.perf_counter()
        if todo:
            print(f"w learn {block}: {len(todo)}/{len(units)} units to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            if check is not None:
                check(a, len(todo))
            batch = todo[a:a + n_w]
            for i, r in zip(batch, self.pool.run_jobs(w_learn_job, [self.kwargs(units[i]) for i in batch])):
                self.cache.put(KIND, ins[i], r, [self.params])
            print(f"w learn {block}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for u, x in zip(units, ins):
            got = self.cache.get(KIND, x)
            if got is None:
                raise RuntimeError(f"w learn {block}: {u['pair']} fly {u['fly']} {u['brain']} missing after the run")
            out.append(dict(unit=u, result=got, cache_key=self.cache.key(KIND, x),
                            cache_file=str(self.cache._path(KIND, x))))
        return out
```

- [ ] **Step 4: Write `w_store`**

```python
"""W's only writer (W.8): raw files under results/w/, the one summary results/summary/w_learning.json, atomic writes
(temporary file + rename), nothing else (SystemExit 2).
- WCache: r_store.RCache (key over kind + inputs + the W measurement key; smoke and real entries never share a root)
  with `put` routed through this module's writer and W's smoke seeds as the smoke scope.
- VReadCache: V's cache root (results/v/cache) read only under V's measurement key (= U's) — the path gate reads V's
  gate ② arm rows by content key; put refuses.
- load_manifest / archive_copy: r_store's (paths and root are arguments)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache, archive_copy, load_manifest  # noqa: F401  (re-exported for w_runner)
from .w_spec import SPEC as W_SPEC

ALLOWED_DIR = "results/w/"
SUMMARY = "results/summary/w_learning.json"
SMOKE_SEEDS = W_SPEC.smoke_seed_set()


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"W writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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


def write_summary_block(path, block: str, obj, params_list, ledger: dict | None = None) -> Path:
    """One block, plus (optionally) one entry appended to the running budget ledger (W.9.6 F)."""
    doc = read_summary(path)
    doc[block] = obj
    if ledger is not None:
        doc["ledger"] = list(doc.get("ledger", [])) + [ledger]
    return write_json(path, doc, params_list)


class WCache(RCache):
    """RCache with W's smoke seeds as the smoke scope and W's writer (a root outside results/w/ refuses on put)."""

    def __init__(self, root, code: dict, smoke_seeds=SMOKE_SEEDS):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)


class VReadCache(RCache):
    """V's raw data, read only (W.3 2): get as RCache; put refuses."""

    def __init__(self, root, code: dict):
        super().__init__(root, code, smoke_seeds=())

    def put(self, kind, inputs, result, params_list) -> None:
        _refuse(f"{self.root} is V's raw data; W reads it by content key and never writes it")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/brain/test_w_measure.py tests/brain/test_w_store.py tests/brain/test_u_measure.py -q`
Expected: all pass (`test_w_measure.py` runs the real connectome: ~2.5 min).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/w_measure.py flymon/brain/w_store.py tests/brain/test_w_measure.py tests/brain/test_w_store.py
git commit -m "feat(w): w_measure (W-only measurement file: sequential R/N/RN job on u_rig via o_jobs.train_x, W key = U files + w_measure.py, cache per pair/fly/brain) and w_store (results/w/ + w_learning.json only, ledger, WCache, V read-only cache)"
```

---

### Task 6: `w_runner` (1) and the runner world — stage 0, reuse, path, pilot, OC, smoke, the budget gate

**Files:**
- Create: `flymon/brain/w_runner.py` (first part), `tests/brain/w_world.py`
- Test: `tests/brain/test_w_runner.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5; `v_runner` (`build_ctx`, `V_HASHED_FILES`, `kc_values`), `v_rules.reuse`, `r_runner.p_items`, `r_measure.RMeasurer`, `h4_formula.pair_stats`, `e_runner.summary_git`, `h3_store` (`canonical`, `sha256_file`, `git_state`).
- Produces: `ORDER`, `GATES`, `EXIT_REFUSE`, `EXIT_KEY`, `W_PIPELINE_FILES`, `W_HASHED_FILES`, `DECISION_FILES`, `JUDGE_MARKER`, `REREAD_MARKER`, `DONE_MARKER`, `PINNED`, `git_state()`, `pipeline_key()`, `decision_key()`, `decision_pins(doc)`, `refuse(msg, code)`, `BudgetStop`, `git_facts(path, commits)`, `build_ctx(spec, npz)`, `Runner(measure, learner, ctx, spec, summary_path=None, code=None, tcode=None, ucode=None, wcode=None, pipeline=None, archive_root=None)` with `units(rows, where, K, F, edit, k0=0, brains=BRAINS, plastic=True)`, `stage_stage0`, `stage_reuse`, `stage_path`, `stage_pilot`, `stage_oc`, `stage_smoke`, `stage_budget`, `_options`, `_plan`, `_elapsed_h`. `measure(z, smoke) -> RMeasurer`-like; `learner(smoke) -> WMeasurer`-like. `w_world`: `World`, `SPEC`, `MAIN`, `PILOT`, `doc()`, `through(w, last)`, `fake_oc`.

- [ ] **Step 1: Write the world and the failing tests**

```python
"""A scripted W world for the runner tests (no engine, no pool): a fake ctx (V's summary with the reused blocks, V's
git facts, 16 pilot rows, a 249-row main set honouring block set, P's stimuli, V's judgement rows), a FakePool whose
w_learn_job is a deterministic count model (paired noise per probe seed; PAM08 lowers MBON05(X) by `a`, PPL105 lowers
MBON13(X) by `b`, no effect with plasticity off; the plastic weights' "sha" names the DANs applied), run through the
real WMeasurer and WCache under results/w/cache, an oracle measurer whose results are r_fixtures.fake_oracle (testable
per plan) written as cache files, fast fakes for w_oc.run / synthetic_validation, V's own reuse decision patched to
PASS (V's tests own it), and V's gate-② rows / judgement raw computed from the same count model so the path gate
reproduces unless a test breaks it."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from flymon.brain import w_measure as WM
from flymon.brain import w_runner as WR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.r_measure import RMeasurer
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W_SPEC
from flymon.brain.w_store import WCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle
from tests.brain.s_fixtures import stimuli

SPEC = dataclasses.replace(W_SPEC, workers=4)
CODE, TCODE, UCODE = {"key": SPEC.r_shared_key}, {"key": SPEC.t_measure_key_t}, {"key": SPEC.u_measure_key_u}
WCODE, PIPE = {"key": "w" * 64}, {"key": "p" * 64}
ZV = {"A": [16.916666666666668, 12.483878492769072], "P": [80.16666666666667, 29.775432639827233]}


def rows(n_b, n_a, turn0, tag):
    out = []
    for i in range(n_b + n_a):
        ax = "b" if i < n_b else "a"
        out.append(dict(axis=ax, turn=turn0 + i, x=f"{tag}x{i}", y=f"{tag}y{i}", odor_x={f"{tag}G": 1.0 + i},
                        odor_y={f"{tag}H": 1.0 + i}, move_x="M", opp_x=["T"], move_y="M", opp_y=["U"]))
    return out


PILOT = rows(15, 1, 0, "p")
MAIN = [dict(r, c=i) for i, r in enumerate(rows(167, 82, 306, "m"))]
VJ = rows(3, 0, 0, "v")


class Model:
    """The count model: effects[pair] = (a, b); naive offset of MBON13(X) per pair (an unbalanced pair)."""

    def __init__(self):
        self.effects, self.offset, self.c_scale, self.train_s, self.probe_s = {}, {}, 0.5, 0.01, 0.001

    def probe(self, pair, seed, edit, applied, plastic):
        r = np.random.default_rng(int(seed))
        n = r.normal(0, 3, 4)
        a, b = self.effects.get(pair, (30.0, 10.0))
        s = 1.0 if edit == V.lever_edit else self.c_scale
        ax = 30 + self.offset.get(pair, 0.0) + n[0]
        px, ay, py = 60 + n[1], 30 + n[2], 60 + n[3]
        if plastic:
            if "PAM08" in applied:
                px -= a * s
            if "PPL105" in applied:
                ax -= b * s
        pr = lambda A, P: dict(seed=int(seed), A=int(max(0, round(A))), P=int(max(0, round(P))), kc_frac=0.05,  # noqa
                               kc_spikes=10, kc_max_win_hz=10.0, apl_out_per_step=0.1, wall_s=0.0, steps=1400)
        return pr(ax, px), pr(ay, py)

    def job(self, kw, pair):
        edit, plastic = kw["edit"], kw["plastic"]
        lever = edit == V.lever_edit
        stages, applied = [], []
        sha = lambda: hashlib.sha256(f"{pair}|{kw['fly']}|{applied}|{plastic}".encode()).hexdigest()[:16]  # noqa

        def snap(name, da):
            xs, ys = zip(*(self.probe(pair, s, edit, applied, plastic) for s in kw["probe_seeds"]))
            return dict(stage=name, x=list(xs), y=list(ys), w_sha256="w0" if not applied or not plastic else sha(),
                        weights_frac=1.0, weights_frac_A=1.0, weights_frac_P=0.9 if "PAM08" in applied else 1.0,
                        da_integral=da)
        stages.append(snap("pre", None))
        for i, (dan, _trials, _base) in enumerate(kw["phases"]):
            applied.append(dan)
            stages.append(snap(f"S{i + 1}", {"PAM08": 0.1}))
        n_tr = sum(int(p[1]) for p in kw["phases"])
        return dict(edit=edit, csc_sha256=V.sha_combined if lever else V.sha_none, edit_edges=2 if lever else 0,
                    block_edges=dict(V.contrast_declared()["chain_entry"]) if lever else {}, w0_sha256="w0",
                    fly=kw["fly"], plastic=plastic, probe_seeds=list(kw["probe_seeds"]), stages=stages,
                    wall_s=n_tr * self.train_s + len(stages) * self.probe_s, train_s=n_tr * self.train_s,
                    probe_s=2 * len(kw["probe_seeds"]) * len(stages) * self.probe_s)


class FakePool:
    def __init__(self, model, n=4):
        self.model, self.n_workers, self.jobs = model, n, 0
        self.pairs = {}

    def run_jobs(self, fn, kws):
        assert fn is WM.w_learn_job
        self.jobs += len(kws)
        return [self.model.job(kw, self.pairs.get(json.dumps([kw["odor_x"], kw["odor_y"]], sort_keys=True), "?"))
                for kw in kws]


class FakeOracle:
    def __init__(self, world, z, smoke):
        self.w, self.z, self.smoke = world, dict(z), smoke
        self._rm = RMeasurer(None, None, V, Params(), READOUT, self.z, TYPES, 100)
        self.last_wall_s, self.last_jobs = 1.0, 1

    def oracle(self, rs, cond, block, seeds):
        self.w.oracle_calls.append((block, cond.name, len(rs), tuple(self.z["A"])))
        root = Path(SPEC.smoke_cache_dir if self.smoke else SPEC.cache_dir) / "r_oracle"
        root.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rs:
            k = row_key(r)
            ok = self.w.testable.get(k, False)
            res = fake_oracle(ok, ok, False, sha=self.w.oracle_sha, edges=2, edit=cond.edit,
                              n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ins = json.loads(canonical(self._rm.inputs(r, cond, block, seeds)))
            ck = hashlib.sha256(canonical(ins).encode()).hexdigest()
            f = root / f"{ck[:24]}.json"
            f.write_text(json.dumps(dict(key=ck, kind="r_oracle", inputs=ins, result=res)))
            out.append(dict(key=k, result=res, cache_key=ck, cache_file=str(f)))
        return out


def fake_oc(pilot, z, spec, cost, n_boot=None, n_rep=None, n_boot_rep=None, log=None):
    sel = dict(q=0.75, K=8, F=8, cost_h=float(cost(8, 8)))
    alt = dict(q=0.75, K=8, F=9, cost_h=float(cost(8, 9)))
    return dict(selected=None if fake_oc.none else sel, ranking=[] if fake_oc.none else [sel, alt], reachable=True,
                calibration={}, theta=dict(n_pairs=len(pilot)), records=dict(mixed_flies={"g0.0": [0.1] * 5}),
                power_lo=[[[0.5] * 25] * 2] * 3, false_hi=[[[0.01] * 25] * 2] * 3,
                timing=dict(point_s=1.0, boot_s=2.0, records_s=1.0, total_s=4.0), n_pilot=len(pilot))


fake_oc.none = False


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.tmp = tmp_path
        self.model = Model()
        self.testable = {row_key(r): (r["c"] % 3 == 0) for r in MAIN}
        self.oracle_calls, self.judge_commits, self.oracle_sha = [], [], V.sha_combined
        self.v = self.v_doc()
        self.v_git = dict(tracked=True, dirty=False, judge_commits=["a279a56"])
        self.facts = dict(last="a279a56" + "0" * 33, ancestors={c: True for _, c in SPEC.v_commits})
        fake_oc.none = False
        monkeypatch.setattr(WR, "summary_git", lambda p: dict(tracked=True, dirty=False,
                                                              judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(WR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(WR.v_rules, "reuse", lambda *a, **k: dict(outcome="PASS", reasons=[]))
        monkeypatch.setattr(WR.w_oc, "run", fake_oc)
        monkeypatch.setattr(WR.w_oc, "synthetic_validation", lambda spec, z: dict(ok=True, zero_effect=dict(ok=True)))
        self.archive = tmp_path / "archive"
        self.pool = FakePool(self.model)
        self.st = stimuli(V.p)
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100,
                        p_inputs=lambda pspec, smoke_: (dict(c1=0.6), self.st),
                        r_doc=dict, r_git=dict, t_doc=dict, t_git=dict, u_doc=dict, u_git=dict,
                        v_doc=lambda: json.loads(json.dumps(self.v)), v_git=lambda: dict(self.v_git),
                        v_facts=lambda: json.loads(json.dumps(self.facts)), pilot_rows=lambda: list(PILOT),
                        w_set=self.w_set, main_rows=self.main_rows, v_gate2_rows=self.v_gate2_rows, v_jm=self.v_jm,
                        clusters=lambda rs: {row_key(r): "X" for r in rs},
                        c3_params=dict(learn_rate=3e-4, recovery_per_pulse=0.0, kc_kc_scale=0.0))
        for r in PILOT + MAIN + VJ:
            self.pool.pairs[json.dumps([r["odor_x"], r["odor_y"]], sort_keys=True)] = row_key(r)
        for d, (x, y) in V.p.pairs().items():
            self.pool.pairs[json.dumps([self.st[x]["odor"], self.st[y]["odor"]], sort_keys=True)] = f"P{d}"

    # ---- V's summary --------------------------------------------------------------------------------------------
    def v_doc(self) -> dict:
        keys = dict(u_measure_key=SPEC.u_measure_key_u, code_key=SPEC.r_shared_key, t_measure_key=SPEC.t_measure_key_t)
        return dict(z=dict(keys, z_V=ZV, outcome="PASS"), kc_input=dict(keys), set=dict(keys, set={}),
                    judge=dict(keys, band="SELECTED"), even=dict(keys, pairs=[]),
                    **{"jm:L": dict(keys, wall_s=1800.0, jobs=64)})

    # ---- the set callables ---------------------------------------------------------------------------------------
    def js(self):
        keys = [row_key(r) for r in MAIN]
        return dict(rows=MAIN, n_b=167, n_a=82, n=249, first_turn=306, last_turn=MAIN[-1]["turn"],
                    skipped=dict(used=1), clusters_b=[], clusters_a=[], digest_keys="k" * 64, digest_e0_b="b" * 64,
                    digest_e0_a="a" * 64, n_odours=10, all_off_pool=True, keys=keys)

    def w_set(self):
        return self.js()

    def main_rows(self, blk):
        from flymon.brain.w_pairs import check_w_set
        bad = check_w_set(self.js(), blk)
        if bad:
            raise ValueError("; ".join(bad))
        return [dict(r) for r in MAIN]

    # ---- V's raw (computed from the same model) ----------------------------------------------------------------
    def v_gate2_rows(self, items):
        n = V.p.o.n
        out = []
        for i in items:
            kw = dict(edit=i["edit"], plastic=True, fly=i["seed"], probe_seeds=[i["seed"]],
                      phases=[[n.h3.punish_type, int(n.h4.teach_trials), n.train_seed_base]])
            res = self.model.job(kw, f"P{i['direction']}")
            st = {s["stage"]: s for s in res["stages"]}
            out.append(dict(pre=dict(x=st["pre"]["x"][0], y=st["pre"]["y"][0]),
                            post=dict(x=st["S1"]["x"][0], y=st["S1"]["y"][0]), w0_sha256="w0",
                            w_post_sha256=st["S1"]["w_sha256"], weights_frac=1.0, weights_frac_A=1.0,
                            weights_frac_P=st["S1"]["weights_frac_P"], da_integral=st["S1"]["da_integral"],
                            csc_sha256=res["csc_sha256"]))
        return out

    def v_jm(self, name, n):
        edit = V.cond(name).edit
        out = []
        for r in VJ[:n]:
            kw = dict(edit=edit, plastic=True, fly=0, probe_seeds=V.judge_seeds()["report"], phases=[])
            st = self.model.job(kw, row_key(r))["stages"][0]
            res = {"report": {"pre": {"A": [[x["A"], y["A"]] for x, y in zip(st["x"], st["y"])],
                                      "P": [[x["P"], y["P"]] for x, y in zip(st["x"], st["y"])]}}}
            out.append((r, res))
        return out

    # ---- measurers ------------------------------------------------------------------------------------------------
    def measure(self, z, smoke=False):
        return FakeOracle(self, z, smoke)

    def learner(self, smoke=False):
        cache = WCache(SPEC.smoke_cache_dir if smoke else SPEC.cache_dir, WCODE)
        return WM.WMeasurer(self.pool, cache, Params(), READOUT, "MBON05", "PAM08", "PPL105",
                            dict(strength=1.0, settle_ms=800.0, read_ms=600.0, window_ms=200),
                            dict(present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, seed_stride=1000))

    def runner(self, spec=None, code=None, wcode=None):
        return WR.Runner(self.measure, self.learner, self.ctx, spec or SPEC, code=code or CODE, tcode=TCODE,
                         ucode=UCODE, wcode=wcode or WCODE, pipeline=PIPE, archive_root=self.archive)


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through(w, last, smoke_spec=None):
    from flymon.brain.w_spec import smoke
    for s in WR.ORDER[:WR.ORDER.index(last) + 1]:
        r = w.runner(smoke(SPEC) if s == "smoke" else None)
        out = getattr(r, f"stage_{s}")()
        if s in WR.GATES:
            assert out["outcome"] == "PASS", (s, out)
        if s == "smoke":
            assert out["problems"] == [], out["problems"]
    return out
```

```python
"""W's stage chain up to the budget gate (W.9.9 순서 0-5a): stage0 -> reuse -> path -> pilot -> oc -> smoke -> budget;
refusals (missing / later / own block, a stopped gate, uncommitted summary), exit 7 when the reuse condition breaks
after reuse, STOP_REUSE, the path gate's three checks (V gate ②'s punish-arm rows, V's oracle naive counts, the reward
path) with STOP_W_PATH_REPRO and INVALID, the pilot's H6 stop, STOP_OC_UNREACHABLE, smoke on its own seed block and
the budget gate's order (C dropped first, then the alternative design, then STOP_BUDGET)."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain import w_rules
from flymon.brain import w_runner as WR
from flymon.brain.r_pairs import row_key
from flymon.brain.w_spec import smoke
from tests.brain.w_world import SPEC, World, doc, fake_oc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_budget(w):
    through(w, "budget")
    d = doc()
    assert set(d) == set(WR.ORDER[:WR.ORDER.index("budget") + 1]) | {"ledger"}
    assert [e["stage"] for e in d["ledger"]] == list(WR.ORDER[:WR.ORDER.index("budget") + 1])
    assert all(d[b]["w_measure_key"] == "w" * 64 and d[b]["pipeline_key"] == "p" * 64 for b in WR.ORDER[:7])
    assert d["stage0"]["outcome"] == "PASS" and d["stage0"]["tables"]["protocol"]["pulse_ms"] == 400.0
    assert len(d["path"]["checks"]) == 8 + 6 + 1 and all(c["diffs"] == [] for c in d["path"]["checks"])
    assert d["pilot"]["n_units"] == 16 * 8 * 3 and d["pilot"]["exploratory"]["label"] == "탐색"
    assert d["oc"]["selected"]["F"] == 8 and d["budget"]["plan"]["with_c"] is True
    assert "set" not in d and w.oracle_calls == [("smoke", "L", 1, tuple(d["reuse"]["z_V"]["A"]))]


def test_refusals_and_order(w):
    with pytest.raises(SystemExit) as e:
        w.runner().stage_reuse()
    assert e.value.code == 2
    through(w, "reuse")
    with pytest.raises(SystemExit):
        w.runner().stage_reuse()
    w.facts["ancestors"]["928eaad"] = False
    with pytest.raises(SystemExit) as e:
        w.runner().stage_path()
    assert e.value.code == 7


def test_reuse_stop(w):
    w.v["judge"]["band"] = "B_Tb"
    through(w, "stage0")
    out = w.runner().stage_reuse()
    assert out["outcome"] == w_rules.STOP_REUSE and "V 판정 B_Tb" in out["sentence"]
    with pytest.raises(SystemExit):
        w.runner().stage_path()


def test_path_stop_on_v_rows_and_naive(w):
    through(w, "reuse")
    real = w.v_gate2_rows
    w.ctx["v_gate2_rows"] = lambda items: [dict(r, w_post_sha256="x") if i == 0 else r
                                           for i, r in enumerate(real(items))]
    out = w.runner().stage_path()
    assert out["outcome"] == w_rules.STOP_W_PATH_REPRO
    assert out["sentence"].startswith("W 학습 경로가 V 관문 ② 처벌 팔 행(L_V r1 25400000)을 재현하지 못했다(")


def test_path_naive_mismatch_and_invalid_edit(w, monkeypatch):
    through(w, "reuse")

    def jm(name, n):
        out = World.v_jm(w, name, n)
        if name == "C":
            out[1][1]["report"]["pre"]["A"][0][0] += 1
        return out
    w.ctx["v_jm"] = jm
    out = w.runner().stage_path()
    assert out["outcome"] == w_rules.STOP_W_PATH_REPRO and [c["ref"] for c in out["failed"]] == [
        f"V 오라클 순진 카운트(C {row_key(World.v_jm(w, 'C', 3)[1][0])})"]
    real = w.model.job
    monkeypatch.setattr(w.model, "job", lambda kw, pair: dict(real(kw, pair), edit_edges=3))
    doc_ = doc()
    doc_.pop("path")
    Path(SPEC.summary).write_text(json.dumps(doc_))
    out = w.runner().stage_path()
    assert out["outcome"] == w_rules.INVALID and "edit_edges 3" in out["reasons"][0]


def test_pilot_no_effect_stops(w):
    for r in w.ctx["pilot_rows"]():
        w.model.effects[row_key(r)] = (30.0, 0.0)                    # reward learns, punishment does nothing
    through(w, "path")
    out = w.runner().stage_pilot()
    assert out["outcome"] == w_rules.STOP_PILOT_NO_EFFECT and out["sentence"].startswith(
        "파일럿에서 조합 지렛대의 F.2 학습 효과가 (i) 보상 연합 d′ ≥ 0.5 쌍 비율 1.000·처벌 연합 d′ ≤ −0.5 쌍 비율 0.000")
    with pytest.raises(SystemExit):
        w.runner().stage_oc()


def test_oc_unreachable(w):
    through(w, "pilot")
    fake_oc.none = True
    out = w.runner().stage_oc()
    assert out["outcome"] == w_rules.STOP_OC_UNREACHABLE
    assert out["sentence"].startswith("파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다(")


def test_budget_drops_c_then_alt_then_stops(w):
    through(w, "smoke")
    d = doc()
    r = w.runner()
    full = r._options(d, 249, None)
    assert [(o["design"]["F"], o["with_c"]) for o in full] == [(8, True), (8, False), (9, False)]
    elapsed = r._elapsed_h(d)
    assert elapsed == sum(e["wall_s"] for e in d["ledger"]) / 3600 and elapsed < SPEC.budget_h
    tight = dataclasses.replace(SPEC, budget_h=elapsed + (full[0]["total_h"] + full[1]["total_h"]) / 2)
    out = w_rules.budget(elapsed, full, tight)
    assert out["outcome"] == "PASS" and out["plan"]["with_c"] is False
    out = w_rules.budget(elapsed, full, dataclasses.replace(SPEC, budget_h=elapsed + full[1]["total_h"] / 2))
    assert out["outcome"] == w_rules.STOP_BUDGET and out["sentence"].startswith("남은 추정 비용 누적 ")


def test_smoke_spec_only_for_smoke(w):
    through(w, "oc")
    with pytest.raises(SystemExit):
        w.runner().stage_smoke()
    out = w.runner(smoke(SPEC)).stage_smoke()
    assert out["problems"] == [] and out["cost"]["total_h"] > 0 and out["design"]["K"] == 8
    assert all(42_100_000 <= s < 42_200_000 for s in doc()["smoke"]["seeds"]["probe"]["0"])
    assert ("smoke", "L", 1, tuple(doc()["reuse"]["z_V"]["A"])) in w.oracle_calls
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_runner.py -q`
Expected: FAIL (ImportError: `flymon.brain.w_runner`).

- [ ] **Step 3: Write `w_runner` (first part)**

Create `flymon/brain/w_runner.py` with exactly this content (Task 8 appends the rest of the class):

```python
"""Spec W's stage chain (W.9.9's order; plan Readings):
stage0 -> reuse -> path -> pilot -> oc -> smoke -> budget -> set -> oracle -> estimate -> naive -> gates -> learn ->
band -> records -> seal -> judge (and after judge only: recompute / invalid_run), one block each in
results/summary/w_learning.json, written only through w_store, plus a running budget ledger (W.9.6 F). Every stage
refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists, its own block exists,
an earlier gate did not pass, the summary has uncommitted changes or a hashed W file is dirty. Every stage after
`reuse` re-checks the reuse condition (V's blocks, R / T / U keys) and refuses with SystemExit 7 when it broke; learn,
seal and judge also re-check that every earlier block carries the current keys (W.8, exit 7).
The main set is used from block oracle on (W.9.9: 순서 7); every stage before it runs on V's raw data, the pilot pairs
or synthetic data only. learn / band / records hold raw manifests and no statistic; the verdict is read once at judge
(the BAND re-measure is measured for every gate pair before the seal, plan Reading 13)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import v_rules, w_oc, w_records, w_rules, w_store, w_verdict
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .h4_formula import pair_stats
from .r_measure import R_MEASURE_FILES
from .r_pairs import row_key
from .r_runner import p_items
from .t_measure import T_MEASURE_FILES
from .u_measure import U_MEASURE_FILES
from .v_runner import V_HASHED_FILES, kc_values
from .v_spec import SPEC as V_SPEC
from .w_measure import W_MEASURE_FILES
from .w_spec import BRAINS

ORDER = ("stage0", "reuse", "path", "pilot", "oc", "smoke", "budget", "set", "oracle", "estimate", "naive", "gates",
         "learn", "band", "records", "seal", "judge")
GATES = ("stage0", "reuse", "path", "pilot", "oc", "budget", "set", "oracle", "estimate", "naive", "gates", "learn",
         "band")
EXIT_REFUSE, EXIT_KEY = 2, 7
W_PIPELINE_FILES = ("flymon/brain/w_spec.py", "flymon/brain/w_pairs.py", "flymon/brain/w_store.py",
                    "flymon/brain/w_verdict.py", "flymon/brain/w_oc.py", "flymon/brain/w_records.py",
                    "flymon/brain/w_rules.py", "flymon/brain/w_runner.py", "scripts/run_w.py")
W_HASHED_FILES = tuple(dict.fromkeys(V_HASHED_FILES + W_MEASURE_FILES + W_PIPELINE_FILES
                                     + ("results/summary/v_lever.json",)))
DECISION_FILES = tuple(f for f in W_HASHED_FILES if f not in R_MEASURE_FILES and f not in T_MEASURE_FILES
                       and f not in U_MEASURE_FILES and f not in W_MEASURE_FILES)
JUDGE_MARKER = "results/w/judge_read.json"
REREAD_MARKER = "results/w/judge_reread.json"
DONE_MARKER = "results/w/judge_done.json"
PINNED = ("oc", "budget", "gates")


def git_state() -> dict:
    return _h3_git_state(files=W_HASHED_FILES)


def _files_key(files) -> dict:
    hashed = {f: sha256_file(_ROOT / f) for f in files}
    return dict(key=hashlib.sha256(canonical(hashed).encode()).hexdigest(), files=hashed)


def pipeline_key() -> dict:
    return _files_key(W_PIPELINE_FILES)


def decision_key() -> dict:
    return _files_key(DECISION_FILES)


def _sha(obj) -> str | None:
    return None if obj is None else hashlib.sha256(canonical(obj).encode()).hexdigest()


def decision_pins(doc: dict) -> dict:
    """W.3 10: the design (block oc's selection and block budget's plan), the gate pairs and z_V."""
    pins = {f"{b}_sha256": _sha(doc.get(b)) for b in PINNED}
    pins["z_V"] = (doc.get("reuse") or {}).get("z_V")
    return pins


def refuse(msg: str, code: int = EXIT_REFUSE):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _z(z: dict) -> dict:
    return {k: (float(v[0]), float(v[1])) for k, v in z.items()}


class BudgetStop(Exception):
    """Raised between rounds when the ledger's estimate passes the budget (W.9.8 H3)."""


def git_facts(path: str, commits) -> dict:
    """The last commit touching V's summary and which declared V commits are in HEAD's history."""
    def run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True)
    last = run("log", "-1", "--format=%H", "--", path).stdout.strip() or None
    anc = {c: run("merge-base", "--is-ancestor", c, "HEAD").returncode == 0 for c in commits}
    return dict(last=last, ancestors=anc)


def build_ctx(spec, npz: str) -> dict:
    """V's context (v_runner.build_ctx on V's spec: C3, block h4's z, the even rows, P's stimuli, R's / T's / U's
    summaries, …) plus W's: V's summary and git facts, V's KC values, the pilot rows, the main set, V's gate-② arm
    rows (read only, by content key) and V's judgement raw (by V's manifest)."""
    from ..agent.config import load_c3_config
    from . import w_pairs
    from .circuits import Populations
    from .connectome import Connectome
    from .r_measure import RMeasurer
    from .u_measure import u_measure_key
    from .v_runner import build_ctx as v_build_ctx
    from .v_store import load_manifest
    ctx = v_build_ctx(V_SPEC, npz)
    cfg = load_c3_config(V_SPEC.m0d_summary)
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    enc, params = ctx["enc"], cfg.params
    v_doc = lambda: json.loads(Path(spec.v_summary).read_text())  # noqa: E731
    vread = w_store.VReadCache(V_SPEC.cache_dir, u_measure_key(npz))
    v_m = RMeasurer(None, vread, V_SPEC, params, cfg.readout, cfg.z, ctx["types"], len(pops.kc))

    def v_gate2_rows(items):
        return v_m.arms(items, ctx["readout"], V_SPEC.p.o.n.h3.punish_type, "gate2", V_SPEC.p.o.n)

    def v_jm(name: str, n: int):
        d = v_doc()
        rows = ctx["judgement_rows"](kc_values(d), d["set"]["set"])[:n]
        got, bad = load_manifest(d[f"jm:{name}"]["manifest"])
        if bad:
            raise ValueError("; ".join(bad[:3]))
        res = {g["key"]: g["result"] for g in got}
        return [(r, res[row_key(r)]) for r in rows]

    ctx.update(
        v_doc=v_doc, v_git=lambda: summary_git(spec.v_summary),
        v_facts=lambda: git_facts(spec.v_summary, [c for _, c in spec.v_commits]),
        pilot_rows=lambda: w_pairs.pilot_rows(ctx["even_rows"], v_doc()["even"]["pairs"], spec),
        w_set=lambda: w_pairs.w_set(pops, enc, params, kc_values(v_doc()), v_doc()["set"]["set"], spec),
        main_rows=lambda blk: w_pairs.main_rows(pops, rc, enc, params, kc_values(v_doc()), v_doc()["set"]["set"],
                                                blk, spec),
        v_gate2_rows=v_gate2_rows, v_jm=v_jm, clusters=w_pairs.cluster_labels,
        c3_params=dict(learn_rate=params.learn_rate, kc_trace_ms=params.kc_trace_ms, da_trace_ms=params.da_trace_ms,
                       da_baseline_ms=params.da_baseline_ms, kc_trace_scale=params.kc_trace_scale,
                       da_trace_scale=params.da_trace_scale, min_weight_frac=params.min_weight_frac,
                       recovery_per_pulse=params.recovery_per_pulse, kc_kc_scale=params.kc_kc_scale,
                       dan_drive_mv=params.dan_drive_mv, core_frac=params.core_frac))
    return ctx


class Runner:
    def __init__(self, measure, learner, ctx: dict, spec, summary_path=None, code: dict | None = None,
                 tcode: dict | None = None, ucode: dict | None = None, wcode: dict | None = None,
                 pipeline: dict | None = None, archive_root=None):
        """measure(z) -> an RMeasurer (or its interface) over U's job copies behind W's cache; learner -> a
        WMeasurer (or its interface). ucode = U's measurement key (= V's); wcode = the W measurement key."""
        self.measure, self.wm, self.ctx, self.spec = measure, learner, ctx, spec
        self.summary_path = str(summary_path or spec.summary)
        self.code_key = (code or {}).get("key")
        self.t_measure_key = (tcode or {}).get("key")
        self.u_measure_key = (ucode or {}).get("key")
        self.w_measure_key = (wcode or {}).get("key")
        self.pipeline_key = (pipeline or {}).get("key")
        self.archive_root = Path(os.path.expanduser(str(archive_root or spec.archive_root)))
        self.plist = [ctx["params"]]
        self._ms = {}

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return w_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed W files are dirty: {gs['dirty_hashed']}")

    def _reuse_dec(self) -> dict:
        c, sp = self.ctx, self.spec
        v_dec = v_rules.reuse(c["r_doc"](), c["r_git"](), c["t_doc"](), c["t_git"](), c["u_doc"](), c["u_git"](),
                              self.code_key, self.t_measure_key, self.u_measure_key, V_SPEC)
        f = c["v_facts"]()
        return w_rules.reuse(v_dec, c["v_doc"](), c["v_git"](), f["ancestors"], f["last"], self.u_measure_key,
                             self.code_key, self.t_measure_key, sp)

    def _require(self, stage: str, allow_own: bool = False) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; W never rewrites an earlier block")
        if stage in doc and not allow_own:
            refuse(f"stage {stage}: block {stage} exists; W never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != w_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — W stops there (W.3)")
        if i > ORDER.index("reuse"):
            r = self._reuse_dec()
            if r["outcome"] != w_rules.PASS:
                refuse(f"stage {stage}: the reuse condition broke ({'; '.join(r['reasons'])})", EXIT_KEY)
        if i > ORDER.index("smoke") and doc["smoke"].get("problems"):
            refuse(f"stage {stage}: smoke found problems {doc['smoke']['problems'][:2]}")
        return doc

    def _keys_chain(self, doc: dict, stage: str) -> None:
        """W.8: the W measurement key (and R's, T's, U's) every earlier block carries is the current one (exit 7)."""
        if not (self.code_key and self.t_measure_key and self.u_measure_key and self.w_measure_key):
            refuse("a measurement key cannot be verified (W.8)", EXIT_KEY)
        for b in ORDER[:ORDER.index(stage)]:
            got = tuple(doc[b].get(k) for k in ("code_key", "t_measure_key", "u_measure_key", "w_measure_key"))
            if got != (self.code_key, self.t_measure_key, self.u_measure_key, self.w_measure_key):
                refuse(f"block {b}'s keys {got} are not the current ones (W.8)", EXIT_KEY)
            d = (doc[b].get("git") or {}).get("dirty_hashed")
            if d is None or d:
                refuse(f"block {b} was written with dirty hashed files (or no git record): {d}")

    def _stamp(self, stage: str, body: dict) -> dict:
        return dict(body, stage=stage, code_key=self.code_key, t_measure_key=self.t_measure_key,
                    u_measure_key=self.u_measure_key, w_measure_key=self.w_measure_key,
                    pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())

    def _write(self, stage: str, body: dict, wall_s: float = 0.0) -> dict:
        block = self._stamp(stage, body)
        ledger = dict(stage=stage, wall_s=float(wall_s), at=block["written_at"])
        w_store.write_summary_block(self.summary_path, stage, block, self.plist, ledger)
        return block

    def _elapsed_h(self, doc: dict) -> float:
        return w_records.elapsed_h(doc.get("ledger", []))

    # ---- progress (resumable wall clock of a pool stage) ------------------------------------------------------------
    def _prog_path(self, stage: str) -> str:
        return f"{self.spec.progress_dir}/{stage}.json"

    def _prog(self, stage: str) -> float:
        p = Path(self._prog_path(stage))
        return float(json.loads(p.read_text())["wall_s"]) if p.exists() else 0.0

    def _prog_add(self, stage: str, s: float) -> float:
        tot = self._prog(stage) + float(s)
        w_store.write_json(self._prog_path(stage), dict(wall_s=tot, at=_now()), self.plist)
        return tot

    # ---- measurers ---------------------------------------------------------------------------------------------------
    def _m(self, z: dict, smoke: bool = False):
        k = (tuple(sorted(_z(z).items())), smoke)
        if k not in self._ms:
            self._ms[k] = self.measure(_z(z), smoke)
        return self._ms[k]

    def _wm(self, smoke: bool = False):
        return self.wm(smoke)

    def _z_v(self, doc) -> dict:
        return _z(doc["reuse"]["z_V"])

    # ---- units -------------------------------------------------------------------------------------------------------
    def _unit(self, r, idx, fly, brain, edit, phases, seeds, plastic=True) -> dict:
        return dict(pair=row_key(r), idx=idx, fly=int(fly), brain=brain, edit=edit, odor_x=dict(r["odor_x"]),
                    odor_y=dict(r["odor_y"]), phases=phases, probe_seeds=[int(s) for s in seeds],
                    plastic=bool(plastic))

    def units(self, rows: list, where: str, K: int, F: int, edit: str, k0: int = 0, brains=BRAINS,
              plastic=True) -> list:
        """where = "main" (row["c"]), "pilot" (position j) or "smoke"; probes k0..K−1; F flies; brains."""
        sp, out = self.spec, []
        for j, r in enumerate(rows):
            for f in range(F):
                if where == "main":
                    c = r["c"]
                    seeds, b0, b1 = sp.probe_seeds(c, f, K, k0), sp.train_base(c, 0), sp.train_base(c, 1)
                elif where == "pilot":
                    c = j
                    seeds, b0, b1 = sp.pilot_probe_seeds(j, f, K, k0), sp.pilot_train_base(j, 0), \
                        sp.pilot_train_base(j, 1)
                else:
                    c = "smoke"
                    seeds, b0, b1 = sp.smoke_probe_seeds(f, K, k0), sp.smoke_train_base(0), sp.smoke_train_base(1)
                for b in brains:
                    ph = [] if b == "naive" else sp.phases(b if b in BRAINS else "R", b0, b1)
                    out.append(self._unit(r, c, f, b, edit, ph, seeds, plastic))
        return out

    def _declared(self, edit: str, seeds_by_fly: dict) -> dict:
        sp = V_SPEC
        lever = edit == sp.lever_edit
        return dict(edit=edit, csc_sha256=sp.sha_combined if lever else sp.sha_none,
                    edit_edges=sp.lever_edges if lever else 0,
                    block_edges=dict(sp.contrast_declared()["chain_entry"]) if lever else {},
                    probe_seeds=seeds_by_fly)

    @staticmethod
    def _by_pair(rows: list) -> dict:
        out = {}
        for r in rows:
            out.setdefault(r["unit"]["pair"], []).append(r)
        return out

    @staticmethod
    def _manifest(rows: list) -> list:
        return [dict(key=f"{r['unit']['pair']}|{r['unit']['fly']}|{r['unit']['brain']}", cache_file=r["cache_file"],
                     cache_key=r["cache_key"], sha256=sha256_file(r["cache_file"])) for r in rows]

    # ================================================================ 0: the verdict code, w_oc, tables (W.9.9 순서 0)
    def stage_stage0(self, n_boot_timing: int = 4) -> dict:
        """No pool, no data: the decision code's hashes, the protocol / seed / design tables, w_oc's synthetic
        validation (W.9.9 P2-11) and the OC's compute time (a synthetic pilot; the bootstrap timed on n_boot_timing
        draws and scaled to boot_draws) — PASS iff every synthetic check passed."""
        self._require("stage0")
        sp = self.spec
        z = sp.z_v()
        t0 = time.perf_counter()
        syn = w_oc.synthetic_validation(sp, z)
        rng = np.random.default_rng(np.random.SeedSequence([sp.oc_seed, w_oc.TAG_SYNTH, 1]))
        pil = w_oc.synthetic_pilot(rng, base=(35.0, 90.0), sd=6.0, corr=0.7, learn=(40.0, 15.0), drift_ax=3.0)
        doc = w_oc.run(pil, z, sp, lambda K, F: K * F, n_boot=n_boot_timing)
        tm = doc["timing"]
        est = tm["point_s"] + tm["records_s"] + tm["boot_s"] * sp.boot_draws / max(n_boot_timing, 1)
        files = {f: sha256_file(_ROOT / f) for f in ("flymon/brain/w_verdict.py", "flymon/brain/w_oc.py",
                                                    "flymon/brain/w_spec.py", "flymon/brain/w_measure.py")}
        tables = dict(protocol=dict(trials=sp.trials, pulse_ms=sp.pulse_ms, train_settle_ms=sp.train_settle_ms,
                                    gap_ms=sp.gap_ms, strength=sp.strength, reward=sp.reward_dan,
                                    punish=sp.punish_dan, probe=dict(settle_ms=V_SPEC.settle_ms,
                                                                     read_ms=V_SPEC.read_ms,
                                                                     window_ms=V_SPEC.window_ms),
                                    c3=self.ctx["c3_params"]),
                      designs=dict(q=list(sp.q_grid), K=list(sp.k_grid), F=[sp.f_min, sp.f_max], k_cap=sp.k_cap),
                      seeds=dict(probe="26_000_000 + c·4_000 + f·100 + k", train="28_000_000 + c·40_000 + f·1_000 + t",
                                 pilot_probe="40_000_000 + j·4_000 + f·100 + k",
                                 pilot_train="41_000_000 + j·40_000 + f·1_000 + t", smoke="42_100_000-42_199_999",
                                 oc=sp.oc_seed, oracle=sp.oracle_seeds()),
                      oc=dict(reps=sp.oc_reps, cal_reps=sp.cal_reps, cal_tol=sp.cal_tol, cal_iter=sp.cal_iter,
                              boot_draws=sp.boot_draws, boot_reps=sp.boot_reps, boot_level=sp.boot_level,
                              cluster_grid=list(sp.cluster_grid), noise="pilot residual vectors, resampled",
                              cal_floor_rule=sp.cal_floor_rule))
        body = dict(outcome=w_rules.PASS if syn["ok"] else w_rules.INVALID,
                    reasons=[] if syn["ok"] else [k for k, v in syn.items() if isinstance(v, dict) and not v["ok"]],
                    synthetic=syn, oc_timing=tm, oc_compute_estimate_s=float(est), decision_files=files, tables=tables,
                    note="W.9.9 순서 0: 판정 코드·w_oc·보정·수치 표·합성 검증, 파일럿 전에 커밋.")
        return self._write("stage0", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 1: reuse (W.3 1)
    def stage_reuse(self) -> dict:
        self._require("reuse")
        dec = self._reuse_dec()
        v = self.ctx["v_doc"]()
        body = dict(dec, z_V=(v.get("z") or {}).get("z_V"), v_judge_band=(v.get("judge") or {}).get("band"),
                    u_measure_key_u=self.spec.u_measure_key_u, v_commits=dict(self.spec.v_commits))
        return self._write("reuse", body)

    # ================================================================ 2: the path gate (W.3 2, W.9.6 P2-13, W.9.9 P2-9)
    def stage_path(self) -> dict:
        doc = self._require("path")
        sp, ctx = self.spec, self.ctx
        t0 = time.perf_counter()
        wm = self._wm()
        checks, raw = [], {}
        # (i) V gate ②'s punish arm, L_V and C, both directions, the first repro_p_count seeds
        _c1, st = ctx["p_inputs"](V_SPEC.p, False)
        seeds = list(V_SPEC.p.seeds[:sp.repro_p_count])
        items = [i for i in p_items(V_SPEC.p, st, V_SPEC.lever_edit, seeds) + p_items(V_SPEC.p_c, st, V_SPEC.no_edit,
                                                                                      seeds)
                 if i["arm"] == sp.repro_p_arm]
        try:
            v_rows = ctx["v_gate2_rows"](items)
        except (RuntimeError, ValueError, AttributeError) as e:
            refuse(f"V's gate-② rows cannot be read: {e}")
        n = V_SPEC.p.o.n
        h4 = n.h4
        p_win = dict(strength=n.h3.strength, settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms,
                     window_ms=int(h4.kc_window_ms))
        p_tim = dict(present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms, train_settle_ms=h4.teach_window.settle_ms,
                     seed_stride=n.train_seed_stride)
        units = [dict(pair=f"P|{i['direction']}|{i['seed']}", idx="p", fly=i["seed"], brain="P", edit=i["edit"],
                      odor_x=i["odor_x"], odor_y=i["odor_y"], phases=[[n.h3.punish_type, int(h4.teach_trials),
                                                                       n.train_seed_base]],
                      probe_seeds=[i["seed"]], plastic=True) for i in items]
        got = wm.run(units, **p_win, **p_tim)
        raw["p"] = got
        for i, v, w in zip(items, v_rows, got):
            tag = f"{'L_V' if i['edit'] == V_SPEC.lever_edit else 'C'} {i['direction']} {i['seed']}"
            checks.append(dict(ref=f"V 관문 ② 처벌 팔 행({tag})", diffs=w_records.p_repro_diffs(w, v, tag),
                               invalid=self._job_invalid(w, i["edit"], tag)))
        # (ii) the oracle's naive probe: V's judgement raw report.pre, rows 0..n−1, L_V and C (report seeds)
        try:
            pairs = {nm: ctx["v_jm"](nm, sp.repro_naive_rows) for nm in ("L", "C")}
        except (KeyError, ValueError) as e:
            refuse(f"V's judgement raw cannot be read: {e}")
        rep = V_SPEC.judge_seeds()["report"]
        nunits, refs = [], []
        for nm, lst in pairs.items():
            edit = V_SPEC.cond(nm).edit
            for r, res in lst:
                nunits.append(dict(pair=row_key(r), idx="v", fly=0, brain="naive", edit=edit, odor_x=r["odor_x"],
                                   odor_y=r["odor_y"], phases=[], probe_seeds=rep, plastic=True))
                refs.append((nm, r, res))
        got2 = wm.run(nunits)
        raw["naive"] = got2
        for (nm, r, res), w in zip(refs, got2):
            tag = f"{nm} {row_key(r)}"
            checks.append(dict(ref=f"V 오라클 순진 카운트({tag})", diffs=w_records.naive_repro_diffs(w, res, tag),
                               invalid=self._job_invalid(w, V_SPEC.cond(nm).edit, tag)))
        # (iii) the reward path (W.9.9 P2-9): pilot pair 0, fly 0, R's reward phase only, K = pilot probes
        pr = ctx["pilot_rows"]()[0]
        u = self.units([pr], "pilot", sp.pilot_probes, 1, V_SPEC.lever_edit, brains=("R",))[0]
        u = dict(u, phases=u["phases"][:1], brain="reward_check")
        w3 = wm.run([u])[0]
        raw["reward"] = w3
        checks.append(dict(ref="보상 경로 양성 검사", diffs=w_records.reward_check(w3),
                           invalid=self._job_invalid(w3, V_SPEC.lever_edit, "reward")))
        dec = w_rules.path(checks)
        p = w_store.write_json(sp.path_detail, dict(w_measure_key=self.w_measure_key, rows=raw), self.plist)
        body = dict(dec, checks=checks, detail_path=sp.path_detail, detail_sha256=sha256_file(p),
                    n_p=len(units), n_naive=len(nunits), z_V=doc["reuse"]["z_V"])
        return self._write("path", body, wall_s=time.perf_counter() - t0)

    def _job_invalid(self, res: dict, edit: str, tag: str) -> list:
        d = self._declared(edit, {})
        return [f"{tag}: {k} {res.get(k)!r} ≠ {d[k]!r}" for k in ("edit", "csc_sha256", "edit_edges", "block_edges")
                if res.get(k) != d[k]]

    # ================================================================ 3 / 3a: the pilot (W.3 3, W.9.4, W.9.8 H6)
    def _pilot_data(self, rows: list, got: list, K: int, F: int) -> tuple:
        by = self._by_pair(got)
        pairs, mach = {}, []
        for j, r in enumerate(rows):
            k = row_key(r)
            seeds = {f: self.spec.pilot_probe_seeds(j, f, K) for f in range(F)}
            mach += [f"{k}: {m}" for m in w_records.machine_reasons(by[k], list(range(F)),
                                                                  self._declared(V_SPEC.lever_edit, seeds))]
            pairs[k] = w_records.pair_data(by[k], list(range(F)))
        return pairs, mach

    def stage_pilot(self) -> dict:
        doc = self._require("pilot")
        sp = self.spec
        rows = self.ctx["pilot_rows"]()
        K, F = sp.pilot_probes, sp.pilot_flies
        units = self.units(rows, "pilot", K, F, V_SPEC.lever_edit)
        t0 = time.perf_counter()
        prev = self._prog("pilot")
        wm = self._wm()

        def tick(done, todo):
            self._prog_add("pilot", time.perf_counter() - tick.t)
            tick.t = time.perf_counter()
        tick.t = time.perf_counter()
        got = wm.learn(units, "pilot", check=tick)
        wall = prev + (time.perf_counter() - t0)
        pairs, mach = self._pilot_data(rows, got, K, F)
        walls = dict(job_s_median=float(np.median([g["result"]["wall_s"] for g in got])),
                     train_s_median=float(np.median([g["result"]["train_s"] for g in got])),
                     probe_s_median=float(np.median([g["result"]["probe_s"] for g in got])))
        rec = w_records.pilot_record(pairs, self._z_v(doc), walls, sp)
        dec = w_rules.pilot(rec, mach, sp)
        exploratory = w_verdict.judge({k: (d, None) for k, d in pairs.items()}, mach, self._z_v(doc), 0.75, F, K, sp)
        body = dict(dec, record=rec, exploratory=dict(verdict=exploratory["verdict"], label="탐색",
                                                    pairs={k: v["status"] for k, v in exploratory["pairs"].items()}),
                    pairs=[row_key(r) for r in rows], n_units=len(units), manifest=self._manifest(got))
        return self._write("pilot", body, wall_s=wall)

    # ================================================================ 4: the OC and the design (W.9.3, H1, W.9.9)
    def _pilot_back(self, doc: dict) -> tuple:
        sp = self.spec
        rows = self.ctx["pilot_rows"]()
        got, bad = w_store.load_manifest(doc["pilot"]["manifest"])
        if bad:
            refuse(f"pilot raw changed since block pilot: {bad[:3]}")
        by = {}
        for g in got:
            pair, fly, brain = g["key"].rsplit("|", 2)
            by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
        pairs = {row_key(r): w_records.pair_data(by[row_key(r)], list(range(sp.pilot_flies))) for r in rows}
        return pairs, got

    def _costs(self, job_results: list, oracle_round_s: float) -> dict:
        return w_records.unit_costs(job_results, 2 * self.spec.trials, oracle_round_s, self.spec.workers)

    def _v_oracle_round_s(self) -> float:
        """V's jm:L (the same oracle job on L_V): its wall clock per worker round."""
        b = self.ctx["v_doc"]()["jm:L"]
        return float(b["wall_s"]) / max(1, -(-int(b["jobs"]) // int(V_SPEC.workers)))

    def stage_oc(self) -> dict:
        doc = self._require("oc")
        sp = self.spec
        pairs, got = self._pilot_back(doc)
        costs = self._costs([g["result"] for g in got], self._v_oracle_round_s())
        n_set = sp.n_b_expected + sp.n_a_expected

        def cost(K, F):
            return w_records.design_cost(costs, K, F, sp, n_set)["total_h"]
        t0 = time.perf_counter()
        oc = w_oc.run(list(pairs.values()), self._z_v(doc), sp, cost,
                      log=lambda m: print(m, file=sys.stderr))
        p = w_store.write_json(sp.oc_detail, oc, self.plist)
        dec = w_rules.oc(oc, sp)
        body = dict(dec, selected=oc["selected"], ranking=oc["ranking"], reachable=oc["reachable"],
                    calibration=oc["calibration"], theta=oc["theta"], records=oc.get("records"), costs=costs,
                    detail_path=sp.oc_detail, detail_sha256=sha256_file(p), timing=oc["timing"],
                    note="작동 특성은 파일럿 잡음 모형 조건부이며 Q → … → W 전체 절차의 오선택률이 아니다. 관문 쌍은 "
                         "오라클 순진·시험 가능으로 고른 조건부 표본이다. 파일럿은 순진 불균형 쌍 위주다.")
        return self._write("oc", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 5: smoke and the cost ledger (W.9.9 순서 5)
    def stage_smoke(self) -> dict:
        doc = self._require("smoke")
        sp = self.spec
        if not sp.smoke:
            refuse("smoke needs the smoke spec (42_1xx_xxx)")
        d = doc["oc"]["selected"]
        K = int(d["K"])
        rows = self.ctx["pilot_rows"]()[:1]
        t0 = time.perf_counter()
        wm = self._wm(smoke=True)
        lv = V_SPEC.lever_edit
        learn = wm.learn(self.units(rows, "smoke", K, sp.smoke_flies, lv), "smoke")
        band = wm.learn(self.units(rows, "smoke", 2 * K, sp.smoke_flies, lv, k0=K), "smoke_band")
        naive = wm.learn(self.units(rows, "smoke", K, sp.smoke_flies, lv, brains=("naive",)), "smoke_naive")
        z_v = self._z_v(doc)
        m = self._m(z_v, smoke=True)
        t1 = time.perf_counter()
        orc = m.oracle(rows, V_SPEC.cond("L"), "smoke", sp.oracle_seeds())
        oracle_s = time.perf_counter() - t1
        flies = list(range(sp.smoke_flies))
        seeds = {f: sp.smoke_probe_seeds(f, K) for f in flies}
        nv = {g["unit"]["fly"]: w_records.counts(g["result"]["stages"][0]) for g in naive}
        problems = w_records.machine_reasons(learn, flies, self._declared(lv, seeds), naive=nv, band=band)
        d_k = w_records.pair_data(learn, flies)
        if bool(w_verdict.rn1_mismatch({s: np.asarray(v)[None] for s, v in d_k.items()})[0]):
            problems.append("RN1 ≠ R1")
        q = orc[0]["result"].get("q", {})
        if q.get("edit") != lv or q.get("csc_sha256") != V_SPEC.sha_combined or q.get("edit_edges") != 2:
            problems.append(f"oracle edit {q.get('edit')} / CSC {q.get('csc_sha256')} / edges {q.get('edit_edges')}")
        if list(m.z.items()) != list(z_v.items()):
            problems.append(f"oracle z {m.z} ≠ z_V {z_v}")
        costs = self._costs([g["result"] for g in learn], oracle_s)
        cost = w_records.design_cost(costs, K, int(d["F"]), sp, sp.n_b_expected + sp.n_a_expected)
        body = dict(problems=problems, design=d, costs=costs, cost=cost, oracle_wall_s=oracle_s,
                    seeds=dict(probe=seeds, oracle=sp.oracle_seeds()), pair=row_key(rows[0]))
        return self._write("smoke", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 5a: the worst-case budget gate (W.9.9 P1-5)
    def _options(self, doc: dict, n_set: int, n_naive: int | None, with_c_first: bool = True) -> list:
        sp, costs = self.spec, doc["smoke"]["costs"]
        sel = doc["oc"]["selected"]
        out = []
        if with_c_first:
            out.append(dict(design=sel, with_c=True, **w_records.design_cost(costs, sel["K"], sel["F"], sp, n_set,
                                                                             n_naive, True)))
        out.append(dict(design=sel, with_c=False, **w_records.design_cost(costs, sel["K"], sel["F"], sp, n_set,
                                                                          n_naive, False)))
        for alt in doc["oc"]["ranking"]:
            if (alt["q"], alt["K"], alt["F"]) != (sel["q"], sel["K"], sel["F"]):
                out.append(dict(design=alt, with_c=False, **w_records.design_cost(costs, alt["K"], alt["F"], sp, n_set,
                                                                                  n_naive, False)))
        return out

    def stage_budget(self) -> dict:
        doc = self._require("budget")
        sp = self.spec
        n_set = sp.n_b_expected + sp.n_a_expected
        dec = w_rules.budget(self._elapsed_h(doc), self._options(doc, n_set, None), sp)
        return self._write("budget", dict(dec, n_set=n_set, note="W.9.9 P1-5: 주 세트 digest 전 최악 비용 관문."))

    def _plan(self, doc: dict) -> dict:
        p = doc["budget"]["plan"]
        e = doc.get("estimate") or {}
        if e.get("plan"):
            p = dict(p, with_c=e["plan"]["with_c"])
        return p
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_w_runner.py -q`
Expected: 9 passed (~1 min).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/w_runner.py tests/brain/w_world.py tests/brain/test_w_runner.py
git commit -m "feat(w): w_runner (1) — stage0 (synthetic checks, tables, OC compute time), reuse (V blocks/keys/commits), path (V gate-2 punish rows, V oracle naive counts, reward path), pilot + H6, OC design, smoke, worst-case budget gate"
```

---

### Task 7: `scripts/run_w.py` — the CLI

**Files:**
- Create: `scripts/run_w.py`
- Test: `tests/brain/test_w_cli.py`

**Interfaces:**
- Consumes: `w_runner` (Task 6: `build_ctx`, `Runner`, `pipeline_key`, `GATES`, `ORDER`), `w_measure` (`WMeasurer`, `w_measure_key`), `u_measure` (`UPool`, `u_measure_key`), `r_measure.RMeasurer`, `t_measure.t_measure_key`, `w_store.WCache`, `w_spec` (`SPEC`, `smoke`).
- Produces: `STAGES`, `POOL_STAGES`, `GATE_STAGES`, `check_args(a)`, `make_measurers(ucode, wcode, pool, ctx) -> (measure, learner)`, `exit_code(stage, out)`, `main(argv)`.

- [ ] **Step 1: Write the failing tests**

```python
"""scripts/run_w.py and W's files: arguments, exit codes, stage lists equal to the runner's; no W file imports the
frozen F v3 verdict.py or loads code by path; the only W measurement file is w_measure.py and no W pipeline module
defines a job; the CLI's pool code sits under `if __name__ == "__main__":`."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from flymon.brain import w_runner as WR

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_w", ROOT / "scripts/run_w.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)
W_FILES = sorted((ROOT / "flymon/brain").glob("w_*.py")) + [ROOT / "scripts/run_w.py"]


def test_args_and_exit_codes():
    a = lambda **k: SimpleNamespace(**dict(dict(stage=None, note=None, workers=None), **k))  # noqa: E731
    assert CLI.check_args(a()) == "--stage is required"
    assert CLI.check_args(a(stage="recompute")).startswith("--note is required")
    assert CLI.check_args(a(stage="path", note="x")).startswith("--note goes with")
    assert CLI.check_args(a(stage="path")) is None
    assert CLI.exit_code("path", dict(outcome="PASS")) == 0
    assert CLI.exit_code("path", dict(outcome="INVALID")) == 5
    assert CLI.exit_code("budget", dict(outcome="STOP_BUDGET")) == 3
    assert CLI.exit_code("smoke", dict(problems=["x"])) == 6 and CLI.exit_code("smoke", dict(problems=[])) == 0
    assert CLI.exit_code("seal", dict(status="NOT_READ")) == 6 and CLI.exit_code("judge", dict(verdict="FAIL")) == 0
    assert set(CLI.GATE_STAGES) == set(WR.GATES) and set(WR.ORDER) <= set(CLI.STAGES)


def test_no_frozen_verdict_and_no_code_by_path():
    for p in W_FILES:
        src = p.read_text()
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "m2-learning-f-v3" not in src and "spec_from_file_location" not in names, p.name
        assert not {"exec", "runpy", "exec_module"} & names, p.name


def test_only_w_measure_defines_a_job():
    for p in W_FILES:
        tree = ast.parse(p.read_text())
        jobs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.endswith("_job")]
        assert jobs == (["w_learn_job"] if p.name == "w_measure.py" else []), p.name


def test_main_guard():
    tree = ast.parse((ROOT / "scripts/run_w.py").read_text())
    last = tree.body[-1]
    assert isinstance(last, ast.If) and "__name__" in ast.unparse(last.test)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_cli.py -q`
Expected: FAIL (FileNotFoundError: `scripts/run_w.py`).

- [ ] **Step 3: Write `scripts/run_w.py`**

```python
#!/usr/bin/env python3
"""Spec appendix W (W.9.9 > W.9.8 > W.9.1-W.9.7 > W.0-W.8): the F v4 learning test on V's combined lever L_V — F.2's
sequential R / N brains and G.5's RN on u_rig, read with z_V, judged by per-fly joint satisfaction with a design
(q, K, F; k cap 8) chosen by W's own operating characteristic from a POOL even pilot. The controller runs every stage;
commit each block before the next.

    uv run python scripts/run_w.py --stage stage0                    # 0 verdict code, w_oc, tables, synthetic checks
    uv run python scripts/run_w.py --stage reuse                     # 1 V's blocks, R / T / U keys (no pool)
    uv run python scripts/run_w.py --stage path                      # 2 V gate ② rows, V oracle naive counts, reward
    uv run python scripts/run_w.py --stage pilot                     # 3 / 3a POOL even pilot (resumes), H6 stop
    uv run python scripts/run_w.py --stage oc                        # 4 the OC, the design and its ranking (no pool)
    uv run python scripts/run_w.py --stage smoke --workers 4         # 5 smoke 42_1xx_xxx, the cost ledger
    uv run python scripts/run_w.py --stage budget                    # 5a worst-case budget gate (no pool)
    uv run python scripts/run_w.py --stage set                       # 6 the main set's digests (no pool)
    uv run python scripts/run_w.py --stage oracle                    # 7 the oracle screen (24_700_xxx; the set is used)
    uv run python scripts/run_w.py --stage estimate                  # 8 the estimate on the testable count (no pool)
    uv run python scripts/run_w.py --stage naive                     # 9 naive probes until 8 gate pairs (resumes)
    uv run python scripts/run_w.py --stage gates                     # 10 the gate pairs, STOP_FEW_PAIRS (no pool)
    uv run python scripts/run_w.py --stage learn                     # 11 R / N / RN on the gate pairs (resumes)
    uv run python scripts/run_w.py --stage band                      # 11 the 2K probes for every gate pair (resumes)
    uv run python scripts/run_w.py --stage records                   # 12 C and the plasticity-off control (resumes)
    uv run python scripts/run_w.py --stage seal                      # 13 validity, manifest, archive (no pool)
    uv run python scripts/run_w.py --stage judge                     # 14 the verdict, once (no pool)
    uv run python scripts/run_w.py --stage recompute --note "..."    # after judge: analysis defect (W.5)
    uv run python scripts/run_w.py --stage invalid_run --note "..."  # after judge: measurement defect (W.5, H8)

Exit 0 recorded (PASS / SEALED / a judgement was read / a record stage), 3 a gate STOP (STOP_REUSE,
STOP_W_PATH_REPRO, STOP_PILOT_NO_EFFECT, STOP_OC_UNREACHABLE, STOP_BUDGET, STOP_FEW_PAIRS; recorded, W stops), 5 a gate
INVALID, 6 seal not SEALED or a smoke with problems, 7 the reuse condition or a measurement key broke (no block), 2 a
refusal (arguments, cwd, connectome sha256, chain, uncommitted summary, dirty hashed file, V's raw data, the set not
reproducing block set)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("stage0", "reuse", "path", "pilot", "oc", "smoke", "budget", "set", "oracle", "estimate", "naive", "gates",
          "learn", "band", "records", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("path", "pilot", "smoke", "oracle", "naive", "learn", "band", "records")
GATE_STAGES = ("stage0", "reuse", "path", "pilot", "oc", "budget", "set", "oracle", "estimate", "naive", "gates",
               "learn", "band")
QUIET = ("manifest", "checks", "record", "records", "screened", "pairs", "judgement", "ranking", "calibration", "theta",
         "synthetic", "tables", "set", "archive", "oc_timing", "decision_files", "costs", "cost", "exploratory")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (W.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    return None


def make_measurers(ucode: dict, wcode: dict, pool, ctx: dict):
    """measure(z, smoke): one RMeasurer per (z, smoke) over U's job copies (u_measure.UPool, unchanged) behind a WCache
    keyed by the W measurement key (results/w/cache or results/w/smoke/cache); learner(smoke): a WMeasurer behind the
    same caches."""
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.u_measure import UPool
    from flymon.brain.v_spec import SPEC as V
    from flymon.brain.w_measure import WMeasurer
    from flymon.brain.w_spec import SPEC
    from flymon.brain.w_store import WCache
    caches = {s: WCache(SPEC.smoke_cache_dir if s else SPEC.cache_dir, wcode) for s in (False, True)}
    upool = UPool(pool)
    windows = dict(strength=SPEC.strength, settle_ms=V.settle_ms, read_ms=V.read_ms, window_ms=V.window_ms)
    timing = dict(present_ms=SPEC.pulse_ms, gap_ms=SPEC.gap_ms, train_settle_ms=SPEC.train_settle_ms,
                  seed_stride=SPEC.fly_train_stride)

    def measure(z, smoke=False):
        return RMeasurer(upool, caches[smoke], V, ctx["params"], ctx["readout"], z, ctx["types"], ctx["n_kc"])

    def learner(smoke=False):
        return WMeasurer(pool, caches[smoke], ctx["params"], ctx["readout"], V.p_type, SPEC.reward_dan,
                         SPEC.punish_dan, windows, timing)
    return measure, learner


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import w_rules as R
    if stage in GATE_STAGES:
        o = out.get("outcome")
        return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)
    if stage == "smoke":
        return EXIT_NOT_READ if out.get("problems") else 0
    if stage == "seal":
        return 0 if out.get("status") == R.SEALED else EXIT_NOT_READ
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--note")
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

    from flymon.brain import w_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import u_measure_key
    from flymon.brain.w_measure import w_measure_key
    from flymon.brain.w_spec import SPEC, smoke

    spec = smoke(SPEC) if a.stage == "smoke" else SPEC
    try:
        ctx = w_runner.build_ctx(spec, NPZ)
        code, ucode, wcode = code_key(NPZ, files=R_MEASURE_FILES), u_measure_key(NPZ), w_measure_key(NPZ)
        workers = a.workers or spec.workers
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_dan,
                           reward_type=SPEC.reward_dan, timeout_s=SPEC.pool_timeout_s)
        try:
            measure, learner = make_measurers(ucode, wcode, pool, ctx)
            r = w_runner.Runner(measure, learner, ctx, spec, code=code, tcode=t_measure_key(NPZ), ucode=ucode,
                                wcode=wcode, pipeline=w_runner.pipeline_key())
            if a.stage in ("recompute", "invalid_run"):
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

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/brain/test_w_cli.py tests/brain/test_w_runner.py -q`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_w.py tests/brain/test_w_cli.py
git commit -m "feat(w): scripts/run_w.py — stages stage0..judge, recompute/invalid_run, W key and caches, exit codes 0/2/3/5/6/7; AST checks (one job, no frozen verdict, no code by path)"
```

---

### Task 8: `w_runner` (2) — set, oracle, estimate, naive, gates, learn, band, records, seal, judge, after judge; the full suite

**Files:**
- Modify: `flymon/brain/w_runner.py` (append the rest of `Runner`)
- Test: `tests/brain/test_w_runner_main.py`, `tests/brain/test_w_judge.py`

**Interfaces:**
- Consumes: Task 6's `Runner` helpers (`_require`, `_keys_chain`, `_write`, `_prog`, `_prog_add`, `_m`, `_wm`, `_z_v`, `units`, `_declared`, `_by_pair`, `_manifest`, `_options`, `_plan`, `_elapsed_h`), `w_pairs.set_summary`, `w_records`, `w_verdict`, `w_rules`, `w_store`.
- Produces: `stage_set`, `_main_rows`, `stage_oracle`, `stage_estimate`, `_checker`, `_stop_budget`, `_later_h`, `stage_naive`, `stage_gates`, `_gate_rows`, `_measure_stage`, `learn_units(doc)`, `band_units(doc)`, `record_units(doc)`, `stage_learn`, `stage_band`, `stage_records`, `_raw`, `_want`, `stage_seal`, `_pair_sets`, `_read`, `_check_pins`, `stage_judge`, `_require_after_judge`, `stage_recompute(note)`, `stage_invalid_run(note)`.

- [ ] **Step 1: Write the failing tests**

```python
"""W's stage chain from the main set to the judgement (W.9.9 순서 6-14): the set check, the oracle on z_V with
24_700_xxx (the set is used from here), the naive screen in declared order stopping at 8 gate pairs (no pair beyond
them measured; unbalanced pairs skipped), STOP_FEW_PAIRS, every RN unit carrying both phases from its own pre, the
learning / band / record manifests holding no statistic, the seal and one judgement (PASS and FAIL)."""
import json
from pathlib import Path

import pytest

from flymon.brain import w_rules
from flymon.brain import w_runner as WR
from flymon.brain.r_pairs import row_key
from tests.brain.w_world import MAIN, SPEC, World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_judge_pass(w):
    out = through(w, "judge")
    d = doc()
    assert set(d) >= set(WR.ORDER) and len(d["ledger"]) == len(WR.ORDER)
    assert all(d[b]["w_measure_key"] == "w" * 64 for b in WR.ORDER)
    assert [c["diffs"] for c in d["path"]["checks"]] == [[]] * len(d["path"]["checks"])
    assert len(d["path"]["checks"]) == 8 + 6 + 1
    assert d["pilot"]["outcome"] == "PASS" and d["pilot"]["exploratory"]["label"] == "탐색"
    assert d["oc"]["selected"] == dict(q=0.75, K=8, F=8, cost_h=d["oc"]["selected"]["cost_h"])
    assert d["budget"]["plan"]["with_c"] is True
    assert d["set"]["set"]["n_b"] == 167 and d["oracle"]["n"] == 249
    assert ("screen", "L", 249, tuple(d["reuse"]["z_V"]["A"])) in w.oracle_calls
    testable = [p for p in d["oracle"]["pairs"] if p["testable"]]
    g = d["gates"]["gates"]
    assert [x["c"] for x in g] == [p["c"] for p in testable[:8]]
    assert [x["c"] for x in d["naive"]["screened"]] == [p["c"] for p in testable[:8]]
    assert all("stats" not in d[s] for s in ("learn", "band", "records"))
    assert d["learn"]["n_units"] == 8 * 8 * 3 and d["band"]["n_units"] == 8 * 8 * 3
    assert d["records"]["n_units"] == 8 * 8 * 3 + 2 * 2
    assert d["seal"]["status"] == "SEALED" and Path(d["seal"]["archive"]["dir"]).exists()
    assert out["verdict"] == "PASS" and out["sentence"].startswith("조합 지렛대 모델(")
    assert "설계 q 0.75 · K 8 · F 8 · k 상한 8" in out["sentence"]
    assert all(n["counts_equal"] and n["weights_equal"] for n in out["records"]["noplast"])
    assert Path(WR.DONE_MARKER).exists()


def test_naive_stops_at_eight_gate_pairs_and_unbalanced_pairs_skip(w):
    testable = [r for r in MAIN if r["c"] % 3 == 0]
    for r in testable[:3]:
        w.model.offset[row_key(r)] = 40.0                               # naive-unbalanced: not a gate pair
    through(w, "gates")
    d = doc()
    sc = d["naive"]["screened"]
    assert [x["gate"] for x in sc[:3]] == [False] * 3 and len(sc) == 11 and sum(x["gate"] for x in sc) == 8
    measured = {json.loads(Path(m["cache_file"]).read_text())["inputs"]["pair"] for m in d["naive"]["manifest"]}
    assert measured == {x["key"] for x in sc}


def test_few_pairs_stop(w):
    for r in MAIN:
        w.model.offset[row_key(r)] = 40.0
    through(w, "naive")
    out = w.runner().stage_gates()
    assert out["outcome"] == w_rules.STOP_FEW_PAIRS and out["sentence"] == (
        "W 주 세트 249쌍에서 오라클 순진 균형·시험 가능 쌍이 0개로 최소 4에 못 미쳤다.")


def test_rn_units_run_both_phases_from_pre(w):
    through(w, "gates")
    r = w.runner()
    us = r.learn_units(doc())
    rn = [u for u in us if u["brain"] == "RN"]
    assert rn and all([p[0] for p in u["phases"]] == ["PAM08", None] for u in rn)
    assert all(u["phases"][0][2] == SPEC.train_base(u["idx"], 0) and u["probe_seeds"][0] ==
               SPEC.probe_seeds(u["idx"], u["fly"], 1)[0] for u in rn)                # RN starts from its own pre
    r0 = [u for u in us if u["brain"] == "R"][0]
    c = r0["idx"]
    assert r0["phases"] == [["PAM08", 20, 28_000_000 + c * 40_000], ["PPL105", 20, 28_000_000 + c * 40_000 + 20]]
    assert r0["probe_seeds"] == [26_000_000 + c * 4_000 + k for k in range(8)]
    b = r.band_units(doc())
    assert b[0]["probe_seeds"] == [26_000_000 + c * 4_000 + k for k in range(8, 16)]


def test_judge_fail_and_once(w):
    gates_c = [r for r in MAIN if r["c"] % 3 == 0][:8]
    w.model.effects[row_key(gates_c[0])] = (30.0, 0.0)                 # one pair learns no punishment
    out = through(w, "judge")
    assert out["verdict"] == "FAIL" and out["judgement"]["failing"] == [row_key(gates_c[0])]
    assert out["sentence"].startswith("F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부(")
    w.judge_commits.append("x")
    with pytest.raises(SystemExit):
        w.runner().stage_judge()
```

```python
"""W's seal and one judgement (W.3 10-11, W.5, W.9.8 H8): the seal checks every raw entry's stored inputs against the
units the declared gate pairs, design and seeds give (a changed entry → INVALID, a missing one → NOT_READ, nothing
read), pins the design, the gate pairs and z_V; judge reads only a sealed set under the sealed decision code and the
pinned blocks, once (the marker; one re-generation after the mark; never after a judge block); recompute and
invalid_run only after judge."""
import json
from pathlib import Path

import pytest

from flymon.brain import w_runner as WR
from tests.brain.w_world import SPEC, World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_seal_invalid_on_an_entry_measured_with_other_inputs(w):
    """An entry whose file still matches its manifest sha but whose stored inputs are not the declared unit's (a
    measurement made with other seeds) is INVALID; a file changed after its block is NOT_READ."""
    from flymon.brain.h3_store import sha256_file
    through(w, "records")
    d = doc()
    m = d["learn"]["manifest"][0]
    e = json.loads(Path(m["cache_file"]).read_text())
    e["inputs"]["probe_seeds"][0] += 1
    Path(m["cache_file"]).write_text(json.dumps(e))
    m["sha256"] = sha256_file(m["cache_file"])
    Path(SPEC.summary).write_text(json.dumps(d))
    out = w.runner().stage_seal()
    assert out["status"] == "INVALID" and any("저장 입력" in r for r in out["reasons"])
    with pytest.raises(SystemExit):
        w.runner().stage_judge()


def test_seal_not_read_on_a_changed_file(w):
    through(w, "records")
    m = doc()["learn"]["manifest"][0]
    Path(m["cache_file"]).write_text(Path(m["cache_file"]).read_text() + " ")
    out = w.runner().stage_seal()
    assert out["status"] == "NOT_READ" and any("sha256" in r for r in out["reasons"])


def test_seal_not_read_on_a_missing_entry(w):
    through(w, "records")
    Path(doc()["band"]["manifest"][0]["cache_file"]).unlink()
    out = w.runner().stage_seal()
    assert out["status"] == "NOT_READ" and out["archive"] is None


def test_judge_once_regeneration_and_pins(w, monkeypatch):
    through(w, "seal")
    real = WR.Runner._read

    def die(self, doc_, mark=None):
        real(self, doc_, mark)
        raise RuntimeError("died after the mark")
    monkeypatch.setattr(WR.Runner, "_read", die)
    with pytest.raises(RuntimeError):
        w.runner().stage_judge()
    assert Path(WR.JUDGE_MARKER).exists() and "judge" not in doc()
    monkeypatch.setattr(WR.Runner, "_read", real)
    out = w.runner().stage_judge()
    assert out["resumed_after_mark"] is True and out["verdict"] == "PASS"
    d = doc()
    d.pop("judge")
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        w.runner().stage_judge()                                 # DONE marker: never a second block


def test_judge_refuses_moved_pins_and_decision_code(w, monkeypatch):
    through(w, "seal")
    d = doc()
    d["gates"]["gates"] = d["gates"]["gates"][::-1]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        w.runner().stage_judge()
    d["gates"]["gates"] = d["gates"]["gates"][::-1]
    Path(SPEC.summary).write_text(json.dumps(d))
    monkeypatch.setattr(WR, "decision_key", lambda: dict(key="other", files={}))
    with pytest.raises(SystemExit):
        w.runner().stage_judge()


def test_after_judge(w):
    with pytest.raises(SystemExit):
        w.runner().stage_recompute("x")
    through(w, "judge")
    e = w.runner().stage_recompute("note")
    assert e["verdict"] == "PASS" and e["differs_from_judge"] is False
    out = w.runner().stage_invalid_run("measurement defect")
    assert out["status"] == "INVALID_RUN" and "같은 관문 쌍으로 다시 돌리지 않는다" in out["rule"]
    with pytest.raises(SystemExit):
        w.runner().stage_recompute("again")


def test_keys_rechecked_before_measurement_seal_and_judge(w):
    through(w, "gates")
    with pytest.raises(SystemExit) as e:
        w.runner(wcode={"key": "z" * 64}).stage_learn()
    assert e.value.code == 7
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brain/test_w_runner_main.py tests/brain/test_w_judge.py -q`
Expected: FAIL (AttributeError: `Runner` has no `stage_set`).

- [ ] **Step 3: Append to `flymon/brain/w_runner.py`**

Append exactly this at the end of the file (it continues `class Runner`; keep one blank line between `_plan` and the new section):

```python
    # ================================================================ 6: the main set (W.2) — no pool
    def stage_set(self) -> dict:
        self._require("set")
        sp = self.spec
        try:
            js = self.ctx["w_set"]()
        except ValueError as e:
            refuse(f"the W set cannot be generated: {e}")
        from .w_pairs import set_summary
        bad = [] if (js["n_b"], js["n_a"]) == (sp.n_b_expected, sp.n_a_expected) else [
            f"(b) {js['n_b']} · (a) {js['n_a']} ≠ 선언 {sp.n_b_expected} · {sp.n_a_expected}"]
        body = dict(outcome=w_rules.INVALID if bad else w_rules.PASS, reasons=bad, set=dict(set_summary(js),
                                                                                          keys=js["keys"]),
                    note="W.2: 어떤 오라클보다 먼저 커밋한다.")
        return self._write("set", body)

    def _main_rows(self, doc: dict) -> list:
        try:
            return self.ctx["main_rows"](doc["set"]["set"])
        except ValueError as e:
            refuse(f"the W set does not reproduce block set (W.2): {e}")

    # ================================================================ 7: the oracle screen (W.2, W.9.8 H3)
    def stage_oracle(self) -> dict:
        doc = self._require("oracle")
        self._keys_chain(doc, "oracle")
        sp = self.spec
        rows = self._main_rows(doc)
        z_v = self._z_v(doc)
        m = self._m(z_v)
        t0 = time.perf_counter()
        prev = self._prog("oracle")
        got = m.oracle(rows, V_SPEC.cond("L"), "screen", sp.oracle_seeds())
        wall = prev + time.perf_counter() - t0
        self._prog_add("oracle", time.perf_counter() - t0)
        per, bad = [], []
        for r, g in zip(rows, got):
            res = g["result"]
            q = res.get("q", {})
            if q.get("edit") != V_SPEC.lever_edit or q.get("csc_sha256") != V_SPEC.sha_combined or \
                    q.get("edit_edges") != V_SPEC.lever_edges:
                bad.append(f"{g['key']}: edit {q.get('edit')} / CSC / edges")
            st = pair_stats(res["report"], z_v, V_SPEC.testable_min)
            if st is None:
                bad.append(f"{g['key']}: d′ 정의 불가")
                continue
            per.append(dict(key=g["key"], c=r["c"], axis=r["axis"], turn=r["turn"], d_pre=st["d_pre"], r=st["r"],
                            p=st["p"], testable=bool(st["testable"])))
        man = [dict(key=g["key"], cache_file=g["cache_file"], cache_key=g["cache_key"],
                    sha256=sha256_file(g["cache_file"])) for g in got]
        body = dict(outcome=w_rules.INVALID if bad else w_rules.PASS, reasons=bad, pairs=per,
                    n_testable=sum(p["testable"] for p in per), n=len(per), manifest=man, seeds=sp.oracle_seeds(),
                    z_V={k: list(v) for k, v in z_v.items()}, note="W.9.8 H3: 이 블록 뒤 주 세트는 사용된 것이다.")
        return self._write("oracle", body, wall_s=wall)

    # ================================================================ 8: the estimate after the oracle (W.9.8 H3)
    def stage_estimate(self) -> dict:
        doc = self._require("estimate")
        n_t = int(doc["oracle"]["n_testable"])
        opts = [o for o in self._options(doc, 0, n_t)
                if (o["design"]["q"], o["design"]["K"], o["design"]["F"]) == tuple(
                    doc["budget"]["plan"]["design"][k] for k in ("q", "K", "F"))]
        if not doc["budget"]["plan"]["with_c"]:
            opts = [o for o in opts if not o["with_c"]]
        dec = w_rules.budget(self._elapsed_h(doc), opts, self.spec)
        return self._write("estimate", dict(dec, n_testable=n_t,
                                            note="W.9.8 H3: 실제 시험 가능 쌍 수로 추정 갱신(C 제외 먼저)."))

    # ---- the in-stage ledger (W.9.8 H3: "배치마다 원장 갱신") ------------------------------------------------------
    def _checker(self, doc: dict, stage: str, part: str, remaining_after: float):
        """check(done, todo): stop when elapsed + this stage's remaining share + the later stages' estimate > 24 h."""
        est = doc["estimate"]["plan"]["parts_h"]
        base = self._elapsed_h(doc)
        t_start = time.perf_counter()
        prev = self._prog(stage)

        def check(done, todo):
            now = time.perf_counter()
            self._prog_add(stage, now - check.t)
            check.t = now
            spent = (prev + now - t_start) / 3600
            left = est[part] * (1 - done / max(todo, 1)) if done else est[part]
            if base + spent + left + remaining_after > self.spec.budget_h:
                raise BudgetStop(f"누적 {base + spent:.2f} h + 남은 {left + remaining_after:.2f} h")
        check.t = time.perf_counter()
        return check

    def _stop_budget(self, stage: str, msg: str, wall: float) -> dict:
        body = dict(outcome=w_rules.STOP_BUDGET, reasons=[msg],
                    sentence=w_rules.sentence(w_rules.STOP_BUDGET, dict(h=msg)))
        return self._write(stage, body, wall_s=wall)

    def _later_h(self, doc: dict, after: tuple) -> float:
        est = doc["estimate"]["plan"]["parts_h"]
        return float(sum(est[p] for p in after if p in est and (p != "c" or self._plan(doc)["with_c"])))

    # ================================================================ 9: the naive screen (W.9.5, W.9.8 H3, W.9.9 P1-4)
    def stage_naive(self) -> dict:
        doc = self._require("naive")
        self._keys_chain(doc, "naive")
        sp = self.spec
        d = self._plan(doc)["design"]
        K, F = int(d["K"]), int(d["F"])
        rows = {r["c"]: r for r in self._main_rows(doc)}
        cand = [p for p in doc["oracle"]["pairs"] if p["testable"]]
        z_v = self._z_v(doc)
        t0 = time.perf_counter()
        prev = self._prog("naive")
        check = self._checker(doc, "naive", "naive", self._later_h(doc, ("learn", "band", "c", "noplast")))
        wm = self._wm()
        screened, gates, man = [], [], []
        i = 0
        try:
            while i < len(cand) and len(gates) < sp.k_cap:
                need = sp.k_cap - len(gates)
                batch = cand[i:i + max(1, min(need, sp.workers // F or 1))]
                units = [u for p in batch for u in self.units([rows[p["c"]]], "main", K, F, V_SPEC.lever_edit,
                                                               brains=("naive",))]
                check(i, len(cand))
                got = wm.learn(units, "naive")
                by = self._by_pair(got)
                for p in batch:
                    g = by[p["key"]]
                    pre = np.stack([w_records.counts(x["result"]["stages"][0]) for x in
                                    sorted(g, key=lambda x: x["unit"]["fly"])])
                    dn = w_verdict.naive_dprime(pre, z_v)
                    ok = bool(abs(dn) < sp.naive_max)
                    screened.append(dict(key=p["key"], c=p["c"], naive_d=dn, gate=ok))
                    man += self._manifest(g)
                    if ok and len(gates) < sp.k_cap:
                        gates.append(dict(key=p["key"], c=p["c"]))
                i += len(batch)
        except BudgetStop as e:
            return self._stop_budget("naive", str(e), prev + time.perf_counter() - t0)
        body = dict(outcome=w_rules.PASS, reasons=[], screened=screened, gates=gates, manifest=man,
                    n_oracle_testable=len(cand), stopped_at=i, design=d)
        return self._write("naive", body, wall_s=prev + time.perf_counter() - t0)

    # ================================================================ 10: the gate pairs (W.9.5, W.9.8 H7)
    def stage_gates(self) -> dict:
        doc = self._require("gates")
        sp = self.spec
        g = doc["naive"]["gates"]
        dec = w_rules.gate_pairs(len(doc["set"]["set"]["keys"]), len(g), sp)
        d = self._plan(doc)["design"]
        body = dict(dec, gates=g, design=dict(q=d["q"], K=d["K"], F=d["F"], k_cap=sp.k_cap),
                    note="생성원 턴 순(같은 턴 (b) 먼저), k 상한 8.")
        return self._write("gates", body)

    # ================================================================ 11-12: learning, BAND probes, records
    def _gate_rows(self, doc: dict) -> list:
        rows = {r["c"]: r for r in self._main_rows(doc)}
        return [rows[g["c"]] for g in doc["gates"]["gates"]]

    def _measure_stage(self, stage: str, part: str, units: list, after: tuple) -> dict:
        doc = self._require(stage)
        self._keys_chain(doc, stage)
        t0 = time.perf_counter()
        prev = self._prog(stage)
        check = self._checker(doc, stage, part, self._later_h(doc, after))
        try:
            got = self._wm().learn(units, stage, check=check)
        except BudgetStop as e:
            return self._stop_budget(stage, str(e), prev + time.perf_counter() - t0)
        body = dict(outcome=w_rules.PASS, reasons=[], manifest=self._manifest(got), n_units=len(units),
                    note="블록은 통계를 담지 않는다 — 판정은 봉인 뒤 1회.")
        return self._write(stage, body, wall_s=prev + time.perf_counter() - t0)

    def learn_units(self, doc: dict) -> list:
        d = doc["gates"]["design"]
        return self.units(self._gate_rows(doc), "main", int(d["K"]), int(d["F"]), V_SPEC.lever_edit)

    def band_units(self, doc: dict) -> list:
        d = doc["gates"]["design"]
        K = int(d["K"])
        return self.units(self._gate_rows(doc), "main", 2 * K, int(d["F"]), V_SPEC.lever_edit, k0=K)

    def record_units(self, doc: dict) -> list:
        sp, d = self.spec, doc["gates"]["design"]
        K, F = int(d["K"]), int(d["F"])
        rows = self._gate_rows(doc)
        out = self.units(rows, "main", K, F, V_SPEC.no_edit) if self._plan(doc)["with_c"] else []
        out += [dict(u, brain="noplast") for u in self.units(rows[:sp.noplast_pairs], "main", K, sp.noplast_flies,
                                                               V_SPEC.lever_edit, brains=("R",), plastic=False)]
        return out

    def stage_learn(self) -> dict:
        doc = self._doc()
        return self._measure_stage("learn", "learn", self.learn_units(doc) if "gates" in doc else [],
                                   ("band", "c", "noplast"))

    def stage_band(self) -> dict:
        doc = self._doc()
        return self._measure_stage("band", "band", self.band_units(doc) if "gates" in doc else [], ("c", "noplast"))

    def stage_records(self) -> dict:
        doc = self._require("records")
        self._keys_chain(doc, "records")
        t0 = time.perf_counter()
        units = self.record_units(doc)
        got = self._wm().learn(units, "records")
        body = dict(manifest=self._manifest(got), n_units=len(units), with_c=self._plan(doc)["with_c"],
                    note="기록 측정(C·가소성 끈 대조) — 판정 아님.")
        return self._write("records", body, wall_s=time.perf_counter() - t0)

    # ================================================================ 13: seal (W.3 10)
    def _raw(self, doc: dict, stage: str) -> tuple:
        got, bad = w_store.load_manifest(doc[stage]["manifest"])
        return got, [f"{stage}: {b}" for b in bad]

    def _want(self, doc: dict) -> dict:
        wm = self._wm()
        out = {}
        for stage, units in (("learn", self.learn_units(doc)), ("band", self.band_units(doc)),
                             ("records", self.record_units(doc))):
            for u in units:
                out[(stage, f"{u['pair']}|{u['fly']}|{u['brain']}")] = wm.inputs(u, stage)
        d = doc["gates"]["design"]
        rows = {r["c"]: r for r in self._main_rows(doc)}
        for p in doc["naive"]["screened"]:
            for u in self.units([rows[p["c"]]], "main", int(d["K"]), int(d["F"]), V_SPEC.lever_edit,
                                brains=("naive",)):
                out[("naive", f"{u['pair']}|{u['fly']}|{u['brain']}")] = wm.inputs(u, "naive")
        return out

    def stage_seal(self) -> dict:
        doc = self._require("seal")
        self._keys_chain(doc, "seal")
        want = self._want(doc)
        reasons, invalid, files = [], [], []
        for stage in ("naive", "learn", "band", "records"):
            got, bad = self._raw(doc, stage)
            reasons += bad
            for g in got:
                files.append(g["cache_file"])
                have = json.loads(Path(g["cache_file"]).read_text()).get("inputs")
                exp = want.get((stage, g["key"]))
                if exp is None or have is None or canonical(have) != canonical(exp):
                    invalid.append(f"{stage}: {g['key']} 저장 입력이 선언과 다름")
        n_exp = len(want)
        if len(files) != n_exp:
            reasons.append(f"원자료 {len(files)}개 ≠ 선언 {n_exp}개")
        status = w_rules.INVALID if invalid else (w_rules.NOT_READ if reasons else w_rules.SEALED)
        body = dict(status=status, reasons=reasons + invalid, invalid=invalid, n_files=len(files),
                    decision=dict(decision_key(), **decision_pins(doc)), archive=None)
        if status == w_rules.SEALED:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_root / f"{stamp}-{(git_state().get('commit') or 'nocommit')[:12]}"
            body["archive"] = dict(dir=str(dest), files=w_store.archive_copy(files, dest, self.archive_root))
        return self._write("seal", body)

    # ================================================================ 14: judge once (W.3 11, W.4 / W.9, W.7)
    def _pair_sets(self, doc: dict, stage: str) -> dict:
        got, bad = self._raw(doc, stage)
        if bad:
            refuse(f"raw files changed since the seal: {bad[:3]}")
        out = {}
        for g in got:
            pair, fly, brain = g["key"].rsplit("|", 2)
            out.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
        return out

    def _read(self, doc: dict, mark=None) -> dict:
        sp = self.spec
        d = doc["gates"]["design"]
        q, K, F = float(d["q"]), int(d["K"]), int(d["F"])
        flies = list(range(F))
        z_v, z_h4 = self._z_v(doc), _z(self.ctx["z"])
        learn, band, naive = (self._pair_sets(doc, s) for s in ("learn", "band", "naive"))
        recs = self._pair_sets(doc, "records")
        if mark is not None:
            mark()
        pairs, machine = {}, []
        for g in doc["gates"]["gates"]:
            k, c = g["key"], g["c"]
            seeds = {f: sp.probe_seeds(c, f, K) for f in flies}
            nv = {x["unit"]["fly"]: w_records.counts(x["result"]["stages"][0]) for x in naive[k]}
            machine += [f"{k}: {m}" for m in w_records.machine_reasons(learn[k], flies,
                                                                     self._declared(V_SPEC.lever_edit, seeds),
                                                                     naive=nv, band=band.get(k, []))]
            d_k = w_records.pair_data(learn[k], flies)
            pairs[k] = (d_k, w_records.extend(d_k, band[k], flies))
        v = w_verdict.judge(pairs, machine, z_v, q, F, K, sp)
        sent = w_rules.verdict_sentence(v, dict(q=q, K=K, F=F), sp)
        c_rec, noplast = {}, []
        for k, rows in recs.items():
            c_rows = [r for r in rows if r["unit"]["brain"] in BRAINS]
            if c_rows:
                d_c = w_records.pair_data(c_rows, flies)
                c_rec[k] = {kk: vv for kk, vv in w_verdict.judge_pair(d_c, None, z_h4, q, F, K, sp).items()
                            if kk in ("status", "code_k", "n_sat", "q_sat", "mech_k_ok", "stats")}
            for r in rows:
                if r["unit"]["brain"] == "noplast":
                    st = {s["stage"]: s for s in r["result"]["stages"]}
                    same = all(np.array_equal(w_records.counts(st[s]), w_records.counts(st["pre"])) for s in st)
                    w_same = all(st[s]["w_sha256"] == r["result"]["w0_sha256"] for s in st)
                    noplast.append(dict(pair=k, fly=r["unit"]["fly"], counts_equal=bool(same),
                                        weights_equal=bool(w_same)))
        gate_rec = w_records.pilot_record({k: v[0] for k, v in pairs.items()}, z_v, {}, sp)
        records = dict(C=c_rec, noplast=noplast, gate_pairs=gate_rec,
                       naive={g["key"]: g["naive_d"] for g in doc["naive"]["screened"]},
                       oracle=dict(n=doc["oracle"]["n"], n_testable=doc["oracle"]["n_testable"]),
                       oc=dict(selected=doc["oc"]["selected"], mixed_flies=(doc["oc"].get("records") or {}).get(
                           "mixed_flies")), label_C="기록 — 판정 아님")
        return dict(status=w_rules.READ, verdict=v["verdict"], sentence=sent["sentence"],
                    consequence=sent["consequence"], judgement=v, records=records, design=dict(q=q, K=K, F=F))

    def _check_pins(self, doc: dict) -> None:
        sd = doc["seal"].get("decision") or {}
        moved = [k for k, v in decision_pins(doc).items() if sd.get(k) is None or sd.get(k) != v]
        if moved:
            refuse(f"the sealed decision record ({', '.join(moved)}) differs from the live blocks (W.3 10)")

    def stage_judge(self) -> dict:
        doc = self._require("judge")
        self._keys_chain(doc, "judge")
        if doc["seal"].get("status") != w_rules.SEALED:
            refuse(f"block seal's status is {doc['seal'].get('status')}: W reads only a sealed set")
        if summary_git(self.summary_path)["judge_commits"]:
            refuse(f"git history of {self.summary_path} already holds a judge block; the set is used once (W.5)")
        sealed = (doc["seal"].get("decision") or {}).get("key")
        now = decision_key()["key"]
        if sealed != now:
            refuse(f"the decision code hash {now} is not the sealed {sealed} (W.5)")
        self._check_pins(doc)
        if Path(DONE_MARKER).exists():
            refuse(f"{DONE_MARKER} exists: a judge block was written once")
        resumed = None
        if Path(JUDGE_MARKER).exists():
            if Path(REREAD_MARKER).exists():
                refuse(f"{REREAD_MARKER} exists: judge was re-generated once after the mark already")
            mk = json.loads(Path(JUDGE_MARKER).read_text())
            if mk.get("seal_written_at") != doc["seal"].get("written_at") or mk.get("decision_key") != sealed:
                refuse(f"{JUDGE_MARKER} belongs to another seal or decision code; no re-generation")
            resumed = mk

        def mark():
            w_store.write_json(REREAD_MARKER if resumed else JUDGE_MARKER, dict(
                seal_written_at=doc["seal"].get("written_at"), decision_key=now, code_key=self.code_key,
                w_measure_key=self.w_measure_key, pipeline_key=self.pipeline_key, read_at=_now()), self.plist)
        out = self._read(doc, mark)
        out = dict(out, resumed_after_mark=resumed is not None, mark_read_at=(resumed or {}).get("read_at"))
        block = self._write("judge", out)
        w_store.write_json(DONE_MARKER, dict(judge_written_at=block["written_at"]), self.plist)
        return block

    # ---- after reading (W.5, W.9.8 H8) -------------------------------------------------------------------------------
    def _require_after_judge(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        if "judge" not in doc:
            refuse(f"stage {stage} needs block judge (W.5: only after the judgement was read)")
        if "invalid_run" in doc:
            refuse("block invalid_run exists: the main set is closed (W.5)")
        return doc

    def stage_recompute(self, note: str) -> dict:
        """W.5 / H8: an analysis defect after reading — the same sealed raw data recomputed."""
        if not note:
            refuse("--note is required (W.5)")
        doc = self._require_after_judge("recompute")
        self._check_pins(doc)
        out = self._read(doc)
        dk = decision_key()["key"]
        entry = dict(note=note, verdict=out["verdict"], sentence=out["sentence"], decision_key=dk,
                     decision_changed_since_seal=bool(dk != (doc["seal"].get("decision") or {}).get("key")),
                     differs_from_judge=bool(out["verdict"] != doc["judge"]["verdict"]
                                             or out["sentence"] != doc["judge"]["sentence"]),
                     pipeline_key=self.pipeline_key, git=git_state(), written_at=_now())
        w_store.write_summary_block(self.summary_path, "recompute", list(doc.get("recompute", [])) + [entry],
                                    self.plist)
        return entry

    def stage_invalid_run(self, note: str) -> dict:
        """W.5 / H8: a measurement defect after reading — INVALID_RUN; the gate pairs are never run again."""
        if not note:
            refuse("--note is required (W.5)")
        doc = self._require_after_judge("invalid_run")
        body = dict(status=w_rules.INVALID_RUN, note=note, judge_verdict=doc["judge"]["verdict"],
                    rule="W.5 / W.9.8 H8: 같은 관문 쌍으로 다시 돌리지 않는다; 대체 세트는 새 선언으로만(사용자).")
        return self._write("invalid_run", body)
```

- [ ] **Step 4: Run the W tests and the collectors**

Run: `uv run pytest tests/brain/test_w_*.py tests/brain/test_p_spec.py tests/agent/test_e_spec_store.py tests/brain/test_v_spec.py tests/brain/test_u_spec.py tests/brain/test_t_spec.py -q`
Expected: all pass (~5 min; `test_w_measure.py` dominates).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (no R / S / T / U / V test changes; only W files and the one `MODULES` entry are new).

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/w_runner.py tests/brain/test_w_runner_main.py tests/brain/test_w_judge.py
git commit -m "feat(w): w_runner (2) — set digest, oracle screen (24_700_xxx), estimate, naive screen to 8 gate pairs, STOP_FEW_PAIRS, learn/band/records manifests, in-stage budget ledger, seal (stored inputs, pins, archive), judge once, recompute, invalid_run"
```

---

## Self-review (done while writing)

- **Spec coverage:**

  | Spec item | Where it lives |
  |---|---|
  | W.0 disclosures (V's limits, F v4 never run, naive bottleneck, DAN compartments, auxiliary contamination, new code, set capacity) and W.9.4 / H9 J.12 precedent | no code; the PASS / FAIL consequences (Task 4), the OC notes (Task 6 `stage_oc`), the W.10 result section (Runs 14) |
  | W.1 engine, L_V, z_V, C on h4 z; W key = U key + W file | Task 1 (`z_v_declared`), Task 5 (`w_measure_key`), Task 4 (`reuse`: z_V at 3 decimals), Task 6 (`_declared`, `_job_invalid`) |
  | W.2 / W.9.5 / P3-15 main set, oracle screen, gate pairs, k cap 8, minimum 4 | Task 1 (`w_set`, `main_rows`), Task 8 (`stage_set`, `stage_oracle`, `stage_naive`, `stage_gates`), Task 4 (`gate_pairs`) |
  | W.3 1 reuse | Task 4 (`reuse`), Task 6 (`stage_reuse`, `_require` exit 7) |
  | W.3 2 / P2-13 / P2-9 path (P rows, naive repro mandatory, reward check), unit tests (edit not applied, RN1 = R1) | Task 4 (`p_repro_diffs`, `naive_repro_diffs`, `reward_check`, `path`), Task 5 (real-connectome tests), Task 6 (`stage_path`) |
  | W.3 3 / W.9.4 / H6 pilot and its stop, W.9.2 pilot pre-check | Task 1 (`pilot_rows`), Task 4 (`pilot_record`, `pilot`), Task 6 (`stage_pilot`) |
  | W.9.3 / H1 / H4 / W.9.9 P0-1 / P1-2 / P1-3 / P1-6 / P2-7 / P2-8 OC | Task 3 (`w_oc`), Task 6 (`stage_oc`), Task 4 (`oc`, OPEN 3 text) |
  | W.9.9 P2-11 synthetic validation, mutations; 순서 0 | Task 3 (`synthetic_validation`), Task 2 (mutation tests), Task 6 (`stage_stage0`) |
  | W.4 / W.9.1 / W.9.2 / H2 / H5 / H7 verdict | Task 2 (`w_verdict`), Task 8 (`_read`) |
  | W.5 / H8 recovery | Task 8 (`stage_recompute`, `stage_invalid_run`), Task 4 (STOP_MACHINE sentence) |
  | W.6 records (gate-pair statistics, BAND, mechanism shares, floors, spillover, naive d′, noise correlations, oracle screen, C, pilot "탐색", OC tables, mixed scenario) | Task 4 (`pilot_record`), Task 3 (`records`), Task 8 (`_read`) |
  | W.7 / W.9.7 / W.9.8 sentences | Task 4 (`SENTENCES`, verbatim tests) |
  | W.8 / W.9.6 files, writes, keys, archive, 24 h budget, ledger | Task 5 (`w_store`), Task 6 (`_write`, ledger, budget), Task 8 (in-stage checker, seal) |
  | W.9.6 P2-10 / P2-11 / P2-12 job unit, RN from pre, candidate seeds, protocol table | Task 1 (seeds, phases), Task 5 (`w_learn_job`), Task 6 (`units`, stage0 tables) |
  | H9 cache key, BAND as its own artefact | Task 5 (`WMeasurer.inputs`), Task 8 (`stage_band`) |
  | W.9.9 P1-4 / P1-5 cost ranking, budget gate before the digest | Task 3 (`select`), Task 4 (`design_cost`, `budget`), Task 6 (`stage_budget`) |
  | W.9.9 순서 | Task 6 `ORDER` |

- **Placeholders:** none. Every code step has its full code; it is the code that ran in the scratch worktree.
- **Type consistency:** the blocks later stages read (`reuse.z_V`, `pilot.manifest`, `oc.selected` / `ranking`, `smoke.costs`, `budget.plan.{design, with_c, parts_h}`, `estimate.plan.parts_h`, `set.set` (+ `keys`), `oracle.pairs` / `n_testable`, `naive.screened` / `gates`, `gates.gates` / `design`, `learn` / `band` / `records` / `naive` `.manifest`, `seal.decision` / `status` / `written_at`) are written by the stages that precede them; `w_world.Model.job` returns `w_learn_job`'s fields and `FakeOracle.oracle` returns `RMeasurer.oracle`'s.
- **Review Focus:** each item has its tests in its owning task (listed in the section).

## Runs (controller)

The controller does every run, from `/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot` (branch `open-fly-brain-connectome`), with Bash `run_in_background: true` and `timeout: 7200000`, after the full suite passes at the final implementation commit.
- Subagents never run these steps.
- When a run finishes, read only its exit code and the last lines of its output.
- **No rule or number changes after any run.** A defect found after a stage is a reviewed fix task before the next stage; `w_measure.py` changing changes the W key and refuses every later stage (exit 7) — never edit it after `path`.
- Each block is committed before the next stage; the stage refuses otherwise.
- `results/summary/w_learning.json` is tracked; `results/w/` is git-ignored. `results/summary/{r,s,t,u,v}_lever.json` and `results/{r,s,t,u,v}/` are never touched.
- Commit messages carry no trailers.
- **OPEN items:** before step 0, write all seven into `/private/tmp/claude-503/p-latest.md` (each with its default and alternative) and stop; do not run stage 0 until answered. An answer that changes code is a reviewed fix task before stage 0.

**The 2 h background limit:**
- Pool stages and rough costs: `path` (~10 min, not cached — a killed run restarts and writes nothing until it finishes), `pilot` (~2.2 h; resumes per unit), `oc` (~25 min, no pool, not resumable — rerun from the start), `smoke` (~10–15 min), `oracle` (~3.1 h; resumes per pair through `RMeasurer`'s cache), `naive` (resumes per unit), `learn` / `band` / `records` (hours at large F; resume per unit).
- If a run is stopped at the limit, or dies, rerun the same command. It resumes from `results/w/cache/` with only the missing units; no block is written until a stage completes; `results/w/progress/<stage>.json` keeps the stage's cumulative wall clock for the ledger.
- Never run two pool stages at once.

**Stop points:** any gate STOP (exit 3: `STOP_REUSE`, `STOP_W_PATH_REPRO`, `STOP_PILOT_NO_EFFECT`, `STOP_OC_UNREACHABLE`, `STOP_BUDGET`, `STOP_FEW_PAIRS`); any exit 7; a gate `INVALID` (exit 5); a refusal that names missing raw data of V; a smoke with problems (exit 6) after one diagnosed fix attempt; seal `NOT_READ` / `INVALID` (exit 6); and the judgement itself. At a stop point:
1. Commit the block (a smoke block with problems is not committed; see step 5).
2. Write the spec section W.10 and the README ledger (step 14) — W.8: "관문 STOP이나 판정 결과에서 W.10 결과 절과 README 원장(ko/en)을 쓴 뒤 멈춘다".
3. Overwrite `/private/tmp/claude-503/p-latest.md` with the full text: the stage and its label or verdict; the closing sentence verbatim (and its consequence); the numbers behind it; the pilot and OC records reached (H6 values, the design, power / false-pass limits, the mixed-fly scenario's P(PASS)); the budget ledger; the commits; what was written (W.10, README ledger); the decision W.7 leaves to the user.
4. Stop. Do not continue to another stage.

0. **Preconditions (no pool):**
   1. `git status` clean, HEAD = the final implementation commit, `uv run pytest -q` passes; the OPEN answers are in (see above).
   2. V's raw data is present: `ls results/v/cache/r_arm | wc -l` → 384, `ls results/v/cache/r_oracle | wc -l` → 237; V's summary unchanged since `a279a56`: `git log -1 --format=%h -- results/summary/v_lever.json` → `a279a56`.
   3. `mkdir -p ~/flymon-archive/w`.
   4. Run `uv run python scripts/run_w.py --stage stage0` (~6 min, no pool). On exit 0, commit: `git add results/summary/w_learning.json && git commit -m "results(w): stage0 — verdict code + w_oc committed (synthetic checks pass: zero <z>, d'4 <b>, one gate <o>, anti-correlated <n>, simple normal F8->32 <p8>-><p32>); OC compute estimate <s> s; protocol/seed/design tables"`. On exit 5: commit, then **stop point**.
1. **Reuse (W.3 1):** run `uv run python scripts/run_w.py --stage reuse` (seconds). On exit 0: `git commit -am "results(w): reuse — V blocks z 928eaad / kc_input 7dc199d / set cf0b3b2 / judge a279a56 (SELECTED) on U key 8a4e0930; shared 3c2699c7, T 7255f872; z_V A 16.917/12.484, P 80.167/29.775"`. On exit 3: commit `results(w): reuse STOP_REUSE — <reasons>`, then **stop point**.
2. **Path (W.3 2):** run `uv run python scripts/run_w.py --stage path` (~10 min). On exit 0: `git commit -am "results(w): path — W job = V gate-2 punish rows (L_V, C; r1/r2; 2 seeds) bit for bit; W naive probes = V oracle report.pre (L, C rows 0-2); reward path moves PAM08-core weights and lowers MBON05(X)"`. On exit 3 / 5: commit `results(w): path <label> — <sentence or reasons>`, then **stop point**.
3. **Pilot (W.3 3, H6):** run `uv run python scripts/run_w.py --stage pilot` (~2.2 h; resumes). Report in one line: H6 shares (reward ≥ 0.5, punishment ≤ −0.5, reversed, X·Y floor) and the exploratory verdict. On exit 0: `git commit -am "results(w): pilot — 16 pairs x 8 flies x R/N/RN (K 8); H6 reward share <r>, punish share <p>, reversed <v>, X+Y floor <f>; exploratory <verdict> (탐색)"`. On exit 3 (`STOP_PILOT_NO_EFFECT`) / 5: commit, then **stop point**.
4. **OC (W.9.3, H1, W.9.9):** run `uv run python scripts/run_w.py --stage oc` (~25 min, no pool). On exit 0: `git commit -am "results(w): oc — design q <q> K <K> F <F> (k cap 8, cost <h> h), <n> qualifying designs ranked; calibration power a <a> b <b>, false a <a'> b <b'>; results/w/oc.json <sha[:8]>"`. On exit 3 (`STOP_OC_UNREACHABLE`): commit, then **stop point**.
5. **Smoke (순서 5):** run `uv run python scripts/run_w.py --stage smoke --workers 4`. Expect exit 0 and `problems: []` (L_V CSC `2d359b8b…`, edges 2, chain entry 7/2/2; RN1 = R1; pre equal across brains and with the naive probe; band weights = main; oracle on z_V). Report the `cost` in one line, then `git commit -am "results(w): smoke — pilot pair 0, 42_1xx_xxx; W path/z_V/seed/sha relations OK; cost <total h> (<parts>)"`. On exit 6: do not commit; diagnose with `superpowers:systematic-debugging`; a reviewed fix task (never `w_measure.py`, never R / T / U / V files); `git checkout -- results/summary/w_learning.json`; rerun once. A second failure is a **stop point**.
6. **Budget gate (5a):** run `uv run python scripts/run_w.py --stage budget` (no pool). On exit 0: `git commit -am "results(w): budget — plan q <q> K <K> F <F>, C <kept|dropped>; elapsed <e> h + worst remaining <r> h <= 24 h"`. On exit 3 (`STOP_BUDGET`, main set unused): commit, then **stop point**.
7. **Main set (W.2):** run `uv run python scripts/run_w.py --stage set` (no pool). On exit 0: `git commit -am "results(w): set — turns 306-<last>, (b) 167 · (a) 82, digests keys <k[:8]> / E0 b <b[:8]> / E0 a <a[:8]>; committed before any oracle on it"`. On exit 5: commit, then **stop point**.
8. **Oracle screen:** run `uv run python scripts/run_w.py --stage oracle` (~3.1 h; resumes). From here the main set is used. On exit 0: `git commit -am "results(w): oracle — 249 pairs on L_V z_V (24_700_xxx), testable <n>/249; the main set is used"`. On exit 5: commit, then **stop point**.
9. **Estimate (순서 8):** run `uv run python scripts/run_w.py --stage estimate` (no pool). On exit 0: `git commit -am "results(w): estimate — <n> testable; elapsed <e> h + remaining <r> h; C <kept|dropped>"`. On exit 3: commit, then **stop point**.
10. **Naive screen (순서 9):** run `uv run python scripts/run_w.py --stage naive` (resumes). On exit 0: `git commit -am "results(w): naive — <s> oracle-testable pairs screened in order, <g> gate pairs (|d'| < 0.5 over F x K)"`. On exit 3 (`STOP_BUDGET`): commit, then **stop point**.
11. **Gate pairs (순서 10):** run `uv run python scripts/run_w.py --stage gates` (no pool). On exit 0: `git commit -am "results(w): gates — <k> gate pairs (turn order, (b) first), design q <q> K <K> F <F>"`. On exit 3 (`STOP_FEW_PAIRS`): commit, then **stop point**.
12. **Learning, BAND probes, records (순서 11–12):** run, one per run, `uv run python scripts/run_w.py --stage learn`, then `--stage band`, then `--stage records` (resume). The blocks hold no statistic; compute nothing from `results/w/cache/` before the seal. Commit after each: `git commit -am "results(w): learn — <k> pairs x F <F> x R/N/RN, K <K> (no statistic read)"`, `"results(w): band — the same flies re-trained, probes K..2K-1 (no statistic read)"`, `"results(w): records — C <kept|dropped>, plasticity-off control (no statistic read)"`. On exit 3 (`STOP_BUDGET`): commit, then **stop point**.
13. **Seal (W.3 10):** run `uv run python scripts/run_w.py --stage seal` (no pool). On exit 0 (`SEALED`), check `find <seal.archive.dir> -type f | wc -l` equals `seal.n_files`, then `git commit -am "results(w): seal — stored inputs = declared units, manifest <n> entries, decision code + oc/budget/gates/z_V pinned, archive ~/flymon-archive/w/<dir>"`. On exit 6: commit, then **stop point** (the judgement is not read).
14. **Judge (once) and the result:**
    1. Run `uv run python scripts/run_w.py --stage judge` (no pool). If it died after writing `results/w/judge_read.json` and before the block (no `judge` block, no `results/w/judge_done.json`), rerun the same command once (`resumed_after_mark: true`). Any further failure is a **stop point**.
    2. Commit: `git commit -am "results(w): judgement — <PASS|FAIL|UNDECIDED|STOP_MACHINE> (<k> gate pairs, design q <q> K <K> F <F>, <failing pairs / reasons>)"`.
    3. Write the result (also at every gate STOP above, with what was reached):
       - the spec section `### W.10 결과 (…, <판정 1회 — **<verdict>**> | <관문 STOP — **<label>**, 판정 없음>)` at the end of appendix W, in V.10's form: implementation range and block commits; reuse; the path gate (8 P rows, 6 naive rows, reward check); the pilot (H6 values, per-pair fly medians, floors, spillover, MBON13(X) presentation-evoked suppression N2 − pre / RN2 − RN1, noise correlations, naive-floor share, |d_pre|-bin noise, the exploratory verdict marked 탐색); the OC (the model summary, calibrations, the design and its power / false-pass limits by k, the alternative ranking, the table P(PASS) at true d′ 0.5 · 1.0 · 1.5 · 2.0, the heterogeneous scenarios and **the mixed-fly scenario's P(PASS)** (W.9.9 P2-12), the notes "작동 특성은 파일럿 잡음 모형 조건부이며 Q → … → W 전체 절차의 오선택률이 아니다." "관문 쌍은 오라클 순진·시험 가능으로 고른 조건부 표본이다."; the disclosures "문턱이 d′ 1이므로 참 d′ = 1에서 관문 통과 확률은 약 절반이다" and "K는 G.6 목록에 글자로 없고 '표본 크기'로 해석한 확장이다(F.2는 프로브 8 고정)"); smoke and the budget ledger; the set; the oracle screen (testable share, d_pre / r / p table reference); the naive screen and the gate pairs; the seal; the judgement (per pair: q_sat, satisfied / BAND / failed flies, failing gates, mechanism shares, BAND resolution); the closing sentence verbatim and its consequence; the records (C, plasticity-off control); "다음: W.7대로 기록하고 사용자가 판단한다".
       - the README ledger lines, Korean (after V's bullet, which ends at ~line 207) and English (after V's bullet, which ends at ~line 358), in V's form: one bullet each with the label or verdict, the numbers and "next is the user's decision".
       - Commit: `git commit -m "docs(w): W.10 result — <verdict or STOP label> (<numbers>); README ledger ko/en"`.
    4. Go to the **stop point** with the result: the closing sentence verbatim and W.7's consequence (PASS: the M2 learning unit stood on the combined lever under the gate-pair conditions, and the POOL battle task (M3) still needs its own declaration; FAIL: M2 no-go recorded per F.7, conditional on this lever, this set and the oracle screen — STD redesign is the user's call; UNDECIDED / STOP_MACHINE: recorded for the user, never rerun on the same gate pairs (H8)). Pushing follows the standing FlyMon push rule (`gh auth switch --user lyutvs`); the verdict itself is reported, not acted on.
