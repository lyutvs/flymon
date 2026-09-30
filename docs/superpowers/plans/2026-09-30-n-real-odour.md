# Spec N — Real Odours (Hallem 2006), APL→KC Block, Lin-like Fine Discrimination: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything spec appendix N (as overridden by N.8) needs to run on the C3 engine, and nothing else:
- real odours from Hallem & Carlson 2006 (DoOR.data, pinned commit);
- the N0f feasibility grid and operating-point rule;
- the N0 similarity gate;
- the N1 punish-only oracle;
- N2.0's pilot calibration with the joint-rule operating characteristic;
- N2's absolute punishment conditioning × APL→KC block, with its verdict.

The plan also gives CLIs with `--smoke`, refusals and summary blocks, and the text writers for N.8a / N.8b. The controller runs the real stages afterwards (section "Runs (controller)").

**Architecture:**
- New files only.
  - `odor_real` is the data adapter.
  - `n_spec` holds every N number.
  - `n_store` guards every write under `results/n/` and `results/summary/n_real_odour.json`.
  - `n_jobs` holds the worker jobs, on a rig cached per worker under `(Params, edit)`.
  - `n_measure` is the content-addressed cache and measurer. N2 checkpoints once per (condition, seed).
  - `n_rules` holds the pure rules and statistics.
  - `n_oc` is the N2.0 operating-characteristic simulator.
  - `n_cli` holds the refusals and the stage skeleton the scripts share.
- Five stage scripts (`run_n0f`, `run_n0`, `run_n1`, `run_n2_pilot`, `run_n2_judge`), plus the data fetcher and the note writer.
- Each stage refuses unless:
  - the earlier blocks exist, are committed and come from the same code;
  - their outcome is a GO;
  - where N.8.9 requires it, the spec at HEAD has the dated N.8a / N.8b paragraph citing the upstream block's run id.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`; `flymon.brain.fly_pool.FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **N** (N.0–N.7) and its red-team amendment **N.8** (N.8.1–N.8.9), which wins wherever they differ:

| N.8 section | Replaces / adds |
|---|---|
| N.8.1 | claim wording ("Lin 2014 유사"), all-output block arm shown next to the verdict |
| N.8.2 | data source, duplication rule, fixture test; Lin totals become a record (N.2 gate 1 dropped) |
| N.8.3 | N0f grid, measurements, blinding, operating-point rule → `STOP_NO_OPERATING_POINT` (replaces N.2's g rule and `STOP_NO_GAIN_BAND`) |
| N.8.4 | N0 gate on a bootstrap CI of Δr, 16 activity seeds |
| N.8.5 | N1: 16 report seeds, sd > 0, raw mean o |
| N.8.6 | N2 raw-unit statistics, c₁ / δ_min / ε, the four clauses, N2.0 OC with 1× / 2× block spread, `STOP_POWER`, `STOP_BUDGET` |
| N.8.7 | `(Params, edit)` rig cache, own training loop (**no** `train_block` change), (condition, seed) checkpoints, "plumbing check" |
| N.8.8 | bistable state of every presentation; unconditional primary analysis, state-conditional record |
| N.8.9 | stop order |

## Global Constraints

- **Commits carry no trailers of any kind.** No `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder.
- **No existing file changes (N.8.7).** Every existing module, script, test and `.gitignore` stays byte-identical. That includes `conditioning.py` (N.8.7 replaces N.4/N.6's `train_block` branch with a loop inside `n_jobs`). N only adds files.
- **`data/` is git-ignored** (`.gitignore`: `data/*`). The files under `data/odor/` are committed with `git add -f`. Tracked files stay tracked, and `.gitignore` is not edited.
- **One configuration object.** Every N constant is a field of `flymon.brain.n_spec.SPEC`. H/J/L constants are read through `SPEC.l` (C3's m0d path and readout), `SPEC.h4` (= `SPEC.l.j.h4`: α grid, teach trials and timings, windows, `testable_min`, `react_zero_share_max`) and `SPEC.h3` (strength 0.35, punish type PPL105, reward type PAM08). No literal duplicates.
- **Numbers (verbatim from N / N.8):**
  - **Data.**
    - DoOR.data `ropensci/DoOR.data` @ `db323a496577c4b4a72b5c2fcd1859e07521ffb5`, column `Hallem.2006.EN`.
    - IA = isopentyl acetate (CAS 123-92-2), EB = ethyl butyrate (105-54-4), δ-DL = δ-decalactone (**705-86-2**, not γ 706-14-9).
    - 24 receptors. Or33b → DM5+DM3 is **duplicated**, and receptors that share a glomerulus add.
    - License CC BY-SA 4.0.
    - Lin 2014 totals IA 2030 · EB 1860 · δ-DL 286 are a **record only** (±1%, both conventions).
  - **Drive.**
    - `drive_hz = g × max(ΣΔ, 0)` per glomerulus. Clipped inhibition is recorded. No spontaneous rate.
    - Mixtures are linear: 4:1 = 0.8·IA + 0.2·EB, 1:4 = 0.2·IA + 0.8·EB. δ-DL is multiplied by c_δ.
    - `s = drive_hz / (max_rate_hz × strength)`, with strength = H.3/H.4's 0.35.
  - **N0f.**
    - Grid: g ∈ {0.125, 0.25, 0.5, 1, 2, 4} × c_δ ∈ {1, 2, 4, 8}.
    - Stimuli {4:1, 1:4, δ-DL, IA, EB}; conditions {on, APL→KC block, all-output block}.
    - Seeds 21_003_000 + i, i < 8.
    - Operating-point rule, all three clauses on the three judged stimuli:
      - ① on: KC median ∈ [3%, 15%];
      - ② on and block: KC median ≤ 30% and the share of presentations with sub-window > 150 Hz ≤ 1/8;
      - ③ on and block: A and P zero share each ≤ 25%.
    - Among passing points: nearest 5.54% (mean of the three on medians); ties → smaller g, then smaller c_δ. Else `STOP_NO_OPERATING_POINT`.
    - State: P < 5 is silent, ≥ 5 is firing. A state-share difference > 0.25 is flagged, not a stop.
  - **Seeds.**
    - activity 21_000_000 + i, i < **16**
    - select 21_000_100 + i, i < 8
    - report 21_000_200 + i, i < **16**
    - pilot 21_001_000 + i, i < 16
    - judge 21_002_000 + i, i < n
    - training seed = 1_000_000 + seed·1000 + trial (`train_block`'s rule)
  - **N0.** Δr = r(4:1, 1:4) − r(4:1, δ-DL) on KC firing-probability vectors. Seed bootstrap 10 000, 95% CI lower > 0 → go. Else `STOP_SIMILARITY_ORDER`.
  - **N1.**
    - Punish-only oracle, α ∈ (0.2, 0.5, 0.8). The chosen α minimises the change; ties → smaller α.
    - p0 = d′(dV_P − dV_pre). Testable ⇔ p0 ≤ −2 ∧ sd(dV_P − dV_pre) > 0.
    - o = mean(dV_P − dV_pre).
    - Both pairs testable, else `STOP_UNTESTABLE`.
  - **N2.**
    - Training: CS+ = 4:1 paired with PPL105, 12 trials, present 800, gap 200, settle 800. No CS− during training.
    - Probes: settle 800 / read 600, with the KC sub-window counted in the same presentation.
    - Statistics:
      - Δ = dV_post − dV_pre (z units), ℓ = −mean Δ.
      - D_sim = ℓ_sim,on − ℓ_sim,off; D_dis likewise.
      - Common-seed bootstrap, 10 000 draws.
    - Thresholds:
      - c₁ = 0.25·|o_pair|
      - δ_min = 0.5·ℓ̂_sim,on
      - ε = 0.25·ℓ̂_dis,on
    - Clauses:
      - ① ℓ_sim,on, ℓ_dis,on ≥ c₁ with 95% CI lower > 0
      - ② D_sim 95% CI lower ≥ δ_min
      - ③ D_dis 90% CI ⊂ [−ε, ε]
      - ④ ℓ_dis,off ≥ c₁
    - Verdicts:
      - `NOT_REPLICATED` ⇔ ① ∧ ¬(② ∧ ③ ∧ ④)
      - `NO_LEARNING` ⇔ ¬①
      - `INVALID` comes first: the plumbing check fails, or a block-condition naive probe breaks N.8.3 ② / ③.
  - **N2.0.**
    - Pilot: on only, both pairs, 16 pilot seeds.
    - OC over rules ①–④ with bootstrap. The block arm keeps the on arm's mean shifted by D and scales its deviations by 1× or 2×.
    - Null (D_sim = 0, D_dis = 0): `SUPPORTED` ≤ 0.05. Alternative (D_sim = 0.75·ℓ̂_sim,on, D_dis = 0): ≥ 0.8. Both must hold under both spreads.
    - The smallest passing n ∈ {16, 24, 32, 48, 64}, else `STOP_POWER`.
    - Judgement-block time > 48 h → `STOP_BUDGET`.
  - **Stop order (N.8.9):** `STOP_DATA_MISMATCH` → N0f → `STOP_NO_OPERATING_POINT` → N.8a committed → `STOP_SIMILARITY_ORDER` → `STOP_UNTESTABLE` → `STOP_POWER` / `STOP_BUDGET` → N.8b committed → N2 verdict.
- **Blinding.**
  - N0f never stores a KC vector: `presentation_job` returns scalars only.
  - N1 never runs APL off.
  - N2.0 reads from block `n0f` only through `n_rules.design_view` (point, wall clock, state shares, validity).
  - Block-condition KC correlations are computed only by the judge, after its verdict.
- **Every output path is guarded.**
  - Raw data and caches go under `results/n/` (git-ignored).
  - The summary is `results/summary/n_real_odour.json` (smoke: `results/n/smoke/n_real_odour.json`).
  - Everything is written through `n_store`. Anything else is refused (SystemExit 2).
- **Worker jobs and pools run only from script files with `if __name__ == "__main__":` or from pytest**, never from a heredoc (spawn re-imports the parent).
- **Subagents never write under `results/` and never start a real run or a smoke run on the connectome.**
  - Tests write only under `tmp_path`.
  - The one network step (Task 1's fetch) is the implementer's, and it writes only `data/odor/`.
  - The controller runs every real step (section "Runs (controller)").
- **Full suite before each commit:** `uv run pytest -q -rfE -o addopts=""`.
  - Baseline at `e37a888`: **1182 passed / 1 skipped / 3 xfailed**.
  - Known flake: `tests/test_run_m0d_h3.py::test_a_complete_run_writes_everything_through_both_guards_and_resumes_from_the_cache`. Rerun that file once.
  - Never run two suites at once.
- **Style:** match `m_*`. Module docstring citing the spec section, dense one-line comments, `from __future__ import annotations`.

## Readings of the spec (decided here)

1. **Activity seeds.** N.1 declares 21_000_000 + i (i < 8), and N.8.4 raises the block to i < 16. One block of 16 is used everywhere activity is measured: N0's similarity vectors, N1's f for the edit, and the judge's post-verdict KC record.
2. **Glomerular sum before clipping.** ΣΔ is summed per glomerulus first (duplicated receptors included). The mixture is taken on that sum (it is linear, so the order does not matter), and only then clipped at 0. The clipped amount is `g × max(−ΣΔ, 0)` per glomerulus.
3. **53-channel odour.** `odor_real.stimuli` returns a strength for **every** model ORN type (0 where no receptor maps). Cache keys and the fixture are therefore the full 53-channel vector.
4. **Refractory cap (record).** A receptor fires at most once per `refrac_steps() + 1` steps (`engine_cpu`: receptors honour `free`), so at dt = 1 ms the cap is 333.3 Hz. Channels commanded above it are listed as `capped`. This is a record, not a stop. At g = 4, IA's DM2 is commanded 944 Hz.
5. **The all-output block** is the CSC edit `apl_all_zero` (every APL out-edge set to 0). It is **value-equal** to building with `Params(apl_scale=0)` (test), and keeps the same `Params` so the rig cache, the guards and the C3 thresholds are untouched. Its sha256 differs from an `apl_scale=0` engine's only by the sign of zero.
6. **N2.0's block arm.** For each simulated experiment the on arm is a resample (with replacement, the same seed indices for both pairs) of the pilot's per-seed Δ. The block arm is an **independent** resample of the same pilot pool: its values are `μ + D + m·(Δ_resampled − μ)` with m ∈ {1, 2}. This assumes no within-seed correlation between on and block, which is conservative. The 2× case covers the "variance ×2" reading of decision ⑥ as well: deviations ×2 is variance ×4.
7. **OC resolution.** Each simulated experiment runs the full rules ①–④ with a 1 000-draw bootstrap (`oc_boot`), and each (n, spread, hypothesis) cell uses 2 000 experiments (`oc_draws`). The Monte-Carlo SE is recorded. The judge itself uses 10 000 bootstrap draws.
8. **Uncalibratable pilot.** If ℓ̂_sim,on ≤ 0 or ℓ̂_dis,on ≤ 0, then δ_min / ε are not positive and the alternative is meaningless. N2.0 ends `STOP_POWER` with that reason and runs no OC.
9. **Plumbing check scope.** "조건마다 같은 시드로" is read as: every condition (the four judged and the two record arms) runs its first `plumbing_seeds = 2` judge seeds a second time with plasticity off. The pre and post probes must be identical (A, P, KC spikes, both odours), and `weights_frac` must be exactly 1.0.
10. **Block validity inside N2** reads the **pre-training** probes of `sim_off` and `dis_off`: 4:1 is sim_off's X, 1:4 is sim_off's Y, δ-DL is dis_off's Y. They must meet N.8.3 ② / ③. The judge also marks the run `INVALID` if one condition's rows carry two CSC sha256s, or if an `_off` arm's sha equals its `_on` arm's (the edit changed nothing).
11. **Budget.**
    - Per (condition, seed) arm: `4·(settle + read) + trials·(train settle + present + gap)` steps, which is 27 200 at H.4's values.
    - Items: `n·6 + plumbing_seeds·6` (four judged + two record conditions).
    - Hours = `items × steps × wall_s_per_step / budget_workers / 3600`.
    - `wall_s_per_step` is N0f's median of `wall_s / steps` over every N0f presentation. Engine builds are not counted.
    - `budget_workers` = 16.
12. **N.8a / N.8b gates.** "N.8a 커밋" is checked mechanically. `git show HEAD:<spec>` must have a line beginning `**N.8a` (or a heading `### N.8a`), and the text after it must contain block `n0f`'s `run_id`. N.8b works the same way with block `n2_0`. `scripts/write_n_notes.py` prints both paragraphs with their run ids.
13. **Smoke.**
    - `n_spec.smoke(SPEC)` uses a 1 × 1 grid, a separate seed block `21_009_xxx` (never a declared stage's), 2–3 seeds per role, n = 3, 200 bootstrap draws and a tiny OC.
    - `--smoke` skips the commit, outcome and note gates; the bypassed outcomes are recorded in `smoke_bypass`.
    - Smoke reads and writes only `results/n/smoke/`.
    - If smoke N0f selects no point, downstream smoke stages use `spec.smoke_point = (1.0, 1.0)`.
14. **Exit codes.** 0 = a GO outcome or any N2 verdict. 5 = a `STOP_*` (the block is written; stop and ask the user). 2 = refusal (nothing written).

## Review Focus

1. **An APL-on job run on a worker whose cached rig is the APL→KC-blocked one (or the reverse).** Expect every row to carry its own arm's CSC sha256. On → block → on in one worker must give identical first and third results, and a block job must equal a fresh block rig. Test in Tasks 4 and 5 (mixed-worker regression) and Task 6 (on and block never share a cache entry).
2. **DoOR names and neighbours.** γ-decalactone sits next to δ-decalactone in every receptor file, IA is listed as "isopentyl acetate", and DoOR's master moves. Expect CAS matching, a pinned commit, pinned sha256 of both CSVs, and `STOP_DATA_MISMATCH` on any NA, unmapped receptor or missing glomerulus. Test in Tasks 1 and 3.
3. **Training seeds of ~2.1·10¹⁰ (> 2³²).** Expect them to reach `np.random.default_rng` unchanged: the stream equals `default_rng(seed)` and differs from `default_rng(seed mod 2³²)`, and JSON cache keys keep the exact integer. Test in Task 2.
4. **Degenerate data at the stats edges.** Examples: sd(ΔdV) = 0 with a negative mean (d′ = −∞), a pilot whose ℓ̂ ≤ 0, and KC vectors that never fire (Pearson undefined). Expect "untestable", `STOP_POWER` and `STOP_SIMILARITY_ORDER` respectively, never a crash or a pass. Test in Tasks 7 and 8.
5. **Blinding leaks.** Examples: an N0f row carrying a KC vector, N2.0 reading N0f's block KC numbers beyond the whitelist, the judge computing block KC correlations before its verdict. Expect no `kc_fired` key in any presentation row, `design_view`'s exact key set, and the judge's call order (arms, verdict, then block KC vectors). Test in Tasks 4, 8 and 10.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/fetch_door_hallem.py` | fetch the pinned DoOR.data files, extract 24 × {IA, EB, δ-DL} by CAS and the mapping rows, check against the model's ORN types, write `data/odor/` |
| `data/odor/hallem2006_subset.csv`, `door_mappings_subset.csv`, `LICENSE-CC-BY-SA-4.0.txt`, `NOTICE`, `provenance.json` | the pinned data (force-added) |
| `flymon/brain/n_spec.py` | `NSpec` (every N number, seed blocks, data pins) + `smoke()` |
| `flymon/brain/n_store.py` | guarded writes under `results/n/` and `results/summary/n_real_odour.json` |
| `flymon/brain/odor_real.py` | `load_table`, `glomerular`, `receptor_totals`, `lin_record`, `mix`, `drive`, `strengths`, `cap_hz`, `stimuli`, `DataMismatch` |
| `flymon/brain/n_jobs.py` | `EDITS`, `apply_edit`, `rig`, `readout_cells`, `presentation_job`, `kc_vectors_job`, `pick_alpha`, `punish_only_oracle_job`, `train_plus_only`, `absolute_arm_job` |
| `flymon/brain/n_measure.py` | `MEASURE_FILES`, `HASHED_FILES`, `NCache`, `NMeasurer` |
| `flymon/brain/n_rules.py` | outcomes, `state`, `cell_stats`, `wall_per_step`, `point_key`, `point_checks`, `select_point`, `state_flags`, `similarity`, `n1_pair`, `n1_outcome`, `delta`, `deltas`, `n2_stats`, `verdict`, `plumbing`, `block_validity`, `csc_checks`, `dprime_record`, `state_conditional`, `thresholds`, `budget_hours`, `design_view`, `n2_0_outcome`, `sentence` |
| `flymon/brain/n_oc.py` | `off_arm`, `simulate`, `choose_n` |
| `flymon/brain/n_cli.py` | `ORDER`, `out_allowed`, `read_previous`, `later_blocks`, `spec_note`, `head_spec`, hooks (`git_state`, `code_keys`, `load_c3`, `m0d_sha`, `model_types`, `make_measurer`), `operating_point`, `stimuli_at`, `write_report`, `Ctx`, `main_stage` |
| `scripts/run_n0f.py`, `run_n0.py`, `run_n1.py`, `run_n2_pilot.py`, `run_n2_judge.py` | the five stages |
| `scripts/write_n_notes.py` | print the dated N.8a / N.8b paragraphs from the summary |
| `tests/test_fetch_door_hallem.py`, `tests/brain/test_n_spec_store.py`, `tests/brain/test_odor_real.py`, `tests/brain/test_n_jobs.py`, `tests/brain/test_n_measure.py`, `tests/brain/test_n_rules.py`, `tests/brain/test_n_verdict_oc.py`, `tests/test_run_n.py`, `tests/test_write_n_notes.py` | tests |

Task order: 1 data → 2 spec/store → 3 adapter → 4–5 jobs → 6 measurer → 7–8 rules/OC → 9–10 CLIs → 11 notes. Later tasks consume only names listed in earlier tasks' **Produces**.

---

### Task 1: The pinned data — `scripts/fetch_door_hallem.py` and `data/odor/`

**Files:**
- Create: `scripts/fetch_door_hallem.py`
- Create (by running the script): `data/odor/hallem2006_subset.csv`, `data/odor/door_mappings_subset.csv`, `data/odor/LICENSE-CC-BY-SA-4.0.txt`, `data/odor/NOTICE`, `data/odor/provenance.json`
- Test: `tests/test_fetch_door_hallem.py`

**Interfaces:**
- Consumes: `flymon.brain.connectome.Connectome`, `flymon.brain.circuits.Populations` (model ORN types, only in `main`).
- Produces (script module, loaded by path in tests): `SHA`, `ODORANTS`, `Mismatch`, `parse(text) -> (header, rows)`, `receptor_values(files: dict[str, str]) -> dict`, `receptor_glomeruli(mapping_text, receptors) -> dict`, `check_model(glomeruli, model_types) -> None`, `render(values, glomeruli) -> (table_text, mapping_text)`, `extract(files, mapping_text, model_types, n_expected=24) -> (table_text, mapping_text)`, `main(argv=None) -> int`.
- Produces (files): the two CSVs with these exact bytes (sha256 pinned in Task 2):
  - `hallem2006_subset.csv` → `d65f2711c73cc7d6e7f547e4e65c0eef5bfaac56b7f08ff4410d69e50c471445`
  - `door_mappings_subset.csv` → `cacf48b235936086f269ddf2dfc1c4e62f3fa2547a410ad4ccc7e98bc2c60005`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch_door_hallem.py
"""Spec N.8.2: odorants by CAS (never by name; δ-decalactone is not γ), 24 receptors with a Hallem.2006.EN column, one
mapping row each, no "?" glomerulus, every glomerulus a model ORN type; the committed data files are the pinned bytes."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE = """receptor,IA,EB,dDL
Or10a,222,43,3
Or19a,110,102,6
Or22a,236,197,15
Or23a,30,24,7
Or2a,73,20,8
Or33b,31,62,40
Or35a,39,138,7
Or43a,38,43,5
Or43b,98,201,4
Or47a,123,75,1
Or47b,9,31,43
Or49b,19,5,4
Or59b,48,41,4
Or65a,28,10,13
Or67a,140,178,13
Or67c,54,118,6
Or7a,-16,18,2
Or82a,55,43,11
Or85a,-2,139,15
Or85b,244,115,37
Or85f,34,33,6
Or88a,25,17,25
Or98a,260,93,20
Or9a,142,124,1
"""
MAPPING = """receptor,glomerulus
Or10a,DL1
Or19a,DC1
Or22a,DM2
Or23a,DA3
Or2a,DA4m
Or33b,DM5+DM3
Or35a,VC3
Or43a,DA4l
Or43b,VM2
Or47a,DM3
Or47b,VA1v
Or49b,VA5
Or59b,DM4
Or65a,DL3
Or67a,DM6
Or67c,VC4
Or7a,DL5
Or82a,VA6
Or85a,DM5
Or85b,VM5d
Or85f,DL4
Or88a,VA1d
Or98a,VM5v
Or9a,VM3
"""
TABLE_SHA = "d65f2711c73cc7d6e7f547e4e65c0eef5bfaac56b7f08ff4410d69e50c471445"
MAPPING_SHA = "cacf48b235936086f269ddf2dfc1c4e62f3fa2547a410ad4ccc7e98bc2c60005"
HEAD = '"Class";"Name";"InChIKey";"CID";"CAS";"Hallem.2006.EN";"Other.2001"\n'


def _script():
    sp = importlib.util.spec_from_file_location("fetch_door_hallem", ROOT / "scripts" / "fetch_door_hallem.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules["fetch_door_hallem"] = mod
    sp.loader.exec_module(mod)
    return mod


def _receptor(ia="236", eb="197", ddl="15", gamma_only=False):
    rows = [f'"1";NA;"sfr";"SFR";"SFR";"SFR";4;NA',
            f'"2";"ester";"isopentyl acetate";"MLF";"31276";"123-92-2";{ia};NA',
            f'"3";"ester";"ethyl butyrate";"OBN";"7762";"105-54-4";{eb};NA',
            f'"4";"O ring";"gamma-decalactone";"IFY";"12813";"706-14-9";20;NA']
    if not gamma_only:
        rows.append(f'"5";"O ring";"delta-decalactone";"GHB";"12810";"705-86-2";{ddl};NA')
    return HEAD + "\n".join(rows) + "\n"


MAP_HEAD = ('"receptor";"sensillum";"OSN";"glomerulus";"co.receptor";"coexpressing";"related1";"related2";"related3";'
            '"related4";"related5";"related6";"Ors";"sensillum.type";"adult";"larva";"dataset.existing";"comment";"code";'
            '"code.OSN"\n')


def _map_row(i, rec, glom):
    return f'"{i}";"{rec}";"ab";"ab1A";"{glom}";"Orco";"";"";"";"";"";"";"";"{rec}";"x";TRUE;TRUE;TRUE;"";"{glom}";"ab1A"\n'


def test_values_are_matched_by_cas_and_delta_is_not_gamma():
    m = _script()
    got = m.receptor_values({"Or22a": _receptor()})
    assert got == {"Or22a": {"IA": "236", "EB": "197", "dDL": "15"}}
    with pytest.raises(m.Mismatch, match="dDL"):
        m.receptor_values({"Or22a": _receptor(gamma_only=True)})          # only γ present: a stop, never a substitute


def test_an_na_value_is_a_mismatch():
    m = _script()
    with pytest.raises(m.Mismatch, match="EB"):
        m.receptor_values({"Or22a": _receptor(eb="NA")})


def test_a_file_without_the_column_or_all_na_is_not_a_receptor():
    m = _script()
    no_col = '"Class";"Name";"InChIKey";"CID";"CAS";"Other.2001"\n"1";"ester";"x";"1";"123-92-2";4\n'
    all_na = _receptor(ia="NA", eb="NA", ddl="NA").replace(";4;NA", ";NA;NA").replace(";20;NA", ";NA;NA")
    assert m.receptor_values({"a": no_col, "b": all_na}) == {}


def test_mapping_duplicates_are_kept_raw_and_unmapped_receptors_stop():
    m = _script()
    text = MAP_HEAD + _map_row(1, "Or33b", "DM5+DM3") + _map_row(2, "Or22a", "DM2")
    assert m.receptor_glomeruli(text, ["Or33b", "Or22a"]) == {"Or33b": "DM5+DM3", "Or22a": "DM2"}
    with pytest.raises(m.Mismatch, match="unmapped"):
        m.receptor_glomeruli(MAP_HEAD + _map_row(1, "Or22a", "?"), ["Or22a"])
    with pytest.raises(m.Mismatch, match="mapping rows"):
        m.receptor_glomeruli(MAP_HEAD + _map_row(1, "Or22a", "DM2"), ["Or22a", "Or7a"])


def test_a_glomerulus_the_model_lacks_is_a_mismatch():
    m = _script()
    m.check_model({"Or33b": "DM5+DM3"}, ["ORN_DM5", "ORN_DM3"])
    with pytest.raises(m.Mismatch, match="DM3"):
        m.check_model({"Or33b": "DM5+DM3"}, ["ORN_DM5"])


def test_extract_counts_receptors_and_renders_sorted_csv():
    m = _script()
    files = {"Or22a": _receptor(), "Or7a": _receptor(ia="-16", eb="18", ddl="2")}
    mapping = MAP_HEAD + _map_row(1, "Or22a", "DM2") + _map_row(2, "Or7a", "DL5")
    table, mp = m.extract(files, mapping, ["ORN_DM2", "ORN_DL5"], n_expected=2)
    assert table == "receptor,IA,EB,dDL\nOr22a,236,197,15\nOr7a,-16,18,2\n"
    assert mp == "receptor,glomerulus\nOr22a,DM2\nOr7a,DL5\n"
    with pytest.raises(m.Mismatch, match="expected 24"):
        m.extract(files, mapping, ["ORN_DM2", "ORN_DL5"])


def test_render_of_the_24_receptors_is_the_pinned_bytes():
    m = _script()
    rows = [ln.split(",") for ln in TABLE.splitlines()[1:]]
    values = {r[0]: {"IA": r[1], "EB": r[2], "dDL": r[3]} for r in rows}
    glom = dict(ln.split(",") for ln in MAPPING.splitlines()[1:])
    table, mp = m.render(values, glom)
    assert (table, mp) == (TABLE, MAPPING)
    assert hashlib.sha256(table.encode()).hexdigest() == TABLE_SHA
    assert hashlib.sha256(mp.encode()).hexdigest() == MAPPING_SHA


def test_the_committed_files_are_the_pinned_bytes_with_license_and_provenance():
    d = ROOT / "data" / "odor"
    assert (d / "hallem2006_subset.csv").read_text() == TABLE
    assert (d / "door_mappings_subset.csv").read_text() == MAPPING
    assert "CC BY-SA 4.0" in (d / "NOTICE").read_text()
    assert "Attribution-ShareAlike 4.0" in (d / "LICENSE-CC-BY-SA-4.0.txt").read_text()
    import json
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["commit"] == "db323a496577c4b4a72b5c2fcd1859e07521ffb5"
    assert prov["outputs"]["hallem2006_subset.csv"] == TABLE_SHA
    assert {k: v["cas"] for k, v in prov["odorants"].items()} == {"IA": "123-92-2", "EB": "105-54-4",
                                                                  "dDL": "705-86-2"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fetch_door_hallem.py -q -o addopts=""`
Expected: FAIL (`FileNotFoundError` for `scripts/fetch_door_hallem.py`).

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""Spec N.8.2: fetch Hallem & Carlson 2006 (DoOR.data column `Hallem.2006.EN`) for isopentyl acetate (IA), ethyl
butyrate (EB) and δ-decalactone (δ-DL) on the receptors that carry that column, with their `door_mappings.csv`
glomerulus, from one pinned DoOR.data commit, and write data/odor/.

    uv run python scripts/fetch_door_hallem.py          # network; then: git add -f data/odor

Odorants are matched by CAS, never by name: DoOR calls IA "isopentyl acetate", and δ-decalactone (705-86-2) sits next
to γ-decalactone (706-14-9) in every receptor file. Exit 5 (STOP_DATA_MISMATCH, nothing written) on a missing or NA
value, a receptor count other than 24, a receptor without exactly one mapping row, an unmapped glomerulus ("?" or
empty) or a glomerulus the model has no ORN_* type for. Exit 2 on a network error. data/ is git-ignored: the files are
added with `git add -f` (.gitignore is not edited, N.8.7). Data rows of a DoOR CSV carry a leading row id, so header
column j is field j + 1."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "ropensci/DoOR.data"
SHA = "db323a496577c4b4a72b5c2fcd1859e07521ffb5"           # master as of 2026-09-30 (commit of 2026-07-17)
RAW = f"https://raw.githubusercontent.com/{REPO}/{SHA}/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"
COLUMN = "Hallem.2006.EN"
ODORANTS = (("IA", "123-92-2"), ("EB", "105-54-4"), ("dDL", "705-86-2"))
N_RECEPTORS = 24
NOT_RECEPTOR = ("door_", "odor", "ORs")                     # data/*.csv tables that are not one receptor's file
NOTICE = f"""data/odor/ is derived from DoOR.data (https://github.com/{REPO}), commit {SHA},
licensed CC BY-SA 4.0 (its DESCRIPTION: "License: CC BY-SA 4.0").
Changes: only the {COLUMN} column for isopentyl acetate (CAS 123-92-2), ethyl butyrate (CAS 105-54-4) and
delta-decalactone (CAS 705-86-2) on the 24 receptors that carry it, and those receptors' glomerulus column of
door_mappings.csv, reformatted as comma-separated files. These files are distributed under CC BY-SA 4.0
(LICENSE-CC-BY-SA-4.0.txt). provenance.json records every source file's sha256.
Sources: Hallem & Carlson 2006, Cell 125:143-160; Muench & Galizia 2016, Sci Rep 6:21841 (DoOR 2.0).
"""


class Mismatch(Exception):
    """STOP_DATA_MISMATCH (N.8.2)."""


def parse(text: str) -> tuple:
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    return rows[0], rows[1:]


def receptor_values(files: dict) -> dict:
    """{receptor: {"IA", "EB", "dDL": value text}} for every file whose Hallem.2006.EN column has a non-NA value."""
    out = {}
    for name, text in sorted(files.items()):
        header, rows = parse(text)
        if COLUMN not in header or "CAS" not in header:
            continue
        j, cas_j = header.index(COLUMN) + 1, header.index("CAS") + 1
        if all(len(r) <= j or r[j] in ("NA", "") for r in rows):
            continue
        vals = {}
        for key, cas in ODORANTS:
            hit = [r for r in rows if len(r) > j and r[cas_j] == cas]
            if len(hit) != 1 or hit[0][j] in ("NA", ""):
                raise Mismatch(f"{name}: {key} (CAS {cas}) has no {COLUMN} value")
            float(hit[0][j])                                    # a non-numeric value raises ValueError: not data
            vals[key] = hit[0][j]
        out[name] = vals
    return out


def receptor_glomeruli(mapping_text: str, receptors) -> dict:
    header, rows = parse(mapping_text)
    ri, gi = header.index("receptor") + 1, header.index("glomerulus") + 1
    by: dict = {}
    for r in rows:
        by.setdefault(r[ri], []).append(r[gi])
    out = {}
    for rec in receptors:
        g = by.get(rec) or []
        if len(g) != 1:
            raise Mismatch(f"receptor {rec} has {len(g)} mapping rows (need exactly 1)")
        if any(p.strip() in ("", "?") for p in g[0].split("+")):
            raise Mismatch(f"receptor {rec} is unmapped ({g[0]!r})")
        out[rec] = g[0]
    return out


def check_model(glomeruli: dict, model_types) -> None:
    have = {str(t) for t in model_types}
    miss = sorted({p for v in glomeruli.values() for p in v.split("+") if "ORN_" + p not in have})
    if miss:
        raise Mismatch(f"glomeruli {miss} have no ORN_* type in the model")


def render(values: dict, glomeruli: dict) -> tuple:
    recs = sorted(values)
    table = "receptor,IA,EB,dDL\n" + "".join(
        f"{r},{values[r]['IA']},{values[r]['EB']},{values[r]['dDL']}\n" for r in recs)
    mapping = "receptor,glomerulus\n" + "".join(f"{r},{glomeruli[r]}\n" for r in recs)
    return table, mapping


def extract(files: dict, mapping_text: str, model_types, n_expected: int = N_RECEPTORS) -> tuple:
    values = receptor_values(files)
    if len(values) != n_expected:
        raise Mismatch(f"{len(values)} receptors carry {COLUMN}, expected {n_expected}")
    glomeruli = receptor_glomeruli(mapping_text, values)
    check_model(glomeruli, model_types)
    return render(values, glomeruli)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flymon-fetch-door/1.0"})   # creativecommons.org 403s urllib's default
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _odorant_ids(text: str) -> dict:
    header, rows = parse(text)
    cas_j, name_j, ink_j = (header.index(k) + 1 for k in ("CAS", "Name", "InChIKey"))
    out = {}
    for key, cas in ODORANTS:
        r = next(r for r in rows if r[cas_j] == cas)
        out[key] = dict(cas=cas, door_name=r[name_j], inchikey=r[ink_j])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default="data/odor")
    a = ap.parse_args(argv)
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    types = sorted(Populations.from_connectome(Connectome.load(a.npz)).receptor_types)
    try:
        tree = json.loads(_get(TREE))["tree"]
        paths = sorted(t["path"] for t in tree if t["path"].startswith("data/") and t["path"].endswith(".csv")
                       and not Path(t["path"]).stem.startswith(NOT_RECEPTOR))
        raw = {p: _get(RAW + p) for p in paths + ["data/door_mappings.csv"]}
        license_text = _get(LICENSE_URL)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"refused: network error {e!r}", file=sys.stderr)
        return 2
    files = {Path(p).stem: raw[p].decode() for p in paths}
    try:
        table, mapping = extract(files, raw["data/door_mappings.csv"].decode(), types)
    except (Mismatch, ValueError) as e:
        print(f"STOP_DATA_MISMATCH: {e}", file=sys.stderr)
        return 5
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hallem2006_subset.csv").write_text(table)
    (out / "door_mappings_subset.csv").write_text(mapping)
    (out / "LICENSE-CC-BY-SA-4.0.txt").write_bytes(license_text)
    (out / "NOTICE").write_text(NOTICE)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    prov = dict(repo=REPO, commit=SHA, column=COLUMN, retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                odorants=_odorant_ids(files["Or22a"]), sources={p: sha(b) for p, b in sorted(raw.items())},
                license=dict(name="CC BY-SA 4.0", declared_in="DESCRIPTION", text_url=LICENSE_URL,
                             text_sha256=sha(license_text)),
                outputs={n: sha((out / n).read_bytes()) for n in ("hallem2006_subset.csv", "door_mappings_subset.csv",
                                                                  "LICENSE-CC-BY-SA-4.0.txt", "NOTICE")},
                citations=["Hallem EA, Carlson JR (2006) Cell 125:143-160",
                           "Muench D, Galizia CG (2016) Sci Rep 6:21841"])
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: {table.count(chr(10)) - 1} receptors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the offline tests (all but the committed-files test)**

Run: `uv run pytest tests/test_fetch_door_hallem.py -q -o addopts="" -k "not committed"`
Expected: PASS (7 tests).

- [ ] **Step 5: Fetch the data (network; this writes only `data/odor/`)**

Run: `uv run python scripts/fetch_door_hallem.py`
Expected: `wrote data/odor: 24 receptors`, exit 0. Exit 5 (`STOP_DATA_MISMATCH`) or bytes that differ from `TABLE` / `MAPPING` mean the pinned commit's data is not what this plan read on 2026-09-30. In that case, **stop and report BLOCKED** with the message and a `diff`. Do not change the pins or the fixture.

- [ ] **Step 6: Run the tests (all pass)**

Run: `uv run pytest tests/test_fetch_door_hallem.py -q -o addopts=""`
Expected: PASS (8 tests).

- [ ] **Step 7: Full suite, then commit (data force-added)**

```bash
uv run pytest -q -rfE -o addopts=""
git add scripts/fetch_door_hallem.py tests/test_fetch_door_hallem.py
git add -f data/odor/hallem2006_subset.csv data/odor/door_mappings_subset.csv data/odor/LICENSE-CC-BY-SA-4.0.txt data/odor/NOTICE data/odor/provenance.json
git commit -m "feat(n): pinned Hallem 2006 subset from DoOR.data db323a4 (24 receptors x IA/EB/dDL by CAS, mapping, CC BY-SA notice)"
```

---

### Task 2: `n_spec`, `n_store` and the seed-width check

**Files:**
- Create: `flymon/brain/n_spec.py`, `flymon/brain/n_store.py`
- Test: `tests/brain/test_n_spec_store.py`

**Interfaces:**
- Consumes: `l_spec.SPEC` (→ `.j.h4`, `.j.h4.h3`, `.m0d_path`, `.readout`); `h3_store.canonical_pretty`, `sha256_file`; `pool_bench.refuse_old_engine_output`, `refuse_modified_engine_output`.
- Produces:
  - `NSpec`, `SPEC`, `smoke(spec) -> NSpec`.
  - Properties `.h4`, `.h3`, `.zero_share_max`, `.testable_min`. Methods `.sha_pins() -> dict | None`, `.mixture(name) -> dict`, `.mixtures_dict() -> dict`, `.edit_of(cond) -> str`, `.pair_stimuli() -> dict[str, tuple]`, `.judge_seeds(n) -> tuple`, `.train_seed(seed, trial) -> int`.
  - `n_store.ALLOWED_DIR = "results/n/"`, `SUMMARY = "results/summary/n_real_odour.json"`, `guard(path, params_list)`, `write_bytes`, `write_json`, `write_summary_block(path, block, obj, params_list)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_n_spec_store.py
"""Spec N.1 / N.8: every N number in one object; seed blocks new and disjoint; training seeds (~2.1e10) reach the
engine's generator unchanged; data pins equal the committed files; writes only under results/n/ and the N summary."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import n_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_store import canonical, sha256_file
from flymon.brain.l_spec import SPEC as L_SPEC
from flymon.brain.n_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]


def _declared(spec):
    return [spec.act_seeds, spec.select_seeds, spec.report_seeds, spec.pilot_seeds, spec.judge_seeds(max(spec.n_grid)),
            spec.n0f_seeds]


def test_declared_sizes_and_grids():
    assert (len(SPEC.act_seeds), len(SPEC.select_seeds), len(SPEC.report_seeds), len(SPEC.pilot_seeds),
            len(SPEC.n0f_seeds)) == (16, 8, 16, 16, 8)
    assert (SPEC.act_seeds[0], SPEC.select_seeds[0], SPEC.report_seeds[0], SPEC.pilot_seeds[0], SPEC.judge_seed0,
            SPEC.n0f_seeds[0]) == (21_000_000, 21_000_100, 21_000_200, 21_001_000, 21_002_000, 21_003_000)
    assert SPEC.g_grid == (0.125, 0.25, 0.5, 1.0, 2.0, 4.0) and SPEC.c_delta_grid == (1.0, 2.0, 4.0, 8.0)
    assert SPEC.n_grid == (16, 24, 32, 48, 64) and SPEC.sd_mults == (1.0, 2.0)
    assert SPEC.judge_seeds(3) == (21_002_000, 21_002_001, 21_002_002)
    assert SPEC.mixture("4:1") == {"IA": 0.8, "EB": 0.2} and SPEC.mixture("1:4") == {"IA": 0.2, "EB": 0.8}
    assert SPEC.pair_stimuli() == {"sim": ("4:1", "1:4"), "dis": ("4:1", "dDL")}
    assert SPEC.edit_of("on") == "none" and SPEC.edit_of("block") == "apl_to_kc_zero"
    assert SPEC.edit_of("all") == "apl_all_zero"


def test_seed_blocks_are_new_and_disjoint_and_training_seeds_never_meet_them():
    flat = [s for b in _declared(SPEC) for s in b]
    assert len(flat) == len(set(flat))
    h4 = SPEC.h4
    old = set(h4.act_seeds) | set(h4.select_seeds) | set(h4.report_seeds) | set(h4.teach_seeds)
    assert not old & set(flat)
    train = [SPEC.train_seed(s, t) for s in flat for t in range(h4.teach_trials)]
    assert len(set(train)) == len(train) and min(train) > max(flat)


def test_training_seed_is_train_blocks_rule():
    assert SPEC.train_seed(21_002_063, 11) == 1_000_000 + 21_002_063 * 1000 + 11 == 21_003_063_011


def test_big_seeds_reach_the_engine_generator_unchanged(synthetic_connectome):
    c = synthetic_connectome()
    e = Engine(c, Populations.from_connectome(c), Params(min_weight=1, balance_hemispheres=False), seed=0)
    big = SPEC.train_seed(max(SPEC.judge_seeds(max(SPEC.n_grid))), SPEC.h4.teach_trials - 1)
    assert big > 2 ** 32
    e.reset(big)
    a = e.rng.random()
    assert a == np.random.default_rng(big).random()
    e.reset(big % 2 ** 32)
    assert e.rng.random() != a                                     # no silent 32-bit truncation
    assert json.loads(canonical({"s": big}))["s"] == big           # cache keys keep the exact integer


def test_constants_come_through_l_and_h4():
    assert SPEC.l is L_SPEC and SPEC.h4 is L_SPEC.j.h4 and SPEC.h3 is L_SPEC.j.h4.h3
    assert (SPEC.zero_share_max, SPEC.testable_min) == (0.25, 2.0)
    assert (SPEC.h4.teach_trials, SPEC.h4.teach_present_ms, SPEC.h4.teach_gap_ms) == (12, 800.0, 200.0)
    assert SPEC.h3.strength == 0.35 and SPEC.h3.punish_type == "PPL105"


def test_data_pins_are_the_committed_files():
    for name, sha in SPEC.sha_pins().items():
        assert sha256_file(ROOT / SPEC.data_dir / name) == sha


def test_smoke_is_small_and_outside_every_declared_block():
    s = smoke(SPEC)
    assert s.g_grid == (1.0,) and s.c_delta_grid == (1.0,) and s.n_grid == (3,)
    declared = {x for b in _declared(SPEC) for x in b}
    smoke_seeds = [x for b in _declared(s) for x in b]
    assert not declared & set(smoke_seeds) and len(smoke_seeds) == len(set(smoke_seeds))
    assert s.sha_pins() == SPEC.sha_pins()                          # smoke runs on the real, pinned data


def test_store_writes_only_under_results_n_and_the_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = Params()
    n_store.write_json("results/n/x/a.json", {"a": 1}, [p])
    assert json.loads(Path("results/n/x/a.json").read_text()) == {"a": 1}
    n_store.write_summary_block(n_store.SUMMARY, "n0f", {"b": 2}, [p])
    n_store.write_summary_block(n_store.SUMMARY, "n0", {"c": 3}, [p])
    assert json.loads(Path(n_store.SUMMARY).read_text()) == {"n0f": {"b": 2}, "n0": {"c": 3}}
    for bad in ("results/m0d/n/a.json", "results/summary/m_readout.json", "results/n_other.json"):
        with pytest.raises(SystemExit):
            n_store.write_json(bad, {}, [p])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_n_spec_store.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.n_spec`).

- [ ] **Step 3: Write `n_spec.py`**

```python
"""The configuration of spec appendix N as amended by N.8 (N.8 wins where they differ): C3 unchanged; Hallem 2006 real
odours (odor_real, data/odor/ pinned by sha256); N0f's feasibility grid and operating-point rule (N.8.3); N0's
similarity gate (N.8.4); N1's punish-only oracle (N.3, N.8.5); N2.0's pilot and joint-rule OC and N2's absolute
punishment conditioning x APL->KC block (N.4, N.8.6). `l` carries C3's m0d path and readout; H.4 / H.3 constants come
through `h4` / `h3` (= l.j.h4 / l.j.h4.h3). Nothing here repeats one of them."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .l_spec import SPEC as L_SPEC, LSpec


@dataclass(frozen=True)
class NSpec:
    l: LSpec = L_SPEC
    # ---- data (N.8.2) ------------------------------------------------------------------------------------------------
    data_dir: str = "data/odor"
    data_sha256: tuple = (("hallem2006_subset.csv", "d65f2711c73cc7d6e7f547e4e65c0eef5bfaac56b7f08ff4410d69e50c471445"),
                          ("door_mappings_subset.csv",
                           "cacf48b235936086f269ddf2dfc1c4e62f3fa2547a410ad4ccc7e98bc2c60005"))
    door_repo: str = "ropensci/DoOR.data"
    door_commit: str = "db323a496577c4b4a72b5c2fcd1859e07521ffb5"
    lin_totals: tuple = (("IA", 2030.0), ("EB", 1860.0), ("dDL", 286.0))   # Lin et al. 2014, a record (N.8.2)
    lin_tol: float = 0.01
    # ---- stimuli (N.2) -------------------------------------------------------------------------------------------------
    mixtures: tuple = (("4:1", (("IA", 0.8), ("EB", 0.2))), ("1:4", (("IA", 0.2), ("EB", 0.8))),
                       ("dDL", (("dDL", 1.0),)), ("IA", (("IA", 1.0),)), ("EB", (("EB", 1.0),)))
    n0f_stimuli: tuple = ("4:1", "1:4", "dDL", "IA", "EB")
    judged_stimuli: tuple = ("4:1", "1:4", "dDL")
    pairs: tuple = (("sim", "4:1", "1:4"), ("dis", "4:1", "dDL"))
    # ---- N0f (N.8.3) ---------------------------------------------------------------------------------------------------
    g_grid: tuple = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    c_delta_grid: tuple = (1.0, 2.0, 4.0, 8.0)
    conditions: tuple = (("on", "none"), ("block", "apl_to_kc_zero"), ("all", "apl_all_zero"))
    n0f_seeds: tuple = tuple(range(21_003_000, 21_003_008))
    kc_band: tuple = (0.03, 0.15)                 # ① on: each judged stimulus' KC median, inclusive
    kc_target: float = 0.0554                     # C3's H.3 reference
    kc_block_max: float = 0.30                    # ② on and block
    runaway_hz: float = 150.0                     # D.6 (a) sub-window threshold
    runaway_share_max: float = 0.125              # ② presentations above it, at most 1/8
    state_p_min: int = 5                          # N.8.8: P < 5 silent, >= 5 firing
    state_diff_flag: float = 0.25
    # ---- N0 (N.8.4) ----------------------------------------------------------------------------------------------------
    act_seeds: tuple = tuple(range(21_000_000, 21_000_016))    # i < 16 (N.8.4); reading 1
    boot_draws: int = 10_000
    boot_seed: int = 20260930
    # ---- N1 (N.3, N.8.5) -----------------------------------------------------------------------------------------------
    select_seeds: tuple = tuple(range(21_000_100, 21_000_108))
    report_seeds: tuple = tuple(range(21_000_200, 21_000_216))
    # ---- N2.0 / N2 (N.4, N.8.6) -----------------------------------------------------------------------------------------
    pilot_seeds: tuple = tuple(range(21_001_000, 21_001_016))
    judge_seed0: int = 21_002_000
    n2_conditions: tuple = (("sim_on", "sim", "none"), ("sim_off", "sim", "apl_to_kc_zero"),
                            ("dis_on", "dis", "none"), ("dis_off", "dis", "apl_to_kc_zero"))
    n2_record_conditions: tuple = (("sim_all", "sim", "apl_all_zero"), ("dis_all", "dis", "apl_all_zero"))
    plumbing_seeds: int = 2                       # reading 9
    train_seed_base: int = 1_000_000              # conditioning.train_block's rule
    train_seed_stride: int = 1000
    c1_frac: float = 0.25
    delta_min_frac: float = 0.5
    eps_frac: float = 0.25
    alt_frac: float = 0.75
    n_grid: tuple = (16, 24, 32, 48, 64)
    sd_mults: tuple = (1.0, 2.0)
    null_max: float = 0.05
    power_min: float = 0.8
    oc_draws: int = 2000                          # reading 7
    oc_boot: int = 1000
    oc_seed: int = 20261001
    budget_h: float = 48.0
    budget_workers: int = 16
    # ---- where -----------------------------------------------------------------------------------------------------------
    spec_path: str = "docs/superpowers/specs/2026-09-14-flymon-design.md"
    smoke_point: tuple = (1.0, 1.0)               # reading 13

    @property
    def h4(self):
        return self.l.j.h4

    @property
    def h3(self):
        return self.l.j.h4.h3

    @property
    def zero_share_max(self) -> float:
        return self.h4.react_zero_share_max

    @property
    def testable_min(self) -> float:
        return self.h4.testable_min

    def sha_pins(self) -> dict | None:
        return dict(self.data_sha256) if self.data_sha256 else None

    def mixture(self, name: str) -> dict:
        return dict(dict(self.mixtures)[name])

    def mixtures_dict(self) -> dict:
        return {n: dict(w) for n, w in self.mixtures}

    def edit_of(self, cond: str) -> str:
        return dict(self.conditions)[cond]

    def pair_stimuli(self) -> dict:
        return {n: (x, y) for n, x, y in self.pairs}

    def judge_seeds(self, n: int) -> tuple:
        return tuple(range(self.judge_seed0, self.judge_seed0 + int(n)))

    def train_seed(self, seed: int, trial: int) -> int:
        return self.train_seed_base + int(seed) * self.train_seed_stride + int(trial)


SPEC = NSpec()


def smoke(spec: NSpec) -> NSpec:
    """A 1 x 1 grid, a seed block of its own (21_009_xxx, never a declared stage's), 2-3 seeds per role, n = 3, few
    bootstrap draws and a tiny OC. The data pins stay: smoke runs on the real data."""
    return dataclasses.replace(
        spec, g_grid=(1.0,), c_delta_grid=(1.0,), n0f_seeds=(21_009_000, 21_009_001),
        act_seeds=(21_009_010, 21_009_011, 21_009_012), select_seeds=(21_009_020, 21_009_021),
        report_seeds=(21_009_030, 21_009_031, 21_009_032), pilot_seeds=(21_009_040, 21_009_041, 21_009_042),
        judge_seed0=21_009_050, n_grid=(3,), plumbing_seeds=1, boot_draws=200, oc_draws=20, oc_boot=50)
```

- [ ] **Step 4: Write `n_store.py`**

```python
"""Guarded writes of spec appendix N: raw data, caches and reports under results/n/ and the summary
results/summary/n_real_odour.json; nothing else (m_store's rule with N's paths). Every write is atomic (tmp +
os.replace)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/n/"
SUMMARY = "results/summary/n_real_odour.json"


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: spec N writes only under {ALLOWED_DIR} and {SUMMARY}", file=sys.stderr)
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_n_spec_store.py -q -o addopts=""`
Expected: PASS (8 tests).

- [ ] **Step 6: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_spec.py flymon/brain/n_store.py tests/brain/test_n_spec_store.py
git commit -m "feat(n): n_spec (every N/N.8 number, seed blocks, data pins, smoke) and n_store (results/n/ guard)"
```

---

### Task 3: `odor_real` — Hallem responses → 53-channel ORN drive

**Files:**
- Create: `flymon/brain/odor_real.py`
- Test: `tests/brain/test_odor_real.py`

**Interfaces:**
- Consumes: `h3_store.sha256_file`; `n_spec.SPEC` (tests only: `sha_pins()`, `mixtures_dict()`, `lin_totals`, `lin_tol`, `data_dir`); `stimuli.present`.
- Produces:
  - `ODORANTS = ("IA", "EB", "dDL")`, `TABLE_FILE`, `MAP_FILE`, `class DataMismatch(ValueError)`.
  - `Table(delta: dict, glomeruli: dict)`, `load_table(data_dir, sha256: dict | None) -> Table`.
  - `glomerular(table, model_types) -> dict[str, dict[str, float]]`.
  - `receptor_totals(table) -> {"signed": {...}, "positive": {...}}`.
  - `lin_record(table, lin_totals: dict, tol) -> dict`.
  - `mix(glom, weights) -> dict[str, float]`.
  - `drive(delta, g) -> (drive_hz: dict, clipped_hz: dict)`.
  - `strengths(drive_hz, max_rate_hz, strength) -> dict`.
  - `cap_hz(params) -> float`.
  - `stimuli(glom, names, g, c_delta, mixtures: dict, max_rate_hz, strength, cap) -> dict[name, {"odor", "drive_hz", "clipped_hz", "clipped_total_hz", "weights", "g", "c_delta", "capped"}]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_odor_real.py
"""Spec N.2 / N.8.2: the hand-computed 53-channel fixture (Or33b duplicated to DM5 and DM3, receptors on one glomerulus
added), linear mixtures, c_δ on δ-DL only, clipping recorded, s = drive_hz / (max_rate_hz x strength), the Lin totals
as a record, and DataMismatch on every data defect."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import odor_real as O
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC
from flymon.brain.stimuli import present

ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPES = [
    "ORN_D", "ORN_DA1", "ORN_DA2", "ORN_DA3", "ORN_DA4l", "ORN_DA4m", "ORN_DC1", "ORN_DC2", "ORN_DC3", "ORN_DC4",
    "ORN_DL1", "ORN_DL2d", "ORN_DL2v", "ORN_DL3", "ORN_DL4", "ORN_DL5", "ORN_DM1", "ORN_DM2", "ORN_DM3", "ORN_DM4",
    "ORN_DM5", "ORN_DM6", "ORN_DP1l", "ORN_DP1m", "ORN_V", "ORN_VA1d", "ORN_VA1v", "ORN_VA2", "ORN_VA3", "ORN_VA4",
    "ORN_VA5", "ORN_VA6", "ORN_VA7l", "ORN_VA7m", "ORN_VC1", "ORN_VC2", "ORN_VC3", "ORN_VC4", "ORN_VC5", "ORN_VL1",
    "ORN_VL2a", "ORN_VL2p", "ORN_VM1", "ORN_VM2", "ORN_VM3", "ORN_VM4", "ORN_VM5d", "ORN_VM5v", "ORN_VM6l",
    "ORN_VM6m", "ORN_VM6v", "ORN_VM7d", "ORN_VM7v"]
# ΣΔ per glomerulus (IA, EB, δ-DL), summed by hand from the committed table: DM5 = Or85a + Or33b, DM3 = Or47a + Or33b
NONZERO = {
    "ORN_DA3": (30, 24, 7), "ORN_DA4l": (38, 43, 5), "ORN_DA4m": (73, 20, 8), "ORN_DC1": (110, 102, 6),
    "ORN_DL1": (222, 43, 3), "ORN_DL3": (28, 10, 13), "ORN_DL4": (34, 33, 6), "ORN_DL5": (-16, 18, 2),
    "ORN_DM2": (236, 197, 15), "ORN_DM3": (154, 137, 41), "ORN_DM4": (48, 41, 4), "ORN_DM5": (29, 201, 55),
    "ORN_DM6": (140, 178, 13), "ORN_VA1d": (25, 17, 25), "ORN_VA1v": (9, 31, 43), "ORN_VA5": (19, 5, 4),
    "ORN_VA6": (55, 43, 11), "ORN_VC3": (39, 138, 7), "ORN_VC4": (54, 118, 6), "ORN_VM2": (98, 201, 4),
    "ORN_VM3": (142, 124, 1), "ORN_VM5d": (244, 115, 37), "ORN_VM5v": (260, 93, 20)}
CAP = 1000.0 / 3.0


@pytest.fixture
def glom():
    return O.glomerular(O.load_table(ROOT / SPEC.data_dir, SPEC.sha_pins()), MODEL_TYPES)


def test_the_53_channel_vector_is_the_hand_computed_fixture(glom):
    assert list(glom) == MODEL_TYPES and len(glom) == 53
    for t in MODEL_TYPES:
        assert tuple(glom[t][k] for k in O.ODORANTS) == pytest.approx(NONZERO.get(t, (0, 0, 0)))
    assert len(NONZERO) == 23                                   # 24 receptors, DM5 and DM3 shared, Or33b on both


def test_or33b_is_duplicated_not_split(glom):
    assert glom["ORN_DM5"]["EB"] == 139 + 62 and glom["ORN_DM3"]["EB"] == 75 + 62


def test_mixture_drive_clip_and_strength_at_g1(glom):
    st = O.stimuli(glom, ("4:1", "1:4", "IA"), 1.0, 1.0, SPEC.mixtures_dict(), 200.0, 0.35, CAP)
    assert st["4:1"]["drive_hz"]["ORN_DM2"] == pytest.approx(0.8 * 236 + 0.2 * 197)       # 228.2
    assert st["4:1"]["drive_hz"]["ORN_DL5"] == 0.0
    assert st["4:1"]["clipped_hz"] == {"ORN_DL5": pytest.approx(9.2)}                      # 0.8 * -16 + 0.2 * 18
    assert st["1:4"]["drive_hz"]["ORN_DL5"] == pytest.approx(11.2) and st["1:4"]["clipped_hz"] == {}
    assert st["IA"]["clipped_hz"] == {"ORN_DL5": pytest.approx(16.0)}     # DM5's IA is -2 + 31 = 29 > 0: not clipped
    assert st["4:1"]["odor"]["ORN_DM2"] == pytest.approx(228.2 / (200.0 * 0.35))
    assert set(st["4:1"]["odor"]) == set(MODEL_TYPES)
    assert all(v == 0.0 for t, v in st["4:1"]["drive_hz"].items() if t not in NONZERO)


def test_c_delta_multiplies_only_ddl_and_g_scales_everything(glom):
    st = O.stimuli(glom, ("dDL", "4:1"), 0.5, 2.0, SPEC.mixtures_dict(), 200.0, 0.35, CAP)
    assert st["dDL"]["drive_hz"]["ORN_DM5"] == pytest.approx(55 * 2.0 * 0.5)
    assert st["4:1"]["drive_hz"]["ORN_DM2"] == pytest.approx(228.2 * 0.5)
    assert st["dDL"]["weights"] == {"dDL": 2.0} and st["4:1"]["weights"] == {"IA": 0.8, "EB": 0.2}


def test_channels_above_the_refractory_cap_are_listed(glom):
    assert O.cap_hz(Params()) == pytest.approx(1000.0 / 3.0)
    st = O.stimuli(glom, ("IA",), 4.0, 1.0, SPEC.mixtures_dict(), 200.0, 0.35, CAP)
    assert "ORN_DM2" in st["IA"]["capped"] and "ORN_DA3" not in st["IA"]["capped"]     # 944 Hz vs 120 Hz


def test_present_commands_drive_hz_on_every_cell_of_the_type(glom, synthetic_connectome):
    from flymon.brain.circuits import Populations
    pops = Populations.from_connectome(synthetic_connectome())

    class Eng:
        p = Params()
        drive_hz = np.zeros(200, np.float32)

    small = {"ORN_DM1": {"IA": 100.0, "EB": 0.0, "dDL": 0.0}, "ORN_DA1": {"IA": -5.0, "EB": 0.0, "dDL": 0.0}}
    st = O.stimuli(small, ("IA",), 1.0, 1.0, SPEC.mixtures_dict(), Eng.p.max_rate_hz, 0.35, CAP)
    present(Eng, pops, st["IA"]["odor"], 0.35)
    assert np.allclose(Eng.drive_hz[pops.receptor_types["ORN_DM1"]], 100.0)
    assert np.all(Eng.drive_hz[pops.receptor_types["ORN_DA1"]] == 0.0)


def test_lin_totals_are_recorded_in_both_conventions():
    rec = O.lin_record(O.load_table(ROOT / SPEC.data_dir, SPEC.sha_pins()), dict(SPEC.lin_totals), SPEC.lin_tol)
    assert rec["totals"]["signed"] == {"IA": 2040.0, "EB": 1870.0, "dDL": 296.0}
    assert rec["totals"]["positive"] == {"IA": 2058.0, "EB": 1870.0, "dDL": 296.0}
    assert rec["rel_error"]["signed"]["dDL"] == pytest.approx(10 / 286)
    assert rec["within_tol"] == {"signed": False, "positive": False}   # δ-DL is +3.5%: a record, not a stop (N.8.2)


def _copy(tmp_path):
    d = tmp_path / "odor"
    shutil.copytree(ROOT / SPEC.data_dir, d)
    return d


def test_a_changed_file_is_a_data_mismatch(tmp_path):
    d = _copy(tmp_path)
    (d / O.TABLE_FILE).write_text((d / O.TABLE_FILE).read_text().replace("Or22a,236", "Or22a,237"))
    with pytest.raises(O.DataMismatch, match="sha256"):
        O.load_table(d, SPEC.sha_pins())


@pytest.mark.parametrize("edit, match", [
    (("t", "Or22a,236,197,15", "Or22a,NA,197,15"), "Or22a"),
    (("t", "Or22a,236,197,15", "Or22a,236,,15"), "Or22a"),
    (("m", "Or22a,DM2", "Or22a,?"), "unmapped"),
    (("m", "Or22a,DM2\n", ""), "without a mapping"),
    (("t", "Or9a,142,124,1\n", "Or9a,142,124,1\nOr9a,1,1,1\n"), "twice"),
])
def test_data_defects_are_data_mismatches(tmp_path, edit, match):
    d = _copy(tmp_path)
    f = d / (O.TABLE_FILE if edit[0] == "t" else O.MAP_FILE)
    f.write_text(f.read_text().replace(edit[1], edit[2]))
    with pytest.raises(O.DataMismatch, match=match):
        O.load_table(d, None)


def test_a_glomerulus_the_model_lacks_is_a_data_mismatch():
    t = O.load_table(ROOT / SPEC.data_dir, SPEC.sha_pins())
    with pytest.raises(O.DataMismatch, match="DM3"):
        O.glomerular(t, [x for x in MODEL_TYPES if x != "ORN_DM3"])


@pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="needs the connectome")
def test_model_types_are_the_real_models():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    types = sorted(str(t) for t in Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz")).receptor_types)
    assert types == MODEL_TYPES
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_odor_real.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.odor_real`).

- [ ] **Step 3: Write `odor_real.py`**

```python
"""Spec N.2 as amended by N.8.2: Hallem & Carlson 2006 responses (data/odor/, from DoOR.data) -> the model's ORN drive.

Receptor -> glomerulus from the mapping file. A receptor listed on several glomeruli ("DM5+DM3") is duplicated to each
(co-expression: every listed ORN class carries it); receptors on one glomerulus add. Mixtures are linear in Δ
(4:1 = 0.8 IA + 0.2 EB); δ-DL is multiplied by c_δ. drive_hz = g x max(ΣΔ, 0) per glomerulus: inhibition is clipped and
the clipped amount recorded; no spontaneous rate. present() gets s = drive_hz / (max_rate_hz x strength), for every
model ORN type (0 where nothing maps). Values are read only from the data files; a file whose sha256 differs from its
pin, a missing or non-numeric value, a receptor without a mapping, an unmapped glomerulus or one the model lacks raises
DataMismatch (STOP_DATA_MISMATCH, the only data stop of N0). Channels commanded above the receptors' refractory cap are
listed (a record)."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .h3_store import sha256_file

ODORANTS = ("IA", "EB", "dDL")
TABLE_FILE = "hallem2006_subset.csv"
MAP_FILE = "door_mappings_subset.csv"


class DataMismatch(ValueError):
    """STOP_DATA_MISMATCH (N.8.2)."""


@dataclass(frozen=True)
class Table:
    delta: dict          # receptor -> {"IA", "EB", "dDL": Δ spikes/s}
    glomeruli: dict      # receptor -> tuple of glomerulus names (no "ORN_" prefix)


def load_table(data_dir, sha256: dict | None) -> Table:
    d = Path(data_dir)
    for name in (TABLE_FILE, MAP_FILE):
        if not (d / name).exists():
            raise DataMismatch(f"{d / name} is missing")
        if sha256 is not None and sha256_file(d / name) != sha256.get(name):
            raise DataMismatch(f"{d / name}: sha256 differs from the pin")
    delta: dict = {}
    with open(d / TABLE_FILE, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["receptor", *ODORANTS]:
            raise DataMismatch(f"{TABLE_FILE}: header {r.fieldnames}")
        for row in r:
            rec = row["receptor"]
            if rec in delta:
                raise DataMismatch(f"receptor {rec} listed twice")
            try:
                vals = {k: float(row[k]) for k in ODORANTS}
            except (TypeError, ValueError):
                raise DataMismatch(f"receptor {rec}: a missing or non-numeric value") from None
            if not all(math.isfinite(v) for v in vals.values()):
                raise DataMismatch(f"receptor {rec}: a non-finite value")
            delta[rec] = vals
    glom: dict = {}
    with open(d / MAP_FILE, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["receptor", "glomerulus"]:
            raise DataMismatch(f"{MAP_FILE}: header {r.fieldnames}")
        for row in r:
            parts = tuple(p.strip() for p in (row["glomerulus"] or "").split("+"))
            if any(p in ("", "?") for p in parts):
                raise DataMismatch(f"receptor {row['receptor']} is unmapped ({row['glomerulus']!r})")
            glom[row["receptor"]] = parts
    if set(glom) != set(delta):
        raise DataMismatch(f"receptors without a mapping {sorted(set(delta) - set(glom))}, "
                           f"mappings without values {sorted(set(glom) - set(delta))}")
    return Table(delta=delta, glomeruli=glom)


def glomerular(table: Table, model_types) -> dict:
    """{"ORN_<glom>": {"IA", "EB", "dDL": ΣΔ}} over every model ORN type, in the given order (0 where nothing maps)."""
    out = {str(t): {k: 0.0 for k in ODORANTS} for t in model_types}
    for rec in sorted(table.delta):
        for gl in table.glomeruli[rec]:
            t = "ORN_" + gl
            if t not in out:
                raise DataMismatch(f"receptor {rec} maps to {gl}, which the model has no ORN type for")
            for k in ODORANTS:
                out[t][k] += table.delta[rec][k]
    return out


def receptor_totals(table: Table) -> dict:
    """Sums over the 24 receptors before any glomerular assignment: Σ Δ and Σ max(Δ, 0)."""
    return {"signed": {k: float(sum(v[k] for v in table.delta.values())) for k in ODORANTS},
            "positive": {k: float(sum(max(v[k], 0.0) for v in table.delta.values())) for k in ODORANTS}}


def lin_record(table: Table, lin_totals: dict, tol: float) -> dict:
    tot = receptor_totals(table)
    rel = {c: {k: (tot[c][k] - lin_totals[k]) / lin_totals[k] for k in ODORANTS} for c in tot}
    return dict(totals=tot, lin=dict(lin_totals), rel_error=rel, tol=tol,
                within_tol={c: all(abs(v) <= tol for v in rel[c].values()) for c in rel}, note="record only (N.8.2)")


def mix(glom: dict, weights: dict) -> dict:
    return {t: float(sum(weights.get(k, 0.0) * v[k] for k in ODORANTS)) for t, v in glom.items()}


def drive(delta: dict, g: float) -> tuple:
    hz = {t: float(g) * max(v, 0.0) for t, v in delta.items()}
    clipped = {t: float(g) * -v for t, v in delta.items() if v < 0}
    return hz, clipped


def strengths(drive_hz: dict, max_rate_hz: float, strength: float) -> dict:
    return {t: v / (max_rate_hz * strength) for t, v in drive_hz.items()}


def cap_hz(params) -> float:
    """A receptor fires at most once per refrac_steps() + 1 steps (engine_cpu: receptors honour `free`)."""
    return 1000.0 / (params.dt * (params.refrac_steps() + 1))


def stimuli(glom: dict, names, g: float, c_delta: float, mixtures: dict, max_rate_hz: float, strength: float,
            cap: float) -> dict:
    out = {}
    for n in names:
        w = {k: v * (c_delta if k == "dDL" else 1.0) for k, v in mixtures[n].items()}
        hz, clipped = drive(mix(glom, w), g)
        out[n] = dict(odor=strengths(hz, max_rate_hz, strength), drive_hz=hz, clipped_hz=clipped,
                      clipped_total_hz=float(sum(clipped.values())), weights=w, g=float(g), c_delta=float(c_delta),
                      capped=sorted(t for t, v in hz.items() if v > cap))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_odor_real.py -q -o addopts=""`
Expected: PASS (15 tests; the model-types test runs when `data/malecns.npz` exists).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/odor_real.py tests/brain/test_odor_real.py
git commit -m "feat(n): odor_real adapter (duplication, linear mixtures, c_delta, clip record, s conversion, 53-channel fixture)"
```

---
### Task 4: `n_jobs` (1) — the `(Params, edit)` rig, presentations, KC vectors

**Files:**
- Create: `flymon/brain/n_jobs.py`
- Test: `tests/brain/test_n_jobs.py`

**Interfaces:**
- Consumes: `engine_cpu.Engine`, `plasticity.Plasticity`, `circuits.compartments`, `h3_jobs.edge_sources`, `h4_jobs._present_kc`, `h4_jobs.type_cells`, `presentation.decide`, `stimuli.present`.
- Produces:
  - `EDITS`, `_RIG`, `apply_edit(eng, pops, edit) -> str` (sha256), `rig(conn, pops, params, edit) -> (Engine, Plasticity, comps, sha)`.
  - `readout_cells(conn, readout) -> (a_cells, p_cells)`.
  - `_present(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells) -> (row, read_vector)`. The row keys are exactly `PRESENTATION_KEYS = ("seed", "A", "P", "kc_frac", "kc_spikes", "kc_max_win_hz", "apl_out_per_step", "wall_s", "steps")`.
  - `presentation_job(eng, pl, pops, comps, ro, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms) -> list[row + edit + csc_sha256]`.
  - `kc_vectors_job(...same...) -> list[row + edit + csc_sha256 + n_kc + kc_fired]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_n_jobs.py
"""Spec N.8.3 / N.8.7: the rig is cached per worker under (Params, edit); "apl_to_kc_zero" zeroes exactly the APL->KC
edges, "apl_all_zero" equals Params(apl_scale=0) by value; graded-APL views follow the edit; APL-on and APL-block jobs
mixed in one worker never see each other's engine; a presentation row is scalars only (blinding) and its KC and readout
counts are h4_jobs._present_kc's and presentation.decide's."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_jobs import edge_sources
from flymon.brain.h4_jobs import _present_kc, rig_for
from flymon.brain.presentation import decide

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
PG = replace(P, apl_mode="graded")
RO = {"A": "MBON03", "P": "MBON01"}                  # synthetic: PPL105 core = MBON03/04, PAM08 core = MBON01/02
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
                w=np.append(c.w, np.int32(20)))                 # one APL -> MBON edge, so the two edits differ
    N._RIG.clear(); H4._RIG.clear()                     # h4_jobs.rig_for is keyed by Params only: never share it
    return c, Populations.from_connectome(c)


def _apl_masks(e, pops):
    src, tgt = edge_sources(e.csc), e.csc.tgt.astype(np.int64)
    is_apl = np.isin(src, pops.apl)
    return is_apl & np.isin(tgt, pops.kc), is_apl & ~np.isin(tgt, pops.kc)


def test_apl_to_kc_zero_zeroes_exactly_the_apl_kc_edges(conn_pops):
    c, pops = conn_pops
    e0 = Engine(c, pops, P, seed=0)
    e1 = Engine(c, pops, P, seed=0)
    sha = N.apply_edit(e1, pops, "apl_to_kc_zero")
    to_kc, to_other = _apl_masks(e1, pops)
    assert to_kc.any() and to_other.any()
    assert np.all(e1.csc.w[to_kc] == 0) and np.all(e0.csc.w[to_kc] != 0)
    assert np.array_equal(e1.csc.w[~to_kc], e0.csc.w[~to_kc])
    assert sha != N.apply_edit(e0, pops, "none")


def test_apl_all_zero_equals_apl_scale_zero_by_value(conn_pops):
    c, pops = conn_pops
    e = Engine(c, pops, P, seed=0)
    N.apply_edit(e, pops, "apl_all_zero")
    assert np.array_equal(e.csc.w, Engine(c, pops, replace(P, apl_scale=0.0), seed=0).csc.w)
    with pytest.raises(ValueError, match="unknown edit"):
        N.apply_edit(e, pops, "apl_half")


def test_graded_apl_views_follow_the_edit(conn_pops):
    c, pops = conn_pops
    e, _, _, _ = N.rig(c, pops, PG, "apl_to_kc_zero")
    is_kc = np.zeros(e.N, bool); is_kc[pops.kc] = True
    for tgt, w in e._apl_edges:
        assert np.all(w[is_kc[tgt]] == 0) and np.any(w[~is_kc[tgt]] != 0)


def test_rig_is_keyed_by_params_and_edit_and_holds_one_entry(conn_pops):
    c, pops = conn_pops
    a = N.rig(c, pops, P, "none")
    b = N.rig(c, pops, P, "apl_to_kc_zero")
    assert a[0] is not b[0] and a[3] != b[3] and len(N._RIG) == 1
    assert N.rig(c, pops, P, "apl_to_kc_zero")[0] is b[0]


@pytest.mark.parametrize("params", [P, PG])
def test_on_block_on_in_one_worker(conn_pops, params):
    c, pops = conn_pops
    kw = dict(params=params, odor=X, seeds=[3, 4], readout=RO, **W)
    on1 = N.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    blk = N.presentation_job(_Stub(c), None, pops, None, None, edit="apl_to_kc_zero", **kw)
    on2 = N.presentation_job(_Stub(c), None, pops, None, None, edit="none", **kw)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_s"} for r in rows]
    assert strip(on1) == strip(on2)
    N._RIG.clear()
    fresh = N.presentation_job(_Stub(c), None, pops, None, None, edit="apl_to_kc_zero", **kw)
    assert strip(blk) == strip(fresh)
    assert {r["csc_sha256"] for r in on1} != {r["csc_sha256"] for r in blk}
    assert [r["edit"] for r in blk] == ["apl_to_kc_zero"] * 2


def test_a_presentation_row_is_scalars_only(conn_pops):
    c, pops = conn_pops
    rows = N.presentation_job(_Stub(c), None, pops, None, None, params=P, edit="apl_to_kc_zero", odor=X, seeds=[3],
                              readout=RO, **W)
    assert set(rows[0]) == set(N.PRESENTATION_KEYS) | {"edit", "csc_sha256"}
    assert all(np.isscalar(v) for v in rows[0].values())


def test_kc_and_readout_counts_are_present_kc_and_decide(conn_pops):
    c, pops = conn_pops
    row = N.presentation_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor=X, seeds=[7],
                             readout=RO, **W)[0]
    e, p, _ = rig_for(c, pops, P)                                     # h4's rig: same Params, unedited engine
    p.set_enabled(False)
    ref = _present_kc(e, p, pops, X, 7, W["strength"], W["settle_ms"], W["read_ms"], W["window_ms"])
    t = np.asarray(c.type).astype(str)
    cnt = decide(e, p, pops, [X], W["strength"], 7, W["settle_ms"], W["read_ms"])[0]
    assert row["kc_frac"] == float((ref["read"] > 0).mean()) and row["kc_spikes"] == int(ref["read"].sum())
    assert row["kc_max_win_hz"] == ref["max_win"] * 1000.0 / W["window_ms"]
    assert (row["A"], row["P"]) == (int(cnt[t == "MBON03"].sum()), int(cnt[t == "MBON01"].sum()))
    assert row["steps"] == 150


def test_kc_vectors_carry_the_fired_positions(conn_pops):
    c, pops = conn_pops
    rows = N.kc_vectors_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor=X, seeds=[3, 4],
                            readout=RO, **W)
    for r in rows:
        assert r["n_kc"] == len(pops.kc) and len(r["kc_fired"]) == round(r["kc_frac"] * len(pops.kc))
        assert all(0 <= i < len(pops.kc) for i in r["kc_fired"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_n_jobs.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.n_jobs`).

- [ ] **Step 3: Write `n_jobs.py` (part 1)**

```python
"""FlyPool worker jobs of spec appendix N (N.8.3, N.3 / N.8.5, N.4 / N.8.7). Signature fn(engine, plasticity, pops,
comps, readout, **kwargs), module-level so the spawn pool can pickle them; the worker's default engine only lends its
connectome.

The rig (Engine, Plasticity, compartments) is cached per worker under (Params, edit), edit in EDITS:
- "none";
- "apl_to_kc_zero": every APL -> KC edge 0, N.7's primary block;
- "apl_all_zero": every APL out-edge 0, the record arm, value-equal to Params(apl_scale=0).
The edit is made in place on a freshly built engine's CSC (the graded-APL out-edge views follow:
h3_jobs.apply_csc_edit's precedent). The sha256 of the weight vector goes with every row. No existing module changes
(N.8.7).

presentation_job: one presentation per seed (reset, clear_drive, present, settle, read), plasticity off. It returns
scalars only:
- KC active fraction and spikes of the read window;
- the largest 200 ms sliding-window KC count over the whole presentation, in Hz (h4_jobs._present_kc's window);
- the readout A / P read counts, the APL output per step and the wall time.
It never returns a KC vector (N.8.3 blinding). kc_vectors_job adds the fired KC positions; N0 calls it with APL on, and
N2's judge only after its verdict."""
from __future__ import annotations

import hashlib
import time
from collections import deque

import numpy as np

from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .h4_formula import dprime, dv
from .h4_jobs import _present_kc, type_cells
from .plasticity import Plasticity
from .presentation import decide
from .stimuli import present

EDITS = ("none", "apl_to_kc_zero", "apl_all_zero")
PRESENTATION_KEYS = ("seed", "A", "P", "kc_frac", "kc_spikes", "kc_max_win_hz", "apl_out_per_step", "wall_s", "steps")
_RIG: dict = {}          # (Params, edit) -> (Engine, Plasticity, comps, csc sha256); one entry per worker


def apply_edit(eng, pops, edit: str) -> str:
    """Zero the edit's APL out-edges in the engine's CSC in place; the sha256 of the weight vector."""
    if edit not in EDITS:
        raise ValueError(f"unknown edit {edit!r}; known: {EDITS}")
    if edit != "none":
        is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
        m = is_apl[edge_sources(eng.csc)]
        if edit == "apl_to_kc_zero":
            is_kc = np.zeros(eng.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
            m &= is_kc[eng.csc.tgt.astype(np.int64)]
        eng.csc.w[m] = np.float32(0.0)
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


def rig(conn, pops, params, edit: str):
    """The worker's rig for (params, edit), built once and reused while jobs keep asking for it (one entry: a switch of
    arm rebuilds, it never edits a cached engine)."""
    key = (params, edit)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha = apply_edit(eng, pops, edit)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha)
    return _RIG[key]


def readout_cells(conn, readout: dict) -> tuple:
    cells = type_cells(conn, (readout["A"], readout["P"]))
    return cells[readout["A"]], cells[readout["P"]]


def _present(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells) -> tuple:
    """One presentation (presentation._fresh, then settle + read steps, as decide and _present_kc step them)."""
    kc = pops.kc
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    apl = np.asarray(pops.apl, np.int64)
    graded = e.p.apl_mode == "graded"
    t0 = time.perf_counter()
    e.reset(int(seed)); p.reset_traces(); e.clear_drive(); p.quiet_dan()
    present(e, pops, odor, strength)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32); counts = np.zeros(e.N, np.int32); apl_out = 0.0
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
            counts[fired] += 1
            if graded:                                  # reading the membrane does not change the engine state
                apl_out += float(e.apl_release(e.v[apl]).sum())
    n_read = n_total - n_settle
    if not graded:                                      # spiking APL: its read-window spikes
        apl_out = float(counts[apl].sum())
    row = dict(seed=int(seed), A=int(counts[a_cells].sum()), P=int(counts[p_cells].sum()),
               kc_frac=float((read > 0).mean()), kc_spikes=int(read.sum()),
               kc_max_win_hz=max_win * 1000.0 / (window_ms * e.p.dt),
               apl_out_per_step=apl_out / (n_read * max(apl.size, 1)), wall_s=time.perf_counter() - t0,
               steps=int(n_total))
    return row, read


def _rows(eng, pops, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms, keep_kc: bool):
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = readout_cells(e.conn, readout)
    p.reset_weights()
    was = p.enabled
    p.set_enabled(False)
    out = []
    try:
        for s in seeds:
            row, read = _present(e, p, pops, odor, s, strength, settle_ms, read_ms, window_ms, a_cells, p_cells)
            row = dict(row, edit=edit, csc_sha256=sha)
            if keep_kc:
                row.update(n_kc=int(len(pops.kc)), kc_fired=np.flatnonzero(read > 0).astype(int).tolist())
            out.append(row)
    finally:
        p.set_enabled(was)
    return out


def presentation_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                     settle_ms: float, read_ms: float, window_ms: int) -> list:
    """N0f's presentations: scalars only, never a KC vector (N.8.3)."""
    return _rows(eng, pops, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms, False)


def kc_vectors_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                   settle_ms: float, read_ms: float, window_ms: int) -> list:
    """The same presentations with the fired KC positions of the read window (N0 with APL on; the judge after its
    verdict)."""
    return _rows(eng, pops, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms, True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_n_jobs.py -q -o addopts=""`
Expected: PASS (9 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_jobs.py tests/brain/test_n_jobs.py
git commit -m "feat(n): n_jobs rig cached per (Params, edit), APL->KC / all-output edits, scalar-only presentations, KC vectors"
```

---

### Task 5: `n_jobs` (2) — punish-only oracle and the absolute conditioning arm

**Files:**
- Modify: `flymon/brain/n_jobs.py` (append; the file is new in this plan)
- Test: `tests/brain/test_n_jobs_arms.py`

**Interfaces:**
- Consumes: Task 4's `rig`, `readout_cells`, `_present`; `h4_formula.dprime`, `dv`; `h4_jobs._present_kc`; `presentation.decide`.
- Produces:
  - `pick_alpha(changes: dict[str, float | None], alphas) -> float`.
  - `punish_only_oracle_job(eng, pl, pops, comps, ro, params, odor_x, odor_y, readout, z, act_seeds, select_seeds, report_seeds, alphas, strength, settle_ms, read_ms, window_ms, punish_type) -> {"alpha_punish", "select": {"pre", "punish": {str(α): {"P", "change"}}}, "report": {"pre", "P"}, "kc": {"x", "y", "jaccard"}, "csc_sha256"}`. Every probe is `{"A": [[x, y] per seed], "P": [[x, y] per seed]}`.
  - `train_plus_only(e, p, pops, cs_plus, strength, seed, punish, trials, present_ms, gap_ms, settle_ms, seed_base, seed_stride) -> None`.
  - `absolute_arm_job(eng, pl, pops, comps, ro, params, edit, odor_x, odor_y, seed, plastic, readout, punish_type, strength, settle_ms, read_ms, window_ms, trials, present_ms, gap_ms, train_settle_ms, seed_base, seed_stride) -> {"seed", "edit", "plastic", "csc_sha256", "pre": {"x": row, "y": row}, "post": {...}, "weights_frac", "wall_s"}`. Each row is a `PRESENTATION_KEYS` row.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_n_jobs_arms.py
"""Spec N.3 / N.8.5: the punish-only oracle is h4_jobs.oracle_job's punishment half (same unedited probes, α by the
smallest change, weights restored). Spec N.4 / N.8.7: the absolute arm trains the CS+ alone with train_block's seed rule
and timings, probes with decide's counts plus the KC sub-window, and with plasticity off changes nothing (plumbing)."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain.circuits import Populations
from flymon.brain.conditioning import train_block
from flymon.brain.config import Params
from flymon.brain.h4_jobs import oracle_job
from flymon.brain.presentation import decide

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}
Z = {"A": [1.0, 2.0], "P": [1.5, 3.0]}
X, Y = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)
ORACLE = dict(act_seeds=(500, 501), select_seeds=(600, 601, 602), report_seeds=(608, 609, 610),
              alphas=(0.2, 0.5, 0.8), **W)
ARM = dict(readout=RO, punish_type="PPL105", trials=2, present_ms=300.0, gap_ms=50.0, train_settle_ms=100.0,
           seed_base=1_000_000, seed_stride=1000, **W)


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    N._RIG.clear(); H4._RIG.clear()                     # h4_jobs.rig_for is keyed by Params only: never share it
    return c, Populations.from_connectome(c)


def test_pick_alpha_takes_the_smallest_change_and_ties_to_the_smaller_alpha():
    assert N.pick_alpha({"0.2": -1.0, "0.5": -3.0, "0.8": -2.0}, (0.2, 0.5, 0.8)) == 0.5
    assert N.pick_alpha({"0.2": -3.0, "0.5": -3.0, "0.8": -2.0}, (0.2, 0.5, 0.8)) == 0.2
    with pytest.raises(ValueError, match="undefined"):
        N.pick_alpha({"0.2": None, "0.5": -1.0, "0.8": -1.0}, (0.2, 0.5, 0.8))


def test_punish_only_pre_is_oracle_jobs_pre(conn_pops):
    c, pops = conn_pops
    got = N.punish_only_oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                                   punish_type="PPL105", **ORACLE)
    ref = oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                     types=("MBON03", "MBON01"), punish_type="PPL105", reward_type="PAM08", **ORACLE)
    assert got["report"]["pre"] == ref["report"]["pre"]
    assert got["select"]["pre"] == {"A": ref["select"]["pre"]["MBON03"], "P": ref["select"]["pre"]["MBON01"]}
    assert got["kc"]["x"]["frac"] == ref["kc"]["x"]["frac"] and got["kc"]["jaccard"] == ref["kc"]["jaccard"]
    assert got["alpha_punish"] in ORACLE["alphas"]


def test_a_zero_alpha_leaves_the_probes_unchanged_and_weights_are_restored(conn_pops):
    c, pops = conn_pops
    got = N.punish_only_oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                                   punish_type="PPL105", **dict(ORACLE, alphas=(0.0,)))
    assert got["report"]["P"] == got["report"]["pre"] and got["select"]["punish"]["0.0"]["change"] == 0.0
    e, p, _, sha = N.rig(c, pops, P, "none")
    assert np.array_equal(e.csc.w[p.edges], p.w0) and got["csc_sha256"] == sha


def test_the_punish_edit_leaves_P_untouched(conn_pops):
    c, pops = conn_pops
    got = N.punish_only_oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                                   punish_type="PPL105", **dict(ORACLE, alphas=(0.8,)))
    assert got["alpha_punish"] == 0.8
    assert got["report"]["P"]["P"] == got["report"]["pre"]["P"]        # only the PPL105 core (A's pool) is edited


def test_train_plus_only_uses_train_blocks_seed_rule(conn_pops, monkeypatch):
    c, pops = conn_pops
    e, p, _, _ = N.rig(c, pops, P, "none")
    seen, orig = [], e.reset
    monkeypatch.setattr(e, "reset", lambda seed=None: (seen.append(seed), orig(seed))[1])
    N.train_plus_only(e, p, pops, X, 3.0, 21_002_063, "PPL105", 3, 100.0, 50.0, 100.0, 1_000_000, 1000)
    assert seen == [1_000_000 + 21_002_063 * 1000 + t for t in range(3)]


def test_one_plus_trial_equals_train_blocks_first_plus_presentation(conn_pops):
    c, pops = conn_pops
    e, p, _, _ = N.rig(c, pops, P, "none")
    snap = {}

    def on_event(kind, **f):
        if f["trial"] == 0 and f["cs"] == "plus":
            snap["w"] = e.csc.w[p.edges].copy()

    p.reset_weights(); p.set_enabled(True)
    train_block(e, p, pops, X, Y, 3.0, 7, "PPL105", None, trials=1, present_ms=300.0, gap_ms=50.0, settle_ms=100.0,
                on_event=on_event)
    p.reset_weights(); p.set_enabled(True)
    N.train_plus_only(e, p, pops, X, 3.0, 7, "PPL105", 1, 300.0, 50.0, 100.0, 1_000_000, 1000)
    assert not np.array_equal(snap["w"], p.w0)            # the pulse depressed something: the comparison is not vacuous
    assert np.array_equal(e.csc.w[p.edges], snap["w"])
    p.reset_weights()


def test_arm_probes_are_decides_counts(conn_pops):
    c, pops = conn_pops
    row = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=X, odor_y=Y, seed=9,
                             plastic=True, **ARM)
    e, p, _, _ = N.rig(c, pops, P, "none")
    t = np.asarray(c.type).astype(str)
    cnt = decide(e, p, pops, [X, Y], W["strength"], 9, W["settle_ms"], W["read_ms"])
    for k, i in (("x", 0), ("y", 1)):
        assert (row["pre"][k]["A"], row["pre"][k]["P"]) == (int(cnt[i][t == "MBON03"].sum()),
                                                              int(cnt[i][t == "MBON01"].sum()))
    assert row["weights_frac"] < 1.0 and np.array_equal(e.csc.w[p.edges], p.w0)   # learned, then restored


def test_plumbing_arm_changes_nothing(conn_pops):
    c, pops = conn_pops
    row = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit="apl_to_kc_zero", odor_x=X, odor_y=Y,
                             seed=9, plastic=False, **ARM)
    strip = lambda d: {k: {f: v for f, v in r.items() if f != "wall_s"} for k, r in d.items()}
    assert strip(row["pre"]) == strip(row["post"]) and row["weights_frac"] == 1.0


def test_arms_on_block_on_in_one_worker(conn_pops):
    c, pops = conn_pops
    run = lambda edit: N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit=edit, odor_x=X, odor_y=Y,
                                          seed=9, plastic=True, **ARM)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s",)} | {
        "pre": {k: {f: v for f, v in x.items() if f != "wall_s"} for k, x in r["pre"].items()},
        "post": {k: {f: v for f, v in x.items() if f != "wall_s"} for k, x in r["post"].items()}}
    a, b, a2 = run("none"), run("apl_to_kc_zero"), run("none")
    assert strip(a) == strip(a2) and a["csc_sha256"] != b["csc_sha256"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_n_jobs_arms.py -q -o addopts=""`
Expected: FAIL (`AttributeError: module 'flymon.brain.n_jobs' has no attribute 'pick_alpha'`).

- [ ] **Step 3: Append to `n_jobs.py`**

```python
# ================================================================ N1: the punish-only oracle (N.3, N.8.5)
def pick_alpha(changes: dict, alphas) -> float:
    """The punishment alpha with the smallest change (the most negative d'), ties -> the smaller alpha."""
    if any(changes.get(str(a)) is None for a in alphas):
        raise ValueError("an undefined change (fewer than 2 select seeds)")
    return min(alphas, key=lambda a: (changes[str(a)], a))


def punish_only_oracle_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, readout: dict, z: dict,
                           act_seeds, select_seeds, report_seeds, alphas, strength: float, settle_ms: float,
                           read_ms: float, window_ms: int, punish_type: str) -> dict:
    """h4_jobs.oracle_job's punishment half:
    - KC activity f of X (and Y, for the Jaccard record) on act_seeds;
    - the punish core's KC -> MBON weights set to w0 (1 - alpha f) for X's KCs, alpha picked on select_seeds by the
      change d'(dV_edit - dV_pre) alone (no reward edit; pick_alpha);
    - pre and P on report_seeds.
    APL on only (N.3 blinding)."""
    e, p, c, sha = rig(eng.conn, pops, params, "none")
    a_cells, p_cells = readout_cells(e.conn, readout)
    idx = np.concatenate([a_cells, p_cells]); n_a = len(a_cells)
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, max_win = [], []
            for s in act_seeds:
                o = _present_kc(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); max_win.append(o["max_win"])
            return fired / len(act_seeds), {"frac": frac, "max_win": max_win}

        def probe(seeds):
            A, Pc = [], []
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                A.append(cnt[:, :n_a].sum(1).tolist()); Pc.append(cnt[:, n_a:].sum(1).tolist())
            return {"A": A, "P": Pc}

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
        pun = np.isin(p.post_mb, p.mb_local[c[punish_type].core])
        w = e.csc.w

        def set_w(a_p):
            wv = p.w0.copy()
            if a_p is not None:
                wv[pun] = p.w0[pun] * (1.0 - a_p * fx)[p.pre_kc[pun]]
            w[p.edges] = wv

        set_w(None); pre_sel = probe(select_seeds)
        punish = {}
        for a in alphas:
            set_w(a); r = probe(select_seeds)
            punish[str(a)] = {"P": r, "change": dprime(dv(r, z) - dv(pre_sel, z))}
        a_p = pick_alpha({k: v["change"] for k, v in punish.items()}, alphas)
        set_w(None); pre = probe(report_seeds)
        set_w(a_p); post = probe(report_seeds)
        w[p.edges] = p.w0
        jac = float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))
        return {"alpha_punish": a_p, "select": {"pre": pre_sel, "punish": punish}, "report": {"pre": pre, "P": post},
                "kc": {"x": kc_x, "y": kc_y, "jaccard": jac}, "csc_sha256": sha}
    finally:
        p.reset_weights(); p.set_enabled(True)


# ================================================================ N2: absolute punishment conditioning (N.4, N.8.7)
def train_plus_only(e, p, pops, cs_plus, strength: float, seed: int, punish: str, trials: int, present_ms: float,
                    gap_ms: float, settle_ms: float, seed_base: int, seed_stride: int) -> None:
    """conditioning.train_block's CS+ half alone (no CS- during training). Per trial:
    - reset to seed_base + seed x seed_stride + trial;
    - settle with the weights frozen;
    - the punishment DAN for present_ms;
    - the gap and one recovery step."""
    for t in range(int(trials)):
        e.reset(int(seed_base) + int(seed) * int(seed_stride) + t)
        p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, cs_plus, strength)
        was = p.enabled
        p.set_enabled(False); e.run(settle_ms); p.set_enabled(was)
        p.drive_dan(punish, e.p.dan_drive_mv)
        e.run(present_ms)
        p.quiet_dan(); e.clear_drive(); e.run(gap_ms)
        p.recover_pulse()


def absolute_arm_job(eng, pl, pops, comps, ro, params, edit: str, odor_x: dict, odor_y: dict, seed: int,
                     plastic: bool, readout: dict, punish_type: str, strength: float, settle_ms: float, read_ms: float,
                     window_ms: int, trials: int, present_ms: float, gap_ms: float, train_settle_ms: float,
                     seed_base: int, seed_stride: int) -> dict:
    """One (condition, seed) of N2, in the edit's rig for the whole arm (pre-test, training and post-test: N.4):
    - probes of X and Y at `seed`, plasticity off (A, P, KC fraction, KC sub-window);
    - training of X alone with the punishment DAN;
    - the probes again.
    With plastic False the training runs with the weights frozen (the plumbing check, N.8.7). Weights are restored."""
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = readout_cells(e.conn, readout)
    t0 = time.perf_counter()

    def probes():
        was = p.enabled
        p.set_enabled(False)
        try:
            return {k: _present(e, p, pops, o, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells)[0]
                    for k, o in (("x", odor_x), ("y", odor_y))}
        finally:
            p.set_enabled(was)

    try:
        p.reset_weights()
        pre = probes()
        p.set_enabled(bool(plastic))
        train_plus_only(e, p, pops, odor_x, strength, int(seed), punish_type, trials, present_ms, gap_ms,
                        train_settle_ms, seed_base, seed_stride)
        p.set_enabled(True)
        post = probes()
        wf = float(p.weights_frac())
    finally:
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), edit=edit, plastic=bool(plastic), csc_sha256=sha, pre=pre, post=post,
                weights_frac=wf, wall_s=time.perf_counter() - t0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_n_jobs.py tests/brain/test_n_jobs_arms.py -q -o addopts=""`
Expected: PASS (18 tests).

If `test_one_plus_trial_equals_train_blocks_first_plus_presentation` fails **only** on the "not vacuous" assert, the synthetic pulse moved no weight. Lengthen `present_ms` in both calls (e.g. 600.0) until the weights move. Never delete the assert.

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_jobs.py tests/brain/test_n_jobs_arms.py
git commit -m "feat(n): punish-only oracle (oracle_job's punishment half) and the CS+-only absolute conditioning arm"
```

---

### Task 6: `n_measure` — content-addressed cache and the measurer

**Files:**
- Create: `flymon/brain/n_measure.py`
- Test: `tests/brain/test_n_measure.py`

**Interfaces:**
- Consumes: `h3_store.MeasureCache`, `canonical`, `MEASURE_FILES`; `n_store.write_json`; Task 4–5 jobs; `n_spec` fields `h3.strength`, `h4.oracle_window`, `h4.kc_window_ms`, `h4.teach_*`, `h4.oracle_alphas`, `act_seeds`, `select_seeds`, `report_seeds`, `train_seed_base`, `train_seed_stride`.
- Produces:
  - `MEASURE_FILES`, `HASHED_FILES` (tuples of repo paths). `HASHED_FILES` names files Tasks 7–11 create, so `n_cli.code_keys` refuses until they all exist. Task 9's CLI tests patch `code_keys`.
  - `NCache(root, code: dict, run_id)` (a `MeasureCache` whose writes go through `n_store`).
  - `NMeasurer(pool, spec, cache)` with `.params_seen` and:
    - `.presentations(params, items: list[(edit, odor)], seeds, readout) -> list[list[row]]`
    - `.kc_vectors(params, items, seeds, readout) -> list[list[row]]`
    - `.punish_oracle(params, pairs: list[{"name", "odor_x", "odor_y"}], readout, z, punish_type) -> list[dict]`
    - `.arms(params, items: list[{"cond", "edit", "odor_x", "odor_y", "seed", "plastic"}], readout, punish_type) -> list[dict + "cond"]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_n_measure.py
"""Spec N.8.7: one cache entry per (condition, seed) arm (the judge's checkpoint), resumable without a pool; APL-on and
APL-block items never share an entry; caches written only under results/n/; rounds of one item per worker."""
import json
from pathlib import Path

import pytest

from flymon.brain import n_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT
from flymon.brain.n_measure import HASHED_FILES, MEASURE_FILES, NCache, NMeasurer
from flymon.brain.n_spec import SPEC

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}
X, Y = {"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


class InProcessPool:
    """Runs the real job in this process on the synthetic connectome, recording the round sizes."""
    n_workers = 3

    def __init__(self, conn, pops):
        self.conn, self.pops, self.rounds = conn, pops, []

    def run_jobs(self, fn, jobs):
        self.rounds.append(len(jobs))
        return [fn(_Stub(self.conn), None, self.pops, None, None, **j) for j in jobs]


class NoPool:
    n_workers = 3

    def run_jobs(self, fn, jobs):
        raise AssertionError("a resumed run must not touch the pool")


def _spec():
    from dataclasses import replace
    h4 = replace(SPEC.h4, teach_trials=1, teach_present_ms=100.0, teach_gap_ms=20.0,
                 teach_window=replace(SPEC.h4.teach_window, settle_ms=50.0),
                 oracle_window=replace(SPEC.h4.oracle_window, settle_ms=30.0, read_ms=50.0))
    return replace(SPEC, l=replace(SPEC.l, j=replace(SPEC.l.j, h4=h4)))


@pytest.fixture
def world(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome(disjoint_kc=True)
    n_jobs._RIG.clear()
    return c, Populations.from_connectome(c)


def _items():
    return [dict(cond=cond, edit=edit, odor_x=X, odor_y=Y, seed=s, plastic=True)
            for cond, edit in (("sim_on", "none"), ("sim_off", "apl_to_kc_zero")) for s in (5, 6)]


def test_one_entry_per_condition_and_seed_and_a_rerun_resumes(world):
    c, pops = world
    code = {"key": "k" * 64}
    pool = InProcessPool(c, pops)
    m = NMeasurer(pool, _spec(), NCache("results/n/cache", code, "run1"))
    rows = m.arms(P, _items(), RO, "PPL105")
    assert [r["cond"] for r in rows] == ["sim_on", "sim_on", "sim_off", "sim_off"]
    assert pool.rounds == [3, 1]                                   # rounds of one item per worker
    files = sorted(Path("results/n/cache/n_arm").glob("*.json"))
    assert len(files) == 4                                         # on and block never share an entry
    again = NMeasurer(NoPool(), _spec(), NCache("results/n/cache", code, "run2")).arms(P, _items(), RO, "PPL105")
    assert [json.dumps(r, sort_keys=True) for r in again] == [json.dumps(json.loads(json.dumps(r)), sort_keys=True)
                                                               for r in rows]


def test_presentations_and_kc_vectors_are_cached_per_edit_and_odour(world):
    c, pops = world
    m = NMeasurer(InProcessPool(c, pops), _spec(), NCache("results/n/cache", {"key": "k" * 64}, "r"))
    rows = m.presentations(P, [("none", X), ("apl_to_kc_zero", X)], (5, 6), RO)
    assert [len(r) for r in rows] == [2, 2] and rows[0][0]["edit"] == "none" and "kc_fired" not in rows[1][0]
    kcv = m.kc_vectors(P, [("none", X)], (5,), RO)
    assert "kc_fired" in kcv[0][0]
    assert len(list(Path("results/n/cache/n_pres").glob("*.json"))) == 2
    assert m.params_seen == [P]


def test_punish_oracle_is_one_entry_per_pair(world):
    c, pops = world
    from dataclasses import replace
    s = replace(_spec(), act_seeds=(1, 2), select_seeds=(3, 4), report_seeds=(5, 6))
    m = NMeasurer(InProcessPool(c, pops), s, NCache("results/n/cache", {"key": "k" * 64}, "r"))
    out = m.punish_oracle(P, [dict(name="sim", odor_x=X, odor_y=Y)], RO, {"A": (1.0, 2.0), "P": (1.0, 2.0)}, "PPL105")
    assert out[0]["alpha_punish"] in SPEC.h4.oracle_alphas
    assert len(list(Path("results/n/cache/n_oracle").glob("*.json"))) == 1


def test_the_cache_refuses_paths_outside_results_n(world):
    c, pops = world
    m = NMeasurer(InProcessPool(c, pops), _spec(), NCache("results/m0d/n/cache", {"key": "k" * 64}, "r"))
    with pytest.raises(SystemExit):
        m.presentations(P, [("none", X)], (5,), RO)


def test_the_file_keys_cover_the_data_and_every_n_file():
    assert "data/odor/hallem2006_subset.csv" in MEASURE_FILES and "flymon/brain/n_jobs.py" in MEASURE_FILES
    for f in ("flymon/brain/n_rules.py", "flymon/brain/n_oc.py", "flymon/brain/n_cli.py", "scripts/run_n2_judge.py"):
        assert f in HASHED_FILES
    assert set(MEASURE_FILES) <= set(HASHED_FILES)
    assert all((ROOT / f).exists() for f in MEASURE_FILES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_n_measure.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.n_measure`).

- [ ] **Step 3: Write `n_measure.py`**

```python
"""The measurement layer of spec appendix N. `NMeasurer` runs n_jobs through a FlyPool and caches every item under
`NCache`: h3_store.MeasureCache's content-addressed entries, but written through n_store's guard under results/n/
(MeasureCache's own write is guarded to results/m0d/).

Items:
- presentations per (edit, odour) over a seed block;
- KC vectors per (edit, odour);
- the punish-only oracle per pair;
- N2's arm per (condition, seed, plastic). This is N.8.7's checkpoint: an interrupted judge loses at most one round of
  one item per worker, and a rerun reads every finished item without starting a job.
Every key carries the Params, the edit, the odour strengths, the seeds and the windows, so an APL-on and an APL-block
result can never share an entry."""
from __future__ import annotations

import json

from . import n_jobs, n_store
from .h3_store import MEASURE_FILES as H3_MEASURE_FILES, MeasureCache, canonical

MEASURE_FILES = tuple(dict.fromkeys(H3_MEASURE_FILES + (
    "flymon/brain/plasticity.py", "flymon/brain/presentation.py", "flymon/brain/h4_jobs.py",
    "flymon/brain/h4_formula.py", "flymon/brain/odor_real.py", "flymon/brain/n_jobs.py", "flymon/brain/n_measure.py",
    "data/odor/hallem2006_subset.csv", "data/odor/door_mappings_subset.csv")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + (
    "flymon/brain/n_spec.py", "flymon/brain/n_store.py", "flymon/brain/n_rules.py", "flymon/brain/n_oc.py",
    "flymon/brain/n_cli.py", "scripts/fetch_door_hallem.py", "scripts/run_n0f.py", "scripts/run_n0.py",
    "scripts/run_n1.py", "scripts/run_n2_pilot.py", "scripts/run_n2_judge.py", "scripts/write_n_notes.py")))


class NCache(MeasureCache):
    """MeasureCache with n_store's guarded write (results/n/ only)."""

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
        n_store.write_json(path, dict(key=k, kind=kind, run_id=self.run_id, inputs=json.loads(canonical(inputs)),
                                      result=result), params_list)
        self.misses += 1
        return json.loads(canonical(result))


class _Missing(Exception):
    pass


def _missing():
    raise _Missing


class NMeasurer:
    def __init__(self, pool, spec, cache):
        self.pool, self.spec, self.cache = pool, spec, cache
        self.params_seen: list = []

    def _items(self, kind, fn, params, items: list, keys: list) -> list:
        """One cache entry per item (keys[i] are its inputs), run in rounds of one item per worker."""
        if params not in self.params_seen:
            self.params_seen.append(params)
        done = {}
        for i, k in enumerate(keys):
            try:
                done[i] = self.cache.get_or_compute(kind, k, _missing, [params])
            except _Missing:
                pass
        todo = [i for i in range(len(items)) if i not in done]
        n = max(1, self.pool.n_workers)
        for r in range(0, len(todo), n):
            batch = todo[r:r + n]
            for i, out in zip(batch, self.pool.run_jobs(fn, [items[i] for i in batch])):
                done[i] = self.cache.get_or_compute(kind, keys[i], lambda out=out: out, [params])
        return [done[i] for i in range(len(items))]

    def _window(self) -> dict:
        h4 = self.spec.h4
        return dict(strength=self.spec.h3.strength, settle_ms=h4.oracle_window.settle_ms,
                    read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms))

    def _pres(self, kind, fn, params, items, seeds, readout) -> list:
        common = dict(params=params, seeds=tuple(int(s) for s in seeds), readout=dict(readout), **self._window())
        jobs = [dict(common, edit=e, odor=dict(o)) for e, o in items]
        return self._items(kind, fn, params, jobs, jobs)

    def presentations(self, params, items, seeds, readout) -> list:
        """items [(edit, odour)] -> one row list per item (presentation_job over `seeds`): scalars only."""
        return self._pres("n_pres", n_jobs.presentation_job, params, items, seeds, readout)

    def kc_vectors(self, params, items, seeds, readout) -> list:
        return self._pres("n_kcv", n_jobs.kc_vectors_job, params, items, seeds, readout)

    def punish_oracle(self, params, pairs, readout, z, punish_type) -> list:
        s = self.spec
        common = dict(params=params, readout=dict(readout), z={k: [float(v) for v in z[k]] for k in ("A", "P")},
                      act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds),
                      report_seeds=tuple(s.report_seeds), alphas=tuple(s.h4.oracle_alphas), punish_type=punish_type,
                      **self._window())
        jobs = [dict(common, odor_x=dict(p["odor_x"]), odor_y=dict(p["odor_y"])) for p in pairs]
        return self._items("n_oracle", n_jobs.punish_only_oracle_job, params, jobs, jobs)

    def arms(self, params, items, readout, punish_type) -> list:
        """items [{"cond", "edit", "odor_x", "odor_y", "seed", "plastic"}] -> absolute_arm_job rows with "cond"."""
        s, h4 = self.spec, self.spec.h4
        common = dict(params=params, readout=dict(readout), punish_type=punish_type, trials=int(h4.teach_trials),
                      present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms,
                      train_settle_ms=h4.teach_window.settle_ms, seed_base=s.train_seed_base,
                      seed_stride=s.train_seed_stride, **self._window())
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     plastic=bool(i["plastic"])) for i in items]
        rows = self._items("n_arm", n_jobs.absolute_arm_job, params, jobs, jobs)
        return [dict(r, cond=i["cond"]) for r, i in zip(rows, items)]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_n_measure.py -q -o addopts=""`
Expected: PASS (5 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_measure.py tests/brain/test_n_measure.py
git commit -m "feat(n): n_measure (NCache under results/n/, per-(condition, seed) arm checkpoints, file keys)"
```

---
### Task 7: `n_rules` (1) — N0f statistics and operating point, N0 similarity, N1 reading

**Files:**
- Create: `flymon/brain/n_rules.py`
- Test: `tests/brain/test_n_rules.py`

**Interfaces:**
- Consumes: `h4_formula.dprime`, `dv`; `n_spec` fields (`kc_band`, `kc_target`, `kc_block_max`, `runaway_hz`, `runaway_share_max`, `state_p_min`, `state_diff_flag`, `zero_share_max`, `judged_stimuli`, `pair_stimuli()`, `boot_draws`, `boot_seed`, `testable_min`).
- Produces:
  - Outcome constants: `OPERATING_POINT`, `STOP_DATA_MISMATCH`, `STOP_NO_OPERATING_POINT`, `SIMILARITY_GO`, `STOP_SIMILARITY_ORDER`, `N1_GO`, `STOP_UNTESTABLE`, `N2_0_GO`, `STOP_POWER`, `STOP_BUDGET`, `SUPPORTED`, `NOT_REPLICATED`, `NO_LEARNING`, `INVALID`.
  - `state(P, spec) -> "firing" | "silent"`.
  - `cell_stats(rows, spec) -> dict` with keys `n`, `kc_frac_median`, `kc_frac`, `runaway_share`, `max_win_hz_max`, `A_median`, `P_median`, `A_zero_share`, `P_zero_share`, `firing_share`, `apl_out_mean`, `wall_s_median`.
  - `wall_per_step(rows) -> float`, `point_key(g, c_delta) -> str`.
  - `point_checks(cells, spec) -> {"band", "runaway", "floor", "ok", "mean_on", "dist"}`.
  - `select_point(grid: {(g, c): cells}, spec) -> {"outcome", "selected": {"g", "c_delta"} | None, "checks": {point_key: checks}}`.
  - `state_flags(cells, spec) -> list[dict]`.
  - `similarity(fired: {stimulus: [positions per seed]}, n_kc, spec, draws=None, seed=None) -> {"outcome", "r_sim", "r_dis", "delta_r", "ci95", "draws", "n_seeds"}`.
  - `n1_pair(row, z, spec) -> {"p0", "sd", "o", "d_pre", "testable", "alpha", "changes", "jaccard", "floor"}`.
  - `n1_outcome(pairs: {"sim": ..., "dis": ...}) -> {"outcome", "note"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_n_rules.py
"""Spec N.8.3 / N.8.4 / N.8.5 / N.8.8: the state threshold, N0f's per-cell numbers and inclusive clause bounds, the
operating-point choice (nearest 5.54%, then smaller g, then smaller c_δ; IA / EB alone never count), the state flags,
the similarity gate's bootstrap (an undefined r stops, never passes), N1's testable rule (sd = 0 is untestable)."""
import numpy as np
import pytest

from flymon.brain import n_rules as R
from flymon.brain.n_spec import SPEC


def _row(kc=0.05, hz=50.0, A=10, P=20, apl=0.1, wall=1.4, steps=1400):
    return dict(seed=0, A=A, P=P, kc_frac=kc, kc_spikes=1, kc_max_win_hz=hz, apl_out_per_step=apl, wall_s=wall,
                steps=steps)


def _cells(on=0.05, block=0.08, **kw):
    st = lambda kc: R.cell_stats([_row(kc=kc, **kw)] * 8, SPEC)
    return {s: {"on": st(on), "block": st(block), "all": st(block)} for s in SPEC.n0f_stimuli}


def test_state_threshold_is_p_below_5_silent():
    assert (R.state(4, SPEC), R.state(5, SPEC)) == ("silent", "firing")


def test_cell_stats():
    rows = [_row(hz=h, A=a, P=p) for h, a, p in [(151, 0, 4), (150, 3, 5), (10, 0, 30), (10, 2, 0)]]
    c = R.cell_stats(rows, SPEC)
    assert c["runaway_share"] == 0.25 and c["A_zero_share"] == 0.5 and c["P_zero_share"] == 0.25
    assert c["firing_share"] == 0.5 and c["max_win_hz_max"] == 151.0 and c["n"] == 4
    assert R.wall_per_step([_row(wall=1.4), _row(wall=2.8)]) == pytest.approx(0.0015)


@pytest.mark.parametrize("on, ok", [(0.03, True), (0.15, True), (0.0299, False), (0.1501, False)])
def test_band_bounds_are_inclusive(on, ok):
    assert R.point_checks(_cells(on=on), SPEC)["band"] is ok


@pytest.mark.parametrize("block, ok", [(0.30, True), (0.3001, False)])
def test_block_kc_bound(block, ok):
    assert R.point_checks(_cells(block=block), SPEC)["runaway"] is ok


def test_runaway_share_and_zero_share_bounds():
    def with_rows(rows_block):
        c = _cells()
        for s in SPEC.judged_stimuli:
            c[s]["block"] = R.cell_stats(rows_block, SPEC)
        return R.point_checks(c, SPEC)
    one = [_row(hz=151)] + [_row()] * 7; two = [_row(hz=151)] * 2 + [_row()] * 6
    assert with_rows(one)["runaway"] and not with_rows(two)["runaway"]
    zeros2 = [_row(A=0)] * 2 + [_row()] * 6; zeros3 = [_row(P=0)] * 3 + [_row()] * 5
    assert with_rows(zeros2)["floor"] and not with_rows(zeros3)["floor"]


def test_ia_and_eb_alone_never_count():
    c = _cells()
    c["IA"]["on"] = R.cell_stats([_row(kc=0.9, hz=900, A=0, P=0)] * 8, SPEC)
    c["EB"]["block"] = c["IA"]["on"]
    assert R.point_checks(c, SPEC)["ok"]


def test_select_nearest_then_smaller_g_then_smaller_c():
    got = R.select_point({(0.5, 1.0): _cells(on=0.05), (1.0, 1.0): _cells(on=0.06), (0.25, 1.0): _cells(on=0.20)},
                         SPEC)
    assert got["outcome"] == R.OPERATING_POINT and got["selected"] == {"g": 1.0, "c_delta": 1.0}   # |.06-.0554| < |.05-.0554|
    tie = R.select_point({(1.0, 2.0): _cells(), (2.0, 1.0): _cells(), (1.0, 1.0): _cells()}, SPEC)
    assert tie["selected"] == {"g": 1.0, "c_delta": 1.0}
    stop = R.select_point({(0.25, 1.0): _cells(on=0.2)}, SPEC)
    assert stop["outcome"] == R.STOP_NO_OPERATING_POINT and stop["selected"] is None
    assert set(stop["checks"]) == {"0.25|1"} and R.point_key(0.125, 8.0) == "0.125|8"


def test_state_flags_are_strictly_above_a_quarter():
    cells = {"4:1": {"on": {"firing_share": 1.0}, "block": {"firing_share": 0.75}, "all": {"firing_share": 0.7}}}
    assert R.state_flags(cells, SPEC) == [dict(stimulus="4:1", condition="all", on=1.0, share=0.7)]


def _fired(x, y_sim, y_dis, n=4):
    return {"4:1": [x + [90 + s] for s in range(n)], "1:4": [y_sim + [90 + s] for s in range(n)],
            "dDL": [y_dis + [95 - s] for s in range(n)]}


def test_similarity_gate_passes_and_stops():
    same, near, far = list(range(20)), list(range(2, 22)), list(range(60, 80))
    go = R.similarity(_fired(same, near, far), 100, SPEC, draws=300)
    assert go["outcome"] == R.SIMILARITY_GO and go["r_sim"] > go["r_dis"] and go["ci95"][0] > 0
    stop = R.similarity(_fired(same, far, near), 100, SPEC, draws=300)
    assert stop["outcome"] == R.STOP_SIMILARITY_ORDER and stop["delta_r"] < 0


def test_similarity_with_silent_kcs_stops_without_crashing():
    empty = {s: [[] for _ in range(4)] for s in ("4:1", "1:4", "dDL")}
    got = R.similarity(empty, 100, SPEC, draws=50)
    assert got["outcome"] == R.STOP_SIMILARITY_ORDER and np.isnan(got["r_sim"]) and np.isnan(got["ci95"][0])
    with pytest.raises(ValueError, match="same"):
        R.similarity({"4:1": [[1]] * 4, "1:4": [[1]] * 3, "dDL": [[1]] * 4}, 10, SPEC, draws=10)


def _oracle(d):
    n = len(d)
    pre = {"A": [[10.0, 10.0]] * n, "P": [[5.0, 5.0]] * n}
    post = {"A": [[10.0 + x, 10.0] for x in d], "P": [[5.0, 5.0]] * n}
    return {"alpha_punish": 0.5, "select": {"punish": {"0.5": {"change": -1.0}}}, "report": {"pre": pre, "P": post},
            "kc": {"jaccard": 0.2}}


ZU = {"A": (0.0, 1.0), "P": (0.0, 1.0)}          # dV = (A_x - P_x) - (A_y - P_y)


def test_n1_testable_needs_p0_at_most_minus_2_and_a_spread():
    ok = R.n1_pair(_oracle([-3.0, -4.0, -5.0]), ZU, SPEC)
    assert ok["testable"] and ok["p0"] == pytest.approx(-4.0) and ok["o"] == pytest.approx(-4.0)
    weak = R.n1_pair(_oracle([-1.0, -2.0, -3.0, -0.5]), ZU, SPEC)
    assert not weak["testable"] and weak["p0"] > -2
    flat = R.n1_pair(_oracle([-3.0, -3.0, -3.0]), ZU, SPEC)
    assert flat["p0"] == -np.inf and flat["sd"] == 0.0 and not flat["testable"]      # N.8.5: sd = 0 is untestable
    assert ok["floor"] == {"A_zero_share": 0.0, "P_zero_share": 0.0} and ok["jaccard"] == 0.2


def test_n1_outcome():
    t, f = {"testable": True}, {"testable": False}
    assert R.n1_outcome({"sim": t, "dis": t}) == {"outcome": R.N1_GO, "note": None}
    dis_only = R.n1_outcome({"sim": f, "dis": t})
    assert dis_only["outcome"] == R.STOP_UNTESTABLE and "비슷한 쌍" in dis_only["note"]
    assert R.n1_outcome({"sim": t, "dis": f}) == {"outcome": R.STOP_UNTESTABLE, "note": None}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_n_rules.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.n_rules`).

- [ ] **Step 3: Write `n_rules.py` (part 1)**

```python
"""The pure rules of spec appendix N as amended by N.8. No engine, no pool.
- The presentation state (N.8.8).
- N0f's per-cell statistics and operating-point rule (N.8.3), with the state-share flags.
- N0's similarity gate (N.8.4).
- N1's per-pair oracle reading and gate (N.3, N.8.5).
- N2's raw-unit statistics, validity checks and verdict (N.8.6).
- N2.0's thresholds, budget and outcome.
- The design view N2.0 may read from block n0f (blinding), and the sentences."""
from __future__ import annotations

import numpy as np

from .h4_formula import dprime, dv

OPERATING_POINT = "OPERATING_POINT"
STOP_DATA_MISMATCH = "STOP_DATA_MISMATCH"
STOP_NO_OPERATING_POINT = "STOP_NO_OPERATING_POINT"
SIMILARITY_GO = "SIMILARITY_GO"
STOP_SIMILARITY_ORDER = "STOP_SIMILARITY_ORDER"
N1_GO = "N1_GO"
STOP_UNTESTABLE = "STOP_UNTESTABLE"
N2_0_GO = "N2_0_GO"
STOP_POWER = "STOP_POWER"
STOP_BUDGET = "STOP_BUDGET"
SUPPORTED = "SUPPORTED"
NOT_REPLICATED = "NOT_REPLICATED"
NO_LEARNING = "NO_LEARNING"
INVALID = "INVALID"


def state(P: int, spec) -> str:
    return "firing" if int(P) >= spec.state_p_min else "silent"


# ================================================================ N0f (N.8.3)
def cell_stats(rows: list, spec) -> dict:
    """One (odour, condition) block of presentation rows -> N.8.3's numbers."""
    kc = np.array([r["kc_frac"] for r in rows], float)
    hz = np.array([r["kc_max_win_hz"] for r in rows], float)
    A = np.array([r["A"] for r in rows], float); P = np.array([r["P"] for r in rows], float)
    return dict(n=len(rows), kc_frac_median=float(np.median(kc)), kc_frac=kc.tolist(),
                runaway_share=float((hz > spec.runaway_hz).mean()), max_win_hz_max=float(hz.max()),
                A_median=float(np.median(A)), P_median=float(np.median(P)),
                A_zero_share=float((A == 0).mean()), P_zero_share=float((P == 0).mean()),
                firing_share=float((P >= spec.state_p_min).mean()),
                apl_out_mean=float(np.mean([r["apl_out_per_step"] for r in rows])),
                wall_s_median=float(np.median([r["wall_s"] for r in rows])))


def wall_per_step(rows: list) -> float:
    return float(np.median([r["wall_s"] / r["steps"] for r in rows]))


def point_key(g: float, c_delta: float) -> str:
    return f"{g:g}|{c_delta:g}"


def point_checks(cells: dict, spec) -> dict:
    """cells {stimulus: {"on" | "block" | "all": cell_stats}} at one (g, c_δ). Returns N.8.3's clauses ① ② ③ on the
    judged stimuli (IA / EB alone are records) and the distance of the on medians' mean from the target."""
    lo, hi = spec.kc_band
    on = [cells[s]["on"] for s in spec.judged_stimuli]
    both = on + [cells[s]["block"] for s in spec.judged_stimuli]
    band = all(lo <= c["kc_frac_median"] <= hi for c in on)
    runaway = all(c["kc_frac_median"] <= spec.kc_block_max and c["runaway_share"] <= spec.runaway_share_max
                  for c in both)
    floor = all(c["A_zero_share"] <= spec.zero_share_max and c["P_zero_share"] <= spec.zero_share_max for c in both)
    mean_on = float(np.mean([c["kc_frac_median"] for c in on]))
    return dict(band=bool(band), runaway=bool(runaway), floor=bool(floor), ok=bool(band and runaway and floor),
                mean_on=mean_on, dist=abs(mean_on - spec.kc_target))


def select_point(grid: dict, spec) -> dict:
    """grid {(g, c_δ): cells} -> the passing point nearest the target; ties -> smaller g, then smaller c_δ."""
    checks = {k: point_checks(v, spec) for k, v in grid.items()}
    table = {point_key(*k): c for k, c in checks.items()}
    ok = [k for k, c in checks.items() if c["ok"]]
    if not ok:
        return dict(outcome=STOP_NO_OPERATING_POINT, selected=None, checks=table)
    g, c = min(ok, key=lambda k: (checks[k]["dist"], k[0], k[1]))
    return dict(outcome=OPERATING_POINT, selected=dict(g=float(g), c_delta=float(c)), checks=table)


def state_flags(cells: dict, spec) -> list:
    """Every (stimulus, block condition) whose firing share differs from the on share by more than 0.25 (a record)."""
    out = []
    for s, by in cells.items():
        for cond in ("block", "all"):
            if abs(by[cond]["firing_share"] - by["on"]["firing_share"]) > spec.state_diff_flag:
                out.append(dict(stimulus=s, condition=cond, on=by["on"]["firing_share"],
                                share=by[cond]["firing_share"]))
    return out


# ================================================================ N0 (N.8.4)
def _pearson_rows(a, b) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True); b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), np.nan)


def similarity(fired: dict, n_kc: int, spec, draws: int | None = None, seed: int | None = None) -> dict:
    """fired {stimulus: [fired KC positions per seed]} (the same seeds in the same order) -> r(4:1, 1:4),
    r(4:1, δ-DL) of the KC firing-probability vectors, Δr, and the common-seed bootstrap 95% CI. SIMILARITY_GO iff
    the lower bound is finite and > 0; an undefined r (a silent vector) fails."""
    x, y_sim = spec.pair_stimuli()["sim"]
    y_dis = spec.pair_stimuli()["dis"][1]
    n = len(fired[x])
    if n < 2 or any(len(fired[s]) != n for s in (y_sim, y_dis)):
        raise ValueError("similarity needs the same >= 2 seeds for every stimulus")
    M = {}
    for s in (x, y_sim, y_dis):
        m = np.zeros((n, int(n_kc)))
        for i, f in enumerate(fired[s]):
            m[i, np.asarray(f, np.int64)] = 1.0
        M[s] = m
    f = {s: M[s].mean(0) for s in M}
    r_sim, r_dis = float(_pearson_rows(f[x], f[y_sim])), float(_pearson_rows(f[x], f[y_dis]))
    draws = spec.boot_draws if draws is None else int(draws)
    idx = np.random.default_rng(spec.boot_seed if seed is None else seed).integers(0, n, (draws, n))
    wt = np.stack([np.bincount(row, minlength=n) for row in idx]) / n                  # draws x seeds weights
    d = np.empty(draws)
    for a in range(0, draws, 500):
        fb = {s: wt[a:a + 500] @ M[s] for s in M}
        d[a:a + 500] = _pearson_rows(fb[x], fb[y_sim]) - _pearson_rows(fb[x], fb[y_dis])
    if np.isfinite(d).all():
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    else:
        lo = hi = float("nan")
    ok = bool(np.isfinite(lo) and lo > 0)
    return dict(outcome=SIMILARITY_GO if ok else STOP_SIMILARITY_ORDER, r_sim=r_sim, r_dis=r_dis,
                delta_r=r_sim - r_dis, ci95=[lo, hi], draws=draws, n_seeds=n)


# ================================================================ N1 (N.3, N.8.5)
def n1_pair(row: dict, z: dict, spec) -> dict:
    """p0 = d'(dV_P - dV_pre) on the report seeds; testable iff p0 <= -testable_min and sd(dV_P - dV_pre) > 0;
    o = mean(dV_P - dV_pre) in z units (N.8.6's c1)."""
    pre, post = dv(row["report"]["pre"], z), dv(row["report"]["P"], z)
    d = post - pre
    p0 = dprime(d)
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    A = np.asarray(row["report"]["pre"]["A"], float); P = np.asarray(row["report"]["pre"]["P"], float)
    return dict(p0=p0, sd=sd, o=float(d.mean()), d_pre=dprime(pre),
                testable=bool(p0 is not None and p0 <= -spec.testable_min and sd > 0), alpha=row["alpha_punish"],
                changes={a: v["change"] for a, v in row["select"]["punish"].items()}, jaccard=row["kc"]["jaccard"],
                floor=dict(A_zero_share=float((A == 0).mean()), P_zero_share=float((P == 0).mean())))


def n1_outcome(pairs: dict) -> dict:
    t = {k: bool(v["testable"]) for k, v in pairs.items()}
    if all(t.values()):
        return dict(outcome=N1_GO, note=None)
    note = ("다른 쌍만 시험 가능: APL이 있어도 비슷한 쌍의 변별이 사전 지정 편집 프로토콜에서 서지 않는다"
            if t.get("dis") and not t.get("sim") else None)
    return dict(outcome=STOP_UNTESTABLE, note=note)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_n_rules.py -q -o addopts=""`
Expected: PASS (16 tests).

- [ ] **Step 5: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_rules.py tests/brain/test_n_rules.py
git commit -m "feat(n): n_rules for N0f (cell stats, operating point, state flags), N0 (bootstrap similarity), N1 (testable)"
```

---

### Task 8: `n_rules` (2) and `n_oc` — N2 statistics, verdict, validity; N2.0 thresholds, OC, budget

**Files:**
- Modify: `flymon/brain/n_rules.py` (append; new in this plan)
- Create: `flymon/brain/n_oc.py`
- Test: `tests/brain/test_n_verdict_oc.py`

**Interfaces:**
- Consumes: Task 7's constants and `cell_stats`; `h4_formula.dprime`, `dv`; `n_spec` fields (`boot_draws`, `boot_seed`, `pair_stimuli()`, `kc_block_max`, `runaway_share_max`, `zero_share_max`, `state_p_min`, `c1_frac`, `delta_min_frac`, `eps_frac`, `alt_frac`, `n_grid`, `sd_mults`, `null_max`, `power_min`, `oc_draws`, `oc_boot`, `oc_seed`, `budget_h`, `budget_workers`, `plumbing_seeds`, `n2_conditions`, `n2_record_conditions`, `h4.*` timings).
- Produces:
  - `N2_ORDER = ("sim_on", "sim_off", "dis_on", "dis_off")`.
  - `delta(row, z) -> float`, `deltas(by: {cond: rows}, conds, z) -> {cond: np.ndarray}`.
  - `boot_index(n, draws, seed)`.
  - `n2_stats(delta, spec, draws=None, seed=None) -> {"n", "ell", "ell_ci95", "D_sim", "D_sim_ci95", "D_dis", "D_dis_ci90"}`.
  - `verdict(st, c1: {"sim", "dis"}, delta_min, eps, invalid=()) -> {"verdict", "clauses": {"one", "two", "three", "four"} | None, "reasons"}`.
  - `plumbing(rows) -> list[str]`, `block_validity(by, spec) -> list[str]`, `csc_checks(by) -> list[str]`.
  - `dprime_record(delta) -> dict`, `state_conditional(by, z, spec) -> dict`.
  - `thresholds(o, pilot, spec) -> {"ell_hat", "pilot_sd", "c1", "delta_min", "eps", "alt_D_sim", "calibratable"}`.
  - `steps_per_arm(spec, dt) -> int`, `budget_hours(n, wall_s_per_step, spec, dt=1.0) -> float`.
  - `DESIGN_KEYS`, `design_view(n0f_block) -> dict`.
  - `n2_0_outcome(th, oc, budget_h, spec) -> {"outcome", "reason"}`.
  - `sentence(name, res) -> str`.
  - `n_oc.off_arm(values, idx, mu, shift, mult) -> np.ndarray`, `n_oc.simulate(pilot: {"sim", "dis"}, c1, delta_min, eps, spec) -> {"rows", "n", "draws", "boot", "alt_D_sim", "pairing"}`, `n_oc.choose_n(rows, spec) -> int | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/brain/test_n_verdict_oc.py
"""Spec N.8.6: raw-unit statistics with common-seed bootstrap CIs; each verdict branch pinned by a mutation of a
SUPPORTED base (INVALID first, then ①, then ② ③ ④), clause bounds inclusive; validity checks; N2.0's thresholds,
budget, blinding view and outcome order; the OC simulator (block arm = shifted mean, deviations x1 / x2, independent
resample) picks the smallest n passing both spreads."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import n_rules as R
from flymon.brain.n_oc import choose_n, off_arm, simulate
from flymon.brain.n_spec import SPEC

E = np.linspace(-0.1, 0.1, 16)
C1 = {"sim": 0.25, "dis": 0.25}
ZU = {"A": (0.0, 1.0), "P": (0.0, 1.0)}


def _d(sim_on=-1.0, sim_off=-0.2, dis_on=-1.0, dis_off=-1.0, spread=E):
    return {"sim_on": sim_on + spread, "sim_off": sim_off + spread, "dis_on": dis_on + spread,
            "dis_off": dis_off + spread}


def _v(d, c1=C1, invalid=()):
    return R.verdict(R.n2_stats(d, SPEC, draws=2000, seed=1), c1, 0.5, 0.25, list(invalid))


def test_base_case_is_supported():
    v = _v(_d())
    assert v["verdict"] == R.SUPPORTED and v["clauses"] == dict(one=True, two=True, three=True, four=True)


def test_invalid_comes_first():
    v = _v(_d(), invalid=["plumbing check failed: sim_on seed 1"])
    assert v["verdict"] == R.INVALID and v["reasons"] == ["plumbing check failed: sim_on seed 1"]


def test_small_on_learning_is_no_learning():
    v = _v(_d(sim_on=-0.2))
    assert v["verdict"] == R.NO_LEARNING and not v["clauses"]["one"]


def test_on_learning_whose_ci_reaches_zero_is_no_learning():
    d = _d(); d["sim_on"] = -0.5 + np.linspace(-3, 3, 16)
    st = R.n2_stats(d, SPEC, draws=2000, seed=1)
    assert st["ell"]["sim_on"] >= 0.25 and st["ell_ci95"]["sim_on"][0] <= 0
    assert R.verdict(st, C1, 0.5, 0.25)["verdict"] == R.NO_LEARNING


def test_a_small_block_effect_fails_clause_two():
    v = _v(_d(sim_off=-0.8))
    assert v["verdict"] == R.NOT_REPLICATED and v["clauses"] == dict(one=True, two=False, three=True, four=True)


def test_a_dissimilar_block_effect_fails_clause_three():
    v = _v(_d(dis_off=-0.5))
    assert v["verdict"] == R.NOT_REPLICATED and v["clauses"] == dict(one=True, two=True, three=False, four=True)


def test_lost_dissimilar_learning_fails_clause_four():
    v = _v(_d(dis_off=-0.9), c1={"sim": 0.25, "dis": 0.95})
    assert v["verdict"] == R.NOT_REPLICATED and v["clauses"] == dict(one=True, two=True, three=True, four=False)


def _st(**over):
    st = dict(n=16, ell={"sim_on": 0.25, "sim_off": 0.0, "dis_on": 0.25, "dis_off": 0.25},
              ell_ci95={k: [0.01, 1.0] for k in R.N2_ORDER}, D_sim=0.6, D_sim_ci95=[0.5, 0.7], D_dis=0.0,
              D_dis_ci90=[-0.25, 0.25])
    st.update(over)
    return st


def test_clause_bounds_are_inclusive_and_ci_lower_is_strict():
    assert R.verdict(_st(), C1, 0.5, 0.25)["verdict"] == R.SUPPORTED
    zero = _st(ell_ci95={k: [0.0, 1.0] for k in R.N2_ORDER})
    assert R.verdict(zero, C1, 0.5, 0.25)["verdict"] == R.NO_LEARNING
    assert R.verdict(_st(D_sim_ci95=[0.4999, 0.7]), C1, 0.5, 0.25)["clauses"]["two"] is False
    assert R.verdict(_st(D_dis_ci90=[-0.2501, 0.1]), C1, 0.5, 0.25)["clauses"]["three"] is False


def test_n2_stats_resamples_seeds_in_common():
    st = R.n2_stats(_d(), SPEC, draws=500, seed=3)
    assert st["D_sim"] == pytest.approx(0.8) and st["D_sim_ci95"] == pytest.approx([0.8, 0.8])
    assert st["D_dis_ci90"] == pytest.approx([0.0, 0.0], abs=1e-12) and st["n"] == 16


def _probe(A=10.0, P=20.0, kc=0.05, hz=50.0):
    return dict(seed=0, A=A, P=P, kc_frac=kc, kc_spikes=5, kc_max_win_hz=hz, apl_out_per_step=0.1, wall_s=0.01,
                steps=1400)


def _arm(cond, seed, k=0.0, plastic=True, sha=None, **pre_kw):
    return dict(cond=cond, seed=seed, plastic=plastic, csc_sha256=sha or cond.split("_")[1], weights_frac=1.0,
                pre={"x": _probe(**pre_kw), "y": _probe(**pre_kw)}, post={"x": _probe(A=10.0 - k), "y": _probe()})


def test_delta_is_dv_post_minus_pre_and_deltas_pair_by_seed():
    assert R.delta(_arm("sim_on", 1, k=2.5), ZU) == pytest.approx(-2.5)
    by = {"sim_on": [_arm("sim_on", 2), _arm("sim_on", 1, k=1.0)], "sim_off": [_arm("sim_off", 1), _arm("sim_off", 2)]}
    got = R.deltas(by, ["sim_on", "sim_off"], ZU)
    assert got["sim_on"].tolist() == [-1.0, 0.0]                     # seed order
    by["sim_off"][1]["seed"] = 3
    with pytest.raises(ValueError, match="seeds"):
        R.deltas(by, ["sim_on", "sim_off"], ZU)


def test_plumbing_flags_changed_probes_and_moved_weights():
    ok = _arm("sim_on", 1, plastic=False)
    moved = dict(_arm("sim_on", 2, plastic=False), weights_frac=0.99)
    changed = _arm("dis_off", 3, k=1.0, plastic=False)
    bad = R.plumbing([ok, moved, changed])
    assert len(bad) == 2 and "sim_on seed 2" in bad[0] and "dis_off seed 3" in bad[1]


def test_block_validity_reads_the_off_conditions_pre_probes():
    good = {"sim_off": [_arm("sim_off", s) for s in range(8)], "dis_off": [_arm("dis_off", s) for s in range(8)]}
    assert R.block_validity(good, SPEC) == []
    hot = {"sim_off": [_arm("sim_off", s, kc=0.4) for s in range(8)], "dis_off": good["dis_off"]}
    assert any("KC median" in b for b in R.block_validity(hot, SPEC))
    floor = {"sim_off": good["sim_off"], "dis_off": [_arm("dis_off", s, P=0.0) for s in range(3)]
             + [_arm("dis_off", s) for s in range(3, 8)]}
    assert any("dDL" in b and "P" in b for b in R.block_validity(floor, SPEC))


def test_csc_checks():
    by = {c: [_arm(c, 1)] for c in R.N2_ORDER}
    assert R.csc_checks(by) == []
    by["sim_off"] = [_arm("sim_off", 1, sha="on")]
    assert any("sim" in b and "changed no weight" in b for b in R.csc_checks(by))
    by["sim_off"] = [_arm("sim_off", 1), _arm("sim_off", 2, sha="x")]
    assert any("2 CSC" in b for b in R.csc_checks(by))


def test_dprime_record_and_state_conditional_are_records():
    rec = R.dprime_record(_d())
    assert rec["L"]["sim_on"] == pytest.approx(-float(np.mean(_d()["sim_on"]) / np.std(_d()["sim_on"], ddof=1)))
    by = {"sim_on": [_arm("sim_on", 1, k=1.0), _arm("sim_on", 2, k=3.0, P=4.0)]}
    sc = R.state_conditional(by, ZU, SPEC)
    assert sc["sim_on"] == dict(n=1, n_all=2, ell=pytest.approx(1.0))


def test_thresholds_and_an_uncalibratable_pilot():
    th = R.thresholds({"sim": -4.0, "dis": -2.0}, {"sim": np.array([-1.0, -1.2]), "dis": np.array([-0.8, -0.8])},
                      SPEC)
    assert th["c1"] == {"sim": 1.0, "dis": 0.5} and th["delta_min"] == pytest.approx(0.55)
    assert th["eps"] == pytest.approx(0.2) and th["alt_D_sim"] == pytest.approx(0.825) and th["calibratable"]
    bad = R.thresholds({"sim": -4.0, "dis": -2.0}, {"sim": np.array([0.1, 0.2]), "dis": np.array([-1.0, -1.0])}, SPEC)
    assert not bad["calibratable"]


def test_budget_counts_every_arm_and_the_plumbing_reruns():
    assert R.steps_per_arm(SPEC, 1.0) == 4 * 1400 + 12 * 1800 == 27200
    assert R.budget_hours(16, 1e-3, SPEC) == pytest.approx((16 * 6 + 2 * 6) * 27200 * 1e-3 / 16 / 3600)


def test_design_view_is_the_whitelist():
    block = dict(outcome="OPERATING_POINT", run_id="r", selected={"g": 1.0, "c_delta": 1.0}, wall_s_per_step=1e-3,
                 state_shares={}, validity={}, state_flags=[], grid={"secret": 1}, apl_shift={}, drives={})
    assert set(R.design_view(block)) == set(R.DESIGN_KEYS) == {"outcome", "run_id", "selected", "wall_s_per_step",
                                                               "state_shares", "validity", "state_flags"}


def test_n2_0_outcome_order():
    th = dict(calibratable=True, ell_hat={"sim": 1.0, "dis": 1.0})
    assert R.n2_0_outcome(dict(th, calibratable=False), None, 1.0, SPEC)["outcome"] == R.STOP_POWER
    assert R.n2_0_outcome(th, {"n": None}, 100.0, SPEC)["outcome"] == R.STOP_POWER       # power before budget
    assert R.n2_0_outcome(th, {"n": 16}, 48.5, SPEC)["outcome"] == R.STOP_BUDGET
    assert R.n2_0_outcome(th, {"n": 16}, 48.0, SPEC) == {"outcome": R.N2_0_GO, "reason": None}


def test_off_arm_shifts_the_mean_and_scales_deviations():
    v = np.array([-1.0, -2.0, -3.0])
    got = off_arm(v, [0, 2], -2.0, 0.5, 2.0)
    assert got.tolist() == [-2.0 + 0.5 + 2.0 * 1.0, -2.0 + 0.5 + 2.0 * -1.0]


SMALL = replace(SPEC, n_grid=(4, 8), oc_draws=40, oc_boot=200)


def test_a_clear_pilot_picks_the_smallest_n():
    clear = {"sim": -1.0 + 0.05 * np.linspace(-1, 1, 16), "dis": -1.0 + 0.05 * np.linspace(-1, 1, 16)}
    oc = simulate(clear, C1, 0.5, 0.25, SMALL)
    assert oc["n"] == 4 and len(oc["rows"]) == 2 * 2 * 2
    assert all(r["p_supported"] == 0.0 for r in oc["rows"] if r["hyp"] == "null")
    assert all(r["p_supported"] == 1.0 for r in oc["rows"] if r["hyp"] == "alt")
    assert oc == simulate(clear, C1, 0.5, 0.25, SMALL)                           # deterministic


def test_a_noisy_pilot_has_no_n():
    noisy = {"sim": -0.1 + 3.0 * np.linspace(-1, 1, 16), "dis": -0.1 + 3.0 * np.linspace(-1, 1, 16)}
    assert simulate(noisy, C1, 0.05, 0.025, SMALL)["n"] is None


def test_choose_n_needs_both_spreads_and_both_hypotheses():
    rows = [dict(sd_mult=m, hyp=h, n=n, p_supported=p) for n in (4, 8) for m in (1.0, 2.0)
            for h, p in (("null", 0.0), ("alt", 0.9 if (n, m) != (4, 2.0) else 0.7))]
    assert choose_n(rows, SMALL) == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_n_verdict_oc.py -q -o addopts=""`
Expected: FAIL (`ModuleNotFoundError: flymon.brain.n_oc`).

- [ ] **Step 3: Append to `n_rules.py`**

```python
# ================================================================ N2 (N.8.6)
N2_ORDER = ("sim_on", "sim_off", "dis_on", "dis_off")


def _block(side: dict) -> dict:
    return {"A": [[side["x"]["A"], side["y"]["A"]]], "P": [[side["x"]["P"], side["y"]["P"]]]}


def delta(row: dict, z: dict) -> float:
    """Δ = dV_post - dV_pre of one (condition, seed) arm; dV = h4_formula.dv (V = z_A - z_P, X - Y), z units."""
    return float(dv(_block(row["post"]), z)[0] - dv(_block(row["pre"]), z)[0])


def deltas(by: dict, conds, z: dict) -> dict:
    """{cond: Δ per seed, in seed order}; ValueError unless every condition has the same seeds (paired by seed)."""
    seeds, out = None, {}
    for c in conds:
        rows = sorted(by[c], key=lambda r: r["seed"])
        s = [int(r["seed"]) for r in rows]
        if seeds is None:
            seeds = s
        elif s != seeds:
            raise ValueError(f"condition {c} has seeds {s}, not {seeds} (N2 pairs the conditions by seed)")
        out[c] = np.array([delta(r, z) for r in rows])
    return out


def boot_index(n: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, int(n), (int(draws), int(n)))


def n2_stats(delta: dict, spec, draws: int | None = None, seed: int | None = None) -> dict:
    """ℓ_k = -mean Δ_k; D_sim = ℓ_sim,on - ℓ_sim,off, D_dis likewise; CIs from one common-seed resample per draw
    (95% for ℓ and D_sim, 90% for D_dis's TOST)."""
    X = np.stack([np.asarray(delta[k], float) for k in N2_ORDER])
    ell = -X.mean(1)
    idx = boot_index(X.shape[1], spec.boot_draws if draws is None else draws, spec.boot_seed if seed is None else seed)
    B = -X[:, idx].mean(2)
    q = lambda v, a, b: [float(np.percentile(v, a)), float(np.percentile(v, b))]
    return dict(n=int(X.shape[1]), ell={k: float(ell[i]) for i, k in enumerate(N2_ORDER)},
                ell_ci95={k: q(B[i], 2.5, 97.5) for i, k in enumerate(N2_ORDER)},
                D_sim=float(ell[0] - ell[1]), D_sim_ci95=q(B[0] - B[1], 2.5, 97.5),
                D_dis=float(ell[2] - ell[3]), D_dis_ci90=q(B[2] - B[3], 5.0, 95.0))


def verdict(st: dict, c1: dict, delta_min: float, eps: float, invalid=()) -> dict:
    """INVALID first. Then:
    - ① ℓ_sim,on >= c1_sim and ℓ_dis,on >= c1_dis, each 95% CI lower > 0; ¬① -> NO_LEARNING;
    - ② D_sim 95% CI lower >= δ_min;
    - ③ D_dis 90% CI inside [-ε, ε];
    - ④ ℓ_dis,off >= c1_dis.
    SUPPORTED iff ② ③ ④ all hold, else NOT_REPLICATED."""
    if invalid:
        return dict(verdict=INVALID, clauses=None, reasons=list(invalid))
    e, ci = st["ell"], st["ell_ci95"]
    one = bool(e["sim_on"] >= c1["sim"] and e["dis_on"] >= c1["dis"] and ci["sim_on"][0] > 0 and ci["dis_on"][0] > 0)
    two = bool(st["D_sim_ci95"][0] >= delta_min)
    three = bool(-eps <= st["D_dis_ci90"][0] and st["D_dis_ci90"][1] <= eps)
    four = bool(e["dis_off"] >= c1["dis"])
    clauses = dict(one=one, two=two, three=three, four=four)
    v = NO_LEARNING if not one else (SUPPORTED if two and three and four else NOT_REPLICATED)
    return dict(verdict=v, clauses=clauses, reasons=[])


def plumbing(rows: list) -> list:
    """N.8.7's plumbing check on the plasticity-off reruns: probes unchanged and weights exactly w0."""
    bad = []
    for r in rows:
        same = all(r["pre"][k][f] == r["post"][k][f] for k in ("x", "y") for f in ("A", "P", "kc_spikes"))
        if not same or r["weights_frac"] != 1.0:
            bad.append(f"plumbing check failed: {r['cond']} seed {r['seed']} (probes changed or weights moved)")
    return bad


def block_validity(by: dict, spec) -> list:
    """N.8.3 ② / ③ on the block conditions' naive (pre-training) probes (plan reading 10)."""
    x, y_sim = spec.pair_stimuli()["sim"]
    y_dis = spec.pair_stimuli()["dis"][1]
    stim = {x: [r["pre"]["x"] for r in by["sim_off"]], y_sim: [r["pre"]["y"] for r in by["sim_off"]],
            y_dis: [r["pre"]["y"] for r in by["dis_off"]]}
    bad = []
    for s, rows in stim.items():
        c = cell_stats(rows, spec)
        if c["kc_frac_median"] > spec.kc_block_max:
            bad.append(f"{s} block: KC median {c['kc_frac_median']:.3f} > {spec.kc_block_max}")
        if c["runaway_share"] > spec.runaway_share_max:
            bad.append(f"{s} block: sub-window > {spec.runaway_hz:g} Hz in {c['runaway_share']:.3f} of presentations")
        for k in ("A", "P"):
            if c[f"{k}_zero_share"] > spec.zero_share_max:
                bad.append(f"{s} block: readout {k} zero share {c[f'{k}_zero_share']:.3f} > {spec.zero_share_max}")
    return bad


def csc_checks(by: dict) -> list:
    sha = {c: sorted({r["csc_sha256"] for r in rows}) for c, rows in by.items()}
    bad = [f"condition {c} ran on {len(s)} CSC weight vectors" for c, s in sha.items() if len(s) != 1]
    for p in ("sim", "dis"):
        if f"{p}_on" in sha and sha.get(f"{p}_on") == sha.get(f"{p}_off"):
            bad.append(f"{p}: the APL->KC edit changed no weight")
    return bad


def dprime_record(delta: dict) -> dict:
    L = {}
    for k in N2_ORDER:
        d = dprime(delta[k])
        L[k] = None if d is None else -d
    I = None if any(v is None for v in L.values()) else (L["sim_on"] - L["sim_off"]) - (L["dis_on"] - L["dis_off"])
    return dict(L=L, I=I, note="d′ 기반 L·I는 기록만 (N.8.6)")


def state_conditional(by: dict, z: dict, spec) -> dict:
    """N.8.8's record: ℓ over the seeds whose four probes (pre / post x X / Y) are all in the firing state."""
    out = {}
    for c, rows in by.items():
        keep = [r for r in rows if all(r[ph][k]["P"] >= spec.state_p_min for ph in ("pre", "post") for k in ("x", "y"))]
        out[c] = dict(n=len(keep), n_all=len(rows), ell=(-float(np.mean([delta(r, z) for r in keep])) if keep else None))
    return out


# ================================================================ N2.0 (N.8.6, decision ⑥)
def thresholds(o: dict, pilot: dict, spec) -> dict:
    """c1 = 0.25 |o_pair| (N1); δ_min = 0.5 ℓ̂_sim,on, ε = 0.25 ℓ̂_dis,on, alternative D_sim = 0.75 ℓ̂_sim,on (pilot)."""
    ell_hat = {k: -float(np.mean(pilot[k])) for k in ("sim", "dis")}
    return dict(ell_hat=ell_hat, pilot_sd={k: float(np.std(pilot[k], ddof=1)) for k in ("sim", "dis")},
                c1={k: spec.c1_frac * abs(float(o[k])) for k in ("sim", "dis")},
                delta_min=spec.delta_min_frac * ell_hat["sim"], eps=spec.eps_frac * ell_hat["dis"],
                alt_D_sim=spec.alt_frac * ell_hat["sim"],
                calibratable=bool(ell_hat["sim"] > 0 and ell_hat["dis"] > 0))


def steps_per_arm(spec, dt: float) -> int:
    h4 = spec.h4
    probe = int(round((h4.oracle_window.settle_ms + h4.oracle_window.read_ms) / dt))
    trial = int(round((h4.teach_window.settle_ms + h4.teach_present_ms + h4.teach_gap_ms) / dt))
    return 4 * probe + int(h4.teach_trials) * trial


def budget_hours(n: int, wall_s_per_step: float, spec, dt: float = 1.0) -> float:
    """Plan reading 11: every judged and record arm, plus the plumbing reruns, over budget_workers workers."""
    conds = len(spec.n2_conditions) + len(spec.n2_record_conditions)
    items = int(n) * conds + int(spec.plumbing_seeds) * conds
    return items * steps_per_arm(spec, dt) * float(wall_s_per_step) / spec.budget_workers / 3600.0


DESIGN_KEYS = ("outcome", "run_id", "selected", "wall_s_per_step", "state_shares", "validity", "state_flags")


def design_view(n0f: dict) -> dict:
    """What N2.0 may read from block n0f (N.8.6: state shares and validity numbers only; no block KC correlation)."""
    return {k: n0f.get(k) for k in DESIGN_KEYS}


def n2_0_outcome(th: dict, oc: dict | None, budget_h: float, spec) -> dict:
    if not th["calibratable"]:
        return dict(outcome=STOP_POWER, reason=f"pilot ℓ̂ ≤ 0 ({th['ell_hat']}): δ_min, ε and the alternative cannot be set")
    if oc is None or oc.get("n") is None:
        return dict(outcome=STOP_POWER, reason=(f"no n in {tuple(spec.n_grid)} with null ≤ {spec.null_max} and power "
                                                f"≥ {spec.power_min} under both spreads {tuple(spec.sd_mults)}"))
    if budget_h > spec.budget_h:
        return dict(outcome=STOP_BUDGET, reason=f"judgement block {budget_h:.1f} h > {spec.budget_h:g} h")
    return dict(outcome=N2_0_GO, reason=None)


# ================================================================ sentences
def sentence(name: str, res: dict) -> str:
    o = res.get("outcome")
    if name == "n0f":
        if o == OPERATING_POINT:
            s = res["selected"]
            return f"N0f: 작동점 g = {s['g']:g}, c_δ = {s['c_delta']:g}. N.8a를 커밋한 뒤 N0로 간다."
        if o == STOP_DATA_MISMATCH:
            return f"N0: 데이터 불일치 — {res.get('reason')} (STOP_DATA_MISMATCH). 사용자 판단으로 넘긴다."
        return ("N0f: 켬 대역·차단 폭주·판독 바닥 세 조건을 모두 만족하는 (g, c_δ)가 없다 (STOP_NO_OPERATING_POINT). "
                "사용자 판단으로 넘긴다.")
    if name == "n0":
        s = res["similarity"]
        head = (f"N0: Δr = r(4:1, 1:4) − r(4:1, δ-DL) = {s['delta_r']:.3f}, 95% CI "
                f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
        return head + (" — 유사도 순서 통과, N1로 간다." if o == SIMILARITY_GO
                       else " — STOP_SIMILARITY_ORDER. 사용자 판단으로 넘긴다.")
    if name == "n1":
        p = res["pairs"]
        head = f"N1: p0 비슷한 쌍 {p['sim']['p0']}, 다른 쌍 {p['dis']['p0']}"
        if o == N1_GO:
            return head + " — 두 쌍 모두 시험 가능, N2.0으로 간다."
        return (head + " — STOP_UNTESTABLE: 사전 지정 편집 프로토콜(편집식 하나, α 3점)에서 시험이 성립하지 않는다."
                + (f" {res['note']}" if res.get("note") else ""))
    if name == "n2_0":
        if o == N2_0_GO:
            return f"N2.0: n = {res['n']}, 판정 블록 예상 {res['budget_h']:.1f} h. N.8b를 커밋한 뒤 N2 판정으로 간다."
        return f"N2.0: {o} — {res.get('reason')}. 사용자 판단으로 넘긴다."
    if name == "n2":
        v = res["verdict"]
        if o == SUPPORTED:
            return ("N2 SUPPORTED: C3 엔진에서 Hallem 2006 실제 냄새로, APL→KC 선택 차단이 비슷한 쌍의 학습된 변별만 "
                    "떨어뜨린다(Lin 2014 유사 패턴).")
        if o == NOT_REPLICATED:
            return f"N2 NOT_REPLICATED: 학습은 있으나 조항 {[k for k, x in v['clauses'].items() if not x]} 불성립."
        if o == NO_LEARNING:
            return "N2 NO_LEARNING: 학습량이 사전 지정 편집 프로토콜 효과의 1/4에 못 미친다(D.6 (c) 성격의 기록, 판정 아님)."
        return f"N2 INVALID: {'; '.join(v['reasons'])}"
    raise ValueError(f"unknown stage {name!r}")
```

- [ ] **Step 4: Write `n_oc.py`**

```python
"""Spec N.8.6 (decision ⑥): N2.0's operating characteristic of the whole judgement rule (clauses ①-④, bootstrap
included), from the APL-on pilot only (plan readings 6-7).
- The on arm is a resample of the pilot's per-seed Δ, with the same indices for both pairs.
- The block arm is an independent resample of the same pool, mean shifted by D and deviations scaled by 1x or 2x.
- Null: D_sim = D_dis = 0. Alternative: D_sim = 0.75 ℓ̂_sim,on, D_dis = 0.
- n = the smallest grid value with null <= 0.05 and alternative >= 0.8 under both spreads, else None (STOP_POWER)."""
from __future__ import annotations

import math

import numpy as np

from .n_rules import SUPPORTED, n2_stats, verdict


def off_arm(values, idx, mu: float, shift: float, mult: float) -> np.ndarray:
    """Block-arm Δ: mean mu + shift (ℓ_off = ℓ_on - D), deviations of the resampled pilot values scaled by mult."""
    v = np.asarray(values, float)[np.asarray(idx, np.int64)]
    return mu + shift + mult * (v - mu)


def simulate(pilot: dict, c1: dict, delta_min: float, eps: float, spec) -> dict:
    sim, dis = np.asarray(pilot["sim"], float), np.asarray(pilot["dis"], float)
    if sim.shape != dis.shape or sim.size < 2:
        raise ValueError("the pilot needs the same >= 2 seeds for both pairs")
    m = sim.size
    mu = {"sim": float(sim.mean()), "dis": float(dis.mean())}
    alt = spec.alt_frac * -mu["sim"]
    rng = np.random.default_rng(spec.oc_seed)
    rows = []
    for mult in spec.sd_mults:
        for hyp, d_sim in (("null", 0.0), ("alt", alt)):
            for n in spec.n_grid:
                wins = 0
                for _ in range(spec.oc_draws):
                    on, off = rng.integers(0, m, n), rng.integers(0, m, n)
                    d = {"sim_on": sim[on], "dis_on": dis[on],
                         "sim_off": off_arm(sim, off, mu["sim"], d_sim, mult),
                         "dis_off": off_arm(dis, off, mu["dis"], 0.0, mult)}
                    st = n2_stats(d, spec, draws=spec.oc_boot, seed=int(rng.integers(2 ** 62)))
                    wins += verdict(st, c1, delta_min, eps)["verdict"] == SUPPORTED
                p = wins / spec.oc_draws
                rows.append(dict(sd_mult=float(mult), hyp=hyp, n=int(n), p_supported=p,
                                 mc_se=math.sqrt(p * (1 - p) / spec.oc_draws)))
    return dict(rows=rows, n=choose_n(rows, spec), draws=spec.oc_draws, boot=spec.oc_boot, alt_D_sim=alt,
                pairing="independent on / block resamples of the pilot pool (plan reading 6)")


def choose_n(rows: list, spec) -> int | None:
    for n in spec.n_grid:
        cell = [r for r in rows if r["n"] == n]
        if cell and all((r["p_supported"] <= spec.null_max) if r["hyp"] == "null" else (r["p_supported"] >= spec.power_min)
                        for r in cell):
            return int(n)
    return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_n_rules.py tests/brain/test_n_verdict_oc.py -q -o addopts=""`
Expected: PASS (16 + 22 tests).

- [ ] **Step 6: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_rules.py flymon/brain/n_oc.py tests/brain/test_n_verdict_oc.py
git commit -m "feat(n): N2 raw-unit statistics and verdict (mutation-pinned), validity checks, N2.0 thresholds, budget, design view, joint-rule OC"
```

---
### Task 9: `n_cli`, `run_n0f.py`, `run_n0.py`

**Files:**
- Create: `flymon/brain/n_cli.py`, `scripts/run_n0f.py`, `scripts/run_n0.py`
- Test: `tests/test_run_n.py`

**Interfaces:**
- Consumes: `l_cli.check_committed`, `load_c3_record`, `other_code`, `refuse`, `run_id`, `same_code`; `h3_store.ROOT`, `code_key`, `git_state`, `sha256_file`; `n_store`; `n_measure.HASHED_FILES`, `MEASURE_FILES`, `NCache`, `NMeasurer`; `n_spec.SPEC`, `smoke`; `odor_real.*`; `n_rules.*`; `fly_pool.FlyPool` (in `make_measurer` only).
- Produces:
  - `n_cli.ORDER = ("n0f", "n0", "n1", "n2_0", "n2")`, `SMOKE_OUT`, `SMOKE_SUMMARY`, `POOL_TIMEOUT_S`.
  - Gate helpers: `out_allowed(out)`, `read_previous(summary, need, smoke, committed) -> (doc, why)`, `later_blocks(name, doc) -> str | None`, `head_spec(path) -> str | None`, `spec_note(text, marker, rid) -> str | None`.
  - Hooks: `git_state(files)`, `code_keys(npz) -> (measure_key: dict, manifest: dict)`, `load_c3(spec) -> (Params, readout, z, pools)`, `m0d_sha(spec) -> str`, `model_types(npz) -> list[str]`, `make_measurer(ctx) -> (NMeasurer, pool)`.
  - Stage helpers: `data_dir(spec) -> Path` (relative to ROOT), `operating_point(doc, spec, smoke) -> (g, c_delta)`, `stimuli_at(ctx, names, g, c_delta) -> dict`, `spec_record(spec) -> dict`, `write_report(out, rid, name, res, params_list) -> Path`.
  - `Ctx` (fields `args, spec, smoke, doc, rid, out, key, c3, readout, z, types, hooks`).
  - `main_stage(name, argv, body, hooks, *, need=(), outcomes=None, note=None, spec=None, require_root=True, doc_help=None) -> int`, where `body(ctx) -> (res: dict with "outcome" and "sentence", exit code)`.
  - Scripts: `run_n0f.main(argv=None, spec=None, require_root=True) -> int` and `run_n0.main(...)`. Each script exposes the hook names at module level (tests patch them there) and `body(ctx)`.
  - Block `n0f`:
    - `outcome`, `selected`, `checks`, `grid`, `lin_totals`, `wall_s_per_step`, `budget_estimate_h`;
    - at a selected point: `state_shares`, `validity`, `state_flags`, `apl_shift`, `drives`;
    - on a data stop: `reason`.
  - Block `n0`: `outcome`, `point`, `similarity`, `kc`, `drives`.
  - Every block also carries `name`, `run_id`, `smoke`, `smoke_bypass`, `measure_key`, `code`, `git`, `inputs` (`m0d_sha256`, `data_sha256`), `upstream`, `c3`, `spec`, `argv`, `wall_s`, `report`, `report_sha256`, `sentence`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_n.py
"""Spec N.8.9 / plan readings 12-14: the N CLIs refuse before any pool on another root, --out, a dirty tree, an
uncommitted or other-code summary, a stopped upstream outcome, a missing N.8a / N.8b paragraph (or one citing another
run), a later block; N0f writes STOP_DATA_MISMATCH / STOP_NO_OPERATING_POINT blocks (exit 5) and never asks for KC
vectors; each stage's flow with a stand-in measurer."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import n_cli
from flymon.brain.config import Params
from flymon.brain.n_rules import (N1_GO, N2_0_GO, OPERATING_POINT, SIMILARITY_GO, STOP_DATA_MISMATCH,
                                  STOP_NO_OPERATING_POINT, STOP_UNTESTABLE, SUPPORTED, INVALID)
from flymon.brain.n_spec import SPEC, smoke
from flymon.brain.odor_real import load_table

ROOT = Path(__file__).resolve().parents[1]
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])
KEY = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}                      # dV = (A_x - P_x) - (A_y - P_y)
TYPES = sorted({"ORN_" + p for v in load_table(ROOT / SPEC.data_dir, SPEC.sha_pins()).glomeruli.values() for p in v})
STAGES = ["run_n0f", "run_n0"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def _prow(A=10.0, P=20.0, kc=0.05, hz=50.0):
    return dict(seed=0, A=A, P=P, kc_frac=kc, kc_spikes=5, kc_max_win_hz=hz, apl_out_per_step=0.1, wall_s=0.014,
                steps=1400)


LEARN = {"sim_on": 1.0, "sim_off": 0.2, "dis_on": 1.0, "dis_off": 1.0, "sim_all": 0.1, "dis_all": 0.1}


class FakePool:
    closed = False

    def close(self):
        FakePool.closed = True


class FakeMeasurer:
    """Stand-in NMeasurer: KC fractions and readouts from constructor values; learning -k per condition."""

    def __init__(self, kc=0.05, learn=LEARN, leak=False, o=(-3.0, -4.0, -5.0)):
        self.kc, self.learn, self.leak, self.o, self.calls = kc, learn, leak, o, []

    def presentations(self, params, items, seeds, readout):
        self.calls.append(("presentations", [e for e, _ in items]))
        return [[dict(_prow(kc=self.kc), seed=s, edit=e, csc_sha256=e) for s in seeds] for e, _ in items]

    def kc_vectors(self, params, items, seeds, readout):
        self.calls.append(("kc_vectors", [e for e, _ in items]))
        out = []
        for i, (e, _) in enumerate(items):
            base = list(range(60, 80)) if i % 3 == 2 else list(range(i % 3 * 2, 20 + i % 3 * 2))
            out.append([dict(_prow(), seed=s, edit=e, csc_sha256=e, n_kc=100, kc_fired=base + [90 + j % 10])
                        for j, s in enumerate(seeds)])
        return out

    def punish_oracle(self, params, pairs, readout, z, punish_type):
        self.calls.append(("punish_oracle", [p["name"] for p in pairs]))
        n = len(self.o)
        pre = {"A": [[10.0, 10.0]] * n, "P": [[5.0, 5.0]] * n}
        post = {"A": [[10.0 + x, 10.0] for x in self.o], "P": [[5.0, 5.0]] * n}
        return [{"alpha_punish": 0.5, "select": {"pre": pre, "punish": {"0.5": {"P": post, "change": -4.0}}},
                 "report": {"pre": pre, "P": post}, "kc": {"x": {}, "y": {}, "jaccard": 0.1}, "csc_sha256": "none"}
                for _ in pairs]

    def arms(self, params, items, readout, punish_type):
        self.calls.append(("arms", sorted({i["cond"] for i in items})))
        rows = []
        for i in items:
            k = (self.learn[i["cond"]] + 0.05 * (i["seed"] % 3 - 1)) if i["plastic"] else 0.0
            post_x = _prow(A=10.0 - k)
            if self.leak and not i["plastic"]:
                post_x["A"] = 9.0
            rows.append(dict(seed=i["seed"], edit=i["edit"], plastic=i["plastic"], csc_sha256=i["edit"],
                             pre={"x": _prow(), "y": _prow()}, post={"x": post_x, "y": _prow()},
                             weights_frac=0.9 if i["plastic"] else 1.0, wall_s=0.1, cond=i["cond"]))
        return rows


def _patch(mod, monkeypatch, measurer=None, head=None, z=ZU):
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda npz: KEY)
    monkeypatch.setattr(mod, "m0d_sha", lambda spec: "s" * 64)
    monkeypatch.setattr(mod, "load_c3", lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, z, None))
    monkeypatch.setattr(mod, "model_types", lambda npz: TYPES)
    monkeypatch.setattr(mod, "head_spec", lambda path: head)
    monkeypatch.setattr(mod, "make_measurer",
                        lambda ctx: (measurer, FakePool()) if measurer else pytest.fail("no pool may start here"))
    monkeypatch.setattr(n_cli, "out_allowed", lambda out: True)


def _block(name, **kw):
    b = dict(outcome=None, run_id=f"rid-{name}", measure_key="k" * 64, code={"key": "m" * 64},
             inputs={"m0d_sha256": "s" * 64}, upstream={})
    b.update(kw)
    return b


N0F = _block("n0f", outcome=OPERATING_POINT, selected={"g": 1.0, "c_delta": 1.0}, wall_s_per_step=1e-5,
             state_shares={}, validity={}, state_flags=[])


def _write(tmp_path, doc, smoke_=True):
    p = tmp_path / ("results/n/smoke/n_real_odour.json" if smoke_ else "results/summary/n_real_odour.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


@pytest.mark.parametrize("name", STAGES)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_out_outside_results_n(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/m0d/n/x"]) == 2 and "results/n/" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_dirty_hashed_files(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/n_jobs.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_spec_note_needs_a_line_start_marker_citing_the_run():
    assert "no N.8a" in n_cli.spec_note("x **N.8a** mentioned mid-line rid-1\n", "N.8a", "rid-1")
    assert "does not cite" in n_cli.spec_note("**N.8a N0f 결과 (run `rid-0`)**\n", "N.8a", "rid-1")
    assert n_cli.spec_note("text\n**N.8a N0f 결과 (2026-10-01, run `rid-1`)** — outcome\n", "N.8a", "rid-1") is None
    assert n_cli.spec_note("### N.8b N2.0\nrun rid-2\n", "N.8b", "rid-2") is None
    assert "not tracked" in n_cli.spec_note(None, "N.8a", "rid-1")


def test_order_later_blocks_and_read_previous(tmp_path):
    assert n_cli.ORDER == ("n0f", "n0", "n1", "n2_0", "n2")
    assert "later blocks ['n1']" in n_cli.later_blocks("n0", {"n0f": {}, "n1": {}})
    assert n_cli.later_blocks("n2", {"n0f": {}}) is None
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"n0f": {}}))
    assert n_cli.read_previous(s, ["n0f"], False, lambda *a: "uncommitted")[1] == "uncommitted"
    assert "lacks the blocks ['n0']" in n_cli.read_previous(s, ["n0f", "n0"], False, lambda *a: None)[1]
    assert n_cli.read_previous(tmp_path / "none.json", [], False, lambda *a: None) == ({}, None)


def test_n0f_smoke_selects_the_point_and_never_asks_for_kc_vectors(tmp_path, monkeypatch, capsys):
    mod = _script("run_n0f")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    doc = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert doc["outcome"] == OPERATING_POINT and doc["selected"] == {"g": 1.0, "c_delta": 1.0}
    assert [c[0] for c in fm.calls] == ["presentations"]
    assert sorted(set(fm.calls[0][1])) == ["apl_all_zero", "apl_to_kc_zero", "none"]
    assert set(doc["state_shares"]["4:1"]) == {"on", "block", "all"} and doc["validity"]["ok"]
    assert doc["lin_totals"]["totals"]["signed"]["IA"] == 2040.0 and doc["smoke"] is True
    assert Path(doc["report"]).exists() and "N0f: 작동점" in capsys.readouterr().out


def test_n0f_writes_a_no_operating_point_block_and_exits_5(tmp_path, monkeypatch):
    mod = _script("run_n0f")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(kc=0.5))
    assert mod.main(["--smoke"], require_root=False) == 5
    doc = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert doc["outcome"] == STOP_NO_OPERATING_POINT and doc["selected"] is None and "state_shares" not in doc


def test_n0f_data_mismatch_is_a_block_and_starts_no_pool(tmp_path, monkeypatch):
    from dataclasses import replace
    mod = _script("run_n0f")
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "odor"
    shutil.copytree(ROOT / SPEC.data_dir, d)
    (d / "hallem2006_subset.csv").write_text((d / "hallem2006_subset.csv").read_text().replace("Or22a,236", "Or22a,NA"))
    _patch(mod, monkeypatch, None)                                     # make_measurer fails the test if called
    assert mod.main(["--smoke"], spec=replace(smoke(SPEC), data_dir=str(d)), require_root=False) == 5
    doc = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert doc["outcome"] == STOP_DATA_MISMATCH and "sha256" in doc["reason"]


def test_n0_refuses_a_stopped_or_unnoted_n0f(tmp_path, monkeypatch, capsys):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None, head="no notes here\n")
    stopped = dict(N0F, outcome=STOP_NO_OPERATING_POINT)
    _write(tmp_path, {"n0f": stopped}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "STOP" in capsys.readouterr().err
    _write(tmp_path, {"n0f": N0F}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "N.8a" in capsys.readouterr().err
    monkeypatch.setattr(mod, "head_spec", lambda p: "**N.8a N0f 결과 (run `rid-other`)**\n")
    assert mod.main([], require_root=False) == 2 and "does not cite" in capsys.readouterr().err


def test_n0_refuses_other_code_another_m0d_and_a_later_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None, head="**N.8a (run `rid-n0f`)**\n")
    _write(tmp_path, {"n0f": dict(N0F, measure_key="x" * 64)}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "other code" in capsys.readouterr().err
    _write(tmp_path, {"n0f": dict(N0F, inputs={"m0d_sha256": "t" * 64})}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "m0d.json" in capsys.readouterr().err
    _write(tmp_path, {"n0f": N0F, "n1": _block("n1")}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "later blocks" in capsys.readouterr().err


def test_n0_passes_the_similarity_gate(tmp_path, monkeypatch):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head="**N.8a N0f 결과 (run `rid-n0f`)**\n")
    _write(tmp_path, {"n0f": N0F}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    doc = json.loads(Path("results/summary/n_real_odour.json").read_text())
    b = doc["n0"]
    assert b["outcome"] == SIMILARITY_GO and b["point"] == {"g": 1.0, "c_delta": 1.0}
    assert b["upstream"] == {"n0f": "rid-n0f"} and fm.calls == [("kc_vectors", ["none"] * 3)]
    assert set(b["drives"]) == {"4:1", "1:4", "dDL"} and doc["n0f"] == N0F
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_run_n.py -q -o addopts=""`
Expected: FAIL (`ImportError: cannot import name 'n_cli'`).

- [ ] **Step 3: Write `n_cli.py`**

```python
"""The helpers every CLI of spec appendix N shares (N.8.9, plan readings 12-14).
- l_cli's refusal, run id, commit check, C3 record and manifest check, re-exported.
- N's --out rule (results/n/), code keys over N's files, the block order, the summary gate and the later-block refusal.
- The spec-note gate: an N.8a / N.8b paragraph committed in the spec at HEAD, citing the upstream block's run id.
- The hooks a script passes in, so its tests patch them on the script module.
- `main_stage`: every refusal before any pool, in order, then the stage body, the report and the summary block."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import h3_store, n_store
from .h3_store import ROOT, code_key, sha256_file
from .l_cli import check_committed, load_c3_record, other_code, refuse, run_id, same_code  # noqa: F401
from .n_measure import HASHED_FILES, MEASURE_FILES, NCache, NMeasurer
from .n_spec import SPEC, NSpec, smoke
from .odor_real import DataMismatch, cap_hz, glomerular, load_table, stimuli

POOL_TIMEOUT_S = 7200                    # one N2 arm is ~27 200 steps; a round of arms well inside two hours
ORDER = ("n0f", "n0", "n1", "n2_0", "n2")
SMOKE_OUT = "results/n/smoke"
SMOKE_SUMMARY = "results/n/smoke/n_real_odour.json"


def out_allowed(out) -> bool:
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(n_store.ALLOWED_DIR)


def git_state(files=HASHED_FILES) -> dict:
    return h3_store.git_state(files=files)


def code_keys(npz) -> tuple:
    """(measure key over MEASURE_FILES, manifest over HASHED_FILES); a missing hashed file is a refusal (exit 2)."""
    missing = [f for f in HASHED_FILES if not (ROOT / f).exists()]
    if missing:
        refuse(f"the hashed files {missing} do not exist (the manifest covers every N module, script and data file)")
        raise SystemExit(2)
    return code_key(npz, files=MEASURE_FILES), code_key(npz, files=HASHED_FILES)


def read_previous(summary, need, smoke: bool, committed) -> tuple:
    """(doc, None) or (None, refusal): outside --smoke the summary is git-tracked and clean at HEAD (`committed`) when
    any block is needed; then every needed block is present. --smoke reads only a summary under results/n/."""
    if smoke:
        if not out_allowed(summary):
            return None, f"a --smoke run reads and writes only a smoke summary under {n_store.ALLOWED_DIR}, not {summary}"
    elif need:
        why = committed(summary, list(need))
        if why:
            return None, why
    try:
        doc = json.loads(Path(summary).read_text()) if need or Path(summary).exists() else {}
    except (OSError, ValueError) as e:
        return None, f"no usable summary at {summary}: {e}"
    missing = [b for b in need if not isinstance(doc.get(b) if isinstance(doc, dict) else None, dict)]
    if missing:
        return None, f"{summary} lacks the blocks {missing} (each earlier stage's block comes first)"
    return doc, None


def later_blocks(name: str, doc: dict) -> str | None:
    later = [b for b in ORDER[ORDER.index(name) + 1:] if isinstance(doc, dict) and b in doc]
    return None if not later else (f"the summary already holds the later blocks {later}: block {name} is not "
                                   f"rewritten under them (start a new summary, or --smoke)")


def head_spec(path) -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def spec_note(text: str | None, marker: str, rid) -> str | None:
    """Plan reading 12: a line starting "**N.8a" (or a heading "### N.8a") in the committed spec, and the upstream
    block's run id somewhere after it."""
    if text is None:
        return "the spec is not tracked at HEAD"
    m = re.search(rf"^(?:#+\s*|\*\*){re.escape(marker)}\b", text, re.M)
    if not m:
        return (f"the spec at HEAD has no {marker} paragraph: write it (scripts/write_n_notes.py) and commit it before "
                f"this stage (N.8.9)")
    if str(rid) not in text[m.start():]:
        return f"the spec's {marker} paragraph does not cite block run_id {rid}"
    return None


def load_c3(spec) -> tuple:
    return load_c3_record(spec.l.m0d_path, spec.l)


def m0d_sha(spec) -> str:
    return sha256_file(spec.l.m0d_path)


def model_types(npz) -> list:
    from .circuits import Populations
    from .connectome import Connectome
    return sorted(str(t) for t in Populations.from_connectome(Connectome.load(str(npz))).receptor_types)


def make_measurer(ctx) -> tuple:
    from .fly_pool import FlyPool
    pool = FlyPool(ctx.args.npz, ctx.c3, flies=[{}] * ctx.args.workers, workers=ctx.args.workers,
                   punish_type=ctx.spec.h3.punish_type, reward_type=ctx.spec.h3.reward_type, timeout_s=POOL_TIMEOUT_S)
    return NMeasurer(pool, ctx.spec, NCache(ctx.out / "cache", ctx.key, ctx.rid)), pool


def operating_point(doc: dict, spec, smoke: bool) -> tuple:
    sel = (doc.get("n0f") or {}).get("selected")
    if sel:
        return float(sel["g"]), float(sel["c_delta"])
    if smoke:
        return tuple(float(v) for v in spec.smoke_point)
    raise ValueError("block n0f selected no operating point")


def data_dir(spec) -> Path:
    """spec.data_dir, relative to the repository root unless absolute (a CLI never depends on its cwd for the data)."""
    p = Path(spec.data_dir)
    return p if p.is_absolute() else ROOT / p


def stimuli_at(ctx, names, g: float, c_delta: float) -> dict:
    glom = glomerular(load_table(data_dir(ctx.spec), ctx.spec.sha_pins()), ctx.types)
    return stimuli(glom, names, g, c_delta, ctx.spec.mixtures_dict(), ctx.c3.max_rate_hz, ctx.spec.h3.strength,
                   cap_hz(ctx.c3))


def spec_record(spec) -> dict:
    return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.name != "l"}


def write_report(out, rid: str, name: str, res: dict, params_list) -> Path:
    base = Path(out) / "runs" / f"{rid}-{name}"
    path = n_store.write_json(Path(f"{base}.json"), res, params_list)
    md = f"# N {name} — {rid}\n\n{res.get('sentence', '')}\n\noutcome: `{res.get('outcome')}`\n"
    n_store.write_bytes(Path(f"{base}.md"), md.encode(), params_list)
    return path


@dataclass
class Ctx:
    args: argparse.Namespace
    spec: NSpec
    smoke: bool
    doc: dict
    rid: str
    out: Path
    key: dict
    c3: object
    readout: dict
    z: dict
    types: list
    hooks: dict


def main_stage(name: str, argv, body, hooks: dict, *, need=(), outcomes=None, note=None, spec=None,
               require_root: bool = True, doc_help: str | None = None) -> int:
    """Refusals (exit 2) before any pool, in order:
    1. another directory;
    2. --out outside results/n/;
    3. dirty hashed files (unless --allow-dirty);
    4. the summary gate (committed outside --smoke; blocks present);
    5. a later block (outside --smoke);
    6. an upstream outcome not in `outcomes` (outside --smoke; recorded in smoke_bypass under --smoke);
    7. the spec note `note` = (marker, block) (outside --smoke);
    8. an upstream block produced on another run of its own upstream;
    9. other code;
    10. another m0d.json than block n0f's;
    11. no usable C3 record.
    Then body(ctx) -> (res, exit code); a DataMismatch inside a later stage's body is a refusal. The report goes under
    <out>/runs/ and block `name` into the summary (a --smoke run: its smoke summary)."""
    ap = argparse.ArgumentParser(description=doc_help, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or (SMOKE_OUT if a.smoke else "results/n/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {n_store.ALLOWED_DIR} of the repository root")
    summary = Path(a.summary or (SMOKE_SUMMARY if a.smoke else n_store.SUMMARY))
    git = hooks["git_state"](HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    doc, why = read_previous(summary, list(need), a.smoke, hooks["check_committed"])
    if why:
        return refuse(why)
    if not a.smoke:
        why = later_blocks(name, doc)
        if why:
            return refuse(why)
    bypass = []
    for b, allowed in (outcomes or {}).items():
        got = doc[b].get("outcome")
        if got not in allowed:
            if not a.smoke:
                return refuse(f"block {b} ended {got}, not {list(allowed)}: every STOP goes to the user (N.6)")
            bypass.append(f"{b}={got}")
    if note and not a.smoke:
        why = spec_note(hooks["head_spec"](spec.spec_path), note[0], doc[note[1]].get("run_id"))
        if why:
            return refuse(why)
    for b in need:
        for ub, urid in (doc[b].get("upstream") or {}).items():
            if (doc.get(ub) or {}).get("run_id") != urid:
                return refuse(f"block {b} was produced on another {ub} run ({urid}; the summary holds "
                              f"{(doc.get(ub) or {}).get('run_id')})")
    key, manifest = hooks["code_keys"](a.npz)
    if need:
        why = other_code(doc, list(need), key["key"], manifest["key"], same_code)
        if why:
            return refuse(why)
    m0d = hooks["m0d_sha"](spec)
    if need and ((doc["n0f"].get("inputs") or {}).get("m0d_sha256") != m0d):
        return refuse(f"results/summary/m0d.json (sha256 {m0d[:12]}) is not the one block n0f ran on")
    try:
        c3, readout, z, _ = hooks["load_c3"](spec)
    except ValueError as e:
        return refuse(str(e))
    rid = run_id()
    ctx = Ctx(args=a, spec=spec, smoke=a.smoke, doc=doc, rid=rid, out=out, key=key, c3=c3, readout=dict(readout),
              z=z, types=hooks["model_types"](a.npz), hooks=hooks)
    t0 = time.time()
    try:
        res, code = body(ctx)
    except DataMismatch as e:
        return refuse(f"the data no longer match the pins block n0f ran on: {e}")
    res = dict(res, name=name, run_id=rid, smoke=a.smoke, smoke_bypass=bypass, measure_key=key["key"], code=manifest,
               git=git, inputs=dict(m0d_sha256=m0d, data_sha256=spec.sha_pins()),
               upstream={b: doc[b].get("run_id") for b in need}, c3=dict(readout=readout, z=z),
               spec=spec_record(spec), argv=list(argv or []), wall_s=time.time() - t0)
    report = write_report(out, rid, name, res, [c3])
    n_store.write_summary_block(summary, name, dict(res, report=str(report), report_sha256=sha256_file(report)), [c3])
    print(res.get("sentence", ""))
    return code
```

- [ ] **Step 4: Write `scripts/run_n0f.py`**

```python
#!/usr/bin/env python3
"""Spec N.8.3 (N0f, record only, no learning; with N.8.2's data stop).

Steps:
1. Load data/odor/ against its pins. A defect ends STOP_DATA_MISMATCH (block written, exit 5, no pool).
2. Record the Lin 2014 totals under both conventions.
3. Present every N0f stimulus of every (g, c_δ) on the N0f seeds, APL on / APL->KC block / all-output block. This is
   presentation_job: scalars only, never a KC vector.
4. Apply the operating-point rule, which ends in OPERATING_POINT or STOP_NO_OPERATING_POINT (exit 5).

    uv run python scripts/run_n0f.py                                          # the controller only (R1)
    uv run python scripts/run_n0f.py --smoke --allow-dirty --workers 4        # results/n/smoke/

Refusals (exit 2) are n_cli.main_stage's. Block "n0f" is written for every outcome."""
from __future__ import annotations

import sys

from flymon.brain.h3_store import canonical
from flymon.brain.n_cli import (check_committed, code_keys, data_dir, git_state, head_spec, load_c3,  # noqa: F401
                                m0d_sha, main_stage, make_measurer, model_types)
from flymon.brain.n_rules import (OPERATING_POINT, STOP_DATA_MISMATCH, budget_hours, cell_stats, point_key,
                                  select_point, sentence, state_flags, wall_per_step)
from flymon.brain.odor_real import DataMismatch, cap_hz, glomerular, lin_record, load_table, stimuli


def hooks() -> dict:
    return dict(check_committed=check_committed, code_keys=code_keys, git_state=git_state, head_spec=head_spec,
                load_c3=load_c3, m0d_sha=m0d_sha, make_measurer=make_measurer, model_types=model_types)


def body(ctx) -> tuple:
    spec = ctx.spec
    try:
        table = load_table(data_dir(spec), spec.sha_pins())
        glom = glomerular(table, ctx.types)
    except DataMismatch as e:
        res = dict(outcome=STOP_DATA_MISMATCH, reason=str(e))
        return dict(res, sentence=sentence("n0f", res)), 5
    grid = {(g, c): stimuli(glom, spec.n0f_stimuli, g, c, spec.mixtures_dict(), ctx.c3.max_rate_hz,
                            spec.h3.strength, cap_hz(ctx.c3))
            for g in spec.g_grid for c in spec.c_delta_grid}
    odours = {}
    for st in grid.values():                       # an odour is its drive: the non-δ stimuli are shared across c_δ
        for rec in st.values():
            odours.setdefault(canonical(rec["odor"]), rec["odor"])
    items = [(edit, o) for _, edit in spec.conditions for o in odours.values()]
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.presentations(ctx.c3, items, spec.n0f_seeds, ctx.readout)
    finally:
        pool.close()
    by = {(e, canonical(o)): r for (e, o), r in zip(items, rows)}
    cells = {gc: {s: {cond: cell_stats(by[(edit, canonical(rec["odor"]))], spec) for cond, edit in spec.conditions}
                  for s, rec in st.items()} for gc, st in grid.items()}
    sel = select_point(cells, spec)
    wps = wall_per_step([r for rr in rows for r in rr])
    res = dict(outcome=sel["outcome"], selected=sel["selected"], checks=sel["checks"],
               lin_totals=lin_record(table, dict(spec.lin_totals), spec.lin_tol),
               grid={point_key(*gc): v for gc, v in cells.items()}, wall_s_per_step=wps,
               budget_estimate_h={str(n): budget_hours(n, wps, spec, ctx.c3.dt) for n in spec.n_grid})
    if sel["selected"]:
        gc = (sel["selected"]["g"], sel["selected"]["c_delta"])
        at = cells[gc]
        res.update(state_shares={s: {cond: at[s][cond]["firing_share"] for cond, _ in spec.conditions} for s in at},
                   validity=sel["checks"][point_key(*gc)], state_flags=state_flags(at, spec),
                   apl_shift={s: {cond: at[s][cond]["apl_out_mean"] - at[s]["on"]["apl_out_mean"]
                                  for cond in ("block", "all")} for s in at},
                   drives={s: {k: rec[k] for k in ("drive_hz", "clipped_hz", "clipped_total_hz", "capped", "weights")}
                           for s, rec in grid[gc].items()})
    return dict(res, sentence=sentence("n0f", res)), (0 if sel["outcome"] == OPERATING_POINT else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n0f", argv, body, hooks(), need=(), spec=spec, require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `scripts/run_n0.py`**

```python
#!/usr/bin/env python3
"""Spec N.8.4 (N0's similarity gate). At block n0f's operating point, with APL on, it presents the three judged
stimuli on the activity seeds (kc_vectors_job) and builds the KC firing-probability vectors. It then computes
Δr = r(4:1, 1:4) - r(4:1, δ-DL) with a common-seed bootstrap 95% CI: SIMILARITY_GO iff the lower bound > 0, else
STOP_SIMILARITY_ORDER (exit 5). Records: r(4:1, 1:4), per-stimulus KC activity and sub-window, the drive tables.

    uv run python scripts/run_n0.py                                           # the controller only (R2)
    uv run python scripts/run_n0.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's, in particular: block n0f not committed, not OPERATING_POINT, or no committed
**N.8a paragraph citing block n0f's run id (outside --smoke)."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (check_committed, code_keys, git_state, head_spec, load_c3, m0d_sha,  # noqa: F401
                                main_stage, make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_rules import OPERATING_POINT, SIMILARITY_GO, cell_stats, sentence, similarity


def hooks() -> dict:
    return dict(check_committed=check_committed, code_keys=code_keys, git_state=git_state, head_spec=head_spec,
                load_c3=load_c3, m0d_sha=m0d_sha, make_measurer=make_measurer, model_types=model_types)


def body(ctx) -> tuple:
    spec = ctx.spec
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.kc_vectors(ctx.c3, [("none", st[s]["odor"]) for s in spec.judged_stimuli], spec.act_seeds,
                            ctx.readout)
    finally:
        pool.close()
    by = dict(zip(spec.judged_stimuli, rows))
    sim = similarity({s: [r["kc_fired"] for r in rr] for s, rr in by.items()}, rows[0][0]["n_kc"], spec)
    res = dict(outcome=sim["outcome"], point=dict(g=g, c_delta=c), similarity=sim,
               kc={s: cell_stats(rr, spec) for s, rr in by.items()},
               drives={s: {k: st[s][k] for k in ("drive_hz", "clipped_hz", "clipped_total_hz", "capped", "weights")}
                       for s in spec.judged_stimuli})
    return dict(res, sentence=sentence("n0", res)), (0 if sim["outcome"] == SIMILARITY_GO else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n0", argv, body, hooks(), need=("n0f",), outcomes={"n0f": (OPERATING_POINT,)},
                      note=("N.8a", "n0f"), spec=spec, require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_n.py -q -o addopts=""`
Expected: PASS (14 tests).

The dirty-file and out tests reach `git_state` / `out_allowed` before `code_keys`, so they pass although Task 10's scripts do not exist yet. The real `code_keys` would refuse, so every flow test patches it.

- [ ] **Step 7: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add flymon/brain/n_cli.py scripts/run_n0f.py scripts/run_n0.py tests/test_run_n.py
git commit -m "feat(n): n_cli stage skeleton (refusals, N.8a/N.8b note gate, hooks) and the N0f / N0 CLIs"
```

---

### Task 10: `run_n1.py`, `run_n2_pilot.py`, `run_n2_judge.py`

**Files:**
- Create: `scripts/run_n1.py`, `scripts/run_n2_pilot.py`, `scripts/run_n2_judge.py`
- Modify: `tests/test_run_n.py` (append; new in this plan)

**Interfaces:**
- Consumes: Task 9's `n_cli` names; `n_rules.*` (Tasks 7–8); `n_oc.simulate`.
- Produces:
  - `main(argv=None, spec=None, require_root=True) -> int` in each script.
  - Block `n1`: `outcome`, `note`, `point`, `pairs: {"sim"|"dis": n1_pair}`.
  - Block `n2_0`:
    - `outcome`, `reason`, `n`, `c1`, `delta_min`, `eps`, `ell_hat`, `pilot_sd`, `alt_D_sim`;
    - `pilot: {"sim"|"dis": [Δ per seed]}`, `pilot_seeds`, `oc` (or None), `budget_h`, `design_view`, `pilot_states`.
  - Block `n2`:
    - `outcome` (= verdict), `verdict`, `stats`, `thresholds: {"n", "c1", "delta_min", "eps"}`, `invalid`;
    - `per_seed`, `dprime_record`, `all_output_block`, `state_conditional`, `csc_sha256`, `post_verdict_kc`.

- [ ] **Step 1: Append the failing tests**

```python
# appended to tests/test_run_n.py
LATER = ["run_n1", "run_n2_pilot", "run_n2_judge"]
N0 = _block("n0", outcome=SIMILARITY_GO, upstream={"n0f": "rid-n0f"})
N1 = _block("n1", outcome=N1_GO, pairs={"sim": {"o": -2.0}, "dis": {"o": -2.0}},
            upstream={"n0f": "rid-n0f", "n0": "rid-n0"})
N2_0 = _block("n2_0", outcome=N2_0_GO, n=3, c1={"sim": 0.25, "dis": 0.25}, delta_min=0.5, eps=0.25,
              upstream={"n0f": "rid-n0f", "n0": "rid-n0", "n1": "rid-n1"})
HEAD_A = "**N.8a N0f 결과 (run `rid-n0f`)**\n"
HEAD_AB = HEAD_A + "**N.8b N2.0 보정 결과 (run `rid-n2_0`)**\n"


@pytest.mark.parametrize("name", LATER)
def test_later_stages_refuse_outside_the_root_and_outside_results_n(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/m0d/n/x"]) == 2 and "results/n/" in capsys.readouterr().err


def test_n1_needs_n8a_and_reads_both_pairs(tmp_path, monkeypatch, capsys):
    mod = _script("run_n1")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head="nothing\n")
    _write(tmp_path, {"n0f": N0F, "n0": N0}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "N.8a" in capsys.readouterr().err
    monkeypatch.setattr(mod, "head_spec", lambda p: HEAD_A)
    assert mod.main([], require_root=False) == 0
    b = json.loads(Path("results/summary/n_real_odour.json").read_text())["n1"]
    assert b["outcome"] == N1_GO and b["pairs"]["sim"]["o"] == pytest.approx(-4.0)
    assert fm.calls == [("punish_oracle", ["sim", "dis"])]


def test_n1_stop_untestable_exits_5(tmp_path, monkeypatch):
    mod = _script("run_n1")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(o=(-1.0, -2.0, 0.5)), head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0}, smoke_=False)
    assert mod.main([], require_root=False) == 5
    assert json.loads(Path("results/summary/n_real_odour.json").read_text())["n1"]["outcome"] == STOP_UNTESTABLE


def test_pilot_refuses_a_stopped_n1_and_runs_on_arms_only(tmp_path, monkeypatch, capsys):
    mod = _script("run_n2_pilot")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": dict(N1, outcome=STOP_UNTESTABLE)}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "STOP" in capsys.readouterr().err
    _write(tmp_path, {"n0f": dict(N0F, grid={"never": "read"}), "n0": N0, "n1": N1}, smoke_=True)
    assert mod.main(["--smoke"], require_root=False) == 0
    b = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n2_0"]
    assert fm.calls == [("arms", ["dis_on", "sim_on"])]                       # APL on only (decision ⑥)
    assert b["outcome"] == N2_0_GO and b["n"] == 3 and b["c1"] == {"sim": 0.5, "dis": 0.5}
    assert set(b["design_view"]) == {"outcome", "run_id", "selected", "wall_s_per_step", "state_shares", "validity",
                                     "state_flags"}
    assert len(b["pilot"]["sim"]) == len(smoke(SPEC).pilot_seeds)


def test_judge_needs_n8b_citing_the_n2_0_run(tmp_path, monkeypatch, capsys):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None, head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "N.8b" in capsys.readouterr().err
    monkeypatch.setattr(mod, "head_spec", lambda p: HEAD_A + "**N.8b (run `rid-old`)**\n")
    assert mod.main([], require_root=False) == 2 and "does not cite" in capsys.readouterr().err


def test_judge_verdict_then_block_kc_record(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    b = json.loads(Path("results/summary/n_real_odour.json").read_text())["n2"]
    assert b["outcome"] == SUPPORTED and b["thresholds"]["n"] == 3 and b["invalid"] == []
    assert fm.calls[0] == ("arms", ["dis_all", "dis_off", "dis_on", "sim_all", "sim_off", "sim_on"])
    assert fm.calls[1] == ("kc_vectors", ["apl_to_kc_zero"] * 3 + ["apl_all_zero"] * 3)    # only after the verdict
    assert len(fm.calls) == 2
    assert set(b["post_verdict_kc"]) == {"apl_to_kc_zero", "apl_all_zero"}
    assert b["all_output_block"]["D_sim"] == pytest.approx(0.9)
    assert b["per_seed"]["sim_on"][0]["seed"] == SPEC.judge_seed0


def test_judge_plumbing_leak_is_invalid(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(leak=True), head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    b = json.loads(Path("results/summary/n_real_odour.json").read_text())["n2"]
    assert b["outcome"] == INVALID and any("plumbing" in r for r in b["invalid"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_run_n.py -q -o addopts=""`
Expected: FAIL (`FileNotFoundError` for `scripts/run_n1.py`).

- [ ] **Step 3: Write `scripts/run_n1.py`**

```python
#!/usr/bin/env python3
"""Spec N.3 as amended by N.8.5 (N1: the punish-only oracle, APL on only).

Steps:
1. For each pair (similar 4:1 / 1:4, dissimilar 4:1 / δ-DL) at block n0f's operating point, run
   n_jobs.punish_only_oracle_job: α on the select seeds, pre and P on the 16 report seeds.
2. Per pair: p0 = d'(dV_P - dV_pre), testable iff p0 <= -2 and sd > 0, and the raw mean o (N.8.6's c1).
3. N1_GO iff both pairs are testable, else STOP_UNTESTABLE (exit 5), with the note when only the dissimilar pair is
   testable.

    uv run python scripts/run_n1.py                                           # the controller only (R3)
    uv run python scripts/run_n1.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's: blocks n0f and n0 committed, n0f OPERATING_POINT, n0 SIMILARITY_GO, and the
committed **N.8a paragraph citing block n0f's run id (outside --smoke)."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (check_committed, code_keys, git_state, head_spec, load_c3, m0d_sha,  # noqa: F401
                                main_stage, make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_rules import N1_GO, OPERATING_POINT, SIMILARITY_GO, n1_outcome, n1_pair, sentence


def hooks() -> dict:
    return dict(check_committed=check_committed, code_keys=code_keys, git_state=git_state, head_spec=head_spec,
                load_c3=load_c3, m0d_sha=m0d_sha, make_measurer=make_measurer, model_types=model_types)


def body(ctx) -> tuple:
    spec = ctx.spec
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    pairs = [dict(name=n, odor_x=st[x]["odor"], odor_y=st[y]["odor"]) for n, (x, y) in spec.pair_stimuli().items()]
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.punish_oracle(ctx.c3, pairs, ctx.readout, ctx.z, spec.h3.punish_type)
    finally:
        pool.close()
    per = {p["name"]: n1_pair(r, ctx.z, spec) for p, r in zip(pairs, rows)}
    out = n1_outcome(per)
    res = dict(outcome=out["outcome"], note=out["note"], point=dict(g=g, c_delta=c), pairs=per)
    return dict(res, sentence=sentence("n1", res)), (0 if out["outcome"] == N1_GO else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n1", argv, body, hooks(), need=("n0f", "n0"),
                      outcomes={"n0f": (OPERATING_POINT,), "n0": (SIMILARITY_GO,)}, note=("N.8a", "n0f"), spec=spec,
                      require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/run_n2_pilot.py`**

```python
#!/usr/bin/env python3
"""Spec N.8.6, decision ⑥ (N2.0: calibration before any APL-block learning data).

Steps:
1. The absolute-conditioning arm with APL on, for both pairs, on the 16 pilot seeds (n_jobs.absolute_arm_job).
2. Per-seed Δ = dV_post - dV_pre.
3. c1 = 0.25 |o| (block n1); δ_min = 0.5 ℓ̂_sim,on; ε = 0.25 ℓ̂_dis,on.
4. The joint-rule OC (n_oc.simulate) and n.
5. The judgement-block budget from block n0f's wall clock (n_rules.design_view: the only n0f fields read here).
Outcomes: N2_0_GO, STOP_POWER (an uncalibratable pilot or no n) or STOP_BUDGET (> 48 h), both exit 5. The numbers go
into N.8b (scripts/write_n_notes.py), which the controller commits before the judge.

    uv run python scripts/run_n2_pilot.py                                     # the controller only (R4)
    uv run python scripts/run_n2_pilot.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's: blocks n0f, n0 and n1 committed with GO outcomes, and the committed **N.8a
paragraph (outside --smoke)."""
from __future__ import annotations

import sys

import numpy as np

from flymon.brain.n_cli import (check_committed, code_keys, git_state, head_spec, load_c3, m0d_sha,  # noqa: F401
                                main_stage, make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_oc import simulate
from flymon.brain.n_rules import (N1_GO, N2_0_GO, OPERATING_POINT, SIMILARITY_GO, budget_hours, deltas, design_view,
                                  n2_0_outcome, sentence, state, thresholds)


def hooks() -> dict:
    return dict(check_committed=check_committed, code_keys=code_keys, git_state=git_state, head_spec=head_spec,
                load_c3=load_c3, m0d_sha=m0d_sha, make_measurer=make_measurer, model_types=model_types)


def body(ctx) -> tuple:
    spec = ctx.spec
    view = design_view(ctx.doc["n0f"])                  # blinding: nothing else of block n0f is read
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    ps = spec.pair_stimuli()
    items = [dict(cond=f"{p}_on", edit="none", odor_x=st[ps[p][0]]["odor"], odor_y=st[ps[p][1]]["odor"], seed=s,
                  plastic=True) for p in ("sim", "dis") for s in spec.pilot_seeds]
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.arms(ctx.c3, items, ctx.readout, spec.h3.punish_type)
    finally:
        pool.close()
    by = {}
    for r in rows:
        by.setdefault(r["cond"], []).append(r)
    d = deltas(by, ["sim_on", "dis_on"], ctx.z)
    pilot = {"sim": d["sim_on"], "dis": d["dis_on"]}
    th = thresholds({k: ctx.doc["n1"]["pairs"][k]["o"] for k in ("sim", "dis")}, pilot, spec)
    oc = simulate(pilot, th["c1"], th["delta_min"], th["eps"], spec) if th["calibratable"] else None
    n = oc["n"] if oc else None
    budget = budget_hours(n if n else max(spec.n_grid), view["wall_s_per_step"], spec, ctx.c3.dt)
    out = n2_0_outcome(th, oc, budget, spec)
    states = {cnd: float(np.mean([state(r[ph][k]["P"], spec) == "firing" for r in rs for ph in ("pre", "post")
                                  for k in ("x", "y")])) for cnd, rs in by.items()}
    res = dict(outcome=out["outcome"], reason=out["reason"], n=n, c1=th["c1"], delta_min=th["delta_min"],
               eps=th["eps"], ell_hat=th["ell_hat"], pilot_sd=th["pilot_sd"], alt_D_sim=th["alt_D_sim"],
               pilot={k: v.tolist() for k, v in pilot.items()}, pilot_seeds=list(spec.pilot_seeds), oc=oc,
               budget_h=budget, design_view=view, pilot_states=states)
    return dict(res, sentence=sentence("n2_0", res)), (0 if out["outcome"] == N2_0_GO else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n2_0", argv, body, hooks(), need=("n0f", "n0", "n1"),
                      outcomes={"n0f": (OPERATING_POINT,), "n0": (SIMILARITY_GO,), "n1": (N1_GO,)},
                      note=("N.8a", "n0f"), spec=spec, require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `scripts/run_n2_judge.py`**

```python
#!/usr/bin/env python3
"""Spec N.4 / N.8.6 / N.8.7 (N2: the judgement).

Steps:
1. On the n judge seeds of block n2_0, run the four judged conditions (similar / dissimilar x APL on /
   APL->KC block) and the two all-output-block record arms (n_jobs.absolute_arm_job).
2. Rerun each condition's first plumbing_seeds seeds with plasticity off (the plumbing check).
3. Checkpoint: every (condition, seed) is one cache entry under results/n/run/cache/, so rerunning the command resumes.
4. INVALID first: plumbing, block validity, CSC sha256 checks. Then the raw-unit statistics and the verdict with c1,
   δ_min and ε from block n2_0: SUPPORTED / NOT_REPLICATED / NO_LEARNING.
5. Records: d'-based L and I, the all-output-block table (same seeds), the state-conditional ℓ, and the per-seed Δ.
6. Only after the verdict: the block conditions' KC vectors on the activity seeds and their correlations.

    uv run python scripts/run_n2_judge.py                                     # the controller only (R5; resumable)
    uv run python scripts/run_n2_judge.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's: blocks n0f, n0, n1 and n2_0 committed with GO outcomes, and the committed
**N.8b paragraph citing block n2_0's run id (outside --smoke). Exit 0 for every verdict (the verdict goes to the
user)."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (check_committed, code_keys, git_state, head_spec, load_c3, m0d_sha,  # noqa: F401
                                main_stage, make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_rules import (N1_GO, N2_0_GO, N2_ORDER, OPERATING_POINT, SIMILARITY_GO, block_validity,
                                  csc_checks, deltas, dprime_record, n2_stats, plumbing, sentence, similarity,
                                  state_conditional, verdict)


def hooks() -> dict:
    return dict(check_committed=check_committed, code_keys=code_keys, git_state=git_state, head_spec=head_spec,
                load_c3=load_c3, m0d_sha=m0d_sha, make_measurer=make_measurer, model_types=model_types)


def body(ctx) -> tuple:
    spec, b0 = ctx.spec, ctx.doc["n2_0"]
    n = int(b0.get("n") or spec.n_grid[0])              # outside --smoke the outcome gate guarantees block n2_0's n
    seeds = spec.judge_seeds(n)
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    ps = spec.pair_stimuli()
    conds = spec.n2_conditions + spec.n2_record_conditions
    item = lambda name, p, edit, s, plastic: dict(cond=name, edit=edit, odor_x=st[ps[p][0]]["odor"],
                                                  odor_y=st[ps[p][1]]["odor"], seed=s, plastic=plastic)
    items = [item(nm, p, e, s, True) for nm, p, e in conds for s in seeds]
    items += [item(nm, p, e, s, False) for nm, p, e in conds for s in seeds[:spec.plumbing_seeds]]
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.arms(ctx.c3, items, ctx.readout, spec.h3.punish_type)
        by = {}
        for r in rows:
            if r["plastic"]:
                by.setdefault(r["cond"], []).append(r)
        judged = {k: by[k] for k in N2_ORDER}
        invalid = plumbing([r for r in rows if not r["plastic"]]) + block_validity(judged, spec) + csc_checks(judged)
        d = deltas(by, [nm for nm, _, _ in conds], ctx.z)
        stats = n2_stats({k: d[k] for k in N2_ORDER}, spec)
        v = verdict(stats, b0["c1"], b0["delta_min"], b0["eps"], invalid)
        all_block = n2_stats({"sim_on": d["sim_on"], "sim_off": d["sim_all"], "dis_on": d["dis_on"],
                              "dis_off": d["dis_all"]}, spec)
        # N.8.6: the block conditions' KC correlations are computed only now, after the verdict
        edits = ("apl_to_kc_zero", "apl_all_zero")
        kcv = m.kc_vectors(ctx.c3, [(e, st[s]["odor"]) for e in edits for s in spec.judged_stimuli], spec.act_seeds,
                           ctx.readout)
    finally:
        pool.close()
    k = len(spec.judged_stimuli)
    post_kc = {}
    for i, e in enumerate(edits):
        part = dict(zip(spec.judged_stimuli, kcv[i * k:(i + 1) * k]))
        sim = similarity({s: [r["kc_fired"] for r in rr] for s, rr in part.items()}, part[spec.judged_stimuli[0]][0]["n_kc"],
                         spec)
        post_kc[e] = {x: sim[x] for x in ("r_sim", "r_dis", "delta_r", "ci95")}
    res = dict(outcome=v["verdict"], verdict=v, stats=stats, invalid=invalid,
               thresholds=dict(n=n, c1=b0["c1"], delta_min=b0["delta_min"], eps=b0["eps"]),
               per_seed={cnd: [dict(seed=int(s), delta=float(x)) for s, x in zip(seeds, d[cnd])] for cnd in d},
               dprime_record=dprime_record({kk: d[kk] for kk in N2_ORDER}), all_output_block=all_block,
               state_conditional=state_conditional(by, ctx.z, spec),
               csc_sha256={cnd: sorted({r["csc_sha256"] for r in rs}) for cnd, rs in by.items()},
               post_verdict_kc=post_kc, n0_r_sim=(ctx.doc.get("n0") or {}).get("similarity", {}).get("r_sim"))
    return dict(res, sentence=sentence("n2", res)), 0


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n2", argv, body, hooks(), need=("n0f", "n0", "n1", "n2_0"),
                      outcomes={"n0f": (OPERATING_POINT,), "n0": (SIMILARITY_GO,), "n1": (N1_GO,),
                                "n2_0": (N2_0_GO,)},
                      note=("N.8b", "n2_0"), spec=spec, require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_n.py -q -o addopts=""`
Expected: PASS (14 + 9 tests).

- [ ] **Step 7: Full suite, then commit**

```bash
uv run pytest -q -rfE -o addopts=""
git add scripts/run_n1.py scripts/run_n2_pilot.py scripts/run_n2_judge.py tests/test_run_n.py
git commit -m "feat(n): N1 punish-only oracle, N2.0 pilot calibration and N2 judge CLIs (N.8a/N.8b gates, checkpointed arms, post-verdict KC record)"
```

---

### Task 11: `scripts/write_n_notes.py` — the N.8a / N.8b paragraphs

**Files:**
- Create: `scripts/write_n_notes.py`
- Test: `tests/test_write_n_notes.py`

**Interfaces:**
- Consumes: block `n0f` (Task 9) and blocks `n1` / `n2_0` (Task 10); `n_cli.spec_note` (round-trip test).
- Produces: `n8a(n0f: dict, date: str) -> str`, `n8b(n2_0: dict, n1: dict, date: str) -> str`, `main(argv=None) -> int` (`--summary`, `--which n8a|n8b`, `--date`), which prints to stdout. The controller pastes the text under N.8 in the spec. The text begins with `**N.8a` / `**N.8b` and cites the block's run id, so `n_cli.spec_note` accepts it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_write_n_notes.py
"""Plan reading 12: the dated N.8a / N.8b paragraphs carry every number N.8.3 / N.8.6 ask the controller to commit and
pass n_cli.spec_note for the block they cite."""
import importlib.util
import json
import sys
from pathlib import Path

from flymon.brain.n_cli import spec_note

ROOT = Path(__file__).resolve().parents[1]
CELL = dict(kc_frac_median=0.05, runaway_share=0.0, A_zero_share=0.1, P_zero_share=0.0, firing_share=0.9,
            apl_out_mean=0.12)
N0F = dict(outcome="OPERATING_POINT", run_id="20261001T000000Z-abc123", selected={"g": 0.5, "c_delta": 4.0},
           checks={"0.5|4": dict(mean_on=0.052, dist=0.0034, ok=True)},
           grid={"0.5|4": {s: {c: CELL for c in ("on", "block", "all")} for s in ("4:1", "1:4", "dDL", "IA", "EB")}},
           state_flags=[dict(stimulus="dDL", condition="block", on=0.9, share=0.5)],
           apl_shift={"4:1": {"block": 0.01, "all": -0.1}},
           lin_totals=dict(totals={"signed": {"IA": 2040.0, "EB": 1870.0, "dDL": 296.0},
                                   "positive": {"IA": 2058.0, "EB": 1870.0, "dDL": 296.0}},
                           rel_error={"signed": {"IA": 0.0049, "EB": 0.0054, "dDL": 0.035},
                                      "positive": {"IA": 0.0138, "EB": 0.0054, "dDL": 0.035}}),
           drives={"4:1": dict(clipped_total_hz=4.6, capped=[])}, wall_s_per_step=0.0021,
           budget_estimate_h={"16": 20.1, "64": 80.4})
N1 = dict(pairs={"sim": {"o": -1.6, "p0": -3.1}, "dis": {"o": -2.4, "p0": -4.0}})
N2_0 = dict(outcome="N2_0_GO", run_id="20261002T000000Z-def456", n=24, c1={"sim": 0.4, "dis": 0.6}, delta_min=0.3,
            eps=0.2, ell_hat={"sim": 0.6, "dis": 0.8}, pilot_sd={"sim": 0.5, "dis": 0.4}, alt_D_sim=0.45,
            budget_h=31.5, oc=dict(draws=2000, boot=1000, rows=[dict(sd_mult=1.0, hyp="null", n=24, p_supported=0.01,
                                                                    mc_se=0.002)]))


def _script():
    sp = importlib.util.spec_from_file_location("write_n_notes", ROOT / "scripts" / "write_n_notes.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules["write_n_notes"] = mod
    sp.loader.exec_module(mod)
    return mod


def test_n8a_carries_the_point_states_lin_and_budget_and_passes_the_gate():
    t = _script().n8a(N0F, "2026-10-01")
    assert t.startswith("**N.8a") and N0F["run_id"] in t and "g = 0.5, c_δ = 4" in t
    for s in ("2040", "+3.5%", "0.9 → 0.5", "80.4 h", "2.1 ms"):
        assert s in t
    assert spec_note("spec\n" + t, "N.8a", N0F["run_id"]) is None


def test_n8a_of_a_stop_says_so():
    t = _script().n8a(dict(N0F, outcome="STOP_NO_OPERATING_POINT", selected=None), "2026-10-01")
    assert "STOP_NO_OPERATING_POINT" in t and "작동점 없음" in t


def test_n8b_carries_the_thresholds_and_the_oc_table_and_passes_the_gate():
    t = _script().n8b(N2_0, N1, "2026-10-02")
    assert t.startswith("**N.8b") and N2_0["run_id"] in t
    for s in ("n = 24", "c₁ = sim 0.4, dis 0.6", "δ_min = 0.3", "ε = 0.2", "o: sim -1.6, dis -2.4", "31.5 h",
              "| 24 | 1 | null | 0.010 |"):
        assert s in t
    assert spec_note(t, "N.8b", N2_0["run_id"]) is None


def test_main_reads_the_summary(tmp_path, capsys):
    s = tmp_path / "n.json"
    s.write_text(json.dumps({"n0f": N0F, "n1": N1, "n2_0": N2_0}))
    m = _script()
    assert m.main(["--summary", str(s), "--which", "n8b", "--date", "2026-10-02"]) == 0
    assert capsys.readouterr().out.startswith("**N.8b")
    assert m.main(["--summary", str(tmp_path / "none.json"), "--which", "n8a"]) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_write_n_notes.py -q -o addopts=""`
Expected: FAIL (`FileNotFoundError` for `scripts/write_n_notes.py`).

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""Print the dated N.8a paragraph (after N0f, N.8.3) or N.8b paragraph (after N2.0, N.8.6) from
results/summary/n_real_odour.json. The controller pastes it at the end of spec N.8 and commits the spec. N0 / N1 /
N2.0 (N.8a) and the N2 judge (N.8b) refuse until the committed spec has the paragraph citing the block's run id
(n_cli.spec_note).

    uv run python scripts/write_n_notes.py --which n8a [--date 2026-10-01]
    uv run python scripts/write_n_notes.py --which n8b"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

CONDS = (("on", "켬"), ("block", "APL→KC 차단"), ("all", "전 출력 차단"))


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def n8a(b: dict, date: str) -> str:
    sel = b.get("selected")
    head = f"**N.8a N0f 결과 ({date}, run `{b['run_id']}`, 기록 전용)** — outcome `{b['outcome']}`."
    lines = [head]
    if sel:
        key = f"{sel['g']:g}|{sel['c_delta']:g}"
        chk = b["checks"][key]
        lines.append(f"- 작동점: g = {sel['g']:g}, c_δ = {sel['c_delta']:g} (켬 세 판정 자극 KC 활성 중앙값 평균 "
                     f"{100 * chk['mean_on']:.2f}%, 목표 5.54%).")
        for s, by in b["grid"][key].items():
            parts = [f"{lab} KC {100 * by[c]['kc_frac_median']:.1f}% · >150 Hz {by[c]['runaway_share']:.3f} · "
                     f"A0 {by[c]['A_zero_share']:.2f} · P0 {by[c]['P_zero_share']:.2f} · 방출 상태 "
                     f"{by[c]['firing_share']:.2f} · APL {by[c]['apl_out_mean']:.4f}" for c, lab in CONDS]
            lines.append(f"- {s}: " + " / ".join(parts))
        flags = b.get("state_flags") or []
        lines.append("- 상태 몫 플래그(> 0.25): " + (", ".join(f"{f['stimulus']} {f['condition']} {f['on']:g} → "
                                                        f"{f['share']:g}" for f in flags) or "없음"))
        lines.append("- APL 출력 이동(차단 − 켬): " + ", ".join(f"{s} {v}" for s, v in (b.get("apl_shift") or {}).items()))
        lines.append("- 잘린 억제량 / 불응기 상한 초과 채널: " + ", ".join(
            f"{s} {d['clipped_total_hz']:.1f} Hz / {d['capped'] or '없음'}" for s, d in (b.get("drives") or {}).items()))
    else:
        lines.append("- 작동점 없음: 세 조건을 모두 만족하는 (g, c_δ)가 없다 (STOP_NO_OPERATING_POINT). 격자 전체는 "
                     "블록 n0f의 grid·checks에 있다.")
    lt = b["lin_totals"]
    lines.append("- Lin 2014 총합 대조(기록): " + "; ".join(
        f"{conv} " + ", ".join(f"{k} {lt['totals'][conv][k]:g} ({_pct(lt['rel_error'][conv][k])})"
                               for k in ("IA", "EB", "dDL")) for conv in ("signed", "positive")))
    lines.append(f"- 벽시계: step당 {1000 * b['wall_s_per_step']:.1f} ms; 판정 블록 예상 "
                 + ", ".join(f"n={n} {h:.1f} h" for n, h in b["budget_estimate_h"].items()) + ".")
    return "\n".join(lines) + "\n"


def n8b(b: dict, n1: dict, date: str) -> str:
    o = {k: n1["pairs"][k]["o"] for k in ("sim", "dis")}
    lines = [f"**N.8b N2.0 보정 결과 ({date}, run `{b['run_id']}`)** — outcome `{b['outcome']}`"
             + (f": {b['reason']}" if b.get("reason") else "."),
             f"- 파일럿(켬, {len(b.get('pilot_seeds') or []) or 16} 시드): ℓ̂_sim,on = {b['ell_hat']['sim']:.4g}, "
             f"ℓ̂_dis,on = {b['ell_hat']['dis']:.4g} (sd sim {b['pilot_sd']['sim']:.4g}, dis {b['pilot_sd']['dis']:.4g}).",
             f"- N1 원단위 효과 o: sim {o['sim']:g}, dis {o['dis']:g} → c₁ = sim {b['c1']['sim']:g}, dis {b['c1']['dis']:g}; "
             f"δ_min = {b['delta_min']:.4g}; ε = {b['eps']:.4g}; 대립 D_sim = {b['alt_D_sim']:.4g}.",
             f"- n = {b['n']}; 판정 블록 예상 {b['budget_h']:.1f} h (한도 48 h)."]
    if b.get("oc"):
        lines += [f"- 운영 특성 (실험 {b['oc']['draws']}회 × 부트스트랩 {b['oc']['boot']}회):", "",
                  "| n | 편차 배수 | 가설 | P(SUPPORTED) | MC SE |", "|---|---|---|---|---|"]
        lines += [f"| {r['n']} | {r['sd_mult']:g} | {r['hyp']} | {r['p_supported']:.3f} | {r['mc_se']:.3f} |"
                  for r in b["oc"]["rows"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results/summary/n_real_odour.json")
    ap.add_argument("--which", choices=("n8a", "n8b"), required=True)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    a = ap.parse_args(argv)
    try:
        doc = json.loads(Path(a.summary).read_text())
        text = n8a(doc["n0f"], a.date) if a.which == "n8a" else n8b(doc["n2_0"], doc["n1"], a.date)
    except (OSError, ValueError, KeyError) as e:
        print(f"refused: no usable block for {a.which} in {a.summary}: {e!r}", file=sys.stderr)
        return 2
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_write_n_notes.py -q -o addopts=""`
Expected: PASS (4 tests).

- [ ] **Step 5: Check the manifest is complete and run the full suite**

Run: `uv run python -c "from flymon.brain.n_cli import code_keys; k, m = code_keys('data/malecns.npz'); print(k['key'][:12], m['key'][:12])"`
Expected: two key prefixes printed and no refusal. Every `HASHED_FILES` entry now exists.

Run: `uv run pytest -q -rfE -o addopts=""`
Expected: baseline + every new test, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add scripts/write_n_notes.py tests/test_write_n_notes.py
git commit -m "feat(n): write_n_notes prints the dated N.8a / N.8b paragraphs the stage gates require"
```

---

## Self-review (done while writing)

1. **Spec coverage:**

   | Spec item | Where |
   |---|---|
   | N.1 seed blocks (activity raised to 16 by N.8.4), training-seed rule, integer width | Task 2 (tests), reading 1 |
   | N.1 / N.8.1 claim wording, all-output arm shown next to the verdict | Tasks 8 (`sentence`), 10 (`all_output_block`), 11 |
   | N.2 / N.8.2 data pinning, license, CAS matching, duplication, clip record, linear mixtures, c_δ, s conversion, hand fixture, Lin totals as record | Tasks 1, 3 |
   | N.8.3 grid, measurements (KC, sub-window, A / P, zero shares, APL output, state, wall clock), blinding, operating-point rule, STOP, flags | Tasks 4, 7, 9 |
   | N.8.4 similarity gate with bootstrap, r record | Tasks 7, 9 |
   | N.3 / N.8.5 punish-only oracle, α rule, p0, sd > 0, o, jaccard, floor, STOP_UNTESTABLE with note | Tasks 5, 7, 10 |
   | N.4 / N.8.7 absolute conditioning without CS−, whole-arm edit, own training loop, no `train_block` change, plumbing check, (condition, seed) checkpoint, CSC sha per arm, mixed-worker regression | Tasks 4, 5, 6, 10 |
   | N.8.6 raw-unit Δ / ℓ / D, common-seed bootstrap, c₁ / δ_min / ε, clauses ①–④, INVALID (plumbing, block validity), d′ record, NO_LEARNING wording | Tasks 8, 10 |
   | N.8.6 N2.0 pilot (on only), OC over ①–④ with 1× / 2×, n grid, STOP_POWER, STOP_BUDGET 48 h, N.8b before the judge | Tasks 8, 10, 11 |
   | N.8.8 state of every presentation, unconditional primary, state-conditional record | Tasks 7, 8, 10 |
   | N.8.9 stop order and the N.8a / N.8b commits | Task 9 (`main_stage` gates), Task 11, Runs |
   | N.6 new files only, summaries under `results/summary/n_*.json`, raw data under `results/n/` | Global Constraints, Task 2 |

2. **Placeholder scan:** every step has its code or command. Task 5 step 4 names the one tuning an implementer may need (a synthetic pulse too short to move a weight) and forbids weakening the assert.
3. **Type consistency:**
   - `presentation_job` / `kc_vectors_job` rows share `PRESENTATION_KEYS`, and `cell_stats` reads exactly those keys.
   - `absolute_arm_job` rows are the `pre` / `post` × `x` / `y` shape that `delta`, `plumbing`, `block_validity` and `state_conditional` read.
   - `NMeasurer.arms` adds `cond`.
   - The block names follow `n_cli.ORDER`, and `design_view`'s keys are what `run_n0f` writes at a selected point.
   - Block `n2_0` carries `c1`, `delta_min`, `eps` and `n`, which is what the judge reads.
4. **Review Focus:**
   - Item 1 → Task 4 `test_on_block_on_in_one_worker`, Task 5 `test_arms_on_block_on_in_one_worker`, Task 6 `test_one_entry_per_condition_and_seed_and_a_rerun_resumes`.
   - Item 2 → Task 1 `test_values_are_matched_by_cas_and_delta_is_not_gamma` and the pinned-bytes tests; Task 3's data-defect tests.
   - Item 3 → Task 2 `test_big_seeds_reach_the_engine_generator_unchanged`.
   - Item 4 → Task 7 `test_n1_testable_needs_p0_at_most_minus_2_and_a_spread` and `test_similarity_with_silent_kcs_stops_without_crashing`; Task 8 `test_thresholds_and_an_uncalibratable_pilot` and `test_n2_0_outcome_order`.
   - Item 5 → Task 4 `test_a_presentation_row_is_scalars_only`, Task 8 `test_design_view_is_the_whitelist`, Task 10 `test_judge_verdict_then_block_kc_record`.

5. **Validated in a scratch worktree (2026-09-30, detached at `e37a888`):**
   - Every code block of this plan was spliced in and run. Task 1's fetch reproduced the pinned bytes, and all N tests pass (Tasks 1–11), also in full-suite order. The fixture-sharing fix is the `h4_jobs._RIG.clear()` in Tasks 4–5.
   - `code_keys` found every hashed file.
   - A real `run_n0f.py --smoke --workers 6` ran end to end in 24 s: 30 presentations at 1.7 ms per step, exit 5 at the smoke's single point g = 1. On-condition KC median was 30–33% for IA / EB / 4:1 / 1:4 but 2.6% for δ-DL (the imbalance c_δ exists for), and no cache row carried `kc_fired`.
   - This is a smoke, not a result.

---

## Runs (controller)

All runs happen from the repository root, in the background (`run_in_background`), after the full suite passes at the final commit. Subagents never run these. N.6 governs timing:
- Real measurement starts only after the re-scope track's S4 pilot has finished.
- **Report each stage to the user before starting the next.**
- Every `STOP_*` (exit 5) goes to the user with no post-hoc change.

- **R0 smoke** (seed block 21_009_xxx, 1 × 1 grid; writes only `results/n/smoke/`):
  1. `uv run python scripts/run_n0f.py --smoke --allow-dirty --workers 4`
  2. `uv run python scripts/run_n0.py --smoke --allow-dirty --workers 4`
  3. `uv run python scripts/run_n1.py --smoke --allow-dirty --workers 4`
  4. `uv run python scripts/run_n2_pilot.py --smoke --allow-dirty --workers 4`
  5. `uv run python scripts/run_n2_judge.py --smoke --allow-dirty --workers 4`
  6. Smoke bypasses outcome and note gates (recorded in `smoke_bypass`), so the chain runs through whatever the smoke outcomes are.
  7. Check each smoke block for:
     - n0f rows with no `kc_fired`;
     - `csc_sha256` differing between on and block;
     - `invalid == []` in the judge's plumbing;
     - the judge's `per_seed` for 3 seeds.
  8. Record the wall clock per presentation and per arm. Rerun the judge smoke to confirm the resume: all cache hits, no job.
- **R1 N0f:**
  1. `uv run python scripts/run_n0f.py` runs 144 items × 8 seeds = 1 152 presentations.
  2. Commit `results/summary/n_real_odour.json` (block `n0f`).
  3. On `STOP_DATA_MISMATCH` or `STOP_NO_OPERATING_POINT`: write the result into the spec and ask the user.
  4. Otherwise run `uv run python scripts/write_n_notes.py --which n8a`, paste the paragraph at the end of spec N.8, commit the spec, and report to the user. This is N.8a, the gate for R2–R4.
- **R2 N0:** `uv run python scripts/run_n0.py`. Commit block `n0`. On `STOP_SIMILARITY_ORDER`, stop and ask the user.
- **R3 N1:** `uv run python scripts/run_n1.py`. Commit block `n1`. On `STOP_UNTESTABLE`, stop and ask the user; the note says whether only the dissimilar pair was testable.
- **R4 N2.0:**
  1. `uv run python scripts/run_n2_pilot.py` runs 2 pairs × 16 pilot seeds, then the OC (minutes).
  2. Commit block `n2_0`.
  3. On `STOP_POWER` or `STOP_BUDGET`, stop and ask the user.
  4. Otherwise run `uv run python scripts/write_n_notes.py --which n8b`, paste the paragraph as N.8b, commit the spec, and report to the user. This is the gate for R5.
- **R5 N2 judge:**
  1. `uv run python scripts/run_n2_judge.py` runs 6 conditions × n seeds, plus 6 × 2 plumbing reruns, then the post-verdict KC record.
  2. It takes hours to ~48 h at the budget; rerun the same command to resume after an interruption.
  3. Commit block `n2`.
  4. Write the result section (verdict sentence, the all-output-block table next to it, state-conditional record, post-verdict block KC correlations) and the README ledger (ko/en).
  5. Every verdict goes to the user.
