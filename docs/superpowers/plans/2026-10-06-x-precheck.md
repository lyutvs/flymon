# Spec X, phase A: point-θ precheck and record-only diagnostics (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run only X's **phase A** (X.9.1.1 구현 단계화, X.9.1.5 order **0** and **0a**). Precedence is **X.9.1 > X.0–X.9**. For the W rules X keeps, it is W.9.10 > W.9.9 > W.9.8 > W.9.1–W.9.7 > W.0–W.8. Phase A covers the following:
- **Order 0 (`stage0`)**: X's set verdict code (`x_verdict`, X.3) and the X OC pieces phase A needs (`x_oc`). This includes calibration with the per-status handling of X.9.1.2 (brackets ×4, a 160-step retry on `no_convergence`, power `floor` → 0, false-pass `floor` → 1). It also includes X.4.6's synthetic validation at every p_set, and **P2-6**: X's evaluate at p_set 1.0 and b = 0 must equal `w_oc.evaluate` bit for bit on W's pilot θ̂, and the unit tests repeat this on a synthetic θ. Stage 0 also writes the numeric tables and the OC compute time. Last comes the **W calibration-failure diagnosis** (X.4.3 + X.9.1.2): W's 200 bootstrap calibrations are re-drawn from W's root 42_000_000 and compared draw by draw with `results/w/oc.json`. Every status is recorded, and failed draws are recalibrated with brackets ×4 and with 160 steps. The diagnosis runs after the reuse checks the precheck depends on. Those checks cover the W measurement key `761274e0…`, W's pipeline key, `w_measure.py`'s hash, the W pilot block `5fbc4c8` and its 384-file manifest, `results/w/oc.json` `7a0a65f3…`, and the pair order = W `_pilot_back`, with θ̂'s summary compared with W's `oc.theta` first. Any break gives `STOP_REUSE`.
- **Order 0a (`precheck`)**: the **point-θ precheck gate** (X.9.1.1). It uses the V0 θ̂ and every design (p_set, q, K, F 8–32) for k = 4–8. It evaluates g ∈ {0, 0.5, 1.0} (worst g), true d′ 1.5 / 0.5, and 4000 experiments per g × target on root 43_200_000. The outcome is PASS, or `STOP_OC_UNREACHABLE` with the spec's sentence, and on STOP the point records at θ̂ are computed (X.4.5 → X.9.1.1, variants V0 · V1 · V2). Order 0a also computes the **record-only diagnostics (i)–(iii)** (X.9.1.4, root 43_300_000, `results/x/precheck_diag.json`), the **P1-5 residual comparison** and the **P2-11 environment hashes**. All of these go into one `precheck` block.

Nothing from phase B is built here: no bootstrap OC with variants, smoke, budget, set, oracle, naive, learn, band, records, seal or judge. Phase B is planned and built **only after a precheck PASS**, and it appends to the phase-A files without changing what they compute (X.9.1.1: if a phase-A file must change, the precheck is recomputed with the same seeds and must be bit-identical).

**Architecture:**
- New files: `flymon/brain/x_spec.py`, `x_verdict.py`, `x_oc.py`, `x_rules.py`, `x_store.py`, `x_runner.py`, `scripts/run_x.py`, `tests/brain/test_x_*.py`. The one existing file touched is `tests/brain/test_p_spec.py`, which gains one `MODULES` entry (the seed collectors must see X's blocks).
- **W code is imported and never modified**: `w_oc`, `w_verdict`, `w_spec`, `w_runner`, `w_store`, `w_records`, `w_pairs` and `w_measure` (`w_measure.py` must never change). `x_oc` reuses `w_oc.fit`, `boot`, `summary`, `simulate`, `slot_means`, `to_stages`, `true_dprimes`, `_bisect`, `_rng`, `_uniform`, `_stack`, `sd_pre`, `synthetic_pilot` and `simple_normal`, together with `w_verdict`'s pair-level functions. The new code covers only these:
  - The set rule.
  - The p_set axis.
  - The per-status calibration.
  - The simulator variants: rejection rounds and filters (diagnostics only; precheck and P2-6 use `w_oc.simulate` itself).
  - The precheck, the records at θ̂, the variants, the diagnostics and the W diagnosis.
- `XSpec` restates the W numbers that W's functions read, with the same field names and values, pinned by a test. An `XSpec` can therefore be passed wherever `w_oc` / `w_verdict` read a spec, while X's own roots and grids live only in `x_spec`.
- Module boundaries for phase B:
  - `x_spec` gains fields.
  - `x_verdict` gains the real-data `judge`.
  - `x_oc` gains the bootstrap, qualification, selection and full records.
  - `x_rules` gains the remaining gates and sentences.
  - `x_store` gains `XCache`.
  - `x_runner.ORDER` gains its stages after `precheck`.
  - `run_x.STAGES` grows to match.

**Tech Stack:** Python 3.13, NumPy, pytest. Use `.venv/bin/python` everywhere, because bare `python` is not on PATH.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **X**: X.0–X.9 and **X.9.1** (X.9.1.1–X.9.1.6), with X.9.1 taking precedence. X reuses appendix **W** (W.0–W.10, the W.9.x amendments) and its code. User decisions are in `/private/tmp/claude-503/x-decisions.md`.

## Global Constraints

- Precedence: X.9.1 > X.0–X.9 > (W rules X keeps: W.9.10 > W.9.9 > W.9.8 > W.9.1–W.9.7 > W.0–W.8).
- Phase A = X.9.1.1's list only: `x_spec` (numbers), `x_verdict` (set rule), `x_oc` (status calibration, point evaluation, precheck, record-only diagnostics, W calibration diagnosis), `x_store`, `x_runner` · `scripts/run_x.py` orders 0 · 0a, and their tests (X.4.6 set-rule fixtures, synthetic validation, the phase-A mutations, P2-6). `x_rules` is added for the gate sentences (Reading 1).
- `flymon/brain/w_*.py` and `scripts/run_w.py` are never edited; `w_measure.py` above all (its sha256 must stay `e23009146ee04e3f6f328445a51685f8f045f6bb194749bf4ebd8f80c6e2be63`). `results/w/` and `results/summary/w_learning.json` are read only.
- X writes only under `results/x/` (git-ignored) and `results/summary/x_learning.json` (tracked).
- Numbers live only in `x_spec.py`. Any other X file holding an int literal with |v| > 16 or a float literal other than 0.0 / 1.0 fails `test_x_spec`.
- Set rule (X.3): n = final PASS + FAIL, m = PASS, b = BAND left. PASS: n ≥ 4 ∧ m ≥ 3 ∧ m/n ≥ p_set ∧ m/(n+b) ≥ p_set. FAIL: n ≥ 4 ∧ ((m+b)/(n+b) < p_set ∨ m+b < 3). Otherwise UNDECIDED ("판정 가능 쌍 < 4" if n < 4 else "BAND 잔존"). Comparisons use `round(x − p_set, 9)`. STOP_MACHINE (machine reason or INVALID pair) comes first.
- p_set ∈ {0.5, 0.625, 0.75, 0.875, 1.0}; q ∈ {0.5, 0.625, 0.75}; K ∈ {8, 16}; F 8–32; k 4–8; g ∈ {0, 0.5, 1.0}.
- G.6 targets: power ≥ 0.80 at true d′ 1.5 (min gate), false pass ≤ 0.05 at 0.5 (max gate, null = zero DAN injection, `cal_floor_rule` "zero").
- Calibration (X.9.1.2): residual draws 2000, tolerance ±0.02, 40 steps, brackets = W's hi_a / hi_b formula **× 4**. On `no_convergence`, retry with **160** steps (same draws, tolerance and brackets). Still failed, or `floor`, or power-side `above_at_zero` → fill power 0 / false pass 1. False-side `above_at_zero` → `zero_floor` (not a failure). The W reproduction alone uses W's brackets (× 1).
- Seeds: X OC root **43_000_000** (synthetic validation, P2-6), precheck root **43_200_000**, diagnostics root **43_300_000**. All streams are `SeedSequence([root, tag, …])`. The W diagnosis uses W's root 42_000_000 read from `w_spec`. No other block exists in phase A.
- Precheck pass: some design has, at **every** k = 4–8, point power (g min) ≥ 0.80 and point false pass (g max) ≤ 0.05.
- Precheck STOP sentence (verbatim, X.9.1.1): "점 θ 사전 점검(X.9.1.1)에서 파일럿 잡음의 점 추정으로도 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다 — p_set · q · K · F 어느 설계도 k = 4–8 모두에서 점 검정력 ≥ 0.80과 점 거짓 통과 ≤ 0.05를 함께 만족하지 않는다(〈…〉). 부트스트랩 작동 특성은 계산하지 않았고, 주 세트를 쓰지 않고 멈춘다." — the parenthesis is the F = 32 table + the best-power design, or "보정 불가 — 〈쪽·손잡이별 상태〉" when a calibration failed.
- STOP_REUSE sentence (X.6): "X 재사용 조건(X.5 1 · X.4.3 진단)이 깨졌다(〈깨진 항목〉). X는 W 파일럿·경로와 V 블록을 다시 재는 경로를 갖지 않으므로 주 세트를 쓰지 않고 멈춘다 — 사용자 몫." It carries `records_unavailable: true` and a reason (P2-10).
- Results go to `results/summary/x_learning.json` (blocks `stage0`, `precheck`, `ledger`), with details in `results/x/w_cal_diag.json`, `results/x/precheck.json`, `results/x/precheck_diag.json` and resumable cells in `results/x/progress/precheck/`.
- Commits carry **no** `Co-Authored-By`, `Claude-Session` or "Generated with" trailers.
- Tests run as `.venv/bin/python -m pytest … > /private/tmp/claude-503/x-tests/<name>.log 2>&1`, then read the tail of the log.

## Review Focus

1. **W raw data or W blocks changed under X.** This covers a manifest sha256 mismatch, a different W measurement key, a reordered `_pilot_back` or a θ̂ summary ≠ W `oc.theta`. Expected: a recorded `STOP_REUSE` block with the broken items, `records_unavailable`, and no precheck computed, not a crash or a silent fit. Owner: Task 5 (`test_stage0_stop_reuse_*`).
2. **θ̂ calibration fails at the precheck**, for example the power side floors. Expected: the power arrays are filled with 0, no design passes, and the sentence's parenthesis reads "보정 불가 — min: a …, b …; max: …". Owner: Task 3 (`test_precheck_calibration_failure_fills_and_sentence`).
3. **The 1–1.5 h precheck run is killed midway.** Expected: rerunning the same command reuses every finished cell under `results/x/progress/precheck/` (same seeds → same values), and the block is identical apart from timing. Owner: Task 5 (`test_precheck_resumes_from_cells`).
4. **A filter that almost never accepts** a naive-balanced pair (F2 at a high level). Expected: rejection stops after 2000 rounds, unfilled slots get the population pair, and the fill share is recorded and flagged above 1 %, with no hang. Owner: Task 4 (`test_pair_bases_reject_all_fills`).
5. **A stage run out of order or twice** (precheck before stage0, stage0 again, precheck after a stage0 STOP, a dirty hashed W file). Expected: refusal with exit 2 and nothing written. Owner: Task 5 (`test_refusals`).

## Readings (spec ambiguities resolved here; each is disclosed in X.10 if X closes)

1. **`x_rules` in phase A.** X.9.1.1's phase-A list omits `x_rules`, but the precheck gate needs its decision and sentence, and Q9 / X.2 name `x_rules` as X's module. Following W's split (`w_rules` holds gate decisions and sentences), phase A creates `x_rules` with only the precheck, reuse and diagnosis decisions.
2. **The reuse checks in phase A** are the subset of X.5 1 the precheck depends on (the user's list plus the facts they rest on). They cover the W measurement key, the W pipeline key `9636cf4c…` (`w_*` unchanged), `w_measure.py` = W stage0's hash, and `w_learning.json` tracked, clean and last changed by `f30ae35`. They also cover W blocks `pilot` (PASS, `5fbc4c8`, 384 units, 16 pairs) and `oc` (`f30ae35`) in HEAD's history on the W key, `oc.json` sha256 `7a0a65f3…`, the manifest sha256 values (`w_store.load_manifest`), the `_pilot_back` order = block pilot's `pairs`, θ̂ summary = W `oc.theta` (JSON round-trip equality, exact), the balanced 3 pairs (X.9.1.3 P1-5) and the V1 count 7. Both `stage0` and `precheck` run them, and a break gives `STOP_REUSE`. The V-block and path checks of X.5 1 are phase B.
3. **The W diagnosis columns**, each record-only:
   - (가) W's brackets and 40 steps, compared with `oc.json` (`ok`, a, b per draw, plus the counts 29 = 14 + 18 − 3).
   - (나) the failed draws with brackets ×4 and 40 steps.
   - (다) the `no_convergence` draws with W's brackets and 160 steps.
   - (라) X's full rule (`calibrate_x`), as what X would do.
4. **P2-6 on θ̂** runs in `stage0`: every g, a and b from θ̂'s X power calibration, `p26_reps` = 200 experiments, root 43_000_000 tag 10. On a synthetic θ it is a unit test. A mismatch makes `stage0` INVALID, because it is a code defect.
5. **Precheck simulation = `w_oc.simulate`** (W's 200 rejection rounds, W.9.9 P2-8's method), so that P2-6 holds for the precheck path. The diagnostics use X's copy (`x_oc.simulate`) with **2000** rounds for every filter, the no-filter one included, so that the six filters are comparable (X.9.1.4 (ii) "거절 표집은 쌍마다 최대 2000회").
6. **Records at θ̂** (precheck STOP only) are evaluated on a one-design spec (only that design's p_set, q, K, F). This is the same model with fewer simulated flies. Streams are `SeedSequence([43_200_000, 5, variant, g, scenario])`. The heterogeneity scenario Σ_v × {0.5, 1, 2} is computed at both 1.5 (power calibration) and 0.5 (false-pass calibration), as X.4.5 reads. On a precheck PASS, no records are computed in phase A (X.4.5 records are phase B's OC).
7. **Diagnostics streams.** Calibration draws use tag 1 on root 43_300_000 (diagnostics have their own calibration; the gate reads only the precheck's). Grid cells use tag 6 (filter, g, target), (i) uses tag 7 (g), and the acceptance draw tag 8.
   - (i) is computed and stored for **every** g with the worst g marked per cell, a superset of "g 0 and g worst".
   - F1's "pass counts" are 3 of 16 and 3 of 3 (F1 is the balanced-pair θ). Its acceptance rate is null because it adds no predicate.
   - The acceptance rate of F2(ℓ) is measured on a separate θ̂ g 0 balanced draw (diag_reps × 16 bases).
   - The diagnostics detail keeps the g-worst and g 0 arrays per filter, not every g.
8. **"OC compute time" (X.5 0)** is one X evaluate at `boot_reps` experiments on θ̂, scaled to the precheck and to phase B's bootstrap.
9. **The STOP sentence's parenthesis** is replaced as a whole by "보정 불가 — min: a 〈status〉, b 〈status〉; max: a …, b …" when either θ̂ calibration fails (X.9.1.1, the W code's form).
10. **Stage 0 INVALID** (synthetic validation or P2-6 failing) is a code defect found before any X data. The controller deletes the uncommitted `x_learning.json`, fixes the code through the task loop and reruns. The block is never committed. A `STOP_REUSE` block is committed.
11. **`XSpec` restates W's shared numbers.** It does not import them. "Numbers only in x_spec" and the duck-typed calls need the fields, and a test pins every one to `w_spec.SPEC`.
12. **The duplicate grid cells (X.9.1.3 P1-3).** The spec lists k 4's duplicates as "0.5 · 0.625 · 0.75". By the spec's own m table, (4, 4) for p_set 0.875 and 1.0 at k 4 is also a duplicate. The code marks every equal cell, so k 4 shows two groups. This is a record mark only; selection is unchanged.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `flymon/brain/x_spec.py` | every X number (phase A), W's shared numbers restated, W facts (keys, hashes, commits), paths | 1 |
| `flymon/brain/x_verdict.py` | the set rule (counts, `set_code_counts`, `set_code`, cause, m-needed table) | 1 |
| `flymon/brain/x_oc.py` | calibration per status, X evaluate (p_set axis), simulator copy, synthetic validation, P2-6, OC timing, W diagnosis (Task 2); variants, precheck, pick, records, residual comparison (Task 3); diagnostics (i)–(iii) (Task 4) | 2–4 |
| `flymon/brain/x_rules.py` | reuse / diagnosis reasons, STOP_REUSE, precheck decision and sentences | 3 |
| `flymon/brain/x_store.py` | the writer (results/x/, x_learning.json) | 5 |
| `flymon/brain/x_runner.py` | context, chain, `stage_stage0`, `stage_precheck`, resumable cells, env hashes | 5 |
| `scripts/run_x.py` | CLI | 5 |
| `tests/brain/test_p_spec.py` | + `MODULES["flymon/brain/x_spec.py"]` | 1 |
| `tests/brain/test_x_spec.py`, `test_x_verdict.py` | Task 1 tests | 1 |
| `tests/brain/test_x_oc.py` | Task 2 tests | 2 |
| `tests/brain/test_x_precheck.py` | Task 3 tests | 3 |
| `tests/brain/test_x_diag.py` | Task 4 tests | 4 |
| `tests/brain/test_x_runner.py`, `test_x_store.py`, `test_x_cli.py` | Task 5 tests | 5 |

Before the first test step: `mkdir -p /private/tmp/claude-503/x-tests`.

---

### Task 1: `x_spec` and `x_verdict` (numbers and the set rule)

**Files:**
- Create: `flymon/brain/x_spec.py`, `flymon/brain/x_verdict.py`
- Modify: `tests/brain/test_p_spec.py:29` (add one `MODULES` entry)
- Test: `tests/brain/test_x_spec.py`, `tests/brain/test_x_verdict.py`

**Interfaces:**
- Consumes: `flymon.brain.w_spec.SPEC`, `flymon.brain.w_verdict` (constants `P_*`, `V_*`, `VERDICT_LABEL`, `overall_code`).
- Produces:
  - `x_spec.XSpec`, `x_spec.SPEC`, and `x_spec.W_SHARED` (the tuple of field names equal to W's).
  - `x_verdict.S_STOP_MACHINE / S_FAIL / S_UNDECIDED / S_PASS` (= W's `V_*` ints) and `SET_LABEL`.
  - `x_verdict.counts(final) -> (n, m, b, invalid)`.
  - `x_verdict.set_code_counts(n, m, b, stop, p_set, xs) -> ndarray`, which broadcasts p_set.
  - `x_verdict.set_code(final, machine, p_set, xs) -> ndarray`.
  - `x_verdict.undecided_cause(n, xs) -> str`.
  - `x_verdict.m_needed(k, p_set, xs) -> int | None` and `x_verdict.m_needed_table(xs) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_p_spec.py`: add `"flymon/brain/x_spec.py": "flymon.brain.x_spec"` to `MODULES` (after the `w_spec` entry).

`tests/brain/test_x_spec.py`:

```python
"""X's numbers (X.2 · X.3 · X.4.1 · X.9.1): W's shared numbers unchanged, X's own grids / roots / calibration rule /
diagnostic grid, the W facts X re-checks; X's seed roots collide with nothing declared; no X file but x_spec holds a
number (literal guard)."""
import ast
import dataclasses
import importlib
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.w_spec import SPEC as W
from flymon.brain.x_spec import SPEC, W_SHARED, XSpec
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]
BASE, STRIDE = 1_000_000, 1000


def test_x_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/x_spec.py") == "flymon.brain.x_spec"


def test_w_shared_numbers_are_ws():
    for f in W_SHARED:
        assert getattr(SPEC, f) == getattr(W, f), f
    assert {"q_grid", "k_grid", "f_min", "f_max", "k_min", "k_cap", "cal_reps", "cal_tol", "cal_iter", "oc_chunk",
            "naive_max", "bar", "band_width", "round_digits", "mech_min", "min_gate_pairs", "cal_floor_rule",
            "cluster_grid", "d_power", "d_false", "p_power", "p_false", "oc_reps", "record_dprimes"} <= set(W_SHARED)


def test_x_numbers():
    assert SPEC.p_set_grid == (0.5, 0.625, 0.75, 0.875, 1.0) and SPEC.min_pass_pairs == 3
    assert (SPEC.bracket_mult, SPEC.w_bracket_mult, SPEC.cal_retry_iter) == (4.0, 1.0, 160)
    assert SPEC.w_rejection_tries == 200 and SPEC.chol_jitter == 1e-9
    assert (SPEC.precheck_reps, SPEC.diag_reps, SPEC.p26_reps) == (4000, 4000, 200)
    assert SPEC.het_scales == (0.5, 1.0, 2.0) and (SPEC.het_low_dprime, SPEC.het_all_dprime) == (0.5, 1.0)
    assert SPEC.variants == ("V0", "V1", "V2") and SPEC.v1_exclude_from == 5.0 and SPEC.min_variant_pairs == 6
    assert SPEC.v1_pairs == 7 and len(SPEC.balanced_pairs) == 3
    assert SPEC.resid_quantiles == (5.0, 25.0, 50.0, 75.0, 95.0)
    assert (SPEC.diag_pairs, SPEC.diag_f_max, SPEC.diag_fs, SPEC.diag_k_max) == (16, 64, (8, 16, 32, 64), 16)
    assert SPEC.diag_k_los == (4, 6, 8, 10, 12) and SPEC.diag_k_width == 4 and SPEC.diag_pairs >= SPEC.diag_k_max
    assert (SPEC.diag_floor_share, SPEC.diag_tries, SPEC.diag_fill_flag) == (0.10, 2000, 0.01)
    assert SPEC.diag_levels == (0.0, 25.0, 50.0, 75.0)
    assert SPEC.w_measure_key == "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    assert SPEC.w_pipeline_key == "9636cf4c7f868338f512cd7cf1a722027bd842c37528188859c14bebb04a2853"
    assert SPEC.w_measure_file_sha == "e23009146ee04e3f6f328445a51685f8f045f6bb194749bf4ebd8f80c6e2be63"
    assert SPEC.w_oc_detail_sha == "7a0a65f3f6b969d47b29f9a41ae45f2d1cfc4bf9bb76d028bf7172a9a6fbeb77"
    assert dict(SPEC.w_commits) == {"pilot": "5fbc4c8", "oc": "f30ae35"} and SPEC.w_summary_commit == "f30ae35"
    assert (SPEC.w_boot_fail_total, SPEC.w_boot_fail_power, SPEC.w_boot_fail_false, SPEC.w_boot_fail_both) == (
        29, 14, 18, 3)
    assert (SPEC.w_pilot_pairs, SPEC.w_pilot_units) == (16, 384)
    assert (SPEC.summary, SPEC.raw_dir, SPEC.archive_root) == ("results/summary/x_learning.json", "results/x",
                                                              "~/flymon-archive/x")
    assert SPEC.diag_detail == "results/x/precheck_diag.json"


def test_seed_roots_are_new():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {43_000_000, 43_200_000, 43_300_000}
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(XSpec))
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/x_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    _collect(importlib.import_module("flymon.brain.p_spec").SPEC, declared, seen)
    assert not any(43_000_000 <= s < 43_400_000 for s in declared)
    assert not any((s - BASE) // STRIDE in declared for s in range(43_000_000, 43_400_000, STRIDE))


def _numbers(path: Path) -> set:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def test_no_x_file_but_x_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("x_*.py")) if p.name != "x_spec.py"]
    if (ROOT / "scripts/run_x.py").exists():
        files.append(ROOT / "scripts/run_x.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
```

`tests/brain/test_x_verdict.py`:

```python
"""X.3's set rule (X.4.6's boundary fixtures with their fixed verdicts, the phase-A mutations each fixture catches, the
p_set 1.0 = W overall_code identity for n ≥ 4) and X.9.1.3's m-needed table with its duplicate marks."""
import numpy as np
import pytest

from flymon.brain import w_verdict as WV
from flymon.brain import x_verdict as XV
from flymon.brain.x_spec import SPEC as XS

P, F, B, I = WV.P_PASS, WV.P_FAIL, WV.P_BAND, WV.P_INVALID
FIX = [  # (n, m, b, p_set) -> verdict, X.4.6 + one BAND-robustness case
    ((4, 2, 0, 0.5), XV.S_FAIL), ((4, 3, 0, 0.75), XV.S_PASS), ((4, 3, 0, 0.875), XV.S_FAIL),
    ((6, 3, 0, 0.625), XV.S_FAIL), ((5, 3, 1, 0.5), XV.S_PASS), ((4, 2, 2, 0.5), XV.S_UNDECIDED),
    ((3, 3, 1, 0.5), XV.S_UNDECIDED), ((4, 1, 1, 0.5), XV.S_FAIL), ((8, 8, 0, 1.0), XV.S_PASS),
    ((8, 7, 0, 1.0), XV.S_FAIL), ((4, 3, 2, 0.75), XV.S_UNDECIDED)]


@pytest.mark.parametrize("args,want", FIX)
def test_boundary_fixtures(args, want):
    n, m, b, p = args
    assert int(XV.set_code_counts(n, m, b, False, p, XS)) == want
    final = np.array([P] * m + [F] * (n - m) + [B] * b)
    assert int(XV.set_code(final, False, p, XS)) == want


def test_causes():
    assert XV.undecided_cause(4, XS) == "BAND 잔존" and XV.undecided_cause(3, XS) == "판정 가능 쌍 < 4"


def test_mutations_are_caught():
    # m ≥ 3 removed → (4, 2, 0, 0.5) would PASS; BAND robustness removed → (4, 3, 2, 0.75) would PASS;
    # the (m + b) < 3 branch removed → (4, 2, 0, 0.5) would be UNDECIDED; BAND counted in n → (3, 3, 1, 0.5) would PASS
    assert int(XV.set_code_counts(4, 2, 0, False, 0.5, XS)) == XV.S_FAIL
    assert int(XV.set_code_counts(4, 3, 2, False, 0.75, XS)) == XV.S_UNDECIDED
    assert int(XV.set_code_counts(3, 3, 1, False, 0.5, XS)) == XV.S_UNDECIDED


def test_stop_machine_first():
    assert int(XV.set_code(np.array([P] * 6), True, 0.5, XS)) == XV.S_STOP_MACHINE
    assert int(XV.set_code(np.array([P] * 5 + [I]), False, 0.5, XS)) == XV.S_STOP_MACHINE


def test_p_set_one_equals_w_overall_code_when_n_at_least_4():
    rng = np.random.default_rng(0)
    for k in range(4, 9):
        fin = rng.choice([P, F, B], size=(4000, k), p=[0.7, 0.2, 0.1])
        mach = rng.random(4000) < 0.05
        n = XV.counts(fin)[0]
        x = XV.set_code(fin, mach, 1.0, XS)
        w = WV.overall_code(fin, mach, XS.min_gate_pairs)
        assert np.array_equal(x[n >= 4], w[n >= 4])


def test_p_set_broadcasts():
    fin = np.array([[P, P, P, F, F, F]])
    code = XV.set_code(fin, np.array([False]), np.asarray(XS.p_set_grid)[:, None], XS)
    assert code.shape == (5, 1) and code[:, 0].tolist() == [XV.S_PASS] + [XV.S_FAIL] * 4


def test_m_needed_table_x913():
    t = XV.m_needed_table(XS)
    assert t["k"] == [4, 5, 6, 7, 8]
    assert t["m"] == {"0.5": [3, 3, 3, 4, 4], "0.625": [3, 4, 4, 5, 5], "0.75": [3, 4, 5, 6, 6],
                      "0.875": [4, 5, 6, 7, 7], "1.0": [4, 5, 6, 7, 8]}
    # Reading 12: X.9.1.3 lists k 4's duplicates as 0.5 · 0.625 · 0.75 only; by its own m table 0.875 and 1.0 also
    # share m = 4 at k 4 — the code marks every equal cell
    assert t["duplicates"] == {"4": [[0.5, 0.625, 0.75], [0.875, 1.0]], "5": [[0.625, 0.75], [0.875, 1.0]],
                               "6": [[0.875, 1.0]], "7": [[0.875, 1.0]], "8": []}
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/brain/test_x_spec.py tests/brain/test_x_verdict.py tests/brain/test_p_spec.py -q > /private/tmp/claude-503/x-tests/t1.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t1.log`
Expected: errors such as `ModuleNotFoundError: No module named 'flymon.brain.x_spec'`.

- [ ] **Step 3: Write `flymon/brain/x_spec.py`**

```python
"""Every number of spec appendix X as amended by X.9.1 (X.9.1 > X.0–X.9; for W's rules X keeps W.9.10 > … > W.0) —
phase A only (X.9.1.1: orders 0 · 0a). Phase B adds its numbers here and changes none of these.
- XSpec is a plain frozen dataclass. W_SHARED lists the fields W's functions read (w_oc.calibrate / simulate /
  simple_normal, w_verdict): they carry W's values unchanged (X.2; test_x_spec pins them to w_spec.SPEC), so an
  XSpec can be passed where w_oc expects a spec. X's generator roots are its own.
- Seeds (X.2, X.9.1.1): X OC root 43_000_000 (phase A: X.4.6's synthetic validation, P2-6), precheck root 43_200_000
  (block 43_200_000–43_299_999), record-only diagnostics root 43_300_000 (block 43_300_000–43_399_999); every stream is
  SeedSequence([root, tag, …]). Phase B adds the probe 44_000_000 · training 46_000_000 · smoke 43_1xx_xxx blocks. The
  oracle seeds 24_700_xxx and W's root 42_000_000 (the W calibration diagnosis) are read from w_spec where used, never
  restated here (the seed collectors would see a collision with W).
- Paths: results/x/ (git-ignored), results/summary/x_learning.json (tracked), ~/flymon-archive/x."""
from __future__ import annotations

from dataclasses import dataclass

W_SHARED = ("bar", "band_width", "naive_max", "mech_min", "round_digits", "min_gate_pairs", "q_grid", "k_grid",
            "f_min", "f_max", "k_min", "k_cap", "envelope", "envelope_solo_from", "d_power", "p_power", "d_false",
            "p_false", "oc_reps", "cal_reps", "cal_tol", "cal_iter", "boot_draws", "boot_reps", "boot_level",
            "cluster_grid", "record_dprimes", "oc_chunk", "cal_floor_rule", "synth_reps", "synth_null_max",
            "synth_big_min", "synth_big_dprime", "synth_drift_dprime_min", "simple_normal_fs", "simple_normal_reps",
            "budget_h")


@dataclass(frozen=True)
class XSpec:
    # ---- W's numbers X keeps (X.2; equal to w_spec.SPEC, test_x_spec) ----------------------------------------------
    bar: float = 1.0
    band_width: float = 0.2
    naive_max: float = 0.5
    mech_min: float = 0.75
    round_digits: int = 9
    min_gate_pairs: int = 4
    q_grid: tuple = (0.5, 0.625, 0.75)
    k_grid: tuple = (8, 16)
    f_min: int = 8
    f_max: int = 32
    k_min: int = 4
    k_cap: int = 8
    envelope: int = 3
    envelope_solo_from: int = 29
    d_power: float = 1.5
    p_power: float = 0.80
    d_false: float = 0.5
    p_false: float = 0.05
    oc_reps: int = 4000
    cal_reps: int = 2000
    cal_tol: float = 0.02
    cal_iter: int = 40
    boot_draws: int = 200
    boot_reps: int = 400
    boot_level: float = 0.95
    cluster_grid: tuple = (0.0, 0.5, 1.0)
    record_dprimes: tuple = (0.5, 1.0, 1.5, 2.0)
    oc_chunk: int = 50
    cal_floor_rule: str = "zero"
    synth_reps: int = 1000
    synth_null_max: float = 0.02
    synth_big_min: float = 0.98
    synth_big_dprime: float = 4.0
    synth_drift_dprime_min: float = 1.5
    simple_normal_fs: tuple = (8, 16, 24, 32)
    simple_normal_reps: int = 2000
    budget_h: float = 24.0                          # X's own 24 h (X.8); W's spend not included
    # ---- the set rule (X.3, Q1 + Q2) ---------------------------------------------------------------------------------
    p_set_grid: tuple = (0.5, 0.625, 0.75, 0.875, 1.0)
    min_pass_pairs: int = 3
    # ---- calibration failure (X.9.1.2) -------------------------------------------------------------------------------
    bracket_mult: float = 4.0                       # W's hi_a / hi_b × 4 for every X calibration
    w_bracket_mult: float = 1.0                     # the W reproduction (diagnosis) only
    cal_retry_iter: int = 160                       # no_convergence → once more with 160 steps
    chol_jitter: float = 1e-9                       # w_oc's Cholesky jitter (x_oc's simulator copy)
    w_rejection_tries: int = 200                    # w_oc._pair_bases's rounds (x_oc.pair_bases = W at this value)
    # ---- synthetic fixtures (W.9.9 P2-11's structure, as w_oc.synthetic_validation) ---------------------------------
    synth_base: tuple = (40.0, 90.0)
    synth_sd: float = 4.0
    synth_corr: float = 0.8
    synth_learn: tuple = (20.0, 10.0)
    synth_drift: float = 15.0
    p26_reps: int = 200                             # stage0's P2-6 check on θ̂ (experiments per g)
    # ---- seeds -------------------------------------------------------------------------------------------------------
    oc_seed: int = 43_000_000
    precheck_seed: int = 43_200_000
    diag_seed: int = 43_300_000
    # ---- the precheck (X.9.1.1) and its records (X.4.5 at θ̂) --------------------------------------------------------
    precheck_reps: int = 4000
    het_scales: tuple = (0.5, 1.0, 2.0)             # Σ_v × (X.4.5)
    het_low_dprime: float = 0.5                     # W's heterogeneous scenarios: one / half pairs at 0.5, rest 1.5
    het_all_dprime: float = 1.0                     # W's "all 1.0"
    # ---- variants and pilot facts (X.4.4, X.0, X.9.1.3) -------------------------------------------------------------
    variants: tuple = ("V0", "V1", "V2")
    v1_exclude_from: float = 5.0                    # V1 drops |d′| ≥ 5
    min_variant_pairs: int = 6                      # fewer → diagonal only, V0's correlations
    v1_pairs: int = 7
    balanced_pairs: tuple = ("a|4|Rock Slide|Strength", "b|2|Mega Drain vs Machamp|Mega Drain vs Tentacruel",
                             "b|10|Surf vs Slowbro|Surf vs Venusaur")
    resid_quantiles: tuple = (5.0, 25.0, 50.0, 75.0, 95.0)
    # ---- record-only diagnostics (X.9.1.4) ---------------------------------------------------------------------------
    diag_reps: int = 4000
    diag_pairs: int = 16
    diag_f_max: int = 64
    diag_fs: tuple = (8, 16, 32, 64)                # (i)
    diag_k_max: int = 16                            # (iii)
    diag_k_los: tuple = (4, 6, 8, 10, 12)
    diag_k_width: int = 4                           # k range [k_lo, k_lo + 4]
    diag_floor_share: float = 0.10                  # floor "caught" when ≥ 10 % of draws truncate
    diag_tries: int = 2000
    diag_fill_flag: float = 0.01
    diag_levels: tuple = (0.0, 25.0, 50.0, 75.0)    # F2(ℓ) percentiles
    diag_chunk: int = 20
    diag_pair_chunk: int = 100
    # ---- W facts re-checked (X.2, X.4.3, X.5 1 in part) -------------------------------------------------------------
    w_measure_key: str = "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    w_pipeline_key: str = "9636cf4c7f868338f512cd7cf1a722027bd842c37528188859c14bebb04a2853"
    w_measure_file_sha: str = "e23009146ee04e3f6f328445a51685f8f045f6bb194749bf4ebd8f80c6e2be63"
    w_oc_detail_sha: str = "7a0a65f3f6b969d47b29f9a41ae45f2d1cfc4bf9bb76d028bf7172a9a6fbeb77"
    w_commits: tuple = (("pilot", "5fbc4c8"), ("oc", "f30ae35"))
    w_summary_commit: str = "f30ae35"
    w_boot_fail_total: int = 29
    w_boot_fail_power: int = 14
    w_boot_fail_false: int = 18
    w_boot_fail_both: int = 3
    w_pilot_pairs: int = 16
    w_pilot_units: int = 384
    # ---- paths and CLI -----------------------------------------------------------------------------------------------
    summary: str = "results/summary/x_learning.json"
    raw_dir: str = "results/x"
    wcal_detail: str = "results/x/w_cal_diag.json"
    precheck_detail: str = "results/x/precheck.json"
    diag_detail: str = "results/x/precheck_diag.json"
    progress_dir: str = "results/x/progress"
    archive_root: str = "~/flymon-archive/x"
    w_summary: str = "results/summary/w_learning.json"
    w_oc_detail: str = "results/w/oc.json"
    cli_print_chars: int = 2000


SPEC = XSpec()
```

- [ ] **Step 4: Write `flymon/brain/x_verdict.py`**

```python
"""X's set verdict (X.3 as amended by X.9.1.3; Q1 + Q2) over W's pair verdict (w_verdict, unchanged: fly gates, q,
2K BAND, mechanism control, RN1 = R1). n = judgeable gate pairs (final PASS or FAIL), m = PASS pairs, b = BAND left
(no 2K data); a machine reason or an INVALID pair → STOP_MACHINE first, so n + b = k.
- PASS: n ≥ 4 ∧ m ≥ 3 ∧ m / n ≥ p_set ∧ m / (n + b) ≥ p_set.
- FAIL: n ≥ 4 ∧ [(m + b) / (n + b) < p_set ∨ (m + b) < 3].
- otherwise UNDECIDED — "판정 가능 쌍 < 4" when n < 4, else "BAND 잔존".
Comparisons round(x − p_set, 9) (digits from the spec). With b = 0 and n ≥ 4 PASS and FAIL are complements; p_set 1.0
equals W's overall_code wherever n ≥ 4. Vectorised over leading axes; p_set broadcasts against them. The OC simulates
this code itself (x_oc.set_hits)."""
from __future__ import annotations

import numpy as np

from . import w_verdict as WV

S_STOP_MACHINE, S_FAIL, S_UNDECIDED, S_PASS = WV.V_STOP_MACHINE, WV.V_FAIL, WV.V_UNDECIDED, WV.V_PASS
SET_LABEL = dict(WV.VERDICT_LABEL)


def counts(final):
    """(n, m, b, invalid) over the pair axis (-1) of final pair codes."""
    f = np.asarray(final)
    m = (f == WV.P_PASS).sum(-1)
    return m + (f == WV.P_FAIL).sum(-1), m, (f == WV.P_BAND).sum(-1), (f == WV.P_INVALID).any(-1)


def _ratio(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def set_code_counts(n, m, b, stop, p_set, xs):
    n, m, b = (np.asarray(v, float) for v in (n, m, b))
    p, d = np.asarray(p_set, float), xs.round_digits
    nb = n + b
    enough = n >= xs.min_gate_pairs
    ok = (enough & (m >= xs.min_pass_pairs) & (np.round(_ratio(m, n) - p, d) >= 0)
          & (np.round(_ratio(m, nb) - p, d) >= 0))
    bad = enough & ((np.round(_ratio(m + b, nb) - p, d) < 0) | (m + b < xs.min_pass_pairs))
    return np.where(np.asarray(stop, bool), S_STOP_MACHINE, np.where(ok, S_PASS, np.where(bad, S_FAIL, S_UNDECIDED)))


def set_code(final, machine, p_set, xs):
    n, m, b, inv = counts(final)
    return set_code_counts(n, m, b, np.asarray(machine, bool) | inv, p_set, xs)


def undecided_cause(n, xs) -> str:
    return f"판정 가능 쌍 < {xs.min_gate_pairs}" if n < xs.min_gate_pairs else "BAND 잔존"


def m_needed(k: int, p_set: float, xs):
    """The smallest m that PASSes with n = k, b = 0 (None: none)."""
    for m in range(k + 1):
        if int(set_code_counts(k, m, 0, False, p_set, xs)) == S_PASS:
            return m
    return None


def m_needed_table(xs) -> dict:
    """X.9.1.3 P1-3: m needed per (p_set, k) and the per-k duplicate cells (records only — selection unchanged)."""
    ks = list(range(xs.k_min, xs.k_cap + 1))
    table = {str(p): [m_needed(k, p, xs) for k in ks] for p in xs.p_set_grid}
    dup = {}
    for j, k in enumerate(ks):
        groups = {}
        for p in xs.p_set_grid:
            groups.setdefault(table[str(p)][j], []).append(p)
        dup[str(k)] = [g for g in groups.values() if len(g) > 1]
    return dict(k=ks, m=table, duplicates=dup, note="중복(기록용) — 선택 규칙은 바꾸지 않는다(X.9.1.3 P1-3)")
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/brain/test_x_spec.py tests/brain/test_x_verdict.py tests/brain/test_p_spec.py tests/brain/test_w_spec.py -q > /private/tmp/claude-503/x-tests/t1.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t1.log`
Expected: all pass. W's own seed test sees X's roots as declared and must not collide with them. If `test_seed_roots_are_new` fails, a block in 43_000_000–43_399_999 is already declared, so stop and report it; do not move the roots.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/x_spec.py flymon/brain/x_verdict.py tests/brain/test_x_spec.py tests/brain/test_x_verdict.py tests/brain/test_p_spec.py
git commit -m "feat(x): x_spec (phase A numbers, W shared numbers pinned, roots 43_000_000/43_200_000/43_300_000) and x_verdict (X.3 set rule, m-needed table)"
```

---

### Task 2: `x_oc` core: calibration per status, X evaluate, synthetic validation, P2-6, W calibration diagnosis

**Files:**
- Create: `flymon/brain/x_oc.py`
- Test: `tests/brain/test_x_oc.py`

**Interfaces:**
- Consumes: `w_oc` (`fit`, `boot`, `summary`, `simulate`, `slot_means`, `to_stages`, `true_dprimes`, `_bisect`, `_rng`, `_uniform`, `sd_pre`, `synthetic_pilot`, `simple_normal`, `SIMPLE_NORMAL_TOL`, `TAG_*`), `w_verdict`, `x_verdict.set_code_counts`, `x_spec.XSpec`.
- Produces (used by Tasks 3–5):
  - Tags: `TAG_CAL, TAG_POINT, TAG_RECORD, TAG_SYNTH` (W's values) and `TAG_P26 = 10`.
  - `rng(root, *tags)` and `grid_shape(xs) -> tuple`, the shape (p_set, q, K, F, k).
  - `brackets(theta, mult) -> (hi_a, hi_b)`.
  - `calibrate(theta, target, mode, idx, z, spec, mult, n_iter) -> dict`, which is W's `calibrate` dict.
  - `failure(c) -> dict | None`, giving `{handle, status}`.
  - `fill_value(mode) -> float`.
  - `calibrate_x(theta, target, mode, idx, z, xs) -> dict`: calibrate's dict plus `first_failure`, `retried`, `failure` and `fill`.
  - `pair_bases(theta, rng_, n_rep, n_pair, g, z, xs, tries, accept=None) -> (bases, filled)`.
  - `simulate(theta, rng_, n_rep, n_pair, n_fly, n_probe, a_pair, b_pair, fly_a, fly_b, g, z, xs, tries, accept=None) -> (stages, bases, filled)`.
  - `pair_finals(d, z, xs, qs, ks, fs)`, a generator of `(qi, ki, fi, fin)`.
  - `set_hits(fin, mach, kk, xs) -> ndarray [p_set, len(kk)]`.
  - `tally(d, z, xs, qs, ks, fs, kk) -> ndarray [p, q, K, F, k]`.
  - `evaluate(theta, rng_, n_rep, a_pair, b_pair, fly_a, fly_b, g, z, xs) -> ndarray [p, q, K, F, k]`.
  - `synthetic_validation(xs, z, n_rep=None) -> dict`, with key `ok`.
  - `bit_identity(theta, z, xs, w_spec, n_rep, a, b) -> dict`, with keys `equal`, `max_p`, `by_g`.
  - `oc_timing(theta, z, xs) -> dict`.
  - `w_cal_diagnosis(theta, z, w_spec, xs, w_boot_cal, n_draws=None, log=None) -> dict`, with keys `rows`, `diffs`, `reproduced`, `counts` and `n_draws`.
  - `run_cell(tag, key, fn)`, the default cell runner, which simply calls `fn()`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_x_oc.py`:

```python
"""x_oc core (X.4.2, X.4.6, X.9.1.2, X.9.1.3 P2-6, X.4.3 diagnosis): W's brackets and steps reproduce w_oc.calibrate
exactly; X's brackets are ×4; floor fills power 0 / false pass 1; no_convergence retries with 160 steps; X's
simulator copy equals w_oc's at W's rounds; set_hits equals the per-k loop; X's evaluate at p_set 1.0 equals
w_oc.evaluate bit for bit; P(PASS) never rises with p_set; the synthetic fixtures pass at every p_set; the W diagnosis
reproduces w_oc.run's stored bootstrap calibrations and flags a tampered one."""
import dataclasses
import json

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain import x_oc
from flymon.brain import x_verdict as XV
from flymon.brain.h3_store import canonical
from flymon.brain.w_spec import SPEC as W
from flymon.brain.x_spec import SPEC as XS

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}


@pytest.fixture(scope="module")
def pilot():
    return w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))


@pytest.fixture(scope="module")
def theta(pilot):
    return w_oc.fit(pilot)


@pytest.fixture(scope="module")
def idx(theta):
    return np.random.default_rng(1).integers(0, len(theta["resid"]), XS.cal_reps)


@pytest.mark.parametrize("target,mode", [(1.5, "min"), (0.5, "max"), (40.0, "min")])
def test_w_brackets_reproduce_w_calibrate(theta, idx, target, mode):
    x = x_oc.calibrate(theta, target, mode, idx, Z, XS, XS.w_bracket_mult, XS.cal_iter)
    assert x == w_oc.calibrate(theta, target, mode, idx, Z, W)


def test_brackets_times_four(theta):
    a1, b1 = x_oc.brackets(theta, 1.0)
    a4, b4 = x_oc.brackets(theta, XS.bracket_mult)
    assert (a4, b4) == (a1 * 4.0, b1 * 4.0)


def test_status_fill(theta, idx):
    p = x_oc.calibrate_x(theta, 40.0, "min", idx, Z, XS)
    assert not p["ok"] and p["fill"] == 0.0 and p["failure"]["status"] in ("floor", "no_convergence")
    f = x_oc.calibrate_x(theta, 40.0, "max", idx, Z, XS)
    assert not f["ok"] and f["fill"] == 1.0           # false-pass floor counts as a pass, never as 0
    ok = x_oc.calibrate_x(theta, 1.5, "min", idx, Z, XS)
    assert ok["ok"] and ok["fill"] is None and ok["failure"] is None


def test_no_convergence_is_retried_with_more_steps(theta, idx):
    one = dataclasses.replace(XS, cal_iter=1, cal_retry_iter=60)
    c = x_oc.calibrate_x(theta, 1.5, "min", idx, Z, one)
    assert c["first_failure"] == dict(handle="a", status="no_convergence") and c["retried"] and c["ok"]
    never = dataclasses.replace(XS, cal_iter=1, cal_retry_iter=1)
    c = x_oc.calibrate_x(theta, 1.5, "min", idx, Z, never)
    assert c["retried"] and not c["ok"] and c["fill"] == 0.0


def test_simulator_copy_equals_w_at_w_rounds(theta):
    sdp = w_oc.sd_pre(theta, Z)
    b_x, filled = x_oc.pair_bases(theta, np.random.default_rng(7), 30, 8, 0.5, Z, XS, XS.w_rejection_tries)
    b_w = w_oc._pair_bases(theta, np.random.default_rng(7), 30, 8, 0.5, Z, XS.naive_max, sdp)
    assert np.array_equal(b_x, b_w) and filled.shape == (30, 8)
    ab = ([3.0] * 8, [2.0] * 8, [1.0] * 12, [1.0] * 12)
    sx, _, _ = x_oc.simulate(theta, np.random.default_rng(9), 6, 8, 12, 16, *ab, 1.0, Z, XS, XS.w_rejection_tries)
    sw = w_oc.simulate(theta, np.random.default_rng(9), 6, 8, 12, 16, *ab, 1.0, Z, XS.naive_max)
    assert all(np.array_equal(sx[s], sw[s]) for s in sw)


def test_set_hits_equals_the_per_k_loop():
    rng = np.random.default_rng(2)
    fin = rng.choice([WV.P_PASS, WV.P_FAIL], size=(200, 8), p=[0.8, 0.2])
    mach = rng.random((200, 8)) < 0.01
    kk = list(range(4, 9))
    got = x_oc.set_hits(fin, mach, kk, XS)
    for j, k in enumerate(kk):
        for pi, p in enumerate(XS.p_set_grid):
            want = (XV.set_code(fin[:, :k], mach[:, :k].any(-1), p, XS) == XV.S_PASS).sum()
            assert got[pi, j] == want


def test_p2_6_bit_identity_synthetic(theta, idx):
    c = x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)
    out = x_oc.bit_identity(theta, Z, XS, W, 60, c["a"]["value"], c["b"]["value"])
    assert out["equal"], out
    assert out["max_p"] > 0.0                         # the comparison is not between two zero arrays


def test_evaluate_shape_and_p_set_order(theta, idx):
    c = x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)
    p = x_oc.evaluate(theta, x_oc.rng(1, 2), 40, *w_oc._uniform(XS, c["a"]["value"], c["b"]["value"]), 0.5, Z, XS)
    assert p.shape == x_oc.grid_shape(XS) == (5, 3, 2, 25, 5)
    assert np.all(np.diff(p, axis=0) <= 0)            # a larger p_set never passes more
    assert np.all(np.diff(p[-1], axis=-1) <= 1e-12)   # p_set 1.0 (all pairs PASS) never rises with k; a ratio
                                                      # rule may (3/4 FAIL at 0.875 → 4/5 PASS), so no k test there


def test_synthetic_validation_at_every_p_set():
    r = x_oc.synthetic_validation(XS, Z, n_rep=300)
    assert r["ok"], r
    assert r["zero_effect"]["max_p"] <= 0.02 and r["big_effect"]["min_p"] >= 0.98
    assert r["one_gate"]["max_p"] <= 0.02 and r["negative_correlation"]["max_p"] <= 0.02


def test_w_cal_diagnosis_reproduces_w_run(pilot, theta):
    doc = json.loads(canonical(w_oc.run(pilot, Z, W, lambda K, F: K * F, n_boot=3, n_rep=2, n_boot_rep=2)))
    d = x_oc.w_cal_diagnosis(theta, Z, W, XS, doc["boot_calibration"])
    assert d["reproduced"] and d["diffs"] == [] and d["n_draws"] == 3 and len(d["rows"]) == 3
    assert set(d["rows"][0]) == {"draw", "min", "max"} and "x_rule" in d["rows"][0]["min"]
    bad = json.loads(json.dumps(doc["boot_calibration"]))
    bad[1]["max"]["ok"] = not bad[1]["max"]["ok"]
    d2 = x_oc.w_cal_diagnosis(theta, Z, W, XS, bad)
    assert not d2["reproduced"] and d2["diffs"][0]["draw"] == 1 and d2["diffs"][0]["side"] == "max"
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/brain/test_x_oc.py -q > /private/tmp/claude-503/x-tests/t2.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t2.log`
Expected: `ModuleNotFoundError: No module named 'flymon.brain.x_oc'`.

- [ ] **Step 3: Write `flymon/brain/x_oc.py` (core section)**

```python
"""X's operating-characteristic pieces, phase A (X.4.2, X.4.4, X.4.6, X.9.1.1–X.9.1.4; X.9.1.2 replaces X.4.3's rule).
W's model is used as is (w_oc: fit, generator, bootstrap draw, true d′); X adds only the following.
- The p_set axis: evaluate runs w_oc.simulate (the same draws as w_oc.evaluate) and W's pair verdict (w_verdict:
  fly gates, q, 2K BAND, mechanism control, RN1 = R1), then X's set rule (x_verdict) on the first k pairs for every
  p_set at once → P(PASS) [p_set, q, K, F, k]. In simulation BAND ends at 2K, so b = 0. At p_set 1.0 this equals
  w_oc.evaluate bit for bit (P2-6).
- Calibration per status (X.9.1.2): W's bisection with the bracket's upper end × bracket_mult (4); a no_convergence
  is retried once with cal_retry_iter steps (same draws, tolerance, brackets). A failure that remains — floor,
  no_convergence after the retry, or above_at_zero on the power side — fills P(PASS) with 0 (power) / 1 (false pass).
  On the false-pass side above_at_zero is W.9.10 2's zero_floor (not a failure).
- A simulator copy (pair_bases / simulate) with a rejection-round count and an extra acceptance predicate, for the
  record-only diagnostics. At W's 200 rounds and no predicate it equals w_oc's draw for draw.
- The W calibration-failure diagnosis (X.4.3 + X.9.1.2): W's bootstrap draws re-drawn from W's root, compared with W's
  stored calibrations, every status recorded; records only.
Every stream is SeedSequence([root, tag, …]); the roots are X's (x_spec) except the W reproduction (w_spec)."""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from . import w_oc
from . import w_verdict as WV
from . import x_verdict as XV

TAG_CAL, TAG_POINT, TAG_RECORD, TAG_SYNTH = w_oc.TAG_CAL, w_oc.TAG_POINT, w_oc.TAG_RECORD, w_oc.TAG_SYNTH
TAG_P26 = 10
MODES = (("min", "d_power"), ("max", "d_false"))


def rng(root, *tags):
    return np.random.default_rng(np.random.SeedSequence([int(root)] + [int(t) for t in tags]))


def run_cell(tag, key, fn):
    """The default cell runner (no cache); x_runner passes a resumable one."""
    return fn()


def grid_shape(xs) -> tuple:
    return (len(xs.p_set_grid), len(xs.q_grid), len(xs.k_grid), xs.f_max - xs.f_min + 1, xs.k_cap - xs.k_min + 1)


# ================================================================ calibration per status (X.9.1.2)
def brackets(theta, mult: float) -> tuple:
    """W's hi_a / hi_b (w_oc.calibrate's formulas) × mult."""
    r = np.abs(theta["resid"]).max()
    hi_a = float((theta["m0s"][WV.P, WV.X] + theta["drift"][0][WV.P, WV.X] + r) * 2 + 1)
    hi_b = float((theta["m0s"][WV.A, WV.X] + np.abs(theta["drift"]).sum() + r) * 2 + 1)
    return hi_a * mult, hi_b * mult


def calibrate(theta, target: float, mode: str, idx, z, spec, mult: float, n_iter: int) -> dict:
    """w_oc.calibrate with brackets × mult and n_iter steps; mult 1.0 with spec.cal_iter is w_oc.calibrate exactly."""
    agg = np.min if mode == "min" else np.max
    hi_a, hi_b = brackets(theta, mult)
    ra = w_oc._bisect(lambda x: agg(w_oc.true_dprimes(theta, x, 0.0, idx, z)[[0, 2]]) - target, hi_a, spec.cal_tol,
                      n_iter)
    if ra["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        ra = dict(value=0.0, status="zero_floor", f0=ra["f0"])
    if ra["value"] is None:
        return dict(ok=False, a=ra, b=None)
    rb = w_oc._bisect(lambda x: agg(w_oc.true_dprimes(theta, ra["value"], x, idx, z)[[1, 3]]) - target, hi_b,
                      spec.cal_tol, n_iter)
    if rb["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        rb = dict(value=0.0, status="zero_floor", f0=rb["f0"])
    if rb["value"] is None:
        return dict(ok=False, a=ra, b=rb)
    td = w_oc.true_dprimes(theta, ra["value"], rb["value"], idx, z)
    return dict(ok=True, a=ra, b=rb, true_dprime=dict(zip(WV.GATES, td.tolist())))


def failure(c: dict):
    """{handle, status} of a failed calibration, None when it succeeded."""
    if c["ok"]:
        return None
    h = "a" if c["a"]["value"] is None else "b"
    return dict(handle=h, status=c[h]["status"])


def fill_value(mode: str) -> float:
    return 0.0 if mode == "min" else 1.0


def calibrate_x(theta, target: float, mode: str, idx, z, xs) -> dict:
    """X.9.1.2: brackets × xs.bracket_mult, xs.cal_iter steps; a no_convergence → once more with xs.cal_retry_iter
    steps; a remaining failure fills (power 0 / false pass 1)."""
    c = calibrate(theta, target, mode, idx, z, xs, xs.bracket_mult, xs.cal_iter)
    first = failure(c)
    retried = first is not None and first["status"] == "no_convergence"
    if retried:
        c = calibrate(theta, target, mode, idx, z, xs, xs.bracket_mult, xs.cal_retry_iter)
    return dict(c, first_failure=first, retried=retried, failure=failure(c),
                fill=None if c["ok"] else fill_value(mode))


# ================================================================ the simulator copy (diagnostics)
def pair_bases(theta, rng_, n_rep, n_pair, g, z, xs, tries: int, accept=None) -> tuple:
    """w_oc._pair_bases with `tries` rounds and an extra predicate accept(base [..., 2, 2]) -> bool [...]; returns
    (bases [n_rep, n_pair, 2, 2], filled [n_rep, n_pair] — the slots given the population pair)."""
    sdp = w_oc.sd_pre(theta, z)
    L = np.linalg.cholesky(theta["pair_cov"] + xs.chol_jitter * np.eye(4))
    w = (rng_.standard_normal((n_rep, 4)) @ L.T * math.sqrt(g)).reshape(n_rep, 1, 2, 2)
    out = np.empty((n_rep, n_pair, 2, 2))
    need = np.ones((n_rep, n_pair), bool)
    for _ in range(tries):
        u = (rng_.standard_normal((n_rep, n_pair, 4)) @ L.T).reshape(n_rep, n_pair, 2, 2)
        base = theta["m0s"] + w + u
        ok = np.abs(WV.dv(base, z)) / sdp < xs.naive_max
        if accept is not None:
            ok &= accept(base)
        take = need & ok
        out[take] = base[take]
        need &= ~ok
        if not need.any():
            return out, need
    out[need] = (theta["m0s"] + w + 0 * out)[need]
    return out, need


def simulate(theta, rng_, n_rep, n_pair, n_fly, n_probe, a_pair, b_pair, fly_a, fly_b, g, z, xs, tries: int,
             accept=None) -> tuple:
    """w_oc.simulate through pair_bases: (stages, bases, filled)."""
    base, filled = pair_bases(theta, rng_, n_rep, n_pair, g, z, xs, tries, accept)
    Lf = np.linalg.cholesky(theta["fly_cov"] + xs.chol_jitter * np.eye(4))
    v = (rng_.standard_normal((n_rep, n_pair, n_fly, 4)) @ Lf.T).reshape(n_rep, n_pair, n_fly, 2, 2)
    a = np.asarray(a_pair, float)[None, :, None] * np.asarray(fly_a, float)[None, None, :]
    b = np.asarray(b_pair, float)[None, :, None] * np.asarray(fly_b, float)[None, None, :]
    mu = w_oc.slot_means(theta, base[:, :, None], v, np.broadcast_to(a, v.shape[:3]), np.broadcast_to(b, v.shape[:3]))
    idx = rng_.integers(0, len(theta["resid"]), (n_rep, n_pair, n_fly, n_probe))
    counts = np.clip(np.rint(mu[:, :, :, None] + theta["resid"][idx]), 0, None).astype(np.int32)
    return w_oc.to_stages(counts), base, filled


# ================================================================ evaluation: W's pair verdict, X's set rule
def pair_finals(d, z, xs, qs, ks, fs):
    """Yields (qi, ki, fi, final pair codes [n, pairs]) — W's pair verdict per design, exactly as w_oc.evaluate."""
    kw = dict(bar=xs.bar, band=xs.band_width, digits=xs.round_digits)
    for ki, K in enumerate(ks):
        dK = {s: v[..., :K, :, :] for s, v in d.items()}
        d2 = {s: v[..., :2 * K, :, :] for s, v in d.items()}
        cK, c2 = WV.fly_class(WV.gate_stats(dK, z), **kw), WV.fly_class(WV.gate_stats(d2, z), **kw)
        fK, f2 = WV.mech_fractions(dK), WV.mech_fractions(d2)
        for fi, F in enumerate(fs):
            mK = WV.mech_ok(fK[..., :F, :], xs.mech_min, xs.round_digits)
            m2 = WV.mech_ok(f2[..., :F, :], xs.mech_min, xs.round_digits)
            for qi, q in enumerate(qs):
                yield qi, ki, fi, WV.pair_final(WV.pair_gate_code(cK[..., :F], q, F, xs.round_digits), mK,
                                                WV.pair_gate_code(c2[..., :F], q, F, xs.round_digits), m2)


def set_hits(fin, mach, kk, xs) -> np.ndarray:
    """[p_set, len(kk)] PASS counts of X's set rule on the first k pairs (nested in k), every p_set and k at once."""
    fin, mach = np.asarray(fin), np.asarray(mach, bool)
    j = np.asarray(kk) - 1
    m = (fin == WV.P_PASS).cumsum(-1)[:, j]
    n = m + (fin == WV.P_FAIL).cumsum(-1)[:, j]
    b = (fin == WV.P_BAND).cumsum(-1)[:, j]
    stop = ((mach | (fin == WV.P_INVALID)).cumsum(-1) > 0)[:, j]
    ps = np.asarray(xs.p_set_grid, float)[:, None, None]
    return (XV.set_code_counts(n, m, b, stop, ps, xs) == XV.S_PASS).sum(1)


def tally(d, z, xs, qs, ks, fs, kk) -> np.ndarray:
    mach = WV.rn1_mismatch(d)
    out = np.zeros((len(xs.p_set_grid), len(qs), len(ks), len(fs), len(kk)))
    for qi, ki, fi, fin in pair_finals(d, z, xs, qs, ks, fs):
        out[:, qi, ki, fi] = set_hits(fin, mach, kk, xs)
    return out


def evaluate(theta, rng_, n_rep, a_pair, b_pair, fly_a, fly_b, g, z, xs) -> np.ndarray:
    """P(PASS) [p_set, q, K, F, k] on w_oc.simulate (the draws of w_oc.evaluate) under X's set rule."""
    fs = list(range(xs.f_min, xs.f_max + 1))
    kk = list(range(xs.k_min, xs.k_cap + 1))
    hits = np.zeros(grid_shape(xs))
    done = 0
    while done < n_rep:
        n = min(xs.oc_chunk, n_rep - done)
        d = w_oc.simulate(theta, rng_, n, xs.k_cap, xs.f_max, 2 * max(xs.k_grid), a_pair, b_pair, fly_a, fly_b, g, z,
                          xs.naive_max)
        hits += tally(d, z, xs, xs.q_grid, xs.k_grid, fs, kk)
        done += n
    return hits / n_rep


# ================================================================ X.4.6 synthetic validation, P2-6, timing
def synthetic_validation(xs, z: dict, n_rep: int | None = None) -> dict:
    """X.4.6: W.9.9 P2-11's five fixtures (w_oc.synthetic_validation's structure) on X's evaluate and calibration,
    checked over every p_set: zero effect ≤ synth_null_max, big effect (d′ synth_big_dprime) ≥ synth_big_min, one
    gate only ≤, negative correlation ≤, the simple normal model not rising from F 8 to 32."""
    n_rep = xs.synth_reps if n_rep is None else int(n_rep)
    r = rng(xs.oc_seed, TAG_SYNTH)
    kw = dict(base=xs.synth_base, sd=xs.synth_sd, corr=xs.synth_corr, learn=xs.synth_learn)
    th0 = w_oc.fit(w_oc.synthetic_pilot(r, **kw))
    th_drift = w_oc.fit(w_oc.synthetic_pilot(r, drift_ax=xs.synth_drift, **kw))
    idx = r.integers(0, len(th0["resid"]), xs.cal_reps)
    lo, hi = xs.synth_null_max, xs.synth_big_min
    res = {}
    z0 = evaluate(th0, r, n_rep, *w_oc._uniform(xs, 0.0, 0.0), 0.0, z, xs)
    res["zero_effect"] = dict(max_p=float(z0.max()), limit=lo, ok=bool(z0.max() <= lo))
    big = calibrate_x(th0, xs.synth_big_dprime, "min", idx, z, xs)
    pb = (evaluate(th0, r, n_rep, *w_oc._uniform(xs, big["a"]["value"], big["b"]["value"]), 0.0, z, xs)
          if big["ok"] else np.zeros(grid_shape(xs)))
    res["big_effect"] = dict(min_p=float(pb.min()), min_p_by_p_set=pb.reshape(len(xs.p_set_grid), -1).min(1).tolist(),
                             limit=hi, ok=bool(big["ok"] and pb.min() >= hi), true_dprime=big.get("true_dprime"))
    one = evaluate(th_drift, r, n_rep, *w_oc._uniform(xs, 0.0, 0.0), 0.0, z, xs)
    td = w_oc.true_dprimes(th_drift, 0.0, 0.0, r.integers(0, len(th_drift["resid"]), xs.cal_reps), z)
    res["one_gate"] = dict(max_p=float(one.max()), limit=lo,
                           ok=bool(one.max() <= lo and td[1] >= xs.synth_drift_dprime_min),
                           true_dprime=dict(zip(WV.GATES, td.tolist())))
    if big["ok"]:
        alt = [1.0 if i % 2 == 0 else 0.0 for i in range(xs.f_max)]
        neg = evaluate(th0, r, n_rep, [big["a"]["value"]] * xs.k_cap, [big["b"]["value"]] * xs.k_cap, alt,
                       [1.0 - x for x in alt], 0.0, z, xs)
        res["negative_correlation"] = dict(max_p=float(neg.max()), limit=lo, ok=bool(neg.max() <= lo))
    else:
        res["negative_correlation"] = dict(max_p=None, ok=False)
    sn = w_oc.simple_normal(xs, r)
    p = np.array([sn[F] for F in xs.simple_normal_fs])
    res["simple_normal"] = dict(p_by_F={str(k): v for k, v in sn.items()}, diffs=np.diff(p).tolist(),
                                tol=w_oc.SIMPLE_NORMAL_TOL, ok=bool(np.all(np.diff(p) <= w_oc.SIMPLE_NORMAL_TOL)))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def bit_identity(theta, z, xs, w_spec, n_rep: int, a: float, b: float) -> dict:
    """X.9.1.3 P2-6: X's evaluate at p_set 1.0 (b = 0 in simulation) equals w_oc.evaluate on the same θ and seed,
    numpy.array_equal over [q, K, F, k], at every cluster level (root oc_seed, tag TAG_P26, g index)."""
    pi = xs.p_set_grid.index(1.0)
    u = w_oc._uniform(xs, a, b)
    by_g, top = {}, 0.0
    for gi, g in enumerate(xs.cluster_grid):
        x = evaluate(theta, rng(xs.oc_seed, TAG_P26, gi), n_rep, *u, g, z, xs)[pi]
        w = w_oc.evaluate(theta, rng(xs.oc_seed, TAG_P26, gi), n_rep, *u, g, z, w_spec)
        by_g[f"g{g}"] = dict(equal=bool(np.array_equal(x, w)), max_abs_diff=float(np.abs(x - w).max()))
        top = max(top, float(w.max()))
    return dict(equal=all(v["equal"] for v in by_g.values()), by_g=by_g, max_p=top, n_rep=int(n_rep), a=a, b=b)


def oc_timing(theta, z, xs) -> dict:
    """X.5 0: one evaluate at boot_reps experiments (θ̂'s power calibration, the last cluster level), scaled to the
    precheck (g × 2 targets × precheck_reps) and to phase B's bootstrap (boot_draws × variants × g × 2 × boot_reps)."""
    idx = rng(xs.oc_seed, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    c = calibrate_x(theta, xs.d_power, "min", idx, z, xs)
    a, b = (c["a"]["value"], c["b"]["value"]) if c["ok"] else (0.0, 0.0)
    t0 = time.perf_counter()
    evaluate(theta, rng(xs.oc_seed, TAG_P26, len(xs.cluster_grid)), xs.boot_reps, *w_oc._uniform(xs, a, b),
             xs.cluster_grid[-1], z, xs)
    s = time.perf_counter() - t0
    per = len(xs.cluster_grid) * len(MODES)
    return dict(evaluate_s=s, reps=xs.boot_reps, precheck_point_s=s * per * xs.precheck_reps / xs.boot_reps,
                phase_b_boot_s=s * per * xs.boot_draws * len(xs.variants))


# ================================================================ W calibration-failure diagnosis (X.4.3 + X.9.1.2)
def _statuses(c: dict) -> dict:
    return dict(a=(c["a"] or {}).get("status"), b=(c["b"] or {}).get("status"))


def _lab(x) -> str:
    return x if isinstance(x, str) else f"{x['handle']}|{x['status']}"


def w_cal_diagnosis(theta, z, w_spec, xs, w_boot_cal: list, n_draws: int | None = None, log=None) -> dict:
    """W's bootstrap draws re-drawn from W's root (w_oc._rng(w_spec, TAG_BOOT, bi): boot, then the calibration
    draws) and calibrated: (가) as W did (brackets × w_bracket_mult, w_spec.cal_iter), compared with W's stored
    (ok, a, b); (나) a failed draw with brackets × bracket_mult; (다) a no_convergence draw with cal_retry_iter steps
    (W's brackets); (라) X's rule (calibrate_x). Records only: the rule is fixed (X.9.1.2)."""
    n = len(w_boot_cal) if n_draws is None else int(n_draws)
    rows, diffs = [], []
    for bi in range(n):
        rb = w_oc._rng(w_spec, w_oc.TAG_BOOT, bi)
        tb = w_oc.boot(theta, rb)
        ib = rb.integers(0, len(tb["resid"]), w_spec.cal_reps)
        row = dict(draw=bi)
        for m, field in MODES:
            t = getattr(w_spec, field)
            c = calibrate(tb, t, m, ib, z, w_spec, xs.w_bracket_mult, w_spec.cal_iter)
            got = dict(ok=c["ok"], a=(c["a"] or {}).get("value"), b=(c["b"] or {}).get("value"))
            if got != w_boot_cal[bi][m]:
                diffs.append(dict(draw=bi, side=m, x=got, w=w_boot_cal[bi][m]))
            e = dict(w=got, status=_statuses(c), failure=failure(c))
            if not c["ok"]:
                e["bracket_x4"] = failure(calibrate(tb, t, m, ib, z, w_spec, xs.bracket_mult, w_spec.cal_iter)) or "ok"
                if e["failure"]["status"] == "no_convergence":
                    e["retry_iter"] = failure(calibrate(tb, t, m, ib, z, w_spec, xs.w_bracket_mult,
                                                        xs.cal_retry_iter)) or "ok"
            e["x_rule"] = failure(calibrate_x(tb, t, m, ib, z, xs)) or "ok"
            row[m] = e
        rows.append(row)
        if log is not None and (bi + 1) % 10 == 0:
            log(f"x w-cal diagnosis {bi + 1}/{n}")

    def side(m):
        out = dict(n_fail=0, by_handle_status={}, bracket_x4={}, retry_iter={}, x_rule={}, statuses={})
        for r in rows:
            e = r[m]
            for h, s in e["status"].items():
                k = f"{h}|{s}"
                out["statuses"][k] = out["statuses"].get(k, 0) + 1
            if not e["w"]["ok"]:
                out["n_fail"] += 1
                for col, v in (("by_handle_status", e["failure"]), ("bracket_x4", e.get("bracket_x4")),
                               ("retry_iter", e.get("retry_iter"))):
                    if v is not None:
                        out[col][_lab(v)] = out[col].get(_lab(v), 0) + 1
            out["x_rule"][_lab(e["x_rule"])] = out["x_rule"].get(_lab(e["x_rule"]), 0) + 1
        return out
    fm = {r["draw"] for r in rows if not r["min"]["w"]["ok"]}
    fx = {r["draw"] for r in rows if not r["max"]["w"]["ok"]}
    counts = dict(total=len(fm | fx), power=len(fm), false=len(fx), both=len(fm & fx), min=side("min"),
                  max=side("max"))
    return dict(rows=rows, diffs=diffs, reproduced=not diffs, counts=counts, n_draws=n)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/brain/test_x_oc.py tests/brain/test_x_spec.py -q > /private/tmp/claude-503/x-tests/t2.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t2.log`
Expected: all pass. `test_no_x_file_but_x_spec_holds_a_number` now also scans `x_oc.py`. If `test_no_convergence_is_retried_with_more_steps` lands within tolerance after one step, use `cal_iter=1` with target 2.0 instead. If target 40.0 turns out reachable under the ×4 brackets in `test_status_fill`, use 400.0. Do not weaken the assertions.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/x_oc.py tests/brain/test_x_oc.py
git commit -m "feat(x): x_oc core — per-status calibration (brackets x4, 160-step retry, fill 0/1), X evaluate with the p_set axis (P2-6 bit-identical to w_oc.evaluate), X.4.6 synthetic validation, W calibration diagnosis"
```

---

### Task 3: precheck, records at θ̂, variants, residual comparison, `x_rules`

**Files:**
- Modify: `flymon/brain/x_oc.py` (append a section; nothing above changes)
- Create: `flymon/brain/x_rules.py`
- Test: `tests/brain/test_x_precheck.py`

**Interfaces:**
- Consumes: Task 2's `rng`, `calibrate_x`, `evaluate`, `grid_shape`, `run_cell` and the `TAG_*` constants. Also `w_oc._uniform`, `w_oc._stack`, `w_oc.fit` and `w_oc.summary`.
- Produces:
  - `x_oc.pair_cov_variant(pair_means, abs_d, name, xs) -> ndarray`, where name ∈ {"V0","V1","V2","F1"}.
  - `x_oc.variant_thetas(theta, abs_d, xs) -> dict`.
  - `x_oc.worst(point, xs) -> (power, false)`.
  - `x_oc.meets(power, false, xs) -> bool ndarray`.
  - `x_oc.pick(power, false, xs, fs, require_false) -> dict`, with keys `rule`, `index`, `p_set`, `q`, `K`, `F`, `power_by_k` and `false_by_k`.
  - `x_oc.table_at_f(power, false, xs, F) -> list`.
  - `x_oc.point_grid(theta, cal, root, n_rep, z, xs) -> dict`.
  - `x_oc.precheck(theta, z, xs, n_rep=None) -> dict`, with keys `passed`, `passing`, `calibration`, `power`, `false`, `point`, `best_power`, `records_target`, `at_f32`, `m_needed`, `axes`, `n_rep`, `seed` and `timing_s`.
  - `x_oc.design_spec(xs, d) -> XSpec` and `x_oc.record_tags(xs) -> list`.
  - `x_oc.point_records(theta, abs_d, target, z, xs, n_rep=None, cell=run_cell, log=None) -> dict`.
  - `x_oc.residual_compare(pilot, balanced, xs) -> dict`.
  - `x_rules`: `PASS`, `INVALID`, `STOP_REUSE`, `STOP_OC_UNREACHABLE`, `SENTENCES`, `sentence()`, `reuse_stop(why) -> dict`, `w_reuse_reasons(w_doc, keys, facts, oc_sha, xs) -> list`, `pair_fact_reasons(w_doc, keys, xs) -> list`, `abs_naive_d(w_doc, keys) -> ndarray`, `diagnosis_reasons(diag, xs) -> list`, `precheck_paren(pc) -> str` and `precheck_decision(pc) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_x_precheck.py`:

```python
"""The point-θ precheck (X.9.1.1): point values worst over g, the pass rule at every k, the best-power design and the
records target with their tie order, a failed calibration filled and named in the sentence, the verbatim STOP
sentence; the variants (X.4.4: V0 = W's pair_cov, V1 |d′| < 5, V2 aweights, the < 6 rule); the point records'
layout; P1-5's residual comparison; x_rules' reuse and diagnosis reasons."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import x_oc, x_rules
from flymon.brain.x_spec import SPEC as XS

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ND = np.array([0.08, -0.05, -0.32, 1.09, 1.43, 2.72, 4.13, 5.9, 6.0, 7.98, -9.89, 10.7, -12.88, 13.3, 14.5, -16.07])
PREFIX = ("점 θ 사전 점검(X.9.1.1)에서 파일럿 잡음의 점 추정으로도 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다 — p_set · "
          "q · K · F 어느 설계도 k = 4–8 모두에서 점 검정력 ≥ 0.80과 점 거짓 통과 ≤ 0.05를 함께 만족하지 않는다(")
SUFFIX = "). 부트스트랩 작동 특성은 계산하지 않았고, 주 세트를 쓰지 않고 멈춘다."


@pytest.fixture(scope="module")
def pilot():
    return w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))


@pytest.fixture(scope="module")
def theta(pilot):
    return w_oc.fit(pilot)


def test_variants(theta):
    pm = theta["pair_means"]
    assert np.array_equal(x_oc.pair_cov_variant(pm, ND, "V0", XS), theta["pair_cov"])
    x = pm.reshape(16, 4)
    assert np.allclose(x_oc.pair_cov_variant(pm, ND, "V1", XS), np.cov(x[np.abs(ND) < 5.0], rowvar=False))
    assert (np.abs(ND) < 5.0).sum() == 7
    assert np.allclose(x_oc.pair_cov_variant(pm, ND, "V2", XS), np.cov(x, rowvar=False, aweights=1 / (1 + np.abs(ND))))
    f1 = x_oc.pair_cov_variant(pm, ND, "F1", XS)                       # 3 pairs < 6: diagonal + V0's correlations
    v0 = theta["pair_cov"]
    r0 = v0 / np.outer(np.sqrt(np.diag(v0)), np.sqrt(np.diag(v0)))
    sd = np.sqrt(np.diag(np.cov(x[np.abs(ND) < 0.5], rowvar=False)))
    assert np.allclose(f1, r0 * np.outer(sd, sd))
    with pytest.raises(ValueError):
        x_oc.pair_cov_variant(pm, ND, "V9", XS)


def test_pick_rules_and_ties():
    shp = x_oc.grid_shape(XS)
    pw, fp = np.zeros(shp), np.ones(shp)
    pw[1, 2, 0, 3] = 0.9                                                 # best power, false fails
    pw[0, 0, 1, 5], fp[0, 0, 1, 5] = 0.5, 0.0                           # false ok
    fs = list(range(8, 33))
    assert x_oc.pick(pw, fp, XS, fs, False)["index"] == [1, 2, 0, 3]
    t = x_oc.pick(pw, fp, XS, fs, True)
    assert t["rule"] == "false_ok_max_power" and t["index"] == [0, 0, 1, 5]
    fp[0, 0, 1, 5] = 0.2
    fp[3, 1, 0, 0] = 0.1
    assert x_oc.pick(pw, fp, XS, fs, True)["index"] == [3, 1, 0, 0]  # none ok → smallest max_k false
    pw2 = np.full(shp, 0.5)                                            # all tied: larger p_set, larger q, small K, F
    assert x_oc.pick(pw2, np.zeros(shp), XS, fs, False)["index"] == [4, 2, 0, 0]


def test_precheck_layout_and_rule(theta):
    pc = x_oc.precheck(theta, Z, XS, n_rep=40)
    pw, fp = np.asarray(pc["power"]), np.asarray(pc["false"])
    pts = {k: np.asarray(v) for k, v in pc["point"].items()}
    assert np.array_equal(pw, np.min([pts[f"g{g}|min"] for g in XS.cluster_grid], 0))
    assert np.array_equal(fp, np.max([pts[f"g{g}|max"] for g in XS.cluster_grid], 0))
    ok = ((np.round(pw - 0.80, 9) >= 0) & (np.round(fp - 0.05, 9) <= 0)).all(-1)
    assert pc["passed"] == bool(ok.any()) and len(pc["passing"]) == int(ok.sum())
    assert len(pc["at_f32"]) == 5 * 3 * 2 and pc["at_f32"][0]["F"] == 32 and pc["seed"] == 43_200_000
    assert pc["best_power"]["rule"] == "max_power" and pc["records_target"]["rule"] != "max_power"


def test_precheck_pass_and_stop_decisions(theta):
    easy = dataclasses.replace(XS, p_power=0.0, p_false=1.0)
    assert x_rules.precheck_decision(x_oc.precheck(theta, Z, easy, n_rep=8))["outcome"] == x_rules.PASS
    hard = dataclasses.replace(XS, p_power=1.01)
    dec = x_rules.precheck_decision(x_oc.precheck(theta, Z, hard, n_rep=8))
    assert dec["outcome"] == x_rules.STOP_OC_UNREACHABLE
    s = dec["sentence"]
    assert s.startswith(PREFIX) and s.endswith(SUFFIX) and "점 검정력 최대 설계 p_set " in s
    assert s.count("·F 32: k 4 점 검정력 ") == 30


def test_precheck_calibration_failure_fills_and_sentence(theta):
    bad = dataclasses.replace(XS, d_power=40.0)
    pc = x_oc.precheck(theta, Z, bad, n_rep=8)
    assert not pc["calibration"]["min"]["ok"] and np.all(np.asarray(pc["power"]) == 0.0) and not pc["passed"]
    s = x_rules.precheck_decision(pc)["sentence"]
    assert s.startswith(PREFIX + "보정 불가 — min: a ") and "; max: a ok, b " in s and s.endswith(SUFFIX)


def test_point_records_layout(theta):
    target = dict(p_set=0.75, q=0.75, K=8, F=8)
    r = x_oc.point_records(theta, np.abs(ND), target, Z, XS, n_rep=6)
    assert set(r["variants"]) == {"V0", "V1", "V2"} and r["seed"] == 43_200_000
    e = r["variants"]["V1"]["g0.5"]
    assert set(e["true_dprime"]) == {"0.5", "1.0", "1.5", "2.0"} and len(e["true_dprime"]["1.5"]) == 5
    assert set(e["heterogeneous_w"]) == {"one_pair_0.5", "half_pairs_0.5", "all_1.0"}
    assert set(e["pair_cov_scaled"]) == {f"{s}|{m}" for s in (0.5, 1.0, 2.0) for m in ("power", "false")}
    assert len(e["mixed_flies"]) == 5 and len(r["tags"]) == 4 + 4 + 6


def test_residual_compare(pilot):
    out = x_oc.residual_compare(pilot, np.abs(ND) < 0.5, XS)
    assert out["n_pairs"] == dict(balanced=3, rest=13)
    g = out["groups"]["balanced"]
    assert len(g) == 6 * 2 * 2 and set(g["R1|MBON05|X"]) == {"sd", "quantiles", "zero_share", "n"}
    assert g["R1|MBON05|X"]["n"] == 3 * 8 * 8 and list(g["pre|MBON13|Y"]["quantiles"]) == ["5.0", "25.0", "50.0",
                                                                                          "75.0", "95.0"]


def _w_doc(keys):
    return dict(stage0=dict(decision_files={"flymon/brain/w_measure.py": XS.w_measure_file_sha}),
                pilot=dict(outcome="PASS", w_measure_key=XS.w_measure_key, pairs=list(keys), manifest=[{}] * 384,
                           record=dict(pairs={k: dict(naive_d=float(d)) for k, d in zip(keys, ND)})),
                oc=dict(w_measure_key=XS.w_measure_key, detail_sha256=XS.w_oc_detail_sha))


def test_reuse_reasons():
    keys = list(XS.balanced_pairs) + [f"p{j}" for j in range(13)]
    doc = _w_doc(keys)
    k = dict(w_measure_key=XS.w_measure_key, pipeline_key=XS.w_pipeline_key, w_measure_sha=XS.w_measure_file_sha)
    facts = dict(last="f30ae35" + "0" * 33, ancestors={"5fbc4c8": True, "f30ae35": True},
                 git=dict(tracked=True, dirty=False))
    assert x_rules.w_reuse_reasons(doc, k, facts, XS.w_oc_detail_sha, XS) == []
    assert x_rules.pair_fact_reasons(doc, keys, XS) == []
    why = x_rules.w_reuse_reasons(doc, dict(k, w_measure_key="z" * 64), dict(facts, ancestors={}), "y" * 64, XS)
    assert any(w.startswith("W 측정 키 ") for w in why) and any("5fbc4c8" in w for w in why)
    assert any("results/w/oc.json" in w for w in why)
    s = x_rules.reuse_stop(why)
    assert s["outcome"] == x_rules.STOP_REUSE and s["records_unavailable"] is True
    assert s["sentence"].startswith("X 재사용 조건(X.5 1 · X.4.3 진단)이 깨졌다(") and s["sentence"].endswith(
        "X는 W 파일럿·경로와 V 블록을 다시 재는 경로를 갖지 않으므로 주 세트를 쓰지 않고 멈춘다 — 사용자 몫.")


def test_diagnosis_reasons():
    ok = dict(diffs=[], n_draws=200, counts=dict(total=29, power=14, false=18, both=3))
    assert x_rules.diagnosis_reasons(ok, XS) == []
    assert x_rules.diagnosis_reasons(dict(ok, counts=dict(total=28, power=14, false=17, both=3)), XS)
    assert x_rules.diagnosis_reasons(dict(ok, diffs=[dict(draw=4, side="min")]), XS)
    assert x_rules.diagnosis_reasons(dict(ok, n_draws=3, counts=dict(total=0, power=0, false=0, both=0)), XS) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/brain/test_x_precheck.py -q > /private/tmp/claude-503/x-tests/t3.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t3.log`
Expected: `AttributeError: module 'flymon.brain.x_oc' has no attribute 'pair_cov_variant'` and `ModuleNotFoundError ... x_rules`.

- [ ] **Step 3: Append the precheck section to `flymon/brain/x_oc.py`**

```python
# ================================================================ variants (X.4.4) and the F1 filter's Σ (X.9.1.4)
def pair_cov_variant(pair_means, abs_d, name: str, xs) -> np.ndarray:
    """Σ_pair from the pilot pair means [J, 2, 2]: V0 all pairs (= w_oc's pair_cov), V1 |d′| < v1_exclude_from, V2
    weights 1 / (1 + |d′|) (numpy aweights), F1 the balanced pairs |d′| < naive_max. A variant on fewer than
    min_variant_pairs pairs keeps only its diagonal, with V0's correlations (Σ = D^½ R_V0 D^½)."""
    x = np.asarray(pair_means, float).reshape(len(pair_means), -1)
    d = np.abs(np.asarray(abs_d, float))
    v0 = np.cov(x, rowvar=False)
    if name == "V0":
        return v0
    if name == "V2":
        return np.cov(x, rowvar=False, aweights=1.0 / (1.0 + d))
    if name not in ("V1", "F1"):
        raise ValueError(f"unknown variant {name}")
    keep = d < (xs.v1_exclude_from if name == "V1" else xs.naive_max)
    c = np.cov(x[keep], rowvar=False)
    if keep.sum() >= xs.min_variant_pairs:
        return c
    s0 = np.sqrt(np.diag(v0))
    return (v0 / np.outer(s0, s0)) * np.outer(np.sqrt(np.diag(c)), np.sqrt(np.diag(c)))


def variant_thetas(theta, abs_d, xs) -> dict:
    return {v: dict(theta, pair_cov=pair_cov_variant(theta["pair_means"], abs_d, v, xs)) for v in xs.variants}


# ================================================================ the point-θ precheck (X.9.1.1)
def point_grid(theta, cal, root, n_rep, z, xs) -> dict:
    """{(g, mode): P(PASS) [p, q, K, F, k]} at θ̂ on w_oc.simulate; tag (TAG_POINT, g index, target); a failed
    calibration fills (X.9.1.2)."""
    out = {}
    for gi, g in enumerate(xs.cluster_grid):
        for m, _f in MODES:
            c = cal[m]
            out[(g, m)] = (evaluate(theta, rng(root, TAG_POINT, gi, int(m == "max")), n_rep,
                                    *w_oc._uniform(xs, c["a"]["value"], c["b"]["value"]), g, z, xs)
                           if c["ok"] else np.full(grid_shape(xs), c["fill"]))
    return out


def worst(point, xs) -> tuple:
    return (np.min([point[(g, "min")] for g in xs.cluster_grid], axis=0),
            np.max([point[(g, "max")] for g in xs.cluster_grid], axis=0))


def meets(power, false, xs) -> np.ndarray:
    d = xs.round_digits
    return (np.round(np.asarray(power) - xs.p_power, d) >= 0) & (np.round(np.asarray(false) - xs.p_false, d) <= 0)


def pick(power, false, xs, fs, require_false: bool) -> dict:
    """power / false [p, q, K, F, k]. require_false False: the largest min_k power ("점 검정력 최대 설계"). True:
    among designs whose max_k false ≤ p_false the largest min_k power, none → the smallest max_k false (then
    power). Ties: larger p_set, larger q, smaller K, smaller F (X.9.1.1; no cost in phase A)."""
    power, false = np.asarray(power), np.asarray(false)
    d = xs.round_digits
    pmin, fmax = np.round(power.min(-1), d), np.round(false.max(-1), d)
    ok = np.round(false.max(-1) - xs.p_false, d) <= 0
    cells = list(np.ndindex(pmin.shape))

    def tie(i):
        return (-xs.p_set_grid[i[0]], -xs.q_grid[i[1]], xs.k_grid[i[2]], fs[i[3]])
    if not require_false:
        rule, i = "max_power", min(cells, key=lambda i: (-pmin[i],) + tie(i))
    elif ok.any():
        rule, i = "false_ok_max_power", min((c for c in cells if ok[c]), key=lambda i: (-pmin[i],) + tie(i))
    else:
        rule, i = "min_false", min(cells, key=lambda i: (fmax[i], -pmin[i]) + tie(i))
    return dict(rule=rule, index=[int(v) for v in i], p_set=xs.p_set_grid[i[0]], q=xs.q_grid[i[1]],
                K=xs.k_grid[i[2]], F=int(fs[i[3]]), power_by_k=power[i].tolist(), false_by_k=false[i].tolist())


def table_at_f(power, false, xs, F: int) -> list:
    fi = F - xs.f_min
    ks = list(range(xs.k_min, xs.k_cap + 1))
    return [dict(p_set=p, q=q, K=K, F=F, k=ks, power=np.asarray(power)[pi, qi, ki, fi].tolist(),
                 false=np.asarray(false)[pi, qi, ki, fi].tolist())
            for pi, p in enumerate(xs.p_set_grid) for qi, q in enumerate(xs.q_grid) for ki, K in enumerate(xs.k_grid)]


def precheck(theta, z, xs, n_rep: int | None = None) -> dict:
    """X.9.1.1 on θ̂ (V0): calibration per status, P(PASS) for every design and k at every g, point power = g min,
    point false pass = g max; passed iff some design meets both targets at every k."""
    t0 = time.perf_counter()
    n_rep = xs.precheck_reps if n_rep is None else int(n_rep)
    root = xs.precheck_seed
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    cal = {m: calibrate_x(theta, getattr(xs, f), m, idx, z, xs) for m, f in MODES}
    point = point_grid(theta, cal, root, n_rep, z, xs)
    power, false = worst(point, xs)
    fs = list(range(xs.f_min, xs.f_max + 1))
    ok = meets(power, false, xs).all(-1)
    passing = [dict(p_set=xs.p_set_grid[i[0]], q=xs.q_grid[i[1]], K=xs.k_grid[i[2]], F=fs[i[3]])
               for i in np.ndindex(ok.shape) if ok[i]]
    return dict(passed=bool(ok.any()), passing=passing, calibration=cal, power=power.tolist(), false=false.tolist(),
                point={f"g{g}|{m}": p.tolist() for (g, m), p in point.items()},
                best_power=pick(power, false, xs, fs, False), records_target=pick(power, false, xs, fs, True),
                at_f32=table_at_f(power, false, xs, xs.f_max), m_needed=XV.m_needed_table(xs),
                axes=dict(p_set=list(xs.p_set_grid), q=list(xs.q_grid), K=list(xs.k_grid), F=fs,
                          k=list(range(xs.k_min, xs.k_cap + 1)), g=list(xs.cluster_grid)),
                n_rep=n_rep, seed=int(root), timing_s=time.perf_counter() - t0)


# ================================================================ records at θ̂ (X.4.5 via X.9.1.1, precheck STOP only)
def design_spec(xs, d: dict):
    """The spec of one design (p_set, q, K, F): evaluate on it simulates F flies and 2K probes."""
    return dataclasses.replace(xs, p_set_grid=(d["p_set"],), q_grid=(d["q"],), k_grid=(d["K"],), f_min=d["F"],
                               f_max=d["F"])


def record_tags(xs) -> list:
    return ([f"t{t}" for t in xs.record_dprimes] + ["one", "half", "all1", "mixed"]
            + [f"het{s}|{m}" for m in ("p", "f") for s in xs.het_scales])


def point_records(theta, abs_d, target: dict, z, xs, n_rep: int | None = None, cell=run_cell, log=None) -> dict:
    """For the target design, P(PASS) by k on every variant (V0 · V1 · V2, records only — P3-12) × cluster level:
    the homogeneous true d′ 0.5 / 1.0 / 1.5 / 2.0 (power-style calibration), W's heterogeneous scenarios (one pair
    0.5 + rest 1.5, every other pair 0.5, all 1.0), the mixed flies (every other fly 0 at 1.5), and Σ_v × het_scales
    at 1.5 (power calibration) and 0.5 (false-pass calibration). Streams (TAG_RECORD, variant, g, scenario). A failed
    calibration → None with its status in `calibrations`."""
    n_rep = xs.oc_reps if n_rep is None else int(n_rep)
    root = xs.precheck_seed
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    xd = design_spec(xs, target)
    cmin = {t: calibrate_x(theta, t, "min", idx, z, xs) for t in sorted(set(xs.record_dprimes) | {xs.d_power})}
    cmax = calibrate_x(theta, xs.d_false, "max", idx, z, xs)
    tags = record_tags(xs)
    thetas = variant_thetas(theta, abs_d, xs)
    K_, F_ = xs.k_cap, xd.f_max
    ones = [1.0] * F_
    half = [1.0 if i % 2 == 0 else 0.0 for i in range(F_)]

    def ab(c):
        return (c["a"]["value"], c["b"]["value"]) if c["ok"] else None

    def one(vi, v):
        th, out = thetas[v], {}
        for gi, g in enumerate(xs.cluster_grid):
            def at(th_, a_p, b_p, fa, fb, tag):
                r = rng(root, TAG_RECORD, vi, gi, tags.index(tag))
                return evaluate(th_, r, n_rep, a_p, b_p, fa, fb, g, z, xd)[0, 0, 0, 0].tolist()
            e = dict(true_dprime={str(t): (at(th, *w_oc._uniform(xd, *ab(cmin[t])), f"t{t}") if cmin[t]["ok"] else None)
                                  for t in xs.record_dprimes})
            lo, hi, mid = ab(cmin[xs.het_low_dprime]), ab(cmin[xs.d_power]), ab(cmin[xs.het_all_dprime])
            het = {}
            if lo and hi:
                het[f"one_pair_{xs.het_low_dprime}"] = at(th, [lo[0]] + [hi[0]] * (K_ - 1),
                                                          [lo[1]] + [hi[1]] * (K_ - 1), ones, ones, "one")
                het[f"half_pairs_{xs.het_low_dprime}"] = at(th, [lo[0] if i % 2 else hi[0] for i in range(K_)],
                                                            [lo[1] if i % 2 else hi[1] for i in range(K_)], ones, ones,
                                                            "half")
            if mid:
                het[f"all_{xs.het_all_dprime}"] = at(th, *w_oc._uniform(xd, *mid), "all1")
            e["heterogeneous_w"] = het
            e["mixed_flies"] = at(th, [hi[0]] * K_, [hi[1]] * K_, half, half, "mixed") if hi else None
            fa = ab(cmax)
            e["pair_cov_scaled"] = {}
            for s in xs.het_scales:
                ts = dict(th, pair_cov=th["pair_cov"] * s)
                e["pair_cov_scaled"][f"{s}|power"] = at(ts, *w_oc._uniform(xd, *hi), f"het{s}|p") if hi else None
                e["pair_cov_scaled"][f"{s}|false"] = at(ts, *w_oc._uniform(xd, *fa), f"het{s}|f") if fa else None
            out[f"g{g}"] = e
        if log is not None:
            log(f"x precheck records {v} done")
        return out
    key = dict(theta=w_oc.summary(theta), target=target, n_rep=n_rep)
    res = {v: cell(f"records_{v}", dict(key, variant=v), lambda vi=vi, v=v: one(vi, v))
           for vi, v in enumerate(xs.variants)}
    cals = {f"min|{t}": dict(ok=c["ok"], failure=c["failure"], retried=c["retried"]) for t, c in cmin.items()}
    cals[f"max|{xs.d_false}"] = dict(ok=cmax["ok"], failure=cmax["failure"], retried=cmax["retried"])
    return dict(target=target, variants=res, calibrations=cals, n_rep=n_rep, seed=int(root), tags=tags,
                note="사전 점검 STOP의 점 records(X.9.1.1) — 변형은 기록 전용(X.9.1.3 P3-12)")


# ================================================================ P1-5 residual comparison (records only)
def residual_compare(pilot: list, balanced, xs) -> dict:
    """Balanced pilot pairs vs the rest: per slot × cell × odour, the sd and quantiles of w_oc.fit's √(K/(K−1))
    residuals and the raw counts' zero share (the floor share)."""
    th = w_oc.fit(pilot)
    raw = w_oc._stack(pilot)
    J = raw.shape[0]
    res = np.stack(th["resid_blocks"])
    raw = raw.reshape(J, -1, *raw.shape[-3:])
    bal = np.asarray(balanced, bool)
    out = {}
    for name, mask in (("balanced", bal), ("rest", ~bal)):
        r, c = res[mask].reshape(-1, *res.shape[-3:]), raw[mask].reshape(-1, *raw.shape[-3:])
        grp = {}
        for si, s in enumerate(w_oc.SLOTS):
            for ci, cell in enumerate(("MBON13", "MBON05")):
                for oi, od in enumerate(("X", "Y")):
                    v = r[:, si, ci, oi]
                    grp[f"{s}|{cell}|{od}"] = dict(
                        sd=float(v.std(ddof=1)), n=int(len(v)), zero_share=float((c[:, si, ci, oi] == 0).mean()),
                        quantiles={str(q): float(x) for q, x in zip(xs.resid_quantiles,
                                                                    np.percentile(v, xs.resid_quantiles))})
        out[name] = grp
    return dict(groups=out, n_pairs=dict(balanced=int(bal.sum()), rest=int((~bal).sum())),
                note="판정·관문에 쓰지 않는다(X.9.1.3 P1-5); '균형 쌍 전용 파일럿'은 다음 결정 목록")
```

- [ ] **Step 4: Write `flymon/brain/x_rules.py`**

```python
"""X's gate decisions and sentences for phase A (X.6, X.9.1.1, X.9.1.3 P2-10; Reading 1): the W reuse facts the
precheck rests on, the W diagnosis check, STOP_REUSE and the point-θ precheck. Phase B adds its gates and the X.3
sentences here. Every number comes from the XSpec passed in."""
from __future__ import annotations

import numpy as np

PASS, INVALID = "PASS", "INVALID"
STOP_REUSE, STOP_OC_UNREACHABLE = "STOP_REUSE", "STOP_OC_UNREACHABLE"

SENTENCES = {
    STOP_REUSE: "X 재사용 조건(X.5 1 · X.4.3 진단)이 깨졌다({why}). X는 W 파일럿·경로와 V 블록을 다시 재는 경로를 갖지 "
                "않으므로 주 세트를 쓰지 않고 멈춘다 — 사용자 몫.",
    STOP_OC_UNREACHABLE: "점 θ 사전 점검(X.9.1.1)에서 파일럿 잡음의 점 추정으로도 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 "
                         "없다 — p_set · q · K · F 어느 설계도 k = 4–8 모두에서 점 검정력 ≥ 0.80과 점 거짓 통과 ≤ 0.05를 "
                         "함께 만족하지 않는다({paren}). 부트스트랩 작동 특성은 계산하지 않았고, 주 세트를 쓰지 않고 "
                         "멈춘다.",
}


def sentence(outcome: str, fields: dict) -> str:
    return SENTENCES[outcome].format(**fields)


def reuse_stop(why: list) -> dict:
    return dict(outcome=STOP_REUSE, reasons=list(why), sentence=sentence(STOP_REUSE, dict(why="; ".join(why))),
                records_unavailable=True, records_reason="OC 전 관문 STOP(X.9.1.3 P2-10)")


def w_reuse_reasons(w_doc: dict, keys: dict, facts: dict, oc_sha, xs) -> list:
    """The W facts the precheck rests on (Reading 2)."""
    why = []
    if keys.get("w_measure_key") != xs.w_measure_key:
        why.append(f"W 측정 키 {keys.get('w_measure_key')} ≠ {xs.w_measure_key}")
    if keys.get("pipeline_key") != xs.w_pipeline_key:
        why.append(f"W 파이프라인 키 {keys.get('pipeline_key')} ≠ {xs.w_pipeline_key}")
    st0 = ((w_doc.get("stage0") or {}).get("decision_files") or {}).get("flymon/brain/w_measure.py")
    if keys.get("w_measure_sha") != xs.w_measure_file_sha or st0 != xs.w_measure_file_sha:
        why.append(f"w_measure.py sha256 {keys.get('w_measure_sha')} / W stage0 {st0} ≠ {xs.w_measure_file_sha}")
    g = facts.get("git") or {}
    if not g.get("tracked") or g.get("dirty"):
        why.append(f"{xs.w_summary} 미커밋")
    if not (facts.get("last") or "").startswith(xs.w_summary_commit):
        why.append(f"{xs.w_summary}의 마지막 커밋 {facts.get('last')} ≠ {xs.w_summary_commit}")
    for b, c in xs.w_commits:
        if not (facts.get("ancestors") or {}).get(c):
            why.append(f"W 커밋 {c}({b})가 HEAD 이력에 없음")
        blk = w_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"W 블록 {b} 없음")
        elif blk.get("w_measure_key") != xs.w_measure_key:
            why.append(f"W 블록 {b}의 W 측정 키 {blk.get('w_measure_key')}")
    p = w_doc.get("pilot") or {}
    if p.get("outcome") != PASS:
        why.append(f"W 파일럿 블록 {p.get('outcome')}")
    if len(p.get("manifest") or []) != xs.w_pilot_units or len(p.get("pairs") or []) != xs.w_pilot_pairs:
        why.append(f"W 파일럿 매니페스트 {len(p.get('manifest') or [])} / 쌍 {len(p.get('pairs') or [])}")
    if (w_doc.get("oc") or {}).get("detail_sha256") != xs.w_oc_detail_sha or oc_sha != xs.w_oc_detail_sha:
        why.append(f"{xs.w_oc_detail} sha256 {oc_sha} ≠ {xs.w_oc_detail_sha}")
    return why


def _naive_d(w_doc: dict, keys) -> dict:
    return {k: float(w_doc["pilot"]["record"]["pairs"][k]["naive_d"]) for k in keys}


def abs_naive_d(w_doc: dict, keys) -> np.ndarray:
    nd = _naive_d(w_doc, keys)
    return np.array([abs(nd[k]) for k in keys])


def pair_fact_reasons(w_doc: dict, keys, xs) -> list:
    """X.9.1.3: the balanced pilot pairs (|naive_d| < naive_max) are the declared three; V1 keeps v1_pairs."""
    nd = _naive_d(w_doc, keys)
    why = []
    bal = sorted(k for k in keys if abs(nd[k]) < xs.naive_max)
    if bal != sorted(xs.balanced_pairs):
        why.append(f"순진 균형 파일럿 쌍 {bal} ≠ {sorted(xs.balanced_pairs)}")
    n1 = sum(abs(v) < xs.v1_exclude_from for v in nd.values())
    if n1 != xs.v1_pairs:
        why.append(f"V1 잔존 쌍 {n1} ≠ {xs.v1_pairs}")
    return why


def diagnosis_reasons(diag: dict, xs) -> list:
    """X.4.3: the reproduction must match W draw for draw, and over all draws give W's counts."""
    why = []
    if diag["diffs"]:
        d0 = diag["diffs"][0]
        why.append(f"W 보정 진단 재현 불일치 {len(diag['diffs'])}건(첫: 추출 {d0['draw']} {d0['side']})")
    c = diag["counts"]
    want = (xs.w_boot_fail_total, xs.w_boot_fail_power, xs.w_boot_fail_false, xs.w_boot_fail_both)
    got = (c["total"], c["power"], c["false"], c["both"])
    if diag["n_draws"] == xs.boot_draws and got != want:
        why.append(f"W 보정 실패 수 {got} ≠ {want}")
    return why


def precheck_paren(pc: dict) -> str:
    cal = pc["calibration"]
    if not (cal["min"]["ok"] and cal["max"]["ok"]):
        def st(m, h):
            return (cal[m].get(h) or {}).get("status")
        return "보정 불가 — " + "; ".join(f"{m}: a {st(m, 'a')}, b {st(m, 'b')}" for m in ("min", "max"))
    rows = "; ".join(f"p_set {r['p_set']}·q {r['q']}·K {r['K']}·F {r['F']}: " + ", ".join(
        f"k {k} 점 검정력 {pw:.3f} / 점 거짓 통과 {fp:.3f}" for k, pw, fp in zip(r["k"], r["power"], r["false"]))
        for r in pc["at_f32"])
    b = pc["best_power"]
    best = f"p_set {b['p_set']} · q {b['q']} · K {b['K']} · F {b['F']}"
    bp = ", ".join(f"k {k} {v:.3f}" for k, v in zip(pc["axes"]["k"], b["power_by_k"]))
    return f"{rows}; 점 검정력 최대 설계 {best}의 k = 4–8 점 검정력 {bp}"


def precheck_decision(pc: dict) -> dict:
    if pc["passed"]:
        return dict(outcome=PASS, reasons=[], n_passing=len(pc["passing"]))
    return dict(outcome=STOP_OC_UNREACHABLE, reasons=["점 θ 사전 점검 미달(X.9.1.1)"],
                sentence=sentence(STOP_OC_UNREACHABLE, dict(paren=precheck_paren(pc))))
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/brain/test_x_precheck.py tests/brain/test_x_oc.py tests/brain/test_x_spec.py -q > /private/tmp/claude-503/x-tests/t3.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t3.log`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/x_oc.py flymon/brain/x_rules.py tests/brain/test_x_precheck.py
git commit -m "feat(x): point-theta precheck (worst g, every k, best-power and records-target picks), STOP sentence verbatim, point records V0/V1/V2, P1-5 residual comparison, x_rules reuse/diagnosis reasons"
```

---

### Task 4: record-only diagnostics (i)–(iii)

**Files:**
- Modify: `flymon/brain/x_oc.py` (append a section; nothing above changes)
- Test: `tests/brain/test_x_diag.py`

**Interfaces:**
- Consumes: Task 2's `rng`, `calibrate_x`, `simulate`, `pair_bases`, `tally`, `run_cell` and `MODES`. Task 3's `pair_cov_variant`, `meets` and `pick`.
- Produces:
  - Constants `TAG_DIAG_GRID = 6`, `TAG_DIAG_PAIR = 7`, `TAG_DIAG_ACCEPT = 8`, `GATE_GROUP`, `ASSIGN` and `STAGE`.
  - `pair_true_dprimes(theta, bases, a, b, idx, z, xs) -> (td [N, 4], floor [N, 2])`.
  - `evaluate_grid(theta, rng_, n_rep, a, b, g, z, xs, accept=None) -> dict(p, fill_share)`.
  - `filters(theta, abs_d, xs) -> {name: dict(theta, accept, pass_counts)}`.
  - `pair_diag(theta, cal, idx, root, n_rep, z, xs, cell=run_cell, log=None) -> dict`.
  - `min_change(worst_by_filter, xs, fs, kk) -> dict`.
  - `diagnostics(theta, abs_d, z, xs, n_rep=None, cell=run_cell, log=None) -> dict`.
  - `diag_summary(diag) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_x_diag.py`:

```python
"""X.9.1.4's record-only diagnostics: per-pair true d′ = w_oc.true_dprimes on that pair's base, the floor flags at
0.10, the filters (F1 Σ from the balanced pairs, F2(ℓ) percentile thresholds on MBON13(X) / MBON05(X)), rejection that
cannot be met fills and is counted, the pair-level counters add up, the minimum-change ranking order, the full run's
layout and that it never changes the gate."""
import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain import x_oc
from flymon.brain.x_spec import SPEC as XS

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ND = np.array([0.08, -0.05, -0.32, 1.09, 1.43, 2.72, 4.13, 5.9, 6.0, 7.98, -9.89, 10.7, -12.88, 13.3, 14.5, -16.07])


@pytest.fixture(scope="module")
def theta():
    return w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                         learn=(20.0, 10.0)))


@pytest.fixture(scope="module")
def cal(theta):
    idx = x_oc.rng(XS.diag_seed, x_oc.TAG_CAL).integers(0, len(theta["resid"]), XS.cal_reps)
    return idx, x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)


def test_pair_true_dprimes_equal_w(theta, cal):
    idx, c = cal
    a, b = c["a"]["value"], c["b"]["value"]
    bases, _ = x_oc.pair_bases(theta, np.random.default_rng(4), 3, 5, 1.0, Z, XS, XS.w_rejection_tries)
    bases = bases.reshape(-1, 2, 2)
    td, fl = x_oc.pair_true_dprimes(theta, bases, a, b, idx, Z, XS)
    for i, bs in enumerate(bases):
        assert np.allclose(td[i], w_oc.true_dprimes(dict(theta, m0s=bs), a, b, idx, Z), rtol=0, atol=1e-12)
        mu = w_oc.slot_means(theta, bs, np.zeros((2, 2)), a, b)
        r = theta["resid"][idx]
        low_r = (np.rint(mu[w_oc.SI["R1"], WV.P, WV.X] + r[:, w_oc.SI["R1"], WV.P, WV.X]) < 0).mean()
        low_p = (np.rint(mu[w_oc.SI["R2"], WV.A, WV.X] + r[:, w_oc.SI["R2"], WV.A, WV.X]) < 0).mean()
        assert fl[i].tolist() == [bool(low_r >= 0.10), bool(low_p >= 0.10)]


def test_pair_bases_reject_all_fills(theta):
    never = lambda b: np.zeros(b.shape[:-2], bool)          # noqa: E731
    bases, filled = x_oc.pair_bases(theta, np.random.default_rng(0), 4, 3, 0.0, Z, XS, 5, never)
    assert filled.all() and np.allclose(bases, theta["m0s"])
    r = x_oc.evaluate_grid(theta, np.random.default_rng(1), 2, 0.0, 0.0, 0.0, Z, XS, never)
    assert r["fill_share"] == 1.0 and r["p"].shape == (5, 3, 2, 57, 13)


def test_filters(theta):
    f = x_oc.filters(theta, ND, XS)
    assert list(f) == ["none", "F1", "F2(0)", "F2(25)", "F2(50)", "F2(75)"]
    assert f["none"]["accept"] is None and f["F1"]["accept"] is None
    assert np.allclose(f["F1"]["theta"]["pair_cov"], x_oc.pair_cov_variant(theta["pair_means"], ND, "F1", XS))
    pm = theta["pair_means"]
    c = f["F2(50)"]["pass_counts"]
    assert c["c_A"] == np.percentile(pm[:, WV.A, WV.X], 50.0) and c["c_P"] == np.percentile(pm[:, WV.P, WV.X], 50.0)
    keep = (pm[:, WV.A, WV.X] >= c["c_A"]) & (pm[:, WV.P, WV.X] >= c["c_P"])
    assert c["pilot"] == int(keep.sum()) and c["balanced"] == int((keep & (np.abs(ND) < 0.5)).sum())
    assert f["F1"]["pass_counts"] == dict(pilot=3, balanced=3)


def test_pair_diag_counts_add_up(theta, cal):
    idx, c = cal
    out = x_oc.pair_diag(theta, c, idx, XS.diag_seed, 2, Z, XS)
    for g in XS.cluster_grid:
        cells = out[f"g{g}"]["cells"]
        assert set(cells) == {f"{q}|{K}|{F}" for q in XS.q_grid for K in XS.k_grid for F in XS.diag_fs}
        for v in cells.values():
            n = v["counts"]
            assert n["n"] == 2 * 16 and n["n_pass"] + n["n_fail"] == n["n"]
            assert sum(sum(r) for r in n["table"]) == n["n_fail"]
            assert n["mech"] + n["joint_only"] <= n["n_fail"]
    assert set(out["worst_g"]) == set(out["g0.0"]["cells"])


def test_min_change_order():
    fs, kk = list(range(8, 65)), list(range(4, 17))
    shp = (5, 3, 2, len(fs), len(kk))
    zero, one = np.zeros(shp), np.ones(shp)
    pw = zero.copy()
    pw[:, :, :, fs.index(40):, :] = 0.9                       # F ≥ 40 passes for "none" at every k
    pw2 = zero.copy()
    pw2[:, :, :, fs.index(20):, kk.index(4):kk.index(8) + 1] = 0.9    # F1: F 20 at k 4-8
    out = x_oc.min_change({"none": (pw, zero), "F1": (pw2, zero), "F2(0)": (zero, one)}, XS, fs, kk)
    r1 = out["rank1"]                                         # knobs: F1 (filter) = 1, none at F 40 (F > 32) = 1
    assert (r1["filter"], r1["k_lo"], r1["F"], r1["knobs"]) == ("F1", 4, 20, 1)
    assert [(e["filter"], e["k_lo"], e["F"]) for e in out["same_knobs"]] == [("F1", 4, 20), ("none", 4, 40)]
    assert r1["p_set"] == 1.0 and r1["q"] == 0.75 and r1["K"] == 8      # ties: larger p_set, larger q, smaller K
    none6 = [e for e in out["entries"] if e["filter"] == "none" and e["k_lo"] == 6][0]
    assert none6["knobs"] == 2 and not [e for e in out["entries"] if e["filter"] == "F2(0)" and e["found"]]


def test_diagnostics_layout(theta):
    d = x_oc.diagnostics(theta, ND, Z, XS, n_rep=2)
    assert set(d) >= {"calibration", "filters", "ii", "iii", "i", "arrays", "axes", "seed", "n_rep", "timing_s"}
    assert d["seed"] == 43_300_000 and list(d["ii"]) == list(d["filters"])
    assert len(d["iii"]["entries"]) == 6 * 5
    ii = d["ii"]["F2(25)"]
    assert set(ii) >= {"best", "best_g0", "fill", "fill_flag", "pass_counts", "acceptance"}
    assert 0.0 <= ii["acceptance"] <= 1.0 and d["ii"]["none"]["acceptance"] is None
    s = x_oc.diag_summary(d)
    assert set(s) == {"ii", "iii", "i", "calibration", "note"}
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/brain/test_x_diag.py -q > /private/tmp/claude-503/x-tests/t4.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t4.log`
Expected: `AttributeError: module 'flymon.brain.x_oc' has no attribute 'pair_true_dprimes'`.

- [ ] **Step 3: Append the diagnostics section to `flymon/brain/x_oc.py`**

```python
# ================================================================ record-only diagnostics (X.9.1.4)
TAG_DIAG_GRID, TAG_DIAG_PAIR, TAG_DIAG_ACCEPT = 6, 7, 8
GATE_GROUP = np.array([0, 1, 0, 1])                    # WV.GATES → 0 reward stage, 1 punishment stage
ASSIGN = ("기계 대조", "순진(바닥)", "순진(쌍 효과·균형)", "마리·프로브 잡음")
STAGE = ("없음", "보상", "처벌", "둘 다")


def _stage_idx(rew, pun):
    return np.asarray(rew, int) + 2 * np.asarray(pun, int)


def pair_true_dprimes(theta, bases, a, b, idx, z, xs) -> tuple:
    """(td [N, 4], floor [N, 2]) for naive bases [N, 2, 2] (fly effect 0), the calibrated a, b and the calibration
    draws idx: td = the four gates' directional true d′ with that base in place of the population pair (w_oc.
    true_dprimes per base); floor = (reward, punishment) "caught" — the share of draws with rint(μ + r) < 0 at
    MBON05(X)·R1 (reward) / MBON13(X)·R2 (punishment) ≥ diag_floor_share."""
    r = theta["resid"][idx]
    s1, s2 = w_oc.SI["R1"], w_oc.SI["R2"]
    td, fl = [], []
    for s in range(0, len(bases), xs.diag_pair_chunk):
        mu = w_oc.slot_means(theta, np.asarray(bases[s:s + xs.diag_pair_chunk], float), np.zeros((2, 2)), a, b)
        c = np.clip(np.rint(mu[:, None] + r[None]), 0, None)
        td.append(WV.SIGNS * WV.gate_stats(w_oc.to_stages(c), z))
        low_r = (np.rint(mu[:, None, s1, WV.P, WV.X] + r[None, :, s1, WV.P, WV.X]) < 0).mean(1)
        low_p = (np.rint(mu[:, None, s2, WV.A, WV.X] + r[None, :, s2, WV.A, WV.X]) < 0).mean(1)
        fl.append(np.round(np.stack([low_r, low_p], -1) - xs.diag_floor_share, xs.round_digits) >= 0)
    return np.concatenate(td), np.concatenate(fl)


def evaluate_grid(theta, rng_, n_rep, a, b, g, z, xs, accept=None) -> dict:
    """P(PASS) [p_set, q, K, F f_min..diag_f_max, k k_min..diag_k_max] with diag_pairs pairs × diag_f_max flies
    (nested in k and F), X's set rule, diag_tries rejection rounds and the filter accept; the fill share."""
    P, Fm = xs.diag_pairs, xs.diag_f_max
    fs, kk = list(range(xs.f_min, Fm + 1)), list(range(xs.k_min, xs.diag_k_max + 1))
    hits = np.zeros((len(xs.p_set_grid), len(xs.q_grid), len(xs.k_grid), len(fs), len(kk)))
    filled, done = 0, 0
    while done < n_rep:
        n = min(xs.diag_chunk, n_rep - done)
        d, _base, fill = simulate(theta, rng_, n, P, Fm, 2 * max(xs.k_grid), [a] * P, [b] * P, [1.0] * Fm, [1.0] * Fm,
                                  g, z, xs, xs.diag_tries, accept)
        hits += tally(d, z, xs, xs.q_grid, xs.k_grid, fs, kk)
        filled += int(fill.sum())
        done += n
    return dict(p=hits / n_rep, fill_share=filled / (n_rep * P))


def filters(theta, abs_d, xs) -> dict:
    """X.9.1.4 (ii): none (V0), F1 (Σ_pair from the balanced pilot pairs, the < 6 rule), F2(ℓ) (base[MBON13, X] ≥
    the ℓ-th percentile of the pilot pair means' MBON13(X) ∧ base[MBON05, X] ≥ MBON05(X)'s)."""
    pm = theta["pair_means"]
    d = np.abs(np.asarray(abs_d, float))
    bal = d < xs.naive_max
    out = {"none": dict(theta=theta, accept=None, pass_counts=dict(pilot=int(len(pm)), balanced=int(bal.sum()))),
           "F1": dict(theta=dict(theta, pair_cov=pair_cov_variant(pm, d, "F1", xs)), accept=None,
                      pass_counts=dict(pilot=int(bal.sum()), balanced=int(bal.sum())))}
    for lv in xs.diag_levels:
        cA = float(np.percentile(pm[:, WV.A, WV.X], lv))
        cP = float(np.percentile(pm[:, WV.P, WV.X], lv))

        def acc(base, cA=cA, cP=cP):
            return (base[..., WV.A, WV.X] >= cA) & (base[..., WV.P, WV.X] >= cP)
        keep = acc(pm)
        out[f"F2({lv:g})"] = dict(theta=theta, accept=acc, pass_counts=dict(
            pilot=int(keep.sum()), balanced=int((keep & bal).sum()), c_A=cA, c_P=cP))
    return out


def _pair_counts() -> dict:
    return dict(n=0, n_pass=0, n_fail=0, solo=[0] * len(WV.GATES), low=[0] * len(WV.GATES), joint_only=0, mech=0,
                floor=[0, 0], table=[[0] * len(STAGE) for _ in ASSIGN])


def _accumulate(c, fin, solo, mcomp, fl, low) -> None:
    fail = fin == WV.P_FAIL
    mok = mcomp.all(-1)
    mf = fail & ~mok
    c["n"] += int(fin.size)
    c["n_pass"] += int((fin == WV.P_PASS).sum())
    c["n_fail"] += int(fail.sum())
    for j in range(len(WV.GATES)):
        c["solo"][j] += int((fail & solo[..., j]).sum())
        c["low"][j] += int((fail & low[..., j]).sum())
    c["joint_only"] += int((fail & mok & ~solo.any(-1)).sum())
    c["mech"] += int(mf.sum())
    for h in range(2):
        c["floor"][h] += int((fail & fl[..., h]).sum())
    live = fail & ~mf
    s = solo & live[..., None]
    fg, lg = s & fl[..., GATE_GROUP], s & low
    a_floor = live & fg.any(-1)
    a_low = live & ~a_floor & lg.any(-1)
    a_noise = live & ~a_floor & ~a_low

    def st(m):
        return _stage_idx(m[..., 0] | m[..., 2], m[..., 1] | m[..., 3])
    for ai, (mask, sidx) in enumerate(((mf, _stage_idx(~mcomp[..., 0], ~mcomp[..., 1])), (a_floor, st(fg)),
                                       (a_low, st(lg)), (a_noise, st(s)))):
        cnt = np.bincount(np.asarray(sidx)[mask].ravel(), minlength=len(STAGE))
        for si in range(len(STAGE)):
            c["table"][ai][si] += int(cnt[si])


def _share(a, b):
    return None if not b else a / b


def _finish(c) -> dict:
    nf = c["n_fail"]
    return dict(pass_share=_share(c["n_pass"], c["n"]), fail_share=_share(nf, c["n"]),
                solo_share=dict(zip(WV.GATES, [_share(v, nf) for v in c["solo"]])),
                low_dprime_share=dict(zip(WV.GATES, [_share(v, nf) for v in c["low"]])),
                joint_only_share=_share(c["joint_only"], nf), mech_share=_share(c["mech"], nf),
                floor_share=dict(reward=_share(c["floor"][0], nf), punish=_share(c["floor"][1], nf)),
                assignment={ASSIGN[a]: dict(zip(STAGE, c["table"][a])) for a in range(len(ASSIGN))}, counts=c)


def pair_diag(theta, cal, idx, root, n_rep, z, xs, cell=run_cell, log=None) -> dict:
    """X.9.1.4 (i) at true d′ 1.5 (cal = the power calibration): diag_pairs gate pairs × diag_f_max flies per
    experiment, every g, every (q, K, F ∈ diag_fs): the final PASS share; for FAIL pairs the single-gate failures (a
    gate whose satisfying flies on the decision data — K, or 2K for a BAND pair — are < q·F), joint-only failures,
    mechanism-control failures, floor caught (reward / punishment), per-pair true d′ < bar, and each failure's
    assignment (mechanism → floor → pair effect / balance → fly / probe noise) × stage; the share of drawn pairs whose
    four per-pair true d′ are all ≥ bar (the F → ∞ ceiling)."""
    a, b = cal["a"]["value"], cal["b"]["value"]
    P, Fm, d = xs.diag_pairs, xs.diag_f_max, xs.round_digits
    kw = dict(bar=xs.bar, band=xs.band_width, digits=d)

    def one(gi, g):
        r = rng(root, TAG_DIAG_PAIR, gi)
        acc, ceil_ok, ceil_n, done = {}, 0, 0, 0
        while done < n_rep:
            n = min(xs.diag_chunk, n_rep - done)
            st, base, _fill = simulate(theta, r, n, P, Fm, 2 * max(xs.k_grid), [a] * P, [b] * P, [1.0] * Fm,
                                       [1.0] * Fm, g, z, xs, xs.diag_tries)
            td, fl = pair_true_dprimes(theta, base.reshape(-1, 2, 2), a, b, idx, z, xs)
            low = (np.round(td - xs.bar, d) < 0).reshape(n, P, -1)
            fl = fl.reshape(n, P, -1)
            ceil_ok += int((~low).all(-1).sum())
            ceil_n += n * P
            for K in xs.k_grid:
                dK = {s: v[..., :K, :, :] for s, v in st.items()}
                d2 = {s: v[..., :2 * K, :, :] for s, v in st.items()}
                sK, s2 = WV.gate_stats(dK, z), WV.gate_stats(d2, z)
                cK, c2 = WV.fly_class(sK, **kw), WV.fly_class(s2, **kw)
                fK, f2 = WV.mech_fractions(dK), WV.mech_fractions(d2)
                for F in xs.diag_fs:
                    oK = np.round(np.median(fK[..., :F, :], axis=-2) - xs.mech_min, d) >= 0
                    o2 = np.round(np.median(f2[..., :F, :], axis=-2) - xs.mech_min, d) >= 0
                    for q in xs.q_grid:
                        codeK = WV.pair_gate_code(cK[..., :F], q, F, d)
                        fin = WV.pair_final(codeK, oK.all(-1), WV.pair_gate_code(c2[..., :F], q, F, d), o2.all(-1))
                        band = codeK == WV.P_BAND
                        used = np.where(band[..., None, None], s2[..., :F, :], sK[..., :F, :])
                        solo = np.round((WV.margins(used, xs.bar, d) >= 0).sum(-2) / F - q, d) < 0
                        _accumulate(acc.setdefault(f"{q}|{K}|{F}", _pair_counts()), fin, solo,
                                    np.where(band[..., None], o2, oK), fl, low)
            done += n
        if log is not None:
            log(f"x diag (i) g {g} done")
        return dict(cells={k: _finish(v) for k, v in acc.items()}, ceiling_share=ceil_ok / ceil_n)
    key = dict(theta=w_oc.summary(theta), a=a, b=b, n_rep=n_rep)
    out = {f"g{g}": cell(f"pair_g{g}", dict(key, g=g), lambda gi=gi, g=g: one(gi, g))
           for gi, g in enumerate(xs.cluster_grid)}
    gs = [f"g{g}" for g in xs.cluster_grid]
    out["worst_g"] = {c: min(gs, key=lambda k: out[k]["cells"][c]["pass_share"]) for c in out[gs[0]]["cells"]}
    return out


def min_change(worst_by_filter: dict, xs, fs, kk) -> dict:
    """X.9.1.4 (iii): per filter and k range [k_lo, k_lo + width], the smallest F whose design meets both targets at
    every k of the range (g worst; ties larger p_set, larger q, smaller K); ranked by the knobs changed (filter ≠
    none, k range ≠ [k_min, k_min + width], F > f_max) → smaller k_lo → smaller F."""
    entries = []
    for name, (pw, fp) in worst_by_filter.items():
        for k_lo in xs.diag_k_los:
            ks = [kk.index(k) for k in range(k_lo, k_lo + xs.diag_k_width + 1)]
            ok = meets(np.asarray(pw)[..., ks], np.asarray(fp)[..., ks], xs).all(-1)
            e = dict(filter=name, k_lo=k_lo, k_hi=k_lo + xs.diag_k_width, found=bool(ok.any()))
            if ok.any():
                i = min((c for c in np.ndindex(ok.shape) if ok[c]),
                        key=lambda i: (fs[i[3]], -xs.p_set_grid[i[0]], -xs.q_grid[i[1]], xs.k_grid[i[2]]))
                F = int(fs[i[3]])
                e.update(F=F, p_set=xs.p_set_grid[i[0]], q=xs.q_grid[i[1]], K=xs.k_grid[i[2]],
                         knobs=int(name != "none") + int(k_lo != xs.k_min) + int(F > xs.f_max),
                         power_by_k=np.asarray(pw)[i][ks].tolist(), false_by_k=np.asarray(fp)[i][ks].tolist())
            entries.append(e)
    found = sorted((e for e in entries if e["found"]), key=lambda e: (e["knobs"], e["k_lo"], e["F"]))
    return dict(entries=entries, ranking=found, rank1=found[0] if found else None,
                same_knobs=[e for e in found if e["knobs"] == found[0]["knobs"]] if found else [])


def diagnostics(theta, abs_d, z, xs, n_rep: int | None = None, cell=run_cell, log=None) -> dict:
    """X.9.1.4 (i)–(iii) on θ̂ (V0 point estimate), root diag_seed, its own calibration (X.9.1.2 per status). Never
    used by a gate, a STOP or a selection."""
    t0 = time.perf_counter()
    n_rep = xs.diag_reps if n_rep is None else int(n_rep)
    root = xs.diag_seed
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    cal = {m: calibrate_x(theta, getattr(xs, f), m, idx, z, xs) for m, f in MODES}
    flt = filters(theta, abs_d, xs)
    fs, kk = list(range(xs.f_min, xs.diag_f_max + 1)), list(range(xs.k_min, xs.diag_k_max + 1))
    shp = (len(xs.p_set_grid), len(xs.q_grid), len(xs.k_grid), len(fs), len(kk))
    key0 = dict(theta=w_oc.summary(theta), n_rep=n_rep,
                cal={m: [(c["a"] or {}).get("value"), (c["b"] or {}).get("value")] for m, c in cal.items()})
    grid = {}
    for fi, (name, f) in enumerate(flt.items()):
        for gi, g in enumerate(xs.cluster_grid):
            for m, _f in MODES:
                c = cal[m]
                if not c["ok"]:
                    grid[(name, g, m)] = dict(p=np.full(shp, c["fill"]), fill_share=None)
                    continue

                def run(f=f, fi=fi, gi=gi, g=g, m=m, c=c):
                    r = evaluate_grid(f["theta"], rng(root, TAG_DIAG_GRID, fi, gi, int(m == "max")), n_rep,
                                      c["a"]["value"], c["b"]["value"], g, z, xs, f["accept"])
                    return dict(p=r["p"].tolist(), fill_share=r["fill_share"])
                v = cell(f"diag_{name}_g{g}_{m}", dict(key0, filter=name, g=g, mode=m), run)
                grid[(name, g, m)] = dict(p=np.asarray(v["p"]), fill_share=v["fill_share"])
                if log is not None:
                    log(f"x diag (ii/iii) {name} g {g} {m} done ({time.perf_counter() - t0:.0f} s)")
    g0 = xs.cluster_grid[0]
    worst_ = {n: (np.min([grid[(n, g, "min")]["p"] for g in xs.cluster_grid], 0),
                  np.max([grid[(n, g, "max")]["p"] for g in xs.cluster_grid], 0)) for n in flt}
    nf, nk = xs.f_max - xs.f_min + 1, xs.k_cap - xs.k_min + 1
    sl = (Ellipsis, slice(0, nf), slice(0, nk))
    base_acc, _ = pair_bases(theta, rng(root, TAG_DIAG_ACCEPT), n_rep, xs.diag_pairs, g0, z, xs, xs.diag_tries)
    ii = {}
    for name, f in flt.items():
        pw, fp = worst_[name]
        best = pick(pw[sl], fp[sl], xs, fs[:nf], True)
        i = tuple(best["index"])
        fill = {f"g{g}|{m}": grid[(name, g, m)]["fill_share"] for g in xs.cluster_grid for m, _f in MODES}
        ii[name] = dict(best=best, best_g0=dict(power_by_k=grid[(name, g0, "min")]["p"][sl][i].tolist(),
                                                false_by_k=grid[(name, g0, "max")]["p"][sl][i].tolist()),
                        fill=fill, fill_flag=any(v is not None and v > xs.diag_fill_flag for v in fill.values()),
                        pass_counts=f["pass_counts"],
                        acceptance=None if f["accept"] is None else float(f["accept"](base_acc).mean()))
    iii = min_change(worst_, xs, fs, kk)
    pair = pair_diag(theta, cal["min"], idx, root, n_rep, z, xs, cell, log) if cal["min"]["ok"] else None
    arrays = {n: dict(power_worst=worst_[n][0].tolist(), false_worst=worst_[n][1].tolist(),
                      power_g0=grid[(n, g0, "min")]["p"].tolist(), false_g0=grid[(n, g0, "max")]["p"].tolist())
              for n in flt}
    return dict(calibration=cal, filters={n: dict(pass_counts=f["pass_counts"]) for n, f in flt.items()}, ii=ii,
                iii=iii, i=pair, arrays=arrays,
                axes=dict(p_set=list(xs.p_set_grid), q=list(xs.q_grid), K=list(xs.k_grid), F=fs, k=kk,
                          g=list(xs.cluster_grid)),
                n_rep=n_rep, seed=int(root), timing_s=time.perf_counter() - t0)


def diag_summary(diag: dict) -> dict:
    """The block's part: (ii) per filter, (iii)'s rank 1 and same-knob entries, (i) the pass share by F per (q, K)
    for every g with the worst g per cell."""
    i = None
    if diag["i"] is not None:
        i = dict(worst_g=diag["i"]["worst_g"],
                 pass_share={g: {c: v["pass_share"] for c, v in diag["i"][g]["cells"].items()}
                             for g in diag["i"] if g != "worst_g"},
                 ceiling_share={g: diag["i"][g]["ceiling_share"] for g in diag["i"] if g != "worst_g"})
    return dict(ii=diag["ii"], iii=dict(rank1=diag["iii"]["rank1"], same_knobs=diag["iii"]["same_knobs"]), i=i,
                calibration={m: dict(ok=c["ok"], failure=c["failure"], retried=c["retried"])
                             for m, c in diag["calibration"].items()},
                note="기록 전용(X.9.1.4) — 관문·STOP·선택에 쓰지 않는다; 거름은 순진(훈련 전) 관찰 지표만으로 정의")
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/brain/test_x_diag.py tests/brain/test_x_precheck.py tests/brain/test_x_oc.py tests/brain/test_x_spec.py -q > /private/tmp/claude-503/x-tests/t4.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t4.log`
Expected: all pass (`test_diagnostics_layout` takes about 10–30 s).

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/x_oc.py tests/brain/test_x_diag.py
git commit -m "feat(x): record-only diagnostics (i)-(iii) — per-pair true d' and floor flags, filters none/F1/F2(l), pair-level failure attribution, minimum-change ranking"
```

---

### Task 5: `x_store`, `x_runner` (stage0 · precheck), `scripts/run_x.py`

**Files:**
- Create: `flymon/brain/x_store.py`, `flymon/brain/x_runner.py`, `scripts/run_x.py`
- Test: `tests/brain/test_x_store.py`, `tests/brain/test_x_runner.py`, `tests/brain/test_x_cli.py`

**Interfaces:**
- Consumes: everything above, plus `w_runner.build_ctx`, `w_runner.Runner._pilot_back`, `w_runner.pipeline_key`, `w_runner.git_facts`, `w_runner.W_PIPELINE_FILES`, `w_store.load_manifest`, `w_measure.w_measure_key`, `e_runner.summary_git` and `h3_store` (`ROOT`, `canonical`, `sha256_file`, `git_state`).
- Produces:
  - `x_store.write_json(path, obj, params_list)`, `x_store.write_summary_block(path, block, obj, params_list, ledger)` and `x_store.read_summary(path)`.
  - `x_runner.ORDER = ("stage0", "precheck")`, `GATES`, `X_FILES`, `W_FILES`, `X_HASHED_FILES`, `env_hashes()`, `build_ctx(npz)` and `Runner(ctx, xs, summary_path=None)` with `stage_stage0()` and `stage_precheck()`.
  - `run_x.main(argv)`, `run_x.exit_code(out)` and `run_x.STAGES`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_x_store.py`:

```python
"""X's writer (X.2 · X.8): results/x/ and results/summary/x_learning.json only, atomic, the ledger appended."""
import json

import pytest

from flymon.brain import x_store as XS
from flymon.brain.config import Params


def test_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    XS.write_json("results/x/a.json", {"a": 1}, [Params()])
    XS.write_summary_block(XS.SUMMARY, "stage0", {"b": 2}, [Params()], ledger=dict(stage="stage0", wall_s=1.0))
    d = json.loads((tmp_path / XS.SUMMARY).read_text())
    assert d["stage0"] == {"b": 2} and d["ledger"][0]["stage"] == "stage0"
    for bad in ("results/w/oc.json", "results/summary/w_learning.json", "results/v/x.json", "x.json",
                "results/xx/a.json"):
        with pytest.raises(SystemExit) as e:
            XS.write_json(bad, {}, [Params()])
        assert e.value.code == 2
    assert not list((tmp_path / "results/x").glob(".*.tmp"))
```

`tests/brain/test_x_runner.py`:

```python
"""X's phase-A chain (X.5 0 · 0a): stage0 then precheck; refusals; STOP_REUSE on every broken W fact (records
unavailable); the W diagnosis checked draw by draw; precheck PASS / STOP blocks with their details; resumable cells."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import x_runner as XR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.w_spec import SPEC as W
from flymon.brain.x_spec import SPEC as X0

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ND = [0.08, -0.05, -0.32, 1.09, 1.43, 2.72, 4.13, 5.9, 6.0, 7.98, -9.89, 10.7, -12.88, 13.3, 14.5, -16.07]
KEYS = [f"p|{j}" for j in range(16)]
XS = dataclasses.replace(X0, w_measure_key="k" * 64, w_pipeline_key="p" * 64, w_measure_file_sha="m" * 64,
                         w_oc_detail_sha="o" * 64, w_summary_commit="c" * 7, balanced_pairs=tuple(KEYS[:3]),
                         precheck_reps=6, oc_reps=4, diag_reps=2, diag_chunk=2, p26_reps=4, boot_reps=4)


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.pilot = w_oc.synthetic_pilot(np.random.default_rng(5), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                          learn=(20.0, 10.0))
        theta = w_oc.fit(self.pilot)
        self.oc = json.loads(canonical(w_oc.run(self.pilot, Z, W, lambda K, F: K * F, n_boot=3, n_rep=2,
                                                n_boot_rep=2)))
        self.w_doc = dict(
            stage0=dict(decision_files={"flymon/brain/w_measure.py": "m" * 64}),
            reuse=dict(z_V={"A": list(Z["A"]), "P": list(Z["P"])}),
            pilot=dict(outcome="PASS", w_measure_key="k" * 64, pairs=list(KEYS), manifest=[{}] * 384,
                       record=dict(pairs={k: dict(naive_d=d) for k, d in zip(KEYS, ND)})),
            oc=dict(w_measure_key="k" * 64, theta=json.loads(canonical(w_oc.summary(theta))), detail_sha256="o" * 64))
        self.keys = dict(w_measure_key="k" * 64, pipeline_key="p" * 64, w_measure_sha="m" * 64)
        self.facts = dict(last="c" * 40, ancestors={"5fbc4c8": True, "f30ae35": True},
                          git=dict(tracked=True, dirty=False))
        self.bad_manifest = []
        self.ctx = dict(w_doc=lambda: self.w_doc, oc_detail=lambda p: (self.oc, "o" * 64),
                        pilot=lambda d: ((None, list(self.bad_manifest)) if self.bad_manifest
                                         else (dict(zip(KEYS, self.pilot)), [])),
                        keys=lambda: dict(self.keys), facts=lambda xs: dict(self.facts), params=lambda: Params())
        monkeypatch.setattr(XR, "summary_git", lambda p: dict(tracked=True, dirty=False, judged=[]))
        monkeypatch.setattr(XR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(XR.x_oc, "synthetic_validation", lambda xs, z, n_rep=None: dict(ok=True))

    def runner(self, xs=XS):
        return XR.Runner(self.ctx, xs)


def doc():
    return json.loads(Path(X0.summary).read_text())


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_stage0_pass(w):
    out = w.runner().stage_stage0()
    assert out["outcome"] == "PASS" and out["p26"]["equal"] and out["w_cal_diagnosis"]["reproduced"]
    assert out["tables"]["m_needed"]["m"]["0.5"] == [3, 3, 3, 4, 4] and out["oc_timing"]["reps"] == 4
    assert Path(X0.wcal_detail).exists() and doc()["ledger"][0]["stage"] == "stage0"
    assert out["w_measure_key"] == "k" * 64 and out["pipeline_key"] == "p" * 64


@pytest.mark.parametrize("breaker,needle", [
    (lambda w: w.keys.update(w_measure_key="z" * 64), "W 측정 키 "),
    (lambda w: w.facts["ancestors"].update({"5fbc4c8": False}), "5fbc4c8"),
    (lambda w: w.bad_manifest.append("b|0|0|R: raw file sha256 differs"), "W 파일럿 원자료"),
    (lambda w: w.w_doc["pilot"].update(pairs=KEYS[::-1]), "쌍 순서"),
    (lambda w: w.w_doc["oc"]["theta"].update(n_fly=9), "θ̂ 요약"),
    (lambda w: w.oc["boot_calibration"][2]["min"].update(ok=not w.oc["boot_calibration"][2]["min"]["ok"]),
     "W 보정 진단 재현 불일치"),
])
def test_stage0_stop_reuse_on_broken_w_facts(w, breaker, needle):
    breaker(w)
    out = w.runner().stage_stage0()
    assert out["outcome"] == "STOP_REUSE" and needle in out["sentence"] and out["records_unavailable"] is True
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2


def test_refusals(w, monkeypatch):
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2 and not Path(X0.summary).exists()
    w.runner().stage_stage0()
    with pytest.raises(SystemExit):
        w.runner().stage_stage0()
    monkeypatch.setattr(XR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=["flymon/brain/w_oc.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2 and "precheck" not in doc()


def test_precheck_stop_writes_records_diag_and_env(w):
    w.runner().stage_stage0()
    out = w.runner(dataclasses.replace(XS, p_power=1.01)).stage_precheck()
    assert out["outcome"] == "STOP_OC_UNREACHABLE" and out["sentence"].startswith("점 θ 사전 점검(X.9.1.1)")
    assert set(out["records"]["variants"]) == {"V0", "V1", "V2"}
    assert out["diag"]["iii"]["rank1"] is None or "knobs" in out["diag"]["iii"]["rank1"]
    assert out["residuals"]["n_pairs"] == dict(balanced=3, rest=13)
    assert set(out["env"]) == {"uv_lock_sha256", "python", "numpy", "x_files", "w_files"}
    assert "flymon/brain/w_measure.py" in out["env"]["w_files"] and "scripts/run_x.py" in out["env"]["x_files"]
    for p in (X0.precheck_detail, X0.diag_detail):
        assert Path(p).exists()
    assert out["diag_detail_sha256"] and out["precheck_detail_sha256"]


def test_precheck_pass_has_no_records(w):
    w.runner().stage_stage0()
    out = w.runner(dataclasses.replace(XS, p_power=0.0, p_false=1.0)).stage_precheck()
    assert out["outcome"] == "PASS" and out["records"] is None and "국면 B" in out["records_note"]


def test_precheck_resumes_from_cells(w, monkeypatch):
    w.runner().stage_stage0()
    saved = Path(X0.summary).read_text()
    hard = dataclasses.replace(XS, p_power=1.01)
    first = w.runner(hard).stage_precheck()
    Path(X0.summary).write_text(saved)                       # as if killed before the block was written

    def boom(*a, **k):
        raise AssertionError("a finished cell was recomputed")
    monkeypatch.setattr(XR.x_oc, "evaluate_grid", boom)
    second = w.runner(hard).stage_precheck()
    assert second["diag"] == first["diag"] and second["records"] == first["records"]
```

`tests/brain/test_x_cli.py`:

```python
"""scripts/run_x.py: stages equal the runner's ORDER, exit codes, refusals outside the repository root."""
import importlib.util
from pathlib import Path

from flymon.brain import x_runner as XR

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_x", ROOT / "scripts/run_x.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


def test_stages_and_exit_codes():
    assert tuple(CLI.STAGES) == XR.ORDER
    assert CLI.exit_code(dict(outcome="PASS")) == 0
    assert CLI.exit_code(dict(outcome="INVALID")) == 5
    assert CLI.exit_code(dict(outcome="STOP_REUSE")) == 3 and CLI.exit_code(dict(outcome="STOP_OC_UNREACHABLE")) == 3


def test_refuses_outside_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert CLI.main(["--stage", "stage0"]) == 2
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/brain/test_x_store.py tests/brain/test_x_runner.py tests/brain/test_x_cli.py -q > /private/tmp/claude-503/x-tests/t5.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t5.log`
Expected: `ModuleNotFoundError` for `x_store` / `x_runner` and a missing `scripts/run_x.py`.

- [ ] **Step 3: Write `flymon/brain/x_store.py`**

```python
"""X's only writer (X.2 · X.8; 글쓴이 해석 (8)): results/x/ and results/summary/x_learning.json, atomic writes
(temporary file + rename), nothing else (SystemExit 2). W's tree is read only (w_store's writer refuses outside
results/w/, so X never uses it). Phase B adds XCache here."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/x/"
SUMMARY = "results/summary/x_learning.json"


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"X writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


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
    doc = read_summary(path)
    doc[block] = obj
    if ledger is not None:
        doc["ledger"] = list(doc.get("ledger", [])) + [ledger]
    return write_json(path, doc, params_list)
```

- [ ] **Step 4: Write `flymon/brain/x_runner.py`**

```python
"""Spec X's stage chain, phase A (X.5 0 · 0a; X.9.1.1): stage0 -> precheck, one block each in
results/summary/x_learning.json, written only through x_store, plus the running ledger (X.8's own 24 h). Phase B
appends its stages to ORDER after precheck (only after a precheck PASS) and never edits these two.
Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block exists, its own
block exists, an earlier gate did not PASS, the summary has uncommitted changes or a hashed X / W file is dirty. Both
stages re-check the W facts the precheck rests on (Reading 2): W measurement key, W pipeline key, w_measure.py's hash,
W's summary (tracked, clean, last commit), W blocks pilot / oc in HEAD's history on the W key, the pilot manifest's
sha256 values, results/w/oc.json's sha256, the pair order (W's own _pilot_back = block pilot's pairs), the balanced /
V1 pair facts and θ̂'s summary = W block oc's theta — any break → STOP_REUSE (records_unavailable, P2-10)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import w_oc, w_runner, x_oc, x_rules, x_store, x_verdict
from .h3_store import ROOT as _ROOT
from .h3_store import canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .w_spec import SPEC as W_SPEC

ORDER = ("stage0", "precheck")
GATES = ("stage0", "precheck")
X_FILES = ("flymon/brain/x_spec.py", "flymon/brain/x_verdict.py", "flymon/brain/x_oc.py", "flymon/brain/x_rules.py",
           "flymon/brain/x_store.py", "flymon/brain/x_runner.py", "scripts/run_x.py")
W_FILES = tuple(dict.fromkeys(tuple(w_runner.W_PIPELINE_FILES) + ("flymon/brain/w_measure.py",)))
X_HASHED_FILES = X_FILES + W_FILES + ("results/summary/w_learning.json",)


def git_state() -> dict:
    return _h3_git_state(files=X_HASHED_FILES)


def files_sha(files) -> dict:
    return {f: sha256_file(_ROOT / f) for f in files}


def env_hashes() -> dict:
    """X.9.1.3 P2-11: uv.lock, Python, numpy, the X files and the W files X imports."""
    return dict(uv_lock_sha256=sha256_file(_ROOT / "uv.lock"), python=platform.python_version(),
                numpy=np.__version__, x_files=files_sha(X_FILES), w_files=files_sha(W_FILES))


def refuse(msg: str, code: int = 2):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def build_ctx(npz: str) -> dict:
    """The real context: W's summary and OC detail (read only), W's pilot through W's own _pilot_back on W's context
    (built once; the manifest's sha256 values checked first), the keys and git facts, the C3 params (writer guard)."""
    from . import w_store
    from .w_measure import w_measure_key
    cache = {}

    def wctx():
        if "c" not in cache:
            cache["c"] = w_runner.build_ctx(W_SPEC, npz)
        return cache["c"]

    def pilot(w_doc):
        _got, bad = w_store.load_manifest(w_doc["pilot"]["manifest"])
        if bad:
            return None, bad
        pairs, _ = w_runner.Runner(None, None, wctx(), W_SPEC)._pilot_back(w_doc)
        return pairs, []

    def oc_detail(path):
        p = Path(path)
        return (json.loads(p.read_text()), sha256_file(p)) if p.exists() else (None, None)
    return dict(w_doc=lambda: json.loads(Path(W_SPEC.summary).read_text()), oc_detail=oc_detail, pilot=pilot,
                keys=lambda: dict(w_measure_key=w_measure_key(npz)["key"], pipeline_key=w_runner.pipeline_key()["key"],
                                  w_measure_sha=sha256_file(_ROOT / "flymon/brain/w_measure.py")),
                facts=lambda xs: dict(w_runner.git_facts(xs.w_summary, [c for _, c in xs.w_commits]),
                                      git=summary_git(xs.w_summary)),
                params=lambda: wctx()["params"])


class Runner:
    def __init__(self, ctx: dict, xs, summary_path=None):
        self.ctx, self.xs = ctx, xs
        self.summary_path = str(summary_path or xs.summary)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return x_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed X / W files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; X never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; X never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != x_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — X stops there")
        return doc

    def _write(self, stage: str, body: dict, wall_s: float) -> dict:
        k = self.ctx["keys"]()
        block = dict(body, stage=stage, w_measure_key=k.get("w_measure_key"), pipeline_key=k.get("pipeline_key"),
                     git=git_state(), written_at=_now())
        x_store.write_summary_block(self.summary_path, stage, block, self.plist,
                                    dict(stage=stage, wall_s=float(wall_s), at=block["written_at"]))
        return block

    # ---- the W facts (Reading 2) -------------------------------------------------------------------------------------
    def _reuse(self) -> tuple:
        xs, c = self.xs, self.ctx
        w_doc = c["w_doc"]()
        why = x_rules.w_reuse_reasons(w_doc, c["keys"](), c["facts"](xs), c["oc_detail"](xs.w_oc_detail)[1], xs)
        pairs = theta = None
        if not why:
            pairs, bad = c["pilot"](w_doc)
            if bad:
                why.append("W 파일럿 원자료: " + "; ".join(bad[:3]))
            elif list(pairs) != list(w_doc["pilot"]["pairs"]):
                why.append("W _pilot_back 쌍 순서 ≠ W 파일럿 블록 pairs")
            else:
                theta = w_oc.fit(list(pairs.values()))
                if json.loads(json.dumps(w_oc.summary(theta))) != w_doc["oc"]["theta"]:
                    why.append("θ̂ 요약 ≠ W oc 블록 theta")
                why += x_rules.pair_fact_reasons(w_doc, list(pairs), xs)
        return why, w_doc, pairs, theta

    @staticmethod
    def _z(w_doc: dict) -> dict:
        return {k: (float(v[0]), float(v[1])) for k, v in w_doc["reuse"]["z_V"].items()}

    def _tables(self) -> dict:
        xs = self.xs
        return dict(designs=dict(p_set=list(xs.p_set_grid), q=list(xs.q_grid), K=list(xs.k_grid),
                                 F=[xs.f_min, xs.f_max], k=[xs.k_min, xs.k_cap],
                                 n=len(xs.p_set_grid) * len(xs.q_grid) * len(xs.k_grid) * (xs.f_max - xs.f_min + 1)),
                    set_rule=dict(min_gate_pairs=xs.min_gate_pairs, min_pass_pairs=xs.min_pass_pairs,
                                  digits=xs.round_digits),
                    m_needed=x_verdict.m_needed_table(xs),
                    seeds=dict(oc=xs.oc_seed, precheck=xs.precheck_seed, diag=xs.diag_seed, w_diagnosis=W_SPEC.oc_seed,
                               tags=dict(cal=x_oc.TAG_CAL, point=x_oc.TAG_POINT, record=x_oc.TAG_RECORD,
                                         synth=x_oc.TAG_SYNTH, p26=x_oc.TAG_P26, diag_grid=x_oc.TAG_DIAG_GRID,
                                         diag_pair=x_oc.TAG_DIAG_PAIR, diag_accept=x_oc.TAG_DIAG_ACCEPT)),
                    calibration=dict(rule="X.9.1.2 상태별", bracket_mult=xs.bracket_mult,
                                     w_bracket_mult=xs.w_bracket_mult, cal_iter=xs.cal_iter,
                                     cal_retry_iter=xs.cal_retry_iter, cal_tol=xs.cal_tol, cal_reps=xs.cal_reps,
                                     fill=dict(power=x_oc.fill_value("min"), false=x_oc.fill_value("max")),
                                     cal_floor_rule=xs.cal_floor_rule),
                    protocol="W 표 그대로(w_spec, W stage0 tables) — X.2")

    # ---- resumable cells (X.9.1.5: 2 h limit, resume) ---------------------------------------------------------------
    def _cell(self, tag: str, key_obj, fn):
        key = hashlib.sha256(canonical(dict(tag=tag, key=key_obj, x=files_sha(X_FILES))).encode()).hexdigest()
        p = Path(self.xs.progress_dir) / "precheck" / (re.sub(r"[^A-Za-z0-9._-]", "_", tag) + ".json")
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("key") == key:
                return d["value"]
        v = fn()
        x_store.write_json(str(p), dict(key=key, value=v, at=_now()), self.plist)
        return json.loads(canonical(v))

    # ================================================================ 0: code checks + W diagnosis (X.5 0)
    def stage_stage0(self) -> dict:
        self._require("stage0")
        xs = self.xs
        t0 = time.perf_counter()
        why, w_doc, _pairs, theta = self._reuse()
        z = self._z(w_doc)
        diag = p26 = timing = None
        detail = {}
        if not why:
            oc, _sha = self.ctx["oc_detail"](xs.w_oc_detail)
            diag = x_oc.w_cal_diagnosis(theta, z, W_SPEC, xs, oc["boot_calibration"], log=_log)
            why += x_rules.diagnosis_reasons(diag, xs)
            p = x_store.write_json(xs.wcal_detail, diag, self.plist)
            detail = dict(wcal_detail_path=xs.wcal_detail, wcal_detail_sha256=sha256_file(p))
        syn = x_oc.synthetic_validation(xs, z)
        if theta is not None and not why:
            idx = x_oc.rng(xs.oc_seed, x_oc.TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
            c = x_oc.calibrate_x(theta, xs.d_power, "min", idx, z, xs)
            a, b = (c["a"]["value"], c["b"]["value"]) if c["ok"] else (0.0, 0.0)
            p26 = x_oc.bit_identity(theta, z, xs, W_SPEC, xs.p26_reps, a, b)
            timing = x_oc.oc_timing(theta, z, xs)
        if why:
            dec = x_rules.reuse_stop(why)
        elif not syn["ok"] or not p26["equal"]:
            dec = dict(outcome=x_rules.INVALID, reasons=([] if syn["ok"] else ["X.4.6 합성 검증 실패"])
                       + ([] if p26["equal"] else ["P2-6 비트 동일 실패"]))
        else:
            dec = dict(outcome=x_rules.PASS, reasons=[])
        body = dict(dec, synthetic=syn, p26=p26, oc_timing=timing, tables=self._tables(),
                    w_cal_diagnosis=None if diag is None else dict(reproduced=diag["reproduced"],
                                                                   n_draws=diag["n_draws"], counts=diag["counts"],
                                                                   diffs=diag["diffs"][:5]),
                    decision_files=files_sha(X_FILES), w_files=files_sha(W_FILES), **detail,
                    note="X.5 순서 0(국면 A): 판정 코드·x_oc·합성 검증·P2-6·W 보정 진단. 진단은 규칙을 바꾸지 않는다"
                         "(X.9.1.2).")
        return self._write("stage0", body, time.perf_counter() - t0)

    # ================================================================ 0a: point-θ precheck + record-only diagnostics
    def stage_precheck(self) -> dict:
        self._require("precheck")
        xs = self.xs
        t0 = time.perf_counter()
        why, w_doc, pairs, theta = self._reuse()
        if why:
            return self._write("precheck", x_rules.reuse_stop(why), time.perf_counter() - t0)
        z = self._z(w_doc)
        abs_d = x_rules.abs_naive_d(w_doc, list(pairs))
        pc = x_oc.precheck(theta, z, xs)
        _log(f"x precheck point done ({pc['timing_s']:.0f} s)")
        dec = x_rules.precheck_decision(pc)
        rec = None
        if dec["outcome"] != x_rules.PASS:
            rec = x_oc.point_records(theta, abs_d, pc["records_target"], z, xs, cell=self._cell, log=_log)
        diag = x_oc.diagnostics(theta, abs_d, z, xs, cell=self._cell, log=_log)
        resid = x_oc.residual_compare(list(pairs.values()), abs_d < xs.naive_max, xs)
        p1 = x_store.write_json(xs.precheck_detail, dict(precheck=pc, records=rec), self.plist)
        p2 = x_store.write_json(xs.diag_detail, diag, self.plist)
        body = dict(dec, theta=w_oc.summary(theta),
                    calibration={m: dict(ok=c["ok"], failure=c["failure"], first_failure=c["first_failure"],
                                         retried=c["retried"], a=(c["a"] or {}).get("value"),
                                         b=(c["b"] or {}).get("value"), true_dprime=c.get("true_dprime"))
                                 for m, c in pc["calibration"].items()},
                    best_power=pc["best_power"], records_target=pc["records_target"], at_f32=pc["at_f32"],
                    n_passing=len(pc["passing"]), m_needed=pc["m_needed"], records=rec,
                    records_note=None if rec is not None else "사전 점검 통과 — records는 국면 B의 OC(X.4.5)",
                    diag=x_oc.diag_summary(diag), residuals=resid, env=env_hashes(),
                    precheck_detail_path=xs.precheck_detail, precheck_detail_sha256=sha256_file(p1),
                    diag_detail_path=xs.diag_detail, diag_detail_sha256=sha256_file(p2),
                    timing=dict(precheck_s=pc["timing_s"], diag_s=diag["timing_s"]),
                    note="점 θ 사전 점검(X.9.1.1): 점 추정·V0·포락선 없음 — 통과해도 자격이 아니다. 기록 전용 진단"
                         "(X.9.1.4)은 관문에 쓰지 않는다.")
        return self._write("precheck", body, time.perf_counter() - t0)
```

Note for the implementer: `_cell` returns `json.loads(canonical(v))` on the first run too, so the first and the resumed run return identical JSON values. This is what `test_precheck_resumes_from_cells` compares.

- [ ] **Step 5: Write `scripts/run_x.py`**

```python
#!/usr/bin/env python3
"""Spec appendix X, phase A (X.9.1 > X.0–X.9; X.9.1.1 구현 단계화): the controller runs each stage and commits its
block before the next.

    .venv/bin/python scripts/run_x.py --stage stage0     # 0  X code checks, X.4.6, P2-6 on θ̂, W calibration diagnosis
    .venv/bin/python scripts/run_x.py --stage precheck   # 0a point-θ precheck + record-only diagnostics (resumes)

Exit 0 PASS, 3 a gate STOP (STOP_REUSE, STOP_OC_UNREACHABLE; recorded, X stops), 5 INVALID (a code defect: do not
commit), 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted summary, dirty hashed file)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID = 3, 5
STAGES = ("stage0", "precheck")
QUIET = ("synthetic", "w_cal_diagnosis", "tables", "decision_files", "w_files", "calibration", "theta", "at_f32",
         "m_needed", "records", "diag", "residuals", "env", "p26", "oc_timing", "git")


def exit_code(out: dict) -> int:
    from flymon.brain import x_rules as R
    o = out.get("outcome")
    return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, required=True)
    a = ap.parse_args(argv)
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    from flymon.brain import x_runner
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.x_spec import SPEC
    try:
        out = getattr(x_runner.Runner(x_runner.build_ctx(NPZ), SPEC), f"stage_{a.stage}")()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    except DataMismatch as e:
        print(f"refusing: the Hallem data do not match their pins: {e}", file=sys.stderr)
        return 2
    if out.get("sentence"):
        print(out["sentence"])
    print(json.dumps({k: v for k, v in out.items() if k not in QUIET}, ensure_ascii=False,
                     default=str)[:SPEC.cli_print_chars])
    return exit_code(out)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `.venv/bin/python -m pytest tests/brain/test_x_store.py tests/brain/test_x_runner.py tests/brain/test_x_cli.py tests/brain/test_x_spec.py -q > /private/tmp/claude-503/x-tests/t5.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/t5.log`
Expected: all pass. The literal guard now scans `x_runner.py`, `x_store.py`, `x_rules.py` and `run_x.py` too.

- [ ] **Step 7: Run the whole X + W + spec-collector suite (regression)**

Run: `.venv/bin/python -m pytest tests/brain/test_x_*.py tests/brain/test_w_*.py tests/brain/test_p_spec.py -q > /private/tmp/claude-503/x-tests/all.log 2>&1; tail -5 /private/tmp/claude-503/x-tests/all.log`
Expected: all pass. W's tests are unchanged and must still pass with `x_spec` in `MODULES`. Also run `git diff --stat main -- flymon/brain/w_*.py scripts/run_w.py` and expect no output.

- [ ] **Step 8: Commit**

```bash
git add flymon/brain/x_store.py flymon/brain/x_runner.py scripts/run_x.py tests/brain/test_x_store.py tests/brain/test_x_runner.py tests/brain/test_x_cli.py
git commit -m "feat(x): x_store, x_runner (stage0: W facts, W calibration diagnosis, X.4.6, P2-6 on theta-hat; precheck: point gate, STOP records, diagnostics, P1-5, P2-11, resumable cells) and scripts/run_x.py"
```

---

## Runs (controller)

Runs are the controller's job and execute in the background (`run_in_background: true`). They do not belong to a subagent: a subagent dies after 600 s without output. Read only the exit code and the tail or error lines of each log, never the whole log.

**Preconditions (all must hold before `stage0`):**
- Tasks 1–5 are committed on `open-fly-brain-connectome` and `git status --porcelain` is empty.
- The Step 7 regression suite passed, and `git diff --stat main -- flymon/brain/w_*.py scripts/run_w.py` is empty.
- `data/malecns.npz` is present with the H3 sha256.
- `results/w/cache/w_learn/` holds the pilot's 384 files (the manifest of W block `pilot`).
- `shasum -a 256 results/w/oc.json` starts with `7a0a65f3`.
- `results/summary/x_learning.json` does not exist.
- `mkdir -p /private/tmp/claude-503/x-runs`.

**Order 0, `stage0`** (about 5–10 min: W context build about 1 min, X.4.6 at 1000 experiments × 5 fixtures about 1–2 min, W diagnosis 200 draws × 2 sides with recalibrations a few minutes, P2-6 on θ̂ 3 g × 2 × 200 experiments, timing):

```bash
.venv/bin/python scripts/run_x.py --stage stage0 > /private/tmp/claude-503/x-runs/stage0.log 2>&1; echo "exit $?" >> /private/tmp/claude-503/x-runs/stage0.log
```

- Exit 0 (PASS): commit `results/summary/x_learning.json` with
  `results(x): stage0 — W calibration diagnosis reproduced (fail 29 = power 14 + false 18 - 3; statuses <a|floor n, b|no_convergence n, …>; brackets x4 <…>; 160 steps <…>; X rule <…>); X.4.6 PASS at every p_set; P2-6 bit-identical on theta-hat`. Fill the brackets from the block's `w_cal_diagnosis.counts`.
- Exit 3 (`STOP_REUSE`): commit the block (`results(x): stage0 STOP_REUSE — <reasons>`), then follow **On STOP** below with the STOP_REUSE sentence and no records (`records_unavailable`).
- Exit 5 (INVALID: X.4.6 or P2-6 failed) is a code defect (Reading 10). Do **not** commit. Delete the untracked `results/summary/x_learning.json`, hand the failing check to a fresh implementer as a fix task with review, then rerun `stage0`.
- Exit 2 is a refusal: read the stderr line, fix the precondition, and rerun.

**Order 0a, `precheck`** (about 1–1.5 h: point grid 6 × 4000 experiments about 2–3 min, records on STOP up to about 20 min, diagnostics 36 grid cells × 4000 experiments × 16 pairs × 64 flies plus (i) 3 × g, about 40–60 min; the block records the actual times):

```bash
.venv/bin/python scripts/run_x.py --stage precheck > /private/tmp/claude-503/x-runs/precheck.log 2>&1; echo "exit $?" >> /private/tmp/claude-503/x-runs/precheck.log
```

- If the run is killed before it writes the block, rerun the same command. Finished cells under `results/x/progress/precheck/` are reused (same seeds, same X files → same values).
- Commit `results/summary/x_learning.json` with
  `results(x): precheck <PASS|STOP_OC_UNREACHABLE> — point power max <best design> k4-8 <values>; false <…>; calibration <ok/status>; diag (ii) best <filter …>, (iii) rank 1 <filter, k range, F>; P1-5 residual sd balanced vs rest <…>`.

**On precheck PASS:** phase A ends here. Write the phase-B plan (`docs/superpowers/plans/2026-10-0x-x-phase-b.md`). It appends to `x_spec` / `x_verdict` / `x_oc` / `x_rules` / `x_store` / `x_runner` / `run_x` and does not change phase-A computations; if it must, recompute the precheck with the same seeds and require bit-identity, X.9.1.1. It covers X.5 1–14: full reuse, bootstrap OC with variants and records, smoke, 5a, digest, oracle, estimate, naive, gates, learn, band, records, seal and judge. Do not implement phase B in the same session.

**On STOP (`STOP_OC_UNREACHABLE`, or `STOP_REUSE` at either stage):**
1. Write **X.10** in the spec, after X.9.1.6: "### X.10 결과 (2026-10-0x, 관문 STOP — **STOP_OC_UNREACHABLE**, 판정 없음)". Include:
   - The sentence verbatim from the block, and the precheck numbers: the best-power design and its k = 4–8 values, the false-pass side, the θ̂ calibration status and the F = 32 summary.
   - The W diagnosis status counts (가)–(라).
   - The records at θ̂ (target design, true d′ table, mixed, heterogeneity × 0.5/1/2, W scenarios, per variant).
   - The diagnostics: (i) where pairs fail (stage / gate / floor), (ii) the best design per filter and the acceptance rates, (iii) rank 1 and the same-knob entries.
   - P1-5's residual comparison and the P2-11 hashes.
   - The disclosures: X.0 (i)–(iv) and X.9.1.6, the readings of this plan, "사용자 사전 승인(2026-10-06)에 따라 권장안 채택", and that the main set was not used and stays available.
2. In the **README ledger**, add a Korean bullet after W's bullet (before "안 된 것(운영)") and an English bullet after W's English bullet. Both follow W's form: "관문 STOP(…, 부록 X.10 — STOP_OC_UNREACHABLE, 판정 없음)" and "gate stop … (appendix X.10)".
3. Commit `docs(x): X.10 result — precheck STOP_OC_UNREACHABLE (<numbers>); README ledger ko/en` with no trailers, then push to `origin` (switch to `gh auth switch --user lyutvs` first if the push is refused).
4. Per the user's process rule, write the diagnostics (i)–(iii) and the next-declaration candidates (balanced-pair filter Y from naive measurements, etc.) into `/private/tmp/claude-503/p-latest.md` (overwrite the full text) and continue as the standing approval of 2026-10-06 says.

## Self-review (done while writing)

- **Spec coverage:**
  - X.9.1.1: θ̂ and the `_pilot_back` order (Task 5 `_reuse`), the θ summary first (Task 5), per-status calibration with 2000 draws, ±0.02, 40 → 160 steps and ×4 brackets (Task 2), and the simulation with the X set rule, 750 designs, k 4–8, three g, two targets and 4000 experiments (Tasks 2–3).
  - X.9.1.1, continued: root 43_200_000 with W's tags (Task 3), the point values and pass rule (Task 3), the verbatim STOP sentence and its calibration-failure form (Task 3), and STOP records on V0 · V1 · V2 with tag 5 (Task 3).
  - X.9.1.2: per-status fills (Task 2) and the diagnosis (가)–(다), plus (라) (Task 2).
  - X.9.1.3: P1-3's table and duplicates (Task 1), V1 = 7 (Task 3), P1-5 (Task 3), P2-6 (Task 2, plus stage0 on θ̂), P2-7 (Task 5), P2-10 (Task 3 `reuse_stop`), P2-11 (Task 5 `env_hashes`) and P3-12 (Task 3 records).
  - X.9.1.4 (i)–(iii): Task 4, root 43_300_000, `results/x/precheck_diag.json`.
  - X.4.6: the set fixtures (Task 1), the synthetic validation at every p_set (Task 2), and the mutations relevant to phase A (Tasks 1–2). The variant-worst, success-only-limits and records-only-on-selected mutations belong to the phase-B bootstrap.
- **Placeholders:** none. Commit messages in **Runs** have `<…>` fields that the controller fills from the block.
- **Type consistency:** checked these names and signatures across the tasks:
  - `calibrate_x` keys `ok`, `a`, `b`, `fill`, `failure`, `first_failure` and `retried`.
  - `evaluate` → [p, q, K, F, k].
  - `pick` keys `index`, `power_by_k` and `false_by_k`.
  - `precheck` keys `calibration`, `at_f32`, `axes`, `best_power`, `records_target` and `passing`.
  - `point_records(theta, abs_d, target, z, xs, n_rep, cell, log)`.
  - `diagnostics(theta, abs_d, z, xs, n_rep, cell, log)` and `diag_summary`.
  - `x_rules.precheck_decision(pc)`.
  - Runner cells `cell(tag, key, fn)`.
