# M0d H.4 Selection Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A resumable runner, `scripts/run_m0d_h4.py`, that selects the M0d combination by spec H.4 as amended — for C0, C1 and C3 (H.3's adopted `Params`): readout reselection (reactivity + teachability), the combination's z constants, G.14's oracle on the even-turn pairs, then the selection rule — and writes block `"h4"` of `results/summary/m0d.json` once, after a complete run.

**Architecture:** Seven small modules under `flymon/brain/` with the prefix `h4_`, reusing H.3's store, cache and measurer unchanged. `h4_spec` is the single configuration object; `h4_pairs` ports the E0 even-turn pair list from the committed calibration code; `h4_formula` is G.14's verdict arithmetic with the z constants passed in; `h4_rules` holds every decision rule as a pure function; `h4_jobs` holds the FlyPool worker jobs (the M0c arm with per-type counts, G.14's oracle with a chosen readout); `h4_measure` serves reference/rest from H.3's cache and teach/oracle from H.4's (per pair); `h4_runner` is the procedure, written against a measurer interface so its tests drive every branch with a scripted measurer. The CLI wires them to a real `FlyPool`.

**Tech Stack:** Python 3.13, NumPy 2.5, pytest, `uv`; `flymon.brain.fly_pool.FlyPool` (spawn start method); `poke_env` (already a dependency) for the Gen 1 move data of the pair list.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` — **H.4** as amended by **H.3a.1** (combinations C0, C1, C3; tie order C0 > C1 > C3), **H.3a.9** (① teachability 6 of 8; ② the APL→MBON ablation, taken out by H.4a.2), **H.3a.10** (the baseline confound sentence, the "all dropped" path), **H.3a.13** (the adopted `Params`) and **H.4a** (committed with this plan: the taught odour, the ablation, the below-bar winner — three user decisions — the readings that change results, and the records kept next to the verdict); **G.14.3–G.14.4** for the oracle and its arithmetic; **F.3** for the z convention; **D.4 / M0c** for the conditioning arm. The spec's text is authoritative; "Readings of the spec" below lists every place the text needed a reading.

## Global Constraints

- **Commits carry no trailers of any kind** — no `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder that asks for them.
- **Nothing that H.3's measurement cache hashes changes**, nor anything the engine is: `engine_cpu.py`, `connectome.py`, `config.py`, `circuits.py`, `measure.py`, `stimuli.py`, `thresholds.py`, `fly_pool.py`, `pool_bench.py`, `pool_jobs.py`, `plasticity.py`, `presentation.py`, `conditioning.py`, every `flymon/brain/h3_*.py`, `scripts/run_m0d_h3.py` and `uv.lock`. H.4 reads H.3's cache (`results/m0d/h3/cache`) for the reactivity; a change there re-measures and the reactivity would no longer be H.3's guard by construction. `results/m0`, `results/m0b`, `results/m0c` are frozen references.
- **The spec amendment and this plan come first.** The planning session committed H.4a, its two diagnostics (`docs/superpowers/specs/m0d-diag/h4_teach_odour.py`, `h4_apl_ablation.py`) and this plan before Task 1 — `git log --oneline` shows `docs(spec): M0d H.4a …` and `docs(plans): M0d H.4 selection runner …`. Every task and every controller step starts from a HEAD that contains both commits (R0 checks it).
- **One configuration object.** Every H.4 constant is a field of `flymon.brain.h4_spec.SPEC` (or of its `h3` field, H.3's `SPEC`). No other module writes one of those numbers as a literal. Tests may build smaller specs with `dataclasses.replace`. Two exceptions, both pinned: `h4_pairs` keeps the ported calibration constants (power and HP bins, 16 turns, the 45-channel limit) exactly as `m2_probe.py` wrote them — the list is pinned by `SPEC.pairs_digest`, so any change refuses the run; the reproduction script takes F.3's constants from the frozen G.14 verdict module (sha256-checked), not from a literal.
- **Every output path goes through `h3_store`** (`write_json`, `write_bytes`, `MeasureCache`, `replace_summary_block`), which calls both `pool_bench.refuse_old_engine_output` and `pool_bench.refuse_modified_engine_output` for every `Params` given and refuses anything outside `results/m0d/` and `results/summary/m0d.json`. The H.4 runner writes `results/m0d/h4/{cache,runs}/` (or `h4-smoke/`), block `"h4"` of the summary, and — only when it is missing — C3's threshold file under `results/m0d/h3/thresholds/` restored from the committed copy.
- **Worker jobs are module-level functions inside `flymon/`** (the spawn pool re-imports them). Never build a `FlyPool` from `python - <<HEREDOC` (it deadlocks); always from a script file with `if __name__ == "__main__":` or from pytest.
- **Subagents never write under `results/` and never start the real H.4 run or the reproduction check.** Tests write only under `tmp_path`. The controller runs the real-data steps at the end of this plan.
- **Full suite before each commit:** `uv run pytest -q -rN -o addopts=""` — currently **385 passed, 1 skipped, 1 xfailed** (at the commit that adds this plan and H.4a). Each task states the count it must leave; no new failure is acceptable. Run it in the background if your harness blocks on long commands; read only the summary line and failures. Two new tests need git-ignored data and skip without it: `data/malecns.npz` (several) and `results/m2/calibration/encoders/E0_even.json` (one); the main worktree has both, and the counts below assume them.
- Style: match the surrounding modules — module docstring that cites the spec section, dense one-line comments, no type-annotated boilerplate beyond what neighbouring files use.

## Readings of the spec (decided here; the red team reviews these)

1. **Combinations and their `Params`.** Block `"h3"` of `results/summary/m0d.json`, `combos[C].adopted.params` for C0, C1, C3; each must be `adopted` (else the run refuses). The `Params` are rebuilt exactly (lists back to tuples) so H.3's cache keys match. C3's threshold file is the recorded path with its sha256; a missing file is restored from the committed copy `results/summary/m0d_h3_c3_thresholds.npz` when the sha256 matches, otherwise the run refuses.
2. **Pools.** The PPL105 and PAM08 core MBON types from `compartments` (A = [MBON13, MBON18], P = [MBON05, MBON21]); the run refuses when they differ from block `"h3"`'s `pools`.
3. **Reactivity (H.4 step 1, first clause) is the H.3 guard.** Same measurement (reference set, 96 presentations, and the same-seed rest), same statistic (`h3_rules.mbon_type_stats`: median of read − rest ≥ 5, zero-read share ≤ 0.25, both inclusive), served from H.3's cache through H.3's measurer. The run refuses (before measuring) when H.3's measurement key — the hash of the files H.3's measurements depend on — differs from block `"h3"`'s `measure_key`, and stops with an error, before any oracle, if any pool type's (median Δ, zero share) differs from block `"h3"`'s guard for that combination. H.3a.4's half-split reproduction is not part of H.4 and is not applied.
4. **Teachability (H.4 step 1, second clause, as fixed by H.4a.1).** The M0c single-channel arms (`punish_only`, `reward_only`: `conditioning.arms`, the `run_arm` sequence, trials 12, present 800 ms, gap 200 ms, settle 800 ms, read 600 ms, the design pair k = 8 seed 0, strength 0.35) at seeds 8–15, in both odour orders, with every pool type's count (all its cells). Per type: the taught odour is the design odour with the larger median naive (pre-probe) count over the seeds — the naive counts do not depend on the arm or the order; a tie keeps M0c's assignment (punishment on odour a, reward on odour b). A types use `punish_only` (PPL105 on the CS+ slot), P types `reward_only` (PAM08 on the CS− slot), in the order that puts the taught odour in that slot. Decreased ⇔ post < pre (strict); teachable ⇔ decreased on ≥ 6 of 8 seeds. The untaught odour's decreases in the same rows are recorded next to it (H.4a.4: the 6-of-8 rule does not tell associative from non-specific depression).
5. **Readout.** A (P) = the pool's types that are both reactive and teachable. Either empty → `dropped_no_readout`: no oracle, not eligible (H.4 step 1 "탈락"). Two types in one pool → `stop_multiple_types`: H.4 does not declare how to combine them, so the run stops there for the user (unreachable on H.3's numbers: one reactive type per pool in every combination).
6. **z constants.** Mean and population SD (ddof 0) of the readout type's count over the 96 reference presentations — F.3's frozen constants are population SDs over its 328 presentations (recomputed: 21.8293 / 18.1032 and 40.6433 / 24.0891; sample SDs would be 18.1309 / 24.1259). Only the SD ratio survives in ΔV. A zero SD raises.
7. **Oracle (H.4 step 2 = G.14.3).** `m2_engine_probe.oracle_job`'s sequence: KC activity of X on seeds 500–507 (settle 800, read 600, stepwise with G.8's 200 ms window recorded); α ∈ {0.2, 0.5, 0.8} picked on 600–607 (reward change max, then punishment change min, ties to the smaller α); pre / R1 / R2 re-probed on 608–615 and only those judged. The engine is the combination's (built from its `Params`, no `configure()`); V uses the combination's readout types and z. The weight edits act on the PAM08-core and PPL105-core KC→MBON edges as in G.14 (independent of which core type reads out). Probe counts are kept for all four pool types, so G.14.3's per-type d′ record can be derived. The pair list is the E0 even-turn list — (a) 18, (b) 21 — ported (`h4_pairs`), compared with the committed calibration modules and pinned by digest.
8. **The formula is G.14's.** `h4_formula` is `engine_probe_verdict`'s `dprime`, `dv`, `pair_stats`, `arm_aggregate` with z passed in; its test loads the frozen module (sha256-checked) and compares every function with F.3's constants.
9. **Invalid rows (G.14.4 row 1).** Per combination: duplicate rows, odd turns, pairs other than the declared list, report probes that are not the declared report seeds, an undefined d′ or a NaN → outcome `INVALID`: no selection, stop for the user.
10. **Selection (H.4 step 3, H.4a.3-5).** Eligible = F_a ≥ 2 among combinations with an oracle. Top = the largest number of testable (b) pairs; "near" = eligible combinations within 2 testable (b) pairs of the top; the first of them in C0 > C1 > C3 is selected. Stops look at the top only: no eligible combination → `STOP_NO_ELIGIBLE`; top T_b < 0.5 → `STOP_LOW_T_B`. A near combination below 0.5 can be selected — the spec's text, confirmed as a user decision (H.4a.3-5) — and is flagged `winner_below_bar` (H.5's T_b ≥ 0.4 takes it from there). C0 selected → `engine_unchanged`. H.5 is a separate plan.
11. **No ablation (H.4a.2).** No APL→MBON variant is measured or judged. The three interpretation sentences (H.3a.10's baseline confound, H.4a.2's direct APL→MBON05 dependence, H.4a.4's readout floors) are `SPEC.notes` and travel in the report and the summary.
12. **No exploratory path (H.4a.3-7).** H.3 adopted all three combinations; H.3a.10's "all dropped" path is not triggered. All combinations dropped at reselection → `STOP_NO_ELIGIBLE`.
13. **Resume = measurement cache (H.3a.12's form).** H.4's cache key hashes `h4_measure.MEASURE_FILES` — `uv.lock`, `circuits`, `config`, `connectome`, `engine_cpu`, `fly_pool`, `stimuli`, `thresholds`, `plasticity`, `presentation`, `conditioning`, `h3_store`, `h4_formula`, `h4_jobs`, `h4_measure` — plus the NPZ, the Python and NumPy versions and the measurement's inputs (the full `Params`, the odours, the seeds, the readout and z). `h4_rules`, `h4_runner`, `h4_spec`, `h4_pairs` and the CLI are in the dirty check and the manifest (`HASHED_FILES`), not in the key. The oracle is cached per pair and run in rounds of one pair per worker, so an interrupted run loses at most one round; the teaching arms are one entry per combination. Every combination is reselected before the first oracle, so a reactivity mismatch or a stop surfaces within minutes. The pool's timeout is 30 min (one round takes 6–8 min), so a lost worker cannot stall the run for hours.
14. **Summary gate.** Block `"h4"` is replaced by a run with all pairs, no `--smoke`/`--pairs`, clean hashed files, the declared configuration (`SPEC`), and an outcome that is neither `INVALID` nor `stop_multiple_types` (a mid-run stop); the connectome is checked before measuring (reading 15). `SELECTED`, `STOP_LOW_T_B` and `STOP_NO_ELIGIBLE` are complete results and are written; the user decides from them. Rerunning a finished run replaces the block again (same verdict from the cache, new run ID). The run report is always written under `<out>/runs/`.
15. **Exit codes.** 0 done (any outcome); 2 refused, always before measuring or writing anything: not at the repository root, dirty hashed files, an NPZ other than the declared connectome, no usable block `"h3"`, core pools or an H.3 measurement key other than block `"h3"`'s, a pair list other than the declared one, a block `"h3"` without three adopted combinations, malformed, or with a C3 threshold file (or copy) that fails its sha256. `adopted()` runs last because it may restore C3's file. A reactivity that differs from block `"h3"`'s guard raises (traceback before any oracle, nothing written).
16. **Records next to the verdict (H.4a.4, from the red team; not judged).** Per combination: the naive readout floor on the pairs' odours (mean count and zero share of each readout type over the report seeds and both candidates, G.14.6), the per-type statistics (G.14.3's "MBON13만·MBON05만의 d′": the other readout term held constant, so it cancels in ΔV; r, p, m per pair and the aggregate), the untaught odour's decreases per type (reading 4), and the testable (b) count on each half of the report seeds (608–611 / 612–615) — the seed noise the 2-pair tie band is compared with. They go into the report, the markdown report and block `"h4"`.

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/h4_spec.py` (create) | `H4Spec`, `SPEC` |
| `flymon/brain/h4_pairs.py` (create) | E0 even-turn pair list (port), `pair_key`, `pairs_digest` |
| `flymon/brain/h4_formula.py` (create) | G.14's `dprime`, `dv`, `pair_stats`, `arm_aggregate` with z passed in |
| `flymon/brain/h4_rules.py` (create) | statuses; reactivity, teachability, readout, z; row checks; selection |
| `flymon/brain/h4_jobs.py` (create) | per-worker rig cache; `teach_job`; `oracle_job` |
| `flymon/brain/h4_measure.py` (create) | `MEASURE_FILES`, `HASHED_FILES`, `H4Measurer` |
| `flymon/brain/h4_runner.py` (create) | `Context`, `reselect`, `run_combo`, `run_h4` |
| `scripts/run_m0d_h4.py` (create) | the CLI |
| `docs/superpowers/specs/m0d-diag/h4_reproduction_check.py` (create) | step R2: the jobs against M0c's and G.12's recorded counts |
| `tests/brain/test_h4_{spec,pairs,formula,rules,jobs,measure,runner}.py`, `tests/brain/h4_scripted.py`, `tests/test_run_m0d_h4.py` (create) | tests |

---

### Task 1: The configuration object and the pair list

**Files:**
- Create: `flymon/brain/h4_spec.py`, `flymon/brain/h4_pairs.py`
- Test: `tests/brain/test_h4_spec.py`, `tests/brain/test_h4_pairs.py`

**Interfaces:**
- Consumes: `flymon.brain.h3_spec.SPEC`, `H3Spec`, `Window`; `flymon.battle.pool.POOL`; `poke_env` Gen 1 data.
- Produces: `h4_spec.H4Spec` (frozen dataclass; fields as below) and `h4_spec.SPEC`; `h4_pairs.even_pairs(pops) -> list[dict]` (keys `axis`, `turn`, `x`, `y`, `odor_x`, `odor_y`), `h4_pairs.pair_key(p) -> (axis, turn, x, y)`, `h4_pairs.pairs_digest(pairs) -> str`, `h4_pairs.pool_vocabulary()`, `h4_pairs.e0_channels(pops, mon_types, move_types)`, `h4_pairs.power_bin`, `h4_pairs.hp_bin`.

The pair list is a port (same calls, same order) of `docs/superpowers/specs/m2-calibration-g/m2_probe.py` (`pool_vocabulary`, `assign_channels`, `build_turns`) and `m2_encoder_compare.py` (`build_encoder("E0")`, `odour`, `alternate_opponent`, `pairs_for`). The test compares the port with those modules on the real connectome, with G.12's raw labels and channels, and pins the digest `4e7298340f2225b839059e0b9508db1351c9e5630998d9deae2b4d0712946338` (validated in a scratch worktree: 39 pairs, identical lists).

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_h4_spec.py`:

```python
"""The H.4 configuration object restates nothing it could take from committed code: the M0c arm, G.14's oracle driver,
the H.3 guard, F.3's z convention and the spec's selection rule (spec H.4, H.3a.1, H.3a.9, H.4a)."""
import importlib.util
import inspect
import sys
from pathlib import Path

from flymon.brain import conditioning, pool_jobs
from flymon.brain.h3_spec import SPEC as H3
from flymon.brain.h4_spec import SPEC

CAL = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/m2-calibration-g"


def _g14_driver():
    sys.path.insert(0, str(CAL))
    try:
        s = importlib.util.spec_from_file_location("m2_engine_probe", CAL / "m2_engine_probe.py")
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(str(CAL))


def _defaults(fn):
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect._empty}


def test_the_combinations_and_their_tie_order():
    assert SPEC.combos == ("C0", "C1", "C3")                    # H.3a.1: C2 dropped; C0 > C1 > C3


def test_reactivity_is_the_h3_guard_and_uses_h3s_reference_set():
    assert (SPEC.react_med_delta_min, SPEC.react_zero_share_max) == (H3.guard_med_delta_min, H3.guard_zero_share_max)
    assert SPEC.h3 == H3


def test_the_teaching_arm_is_m0cs():
    job, arm = _defaults(pool_jobs.conditioning_arm_job), _defaults(conditioning.run_arm)
    assert SPEC.teach_seeds == tuple(range(8, 16)) and SPEC.teach_min_decreased == 6      # D.4 seeds; H.3a.9 (1)
    assert (SPEC.teach_trials, SPEC.teach_present_ms, SPEC.teach_window.settle_ms) == \
        (job["trials"], job["present_ms"], job["settle_ms"])
    assert (SPEC.teach_gap_ms, SPEC.teach_window.read_ms) == (arm["gap_ms"], arm["read_ms"])
    assert (job["strength"], job["k"], job["odor_seed"]) == (H3.strength, H3.design_k, H3.design_odor_seed)
    assert (job["punish_type"], job["reward_type"]) == (H3.punish_type, H3.reward_type)
    assert SPEC.teach_orders == ("ab", "ba")                                               # H.4a.1


def test_the_oracle_is_g14s_driver():
    d = _g14_driver()
    assert list(SPEC.act_seeds) == d.ACT_SEEDS and list(SPEC.select_seeds) == d.SELECT_SEEDS
    assert list(SPEC.report_seeds) == d.REPORT_SEEDS and list(SPEC.oracle_alphas) == d.ALPHAS
    assert H3.strength == d.STRENGTH and SPEC.kc_window_ms == d.WINDOW_MS
    assert (SPEC.oracle_window.settle_ms, SPEC.oracle_window.read_ms) == (d.SETTLE_MS, d.READ_MS)


def test_the_selection_rule_and_the_notes():
    assert (SPEC.f_a_min, SPEC.t_b_min, SPEC.tie_pairs, SPEC.z_ddof) == (2, 0.5, 2, 0)
    assert len(SPEC.notes) == 3 and "H.3a.10" in SPEC.notes[0] and "H.4a.2" in SPEC.notes[1] and "H.4a.4" in SPEC.notes[2]
```

`tests/brain/test_h4_pairs.py`:

```python
"""The oracle's pair list (spec H.4 step 2 = G.14.3): the port equals the committed calibration code and G.12's list."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flymon.brain import h4_pairs as H
from flymon.brain.circuits import Populations
from flymon.brain.connectome import Connectome
from flymon.brain.h4_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
CAL = ROOT / "docs/superpowers/specs/m2-calibration-g"
G12_RAW = ROOT / "results/m2/calibration/encoders/E0_even.json"      # git-excluded: G.12's E0 even-turn rows
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="data/malecns.npz not present")


@pytest.fixture(scope="module")
def pops():
    return Populations.from_connectome(Connectome.load(NPZ))


@pytest.fixture(scope="module")
def pairs(pops):
    return H.even_pairs(pops)


def _load(name):
    sys.path.insert(0, str(CAL))
    try:
        s = importlib.util.spec_from_file_location(name, CAL / f"{name}.py")
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(str(CAL))


@needs_npz
def test_the_port_equals_the_committed_calibration_code(pops, pairs):
    P, E = _load("m2_probe"), _load("m2_encoder_compare")
    st, mi, mt, vt = P.pool_vocabulary()
    turns = [t for t in P.build_turns(st, mi, 16) if t["turn"] % 2 == 0]
    assert pairs == E.pairs_for("E0", pops, E.build_encoder("E0", pops, mt, vt), turns, st)


@needs_npz
def test_the_list_is_the_declared_one(pairs):
    assert sum(p["axis"] == "a" for p in pairs) == SPEC.n_pairs_a == 18
    assert sum(p["axis"] == "b" for p in pairs) == SPEC.n_pairs_b == 21
    assert all(p["turn"] % 2 == 0 for p in pairs)
    assert H.pairs_digest(pairs) == SPEC.pairs_digest


@needs_npz
@pytest.mark.skipif(not G12_RAW.exists(), reason="G.12 raw rows not present (git-excluded)")
def test_labels_and_channels_are_g12s(pops, pairs):
    raw = json.loads(G12_RAW.read_text())
    assert [H.pair_key(p) for p in pairs] == [(r["axis"], r["turn"], r["x"], r["y"]) for r in raw["rows"]]
    _, _, mt, vt = H.pool_vocabulary()
    assert H.e0_channels(pops, mt, vt) == raw["channels"]


def test_the_digest_sees_every_field():
    p = dict(axis="a", turn=0, x="Surf", y="Earthquake", odor_x={"ORN_A": 1.0}, odor_y={"ORN_B": 1.0})
    d = H.pairs_digest([p])
    for k, v in [("axis", "b"), ("turn", 2), ("x", "Cut"), ("y", "Cut"), ("odor_x", {"ORN_A": 1.5}),
                 ("odor_y", {"ORN_C": 1.0})]:
        assert H.pairs_digest([dict(p, **{k: v})]) != d, k
    assert H.pairs_digest([dict(p, odor_x={"ORN_A": 1.0, "ORN_B": 2.0})]) == \
        H.pairs_digest([dict(p, odor_x={"ORN_B": 2.0, "ORN_A": 1.0})])


@pytest.mark.parametrize("bp, bin_", [(59, "lt60"), (60, "60to89"), (89, "60to89"), (90, "ge90")])
def test_power_bins(bp, bin_):
    assert H.power_bin(bp) == bin_


@pytest.mark.parametrize("frac, bin_", [(0.34, "low"), (0.35, "mid"), (0.67, "mid"), (0.68, "high")])
def test_hp_bins(frac, bin_):
    assert H.hp_bin(frac) == bin_
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_spec.py tests/brain/test_h4_pairs.py`
Expected: collection errors — `ModuleNotFoundError: No module named 'flymon.brain.h4_spec'`.

- [ ] **Step 3: Write the configuration object**

`flymon/brain/h4_spec.py`:

```python
"""The M0d H.4 runner's one configuration object (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a).

H.3a.11's rule carries over: every number the H.4 runner, its rules and its records use is a field of `SPEC` (or of a
`dataclasses.replace` of it, for smoke runs and tests), and no other module restates one. The reference set, its
windows, the design pair and the connectome are H.3's (`h3`): H.4's reactivity and z constants are defined on H.3's
reference set (H.3: "활성 조정과 H.4의 반응성·z 상수는 이 집합에서만 한다"), and the reactivity is the H.3 guard's
measurement, served from H.3's measurement cache.
"""
from __future__ import annotations

from dataclasses import dataclass

from .h3_spec import SPEC as H3_SPEC, H3Spec, Window


@dataclass(frozen=True)
class H4Spec:
    h3: H3Spec = H3_SPEC
    # ---- the combinations (H.3a.1: C2 dropped); the order is also H.4's tie order C0 > C1 > C3 -----------------
    combos: tuple = ("C0", "C1", "C3")
    h3_block: str = "h3"                        # results/summary/m0d.json block holding the adopted Params
    # ---- H.4 step 1: readout reselection ---------------------------------------------------------------------
    react_med_delta_min: float = 5.0            # median of (read - same-seed rest) over the reference set
    react_zero_share_max: float = 0.25          # share of reference presentations with a zero read count
    teach_seeds: tuple = tuple(range(8, 16))    # M0c's judged seeds
    teach_min_decreased: int = 6                # H.3a.9 (1): 6 of 8 (was 7)
    teach_orders: tuple = ("ab", "ba")          # H.4a: both odour orders run; each type is judged on its odour
    teach_trials: int = 12                      # the M0c arm (pool_jobs.conditioning_arm_job)
    teach_present_ms: float = 800.0
    teach_gap_ms: float = 200.0
    teach_window: Window = Window(800.0, 600.0)
    z_ddof: int = 0                             # F.3's frozen constants are population SDs
    # ---- H.4 step 2: the oracle (G.14.3) -------------------------------------------------------------------------
    oracle_alphas: tuple = (0.2, 0.5, 0.8)
    act_seeds: tuple = tuple(range(500, 508))
    select_seeds: tuple = tuple(range(600, 608))
    report_seeds: tuple = tuple(range(608, 616))
    oracle_window: Window = Window(800.0, 600.0)
    kc_window_ms: int = 200                     # G.8's sliding window, recorded
    n_pairs_a: int = 18
    n_pairs_b: int = 21
    pairs_digest: str = "4e7298340f2225b839059e0b9508db1351c9e5630998d9deae2b4d0712946338"
    # ---- G.14.4: the per-pair formula and the per-combination aggregate ------------------------------------------
    testable_min: float = 2.0                   # m = min(r, -p) >= 2
    naive_max: float = 0.5                      # |d_pre| < 0.5
    # ---- H.4 step 3: selection -------------------------------------------------------------------------------------
    f_a_min: int = 2
    t_b_min: float = 0.5
    tie_pairs: int = 2                          # within 2 testable (b) pairs of the top -> the earlier combination
    # ---- interpretation notes carried with the result (H.3a.10, H.4a.2, H.4a.4) ------------------------------------
    notes: tuple = (
        "C0 대 C1·C3 비교는 기저에서 교락돼 있다(H.3a.10): C1·C3는 mbon_hold_frac을 3.5 Hz로 재보정했고 C0은 동결 0.85로 간다. "
        "기저는 hold에 약 27.9 Hz/unit로 반응하고 H.4의 판독이 MBON 발화율이다.",
        "V는 APL→MBON05 억제에 직접 반응할 수 있다(H.3a.9 ②, H.4a.2): A·P 풀이 모두 APL 표적이고 MBON05는 APL의 MBON 표적 1위다. "
        "APL→MBON05를 끊으면 MBON05가 풀려 MBON13이 바닥으로 내려가므로 절제판 V로는 이 교락을 가를 수 없다.",
        "판독 바닥이 조합마다 다르다(H.4a.4): H.3 가드의 MBON13 중앙값 Δ가 C0 17.5 · C1 9.5 · C3 6.0(문턱 5)이라 T_b 차이의 일부는 "
        "엔진이 아니라 판독 바닥의 차이일 수 있다(G.14.8과 같은 종류). 쌍 냄새의 순진 판독 수준·0 비율과 타입별 d′를 함께 싣는다.")


SPEC = H4Spec()
```

- [ ] **Step 4: Write the pair list**

`flymon/brain/h4_pairs.py`:

```python
"""The oracle's pair list for M0d H.4 (spec H.4 step 2 = G.14.3: encoder E0, even turns, (a) 18 pairs and (b) 21 pairs).

Ported from the committed calibration code with the same calls in the same order — the turns and vocabulary from
`m2-calibration-g/m2_probe.py` (`pool_vocabulary`, `assign_channels`, `build_turns`) and the E0 encoder and the pairs
from `m2-calibration-g/m2_encoder_compare.py` (`build_encoder("E0")`, `odour`, `alternate_opponent`, `pairs_for`),
which produced G.12's and G.14's pair lists. Only E0 is ported (H.4 names no other encoder). tests/brain/test_h4_pairs.py
checks the port against those modules and pins the list by digest (`H4Spec.pairs_digest`).
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from ..battle.pool import POOL
from .circuits import Populations

EXCLUDE = ("ORN_DA1", "ORN_V")          # spec 3.3: cVA and CO2 channels are not free to reassign
POWER, HP = ["lt60", "60to89", "ge90"], ["low", "mid", "high"]
POWER_EDGES = (60, 90)
HP_EDGES = (0.34, 0.67)


def pool_vocabulary():
    """Species types, move (type, base power), the sorted mon types and move types of the 16-mon pool (Gen 1)."""
    from poke_env.battle import Move
    from poke_env.data import GenData
    from poke_env.data.normalize import to_id_str
    gd = GenData.from_gen(1)
    species_types, move_info = {}, {}
    for m in POOL:
        species_types[m.species] = tuple(t.upper() for t in gd.pokedex[to_id_str(m.species)]["types"])
        for a in m.attacks:
            mv = Move(to_id_str(a), gen=1)
            move_info[a] = (mv.type.name, int(mv.base_power))
    mon_types = sorted({t for v in species_types.values() for t in v})
    move_types = sorted({v[0] for v in move_info.values()})
    return species_types, move_info, mon_types, move_types


def assign_channels(pops: Populations, groups: list) -> dict:
    """m2_probe.assign_channels: receptor types by receptor count, trimmed symmetrically, spread per group."""
    cand = sorted((t for t in pops.receptor_types if t not in EXCLUDE),
                  key=lambda t: (len(pops.receptor_types[t]), str(t)))
    need = sum(len(names) for _, names in groups)
    if need > 45:
        raise ValueError(f"spec 3.3 allows at most 45 channels, this vocabulary needs {need}")
    if need > len(cand):
        raise ValueError(f"need {need} receptor types, have {len(cand)}")
    lo = (len(cand) - need) // 2
    band = cand[lo:lo + need]
    free = set(range(need))
    out = {}
    for gname, names in sorted(groups, key=lambda g: (len(g[1]), g[0])):
        q = len(names)
        for j, nm in enumerate(names):
            ideal = (j + 0.5) * need / q
            pos = min(free, key=lambda p: (abs(p + 0.5 - ideal), p))
            free.discard(pos)
            out[f"{gname}:{nm}"] = str(band[pos])
    return out


def power_bin(bp: int) -> str:
    return "lt60" if bp < POWER_EDGES[0] else ("60to89" if bp < POWER_EDGES[1] else "ge90")


def hp_bin(frac: float) -> str:
    return "low" if frac <= HP_EDGES[0] else ("mid" if frac <= HP_EDGES[1] else "high")


def build_turns(species_types, move_info, n_turns: int) -> list:
    hps = [(0.9, 0.9), (0.9, 0.3), (0.5, 0.6), (0.2, 0.8), (0.6, 0.2), (0.3, 0.5)]
    turns = []
    for i in range(n_turns):
        me, opp = POOL[i % len(POOL)], POOL[(i * 5 + 3) % len(POOL)]
        my_hp, op_hp = hps[i % len(hps)]
        cands = [{"move": a, "type": move_info[a][0], "bp": move_info[a][1]} for a in me.attacks][:4]
        turns.append({"turn": i, "me": me.species, "opp": opp.species,
                      "my_types": list(species_types[me.species]), "opp_types": list(species_types[opp.species]),
                      "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands})
    return turns


def e0_channels(pops: Populations, mon_types, move_types) -> dict:
    """m2_encoder_compare.build_encoder("E0"): {"group:symbol": [receptor type]} by receptor count."""
    syms = ([("my", t) for t in mon_types] + [("opp", t) for t in mon_types] + [("move", t) for t in move_types]
            + [("pow", b) for b in POWER] + [("myhp", b) for b in HP] + [("opphp", b) for b in HP])
    groups = {}
    for g, sym in syms:
        groups.setdefault(g, []).append(sym)
    flat = assign_channels(pops, [(g, v) for g, v in groups.items()])
    return {k: [v] for k, v in flat.items()}


def odour(pops, chan, my_types, opp_types, move, bp, my_hp, opp_hp) -> dict:
    """m2_encoder_compare.odour with every E0 group present: strengths proportional to 1/receptor count, mean 1."""
    keys = ([f"my:{t}" for t in my_types] + [f"opp:{t}" for t in opp_types] + [f"move:{move}", f"pow:{power_bin(bp)}",
            f"myhp:{hp_bin(my_hp)}", f"opphp:{hp_bin(opp_hp)}"])
    glom = [g for k in keys for g in chan[k]]
    if len(set(glom)) != len(glom):
        raise ValueError(f"glomerulus collision in {keys}")
    inv = np.array([1.0 / len(pops.receptor_types[g]) for g in glom]); inv /= inv.mean()
    return {g: float(s) for g, s in zip(glom, inv)}


def alternate_opponent(turn_index: int, me: str, opp_types, species_types) -> str:
    """G.11: POOL[(5i + 3 + k) mod 16], the first k >= 1 with no shared type and not my own species."""
    for k in range(1, len(POOL)):
        cand = POOL[(5 * turn_index + 3 + k) % len(POOL)].species
        if cand != me and not (set(species_types[cand]) & set(opp_types)):
            return cand
    raise ValueError(f"turn {turn_index}: no type-disjoint alternate opponent")


def even_pairs(pops: Populations, n_turns: int = 16) -> list:
    """[{axis, turn, x, y, odor_x, odor_y}] for the even turns, in G.12/G.14 order ((a) then (b) within each turn)."""
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    chan = e0_channels(pops, mon_types, move_types)
    out = []
    for t in build_turns(species_types, move_info, n_turns):
        if t["turn"] % 2:
            continue
        cands = t["candidates"]
        od = lambda c, opp_types: odour(pops, chan, t["my_types"], opp_types, c["type"], c["bp"], t["my_hp"], t["opp_hp"])
        for i, j in itertools.combinations(range(len(cands)), 2):
            out.append({"axis": "a", "turn": t["turn"], "x": cands[i]["move"], "y": cands[j]["move"],
                        "odor_x": od(cands[i], t["opp_types"]), "odor_y": od(cands[j], t["opp_types"])})
        alt = alternate_opponent(t["turn"], t["me"], t["opp_types"], species_types)
        for c in cands:
            out.append({"axis": "b", "turn": t["turn"], "x": f"{c['move']} vs {t['opp']}", "y": f"{c['move']} vs {alt}",
                        "odor_x": od(c, t["opp_types"]), "odor_y": od(c, list(species_types[alt]))})
    return out


def pair_key(p: dict) -> tuple:
    return (p["axis"], int(p["turn"]), p["x"], p["y"])


def pairs_digest(pairs: list) -> str:
    rows = [[p["axis"], int(p["turn"]), p["x"], p["y"], sorted((k, float(v)) for k, v in p["odor_x"].items()),
             sorted((k, float(v)) for k, v in p["odor_y"].items())] for p in pairs]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_spec.py tests/brain/test_h4_pairs.py`
Expected: 17 passed (on a checkout without `results/m2/calibration/encoders/E0_even.json`: 16 passed, 1 skipped).

- [ ] **Step 6: Full suite, then commit**

Run: `uv run pytest -q -rN -o addopts=""` → **402 passed, 1 skipped, 1 xfailed**.

```bash
git add flymon/brain/h4_spec.py flymon/brain/h4_pairs.py tests/brain/test_h4_spec.py tests/brain/test_h4_pairs.py
git commit -m "feat(brain): h4_spec and h4_pairs — the H.4 configuration object and the E0 even-turn pair list"
```

---

### Task 2: The oracle's arithmetic and the decision rules

**Files:**
- Create: `flymon/brain/h4_formula.py`, `flymon/brain/h4_rules.py`
- Test: `tests/brain/test_h4_formula.py`, `tests/brain/test_h4_rules.py`

**Interfaces:**
- Consumes: `h4_spec.SPEC` (Task 1); `h3_rules.mbon_type_stats(stim, rest_by_seed, name, med_min, zero_max) -> dict` (keys `median_delta`, `zero_share`, `passes`, …).
- Produces:
  - `h4_formula.dprime(x) -> float | None`; `dv(probe, z) -> np.ndarray` with `probe = {"A": [[x, y] per seed], "P": …}` and `z = {"A": (mean, sd), "P": (mean, sd)}`; `pair_stats(report, z, testable_min) -> dict | None` (keys `d_pre`, `r`, `p`, `m`, `testable`); `arm_aggregate(stats, naive_max, t_b_min, f_a_min) -> dict` (keys `n_a`, `n_b`, `T_b`, `F_a`, `testable_a`, `testable_b`, `naive_a`, `naive_b`, `bar`).
  - `h4_rules` statuses `READOUT_SELECTED`, `DROPPED_NO_READOUT`, `STOP_MULTI_TYPE`, outcomes `INVALID`, `SELECTED`, `STOP_NO_ELIGIBLE`, `STOP_LOW_T_B`; `reactivity(ref_rows, rest_by_seed, types, spec) -> {type: stats}`; `teach_type(rows, t, slot, spec) -> dict` (keys `slot`, `pre`, `post`, `n_decreased`, `teachable`); `teach_choice(rows, t, arm, spec) -> dict` (adds `arm`, `odour`, `order`, `naive_median`); `pick_readout(react, teach, pools) -> dict` (keys `status`, `readout` = `{"A": type, "P": type}` or None, `passing`); `z_constants(ref_rows, readout, ddof) -> {"A": (mean, sd), "P": (mean, sd)}`; `check_rows(rows, expected) -> list[str]`; `combo_stats(rows, z, expected, spec) -> dict` (keys `reasons`, `aggregate`, `pairs`); `combo_records(rows, z, spec) -> dict` (keys `naive_floor` = `{"A"|"P": {mean, zero_share, n}}`, `single_type` = `{"A"|"P": {aggregate, pairs}}`, `report_halves` = `[{seeds, testable_b, undefined}] × 2`); `select(aggs, spec) -> dict` (keys `outcome`, `winner`, `eligible`, `near`, and `top_T_b`, `engine_unchanged`, `winner_below_bar` when they apply). `teach_choice` also returns `untaught_n_decreased`.

Teach rows (Task 3 produces them) are `{"seed", "arm", "order", "pre": {"plus": {type: n}, "minus": {…}}, "post": {…}}`; "plus" is the CS+ slot (odour a in order "ab", odour b in "ba"). Oracle rows are `{"axis", "turn", "x", "y", "report": {"pre", "R1", "R2"}}` with each probe `{"A": [[x, y] per seed], "P": …}`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_h4_formula.py`:

```python
"""h4_formula is G.14's verdict arithmetic with z passed in (spec H.4 step 2): checked against the frozen module."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import h4_formula as F
from flymon.brain.h4_spec import SPEC

FROZEN = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/m2-calibration-g/engine_probe_verdict.py"
FROZEN_SHA = "c9c81ab97d13e7bc163210564e1114d3caafd60c63431bbef8eb977f8fb334d7"   # spec G.14


@pytest.fixture(scope="module")
def V():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA
    s = importlib.util.spec_from_file_location("engine_probe_verdict", FROZEN)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def _probe(rng, n, lo=0, hi=60):
    return {"A": rng.integers(lo, hi, size=(n, 2)).tolist(), "P": rng.integers(lo, hi, size=(n, 2)).tolist()}


def test_thresholds_are_g14s(V):
    assert (SPEC.testable_min, SPEC.naive_max, SPEC.t_b_min, SPEC.f_a_min) == (V.TESTABLE, V.NAIVE, V.BAR_T_B, V.BAR_F_A)


@pytest.mark.parametrize("x", [[1.0], [], [0.0, 0.0], [2.0, 2.0], [-3.0, -3.0, -3.0], [1.0, 2.0, 4.0], [0.5, -0.5]])
def test_dprime_is_g14s_including_its_limits(V, x):
    assert F.dprime(x) == V.dprime(x)


def test_pair_stats_with_f3_constants_is_g14s_on_random_and_degenerate_reports(V):
    rng = np.random.default_rng(7)
    reports = [{k: _probe(rng, 8) for k in ("pre", "R1", "R2")} for _ in range(300)]
    same = _probe(rng, 8)
    reports.append({"pre": same, "R1": same, "R2": same})                          # every d' is the 0 limit
    shifted = {"A": [[a + 5, b] for a, b in same["A"]], "P": same["P"]}
    reports.append({"pre": same, "R1": shifted, "R2": shifted})                     # r = +inf, p = 0
    reports.append({k: _probe(rng, 1) for k in ("pre", "R1", "R2")})               # undefined
    for rep in reports:
        assert F.pair_stats(rep, V.Z, SPEC.testable_min) == V.pair_stats(rep)


def test_arm_aggregate_is_g14s(V):
    rng = np.random.default_rng(3)
    for _ in range(50):
        keys = [(ax, t, f"x{j}", f"y{j}") for ax, n in (("a", 18), ("b", 21)) for j, t in enumerate(range(n))]
        stats = {k: V.pair_stats({p: _probe(rng, 8) for p in ("pre", "R1", "R2")}) for k in keys}
        assert F.arm_aggregate(stats, SPEC.naive_max, SPEC.t_b_min, SPEC.f_a_min) == V.arm_aggregate(stats)


def test_dv_uses_the_given_constants():
    probe = {"A": [[30, 10], [20, 20]], "P": [[5, 25], [40, 40]]}
    z = {"A": (100.0, 10.0), "P": (-7.0, 5.0)}
    # V(X) - V(Y) = (A_x - A_y) / sd_A - (P_x - P_y) / sd_P; the means cancel
    assert F.dv(probe, z).tolist() == [20 / 10 - (-20) / 5, 0.0]
    with pytest.raises(ValueError):
        F.dv({"A": [[1, 2, 3]], "P": [[1, 2, 3]]}, z)


def test_testable_is_inclusive_at_the_threshold():
    base = {"A": [[20, 20]] * 4, "P": [[40, 40]] * 4}
    r1 = {"A": base["A"], "P": [[40 - d, 40] for d in (2, 4, 2, 4)]}                  # dV rises by d / sd_P
    z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}
    r2 = {"A": [[20 - d, 20] for d in (2, 4, 2, 4)], "P": r1["P"]}
    s = F.pair_stats({"pre": base, "R1": r1, "R2": r2}, z, testable_min=float(np.mean([2, 4]) / np.std([2, 4, 2, 4], ddof=1)))
    assert s["r"] == -s["p"] == s["m"] and s["testable"]
```

`tests/brain/test_h4_rules.py`:

```python
"""The H.4 decision rules on synthetic inputs (spec H.4 as amended by H.3a.1, H.3a.9 and H.4a)."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import h4_rules as R
from flymon.brain.h4_spec import SPEC

SEEDS = list(SPEC.teach_seeds)


# ---- reactivity ----------------------------------------------------------------------------------------------------
def test_reactivity_is_the_h3_guard_statistic_with_h4s_thresholds():
    stim = [dict(odor=f"R{j:02d}", seed=s, types={"T": c}) for j, (s, c) in enumerate([(1, 7), (2, 0), (3, 6), (4, 6)])]
    rest = {s: dict(types={"T": 1}) for s in (1, 2, 3, 4)}
    got = R.reactivity(stim, rest, ["T"], SPEC)["T"]
    assert got["median_delta"] == 5.0 and got["zero_share"] == 0.25 and got["passes"]          # both edges inclusive
    strict = R.reactivity(stim, rest, ["T"], dataclasses.replace(SPEC, react_zero_share_max=0.24))["T"]
    assert not strict["passes"]


# ---- teachability -------------------------------------------------------------------------------------------------
def _rows(arm, order, pre_plus, pre_minus, post_plus, post_minus, t="T"):
    return [dict(seed=s, arm=arm, order=order, pre={"plus": {t: a}, "minus": {t: b}},
                 post={"plus": {t: c}, "minus": {t: d}})
            for s, a, b, c, d in zip(SEEDS, pre_plus, pre_minus, post_plus, post_minus)]


def test_teach_type_counts_strict_decreases_against_six_of_eight():
    pre, post = [10] * 8, [9] * 6 + [10, 11]
    r = R.teach_type(_rows("punish_only", "ab", pre, [0] * 8, post, [0] * 8), "T", "plus", SPEC)
    assert r["n_decreased"] == 6 and r["teachable"]
    r = R.teach_type(_rows("punish_only", "ab", pre, [0] * 8, [9] * 5 + [10] * 3, [0] * 8), "T", "plus", SPEC)
    assert r["n_decreased"] == 5 and not r["teachable"]
    r = R.teach_type(_rows("punish_only", "ab", [0] * 8, [0] * 8, [0] * 8, [0] * 8), "T", "plus", SPEC)
    assert r["n_decreased"] == 0 and not r["teachable"]                                     # silent: 0 -> 0 is no decrease
    with pytest.raises(ValueError):
        R.teach_type(_rows("punish_only", "ab", pre, pre, post, post)[:7], "T", "plus", SPEC)


@pytest.mark.parametrize("arm, naive_a, naive_b, odour, order, slot", [
    ("punish_only", 0, 25, "b", "ba", "plus"),       # MBON13 in C0: silent to a -> taught on b, CS+ = b
    ("punish_only", 15, 5, "a", "ab", "plus"),
    ("punish_only", 7, 7, "a", "ab", "plus"),        # tie -> M0c's assignment (punish on odour a)
    ("reward_only", 40, 3, "a", "ba", "minus"),      # reward on the CS- slot: odour a is CS- in order "ba"
    ("reward_only", 3, 40, "b", "ab", "minus"),
    ("reward_only", 9, 9, "b", "ab", "minus"),       # tie -> M0c's assignment (reward on odour b)
])
def test_teach_choice_judges_the_odour_the_type_answers(arm, naive_a, naive_b, odour, order, slot):
    rows = []
    for o in ("ab", "ba"):
        na, nb = (naive_a, naive_b) if o == "ab" else (naive_b, naive_a)      # "plus" is odour a in "ab", b in "ba"
        post_plus, post_minus = ([0] * 8, [nb] * 8) if slot == "plus" else ([na] * 8, [0] * 8)
        rows += _rows(arm, o, [na] * 8, [nb] * 8, post_plus, post_minus)
    got = R.teach_choice(rows, "T", arm, SPEC)
    assert (got["odour"], got["order"], got["slot"]) == (odour, order, slot)
    assert got["untaught_n_decreased"] == 0                                    # the other slot's counts never fall here
    assert got["naive_median"] == {"a": float(naive_a), "b": float(naive_b)}
    assert got["teachable"] == (max(naive_a, naive_b) > 0 if (naive_a != naive_b) else
                                (naive_a if slot == "plus" else naive_b) > 0)


# ---- readout ---------------------------------------------------------------------------------------------------------
POOLS = {"A": ["MA1", "MA2"], "P": ["MP1", "MP2"]}


def _flags(react, teach):
    return ({t: {"passes": t in react} for t in ("MA1", "MA2", "MP1", "MP2")},
            {t: {"teachable": t in teach} for t in ("MA1", "MA2", "MP1", "MP2")})


@pytest.mark.parametrize("react, teach, status, readout", [
    ({"MA1", "MP1"}, {"MA1", "MA2", "MP1", "MP2"}, R.READOUT_SELECTED, {"A": "MA1", "P": "MP1"}),
    ({"MA1", "MA2", "MP2"}, {"MA2", "MP2"}, R.READOUT_SELECTED, {"A": "MA2", "P": "MP2"}),
    ({"MA1", "MP1"}, {"MP1"}, R.DROPPED_NO_READOUT, None),                  # A empty: reactive but not teachable
    ({"MA1"}, {"MA1", "MP1"}, R.DROPPED_NO_READOUT, None),                  # P empty
    ({"MA1", "MA2", "MP1"}, {"MA1", "MA2", "MP1"}, R.STOP_MULTI_TYPE, None),
])
def test_pick_readout(react, teach, status, readout):
    got = R.pick_readout(*_flags(react, teach), POOLS)
    assert got["status"] == status and got["readout"] == readout


def test_z_constants_are_population_moments_of_the_type_count():
    rows = [dict(types={"MA1": a, "MP1": p}) for a, p in [(1, 10), (3, 10), (5, 40), (7, 40)]]
    z = R.z_constants(rows, {"A": "MA1", "P": "MP1"}, SPEC.z_ddof)
    assert z == {"A": (4.0, float(np.sqrt(5.0))), "P": (25.0, 15.0)}
    with pytest.raises(ValueError):
        R.z_constants([dict(types={"MA1": 2, "MP1": 1})] * 3, {"A": "MA1", "P": "MP1"}, 0)


# ---- oracle rows -----------------------------------------------------------------------------------------------------
Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}


def report(testable=True, naive=True, n=8):
    """pre: dV alternates +-1 (d_pre 0) or +5 +-1 (not naive); R1 lowers P(X) by 20 + (s % 2), R2 lowers A(X) the same."""
    step = [20 + (s % 2) for s in range(n)] if testable else [0] * n
    off = 0 if naive else 5
    pre = {"A": [[20 + off + (1 if s % 2 else -1), 20] for s in range(n)], "P": [[40, 40]] * n}
    r1 = {"A": pre["A"], "P": [[40 - d, 40] for d in step]}
    r2 = {"A": [[a - d, b] for (a, b), d in zip(pre["A"], step)], "P": r1["P"]}
    return {"pre": pre, "R1": r1, "R2": r2}


def rows_for(n_b_testable, n_a_testable_naive, n_a=4, n_b=5):
    rows = [dict(axis="a", turn=2 * (j % 4), x=f"ax{j}", y=f"ay{j}", report=report(j < n_a_testable_naive))
            for j in range(n_a)]
    rows += [dict(axis="b", turn=2 * (j % 4), x=f"bx{j}", y=f"by{j}", report=report(j < n_b_testable))
             for j in range(n_b)]
    return rows


def test_the_report_helper_hits_the_intended_statistics():
    from flymon.brain.h4_formula import pair_stats
    assert pair_stats(report(True), Z, 2.0)["testable"] and abs(pair_stats(report(True), Z, 2.0)["d_pre"]) < 0.5
    assert not pair_stats(report(False), Z, 2.0)["testable"]
    assert abs(pair_stats(report(True, naive=False), Z, 2.0)["d_pre"]) >= 0.5


def test_combo_stats_aggregates_clean_rows():
    rows = rows_for(3, 2)
    got = R.combo_stats(rows, Z, [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows], SPEC)
    assert got["reasons"] == [] and got["aggregate"]["testable_b"] == 3 and got["aggregate"]["F_a"] == 2
    assert len(got["pairs"]) == len(rows)


def test_combo_stats_names_every_invalid_row_set():
    rows = rows_for(3, 2)
    exp = [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows]
    assert "duplicate pair rows" in R.combo_stats(rows + rows[:1], Z, exp, SPEC)["reasons"]
    odd = [dict(rows[0], turn=3)] + rows[1:]
    got = R.combo_stats(odd, Z, exp, SPEC)["reasons"]
    assert "odd-turn rows present" in got and any("differ from the declared list" in r for r in got)
    short = [dict(r, report={k: {"A": v["A"][:1], "P": v["P"][:1]} for k, v in r["report"].items()}) for r in rows]
    got = R.combo_stats(short, Z, exp, SPEC)
    assert got["aggregate"] is None and any("undefined d'" in r for r in got["reasons"])
    assert any("not the 8 report seeds" in r for r in got["reasons"])
    seven = [dict(r, report={k: {"A": v["A"][:7], "P": v["P"][:7]} for k, v in r["report"].items()}) for r in rows]
    got = R.combo_stats(seven, Z, exp, SPEC)                                    # d' defined, still not the declared seeds
    assert got["aggregate"] is None and got["reasons"] == [f"{len(rows)} pairs whose report probes are not the 8 report seeds"]
    nanz = {"A": (0.0, float("nan")), "P": (0.0, 1.0)}
    assert any("NaN" in r for r in R.combo_stats(rows, nanz, exp, SPEC)["reasons"])


# ---- records ---------------------------------------------------------------------------------------------------------
def test_combo_records_floor_single_types_and_halves():
    rows = rows_for(3, 2)
    rec = R.combo_records(rows, Z, SPEC)
    a = [c for r in rows for pair in r["report"]["pre"]["A"] for c in pair]
    assert rec["naive_floor"]["A"] == dict(mean=float(np.mean(a)), zero_share=0.0, n=len(a))
    assert rec["naive_floor"]["P"] == dict(mean=40.0, zero_share=0.0, n=len(a))
    # report(): reward lowers P(X), punishment lowers A(X): alone, the P term sees only the reward (p = 0 -> m <= 0), the
    # A term only the punishment (r = 0 -> m <= 0), so neither type alone makes a pair testable
    assert rec["single_type"]["A"]["aggregate"]["testable_b"] == 0 and rec["single_type"]["P"]["aggregate"]["testable_b"] == 0
    pa = next(x for x in rec["single_type"]["A"]["pairs"] if x["x"] == "bx0")
    pp = next(x for x in rec["single_type"]["P"]["pairs"] if x["x"] == "bx0")
    assert pa["r"] == 0.0 and pa["p"] < -2 and pp["r"] > 2 and pp["p"] == 0.0
    assert [h["seeds"] for h in rec["report_halves"]] == [list(SPEC.report_seeds[:4]), list(SPEC.report_seeds[4:])]
    assert [h["testable_b"] for h in rec["report_halves"]] == [3, 3]


# ---- selection -------------------------------------------------------------------------------------------------------
def agg(testable_b, f_a, n_b=21):
    return dict(testable_b=testable_b, n_b=n_b, T_b=testable_b / n_b, F_a=f_a)


@pytest.mark.parametrize("aggs, outcome, winner, near", [
    ({"C0": agg(12, 2), "C1": agg(15, 3), "C3": agg(9, 5)}, R.SELECTED, "C1", ["C1"]),
    ({"C0": agg(13, 2), "C1": agg(15, 3), "C3": agg(14, 5)}, R.SELECTED, "C0", ["C0", "C1", "C3"]),   # within 2 pairs
    ({"C0": agg(12, 2), "C1": agg(15, 3), "C3": agg(13, 5)}, R.SELECTED, "C1", ["C1", "C3"]),         # 3 pairs: out
    ({"C0": agg(20, 1), "C1": agg(15, 3), "C3": agg(14, 2)}, R.SELECTED, "C1", ["C1", "C3"]),         # C0 ineligible
    ({"C0": None, "C1": agg(11, 2), "C3": agg(10, 2)}, R.SELECTED, "C1", ["C1", "C3"]),              # 11/21 >= 0.5
    ({"C0": agg(10, 2), "C1": agg(10, 3), "C3": None}, R.STOP_LOW_T_B, None, []),                   # 10/21 < 0.5
    ({"C0": agg(15, 1), "C1": agg(15, 0), "C3": None}, R.STOP_NO_ELIGIBLE, None, []),
    ({"C0": None, "C1": None, "C3": None}, R.STOP_NO_ELIGIBLE, None, []),
])
def test_select(aggs, outcome, winner, near):
    got = R.select(aggs, SPEC)
    assert (got["outcome"], got["winner"], got["near"]) == (outcome, winner, near)
    if winner:
        assert got["engine_unchanged"] == (winner == "C0")


def test_select_keeps_a_winner_whose_own_t_b_is_below_half_when_the_top_clears_it():
    """H.4 step 3 as written (H.4a.3-5, user decision): the stop looks at the best T_b; the tie band may pick an
    earlier combination below 0.5, which is flagged."""
    got = R.select({"C0": agg(9, 2), "C1": agg(11, 2), "C3": None}, SPEC)
    assert got["outcome"] == R.SELECTED and got["winner"] == "C0" and got["top_T_b"] == 11 / 21
    assert got["winner_below_bar"]
    assert not R.select({"C0": agg(11, 2), "C1": agg(12, 2), "C3": None}, SPEC)["winner_below_bar"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_formula.py tests/brain/test_h4_rules.py`
Expected: collection errors — `No module named 'flymon.brain.h4_formula'`.

- [ ] **Step 3: Write the formula**

`flymon/brain/h4_formula.py`:

```python
"""G.14's oracle arithmetic for M0d H.4, with the readout's z constants passed in (spec H.4 step 2: "판정 산식
pair_stats·arm_aggregate는 바꾸지 않는다").

This is `docs/superpowers/specs/m2-calibration-g/engine_probe_verdict.py` (sha256 c9c81ab9…) — `dprime`, `dv`,
`pair_stats`, `arm_aggregate` — except that `dv` takes z = {"A": (mean, sd), "P": (mean, sd)} instead of reading F.3's
frozen constants, and the thresholds come from the configuration object. tests/brain/test_h4_formula.py loads the frozen
module and checks every function against it with F.3's constants. The oracle job imports `dprime` and `dv` (its alpha
choice), so this module is part of the measurement key; the decision rules live in h4_rules, outside it.
"""
from __future__ import annotations

import numpy as np


def dprime(x) -> float | None:
    """mean / sd (ddof 1). sd = 0 is a limit: 0 if the mean is 0, else +-inf. Fewer than 2 seeds: undefined."""
    x = np.asarray(x, float)
    if x.size < 2:
        return None
    sd, m = float(x.std(ddof=1)), float(x.mean())
    if sd == 0:
        return 0.0 if m == 0 else float(np.copysign(np.inf, m))
    return m / sd


def dv(probe: dict, z: dict) -> np.ndarray:
    """V(X) - V(Y) per probe seed, V = z_A - z_P."""
    A = np.asarray(probe["A"], float); P = np.asarray(probe["P"], float)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape != P.shape:
        raise ValueError(f"probe must be [seeds x 2] for A and P, got {A.shape} / {P.shape}")
    V = (A - z["A"][0]) / z["A"][1] - (P - z["P"][0]) / z["P"][1]
    return V[:, 0] - V[:, 1]


def pair_stats(report: dict, z: dict, testable_min: float) -> dict | None:
    """report = {"pre", "R1", "R2"} on the same seeds: d_pre = d'(dV_pre), r = d'(dV_R1 - dV_pre),
    p = d'(dV_R2 - dV_R1), m = min(r, -p), testable <=> m >= testable_min. None if any d' is undefined."""
    pre, r1, r2 = dv(report["pre"], z), dv(report["R1"], z), dv(report["R2"], z)
    if not (len(pre) == len(r1) == len(r2)):
        raise ValueError("pre / R1 / R2 must share the probe seeds")
    d_pre, r, p = dprime(pre), dprime(r1 - pre), dprime(r2 - r1)
    if d_pre is None or r is None or p is None:
        return None
    m = min(r, -p)
    return {"d_pre": d_pre, "r": r, "p": p, "m": m, "testable": bool(m >= testable_min)}


def arm_aggregate(stats: dict, naive_max: float, t_b_min: float, f_a_min: int) -> dict:
    """stats: {(axis, turn, x, y): pair_stats}. T_b = testable share on axis (b); F_a = axis-(a) pairs testable with
    |d_pre| < naive_max; bar = T_b >= t_b_min and F_a >= f_a_min."""
    b = [s for (ax, *_), s in stats.items() if ax == "b"]
    a = [s for (ax, *_), s in stats.items() if ax == "a"]
    naive_ok = lambda s: abs(s["d_pre"]) < naive_max
    t_b = sum(s["testable"] for s in b) / len(b)
    f_a = sum(s["testable"] and naive_ok(s) for s in a)
    return {"n_a": len(a), "n_b": len(b), "T_b": t_b, "F_a": f_a,
            "testable_a": sum(s["testable"] for s in a), "testable_b": sum(s["testable"] for s in b),
            "naive_a": sum(naive_ok(s) for s in a), "naive_b": sum(naive_ok(s) for s in b),
            "bar": bool(t_b >= t_b_min and f_a >= f_a_min)}
```

- [ ] **Step 4: Write the rules**

`flymon/brain/h4_rules.py`:

```python
"""The M0d H.4 decision rules (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a) as pure functions over measured rows.

Nothing here runs the engine: readout reselection (reactivity, teachability, the readout, z constants), the row checks
of G.14.4's first row, the records kept next to the verdict and the selection. The per-pair arithmetic is h4_formula's (G.14's code).
Editing this module never invalidates a cached measurement (it is outside h4_measure.MEASURE_FILES).
"""
from __future__ import annotations

import numpy as np

from .h3_rules import mbon_type_stats
from .h4_formula import arm_aggregate, pair_stats

# ---- combination statuses at reselection ----------------------------------------------------------------------
READOUT_SELECTED = "readout_selected"
DROPPED_NO_READOUT = "dropped_no_readout"       # A or P empty after reselection (H.4 step 1: 탈락)
STOP_MULTI_TYPE = "stop_multiple_types"         # two types of one pool pass: H.4 does not say how to combine them
# ---- run outcomes ------------------------------------------------------------------------------------------------
INVALID = "INVALID"                             # G.14.4 row 1: missing / extra / duplicate / odd-turn rows, undefined d', NaN
SELECTED = "SELECTED"
STOP_NO_ELIGIBLE = "STOP_NO_ELIGIBLE"           # no combination with F_a >= 2 (H.4 step 3: 멈추고 사용자 판단)
STOP_LOW_T_B = "STOP_LOW_T_B"                   # the best T_b < 0.5 (멈추고 사용자 판단)


# ================================================================ readout reselection (H.4 step 1)
def reactivity(ref_rows: list, rest_by_seed: dict, types: list, spec) -> dict:
    """{type: h3_rules.mbon_type_stats}: the H.3 guard statistic — median over the reference presentations of
    (read - same-seed rest) >= react_med_delta_min and a zero-read share <= react_zero_share_max."""
    return {t: mbon_type_stats(ref_rows, rest_by_seed, t, spec.react_med_delta_min, spec.react_zero_share_max)
            for t in types}


def teach_type(rows: list, t: str, slot: str, spec) -> dict:
    """One type on one arm-and-order's rows: decreased <=> post < pre on the odour in `slot` ("plus" / "minus");
    teachable <=> decreased on at least teach_min_decreased of the declared seeds."""
    by_seed = {int(r["seed"]): r for r in rows}
    if sorted(by_seed) != sorted(spec.teach_seeds) or len(by_seed) != len(rows):
        raise ValueError(f"teach rows must cover the seeds {list(spec.teach_seeds)} once each")
    pre = [int(by_seed[s]["pre"][slot][t]) for s in spec.teach_seeds]
    post = [int(by_seed[s]["post"][slot][t]) for s in spec.teach_seeds]
    n = sum(b < a for a, b in zip(pre, post))
    return dict(slot=slot, pre=pre, post=post, n_decreased=n, teachable=bool(n >= spec.teach_min_decreased))


def teach_choice(rows: list, t: str, arm: str, spec) -> dict:
    """H.4a: a type is judged on the design odour it answers more — the larger median naive (pre) count over the teach
    seeds; a tie keeps M0c's assignment — taught in the order that pairs that odour with the arm's DAN. arms():
    punish_only drives the punishment DAN on the CS+ slot, reward_only the reward DAN on the CS- slot; order "ab" has
    CS+ = odour a (M0c), "ba" exchanges the odours. Naive counts do not depend on the arm or the order."""
    slot = "plus" if arm == "punish_only" else "minus"
    ab = [r for r in rows if r["arm"] == arm and r["order"] == "ab"]
    naive = {"a": float(np.median([r["pre"]["plus"][t] for r in ab])),
             "b": float(np.median([r["pre"]["minus"][t] for r in ab]))}
    m0c = "a" if slot == "plus" else "b"
    odour = m0c if naive["a"] == naive["b"] else max(naive, key=naive.get)
    order = "ab" if odour == m0c else "ba"
    mine = [r for r in rows if r["arm"] == arm and r["order"] == order]
    res = teach_type(mine, t, slot, spec)
    other = teach_type(mine, t, "minus" if slot == "plus" else "plus", spec)      # recorded, not judged
    return dict(res, arm=arm, odour=odour, order=order, naive_median=naive, untaught_n_decreased=other["n_decreased"])


def pick_readout(react: dict, teach: dict, pools: dict) -> dict:
    """A (P) = the pool's types that are reactive and teachable. Either empty -> dropped; two in one pool -> stop."""
    chosen = {k: [t for t in pools[k] if react[t]["passes"] and teach[t]["teachable"]] for k in ("A", "P")}
    if not chosen["A"] or not chosen["P"]:
        return dict(status=DROPPED_NO_READOUT, readout=None, passing=chosen)
    if len(chosen["A"]) > 1 or len(chosen["P"]) > 1:
        return dict(status=STOP_MULTI_TYPE, readout=None, passing=chosen)
    return dict(status=READOUT_SELECTED, readout={"A": chosen["A"][0], "P": chosen["P"][0]}, passing=chosen)


def z_constants(ref_rows: list, readout: dict, ddof: int) -> dict:
    """{"A": (mean, sd), "P": (mean, sd)} of the readout type's count (all its cells) over the reference presentations
    (H.4: 기준 집합의 순진 응답, 타입별 좌우 세포 합). F.3's frozen constants are population SDs (ddof 0)."""
    out = {}
    for k, t in readout.items():
        x = np.array([r["types"][t] for r in ref_rows], float)
        out[k] = (float(x.mean()), float(x.std(ddof=ddof)))
        if not out[k][1] > 0:
            raise ValueError(f"readout {t}: zero SD over the reference set")
    return out


# ================================================================ the oracle rows (G.14.4)
def check_rows(rows: list, expected: list) -> list:
    """G.14.4 row 1 for one combination and variant: the reasons the rows cannot be judged (empty = fine)."""
    reasons = []
    keys = [(r["axis"], int(r["turn"]), r["x"], r["y"]) for r in rows]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    if any(k[1] % 2 for k in keys):
        reasons.append("odd-turn rows present")
    exp = {tuple(k) for k in expected}
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    return reasons


def combo_stats(rows: list, z: dict, expected: list, spec) -> dict:
    """pair_stats for every row, the INVALID reasons and, when there are none, the aggregate."""
    reasons = check_rows(rows, expected)
    n = len(spec.report_seeds)
    short = [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows
             if any(len(r["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P"))]
    if short:
        reasons.append(f"{len(short)} pairs whose report probes are not the {n} report seeds")
    stats = {(r["axis"], int(r["turn"]), r["x"], r["y"]): pair_stats(r["report"], z, spec.testable_min) for r in rows}
    undefined = [k for k, s in stats.items() if s is None]
    if undefined:
        reasons.append(f"{len(undefined)} pairs with an undefined d'")
    nan = [k for k, s in stats.items() if s is not None and any(np.isnan(s[f]) for f in ("d_pre", "r", "p", "m"))]
    if nan:
        reasons.append(f"{len(nan)} pairs with a NaN statistic")
    agg = None if reasons else arm_aggregate(stats, spec.naive_max, spec.t_b_min, spec.f_a_min)
    return dict(reasons=reasons, aggregate=agg,
                pairs=[dict(axis=k[0], turn=k[1], x=k[2], y=k[3], **(s or {})) for k, s in stats.items()])


# ================================================================ records (not judged)
def _only(report: dict, keep: str) -> dict:
    """The report with the other readout type's counts set to 0: its term is then constant and cancels in V(X) - V(Y),
    so pair_stats measures the kept type alone (G.14.3: MBON13만·MBON05만의 d′)."""
    drop = "P" if keep == "A" else "A"
    return {ph: {keep: pr[keep], drop: [[0, 0] for _ in pr[drop]]} for ph, pr in report.items()}


def combo_records(rows: list, z: dict, spec) -> dict:
    """Per combination: the naive readout floor on the pairs' odours (mean count and zero share of each readout type over
    the report seeds and both candidates, G.14.6), the per-type statistics (G.14.3) and the testable (b) count on each
    half of the report seeds (the selection's seed noise). None of these enters the selection."""
    floor = {}
    for k in ("A", "P"):
        x = np.array([c for r in rows for pair in r["report"]["pre"][k] for c in pair], float)
        floor[k] = dict(mean=float(x.mean()), zero_share=float((x == 0).mean()), n=int(x.size))
    single = {}
    for k in ("A", "P"):
        st = {(r["axis"], int(r["turn"]), r["x"], r["y"]): pair_stats(_only(r["report"], k), z, spec.testable_min)
              for r in rows}
        ok = {key: s for key, s in st.items() if s is not None}
        single[k] = dict(aggregate=arm_aggregate(ok, spec.naive_max, spec.t_b_min, spec.f_a_min) if ok else None,
                         pairs=[dict(axis=key[0], turn=key[1], x=key[2], y=key[3], **s) for key, s in ok.items()])
    h = len(spec.report_seeds) // 2
    halves = []
    for part in (slice(0, h), slice(h, None)):
        st = [pair_stats({ph: {k: pr[k][part] for k in ("A", "P")} for ph, pr in r["report"].items()}, z,
                         spec.testable_min) for r in rows if r["axis"] == "b"]
        halves.append(dict(seeds=list(spec.report_seeds[part]), testable_b=sum(bool(s and s["testable"]) for s in st),
                           undefined=sum(s is None for s in st)))
    return dict(naive_floor=floor, single_type=single, report_halves=halves)


# ================================================================ selection (H.4 step 3)
def select(aggs: dict, spec) -> dict:
    """aggs: {combo: aggregate, or None for a combination without an oracle}, in spec.combos order. Eligible = F_a >= 2;
    the best T_b; every eligible combination within tie_pairs testable (b) pairs of the best is 'near', and the first
    of them in spec.combos (C0 > C1 > C3) is selected; no eligible combination or a best T_b < 0.5 -> stop. The stop looks
    at the best only, so the selected combination can itself be below 0.5 (H.4a.3-5, user decision): it is flagged."""
    elig = {c: a for c, a in aggs.items() if a is not None and a["F_a"] >= spec.f_a_min}
    if not elig:
        return dict(outcome=STOP_NO_ELIGIBLE, winner=None, eligible=[], near=[])
    top = max(a["testable_b"] for a in elig.values())
    top_t_b = max(a["T_b"] for a in elig.values())
    eligible = [c for c in spec.combos if c in elig]
    if top_t_b < spec.t_b_min:
        return dict(outcome=STOP_LOW_T_B, winner=None, eligible=eligible, near=[], top_T_b=top_t_b)
    near = [c for c in eligible if top - elig[c]["testable_b"] <= spec.tie_pairs]
    return dict(outcome=SELECTED, winner=near[0], eligible=eligible, near=near, top_T_b=top_t_b,
                engine_unchanged=bool(near[0] == "C0"), winner_below_bar=bool(elig[near[0]]["T_b"] < spec.t_b_min))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_formula.py tests/brain/test_h4_rules.py`
Expected: 39 passed.

- [ ] **Step 6: Full suite, then commit**

Run: `uv run pytest -q -rN -o addopts=""` → **441 passed, 1 skipped, 1 xfailed**.

```bash
git add flymon/brain/h4_formula.py flymon/brain/h4_rules.py tests/brain/test_h4_formula.py tests/brain/test_h4_rules.py
git commit -m "feat(brain): h4_formula and h4_rules — G.14's arithmetic with the readout's z, reselection, records and selection rules"
```

---

### Task 3: The worker jobs

**Files:**
- Create: `flymon/brain/h4_jobs.py`
- Test: `tests/brain/test_h4_jobs.py`

**Interfaces:**
- Consumes: `h4_formula.dprime`, `h4_formula.dv` (Task 2); `conditioning.arms`, `conditioning.train_block`; `presentation.decide`; `stimuli.design_odor_pair`, `stimuli.present`; `circuits.compartments`; `Engine`, `Plasticity`.
- Produces: `h4_jobs._RIG` (one entry per worker), `rig_for(conn, pops, params) -> (Engine, Plasticity, comps)`, `type_cells(conn, types) -> {type: indices}`; `teach_job(eng, pl, pops, comps, ro, params, seed, arm, order, types, punish_type, reward_type, k, odor_seed, strength, trials, present_ms, gap_ms, settle_ms, read_ms) -> dict` (a teach row, Task 2's shape); `oracle_job(eng, pl, pops, comps, ro, params, odor_x, odor_y, readout, z, types, act_seeds, select_seeds, report_seeds, alphas, strength, settle_ms, read_ms, window_ms, punish_type, reward_type) -> dict` (keys `select`, `alpha_reward`, `alpha_punish`, `counts` = per-type probes for pre/R1/R2, `report` = the readout's `{"A", "P"}` probes for pre/R1/R2, `kc`).

Both jobs reset the rig's weights and re-enable plasticity on the way out, whatever happens. The real-data equivalences were validated in a scratch worktree on HEAD `95ee094`: `teach_job` on C0, order "ab", reproduces `results/m0c/conditioning.json`'s A/P counts for seeds 8–15 in both arms bit for bit; `oracle_job` on C0 with readout MBON13 / MBON05 and F.3's constants reproduces G.12's E0 raw counts (pre and R1 at every α on 600–607, KC activity on 500–507) bit for bit on the first two pairs of each axis. Step R2 re-runs both on the committed code and adds the two comparisons the verdict's path needs: all 96 teaching arms against the H.4a.1 diagnostic, and G.14's S-ref smoke row (the punishment choice, R2 and the report seeds).

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_h4_jobs.py`:

```python
"""The H.4 worker jobs (spec H.4): the M0c arm with per-type counts, G.14's oracle with a chosen readout, the rig cache,
the weight reset and pool = in-process."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import h4_jobs as J
from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import Readout, run_arm
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h4_formula import dprime, dv
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide
from flymon.brain.stimuli import design_odor_pair, present

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")              # synthetic PPL105 core, then PAM08 core
TEACH = dict(types=TYPES, punish_type="PPL105", reward_type="PAM08", k=2, odor_seed=0, strength=3.0, trials=3,
             present_ms=150.0, gap_ms=50.0, settle_ms=50.0, read_ms=100.0)
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), strength=3.0, settle_ms=50.0, read_ms=100.0,
              window_ms=20, punish_type="PPL105", reward_type="PAM08")


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    return c, Populations.from_connectome(c)


def _fresh(c, pops, params=P):
    eng = Engine(c, pops, params, seed=0)
    comps = compartments(c, pops, params.core_frac)
    return eng, Plasticity(eng, pops, comps), comps


@pytest.mark.parametrize("arm", ["punish_only", "reward_only"])
@pytest.mark.parametrize("order", ["ab", "ba"])
def test_teach_job_is_run_arm_with_per_type_counts(conn_pops, arm, order):
    c, pops = conn_pops
    J._RIG.clear()
    got = J.teach_job(_Stub(c), None, pops, None, None, params=P, seed=9, arm=arm, order=order, **TEACH)
    eng, pl, comps = _fresh(c, pops)
    a, b = design_odor_pair(pops, k=2, seed=0)
    cs_plus, cs_minus = (a, b) if order == "ab" else (b, a)
    ref = run_arm(eng, pl, pops, Readout.from_compartments(comps), cs_plus, cs_minus, 3.0, 9, arm, trials=3,
                  present_ms=150.0, gap_ms=50.0, settle_ms=50.0, read_ms=100.0)["counts"]
    for ph in ("pre", "post"):
        for cs in ("plus", "minus"):
            g = got[ph][cs]
            assert (g["MBON03"] + g["MBON04"], g["MBON01"] + g["MBON02"]) == \
                (ref[f"{ph}_{cs}"]["A"], ref[f"{ph}_{cs}"]["P"]), (ph, cs)
    assert any(got["pre"][cs][t] for cs in ("plus", "minus") for t in TYPES), "the synthetic regime must drive MBONs"
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled


def _oracle_reference(c, pops, odor_x, odor_y):
    """G.14.3's call sequence (m2_engine_probe.oracle_job) written out on a fresh rig, with this readout and z."""
    eng, pl, comps = _fresh(c, pops)
    t = np.asarray(c.type).astype(str)
    ia, ip = np.flatnonzero(t == READOUT["A"]), np.flatnonzero(t == READOUT["P"])
    idx = np.concatenate([ia, ip])
    pl.reset_weights(); pl.set_enabled(False)
    fired = np.zeros(len(pops.kc))
    for s in ORACLE["act_seeds"]:
        eng.reset(s); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan(); present(eng, pops, odor_x, 3.0)
        eng.run(50.0); fired += eng.run(100.0)[pops.kc] > 0
    fx = fired / len(ORACLE["act_seeds"])

    def probe(seeds):
        out = [decide(eng, pl, pops, [odor_x, odor_y], 3.0, s, 50.0, 100.0, idx=idx) for s in seeds]
        return {"A": [o[:, :len(ia)].sum(1).tolist() for o in out], "P": [o[:, len(ia):].sum(1).tolist() for o in out]}

    rew = np.isin(pl.post_mb, pl.mb_local[comps["PAM08"].core]); pun = np.isin(pl.post_mb, pl.mb_local[comps["PPL105"].core])

    def set_w(a_r, a_p=None):
        wv = pl.w0.copy()
        if a_r is not None:
            wv[rew] = pl.w0[rew] * (1.0 - a_r * fx)[pl.pre_kc[rew]]
        if a_p is not None:
            wv[pun] = pl.w0[pun] * (1.0 - a_p * fx)[pl.pre_kc[pun]]
        eng.csc.w[pl.edges] = wv

    set_w(None); pre_sel = probe(ORACLE["select_seeds"])
    r1 = {}
    for a in ORACLE["alphas"]:
        set_w(a); r1[a] = probe(ORACLE["select_seeds"])
    a_r = max(ORACLE["alphas"], key=lambda a: (dprime(dv(r1[a], Z) - dv(pre_sel, Z)), -a))
    ch = {}
    for a in ORACLE["alphas"]:
        set_w(a_r, a); ch[a] = dprime(dv(probe(ORACLE["select_seeds"]), Z) - dv(r1[a_r], Z))
    a_p = min(ORACLE["alphas"], key=lambda a: (ch[a], a))
    set_w(None); pre = probe(ORACLE["report_seeds"])
    set_w(a_r); R1 = probe(ORACLE["report_seeds"])
    set_w(a_r, a_p); R2 = probe(ORACLE["report_seeds"])
    return a_r, a_p, {"pre": pre, "R1": R1, "R2": R2}


def test_oracle_job_is_g14s_sequence_with_the_given_readout(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    got = J.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    a_r, a_p, rep = _oracle_reference(c, pops, a, b)
    assert (got["alpha_reward"], got["alpha_punish"]) == (a_r, a_p)
    assert got["report"] == rep
    assert got["counts"]["pre"]["MBON03"] == rep["pre"]["A"] and set(got["counts"]["pre"]) == set(TYPES)
    assert any(x for row in rep["pre"]["P"] for x in row), "the synthetic regime must drive the readout"
    assert len(got["kc"]["x"]["frac"]) == 2 and 0.0 <= got["kc"]["jaccard"] <= 1.0
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled


def test_the_rig_is_built_once_per_params(conn_pops):
    c, pops = conn_pops
    J._RIG.clear()
    kw = dict(seed=9, arm="punish_only", order="ab", **TEACH)
    first = J.teach_job(_Stub(c), None, pops, None, None, params=P, **kw)
    rig = J._RIG[P]
    assert J.teach_job(_Stub(c), None, pops, None, None, params=P, **kw) == first and J._RIG[P] is rig
    other = dataclasses.replace(P, kc_thresh=0.6)
    J.teach_job(_Stub(c), None, pops, None, None, params=other, **kw)
    assert list(J._RIG) == [other]


def test_pool_equals_in_process(conn_pops, synthetic_npz):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    teach = J.teach_job(_Stub(c), None, pops, None, None, params=P, seed=9, arm="reward_only", order="ba", **TEACH)
    oracle = J.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    with FlyPool(synthetic_npz, Params(), [{}], workers=1) as pool:
        t = pool.run_jobs(J.teach_job, [dict(params=P, seed=9, arm="reward_only", order="ba", **TEACH)])[0]
        o = pool.run_jobs(J.oracle_job, [dict(params=P, odor_x=a, odor_y=b, **ORACLE)])[0]
    assert t == teach and o["report"] == oracle["report"] and o["counts"] == oracle["counts"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_jobs.py`
Expected: collection error — `No module named 'flymon.brain.h4_jobs'`.

- [ ] **Step 3: Write the jobs**

`flymon/brain/h4_jobs.py`:

```python
"""FlyPool worker jobs for the M0d H.4 runner (spec appendix H.4): the conditioning arms of the readout reselection and
the oracle.

Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them.
Every job builds (or reuses) its own rig — Engine, Plasticity, compartments — from the `Params` it is given
(h3_jobs' pattern); the worker's default engine only lends its connectome. The call sequences are the committed
ones, so the recorded numbers reproduce: `teach_job` is `conditioning.run_arm` (the M0c arm, as
`pool_jobs.conditioning_arm_job` runs it) with per-type counts; `oracle_job` is `m2_engine_probe.oracle_job` (G.14.3)
with the readout types, their z constants and the combination's engine passed in instead of MBON13 / MBON05, F.3's
constants and `configure()`.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .circuits import compartments
from .conditioning import arms, train_block
from .engine_cpu import Engine
from .h4_formula import dprime, dv
from .plasticity import Plasticity
from .presentation import decide
from .stimuli import design_odor_pair, present

_RIG: dict = {}          # Params -> (Engine, Plasticity, comps); one entry per worker


def rig_for(conn, pops, params):
    """The worker's engine, plasticity and compartments for `params`, built as FlyPool's worker builds them
    (fly_pool._WorkerState.get) and reused while jobs keep asking for the same Params."""
    if params not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[params] = (eng, Plasticity(eng, pops, comps), comps)
    return _RIG[params]


def type_cells(conn, types) -> dict:
    t = np.asarray(conn.type).astype(str)
    return {n: np.flatnonzero(t == n) for n in types}


# ================================================================ readout reselection: the M0c single-channel arms
def teach_job(eng, pl, pops, comps, ro, params, seed: int, arm: str, order: str, types, punish_type: str,
              reward_type: str, k: int, odor_seed: int, strength: float, trials: int, present_ms: float, gap_ms: float,
              settle_ms: float, read_ms: float) -> dict:
    """conditioning.run_arm at one seed — probes (settle, read) of CS+ and CS- with the probe seed, the training block,
    the probes again — with every listed MBON type's count. order "ab": CS+ = odour a, CS- = odour b (M0c);
    "ba" exchanges them. The punish type is paired with the CS+ slot, the reward type with the CS- slot (arms())."""
    e, p, c = rig_for(eng.conn, pops, params)
    cells = type_cells(e.conn, types)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    cs_plus, cs_minus = (a, b) if order == "ab" else (b, a)

    def probe(odor):
        was = p.enabled
        p.set_enabled(False)
        e.reset(seed); p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, odor, strength)
        e.run(settle_ms)
        counts = e.run(read_ms)
        p.set_enabled(was)
        return {n: int(counts[i].sum()) for n, i in cells.items()}

    punish, reward, plastic = arms(punish_type, reward_type)[arm]
    try:
        p.reset_weights()
        pre = {"plus": probe(cs_plus), "minus": probe(cs_minus)}
        p.set_enabled(plastic)
        train_block(e, p, pops, cs_plus, cs_minus, strength, seed, punish, reward, trials, present_ms, gap_ms, settle_ms)
        p.set_enabled(True)
        post = {"plus": probe(cs_plus), "minus": probe(cs_minus)}
    finally:
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), arm=arm, order=order, pre=pre, post=post)


# ================================================================ the oracle (G.14.3)
def _present_kc(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms: int) -> dict:
    """m2_engine_probe.present_kc: one presentation; KC counts of the read window and the maximum sliding-window KC
    count over the whole presentation (G.8's window, recorded)."""
    kc = pops.kc
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    e.reset(seed); p.reset_traces(); e.clear_drive(); p.quiet_dan()
    present(e, pops, odor, strength)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32)
    n_settle, n_total = int(round(settle_ms / e.p.dt)), int(round((settle_ms + read_ms) / e.p.dt))
    for step in range(n_total):
        f = pos[e.step()]; f = f[f >= 0]
        win[f] += 1; hist.append(f)
        if len(hist) > window_ms:
            win[hist.popleft()] -= 1
        if f.size:
            max_win = max(max_win, int(win.max()))
        if step >= n_settle:
            read[f] += 1
    return {"read": read, "max_win": max_win}


def oracle_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, readout: dict, z: dict, types,
               act_seeds, select_seeds, report_seeds, alphas, strength: float, settle_ms: float, read_ms: float,
               window_ms: int, punish_type: str, reward_type: str) -> dict:
    """One pair: KC activity f_i of X on act_seeds; alpha picked on select_seeds (reward change max, then punishment
    change min, ties -> smaller alpha); pre / R1 / R2 on report_seeds. Probe counts are kept per listed type; V is
    built from readout = {"A": type, "P": type} with z = {"A": (mean, sd), "P": (mean, sd)}."""
    e, p, c = rig_for(eng.conn, pops, params)
    cells = type_cells(e.conn, types)
    idx = np.concatenate([cells[n] for n in types])
    bounds = np.cumsum([0] + [len(cells[n]) for n in types])
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win = [], [], []
            for s in act_seeds:
                o = _present_kc(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}

        def probe(seeds):
            out = {n: [] for n in types}
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                for j, n in enumerate(types):
                    out[n].append(cnt[:, bounds[j]:bounds[j + 1]].sum(1).tolist())
            return out

        def ap(pr):
            return {"A": pr[readout["A"]], "P": pr[readout["P"]]}

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
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
        w[p.edges] = p.w0
        return {"select": {"pre": pre_sel, "reward": reward, "punish": punish},
                "alpha_reward": a_r, "alpha_punish": a_p,
                "counts": {"pre": pre, "R1": R1, "R2": R2},
                "report": {"pre": ap(pre), "R1": ap(R1), "R2": ap(R2)},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}
    finally:
        p.reset_weights(); p.set_enabled(True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_jobs.py`
Expected: 7 passed.

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q -rN -o addopts=""` → **448 passed, 1 skipped, 1 xfailed**.

```bash
git add flymon/brain/h4_jobs.py tests/brain/test_h4_jobs.py
git commit -m "feat(brain): h4_jobs — the M0c arm with per-type counts and G.14's oracle with a chosen readout"
```

---

### Task 4: The measurer and the procedure

**Files:**
- Create: `flymon/brain/h4_measure.py`, `flymon/brain/h4_runner.py`
- Test: `tests/brain/h4_scripted.py`, `tests/brain/test_h4_measure.py`, `tests/brain/test_h4_runner.py`

**Interfaces:**
- Consumes: `h4_jobs.teach_job`, `h4_jobs.oracle_job` (Task 3); every `h4_rules` function (Task 2); `h3_store.MEASURE_FILES`, `h3_store.MeasureCache.get_or_compute(kind, inputs, compute, params_list)`; an H.3 measurer with `reference(params)` and `rest(params, seeds)` (`h3_measure.PoolMeasurer`).
- Produces: `h4_measure.MEASURE_FILES`, `h4_measure.HASHED_FILES`, `h4_measure.H4Measurer(pool, spec, pairs, pools, cache, h3_measurer)` with `reference(params)`, `rest(params, seeds)`, `teach(params) -> list[teach row]`, `oracle(params, readout, z) -> list[oracle row]` (pair order; each row = the job's result plus the pair's `axis`, `turn`, `x`, `y`), `params_seen`; `h4_runner.Context(spec, combos, pools, probe_seeds, expected, h3_guard, log)`, `reselect(m, ctx, name, params) -> dict`, `run_oracle(m, ctx, name, params, r) -> dict` (adds `oracle` = `combo_stats(...)` and `records` = `combo_records(...)` to a reselected combination), `run_h4(m, ctx) -> dict` (keys `combos`, `selection`, `outcome`, and `invalid` when it applies; every combination is reselected before the first oracle).

`H4Measurer.oracle` asks the cache for every pair first (a stand-in `compute` that raises marks the missing ones), runs the missing pairs in rounds of `pool.n_workers`, and stores each result as its own entry — the pair's labels are merged last so a job's own keys can never overwrite them (a scratch test caught that order).

- [ ] **Step 1: Write the scripted measurer and the failing tests**

`tests/brain/h4_scripted.py`:

```python
"""A scripted stand-in for h4_measure.H4Measurer: every row is a simple function of the combination, so the H.4
procedure's branches can be steered one at a time without running the engine. Combinations are told apart by their
Params' kc_thresh (C0 1.5, C1 1.6, C3 1.7)."""
from flymon.brain.config import Params
from flymon.brain.h4_spec import SPEC

POOLS = {"A": ["MA1", "MA2"], "P": ["MP1", "MP2"]}
TYPES = POOLS["A"] + POOLS["P"]
COMBOS = {"C0": Params(), "C1": Params(kc_thresh=1.6), "C3": Params(kc_thresh=1.7)}
NAME = {p: n for n, p in COMBOS.items()}
SEEDS = [1000 + j for j in range(8)]                     # the reference set's seeds (8 presentations here)
PAIRS = ([("a", 2 * (j % 4), f"ax{j}", f"ay{j}") for j in range(4)]
         + [("b", 2 * (j % 4), f"bx{j}", f"by{j}") for j in range(5)])


def report(testable=True, n=8):
    """d_pre 0; testable: R1 lowers P(X) by 20 + (s % 2), R2 lowers A(X) the same (m ~ 29); else no change (m = 0)."""
    step = [20 + (s % 2) for s in range(n)] if testable else [0] * n
    pre = {"A": [[20 + (1 if s % 2 else -1), 20] for s in range(n)], "P": [[40, 40]] * n}
    r1 = {"A": pre["A"], "P": [[40 - d, 40] for d in step]}
    r2 = {"A": [[a - d, b] for (a, b), d in zip(pre["A"], step)], "P": r1["P"]}
    return {"pre": pre, "R1": r1, "R2": r2}


def guard_of(scripted, name):
    """The H.3 guard statistics the scripted reference/rest rows give (what block "h3" would hold)."""
    from flymon.brain.h3_rules import mbon_type_stats
    p = COMBOS[name]
    ref, rest = scripted.reference(p), {r["seed"]: r for r in scripted.rest(p, SEEDS)}
    scripted.calls.clear()
    return {t: mbon_type_stats(ref, rest, t, 5.0, 0.25) for t in TYPES}


class Scripted:
    def __init__(self, react=None, teach=None, testable=None, rows=None):
        """react(name) -> set of reactive types; teach(name) -> set of teachable types; testable(name) -> (n_b of 5
        testable, n_a of 4 testable and naive); rows(name, rows) -> rows (to corrupt them)."""
        self.react_fn = react or (lambda n: {"MA1", "MP1"})
        self.teach_fn = teach or (lambda n: set(TYPES))
        self.testable_fn = testable or (lambda n: (3, 2))
        self.rows_fn = rows or (lambda n, r: r)
        self.calls = []

    def reference(self, params):
        self.calls.append(("reference", NAME[params]))
        on = self.react_fn(NAME[params])
        return [dict(odor=f"R{j:02d}", seed=s, types={t: (10 + j if t in on else 0) for t in TYPES})
                for j, s in enumerate(SEEDS)]

    def rest(self, params, seeds):
        self.calls.append(("rest", NAME[params]))
        return [dict(seed=int(s), types={t: 1 for t in TYPES}) for s in seeds]

    def teach(self, params):
        self.calls.append(("teach", NAME[params]))
        ok = self.teach_fn(NAME[params])
        rows = []
        for order in ("ab", "ba"):
            for arm in ("punish_only", "reward_only"):
                for s in SPEC.teach_seeds:
                    pre = {cs: {t: 10 for t in TYPES} for cs in ("plus", "minus")}
                    post = {cs: {t: (5 if t in ok else 10) for t in TYPES} for cs in ("plus", "minus")}
                    rows.append(dict(seed=s, arm=arm, order=order, pre=pre, post=post))
        return rows

    def oracle(self, params, readout, z):
        name = NAME[params]
        self.calls.append(("oracle", name, tuple(sorted(readout.items()))))
        n_b, n_a = self.testable_fn(name)
        rows, ia, ib = [], 0, 0
        for ax, t, x, y in PAIRS:
            ok = (ia < n_a) if ax == "a" else (ib < n_b)
            ia, ib = ia + (ax == "a"), ib + (ax == "b")
            rows.append(dict(axis=ax, turn=t, x=x, y=y, report=report(ok)))
        return self.rows_fn(name, rows)
```

`tests/brain/test_h4_measure.py`:

```python
"""The H.4 measurement layer: teach and the oracle through the content-addressed cache (one entry per oracle pair, run
in rounds of one pair per worker), reference and rest delegated to H.3's measurer."""
import pytest

from flymon.brain import h4_jobs
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_spec import SPEC

PAIRS = [dict(axis=ax, turn=2 * j, x=f"{ax}x{j}", y=f"{ax}y{j}", odor_x={"ORN_A": 1.0}, odor_y={"ORN_B": 1.0 + j})
         for ax in ("a", "b") for j in range(3)]
POOLS = {"A": ["MA1"], "P": ["MP1"]}


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kwargs_list):
        self.calls.append((fn.__name__, len(kwargs_list)))
        return [dict(tag=fn.__name__, b=kw.get("odor_y", {}).get("ORN_B"), seed=kw.get("seed"), x="job's own x")
                for kw in kwargs_list]


class FakeH3:
    def __init__(self):
        self.calls = []

    def reference(self, params):
        self.calls.append("reference"); return ["ref"]

    def rest(self, params, seeds):
        self.calls.append(("rest", tuple(seeds))); return ["rest"]


@pytest.fixture
def m(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    cache = MeasureCache("results/m0d/h4/cache", dict(key="k" * 64), "run")
    return H4Measurer(pool, SPEC, PAIRS, POOLS, cache, FakeH3())


def test_the_oracle_runs_missing_pairs_in_rounds_and_resumes_per_pair(m, tmp_path):
    p = Params(kc_thresh=1.6)
    z = {"A": (1.0, 2.0), "P": (3.0, 4.0)}
    rows = m.oracle(p, {"A": "MA1", "P": "MP1"}, z)
    assert m.pool.calls == [("oracle_job", 2), ("oracle_job", 2), ("oracle_job", 2)]
    assert [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows] == [(q["axis"], q["turn"], q["x"], q["y"]) for q in PAIRS]
    assert [r["b"] for r in rows] == [q["odor_y"]["ORN_B"] for q in PAIRS] and rows[0]["tag"] == "oracle_job"
    files = sorted((tmp_path / "results/m0d/h4/cache/oracle").glob("*.json"))
    assert len(files) == len(PAIRS) and m.cache.misses == len(PAIRS)
    m.pool.calls.clear()
    assert m.oracle(p, {"A": "MA1", "P": "MP1"}, z) == rows and m.pool.calls == []
    files[0].unlink()
    assert m.oracle(p, {"A": "MA1", "P": "MP1"}, z) == rows and m.pool.calls == [("oracle_job", 1)]
    assert m.oracle(p, {"A": "MA1", "P": "MP1"}, {"A": (1.0, 2.5), "P": (3.0, 4.0)}) != [] and \
        m.pool.calls[-3:] == [("oracle_job", 2)] * 3                   # other z constants are another measurement


def test_teach_runs_every_arm_order_and_seed_once_and_is_cached(m):
    p = Params()
    rows = m.teach(p)
    assert m.pool.calls == [("teach_job", 2 * 2 * len(SPEC.teach_seeds))]
    assert len(rows) == 32 and m.teach(p) == rows and len(m.pool.calls) == 1


def test_reference_and_rest_are_h3s_measurements(m):
    p = Params(kc_thresh=1.7)
    assert m.reference(p) == ["ref"] and m.rest(p, [5, 6]) == ["rest"]
    assert m.h3.calls == ["reference", ("rest", (5, 6))] and m.params_seen == [p]
```

`tests/brain/test_h4_runner.py`:

```python
"""The H.4 procedure on a scripted measurer (spec H.4 as amended by H.3a.1, H.3a.9 and H.4a): every branch."""
import numpy as np
import pytest

from flymon.brain import h4_rules as R
from flymon.brain.h4_runner import Context, run_h4
from flymon.brain.h4_spec import SPEC

from h4_scripted import COMBOS, PAIRS, POOLS, SEEDS, Scripted, guard_of


def ctx_for(m, **kw):
    guards = {n: guard_of(m, n) for n in COMBOS}
    return Context(spec=SPEC, combos=dict(COMBOS), pools=POOLS, probe_seeds=SEEDS, expected=list(PAIRS),
                   h3_guard=guards, log=lambda s: None, **kw)


def run(m):
    return run_h4(m, ctx_for(m))


def test_equal_combinations_select_c0_and_every_combination_is_measured_in_order():
    m = Scripted()
    out = run(m)
    assert out["outcome"] == R.SELECTED and out["selection"]["winner"] == "C0" and out["selection"]["engine_unchanged"]
    assert out["selection"]["near"] == ["C0", "C1", "C3"]
    assert [c for c in m.calls if c[0] == "oracle"] == [("oracle", n, (("A", "MA1"), ("P", "MP1"))) for n in COMBOS]
    first_oracle = next(i for i, c in enumerate(m.calls) if c[0] == "oracle")
    assert {c[1] for c in m.calls[:first_oracle] if c[0] == "teach"} == set(COMBOS)      # every reselection first
    c0 = out["combos"]["C0"]
    assert c0["readout"] == {"A": "MA1", "P": "MP1"} and c0["oracle"]["aggregate"]["testable_b"] == 3
    assert set(c0["records"]) == {"naive_floor", "single_type", "report_halves"}
    assert out["selection"]["winner_below_bar"] is False
    x = np.arange(10, 18, dtype=float)
    assert c0["z"]["A"] == (float(x.mean()), float(x.std()))
    assert c0["teach"]["MA1"]["order"] == "ab" and c0["teach"]["MA1"]["teachable"]


def test_the_best_t_b_wins_outside_the_tie_band():
    m = Scripted(testable=lambda n: {"C0": (2, 2), "C1": (5, 2), "C3": (3, 2)}[n])
    out = run(m)
    assert out["selection"]["winner"] == "C1" and out["selection"]["near"] == ["C1", "C3"]


def test_a_combination_without_a_readout_is_dropped_and_gets_no_oracle():
    m = Scripted(teach=lambda n: {"MP1", "MP2"} if n == "C0" else {"MA1", "MA2", "MP1", "MP2"})
    out = run(m)
    assert out["combos"]["C0"]["status"] == R.DROPPED_NO_READOUT and "oracle" not in out["combos"]["C0"]
    assert ("oracle", "C0") not in [c[:2] for c in m.calls]
    assert out["selection"]["eligible"] == ["C1", "C3"] and out["selection"]["winner"] == "C1"


def test_two_passing_types_in_a_pool_stop_the_run():
    m = Scripted(react=lambda n: {"MA1", "MA2", "MP1"} if n == "C1" else {"MA1", "MP1"})
    out = run(m)
    assert out["outcome"] == R.STOP_MULTI_TYPE and list(out["combos"]) == ["C0", "C1"]
    assert out["selection"] is None and not [c for c in m.calls if c[0] == "oracle"]      # before any oracle


def test_invalid_rows_make_the_run_invalid_without_a_selection():
    m = Scripted(rows=lambda n, rows: rows[:-1] if n == "C3" else rows)
    out = run(m)
    assert out["outcome"] == R.INVALID and set(out["invalid"]) == {"C3"} and out["selection"] is None


@pytest.mark.parametrize("testable, outcome", [((2, 2), R.STOP_LOW_T_B), ((5, 1), R.STOP_NO_ELIGIBLE)])
def test_the_stops_for_the_users_decision(testable, outcome):
    out = run(Scripted(testable=lambda n: testable))
    assert out["outcome"] == outcome and out["selection"]["winner"] is None


def test_reactivity_must_be_the_h3_guard():
    m = Scripted()
    ctx = ctx_for(m)
    ctx.h3_guard["C3"]["MA1"] = dict(ctx.h3_guard["C3"]["MA1"], zero_share=0.5)
    with pytest.raises(RuntimeError, match="differs from the H.3 guard"):
        run_h4(m, ctx)
    assert not [c for c in m.calls if c[0] == "oracle"]                              # found before any oracle
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_measure.py tests/brain/test_h4_runner.py`
Expected: collection errors — `No module named 'flymon.brain.h4_measure'` / `'flymon.brain.h4_runner'`.

- [ ] **Step 3: Write the measurer**

`flymon/brain/h4_measure.py`:

```python
"""The measurement layer of the M0d H.4 runner: the reference and rest measurements come from H.3's measurer and cache
(the H.3 guard's own entries), the conditioning arms and the oracle from H.4's cache. The procedure (h4_runner) only
sees this interface, so its tests substitute a scripted measurer and never run the engine.

Resume (spec H.3a.12's form, carried over): every measurement is cached under a content key — `h3_store.MeasureCache`
keyed by the files a measurement's result depends on (`MEASURE_FILES`), the NPZ, the versions and the inputs. The
oracle is cached per pair and run in rounds of one pair per worker, so an interrupted run loses at most one round.
"""
from __future__ import annotations

from . import h4_jobs
from .h3_store import MEASURE_FILES as H3_MEASURE_FILES

MEASURE_FILES = ("uv.lock", "flymon/brain/circuits.py", "flymon/brain/config.py", "flymon/brain/connectome.py",
                 "flymon/brain/engine_cpu.py", "flymon/brain/fly_pool.py", "flymon/brain/stimuli.py",
                 "flymon/brain/thresholds.py", "flymon/brain/plasticity.py", "flymon/brain/presentation.py",
                 "flymon/brain/conditioning.py", "flymon/brain/h3_store.py", "flymon/brain/h4_formula.py",
                 "flymon/brain/h4_jobs.py", "flymon/brain/h4_measure.py")
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + H3_MEASURE_FILES + (
    "flymon/brain/h3_rules.py", "flymon/brain/h3_spec.py", "flymon/brain/pool_bench.py", "flymon/brain/h4_spec.py",
    "flymon/brain/h4_pairs.py", "flymon/brain/h4_rules.py", "flymon/brain/h4_runner.py", "scripts/run_m0d_h4.py")))


class _Missing(Exception):
    pass


def _missing(*_):
    raise _Missing


class H4Measurer:
    def __init__(self, pool, spec, pairs: list, pools: dict, cache, h3_measurer):
        """pairs: h4_pairs.even_pairs; pools: {"A": [types], "P": [types]}; h3_measurer: h3_measure.PoolMeasurer on
        H.3's cache (reference / rest)."""
        self.pool, self.spec, self.pairs, self.pools = pool, spec, pairs, pools
        self.cache, self.h3 = cache, h3_measurer
        self.types = [t for k in ("A", "P") for t in pools[k]]
        self.params_seen: list = []

    def _seen(self, params):
        if params not in self.params_seen:
            self.params_seen.append(params)

    def reference(self, params) -> list:
        self._seen(params)
        return self.h3.reference(params)

    def rest(self, params, seeds) -> list:
        self._seen(params)
        return self.h3.rest(params, seeds)

    def teach(self, params) -> list:
        """Both single-channel arms, every declared odour order and seed: one cache entry per combination."""
        s, h3 = self.spec, self.spec.h3
        self._seen(params)
        common = dict(params=params, types=tuple(self.types), punish_type=h3.punish_type, reward_type=h3.reward_type,
                      k=h3.design_k, odor_seed=h3.design_odor_seed, strength=h3.strength, trials=s.teach_trials,
                      present_ms=s.teach_present_ms, gap_ms=s.teach_gap_ms, settle_ms=s.teach_window.settle_ms,
                      read_ms=s.teach_window.read_ms)
        jobs = [dict(common, seed=int(seed), arm=arm, order=order) for order in s.teach_orders
                for arm in ("punish_only", "reward_only") for seed in s.teach_seeds]
        inputs = dict(common, jobs=[(j["seed"], j["arm"], j["order"]) for j in jobs])
        return self.cache.get_or_compute("teach", inputs, lambda: self.pool.run_jobs(h4_jobs.teach_job, jobs), [params])

    def oracle(self, params, readout: dict, z: dict) -> list:
        """One row per pair ({axis, turn, x, y} + oracle_job's result), in pair order."""
        s, h3 = self.spec, self.spec.h3
        self._seen(params)
        common = dict(params=params, readout=dict(readout), z={k: tuple(v) for k, v in z.items()},
                      types=tuple(self.types), act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds),
                      report_seeds=tuple(s.report_seeds), alphas=tuple(s.oracle_alphas), strength=h3.strength,
                      settle_ms=s.oracle_window.settle_ms, read_ms=s.oracle_window.read_ms, window_ms=s.kc_window_ms,
                      punish_type=h3.punish_type, reward_type=h3.reward_type)
        items = [(p, dict(common, odor_x=p["odor_x"], odor_y=p["odor_y"])) for p in self.pairs]
        inputs = lambda p, kw: dict(kw, pair=[p["axis"], int(p["turn"]), p["x"], p["y"]])
        done = {}
        for i, (p, kw) in enumerate(items):
            try:
                done[i] = self.cache.get_or_compute("oracle", inputs(p, kw), _missing, [params])
            except _Missing:
                pass
        todo = [i for i in range(len(items)) if i not in done]
        n = max(1, self.pool.n_workers)
        for r in range(0, len(todo), n):
            batch = todo[r:r + n]
            res = self.pool.run_jobs(h4_jobs.oracle_job, [items[i][1] for i in batch])
            for i, out in zip(batch, res):
                done[i] = self.cache.get_or_compute("oracle", inputs(*items[i]), lambda out=out: out, [params])
        return [done[i] | {k: items[i][0][k] for k in ("axis", "turn", "x", "y")} for i in range(len(items))]
```

- [ ] **Step 4: Write the procedure**

`flymon/brain/h4_runner.py`:

```python
"""M0d H.4 procedure (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a): per combination the readout reselection,
the z constants and the oracle; then the selection over the combinations. H.4a.2 took the APL->MBON ablation of
H.3a.9 (2) out of the verdict, so there is no ablated oracle; its reason travels with the result (SPEC.notes).

    every combination first: reactivity (the H.3 guard measurement) + teachability (M0c arms) -> readout (A, P), z
       |- A or P empty              -> dropped_no_readout: no oracle, not eligible
       '- two types pass in a pool  -> stop_multiple_types: the run stops before any oracle (H.4 does not combine them)
    then every combination with a readout: oracle -> statistics and records
    INVALID rows? -> selection (F_a >= 2; top T_b; tie band; C0 > C1 > C3)

Reselecting every combination before the first oracle makes a reactivity mismatch or a stop surface within minutes,
not after an hour of oracle work.

The measurer is an interface (h4_measure.H4Measurer in a run, a scripted stand-in in the tests).
"""
from __future__ import annotations

import dataclasses

from .h4_rules import (INVALID, READOUT_SELECTED, STOP_MULTI_TYPE, combo_records, combo_stats, pick_readout, reactivity,
                       select, teach_choice, z_constants)


@dataclasses.dataclass
class Context:
    spec: object
    combos: dict                     # {name: adopted Params} in spec.combos order
    pools: dict                      # {"A": [MBON types], "P": [MBON types]}
    probe_seeds: list                # the reference set's seeds, generator order (the rest measurement's seeds)
    expected: list                   # [(axis, turn, x, y)] of the pair list
    h3_guard: dict                   # {name: {type: H.3 guard stats}} from block "h3": reactivity must equal it
    log: object = print


def reselect(m, ctx: Context, name: str, params) -> dict:
    spec = ctx.spec
    ref = m.reference(params)
    rest_by_seed = {r["seed"]: r for r in m.rest(params, ctx.probe_seeds)}
    types = [t for k in ("A", "P") for t in ctx.pools[k]]
    react = reactivity(ref, rest_by_seed, types, spec)
    for t in types:                          # the same measurement as H.3's guard, so the same numbers
        g = ctx.h3_guard[name].get(t)
        if g is None or (react[t]["median_delta"], react[t]["zero_share"]) != (g["median_delta"], g["zero_share"]):
            raise RuntimeError(f"{name} {t}: reactivity {react[t]} differs from the H.3 guard {g}")
    rows = m.teach(params)
    teach = {t: teach_choice(rows, t, "punish_only" if t in ctx.pools["A"] else "reward_only", spec) for t in types}
    pick = pick_readout(react, teach, ctx.pools)
    z = z_constants(ref, pick["readout"], spec.z_ddof) if pick["status"] == READOUT_SELECTED else None
    return dict(reactivity=react, teach=teach, **pick, z=z)


def run_oracle(m, ctx: Context, name: str, params, r: dict) -> dict:
    rows = m.oracle(params, r["readout"], r["z"])
    r["oracle"] = combo_stats(rows, r["z"], ctx.expected, ctx.spec)
    r["records"] = combo_records(rows, r["z"], ctx.spec)
    agg = r["oracle"]["aggregate"]
    ctx.log(f"{name} oracle: " + (f"T_b {agg['testable_b']}/{agg['n_b']}  F_a {agg['F_a']}" if agg
                                  else f"INVALID {r['oracle']['reasons']}"))
    return r


def run_h4(m, ctx: Context) -> dict:
    out = dict(combos={}, selection=None, outcome=None)
    for name, params in ctx.combos.items():
        r = out["combos"][name] = reselect(m, ctx, name, params)
        ctx.log(f"{name}: {r['status']} readout {r['readout']} z {r['z']}")
        if r["status"] == STOP_MULTI_TYPE:
            out["outcome"] = STOP_MULTI_TYPE
            return out
    for name, params in ctx.combos.items():
        if out["combos"][name]["status"] == READOUT_SELECTED:
            run_oracle(m, ctx, name, params, out["combos"][name])
    bad = {n: c["oracle"]["reasons"] for n, c in out["combos"].items()
           if c["status"] == READOUT_SELECTED and c["oracle"]["reasons"]}
    if bad:
        out.update(outcome=INVALID, invalid=bad)
        return out
    live = {n: c for n, c in out["combos"].items() if c["status"] == READOUT_SELECTED}
    out["selection"] = select({n: (live[n]["oracle"]["aggregate"] if n in live else None) for n in ctx.combos}, ctx.spec)
    out["outcome"] = out["selection"]["outcome"]
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -o addopts="" tests/brain/test_h4_measure.py tests/brain/test_h4_runner.py`
Expected: 11 passed.

- [ ] **Step 6: Full suite, then commit**

Run: `uv run pytest -q -rN -o addopts=""` → **459 passed, 1 skipped, 1 xfailed**.

```bash
git add flymon/brain/h4_measure.py flymon/brain/h4_runner.py tests/brain/h4_scripted.py tests/brain/test_h4_measure.py tests/brain/test_h4_runner.py
git commit -m "feat(brain): h4_measure and h4_runner — H.3's cache for reactivity, per-pair oracle cache, the H.4 procedure"
```

---

### Task 5: The CLI, the end-to-end test and the reproduction check

**Files:**
- Create: `scripts/run_m0d_h4.py`, `docs/superpowers/specs/m0d-diag/h4_reproduction_check.py`
- Test: `tests/test_run_m0d_h4.py`

**Interfaces:**
- Consumes: everything above; `h3_measure.PoolMeasurer(pool, spec, odors, all51, cache, diag_cache)` (only `reference` and `rest` are called, so `all51` and `diag_cache` are `None`); `h3_spec.make_odors`; `h3_store.code_key`, `git_state`, `MeasureCache`, `write_json`, `write_bytes`, `replace_summary_block`, `sha256_file`, `ROOT`.
- Produces: `run_m0d_h4.main(argv=None, spec=None, summary_spec=SPEC, require_root=True, pairs=None, pools=None) -> int` (`pairs` / `pools` exist for the synthetic-connectome test only; `pools` replaces the core pools after they are checked against block `"h3"`), `run_m0d_h4.adopted(summary, spec) -> (combos, guards)` (ValueError when the block cannot be used), `run_m0d_h4.refuse(msg) -> 2`, `run_m0d_h4.params_from_json(d) -> Params`, `run_m0d_h4.smoke_spec(spec)`, `run_m0d_h4.md_report(res)`, `run_m0d_h4.C3_COPY`, `run_m0d_h4.POOL_TIMEOUT_S`.

The end-to-end test builds block `"h3"` for three synthetic combinations by measuring the reference and rest into H.3's cache under `tmp_path` (so the runner's reactivity check sees the same numbers), then runs the CLI on the synthetic connectome with single-type pools and two synthetic pairs. Every refusal is exercised on its own and must leave `results/m0d/h4` absent.

- [ ] **Step 1: Write the failing end-to-end test**

`tests/test_run_m0d_h4.py`:

```python
"""scripts/run_m0d_h4.py end to end on the synthetic connectome: real jobs on a FlyPool, H.3's cache for the reference
and rest, the resume, both guards on every written file, the refusals and when block "h4" may be written (spec H.4,
H.4a)."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import h3_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_rules import mbon_type_stats
from flymon.brain.h3_spec import SPEC as H3, OdorSet, Window, make_odors
from flymon.brain.h4_measure import HASHED_FILES, MEASURE_FILES
from flymon.brain.h4_pairs import pairs_digest
from flymon.brain.h4_spec import SPEC

_spec = importlib.util.spec_from_file_location("run_m0d_h4", Path("scripts/run_m0d_h4.py").resolve())
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

CLEAN = dict(commit="t", dirty_hashed=[], dirty_other=[])
BASE = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05)
COMBOS = {"C0": Params(**BASE, kc_thresh=0.5), "C1": Params(**BASE, kc_thresh=0.55), "C3": Params(**BASE, kc_thresh=0.6)}
POOLS = {"A": ["MBON03"], "P": ["MBON01"]}
CORES = {"A": ["MBON03", "MBON04"], "P": ["MBON01", "MBON02"]}          # the synthetic PPL105 and PAM08 core types
PAIRS = [dict(axis="a", turn=0, x="X0", y="Y0", odor_x={"ORN_DM1": 1.0}, odor_y={"ORN_VA2": 1.0}),
         dict(axis="b", turn=2, x="X1", y="Y1", odor_x={"ORN_DM6": 1.0}, odor_y={"ORN_DM1": 0.5, "ORN_VA2": 0.5})]


def lenient_spec(npz, pairs=PAIRS):
    """Small windows and seed sets, thresholds wide enough that every combination reaches the oracle and a selection."""
    r = dataclasses.replace
    h3 = r(H3, connectome_sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest(), strength=3.0, design_k=2,
           reference=OdorSet("S{j:02d}", 3, 11, 1000, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=None),
           reference_window=Window(50.0, 100.0))
    return r(SPEC, h3=h3, react_med_delta_min=-1e9, react_zero_share_max=1.0, teach_seeds=(8, 9), teach_min_decreased=0,
             teach_trials=2, teach_present_ms=100.0, teach_gap_ms=50.0, teach_window=Window(50.0, 100.0),
             act_seeds=(500, 501), select_seeds=(600, 601), report_seeds=(608, 609), oracle_window=Window(50.0, 100.0),
             kc_window_ms=20, n_pairs_a=1, n_pairs_b=1, pairs_digest=pairs_digest(pairs), t_b_min=0.0, f_a_min=0)


def write_h3_block(npz, spec):
    """Block "h3" as the H.3 runner leaves it: every combination adopted, with the guard statistics of H.3's cache."""
    conn = Connectome.load(npz); pops = Populations.from_connectome(conn)
    odors = make_odors(pops, spec.h3.reference)
    seeds = [int(s) for o in odors for s in o["seeds"]]
    code = h3_store.code_key(npz)
    combos = {}
    with FlyPool(npz, Params(), [{}, {}], workers=2) as pool:
        m = PoolMeasurer(pool, spec.h3, {"reference": odors}, None, h3_store.MeasureCache("results/m0d/h3/cache", code, "h3"),
                         None)
        for name, p in COMBOS.items():
            ref, rest = m.reference(p), {r["seed"]: r for r in m.rest(p, seeds)}
            types = {k: {"types": {t: mbon_type_stats(ref, rest, t, 5.0, 0.25) for t in CORES[k]}} for k in ("A", "P")}
            combos[name] = dict(status="adopted", adopted=dict(params=dataclasses.asdict(p)),
                                cells=[dict(status="adopted", stage3=dict(guard=types))])
    Path("results/summary").mkdir(parents=True, exist_ok=True)
    Path("results/summary/m0d.json").write_text(json.dumps({"h3": dict(combos=combos, measure_key=code["key"],
                                                                       run_id="h3-test", pools=CORES)}))


@pytest.fixture
def workdir(synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "git_state", lambda **kw: dict(CLEAN))
    seen = []
    old, mod = h3_store.refuse_old_engine_output, h3_store.refuse_modified_engine_output
    monkeypatch.setattr(h3_store, "refuse_old_engine_output", lambda out, k: (seen.append(("old", out)), old(out, k)))
    monkeypatch.setattr(h3_store, "refuse_modified_engine_output", lambda out, p: (seen.append(("mod", out)), mod(out, p)))
    spec = lenient_spec(synthetic_npz)
    write_h3_block(str(synthetic_npz), spec)
    seen.clear()

    def main(*extra, spec_=spec, summary_spec=spec, pairs=PAIRS, pools=POOLS):
        return run.main(["--npz", str(synthetic_npz), "--workers", "2", *extra], spec=spec_, summary_spec=summary_spec,
                        require_root=False, pairs=pairs, pools=pools)
    return tmp_path, main, seen, spec


def _reports(root):
    return sorted((root / "results/m0d/h4/runs").glob("*.json"))


def test_a_complete_run_writes_block_h4_through_both_guards_and_resumes_from_the_cache(workdir):
    root, main, seen, _ = workdir
    assert main() == 0
    s = json.loads((root / "results/summary/m0d.json").read_text())
    assert set(s) == {"h3", "h4"}
    h4 = s["h4"]["h4"]
    assert h4["outcome"] == "SELECTED" and h4["selection"]["winner"] == "C0"
    assert all(c["status"] == "readout_selected" and c["readout"] == {"A": "MBON03", "P": "MBON01"}
               and set(c["records"]) == {"naive_floor", "single_type", "report_halves"} for c in h4["combos"].values())
    assert "winner_below_bar" in h4["selection"]
    assert s["h4"]["notes"] == list(SPEC.notes) and s["h4"]["h3_run_id"] == "h3-test"
    written = sorted(str(p.relative_to(root)) for p in (root / "results/m0d/h4").rglob("*") if p.is_file())
    for f in written + ["results/summary/m0d.json"]:
        assert ("old", f) in seen and ("mod", f) in seen, f
    first = json.loads(_reports(root)[0].read_text())
    assert first["cache"]["hits"] >= 2 * len(COMBOS)                    # reference and rest came from H.3's cache
    assert main() == 0
    second = json.loads(_reports(root)[-1].read_text())
    assert second["cache"]["misses"] == 0
    assert h3_store.canonical(first["h4"]) == h3_store.canonical(second["h4"])


@pytest.mark.parametrize("blocker", ["smoke", "pairs", "dirty", "spec", "invalid", "incomplete"])
def test_each_blocker_alone_keeps_block_h4_unwritten(workdir, monkeypatch, blocker):
    root, main, _, spec = workdir
    extra = {"smoke": ["--smoke", "--out", "results/m0d/h4-test"], "pairs": ["--pairs", "1"],
             "dirty": ["--allow-dirty"]}.get(blocker, [])
    kw = {}
    if blocker == "dirty":
        monkeypatch.setattr(run, "git_state", lambda **k: dict(CLEAN, dirty_hashed=["flymon/brain/h4_rules.py"]))
    if blocker == "spec":
        kw = dict(summary_spec=dataclasses.replace(spec, tie_pairs=3))
    if blocker == "invalid":
        odd = [PAIRS[0], dict(PAIRS[1], turn=3)]
        s2 = dataclasses.replace(spec, pairs_digest=pairs_digest(odd))
        kw = dict(spec_=s2, summary_spec=s2, pairs=odd)
    if blocker == "incomplete":
        kw = dict(pools=CORES)                            # both types of a pool pass the lenient rules: the run stops
    assert main(*extra, **kw) == 0
    assert "h4" not in json.loads((root / "results/summary/m0d.json").read_text())


def _edit_h3(root, fn):
    path = root / "results/summary/m0d.json"
    good = path.read_text()
    d = json.loads(good); fn(d["h3"]); path.write_text(json.dumps(d))
    return lambda: path.write_text(good)


@pytest.mark.parametrize("case", ["connectome", "digest", "pools", "measure_key", "not_adopted", "malformed", "no_block",
                                  "dirty", "root"])
def test_every_refusal_exits_2_before_measuring_or_writing(workdir, monkeypatch, synthetic_npz, case):
    root, main, _, spec = workdir
    kw, restore = {}, lambda: None
    if case == "connectome":
        bad = dataclasses.replace(spec, h3=dataclasses.replace(spec.h3, connectome_sha256="0" * 64))
        kw = dict(spec_=bad, summary_spec=bad)
    if case == "digest":
        kw = dict(spec_=dataclasses.replace(spec, pairs_digest="0" * 64))
    if case == "pools":
        restore = _edit_h3(root, lambda b: b.__setitem__("pools", POOLS))
    if case == "measure_key":
        restore = _edit_h3(root, lambda b: b.__setitem__("measure_key", "0" * 64))
    if case == "not_adopted":
        restore = _edit_h3(root, lambda b: b["combos"]["C1"].__setitem__("status", "dropped"))
    if case == "malformed":
        restore = _edit_h3(root, lambda b: b["combos"]["C3"].pop("cells"))
    if case == "no_block":
        (root / "results/summary/m0d.json").write_text("{}")
    if case == "dirty":
        monkeypatch.setattr(run, "git_state", lambda **k: dict(CLEAN, dirty_hashed=["flymon/brain/h4_rules.py"]))
    if case == "root":
        assert run.main(["--npz", str(synthetic_npz)], spec=spec) == 2                  # not at the repository root
    else:
        assert main(**kw) == 2
    restore()
    assert not (root / "results/m0d/h4").exists()


def test_adopted_restores_a_missing_c3_threshold_file_from_the_committed_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = b"thresholds"
    sha = hashlib.sha256(data).hexdigest()
    Path("results/summary").mkdir(parents=True)
    Path(run.C3_COPY).write_bytes(data)
    p = Params(kc_thresh_mode="homeostatic", kc_thresh_file="results/m0d/h3/thresholds/theta-x.npz", kc_thresh_sha256=sha)
    guard = {"A": {"types": {"MBON13": {"median_delta": 1.0, "zero_share": 0.0}}}, "P": {"types": {}}}
    block = {"combos": {n: dict(status="adopted", adopted=dict(params=json.loads(h3_store.canonical(p))),
                                cells=[dict(status="adopted", stage3=dict(guard=guard))]) for n in SPEC.combos}}
    params, guards = run.adopted({"h3": block}, SPEC)
    assert params["C3"] == p and Path(p.kc_thresh_file).read_bytes() == data
    assert guards["C1"] == {"MBON13": {"median_delta": 1.0, "zero_share": 0.0}}
    Path(p.kc_thresh_file).write_bytes(b"other")
    with pytest.raises(ValueError, match="sha256 differs"):
        run.adopted({"h3": block}, SPEC)
    Path(p.kc_thresh_file).write_bytes(data)
    block["combos"]["C1"]["status"] = "dropped"
    with pytest.raises(ValueError, match="not adopted"):
        run.adopted({"h3": block}, SPEC)
    block["combos"]["C1"]["status"] = "adopted"
    Path(p.kc_thresh_file).unlink(); Path(run.C3_COPY).unlink()
    with pytest.raises(ValueError, match="is missing"):                                  # no copy: a refusal, not a traceback
        run.adopted({"h3": block}, SPEC)
    del block["combos"]["C0"]["cells"]
    with pytest.raises(ValueError, match="malformed"):
        run.adopted({"h3": block}, SPEC)


def test_the_smoke_spec_shrinks_the_samples_but_keeps_the_rules():
    s = run.smoke_spec(SPEC)
    assert len(s.teach_seeds) < len(SPEC.teach_seeds) and len(s.report_seeds) < len(SPEC.report_seeds)
    assert (s.t_b_min, s.f_a_min, s.tie_pairs, s.pairs_digest) == (SPEC.t_b_min, SPEC.f_a_min, SPEC.tie_pairs,
                                                                  SPEC.pairs_digest)


def test_the_key_and_manifest_files_exist_and_the_rules_stay_outside_the_key():
    for f in HASHED_FILES:
        assert (h3_store.ROOT / f).exists(), f
    assert set(MEASURE_FILES) <= set(HASHED_FILES) and len(set(HASHED_FILES)) == len(HASHED_FILES)
    for f in ("flymon/brain/h4_rules.py", "flymon/brain/h4_runner.py", "flymon/brain/h4_spec.py",
              "flymon/brain/h4_pairs.py", "scripts/run_m0d_h4.py"):
        assert f not in MEASURE_FILES and f in HASHED_FILES, f
    for f in ("flymon/brain/h4_formula.py", "flymon/brain/h4_jobs.py", "flymon/brain/plasticity.py",
              "flymon/brain/presentation.py", "flymon/brain/conditioning.py"):
        assert f in MEASURE_FILES, f
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q -o addopts="" tests/test_run_m0d_h4.py`
Expected: `FileNotFoundError` for `scripts/run_m0d_h4.py` at import.

- [ ] **Step 3: Write the CLI**

`scripts/run_m0d_h4.py`:

```python
#!/usr/bin/env python3
"""M0d H.4: select the combination (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a) — readout reselection and
the oracle for C0, C1 and C3, then the selection.

    uv run python scripts/run_m0d_h4.py                          # the full run (~1.5 h on 16 workers)
    uv run python scripts/run_m0d_h4.py --smoke --allow-dirty    # minutes; results/m0d/h4-smoke/, no summary

Run it from the repository root, and leave the hashed files alone while it runs. The adopted Params come from block
"h3" of results/summary/m0d.json. Reference and rest measurements are H.3's cache entries (results/m0d/h3/cache); the
conditioning arms and the oracle are cached under --out by content, per pair for the oracle, so an interrupted run
resumes by running the same command. Every combination is reselected before the first oracle.
The run report goes to <out>/runs/<run id>.{json,md}. Block "h4" of results/summary/m0d.json is replaced (rerunning a
finished run replaces it again, with the same verdict from the cache) only by a complete run: all pairs, no --smoke /
--pairs, clean hashed files, the declared configuration, and an outcome that is neither INVALID nor the mid-run stop
on two passing types in a pool (the stops for the user's decision after a selection are complete results).
Exit codes: 0 done (whatever the outcome); 2 refused before measuring anything — not at the repository root, dirty
hashed files, an NPZ other than the declared connectome, no usable block "h3", core pools or an H.3 measurement key
other than block "h3"'s, a pair list other than the declared one, a block "h3" without three adopted combinations or
with a C3 threshold file that fails its sha256. A reactivity that differs from block "h3"'s guard raises: the run stops
with a traceback before any oracle and writes nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_spec import make_odors
from flymon.brain.h3_store import (MEASURE_FILES as H3_MEASURE_FILES, ROOT, MeasureCache, code_key, git_state,
                                  replace_summary_block, sha256_file, write_bytes, write_json)
from flymon.brain.h4_measure import HASHED_FILES, MEASURE_FILES, H4Measurer
from flymon.brain.h4_pairs import even_pairs, pair_key, pairs_digest
from flymon.brain.h4_rules import INVALID, STOP_MULTI_TYPE
from flymon.brain.h4_runner import Context, run_h4
from flymon.brain.h4_spec import SPEC, H4Spec

H3_CACHE = "results/m0d/h3/cache"
C3_COPY = "results/summary/m0d_h3_c3_thresholds.npz"
POOL_TIMEOUT_S = 1800.0          # one oracle round is ~6-8 min; a lost worker should not stall the run for hours


def refuse(msg: str) -> int:
    print(f"refusing to run: {msg}", file=sys.stderr)
    return 2


def params_from_json(d: dict) -> Params:
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def adopted(summary: dict, spec: H4Spec) -> tuple[dict, dict]:
    """{name: Params} and {name: {type: guard stats}} from block "h3". C3's threshold file must exist with its sha256;
    a missing source is restored from the committed copy (same sha256). ValueError when the block cannot be used."""
    try:
        h3 = summary[spec.h3_block]
        params, guards = {}, {}
        for name in spec.combos:
            c = h3["combos"][name]
            if c["status"] != "adopted":
                raise ValueError(f"block {spec.h3_block}: {name} is {c['status']}, not adopted")
            params[name] = params_from_json(c["adopted"]["params"])
            cell = next(x for x in c["cells"] if x.get("status") == "adopted")
            guards[name] = {t: st for k in ("A", "P") for t, st in cell["stage3"]["guard"][k]["types"].items()}
    except (KeyError, StopIteration, TypeError) as e:
        raise ValueError(f"block {spec.h3_block} is malformed: {e!r}") from None
    for name, p in params.items():
        if p.kc_thresh_mode != "homeostatic":
            continue
        src = Path(p.kc_thresh_file)
        if not src.exists():
            if not Path(C3_COPY).exists() or sha256_file(C3_COPY) != p.kc_thresh_sha256:
                raise ValueError(f"{src} is missing and {C3_COPY} is missing or does not match {name}'s sha256")
            write_bytes(src, Path(C3_COPY).read_bytes(), [p])
        if sha256_file(src) != p.kc_thresh_sha256:
            raise ValueError(f"{src}: sha256 differs from {name}'s Params")
    return params, guards


def smoke_spec(spec: H4Spec) -> H4Spec:
    """A few-minute pass through every code path (with --pairs 1); its numbers are not judgments."""
    r = dataclasses.replace
    return r(spec, teach_seeds=(8, 9), teach_min_decreased=1, act_seeds=(500, 501), select_seeds=(600, 601),
             report_seeds=(608, 609))


def md_report(res: dict) -> str:
    L = [f"# M0d H.4 run {res['run_id']}", "", f"commit {res['git']['commit']}, dirty hashed files "
         f"{res['git']['dirty_hashed'] or 'none'}, smoke {res['smoke']}, wall {res['wall_s'] / 60:.1f} min", "",
         f"**outcome: {res['h4']['outcome']}**", "", "| combination | status | readout | z | testable (b) / T_b | F_a |"
         " naive floor A / P (mean, zero share) | A only / P only testable (b) | report-seed halves |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, c in res["h4"]["combos"].items():
        agg, rec = (c.get("oracle") or {}).get("aggregate"), c.get("records")
        cells = [name, c["status"], str(c.get("readout") or ""), str(c.get("z") or ""),
                 f"{agg['testable_b']}/{agg['n_b']} / {agg['T_b']:.3f}" if agg else "", str(agg["F_a"]) if agg else ""]
        if rec:
            fl, st = rec["naive_floor"], rec["single_type"]
            cells += [" / ".join(f"({fl[k]['mean']:.1f}, {fl[k]['zero_share']:.2f})" for k in ("A", "P")),
                      " / ".join(str(st[k]["aggregate"]["testable_b"]) if st[k]["aggregate"] else "-" for k in ("A", "P")),
                      " / ".join(str(h["testable_b"]) for h in rec["report_halves"])]
        else:
            cells += ["", "", ""]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", f"selection: {res['h4'].get('selection')}", ""] + [f"- {n}" for n in res["notes"]]
    return "\n".join(L) + "\n"


def main(argv=None, spec: H4Spec | None = None, summary_spec: H4Spec = SPEC, require_root: bool = True,
         pairs: list | None = None, pools: dict | None = None) -> int:
    """pairs / pools replace the E0 pair list and the core-type pools in the tests only: the synthetic connectome has
    neither the pool's receptor types nor one type per pool. Every refusal comes before any measurement or write."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)          # verification only: the first N pairs of each axis
    a = ap.parse_args(argv)
    if a.smoke and a.pairs is None:
        a.pairs = 1

    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke_spec(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/h4-smoke" if a.smoke else "results/m0d/h4"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    h3_code = code_key(a.npz, files=H3_MEASURE_FILES)
    code = code_key(a.npz, files=MEASURE_FILES)
    manifest = code_key(a.npz, files=HASHED_FILES)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    try:
        summary = json.loads(Path(a.summary).read_text())
        block = summary[spec.h3_block]
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable block {spec.h3_block} in {a.summary}: {e!r}")

    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    types_of = lambda dan: sorted({str(np.asarray(conn.type).astype(str)[int(i)]) for i in comps[dan].core})
    core = dict(A=types_of(spec.h3.punish_type), P=types_of(spec.h3.reward_type))
    if block.get("pools") != core:
        return refuse(f"the core pools {core} differ from block {spec.h3_block}'s {block.get('pools')}")
    if block.get("measure_key") != h3_code["key"]:
        return refuse(f"H.3's measurement key is {h3_code['key'][:12]} but block {spec.h3_block} was measured with "
                      f"{str(block.get('measure_key'))[:12]}: the reactivity would not be H.3's guard")
    pools = pools or core
    pairs = even_pairs(pops) if pairs is None else pairs
    digest = pairs_digest(pairs)
    if digest != spec.pairs_digest or sum(p["axis"] == "a" for p in pairs) != spec.n_pairs_a \
            or sum(p["axis"] == "b" for p in pairs) != spec.n_pairs_b:
        return refuse(f"the pair list differs from the declared one ({digest})")
    if a.pairs:
        pairs = [p for ax in ("a", "b") for p in [q for q in pairs if q["axis"] == ax][:a.pairs]]
    try:
        combos, guards = adopted(summary, spec)                 # last: it may restore C3's threshold file
    except ValueError as e:
        return refuse(str(e))

    t0 = time.time()
    odors = make_odors(pops, spec.h3.reference)
    ctx = Context(spec=spec, combos=combos, pools=pools, probe_seeds=[int(s) for o in odors for s in o["seeds"]],
                  expected=[pair_key(p) for p in pairs], h3_guard=guards, log=lambda s: print(s, flush=True))
    cache = MeasureCache(out / "cache", code, run_id)
    h3_cache = MeasureCache(H3_CACHE, h3_code, run_id)
    print(f"run {run_id}: pools {pools}, {len(pairs)} pairs, H.3 measurement key {h3_code['key'][:12]} "
          f"(= block {spec.h3_block}'s), H.4 key {code['key'][:12]}", flush=True)

    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], h3_measure_key=h3_code["key"], spec=spec, pools=pools,
               pairs_digest=digest, n_pairs=len(pairs), h3_run_id=block.get("run_id"), notes=list(spec.notes))
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers, punish_type=spec.h3.punish_type,
                 reward_type=spec.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        h3m = PoolMeasurer(pool, spec.h3, {"reference": odors}, None, h3_cache, None)
        m = H4Measurer(pool, spec, pairs, pools, cache, h3m)
        res["h4"] = run_h4(m, ctx)
        guard_params = [Params()] + [p for p in m.params_seen if p != Params()]
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits + h3_cache.hits, misses=cache.misses + h3_cache.misses)
    used = sorted(set(cache.used) | set(h3_cache.used))
    res["artifacts"] = {p: sha256_file(p) for p in used if Path(p).exists()}
    report = write_json(out / "runs" / f"{run_id}.json", res, guard_params)
    write_bytes(out / "runs" / f"{run_id}.md", md_report(res).encode(), guard_params)
    print(f"wrote {report}: outcome {res['h4']['outcome']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, pairs=a.pairs is not None, dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec,
                    invalid=res["h4"]["outcome"] == INVALID, incomplete=res["h4"]["outcome"] == STOP_MULTI_TYPE)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        replace_summary_block(a.summary, "h4", dict(res, report=str(report), report_sha256=sha256_file(report)),
                              guard_params)
        print(f"replaced block 'h4' of {a.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write the reproduction check (step R2's script; no test — the controller runs it on real data; it reads four git-ignored records: M0c's conditioning, the H.4a.1 diagnostic, G.12's E0 rows and G.14's S-ref smoke row)**

`docs/superpowers/specs/m0d-diag/h4_reproduction_check.py`:

```python
"""M0d H.4 reproduction check (plan step R2): the committed H.4 worker jobs against recorded numbers, before the H.4 run.

1. `h4_jobs.teach_job` on C0 (`Params()`), order "ab", both single-channel arms, seeds 8-15: the PPL105-core and
   PAM08-core sums of its per-type counts equal the A/P probe counts recorded in results/m0c/conditioning.json.
2. `h4_jobs.teach_job` on the three adopted engines (block "h3"), both orders, both arms, seeds 8-15 (96 arms): every
   pool type's pre/post counts equal the H.4a.1 diagnostic, results/m0d/diag/h4_teach_odour.json (git-ignored).
3. `h4_jobs.oracle_job` on C0 with readout MBON13 / MBON05 and F.3's frozen constants, the first --pairs pairs of each
   axis: the selection-seed probes (pre, and R1 at every alpha; seeds 600-607) and the KC activity (seeds 500-507)
   equal G.12's E0 rows, results/m2/calibration/encoders/E0_even.json (git-ignored).
4. The same job on G.14's smoke pair and seed blocks (500-501, 600-601, 608-609): the selection probes and changes,
   both alphas, the report pre / R1 / R2 and the KC records equal G.14's S-ref smoke row,
   results/m2/calibration/engine_probe/smoke/S-ref.json (git-ignored; driver sha256 81ad9e82..., spiking, kc_thresh 1.5
   = C0) — the punishment choice and the report seeds, which the verdict reads, against a recorded run.
All bit for bit. Exit 0 only when everything matches. Writes nothing.

    uv run python docs/superpowers/specs/m0d-diag/h4_reproduction_check.py [--workers 16] [--pairs 2]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from flymon.brain import h4_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h4_pairs import even_pairs, pair_key
from flymon.brain.h4_spec import SPEC

NPZ = "data/malecns.npz"
M0C = Path("results/m0c/conditioning.json")
TEACH_DIAG = Path("results/m0d/diag/h4_teach_odour.json")
G12 = Path("results/m2/calibration/encoders/E0_even.json")
SREF = Path("results/m2/calibration/engine_probe/smoke/S-ref.json")
VERDICT = Path("docs/superpowers/specs/m2-calibration-g/engine_probe_verdict.py")
VERDICT_SHA = "c9c81ab97d13e7bc163210564e1114d3caafd60c63431bbef8eb977f8fb334d7"      # spec G.14


def f3_z() -> dict:
    """F.3's frozen constants, from the frozen G.14 verdict module (sha256-checked), not restated."""
    if hashlib.sha256(VERDICT.read_bytes()).hexdigest() != VERDICT_SHA:
        raise SystemExit(f"{VERDICT} is not the frozen G.14 verdict module")
    s = importlib.util.spec_from_file_location("engine_probe_verdict", VERDICT)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return {k: tuple(v) for k, v in m.Z.items()}


def params_from_json(d: dict) -> Params:
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=2)
    a = ap.parse_args()
    s, h3 = SPEC, SPEC.h3
    block = json.loads(Path("results/summary/m0d.json").read_text())["h3"]
    pools = block["pools"]
    types = tuple(pools["A"] + pools["P"])
    combos = {n: params_from_json(block["combos"][n]["adopted"]["params"]) for n in s.combos}
    pops = Populations.from_connectome(Connectome.load(NPZ))
    pairs = even_pairs(pops)
    pick = [p for ax in ("a", "b") for p in [q for q in pairs if q["axis"] == ax][:a.pairs]]
    sref = json.loads(SREF.read_text())["rows"][0]
    sref_pair = next(p for p in pairs if pair_key(p) == (sref["axis"], sref["turn"], sref["x"], sref["y"]))
    teach = dict(types=types, punish_type=h3.punish_type, reward_type=h3.reward_type, k=h3.design_k,
                 odor_seed=h3.design_odor_seed, strength=h3.strength, trials=s.teach_trials,
                 present_ms=s.teach_present_ms, gap_ms=s.teach_gap_ms, settle_ms=s.teach_window.settle_ms,
                 read_ms=s.teach_window.read_ms)
    oracle = dict(params=Params(), readout={"A": "MBON13", "P": "MBON05"}, z=f3_z(), types=types,
                  act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds), report_seeds=tuple(s.report_seeds),
                  alphas=tuple(s.oracle_alphas), strength=h3.strength, settle_ms=s.oracle_window.settle_ms,
                  read_ms=s.oracle_window.read_ms, window_ms=s.kc_window_ms, punish_type=h3.punish_type,
                  reward_type=h3.reward_type)
    smoke = dict(oracle, act_seeds=(500, 501), select_seeds=(600, 601), report_seeds=(608, 609))
    t_rows = {}
    with FlyPool(NPZ, Params(), [{} for _ in range(a.workers)], workers=a.workers, timeout_s=3600.0) as pool:
        for name, p in combos.items():                  # one combination at a time: each worker builds one rig per Params
            t_rows[name] = pool.run_jobs(h4_jobs.teach_job, [dict(teach, params=p, seed=seed, arm=arm, order=order)
                                                             for order in ("ab", "ba") for arm in ("punish_only", "reward_only")
                                                             for seed in s.teach_seeds])
        o_rows = pool.run_jobs(h4_jobs.oracle_job, [dict(oracle, odor_x=p["odor_x"], odor_y=p["odor_y"]) for p in pick]
                               + [dict(smoke, odor_x=sref_pair["odor_x"], odor_y=sref_pair["odor_y"])])
    bad = []
    m0c = json.loads(M0C.read_text())["per_seed"]
    for r in t_rows["C0"]:
        if r["order"] != "ab":
            continue
        ref = m0c[str(r["seed"])][r["arm"]]["counts"]
        for ph in ("pre", "post"):
            for cs in ("plus", "minus"):
                got = (sum(r[ph][cs][t] for t in pools["A"]), sum(r[ph][cs][t] for t in pools["P"]))
                if got != (ref[f"{ph}_{cs}"]["A"], ref[f"{ph}_{cs}"]["P"]):
                    bad.append(f"M0c: seed {r['seed']} {r['arm']} {ph}_{cs}: {got}")
    diag = json.loads(TEACH_DIAG.read_text())
    for name, rows in t_rows.items():
        want = {(d["seed"], d["arm"], d["order"]): d for d in diag[name]}
        for r in rows:
            d = want[(r["seed"], r["arm"], r["order"])]
            for ph in ("pre", "post"):
                for cs in ("plus", "minus"):
                    if any(r[ph][cs][t] != d[ph][cs][t] for t in types):
                        bad.append(f"H.4a.1 diagnostic: {name} seed {r['seed']} {r['arm']} {r['order']} {ph}_{cs}")
    print(f"teach: {sum(len(v) for v in t_rows.values())} arms against the H.4a.1 diagnostic, 16 of them against M0c")
    g12 = {(r["axis"], r["turn"], r["x"], r["y"]): r for r in json.loads(G12.read_text())["rows"]}
    for p, r in zip(pick, o_rows[:-1]):
        g, sel = g12[pair_key(p)], r["select"]
        checks = {"pre": sel["pre"]["MBON13"] == g["pre"]["A"] and sel["pre"]["MBON05"] == g["pre"]["P"],
                  "R1": all(sel["reward"][k]["R1"]["MBON13"] == g["reward"][k]["R1"]["A"]
                            and sel["reward"][k]["R1"]["MBON05"] == g["reward"][k]["R1"]["P"] for k in g["reward"]),
                  "kc": r["kc"]["x"]["frac"] == g["kc"]["frac_x"] and r["kc"]["y"]["frac"] == g["kc"]["frac_y"]
                  and r["kc"]["x"]["spikes"] == g["kc"]["spikes_x"] and r["kc"]["y"]["spikes"] == g["kc"]["spikes_y"]}
        bad += [f"G.12 {pair_key(p)}: {k}" for k, ok in checks.items() if not ok]
        print(f"G.12 {pair_key(p)}: {checks}")
    r, ap_ = o_rows[-1], lambda pr: {"A": pr["MBON13"], "P": pr["MBON05"]}
    sel = r["select"]
    checks = {"alphas": (r["alpha_reward"], r["alpha_punish"]) == (sref["alpha_reward"], sref["alpha_punish"]),
              "select pre": ap_(sel["pre"]) == sref["select"]["pre"],
              "select reward": all(ap_(sel["reward"][k]["R1"]) == v["R1"] and sel["reward"][k]["change"] == v["change"]
                                   for k, v in sref["select"]["reward"].items()),
              "select punish": all(ap_(sel["punish"][k]["R2"]) == v["R2"] and sel["punish"][k]["change"] == v["change"]
                                   for k, v in sref["select"]["punish"].items()),
              "report": r["report"] == sref["report"],
              "kc": all(r["kc"][c][f] == sref["kc"][c][f] for c in ("x", "y") for f in ("frac", "spikes", "max_win"))}
    bad += [f"G.14 S-ref: {k}" for k, ok in checks.items() if not ok]
    print(f"G.14 S-ref {pair_key(sref_pair)}: {checks}")
    print("ALL MATCH" if not bad else "MISMATCH\n" + "\n".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -o addopts="" tests/test_run_m0d_h4.py`
Expected: 19 passed (about 10–15 s: every end-to-end case starts pools on the synthetic connectome). The last test checks that every file the cache key and the manifest hash exists — it lives here because the list names this task's CLI.

- [ ] **Step 6: Full suite, then commit**

Run: `uv run pytest -q -rN -o addopts=""` → **478 passed, 1 skipped, 1 xfailed**.

```bash
git add scripts/run_m0d_h4.py docs/superpowers/specs/m0d-diag/h4_reproduction_check.py tests/test_run_m0d_h4.py
git commit -m "feat(scripts): run_m0d_h4 — the resumable M0d H.4 selection runner; h4_reproduction_check"
```

---

## Controller steps after Task 5 (not subagent tasks)

These run real data from the repository root. The controller runs each with Bash `run_in_background: true` and reads only exit codes and summary lines. Commit everything first: the runner refuses dirty hashed files.

- [ ] **R0: Preconditions.** `git log --oneline` contains `docs(spec): M0d H.4a …` and `docs(plans): M0d H.4 selection runner …` below Task 5's commit; the full suite is at 478 / 1 / 1; `git status --short` shows nothing under `flymon/`, `scripts/` or `uv.lock`.

- [ ] **R1: Smoke.** `uv run python scripts/run_m0d_h4.py --smoke` → exit 0, a report under `results/m0d/h4-smoke/runs/`, `summary not written: ['smoke', 'pairs', 'spec']`. It runs every path on one pair per axis with two seeds per block; its numbers are not judgments. If the smoke shows a design flaw, record an amendment before R3 (spec H.7, G.14.7 precedent).

  Validated 2026-09-22 on this plan's code in a scratch worktree, `--smoke --allow-dirty`: first on the pre-review code (run `20260921T151741Z-57753f`,
  11.2 min on 16 workers, about 4 min of it re-measuring H.3's reference and rest, which that checkout's cache lacked), then on this code (run
  `20260921T164303Z-211774`, every measurement a cache hit — the review changed only `h4_rules`, `h4_runner`, `h4_spec` and the CLI, which are outside the
  measurement key, so nothing was re-measured). Both: exit 0, `summary not written: ['smoke', 'pairs', 'dirty', 'spec']` (`dirty` only because the scratch
  files were uncommitted); the re-measured reactivity equalled block `"h3"`'s guard for all three combinations; C3's threshold file was restored from the
  committed copy; every combination read out MBON13 / MBON05; outcome `STOP_NO_ELIGIBLE` (one pair per axis, two seeds per block — not a judgment); the
  markdown report carries the H.4a.4 records.
- [ ] **R2: Reproduction check.** `uv run python docs/superpowers/specs/m0d-diag/h4_reproduction_check.py` → must print `ALL MATCH` and exit 0. Four comparisons, all bit for bit: the committed `teach_job` against M0c's recorded counts (C0, order "ab", seeds 8–15, both arms) and against the H.4a.1 diagnostic (all three engines, both orders, both arms: 96 arms, every pool type); the committed `oracle_job` against G.12's E0 raw counts (C0, readout MBON13 / MBON05, F.3's constants, the first two pairs of each axis: selection-seed pre and R1 at every α, KC activity) and against G.14's S-ref smoke row (the same engine on G.14's smoke seed blocks: both α choices, the selection changes, the report pre / R1 / R2, the KC records) — the punishment and report path the verdict reads, against a recorded run. A mismatch stops the plan.

  Validated 2026-09-22 on this plan's code in a scratch worktree (HEAD `95ee094` + the plan's files), 20.1 min on 16 workers: `teach: 96 arms against the
  H.4a.1 diagnostic, 16 of them against M0c`; G.12 pairs (a, 0, Surf, Earthquake), (a, 0, Surf, Strength), (b, 0, Surf vs Chansey, Surf vs Rhydon),
  (b, 0, Earthquake vs Chansey, Earthquake vs Rhydon) — `pre`, `R1` (all three α), `kc` equal; G.14 S-ref (b, 0, Surf vs Chansey, Surf vs Rhydon) —
  `alphas` (0.2, 0.5), `select pre`, `select reward`, `select punish` (probes and changes), `report`, `kc` (frac, spikes, max_win) equal → `ALL MATCH`.
- [ ] **R3: The run.** `uv run python scripts/run_m0d_h4.py` (estimate below). Do not touch the hashed files while it runs. If it is interrupted, run the same command again (the oracle resumes per pair); a rerun after it finished replaces block `"h4"` again — diff `results/summary/m0d.json` before R4's commit. On any outcome except `INVALID` and `stop_multiple_types`, block `"h4"` of `results/summary/m0d.json` is written. Then bring the outcome to the user: `SELECTED` goes to H.5 (a separate plan; with C0, the engine is unchanged and only the readout swap is confirmed); `STOP_LOW_T_B`, `STOP_NO_ELIGIBLE`, `INVALID` and `stop_multiple_types` are the user's decision (spec H.4 step 3).
- [ ] **R4: Record.** Append an "H.4 결과 (날짜)" paragraph after H.4a: per combination the reactivity (equal to H.3's guard), the teach choice (odour, order, n decreased, untaught n decreased) per type, the readout and z constants, T_b (testable (b) of 21), F_a, the naive counts, the H.4a.4 records (readout floor, per-type statistics, report-seed halves), the selection with its near set and `winner_below_bar`, the three interpretation sentences (`SPEC.notes`), wall time, run ID, the R2 check. Commit `results/summary/m0d.json` with the spec.

**Cost estimate** (measured in the scratch validation on a loaded 18-core machine): one oracle pair ≈ 6 min of worker time; 39 pairs on 16 workers = 3 rounds ≈ 20–25 min per combination; the teaching arms ≈ 5 min per combination (32 arms); reactivity and z come from H.3's cache (seconds). **Full run ≈ 1.5 h**; R1 ≈ 7 min (11 min when H.3's reference and rest must be re-measured); R2 ≈ 20 min.
