# Spec K.8 — Fast KC→KC Inhibition Scan and Judgement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spec appendix K as amended by K.8 — for each g in the declared grid, re-converge C3's rule with fast KC→KC inhibition (`kc_kc_scale = g < 0`), rank the ADOPTED engines by the absolute X-only drive onto MBON13 (N) on the odd-turn (b) pairs, gate on N / N_C3 > 1.0, and only then judge the top engine on the 21 even-turn pairs with J's stage-2 path and J.12.9's bands.

**Architecture:** Nine small `flymon/brain/k_*` modules plus two scripts, reusing J's `reconverge` / `judge` / `measure_d6` and H.3/H.4 measurers unchanged. `k_spec` is the one configuration object; `k_params` builds the engine (`with_kc`); `k_store` guards report/summary writes; `k_pairs` builds the odd-turn (b) pairs; `k_metrics` holds the pure metrics (N, S, D₁₃, reliability, lobes, KC→KC input); `k_jobs` / `k_measure` measure per-KC activity and one raster through FlyPool with a content-keyed cache; `k_rules` holds selection, gate and closing states; `k_runner` drives stage 1 and stage 2. `scripts/run_k_scan.py` runs stage 1 (and self-check i); `scripts/run_k_judge.py` runs stage 2.

**Tech Stack:** Python 3.13, NumPy 2.5, pytest, `uv`; `flymon.brain.fly_pool.FlyPool` (spawn start method).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` — appendix **K** (K.0–K.7) and its amendment **K.8** (K.8 wins where they differ: K.8.1 odd-turn use and G.2, K.8.2 stage 1 with per-setting re-convergence, K.8.3 metric N, K.8.4 gate and stage 2, K.8.5 closing sentences, K.8.6 prior expectation, K.8.7 storage/self-checks/tests/cost), with **J.10.4** (re-convergence state machine), **J.11.4** (stage-2 steps), **J.12.9** (bands), **H.3a.3–4** (measurement table, qualification), **H.4a.7** (f definition, presentation), **G.2** (turn split).

## Global Constraints

- **Commits carry no trailers of any kind** — no `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder that asks for them.
- **No existing module changes.** `Params`, the engine, FlyPool, every `j_*` / `h3_*` / `h4_*` / `d6a` module and the J scripts stay byte-identical (their hashes are in J's recorded keys; K.1). K only adds files.
- **One configuration object.** Every K constant is a field of `flymon.brain.k_spec.SPEC`; J/H constants are read through `SPEC.j` (a `JSpec`). No other module writes one as a literal. Tests may build smaller specs with `dataclasses.replace`.
- **Grid** `(-0.05, -0.1, -0.2, -0.4, -0.8)`; tie tolerance `0.01` (relative, on N); gate `N / N_C3 > 1.0`; target readout `MBON13`, reward-side record `MBON05`; synapse floor `5`; f = fraction of act seeds 500–507 on which a KC fired at least once in the read window (H.4a.7); presentation strength `SPEC.j.h4.h3.strength`, window `SPEC.j.h4.oracle_window` (800 ms settle, 600 ms read), sliding window `SPEC.j.h4.kc_window_ms`.
- **Every output path is guarded.** Cache, threshold files: under `results/m0d/k/` through `h3_store` (its guard allows `results/m0d/`). Run reports and self-check reports under `results/m0d/k/`, and the summary `results/summary/k_engine.json`, through `k_store` (allows only those). J/old paths (`results/summary/m2_engine.json`, `std_scan.json`, `results/m0d/j/`, `results/m0*/` outside `m0d/k/`) are refused.
- **Worker jobs / pools only from script files with `if __name__ == "__main__":` or from pytest**; never from a heredoc.
- **Subagents never write under `results/` and never start a real run** (smoke, self-check, scan, judgement). Tests write only under `tmp_path`. The controller runs the real-data steps (section "Run order") after Task 7.
- **Full suite before each commit:** `uv run pytest -q -rfE -o addopts=""` — currently **716 passed, 1 skipped, 3 xfailed** (HEAD `aa3581e`, this worktree with `data/malecns.npz` and the git-excluded results). Known flake: `tests/test_run_m0d_h3.py::test_a_complete_run_writes_everything_through_both_guards_and_resumes_from_the_cache` — rerun that file once before calling it a regression; never run two suites at once.
- Style: match the surrounding modules — module docstring that cites the spec section, dense one-line comments, `from __future__ import annotations`.

## Readings of the spec (decided here)

1. **C3's recorded `kc_kc_scale` is 0.0** (KC→KC edges absent from the CSC). `with_kc(C3, 0.0)` is therefore C3 itself (dataclass-equal), and `k_make(h3spec, 0.0)` is `c1_params` unchanged — self-check i re-converges exactly J's self-check (ii) engine. A negative g puts the 33,247 KC→KC edges back with weight × g (`build_csc`).
2. **Engine base for re-convergence:** `make(kc) = with_kc(c1_params(h3spec, kc, 1.0), g)` — C3's rule with the H.3a grid, first candidate `SPEC.j.first_kc` (1.65), target `SPEC.j.homeo_target` (0.060), via `j_runner.reconverge`. `j_runner.stage2` is not used (it forces STD on); only its loop shape is mirrored.
3. **N_C3** is measured on C3's recorded engine (`j_store.load_c3`), not on a re-converged g = 0 engine; self-check i shows the two agree.
4. **f and counts** come from one measurement per (odour, seed): `h4_jobs._present_kc` with plasticity disabled and weights reset (H.4a.7's `ceiling_job`), keeping the read-window count per KC. f_x = fraction of the 8 seeds with count > 0; c_x = mean count. Odours are deduplicated by their normalised glomerulus dict.
5. **w₁₃, w₀₅** are synapse counts (`conn.w`, edges ≥ 5) from every KC to every cell of that type — not the plasticity `w0` in mV. N_p = Σ w₁₃·f_x·[f_y = 0]; D₁₃ = Σ w₁₃·f_x; S_p = N_p / D₁₃ (0 when D₁₃ = 0); N = Σ_p N_p over the fixed odd pair list; N05 likewise with w₀₅.
6. **KC→KC input record** (K.8.2 manipulation record) = per-KC Σ_pre c̄_pre · w_pre→post · |g| · `mv_per_synapse` over KC→KC edges ≥ 5, from the pair-mean X counts — a spike-count proxy of the inhibitory charge (hemisphere factor not applied), recorded per lobe, never judged.
7. **Lobes** by KC type prefix: `KCa'b'` (α'β'), `KCg` (γ), `KCab` (αβ).
8. **Raster** (K.8.2): at `SPEC.raster_g` (−0.8), if that setting is ADOPTED, one presentation of the first odd pair's X odour at seed 500: KC spikes (step, KC position) over settle + read, and the mean membrane of the α'β' KCs per step. If −0.8 is not ADOPTED the record says so.
9. **Selection** (K.8.3): among ADOPTED settings, repeatedly take the largest N; every setting with N ≥ best·(1 − tie_tol) is tied; among tied, larger S median wins, then smaller |g|. The full order is recorded. `N_C3 ≤ 0` raises (the ratio is undefined; stop and ask).
10. **Scan outcomes:** `COMPUTE_ABORTED` (budget) → `STOP_NO_QUALIFIED_SETTING` (no ADOPTED) → `STOP_NO_TARGET_GAIN` (top N / N_C3 ≤ 1.0) → `SCAN_GO`.
11. **Closing states** (K.8.5) from the judgement: `stop_multiple_types`; `dropped_no_readout` (J's `judge` returns B with `reading` None); `INVALID`; `COMPUTE_ABORTED`; otherwise the J.12.9 band (`SELECTED`, `B_Tb`, `B_NO_CONCLUSION`, `B_Fa`).
12. **H.5 comparison deferred.** No H.5 turn-set builder exists; K.8.1's odour-level overlap report compares the odd pairs with the 21 even (b) pairs (and the 18 (a) pairs) now, and with H.5 only if a `SELECTED` engine reaches confirmation (the confirmation builder is new work then). Task 2 adds a dated line to K.8.1 saying so.
13. **Key and manifest:** measurement key = `h3_store.code_key` over `k_measure.MEASURE_FILES` (J's MEASURE_FILES + `k_params`, `k_pairs`, `k_jobs`, `k_measure`) and the NPZ; manifest over `k_measure.HASHED_FILES` (+ J's HASHED_FILES, `k_spec`, `k_metrics`, `k_rules`, `k_runner`, `k_store`, both K scripts). The judgement refuses a scan under another key, manifest or spec (K.3-7).
14. **Shared output dir:** both scripts default to `results/m0d/k/run` (smoke: `results/m0d/k/smoke`), so the judgement finds the scan's threshold files and reuses its cache.

## Review Focus

1. **N_C3 = 0 or tiny** (C3's X-only MBON13 drive near zero on the odd pairs) — expect a hard stop with a clear message, not a division by zero or an infinite ratio that passes the gate. Test in Task 5.
2. **An odd turn whose alternate opponent cannot be found** — expect `odd_pairs` to raise with the turn number (G.11 gives no fallback), not to silently drop the turn and change the pair count. Test in Task 2.
3. **Judgement run against a scan whose threshold file was deleted or regenerated** — expect refusal before any measurement (sha mismatch), not a judgement on another engine. Test in Task 7.
4. **A setting that aborts mid-scan and is resumed** — expect the same cache keys on rerun (no remeasurement of finished settings) and the abort recorded as `COMPUTE_ABORTED`, never as `INVALID_ENGINE`. Test in Task 6.
5. **Duplicate odours across pairs** (the same glomerulus dict under two names) — expect one measurement shared by both pairs and the overlap report to list it, not two cache entries with different results. Test in Task 4 and Task 2.

---

## File Structure

| File | Responsibility |
|---|---|
| `flymon/brain/k_spec.py` | `KSpec` (grid, readouts, tie, gate, raster, pinned odd-pair digest) + `smoke()` |
| `flymon/brain/k_params.py` | `with_kc(base, g)`, `k_make(h3spec, g)` |
| `flymon/brain/k_store.py` | `guard`, `write_json`, `write_bytes`, `write_summary_block` for reports and `k_engine.json` |
| `flymon/brain/k_pairs.py` | `odd_pairs(pops)`, `odour_key`, `overlap_report` |
| `flymon/brain/k_metrics.py` | `readout_weights`, `lobe_masks`, `kc_kc_edges`, `pair_metrics`, `engine_metrics`, `kc_kc_input`, `removed_fraction` |
| `flymon/brain/k_jobs.py` | FlyPool jobs `activity_job`, `raster_job` |
| `flymon/brain/k_measure.py` | `MEASURE_FILES`, `HASHED_FILES`, `KMeasurer` (activity, raster; cached) |
| `flymon/brain/k_rules.py` | outcome constants, `select_order`, `scan_outcome`, `closing_state` |
| `flymon/brain/k_runner.py` | `KArrays`, `arrays`, `measure`, `stage1`, `stage2` |
| `scripts/run_k_scan.py` | stage 1 CLI, self-check i |
| `scripts/run_k_judge.py` | stage 2 CLI with key / gate / threshold checks |
| `tests/brain/test_k_spec_params_store.py`, `test_k_pairs.py`, `test_k_metrics.py`, `test_k_measure.py`, `test_k_rules.py`, `test_k_runner.py`, `tests/test_run_k.py` | tests |

---

### Task 1: `k_spec`, `k_params`, `k_store`

**Files:**
- Create: `flymon/brain/k_spec.py`, `flymon/brain/k_params.py`, `flymon/brain/k_store.py`
- Test: `tests/brain/test_k_spec_params_store.py`

**Interfaces:**
- Consumes: `flymon.brain.j_spec.SPEC`/`JSpec`; `flymon.brain.h3_runner.c1_params(spec, kc, scale, **kw)`; `flymon.brain.pool_bench.refuse_old_engine_output(out, kc_kc_scale)`, `refuse_modified_engine_output(out, params)`; `flymon.brain.h3_store.canonical_pretty(obj)`.
- Produces: `KSpec` fields `j, grid, target_type, reward_type, min_weight, tie_tol, gate_ratio, raster_g, odd_pairs_digest`; `SPEC`; `smoke(spec) -> KSpec`; `with_kc(base: Params, g: float) -> Params`; `k_make(h3spec, g) -> Callable[[float], Params]`; `k_store.ALLOWED_DIR`, `ALLOWED_FILES`, `guard(path, params_list)`, `write_json(path, obj, params_list) -> Path`, `write_bytes(path, data, params_list) -> Path`, `write_summary_block(path, block, obj, params_list) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K.8: the configuration object, the engine builder and the guarded writes."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain.config import Params
from flymon.brain.h3_runner import c1_params
from flymon.brain.j_params import StdParams
from flymon.brain.j_spec import SPEC as J_SPEC
from flymon.brain.k_params import k_make, with_kc
from flymon.brain.k_spec import SPEC, smoke
from flymon.brain.k_store import guard, write_json, write_summary_block


def test_the_declared_configuration():
    assert SPEC.grid == (-0.05, -0.1, -0.2, -0.4, -0.8)
    assert (SPEC.target_type, SPEC.reward_type, SPEC.min_weight) == ("MBON13", "MBON05", 5)
    assert (SPEC.tie_tol, SPEC.gate_ratio, SPEC.raster_g) == (0.01, 1.0, -0.8)
    assert SPEC.j == J_SPEC and SPEC.j.first_kc == 1.65 and SPEC.j.homeo_target == 0.060
    assert smoke(SPEC) != SPEC and smoke(SPEC).grid == (-0.8,)
    assert SPEC.even_pair_uses == ("H.4 C0-C3", "H.4a.8 ceiling freq", "H.4a.8 ceiling all", "J.13")


def test_with_kc_zero_is_the_engine_itself():
    base = c1_params(J_SPEC.h4.h3, 1.65, 1.0)
    assert base.kc_kc_scale == 0.0 and with_kc(base, 0.0) == base
    assert with_kc(base, -0.2).kc_kc_scale == -0.2
    assert dataclasses.replace(with_kc(base, -0.2), kc_kc_scale=0.0) == base


@pytest.mark.parametrize("g", [0.1, float("nan"), float("-inf")])
def test_with_kc_refuses_excitation_and_non_finite(g):
    with pytest.raises(ValueError):
        with_kc(Params(), g)


def test_with_kc_refuses_depression_engines():
    with pytest.raises(TypeError):
        with_kc(StdParams(receptor_scale=2.0), -0.1)
    with pytest.raises(ValueError):
        with_kc(Params(orn_std=True), -0.1)


def test_k_make_is_c3s_rule_with_g():
    make = k_make(J_SPEC.h4.h3, -0.4)
    assert make(1.6) == dataclasses.replace(c1_params(J_SPEC.h4.h3, 1.6, 1.0), kc_kc_scale=-0.4)


@pytest.mark.parametrize("path", ["results/m0d/k/run/runs/x.json", "results/summary/k_engine.json"])
def test_allowed_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(path, {"a": 1}, [Params(), with_kc(Params(), -0.2)])
    assert json.loads(Path(path).read_text()) == {"a": 1}


@pytest.mark.parametrize("path", ["results/summary/m2_engine.json", "results/summary/std_scan.json",
                                  "results/m0d/j/std/runs/x.json", "results/summary/m0d.json", "results/m0c/x.json",
                                  "results/m0d/x.json", "results/k/x.json", "x.json"])
def test_refused_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        guard(path, [with_kc(Params(), -0.2)])
    assert not Path(path).exists()


def test_summary_blocks_are_merged_atomically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = "results/summary/k_engine.json"
    write_summary_block(p, "scan", {"outcome": "scan_go"}, [Params()])
    write_summary_block(p, "judge", {"outcome": "B"}, [Params()])
    assert json.loads(Path(p).read_text()) == {"scan": {"outcome": "scan_go"}, "judge": {"outcome": "B"}}
    assert not list(Path("results/summary").glob(".*.tmp"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_k_spec_params_store.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.brain.k_spec'`

- [ ] **Step 3: Write `flymon/brain/k_spec.py`**

```python
"""The configuration of spec appendix K (K.8): fast KC->KC inhibition, g = kc_kc_scale < 0, scanned over a declared
grid; every setting re-converged by C3's rule (J.10.4 through `j`), ranked by the X-only MBON13 drive N on the odd-turn
(b) pairs, gated on N / N_C3 > 1.0, the top one judged by J's stage-2 path and J.12.9's bands.

`j` carries every J/H constant K reuses (C3's rule, first candidate, homeostasis target, the H.4 oracle, the bands);
nothing here repeats one of them.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .j_spec import SPEC as J_SPEC, JSpec


@dataclass(frozen=True)
class KSpec:
    j: JSpec = J_SPEC
    grid: tuple = (-0.05, -0.1, -0.2, -0.4, -0.8)      # K.2 / K.8.2
    target_type: str = "MBON13"                         # the punishment readout N is measured on (K.8.3)
    reward_type: str = "MBON05"                         # the reward-side N05, recorded only
    min_weight: int = 5                                 # synapse floor of w13 / w05 / KC->KC edges (Params.min_weight)
    tie_tol: float = 0.01                               # relative tie on N (K.8.3)
    gate_ratio: float = 1.0                             # stage 2 only if N / N_C3 > this (K.8.4)
    raster_g: float = -0.8                              # the setting whose raster and membrane are recorded (K.8.2)
    odd_pairs_digest: str = ""                          # pinned in Task 2 (K.8.7 self-check iii)
    even_pair_uses: tuple = ("H.4 C0-C3", "H.4a.8 ceiling freq", "H.4a.8 ceiling all", "J.13")   # K.8.4 record


SPEC = KSpec()


def smoke(spec: KSpec) -> KSpec:
    """J's smoke configuration (scripts/run_j_std_judge.smoke_spec, copied: scripts are not importable) on one g."""
    r = dataclasses.replace
    h3 = r(spec.j.h4.h3, reference=r(spec.j.h4.h3.reference, n=4), all51_seeds=(200,), kc_grid=(1.65,),
           scale_bisect_steps=2, hold_bisect_steps=2, membrane_tol_mv=5.0, design_seeds=(100, 101),
           design_extra_seeds=(102, 103), baseline_cal_seeds=(100, 101, 102, 103), baseline_gate_seeds=(116, 117),
           runaway_rest_seeds=(100, 101), runaway_odor_seeds=(100, 101), boot_draws=200, homeo_max_iter=2,
           c3_max_cycles=1)
    h4 = r(spec.j.h4, h3=h3, teach_seeds=(8, 9), teach_min_decreased=1, act_seeds=(500, 501), select_seeds=(600, 601),
           report_seeds=(608, 609))
    return r(spec, j=r(spec.j, h4=h4, max_settings=1), grid=(-0.8,))
```

- [ ] **Step 4: Write `flymon/brain/k_params.py`**

```python
"""The engine of spec appendix K (K.8.2): C3's `Params` with `kc_kc_scale` = g <= 0. `build_csc` multiplies every
KC->KC edge by g and drops them at 0 (C3's recorded value), so g < 0 is fast inhibition with no engine change and
g = 0 is C3 itself. J's `with_std` is not used: it forces the ORN depression on."""
from __future__ import annotations

import dataclasses
import math

from .config import Params
from .h3_runner import c1_params


def with_kc(base: Params, g: float) -> Params:
    if type(base) is not Params:
        raise TypeError(f"K builds on a plain Params, got {type(base).__name__} (no depression / receptor scale)")
    if base.orn_std:
        raise ValueError("K's engine has no ORN depression (orn_std must be False)")
    if not math.isfinite(g) or g > 0:
        raise ValueError(f"g must be finite and <= 0 (inhibition), got {g!r}")
    return dataclasses.replace(base, kc_kc_scale=float(g) + 0.0)       # + 0.0: -0.0 -> 0.0


def k_make(h3spec, g: float):
    """make_base for j_runner.reconverge: the cell's Params at apl_input_scale 1.0 with g applied."""
    return lambda kc: with_kc(c1_params(h3spec, kc, 1.0), g)
```

- [ ] **Step 5: Write `flymon/brain/k_store.py`**

```python
"""Guarded writes of spec appendix K (K.8.7): run reports and self-checks under results/m0d/k/ and the summary
results/summary/k_engine.json; nothing else. The cache and threshold files also live under results/m0d/k/ but go
through h3_store (whose guard allows results/m0d/). Every write is atomic (tmp + os.replace)."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from .h3_store import canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/m0d/k/"
ALLOWED_FILES = ("results/summary/k_engine.json",)


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel in ALLOWED_FILES):
        print(f"refusing to write {path}: spec K writes only under {ALLOWED_DIR} and {ALLOWED_FILES}", file=sys.stderr)
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
    """Replace one top-level block of the summary, keeping the others (scan, judge)."""
    guard(path, params_list)
    doc = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    doc[block] = json.loads(canonical_pretty(obj))
    return write_json(path, doc, params_list)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_k_spec_params_store.py -q -o addopts=""`
Expected: PASS (all). If `smoke()` fails because a replaced field name does not exist on `H3Spec`/`H4Spec`, compare with `scripts/run_j_std_judge.py:smoke_spec` — the copy must match it field for field.

- [ ] **Step 7: Full suite, then commit**

Run: `uv run pytest -q -rfE -o addopts=""` — expected: baseline + the new tests, no new failures.

```bash
git add flymon/brain/k_spec.py flymon/brain/k_params.py flymon/brain/k_store.py tests/brain/test_k_spec_params_store.py
git commit -m "feat(k): K.8 configuration, fast KC->KC inhibition engine builder (with_kc, k_make) and guarded K writes"
```

---

### Task 2: odd-turn (b) pairs, digest pin, odour-level overlap report

**Files:**
- Create: `flymon/brain/k_pairs.py`
- Modify: `flymon/brain/k_spec.py` (the `odd_pairs_digest` value only)
- Modify: `docs/superpowers/specs/2026-09-14-flymon-design.md` (one dated line at the end of K.8.1)
- Test: `tests/brain/test_k_pairs.py`

**Interfaces:**
- Consumes: `flymon.brain.h4_pairs.{pool_vocabulary, build_turns, e0_channels, odour, alternate_opponent, even_pairs, pair_key, pairs_digest}`; `Populations.from_connectome`.
- Produces: `odd_pairs(pops, n_turns=16) -> list[{axis, turn, x, y, odor_x, odor_y}]`; `odour_key(odor: dict) -> tuple`; `overlap_report(odd: list, others: dict[str, list]) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K.8.1 / K.8.7 (iii): the odd-turn (b) pairs and their odour-level overlap with the judged pairs."""
from pathlib import Path

import pytest

from flymon.brain import h4_pairs
from flymon.brain.k_pairs import odd_pairs, odour_key, overlap_report
from flymon.brain.k_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


def test_odour_key_ignores_order_and_float_noise():
    assert odour_key({"DA1": 0.5, "VA1d": 1.0}) == odour_key({"VA1d": 1.0 + 1e-15, "DA1": 0.5})
    assert odour_key({"DA1": 0.5}) != odour_key({"DA1": 0.25})


def test_overlap_report_finds_shared_pairs_and_odours():
    a, b, c = {"g1": 1.0}, {"g2": 1.0}, {"g3": 1.0}
    odd = [dict(axis="b", turn=1, x="m vs P", y="m vs Q", odor_x=a, odor_y=b)]
    even = [dict(axis="b", turn=0, x="n vs R", y="n vs S", odor_x=b, odor_y=a),
            dict(axis="b", turn=2, x="k vs T", y="k vs U", odor_x=c, odor_y=b)]
    rep = overlap_report(odd, {"even": even})
    assert rep["even"]["pairs"] == [{"odd": ["b", 1, "m vs P", "m vs Q"], "other": ["b", 0, "n vs R", "n vs S"]}]
    assert rep["even"]["n_shared_odours"] == 2


@needs_npz
def test_a_missing_alternate_opponent_raises_with_the_turn(monkeypatch):
    def boom(turn_index, me, opp_types, species_types):
        raise ValueError(f"turn {turn_index}: no type-disjoint alternate opponent")
    monkeypatch.setattr("flymon.brain.k_pairs.alternate_opponent", boom)
    with pytest.raises(ValueError, match="turn 1"):
        odd_pairs(_pops())


def _pops():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    return Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))


@needs_npz
def test_odd_pairs_are_the_even_rule_on_odd_turns_and_pinned():
    pops = _pops()
    odd = odd_pairs(pops)
    assert odd and all(p["axis"] == "b" and p["turn"] % 2 == 1 for p in odd)
    even_b = [p for p in h4_pairs.even_pairs(pops) if p["axis"] == "b"]
    assert len(even_b) == 21
    assert h4_pairs.pairs_digest(odd) == SPEC.odd_pairs_digest


@needs_npz
def test_odd_and_even_keys_are_disjoint_and_the_report_is_built():
    pops = _pops()
    odd, even = odd_pairs(pops), h4_pairs.even_pairs(pops)
    assert not {h4_pairs.pair_key(p) for p in odd} & {h4_pairs.pair_key(p) for p in even}
    rep = overlap_report(odd, {"even_b": [p for p in even if p["axis"] == "b"],
                               "even_a": [p for p in even if p["axis"] == "a"]})
    assert set(rep) == {"even_b", "even_a"} and all("n_shared_odours" in v for v in rep.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/brain/test_k_pairs.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.brain.k_pairs'`

- [ ] **Step 3: Write `flymon/brain/k_pairs.py`**

```python
"""The odd-turn (b) pairs of spec appendix K (K.2 / K.8.1): h4_pairs.even_pairs' (b) rule (E0 channels, G.11's
alternate opponent) on the odd turns 1, 3, ..., 15, used only to select the engine (no learning). Independence from the
even pairs is partial — the opponents are species the even turns also meet — so the check is at odour level: identical
normalised (odour X, odour Y) pairs and shared odours are reported, never used to drop a pair."""
from __future__ import annotations

from .h4_pairs import alternate_opponent, build_turns, e0_channels, odour, pair_key, pool_vocabulary


def odd_pairs(pops, n_turns: int = 16) -> list:
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    chan = e0_channels(pops, mon_types, move_types)
    out = []
    for t in build_turns(species_types, move_info, n_turns):
        if not t["turn"] % 2:
            continue
        alt = alternate_opponent(t["turn"], t["me"], t["opp_types"], species_types)   # raises with the turn number
        for c in t["candidates"]:
            od = lambda opp_types, c=c: odour(pops, chan, t["my_types"], opp_types, c["type"], c["bp"], t["my_hp"],
                                             t["opp_hp"])
            out.append({"axis": "b", "turn": t["turn"], "x": f"{c['move']} vs {t['opp']}", "y": f"{c['move']} vs {alt}",
                        "odor_x": od(t["opp_types"]), "odor_y": od(list(species_types[alt]))})
    return out


def odour_key(odor: dict) -> tuple:
    return tuple(sorted((str(g), round(float(v), 12)) for g, v in odor.items()))


def overlap_report(odd: list, others: dict) -> dict:
    """{name: {pairs: [{odd: key, other: key}], n_shared_odours}} — a pair matches when its two odours equal the other
    pair's two odours in either order."""
    def both(p):
        return frozenset((odour_key(p["odor_x"]), odour_key(p["odor_y"])))
    odd_odours = {odour_key(p[k]) for p in odd for k in ("odor_x", "odor_y")}
    rep = {}
    for name, pairs in others.items():
        index = {}
        for q in pairs:
            index.setdefault(both(q), []).append(q)
        hits = [{"odd": list(pair_key(p)), "other": list(pair_key(q))} for p in odd for q in index.get(both(p), [])]
        shared = odd_odours & {odour_key(q[k]) for q in pairs for k in ("odor_x", "odor_y")}
        rep[name] = {"pairs": hits, "n_shared_odours": len(shared)}
    return rep
```

- [ ] **Step 4: Pin the digest**

Run: `uv run python -c "from flymon.brain.connectome import Connectome; from flymon.brain.circuits import Populations; from flymon.brain.k_pairs import odd_pairs; from flymon.brain.h4_pairs import pairs_digest; p=odd_pairs(Populations.from_connectome(Connectome.load('data/malecns.npz'))); print(len(p), pairs_digest(p))"`
Expected: a pair count and a 64-hex digest. Put the digest into `KSpec.odd_pairs_digest` (replace `""`), with the count in the comment, e.g. `odd_pairs_digest: str = "<digest>"   # N odd-turn (b) pairs (Task 2)`. If this raises `ValueError: turn t: no type-disjoint alternate opponent`, stop and report BLOCKED with the message (the spec has no fallback).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_k_pairs.py -q -o addopts=""`
Expected: PASS (all, including the two real-data tests).

- [ ] **Step 6: Add the spec note** at the end of K.8.1 (after the line ending "겹침은 선택을 막지 않고 보고한다)."):

```markdown
**2026-09-26 (구현 계획)**: H.5 세트의 생성기는 아직 없다. 냄새 수준 비교는 짝수 턴 (b) 21쌍·(a) 18쌍과 하고, H.5와의 비교는 `SELECTED`일 때 확인 세트 생성기와 함께 한다.
홀수 턴 (b) 쌍은 N개이고 digest는 `k_spec.SPEC.odd_pairs_digest`에 고정했다.
```
(replace N with the count from Step 4).

- [ ] **Step 7: Full suite, then commit**

```bash
git add flymon/brain/k_pairs.py flymon/brain/k_spec.py tests/brain/test_k_pairs.py docs/superpowers/specs/2026-09-14-flymon-design.md
git commit -m "feat(k): K.8.1 odd-turn (b) pairs (even rule on odd turns), pinned digest, odour-level overlap report; spec note: H.5 comparison deferred to confirmation"
```

---

### Task 3: pure metrics

**Files:**
- Create: `flymon/brain/k_metrics.py`
- Test: `tests/brain/test_k_metrics.py`

**Interfaces:**
- Consumes: `Connectome` (`type`, `pre`, `post`, `w`, `N`), `Populations` (`kc`).
- Produces:
  - `readout_weights(conn, pops, mbon_type: str, min_weight: int) -> np.ndarray` (float64, len n_kc, synapse counts)
  - `lobe_masks(conn, pops) -> dict[str, np.ndarray]` keys `"apbp"`, `"g"`, `"ab"` (bool, len n_kc)
  - `kc_kc_edges(conn, pops, min_weight) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (pre KC position, post KC position, synapse count)
  - `pair_metrics(fx, fy, w13, w05) -> dict` keys `N, D13, S, N05`
  - `engine_metrics(acts: list[dict], w13, w05, lobes) -> dict` keys `N, N05, S_median, S, D13, N_p, jaccard_median, lobes, reliability`
  - `kc_kc_input(counts, edges, g, mv_per_synapse) -> np.ndarray`
  - `removed_fraction(fx, fy) -> float`

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K.8.3: N, S, D13 and the records, on synthetic masks and on H.4a.8's recorded even-turn masks."""
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.k_metrics import (engine_metrics, kc_kc_edges, kc_kc_input, lobe_masks, pair_metrics,
                                    readout_weights, removed_fraction)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/m0d/diag/h4_specificity_ceiling.json"
SUM = ROOT / "results/m0d/diag/h4_specificity_ceiling_summary.json"


def test_pair_metrics_on_a_known_mask():
    fx = np.array([1.0, 0.5, 0.25, 0.0])
    fy = np.array([0.0, 0.5, 0.0, 1.0])
    w13 = np.array([2.0, 4.0, 8.0, 16.0])
    w05 = np.array([1.0, 1.0, 1.0, 1.0])
    m = pair_metrics(fx, fy, w13, w05)
    assert m["N"] == 2.0 * 1.0 + 8.0 * 0.25                 # KCs 0 and 2 are X-only
    assert m["D13"] == 2.0 + 4.0 * 0.5 + 8.0 * 0.25
    assert m["S"] == pytest.approx(4.0 / 6.0)
    assert m["N05"] == 1.0 + 0.25


def test_no_drive_gives_zero_not_nan():
    z = np.zeros(3)
    m = pair_metrics(z, z, np.ones(3), np.ones(3))
    assert (m["N"], m["D13"], m["S"], m["N05"]) == (0.0, 0.0, 0.0, 0.0)


def test_engine_metrics_sums_n_keeps_every_pair_and_records():
    lobes = {"apbp": np.array([True, True, False, False]), "g": np.array([False, False, True, True]),
             "ab": np.zeros(4, bool)}
    w = np.array([1.0, 1.0, 1.0, 1.0])
    acts = [dict(fx=np.array([1.0, 0.0, 0.0, 0.0]), fy=np.zeros(4), cx=np.array([2.0, 0, 0, 0])),
            dict(fx=np.zeros(4), fy=np.zeros(4), cx=np.zeros(4)),                  # no drive: N 0, S 0, kept
            dict(fx=np.array([0.125, 1.0, 0, 0]), fy=np.array([1.0, 0, 0, 0]), cx=np.array([1.0, 3.0, 0, 0]))]
    m = engine_metrics(acts, w, w, lobes)
    assert m["N_p"] == [1.0, 0.0, 1.0] and m["N"] == 2.0 and len(m["S"]) == 3 and m["S"][1] == 0.0
    assert m["S_median"] == pytest.approx(float(np.median([1.0, 0.0, 1.0 / 1.125])))
    assert m["lobes"]["apbp"] == pytest.approx(np.mean([0.5, 0.0, 1.0]))
    assert m["reliability"]["n_active"] == 3 and m["reliability"]["hist"]["8"] == 2 and m["reliability"]["hist"]["1"] == 1
    assert m["jaccard_median"] == pytest.approx(float(np.median([0.0, 0.0, 0.5])))


def test_kc_kc_input_is_the_spike_count_proxy():
    edges = (np.array([0, 1]), np.array([2, 2]), np.array([10.0, 20.0]))
    got = kc_kc_input(np.array([1.0, 2.0, 0.0]), edges, -0.5, 0.275)
    assert got == pytest.approx([0.0, 0.0, 0.5 * 0.275 * (1.0 * 10 + 2.0 * 20)])


def test_connectome_helpers_on_the_synthetic_connectome(synthetic_connectome):
    from flymon.brain.circuits import Populations
    conn = synthetic_connectome()
    pops = Populations.from_connectome(conn)
    w = readout_weights(conn, pops, "MBON01", 5)
    assert w.shape == (len(pops.kc),) and (w >= 0).all()
    lobes = lobe_masks(conn, pops)
    assert lobes["ab"].all() and not lobes["g"].any()       # the synthetic KCs are all "KCab-m"
    pre, post, ww = kc_kc_edges(conn, pops, 5)
    assert pre.size == post.size == ww.size == 0             # no KC->KC edges in the synthetic build


@pytest.mark.skipif(not (RAW.exists() and SUM.exists()), reason="H.4a.8 raw records are git-excluded")
def test_removed_fraction_reproduces_h4a8_on_the_recorded_even_masks():
    """K.8.7 (ii): the mask measure on already-used data (not selection) equals H.4a.8's recorded values."""
    raw, summ = json.loads(RAW.read_text()), json.loads(SUM.read_text())
    rows = [r for r in raw["rows"]["C3"] if r["mode"] == "x_only"]
    rec = {(p["axis"], p["turn"], p["x"], p["y"]): p["mask"]["removed"] for p in summ["engines"]["C3"]["pairs"]}
    got = {(r["axis"], r["turn"], r["x"], r["y"]): removed_fraction(np.array(r["fx"]), np.array(r["fy"])) for r in rows}
    assert set(got) == set(rec)
    assert all(abs(got[k] - rec[k]) < 1e-9 for k in got)
    b = [got[k] for k in got if k[0] == "b"]
    assert abs(float(np.median(b)) - summ["engines"]["C3"]["groups"]["b"]["removed"]) < 1e-9
```

Note for the implementer: the summary key path (`engines → C3 → pairs[] → mask.removed`, `engines → C3 → groups → b → removed`) was read from the file's structure; if it differs, open the file with a short Python snippet and fix the **path** in the test, never the formula. `removed` in H.4a.8 is Σ f_x over KCs with f_y > 0 divided by Σ f_x (`h4_specificity_ceiling.py:517-535`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/brain/test_k_metrics.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.brain.k_metrics'`

- [ ] **Step 3: Write `flymon/brain/k_metrics.py`**

```python
"""The metrics of spec appendix K (K.8.3), pure functions over per-KC fire fractions f (fraction of act seeds with at
least one read-window spike, H.4a.7) and synapse-count weights (edges >= min_weight).

N_p = sum w13 f_x [f_y = 0] (X-only drive onto the target readout), D13 = sum w13 f_x, S = N_p / D13 (0 without drive),
N05 = the reward readout's N. An engine's N is the sum over the fixed pair list: no pair is dropped. Everything else is
recorded, never judged."""
from __future__ import annotations

import numpy as np

LOBES = {"apbp": "KCa'b'", "g": "KCg", "ab": "KCab"}


def _kc_pos(conn, pops) -> np.ndarray:
    pos = np.full(conn.N, -1, np.int64)
    pos[np.asarray(pops.kc, np.int64)] = np.arange(len(pops.kc))
    return pos


def readout_weights(conn, pops, mbon_type: str, min_weight: int) -> np.ndarray:
    pos = _kc_pos(conn, pops)
    cells = np.flatnonzero(np.asarray(conn.type).astype(str) == mbon_type)
    sel = (pos[conn.pre] >= 0) & np.isin(conn.post, cells) & (conn.w >= min_weight)
    return np.bincount(pos[conn.pre[sel]], weights=conn.w[sel].astype(np.float64), minlength=len(pops.kc))


def lobe_masks(conn, pops) -> dict:
    t = np.asarray(conn.type).astype(str)[np.asarray(pops.kc, np.int64)]
    return {k: np.char.startswith(t, prefix) for k, prefix in LOBES.items()}


def kc_kc_edges(conn, pops, min_weight: int) -> tuple:
    pos = _kc_pos(conn, pops)
    sel = (pos[conn.pre] >= 0) & (pos[conn.post] >= 0) & (conn.w >= min_weight)
    return pos[conn.pre[sel]], pos[conn.post[sel]], conn.w[sel].astype(np.float64)


def pair_metrics(fx, fy, w13, w05) -> dict:
    only = fx * (fy == 0)
    n, d = float(np.dot(w13, only)), float(np.dot(w13, fx))
    return dict(N=n, D13=d, S=n / d if d > 0 else 0.0, N05=float(np.dot(w05, only)))


def removed_fraction(fx, fy) -> float:
    """H.4a.8's mask measure: the share of X's activity on KCs Y also fires."""
    s = float(fx.sum())
    return float(fx[fy > 0].sum()) / s if s > 0 else 0.0


def engine_metrics(acts: list, w13, w05, lobes: dict) -> dict:
    per = [pair_metrics(a["fx"], a["fy"], w13, w05) for a in acts]
    jac = []
    for a in acts:
        x, y = a["fx"] > 0, a["fy"] > 0
        u = int((x | y).sum())
        jac.append(int((x & y).sum()) / u if u else 0.0)
    fx_all = np.concatenate([a["fx"] for a in acts]) if acts else np.zeros(0)
    act = fx_all[fx_all > 0]
    eighths = np.rint(act * 8).astype(int)
    return dict(N=float(sum(m["N"] for m in per)), N05=float(sum(m["N05"] for m in per)),
                S_median=float(np.median([m["S"] for m in per])) if per else 0.0,
                S=[m["S"] for m in per], D13=[m["D13"] for m in per], N_p=[m["N"] for m in per],
                jaccard_median=float(np.median(jac)) if jac else 0.0,
                lobes={k: float(np.mean([float((a["fx"][m] > 0).mean()) if m.any() else 0.0 for a in acts]))
                       for k, m in lobes.items()},
                reliability=dict(n_active=int(act.size), frac_always=float((act == 1.0).mean()) if act.size else 0.0,
                                 hist={str(k): int((eighths == k).sum()) for k in range(1, 9)}))


def kc_kc_input(counts, edges, g: float, mv_per_synapse: float) -> np.ndarray:
    """Per-KC inhibitory charge proxy: sum over KC->KC in-edges of the presynaptic mean count x synapses x |g| x mV."""
    pre, post, w = edges
    return np.bincount(post, weights=counts[pre] * w * abs(g) * mv_per_synapse, minlength=len(counts))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_k_metrics.py -q -o addopts=""`
Expected: PASS. The H.4a.8 test must run (not skip) on the controller's worktree; if it is skipped for you, say so in the report.

- [ ] **Step 5: Full suite, then commit**

```bash
git add flymon/brain/k_metrics.py tests/brain/test_k_metrics.py
git commit -m "feat(k): K.8.3 metrics — X-only MBON13 drive N over the fixed pair list, S, D13, N05, reliability, lobes, KC->KC input proxy; H.4a.8 mask measure reproduced"
```

---

### Task 4: activity and raster jobs, cached measurer, key files

**Files:**
- Create: `flymon/brain/k_jobs.py`, `flymon/brain/k_measure.py`
- Test: `tests/brain/test_k_measure.py`

**Interfaces:**
- Consumes: `h4_jobs.rig_for(conn, pops, params)`, `h4_jobs._present_kc(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms) -> {read, max_win}`; `stimuli.present`; `h3_measure.chunks(items, n)`; `MeasureCache.get_or_compute(kind, inputs, compute, params_list)`; `pool.run_jobs(fn, kwargs_list) -> list`, `pool.n_workers`; `j_measure.MEASURE_FILES`, `HASHED_FILES`; `k_pairs.odour_key`.
- Produces: `k_jobs.activity_job(eng, pl, pops, comps, ro, params, items, strength, settle_ms, read_ms, window_ms) -> list[{i, seed, kc, n, max_win}]`; `k_jobs.raster_job(eng, pl, pops, comps, ro, params, odor, seed, strength, settle_ms, read_ms, probe) -> {spikes, v_mean}`; `k_measure.MEASURE_FILES`, `HASHED_FILES`; `KMeasurer(pool, spec, pairs, n_kc, cache)` with `.odours`, `.index`, `.params_seen`, `.activity(params) -> list[{fx, fy, cx}]`, `.raster(params, odor, seed, probe) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K.8.2: per-KC activity measured once per (odour, seed), cached under a content key, shared by duplicate odours."""
import numpy as np
import pytest

from flymon.brain import k_jobs
from flymon.brain.config import Params
from flymon.brain.h3_store import MeasureCache
from flymon.brain.k_measure import HASHED_FILES, MEASURE_FILES, KMeasurer
from flymon.brain.k_spec import SPEC


class FakePool:
    n_workers = 2

    def __init__(self):
        self.calls = []

    def run_jobs(self, fn, kwargs_list):
        assert fn is k_jobs.activity_job
        self.calls.append(kwargs_list)
        out = []
        for kw in kwargs_list:
            part = []
            for i, odor, seed in kw["items"]:
                k = int(sum(odor.values()) * 10) % 5                    # deterministic fake: KC k fires on even seeds
                part.append(dict(i=i, seed=seed, kc=[k] if seed % 2 == 0 else [], n=[3] if seed % 2 == 0 else [],
                                 max_win=1))
            out.append(part)
        return out


PAIRS = [dict(axis="b", turn=1, x="a", y="b", odor_x={"g1": 0.1}, odor_y={"g2": 0.2}),
         dict(axis="b", turn=3, x="c", y="d", odor_x={"g2": 0.2}, odor_y={"g3": 0.3})]     # odour g2 appears twice


def test_files_extend_js_and_name_k_modules():
    assert "flymon/brain/k_jobs.py" in MEASURE_FILES and "flymon/brain/j_jobs.py" in MEASURE_FILES
    assert "flymon/brain/k_metrics.py" not in MEASURE_FILES and "flymon/brain/k_metrics.py" in HASHED_FILES
    assert {"scripts/run_k_scan.py", "scripts/run_k_judge.py", "flymon/brain/k_rules.py"} <= set(HASHED_FILES)


def test_duplicate_odours_are_measured_once_and_fractions_are_per_seed(tmp_path):
    code = {"key": "k" * 64}
    pool = FakePool()
    km = KMeasurer(pool, SPEC, PAIRS, 5, MeasureCache(tmp_path / "cache", code, "r1"))
    assert len(km.odours) == 3 and km.index == [(0, 1), (1, 2)]
    acts = km.activity(Params())
    n_items = sum(len(kw["items"]) for kw in pool.calls[0])
    assert n_items == 3 * len(SPEC.j.h4.act_seeds)
    k1 = int(0.1 * 10) % 5
    assert acts[0]["fx"][k1] == 0.5 and acts[0]["cx"][k1] == 1.5      # 4 of 8 seeds, 3 spikes each
    assert np.array_equal(acts[0]["fy"], acts[1]["fx"])                # the shared odour is one measurement
    KMeasurer(pool, SPEC, PAIRS, 5, MeasureCache(tmp_path / "cache", code, "r2")).activity(Params())
    assert len(pool.calls) == 1                                        # the rerun is a cache hit (resume)


@pytest.mark.skipif(True, reason="engine run; exercised by the controller's smoke run (Run order R0)")
def test_activity_job_on_the_engine():
    pass
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/brain/test_k_measure.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.brain.k_jobs'`

- [ ] **Step 3: Write `flymon/brain/k_jobs.py`**

```python
"""FlyPool worker jobs of spec appendix K (K.8.2): per-KC read-window counts for a batch of (odour, seed) items on the
naive brain (plasticity off, weights reset — H.4a.7's ceiling_job presentation through h4_jobs._present_kc), and one
raster with the mean membrane of probe KCs.

Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them. The
rig is built from the Params given (h4_jobs.rig_for), so a job carries its g; the worker's engine only lends its
connectome."""
from __future__ import annotations

import numpy as np

from .h4_jobs import _present_kc, rig_for
from .stimuli import present


def activity_job(eng, pl, pops, comps, ro, params, items, strength: float, settle_ms: float, read_ms: float,
                 window_ms: int) -> list:
    e, p, _ = rig_for(eng.conn, pops, params)
    out = []
    try:
        p.reset_weights()
        p.set_enabled(False)
        for i, odor, seed in items:
            o = _present_kc(e, p, pops, odor, int(seed), strength, settle_ms, read_ms, int(window_ms))
            nz = np.flatnonzero(o["read"])
            out.append(dict(i=int(i), seed=int(seed), kc=nz.tolist(), n=o["read"][nz].astype(int).tolist(),
                            max_win=int(o["max_win"])))
    finally:
        p.reset_weights()
        p.set_enabled(True)
    return out


def raster_job(eng, pl, pops, comps, ro, params, odor, seed: int, strength: float, settle_ms: float, read_ms: float,
               probe) -> dict:
    """KC spikes [(step, KC position)] over settle + read and the mean membrane of `probe` (global indices) per step."""
    e, p, _ = rig_for(eng.conn, pops, params)
    kc = np.asarray(pops.kc, np.int64)
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    probe = np.asarray(probe, np.int64)
    spikes, v_mean = [], []
    try:
        p.reset_weights(); p.set_enabled(False)
        e.reset(int(seed)); p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, odor, strength)
        for step in range(int(round((settle_ms + read_ms) / e.p.dt))):
            f = pos[e.step()]; f = f[f >= 0]
            spikes += [(step, int(k)) for k in f]
            v_mean.append(float(e.v[probe].mean()) if probe.size else 0.0)
    finally:
        p.reset_weights(); p.set_enabled(True)
    return dict(spikes=spikes, v_mean=v_mean)
```

- [ ] **Step 4: Write `flymon/brain/k_measure.py`**

```python
"""The measurement layer of spec appendix K: H.3's and H.4's measurers run unchanged (through J's files), plus the
per-KC activity on the odd-turn (b) pairs and one raster (K.8.2).

Resume (H.3a.12's form): every measurement is cached under a content key — `h3_store.MeasureCache` keyed by the files a
result depends on (`MEASURE_FILES`), the NPZ and the inputs (the full Params, which carry g). The procedure modules
(metrics, rules, runner, store, scripts) are in `HASHED_FILES` (dirty check and manifest), not in the key."""
from __future__ import annotations

import numpy as np

from . import k_jobs
from .h3_measure import chunks
from .j_measure import HASHED_FILES as J_HASHED_FILES, MEASURE_FILES as J_MEASURE_FILES
from .k_pairs import odour_key

MEASURE_FILES = tuple(dict.fromkeys(J_MEASURE_FILES + (
    "flymon/brain/k_params.py", "flymon/brain/k_pairs.py", "flymon/brain/k_jobs.py", "flymon/brain/k_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + J_HASHED_FILES + (
    "flymon/brain/k_spec.py", "flymon/brain/k_metrics.py", "flymon/brain/k_rules.py", "flymon/brain/k_runner.py",
    "flymon/brain/k_store.py", "scripts/run_k_scan.py", "scripts/run_k_judge.py")))


class KMeasurer:
    def __init__(self, pool, spec, pairs: list, n_kc: int, cache):
        """pairs: k_pairs.odd_pairs; identical odours (odour_key) are measured once and shared."""
        self.pool, self.spec, self.pairs, self.n_kc, self.cache = pool, spec, pairs, int(n_kc), cache
        self.params_seen: list = []
        self.odours, self.index, seen = [], [], {}
        for p in pairs:
            ij = []
            for k in ("odor_x", "odor_y"):
                key = odour_key(p[k])
                if key not in seen:
                    seen[key] = len(self.odours)
                    self.odours.append({str(g): float(v) for g, v in p[k].items()})
                ij.append(seen[key])
            self.index.append(tuple(ij))

    def _seen(self, params):
        if params not in self.params_seen:
            self.params_seen.append(params)

    def _common(self, params) -> dict:
        h4 = self.spec.j.h4
        return dict(params=params, strength=h4.h3.strength, settle_ms=h4.oracle_window.settle_ms,
                    read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms))

    def activity(self, params) -> list:
        """[{fx, fy, cx}] per pair: fire fraction over act seeds and mean read-window count (cx for X)."""
        self._seen(params)
        seeds = [int(s) for s in self.spec.j.h4.act_seeds]
        common = self._common(params)
        items = [(i, o, s) for i, o in enumerate(self.odours) for s in seeds]
        jobs = [dict(common, items=c) for c in chunks(items, self.pool.n_workers)]
        rows = self.cache.get_or_compute("k_act", dict(common, odours=self.odours, seeds=seeds),
                                         lambda: [r for part in self.pool.run_jobs(k_jobs.activity_job, jobs)
                                                  for r in part], [params])
        fired = np.zeros((len(self.odours), self.n_kc))
        counts = np.zeros((len(self.odours), self.n_kc))
        for r in rows:
            fired[r["i"], r["kc"]] += 1
            counts[r["i"], r["kc"]] += r["n"]
        fired /= len(seeds)
        counts /= len(seeds)
        return [dict(fx=fired[ix], fy=fired[iy], cx=counts[ix]) for ix, iy in self.index]

    def raster(self, params, odor: dict, seed: int, probe) -> dict:
        self._seen(params)
        c = self._common(params)
        kw = dict(params=params, odor=odor, seed=int(seed), strength=c["strength"], settle_ms=c["settle_ms"],
                  read_ms=c["read_ms"], probe=[int(x) for x in probe])
        return self.cache.get_or_compute("k_raster", kw, lambda: self.pool.run_jobs(k_jobs.raster_job, [kw])[0],
                                         [params])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_k_measure.py -q -o addopts=""`
Expected: PASS (the engine test is skipped by design). If `MeasureCache.get_or_compute` requires a `run_jobs` result that is JSON-canonical, the lists above already are.

- [ ] **Step 6: Full suite, then commit**

```bash
git add flymon/brain/k_jobs.py flymon/brain/k_measure.py tests/brain/test_k_measure.py
git commit -m "feat(k): K.8.2 activity and raster jobs (naive brain, H.4a.7 presentation), cached KMeasurer sharing duplicate odours, K key files"
```

---

### Task 5: selection, gate, closing states

**Files:**
- Create: `flymon/brain/k_rules.py`
- Test: `tests/brain/test_k_rules.py`

**Interfaces:**
- Consumes: `h3_rules.COMBO_ADOPTED`; `j_rules.{SELECTED, B, INVALID, STOP_MULTI_TYPE, COMPUTE_ABORTED}`.
- Produces: constants `SCAN_GO = "scan_go"`, `STOP_NO_QUALIFIED_SETTING = "stop_no_qualified_setting"`, `STOP_NO_TARGET_GAIN = "stop_no_target_gain"`, `DROPPED_NO_READOUT = "dropped_no_readout"`; `select_order(settings, n_c3, tie_tol) -> list[int]`; `scan_outcome(settings, order, n_c3, gate_ratio) -> str`; `closing_state(judge: dict) -> str`. A setting is `{g, status, metrics: {N, S_median, ...} | None, ...}`.

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K.8.3–K.8.5: ranking of ADOPTED engines by N, the gate, and the closing state of every branch."""
import pytest

from flymon.brain.h3_rules import COMBO_ADOPTED, COMBO_DROPPED
from flymon.brain.j_rules import B, COMPUTE_ABORTED, INVALID, SELECTED, STOP_MULTI_TYPE
from flymon.brain.k_rules import (DROPPED_NO_READOUT, SCAN_GO, STOP_NO_QUALIFIED_SETTING, STOP_NO_TARGET_GAIN,
                                  closing_state, scan_outcome, select_order)


def st(g, n=None, s=0.5, status=COMBO_ADOPTED):
    return dict(g=g, status=status, metrics=None if n is None else dict(N=n, S_median=s))


def test_order_by_n_only_adopted():
    s = [st(-0.05, 10), st(-0.1, 30), st(-0.2, status=COMBO_DROPPED), st(-0.4, 20)]
    assert select_order(s, 10.0, 0.01) == [1, 3, 0]


def test_ties_within_one_percent_go_to_larger_s_then_smaller_g():
    s = [st(-0.4, 100.0, s=0.4), st(-0.1, 99.5, s=0.6), st(-0.2, 99.5, s=0.6), st(-0.05, 50.0)]
    assert select_order(s, 10.0, 0.01) == [1, 2, 0, 3]


def test_a_non_positive_c3_drive_stops():
    with pytest.raises(ValueError, match="N_C3"):
        select_order([st(-0.1, 5)], 0.0, 0.01)


def test_scan_outcomes():
    assert scan_outcome([st(-0.1, status=COMBO_DROPPED)], [], 10.0, 1.0) == STOP_NO_QUALIFIED_SETTING
    s = [st(-0.1, 10.0), st(-0.2, 9.0)]
    assert scan_outcome(s, select_order(s, 10.0, 0.01), 10.0, 1.0) == STOP_NO_TARGET_GAIN   # ratio exactly 1.0
    s = [st(-0.1, 10.0001)]
    assert scan_outcome(s, [0], 10.0, 1.0) == SCAN_GO


@pytest.mark.parametrize("judge,state", [
    (dict(outcome=STOP_MULTI_TYPE, reading=None), "stop_multiple_types"),
    (dict(outcome=B, reading=None, reason="no reactive and teachable readout in a pool"), DROPPED_NO_READOUT),
    (dict(outcome=INVALID, reading=None), INVALID),
    (dict(outcome=COMPUTE_ABORTED), COMPUTE_ABORTED),
    (dict(outcome=SELECTED, reading=dict(band=SELECTED)), SELECTED),
    (dict(outcome=B, reading=dict(band="B_Tb")), "B_Tb"),
    (dict(outcome=B, reading=dict(band="B_NO_CONCLUSION")), "B_NO_CONCLUSION"),
    (dict(outcome=B, reading=dict(band="B_Fa")), "B_Fa"),
])
def test_every_branch_has_a_state(judge, state):
    assert closing_state(judge) == state


def test_a_bandless_reading_is_an_error():
    with pytest.raises(ValueError, match="band"):
        closing_state(dict(outcome=B, reading=dict(band=None)))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/brain/test_k_rules.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.brain.k_rules'`

- [ ] **Step 3: Write `flymon/brain/k_rules.py`**

```python
"""The rules of spec appendix K (K.8.3–K.8.5): rank the ADOPTED engines by N (ties within tie_tol relative -> larger S
median, then smaller |g|), gate on N / N_C3 > gate_ratio, and name the closing state of a judgement. Pure functions."""
from __future__ import annotations

from .h3_rules import COMBO_ADOPTED
from .j_rules import COMPUTE_ABORTED, INVALID, STOP_MULTI_TYPE

SCAN_GO = "scan_go"
STOP_NO_QUALIFIED_SETTING = "stop_no_qualified_setting"
STOP_NO_TARGET_GAIN = "stop_no_target_gain"
DROPPED_NO_READOUT = "dropped_no_readout"


def select_order(settings: list, n_c3: float, tie_tol: float) -> list:
    if not n_c3 > 0:
        raise ValueError(f"N_C3 = {n_c3!r}: the ratio N / N_C3 is undefined (stop and ask the user, K.8.3)")
    left = [i for i, s in enumerate(settings) if s["status"] == COMBO_ADOPTED]
    order = []
    while left:
        best = max(settings[i]["metrics"]["N"] for i in left)
        tied = [i for i in left if settings[i]["metrics"]["N"] >= best * (1.0 - tie_tol)]
        pick = max(tied, key=lambda i: (settings[i]["metrics"]["S_median"], -abs(settings[i]["g"])))
        order.append(pick)
        left.remove(pick)
    return order


def scan_outcome(settings: list, order: list, n_c3: float, gate_ratio: float) -> str:
    if not order:
        return STOP_NO_QUALIFIED_SETTING
    return SCAN_GO if settings[order[0]]["metrics"]["N"] / n_c3 > gate_ratio else STOP_NO_TARGET_GAIN


def closing_state(judge: dict) -> str:
    o = judge["outcome"]
    if o in (STOP_MULTI_TYPE, INVALID, COMPUTE_ABORTED):
        return o
    rd = judge.get("reading")
    if rd is None:
        return DROPPED_NO_READOUT                     # j_runner.judge: B without a band (no readout in a pool)
    if rd.get("band") is None:
        raise ValueError(f"reading without a band: {rd!r} (off the declared 21 pairs is not a judgement)")
    return rd["band"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_k_rules.py -q -o addopts=""`
Expected: PASS

- [ ] **Step 5: Full suite, then commit**

```bash
git add flymon/brain/k_rules.py tests/brain/test_k_rules.py
git commit -m "feat(k): K.8.3-K.8.5 rules — rank ADOPTED engines by N with declared ties, gate N/N_C3 > 1, closing state of every judgement branch"
```

---

### Task 6: runner — stage 1 and stage 2

**Files:**
- Create: `flymon/brain/k_runner.py`
- Test: `tests/brain/test_k_runner.py`

**Interfaces:**
- Consumes: `j_runner.{reconverge, judge, measure_d6, params_json}`; `h3_runner.{Deadline, ComputeAborted}`; `h3_rules.{COMBO_ABORTED, COMBO_ADOPTED}`; `j_params.params_from_json`; `j_rules.{B, SELECTED, COMPUTE_ABORTED}`; Task 1 `k_make`; Task 3 `readout_weights, lobe_masks, kc_kc_edges, engine_metrics, kc_kc_input`; Task 5 `select_order, scan_outcome, closing_state, SCAN_GO`.
- Produces: `KArrays(w13, w05, lobes, kk, probe, mv)`; `arrays(conn, pops, spec, mv_per_synapse) -> KArrays`; `measure(km, params, arr, g) -> dict`; `stage1(m3, km, jctx, spec, c3, arr, jm=None) -> {outcome, base, settings, order, ratios}` (with a `JMeasurer`, each metrics dict gains `all51_kc_on_log10_var`); `stage2(m4, jm, jctx, setting, with_d6=True, d6_seeds=None) -> {name, g, params, judge, outcome, d6, state}`.

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K.8.2 / K.8.4: stage 1 re-converges every g, measures only ADOPTED engines, ranks and gates; stage 2 judges one."""
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.brain import k_runner as K
from flymon.brain.config import Params
from flymon.brain.h3_rules import COMBO_ABORTED, COMBO_ADOPTED, COMBO_DROPPED
from flymon.brain.j_rules import B, COMPUTE_ABORTED
from flymon.brain.k_params import with_kc
from flymon.brain.k_rules import SCAN_GO, STOP_NO_QUALIFIED_SETTING, STOP_NO_TARGET_GAIN
from flymon.brain.k_spec import SPEC

ARR = K.KArrays(w13=np.ones(4), w05=np.ones(4), lobes={"apbp": np.array([1, 1, 0, 0], bool)},
                kk=(np.array([0]), np.array([1]), np.array([10.0])), probe=np.array([0, 1]), mv=0.275)
JCTX = SimpleNamespace(log=lambda s: None, h3=SimpleNamespace(spec=SPEC.j.h4.h3, deadline=None))


class FakeKM:
    """One pair whose X-only mass on KC 0 is n_of_g(g) (default 1 + 10|g|; w13 = 1, so N = n_of_g(g)). The value is a
    stand-in, not a fraction: the runner only sums it."""
    def __init__(self, n_of_g=None):
        self.params_seen, self.odours, self.index = [], [{"g1": 1.0}], [(0, 0)]
        self.n_of_g = n_of_g or (lambda g: 1.0 + 10 * abs(g))
        self.rasters = []

    def activity(self, params):
        n = self.n_of_g(params.kc_kc_scale)
        return [dict(fx=np.array([n, 0.0, 0.0, 0.0]), fy=np.zeros(4), cx=np.array([2.0, 0.0, 0.0, 0.0]))]

    def raster(self, params, odor, seed, probe):
        self.rasters.append(params.kc_kc_scale)
        return dict(spikes=[], v_mean=[])


def fake_reconverge(adopt):
    def rc(m3, ctx, make):
        p = make(1.65)
        g = p.kc_kc_scale
        if adopt(g) == "abort":
            return dict(status=COMBO_ABORTED, cells=[], adopted=None, guard=None, note="budget")
        if adopt(g):
            return dict(status=COMBO_ADOPTED, cells=[dict(label="G", boundary_share=0.01)],
                        adopted=dict(label="G", params=p, reference_median_kc_pct=6.0), guard={"MBON13": {}})
        return dict(status=COMBO_DROPPED, cells=[], adopted=None, guard=None)
    return rc


def test_stage1_ranks_adopted_engines_by_n_and_goes(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: g != -0.4))
    km = FakeKM()
    r = K.stage1(object(), km, JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == SCAN_GO
    gs = [s["g"] for s in r["settings"]]
    assert gs == list(SPEC.grid) and r["settings"][3]["metrics"] is None
    assert [r["settings"][i]["g"] for i in r["order"]] == [-0.8, -0.2, -0.1, -0.05]
    assert r["ratios"][r["order"][0]] == pytest.approx(9.0)
    assert km.rasters == [-0.8] and "kc_kc_input" in r["settings"][4]["metrics"]
    assert r["base"]["metrics"]["all51_kc_on_log10_var"] is None       # no JMeasurer given


def test_stage1_without_gain_stops_before_any_even_pair(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: True))
    r = K.stage1(object(), FakeKM(lambda g: 1.0 - abs(g)), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == STOP_NO_TARGET_GAIN


def test_stage1_without_an_adopted_engine(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: False))
    r = K.stage1(object(), FakeKM(), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == STOP_NO_QUALIFIED_SETTING and r["order"] == []


def test_an_abort_is_compute_aborted_not_invalid(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: "abort" if g == -0.2 else True))
    r = K.stage1(object(), FakeKM(), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == COMPUTE_ABORTED and [s["g"] for s in r["settings"]] == [-0.05, -0.1, -0.2]


def test_stage2_judges_the_setting_and_names_the_state(monkeypatch):
    p = with_kc(Params(apl_mode="graded"), -0.2)
    seen = {}
    def judge(m4, ctx, name, params, guard):
        seen.update(name=name, params=params, guard=guard)
        return dict(outcome=B, reading=None, reason="no reactive and teachable readout in a pool")
    monkeypatch.setattr(K, "judge", judge)
    monkeypatch.setattr(K, "measure_d6", lambda jm, ctx, params, seeds: {"a": 1})
    setting = dict(g=-0.2, reconverge=dict(adopted=dict(params=K.params_json(p)), guard={"MBON13": {"x": 1}}))
    r = K.stage2(object(), object(), JCTX, setting, with_d6=True, d6_seeds=(1, 2))
    assert seen["params"] == p and seen["guard"] == {"MBON13": {"x": 1}} and "g=-0.2" in seen["name"]
    assert r["state"] == "dropped_no_readout" and r["d6"] == {"a": 1}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/brain/test_k_runner.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: No module named 'flymon.brain.k_runner'`

- [ ] **Step 3: Write `flymon/brain/k_runner.py`**

```python
"""The runner of spec appendix K (K.8.2 / K.8.4).

stage1: for every g in the grid, re-converge C3's rule (j_runner.reconverge with k_make, J.10.4's state machine: a
setting whose candidates all fail is COMBO_DROPPED = INVALID_ENGINE); measure N and the records on the odd-turn (b)
pairs for ADOPTED engines only; rank (k_rules.select_order) against C3's recorded engine and gate (scan_outcome).
stage2: J.11.4 steps 2-3 on the chosen engine — j_runner.judge (H.4 readout reselection, z, oracle on the even pairs,
stage2_reading) and D.6 when the outcome is SELECTED or B. j_runner.stage2 is not used: it forces the depression on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import d6a
from .h3_rules import COMBO_ABORTED, COMBO_ADOPTED
from .h3_runner import ComputeAborted, Deadline
from .j_params import params_from_json
from .j_rules import B, COMPUTE_ABORTED, SELECTED, scan_metrics
from .j_runner import judge, measure_d6, params_json, reconverge
from .k_metrics import engine_metrics, kc_kc_edges, kc_kc_input, lobe_masks, readout_weights
from .k_params import k_make
from .k_rules import SCAN_GO, closing_state, scan_outcome, select_order


@dataclass
class KArrays:
    w13: np.ndarray
    w05: np.ndarray
    lobes: dict
    kk: tuple
    probe: np.ndarray          # global indices of the alpha'beta' KCs (raster membrane)
    mv: float


def arrays(conn, pops, spec, mv_per_synapse: float) -> KArrays:
    lobes = lobe_masks(conn, pops)
    return KArrays(w13=readout_weights(conn, pops, spec.target_type, spec.min_weight),
                   w05=readout_weights(conn, pops, spec.reward_type, spec.min_weight), lobes=lobes,
                   kk=kc_kc_edges(conn, pops, spec.min_weight),
                   probe=np.asarray(pops.kc, np.int64)[lobes["apbp"]], mv=float(mv_per_synapse))


def measure(km, params, arr: KArrays, g: float) -> dict:
    acts = km.activity(params)
    met = engine_metrics(acts, arr.w13, arr.w05, arr.lobes)
    cur = kc_kc_input(np.mean([a["cx"] for a in acts], axis=0), arr.kk, g, arr.mv)
    met["kc_kc_input"] = {k: float(cur[m].mean()) if m.any() else 0.0 for k, m in arr.lobes.items()}
    return met


def _all51(jm, params):
    """K.8.3's all51 kc_on log10 variance record (J's all51 measurement), or None without a JMeasurer."""
    return None if jm is None else scan_metrics(jm.all51_uni(params))["log10_var"]["kc_on"]


def stage1(m3, km, jctx, spec, c3, arr: KArrays, jm=None) -> dict:
    first_x = km.odours[km.index[0][0]] if km.index else None
    seed0 = int(spec.j.h4.act_seeds[0])
    m3, kmd = Deadline(m3, jctx.h3), Deadline(km, jctx.h3)
    jmd = None if jm is None else Deadline(jm, jctx.h3)
    base, settings = None, []
    try:
        base = dict(g=0.0, name="C3", params=params_json(c3), metrics=measure(kmd, c3, arr, 0.0))
        base["metrics"]["all51_kc_on_log10_var"] = _all51(jmd, c3)
        jctx.log(f"K stage 1: C3 N {base['metrics']['N']:.4g}")
        for g in spec.grid:
            jctx.log(f"K stage 1: g = {g:g}, C3 rule re-convergence")
            rc = reconverge(m3, jctx, k_make(jctx.h3.spec, g))
            st = dict(g=g, status=rc["status"], reconverge=rc, metrics=None)
            settings.append(st)
            if rc["status"] == COMBO_ABORTED:
                return dict(outcome=COMPUTE_ABORTED, base=base, settings=settings, order=[], ratios={},
                            note=rc.get("note"))
            if rc["status"] != COMBO_ADOPTED:
                jctx.log(f"K stage 1: g = {g:g} INVALID_ENGINE")
                continue
            p = rc["adopted"]["params"]
            st["metrics"] = measure(kmd, p, arr, g)
            st["metrics"]["all51_kc_on_log10_var"] = _all51(jmd, p)
            st["boundary_share"] = next((c.get("boundary_share") for c in rc["cells"]
                                         if c.get("label") == rc["adopted"]["label"]), None)
            if g == spec.raster_g and first_x is not None:
                st["raster"] = kmd.raster(p, first_x, seed0, arr.probe)
            jctx.log(f"K stage 1: g = {g:g} ADOPTED kc_thresh {p.kc_thresh:g}, N {st['metrics']['N']:.4g}")
    except ComputeAborted as e:
        return dict(outcome=COMPUTE_ABORTED, base=base, settings=settings, order=[], ratios={}, note=str(e))
    if not any(s["g"] == spec.raster_g and "raster" in s for s in settings):
        for s in settings:
            if s["g"] == spec.raster_g:
                s["raster"] = None                     # recorded: the raster setting was not ADOPTED
    n_c3 = base["metrics"]["N"]
    order = select_order(settings, n_c3, spec.tie_tol)
    ratios = {i: settings[i]["metrics"]["N"] / n_c3 for i in order}
    return dict(outcome=scan_outcome(settings, order, n_c3, spec.gate_ratio), base=base, settings=settings,
                order=order, ratios=ratios)


def stage2(m4, jm, jctx, setting: dict, with_d6: bool = True, d6_seeds=None) -> dict:
    p = params_from_json(setting["reconverge"]["adopted"]["params"])
    guard = setting["reconverge"]["guard"]
    name = f"KC-KC g={setting['g']:g} kc_thresh={p.kc_thresh:g}"
    m4, jm = Deadline(m4, jctx.h3), Deadline(jm, jctx.h3)
    try:
        j = judge(m4, jctx, name, p, guard)
        d6 = measure_d6(jm, jctx, p, d6a.SEEDS if d6_seeds is None else d6_seeds) \
            if with_d6 and j["outcome"] in (SELECTED, B) else None
    except ComputeAborted as e:
        return dict(name=name, g=setting["g"], params=params_json(p), judge=None, outcome=COMPUTE_ABORTED, d6=None,
                    state=COMPUTE_ABORTED, note=str(e))
    return dict(name=name, g=setting["g"], params=params_json(p), judge=j, outcome=j["outcome"], d6=d6,
                state=closing_state(j), scan=SCAN_GO)
```

Note: in `stage2`, `measure_d6` is called positionally with the seeds as its 4th argument, which matches `j_runner.measure_d6(jm, ctx, params, seeds=d6a.SEEDS, judged=True)`; the test's stub takes `(jm, ctx, params, seeds)`. If `Deadline` does not proxy plain attributes (`km.odours`), that is why `first_x` is read before wrapping.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_k_runner.py -q -o addopts=""`
Expected: PASS. If `Deadline(m, ctx)` needs more than `ctx.deadline` (read `h3_runner.py:50-63`), extend `JCTX.h3` in the test with the attribute it reads — do not change `Deadline`.

- [ ] **Step 5: Full suite, then commit**

```bash
git add flymon/brain/k_runner.py tests/brain/test_k_runner.py
git commit -m "feat(k): K.8.2/K.8.4 runner — stage 1 re-converges every g by C3's rule, measures ADOPTED engines, ranks and gates; stage 2 judges one engine through J's judge with D.6"
```

---

### Task 7: the two CLIs

**Files:**
- Create: `scripts/run_k_scan.py`, `scripts/run_k_judge.py`
- Test: `tests/test_run_k.py`

**Interfaces:**
- Consumes: everything above; `j_setup.build(npz, spec: JSpec, out, deadline, log, pools, pairs)`; `j_store.load_c3(summary, block, name)`; `h3_store.{ROOT, MeasureCache, canonical, code_key, git_state, sha256_file, write_json, write_bytes}`; `h3_measure.PoolMeasurer`; `h4_measure.H4Measurer`; `j_measure.JMeasurer`; `j_rules.{c3_mismatch, cycle_divergence, SELECTED}`; `FlyPool`.
- Produces: `run_k_scan.main(argv=None, spec=None, summary_spec=SPEC, require_root=True, pools=None, odd=None) -> int` (exit 0 done, 2 refused, 3 aborted, 5 self-check i mismatch); `run_k_judge.main(argv=None, spec=None, summary_spec=SPEC, require_root=True, pools=None, pairs=None) -> int` (0, 2, 3).

- [ ] **Step 1: Write the failing tests**

```python
"""Spec K's CLIs: every refusal comes before any measurement or write (the runs themselves are controller steps)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SCAN, JUDGE = _load("run_k_scan"), _load("run_k_judge")
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])


@pytest.mark.parametrize("cli", [SCAN, JUDGE])
def test_outside_the_repository_root_is_refused(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("cli", [SCAN, JUDGE])
def test_dirty_hashed_files_are_refused(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/k_rules.py"],
                                                             dirty_other=[]))
    assert cli.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_a_foreign_connectome_is_refused(tmp_path, monkeypatch, capsys):
    npz = tmp_path / "other.npz"
    npz.write_bytes(b"not the connectome")
    monkeypatch.setattr(SCAN, "git_state", CLEAN)
    assert SCAN.main(["--npz", str(npz)], require_root=False) == 2 and "declared connectome" in capsys.readouterr().err


def _scan(key, manifest, **kw):
    doc = {"measure_key": key, "code": {"key": manifest}, "smoke": False, "git": {"dirty_hashed": []},
           "spec": json.loads(JUDGE.canonical(JUDGE.SPEC)),
           "stage1": {"outcome": "scan_go", "order": [0], "settings": [
               {"g": -0.2, "reconverge": {"adopted": {"params": {"kc_thresh": 1.65, "kc_thresh_file": "missing.npz",
                                                                  "kc_thresh_sha256": "0" * 64}},
                                          "guard": {}}}]}}
    doc.update(kw)
    return doc


@needs_npz
def test_the_judgement_refuses_other_code_a_stopped_scan_and_a_missing_threshold_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(JUDGE, "git_state", CLEAN)
    key = JUDGE.code_key("data/malecns.npz", files=JUDGE.MEASURE_FILES)["key"]
    manifest = JUDGE.code_key("data/malecns.npz", files=JUDGE.HASHED_FILES)["key"]
    f = tmp_path / "scan.json"
    f.write_text(json.dumps(_scan("0" * 64, manifest)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "K key" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, manifest, stage1={"outcome": "stop_no_target_gain"})))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "K.8.4" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, "0" * 64)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "procedure code" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, manifest, smoke=True)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "smoke" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, manifest)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "threshold file" in capsys.readouterr().err


@needs_npz
def test_the_judgement_reads_the_scan_block_of_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(JUDGE, "git_state", CLEAN)
    manifest = JUDGE.code_key("data/malecns.npz", files=JUDGE.HASHED_FILES)["key"]
    f = tmp_path / "k_engine.json"
    f.write_text(json.dumps({"scan": _scan("0" * 64, manifest)}))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "K key" in capsys.readouterr().err


def test_the_smoke_spec_differs_from_the_declared_one():
    assert SCAN.smoke(SCAN.SPEC) != SCAN.SPEC
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_run_k.py -q -o addopts=""`
Expected: FAIL — `FileNotFoundError` / no `scripts/run_k_scan.py`

- [ ] **Step 3: Write `scripts/run_k_scan.py`**

```python
#!/usr/bin/env python3
"""Spec K.8.2, stage 1: for every g, re-converge C3's rule with fast KC->KC inhibition, measure the X-only MBON13
drive N on the odd-turn (b) pairs for ADOPTED engines, rank and gate (K.8.3-4); or K.8.7's self-check (i).

    uv run python scripts/run_k_scan.py                        # stage 1 (~5-7 h; resumable: rerun the same command)
    uv run python scripts/run_k_scan.py --self-check i         # g = 0 re-converged = C3's recorded thresholds (~1 h)
    uv run python scripts/run_k_scan.py --smoke --allow-dirty  # minutes

Run it from the repository root. Report: <out>/runs/<run id>-scan.{json,md} (self-check: <out>/selfcheck/<id>-i.json).
results/summary/k_engine.json block "scan" is replaced only by a complete stage 1 (no --smoke / --grid / --pairs,
clean hashed files, the declared configuration, not aborted). Exit codes: 0 done (any outcome; a matching self-check),
2 refused before measuring, 3 compute aborted, 5 self-check (i) mismatch (stop and ask the user).
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

from flymon.brain import j_store, k_store
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_runner import Deadline
from flymon.brain.h3_store import ROOT, MeasureCache, code_key, git_state, sha256_file
from flymon.brain.h4_pairs import pairs_digest
from flymon.brain.j_rules import COMPUTE_ABORTED, c3_mismatch, cycle_divergence
from flymon.brain.j_measure import JMeasurer
from flymon.brain.j_runner import params_json, reconverge
from flymon.brain.j_setup import build
from flymon.brain.k_measure import HASHED_FILES, MEASURE_FILES, KMeasurer
from flymon.brain.k_pairs import odd_pairs, overlap_report
from flymon.brain.k_params import k_make
from flymon.brain.k_runner import arrays, stage1
from flymon.brain.k_spec import SPEC, KSpec, smoke

POOL_TIMEOUT_S = 1800


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def main(argv=None, spec: KSpec | None = None, summary_spec: KSpec = SPEC, require_root: bool = True,
         pools: dict | None = None, odd: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="results/summary/k_engine.json")
    ap.add_argument("--m0d", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--grid", type=float, nargs="+", default=None)   # verification only
    ap.add_argument("--pairs", type=int, default=None)               # verification only: the first N odd pairs
    ap.add_argument("--self-check", choices=["i"], default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    if a.grid:
        spec = dataclasses.replace(spec, grid=tuple(a.grid))
    out = Path(a.out or ("results/m0d/k/smoke" if a.smoke else "results/m0d/k/run"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    code, manifest = code_key(a.npz, files=MEASURE_FILES), code_key(a.npz, files=HASHED_FILES)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    try:
        summary = json.loads(Path(a.m0d).read_text())
        c3, _, _ = j_store.load_c3(summary, spec.j.h3_block, spec.j.c3_name)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable C3 in block {spec.j.h3_block} of {a.m0d}: {e}")
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours
    s = build(a.npz, spec.j, out, deadline=deadline, log=lambda x: print(x, flush=True), pools=pools)
    if pools is None and summary[spec.j.h3_block].get("pools") != s["core"]:
        return refuse(f"the core pools {s['core']} differ from block {spec.j.h3_block}'s")
    odd = odd if odd is not None else odd_pairs(s["pops"])
    digest = pairs_digest(odd)
    if not a.smoke and a.pairs is None and digest != spec.odd_pairs_digest:
        return refuse(f"the odd pair list differs from the declared one ({digest[:12]})")
    if a.pairs:
        odd = odd[:a.pairs]
    print(f"run {run_id}: K key {code['key'][:12]}, {len(odd)} odd (b) pairs", flush=True)
    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], spec=spec, self_check=a.self_check, odd_pairs_digest=digest,
               n_odd=len(odd), overlap=overlap_report(odd, {"even_b": [p for p in s["pairs"] if p["axis"] == "b"],
                                                             "even_a": [p for p in s["pairs"] if p["axis"] == "a"]}))
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, run_id)
        m3 = PoolMeasurer(pool, spec.j.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
        km = KMeasurer(pool, spec, odd, len(s["pops"].kc), cache)
        jm = JMeasurer(pool, spec.j, s["all51"], cache)
        seen = [m3, km, jm]
        if a.self_check == "i":
            rc = reconverge(Deadline(m3, s["ctx"].h3), s["ctx"], k_make(spec.j.h4.h3, 0.0))
            want = summary[spec.j.h3_block]["combos"][spec.j.c3_name]
            got = params_json(rc["adopted"]["params"]) if rc["adopted"] else {}
            bad = c3_mismatch(got, want["adopted"]["params"])
            res["check"] = dict(ok=not bad, status=rc["status"], mismatched=bad, got=got,
                                want=want["adopted"]["params"],
                                divergence=None if not bad else cycle_divergence(rc["cells"], want["cells"]))
        else:
            res["stage1"] = stage1(m3, km, s["ctx"], spec, c3, arrays(s["conn"], s["pops"], spec, c3.mv_per_synapse), jm)
    guard_params = [Params()] + list(dict.fromkeys(p for m in seen for p in m.params_seen if p != Params()))
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    if a.self_check:
        rep = k_store.write_json(out / "selfcheck" / f"{run_id}-i.json", res, guard_params)
        print(f"wrote {rep}: self-check (i) {'MATCHES' if res['check']['ok'] else 'DIFFERS'} "
              f"in {res['wall_s'] / 60:.1f} min", flush=True)
        return 0 if res["check"]["ok"] else 5
    st1 = res["stage1"]
    report = k_store.write_json(out / "runs" / f"{run_id}-scan.json", res, guard_params)
    lines = [f"# K.8.2 stage 1 {run_id}\n", f"**outcome: {st1['outcome']}**\n",
             f"C3 N: {st1['base']['metrics']['N'] if st1['base'] else None}\n"]
    for i, t in enumerate(st1["settings"]):
        n = t["metrics"]["N"] if t["metrics"] else None
        lines.append(f"- g {t['g']:g}: {t['status']}, N {n}, ratio {st1['ratios'].get(i)}\n")
    lines.append(f"\norder: {[st1['settings'][i]['g'] for i in st1['order']]}\n")
    k_store.write_bytes(out / "runs" / f"{run_id}-scan.md", "".join(lines).encode(), guard_params)
    print(f"wrote {report}: {st1['outcome']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, grid=a.grid is not None, pairs=a.pairs is not None,
                    dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec, aborted=st1["outcome"] == COMPUTE_ABORTED)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        k_store.write_summary_block(a.summary, "scan", dict(res, report=str(report), report_sha256=sha256_file(report)),
                                    guard_params)
        print(f"wrote {a.summary} (block scan)", flush=True)
    return 3 if st1["outcome"] == COMPUTE_ABORTED else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/run_k_judge.py`**

```python
#!/usr/bin/env python3
"""Spec K.8.4, stage 2: judge the top setting of a complete stage 1 whose outcome is scan_go — H.4 readout
reselection, z, the oracle on the 21 even-turn (b) and 18 (a) pairs, J.12.9's bands, D.6 on the judged engine.

    uv run python scripts/run_k_judge.py                       # ~1.5 h
    uv run python scripts/run_k_judge.py --smoke --allow-dirty --scan <a smoke scan report>

Run it from the repository root. --scan is results/summary/k_engine.json (its block "scan") or a scan report. Refused
before measuring: a scan under another K key, manifest (HASHED_FILES) or configuration; a smoke or dirty scan outside
--smoke; a scan whose outcome is not scan_go (K.8.4: no judgement); a chosen engine whose threshold file is missing or
has another sha256. Report: <out>/runs/<run id>-judge.{json,md}; block "judge" of the summary is replaced only by a
complete judgement (every pair, D.6 measured, clean, declared configuration, not aborted). Exit codes: 0, 2, 3.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path

from flymon.brain import d6a, k_store
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_store import ROOT, MeasureCache, canonical, code_key, git_state, sha256_file
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import pair_key
from flymon.brain.j_measure import JMeasurer
from flymon.brain.j_params import params_from_json
from flymon.brain.j_rules import COMPUTE_ABORTED, SELECTED
from flymon.brain.j_setup import build
from flymon.brain.k_measure import HASHED_FILES, MEASURE_FILES
from flymon.brain.k_rules import SCAN_GO
from flymon.brain.k_runner import stage2
from flymon.brain.k_spec import SPEC, KSpec, smoke

POOL_TIMEOUT_S = 1800


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def main(argv=None, spec: KSpec | None = None, summary_spec: KSpec = SPEC, require_root: bool = True,
         pools: dict | None = None, pairs: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scan", default="results/summary/k_engine.json")
    ap.add_argument("--summary", default="results/summary/k_engine.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)       # verification only
    ap.add_argument("--no-d6", action="store_true")          # verification only
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.pairs is not None and a.pairs < 1:
        return refuse(f"--pairs must be at least 1, got {a.pairs}")
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/k/smoke" if a.smoke else "results/m0d/k/run"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    code, manifest = code_key(a.npz, files=MEASURE_FILES), code_key(a.npz, files=HASHED_FILES)
    try:
        doc = json.loads(Path(a.scan).read_text())
    except (OSError, ValueError) as e:
        return refuse(f"no usable scan at {a.scan}: {e}")
    scan = doc["scan"] if "scan" in doc and "stage1" not in doc else doc
    if scan.get("measure_key") != code["key"]:
        return refuse(f"the scan was measured under K key {str(scan.get('measure_key'))[:12]}, this code's is "
                      f"{code['key'][:12]}: its engines are not this code's")
    if not a.smoke and scan.get("code", {}).get("key") != manifest["key"]:
        return refuse(f"the scan was run under another procedure code {str(scan.get('code', {}).get('key'))[:12]}, "
                      f"this code's is {manifest['key'][:12]}")
    if not a.smoke and (scan.get("smoke") or scan.get("git", {}).get("dirty_hashed")):
        why = "a smoke run" if scan.get("smoke") else f"measured with dirty hashed files {scan['git']['dirty_hashed']}"
        return refuse(f"the scan is {why}: only a --smoke judgement may use it")
    if not a.smoke and canonical(scan.get("spec")) != canonical(spec):
        return refuse("the scan was run under another K configuration (its spec differs from this judgement's)")
    st1 = scan.get("stage1", {})
    if st1.get("outcome") != SCAN_GO:
        return refuse(f"the scan's outcome is {st1.get('outcome')}: K.8.4 judges only after {SCAN_GO}")
    setting = st1["settings"][st1["order"][0]]
    p = params_from_json(setting["reconverge"]["adopted"]["params"])
    tf = Path(p.kc_thresh_file)
    if not tf.exists() or sha256_file(tf) != p.kc_thresh_sha256:
        return refuse(f"the chosen engine's threshold file {tf} is missing or has another sha256")
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours
    s = build(a.npz, spec.j, out, deadline=deadline, log=lambda x: print(x, flush=True), pools=pools, pairs=pairs)
    if pairs is None and s["pairs_digest"] != spec.j.h4.pairs_digest:
        return refuse(f"the pair list differs from the declared one ({s['pairs_digest'][:12]})")
    if a.pairs:
        s["pairs"] = [q for ax in ("a", "b") for q in [r for r in s["pairs"] if r["axis"] == ax][:a.pairs]]
        s["ctx"].expected = [pair_key(q) for q in s["pairs"]]
    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], spec=spec, scan_run_id=scan.get("run_id"),
               n_pairs=len(s["pairs"]), setting_g=setting["g"], even_pair_uses=list(spec.even_pair_uses) + ["K.8"])
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, run_id)
        m3 = PoolMeasurer(pool, spec.j.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
        m4 = H4Measurer(pool, spec.j.h4, s["pairs"], s["ctx"].pools, cache, m3)
        jm = JMeasurer(pool, spec.j, s["all51"], cache)
        seen = [m3, m4, jm]
        res["stage2"] = stage2(m4, jm, s["ctx"], setting, with_d6=not a.no_d6,
                               d6_seeds=d6a.SEEDS[:2] if a.smoke else d6a.SEEDS)
    guard_params = [Params()] + list(dict.fromkeys(q for m in seen for q in m.params_seen if q != Params()))
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {q: sha256_file(q) for q in sorted(cache.used) if Path(q).exists()}
    st2 = res["stage2"]
    report = k_store.write_json(out / "runs" / f"{run_id}-judge.json", res, guard_params)
    k_store.write_bytes(out / "runs" / f"{run_id}-judge.md",
                        (f"# K.8.4 judgement {run_id}\n\n**state: {st2['state']}** ({st2['name']})\n\n"
                         f"reading: {(st2.get('judge') or {}).get('reading')}\n\nD.6: {st2.get('d6')}\n").encode(),
                        guard_params)
    print(f"wrote {report}: {st2['state']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, pairs=a.pairs is not None, no_d6=a.no_d6, dirty=bool(git["dirty_hashed"]),
                    spec=spec != summary_spec, aborted=st2["state"] == COMPUTE_ABORTED)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        block = dict(res, report=str(report), report_sha256=sha256_file(report))
        if st2["state"] == SELECTED:
            rd = st2["judge"]["reading"]
            block["selection"] = dict(setting=st2["name"], params=st2["params"], T_b=rd["T_b"], F_a=rd["F_a"],
                                      testable_b=rd["testable_b"], n_b=rd["n_b"], confirmed=False)
        k_store.write_summary_block(a.summary, "judge", block, guard_params)
        print(f"wrote {a.summary} (block judge)", flush=True)
    return 3 if st2["state"] == COMPUTE_ABORTED else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_run_k.py -q -o addopts=""`
Expected: PASS. In the judge, the order of checks matters for the tests: key → manifest → smoke/dirty → spec → outcome → threshold file. The `_scan` fixture's spec is the canonical `SPEC`, so a real-data run reaches the later checks.

- [ ] **Step 6: Full suite, then commit**

```bash
git add scripts/run_k_scan.py scripts/run_k_judge.py tests/test_run_k.py
git commit -m "feat(k): K.8 CLIs — stage 1 scan with self-check (i) and odd-pair digest/overlap record; stage 2 judgement refusing other keys, stopped scans and changed threshold files; summary blocks through k_store"
```

---

## Run order (controller only, after Task 7; each long step in the background, never from a subagent)

- **R0 smoke:** `uv run python scripts/run_k_scan.py --smoke --allow-dirty` then, if its outcome is `scan_go`, `uv run python scripts/run_k_judge.py --smoke --allow-dirty --scan <the smoke scan report>`. Pass = both exit 0 with reports under `results/m0d/k/smoke/`. If the smoke scan reads `stop_*`, that is a valid smoke result: run the judge smoke with a hand-edited copy of the report whose outcome is `scan_go` only to exercise the path (never under `results/m0d/k/run`).
- **R1 self-check (i) and timing:** `uv run python scripts/run_k_scan.py --self-check i` → `MATCHES` (exit 0). Exit 5 = stop and ask the user. Its wall time is the per-setting re-convergence cost; if 5 × that exceeds 10 h, report to the user before R2.
- **R1b metric self-check (ii):** `uv run pytest tests/brain/test_k_metrics.py -q -o addopts="" -rs` → the H.4a.8 test passes (not skipped).
- **R2 stage 1:** `uv run python scripts/run_k_scan.py` → block `scan` in `results/summary/k_engine.json`. `COMPUTE_ABORTED` → rerun the same command (cache resumes).
- **R3 stage 2:** only if R2's outcome is `scan_go`: `uv run python scripts/run_k_judge.py` → block `judge`.
- **R4 record:** K results section (dated, K.8.5's sentence for the state reached, with setting, N / N_C3, S change, readout type, naive_a and J.12.9's matching sensitivity row) + README ledger (ko/en) + push. Spec-reserved verdicts (SELECTED → confirmation, B_NO_CONCLUSION, STOP_MULTI_TYPE, any N_C3 ≤ 0 stop) go to the user.
