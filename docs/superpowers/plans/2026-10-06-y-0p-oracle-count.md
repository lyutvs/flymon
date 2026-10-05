# Spec Y, step 0p: main-set oracle count (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build only Y.7 **0p** (runs in parallel with Y's red-team). Two stages are built: `digest` (0p-b: partial reuse and the main-set digest) and `oracle` (0p-c: the L_V oracle on all 249 main-set pairs with seeds 24_700_xxx). The tracked summary holds **only a count table**. Per-pair values go only to the git-ignored `results/y/oracle.json`. If the lenient pre-filter passes fewer than 4 pairs, the stage ends with the early `STOP_FEW_PAIRS`.

**Architecture:**
- New files: `flymon/brain/y_spec.py` (every Y number used so far), `y_rules.py` (0p filters, count table, decisions, Y.8 sentences), `y_store.py` (the writer guard and `YCache`), `y_runner.py` (minimal: `ORDER = ("digest", "oracle")`), `scripts/run_y.py`, and `tests/brain/test_y_*.py`.
- One existing file changes: `tests/brain/test_p_spec.py` gains a `MODULES` entry so the seed collectors see Y's blocks.
- Reused code is imported and never modified:
  - `w_runner.build_ctx`, which provides `w_set`, `main_rows`, `params`, `readout`, `types`, `n_kc` and the V reuse context.
  - `w_runner.Runner(...)._reuse_dec()`, which checks the V blocks, the U key and the shared and T keys, the same way X uses `_pilot_back`.
  - `w_pairs.set_summary`.
  - `r_measure.RMeasurer.oracle(rows, V_SPEC.cond("L"), "screen", W_SPEC.oracle_seeds())`.
  - `h4_formula.pair_stats`.
  - `u_measure.UPool` and `fly_pool.FlyPool`.
- Phase A later appends to these files. It extends `ORDER` after `oracle`, adds fields to `y_spec`, and adds gates and sentences to `y_rules`. It never changes a 0p computation. If one must change, the 0p blocks are recomputed with the same seeds and must be bit-identical (Y.7 0p-a).

**Tech Stack:** Python 3.13, NumPy, pytest. Always use `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md`, appendix **Y** (lines 5182–5477, commit `301aa29`). This plan mainly uses Y.7 0p, Y.3.1–Y.3.3, Y.2 (reuse, seeds, code boundary), Y.8 (`STOP_REUSE`, early `STOP_FEW_PAIRS`) and Y.9.1, author's reading 6.

## Global Constraints

- `flymon/brain/w_*.py`, `x_*.py`, `scripts/run_w.py` and `scripts/run_x.py` are never edited. `results/w/`, `results/x/`, `results/v/` and every `results/summary/*` other than `y_learning.json` are read only.
- Y writes only under `results/y/` (git-ignored) and to `results/summary/y_learning.json` (tracked). Every other path gets SystemExit 2.
- Numbers live only in `y_spec.py`. Any other Y file holding an int literal with |v| > 16, or a float literal other than 0.0 or 1.0, fails `test_y_spec`. The oracle seeds 24_700_xxx are never restated in Y code; they are read from `w_spec.SPEC.oracle_seeds()`.
- Thresholds (Y.3.2 / Y.3.3 / Y.7 0p-c): balance |d_pre| < 0.5 (strict); floors c_A 20 and c_P 43 (≥); lenient pre-filter |d_pre| < 1.0 ∧ L_A^or ≥ 16 ∧ L_P^or ≥ 34.4. Comparisons round to 9 digits. "Testable" means min(r, −p) ≥ 2 (`V_SPEC.testable_min`).
- Main set: `w_pairs.w_set` regenerated. **`digest_keys` = `65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a`**, (b) 167 · (a) 82, last turn 1967. Any mismatch gives `STOP_REUSE`.
- Keys: U measurement key `8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523`, W measurement key `761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1`, V blocks z `928eaad` · kc_input `7dc199d` · set `cf0b3b2` · judge `a279a56` (SELECTED), checked through `w_rules.reuse`.
- Y seed roots (Y.2, author's reading 2): 60_000_000 · 61_000_000 · 62_000_000 · 64_000_000 · 77_000_000 · 77_100_000 · 77_110_000 · 77_150_000 · 77_200_000 · 77_300_000. No other declared seed may fall in [60_000_000, 77_400_000).
- Read rule (Y.0, Y.7 0p-c): until Y.9.2 is committed, the controller and the red-team read **only the count table**. The block, the CLI output and commit messages never carry a per-pair d_pre, r, p or level value.
- Commits: no `Co-Authored-By`, `Claude-Session` or "Generated with" trailers.
- Tests: `.venv/bin/python -m pytest <paths> -q > /private/tmp/claude-503/y0p-<task>.log 2>&1; tail -5 /private/tmp/claude-503/y0p-<task>.log`.

## Readings (decisions this plan makes; record them in the module docstrings)

1. **The digest field is `digest_keys`.** I regenerated it read-only on 2026-10-06 (`w_runner.build_ctx(W_SPEC, NPZ)["w_set"]()`, 1.9 s):
   - `digest_keys` = `65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a`
   - `digest_e0_b` = `29773323…`, `digest_e0_a` = `d599c04b…`
   - n_b 167, n_a 82, n_odours 155, first turn 306, last turn 1967
   - The spec's `65dbf001…` is therefore `w_pairs.w_set(...)["digest_keys"]`, the sha256 of the declared-order key list. W never wrote a `set` block (it stopped at `oc`), so the `digest` block is the first committed record of the set. It stores `set_summary(js)` plus `keys`, and `oracle` regenerates the rows against it via `main_rows`.
2. **F2 columns use the exact X values.** `results/x/precheck_diag.json` (sha256 `33f83895…`) gives `filters.F2(0).pass_counts` c_A 8.875 / c_P 51.640625 and `F2(25)` 30.7421875 / 94.421875. The spec prints these as 8.9 · 51.6 / 30.7 · 94.4. Oracle levels are medians of 8 integer counts, which are half-integers, so both readings give the same counts.
3. **Oracle levels:** L_A^or = median over the 8 report seeds of `report["pre"]["A"][:, 0]`, and L_P^or uses `["P"][:, 0]` the same way. Column 0 is odour X. `ap()` in `q_jobs` holds the readout type's cell-sum counts (Y.3.1).
4. **Failures:**
   - A lever mismatch (edit, CSC sha or edge count ≠ `V_SPEC`) is counted under `failures` and makes the block `INVALID`. It is a machine defect: exit 5, do not commit (as W did).
   - `pair_stats` = None (an undefined d′) is counted under `no_value` and treated as not testable. It does not change the outcome.
5. **Caches:**
   - The cache is `results/y/cache` under the W measurement key, which is unchanged because `w_measure.py` is untouched.
   - V's oracle cache is not chained in 0p. The main set (turns 306–1985) is disjoint from V's set by construction (W.2), so the V cache can have no hits there. V-cache reuse belongs to phase A's pilot rows.
6. **Ledger:** `y_learning.json["budget"]["ledger"]` is a list of `{stage, wall_s, at}` (Y.7: "원장은 `budget` 블록"). The oracle stage's wall time accumulates across resumed runs through `results/y/progress/oracle.json`.

## Review Focus

1. **Per-pair values leaking to the controller.** The `oracle` block, the CLI stdout and the ledger must hold only counts, the detail sha and seeds. Task 2 asserts that no `d_pre`, `L_A` or `pairs` key appears in the block and that stdout contains no per-pair key.
2. **A 3 h run killed midway.** No block may be written, and a rerun must neither refuse nor duplicate work. Task 2 has a test where the fake measurer raises; it then checks that no `oracle` block was written and that the progress wall time was kept.
3. **Boundary values.** Half-integer medians and 9-digit rounding must behave exactly at the edges: |d| = 0.5 is not balanced, d = −1.0 fails the one-sided test, L_A 20.0 passes strict, 19.5 fails, 34.4 passes lenient, and 0.9999999999 rounds to 1.0 and fails lenient. Task 1 tests each.
4. **Infinite or undefined d′ from `pair_stats`.** ±inf must never count as balanced, and None goes to `no_value`. Task 1 covers this.
5. **The main set changing after `digest` was committed, or a dirty hashed W, V or Y file.** `oracle` must refuse (exit 2) and write nothing. Task 2 covers this.

---

### Task 1: `y_spec`, `y_rules`, `y_store` (numbers, filters, count table, sentences, writer)

**Files:**
- Create: `flymon/brain/y_spec.py`, `flymon/brain/y_rules.py`, `flymon/brain/y_store.py`
- Modify: `tests/brain/test_p_spec.py:21-30` (add `"flymon/brain/y_spec.py": "flymon.brain.y_spec"` to `MODULES`)
- Test: `tests/brain/test_y_spec.py`, `tests/brain/test_y_rules.py`, `tests/brain/test_y_store.py`

**Interfaces:**
- Produces:
  - `y_spec.SPEC: YSpec`
  - `y_rules.PASS`, `INVALID`, `STOP_REUSE`, `STOP_FEW_PAIRS`, and `COLUMNS: tuple[str, ...]`
  - `y_rules.reuse_stop(why: list, where: str) -> dict`
  - `y_rules.key_reasons(keys: dict, ys) -> list`
  - `y_rules.digest_reasons(js: dict, ys) -> list`
  - `y_rules.levels(report: dict) -> tuple[float, float]`
  - `y_rules.passes(p: dict, ys) -> dict[str, bool]`
  - `y_rules.count_table(per: list, ys) -> dict`
  - `y_rules.oracle_decision(table: dict, ys) -> dict`
  - `y_store.write_json`, `read_summary`, `write_summary_block(path, block, obj, params_list, ledger=None)`, `to_json`, `SUMMARY`, and `YCache(root, code)`

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_y_spec.py`:
```python
"""Y's 0p numbers (Y.2 · Y.3.2 · Y.3.3 · Y.7 0p): thresholds, keys, the main-set facts, seed roots that collide with
nothing declared, and no Y file but y_spec holding a number (literal guard)."""
import ast
import dataclasses
import importlib
import json
from pathlib import Path

import pytest

from flymon.brain import d6a
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC, YSpec
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]


def test_y_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/y_spec.py") == "flymon.brain.y_spec"


def test_thresholds():
    assert (SPEC.naive_max, SPEC.c_a, SPEC.c_p) == (0.5, 20.0, 43.0)
    assert (SPEC.lenient_d, SPEC.lenient_a, SPEC.lenient_p) == (1.0, 16.0, 34.4)
    assert round(0.8 * SPEC.c_a, 9) == SPEC.lenient_a and round(0.8 * SPEC.c_p, 9) == SPEC.lenient_p
    assert SPEC.round_digits == 9 and V.testable_min == 2.0
    assert SPEC.f2_0 == (8.875, 51.640625) and SPEC.f2_25 == (30.7421875, 94.421875)
    assert (SPEC.user_a, SPEC.user_onesided_d) == (31.0, -1.0)
    assert SPEC.k_ranges == ((4, 8), (6, 10)) and min(lo for lo, _ in SPEC.k_ranges) == 4
    assert SPEC.budget_h == 24.0


def test_f2_values_are_xs_precheck_diag():
    p = ROOT / "results/x/precheck_diag.json"
    if not p.exists():
        pytest.skip("X detail not present in this checkout")
    f = json.loads(p.read_text())["filters"]
    assert SPEC.f2_0 == (f["F2(0)"]["pass_counts"]["c_A"], f["F2(0)"]["pass_counts"]["c_P"])
    assert SPEC.f2_25 == (f["F2(25)"]["pass_counts"]["c_A"], f["F2(25)"]["pass_counts"]["c_P"])


def test_keys_and_main_set_facts():
    assert SPEC.u_measure_key == W.u_measure_key_u == \
        "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    assert SPEC.w_measure_key == "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    assert SPEC.main_digest_keys == "65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a"
    assert (SPEC.main_n_b, SPEC.main_n_a, SPEC.main_last_turn) == (167, 82, 1967)
    assert (SPEC.summary, SPEC.raw_dir, SPEC.cache_dir, SPEC.oracle_detail) == (
        "results/summary/y_learning.json", "results/y", "results/y/cache", "results/y/oracle.json")


def test_seed_roots_are_new():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {60_000_000, 61_000_000, 62_000_000, 64_000_000, 77_000_000, 77_100_000, 77_110_000,
                   77_150_000, 77_200_000, 77_300_000}
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(YSpec))
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/y_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    assert not any(60_000_000 <= s < 77_400_000 for s in declared)
    o = W.oracle_seeds()
    assert not any(60_000_000 <= s < 77_400_000 for s in o["act"] + o["select"] + o["report"])


def _numbers(path: Path) -> set:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def test_no_y_file_but_y_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("y_*.py")) if p.name != "y_spec.py"]
    if (ROOT / "scripts/run_y.py").exists():
        files.append(ROOT / "scripts/run_y.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
```

`tests/brain/test_y_rules.py`:
```python
"""0p's filters and count table (Y.3.2 · Y.3.3 · Y.7 0p-c), the digest / key checks and Y.8's sentences verbatim."""
import math

import numpy as np

from flymon.brain import y_rules as R
from flymon.brain.y_spec import SPEC as Y


def _p(d, a, q, testable=True, axis="b", value=True, failure=False):
    return dict(axis=axis, value=value, failure=failure, testable=testable, d_pre=d, L_A=a, L_P=q)


def test_columns():
    assert R.COLUMNS == ("testable", "balanced", "y_strict", "y_lenient", "f2_0", "f2_25", "user31",
                         "user_onesided31")


def test_boundaries():
    assert R.passes(_p(0.5, 50, 200), Y)["balanced"] is False
    assert R.passes(_p(-0.4999, 50, 200), Y)["balanced"] is True
    s = R.passes(_p(0.1, 20.0, 43.0), Y)
    assert s["y_strict"] and s["y_lenient"]
    assert not R.passes(_p(0.1, 19.5, 43.0), Y)["y_strict"]
    assert not R.passes(_p(0.1, 20.0, 42.5), Y)["y_strict"]
    assert R.passes(_p(0.9, 16.0, 34.4), Y)["y_lenient"]
    assert not R.passes(_p(0.9999999999, 30, 100), Y)["y_lenient"]        # rounds to 1.0
    assert not R.passes(_p(0.9, 15.5, 100), Y)["y_lenient"]
    assert R.passes(_p(-0.99, 31.0, 0.0), Y)["user_onesided31"]
    assert not R.passes(_p(-1.0, 31.0, 0.0), Y)["user_onesided31"]
    assert R.passes(_p(5.0, 31.0, 0.0), Y)["user_onesided31"] and not R.passes(_p(5.0, 31.0, 0.0), Y)["user31"]
    assert R.passes(_p(0.0, 9.0, 52.0), Y)["f2_0"] and not R.passes(_p(0.0, 8.5, 52.0), Y)["f2_0"]
    assert R.passes(_p(0.0, 31.0, 94.5), Y)["f2_25"] and not R.passes(_p(0.0, 31.0, 94.0), Y)["f2_25"]


def test_not_testable_infinite_and_missing_count_nothing():
    assert not any(R.passes(_p(0.0, 50, 200, testable=False), Y).values())
    assert not any(R.passes(_p(None, None, None, testable=False, value=False), Y).values())
    s = R.passes(_p(math.inf, 50, 200), Y)
    assert s["testable"] and not s["balanced"] and not s["y_lenient"]
    assert not R.passes(_p(-math.inf, 50, 200), Y)["balanced"]


def test_count_table_rows_and_columns():
    per = [_p(0.1, 25, 50, axis="b"), _p(0.7, 17, 35, axis="a"), _p(0.1, 25, 50, testable=False, axis="a"),
           _p(None, None, None, testable=False, value=False, axis="b"), _p(0.0, 40, 100, axis="b", failure=True)]
    t = R.count_table(per, Y)
    assert set(t) == {"all", "b", "a"}
    assert t["all"]["n"] == 5 and t["b"]["n"] == 3 and t["a"]["n"] == 2
    assert t["all"]["no_value"] == 1 and t["all"]["failures"] == 1
    assert t["all"]["testable"] == 3 and t["all"]["balanced"] == 2 and t["all"]["y_strict"] == 2
    assert t["all"]["y_lenient"] == 3 and t["a"]["y_lenient"] == 1
    assert all(t["all"][c] == t["b"][c] + t["a"][c] for c in R.COLUMNS + ("n", "no_value", "failures"))


def test_oracle_decision():
    def tab(lenient, failures=0):
        return {"all": dict({c: 0 for c in R.COLUMNS}, n=249, no_value=0, failures=failures, y_lenient=lenient)}
    assert R.oracle_decision(tab(4), Y)["outcome"] == R.PASS
    d = R.oracle_decision(tab(3), Y)
    assert d["outcome"] == R.STOP_FEW_PAIRS and d["stage"] == "early" and d["records_unavailable"] is True
    assert d["sentence"] == ("Y 주 세트 249쌍에서 오라클 사전 거름(시험 가능 ∧ |d_pre| < 1.0 ∧ 순진 MBON13(X) ≥ 16 ∧ "
                             "순진 MBON05(X) ≥ 34.4)을 통과한 쌍이 3개로 최소 관문 쌍 수 4에 못 미쳤다 — 주 세트 "
                             "학습 측정 없이 멈춘다.")
    assert R.oracle_decision(tab(0, failures=1), Y)["outcome"] == R.INVALID


def test_reuse_stop_sentence():
    d = R.reuse_stop(["W 측정 키 x ≠ y"], "0p-b")
    assert d["outcome"] == R.STOP_REUSE and d["records_unavailable"] is True
    assert d["sentence"] == ("Y 재사용 조건(Y.7 0p-b)이 깨졌다(W 측정 키 x ≠ y). Y는 W 파일럿 · 경로, X 블록, V 블록, "
                             "주 세트 생성원을 다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 없이 멈춘다 "
                             "— 사용자 몫.")


def test_key_and_digest_reasons():
    keys = dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key)
    assert R.key_reasons(keys, Y) == []
    assert len(R.key_reasons(dict(keys, w_measure_key="x"), Y)) == 1
    assert len(R.key_reasons(dict(keys, u_measure_key="x"), Y)) == 1
    js = dict(digest_keys=Y.main_digest_keys, n_b=167, n_a=82, last_turn=1967)
    assert R.digest_reasons(js, Y) == []
    for k, v in (("digest_keys", "0" * 64), ("n_b", 166), ("n_a", 83), ("last_turn", 1985)):
        assert len(R.digest_reasons(dict(js, **{k: v}), Y)) == 1, k


def test_levels_are_medians_of_odour_x():
    rep = {"pre": {"A": [[20, 99], [21, 0], [19, 0], [22, 0], [18, 0], [20, 0], [21, 0], [23, 0]],
                   "P": [[40, 0], [44, 0], [43, 0], [42, 0], [45, 0], [41, 0], [43, 0], [46, 0]]}}
    assert R.levels(rep) == (float(np.median([20, 21, 19, 22, 18, 20, 21, 23])),
                             float(np.median([40, 44, 43, 42, 45, 41, 43, 46])))
```

`tests/brain/test_y_store.py`:
```python
"""Y's writer: results/y/ and results/summary/y_learning.json only, atomic, the ledger under budget; YCache's put goes
through it."""
import json

import pytest

from flymon.brain import y_store as YS
from flymon.brain.config import Params


def test_guard_and_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    YS.write_json("results/y/a.json", {"a": 1}, [Params()])
    YS.write_summary_block(YS.SUMMARY, "digest", {"b": 2}, [Params()], ledger=dict(stage="digest", wall_s=1.0))
    YS.write_summary_block(YS.SUMMARY, "oracle", {"c": 3}, [Params()], ledger=dict(stage="oracle", wall_s=2.0))
    d = json.loads((tmp_path / YS.SUMMARY).read_text())
    assert d["digest"] == {"b": 2} and [e["stage"] for e in d["budget"]["ledger"]] == ["digest", "oracle"]
    for bad in ("results/w/x.json", "results/x/x.json", "results/v/cache/x.json", "results/summary/x_learning.json",
                "results/yy/a.json", "y.json"):
        with pytest.raises(SystemExit) as e:
            YS.write_json(bad, {}, [Params()])
        assert e.value.code == 2
    assert not list((tmp_path / "results/y").glob(".*.tmp"))


def test_ycache_round_trip_and_refuses_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = YS.YCache("results/y/cache", {"key": "k" * 64})
    ins = dict(pair="b|1|x|y", report_seeds=[24_700_200])
    assert c.get("r_oracle", ins) is None
    c.put("r_oracle", ins, {"v": 1}, [Params()])
    assert c.get("r_oracle", ins) == {"v": 1}
    with pytest.raises(SystemExit):
        YS.YCache("results/w/cache", {"key": "k" * 64}).put("r_oracle", ins, {"v": 1}, [Params()])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/brain/test_y_spec.py tests/brain/test_y_rules.py tests/brain/test_y_store.py -q > /private/tmp/claude-503/y0p-t1.log 2>&1; tail -5 /private/tmp/claude-503/y0p-t1.log`
Expected: errors with `ModuleNotFoundError: No module named 'flymon.brain.y_spec'`.

- [ ] **Step 3: Implement**

`flymon/brain/y_spec.py`:
```python
"""Every number of spec appendix Y used so far — step 0p only (Y.7 0p-a). Phase A appends its numbers here and changes
none of these (a change means recomputing the 0p blocks with the same seeds, bit for bit).
- Thresholds: Y.3.2 (balance < 0.5, c_A 20, c_P 43, 9-digit rounding), Y.3.3 1 (lenient 1.0 · 16 · 34.4), the record-
  only columns of Y.7 0p-c: F2(0) / F2(25) exactly as X's results/x/precheck_diag.json filters (spec prints 8.9 · 51.6 /
  30.7 · 94.4), the user's 31 and one-sided −1. Testable = V_SPEC.testable_min (2.0), not restated.
- Main set (Y.2): w_pairs.w_set's digest_keys (Reading 1 of the 0p plan), counts and last turn.
- Seeds (Y.2, author's reading 2): Y's roots only; the oracle seeds 24_700_xxx are W's (w_spec.SPEC.oracle_seeds()) and
  never restated here (the seed collectors would see a collision with W).
- Paths: results/y/ (git-ignored), results/summary/y_learning.json (tracked), ~/flymon-archive/y."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YSpec:
    # ---- the filter (Y.3.2) and the oracle lenient pre-filter (Y.3.3 1) ---------------------------------------------
    naive_max: float = 0.5
    c_a: float = 20.0
    c_p: float = 43.0
    lenient_d: float = 1.0
    lenient_a: float = 16.0                         # 0.8 c_A
    lenient_p: float = 34.4                         # 0.8 c_P
    round_digits: int = 9
    # ---- record-only count-table columns (Y.7 0p-c) ------------------------------------------------------------------
    f2_0: tuple = (8.875, 51.640625)
    f2_25: tuple = (30.7421875, 94.421875)
    user_a: float = 31.0
    user_onesided_d: float = -1.0
    # ---- the set rule's k ranges (Y.5); 0p uses min k_lo for the early STOP_FEW_PAIRS -------------------------------
    k_ranges: tuple = ((4, 8), (6, 10))
    # ---- reused keys and the main set (Y.2, Y.7 0p-b) -----------------------------------------------------------------
    u_measure_key: str = "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    w_measure_key: str = "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    main_digest_keys: str = "65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a"
    main_n_b: int = 167
    main_n_a: int = 82
    main_last_turn: int = 1967
    # ---- seeds (Y.2) --------------------------------------------------------------------------------------------------
    pilot_probe_seed0: int = 60_000_000             # + j × 4_000 + f × 100 + k
    pilot_train_seed0: int = 61_000_000             # + j × 40_000 + f × 1_000 + t
    probe_seed0: int = 62_000_000                   # + c × 4_000 + f × 100 + k (f, k < 32)
    train_seed0: int = 64_000_000                   # + c × 40_000 + f × 1_000 + t
    oc_seed: int = 77_000_000
    smoke_probe_seed0: int = 77_100_000
    smoke_train_seed0: int = 77_110_000
    smoke_oracle_seed0: int = 77_150_000
    precheck_seed: int = 77_200_000
    compare_seed: int = 77_300_000
    # ---- budget, pool, paths, CLI -------------------------------------------------------------------------------------
    budget_h: float = 24.0
    workers: int = 16
    pool_timeout_s: float = 3600.0
    summary: str = "results/summary/y_learning.json"
    raw_dir: str = "results/y"
    cache_dir: str = "results/y/cache"
    oracle_detail: str = "results/y/oracle.json"
    progress_dir: str = "results/y/progress"
    archive_root: str = "~/flymon-archive/y"
    cli_print_chars: int = 2000


SPEC = YSpec()
```

`flymon/brain/y_rules.py`:
```python
"""Y's 0p decisions and sentences (Y.3.1–Y.3.3, Y.7 0p-b / 0p-c, Y.8): key and digest reasons, STOP_REUSE, the oracle
levels, the filters as count-table columns, the count table (rows all / (b) / (a)) and the early STOP_FEW_PAIRS. Every
number comes from the YSpec passed in. Phase A adds its gates and sentences here and changes none of these.
- Every column requires testable. Balance is strict (<), floors include the bound (≥), all after rounding to
  round_digits. A pair with no oracle value (pair_stats None) counts in no_value and nowhere else; a lever mismatch
  counts in failures and makes the block INVALID (0p plan Reading 4)."""
from __future__ import annotations

import numpy as np

PASS, INVALID = "PASS", "INVALID"
STOP_REUSE, STOP_FEW_PAIRS = "STOP_REUSE", "STOP_FEW_PAIRS"
COLUMNS = ("testable", "balanced", "y_strict", "y_lenient", "f2_0", "f2_25", "user31", "user_onesided31")
ROWS = ("all", "b", "a")

SENTENCES = {
    STOP_REUSE: "Y 재사용 조건(Y.7 {where})이 깨졌다({why}). Y는 W 파일럿 · 경로, X 블록, V 블록, 주 세트 생성원을 "
                "다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 없이 멈춘다 — 사용자 몫.",
    (STOP_FEW_PAIRS, "early"): "Y 주 세트 {n}쌍에서 오라클 사전 거름(시험 가능 ∧ |d_pre| < 1.0 ∧ 순진 MBON13(X) ≥ 16 ∧ "
                               "순진 MBON05(X) ≥ 34.4)을 통과한 쌍이 {k}개로 최소 관문 쌍 수 4에 못 미쳤다 — 주 세트 "
                               "학습 측정 없이 멈춘다.",
}
RECORDS_REASON = "OC 전 관문 STOP(X.9.1.3 P2-10)"


def reuse_stop(why: list, where: str) -> dict:
    return dict(outcome=STOP_REUSE, reasons=list(why),
                sentence=SENTENCES[STOP_REUSE].format(where=where, why="; ".join(why)),
                records_unavailable=True, records_reason=RECORDS_REASON)


def key_reasons(keys: dict, ys) -> list:
    why = []
    if keys.get("w_measure_key") != ys.w_measure_key:
        why.append(f"W 측정 키 {keys.get('w_measure_key')} ≠ {ys.w_measure_key}")
    if keys.get("u_measure_key") != ys.u_measure_key:
        why.append(f"U 측정 키 {keys.get('u_measure_key')} ≠ {ys.u_measure_key}")
    return why


def digest_reasons(js: dict, ys) -> list:
    want = dict(digest_keys=ys.main_digest_keys, n_b=ys.main_n_b, n_a=ys.main_n_a, last_turn=ys.main_last_turn)
    return [f"주 세트 {k}: 재생성 {js.get(k)!r} ≠ 선언 {v!r}" for k, v in want.items() if js.get(k) != v]


def levels(report: dict) -> tuple:
    """(L_A^or(X), L_P^or(X)): medians over the report seeds of odour X's cell-sum counts in report pre (Y.3.1)."""
    pre = report["pre"]
    return (float(np.median(np.asarray(pre["A"], float)[:, 0])), float(np.median(np.asarray(pre["P"], float)[:, 0])))


def passes(p: dict, ys) -> dict:
    if not p.get("value") or not p.get("testable"):
        return {c: False for c in COLUMNS}
    r = lambda x: round(float(x), ys.round_digits)  # noqa: E731
    d, a, q = r(p["d_pre"]), r(p["L_A"]), r(p["L_P"])
    bal = abs(d) < ys.naive_max
    return dict(testable=True, balanced=bal,
                y_strict=bal and a >= ys.c_a and q >= ys.c_p,
                y_lenient=abs(d) < ys.lenient_d and a >= ys.lenient_a and q >= ys.lenient_p,
                f2_0=bal and a >= ys.f2_0[0] and q >= ys.f2_0[1],
                f2_25=bal and a >= ys.f2_25[0] and q >= ys.f2_25[1],
                user31=bal and a >= ys.user_a,
                user_onesided31=d > ys.user_onesided_d and a >= ys.user_a)


def count_table(per: list, ys) -> dict:
    out = {}
    for row in ROWS:
        sel = [p for p in per if row == "all" or p["axis"] == row]
        s = [passes(p, ys) for p in sel]
        out[row] = dict({c: sum(x[c] for x in s) for c in COLUMNS}, n=len(sel),
                        no_value=sum(not p["value"] for p in sel), failures=sum(bool(p["failure"]) for p in sel))
    return out


def oracle_decision(table: dict, ys) -> dict:
    t = table["all"]
    if t["failures"]:
        return dict(outcome=INVALID, reasons=[f"지렛대 검사 실패 {t['failures']}쌍(edit / CSC / edges)"])
    k_min = min(lo for lo, _ in ys.k_ranges)
    if t["y_lenient"] < k_min:
        return dict(outcome=STOP_FEW_PAIRS, stage="early", reasons=[],
                    sentence=SENTENCES[(STOP_FEW_PAIRS, "early")].format(n=t["n"], k=t["y_lenient"]),
                    records_unavailable=True, records_reason=RECORDS_REASON,
                    main_set="미사용(Y.0) — 다음 선언은 이 오라클 값을 공개해야 한다(Y.9)")
    return dict(outcome=PASS, reasons=[])
```

`flymon/brain/y_store.py`:
```python
"""Y's only writer (Y.2 code boundary): results/y/ and results/summary/y_learning.json, atomic writes (temporary file +
rename), nothing else (SystemExit 2). The running ledger lives in the summary's `budget` block (Y.7 예산). YCache =
r_store.RCache keyed by the W measurement key (w_measure.py unchanged, Y.2) with `put` through this writer. Phase A
adds here and changes none of these."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

from .h3_store import canonical, canonical_pretty
from .pool_bench import refuse_modified_engine_output, refuse_old_engine_output
from .r_store import RCache

ALLOWED_DIR = "results/y/"
SUMMARY = "results/summary/y_learning.json"


def _refuse(msg: str):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _plain(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _plain(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def to_json(obj):
    return json.loads(json.dumps(_plain(obj), sort_keys=True))


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(os.getcwd())).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        _refuse(f"Y writes only under {ALLOWED_DIR} and {SUMMARY}, not {path}")


def write_bytes(path, data: bytes, params_list) -> Path:
    guard(path, params_list)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return p


def write_json(path, obj, params_list) -> Path:
    return write_bytes(path, (canonical_pretty(to_json(obj)) + "\n").encode(), params_list)


def read_summary(path=SUMMARY) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def write_summary_block(path, block: str, obj, params_list, ledger: dict | None = None) -> Path:
    doc = read_summary(path)
    doc[block] = obj
    if ledger is not None:
        b = dict(doc.get("budget") or {})
        b["ledger"] = list(b.get("ledger", [])) + [ledger]
        doc["budget"] = b
    return write_json(path, doc, params_list)


class YCache(RCache):
    """RCache under results/y/ (a root elsewhere refuses on put); no smoke seeds in 0p (phase A passes its own)."""

    def __init__(self, root, code: dict, smoke_seeds=()):
        super().__init__(root, code, smoke_seeds)

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"key": self.key(kind, inputs), "kind": kind,
                                              "inputs": json.loads(canonical(inputs)), "result": result}, params_list)
```

In `tests/brain/test_p_spec.py`, extend `MODULES`:
```python
           "flymon/brain/v_spec.py": "flymon.brain.v_spec", "flymon/brain/w_spec.py": "flymon.brain.w_spec",
           "flymon/brain/x_spec.py": "flymon.brain.x_spec", "flymon/brain/y_spec.py": "flymon.brain.y_spec"}
```

- [ ] **Step 4: Run the tests to verify they pass, along with the collectors**

Run: `.venv/bin/python -m pytest tests/brain/test_y_spec.py tests/brain/test_y_rules.py tests/brain/test_y_store.py tests/brain/test_p_spec.py tests/brain/test_x_spec.py tests/brain/test_w_spec.py -q > /private/tmp/claude-503/y0p-t1.log 2>&1; tail -5 /private/tmp/claude-503/y0p-t1.log`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/y_spec.py flymon/brain/y_rules.py flymon/brain/y_store.py tests/brain/test_y_spec.py \
        tests/brain/test_y_rules.py tests/brain/test_y_store.py tests/brain/test_p_spec.py
git commit -m "feat(y): 0p-a y_spec / y_rules / y_store — filters, count table, Y.8 sentences, writer guard"
```

---

### Task 2: `y_runner` (`digest`, `oracle`) and `scripts/run_y.py`

**Files:**
- Create: `flymon/brain/y_runner.py`, `scripts/run_y.py`
- Test: `tests/brain/test_y_runner.py`, `tests/brain/test_y_cli.py`

**Interfaces:**
- Consumes everything Task 1 produces, plus the following:
  - `w_runner.build_ctx(W_SPEC, npz)`, whose keys include `w_set`, `main_rows`, `params`, `readout`, `types` and `n_kc`
  - `w_runner.Runner(None, None, wctx, W_SPEC, code=, tcode=, ucode=)._reuse_dec() -> dict(outcome, reasons)`
  - `w_pairs.set_summary`, `h4_formula.pair_stats` and `RMeasurer.oracle`
- Produces:
  - `y_runner.ORDER = ("digest", "oracle")`, `GATES`, `Y_FILES`, `Y_HASHED_FILES`
  - `y_runner.build_ctx(npz) -> dict`, with keys `keys`, `reuse`, `w_set`, `main_rows`, `params`, `measurer(pool)`
  - `y_runner.Runner(ctx, ys, measure=None, summary_path=None)`, with `.stage_digest()` and `.stage_oracle()`, each returning the written block
  - `scripts/run_y.py --stage {digest,oracle} [--workers N]`, with exit codes 0 PASS, 3 STOP, 5 INVALID and 2 refusal

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_y_runner.py`:
```python
"""Y's 0p chain: digest (keys, V reuse through W's _reuse_dec, the main-set digest) then oracle (the count table only in
the block, per-pair values only in results/y/oracle.json); STOP_REUSE, early STOP_FEW_PAIRS, INVALID; refusals; an
interrupted oracle writes no block."""
import json
from pathlib import Path

import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.config import Params
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y

GOOD_Q = dict(edit=V.lever_edit, csc_sha256=V.sha_combined, edit_edges=V.lever_edges)


def _rep(la, lp, st):
    return {"pre": {"A": [[la, 0]] * 8, "P": [[lp, 0]] * 8}, "R1": {}, "R2": {}, "_st": st}


def _st(d, testable=True):
    return dict(d_pre=d, r=3.0, p=-3.0, m=3.0, testable=testable)


# (axis, d_pre, L_A, L_P, testable) — 5 lenient passes by default
PAIRS = [("b", 0.1, 25, 50, True), ("b", 0.7, 17, 35, True), ("a", 0.2, 40, 100, True), ("a", -0.9, 30, 60, True),
         ("b", 0.0, 22, 44, True), ("a", 3.0, 50, 200, True), ("b", 0.1, 25, 50, False)]


class FakeM:
    def __init__(self, pairs, q=GOOD_Q, raise_after=None, none_at=()):
        self.pairs, self.q, self.raise_after, self.none_at, self.calls = pairs, q, raise_after, set(none_at), []

    def oracle(self, rows, cond, block, seeds):
        self.calls.append((cond.name, block, seeds))
        if self.raise_after is not None:
            raise RuntimeError("pool died")
        out = []
        for i, (r, (ax, d, la, lp, t)) in enumerate(zip(rows, self.pairs)):
            p = Path(f"results/y/cache/r_oracle/{i}.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
            out.append(dict(key=f"{ax}|{r['turn']}|x|y", cache_file=str(p), cache_key=f"k{i}",
                            result=dict(q=dict(self.q), report=_rep(la, lp, None if i in self.none_at
                                                                    else _st(d, t)))))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch, pairs=PAIRS):
        monkeypatch.chdir(tmp_path)
        self.pairs = pairs
        self.js = dict(n_b=167, n_a=82, n=249, first_turn=306, last_turn=1967, skipped={}, clusters_b=[],
                       clusters_a=[], digest_keys=Y.main_digest_keys, digest_e0_b="b" * 64, digest_e0_a="a" * 64,
                       n_odours=155, all_off_pool=True, keys=[f"k{i}" for i in range(249)], rows=[])
        self.keys = dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key)
        self.reuse = dict(outcome="PASS", reasons=[])
        self.set_ok = True
        rows = [dict(axis=ax, turn=306 + i, c=i) for i, (ax, *_rest) in enumerate(pairs)]

        def main_rows(blk):
            if not self.set_ok:
                raise ValueError("W set keys differ from block set")
            assert blk["digest_keys"] == Y.main_digest_keys
            return rows
        self.m = FakeM(pairs)
        self.ctx = dict(keys=lambda: dict(self.keys), reuse=lambda: dict(self.reuse), w_set=lambda: dict(self.js),
                        main_rows=main_rows, params=lambda: Params())
        monkeypatch.setattr(YR, "summary_git", lambda p: dict(tracked=True, dirty=False, judged=[]))
        monkeypatch.setattr(YR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(YR, "pair_stats", lambda rep, z, t: rep["_st"])

    def runner(self):
        return YR.Runner(self.ctx, Y, measure=lambda: self.m)


def doc():
    return json.loads(Path(Y.summary).read_text())


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_digest_pass_records_the_set(w):
    out = w.runner().stage_digest()
    assert out["outcome"] == R.PASS and out == doc()["digest"]
    assert out["set"]["digest_keys"] == Y.main_digest_keys and len(out["set"]["keys"]) == 249
    assert "rows" not in out["set"] and doc()["budget"]["ledger"][0]["stage"] == "digest"


@pytest.mark.parametrize("breaker", ["w_key", "u_key", "reuse", "digest", "n_b", "last_turn", "gen"])
def test_digest_stop_reuse(w, breaker):
    if breaker == "w_key":
        w.keys["w_measure_key"] = "x"
    elif breaker == "u_key":
        w.keys["u_measure_key"] = "x"
    elif breaker == "reuse":
        w.reuse = dict(outcome="STOP_REUSE", reasons=["V 커밋 cf0b3b2(set)가 HEAD 이력에 없음"])
    elif breaker == "digest":
        w.js["digest_keys"] = "0" * 64
    elif breaker == "n_b":
        w.js["n_b"] = 166
    elif breaker == "last_turn":
        w.js["last_turn"] = 1985
    else:
        def boom():
            raise ValueError("V's set does not reproduce")
        w.ctx["w_set"] = boom
    out = w.runner().stage_digest()
    assert out["outcome"] == R.STOP_REUSE and out["records_unavailable"] is True
    assert out["sentence"].startswith("Y 재사용 조건(Y.7 0p-b)이 깨졌다(")
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oracle()
    assert e.value.code == 2


def test_oracle_counts_only_in_block_and_values_in_detail(w):
    w.runner().stage_digest()
    out = w.runner().stage_oracle()
    assert out["outcome"] == R.PASS and out == doc()["oracle"]
    assert w.m.calls == [("L", "screen", W.oracle_seeds())]
    det = json.loads(Path(Y.oracle_detail).read_text())
    assert out["counts"] == R.count_table(det["pairs"], Y)
    assert out["counts"]["all"]["y_lenient"] == 5 and out["counts"]["all"]["testable"] == 6
    from flymon.brain.h3_store import sha256_file
    assert out["detail_sha256"] == sha256_file(Y.oracle_detail) and len(det["manifest"]) == len(PAIRS)
    text = json.dumps(out)
    for k in ("d_pre", "L_A", "L_P", '"pairs"', "manifest", "cache_file"):
        assert k not in text, k
    assert det["pairs"][0]["L_A"] == 25.0 and det["pairs"][0]["d_pre"] == 0.1


def test_oracle_early_stop_few_pairs(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, pairs=PAIRS[:3] + PAIRS[5:])     # lenient passes: 3
    w.runner().stage_digest()
    out = w.runner().stage_oracle()
    assert out["outcome"] == R.STOP_FEW_PAIRS and out["stage"] == "early"
    assert "통과한 쌍이 3개로 최소 관문 쌍 수 4에 못 미쳤다" in out["sentence"]


def test_oracle_lever_mismatch_is_invalid_and_no_value_is_counted(w):
    w.runner().stage_digest()
    w.m = FakeM(PAIRS, none_at={0})
    out = w.runner().stage_oracle()
    assert out["counts"]["all"]["no_value"] == 1 and out["outcome"] == R.PASS
    Path(Y.summary).write_text(json.dumps({k: v for k, v in doc().items() if k != "oracle"}))
    w.m = FakeM(PAIRS, q=dict(GOOD_Q, edit_edges=-1))
    out = w.runner().stage_oracle()
    assert out["outcome"] == R.INVALID and out["counts"]["all"]["failures"] == len(PAIRS)


def test_interrupted_oracle_writes_no_block_and_keeps_wall(w):
    w.runner().stage_digest()
    w.m = FakeM(PAIRS, raise_after=0)
    with pytest.raises(RuntimeError):
        w.runner().stage_oracle()
    assert "oracle" not in doc()
    w.m = FakeM(PAIRS)
    assert w.runner().stage_oracle()["outcome"] == R.PASS


def test_refusals(w, monkeypatch):
    with pytest.raises(SystemExit):
        w.runner().stage_oracle()                                  # no digest
    w.runner().stage_digest()
    with pytest.raises(SystemExit):
        w.runner().stage_digest()                                  # block exists
    w.set_ok = False
    with pytest.raises(SystemExit) as e:
        w.runner().stage_oracle()                                  # set no longer reproduces block digest
    assert e.value.code == 2 and "oracle" not in doc()
    w.set_ok = True
    monkeypatch.setattr(YR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/w_pairs.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        w.runner().stage_oracle()


def test_hashed_files_cover_y_and_the_w_code_it_imports():
    for f in ("flymon/brain/y_spec.py", "flymon/brain/y_rules.py", "flymon/brain/y_store.py",
              "flymon/brain/y_runner.py", "scripts/run_y.py", "flymon/brain/w_pairs.py", "flymon/brain/w_runner.py",
              "flymon/brain/w_measure.py", "results/summary/v_lever.json"):
        assert f in YR.Y_HASHED_FILES, f
```

`tests/brain/test_y_cli.py`:
```python
"""run_y.py: arguments, cwd refusal, exit codes, quiet output (no per-pair key printed)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_y", ROOT / "scripts/run_y.py")
run_y = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_y)


def test_stages_and_exit_codes():
    assert run_y.STAGES == ("digest", "oracle") and run_y.POOL_STAGES == ("oracle",)
    assert run_y.exit_code({"outcome": "PASS"}) == 0
    assert run_y.exit_code({"outcome": "STOP_FEW_PAIRS"}) == 3 and run_y.exit_code({"outcome": "STOP_REUSE"}) == 3
    assert run_y.exit_code({"outcome": "INVALID"}) == 5


def test_refuses_outside_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_y.main(["--stage", "digest"]) == 2


def test_quiet_keys_hide_records():
    assert {"set", "reuse", "git", "seeds", "z_V"} <= set(run_y.QUIET)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/brain/test_y_runner.py tests/brain/test_y_cli.py -q > /private/tmp/claude-503/y0p-t2.log 2>&1; tail -5 /private/tmp/claude-503/y0p-t2.log`
Expected: errors with `ModuleNotFoundError: No module named 'flymon.brain.y_runner'` and a missing `scripts/run_y.py`.

- [ ] **Step 3: Implement**

`flymon/brain/y_runner.py`:
```python
"""Spec Y's stage chain, step 0p (Y.7 0p-b · 0p-c): digest -> oracle, one block each in results/summary/y_learning.json,
written only through y_store, plus the running ledger under `budget` (Y's own 24 h). Phase A appends its stages to
ORDER after oracle and never edits these two.
- digest: the U / W measurement keys, V's blocks through W's own reuse decision (w_runner.Runner._reuse_dec: V's
  z / kc_input / set / judge on V's keys, SELECTED, z_V as declared; R shared / T / U keys), then w_pairs.w_set
  regenerated: digest_keys 65dbf001… (0p plan Reading 1), (b) 167 · (a) 82, last turn 1967. Any break -> STOP_REUSE.
  The block records set_summary + keys (W never wrote a set block).
- oracle: W's stage_oracle path — main_rows checked against block digest's set, RMeasurer.oracle(rows,
  V_SPEC.cond("L"), "screen", W_SPEC.oracle_seeds()) over results/y/cache (W measurement key), pair_stats on z_V,
  the lever check per pair, L_A^or / L_P^or from report pre. Per-pair values go only to results/y/oracle.json
  (git-ignored, sha256 in the block); the block holds the count table only (author's reading 6). Resumable: the cache
  holds every finished pair; a killed run writes no block, and the wall time accumulates in results/y/progress.
Every stage refuses (SystemExit 2, nothing written) when an earlier block is missing, a later block or its own block
exists, an earlier gate did not PASS, the summary has uncommitted changes, or a hashed Y / W / V file is dirty."""
from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

from ..agent.e_runner import summary_git
from . import w_runner, y_rules, y_store
from .h3_store import git_state as _h3_git_state
from .h3_store import sha256_file
from .h4_formula import pair_stats
from .v_spec import SPEC as V_SPEC
from .w_pairs import set_summary
from .w_spec import SPEC as W_SPEC

ORDER = ("digest", "oracle")
GATES = ("digest", "oracle")
Y_FILES = ("flymon/brain/y_spec.py", "flymon/brain/y_rules.py", "flymon/brain/y_store.py", "flymon/brain/y_runner.py",
           "scripts/run_y.py")
W_FILES = tuple(dict.fromkeys(tuple(w_runner.W_PIPELINE_FILES) + ("flymon/brain/w_measure.py",)))
Y_HASHED_FILES = tuple(dict.fromkeys(Y_FILES + W_FILES + tuple(w_runner.V_HASHED_FILES)
                                     + ("results/summary/v_lever.json", "results/summary/w_learning.json")))


def git_state() -> dict:
    return _h3_git_state(files=Y_HASHED_FILES)


def refuse(msg: str, code: int = 2):
    print(f"refusing: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def build_ctx(npz: str) -> dict:
    """W's context (built once, read only) and the keys; measurer(pool) = an RMeasurer over U's job copies behind
    YCache(results/y/cache, W measurement key) with z_V as declared."""
    from .h3_store import code_key
    from .r_measure import R_MEASURE_FILES, RMeasurer
    from .t_measure import t_measure_key
    from .u_measure import UPool, u_measure_key
    from .w_measure import w_measure_key
    from .y_spec import SPEC as YS
    cache = {}

    def wctx():
        if "c" not in cache:
            cache["c"] = w_runner.build_ctx(W_SPEC, npz)
        return cache["c"]

    def reuse():
        r = w_runner.Runner(None, None, wctx(), W_SPEC, code=code_key(npz, files=R_MEASURE_FILES),
                            tcode=t_measure_key(npz), ucode=u_measure_key(npz))
        return r._reuse_dec()

    def measurer(pool):
        c = wctx()
        return RMeasurer(UPool(pool), y_store.YCache(YS.cache_dir, w_measure_key(npz)), V_SPEC, c["params"],
                         c["readout"], W_SPEC.z_v(), c["types"], c["n_kc"])
    return dict(keys=lambda: dict(w_measure_key=w_measure_key(npz)["key"], u_measure_key=u_measure_key(npz)["key"]),
                reuse=reuse, w_set=lambda: wctx()["w_set"](), main_rows=lambda blk: wctx()["main_rows"](blk),
                params=lambda: wctx()["params"], measurer=measurer)


class Runner:
    def __init__(self, ctx: dict, ys, measure=None, summary_path=None):
        self.ctx, self.ys, self.measure = ctx, ys, measure
        self.summary_path = str(summary_path or ys.summary)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return y_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed Y / W / V files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        self._clean(stage)
        doc = self._doc()
        i = ORDER.index(stage)
        missing = [b for b in ORDER[:i] if b not in doc]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in ORDER[i + 1:] if b in doc]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; Y never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; Y never rewrites a recorded block")
        stopped = [g for g in GATES if g in ORDER[:i] and doc[g].get("outcome") != y_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {stopped[0]} outcome {doc[stopped[0]].get('outcome')} — Y stops there")
        return doc

    def _write(self, stage: str, body: dict, wall_s: float) -> dict:
        k = self.ctx["keys"]()
        block = y_store.to_json(dict(body, stage=stage, w_measure_key=k.get("w_measure_key"),
                                     u_measure_key=k.get("u_measure_key"), git=git_state(), written_at=_now()))
        y_store.write_summary_block(self.summary_path, stage, block, self.plist,
                                    dict(stage=stage, wall_s=float(wall_s), at=block["written_at"]))
        return block

    def _prog_path(self, stage: str) -> str:
        return f"{self.ys.progress_dir}/{stage}.json"

    def _prog(self, stage: str) -> float:
        p = Path(self._prog_path(stage))
        return float(json.loads(p.read_text())["wall_s"]) if p.exists() else 0.0

    def _prog_add(self, stage: str, s: float) -> float:
        tot = self._prog(stage) + float(s)
        y_store.write_json(self._prog_path(stage), dict(wall_s=tot, at=_now()), self.plist)
        return tot

    # ================================================================ 0p-b: partial reuse + the main-set digest
    def stage_digest(self) -> dict:
        self._require("digest")
        ys, t0 = self.ys, time.perf_counter()
        keys = self.ctx["keys"]()
        why = y_rules.key_reasons(keys, ys)
        dec = self.ctx["reuse"]()
        if dec.get("outcome") != y_rules.PASS:
            why += list(dec.get("reasons") or [dec.get("outcome")])
        js = None
        try:
            js = self.ctx["w_set"]()
        except ValueError as e:
            why.append(f"주 세트 재생성 불가: {e}")
        if js is not None:
            why += y_rules.digest_reasons(js, ys)
        body = y_rules.reuse_stop(why, "0p-b") if why else dict(outcome=y_rules.PASS, reasons=[])
        body.update(set=None if js is None else dict(set_summary(js), keys=js["keys"]), keys=keys,
                    v_reuse=dict(outcome=dec.get("outcome"), reasons=list(dec.get("reasons") or [])),
                    note="Y.7 0p-b: U · W 측정 키, V 블록(w_rules.reuse), 주 세트 digest_keys · (b) · (a) · 마지막 턴.")
        return self._write("digest", body, time.perf_counter() - t0)

    # ================================================================ 0p-c: the main-set oracle (count table only)
    def stage_oracle(self) -> dict:
        doc = self._require("oracle")
        ys = self.ys
        try:
            rows = self.ctx["main_rows"](doc["digest"]["set"])
        except ValueError as e:
            refuse(f"the main set does not reproduce block digest: {e}")
        z = W_SPEC.z_v()
        seeds = W_SPEC.oracle_seeds()
        m = self.measure()
        t0 = time.perf_counter()
        try:
            got = m.oracle(rows, V_SPEC.cond("L"), "screen", seeds)
        finally:
            wall = self._prog_add("oracle", time.perf_counter() - t0)
        want_q = (V_SPEC.lever_edit, V_SPEC.sha_combined, V_SPEC.lever_edges)
        per, man = [], []
        for r, g in zip(rows, got):
            res = g["result"]
            q = res.get("q", {})
            st = pair_stats(res["report"], z, V_SPEC.testable_min)
            la, lp = y_rules.levels(res["report"])
            per.append(dict(key=g["key"], c=r["c"], axis=r["axis"], turn=r["turn"], value=st is not None,
                            failure=(q.get("edit"), q.get("csc_sha256"), q.get("edit_edges")) != want_q,
                            testable=bool(st and st["testable"]), d_pre=st and st["d_pre"], r=st and st["r"],
                            p=st and st["p"], L_A=la, L_P=lp))
            man.append(dict(key=g["key"], cache_file=g["cache_file"], cache_key=g["cache_key"],
                            sha256=sha256_file(g["cache_file"])))
        table = y_rules.count_table(per, ys)
        dec = y_rules.oracle_decision(table, ys)
        p = y_store.write_json(ys.oracle_detail, dict(pairs=per, manifest=man, seeds=seeds,
                                                      z_V={k: list(v) for k, v in z.items()}), self.plist)
        body = dict(dec, counts=table, n=len(per), detail_path=ys.oracle_detail, detail_sha256=sha256_file(p),
                    seeds=seeds, z_V={k: list(v) for k, v in z.items()}, condition="L", block="screen",
                    note="Y.7 0p-c: 개수 표만(쌍별 값은 git 제외 상세 파일, 글쓴이 해석 6). 오라클 · 순진 pre만으로는 "
                         "주 세트를 사용한 것이 아니다(Y.0).")
        return self._write("oracle", body, wall)
```

Note that `test_interrupted_oracle_writes_no_block_and_keeps_wall` passes because the `finally` records wall time even when the measurer raises, and the exception propagates before `_write` runs.

`scripts/run_y.py`:
```python
#!/usr/bin/env python3
"""Spec appendix Y, step 0p (Y.7 0p, in parallel with the red-team): the controller runs each stage and commits its block
before the next.

    .venv/bin/python scripts/run_y.py --stage digest                 # 0p-b keys, V blocks, main-set digest (no pool)
    .venv/bin/python scripts/run_y.py --stage oracle --workers 16    # 0p-c 249-pair oracle (24_700_xxx; resumes)

Exit 0 PASS, 3 a gate STOP (STOP_REUSE, STOP_FEW_PAIRS; recorded, Y stops), 5 INVALID (a code / machine defect: do not
commit), 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted summary, dirty hashed file). Output: the
outcome, sentence and the count table only (Y.7 0p-c read rule)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID = 3, 5
STAGES = ("digest", "oracle")
POOL_STAGES = ("oracle",)
QUIET = ("set", "reuse", "v_reuse", "keys", "git", "seeds", "z_V", "detail_path")


def exit_code(out: dict) -> int:
    from flymon.brain import y_rules as R
    o = out.get("outcome")
    return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    from flymon.brain import y_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.w_spec import SPEC as W
    from flymon.brain.y_spec import SPEC
    pool = None
    try:
        ctx = y_runner.build_ctx(NPZ)
        if a.stage in POOL_STAGES:
            workers = a.workers or SPEC.workers
            pool = FlyPool(NPZ, ctx["params"](), flies=[{}] * workers, workers=workers, punish_type=W.punish_dan,
                           reward_type=W.reward_dan, timeout_s=SPEC.pool_timeout_s)
        try:
            out = getattr(y_runner.Runner(ctx, SPEC, measure=lambda: ctx["measurer"](pool)), f"stage_{a.stage}")()
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
    print(json.dumps({k: v for k, v in out.items() if k not in QUIET}, ensure_ascii=False,
                     default=str)[:SPEC.cli_print_chars])
    return exit_code(out)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the Y tests, the literal guard and the W/X regression slice**

Run: `.venv/bin/python -m pytest tests/brain/test_y_*.py tests/brain/test_p_spec.py tests/brain/test_w_pairs.py tests/brain/test_x_spec.py -q > /private/tmp/claude-503/y0p-t2.log 2>&1; tail -5 /private/tmp/claude-503/y0p-t2.log`
Expected: all pass. `test_no_y_file_but_y_spec_holds_a_number` now also covers `y_runner.py` and `run_y.py`.

Then confirm W and X are untouched: `git diff --stat main...HEAD -- flymon/brain/w_*.py flymon/brain/x_*.py scripts/run_w.py scripts/run_x.py` should print nothing.

- [ ] **Step 5: Commit**

```bash
git add flymon/brain/y_runner.py scripts/run_y.py tests/brain/test_y_runner.py tests/brain/test_y_cli.py
git commit -m "feat(y): 0p y_runner digest / oracle + run_y.py — count table only, detail git-ignored, resumable"
```

---

## Runs (controller)

**Preconditions**
- Tasks 1–2 are committed and reviewed. `git status` shows no dirty file in `y_runner.Y_HASHED_FILES`. `results/summary/y_learning.json` does not exist yet.
- `data/malecns.npz` is present and matches its sha256.
- No other FlyPool job is running on this machine. The red-team is text-only and may run concurrently.
- Run from the repository root. The background run uses Bash `run_in_background: true` and is not polled.

**1. `digest` (0p-b, seconds, no pool)**
```bash
.venv/bin/python scripts/run_y.py --stage digest > /private/tmp/claude-503/y0p-digest.log 2>&1; echo "exit $?"
```
- **Exit 0:** commit the block:
  ```bash
  git add results/summary/y_learning.json
  git commit -m "results(y): 0p-b digest — keys U 8a4e0930… / W 761274e0…, V blocks OK, main set digest_keys 65dbf001…, (b) 167 · (a) 82, last turn 1967"
  ```
- **Exit 3 (`STOP_REUSE`):** follow the STOP path below.
- **Exit 2:** fix the precondition and rerun.

**2. `oracle` (0p-c, about 3 h on 16 workers: 249 × 704 s / 16)**
```bash
.venv/bin/python scripts/run_y.py --stage oracle --workers 16 > /private/tmp/claude-503/y0p-oracle.log 2>&1; echo "exit $?"
```
- Run it in the background. If the run is killed (for example by a 2 h limit), rerun the same command. `results/y/cache` keeps every finished pair, so only the missing pairs are measured, with the same seeds and the same values. The wall time accumulates in `results/y/progress/oracle.json`.
- On completion, read only the exit code and the last two lines of the log (the sentence and the count JSON). **Do not open `results/y/oracle.json`** until Y.9.2 is committed.
- **Exit 0 (PASS, lenient ≥ 4):** commit, putting the count table's `all` row in the message:
  ```bash
  git add results/summary/y_learning.json
  git commit -m "results(y): 0p-c oracle — counts only: all n 249, testable …, balanced …, Y strict …, Y lenient …, F2(0) …, F2(25) …, user31 …, one-sided+31 … ((b)/(a) in block); detail sha256 …"
  ```
  Then wait for the red-team (Y.9.2) before planning phase A.
- **Exit 3 (early `STOP_FEW_PAIRS`):** commit the block the same way, then follow the STOP path.
- **Exit 5 (`INVALID`: lever mismatch):** do not commit. Run `git checkout results/summary/y_learning.json`, keep `results/y/` for diagnosis, and treat it as an infrastructure defect (systematic debugging), not a rerun.

**STOP path (`STOP_REUSE` or early `STOP_FEW_PAIRS`; the user's standing approval of 2026-10-06 applies)**
1. Write **Y.10** in the spec. It records the verbatim sentence, the count table (all / (b) / (a)), the detail sha256 and `records_unavailable` with its reason. It states that the main set is **unused** (Y.0 / Y.9) and that the next declaration must disclose these oracle values. It also lists the next-declaration candidates per Y.9 (the user's one-sided replacement, a pre-declared split of the main set, a new generator).
2. Add the README ledger row in ko and en.
3. Commit with `docs(y): Y.10 result — <STOP> …`. Switch the account with `gh auth switch --user lyutvs`, then `git push origin HEAD`.
4. Continue per the standing approval by writing the next-declaration candidates into the brainstorm record (Y.9 last bullet). Stop for the user only when no recommended option exists.
