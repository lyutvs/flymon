# `apl_input_scale` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one engine parameter, `Params.apl_input_scale`, that multiplies the weight of every CSC edge whose target is an APL cell, so the H.3 runner can place APL on the usable part of its release curve.

**Architecture:** The multiply happens once, inside `build_csc`, immediately after the `apl_scale` multiply and after the hemisphere correction — the same float32 order the calibration prototypes used, so their numbers reproduce bit for bit. The default 1.0 skips the multiply entirely, which keeps every existing CSC bit-identical. Three call sites that key on engine settings learn about the new field: the M0d write guard, the sparsity-grid row lookup in `pool_bench`, and the M0c summary writer's engine check.

**Tech Stack:** Python 3.12, NumPy, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` — appendix **H.3a.2** (the parameter, its position in `build_csc`, and the seven tests), with H.2 for the mode table this extends.

## Global Constraints

- **Commits carry no trailers of any kind** — no `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder that asks for them.
- **Default behaviour is bit-identical.** `Params()` unchanged means every existing CSC, step and reference result stays exactly as it is; `results/m0`, `results/m0b`, `results/m0c` are frozen references.
- **Valid range is `0 < apl_input_scale <= 1`.** Values outside that, and non-finite values, are rejected at engine construction.
- **Full suite before each commit:** `uv run pytest -q -rN -o addopts=""` — currently **239 passed, 1 skipped, 1 xfailed** (241 collected). Every task must leave that count at or above this, with no new failures.
- Tests for the engine modes live in `tests/brain/test_engine_modes.py` and use the `synthetic_connectome` fixture from `tests/conftest.py`; follow that file's style (module-level `BASE` params, `_engine(...)` helper).
- Do not touch `flymon/brain/engine_cpu.py`'s step loop: this parameter changes weights at build time only.

---

### Task 1: The parameter and the multiply in `build_csc`

**Files:**
- Modify: `flymon/brain/config.py` (the M0d mode block, after `kc_thresh_sha256`)
- Modify: `flymon/brain/connectome.py:101-136` (`build_csc`)
- Modify: `flymon/brain/engine_cpu.py:189-197` (`_validate_modes`)
- Test: `tests/brain/test_engine_modes.py`

**Interfaces:**
- Consumes: `Params` (dataclass in `flymon/brain/config.py`), `build_csc(conn, params, apl_idx, kc_idx) -> CSC`, `CSC(ptr, tgt, w)`.
- Produces: `Params.apl_input_scale: float = 1.0`; `build_csc` honouring it; `_validate_modes` raising `ValueError` for values outside `(0, 1]` or non-finite.

- [ ] **Step 1: Write the failing tests**

Add to `tests/brain/test_engine_modes.py`, after the defaults section:

```python
# ---- apl_input_scale (spec H.3a.2) --------------------------------------------------------------------------
def _csc_of(synthetic_connectome, **kw):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    return build_csc(c, Params(**{**BASE, **kw}), pops.apl, pops.kc), c, pops


def test_apl_input_scale_defaults_to_one_and_keeps_the_csc_bit_identical(synthetic_connectome):
    assert Params().apl_input_scale == 1.0
    a, _, _ = _csc_of(synthetic_connectome)
    b, _, _ = _csc_of(synthetic_connectome, apl_input_scale=1.0)
    assert np.array_equal(a.w, b.w) and np.array_equal(a.tgt, b.tgt) and np.array_equal(a.ptr, b.ptr)


def test_apl_input_scale_scales_exactly_the_edges_into_apl(synthetic_connectome):
    base, c, pops = _csc_of(synthetic_connectome)
    scaled, _, _ = _csc_of(synthetic_connectome, apl_input_scale=0.25)
    into_apl = np.isin(base.tgt, np.asarray(pops.apl, np.int64))
    assert into_apl.any(), "the synthetic connectome must have edges into APL"
    assert np.array_equal(scaled.w[into_apl], (base.w[into_apl] * np.float32(0.25)).astype(np.float32))
    assert np.array_equal(scaled.w[~into_apl], base.w[~into_apl])


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_apl_input_scale_outside_the_declared_range_is_rejected(synthetic_connectome, bad):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    with pytest.raises(ValueError, match="apl_input_scale"):
        Engine(c, pops, Params(**{**BASE, "apl_input_scale": bad}), seed=1)
```

Add `build_csc` to the imports at the top of the file:

```python
from flymon.brain.connectome import Connectome, build_csc
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_engine_modes.py -k apl_input_scale -v -o addopts=""`
Expected: FAIL — `TypeError: Params.__init__() got an unexpected keyword argument 'apl_input_scale'`.

- [ ] **Step 3: Add the field**

In `flymon/brain/config.py`, in the M0d block, directly after `kc_thresh_sha256: str = ""`:

```python
    apl_input_scale: float = 1.0   # multiplier on every CSC edge INTO an APL cell (spec H.3a.2); 1.0 = no multiply.
                                   # Stands in for the summation saturation the point-neuron APL lacks: its membrane
                                   # is the unattenuated sum of ~2,059 KC inputs. Valid range (0, 1].
```

- [ ] **Step 4: Apply it in `build_csc`**

In `flymon/brain/connectome.py`, inside `build_csc`, the APL block currently reads:

```python
    is_apl = np.zeros(conn.N, bool)
    is_apl[apl_idx] = True
    m = is_apl[pre]
    mv[m] *= np.float32(params.apl_scale)
```

Replace it with (note the order: hemisphere correction and `apl_scale` first, then the input scale — the calibration prototypes multiplied the finished CSC, so this order reproduces them):

```python
    is_apl = np.zeros(conn.N, bool)
    is_apl[apl_idx] = True
    m = is_apl[pre]
    mv[m] *= np.float32(params.apl_scale)
    if params.apl_input_scale != 1.0:      # 1.0 is the pre-H.3a engine: no mask, no multiply, bit-identical CSC
        m = is_apl[post]
        mv[m] *= np.float32(params.apl_input_scale)
```

- [ ] **Step 5: Validate the range**

In `flymon/brain/engine_cpu.py`, at the end of `_validate_modes`:

```python
    if not (math.isfinite(p.apl_input_scale) and 0 < p.apl_input_scale <= 1):
        raise ValueError(f"apl_input_scale must be finite and in (0, 1], got {p.apl_input_scale}")
```

Add `import math` to that module's imports.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/brain/test_engine_modes.py -k apl_input_scale -v -o addopts=""`
Expected: 7 passed (3 tests, the range one parametrised 5 ways).

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q -rN -o addopts=""`
Expected: 246 passed, 1 skipped, 1 xfailed (239 + 7 new).

- [ ] **Step 8: Commit**

```bash
git add flymon/brain/config.py flymon/brain/connectome.py flymon/brain/engine_cpu.py tests/brain/test_engine_modes.py
git commit -m "feat(brain): apl_input_scale, a multiplier on the edges into APL"
```

---

### Task 2: Prototype equivalence and the pool = in-process check

**Files:**
- Test: `tests/brain/test_engine_modes.py`

**Interfaces:**
- Consumes: `Params.apl_input_scale` and `build_csc` from Task 1; `FlyPool`, `FlySpec`, `decide` as already imported in this test module.
- Produces: no source changes — two tests that pin the property the calibration record depends on.

The calibration runs (`docs/superpowers/specs/m0d-diag/*.py`) did **not** use this parameter: they built a default CSC and multiplied it afterwards, `e.csc.w[is_apl[e.csc.tgt]] *= np.float32(s)`. Spec H.3a.2 test (6) requires the two paths to agree bit for bit, otherwise the adopted operating point in H.3a.7 does not carry over to the implemented engine.

- [ ] **Step 1: Write the failing tests**

```python
def test_apl_input_scale_matches_the_diagnostic_prototype_path(synthetic_connectome):
    """Spec H.3a.2 test (6): the calibration scripts multiplied the finished CSC; the parameter must agree bit
    for bit, or the operating point recorded in H.3a.7 does not carry over."""
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    s = 0.11863                                   # the adopted operating point's value (spec H.3a.7)
    prototype = build_csc(c, Params(**BASE), pops.apl, pops.kc)
    is_apl = np.zeros(c.N, bool)
    is_apl[np.asarray(pops.apl, np.int64)] = True
    prototype.w[is_apl[prototype.tgt]] *= np.float32(s)
    implemented = build_csc(c, Params(**{**BASE, "apl_input_scale": s}), pops.apl, pops.kc)
    assert np.array_equal(implemented.w, prototype.w)


def test_pool_decide_equals_in_process_with_apl_input_scale(synthetic_npz, tmp_path):
    """Every engine mode has to survive the spawn boundary identically (spec H.2 test 5)."""
    params = Params(**{**BASE, "apl_mode": "graded", "apl_r_max": 0.333, "apl_input_scale": 0.25})
    conn = Connectome.load(synthetic_npz)
    pops = Populations.from_connectome(conn)
    eng = Engine(conn, pops, params, seed=7)
    pl = Plasticity(eng, pops, compartments(conn, pops, params.core_frac), params)
    ro = sorted(pops.receptor_types)[:2]
    odors = [{ro[0]: 1.0}, {ro[1]: 1.0}]
    in_process = decide(eng, pl, pops, odors, strength=0.35, seed=11, settle_ms=20.0, read_ms=20.0)
    with FlyPool(synthetic_npz, params, [FlySpec()], workers=1) as pool:
        pooled = pool.decide_batch([(0, odors, 11)], strength=0.35, settle_ms=20.0, read_ms=20.0)[0]
    assert np.array_equal(np.asarray(pooled), np.asarray(in_process))
```

The call shape above is the one `test_pool_decide_equals_in_process_in_every_new_mode` already uses in this file; keep it identical so the two tests fail for the same reasons.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brain/test_engine_modes.py -k "prototype or in_process_with_apl" -v -o addopts=""`
Expected: the prototype test FAILs only if Task 1's multiply sits in the wrong place (this is the point of the test); the pool test FAILs if the parameter does not survive the spawn. If both pass immediately, that is the expected outcome of a correct Task 1 — record that in the commit message rather than weakening the tests.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q -rN -o addopts=""`
Expected: 248 passed, 1 skipped, 1 xfailed.

- [ ] **Step 4: Commit**

```bash
git add tests/brain/test_engine_modes.py
git commit -m "test(brain): apl_input_scale matches the diagnostic path and survives the pool"
```

---

### Task 3: The three settings-aware call sites

**Files:**
- Modify: `flymon/brain/pool_bench.py:115-127` (`refuse_modified_engine_output`) and `:150-175` (`match_sparsity_row`)
- Modify: `scripts/write_m0c_summary.py:86-92` (the sparsity grid row lookup)
- Test: `tests/brain/test_pool_bench.py`

**Interfaces:**
- Consumes: `Params.apl_input_scale` from Task 1.
- Produces: `refuse_modified_engine_output` treating `apl_input_scale != 1.0` as a modified engine; `match_sparsity_row` and `write_m0c_summary`'s row lookup keying on `apl_input_scale` as well, with rows lacking the key read as 1.0.

Without this, a run with a scaled APL can overwrite the frozen M0/M0b/M0c references, and two runs that differ only in this parameter match the same sparsity-grid row.

- [ ] **Step 1: Write the failing test**

Add to `tests/brain/test_pool_bench.py`:

```python
def test_refuse_modified_engine_output_covers_apl_input_scale(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from flymon.brain.config import Params
    from flymon.brain.pool_bench import refuse_modified_engine_output
    refuse_modified_engine_output("results/m0c/sparsity.json", Params())          # default: allowed
    with pytest.raises(SystemExit) as e:
        refuse_modified_engine_output("results/m0c/sparsity.json", Params(apl_input_scale=0.5))
    assert e.value.code == 2
    assert "apl_input_scale=0.5" in capsys.readouterr().err
    refuse_modified_engine_output("results/m0d/diag/x.json", Params(apl_input_scale=0.5))   # m0d path: allowed
```

`tests/brain/test_pool_bench.py` already imports `pytest`, `Params` and `refuse_modified_engine_output` at module level, so drop the local imports above and use the module's.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/brain/test_pool_bench.py -k apl_input_scale -v -o addopts=""`
Expected: FAIL — no `SystemExit`, because the guard does not look at the new field.

- [ ] **Step 3: Extend the guard**

In `flymon/brain/pool_bench.py`, `refuse_modified_engine_output`:

```python
    modified = (params.apl_mode != "spiking" or params.orn_std or params.kc_thresh_mode != "pn_norm"
                or params.apl_input_scale != 1.0)
```

and add the value to the message:

```python
        print(f"refusing to write {out} with M0d modes on (apl_mode={params.apl_mode}, orn_std={params.orn_std}, "
              f"kc_thresh_mode={params.kc_thresh_mode}, apl_input_scale={params.apl_input_scale}): that path holds "
              f"a pre-M0d engine's reference; use results/m0d/", file=sys.stderr)
```

- [ ] **Step 4: Extend the two grid-row lookups**

In `flymon/brain/pool_bench.py`, `match_sparsity_row`:

```python
    rows = [g for g in reference["grid"]
            if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac"), g.get("kc_kc_scale", 1.0),
                g.get("apl_input_scale", 1.0))
            == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac, p.kc_kc_scale, p.apl_input_scale)]
    if not rows:
        return {"ok": False, "note": f"no reference grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} "
                                     f"mbon_hold_frac={p.mbon_hold_frac} kc_kc_scale={p.kc_kc_scale} "
                                     f"apl_input_scale={p.apl_input_scale}"}
```

In `scripts/write_m0c_summary.py`, the same change in its row lookup (the `raise SystemExit` branch keeps its shape, with `apl_input_scale` added to the message).

- [ ] **Step 5: Run the test and the full suite**

Run: `uv run pytest tests/brain/test_pool_bench.py -k apl_input_scale -v -o addopts=""`
Expected: PASS.

Run: `uv run pytest -q -rN -o addopts=""`
Expected: 249 passed, 1 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add flymon/brain/pool_bench.py scripts/write_m0c_summary.py tests/brain/test_pool_bench.py
git commit -m "feat(bench): the write guard and grid-row lookups know apl_input_scale"
```

---

### Task 4: Old-config migration and the real-data reproduction check

**Files:**
- Test: `tests/brain/test_config.py`
- Create: `results/m0d/diag/apl_input_scale_impl_check.json` (git-excluded output of the check command below)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: a test pinning that a `Params` dict saved before this change still loads (as 1.0), and a recorded measurement showing the implemented parameter reproduces the calibration numbers on the real connectome.

- [ ] **Step 1: Write the failing test**

Add to `tests/brain/test_config.py`:

```python
def test_params_from_an_older_run_loads_with_apl_input_scale_one():
    """Summaries written before H.3a have no apl_input_scale; they are the unscaled engine."""
    import dataclasses
    from flymon.brain.config import Params
    old = dataclasses.asdict(Params())
    old.pop("apl_input_scale")
    assert Params(**old).apl_input_scale == 1.0
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/brain/test_config.py -k older_run -v -o addopts=""`
Expected: PASS (dataclass defaults already give this). If it FAILs, the field was added with no default — fix Task 1 rather than the test.

- [ ] **Step 3: Reproduce the adopted operating point on the real connectome**

Spec H.3a.2 asks for one confirmation run after implementation: the adopted point must reproduce its recorded numbers. Run this from the repo root — it takes about a minute on 16 workers:

```bash
uv run python - <<'PY'
import json, numpy as np
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
import sys; sys.path.insert(0, "docs/superpowers/specs/m0d-diag")
from apl_input_scale_sweep import NPZ, STRENGTH, SETTLE_MS, READ_STEPS, reference_odors, job
pops = Populations.from_connectome(Connectome.load(NPZ))
odors = reference_odors(pops)
params = dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.6, apl_input_scale=0.11863)
with FlyPool(NPZ, Params(), [{} for _ in range(16)], workers=16) as pool:
    rows = pool.run_jobs(job, [dict(params=params, scale=1, odors=odors)])
pres = rows[0]
kc = 100 * float(np.median([p["kc_active_frac"] for p in pres]))
mv = float(np.median([p["apl_v_mean"] for p in pres]))
print(json.dumps({"kc_active_pct_median": kc, "apl_mv_median": mv}, indent=1))
PY
```

Those names are the committed ones in `docs/superpowers/specs/m0d-diag/apl_input_scale_sweep.py` (`8b51594`): `NPZ`, `STRENGTH`, `SETTLE_MS`, `READ_STEPS`, `reference_odors(pops)`, `job(...)`. Note `job` takes `scale` and applies the post-hoc multiply when it is not 1 — passing `scale=1` with `apl_input_scale` in `params` is what puts the new parameter on the measured path.

Expected, from spec H.3a.7 (measured at `mbon_hold_frac` 0.85, which is this command's default): **KC active median ≈ 6.37%**, **APL membrane median ≈ 10.90 mV**. Agreement to 0.05 pp and 0.05 mV is a pass — the parameter path and the prototype path are bit-identical, so any larger difference means the multiply landed in the wrong place.

- [ ] **Step 4: Record the result**

Write the printed JSON, plus the commit hash and the two expected values, to `results/m0d/diag/apl_input_scale_impl_check.json`. `results/` is git-ignored; this file is the local record that H.3a.2's confirmation run happened.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q -rN -o addopts=""`
Expected: 250 passed, 1 skipped, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add tests/brain/test_config.py
git commit -m "test(config): params saved before H.3a load with apl_input_scale 1.0"
```

---

## What this plan does not do

- The H.3 runner (grid scan, bisections, qualification judging, bootstrap CIs, resume keys) is a separate plan — spec H.3a.11 asks for it as its own SDD.
- No `results/m0`, `results/m0b`, `results/m0c` output changes: the default path stays bit-identical, which is what Task 1's first test pins.
- The reparameterisation comparison H.3a.2 mentions (`apl_v_mid` 11/s, `apl_slope` 5/s at the adopted point) is a record, not a gate; it belongs with the runner's diagnostics, not here.
