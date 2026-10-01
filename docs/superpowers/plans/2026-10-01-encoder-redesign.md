# Encoder Redesign (E-grid) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the E-grid encoder (glomerulus-disjoint conjunctive code over move type × opponent type), its calibration, the even-turn selection and the one judgement on a fresh pair set, exactly as the spec fixes them, so the controller can run stages ⓪–⑥ in the background.

**Architecture:** Pure modules under `flymon/agent/` (spec constants, codebook construction, odours, pair sets, rules) are tested without the engine. One measurer wraps `FlyPool` and calls existing brain jobs (`k_jobs.activity_job`, `h4_jobs.oracle_job`) through a track-local content-key cache. A stage runner (procedure) talks only to a measurer interface; the CLI script builds the real measurer. Nothing under `flymon/brain/` is edited.

**Tech Stack:** Python 3.13, numpy, pytest, `uv run`; existing `flymon.brain` engine (C3 Params from `results/summary/m0d.json`), `FlyPool` (spawn).

**Spec:** `docs/superpowers/specs/2026-10-01-encoder-redesign-design.md` (read it whole before any task; section numbers below are that spec's).

## Global Constraints

- Never modify `flymon/brain/*`, `flymon/agent/encode.py`, the main spec `docs/superpowers/specs/2026-09-14-flymon-design.md`, `README.md`, `data/odor/`, `tests/test_m2_go.py`. Call brain code only. If a task seems to need a brain change, stop and report `BLOCKED`.
- No new data directory. Raw outputs only under `results/encoder/` (git-ignored); the one committed summary is `results/summary/encoder_grid.json`.
- Commit messages: plain body text, **no** `Co-Authored-By`, `Claude-Session` or "Generated with" lines.
- Excluded glomeruli: `ORN_DA1`, `ORN_V`. Alphabet k = 3: all 49 others; k = 2: the 40 closest to the drive band (spec 3.1).
- All logs are `log10(x + 1)`.
- Seeds (spec 4.5): drive `24_001_000–24_001_007`; strength `24_002_000–24_002_007`; judgement oracle act `24_100_000–24_100_007`, select `24_100_100–24_100_107`, report `24_100_200–24_100_207`; smoke `24_009_000–24_009_099`; even-turn oracle = H.4's 500–507 / 600–607 / 608–615.
- Strength grid `(0.175, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)`; KC activity target `0.0554`, band `[0.05, 0.09]`, tails `< 0.03` and `> 0.15` each `≤ 10 %` per type; ORN cap hard: `max_rate_hz × s × s_g ≤ cap_hz` (`odor_real.cap_hz`).
- Config order `("k3-full", "k3-norm", "k2-full", "k2-norm")`; tie window 2 testable (b) pairs; bar testable_b ≥ 11 ∧ F_a ≥ 2; margin δ = 2.
- Codebook search: DSATUR random tie seeds 0–999, first C = 16; exact-search node budget 10⁷; anneal 200 000 iterations × restarts r = 0..7, rng `default_rng(20261001 + 1000·k + r)`, temperature linear 2.0 → 0.03, scalar J = 10⁶·dup + 10³·soft + logvar.
- Judgement set: `l_pairs.new_turns(…, 210, 20260927)` turns 64..209, E-grid key dedupe, stop at the 21st (b), `STOP_SET_SHORT` if not reached by turn 209.
- Windows: settle 800 ms + read 600 ms, sliding window 200 ms; drive measurement and E0 strength 0.35.
- Tests that need the connectome use `needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="no connectome")`.

## Review Focus

- Odour-pair duplicates across sets when two configs share a codebook: the judgement-set overlap test must run per config on glomerulus sets, not only on E-grid keys (Task 4).
- A stage run out of order (e.g. `judge` before `even`, or after a codebook digest changed): the runner must refuse with exit code 2 and write nothing (Task 7).
- Interrupted measurement: re-running a stage must reuse cached pairs and only compute the missing ones; partial pair sets must never be read (`NOT_READ`) (Tasks 6, 7).
- Smoke runs must never touch the judgement set or write the committed summary (Task 7).
- Strength values excluded by the ORN cap must be dropped before any engine call (Task 5 rule + Task 7 runner test).

---

## File Structure

| File | Responsibility |
|---|---|
| `flymon/agent/e_spec.py` | `ESpec` frozen dataclass: every number above; `smoke()` variant; `all_track_seeds()` |
| `flymon/agent/e_store.py` | path guard (`results/encoder/`, `results/summary/encoder_grid.json`), atomic JSON writes, summary blocks, `ECache` content-key cache |
| `flymon/agent/e_codebook.py` | cells, dual set D, hard-conflict graph, clique size, randomized DSATUR, constructive start, exact fallback, anneal, uniqueness, digest |
| `flymon/agent/encode_grid.py` | `odour()`, reachable situation odours (112), ORN-cap check, `GridEncoder` (battle interface) |
| `flymon/agent/e_pairs.py` | even-turn situations re-encoded (E0 digest check), judgement-set generation, E-grid keys, overlap checks, shared-odour count |
| `flymon/agent/e_rules.py` | strength choice, config selection, band reading, closing sentences, operating characteristics |
| `flymon/agent/e_measure.py` | `EMeasurer` (FlyPool + ECache): `drive`, `activity`, `oracle` |
| `flymon/agent/e_runner.py` | stages ⓪–⑥ on a measurer interface, chain checks, manifest |
| `scripts/run_encoder_grid.py` | CLI: builds pool/pops/C3/measurer, runs one stage |
| `tests/agent/test_e_*.py` | tests per module; `tests/agent/e_scripted.py` scripted measurer |

---

### Task 1: Spec constants, store, seed disjointness

**Files:**
- Create: `flymon/agent/e_spec.py`, `flymon/agent/e_store.py`
- Test: `tests/agent/test_e_spec_store.py`
- Setup (not committed — `data/*` is git-ignored): `ln -s /Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot/data/malecns.npz data/malecns.npz`; verify `uv run python -c "from flymon.brain.h3_store import sha256_file; from flymon.brain.h3_spec import SPEC; print(sha256_file('data/malecns.npz')==SPEC.connectome_sha256)"` prints `True`.

**Interfaces:**
- Produces: `ESpec` (fields listed in Step 3), `SPEC = ESpec()`, `smoke(spec) -> ESpec`, `track_seeds(spec) -> set[int]`; `e_store.guard(path, params_list)`, `write_json(path, obj, params_list) -> Path`, `write_summary_block(path, block, obj, params_list) -> Path`, `read_summary(path=SUMMARY) -> dict`, `class ECache(root: str, code: dict)` with `.key(kind, inputs) -> str`, `.get(kind, inputs) -> dict|None`, `.put(kind, inputs, result, params_list) -> None`, `.get_or_compute(kind, inputs, compute, params_list)`, counters `.hits`, `.misses`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_e_spec_store.py
import importlib, inspect, json, os
import pytest
from flymon.agent import e_spec, e_store
from flymon.agent.e_spec import SPEC

def _seed_ints(obj):
    out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int):
        out.add(obj)
    elif isinstance(obj, (tuple, list, range, set, frozenset)):
        for v in obj:
            out |= _seed_ints(v)
    return out

def _module_seeds(modname):
    m = importlib.import_module(modname)
    out = set()
    for name, cls in inspect.getmembers(m, inspect.isclass):
        if cls.__module__ != m.__name__ or not hasattr(cls, "__dataclass_fields__"):
            continue
        for f, fld in cls.__dataclass_fields__.items():
            if "seed" in f:
                default = fld.default if fld.default is not inspect._empty else None
                if default is None and fld.default_factory is not inspect._empty:  # pragma: no cover
                    default = fld.default_factory()
                out |= _seed_ints(default)
    for name, v in vars(m).items():
        if "SEED" in name.upper():
            out |= _seed_ints(v)
    return out

def test_track_seeds_disjoint_from_every_spec_module():
    import pkgutil, flymon.brain, flymon.rescope
    mods = [f"flymon.brain.{n}" for _, n, _ in pkgutil.iter_modules(flymon.brain.__path__) if n.endswith("_spec")]
    mods += ["flymon.rescope.spec", "flymon.rescope.blocks", "flymon.brain.d6a"]
    other = set().union(*(_module_seeds(m) for m in mods))
    other |= set(range(500, 616))
    mine = e_spec.track_seeds(SPEC)
    assert mine and not (mine & other)

def test_spec_numbers():
    assert SPEC.s_grid == (0.175, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
    assert SPEC.configs == ("k3-full", "k3-norm", "k2-full", "k2-norm")
    assert SPEC.drive_seeds == tuple(range(24_001_000, 24_001_008))
    assert SPEC.judge_act_seeds == tuple(range(24_100_000, 24_100_008))
    assert SPEC.even_report_seeds == tuple(range(608, 616))
    assert SPEC.bar_b == 11 and SPEC.f_a_min == 2 and SPEC.margin == 2 and SPEC.tie_pairs == 2

def test_guard_refuses_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        e_store.write_json("results/m0d/x.json", {"a": 1}, [])
    p = e_store.write_json("results/encoder/x.json", {"a": 1}, [])
    assert json.loads(p.read_text()) == {"a": 1}
    e_store.write_summary_block(e_store.SUMMARY, "set", {"n": 1}, [])
    e_store.write_summary_block(e_store.SUMMARY, "drive", {"m": 2}, [])
    assert e_store.read_summary() == {"set": {"n": 1}, "drive": {"m": 2}}

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = e_store.ECache("results/encoder/cache", {"key": "abc"})
    calls = []
    f = lambda: calls.append(1) or {"v": 3}
    assert c.get_or_compute("oracle", {"x": 1}, f, []) == {"v": 3}
    assert c.get_or_compute("oracle", {"x": 1}, f, []) == {"v": 3}
    assert calls == [1] and c.hits == 1 and c.misses == 1
    assert c.get("oracle", {"x": 2}) is None
```

- [ ] **Step 2: Run** `uv run pytest tests/agent/test_e_spec_store.py -q` — expect FAIL (`ModuleNotFoundError: flymon.agent.e_spec`).

- [ ] **Step 3: Implement**

```python
# flymon/agent/e_spec.py
"""Every number of the encoder-redesign track (docs/superpowers/specs/2026-10-01-encoder-redesign-design.md).
No other module restates one; smoke runs and tests use dataclasses.replace of SPEC."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ESpec:
    exclude: tuple = ("ORN_DA1", "ORN_V")
    k2_alphabet: int = 40
    band_lo: float = 400.0
    band_hi: float = 3200.0
    # codebook (3.3)
    dsatur_tie_seeds: int = 1000
    exact_node_budget: int = 10_000_000
    anneal_iters: int = 200_000
    anneal_restarts: int = 8
    anneal_seed0: int = 20261001
    t_start: float = 2.0
    t_end: float = 0.03
    w_dup: float = 1e6
    w_soft: float = 1e3
    # measurement windows (4.1, 4.4)
    drive_strength: float = 0.35
    e0_strength: float = 0.35
    settle_ms: float = 800.0
    read_ms: float = 600.0
    window_ms: int = 200
    drive_seeds: tuple = tuple(range(24_001_000, 24_001_008))
    strength_seeds: tuple = tuple(range(24_002_000, 24_002_008))
    smoke_seeds: tuple = tuple(range(24_009_000, 24_009_100))
    # strength calibration (4.3)
    s_grid: tuple = (0.175, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
    kc_target: float = 0.0554
    kc_band: tuple = (0.05, 0.09)
    tail_lo: float = 0.03
    tail_hi: float = 0.15
    tail_share_max: float = 0.10
    # grid and selection (4.2, 4.4)
    configs: tuple = ("k3-full", "k3-norm", "k2-full", "k2-norm")
    alphas: tuple = (0.2, 0.5, 0.8)
    even_act_seeds: tuple = tuple(range(500, 508))
    even_select_seeds: tuple = tuple(range(600, 608))
    even_report_seeds: tuple = tuple(range(608, 616))
    testable_min: float = 2.0
    naive_max: float = 0.5
    bar_b: int = 11
    f_a_min: int = 2
    tie_pairs: int = 2
    n_b: int = 21
    # judgement (5.1-5.5)
    l_rng_seed: int = 20260927
    l_total_turns: int = 210
    judge_first_turn: int = 64
    judge_last_turn: int = 209
    judge_act_seeds: tuple = tuple(range(24_100_000, 24_100_008))
    judge_select_seeds: tuple = tuple(range(24_100_100, 24_100_108))
    judge_report_seeds: tuple = tuple(range(24_100_200, 24_100_208))
    margin: int = 2
    oc_q: tuple = (0.33, 0.5, 0.6, 0.7)
    oc_c: tuple = (2, 4, 7)
    oc_c_delta: int = 2
    oc_naive_max: int = 12
    oc_n_a_rows: tuple = (18, 32)
    # paths
    raw_dir: str = "results/encoder"
    m0d_summary: str = "results/summary/m0d.json"

    @staticmethod
    def k_of(config: str) -> int:
        return int(config[1])

    @staticmethod
    def dual_rule(config: str) -> str:
        return config.split("-")[1]


SPEC = ESpec()


def smoke(spec: ESpec = SPEC) -> ESpec:
    """Two seeds per block from the smoke range, two strengths, few iterations; raw dir results/encoder/smoke."""
    s = spec.smoke_seeds
    return dataclasses.replace(spec, drive_seeds=s[0:2], strength_seeds=s[2:4], even_act_seeds=s[4:6],
                               even_select_seeds=s[6:8], even_report_seeds=s[8:10], s_grid=(0.35, 0.7),
                               anneal_iters=2_000, anneal_restarts=2, raw_dir="results/encoder/smoke")


def track_seeds(spec: ESpec = SPEC) -> set:
    return set(spec.drive_seeds) | set(spec.strength_seeds) | set(spec.smoke_seeds) | set(spec.judge_act_seeds) \
        | set(spec.judge_select_seeds) | set(spec.judge_report_seeds)
```

```python
# flymon/agent/e_store.py
"""The encoder track's only writer (spec 6): raw files under results/encoder/, the one summary
results/summary/encoder_grid.json, atomic writes, and a content-key cache (h3_store.MeasureCache's form with this
track's path guard — h3_store refuses paths outside results/m0d/)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from ..brain.h3_store import canonical, canonical_pretty
from ..brain.pool_bench import refuse_modified_engine_output, refuse_old_engine_output

ALLOWED_DIR = "results/encoder/"
SUMMARY = "results/summary/encoder_grid.json"


def guard(path, params_list) -> None:
    for p in params_list:
        refuse_old_engine_output(str(path), p.kc_kc_scale)
        refuse_modified_engine_output(str(path), p)
    rel = os.path.relpath(os.path.abspath(str(path)), os.getcwd()).replace(os.sep, "/")
    if not (rel.startswith(ALLOWED_DIR) or rel == SUMMARY):
        print(f"refusing to write {path}: the encoder track writes only under {ALLOWED_DIR} and {SUMMARY}",
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
    guard(path, params_list)
    doc = read_summary(path)
    doc[block] = obj
    return write_json(path, doc, params_list)


class ECache:
    def __init__(self, root: str, code: dict):
        self.root, self.code = str(root), dict(code)
        self.hits = self.misses = 0

    def key(self, kind: str, inputs) -> str:
        blob = json.dumps(canonical({"kind": kind, "inputs": inputs, "code": self.code}), sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, kind, inputs) -> Path:
        return Path(self.root) / kind / f"{self.key(kind, inputs)[:24]}.json"

    def get(self, kind, inputs):
        p = self._path(kind, inputs)
        return json.loads(p.read_text())["result"] if p.exists() else None

    def put(self, kind, inputs, result, params_list) -> None:
        write_json(self._path(kind, inputs), {"kind": kind, "result": result}, params_list)

    def get_or_compute(self, kind, inputs, compute, params_list):
        got = self.get(kind, inputs)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        res = compute()
        self.put(kind, inputs, res, params_list)
        return json.loads(canonical_pretty(res))
```

- [ ] **Step 4: Run** `uv run pytest tests/agent/test_e_spec_store.py -q` — expect PASS. If the disjointness test finds an overlap, report it (do not change the spec's seeds silently).

- [ ] **Step 5: Commit** `git add flymon/agent/e_spec.py flymon/agent/e_store.py tests/agent/test_e_spec_store.py && git commit -m "encoder track: spec constants, track store and cache, seed-disjointness test"`

---

### Task 2: Codebook construction

**Files:**
- Create: `flymon/agent/e_codebook.py`
- Test: `tests/agent/test_e_codebook.py`

**Interfaces:**
- Consumes: `ESpec`; `h4_pairs.pool_vocabulary()` → `(species_types, move_info, mon_types, move_types)`.
- Produces:
  - `cells(move_types, mon_types) -> list[tuple[str,str]]` (move-major order).
  - `dual_set(species_types) -> set[frozenset]` (10 in the pool).
  - `hard_conflict(a, b, duals) -> bool`; `conflict_graph(cells, duals) -> list[set[int]]`.
  - `max_clique_size(adj) -> int`.
  - `dsatur(adj, rng) -> list[int]` (colour per cell); `find_colouring(adj, target, seeds) -> (seed|None, colours|None)`.
  - `alphabet(drive: dict[str,float], k: int, spec) -> list[str]` (ordered: drive descending, then name).
  - `construct(colours, alpha, k) -> list[tuple[str,...]]` (codeword per cell; sorted tuple of glomerulus names).
  - `exact_search(adj, alpha, k, budget) -> list|None|"INFEASIBLE"`.
  - `hard_violations(codebook, adj) -> int`; `soft_violations(codebook, adj) -> int`; `dup_pairs(codebook) -> int`.
  - `anneal(start, adj, alpha, drive, k, spec) -> dict(codebook, J, restart, dup, soft, logvar)`.
  - `build(drive, k, spec) -> dict(status in {"OK","INFEASIBLE","UNDECIDED","NOT_UNIQUE"}, colouring, colour_seed, C, omega, codebook, anneal, digest)`; `digest(codebook) -> str`.
  - The 112-odour uniqueness check lives in Task 3 (`encode_grid.unique_odours`), and `build` calls it through an injected function parameter `unique_check(codebook) -> bool` (default: codewords distinct only) so this module stays pops-free.

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_e_codebook.py
import itertools
import numpy as np
import pytest
from flymon.agent import e_codebook as cb
from flymon.agent.e_spec import SPEC, smoke
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, MON, MOVE = pool_vocabulary()
CELLS = cb.cells(MOVE, MON)
DUALS = cb.dual_set(ST)
ADJ = cb.conflict_graph(CELLS, DUALS)
GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
DRIVE = {g: float(10 + 50 * i) for i, g in enumerate(GLOMS)}

def test_cells_and_duals():
    assert len(CELLS) == 96 and CELLS[0] == (MOVE[0], MON[0])
    assert len(DUALS) == 10 and frozenset({"GRASS", "POISON"}) in DUALS

def test_hard_conflict_rules():
    m1, m2 = MOVE[0], MOVE[1]
    assert cb.hard_conflict((m1, "FIRE"), (m1, "ICE"), DUALS)            # same row
    assert cb.hard_conflict((m1, "FIRE"), (m2, "FIRE"), DUALS)           # same column
    assert cb.hard_conflict((m1, "GRASS"), (m2, "POISON"), DUALS)        # dual cross term
    assert not cb.hard_conflict((m1, "FIRE"), (m2, "ICE"), DUALS)

def test_clique_is_16():
    assert cb.max_clique_size(ADJ) == 16

def test_dsatur_valid_and_colouring_found():
    seed, col = cb.find_colouring(ADJ, 16, range(SPEC.dsatur_tie_seeds))
    assert seed is not None and max(col) + 1 == 16
    assert all(col[i] != col[j] for i in range(96) for j in ADJ[i])

def test_construct_has_no_hard_violation():
    _, col = cb.find_colouring(ADJ, 16, range(1000))
    alpha = cb.alphabet(DRIVE, 3, SPEC)
    book = cb.construct(col, alpha, 3)
    assert cb.hard_violations(book, ADJ) == 0 and all(len(w) == 3 for w in book)

def test_alphabet_k2_band_and_order():
    drive = {**DRIVE, "ORN_DA1": 1131.0, "ORN_V": 1131.0}
    a2 = cb.alphabet(drive, 2, SPEC)
    assert len(a2) == 40 and "ORN_DA1" not in a2 and "ORN_V" not in a2
    target = np.log10(np.sqrt(400 * 3200) + 1)
    far = max(abs(np.log10(drive[g] + 1) - target) for g in a2)
    out = [g for g in GLOMS if g not in a2]
    assert all(abs(np.log10(drive[g] + 1) - target) >= far for g in out)
    a3 = cb.alphabet(drive, 3, SPEC)
    assert len(a3) == 49 and a3 == sorted(a3, key=lambda g: (-drive[g], g))

def test_anneal_keeps_hard_zero_and_removes_duplicates():
    sp = smoke(SPEC)
    res = cb.build(DRIVE, 2, sp)
    assert res["status"] == "OK"
    book = [tuple(w) for w in res["codebook"]]
    assert cb.hard_violations(book, ADJ) == 0
    assert cb.dup_pairs(book) == 0 and len(set(book)) == 96

def test_build_is_deterministic():
    sp = smoke(SPEC)
    assert cb.build(DRIVE, 2, sp)["digest"] == cb.build(DRIVE, 2, sp)["digest"]

def test_infeasible_when_alphabet_too_small():
    small = {g: DRIVE[g] for g in GLOMS[:40]}
    res = cb.build(small, 3, smoke(SPEC))         # omega*k = 48 > 40
    assert res["status"] == "INFEASIBLE"

def test_not_unique_status():
    res = cb.build(DRIVE, 2, smoke(SPEC), unique_check=lambda book: False)
    assert res["status"] == "NOT_UNIQUE"
```

- [ ] **Step 2: Run** `uv run pytest tests/agent/test_e_codebook.py -q` — expect FAIL (module missing).

- [ ] **Step 3: Implement** `flymon/agent/e_codebook.py`:

```python
"""Spec 3.1-3.3: the 96-cell glomerulus-disjoint conjunctive codebook. Pure (no engine, no pops)."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def cells(move_types, mon_types) -> list:
    return [(m, t) for m in move_types for t in mon_types]


def dual_set(species_types) -> set:
    return {frozenset(v) for v in species_types.values() if len(v) == 2}


def hard_conflict(a, b, duals) -> bool:
    (m1, t1), (m2, t2) = a, b
    return m1 == m2 or t1 == t2 or frozenset((t1, t2)) in duals


def conflict_graph(cs, duals) -> list:
    adj = [set() for _ in cs]
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if hard_conflict(cs[i], cs[j], duals):
                adj[i].add(j); adj[j].add(i)
    return adj


def max_clique_size(adj) -> int:
    """Bron-Kerbosch with pivoting (96 vertices: fast enough)."""
    best = 0

    def bk(r, p, x):
        nonlocal best
        if not p and not x:
            best = max(best, r); return
        if r + len(p) <= best:
            return
        u = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[u]):
            bk(r + 1, p & adj[v], x & adj[v]); p = p - {v}; x = x | {v}

    bk(0, set(range(len(adj))), set())
    return best


def dsatur(adj, rng) -> list:
    n = len(adj); col = [-1] * n; tie = rng.permutation(n)
    for _ in range(n):
        cand = [v for v in range(n) if col[v] < 0]
        sat = {v: len({col[u] for u in adj[v] if col[u] >= 0}) for v in cand}
        v = max(cand, key=lambda v: (sat[v], len(adj[v]), -int(np.where(tie == v)[0][0])))
        used = {col[u] for u in adj[v] if col[u] >= 0}
        col[v] = next(c for c in range(n) if c not in used)
    return col


def find_colouring(adj, target: int, seeds):
    for s in seeds:
        col = dsatur(adj, np.random.default_rng(int(s)))
        if max(col) + 1 <= target:
            return int(s), col
    return None, None


def _lg(x) -> float:
    return math.log10(float(x) + 1.0)


def alphabet(drive: dict, k: int, spec) -> list:
    gl = [g for g in drive if g not in spec.exclude]
    order = lambda gs: sorted(gs, key=lambda g: (-drive[g], g))
    if k == 3:
        return order(gl)
    target = _lg(math.sqrt(spec.band_lo * spec.band_hi))
    near = sorted(gl, key=lambda g: (abs(_lg(drive[g]) - target), g))[:spec.k2_alphabet]
    return order(near)


def construct(colours, alpha, k) -> list:
    return [tuple(sorted(alpha[c * k + j] for j in range(k))) for c in colours]


def hard_violations(book, adj) -> int:
    return sum(len(set(book[i]) & set(book[j])) for i in range(len(book)) for j in adj[i] if j > i)


def soft_violations(book, adj) -> int:
    n = len(book)
    return sum(max(0, len(set(book[i]) & set(book[j])) - 1) for i in range(n) for j in range(i + 1, n)
               if j not in adj[i])


def dup_pairs(book) -> int:
    from collections import Counter
    return sum(c * (c - 1) // 2 for c in Counter(tuple(sorted(w)) for w in book).values())


def logvar(book, drive) -> float:
    return float(np.var([_lg(sum(drive[g] for g in w)) for w in book]))


def objective(book, adj, drive, spec) -> tuple:
    d, s, v = dup_pairs(book), soft_violations(book, adj), logvar(book, drive)
    return spec.w_dup * d + spec.w_soft * s + v, d, s, v


def exact_search(adj, alpha, k, budget):
    """Backtracking over cells in DSATUR-like order, each cell picking a k-subset disjoint from its conflicting
    neighbours' glomeruli. Returns a codebook, None (budget spent) — infeasibility is proven only by the clique bound."""
    import itertools
    n = len(adj); book = [None] * n; nodes = 0
    order = sorted(range(n), key=lambda v: -len(adj[v]))

    def rec(i):
        nonlocal nodes
        if i == n:
            return True
        v = order[i]
        banned = set().union(*(set(book[u]) for u in adj[v] if book[u] is not None)) if adj[v] else set()
        for combo in itertools.combinations([g for g in alpha if g not in banned], k):
            nodes += 1
            if nodes > budget:
                raise TimeoutError
            book[v] = tuple(sorted(combo))
            if rec(i + 1):
                return True
            book[v] = None
        return False

    try:
        return list(book) if rec(0) else "INFEASIBLE"
    except TimeoutError:
        return None


def anneal(start, adj, alpha, drive, k, spec) -> dict:
    """Moves swap one glomerulus of one cell for an alphabet glomerulus not in it; moves creating a hard violation
    are rejected. Full objective recomputed per move only on the touched cell's terms (incremental)."""
    best = None
    for r in range(spec.anneal_restarts):
        rng = np.random.default_rng(spec.anneal_seed0 + 1000 * k + r)
        book = [list(w) for w in start]
        J, *_ = objective([tuple(w) for w in book], adj, drive, spec)
        n = len(book)
        for it in range(spec.anneal_iters):
            T = spec.t_start + (spec.t_end - spec.t_start) * it / max(1, spec.anneal_iters - 1)
            i = int(rng.integers(n)); old = book[i][:]
            out_g = old[int(rng.integers(k))]
            choices = [g for g in alpha if g not in old]
            in_g = choices[int(rng.integers(len(choices)))]
            new = sorted([g for g in old if g != out_g] + [in_g])
            if any(in_g in book[j] for j in adj[i]):
                continue
            book[i] = new
            J2, *_ = objective([tuple(w) for w in book], adj, drive, spec)
            if J2 <= J or rng.random() < math.exp(-(J2 - J) / T):
                J = J2
            else:
                book[i] = old
        J, d, s, v = objective([tuple(w) for w in book], adj, drive, spec)
        if best is None or J < best["J"]:
            best = dict(codebook=[tuple(w) for w in book], J=J, restart=r, dup=d, soft=s, logvar=v)
    return best


def digest(book) -> str:
    return hashlib.sha256(json.dumps([list(w) for w in book], separators=(",", ":")).encode()).hexdigest()


def build(drive: dict, k: int, spec, unique_check=None) -> dict:
    from ..brain.h4_pairs import pool_vocabulary
    st, _, mon, move = pool_vocabulary()
    cs = cells(move, mon); adj = conflict_graph(cs, dual_set(st))
    omega = max_clique_size(adj); alpha = alphabet(drive, k, spec)
    out = dict(k=k, omega=omega, alphabet=alpha, colour_seed=None, colouring=None, C=None)
    if omega * k > len(alpha):
        return dict(out, status="INFEASIBLE", codebook=None, anneal=None, digest=None)
    seed, col = find_colouring(adj, omega, range(spec.dsatur_tie_seeds))      # spec 3.3: first C = omega
    if col is not None:
        start = construct(col, alpha, k)
        out.update(colour_seed=seed, colouring=col, C=max(col) + 1)
    else:
        start = exact_search(adj, alpha, k, spec.exact_node_budget)
        if start == "INFEASIBLE":
            return dict(out, status="INFEASIBLE", codebook=None, anneal=None, digest=None)
        if start is None:
            return dict(out, status="UNDECIDED", codebook=None, anneal=None, digest=None)
    an = anneal(start, adj, alpha, drive, k, spec)
    book = an["codebook"]
    assert hard_violations(book, adj) == 0
    uniq = (unique_check or (lambda b: dup_pairs(b) == 0))(book)
    status = "OK" if (dup_pairs(book) == 0 and uniq) else "NOT_UNIQUE"
    return dict(out, status=status, codebook=[list(w) for w in book], anneal={kk: an[kk] for kk in
                ("J", "restart", "dup", "soft", "logvar")}, digest=digest(book))
```

Performance: `objective` recomputes everything per move (O(96²)); if `anneal` with 200 000 iterations exceeds ~10 minutes per k in a quick timing run, replace it with incremental deltas over the touched cell's terms (same accept rule, same RNG call order) and add a test that the incremental J equals the full `objective` after 1 000 random moves.

- [ ] **Step 4: Run** `uv run pytest tests/agent/test_e_codebook.py -q` — expect PASS. Time one full-size `anneal` (k = 3, `SPEC`, synthetic drive) and report the wall time in the task report.

- [ ] **Step 5: Commit** `git commit -m "encoder track: E-grid codebook — conflict graph, committed-colouring DSATUR, constructive start, exact fallback, anneal with uniqueness"`

---

### Task 3: Odours, reachable set, ORN cap, battle encoder

**Files:**
- Create: `flymon/agent/encode_grid.py`
- Test: `tests/agent/test_encode_grid.py`

**Interfaces:**
- Consumes: codebook as `{"cells": [[move, type], ...], "codebook": [[glom, ...], ...]}` (Task 2 `build` output plus `cells`).
- Produces:
  - `Codebook(cells, words)` with `.word(move_type, opp_type) -> tuple`.
  - `glomeruli(cb, move_type, opp_types) -> tuple[str,...]`.
  - `odour(receptor_counts: dict[str,int], cb, move_type, opp_types, dual_rule: str) -> dict[str,float]`: strengths ∝ 1/receptor count, mean 1; `norm` multiplies by `k / len(glomeruli)` when the opponent is dual-typed.
  - `reachable(species_types, move_types) -> list[(move_type, tuple(opp_types))]` (112; single-type opponent sets from the pool + the 10 dual sets, each × 8 moves).
  - `unique_odours(cb, species_types, move_types) -> bool`.
  - `cap_ok(receptor_counts, cb, species_types, move_types, dual_rule, s, max_rate_hz, cap_hz) -> bool`.
  - `class GridEncoder(pops, cb, dual_rule)` with `odour(battle, move) -> dict` and `situation_key(battle, cands) -> tuple` (same shape as `encode.Encoder.situation_key`).

- [ ] **Step 1: Failing tests**

```python
# tests/agent/test_encode_grid.py
import numpy as np
import pytest
from flymon.agent import encode_grid as eg, e_codebook as cb
from flymon.agent.e_spec import SPEC, smoke
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, MON, MOVE = pool_vocabulary()
GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
RC = {g: 1 + (i % 5) for i, g in enumerate(GLOMS)}
DRIVE = {g: float(10 + 50 * i) for i, g in enumerate(GLOMS)}
BUILT = cb.build(DRIVE, 2, smoke(SPEC))
CB = eg.Codebook(cb.cells(MOVE, MON), BUILT["codebook"])

def test_reachable_is_112():
    r = eg.reachable(ST, MOVE)
    assert len(r) == 112 and len(set(r)) == 112
    singles = {o for _, o in r if len(o) == 1}
    assert singles == {("FIGHTING",), ("NORMAL",), ("PSYCHIC",), ("WATER",)}

def test_odour_full_and_norm():
    single = eg.odour(RC, CB, "WATER", ("NORMAL",), "full")
    assert len(single) == 2 and np.isclose(np.mean(list(single.values())), 1.0)
    dual_f = eg.odour(RC, CB, "WATER", ("GRASS", "POISON"), "full")
    dual_n = eg.odour(RC, CB, "WATER", ("GRASS", "POISON"), "norm")
    assert len(dual_f) == 4 and np.isclose(np.mean(list(dual_f.values())), 1.0)
    assert all(np.isclose(dual_n[g], 0.5 * dual_f[g]) for g in dual_f)
    assert eg.odour(RC, CB, "WATER", ("NORMAL",), "norm") == single
    inv = {g: 1 / RC[g] for g in single}; m = np.mean(list(inv.values()))
    assert all(np.isclose(single[g], inv[g] / m) for g in single)

def test_axis_pairs_share_no_glomerulus():
    a = set(eg.glomeruli(CB, "WATER", ("GRASS", "POISON")))
    b = set(eg.glomeruli(CB, "WATER", ("FIRE", "FLYING")))
    c = set(eg.glomeruli(CB, "GROUND", ("GRASS", "POISON")))
    assert not (a & b) and not (a & c)

def test_unique_odours():
    assert eg.unique_odours(CB, ST, MOVE)

def test_cap_ok():
    assert eg.cap_ok(RC, CB, ST, MOVE, "full", 0.35, 200.0, 333.3)
    assert not eg.cap_ok(RC, CB, ST, MOVE, "full", 2.0, 200.0, 333.3)

def test_grid_encoder_matches_odour():
    class Mon:
        def __init__(self, species, hp=1.0): self.species, self.current_hp_fraction = species, hp
    class Mv:
        def __init__(self, id): self.id = id
    class B:
        active_pokemon = Mon("Blastoise"); opponent_active_pokemon = Mon("Venusaur")
    class P:
        receptor_types = {g: list(range(RC[g])) for g in GLOMS}
    enc = eg.GridEncoder(P(), CB, "norm")
    assert enc.odour(B(), Mv("surf")) == eg.odour(RC, CB, "WATER", ("GRASS", "POISON"), "norm")
```

- [ ] **Step 2: Run** — expect FAIL.

- [ ] **Step 3: Implement**

```python
# flymon/agent/encode_grid.py
"""Spec 3.4: E-grid situation odours. odour(move type, opponent types) = union of the cells' codewords; strengths
proportional to 1/receptor count with mean 1 (the E0 convention, h4_pairs.odour); `norm` halves a dual-typed odour
(k / |glomeruli|). My type, HP and power are not encoded."""
from __future__ import annotations

import numpy as np
from poke_env.data.normalize import to_id_str

from ..battle.pool import POOL


class Codebook:
    def __init__(self, cells, words):
        self.cells = [tuple(c) for c in cells]
        self.words = [tuple(w) for w in words]
        self._ix = {c: i for i, c in enumerate(self.cells)}
        self.k = len(self.words[0])

    def word(self, move_type: str, opp_type: str) -> tuple:
        return self.words[self._ix[(move_type, opp_type)]]


def glomeruli(cb: Codebook, move_type: str, opp_types) -> tuple:
    out = [g for t in opp_types for g in cb.word(move_type, t)]
    if len(set(out)) != len(out):
        raise ValueError(f"glomerulus collision for {move_type} vs {opp_types}")
    return tuple(out)


def odour(receptor_counts: dict, cb: Codebook, move_type: str, opp_types, dual_rule: str) -> dict:
    gl = glomeruli(cb, move_type, opp_types)
    inv = np.array([1.0 / receptor_counts[g] for g in gl]); inv /= inv.mean()
    if dual_rule == "norm" and len(opp_types) == 2:
        inv *= cb.k / len(gl)
    elif dual_rule not in ("full", "norm"):
        raise ValueError(dual_rule)
    return {g: float(s) for g, s in zip(gl, inv)}


def reachable(species_types, move_types) -> list:
    opp_sets = sorted({tuple(sorted(v)) for v in species_types.values()})
    return [(m, o) for o in opp_sets for m in move_types]


def unique_odours(cb: Codebook, species_types, move_types) -> bool:
    sets = [frozenset(glomeruli(cb, m, o)) for m, o in reachable(species_types, move_types)]
    return len(set(sets)) == len(sets)


def cap_ok(receptor_counts, cb, species_types, move_types, dual_rule, s, max_rate_hz, cap_hz) -> bool:
    return all(max_rate_hz * s * v <= cap_hz
               for m, o in reachable(species_types, move_types)
               for v in odour(receptor_counts, cb, m, o, dual_rule).values())


class GridEncoder:
    """encode.Encoder's interface on E-grid odours."""

    def __init__(self, pops, cb: Codebook, dual_rule: str):
        from ..brain.h4_pairs import pool_vocabulary
        self.rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
        self.cb, self.rule = cb, dual_rule
        self.species_types, self.move_info, _, _ = pool_vocabulary()
        self.species_by_id = {to_id_str(m.species): m.species for m in POOL}
        self.move_by_id = {to_id_str(a): a for m in POOL for a in m.attacks}

    def _species(self, mon) -> str:
        return self.species_by_id[to_id_str(mon.species)]

    def odour(self, battle, move) -> dict:
        opp = self._species(battle.opponent_active_pokemon)
        mtype = self.move_info[self.move_by_id[move.id]][0]
        return odour(self.rc, self.cb, mtype, tuple(sorted(self.species_types[opp])), self.rule)

    def situation_key(self, battle, cands) -> tuple:
        return (self._species(battle.active_pokemon), self._species(battle.opponent_active_pokemon),
                tuple(sorted(m.id for m in cands)))
```

Note: `reachable` sorts types inside each opponent set; `glomeruli` order follows that sorted tuple. Everywhere odours are built for pairs (Task 4) pass `tuple(sorted(opp_types))` so the same situation always yields the identical dict.

- [ ] **Step 4: Run** `uv run pytest tests/agent/test_encode_grid.py -q` — PASS. In `e_codebook.build` callers (Task 7) pass `unique_check=lambda book: unique_odours(Codebook(cells, book), ST, MOVE)`.

- [ ] **Step 5: Commit** `git commit -m "encoder track: E-grid odours, 112 reachable situations, ORN cap check, GridEncoder"`

---

### Task 4: Pair sets — even selection set and the judgement set

**Files:**
- Create: `flymon/agent/e_pairs.py`
- Test: `tests/agent/test_e_pairs.py`

**Interfaces:**
- Consumes: `h4_pairs` (`pool_vocabulary`, `build_turns`, `alternate_opponent`, `e0_channels`, `odour`, `pairs_digest`), `h4_spec.SPEC.pairs_digest`, `l_pairs` (`new_turns`, `alternate_from`, `_turn_pairs`), Task 3 `odour`, `Codebook`.
- Produces:
  - Situation row: `{"axis", "turn", "x", "y", "move_x", "move_y", "opp_x": tuple, "opp_y": tuple, "odor_x_e0", "odor_y_e0"}` (move = move type; opp = sorted opponent types).
  - `even_situations(pops) -> list[row]` — same order as `h4_pairs.even_pairs`; raises `ValueError` unless `pairs_digest` of the E0 odours equals `H4Spec.pairs_digest`.
  - `egrid_key(row) -> frozenset` — `{(move_x, opp_x), (move_y, opp_y)}`.
  - `judgement_set(pops, spec) -> dict(b, a, n_a, last_turn, skipped, status in {"OK","STOP_SET_SHORT"}, digest_e0_b, digest_e0_a, digest_keys)`.
  - `attach_odours(rows, rc, cb, dual_rule) -> list[row + odor_x, odor_y]`.
  - `overlap_report(judge_rows, used_rows, rc, cb, dual_rule) -> dict(cross=int, within=int)` — on glomerulus-set level for one config.
  - `used_situations(pops) -> list[row]` — all (a)/(b) rows of `build_turns` 16 turns and L turns 0–63 (every turn with (a)).
  - `shared_odour_count(judge_b, even_rows) -> int` — judgement (b) pairs with at least one odour key `(move, opp)` equal to an odour key in the even pairs.

- [ ] **Step 1: Failing tests** (all `needs_npz` except key logic)

```python
# tests/agent/test_e_pairs.py
from pathlib import Path
import pytest
from flymon.agent import e_pairs as ep, e_codebook as cb, encode_grid as eg
from flymon.agent.e_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="no connectome")

@pytest.fixture(scope="module")
def pops():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    return Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))

@needs_npz
def test_even_situations_reproduce_h4_digest(pops):
    rows = ep.even_situations(pops)
    assert sum(r["axis"] == "a" for r in rows) == 18 and sum(r["axis"] == "b" for r in rows) == 21

@needs_npz
def test_judgement_set_rules(pops):
    s = ep.judgement_set(pops, SPEC)
    assert s["status"] == "OK" and len(s["b"]) == 21
    assert all(SPEC.judge_first_turn <= r["turn"] <= s["last_turn"] <= SPEC.judge_last_turn for r in s["b"] + s["a"])
    used = {ep.egrid_key(r) for r in ep.used_situations(pops)}
    keys = [ep.egrid_key(r) for r in s["b"] + s["a"]]
    assert not (set(keys) & used) and len(set(keys)) == len(keys)
    assert s["n_a"] == len(s["a"])

@needs_npz
def test_overlap_zero_on_glomerulus_sets(pops):
    s = ep.judgement_set(pops, SPEC)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    gl = [g for g in rc if g not in SPEC.exclude]
    drive = {g: float(i) for i, g in enumerate(gl)}
    from flymon.brain.h4_pairs import pool_vocabulary
    st, _, mon, move = pool_vocabulary()
    book = cb.build(drive, 2, smoke(SPEC))["codebook"]
    C = eg.Codebook(cb.cells(move, mon), book)
    rep = ep.overlap_report(s["b"] + s["a"], ep.used_situations(pops), rc, C, "full")
    assert rep == {"cross": 0, "within": 0}
    for r in ep.attach_odours(s["b"] + s["a"], rc, C, "full"):
        assert not (set(r["odor_x"]) & set(r["odor_y"]))

def test_egrid_key_unordered():
    r1 = dict(move_x="WATER", opp_x=("NORMAL",), move_y="WATER", opp_y=("GROUND", "ROCK"))
    r2 = dict(move_x="WATER", opp_x=("GROUND", "ROCK"), move_y="WATER", opp_y=("NORMAL",))
    assert ep.egrid_key(r1) == ep.egrid_key(r2)
```

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement** `flymon/agent/e_pairs.py`. Required behaviour (write the code exactly to this):

```python
"""Spec 4.4 and 5.1: the even selection situations (H.4's list, re-encoded) and the judgement set from the L
generator's turns 64-209, deduplicated by E-grid key against every earlier pair and within itself."""
from __future__ import annotations

import hashlib
import itertools
import json

from ..brain import h4_pairs, l_pairs
from ..brain.h4_spec import SPEC as H4
from .encode_grid import glomeruli, odour


def _rows_for_turn(pops, chan, t, alt, species_types, with_a: bool) -> list:
    """l_pairs._turn_pairs' rows (same labels, same order) with the E-grid situation fields added."""
    e0 = l_pairs._turn_pairs(pops, chan, t, alt, species_types, with_a)
    cands = t["candidates"]; opp = tuple(sorted(t["opp_types"])); alt_t = tuple(sorted(species_types[alt]))
    meta = []
    if with_a:
        for i, j in itertools.combinations(range(len(cands)), 2):
            meta.append((cands[i]["type"], opp, cands[j]["type"], opp))
    for c in cands:
        meta.append((c["type"], opp, c["type"], alt_t))
    assert len(meta) == len(e0)
    return [dict(axis=r["axis"], turn=int(r["turn"]), x=r["x"], y=r["y"], move_x=mx, opp_x=ox, move_y=my, opp_y=oy,
                 odor_x_e0=r["odor_x"], odor_y_e0=r["odor_y"]) for r, (mx, ox, my, oy) in zip(e0, meta)]


def _e0_digest(rows) -> str:
    return h4_pairs.pairs_digest([dict(axis=r["axis"], turn=r["turn"], x=r["x"], y=r["y"], odor_x=r["odor_x_e0"],
                                       odor_y=r["odor_y_e0"]) for r in rows])


def even_situations(pops) -> list:
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    out = []
    for t in h4_pairs.build_turns(st, mi, 16):
        if t["turn"] % 2:
            continue
        alt = h4_pairs.alternate_opponent(t["turn"], t["me"], t["opp_types"], st)
        out += _rows_for_turn(pops, chan, t, alt, st, with_a=True)
    if _e0_digest(out) != H4.pairs_digest:
        raise ValueError("even situations do not reproduce H4Spec.pairs_digest")
    return out
```

and in the same file: `egrid_key(r) = frozenset({(r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"]))})`;
`used_situations(pops)` = `_rows_for_turn` over all 16 `build_turns` turns (alternate via `h4_pairs.alternate_opponent`) plus `l_pairs.new_turns(st, mi, 64, spec.l_rng_seed)` turns (alternate via `l_pairs.alternate_from`), all `with_a=True`;
`judgement_set(pops, spec)`: iterate `l_pairs.new_turns(st, mi, spec.l_total_turns, spec.l_rng_seed)[spec.judge_first_turn:spec.judge_last_turn + 1]`; per turn build rows with `with_a=True`; for each row in order, skip (record `{"key": [axis, turn, x, y], "reason": "used"|"in_set"}`) if its key is in the used set or already taken; keep (b) rows until 21 are kept — when the 21st (b) is kept, stop after finishing nothing more of that turn (later (b) rows of that turn are dropped and the turn's later (a) rows too — record them as `"after_21st"`); (a) rows of earlier positions in that turn are kept. If the turns run out, `status = "STOP_SET_SHORT"`. Return digests: `digest_e0_b = _e0_digest(b)`, `digest_e0_a = _e0_digest(a)`, `digest_keys = sha256(json of [[axis, turn, sorted key items as lists]])`;
`attach_odours(rows, rc, cb, rule)` adds `odor_x = odour(rc, cb, move_x, opp_x, rule)`, `odor_y` likewise;
`overlap_report(judge, used, rc, cb, rule)`: glomerulus-set pair key `frozenset({frozenset(glomeruli(cb, mx, ox)), frozenset(glomeruli(cb, my, oy))})`; `cross` = judge pairs whose set key is in the used rows' set keys; `within` = duplicates among judge set keys;
`shared_odour_count(judge_b, even_rows)` per the interface.

- [ ] **Step 4: Run** `uv run pytest tests/agent/test_e_pairs.py -q` — PASS. Print in the report: `len(b)`, `n_a`, `last_turn` (spec scratch expectation: 21, 32, 103) and `shared_odour_count` (scratch 10).

- [ ] **Step 5: Commit** `git commit -m "encoder track: even situations and the turn-64+ judgement set with E-grid-key dedupe and glomerulus-level overlap checks"`

---

### Task 5: Rules — strength, selection, bands, sentences, operating characteristics

**Files:**
- Create: `flymon/agent/e_rules.py`
- Test: `tests/agent/test_e_rules.py`

**Interfaces:**
- Produces:
  - Constants `SELECTED="SELECTED"`, `B_TB="B_Tb"`, `B_FA="B_Fa"`, `B_NC="B_결론없음"`, `NOT_READ="NOT_READ"`, `STOP_NO_ELIGIBLE`, `STOP_EVEN_LOW`, `STOP_SET_SHORT`, and reason codes `E0_ABOVE_BAR`, `BELOW_BAR`, `MARGIN`.
  - `odour_activity(per_seed: dict[odour_id, list[float]]) -> dict[odour_id, float]` (median over seeds).
  - `strength_ok(single: list[float], dual: list[float], spec) -> dict(ok, median, lo_single, lo_dual, hi_single, hi_dual)` — on per-odour activities; `ok` iff the median of all in band (inclusive) and each of the four tail shares ≤ `tail_share_max`.
  - `choose_strength(table: dict[s, dict(cap_ok: bool, single: list|None, dual: list|None)], spec) -> dict(s|None, reasons)`; `cap_ok=False` → not eligible; then `strength_ok`; pick closest median to `kc_target`, ties smaller s.
  - `select_config(results: dict[config, dict(eligible: bool, testable_b: int, F_a: int)], spec) -> dict(outcome, winner, passing, near)`.
  - `read_band(n, c, f_a, n_b, n_a, n_a_declared, naive_a, spec) -> dict(band, reason)`.
  - `sentence(outcome: str, fields: dict) -> str` (the spec 5.4 Korean sentences with fields filled; missing field → `KeyError`).
  - `operating_characteristics(n_a: int, spec) -> dict` rows over q × c × naive_a, c ± δ rows, n_a 18/32 rows.

- [ ] **Step 1: Failing tests**

```python
# tests/agent/test_e_rules.py
import itertools
import pytest
from flymon.agent import e_rules as R
from flymon.agent.e_spec import SPEC

def test_band_order_and_full_cover():
    n_a = 32
    for n, c in itertools.product(range(22), range(22)):
        for fa in range(n_a + 1):
            b = R.read_band(n, c, fa, 21, n_a, n_a, naive_a=fa, spec=SPEC)
            assert b["band"] in (R.SELECTED, R.B_TB, R.B_FA, R.B_NC)
            if c >= 11:
                assert b == {"band": R.B_NC, "reason": R.E0_ABOVE_BAR}
            elif n <= c:
                assert b["band"] == R.B_TB
            elif n < 11:
                assert b == {"band": R.B_NC, "reason": R.BELOW_BAR}
            elif n - c < 2:
                assert b == {"band": R.B_NC, "reason": R.MARGIN}
            elif fa < 2:
                assert b["band"] == R.B_FA
            else:
                assert b["band"] == R.SELECTED

@pytest.mark.parametrize("n,c,fa,band", [(11, 10, 2, R.B_NC), (12, 10, 2, R.SELECTED), (11, 11, 5, R.B_NC),
                                         (9, 9, 5, R.B_TB), (10, 2, 5, R.B_NC), (11, 2, 1, R.B_FA)])
def test_band_edges(n, c, fa, band):
    assert R.read_band(n, c, fa, 21, 32, 32, fa, SPEC)["band"] == band

def test_not_read_on_counts():
    assert R.read_band(15, 2, 5, 20, 32, 32, 5, SPEC)["band"] == R.NOT_READ
    assert R.read_band(15, 2, 5, 21, 31, 32, 5, SPEC)["band"] == R.NOT_READ

def test_select_pass_first():
    res = {"k3-full": dict(eligible=True, testable_b=10, F_a=3), "k3-norm": dict(eligible=True, testable_b=9, F_a=3),
           "k2-full": dict(eligible=False, testable_b=0, F_a=0), "k2-norm": dict(eligible=True, testable_b=12, F_a=2)}
    assert R.select_config(res, SPEC)["winner"] == "k2-norm"
    res["k3-norm"] = dict(eligible=True, testable_b=11, F_a=2)
    assert R.select_config(res, SPEC)["winner"] == "k3-norm"          # within 2 of 12, earlier in order
    for v in res.values(): v["testable_b"] = 10
    assert R.select_config(res, SPEC)["outcome"] == R.STOP_EVEN_LOW
    assert R.select_config({k: dict(eligible=False, testable_b=0, F_a=0) for k in SPEC.configs},
                           SPEC)["outcome"] == R.STOP_NO_ELIGIBLE

def test_strength_eligible_first():
    good = dict(cap_ok=True, single=[0.06] * 32, dual=[0.06] * 80)
    near_bad = dict(cap_ok=True, single=[0.02] * 32, dual=[0.0554] * 80)
    capped = dict(cap_ok=False, single=None, dual=None)
    out = R.choose_strength({0.35: near_bad, 0.5: good, 2.0: capped}, SPEC)
    assert out["s"] == 0.5
    assert R.choose_strength({0.35: near_bad}, SPEC)["s"] is None

def test_strength_tails_per_type():
    single = [0.06] * 28 + [0.02] * 4        # 12.5 % low -> fails
    assert not R.strength_ok(single, [0.06] * 80, SPEC)["ok"]
    single = [0.06] * 29 + [0.02] * 3        # 9.4 %
    assert R.strength_ok(single, [0.06] * 80, SPEC)["ok"]
    assert not R.strength_ok([0.06] * 32, [0.06] * 71 + [0.2] * 9, SPEC)["ok"]

def test_sentences_fill():
    s = R.sentence(R.STOP_EVEN_LOW, dict(table="k3-full 9/21·F_a 2"))
    assert "k3-full 9/21" in s
    with pytest.raises(KeyError):
        R.sentence(R.SELECTED, {})

def test_oc_rows_sum_to_one():
    oc = R.operating_characteristics(32, SPEC)
    for row in oc["rows"]:
        assert abs(sum(row["P"].values()) - 1.0) < 1e-9
    assert {r["n_a"] for r in oc["rows"]} >= {18, 32}
```

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement** `flymon/agent/e_rules.py`:

```python
"""Spec 4.3, 4.4, 5.3-5.5: the encoder track's decisions. The judgement code is the authoritative source."""
from __future__ import annotations

from math import comb
from statistics import median

SELECTED, B_TB, B_FA, B_NC, NOT_READ = "SELECTED", "B_Tb", "B_Fa", "B_결론없음", "NOT_READ"
STOP_NO_ELIGIBLE, STOP_EVEN_LOW, STOP_SET_SHORT = "STOP_NO_ELIGIBLE", "STOP_EVEN_LOW", "STOP_SET_SHORT"
E0_ABOVE_BAR, BELOW_BAR, MARGIN = "E0_ABOVE_BAR", "BELOW_BAR", "MARGIN"


def odour_activity(per_seed: dict) -> dict:
    return {k: float(median(v)) for k, v in per_seed.items()}


def strength_ok(single, dual, spec) -> dict:
    share = lambda xs, f: sum(1 for x in xs if f(x)) / len(xs)
    med = float(median(list(single) + list(dual)))
    d = dict(median=med, lo_single=share(single, lambda x: x < spec.tail_lo), lo_dual=share(dual, lambda x: x < spec.tail_lo),
             hi_single=share(single, lambda x: x > spec.tail_hi), hi_dual=share(dual, lambda x: x > spec.tail_hi))
    d["ok"] = bool(spec.kc_band[0] <= med <= spec.kc_band[1] and
                   all(d[k] <= spec.tail_share_max for k in ("lo_single", "lo_dual", "hi_single", "hi_dual")))
    return d


def choose_strength(table: dict, spec) -> dict:
    reasons, ok = {}, {}
    for s, row in sorted(table.items()):
        if not row["cap_ok"]:
            reasons[s] = "ORN_CAP"; continue
        st = strength_ok(row["single"], row["dual"], spec)
        reasons[s] = "OK" if st["ok"] else "KC"
        if st["ok"]:
            ok[s] = st["median"]
    if not ok:
        return dict(s=None, reasons=reasons)
    s = min(ok, key=lambda v: (abs(ok[v] - spec.kc_target), v))
    return dict(s=s, reasons=reasons)


def select_config(results: dict, spec) -> dict:
    elig = {c: r for c, r in results.items() if r["eligible"]}
    if not elig:
        return dict(outcome=STOP_NO_ELIGIBLE, winner=None, passing=[], near=[])
    passing = [c for c in spec.configs if c in elig and elig[c]["testable_b"] >= spec.bar_b and elig[c]["F_a"] >= spec.f_a_min]
    if not passing:
        return dict(outcome=STOP_EVEN_LOW, winner=None, passing=[], near=[])
    top = max(elig[c]["testable_b"] for c in passing)
    near = [c for c in passing if top - elig[c]["testable_b"] <= spec.tie_pairs]
    return dict(outcome=SELECTED, winner=near[0], passing=passing, near=near)


def read_band(n, c, f_a, n_b, n_a, n_a_declared, naive_a, spec) -> dict:
    if n_b != spec.n_b or n_a != n_a_declared:
        return dict(band=NOT_READ, reason="COUNTS")
    if c >= spec.bar_b:
        return dict(band=B_NC, reason=E0_ABOVE_BAR)
    if n <= c:
        return dict(band=B_TB, reason="NO_GAIN")
    if n < spec.bar_b:
        return dict(band=B_NC, reason=BELOW_BAR)
    if n - c < spec.margin:
        return dict(band=B_NC, reason=MARGIN)
    if f_a < spec.f_a_min:
        return dict(band=B_FA, reason="F_A", naive_a=naive_a, f_a_possible=naive_a >= spec.f_a_min) | {}
    return dict(band=SELECTED, reason="PASS")
```

(Fix for the fixtures: `read_band` returns exactly `{"band", "reason"}` for B_NC; for B_Fa it may carry `naive_a`, `f_a_possible` — the full-cover test only checks `["band"]` for B_Fa.)

`sentence(outcome, fields)`: a dict of templates with `str.format(**fields)` — templates copied verbatim from spec 5.4 with 〈…〉 replaced by `{name}`:
- `STOP_NO_ELIGIBLE`: "결합 부호 E-grid의 선언한 4설정 가운데 하드 제약 위반 0과 KC 활성 자격을 함께 만족한 설정이 없었다({reasons})."
- `STOP_EVEN_LOW`: "E-grid 자격 설정 가운데 짝수 (b) 21쌍에서 M2 기준(testable_b ≥ 11 ∧ F_a ≥ 2)을 넘은 설정이 없었다({table})."
- `STOP_SET_SHORT`: "L 생성기 턴 64–209에서 E-grid 키 중복을 뺀 (b) 쌍이 21개에 못 미쳤다({m}쌍)."
- `UNDECIDED`: "k = {k}의 하드 제약 해를 구성·정확 탐색으로 찾지 못했고 불가능도 증명하지 못했다."
- `B_TB`: "선택된 설정 {config}(4설정에서 짝수 턴으로 선택, 짝수 {k_even}/21)이 판정 세트(L 생성기 턴 64–{T})에서 {n}/21로 같은 세트 E0의 {c}/21보다 오르지 않았다."
- `B_NC` with reason E0_ABOVE_BAR / BELOW_BAR / MARGIN: the 5.3 sentence + " ({n}/21 대 {c}/21, F_a {f_a}/{n_a})".
- `B_FA`: "M2 (b) 기준과 여유는 넘었지만 순진 균형 (a) 쌍의 F_a가 기준에 못 미쳤다({n}/21 대 {c}/21, F_a {f_a}/{n_a}, naive_a {naive_a})."
- `SELECTED`: the spec 5.4 sentence with `{config} {T} {n} {c} {f_a} {n_a} {k_even}`.
B_NC/B_FA templates are keyed `"B_결론없음:E0_ABOVE_BAR"` etc.

`operating_characteristics(n_a, spec)`: for q in `oc_q`, c in `oc_c` plus `c ± oc_c_delta` (clipped to 0..21, tagged `"c_delta"`), naive_a in `0..min(n_a, oc_naive_max)`, and n_a rows `oc_n_a_rows` (naive_a fixed at `round(q·n_a_row/4)`… no: for the n_a rows use naive_a = n_a_row · 4/18 rounded, i.e. scale the even-set naive share; record the formula): P(n = i) = Bin(21, q); P(F_a = j) = Bin(naive_a, q); sum `read_band(i, c, j, 21, n_a, n_a, naive_a, spec)["band"]` weighted. Return `{"assumption": "pairs independent; q_b = q_a; conditional on fixed c", "rows": [dict(q, c, naive_a, n_a, tag, P={band: p})]}`.

- [ ] **Step 4: Run** `uv run pytest tests/agent/test_e_rules.py -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "encoder track: rules — eligible-first strength, pass-first selection, fixed band order with margin, sentences, operating characteristics"`

---

### Task 6: Measurer (FlyPool + cache)

**Files:**
- Create: `flymon/agent/e_measure.py`, `tests/agent/e_scripted.py`
- Test: `tests/agent/test_e_measure.py`

**Interfaces:**
- Consumes: `k_jobs.activity_job(params, items=[(i, odor, seed)], strength, settle_ms, read_ms, window_ms) -> [{i, seed, kc, n, max_win}]`; `h4_jobs.oracle_job(params, odor_x, odor_y, readout, z, types, act_seeds, select_seeds, report_seeds, alphas, strength, settle_ms, read_ms, window_ms, punish_type, reward_type) -> dict`; `ECache`.
- Produces `class EMeasurer(pool, cache: ECache, params, n_kc: int, spec)`:
  - `drive(glomeruli: list[str], c_norm: float, receptor_counts: dict) -> dict[glom, dict(mean_spikes, kc_frac_mean)]` — single-glomerulus odour `{g: c_norm / receptor_counts[g]}` at `spec.drive_strength`, seeds `spec.drive_seeds`; mean total KC read-window spikes over seeds.
  - `activity(odours: dict[str, dict], s: float, seeds) -> dict[odour_id, dict(frac: list[float], max_win: list[int])]` — one cache entry per (odour_id, odour dict, s, seed batch); batches of ≤ 64 items per job, one job per worker round.
  - `oracle(rows, strength, readout, z, types, seeds: dict(act, select, report), tag) -> list[row + oracle result]` — one cache entry per pair (inputs = odours, strength, seeds, readout, z, types, alphas, windows); runs missing pairs in rounds of `pool.n_workers`; never returns a partial list (raises if any pair is missing after the run).
- `tests/agent/e_scripted.py`: `ScriptedMeasurer` with the same three methods, deterministic fake values (e.g. activity = 0.01 × s × |odour|), and `.calls`.

- [ ] **Step 1: Failing tests** — use a fake pool:

```python
# tests/agent/test_e_measure.py
from flymon.agent.e_measure import EMeasurer
from flymon.agent.e_store import ECache
from flymon.agent.e_spec import SPEC, smoke

class FakePool:
    n_workers = 2
    def __init__(self): self.calls = []
    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        if fn.__name__ == "activity_job":
            return [[dict(i=i, seed=s, kc=list(range(int(10 * len(o)))), n=[1] * int(10 * len(o)), max_win=3)
                     for i, o, s in kw["items"]] for kw in kws]
        return [dict(report={"pre": {}, "R1": {}, "R2": {}}, kc={"jaccard": 0.1}, odor=len(kw["odor_x"])) for kw in kws]

class P:  # params stand-in accepted by the store guard functions
    kc_kc_scale = 0.0

def test_activity_cached_and_resumed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool(); sp = smoke(SPEC)
    m = EMeasurer(pool, ECache("results/encoder/cache", {"k": 1}), P(), n_kc=100, spec=sp)
    od = {"a": {"ORN_X": 1.0}, "b": {"ORN_X": 1.0, "ORN_Y": 1.0}}
    r1 = m.activity(od, 0.35, sp.strength_seeds)
    n_calls = len(pool.calls)
    r2 = m.activity(od, 0.35, sp.strength_seeds)
    assert r1 == r2 and len(pool.calls) == n_calls
    assert r1["b"]["frac"][0] == 0.2

def test_oracle_rounds_and_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = EMeasurer(pool, ECache("results/encoder/cache", {"k": 1}), P(), n_kc=100, spec=SPEC)
    rows = [dict(axis="b", turn=i, x=f"x{i}", y=f"y{i}", odor_x={"ORN_X": 1.0}, odor_y={"ORN_Y": 1.0}) for i in range(5)]
    seeds = dict(act=SPEC.even_act_seeds, select=SPEC.even_select_seeds, report=SPEC.even_report_seeds)
    out = m.oracle(rows, 0.35, {"A": "MBON13", "P": "MBON05"}, {"A": (1, 1), "P": (1, 1)}, ["MBON13", "MBON05"], seeds, "t")
    assert len(out) == 5 and [c[1] for c in pool.calls] == [2, 2, 1]
    m.oracle(rows, 0.35, {"A": "MBON13", "P": "MBON05"}, {"A": (1, 1), "P": (1, 1)}, ["MBON13", "MBON05"], seeds, "t")
    assert len(pool.calls) == 3
```

(`refuse_old_engine_output` / `refuse_modified_engine_output` take `params`; if the `P` stand-in is rejected, pass `params_list=[]` into cache writes from the measurer for tests by constructing `EMeasurer(..., guard_params=False)` — add that keyword, default `True`.)

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement** `EMeasurer` per the interface (imports `from ..brain import k_jobs, h4_jobs`; frac = `len(row["kc"]) / n_kc`; cache kinds `"drive"`, `"activity"`, `"oracle"`; the oracle job kwargs exactly as in Consumes, `alphas=spec.alphas`, `settle_ms/read_ms/window_ms` from spec, `punish_type="PPL105"`, `reward_type="PAM08"`). `ScriptedMeasurer` in `tests/agent/e_scripted.py`: activity frac = `min(0.2, 0.012 * s * len(odour))`; oracle returns a `report` made with `tests.brain.h4_scripted.report(testable=<bool from a dict keyed by (axis, turn, x, y), default False>)` plus `kc={"jaccard": 0.2}`.

- [ ] **Step 4: Run** `uv run pytest tests/agent/test_e_measure.py -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "encoder track: measurer over FlyPool with content-key cache and per-pair resume"`

---

### Task 7: Stage runner, CLI, summary chain

**Files:**
- Create: `flymon/agent/e_runner.py`, `scripts/run_encoder_grid.py`
- Test: `tests/agent/test_e_runner.py`

**Interfaces:**
- Consumes: everything above; `h4_formula.pair_stats(report, z, testable_min)`, `h4_formula.arm_aggregate(stats, naive_max, t_b_min, f_a_min)` (with `stats` keyed by `(axis, turn, x, y)`); `h3_spec.all51_glomeruli(pops)` → `(types, c_norm)`; `odor_real.cap_hz(params)`; `l_cli.provenance(files, args, argv)`; `h3_store.git_state()`.
- Produces `class Runner(measurer, pops_info, spec, summary_path=e_store.SUMMARY, smoke=False)` with methods `stage_set()`, `stage_drive()`, `stage_codebook()`, `stage_strength()`, `stage_oc()`, `stage_even()`, `stage_judge()`; each writes one summary block named after the stage (smoke: writes `results/encoder/smoke/summary.json` instead and never calls `stage_judge`). `pops_info = dict(receptor_counts, c_norm, glomeruli, n_kc, max_rate_hz, cap_hz)` so the runner needs no pops.
  - Chain (`_require(*blocks)`): each stage refuses (`SystemExit(2)`, nothing written) unless prior blocks exist; `stage_judge` additionally requires: `set.status == "OK"`, `even.outcome == "SELECTED"`, codebook digest of the winner equal to the one recorded in `codebook`, strength block recorded for the winner, and every prior block's `git.dirty_hashed` empty.
  - `stage_even` result per config: `eligible` (codebook status OK ∧ strength `s` not None), oracle rows aggregated by `h4_rules.combo_stats`-equivalent via `pair_stats`/`arm_aggregate` with `report_seeds=spec.even_report_seeds`; `testable_b = aggregate["testable_b"]`, `F_a = aggregate["F_a"]`; then `e_rules.select_config`; on a stop outcome writes the block and returns the outcome (the CLI then writes `/private/tmp/claude-503/encoder-latest.md`? — no: the CLI prints the sentence; the controller writes that file).
  - `stage_judge`: oracle on judgement (b)+(a) rows for the winner (odours from its codebook/rule, strength = chosen s) and for E0 (`odor_x_e0`/`odor_y_e0`, strength `spec.e0_strength`), seeds `judge_*`; counts must equal 21 and `n_a` else `NOT_READ` written as status `INCOMPLETE` (no band); aggregate n, c, F_a, naive_a; `read_band`; sentence; KC Jaccard per pair (from `kc.jaccard`); manifest = list of `{tag, axis, turn, x, y, cache_key}` + sha256 of each cache file.
  - Every block includes `git = h3_store.git_state()`, `provenance` (sha256 of `flymon/agent/e_*.py`, `encode_grid.py`, the script), timestamps.
- CLI: `uv run python scripts/run_encoder_grid.py --stage {set,drive,codebook,strength,oc,even,judge} [--smoke] [--workers 16]`; requires cwd = repo root and `data/malecns.npz` with the connectome sha256 (`h3_spec.SPEC.connectome_sha256`); builds `FlyPool(npz, params, flies=[{}]*W, workers=W, punish_type="PPL105", reward_type="PAM08", timeout_s=3600)` inside `if __name__ == "__main__":` → `sys.exit(main())`; loads C3 via `flymon.agent.config.load_c3_config()`, pools via `json.load(m0d)["h4"]["pools"]` → `types = pools["A"] + pools["P"]`.

- [ ] **Step 1: Failing tests** (scripted measurer, tmp cwd with a copied `results/summary/m0d.json` not needed — the runner receives readout/z/types as arguments):

```python
# tests/agent/test_e_runner.py
import json
import pytest
from tests.agent.e_scripted import ScriptedMeasurer
from flymon.agent.e_runner import Runner
from flymon.agent.e_spec import SPEC, smoke
from flymon.agent import e_store

GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
INFO = dict(receptor_counts={g: 1 + i % 5 for i, g in enumerate(GLOMS)}, c_norm=3.0, glomeruli=GLOMS,
            n_kc=1000, max_rate_hz=200.0, cap_hz=333.3)
ORACLE = dict(readout={"A": "MBON13", "P": "MBON05"}, z={"A": (10.0, 9.0), "P": (26.0, 19.0)},
              types=["MBON13", "MBON18", "MBON05", "MBON21"])

def runner(tmp_path, monkeypatch, sm=False, testable=None):
    monkeypatch.chdir(tmp_path)
    return Runner(ScriptedMeasurer(testable=testable or {}), INFO, smoke(SPEC) if sm else SPEC, oracle=ORACLE,
                  situations=FAKE_SITUATIONS, smoke=sm)

def test_out_of_order_refused(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert not (tmp_path / e_store.SUMMARY).exists()

def test_full_chain_to_stop_even_low(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    for st in ("set", "drive", "codebook", "strength", "oc"):
        getattr(r, f"stage_{st}")()
    out = r.stage_even()
    assert out["outcome"] == "STOP_EVEN_LOW"
    with pytest.raises(SystemExit):
        r.stage_judge()

def test_smoke_never_writes_summary(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, sm=True)
    for st in ("set", "drive", "codebook", "strength", "even"):
        getattr(r, f"stage_{st}")()
    assert not (tmp_path / e_store.SUMMARY).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()

def test_capped_strengths_never_measured(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    for st in ("set", "drive", "codebook"):
        getattr(r, f"stage_{st}")()
    r.stage_strength()
    measured = {c["s"] for c in r.measurer.calls if c["kind"] == "activity"}
    blk = e_store.read_summary()["strength"]
    capped = {float(s) for cfg in blk["configs"].values() for s, why in cfg["reasons"].items() if why == "ORN_CAP"}
    assert not (measured & capped)
```

`FAKE_SITUATIONS` (defined in the test module): an object with `.even()` returning 18 (a) + 21 (b) situation rows and `.judgement()` returning a `judgement_set`-shaped dict with 21 (b) and 32 (a) rows, built from `e_pairs` row fields with synthetic move/opp values taken from `pool_vocabulary()` — so the runner takes situations by injection (`situations=` argument; the CLI passes a real adapter calling `e_pairs.even_situations(pops)` / `e_pairs.judgement_set(pops, spec)`).

Add one more test: `stage_judge` after a SELECTED `stage_even` (scripted testable for ≥ 12 even (b) and ≥ 2 naive (a) pairs for `k3-full`, and judgement testable for 14 (b) with E0 testable for 3) writes band `SELECTED`, a manifest with 2 × (21 + 32) entries, and refuses a second run (`SystemExit`) because the `judge` block exists (D8: no re-run on the same set).

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement** `Runner` and the CLI per the interface. Stage details:
  - `stage_set`: `judgement_set` via `situations.judgement()`; block `set` = counts, last_turn, n_a, status, digests, skipped, `shared_odour_count`. If status `STOP_SET_SHORT`, write the block and return the outcome.
  - `stage_drive`: `measurer.drive(INFO["glomeruli"] minus exclude, c_norm, receptor_counts)` → block `drive`.
  - `stage_codebook`: for k in (3, 2): `e_codebook.build(drive_means, k, spec, unique_check=...)` → block `codebook` = `{k: {status, colour_seed, colouring, C, omega, alphabet, codebook, anneal, digest}}` plus per-config glomerulus-level `overlap_report` against `used` rows (must be 0/0 else that config gets status `NOT_UNIQUE`).
  - `stage_strength`: per config with codebook OK: odours for the 112 reachable situations; per s in `s_grid`: `cap_ok` → if False record `ORN_CAP` and skip measuring; else `measurer.activity(odours, s, spec.strength_seeds)`, per-odour medians split single/dual → `choose_strength`; record D.6 (a) count (`max_win > 31` presentations — 150 Hz over 200 ms), band share, and 176-odour statistics only if measured (record "not measured" otherwise — do not add engine runs for the 176 record).
  - `stage_oc`: `operating_characteristics(set.n_a, spec)` → block `oc` (+ sha256 of its canonical JSON).
  - `stage_even` and `stage_judge` as in Interfaces.

- [ ] **Step 4: Run** `uv run pytest tests/agent -q` — all PASS; then `uv run pytest -q -x tests/brain/test_h4_pairs.py tests/test_m2_go.py` to confirm nothing shared changed (expect the two strict xfails to stay xfail).

- [ ] **Step 5: Commit** `git commit -m "encoder track: stage runner with chain checks, manifest and smoke mode; CLI"`

---

## Execution (controller, after Task 7 — not SDD tasks)

All long runs go through Bash `run_in_background: true` from the controller (subagents die after 600 s without output). Order, each followed by a commit of `results/summary/encoder_grid.json` (smoke excepted) and a push to `lyutvs/encoder-redesign`:

1. `uv run python scripts/run_encoder_grid.py --stage set` → record digests, T, n_a in spec 12 (commit "results(encoder): judgement set fixed").
2. Smoke: `--smoke` for `set drive codebook strength even` (wall times → spec 11).
3. `--stage drive`, `--stage codebook`, `--stage strength`, `--stage oc` (commit each).
4. `--stage even` → if `STOP_*`: write `/private/tmp/claude-503/encoder-latest.md` and stop (user).
5. `--stage judge` → write the band, sentence and numbers to `encoder-latest.md` and stop (user).
