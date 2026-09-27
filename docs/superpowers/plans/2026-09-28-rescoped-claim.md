# Re-scoped Claim (reward-side machine control + battle win-rate contribution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Execution method (fixed by the user, 2026-09-28): **subagent-driven** — `sdd-implementer` per task, `sdd-reviewer` per task, final whole-branch review on opus.

**Goal:** Build the code that runs the re-scoped claim's five stages — τ_rec synthetic calibration, designed-pair qualification, the primary (reward-side machine control) verdict, the battle power pilot, and the M4 verdict — with smoke modes and self-checks, but run none of the real stages.

**Architecture:** Everything new lives in a new package `flymon/rescope/` plus `scripts/run_rescope_*.py`. The brain side reuses `flymon.brain.b_runner.probe/train`, `flymon.brain.b_rules.dprime/v_of/check_records/noplast_ok` and `flymon.brain.fly_pool.FlyPool` unmodified, with the C3 `Params` and a new spec object. The battle side reuses `flymon.agent` (`AgentPlayer`, `BrainSwarm`, `run_cohort`, `CheckpointStore`) and `flymon.battle` (`make_schedule`, providers, server) unmodified, adding a yoked-pulse player subclass and a two-block (learn → eval) driver.

**Tech Stack:** Python 3.13 (uv), numpy, scipy (betabinom), poke-env 0.16.1, pokemon-showdown 0.11.11 (node), pytest (`asyncio_mode = auto`).

**Spec:** `docs/superpowers/specs/2026-09-28-rescoped-claim-design.md` (sections 0–9 and the **section 10 amendment, which overrides 2/4/6**). Background: `docs/superpowers/specs/2026-09-14-flymon-design.md` (the "original spec"; read-only).

## Global Constraints

- Never modify: `docs/superpowers/specs/2026-09-14-flymon-design.md`, anything under `flymon/brain/`, existing `scripts/*` files, judged files under `results/summary/`, M's HASHED_FILES. `flymon/agent/*` and `flymon/battle/*` are also left unmodified (subclass / wrap instead).
- Never use the even-turn (b) 21 / (a) 18 pairs, odd-turn pairs, L's new set, M's judgement/confirmation set, or the original spec 4.4 confirmation set.
- Engine = C3: `flymon.agent.config.load_c3_config()` (Params from `results/summary/m0d.json` block `"h3"`, z/readout from `["h4"]["h4"]["combos"]["C3"]`: z A = (10.78125, 9.412096743243064), P = (26.25, 19.30889259728101); readout A = MBON13, P = MBON05).
- Primary test: `recovery_per_pulse = 0` (C3 value). Only the M4 arms use the τ_rec picked in stage 1.
- New output directories only: `results/rescope/`, `results/rescope-smoke/`, `results/summary/rescope_*.json`. Refuse any other path.
- Seeds (new, verified unused in the repo): primary probe `900_000 + 10_000*p + 100*fly + k`; primary train `9_000_000 + 100_000*p + 1000*fly + t`; N′ (brain name `N2`) train `9_800_000 + 100_000*p + 1000*fly + t`; qualification act/select/report `980_000 + 1000*p + {0,100,200} + i`; τ_rec odour generator `990_000`, τ_rec pulse seeds `991_000 + i`. Battle schedules: pilot learn 101, pilot eval 102, judgement learn 201, judgement eval 202 (`make_schedule` seeds; 0/7/8 are taken).
- Pair index p: `seed0` → 0, `p1000` → 1, …, `p1005` → 6.
- Tests: `uv run pytest -q -rfE -o addopts=""`. Real-data tests skip when `data/malecns.npz` or `results/summary/m0d.json` is missing (`pytest.mark.skipif`). This worktree has `data/malecns.npz` as a symlink to the guillemot worktree (read-only use).
- Commits: no `Co-Authored-By`, `Claude-Session` or "Generated with" lines. Commit only to branch `lyutvs/rescope-claim`; no push, no merge.
- **Execution boundary (user, 2026-09-28):** implementers and reviewers write code, tests and smoke commands only. Smoke runs and self-checks are run by the controller in the background. The real stages (τ_rec calibration, qualification, primary verdict, pilot, M4) are **not** started by this plan: before any of them the controller checks `orca worktree ps` for M's oracle/scan and reports to the user.

## Review Focus

1. **Y moving the "wrong way" or X not moving** (e_X ≤ 0, e_Y < 0) — must FAIL the pair, never pass by clipping. Tested in Task 3.
2. **A yoked RS fly having more or fewer fly turns than its donor** — queue carry-over across battles, residual ≤ 5 % of donor pulse ms or the pair is INVALID; empty (no-signal) turns never enter the queue. Tested in Task 7.
3. **The eval block leaking learning** — plasticity must be off and no pulse delivered in eval; weights before and after the eval block bit-identical. Tested in Task 8.
4. **Resuming an interrupted run** (primary pair checkpoint; battle checkpoint) — resumed run gives the same records as an uninterrupted fake run; a rerun FLY k forces RS k rerun (sha256 mismatch refuses). Tested in Tasks 5 and 8.
5. **A partial or malformed record set** (missing seed, duplicate, NaN, < 6 valid flies, one INVALID pair among PASSes) — overall verdict INVALID, never PASS/FAIL. Tested in Task 3.

## Readings of the spec (decisions the plan makes where the spec is silent)

- R1. Qualification X = the MBON05-median rule applied to the qualification pre probes (select + report seeds). The judgement's own X (Rr pre) is recorded; a mismatch is recorded, not a verdict.
- R2. Qualification floor guard ("all qualification probe seeds") = naive MBON05 of X and of Y ≥ 5 on every select and report seed.
  - R2 (2026-09-28 supplement, before any real stage; replaced by spec 10.3 "2026-09-28 amendment (floor rule B)"). The every-seed guard above is superseded: per odour (X and Y separately), identically for every pair (seed0 and candidates), over the select + report seeds, the naive MBON05 seed median ≥ 5 **and** the share of seeds with a count < 5 ≤ 1/8 (`oracle.floor_rule_b`, `RSpec.floor_silent_max = 0.125`); `qualify` returns per odour `floor` (median, n_silent, silent_share, median_ok, silent_ok, ok) and `silent_share`, and its reasons name the odour, its role (X / Y) and the failing part. Recorded only: the silent-seed share per pair and odour in `rescope_qualify.json`; per-probe active / silent state (silent := MBON05 < 5) in `rescope_primary.json` `recorded.<pair>.silent` (`rules.silent_states`); silent decision shares per arm / fly / block in `rescope_m4.json` `recorded.silent_decisions` (`stats.silent_decisions`, from the decision records' `p` counts, which every fly decision incl. FLY's learning block carries). The primary's `NOT_CONSTRUCTIBLE` (spec 10.4, median < 5) is already a median rule and is unchanged.
- R3. Qualification oracle = a reward-only copy of `h4_jobs.oracle_job` (G.14.3 "freq" edit on PAM08-core edges, α grid (0.2, 0.5, 0.8) chosen on select seeds by max change, statistic on report seeds) — pinned by a test to `oracle_job`'s reward half. Statistic `d′(dv_R1 − dv_pre)` with C3 z; qualified iff ≥ 2.0.
- R4. Pair verdict's "valid fly" = `fly_stats` not None and no NaN (±inf allowed, F.5 limit rule).
- R5. τ_rec: two flies per grid pool — fly 0 = the alternating-odour trajectory (selection), fly 1 = the same-odour trajectory (record). Ratios are sampled every 10 pulses and after the last one; "path minimum" = min over samples.
- R6. Battle ids carry the block: learn `L-fNN-bNNN`, eval `E-fNN-bNNN` (so `derive_seed` never repeats across blocks; opponent account `fm-h-<id>` ≤ 18 chars). tau uses the within-block battle number.
  - R6 (2026-09-28 supplement, final review, before S4). The ids also carry the phase: pilot learn / eval `PL-` / `PE-`, judge learn / eval `JL-` / `JE-` (`blocks.block_tag`, `blocks.BLOCKS`), so `derive_seed` differs between the pilot and the judge run as well as between blocks; `battle_index` parses only `<PL|PE|JL|JE>-fNN-bNNN`. The longest opponent account `fm-h-PL-f00-b000-3` is 18 characters; the battle CLI refuses a schedule whose last-attempt name would exceed 18. RND reseeds its `RandomProvider` at every battle start from `derive_seed("rnd", phase, fly, battle_id)`, so a resumed or retried RND fly replays an uninterrupted one.
- R7. The paired bootstrap resamples fly index k jointly for both arms of a comparison, then battles within each fly, 10,000 draws, seed 20260928.
- R9. Pilot size: 6 flies per arm (FLY, RS, COFF, RND), L = 40, E = 20 (spec 10.7 "FLY·RS ≥ 6"; 4.4's example 20 eval battles).
- R8. Power: normal approximation with variance components (between-fly variance × 1.5 margin, binomial within), joint (2b) power by Monte Carlo of two correlated-by-FLY z statistics (FLY noise shared), 20,000 draws, seed 20260928.
- R10 (2026-09-28 supplement). M4 valid pairs: each comparison (FLY − RND, FLY − C-off, FLY − RS) needs at least ceil(F · 6/8) valid pairs, and never fewer than 2 (`stats.min_valid_pairs`). A pair counts as INVALID when fly k is INVALID in either arm (RS: `invalid`, `donor_invalid`, residual > 5 %, eval weights changed). A shortage is per verdict: (2a) is INVALID only when FLY/RND is short; (2b) is INVALID when FLY/C-off or FLY/RS is short; the other verdict is still judged. A missing or malformed arm file, or eval schedule digests that differ, make the whole M4 summary INVALID. An INVALID verdict is never PASS or FAIL.
- R11 (2026-09-28 supplement). Wall-clock extrapolation (spec 10.7 partial-batch loss): batch model. An arm plays all its flies concurrently and the brain arms' decisions go through a FlyPool of W workers, so an arm's hours scale with ceil(F / W) batches, where W is the judge's workers (`write_rescope_power.py --workers`, default 16, recorded in rescope_power.json). Per-batch-battle rate from the pilot = arm wall_clock_s / (ceil(F_pilot / W_pilot) · battles per fly), W_pilot = the pilot result.json's `workers`: brain from FLY (learn + eval battles), no-brain from RND (eval battles; RND / MAX run their flies concurrently the same way, so the same form applies). hours = ceil(F / W) · [2 · (L + E) · s_brain + E · s_brain + 2 · E · s_nobrain] / 3600. wall_clock_s is the sum over every session in the arm's wall_clock.json, aborted sessions included (each session is ended in a `finally`); the power writer refuses to size when the FLY or RND session log is missing, any session has no end record, or the sessions do not sum to wall_clock_s.

---

## File Structure

| File | Responsibility |
|---|---|
| `flymon/rescope/__init__.py` | package marker |
| `flymon/rescope/spec.py` | `RSpec` frozen dataclass: all constants, seed functions, pair names, digest |
| `flymon/rescope/pairs.py` | new designed-pair generator, skip rule, digest, odours by pair name |
| `flymon/rescope/store.py` | output-path guard, JSON writer, primary checkpoint |
| `flymon/rescope/rules.py` | pure primary verdict: X rule, fly stats, pair/control/overall verdicts, t(m), OC table |
| `flymon/rescope/oracle.py` | `reward_oracle_job` (reward-only copy of `oracle_job`) and pure qualification decision |
| `flymon/rescope/primary.py` | Rr/N/N2/noplast layout and resumable pair runner on `b_runner.probe/train` |
| `flymon/rescope/taurec.py` | synthetic odours, pulse schedule, trajectory runner, grid selection |
| `flymon/rescope/yoke.py` | yoked pulse queue from a donor log, residual accounting |
| `flymon/rescope/players.py` | `YokedPlayer(AgentPlayer)` |
| `flymon/rescope/blocks.py` | two-block schedules (learn/eval), block battle ids, digests, disjointness |
| `flymon/rescope/stats.py` | per-fly win tables, paired hierarchical bootstrap, (2a)/(2b) verdict |
| `flymon/rescope/power.py` | pilot variance components, joint power, (F, E) choice, wall-clock model |
| `scripts/run_rescope_taurec.py` | stage 1 CLI |
| `scripts/run_rescope_qualify.py` | stage 2 CLI |
| `scripts/run_rescope_primary.py` | stage 3 CLI |
| `scripts/run_rescope_battles.py` | stages 4–5 battle CLI (arms FLY / RS / C-off / RND / MAX, pilot or judge) |
| `scripts/write_rescope_power.py` | stage 4 summary: variance, (F, E), budget |
| `scripts/write_rescope_m4.py` | stage 5 summary: (2a)/(2b) verdict |
| `tests/rescope/…` | one test module per source module |

---

### Task 1: Spec constants and the new designed-pair generator

**Files:**
- Create: `flymon/rescope/__init__.py`, `flymon/rescope/spec.py`, `flymon/rescope/pairs.py`
- Test: `tests/rescope/__init__.py`, `tests/rescope/test_spec.py`, `tests/rescope/test_pairs.py`

**Interfaces:**
- Consumes: `flymon.brain.stimuli.channel_strengths(pops, types) -> dict`, `flymon.brain.stimuli.design_odor_pair(pops, k, seed)`, `flymon.brain.circuits.Populations`, `flymon.brain.connectome.Connectome.load`.
- Produces:
  - `RSpec` (frozen dataclass) and `SPEC = RSpec()` with fields listed in Step 3; methods `pair_names() -> tuple[str, ...]`, `pair_index(name) -> int`, `probe_seeds(name, fly) -> list[int]`, `train_seed(name, fly, trial, second_null=False) -> int`, `qual_seeds(name) -> dict[str, list[int]]` (keys `act`, `select`, `report`).
  - `candidates(pops, exclude) -> list[str]`; `sample_pair(pops, seed, k=8) -> tuple[list[str], list[str]]`; `new_pairs(pops, spec) -> list[dict]` (dicts `{seed, a, b}`); `pairs_digest(pairs) -> str`; `pair_odors(pops, name, spec) -> {"a": dict, "b": dict}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_spec.py
import json
from pathlib import Path
import pytest
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]

def test_pair_names_and_index():
    assert SPEC.pair_names() == ("seed0", "p1000", "p1001", "p1002", "p1003", "p1004", "p1005")
    assert SPEC.pair_index("seed0") == 0 and SPEC.pair_index("p1005") == 6

def test_seed_blocks_disjoint():
    probe, train, n2, qual = set(), set(), set(), set()
    for name in SPEC.pair_names():
        for f in range(SPEC.n_flies):
            probe |= set(SPEC.probe_seeds(name, f))
            for t in range(SPEC.trials):
                train.add(SPEC.train_seed(name, f, t))
                n2.add(SPEC.train_seed(name, f, t, second_null=True))
        for v in SPEC.qual_seeds(name).values():
            qual |= set(v)
    blocks = [probe, train, n2, qual]
    assert sum(map(len, blocks)) == len(set().union(*blocks))
    assert len(probe) == 7 * 8 * 8 and len(train) == 7 * 8 * 20 and len(qual) == 7 * 24
    taurec = set(range(SPEC.taurec_seed_base, SPEC.taurec_seed_base + SPEC.taurec_pulses)) | {SPEC.taurec_gen_seed}
    assert not taurec & set().union(*blocks)

@pytest.mark.skipif(not (ROOT / "results/summary/m0d.json").exists(), reason="m0d summary missing")
def test_z_matches_c3_record():
    c3 = json.loads((ROOT / "results/summary/m0d.json").read_text())["h4"]["h4"]["combos"]["C3"]
    assert tuple(c3["z"]["A"]) == SPEC.z_a and tuple(c3["z"]["P"]) == SPEC.z_p
    assert c3["readout"] == {"A": SPEC.a_type, "P": SPEC.p_type}
```

```python
# tests/rescope/test_pairs.py
from pathlib import Path
import pytest
from flymon.rescope.pairs import new_pairs, pairs_digest, sample_pair, pair_odors
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")

@pytest.fixture(scope="module")
def pops():
    from flymon.brain.connectome import Connectome
    from flymon.brain.circuits import Populations
    return Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))

@needs_npz
def test_digest_pinned(pops):
    pairs = new_pairs(pops, SPEC)
    assert [p["seed"] for p in pairs] == [1000, 1001, 1002, 1003, 1004, 1005]
    assert pairs_digest(pairs) == SPEC.pairs_digest

@needs_npz
def test_pair_shape(pops):
    a, b = sample_pair(pops, 1000)
    assert len(a) == len(b) == 8 and not set(a) & set(b)
    assert "ORN_DA1" not in a + b and "ORN_V" not in a + b

@needs_npz
def test_skip_rule_skips_used_band(pops, monkeypatch):
    from flymon.rescope import pairs as mod
    used = mod.used_sets(pops, SPEC)
    real = mod.sample_pair
    calls = []
    def fake(p, seed, k=8):
        calls.append(seed)
        if seed == 1000:  # pretend seed 1000 reproduced the seed-0 band
            s = sorted(next(iter(used)))
            return s[0::2], s[1::2]
        return real(p, seed, k)
    monkeypatch.setattr(mod, "sample_pair", fake)
    got = mod.new_pairs(pops, SPEC)
    assert got[0]["seed"] == 1001 and len(got) == 6 and calls[:2] == [1000, 1001]

@needs_npz
def test_pair_odors_seed0_is_design_pair(pops):
    from flymon.brain.stimuli import design_odor_pair
    a, b = design_odor_pair(pops, k=8, seed=0)
    assert pair_odors(pops, "seed0", SPEC) == {"a": a, "b": b}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q -rfE -o addopts="" tests/rescope/test_spec.py tests/rescope/test_pairs.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'flymon.rescope'`.

- [ ] **Step 3: Implement**

```python
# flymon/rescope/spec.py
"""Constants of the re-scoped claim (docs/superpowers/specs/2026-09-28-rescoped-claim-design.md, section 10 overrides)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RSpec:
    k: int = 8
    pair_seeds: tuple = (1000, 1001, 1002, 1003, 1004, 1005)
    gen_start: int = 1000
    n_pairs: int = 6
    used_design_seeds: tuple = (0, 7, 23)
    pairs_digest: str = "2c5d8076c912e069bdc61b29221c53007bea7f1d065b5e20d9ddeb1952d14133"
    control: str = "seed0"
    strength: float = 0.35
    n_flies: int = 8
    n_probe: int = 8
    trials: int = 20
    pulse_ms: float = 400.0
    reward_dan: str = "PAM08"
    punish_dan: str = "PPL105"
    probe_settle_ms: float = 800.0
    probe_read_ms: float = 600.0
    train_settle_ms: float = 800.0
    train_gap_ms: float = 200.0
    a_type: str = "MBON13"
    p_type: str = "MBON05"
    z_a: tuple = (10.78125, 9.412096743243064)
    z_p: tuple = (26.25, 19.30889259728101)
    x_tie: str = "b"
    floor_spikes: float = 5.0
    valid_min: int = 6
    assoc_min: float = 1.0
    spill_max: float = 0.5
    sign_min: float = 0.75
    oracle_min: float = 2.0
    oracle_alphas: tuple = (0.2, 0.5, 0.8)
    oracle_settle_ms: float = 800.0
    oracle_read_ms: float = 600.0
    kc_window_ms: float = 200.0
    n_qual_seeds: int = 8
    min_pairs: int = 4
    fp_cap: float = 0.10
    q_null: float = 0.2
    rho: float = 0.3
    probe_base: int = 900_000
    train_base: int = 9_000_000
    train_base_n2: int = 9_800_000
    qual_base: int = 980_000
    recovery_grid: tuple = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02)
    median_floor: float = 0.5
    taurec_pulses: int = 1000
    taurec_odours: int = 50
    taurec_gen_seed: int = 990_000
    taurec_seed_base: int = 991_000
    taurec_sample_every: int = 10
    taurec_reward_ms: float = 600.0
    taurec_punish_ms: float = 400.0
    residual_max: float = 0.05
    boot_draws: int = 10_000
    boot_seed: int = 20260928
    power_draws: int = 20_000
    mie: float = 0.05
    power_target: float = 0.8
    var_margin: float = 1.5
    budget_hours: float = 60.0
    learn_battles: int = 40
    retry_max: int = 3
    schedule_seeds: tuple = (("pilot", 101, 102), ("judge", 201, 202))

    def pair_names(self) -> tuple:
        return (self.control,) + tuple(f"p{s}" for s in self.pair_seeds)

    def pair_index(self, name: str) -> int:
        return self.pair_names().index(name)

    def probe_seeds(self, name: str, fly: int) -> list:
        p = self.pair_index(name)
        return [self.probe_base + 10_000 * p + 100 * fly + k for k in range(self.n_probe)]

    def train_seed(self, name: str, fly: int, trial: int, second_null: bool = False) -> int:
        base = self.train_base_n2 if second_null else self.train_base
        return base + 100_000 * self.pair_index(name) + 1000 * fly + trial

    def qual_seeds(self, name: str) -> dict:
        b = self.qual_base + 1000 * self.pair_index(name)
        return {key: [b + off + i for i in range(self.n_qual_seeds)]
                for key, off in (("act", 0), ("select", 100), ("report", 200))}


SPEC = RSpec()
```

```python
# flymon/rescope/pairs.py
"""New designed odour pairs (spec 4.2): 16 glomeruli drawn per seed from the whole candidate set, not a band."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..brain.stimuli import channel_strengths, design_odor_pair

EXCLUDE = ("ORN_DA1", "ORN_V")


def candidates(pops, exclude=EXCLUDE) -> list:
    cand = [t for t in pops.receptor_types if t not in exclude]
    return sorted(cand, key=lambda t: (len(pops.receptor_types[t]), t))


def sample_pair(pops, seed: int, k: int = 8) -> tuple:
    cand = candidates(pops)
    pick = sorted(np.random.default_rng(seed).choice(len(cand), size=2 * k, replace=False).tolist())
    band = [cand[i] for i in pick]
    return band[0::2], band[1::2]


def used_sets(pops, spec) -> list:
    out = []
    for s in spec.used_design_seeds:
        a, b = design_odor_pair(pops, k=spec.k, seed=s)
        out.append(frozenset(a) | frozenset(b))
    return out


def new_pairs(pops, spec) -> list:
    seen, pairs, seed = used_sets(pops, spec), [], spec.gen_start
    while len(pairs) < spec.n_pairs:
        a, b = sample_pair(pops, seed, spec.k)
        u = frozenset(a) | frozenset(b)
        if u not in seen:
            seen.append(u)
            pairs.append({"seed": seed, "a": list(a), "b": list(b)})
        seed += 1
    return pairs


def pairs_digest(pairs: list) -> str:
    canon = json.dumps([{"seed": p["seed"], "a": p["a"], "b": p["b"]} for p in pairs], sort_keys=True,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def pair_odors(pops, name: str, spec) -> dict:
    if name == spec.control:
        a, b = design_odor_pair(pops, k=spec.k, seed=0)
        return {"a": a, "b": b}
    seed = int(name[1:])
    p = next(p for p in new_pairs(pops, spec) if p["seed"] == seed)
    return {"a": channel_strengths(pops, p["a"]), "b": channel_strengths(pops, p["b"])}
```

Also create empty `flymon/rescope/__init__.py` and `tests/rescope/__init__.py`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q -rfE -o addopts="" tests/rescope/test_spec.py tests/rescope/test_pairs.py`
Expected: all PASS (real-data tests run here because `data/malecns.npz` exists).

- [ ] **Step 5: Commit**

```bash
git add flymon/rescope/__init__.py flymon/rescope/spec.py flymon/rescope/pairs.py tests/rescope
git commit -m "feat(rescope): RSpec constants, disjoint seed blocks and the new designed-pair generator (digest pinned)"
```

---

### Task 2: Output guard, JSON writer and primary checkpoint

**Files:**
- Create: `flymon/rescope/store.py`
- Test: `tests/rescope/test_store.py`

**Interfaces:**
- Consumes: `flymon.brain.pool_bench.refuse_modified_engine_output(path, params)` (existing guard; import only). `flymon.brain.fly_pool.FlyPool.state()` shape `{"flies": [dict(enabled, shuffle_seed, w)]}`.
- Produces: `guard(path, params_list) -> Path`; `write_json(path, obj, params_list) -> Path` (atomic, sorted keys, indent 1); `git_provenance(files) -> dict` (`{"commit", "dirty": bool, "dirty_files": [...]}`); `class Checkpoint(directory, key, params_list)` with `load() -> dict | None` and `save(state: dict) -> None` (state keys `records, x, done, provenance, weights`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_store.py
import json
import numpy as np
import pytest
from flymon.brain.config import Params
from flymon.rescope import store

def test_guard_allows_only_rescope_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for ok in ("results/rescope/a.json", "results/rescope-smoke/x/y.json", "results/summary/rescope_primary.json"):
        assert store.guard(ok, [Params()])
    for bad in ("results/b/a.json", "results/summary/m0d.json", "results/m0d/x.json", "elsewhere.json"):
        with pytest.raises(SystemExit):
            store.guard(bad, [Params()])

def test_write_json_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = store.write_json("results/rescope/t.json", {"b": 1, "a": [1, 2]}, [Params()])
    assert json.loads(p.read_text()) == {"a": [1, 2], "b": 1}

def test_checkpoint_roundtrip_and_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ck = store.Checkpoint("results/rescope/pairX", "key1", [Params()])
    assert ck.load() is None
    st = dict(records=[{"brain": "Rr"}], x="a", done=["pre"], provenance={"commit": "c"},
              weights={"flies": [dict(enabled=True, shuffle_seed=None, w=np.arange(3, dtype=np.float32))]})
    ck.save(st)
    got = ck.load()
    assert got["x"] == "a" and got["done"] == ["pre"] and got["records"] == [{"brain": "Rr"}]
    assert np.array_equal(got["weights"]["flies"][0]["w"], np.arange(3, dtype=np.float32))
    assert store.Checkpoint("results/rescope/pairX", "key2", [Params()]).load() is None
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_store.py` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# flymon/rescope/store.py
"""Where re-scope outputs may go, and the primary test's per-pair checkpoint (same shape as b_store.Checkpoint)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..brain.pool_bench import refuse_modified_engine_output

ALLOWED_DIRS = ("results/rescope/", "results/rescope-smoke/")
ALLOWED_SUMMARY_PREFIX = "results/summary/rescope_"


def guard(path, params_list) -> Path:
    p = Path(path)
    rel = p.resolve().relative_to(Path.cwd().resolve()).as_posix() if p.is_absolute() else p.as_posix()
    ok = rel.startswith(ALLOWED_DIRS) or (rel.startswith(ALLOWED_SUMMARY_PREFIX) and rel.endswith(".json"))
    if not ok:
        raise SystemExit(f"refusing to write {rel}: re-scope outputs go under {ALLOWED_DIRS} or {ALLOWED_SUMMARY_PREFIX}*.json")
    for params in params_list:
        refuse_modified_engine_output(p, params)
    return p


def _atomic_write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, p)


def write_json(path, obj, params_list) -> Path:
    p = guard(path, params_list)
    _atomic_write(p, json.dumps(obj, sort_keys=True, indent=1, default=float).encode())
    return p


def git_provenance(files=()) -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    dirty = [ln[3:] for ln in run("status", "--porcelain", "--", *files).splitlines()] if files else \
        [ln[3:] for ln in run("status", "--porcelain").splitlines()]
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty), "dirty_files": dirty}


class Checkpoint:
    def __init__(self, directory, key: str, params_list):
        self.path = guard(Path(directory) / "checkpoint.npz", params_list)
        self.key = key

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        with np.load(self.path, allow_pickle=False) as z:
            prog = json.loads(str(z["progress"]))
            if prog["key"] != self.key:
                return None
            flies = [dict(enabled=e, shuffle_seed=None, w=z[f"w{i}"].copy()) for i, e in enumerate(prog["enabled"])]
        return dict(records=prog["records"], x=prog["x"], done=prog["done"], provenance=prog["provenance"],
                    weights={"flies": flies})

    def save(self, st: dict) -> None:
        flies = st["weights"]["flies"]
        prog = dict(key=self.key, records=st["records"], x=st["x"], done=st["done"], provenance=st.get("provenance"),
                    enabled=[bool(f["enabled"]) for f in flies])
        arrays = {f"w{i}": np.asarray(f["w"], dtype=np.float32) for i, f in enumerate(flies)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".npz")
        os.close(fd)
        np.savez(tmp, progress=np.array(json.dumps(prog)), **arrays)
        os.replace(tmp, self.path)
```

If `refuse_modified_engine_output` has a different name or signature in `flymon/brain/pool_bench.py`, use the function `flymon/brain/b_store.py:guard` calls for the same purpose (read `b_store.py:24-33`) and note it in the report.

- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add flymon/rescope/store.py tests/rescope/test_store.py && git commit -m "feat(rescope): output guard (results/rescope*, rescope_* summaries), atomic JSON and primary checkpoint"`

---

### Task 3: Primary verdict rules (pure)

**Files:**
- Create: `flymon/rescope/rules.py`
- Test: `tests/rescope/rescope_fixtures.py`, `tests/rescope/test_rules.py`, `tests/rescope/test_rules_mutations.py`

**Interfaces:**
- Consumes: `flymon.brain.b_rules.dprime(x) -> float|None`, `v_of(counts, spec) -> float`, `check_records(records, brains, spec, seeds_of) -> list[str]`, `noplast_ok(records) -> bool`; `SPEC` (Task 1). Records are b_runner.probe records: `dict(brain, fly, stage, seed, counts={"a": {A: int, P: int}, "b": {...}})`, stages `"pre"`, `"S1"`, brains `Rr`, `N`, `N2`, `noplast` (fly 0 only).
- Produces:
  - constants `PASS FAIL INVALID NOT_CONSTRUCTIBLE STOP_MACHINE STOP_PROTOCOL STOP_CONTROL_INVALID STOP_FEW_PAIRS OK`, `BRAINS = ("Rr", "N", "N2", "noplast")`.
  - `choose_x(records, spec) -> str`; `naive_p(records, x, spec) -> float` (median MBON05 over Rr pre for odour x).
  - `fly_stats(records, x, spec, fly) -> dict | None` keys `assoc, null, e_x, e_y, spill, level, sign`.
  - `pair_verdict(records, x, spec, pair) -> dict` keys `status, reasons, n_valid, assoc_median, null_max, spill_median, level_median, sign_median, x`.
  - `control_verdict(records, x, spec, qualified: bool) -> dict` keys `status, sign_median, pair`.
  - `threshold_t(m, spec) -> int | None`; `false_pass(t, m, q, rho) -> float`; `oc_table(spec) -> dict`.
  - `overall(control: dict, pairs: dict[str, dict], spec) -> dict` keys `status, m, t, n_pass`.

- [ ] **Step 1: Write the fixture builder**

```python
# tests/rescope/rescope_fixtures.py
"""Synthetic probe records for the re-scope primary test. V moves through the P (MBON05) count: lowering P raises V."""
import numpy as np
from flymon.rescope.spec import SPEC


def records(x="a", p_naive=30, a_naive=10, dx=0.0, dy=0.0, null=0.0, noise=0.0, seed=0, pair="p1000",
            n_flies=None, noplast_move=0, drop=None):
    """dx / dy: drop of MBON05 count on X / Y in Rr at S1 relative to N (positive = learning).
    null: extra P drop of N2 on X relative to N (the null's association)."""
    rng = np.random.default_rng(seed)
    n_flies = SPEC.n_flies if n_flies is None else n_flies
    y = "b" if x == "a" else "a"
    out = []
    def rec(brain, fly, stage, s, px, py):
        c = {o: {SPEC.a_type: a_naive, SPEC.p_type: int(max(0, round(v + rng.normal(0, noise))))}
             for o, v in ((x, px), (y, py))}
        out.append(dict(brain=brain, fly=fly, stage=stage, seed=s, counts=c))
    for f in range(n_flies):
        for s in SPEC.probe_seeds(pair, f):
            for b in ("Rr", "N", "N2"):
                rec(b, f, "pre", s, p_naive, p_naive)
            rec("Rr", f, "S1", s, p_naive - dx, p_naive - dy)
            rec("N", f, "S1", s, p_naive, p_naive)
            rec("N2", f, "S1", s, p_naive - null, p_naive)
    for s in SPEC.probe_seeds(pair, 0):
        rec("noplast", 0, "pre", s, p_naive, p_naive)
        rec("noplast", 0, "S1", s, p_naive + noplast_move, p_naive)
    if drop is not None:
        out = [r for i, r in enumerate(out) if i != drop]
    return out
```

(With `noise > 0` the per-seed ΔV varies, so d′ is finite; with `noise = 0` and `dx > 0` d′ is +inf, which F.5 allows.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/rescope/test_rules.py
import math
import pytest
from flymon.rescope import rules as R
from flymon.rescope.spec import SPEC
from tests.rescope.rescope_fixtures import records

def test_specific_learning_passes():
    v = R.pair_verdict(records(dx=12, dy=0, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.PASS and v["spill_median"] <= 0.5

def test_presentation_drift_only_fails():
    v = R.pair_verdict(records(dx=0, dy=0, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL

def test_full_generalisation_fails_on_spill():
    v = R.pair_verdict(records(dx=12, dy=12, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL and v["spill_median"] > 0.5

def test_y_reverse_spill_fails():  # X +small, Y moves the other way: clipping must not pass it
    v = R.pair_verdict(records(dx=1, dy=-10, noise=0.5), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL

def test_x_worse_but_y_reverse_fails():  # e_X < 0 while assoc is large: only the e_X <= 0 -> inf rule catches it
    v = R.pair_verdict(records(dx=-0.5, dy=-10, noise=0.5), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL and v["assoc_median"] >= 1.0

def test_x_not_moving_is_infinite_spill():
    st = R.fly_stats(records(dx=-3, dy=0, noise=1), "a", SPEC, 0)
    assert st["e_x"] <= 0 and math.isinf(st["spill"])

def test_null_bigger_than_assoc_fails():
    v = R.pair_verdict(records(dx=6, null=30, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL and v["null_max"] >= v["assoc_median"]

def test_naive_negative_level_still_passes():  # no level gate: X-only learning over a negative naive dV passes
    recs = records(dx=10, noise=2)
    for r in recs:  # make X naively much more P-driven than Y (naive dV strongly negative)
        r["counts"]["a"][SPEC.p_type] += 20
    v = R.pair_verdict(recs, "a", SPEC, "p1000")
    assert v["status"] == R.PASS and v["level_median"] < 1.0

def test_x_rule_uses_mbon05_median():
    recs = records(x="b", p_naive=30)
    for r in recs:
        if r["brain"] == "Rr" and r["stage"] == "pre":
            r["counts"]["b"][SPEC.p_type] = 40
    assert R.choose_x(recs, SPEC) == "b"
    assert R.choose_x(records(x="a"), SPEC) == SPEC.x_tie  # equal medians -> tie rule

def test_x_mismatch_invalid():
    v = R.pair_verdict(records(x="a", dx=12, noise=2), "b", SPEC, "p1000")
    assert v["status"] == R.INVALID

def test_floor_not_constructible():
    v = R.pair_verdict(records(p_naive=3, dx=1, noise=0.5), "a", SPEC, "p1000")
    assert v["status"] == R.NOT_CONSTRUCTIBLE

def test_missing_record_invalid():
    v = R.pair_verdict(records(dx=12, noise=2, drop=5), "a", SPEC, "p1000")
    assert v["status"] == R.INVALID

def test_noplast_move_stop_machine():
    v = R.pair_verdict(records(dx=12, noise=2, noplast_move=1), "a", SPEC, "p1000")
    assert v["status"] == R.STOP_MACHINE

def test_control_sign_and_qualification():
    good = records(dx=12, noise=2, pair="seed0")
    assert R.control_verdict(good, "a", SPEC, qualified=True)["status"] == R.OK
    assert R.control_verdict(good, "a", SPEC, qualified=False)["status"] == R.STOP_CONTROL_INVALID
    bad = records(dx=-12, noise=2, pair="seed0")
    assert R.control_verdict(bad, "a", SPEC, qualified=True)["status"] == R.STOP_PROTOCOL

def test_threshold_table():
    assert [R.threshold_t(m, SPEC) for m in (4, 5, 6)] == [4, 4, 5]
    assert R.false_pass(4, 4, 0.2, 0.3) == pytest.approx(0.033, abs=1e-3)
    assert R.false_pass(5, 6, 0.2, 0.3) == pytest.approx(0.052, abs=1e-3)

def _p(status): return {"status": status}

def test_overall_propagation():
    ok = {"status": R.OK}
    four_pass = {f"p{1000+i}": _p(R.PASS) for i in range(4)}
    assert R.overall(ok, four_pass, SPEC)["status"] == R.PASS
    three = dict(four_pass, p1003=_p(R.FAIL))
    assert R.overall(ok, three, SPEC)["status"] == R.FAIL          # m = 4 needs 4/4
    assert R.overall(ok, dict(four_pass, p1004=_p(R.INVALID)), SPEC)["status"] == R.INVALID
    assert R.overall(ok, dict(four_pass, p1004=_p(R.NOT_CONSTRUCTIBLE)), SPEC)["status"] == R.NOT_CONSTRUCTIBLE
    assert R.overall(ok, dict(four_pass, p1004=_p(R.STOP_MACHINE)), SPEC)["status"] == R.STOP_MACHINE
    assert R.overall({"status": R.STOP_PROTOCOL}, four_pass, SPEC)["status"] == R.STOP_PROTOCOL
    assert R.overall(ok, {k: four_pass[k] for k in list(four_pass)[:3]}, SPEC)["status"] == R.STOP_FEW_PAIRS
```

`tests/rescope/test_rules_mutations.py`: each mutation is a monkeypatch of one rule in `rules.py`, and the test asserts at least one fixture's verdict changes (a surviving mutant means that rule is untested):

```python
# tests/rescope/test_rules_mutations.py
import math
import pytest
from flymon.rescope import rules as R
from flymon.rescope.spec import SPEC
from tests.rescope.rescope_fixtures import records

CASES = [records(dx=12, noise=2), records(dx=1, dy=-10, noise=0.5), records(dx=-0.5, dy=-10, noise=0.5),
         records(dx=6, null=30, noise=2), records(p_naive=30, dx=12, noise=2)]
# case 4: make Y's naive MBON05 low so only the Y floor catches it
for r in CASES[4]:
    if r["stage"] == "pre":
        r["counts"]["b"][SPEC.p_type] = 3

MUTANTS = {
    "no_inf_on_nonpositive_ex": ("_spill", lambda ex, ey: 0.0 if ex <= 0 else abs(ey) / ex),
    "clip_instead_of_abs": ("_spill", lambda ex, ey: math.inf if ex <= 0 else max(0.0, ey / ex)),
    "no_null_compare": ("_beats_null", lambda med, null_max: True),
    "no_y_floor": ("_floor_odours", lambda x, y: (x,)),
}

def verdicts():
    return [R.pair_verdict(c, "a", SPEC, "p1000")["status"] for c in CASES]

@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name):
    base = verdicts()
    attr, fn = MUTANTS[name]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, attr, fn)
        assert verdicts() != base, f"mutant {name} survived"

def test_threshold_mutant_killed():
    assert [math.ceil(2 * m / 3) for m in (4, 6)] != [R.threshold_t(m, SPEC) for m in (4, 6)]
```


- [ ] **Step 3: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_rules.py tests/rescope/test_rules_mutations.py` → FAIL (module missing).

- [ ] **Step 4: Implement**

```python
# flymon/rescope/rules.py
"""The primary (reward-side machine control) verdict — spec 10.4. Pure functions over b_runner probe records."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import betabinom

from ..brain.b_rules import check_records, dprime, noplast_ok, v_of

PASS, FAIL, INVALID, NOT_CONSTRUCTIBLE = "PASS", "FAIL", "INVALID", "NOT_CONSTRUCTIBLE"
STOP_MACHINE, STOP_PROTOCOL, STOP_CONTROL_INVALID, STOP_FEW_PAIRS, OK = (
    "STOP_MACHINE", "STOP_PROTOCOL", "STOP_CONTROL_INVALID", "STOP_FEW_PAIRS", "OK")
BRAINS = ("Rr", "N", "N2", "noplast")


def _other(x): return "b" if x == "a" else "a"


def _index(records):
    return {(r["brain"], r["fly"], r["stage"], r["seed"]): r["counts"] for r in records}


def naive_p(records, x, spec) -> float:
    return float(np.median([r["counts"][x][spec.p_type] for r in records if r["brain"] == "Rr" and r["stage"] == "pre"]))


def choose_x(records, spec) -> str:
    a, b = naive_p(records, "a", spec), naive_p(records, "b", spec)
    return spec.x_tie if a == b else ("a" if a > b else "b")


def _spill(e_x, e_y) -> float:
    return math.inf if e_x <= 0 else abs(e_y) / e_x


def _beats_null(med, null_max) -> bool:
    return med > null_max


def _floor_odours(x, y) -> tuple:
    return (x, y)


def fly_stats(records, x, spec, fly) -> dict | None:
    y, idx = _other(x), _index(records)
    seeds = sorted({r["seed"] for r in records if r["brain"] == "Rr" and r["fly"] == fly and r["stage"] == "pre"})
    try:
        V = {(b, st): np.array([[v_of(idx[(b, fly, st, s)][o], spec) for s in seeds] for o in (x, y)])
             for b in ("Rr", "N", "N2") for st in ("pre", "S1")}
        P = {b: np.array([idx[(b, fly, "S1", s)][x][spec.p_type] for s in seeds]) for b in ("Rr", "N")}
    except KeyError:
        return None
    dv = {b: V[(b, "S1")][0] - V[(b, "S1")][1] for b in ("Rr", "N", "N2")}
    assoc, null, level = dprime(dv["Rr"] - dv["N"]), dprime(dv["N2"] - dv["N"]), dprime(dv["Rr"])
    if assoc is None or null is None:
        return None
    e_x = float(np.mean(V[("Rr", "S1")][0] - V[("N", "S1")][0]))
    e_y = float(np.mean(V[("Rr", "S1")][1] - V[("N", "S1")][1]))
    sign = float(np.mean(np.where(P["Rr"] < P["N"], 1.0, np.where(P["Rr"] == P["N"], 0.5, 0.0))))
    return dict(assoc=assoc, null=null, e_x=e_x, e_y=e_y, spill=_spill(e_x, e_y), level=level, sign=sign)


def _seeds_of(spec, pair):
    return lambda f: spec.probe_seeds(pair, f)


def pair_verdict(records, x, spec, pair) -> dict:
    out = dict(pair=pair, x=x, reasons=[], n_valid=0, assoc_median=None, null_max=None, spill_median=None,
               level_median=None, sign_median=None)
    reasons = check_records(records, BRAINS, spec, _seeds_of(spec, pair))
    if reasons:
        return dict(out, status=INVALID, reasons=reasons)
    if not noplast_ok(records):
        return dict(out, status=STOP_MACHINE, reasons=["noplast counts moved"])
    if x != choose_x(records, spec):
        return dict(out, status=INVALID, reasons=[f"X {x} is not the rule's {choose_x(records, spec)}"])
    if min(naive_p(records, o, spec) for o in _floor_odours(x, _other(x))) < spec.floor_spikes:
        return dict(out, status=NOT_CONSTRUCTIBLE, reasons=["naive MBON05 median below floor"])
    stats = [fly_stats(records, x, spec, f) for f in range(spec.n_flies)]
    valid = [s for s in stats if s is not None and not any(isinstance(v, float) and math.isnan(v) for v in s.values())]
    out["n_valid"] = len(valid)
    if len(valid) < spec.valid_min:
        return dict(out, status=INVALID, reasons=[f"{len(valid)} valid flies < {spec.valid_min}"])
    med = lambda k: float(np.median([s[k] for s in valid]))
    out.update(assoc_median=med("assoc"), null_max=float(max(s["null"] for s in valid)), spill_median=med("spill"),
               level_median=med("level"), sign_median=med("sign"))
    ok = (out["assoc_median"] >= spec.assoc_min and _beats_null(out["assoc_median"], out["null_max"])
          and out["spill_median"] <= spec.spill_max)
    return dict(out, status=PASS if ok else FAIL)


def control_verdict(records, x, spec, qualified: bool) -> dict:
    if not qualified:
        return dict(status=STOP_CONTROL_INVALID, pair=spec.control, sign_median=None)
    v = pair_verdict(records, x, spec, spec.control)
    if v["status"] in (INVALID, STOP_MACHINE, NOT_CONSTRUCTIBLE):
        return dict(status=v["status"], pair=spec.control, sign_median=None, reasons=v["reasons"])
    status = OK if v["sign_median"] >= spec.sign_min else STOP_PROTOCOL
    return dict(status=status, pair=spec.control, sign_median=v["sign_median"])


def false_pass(t: int, m: int, q: float, rho: float) -> float:
    if rho <= 0:
        from scipy.stats import binom
        return float(binom.sf(t - 1, m, q))
    ab = (1 - rho) / rho
    return float(betabinom.sf(t - 1, m, q * ab, (1 - q) * ab))


def threshold_t(m: int, spec) -> int | None:
    t = math.ceil(2 * m / 3)
    while t <= m and false_pass(t, m, spec.q_null, spec.rho) > spec.fp_cap:
        t += 1
    return t if t <= m else None


def oc_table(spec) -> dict:
    qs = (0.2, 0.4, 0.6, 0.8, 0.9)
    return {str(m): {"t": threshold_t(m, spec),
                     "p_pass": {str(q): [false_pass(threshold_t(m, spec), m, q, 1e-4),
                                         false_pass(threshold_t(m, spec), m, q, spec.rho)] for q in qs}}
            for m in range(spec.min_pairs, spec.n_pairs + 1)}


def overall(control: dict, pairs: dict, spec) -> dict:
    statuses = [p["status"] for p in pairs.values()]
    m = len(pairs)
    base = dict(m=m, t=None, n_pass=statuses.count(PASS))
    if STOP_MACHINE in statuses or control["status"] == STOP_MACHINE:
        return dict(base, status=STOP_MACHINE)
    for st in (INVALID, NOT_CONSTRUCTIBLE):
        if control["status"] == st:
            return dict(base, status=st)
    if control["status"] in (STOP_CONTROL_INVALID, STOP_PROTOCOL):
        return dict(base, status=control["status"])
    for st in (INVALID, NOT_CONSTRUCTIBLE):
        if st in statuses:
            return dict(base, status=st)
    if m < spec.min_pairs:
        return dict(base, status=STOP_FEW_PAIRS)
    t = threshold_t(m, spec)
    return dict(base, t=t, status=PASS if base["n_pass"] >= t else FAIL)
```

Before relying on `check_records`, read `flymon/brain/b_rules.py:45-75` and confirm its `brains` argument handles `N2` like any non-noplast brain (range(n_flies)); if it rejects unknown brain names, report it (do not modify `b_rules.py`).

- [ ] **Step 5: Run to verify pass** — same command → PASS. If a fixture needs different `noise`/`dx` values to hit its intended branch, change the fixture values, never the rule.
- [ ] **Step 6: Commit** — `git add flymon/rescope/rules.py tests/rescope/rescope_fixtures.py tests/rescope/test_rules.py tests/rescope/test_rules_mutations.py && git commit -m "feat(rescope): primary verdict rules — reward association vs N and N' null, |e_Y|/e_X spill with e_X<=0 -> inf, X/Y floor, state propagation, t(m) at false-PASS cap 0.10, mutation tests"`

---

### Task 4: Reward-only oracle job and qualification decision

**Files:**
- Create: `flymon/rescope/oracle.py`
- Test: `tests/rescope/test_oracle.py`

**Interfaces:**
- Consumes: `flymon/brain/h4_jobs.py` (`rig_for`, the KC-activity helper `_present_kc`, `decide`, the `oracle_job` body at `:103-200` — read it and copy the reward half); `flymon.brain.h4_formula.dv(probe, z)`; `flymon.brain.b_rules.dprime`; `tests/conftest.py` fixtures `synthetic_connectome`, `synthetic_npz`.
- Produces:
  - `reward_oracle_job(eng, pl, pops, comps, ro, params, odor_x, odor_y, readout, z, types, act_seeds, select_seeds, report_seeds, alphas, strength, settle_ms, read_ms, window_ms, reward_type) -> dict` with keys `alpha_reward`, `select` (`{"pre", "reward": {str(a): {"R1", "change"}}}`), `report` (`{"pre", "R1"}`, same shape as `oracle_job`'s `report`), `kc`.
  - `qualify(naive_x_rule_x: str, report: dict, naive: dict, spec, z) -> dict` keys `qualified: bool, r: float, floor_ok: bool, x: str, reasons: list`, where `naive = {"a": [P counts per select+report seed], "b": [...]}` and `r = dprime(dv(report["R1"], z) - dv(report["pre"], z))`.
  - `qual_x(naive: dict, spec) -> str` (MBON05 median rule, tie → `spec.x_tie`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_oracle.py
import numpy as np
import pytest
from flymon.brain.config import Params
from flymon.brain.h4_jobs import oracle_job
from flymon.rescope.oracle import reward_oracle_job, qualify, qual_x
from flymon.rescope.spec import SPEC

def test_reward_half_equals_oracle_job(synthetic_connectome):
    """The accepted copy: same alpha_reward, same select reward block, same report pre/R1 as oracle_job."""
    rig, kw = _oracle_kwargs(synthetic_connectome)   # helper defined in this test module, mirroring test_h4_jobs
    full = oracle_job(*rig, **kw)
    kw_r = {k: v for k, v in kw.items() if k != "punish_type"}
    mine = reward_oracle_job(*rig, **kw_r)
    assert mine["alpha_reward"] == full["alpha_reward"]
    assert mine["select"]["reward"] == full["select"]["reward"]
    assert mine["report"]["pre"] == full["report"]["pre"] and mine["report"]["R1"] == full["report"]["R1"]

def test_qualify_decision():
    z = {"A": SPEC.z_a, "P": SPEC.z_p}
    pre = {"A": [[10, 10]] * 8, "P": [[30, 30]] * 8}
    r1 = {"A": [[10, 10]] * 8, "P": [[30 - 12 - (i % 2), 30] for i in range(8)]}
    naive = {"a": [30] * 16, "b": [30] * 16}
    q = qualify("b", {"pre": pre, "R1": r1}, naive, SPEC, z)
    assert q["qualified"] and q["r"] >= 2.0 and q["floor_ok"]
    q2 = qualify("b", {"pre": pre, "R1": r1}, {"a": [30] * 16, "b": [4] + [30] * 15}, SPEC, z)
    assert not q2["qualified"] and not q2["floor_ok"]      # Y (or X) floor on any seed fails
    flat = {"A": [[10, 10]] * 8, "P": [[30, 30]] * 8}
    assert not qualify("b", {"pre": pre, "R1": flat}, naive, SPEC, z)["qualified"]

def test_qual_x_rule():
    assert qual_x({"a": [40] * 16, "b": [30] * 16}, SPEC) == "a"
    assert qual_x({"a": [30] * 16, "b": [30] * 16}, SPEC) == SPEC.x_tie
```

Write `_oracle_kwargs(synthetic_connectome)` by copying the setup `tests/brain/test_h4_jobs.py` uses to call `oracle_job` on the synthetic connectome (same readout/z/types/seeds/alphas/windows), returning `(rig_tuple, kwargs_dict)`.

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_oracle.py` → FAIL (module missing).

- [ ] **Step 3: Implement**

Copy `oracle_job` from `flymon/brain/h4_jobs.py:103` into `flymon/rescope/oracle.py` as `reward_oracle_job`, then delete: the `punish_type` parameter, the `pun` mask, the punish α sweep, and the R2 report/counts. Keep the reward mask (`np.isin(p.post_mb, p.mb_local[c[reward_type].core])`), the X-KC-activity edit `w0 * (1 - a_r * fx)`, the α choice (argmax change, ties to the smaller α) on `select_seeds`, the report (pre, R1) on `report_seeds`, the `kc` block, and the `finally` that resets weights and re-enables plasticity. Put this docstring on it: `"""Reward half of flymon.brain.h4_jobs.oracle_job (accepted copy, pinned by tests/rescope/test_oracle.py): the G.14.3 'freq' edit on reward-core KC->MBON edges only."""`

```python
# the pure part, in the same module
from ..brain.b_rules import dprime
from ..brain.h4_formula import dv


def qual_x(naive: dict, spec) -> str:
    a, b = float(np.median(naive["a"])), float(np.median(naive["b"]))
    return spec.x_tie if a == b else ("a" if a > b else "b")


def qualify(x: str, report: dict, naive: dict, spec, z) -> dict:
    floor_ok = min(min(naive["a"]), min(naive["b"])) >= spec.floor_spikes
    r = dprime(dv(report["R1"], z) - dv(report["pre"], z))
    reasons = ([] if floor_ok else ["naive MBON05 below floor on some seed"]) + \
              ([] if (r is not None and r >= spec.oracle_min) else [f"oracle reward change d' {r} < {spec.oracle_min}"])
    return dict(qualified=not reasons, r=r, floor_ok=floor_ok, x=x, reasons=reasons)
```

Check `h4_formula.dv`'s expected probe shape (`{"A": [[x, y]...], "P": [...]}` keyed by readout role vs by type) against what `oracle_job` puts in `report`, and adapt the `qualify` test inputs to that exact shape.

- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add flymon/rescope/oracle.py tests/rescope/test_oracle.py && git commit -m "feat(rescope): reward-only oracle job (accepted copy of oracle_job's reward half, pinned) and the qualification decision"`

---

### Task 5: Primary pair runner and the qualification / primary CLIs

**Files:**
- Create: `flymon/rescope/primary.py`, `scripts/run_rescope_qualify.py`, `scripts/run_rescope_primary.py`
- Test: `tests/rescope/test_primary.py`, `tests/rescope/test_cli_primary.py`

**Interfaces:**
- Consumes: `flymon.brain.b_runner.probe(pool, spec, pair, lay, odors, cells, stage)`, `b_runner.train(pool, spec, pair, lay, x_odor, trial)`, `b_runner.steps(spec)`, `b_runner.fly_specs(lay)`; `rules.choose_x`, `rules.pair_verdict`, `rules.control_verdict`, `rules.overall`, `rules.oc_table`; `oracle.reward_oracle_job`, `oracle.qualify`, `oracle.qual_x`; `store.Checkpoint`, `store.write_json`, `store.git_provenance`; `pairs.pair_odors`; `flymon.agent.config.load_c3_config`; `FlyPool(npz, params, flies, workers, timeout_s)` and `FlyPool.run_jobs(fn, kwargs_list)`; `h4_jobs.type_cells(conn, types)`; the `FakePool` in `tests/brain/test_b_runner.py:18` (copy it into the new test module).
- Produces:
  - `layout(spec) -> list[tuple[str, int]]` = `[Rr×n] + [N×n] + [N2×n] + [("noplast", 0)]`.
  - `run_pair(pool, spec, pair, odors, cells, checkpoint=None, log=print, provenance=None) -> dict` keys `pair, x, layout, records, provenance, replayed`.
  - CLI `scripts/run_rescope_qualify.py --npz data/malecns.npz --workers 16 --out results/rescope/qualify [--smoke] [--allow-dirty]` → writes `results/summary/rescope_qualify.json` (`{pairs: {name: qualify-dict + naive + alpha_reward}, qualified: [names], m, stop: "STOP_FEW_PAIRS"|None, control_qualified: bool, oc_table, provenance}`; smoke writes under `results/rescope-smoke/`).
  - CLI `scripts/run_rescope_primary.py --npz ... --workers 16 --out results/rescope/primary [--smoke] [--allow-dirty]` → refuses unless `rescope_qualify.json` exists with `stop is None` and `control_qualified`; runs `seed0` then every qualified pair; writes `results/summary/rescope_primary.json` (`{control, pairs, overall, oc_table, recorded: {level, sign, naive, qual_x_vs_x}, provenance}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_primary.py
import dataclasses
from flymon.rescope import primary, rules
from flymon.rescope.spec import SPEC
# copy class FakePool from tests/brain/test_b_runner.py:18 here (unchanged)

SMALL = dataclasses.replace(SPEC, n_flies=2, n_probe=2, trials=3, valid_min=1)

def test_layout_has_no_rp_and_has_n2():
    lay = primary.layout(SMALL)
    assert [b for b, _ in lay] == ["Rr", "Rr", "N", "N", "N2", "N2", "noplast"]

def test_run_pair_dans_and_seeds():
    pool = FakePool(len(primary.layout(SMALL)))
    out = primary.run_pair(pool, SMALL, "p1000", ODORS, CELLS)
    dans = {(i, dan) for (i, dan, _, _) in pool.reinforced}
    lay = out["layout"]
    for i, (b, f) in enumerate(lay):
        want = {"Rr": "PAM08", "noplast": "PAM08", "N": None, "N2": None}[b]
        assert all(d == want for (j, d) in dans if j == i)
    n2_seeds = {s for (i, _, _, s) in pool.reinforced if lay[i][0] == "N2"}
    assert min(n2_seeds) >= SPEC.train_base_n2
    assert {r["stage"] for r in out["records"]} == {"pre", "S1"}
    assert out["x"] == rules.choose_x(out["records"], SMALL)

def test_resume_gives_same_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from flymon.brain.config import Params
    from flymon.rescope.store import Checkpoint
    full = primary.run_pair(FakePool(7), SMALL, "p1000", ODORS, CELLS)
    ck = Checkpoint("results/rescope/p1000", "k", [Params()])
    broken = FakePool(7, fail_at=2)
    try:
        primary.run_pair(broken, SMALL, "p1000", ODORS, CELLS, checkpoint=ck)
    except RuntimeError:
        pass
    resumed = primary.run_pair(FakePool(7), SMALL, "p1000", ODORS, CELLS, checkpoint=ck)
    assert resumed["records"] == full["records"] and resumed["x"] == full["x"]
```

Define `ODORS = {"a": {"ORN_DM1": 1.0}, "b": {"ORN_VA2": 1.0}}` and `CELLS = {"MBON13": np.array([0]), "MBON05": np.array([1])}` (or whatever shape the copied FakePool expects — read it). If FakePool's `fail_at` raises a different exception type, catch that one.

```python
# tests/rescope/test_cli_primary.py
import importlib.util, json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_primary_refuses_without_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])

def test_primary_refuses_on_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": "STOP_FEW_PAIRS", "control_qualified": True, "qualified": []}))
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])

def test_qualify_refuses_outside_rescope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load("run_rescope_qualify").main(["--out", "results/b/x", "--allow-dirty"])
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_primary.py tests/rescope/test_cli_primary.py` → FAIL.

- [ ] **Step 3: Implement `flymon/rescope/primary.py`**

```python
"""Spec 10.2's primary protocol: independent arms Rr / N / N' (brain name N2) / noplast from naive, pre -> X -> 20 trials
-> S1, on b_runner.probe / b_runner.train (used unmodified). No punishment arm."""
from __future__ import annotations

from ..brain.b_runner import probe, steps, train
from .rules import choose_x


def layout(spec) -> list:
    n = range(spec.n_flies)
    return [("Rr", f) for f in n] + [("N", f) for f in n] + [("N2", f) for f in n] + [("noplast", 0)]


def run_pair(pool, spec, pair, odors, cells, checkpoint=None, log=print, provenance=None) -> dict:
    lay = layout(spec)
    st = (checkpoint.load() if checkpoint else None) or dict(records=[], x=None, done=[])
    if not st["done"]:
        st["provenance"] = provenance
    replayed = all(s in st["done"] for s in steps(spec))
    if st["done"] and "weights" in st:
        pool.load_state(st["weights"])
    for step in steps(spec):
        if step in st["done"]:
            continue
        if step in ("pre", "S1"):
            st["records"] += probe(pool, spec, pair, lay, odors, cells, step)
            if step == "pre":
                st["x"] = choose_x(st["records"], spec)
                log(f"{pair}: X = odour {st['x']}")
        else:
            train(pool, spec, pair, lay, odors[st["x"]], int(step.split(":")[1]))
        st["done"].append(step)
        if checkpoint:
            checkpoint.save(dict(st, weights=pool.state()))
        log(f"{pair}: {step} done")
    return dict(pair=pair, x=st["x"], layout=lay, records=st["records"], provenance=st.get("provenance"),
                replayed=replayed)
```

- [ ] **Step 4: Implement the two CLIs**

Both scripts: `main(argv=None)` with argparse; `--smoke` replaces the spec with `dataclasses.replace(SPEC, n_flies=2, n_probe=2, trials=2, valid_min=1, n_qual_seeds=2)` and forces `--out` under `results/rescope-smoke/`; `--allow-dirty` skips the clean-tree refusal (otherwise `git_provenance(files=[flymon/rescope/*.py, the script])["dirty"]` → `SystemExit`). Load C3 with `cfg = load_c3_config()`; connectome/pops with `Connectome.load(npz)`, `Populations.from_connectome`; cells with `h4_jobs.type_cells(conn, [SPEC.a_type, SPEC.p_type])`.

`run_rescope_qualify.py` flow, per pair name in `SPEC.pair_names()`:
1. `odors = pair_odors(pops, name, spec)`.
2. Naive probe: with a `FlyPool(npz, cfg.params, [FlySpec()], workers)` run `decide_batch([(0, [odors["a"], odors["b"]], s)], spec.strength, spec.probe_settle_ms, spec.probe_read_ms, idx=cells[P])` for every `s` in `qual_seeds(name)["select"] + ["report"]`; `naive = {"a": [sum P counts], "b": [...]}`.
3. `x = qual_x(naive, spec)`.
4. Oracle: `pool.run_jobs(reward_oracle_job, [dict(params=cfg.params, odor_x=odors[x], odor_y=odors[other], readout={"A": SPEC.a_type, "P": SPEC.p_type}, z={"A": SPEC.z_a, "P": SPEC.z_p}, types=[SPEC.a_type, SPEC.p_type], act_seeds=..., select_seeds=..., report_seeds=..., alphas=SPEC.oracle_alphas, strength=SPEC.strength, settle_ms=SPEC.oracle_settle_ms, read_ms=SPEC.oracle_read_ms, window_ms=SPEC.kc_window_ms, reward_type=SPEC.reward_dan)])[0]`. (Pool built with `Params()` as in the L scripts, since the job takes C3 per call — read `scripts/run_l_stage0.py:177-196` for the exact pool construction and copy it.)
5. `qualify(x, row["report"], naive, spec, z)`.
Then `qualified = [n for n in pair_names()[1:] if ok]`, `m = len(qualified)`, `stop = "STOP_FEW_PAIRS" if m < spec.min_pairs else None`, `control_qualified = pairs["seed0"]["qualified"]`, write the summary (also `oc_table(spec)`, `pairs_digest`, provenance). Print one line per pair and the stop state.

`run_rescope_primary.py` flow: read `results/summary/rescope_qualify.json` (smoke: the smoke one); refuse (`SystemExit`) if missing, `stop` set, or not `control_qualified`. Build `pool = FlyPool(npz, cfg.params, [FlySpec(**d) for d in b_runner.fly_specs(primary.layout(spec))], workers, timeout_s=1800)`. For `seed0` and each qualified pair: `Checkpoint(out/pair, f"{commit}-{pair}", [cfg.params])`, `run_pair(...)`, then `pair_verdict` (control: `control_verdict(records, x, spec, qualified=True)`); between pairs reset weights with `pool.load_state` of the pool's initial `state()` captured before the first pair. Write raw records to `out/<pair>/records.json` and the summary with `overall(control, pairs, spec)`; `recorded` holds level/sign medians, naive medians, and whether the judgement X equals the qualification X.

- [ ] **Step 5: Run to verify pass** — `uv run pytest -q -rfE -o addopts="" tests/rescope/` → PASS.
- [ ] **Step 6: Commit** — `git add flymon/rescope/primary.py scripts/run_rescope_qualify.py scripts/run_rescope_primary.py tests/rescope/test_primary.py tests/rescope/test_cli_primary.py && git commit -m "feat(rescope): primary pair runner (Rr/N/N'/noplast on b_runner, resumable) and qualification/primary CLIs with smoke mode and refusals"`

---

### Task 6: τ_rec synthetic calibration (stage 1)

**Files:**
- Create: `flymon/rescope/taurec.py`, `scripts/run_rescope_taurec.py`
- Test: `tests/rescope/test_taurec.py`

**Interfaces:**
- Consumes: `flymon.agent.encode.Encoder(pops)` (`species_types`, `move_info`, `chan`), `flymon.brain.h4_pairs.odour(pops, chan, my_types, opp_types, move, bp, my_hp, opp_hp)`, `flymon.battle.pool.POOL`; `FlyPool(npz, params, flies, workers)`, `reinforce_batch`, `decide_batch(idx=pops.kc)`, `w`, `w0[None]`, `run_jobs`; `flymon.agent.jobs.edge_compartments_job` (read its signature at `flymon/agent/jobs.py`); `dataclasses.replace(params, recovery_per_pulse=r)`.
- Produces:
  - `synthetic_odours(pops, n, seed) -> list[dict]` (random my/opp species from POOL, one of my attacks, hp fractions uniform in (0, 1]).
  - `pulse_plan(spec) -> list[tuple[str, float]]` (length `taurec_pulses`, alternating `(reward_dan, taurec_reward_ms)`, `(punish_dan, taurec_punish_ms)`, starting with reward).
  - `trajectory(pool, odours, plan, spec, taught_mask) -> dict` keys `alt: {"ratio": [...], "q10_taught": [...], "floor_frac_taught": [...]}`, `same: {...}` (fly 0 cycles odours, fly 1 repeats `odours[0]`), sampled every `taurec_sample_every` pulses and after the last.
  - `select(results: dict[float, dict], spec) -> dict` keys `status` (`"SELECTED"` or `"STOP_NO_RECOVERY"`), `recovery_per_pulse`, `path_min` per grid value.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_taurec.py
import numpy as np
import pytest
from flymon.rescope import taurec
from flymon.rescope.spec import SPEC

def test_pulse_plan():
    plan = taurec.pulse_plan(SPEC)
    assert len(plan) == 1000
    assert plan[0] == ("PAM08", 600.0) and plan[1] == ("PPL105", 400.0)
    assert sum(1 for d, _ in plan if d == "PAM08") == 500

def test_select_smallest_passing():
    res = {0.0: {"alt": {"ratio": [1.0, 0.45]}}, 0.001: {"alt": {"ratio": [1.0, 0.6, 0.55]}},
           0.002: {"alt": {"ratio": [1.0, 0.9]}}}
    out = taurec.select(res, SPEC)
    assert out["status"] == "SELECTED" and out["recovery_per_pulse"] == 0.001
    assert out["path_min"]["0.0"] == 0.45

def test_select_zero_when_zero_passes():
    assert taurec.select({0.0: {"alt": {"ratio": [1.0, 0.7]}}, 0.001: {"alt": {"ratio": [1.0, 0.9]}}}, SPEC)["recovery_per_pulse"] == 0.0

def test_select_stop():
    assert taurec.select({0.0: {"alt": {"ratio": [0.3]}}, 0.02: {"alt": {"ratio": [0.4]}}}, SPEC)["status"] == "STOP_NO_RECOVERY"

def test_trajectory_on_synthetic(synthetic_npz):
    from flymon.brain.config import Params
    from flymon.brain.fly_pool import FlyPool, FlySpec
    small = __import__("dataclasses").replace(SPEC, taurec_pulses=6, taurec_sample_every=2, taurec_odours=3)
    odours = [{"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}, {"ORN_DM6": 1.0}]
    with FlyPool(synthetic_npz, Params(), [FlySpec(), FlySpec()], workers=1) as pool:
        mask = np.ones(len(pool.w[0]), dtype=bool)
        out = taurec.trajectory(pool, odours, taurec.pulse_plan(small), small, mask)
    assert len(out["alt"]["ratio"]) == 4 and out["alt"]["ratio"][0] <= 1.0   # samples at 2, 4, 6 and the end marker
```

Adjust the expected sample count in the last test to the sampling rule you implement (samples after pulses 2, 4, 6 → 3 samples if the last coincides; document the rule in the docstring and make the test match it exactly). Use the synthetic connectome's receptor type names from `tests/conftest.py` (`ORN_DM1`, `ORN_DA1`, `ORN_VA2`, `ORN_DM6`, `ORN_VC1`) and DAN names `PAM08`/`PPL105` (present in the synthetic build).

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_taurec.py` → FAIL.

- [ ] **Step 3: Implement `flymon/rescope/taurec.py`**

```python
"""Stage 1 (spec 10.6): pick recovery_per_pulse without battles — 1,000 synthetic pulses, the runtime safety population
(median w/w0 over all plastic KC->MBON edges, flymon.agent.swarm.BrainSwarm.median_ratio), path minimum >= 0.5."""
from __future__ import annotations

import numpy as np

from ..agent.encode import Encoder
from ..battle.pool import POOL
from ..brain import h4_pairs


def synthetic_odours(pops, n: int, seed: int) -> list:
    enc, rng, out = Encoder(pops), np.random.default_rng(seed), []
    for _ in range(n):
        me, opp = rng.choice(len(POOL), 2, replace=True)
        mon = POOL[int(me)]
        move = mon.attacks[int(rng.integers(len(mon.attacks)))]
        mtype, bp = enc.move_info[move]
        out.append(h4_pairs.odour(pops, enc.chan, list(enc.species_types[mon.species]),
                                  list(enc.species_types[POOL[int(opp)].species]), mtype, bp,
                                  float(rng.uniform(0.01, 1.0)), float(rng.uniform(0.01, 1.0))))
    return out


def pulse_plan(spec) -> list:
    one = [(spec.reward_dan, spec.taurec_reward_ms), (spec.punish_dan, spec.taurec_punish_ms)]
    return [one[i % 2] for i in range(spec.taurec_pulses)]


def _sample(pool, fly, mask) -> tuple:
    r = pool.w[fly] / pool.w0[None]
    taught = r[mask]
    return float(np.median(r)), float(np.quantile(taught, 0.1)), float(np.mean(taught <= 0.2 + 1e-9))


def trajectory(pool, odours, plan, spec, taught_mask) -> dict:
    out = {k: {"ratio": [], "q10_taught": [], "floor_frac_taught": []} for k in ("alt", "same")}
    for i, (dan, ms) in enumerate(plan):
        seed = spec.taurec_seed_base + i
        pool.reinforce_batch([(0, odours[i % len(odours)], dan, ms, seed), (1, odours[0], dan, ms, seed)], spec.strength)
        if (i + 1) % spec.taurec_sample_every == 0 or i + 1 == len(plan):
            for fly, key in ((0, "alt"), (1, "same")):
                med, q10, fl = _sample(pool, fly, taught_mask)
                out[key]["ratio"].append(med); out[key]["q10_taught"].append(q10); out[key]["floor_frac_taught"].append(fl)
    return out


def select(results: dict, spec) -> dict:
    path_min = {str(r): float(min(res["alt"]["ratio"])) for r, res in results.items()}
    ok = sorted(r for r, res in results.items() if min(res["alt"]["ratio"]) >= spec.median_floor)
    if not ok:
        return dict(status="STOP_NO_RECOVERY", recovery_per_pulse=None, path_min=path_min)
    return dict(status="SELECTED", recovery_per_pulse=ok[0], path_min=path_min)
```

Check `synthetic_odours` against `Encoder`: `species_types` is keyed by species name and `move_info` by the attack name as written in `POOL` (confirm in `flymon/brain/h4_pairs.py:26 pool_vocabulary`); adapt the keys if they differ. `pool.w0` is keyed by `shuffle_seed` (`None` here) — confirm in `fly_pool.py`.

- [ ] **Step 4: Implement `scripts/run_rescope_taurec.py`**

`main(argv)`: `--npz`, `--out results/rescope/taurec`, `--smoke` (spec with `taurec_pulses=20`, `taurec_sample_every=5`, `taurec_odours=4`, grid `(0.0, 0.02)`, out under `results/rescope-smoke/`), `--allow-dirty`. Steps: `cfg = load_c3_config()`; `odours = synthetic_odours(pops, spec.taurec_odours, spec.taurec_gen_seed)`; taught mask: a one-fly pool, `decide_batch([(0, [odours[0]], spec.taurec_seed_base - 1)], strength, 800, 600, idx=pops.kc)` → active KC set; edge masks via `pool.run_jobs(edge_compartments_job, [dict(types=[reward_dan, punish_dan])])` plus the edges' pre-KC index (`pl.pre_kc`, obtained with a tiny module-level job `pre_kc_job(eng, pl, pops, comps, ro) -> pl.pre_kc` in `taurec.py`); `taught = (mask_PAM08 | mask_PPL105) & isin(pre_kc, active)`. For each `r` in the grid: `FlyPool(npz, replace(cfg.params, recovery_per_pulse=r), [FlySpec(), FlySpec()], workers=2)` → `trajectory(...)`. Then `select`. Write `results/summary/rescope_taurec.json`: `{status, recovery_per_pulse, path_min, trajectories, odours_seed, n_taught_edges, provenance}`. Print the selected value or `STOP_NO_RECOVERY`.

Add to `tests/rescope/test_taurec.py`: `test_cli_refuses_outside_rescope` (same pattern as Task 5's CLI test).

- [ ] **Step 5: Run to verify pass** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_taurec.py` → PASS.
- [ ] **Step 6: Commit** — `git add flymon/rescope/taurec.py scripts/run_rescope_taurec.py tests/rescope/test_taurec.py && git commit -m "feat(rescope): stage 1 tau_rec — synthetic odours from the battle encoder, 1,000 alternating pulses, runtime-population path minimum, same-odour trajectory and taught-KC quantiles recorded"`

---

### Task 7: Yoked pulse queue and the FLY-RS player

**Files:**
- Create: `flymon/rescope/yoke.py`, `flymon/rescope/players.py`
- Test: `tests/rescope/test_yoke.py`, `tests/rescope/test_players.py`

**Interfaces:**
- Consumes: FLY log records `kind == "reinforce"` with `pulses: [[dan, ms], ...]` in `logs/flyNN.jsonl` (written by `AgentPlayer._on_outcome`, `flymon/agent/player.py:104-115`); `AgentPlayer` internals `_choice`, `_queue`, `no_signal`, `_write`, `fly`, `battle_id`; `policy.derive_seed`; test helpers from `tests/battle/test_agent_player.py` (`FakeEncoder`, `make_player`, `stub_forfeit`) and `Outcome` from `flymon.battle.attribution`.
- Produces:
  - `class YokedQueue`: `from_log(path, battle_ids: set[str]) -> YokedQueue` (bundles in log order, only learning-block battles, only non-empty `pulses`), `pop() -> list[tuple[str, float]] | None`, `total_ms() -> float`, `delivered_ms() -> float`, `residual_frac() -> float` (= remaining ms / donor total ms), `donor_sha256: str`, `state() -> dict`, `load_state(d)` (index into the list; for checkpoint/resume).
  - `class YokedPlayer(AgentPlayer)`: `__init__(fly, encoder, table, rbarrier, queue: YokedQueue, **kw)`; overrides `_on_outcome(tag, turn, outcome)` to deliver `queue.pop()` instead of `pulses_for(outcome)` on a fly turn (outcome ignored except for logging); log record adds `yoked: true` and `donor_sha256`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_yoke.py
import json
from flymon.rescope.yoke import YokedQueue

def _log(tmp_path, rows):
    p = tmp_path / "fly00.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p

def test_queue_keeps_nonempty_learning_bundles_in_order(tmp_path):
    rows = [dict(kind="reinforce", battle_id="L-f00-b000", pulses=[["PAM08", 400.0]]),
            dict(kind="reinforce", battle_id="L-f00-b000", pulses=[]),
            dict(kind="decision", battle_id="L-f00-b000"),
            dict(kind="reinforce", battle_id="L-f00-b001", pulses=[["PAM08", 200.0], ["PPL105", 200.0]]),
            dict(kind="reinforce", battle_id="E-f00-b000", pulses=[["PAM08", 400.0]])]
    q = YokedQueue.from_log(_log(tmp_path, rows), {"L-f00-b000", "L-f00-b001"})
    assert q.total_ms() == 800.0
    assert q.pop() == [("PAM08", 400.0)]
    assert q.pop() == [("PAM08", 200.0), ("PPL105", 200.0)]
    assert q.pop() is None and q.residual_frac() == 0.0

def test_residual_and_resume(tmp_path):
    rows = [dict(kind="reinforce", battle_id="L-f00-b000", pulses=[["PAM08", 100.0]]) for _ in range(20)]
    q = YokedQueue.from_log(_log(tmp_path, rows), {"L-f00-b000"})
    for _ in range(19): q.pop()
    assert abs(q.residual_frac() - 0.05) < 1e-12
    st = q.state()
    q2 = YokedQueue.from_log(_log(tmp_path, rows), {"L-f00-b000"}); q2.load_state(st)
    assert q2.pop() == [("PAM08", 100.0)] and q2.pop() is None

def test_sha_mismatch_refused(tmp_path):
    rows = [dict(kind="reinforce", battle_id="L-f00-b000", pulses=[["PAM08", 100.0]])]
    q = YokedQueue.from_log(_log(tmp_path, rows), {"L-f00-b000"})
    st = q.state()
    _log(tmp_path, rows + rows)                       # donor rerun: log changed
    q3 = YokedQueue.from_log(tmp_path / "fly00.jsonl", {"L-f00-b000"})
    import pytest
    with pytest.raises(ValueError):
        q3.load_state(st)
```

```python
# tests/rescope/test_players.py
from flymon.battle.attribution import Outcome
from flymon.rescope.players import YokedPlayer
from flymon.rescope.yoke import YokedQueue
# reuse FakeEncoder / make_player pattern from tests/battle/test_agent_player.py (copy the helpers here, building a YokedPlayer)

def test_yoked_delivers_queue_not_outcome(tmp_path):
    q = YokedQueue.from_bundles([[("PPL105", 400.0)]], donor_sha256="x")   # test constructor
    p = make_yoked_player(tmp_path, q)
    tag, turn = "battle-gen1ou-1", 3
    p._choice[tag] = {"turn": turn, "odour": {"ORN_DM1": 1.0}, "move": "surf"}
    p._on_outcome(tag, turn, Outcome(dealt_frac=1.0, effectiveness="super"))   # outcome would give PAM08
    assert [(j["dan"], j["ms"]) for j in p.pending_pulses(tag)] == [("PPL105", 400.0)]

def test_empty_queue_gives_no_pulse(tmp_path):
    p = make_yoked_player(tmp_path, YokedQueue.from_bundles([], donor_sha256="x"))
    tag = "battle-gen1ou-1"
    p._choice[tag] = {"turn": 1, "odour": {"ORN_DM1": 1.0}, "move": "surf"}
    p._on_outcome(tag, 1, Outcome(dealt_frac=1.0, effectiveness="super"))
    assert p.pending_pulses(tag) == []

def test_coach_turn_does_not_consume(tmp_path):
    q = YokedQueue.from_bundles([[("PAM08", 400.0)]], donor_sha256="x")
    p = make_yoked_player(tmp_path, q)
    p._on_outcome("battle-gen1ou-1", 2, Outcome(dealt_frac=1.0, effectiveness="super"))   # no _choice: coach turn
    assert q.pop() == [("PAM08", 400.0)]
```

Add `YokedQueue.from_bundles(bundles, donor_sha256)` as a classmethod used by `from_log` and tests.

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_yoke.py tests/rescope/test_players.py` → FAIL.

- [ ] **Step 3: Implement**

```python
# flymon/rescope/yoke.py
"""FLY-RS pulse queue (spec 10.8): the donor FLY k's non-empty learning-block pulse bundles in order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class YokedQueue:
    def __init__(self, bundles: list, donor_sha256: str):
        self.bundles = [[(str(d), float(ms)) for d, ms in b] for b in bundles]
        self.donor_sha256 = donor_sha256
        self.i = 0

    @classmethod
    def from_bundles(cls, bundles, donor_sha256):
        return cls(bundles, donor_sha256)

    @classmethod
    def from_log(cls, path, battle_ids) -> "YokedQueue":
        raw = Path(path).read_bytes()
        bundles = []
        for line in raw.decode().splitlines():
            r = json.loads(line)
            if r.get("kind") == "reinforce" and r.get("battle_id") in battle_ids and r.get("pulses"):
                bundles.append(r["pulses"])
        return cls(bundles, hashlib.sha256(raw).hexdigest())

    def pop(self):
        if self.i >= len(self.bundles):
            return None
        self.i += 1
        return self.bundles[self.i - 1]

    def total_ms(self) -> float:
        return float(sum(ms for b in self.bundles for _, ms in b))

    def delivered_ms(self) -> float:
        return float(sum(ms for b in self.bundles[:self.i] for _, ms in b))

    def residual_frac(self) -> float:
        tot = self.total_ms()
        return 0.0 if tot == 0 else (tot - self.delivered_ms()) / tot

    def state(self) -> dict:
        return {"i": self.i, "donor_sha256": self.donor_sha256}

    def load_state(self, d: dict) -> None:
        if d["donor_sha256"] != self.donor_sha256:
            raise ValueError("donor log changed since this RS fly started (rerun FLY k and RS k together)")
        self.i = int(d["i"])
```

```python
# flymon/rescope/players.py
"""FLY-RS: an AgentPlayer whose fly-turn reinforcement comes from the donor's queue, not from its own outcome."""
from __future__ import annotations

from ..agent.player import AgentPlayer
from ..agent.policy import derive_seed


class YokedPlayer(AgentPlayer):
    def __init__(self, fly, encoder, table, rbarrier, queue, **kw):
        super().__init__(fly, encoder, table, rbarrier, **kw)
        self.yoke = queue

    def _on_outcome(self, tag, turn, outcome):
        choice = self._choice.pop(tag, None)
        if choice is None or choice["turn"] != turn:
            return
        pulses = self.yoke.pop() or []
        if not pulses:
            self.no_signal[tag] = self.no_signal.get(tag, 0) + 1
        for j, (dan, ms) in enumerate(pulses):
            self._queue.setdefault(tag, []).append(dict(fly=self.fly, odour=choice["odour"], dan=dan, ms=ms,
                                                        seed=derive_seed("reinforce", self.fly, self.battle_id, turn, j)))
        self._write(dict(kind="reinforce", battle_tag=tag, turn=turn, odour_move=choice["move"],
                         pulses=[[d, ms] for d, ms in pulses], yoked=True, donor_sha256=self.yoke.donor_sha256))
```

Before finalising, read `flymon/agent/player.py:97-131` and make `_on_outcome` mirror its exact bookkeeping (the `no_signal` container type, the queue item keys, the `_write` fields) so `logschema.validate` accepts the record; only the pulse source differs.

- [ ] **Step 4: Run to verify pass** — same command → PASS; also `uv run pytest -q -rfE -o addopts="" tests/battle/test_agent_player.py` stays green.
- [ ] **Step 5: Commit** — `git add flymon/rescope/yoke.py flymon/rescope/players.py tests/rescope/test_yoke.py tests/rescope/test_players.py && git commit -m "feat(rescope): yoked FLY-RS — donor log pulse queue (non-empty learning bundles, carry-over, residual, sha256-checked resume) and YokedPlayer"`

---

### Task 8: Two-block battle driver (learn → eval) and arms FLY / RS / C-off / RND / MAX

**Files:**
- Create: `flymon/rescope/blocks.py`, `scripts/run_rescope_battles.py`
- Test: `tests/rescope/test_blocks.py`, `tests/rescope/test_run_battles.py`

**Interfaces:**
- Consumes: `flymon.battle.schedule.make_schedule(n_flies, n_battles, opponent, seed)`, `ScheduledBattle`, `save/load`; `run_m3_smoke.py`'s `play_one` pattern (read `scripts/run_m3_smoke.py:87-160` and follow it); `BrainSwarm` (`mode`, `compartment_fracs`, `median_ratio`), `FlyPool.set_enabled`, `run_cohort`, `CheckpointStore`, `config_hash`; `YokedPlayer`, `YokedQueue`; providers `RandomProvider(seed)`, `MaxDamageProvider()` and `FlyCoachPlayer` (the no-brain arms, as in `scripts/pilot_no_brain.py:47`); `store.guard`; `taurec` summary for `recovery_per_pulse`.
- Produces:
  - `block_schedule(n_flies, n_battles, seed, block: "L"|"E") -> list[ScheduledBattle]` with ids `f"{block}-f{fly:02d}-b{b:03d}"`; `schedule_digest(sched) -> str`; `assert_disjoint(learn, eval_) -> None` (raises if any `(my_team, opp_team)` pair repeats across blocks).
  - `battle_index(battle_id) -> int` (= int of the last 3 chars).
  - CLI `scripts/run_rescope_battles.py --phase pilot|judge --arm FLY|RS|COFF|RND|MAX --flies F --eval E [--learn 40] --out results/rescope/<phase>/<arm> [--resume] [--smoke] [--workers 16] [--allow-dirty]`:
    - FLY: learn block (`mode="learn"`, plasticity on, pulses per `pulses_for`) then eval block (`swarm.mode = "eval"`, `pool.set_enabled(f, False)` for all flies, no pulses delivered).
    - RS: requires `results/rescope/<phase>/FLY/logs/flyNN.jsonl` for every fly k and FLY's learn block complete; learn block with `YokedPlayer` (queue from FLY k's log, learn battle ids), then eval block as FLY. Writes per fly `residual_frac` and `donor_sha256`; marks the pair INVALID in `result.json` when `residual_frac > spec.residual_max`.
    - COFF: eval block only, `FlySpec(enabled=False)`, `mode="eval"`.
    - RND / MAX: eval block only, `FlyCoachPlayer(RandomProvider(seed=fly) | MaxDamageProvider(), coach)`.
    - M4 arms load `recovery_per_pulse` from `results/summary/rescope_taurec.json` (`status == "SELECTED"`, else `SystemExit`) and build `FlyPool(npz, replace(cfg.params, recovery_per_pulse=r), ...)`.
    - Server-error retries: a battle that ends unfinished (`battle_stats(tag)["finished"] is False`) is replayed with the same schedule entry up to `spec.retry_max` times; each retry counted in the log; beyond that the fly is marked INVALID in `result.json`.
    - Writes `out/result.json`: `{phase, arm, flies, learn, eval, recovery_per_pulse, schedule_digests: {learn, eval}, per_fly: [{fly, eval_battles: [{battle_id, won, finished, retries}], invalid, residual_frac?, donor_sha256?, weights_bit_identical_across_eval}], wall_clock_s, m_overlap_note, provenance}`.
    - Schedules: learn seed / eval seed from `spec.schedule_seeds[phase]`; the eval schedule is the same file for every arm of a phase (`results/rescope/<phase>/eval_schedule.json`, created once, digest checked by every arm).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_blocks.py
from flymon.rescope.blocks import block_schedule, schedule_digest, assert_disjoint, battle_index

def test_block_ids_and_index():
    s = block_schedule(2, 3, 201, "L")
    assert [b.battle_id for b in s[:3]] == ["L-f00-b000", "L-f00-b001", "L-f00-b002"]
    assert battle_index("E-f01-b017") == 17

def test_eval_schedule_same_for_all_arms():
    assert schedule_digest(block_schedule(4, 5, 202, "E")) == schedule_digest(block_schedule(4, 5, 202, "E"))

def test_disjoint_detects_repeat():
    import pytest
    learn = block_schedule(2, 3, 201, "L")
    assert_disjoint(learn, block_schedule(2, 3, 202, "E"))
    with pytest.raises(ValueError):
        assert_disjoint(learn, [learn[0]])
```

```python
# tests/rescope/test_run_battles.py — no Showdown
import asyncio, importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from flymon.rescope import blocks
from flymon.rescope.spec import SPEC
from flymon.rescope.yoke import YokedQueue

ROOT = Path(__file__).resolve().parents[2]

def load():
    spec = importlib.util.spec_from_file_location("rb", ROOT / "scripts/run_rescope_battles.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_eval_block_freezes_weights(synthetic_npz):
    # Build the pool and swarm exactly as tests/agent/test_swarm.py does (copy its small AgentConfig and cells).
    from tests.agent.test_swarm import make_swarm          # if absent there, copy its setup into a local helper
    pool, swarm, odours = make_swarm(synthetic_npz)
    req = lambda b: SimpleNamespace(context=dict(battle_tag="t", turn=1, fly=0, battle_id=b, battle_index=0, k=0,
                                                 odours=odours))
    asyncio.run(swarm.reinforce_run_batch([SimpleNamespace(context=dict(fly=0, odour=odours[0], dan="PAM08",
                                                                        ms=400.0, seed=1))]))
    before = pool.w[0].copy()
    blocks.enter_eval(swarm, pool)
    asyncio.run(swarm.decide_run_batch([req("E-f00-b000")]))
    assert swarm.mode == "eval" and np.array_equal(pool.w[0], before)
    assert all(not f["enabled"] for f in pool.state()["flies"])

def test_rs_refuses_without_donor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "RS", "--flies", "2", "--eval", "2",
                     "--out", "results/rescope/pilot/RS", "--allow-dirty"])

def test_arm_requires_taurec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "FLY", "--flies", "2", "--eval", "2",
                     "--out", "results/rescope/pilot/FLY", "--allow-dirty"])

def test_rs_marks_invalid_on_residual():
    q = YokedQueue.from_bundles([[("PAM08", 94.0)], [("PAM08", 6.0)]], donor_sha256="x"); q.pop()
    assert blocks.fly_result(0, [], queue=q, spec=SPEC)["invalid"] is True            # residual 0.06
    q2 = YokedQueue.from_bundles([[("PAM08", 95.0)], [("PAM08", 5.0)]], donor_sha256="x"); q2.pop()
    assert blocks.fly_result(0, [], queue=q2, spec=SPEC)["invalid"] is False          # residual 0.05

def test_retry_limit():
    calls = []
    async def never(sb): calls.append(1); return {"finished": False}
    out = asyncio.run(blocks.play_with_retries(never, None, 3))
    assert out["invalid"] and out["retries"] == 3 and len(calls) == 4
    seq = iter([False, False, True])
    async def third(sb): return {"finished": next(seq)}
    out = asyncio.run(blocks.play_with_retries(third, None, 3))
    assert not out["invalid"] and out["retries"] == 2

def test_eval_schedule_digest_mismatch_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = tmp_path / "results/summary/rescope_taurec.json"; t.parent.mkdir(parents=True)
    t.write_text(json.dumps({"status": "SELECTED", "recovery_per_pulse": 0.0}))
    e = tmp_path / "results/rescope/pilot/eval_schedule.json"; e.parent.mkdir(parents=True)
    e.write_text(json.dumps({"digest": "not-the-one", "schedule": []}))
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "RND", "--flies", "2", "--eval", "2",
                     "--out", "results/rescope/pilot/RND", "--allow-dirty"])
```

`make_swarm(synthetic_npz) -> (pool, swarm, odours)`: if `tests/agent/test_swarm.py` has no such helper, write it in this test module by copying that file's pool/`AgentConfig`/`BrainSwarm` setup (2 flies, synthetic readout MBON03/MBON01); the reinforce request shape must match what `BrainSwarm.reinforce_run_batch` reads (check `flymon/agent/swarm.py:53`). The CLI must check the eval-schedule digest (and the taurec summary for brain arms) **before** starting any server, so these refusal tests never need Showdown. Expose from `blocks.py`: `enter_eval(swarm, pool) -> None`, `fly_result(fly, eval_records, queue=None, spec=None, invalid=False) -> dict`, `async play_with_retries(play, sb, retry_max) -> dict`. `eval_schedule.json` format: `{"digest": str, "schedule": [ScheduledBattle as dict]}`.

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_blocks.py tests/rescope/test_run_battles.py` → FAIL.

- [ ] **Step 3: Implement `flymon/rescope/blocks.py`**

```python
"""Learn / eval blocks for the M4 arms (spec 4.5, 10.7-10.8)."""
from __future__ import annotations

import dataclasses
import hashlib
import json

from ..battle.schedule import make_schedule


def block_schedule(n_flies, n_battles, seed, block) -> list:
    return [dataclasses.replace(sb, battle_id=f"{block}-{sb.battle_id}")
            for sb in make_schedule(n_flies, n_battles, "heuristic", seed=seed)]


def schedule_digest(sched) -> str:
    rows = [dict(battle_id=s.battle_id, fly_id=s.fly_id, my_team=list(s.my_team), opp_team=list(s.opp_team),
                 opponent=s.opponent) for s in sched]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_disjoint(learn, eval_) -> None:
    key = lambda s: (tuple(s.my_team), tuple(s.opp_team))
    both = {key(s) for s in learn} & {key(s) for s in eval_}
    if both:
        raise ValueError(f"{len(both)} learn/eval battles share teams and order")


def battle_index(battle_id: str) -> int:
    return int(battle_id[-3:])


def enter_eval(swarm, pool) -> None:
    for f in range(pool.n_flies):
        pool.set_enabled(f, False)
    swarm.mode = "eval"


def fly_result(fly, eval_records, queue=None, spec=None, invalid=False) -> dict:
    out = dict(fly=fly, eval_battles=eval_records, invalid=bool(invalid))
    if queue is not None:
        out.update(residual_frac=queue.residual_frac(), donor_sha256=queue.donor_sha256)
        out["invalid"] = out["invalid"] or queue.residual_frac() > spec.residual_max
    return out


async def play_with_retries(play, sb, retry_max) -> dict:
    for attempt in range(retry_max + 1):
        res = await play(sb)
        if res.get("finished", False):
            return dict(res, retries=attempt, invalid=False)
    return dict(res, retries=retry_max, invalid=True)
```

If `ScheduledBattle` is not a dataclass, build the renamed copies with its constructor instead of `dataclasses.replace` (read `flymon/battle/schedule.py`). Check `ScheduledBattle.my_team` element type and make the digest JSON-serialisable.

- [ ] **Step 4: Implement `scripts/run_rescope_battles.py`**

Follow `scripts/run_m3_smoke.py` for: C3 config, connectome/encoder, `FlyPool`, `BrainSwarm`, `ShowdownServer(port=_port())`, `ServerConfiguration`, `BatchBarrier`s, `CheckpointStore(out/"checkpoints", config_hash(cfg))`, `run_cohort(sched, store, out/"logs", play_one, pool.state, stop_after)`, `on_player_loop(p, p.drain())`, `write_battle_summary`. Differences:
1. Accounts: `fm-r-{arm}-f{NN}` truncated to 18 chars (e.g. `fm-rFLY-f00`, `fm-rRS-f00`, `fm-rCO-f00`, `fm-rRN-f00`, `fm-rMX-f00`).
2. `p.start_battle(sb.battle_id, battle_index(sb.battle_id))`.
3. Two cohorts per brain arm: learn schedule with a checkpoint dir `checkpoints/learn`, then `enter_eval(swarm, pool)` and the eval schedule with `checkpoints/eval` (resume re-enters eval if the learn store is complete). Record each fly's `pool.w[f]` sha256 before and after the eval cohort → `weights_bit_identical_across_eval`.
4. RS: before the learn cohort, for each fly load `YokedQueue.from_log(FLY_out/"logs"/f"fly{k:02d}.jsonl", learn_ids_k)`; refuse if FLY's learn `result.json`/checkpoint is incomplete; the queue state is saved in `out/yoke_state.json` after each battle (commit hook in `play_one`) and restored on `--resume` via `load_state` (sha256 check).
5. COFF: `FlySpec(enabled=False)`, `BrainSwarm(..., mode="eval")`, eval cohort only.
6. RND/MAX: no pool/swarm; per fly `FlyCoachPlayer(RandomProvider(seed=fly) | MaxDamageProvider(), Coach(), ...)` as in `pilot_no_brain.run_fly`, eval schedule only.
7. `--smoke`: `--flies 2 --learn 2 --eval 2 --workers 2`, out under `results/rescope-smoke/`, taurec summary may be the smoke one.
8. Every battle goes through `play_with_retries`.
9. Refuse `--phase judge` unless `results/summary/rescope_power.json` exists with `status == "SIZED"` and `--flies/--eval` equal its `F`/`E`.

- [ ] **Step 5: Run to verify pass** — `uv run pytest -q -rfE -o addopts="" tests/rescope/` → PASS; `uv run pytest -q -rfE -o addopts="" tests/agent tests/battle` stays green.
- [ ] **Step 6: Commit** — `git add flymon/rescope/blocks.py scripts/run_rescope_battles.py tests/rescope/test_blocks.py tests/rescope/test_run_battles.py && git commit -m "feat(rescope): learn/eval block driver — FLY, yoked RS (after FLY), C-off eval-only, RND/MAX; frozen eval weights, retries, shared eval schedule digest"`

---

### Task 9: Win-rate statistics and the M4 verdict

**Files:**
- Create: `flymon/rescope/stats.py`, `scripts/write_rescope_m4.py`
- Test: `tests/rescope/test_stats.py`

**Interfaces:**
- Consumes: arm `result.json` files from Task 8 (`per_fly[].eval_battles[].won`, `invalid`).
- Produces:
  - `win_table(result: dict) -> dict[int, np.ndarray]` (fly → 0/1 array of eval wins; unfinished counts as a loss, as in `pilot_no_brain`; INVALID flies excluded).
  - `paired_boot(a: dict, b: dict, draws, seed) -> dict` keys `diff, lo, hi, n_pairs` — resamples fly index k over the flies present and valid in both, then battles within each fly for each arm; statistic = mean over flies of per-fly win rate (a) − (b).
  - `m4_verdict(arms: dict[str, dict], spec) -> dict` keys `2a: {pass, diff, lo, hi}`, `2b: {pass, vs_coff: {...}, vs_rs: {...}}`, `invalid_pairs: {...}`, `recorded: {max_minus_fly, fly_rnd_ratio...}`.
  - CLI `scripts/write_rescope_m4.py --phase judge` → `results/summary/rescope_m4.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_stats.py
import numpy as np
from flymon.rescope import stats
from flymon.rescope.spec import SPEC

def arm(rates, n=200, seed=0, invalid=()):
    rng = np.random.default_rng(seed)
    return {"per_fly": [{"fly": k, "invalid": k in invalid,
                         "eval_battles": [{"won": bool(w), "finished": True} for w in rng.random(n) < r]}
                        for k, r in enumerate(rates)]}

def test_clear_difference_detected():
    out = stats.paired_boot(stats.win_table(arm([0.5] * 12, seed=1)), stats.win_table(arm([0.2] * 12, seed=2)), 2000, 0)
    assert out["lo"] > 0 and abs(out["diff"] - 0.3) < 0.05

def test_no_difference_ci_contains_zero():
    out = stats.paired_boot(stats.win_table(arm([0.3] * 12, seed=3)), stats.win_table(arm([0.3] * 12, seed=4)), 2000, 0)
    assert out["lo"] < 0 < out["hi"]

def test_invalid_flies_dropped_pairwise():
    a = stats.win_table(arm([0.5] * 4, invalid=(1,)))
    b = stats.win_table(arm([0.2] * 4))
    assert stats.paired_boot(a, b, 200, 0)["n_pairs"] == 3

def test_2b_needs_both():
    arms = {"FLY": arm([0.5] * 12, seed=1), "COFF": arm([0.2] * 12, seed=2), "RS": arm([0.5] * 12, seed=3),
            "RND": arm([0.15] * 12, seed=4), "MAX": arm([0.47] * 12, seed=5)}
    v = stats.m4_verdict(arms, SPEC)
    assert v["2a"]["pass"] and v["2b"]["vs_coff"]["lo"] > 0 and not v["2b"]["pass"]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_stats.py` → FAIL.

- [ ] **Step 3: Implement `flymon/rescope/stats.py`**

```python
"""Evaluation-block win rates, fly -> battle paired hierarchical bootstrap (Readings R7), (2a)/(2b) verdict."""
from __future__ import annotations

import numpy as np


def win_table(result: dict) -> dict:
    return {int(f["fly"]): np.array([1.0 if b["won"] is True else 0.0 for b in f["eval_battles"]])
            for f in result["per_fly"] if not f.get("invalid")}


def paired_boot(a: dict, b: dict, draws: int, seed: int) -> dict:
    ks = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    point = float(np.mean([a[k].mean() - b[k].mean() for k in ks]))
    stat = np.empty(draws)
    for d in range(draws):
        pick = rng.choice(ks, size=len(ks), replace=True)
        stat[d] = np.mean([rng.choice(a[k], a[k].size).mean() - rng.choice(b[k], b[k].size).mean() for k in pick])
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return dict(diff=point, lo=float(lo), hi=float(hi), n_pairs=len(ks))


def m4_verdict(arms: dict, spec) -> dict:
    t = {k: win_table(v) for k, v in arms.items()}
    cmp = lambda x, y: paired_boot(t[x], t[y], spec.boot_draws, spec.boot_seed)
    a = cmp("FLY", "RND")
    c, r = cmp("FLY", "COFF"), cmp("FLY", "RS")
    rate = lambda k: float(np.mean([v.mean() for v in t[k].values()]))
    invalid = {k: [f["fly"] for f in v["per_fly"] if f.get("invalid")] for k, v in arms.items()}
    return {"2a": dict(a, **{"pass": a["lo"] > 0}),
            "2b": {"pass": c["lo"] > 0 and r["lo"] > 0, "vs_coff": c, "vs_rs": r},
            "invalid_flies": invalid,
            "recorded": {"win_rate": {k: rate(k) for k in t}, "max_minus_fly": rate("MAX") - rate("FLY")}}
```

`scripts/write_rescope_m4.py`: reads the five `results/rescope/judge/<ARM>/result.json`, refuses if any is missing or if the eval schedule digests differ, writes `results/summary/rescope_m4.json` = `m4_verdict(...)` + the digests + per-arm `n_flies`, `n_eval`, RS residuals and donor sha256s, + provenance. It also records the information-turn match rate (spec 4.2 statistic: among fly turns whose candidates' max type multiplier differs, the fraction choosing the max) from each brain arm's eval-block `decision` records (fields per `flymon/agent/logschema.py`), as a record only. Add a test for that function (`info_turn_match(records) -> float`) with three hand-made decision records.

- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add flymon/rescope/stats.py scripts/write_rescope_m4.py tests/rescope/test_stats.py && git commit -m "feat(rescope): eval-block win tables, paired fly->battle bootstrap, (2a)/(2b) verdict and the M4 summary writer"`

---

### Task 10: Pilot variance, joint power, (F, E) choice and budget

**Files:**
- Create: `flymon/rescope/power.py`, `scripts/write_rescope_power.py`
- Test: `tests/rescope/test_power.py`

**Interfaces:**
- Consumes: pilot arm `result.json`s (FLY, RS, COFF, RND) from Task 8 and their `wall_clock_s`, fly decisions per battle (count `decision` records with `decider == "fly"` in the logs); `SPEC` (`mie`, `power_target`, `var_margin`, `power_draws`, `boot_seed`, `budget_hours`, `learn_battles`).
- Produces:
  - `components(table: dict) -> dict` keys `p, var_between, n_flies, n_battles` (method of moments: `var_between = max(0, var(per-fly rates) − mean(p_k(1−p_k)/n_k))`).
  - `joint_power(comp: dict[str, dict], F, E, spec) -> dict` keys `p_2b, p_coff, p_rs, p_2a` (Readings R8: per-arm SE² = (var_between·var_margin)/F + p(1−p)/(F·E); FLY noise drawn once per draw and shared by both (2b) comparisons; effect `mie` added to FLY; pass = z > 1.96).
  - `wall_hours(F, E, rates: dict, spec) -> float` with `rates = {"sec_per_fly_battle_brain": s_b, "sec_per_battle_nobrain": s_n}` from the pilot: `hours = [F·(L+E)·s_b (FLY) + F·(L+E)·s_b (RS, after FLY) + F·E·s_b (COFF) + 2·F·E·s_n (RND, MAX)] / 3600`.
  - `choose(comp, rates, spec, grid_F=(8, 12, 16, 24, 32), grid_E=(20, 40, 60, 80, 100, 150, 200, 300)) -> dict` keys `status` (`"SIZED"` | `"STOP_BUDGET"` | `"STOP_POWER"`), `F, E, hours, p_2b, table`: among grid points with `p_2b >= power_target`, the smallest `wall_hours`; if none reach power → `STOP_POWER`; if the best exceeds `budget_hours` → `STOP_BUDGET` (still reporting it).
  - CLI `scripts/write_rescope_power.py` → `results/summary/rescope_power.json` with the above, the pilot's per-arm components, the rates, whether the pilot overlapped with M (`--m-overlap yes|no`, required flag, recorded verbatim), and the pre-estimate text from spec 10.7.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rescope/test_power.py
import numpy as np
from flymon.rescope import power
from flymon.rescope.spec import SPEC

def comp(p, vb=0.0): return {"p": p, "var_between": vb, "n_flies": 6, "n_battles": 20}

def test_components_moment_estimate():
    rng = np.random.default_rng(0)
    table = {k: (rng.random(400) < 0.3).astype(float) for k in range(40)}
    c = power.components(table)
    assert abs(c["p"] - 0.3) < 0.02 and c["var_between"] < 0.002

def test_power_grows_with_E():
    cs = {"FLY": comp(0.3), "RS": comp(0.3), "COFF": comp(0.3), "RND": comp(0.15)}
    lo = power.joint_power(cs, 12, 40, SPEC)["p_2b"]; hi = power.joint_power(cs, 12, 400, SPEC)["p_2b"]
    assert hi > lo and 0 <= lo <= 1

def test_joint_below_single():
    cs = {"FLY": comp(0.3, 0.001), "RS": comp(0.3, 0.001), "COFF": comp(0.3), "RND": comp(0.15)}
    r = power.joint_power(cs, 16, 200, SPEC)
    assert r["p_2b"] <= min(r["p_coff"], r["p_rs"]) + 1e-9

def test_choose_min_wall_and_stops():
    cs = {"FLY": comp(0.3), "RS": comp(0.3), "COFF": comp(0.3), "RND": comp(0.15)}
    rates = {"sec_per_fly_battle_brain": 1.0, "sec_per_battle_nobrain": 0.1}
    out = power.choose(cs, rates, SPEC)
    assert out["status"] == "SIZED" and out["p_2b"] >= 0.8
    slow = {"sec_per_fly_battle_brain": 600.0, "sec_per_battle_nobrain": 1.0}
    assert power.choose(cs, slow, SPEC)["status"] == "STOP_BUDGET"
    tiny = dict(cs, FLY=comp(0.3, 0.2), RS=comp(0.3, 0.2))
    assert power.choose(tiny, rates, SPEC, grid_F=(2,), grid_E=(20,))["status"] == "STOP_POWER"
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_power.py` → FAIL.

- [ ] **Step 3: Implement `flymon/rescope/power.py`**

```python
"""Stage 4 (spec 10.7): pilot variance components -> joint (2b) power -> the cheapest (F, E) -> budget."""
from __future__ import annotations

import numpy as np


def components(table: dict) -> dict:
    rates = np.array([v.mean() for v in table.values()])
    ns = np.array([v.size for v in table.values()])
    p = float(rates.mean())
    within = float(np.mean(rates * (1 - rates) / ns))
    return dict(p=p, var_between=float(max(0.0, rates.var(ddof=1) - within)), n_flies=len(rates),
                n_battles=int(ns.mean()))


def _se2(c, F, E, spec) -> float:
    return c["var_between"] * spec.var_margin / F + c["p"] * (1 - c["p"]) / (F * E)


def joint_power(comp: dict, F: int, E: int, spec) -> dict:
    rng = np.random.default_rng(spec.boot_seed)
    n = spec.power_draws
    fly = dict(comp["FLY"], p=min(comp["FLY"]["p"] + spec.mie, 1.0))
    e_fly = rng.normal(0, np.sqrt(_se2(fly, F, E, spec)), n)
    out = {}
    for name in ("COFF", "RS", "RND"):
        other = comp[name]
        e_o = rng.normal(0, np.sqrt(_se2(other, F, E, spec)), n)
        centre = spec.mie if name != "RND" else comp["FLY"]["p"] - other["p"]   # (2b): FLY = other + MIE; (2a): pilot gap
        diff = centre + e_fly - e_o
        se = np.sqrt(_se2(fly, F, E, spec) + _se2(other, F, E, spec))
        out[name] = diff / se > 1.96
    return dict(p_coff=float(out["COFF"].mean()), p_rs=float(out["RS"].mean()),
                p_2b=float((out["COFF"] & out["RS"]).mean()), p_2a=float(out["RND"].mean()))


def wall_hours(F, E, rates, spec) -> float:
    b, s = rates["sec_per_fly_battle_brain"], rates["sec_per_battle_nobrain"]
    L = spec.learn_battles
    return (2 * F * (L + E) * b + F * E * b + 2 * F * E * s) / 3600.0


def choose(comp, rates, spec, grid_F=(8, 12, 16, 24, 32), grid_E=(20, 40, 60, 80, 100, 150, 200, 300)) -> dict:
    table = []
    for F in grid_F:
        for E in grid_E:
            jp = joint_power(comp, F, E, spec)
            table.append(dict(F=F, E=E, hours=wall_hours(F, E, rates, spec), **jp))
    ok = [r for r in table if r["p_2b"] >= spec.power_target]
    if not ok:
        return dict(status="STOP_POWER", F=None, E=None, hours=None, p_2b=None, table=table)
    best = min(ok, key=lambda r: (r["hours"], r["F"]))
    status = "SIZED" if best["hours"] <= spec.budget_hours else "STOP_BUDGET"
    return dict(status=status, F=best["F"], E=best["E"], hours=best["hours"], p_2b=best["p_2b"], table=table)
```

(Power is computed at the MIE alternative: the true FLY − COFF and FLY − RS differences are both 0.05; FLY's sampling noise is shared by the two (2b) comparisons. (2a)'s power uses the pilot's measured FLY − RND gap and is recorded only.)

`scripts/write_rescope_power.py`: reads the four pilot `result.json`s and logs, builds `components` per arm, rates (`sec_per_fly_battle_brain` = FLY learn+eval wall-clock / (flies × battles); `sec_per_battle_nobrain` = RND wall-clock / battles), calls `choose`, writes `results/summary/rescope_power.json`. Refuses if any pilot arm result is missing or its schedule digests are not the pilot ones.

- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add flymon/rescope/power.py scripts/write_rescope_power.py tests/rescope/test_power.py && git commit -m "feat(rescope): pilot variance components, joint (2b) power at MIE 0.05 with 1.5x margin, cheapest (F,E), STOP_BUDGET/STOP_POWER, power summary writer"`

---

### Task 11: Smoke commands, self-checks and the runbook

**Files:**
- Create: `scripts/rescope_selfcheck.py`, `docs/superpowers/plans/2026-09-28-rescoped-claim-runbook.md`
- Test: `tests/rescope/test_selfcheck.py`

**Interfaces:**
- Consumes: everything above; `flymon.brain.h4_jobs.oracle_job`; `flymon.brain.fly_pool.FlyPool`; in-process `flymon.brain.engine_cpu.Engine` (M0b equivalence pattern, see `tests/agent/test_swarm_c3.py`).
- Produces:
  - `scripts/rescope_selfcheck.py [--allow-dirty]` → `results/rescope/selfcheck.json` with three checks, each `{"ok": bool, ...}`:
    1. `pairs_digest`: `new_pairs` on the real connectome reproduces `SPEC.pairs_digest`.
    2. `oracle_copy_real`: on C3 and the `seed0` pair with 2 select and 2 report seeds, `reward_oracle_job`'s `alpha_reward`, `select["reward"]` and `report` pre/R1 equal `oracle_job`'s.
    3. `pool_equivalence_c3`: one `decide_batch` on the C3 pool equals the in-process engine for the same seed (copy the method of `tests/agent/test_swarm_c3.py`).
  - The runbook: exact smoke commands in order, then the real-stage commands in order with "controller: check `orca worktree ps` for M's oracle/scan and report to the user first" before each real stage.

- [ ] **Step 1: Write the failing test** — `tests/rescope/test_selfcheck.py`: load the script with importlib, monkeypatch its three check functions to return `{"ok": True}`, `{"ok": False}`, `{"ok": True}`; `main(["--allow-dirty"])` in a tmp cwd writes `results/rescope/selfcheck.json` with `all_ok == False` and exits non-zero (`SystemExit` with code 1).
- [ ] **Step 2: Run to verify failure** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_selfcheck.py` → FAIL.
- [ ] **Step 3: Implement** `scripts/rescope_selfcheck.py` (three functions `check_pairs_digest()`, `check_oracle_copy()`, `check_pool_equivalence()`, and `main`), and write the runbook:

```markdown
# Re-scoped claim — runbook (controller only)

## Smoke (after all tasks; controller runs these in the background, reads exit code + error lines only)
1. uv run pytest -q -rfE -o addopts="" tests/rescope tests/agent tests/battle
2. uv run python scripts/rescope_selfcheck.py
3. uv run python scripts/run_rescope_taurec.py --smoke --out results/rescope-smoke/taurec
4. uv run python scripts/run_rescope_qualify.py --smoke --workers 4 --out results/rescope-smoke/qualify
5. uv run python scripts/run_rescope_primary.py --smoke --workers 4 --out results/rescope-smoke/primary
6. for ARM in FLY RS COFF RND MAX: uv run python scripts/run_rescope_battles.py --smoke --phase pilot --arm $ARM --out results/rescope-smoke/pilot/$ARM
7. uv run python scripts/write_rescope_power.py --smoke --m-overlap <yes|no>
Record wall-clock of each smoke step in the ledger (feeds the cost estimates of spec 5).

## Real stages — NOT started without the user's go
Before each: `orca worktree ps` → is M's oracle/scan running in guillemot? Report to the user; wait for the go.
S1  uv run python scripts/run_rescope_taurec.py --out results/rescope/taurec
S2  uv run python scripts/run_rescope_qualify.py --workers 16 --out results/rescope/qualify    (stop if STOP_FEW_PAIRS / control not qualified)
S3  uv run python scripts/run_rescope_primary.py --workers 16 --out results/rescope/primary
S4  pilot (6 flies, L 40, E 20 for every arm): FLY, then RS, then COFF, RND
      uv run python scripts/run_rescope_battles.py --phase pilot --arm <ARM> --flies 6 --eval 20 --learn 40 --out results/rescope/pilot/<ARM>
    then uv run python scripts/write_rescope_power.py --m-overlap <yes|no>
S5  judge (only if rescope_power.json status SIZED): FLY, RS, COFF, RND, MAX with --flies F --eval E, then write_rescope_m4.py --phase judge
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -q -rfE -o addopts="" tests/rescope/test_selfcheck.py` → PASS; full suite `uv run pytest -q -rfE -o addopts=""` → no new failures (the pre-existing strict xfails in `tests/test_m2_go.py` stay xfail).
- [ ] **Step 5: Commit** — `git add scripts/rescope_selfcheck.py docs/superpowers/plans/2026-09-28-rescoped-claim-runbook.md tests/rescope/test_selfcheck.py && git commit -m "feat(rescope): self-checks (pair digest, oracle copy on C3, pool equivalence) and the controller runbook"`

---

## After the tasks (controller)

1. Final whole-branch review (opus reviewer).
2. Smoke and self-checks per the runbook, in the background; record results and wall-clocks in the ledger `docs/superpowers/plans/2026-09-28-rescoped-claim-ledger.md`.
3. Report to the user. Do not start S1–S5. Do not merge or push to main.
