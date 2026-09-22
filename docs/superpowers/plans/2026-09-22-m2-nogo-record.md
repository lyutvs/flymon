# M2 No-Go Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record "M2 no-go — 시험 불성립" (spec route 3: 범위 축소 / M2 no-go 기록 + 주장 재설계) — re-judge D.6 (a) by G.8 for the record, derive `results/summary/m2_nogo.json` from the recorded summaries by code, keep the M2 go condition as strict xfail tests, and write spec appendix I, the README ledger and a handoff note for the claim redesign.

**Architecture:** One small module `flymon/brain/d6a.py` holds G.8's odours, worker job and verdict (`judge`); `scripts/run_m2_d6a.py` runs it on the pool and writes the git-excluded raw record. `scripts/write_m2_nogo_summary.py` derives the no-go chain (G.10 → G.12 → H.4 → H.4a.8 ceiling read by the H.4a.7 rule → D.6 status) from committed summaries plus the two git-excluded recorded measurements, and refuses to write if any link does not hold. `tests/test_m2_go.py` keeps spec 5's go condition as strict expected failures (the A.5 precedent). Three documentation tasks follow from the recorded numbers.

**Tech Stack:** Python 3.13, NumPy, pytest, `uv`; `flymon.brain.fly_pool.FlyPool` (spawn start method); `poke_env` (already a dependency, used by `h4_pairs.pool_vocabulary`).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` — spec 0 (claims), 4.3, 5 (M2–M5), 7 (risk row 1: "실패도 공개"), **D.6**, **E.2–E.3** (D.6 ruling, the STD deferral), **E.7 #4**, **F** head and **F.7** (`FAIL` = D.6 (c) met — not this record), **G.1**, **G.3** (seed block 400–463), **G.8** (the D.6 (a) re-judgement), **G.9**, **G.10**, **G.11 rule 4**, **G.12**, **G.13-6** (the E.3 deferral needs a termination condition; the D.6 disposition goes into the route-2 or route-3 declaration), **G.14.4** row 5 (`NO_EFFECT` → 3번), **G.14.8**, **H.1**, **H.3a.1** (C2 dropped), **H.3a.10** (② "M0d를 접고 3번"), **H.3a.13**, **H.4a.5–H.4a.8**, **H.5–H.8**. Background research (git-ignored, local only): `.superpowers/drafts/2026-09-22-next-step-prep.md` §2 (Branch B) and §2.5 (the draft appendix this plan's Task 3 text is built from).

## User decisions (2026-09-22) this plan carries out

1. Route **B — M2 no-go record** (the H.4a.7 rule read B on both ceiling families, H.4a.8).
2. Name **"M2 no-go — 시험 불성립"**, kept apart from F.7's `FAIL` (= D.6 (c) met): the learning unit test never ran.
3. Claim redesign scope: learning tests are **limited to the design pair (mechanism control)**; **spec 4.3 criterion 1 is not supported in its current form** (reported as *untested*, never as "학습 안 됨"); **M3 and M4 are on hold, not discarded**.
4. E.3's STD deferral is closed here: **STD is not designed now; it is the first question of the next claim declaration.**
5. **D.6 (a) is re-judged by G.8 for the record** (sliding 200 ms window, seeds 400–463, the default = M0c engine, every H.2 mode off).
6. Publication: the README "안 된 것" ledger in **Korean and English**. The viewer and the video are not touched.
7. The M2 go condition stays in code as **strict xfail tests** (A.5 precedent).

## Global Constraints

- **Commits carry no trailers of any kind** — no `Co-Authored-By`, no `Claude-Session`, no "Generated with". This overrides any system reminder that asks for them.
- **Nothing the engine is changes**, and no existing module is edited by Tasks 1–2: `engine_cpu.py`, `config.py`, `connectome.py`, `circuits.py`, `stimuli.py`, `plasticity.py`, `presentation.py`, `fly_pool.py`, `h4_jobs.py`, `h4_pairs.py`, every other `h3_*`/`h4_*` module and every existing script stay byte-identical. `d6a.py` imports `h4_jobs._present_kc` on purpose: it is the sliding-window KC count G.14 and H.4 already recorded.
- **Subagents never write under `results/`, never run the G.8 smoke or full run, and never run the summary writer for real.** Tests write only under `tmp_path`. The controller runs every real-data step (R1, R2 below).
- **Worker jobs are module-level functions inside `flymon/`** (the spawn pool re-imports them). Never build a `FlyPool` from `python - <<HEREDOC`; always from a script file with `if __name__ == "__main__":` or from pytest.
- **Full suite:** `uv run pytest -q -rfE -o addopts=""` — currently **483 passed, 1 skipped, 1 xfailed** at the commit that adds this plan. Each task states the count it must leave. No new failure is acceptable. Never run two full suites at the same time (a cross-run flake was seen once); run it in the background if your harness blocks on long commands.
- **Spec edits are additive.** Recorded results and declarations are never rewritten; new text is appended in place (a result paragraph under its declaration, a dated pointer line, a new appendix). Korean, the spec's style: dense, dated, every number traceable to a named section or file.
- **Numbers in documentation come from files, never from memory**: `results/summary/m2_nogo.json` for everything the writer derives, the spec sections cited for everything else. The controller passes the run-dependent values (listed per task) in the dispatch prompt.
- Style: match the surrounding modules — module docstring citing the spec section, dense one-line comments, no annotation boilerplate beyond what neighbouring files use.

## Readings of the spec (decided here)

1. **This is not D.6 (c).** Spec 5's learning unit test never ran (F v3 withdrawn unrun, F v4 never written); G.1 and H.8 say no G or H data judges (c). F.7's `FAIL` row and H.6's D.6 table row 1 therefore do not fire. What exists is route 3, named by G.14.4 row 5 and H.3a.10 ②, taken by user decision.
2. **G.8's odours are E.1's 41 candidates**, not H.4's pair odours: 16 turns of the provisional spec-3.3 encoder (E0), candidates in turn order, strengths built exactly as `m2-calibration-g/m2_probe.turn_odors` did (shared my/opp/HP keys first, then move and power). Checked bit for bit in the planning session (ULP differences appear if the keys are ordered as in `h4_pairs.odour`, so the E.1 order is used). Digest `29992673f80a629ba09dbb3cf43eb257846c55bc654cbc75f3a74f4af3d6c0c9`.
3. **G.8's window is `h4_jobs._present_kc`'s**: per KC, spikes in a 200-step window sliding by one 1 ms step over settle 800 + read 600 ms (partial windows at the start are subsets of the first full window, so they cannot raise the maximum). Over = **≥ 31 spikes** (> 150 Hz; 30 spikes is exactly 150 Hz and is not over — E.2's record maximum). D.6 (a) is met iff any of the 41 × 64 = 2,624 presentations is over.
4. **Paired noise and hygiene as E.1**: every candidate of a turn from the same reset seed, plasticity off, weights reset around the job. On E.1's seeds this reproduces `results/m2/candidate_map.json`'s read-window KC activity and spike count bit for bit (planning session: 48 of 48 presentations; turn 12 seed 203 candidate 1 — E.2's 150.0 Hz maximum — reads exactly 30 spikes in the sliding window). The runner repeats two of these cells as a self-check before measuring.
5. **The engine is `Params()`**, which equals H.3's C0 adopted params (the M0c engine, every H.2 mode off; checked against block `"h3"`). Nothing on the command line changes it.
6. **The no-go chain is derived, not asserted.** The writer refuses unless: no learning-test summary exists; G.12 stopped (no winner, best score < 0.5); H.4 is `STOP_LOW_T_B` and no combination reaches T_b ≥ 0.5 ∧ F_a ≥ 2 (recomputed from T_b and F_a, never the stored flag); both ceiling summaries are the full runs of H.4's run with the right family, bound to committed code (the commit they name is an ancestor of HEAD and the ceiling script at that commit hashes to their recorded sha256); the H.4a.7 rule reads **B** on every family (C3 testable (b) ≤ 10/21; A iff ≥ 14/21 ∧ F_a ≥ 2; otherwise the user's call); and the G.8 record passes `d6a.judge`. D.6 (a) may read either way — it is recorded, not required.
7. **Git-excluded inputs** (`results/m0d/diag/h4_specificity_ceiling{,_all}_summary.json`, `results/m2/d6a/g8.json`) are bound by `inputs_sha256`, and the numbers the record needs are copied into the committed `m2_nogo.json` (the D.5 rule: raw git-excluded, summary committed).
8. **The go condition, as code.** `test_m2_learning_unit_test_can_be_built` asserts that some H.4 combination reaches T_b ≥ 0.5 ∧ F_a ≥ 2 (G.14.4/H.4's bar for building the test); `test_m2_go` asserts that F.10 #4's summary `results/summary/m2_learning.json` exists with outcome `PASS` (F.7). Both are strict xfail: they break as XPASS the day either becomes true.

## File Structure

| File | Task | Responsibility |
|---|---|---|
| `flymon/brain/d6a.py` (create) | 1 | G.8 constants, E.1's 41 odours, the worker job, `judge()` |
| `scripts/run_m2_d6a.py` (create) | 1 | CLI: refusals, E.2 self-check, pool calls per 8-seed block, raw record `results/m2/d6a/{g8,smoke}.json` |
| `tests/brain/test_d6a.py` (create) | 1 | window vs brute force, hygiene, pool = in-process, `judge` at 30/31 and on broken records, the real 41 odours, engine = C0 |
| `tests/test_run_m2_d6a.py` (create) | 1 | CLI refusals before measuring |
| `scripts/write_m2_nogo_summary.py` (create) | 2 | derive and write `results/summary/m2_nogo.json`; refuse on any broken link |
| `tests/test_write_m2_nogo_summary.py` (create) | 2 | every link and boundary of the derivation; provenance; CLI refusals; the committed record = what the code derives |
| `tests/test_m2_go.py` (create) | 2 | spec 5's go condition as two strict xfails |
| `docs/superpowers/specs/2026-09-14-flymon-design.md` (modify) | 3 | appendix I; G.8 result; dated pointers in 0, 4.3, 5, E.2, E.3, F head, G.9, H.4a.5, H.5, H.6 |
| `README.md` (modify) | 4 | header status, ledger catch-up (G.10–H.4a.8) and the no-go entry, an English section |
| `docs/handoffs/2026-09-22-m2-nogo-claim-redesign.md` (create) | 5 | starting point for the next claim declaration |

Order: Task 1 → **R1** (controller: G.8 smoke and full run, in the background) → Task 2 (in parallel with R1's full run) → **R2** (controller: write and commit `m2_nogo.json`) → Task 3 → Task 4 → Task 5 → **R3** (controller: full suite, final review, push).

---

### Task 1: G.8 — the D.6 (a) re-judgement code

**Files:**
- Create: `flymon/brain/d6a.py`
- Create: `scripts/run_m2_d6a.py`
- Test: `tests/brain/test_d6a.py`, `tests/test_run_m2_d6a.py`

**Interfaces:**
- Consumes (existing, unchanged): `flymon.brain.h4_pairs.pool_vocabulary()`, `.e0_channels(pops, mon_types, move_types) -> {"group:symbol": [receptor type]}`, `.build_turns(species_types, move_info, n_turns) -> [turn dict]`, `.hp_bin`, `.power_bin`; `flymon.brain.h4_jobs._present_kc(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms) -> {"read": int32[n_kc], "max_win": int}`; `FlyPool(npz, params, flies, workers).run_jobs(fn, kwargs_list)` (results in order; `fn(eng, pl, pops, comps, ro, **kw)`).
- Produces (Task 2 relies on these exact names): `d6a.SEEDS` (tuple 400..463), `d6a.N_ODOURS` (41), `d6a.OVER_SPIKES` (31), `d6a.WINDOW_MS` (200), `d6a.ODOURS_DIGEST` (str), `d6a.candidate_odours(pops) -> [{"turn": int, "move": str, "odor": {receptor: float}}]`, `d6a.odours_digest(rows) -> str`, `d6a.engine_params() -> dict`, `d6a.d6a_job(eng, pl, pops, comps, ro, odors, seed, strength=0.35, settle_ms=800.0, read_ms=600.0, window_ms=200) -> {"seed", "max_win": [int], "kc_active_frac": [float], "kc_spikes": [int]}`, and **`d6a.judge(raw: dict) -> {"condition", "fired": bool, "n_presentations", "n_over", "max_win_spikes", "max_win_hz", "margin_spikes", "over": [{"turn", "move", "seed", "max_win"}]}`**, raising `ValueError` naming every problem when `raw` is not a complete full G.8 run. The raw record `results/m2/d6a/g8.json` has keys `what, smoke, started_utc, wall_s, git{commit, dirty}, code_sha256, params, odours, odours_digest, seeds, strength, settle_ms, read_ms, window_ms, over_spikes, workers, e2_check{ok, cells}, rows[{turn, seed, max_win, kc_active_frac, kc_spikes}]`.

- [ ] **Step 1: Write the failing tests**

`tests/brain/test_d6a.py`:

```python
"""Spec G.8, the D.6 (a) re-judgement: the sliding-window count against a brute-force reference, the job's plasticity
hygiene, pool = in-process, the 41 odours = E.1's, and judge() at the 30/31-spike boundary and on incomplete records."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import d6a
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair, present

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
E1_META = ROOT / "results/m2/candidate_map_meta.json"          # git-excluded: E.1's encoder and turns
M2_PROBE = ROOT / "docs/superpowers/specs/m2-calibration-g/m2_probe.py"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="data/malecns.npz not present")

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
SYN = dict(strength=6.0, settle_ms=50.0, read_ms=100.0, window_ms=60)   # windows shorter than the presentation


@pytest.fixture
def rig(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, P, seed=0)
    return c, pops, eng, Plasticity(eng, pops, compartments(c, pops, P.core_frac))


def _reference_max_window(eng, pl, pops, odor, seed, strength, settle_ms, read_ms, window_ms):
    """Record every KC spike of the presentation, then take the largest count over all 1 ms-step windows."""
    kc = pops.kc
    pos = np.full(eng.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    pl.set_enabled(False)
    eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
    present(eng, pops, odor, strength)
    n = int(round((settle_ms + read_ms) / eng.p.dt))
    spikes = np.zeros((n, len(kc)), np.int64)
    for step in range(n):
        f = pos[eng.step()]; spikes[step, f[f >= 0]] = 1
    cum = np.vstack([np.zeros((1, len(kc)), np.int64), np.cumsum(spikes, 0)])
    pl.set_enabled(True)
    return max(int((cum[i] - cum[max(0, i - window_ms)]).max()) for i in range(1, n + 1))


def test_window_count_equals_a_brute_force_reference(rig):
    c, pops, eng, pl = rig
    a, b = design_odor_pair(pops, k=2, seed=0)
    got = d6a.d6a_job(eng, pl, pops, None, None, odors=[a, b], seed=7, **SYN)
    ref = [_reference_max_window(eng, pl, pops, o, 7, **SYN) for o in (a, b)]
    assert got["max_win"] == ref
    assert max(ref) > 1, "the synthetic regime must make a KC fire more than once per window"
    short = [_reference_max_window(eng, pl, pops, o, 7, **{**SYN, "window_ms": 20}) for o in (a, b)]
    assert short != ref, "the window length must matter in this regime, or a wrong length would pass"
    assert got["seed"] == 7 and len(got["kc_active_frac"]) == 2 and all(s > 0 for s in got["kc_spikes"])


def test_job_leaves_weights_reset_and_plasticity_on(rig):
    c, pops, eng, pl = rig
    a, b = design_odor_pair(pops, k=2, seed=0)
    pl.set_enabled(True)
    d6a.d6a_job(eng, pl, pops, None, None, odors=[a], seed=3, **SYN)
    assert pl.enabled and pl.weights_frac() == 1.0


def test_pool_equals_in_process(rig, synthetic_npz):
    c, pops, eng, pl = rig
    a, b = design_odor_pair(pops, k=2, seed=0)
    here = d6a.d6a_job(eng, pl, pops, None, None, odors=[a, b], seed=11, **SYN)
    with FlyPool(synthetic_npz, P, [{}], workers=1) as pool:
        there = pool.run_jobs(d6a.d6a_job, [dict(odors=[a, b], seed=11, **SYN)])[0]
    assert there == here


# ---------------------------------------------------------------- judge()
def _fake_odours():
    """41 odours over 16 turns (9 turns of 3 candidates, 7 of 2), like E.1's."""
    rows = []
    for t in range(16):
        for j in range(3 if t < 9 else 2):
            rows.append({"turn": t, "move": f"m{t}_{j}", "odor": {f"ORN_{t}_{j}": 1.0}})
    return rows


def _raw(max_win=30):
    odours = _fake_odours()
    per_turn = {}
    for o in odours:
        per_turn[o["turn"]] = per_turn.get(o["turn"], 0) + 1
    rows = [{"turn": t, "seed": s, "max_win": [max_win] * n, "kc_active_frac": [0.05] * n, "kc_spikes": [100] * n}
            for t, n in per_turn.items() for s in d6a.SEEDS]
    return {"smoke": False, "seeds": list(d6a.SEEDS), "params": d6a.engine_params(), "odours": odours,
            "strength": 0.35, "settle_ms": 800.0, "read_ms": 600.0, "window_ms": 200, "over_spikes": 31,
            "e2_check": {"ok": True, "cells": []}, "rows": rows}


@pytest.fixture
def fake_digest(monkeypatch):
    monkeypatch.setattr(d6a, "ODOURS_DIGEST", d6a.odours_digest(_fake_odours()))


def test_judge_30_spikes_is_not_over(fake_digest):
    v = d6a.judge(_raw(30))
    assert (v["fired"], v["n_over"], v["n_presentations"]) == (False, 0, 41 * 64)
    assert (v["max_win_spikes"], v["max_win_hz"], v["margin_spikes"]) == (30, 150.0, 0)


def test_judge_one_presentation_at_31_spikes_fires(fake_digest):
    raw = _raw(30)
    row = next(r for r in raw["rows"] if r["turn"] == 12 and r["seed"] == 431)
    row["max_win"][1] = 31
    v = d6a.judge(raw)
    assert (v["fired"], v["n_over"], v["max_win_spikes"], v["margin_spikes"]) == (True, 1, 31, -1)
    assert v["over"] == [{"turn": 12, "move": "m12_1", "seed": 431, "max_win": 31}]


@pytest.mark.parametrize("breakage, message", [
    (lambda r: r["rows"].pop(), "cover"),
    (lambda r: r["rows"].append(copy.deepcopy(r["rows"][0])), "duplicate"),
    (lambda r: r.update(smoke=True), "smoke"),
    (lambda r: r.update(seeds=list(range(400, 463))), "400-463"),
    (lambda r: r["params"].update(kc_kc_scale=1.0), "M0c"),
    (lambda r: r.update(window_ms=100), "window_ms"),
    (lambda r: r.update(over_spikes=30), "over_spikes"),
    (lambda r: r.update(e2_check={"ok": False}), "self-check"),
    (lambda r: r.pop("e2_check"), "self-check"),
    (lambda r: r["odours"].pop(), "41"),
    (lambda r: r["rows"][0]["max_win"].pop(), "does not match"),
    (lambda r: r["rows"][0]["max_win"].__setitem__(0, 30.5), "does not match"),
])
def test_judge_refuses_an_incomplete_or_foreign_record(fake_digest, breakage, message):
    raw = _raw(30)
    breakage(raw)
    with pytest.raises(ValueError, match=message):
        d6a.judge(raw)


def test_the_real_digest_rejects_fake_odours():
    with pytest.raises(ValueError, match="41"):
        d6a.judge(_raw(30))


# ---------------------------------------------------------------- the real 41 odours
@needs_npz
def test_candidate_odours_are_e1s_41():
    pops = Populations.from_connectome(Connectome.load(NPZ))
    rows = d6a.candidate_odours(pops)
    assert len(rows) == d6a.N_ODOURS == 41 and len({r["turn"] for r in rows}) == 16
    assert d6a.odours_digest(rows) == d6a.ODOURS_DIGEST
    assert all(6 <= len(r["odor"]) <= 8 for r in rows)
    if E1_META.exists():            # bit for bit, in order, against the committed calibration code on E.1's record
        spec = importlib.util.spec_from_file_location("m2_probe", M2_PROBE)
        m2_probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2_probe)
        meta = json.loads(E1_META.read_text())
        ref = [o for t in meta["turns"] for o in m2_probe.turn_odors(pops, meta["channels"], t)]
        assert [list(r["odor"].items()) for r in rows] == [list(o.items()) for o in ref]


def test_engine_is_h3s_c0():
    """Params() = the M0c engine = H.3's C0 (every H.2 mode off), as recorded in the committed m0d.json."""
    h3 = json.loads((ROOT / "results/summary/m0d.json").read_text())["h3"]
    assert h3["combos"]["C0"]["adopted"]["params"] == d6a.engine_params()
```

`tests/test_run_m2_d6a.py`:

```python
"""scripts/run_m2_d6a.py (spec G.8) refuses before measuring anything: off the repository root, an existing output,
dirty tracked files on a full run, and a full run without E.1's record for the self-check."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("run_m2_d6a", Path("scripts/run_m2_d6a.py").resolve())
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fake repository root: the files main() checks for, nothing else."""
    (tmp_path / "flymon/brain").mkdir(parents=True); (tmp_path / "flymon/brain/d6a.py").write_text("")
    (tmp_path / "data").mkdir(); (tmp_path / "data/malecns.npz").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "git_state", lambda: {"commit": "abc", "dirty": []})
    monkeypatch.setattr(run.Connectome, "load", lambda *a, **k: pytest.fail("measured after a refusal"))
    return tmp_path


def test_refuses_off_the_repository_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert run.main([]) == 2 and "repository root" in capsys.readouterr().err


@pytest.mark.parametrize("smoke, name", [(False, "g8.json"), (True, "smoke.json")])
def test_never_overwrites_an_output(root, capsys, smoke, name):
    (root / "results/m2/d6a").mkdir(parents=True); (root / "results/m2/d6a" / name).write_text("{}")
    assert run.main(["--smoke"] if smoke else []) == 2 and "exists" in capsys.readouterr().err


def test_full_run_refuses_dirty_tracked_files(root, monkeypatch, capsys):
    monkeypatch.setattr(run, "git_state", lambda: {"commit": "abc", "dirty": ["flymon/brain/d6a.py"]})
    assert run.main([]) == 2 and "dirty" in capsys.readouterr().err


def test_full_run_needs_e1s_record(root, capsys):
    assert run.main([]) == 2 and "self-check" in capsys.readouterr().err
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q -o addopts="" tests/brain/test_d6a.py tests/test_run_m2_d6a.py`
Expected: collection errors — `ModuleNotFoundError: No module named 'flymon.brain.d6a'` and `FileNotFoundError` for `scripts/run_m2_d6a.py`.

- [ ] **Step 3: Write `flymon/brain/d6a.py`**

```python
"""Spec G.8: the D.6 (a) re-judgement (E.7 #4) — a D.6 ruling, not calibration.

The 41 candidate odours of the first M2 measurement (E.1: the provisional spec-3.3 encoder, 16 turns, candidates in
turn order) x seeds 400-463 (G.3). Each is presented for settle 800 + read 600 ms on the default engine (Params() =
the M0c engine = H.3's C0, every H.2 mode off), plasticity off, weights reset, every candidate of a turn from the same
reset seed (E.1's paired noise). Per KC, the spikes in a 200 ms window sliding by 1 ms over the whole presentation
(h4_jobs._present_kc, the window G.14 and H.4 recorded); a presentation exceeds if some KC has >= 31 spikes in some
window (> 150 Hz). D.6 (a) is met iff any presentation exceeds.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from . import h4_pairs
from .config import Params
from .h4_jobs import _present_kc

N_TURNS = 16
N_ODOURS = 41
SEEDS = tuple(range(400, 464))                  # G.3: the D.6 (a) re-judgement block
STRENGTH, SETTLE_MS, READ_MS = 0.35, 800.0, 600.0
WINDOW_MS = 200
OVER_SPIKES = 31                                # > 150 Hz in 200 ms
ODOURS_DIGEST = "29992673f80a629ba09dbb3cf43eb257846c55bc654cbc75f3a74f4af3d6c0c9"
CONDITION = "some KC has >= 31 spikes in a 200 ms window sliding by 1 ms over settle + read (> 150 Hz)"


def candidate_odours(pops) -> list:
    """[{turn, move, odor}] for E.1's 41 candidates, bit-exact to m2_probe.turn_odors (shared my/opp/HP keys first,
    then move and power; strengths 1/receptor count over the odour's glomeruli, mean 1)."""
    species_types, move_info, mon_types, move_types = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon_types, move_types)
    rows = []
    for t in h4_pairs.build_turns(species_types, move_info, N_TURNS):
        shared = ([f"my:{x}" for x in t["my_types"]] + [f"opp:{x}" for x in t["opp_types"]]
                  + [f"myhp:{h4_pairs.hp_bin(t['my_hp'])}", f"opphp:{h4_pairs.hp_bin(t['opp_hp'])}"])
        for c in t["candidates"]:
            keys = shared + [f"move:{c['type']}", f"pow:{h4_pairs.power_bin(c['bp'])}"]
            glom = [chan[k][0] for k in keys]
            if len(set(glom)) != len(glom):
                raise ValueError(f"channel collision in {keys}")
            inv = np.array([1.0 / len(pops.receptor_types[g]) for g in glom])
            inv = inv / inv.mean()
            rows.append({"turn": int(t["turn"]), "move": c["move"], "odor": {g: float(s) for g, s in zip(glom, inv)}})
    return rows


def odours_digest(rows: list) -> str:
    blob = json.dumps([[int(r["turn"]), r["move"], sorted((k, float(v)) for k, v in r["odor"].items())] for r in rows],
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def engine_params() -> dict:
    """Params() as it round-trips through JSON (tuples become lists)."""
    return json.loads(json.dumps(dataclasses.asdict(Params())))


def d6a_job(eng, pl, pops, comps, ro, odors: list, seed: int, strength: float = STRENGTH, settle_ms: float = SETTLE_MS,
            read_ms: float = READ_MS, window_ms: int = WINDOW_MS) -> dict:
    """One turn, one seed: every candidate from the same reset seed; plasticity off and weights reset around it."""
    pl.reset_weights(); pl.set_enabled(False)
    try:
        out = [_present_kc(eng, pl, pops, o, int(seed), strength, settle_ms, read_ms, int(window_ms)) for o in odors]
    finally:
        pl.reset_weights(); pl.set_enabled(True)
    return {"seed": int(seed), "max_win": [int(o["max_win"]) for o in out],
            "kc_active_frac": [float((o["read"] > 0).mean()) for o in out],
            "kc_spikes": [int(o["read"].sum()) for o in out]}


def judge(raw: dict) -> dict:
    """D.6 (a) from a raw G.8 record. ValueError names every way the record is not a complete G.8 run."""
    problems = []
    if raw.get("smoke") is not False:
        problems.append("not a full run (smoke is not False)")
    if list(raw.get("seeds", [])) != list(SEEDS):
        problems.append("seeds are not 400-463")
    if not (raw.get("e2_check") or {}).get("ok"):
        problems.append("the E.2 self-check is missing or failed")
    if raw.get("params") != engine_params():
        problems.append("engine is not Params() (the M0c engine)")
    fixed = dict(strength=STRENGTH, settle_ms=SETTLE_MS, read_ms=READ_MS, window_ms=WINDOW_MS, over_spikes=OVER_SPIKES)
    problems += [f"{k} {raw.get(k)!r} != {v!r}" for k, v in fixed.items() if raw.get(k) != v]
    odours = raw.get("odours", [])
    if len(odours) != N_ODOURS or odours_digest(odours) != ODOURS_DIGEST:
        problems.append(f"odours are not E.1's 41 candidates (n {len(odours)})")
    moves = {}
    for o in odours:
        moves.setdefault(int(o["turn"]), []).append(o["move"])
    rows = raw.get("rows", [])
    keys = [(int(r["turn"]), int(r["seed"])) for r in rows]
    if len(keys) != len(set(keys)):
        problems.append("duplicate (turn, seed) rows")
    want = {(t, s) for t in moves for s in SEEDS}
    if set(keys) != want:
        problems.append(f"rows cover {len(set(keys) & want)} of {len(want)} (turn, seed) cells, "
                        f"{len(set(keys) - want)} foreign")
    for r in rows:
        w = r.get("max_win", [])
        if len(w) != len(moves.get(int(r["turn"]), [])) or any(not isinstance(x, int) or x < 0 for x in w):
            problems.append(f"row (turn {r['turn']}, seed {r['seed']}): max_win {w!r} does not match its candidates")
    if problems:
        raise ValueError("; ".join(problems))
    over = [{"turn": int(r["turn"]), "move": moves[int(r["turn"])][j], "seed": int(r["seed"]), "max_win": int(w)}
            for r in rows for j, w in enumerate(r["max_win"]) if w >= OVER_SPIKES]
    top = max(w for r in rows for w in r["max_win"])
    return {"condition": CONDITION, "fired": bool(over), "n_presentations": sum(len(r["max_win"]) for r in rows),
            "n_over": len(over), "max_win_spikes": int(top), "max_win_hz": top * 1000.0 / WINDOW_MS,
            "margin_spikes": OVER_SPIKES - 1 - int(top), "over": sorted(over, key=lambda o: -o["max_win"])[:50]}
```

- [ ] **Step 4: Write `scripts/run_m2_d6a.py`**

```python
#!/usr/bin/env python3
"""Spec G.8: run the D.6 (a) re-judgement and write its raw record (git-excluded; flymon/brain/d6a.py has the rule).

    uv run python scripts/run_m2_d6a.py --smoke     # turn 0 x seeds 400-401 -> results/m2/d6a/smoke.json
    uv run python scripts/run_m2_d6a.py             # 41 odours x seeds 400-463 -> results/m2/d6a/g8.json

The engine is Params() (the M0c engine, every H.2 mode off); nothing on the command line changes it. Before measuring,
a full run re-presents two of E.1's recorded cells (turn 12 seed 203 — E.2's 150.0 Hz maximum — and turn 0 seed 200)
and requires the read-window KC record to equal results/m2/candidate_map.json bit for bit and the sliding maximum to be
at least E.2's 200 ms tile maximum. The pool is called once per block of 8 seeds, far below FlyPool's 3600 s timeout.
Exit codes: 0 done, whatever D.6 (a) reads; 2 refused — not at the repository root, tracked files dirty on a full run,
the odours are not E.1's, the output exists (move it aside first), or the E.2 self-check is missing or fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flymon.brain import d6a
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool

NPZ = "data/malecns.npz"
OUT_DIR = Path("results/m2/d6a")
CODE = ("flymon/brain/d6a.py", "flymon/brain/h4_jobs.py", "flymon/brain/h4_pairs.py", "scripts/run_m2_d6a.py")
SEED_BLOCK = 8
E2_RAW = Path("results/m2/candidate_map.json")     # E.1's recorded candidate map (git-excluded)
E2_CELLS = ((12, 203), (0, 200))                   # E.2's 150.0 Hz maximum, and the first recorded cell


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    dirty = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": dirty}


def e2_check(pool, by_turn: dict) -> dict:
    """Re-present E2_CELLS: read-window KC activity and spike count equal E.1's record; sliding max >= its tile max."""
    rec = json.loads(E2_RAW.read_text())
    ref = {tuple(k): r for k, r in zip(rec["index"], rec["results"])}
    res = pool.run_jobs(d6a.d6a_job, [dict(odors=by_turn[t], seed=s) for t, s in E2_CELLS])
    cells, ok = [], True
    for (t, s), r in zip(E2_CELLS, res):
        for j, c in enumerate(ref[(t, s)]["candidates"]):
            same = r["kc_active_frac"][j] == c["kc_active_frac"] and r["kc_spikes"][j] == c["kc_spikes"]
            ge = r["max_win"][j] * 1000.0 / d6a.WINDOW_MS >= c["kc_max_hz_sub"]
            ok = ok and same and ge
            cells.append({"turn": t, "seed": s, "candidate": j, "read_equal": same, "max_win": r["max_win"][j],
                          "e2_tile_max_hz": c["kc_max_hz_sub"]})
    return {"ok": ok, "cells": cells}


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if not (Path("flymon/brain/d6a.py").is_file() and Path(NPZ).is_file()):
        return refuse("run from the repository root with data/malecns.npz present")
    out = OUT_DIR / ("smoke.json" if a.smoke else "g8.json")
    if out.exists():
        return refuse(f"{out} exists; move it aside first")
    git = git_state()
    if git["dirty"] and not a.smoke:
        return refuse(f"tracked files are dirty {git['dirty']}")
    if not a.smoke and not E2_RAW.is_file():
        return refuse(f"{E2_RAW} (E.1's record) is needed for the self-check")
    code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in CODE}      # hashed before measuring
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    odours = d6a.candidate_odours(pops)
    if len(odours) != d6a.N_ODOURS or d6a.odours_digest(odours) != d6a.ODOURS_DIGEST:
        return refuse("the candidate odours are not E.1's 41 (digest differs)")
    del conn
    by_turn = {}
    for o in odours:
        by_turn.setdefault(o["turn"], []).append(o["odor"])
    turns = [0] if a.smoke else sorted(by_turn)
    seeds = list(d6a.SEEDS[:2] if a.smoke else d6a.SEEDS)
    started, t0 = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), time.perf_counter()
    rows = []
    with FlyPool(NPZ, Params(), [{} for _ in range(a.workers)], workers=a.workers) as pool:
        print(f"pool up in {time.perf_counter() - t0:.0f}s; {len(turns)} turns x {len(seeds)} seeds", flush=True)
        check = None if a.smoke else e2_check(pool, by_turn)
        if check is not None:
            print(f"E.2 self-check {'ok' if check['ok'] else 'FAILED'}: {check['cells']}", flush=True)
            if not check["ok"]:
                return refuse("the E.2 self-check failed; nothing measured")
        for i in range(0, len(seeds), SEED_BLOCK):
            block = seeds[i:i + SEED_BLOCK]
            cells = [(t, s) for t in turns for s in block]
            res = pool.run_jobs(d6a.d6a_job, [dict(odors=by_turn[t], seed=s) for t, s in cells])
            rows += [{"turn": t, **r} for (t, _), r in zip(cells, res)]
            top = max(w for r in rows for w in r["max_win"])
            print(f"seeds {block[0]}-{block[-1]} done, {time.perf_counter() - t0:.0f}s, max window so far {top}",
                  flush=True)
    raw = {"what": "spec G.8: D.6 (a) re-judgement (E.7 #4) - raw record; the rule is flymon/brain/d6a.py judge()",
           "smoke": bool(a.smoke), "started_utc": started, "wall_s": round(time.perf_counter() - t0, 1), "git": git,
           "code_sha256": code,
           "params": d6a.engine_params(), "odours": odours, "odours_digest": d6a.ODOURS_DIGEST, "seeds": seeds,
           "strength": d6a.STRENGTH, "settle_ms": d6a.SETTLE_MS, "read_ms": d6a.READ_MS, "window_ms": d6a.WINDOW_MS,
           "over_spikes": d6a.OVER_SPIKES, "workers": a.workers, "e2_check": check, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=1) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out} ({len(rows)} rows, {raw['wall_s'] / 60:.1f} min)", flush=True)
    if not a.smoke:
        v = d6a.judge(raw)
        print(f"D.6 (a) {'MET' if v['fired'] else 'not met'}: {v['n_over']}/{v['n_presentations']} presentations over, "
              f"max window {v['max_win_spikes']} spikes ({v['max_win_hz']:.0f} Hz), margin {v['margin_spikes']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest -q -o addopts="" tests/brain/test_d6a.py tests/test_run_m2_d6a.py`
Expected: `25 passed`. `test_candidate_odours_are_e1s_41` needs `data/malecns.npz` and, for its bit-exact branch, the git-excluded `results/m2/candidate_map_meta.json` — both are present in this worktree; if the npz were missing it would skip, which is not acceptable here.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q -rfE -o addopts=""`
Expected: **508 passed, 1 skipped, 1 xfailed** (483 + 25).

- [ ] **Step 7: Commit**

```bash
git add flymon/brain/d6a.py scripts/run_m2_d6a.py tests/brain/test_d6a.py tests/test_run_m2_d6a.py
git commit -m "feat(m2): G.8 D.6 (a) re-judgement — E.1's 41 odours x seeds 400-463, sliding 200 ms window, judge() and runner with an E.2 self-check"
```

---

### R1 (controller, not a subagent): run G.8

Only after Task 1 is reviewed and the tree is clean (`git status --porcelain` empty — the runner refuses dirty tracked files on a full run).

1. Smoke (about 20 s): `uv run python scripts/run_m2_d6a.py --smoke > .superpowers/runs/2026-09-22-m2-g8-smoke.log 2>&1; echo "exit $?" >> .superpowers/runs/2026-09-22-m2-g8-smoke.log` with `run_in_background: true`. Expect `exit 0`, turn 0 × seeds 400–401, a max window in the high 20s (planning smoke: 29, 28, 27 / 29, 28, 25).
2. Full run: `uv run python scripts/run_m2_d6a.py > .superpowers/runs/2026-09-22-m2-g8.log 2>&1; echo "exit $?" >> .superpowers/runs/2026-09-22-m2-g8.log` with `run_in_background: true` and **no timeout argument**. Estimate 8–15 min (planning smoke: about 2.7 s per presentation with 16 workers, 2,624 presentations; an estimate, not a measurement of the full run). The log must show `E.2 self-check ok` before the first seed block; a failed self-check exits 2 having measured nothing.
3. On `exit 0`, read only the last lines (`tail -3`): the `D.6 (a) MET / not met` line. Record in the ledger: commit, wall time, `n_over/2624`, max window, margin. **Either outcome is a record, not a stop** (reading 6) — continue with R2 when Task 2 is done.

---

### Task 2: the no-go summary writer and the go condition as strict xfails

**Files:**
- Create: `scripts/write_m2_nogo_summary.py`
- Test: `tests/test_write_m2_nogo_summary.py`, `tests/test_m2_go.py`

**Interfaces:**
- Consumes: `d6a.judge(raw)` (Task 1; `ValueError` on a broken record). Committed summaries: `results/summary/m2_oracle.json` (`counts.{testable, pairs, odd_gate_pool, even_pilot_pool}`, `design_pair_testable`), `m2_encoders.json` (`even.scores[E].score`, `even.decision.winner`), `m0d.json` (`h4.run_id`, `h4.git.commit`, `h4.h4.{outcome, selection.eligible, combos[C].oracle.aggregate.{testable_b, n_b, T_b, F_a, testable_a, n_a}}`), `m2_probe.json` (`d6_ruling.a_runaway.{fired, n_presentations, max_kc_sub_window_hz, margin_hz}`, `d6_ruling.b_candidate_ratio.{fired, n_turns_over, n_turns, ratio_max}`). Git-excluded: the two ceiling summaries (`commit`, `script`, `script_sha256`, `smoke`, `h4_run`, optional `family`, `engines.{C3,C0}.aggregate`) and `results/m2/d6a/g8.json`.
- Produces: `results/summary/m2_nogo.json` (written by the controller in R2) with keys `what, written_utc, git_commit, git_dirty, inputs_sha256, code_sha256, name, not_a_d6c_fail, learning_unit_test, chain{g10_oracle, g12_encoders, h4, h4a8_ceiling{rule, families{freq, all}, reading}}, d6{a, b, c}, std_condition_fired, decisions`. Tasks 3–5 quote numbers from it.

- [ ] **Step 1: Write the failing tests**

`tests/test_write_m2_nogo_summary.py`:

```python
"""scripts/write_m2_nogo_summary.py (spec appendix I): the no-go chain is derived from the recorded summaries and every
link refuses when it does not hold — the H.4a.7 reading at its boundaries, H.4's outcome and bar, G.12's stop, the
ceilings' identity and provenance, the G.8 record, and a learning unit test that ran."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("write_m2_nogo_summary", Path("scripts/write_m2_nogo_summary.py").resolve())
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ("m2_oracle", "m2_encoders", "m0d", "m2_probe")
FREQ_COMMIT = "ad77490bee47f4e5d744092bace300cd85a4f4e1"      # H.4a.8: the freq ceiling ran at ad77490
FREQ_SHA = "f1589bfe9f9d98ede5fbb1b725b2a001e1fca8c7bdb0bbe4f73d77e3b247ad07"
A_OK = {"condition": "c", "fired": False, "n_presentations": 2624, "n_over": 0, "max_win_spikes": 30,
        "max_win_hz": 150.0, "margin_spikes": 0, "over": []}


def _agg(tb, fa, n_b=21):
    return {"testable_b": tb, "n_b": n_b, "T_b": tb / n_b, "F_a": fa, "testable_a": 2, "n_a": 18, "bar": False}


def _ceiling(family, c3=(2, 0), c0=(2, 1)):
    s = {"commit": FREQ_COMMIT, "script": w.CEILING_SCRIPT, "script_sha256": FREQ_SHA, "smoke": False,
         "h4_run": "results/m0d/h4/runs/20260921T174430Z-7986f4.json",
         "engines": {"C3": {"aggregate": _agg(*c3)}, "C0": {"aggregate": _agg(*c0)}}}
    if family == "all":
        s["family"] = "all"
    return s


@pytest.fixture
def src(monkeypatch):
    """The committed summaries as they are, the two ceilings as H.4a.8 recorded them, and a G.8 verdict stand-in."""
    s = {k: json.loads((ROOT / w.INPUTS[k]).read_text()) for k in COMMITTED}
    s["ceiling_freq"], s["ceiling_all"] = _ceiling("freq", (2, 0), (2, 1)), _ceiling("all", (3, 0), (4, 1))
    s["g8"] = {"stand-in": True}
    monkeypatch.setattr(w.d6a, "judge", lambda raw: dict(A_OK))
    return s


@pytest.mark.parametrize("tb, fa, want", [
    (14, 2, "A"), (21, 4, "A"), (14, 1, "user"), (13, 4, "user"), (11, 2, "user"), (10, 4, "B"), (3, 0, "B"), (0, 0, "B")])
def test_ceiling_reading_at_the_h4a7_boundaries(tb, fa, want):
    assert w.ceiling_reading(tb, 21, fa) == want


def test_ceiling_reading_refuses_another_pair_count():
    with pytest.raises(w.Refused, match="21"):
        w.ceiling_reading(3, 20, 0)


def test_derive_reads_b_and_records_the_chain(src):
    r = w.derive(src, learning_ran=False)
    ch = r["chain"]
    assert r["not_a_d6c_fail"] is True and r["learning_unit_test"]["ran"] is False
    assert (ch["g10_oracle"]["testable"], ch["g10_oracle"]["pairs"], ch["g10_oracle"]["design_pair_testable"]) == (8, 34, True)
    assert ch["g12_encoders"]["winner"] is None and ch["g12_encoders"]["best"] < 0.5
    assert ch["h4"]["outcome"] == "STOP_LOW_T_B" and ch["h4"]["eligible"] == ["C3"]
    assert {c: (a["testable_b"], a["F_a"]) for c, a in ch["h4"]["combos"].items()} == {"C0": (4, 1), "C1": (2, 0), "C3": (7, 2)}
    fams = ch["h4a8_ceiling"]["families"]
    assert {f: (v["C3"]["testable_b"], v["C3"]["F_a"], v["reading"]) for f, v in fams.items()} == \
        {"freq": (2, 0, "B"), "all": (3, 0, "B")}
    assert ch["h4a8_ceiling"]["reading"] == "B"
    assert r["d6"]["c"]["status"] == "not measured" and r["d6"]["b"]["fired"] is True
    assert r["d6"]["a"]["fired"] is False and r["d6"]["a"]["e2_record"]["max_kc_sub_window_hz"] == 150.0
    assert r["std_condition_fired"] is True          # (b) fired in E.2; D.6 is an OR


def test_d6a_met_is_recorded(src, monkeypatch):
    monkeypatch.setattr(w.d6a, "judge", lambda raw: {**A_OK, "fired": True, "n_over": 3, "max_win_spikes": 33})
    assert w.derive(src, learning_ran=False)["d6"]["a"]["fired"] is True


@pytest.mark.parametrize("family, c3", [("freq", (11, 0)), ("all", (11, 0)), ("all", (14, 2)), ("freq", (14, 1))])
def test_refuses_when_a_ceiling_family_does_not_read_b(src, family, c3):
    src[f"ceiling_{family}"]["engines"]["C3"]["aggregate"] = _agg(*c3)
    with pytest.raises(w.Refused, match="does not read B"):
        w.derive(src, learning_ran=False)


def test_ten_of_21_still_reads_b(src):
    src["ceiling_all"]["engines"]["C3"]["aggregate"] = _agg(10, 4)
    assert w.derive(src, learning_ran=False)["chain"]["h4a8_ceiling"]["families"]["all"]["reading"] == "B"


def test_refuses_when_the_learning_unit_test_ran(src):
    with pytest.raises(w.Refused, match="ran"):
        w.derive(src, learning_ran=True)


def test_refuses_when_h4_did_not_stop(src):
    src["m0d"]["h4"]["h4"]["outcome"] = "SELECTED"
    with pytest.raises(w.Refused, match="SELECTED"):
        w.derive(src, learning_ran=False)


def test_refuses_when_a_combination_is_at_the_bar_whatever_the_stored_flag(src):
    agg = src["m0d"]["h4"]["h4"]["combos"]["C3"]["oracle"]["aggregate"]
    agg.update(T_b=0.5, testable_b=11, F_a=2, bar=False)
    with pytest.raises(w.Refused, match="C3"):
        w.derive(src, learning_ran=False)


def test_just_below_the_bar_is_not_at_it(src):
    src["m0d"]["h4"]["h4"]["combos"]["C3"]["oracle"]["aggregate"].update(T_b=10 / 21, F_a=4)
    assert w.derive(src, learning_ran=False)["chain"]["h4"]["outcome"] == "STOP_LOW_T_B"


@pytest.mark.parametrize("winner, score", [("E1", 0.3), (None, 0.5)])
def test_refuses_when_g12_did_not_stop(src, winner, score):
    src["m2_encoders"]["even"]["decision"]["winner"] = winner
    src["m2_encoders"]["even"]["scores"]["E2"]["score"] = score
    with pytest.raises(w.Refused, match="G.12"):
        w.derive(src, learning_ran=False)


@pytest.mark.parametrize("breakage", [
    lambda s: s.update(smoke=True),
    lambda s: s.update(h4_run="results/m0d/h4/runs/other.json"),
    lambda s: s.update(family="freq"),                 # the all-family summary must say all
])
def test_refuses_a_ceiling_that_is_not_h4a8s(src, breakage):
    breakage(src["ceiling_all"])
    with pytest.raises(w.Refused, match="all-family ceiling"):
        w.derive(src, learning_ran=False)


def test_a_g8_problem_refuses(src, monkeypatch):
    def bad(raw):
        raise ValueError("seeds are not 400-463")
    monkeypatch.setattr(w.d6a, "judge", bad)
    with pytest.raises(w.Refused, match="G.8 record: seeds"):
        w.derive(src, learning_ran=False)


def test_ceiling_provenance_binds_to_committed_code():
    w.ceiling_provenance(_ceiling("freq"))                              # ad77490's script hashes to f1589bfe…
    with pytest.raises(w.Refused, match="hash"):
        w.ceiling_provenance({**_ceiling("freq"), "script_sha256": "0" * 64})
    with pytest.raises(w.Refused, match="ancestor"):
        w.ceiling_provenance({**_ceiling("freq"), "commit": "f" * 40})
    with pytest.raises(w.Refused, match="script"):
        w.ceiling_provenance({**_ceiling("freq"), "script": "scripts/other.py"})


def test_main_refuses_a_dirty_tree_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w, "git_state", lambda out: {"git_commit": "abc", "dirty": ["README.md"]})
    out = tmp_path / "m2_nogo.json"
    assert w.main(["--out", str(out)]) == 2 and "dirty" in capsys.readouterr().err and not out.exists()


def test_main_refuses_missing_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w, "git_state", lambda out: {"git_commit": "abc", "dirty": []})
    monkeypatch.chdir(tmp_path)
    assert w.main(["--out", str(tmp_path / "x.json")]) == 2 and "missing inputs" in capsys.readouterr().err


# ---------------------------------------------------------------- the recorded inputs (git-excluded ones present)
REAL = [ROOT / p for p in w.INPUTS.values()]


@pytest.mark.skipif(not all(p.exists() for p in REAL), reason="git-excluded inputs (ceilings, G.8) not present")
def test_the_recorded_inputs_read_b_and_the_committed_summary_matches():
    src = {k: json.loads((ROOT / p).read_text()) for k, p in w.INPUTS.items()}
    body = w.derive(src, learning_ran=(ROOT / w.LEARNING_SUMMARY).exists())
    assert body["chain"]["h4a8_ceiling"]["reading"] == "B"
    committed = ROOT / w.OUT
    if committed.exists():                       # the committed record is what the code derives from its inputs now
        rec = json.loads(committed.read_text())
        assert {k: rec[k] for k in body} == json.loads(json.dumps(body, ensure_ascii=False))
        assert rec["inputs_sha256"] == {p: w._sha256(ROOT / p) for p in w.INPUTS.values()}
```

`tests/test_m2_go.py`:

```python
"""Spec 5's M2 go condition, kept as strict expected failures (spec appendix I; the A.5 precedent). They are not removed
or relaxed: the day a tested engine reaches the bar, or a learning unit test passes, they XPASS and the suite breaks."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M0D = ROOT / "results/summary/m0d.json"
LEARNING = ROOT / "results/summary/m2_learning.json"      # F.10 #4's summary: written only if the learning test runs


@pytest.mark.xfail(strict=True, reason="M2 no-go — 시험 불성립 (spec appendix I): no tested engine reaches "
                                       "T_b >= 0.5 and F_a >= 2 (H.4 STOP_LOW_T_B; best C3 7/21)")
def test_m2_learning_unit_test_can_be_built():
    """G.14.4 / H.4: spec 5's learning unit test can be built only on an engine whose idealised specific edit makes at
    least half of the (b) pairs testable (T_b >= 0.5) and at least two naive-balanced (a) pairs testable (F_a >= 2)."""
    combos = json.loads(M0D.read_text())["h4"]["h4"]["combos"]
    aggs = [c["oracle"]["aggregate"] for c in combos.values()]
    assert any(a["T_b"] >= 0.5 and a["F_a"] >= 2 for a in aggs)


@pytest.mark.xfail(strict=True, reason="M2 no-go — 시험 불성립 (spec appendix I): spec 5's learning unit test never "
                                       "ran (F v3 withdrawn unrun, F v4 not written)")
def test_m2_go():
    """Spec 5, M2: naive d' ~ 0, d' >= 1 after 20 rewards, a drop after 20 punishments — F.7's overall verdict PASS."""
    assert json.loads(LEARNING.read_text())["outcome"] == "PASS"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q -rxX -o addopts="" tests/test_write_m2_nogo_summary.py tests/test_m2_go.py`
Expected: a collection error for `tests/test_write_m2_nogo_summary.py` (`FileNotFoundError: scripts/write_m2_nogo_summary.py`); `tests/test_m2_go.py` already reports `2 xfailed` — those two tests assert on recorded data, not on new code, and must stay xfailed.

- [ ] **Step 3: Write `scripts/write_m2_nogo_summary.py`**

```python
#!/usr/bin/env python3
"""Write results/summary/m2_nogo.json: the M2 no-go record (spec appendix I, "M2 no-go — 시험 불성립").

This is NOT a D.6 (c) FAIL. Spec 5's learning unit test never ran (F v3 withdrawn unrun, F v4 not written), and G.1 and
H.8 say no G or H data judges (c). The record derives from the committed summaries and two git-excluded recorded
measurements the chain the no-go rests on and D.6's status, and refuses to write when a link does not hold, so a claim
in prose cannot survive into the summary (the spec E rule, as in write_m2_probe_summary.py):

  G.10    oracle: testable pairs / all pairs                     results/summary/m2_oracle.json
  G.12    encoder comparison: best score < 0.5 (G.11 rule 4)     results/summary/m2_encoders.json
  H.4     selection STOP_LOW_T_B, no combination at the bar      results/summary/m0d.json block "h4"
  H.4a.8  specificity ceiling read by the H.4a.7 rule: B         results/m0d/diag/h4_specificity_ceiling{,_all}_summary.json
  D.6     (a) G.8 re-judgement, (b) E.2, (c) not measured        results/m2/d6a/g8.json, results/summary/m2_probe.json

The ceiling summaries are git-excluded, so each is bound to committed code: the commit it names must be an ancestor of
HEAD, and the ceiling script at that commit must hash to the sha256 it recorded.

Run from the repository root on a clean tree:  uv run python scripts/write_m2_nogo_summary.py
Exit 2: refused (dirty tree, a missing input, or a link that does not hold); nothing is written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from flymon.brain import d6a

OUT = Path("results/summary/m2_nogo.json")
INPUTS = {
    "m2_oracle": "results/summary/m2_oracle.json",
    "m2_encoders": "results/summary/m2_encoders.json",
    "m0d": "results/summary/m0d.json",
    "m2_probe": "results/summary/m2_probe.json",
    "ceiling_freq": "results/m0d/diag/h4_specificity_ceiling_summary.json",
    "ceiling_all": "results/m0d/diag/h4_specificity_ceiling_all_summary.json",
    "g8": "results/m2/d6a/g8.json",
}
CEILINGS = (("freq", "ceiling_freq"), ("all", "ceiling_all"))
CEILING_SCRIPT = "docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py"
LEARNING_SUMMARY = Path("results/summary/m2_learning.json")   # F.10 #4's summary; it exists only if the test ran
CODE = ("scripts/write_m2_nogo_summary.py", "flymon/brain/d6a.py")
BAR_T_B, BAR_F_A = 0.5, 2              # G.14.4 / H.4 absolute bar
ENCODER_STOP = 0.5                     # G.11 rule 4
CEIL_A, CEIL_B, N_B = 14, 10, 21       # H.4a.7, C3's (b) axis
NAME = "M2 no-go — 시험 불성립 (the learning unit test could not be built)"
DECISIONS = {                          # the user's, 2026-09-22 (spec appendix I) — recorded, not derived
    "route": "3번 (범위 축소 / M2 no-go 기록 + 주장 재설계), G.14.4 row 5 and H.3a.10 ②, under the H.4a.7 rule",
    "name": "M2 no-go — 시험 불성립, kept apart from F.7's FAIL (= D.6 (c) met)",
    "claims": "learning tests limited to the design pair (a mechanism control); 4.3 criterion 1 is not supported in "
              "its current form and is reported as untested; M3 and M4 are on hold, not discarded",
    "e3_std": "STD is not designed now; it is the first question of the next claim declaration",
    "g8": "D.6 (a) re-judged by G.8 for the record",
}


class Refused(Exception):
    """A link of the no-go chain does not hold; the message names it."""


def ceiling_reading(testable_b: int, n_b: int, f_a: int) -> str:
    """H.4a.7 on C3: "A" iff testable (b) >= 14/21 and F_a >= 2; "B" iff testable (b) <= 10/21; else "user"."""
    if n_b != N_B:
        raise Refused(f"a ceiling has {n_b} (b) pairs; H.4a.7 reads {N_B}")
    if testable_b >= CEIL_A and f_a >= BAR_F_A:
        return "A"
    return "B" if testable_b <= CEIL_B else "user"


def at_bar(agg: dict) -> bool:
    """G.14.4 / H.4's absolute bar, recomputed from T_b and F_a (never read from the stored flag)."""
    return agg["T_b"] >= BAR_T_B and agg["F_a"] >= BAR_F_A


def _agg(agg: dict) -> dict:
    return {k: agg[k] for k in ("testable_b", "n_b", "T_b", "F_a", "testable_a", "n_a")}


def ceiling_provenance(summary: dict) -> None:
    """The ceiling ran at a commit on HEAD's history, and the ceiling script there hashes to what it recorded."""
    commit = summary.get("commit", "")
    if summary.get("script") != CEILING_SCRIPT:
        raise Refused(f"ceiling script {summary.get('script')!r} is not {CEILING_SCRIPT}")
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    capture_output=True).returncode != 0:
        raise Refused(f"ceiling commit {commit[:12]!r} is not an ancestor of HEAD")
    blob = subprocess.run(["git", "show", f"{commit}:{CEILING_SCRIPT}"], capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest() != summary.get("script_sha256"):
        raise Refused(f"the ceiling script at {commit[:12]} does not hash to the recorded sha256")


def derive(src: dict, learning_ran: bool) -> dict:
    """The record body from the parsed inputs (keys of INPUTS). Raises Refused when a link of the chain does not hold."""
    if learning_ran:
        raise Refused(f"{LEARNING_SUMMARY} exists: the learning unit test ran, so the test was not 'not constructible'")
    oracle = src["m2_oracle"]
    g10 = {k: oracle["counts"][k] for k in ("testable", "pairs", "odd_gate_pool", "even_pilot_pool")}
    g10["design_pair_testable"] = oracle["design_pair_testable"]

    enc = src["m2_encoders"]["even"]
    scores = {e: s["score"] for e, s in sorted(enc["scores"].items())}
    best = max(scores.values())
    if enc["decision"]["winner"] is not None or best >= ENCODER_STOP:
        raise Refused(f"G.12 did not stop: winner {enc['decision']['winner']!r}, best score {best:.3f}")
    g12 = {"scores": scores, "best": best, "winner": None, "rule": "G.11 rule 4: best score < 0.5 -> stop and report"}

    blk = src["m0d"]["h4"]
    h4 = blk["h4"]
    combos = {c: _agg(v["oracle"]["aggregate"]) for c, v in sorted(h4["combos"].items())}
    reached = sorted(c for c, a in combos.items() if at_bar(a))
    if h4["outcome"] != "STOP_LOW_T_B" or reached:
        raise Refused(f"H.4 is not a STOP below the bar: outcome {h4['outcome']}, at the bar {reached}")
    h4_rec = {"run_id": blk["run_id"], "commit": blk["git"]["commit"], "outcome": h4["outcome"],
              "eligible": h4["selection"]["eligible"], "combos": combos,
              "bar": f"T_b >= {BAR_T_B} and F_a >= {BAR_F_A} (G.14.4, H.4)"}

    h4_run = f"results/m0d/h4/runs/{blk['run_id']}.json"
    families = {}
    for fam, key in CEILINGS:
        s = src[key]
        if s.get("smoke") is not False or s.get("h4_run") != h4_run or s.get("family", "freq") != fam:
            raise Refused(f"{INPUTS[key]} is not the full {fam}-family ceiling of H.4 run {blk['run_id']}")
        c3 = s["engines"]["C3"]["aggregate"]
        families[fam] = {"commit": s["commit"], "script_sha256": s["script_sha256"], "C3": _agg(c3),
                         "C0": _agg(s["engines"]["C0"]["aggregate"]),
                         "reading": ceiling_reading(c3["testable_b"], c3["n_b"], c3["F_a"])}
    readings = {f: v["reading"] for f, v in families.items()}
    if set(readings.values()) != {"B"}:
        raise Refused(f"the H.4a.7 rule does not read B on every ceiling family: {readings}")
    ceiling = {"rule": f"H.4a.7 on C3: A iff testable (b) >= {CEIL_A}/{N_B} and F_a >= {BAR_F_A}; "
                       f"B iff <= {CEIL_B}/{N_B}; otherwise the user's call",
               "families": families, "reading": "B"}

    try:
        a = d6a.judge(src["g8"])
    except ValueError as e:
        raise Refused(f"G.8 record: {e}") from None
    e2 = src["m2_probe"]["d6_ruling"]
    b = e2["b_candidate_ratio"]
    d6 = {
        "a": {"source": "G.8 re-judgement (E.7 #4), results/m2/d6a/g8.json", **a,
              "e2_record": {k: e2["a_runaway"][k] for k in ("fired", "n_presentations", "max_kc_sub_window_hz",
                                                             "margin_hz")}},
        "b": {"source": "E.2, results/summary/m2_probe.json", "fired": b["fired"], "n_turns_over": b["n_turns_over"],
              "n_turns": b["n_turns"], "ratio_max": b["ratio_max"]},
        "c": {"status": "not measured", "why": "spec 5's learning unit test never ran (F v3 withdrawn unrun, F v4 not "
                                               "written); G.1 and H.8: no G or H data judges (c)"},
    }
    return {"name": NAME, "not_a_d6c_fail": True, "learning_unit_test": {"ran": False, "summary": str(LEARNING_SUMMARY)},
            "chain": {"g10_oracle": g10, "g12_encoders": g12, "h4": h4_rec, "h4a8_ceiling": ceiling},
            "d6": d6, "std_condition_fired": bool(a["fired"] or b["fired"]), "decisions": DECISIONS}


def git_state(out: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    dirty = [ln[3:] for ln in run("status", "--porcelain").splitlines() if ln.strip() and ln[3:] != out.as_posix()]
    return {"git_commit": run("rev-parse", "HEAD").strip(), "dirty": dirty}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    try:
        git = git_state(a.out)
        if git["dirty"]:
            raise Refused(f"the tree is dirty: {git['dirty']}")
        missing = [p for p in INPUTS.values() if not Path(p).is_file()]
        if missing:
            raise Refused(f"missing inputs {missing}")
        src = {k: json.loads(Path(p).read_text()) for k, p in INPUTS.items()}
        for _, key in CEILINGS:
            ceiling_provenance(src[key])
        body = derive(src, LEARNING_SUMMARY.exists())
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    record = {"what": "spec appendix I: the M2 no-go record, derived by scripts/write_m2_nogo_summary.py",
              "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git["git_commit"], "git_dirty": False,
              "inputs_sha256": {p: _sha256(p) for p in INPUTS.values()}, "code_sha256": {p: _sha256(p) for p in CODE},
              **body}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    ch, d = body["chain"], body["d6"]
    print(f"wrote {a.out}: H.4 {ch['h4']['outcome']}, ceiling C3 "
          + ", ".join(f"{f} {v['C3']['testable_b']}/{v['C3']['n_b']}" for f, v in ch["h4a8_ceiling"]["families"].items())
          + f" -> B; D.6 (a) {'met' if d['a']['fired'] else 'not met'} (max window {d['a']['max_win_spikes']} spikes), "
          f"(b) {'met' if d['b']['fired'] else 'not met'}, (c) not measured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest -q -rsxX -o addopts="" tests/test_write_m2_nogo_summary.py tests/test_m2_go.py`
Expected: `29 passed, 1 skipped, 2 xfailed` while `results/m2/d6a/g8.json` does not exist yet (the skip is `test_the_recorded_inputs_read_b_and_the_committed_summary_matches`); `30 passed, 2 xfailed` once R1 has written it. `test_ceiling_provenance_binds_to_committed_code` runs `git` against this repository's history (commit `ad77490` must be an ancestor of HEAD).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q -rfE -o addopts=""` (not while another full suite runs; R1's G.8 run may be using the CPU — that only slows it).
Expected: **537 passed, 2 skipped, 3 xfailed** before R1 finishes, or **538 passed, 1 skipped, 3 xfailed** after.

- [ ] **Step 6: Commit**

```bash
git add scripts/write_m2_nogo_summary.py tests/test_write_m2_nogo_summary.py tests/test_m2_go.py
git commit -m "feat(m2): write_m2_nogo_summary derives the M2 no-go chain (G.10 -> G.12 -> H.4 -> H.4a.8 read by H.4a.7 -> D.6) and refuses on a broken link; M2 go condition as strict xfails"
```

---

### R2 (controller, not a subagent): write and commit the record

After R1 exited 0 and Task 2 is reviewed, on a clean tree:

1. `uv run python scripts/write_m2_nogo_summary.py` (seconds). Expect exit 0 and one line: `wrote results/summary/m2_nogo.json: H.4 STOP_LOW_T_B, ceiling C3 freq 2/21, all 3/21 -> B; D.6 (a) … (b) met, (c) not measured`. Exit 2 means a link does not hold: stop, read the `refused:` line, and take it to the user (it would contradict the recorded chain).
2. `uv run pytest -q -o addopts="" tests/test_write_m2_nogo_summary.py` → `30 passed` (the real-data test now also checks that the committed record equals what the code derives).
3. Commit: `git add results/summary/m2_nogo.json && git commit -m "results(m2): m2_nogo.json — M2 no-go (시험 불성립) derived; D.6 (a) <met|not met> by G.8 (<n_over>/2624, max window <k> spikes)"`.
4. Put into the Task 3–5 dispatch prompts: the Task 1 commit, the `m2_nogo.json` commit, `sha256sum scripts/write_m2_nogo_summary.py`, R1's wall time in minutes, and from `m2_nogo.json` `d6.a`: `fired`, `n_over`, `n_presentations`, `max_win_spikes`, `max_win_hz`, `margin_spikes` (and the first `over` entries if fired).

---

### Task 3: spec — appendix I, the G.8 result and dated pointers

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-flymon-design.md`

**Interfaces:**
- Consumes (from the dispatch prompt, all from R2): `T1_COMMIT` (Task 1's commit, short hash), `NOGO_COMMIT` (the `m2_nogo.json` commit), `WRITER_SHA` (full sha256 of `scripts/write_m2_nogo_summary.py`), `G8_MIN` (R1's wall time, minutes, one decimal), and `d6.a` from `results/summary/m2_nogo.json`: `G8_FIRED`, `G8_NOVER`, `G8_MAX` (`max_win_spikes`), `G8_HZ` (`max_win_hz`), `G8_MARGIN` (`margin_spikes`).
- Produces: appendix I with subsections I.1–I.7 (Tasks 4–5 cite "부록 I", "I.3", "I.5", "I.6", "I.7").

Every edit is an insertion; no existing sentence is changed or deleted. Where the text below branches on `G8_FIRED`, write only the branch that matches. After each step, `git diff --stat` must show only this file.

- [ ] **Step 1: The G.8 result.** Directly after the G.8 paragraph (the one ending "초과 제시가 하나라도 있으면 (a) 충족.") and before `### G.9 그 다음`, insert a blank line and:

```markdown
**결과(2026-09-22, 기록용 — 사용자 결정, 부록 I)**: `scripts/run_m2_d6a.py`(커밋 `{T1_COMMIT}`, 규칙은 `flymon/brain/d6a.py`의 `judge()`), 엔진 = `Params()`
(M0c 엔진 = H.3의 C0, H.2 모드 전부 꺼짐), 16워커 {G8_MIN}분. E.1의 41냄새(다이제스트 `29992673…`, `m2_probe.turn_odors`와 비트 동일) × 시드 400–463 =
{G8_NOVER 포함 문장 — 아래 둘 중 하나}.
실행 전 자기 검사: E.1 기록 두 칸(턴 12 시드 203 — E.2의 최대 150.0 Hz — 과 턴 0 시드 200)의 읽기 창 KC 활성·스파이크 수가 `results/m2/candidate_map.json`과
비트 동일했고, 이동 창 최대가 타일 최대 이상이었다(턴 12 시드 203 후보 1: 30 스파이크 = 150 Hz — 이동 창으로도 초과가 아니다). 원자료 `results/m2/d6a/g8.json`(git 제외),
요약 `results/summary/m2_nogo.json`의 `d6.a`.
```

with the sentence for `G8_FIRED = false`: `2624 제시에서 150 Hz 초과 0건, 최대 이동 창 {G8_MAX} 스파이크({G8_HZ} Hz, 여유 {G8_MARGIN} 스파이크) → **(a) 미충족**`; for `G8_FIRED = true`: `2624 제시 중 {G8_NOVER}건이 150 Hz 초과(최대 이동 창 {G8_MAX} 스파이크 = {G8_HZ} Hz) → **(a) 충족**`.

- [ ] **Step 2: E.2 and E.3 pointers.** After the E.2 table (the line starting `| (c) M2 학습 단위 시험 실패 |`) insert a blank line and `→ **(a) 재판정(2026-09-22, G.8 결과)**: {미충족 — 최대 이동 창 {G8_MAX} 스파이크 | 충족 — {G8_NOVER}/2624 제시}. (c)는 끝내 측정되지 않았다(부록 I.3).` At the end of E.3 (after the bullet "이 개정은 학습 단위 시험 **실행 전에** 기록됐다.") insert a blank line and `**종결(2026-09-22, 부록 I.4)**: (c)는 이 경로에서 생기지 않으므로 "함께 한 번에"의 판정은 오지 않는다. STD는 지금 설계하지 않고 다음 주장 선언의 첫 질문으로 넘긴다(사용자 결정).`

- [ ] **Step 3: F head and G.9.** Inside the blockquote at the head of appendix F, after its last line (`>   셋 다 판독 세포가 일부 시드에서 사실상 무발화라 **바닥 가드가 제외했을 뿐**이다.`), add the line `> **2026-09-22**: F v4는 쓰지 않는다 — M2 no-go(시험 불성립), 부록 I.` At the end of G.9's paragraph append ` (2026-09-22: F v4는 쓰지 않는다 — 부록 I.6.)`.

- [ ] **Step 4: H.4a.5, H.5, H.6.** In H.4a.5, directly after the bold result paragraph (ending "결정은 사용자의 몫이며 아직 내려지지 않았다.**") insert a blank line and `**결정(사용자, 2026-09-22)**: 특이성 천장(H.4a.7 판독 규칙을 결과 전에 고정, H.4a.8 결과 B)을 먼저 쟀고, 그 규칙대로 **M2 no-go 기록**으로 갔다 — 부록 I.` Directly under the headings `### H.5 확인 (한 번)` and `### H.6 동결과 인계` insert a blank line and `> **상태(2026-09-22)**: 실행하지 않는다 — M2 no-go, 부록 I.6.`

- [ ] **Step 5: spec 0, 4.3 and 5.** In section 0, after the **2차 주장** paragraph and before **정확히 구분해서 적을 것**, insert:

```markdown
> **상태(2026-09-22, 부록 I)**: **M2 no-go — 시험 불성립.** 1차 주장을 시험할 쌍이 이 커넥톰·판독·인코더에서 선언된 기준에 못 미쳐 학습 단위 시험(스펙 5)이
> 세워지지 않았다. 4.3 기준 1은 현재 형태로 지지되지 않는다(시험하지 않음 — "학습 안 됨"이 아니다). 학습 시험은 설계쌍(기계 대조)으로 한정한다.
> M3·M4와 2차 주장은 보류(폐기 아님)이며, 다시 여는 것은 새 주장 선언뿐이다.
```

In 4.3, after the paragraph starting "기준 1을 못 넘으면" insert a blank line and `**2026-09-22**: M4를 돌리지 않았으므로 기준 1–5는 **시험하지 않음**이다. 위의 "타입 조건부 학습 안 됨"은 M4 실패에 쓰는 문장이라 쓰지 않는다(부록 I.5).` In spec 5, append to the end of the M2 bullet ` **상태: no-go — 시험 불성립(2026-09-22, 부록 I). 학습 단위 시험은 돌지 않았다.**`, to the M3 and M4 bullets ` **상태: 보류(2026-09-22, 부록 I.6).**`, and to the M5 bullet ` **상태(2026-09-22): README "안 된 것"에 no-go를 싣는다. 뷰어·영상은 이번에 손대지 않는다(부록 I.7).**`

- [ ] **Step 6: Appendix I.** At the end of the file (after H.8's last bullet), insert a blank line and the appendix below, with the `{…}` values filled and the (a) row written for the actual `G8_FIRED`:

```markdown
## 부록 I. M2 no-go — 시험 불성립 (2026-09-22, 사용자 결정, 판정 코드 `scripts/write_m2_nogo_summary.py` sha256 `{WRITER_SHA}`)

**결정(사용자, 2026-09-22)**: G.14.4 5행·H.3a.10 ②가 이름 붙인 **3번(범위 축소 / M2 no-go 기록 + 주장 재설계)**으로 간다. 근거는 H.4의 `STOP_LOW_T_B`
(자격 조합 C3 하나, T_b 7/21 = 0.333, H.4a.5)와 특이성 천장을 H.4a.7 규칙으로 읽은 **B**(H.4a.8: C3 `freq` 2/21 · F_a 0, `all` 3/21 · F_a 0; C0 `freq` 2/21 · F_a 1,
`all` 4/21 · F_a 1)다. 요약 `results/summary/m2_nogo.json`(커밋 `{NOGO_COMMIT}`, `git_commit`·`git_dirty`·`inputs_sha256`)은 이 사슬과 D.6 상태를 기록된 요약에서
파생하고, 한 고리라도 성립하지 않으면 쓰기를 거부한다 — 학습 시험 요약이 있음, G.12가 멈추지 않음, H.4가 기준 아래 STOP이 아님(T_b·F_a에서 다시 계산),
천장 요약이 H.4 실행의 전체 실행이 아니거나 커밋된 코드와 묶이지 않음(이름 붙인 커밋이 HEAD의 조상이고 그 커밋의 천장 스크립트가 기록된 sha256과 같아야 한다),
어느 천장 가족이든 B가 아님, G.8 원자료가 완전하지 않음.

**이것은 D.6 (c) FAIL이 아니다.** 학습 단위 시험(스펙 5)은 한 번도 돌지 않았다 — F v3는 실행 전에 철회됐고(부록 F 머리) F v4는 쓰지 않았다(G.9, H.6).
G.1·H.8에 따라 G·H의 어떤 데이터도 (c)를 판정하지 않는다. 그래서 F.7의 `FAIL` 행(= D.6 (c) 충족 → STD 재설계)과 H.6 조치표 1행은 발동하지 않는다.
기록하는 것은 **시험 불성립**이다: 시험한 인코더(E0–E3, G.12)와 엔진(C0·C1·C3, H.4)에서, 이상화한 특이적 감소로 스펙 5의 시험을 세울 수 있는 쌍이 선언된 기준
(T_b ≥ 0.5 ∧ F_a ≥ 2, G.14.4)에 못 미쳤고, X에만 활성인 KC만 깎는 천장에서도 C3가 2–3/21에 그쳤다.

### I.1 근거 사슬 (측정된 것, 순서대로)

- **G.10**: 이상화한 최대 특이적 감소로도 후보쌍 8/34. 순진 d′ ≈ 0 ∧ 시험 가능 ∧ 가드 통과를 함께 만족하는 홀수 턴 게이트 후보 0쌍.
- **G.12·G.13**: 네 인코더 모두 점수 0.5 미만(최고 E0 0.222) — G.11 규칙 4의 멈춤. 병목은 (b) 축과 처벌 판독.
- **G.14.8**: 등급 APL을 그대로 바꿔 끼우면 판독 작동점이 무너진다(MBON05 직접 억제, KC 약 3%에서 MBON13 바닥) — 프로브는 판정 없이 중단, M0d로.
- **H.4**: C3 7/21(F_a 2) · C0 4/21(F_a 1) · C1 2/21(F_a 0) → `STOP_LOW_T_B`. C2는 H.3a.1에서 이미 탈락.
- **H.4a.6**: 학습 세기·시드 잡음·판독 가중은 원인이 아니다. 판독 바닥은 C3가 대부분 고쳤다(바닥 통과 14쌍 중 시험 가능 7). 남은 한계는 편집된 시냅스의
  X/Y 특이성(Y의 판독이 X의 54–80%만큼 같이 움직임).
- **H.4a.8**: 편집을 X 전용 KC로 제한하면 특이성은 얻지만(Y/X → 0) 처벌 조건이 무너진다((b)에서 −p ≥ 2인 쌍 C3 11 → 3 · 4). MBON13을 움직이는 X의 구동은
  대부분 Y와 공유된 KC에 있다.

### I.2 원인을 하나로 귀속하지 않는다 — 대안 설명 (F.7의 방식)

- 판독이 MBON13·MBON05 두 타입뿐이다. 풀 밖 MBON은 재지 않았다.
- 오라클은 이상화된 편집이다. 천장은 **현재 KC 부호**에서의 상한이라, 부호를 바꾸는 변경(인코더, 국소 APL, 발화율 정규화, LHPV3c1 이득, KC→KC 억제, 희소화)을
  묶지 않는다. 이들은 시험하지 않았다.
- 짝수 턴 8턴·(b) 21쌍뿐이고, 짝·홀 분할은 종족 분할이다(G.13-2).
- 채널→사구체 배정이 임의적이다(동일 총 ORN 구동에서 사구체 간 KC 구동 2867배, E.1-B).
- C0 대 C1·C3는 기저에서 교락돼 있다(H.3a.10). V가 APL→MBON05 억제에 직접 반응할 수 있다(H.4a.2).

### I.3 D.6 상태

| 조건 | 상태 | 근거 |
|---|---|---|
| (a) KC 하나라도 200 ms 이상 150 Hz 초과 | **{미충족 | 충족}** | G.8 재판정(G.8 결과): 2624 제시, 최대 이동 창 {G8_MAX} 스파이크({G8_HZ} Hz), 초과 {G8_NOVER}건. E.2의 격자 타일 기록(최대 150.0 Hz, 여유 0)을 대체한다 |
| (b) 같은 턴 후보 KC 비율 > 2 | **발동** | E.2: 16턴 중 3턴, 최대 2.50. 되돌리지 않는다(E.3) |
| (c) M2 학습 단위 시험 실패 | **미측정** | 학습 단위 시험이 돌지 않았다(시험 불성립) |

D.6은 OR이므로 STD 재검토 조건은 (b)만으로 이미 발동해 있다(`std_condition_fired` = true). H.6의 조치표는 채택 엔진을 전제하므로 적용하지 않고, 처리는 I.4가 한다.
{G8_FIRED = true일 때만 이 문장: (a)도 발동했으므로 H.6 조치표 2행의 내용("M3 전에 KC 폭주 대응을 새 선언으로")을 새 주장 선언의 입력으로 넘긴다 — M3는 어차피 보류다.}

### I.4 E.3의 종결

E.3은 STD 설계 여부를 (c)와 "함께 한 번에" 판정하기로 미뤘다. (c)는 이 경로에서 생기지 않으므로 그 미룸을 여기서 닫는다(G.13-6이 요구한 종결 조건).
**STD는 지금 설계하지 않고, 다음 주장 선언의 첫 질문으로 넘긴다**(사용자 결정). (b)의 후보 원인 자리는 ORN→PN 단계로 기록하되(E.3·E.5), 문헌 상수(f 0.78, τ 893 ms)로는
PN 출력이 무너져 C2가 자격점을 못 찾았다는 H.3a.1의 진단을 함께 적는다.

### I.5 이 기록이 닫지 않는 것

- D.6 (c).
- 4.3 기준 1–5 — **시험하지 않음**. "타입 조건부 학습 안 됨"은 4.3이 M4 실패에 쓰는 문장이라 여기 쓰지 않는다.
- 2채널 부류, 실제 가소성 규칙의 도달 정도(오라클은 T_b를 낮출 수만 있다), 풀 밖 판독.
- I.2의 KC 부호 변경들.
- M0d 확인 턴 세트(H.5)와 스펙 4.4의 M4 확인 세트 — 둘 다 쓰지 않았다. G.11 이후의 선택에는 짝수 턴만 썼다(홀수 턴은 G.10 오라클에서만 쟀다).

### I.6 멈추는 것

| 항목 | 상태 |
|---|---|
| F v4 (G.9, H.6) | 쓰지 않는다 |
| M0d H.5–H.7 | 돌리지 않는다. 엔진 기본값은 M0c 엔진 그대로(H.2 모드 전부 꺼짐). C3 역치 사본 `results/summary/m0d_h3_c3_thresholds.npz`는 기록으로 남는다 |
| 4.4 M2 동결 | 동결할 것이 없다. F.8의 무효 조건도 대상이 없다 |
| H.6 D.6 조치표 | 채택 엔진을 전제하므로 적용하지 않는다. D.6 처리는 I.3·I.4 |
| M3 | **보류**(F.7상 비-PASS는 M3를 막는다). 폐기가 아니다 |
| M4, 4.1 팔, 4.3 기준 | **보류**. 폐기가 아니다 |
| 스펙 5 M2 | "no-go — 시험 불성립, 2026-09-22"로 표시 |

### I.7 주장 재설계 (사용자 결정, 2026-09-22)

- **학습 시험은 설계쌍(기계 대조)으로 한정한다.** 설계쌍은 G.10에서 시험 가능(보상 18.69 / 처벌 −6.27, Y 번짐 0.03)이고 M0·M0c에서 채널별 냄새 특이 억제를
  보였다. 이것은 가소성 경로가 작동한다는 기계 대조이지 0절 1차 주장의 근거가 아니다.
- **4.3 기준 1은 현재 형태로 지지되지 않는다.** 이 커넥톰·판독·인코더에서 같은 턴 후보의 상대 타입 조건부 학습 분리를 세울 쌍이 기준에 못 미친다. 1차 주장을
  다시 세우려면 새 주장 선언이 KC 부호를 바꾸는 변경으로 MBON13을 움직이는 X 전용 구동을 새로 만들어야 한다(H.4a.8).
- **M3·M4는 보류**(폐기 아님). 2차 주장(승률 기여)은 M4에서 재므로 함께 보류된다. 다시 여는 것은 새 주장 선언뿐이다.
- 새 주장 선언의 첫 질문은 STD다(I.4). 출발점은 인계 노트 `docs/handoffs/2026-09-22-m2-nogo-claim-redesign.md`.
- 공개(스펙 7 "실패도 공개"): README "측정된 것 / 우리가 정한 것 / 안 된 것"에 이 부록을 한국어·영어로 요약한다. 뷰어·영상(3.7)은 이번에 손대지 않는다.
- M2 go 조건은 strict xfail로 코드에 남는다: `tests/test_m2_go.py::test_m2_learning_unit_test_can_be_built`(어떤 엔진이 T_b ≥ 0.5 ∧ F_a ≥ 2에 닿음)와
  `::test_m2_go`(`results/summary/m2_learning.json`의 판정 `PASS`). A.5 선례대로 삭제·완화하지 않고, 조건이 충족되는 날 XPASS로 깨진다.
```

- [ ] **Step 7: Check and commit**

Run: `git diff --stat` → only the spec, insertions only (`git diff | grep '^-[^-]'` prints nothing). `grep -n '{' docs/superpowers/specs/2026-09-14-flymon-design.md | grep -E '\{(T1_|NOGO_|WRITER_|G8_)'` prints nothing (every value filled). Check every number in appendix I against `results/summary/m2_nogo.json` and the cited sections.

```bash
git add docs/superpowers/specs/2026-09-14-flymon-design.md
git commit -m "docs(spec): appendix I — M2 no-go (시험 불성립) by user decision under H.4a.7; G.8 D.6 (a) result; E.3 closed to the next claim declaration; dated pointers in 0, 4.3, 5, E, F, G.9, H.4a.5, H.5, H.6"
```

---

### Task 4: README — the ledger catch-up, the no-go, English

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the same `G8_*` values as Task 3 (dispatch prompt); appendix I (Task 3).
- Produces: nothing later tasks read.

Edits are insertions; existing README text is not changed. The ledger section is `## 측정된 것 / 우리가 정한 것 / 안 된 것`; its last bullet is `- **안 된 것(운영)**: …` (two lines).

- [ ] **Step 1: Header.** In line 4 (`M0: 엔진과 flybrain 측정값 재현. M1: Showdown 배틀 환경과 뇌 없는 파일럿.`) append ` M2: 인코더·판독·학습 단위 시험 — no-go(시험 불성립). M3·M4는 보류.` After the line `**M1 결과: …**` insert a blank line and `**M2 결과: no-go — 시험 불성립 (2026-09-22). 학습 단위 시험을 세울 쌍이 선언된 기준에 못 미쳐 시험이 돌지 않았다. "학습 안 됨"이 아니다 — 스펙 부록 I.**`

- [ ] **Step 2: Ledger catch-up.** Insert these bullets, in this order, directly before `- **안 된 것(운영)**`:

```markdown
- **실제 M2 보정 — 오라클 상한**(2026-09-16, 스펙 G.10, D.6 (c) 판정 아님): 학습 대신 KC→MBON 가중치를 직접 깎아 "가르친 냄새에 최대로 특이적인 감소가 일어났다면
  판독이 기준(보상 d′ ≥ +2, 처벌 하락 d′ ≤ −2)에 닿는가"를 쟀다. 후보쌍 **8/34**만 시험 가능했고(1채널 2/19, 2채널 6/15), 시험 가능한 쌍은 NORMAL 기술이 끼는 쌍에
  몰렸다(7/8 — KC를 거의 구동하지 못하는 사구체가 만든 비대칭일 수 있다, 기전 미확인). F v3의 게이트 3쌍은 전부 시험 불가였다 — v3를 돌렸다면 학습 실패가 아니라
  구조 때문에 FAIL이 났을 것이다. 순진 d′ ≈ 0 ∧ 시험 가능 ∧ 바닥 가드 통과인 홀수 턴 게이트 후보는 0쌍. 설계쌍은 시험 가능(보상 18.69 / 처벌 −6.27).
  요약 `results/summary/m2_oracle.json`.
- **실제 M2 인코더 비교**(2026-09-17, 스펙 G.12·G.13, 짝수 턴): 인코더 네 개(E0 스펙 3.3, E1 KC 구동 기준 배정, E2·E3 기술 채널 복제)의 점수 = min(턴 내 기술 축,
  상대 타입 축의 시험 가능 비율)이 E0 0.222 · E1 0.143 · E2 0.143 · E3 0.190으로 모두 0.5 미만 → 사전 선언한 규칙대로 멈췄다. 처벌 판독(MBON13)이 늘 마지막 관문이고,
  강한 사구체를 두 축이 나눠 갖는 제로섬이라 어느 하나만 고쳐서는 0.5에 닿지 않는다. 요약 `results/summary/m2_encoders.json`.
- **안 된 것(엔진 전제 프로브, 스펙 G.14.8)**: 등급 APL(E.5)이 상대 타입 축 병목을 줄이는지 보려던 프로브는 스모크에서 설계 결함이 드러나 전체 실행 전에 중단했다.
  등급 APL은 보상 판독 MBON05를 직접 억제하고(APL의 MBON 표적 1위), 활성을 맞춘 스파이킹 팔에서는 처벌 판독 MBON13이 바닥이라, 대비가 편차 완화가 아니라
  판독 세포의 바닥 효과를 쟀을 것이다. 판정은 내지 않았다. 등급 APL을 그대로 바꿔 끼우는 것은 엔진 교정이 아니다 → M0d(스펙 부록 H).
- **우리가 정한 것(M0d, 스펙 H.1–H.3a)**: 조합 C0 현재 엔진(대조) · C1 포화형 등급 APL + MBON 재조정 · C3 C1 + KC 활성 항상성 역치. C2(C1 + ORN→PN STD)는
  문헌 상수(f 0.78, τ 893 ms)에서 ALPN 스파이크가 C1의 0.08–0.11배로 무너져 자격점이 없어 탈락했다(이 엔진·이 상수·이 탐색 범위에서). 모든 변경은 기본값 꺼짐이고,
  엔진 기본값은 M0c 엔진 그대로다.
- **실제 M0d 측정 — 작동점 고정**(2026-09-21, 스펙 H.3a.13): 세 조합 모두 채택. C1 `kc_thresh` 1.65 · `apl_input_scale` 0.137 · `mbon_hold_frac` 0.853
  (기준 KC 활성 5.95%, 게이트 32시드 기저 3.66 Hz), C3 같은 `kc_thresh` + 항상성 역치 · `apl_input_scale` 0.609 · `mbon_hold_frac` 0.849(5.54%, 3.08 Hz),
  세 조합 모두 폭주 0. 요약 `results/summary/m0d.json` 블록 `"h3"`.
- **실제 M0d 측정 — 조합 선택**(2026-09-22, 스펙 H.4a.5, 짝수 턴 39쌍): 세 조합 모두 판독 A = MBON13, P = MBON05. 오라클로 시험 가능한 상대 타입 축 쌍
  C3 **7/21**(F_a 2) · C0 4/21(F_a 1) · C1 2/21(F_a 0) → 자격은 C3 하나, T_b 0.333 < 0.5 → `STOP_LOW_T_B`. 요약 `results/summary/m0d.json` 블록 `"h4"`.
- **실제 M0d 진단과 특이성 천장**(2026-09-22, 스펙 H.4a.6–H.4a.8, 선택에 쓴 짝수 턴 데이터의 사후 분석): 학습 세기·시드 잡음·판독 가중은 원인이 아니고, 판독 바닥은
  C3가 대부분 고쳤다(바닥 통과 14쌍 중 시험 가능 7). 남은 한계는 편집된 시냅스의 X/Y 특이성 — 가르치지 않은 Y의 판독이 X의 54–80%만큼 같이 움직인다.
  편집을 X에만 활성인 KC로 제한한 천장(판독 규칙을 결과 전에 고정: C3 ≥ 14/21이면 새 선언, ≤ 10/21이면 no-go)은 특이성을 얻었으나(Y/X → 0) 처벌 조건이 무너져
  (−p ≥ 2인 쌍 C3 11 → 3 · 4) C3 `freq` 2/21 · `all` 3/21 → **no-go**. MBON13(처벌 판독)을 움직이는 X의 구동은 대부분 Y와 공유된 KC에 있다.
- **안 된 것(M2 no-go — 시험 불성립, 2026-09-22, 스펙 부록 I)**: 스펙 5의 M2 학습 단위 시험(순진 d′ ≈ 0 → 보상 20회 뒤 d′ ≥ 1 → 처벌 20회 뒤 하락)은
  **한 번도 돌지 않았다.** 시험한 인코더(E0–E3)와 엔진(C0·C1·C3)에서, 이상화한 특이적 가중치 감소로도 시험을 세울 수 있는 쌍이 선언된 기준(상대 타입 축 시험 가능
  비율 ≥ 0.5 ∧ 순진 균형 턴 내 쌍 ≥ 2)에 못 미쳤다. 그래서 이것은 "학습 안 됨"(D.6 (c) 실패)이 **아니라** 시험 불성립이다. 사용자 결정으로 스펙이 이름 붙인
  3번 경로(범위 축소 / M2 no-go 기록 + 주장 재설계)를 택했다.
  - 학습 시험은 설계쌍(기계 대조)으로 한정한다. 0절의 1차 주장(상대 타입 조건부 선호 학습)은 4.3 기준 1이 현재 형태로 **지지되지 않는다** — 시험하지 않았다.
    M3(에이전트 루프)·M4(실험)와 2차 주장(승률 기여)은 **보류**(폐기 아님).
  - D.6: (a) {G8_SENTENCE} · (b) 발동(E.2: 16턴 중 3턴이 비율 2 초과) · (c) 미측정. STD 재설계 여부는 새 주장 선언의 첫 질문으로 넘겼다.
  - 원인을 하나로 귀속하지 않는다: 판독이 두 타입뿐, 오라클은 이상화된 편집이고 천장은 현재 KC 부호의 상한(인코더·국소 APL·발화율 정규화 같은 KC 부호 변경은
    시험하지 않음), 짝수 턴 21쌍뿐, 채널→사구체 배정이 임의적(2867배), C0 대 C1·C3 기저 교락.
  - 판정은 `scripts/write_m2_nogo_summary.py`가 기록된 요약에서 파생한다(한 고리라도 성립하지 않으면 쓰기를 거부). 요약 `results/summary/m2_nogo.json`.
    M2 go 조건은 strict xfail 테스트 `tests/test_m2_go.py`로 남는다 — 조건이 충족되는 날 XPASS로 깨진다.
```

`{G8_SENTENCE}` is, for `G8_FIRED = false`: `미충족 — G.8 재판정(41냄새 × 64시드, 1 ms씩 움직이는 200 ms 창)에서 최대 {G8_MAX} 스파이크({G8_HZ} Hz), 150 Hz 초과 0건`; for `G8_FIRED = true`: `충족 — G.8 재판정(41냄새 × 64시드, 1 ms씩 움직이는 200 ms 창)에서 2624 제시 중 {G8_NOVER}건이 150 Hz 초과(최대 {G8_MAX} 스파이크)`.

- [ ] **Step 3: English.** Directly before `## 실행`, insert:

```markdown
## In English

FlyMon runs a leaky integrate-and-fire simulation of the whole MaleCNS v1.0 fruit-fly connectome as the attacking-move
chooser in a restricted task on Pokémon Showdown's Gen 1 OU rules (16 species, a constrained move pool), next to a
heuristic coach. The ledger above separates what was measured, what we chose, and what did not work; this is its summary.

- **M0 (engine, reproducing flybrain's measurements): partial pass.** Kenyon-cell sparsity, the MBON baseline and
  channel-specific odour depression pass; the pre-registered composite-index flip does not (graded n_flip 0/8).
- **M0c (KC→KC fast excitation removed): pass** on sparsity, baseline, runaway, equivalence and throughput; the
  conditioning criterion still fails, as predicted.
- **M1 (battle environment, brain-free pilot): pass** — MAX − RND win rate 0.319 ≥ 0.15.
- **M2 (encoder, readout, learning unit test): no-go — the test could not be built (2026-09-22).** Spec 5's learning
  unit test (naive d′ ≈ 0, d′ ≥ 1 after 20 rewards, a drop after 20 punishments) never ran. On every encoder (E0–E3)
  and engine (C0, C1, C3) we tried, too few candidate pairs reach the declared bar even under an idealised, perfectly
  targeted weight edit: at best 7 of 21 opponent-type pairs (bar: half), and 2–3 of 21 when the edit is confined to
  Kenyon cells active only for the taught odour, because the punishment readout (MBON13) is driven mostly by Kenyon
  cells the two odours share. This is **not** a "no learning" result.
  - Learning tests are limited to the designed odour pair (a mechanism control). The primary claim (type-conditional
    move preference, spec 4.3 criterion 1) is not supported in its current form and is reported as untested. M3 (agent
    loop), M4 (experiment) and the secondary claim (win-rate contribution) are on hold, not discarded. Whether to design
    short-term depression (ORN→PN) is the first question of the next claim declaration.
  - Runaway and ratio conditions (spec D.6): (a) {G8_ENGLISH}; (b) met — the within-turn candidate KC spike ratio exceeds
    2 on 3 of 16 turns; (c) the learning unit test itself: not measured.
  - We do not attribute the no-go to one cause: only two readout types, an idealised oracle whose ceiling bounds only the
    current Kenyon-cell code (encoder and coding changes such as local APL or rate normalisation were not tested), 21
    even-turn pairs, an arbitrary channel-to-glomerulus assignment, and a baseline confound between C0 and C1/C3.
  - Derived by `scripts/write_m2_nogo_summary.py` into `results/summary/m2_nogo.json`; the M2 go condition stays in the
    code as strict expected failures (`tests/test_m2_go.py`). Details: spec appendix I (in Korean).
```

`{G8_ENGLISH}` is, for `G8_FIRED = false`: `not met — no Kenyon cell exceeded 150 Hz in any 200 ms window sliding by 1 ms over 41 odours × 64 seeds (maximum {G8_MAX} spikes, {G8_HZ} Hz)`; for `G8_FIRED = true`: `met — {G8_NOVER} of 2,624 presentations had a Kenyon cell above 150 Hz in some 200 ms window (maximum {G8_MAX} spikes)`.

- [ ] **Step 4: Check and commit**

Run: `git diff --stat` → only `README.md`; `git diff README.md | grep '^-[^-]'` prints nothing. Every number in the new bullets must match the cited spec section or `results/summary/m2_nogo.json`.

```bash
git add README.md
git commit -m "docs(readme): ledger catch-up (G.10, G.12/13, G.14.8, H.1-H.4a.8) and the M2 no-go (시험 불성립) entry; English summary"
```

---

### Task 5: the handoff note for the claim redesign

**Files:**
- Create: `docs/handoffs/2026-09-22-m2-nogo-claim-redesign.md`

**Interfaces:**
- Consumes: appendix I (Task 3), `results/summary/m2_nogo.json`, the commits of Tasks 1–4 and R2 (dispatch prompt). Style reference: `docs/handoffs/2026-09-17-m2-calibration-handoff.md` (read it first; Korean, dense, sections with pointers).
- Produces: the starting point appendix I.7 names.

- [ ] **Step 1: Write the note** with exactly these sections, in Korean, every claim pointing at a spec section, a committed file or a commit:

1. **상태** — one paragraph: M2 no-go — 시험 불성립 recorded (commits of Tasks 1–4 and R2, `m2_nogo.json`), branch `open-fly-brain-connectome` = `origin/main`. Nothing is running. D.6: (a) as recorded (G.8), (b) fired, (c) not measured.
2. **새 주장 선언의 첫 질문: STD** (I.4) — the question as it stands: E.3 withdrew the "STD does not fix drive differences" claim and pointed at ORN→PN (E.3, E.5); D.6 (b) fired in E.2 {and (a) by G.8, if `G8_FIRED`}; with literature constants (f 0.78, τ 893 ms) ORN→PN depression collapses PN output and C2 found no qualifying point (H.3a.1 — "현재 엔진·이 문헌 상수·명시한 탐색 범위에서"). The declaration must decide: design STD (where, with which constants, judged how) or close it with a stated reason. Do not answer it in the note.
3. **새 선언이 물려받는 제약** — (i) H.4a.8: any lever must create X-only drive onto MBON13; the ceiling bounds only levers that keep the current KC code (plasticity specificity, readout at the edited synapses); (ii) untested KC-code levers named in I.2 — encoder or vocabulary aimed at the (b)-axis opponent channel, local APL, firing-rate normalisation (ORN→PN divisive normalisation), LHPV3c1 gain or sign, KC→KC inhibition, sparsening — each with its spec pointer (E.5, H.1's exclusion row, H.4a.6 "선택지에 대한 함의") and marked **미검증**; (iii) learning tests limited to the design pair; 4.3 criterion 1 untested; M3/M4 and the secondary claim on hold; (iv) the declaration habits to keep: fix reading rules before results (H.4a.7), exploration/confirmation separation (4.4), odd turns used only by the G.10 oracle so far, the M4 confirmation set and the M0d confirmation set (H.5) never used.
4. **재사용할 것** — `flymon/brain/d6a.py` + `scripts/run_m2_d6a.py` (G.8), `scripts/write_m2_nogo_summary.py`, `tests/test_m2_go.py` (the xfails a new go must turn into XPASS), `scripts/run_m0d_h4.py` and the `flymon/brain/h4_*` modules (readout reselection, oracle, selection), `docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py` (`--family`), `h4_testability_diag.py`; summaries `m2_oracle.json`, `m2_encoders.json`, `m0d.json` (`h3`, `h4`), `m2_nogo.json`, `m0d_h3_c3_thresholds.npz`.
5. **닫지 않은 것** — copy I.5's list as pointers (not re-argued).
6. **운영 규칙** — commits without trailers; push `gh auth switch --hostname github.com --user lyutvs` → `git push origin HEAD:main` → `gh auth switch --hostname github.com --user asher_wrtn`; long runs from the controller with `run_in_background` and no timeout argument; subagents never write `results/` or run real data; FlyPool scripts need a `__main__` guard; full suite `uv run pytest -q -rfE -o addopts=""` and its count at the end of this plan.
7. **로컬 전용 참고(git 제외)** — `.superpowers/drafts/2026-09-22-next-step-prep.md` §1 (Branch A candidates, ranked; background only, not a decision), `.superpowers/drafts/2026-09-22-h4-testability-diag.md`, `results/m0d/diag/h4_specificity_ceiling{,_all}{,_summary}.json`, `results/m2/d6a/g8.json`.

- [ ] **Step 2: Check and commit**

Run: `git diff --stat HEAD` → only the new file. Every commit hash in the note exists (`git cat-file -t <hash>` prints `commit`), every path exists or is marked git-excluded.

```bash
git add docs/handoffs/2026-09-22-m2-nogo-claim-redesign.md
git commit -m "docs(handoff): M2 no-go -> claim redesign starting point (STD first, H.4a.8 constraint, untested KC-code levers, what to reuse)"
```

---

### R3 (controller, not a subagent): verify, review, push

1. Full suite in the background: `uv run pytest -q -rfE -o addopts=""` → **538 passed, 1 skipped, 3 xfailed** (483 + 25 + 30 passed, the 2 new strict xfails). Read only the last lines.
2. Final review (sdd-reviewer) over `origin/main..HEAD`: spec compliance with this plan and appendix I, numbers in the spec, README and handoff against `m2_nogo.json` and the cited sections, no trailers (`git log origin/main..HEAD --format=%B | grep -iE 'co-authored|claude-session|generated with'` prints nothing).
3. Push: `gh auth switch --hostname github.com --user lyutvs && git push origin HEAD:main; gh auth switch --hostname github.com --user asher_wrtn`.
4. Ledger: final entry (commits, review verdict, suite count, D.6 (a) result). Remove the planning scratch worktree (`git worktree remove --force <scratchpad>/wt`).

## Self-review notes (planning session)

- All code in Tasks 1–2 ran in a scratch worktree at `0967a2c` before this plan was written: `tests/brain/test_d6a.py` 20 passed, `tests/test_run_m2_d6a.py` 5 passed, `tests/test_write_m2_nogo_summary.py` 29 passed + 1 skipped (no git-excluded inputs there), `tests/test_m2_go.py` 2 xfailed; the G.8 smoke ran (exit 0, 12 s); `d6a_job` reproduced E.1's read-window KC record on 48 presentations (turns 0, 6, 8, 13 × seeds 200–203); the runner's `e2_check` passed on real data; `derive()` on the real ceiling summaries read B with both provenance checks passing (G.8 stubbed).
- Spec coverage: reading 1 → appendix I head; G.8 → Task 1 + R1 + Task 3 step 1; the derivation rule (E head, S:626–628 precedent) → Task 2 + R2; A.5 precedent → `tests/test_m2_go.py`; G.13-6's E.3 termination → Task 3 step 2 and I.4; H.6 D.6 table → I.3/I.6; spec 7 "실패도 공개" → Task 4; the handoff → Task 5.
