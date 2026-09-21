"""scripts/run_m0d_h3.py end to end on the synthetic connectome: every code path with real jobs on a FlyPool, the
resume, the guards on every written file, the stop rule, the exit codes and when the summary may be written
(spec H.3a.4, H.3a.11)."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import h3_store
from flymon.brain.h3_spec import SPEC, OdorSet, Window

_spec = importlib.util.spec_from_file_location("run_m0d_h3", Path("scripts/run_m0d_h3.py").resolve())
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

CLEAN = dict(commit="t", dirty_hashed=[], dirty_other=[])


def lenient_spec(npz):
    """Small windows and seed sets, and verdict thresholds wide enough that every stage runs on the synthetic brain;
    the connectome identity is the synthetic NPZ's."""
    r = dataclasses.replace
    return r(SPEC, connectome_sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest(), strength=3.0,
             reference=OdorSet("S{j:02d}", 4, 11, 1000, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=None),
             extended=OdorSet("T{j:02d}", 4, 15, 2000, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=None),
             reference_window=Window(50.0, 100.0), design_window=Window(50.0, 100.0), all51_window=Window(50.0, 100.0),
             runaway_odor_window=Window(50.0, 100.0), baseline_ms=200.0, design_k=2,
             design_seeds=(100, 101), design_extra_seeds=(102, 103), baseline_cal_seeds=(100, 101),
             baseline_gate_seeds=(116, 117, 118), runaway_rest_seeds=(100,), runaway_odor_seeds=(100, 101),
             all51_seeds=(200,), boot_draws=50, guard_half=2, kc_grid=(0.5,), scale_bisect_steps=1,
             hold_bisect_steps=1, membrane_tol_mv=1e9, d4_band=(0.0, 1.0), guard_med_delta_min=-1e9,
             guard_zero_share_max=1.0, baseline_band_hz=(-1e9, 1e9), runaway_rest_sat_hz=1e9,
             runaway_odor_sat_hz=1e9, homeo_median_tol=1.0, homeo_dtheta_max=1e9, homeo_max_iter=3,
             homeo_boundary_share=1.0, c3_max_cycles=1)


@pytest.fixture
def workdir(synthetic_npz, tmp_path, monkeypatch):
    """Working directory tmp_path, a clean tree, both engine guards recorded; returns a runner bound to them."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "git_state", lambda: dict(CLEAN))
    seen = []
    old, mod = h3_store.refuse_old_engine_output, h3_store.refuse_modified_engine_output
    monkeypatch.setattr(h3_store, "refuse_old_engine_output", lambda out, k: (seen.append(("old", out)), old(out, k)))
    monkeypatch.setattr(h3_store, "refuse_modified_engine_output", lambda out, p: (seen.append(("mod", out)), mod(out, p)))
    spec = lenient_spec(synthetic_npz)

    def main(*extra, spec_=spec, summary_spec=spec):
        return run.main(["--npz", str(synthetic_npz), "--workers", "2", *extra], spec=spec_, summary_spec=summary_spec,
                        require_root=False)
    return tmp_path, main, seen, spec


def _files(root):
    return sorted(str(p.relative_to(root)) for p in (root / "results").rglob("*") if p.is_file())


def test_a_complete_run_writes_everything_through_both_guards_and_resumes_from_the_cache(workdir):
    root, main, seen, _ = workdir
    assert main("--continue-after-drop") == 0
    files = _files(root)
    s = json.loads((root / "results/summary/m0d.json").read_text())["h3"]
    assert set(s["combos"]) == {"C0", "C1", "C3"}
    assert all(c["status"] in ("adopted", "dropped") for c in s["combos"].values())
    assert set(s["records"]) == {k for k, c in s["combos"].items() if c["status"] == "adopted"}
    assert ("c3_thresholds" in s) == (s["combos"]["C3"]["status"] == "adopted")
    assert s["combos"]["C3"]["homeo_target"] is not None
    for f in files:
        assert ("old", f) in seen and ("mod", f) in seen, f
    assert any("/thresholds/theta-" in f for f in files) and any(f.startswith("results/m0d/diag/h3/") for f in files)
    for k, rec in s["records"].items():
        assert set(rec["apl_to_mbon_zero"]) == {"csc_edit", "csc_sha256", "path", "file_sha256"}, k

    assert main("--continue-after-drop") == 0
    reports = sorted((root / "results/m0d/h3/runs").glob("*.json"))
    second = json.loads(reports[-1].read_text())
    assert second["cache"]["misses"] == 0 and second["cache"]["hits"] > 0
    first = json.loads(reports[0].read_text())
    assert {k: c["status"] for k, c in first["combos"].items()} == {k: c["status"] for k, c in second["combos"].items()}
    assert h3_store.canonical(first["combos"]) == h3_store.canonical(second["combos"])


def test_the_runner_stops_at_the_first_dropped_combination(workdir):
    root, main, _, _ = workdir
    assert main() == 0
    rep = json.loads(next((root / "results/m0d/h3/runs").glob("*.json")).read_text())
    first_drop = next(k for k in ("C0", "C1", "C3") if rep["combos"].get(k, {}).get("status") == "dropped")
    assert rep["stopped_after"] == first_drop and list(rep["combos"])[-1] == first_drop
    assert not (root / "results/summary/m0d.json").exists()


@pytest.mark.parametrize("blocker", ["no_records", "kc_cells", "stop_after", "combos", "smoke", "dirty", "connectome",
                                     "spec"])
def test_each_blocker_alone_keeps_the_summary_unwritten(workdir, monkeypatch, blocker):
    root, main, _, spec = workdir
    extra = {"no_records": ["--no-records"], "kc_cells": ["--kc-cells", "0.5"], "stop_after": ["--stop-after", "stage1"],
             "combos": ["--combos", "C0,C1"], "smoke": ["--smoke", "--out", "results/m0d/h3-test"],
             "dirty": ["--allow-dirty"]}.get(blocker, [])
    if blocker == "dirty":
        monkeypatch.setattr(run, "git_state", lambda: dict(CLEAN, dirty_hashed=["flymon/brain/h3_rules.py"]))
    kw = {}
    if blocker == "connectome":
        kw = dict(spec_=dataclasses.replace(spec, connectome_sha256="0" * 64),
                  summary_spec=dataclasses.replace(spec, connectome_sha256="0" * 64))
    if blocker == "spec":
        kw = dict(summary_spec=dataclasses.replace(spec, boot_draws=51))
    assert main("--continue-after-drop", *extra, **kw) == 0
    assert not (root / "results/summary/m0d.json").exists()


def test_stop_after_stage1_ends_the_run_after_c1(workdir):
    root, main, _, _ = workdir
    assert main("--continue-after-drop", "--stop-after", "stage1") == 0
    rep = json.loads(next((root / "results/m0d/h3/runs").glob("*.json")).read_text())
    assert rep["combos"]["C1"]["status"] == "stopped_after_stage1" and "C3" not in rep["combos"]
    assert rep["stopped_after"] is None
    assert not (root / "results/summary/m0d.json").exists()


def test_a_dirty_hashed_file_refuses_the_run(workdir, monkeypatch):
    root, main, _, _ = workdir
    monkeypatch.setattr(run, "git_state", lambda: dict(CLEAN, dirty_hashed=["flymon/brain/h3_rules.py"]))
    assert main() == 2
    assert not (root / "results").exists()


def test_a_passed_deadline_exits_3_without_a_summary(workdir):
    root, main, _, _ = workdir
    assert main("--continue-after-drop", "--max-hours", "0") == 3
    rep = json.loads(next((root / "results/m0d/h3/runs").glob("*.json")).read_text())
    assert rep["combos"]["C0"]["status"] == "compute_aborted"
    assert not (root / "results/summary/m0d.json").exists()


def test_the_runner_refuses_to_run_outside_the_repository_root(synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run.main(["--npz", str(synthetic_npz), "--allow-dirty"], spec=lenient_spec(synthetic_npz)) == 2
    assert not (tmp_path / "results").exists()


def test_the_smoke_spec_shrinks_every_sample_but_keeps_the_generators():
    s = run.smoke_spec(SPEC)
    assert s.reference.rng_seed == SPEC.reference.rng_seed and s.reference.n == 4
    assert len(s.design_seeds) < len(SPEC.design_seeds) and len(s.kc_grid) == 1


def test_every_file_the_resume_key_hashes_exists():
    for f in h3_store.HASHED_FILES:
        assert (h3_store.ROOT / f).exists(), f
