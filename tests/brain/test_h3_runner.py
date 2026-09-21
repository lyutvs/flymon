"""The H.3 candidate state machine for C0 and C1 (spec H.3a.4-H.3a.5), driven by a scripted measurer."""
import dataclasses
import math
import time

import pytest

from flymon.brain import h3_rules as R
from flymon.brain.config import Params
from flymon.brain.h3_runner import Context, c1_params, run_c0, run_c1
from flymon.brain.h3_spec import SPEC

from h3_scripted import N_KC, ODORS, POOLS, Scripted


def ctx(**kw):
    return Context(spec=kw.pop("spec", SPEC), odors=ODORS, pools=POOLS, n_kc=N_KC, log=lambda s: None, **kw)


def test_c1_takes_every_converged_cell_through_stages_2_and_3_and_adopts_the_top_rank():
    # the sparsity margin (pp) grows with kc_thresh and stays below the overlap margin (SD units): G(1.7) ranks first
    m = Scripted(pct=lambda p: (0.07 - 0.01 * (p.kc_thresh - 1.5), 0.045),
                 margin=lambda p, s: 0.006 - 0.01 * (p.kc_thresh - 1.55) + 0.0005 * math.sin(s))
    r = run_c1(m, ctx())
    assert r["status"] == R.COMBO_ADOPTED
    cells = {c["kc"]: c for c in r["cells"]}
    assert all("stage2" in c and "stage3" in c for c in cells.values())
    ranked = sorted(cells.values(), key=lambda c: c["rank"])
    assert [c["kc"] for c in ranked] == [1.7, 1.65, 1.6, 1.55]
    assert [c["status"] for c in ranked] == [R.ADOPTED, R.NOT_REACHED, R.NOT_REACHED, R.NOT_REACHED]
    assert ranked[0]["stage3"]["score"] == max(c["stage3"]["score"] for c in cells.values())
    assert sum(1 for c in cells.values() if "stage4" in c) == 1
    assert r["adopted"]["params"] == ranked[0]["final_params"]
    assert r["adopted"]["reference_median_kc_pct"] == ranked[0]["stage3"]["reference"]["median_kc_pct"] == 6.0
    assert r["adopted"]["params"].mbon_hold_frac == ranked[0]["stage2"]["accepted"]


def test_the_declared_seed_sets_are_the_only_ones_measured():
    m = Scripted()
    run_c1(m, ctx())
    assert m.seeds_asked("design") == {tuple(SPEC.design_seeds)}
    assert m.seeds_asked("baseline") == {tuple(SPEC.baseline_cal_seeds), tuple(SPEC.baseline_gate_seeds),
                                         tuple(SPEC.runaway_rest_seeds)}
    assert m.seeds_asked("odor_runaway") == {tuple(SPEC.runaway_odor_seeds)}
    assert m.seeds_asked("rest") == {tuple(s for o in ODORS for s in o["seeds"])}
    assert all(c[2] == "reference" and c[3] is None for c in m.calls if c[0] == "reference")


def test_stage1_runs_on_the_default_hold_and_stage3_on_the_final_params():
    m = Scripted()
    r = run_c1(m, ctx())
    cell = r["cells"][0]
    refs = [c[1] for c in m.calls if c[0] == "reference" and c[1].kc_thresh == cell["kc"]]
    assert refs[0].mbon_hold_frac == Params().mbon_hold_frac
    assert refs[-1] == cell["final_params"] and cell["final_params"].mbon_hold_frac != Params().mbon_hold_frac


def test_a_stage4_failure_moves_to_the_next_ranked_candidate():
    top = {}

    def hz(p, s):
        base = 28.0 * (p.mbon_hold_frac - 0.725) + 0.3 * math.sin(s)
        return base + (5.0 if s >= 116 and p.kc_thresh == top.get("kc") else 0.0)   # gate seeds only
    m = Scripted(pct=lambda p: (0.07 - 0.01 * (p.kc_thresh - 1.5), 0.045),
                 margin=lambda p, s: 0.006 - 0.01 * (p.kc_thresh - 1.55) + 0.0005 * math.sin(s), hz=hz)
    top["kc"] = 1.7                                   # the top-ranked cell under this script
    r = run_c1(m, ctx())
    by = {c["kc"]: c for c in r["cells"]}
    assert by[1.7]["rank"] == 1 and by[1.7]["status"] == R.GATE_FAILED
    assert not by[1.7]["stage4"]["baseline_ci"]["ok"] and by[1.7]["stage4"]["runaway"]["ok"]
    assert by[1.65]["rank"] == 2 and by[1.65]["status"] == R.ADOPTED and r["adopted"]["label"] == by[1.65]["label"]
    assert by[1.6]["status"] == R.NOT_REACHED and "stage4" not in by[1.6]


def test_overlap_holding_zero_takes_the_extra_seeds_then_passes_or_is_indeterminate():
    def margin_then_pass(p, s):
        return (0.004 if s % 2 else -0.004) if s < 108 else 0.01
    m = Scripted(margin=margin_then_pass)
    r = run_c1(m, ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    ov = r["cells"][0]["stage3"]["overlap"]
    assert ov["clause8"] == "holds_zero" and ov["clause"] == "pass" and ov["ci16"]["n"] == 16
    assert (tuple(SPEC.design_extra_seeds)) in m.seeds_asked("design")

    m = Scripted(margin=lambda p, s: 0.004 if s % 2 else -0.004)
    r = run_c1(m, ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    c = r["cells"][0]
    assert c["status"] == R.INDETERMINATE and c["stage3"]["reasons"] == ["overlap"]
    assert r["status"] == R.COMBO_DROPPED


def test_a_single_guard_type_whose_halves_split_is_indeterminate():
    def counts(p, name, j, s):
        if name == "MA1":
            return 0 if (j >= 24 and (2 * (j - 24) + s) < 20) or (j < 24 and 2 * j + s < 4) else 10
        return 0 if name == "MA2" else 10
    r = run_c1(Scripted(type_count=counts), ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    c = r["cells"][0]
    assert c["stage3"]["guard"]["A"]["passing"] == ["MA1"]
    assert c["status"] == R.INDETERMINATE and c["stage3"]["reasons"] == ["guard_A"]


def test_the_membrane_is_rechecked_on_the_final_params():
    def mv(p):   # on target at the default hold, 3 mV higher once the hold moves
        return 50.0 * p.apl_input_scale + (3.0 if p.mbon_hold_frac != 0.85 else 0.0)
    r = run_c1(Scripted(mv=mv), ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    c = r["cells"][0]
    assert c["search"]["converged"] and c["status"] == R.QUAL_FAILED and c["stage3"]["reasons"] == ["membrane"]


def test_a_cell_that_cannot_reach_the_membrane_target_is_a_search_failure_not_a_candidate():
    r = run_c1(Scripted(mv=lambda p: 30.0 + p.apl_input_scale), ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    c = r["cells"][0]
    assert c["status"] == R.SEARCH_FAILED and "stage2" not in c and r["status"] == R.COMBO_DROPPED


def test_sparsity_needs_a_positive_margin():
    r = run_c1(Scripted(pct=lambda p: (0.07, 0.045)), ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    assert r["cells"][0]["status"] == R.QUAL_FAILED and "sparsity" in r["cells"][0]["stage3"]["reasons"]


def test_runaway_fails_stage4():
    r = run_c1(Scripted(odor_kc_over=lambda p, s: 1 if s == 163 else 0),
               ctx(spec=dataclasses.replace(SPEC, kc_grid=(1.6,))))
    c = r["cells"][0]
    assert c["status"] == R.GATE_FAILED and not c["stage4"]["runaway"]["ok"]
    assert len(c["stage4"]["runaway"]["odor_detail"]) == len(SPEC.runaway_odor_seeds)
    assert c["stage4"]["runaway"]["odor_record_hz"] == SPEC.runaway_odor_record_hz


def test_a_passed_deadline_is_a_compute_abort_not_a_drop():
    r = run_c1(Scripted(), ctx(deadline=time.time() - 1))
    assert r["status"] == R.COMBO_ABORTED and "deadline" in r["note"]


def test_c0_is_judged_without_stage_1_2_or_the_membrane_and_records_its_baseline():
    m = Scripted(hz=lambda p, s: 4.8)        # far outside 3-4 Hz: C0's baseline is recorded, not judged
    r = run_c0(m, ctx())
    c = r["cells"][0]
    assert r["status"] == R.COMBO_ADOPTED and r["adopted"]["params"] == Params()
    assert c["stage3"]["membrane"] is None and "stage2" not in c and "search" not in c
    assert not c["stage4"]["baseline_ci"]["ok"] and c["stage4"]["baseline_judged"] is False
    assert tuple(SPEC.baseline_cal_seeds) not in m.seeds_asked("baseline")
    assert all(call[1] == Params() for call in m.calls)


def test_c0_fails_on_a_qualification():
    assert run_c0(Scripted(pct=lambda p: (0.08, 0.045)), ctx())["status"] == R.COMBO_DROPPED


def test_stop_after_stage1_measures_nothing_else():
    m = Scripted()
    r = run_c1(m, ctx(extra=dict(stop_after="stage1")))
    assert r["status"] == "stopped_after_stage1" and {c[0] for c in m.calls} == {"reference"}


def test_c1_params_are_graded_at_the_declared_cap():
    p = c1_params(SPEC, 1.6, 0.11863)
    assert (p.apl_mode, p.apl_r_max, p.kc_thresh, p.apl_input_scale) == ("graded", 0.333, 1.6, 0.11863)
