import json
from pathlib import Path

import pytest

from flymon.brain.config import Params

P = Path("results/summary/m0.json")


@pytest.mark.skipif(not P.exists(), reason="M0 summary not written yet")
def test_summary_matches_frozen_params():
    d = json.loads(P.read_text())
    frozen = d["params_frozen"]
    live = Params().__dict__
    for k in ("kc_thresh", "apl_scale", "learn_rate", "kc_trace_scale", "da_trace_scale", "mbon_hold_frac"):
        assert frozen[k] == live[k], f"{k}: summary {frozen[k]} != Params default {live[k]}"


@pytest.mark.skipif(not P.exists(), reason="M0 summary not written yet")
def test_summary_records_the_partial_gate_outcome():
    """M0 closed as a PARTIAL pass: sparsity and per-channel specificity hold, the pre-registered
    composite index flip does not. The summary must say which of the two it is."""
    gate = json.loads(P.read_text())["gate"]
    assert set(gate) == {"sparsity_ok", "conditioning_index_flip_ok", "conditioning_channel_specific_ok",
                         "channel_specific_seeds", "passed", "partial"}
    assert gate["sparsity_ok"] is True
    assert gate["passed"] or gate["partial"]


P_M0B = Path("results/summary/m0b.json")


@pytest.mark.skipif(not P_M0B.exists(), reason="M0b summary not written yet")
def test_m0b_summary_records_gate_and_budget():
    """M0b gate = budget <= 60 h AND exact reproduction of M0 (conditioning, sparsity) AND pool decide == in-process."""
    d = json.loads(P_M0B.read_text())
    assert set(d["gate"]) == {"throughput_ok", "conditioning_exact_ok", "sparsity_exact_ok", "decide_equal_ok", "passed"}
    assert d["budget"]["limit_hours"] == 60.0
    assert d["gate"]["throughput_ok"] == (d["budget"]["baseline_hours"] <= 60.0)
    assert d["gate"]["passed"] == all(d["gate"][k] for k in ("throughput_ok", "conditioning_exact_ok", "sparsity_exact_ok", "decide_equal_ok"))
    assert d["params_frozen"]["kc_thresh"] == Params().kc_thresh
    assert d["runaway"]["n_kc_over_sat"] >= 0.0
    assert {r["workers"] for r in d["throughput"]} >= {4, 8, 16}


P_M0C = Path("results/summary/m0c.json")


@pytest.mark.skipif(not P_M0C.exists(), reason="M0c summary not written yet")
def test_m0c_summary_records_the_pre_registered_gate():
    """M0c (spec D.4): PASS = sparsity and baseline and runaway and equivalence and throughput; the conditioning
    criterion is recorded alongside. The summary is the new engine (kc_kc_scale 0.0) and names the old one."""
    d = json.loads(P_M0C.read_text())
    g = d["gate"]
    assert set(g) == {"sparsity_ok", "baseline_ok", "runaway_ok", "equivalence_ok", "throughput_ok",
                      "conditioning_index_flip_ok", "channel_specific_seeds", "passed"}
    assert g["passed"] == all(g[k] for k in ("sparsity_ok", "baseline_ok", "runaway_ok", "equivalence_ok", "throughput_ok"))
    assert d["params_frozen"]["kc_kc_scale"] == Params().kc_kc_scale == 0.0 and d["kc_kc_scale_old_engine"] == 1.0
    assert d["sparsity"]["kc_kc_scale"] == 0.0
    assert d["seeds"]["conditioning_judged"] == list(range(8, 16)) and d["seeds"]["sparsity"] == [100, 101, 102]
    assert d["runaway"]["rest_seeds"] == list(range(100, 108)) and len(d["runaway"]["rest_n_kc_over_sat_per_seed"]) == 8
    assert d["runaway"]["odor_B_n_seeds"] == 64 and d["runaway"]["odor_B_sat_hz"] == 150.0
    assert d["equivalence"]["old_conditioning"]["n_results"] == 40
    # the terms the milestone exists for, predicted PASS in spec D.4: asserted, not merely recorded
    assert d["equivalence"]["old_conditioning"]["ok"] is True and d["gate"]["equivalence_ok"] is True
    assert d["gate"]["runaway_ok"] is True
    assert d["budget"]["limit_hours"] == 60.0 and {r["workers"] for r in d["throughput"]} >= {4, 8, 16}
    assert d["git_commit"] and set(d["inputs_sha256"]) >= {"sparsity", "reproduce_old", "reproduce", "throughput"}
