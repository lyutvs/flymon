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
    assert d["conditioning"]["n_flip"] == 8
