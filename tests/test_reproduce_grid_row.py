"""The sparsity-grid writer keys each row by every engine setting the reader matches on (spec H.3a.2)."""
import importlib.util
from pathlib import Path

from flymon.brain.config import Params
from flymon.brain.pool_bench import SPARSITY_MATCH_KEYS, match_sparsity_row

_spec = importlib.util.spec_from_file_location("reproduce_flybrain_measurements",
                                               Path("scripts/reproduce_flybrain_measurements.py"))
rfm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfm)

MEAN = {k: 0.05 for k in SPARSITY_MATCH_KEYS}
REST = {"mbon_hz_rest_trimmed": 3.5}


def test_a_row_carries_apl_input_scale_and_is_found_only_for_its_own_engine():
    p = Params(apl_input_scale=0.5)
    row = rfm.grid_row(p, [100, 101, 102], MEAN, REST)
    assert row["apl_input_scale"] == 0.5 and row["kc_kc_scale"] == p.kc_kc_scale
    sparsity = dict(MEAN, per_seed=[{}])
    assert match_sparsity_row(sparsity, REST, {"grid": [row]}, p)["ok"]
    assert "no reference grid row" in match_sparsity_row(sparsity, REST, {"grid": [row]}, Params())["note"]


def test_the_row_key_fields_come_from_the_params():
    p = Params(kc_thresh=1.25, apl_scale=0.2, mbon_hold_frac=0.9, kc_kc_scale=1.0)
    row = rfm.grid_row(p, [100], MEAN, REST)
    assert (row["kc_thresh"], row["apl_scale"], row["mbon_hold_frac"], row["kc_kc_scale"], row["apl_input_scale"]) == \
        (1.25, 0.2, 0.9, 1.0, 1.0)
