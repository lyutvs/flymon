import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_m3_smoke", Path("scripts/run_m3_smoke.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_out_must_be_under_results_m3():
    with pytest.raises(SystemExit):
        m.parse_args(["--out", "results/m0d/x"])
    assert m.parse_args(["--out", "results/m3/smoke/x"]).flies == 2
