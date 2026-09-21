"""H.3a.8 records: every item is measured for an adopted candidate and none of them judges."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.config import Params
from flymon.brain.h3_records import clamped, collect, gain_control
from flymon.brain.h3_runner import Context, c1_params
from flymon.brain.h3_spec import SPEC

from h3_scripted import N_KC, ODORS, POOLS, Scripted

SPEC_FAST = dataclasses.replace(SPEC, boot_draws=200)


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def ctx(update=True):
    extra = dict(diag_dir="results/m0d/diag/h3/contrasts")
    if update:
        extra["update_mask"] = np.arange(N_KC) < 80
    return Context(spec=SPEC_FAST, odors=ODORS, pools=POOLS, n_kc=N_KC, log=lambda s: None, extra=extra)


def test_graded_records_cover_every_declared_item():
    m = Scripted()
    p = c1_params(SPEC, 1.6, 0.11863, mbon_hold_frac=0.84082)
    rec = collect(m, ctx(), p)
    assert set(rec) == {"membrane", "apl_input", "half_split", "guard_ci", "a_i", "gain_control", "all51",
                        "apl_to_mbon_zero", "kc_to_apl_only"}
    assert set(rec["membrane"]["instantaneous"]) == {"p5", "p25", "p50", "p75", "p95"}
    assert rec["membrane"]["release"]["rel_share_in"] == 1.0
    assert sum(rec["apl_input"]["share"].values()) == pytest.approx(1.0)
    assert set(rec["guard_ci"]) == {n for k in POOLS for n in POOLS[k]}
    assert rec["gain_control"]["release_match_ok"] and "iqr_diff" in rec["gain_control"]
    assert "log10_var_diff" in rec["all51"]
    edits = [(c[1], c[3]) for c in m.calls if c[0] == "reference" and c[3] is not None]
    assert (p, ("apl_to_mbon_zero",)) in edits
    assert (dataclasses.replace(p, apl_input_scale=1.0), ("kc_to_apl_scale", 0.11863)) in edits
    assert any(c[0] == "reference" and c[2] == "extended" for c in m.calls)
    for key in ("apl_to_mbon_zero", "kc_to_apl_only"):
        c = rec[key]
        assert set(c) == {"csc_edit", "csc_sha256", "path", "file_sha256"}               # numbers stay in diag
        assert c["path"].startswith("results/m0d/diag/h3/contrasts/")
        assert json.loads(Path(c["path"]).read_text())["guard"]["A"]["verdict"] in ("pass", "fail", "indeterminate")


def test_spiking_records_leave_out_the_graded_only_items():
    m = Scripted()
    rec = collect(m, ctx(update=False), Params())
    assert rec["gain_control"] is None and "kc_to_apl_only" not in rec and "a_i" not in rec
    assert "log10_var_b" not in rec["all51"] and not any(c[0] == "reference" and c[2] == "extended" for c in m.calls)


def test_the_clamped_engine_flattens_the_sigmoid_at_the_mean_release():
    p = c1_params(SPEC, 1.6, 0.11863)
    b = clamped(p, 0.5058, SPEC)
    assert b.apl_slope == 1e9 and b.apl_r_max == 2.0 * 0.5058 * 0.333
    assert dataclasses.replace(b, apl_slope=p.apl_slope, apl_r_max=p.apl_r_max) == p


def test_a_release_mismatch_withholds_the_comparison():
    class Off(Scripted):
        def reference(self, params, which="reference", csc_edit=None):
            rows = super().reference(params, which, csc_edit)
            if params.apl_slope > 1e6:
                for r in rows:
                    r["release_mean"] *= 1.05
            return rows
    gc = gain_control(Off(), ctx(), c1_params(SPEC, 1.6, 0.11863))
    assert not gc["release_match_ok"] and "iqr_diff" not in gc
