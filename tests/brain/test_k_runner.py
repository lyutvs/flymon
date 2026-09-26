"""Spec K.8.2 / K.8.4: stage 1 re-converges every g, measures only ADOPTED engines, ranks and gates; stage 2 judges one."""
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.brain import k_runner as K
from flymon.brain.config import Params
from flymon.brain.h3_rules import COMBO_ABORTED, COMBO_ADOPTED, COMBO_DROPPED
from flymon.brain.j_rules import B, COMPUTE_ABORTED
from flymon.brain.k_params import with_kc
from flymon.brain import d6a
from flymon.brain.j_rules import SELECTED
from flymon.brain.k_rules import SCAN_GO, STOP_C3_NO_DRIVE, STOP_NO_QUALIFIED_SETTING, STOP_NO_TARGET_GAIN
from flymon.brain.k_spec import SPEC

ARR = K.KArrays(w13=np.ones(4), w05=np.ones(4), lobes={"apbp": np.array([1, 1, 0, 0], bool)},
                kk=(np.array([0]), np.array([1]), np.array([10.0])), probe=np.array([0, 1]), mv=0.275)
JCTX = SimpleNamespace(log=lambda s: None, h3=SimpleNamespace(spec=SPEC.j.h4.h3, deadline=None))


class FakeKM:
    """One pair whose X-only mass on KC 0 is n_of_g(g) (default 1 + 10|g|; w13 = 1, so N = n_of_g(g)). The value is a
    stand-in, not a fraction: the runner only sums it."""
    def __init__(self, n_of_g=None):
        self.params_seen, self.odours, self.index = [], [{"g1": 1.0}], [(0, 0)]
        self.n_of_g = n_of_g or (lambda g: 1.0 + 10 * abs(g))
        self.rasters = []

    def activity(self, params):
        n = self.n_of_g(params.kc_kc_scale)
        return [dict(fx=np.array([n, 0.0, 0.0, 0.0]), fy=np.zeros(4), cx=np.array([2.0, 0.0, 0.0, 0.0]))]

    def raster(self, params, odor, seed, probe):
        self.rasters.append(params.kc_kc_scale)
        return dict(spikes=[], v_mean=[])


def fake_reconverge(adopt):
    def rc(m3, ctx, make):
        p = make(1.65)
        g = p.kc_kc_scale
        if adopt(g) == "abort":
            return dict(status=COMBO_ABORTED, cells=[], adopted=None, guard=None, note="budget")
        if adopt(g):
            return dict(status=COMBO_ADOPTED, cells=[dict(label="G", boundary_share=0.01)],
                        adopted=dict(label="G", params=p, reference_median_kc_pct=6.0), guard={"MBON13": {}})
        return dict(status=COMBO_DROPPED, cells=[], adopted=None, guard=None)
    return rc


def test_stage1_ranks_adopted_engines_by_n_and_goes(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: g != -0.4))
    km = FakeKM()
    r = K.stage1(object(), km, JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == SCAN_GO
    gs = [s["g"] for s in r["settings"]]
    assert gs == list(SPEC.grid) and r["settings"][3]["metrics"] is None
    assert [r["settings"][i]["g"] for i in r["order"]] == [-0.8, -0.2, -0.1, -0.05]
    assert r["ratios"][r["order"][0]] == pytest.approx(9.0)
    assert km.rasters == [-0.8] and "kc_kc_input" in r["settings"][4]["metrics"]
    assert r["base"]["metrics"]["all51_kc_on_log10_var"] is None       # no JMeasurer given


def test_stage1_without_gain_stops_before_any_even_pair(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: True))
    r = K.stage1(object(), FakeKM(lambda g: 1.0 - abs(g)), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == STOP_NO_TARGET_GAIN


def test_stage1_without_an_adopted_engine(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: False))
    r = K.stage1(object(), FakeKM(), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == STOP_NO_QUALIFIED_SETTING and r["order"] == []


def test_an_abort_is_compute_aborted_not_invalid(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: "abort" if g == -0.2 else True))
    r = K.stage1(object(), FakeKM(), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == COMPUTE_ABORTED and [s["g"] for s in r["settings"]] == [-0.05, -0.1, -0.2]


def test_stage2_judges_the_setting_and_names_the_state(monkeypatch):
    p = with_kc(Params(apl_mode="graded"), -0.2)
    seen = {}
    def judge(m4, ctx, name, params, guard):
        seen.update(name=name, params=params, guard=guard)
        return dict(outcome=B, reading=None, reason="no reactive and teachable readout in a pool")
    monkeypatch.setattr(K, "judge", judge)
    monkeypatch.setattr(K, "measure_d6", lambda jm, ctx, params, seeds, judged: {"a": 1})
    setting = dict(g=-0.2, reconverge=dict(adopted=dict(params=K.params_json(p)), guard={"MBON13": {"x": 1}}))
    r = K.stage2(object(), object(), JCTX, setting, with_d6=True, d6_seeds=(1, 2))
    assert seen["params"] == p and seen["guard"] == {"MBON13": {"x": 1}} and "g=-0.2" in seen["name"]
    assert r["state"] == "dropped_no_readout" and r["d6"] == {"a": 1}


def test_stage1_records_all51_kc_on_variance_with_js_count_floor(monkeypatch):
    monkeypatch.setattr(K, "reconverge", fake_reconverge(lambda g: g == -0.8))
    floors = []
    def sm(rows, floor):
        floors.append(floor)
        return {"log10_var": {"kc_on": 0.2 + rows[0]}}
    monkeypatch.setattr(K, "scan_metrics", sm)
    jm = SimpleNamespace(all51_uni=lambda p: [p.kc_kc_scale])
    r = K.stage1(object(), FakeKM(), JCTX, SPEC, Params(), ARR, jm=jm)
    assert r["base"]["metrics"]["all51_kc_on_log10_var"] == pytest.approx(0.2)
    assert r["settings"][4]["metrics"]["all51_kc_on_log10_var"] == pytest.approx(0.2 - 0.8)
    assert floors == [SPEC.j.count_floor] * 2


def test_a_c3_without_drive_stops_before_any_reconvergence(monkeypatch):
    calls = []
    monkeypatch.setattr(K, "reconverge", lambda *a: calls.append(a))
    r = K.stage1(object(), FakeKM(lambda g: 0.0 if g == 0 else 5.0), JCTX, SPEC, Params(), ARR)
    assert r["outcome"] == STOP_C3_NO_DRIVE == "stop_c3_no_drive" and calls == []
    assert r["settings"] == [] and r["order"] == [] and r["ratios"] == {} and r["base"]["metrics"]["N"] == 0.0


@pytest.mark.parametrize("seeds,judged", [(d6a.SEEDS[:2], False), (d6a.SEEDS, True), (None, True)])
def test_stage2_judges_d6a_only_on_the_full_seed_block(monkeypatch, seeds, judged):
    p = with_kc(Params(apl_mode="graded"), -0.2)
    monkeypatch.setattr(K, "judge", lambda m4, ctx, name, params, guard: dict(outcome=SELECTED,
                                                                             reading=dict(band=SELECTED)))
    got = {}
    monkeypatch.setattr(K, "measure_d6", lambda jm, ctx, params, s, judged: got.update(seeds=tuple(s), judged=judged))
    setting = dict(g=-0.2, reconverge=dict(adopted=dict(params=K.params_json(p)), guard={}))
    kw = {} if seeds is None else dict(d6_seeds=seeds)
    r = K.stage2(object(), object(), JCTX, setting, with_d6=True, **kw)
    assert got == dict(seeds=tuple(d6a.SEEDS if seeds is None else seeds), judged=judged)
    assert r["state"] == SELECTED
