"""C3: homeostatic thresholds, cycles and the inherited-then-own-grid order (spec H.3a.6)."""
import dataclasses
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import h3_rules as R
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.h3_c3 import ThresholdFiles, c3_cell, homeostasis, rule_thresholds, run_c3, run_cycles, update_mask
from flymon.brain.h3_runner import Context, c1_params, run_c1
from flymon.brain.h3_spec import SPEC
from flymon.brain.h3_store import sha256_file

from h3_scripted import N_KC, ODORS, POOLS, Scripted

RULE = np.full(N_KC, 10.0, np.float32)
HAS_PN = np.arange(N_KC) < 90
UPDATE = np.arange(N_KC) < 80
IDS = np.arange(5000, 5000 + N_KC, dtype=np.int64)


def theta_of(p):
    if p.kc_thresh_mode != "homeostatic":
        return RULE
    with np.load(p.kc_thresh_file) as d:
        assert np.array_equal(d["kc_body_ids"], IDS)
        return d["v_th"]


def a_model(p, gain=0.12):
    """a_i = 0.12 (rule / theta)^4 on the update set: 0.12 at the rule, A0 = 0.062 near theta = 1.18 rule."""
    th = theta_of(p).astype(np.float64)
    return np.where(UPDATE, np.minimum(1.0, gain * (RULE / th) ** 4), 0.0)


def fired_model(gain=0.12):
    def fired(p, j, s):
        n = np.round(a_model(p, gain) * 96).astype(int)
        return [int(i) for i in np.flatnonzero(n > 2 * j + s)]
    return fired


def mv_model(p):
    """Membrane falls as the thresholds rise (fewer KC spikes into APL), rises with the scale."""
    return 50.0 * p.apl_input_scale * float(RULE[UPDATE].mean() / theta_of(p)[UPDATE].mean())


@pytest.fixture
def c3ctx(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = ThresholdFiles("results/m0d/h3/thresholds", IDS)
    return Context(spec=SPEC, odors=ODORS, pools=POOLS, n_kc=N_KC, log=lambda s: None,
                   extra=dict(update_mask=UPDATE, rule_thresholds=lambda kc: (RULE, HAS_PN), threshold_files=files))


# ---- the cycle loop (pure) ------------------------------------------------------------------------------------------
def _search(ok=True):
    return lambda theta: dict(converged=ok, accepted=dict(x=0.2))


def _homeo(ends, converged=True):
    it = iter(ends)

    def h(s, theta):
        mv, kc = next(it)
        return dict(converged=converged, theta=theta, params=f"P{mv}", trace=[],
                    stats=dict(median_mv=mv, median_kc_pct=kc))
    return h


def test_run_cycles_stops_at_the_first_cycle_inside_the_band():
    r = run_cycles(SPEC, _search(), _homeo([(13.0, 6.0), (11.5, 6.2)]), RULE, lambda th: 0.0)
    assert r["status"] is None and r["params"] == "P11.5" and len(r["cycles"]) == 2


def test_run_cycles_stalled_exhausted_search_failed_unconverged_boundary():
    assert run_cycles(SPEC, _search(), _homeo([(13.0, 6.0), (13.1, 6.1)]), RULE, lambda th: 0)["status"] == R.STALLED
    assert run_cycles(SPEC, _search(), _homeo([(13.0, 6.0), (13.5, 6.0), (14.0, 6.0)]), RULE,
                      lambda th: 0)["status"] == R.CYCLES_EXHAUSTED
    assert run_cycles(SPEC, _search(False), _homeo([]), RULE, lambda th: 0)["status"] == R.SEARCH_FAILED
    assert run_cycles(SPEC, _search(), _homeo([(11.0, 6.0)], converged=False), RULE,
                      lambda th: 0)["status"] == R.HOMEOSTASIS_UNCONVERGED
    r = run_cycles(SPEC, _search(), _homeo([(11.0, 6.0)]), RULE, lambda th: 0.11)
    assert r["status"] == R.BOUNDARY_LIMITED and r["boundary_share"] == 0.11


def test_run_cycles_needs_both_changes_small_to_call_a_stall():
    r = run_cycles(SPEC, _search(), _homeo([(13.0, 6.0), (13.1, 6.5), (13.05, 6.6)]), RULE, lambda th: 0)
    assert r["status"] == R.STALLED and len(r["cycles"]) == 3


# ---- homeostasis with real threshold files ----------------------------------------------------------------------
def test_homeostasis_converges_and_writes_content_addressed_files(c3ctx):
    m = Scripted(fired=fired_model(), mv=mv_model)
    base = c1_params(SPEC, 1.6, 0.22)
    h = homeostasis(m, c3ctx, base, RULE, RULE, UPDATE, c3ctx.extra["threshold_files"])
    assert h["converged"] and h["trace"][-1]["median_ok"] and h["trace"][-1]["dtheta_ok"]
    assert h["trace"][0]["dtheta_q"] is None and m.calls[0][1] == base          # iteration 0 is the rule, pn_norm
    p = h["params"]
    assert p.kc_thresh_mode == "homeostatic" and p.kc_thresh_sha256 == sha256_file(p.kc_thresh_file)
    assert np.array_equal(theta_of(p), h["theta"]) and (h["theta"][~UPDATE] == RULE[~UPDATE]).all()
    again = c3ctx.extra["threshold_files"].params(base, h["theta"])
    assert again == p


def test_homeostasis_that_cannot_reach_the_target_is_unconverged(c3ctx):
    spec = dataclasses.replace(SPEC, homeo_max_iter=5)
    m = Scripted(fired=lambda p, j, s: list(range(80)) if 2 * j + s < 20 else [], mv=mv_model)   # a = 0.21 always
    h = homeostasis(m, dataclasses.replace(c3ctx, spec=spec), c1_params(SPEC, 1.6, 0.22), RULE, RULE, UPDATE,
                    c3ctx.extra["threshold_files"])
    assert not h["converged"] and len(h["trace"]) == 5
    measured = [c[1] for c in m.calls if c[0] == "reference"]
    assert h["params"] == measured[-1] and np.array_equal(theta_of(h["params"]), h["theta"])
    files = sorted(Path("results/m0d/h3/thresholds").glob("theta-*.npz"))
    assert len(files) == 4                      # iterations 1-4; no file for an update that was never measured
    assert h["trace"][2]["n_moving"] is not None and h["trace"][2]["n_oscillating"] is not None


# ---- a cell and the combination ------------------------------------------------------------------------------------
def test_c3_cell_starts_from_c1_s_stage1_and_re_bisects_after_the_thresholds_move(c3ctx):
    m = Scripted(fired=fired_model(), mv=mv_model)
    cell = c3_cell(m, c3ctx, 1.6, 0)
    assert "params" in cell and len(cell["cycles"]) == 2
    first = [c[1] for c in m.calls if c[0] == "reference"][:10]
    c1 = Scripted(fired=fired_model(), mv=mv_model)
    run_c1(c1, dataclasses.replace(c3ctx, spec=dataclasses.replace(SPEC, kc_grid=(1.6,)), extra=dict(stop_after="stage1")))
    assert first == [c[1] for c in c1.calls if c[0] == "reference"]            # cycle 1 = C1's stage 1, same Params
    assert cell["params"].kc_thresh_mode == "homeostatic"
    assert cell["cycles"][1]["search"]["accepted"]["x"] > cell["cycles"][0]["search"]["accepted"]["x"]   # higher thresholds, less KC drive


def test_c3_cell_refuses_an_update_set_outside_the_pn_input_kcs(c3ctx):
    bad = dataclasses.replace(c3ctx, extra=dict(c3ctx.extra, update_mask=np.arange(N_KC) < 95))
    with pytest.raises(ValueError, match="update set"):
        c3_cell(Scripted(), bad, 1.6, 0)


def _c1_adopted(kc, m_factory, ctx):
    c1 = run_c1(m_factory(), dataclasses.replace(ctx, spec=dataclasses.replace(SPEC, kc_grid=(kc,))))
    assert c1["status"] == R.COMBO_ADOPTED
    return c1


def test_run_c3_tries_the_inherited_point_first_and_stops_when_it_is_adopted(c3ctx):
    make = lambda: Scripted(fired=fired_model(), mv=mv_model)
    c1 = _c1_adopted(1.65, make, c3ctx)
    r = run_c3(make(), c3ctx, c1)
    assert r["status"] == R.COMBO_ADOPTED and [c["kc"] for c in r["cells"]] == [1.65]
    assert r["cells"][0]["inherited_from_c1"] and r["cells"][0]["inherited_scale_equal"]


def test_run_c3_runs_its_own_grid_when_the_inherited_point_fails(c3ctx):
    make = lambda: Scripted(fired=fired_model(), mv=mv_model)
    c1 = _c1_adopted(1.65, make, c3ctx)
    fails_at_165 = Scripted(fired=fired_model(), mv=mv_model,
                            pct=lambda p: (0.08, 0.045) if p.kc_thresh == 1.65 else (0.058, 0.045))
    r = run_c3(fails_at_165, c3ctx, c1)
    assert [c["kc"] for c in r["cells"]] == [1.65, 1.55, 1.6, 1.7]
    assert r["cells"][0]["status"] == R.QUAL_FAILED and r["status"] == R.COMBO_ADOPTED


def test_a0_follows_c1_s_adopted_point_and_falls_back_to_the_declared_value(c3ctx):
    make = lambda: Scripted(fired=fired_model(), mv=mv_model, kc=lambda p: 0.0596)
    c1 = _c1_adopted(1.6, make, c3ctx)
    assert c1["adopted"]["reference_median_kc_pct"] == pytest.approx(5.96)
    r = run_c3(make(), c3ctx, c1)
    assert r["homeo_target"] == 0.06 and "C1 adopted" in r["homeo_target_source"]
    dropped = run_c3(make(), dataclasses.replace(c3ctx, spec=dataclasses.replace(SPEC, kc_grid=(1.6,))),
                     dict(status=R.COMBO_DROPPED))
    assert dropped["homeo_target"] == SPEC.homeo_target == 0.062


def test_an_abort_keeps_the_cells_already_finished(c3ctx):
    from flymon.brain.h3_runner import ComputeAborted

    class Stops(Scripted):
        def reference(self, params, which="reference", csc_edit=None):
            if params.kc_thresh == 1.6:
                raise ComputeAborted("budget")
            return super().reference(params, which, csc_edit)
    r = run_c3(Stops(fired=fired_model(), mv=mv_model), c3ctx, dict(status=R.COMBO_DROPPED))
    assert r["status"] == R.COMBO_ABORTED and [c["kc"] for c in r["cells"]] == [1.55]


def test_a_tampered_threshold_file_is_rewritten(c3ctx):
    files = c3ctx.extra["threshold_files"]
    theta = RULE * np.float32(1.1)
    p = files.params(c1_params(SPEC, 1.6, 0.2), theta)
    np.savez(p.kc_thresh_file, kc_body_ids=IDS, v_th=RULE)                  # same name, other content
    again = files.params(c1_params(SPEC, 1.6, 0.2), theta)
    assert np.array_equal(theta_of(again), theta) and again.kc_thresh_sha256 == sha256_file(again.kc_thresh_file)


def test_threshold_files_are_byte_identical_whenever_they_are_written(c3ctx):
    files = c3ctx.extra["threshold_files"]
    theta = RULE * np.float32(1.1)
    p = files.params(c1_params(SPEC, 1.6, 0.2), theta)
    with zipfile.ZipFile(p.kc_thresh_file) as z:
        assert [i.filename for i in z.infolist()] == ["kc_body_ids.npy", "v_th.npy"]
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in z.infolist())
    first = Path(p.kc_thresh_file).read_bytes()
    Path(p.kc_thresh_file).unlink()
    again = files.params(c1_params(SPEC, 1.6, 0.2), theta)
    assert again.kc_thresh_sha256 == p.kc_thresh_sha256 and Path(again.kc_thresh_file).read_bytes() == first
    assert np.array_equal(theta_of(again), theta)


def test_run_c3_runs_every_cell_when_c1_was_dropped(c3ctx):
    r = run_c3(Scripted(fired=fired_model(), mv=mv_model), c3ctx, dict(status=R.COMBO_DROPPED))
    assert [c["kc"] for c in r["cells"]] == list(SPEC.kc_grid) and not any(c.get("inherited_from_c1") for c in r["cells"])


# ---- the update set and the rule on connectomes ---------------------------------------------------------------------
def test_update_mask_drops_kcs_whose_pn_synapses_are_all_below_min_weight(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    k0 = int(pops.kc[0])
    w = c.w.copy()
    w[(np.isin(c.pre, pops.alpn)) & (c.post == k0)] = 3
    c2 = dataclasses.replace(c, w=w)
    m = update_mask(c2, pops, Params())
    assert not m[0] and m[1:].all()
    rule, has_pn = rule_thresholds(c2, pops, Params(kc_thresh=1.6))
    assert has_pn[0] and np.array_equal(rule, Engine(c2, pops, Params(kc_thresh=1.6)).v_th[pops.kc])


def test_a_threshold_file_of_the_rule_loads_into_the_engine(synthetic_connectome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    base = Params(kc_thresh=1.6)
    rule, _ = rule_thresholds(c, pops, base)
    p = ThresholdFiles("results/m0d/h3/thresholds", c.bodyId[pops.kc]).params(base, rule * np.float32(1.0))
    e = Engine(c, pops, p)
    assert np.array_equal(e.v_th[pops.kc], rule)


@pytest.mark.skipif(not Path("data/malecns.npz").exists(), reason="MaleCNS connectome not built")
def test_the_real_update_set_is_the_declared_3768_kcs():
    c = Connectome.load("data/malecns.npz")
    pops = Populations.from_connectome(c)
    m = update_mask(c, pops, Params())
    _, has_pn = rule_thresholds(c, pops, Params(kc_thresh=1.6))
    assert int(m.sum()) == 3768 and int(has_pn.sum()) == 3812 and not (m & ~has_pn).any()
    ids = np.asarray(c.bodyId[pops.kc][m], np.int64)
    assert hashlib.sha256(ids.tobytes()).hexdigest() == \
        "f069a2e55aee3314d5d3599fbdfe63ff94f0566a60dff8d0a504ccbdf4a45f32"
