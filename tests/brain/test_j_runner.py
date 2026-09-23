"""Spec J.11.3-J.11.4 procedures on scripted measurers: every branch of the scan and the judgement."""
import dataclasses

import pytest

from flymon.brain import h3_rules as H3R
from flymon.brain import j_runner as J
from flymon.brain import j_rules as R
from flymon.brain.config import Params
from flymon.brain.h3_c3 import ThresholdFiles
from flymon.brain.h3_runner import Context as H3Context
from flymon.brain.h3_spec import SPEC as H3_SPEC
from flymon.brain.j_params import StdParams
from flymon.brain.j_spec import SPEC

import h4_scripted as S4
from h3_scripted import N_KC, ODORS, POOLS, Scripted as Scripted3
from test_h3_c3 import HAS_PN, IDS, RULE, UPDATE, fired_model, mv_model

GLOMS = ["G0", "G1", "G2", "G3", "G4"]
KEEP = {(0.78, 893.0): 0.01, (0.78, 100.0): 0.1, (0.9, 893.0): 0.2, (0.9, 100.0): 0.4, (0.95, 893.0): 0.5,
        (0.95, 100.0): 0.9}                          # transmission the depression leaves (0.01: not restorable in 64x)
SPREAD = {0.78: 0.5, 0.9: 0.7, 0.95: 0.9}           # log10 kc_on step between glomeruli (no depression: 1.0)


class ScriptedJ:
    """all51 rows: ALPN = 100 (i + 1) x kept transmission x receptor_scale; kc_on = 10^(spread i)."""
    def __init__(self):
        self.calls = []

    def all51_uni(self, p):
        self.calls.append(("all51", p))
        std = getattr(p, "orn_std", False)
        gain = (KEEP[(p.orn_std_f, p.orn_std_tau_ms)] if std else 1.0) * getattr(p, "receptor_scale", 1.0)
        c = SPREAD[p.orn_std_f] if std else 1.0
        return [dict(g=g, seed=s, kc=10 * 10 ** (c * i), kc_on=10 ** (c * i), pn=100 * (i + 1) * gain, uni_n=1,
                     uni_hz=10.0 * (i + 1), orn_hz=60.0, std_r=(0.5 if std else None))
                for i, g in enumerate(GLOMS) for s in (200, 201)]

    def d6(self, p, by_turn, seeds):
        self.calls.append(("d6", p))
        return [dict(turn=t, seed=s, max_win=[1] * len(o), kc_active_frac=[0.05] * len(o), kc_spikes=[10] * len(o))
                for t, o in sorted(by_turn.items()) for s in seeds]


class ScriptedRef:
    """The reference set's KC median: 6% everywhere, 12% at (0.95, 100 ms) — outside the band."""
    def reference(self, p):
        frac = 0.12 if getattr(p, "orn_std", False) and (p.orn_std_f, p.orn_std_tau_ms) == (0.95, 100.0) else 0.06
        return [dict(apl_v_mean=11.0, kc_active_frac=frac)]


def jctx(tmp_path=None, deadline=None, spec=None):
    files = ThresholdFiles("results/m0d/j/std/thresholds", IDS)
    h3 = H3Context(spec=dataclasses.replace(H3_SPEC, homeo_target=SPEC.homeo_target), odors=ODORS, pools=POOLS,
                   n_kc=N_KC, deadline=deadline, log=lambda s: None,
                   extra=dict(update_mask=UPDATE, rule_thresholds=lambda kc: (RULE, HAS_PN), threshold_files=files))
    return J.JContext(spec=spec or dataclasses.replace(SPEC, std_tau_ms=(893.0, 100.0)), h3=h3, pools=S4.POOLS,
                      probe_seeds=S4.SEEDS, expected=list(S4.PAIRS), odours=[dict(turn=0, move="m", odor={"X": 1.0})],
                      log=lambda s: None)


# ---- stage 1 ----------------------------------------------------------------------------------------------------
def test_stage1_restores_the_gain_and_orders_the_usable_settings():
    out = J.stage1(ScriptedJ(), ScriptedRef(), jctx(), Params())
    st = out["settings"]
    assert out["outcome"] == R.SCAN_COMPLETE and out["target_alpn"] == 300.0 and out["base"]["kc_pct"] == 6.0
    assert [(s["f"], s["tau_ms"]) for s in st] == list(KEEP)
    assert not st[0]["restorable"] and st[0]["scale"] is None                       # 0.01 needs 100x > 64x
    for s in st[1:]:
        assert s["restorable"] and abs(s["metrics"]["alpn_median"] / 300.0 - 1) <= SPEC.alpn_match_tol
        assert type(s["params"]) is StdParams and s["params"].receptor_scale == s["scale"]
    assert not st[5]["feasible"] and st[5]["kc_pct"] == 12.0
    assert out["order"] == [1, 2, 3, 4]              # 0.75 > 0.51 = 0.51 (the tie goes to tau 893) > 0.19


def test_stage1_without_a_usable_setting_reports_no_feasible_setting():
    spec = dataclasses.replace(SPEC, std_f=(0.78,), std_tau_ms=(893.0,))
    assert J.stage1(ScriptedJ(), ScriptedRef(), jctx(spec=spec), Params())["outcome"] == R.NO_FEASIBLE_SETTING


def test_stage1_past_the_deadline_is_aborted_not_judged():
    out = J.stage1(ScriptedJ(), ScriptedRef(), jctx(deadline=0.0), Params())
    assert out["outcome"] == R.COMPUTE_ABORTED and out["order"] == []


# ---- stage 2: the control flow (reconverge / judge / D.6 stubbed) ---------------------------------------------------
SETTINGS = [dict(f=0.78, tau_ms=893.0, scale=20.0), dict(f=0.9, tau_ms=300.0, scale=4.0),
            dict(f=0.95, tau_ms=100.0, scale=1.5), dict(f=0.9, tau_ms=100.0, scale=2.0)]


@pytest.fixture
def stubs(monkeypatch):
    log = dict(bases=[], judged=[], d6=[])

    def install(statuses, outcome="SELECTED"):
        it = iter(statuses)

        def reconverge(m3, ctx, make_base):
            log["bases"].append(make_base(1.65))
            status = next(it)
            adopted = dict(params=make_base(1.65)) if status == H3R.COMBO_ADOPTED else None
            return dict(status=status, cells=[], adopted=adopted, guard={"MA1": {}}, note="x")
        monkeypatch.setattr(J, "reconverge", reconverge)
        monkeypatch.setattr(J, "judge", lambda m4, ctx, name, p, g: log["judged"].append(p) or
                            dict(outcome=outcome, reading={"T_b": 0.6}))
        monkeypatch.setattr(J, "measure_d6", lambda jm, ctx, p, seeds, judged: log["d6"].append(p) or {"a": 1})
    return log, install


def test_stage2_skips_dropped_settings_and_judges_the_first_adopted_one(stubs):
    log, install = stubs
    install([H3R.COMBO_DROPPED, H3R.COMBO_ADOPTED])
    out = J.stage2(None, None, ScriptedJ(), jctx(), SETTINGS, [1, 0, 2])
    assert out["outcome"] == R.SELECTED and [t["setting"] for t in out["tried"]] == [1, 0] and out["judged"] == 1
    b = log["bases"][0]
    assert type(b) is StdParams and (b.orn_std, b.orn_std_f, b.orn_std_tau_ms, b.receptor_scale) == (True, 0.9, 300.0, 4.0)
    assert (b.apl_mode, b.kc_thresh, b.apl_input_scale) == ("graded", 1.65, 1.0)      # C1's base, as C3's rule builds it
    assert log["d6"] == log["judged"] and "all51_after" in out["tried"][1] and out["d6"] == {"a": 1}


def test_stage2_tries_at_most_three_settings_then_stops_without_an_operating_point(stubs):
    log, install = stubs
    install([H3R.COMBO_DROPPED] * 4)
    out = J.stage2(None, None, ScriptedJ(), jctx(), SETTINGS, [0, 1, 2, 3])
    assert out["outcome"] == R.STOP_NO_OPERATING_POINT and len(out["tried"]) == SPEC.max_settings == 3
    assert log["judged"] == [] and log["d6"] == []


def test_stage2_abort_and_the_non_verdict_outcomes_skip_d6(stubs):
    log, install = stubs
    install([H3R.COMBO_ABORTED])
    assert J.stage2(None, None, ScriptedJ(), jctx(), SETTINGS, [0])["outcome"] == R.COMPUTE_ABORTED
    for outcome, measured in ((R.B, True), (R.INVALID, False), (R.STOP_MULTI_TYPE, False)):
        log["d6"].clear()
        install([H3R.COMBO_ADOPTED], outcome)
        assert J.stage2(None, None, ScriptedJ(), jctx(), SETTINGS, [0])["outcome"] == outcome
        assert bool(log["d6"]) == measured


# ---- stage 2: C3's rule on another engine -------------------------------------------------------------------------
def test_reconverge_is_c3_s_rule_from_the_first_kc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = J.reconverge(Scripted3(fired=fired_model(), mv=mv_model), jctx(), None)
    assert r["status"] == H3R.COMBO_ADOPTED and [c["kc"] for c in r["cells"]] == [1.65]
    assert type(r["adopted"]["params"]) is Params and set(r["guard"]) == set(POOLS["A"] + POOLS["P"])
    fails = Scripted3(fired=fired_model(), mv=mv_model,
                      pct=lambda p: (0.08, 0.045) if p.kc_thresh == 1.65 else (0.058, 0.045))
    r = J.reconverge(fails, jctx(), None)
    assert [c["kc"] for c in r["cells"]] == [1.65, 1.55, 1.6, 1.7] and r["status"] == H3R.COMBO_ADOPTED


def test_reconverge_on_a_depression_engine_adopts_a_depression_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from flymon.brain.h3_runner import c1_params
    from flymon.brain.j_params import with_std
    make = lambda kc: with_std(c1_params(H3_SPEC, kc, 1.0), 0.9, 300.0, 4.0)
    r = J.reconverge(Scripted3(fired=fired_model(), mv=mv_model), jctx(), make)
    p = r["adopted"]["params"]
    assert type(p) is StdParams and (p.orn_std_f, p.receptor_scale, p.kc_thresh_mode) == (0.9, 4.0, "homeostatic")


# ---- stage 2: H.4's reselection, oracle and the M2 bar ----------------------------------------------------------------
@pytest.mark.parametrize("testable, outcome", [((3, 2), R.SELECTED), ((2, 2), R.B), ((3, 1), R.B)])
def test_judge_reads_the_oracle_by_the_m2_bar(testable, outcome):
    m = S4.Scripted(testable=lambda n: testable)
    out = J.judge(m, jctx(), "C3", S4.COMBOS["C3"], S4.guard_of(m, "C3"))
    assert out["outcome"] == outcome and out["reading"]["n_b"] == 5


def test_judge_without_a_readout_is_b_and_two_types_stop():
    m = S4.Scripted(teach=lambda n: set())
    out = J.judge(m, jctx(), "C3", S4.COMBOS["C3"], S4.guard_of(m, "C3"))
    assert out["outcome"] == R.B and out["reading"] is None and "readout" in out["reason"]
    m = S4.Scripted(react=lambda n: set(S4.POOLS["A"] + S4.POOLS["P"]))
    assert J.judge(m, jctx(), "C3", S4.COMBOS["C3"], S4.guard_of(m, "C3"))["outcome"] == R.STOP_MULTI_TYPE


def test_judge_invalid_rows_and_a_guard_that_is_not_the_measurement():
    m = S4.Scripted(rows=lambda n, rows: rows[:-1])
    assert J.judge(m, jctx(), "C3", S4.COMBOS["C3"], S4.guard_of(m, "C3"))["outcome"] == R.INVALID
    m = S4.Scripted()
    with pytest.raises(RuntimeError, match="H.3 guard"):
        J.judge(m, jctx(), "C3", S4.COMBOS["C3"], {t: {"median_delta": -1, "zero_share": 1} for t in S4.TYPES})


def test_d6_on_other_seeds_is_recorded_but_not_judged():
    """A smoke run's D.6 (two seeds) cannot pass G.8's seed check: (b) is computed, (a) is left unjudged."""
    from flymon.brain import d6a
    ctx = jctx()
    out = J.measure_d6(ScriptedJ(), ctx, Params(), d6a.SEEDS[:2], judged=False)
    assert out["a"] is None and out["b"]["n_turns"] == 1 and out["n_rows"] == 2
    assert out["seed_block"] == "verification seeds (not judged)"
