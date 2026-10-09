"""AD.3's contrasts on synthetic records: the decision rule only on the two primaries, the crossed CI everywhere on the
situation contrasts, point-wise minimum valid flies, the difference of differences within fly, the stage-1 C-off and FLY
columns, the paired OS evaluation, the retry sensitivity rows, and criteria 2-3 unmeasured without an evaluation block."""
import dataclasses

import pytest

from flymon.ad import analysis as an
from flymon.ad import stats
from flymon.ad.spec import SPEC

S = dataclasses.replace(SPEC, boot_draws=400)


def sw(n_on):
    return [1] * n_on + [0] * (20 - n_on)


def rec(g, point, n_on, arm=None, set_="confirm", mode="own"):
    return dict(fly=g, point=point, arm=arm or ("FLYL" if g < 36 else "FLYOS"), set=set_, mode=mode, frozen=True,
                switched=sw(n_on))


def arms(point, l_on, os_on, n_l=12, n_os=12, **kw):
    return ([rec(24 + i, point, l_on, **kw) for i in range(n_l)] + [rec(36 + i, point, os_on, **kw) for i in range(n_os)])


def test_two_stage_mean():
    m = stats.two_stage_mean({1: [1, 1, 0, 0], 2: [1, 0, 0, 0]}, 300, 304)
    assert m["mean"] == pytest.approx(0.375) and m["lo"] <= m["mean"] <= m["hi"] and m["n"] == 2


def test_primary_rule_and_crossed_ci():
    sits = arms(0, 4, 4) + arms(40, 6, 4) + arms(120, 12, 2)
    r = an.situation_readings(sits, S, final=120, new_points=[], coff=[], fly1=[], retried=[])
    p1, p2 = r["primary"]
    assert (p1["point"], p1["status"], p1["diff"]) == (120, "충족", pytest.approx(0.5))
    assert p1["crossed"]["lo"] <= p1["diff"] <= p1["crossed"]["hi"]
    assert (p2["point"], p2["status"]) == (40, "미충족")              # 0.1 < 0.15
    assert r["reference"]["status"] == "기술" and r["reference"]["diff"] == 0.0
    assert [d["point"] for d in r["did"]] == [40, 120] and all(d["status"] == "기술" for d in r["did"])


def test_point_wise_minimum_makes_an_item_unmeasured():
    sits = arms(0, 4, 4) + arms(40, 6, 4) + arms(120, 12, 2, n_os=8)
    p1, p2 = an.situation_readings(sits, S, final=120, new_points=[], coff=[], fly1=[], retried=[])["primary"]
    assert p1["status"] == "측정 불성립" and (p1["n_b"], p1["min_b"]) == (8, 9) and p2["status"] == "미충족"


def test_did_units_are_within_fly():
    sits = [rec(24, 0, 2), rec(24, 40, 5), rec(25, 40, 9)]                 # fly 25 has no point-0 record
    u = an.did_units(sits, "FLYL", 40)
    assert list(u) == [24] and sum(u[24]) == 3


def test_aux_uses_the_stage1_columns():
    coff = [dict(fly=12 + i, point=0, arm="COFF", frozen=True, switched=sw(1)) for i in range(6)]
    fly1 = [dict(fly=i, point=40, arm="FLY", frozen=True, switched=sw(5)) for i in range(12)]
    sits = arms(0, 4, 4) + arms(40, 6, 4)
    r = an.situation_readings(sits, S, final=40, new_points=[], coff=coff, fly1=fly1, retried=[])
    names = [a["name"] for a in r["aux"]]
    assert names == ["FLY-L(40) − C-off", "FLY-L(40) − 1단계 FLY(40)"]
    assert r["aux"][0]["diff"] == pytest.approx(0.25) and r["aux"][0]["n_b"] == 6 and r["aux"][0]["status"] == "기술"
    assert r["aux"][1]["diff"] == pytest.approx(0.05)
    assert r["primary"][0]["point"] == r["primary"][1]["point"] == 40          # STOP_FUTILE: 1차 1 = 1차 2


def test_new_set_and_paired_os():
    sits = (arms(0, 4, 4) + arms(40, 6, 4) + arms(0, 3, 3, set_="new") + arms(40, 8, 3, set_="new")
            + [rec(24 + i, 40, 2, mode="os_paired") for i in range(12)] + [rec(24 + i, 0, 4, mode="os_paired")
                                                                           for i in range(12)])
    r = an.situation_readings(sits, S, final=40, new_points=[0, 40], coff=[], fly1=[], retried=[])
    assert [c["point"] for c in r["new_set"]] == [0, 40] and r["new_set"][1]["diff"] == pytest.approx(0.25)
    po = {c["point"]: c for c in r["paired_os"]}
    assert set(po) == {0, 40} and po[40]["diff"] == pytest.approx(0.2) and po[0]["diff"] == pytest.approx(0.0)
    assert po[40]["status"] == "기술" and po[40]["n_a"] == 12 and "crossed" in po[40]


def test_sensitivity_excludes_retried_flies():
    sits = arms(0, 4, 4) + arms(40, 6, 4) + arms(120, 12, 2)
    r = an.situation_readings(sits, S, final=120, new_points=[], coff=[], fly1=[], retried=[24, 25, 26, 27])
    assert [c["status"] for c in r["sensitivity"]] == ["측정 불성립", "측정 불성립"]   # 8 FLY-L flies < 9
    r = an.situation_readings(sits, S, final=120, new_points=[], coff=[], fly1=[], retried=[24])
    assert r["sensitivity"][0]["n_a"] == 11 and r["sensitivity"][0]["status"] == "기술"


def test_budget_stop_marks_criteria_2_3_unmeasured():
    out = an.eval_readings({"status": "STOP_BUDGET", "per_fly": None}, [], [], S)
    assert [c["criterion"] for c in out] == [2, 3, 3, 3]
    assert all(c["status"] == "측정 불성립" and "평가 블록" in c["reason"] for c in out)


def test_eval_readings_on_complete_rows():
    rows = [dict(fly=i, arm="FLYL" if i < 12 else "FLYOS", k=i % 12, invalid=False,
                 eval_battles=[dict(won=(b < (10 if i < 12 else 5))) for b in range(20)]) for i in range(24)]
    nob = [dict(fly=k, arm=arm, k=k, invalid=False, eval_battles=[dict(won=b < 4) for b in range(20)])
           for arm in ("RND", "MAX") for k in range(16)]
    ev = [dict(kind="decision", decider="fly", fly=i, candidates=["a", "b"], multipliers=[2.0, 0.5],
               chosen="a" if i < 12 else "b") for i in range(24)]
    out = {(c["criterion"], c["a"], c["b"]): c for c in an.eval_readings({"status": "COMPLETE", "per_fly": rows}, ev,
                                                                          nob, S)}
    assert out[(2, "FLYL", "FLYOS")]["diff"] == pytest.approx(1.0) and out[(2, "FLYL", "FLYOS")]["status"] == "기술"
    assert out[(3, "FLYL", "FLYOS")]["diff"] == pytest.approx(0.25)
    assert out[(3, "FLYL", "RND")]["diff"] == pytest.approx(0.3) and out[(3, "MAX", "FLYL")]["diff"] == pytest.approx(-0.3)
    assert "crossed" not in out[(3, "FLYL", "RND")]
