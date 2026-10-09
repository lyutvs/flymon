"""Point-wise validity, the futility check at 40 and the final reading point (AD.2, AD.4, AD.7) on synthetic records."""
import json

import pytest

from flymon.ad import validity as va
from flymon.ad.spec import SPEC

LAY = SPEC.layout()
PAIRS = [dict(cands=["m0", "m1"], best1="m0", best2="m1") for _ in range(4)]


def rec(g, point, pos, arm=None, set_="confirm", mode="own", frozen=True):
    sits = []
    for i, p in enumerate(pos):
        sits.append(dict(pair=i, side=0, v=[1.0, 0.0] if p else [0.0, 1.0]))
        sits.append(dict(pair=i, side=1, v=[0.0, 1.0] if p else [1.0, 0.0]))
    return dict(fly=g, point=point, arm=arm or ("FLYL" if g < 36 else "FLYOS"), set=set_, mode=mode, frozen=frozen,
                situations=sits, switched=[int(p) for p in pos])


def test_records_and_share():
    sits = [rec(24, 0, [1, 0, 0, 0]), rec(25, 0, [1, 1, 0, 0]), rec(36, 0, [1, 1, 1, 1]),
            rec(26, 0, [1, 1, 1, 1], frozen=False), rec(24, 0, [1, 1, 1, 1], mode="os_paired")]
    r = va.records(sits, point=0, arm="FLYL")
    assert sorted(r) == [24, 25] and va.share(r, PAIRS) == pytest.approx(0.375)
    assert va.share({}, PAIRS) is None
    with pytest.raises(ValueError):
        va.records(sits + [rec(24, 0, [0, 0, 0, 0])], point=0)


def test_futility_rule():
    base = [rec(24, 0, [1, 1, 0, 0]), rec(25, 0, [1, 0, 0, 0])]                 # share at 0 = 0.375
    same = base + [rec(24, 40, [1, 0, 0, 0]), rec(25, 40, [1, 1, 0, 0])]        # 0.375 <= 0.375
    f = va.futility(same, PAIRS, SPEC)
    assert f["status"] == "STOP_FUTILE" and f["share_at_point"] == f["share_at_0"] == pytest.approx(0.375)
    up = base + [rec(24, 40, [1, 1, 1, 0]), rec(25, 40, [1, 0, 0, 0])]
    assert va.futility(up, PAIRS, SPEC)["status"] == "CONTINUE"
    osonly = base + [rec(36, 40, [1, 1, 1, 1])]
    assert va.futility(osonly, PAIRS, SPEC)["status"] == "NO_FLYL"
    line = va.futility_line(f)
    assert line == "futility check at 40: FLY-L right-direction share 0.375 vs point 0 0.375 -> STOP_FUTILE"


def test_futility_decision_is_kept_across_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = "results/m4/ad/futility.json"
    f = va.futility([rec(24, 0, [1, 0, 0, 0]), rec(24, 40, [1, 1, 0, 0])], PAIRS, SPEC)
    assert va.persist_futility(p, f) == f
    again = va.persist_futility(p, dict(f))
    assert again == json.loads((tmp_path / p).read_text())
    with pytest.raises(SystemExit, match="STOP_INFRA"):
        va.persist_futility(p, dict(f, status="STOP_FUTILE"))


def test_budget_stop_reads_the_latest_common_point():
    sits = ([rec(24, p, [1, 0, 0, 0]) for p in (0, 20, 40, 60)] + [rec(25, p, [1, 0, 0, 0]) for p in (0, 20, 40, 60, 80)]
            + [rec(36, p, [0, 0, 0, 0]) for p in (0, 20, 40, 60)] + [rec(37, p, [0, 0, 0, 0]) for p in (0, 20)])
    invalid = {13: "DL-f37-b030: unfinished after 3 retries"}                   # pool index 13 = fly 37
    assert va.reached(sits) == {24: 60, 25: 80, 36: 60, 37: 20}
    lay = [x for x in LAY if x[2] in (24, 25, 36, 37)]
    assert va.latest_common_point(sits, LAY[:2] + LAY[12:14], {3: invalid[13]}) == 60
    fp = va.final_point(SPEC, 120, "CONTINUE", "STOP_BUDGET", sits, LAY[:2] + LAY[12:14], {3: invalid[13]})
    assert fp == {"point": 60, "reason": "STOP_BUDGET"}
    assert va.final_point(SPEC, 120, "STOP_FUTILE", "COMPLETE", sits, lay, {}) == {"point": 40, "reason": "STOP_FUTILE"}
    assert va.final_point(SPEC, 120, "CONTINUE", "COMPLETE", sits, lay, {}) == {"point": 120, "reason": "complete"}
    assert va.final_point(SPEC, 80, "CONTINUE", "COMPLETE", sits, lay, {}) == {"point": 80, "reason": "fallback"}


def test_dropouts_and_valid_counts():
    sits = [rec(24, p, [1, 0, 0, 0]) for p in (0, 20, 40)] + [rec(36, p, [0, 0, 0, 0]) for p in (0, 20)]
    d = va.dropouts({12: "DL-f36-b025: unfinished after 3 retries"}, LAY, sits)
    assert d == [dict(fly=36, arm="FLYOS", reason="DL-f36-b025: unfinished after 3 retries", last_point=20)]
    n = va.n_valid_by_point(sits, LAY, (0, 20, 40))
    assert n == {0: {"FLYL": 1, "FLYOS": 1}, 20: {"FLYL": 1, "FLYOS": 1}, 40: {"FLYL": 1, "FLYOS": 0}}


def test_stopped_arm_fly_without_mark_does_not_lower_the_common_point():
    # FLY-OS stopped (STOP_INFRA) after fly 37 went invalid; fly 36's next battle never ran, so hooks never marked it
    # "arm_stopped" - it has no later record and no invalid mark, and must not pull the common point down to 20.
    sits = ([rec(24, p, [1, 0, 0, 0]) for p in (0, 20, 40, 60)] + [rec(25, p, [1, 0, 0, 0]) for p in (0, 20, 40, 60)]
            + [rec(36, p, [0, 0, 0, 0]) for p in (0, 20)] + [rec(37, p, [0, 0, 0, 0]) for p in (0,)])
    lay = LAY[:2] + LAY[12:14]
    invalid = {3: "DL-f37-b005: unfinished after 3 retries"}
    stopped = {"FLYOS": dict(status="STOP_INFRA", n_invalid=1, size=12, min_valid=9)}
    assert va.latest_common_point(sits, lay, invalid) == 20                     # without the stop the fly still counts
    assert va.latest_common_point(sits, lay, invalid, stopped=stopped) == 60
    assert va.latest_common_point(sits, lay, invalid, stopped=["FLYOS"]) == 60
    fp = va.final_point(SPEC, 120, "CONTINUE", "STOP_BUDGET", sits, lay, invalid, stopped=stopped)
    assert fp == {"point": 60, "reason": "STOP_BUDGET"}
    assert va.latest_common_point(sits, lay, invalid, stopped={"FLYOS": {}, "FLYL": {}}) is None
