import json

import pytest

from flymon.rescope.spec import SPEC
from flymon.rescope.yoke import YokedQueue


def _log(tmp_path, rows):
    p = tmp_path / "fly00.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_queue_keeps_nonempty_learning_bundles_in_order(tmp_path):
    rows = [dict(kind="reinforce", battle_id="PL-f00-b000", pulses=[["PAM08", 400.0]]),
            dict(kind="reinforce", battle_id="PL-f00-b000", pulses=[]),
            dict(kind="decision", battle_id="PL-f00-b000"),
            dict(kind="reinforce", battle_id="PL-f00-b001", pulses=[["PAM08", 200.0], ["PPL105", 200.0]]),
            dict(kind="reinforce", battle_id="PE-f00-b000", pulses=[["PAM08", 400.0]])]
    q = YokedQueue.from_log(_log(tmp_path, rows), {"PL-f00-b000", "PL-f00-b001"})
    assert q.total_ms() == 800.0
    assert q.pop() == [("PAM08", 400.0)]
    assert q.pop() == [("PAM08", 200.0), ("PPL105", 200.0)]
    assert q.pop() is None and q.residual_frac() == 0.0


def test_empty_turns_never_enter_the_queue(tmp_path):
    rows = [dict(kind="reinforce", battle_id="PL-f00-b000", pulses=[]) for _ in range(5)]
    rows.insert(2, dict(kind="reinforce", battle_id="PL-f00-b000", pulses=[["PPL105", 400.0]]))
    q = YokedQueue.from_log(_log(tmp_path, rows), {"PL-f00-b000"})
    assert len(q.bundles) == 1 and q.pop() == [("PPL105", 400.0)] and q.pop() is None
    with pytest.raises(ValueError):
        YokedQueue.from_bundles([[("PAM08", 1.0)], []], donor_sha256="x")


def test_residual_and_resume(tmp_path):
    rows = [dict(kind="reinforce", battle_id="PL-f00-b000", pulses=[["PAM08", 100.0]]) for _ in range(20)]
    q = YokedQueue.from_log(_log(tmp_path, rows), {"PL-f00-b000"})
    for _ in range(19): q.pop()
    assert abs(q.residual_frac() - 0.05) < 1e-12
    st = q.state()
    assert json.loads(json.dumps(st)) == st
    q2 = YokedQueue.from_log(_log(tmp_path, rows), {"PL-f00-b000"}); q2.load_state(st)
    assert q2.pop() == [("PAM08", 100.0)] and q2.pop() is None


def test_sha_mismatch_refused(tmp_path):
    rows = [dict(kind="reinforce", battle_id="PL-f00-b000", pulses=[["PAM08", 100.0]])]
    q = YokedQueue.from_log(_log(tmp_path, rows), {"PL-f00-b000"})
    st = q.state()
    _log(tmp_path, rows + rows)                       # donor rerun: log changed
    q3 = YokedQueue.from_log(tmp_path / "fly00.jsonl", {"PL-f00-b000"})
    with pytest.raises(ValueError):
        q3.load_state(st)


def test_out_of_range_state_refused():
    q = YokedQueue.from_bundles([[("PAM08", 1.0)]], donor_sha256="x")
    with pytest.raises(ValueError):
        q.load_state({"i": 2, "exhausted": 0, "n_bundles": 1, "donor_sha256": "x"})


def test_fewer_rs_turns_residual_boundary():
    """RS gets fewer fly turns than the donor: leftovers are dropped; <= 5 % of donor ms is valid, above is INVALID."""
    q = YokedQueue.from_bundles([[("PAM08", 100.0)]] * 20, donor_sha256="x")
    for _ in range(19): q.pop()
    assert q.dropped_bundles() == 1 and q.residual_ok(SPEC.residual_max)          # exactly 5 %: still valid
    q = YokedQueue.from_bundles([[("PAM08", 100.0)]] * 19 + [[("PAM08", 100.0), ("PPL105", 1.0)]], donor_sha256="x")
    for _ in range(19): q.pop()
    assert q.residual_frac() > SPEC.residual_max and not q.residual_ok(SPEC.residual_max)
    s = q.summary(SPEC.residual_max)
    assert s["valid"] is False and s["dropped_bundles"] == 1 and s["exhausted_turns"] == 0


def test_more_rs_turns_counts_exhausted_turns():
    """RS gets more fly turns than the donor: the extra turns get no pulse, are counted, and residual is 0."""
    q = YokedQueue.from_bundles([[("PAM08", 400.0)], [("PPL105", 400.0)]], donor_sha256="x")
    got = [q.pop() for _ in range(5)]
    assert got[2:] == [None, None, None] and q.exhausted == 3
    assert q.residual_frac() == 0.0 and q.delivered_ms() == q.total_ms() == 800.0
    st = q.state()
    q2 = YokedQueue.from_bundles([[("PAM08", 400.0)], [("PPL105", 400.0)]], donor_sha256="x"); q2.load_state(st)
    assert q2.exhausted == 3 and q2.pop() is None and q2.exhausted == 4
