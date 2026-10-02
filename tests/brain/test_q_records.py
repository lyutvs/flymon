# tests/brain/test_q_records.py
"""Readings 5, 16, 17: per-pair values (r_P = combo_records' single_type P; r = dV mean / SD; ratio None when naive P_X
is 0), the Q0 record (sha gate, 12/9 split, borderline, A·Y cancel) and the per-condition gate (completeness,
uniqueness, finite values, one CSC sha, edit edge counts, the KC band for gated conditions)."""
import math

import numpy as np
import pytest

from flymon.brain import q_records as R
from flymon.brain.h4_rules import combo_records
from flymon.brain.h4_spec import SPEC as H4
from flymon.brain.q_spec import SPEC
from tests.brain.q_fixtures import KEYS, Z, fake_result

COND = {c.name: c for c in SPEC.conditions(1.3, 0.8, 1.25)}


def test_r_p_is_combo_records_single_type_and_r_is_mean_over_sd():
    res = fake_result(8, 8, px=20, dpx=-6, sd=2.0, seed=1)
    v = R.pair_values(res, Z, SPEC)
    rows = [dict(axis="b", turn=0, x="a", y="b", report=res["report"])]
    single = combo_records(rows, Z, H4)["single_type"]["P"]["pairs"][0]
    assert v["r_P"] == pytest.approx(single["r"])
    assert v["r"] == pytest.approx(v["dV_mean"] / v["dV_sd"])
    assert v["naive_px"] == pytest.approx(np.mean([p[0] for p in res["report"]["pre"]["P"]]))
    assert set(v["dpx_fixed"]) == {"0.8", "1.0"} and v["ratio"] == pytest.approx(abs(v["dpx_fixed"]["0.8"]) / v["naive_px"])
    assert 0.0 <= v["stop_naive"] <= 1.0 and v["W_X"] == 1.0


def test_ratio_is_none_when_x_is_silent():
    res = fake_result(8, 8, px=0, dpx=0, sd=0.0, seed=2)
    for ph in ("pre", "R1", "R2"):
        res["report"][ph]["P"] = [[0, 10 + i % 3] for i in range(8)]
    res["q"]["fixed"]["0.8"]["report"]["P"] = [[0, 10] for _ in range(8)]
    v = R.pair_values(res, Z, SPEC)
    assert v is not None and v["naive_px"] == 0 and v["ratio"] is None   # silent X: no ratio, the pair is still read
    assert v["dpx_fixed"]["0.8"] == 0.0 and v["r"] == 0.0                # dV unchanged: d' of an all-zero change is 0


def test_q0_record_without_q_fields_and_split():
    res = [fake_result(8, 8, px=20, dpx=(-12 if i >= 12 else -0.5), sd=(1.0 if i >= 12 else 3.0), seed=i, q=False)
           for i in range(21)]
    files = [dict(key=k, ok=True, path=f"p{i}") for i, k in enumerate(KEYS)]
    rec = R.q0_record(files, res, SPEC, Z)
    assert rec["status"] == R.OK, rec["reasons"]
    assert len(rec["split"]["F"]) == 12 and len(rec["split"]["S"]) == 9
    assert "dpx_fixed" not in next(iter(rec["vals"].values()))
    files[3]["ok"] = False
    bad = R.q0_record(files, res, SPEC, Z)
    assert bad["status"] == R.INVALID and any("digest" in r for r in bad["reasons"])


def _got(cond, n=21, **kw):
    return [dict(key=KEYS[i], result=fake_result(8, 8, px=20, dpx=-6, sd=2.0, seed=i, **kw), cache_key="c",
                 cache_file="f") for i in range(n)]


def test_condition_ok_and_records():
    rec = R.condition_record(_got(COND["base"]), COND["base"], SPEC, Z, KEYS, same_sha="sha-base")
    assert rec["status"] == R.OK and rec["kc_median"] == 0.05 and rec["edit_edges"] == [0]
    assert rec["d6a_over_share"] == 0.0 and rec["apl_out_median"] == 0.1
    assert rec["naive_px_zero"] == 0


def test_naive_px_zero_counts_pairs_already_at_floor():
    g = _got(COND["base"])
    for i in (4, 9):
        for ph in ("pre", "R1", "R2"):
            g[i]["result"]["report"][ph]["P"] = [[0, 10 + j % 3] for j in range(8)]
    rec = R.condition_record(g, COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.OK and rec["naive_px_zero"] == 2


@pytest.mark.parametrize("mutate,needle", [
    (lambda g: g[:-1], "missing"),
    (lambda g: g + g[:1], "duplicate"),
    (lambda g: [dict(x, result=dict(x["result"], q={**x["result"]["q"], "csc_sha256": f"s{i}"})) for i, x in enumerate(g)],
     "sha"),
])
def test_condition_invalid_on_defects(mutate, needle):
    rec = R.condition_record(mutate(_got(COND["base"])), COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.INVALID and any(needle in r for r in rec["reasons"]), rec["reasons"]


def test_condition_invalid_on_short_probes_and_missing_fixed_arm():
    g = _got(COND["base"])
    g[0]["result"]["report"]["pre"]["P"] = g[0]["result"]["report"]["pre"]["P"][:7]
    del g[1]["result"]["q"]["fixed"]["1.0"]
    rec = R.condition_record(g, COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.INVALID and sum(KEYS[0] in r or KEYS[1] in r for r in rec["reasons"]) == 2


def test_non_finite_r_is_invalid_not_a_crash():
    g = _got(COND["base"])
    rep = g[2]["result"]["report"]
    n = len(rep["pre"]["P"])
    for ph in ("pre", "R1", "R2"):                                     # every seed identical, so dV is exactly
        rep[ph]["A"] = [[30, 30] for _ in range(n)]                    # constant per phase (no rounding noise)
    rep["pre"]["P"] = [[20, 10] for _ in range(n)]
    rep["R1"]["P"] = rep["R2"]["P"] = [[17, 10] for _ in range(n)]     # R1 = pre - 3 on every seed: SD exactly 0 -> r = inf
    rec = R.condition_record(g, COND["base"], SPEC, Z, KEYS)
    assert rec["status"] == R.INVALID and any(KEYS[2] in r for r in rec["reasons"])


def test_edit_edge_counts_and_kc_band():
    ok = R.condition_record(_got(COND["apl_mbon05"], sha="sha-m05", edges=2), COND["apl_mbon05"], SPEC, Z, KEYS,
                            other_sha="sha-base")
    assert ok["status"] == R.OK
    three = R.condition_record(_got(COND["apl_mbon05"], sha="sha-m05", edges=3), COND["apl_mbon05"], SPEC, Z, KEYS)
    assert three["status"] == R.INVALID
    same = R.condition_record(_got(COND["apl_mbon05"], edges=2), COND["apl_mbon05"], SPEC, Z, KEYS,
                              other_sha="sha-base")
    assert same["status"] == R.INVALID
    hot = R.condition_record(_got(COND["s_up"], kc=0.2), COND["s_up"], SPEC, Z, KEYS, same_sha="sha-base")
    assert hot["status"] == R.INVALID and any("KC" in r for r in hot["reasons"])
    base_hot = R.condition_record(_got(COND["base"], kc=0.2), COND["base"], SPEC, Z, KEYS)
    assert base_hot["status"] == R.OK                                   # the base is recorded, not gated
    assert math.isclose(hot["kc_median"], 0.2)
