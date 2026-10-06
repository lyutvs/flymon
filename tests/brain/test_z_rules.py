"""Z's order-0 rules (Z.9.2 P1-3 (가) / (나), P0-1 key pre-check, P2-11 env, Z.7 0b compare, Z.8 / P3-14 sentences)."""
import dataclasses
import re
from pathlib import Path

import pytest

from flymon.brain import z_rules as R
from flymon.brain.z_spec import SPEC as Z

SPEC_MD = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/2026-09-14-flymon-design.md"


def _L(v12, v24, other=0.9):
    """L[(δ index, n)]: the −1 SE row is (v12, v24); δ 0 and −2 SE rows are `other`."""
    L = {(di, n): other for di in range(len(Z.sens_deltas)) for n in Z.sens_ns}
    L[(Z.sens_cont_delta, 12)], L[(Z.sens_cont_delta, 24)] = v12, v24
    return L


def test_rule_a_needs_both_n_below_the_bar():
    assert R.plan_rule_a(_L(0.79, 0.81), Z)["outcome"] == R.PASS
    assert R.plan_rule_a(_L(0.81, 0.79), Z)["outcome"] == R.PASS
    out = R.plan_rule_a(_L(0.79, 0.799), Z)
    assert out["outcome"] == R.STOP_PLAN_UNREACHABLE and out["where"] == "0c′" and out["records_unavailable"]
    assert "n 12 0.790 · n 24 0.799" in out["sentence"] and "분할 전" in out["sentence"]
    assert "n 24 0.799로 0.80에 못 미쳤다" in out["sentence"] and "1 SE(2.41)" in out["sentence"]
    assert out["p314"]["halves"] == dict(pilot="미사용", confirm="미사용")


def test_rule_a_bar_is_inclusive_and_uses_minus_one_se():
    assert R.plan_rule_a(_L(0.80, 0.0), Z)["outcome"] == R.PASS             # 0.80 passes (T8)
    assert R.plan_rule_a(_L(0.8 - 1e-12, 0.8 - 1e-12), Z)["outcome"] == R.PASS   # 9-digit rounding
    L = _L(0.9, 0.9, other=0.0)                                               # δ 0 / −2 SE rows never decide
    assert R.plan_rule_a(L, Z)["outcome"] == R.PASS


def test_n_star_steps():
    assert [R.n_star(j, Z) for j in (12, 23, 24, 30)] == [12, 12, 24, 24]


def test_key_precheck_counts_keys_only():
    p = [f"b|{900 + i}|x{i}|y{i}" for i in range(5)] + ["a|950|Earthquake|Strength", "a|951|Surf|Psychic"]
    pre = R.key_precheck(p, Z)
    assert pre == dict(h=7, j_max=13, n_sigma_max=5 + 6)       # Y 6 merge to 5; Earthquake joins Y's group
    assert R.pilot_few_0d(pre, 24, 3, Z) is None


def test_pilot_few_0d_below_twelve():
    pre = R.key_precheck(["b|900|x|y"] * 0 + [f"b|{900 + i}|x{i}|y{i}" for i in range(5)], Z)
    out = R.pilot_few_0d(pre, 26, 14, Z)
    assert pre["j_max"] == 11 and out["outcome"] == R.STOP_PILOT_FEW and out["where"] == "0d"
    assert "파일럿 반이 5쌍" in out["sentence"] and "6 + 5 = 11" in out["sentence"] and "강제 묶음 14쌍" in out["sentence"]
    assert R.pilot_few_0d(R.key_precheck([f"b|{900 + i}|x{i}|y{i}" for i in range(6)], Z), 25, 0, Z) is None


def test_rule_b_uses_n_star_and_minus_one_se():
    L = _L(0.79, 0.85)
    assert R.plan_rule_b(L, 30, Z)["outcome"] == R.PASS                # n* 24 → 0.85
    out = R.plan_rule_b(L, 20, Z)                                       # n* 12 → 0.79
    assert out["outcome"] == R.STOP_PLAN_UNREACHABLE and out["n_star"] == 12 and out["where"] == "0d"
    assert "최대 J 20 → n 12에서 0.790" in out["sentence"] and "분할 뒤" in out["sentence"]
    assert "0.790로 0.80에 못 미쳤다" in out["sentence"] and "1 SE(2.41)" in out["sentence"]
    assert out["records_unavailable"] is True and out["records_reason"] == R.RECORDS_REASON
    assert R.plan_rule_b(_L(0.80, 0.0), 13, Z)["outcome"] == R.PASS


def test_diag_compare_numpy_rounding_and_missing():
    zs = dataclasses.replace(Z, printed=(("a", 0.962, 3), ("b", 37, 0), ("c", 1.2, 2), ("d", 0.5, 1)))
    out = R.diag_compare(dict(a=0.9625, b=37, c=1.1948), zs)
    rows = {r["name"]: r for r in out["rows"]}
    assert rows["a"]["equal"] and rows["b"]["equal"] and not rows["c"]["equal"]
    assert rows["d"]["computed"] is None and out["differ"] == ["c", "d"] and out["n_equal"] == 2


def test_env_rules():
    old = dict(numpy="2", z_files={"a.py": "1"}, yxw_files={"y.py": "2"})
    assert R.env_diff(old, dict(numpy="2", z_files={"a.py": "1", "b.py": "9"}, yxw_files={"y.py": "2"})) == []
    assert R.env_diff(old, dict(numpy="3", z_files={"a.py": "1"}, yxw_files={"y.py": "2"})) == ["numpy"]
    assert R.env_diff(old, dict(numpy="2", z_files={}, yxw_files={"y.py": "2"})) == ["z_files:a.py"]
    assert R.env_reasons([("stage0", old)], dict(old, numpy="3")) == ["stage0 대비 numpy"]
    later = dict(old, numpy="3")                     # only the earlier block differs from now
    assert R.env_reasons([("stage0", old), ("split", later)], later) == ["stage0 대비 numpy"]


def test_reuse_stop_rows():
    for where, first in (("0b", "부"), ("0c", "필")):
        out = R.reuse_stop(["x"], where, Z)
        assert out["outcome"] == R.STOP_REUSE and out["p314"]["outputs"]["ydiag"] == first
        assert f"Z.7 {where}" in out["sentence"] and out["records_unavailable"]


def _p314_spec_rows() -> dict:
    """P3-14's table as printed in the spec: {(label, 단계): (①…⑧, ⑨)} for every label of every row."""
    text = SPEC_MD.read_text()
    tab = text[text.index("**P3-14 STOP 표식별 산출물"):]
    rows = {}
    lines = tab.splitlines()
    start = lines.index("|---|---|---|---|---|---|---|---|---|---|") + 1
    for line in lines[start:]:
        if not line.startswith("| `"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for label, where in re.findall(r"`(\w+)`〈([^〉]+)〉", cells[0]):
            rows[(label, where)] = (tuple(cells[1:9]), cells[9])
    return rows


def test_p314_rows_match_the_spec_table():
    rows = _p314_spec_rows()
    assert len(R.P314) == 5
    for (outcome, where), got in R.P314.items():
        want, halves = rows[(outcome, where)]
        assert got == want, (outcome, where)
        out = R.p314(outcome, where)
        assert tuple(out["outputs"].values()) == want
        assert f"{out['halves']['pilot']} / {out['halves']['confirm']}" == halves
    assert rows[("STOP_REUSE", "0b")][0] == ("부",) + ("불",) * 7                      # the parse itself is pinned
    assert rows[("STOP_PLAN_UNREACHABLE", "0d")][0] == ("필",) * 3 + ("불",) * 5


@pytest.mark.parametrize("mut", ["delta_zero", "always_24"])
def test_mutants_are_caught(mut, monkeypatch):
    if mut == "delta_zero":
        zs = dataclasses.replace(Z, sens_cont_delta=0)
        assert R.plan_rule_a(_L(0.7, 0.7, other=0.9), zs)["outcome"] == R.PASS     # the mutant passes …
        assert R.plan_rule_a(_L(0.7, 0.7, other=0.9), Z)["outcome"] != R.PASS      # … the rule does not
    else:
        monkeypatch.setattr(R, "n_star", lambda j, zs: 24)
        assert R.plan_rule_b(_L(0.79, 0.85), 20, Z)["outcome"] == R.PASS
        monkeypatch.undo()
        assert R.plan_rule_b(_L(0.79, 0.85), 20, Z)["outcome"] != R.PASS
