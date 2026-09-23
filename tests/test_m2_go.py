"""Spec 5's M2 go condition, kept as strict expected failures (spec appendix I; the A.5 precedent). They are not removed
or relaxed: the day a tested engine reaches the bar, or a learning unit test passes, they XPASS and the suite breaks."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M0D = ROOT / "results/summary/m0d.json"
LEARNING = ROOT / "results/summary/m2_learning.json"      # F.10 #4's summary: written only if the learning test runs
ENGINE = ROOT / "results/summary/m2_engine.json"          # spec J.11.4-4: a depression engine's selection, if any


@pytest.mark.xfail(strict=True, reason="M2 no-go — 시험 불성립 (spec appendix I): no tested engine reaches "
                                       "T_b >= 0.5 and F_a >= 2 (H.4 STOP_LOW_T_B; best C3 7/21)")
def test_m2_learning_unit_test_can_be_built():
    """G.14.4 / H.4: spec 5's learning unit test can be built only on an engine whose idealised specific edit makes at
    least half of the (b) pairs testable (T_b >= 0.5) and at least two naive-balanced (a) pairs testable (F_a >= 2)."""
    combos = json.loads(M0D.read_text())["h4"]["h4"]["combos"]
    aggs = [c["oracle"]["aggregate"] for c in combos.values()]
    sel = json.loads(ENGINE.read_text()).get("selection") if ENGINE.exists() else None      # J.10.6: widened, not relaxed
    assert any(a["T_b"] >= 0.5 and a["F_a"] >= 2 for a in aggs) or \
        bool(sel and sel["confirmed"] and sel["T_b"] >= 0.5 and sel["F_a"] >= 2)


@pytest.mark.xfail(strict=True, reason="M2 no-go — 시험 불성립 (spec appendix I): spec 5's learning unit test never "
                                       "ran (F v3 withdrawn unrun, F v4 not written)")
def test_m2_go():
    """Spec 5, M2: naive d' ~ 0, d' >= 1 after 20 rewards, a drop after 20 punishments — F.7's overall verdict PASS."""
    assert json.loads(LEARNING.read_text())["outcome"] == "PASS"
