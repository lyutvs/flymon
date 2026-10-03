# tests/brain/test_s_records.py
"""S.3 ④ / S.9.7: the gate-② ratio is ℓ_L / ℓ_C per direction on the same seeds (ℓ = p_rules' ℓ), with a paired seed
bootstrap CI; unequal seeds refuse; gate ②'s OC from P's block gives P(STOP_PUNISH_WEAKENED) = 1 − 0.5² at a true
ratio of exactly 0.5 and falls as the true ratio grows."""
import numpy as np
import pytest

from flymon.brain import s_records as SR
from flymon.brain.p_rules import p_judge
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.s_spec import SPEC
from tests.brain.s_fixtures import C1, Z, p_rows


def test_fixture_rows_pass_p_judge():
    res = p_judge(p_rows(SPEC.p_c, "none", 16), Z, C1, SPEC.p_c)
    assert res["label"] == "LEARNS_CONFIRMATORY", res.get("reasons")
    res = p_judge(p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2), Z, C1, SPEC.p)
    assert res["label"] == "LEARNS_CONFIRMATORY", res.get("reasons")


def test_ratio_is_ell_l_over_ell_c_on_the_same_seeds():
    rl = p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2)
    rc = p_rows(SPEC.p_c, "none", 16)
    out = SR.ratio(rl, rc, Z, SPEC)
    k = np.array([s % 3 for s in SPEC.p.seeds], float)
    want = float(np.mean(8 + k) / np.mean(16 + k))
    jl, jc = p_judge(rl, Z, C1, SPEC.p), p_judge(rc, Z, C1, SPEC.p_c)
    for d in SPEC.p.directions:
        assert out[d]["ratio"] == pytest.approx(want)
        assert out[d]["ell_L"] == pytest.approx(jl["directions"][d]["ell"])
        assert out[d]["ell_C"] == pytest.approx(jc["directions"][d]["ell"])
        lo, hi = out[d]["ratio_ci"]
        assert lo <= out[d]["ratio"] <= hi and out[d]["n"] == 32 and out[d]["seeds"] == list(SPEC.p.seeds)


def test_ratio_refuses_unequal_seeds():
    rl = p_rows(SPEC.p, SPEC.lever_edit, 8, sha="sha-L", edges=2)
    rc = [dict(r, seed=r["seed"] + 1) if r["seed"] == SPEC.p.seeds[0] else r for r in p_rows(SPEC.p_c, "none", 16)]
    rc = [r for r in rc if not (r["seed"] == SPEC.p.seeds[1])]
    with pytest.raises(ValueError):
        SR.ratio(rl, rc, Z, SPEC)


def test_gate2_oc_from_ps_block():
    oc = SR.gate2_oc(p_rows(P_SPEC, "none", 16), Z, P_SPEC, SPEC)
    assert oc["source"]["seeds"] == list(P_SPEC.seeds) and oc["source"]["n_rows"] == 192
    assert [r["rho"] for r in oc["rows"]] == list(SPEC.gate2_oc_rhos)
    at = {r["rho"]: r["p_stop_weakened"] for r in oc["rows"]}
    assert at[0.5] == pytest.approx(0.75)                       # Φ(0) in each of two directions
    ps = [at[r] for r in SPEC.gate2_oc_rhos]
    assert ps == sorted(ps, reverse=True) and at[0.4] > 0.75 and at[1.0] < 0.05
    for d, s in oc["directions"].items():
        assert s["n"] == 32 and s["se"] > 0 and s["ell"] == pytest.approx((16 + 1) / 9, rel=0.05)
