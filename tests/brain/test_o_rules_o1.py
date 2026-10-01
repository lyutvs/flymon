# tests/brain/test_o_rules_o1.py
"""Spec O.7.2 / O.7.5: the validity gate before any judgement; mixed cells (both shares >= 0.10, fewer than 2 ->
STOP_NO_SILENT_STATE); the state nature from the P mid band and the APL / KC AUCs; the pathway 2x2 with its 0.25 ratio
and CI clause; per-stimulus drive labels with 99% seed-cluster slope CIs; the records. The state is n_rules.state."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import n_rules
from flymon.brain import o_rules as R
from flymon.brain.o_spec import SPEC

S0 = 100


def _spec(g=(0.25, 1.0), stim=("4:1", "dDL"), n=40, draws=500):
    return replace(SPEC, o1_g_grid=g, o1_stimuli=stim, o1_seed0=S0, o1_n_seeds=n, boot_draws=draws)


def _row(cond, edit, g, stim, seed, P, apl, kc):
    return dict(seed=seed, A=10, P=P, kc_frac=0.05, kc_spikes=kc, kc_max_win_hz=50.0, apl_out_per_step=apl,
                wall_s=0.01, steps=1400, edit=edit, csc_sha256="sha-" + edit, cond=cond, g=float(g), stim=stim)


def _rows(spec, sil, fire_P=40, same_net=False):
    """sil(cond, g, stim, i) -> silent? Silent: P 0, APL 1.0, KC 10; firing: P fire_P, APL 0.1, KC 100 (same_net: APL /
    KC equal in both states)."""
    out = []
    for cond, edit in spec.o1_conditions:
        for g in spec.o1_g_grid:
            for s in spec.o1_stimuli:
                for i, seed in enumerate(spec.o1_seeds):
                    q = sil(cond, g, s, i)
                    apl, kc = (0.5, 50) if same_net else ((1.0, 10) if q else (0.1, 100))
                    out.append(_row(cond, edit, g, s, seed, 0 if q else fire_P, apl, kc))
    return out


def test_state_and_cell_stats_are_ns():
    assert R.state is n_rules.state and R.cell_stats is n_rules.cell_stats
    sp = _spec()
    rows = [dict(P=p) for p in (0, 4, 5, 40)]
    assert R.silent(rows, sp).tolist() == [True, True, False, False]


def test_auc_and_boot_weights():
    assert R.auc([2, 3], [1, 1]) == 1.0 and R.auc([1], [1]) == 0.5 and R.auc([], [1]) is None
    W = R.boot_weights(5, 7, 11)
    idx = n_rules.boot_index(5, 7, 11)
    assert np.allclose(W.sum(1), 1.0) and np.allclose(W[3] * 5, np.bincount(idx[3], minlength=5))


def test_the_gate_names_every_defect_and_judges_nothing():
    sp = _spec()
    rows = _rows(sp, lambda c, g, s, i: c == "on" and i < 12)
    assert R.o1_judge(rows, sp)["outcome"] == R.JUDGED
    cases = {
        "no row": rows[1:],
        "more than once": rows + [rows[0]],
        "undeclared seed": [dict(rows[0], seed=999)] + rows[1:],
        "non-finite": [dict(rows[0], apl_out_per_step=float("nan"))] + rows[1:],
        "several CSC": [dict(rows[0], csc_sha256="other")] + rows[1:],
        "share one CSC": [dict(r, csc_sha256="sha-none") if r["edit"] == "apl_to_nonkc_zero" else r for r in rows],
    }
    for why, bad in cases.items():
        res = R.o1_judge(bad, sp)
        assert res["outcome"] == R.INVALID and any(why in x for x in res["reasons"]), why
        assert "nature" not in res and "pathway" not in res


def test_fewer_than_two_mixed_cells_stops():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda *a: False), sp)
    assert res["outcome"] == R.STOP_NO_SILENT_STATE and res["mixed_cells"] == []
    assert res["nature"] is None and res["pathway"] is None and res["drive"] is None and res["cells"]["on"]
    one = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and g == 0.25 and s == "4:1" and i < 8), sp)
    assert one["outcome"] == R.STOP_NO_SILENT_STATE and one["mixed_cells"] == ["0.25|4:1"]


@pytest.mark.parametrize("k, mixed", [(4, True), (3, False)])
def test_the_mixed_threshold_is_ten_percent(k, mixed):
    sp = _spec()                                                    # 40 seeds: 4 -> 0.10 (mixed), 3 -> 0.075
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < k), sp)
    assert (res["outcome"] == R.JUDGED) is mixed and len(res["mixed_cells"]) == (4 if mixed else 0)


def test_bistable_network_and_readout_path():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c in ("on", "kc") and i < 12), sp)
    assert res["outcome"] == R.JUDGED and len(res["mixed_cells"]) == 4
    assert res["nature"]["label"] == R.BISTABLE_NETWORK
    assert all(c["auc_apl"] == 1.0 and c["auc_apl_silent_above"] == 1.0 for c in res["nature"]["cells"])
    assert res["pathway"]["label"] == R.READOUT_PATH and res["pathway"]["q"]["on"] == pytest.approx(0.3)
    assert {s: v["label"] for s, v in res["drive"].items()} == {"4:1": R.FLAT, "dDL": R.FLAT}
    assert res["phi_on"] and all(p["phi"] == pytest.approx(1.0) for p in res["phi_on"])
    assert res["cells"]["on"]["0.25|4:1"]["P"][:12] == [0] * 12


def test_bimodal_p_readout_when_the_network_does_not_split():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < 12, same_net=True), sp)
    assert res["nature"]["label"] == R.BIMODAL_P_READOUT
    assert all(c["auc_apl"] == 0.5 and c["a"] and not c["b"] for c in res["nature"]["cells"])


def test_graded_when_p_sits_in_the_mid_band():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < 12, fire_P=10), sp)
    assert res["nature"]["label"] == R.GRADED and all(c["mid_share"] == pytest.approx(0.7)
                                                      for c in res["nature"]["cells"])


@pytest.mark.parametrize("silent_conds, label", [
    (("on", "kc"), R.READOUT_PATH), (("on", "nonkc"), R.KC_NETWORK_PATH), (("on",), R.BOTH_PATHS),
    (("on", "kc", "nonkc"), R.NEITHER_PATH)])
def test_pathway_labels(silent_conds, label):
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c in silent_conds and i < 12), sp)
    assert res["pathway"]["label"] == label and res["pathway"]["record_all"] == 0.0


def test_a_partial_drop_above_a_quarter_does_not_remove():
    sp = _spec()
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: (c == "on" and i < 12) or (c == "kc" and i < 6)), sp)
    kc = res["pathway"]["arms"]["kc"]
    assert kc["diff_ci95"][1] < 0 and not kc["removes"]           # 0.15 > 0.25 x 0.3: the ratio clause fails
    assert res["pathway"]["label"] == R.READOUT_PATH


def test_drive_labels_per_stimulus():
    sp = _spec()
    up = {(0.25, "4:1"): 12, (1.0, "4:1"): 24, (0.25, "dDL"): 24, (1.0, "dDL"): 12}
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < up[(g, s)]), sp)
    d = res["drive"]
    assert d["4:1"]["label"] == R.INCREASING and d["4:1"]["slope_ci"][0] > 0
    assert d["dDL"]["label"] == R.DECREASING and d["dDL"]["slope"] == pytest.approx(-0.3)
    flat = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < (12 if g == 0.25 else 16)), sp)
    assert {v["label"] for v in flat["drive"].values()} == {R.FLAT}


def test_a_non_monotone_stimulus_is_unclear():
    sp = _spec(g=(0.25, 1.0, 4.0))
    k = {0.25: 8, 1.0: 24, 4.0: 8}
    res = R.o1_judge(_rows(sp, lambda c, g, s, i: c == "on" and i < k[g]), sp)
    assert {v["label"] for v in res["drive"].values()} == {R.NONMONOTONIC_OR_UNCLEAR}
    assert all(v["slope"] == 0.0 and v["range"] == pytest.approx(0.4) for v in res["drive"].values())


def test_records_and_determinism():
    sp = _spec()
    rows = _rows(sp, lambda c, g, s, i: c in ("on", "kc") and i < 12)
    a, b = R.o1_judge(rows, sp), R.o1_judge(list(reversed(rows)), sp)
    assert a == b                                                   # row order and the bootstrap never matter
    assert a["kc_block_record"]["nature"]["label"] == R.BISTABLE_NETWORK
    assert a["sha"] == {c: "sha-" + e for c, e in sp.o1_conditions}
    rec = a["cells"]["on"]["0.25|4:1"]
    assert rec["silent_share"] == pytest.approx(0.3) and rec["apl_out"]["silent"][0] == 1.0
    assert "kc_frac" not in rec and rec["firing_share"] == pytest.approx(0.7)


# ---- added beyond the brief: missing input never yields a label; the g rank never depends on the grid's order --------
def test_a_row_on_the_wrong_edit_is_invalid():
    sp = _spec()
    rows = _rows(sp, lambda c, g, s, i: c == "on" and i < 12)
    swap = [dict(r, edit="apl_to_kc_zero", csc_sha256="sha-apl_to_kc_zero") if r["cond"] == "on" else r for r in rows]
    res = R.o1_judge(swap, sp)
    assert res["outcome"] == R.INVALID and any("does not declare" in x for x in res["reasons"])


@pytest.mark.parametrize("kw", [dict(n=0), dict(g=()), dict(stim=())])
def test_an_empty_declaration_is_invalid_not_a_label(kw):
    sp = _spec(**kw)
    res = R.o1_judge([], sp)
    assert res["outcome"] == R.INVALID and any("empty" in x for x in res["reasons"])


def test_a_repeated_grid_value_raises():
    with pytest.raises(ValueError):
        R.o1_judge([], _spec(g=(0.25, 0.25)))


def test_the_g_rank_follows_g_not_the_grid_order():
    up = {0.25: 12, 1.0: 24}
    a, b = _spec(), _spec(g=(1.0, 0.25))
    ra = R.o1_judge(_rows(a, lambda c, g, s, i: c == "on" and i < up[g]), a)
    rb = R.o1_judge(_rows(b, lambda c, g, s, i: c == "on" and i < up[g]), b)
    assert {v["label"] for v in rb["drive"].values()} == {R.INCREASING}
    assert ra["drive"] == rb["drive"]


def test_the_old_da_judgement_constants_are_gone():
    assert not hasattr(R, "DA_DEPENDENT") and not hasattr(R, "DA_INDEPENDENT")       # O.7.8: a record now
