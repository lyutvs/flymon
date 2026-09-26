"""Spec K.8.3: N, S, D13 and the records, on synthetic masks and on H.4a.8's recorded even-turn masks."""
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.k_metrics import (engine_metrics, kc_kc_edges, kc_kc_input, lobe_masks, pair_metrics,
                                    readout_weights, removed_fraction)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/m0d/diag/h4_specificity_ceiling.json"
SUM = ROOT / "results/m0d/diag/h4_specificity_ceiling_summary.json"


def test_pair_metrics_on_a_known_mask():
    fx = np.array([1.0, 0.5, 0.25, 0.0])
    fy = np.array([0.0, 0.5, 0.0, 1.0])
    w13 = np.array([2.0, 4.0, 8.0, 16.0])
    w05 = np.array([1.0, 1.0, 1.0, 1.0])
    m = pair_metrics(fx, fy, w13, w05)
    assert m["N"] == 2.0 * 1.0 + 8.0 * 0.25                 # KCs 0 and 2 are X-only
    assert m["D13"] == 2.0 + 4.0 * 0.5 + 8.0 * 0.25
    assert m["S"] == pytest.approx(4.0 / 6.0)
    assert m["N05"] == 1.0 + 0.25


def test_no_drive_gives_zero_not_nan():
    z = np.zeros(3)
    m = pair_metrics(z, z, np.ones(3), np.ones(3))
    assert (m["N"], m["D13"], m["S"], m["N05"]) == (0.0, 0.0, 0.0, 0.0)


def test_engine_metrics_sums_n_keeps_every_pair_and_records():
    lobes = {"apbp": np.array([True, True, False, False]), "g": np.array([False, False, True, True]),
             "ab": np.zeros(4, bool)}
    w = np.array([1.0, 1.0, 1.0, 1.0])
    acts = [dict(fx=np.array([1.0, 0.0, 0.0, 0.0]), fy=np.zeros(4), cx=np.array([2.0, 0, 0, 0])),
            dict(fx=np.zeros(4), fy=np.zeros(4), cx=np.zeros(4)),                  # no drive: N 0, S 0, kept
            dict(fx=np.array([0.125, 1.0, 0, 0]), fy=np.array([1.0, 0, 0, 0]), cx=np.array([1.0, 3.0, 0, 0]))]
    m = engine_metrics(acts, w, w, lobes)
    assert m["N_p"] == [1.0, 0.0, 1.0] and m["N"] == 2.0 and len(m["S"]) == 3 and m["S"][1] == 0.0
    assert m["S_median"] == pytest.approx(float(np.median([1.0, 0.0, 1.0 / 1.125])))
    assert m["lobes"]["apbp"] == pytest.approx(np.mean([0.5, 0.0, 1.0]))
    assert m["reliability"]["n_active"] == 3 and m["reliability"]["hist"]["8"] == 2 and m["reliability"]["hist"]["1"] == 1
    assert m["jaccard_median"] == pytest.approx(float(np.median([0.0, 0.0, 0.5])))


def test_kc_kc_input_is_the_spike_count_proxy():
    edges = (np.array([0, 1]), np.array([2, 2]), np.array([10.0, 20.0]))
    got = kc_kc_input(np.array([1.0, 2.0, 0.0]), edges, -0.5, 0.275)
    assert got == pytest.approx([0.0, 0.0, 0.5 * 0.275 * (1.0 * 10 + 2.0 * 20)])


def test_connectome_helpers_on_the_synthetic_connectome(synthetic_connectome):
    from flymon.brain.circuits import Populations
    conn = synthetic_connectome()
    pops = Populations.from_connectome(conn)
    w = readout_weights(conn, pops, "MBON01", 5)
    assert w.shape == (len(pops.kc),) and (w >= 0).all()
    lobes = lobe_masks(conn, pops)
    assert lobes["ab"].all() and not lobes["g"].any()       # the synthetic KCs are all "KCab-m"
    pre, post, ww = kc_kc_edges(conn, pops, 5)
    assert pre.size == post.size == ww.size == 0             # no KC->KC edges in the synthetic build


@pytest.mark.skipif(not (RAW.exists() and SUM.exists()), reason="H.4a.8 raw records are git-excluded")
def test_removed_fraction_reproduces_h4a8_on_the_recorded_even_masks():
    """K.8.7 (ii): the mask measure on already-used data (not selection) equals H.4a.8's recorded values."""
    raw, summ = json.loads(RAW.read_text()), json.loads(SUM.read_text())
    rows = [r for r in raw["rows"]["C3"] if r["mode"] == "x_only"]
    rec = {(p["axis"], p["turn"], p["x"], p["y"]): p["mask"]["removed"] for p in summ["engines"]["C3"]["pairs"]}
    got = {(r["axis"], r["turn"], r["x"], r["y"]): removed_fraction(np.array(r["fx"]), np.array(r["fy"])) for r in rows}
    assert set(got) == set(rec)
    assert all(abs(got[k] - rec[k]) < 1e-9 for k in got)
    b = [got[k] for k in got if k[0] == "b"]
    assert abs(float(np.median(b)) - summ["engines"]["C3"]["groups"]["b"]["removed"]) < 1e-9
