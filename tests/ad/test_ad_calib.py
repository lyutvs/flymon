"""AD.5 3: the KC-activity calibration of FLY-OS on the naive brain. Requests (FLY + the 16 OS odours per situation,
one decision seed per (rep, pair, side)), the per-candidate KC counts, the measured ratios, and the pre-registered
band 1 +- max(0.15, 3 SD_R) applied after the measurement is committed."""
import numpy as np
import pytest

from flymon.ad import calib, os_encoder
from flymon.ad.spec import SPEC
from flymon.agent.policy import derive_seed
from tests.ac.ac_fakes import MI, ST, stub_grid

PAIRS = [dict(me="Lapras", o1="Charizard", o2="Exeggutor", cands=["Surf", "Psychic", "Thunderbolt"])]


def test_requests_share_one_seed_per_situation_and_cover_16_species():
    meta, reqs = calib.requests(stub_grid(), PAIRS, 1, 305)
    assert len(reqs) == 2 * 17 and [m["variant"] for m in meta[:17]] == ["fly"] + list(os_encoder.SPECIES)
    assert {s for _, _, s in reqs[:17]} == {derive_seed(305, "os-calib", 1, 0, 0)}
    assert reqs[0][1] == [os_encoder.odour_for_species(stub_grid(), MI[m][0], "Charizard") for m in PAIRS[0]["cands"]]
    assert meta[17]["side"] == 1 and meta[17]["true"] == "Exeggutor"


def test_kc_rows_count_active_cells_after_the_readout():
    meta = [dict(rep=0, pair=0, side=0, variant="fly", true="Charizard", cands=["Surf", "Psychic"])]
    counts = [np.array([[9, 9, 1, 0, 2], [9, 9, 0, 0, 0]])]
    rows = calib.kc_rows(meta, counts, n_skip=2)
    assert rows[0]["kc"] == [2, 0]


def rows_for(ratio, reps=(0, 1, 2), jitter=0.0):
    out = []
    for r in reps:
        k = ratio + jitter * (r - 1)
        out.append(dict(rep=r, pair=0, side=0, variant="fly", true="Charizard", cands=["Surf", "Psychic"],
                        kc=[100, 100]))
        for sp in os_encoder.SPECIES:
            out.append(dict(rep=r, pair=0, side=0, variant=sp, true="Charizard", cands=["Surf", "Psychic"],
                            kc=[round(100 * k), round(100 * k)]))
    return out


def test_measure_and_band():
    m = calib.measure(rows_for(1.1, jitter=0.0), MI, ST)
    assert m["R_by_rep"] == [1.1, 1.1, 1.1] and m["R"] == pytest.approx(1.1) and m["sd_R"] == 0.0
    assert set(m["by_move_type"]) == {MI["Surf"][0], MI["Psychic"][0]} and "single" in m["by_true_dual"]
    assert set(m["percentiles"]) == {"fly", "os"} and m["ks"] == pytest.approx(1.0)
    b = calib.band(m, SPEC)
    assert b["status"] == "OK" and b["half_width"] == 0.15
    assert calib.band(calib.measure(rows_for(0.27), MI, ST), SPEC)["status"] == "STOP_BALANCE"   # FLY-TB's ratio
    wide = calib.measure(rows_for(1.2, jitter=0.1), MI, ST)
    assert wide["sd_R"] == pytest.approx(0.1) and calib.band(wide, SPEC)["half_width"] == pytest.approx(0.3)
    assert calib.band(wide, SPEC)["status"] == "OK"


def test_ks_distance():
    assert calib.ks([1, 2, 3], [1, 2, 3]) == 0.0
    assert calib.ks([1, 1, 1], [5, 5, 5]) == 1.0
