"""Spec N.2 / N.8.2: the hand-computed 53-channel fixture (Or33b duplicated to DM5 and DM3, receptors on one glomerulus
added), linear mixtures, c_δ on δ-DL only, clipping recorded, s = drive_hz / (max_rate_hz x strength), the Lin totals
as a record, and DataMismatch on every data defect."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import odor_real as O
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC
from flymon.brain.stimuli import present

ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPES = [
    "ORN_D", "ORN_DA1", "ORN_DA2", "ORN_DA3", "ORN_DA4l", "ORN_DA4m", "ORN_DC1", "ORN_DC2", "ORN_DC3", "ORN_DC4",
    "ORN_DL1", "ORN_DL2d", "ORN_DL2v", "ORN_DL3", "ORN_DL4", "ORN_DL5", "ORN_DM1", "ORN_DM2", "ORN_DM3", "ORN_DM4",
    "ORN_DM5", "ORN_DM6", "ORN_DP1l", "ORN_DP1m", "ORN_V", "ORN_VA1d", "ORN_VA1v", "ORN_VA2", "ORN_VA3", "ORN_VA4",
    "ORN_VA5", "ORN_VA6", "ORN_VA7l", "ORN_VA7m", "ORN_VC1", "ORN_VC2", "ORN_VC3", "ORN_VC4", "ORN_VC5", "ORN_VL1",
    "ORN_VL2a", "ORN_VL2p", "ORN_VM1", "ORN_VM2", "ORN_VM3", "ORN_VM4", "ORN_VM5d", "ORN_VM5v", "ORN_VM6l",
    "ORN_VM6m", "ORN_VM6v", "ORN_VM7d", "ORN_VM7v"]
# ΣΔ per glomerulus (IA, EB, δ-DL), summed by hand from the committed table: DM5 = Or85a + Or33b, DM3 = Or47a + Or33b
NONZERO = {
    "ORN_DA3": (30, 24, 7), "ORN_DA4l": (38, 43, 5), "ORN_DA4m": (73, 20, 8), "ORN_DC1": (110, 102, 6),
    "ORN_DL1": (222, 43, 3), "ORN_DL3": (28, 10, 13), "ORN_DL4": (34, 33, 6), "ORN_DL5": (-16, 18, 2),
    "ORN_DM2": (236, 197, 15), "ORN_DM3": (154, 137, 41), "ORN_DM4": (48, 41, 4), "ORN_DM5": (29, 201, 55),
    "ORN_DM6": (140, 178, 13), "ORN_VA1d": (25, 17, 25), "ORN_VA1v": (9, 31, 43), "ORN_VA5": (19, 5, 4),
    "ORN_VA6": (55, 43, 11), "ORN_VC3": (39, 138, 7), "ORN_VC4": (54, 118, 6), "ORN_VM2": (98, 201, 4),
    "ORN_VM3": (142, 124, 1), "ORN_VM5d": (244, 115, 37), "ORN_VM5v": (260, 93, 20)}
CAP = 1000.0 / 3.0


@pytest.fixture
def glom():
    return O.glomerular(O.load_table(ROOT / SPEC.data_dir, SPEC.sha_pins()), MODEL_TYPES)


def test_the_53_channel_vector_is_the_hand_computed_fixture(glom):
    assert list(glom) == MODEL_TYPES and len(glom) == 53
    for t in MODEL_TYPES:
        assert tuple(glom[t][k] for k in O.ODORANTS) == pytest.approx(NONZERO.get(t, (0, 0, 0)))
    assert len(NONZERO) == 23                                   # 24 receptors, DM5 and DM3 shared, Or33b on both


def test_or33b_is_duplicated_not_split(glom):
    assert glom["ORN_DM5"]["EB"] == 139 + 62 and glom["ORN_DM3"]["EB"] == 75 + 62


def test_mixture_drive_clip_and_strength_at_g1(glom):
    st = O.stimuli(glom, ("4:1", "1:4", "IA"), 1.0, 1.0, SPEC.mixtures_dict(), 200.0, 0.35, CAP)
    assert st["4:1"]["drive_hz"]["ORN_DM2"] == pytest.approx(0.8 * 236 + 0.2 * 197)       # 228.2
    assert st["4:1"]["drive_hz"]["ORN_DL5"] == 0.0
    assert st["4:1"]["clipped_hz"] == {"ORN_DL5": pytest.approx(9.2)}                      # 0.8 * -16 + 0.2 * 18
    assert st["1:4"]["drive_hz"]["ORN_DL5"] == pytest.approx(11.2) and st["1:4"]["clipped_hz"] == {}
    assert st["IA"]["clipped_hz"] == {"ORN_DL5": pytest.approx(16.0)}     # DM5's IA is -2 + 31 = 29 > 0: not clipped
    assert st["4:1"]["odor"]["ORN_DM2"] == pytest.approx(228.2 / (200.0 * 0.35))
    assert set(st["4:1"]["odor"]) == set(MODEL_TYPES)
    assert all(v == 0.0 for t, v in st["4:1"]["drive_hz"].items() if t not in NONZERO)


def test_c_delta_multiplies_only_ddl_and_g_scales_everything(glom):
    st = O.stimuli(glom, ("dDL", "4:1"), 0.5, 2.0, SPEC.mixtures_dict(), 200.0, 0.35, CAP)
    assert st["dDL"]["drive_hz"]["ORN_DM5"] == pytest.approx(55 * 2.0 * 0.5)
    assert st["4:1"]["drive_hz"]["ORN_DM2"] == pytest.approx(228.2 * 0.5)
    assert st["dDL"]["weights"] == {"dDL": 2.0} and st["4:1"]["weights"] == {"IA": 0.8, "EB": 0.2}


def test_channels_above_the_refractory_cap_are_listed(glom):
    assert O.cap_hz(Params()) == pytest.approx(1000.0 / 3.0)
    st = O.stimuli(glom, ("IA",), 4.0, 1.0, SPEC.mixtures_dict(), 200.0, 0.35, CAP)
    assert "ORN_DM2" in st["IA"]["capped"] and "ORN_DA3" not in st["IA"]["capped"]     # 944 Hz vs 120 Hz


def test_present_commands_drive_hz_on_every_cell_of_the_type(glom, synthetic_connectome):
    from flymon.brain.circuits import Populations
    pops = Populations.from_connectome(synthetic_connectome())

    class Eng:
        p = Params()
        drive_hz = np.zeros(200, np.float32)

    small = {"ORN_DM1": {"IA": 100.0, "EB": 0.0, "dDL": 0.0}, "ORN_DA1": {"IA": -5.0, "EB": 0.0, "dDL": 0.0}}
    st = O.stimuli(small, ("IA",), 1.0, 1.0, SPEC.mixtures_dict(), Eng.p.max_rate_hz, 0.35, CAP)
    present(Eng, pops, st["IA"]["odor"], 0.35)
    assert np.allclose(Eng.drive_hz[pops.receptor_types["ORN_DM1"]], 100.0)
    assert np.all(Eng.drive_hz[pops.receptor_types["ORN_DA1"]] == 0.0)


def test_lin_totals_are_recorded_in_both_conventions():
    rec = O.lin_record(O.load_table(ROOT / SPEC.data_dir, SPEC.sha_pins()), dict(SPEC.lin_totals), SPEC.lin_tol)
    assert rec["totals"]["signed"] == {"IA": 2040.0, "EB": 1870.0, "dDL": 296.0}
    assert rec["totals"]["positive"] == {"IA": 2058.0, "EB": 1870.0, "dDL": 296.0}
    assert rec["rel_error"]["signed"]["dDL"] == pytest.approx(10 / 286)
    assert rec["within_tol"] == {"signed": False, "positive": False}   # δ-DL is +3.5%: a record, not a stop (N.8.2)


def _copy(tmp_path):
    d = tmp_path / "odor"
    shutil.copytree(ROOT / SPEC.data_dir, d)
    return d


def test_a_changed_file_is_a_data_mismatch(tmp_path):
    d = _copy(tmp_path)
    (d / O.TABLE_FILE).write_text((d / O.TABLE_FILE).read_text().replace("Or22a,236", "Or22a,237"))
    with pytest.raises(O.DataMismatch, match="sha256"):
        O.load_table(d, SPEC.sha_pins())


@pytest.mark.parametrize("edit, match", [
    (("t", "Or22a,236,197,15", "Or22a,NA,197,15"), "Or22a"),
    (("t", "Or22a,236,197,15", "Or22a,236,,15"), "Or22a"),
    (("m", "Or22a,DM2", "Or22a,?"), "unmapped"),
    (("m", "Or22a,DM2\n", ""), "without a mapping"),
    (("t", "Or9a,142,124,1\n", "Or9a,142,124,1\nOr9a,1,1,1\n"), "twice"),
])
def test_data_defects_are_data_mismatches(tmp_path, edit, match):
    d = _copy(tmp_path)
    f = d / (O.TABLE_FILE if edit[0] == "t" else O.MAP_FILE)
    f.write_text(f.read_text().replace(edit[1], edit[2]))
    with pytest.raises(O.DataMismatch, match=match):
        O.load_table(d, None)


def test_a_glomerulus_the_model_lacks_is_a_data_mismatch():
    t = O.load_table(ROOT / SPEC.data_dir, SPEC.sha_pins())
    with pytest.raises(O.DataMismatch, match="DM3"):
        O.glomerular(t, [x for x in MODEL_TYPES if x != "ORN_DM3"])


@pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="needs the connectome")
def test_model_types_are_the_real_models():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    types = sorted(str(t) for t in Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz")).receptor_types)
    assert types == MODEL_TYPES
