import dataclasses
import json

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments, export_compartments, validate_populations


def test_populations_from_synthetic(synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    assert len(p.kc) == 40 and len(p.mbon) == 4 and len(p.apl) == 1
    assert set(p.dan_types) == {"PAM08", "PPL105"}
    assert len(p.pam) == 2 and len(p.ppl1) == 2
    assert set(p.receptor_types) == {"ORN_DM1", "ORN_DA1", "ORN_VA2", "ORN_DM6", "ORN_VC1"}
    assert sum(len(v) for v in p.receptor_types.values()) == 20
    assert len(p.sensory) == 20 and len(p.dn) == 6 and len(p.mn) == 4
    # receptor sides: synthetic ORNs are 'M'
    assert len(p.receptor_side["ORN_DM1"]["M"]) == 4


def test_compartments_core_excludes_strays(synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    comps = compartments(c, p, core_frac=0.2)
    pam, ppl = comps["PAM08"], comps["PPL105"]
    assert pam.family == "PAM" and ppl.family == "PPL1"
    # synthetic: PAM08 -> MBON01/02 (60 synapses), stray 3 synapses -> not core
    assert set(c.type[pam.core]) == {"MBON01", "MBON02"}
    assert set(c.type[ppl.core]) == {"MBON03", "MBON04"}
    assert pam.w_mbon.max() == 1.0
    assert (pam.w_mbon[ppl.core] < 0.2).all()


def test_export_compartments(tmp_path, synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    comps = compartments(c, p, 0.2)
    path = tmp_path / "comps.json"
    export_compartments(comps, c, path)
    d = json.loads(path.read_text())
    assert d["PAM08"]["core_mbon_types"] == ["MBON01", "MBON02"]
    assert d["PPL105"]["n_cells"] == 2


def _valid(synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    return c, p, compartments(c, p, 0.2)


def test_validate_populations_accepts_synthetic(synthetic_connectome):
    c, p, comps = _valid(synthetic_connectome)
    assert validate_populations(c, p, comps) is None


def test_validate_populations_rejects_empty_populations(synthetic_connectome):
    c, p, comps = _valid(synthetic_connectome)
    for field, msg in (("kc", "Kenyon"), ("mbon", "MBON"), ("receptor_types", "receptor types")):
        bad = dataclasses.replace(p, **{field: {} if field == "receptor_types" else np.zeros(0, np.int64)})
        with pytest.raises(ValueError, match=msg):
            validate_populations(c, bad, comps)


def test_validate_populations_rejects_missing_dan_type(synthetic_connectome):
    c, p, comps = _valid(synthetic_connectome)
    with pytest.raises(ValueError, match="PAM08"):
        validate_populations(c, p, {k: v for k, v in comps.items() if k != "PAM08"})


def test_validate_populations_rejects_overlapping_cores(synthetic_connectome):
    c, p, comps = _valid(synthetic_connectome)
    overlapping = dict(comps)
    overlapping["PAM08"] = dataclasses.replace(comps["PAM08"], core=comps["PPL105"].core)
    with pytest.raises(ValueError, match="overlap"):
        validate_populations(c, p, overlapping)
