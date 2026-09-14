import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flymon.brain.connectome import Connectome
from flymon.brain.data_build import build

RAW = Path("data/raw")
FILES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
]


def _write_fake_dataset(d: Path, dictionary: bool = False):
    d.mkdir(parents=True)

    def s(values):
        a = pa.array(values, pa.string())
        return a.dictionary_encode() if dictionary else a

    ann = pa.table({
        "bodyId": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "type": s(["ORN_DM1", "KCab-m", "MBON01", None, "lLN1_a"]),
        "class": s(["olfactory", "Kenyon_Cell", "MBON", None, "ALLN"]),
        "superclass": s(["cb_sensory", "cb_intrinsic", "cb_intrinsic", None, "cb_intrinsic"]),
        "somaSide": s(["M", "L", "R", "L", "L"]),
        "rootSide": s(["L", None, None, None, None]),
        "status": s(["Traced", "Traced", "Traced", "Traced", "Orphan"]),
    })
    feather.write_feather(ann, d / FILES[1])
    nt = pa.table({"body": pa.array([1, 2, 3, 3], pa.int64()),
                   "consensus_nt": s(["acetylcholine", "acetylcholine", "glutamate", "glutamate"])})
    feather.write_feather(nt, d / FILES[2])
    wt = pa.table({"body_pre": pa.array([1, 2, 2, 3, 9], pa.int64()),
                   "body_post": pa.array([2, 3, 3, 1, 1], pa.int64()),
                   "weight": pa.array([12, 7, 2, 1, 5], pa.int64())})
    # write as IPC file so record-batch streaming works
    with pa.OSFile(str(d / FILES[0]), "wb") as sink:
        with pa.ipc.new_file(sink, wt.schema) as writer:
            writer.write_table(wt, max_chunksize=2)


def test_build_from_fake_feathers(tmp_path):
    raw = tmp_path / "raw"
    _write_fake_dataset(raw)
    out = tmp_path / "c.npz"
    manifest = build(raw, out, min_weight=1)
    c = Connectome.load(out)
    # neuron 4 (no type) and neuron 5 (not Traced) dropped
    assert c.N == 3 and list(c.bodyId) == [1, 2, 3]
    assert list(c.type) == ["ORN_DM1", "KCab-m", "MBON01"]
    assert list(c.sign) == [1, 1, -1]
    assert list(c.side) == ["L", "L", "R"]           # ORN side from rootSide
    # edges: 1->2 (12), 2->3 (7), 2->3 (2), 3->1 (1); 9->1 dropped (unknown body)
    assert c.E == 4
    assert manifest["n_neurons"] == 3 and manifest["n_edges"] == 4
    assert set(manifest["inputs"]) == set(FILES)
    assert all(len(v["sha256"]) == 64 for v in manifest["inputs"].values())
    assert json.loads((out.with_suffix(".manifest.json")).read_text())["n_edges"] == 4


def test_build_from_dictionary_encoded_feathers(tmp_path):
    """Janelia dictionary-encodes string columns; pandas hands those back as Categorical."""
    raw = tmp_path / "raw"
    _write_fake_dataset(raw, dictionary=True)
    out = tmp_path / "c.npz"
    manifest = build(raw, out, min_weight=1)
    c = Connectome.load(out)
    assert c.N == 3 and list(c.bodyId) == [1, 2, 3]
    assert list(c.type) == ["ORN_DM1", "KCab-m", "MBON01"]
    assert list(c.cls) == ["olfactory", "Kenyon_Cell", "MBON"]
    assert list(c.sign) == [1, 1, -1]
    assert list(c.side) == ["L", "L", "R"]
    assert c.E == 4 and manifest["n_edges"] == 4


@pytest.mark.skipif(not all((RAW / f).exists() for f in FILES), reason="MaleCNS raw files not downloaded")
def test_build_real_dataset_counts(tmp_path):
    out = tmp_path / "malecns.npz"
    m = build(RAW, out, min_weight=1)
    assert 160_000 <= m["n_neurons"] <= 170_000
    assert m["counts"]["Kenyon_Cell"] == 4064
    assert m["counts"]["MBON"] == 97
    assert m["counts"]["APL"] == 2
    assert m["counts"]["lLN"] == 151
