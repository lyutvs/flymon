# tests/brain/test_q_pairs.py
"""Q.3 / Q.6.4 / Q.6.8: the k2-norm codebook is the committed one (digest), only H.4's even turns 0-14 are accepted (the
L judgement set never), the Q0 file of each (b) pair is the manifest's basename under results/q/q0_cache with its
sha256 checked, and the largest cap-limited 0.05-grid s is a record only (Q.6.9)."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from flymon.brain import q_pairs as Q
from flymon.brain.q_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
ENC = json.loads((ROOT / "results/summary/encoder_grid.json").read_text())
NPZ = ROOT / "data/malecns.npz"


def test_codebook_is_the_committed_k2_norm_digest():
    cb = Q.codebook(ENC, SPEC)
    assert cb.k == 2 and len(cb.words) == 96
    bad = copy.deepcopy(ENC)
    bad["codebook"]["k"]["2"]["codebook"][0] = list(reversed(bad["codebook"]["k"]["2"]["codebook"][1]))
    with pytest.raises(ValueError, match="digest"):
        Q.codebook(bad, SPEC)


def test_strength_is_s_1():
    Q.check_strength(ENC, SPEC)
    bad = copy.deepcopy(ENC)
    bad["strength"]["configs"]["k2-norm"]["s"] = 1.4
    with pytest.raises(ValueError):
        Q.check_strength(bad, SPEC)


@pytest.mark.parametrize("turn", [64, 103, 16, 3])
def test_turn_guard_refuses_anything_but_h4_even_turns(turn):
    rows = [dict(axis="b", turn=0, x="a", y="b"), dict(axis="b", turn=turn, x="c", y="d")]
    with pytest.raises(ValueError, match="judgement set"):
        Q.check_turns(rows, SPEC)


def test_manifest_has_the_21_b_pairs():
    m = Q.manifest(ENC, SPEC)
    assert len(m) == 21 and all(k.startswith("b|") for k in m)
    assert all(int(k.split("|")[1]) % 2 == 0 for k in m)


def test_q0_files_check_the_declared_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    enc = copy.deepcopy(ENC)
    m = [e for e in enc["even"]["configs"]["k2-norm"]["manifest"] if e["axis"] == "b"]
    d = tmp_path / SPEC.q0_cache_dir
    d.mkdir(parents=True)
    for e in m:
        (d / Path(e["cache_file"]).name).write_text('{"kind": "oracle", "result": {}}')
    good = m[0]
    blob = (d / Path(good["cache_file"]).name).read_bytes()
    good["cache_sha256"] = hashlib.sha256(blob).hexdigest()
    rows = [dict(axis=e["axis"], turn=e["turn"], x=e["x"], y=e["y"]) for e in m]
    files = Q.q0_files(rows, enc, SPEC)
    assert files[0]["ok"] and not any(f["ok"] for f in files[1:])
    assert all(f["path"].startswith(SPEC.q0_cache_dir + "/") for f in files)
    assert "results/encoder" not in json.dumps(files)


def test_cap_s_grid_and_weak_rule():
    odours = [{"g1": 1.25, "g2": 0.75}, {"g3": 0.5}]
    got = Q.cap_s(odours, 200.0, 1000.0 / 3.0, SPEC)          # s * 1.25 <= 1.6667 -> 1.3
    assert got["s"] == 1.3 and got["weak"] is False and got["vmax"] == 1.25
    got = Q.cap_s([{"g": 1.6}], 200.0, 1000.0 / 3.0, SPEC)     # s <= 1.0417 -> 1.0, weak
    assert got["s"] == 1.0 and got["weak"] is True
    got = Q.cap_s([{"g": 1.5}], 200.0, 1000.0 / 3.0, SPEC)     # s <= 1.1111 -> 1.1, weak (0.1 < 0.15)
    assert got["s"] == 1.1 and got["weak"] is True
    with pytest.raises(ValueError, match="cap"):
        Q.cap_s([{"g": 2.0}], 200.0, 1000.0 / 3.0, SPEC)        # s 1.0 itself is over the cap


@pytest.mark.skipif(not NPZ.exists(), reason="needs data/malecns.npz")
def test_real_rows_are_the_encoder_odours():
    from flymon.agent import e_pairs
    from flymon.agent.config import load_c3_config
    from flymon.agent.e_runner import MEASURE_FILES_E
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_store import code_key
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    rows = Q.b_rows(e_pairs.even_situations(pops), rc, ENC, SPEC)
    assert len(rows) == 21 and all(r["axis"] == "b" and r["turn"] % 2 == 0 and r["turn"] < 16 for r in rows)
    assert [Q.key_str(r) for r in rows] == list(Q.manifest(ENC, SPEC))
    code = code_key(str(NPZ), files=MEASURE_FILES_E)
    if code["key"] != ENC["even"]["code_key"]:
        pytest.skip("encoder code key moved; the reproduction gate covers it")
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    m0d = json.loads((ROOT / SPEC.m0d_summary).read_text())
    types = m0d["h4"]["pools"]["A"] + m0d["h4"]["pools"]["P"]
    assert all(Q.encoder_key_match(rows, ENC, SPEC, code, cfg.params, cfg.readout, cfg.z, types, len(pops.kc)))
