"""Y.7 11 · 12: the seal (stored inputs = declared, file count, decision key + pins, the order-5 / order-6
environment check, the archive copy; NOT_SEALED otherwise) and the one judgement (W's pair verdict, X's set rule at
the design's p_set, b = 0, read / done markers, one regeneration after the mark, Y.8's sentences)."""
import json
from pathlib import Path

import pytest

from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain.y_spec import SPEC as Y
from tests.brain.y_world_b import World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_judge_pass_once(w):
    out = through(w, "judge")
    assert out["verdict"] == R.PASS and (out["n"], out["m"], out["b"]) == (8, 8, 0) and out["p_set"] == 1.0
    assert out["sentence"].startswith("조합 지렛대 모델(C3") and "첫 8 관문 쌍 중 전부에서 섰다" in out["sentence"]
    assert out["consequence"] == R.CONSEQUENCE[R.PASS] and out["main_set"].startswith("사용됨")
    assert out["oc_records"]["records"]["target"] == {k: doc()["oc"]["design"][k] for k in ("p_set", "q", "K", "F")}
    assert Path(Y.done_marker).exists() and Path(Y.judge_marker).exists()
    with pytest.raises(SystemExit):
        w.runner().stage_judge()
    det = json.loads(Path(Y.judge_detail).read_text())
    assert set(det) == {"judgement", "records", "set"} and det["records"]["C"]


def test_seal_records_decision_env_and_archive(w):
    through(w, "records")
    out = w.runner().stage_seal()
    assert out["outcome"] == R.PASS and out["n_files"] == out["n_declared"] == 8 * 8 + 192 + 192 + 196
    assert set(out["decision"]) == {"key", "files", "oc_sha256", "gates_sha256", "z_V"}
    assert "flymon/brain/y_runner.py" in out["decision"]["files"] and "flymon/brain/x_oc.py" in out["decision"]["files"]
    assert out["env_mismatch"] == [] and len(out["archive"]) == 1 + 4 + out["n_files"]


def test_seal_refuses_to_seal_on_an_environment_change(w, monkeypatch):
    through(w, "records")
    real = YR.env_hashes
    monkeypatch.setattr(YR, "env_hashes", lambda: dict(real(), numpy="0.0"))
    out = w.runner().stage_seal()
    assert out["outcome"] == R.NOT_SEALED and "순서 5 대비 numpy" in out["env_mismatch"] and "archive" not in out
    with pytest.raises(SystemExit):
        w.runner().stage_judge()


def test_seal_refuses_to_seal_on_changed_raw(w):
    through(w, "records")
    f = Path(doc()["learn"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"fly": 0', '"fly": 0 ', 1))
    out = w.runner().stage_seal()
    assert out["outcome"] == R.NOT_SEALED and any("learn:" in r for r in out["reasons"])


def test_judge_fail_names_the_failing_pairs(w):
    through(w, "gates")
    k0 = doc()["gates"]["gates"][0]["key"]
    w.model.effects[k0] = (30.0, 0.0)                     # no punishment effect on the first gate pair
    for s in ("estimate", "learn", "band", "records", "seal"):
        getattr(w.runner(), f"stage_{s}")()
    out = w.runner().stage_judge()
    assert out["verdict"] == R.FAIL and out["m"] == 7 and out["pairs"][k0]["status"] == "FAIL"
    assert "m/n 0.875 < p_set 1.0" in out["sentence"] and k0 in out["sentence"]


def test_judge_stop_machine_on_rn_sharing(w):
    through(w, "estimate")
    w.model.share_state = True
    for s in ("learn", "band", "records", "seal"):
        getattr(w.runner(), f"stage_{s}")()
    out = w.runner().stage_judge()
    assert out["verdict"] == R.STOP_MACHINE and "RN1" in out["sentence"]


def test_judge_refuses_when_a_pinned_block_moved(w):
    through(w, "seal")
    d = doc()
    d["gates"]["note"] = "edited"
    Path(Y.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and not Path(Y.judge_marker).exists()


def test_judge_regenerates_once_after_the_mark(w, monkeypatch):
    through(w, "seal")
    real = YR.y_rules.set_verdict
    monkeypatch.setattr(YR.y_rules, "set_verdict", lambda *a: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        w.runner().stage_judge()
    assert Path(Y.judge_marker).exists() and "judge" not in doc()
    monkeypatch.setattr(YR.y_rules, "set_verdict", real)
    out = w.runner().stage_judge()
    assert out["resumed_after_mark"] is True and Path(Y.reread_marker).exists()
