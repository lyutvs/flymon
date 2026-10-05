"""W's seal and one judgement (W.3 10-11, W.5, W.9.8 H8): the seal checks every raw entry's stored inputs against the
units the declared gate pairs, design and seeds give (a changed entry → INVALID, a missing one → NOT_READ, nothing
read), pins the design, the gate pairs and z_V; judge reads only a sealed set under the sealed decision code and the
pinned blocks, once (the marker; one re-generation after the mark; never after a judge block); recompute and
invalid_run only after judge."""
import json
from pathlib import Path

import pytest

from flymon.brain import w_runner as WR
from tests.brain.w_world import SPEC, World, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_seal_invalid_on_an_entry_measured_with_other_inputs(w):
    """An entry whose file still matches its manifest sha but whose stored inputs are not the declared unit's (a
    measurement made with other seeds) is INVALID; a file changed after its block is NOT_READ."""
    from flymon.brain.h3_store import sha256_file
    through(w, "records")
    d = doc()
    m = d["learn"]["manifest"][0]
    e = json.loads(Path(m["cache_file"]).read_text())
    e["inputs"]["probe_seeds"][0] += 1
    Path(m["cache_file"]).write_text(json.dumps(e))
    m["sha256"] = sha256_file(m["cache_file"])
    Path(SPEC.summary).write_text(json.dumps(d))
    out = w.runner().stage_seal()
    assert out["status"] == "INVALID" and any("저장 입력" in r for r in out["reasons"])
    with pytest.raises(SystemExit):
        w.runner().stage_judge()


def test_seal_not_read_on_a_changed_file(w):
    through(w, "records")
    m = doc()["learn"]["manifest"][0]
    Path(m["cache_file"]).write_text(Path(m["cache_file"]).read_text() + " ")
    out = w.runner().stage_seal()
    assert out["status"] == "NOT_READ" and any("sha256" in r for r in out["reasons"])


def test_seal_not_read_on_a_missing_entry(w):
    through(w, "records")
    Path(doc()["band"]["manifest"][0]["cache_file"]).unlink()
    out = w.runner().stage_seal()
    assert out["status"] == "NOT_READ" and out["archive"] is None


def test_judge_once_regeneration_and_pins(w, monkeypatch):
    through(w, "seal")
    real = WR.Runner._read

    def die(self, doc_, mark=None):
        real(self, doc_, mark)
        raise RuntimeError("died after the mark")
    monkeypatch.setattr(WR.Runner, "_read", die)
    with pytest.raises(RuntimeError):
        w.runner().stage_judge()
    assert Path(WR.JUDGE_MARKER).exists() and "judge" not in doc()
    monkeypatch.setattr(WR.Runner, "_read", real)
    out = w.runner().stage_judge()
    assert out["resumed_after_mark"] is True and out["verdict"] == "PASS"
    d = doc()
    d.pop("judge")
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        w.runner().stage_judge()                                 # DONE marker: never a second block


def test_judge_refuses_moved_pins_and_decision_code(w, monkeypatch):
    through(w, "seal")
    d = doc()
    d["gates"]["gates"] = d["gates"]["gates"][::-1]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        w.runner().stage_judge()
    d["gates"]["gates"] = d["gates"]["gates"][::-1]
    Path(SPEC.summary).write_text(json.dumps(d))
    monkeypatch.setattr(WR, "decision_key", lambda: dict(key="other", files={}))
    with pytest.raises(SystemExit):
        w.runner().stage_judge()


def test_after_judge(w):
    with pytest.raises(SystemExit):
        w.runner().stage_recompute("x")
    through(w, "judge")
    e = w.runner().stage_recompute("note")
    assert e["verdict"] == "PASS" and e["differs_from_judge"] is False
    out = w.runner().stage_invalid_run("measurement defect")
    assert out["status"] == "INVALID_RUN" and "같은 관문 쌍으로 다시 돌리지 않는다" in out["rule"]
    with pytest.raises(SystemExit):
        w.runner().stage_recompute("again")


def test_keys_rechecked_before_measurement_seal_and_judge(w):
    through(w, "gates")
    with pytest.raises(SystemExit) as e:
        w.runner(wcode={"key": "z" * 64}).stage_learn()
    assert e.value.code == 7
