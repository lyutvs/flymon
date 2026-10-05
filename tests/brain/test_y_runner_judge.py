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


# ---- judge once ------------------------------------------------------------------------------------------------------
def _crash(monkeypatch):
    monkeypatch.setattr(YR.y_rules, "set_verdict", lambda *a: (_ for _ in ()).throw(RuntimeError("crash")))


def test_judge_third_attempt_after_a_regenerated_crash_refuses(w, monkeypatch):
    through(w, "seal")
    real = YR.y_rules.set_verdict
    _crash(monkeypatch)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            w.runner().stage_judge()
    assert Path(Y.judge_marker).exists() and Path(Y.reread_marker).exists() and "judge" not in doc()
    monkeypatch.setattr(YR.y_rules, "set_verdict", real)
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(Y.done_marker).exists()


def test_judge_refuses_with_the_done_marker_and_no_block(w):
    through(w, "judge")
    d = doc()
    del d["judge"]
    Path(Y.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(Y.reread_marker).exists()


def test_judge_refuses_when_git_history_holds_a_judge_block(w):
    through(w, "seal")
    w.judge_commits = ["x"]
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and not Path(Y.judge_marker).exists() and "judge" not in doc()


@pytest.mark.parametrize("field", ["seal_written_at", "decision_key"])
def test_judge_refuses_a_marker_of_another_seal(w, monkeypatch, field):
    through(w, "seal")
    real = YR.y_rules.set_verdict
    _crash(monkeypatch)
    with pytest.raises(RuntimeError):
        w.runner().stage_judge()
    mk = json.loads(Path(Y.judge_marker).read_text())
    mk[field] = "other"
    Path(Y.judge_marker).write_text(json.dumps(mk))
    monkeypatch.setattr(YR.y_rules, "set_verdict", real)
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and not Path(Y.reread_marker).exists() and "judge" not in doc()


# ---- b = 0 -------------------------------------------------------------------------------------------------------------
def test_judge_refuses_b_nonzero_as_invalid_run(w, monkeypatch, capsys):
    through(w, "seal")
    real = YR.y_rules.set_verdict
    monkeypatch.setattr(YR.y_rules, "set_verdict", lambda *a: dict(real(*a), b=1))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(Y.done_marker).exists()
    err = capsys.readouterr().err
    assert "b = 1" in err and "INVALID_RUN" in err and "resume band" not in err


# ---- seal --------------------------------------------------------------------------------------------------------------
def _patch_want(monkeypatch, edit):
    real = YR.w_runner.Runner._want

    def want(self, view):
        return edit(dict(real(self, view)))
    monkeypatch.setattr(YR.w_runner.Runner, "_want", want)


def test_seal_refuses_on_a_file_count_mismatch(w, monkeypatch):
    through(w, "records")
    _patch_want(monkeypatch, lambda d: {**d, ("learn", "extra|0|C"): {"x": 1}})
    out = w.runner().stage_seal()
    n = 8 * 8 + 192 + 192 + 196
    assert out["outcome"] == R.NOT_SEALED and out["invalid"] == [] and out["env_mismatch"] == []
    assert out["reasons"] == [f"원자료 {n}개 ≠ 선언 {n + 1}개"] and "archive" not in out


def test_seal_refuses_when_stored_inputs_differ_from_declared(w, monkeypatch):
    through(w, "records")

    def edit(d):
        k = sorted(k for k in d if k[0] == "band")[0]
        d[k] = dict(d[k], edited=True)
        return d
    _patch_want(monkeypatch, edit)
    out = w.runner().stage_seal()
    assert out["outcome"] == R.NOT_SEALED and len(out["invalid"]) == 1 and out["invalid"][0].startswith("band: ")
    assert out["reasons"] == out["invalid"] and "archive" not in out


def test_seal_refuses_a_y_files_change_against_order_6(w, monkeypatch):
    through(w, "records")
    real = YR.env_hashes

    def env():
        e = real()
        f = sorted(e["y_files"])[0]
        return dict(e, y_files=dict(e["y_files"], **{f: "0" * 64}))
    f0 = sorted(real()["y_files"])[0]
    monkeypatch.setattr(YR, "env_hashes", env)
    out = w.runner().stage_seal()
    assert out["outcome"] == R.NOT_SEALED and out["env_mismatch"] == [f"순서 6 대비 y_files:{f0}"]


def test_seal_strips_a_pre_only_y_files_change(w):
    through(w, "records")
    d = doc()
    f0 = sorted(d["precheck"]["env"]["y_files"])[0]
    d["precheck"]["env"]["y_files"][f0] = "0" * 64
    Path(Y.summary).write_text(json.dumps(d))
    out = w.runner().stage_seal()
    assert out["outcome"] == R.PASS and out["env_mismatch"] == []


@pytest.mark.parametrize("edit", ["oc", "z_V"])
def test_judge_refuses_an_edited_pin_after_the_seal(w, edit):
    through(w, "seal")
    d = doc()
    if edit == "oc":
        d["oc"]["note"] = "edited"
    else:
        d["pilot"]["z_V"] = {k: [v[0] + 1.0, v[1]] for k, v in d["pilot"]["z_V"].items()}
    Path(Y.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_judge()
    assert e.value.code == 2 and not Path(Y.judge_marker).exists()


@pytest.mark.parametrize("stage", ["seal", "judge"])
def test_seal_and_judge_refuse_changed_measurement_keys(w, stage):
    through(w, "records" if stage == "seal" else "seal")
    before = Path(Y.summary).read_text()
    w.ctx["keys"] = lambda: dict(w_measure_key="other", u_measure_key=Y.u_measure_key)
    with pytest.raises(SystemExit) as e:
        getattr(w.runner(), f"stage_{stage}")()
    assert e.value.code == YR.EXIT_KEY and Path(Y.summary).read_text() == before
    assert not Path(Y.judge_marker).exists()
