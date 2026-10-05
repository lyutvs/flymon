"""X's phase-A chain (X.5 0 · 0a): stage0 then precheck; refusals; STOP_REUSE on every broken W fact (records
unavailable); the W diagnosis checked draw by draw; precheck PASS / STOP blocks with their details; resumable cells."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import x_runner as XR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.w_spec import SPEC as W
from flymon.brain.x_spec import SPEC as X0

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ND = [0.08, -0.05, -0.32, 1.09, 1.43, 2.72, 4.13, 5.9, 6.0, 7.98, -9.89, 10.7, -12.88, 13.3, 14.5, -16.07]
KEYS = [f"p|{j}" for j in range(16)]
XS = dataclasses.replace(X0, w_measure_key="k" * 64, w_pipeline_key="p" * 64, w_measure_file_sha="m" * 64,
                         w_oc_detail_sha="o" * 64, w_summary_commit="c" * 7, balanced_pairs=tuple(KEYS[:3]),
                         precheck_reps=6, oc_reps=4, diag_reps=2, diag_chunk=2, p26_reps=4, boot_reps=4)


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.pilot = w_oc.synthetic_pilot(np.random.default_rng(5), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                          learn=(20.0, 10.0))
        theta = w_oc.fit(self.pilot)
        self.oc = json.loads(canonical(w_oc.run(self.pilot, Z, W, lambda K, F: K * F, n_boot=3, n_rep=2,
                                                n_boot_rep=2)))
        self.w_doc = dict(
            stage0=dict(decision_files={"flymon/brain/w_measure.py": "m" * 64}),
            reuse=dict(z_V={"A": list(Z["A"]), "P": list(Z["P"])}),
            pilot=dict(outcome="PASS", w_measure_key="k" * 64, pairs=list(KEYS), manifest=[{}] * 384,
                       record=dict(pairs={k: dict(naive_d=d) for k, d in zip(KEYS, ND)})),
            oc=dict(w_measure_key="k" * 64, theta=json.loads(canonical(w_oc.summary(theta))), detail_sha256="o" * 64))
        self.keys = dict(w_measure_key="k" * 64, pipeline_key="p" * 64, w_measure_sha="m" * 64)
        self.facts = dict(last="c" * 40, ancestors={"5fbc4c8": True, "f30ae35": True},
                          git=dict(tracked=True, dirty=False))
        self.bad_manifest = []
        self.ctx = dict(w_doc=lambda: self.w_doc, oc_detail=lambda p: (self.oc, "o" * 64),
                        pilot=lambda d: ((None, list(self.bad_manifest)) if self.bad_manifest
                                         else (dict(zip(KEYS, self.pilot)), [])),
                        keys=lambda: dict(self.keys), facts=lambda xs: dict(self.facts), params=lambda: Params())
        monkeypatch.setattr(XR, "summary_git", lambda p: dict(tracked=True, dirty=False, judged=[]))
        monkeypatch.setattr(XR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(XR.x_oc, "synthetic_validation", lambda xs, z, n_rep=None: dict(ok=True))

    def runner(self, xs=XS):
        return XR.Runner(self.ctx, xs)


def doc():
    return json.loads(Path(X0.summary).read_text())


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_stage0_pass(w):
    out = w.runner().stage_stage0()
    assert out["outcome"] == "PASS" and out["p26"]["equal"] and out["w_cal_diagnosis"]["reproduced"]
    assert out["tables"]["m_needed"]["m"]["0.5"] == [3, 3, 3, 4, 4] and out["oc_timing"]["reps"] == 4
    assert Path(X0.wcal_detail).exists() and doc()["ledger"][0]["stage"] == "stage0"
    assert out["w_measure_key"] == "k" * 64 and out["pipeline_key"] == "p" * 64


@pytest.mark.parametrize("breaker,needle", [
    (lambda w: w.keys.update(w_measure_key="z" * 64), "W 측정 키 "),
    (lambda w: w.facts["ancestors"].update({"5fbc4c8": False}), "5fbc4c8"),
    (lambda w: w.bad_manifest.append("b|0|0|R: raw file sha256 differs"), "W 파일럿 원자료"),
    (lambda w: w.w_doc["pilot"].update(pairs=KEYS[::-1]), "쌍 순서"),
    (lambda w: w.w_doc["oc"]["theta"].update(n_fly=9), "θ̂ 요약"),
    (lambda w: w.oc["boot_calibration"][2]["min"].update(ok=not w.oc["boot_calibration"][2]["min"]["ok"]),
     "W 보정 진단 재현 불일치"),
])
def test_stage0_stop_reuse_on_broken_w_facts(w, breaker, needle):
    breaker(w)
    out = w.runner().stage_stage0()
    assert out["outcome"] == "STOP_REUSE" and needle in out["sentence"] and out["records_unavailable"] is True
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2


def test_refusals(w, monkeypatch):
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2 and not Path(X0.summary).exists()
    w.runner().stage_stage0()
    with pytest.raises(SystemExit):
        w.runner().stage_stage0()
    monkeypatch.setattr(XR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=["flymon/brain/w_oc.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2 and "precheck" not in doc()


def test_precheck_stop_writes_records_diag_and_env(w):
    w.runner().stage_stage0()
    out = w.runner(dataclasses.replace(XS, p_power=1.01)).stage_precheck()
    assert out["outcome"] == "STOP_OC_UNREACHABLE" and out["sentence"].startswith("점 θ 사전 점검(X.9.1.1)")
    assert set(out["records"]["variants"]) == {"V0", "V1", "V2"}
    assert out["diag"]["iii"]["rank1"] is None or "knobs" in out["diag"]["iii"]["rank1"]
    assert out["residuals"]["n_pairs"] == dict(balanced=3, rest=13)
    assert set(out["env"]) == {"uv_lock_sha256", "python", "numpy", "x_files", "w_files"}
    assert "flymon/brain/w_measure.py" in out["env"]["w_files"] and "scripts/run_x.py" in out["env"]["x_files"]
    for p in (X0.precheck_detail, X0.diag_detail):
        assert Path(p).exists()
    assert out["diag_detail_sha256"] and out["precheck_detail_sha256"]


def test_precheck_pass_has_no_records(w):
    w.runner().stage_stage0()
    out = w.runner(dataclasses.replace(XS, p_power=0.0, p_false=1.0)).stage_precheck()
    assert out["outcome"] == "PASS" and out["records"] is None and "국면 B" in out["records_note"]


def test_precheck_resumes_from_cells(w, monkeypatch):
    w.runner().stage_stage0()
    saved = Path(X0.summary).read_text()
    hard = dataclasses.replace(XS, p_power=1.01)
    first = w.runner(hard).stage_precheck()
    Path(X0.summary).write_text(saved)                       # as if killed before the block was written

    def boom(*a, **k):
        raise AssertionError("a finished cell was recomputed")
    monkeypatch.setattr(XR.x_oc, "evaluate_grid", boom)
    second = w.runner(hard).stage_precheck()
    assert second["diag"] == first["diag"] and second["records"] == first["records"]


def _plain_json(path):
    def no_nan(c):
        raise ValueError(c)
    return json.loads(Path(path).read_text(), parse_constant=no_nan)


def test_blocks_and_details_are_plain_json_equal_to_the_returns(w):
    out0 = w.runner().stage_stage0()
    out1 = w.runner(dataclasses.replace(XS, p_power=1.01)).stage_precheck()
    d = _plain_json(X0.summary)
    assert d["stage0"] == out0 and d["precheck"] == out1
    for p in (X0.wcal_detail, X0.precheck_detail, X0.diag_detail):
        _plain_json(p)
    det = _plain_json(X0.precheck_detail)
    assert set(det["precheck"]["calibration"]) == {"min", "max"} and set(det["records"]["variants"]) == {"V0", "V1",
                                                                                                         "V2"}
    for c in Path(X0.progress_dir).glob("*/*.json"):
        _plain_json(c)


def test_stage0_resumes_from_cells(w, monkeypatch):
    first = w.runner().stage_stage0()
    Path(X0.summary).unlink()
    assert {p.stem for p in (Path(X0.progress_dir) / "stage0").glob("*.json")} == {"w_cal_diagnosis", "synthetic",
                                                                                   "p26"}

    def boom(*a, **k):
        raise AssertionError("a finished cell was recomputed")
    for f in ("w_cal_diagnosis", "bit_identity", "synthetic_validation"):
        monkeypatch.setattr(XR.x_oc, f, boom)
    second = w.runner().stage_stage0()
    for k in ("w_cal_diagnosis", "synthetic", "p26", "outcome", "tables"):
        assert second[k] == first[k]


def test_precheck_point_cells_and_a_changed_spec_or_corrupt_cell_recomputes(w, monkeypatch):
    w.runner().stage_stage0()
    saved = Path(X0.summary).read_text()
    hard = dataclasses.replace(XS, p_power=1.01)
    first = w.runner(hard).stage_precheck()
    cells = {p.stem for p in (Path(X0.progress_dir) / "precheck").glob("*.json")}
    assert {f"point_g{g}_{m}" for g in XS.cluster_grid for m in ("min", "max")} <= cells
    Path(X0.summary).write_text(saved)
    monkeypatch.setattr(XR.x_oc, "evaluate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    monkeypatch.setattr(XR.x_oc, "evaluate_grid", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    second = w.runner(hard).stage_precheck()
    for k in ("outcome", "sentence", "best_power", "records_target", "at_f32", "records", "diag"):
        assert second[k] == first[k]
    Path(X0.summary).write_text(saved)
    (Path(X0.progress_dir) / "precheck" / f"point_g{XS.cluster_grid[0]}_min.json").write_text("{trunc")
    with pytest.raises(AssertionError, match="recomputed"):                # a half-written cell is not trusted
        w.runner(hard).stage_precheck()
    with pytest.raises(AssertionError, match="recomputed"):                # another XSpec never reuses a cell
        w.runner(dataclasses.replace(hard, precheck_reps=7)).stage_precheck()
