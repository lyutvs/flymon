"""Y's phase-A chain (Y.7 0 · 1 · 4 · 5): stage0 (fixtures, bit identity, thresholds, comparison, tables; INVALID on a
failed check; STOP_REUSE on a W or X fact), reuse, pilot (admission, 4a in order, θ, records), precheck (kept k
ranges, STOP records, P2-10 / P2-12, env hashes, resumable cells, θ̂ = pilot block); refusals; archives."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import w_oc, w_rules
from flymon.brain import w_verdict as WV
from flymon.brain import y_rules as R
from flymon.brain import y_runner as YR
from flymon.brain import y_store as YS
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
OK = dict(ok=True)


def _rep(la, lp, st):
    return {"pre": {"A": [[la, 0]] * 8, "P": [[lp, 0]] * 8}, "_st": st}


class World:
    def __init__(self, tmp_path, monkeypatch, k_ranges=((4, 8), (6, 10))):
        monkeypatch.chdir(tmp_path)
        self.ys = dataclasses.replace(Y, archive_root=str(tmp_path / "arch"), precheck_reps=2, oc_reps=2,
                                      compare_reps=2, p26_reps=2, small_boot_draws=1, oc_chunk=2, boot_reps=2)
        self.arch = tmp_path / "arch"
        kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
        self.wpilot = w_oc.synthetic_pilot(np.random.default_rng(5), n_pair=16, **kw)
        self.wkeys = list(Y.pilot_w_pairs) + [f"b|{40 + i}|w{i}|y{i}" for i in range(13)]
        self.theta_w = w_oc.fit(self.wpilot)
        self.vdata = dict(zip(Y.pilot_v_pairs, w_oc.synthetic_pilot(np.random.default_rng(6), n_pair=4, **kw)))
        self.why, self.machine = [], []
        self.w_doc = dict(reuse=dict(z_V={"A": list(Z["A"]), "P": list(Z["P"])}),
                          pilot=dict(record=dict(pairs={k: dict(naive_d=0.1, naive_median=dict(MBON13_X=40.0,
                                                                                                  MBON05_X=90.0))
                                                        for k in self.wkeys})))
        self.facts = dict(x_doc=dict(stage0=dict(decision_files={"x.py": "1"}),
                                     precheck=dict(calibration=dict(min=dict(a=10.0, b=3.0)))),
                          ancestors={"868771a": True, "4b81035": True}, last="4b81035" + "0" * 33,
                          git=dict(tracked=True, dirty=False), diag_sha=Y.x_precheck_diag_sha256, x_files={"x.py": "1"},
                          w_ancestors={c: True for _, c in Y.w_commits},
                          w_blocks={b: Y.w_measure_key for b, _ in Y.w_commits})
        self.oc = json.loads(canonical(w_oc.run(self.wpilot, Z, W, lambda K, F: K * F, n_boot=2, n_rep=2,
                                                n_boot_rep=2)))
        self.ctx = dict(keys=lambda: dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key),
                        params=lambda: Params(),
                        x_reuse=lambda: (list(self.why), self.w_doc, dict(zip(self.wkeys, self.wpilot)), self.theta_w),
                        x_facts=lambda ys: dict(self.facts), w_oc_detail=lambda: (self.oc, "o" * 64),
                        v_candidates=lambda ys: [(dict(key=k), dict(report=_rep(46.0, 109.0, dict(d_pre=0.1, testable=True))))
                                                 for k in Y.pilot_v_pairs],
                        v_even_oracle=lambda keys, z: {}, pilot_measure=self._measure,
                        pilot_back=lambda man: (dict(self.vdata), []))
        monkeypatch.setattr(YR, "summary_git", lambda p: dict(tracked=True, dirty=False, judged=[]))
        monkeypatch.setattr(YR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(YR, "pair_stats", lambda rep, z, t: rep["_st"])
        for name in ("synthetic_validation", "generator_fixtures"):
            monkeypatch.setattr(YR.y_oc, name, lambda *a, **k: dict(OK))
        Path(Y.oracle_detail).parent.mkdir(parents=True, exist_ok=True)
        Path(Y.oracle_detail).write_text("{}")
        Path(Y.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(Y.summary).write_text(json.dumps(dict(
            digest=dict(outcome="PASS"),
            oracle=dict(outcome="PASS", n=249, k_ranges=[list(k) for k in k_ranges],
                        derived=dict(lenient_levels=dict(n=9, quantiles=[0.0, 0.25, 0.5, 0.75, 1.0],
                                                         A=[16.0, 20.0, 30.0, 40.0, 50.0],
                                                         P=[35.0, 50.0, 90.0, 120.0, 150.0]))),
            budget=dict(ledger=[dict(stage="oracle", wall_s=3600.0)]))))

    def _measure(self, rows, pool, check=None):
        man = []
        for i, k in enumerate(Y.pilot_v_pairs):
            p = Path(f"results/y/cache/w_learn/u{i}.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(dict(k=k)))
            man.append(dict(key=f"{k}|0|R", cache_file=str(p), cache_key=f"c{i}", sha256="s"))
        job = dict(wall_s=1.0, train_s=1.0, probe_s=1.0, probe_seeds=[0] * 8, stages=[{}] * 3)
        return dict(pairs=dict(self.vdata), machine=list(self.machine), manifest=man, jobs=[job] * 4, n_units=96)

    def runner(self):
        return YR.Runner(self.ctx, self.ys)


def doc():
    return json.loads(Path(Y.summary).read_text())


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_stage0_pass(w):
    out = w.runner().stage_stage0()
    assert out["outcome"] == R.PASS and out == doc()["stage0"]
    assert out["bit_identity"]["equal"] and out["thresholds"]["n_resid"] == len(w.theta_w["resid"])
    assert set(out["compare"]["candidates"]) == set("abcdef") and out["tables"]["designs"]["n"] == 1500
    assert out["tables"]["m_needed"]["k"] == list(range(4, 11)) and "flymon/brain/y_oc.py" in out["decision_files"]
    assert {a["src"] for a in out["archive"]} == {Y.oracle_detail, Y.stage0_detail}


def test_stage0_invalid_on_a_failed_fixture(w, monkeypatch):
    monkeypatch.setattr(YR.y_oc, "generator_fixtures", lambda *a, **k: dict(ok=False, accept=dict(ok=False)))
    out = w.runner().stage_stage0()
    assert out["outcome"] == R.INVALID and "픽스처 generator 실패" in out["reasons"] and "archive" not in out


@pytest.mark.parametrize("breaker", ["w", "w_block", "x"])
def test_stage0_and_reuse_stop_reuse(w, breaker):
    if breaker == "w":
        w.why = ["W 측정 키 x ≠ 761274e0…"]
    elif breaker == "w_block":
        w.facts["w_ancestors"] = {"5fbc4c8": True, "f30ae35": True}
    else:
        w.facts["diag_sha"] = "0" * 64
    out = w.runner().stage_stage0()
    assert out["outcome"] == R.STOP_REUSE and out["records_unavailable"] is True
    assert out["sentence"].startswith("Y 재사용 조건(Y.7 1")
    with pytest.raises(SystemExit) as e:
        w.runner().stage_reuse()
    assert e.value.code == 2


def test_reuse_pass_then_pilot_pass_then_precheck(w):
    w.runner().stage_stage0()
    assert w.runner().stage_reuse()["outcome"] == R.PASS
    pil = w.runner().stage_pilot()
    assert pil["outcome"] == R.PASS and pil["n_admitted"] == 7 and pil["n_sigma"] == 6
    assert pil["candidates"] == list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs)   # blocks sort keys; order lives here
    assert set(pil["admission"]) == set(pil["candidates"])
    assert pil["theta"] == json.loads(canonical(w_oc.summary(YR.y_oc.fit_y(
        [dict(zip(w.wkeys, w.wpilot))[k] for k in Y.pilot_w_pairs] + [w.vdata[k] for k in Y.pilot_v_pairs],
        list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs), YR.y_oc.r_v0(w.theta_w)))))
    assert pil["level_compare"]["oracle_lenient"]["n"] == 9 and pil["level_compare"]["pilot"]["n"] == 7
    assert pil["exploratory"]["label"] == "탐색" and pil["costs"]["workers"] == W.cost_workers()
    assert pil["flip"]["available"] is False                    # the fake V cache has no even-block entry
    assert {a["src"] for a in pil["archive"]} >= {Y.pilot_detail, "results/y/cache/w_learn/u0.json"}
    pc = w.runner().stage_precheck()
    assert pc["outcome"] in (R.PASS, R.STOP_OC_UNREACHABLE) and pc == doc()["precheck"]
    assert pc["k_ranges"] == [[4, 8], [6, 10]] and set(pc["env"]) >= {"uv_lock_sha256", "python", "numpy", "y_files"}
    assert pc["small_bootstrap"]["n_draws"] == 1 and "lo" in pc["thresholds"]


def test_pilot_stop_few(w):
    for k in (Y.pilot_v_pairs[0], Y.pilot_v_pairs[3]):
        w.vdata[k] = dict(w.vdata[k], pre=np.asarray(w.vdata[k]["pre"]).copy())
        w.vdata[k]["pre"][..., WV.A, WV.X] = 5
    w.runner().stage_stage0()
    w.runner().stage_reuse()
    out = w.runner().stage_pilot()
    assert out["outcome"] == R.STOP_PILOT_FEW and out["n_sigma"] == 4 and "theta" not in out
    assert out["records_unavailable"] is True and out["archive"]
    with pytest.raises(SystemExit):
        w.runner().stage_precheck()


def test_pilot_no_effect_and_invalid(w, monkeypatch):
    w.runner().stage_stage0()
    w.runner().stage_reuse()
    w.machine = ["fly 0: N pre ≠ R pre"]
    out = w.runner().stage_pilot()
    assert out["outcome"] == R.INVALID and "archive" not in out
    Path(Y.summary).write_text(json.dumps({k: v for k, v in doc().items() if k != "pilot"}))
    w.machine = []
    monkeypatch.setattr(w_rules, "pilot", lambda rec, mach, spec: dict(outcome="STOP_PILOT_NO_EFFECT",
                                                                       reasons=["(i) 보상 연합 …"], values={}))
    out = w.runner().stage_pilot()
    assert out["outcome"] == R.STOP_PILOT_NO_EFFECT and out["sentence"].endswith("(i) 보상 연합 … — 주 세트 학습 측정 없이 멈춘다.")


def test_precheck_stop_records_and_reads_oracle_k_ranges(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, k_ranges=((4, 8),))
    w.ys = dataclasses.replace(w.ys, p_power=1.01)                  # nothing can pass → STOP with records
    for st in ("stage0", "reuse", "pilot"):
        getattr(w.runner(), f"stage_{st}")()
    pc = w.runner().stage_precheck()
    assert pc["outcome"] == R.STOP_OC_UNREACHABLE and pc["k_ranges"] == [[4, 8]]
    assert "k 범위 [6, 10]" not in pc["sentence"] and pc["records_target"]["k_range"] == [4, 8]
    assert set(pc["records"]["variants"]) == {"Y", "R-W", "R-Σ2", "R-un", "R-V", "R-pre"}


def test_precheck_refuses_a_changed_theta(w):
    for st in ("stage0", "reuse", "pilot"):
        getattr(w.runner(), f"stage_{st}")()
    d = doc()
    d["pilot"]["theta"]["m0"][0][0] += 1.0
    Path(Y.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        w.runner().stage_precheck()
    assert e.value.code == 2


def test_precheck_resumes_from_cells(w, monkeypatch):
    for st in ("stage0", "reuse", "pilot"):
        getattr(w.runner(), f"stage_{st}")()
    real_sb = YR.y_oc.small_bootstrap

    def boom(*a, **k):
        raise RuntimeError("killed")
    monkeypatch.setattr(YR.y_oc, "small_bootstrap", boom)
    with pytest.raises(RuntimeError):
        w.runner().stage_precheck()
    cells = sorted(Path(Y.progress_dir, "precheck").glob("point_*.json"))
    assert "precheck" not in doc() and len(cells) == 12
    before = {c: c.stat().st_mtime_ns for c in cells}
    monkeypatch.setattr(YR.y_oc, "small_bootstrap", real_sb)
    out = w.runner().stage_precheck()
    assert out["outcome"] in (R.PASS, R.STOP_OC_UNREACHABLE) and out == doc()["precheck"]
    assert {c: c.stat().st_mtime_ns for c in cells} == before          # every point cell reused, none rewritten
    assert doc()["budget"]["ledger"][-1]["wall_s"] >= 0.0


def test_refusals(w):
    for st in ("reuse", "pilot", "precheck"):
        with pytest.raises(SystemExit) as e:
            getattr(w.runner(), f"stage_{st}")()
        assert e.value.code == 2
    w.runner().stage_stage0()
    with pytest.raises(SystemExit):
        w.runner().stage_stage0()


def test_stage_files():
    assert YS.stage_files("digest", Y) == [] and YS.stage_files("oracle", Y) == [Y.oracle_detail]
    assert YS.stage_files("stage0", Y) == [Y.oracle_detail, Y.stage0_detail]
    assert YS.stage_files("reuse", Y) == [Y.oracle_detail]
    assert YS.stage_files("precheck", Y) == [Y.oracle_detail, Y.precheck_detail]
