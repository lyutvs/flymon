"""aa_runner (E): the pre-seal restart (INVALID stage0 → aa_* fix → restart_preseal → stage0 PASS; a tracked summary
rewritten to a preseal_restarts record; refused once seal_code exists or after a STOP), the smoke raw archived like
the other stages, the smoke z_V check against W block reuse (AA.2), numerical failures in θ_AA's fit_y and the G.6
rows as record-only statuses, and the CLI's QUIET per-pair containers."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import aa_estimate, aa_rules, w_oc, y_oc
from flymon.brain import aa_runner as AR
from flymon.brain.aa_spec import SPEC
from tests.brain.aa_world import World, doc

ROOT = Path(__file__).resolve().parents[2]
RUNNER = "flymon/brain/aa_runner.py"


def _run_aa():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_aa", ROOT / "scripts/run_aa.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _world(tmp_path, monkeypatch, **kw):
    w = World(tmp_path, monkeypatch)
    w.s = dataclasses.replace(w.s, workers=1, **kw)
    monkeypatch.setattr(AR, "_head_summary", lambda path: None)        # untracked unless a test says otherwise
    return w


def _fixed(monkeypatch):
    """An aa_* fix: aa_runner.py's hash changes (as after a patch)."""
    real = AR.env_now
    monkeypatch.setattr(AR, "env_now", lambda: (lambda d: dict(d, aa_files=dict(d["aa_files"], **{RUNNER: "f" * 64})))(
        real()))


def _log(w, n=1234):
    Path(w.s.tests_log).parent.mkdir(parents=True, exist_ok=True)
    Path(w.s.tests_log).write_text(f"{n} passed in 600.00s\nexit 0\n")


def _code(fn):
    with pytest.raises(SystemExit) as e:
        fn()
    return e.value.code


# ================================================================ pre-seal restart
def test_restart_after_invalid_stage0_then_stage0_pass(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    Path(w.s.tests_log).write_text("3 failed, 10 passed\nexit 1\n")
    out = w.runner().run("stage0")
    assert out["outcome"] == aa_rules.INVALID and "archive" not in out
    assert (Path(w.s.progress_dir) / "stage0.env.json").exists()
    _fixed(monkeypatch)
    _log(w, 1235)
    assert _code(lambda: w.runner().run("stage0")) == 2                   # the INVALID block is still there
    Path(w.s.summary).unlink()                                          # even without it, the first-start env refuses
    assert _code(lambda: w.runner().run("stage0")) == AR.EXIT_ENV
    res = w.runner().restart_preseal()
    assert res["outcome"] == "PASS"
    r = res["restart"]
    inv = Path(f"{w.s.raw_dir}.invalid-1")
    assert r["n"] == 1 and r["invalid_dir"] == str(inv) and r["archive"] is None
    assert r["summary"].startswith("removed") and not Path(w.s.summary).exists()
    assert not Path(w.s.raw_dir).exists() and (inv / "progress" / "stage0.env.json").exists()
    assert (inv / "stage0.json").exists() and (inv / "tests_0a.log").exists()
    note = json.loads((inv / "restart_preseal.json").read_text())
    assert note["n"] == 1 and note["raw"]["dst"] == str(inv) and note["summary"] == r["summary"]
    kept = json.loads(Path(note["summary_copy"]).read_text())
    assert any("env_mismatch" in e for e in kept["budget"]["ledger"])  # the summary as it was, kept
    _log(w, 1236)                                                       # §2 again (results/aa moved aside)
    out = w.runner().run("stage0")
    assert out["outcome"] == "PASS" and out["env"]["aa_files"][RUNNER] == "f" * 64
    assert Path(w.s.archive_root, "stage0").exists()


def test_restart_tracked_summary_after_committed_smoke(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("smoke")
    led0 = list(doc()["budget"]["ledger"])
    arch = Path(w.s.archive_root)
    assert (arch / "smoke").exists() and (arch / "stage0").exists()
    monkeypatch.setattr(AR, "_head_summary", lambda path: Path(path).read_bytes())    # committed at HEAD
    _fixed(monkeypatch)
    assert _code(lambda: w.runner().run("seal_code")) == AR.EXIT_ENV               # committed env.aa_files refuse
    res = w.runner().restart_preseal()
    r = res["restart"]
    d = doc()
    assert set(d) == {"preseal_restarts", "budget"} and r["summary"].startswith("rewritten")
    assert d["budget"]["ledger"][:len(led0)] == led0 and "env_mismatch" in d["budget"]["ledger"][-1]
    pr = d["preseal_restarts"]
    assert len(pr) == 1 and pr[0]["head_blocks"] == ["reuse", "smoke", "stage0"] == pr[0]["dropped_blocks"]
    assert not arch.exists() and r["archive"]["dst"] == str(arch.with_name(arch.name + ".invalid-1"))
    assert (arch.with_name(arch.name + ".invalid-1") / "smoke").exists()
    _log(w, 1235)
    out = w.chain("smoke")
    assert out["outcome"] == "PASS" and doc()["preseal_restarts"] == pr
    assert len(doc()["budget"]["ledger"]) == len(led0) + 1 + 3         # carried + mismatch + the new three
    res2 = w.runner().restart_preseal()                                 # a second restart: slot 2, record appended
    assert res2["restart"]["n"] == 2 and len(doc()["preseal_restarts"]) == 2


def test_restart_refused_after_seal_code(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    before = json.loads(Path(w.s.summary).read_text())
    capsys.readouterr()
    assert _code(lambda: w.runner().restart_preseal()) == 2
    assert "seal_code" in capsys.readouterr().err
    assert Path(w.s.raw_dir).exists() and not Path(f"{w.s.raw_dir}.invalid-1").exists()
    assert Path(w.s.archive_root).exists() and doc() == before


def test_restart_refused_when_head_holds_seal_code(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("seal_code")
    head = Path(w.s.summary).read_bytes()
    d = doc()
    d.pop("seal_code")
    Path(w.s.summary).write_text(json.dumps(d))                         # the working copy lost it; HEAD has it
    monkeypatch.setattr(AR, "_head_summary", lambda path: head)
    assert _code(lambda: w.runner().restart_preseal()) == 2
    assert Path(w.s.raw_dir).exists()


def test_restart_refused_after_stop_or_main_raw_or_nothing(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("stage0")
    w.facts_ok = False
    assert w.runner().run("reuse")["outcome"] == aa_rules.STOP_REUSE
    assert _code(lambda: w.runner().restart_preseal()) == 2             # a STOP is final
    d = doc()
    d.pop("reuse")
    Path(w.s.summary).write_text(json.dumps(d))
    p = Path(w.s.cache_dir, "w_learn", "x.json")
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    assert _code(lambda: w.runner().restart_preseal()) == 2             # main-set raw exists
    p.unlink()
    assert w.runner().restart_preseal()["outcome"] == "PASS"
    assert _code(lambda: w.runner().restart_preseal()) == 2             # nothing left to restart


def test_restart_cli_arguments():
    import argparse
    m = _run_aa()
    assert m.RESTART == "restart_preseal"
    assert m.arg_reasons(argparse.Namespace(stage="restart_preseal", note=None, name=None), SPEC) == []
    assert m.arg_reasons(argparse.Namespace(stage="restart_preseal", note=None, name="1"), SPEC)
    assert m.main(["--stage", "restart_preseal", "--name", "1"]) == 2


# ================================================================ smoke: archive and z_V
def test_smoke_raw_archived(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("smoke")
    blk = doc()["smoke"]
    det = json.loads(Path(blk["detail_path"]).read_text())
    assert det["manifest"] == blk["manifest"] and len(det["manifest"]) == 4
    assert len(blk["archive"]) == 5                                     # the detail + R · N · RN + naive raw
    again = w.runner().archive("smoke")                                 # `--stage archive --name smoke`
    assert len(again) > 0 and again == blk["archive"]
    assert all(Path(a["dst"]).exists() and Path(a["dst"]).is_relative_to(Path(w.s.archive_root, "smoke"))
               for a in again)
    assert {a["src"] for a in again} == {blk["detail_path"], *(m["cache_file"] for m in blk["manifest"])}


@pytest.mark.parametrize("case", ["w_reuse", "aa_z"])
def test_smoke_z_v_checked_against_w_reuse(tmp_path, monkeypatch, case):
    w = _world(tmp_path, monkeypatch)
    w.chain("reuse")
    if case == "w_reuse":
        w.wz = dict(w.wz, A=(w.wz["A"][0] + 1e-9, w.wz["A"][1]))
    else:
        w.z = dict(w.z, P=(w.z["P"][0], w.z["P"][1] * 1.5))
    out = w.runner().run("smoke")
    assert out["outcome"] == aa_rules.INVALID
    want = "Y 블록 pilot의 z_V ≠ W 블록 reuse의 z_V" if case == "w_reuse" else "AA가 쓰는 z(Y _z_b) ≠ W 블록 reuse의 z_V"
    assert want in out["problems"]


# ================================================================ G.6: numerical failures are record-only
def _theta():
    pil = w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, base=(40.0, 90.0), sd=4.0, corr=0.8,
                               learn=(20.0, 10.0))
    return y_oc.fit_y(pil, [f"a|{i}|O{i % 4}|y{i}" for i in range(6)], np.eye(4))


def test_g6_numerical_failures_are_statuses(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    th = _theta()
    w.ctx["y_theta"] = lambda where: dict(why=[], theta=th, theta_w=None, r=np.eye(4), keys=[], z=dict(w.z))

    def bad_fit(*a, **k):
        raise FloatingPointError("overflow in exp")
    monkeypatch.setattr(y_oc, "fit_y", bad_fit)
    seen = []

    def row(theta, z, tgt, costs, n, s, ys=None, cell=None):
        seen.append(tgt)
        if len(seen) == 1:
            raise np.linalg.LinAlgError("singular matrix")
        return dict(status="ok", target=tgt, false=[[0.0, 0.25]], fill_bad=[False, True], n_pass=7)

    def het(theta, z, mu, tau, ref, costs, n, s, ys=None):
        raise ZeroDivisionError("division by zero")
    monkeypatch.setattr(aa_estimate, "g6_row", row)
    monkeypatch.setattr(aa_estimate, "g6_hetero", het)
    w.chain("records")
    out = doc()["records"]
    assert out["outcome"] == "PASS" and out["reasons"] == []
    g6 = out["g6"]
    assert g6["theta_status"]["AA"] == "θ_AA 적합 실패 FloatingPointError: overflow in exp"
    live = [r for r in g6["rows"] if r["status"] != aa_estimate.TARGET_LOW]
    aa = [r for r in live if r["theta"] == "AA"]
    assert aa and all(r["status"] == g6["theta_status"]["AA"] for r in aa)
    y = [r for r in live if r["theta"] == "Y"]
    assert y[0]["status"] == "계산 실패 LinAlgError: singular matrix" and all(r["status"] == "ok" for r in y[1:])
    for h in g6["hetero"]:
        if h["theta"] == "AA" or h["status"] == aa_estimate.TARGET_LOW or h["status"].startswith("DL"):
            continue
        pt = [r for r in g6["rows"] if (r["theta"], r["row"], r["scale"]) == ("Y", "point", h["scale"])][0]
        want = "계산 실패 ZeroDivisionError: division by zero" if pt["status"] == "ok" else \
            f"점 행 상태 {pt['status']} — 계산 안 함"
        assert h["status"] == want


def test_g6_job_does_not_swallow_other_errors(monkeypatch):
    def boom(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(aa_estimate, "g6_row", boom)
    with pytest.raises(KeyError):
        AR._g6_job("row", None, None, 0.5, None, {}, 31, SPEC, None)


# ================================================================ CLI: per-pair containers stay quiet
def test_report_quiet_per_pair_containers():
    m = _run_aa()
    for q in ("pairs", "primary", "s1_records", "sets"):
        assert q in m.QUIET
    est = dict(outcome="PASS", sentence="s", primary=dict(pairs=[dict(key="b|1|x|y", naive_d=0.1)], range=[0, 1]),
               compare={"reward_assoc": []}, coverage_block="coverage")
    rec = dict(outcome="PASS", pairs={"b|1|x|y": dict(naive_d=0.1)}, s1_records=dict(range=[0, 1]),
               sets=dict(S31=dict(range=[0, 1])), phi_quantiles=dict(qs=[0.5]), nulls=[])
    for stage, out in (("estimate", est), ("records", rec)):
        body = json.loads(m.report_lines(stage, out, SPEC.cli_print_chars)[-1])
        assert not {"pairs", "primary", "s1_records", "sets"} & set(body)
        assert "b|1|x|y" not in json.dumps(body) and "naive_d" not in json.dumps(body)
