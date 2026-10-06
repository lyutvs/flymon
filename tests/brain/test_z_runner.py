"""Z's order-0 chain (Z.7 0a–0d, Z.9.2 P1-5 T3 · T6 · T8 · T11 · T14): stage0 → ydiag → reuse → sens → split on a
synthetic world (fake ctx, synthetic Y cells / oracle.json in a temporary cwd), refusals, env checks, archives."""
import ast
import dataclasses
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import w_oc, y_oc, y_spec, y_store
from flymon.brain import z_oc as O
from flymon.brain import z_rules as R
from flymon.brain import z_runner as ZR
from flymon.brain import z_store
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT, sha256_file
from flymon.brain.y_spec import SPEC as Y
from flymon.brain.z_spec import SPEC as Z

ZV = {"A": [16.917, 12.484], "P": [80.167, 29.775]}
N_CELLS = 4


def _cal():
    return dict(ok=True, failure=None, corners=[dict(a=30.0, b=6.0, w=1.0, label="a0b0")],
                a=dict(status="ok", coarse_step=False, stage=1), b=[dict(status="ok", coarse_step=False, stage=1)])


def _save_cell(path, d):
    buf = io.BytesIO()
    np.savez(buf, __key__=np.array("k"), __meta__=np.array(json.dumps(d["meta"])), **d["arrays"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.getvalue())


def _lenient_keys():
    b = [f"b|{i}|xb{i}|yb{i}" for i in range(11)]
    a = [f"a|{20 + i}|{('Earthquake', 'Surf', 'Psychic', 'Flamethrower')[i % 4]}|ya{i}" for i in range(20)]
    return b + a


class World:
    def __init__(self, tmp_path, monkeypatch, n_p_keys=None):
        monkeypatch.chdir(tmp_path)
        self.tmp = tmp_path
        rng = np.random.default_rng(1)
        kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
        pilot = w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw)
        self.theta = y_oc.fit_y(pilot, list(Z.y_admitted), np.eye(4))
        # Y cells and Y's oc.json
        shp = (3, 2, 2) + y_oc.x_oc.grid_shape(Y)
        self.draws = [dict(arrays=dict(hits=rng.integers(0, 400, shp).astype(np.uint16),
                                       fills=np.zeros((3, 2, 2, Y.k_cap - Y.k_min + 1))),
                           meta=dict(cal=dict(min=_cal(), max=_cal()), n_rep=400)) for _ in range(N_CELLS)]
        for bi, d in enumerate(self.draws):
            _save_cell(f"{Z.y_cells_dir}/boot_{bi}.npz", d)
        lim = y_oc.limits(self.draws, Y, [tuple(k) for k in Y.k_ranges])
        self.det = z_store.to_json(O.limits_json(lim))
        i, ks = O.design_index(Z)
        # oracle.json: 31 lenient + 5 not
        keys = _lenient_keys()
        recs = [dict(key=k, c=c, axis=k[0], turn=c, value=True, testable=True, d_pre=0.1, L_A=30.0, L_P=60.0, r=0.0,
                     p=0.0, failure=None) for c, k in enumerate(keys)]
        recs += [dict(key=f"b|{90 + j}|n{j}|m{j}", c=31 + j, axis="b", turn=90 + j, value=True, testable=True,
                      d_pre=2.0, L_A=30.0, L_P=60.0, r=0.0, p=0.0, failure=None) for j in range(5)]
        Path("results/y").mkdir(parents=True, exist_ok=True)
        Path(Y.oracle_detail).write_text(json.dumps(dict(pairs=recs)))
        Path(Y.pilot_detail).write_text(json.dumps(dict(manifest=[])))
        self.zs = dataclasses.replace(
            Z, archive_root=str(tmp_path / "arch"), y_cells_n=N_CELLS, redraw_n=2,
            oracle_sha256=sha256_file(Y.oracle_detail), sens_draws=5, workers=1,
            gate_by_k=tuple(np.round(lim["power_lo_by_k"][i][ks], 3).tolist()),
            gate_sim=round(float(lim["sim"][(4, 8)]["power"][i]), 3),
            gate_false=float(np.round(lim["false_hi_by_k"][i][ks], 3).max()), records_budget_h=0.0)
        self.ydoc = dict(
            digest=dict(outcome="PASS", set=dict(digest_keys=Y.main_digest_keys, n_b=Y.main_n_b, n_a=Y.main_n_a,
                                                 last_turn=Y.main_last_turn)),
            oracle=dict(outcome="PASS", detail_sha256=self.zs.oracle_sha256, derived=dict(yield_rule=dict(n_len=31))),
            stage0=dict(outcome="PASS"), reuse=dict(outcome="PASS"),
            pilot=dict(outcome="PASS", z_V=ZV, admitted=list(Z.y_admitted), n_sigma=5,
                       detail_sha256=sha256_file(Y.pilot_detail)),
            precheck=dict(outcome="PASS", calibration=dict(min=_cal())),
            oc=dict(outcome="STOP_OC_UNREACHABLE", detail_sha256="o" * 64))
        self.why, self.theta_why, self.decl_change = [], [], []
        self.ctx = dict(
            keys=lambda: dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key),
            params=lambda: Params(), y_doc=lambda: json.loads(json.dumps(self.ydoc)),
            y_theta=lambda where: (dict(why=list(self.theta_why)) if self.theta_why else
                                   dict(why=[], theta=self.theta, theta_w=self.theta, r=np.eye(4),
                                        keys=list(Z.y_admitted), z={k: tuple(v) for k, v in ZV.items()})),
            y_reuse=lambda where: list(self.why),
            git_facts=lambda path, commits: dict(ancestors={c: True for c in commits}),
            decl_sha=lambda files, commit: {f: ("x" if f in self.decl_change else sha256_file(ROOT / f)) for f in files},
            oc_detail=lambda: (self.det, "o" * 64), y_pilot_back=lambda man: ({}, []))
        monkeypatch.setattr(ZR, "summary_git", lambda p: dict(tracked=True, dirty=False, judged=[]))
        monkeypatch.setattr(ZR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(O, "redraw", lambda th, z, r, bi: self.draws[bi])
        Path(Z.tests_log).parent.mkdir(parents=True, exist_ok=True)
        Path(Z.tests_log).write_text("... 900 passed\nexit 0\n")
        self.L = {}                                                 # (n, δ index) -> constant power for sens

        def fake_sens(th, r, z, n, di, delta, bi, root, share):
            v = self.L.get((n, di), 0.95)
            return dict(arrays=dict(p=np.full((3, 2) + y_oc.x_oc.grid_shape(Y), v)),
                        meta=dict(ok=True, a=30.0, drift_ax=-11.0 + delta, status=None))
        monkeypatch.setattr(O, "sens_job", fake_sens)

    def run(self, stage, zs=None):
        r = ZR.Runner(self.ctx, zs or self.zs)
        r.workers = 1
        return r.run(stage)

    def chain(self, upto):
        out = {}
        for s in Z.order0[:Z.order0.index(upto) + 1]:
            out[s] = self.run(s)
        return out


def test_full_chain_pass(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    out = w.chain("split")
    assert [out[s]["outcome"] for s in Z.order0] == [R.PASS] * 5
    doc = z_store.read_summary()
    assert set(Z.order0) <= set(doc) and len(doc["budget"]["ledger"]) == 5
    assert all(doc[s]["env"] == doc["stage0"]["env"] for s in Z.order0)
    sp = doc["split"]
    assert sp["n_conf"] + sp["precheck"]["h"] == 31 and "|" not in json.dumps(sp["split"])
    assert json.loads(Path(Z.split_detail).read_text())["C"][0].count("|") == 3
    assert Path(w.tmp / "arch/y_cells/oc/boot_0.npz").exists() and Path(w.tmp / "arch/split/z/split.json").exists()
    assert doc["ydiag"]["records_null"] and all(v == "records 상한, Z.9.2 P2-10"
                                                 for v in doc["ydiag"]["records_null"].values())


def test_stage0_invalid_without_a_passing_test_log(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    Path(Z.tests_log).write_text("1 failed\nexit 1\n")
    out = w.run("stage0")
    assert out["outcome"] == R.INVALID and "archive" not in out and "exit 1" in out["reasons"][0]


def test_stage0_invalid_without_a_test_log(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    Path(Z.tests_log).unlink()
    out = w.run("stage0")
    assert out["outcome"] == R.INVALID and out["tests"]["sha256"] is None


def test_ydiag_gate_failures_stop_reuse(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run("stage0")
    w.theta_why = ["θ̂ ≠ Y pilot"]
    out = w.run("ydiag")
    assert out["outcome"] == R.STOP_REUSE and out["where"] == "0b" and "(iii)" in out["sentence"]
    assert out["p314"]["outputs"]["ydiag"] == "부"


def test_ydiag_redraw_mismatch_and_limits_mismatch(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run("stage0")
    bad = dict(w.draws[1], arrays=dict(w.draws[1]["arrays"], hits=w.draws[1]["arrays"]["hits"] + 1))
    monkeypatch.setattr(O, "redraw", lambda th, z, r, bi: bad if bi == 1 else w.draws[bi])
    out = w.run("ydiag")
    assert out["outcome"] == R.STOP_REUSE and out["gates"]["ii"]["unequal"] == [1]


def test_ydiag_limits_mismatch(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run("stage0")
    w.det = dict(w.det, sim={})
    out = w.run("ydiag")
    assert out["outcome"] == R.STOP_REUSE and "(i)" in out["sentence"]


def test_ydiag_records_fill_and_compare(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run("stage0")
    for name in ("rec_distribution", "rec_base_only", "rec_rho", "rec_near", "rec_reach", "point_near",
                 "draw_thresholds", "drift_facts", "rec_selfbase", "rec_cf", "rec_sub", "rec_psize"):
        monkeypatch.setattr(O, name, lambda *a, _n=name, **k: {f"{_n}_v": 1.0, "draw_q5": 0.040})
    for name in ("components", "near_truedprimes", "reach"):
        monkeypatch.setattr(O, name, lambda *a, **k: [])
    zs = dataclasses.replace(w.zs, records_budget_h=6.0, selfbase_ranks=(0,), cf_p_ranks=(0,), cf_ranks=(0,),
                             psize_draws=1, psize_ns=(6,))
    for name in ("selfbase_job", "cf_job", "sub_job", "psize_job"):
        monkeypatch.setattr(O, name, lambda *a, **k: dict(arrays=dict(p=np.zeros(1)), meta=dict(status="floor")))
    out = ZR.Runner(w.ctx, zs).run("ydiag")
    assert out["outcome"] == R.PASS and out["records_null"] == {}
    assert out["compare"]["n"] == len(Z.printed) and "draw_q5" not in out["compare"]["differ"]
    doc = z_store.read_summary()
    assert doc["budget"]["records_ledger"][-1]["stage"] == "ydiag"


def test_reuse_breaks(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("ydiag")
    w.ydoc["pilot"]["admitted"] = list(Z.y_admitted[:5])
    w.decl_change = ["flymon/brain/y_oc.py"]
    out = w.run("reuse")
    assert out["outcome"] == R.STOP_REUSE and out["where"] == "0c"
    assert any("입장" in x for x in out["reasons"]) and any("y_oc.py" in x for x in out["reasons"])


def test_reuse_oracle_count(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("ydiag")
    out = w.run("reuse", dataclasses.replace(w.zs, n_len=30))
    assert out["outcome"] == R.STOP_REUSE and any("oracle.json" in x for x in out["reasons"])


def test_sens_rule_a_stop(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    w.L = {(12, 1): 0.79, (24, 1): 0.7999}
    out = w.run("sens")
    assert out["outcome"] == R.STOP_PLAN_UNREACHABLE and out["where"] == "0c′"
    lim = {(x["n"], x["delta_index"]): x["limit"] for x in out["limits"]}
    assert lim[(12, 1)] == pytest.approx(0.79) and lim[(24, 0)] == pytest.approx(0.95)
    with pytest.raises(SystemExit):
        w.run("split")                                              # Z stops at a STOP


def test_split_rule_b_stop_at_n_star(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.L = {(12, 1): 0.79, (24, 1): 0.95}                           # (가) passes (24 reaches the bar)
    w.chain("sens")
    out = w.run("split")
    pre = out["precheck"]
    assert pre["j_max"] == 6 + pre["h"]
    want = R.STOP_PLAN_UNREACHABLE if pre["j_max"] < 24 else R.PASS
    assert out["outcome"] == want


def test_split_pilot_few(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("sens")
    monkeypatch.setattr(ZR.z_split, "split", lambda items, forced, root, strata=("b", "a"): dict(
        C=[k for k, _c, _a in items[:26]], P=[k for k, _c, _a in items[26:]], C_c=[], P_c=[], forced_n=3))
    out = w.run("split")
    assert out["outcome"] == R.STOP_PILOT_FEW and out["where"] == "0d" and out["precheck"]["j_max"] == 11


def test_chain_refusals(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        w.run("ydiag")                                              # needs stage0
    w.run("stage0")
    with pytest.raises(SystemExit):
        w.run("stage0")                                             # never rewritten
    monkeypatch.setattr(ZR, "summary_git", lambda p: dict(tracked=True, dirty=True, judged=[]))
    with pytest.raises(SystemExit):
        w.run("ydiag")


def test_later_stage_dispatch(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("split")
    with pytest.raises(SystemExit) as e:
        w.run("pilot")
    assert e.value.code == 2
    import sys
    import types
    m = types.ModuleType("z_fake_later")
    m.STAGES = dict(pilot=lambda runner: dict(outcome="PASS", via="later"))
    monkeypatch.setitem(sys.modules, "z_fake_later", m)
    assert w.run("pilot", dataclasses.replace(w.zs, later_modules=("z_fake_later",)))["via"] == "later"


def test_env_mismatch_at_start_and_resume(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run("stage0")
    real = ZR.env_now()
    monkeypatch.setattr(ZR, "env_now", lambda: dict(real, numpy="0.0"))
    with pytest.raises(SystemExit) as e:
        w.run("ydiag")
    assert e.value.code == ZR.EXIT_ENV
    led = z_store.read_summary()["budget"]["ledger"]
    assert led[-1]["stage"] == "ydiag" and "stage0 대비 numpy" in led[-1]["env_mismatch"]
    assert "ydiag" not in z_store.read_summary() and not Path(f"{Z.progress_dir}/ydiag.env.json").exists()


def test_env_resume_compares_first_start(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    real = ZR.env_now()
    Path(Z.progress_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{Z.progress_dir}/stage0.env.json").write_text(json.dumps(dict(real, scipy="0")))
    with pytest.raises(SystemExit) as e:
        w.run("stage0")
    assert e.value.code == ZR.EXIT_ENV


def test_added_z_file_is_not_a_mismatch(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run("stage0")
    real = ZR.env_now()
    monkeypatch.setattr(ZR, "env_now", lambda: dict(real, z_files=dict(real["z_files"], **{"flymon/brain/z_new.py": "1"})))
    assert w.run("ydiag")["outcome"] == R.PASS


def test_t14_no_monkeypatching_and_y_untouched(tmp_path, monkeypatch):
    for p in [*sorted((ROOT / "flymon/brain").glob("z_*.py")), ROOT / "scripts/run_z.py"]:
        tree = ast.parse(p.read_text())
        src = p.read_text()
        assert "setattr(" not in src and "mock" not in src and "reload(" not in src and "monkeypatch" not in src, p
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for a in ast.walk(t):
                        if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name):
                            assert not a.value.id.startswith(("y_", "x_", "w_", "Y_", "X_", "W_")), (p, a.value.id)
    before = (y_store.ALLOWED_DIR, y_store.SUMMARY, y_spec.SPEC == y_spec.YSpec())
    w = World(tmp_path, monkeypatch)
    ysha = {str(p): sha256_file(p) for p in Path("results/y").rglob("*") if p.is_file()}
    w.chain("split")
    assert (y_store.ALLOWED_DIR, y_store.SUMMARY, y_spec.SPEC == y_spec.YSpec()) == before == (
        "results/y/", "results/summary/y_learning.json", True)
    assert {str(p): sha256_file(p) for p in Path("results/y").rglob("*") if p.is_file()} == ysha
    assert not Path(Y.summary).exists()


class Killed(Exception):
    pass


def test_sens_resumes_from_cells(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("reuse")
    calls, state = [], dict(kill=True)
    real = O.sens_job

    def counting(*a):
        calls.append(a[3:6])
        if state["kill"] and len(calls) == 7:
            state["kill"] = False
            raise Killed                                            # the run is killed at the 7th cell
        return real(*a)
    monkeypatch.setattr(O, "sens_job", counting)
    with pytest.raises(Killed):
        w.run("sens")
    assert "sens" not in z_store.read_summary() and len(calls) == 7
    calls.clear()
    out = w.run("sens")
    assert out["outcome"] == R.PASS and len(calls) == len(Z.sens_ns) * len(Z.sens_deltas) * w.zs.sens_draws - 6
    assert z_store.read_summary()["budget"]["ledger"][-1]["wall_s"] >= 0.0


def test_split_refuses_when_oracle_changed_after_reuse(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.chain("sens")
    p = Path(Y.oracle_detail)
    p.write_text(p.read_text() + " ")
    with pytest.raises(SystemExit) as e:
        w.run("split")
    assert e.value.code == 2 and not Path(Z.split_detail).exists() and "split" not in z_store.read_summary()
