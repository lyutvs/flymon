"""ab_runner (B1, plan Task 8a): the pooled checkpointing cell executor, cal_gate (0e: reservation incl. the 0f term,
AB.5 on g 5 · 6 · 7, STOP_CALIBRATION, the measured cap, the synthetic validation, resume bit for bit) and futility
(0f, AB.9.3: reservation, the AA source bit check, the point decision on g 6 · icc at 0e's levels, STOP_FUTILE with the
record grid on the records ledger and its cap, resume bit for bit), and the chain through 0f with restart_preseal
reproducing 0e / 0f bit for bit (Operator §7 at world scale). Test scale: n_sel / n_ver 10, B 200, workers 1 (one
test runs the spawn pool); every rule number unchanged except the stated targets."""
import dataclasses
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import ab_estimate as E
from flymon.brain import ab_rules, ab_store
from flymon.brain import ab_runner as R
from flymon.brain.r_pairs import row_key
from tests.brain import ab_world
from tests.brain.ab_world import World, doc
from tests.brain.test_ab_rules import _pattern, _quoted
from tests.brain.w_world import Model

ROOT = Path(__file__).resolve().parents[2]
_cli = importlib.util.spec_from_file_location("run_ab", ROOT / "scripts/run_ab.py")
run_ab = importlib.util.module_from_spec(_cli)
_cli.loader.exec_module(run_ab)

FAST = dict(n_sel=10, n_ver=10, workers=1)
OK = dict(FAST, p_target=0.9, f_target=0.9)                     # test scale: every structure reaches a verified α
H = 3600.0


def _code(fn):
    with pytest.raises(SystemExit) as e:
        fn()
    return e.value.code


class NoisyModel(Model):
    """w_world's count model plus stage-dependent count noise (seeded by probe seed and stage), so the AA source's
    gates are not all clipped at ±10 and its raw contrasts vary between pairs."""

    def probe(self, pair, seed, edit, applied, plastic):
        x, y = super().probe(pair, seed, edit, applied, plastic)
        n = np.random.default_rng([int(seed), len(applied)]).normal(0.0, 4.0, 4)
        add = lambda d, a, p: dict(d, A=int(max(0, d["A"] + round(a))), P=int(max(0, d["P"] + round(p))))  # noqa
        return add(x, n[0], n[1]), add(y, n[2], n[3])


class NoisyWorld(World):
    """ab_world.World with per-pair effects and NoisyModel for AA's raw. The plain world's AA S1 (one effect for every
    pair, deterministic contrasts) has three gates clipped at ±10 and constant raw contrasts in every pair, so
    fut_model's 8 × 8 correlation is NaN and its Cholesky fails; here fut_check still holds (the records are computed
    from the same raw). With 6 source pairs C has rank ≤ 5 and is PD only through fut_chol_eps — a world-scale fact."""

    def _write_aa(self):
        self.model.__class__ = NoisyModel
        for i, r in enumerate(ab_world.AA_ROWS):
            self.model.effects[row_key(r)] = (1.0 + 0.6 * i, 1.0 + 0.5 * ((i * 3) % 5))
        super()._write_aa()


def _w(tmp_path, monkeypatch, sub=None, **kw):
    d = tmp_path / sub if sub else tmp_path
    d.mkdir(exist_ok=True)
    w = NoisyWorld(d, monkeypatch)
    w.s = dataclasses.replace(w.s, **kw)
    return w


def _ab8(prefix: str, sentence: str) -> bool:
    body = sentence.split(" (후보 상태")[0].replace("**", "")
    return re.fullmatch(_pattern(_quoted(prefix)), body, re.S) is not None


def _ledger(which="ledger"):
    return [e for e in doc().get("budget", {}).get(which, [])]


def _add_core(w, hours):
    ab_store.append_ledger(w.s.summary, dict(stage="test_extra", wall_s=hours * H), [w.ctx["params"]()])


def _note(tmp_path):
    p = tmp_path / "note.json"
    p.write_text(json.dumps(dict(symptom="Stage II code added (staged build)", clause="AB.7 0 (order-0 completeness)",
                                 cause="plan reading 33")))
    return str(p)


class _Kill(Exception):
    pass


CAL_KEYS = ("structures", "synth", "synth_reason")
FUT_KEYS = ("model", "rows", "judge", "alpha", "source")


# ================================================================ 0e cal_gate
def test_cal_gate_pass_block(tmp_path, monkeypatch):
    """Serial and spawn-pool runs give the same block; every structure carries α and every cell × grid level."""
    got = {}
    for sub, workers in (("serial", 1), ("spawn", 2)):
        w = _w(tmp_path, monkeypatch, sub, **dict(OK, workers=workers))
        w.chain("seal_code")
        out = w.runner().run("cal_gate")
        assert out["outcome"] == ab_rules.PASS and run_ab.exit_code(out) == 0, out.get("sentence")
        d = doc()["cal_gate"]
        got[sub] = {k: d[k] for k in CAL_KEYS}
        s = w.s
        assert set(d["structures"]) == {"g5", "g6", "g7"}
        for tag, sm in d["structures"].items():
            assert sm["pass_ok"] and all(sm["alpha"][n]["P"] in s.p_grid for n in E.STATS)
            assert sm["sizes"] == list(s.rep_sizes(tag)) and sm["fail_flag"] is None
            assert sorted(map(int, sm["sel_counts"])) == list(range(1, 25)) and len(sm["truth"]) == 24
            for c in sm["sel_counts"].values():
                assert len(c["D"]["P"]) == len(c["Dfin"]["P"]) == len(c["R"]["P"]) == len(s.p_grid)
                assert len(c["D"]["F"]) == len(s.f_grid) and {"P_TS", "P_CG", "F_TS", "F_CG"} <= set(c["D"])
            assert sorted(map(int, sm["ver_counts"])) == list(range(1, 25)) and sm["n_sel"] == sm["n_ver"] == 10
        syn = d["synth"]
        assert d["synth_reason"] is None and syn["label"] == "기록 — 방법 · 수치를 바꾸지 않는다"
        assert set(syn["results"]) == {"bar_one", "bar_all", "eff1.5", "eff2.5", "eff3.5"}
        assert all(sum(v["counts"].values()) == s.synth_reps for c in syn["results"].values() for v in c.values())
        assert set(syn["results"]["eff2.5"]) == {"1", "4", "12", "19"}
        est = d["estimate_h"]
        assert est["total"] == pytest.approx(est["cal"] + est["synth"] + est["futility"] + est["rest"])
        assert est["futility"] > 0 and d["seal_key"] == doc()["seal_code"]["seal"]["key"]
        assert not Path(s.cache_dir).exists()                                   # no Gen-2 measurement
        cal = Path(s.cal_dir)
        assert len(list(cal.glob("g*/*.json"))) == 3 * 3 * 24 and len(list(cal.glob("synth/*.json"))) == 20
        assert json.loads((cal / "g6" / "ver_24.json").read_text())["final"] is True
        assert [e for e in _ledger() if e["stage"] == "cal_gate"][0]["wall_s"] > 0
        assert not [e for e in _ledger("records_ledger") if e["stage"] == "cal_gate"]
    assert R._same(got["serial"], got["spawn"])


def test_cal_gate_stop_calibration(tmp_path, monkeypatch, capsys):
    """p_target 0.025 at n_sel 10: no α passes the CP rule → STOP_CALIBRATION〈0e〉 (the mutant "0e를 기록 전용으로"
    returns PASS and fails here); then futility and kc_input are refused."""
    w = _w(tmp_path, monkeypatch, **FAST)
    w.chain("seal_code")
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.STOP_CALIBRATION and out["stop_stage"] == "0e" and run_ab.exit_code(out) == 3
    assert len(out["reasons"]) == 9 and out["reasons"][0].startswith("g5 D sel 칸 ")
    f = out["failures"][0]
    assert (f["tag"], f["stat"], f["phase"], f["n"]) == ("g5", "D", "sel", 10)
    sm = doc()["cal_gate"]["structures"]["g5"]
    assert f["x"] == sm["sel_counts"][str(f["cell"])]["D"]["P"][-1]
    assert f["cp"] == max(v[-1] for v in sm["sel_cp"]["D"]["P"].values())
    assert _ab8("- **`STOP_CALIBRATION`**〈0e〉:", out["sentence"]) and ab_rules.CLOSURE in out["sentence"]
    assert "대표 구조 g 5, X 라벨 묶음 크기 (5, 4, 3, 3, 2)의 D" in out["sentence"]
    assert out["synth"] is None and "합성 검증을 하지 않음" in out["synth_reason"]
    assert not Path(w.s.cache_dir).exists()
    assert _code(lambda: w.runner().run("futility")) == 2
    assert "AB stops there" in capsys.readouterr().err
    assert _code(lambda: w.runner()._require("kc_input")) == 2


def test_cal_gate_stop_calibration_on_verification(tmp_path, monkeypatch):
    """Only g 5's R side fails its verification → STOP_CALIBRATION with the chosen α and the worst cell of "ver"."""
    w = _w(tmp_path, monkeypatch, **OK)
    w.chain("seal_code")
    real = E.calibrate_many

    def cm(items, s, cell):
        out = real(items, s, cell)
        out[0]["alpha"]["R"]["P"] = None
        out[0]["verified"]["R"]["P"]["ok"] = False
        out[0]["pass_ok"] = False
        return out
    monkeypatch.setattr(E, "calibrate_many", cm)
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.STOP_CALIBRATION and out["reasons"] == [out["reasons"][0]]
    f = out["failures"][0]
    v = doc()["cal_gate"]["structures"]["g5"]["verified"]["R"]["P"]
    assert (f["tag"], f["stat"], f["phase"], f["cell"], f["x"], f["n"]) == ("g5", "R", "ver", v["worst"]["cell"],
                                                                            v["worst"]["x"], 10)
    assert f["alpha"] == v["alpha"] and f"고른 α {v['alpha']}에서 칸 {v['worst']['cell']}" in out["sentence"]
    assert out["synth"] is not None                                             # g 7 passed: the record runs


def _run_cal(w):
    w.chain("seal_code")
    return w.runner().run("cal_gate")


def test_cal_gate_resumes_checkpoints_bit_equal(tmp_path, monkeypatch):
    """A kill in the middle of the 5th selection job (after its first chunk) → rerun reads the 72 truths and four
    selection cells back, resumes the 5th from its PCG64 state, and the block equals an uninterrupted run's."""
    w = _w(tmp_path, monkeypatch, "killed", **OK)
    w.chain("seal_code")
    real, calls = E.cal_run, []

    def kill_5th(*a, **kw):
        calls.append(a[2:4])
        if len(calls) == 5:
            on = kw["on_chunk"]

            def boom(pay):
                on(pay)
                raise _Kill()
            kw = dict(kw, on_chunk=boom)
        return real(*a, **kw)
    monkeypatch.setattr(E, "cal_run", kill_5th)
    with pytest.raises(_Kill):
        w.runner().run("cal_gate")
    part = json.loads((Path(w.s.cal_dir) / "g5" / "sel_5.json").read_text())
    assert part["final"] is False and part["value"]["done"] == w.s.cal_chunk
    assert len(list(Path(w.s.cal_dir).glob("g*/truth_*.json"))) == 72 and "cal_gate" not in doc()
    seen, truths = [], []
    real_truth = E.truth

    def count(*a, **kw):
        seen.append((a[2], a[3], (kw.get("resume") or {}).get("done")))
        return real(*a, **kw)
    monkeypatch.setattr(E, "cal_run", count)
    monkeypatch.setattr(E, "truth", lambda *a, **kw: truths.append(a) or real_truth(*a, **kw))
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.PASS and truths == []
    assert len(seen) == (72 - 4) + 72 and seen[0] == ("sel", 5, w.s.cal_chunk)       # only the missing jobs
    killed = {k: doc()["cal_gate"][k] for k in CAL_KEYS}
    monkeypatch.setattr(E, "cal_run", real)
    monkeypatch.setattr(E, "truth", real_truth)
    w2 = _w(tmp_path, monkeypatch, "clean", **OK)
    assert _run_cal(w2)["outcome"] == ab_rules.PASS
    assert R._same(killed, {k: doc()["cal_gate"][k] for k in CAL_KEYS})


def test_cal_gate_reservation_stop_before_start(tmp_path, monkeypatch):
    w = _w(tmp_path, monkeypatch, **OK)
    w.chain("seal_code")
    _add_core(w, 23.0)
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0e 전" and run_ab.exit_code(out) == 3
    assert "— 2세대 측정 없음. 사용자 몫." in out["sentence"] and " h + 2.0 × " in out["sentence"]
    assert _ab8("- **`STOP_BUDGET`**(0e 전", out["sentence"])
    assert set(out["estimate_h"]) >= {"cal", "synth", "futility", "rest", "total"}
    assert not Path(w.s.cal_dir).exists()


def test_cal_gate_reservation_includes_futility(tmp_path, monkeypatch):
    """A core ledger where the reservation passes without the 0f term and fails with it → STOP_BUDGET〈0e 전〉."""
    w = _w(tmp_path, monkeypatch, **OK)
    w.chain("seal_code")
    r = w.runner()
    est = r._cal_est(doc())
    core = r._core_h(doc())
    no_fut = est["cal"] + est["synth"] + est["rest"]                                  # the terms, read one by one
    target = w.s.core_cap_h - 2.0 * no_fut - est["futility"]                       # passes by F, fails by F
    _add_core(w, target - core)
    c2 = r._core_h(doc())
    assert est["futility"] > 1e-6
    assert ab_rules.core_ok(c2, no_fut, w.s) and not ab_rules.core_ok(c2, est["total"], w.s)
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0e 전"
    assert out["estimate_h"]["futility"] == est["futility"]


def test_cal_gate_measured_cap(tmp_path, monkeypatch):
    """The measured core (ledger + this stage's progress) past 24 h between cells → STOP_BUDGET〈0e〉; every
    checkpoint written so far stays byte for byte."""
    w = _w(tmp_path, monkeypatch, **OK)
    w.chain("seal_code")
    real, n = E.truth, []

    def kill(*a, **kw):
        n.append(1)
        if len(n) == 30:
            raise _Kill()
        return real(*a, **kw)
    monkeypatch.setattr(E, "truth", kill)
    with pytest.raises(_Kill):
        w.runner().run("cal_gate")
    snap = {p: p.read_bytes() for p in Path(w.s.cal_dir).rglob("*.json")}
    assert len(snap) == 29
    prog = Path(w.s.progress_dir) / "cal_gate.json"
    bumped = []

    def bump(*a, **kw):                                  # past the reservation: the first new cell of the resumed run
        if not bumped:                                   # finds 0e has run long (the reservation counts progress
            bumped.append(1)                             # seconds, so the bump must come after it)
            d = json.loads(prog.read_text())
            ab_store.write_json(str(prog), dict(d, core=d["core"] + 24.5 * H), [w.ctx["params"]()])
        return real(*a, **kw)
    monkeypatch.setattr(E, "truth", bump)
    out = w.runner().run("cal_gate")
    monkeypatch.setattr(E, "truth", real)
    assert bumped
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0e" and run_ab.exit_code(out) == 3
    assert "실측 누적 " in out["sentence"] and "— 2세대 측정 없음. 사용자 몫." in out["sentence"]
    assert _ab8("- **`STOP_BUDGET`**(0e 전", out["sentence"])
    assert all(p.read_bytes() == v for p, v in snap.items())                  # the earlier checkpoints stay
    assert len(list(Path(w.s.cal_dir).rglob("*.json"))) == 30                  # + the one cell in flight at the bump
    assert [e for e in _ledger() if e["stage"] == "cal_gate"][0]["wall_s"] > 24.5 * H


def test_cal_gate_resumed_reservation_counts_killed_progress(tmp_path, monkeypatch):
    """AB.7 0e: the reservation ("0e 전") is decided once, on the first start, and counts this stage's progress
    seconds already there; it is written to progress/cal_gate.reserve.json. A resume after a kill does not re-reserve
    (re-adding the full estimate on top of the killed run's seconds would stop a late resume spuriously); the
    measured cap ("0e 실행 중", ledger + every run's progress) still stops it once the total passes 24 h."""
    # (1) first start: earlier progress seconds count in the reservation → STOP_BUDGET〈0e 전〉, nothing measured
    w = _w(tmp_path, monkeypatch, "first", **OK)
    w.chain("seal_code")
    r = w.runner()
    est, core = r._cal_est(doc()), r._core_h(doc())
    assert ab_rules.core_ok(core, est["total"], w.s)                           # the ledger alone passes
    assert not ab_rules.core_ok(core + 23.5, est["total"], w.s) and core + 23.5 < w.s.core_cap_h
    prog = Path(w.s.progress_dir) / "cal_gate.json"
    ab_store.write_json(str(prog), dict(core=23.5 * H, records=0.0), [w.ctx["params"]()])
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0e 전" and run_ab.exit_code(out) == 3
    assert not Path(w.s.cal_dir).exists()
    res = json.loads((Path(w.s.progress_dir) / "cal_gate.reserve.json").read_text())
    assert res["ok"] is False and res["seal_key"] == doc()["seal_code"]["seal"]["key"]

    # (2) killed after the reservation; the killed run's spend makes a re-reservation fail, the measured cap not
    w = _w(tmp_path, monkeypatch, "resume", **OK)
    w.chain("seal_code")
    real, n = E.truth, []

    def kill(*a, **kw):
        n.append(1)
        if len(n) == 5:
            raise _Kill()
        return real(*a, **kw)
    monkeypatch.setattr(E, "truth", kill)
    with pytest.raises(_Kill):
        w.runner().run("cal_gate")
    monkeypatch.setattr(E, "truth", real)
    rp = Path(w.s.progress_dir) / "cal_gate.reserve.json"
    rbytes = rp.read_bytes()
    assert json.loads(rbytes)["ok"] is True
    snap = {p: p.read_bytes() for p in Path(w.s.cal_dir).rglob("*.json")}
    assert len(snap) == 4
    prog = Path(w.s.progress_dir) / "cal_gate.json"
    d = json.loads(prog.read_text())
    ab_store.write_json(str(prog), dict(d, core=d["core"] + 23.5 * H), [w.ctx["params"]()])   # the killed run's spend
    r = w.runner()
    est, spent = r._cal_est(doc()), r._spent_core_h(doc(), "cal_gate")
    assert not ab_rules.core_ok(spent, est["total"], w.s) and ab_rules.spent_ok(spent, w.s)
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.PASS and run_ab.exit_code(out) == 0                # no re-reservation
    assert rp.read_bytes() == rbytes
    assert all(p.read_bytes() == v for p, v in snap.items())

    # (3) killed after the reservation; the measured total past 24 h → STOP_BUDGET〈0e〉 on resume, no new checkpoint
    w = _w(tmp_path, monkeypatch, "cap", **OK)
    w.chain("seal_code")
    n.clear()
    monkeypatch.setattr(E, "truth", kill)
    with pytest.raises(_Kill):
        w.runner().run("cal_gate")
    monkeypatch.setattr(E, "truth", real)
    snap = {p: p.read_bytes() for p in Path(w.s.cal_dir).rglob("*.json")}
    prog = Path(w.s.progress_dir) / "cal_gate.json"
    d = json.loads(prog.read_text())
    ab_store.write_json(str(prog), dict(d, core=d["core"] + 24.5 * H), [w.ctx["params"]()])
    out = w.runner().run("cal_gate")
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0e" and run_ab.exit_code(out) == 3
    assert {p: p.read_bytes() for p in Path(w.s.cal_dir).rglob("*.json")} == snap


def test_cal_gate_fail_side_unreached_is_flag_not_stop(tmp_path, monkeypatch):
    w = _w(tmp_path, monkeypatch, **dict(OK, f_target=1e-9))
    out = _run_cal(w)
    assert out["outcome"] == ab_rules.PASS
    for sm in doc()["cal_gate"]["structures"].values():
        assert sm["alpha"]["D"]["F"] is None and not sm["fail_ok"] and sm["fail_flag"] == "FAIL 쪽 미도달 위험"
        assert sm["pass_ok"]
    assert doc()["cal_gate"]["synth"]["alpha"]["F"] is None


def test_cal_gate_checks_seal_first(tmp_path, monkeypatch):
    w = _w(tmp_path, monkeypatch, **OK)
    w.chain("seal_code")
    w.s = dataclasses.replace(w.s, fut_reps=w.s.fut_reps + 1)
    assert _code(lambda: w.runner().run("cal_gate")) == R.EXIT_SEAL
    assert not Path(w.s.cal_dir).exists() and "cal_gate" not in doc()


def test_job_paths_and_keys(tmp_path, monkeypatch):
    s = _w(tmp_path, monkeypatch, **OK).s
    st = E.rep_structure(s.rep_sizes("g6"), s)
    assert R.job_path("r", "truth", (st, "g6", 3, s)) == "r/g6/truth_3.json"
    assert R.job_path("r", "sel", (st, "g6", "sel", 4, {}, s)) == "r/g6/sel_4.json"
    assert R.job_path("r", "fut", (st, "g6", "icc", {}, {}, s)) == "r/fut_g6_icc.json"
    assert R.job_path("r", "fut", (st, "g5", "icc", {}, {}, s, 2, None, None, None, ["fut", "grid", "all", 5, 8])) \
        == "r/fut_grid_all_5_8.json"
    assert R.job_path("r", "synth", (st, "eff2.5", 12, (2.5,) * 4, {}, s)) == "r/synth/eff2.5_12.json"
    k = R.job_key("sel", (st, "g6", "sel", 4, {}, s), "x")
    assert k == R.job_key("sel", (st, "g6", "sel", 4, {}, dataclasses.replace(s, cli_print_chars=3)), "x")
    assert k != R.job_key("sel", (st, "g6", "sel", 4, {}, s), "y") != R.job_key("sel", (st, "g6", "sel", 5, {}, s), "x")


# ================================================================ 0f futility
def _fut_world(tmp_path, monkeypatch, sub=None, **kw):
    w = _w(tmp_path, monkeypatch, sub, **dict(OK, **kw))
    w.chain("cal_gate")
    return w


def test_futility_pass_block(tmp_path, monkeypatch):
    w = _fut_world(tmp_path, monkeypatch, fut_threshold=0.0)
    full = doc()                                    # 0e's levels differ by structure and statistic (the mutant
    for i, tag in enumerate(w.s.fut_tags):          # "0e 수준 대신 고정 α" cannot match them all)
        full["cal_gate"]["structures"][tag]["alpha"] = {"D": {"P": w.s.p_grid[1 + i], "F": None},
                                                        "Dfin": {"P": w.s.p_grid[2 + i]}, "R": {"P": w.s.p_grid[3 + i]}}
    Path(w.s.summary).write_text(json.dumps(full))
    out = w.runner().run("futility")
    assert out["outcome"] == ab_rules.PASS and run_ab.exit_code(out) == 0
    d = doc()["futility"]
    cal = doc()["cal_gate"]["structures"]
    assert d["alpha"]["g6"] == dict(D=w.s.p_grid[2], Dfin=w.s.p_grid[3], R=w.s.p_grid[4], F=None)
    assert len(d["rows"]) == 9 and d["judge"]["row"] == "g6|icc" and d["judge"]["futile"] is False
    assert all(r["n"] == w.s.fut_reps and r["alpha"] == E.fut_alpha(cal[r["tag"]]["alpha"]) for r in d["rows"].values())
    assert d["alpha"] == {t: E.fut_alpha(cal[t]["alpha"]) for t in w.s.fut_tags}           # 0e's levels, not fixed
    assert d["grid"] is None and d["grid_reason"].startswith("STOP_FUTILE이 아니므로")
    assert d["source"]["n_units"] == w.s.fut_src_units and d["source"]["missing"] == []
    assert set(d["model"]) == {"keys", "mu", "tau", "tau_src", "rho", "C", "x_sizes"}
    assert not Path(w.s.cache_dir).exists() and (Path(w.s.fut_dir) / "fut_g6_icc.json").exists()
    assert len(list(Path(w.s.fut_dir).glob("*.json"))) == 9
    assert json.loads(Path(w.s.fut_detail).read_text())["grid"] is None
    assert [e for e in _ledger() if e["stage"] == "futility"][0]["wall_s"] > 0
    assert not [e for e in _ledger("records_ledger") if e["stage"] == "futility"]


def test_futility_stop_futile(tmp_path, monkeypatch):
    """fut_threshold 1.01: P̂ < 1.01 always → STOP_FUTILE〈0f〉 with the record grid (the mutant fut_record_only — the
    STOP_FUTILE branch → PASS — is killed here); closure afterwards."""
    w = _fut_world(tmp_path, monkeypatch, fut_threshold=1.01)
    out = w.runner().run("futility")
    assert out["outcome"] == ab_rules.STOP_FUTILE and out["stop_stage"] == "0f" and run_ab.exit_code(out) == 3
    jr = out["rows"]["g6|icc"]
    frac = f"({jr['passed']}/{jr['n']}, CP 95%"
    assert frac in out["sentence"] and ab_rules.CLOSURE in out["sentence"]
    at_scale = out["sentence"].replace(frac, f"({jr['passed']}/2 000, CP 95%")      # AB.8's text has M = 2 000
    assert _ab8("- **`STOP_FUTILE`**〈0f〉:", at_scale)
    assert "all 없음; noskew 없음; sd2 없음; sd1 없음" in out["sentence"]
    assert set(out["grid"]) == {"all", "noskew", "sd2", "sd1"}
    assert all(v["pareto"] == [] and v["cells"] == v["done"] == 4 for v in out["grid"].values())
    det = json.loads(Path(w.s.fut_detail).read_text())
    assert [(c["g"], c["k"]) for c in det["grid"]["all"]["cells"]] == [(5, 8), (5, 12), (6, 8), (6, 12)]
    assert all(c["n"] == w.s.fut_grid_reps for v in det["grid"].values() for c in v["cells"])
    assert (Path(w.s.fut_dir) / "fut_grid_sd1_6_12.json").exists()
    assert [e for e in _ledger("records_ledger") if e["stage"] == "futility"][0]["wall_s"] > 0
    assert [e for e in _ledger() if e["stage"] == "futility"][0]["wall_s"] > 0
    assert _code(lambda: w.runner()._require("kc_input")) == 2


def test_futility_resumes_checkpoints_bit_equal(tmp_path, monkeypatch):
    w = _fut_world(tmp_path, monkeypatch, "killed", fut_threshold=0.0)
    real, calls = E.fut_run, []

    def kill(*a, **kw):
        calls.append(a[1:3])
        if len(calls) == 4:
            raise _Kill()
        return real(*a, **kw)
    monkeypatch.setattr(E, "fut_run", kill)
    with pytest.raises(_Kill):
        w.runner().run("futility")
    assert len(list(Path(w.s.fut_dir).glob("*.json"))) == 3 and "futility" not in doc()
    calls.clear()
    monkeypatch.setattr(E, "fut_run", lambda *a, **kw: calls.append(a[1:3]) or real(*a, **kw))
    assert w.runner().run("futility")["outcome"] == ab_rules.PASS
    assert len(calls) == 6 and calls[0] == ("g6", "icc")                                 # only the missing rows
    killed = {k: doc()["futility"][k] for k in FUT_KEYS}
    monkeypatch.setattr(E, "fut_run", real)
    w2 = _fut_world(tmp_path, monkeypatch, "clean", fut_threshold=0.0)
    assert w2.runner().run("futility")["outcome"] == ab_rules.PASS
    assert R._same(killed, {k: doc()["futility"][k] for k in FUT_KEYS})


def test_futility_reservation_stop(tmp_path, monkeypatch):
    w = _fut_world(tmp_path, monkeypatch, fut_threshold=0.0)
    _add_core(w, 23.0)                                     # as if 0e had run far longer than its estimate
    out = w.runner().run("futility")
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0f 전" and run_ab.exit_code(out) == 3
    assert "— 2세대 측정 없음. 사용자 몫." in out["sentence"] and _ab8("- **`STOP_BUDGET`**(0e 전", out["sentence"])
    assert set(out["estimate_h"]) >= {"futility", "rest", "total"}
    assert not Path(w.s.fut_dir).exists()


def test_futility_resumed_reservation_counts_killed_progress(tmp_path, monkeypatch):
    """AB.7 0f: the reservation ("0f 전") is decided once, on the first start, and counts this stage's progress
    seconds already there (progress/futility.reserve.json); a resume after a kill does not re-reserve — the killed
    run's seconds stay in the ledger (T8a review), and 0f has no measured cap (AB.7 예산)."""
    # (1) first start: earlier progress seconds count → STOP_BUDGET〈0f 전〉, nothing measured
    w = _fut_world(tmp_path, monkeypatch, "first", fut_threshold=0.0)
    r = w.runner()
    core = r._core_h(doc())
    est = r._fut_est_h(doc()) + r._rest_terms(doc())["total"]
    assert ab_rules.core_ok(core, est, w.s) and not ab_rules.core_ok(core + 23.5, est, w.s)
    prog = Path(w.s.progress_dir) / "futility.json"
    ab_store.write_json(str(prog), dict(core=23.5 * H, records=0.0), [w.ctx["params"]()])
    out = w.runner().run("futility")
    assert out["outcome"] == ab_rules.STOP_BUDGET and out["stop_stage"] == "0f 전" and run_ab.exit_code(out) == 3
    assert not Path(w.s.fut_dir).exists()
    assert json.loads((Path(w.s.progress_dir) / "futility.reserve.json").read_text())["ok"] is False

    # (2) killed after the reservation; a re-reservation would fail, the resume does not re-reserve
    w = _fut_world(tmp_path, monkeypatch, "resume", fut_threshold=0.0)
    real, calls = E.fut_run, []

    def kill(*a, **kw):
        calls.append(1)
        if len(calls) == 3:
            raise _Kill()
        return real(*a, **kw)
    monkeypatch.setattr(E, "fut_run", kill)
    with pytest.raises(_Kill):
        w.runner().run("futility")
    monkeypatch.setattr(E, "fut_run", real)
    rp = Path(w.s.progress_dir) / "futility.reserve.json"
    rbytes = rp.read_bytes()
    assert json.loads(rbytes)["ok"] is True
    snap = {p: p.read_bytes() for p in Path(w.s.fut_dir).glob("*.json")}
    assert len(snap) == 2 and "futility" not in doc()
    prog = Path(w.s.progress_dir) / "futility.json"
    d = json.loads(prog.read_text())
    assert d["core"] > 0.0                                                     # the killed run's seconds are there
    ab_store.write_json(str(prog), dict(d, core=d["core"] + 23.5 * H), [w.ctx["params"]()])
    r = w.runner()
    assert not ab_rules.core_ok(r._spent_core_h(doc(), "futility"), est, w.s)
    out = w.runner().run("futility")
    assert out["outcome"] == ab_rules.PASS and run_ab.exit_code(out) == 0
    assert rp.read_bytes() == rbytes
    assert all(p.read_bytes() == v for p, v in snap.items())
    assert [e for e in _ledger() if e["stage"] == "futility"][0]["wall_s"] > 23.5 * H   # the killed spend counts


def test_futility_records_cap_nulls(tmp_path, monkeypatch):
    w = _fut_world(tmp_path, monkeypatch, fut_threshold=1.01, records_cap_h=1e-6)
    out = w.runner().run("futility")
    assert out["outcome"] == ab_rules.STOP_FUTILE
    g = out["grid"]
    assert g["all"]["pareto"] == [] and g["all"]["done"] == 4
    for sc in ("noskew", "sd2", "sd1"):
        assert g[sc]["pareto"] is None and g[sc]["done"] == 0 and g[sc]["reason"] == ab_rules.RECORDS_REASON
    det = json.loads(Path(w.s.fut_detail).read_text())["grid"]
    assert all(c["reason"] == ab_rules.RECORDS_REASON for c in det["sd1"]["cells"])
    assert "all 없음; noskew records 상한; sd2 records 상한; sd1 records 상한" in out["sentence"]


def test_futility_source_mismatch_refuses(tmp_path, monkeypatch, capsys):
    w = _fut_world(tmp_path, monkeypatch, fut_threshold=0.0)
    p = Path(w.s.aa_summary)
    orig = p.read_text()
    d = json.loads(orig)
    k = d["records"]["s1_records"]["keys"][0]
    d["records"]["pairs"][k]["gates"]["reward_level"]["mean"] += 1.0
    p.write_text(json.dumps(d))
    capsys.readouterr()
    assert _code(lambda: w.runner().run("futility")) == 2
    assert "not bit-equal to AA" in capsys.readouterr().err and "futility" not in doc()
    p.write_text(orig)
    man = json.loads(Path(w.s.aa_learn_detail).read_text())["manifest"]
    Path(next(m["cache_file"] for m in man if m["key"].startswith(k + "|"))).unlink()
    assert _code(lambda: w.runner().run("futility")) == 2
    assert "differs from the reuse block" in capsys.readouterr().err and "futility" not in doc()
    assert not Path(w.s.fut_dir).exists()


def test_futility_checks_seal_first(tmp_path, monkeypatch):
    w = _fut_world(tmp_path, monkeypatch, fut_threshold=0.0)
    w.s = dataclasses.replace(w.s, fut_threshold=0.25)
    assert _code(lambda: w.runner().run("futility")) == R.EXIT_SEAL
    assert not Path(w.s.fut_dir).exists()


def test_futility_needs_cal_gate(tmp_path, monkeypatch, capsys):
    w = _w(tmp_path, monkeypatch, **OK)
    w.chain("seal_code")
    capsys.readouterr()
    assert _code(lambda: w.runner().run("futility")) == 2
    assert "needs block(s) ['cal_gate']" in capsys.readouterr().err


# ================================================================ the chain through 0f and the restart
def test_chain_through_futility(tmp_path, monkeypatch):
    """chain 0a–0f PASS; restart_preseal moves everything aside; the rerun's cal_gate and futility bodies equal the
    INVALID copies bit for bit (Operator §7 step 5 at world scale)."""
    w = _w(tmp_path, monkeypatch, **dict(OK, fut_threshold=0.0))
    assert w.chain("futility")["outcome"] == ab_rules.PASS
    monkeypatch.setattr(R, "_head_summary", lambda path: Path(path).read_bytes())          # committed at HEAD
    res = w.runner().restart_preseal(_note(tmp_path))
    assert res["outcome"] == ab_rules.PASS
    inv = doc()["restart_preseal_1"]["invalidated"]
    assert set(inv) == set(w.s.stages[:w.s.stages.index("futility") + 1])
    assert not Path(w.s.cal_dir).exists() and not Path(w.s.fut_dir).exists()
    w.write_log("1300 passed in 700.00s\nexit 0\n")
    assert w.chain("futility")["outcome"] == ab_rules.PASS
    now = doc()
    assert R._same({k: now["cal_gate"][k] for k in CAL_KEYS}, {k: inv["cal_gate"][k] for k in CAL_KEYS})
    assert R._same({k: now["futility"][k] for k in FUT_KEYS}, {k: inv["futility"][k] for k in FUT_KEYS})
    assert now["futility"]["seal_key"] == inv["futility"]["seal_key"] == now["seal_code"]["seal"]["key"]
