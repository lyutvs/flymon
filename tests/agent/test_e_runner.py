"""The stage runner on the scripted measurer and synthetic situations (no engine, no pops)."""
import dataclasses
import functools
import itertools
import json

import pytest

from flymon.agent import e_codebook, e_pairs, e_runner, e_rules, e_store
from flymon.agent.e_runner import Runner
from flymon.agent.e_spec import SPEC, smoke
from flymon.brain.h4_pairs import pool_vocabulary
from tests.agent.e_scripted import ScriptedMeasurer

GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
INFO = dict(receptor_counts={g: 1 + i % 5 for i, g in enumerate(GLOMS)}, c_norm=3.0, glomeruli=GLOMS,
            n_kc=1000, max_rate_hz=200.0, cap_hz=333.3)
ORACLE = dict(readout={"A": "MBON13", "P": "MBON05"}, z={"A": (10.0, 9.0), "P": (26.0, 19.0)},
              types=["MBON13", "MBON18", "MBON05", "MBON21"])
CLEAN = dict(commit="test", dirty_hashed=[], dirty_other=[])
SUMMARY_CLEAN = dict(tracked=True, dirty=False, judge_commits=[])
INPUTS = {"data/malecns.npz": "npzsha", "results/summary/m0d.json": "m0dsha"}
CODE = dict(key="codekey")

ST, _MI, _MON, MOVE = pool_vocabulary()
OPPS = sorted({tuple(sorted(v)) for v in ST.values()})
A_KEYS = [(m1, m2, o) for o in OPPS for m1, m2 in itertools.combinations(MOVE, 2)]
B_KEYS = [(m, o1, o2) for m in MOVE for o1, o2 in itertools.combinations(OPPS, 2) if not set(o1) & set(o2)]
E0 = {"ORN_G00": 1.0, "ORN_G01": 1.0}


def _row(axis, turn, i, mx, ox, my, oy):
    return dict(axis=axis, turn=turn, x=f"{axis}x{i}", y=f"{axis}y{i}", move_x=mx, opp_x=ox, move_y=my, opp_y=oy,
                odor_x_e0=E0, odor_y_e0=E0)


def _a(keys, turns, off):
    return [_row("a", turns[i % len(turns)], off + i, m1, o, m2, o) for i, (m1, m2, o) in enumerate(keys)]


def _b(keys, turns, off):
    return [_row("b", turns[i % len(turns)], off + i, m, o1, m, o2) for i, (m, o1, o2) in enumerate(keys)]


class FakeSituations:
    """18 (a) + 21 (b) even rows, a judgement_set-shaped dict with 21 (b) + 32 (a) rows; every E-grid key distinct;
    used = the even rows."""

    def __init__(self, n_b_judge=21):
        evt = list(range(0, 16, 2))
        self._even = _a(A_KEYS[:18], evt, 0) + _b(B_KEYS[:21], evt, 0)
        jt = list(range(64, 104))
        b, a = _b(B_KEYS[21:21 + n_b_judge], jt, 100), _a(A_KEYS[18:50], jt, 100)
        keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in b + a]
        import hashlib
        self._js = dict(b=b, a=a, n_a=len(a), last_turn=103, skipped=[{"key": ["b", 64, "q", "r"], "reason": "used"}],
                        status="OK" if len(b) == 21 else e_rules.STOP_SET_SHORT,
                        digest_e0_b="e0b", digest_e0_a="e0a",
                        digest_keys=hashlib.sha256(json.dumps(keys).encode()).hexdigest())

    def even(self):
        return [dict(r) for r in self._even]

    def judgement(self):
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self._js.items()}

    def used(self):
        return self.even()


FAKE_SITUATIONS = FakeSituations()
_real_build = e_codebook.build
_real_summary_git = e_runner.summary_git          # the autouse fixture patches e_runner.summary_git


@functools.lru_cache(maxsize=None)
def _fast(drive_items, k, iters, restarts, n_alpha_glom):
    drive = dict(drive_items)
    sp = dataclasses.replace(SPEC, anneal_iters=iters, anneal_restarts=restarts)
    from flymon.agent.encode_grid import Codebook, unique_odours
    cells = e_codebook.cells(MOVE, _MON)
    return _real_build(drive, k, sp, unique_check=lambda b: unique_odours(Codebook(cells, b), ST, MOVE))


def fast_build(drive, k, spec, unique_check=None):
    """The real build with a test-sized anneal (k3 reaches dup 0 at 5M iterations on a flat drive), memoised."""
    if spec.anneal_iters <= 20_000:                     # smoke: run as declared
        return _real_build(drive, k, spec, unique_check=unique_check)
    iters = 5_000_000 if k == 3 else 200_000
    return _fast(tuple(sorted(drive.items())), k, iters, 1, len(drive))


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(e_runner.e_codebook, "build", fast_build)
    monkeypatch.setattr(e_runner, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(e_runner, "provenance", lambda args: dict(sha256={}, args=args))
    monkeypatch.setattr(e_runner, "summary_git", lambda path: dict(SUMMARY_CLEAN))


def runner(tmp_path, monkeypatch, sm=False, testable=None, info=INFO, situations=FAKE_SITUATIONS, inputs=INPUTS,
           code=CODE):
    monkeypatch.chdir(tmp_path)
    spec = smoke(SPEC) if sm else SPEC
    return Runner(ScriptedMeasurer(testable=testable or {}, n_report=len(spec.even_report_seeds)), info, spec,
                  oracle=ORACLE, situations=situations, smoke=sm, inputs=inputs, code=code)


def _upto(r, last):
    for st in e_runner.ORDER[:e_runner.ORDER.index(last) + 1]:
        getattr(r, f"stage_{st}")()


def test_out_of_order_refused(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        r.stage_judge()
    with pytest.raises(SystemExit):
        r.stage_drive()
    assert not (tmp_path / e_store.SUMMARY).exists()


def test_full_chain_to_stop_even_low(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    for st in ("set", "drive", "codebook", "strength", "oc"):
        getattr(r, f"stage_{st}")()
    out = r.stage_even()
    assert out["outcome"] == "STOP_EVEN_LOW"
    assert out["sentence"].endswith("(설정별 k3-full 0/21·F_a 0, k3-norm 0/21·F_a 0, k2-full 0/21·F_a 0, "
                                    "k2-norm 0/21·F_a 0).")
    doc = e_store.read_summary()
    assert set(doc) == {"set", "drive", "codebook", "strength", "oc", "even"}
    cbk = doc["codebook"]
    for k in ("3", "2"):
        assert cbk["k"][k]["status"] == "OK" and cbk["k"][k]["C"] == 16 and cbk["k"][k]["colouring_digest"]
        assert len(cbk["k"][k]["colouring"]) == 96 and cbk["k"][k]["digest"]
        rs = cbk["k"][k]["anneal"]["restarts"]                                       # final review 7
        assert [x["r"] for x in rs] == [0] and {"J", "dup", "soft", "logvar"} <= set(rs[0])
    assert all(b["code_key"] == "codekey" and b["inputs"] == INPUTS for b in doc.values())   # final review 3
    assert all(c["overlap"] == {"cross": 0, "within": 0} for c in cbk["configs"].values())
    assert all(b["git"] == CLEAN and "written_at" in b for b in doc.values())
    with pytest.raises(SystemExit):
        r.stage_judge()
    with pytest.raises(SystemExit):                     # rewriting an earlier block would orphan later ones
        r.stage_codebook()


def test_smoke_never_writes_summary(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, sm=True)
    for st in ("set", "drive", "codebook", "strength", "even"):
        getattr(r, f"stage_{st}")()
    assert not (tmp_path / e_store.SUMMARY).exists()
    doc = e_store.read_summary(tmp_path / "results/encoder/smoke/summary.json")
    assert {"set", "drive", "codebook", "strength", "even"} <= set(doc) and doc["even"]["smoke"]
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_capped_strengths_never_measured(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    for st in ("set", "drive", "codebook"):
        getattr(r, f"stage_{st}")()
    r.stage_strength()
    from collections import Counter
    measured = Counter(c[2] for c in r.measurer.calls if c[0] == "activity")
    blk = e_store.read_summary()["strength"]
    capped = Counter(float(s) for cfg in blk["configs"].values() for s, why in cfg["reasons"].items() if why == "ORN_CAP")
    n_cfg = len(blk["configs"])
    assert capped and measured
    assert all(measured[s] == n_cfg - capped[s] for s in SPEC.s_grid)       # every capped (config, s) skipped
    for cfg in blk["configs"].values():
        for s, why in cfg["reasons"].items():
            assert cfg["table"][s]["measured"] == (why != "ORN_CAP")
    for cfg in blk["configs"].values():
        assert cfg["s"] == 0.175 and cfg["n_odours"] == 112 and cfg["n_single"] == 32 and cfg["n_dual"] == 80
        assert cfg["stats_176"] == "not measured (spec 4.3 decision 95258dd)"          # final review 11


def _selected_testable():
    even = FAKE_SITUATIONS.even()
    js = FAKE_SITUATIONS.judgement()
    key = lambda r: (r["axis"], r["turn"], r["x"], r["y"])
    t = {key(r): True for r in [r for r in even if r["axis"] == "b"][:12]}
    t.update({key(r): True for r in [r for r in even if r["axis"] == "a"][:2]})
    t.update({("judge:k3-full", *key(r)): True for r in js["b"][:14]})
    t.update({("judge:k3-full", *key(r)): True for r in js["a"][:2]})
    t.update({("judge:E0", *key(r)): True for r in js["b"][:3]})
    return t


def test_judge_selected_manifest_and_no_rerun(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, testable=_selected_testable())
    _upto(r, "oc")
    ev = r.stage_even()
    assert ev["outcome"] == "SELECTED" and ev["winner"] == "k3-full"
    out = r.stage_judge()
    assert out["band"] == "SELECTED" and (out["n"], out["c"], out["F_a"]) == (14, 3, 2)
    blk = e_store.read_summary()["judge"]
    assert len(blk["manifest"]) == 2 * (21 + 32)
    assert {m["tag"] for m in blk["manifest"]} == {"judge:k3-full", "judge:E0"}
    assert all(m["cache_key"] for m in blk["manifest"])
    assert blk["sentence"].startswith("C3 엔진·H.4 판독(MBON13/05)에서 결합 부호 E-grid k3-full이")
    assert len(blk["kc_jaccard"]["winner"]) == 53
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_judge_b_tb_sentence_closes(tmp_path, monkeypatch):
    t = _selected_testable()
    js = FAKE_SITUATIONS.judgement()
    key = lambda r: (r["axis"], r["turn"], r["x"], r["y"])
    t.update({("judge:E0", *key(r)): True for r in js["b"][:14]})       # c = 14 >= 11 -> B_결론없음; make c = 10
    for row in js["b"][10:14]:
        t[("judge:E0", *key(row))] = False
    for row in js["b"][10:14]:
        t[("judge:k3-full", *key(row))] = False                          # n = 10 <= c = 10 -> B_Tb
    r = runner(tmp_path, monkeypatch, testable=t)
    _upto(r, "even")
    out = r.stage_judge()
    assert out["band"] == "B_Tb"
    assert out["sentence"].endswith(e_runner.B_TB_CLOSE)


def test_judge_interrupted_writes_nothing_then_resumes(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, testable=_selected_testable())
    _upto(r, "even")
    r.measurer.fail_oracle.add("judge:E0")
    with pytest.raises(RuntimeError):
        r.stage_judge()
    assert "judge" not in e_store.read_summary()
    assert r.stage_judge()["band"] == "SELECTED"


def test_judge_incomplete_counts_write_nothing(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, testable=_selected_testable())
    _upto(r, "even")
    real = r.measurer.oracle
    r.measurer.oracle = lambda rows, *a: real(rows, *a)[:-1]           # one pair missing
    out = r.stage_judge()
    assert out["status"] == "INCOMPLETE" and out["band"] is None
    assert "judge" not in e_store.read_summary()


def test_judge_refuses_dirty_prior_block(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, testable=_selected_testable())
    monkeypatch.setattr(e_runner, "git_state", lambda: dict(CLEAN, dirty_hashed=["flymon/agent/e_runner.py"]))
    _upto(r, "even")
    monkeypatch.setattr(e_runner, "git_state", lambda: dict(CLEAN))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in e_store.read_summary()


def test_judge_refuses_codebook_digest_mismatch(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, testable=_selected_testable())
    _upto(r, "even")
    doc = e_store.read_summary()
    doc["even"]["winner_digest"] = "0" * 64
    (tmp_path / e_store.SUMMARY).write_text(json.dumps(doc))
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_infeasible_k_disables_only_its_configs(tmp_path, monkeypatch):
    g45 = GLOMS[:45]                                     # k = 3: omega*k = 48 > 45 -> INFEASIBLE; k = 2 keeps 40
    info = dict(INFO, glomeruli=g45, receptor_counts={g: INFO["receptor_counts"][g] for g in g45})
    r = runner(tmp_path, monkeypatch, info=info)
    _upto(r, "oc")
    doc = e_store.read_summary()
    st = {c: v["status"] for c, v in doc["codebook"]["configs"].items()}
    assert st == {"k3-full": "INFEASIBLE", "k3-norm": "INFEASIBLE", "k2-full": "OK", "k2-norm": "OK"}
    assert doc["strength"]["configs"]["k3-full"]["s"] is None
    out = r.stage_even()
    res = e_store.read_summary()["even"]["configs"]
    assert not res["k3-full"]["eligible"] and res["k2-full"]["eligible"]
    assert out["outcome"] == "STOP_EVEN_LOW" and "k3-full 자격 없음(INFEASIBLE)" in out["sentence"]


def test_set_short_stops(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, situations=FakeSituations(n_b_judge=20))
    out = r.stage_set()
    assert out["outcome"] == "STOP_SET_SHORT" and "(20쌍)" in out["sentence"]
    with pytest.raises(SystemExit):
        r.stage_drive()


def test_changed_judgement_list_refused(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch)
    _upto(r, "drive")
    r.sit = FakeSituations()
    r.sit._js["digest_keys"] = "changed"
    with pytest.raises(SystemExit):
        r.stage_codebook()


def test_cli_refuses_outside_repo_root(tmp_path, monkeypatch):
    import importlib.util
    from flymon.brain.h3_store import ROOT
    spec = importlib.util.spec_from_file_location("run_encoder_grid", ROOT / "scripts/run_encoder_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    assert mod.main(["--stage", "set"]) == 2
    assert not (tmp_path / "results").exists()


# ---- final review fixes ------------------------------------------------------------------------------------------
def _judge_ready(tmp_path, monkeypatch, **kw):
    r = runner(tmp_path, monkeypatch, testable=_selected_testable(), **kw)
    _upto(r, "even")
    return r


def _edit_summary(tmp_path, fn):
    doc = e_store.read_summary()
    fn(doc)
    (tmp_path / e_store.SUMMARY).write_text(json.dumps(doc))


@pytest.mark.parametrize("state", [dict(dirty=True), dict(tracked=False)])
def test_stage_refuses_uncommitted_summary(tmp_path, monkeypatch, state):
    """Final review 4: spec 4.0 — every full-run stage refuses while the summary has uncommitted changes."""
    r = runner(tmp_path, monkeypatch)
    r.stage_set()
    monkeypatch.setattr(e_runner, "summary_git", lambda path: dict(SUMMARY_CLEAN, **state))
    before = (tmp_path / e_store.SUMMARY).read_text()
    with pytest.raises(SystemExit):
        r.stage_drive()
    with pytest.raises(SystemExit):
        r.stage_set()
    assert (tmp_path / e_store.SUMMARY).read_text() == before


def test_smoke_ignores_summary_git(tmp_path, monkeypatch):
    r = runner(tmp_path, monkeypatch, sm=True)
    monkeypatch.setattr(e_runner, "summary_git", lambda path: dict(SUMMARY_CLEAN, dirty=True, tracked=False))
    r.stage_set()
    r.stage_drive()


@pytest.mark.parametrize("state", [dict(dirty=True), dict(tracked=False), dict(judge_commits=["abc123"])])
def test_judge_refuses_summary_not_committed_or_already_judged(tmp_path, monkeypatch, state):
    """Final review 4: the judge needs a committed, clean summary whose history never held a judge block."""
    r = _judge_ready(tmp_path, monkeypatch)
    monkeypatch.setattr(e_runner, "summary_git", lambda path: dict(SUMMARY_CLEAN, **state))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in e_store.read_summary()
    assert not any(c[0] == "oracle" and c[1].startswith("judge") for c in r.measurer.calls)     # nothing measured


@pytest.mark.parametrize("edit", [lambda d: d["drive"].update(smoke=True), lambda d: d["oc"].pop("smoke")])
def test_judge_refuses_smoke_blocks(tmp_path, monkeypatch, edit):
    """Final review 9: a smoke block (or one with no smoke flag) never feeds a judgement."""
    r = _judge_ready(tmp_path, monkeypatch)
    _edit_summary(tmp_path, edit)
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in e_store.read_summary()


def test_judge_refuses_code_key_or_inputs_change(tmp_path, monkeypatch):
    """Final review 3: the judge's code key and inputs must be the ones every block (the selection) recorded."""
    r = _judge_ready(tmp_path, monkeypatch)
    r.code_key = "other"
    with pytest.raises(SystemExit):
        r.stage_judge()
    r.code_key = CODE["key"]
    r.inputs = dict(INPUTS, **{"results/summary/m0d.json": "changed"})
    with pytest.raises(SystemExit):
        r.stage_judge()
    r.inputs = {"data/malecns.npz": "npzsha"}                       # m0d.json not hashed at all
    with pytest.raises(SystemExit):
        r.stage_judge()
    r.inputs = dict(INPUTS)
    _edit_summary(tmp_path, lambda d: d["even"].pop("code_key"))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in e_store.read_summary()


def test_runner_without_code_key_refuses_judge(tmp_path, monkeypatch):
    r = _judge_ready(tmp_path, monkeypatch, code=None)
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_judge_provenance_hashed_refuses_record_only_recorded(tmp_path, monkeypatch):
    """Final review 3: a hashed file whose sha256 changed since block even refuses; a record-only file is recorded."""
    sha = {"flymon/agent/e_rules.py": "a", "flymon/brain/h4_jobs.py": "b", e_runner.RECORD_ONLY_FILES[0]: "c"}
    monkeypatch.setattr(e_runner, "provenance", lambda args: dict(sha256=dict(sha), args=args))
    r = _judge_ready(tmp_path, monkeypatch)
    sha["flymon/agent/e_rules.py"] = "changed"
    with pytest.raises(SystemExit):
        r.stage_judge()
    sha["flymon/agent/e_rules.py"] = "a"
    sha[e_runner.RECORD_ONLY_FILES[0]] = "changed"
    assert r.stage_judge()["band"] == "SELECTED"
    assert e_store.read_summary()["judge"]["record_only_changed_since_even"] == [e_runner.RECORD_ONLY_FILES[0]]


def test_judge_refuses_used_codebook_not_matching_digest(tmp_path, monkeypatch):
    """Final review 5: the codebook list actually used must hash to winner_digest (digests alone agree here)."""
    r = _judge_ready(tmp_path, monkeypatch)

    def swap(d):
        book = d["codebook"]["k"]["3"]["codebook"]
        book[0], book[1] = book[1], book[0]
    _edit_summary(tmp_path, swap)
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in e_store.read_summary()


def test_hashed_files_and_code_key_files():
    """Final review 1-2: the code key covers every oracle_job/activity_job dependency; the hashed files cover the
    modules the track imports and the M0d summary."""
    import importlib.util
    from flymon.brain.h3_store import ROOT
    for f in ("flymon/brain/presentation.py", "flymon/brain/plasticity.py", "flymon/brain/conditioning.py",
              "flymon/brain/h4_jobs.py", "flymon/brain/k_jobs.py", "flymon/brain/engine_cpu.py"):
        assert f in e_runner.MEASURE_FILES_E
    h = set(e_runner.hashed_files())
    assert set(e_runner.MEASURE_FILES_E) <= h
    for f in ("flymon/agent/config.py", "flymon/brain/h3_c3.py", "flymon/brain/h4_spec.py", "flymon/brain/k_pairs.py",
              "flymon/brain/j_store.py", "flymon/brain/j_params.py", "flymon/battle/pool.py",
              "results/summary/m0d.json", "flymon/agent/e_runner.py", "scripts/run_encoder_grid.py"):
        assert f in h
    assert all((ROOT / f).exists() for f in h)
    spec = importlib.util.spec_from_file_location("run_encoder_grid", ROOT / "scripts/run_encoder_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.measure_files() == e_runner.MEASURE_FILES_E


def _git(cwd, *args):
    import subprocess
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_summary_git_on_a_temp_repo(tmp_path, monkeypatch):
    """Final review 4: the real git helper — untracked, dirty, clean, and a judge block anywhere in history."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    monkeypatch.chdir(tmp_path)
    p = tmp_path / e_store.SUMMARY
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"set": {}}))
    g = _real_summary_git(e_store.SUMMARY)
    assert not g["tracked"] and g["dirty"] and g["judge_commits"] == []
    _git(tmp_path, "add", e_store.SUMMARY)
    _git(tmp_path, "commit", "-qm", "set")
    assert _real_summary_git(e_store.SUMMARY) == dict(tracked=True, dirty=False, judge_commits=[])
    p.write_text(json.dumps({"set": {}, "judge": {}}))
    assert _real_summary_git(e_store.SUMMARY)["dirty"]
    _git(tmp_path, "commit", "-qam", "judge")
    p.write_text(json.dumps({"set": {}}))                          # judge block removed again later
    _git(tmp_path, "commit", "-qam", "drop judge")
    g = _real_summary_git(e_store.SUMMARY)
    assert g["tracked"] and not g["dirty"] and len(g["judge_commits"]) == 1


def _cli():
    import importlib.util
    from flymon.brain.h3_store import ROOT
    spec = importlib.util.spec_from_file_location("run_encoder_grid", ROOT / "scripts/run_encoder_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_e0_strength_check():
    """Final review 6: E0's strength must be the M0d H.4 strength recorded in results/summary/m0d.json."""
    from flymon.brain.h3_store import ROOT
    mod = _cli()
    m0d = json.loads((ROOT / "results/summary/m0d.json").read_text())
    assert mod.check_e0_strength(SPEC, m0d) is None
    assert mod.check_e0_strength(dataclasses.replace(SPEC, e0_strength=0.5), m0d)


def test_cli_exit_codes():
    """Final review 8: judge INCOMPLETE exits 3; STOP outcomes and READ bands exit 0."""
    mod = _cli()
    assert mod.exit_code("judge", dict(status="INCOMPLETE", band=None)) == 3
    assert mod.exit_code("judge", dict(status="READ", band="B_Tb")) == 0
    assert mod.exit_code("even", dict(outcome="STOP_EVEN_LOW")) == 0
    assert mod.exit_code("set", dict(outcome="STOP_SET_SHORT")) == 0
