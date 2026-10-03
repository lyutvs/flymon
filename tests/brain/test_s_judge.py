# tests/brain/test_s_judge.py
"""The judgement end of the S chain (S.3 ⑤-⑦, S.4, S.5, S.9.2, S.9.5; Readings 4, 10-12) and its mutation tests: jm
blocks carry no pair statistic and are written only when all 64 pairs are back; the reuse condition is re-checked
before the judgement measurement, the seal and the judge (exit 7); the seal re-checks every raw file, runs the pre-read
validity, archives 192 raw files and pins the decision code; the judge reads only a sealed set under the sealed
decision code, once — with S.9.2's one re-generation after an interruption between the mark and the block (fault
injection); G_fail_S fires on a net drop of 3 and never on a symmetric swap; the CLI's refusals and exit codes; S never
imports a modified copy of a shared measurement file; the judgement set is touched only from jm on."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_measure, r_store, s_store
from flymon.brain import s_runner as SR
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.s_spec import SPEC
from flymon.brain.s_store import SCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.s_world import CODE, Scripted, World, doc, through_gate2

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through_gate2(w, m)
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def _sealed(w, plan=None):
    r = _to_seal(w, Scripted(plan=plan or w.pass_plan(n=14, c=8, f_a=3)))
    assert r.stage_seal()["status"] == "SEALED"
    return r


def test_full_chain_reads_selected(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan(n=14, c=8, f_a=3)))
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 64 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 64
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 192 and seal["set"]["n_a"] == 43
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive)) and seal["decision"]["key"] == SR.decision_key()["key"]
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    rho = doc()["gate2"]["ratio"]
    assert (f"(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 "
            f"{rho['r1']['ratio']:.3f}·ℓ_r2 {rho['r2']['ratio']:.3f} ≥ 0.5(약화 정도 기록), 짝수 16/21") in out["sentence"]
    assert out["resumed_after_mark"] is False and out["mark_read_at"] is None
    assert out["g_fail"]["g_fail"] is False and out["records"]["transitions_C_to_L"]["b"]["testable"]["fp"] == 6
    assert out["oc_sha256"] == doc()["oc"]["sha256"] and set(doc()) == set(SR.ORDER)
    assert Path(SR.JUDGE_MARKER).exists() and Path(SR.DONE_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()                                     # once


def test_net_drop_3_on_a_reads_the_punish_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", "L")].update({k: (False, False, False) for k in ja[30:33]})       # 3 (a) pass->fail, none back
    out = _sealed(w, plan).stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["a"]["net_drop"] == 3
    assert out["sentence"] == ("M2 (b) 기준과 여유는 넘었지만 지렛대 아래 처벌 통과가 순감소했다(처벌 통과 (b) 21 대 21, "
                               "(a) 40 대 43; 순감소 (b) 0, (a) 3).")


def test_a_symmetric_swap_of_3_does_not_fire_the_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", "L")].update({k: (False, False, False) for k in ja[30:33]})       # 3 pass->fail
    plan[("judge", "C")] = dict(plan[("judge", "C")], **{k: (False, False, False) for k in ja[33:36]})  # 3 fail->pass
    out = _sealed(w, plan).stage_judge()
    ax = out["g_fail"]["axes"]["a"]
    assert (ax["pass_to_fail"], ax["fail_to_pass"], ax["net_drop"]) == (3, 3, 0) and out["band"] == "SELECTED"


def test_judgement_set_is_touched_only_from_jm_on(w):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = Scripted(plan=w.pass_plan())
    r = through_gate2(w, m)
    m.fail_once = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 64


@pytest.mark.parametrize("where", ["jm", "seal", "judge"])
def test_reuse_is_rechecked_before_measurement_seal_and_judge(w, where):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    w.r["gate3"]["code_key"] = "x" * 64                      # R's block rewritten under another key
    with pytest.raises(SystemExit) as e:
        {"jm": lambda: r.stage_jm("L"), "seal": r.stage_seal, "judge": r.stage_judge}[where]()
    assert e.value.code == SR.EXIT_REUSE
    assert not Path(SR.JUDGE_MARKER).exists()


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))
    d = doc()
    del d["gate2"]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and w.judgement_calls == 0


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    r = _to_seal(w, Scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)}))
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_an_edit_in_c_fails_the_preread_validity(w):
    m = Scripted(plan=w.pass_plan(), override={("judge", "C"): dict(edges=2, edit=SPEC.lever_edit, sha="sha-L")})
    seal = _to_seal(w, m).stage_seal()
    assert seal["status"] == "NOT_READ" and any("C" in x for x in seal["reasons"])


def test_mutation_a_tampered_set_refuses(w):
    r = through_gate2(w, Scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("S set digest_keys: generated '00', declared '884d'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_mutation_a_manifest_pointing_c_at_l_files_fails_the_preread_validity(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    d = doc()
    d["jm:C"]["manifest"] = d["jm:L"]["manifest"]
    Path(SPEC.summary).write_text(json.dumps(d))
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any(x.startswith("C: ") and "stored inputs" in x for x in seal["reasons"])


def test_judge_refuses_a_raw_file_changed_after_the_seal(w):
    r = _sealed(w)
    f = Path(doc()["jm:C"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"sha-C"', '"sha-X"'))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in doc() and not Path(SR.JUDGE_MARKER).exists()


def test_an_earlier_committed_judge_or_another_code_key_refuses(w):
    m = Scripted(plan=w.pass_plan())
    r = through_gate2(w, m)
    w.judge_commits.append("abc")
    with pytest.raises(SystemExit):
        r.stage_jm("L")
    w.judge_commits.clear()
    with pytest.raises(SystemExit):
        w.runner(m, code={"key": "z" * 64}).stage_jm("L")


def test_judge_twice_with_the_first_block_discarded_refuses(w):
    r = _sealed(w)
    assert r.stage_judge()["status"] == "READ"
    marker = json.loads(Path(SR.JUDGE_MARKER).read_text())
    assert marker["seal_written_at"] == doc()["seal"]["written_at"] and marker["read_at"]
    d = doc()
    del d["judge"]                                          # the judge block discarded (git checkout)
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()                                     # DONE marker: no re-generation for a written block
    assert e.value.code == 2 and "judge" not in doc()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(SR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(SR.JUDGE_MARKER).exists()


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(SR.s_rules, "read_band", lambda *a, **k: dict(band=SR.s_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == SR.s_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(SR.JUDGE_MARKER).exists()


class _Kill(Exception):
    pass


def _kill_judge_write(monkeypatch, seen):
    """Fault injection (S.9.2): the process dies after the mark, while the judge block is being written."""
    orig = s_store.write_summary_block

    def boom(path, block, obj, plist):
        if block == "judge":
            seen.append(obj)
            raise _Kill()
        return orig(path, block, obj, plist)
    monkeypatch.setattr(SR.s_store, "write_summary_block", boom)
    return orig


def test_resume_after_mark_once(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    mark = json.loads(Path(SR.JUDGE_MARKER).read_text())
    assert "judge" not in doc() and not Path(SR.DONE_MARKER).exists() and not Path(SR.REREAD_MARKER).exists()
    monkeypatch.setattr(SR.s_store, "write_summary_block", orig)
    out = r.stage_judge()                                   # the one re-generation
    assert out["status"] == "READ" and out["resumed_after_mark"] is True and out["mark_read_at"] == mark["read_at"]
    assert (out["band"], out["sentence"]) == (seen[0]["band"], seen[0]["sentence"])
    assert doc()["judge"]["resumed_after_mark"] is True and Path(SR.REREAD_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_a_second_regeneration_refuses(w, monkeypatch):
    r = _sealed(w)
    seen = []
    _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    with pytest.raises(_Kill):
        r.stage_judge()                                     # the re-generation dies too
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and len(seen) == 2 and "judge" not in doc()


def test_regeneration_refuses_another_decision_code_or_seal(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    monkeypatch.setattr(SR.s_store, "write_summary_block", orig)
    real = SR.decision_key
    monkeypatch.setattr(SR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit):
        r.stage_judge()                                     # another decision code
    monkeypatch.setattr(SR, "decision_key", real)
    m = json.loads(Path(SR.JUDGE_MARKER).read_text())
    Path(SR.JUDGE_MARKER).write_text(json.dumps(dict(m, seal_written_at="another seal")))
    with pytest.raises(SystemExit):                         # a marker from another seal
        r.stage_judge()
    assert "judge" not in doc() and not Path(SR.REREAD_MARKER).exists()


def test_recovery_after_reading(w):
    r = _sealed(w)
    r.stage_judge()
    with pytest.raises(SystemExit):
        r.stage_recompute("")
    e = r.stage_recompute("records code fix (test)")
    assert e["band"] == "SELECTED" and e["differs_from_judge"] is False and e["decision_changed_since_seal"] is False
    assert r.stage_invalid_run("measurement defect (test)")["status"] == "INVALID_RUN"
    with pytest.raises(SystemExit):
        r.stage_invalid_run("again")
    with pytest.raises(SystemExit):
        r.stage_recompute("after invalid")


class OraclePool:
    """r_oracle_job-shaped outputs for a real RMeasurer + SCache; the pair key travels in the odour name ("G|<key>"
    E-grid, "E|<key>" E0)."""

    def __init__(self, plan):
        self.n_workers, self.plan, self.jobs = 4, plan, 0

    def run_jobs(self, fn, kws):
        assert fn is r_jobs.r_oracle_job
        out = []
        for kw in kws:
            self.jobs += 1
            lever = kw["edit"] == SPEC.lever_edit
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            flags = self.plan.get(("judge", cond), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha="sha-L" if lever else "sha-C", edges=SPEC.lever_edges if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_real_cache_round_trip_seals_and_reads(w):
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through_gate2(w, Scripted(plan=plan))
    pool = OraclePool(plan)
    real = w.runner(RMeasurer(pool, SCache(SPEC.cache_dir, CODE), SPEC, w.ctx["params"], READOUT, Z, TYPES, 100))
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 64
    assert pool.jobs == 192
    f = doc()["jm:L"]["manifest"][0]["cache_file"]
    stored = json.loads(Path(f).read_text())["inputs"]
    assert f.startswith("results/s/cache/r_oracle/") and stored["act_seeds"] == SPEC.judge_seeds()["act"]
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds", "s_set"}
# s_spec's None: the judgement seed fields at class level
ALLOWED = {"s_spec.py": {None, "judge_seeds"}, "s_pairs.py": {"judgement_rows"},
           "s_runner.py": {"build_ctx", "_judgement_rows", "stage_set", "stage_jm", "stage_seal", "_read", "_cost"}}


def _owners(tree) -> set:
    out = set()

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            if (isinstance(ch, ast.Name) and ch.id in WATCH) or (isinstance(ch, ast.Attribute) and ch.attr in WATCH):
                out.add(fn)
            visit(ch, ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn)
    visit(tree, None)
    return out


def test_only_the_judgement_stages_name_the_judgement_set():
    files = sorted((ROOT / "flymon/brain").glob("s_*.py")) + [ROOT / "scripts/run_s.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def test_s_never_imports_a_modified_copy_of_a_shared_measurement_file():
    """S.3 ① / S.9.5: S runs R's jobs, measurer and cache themselves — the objects S uses are the ones defined in the
    R_MEASURE_FILES modules, no S file defines a job, a measurer or a cache of its own (SCache only subclasses
    RCache's writer), and no S file loads code by path."""
    assert s_store.RCache is r_store.RCache and issubclass(SCache, r_store.RCache)
    assert {k for k in vars(SCache) if not k.startswith("__")} == {"put"}
    for obj in (r_jobs.kc_activity_job, r_jobs.r_arm_job, r_jobs.r_oracle_job, RMeasurer, r_store.RCache):
        src = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
        assert src in R_MEASURE_FILES, src
    shared = {"kc_activity_job", "r_arm_job", "r_oracle_job", "RMeasurer", "RCache", "ECache", "q_oracle_job",
              "arm_job", "q_rig"}
    for p in sorted((ROOT / "flymon/brain").glob("s_*.py")) + [ROOT / "scripts/run_s.py"]:
        tree = ast.parse(p.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not defs & shared, (p.name, defs & shared)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not names & {"importlib", "exec", "spec_from_file_location", "runpy"}, p.name


def _cli():
    spec = importlib.util.spec_from_file_location("run_s_for_test", ROOT / "scripts/run_s.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_argument_refusals_and_exit_codes(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    for argv in (["--stage", "reuse", "--condition", "L"], ["--stage", "jm"], ["--stage", "recompute"],
                 ["--stage", "smoke", "--note", "x"], ["--stage", "set", "--rerun-after-invalid"], [],
                 ["--stage", "reuse"]):                                   # the last: wrong cwd
        assert cli.main(argv) == 2, argv
    assert cli.exit_code("reuse", {"outcome": "PASS"}) == 0 and cli.exit_code("reuse", {"outcome": "STOP_REUSE"}) == 3
    assert cli.exit_code("set", {"outcome": "STOP_SET_SHORT"}) == 3
    assert cli.exit_code("gate2", {"outcome": "STOP_PUNISH_WEAKENED"}) == 3
    assert cli.exit_code("gate2", {"outcome": "STOP_P_REFERENCE"}) == 3
    assert cli.exit_code("gate2", {"outcome": "INVALID"}) == 5
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    assert cli.exit_code("oc", {}) == 0 and cli.exit_code("gate2_oc", {}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "INVALID"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_s.py").read_text()


def test_hashed_pipeline_and_decision_files():
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.r_runner import R_HASHED_FILES
    assert set(SR.S_PIPELINE_FILES) == {f"flymon/brain/s_{n}.py" for n in
                                        ("spec", "pairs", "store", "records", "rules", "runner")} | {"scripts/run_s.py"}
    assert not set(SR.S_PIPELINE_FILES) & set(R_MEASURE_FILES)
    assert set(R_HASHED_FILES) | set(SR.S_PIPELINE_FILES) <= set(SR.S_HASHED_FILES)
    assert "results/summary/r_lever.json" in SR.S_HASHED_FILES
    assert set(SR.DECISION_FILES) == set(SR.S_HASHED_FILES) - set(R_MEASURE_FILES)
    assert len(SR.pipeline_key()["key"]) == 64 and set(SR.pipeline_key()["files"]) == set(SR.S_PIPELINE_FILES)
