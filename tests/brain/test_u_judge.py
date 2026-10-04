"""The judgement end of the U chain (U.3 9-11, U.4, U.5, U.6, U.8) and its mutation tests: jm:L runs L_f* on z_f* and
jm:C / jm:E0 on block h4's z, every block carries no pair statistic and is written only when all 64 pairs are back; the
reuse condition and the T / U measurement keys are re-checked before the judgement measurement, the seal and the judge
(exit 7); the seal re-checks every raw file and its stored inputs (L's carry u_edit(f*) and z_f*), archives 192 raw
files and pins the decision code; the judge reads once — with the one re-generation after an interruption between the
mark and the block — and fills U.7's sentence with f*; the records hold the clusters, the α-fixed sensitivity, z_f*,
the f choice and the contrast readings; the CLI's refusals and exit codes; job copies live only in U's measurement
file; the judgement set is touched only from the KC band (its odours) and jm on."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_store, t_measure, u_store
from flymon.brain import u_measure as UM
from flymon.brain import u_runner as UR
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.t_measure import T_MEASURE_FILES
from flymon.brain.u_measure import U_MEASURE_FILES, u_edit
from flymon.brain.u_store import UCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.u_world import CODE, SPEC, TCODE, UCODE, ZF, World, ZScripted, doc, through

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through(w, m, zm=ZScripted())
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def _sealed(w, plan=None):
    m = w.scripted(plan=plan or w.pass_plan(n=14, c=8, f_a=3))
    r = _to_seal(w, m)
    assert r.stage_seal()["status"] == "SEALED"
    return r


def test_full_chain_reads_selected(w):
    r = _to_seal(w, w.scripted(plan=w.pass_plan(n=14, c=8, f_a=3)))
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 64 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 64
    assert jm["edit"] == u_edit(0.6) and jm["f_star"] == 0.6
    assert jm["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and doc()["jm:C"]["z"] == {"A": [10.0, 9.0],
                                                                                     "P": [26.0, 19.0]}
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm
    stored = json.loads(Path(jm["manifest"][0]["cache_file"]).read_text())["inputs"]
    assert stored["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and stored["edit"] == u_edit(0.6)
    assert stored["act_seeds"] == SPEC.judge_seeds()["act"] == list(range(24_500_000, 24_500_008))
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 192 and seal["set"]["n_a"] == 43 and seal["f_star"] == 0.6
    assert seal["z"]["z_f_star"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive)) and seal["decision"]["key"] == UR.decision_key()["key"]
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"], out["f_star"]) == (
        "READ", "SELECTED", 14, 8, 3, 0.6)
    rho = doc()["gate2"]["ratio"]
    assert (f"(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 "
            f"{rho['r1']['ratio']:.3f}·ℓ_r2 {rho['r2']['ratio']:.3f} ≥ 0.5(약화 정도 기록), 짝수 13/21, 판정 시드 "
            f"24_500_xxx)") in out["sentence"]
    assert "f = 0.6배로" in out["sentence"] and "생성원 턴 0–103" in out["sentence"]
    rec = out["records"]
    assert rec["clusters"]["GROUND* 대 NORMAL"]["L"] == dict(n=21, testable=14, reward_pass=14, punish_pass=21)
    assert rec["alpha_fixed"]["n"] == 14 and "판정 아님" in rec["alpha_fixed"]["note"]
    assert rec["z"]["z_f_star"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert rec["choose"] == dict(kept=[0.5, 0.6], checked=[0.5, 0.6, 0.7])
    assert rec["contrast"]["block"]["reading"] == "사슬 비지지"
    assert out["resumed_after_mark"] is False and out["k_even"] == 13
    assert out["oc_sha256"] == doc()["oc"]["independent"]["sha256"] and set(doc()) == set(UR.ORDER)
    assert Path(UR.JUDGE_MARKER).exists() and Path(UR.DONE_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_net_drop_3_on_a_reads_the_punish_guard_with_f(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", u_edit(0.6))].update({k: (False, False, False) for k in ja[30:33]})
    out = _sealed(w, plan).stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["a"]["net_drop"] == 3
    assert out["sentence"].endswith(" (f = 0.6)")


def test_b_tb_closes_the_partial_lever_claim_only(w):
    out = _sealed(w, w.pass_plan(n=8, c=8, f_a=3)).stage_judge()
    assert out["band"] == "B_Tb" and out["sentence"] == (
        "APL→MBON05 부분 제거(f = 0.6)가 넓힌 상대 풀 판정 세트(생성원 턴 0–103)에서 8/21로 지렛대 없는 같은 세트 8/21보다 "
        "오르지 않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 이 부분 제거 주장을 닫는다.")


def test_b_nc_names_f(w):
    out = _sealed(w, w.pass_plan(n=10, c=8, f_a=3)).stage_judge()
    assert out["band"] == "B_결론없음" and out["sentence"].endswith("(10/21 대 8/21, F_a 3/43, f = 0.6)")


def test_judgement_set_is_touched_only_from_kc_and_jm_on(w):
    r = through(w, w.scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0 and w.odour_calls == 1
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = w.scripted(plan=w.pass_plan())
    r = through(w, m)
    m.st["fail_once"] = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 64


@pytest.mark.parametrize("where", ["jm", "seal", "judge"])
@pytest.mark.parametrize("what", ["reuse", "tkey", "ukey", "no_ukey"])
def test_keys_are_rechecked_before_measurement_seal_and_judge(w, where, what):
    m = w.scripted(plan=w.pass_plan())
    r = through(w, m)
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    if what == "reuse":
        w.t["z"]["detail_sha256"] = "x" * 64
    elif what == "tkey":
        r = w.runner(m, tcode={"key": "x" * 64})
    elif what == "ukey":
        r = w.runner(m, ucode={"key": "v" * 64})
    else:
        r = UR.Runner(m.at, None, w.ctx, SPEC, code=CODE, tcode=TCODE, ucode=None, pipeline={"key": "p" * 64},
                      archive_root=w.archive)
    before = doc()
    with pytest.raises(SystemExit) as e:
        {"jm": lambda: r.stage_jm("L"), "seal": r.stage_seal, "judge": r.stage_judge}[where]()
    assert e.value.code == UR.EXIT_KEY and doc() == before
    assert not Path(UR.JUDGE_MARKER).exists()


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    r = _to_seal(w, w.scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)}))
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_l_raw_on_h4_z_fails_the_preread_validity(w):
    m = w.scripted(plan=w.pass_plan())
    r = through(w, m)
    real_m_for = r._m_for
    r._m_for = lambda name, doc=None: m.at(Z) if name == "L" else real_m_for(name, doc)
    r.stage_jm("L")
    r._m_for = real_m_for
    r.stage_jm("C")
    r.stage_jm("E0")
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any(x.startswith("L: ") and "stored inputs" in x for x in seal["reasons"])


def test_mutation_l_raw_of_another_f_fails_the_preread_validity(w):
    """A jm:L measured on another f (its edit string) is caught by raw_check and the stored inputs at the seal."""
    m = w.scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edit=u_edit(0.5))})
    seal = _to_seal(w, m).stage_seal()
    assert seal["status"] == "NOT_READ" and any("ran edit u_apl_mbon05_x0.5" in x for x in seal["reasons"])


def test_mutation_a_tampered_set_refuses(w):
    r = through(w, w.scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("T set digest_keys: generated '00', declared '37dd'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through(w, w.scripted(plan=w.pass_plan()))
    d = doc()
    del d["choose"]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and w.judgement_calls == 0


def test_judge_refuses_a_raw_file_changed_after_the_seal(w):
    r = _sealed(w)
    f = Path(doc()["jm:C"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"sha-C"', '"sha-X"'))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in doc() and not Path(UR.JUDGE_MARKER).exists()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(UR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(UR.JUDGE_MARKER).exists()


def _edit_summary(fn):
    d = doc()
    fn(d)
    Path(SPEC.summary).write_text(json.dumps(d))


@pytest.mark.parametrize("edit", ["f_star", "z_f_star"])
def test_judge_refuses_a_choose_or_scan_point_changed_after_the_seal(w, edit):
    """U.3 10: the seal's decision hash covers the z_f* scan point and the choose block; judge refuses on change."""
    r = _sealed(w)
    seal = doc()["seal"]
    assert seal["decision"]["choose_sha256"] and seal["decision"]["scan_f_star_sha256"]

    def ch(d):
        if edit == "f_star":
            d["choose"]["f_star"] = 0.5
        else:
            d["scan"]["points"][UR.fk(0.6)]["z"]["A"][0] = 7.0
    _edit_summary(ch)
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(UR.JUDGE_MARKER).exists()


def test_regeneration_refuses_a_scan_point_changed_after_the_mark(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    monkeypatch.setattr(UR.u_store, "write_summary_block", orig)
    _edit_summary(lambda d: d["scan"]["points"][UR.fk(0.6)]["z"]["P"].__setitem__(0, 81.0))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(UR.REREAD_MARKER).exists()


@pytest.mark.parametrize("broken", [False, True])
def test_z_h4_reproduced_reads_the_reuse_block(w, broken):
    """records.z.z_h4_reproduced is read from the reuse block (T's unedited z = block h4's z), not a literal."""
    r = _sealed(w)
    if broken:
        _edit_summary(lambda d: d["reuse"]["records"]["t_z"]["none_z"]["A"].__setitem__(0, 99.0))
    out = r.stage_judge()
    assert out["records"]["z"]["z_h4_reproduced"] is (not broken)


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(UR.u_rules, "read_band", lambda *a, **k: dict(band=UR.u_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == UR.u_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(UR.JUDGE_MARKER).exists()


class _Kill(Exception):
    pass


def _kill_judge_write(monkeypatch, seen):
    orig = u_store.write_summary_block

    def boom(path, block, obj, plist):
        if block == "judge":
            seen.append(obj)
            raise _Kill()
        return orig(path, block, obj, plist)
    monkeypatch.setattr(UR.u_store, "write_summary_block", boom)
    return orig


def test_resume_after_mark_once(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    mark = json.loads(Path(UR.JUDGE_MARKER).read_text())
    assert "judge" not in doc() and not Path(UR.DONE_MARKER).exists() and mark["u_measure_key"] == "u" * 64
    monkeypatch.setattr(UR.u_store, "write_summary_block", orig)
    out = r.stage_judge()
    assert out["status"] == "READ" and out["resumed_after_mark"] is True and out["mark_read_at"] == mark["read_at"]
    assert (out["band"], out["sentence"]) == (seen[0]["band"], seen[0]["sentence"])
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_a_second_regeneration_refuses(w, monkeypatch):
    r = _sealed(w)
    seen = []
    _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    with pytest.raises(_Kill):
        r.stage_judge()
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and len(seen) == 2 and "judge" not in doc()


def test_recovery_after_reading(w):
    r = _sealed(w)
    with pytest.raises(SystemExit):
        r.stage_recompute("before judge")
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


@pytest.mark.parametrize("edit", ["f_star", "z_f_star"])
def test_recompute_refuses_a_choose_or_scan_point_changed_after_the_seal(w, edit):
    """U.3 10 / U.5: recompute reads the sealed f* and z_f* only; a moved choose block or scan point refuses (exit 2)."""
    r = _sealed(w)
    r.stage_judge()

    def ch(d):
        if edit == "f_star":
            d["choose"]["f_star"] = 0.5
        else:
            d["scan"]["points"][UR.fk(0.6)]["z"]["A"][0] = 7.0
    _edit_summary(ch)
    with pytest.raises(SystemExit) as e:
        r.stage_recompute("records code fix (test)")
    assert e.value.code == 2 and "recompute" not in doc()


def test_recompute_records_a_decision_key_change_alone(w, monkeypatch):
    """U.5: recompute after a code fix — a changed decision key alone is recorded, not refused."""
    r = _sealed(w)
    r.stage_judge()
    monkeypatch.setattr(UR, "decision_key", lambda: dict(key="d" * 64, files={}))
    e = r.stage_recompute("decision code fix (test)")
    assert e["decision_changed_since_seal"] is True and e["decision_key"] == "d" * 64 and e["band"] == "SELECTED"
    assert doc()["recompute"][-1]["decision_changed_since_seal"] is True


class OraclePool:
    """u_oracle_job-shaped outputs for real RMeasurers over UPool + UCache; the pair key travels in the odour name
    ("G|<key>" E-grid, "E|<key>" E0); the z and edit each job received are recorded."""

    def __init__(self, plan):
        self.n_workers, self.plan, self.jobs, self.z, self.edits = 4, plan, 0, {}, set()

    def run_jobs(self, fn, kws):
        assert fn is UM.u_oracle_job
        out = []
        for kw in kws:
            self.jobs += 1
            lever = UM.is_u_edit(kw["edit"])
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            self.z[cond] = kw["z"]
            self.edits.add(kw["edit"])
            flags = self.plan.get(("judge", kw["edit"]), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha=f"sha-{kw['edit']}" if lever else "sha-C", edges=2 if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_real_cache_round_trip_seals_and_reads(w):
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through(w, w.scripted(plan=plan))
    pool, cache, ms = OraclePool(plan), UCache(SPEC.cache_dir, UCODE), {}

    def measure(z):
        k = json.dumps({a: list(b) for a, b in sorted(z.items())})
        return ms.setdefault(k, RMeasurer(UM.UPool(pool), cache, SPEC, w.ctx["params"], READOUT, z, TYPES, 100))
    real = UR.Runner(measure, None, w.ctx, SPEC, code=CODE, tcode=TCODE, ucode=UCODE, pipeline={"key": "p" * 64},
                     archive_root=w.archive)
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 64
    assert pool.jobs == 192 and pool.edits == {u_edit(0.6), "none"}
    assert pool.z["L"] == {k: list(v) for k, v in ZF.items()} and pool.z["C"] == pool.z["E0"] == {
        k: list(v) for k, v in Z.items()}
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds", "t_set", "set_odours", "_checked"}
ALLOWED = {"u_spec.py": {None, "judge_seeds"},
           "u_runner.py": {"build_ctx", "_judgement_rows", "stage_set", "stage_kc", "stage_jm", "stage_seal", "_read",
                           "_cost"}}


def _owners(tree) -> set:
    out = set()

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            if ((isinstance(ch, ast.Name) and ch.id in WATCH) or (isinstance(ch, ast.Attribute) and ch.attr in WATCH)
                    or (isinstance(ch, ast.Constant) and ch.value in WATCH)):
                out.add(fn)
            visit(ch, ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn)
    visit(tree, None)
    return out


def test_only_the_judgement_stages_name_the_judgement_set():
    files = sorted((ROOT / "flymon/brain").glob("u_*.py")) + [ROOT / "scripts/run_u.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def test_job_copies_live_only_in_us_measurement_file():
    """U.8 / U.9.4 P2-5: U's copies of the oracle, P arm, KC activity and reference / rest jobs are defined only in
    u_measure.py (in the U measurement key); every other U file defines no job, measurer or cache copy; RMeasurer and
    the R / T jobs U copies stay in the shared / T measurement files; UCache overrides only put; no U file loads code by
    path."""
    assert issubclass(UCache, r_store.RCache) and {k for k in vars(UCache) if not k.startswith("__")} == {"put"}
    for obj in (r_jobs.kc_activity_job, r_jobs.r_arm_job, r_jobs.r_oracle_job, RMeasurer, r_store.RCache):
        assert Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix() in R_MEASURE_FILES
    for obj in (t_measure.t_ref_job, t_measure.t_rest_job):
        assert Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix() in T_MEASURE_FILES
    copies = {"u_kc_activity_job", "u_arm_job", "u_oracle_job", "u_ref_job", "u_rest_job", "u_rig", "u_engine",
              "apply_u_edit", "UPool", "UZMeasurer"}
    shared = {"kc_activity_job", "r_arm_job", "r_oracle_job", "RMeasurer", "RCache", "ECache", "q_oracle_job",
              "arm_job", "q_rig", "reference_job", "rest_job", "engine_for", "apply_q_edit", "apply_csc_edit",
              "t_ref_job", "t_rest_job", "z_engine", "ZMeasurer"}
    for p in sorted((ROOT / "flymon/brain").glob("u_*.py")) + [ROOT / "scripts/run_u.py"]:
        tree = ast.parse(p.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not defs & shared, (p.name, defs & shared)
        if p.name != "u_measure.py":
            assert not defs & copies, (p.name, defs & copies)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not names & {"importlib", "exec", "spec_from_file_location", "runpy"}, p.name
    assert U_MEASURE_FILES == ("flymon/brain/u_measure.py",)
    assert set(UM.JOBS) == {r_jobs.kc_activity_job, r_jobs.r_arm_job, r_jobs.r_oracle_job}


def _cli():
    spec = importlib.util.spec_from_file_location("run_u_for_test", ROOT / "scripts/run_u.py")
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
    for st in ("reuse", "path", "set", "scan", "kc", "even", "choose", "gate2"):
        assert cli.exit_code(st, {"outcome": "PASS"}) == 0 and cli.exit_code(st, {"outcome": "INVALID"}) == 5
    for st, o in (("reuse", "STOP_REUSE"), ("path", "STOP_U_PATH_REPRO"), ("set", "STOP_SET_SHORT"),
                  ("scan", "STOP_NO_QUALIFIED_F"), ("kc", "STOP_NO_QUALIFIED_F"), ("even", "STOP_EVEN_REPRO"),
                  ("choose", "STOP_EVEN_PUNISH"), ("choose", "STOP_EVEN_LOW_LEVER"),
                  ("gate2", "STOP_PUNISH_WEAKENED")):
        assert cli.exit_code(st, {"outcome": o}) == 3, (st, o)
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    assert cli.exit_code("oc", {}) == 0 and cli.exit_code("gate2_oc", {}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "INVALID"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert set(cli.POOL_STAGES) == {"path", "scan", "kc", "even", "smoke", "gate2", "jm"}
    assert set(cli.STAGES) == {s.split(":")[0] for s in UR.ORDER} | {"recompute", "invalid_run"}
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_u.py").read_text()


def test_hashed_pipeline_and_decision_files():
    from flymon.brain.t_runner import T_HASHED_FILES
    assert set(UR.U_PIPELINE_FILES) == {f"flymon/brain/u_{n}.py" for n in
                                        ("spec", "store", "records", "rules", "runner")} | {"scripts/run_u.py"}
    assert not set(UR.U_PIPELINE_FILES) & (set(R_MEASURE_FILES) | set(T_MEASURE_FILES) | set(U_MEASURE_FILES))
    assert set(T_HASHED_FILES) | set(UR.U_PIPELINE_FILES) | set(U_MEASURE_FILES) <= set(UR.U_HASHED_FILES)
    assert "results/summary/t_lever.json" in UR.U_HASHED_FILES
    assert set(UR.DECISION_FILES) == (set(UR.U_HASHED_FILES) - set(R_MEASURE_FILES) - set(T_MEASURE_FILES)
                                      - set(U_MEASURE_FILES))
    assert [f for f in UR.U_HASHED_FILES if not (ROOT / f).exists()] == []
    assert len(UR.pipeline_key()["key"]) == 64 and set(UR.pipeline_key()["files"]) == set(UR.U_PIPELINE_FILES)


def test_cli_measure_factory_builds_rmeasurers_over_upool_and_ucache_only(tmp_path, monkeypatch):
    from flymon.brain.config import Params
    from flymon.brain.u_spec import SPEC as U_SPEC
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    ctx = dict(params=Params(), readout=READOUT, types=TYPES, n_kc=100)
    measure = cli.make_measure("even", UCODE, None, ctx)
    m_h4, m_f = measure(Z), measure(ZF)
    assert measure({k: list(v) for k, v in Z.items()}) is m_h4 and m_f is not m_h4 and m_f.cache is m_h4.cache
    for m in (m_h4, m_f):
        assert isinstance(m, RMeasurer) and type(m.cache) is UCache and m.spec is U_SPEC
        assert isinstance(m.pool, UM.UPool) and m.cache.root == U_SPEC.cache_dir and m.cache.code == UCODE
    m_f.cache.put("r_oracle", {"act_seeds": [500], "edit": u_edit(0.3)}, {"x": 1}, [Params()])
    written = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.startswith(u_store.ALLOWED_DIR + "cache/r_oracle/") for p in written), written
    sm = cli.make_measure("smoke", UCODE, None, ctx)(Z)
    assert sm.cache.root == U_SPEC.smoke_cache_dir and sm.cache.root.startswith(u_store.ALLOWED_DIR)
