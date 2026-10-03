"""The judgement end of the T chain (T.3 8-10, T.4, T.5, T.6, T.9.2, T.9.6; Readings 4, 8, 10-15) and its mutation
tests: jm:L runs on z_lever and jm:C / jm:E0 on block h4's z, and every block carries no pair statistic and is written
only when all 64 pairs are back; R's reuse condition and the T measurement key are re-checked before the judgement
measurement, the seal and the judge (exit 7); the seal re-checks every raw file and its stored inputs (L's carry
z_lever), archives 192 raw files and pins the decision code; the judge reads L on z_lever and C / E0 on h4 z, once —
with the one re-generation after an interruption between the mark and the block; the records hold the clusters, the
α-fixed sensitivity and z; rs_reread runs only after judge; the CLI's refusals and exit codes; T never defines a copy
of a shared measurement job, measurer or cache; the judgement set is touched only from gate1s (its odours) and jm on."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_measure, r_store, t_store
from flymon.brain import t_runner as TR
from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
from flymon.brain.t_measure import T_MEASURE_FILES
from flymon.brain.t_store import RReadCache, TCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.t_world import CODE, SPEC, TCODE, ZL, World, doc, through_gate3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through_gate3(w, m)
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
    assert jm["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and doc()["jm:C"]["z"] == {"A": [10.0, 9.0],
                                                                                     "P": [26.0, 19.0]}
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm
    stored = json.loads(Path(jm["manifest"][0]["cache_file"]).read_text())["inputs"]
    assert stored["z"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and stored["act_seeds"] == SPEC.judge_seeds()["act"]
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 192 and seal["set"]["n_a"] == 43
    assert seal["z"]["z_lever"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive)) and seal["decision"]["key"] == TR.decision_key()["key"]
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    rho = doc()["gate2"]["ratio"]
    assert (f"(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 "
            f"{rho['r1']['ratio']:.3f}·ℓ_r2 {rho['r2']['ratio']:.3f} ≥ 0.5(약화 정도 기록), 짝수 재판독 12/21") in out["sentence"]
    assert "생성원 턴 0–103" in out["sentence"] and "세 번째 판정" in out["sentence"]
    rec = out["records"]
    assert rec["clusters"]["GROUND* 대 NORMAL"]["L"] == dict(n=21, testable=14, reward_pass=14, punish_pass=21)
    assert rec["alpha_fixed"]["n"] == 14 and "판정 아님" in rec["alpha_fixed"]["note"]
    assert rec["z"]["z_lever"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]}
    assert out["resumed_after_mark"] is False and out["k_even"] == 12
    assert out["oc_sha256"] == doc()["oc"]["independent"]["sha256"] and set(doc()) == set(TR.ORDER)
    assert Path(TR.JUDGE_MARKER).exists() and Path(TR.DONE_MARKER).exists()
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_net_drop_3_on_a_reads_the_punish_guard(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    ja = w.keys(w.judge, "a")
    plan[("judge", "L")].update({k: (False, False, False) for k in ja[30:33]})
    out = _sealed(w, plan).stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["a"]["net_drop"] == 3


def test_b_tb_closes_the_t_claim_only(w):
    out = _sealed(w, w.pass_plan(n=8, c=8, f_a=3)).stage_judge()
    assert out["band"] == "B_Tb" and out["sentence"].endswith(
        "→ 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 결과는 R·S 그대로).")


def test_judgement_set_is_touched_only_from_gate1s_and_jm_on(w):
    r = through_gate3(w, w.scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0 and w.odour_calls == 1
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    m.st["fail_once"] = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 64


@pytest.mark.parametrize("where", ["jm", "seal", "judge"])
@pytest.mark.parametrize("what", ["reuse", "tkey"])
def test_keys_are_rechecked_before_measurement_seal_and_judge(w, where, what):
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    if what == "reuse":
        w.r["gate1"]["code_key"] = "x" * 64
    else:
        r = w.runner(m, tcode={"key": "u" * 64})
    with pytest.raises(SystemExit) as e:
        {"jm": lambda: r.stage_jm("L"), "seal": r.stage_seal, "judge": r.stage_judge}[where]()
    assert e.value.code == TR.EXIT_KEY
    assert not Path(TR.JUDGE_MARKER).exists()


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    r = _to_seal(w, w.scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)}))
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_l_raw_on_h4_z_fails_the_preread_validity(w):
    """A jm:L measured with the wrong z (block h4's) is caught by the stored inputs at the seal."""
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    real_m_for = r._m_for
    r._m_for = lambda name, doc=None: m.at(Z) if name == "L" else real_m_for(name, doc)
    r.stage_jm("L")
    r._m_for = real_m_for
    r.stage_jm("C")
    r.stage_jm("E0")
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any(x.startswith("L: ") and "stored inputs" in x for x in seal["reasons"])


def test_mutation_a_tampered_set_refuses(w):
    r = through_gate3(w, w.scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("T set digest_keys: generated '00', declared '37dd'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through_gate3(w, w.scripted(plan=w.pass_plan()))
    d = doc()
    del d["gate3"]
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
    assert "judge" not in doc() and not Path(TR.JUDGE_MARKER).exists()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(TR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(TR.JUDGE_MARKER).exists()


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(TR.t_rules, "read_band", lambda *a, **k: dict(band=TR.t_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == TR.t_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(TR.JUDGE_MARKER).exists()


class _Kill(Exception):
    pass


def _kill_judge_write(monkeypatch, seen):
    orig = t_store.write_summary_block

    def boom(path, block, obj, plist):
        if block == "judge":
            seen.append(obj)
            raise _Kill()
        return orig(path, block, obj, plist)
    monkeypatch.setattr(TR.t_store, "write_summary_block", boom)
    return orig


def test_resume_after_mark_once(w, monkeypatch):
    r = _sealed(w)
    seen = []
    orig = _kill_judge_write(monkeypatch, seen)
    with pytest.raises(_Kill):
        r.stage_judge()
    mark = json.loads(Path(TR.JUDGE_MARKER).read_text())
    assert "judge" not in doc() and not Path(TR.DONE_MARKER).exists() and mark["t_measure_key"] == "t" * 64
    monkeypatch.setattr(TR.t_store, "write_summary_block", orig)
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
        r.stage_rs_reread()                                 # only after judge
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


def _other_track(rows, plan, root: Path) -> dict:
    """An R- or S-shaped summary: jm:L / jm:C blocks with manifests over written raw files, and a judge block."""
    out = {}
    for n in ("L", "C"):
        man = []
        for r in rows:
            k = key_of(r)
            res = fake_oracle(*plan.get((n, k), (False, True, False)), sha="sha-L" if n == "L" else "sha-C",
                              edges=2 if n == "L" else 0, edit=SPEC.lever_edit if n == "L" else SPEC.no_edit)
            ck = hashlib.sha256(f"{root}|{n}|{k}".encode()).hexdigest()
            f = root / n / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "result": res}))
            man.append(dict(key=k, cache_key=ck, cache_file=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        out[f"jm:{n}"] = dict(manifest=man, seeds=dict(act=list(range(8)), select=list(range(8)),
                                                        report=list(range(8))))
    out["judge"] = dict(band="B_Fa")
    return out


def test_rs_reread_records_rs_and_s_raw_under_ts_z(w, tmp_path):
    from tests.brain.r_fixtures import fake_rows
    rows = fake_rows(21, 32, turn0=64)
    plan = {("L", key_of(r)): (True, True, False) for r in rows[:12]}
    plan.update({("C", key_of(r)): (True, True, False) for r in rows[:6]})
    w.r.update(_other_track(rows, plan, tmp_path / "rraw"))
    w.s = _other_track(fake_rows(21, 43, turn0=104), {}, tmp_path / "sraw")
    r = _sealed(w)
    r.stage_judge()
    out = r.stage_rs_reread()
    assert (out["rows"]["R"]["n"], out["rows"]["R"]["c"], out["rows"]["R"]["n_a"]) == (12, 6, 32)
    assert out["rows"]["S"]["n_a"] == 43 and out["rows"]["R"]["band_judged"] == "B_Fa" and "기록 전용" in out["note"]
    with pytest.raises(SystemExit):
        r.stage_rs_reread()


class OraclePool:
    """r_oracle_job-shaped outputs for real RMeasurers + TCache; the pair key travels in the odour name ("G|<key>"
    E-grid, "E|<key>" E0); the z each job received is recorded."""

    def __init__(self, plan):
        self.n_workers, self.plan, self.jobs, self.z = 4, plan, 0, {}

    def run_jobs(self, fn, kws):
        assert fn is r_jobs.r_oracle_job
        out = []
        for kw in kws:
            self.jobs += 1
            lever = kw["edit"] == SPEC.lever_edit
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            self.z[cond] = kw["z"]
            flags = self.plan.get(("judge", cond), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha="sha-L" if lever else "sha-C", edges=SPEC.lever_edges if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_real_cache_round_trip_seals_and_reads(w):
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through_gate3(w, w.scripted(plan=plan))
    pool, cache, ms = OraclePool(plan), TCache(SPEC.cache_dir, CODE), {}

    def measure(z):
        k = json.dumps({a: list(b) for a, b in sorted(z.items())})
        return ms.setdefault(k, RMeasurer(pool, cache, SPEC, w.ctx["params"], READOUT, z, TYPES, 100))
    real = TR.Runner(measure, None, w.ctx, SPEC, code=CODE, tcode=TCODE, pipeline={"key": "p" * 64},
                     archive_root=w.archive)
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 64
    assert pool.jobs == 192
    assert pool.z["L"] == {k: list(v) for k, v in ZL.items()} and pool.z["C"] == pool.z["E0"] == {
        k: list(v) for k, v in Z.items()}
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds", "judge_select_seeds",
         "judge_report_seeds", "t_set", "set_odours", "_checked"}
ALLOWED = {"t_spec.py": {None, "judge_seeds"}, "t_pairs.py": {"_checked", "judgement_rows", "set_odours"},
           "t_runner.py": {"build_ctx", "_judgement_rows", "stage_set", "stage_gate1s", "stage_jm", "stage_seal",
                           "_read", "_cost"}}


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
    files = sorted((ROOT / "flymon/brain").glob("t_*.py")) + [ROOT / "scripts/run_t.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def test_t_never_defines_a_copy_of_shared_measurement_code():
    """T.3 1 / T.8: T runs R's jobs, measurer and cache themselves; its caches only override put (TCache: T's
    writer; RReadCache: refuse); its own measurement file defines only the reference-set jobs; no T file loads code by
    path."""
    assert t_store.RCache is r_store.RCache and issubclass(TCache, r_store.RCache) and issubclass(RReadCache,
                                                                                                r_store.RCache)
    for cls in (TCache, RReadCache):
        assert {k for k in vars(cls) if not k.startswith("__")} == {"put"}
    for obj in (r_jobs.kc_activity_job, r_jobs.r_arm_job, r_jobs.r_oracle_job, RMeasurer, r_store.RCache):
        src = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT).as_posix()
        assert src in R_MEASURE_FILES, src
    shared = {"kc_activity_job", "r_arm_job", "r_oracle_job", "RMeasurer", "RCache", "ECache", "q_oracle_job",
              "arm_job", "q_rig", "reference_job", "rest_job", "engine_for", "apply_q_edit", "apply_csc_edit"}
    for p in sorted((ROOT / "flymon/brain").glob("t_*.py")) + [ROOT / "scripts/run_t.py"]:
        tree = ast.parse(p.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert not defs & shared, (p.name, defs & shared)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not names & {"importlib", "exec", "spec_from_file_location", "runpy"}, p.name
    assert not set(T_MEASURE_FILES) & set(R_MEASURE_FILES) and r_measure.R_MEASURE_FILES == R_MEASURE_FILES


def _cli():
    spec = importlib.util.spec_from_file_location("run_t_for_test", ROOT / "scripts/run_t.py")
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
    for st in ("reuse", "set", "z", "gate1s", "gate2", "gate3"):
        assert cli.exit_code(st, {"outcome": "PASS"}) == 0 and cli.exit_code(st, {"outcome": "INVALID"}) == 5
    for st, o in (("reuse", "STOP_REUSE"), ("set", "STOP_SET_SHORT"), ("z", "STOP_Z_REPRO"),
                  ("z", "STOP_Z_DEGENERATE"), ("gate1s", "STOP_STRENGTH_LEVER"), ("gate2", "STOP_PUNISH_WEAKENED"),
                  ("gate3", "STOP_EVEN_REPRO"), ("gate3", "STOP_EVEN_LOW_LEVER")):
        assert cli.exit_code(st, {"outcome": o}) == 3, (st, o)
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    assert cli.exit_code("oc", {}) == 0 and cli.exit_code("rs_reread", {}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "INVALID"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert set(cli.POOL_STAGES) == {"z", "gate1s", "smoke", "gate2", "gate3", "jm"}
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_t.py").read_text()


def test_hashed_pipeline_and_decision_files():
    from flymon.brain.s_runner import S_HASHED_FILES
    assert set(TR.T_PIPELINE_FILES) == {f"flymon/brain/t_{n}.py" for n in
                                        ("spec", "pairs", "store", "records", "rules", "runner")} | {"scripts/run_t.py"}
    assert not set(TR.T_PIPELINE_FILES) & (set(R_MEASURE_FILES) | set(T_MEASURE_FILES))
    assert set(S_HASHED_FILES) | set(TR.T_PIPELINE_FILES) | set(T_MEASURE_FILES) <= set(TR.T_HASHED_FILES)
    assert {"results/summary/s_lever.json", "results/summary/r_lever.json",
            "tests/brain/fixtures/t_oc_cluster.json"} <= set(TR.T_HASHED_FILES)
    assert set(TR.DECISION_FILES) == set(TR.T_HASHED_FILES) - set(R_MEASURE_FILES) - set(T_MEASURE_FILES)
    assert [f for f in TR.T_HASHED_FILES if not (ROOT / f).exists()] == []
    assert len(TR.pipeline_key()["key"]) == 64 and set(TR.pipeline_key()["files"]) == set(TR.T_PIPELINE_FILES)


# ---- additions from the Task 5 review (carried into Task 6) ----------------------------------------------------------
@pytest.mark.parametrize("where", ["jm", "seal", "judge"])
@pytest.mark.parametrize("side", ["runner", "block", "both"])
def test_a_missing_t_measurement_key_on_either_side_refuses(w, where, side):
    """Never None == None: a runner without a T measurement key, or a block z without one, refuses with exit 7."""
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    if where in ("seal", "judge"):
        for n in SPEC.cond_names:
            r.stage_jm(n)
    if where == "judge":
        assert r.stage_seal()["status"] == "SEALED"
    if side == "runner":
        r = TR.Runner(m.at, None, w.ctx, SPEC, code=CODE, tcode=None, pipeline={"key": "p" * 64},
                      archive_root=w.archive)
    else:
        d = doc()
        d["z"]["t_measure_key"] = None
        Path(SPEC.summary).write_text(json.dumps(d))
        if side == "both":
            r.t_measure_key = None                                      # None == None: still refused
    before = doc()
    with pytest.raises(SystemExit) as e:
        {"jm": lambda: r.stage_jm("L"), "seal": r.stage_seal, "judge": r.stage_judge}[where]()
    assert e.value.code == TR.EXIT_KEY and doc() == before and not Path(TR.JUDGE_MARKER).exists()


def test_cli_measure_factory_builds_rmeasurers_over_ts_cache_only(tmp_path, monkeypatch):
    """The CLI's measure(z): one RMeasurer per z, T's spec, one TCache rooted at results/t/cache (smoke: its smoke
    root); an entry put through it — gate ③'s L even pairs included — lands only under results/t/."""
    from flymon.brain.config import Params
    from flymon.brain.t_spec import SPEC as T_SPEC
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    ctx = dict(params=Params(), readout=READOUT, types=TYPES, n_kc=100)
    measure = cli.make_measure("gate3", CODE, None, ctx)
    m_h4, m_l = measure(Z), measure(ZL)
    assert measure({k: list(v) for k, v in Z.items()}) is m_h4 and m_l is not m_h4 and m_l.cache is m_h4.cache
    for m in (m_h4, m_l):
        assert isinstance(m, RMeasurer) and type(m.cache) is TCache and m.spec is T_SPEC
        assert m.cache.root == T_SPEC.cache_dir and m.cache.root.startswith(t_store.ALLOWED_DIR)
        assert m.cache.code == CODE
    assert dict(m_l.z) == ZL
    m_l.cache.put("r_oracle", {"act_seeds": [500], "z": {k: list(v) for k, v in ZL.items()}}, {"x": 1}, [Params()])
    written = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.startswith(t_store.ALLOWED_DIR + "cache/r_oracle/") for p in written), written
    sm = cli.make_measure("smoke", CODE, None, ctx)(Z)
    assert sm.cache.root == T_SPEC.smoke_cache_dir and sm.cache.root.startswith(t_store.ALLOWED_DIR)
    assert T_SPEC.cache_dir != T_SPEC.r_cache_dir and not T_SPEC.cache_dir.startswith(("results/r/", "results/s/"))


def test_judgement_set_ast_admits_r_even_reader_and_catches_any_other_access():
    """r_even_reader (Task 5) reads R's even raw and names no part of the judgement set, so the access test admits it;
    any other function naming the judgement set is caught."""
    tree = ast.parse((ROOT / "flymon/brain/t_runner.py").read_text())
    assert "r_even_reader" in {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "r_even_reader" not in _owners(tree) and "stage_rs_reread" not in _owners(tree)
    for src in ("def stage_gate3(self):\n    return self.ctx['judgement_rows']()\n",
                "def r_even_reader(c):\n    return c.judge_seeds()\n",
                "def helper(sp):\n    return sp.judge_report_seeds\n",
                "x = t_pairs.set_odours\n"):
        owners = _owners(ast.parse(src))
        assert owners and not owners <= ALLOWED["t_runner.py"], (src, owners)


def test_rs_reread_is_refused_before_judge_at_every_earlier_point(w):
    """T.9.6: R's and S's judgement raw are read under T's z only after judge — never before (no read, no block)."""
    m = w.scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    reads = []
    w.ctx["r_doc"] = (lambda orig: lambda: reads.append("r") or orig())(w.ctx["r_doc"])
    w.ctx["s_doc"] = lambda: reads.append("s") or {}
    with pytest.raises(SystemExit):
        r.stage_rs_reread()
    for n in SPEC.cond_names:
        r.stage_jm(n)
    assert r.stage_seal()["status"] == "SEALED"
    with pytest.raises(SystemExit):
        r.stage_rs_reread()
    assert "s" not in reads and "rs_reread" not in doc()
