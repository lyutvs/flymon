# tests/brain/test_r_judge.py
"""The judgement end of the R chain (R.3, R.5, R.9.7; Readings 10, 14, 15, 17, 18) and its mutation tests: jm blocks
carry no pair statistic and are written only when every pair is back; the seal re-checks every raw file, runs the
pre-read validity and archives the raw files; the judge reads only a sealed set, re-checks the raw files and writes the
band, the sentence and the records; a gate block deleted, L edges 1 / 3, an edit in C, a tampered digest, a changed
raw file or an earlier committed judge all refuse or mark the set; the judgement set is touched only from jm on."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import r_jobs
from flymon.brain import r_runner as RR
from flymon.brain.r_measure import RMeasurer
from flymon.brain.r_spec import SPEC
from flymon.brain.r_store import RCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.r_world import Scripted, World, doc, through_gate3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to_seal(w, m):
    r = through_gate3(w, m)
    for n in SPEC.cond_names:
        r.stage_jm(n)
    return r


def test_full_chain_reads_selected(w):
    m = Scripted(plan=w.pass_plan(n=14, c=8, f_a=3))
    r = _to_seal(w, m)
    jm = doc()["jm:L"]
    assert jm["n_pairs"] == 53 and jm["edit_edges"] == [2] and len(jm["manifest"]) == 53
    assert "pairs" not in jm and "aggregate" not in jm and "counts" not in jm       # nothing to read before the seal
    seal = r.stage_seal()
    assert seal["status"] == "SEALED" and seal["n_files"] == 159
    assert all(Path(f["dst"]).exists() for f in seal["archive"]["files"])
    assert seal["archive"]["dir"].startswith(str(w.archive))
    out = r.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)
    assert "(14/21 대 8/21, 여유 ≥ 2, F_a 3/32, 처벌 가드 통과, 짝수 12/21" in out["sentence"]
    assert out["g_fail"]["g_fail"] is False and out["records"]["transitions_C_to_L"]["b"]["testable"]["fp"] == 6
    assert out["oc_sha256"] == doc()["oc"]["sha256"] and set(doc()) == set(RR.ORDER)
    with pytest.raises(SystemExit):
        r.stage_judge()                                     # once


def test_punish_guard_band_through_the_runner(w):
    plan = w.pass_plan(n=14, c=8, f_a=3)
    jb = w.keys(w.judge, "b")
    plan[("judge", "L")].update({k: (False, False, False) for k in jb[18:21]})       # 3 pass->fail on (b)
    r = _to_seal(w, Scripted(plan=plan))
    r.stage_seal()
    out = r.stage_judge()
    assert out["band"] == "B_처벌가드" and out["g_fail"]["axes"]["b"]["pass_to_fail"] == 3
    assert "처벌 통과 (b) 18 대 21" in out["sentence"]


def test_judgement_set_is_touched_only_from_jm_on(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))
    assert w.judgement_calls == 0
    r.stage_jm("L")
    assert w.judgement_calls == 1


def test_jm_writes_nothing_until_complete(w):
    m = Scripted(plan=w.pass_plan())
    r = through_gate3(w, m)
    m.fail_once = True
    with pytest.raises(RuntimeError):
        r.stage_jm("L")
    assert "jm:L" not in doc()
    assert r.stage_jm("L")["n_pairs"] == 53


def test_mutation_a_deleted_gate_block_refuses_the_judgement(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))
    d = doc()
    del d["gate2"]
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and w.judgement_calls == 0


@pytest.mark.parametrize("edges", [1, 3])
def test_mutation_l_edges_other_than_2_seal_invalid(w, edges):
    m = Scripted(plan=w.pass_plan(), override={("judge", "L"): dict(edges=edges)})
    r = _to_seal(w, m)
    seal = r.stage_seal()
    assert seal["status"] == "INVALID" and seal["archive"] is None
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_an_edit_in_c_fails_the_preread_validity(w):
    m = Scripted(plan=w.pass_plan(), override={("judge", "C"): dict(edges=2, edit=SPEC.lever_edit, sha="sha-L")})
    r = _to_seal(w, m)
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and any("C" in x for x in seal["reasons"])
    with pytest.raises(SystemExit):
        r.stage_judge()


def test_mutation_a_tampered_digest_refuses(w):
    """The digest-check split (D8): this test pins only the runner's side — a ValueError from ctx["judgement_rows"]
    (what r_pairs.judgement_rows raises when check_set finds a digest, n_a or last turn other than R's declared values
    and block set's) becomes a refusal (exit 2) with no jm block written. The digests themselves are checked in
    r_pairs.check_set / judgement_rows, on every call (jm, seal, judge, recompute), and pinned by Task 2's
    test_judgement_rows_check_the_digests_and_refuse_smoke; the runner never re-derives or skips them."""
    r = through_gate3(w, Scripted(plan=w.pass_plan()))

    def bad():
        raise ValueError("judgement set digest_keys: generated '00', block set '33be', declared '33be'")
    w.ctx["judgement_rows"] = bad
    with pytest.raises(SystemExit) as e:
        r.stage_jm("L")
    assert e.value.code == 2 and "jm:L" not in doc()


def test_judge_refuses_a_raw_file_changed_after_the_seal(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    assert r.stage_seal()["status"] == "SEALED"
    f = Path(doc()["jm:C"]["manifest"][0]["cache_file"])
    f.write_text(f.read_text().replace('"sha-C"', '"sha-X"'))
    with pytest.raises(SystemExit):
        r.stage_judge()
    assert "judge" not in doc()


def test_an_earlier_committed_judge_refuses_the_judgement(w):
    r = through_gate3(w, Scripted(plan=w.pass_plan()))
    w.judge_commits.append("abc")
    with pytest.raises(SystemExit):
        r.stage_jm("L")


def test_a_changed_code_key_refuses_the_judgement(w):
    m = Scripted(plan=w.pass_plan())
    through_gate3(w, m)
    with pytest.raises(SystemExit):
        w.runner(m, code="z" * 64).stage_jm("L")


def test_recovery_after_reading(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    r.stage_seal()
    r.stage_judge()
    with pytest.raises(SystemExit):
        r.stage_recompute("")
    e = r.stage_recompute("records code fix (test)")
    assert e["band"] == "SELECTED" and e["differs_from_judge"] is False
    r.stage_recompute("second")
    assert len(doc()["recompute"]) == 2
    assert r.stage_invalid_run("measurement defect (test)")["status"] == "INVALID_RUN"
    with pytest.raises(SystemExit):
        r.stage_invalid_run("again")
    with pytest.raises(SystemExit):
        r.stage_recompute("after invalid")


def test_mutation_a_manifest_pointing_c_at_l_files_fails_the_preread_validity(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    d = doc()
    d["jm:C"]["manifest"] = d["jm:L"]["manifest"]
    Path(SPEC.summary).write_text(json.dumps(d))
    seal = r.stage_seal()
    assert seal["status"] == "NOT_READ" and seal["archive"] is None
    assert any(x.startswith("C: ") and "stored inputs" in x for x in seal["reasons"])


def test_seal_refuses_after_an_earlier_committed_judge(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan()))
    w.judge_commits.append("abc")
    with pytest.raises(SystemExit):
        r.stage_seal()
    assert "seal" not in doc()


class OraclePool:
    """r_oracle_job-shaped outputs for a real RMeasurer + RCache; the pair key travels in the odour name ("G|<key>"
    E-grid, "E|<key>" E0); plan[("judge", cond)][key] = fake_oracle flags."""

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
    """A real RMeasurer + RCache writes the judgement entries (inputs with a real Params, canonicalised by
    RCache.put); the seal's stored-inputs check (canonical(stored) == canonical(judge_inputs(...))) passes on them,
    and the judge reads the band from the re-loaded files."""
    for r in w.judge:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan(n=14, c=8, f_a=3)
    through_gate3(w, Scripted(plan=plan))
    pool = OraclePool(plan)
    real = w.runner(RMeasurer(pool, RCache(SPEC.cache_dir, {"key": "r" * 64}), SPEC, w.ctx["params"], READOUT, Z,
                              TYPES, 100))
    for n in SPEC.cond_names:
        assert real.stage_jm(n)["n_pairs"] == 53
    assert pool.jobs == 159
    stored = json.loads(Path(doc()["jm:L"]["manifest"][0]["cache_file"]).read_text())["inputs"]
    assert isinstance(stored["params"], dict) and stored["block"] == "judge" and stored["condition"] == "L"
    seal = real.stage_seal()
    assert seal["status"] == "SEALED" and seal["reasons"] == [], seal["reasons"][:3]
    out = real.stage_judge()
    assert (out["status"], out["band"], out["n"], out["c"], out["F_a"]) == ("READ", "SELECTED", 14, 8, 3)


WATCH = {"judgement_set", "judgement_rows", "_judgement_rows", "judge_seeds", "judge_act_seeds",
         "judge_select_seeds", "judge_report_seeds"}
ALLOWED = {"r_spec.py": {"judge_seeds"}, "r_pairs.py": {"judgement_rows"},
           "r_records.py": {"preread_validity", "judge_inputs"},
           "r_runner.py": {"build_ctx", "_judgement_rows", "stage_jm", "stage_seal", "_read"}}


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
    files = sorted((ROOT / "flymon/brain").glob("r_*.py")) + [ROOT / "scripts/run_r.py"]
    for p in files:
        owners = _owners(ast.parse(p.read_text()))
        assert owners <= ALLOWED.get(p.name, set()), (p.name, owners)


def _cli():
    spec = importlib.util.spec_from_file_location("run_r_for_test", ROOT / "scripts/run_r.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_argument_refusals_and_exit_codes(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.chdir(tmp_path)
    for argv in (["--stage", "gate1", "--condition", "L"], ["--stage", "jm"], ["--stage", "recompute"],
                 ["--stage", "smoke", "--note", "x"], ["--stage", "gate1", "--rerun-after-invalid"], [],
                 ["--list-refs", "--stage", "repro"], ["--stage", "repro"]):     # the last: wrong cwd
        assert cli.main(argv) == 2, argv
    assert cli.exit_code("repro", {"passed": True}) == 0 and cli.exit_code("repro", {"passed": False}) == 4
    assert cli.exit_code("gate1", {"outcome": "PASS"}) == 0
    assert cli.exit_code("gate1", {"outcome": "STOP_STRENGTH_LEVER"}) == 3
    assert cli.exit_code("gate2", {"outcome": "INVALID"}) == 5
    assert cli.exit_code("gate3", {"outcome": "STOP_C_EVEN_MISMATCH"}) == 3
    assert cli.exit_code("smoke", {"problems": []}) == 0 and cli.exit_code("smoke", {"problems": ["x"]}) == 6
    assert cli.exit_code("oc", {}) == 0 and cli.exit_code("recompute", {"band": "SELECTED"}) == 0
    assert cli.exit_code("seal", {"status": "SEALED"}) == 0 and cli.exit_code("seal", {"status": "NOT_READ"}) == 6
    assert cli.exit_code("judge", {"status": "READ"}) == 0 and cli.exit_code("judge", {"status": "NOT_READ"}) == 6
    assert "if __name__ == \"__main__\":" in (ROOT / "scripts/run_r.py").read_text()


# ---- final-review fixes: the judgement read once (I1), NOT_READ prints nothing (M1) ------------------------------
def _sealed(w):
    r = _to_seal(w, Scripted(plan=w.pass_plan(n=14, c=8, f_a=3)))
    assert r.stage_seal()["status"] == "SEALED"
    return r


def test_seal_pins_the_decision_files():
    from flymon.brain.r_measure import R_MEASURE_FILES
    assert set(RR.DECISION_FILES) == set(RR.R_HASHED_FILES) - set(R_MEASURE_FILES)
    assert "flymon/brain/r_rules.py" in RR.DECISION_FILES and "flymon/brain/r_records.py" in RR.DECISION_FILES
    assert not set(RR.DECISION_FILES) & set(R_MEASURE_FILES)


def test_judge_twice_with_the_first_block_discarded_refuses(w):
    r = _sealed(w)
    seal = doc()["seal"]
    assert seal["decision"]["key"] == RR.decision_key()["key"]
    assert r.stage_judge()["status"] == "READ"
    marker = json.loads(Path(RR.JUDGE_MARKER).read_text())
    assert marker["seal_written_at"] == seal["written_at"] and marker["read_at"]
    d = doc()
    del d["judge"]                                  # the uncommitted judge block discarded (git checkout)
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc()


def test_judge_refuses_a_decision_file_changed_after_the_seal(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(RR, "decision_key", lambda: dict(key="d" * 64, files={}))
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2 and "judge" not in doc() and not Path(RR.JUDGE_MARKER).exists()


def test_recompute_and_invalid_run_work_after_a_committed_judge_and_its_marker(w):
    r = _sealed(w)
    r.stage_judge()
    assert Path(RR.JUDGE_MARKER).exists()
    e = r.stage_recompute("records code fix (test)")
    assert e["band"] == "SELECTED" and e["decision_key"] == RR.decision_key()["key"]
    assert e["decision_changed_since_seal"] is False
    assert r.stage_invalid_run("measurement defect (test)")["status"] == "INVALID_RUN"


def test_not_read_judge_returns_no_numbers_and_marks_the_read(w, monkeypatch):
    r = _sealed(w)
    monkeypatch.setattr(RR.r_rules, "read_band", lambda *a, **k: dict(band=RR.r_rules.NOT_READ, reason="COUNTS"))
    out = r.stage_judge()
    assert out["status"] == RR.r_rules.NOT_READ and set(out) == {"status", "band", "reason", "reasons"}
    assert "judge" not in doc() and Path(RR.JUDGE_MARKER).exists()
    with pytest.raises(SystemExit) as e:
        r.stage_judge()
    assert e.value.code == 2
