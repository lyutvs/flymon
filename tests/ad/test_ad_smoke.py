"""AD.5's smoke gates: the gate test ids name real tests, the FLY-OS shadow schema, the record-only draw summary, the
stage-1 reproduction comparison, the POOL-response identity with stage 1, and the status precedence."""
import ast
import importlib.util
import json
from pathlib import Path

from flymon.ad import inputs, os_encoder, reuse, smoke
from flymon.ad.spec import LABEL, SPEC
from flymon.ad.spec import smoke as smoke_spec
from flymon.rescope import blocks
from tests.ad.test_ad_reuse import fake_stage1

ROOT = Path(__file__).resolve().parents[2]


def test_gate_test_ids_name_real_tests():
    from flymon.ac import smoke as ac_smoke
    for tid in (smoke.STALE_TESTS + smoke.OS_TESTS + smoke.SEED_TESTS + smoke.MANIFEST_TESTS + ac_smoke.SAFETY_TESTS
                + ac_smoke.EQUALITY_TESTS):
        path, _, name = tid.partition("::")
        assert (ROOT / path).exists(), tid
        if name:
            tree = ast.parse((ROOT / path).read_text())
            names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            assert name in names, tid


def write(p, recs):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))


def dec(fly, os=None):
    r = dict(kind="decision", decider="fly", fly=fly, battle_id="XL-f26-b000", turn=1)
    if os is not None:
        r["os"] = os
    return r


def test_os_shadow_errors_and_draw_record(tmp_path):
    good = dict(true="Snorlax", drawn="Chansey", same_types=True)
    write(tmp_path / "logs/fly02.jsonl", [dec(2, good), dec(2, dict(true="Snorlax", drawn="Lapras", same_types=False)),
                                          dict(kind="decision", decider="coach", fly=2)])
    write(tmp_path / "logs/fly00.jsonl", [dec(0)])
    paths = sorted((tmp_path / "logs").glob("fly*.jsonl"))
    assert smoke.os_shadow_errors(paths, {2}) == []
    write(tmp_path / "logs/fly03.jsonl", [dec(3), dec(3, dict(true="Snorlax", drawn="Lapras"))])
    paths = sorted((tmp_path / "logs").glob("fly*.jsonl"))
    assert len(smoke.os_shadow_errors(paths, {2, 3})) == 2
    rec = smoke.os_draw_record(paths, {2})
    assert rec["n"] == 2 and rec["same_types_share"] == 0.5 and rec["by_true"] == {"Snorlax": {"Chansey": 1, "Lapras": 1}}


def test_os_shadow_errors_flags_a_shadow_on_a_flyl_fly(tmp_path):
    write(tmp_path / "logs/fly00.jsonl", [dec(0, dict(true="Snorlax", drawn="Chansey", same_types=True))])
    assert len(smoke.os_shadow_errors(sorted((tmp_path / "logs").glob("fly*.jsonl")), {2})) == 1


def test_os_shadow_drawn_is_the_seeded_draw_of_the_record(tmp_path):
    """The shape _ADCommon._log writes: top-level `os`, `fly` = pool index, `gfly` = global, the schedule battle id."""
    want = os_encoder.SPECIES[os_encoder.draw_index(os_encoder.battle_seed(SPEC.os_seed, 38, "XL-f38-b000", 3))]
    other = next(s for s in os_encoder.SPECIES if s != want)
    base = dict(kind="decision", decider="fly", fly=2, gfly=38, battle_id="XL-f38-b000", turn=3)
    write(tmp_path / "logs/fly02.jsonl", [dict(base, os=dict(true="Snorlax", drawn=want, same_types=False))])
    paths = [tmp_path / "logs/fly02.jsonl"]
    assert smoke.os_shadow_errors(paths, {2}, os_seed=SPEC.os_seed) == []
    write(tmp_path / "logs/fly02.jsonl", [dict(base, os=dict(true="Snorlax", drawn=other, same_types=False))])
    bad = smoke.os_shadow_errors(paths, {2}, os_seed=SPEC.os_seed)
    assert len(bad) == 1 and "seeded draw" in bad[0]


def test_reproduction_diff_is_byte_level():
    s = [dict(pair=0, side=0, v=[1.0, 2.0], a=[1, 2], p=[3, 4], kc_active=[5, 6], pick=1, tie=False, correct=True)]
    assert smoke.reproduction_diff([dict(fly=0, situations=s)], [dict(fly=0, situations=s)]) == []
    t = [dict(s[0], v=[1.0, 2.0000000000000004])]
    assert smoke.reproduction_diff([dict(fly=0, situations=s)], [dict(fly=0, situations=t)]) == ["fly 0 pair 0 side 0: v"]


def test_reproduction_diff_needs_both_stage1_flies():
    assert smoke.reproduction_diff([], []) == ["no stage-1 records"]


def test_stage1_fly_records_picks_the_flies_at_one_point(tmp_path):
    """reuse.stage1_fly_records (Task 6) - the smoke's stage-1 reproduction is its first consumer."""
    p = tmp_path / "situations_init.jsonl"
    write(p, [dict(fly=g, point=pt, x=g * 10 + pt) for g in (12, 0, 5) for pt in (0, 20)])
    got = reuse.stage1_fly_records(p, ("12", 0), 0)
    assert [(r["fly"], r["point"]) for r in got] == [(0, 0), (12, 0)]
    assert [r["x"] for r in reuse.stage1_fly_records(p, [5], 20)] == [70]
    assert reuse.stage1_fly_records(tmp_path / "missing.jsonl", [0], 0) == []


def test_same_response():
    a = dict(n_situations=79, n_candidates=200, tie_ratio=0.1, a0_ratio=0.2, p0_ratio=0.0)
    assert smoke.same_response(dict(a), a) and not smoke.same_response(dict(a, tie_ratio=0.11), a)


def test_status_precedence():
    groups = dict(smoke=["a", "b"], reuse=["r"], balance=["k"])
    assert smoke.status(dict(a=True, b=True, r=True, k=True), groups) == "OK"
    assert smoke.status(dict(a=True, b=True, r=True, k=False), groups) == "STOP_BALANCE"
    assert smoke.status(dict(a=True, b=True, r=False, k=False), groups) == "STOP_REUSE"
    assert smoke.status(dict(a=False, b=True, r=False, k=False), groups) == "STOP_SMOKE"


# ---- ruling D1: the evaluation digest of what was played -----------------------------------------------------------
def played_tree(tmp_path, drop=None, swap=False):
    s = smoke_spec()
    doc = inputs.inputs_doc(s)
    learn, ev = inputs.schedules_for(s, doc, s.learn_battles)
    recs = [dict(battle_id=sb.battle_id, fly=int(sb.fly_id), block="E", won=True, finished=True, retries=0,
                 invalid=False) for sb in ev if sb.battle_id != drop]
    if swap:                                            # fly 1 logs fly 0's ids: the global fly in the id is wrong
        recs = [dict(r, battle_id=r["battle_id"].replace(f"f{s.layout()[1][2]}", f"f{s.layout()[0][2]}"))
                if r["fly"] == 1 else r for r in recs]
    write(tmp_path / "logs/eval/battles.jsonl", recs)
    return s, doc, ev


def test_eval_played_is_one_digest_equal_to_the_document(tmp_path):
    s, doc, ev = played_tree(tmp_path)
    got = smoke.eval_played(tmp_path / "logs/eval/battles.jsonl", doc["eval"], s.layout(), s.tags[1])
    assert got["problems"] == [] and set(got["digests"].values()) == {doc["digests"]["eval"]}
    assert len(got["digests"]) == len(s.layout())
    assert smoke.eval_shared(got, doc["digests"]["eval"], doc["digests"]["eval"], len(s.layout()))
    assert not smoke.eval_shared(got, doc["digests"]["eval"], "0" * 64, len(s.layout()))


def test_eval_played_flags_a_missing_battle_and_a_wrong_global_fly(tmp_path):
    s, doc, ev = played_tree(tmp_path, drop=ev_id(smoke_spec(), 2, 1))
    got = smoke.eval_played(tmp_path / "logs/eval/battles.jsonl", doc["eval"], s.layout(), s.tags[1])
    assert any("fly 2" in p for p in got["problems"])
    assert not smoke.eval_shared(got, doc["digests"]["eval"], doc["digests"]["eval"], len(s.layout()))
    s, doc, ev = played_tree(tmp_path, swap=True)
    got = smoke.eval_played(tmp_path / "logs/eval/battles.jsonl", doc["eval"], s.layout(), s.tags[1])
    assert any("pool fly 1" in p for p in got["problems"])


def test_eval_played_with_no_battles_is_not_shared(tmp_path):
    s = smoke_spec()
    doc = inputs.inputs_doc(s)
    got = smoke.eval_played(tmp_path / "none.jsonl", doc["eval"], s.layout(), s.tags[1])
    assert not smoke.eval_shared(got, doc["digests"]["eval"], doc["digests"]["eval"], len(s.layout()))


def ev_id(s, i, b):
    return f"{s.tags[1]}-f{s.layout()[i][2]:02d}-b{b:03d}"


# ---- AD.2 (가) before (나); assert_disjoint as a gate; check_inputs without a document --------------------------------
def test_run_inputs_rebuilt_reports_overlapping_teams_as_a_gate(monkeypatch):
    def boom(learn, ev):
        raise ValueError("3 learn/eval battles share teams and order")
    monkeypatch.setattr(blocks, "assert_disjoint", boom)
    doc, err = smoke.run_inputs_rebuilt(smoke_spec())
    assert doc is None and "learning and evaluation schedules share teams" in err and "3 learn/eval" in err


def test_run_inputs_rebuilt_ok():
    doc, err = smoke.run_inputs_rebuilt(smoke_spec())
    assert err is None and doc["digests"]["eval"]


def test_reuse_check_uses_stage1_and_the_run_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_stage1(tmp_path)
    d = dict(confirm=reuse.CONFIRM_DIGEST, eval=reuse.EVAL_DIGEST)
    f0, bad = smoke.reuse_check(d, SPEC)
    assert bad == [] and f0["coff_flies"] == list(SPEC.coff_flies)
    f, bad = smoke.reuse_check(dict(d, eval="0" * 64), SPEC)
    assert bad == ["eval_digest"]
    f, bad = smoke.reuse_check(d, SPEC, run_manifest=dict(reuse=dict(f0, coff_content_hash="x")))
    assert bad == ["coff_content_hash"]
    (tmp_path / "results/m4/stage1/BRAIN/result.json").unlink()
    f, bad = smoke.reuse_check(d, SPEC)
    assert bad == ["missing_stage1"]


def test_check_inputs_tolerates_an_absent_document(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = smoke_spec()
    d = inputs.inputs_doc(s)
    man = inputs.input_manifest(d, tmp_path / "absent.json")
    assert man["inputs_sha256"] is None
    assert inputs.check_inputs(d, dict(man, inputs_sha256="x"), s, tmp_path / "absent.json") == ["inputs_sha256"]


# ---- the script on an empty tree: every gate fails cleanly, the summary is written ---------------------------------
def load_script():
    sp = importlib.util.spec_from_file_location("cads", ROOT / "scripts/check_ad_smoke.py")
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); return mod


def test_script_on_an_empty_tree_is_stop_smoke_with_a_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = load_script()
    monkeypatch.setattr(mod, "run_pytest", lambda ids: 0)
    monkeypatch.setattr(mod.store, "git_provenance", lambda: dict(commit="test"))
    (tmp_path / "suite_exit.txt").write_text("0\n")
    assert mod.main(["--suite-exit", str(tmp_path / "suite_exit.txt")]) == 2
    out = json.loads((tmp_path / smoke.SMOKE).read_text())
    assert out["status"] == "STOP_SMOKE" and out["label"] == LABEL
    g = out["gates"]
    assert g["full_suite"] and g["safety_tests"] and not g["eval_digest_shared"] and not g["stage1_reproduction"]
    assert not g["reuse"] and not g["kc_balance"] and "learn_eval_disjoint" in g
    assert "missing_stage1" in out["reuse_problems"] and "ac_inputs_digests" in out["reuse_problems"]


def test_script_maps_status_to_exit_codes():
    mod = load_script()
    assert mod.EXIT == {"OK": 0, "STOP_SMOKE": 2, "STOP_REUSE": 4, "STOP_BALANCE": 5}
