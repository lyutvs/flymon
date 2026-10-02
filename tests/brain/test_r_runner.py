# tests/brain/test_r_runner.py
"""The R stage chain up to gate ③ (R.2, R.5, R.9.6; Readings 3, 4, 6, 14-16): repro -> smoke -> oc -> gate1 -> gate2
-> even -> gate3; each stage refuses (exit 2, nothing written) when an earlier block is missing, a later one exists,
its own block exists (a failed repro and gate ②'s one INVALID rerun excepted), the summary is uncommitted or a hashed
file is dirty; a failed repro, a smoke problem or a gate STOP / INVALID blocks every later stage; the judgement set is
never touched before the judgement stages."""
import importlib.util
from pathlib import Path

import pytest

from flymon.brain import r_jobs, r_rules
from flymon.brain import r_runner as RR
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.q_spec import SPEC as Q_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.r_spec import SPEC
from flymon.brain.r_store import RCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, key_of
from tests.brain.r_world import Scripted, World, doc, ref_name, through_gate3

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_chain_to_gate3_never_touches_the_judgement_set(w):
    m = Scripted(plan=w.pass_plan())
    through_gate3(w, m)
    d = doc()
    assert set(d) == set(RR.ORDER[:RR.ORDER.index("jm:L")])
    assert d["repro"]["kc"]["passed"] and d["repro"]["p"]["passed"] and d["repro"]["oracle"]["passed"]
    assert d["repro"]["csc_sha256_none"] == "sha-C" and len(d["repro"]["p"]["rows"]) == 18
    assert d["smoke"]["oracle"]["L"]["edit_edges"] == [2] and d["smoke"]["p"]["edit_edges"] == [2]
    assert d["gate1"]["record"]["median"] == pytest.approx(0.045) and d["gate1"]["record"]["original"]["ok"] is False
    assert d["gate2"]["label"] == "LEARNS_CONFIRMATORY" and d["gate2"]["seeds"] == list(SPEC.p.seeds)
    assert (d["gate3"]["testable_b"], d["gate3"]["c_even"]) == (12, 7)
    assert d["gate3"]["records"]["transitions_C_to_L"]["b"]["testable"]["fp"] == 5
    assert w.judgement_calls == 0
    assert not any(c[1] == "judge" for c in m.calls if c[0] == "oracle")
    assert all(c[3] == 2 for c in m.calls if c[0] == "oracle" and c[1] == "smoke")


def test_repro_refuses_without_refs(w):
    (Path(SPEC.ref_dir) / ref_name(SPEC.kc_repro_odours[1])).unlink()
    with pytest.raises(SystemExit) as e:
        w.runner(Scripted()).stage_repro()
    assert e.value.code == 2 and not Path(SPEC.summary).exists()


def test_repro_fails_on_a_wrong_reference_and_can_be_rerun(w):
    m = Scripted()
    m.kc_override[SPEC.kc_repro_odours[0]] = [0.05] * 7 + [0.06]
    r = w.runner(m)
    out = r.stage_repro()
    assert out["passed"] is False and not out["kc"]["odours"][0]["equal"]
    with pytest.raises(SystemExit) as e:
        r.stage_smoke()
    assert e.value.code == 4                                  # a later stage after a failed repro (Reading 4)
    m.kc_override.clear()
    assert r.stage_repro()["passed"] is True                  # a failed repro block may be rewritten (R.9.6)
    with pytest.raises(SystemExit):
        r.stage_repro()                                       # a passed one may not


def test_order_and_rewrite_refusals(w):
    r = w.runner(Scripted(plan=w.pass_plan()))
    with pytest.raises(SystemExit):
        r.stage_gate1()                                       # repro, smoke, oc missing
    assert not Path(SPEC.summary).exists()
    r.stage_repro()
    r.stage_smoke()
    with pytest.raises(SystemExit):
        r.stage_smoke()                                       # own block exists
    with pytest.raises(SystemExit):
        r.stage_even()                                        # oc, gate1, gate2 missing


def test_dirty_summary_or_hashed_file_refuses(w, monkeypatch):
    r = w.runner(Scripted())
    r.stage_repro()
    monkeypatch.setattr(RR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_smoke()
    monkeypatch.setattr(RR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(RR, "git_state", lambda: dict(commit="c", dirty_hashed=["flymon/brain/r_jobs.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_smoke()


def test_smoke_problems_block_later_stages(w):
    m = Scripted(override={("smoke", "C"): dict(edges=2, edit=SPEC.lever_edit, sha="sha-L")})
    r = w.runner(m)
    r.stage_repro()
    sm = r.stage_smoke()
    assert sm["problems"]
    with pytest.raises(SystemExit):
        r.stage_oc()


def test_gate1_stop_is_recorded_and_blocks_gate2(w):
    r = w.runner(Scripted(kc=0.02))
    r.stage_repro(); r.stage_smoke(); r.stage_oc()
    g1 = r.stage_gate1()
    assert g1["outcome"] == "STOP_STRENGTH_LEVER" and "0.0200" in g1["sentence"]
    with pytest.raises(SystemExit):
        r.stage_gate2()


def test_gate2_stop_and_the_one_invalid_rerun(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, p_label="NO_LEARNING")
    r = w.runner(Scripted())
    r.stage_repro(); r.stage_smoke(); r.stage_oc(); r.stage_gate1()
    g2 = r.stage_gate2()
    assert g2["outcome"] == "STOP_PUNISH_BROKEN" and "NO_LEARNING" in g2["sentence"]
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                             # only an INVALID block reruns
    with pytest.raises(SystemExit):
        r.stage_even()


def test_gate2_invalid_reruns_once_with_new_code_and_the_judgement_chain_accepts_it(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)                          # smoke's P arms judged with the default label
    m = Scripted(plan=w.pass_plan())
    r = w.runner(m)
    r.stage_repro(); r.stage_smoke(); r.stage_oc(); r.stage_gate1()
    w.p_label = "INVALID"
    assert r.stage_gate2()["outcome"] == "INVALID"
    with pytest.raises(SystemExit):
        r.stage_gate2(rerun=True)                             # same code key
    w.p_label = "LEARNS_CONFIRMATORY"
    before = Path(SPEC.summary).read_text()
    broken = Scripted(plan=w.pass_plan())
    broken.arms = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("worker died"))
    with pytest.raises(RuntimeError):
        w.runner(broken, code="t" * 64).stage_gate2(rerun=True)
    assert Path(SPEC.summary).read_text() == before           # one write, after measuring (nothing moved yet)
    r2 = w.runner(m, code="s" * 64)
    assert r2.stage_gate2(rerun=True)["outcome"] == "PASS"
    d = doc()
    assert d["gate2_invalid"]["outcome"] == "INVALID" and d["gate2"]["outcome"] == "PASS"
    with pytest.raises(SystemExit):
        r2.stage_gate2(rerun=True)                            # once only
    r2.stage_even()
    assert r2.stage_gate3()["outcome"] == "PASS"
    assert r2.stage_jm("L")["n_pairs"] == 53                  # pre-gate2 blocks may carry the INVALID run's key


@pytest.mark.parametrize("even_l,even_c,outcome", [(16, 8, "STOP_C_EVEN_MISMATCH"), (10, 7, "STOP_EVEN_LOW_LEVER")])
def test_gate3_stops(w, even_l, even_c, outcome):
    r = w.runner(Scripted(plan=w.pass_plan(even_l=even_l, even_c=even_c)))
    r.stage_repro(); r.stage_smoke(); r.stage_oc(); r.stage_gate1(); r.stage_gate2(); r.stage_even()
    g3 = r.stage_gate3()
    assert g3["outcome"] == outcome and g3["sentence"]
    with pytest.raises(SystemExit):
        r.stage_jm("L")
    assert w.judgement_calls == 0



class JobPool:
    """A pool returning r_jobs-shaped outputs, so a real RMeasurer (and RCache) produces gate ① / ③'s inputs. The pair
    key travels in the odour name ("G|<key>" E-grid, "E|<key>" E0); plan[(block, cond)][key] = fake_oracle flags."""

    def __init__(self, plan):
        self.n_workers, self.plan = 4, plan

    def run_jobs(self, fn, kws):
        if fn is r_jobs.kc_activity_job:
            return [[dict(i=i, seed=s, kc=list(range(5)), n=[1] * 5, max_win=5,
                          csc_sha256="sha-L" if kw["edit"] == SPEC.lever_edit else "sha-C",
                          edit_edges=SPEC.lever_edges if kw["edit"] == SPEC.lever_edit else 0)
                     for i, _, s in kw["items"]] for kw in kws]
        assert fn is r_jobs.r_oracle_job
        out = []
        for kw in kws:
            lever = kw["edit"] == SPEC.lever_edit
            kind, key = next(iter(kw["odor_x"])).split("|", 1)
            cond = "L" if lever else ("C" if kind == "G" else "E0")
            flags = self.plan.get(("even", cond), {}).get(key, (False, True, False))
            out.append(fake_oracle(*flags, sha="sha-L" if lever else "sha-C", edges=SPEC.lever_edges if lever else 0,
                                   edit=kw["edit"], n_rep=len(kw["report_seeds"]), n_act=len(kw["act_seeds"])))
        return out


def test_gate1_and_gate3_pass_on_what_the_real_measurer_produces(w):
    """The records stage_gate1 / stage_even build from RMeasurer's own output (not a hand-written dict) have the
    shapes r_rules.gate1 / gate3 read: edit_edges lists, one CSC sha, n_seeds [8]; C's sha is the repro's."""
    for r in w.even:
        r["odor_x"], r["odor_x_e0"] = {f"G|{key_of(r)}": 1.0}, {f"E|{key_of(r)}": 1.0}
    plan = w.pass_plan()
    s = w.runner(Scripted(plan=plan))
    s.stage_repro(); s.stage_smoke(); s.stage_oc()
    real = w.runner(RMeasurer(JobPool(plan), RCache(SPEC.cache_dir, {"key": "k"}), SPEC, w.ctx["params"], READOUT,
                              Z, TYPES, 100))
    g1 = real.stage_gate1()
    rec = doc()["gate1"]["record"]
    assert g1["outcome"] == "PASS" and rec["median"] == pytest.approx(0.05)
    assert (rec["edit_edges"], rec["csc_sha256"], rec["n_seeds"], rec["n_odours"]) == ([2], ["sha-L"], [8], 112)
    assert r_rules.gate1(rec, True, SPEC)["outcome"] == "PASS"
    assert s.stage_gate2()["outcome"] == "PASS"
    real.stage_even()
    ev = doc()["even"]["conditions"]
    assert (ev["L"]["edit_edges"], ev["L"]["csc_sha256"]) == ([2], "sha-L")
    assert (ev["C"]["edit_edges"], ev["C"]["csc_sha256"]) == ([0], "sha-C")
    assert (ev["E0"]["edit_edges"], ev["E0"]["csc_sha256"]) == ([0], "sha-C")
    assert all(not ev[n]["reasons"] for n in SPEC.cond_names)
    g3 = real.stage_gate3()
    assert (g3["outcome"], g3["testable_b"], g3["c_even"]) == ("PASS", 12, 7)
    assert r_rules.gate3(ev["L"], ev["C"], SPEC, doc()["repro"]["csc_sha256_none"])["outcome"] == "PASS"

def test_p_items_are_run_p_items_with_the_edit():
    spec = importlib.util.spec_from_file_location("run_p_for_test", ROOT / "scripts/run_p.py")
    run_p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_p)
    st = {n: {"odor": {f"G_{n}": 1.0}} for xy in P_SPEC.pairs().values() for n in xy}
    assert RR.p_items(P_SPEC, st, "none") == run_p.items_p(P_SPEC, st)
    lev = RR.p_items(SPEC.p, st, SPEC.lever_edit)
    assert len(lev) == 192 and {i["edit"] for i in lev} == {SPEC.lever_edit}
    assert {i["seed"] for i in lev} == set(SPEC.p.seeds)


NPZ = ROOT / "data/malecns.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="no connectome")
def test_build_ctx_on_the_real_data(monkeypatch):
    monkeypatch.chdir(ROOT)
    ctx = RR.build_ctx(SPEC, str(NPZ))
    assert len(ctx["even_rows"]) == 39 and len(ctx["calib"][0]) == 112 and ctx["cap_ok"] is True
    assert ctx["readout"] == {"A": "MBON13", "P": "MBON05"} and ctx["p_block"]["label"] == "LEARNS_CONFIRMATORY"
    names = [ctx["kc_ref"](o) for o in SPEC.kc_repro_odours]
    assert all(n and n.endswith(".json") for n in names) and len(set(names)) == 3
    f = ctx["q0_ref"]([r for r in ctx["even_rows"] if r["axis"] == "b"][0])
    assert f["path"].startswith(Q_SPEC.q0_cache_dir)
