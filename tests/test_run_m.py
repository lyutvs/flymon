"""Spec M.10: the M CLIs refuse before any pool on another root, --out, a dirty tree, an uncommitted or other-code
summary, a later block, another m0d.json; run_m_spec's pure path on the synthetic connectome; run_m_stage0's flow on
the synthetic connectome with a stand-in measurer (self-check, incumbent guard, arms, teach record, block write)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import m_cli
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.h3_store import sha256_file
from flymon.brain.m_cands import SPEC_GO, cell_weights, cells_digest, core_groups, pair_s, spec_check
from flymon.brain.m_measure import group_react, group_rows
from flymon.brain.m_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[1]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="needs the connectome")
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])
ALL = ["run_m_spec", "run_m_stage0", "run_m_oc", "run_m_stage1", "run_m_stage2", "run_m_list", "run_m_stage3"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ALL)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2
    assert "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", ALL)
def test_refuses_an_out_outside_m(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/m0d/l/x"]) == 2
    assert "results/m0d/m/" in capsys.readouterr().err


@pytest.mark.parametrize("name", ALL)
def test_refuses_dirty_hashed_files(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/m_cli.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_stage0_refuses_an_uncommitted_spec_check(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {"outcome": "SPEC_GO"}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert "tracked" in capsys.readouterr().err


def test_stage0_refuses_a_stopped_spec_check(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {"outcome": "STOP_NO_SPECIFICITY"}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert "STOP_NO_SPECIFICITY" in capsys.readouterr().err


def test_stage0_refuses_a_spec_check_from_other_code(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)                                 # code keys stood in; same_code real
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": _sc_block(measure_key="x")}))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2
    assert "other code" in capsys.readouterr().err


def test_stage0_smoke_reads_only_a_smoke_summary(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {"outcome": "SPEC_GO"}}))
    assert mod.main(["--summary", str(s), "--smoke", "--allow-dirty"]) == 2
    assert "smoke summary under results/m0d/m/" in capsys.readouterr().err


# ---------------------------------------------------------------- the later refusals (code keys stood in)
def _keys(monkeypatch, mod, tmp_path):
    """A stand-in connectome file whose code keys carry the declared sha256 (code_keys refuses until every M script
    exists), and a same_code that accepts the stand-in keys."""
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(b"stand-in")
    code = dict(key="k" * 64, files={"npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256})
    monkeypatch.setattr(mod, "code_keys", lambda p: (code, {"key": "m" * 64, "files": {}}))   # the real shape
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    return str(npz), code


def _sc_block(**kw):
    return dict(dict(outcome=SPEC_GO, measure_key="k" * 64, code={"key": "m" * 64}, run_id="r0",
                     inputs=dict(m0d=dict(path=SPEC.m0d_path, sha256=sha256_file(ROOT / SPEC.m0d_path)))), **kw)


def test_stage0_refuses_a_summary_with_a_later_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": _sc_block(), "stage1": {}}))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2
    assert "later blocks ['stage1']" in capsys.readouterr().err


def test_stage0_refuses_another_m0d(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)
    other = tmp_path / "m0d.json"
    other.write_text((ROOT / SPEC.m0d_path).read_text() + "\n")
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": _sc_block()}))
    assert mod.main(["--npz", npz, "--summary", str(s), "--m0d", str(other)]) == 2
    assert "not the one block spec_check ran on" in capsys.readouterr().err


def test_spec_refuses_a_summary_with_a_later_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_spec")
    monkeypatch.chdir(ROOT)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {}, "stage0": {}}))
    monkeypatch.setattr(mod, "git_state", CLEAN)
    assert mod.main(["--summary", str(s)]) == 2
    assert "later blocks ['stage0']" in capsys.readouterr().err


@needs_npz
def test_stage0_refuses_candidates_other_than_the_pinned_ones(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    _keys(monkeypatch, mod, tmp_path)                                          # the real connectome: default --npz
    real = mod.candidates

    def moved(types, comps, spec):
        out = real(types, comps, spec)
        out["reward"][0] = dict(out["reward"][0], digest="0" * 64)
        return out
    conn = Connectome.load(str(ROOT / "data/malecns.npz"))
    pinned = real(conn.type, compartments(conn, Populations.from_connectome(conn), Params().core_frac), SPEC)
    block = _sc_block(candidates={arm: [dict(name=c["name"], digest=c["digest"]) for c in pinned[arm]]
                                  for arm in pinned}, passing={"reward": ["PAM10"], "punish": []})
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": block}))
    monkeypatch.setattr(mod, "candidates", moved)
    monkeypatch.setattr(mod, "build", lambda *a, **k: pytest.fail("built before the digest refusal"))
    assert mod.main(["--summary", str(s)]) == 2
    assert "digest" in capsys.readouterr().err


# ---------------------------------------------------------------- m_cli
def test_order_out_rule_and_later_blocks(tmp_path):
    assert m_cli.ORDER[0] == "spec_check" and m_cli.ORDER.index("stage0") == 1 and m_cli.POOL_TIMEOUT_S == 3600
    assert m_cli.out_allowed(ROOT / "results/m0d/m/run") and not m_cli.out_allowed(ROOT / "results/m0d/l/run")
    assert not m_cli.out_allowed(ROOT / "results/m0d/mm")
    s = tmp_path / "m.json"
    assert m_cli.later_blocks(s, "spec_check") is None                          # no summary yet
    s.write_text(json.dumps({"spec_check": {}, "oc": {}}))
    assert "['oc']" in m_cli.later_blocks(s, "stage0") and m_cli.later_blocks(s, "oc") is None
    s.write_text("{")
    assert "no usable summary" in m_cli.later_blocks(s, "stage0")


def test_read_previous_gate_order(tmp_path):
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {"outcome": SPEC_GO}}))
    assert m_cli.read_previous(s, ["spec_check"], False, lambda p, b: "not tracked")[1] == "not tracked"
    doc, why = m_cli.read_previous(s, ["spec_check"], False, lambda p, b: None)
    assert why is None and doc["spec_check"]["outcome"] == SPEC_GO
    assert "['stage0']" in m_cli.read_previous(s, ["spec_check", "stage0"], False, lambda p, b: None)[1]
    assert "results/m0d/m/" in m_cli.read_previous(s, ["spec_check"], True, lambda p, b: None)[1]


def test_code_keys_refuse_a_missing_hashed_file(monkeypatch, capsys):
    monkeypatch.setattr(m_cli, "HASHED_FILES", m_cli.HASHED_FILES + ("scripts/no_such_m_script.py",))
    with pytest.raises(SystemExit) as e:
        m_cli.code_keys(ROOT / "pyproject.toml")
    assert e.value.code == 2 and "no_such_m_script.py" in capsys.readouterr().err


# ---------------------------------------------------------------- the synthetic world (no pool, no engine)
def _world(npz):
    """The synthetic connectome's two real core groups (PAM08: MBON01-02, PPL105: MBON03-04) plus two stand-in
    candidates on a single core cell each, under the smoke spec's names (PAM08 PAM10 / PPL103 PPL105)."""
    conn = Connectome.load(str(npz))
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    pam = core_groups(conn.type, comps, "PAM")[0]
    ppl = core_groups(conn.type, comps, "PPL1")[0]
    t = np.asarray(conn.type).astype(str)

    def fake(name, fam, cell):
        return dict(name=name, dans=[name], family=fam, cells=[cell], types=[str(t[cell])], n_dan_cells=3,
                    digest=cells_digest([cell]), w_mbon=[0.5])
    m01, m03 = int(np.flatnonzero(t == "MBON01")[0]), int(np.flatnonzero(t == "MBON03")[0])
    cands = {"reward": [pam, fake("PAM10", "PAM", m01)], "punish": [fake("PPL103", "PPL1", m03), ppl]}
    return conn, pops, cands


def test_spec_pure_path_on_the_synthetic_connectome(synthetic_npz, tmp_path, monkeypatch, capsys):
    mod = _script("run_m_spec")
    conn, pops, cands = _world(synthetic_npz)
    w = {c["name"]: cell_weights(conn, pops, c["cells"], SPEC.k.min_weight) for arm in cands for c in cands[arm]}
    p8, p10 = w["PAM08"] / w["PAM08"].sum(), w["PAM10"] / w["PAM10"].sum()
    fx = np.ones(len(pops.kc))
    even = [dict(key=("b", 0, "x0", "y0"), fx=fx, fy=(p10 < p8).astype(float), pre={}),     # PAM10 strictly above
            dict(key=("b", 2, "x1", "y1"), fx=fx, fy=(p10 < p8).astype(float), pre={}),
            dict(key=("b", 4, "x2", "y2"), fx=fx * 0.5, fy=np.zeros(len(pops.kc)), pre={})]
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(Path(synthetic_npz).read_bytes())
    m0d = tmp_path / "m0d.json"
    m0d.write_text("{}")
    spec = smoke(SPEC)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64, "files": {}}))
    monkeypatch.setattr(mod, "load_c3_record", lambda path, s: (Params(), {"A": "MBON13", "P": "MBON05"}, None, None))
    monkeypatch.setattr(mod, "load_even", lambda s, c3: even)
    monkeypatch.setattr(mod, "candidates", lambda types, comps, s: cands)
    assert mod.main(["--npz", str(npz), "--m0d", str(m0d)], spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads((tmp_path / "results/summary/m_readout.json").read_text())["spec_check"]
    s_all = {n: [pair_s(e["fx"], e["fy"], w[n]) for e in even] for n in w}
    want = spec_check(s_all, spec)
    assert b["outcome"] == want["outcome"] == SPEC_GO and b["passing"]["reward"] == ["PAM10"]
    assert b["medians"] == pytest.approx(want["medians"]) and b["n_pairs"] == 3
    rec = {r["name"]: r for arm in b["candidates"] for r in b["candidates"][arm]}
    assert [r["name"] for r in b["candidates"]["reward"]] == ["PAM08", "PAM10"]
    assert rec["PAM08"]["digest"] == cands["reward"][0]["digest"] and rec["PAM08"]["cells"] == cands["reward"][0]["cells"]
    assert rec["PAM10"]["S"] == pytest.approx(s_all["PAM10"]) and rec["PAM10"]["kc_input"] == pytest.approx(w["PAM10"].sum())
    assert rec["PAM08"]["lobes"]["ab"] == pytest.approx(1.0)                   # every synthetic KC is KCab-m
    assert rec["PAM08"]["n_dan_cells"] == 2 and rec["PAM08"]["w_mbon_stats"]["max"] == pytest.approx(1.0)
    assert b["incumbent_values"]["reward"]["median"] == rec["PAM08"]["median"]
    assert b["inputs"]["m0d"]["sha256"] == sha256_file(m0d) and b["inputs"]["ceiling"]["sha256"] == spec.l.ceiling_sha256
    assert b["sentence"] is None and Path(b["report"]).exists() and b["report_sha256"]
    assert "SPEC_GO" in capsys.readouterr().out


def test_spec_stop_prints_the_sentence_and_smoke_writes_no_block(synthetic_npz, tmp_path, monkeypatch, capsys):
    mod = _script("run_m_spec")
    conn, pops, cands = _world(synthetic_npz)
    even = [dict(key=("b", 0, "x", "y"), fx=np.ones(len(pops.kc)), fy=np.zeros(len(pops.kc)), pre={})]   # S = 1 for all
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(Path(synthetic_npz).read_bytes())
    (tmp_path / "m0d.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64, "files": {}}))
    monkeypatch.setattr(mod, "load_c3_record", lambda path, s: (Params(), {"A": "MBON13", "P": "MBON05"}, None, None))
    monkeypatch.setattr(mod, "load_even", lambda s, c3: even)
    monkeypatch.setattr(mod, "candidates", lambda types, comps, s: cands)
    assert mod.main(["--npz", str(npz), "--m0d", "m0d.json", "--smoke"], require_root=False) == 0
    out = capsys.readouterr().out
    assert "STOP_NO_SPECIFICITY" in out and "X 전용 가중 몫" in out and "summary not written: ['smoke'" in out
    assert not (tmp_path / "results/summary/m_readout.json").exists()
    rep = json.loads(next((tmp_path / "results/m0d/m/smoke/runs").glob("*-spec.json")).read_text())
    assert rep["outcome"] == "STOP_NO_SPECIFICITY" and rep["passing"] == {"reward": [], "punish": []}


class _Pool:
    n_workers = 2

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


ODORS = [dict(name=f"S{i}", seeds=[10 * i + j for j in range(4)]) for i in range(3)]


def _counts(cells, seed, t, fail=()):
    """Per-cell reference counts: 8-20 spikes varying with the seed, 0 for the cells of the `fail` types."""
    return [0 if t[c] in fail else 8 + (seed * (k + 3)) % 13 for k, c in enumerate(cells)]


def _stage0_world(monkeypatch, tmp_path, synthetic_npz, fail=(), bad_z=False, teach_error=False):
    mod = _script("run_m_stage0")
    conn, pops, cands = _world(synthetic_npz)
    t = np.asarray(conn.type).astype(str)
    cells = sorted(int(i) for i in pops.mbon)
    ref = [dict(odor=o["name"], seed=s, counts=_counts(cells, s, t, fail)) for o in ODORS for s in o["seeds"]]
    rest = [dict(seed=s, counts=[1] * len(cells)) for o in ODORS for s in o["seeds"]]
    readout = {"A": "MBON03", "P": "MBON01"}
    tc = {k: [int(i) for i in np.flatnonzero(t == v)] for k, v in readout.items()}
    react = {v: group_react(ref, rest, cells, {v: tc[k]}, SPEC.j.h4)[v] for k, v in readout.items()}
    z = {}
    for k, v in readout.items():
        x = np.array([r["types"][v] for r in group_rows(ref, cells, {v: tc[k]})], float)
        z[k] = [float(x.mean()), float(x.std()) + (1e-12 if bad_z and k == "A" else 0.0)]
    m0d = tmp_path / "m0d.json"
    m0d.write_text(json.dumps({"h4": {"h4": {"combos": {"C3": {"reactivity": react}}}}}))
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(Path(synthetic_npz).read_bytes())
    block = dict(outcome=SPEC_GO, measure_key="k" * 64, code={"key": "m" * 64}, run_id="spec-r",
                 inputs=dict(m0d=dict(path=str(m0d), sha256=sha256_file(m0d))),
                 candidates={arm: [dict(name=c["name"], digest=c["digest"]) for c in cands[arm]] for arm in cands},
                 passing={"reward": ["PAM10"], "punish": ["PPL103"]})
    (tmp_path / "results/summary").mkdir(parents=True)
    (tmp_path / "results/summary/m_readout.json").write_text(json.dumps({"spec_check": block}))
    calls = {"teach": [], "choice": []}

    class M:
        def __init__(self, pool, spec, cache):
            self.params_seen = [Params()]

        def reference(self, params, odors, cells_):
            assert cells_ == cells and odors == ODORS
            return ref

        def rest(self, params, seeds, cells_):
            assert seeds == [s for o in ODORS for s in o["seeds"]]
            return rest

        def teach(self, params, types, arm, punish_type, reward_type):
            calls["teach"].append((tuple(types), arm, punish_type, reward_type))
            return [dict(arm=arm)]

    def choice(rows, t_, arm, h4):
        calls["choice"].append((t_, arm))
        if teach_error:
            raise ValueError("rows must cover the seeds")
        return dict(teachable=True, arm=arm)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64, "files": {}}))
    monkeypatch.setattr(mod, "load_c3_record", lambda path, s: (Params(), readout, z, {"A": ["MBON03"], "P": ["MBON01"]}))
    monkeypatch.setattr(mod, "candidates", lambda types, comps, s: cands)
    monkeypatch.setattr(mod, "build", lambda *a, **k: dict(odors=ODORS))
    monkeypatch.setattr(mod, "FlyPool", _Pool)
    monkeypatch.setattr(mod, "MMeasurer", M)
    monkeypatch.setattr(mod, "teach_choice", choice)
    spec = smoke(SPEC)
    run = lambda *extra: mod.main(["--npz", str(npz), "--m0d", str(m0d), *extra], spec=spec, summary_spec=spec,
                                  require_root=False)
    return run, calls, cands


def test_stage0_done_measures_groups_records_teach_and_writes_the_block(synthetic_npz, tmp_path, monkeypatch):
    run, calls, cands = _stage0_world(monkeypatch, tmp_path, synthetic_npz)
    assert run() == 0
    b = json.loads((tmp_path / "results/summary/m_readout.json").read_text())["stage0"]
    assert b["status"] == "done" and b["self_check"]["ok"] and b["spec_check_run_id"] == "spec-r"
    assert b["inputs"]["m0d"]["sha256"] == sha256_file(tmp_path / "m0d.json")
    g = {x["name"]: x for x in b["groups"]}
    assert [x["name"] for x in b["groups"]] == ["PPL105", "PAM08", "PAM10", "PPL103"]
    assert g["PPL105"]["incumbent"] and g["PAM08"]["incumbent"] and g["PAM10"]["passes"]
    assert g["PAM08"]["cells"] == cands["reward"][0]["cells"] and g["PAM08"]["n_dan_cells"] == 2
    assert g["PAM08"]["w_mbon_stats"]["max"] == pytest.approx(1.0) and len(g["PAM08"]["w_mbon"]) == 2
    assert len(g["PAM08"]["cell_records"]["per_cell"]) == 2 and g["PAM08"]["z"][1] > 0
    assert b["arms"] == {"reward": ["PAM10"], "punish": ["PPL103"]} and b["fixed"] == {}
    assert sorted(calls["teach"]) == [(("MBON01",), "reward_only", "PPL105", "PAM10"),
                                      (("MBON03",), "punish_only", "PPL103", "PAM08")]
    assert g["PAM10"]["teach"]["types"]["MBON01"] == {"teachable": True, "arm": "reward_only"}
    assert "teach" not in g["PAM08"] and b["report_sha256"]


def test_stage0_records_a_teach_error_and_fixes_an_empty_arm(synthetic_npz, tmp_path, monkeypatch):
    run, calls, cands = _stage0_world(monkeypatch, tmp_path, synthetic_npz, teach_error=True)
    mod = sys.modules["run_m_stage0"]
    orig, ppl103 = mod.group_z, cands["punish"][0]["cells"]
    monkeypatch.setattr(mod, "group_z", lambda ref, cells, g, ddof: None if list(g) == ppl103 else orig(ref, cells, g, ddof))
    assert run() == 0
    b = json.loads((tmp_path / "results/summary/m_readout.json").read_text())["stage0"]
    g = {x["name"]: x for x in b["groups"]}
    assert g["PPL103"]["z"] is None and not g["PPL103"]["passes"]                # z invalid: a guard fail
    assert b["arms"] == {"reward": ["PAM10"], "punish": []} and b["fixed"] == {"punish": "PPL105"}
    assert g["PAM10"]["teach"]["types"]["MBON01"] == "rows must cover the seeds" and "teach" not in g["PPL103"]


def test_stage0_self_check_mismatch_exits_5_without_the_block(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _ = _stage0_world(monkeypatch, tmp_path, synthetic_npz, bad_z=True)
    assert run() == 5
    assert "stage0" not in json.loads((tmp_path / "results/summary/m_readout.json").read_text())
    rep = json.loads(next((tmp_path / "results/m0d/m/run/runs").glob("*-stage0.json")).read_text())
    assert rep["status"] == "mismatch" and rep["self_check"]["mismatched"] == ["MBON03"] and "groups" not in rep
    assert "DIFFERS" in capsys.readouterr().out and not calls["teach"]


def test_stage0_a_failing_incumbent_stops_with_exit_5(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _ = _stage0_world(monkeypatch, tmp_path, synthetic_npz, fail=("MBON01", "MBON02"))
    assert run() == 5                                             # PAM08's core is silent; the self-check still holds
    assert "stage0" not in json.loads((tmp_path / "results/summary/m_readout.json").read_text())
    rep = json.loads(next((tmp_path / "results/m0d/m/run/runs").glob("*-stage0.json")).read_text())
    assert rep["status"] == "STOP_INCUMBENT" and rep["self_check"]["ok"] and not calls["teach"]
    assert "STOP_INCUMBENT" in capsys.readouterr().out


def test_stage0_no_candidate_in_either_arm_writes_the_block(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _ = _stage0_world(monkeypatch, tmp_path, synthetic_npz)
    mod = sys.modules["run_m_stage0"]
    orig = mod.group_z
    monkeypatch.setattr(mod, "group_z", lambda ref, cells, g, ddof: None if len(g) == 1 else orig(ref, cells, g, ddof))
    assert run() == 0
    b = json.loads((tmp_path / "results/summary/m_readout.json").read_text())["stage0"]
    assert b["status"] == "STOP_NO_CANDIDATE" and b["arms"] == {"reward": [], "punish": []}
    assert b["fixed"] == {"reward": "PAM08", "punish": "PPL105"} and not calls["teach"]
    assert "반응 가드를 통과한 core 집단이 없었다" in capsys.readouterr().out


def test_stage0_smoke_scans_the_smoke_candidates_and_writes_into_the_smoke_summary(synthetic_npz, tmp_path,
                                                                                    monkeypatch):
    run, calls, _ = _stage0_world(monkeypatch, tmp_path, synthetic_npz)
    monkeypatch.setattr(m_cli, "out_allowed", lambda out: True)              # read_previous's smoke rule (tmp root)
    smoke_sum = tmp_path / "results/m0d/m/smoke/m_readout.json"
    smoke_sum.parent.mkdir(parents=True)
    doc = json.loads((tmp_path / "results/summary/m_readout.json").read_text())
    doc["spec_check"]["passing"] = {"reward": [], "punish": []}               # smoke ignores it
    doc["stage1"] = {}                                                         # and the later-block refusal
    smoke_sum.write_text(json.dumps(doc))
    assert run("--smoke", "--summary", "results/m0d/m/smoke/m_readout.json") == 0
    b = json.loads(smoke_sum.read_text())["stage0"]
    assert b["smoke"] and b["scanned"] == {"reward": ["PAM10"], "punish": ["PPL103"]} and b["status"] == "done"
    assert "stage0" not in json.loads((tmp_path / "results/summary/m_readout.json").read_text())


def test_stage0_refuses_a_passing_name_it_does_not_hold(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _ = _stage0_world(monkeypatch, tmp_path, synthetic_npz)
    p = tmp_path / "results/summary/m_readout.json"
    doc = json.loads(p.read_text())
    doc["spec_check"]["passing"]["reward"] = ["PAM13"]
    p.write_text(json.dumps(doc))
    assert run() == 2 and "PAM13" in capsys.readouterr().err


# ---------------------------------------------------------------- run_m_oc (pure)
def test_oc_rows_are_written_to_a_smoke_summary(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_oc")
    monkeypatch.chdir(tmp_path)                                                # m_store's guard is cwd-relative
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(m_cli, "out_allowed", lambda out: True)               # read_previous's smoke rule (tmp root)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "code_keys", lambda npz: ({"key": None, "files": {}}, {"key": None}))
    monkeypatch.setattr(mod, "same_code", lambda *a: True)
    (tmp_path / "malecns.npz").write_bytes(b"stand-in")
    s = tmp_path / "results/m0d/m/smoke/m.json"
    s.parent.mkdir(parents=True)
    s.write_text(json.dumps({"stage0": {"run_id": "r0", "measure_key": None, "code": None}, "stage1": {}}))
    assert mod.main(["--smoke", "--allow-dirty", "--npz", "malecns.npz", "--summary", str(s)], require_root=False) == 0
    b = json.loads(s.read_text())["oc"]
    assert b["stage0_run_id"] == "r0" and b["smoke"] and b["n_b"] == SPEC.judge_n_b and "smoke" in b["n_b_note"]
    assert len(b["rows"]) == len(SPEC.oc_q_b) * len(SPEC.oc_ratio) * len(SPEC.oc_naive_a) * len(SPEC.oc_c)
    assert all(sum(r["P"].values()) == pytest.approx(1.0) for r in b["rows"]) and b["report_sha256"]
    assert b["assumption"].startswith("u = 0") and b["provenance"]["sha256"]
    assert sum("P(SELECTED)" in ln for ln in capsys.readouterr().out.splitlines()) == len(SPEC.oc_q_b)


def test_oc_refuses_a_missing_stage0_other_code_and_a_later_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_oc")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={}), {"key": "m" * 64}))
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(b"stand-in")
    npz = str(npz)
    s = tmp_path / "m.json"                                   # untracked: OC needs stage0 present only (reading 16)
    s.write_text(json.dumps({"spec_check": {}}))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2 and "['stage0']" in capsys.readouterr().err
    s.write_text(json.dumps({"stage0": dict(measure_key="x", code={"key": "m" * 64})}))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2 and "other code" in capsys.readouterr().err
    s.write_text(json.dumps({"stage0": dict(measure_key="k" * 64, code={"key": "m" * 64}), "stage1": {}}))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2 and "later blocks ['stage1']" in capsys.readouterr().err


def test_oc_table_spec_keeps_the_declared_n_b():
    mod = _script("run_m_oc")
    assert mod.table_spec(SPEC) is SPEC and mod.table_spec(smoke(SPEC)).judge_n_b == SPEC.judge_n_b


# ---------------------------------------------------------------- run_m_stage1: refusals before any pool
def test_stage1_refuses_missing_blocks_listing_each(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage1")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    err = capsys.readouterr().err
    assert "stage0" in err and "oc" in err


def test_stage1_refuses_an_uncommitted_summary(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage1")
    monkeypatch.chdir(ROOT)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {}, "stage0": {}, "oc": {}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2 and "tracked" in capsys.readouterr().err


def test_stage1_refuses_a_broken_run_id_chain(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage1")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "same_code", lambda *a: True)
    monkeypatch.setattr(mod, "code_keys", lambda npz: ({"key": "k", "files": {}}, {"key": "m"}))
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {"run_id": "a"}, "stage0": {"run_id": "b", "spec_check_run_id": "zzz"},
                             "oc": {"run_id": "c", "stage0_run_id": "b"}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert "other upstream runs" in capsys.readouterr().err
    s.write_text(json.dumps({"spec_check": {"run_id": "a"}, "stage0": {"run_id": "b", "spec_check_run_id": "a"},
                             "oc": {"run_id": "c", "stage0_run_id": "q"}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert "block oc was produced on other upstream runs" in capsys.readouterr().err


def _chain(**s0):
    return {"spec_check": _sc_block(), "stage0": dict(_sc_block(run_id="r1", spec_check_run_id="r0"), **s0),
            "oc": _sc_block(run_id="r2", stage0_run_id="r1")}


def test_stage1_refuses_other_code_a_later_block_and_a_stage0_not_done(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage1")
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64}))
    s = tmp_path / "m.json"
    doc = _chain(status="done")
    doc["oc"]["measure_key"] = "x"
    s.write_text(json.dumps(doc))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2 and "['oc'] were produced under other code" in \
        capsys.readouterr().err
    s.write_text(json.dumps(dict(_chain(status="done"), stage2={})))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2 and "later blocks ['stage2']" in capsys.readouterr().err
    s.write_text(json.dumps(_chain(status="STOP_NO_CANDIDATE")))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2 and "STOP_NO_CANDIDATE" in capsys.readouterr().err


def test_stage1_refuses_another_m0d(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage1")
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64}))
    monkeypatch.setattr(mod, "FlyPool", lambda *a, **k: pytest.fail("pool started before the m0d refusal"))
    other = tmp_path / "m0d.json"
    other.write_text((ROOT / SPEC.m0d_path).read_text() + "\n")
    s = tmp_path / "m.json"
    s.write_text(json.dumps(_chain(status="done")))
    assert mod.main(["--npz", npz, "--summary", str(s), "--m0d", str(other)]) == 2
    assert "not the one stage 0 ran on" in capsys.readouterr().err


# ---------------------------------------------------------------- run_m_stage1: the flow with stand-in measurers
BIG = [[3, 0], [4, 0]]                    # dV of these probes (z = (0, 1)) has d' 4.95: r (or -p) passes
ZERO = [[0, 0], [0, 0]]


def _rep(r_ok: bool, p_ok: bool) -> dict:
    """A report whose r >= 2 iff r_ok and -p >= 2 iff p_ok (the P readout silent, z = (0, 1))."""
    r1 = BIG if r_ok else ZERO
    r2 = [[a - (b if p_ok else 0) for a, b in zip(x, y)] for x, y in zip(r1, BIG)]
    return {"pre": {"A": ZERO, "P": ZERO}, "R1": {"A": r1, "P": ZERO}, "R2": {"A": r2, "P": ZERO}}


def _pairs():
    od = lambda k: {"g0": float(k), "g1": 1.0}
    return ([dict(axis="a", turn=0, x="m0", y="m1", odor_x=od(0), odor_y=od(1))]
            + [dict(axis="b", turn=0, x=f"m{i} vs A", y=f"m{i} vs B", odor_x=od(i), odor_y=od(i + 5)) for i in range(3)])


def _stage1_world(monkeypatch, tmp_path, synthetic_npz, plan: dict, self_ok=True, overlap_ppl103=False):
    """plan: {(reward_type, punish_type): [(r_ok, p_ok) per scanned pair]}."""
    import dataclasses
    from flymon.brain.h4_formula import pair_stats
    from flymon.brain.h4_pairs import pair_key, pairs_digest
    from flymon.brain.j_runner import params_json
    mod = _script("run_m_stage1")
    conn, pops, cands = _world(synthetic_npz)
    if overlap_ppl103:                                                         # PPL103's core cell inside PAM08's
        c = cands["punish"][0]
        cands["punish"][0] = dict(c, cells=[cands["reward"][0]["cells"][0]],
                                  digest=cells_digest([cands["reward"][0]["cells"][0]]))
    cells = sorted(int(i) for i in pops.mbon)
    pairs = _pairs()
    even = [p for p in pairs if p["axis"] == "b"][:2]
    base = smoke(SPEC)
    spec = dataclasses.replace(base, j=dataclasses.replace(base.j, h4=dataclasses.replace(
        base.j.h4, pairs_digest=pairs_digest(pairs))))
    readout, z_c3, pools = {"A": "MBON13", "P": "MBON05"}, {"A": [0.0, 1.0], "P": [0.0, 1.0]}, {"A": ["x"], "P": ["y"]}
    c3_rep = _rep(True, False)
    want = dict(zip(("axis", "turn", "x", "y"), pair_key(even[0])), **pair_stats(c3_rep, z_c3, 2.0))
    rec = [want if self_ok else dict(want, r=want["r"] + 1.0),
           dict(zip(("axis", "turn", "x", "y"), pair_key(even[1])), r=0.0, p=-3.0, testable=False)]
    m0d = tmp_path / "m0d.json"
    m0d.write_text(json.dumps({"h4": {"h4": {"combos": {"C3": {"oracle": {"pairs": rec}}}}}}))
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(Path(synthetic_npz).read_bytes())
    kc = {"PAM08": 5.0, "PAM10": 1.0, "PPL103": 2.0, "PPL105": 4.0}
    sc = dict(outcome=SPEC_GO, measure_key="k" * 64, code={"key": "m" * 64}, run_id="r0",
              candidates={arm: [dict(name=c["name"], digest=c["digest"], kc_input=kc[c["name"]]) for c in cands[arm]]
                          for arm in cands})
    groups = [dict(name=c["name"], arm=arm, cells=c["cells"], digest=c["digest"], z=[0.0, 1.0])
              for arm in cands for c in cands[arm]]
    s0 = dict(status="done", measure_key="k" * 64, code={"key": "m" * 64}, run_id="r1", spec_check_run_id="r0",
              inputs=dict(m0d=dict(path=str(m0d), sha256=sha256_file(m0d))), c3=params_json(Params()),
              readout=readout, z_c3=z_c3, pools=pools, cells=cells, groups=groups,
              arms={"reward": ["PAM10"], "punish": ["PPL103"]})
    oc = dict(measure_key="k" * 64, code={"key": "m" * 64}, run_id="r2", stage0_run_id="r1")
    summ = tmp_path / "results/summary/m_readout.json"
    summ.parent.mkdir(parents=True)
    summ.write_text(json.dumps({"spec_check": sc, "stage0": s0, "oc": oc}))
    calls = {"pre": [], "edit": [], "h4": []}

    class H4:
        def __init__(self, pool, h4spec, pairs_, pools_, cache, h3):
            assert pairs_ == even[:1] and pools_ == pools and h3 is None
            self.params_seen = [Params()]

        def oracle(self, params, ro, z):
            calls["h4"].append((ro, z))
            return [dict(report=c3_rep)]

    class M:
        def __init__(self, pool, spec_, cache):
            self.params_seen = [Params()]

        def pre(self, params, pairs_, cells_):
            assert pairs_ == even and cells_ == cells
            calls["pre"].append(len(pairs_))
            return [dict(i=i) for i in range(len(pairs_))]

        def edit(self, params, pairs_, pres, cells_, groups_, ro, z, reward_type, punish_type):
            assert pairs_ == even and pres == [dict(i=i) for i in range(len(pairs_))]
            calls["edit"].append(dict(groups=groups_, readout=ro, z=z, reward=reward_type, punish=punish_type))
            ed = {"reward": {"group": ro["P"], "cells": groups_[ro["P"]]},
                  "punish": {"group": ro["A"], "cells": groups_[ro["A"]]}}
            return [dict(report=_rep(*ok), edited=ed) for ok in plan[(reward_type, punish_type)]]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64}))
    monkeypatch.setattr(mod, "load_c3_record", lambda path, s: (Params(), readout, z_c3, pools))
    monkeypatch.setattr(mod, "even_pairs", lambda pops_: pairs)
    monkeypatch.setattr(mod, "FlyPool", _Pool)
    monkeypatch.setattr(mod, "H4Measurer", H4)
    monkeypatch.setattr(mod, "MMeasurer", M)
    run = lambda *extra: mod.main(["--npz", str(npz), "--m0d", str(m0d), *extra], spec=spec, summary_spec=spec,
                                  require_root=False)
    return run, calls, cands, summ


PLAN = {("PAM08", "PPL105"): [(True, False), (False, True)],                    # reference: n_r 1, n_p 1
        ("PAM10", "PPL105"): [(True, True), (True, False)],                     # reward PAM10: n 2
        ("PAM08", "PPL103"): [(False, True), (True, True)]}                     # punish PPL103: n 2


def test_stage1_scans_both_arms_ranks_and_writes_the_block(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, cands, summ = _stage1_world(monkeypatch, tmp_path, synthetic_npz, PLAN)
    assert run() == 0
    b = json.loads(summ.read_text())["stage1"]
    assert b["status"] == "done" and b["self_check"]["ok"] and calls["pre"] == [2]            # pre once, shared
    assert calls["h4"] == [({"A": "MBON13", "P": "MBON05"}, {"A": [0.0, 1.0], "P": [0.0, 1.0]})]
    assert b["stage0_run_id"] == "r1" and b["oc_run_id"] == "r2" and b["spec_check_run_id"] == "r0"
    assert b["n_pairs"] == 2 and [k[0] for k in b["pairs"]] == ["b", "b"] and len(b["pairs_digest"]) == 64
    ref = next(c for c in calls["edit"] if c["reward"] == "PAM08" and c["punish"] == "PPL105")
    assert ref["readout"] == {"A": "PPL105", "P": "PAM08"} and len(calls["edit"]) == 3          # reference measured once
    rw = next(c for c in calls["edit"] if c["reward"] == "PAM10")
    assert rw["readout"] == {"A": "PPL105", "P": "PAM10"} and rw["punish"] == "PPL105"
    assert rw["groups"] == {"PAM10": cands["reward"][1]["cells"], "PPL105": cands["punish"][1]["cells"]}
    pu = next(c for c in calls["edit"] if c["punish"] == "PPL103")
    assert pu["readout"] == {"A": "PPL103", "P": "PAM08"} and pu["reward"] == "PAM08"
    assert b["reference"]["n_r"] == 1 and b["reference"]["n_p"] == 1 and b["reference"]["edited_ok"]
    e = {x["name"]: x for arm in b["entries"] for x in b["entries"][arm]}
    assert e["PAM10"]["n"] == 2 and e["PPL103"]["n"] == 2 and e["PAM10"]["kc_input"] == 1.0
    assert len(e["PAM10"]["r"]) == 2 and len(e["PPL103"]["p"]) == 2 and e["PAM10"]["edited_ok"]
    assert e["PAM10"]["edited"]["reward"] == {"group": "PAM10", "cells": cands["reward"][1]["cells"]}
    assert b["ranked"] == {"reward": ["PAM10"], "punish": ["PPL103"]}
    assert [t["name"] for t in b["top"]["reward"]] == ["PAM10"] and b["top"]["reward"][0]["digest"]
    pj = b["predicted_joint"]
    assert pj == {"PAM10|PPL103": 2, "PAM10|PPL105": 1, "PAM08|PPL103": 1, "PAM08|PPL105": 0}
    assert b["c3_baseline"] == dict(b["c3_baseline"], n_pairs=2, n_r=1, n_p=1)
    assert b["skipped"] == {"reward": [], "punish": []} and b["report_sha256"] and "cache" in b


def test_stage1_skips_an_overlapping_candidate_and_falls_back_to_the_incumbent(synthetic_npz, tmp_path, monkeypatch):
    run, calls, cands, summ = _stage1_world(monkeypatch, tmp_path, synthetic_npz, PLAN, overlap_ppl103=True)
    assert run() == 0
    b = json.loads(summ.read_text())["stage1"]
    assert b["skipped"] == {"reward": [], "punish": [{"name": "PPL103", "overlaps": "PAM08"}]}
    assert not any(c["punish"] == "PPL103" for c in calls["edit"]) and b["entries"]["punish"] == []
    assert b["top"]["punish"] == [dict(name="PPL105", cells=cands["punish"][1]["cells"],
                                       digest=cands["punish"][1]["digest"], n=1, med=b["top"]["punish"][0]["med"],
                                       incumbent=True)]
    assert set(b["predicted_joint"]) == {"PAM10|PPL105", "PAM08|PPL105"}


def test_stage1_self_check_mismatch_exits_5_before_the_scan(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, summ = _stage1_world(monkeypatch, tmp_path, synthetic_npz, PLAN, self_ok=False)
    assert run() == 5
    assert "stage1" not in json.loads(summ.read_text()) and not calls["pre"] and not calls["edit"]
    rep = json.loads(next((tmp_path / "results/m0d/m/run/runs").glob("*-stage1.json")).read_text())
    assert rep["status"] == "mismatch" and rep["self_check"]["mismatched"] == ["r"] and "entries" not in rep
    assert "DIFFERS" in capsys.readouterr().out


def test_stage1_refuses_a_stage0_group_whose_digest_moved(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, summ = _stage1_world(monkeypatch, tmp_path, synthetic_npz, PLAN)
    doc = json.loads(summ.read_text())
    doc["stage0"]["groups"][1]["cells"] = doc["stage0"]["groups"][0]["cells"]          # PAM10's cells, digest kept
    summ.write_text(json.dumps(doc))
    assert run() == 2 and "['PAM10']" in capsys.readouterr().err and not calls["h4"]


def test_stage1_refuses_another_even_pair_list(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, _ = _stage1_world(monkeypatch, tmp_path, synthetic_npz, PLAN)
    monkeypatch.setattr(sys.modules["run_m_stage1"], "even_pairs", lambda pops: _pairs()[:3])
    assert run() == 2 and "even pair list differs" in capsys.readouterr().err and not calls["h4"]


# ---------------------------------------------------------------- Task 9: the carried stage-0 fix, the commit gate
def test_stage0_accepts_a_spec_check_from_this_code_with_the_real_code_keys_shape(tmp_path, monkeypatch, capsys):
    """code_keys' real (dict) manifest: a spec_check block written under this code (code = the manifest) passes
    other_code, so stage 0 reaches the later-block refusal (before the fix it read "other code")."""
    mod = _script("run_m_stage0")
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)
    code, manifest = m_cli.code_keys(npz)
    code = dict(code, files=dict(code["files"], **{"npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}))
    monkeypatch.setattr(mod, "code_keys", lambda p: (code, manifest))
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": _sc_block(measure_key=code["key"], code=manifest), "stage1": {}}))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2
    err = capsys.readouterr().err
    assert "later blocks ['stage1']" in err and "other code" not in err


def test_blocks_committed_checks_only_the_named_blocks(tmp_path):
    s = tmp_path / "m.json"
    head = {"spec_check": {"a": 1}, "stage0": {"b": 2}, "oc": {}}
    s.write_text(json.dumps(dict(head, stage1={"new": True})))                  # stage1 written after the commit
    show = lambda p: json.dumps(head)
    assert m_cli.blocks_committed(s, ["spec_check", "stage0", "oc"], show) is None
    s.write_text(json.dumps(dict(head, stage0={"b": 3}, stage1={})))
    assert "uncommitted changes in the blocks ['stage0']" in m_cli.blocks_committed(s, ["spec_check", "stage0"], show)
    assert "not tracked" in m_cli.blocks_committed(s, ["stage0"], lambda p: None)
    assert "['oc']" in m_cli.blocks_committed(s, ["oc"], lambda p: json.dumps({"stage0": {}}))


def test_stage2_commit_gate_names_spec_check_stage0_and_oc(tmp_path, monkeypatch):
    mod = _script("run_m_stage2")
    seen = []
    monkeypatch.setattr(mod, "blocks_committed", lambda summary, blocks: seen.append(list(blocks)))
    mod.check_committed("x", ["spec_check", "stage0", "oc", "stage1"])
    assert seen == [["spec_check", "stage0", "oc"]]


# ---------------------------------------------------------------- Task 9: refusals of stage 2 / list / stage 3
@pytest.mark.parametrize("name,need", [("run_m_stage2", "stage1"), ("run_m_list", "stage2"), ("run_m_stage3", "list")])
def test_later_stages_refuse_missing_blocks(name, need, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert need in capsys.readouterr().err


@pytest.mark.parametrize("name", ["run_m_list", "run_m_stage3"])
def test_refuse_a_stopped_stage2(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "same_code", lambda *a: True)
    monkeypatch.setattr(mod, "code_keys", lambda npz: ({"key": "k", "files": {}}, {"key": "m"}))
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"stage0": {"run_id": "0"}, "stage2": {"run_id": "2", "gate": "STOP_NO_GAIN"},
                             "list": {"run_id": "l", "stage2_run_id": "2"}}))
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert "STAGE2_GO" in capsys.readouterr().err


def test_stage3_refuses_an_edited_summary_after_commit(tmp_path, monkeypatch, capsys):
    """Review Focus 5: a real temp git repo; the summary committed, then edited -> refused before code_keys."""
    import subprocess
    mod = _script("run_m_stage3")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    s = repo / "m.json"
    s.write_text(json.dumps({"stage2": {"gate": "STAGE2_GO"}, "list": {}}))
    subprocess.run(["git", "-C", str(repo), "add", "m.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"], check=True)
    s.write_text(json.dumps({"stage2": {"gate": "STAGE2_GO"}, "list": {"edited": True}}))
    monkeypatch.setattr(mod, "check_committed", lambda summary, blocks: (
        "has uncommitted changes" if subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "m.json"])
        .returncode else None))
    monkeypatch.setattr(mod, "code_keys", lambda npz: pytest.fail("code_keys reached before the commit gate"))
    monkeypatch.chdir(ROOT)
    assert mod.main(["--summary", str(s), "--allow-dirty"]) == 2
    assert "uncommitted" in capsys.readouterr().err


def test_stage3_refuses_another_m0d(tmp_path, monkeypatch, capsys):
    mod = _script("run_m_stage3")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "same_code", lambda *a: True)
    monkeypatch.setattr(mod, "code_keys", lambda npz: ({"key": "k", "files": {"npz:malecns.npz": "x"}}, {"key": "m"}))
    other = tmp_path / "m0d.json"
    other.write_text("{}")
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"stage0": {"run_id": "0", "inputs": {"m0d": {"sha256": "nope"}}},
                             "stage2": {"run_id": "2", "gate": "STAGE2_GO"}, "list": {"run_id": "l", "stage2_run_id": "2"}}))
    rc = mod.main(["--summary", str(s), "--allow-dirty", "--m0d", str(other)])
    err = capsys.readouterr().err
    assert rc == 2 and ("not the one stage 0 ran on" in err or "connectome" in err)


def test_stage3_refuses_the_m0d_with_the_declared_connectome(tmp_path, monkeypatch, capsys):
    """The m0d sha refusal itself (the npz check passes with the declared sha)."""
    mod = _script("run_m_stage3")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "same_code", lambda *a: True)
    monkeypatch.setattr(mod, "code_keys", lambda npz: ({"key": "k", "files": {
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}}, {"key": "m"}))
    monkeypatch.setattr(mod, "FlyPool", lambda *a, **k: pytest.fail("pool started before the m0d refusal"))
    other = tmp_path / "m0d.json"
    other.write_text("{}")
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"stage0": {"run_id": "0", "inputs": {"m0d": {"sha256": "nope"}}},
                             "stage2": {"run_id": "2", "gate": "STAGE2_GO"}, "list": {"run_id": "l", "stage2_run_id": "2"}}))
    assert mod.main(["--summary", str(s), "--allow-dirty", "--m0d", str(other)]) == 2
    assert "not the one stage 0 ran on" in capsys.readouterr().err


@pytest.mark.parametrize("name,block", [("run_m_list", "stage2"), ("run_m_stage3", "list")])
def test_later_stages_refuse_other_code_and_a_broken_chain(name, block, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda npz: ({"key": "k" * 64, "files": {}}, {"key": "m" * 64}))
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(b"stand-in")
    ok = dict(measure_key="k" * 64, code={"key": "m" * 64})
    doc = {"stage0": dict(ok, run_id="0"), "stage2": dict(ok, run_id="2", gate="STAGE2_GO"),
           "list": dict(ok, run_id="l", stage2_run_id="2")}
    doc[block]["measure_key"] = "x"
    s = tmp_path / "m.json"
    s.write_text(json.dumps(doc))
    assert mod.main(["--npz", str(npz), "--summary", str(s), "--allow-dirty"]) == 2
    assert f"['{block}'] were produced under other code" in capsys.readouterr().err
    if name == "run_m_stage3":
        doc[block]["measure_key"] = "k" * 64
        doc["list"]["stage2_run_id"] = "zzz"
        s.write_text(json.dumps(doc))
        assert mod.main(["--npz", str(npz), "--summary", str(s), "--allow-dirty"]) == 2
        assert "block list was produced on other upstream runs" in capsys.readouterr().err


@pytest.mark.parametrize("name,later", [("run_m_stage2", "list"), ("run_m_list", "stage3")])
def test_later_stages_refuse_a_later_block(name, later, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    npz, _ = _keys(monkeypatch, mod, tmp_path)
    ok = dict(measure_key="k" * 64, code={"key": "m" * 64})
    doc = {"spec_check": dict(ok, run_id="r0"), "stage0": dict(ok, run_id="r1", spec_check_run_id="r0"),
           "oc": dict(ok, run_id="r2", stage0_run_id="r1"), "stage1": dict(ok, run_id="r3", stage0_run_id="r1",
                                                                          oc_run_id="r2", status="done"),
           "stage2": dict(ok, run_id="r4", gate="STAGE2_GO"), later: {}}
    s = tmp_path / "m.json"
    s.write_text(json.dumps(doc))
    assert mod.main(["--npz", npz, "--summary", str(s)]) == 2
    assert f"later blocks ['{later}']" in capsys.readouterr().err


# ---------------------------------------------------------------- Task 9: stage 2 / list / stage 3 flows (stand-ins)
N_REP = len(SPEC.j.h4.report_seeds)


def _rep8(ok: bool) -> dict:
    """A report on the report seeds: testable (r = -p = d' 6.5) iff ok; d_pre 0 (naive)."""
    x = [[3 + i % 2, 0] for i in range(N_REP)] if ok else [[0, 0]] * N_REP
    zero = [[0, 0]] * N_REP
    return {"pre": {"A": zero, "P": zero}, "R1": {"A": x, "P": zero},
            "R2": {"A": [[0, 0]] * N_REP if ok else zero, "P": zero}}


def _row(p, ok, **kw):
    return dict(axis=p["axis"], turn=int(p["turn"]), x=p["x"], y=p["y"], report=_rep8(ok), **kw)


def _s0_groups(cands):
    return [dict(name=c["name"], arm=arm, cells=c["cells"], digest=c["digest"], z=[0.0, 1.0],
                 n_dan_cells=10 + i, w_mbon_stats=dict(min=0.1, median=0.5 + i, max=1.0))
            for arm in cands for i, c in enumerate(cands[arm])]


def _stage2_world(monkeypatch, tmp_path, synthetic_npz, plan, bar=11, overlap=False, bad_odd=False):
    """plan: {(reward, punish): [ok per even pair]}; odd: the winner's oks (all True). Even = (a) 2 + (b) 2."""
    import dataclasses
    from flymon.brain.h4_pairs import pairs_digest
    from flymon.brain.j_runner import params_json
    mod = _script("run_m_stage2")
    conn, pops, cands = _world(synthetic_npz)
    if overlap:                                                                # PPL103's core cell = PAM10's
        c = cands["punish"][0]
        cands["punish"][0] = dict(c, cells=list(cands["reward"][1]["cells"]),
                                  digest=cells_digest(cands["reward"][1]["cells"]))
    cells = sorted(int(i) for i in pops.mbon)
    od = lambda k: {"g0": float(k), "g1": 1.0}
    pairs = ([dict(axis="a", turn=t, x=f"a{t}", y=f"a{t}'", odor_x=od(t), odor_y=od(t + 9)) for t in (0, 2)]
             + [dict(axis="b", turn=t, x=f"m{t} vs A", y=f"m{t} vs B", odor_x=od(t), odor_y=od(t + 5)) for t in (0, 2, 4)])
    odd = [dict(axis="b", turn=t, x=f"o{t} vs A", y=f"o{t} vs B", odor_x=od(t), odor_y=od(t + 3)) for t in (1, 3, 5)]
    even = pairs[:4]
    base = smoke(SPEC)
    spec = dataclasses.replace(base, j=dataclasses.replace(base.j, stage2_select_testable_b=bar, h4=dataclasses.replace(
        base.j.h4, pairs_digest=pairs_digest(pairs))), k=dataclasses.replace(base.k, odd_pairs_digest=pairs_digest(
            odd) if not bad_odd else "0" * 64))
    readout, z_c3, pools = {"A": "MBON13", "P": "MBON05"}, {"A": [0.0, 1.0], "P": [0.0, 1.0]}, {"A": ["x"], "P": ["y"]}
    m0d = tmp_path / "m0d.json"
    m0d.write_text("{}")
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(Path(synthetic_npz).read_bytes())
    ok = dict(measure_key="k" * 64, code={"key": "m" * 64})
    sc = dict(ok, outcome=SPEC_GO, run_id="r0", passing={"reward": ["PAM10"], "punish": ["PPL103"]})
    s0 = dict(ok, status="done", run_id="r1", spec_check_run_id="r0",
              inputs=dict(m0d=dict(path=str(m0d), sha256=sha256_file(m0d))), c3=params_json(Params()),
              readout=readout, z_c3=z_c3, pools=pools, cells=cells, groups=_s0_groups(cands))
    top = lambda arm: [dict(name=c["name"], cells=c["cells"], digest=c["digest"]) for c in cands[arm]]
    s1 = dict(ok, status="done", run_id="r3", stage0_run_id="r1", oc_run_id="r2",
              top={"reward": top("reward")[::-1], "punish": top("punish")})             # PAM10 first
    summ = tmp_path / "results/summary/m_readout.json"
    summ.parent.mkdir(parents=True)
    summ.write_text(json.dumps({"spec_check": sc, "stage0": s0, "oc": dict(ok, run_id="r2", stage0_run_id="r1"),
                                "stage1": s1}))
    calls = {"pre": [], "edit": []}

    class M:
        def __init__(self, pool, spec_, cache):
            self.params_seen = [Params()]

        def pre(self, params, pairs_, cells_):
            calls["pre"].append([p["x"] for p in pairs_])
            return [dict(i=i) for i in range(len(pairs_))]

        def edit(self, params, pairs_, pres, cells_, groups_, ro, z, reward_type, punish_type):
            calls["edit"].append(dict(pairs=[p["x"] for p in pairs_], groups=groups_, readout=ro, z=z,
                                      reward=reward_type, punish=punish_type))
            oks = plan[(reward_type, punish_type)] if pairs_ == even else [True] * len(pairs_)
            ed = {"reward": {"group": ro["P"], "cells": groups_[ro["P"]]},
                  "punish": {"group": ro["A"], "cells": groups_[ro["A"]]}}
            return [_row(p, o, edited=ed) for p, o in zip(pairs_, oks)]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64}))
    monkeypatch.setattr(mod, "load_c3_record", lambda path, s: (Params(), readout, z_c3, pools))
    monkeypatch.setattr(mod, "even_pairs", lambda pops_: pairs)
    monkeypatch.setattr(mod, "odd_pairs", lambda pops_: odd)
    monkeypatch.setattr(mod, "FlyPool", _Pool)
    monkeypatch.setattr(mod, "MMeasurer", M)
    run = lambda *extra: mod.main(["--npz", str(npz), "--m0d", str(m0d), *extra], spec=spec, summary_spec=spec,
                                  require_root=False)
    return run, calls, cands, summ


P2 = {("PAM10", "PPL103"): [True, True, True, False],        # (a) 2 testable, (b) 1: testable_b 1, F_a 2
      ("PAM10", "PPL105"): [False, False, True, True],       # testable_b 2, F_a 0
      ("PAM08", "PPL103"): [True, True, True, True],         # testable_b 2, F_a 2: the bar (2) met
      ("PAM08", "PPL105"): [True, False, True, True]}        # testable_b 2, F_a 1


def test_stage2_bar_first_choice_go_gate_and_odd_record(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, cands, summ = _stage2_world(monkeypatch, tmp_path, synthetic_npz, P2, bar=2)
    assert run() == 0
    b = json.loads(summ.read_text())["stage2"]
    assert calls["pre"][0] == ["a0", "a2", "m0 vs A", "m2 vs A"] and calls["pre"][1] == ["o1 vs A", "o3 vs A"]
    assert [c["name"] for c in b["combos"]] == ["PAM10|PPL103", "PAM10|PPL105", "PAM08|PPL103", "PAM08|PPL105"]
    e = next(c for c in calls["edit"] if c["reward"] == "PAM10" and c["punish"] == "PPL103")
    assert e["readout"] == {"A": "PPL103", "P": "PAM10"} and e["groups"] == {
        "PAM10": cands["reward"][1]["cells"], "PPL103": cands["punish"][0]["cells"]}
    assert b["ranked"][0] == "PAM08|PPL103" and b["gate"] == "STAGE2_GO" and b["sentence"] is None
    w = b["winner"]
    assert w["label"] == "PPL103·PAM08" and w["testable_b"] == 2 and w["aggregate"]["F_a"] == 2
    assert w["digests"] == {"PAM08": cands["reward"][0]["digest"], "PPL103": cands["punish"][0]["digest"]}
    assert w["groups"]["PPL103"] == {"n_dan_cells": 10, "w_mbon_median": 0.5}
    assert b["odd"]["n"] == 2 and b["odd"]["n_testable"] == 2 and b["odd"]["testable"] == [True, True]
    assert calls["edit"][-1]["pairs"] == ["o1 vs A", "o3 vs A"] and calls["edit"][-1]["reward"] == "PAM08"
    assert b["stage1_run_id"] == "r3" and b["stage0_run_id"] == "r1" and b["excluded"] == []
    assert b["n_spec_passing"] == 2 and b["n_b"] == 2 and "m_median" in w["aggregate"]
    assert b["n_cands"] == 2                                                   # the smoke summary spec scans 2


def test_stage2_stop_no_gain_still_records_the_odd_winner(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, summ = _stage2_world(monkeypatch, tmp_path, synthetic_npz, P2)          # the declared bar 11
    assert run() == 0
    b = json.loads(summ.read_text())["stage2"]
    assert b["gate"] == "STOP_NO_GAIN" and "최대 2/21·F_a 2" in b["sentence"] and b["odd"]["n_testable"] == 2
    assert b["winner"]["name"] == "PAM08|PPL103"                                   # testable_b, then F_a


def test_stage2_excludes_overlapping_combinations(synthetic_npz, tmp_path, monkeypatch):
    run, calls, _, summ = _stage2_world(monkeypatch, tmp_path, synthetic_npz, P2, bar=2, overlap=True)
    assert run() == 0
    b = json.loads(summ.read_text())["stage2"]
    assert b["excluded"] == [["PAM10", "PPL103"], ["PAM08", "PPL103"]]                    # PAM10's cell is in PAM08's core
    assert [c["name"] for c in b["combos"]] == ["PAM10|PPL105", "PAM08|PPL105"]


def test_stage2_invalid_rows_exit_3_without_the_block(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, summ = _stage2_world(monkeypatch, tmp_path, synthetic_npz, P2, bar=2)
    mod = sys.modules["run_m_stage2"]
    orig = mod.MMeasurer

    class Short(orig):
        def edit(self, *a, **k):
            rows = super().edit(*a, **k)
            return rows[:-1]                                                   # a missing pair row
    monkeypatch.setattr(mod, "MMeasurer", Short)
    assert run() == 3 and "stage2" not in json.loads(summ.read_text())


def test_stage2_refuses_another_odd_list_before_the_pool(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, _ = _stage2_world(monkeypatch, tmp_path, synthetic_npz, P2, bad_odd=True)
    assert run() == 2 and "odd pair list differs" in capsys.readouterr().err and not calls["pre"]


def test_stage2_refuses_a_top_entry_that_moved(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, _, summ = _stage2_world(monkeypatch, tmp_path, synthetic_npz, P2)
    doc = json.loads(summ.read_text())
    doc["stage1"]["top"]["reward"][0]["cells"] = [0]
    summ.write_text(json.dumps(doc))
    assert run() == 2 and "top entries" in capsys.readouterr().err and not calls["pre"]


def _new_all(n_b=21, turns=(4, 5, 7, 8, 9, 10, 11, 12), early=True):
    """A new_pairs-shaped set: (b) pairs on turns 0-3 (excluded) and the judgement turns, (a) 2 per judgement turn."""
    od = lambda k: {"g0": float(k), "g1": 1.0}
    moves = ["Surf", "Surf", "Ember", "Tackle"]
    b = [dict(axis="b", turn=t, x=f"Early{t} vs A", y=f"Early{t} vs B", odor_x=od(t), odor_y=od(t + 1))
         for t in (range(4) if early else [])]
    i = 0
    while len(b) - (4 if early else 0) < n_b:
        t = turns[i % len(turns)]
        b.append(dict(axis="b", turn=t, x=f"{moves[i % 4]} vs O{i}", y=f"{moves[i % 4]} vs Q{i}", odor_x=od(i),
                      odor_y=od(i + 50)))
        i += 1
    a = [dict(axis="a", turn=t, x=f"a{t}{k}", y=f"a{t}{k}'", odor_x=od(t), odor_y=od(t + 70)) for t in turns
         for k in range(2)]
    return dict(b=b, a=a, skipped=[])


def test_list_writes_the_judgement_list(tmp_path, monkeypatch, capsys):
    import dataclasses
    mod = _script("run_m_list")
    spec = dataclasses.replace(SPEC, judge_b_digest="", judge_a_digest="", judge_n_a=16)
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(b"stand-in")
    s = tmp_path / "results/summary/m_readout.json"
    s.parent.mkdir(parents=True)
    ok = dict(measure_key="k" * 64, code={"key": "m" * 64})
    s.write_text(json.dumps({"stage2": dict(ok, run_id="r4", gate="STAGE2_GO")}))
    got = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64}))
    monkeypatch.setattr(mod, "Connectome", type("C", (), {"load": staticmethod(lambda p: None)}))
    monkeypatch.setattr(mod, "Populations", type("P", (), {"from_connectome": staticmethod(lambda c: None)}))
    monkeypatch.setattr(mod, "new_pairs", lambda pops, lspec: got.setdefault("l", lspec) and _new_all())
    run = lambda: mod.main(["--npz", str(npz)], spec=spec, summary_spec=spec, require_root=False)
    assert run() == 0
    b = json.loads(s.read_text())["list"]
    assert got["l"].a_turns == SPEC.l.n_turns
    assert b["n_b"] == 21 and b["n_a"] == 16 and b["turns"] == [4, 5, 7, 8, 9, 10, 11, 12] and b["stage2_run_id"] == "r4"
    assert all(k[1] >= 4 for k in b["b"]) and len(b["b_digest"]) == 64 and "turns 0-3" in b["exposure_note"]
    assert b["diversity"]["n_moves"] == 3 and b["diversity"]["top_move_text"] == "Surf 11/21"
    wrong = dataclasses.replace(spec, judge_n_a=18)
    s.write_text(json.dumps({"stage2": dict(ok, run_id="r4", gate="STAGE2_GO")}))
    assert mod.main(["--npz", str(npz)], spec=wrong, summary_spec=wrong, require_root=False) == 2
    assert "(a) 16" in capsys.readouterr().err
    pinned = dataclasses.replace(spec, judge_b_digest="0" * 64, judge_a_digest="0" * 64)
    assert mod.main(["--npz", str(npz)], spec=pinned, summary_spec=pinned, require_root=False) == 2
    assert "not the pinned" in capsys.readouterr().err


def _stage3_world(monkeypatch, tmp_path, synthetic_npz, sel_b=12, c3_b=3, fa=2, spec=None, invalid=False):
    """Judgement list 21 (b) + 16 (a) (a stand-in spec: no digests, n_a 16); the winner PPL103·PAM10."""
    import dataclasses
    from flymon.brain.h4_pairs import pair_key, pairs_digest
    from flymon.brain.j_runner import params_json
    from flymon.brain.m_rules import judgement_set
    mod = _script("run_m_stage3")
    conn, pops, cands = _world(synthetic_npz)
    cells = sorted(int(i) for i in pops.mbon)
    spec = spec or dataclasses.replace(SPEC, judge_b_digest="", judge_a_digest="", judge_n_a=16)
    new_all = _new_all()
    js = judgement_set(new_all, spec)
    readout, z_c3, pools = {"A": "MBON13", "P": "MBON05"}, {"A": [0.0, 1.0], "P": [0.0, 1.0]}, {"A": ["x"], "P": ["y"]}
    m0d = tmp_path / "m0d.json"
    m0d.write_text("{}")
    npz = tmp_path / "malecns.npz"
    npz.write_bytes(Path(synthetic_npz).read_bytes())
    ok = dict(measure_key="k" * 64, code={"key": "m" * 64})
    groups = _s0_groups(cands)
    s0 = dict(ok, status="done", run_id="r1", inputs=dict(m0d=dict(path=str(m0d), sha256=sha256_file(m0d))),
              c3=params_json(Params()), readout=readout, z_c3=z_c3, pools=pools, cells=cells, groups=groups)
    g = {x["name"]: x for x in groups}
    w = dict(name="PAM10|PPL103", label="PPL103·PAM10", reward="PAM10", punish="PPL103",
             readout={"A": "PPL103", "P": "PAM10"}, z={"A": [0.0, 1.0], "P": [0.0, 1.0]},
             cells={n: g[n]["cells"] for n in ("PAM10", "PPL103")}, digests={n: g[n]["digest"] for n in ("PAM10", "PPL103")},
             testable_b=13)
    s2 = dict(ok, run_id="r4", stage0_run_id="r1", gate="STAGE2_GO", winner=w, odd=dict(n=20, n_testable=4),
              n_cands=19, n_spec_passing=6)
    lst = dict(ok, run_id="r5", stage2_run_id="r4", b=[list(pair_key(p)) for p in js["b"]],
               a=[list(pair_key(p)) for p in js["a"]], b_digest=pairs_digest(js["b"]), a_digest=pairs_digest(js["a"]),
               diversity=dict(n_moves=3, top_move_text="Surf 11/21"))
    summ = tmp_path / "results/summary/m_readout.json"
    summ.parent.mkdir(parents=True)
    summ.write_text(json.dumps({"stage0": s0, "stage2": s2, "list": lst}))
    calls = {"pre": [], "edit": [], "h4": []}
    oks = lambda nb: ([True] * nb + [False] * (len(js["b"]) - nb))

    class M:
        def __init__(self, pool, spec_, cache):
            self.params_seen = [Params()]

        def pre(self, params, pairs_, cells_):
            calls["pre"].append(len(pairs_))
            return [dict(i=i) for i in range(len(pairs_))]

        def edit(self, params, pairs_, pres, cells_, groups_, ro, z, reward_type, punish_type):
            calls["edit"].append(dict(groups=groups_, readout=ro, z=z, reward=reward_type, punish=punish_type))
            o = oks(sel_b) + [True] * fa + [False] * (len(pairs_) - len(js["b"]) - fa)
            rows = [_row(p, x, edited={}) for p, x in zip(pairs_, o)]
            return rows[:-1] if invalid else rows

    class H4:
        def __init__(self, pool, h4spec, pairs_, pools_, cache, h3):
            assert pairs_ == js["b"] + js["a"] and pools_ == pools and h3 is None
            self.params_seen = [Params()]

        def oracle(self, params, ro, z):
            calls["h4"].append((ro, z))
            return [_row(p, x) for p, x in zip(js["b"] + js["a"], oks(c3_b) + [False] * len(js["a"]))]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "out_allowed", lambda out: True)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda p: (dict(key="k" * 64, files={
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), {"key": "m" * 64}))
    monkeypatch.setattr(mod, "load_c3_record", lambda path, s: (Params(), readout, z_c3, pools))
    monkeypatch.setattr(mod, "new_pairs", lambda pops_, lspec: new_all)
    monkeypatch.setattr(mod, "FlyPool", _Pool)
    monkeypatch.setattr(mod, "MMeasurer", M)
    monkeypatch.setattr(mod, "H4Measurer", H4)
    run = lambda *extra: mod.main(["--npz", str(npz), "--m0d", str(m0d), *extra], spec=spec, summary_spec=spec,
                                  require_root=False)
    return run, calls, summ


def test_stage3_selected_reading_and_sentence(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, summ = _stage3_world(monkeypatch, tmp_path, synthetic_npz)
    assert run() == 0
    b = json.loads(summ.read_text())["stage3"]
    assert calls["pre"] == [37] and calls["h4"] == [({"A": "MBON13", "P": "MBON05"}, {"A": [0.0, 1.0], "P": [0.0, 1.0]})]
    e = calls["edit"][0]
    assert e["readout"] == {"A": "PPL103", "P": "PAM10"} and e["reward"] == "PAM10" and e["punish"] == "PPL103"
    assert b["reading"]["outcome"] == "SELECTED" and b["reading"]["n"] == 12 and b["reading"]["c"] == 3
    assert b["aggregate_selected"]["F_a"] == 2 and b["aggregate_c3"]["testable_b"] == 3
    for part in ("PPL103·PAM10", "12/21 대 3/21", "홀수 기록 4/20", "19후보", "특이성 통과 6개", "기술 3종·Surf 11/21",
                 "DAN 세포 수 PPL 10·PAM 11", "core w_mbon PPL103 0.5·PAM10 1.5"):
        assert part in b["sentence"], part
    assert b["context"]["k"] == 13 and b["list_run_id"] == "r5" and b["stage2_run_id"] == "r4"
    assert b["attempts"][-1]["claim"] == "M" and b["even_usage"] == "G·H·J·K·L·M"


def test_stage3_n_equal_c_at_the_bar_reads_no_conclusion(synthetic_npz, tmp_path, monkeypatch):
    run, _, summ = _stage3_world(monkeypatch, tmp_path, synthetic_npz, sel_b=12, c3_b=12)
    assert run() == 0
    b = json.loads(summ.read_text())["stage3"]
    assert b["reading"]["outcome"] == "B_NO_CONCLUSION" and "C3 자체가" in b["sentence"] and "짝수 선택 13/21" in b["sentence"]


def test_stage3_invalid_rows_exit_3_without_a_block(synthetic_npz, tmp_path, monkeypatch):
    run, _, summ = _stage3_world(monkeypatch, tmp_path, synthetic_npz, invalid=True)
    assert run() == 3 and "stage3" not in json.loads(summ.read_text())


def test_stage3_smoke_sized_list_gives_no_reading_and_no_block(synthetic_npz, tmp_path, monkeypatch, capsys):
    import dataclasses
    sm = dataclasses.replace(smoke(SPEC), judge_from_turn=4, judge_n_b=2)
    run, _, summ = _stage3_world(monkeypatch, tmp_path, synthetic_npz, sel_b=2, c3_b=0, spec=sm)
    assert run() == 0
    assert "stage3" not in json.loads(summ.read_text())
    rep = json.loads(next((tmp_path / "results/m0d/m/run/runs").glob("*-stage3.json")).read_text())
    assert rep["reading"]["outcome"] is None and rep["sentence"] is None and "no_reading" in capsys.readouterr().out


def test_stage3_refuses_a_list_that_differs_from_the_regenerated_one(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, summ = _stage3_world(monkeypatch, tmp_path, synthetic_npz)
    doc = json.loads(summ.read_text())
    doc["list"]["b"] = doc["list"]["b"][1:]
    summ.write_text(json.dumps(doc))
    assert run() == 2 and "block list's keys" in capsys.readouterr().err and not calls["pre"]


def test_stage3_refuses_a_winner_whose_cells_moved(synthetic_npz, tmp_path, monkeypatch, capsys):
    run, calls, summ = _stage3_world(monkeypatch, tmp_path, synthetic_npz)
    doc = json.loads(summ.read_text())
    doc["stage2"]["winner"]["cells"]["PAM10"] = [0]
    summ.write_text(json.dumps(doc))
    assert run() == 2 and "winner" in capsys.readouterr().err and not calls["pre"]
