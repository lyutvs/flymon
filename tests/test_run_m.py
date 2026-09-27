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


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["run_m_spec", "run_m_stage0"])
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2
    assert "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["run_m_spec", "run_m_stage0"])
def test_refuses_an_out_outside_m(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/m0d/l/x"]) == 2
    assert "results/m0d/m/" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["run_m_spec", "run_m_stage0"])
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
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    s = tmp_path / "m.json"
    s.write_text(json.dumps({"spec_check": {"outcome": "SPEC_GO", "measure_key": "x", "code": {"key": "y"}}}))
    rc = mod.main(["--summary", str(s), "--allow-dirty"])
    err = capsys.readouterr().err
    assert rc == 2 and ("other code" in err or "do not exist" in err or "does not exist" in err)


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
    monkeypatch.setattr(mod, "code_keys", lambda p: (code, "m" * 64))
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
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), "m" * 64))
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
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), "m" * 64))
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
        "npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}), "m" * 64))
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
