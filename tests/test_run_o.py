# tests/test_run_o.py
"""Spec O.5 / O.7.5 and readings 13, 15, 18, 19: the O CLIs refuse before any pool on another root, an --out outside
results/o/, dirty hashed files, a block already in the real summary (no post-hoc rewrite), an uncommitted spec, and (O2)
a block n1 whose oracle o disagrees with O.7.4's; O1 asks for every (condition, cell, seed) once with the four edits and
writes its block (JUDGED exit 0, STOP_NO_SILENT_STATE exit 5); O2 asks for every (X, arm, seed) with the declared flags
and writes its block. Every O hashed file exists."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flymon.brain import o_cli, o_measure
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_measure import HASHED_FILES
from flymon.brain.o_rules import BISTABLE_NETWORK, DEPRESSION_PRESENT, JUDGED, READOUT_PATH, STOP_NO_SILENT_STATE
from flymon.brain.o_spec import SPEC, smoke
from flymon.brain.odor_real import load_table

ROOT = Path(__file__).resolve().parents[1]
SM = smoke(SPEC)
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])
KEY = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}
TYPES = sorted({"ORN_" + p for v in load_table(ROOT / N_SPEC.data_dir, N_SPEC.sha_pins()).glomeruli.values() for p in v})
ORACLE = dict(declared=dict(SPEC.oracle_o), exact={"sim": -2.3482, "dis": -2.4325})
STAGES = ["run_o1", "run_o2"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def _prow(seed, A=20.0, P=40, apl=0.1, kc=100):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=kc, kc_max_win_hz=50.0, apl_out_per_step=apl,
                wall_s=0.01, steps=1400)


class FakePool:
    def close(self):
        pass


class FakeMeasurer:
    """O1: on / kc silent on every second seed (P 0, APL 1.0, KC 10), else firing; O2: arm drops plastic 10, punish 15."""

    def __init__(self, silent=lambda i: i % 2 == 0):
        self.silent, self.calls = silent, []

    def o1_presentations(self, params, items, readout):
        self.calls.append(("o1", sorted({i["edit"] for i in items}), len(items)))
        out = []
        for it in items:
            q = self.silent(it["seed"] - SM.o1_seed0) and it["cond"] in ("on", "kc")
            out.append(dict(_prow(it["seed"], P=0 if q else 40, apl=1.0 if q else 0.1, kc=10 if q else 100),
                            edit=it["edit"], csc_sha256="sha-" + it["edit"], cond=it["cond"], g=it["g"],
                            stim=it["stim"]))
        return out

    def o2_arms(self, params, items, readout, punish_type):
        self.calls.append(("o2", sorted({(i["arm"], i["punish"], i["plastic"], i["da_zero"]) for i in items}),
                           len(items)))
        out = []
        for it in items:
            i = it["seed"] - SM.o2_seed0
            drop = {"plastic": 10.0, "frozen": 0.0, "punish": 15.0, "da_zero": 0.0}[it["arm"]]
            jit = 0.5 * (i % 2) if it["arm"] in ("plastic", "punish") else 0.0
            moved = it["arm"] in ("plastic", "punish")
            out.append(dict(seed=it["seed"], edit=it["edit"], arm=it["arm"], punish=it["punish"],
                            plastic=it["plastic"], da_zero=it["da_zero"], csc_sha256="sha-none",
                            pre={"x": _prow(it["seed"]), "y": _prow(it["seed"])},
                            post={"x": _prow(it["seed"], A=20.0 - drop - jit), "y": _prow(it["seed"])},
                            weights_frac=0.9 if moved else 1.0, weights_frac_A=0.8 if moved else 1.0,
                            weights_frac_P=1.0, w0_sha256="w0", w_post_sha256="w1" if moved else "w0",
                            da_integral={"PPL105": 1.0}, wall_s=1.0, x=it["x"], y=it["y"]))
        return out


def _patch(mod, monkeypatch, measurer=None, commit="c0ffee", oracle=(ORACLE, None)):
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "code_keys", lambda npz: KEY)
    monkeypatch.setattr(mod, "m0d_sha", lambda spec: "s" * 64)
    monkeypatch.setattr(mod, "load_c3", lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, ZU, None))
    monkeypatch.setattr(mod, "model_types", lambda npz: TYPES)
    monkeypatch.setattr(mod, "spec_commit", lambda path: commit)
    monkeypatch.setattr(mod, "n_oracle", lambda spec, smoke: oracle)
    monkeypatch.setattr(mod, "make_measurer",
                        lambda ctx: (measurer, FakePool()) if measurer else pytest.fail("no pool may start here"))
    monkeypatch.setattr(mod, "nonkc_edit", lambda ctx: dict(NONKC))
    monkeypatch.setattr(o_cli, "out_allowed", lambda out: True)


NONKC = dict(edit="apl_to_nonkc_zero", readout_p="MBON05", apl_to_p_edges=2, apl_to_p_zeroed=2,
             apl_to_nonkc_zeroed=394)


def test_every_hashed_file_exists():
    assert [f for f in HASHED_FILES if not (ROOT / f).exists()] == []


@pytest.mark.parametrize("name", STAGES)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_out_outside_results_o(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/n/x"]) == 2 and "results/o/" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_dirty_hashed_files(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/o_jobs.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


@pytest.mark.parametrize("name, block", [("run_o1", "o1"), ("run_o2", "o2")])
def test_never_rewrites_a_block_of_the_real_summary(name, block, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    p = tmp_path / "results/summary/o_states.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({block: {"outcome": JUDGED}}))
    assert mod.main([], require_root=False) == 2 and "post-hoc" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_uncommitted_spec(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, commit=None)
    assert mod.main([], require_root=False) == 2 and "spec" in capsys.readouterr().err


def test_o2_refuses_a_disagreeing_oracle(tmp_path, monkeypatch, capsys):
    mod = _script("run_o2")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, oracle=(None, "block n1's oracle effects disagree"))
    assert mod.main(["--smoke"], require_root=False) == 2 and "disagree" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_n_oracle_reads_block_n1(tmp_path):
    s = tmp_path / "n.json"
    s.write_text(json.dumps({"n1": {"pairs": {"sim": {"o": -2.3482251}, "dis": {"o": -2.4325499}}}}))
    ok, why = o_cli.n_oracle(SPEC, False, committed=lambda *a: None, path=s)
    assert why is None and ok["declared"] == dict(SPEC.oracle_o) and ok["exact"]["sim"] == -2.3482251
    s.write_text(json.dumps({"n1": {"pairs": {"sim": {"o": -2.36}, "dis": {"o": -2.4325}}}}))
    assert "disagree" in o_cli.n_oracle(SPEC, False, committed=lambda *a: None, path=s)[1]
    assert o_cli.n_oracle(SPEC, False, committed=lambda *a: "not tracked", path=s)[1] == "not tracked"
    ok, why = o_cli.n_oracle(SPEC, True, committed=lambda *a: "never asked", path=tmp_path / "none.json")
    assert why is None and ok["exact"] is None


def test_o1_smoke_judges_and_asks_for_every_unit_once(tmp_path, monkeypatch, capsys):
    mod = _script("run_o1")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    assert fm.calls == [("o1", sorted(e for _, e in SPEC.o1_conditions), SM.n_o1_presentations())]
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o1"]
    assert doc["outcome"] == JUDGED and doc["nature"]["label"] == BISTABLE_NETWORK
    assert doc["pathway"]["label"] == READOUT_PATH and doc["spec_commit"] == "c0ffee" and doc["smoke"] is True
    assert Path(doc["report"]).exists() and "O1: 혼합 칸" in capsys.readouterr().out


def test_o1_without_silence_stops_with_exit_5(tmp_path, monkeypatch):
    mod = _script("run_o1")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(silent=lambda i: False))
    assert mod.main(["--smoke"], require_root=False) == 5
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o1"]
    assert doc["outcome"] == STOP_NO_SILENT_STATE and doc["nature"] is None


def test_o2_smoke_judges_both_x(tmp_path, monkeypatch, capsys):
    mod = _script("run_o2")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    assert fm.calls == [("o2", sorted(SPEC.o2_arms), SM.n_o2_arms())]
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o2"]
    assert doc["outcome"] == JUDGED and set(doc["x"]) == {x for x, _, _ in SPEC.o2_pairs}
    assert doc["x"]["4:1"]["labels"]["depression"] == DEPRESSION_PRESENT
    assert doc["oracle"] == json.loads(json.dumps(ORACLE)) and doc["point"] == {"g": 0.25, "c_delta": 8.0}
    assert "O2: X = 4:1" in capsys.readouterr().out


# ---- Task 7 rulings (Task 2 / Task 4 reviews): the APL->MBON05 edit check, separate cache roots, missing hashed files
def test_o1_is_invalid_without_a_zeroed_apl_to_mbon05_edge_and_starts_no_pool(tmp_path, monkeypatch):
    mod = _script("run_o1")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)                                   # no measurer: a pool start fails the test
    monkeypatch.setattr(mod, "nonkc_edit", lambda ctx: dict(NONKC, apl_to_p_zeroed=0))
    assert mod.main(["--smoke"], require_root=False) == 5
    doc = json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o1"]
    assert doc["outcome"] == "INVALID" and "MBON05" in " ".join(doc["reasons"])
    assert doc["nonkc_edit"]["apl_to_p_zeroed"] == 0 and doc["n_presentations"] == 0


def test_o1_records_the_nonkc_edit_check(tmp_path, monkeypatch):
    mod = _script("run_o1")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer())
    assert mod.main(["--smoke"], require_root=False) == 0
    assert json.loads(Path(o_cli.SMOKE_SUMMARY).read_text())["o1"]["nonkc_edit"] == NONKC


class _Conn:
    def __init__(self, types):
        self.type = types


def test_nonkc_edit_counts_the_apl_to_p_edges_the_rig_zeroed(conn_pops_fake):
    got = o_cli.count_nonkc_edit(*conn_pops_fake)
    assert got["apl_to_p_edges"] == 2 and got["apl_to_p_zeroed"] == 2 and got["apl_to_nonkc_zeroed"] == 3
    assert got["readout_p"] == "MBON05" and got["edit"] == "apl_to_nonkc_zero"


@pytest.fixture
def conn_pops_fake():
    """A stand-in rig: cells 0 APL, 1-2 KC, 3-4 MBON05, 5 other; APL -> {1, 3, 4, 5}, KC 1 -> 3."""
    import numpy as np
    from types import SimpleNamespace
    csc = SimpleNamespace(ptr=np.array([0, 4, 5, 5, 5, 5, 5]), tgt=np.array([1, 3, 4, 5, 3], np.int32),
                          w=np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32))
    eng = SimpleNamespace(N=6, csc=csc)
    pops = SimpleNamespace(apl=np.array([0]), kc=np.array([1, 2]))
    conn = _Conn(np.array(["APL", "KC", "KC", "MBON05", "MBON05", "PN"]))
    return conn, pops, eng, {"A": "MBON13", "P": "MBON05"}


@pytest.mark.parametrize("name", STAGES)
def test_smoke_and_real_runs_use_separate_cache_roots(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    seen = []
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "make_measurer", lambda ctx: seen.append(Path(ctx.out)) or (_Stop(), FakePool()))
    for argv in (["--smoke"], []):
        with pytest.raises(_Halt):
            mod.main(argv, require_root=False)
    assert seen == [Path(o_cli.SMOKE_OUT), Path(o_cli.RUN_OUT)]
    assert not (tmp_path / o_cli.RUN_OUT).exists() and not (tmp_path / o_cli.SMOKE_OUT).exists()
    assert mod.main(["--smoke", "--out", o_cli.RUN_OUT], require_root=False) == 2
    assert mod.main(["--out", o_cli.SMOKE_OUT], require_root=False) == 2
    assert mod.main(["--smoke", "--summary", o_measure.SUMMARY], require_root=False) == 2
    assert mod.main(["--summary", o_cli.SMOKE_SUMMARY], require_root=False) == 2
    assert capsys.readouterr().err.count("smoke") >= 4 and seen == [Path(o_cli.SMOKE_OUT), Path(o_cli.RUN_OUT)]


class _Halt(Exception):
    pass


class _Stop:
    """A measurer that stops the stage once the cache root has been seen."""

    def o1_presentations(self, *a):
        raise _Halt

    def o2_arms(self, *a):
        raise _Halt


def test_o_cli_refuses_when_a_hashed_file_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(o_cli, "HASHED_FILES", o_cli.HASHED_FILES + ("flymon/brain/no_such_file.py",))
    with pytest.raises(SystemExit) as e:
        o_cli.code_keys(o_cli.NPZ)
    assert e.value.code == 2 and "no_such_file" in capsys.readouterr().err


def test_pool_settings_come_from_the_spec(monkeypatch):
    from types import SimpleNamespace
    from flymon.brain import fly_pool
    got = {}
    monkeypatch.setattr(fly_pool, "FlyPool", lambda *a, **k: got.update(k) or FakePool())
    ctx = SimpleNamespace(args=SimpleNamespace(npz="x.npz", workers=3), c3=Params(), spec=SM, out=Path("results/o/x"),
                          key=KEY[0], rid="r", spec_commit="c")
    m, pool = o_cli.make_measurer(ctx)
    assert got["timeout_s"] == N_SPEC.pool_timeout_s and got["workers"] == 3
    assert got["punish_type"] == N_SPEC.h3.punish_type and got["reward_type"] == N_SPEC.h3.reward_type
    assert m.cache.root == Path("results/o/x/cache") and m.cache.spec_commit == "c"
    assert o_cli.parser(None).parse_args([]).workers == N_SPEC.workers


def test_o2_passes_the_c3_z_and_the_declared_oracle(tmp_path, monkeypatch):
    mod = _script("run_o2")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer())
    got = {}
    real = mod.o2_judge
    monkeypatch.setattr(mod, "o2_judge", lambda rows, z, o, spec: got.update(z=z, o=o) or real(rows, z, o, spec))
    assert mod.main(["--smoke"], require_root=False) == 0
    assert got == {"z": ZU, "o": dict(SPEC.oracle_o)}
