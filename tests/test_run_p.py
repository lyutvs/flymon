"""Spec P.4 / P.6.4 / P.6.5: the P CLIs refuse before any pool on another root, an --out outside results/p/, dirty
hashed files, a block already in the real summary, an uncommitted spec, a disagreeing block n1; run_p refuses without a
committed block oc, asks for every (direction, arm, seed) once with the declared pair, flags and point, and writes its
block (judged exit 0, INVALID exit 5); P.6.5's single rerun after a technical INVALID; p_oc records block oc from O2's
rows without a pool. Every P hashed file exists."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flymon.brain import p_cli, p_measure
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.odor_real import load_table
from flymon.brain.p_measure import HASHED_FILES
from flymon.brain.p_rules import DIRECTION_DEPENDENT, LEARNS_CONFIRMATORY, OC_RECORDED
from flymon.brain.p_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[1]
SM = smoke(SPEC)
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])      # noqa: E731
KEY = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}
TYPES = sorted({"ORN_" + p for v in load_table(ROOT / N_SPEC.data_dir, N_SPEC.sha_pins()).glomeruli.values()
                for p in v})
C1 = dict(pair="dis", declared=-2.433, exact=-2.4325, o=-2.4325, c1=0.25 * 2.4325, tol=0.0005)
STAGES = ["run_p", "p_oc"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def _probe(seed, A=20.0, P=10):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1,
                wall_s=0.01, steps=1400)


class FakePool:
    def close(self):
        pass


class FakeMeasurer:
    """The punish arm lowers A_X by 2 + effect[direction] + jitter, the plastic arm by 2, the frozen arm not at all."""

    def __init__(self, effect=None, frozen_moves=False):
        self.effect = effect or {"r1": 2.4, "r2": 2.4}
        self.frozen_moves, self.calls = frozen_moves, []

    def p_arms(self, params, items, readout, punish_type):
        self.calls.append((sorted({(i["direction"], i["x"], i["y"]) for i in items}),
                           sorted({(i["arm"], i["punish"], i["plastic"], i["da_zero"]) for i in items}),
                           sorted({tuple(i["point"]) for i in items}), len(items)))
        out = []
        for it in items:
            i = it["seed"] - SM.seeds[0]
            punish = 2.0 + self.effect[it["direction"]] + 0.5 * (i % 2)
            ax = {"plastic": 2.0, "frozen": 0.0, "punish": punish}[it["arm"]]
            moved = it["arm"] != "frozen" or self.frozen_moves
            out.append(dict(seed=it["seed"], edit=it["edit"], arm=it["arm"], punish=it["punish"],
                            plastic=it["plastic"], da_zero=it["da_zero"], csc_sha256="sha-none",
                            pre={"x": _probe(it["seed"]), "y": _probe(it["seed"])},
                            post={"x": _probe(it["seed"], A=20.0 - ax), "y": _probe(it["seed"])},
                            weights_frac=0.9, weights_frac_A=0.8, weights_frac_P=1.0, w0_sha256="w0",
                            w_post_sha256="w1" if moved else "w0", da_integral={"PPL105": 1.0}, wall_s=1.0,
                            direction=it["direction"], x=it["x"], y=it["y"], point=list(it["point"])))
        return out


def _o2_source(spec, smoke):
    """Two O2-like X with 32 seeds: punish lowers A_X by 2 + 2.4 + jitter, plastic by 2."""
    rows = []
    for x, y, _ in O_SPEC.o2_pairs:
        for a in ("plastic", "punish"):
            for i, s in enumerate(O_SPEC.o2_seeds):
                ax = 2.0 if a == "plastic" else 4.4 + 0.3 * (i % 3)
                rows.append(dict(x=x, y=y, arm=a, seed=s, pre={"x": _probe(s), "y": _probe(s)},
                                 post={"x": _probe(s, A=20.0 - ax), "y": _probe(s)}))
    return dict(run_id="run-o2", measure_key="q" * 64, z=ZU, rows=rows, D={}, summary="s", cache="c",
                n_rows=len(rows), seeds=list(O_SPEC.o2_seeds), null=p_cli.OC_NULL), None


def _patch(mod, monkeypatch, measurer=None, commit="c0ffee", c1=(C1, None), committed=None, key=KEY):
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "code_keys", lambda npz: key)
    monkeypatch.setattr(mod, "m0d_sha", lambda spec: "s" * 64)
    monkeypatch.setattr(mod, "load_c3", lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, ZU, None))
    monkeypatch.setattr(mod, "model_types", lambda npz: TYPES)
    monkeypatch.setattr(mod, "spec_commit", lambda path: commit)
    monkeypatch.setattr(mod, "c1_source", lambda spec, smoke: c1)
    monkeypatch.setattr(mod, "o2_source", _o2_source)
    monkeypatch.setattr(mod, "check_committed", committed or (lambda summary, blocks: None))
    monkeypatch.setattr(mod, "make_measurer",
                        lambda ctx: (measurer, FakePool()) if measurer else pytest.fail("no pool may start here"))
    monkeypatch.setattr(p_cli, "out_allowed", lambda out: True)
    monkeypatch.setattr(p_cli, "real_summary", lambda p: Path(p).resolve() == Path.cwd().resolve() / p_measure.SUMMARY)


def _real_summary(tmp_path, doc):
    p = tmp_path / p_measure.SUMMARY
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def test_every_hashed_file_exists():
    assert [f for f in HASHED_FILES if not (ROOT / f).exists()] == []


@pytest.mark.parametrize("name", STAGES)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_out_outside_results_p(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/o/x"]) == 2 and "results/p/" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_dirty_hashed_files(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/p_rules.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


@pytest.mark.parametrize("name, block", [("run_p", "p"), ("p_oc", "oc")])
def test_never_rewrites_a_judged_block(name, block, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}, block: {"outcome": "JUDGED"}})
    assert mod.main([], require_root=False) == 2 and "post-hoc" in capsys.readouterr().err
    assert mod.main(["--rerun-after-invalid"], require_root=False) == 2


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_uncommitted_spec(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, commit=None)
    if name == "run_p":
        _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}})
    assert mod.main([], require_root=False) == 2 and "spec" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_a_disagreeing_block_n1(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, c1=(None, "block n1's oracle effects disagree"))
    assert mod.main(["--smoke"], require_root=False) == 2 and "disagree" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_run_p_refuses_without_a_committed_oc_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    assert mod.main([], require_root=False) == 2 and "block oc" in capsys.readouterr().err
    _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED, "run_id": "oc1"}})
    _patch(mod, monkeypatch, committed=lambda summary, blocks: "has uncommitted changes")
    assert mod.main([], require_root=False) == 2 and "uncommitted" in capsys.readouterr().err


def test_run_p_smoke_asks_for_every_unit_once_and_judges(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    assert fm.calls == [(sorted((d, x, y) for d, (x, y) in SPEC.pairs().items()),
                         sorted(SPEC.o.o2_arms), [(0.25, 8.0)], SM.n_arms())]
    doc = json.loads(Path(p_cli.SMOKE_SUMMARY).read_text())["p"]
    assert doc["outcome"] == "JUDGED" and doc["label"] == LEARNS_CONFIRMATORY and doc["smoke"] is True
    assert doc["c1"] == pytest.approx(C1["c1"]) and doc["c1_source"] == C1 and doc["spec_commit"] == "c0ffee"
    assert doc["point"] == {"g": 0.25, "c_delta": 8.0} and Path(doc["report"]).exists()
    assert "P: LEARNS_CONFIRMATORY" in capsys.readouterr().out


def test_run_p_real_run_needs_oc_first_and_writes_block_p(tmp_path, monkeypatch):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer(effect={"r1": 2.4, "r2": 0.0})
    _patch(mod, monkeypatch, fm)
    summ = _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED, "run_id": "oc1"}})
    assert mod.main([], require_root=False, spec=SM) == 0          # a declared run at the fake measurer's scale
    doc = json.loads(summ.read_text())
    assert doc["p"]["label"] == DIRECTION_DEPENDENT and doc["p"]["oc_run_id"] == "oc1" and doc["oc"]["run_id"] == "oc1"


def test_an_invalid_block_is_rerun_once_with_fixed_code(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(frozen_moves=True))
    summ = _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED, "run_id": "oc1"}})
    assert mod.main([], require_root=False, spec=SM) == 5
    first = json.loads(summ.read_text())["p"]
    assert first["outcome"] == "INVALID" and "plumbing" in first["reasons"][0]
    assert mod.main([], require_root=False, spec=SM) == 2 and "--rerun-after-invalid" in capsys.readouterr().err
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 2       # same code manifest
    assert "fix the code" in capsys.readouterr().err
    fixed = ({"key": "k" * 64, "files": {}}, {"key": "n" * 64, "files": {}})
    _patch(mod, monkeypatch, FakeMeasurer(), key=fixed)
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 0
    doc = json.loads(summ.read_text())
    assert doc["p_invalid"]["run_id"] == first["run_id"] and doc["p"]["rerun_of"] == first["run_id"]
    assert doc["p"]["label"] == LEARNS_CONFIRMATORY
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 2       # final now


def test_a_second_invalid_is_final(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(frozen_moves=True))
    summ = _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}, "p": {"outcome": "INVALID", "run_id": "a",
                                                                            "code": {"key": "old"}},
                                    "p_invalid": {"outcome": "INVALID", "run_id": "b"}})
    assert mod.main(["--rerun-after-invalid"], require_root=False, spec=SM) == 2
    assert "one rerun only" in capsys.readouterr().err and json.loads(summ.read_text())["p"]["run_id"] == "a"


def test_rerun_flag_without_an_invalid_block_is_refused(tmp_path, monkeypatch, capsys):
    mod = _script("run_p")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    _real_summary(tmp_path, {"oc": {"outcome": OC_RECORDED}})
    assert mod.main(["--rerun-after-invalid"], require_root=False) == 2
    assert "needs an INVALID" in capsys.readouterr().err


def test_p_oc_smoke_records_block_oc_without_a_pool(tmp_path, monkeypatch, capsys):
    mod = _script("p_oc")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)                                   # no measurer: a pool start fails the test
    assert mod.main(["--smoke"], require_root=False) == 0
    doc = json.loads(Path(p_cli.SMOKE_SUMMARY).read_text())["oc"]
    assert doc["outcome"] == OC_RECORDED and doc["draws"] == SM.oc_draws and doc["n"] == SM.o.o2_n_seeds
    assert set(doc["scenarios"]) == {"observed", "null_both", "null_r1", "null_r2"}
    assert doc["o2_source"]["run_id"] == "run-o2" and "Y = 1:4" in doc["limitation"]
    assert doc["null"] == p_cli.OC_NULL and doc["o2_source"]["null"] == p_cli.OC_NULL       # Task 6 ruling
    assert "P OC:" in capsys.readouterr().out


def test_p_oc_refuses_when_c3_z_differs_from_o2s(tmp_path, monkeypatch, capsys):
    mod = _script("p_oc")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    monkeypatch.setattr(mod, "o2_source", lambda spec, smoke: (dict(_o2_source(spec, smoke)[0],
                                                                    z={"A": [1.0, 1.0], "P": [0.0, 1.0]}), None))
    assert mod.main(["--smoke"], require_root=False) == 2 and "differs from O2's" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_smoke_and_real_runs_use_separate_roots(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch)
    assert mod.main(["--smoke", "--out", p_cli.RUN_OUT], require_root=False) == 2
    assert mod.main(["--out", p_cli.SMOKE_OUT], require_root=False) == 2
    assert mod.main(["--smoke", "--summary", p_measure.SUMMARY], require_root=False) == 2
    assert mod.main(["--summary", p_cli.SMOKE_SUMMARY], require_root=False) == 2
    assert capsys.readouterr().err.count("smoke") >= 4 and not (tmp_path / "results").exists()
