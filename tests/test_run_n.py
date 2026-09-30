# tests/test_run_n.py
"""Spec N.8.9 / plan readings 12-14: the N CLIs refuse before any pool on another root, --out, a dirty tree, an
uncommitted or other-code summary, a stopped upstream outcome, a missing N.8a / N.8b paragraph (or one citing another
run), a later block; N0f writes STOP_DATA_MISMATCH / STOP_NO_OPERATING_POINT blocks (exit 5) and never asks for KC
vectors; each stage's flow with a stand-in measurer. Blinding (N.8): N0f never
asks for a KC vector, N0 only with edit "none". The shared helpers live once in n_cli and read their numbers from
n_spec."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import n_cli
from flymon.brain.config import Params
from flymon.brain.n_rules import (N1_GO, N2_0_GO, OPERATING_POINT, SIMILARITY_GO, STOP_DATA_MISMATCH,
                                  STOP_NO_OPERATING_POINT, STOP_UNTESTABLE, SUPPORTED, INVALID)
from flymon.brain.n_spec import SPEC, smoke
from flymon.brain.odor_real import load_table

ROOT = Path(__file__).resolve().parents[1]
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])
KEY = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}                      # dV = (A_x - P_x) - (A_y - P_y)
TYPES = sorted({"ORN_" + p for v in load_table(ROOT / SPEC.data_dir, SPEC.sha_pins()).glomeruli.values() for p in v})
STAGES = ["run_n0f", "run_n0"]


def _script(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


def _prow(A=10.0, P=20.0, kc=0.05, hz=50.0):
    return dict(seed=0, A=A, P=P, kc_frac=kc, kc_spikes=5, kc_max_win_hz=hz, apl_out_per_step=0.1, wall_s=0.014,
                steps=1400)


LEARN = {"sim_on": 1.0, "sim_off": 0.2, "dis_on": 1.0, "dis_off": 1.0, "sim_all": 0.1, "dis_all": 0.1}


class FakePool:
    closed = False

    def close(self):
        FakePool.closed = True


class FakeMeasurer:
    """Stand-in NMeasurer: KC fractions and readouts from constructor values; learning -k per condition."""

    def __init__(self, kc=0.05, learn=LEARN, leak=False, o=(-3.0, -4.0, -5.0)):
        self.kc, self.learn, self.leak, self.o, self.calls = kc, learn, leak, o, []

    def presentations(self, params, items, seeds, readout):
        self.calls.append(("presentations", [e for e, _ in items]))
        return [[dict(_prow(kc=self.kc), seed=s, edit=e, csc_sha256=e) for s in seeds] for e, _ in items]

    def kc_vectors(self, params, items, seeds, readout):
        self.calls.append(("kc_vectors", [e for e, _ in items]))
        out = []
        for i, (e, _) in enumerate(items):
            base = list(range(60, 80)) if i % 3 == 2 else list(range(i % 3 * 2, 20 + i % 3 * 2))
            out.append([dict(_prow(), seed=s, edit=e, csc_sha256=e, n_kc=100, kc_fired=base + [90 + j % 10])
                        for j, s in enumerate(seeds)])
        return out

    def punish_oracle(self, params, pairs, readout, z, punish_type):
        self.calls.append(("punish_oracle", [p["name"] for p in pairs]))
        n = len(self.o)
        pre = {"A": [[10.0, 10.0]] * n, "P": [[5.0, 5.0]] * n}
        post = {"A": [[10.0 + x, 10.0] for x in self.o], "P": [[5.0, 5.0]] * n}
        return [{"alpha_punish": 0.5, "select": {"pre": pre, "punish": {"0.5": {"P": post, "change": -4.0}}},
                 "report": {"pre": pre, "P": post}, "kc": {"x": {}, "y": {}, "jaccard": 0.1}, "csc_sha256": "none"}
                for _ in pairs]

    def arms(self, params, items, readout, punish_type):
        self.calls.append(("arms", sorted({i["cond"] for i in items})))
        rows = []
        for i in items:
            k = (self.learn[i["cond"]] + 0.05 * (i["seed"] % 3 - 1)) if i["plastic"] else 0.0
            post_x = _prow(A=10.0 - k)
            if self.leak and not i["plastic"]:
                post_x["A"] = 9.0
            rows.append(dict(seed=i["seed"], edit=i["edit"], plastic=i["plastic"], csc_sha256=i["edit"],
                             pre={"x": _prow(), "y": _prow()}, post={"x": post_x, "y": _prow()},
                             weights_frac=0.9 if i["plastic"] else 1.0, wall_s=0.1, cond=i["cond"]))
        return rows


def _patch(mod, monkeypatch, measurer=None, head=None, z=ZU):
    monkeypatch.setattr(mod, "git_state", CLEAN)
    monkeypatch.setattr(mod, "check_committed", lambda *a: None)
    monkeypatch.setattr(mod, "code_keys", lambda npz: KEY)
    monkeypatch.setattr(mod, "m0d_sha", lambda spec: "s" * 64)
    monkeypatch.setattr(mod, "load_c3", lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, z, None))
    monkeypatch.setattr(mod, "model_types", lambda npz: TYPES)
    monkeypatch.setattr(mod, "head_spec", lambda path: head)
    monkeypatch.setattr(mod, "make_measurer",
                        lambda ctx: (measurer, FakePool()) if measurer else pytest.fail("no pool may start here"))
    monkeypatch.setattr(n_cli, "out_allowed", lambda out: True)


def _block(name, **kw):
    b = dict(outcome=None, run_id=f"rid-{name}", measure_key="k" * 64, code={"key": "m" * 64},
             inputs={"m0d_sha256": "s" * 64}, upstream={})
    b.update(kw)
    return b


N0F = _block("n0f", outcome=OPERATING_POINT, selected={"g": 1.0, "c_delta": 1.0}, wall_s_per_step=1e-5,
             state_shares={}, validity={}, state_flags=[])


def _write(tmp_path, doc, smoke_=True):
    p = tmp_path / ("results/n/smoke/n_real_odour.json" if smoke_ else "results/summary/n_real_odour.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


@pytest.mark.parametrize("name", STAGES)
def test_refuses_outside_the_root(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", STAGES)
def test_refuses_an_out_outside_results_n(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/m0d/n/x"]) == 2 and "results/n/" in capsys.readouterr().err


@pytest.mark.parametrize("name", STAGES)
def test_refuses_dirty_hashed_files(name, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.setattr(mod, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/n_jobs.py"],
                                                            dirty_other=[]))
    assert mod.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_spec_note_needs_a_line_start_marker_citing_the_run():
    assert "no N.8a" in n_cli.spec_note("x **N.8a** mentioned mid-line rid-1\n", "N.8a", "rid-1")
    assert "does not cite" in n_cli.spec_note("**N.8a N0f 결과 (run `rid-0`)**\n", "N.8a", "rid-1")
    assert n_cli.spec_note("text\n**N.8a N0f 결과 (2026-10-01, run `rid-1`)** — outcome\n", "N.8a", "rid-1") is None
    assert n_cli.spec_note("### N.8b N2.0\nrun rid-2\n", "N.8b", "rid-2") is None
    assert "not tracked" in n_cli.spec_note(None, "N.8a", "rid-1")


def test_order_later_blocks_and_read_previous(tmp_path):
    assert n_cli.ORDER == ("n0f", "n0", "n1", "n2_0", "n2")
    assert "later blocks ['n1']" in n_cli.later_blocks("n0", {"n0f": {}, "n1": {}})
    assert n_cli.later_blocks("n2", {"n0f": {}}) is None
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"n0f": {}}))
    assert n_cli.read_previous(s, ["n0f"], False, lambda *a: "uncommitted")[1] == "uncommitted"
    assert "lacks the blocks ['n0']" in n_cli.read_previous(s, ["n0f", "n0"], False, lambda *a: None)[1]
    assert n_cli.read_previous(tmp_path / "none.json", [], False, lambda *a: None) == ({}, None)


def test_n0f_smoke_selects_the_point_and_never_asks_for_kc_vectors(tmp_path, monkeypatch, capsys):
    mod = _script("run_n0f")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    assert mod.main(["--smoke"], require_root=False) == 0
    doc = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert doc["outcome"] == OPERATING_POINT and doc["selected"] == {"g": 1.0, "c_delta": 1.0}
    assert [c[0] for c in fm.calls] == ["presentations"]
    assert sorted(set(fm.calls[0][1])) == ["apl_all_zero", "apl_to_kc_zero", "none"]
    assert set(doc["state_shares"]["4:1"]) == {"on", "block", "all"} and doc["validity"]["ok"]
    assert doc["lin_totals"]["totals"]["signed"]["IA"] == 2040.0 and doc["smoke"] is True
    assert Path(doc["report"]).exists() and "N0f: 작동점" in capsys.readouterr().out


def test_n0f_writes_a_no_operating_point_block_and_exits_5(tmp_path, monkeypatch):
    mod = _script("run_n0f")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(kc=0.5))
    assert mod.main(["--smoke"], require_root=False) == 5
    doc = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert doc["outcome"] == STOP_NO_OPERATING_POINT and doc["selected"] is None and "state_shares" not in doc


def test_n0f_data_mismatch_is_a_block_and_starts_no_pool(tmp_path, monkeypatch):
    from dataclasses import replace
    mod = _script("run_n0f")
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "odor"
    shutil.copytree(ROOT / SPEC.data_dir, d)
    (d / "hallem2006_subset.csv").write_text((d / "hallem2006_subset.csv").read_text().replace("Or22a,236", "Or22a,NA"))
    _patch(mod, monkeypatch, None)                                     # make_measurer fails the test if called
    assert mod.main(["--smoke"], spec=replace(smoke(SPEC), data_dir=str(d)), require_root=False) == 5
    doc = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert doc["outcome"] == STOP_DATA_MISMATCH and "sha256" in doc["reason"]


def test_n0_refuses_a_stopped_or_unnoted_n0f(tmp_path, monkeypatch, capsys):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None, head="no notes here\n")
    stopped = dict(N0F, outcome=STOP_NO_OPERATING_POINT)
    _write(tmp_path, {"n0f": stopped}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "STOP" in capsys.readouterr().err
    _write(tmp_path, {"n0f": N0F}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "N.8a" in capsys.readouterr().err
    monkeypatch.setattr(mod, "head_spec", lambda p: "**N.8a N0f 결과 (run `rid-other`)**\n")
    assert mod.main([], require_root=False) == 2 and "does not cite" in capsys.readouterr().err


def test_n0_refuses_other_code_another_m0d_and_a_later_block(tmp_path, monkeypatch, capsys):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None, head="**N.8a (run `rid-n0f`)**\n")
    _write(tmp_path, {"n0f": dict(N0F, measure_key="x" * 64)}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "other code" in capsys.readouterr().err
    _write(tmp_path, {"n0f": dict(N0F, inputs={"m0d_sha256": "t" * 64})}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "m0d.json" in capsys.readouterr().err
    _write(tmp_path, {"n0f": N0F, "n1": _block("n1")}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "later blocks" in capsys.readouterr().err


def test_n0_passes_the_similarity_gate(tmp_path, monkeypatch):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head="**N.8a N0f 결과 (run `rid-n0f`)**\n")
    _write(tmp_path, {"n0f": N0F}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    doc = json.loads(Path("results/summary/n_real_odour.json").read_text())
    b = doc["n0"]
    assert b["outcome"] == SIMILARITY_GO and b["point"] == {"g": 1.0, "c_delta": 1.0}
    assert b["upstream"] == {"n0f": "rid-n0f"} and fm.calls == [("kc_vectors", ["none"] * 3)]
    assert set(b["drives"]) == {"4:1", "1:4", "dDL"} and doc["n0f"] == N0F


# ---------------------------------------------------------------- blinding, no GO on missing input, one copy (rulings)
def test_n0f_source_never_names_the_kc_vector_measurement():
    src = (ROOT / "scripts" / "run_n0f.py").read_text()
    assert "kc_vectors" not in src and "n_kcv" not in src and "kc_fired" not in src


def test_n0_asks_for_kc_vectors_only_with_the_unedited_engine(tmp_path, monkeypatch):
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm)
    _write(tmp_path, {"n0f": N0F})
    assert mod.main(["--smoke"], require_root=False) == 0
    assert [k for k, _ in fm.calls] == ["kc_vectors"] and {e for _, es in fm.calls for e in es} == {"none"}
    src = (ROOT / "scripts" / "run_n0.py").read_text()
    assert "apl_to_kc_zero" not in src and "apl_all_zero" not in src and "spec.conditions" not in src


class SilentMeasurer(FakeMeasurer):
    def kc_vectors(self, params, items, seeds, readout):
        return [[dict(r, kc_fired=[]) for r in rr] for rr in super().kc_vectors(params, items, seeds, readout)]


def test_n0_silent_vectors_stop_and_exit_5(tmp_path, monkeypatch):
    from flymon.brain.n_rules import STOP_SIMILARITY_ORDER
    mod = _script("run_n0")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, SilentMeasurer())
    _write(tmp_path, {"n0f": N0F})
    assert mod.main(["--smoke"], require_root=False) == 5
    assert json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0"]["outcome"] == STOP_SIMILARITY_ORDER


class ShortMeasurer(FakeMeasurer):
    """Returns only the first `keep` seeds of every item (0: nothing)."""
    keep = 0

    def presentations(self, params, items, seeds, readout):
        return [rr[:self.keep] for rr in super().presentations(params, items, seeds, readout)]

    def kc_vectors(self, params, items, seeds, readout):
        return [rr[:self.keep] for rr in super().kc_vectors(params, items, seeds, readout)]


@pytest.mark.parametrize("keep", [0, 1])
@pytest.mark.parametrize("name,block", [("run_n0f", "n0f"), ("run_n0", "n0")])
def test_missing_measurements_raise_and_write_no_block(name, block, keep, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    FakePool.closed = False
    fm = ShortMeasurer()
    fm.keep = keep
    _patch(mod, monkeypatch, fm)
    if block == "n0":
        _write(tmp_path, {"n0f": N0F})
    with pytest.raises(ValueError, match="missing"):
        mod.main(["--smoke"], require_root=False)
    p = Path(n_cli.SMOKE_SUMMARY)
    assert FakePool.closed and (not p.exists() or block not in json.loads(p.read_text()))
    assert "GO" not in capsys.readouterr().out


def test_n0_without_an_operating_point_raises_outside_smoke(tmp_path):
    with pytest.raises(ValueError):
        n_cli.operating_point({"n0f": {"selected": None}}, SPEC, False)
    assert n_cli.operating_point({"n0f": {"selected": None}}, SPEC, True) == tuple(SPEC.smoke_point)


def test_scripts_hold_no_copy_of_a_shared_helper_and_hooks_resolve_on_the_script_module(monkeypatch):
    for name in STAGES:
        src = (ROOT / "scripts" / f"{name}.py").read_text()
        assert "def hooks" not in src and "drive_hz" not in src and "FlyPool" not in src
        assert '__name__ == "__main__"' in src
        mod = _script(name)
        assert set(n_cli.hooks(mod)) == set(n_cli.HOOK_NAMES)
        monkeypatch.setattr(mod, "m0d_sha", "patched")
        assert n_cli.hooks(mod)["m0d_sha"] == "patched"


def test_cli_defaults_and_pool_settings_come_from_n_spec(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from flymon.brain import fly_pool
    assert n_cli.parser(None).parse_args([]).workers == SPEC.workers
    seen = {}

    class Pool:
        def __init__(self, npz, params, flies, **kw):
            seen.update(kw, flies=len(flies))

    monkeypatch.setattr(fly_pool, "FlyPool", Pool)
    spec = replace(SPEC, pool_timeout_s=123.0)
    ctx = SimpleNamespace(args=SimpleNamespace(npz="x.npz", workers=3), spec=spec, c3=Params(), out=tmp_path,
                          key={"key": "k"}, rid="r")
    m, pool = n_cli.make_measurer(ctx)
    assert seen == dict(flies=3, workers=3, punish_type=spec.h3.punish_type, reward_type=spec.h3.reward_type,
                        timeout_s=123.0)
    assert m.pool is pool and m.cache.root == tmp_path / "cache"
    src = (ROOT / "flymon" / "brain" / "n_cli.py").read_text()
    assert "7200" not in src and "3600" not in src and "default=16" not in src


def test_code_keys_cover_the_measure_files_and_the_manifest_and_refuse_a_missing_file(tmp_path, monkeypatch, capsys):
    from flymon.brain.n_measure import HASHED_FILES, MEASURE_FILES
    assert "flymon/brain/n_spec.py" in HASHED_FILES and "flymon/brain/n_spec.py" not in MEASURE_FILES
    for f in HASHED_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    monkeypatch.setattr(n_cli, "ROOT", tmp_path)
    monkeypatch.setattr(n_cli, "code_key", lambda npz, files: tuple(files))
    assert n_cli.code_keys("x.npz") == (MEASURE_FILES, HASHED_FILES)
    (tmp_path / "scripts" / "run_n0.py").unlink()
    with pytest.raises(SystemExit) as e:
        n_cli.code_keys("x.npz")
    assert e.value.code == 2 and "run_n0.py" in capsys.readouterr().err
