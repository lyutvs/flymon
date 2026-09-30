# tests/test_run_n.py
"""Spec N.8.9 / plan readings 12-14: the N CLIs refuse before any pool on another root, --out, a dirty tree, an
uncommitted or other-code summary, a stopped upstream outcome, a missing N.8a / N.8b paragraph (or one citing another
run), a later block; N0f writes STOP_DATA_MISMATCH / STOP_NO_OPERATING_POINT blocks (exit 5) and never asks for KC
vectors; each stage's flow with a stand-in measurer. Blinding (N.8): N0f never
asks for a KC vector, N0 only with edit "none". The shared helpers live once in n_cli and read their numbers from
n_spec.
N1 / N2.0 / N2 (the later stages): the N.8a / N.8b gates; N1 and the pilot run the unedited engine only and never ask
for a KC vector; the pilot reads block n0f only through design_view and refuses a partial OC; the judge asks for the
block conditions' KC vectors only after its verdict block is on disk; short or empty measurements raise and write no
block."""
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


# ================================================================ N1, N2.0, N2 (Task 10)
LATER = ["run_n1", "run_n2_pilot", "run_n2_judge"]
N0 = _block("n0", outcome=SIMILARITY_GO, upstream={"n0f": "rid-n0f"})
N1 = _block("n1", outcome=N1_GO, pairs={"sim": {"o": -2.0}, "dis": {"o": -2.0}},
            upstream={"n0f": "rid-n0f", "n0": "rid-n0"})
N2_0 = _block("n2_0", outcome=N2_0_GO, n=3, c1={"sim": 0.25, "dis": 0.25}, delta_min=0.5, eps=0.25,
              upstream={"n0f": "rid-n0f", "n0": "rid-n0", "n1": "rid-n1"})
HEAD_A = "**N.8a N0f 결과 (run `rid-n0f`)**\n"
HEAD_AB = HEAD_A + "**N.8b N2.0 보정 결과 (run `rid-n2_0`)**\n"
FULL = "results/summary/n_real_odour.json"
BLOCK_EDITS = ["apl_to_kc_zero", "apl_all_zero"]


def _report3():
    """SPEC on three report seeds: FakeMeasurer's oracle answers on len(o) = 3 seeds, and N1 checks the count."""
    from dataclasses import replace
    return replace(SPEC, report_seeds=SPEC.report_seeds[:3])


@pytest.mark.parametrize("name", LATER)
def test_later_stages_refuse_outside_the_root_and_outside_results_n(name, tmp_path, monkeypatch, capsys):
    mod = _script(name)
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 2 and "repository root" in capsys.readouterr().err
    monkeypatch.chdir(ROOT)
    assert mod.main(["--out", "results/m0d/n/x"]) == 2 and "results/n/" in capsys.readouterr().err


def test_n1_needs_n8a_and_reads_both_pairs(tmp_path, monkeypatch, capsys):
    mod = _script("run_n1")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head="nothing\n")
    _write(tmp_path, {"n0f": N0F, "n0": N0}, smoke_=False)
    assert mod.main([], spec=_report3(), require_root=False) == 2 and "N.8a" in capsys.readouterr().err
    monkeypatch.setattr(mod, "head_spec", lambda p: HEAD_A)
    assert mod.main([], spec=_report3(), require_root=False) == 0
    b = json.loads(Path(FULL).read_text())["n1"]
    assert b["outcome"] == N1_GO and b["pairs"]["sim"]["o"] == pytest.approx(-4.0)
    assert fm.calls == [("punish_oracle", ["sim", "dis"])]


def test_n1_stop_untestable_exits_5(tmp_path, monkeypatch):
    mod = _script("run_n1")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(o=(-1.0, -2.0, 0.5)), head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0}, smoke_=False)
    assert mod.main([], spec=_report3(), require_root=False) == 5
    assert json.loads(Path(FULL).read_text())["n1"]["outcome"] == STOP_UNTESTABLE


def test_pilot_refuses_a_stopped_n1_and_runs_on_arms_only(tmp_path, monkeypatch, capsys):
    mod = _script("run_n2_pilot")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": dict(N1, outcome=STOP_UNTESTABLE)}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "STOP" in capsys.readouterr().err
    _write(tmp_path, {"n0f": dict(N0F, grid={"never": "read"}), "n0": N0, "n1": N1}, smoke_=True)
    assert mod.main(["--smoke"], require_root=False) == 0
    b = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n2_0"]
    assert fm.calls == [("arms", ["dis_on", "sim_on"])]                       # APL on only (decision ⑥)
    assert b["outcome"] == N2_0_GO and b["n"] == 3 and b["c1"] == {"sim": 0.5, "dis": 0.5}
    assert set(b["design_view"]) == {"outcome", "run_id", "selected", "wall_s_per_step", "state_shares", "validity",
                                     "state_flags"}
    assert len(b["pilot"]["sim"]) == len(smoke(SPEC).pilot_seeds)
    # the block keeps all four per-scenario results and the OC's own bootstrap draws (beside the judgement's)
    sp = smoke(SPEC)
    four = {f"{p}|{m:g}x" for p in SPEC.oc_pairings for m in SPEC.sd_mults}
    assert len(four) == 4 and set(b["scenario_n"]) == four == set(b["oc"]["scenario_n"])
    assert b["oc"]["partial"] is False and b["oc"]["pairings"] == list(SPEC.oc_pairings)
    assert b["oc_boot"] == b["oc"]["boot"] == sp.oc_boot and b["judge_boot"] == b["oc"]["judge_boot"] == sp.boot_draws
    assert len(b["oc"]["rows"]) == 4 * 2 * len(sp.n_grid) and "boot_note" in b["oc"]
    assert b["pilot_states"] == {"sim_on": 1.0, "dis_on": 1.0}                 # an aggregate, no per-presentation row


def test_judge_needs_n8b_citing_the_n2_0_run(tmp_path, monkeypatch, capsys):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None, head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 2 and "N.8b" in capsys.readouterr().err
    monkeypatch.setattr(mod, "head_spec", lambda p: HEAD_A + "**N.8b (run `rid-old`)**\n")
    assert mod.main([], require_root=False) == 2 and "does not cite" in capsys.readouterr().err


def test_judge_verdict_then_block_kc_record(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    b = json.loads(Path(FULL).read_text())["n2"]
    assert b["outcome"] == SUPPORTED and b["thresholds"]["n"] == 3 and b["invalid"] == []
    assert fm.calls[0] == ("arms", ["dis_all", "dis_off", "dis_on", "sim_all", "sim_off", "sim_on"])
    assert fm.calls[1] == ("kc_vectors", ["apl_to_kc_zero"] * 3 + ["apl_all_zero"] * 3)    # only after the verdict
    assert len(fm.calls) == 2
    assert set(b["post_verdict_kc"]) == {"apl_to_kc_zero", "apl_all_zero"}
    assert b["all_output_block"]["D_sim"] == pytest.approx(0.9)
    assert b["per_seed"]["sim_on"][0]["seed"] == SPEC.judge_seed0
    assert set(b["state_shares"]) == set(LEARN) and "pre" not in json.dumps(b["per_seed"])   # aggregates only


def test_judge_plumbing_leak_is_invalid(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer(leak=True)
    _patch(mod, monkeypatch, fm, head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    b = json.loads(Path(FULL).read_text())["n2"]
    assert b["outcome"] == INVALID and any("plumbing" in r for r in b["invalid"])
    # an INVALID run's judgement is not over: no block-condition KC vector is asked for
    assert [k for k, _ in fm.calls] == ["arms"] and b["post_verdict_kc"] is None and "INVALID" in b["post_verdict_kc_note"]


# ---------------------------------------------------------------- rulings: blinding, ordering, no outcome on missing input
class Sealed(dict):
    """A summary block whose fields outside `open_` may not be read: any such read (or a walk over it) fails."""

    def __init__(self, data, open_):
        super().__init__(data)
        self.open_, self.read = set(open_), []

    def _ask(self, k):
        self.read.append(k)
        if k not in self.open_:
            raise AssertionError(f"blinded field {k!r} was read")

    def __getitem__(self, k):
        self._ask(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._ask(k)
        return super().get(k, default)

    def _walk(self, *a, **kw):
        raise AssertionError("a sealed block was walked")

    __iter__ = keys = items = values = __contains__ = copy = _walk


class EditMeasurer(FakeMeasurer):
    """Also records the edit of every arm item."""

    def arms(self, params, items, readout, punish_type):
        self.edits = getattr(self, "edits", []) + [i["edit"] for i in items]
        return super().arms(params, items, readout, punish_type)


def _ctx(spec, doc, fm, tmp_path):
    from types import SimpleNamespace
    return n_cli.Ctx(args=SimpleNamespace(npz="x.npz", workers=1), spec=spec, smoke=True, doc=doc, rid="rid", out=tmp_path,
                     key=KEY[0], c3=Params(), readout={"A": "MBON13", "P": "MBON05"}, z=ZU, types=TYPES,
                     hooks=dict(make_measurer=lambda ctx: (fm, FakePool())))


def test_pilot_never_reads_a_blinded_field(tmp_path, monkeypatch):
    from flymon.brain.n_rules import DESIGN_KEYS
    mod = _script("run_n2_pilot")
    monkeypatch.chdir(tmp_path)
    secret = "SENTINEL-BLOCK-KC"
    hidden = dict(grid={"1|1": {"4:1": {"block": {"kc_frac": secret}}}}, checks={"1|1": secret}, apl_shift=secret,
                  drives=secret, lin_totals=secret, budget_estimate_h=secret, block_kc_corr=secret)
    n0f = Sealed(dict(N0F, **hidden), DESIGN_KEYS)
    n0 = Sealed(dict(N0, similarity={"r_sim": secret, "delta_r": secret}, kc=secret), ())
    pairs = {p: Sealed({"o": -2.0, "p0": secret, "jaccard": secret, "changes": secret}, ("o",)) for p in ("sim", "dis")}
    n1 = Sealed(dict(N1, pairs=Sealed(pairs, ("sim", "dis"))), ("pairs",))
    fm = EditMeasurer()
    res, code = mod.body(_ctx(smoke(SPEC), {"n0f": n0f, "n0": n0, "n1": n1}, fm, tmp_path))
    assert code == 0 and res["outcome"] == N2_0_GO
    assert set(n0f.read) == set(DESIGN_KEYS) and n0.read == [] and set(n1.read) == {"pairs"}
    assert all(set(v.read) == {"o"} for v in pairs.values())
    assert secret not in json.dumps(res, default=str) and set(res["design_view"]) == set(DESIGN_KEYS)
    assert set(fm.edits) == {"none"} and [k for k, _ in fm.calls] == ["arms"]
    with pytest.raises(AssertionError, match="blinded field 'grid'"):          # the seal itself works
        n0f["grid"]


def test_n1_and_the_pilot_never_ask_for_a_kc_vector_or_a_block_arm(tmp_path, monkeypatch):
    for name in ("run_n1", "run_n2_pilot"):
        src = (ROOT / "scripts" / f"{name}.py").read_text()
        assert "kc_vectors" not in src and "kc_fired" not in src and "kc_blocks" not in src
        assert not any(e in src for e in BLOCK_EDITS) and "n2_record_conditions" not in src
        assert "spec.conditions" not in src
    pilot = (ROOT / "scripts" / "run_n2_pilot.py").read_text()
    assert pilot.count('ctx.doc["n0f"]') == 1 and 'design_view(ctx.doc["n0f"])' in pilot
    assert "ctx.doc, " not in pilot and 'ctx.doc["n0"]' not in pilot and "ctx.doc.get" not in pilot
    mod = _script("run_n1")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0}, smoke_=False)
    assert mod.main([], spec=_report3(), require_root=False) == 0
    assert [k for k, _ in fm.calls] == ["punish_oracle"]


def test_pilot_refuses_a_partial_oc_before_any_outcome(tmp_path, monkeypatch, capsys):
    from flymon.brain import n_oc
    mod = _script("run_n2_pilot")
    monkeypatch.chdir(tmp_path)
    FakePool.closed = False
    _patch(mod, monkeypatch, FakeMeasurer())
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1})
    seen = {}

    def one_pairing(pilot, c1, delta_min, eps, spec, pairings=None):
        seen["asked"] = pairings
        oc = n_oc.simulate(pilot, c1, delta_min, eps, spec, SPEC.oc_pairings[:1])
        assert oc["partial"] is True and oc["n"] is None
        return oc

    monkeypatch.setattr(mod, "simulate", one_pairing)
    monkeypatch.setattr(mod, "n2_0_outcome", lambda *a: pytest.fail("no outcome may be read off a partial OC"))
    with pytest.raises(ValueError, match="partial"):
        mod.main(["--smoke"], require_root=False)
    assert tuple(seen["asked"]) == SPEC.oc_pairings                           # the script asks for every pairing
    assert "n2_0" not in json.loads(Path(n_cli.SMOKE_SUMMARY).read_text()) and FakePool.closed
    assert "STOP_POWER" not in capsys.readouterr().out


def test_pilot_uncalibratable_is_stop_power_without_an_oc(tmp_path, monkeypatch):
    from flymon.brain.n_rules import STOP_POWER
    mod = _script("run_n2_pilot")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, FakeMeasurer(learn=dict(LEARN, sim_on=-1.0)))
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1})
    monkeypatch.setattr(mod, "simulate", lambda *a, **k: pytest.fail("no OC on an uncalibratable pilot"))
    assert mod.main(["--smoke"], require_root=False) == 5
    b = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n2_0"]
    assert b["outcome"] == STOP_POWER and b["n"] is None and b["oc"] is None and b["scenario_n"] is None


class OrderMeasurer(FakeMeasurer):
    """Reads the summary at the moment block-condition KC vectors are asked for."""

    def __init__(self, summary, **kw):
        super().__init__(**kw)
        self.summary, self.at_kc = summary, None

    def kc_vectors(self, params, items, seeds, readout):
        self.at_kc = json.loads(Path(self.summary).read_text()).get("n2")
        return super().kc_vectors(params, items, seeds, readout)


def test_judge_asks_for_block_kc_vectors_only_after_the_verdict_is_written(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    fm = OrderMeasurer(FULL)
    _patch(mod, monkeypatch, fm, head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    assert mod.main([], require_root=False) == 0
    then, now = fm.at_kc, json.loads(Path(FULL).read_text())["n2"]
    assert then is not None and then["outcome"] == SUPPORTED and then["verdict"]["verdict"] == SUPPORTED
    assert then["post_verdict_kc"] is None and Path(then["report"]).exists()
    assert now["verdict"] == then["verdict"] and now["stats"] == then["stats"] and now["run_id"] == then["run_id"]
    assert set(now["post_verdict_kc"]) == set(BLOCK_EDITS)
    assert set(now["post_verdict_kc"]["apl_to_kc_zero"]) == {"r_sim", "r_dis", "delta_r", "ci95"}
    assert [k for k, _ in fm.calls] == ["arms", "kc_vectors"] and "none" not in fm.calls[1][1]


def test_main_stage_after_runs_once_the_block_is_on_disk_and_a_failure_keeps_it(tmp_path, monkeypatch):
    mod = _script("run_n0f")                                # any hook holder: the stage here is a stand-in body
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, None)
    seen = []

    def after(ctx, res):
        seen.append(json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]["outcome"])
        return dict(extra=1)

    body = lambda ctx: (dict(outcome="X", sentence="s"), 0)
    assert n_cli.main_stage("n0f", ["--smoke"], body, n_cli.hooks(mod), require_root=False, after=after) == 0
    assert seen == ["X"] and json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]["extra"] == 1

    def broken(ctx, res):
        raise ValueError("record failed")

    body = lambda ctx: (dict(outcome="Y", sentence="s"), 0)
    with pytest.raises(ValueError, match="record failed"):
        n_cli.main_stage("n0f", ["--smoke"], body, n_cli.hooks(mod), require_root=False, after=broken)
    b = json.loads(Path(n_cli.SMOKE_SUMMARY).read_text())["n0f"]
    assert b["outcome"] == "Y" and "extra" not in b


class DropMeasurer(FakeMeasurer):
    """Drops rows: `drop(row)` true -> the row never comes back; short_kc -> only the first seed of every KC block."""

    def __init__(self, drop=lambda r: False, pairs=None, report=None, short_kc=False, **kw):
        super().__init__(**kw)
        self.drop, self.pairs, self.report, self.short_kc = drop, pairs, report, short_kc

    def arms(self, params, items, readout, punish_type):
        return [r for r in super().arms(params, items, readout, punish_type) if not self.drop(r)]

    def punish_oracle(self, params, pairs, readout, z, punish_type):
        rows = super().punish_oracle(params, pairs, readout, z, punish_type)[:self.pairs]
        if self.report is not None:
            for r in rows:
                r["report"] = {k: {f: v[:self.report] for f, v in b.items()} for k, b in r["report"].items()}
        return rows

    def kc_vectors(self, params, items, seeds, readout):
        rows = super().kc_vectors(params, items, seeds, readout)
        return [rr[:1] for rr in rows] if self.short_kc else rows


@pytest.mark.parametrize("kw", [dict(pairs=0), dict(pairs=1), dict(report=2), dict(report=0)])
def test_n1_missing_oracle_rows_raise_and_write_no_block(kw, tmp_path, monkeypatch, capsys):
    mod = _script("run_n1")
    monkeypatch.chdir(tmp_path)
    FakePool.closed = False
    _patch(mod, monkeypatch, DropMeasurer(**kw), head=HEAD_A)
    _write(tmp_path, {"n0f": N0F, "n0": N0}, smoke_=False)
    with pytest.raises(ValueError, match="missing"):
        mod.main([], spec=_report3(), require_root=False)
    assert "n1" not in json.loads(Path(FULL).read_text()) and FakePool.closed
    assert "GO" not in capsys.readouterr().out


LAST_PILOT = smoke(SPEC).pilot_seeds[-1]


@pytest.mark.parametrize("drop", [lambda r: True, lambda r: r["seed"] == LAST_PILOT,
                                  lambda r: r["cond"] == "dis_on" and r["seed"] == LAST_PILOT,
                                  lambda r: r["cond"] == "sim_on"])
def test_pilot_missing_arm_rows_raise_and_write_no_block(drop, tmp_path, monkeypatch, capsys):
    mod = _script("run_n2_pilot")
    monkeypatch.chdir(tmp_path)
    FakePool.closed = False
    _patch(mod, monkeypatch, DropMeasurer(drop=drop))
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1})
    monkeypatch.setattr(mod, "simulate", lambda *a, **k: pytest.fail("no OC on a short pilot"))
    with pytest.raises(ValueError, match="missing"):
        mod.main(["--smoke"], require_root=False)
    assert "n2_0" not in json.loads(Path(n_cli.SMOKE_SUMMARY).read_text()) and FakePool.closed
    out = capsys.readouterr().out
    assert "GO" not in out and "STOP" not in out


@pytest.mark.parametrize("drop", [lambda r: True, lambda r: not r["plastic"],
                                  lambda r: not r["plastic"] and r["cond"] == "dis_all",
                                  lambda r: not r["plastic"] and r["seed"] == SPEC.judge_seed0 + 1,
                                  lambda r: r["plastic"] and r["seed"] == SPEC.judge_seed0 + 2,
                                  lambda r: r["plastic"] and r["cond"] == "sim_all"])
def test_judge_missing_arms_or_reruns_raise_and_write_no_verdict(drop, tmp_path, monkeypatch, capsys):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    FakePool.closed = False
    fm = DropMeasurer(drop=drop)
    _patch(mod, monkeypatch, fm, head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    monkeypatch.setattr(mod, "plumbing", lambda rows: pytest.fail("the rerun count is checked before the plumbing rule"))
    with pytest.raises(ValueError, match="missing"):
        mod.main([], require_root=False)
    assert "n2" not in json.loads(Path(FULL).read_text()) and FakePool.closed
    assert [k for k, _ in fm.calls] == ["arms"]                               # and no KC vector was asked for
    out = capsys.readouterr().out
    assert not any(v in out for v in ("SUPPORTED", "NOT_REPLICATED", "NO_LEARNING", "INVALID"))


def test_judge_checks_the_rerun_count_against_n_spec(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    got = {}
    _patch(mod, monkeypatch, FakeMeasurer(), head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    real = mod.plumbing
    monkeypatch.setattr(mod, "plumbing", lambda rows: got.update(n=len(rows), plastic={r["plastic"] for r in rows})
                        or real(rows))
    assert mod.main([], require_root=False) == 0
    conds = len(SPEC.n2_conditions) + len(SPEC.n2_record_conditions)
    assert got == dict(n=conds * SPEC.plumbing_seeds, plastic={False})


@pytest.mark.parametrize("bad", [dict(n=None), dict(c1=None), dict(delta_min=0.0), dict(eps=None)])
def test_judge_without_n_or_a_threshold_raises_and_writes_no_verdict(bad, tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    fm = FakeMeasurer()
    _patch(mod, monkeypatch, fm, head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": dict(N2_0, **bad)}, smoke_=False)
    with pytest.raises((ValueError, TypeError)):
        mod.main([], require_root=False)
    assert "n2" not in json.loads(Path(FULL).read_text()) and "kc_vectors" not in [k for k, _ in fm.calls]


def test_judge_short_post_verdict_kc_raises_and_the_written_verdict_stays(tmp_path, monkeypatch):
    mod = _script("run_n2_judge")
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, DropMeasurer(short_kc=True), head=HEAD_AB)
    _write(tmp_path, {"n0f": N0F, "n0": N0, "n1": N1, "n2_0": N2_0}, smoke_=False)
    with pytest.raises(ValueError, match="missing"):
        mod.main([], require_root=False)
    b = json.loads(Path(FULL).read_text())["n2"]
    assert b["outcome"] == SUPPORTED and b["post_verdict_kc"] is None


def test_arm_helpers_refuse_short_rows_and_read_their_numbers_from_n_spec():
    st = {s: {"odor": {"ORN_" + s: 1.0}} for s in SPEC.judged_stimuli}
    items = n_cli.arm_items(st, SPEC, SPEC.n2_conditions, (5, 6), True)
    assert [(i["cond"], i["edit"], i["seed"]) for i in items[:3]] == [("sim_on", "none", 5), ("sim_on", "none", 6),
                                                                      ("sim_off", "apl_to_kc_zero", 5)]
    assert items[0]["odor_x"] == st["4:1"]["odor"] and items[0]["odor_y"] == st["1:4"]["odor"]
    assert items[-1]["odor_y"] == st["dDL"]["odor"] and all(i["plastic"] is True for i in items)
    rows = [dict(cond="a", seed=s) for s in (2, 1)]
    assert [r["seed"] for r in n_cli.by_condition(rows, ["a"], (1, 2), "t")["a"]] == [1, 2]
    for bad, names, seeds in [(rows, ["a"], (1, 2, 3)), (rows + rows[:1], ["a"], (1, 2)), (rows, ["a", "b"], (1, 2)),
                              ([], ["a"], (1,)), (rows, ["b"], (1, 2)), (rows, [], (1, 2)), (rows, ["a"], ())]:
        with pytest.raises(ValueError):
            n_cli.by_condition(bad, names, seeds, "t")
    probe = lambda P: {"x": {"P": P}, "y": {"P": P}}
    by = {"a": [dict(pre=probe(SPEC.state_p_min), post=probe(SPEC.state_p_min - 1))]}
    assert n_cli.firing_shares(by, SPEC) == {"a": 0.5}
    with pytest.raises(ValueError):
        n_cli.firing_shares({"a": []}, SPEC)
    assert n_cli.UNEDITED == SPEC.conditions[0][1] == SPEC.n2_conditions[0][2]


def test_later_scripts_hold_no_copy_of_a_shared_helper():
    import re
    for name in LATER:
        src = (ROOT / "scripts" / f"{name}.py").read_text()
        assert "def hooks" not in src and "FlyPool" not in src and "drive_hz" not in src
        assert "timeout" not in src and "cwd" not in src and "argparse" not in src
        assert '__name__ == "__main__"' in src and "hooks(sys.modules[__name__])" in src
        code = src.split('"""', 2)[2]                                          # after the module docstring
        assert not re.search(r"(?<![\w.])\d+\.\d+|\b\d{2,}\b", code), name   # no number of its own: n_spec's
        assert set(n_cli.hooks(_script(name))) == set(n_cli.HOOK_NAMES)
