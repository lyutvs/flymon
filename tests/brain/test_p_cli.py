# tests/brain/test_p_cli.py
"""Spec P.3 / P.6.4 / P.6.5: c1 comes from block n1's dissimilar-pair o (within 5e-4 of the declared -2.433; the other
pair is not P's business), O2's per-seed source is exactly block o2's run (its measure key, complete, D reproduced),
and the pool is built from the spec."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flymon.brain import p_cli
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC as N_SPEC
from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.p_spec import SPEC, smoke

OK = lambda *a: None          # noqa: E731  (a committed summary)
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}


def _n1(tmp_path, dis=-2.4325499, sim=-9.0):
    s = tmp_path / "n.json"
    s.write_text(json.dumps({"n1": {"pairs": {"sim": {"o": sim}, "dis": {"o": dis}}}}))
    return s


def test_c1_reads_the_dissimilar_pair_of_block_n1(tmp_path):
    rec, why = p_cli.c1_source(SPEC, False, committed=OK, path=_n1(tmp_path))
    assert why is None and rec["pair"] == "dis" and rec["exact"] == -2.4325499 and rec["declared"] == -2.433
    assert rec["c1"] == pytest.approx(0.25 * 2.4325499) and round(rec["c1"], 3) == 0.608 and rec["tol"] == 0.0005


def test_c1_refuses_a_disagreeing_or_uncommitted_block(tmp_path):
    assert "disagree" in p_cli.c1_source(SPEC, False, committed=OK, path=_n1(tmp_path, dis=-2.4400))[1]
    assert p_cli.c1_source(SPEC, False, committed=lambda *a: "not tracked", path=_n1(tmp_path))[1] == "not tracked"
    assert "no usable block n1" in p_cli.c1_source(SPEC, False, committed=OK, path=tmp_path / "none.json")[1]


def test_c1_in_smoke_without_a_block_uses_the_declared_o(tmp_path):
    rec, why = p_cli.c1_source(smoke(SPEC), True, committed=lambda *a: "never asked", path=tmp_path / "none.json")
    assert why is None and rec["exact"] is None and rec["c1"] == pytest.approx(0.25 * 2.433)


def _probe(seed, A=20.0, P=10):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1,
                wall_s=0.01, steps=1400)


def _o2_world(tmp_path, key="m" * 64, D_shift=0.0, drop=0, other_key_extra=True):
    """A committed-looking block o2 and its cache: punish lowers A_X by 2 + 2.4, plastic by 2 (D = -2.4 with ZU)."""
    cache = tmp_path / "o2_arm"
    cache.mkdir()
    n = 0
    for x, y, _ in O_SPEC.o2_pairs:
        for a, pu, pl, dz in O_SPEC.o2_arms:
            for s in O_SPEC.o2_seeds:
                ax = {"plastic": 2.0, "frozen": 0.0, "punish": 4.4, "da_zero": 0.0}[a]
                res = dict(seed=s, edit="none", arm=a, punish=pu, plastic=pl, da_zero=dz, csc_sha256="sha",
                           pre={"x": _probe(s), "y": _probe(s)}, post={"x": _probe(s, A=20.0 - ax), "y": _probe(s)},
                           w0_sha256="w0", w_post_sha256="w0")
                n += 1
                if n <= drop:
                    continue
                (cache / f"{n:04d}.json").write_text(json.dumps(dict(
                    key=str(n), kind="o2_arm", run_id="run-o2", code_key=key, inputs=dict(x=x, y=y, arm=a, seed=s),
                    result=res)))
    if other_key_extra:                                    # an entry of another run: never read
        (cache / "zzzz.json").write_text(json.dumps(dict(key="z", kind="o2_arm", run_id="old", code_key="o" * 64,
                                                         inputs=dict(x="4:1", y="1:4", arm="punish", seed=1),
                                                         result=dict(seed=1))))
    summ = tmp_path / "o_states.json"
    summ.write_text(json.dumps({"o2": dict(outcome="JUDGED", smoke=False, run_id="run-o2", measure_key="m" * 64,
                                           c3=dict(z=ZU), x={x: dict(stats=dict(D=-2.4 + D_shift))
                                                             for x, _, _ in O_SPEC.o2_pairs})}))
    return summ, cache


def test_o2_source_reads_exactly_block_o2s_run(tmp_path):
    summ, cache = _o2_world(tmp_path)
    rec, why = p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)
    assert why is None and rec["n_rows"] == O_SPEC.n_o2_arms() and rec["run_id"] == "run-o2" and rec["z"] == ZU
    assert rec["D"]["dDL"]["recomputed"] == pytest.approx(-2.4)


def test_o2_source_refuses_a_cache_that_is_not_the_blocks_run(tmp_path):
    summ, cache = _o2_world(tmp_path, D_shift=0.01)
    assert "block o2 says" in p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]


def test_o2_source_refuses_a_missing_entry(tmp_path):
    summ, cache = _o2_world(tmp_path, drop=1)
    assert "no row" in p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]


def test_o2_source_refuses_another_measure_key_or_an_uncommitted_summary(tmp_path):
    summ, cache = _o2_world(tmp_path, key="q" * 64)
    assert "no row" in p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]
    assert p_cli.o2_source(SPEC, False, committed=lambda *a: "dirty", summary=summ, cache=cache)[1] == "dirty"


def test_pool_settings_come_from_the_spec(monkeypatch):
    from flymon.brain import fly_pool
    got = {}
    monkeypatch.setattr(fly_pool, "FlyPool", lambda *a, **k: got.update(k) or SimpleNamespace(close=lambda: None))
    ctx = SimpleNamespace(args=SimpleNamespace(npz="x.npz", workers=3), c3=Params(), spec=smoke(SPEC),
                          out=Path("results/p/x"), key={"key": "k" * 64}, rid="r", spec_commit="c")
    m, pool = p_cli.make_measurer(ctx)
    assert got["timeout_s"] == N_SPEC.pool_timeout_s and got["workers"] == 3
    assert got["punish_type"] == N_SPEC.h3.punish_type and got["reward_type"] == N_SPEC.h3.reward_type
    assert m.cache.root == Path("results/p/x/cache") and m.cache.spec_commit == "c" and m.pspec == smoke(SPEC)
    assert p_cli.parser(None).parse_args([]).workers == N_SPEC.workers


def test_p_cli_refuses_when_a_hashed_file_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(p_cli, "HASHED_FILES", p_cli.HASHED_FILES + ("flymon/brain/no_such_file.py",))
    with pytest.raises(SystemExit) as e:
        p_cli.code_keys(p_cli.NPZ)
    assert e.value.code == 2 and "no_such_file" in capsys.readouterr().err


# ---- controller rulings (Task 4 / Task 5 reviews, reading 11): O2's seed order, the OC null, the code keys, the
# separate smoke / real roots in both directions, P.6.5's single rerun, no block on a refused input
def test_o2_source_refuses_x_rows_whose_seeds_differ(tmp_path, monkeypatch):
    summ, cache = _o2_world(tmp_path)
    f = sorted(cache.glob("0*.json"))[0]                    # the first 4:1 row: another seed, past o_rules' gate
    d = json.loads(f.read_text())
    d["result"]["seed"] = d["inputs"]["seed"] = 99
    f.write_text(json.dumps(d))
    monkeypatch.setattr(p_cli, "validity", lambda *a: [])
    why = p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)[1]
    assert "identical seeds" in why


def test_o2_source_states_the_oc_null_and_returns_rows_in_key_order(tmp_path):
    summ, cache = _o2_world(tmp_path)
    rec, _ = p_cli.o2_source(SPEC, False, committed=OK, summary=summ, cache=cache)
    assert rec["null"] == p_cli.OC_NULL == "only s is shifted by mean(ℓ) − c₁; t fixed"
    keys = [(r["x"], r["arm"], int(r["seed"])) for r in rec["rows"]]
    assert keys == sorted(keys) and rec["seeds"] == list(O_SPEC.o2_seeds)


def test_code_keys_cover_measure_files_for_the_cache_and_hashed_files_for_the_manifest(monkeypatch):
    seen, root = [], Path(__file__).resolve().parents[2]
    monkeypatch.setattr(p_cli, "HASHED_FILES", tuple(f for f in p_cli.HASHED_FILES if (root / f).exists()))  # Task 7
    monkeypatch.setattr(p_cli, "code_key", lambda npz, files: seen.append(files) or {"key": str(len(seen))})
    key, manifest = p_cli.code_keys("x.npz")
    assert seen == [p_cli.MEASURE_FILES, p_cli.HASHED_FILES] and key == {"key": "1"} and manifest == {"key": "2"}
    assert "flymon/brain/p_measure.py" in p_cli.MEASURE_FILES


def test_every_p_hashed_file_but_the_scripts_exists():
    root = Path(__file__).resolve().parents[2]
    missing = [f for f in p_cli.HASHED_FILES if not (root / f).exists()]
    assert set(missing) <= {"scripts/run_p.py", "scripts/p_oc.py"}          # Task 7 adds the two scripts


CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])   # noqa: E731
KEYS = ({"key": "k" * 64, "files": {}}, {"key": "m" * 64, "files": {}})


class _Halt(Exception):
    pass


def _hooks(**over):
    h = dict(check_committed=OK, code_keys=lambda npz: KEYS, git_state=CLEAN,
             load_c3=lambda spec: (Params(), {"A": "MBON13", "P": "MBON05"}, ZU, None), m0d_sha=lambda spec: "s" * 64,
             make_measurer=lambda ctx: pytest.fail("no pool may start here"), model_types=lambda npz: [],
             spec_commit=lambda path: "c0ffee", c1_source=None, o2_source=None)
    h.update(over)
    return h


def _body(outcome="LEARNS_CONFIRMATORY", code=0, seen=None):
    def body(ctx):
        if seen is not None:
            seen.append(ctx)
        return dict(outcome=outcome, label=outcome, sentence=f"P: {outcome}"), code
    return body


def _stage(argv, body=None, rerun_once=True, **over):
    return p_cli.main_stage("p", argv, body or _body(), _hooks(**over), require_root=False, rerun_once=rerun_once)


@pytest.fixture
def at_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(p_cli, "out_allowed", lambda out: True)
    return tmp_path


def _real(doc):
    p = Path("results/summary/p_learning.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def test_main_stage_refuses_another_directory_and_an_out_outside_results_p(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert p_cli.main_stage("p", [], _body(), _hooks()) == 2 and "repository root" in capsys.readouterr().err
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    assert p_cli.main_stage("p", ["--out", "results/o/x"], _body(), _hooks()) == 2
    assert "results/p/" in capsys.readouterr().err and not any(tmp_path.iterdir())


def test_smoke_and_real_runs_use_separate_roots_both_ways(at_tmp, capsys):
    seen = []
    assert _stage(["--smoke"], _body(seen=seen)) == 0 and _stage([], _body(seen=seen)) == 0
    assert [c.out for c in seen] == [Path(p_cli.SMOKE_OUT), Path(p_cli.RUN_OUT)] and [c.smoke for c in seen] == [
        True, False]
    assert json.loads(Path(p_cli.SMOKE_SUMMARY).read_text())["p"]["smoke"] is True
    assert json.loads(Path("results/summary/p_learning.json").read_text())["p"]["smoke"] is False
    for argv in (["--smoke", "--out", p_cli.RUN_OUT], ["--smoke", "--summary", "results/summary/p_learning.json"],
                 ["--out", p_cli.SMOKE_OUT + "/x"], ["--summary", p_cli.SMOKE_SUMMARY]):
        capsys.readouterr()
        assert _stage(argv, _body(seen=seen)) == 2 and "smoke" in capsys.readouterr().err
    assert len(seen) == 2


def test_main_stage_refuses_dirty_files_and_an_uncommitted_spec_before_any_input(at_tmp, capsys):
    dirty = lambda files: dict(commit="x", dirty_hashed=["flymon/brain/p_rules.py"], dirty_other=[])  # noqa: E731
    assert _stage([], git_state=dirty) == 2 and "dirty" in capsys.readouterr().err
    assert _stage([], spec_commit=lambda path: None) == 2 and "no commit" in capsys.readouterr().err
    assert not (at_tmp / "results").exists()


def test_a_refused_input_writes_no_block_and_starts_no_pool(at_tmp, capsys):
    called = []
    rc = p_cli.main_stage("p", [], _body(seen=called), _hooks(), require_root=False,
                          prepare=lambda spec, smoke, hooks, doc, summary, z: (None, "no usable block n1 in x"))
    assert rc == 2 and "block n1" in capsys.readouterr().err and called == [] and not (at_tmp / "results").exists()


def test_prepare_sees_the_summary_and_c3_z_and_its_extra_reaches_the_body(at_tmp):
    got, seen = {}, []

    def prepare(spec, smoke, hooks, doc, summary, z):
        got.update(doc=doc, summary=summary, z=z, smoke=smoke)
        return {"c1": 0.6}, None
    assert p_cli.main_stage("oc", [], _body(seen=seen), _hooks(), require_root=False, prepare=prepare) == 0
    assert got == dict(doc={}, summary=Path("results/summary/p_learning.json"), z=ZU, smoke=False)
    assert seen[0].extra == {"c1": 0.6}


def test_an_invalid_block_is_written_with_exit_5(at_tmp):
    assert _stage([], _body("INVALID", 5)) == 5
    assert json.loads(Path("results/summary/p_learning.json").read_text())["p"]["outcome"] == "INVALID"


def test_a_judged_block_is_final(at_tmp, capsys):
    _real({"p": {"outcome": "NO_LEARNING", "run_id": "r0", "code": {"key": "q" * 64}}})
    assert _stage([]) == 2 and "post-hoc" in capsys.readouterr().err
    assert _stage(["--rerun-after-invalid"]) == 2 and "post-hoc" in capsys.readouterr().err


def test_an_invalid_block_needs_the_flag_new_code_and_a_stage_that_allows_a_rerun(at_tmp, capsys):
    _real({"p": {"outcome": "INVALID", "run_id": "r0", "code": {"key": "m" * 64}}})
    assert _stage([]) == 2 and "--rerun-after-invalid" in capsys.readouterr().err
    assert _stage(["--rerun-after-invalid"]) == 2 and "fix the code" in capsys.readouterr().err
    assert _stage(["--rerun-after-invalid"], rerun_once=False) == 2 and "post-hoc" in capsys.readouterr().err


def test_the_one_rerun_keeps_the_invalid_block_and_is_final(at_tmp, capsys):
    prior = {"outcome": "INVALID", "run_id": "r0", "code": {"key": "q" * 64}}
    p = _real({"p": prior, "oc": {"outcome": "OC_RECORDED"}})
    assert _stage(["--rerun-after-invalid"], _body("INVALID", 5)) == 5
    doc = json.loads(p.read_text())
    assert doc["p_invalid"] == prior and doc["p"]["rerun_of"] == "r0" and doc["p"]["outcome"] == "INVALID"
    assert doc["oc"] == {"outcome": "OC_RECORDED"}
    capsys.readouterr()
    assert _stage(["--rerun-after-invalid"], code_keys=lambda npz: (KEYS[0], {"key": "n" * 64})) == 2
    assert "one rerun only" in capsys.readouterr().err and json.loads(p.read_text()) == doc


def test_the_rerun_flag_without_an_invalid_block_is_refused(at_tmp, capsys):
    assert _stage(["--rerun-after-invalid"]) == 2 and "needs an INVALID block" in capsys.readouterr().err
    assert not (at_tmp / "results").exists()
