"""Spec L's CLIs: every refusal comes before any measurement or write (the runs themselves are controller steps)."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import l_cli

ROOT = Path(__file__).resolve().parents[1]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


S0, OC = _load("run_l_stage0"), _load("run_l_oc")
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])


@pytest.mark.parametrize("cli", [S0, OC])
def test_outside_the_repository_root_is_refused(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("cli", [S0, OC])
def test_dirty_hashed_files_are_refused(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/l_rules.py"],
                                                             dirty_other=[]))
    assert cli.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


@pytest.mark.parametrize("cli", [S0, OC])
def test_an_out_dir_elsewhere_is_refused(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", CLEAN)
    assert cli.main(["--out", "results/m0d/k/run"], require_root=False) == 2
    assert "results/m0d/l/" in capsys.readouterr().err


def test_stage0_refuses_a_foreign_connectome(tmp_path, monkeypatch, capsys):
    npz = tmp_path / "other.npz"; npz.write_bytes(b"not the connectome")
    monkeypatch.setattr(S0, "git_state", CLEAN)
    assert S0.main(["--npz", str(npz)], require_root=False) == 2 and "declared connectome" in capsys.readouterr().err


@needs_npz
def test_stage0_refuses_a_foreign_pinned_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(S0, "git_state", CLEAN)
    bad = tmp_path / "c.json"; bad.write_text("{}")
    assert S0.main(["--ceiling", str(bad)], require_root=False) == 2 and "sha256" in capsys.readouterr().err
    assert S0.main(["--k-act", str(bad)], require_root=False) == 2 and "sha256" in capsys.readouterr().err


def _m0d(tmp_path, edit):
    doc = json.loads((ROOT / "results/summary/m0d.json").read_text())
    edit(doc["h4"]["h4"]["combos"]["C3"])
    f = tmp_path / "m0d.json"; f.write_text(json.dumps(doc))
    return str(f)


@needs_npz
def test_stage0_refuses_another_readout_or_a_missing_z(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(S0, "git_state", CLEAN)
    m = _m0d(tmp_path, lambda c: c.update(readout={"A": "MBON18", "P": "MBON05"}))
    assert S0.main(["--m0d", m], require_root=False) == 2 and "readout" in capsys.readouterr().err
    m = _m0d(tmp_path, lambda c: c.pop("z"))
    assert S0.main(["--m0d", m], require_root=False) == 2 and "z" in capsys.readouterr().err


@needs_npz
def test_stage0_refuses_another_odd_pair_list_before_building(monkeypatch, capsys):
    monkeypatch.setattr(S0, "git_state", CLEAN)
    monkeypatch.setattr(S0, "odd_pairs", lambda pops: [])
    monkeypatch.setattr(S0, "build", lambda *a, **k: pytest.fail("build ran before the digest refusal"))
    assert S0.main([], require_root=False) == 2 and "odd pair list" in capsys.readouterr().err


def _k_rows():
    """Odour 0 fires KCs {1, 2} (spikes 3, 1) on seed 500 and {2} (2) on 501; odour 1 fires {0} (5) on both."""
    return [dict(i=0, seed=500, kc=[1, 2], n=[3, 1]), dict(i=0, seed=501, kc=[2], n=[2]),
            dict(i=1, seed=500, kc=[0], n=[5]), dict(i=1, seed=501, kc=[0], n=[5])]


def test_check_ii_recomputes_frac_and_spikes_per_seed_from_the_k_cache():
    row = dict(axis="b", turn=1, x="a", y="b", kc=dict(x=dict(frac=[2 / 4, 1 / 4], spikes=[4, 2]),
                                                    y=dict(frac=[1 / 4, 1 / 4], spikes=[5, 5])))
    assert S0.kc_mismatch([row], _k_rows(), [(0, 1)], 4, [500, 501]) == []
    row["kc"]["y"]["spikes"] = [5, 4]
    assert S0.kc_mismatch([row], _k_rows(), [(0, 1)], 4, [500, 501]) == [
        dict(key=["b", 1, "a", "b"], side="y", field="spikes", got=[5, 4], want=[5, 5])]
    row["kc"]["x"]["frac"] = [2 / 4, 2 / 4]
    assert [m["field"] for m in S0.kc_mismatch([row], _k_rows(), [(0, 1)], 4, [500, 501])] == ["frac", "spikes"]


def test_the_oc_json_forms_and_its_row_coverage_refusal():
    gate_ps = {(0.4, 0.0): {"SCREEN_GO": 0.5}, (0.4, 0.1): {"SCREEN_GO": 0.25}}
    assert OC.gate_records(gate_ps) == [dict(q=0.4, q_fail=0.0, P={"SCREEN_GO": 0.5}),
                                        dict(q=0.4, q_fail=0.1, P={"SCREEN_GO": 0.25})]
    json.dumps(OC.gate_records(gate_ps))
    rows = [dict(q_b=0.5), dict(q_b=0.4)]
    assert OC.uncovered_q((0.4, 0.5), rows) == []
    assert OC.uncovered_q((0.4, 0.5, 0.1 + 0.2), rows) == [0.1 + 0.2]      # no silent float match


def test_oc_refuses_without_a_stage0_block(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(OC, "git_state", CLEAN)
    s = tmp_path / "l_screen.json"; s.write_text("{}")
    assert OC.main(["--summary", str(s)], require_root=False) == 2 and "stage0" in capsys.readouterr().err


@needs_npz
def test_oc_refuses_a_stage0_under_other_code(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(OC, "git_state", CLEAN)
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps({"stage0": {"measure_key": "k", "code": {"key": "m"}}}))
    assert OC.main(["--summary", str(s)], require_root=False) == 2 and "other code" in capsys.readouterr().err


@needs_npz
def test_oc_refuses_an_uncommitted_summary_and_a_smoke_summary_outside_l(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(OC, "git_state", CLEAN)
    monkeypatch.setattr(OC, "same_code", lambda block, key, manifest: True)
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps({"stage0": {"measure_key": "k", "code": {"key": "m"}}}))
    assert OC.main(["--summary", str(s)], require_root=False) == 2 and "tracked" in capsys.readouterr().err
    assert OC.main(["--smoke", "--summary", str(s)], require_root=False) == 2
    assert "results/m0d/l/" in capsys.readouterr().err


def test_same_code_needs_both_keys():
    b = {"measure_key": "k", "code": {"key": "m"}}
    assert l_cli.same_code(b, "k", "m")
    assert not l_cli.same_code(b, "k", "x") and not l_cli.same_code(b, "x", "m") and not l_cli.same_code({}, "k", "m")


def test_check_committed_names_an_untracked_or_edited_summary(tmp_path):
    f = tmp_path / "l_screen.json"; f.write_text("{}")
    assert "tracked" in l_cli.check_committed(f, ["stage0"])
    assert l_cli.check_committed(ROOT / "pyproject.toml", ["stage0"]) is None


def test_code_keys_refuse_a_missing_hashed_file(tmp_path, monkeypatch, capsys):
    """Every L file exists now: a missing hashed file is a refusal naming it, never a smaller manifest."""
    npz = tmp_path / "x.npz"; npz.write_bytes(b"x")
    code, manifest = l_cli.code_keys(str(npz))
    assert set(manifest["files"]) - {"npz:x.npz"} == set(l_cli.HASHED_FILES) and set(code["files"]) <= set(manifest["files"])
    monkeypatch.setattr(l_cli, "HASHED_FILES", l_cli.HASHED_FILES + ("scripts/run_l_stage9.py",))
    with pytest.raises(SystemExit) as e:
        l_cli.code_keys(str(npz))
    assert e.value.code == 2 and "scripts/run_l_stage9.py" in capsys.readouterr().err


# ---------------------------------------------------------------- stage 0's flow on stand-ins (no pool, no engine)
class _Pool:
    n_workers = 4

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stage0_world(monkeypatch, tmp_path, iii_ok=True, ii_ok=True):
    """Real inputs (connectome, pinned files, m0d.json) up to the pool; the pool, the oracle, D.6 and the formula's
    statistics are stand-ins: the oracle's kc rows are rebuilt from the K cache so self-check (ii) is exercised."""
    import dataclasses
    from flymon.brain import j_store
    from flymon.brain.h4_pairs import even_pairs, pair_key
    from flymon.brain.k_measure import KMeasurer
    from flymon.brain.l_spec import SPEC
    m0d = json.loads((ROOT / "results/summary/m0d.json").read_text())
    c3, _, _ = j_store.load_c3(m0d)
    rec = m0d["h4"]["h4"]["combos"]["C3"]
    spec = dataclasses.replace(SPEC, ceiling_path=str(ROOT / SPEC.ceiling_path), k_act_path=str(ROOT / SPEC.k_act_path),
                               m0d_path=str(ROOT / SPEC.m0d_path))
    k_raw = json.loads(Path(spec.k_act_path).read_text())["result"]
    per = {(r["i"], r["seed"]): r for r in k_raw}
    seeds = list(SPEC.j.h4.act_seeds)
    conn = S0.Connectome.load(str(ROOT / "data/malecns.npz"))
    pops = S0.Populations.from_connectome(conn)
    n_kc = len(pops.kc)
    km = KMeasurer(None, None, S0.odd_pairs(pops), n_kc, None)
    idx = {pair_key(p): ij for p, ij in zip(km.pairs, km.index)}

    def side(i):
        return dict(frac=[len(per[(i, s)]["kc"]) / n_kc for s in seeds], spikes=[sum(per[(i, s)]["n"]) for s in seeds])

    class M4:
        def __init__(self, pool, spec4, pairs, pools, cache, h3):
            self.pairs, self.params_seen = pairs, [c3]

        def oracle(self, params, readout, z):
            out = []
            for p in self.pairs:
                k = pair_key(p)
                want = next((q for q in rec["oracle"]["pairs"] if pair_key(q) == k), None)
                stats = {f: want[f] for f in ("d_pre", "r", "p", "m", "testable")} if want else None
                if stats and not iii_ok:
                    stats = dict(stats, m=stats["m"] + 1e-12)
                ij = idx.get(k)
                kcs = dict(x=side(ij[0]), y=side(ij[1])) if ij else {}
                if ij and not ii_ok:
                    kcs["y"] = dict(kcs["y"], spikes=[s + 1 for s in kcs["y"]["spikes"]])
                pre = {t: [[6, 7]] * 8 for t in ("MBON13", "MBON18", "MBON05", "MBON21")}
                out.append(dict(axis=p["axis"], turn=p["turn"], x=p["x"], y=p["y"], report=dict(stats=stats),
                                select=dict(pre=pre), counts=dict(pre=pre), kc=kcs))
            return out

    def fake_rows_stats(rows, z, expected, spec4):
        st = {pair_key(r): dict(d_pre=0.0, r=1.0, p=-1.0, m=1.0, testable=bool(r["turn"] % 4 == 1)) for r in rows}
        return dict(reasons=[], stats=st, pairs=[])

    s = dict(conn=conn, pops=pops, core=m0d["h3"]["pools"], pools=m0d["h3"]["pools"], pairs=even_pairs(pops),
             pairs_digest=SPEC.j.h4.pairs_digest, all51=None, ctx=None)
    monkeypatch.setattr(S0, "git_state", CLEAN)
    monkeypatch.setattr(S0, "out_allowed", lambda out: True)
    monkeypatch.setattr(S0.j_store, "load_c3", lambda summary, block, name: (c3, None, None))
    monkeypatch.setattr(S0, "build", lambda *a, **k: s)
    monkeypatch.setattr(S0, "FlyPool", _Pool)
    monkeypatch.setattr(S0, "H4Measurer", M4)
    monkeypatch.setattr(S0, "pair_stats", lambda report, z, tm: report["stats"])
    monkeypatch.setattr(S0, "pair_rows_stats", fake_rows_stats)
    monkeypatch.setattr(S0, "JMeasurer", lambda *a: type("J", (), {"params_seen": [c3]})())
    monkeypatch.setattr(S0, "measure_d6", lambda jm, ctx, params, seeds, judged: dict(a="A", b="B", seeds=list(seeds)))
    monkeypatch.chdir(tmp_path)
    return spec


@needs_npz
def test_stage0_complete_run_writes_41_feature_records_and_the_block(monkeypatch, tmp_path):
    spec = _stage0_world(monkeypatch, tmp_path)
    npz = str(ROOT / "data/malecns.npz")
    assert S0.main(["--npz", npz], spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads((tmp_path / "results/summary/l_screen.json").read_text())["stage0"]
    assert b["checks"]["i"]["ok"] and b["checks"]["ii"]["ok"] and b["checks"]["iii"]["ok"]
    assert len(b["feats"]) == 41 and [f["set"] for f in b["feats"]].count("odd") == 20
    assert {f["turn"] % 2 for f in b["feats"] if f["set"] == "odd"} == {1}
    assert all(f["feat"]["G"] for f in b["feats"] if f["set"] == "odd")        # [6, 7] on every select seed
    assert b["d6"]["seeds"] == [int(x) for x in S0.d6a.SEEDS] and len(b["odd"]) == 20
    assert b["report_sha256"] and Path(b["report"]).exists()


@needs_npz
@pytest.mark.parametrize("which", ["iii", "ii"])
def test_stage0_self_check_mismatch_exits_5_without_the_block(which, monkeypatch, tmp_path, capsys):
    spec = _stage0_world(monkeypatch, tmp_path, iii_ok=which != "iii", ii_ok=which != "ii")
    assert S0.main(["--npz", str(ROOT / "data/malecns.npz")], spec=spec, summary_spec=spec, require_root=False) == 5
    assert not (tmp_path / "results/summary/l_screen.json").exists()
    rep = json.loads(next((tmp_path / "results/m0d/l/run/runs").glob("*-stage0.json")).read_text())
    assert rep["status"] == f"mismatch_{which}" and not rep["checks"][which]["ok"]
    assert "DIFFERS" in capsys.readouterr().out


@needs_npz
def test_stage0_smoke_measures_two_odd_pairs_and_writes_no_block(monkeypatch, tmp_path):
    spec = _stage0_world(monkeypatch, tmp_path)
    assert S0.main(["--npz", str(ROOT / "data/malecns.npz"), "--smoke"], spec=spec, summary_spec=spec,
                   require_root=False) == 0
    assert not (tmp_path / "results/summary/l_screen.json").exists()
    rep = json.loads(next((tmp_path / "results/m0d/l/smoke/runs").glob("*-stage0.json")).read_text())
    assert rep["n_odd"] == 2 and len(rep["feats"]) == 23 and rep["d6"]["seeds"] == [int(x) for x in S0.d6a.SEEDS[:2]]


@needs_npz
def test_oc_smoke_writes_its_block_into_the_smoke_summary_with_provenance(monkeypatch, tmp_path):
    import dataclasses
    from flymon.brain.l_spec import SPEC
    spec = dataclasses.replace(SPEC, oc_draws=3, m0d_path=str(ROOT / SPEC.m0d_path))
    feats = [dict(turn=t, set="even" if t % 2 == 0 else "odd", testable=t % 3 == 0,
                  feat=dict(G=t % 4 != 0, S=0.1 * t, f2=float(t), f3=0.5)) for t in range(16) for _ in range(2)]
    monkeypatch.setattr(OC, "git_state", CLEAN)
    monkeypatch.setattr(OC, "same_code", lambda block, key, manifest: True)
    monkeypatch.setattr(OC, "out_allowed", lambda out: str(out).startswith("results/m0d/l/"))   # root = tmp_path
    monkeypatch.chdir(tmp_path)
    s = Path("results/m0d/l/smoke/l_screen.json"); s.parent.mkdir(parents=True)
    s.write_text(json.dumps({"stage0": {"run_id": "r0", "feats": feats}}))
    assert OC.main(["--smoke", "--summary", str(s), "--npz", str(ROOT / "data/malecns.npz")], spec=spec,
                   require_root=False) == 0
    doc = json.loads(s.read_text())
    oc = doc["oc"]
    assert doc["stage0"]["run_id"] == "r0" and oc["stage0_run_id"] == "r0"
    assert [(g["q"], g["q_fail"]) for g in oc["gate"]] == [(q, f) for q in SPEC.oc_q for f in SPEC.oc_q_fail]
    assert {r["q_b"] for r in oc["stage3"]} == set(SPEC.oc_q) and {r["u"] for r in oc["stage3"]} == {0, "C3"}
    assert oc["whole"]["rows"] and oc["report_sha256"]
    sha = oc["provenance"]["sha256"]
    assert "flymon/brain/l_oc.py" in sha and "docs/superpowers/specs/j-diag/stage2_oc.py" in sha
    assert oc["provenance"]["summary_at_head"]["sha256"] is None           # a smoke summary is not tracked


# ================================================================ stages 1-3 (Task 8)
S1, S2, S3 = _load("run_l_stage1"), _load("run_l_stage2"), _load("run_l_stage3")
KEYS = {"measure_key": "k", "code": {"key": "m"}}
DIRTY = lambda files: dict(commit="x", dirty_hashed=["flymon/brain/l_rules.py"], dirty_other=[])
FULL = {"stage0": dict(KEYS), "oc": dict(KEYS), "stage1": dict(KEYS, gate={"outcome": "SCREEN_GO"}),
        "stage2": dict(KEYS, outcome="SCREENED")}


@pytest.mark.parametrize("cli", [S1, S2, S3])
def test_later_stages_refuse_outside_root_and_dirty(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(cli, "git_state", DIRTY)
    assert cli.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


@pytest.mark.parametrize("cli,need", [(S1, "oc"), (S2, "stage1"), (S3, "stage2")])
def test_each_stage_refuses_without_its_previous_block(cli, need, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", CLEAN)
    monkeypatch.setattr(cli, "check_committed", lambda summary, blocks: None)
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps({"stage0": {"measure_key": "k", "code": {"key": "m"}}}))
    assert cli.main(["--summary", str(s)], require_root=False) == 2 and need in capsys.readouterr().err


def test_the_missing_block_refusal_lists_every_missing_block(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S3, "git_state", CLEAN)
    monkeypatch.setattr(S3, "check_committed", lambda summary, blocks: None)
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps({"stage0": dict(KEYS)}))
    assert S3.main(["--summary", str(s)], require_root=False) == 2
    err = capsys.readouterr().err
    assert all(b in err for b in ("oc", "stage1", "stage2"))


@pytest.mark.parametrize("cli", [S1, S2, S3])
def test_each_stage_refuses_an_uncommitted_summary(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", CLEAN)
    s = tmp_path / "l_screen.json"; s.write_text("{}")
    assert cli.main(["--summary", str(s)], require_root=False) == 2
    err = capsys.readouterr().err
    assert "commit" in err or "tracked" in err


@pytest.mark.parametrize("cli", [S1, S2, S3])
def test_each_stage_refuses_a_summary_edited_after_its_commit(cli, tmp_path, monkeypatch, capsys):
    """Review Focus 4: a committed summary edited afterwards is refused before the code keys or any measurement."""
    import subprocess
    repo = tmp_path / "repo"; (repo / "results/summary").mkdir(parents=True)
    s = repo / "results/summary/l_screen.json"; s.write_text(json.dumps(FULL))
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                                     "-c", "commit.gpgsign=false", *a], check=True, capture_output=True)
    git("init", "-q"); git("add", "."); git("commit", "-qm", "summary")
    monkeypatch.setattr(l_cli, "ROOT", repo)
    assert l_cli.check_committed(s, ["stage0"]) is None
    s.write_text(json.dumps(dict(FULL, stage1=dict(KEYS, gate={"outcome": "SCREEN_GO", "edited": True}))))
    monkeypatch.setattr(cli, "git_state", CLEAN)
    monkeypatch.setattr(cli, "out_allowed", lambda out: True)
    monkeypatch.setattr(cli, "code_keys", lambda npz: pytest.fail("code keys taken before the commit gate"))
    assert cli.main(["--summary", str(s)], require_root=False) == 2
    err = capsys.readouterr().err
    assert "uncommitted" in err and "commit" in err


@needs_npz
@pytest.mark.parametrize("cli", [S1, S2, S3])
def test_each_stage_refuses_blocks_under_other_code(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", CLEAN)
    monkeypatch.setattr(cli, "check_committed", lambda summary, blocks: None)
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps(FULL))
    assert cli.main(["--summary", str(s)], require_root=False) == 2
    err = capsys.readouterr().err
    assert "other code" in err and "stage0" in err and "oc" in err


@needs_npz
def test_stage2_refuses_a_stage1_that_did_not_go(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S2, "git_state", CLEAN)
    monkeypatch.setattr(S2, "check_committed", lambda summary, blocks: None)
    monkeypatch.setattr(S2, "same_code", lambda block, key, manifest: True)
    doc = {b: {"measure_key": "k", "code": {"key": "m"}} for b in ("stage0", "oc")}
    doc["stage1"] = {"measure_key": "k", "code": {"key": "m"}, "gate": {"outcome": "SCREEN_IMPRECISE"}}
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps(doc))
    assert S2.main(["--summary", str(s)], require_root=False) == 2 and "SCREEN_GO" in capsys.readouterr().err


@needs_npz
def test_stage3_refuses_a_stage2_that_did_not_screen(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S3, "git_state", CLEAN)
    monkeypatch.setattr(S3, "check_committed", lambda summary, blocks: None)
    monkeypatch.setattr(S3, "same_code", lambda block, key, manifest: True)
    s = tmp_path / "l_screen.json"; s.write_text(json.dumps(dict(FULL, stage2=dict(KEYS, outcome="COVERAGE_SHORT"))))
    assert S3.main(["--summary", str(s)], require_root=False) == 2 and "SCREENED" in capsys.readouterr().err


def test_smoke_stages_read_only_a_smoke_summary(monkeypatch, capsys):
    for cli in (S1, S2, S3):
        monkeypatch.setattr(cli, "git_state", CLEAN)
        assert cli.main(["--smoke"], require_root=False) == 2
        assert "results/m0d/l/" in capsys.readouterr().err


@needs_npz
def test_new_set_is_the_declared_prefix_and_refuses_another_digest():
    import dataclasses
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.l_spec import SPEC, smoke
    pops = Populations.from_connectome(Connectome.load(str(ROOT / "data/malecns.npz")))
    full = l_cli.new_set(pops, SPEC, SPEC)
    sm = l_cli.new_set(pops, smoke(SPEC), SPEC)
    assert len(full["b"]) == 160 and len(full["a"]) == 20 and len(full["skipped"]) == 6
    assert sm["b"] == full["b"][:len(sm["b"])] and {p["turn"] for p in sm["b"]} <= set(range(4))
    assert {p["turn"] for p in sm["a"]} == {0}
    with pytest.raises(ValueError, match="digest"):
        l_cli.new_set(pops, SPEC, dataclasses.replace(SPEC, b_digest="0" * 64))


def test_rule_text():
    assert l_cli.rule_text({"family": "G"}) == "G" and l_cli.rule_text(None) == "none"
    assert l_cli.rule_text({"family": "f2", "op": ">", "t": 13.5}) == "G ∧ f2 > 13.5"


# ---------------------------------------------------------------- shared stand-ins for the stage flows
def _keys(monkeypatch, cli, tmp_path):
    """code_keys / same_code / commit / dirty / --out stand-ins; returns the argv prefix (a dummy --npz)."""
    from flymon.brain.l_spec import SPEC
    npz = tmp_path / "malecns.npz"; npz.write_bytes(b"x")
    monkeypatch.setattr(cli, "code_keys", lambda n: (dict(key="k", files={"npz:malecns.npz": SPEC.j.h4.h3.connectome_sha256}),
                                                     dict(key="m")))
    monkeypatch.setattr(cli, "git_state", CLEAN)
    monkeypatch.setattr(cli, "check_committed", lambda summary, blocks: None)
    monkeypatch.setattr(cli, "same_code", lambda block, key, manifest: True)
    monkeypatch.setattr(cli, "out_allowed", lambda out: True)
    monkeypatch.chdir(tmp_path)
    return ["--npz", str(npz)]


def _summary(tmp_path, doc):
    s = tmp_path / "results/summary/l_screen.json"; s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps(doc))
    return s


# ---------------------------------------------------------------- stage 1
def _stage1_doc(label=lambda f: f["G"], g=None):
    """21 even feature records on the ceiling file's (b) pairs (select G from its select.pre), 20 odd ones (G true, the
    first odd pair's report G false); testable = label(feat)."""
    from flymon.brain.l_spec import SPEC
    d = json.loads((ROOT / SPEC.ceiling_path).read_text())
    rows = [r for r in d["rows"]["C3"] if r["mode"] == "x_only" and r["axis"] == "b"]
    G = lambda pre: all(min(xy) >= 5 for t in ("MBON13", "MBON05") for xy in pre[t])
    feats = [dict(key=["b", r["turn"], r["x"], r["y"]], turn=r["turn"], set="even",
                  feat=dict(G=G(r["select"]["pre"]) if g is None else g, S=0.1, f2=float(i), f3=0.5))
             for i, r in enumerate(rows)]
    odd = [dict(key=["b", 1 + 2 * (i % 8), f"m{i} vs o", f"m{i} vs p"],
                report_pre={"MBON13": [[6, 6]] * 8, "MBON05": [[0 if i == 0 else 6, 6]] * 8}) for i in range(20)]
    feats += [dict(key=o["key"], turn=o["key"][1], set="odd", feat=dict(G=True if g is None else g, S=0.2, f2=float(i),
                                                                         f3=0.4)) for i, o in enumerate(odd)]
    for f in feats:
        f["testable"] = bool(label(f["feat"]))
    return {"stage0": dict(KEYS, run_id="r0", feats=feats, odd=odd), "oc": dict(KEYS, run_id="r1")}


def _s1_spec():
    import dataclasses
    from flymon.brain.l_spec import SPEC
    return dataclasses.replace(SPEC, ceiling_path=str(ROOT / SPEC.ceiling_path))


def test_stage1_go_writes_the_gate_the_loto_oc_row_and_the_g_disagreements(tmp_path, monkeypatch, capsys):
    argv = _keys(monkeypatch, S1, tmp_path)
    s = _summary(tmp_path, _stage1_doc())
    spec = _s1_spec()
    assert S1.main(argv + ["--summary", str(s)], spec=spec, summary_spec=spec, require_root=False) == 0
    out = capsys.readouterr().out
    b = json.loads(s.read_text())["stage1"]
    assert b["gate"]["outcome"] == "SCREEN_GO" and b["sentence"] is None
    assert b["gate"]["final"]["rule"] == {"family": "G"} and b["gate"]["loto"]["precision"] == 1.0
    assert {r["q_b"] for r in b["oc_row"]} == {1.0} and len(b["oc_row"]) == len(spec.oc_naive_a) * len(spec.oc_ratio)
    dis = b["g_disagree"]
    assert len([x for x in dis if x["set"] == "even"]) == 3                 # L.11.2: select G 13, report G 14 (3 differ)
    assert [x["key"] for x in dis if x["set"] == "odd"] == [["b", 1, "m0 vs o", "m0 vs p"]]
    assert all(x["select_G"] != x["report_G"] for x in dis)
    assert "commit results/summary/l_screen.json before stage 2" in out
    assert b["stage0_run_id"] == "r0" and b["oc_run_id"] == "r1" and b["report_sha256"]


def test_stage1_imprecise_writes_its_sentence_and_the_oc_row(tmp_path, monkeypatch, capsys):
    argv = _keys(monkeypatch, S1, tmp_path)
    s = _summary(tmp_path, _stage1_doc(label=lambda f: False))
    spec = _s1_spec()
    assert S1.main(argv + ["--summary", str(s)], spec=spec, summary_spec=spec, require_root=False) == 0
    out = capsys.readouterr().out
    b = json.loads(s.read_text())["stage1"]
    assert b["gate"]["outcome"] == "SCREEN_IMPRECISE" and "LOTO" in b["sentence"]
    assert {r["q_b"] for r in b["oc_row"]} == {0.0} and "before stage 2" not in out


def test_stage1_without_a_rule_has_no_oc_row(tmp_path, monkeypatch, capsys):
    argv = _keys(monkeypatch, S1, tmp_path)
    s = _summary(tmp_path, _stage1_doc(g=False))
    spec = _s1_spec()
    assert S1.main(argv + ["--summary", str(s)], spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads(s.read_text())["stage1"]
    assert b["gate"]["outcome"] == "SCREEN_NO_RULE" and b["gate"]["loto"]["precision"] is None
    assert b["oc_row"] is None and "0.35" in b["sentence"]


def test_stage1_refuses_a_feature_record_with_no_report_probe(tmp_path, monkeypatch, capsys):
    argv = _keys(monkeypatch, S1, tmp_path)
    doc = _stage1_doc(); doc["stage0"]["odd"] = doc["stage0"]["odd"][1:]
    s = _summary(tmp_path, doc)
    spec = _s1_spec()
    assert S1.main(argv + ["--summary", str(s)], spec=spec, summary_spec=spec, require_root=False) == 2
    assert "report" in capsys.readouterr().err and "stage1" not in json.loads(s.read_text())


# ---------------------------------------------------------------- stage 2
TYPES = ("MBON13", "MBON18", "MBON05", "MBON21")
PASS_PRE = {t: [[6, 6]] * 8 for t in TYPES}
FAIL_PRE = {t: [[0, 6]] * 8 for t in TYPES}


class _Pops:
    kc = list(range(10))


def _s2_world(monkeypatch, tmp_path, pattern, selfcheck_ok=True, **spec_kw):
    """New (b) pair i passes G iff pattern[i]; four pairs per turn; one (a) pair. The world (connectome), the new set,
    the first even pair with its ceiling row, the readout weights, the pool and the naive measurer are stand-ins."""
    import dataclasses
    import numpy as np
    from flymon.brain.l_spec import SPEC
    spec = dataclasses.replace(SPEC, m0d_path=str(ROOT / SPEC.m0d_path), **spec_kw)
    argv = _keys(monkeypatch, S2, tmp_path)
    b = [dict(axis="b", turn=i // 4, x=f"m{i} vs o", y=f"m{i} vs p", odor_x={}, odor_y={}) for i in range(len(pattern))]
    new = dict(b=b, a=[dict(axis="a", turn=0, x="m0", y="m1", odor_x={}, odor_y={})],
               skipped=[{"key": ["b", 0, "q vs o", "q vs p"], "reason": "old_set"}])
    first = dict(axis="b", turn=0, x="Surf vs Chansey", y="Surf vs Rhydon", odor_x={}, odor_y={})
    fx = np.array([0, 0.5, 0, 0.25, 0, 0, 0, 0, 0, 0.125]); fy = np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.125])
    row = dict(key=("b", 0, first["x"], first["y"]), fx=fx, fy=fy, pre=PASS_PRE)
    calls = []

    class LM:
        def __init__(self, pool, spec_, types, cache):
            assert tuple(types) == TYPES
            self.params_seen = []

        def naive(self, params, pairs):
            calls.append([p["x"] for p in pairs])
            out = []
            for p in pairs:
                if p is first:
                    f = fx if selfcheck_ok else fx * 0.5
                    out.append(dict(fx_idx=np.flatnonzero(f).tolist(), fx_val=f[f > 0].tolist(),
                                    fy_idx=np.flatnonzero(fy).tolist(), fy_val=fy[fy > 0].tolist(), pre=PASS_PRE))
                else:
                    i = int(p["x"].split()[0][1:])
                    out.append(dict(fx_idx=[1], fx_val=[0.5], fy_idx=[2], fy_val=[0.5],
                                    pre=PASS_PRE if pattern[i] else FAIL_PRE))
            return out

    monkeypatch.setattr(S2.j_store, "load_c3", lambda summary, block, name: ("C3", None, None))
    monkeypatch.setattr(S2, "params_json", lambda p: {"p": p})
    monkeypatch.setattr(S2, "world", lambda npz: (None, _Pops()))
    monkeypatch.setattr(S2, "new_set", lambda pops, spec_, declared: new)
    monkeypatch.setattr(S2, "first_even", lambda pops, spec_, c3_json: (first, row))
    monkeypatch.setattr(S2, "readout_weights", lambda conn, pops, t, w: np.ones(10))
    monkeypatch.setattr(S2, "FlyPool", _Pool)
    monkeypatch.setattr(S2, "LMeasurer", LM)
    doc = {"stage0": dict(KEYS), "oc": dict(KEYS),
           "stage1": dict(KEYS, run_id="r2", gate={"outcome": "SCREEN_GO", "final": {"rule": {"family": "G"}}})}
    return spec, argv + ["--summary", str(_summary(tmp_path, doc))], calls


def test_stage2_passes_after_the_nth_in_the_last_batch_are_neither_screened_nor_counted(tmp_path, monkeypatch, capsys):
    """Review Focus 3 through the CLI: n_pass 3, batches of 4; the 3rd pass is position 5 (index 4), passes at 6 and 7
    were measured in the same batch but are not screened, not passed, not in the coverage denominator, not in the lift."""
    pattern = [True, True, False, False, True, True, True, False, True, True, True, True]
    spec, argv, calls = _s2_world(monkeypatch, tmp_path, pattern, n_pass=3, max_screened=12, n_lift=10)
    s = argv[-1]
    assert S2.main(argv, spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads(Path(s).read_text())["stage2"]
    assert calls[0] == ["Surf vs Chansey"] and [len(c) for c in calls[1:]] == [4, 4]      # self-check, 2 batches
    assert b["outcome"] == "SCREENED" and b["n_star"] == 5 and b["coverage"] == 3 / 5
    assert [x["key"][2] for x in b["screened"]] == [f"m{i} vs o" for i in range(5)]
    assert [k[2] for k in b["passed"]] == ["m0 vs o", "m1 vs o", "m4 vs o"]
    assert sorted(k[2] for k in b["lift"]) == ["m2 vs o", "m3 vs o"] and b["measured"] == 8
    assert b["selfcheck_naive"]["ok"] and b["selfcheck_naive"]["mismatched"] == [] and b["sentence"] is None
    assert b["skipped"] == [{"key": ["b", 0, "q vs o", "q vs p"], "reason": "old_set"}]
    assert b["b_digest"] == spec.b_digest and b["a_digest"] == spec.a_digest and b["rule"] == {"family": "G"}
    assert "stage 3" in capsys.readouterr().out


def test_stage2_coverage_short_writes_its_sentence(tmp_path, monkeypatch, capsys):
    pattern = [True, False, False, False, False, True, False, False, False]
    spec, argv, calls = _s2_world(monkeypatch, tmp_path, pattern, n_pass=3, max_screened=12)
    assert S2.main(argv, spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads(Path(argv[-1]).read_text())["stage2"]
    assert b["outcome"] == "COVERAGE_SHORT" and b["n_star"] is None and len(b["screened"]) == 9 and b["lift"] == []
    assert "0.25" in b["sentence"] and "(2/9)" in b["sentence"]
    assert "COVERAGE_SHORT" in capsys.readouterr().out


def test_stage2_naive_self_check_mismatch_exits_5_before_screening(tmp_path, monkeypatch, capsys):
    spec, argv, calls = _s2_world(monkeypatch, tmp_path, [True] * 8, selfcheck_ok=False, n_pass=3)
    assert S2.main(argv, spec=spec, summary_spec=spec, require_root=False) == 5
    assert calls == [["Surf vs Chansey"]] and "stage2" not in json.loads(Path(argv[-1]).read_text())
    rep = json.loads(next((tmp_path / "results/m0d/l/run/runs").glob("*-stage2.json")).read_text())
    assert rep["selfcheck_naive"]["mismatched"] == ["fx"] and rep["status"] == "mismatch_naive"


def test_stage2_refuses_other_new_pair_digests_before_the_pool(tmp_path, monkeypatch, capsys):
    spec, argv, calls = _s2_world(monkeypatch, tmp_path, [True] * 8, n_pass=3)

    def bad(pops, spec_, declared):
        raise ValueError("new (b) list digest abc differs from spec.b_digest")
    monkeypatch.setattr(S2, "new_set", bad)
    monkeypatch.setattr(S2, "FlyPool", lambda *a, **k: pytest.fail("the pool started before the digest refusal"))
    assert S2.main(argv, spec=spec, summary_spec=spec, require_root=False) == 2
    assert "digest" in capsys.readouterr().err and calls == []


def test_stage2_naive_mismatch_names_each_field():
    import numpy as np
    row = dict(fx=np.array([0.0, 0.5]), fy=np.array([0.25, 0.0]), pre={"MBON13": [[1, 2]]})
    got = dict(fx_idx=[1], fx_val=[0.5], fy_idx=[0], fy_val=[0.25], pre={"MBON13": [[1, 2]]})
    assert S2.naive_mismatch(got, row, 2) == []
    assert S2.naive_mismatch(dict(got, fy_val=[0.26], pre={"MBON13": [[1, 3]]}), row, 2) == ["fy", "pre"]


# ---------------------------------------------------------------- stage 3
def _s3_world(monkeypatch, tmp_path, testable_b=12, fa_ok=3, reasons=(), n_lift_testable=2, **spec_kw):
    """21 passed (b) pairs on new turns 0-20 (first `testable_b` testable), 20 (a) pairs of turns 0-7 (first `fa_ok`
    testable and naive), 10 lift pairs (first `n_lift_testable` testable). Oracle and row stats are stand-ins; the
    turn records for the diversity are the real new_turns."""
    import dataclasses
    from flymon.brain.l_spec import SPEC
    spec = dataclasses.replace(SPEC, m0d_path=str(ROOT / SPEC.m0d_path), **spec_kw)
    argv = _keys(monkeypatch, S3, tmp_path)
    npass = spec.n_pass
    b = [dict(axis="b", turn=i, x=f"Mv{i % 5} vs o", y=f"Mv{i % 5} vs p", odor_x={}, odor_y={}) for i in range(npass)]
    lift = [dict(axis="b", turn=30 + i, x=f"L{i} vs o", y=f"L{i} vs p", odor_x={}, odor_y={}) for i in range(10)]
    a = [dict(axis="a", turn=i // 3, x=f"A{i}", y=f"B{i}", odor_x={}, odor_y={}) for i in range(20)]
    new = dict(b=b + lift, a=a, skipped=[])
    key = lambda p: [p["axis"], p["turn"], p["x"], p["y"]]
    seen = []

    class M4:
        def __init__(self, pool, spec4, pairs, pools, cache, h3):
            self.pairs, self.params_seen = pairs, []
            seen.append([key(p) for p in pairs])

        def oracle(self, params, readout, z):
            return [dict(axis=p["axis"], turn=p["turn"], x=p["x"], y=p["y"]) for p in self.pairs]

    def rows_stats(rows, z, expected, spec4):
        st = {}
        for r in rows:
            k = (r["axis"], r["turn"], r["x"], r["y"])
            if r["axis"] == "a":
                i = int(r["x"][1:]); st[k] = dict(d_pre=0.0 if i < fa_ok else 1.0, r=3, p=-3, m=3, testable=i < fa_ok)
            elif r["x"].startswith("L"):
                st[k] = dict(d_pre=0.0, r=3, p=-3, m=3, testable=int(r["x"][1:].split()[0]) < n_lift_testable)
            else:
                st[k] = dict(d_pre=0.0, r=3, p=-3, m=3, testable=r["turn"] < testable_b)
        return dict(reasons=list(reasons) if any(r["axis"] == "a" for r in rows) else [], stats=st, pairs=[])

    monkeypatch.setattr(S3.j_store, "load_c3", lambda summary, block, name: ("C3", None, None))
    monkeypatch.setattr(S3, "params_json", lambda p: {"p": p})
    monkeypatch.setattr(S3, "world", lambda npz: (None, _Pops()))
    monkeypatch.setattr(S3, "new_set", lambda pops, spec_, declared: new)
    monkeypatch.setattr(S3, "FlyPool", _Pool)
    monkeypatch.setattr(S3, "H4Measurer", M4)
    monkeypatch.setattr(S3, "pair_rows_stats", rows_stats)
    rule = {"family": "f2", "op": ">", "t": 13.5}
    doc = {"stage0": dict(KEYS), "oc": dict(KEYS), "stage1": dict(KEYS, gate={"outcome": "SCREEN_GO",
                                                                            "final": {"rule": rule}}),
           "stage2": dict(KEYS, run_id="r3", outcome="SCREENED", rule=rule, b_digest=spec.b_digest,
                          a_digest=spec.a_digest, coverage=0.3, passed=[key(p) for p in b],
                          lift=[key(p) for p in lift])}
    return spec, argv + ["--summary", str(_summary(tmp_path, doc))], seen


def test_stage3_selected_reads_21_plus_a_records_the_lift_and_the_diversity(tmp_path, monkeypatch, capsys):
    spec, argv, seen = _s3_world(monkeypatch, tmp_path)
    assert S3.main(argv, spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads(Path(argv[-1]).read_text())["stage3"]
    assert len(seen[0]) == 21 + 20 + 10                                          # one oracle call: judged + lift
    assert b["aggregate"]["n_b"] == 21 and b["aggregate"]["n_a"] == 20 and b["aggregate"]["testable_b"] == 12
    assert b["reading"]["band"] == "SELECTED" and b["reading"]["F_a"] == 3
    assert b["lift"]["n"] == 10 and b["lift"]["k"] == 2
    assert b["diversity"]["n_moves"] == 5 and b["diversity"]["n_me"] >= 1 and b["diversity"]["n_opp_types"] >= 1
    assert "12/21" in b["sentence"] and "2/10" in b["sentence"] and "G ∧ f2 > 13.5" in b["sentence"]
    assert [x["claim"] for x in b["attempts"]] == ["I", "J", "K", "L"] and b["rule"]["family"] == "f2"
    assert b["c3"] == {"p": "C3"} and b["stage2_run_id"] == "r3"


@pytest.mark.parametrize("tb,band", [(4, "B_Tb"), (9, "B_NO_CONCLUSION")])
def test_stage3_bands_below_the_bar(tb, band, tmp_path, monkeypatch):
    spec, argv, seen = _s3_world(monkeypatch, tmp_path, testable_b=tb)
    assert S3.main(argv, spec=spec, summary_spec=spec, require_root=False) == 0
    b = json.loads(Path(argv[-1]).read_text())["stage3"]
    assert b["reading"]["band"] == band and f"{tb}/21" in b["sentence"]


def test_stage3_invalid_rows_exit_3_without_the_block(tmp_path, monkeypatch, capsys):
    spec, argv, seen = _s3_world(monkeypatch, tmp_path, reasons=("1 pairs with an undefined d'",))
    assert S3.main(argv, spec=spec, summary_spec=spec, require_root=False) == 3
    assert "stage3" not in json.loads(Path(argv[-1]).read_text())
    rep = json.loads(next((tmp_path / "results/m0d/l/run/runs").glob("*-stage3.json")).read_text())
    assert rep["reading"]["outcome"] == "INVALID" and rep["reading"]["reasons"] == ["1 pairs with an undefined d'"]


def test_stage3_refuses_a_stage2_whose_pairs_are_not_in_the_new_set(tmp_path, monkeypatch, capsys):
    spec, argv, seen = _s3_world(monkeypatch, tmp_path)
    doc = json.loads(Path(argv[-1]).read_text())
    doc["stage2"]["passed"][0] = ["b", 0, "X vs o", "X vs p"]
    Path(argv[-1]).write_text(json.dumps(doc))
    assert S3.main(argv, spec=spec, summary_spec=spec, require_root=False) == 2
    assert "X vs o" in capsys.readouterr().err and seen == []


def test_stage3_refuses_another_rule_or_digest_than_recorded(tmp_path, monkeypatch, capsys):
    spec, argv, seen = _s3_world(monkeypatch, tmp_path)
    doc = json.loads(Path(argv[-1]).read_text())
    Path(argv[-1]).write_text(json.dumps(dict(doc, stage2=dict(doc["stage2"], b_digest="0" * 64))))
    assert S3.main(argv, spec=spec, summary_spec=spec, require_root=False) == 2 and "digest" in capsys.readouterr().err
    Path(argv[-1]).write_text(json.dumps(dict(doc, stage2=dict(doc["stage2"], rule={"family": "G"}))))
    assert S3.main(argv, spec=spec, summary_spec=spec, require_root=False) == 2 and "rule" in capsys.readouterr().err
    assert seen == []


def test_stage3_smoke_writes_no_block(tmp_path, monkeypatch):
    from flymon.brain.l_spec import smoke
    spec, argv, seen = _s3_world(monkeypatch, tmp_path, testable_b=1, n_pass=2, n_lift=1, a_turns=1)
    monkeypatch.setattr(S3, "smoke", lambda s: spec)
    sm = tmp_path / "results/m0d/l/smoke/l_screen.json"; sm.parent.mkdir(parents=True)
    doc = json.loads(Path(argv[-1]).read_text())
    doc["stage2"]["passed"] = doc["stage2"]["passed"][:2]; doc["stage2"]["lift"] = doc["stage2"]["lift"][:1]
    sm.write_text(json.dumps(doc))
    assert S3.main(argv[:-1] + [str(sm), "--smoke"], summary_spec=smoke(spec), require_root=False) == 0
    assert "stage3" not in json.loads(sm.read_text())
    rep = json.loads(next((tmp_path / "results/m0d/l/smoke/runs").glob("*-stage3.json")).read_text())
    assert rep["reading"]["band"] is None and rep["sentence"] is None
