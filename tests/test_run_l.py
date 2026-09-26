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


def test_the_manifest_hashes_only_files_that_exist(tmp_path):
    """HASHED_FILES names later tasks' scripts up front; absent files are left out of the manifest (their absence
    still changes the key, since the key hashes the file map)."""
    npz = tmp_path / "x.npz"; npz.write_bytes(b"x")
    code, manifest = l_cli.code_keys(str(npz))
    assert all((ROOT / f).exists() for f in manifest["files"] if not f.startswith("npz:"))
    assert set(code["files"]) <= set(manifest["files"])


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
