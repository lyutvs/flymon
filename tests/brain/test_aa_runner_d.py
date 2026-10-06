"""aa_runner (D): records (S31 / out, record estimators, per-pair records, floor sensitivity + coverage, plasticity-off
control, records cap, G.6 recompute record with its own measured 2 h cap, TARGET_LOW skip, per-row resume) and the
defect procedure (defect → reseal → recompute from the cache, `invalid`, both archived, env rebased on aa_files only)."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import aa_estimate, aa_rules, w_oc, y_oc
from flymon.brain import aa_runner as AR
from flymon.brain.r_pairs import row_key
from tests.brain.aa_world import LENIENT_C, MAIN, World, doc

LKEYS = [row_key(MAIN[c]) for c in LENIENT_C]
GATES = aa_estimate.GATES
ITEMS = ["s1", "pairs", "S31", "out", "phi", "sens", "noplast", "g6"]


def _world(tmp_path, monkeypatch, **kw):
    w = World(tmp_path, monkeypatch)
    w.s = dataclasses.replace(w.s, workers=1, **kw)                    # cell stages in-process (speed)
    return w


def _theta():
    pil = w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, base=(40.0, 90.0), sd=4.0, corr=0.8,
                               learn=(20.0, 10.0))
    return y_oc.fit_y(pil, [f"a|{i}|O{i % 4}|y{i}" for i in range(6)], np.eye(4))


def _with_theta(w):
    th = _theta()
    w.ctx["y_theta"] = lambda where: dict(why=[], theta=th, theta_w=None, r=np.eye(4), keys=[], z=dict(w.z))
    return th


def _fake_g6(monkeypatch, hetero_fail=0):
    calls = dict(row=[], het=[], fail=hetero_fail)

    def row(theta, z, tgt, costs, n, s, ys=None, cell=None):
        calls["row"].append(round(tgt, 9))
        return dict(status="ok", target=tgt, false=[[0.0, 0.25]], fill_bad=[False, True], n_pass=7,
                    counts={"4-8|0.5": 7}, first=dict(p_set=0.5), designs=[dict(x=1)], n_pass_base_only=8,
                    base_minus_all=1, target_design_power={"4": 0.9})

    def het(theta, z, mu, tau, ref, costs, n, s, ys=None):
        if calls["fail"]:
            calls["fail"] -= 1
            raise RuntimeError("killed")
        calls["het"].append(dict(mu=mu, tau=tau, ref=ref))
        return dict(status="ok", mu=mu, tau=tau, n_pass=3, first=None, odd_k_low_extra=True)
    monkeypatch.setattr(aa_estimate, "g6_row", row)
    monkeypatch.setattr(aa_estimate, "g6_hetero", het)
    return calls


def _inputs(path):
    return json.loads(Path(path).read_text())["inputs"]


def _new_log(w, n=1235):
    Path(w.s.tests_log).write_text(f"{n} passed in 601.00s\nexit 0\n")


def _note(tmp_path, **drop):
    p = tmp_path / "note.json"
    d = dict(symptom="추정 재계산 불일치", clause="AA.7 6", cause="부동소수 순서", affected=["estimate"])
    for k in drop:
        d.pop(k)
    p.write_text(json.dumps(d))
    return str(p)


# ================================================================ records
def test_records_pass_full(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    for c in LENIENT_C[:2]:                                           # two (b) pairs fail: out has 2
        w.model.offset[row_key(MAIN[c])] = 40.0
    w.chain("estimate")
    core0 = list(doc()["budget"]["ledger"])
    j0 = w.pool.jobs
    out = w.runner().run("records")
    assert out["outcome"] == "PASS" and out["nulls"] == [] and out["order"] == "7"
    assert out["primary_block"] == "estimate" and out["coverage_block"] == "coverage"
    sc = doc()["screen"]["pairs"]
    s1 = [p["key"] for p in sc if p["passed"]]
    assert len(s1) == 29
    assert out["sets"]["S31"]["k"] == 31 and out["sets"]["out"]["k"] == 31 - len(s1) == 2
    assert out["sets"]["out"]["plan"]["primary"] == "fly"
    assert out["sets"]["S31"]["coverage"] == doc()["coverage"]["by_set"]["S31"]
    r = out["s1_records"]
    assert r["k"] == 29 and set(r["ci"]) == {"two_stage", "fly", "pair"} and "_reps" not in r
    assert set(r["raw"]["ci"]) == set(aa_estimate.RAW) and set(r["drop"]) == set(GATES) and set(r["dl"]) == set(GATES)
    for p in sc:
        x = out["pairs"][p["key"]]
        assert x["naive_d"] == p["naive_d"] and x["passed"] == p["passed"] and x["x_odour"] == p["x_odour"]
        assert len(x["gates"]["punish_assoc"]["fly"]) == 8 and x["gates"]["reward_assoc"]["t_ci"] is not None
        assert "phi_R" in x["floors"] and set(x["ci"]) == set(aa_estimate.COLS)
        for k in AR.PILOT_ITEMS:
            assert k in x
        assert "q_sat" not in x and "naive_median" not in x
    assert out["phi_quantiles"]["qs"] == list(w.s.floor_qs) and len(out["phi_quantiles"]["phi_R"]) == 5
    np_ = out["noplast"]
    assert np_["r1_equals_pre"] is True and np_["mismatches"] == [] and np_["pairs"] == s1[:2]
    assert np_["source"] == "S1" and np_["n_units"] == 4 and w.pool.jobs - j0 == 4
    det = json.loads(Path(w.s.raw_dir, "records.json").read_text())
    assert len(det["manifest"]) == 4
    for m in det["manifest"]:
        ins = _inputs(m["cache_file"])
        assert ins["brain"] == "noplast" and ins["plastic"] is False and ins["block"] == "records"
        assert ins["fly"] in (0, 1) and ins["pair"] in s1[:2]
        c = MAIN[[row_key(r) for r in MAIN].index(ins["pair"])]["c"]
        assert ins["phases"][0][2] == 64_000_000 + c * 40_000 and ins["phases"][0][0] == "PAM08"
    g6 = out["g6"]
    for row in g6["rows"]:
        if row["status"] == aa_estimate.TARGET_LOW:
            continue
        assert row["status"] == ("Y θ̂ 없음(시험 세계)" if row["theta"] == "Y" else "θ_W 없음"), row
        assert "n_pass" not in row
    assert len(g6["rows"]) == 12 and [(r["row"], r["scale"]) for r in g6["rows"][:6]] == \
        [(r, s) for s in ("hedges", "raw") for r in ("point", "lo", "hi")]
    assert g6["capped"] is False and len(g6["disclosures"]) == len(AR.G6_DISCLOSURES)
    d = doc()
    assert d["budget"]["ledger"] == core0
    assert len(d["budget"]["records_ledger"]) == 1 and d["budget"]["records_ledger"][0]["stage"] == "records"
    assert len(out["archive"]) == 5 and Path(w.s.archive_root, "records").exists()


def test_records_floor_sensitivity(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    w = _world(tmp_path / "a", monkeypatch)
    floored = set(LKEYS[::2])
    for k in floored:                                                 # PPL105 drives R2 MBON13(X) to 0
        w.model.effects[k] = (30.0, 60.0)
    w.chain("records")
    out = doc()["records"]
    want = [k for k in LKEYS if out["pairs"][k]["floors"]["phi_R"] < 0.1 and out["pairs"][k]["floors"]["phi_P"] < 0.1]
    assert set(want) == set(LKEYS) - floored
    sens = out["sets"]["sens"]
    assert sens["keys"] == want and sens["flow"] == "sens_floor" and sens["k"] == len(want)
    ms = AR.Runner._cov_methods(sens["plan"])
    assert set(sens["coverage"]) == set(ms)
    sizes = tuple(sens["sizes"])
    one = aa_estimate.coverage_cell(sizes, sens["plan"]["primary"], "sens", 0, w.s)
    assert sens["coverage"][sens["plan"]["primary"]]["cells"][0]["share"] == one
    (tmp_path / "b").mkdir()
    w2 = _world(tmp_path / "b", monkeypatch)
    for k in LKEYS:
        w2.model.effects[k] = (30.0, 60.0)
    w2.chain("records")
    out2 = doc()["records"]
    assert out2["outcome"] == "PASS" and out2["sets"]["sens"] is None
    assert dict(item="sens", reason="k = 0") in out2["nulls"]


def test_records_cap_nulls_not_stop(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, records_cap_h=0.0)
    w.chain("estimate")
    j0 = w.pool.jobs
    out = w.runner().run("records")
    assert out["outcome"] == "PASS"
    assert out["nulls"] == [dict(item=i, reason=aa_rules.RECORDS_REASON) for i in ITEMS]
    assert out["s1_records"] is None and out["pairs"] is None and out["g6"] is None and out["noplast"] is None
    assert out["sets"] == dict(S31=None, out=None, sens=None) and w.pool.jobs == j0


def test_g6_cap(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch, g6_cap_h=0.0)
    _with_theta(w)
    calls = _fake_g6(monkeypatch)
    w.chain("records")
    g6 = doc()["records"]["g6"]
    assert calls["row"] == [] and calls["het"] == []
    live = [r for r in g6["rows"] if r["status"] != aa_estimate.TARGET_LOW]
    assert live and all(r["status"] == aa_rules.G6_REASON and "n_pass" not in r for r in live)
    assert all(h["status"] in (aa_rules.G6_REASON, aa_estimate.TARGET_LOW) for h in g6["hetero"])
    assert g6["capped"] is True and g6["cap_reason"] == aa_rules.G6_REASON
    nul = doc()["records"]["nulls"]
    assert len([x for x in nul if x["reason"] == aa_rules.G6_REASON]) == len(live)


def test_g6_pool_cap_terminates_running_rows(tmp_path, monkeypatch):
    """Spawn pool, the real g6_row at Y's full precheck size: the measured cap (2 s) terminates the pool."""
    w = _world(tmp_path, monkeypatch, g6_cap_h=2.0 / 3600)
    _with_theta(w)
    w.chain("estimate")
    r = w.runner()
    r.workers = 2
    out = r.run("records")
    g6 = out["g6"]
    live = [x for x in g6["rows"] if x["status"] != aa_estimate.TARGET_LOW]
    assert live and all(x["status"] == aa_rules.G6_REASON for x in live)
    assert g6["capped"] is True and 2.0 <= g6["seconds"] < 60
    assert not list(Path(w.s.progress_dir, "records").glob("g6_*.json"))


@pytest.mark.parametrize("k", [29, 1])
def test_g6_inputs_from_primary_resample(tmp_path, monkeypatch, k):
    w = _world(tmp_path, monkeypatch)
    keep = set(LKEYS[2:]) if k == 29 else {LKEYS[11]}
    for x in LKEYS:
        if x not in keep:
            w.model.offset[x] = 40.0
    w.chain("records")
    d = doc()
    g6 = d["records"]["g6"]
    data = w.runner()._learn_data(d)
    S1 = [p for p in d["screen"]["pairs"] if p["passed"]]
    keys = [p["key"] for p in S1]
    assert len(keys) == k
    arrs = {x: aa_estimate.pair_arrays(data[x], w.z, w.s) for x in keys}
    if k > 1:
        est = aa_estimate.estimate_set(arrs, keys, "S1", w.s)
        mog = aa_estimate.min_of_gates([est["point"][g] for g in GATES], est["_reps"][:, :4], w.s)
        assert g6["resample"].startswith("S1 two_stage")
    else:
        reps = aa_estimate.pair_ci(arrs[keys[0]], S1[0]["c"], w.s)["reps"][:, :4]
        mog = aa_estimate.min_of_gates(arrs[keys[0]]["w"].mean(0)[:4], reps, w.s)
        assert g6["theta_status"]["AA"] == "k < 2"
        assert all(r["status"] in ("k < 2", aa_estimate.TARGET_LOW) for r in g6["rows"] if r["theta"] == "AA")
        assert all(h["status"] in ("k < 2", aa_estimate.TARGET_LOW) for h in g6["hetero"] if h["theta"] == "AA")
        assert d["records"]["noplast"]["source"] == "S31"
        assert d["records"]["noplast"]["pairs"] == LKEYS[:2]
    assert g6["mog"] == AR._js(mog)
    assert g6["inputs"] == AR._js(aa_estimate.g6_inputs(mog, w.s))
    assert [r["target"] for r in g6["rows"][:6]] == [x["target"] for x in g6["inputs"]]


def test_g6_target_le_false_not_computed(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    for x in LKEYS:
        w.model.effects[x] = (0.0, 0.0)
    _with_theta(w)
    calls = _fake_g6(monkeypatch)
    w.chain("records")
    g6 = doc()["records"]["g6"]
    low = [r for r in g6["rows"] if round(r["target"] - 0.5, 9) <= 0]
    assert low and all(r["status"] == aa_estimate.TARGET_LOW and "n_pass" not in r for r in low)
    assert all(t > 0.5 for t in calls["row"])
    hi = [r for r in g6["rows"] if r not in low]
    assert all(r["status"] == "ok" and r["n_pass"] == 7 for r in hi)
    assert len(calls["row"]) == len(hi)
    for h in g6["hetero"]:
        pt = [r for r in g6["rows"] if (r["theta"], r["row"], r["scale"]) == (h["theta"], "point", h["scale"])][0]
        if pt["status"] == aa_estimate.TARGET_LOW:
            assert h["status"] == aa_estimate.TARGET_LOW and "n_pass" not in h


def test_g6_rows_resume_and_hetero_ref(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    _with_theta(w)
    calls = _fake_g6(monkeypatch, hetero_fail=1)
    w.chain("estimate")
    with pytest.raises(RuntimeError):
        w.runner().run("records")
    assert "records" not in doc()
    n_rows = len(calls["row"])
    assert n_rows == 12                                               # every row computed (and kept) before hetero
    out = w.runner().run("records")
    assert len(calls["row"]) == n_rows                                # rows read back from their cells
    g6 = out["g6"]
    assert all(r["status"] == "ok" and r["n_pass"] == 7 for r in g6["rows"]) and "designs" not in g6["rows"][0]
    d = doc()
    data = w.runner()._learn_data(d)
    keys = [p["key"] for p in d["screen"]["pairs"] if p["passed"]]
    est = aa_estimate.estimate_set({x: aa_estimate.pair_arrays(data[x], w.z, w.s) for x in keys}, keys, "S1", w.s)
    tau = est["dl"][g6["mog"]["gate"]]["tau"]
    assert len(calls["het"]) == len(g6["hetero"]) == 4
    for h in g6["hetero"]:
        pt = [r for r in g6["rows"] if (r["theta"], r["row"], r["scale"]) == (h["theta"], "point", h["scale"])][0]
        f = aa_estimate.hedges_j(8) if h["scale"] == "hedges" else 1.0
        assert h["status"] == "ok" and h["mu"] == pt["target"] and h["tau"] == pytest.approx(f * tau, abs=1e-12)
        assert h["odd_k_low_extra"] is True
    assert all(c["ref"] == dict(false=[[0.0, 0.25]], fill_bad=[False, True]) for c in calls["het"])
    det = json.loads(Path(w.s.raw_dir, "records.json").read_text())
    assert det["g6"]["g6_row_Y_point_hedges"]["designs"] == [dict(x=1)]


# ================================================================ defect procedure
def test_defect_procedure(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("estimate")
    orig = doc()["estimate"]
    with pytest.raises(SystemExit) as e:
        w.runner().defect(_note(tmp_path, cause=True))
    assert e.value.code == 2
    w.runner().defect(_note(tmp_path))
    d1 = doc()["defect_1"]
    assert (d1["symptom"], d1["clause"], d1["cause"], d1["affected"]) == ("추정 재계산 불일치", "AA.7 6", "부동소수 순서",
                                                                          ["estimate"])
    assert len(doc()["budget"]["ledger"]) == len([b for b in AR.AA.stages[:8]])
    real = AR.seal_now
    monkeypatch.setattr(AR, "seal_now", lambda s=w.s: dict(real(s), key="1" * 64))
    with pytest.raises(SystemExit) as e:
        w.runner().run("records")
    assert e.value.code == AR.EXIT_SEAL
    with pytest.raises(SystemExit) as e:                             # no reseal yet
        w.runner().recompute("estimate")
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:                             # the tests log was not rerun after the patch
        w.runner().reseal(1)
    assert e.value.code == 2
    _new_log(w)
    w.runner().reseal(1)
    assert doc()["reseal_1"]["seal"]["key"] == "1" * 64 and doc()["reseal_1"]["defect"] == 1
    j0 = w.pool.jobs
    with pytest.raises(SystemExit):
        w.runner().recompute("learn")                                 # not a sealed stage
    out = w.runner().recompute("estimate")
    assert out["outcome"] == "PASS" and w.pool.jobs == j0
    d = doc()
    assert d["invalid"] == ["estimate"] and d["estimate_v1"]["replaces"] == "estimate"
    assert d["estimate_v1"]["defect"] == 1 and d["estimate_v1"]["seal_key"] == "1" * 64
    assert d["estimate_v1"]["primary"] == orig["primary"] and d["estimate"] == orig
    assert d["budget"]["ledger"][-1]["stage"] == "estimate_v1"
    arch = Path(w.s.archive_root)
    assert (arch / "estimate").exists() and (arch / "estimate_v1").exists()
    with pytest.raises(SystemExit):
        w.runner().recompute("estimate")                              # estimate_v1 exists
    rec = w.runner().run("records")
    assert rec["outcome"] == "PASS" and rec["primary_block"] == "estimate_v1"


def test_recompute_refuses_measurement(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("estimate")
    w.runner().defect(_note(tmp_path))
    _new_log(w)
    w.runner().reseal(1)
    man = json.loads(Path(w.s.raw_dir, "learn.json").read_text())["manifest"]
    Path(man[5]["cache_file"]).unlink()
    j0 = w.pool.jobs
    with pytest.raises(SystemExit) as e:
        w.runner().recompute("estimate")
    assert e.value.code == 2 and w.pool.jobs == j0
    assert "estimate_v1" not in doc() and "invalid" not in doc()


def test_recompute_records_cache_only(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("records")
    orig = doc()["records"]
    w.runner().defect(_note(tmp_path))
    _new_log(w)
    w.runner().reseal(1)
    j0 = w.pool.jobs
    out = w.runner().recompute("records")
    assert out["outcome"] == "PASS" and w.pool.jobs == j0
    assert out["noplast"] == orig["noplast"] and out["s1_records"] == orig["s1_records"]
    d = doc()
    assert d["invalid"] == ["records"] and [e["stage"] for e in d["budget"]["records_ledger"]] == ["records",
                                                                                                    "records_v1"]
    assert (Path(w.s.archive_root) / "records_v1").exists()
    noplast = json.loads(Path(w.s.raw_dir, "records.json").read_text())["manifest"]
    Path(noplast[0]["cache_file"]).unlink()
    w.runner().defect(_note(tmp_path))
    _new_log(w, 1236)
    w.runner().reseal(2)
    with pytest.raises(SystemExit) as e:                             # a noplast cache miss is never re-measured
        w.runner().recompute("records")
    assert e.value.code == 2 and w.pool.jobs == j0


def test_reseal_rebases_aa_files_env_only(tmp_path, monkeypatch):
    w = _world(tmp_path, monkeypatch)
    w.chain("estimate")
    w.runner().defect(_note(tmp_path))
    real_env, real_seal = AR.env_now, AR.seal_now
    patched = lambda: dict(real_env(), aa_files=dict(real_env()["aa_files"], **{"flymon/brain/aa_estimate.py": "f" * 64}))  # noqa: E731
    monkeypatch.setattr(AR, "seal_now", lambda s=w.s: dict(real_seal(s), key="2" * 64))
    monkeypatch.setattr(AR, "env_now", lambda: dict(patched(), uv_lock_sha256="0" * 64))
    _new_log(w)
    with pytest.raises(SystemExit) as e:                             # anything but aa_files still refuses
        w.runner().reseal(1)
    assert e.value.code == AR.EXIT_ENV
    monkeypatch.setattr(AR, "env_now", patched)
    w.runner().reseal(1)
    out = w.runner().run("records")                                   # pre-reseal blocks compared without aa_files
    assert out["outcome"] == "PASS"


def test_reseal_block_compared_in_full_and_changed_aa_files(tmp_path, monkeypatch):
    """The reseal block (and later ones) keep aa_files: after reseal_1 any further aa_* change refuses (EXIT_ENV);
    reseal_<n> records which aa_* files changed against the previous reseal (else seal_code)."""
    w = _world(tmp_path, monkeypatch)
    w.chain("estimate")
    w.runner().defect(_note(tmp_path))
    real_env, real_seal = AR.env_now, AR.seal_now
    est, rules = "flymon/brain/aa_estimate.py", "flymon/brain/aa_rules.py"

    def patched(**files):
        return lambda: dict(real_env(), aa_files=dict(real_env()["aa_files"], **files))
    monkeypatch.setattr(AR, "seal_now", lambda s=w.s: dict(real_seal(s), key="2" * 64))
    monkeypatch.setattr(AR, "env_now", patched(**{est: "f" * 64}))
    _new_log(w)
    w.runner().reseal(1)
    r1 = doc()["reseal_1"]
    assert r1["changed_aa_files"] == [est] and r1["changed_against"] == "seal_code"
    monkeypatch.setattr(AR, "env_now", patched(**{est: "f" * 64, rules: "e" * 64}))
    with pytest.raises(SystemExit) as e:                             # reseal_1 itself is compared with aa_files
        w.runner().run("records")
    assert e.value.code == AR.EXIT_ENV
    led = doc()["budget"]["ledger"][-1]
    assert led["stage"] == "records" and any(x.startswith("reseal_1 대비") for x in led["env_mismatch"])
    assert not any(x.startswith("estimate 대비") for x in led["env_mismatch"])
    w.runner().defect(_note(tmp_path))                               # the procedure again: reseal_2 rebases
    _new_log(w, 1236)
    w.runner().reseal(2)
    r2 = doc()["reseal_2"]
    assert r2["changed_aa_files"] == [rules] and r2["changed_against"] == "reseal_1"
    assert w.runner().run("records")["outcome"] == "PASS"


def test_run_refuses_stage_with_recomputed_block(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path, monkeypatch)
    w.chain("learn")
    w.runner().defect(_note(tmp_path))
    _new_log(w)
    w.runner().reseal(1)
    out = w.runner().recompute("estimate")                            # no estimate block: estimate_v1 replaces none
    assert out["outcome"] == "PASS" and doc()["estimate_v1"]["replaces"] is None and "estimate" not in doc()
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        w.runner().run("estimate")
    assert e.value.code == 2 and "estimate_v1 exists" in capsys.readouterr().err
    assert "estimate" not in doc()
    assert w.runner().run("records")["primary_block"] == "estimate_v1"


def test_g6_clock_includes_theta_aa_fit(tmp_path, monkeypatch):
    """θ_AA's fit_y runs on the G.6 clock: a fit longer than the cap leaves every computed row capped."""
    import time
    w = _world(tmp_path, monkeypatch, g6_cap_h=0.25 / 3600)
    _with_theta(w)
    calls = _fake_g6(monkeypatch)
    real = y_oc.fit_y

    def slow(*a, **k):
        time.sleep(0.5)
        return real(*a, **k)
    monkeypatch.setattr(y_oc, "fit_y", slow)
    w.chain("records")
    g6 = doc()["records"]["g6"]
    assert g6["theta_status"]["AA"] is None                           # θ_AA was fitted
    assert calls["row"] == [] and g6["capped"] is True and g6["cap_reason"] == aa_rules.G6_REASON
    assert g6["seconds"] >= 0.5
    live = [r for r in g6["rows"] if r["status"] != aa_estimate.TARGET_LOW]
    assert live and all(r["status"] == aa_rules.G6_REASON for r in live)


def test_noplast_mismatch_recorded_outcome_pass(tmp_path, monkeypatch):
    import copy
    w = _world(tmp_path, monkeypatch)
    real = AR.Runner._measure

    def bump(self, wm, U, block):
        got = real(self, wm, U, block)
        if block != "records":
            return got
        got = copy.deepcopy(got)
        st = got[0]["result"]["stages"][1]
        st["x"][0]["A"] = float(st["x"][0]["A"]) + 1.0
        return got
    monkeypatch.setattr(AR.Runner, "_measure", bump)
    w.chain("records")
    out = doc()["records"]
    np_ = out["noplast"]
    assert out["outcome"] == "PASS" and out["reasons"] == []
    assert np_["r1_equals_pre"] is False and len(np_["mismatches"]) == 1
    assert np_["mismatches"][0]["fly"] in (0, 1) and np_["mismatches"][0]["pair"] in np_["pairs"]
    assert "1차 추정은 바꾸지 않는다" in np_["note"]
