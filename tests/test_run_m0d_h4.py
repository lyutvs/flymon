"""scripts/run_m0d_h4.py end to end on the synthetic connectome: real jobs on a FlyPool, H.3's cache for the reference
and rest, the resume, both guards on every written file, the refusals and when block "h4" may be written (spec H.4,
H.4a)."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.brain import h3_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_rules import mbon_type_stats
from flymon.brain.h3_spec import SPEC as H3, OdorSet, Window, make_odors
from flymon.brain.h4_measure import HASHED_FILES, MEASURE_FILES
from flymon.brain.h4_pairs import pairs_digest
from flymon.brain.h4_spec import SPEC

_spec = importlib.util.spec_from_file_location("run_m0d_h4", Path("scripts/run_m0d_h4.py").resolve())
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

CLEAN = dict(commit="t", dirty_hashed=[], dirty_other=[])
BASE = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05)
COMBOS = {"C0": Params(**BASE, kc_thresh=0.5), "C1": Params(**BASE, kc_thresh=0.55), "C3": Params(**BASE, kc_thresh=0.6)}
POOLS = {"A": ["MBON03"], "P": ["MBON01"]}
CORES = {"A": ["MBON03", "MBON04"], "P": ["MBON01", "MBON02"]}          # the synthetic PPL105 and PAM08 core types
PAIRS = [dict(axis="a", turn=0, x="X0", y="Y0", odor_x={"ORN_DM1": 1.0}, odor_y={"ORN_VA2": 1.0}),
         dict(axis="b", turn=2, x="X1", y="Y1", odor_x={"ORN_DM6": 1.0}, odor_y={"ORN_DM1": 0.5, "ORN_VA2": 0.5})]


def lenient_spec(npz, pairs=PAIRS):
    """Small windows and seed sets, thresholds wide enough that every combination reaches the oracle and a selection."""
    r = dataclasses.replace
    h3 = r(H3, connectome_sha256=hashlib.sha256(Path(npz).read_bytes()).hexdigest(), strength=3.0, design_k=2,
           reference=OdorSet("S{j:02d}", 3, 11, 1000, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=None),
           reference_window=Window(50.0, 100.0))
    return r(SPEC, h3=h3, react_med_delta_min=-1e9, react_zero_share_max=1.0, teach_seeds=(8, 9), teach_min_decreased=0,
             teach_trials=2, teach_present_ms=100.0, teach_gap_ms=50.0, teach_window=Window(50.0, 100.0),
             act_seeds=(500, 501), select_seeds=(600, 601), report_seeds=(608, 609), oracle_window=Window(50.0, 100.0),
             kc_window_ms=20, n_pairs_a=1, n_pairs_b=1, pairs_digest=pairs_digest(pairs), t_b_min=0.0, f_a_min=0)


def write_h3_block(npz, spec):
    """Block "h3" as the H.3 runner leaves it: every combination adopted, with the guard statistics of H.3's cache."""
    conn = Connectome.load(npz); pops = Populations.from_connectome(conn)
    odors = make_odors(pops, spec.h3.reference)
    seeds = [int(s) for o in odors for s in o["seeds"]]
    code = h3_store.code_key(npz)
    combos = {}
    with FlyPool(npz, Params(), [{}, {}], workers=2) as pool:
        m = PoolMeasurer(pool, spec.h3, {"reference": odors}, None, h3_store.MeasureCache("results/m0d/h3/cache", code, "h3"),
                         None)
        for name, p in COMBOS.items():
            ref, rest = m.reference(p), {r["seed"]: r for r in m.rest(p, seeds)}
            types = {k: {"types": {t: mbon_type_stats(ref, rest, t, 5.0, 0.25) for t in CORES[k]}} for k in ("A", "P")}
            combos[name] = dict(status="adopted", adopted=dict(params=dataclasses.asdict(p)),
                                cells=[dict(status="adopted", stage3=dict(guard=types))])
    Path("results/summary").mkdir(parents=True, exist_ok=True)
    Path("results/summary/m0d.json").write_text(json.dumps({"h3": dict(combos=combos, measure_key=code["key"],
                                                                       run_id="h3-test", pools=CORES)}))


@pytest.fixture
def workdir(synthetic_npz, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "git_state", lambda **kw: dict(CLEAN))
    seen = []
    old, mod = h3_store.refuse_old_engine_output, h3_store.refuse_modified_engine_output
    monkeypatch.setattr(h3_store, "refuse_old_engine_output", lambda out, k: (seen.append(("old", out)), old(out, k)))
    monkeypatch.setattr(h3_store, "refuse_modified_engine_output", lambda out, p: (seen.append(("mod", out)), mod(out, p)))
    spec = lenient_spec(synthetic_npz)
    write_h3_block(str(synthetic_npz), spec)
    seen.clear()

    def main(*extra, spec_=spec, summary_spec=spec, pairs=PAIRS, pools=POOLS):
        return run.main(["--npz", str(synthetic_npz), "--workers", "2", *extra], spec=spec_, summary_spec=summary_spec,
                        require_root=False, pairs=pairs, pools=pools)
    return tmp_path, main, seen, spec


def _reports(root):
    return sorted((root / "results/m0d/h4/runs").glob("*.json"))


def test_a_complete_run_writes_block_h4_through_both_guards_and_resumes_from_the_cache(workdir):
    root, main, seen, _ = workdir
    assert main() == 0
    s = json.loads((root / "results/summary/m0d.json").read_text())
    assert set(s) == {"h3", "h4"}
    h4 = s["h4"]["h4"]
    assert h4["outcome"] == "SELECTED" and h4["selection"]["winner"] == "C0"
    assert all(c["status"] == "readout_selected" and c["readout"] == {"A": "MBON03", "P": "MBON01"}
               and set(c["records"]) == {"naive_floor", "single_type", "report_halves"} for c in h4["combos"].values())
    assert "winner_below_bar" in h4["selection"]
    assert s["h4"]["notes"] == list(SPEC.notes) and s["h4"]["h3_run_id"] == "h3-test"
    md = _reports(root)[0].with_suffix(".md").read_text()                # reading 16: the teach records, per type
    for name, c in h4["combos"].items():
        for t in POOLS["A"] + POOLS["P"]:
            tc = c["teach"][t]
            assert (f"- {name} {t}: taught odour {tc['odour']}, order {tc['order']}, arm {tc['arm']}, decreased "
                    f"{tc['n_decreased']}/2, teachable {tc['teachable']}, untaught decreased "
                    f"{tc['untaught_n_decreased']}") in md, (name, t)
    written = sorted(str(p.relative_to(root)) for p in (root / "results/m0d/h4").rglob("*") if p.is_file())
    for f in written + ["results/summary/m0d.json"]:
        assert ("old", f) in seen and ("mod", f) in seen, f
    first = json.loads(_reports(root)[0].read_text())
    assert first["cache"]["hits"] >= 2 * len(COMBOS)                    # reference and rest came from H.3's cache
    assert main() == 0
    second = json.loads(_reports(root)[-1].read_text())
    assert second["cache"]["misses"] == 0
    assert h3_store.canonical(first["h4"]) == h3_store.canonical(second["h4"])


@pytest.mark.parametrize("blocker", ["smoke", "pairs", "dirty", "spec", "invalid", "incomplete"])
def test_each_blocker_alone_keeps_block_h4_unwritten(workdir, monkeypatch, blocker):
    root, main, _, spec = workdir
    extra = {"smoke": ["--smoke", "--out", "results/m0d/h4-test"], "pairs": ["--pairs", "1"],
             "dirty": ["--allow-dirty"]}.get(blocker, [])
    kw = {}
    if blocker == "dirty":
        monkeypatch.setattr(run, "git_state", lambda **k: dict(CLEAN, dirty_hashed=["flymon/brain/h4_rules.py"]))
    if blocker == "spec":
        kw = dict(summary_spec=dataclasses.replace(spec, tie_pairs=3))
    if blocker == "invalid":
        odd = [PAIRS[0], dict(PAIRS[1], turn=3)]
        s2 = dataclasses.replace(spec, pairs_digest=pairs_digest(odd))
        kw = dict(spec_=s2, summary_spec=s2, pairs=odd)
    if blocker == "incomplete":
        kw = dict(pools=CORES)                            # both types of a pool pass the lenient rules: the run stops
    assert main(*extra, **kw) == 0
    assert "h4" not in json.loads((root / "results/summary/m0d.json").read_text())


def _edit_h3(root, fn):
    path = root / "results/summary/m0d.json"
    good = path.read_text()
    d = json.loads(good); fn(d["h3"]); path.write_text(json.dumps(d))
    return lambda: path.write_text(good)


@pytest.mark.parametrize("case", ["connectome", "digest", "pools", "measure_key", "not_adopted", "malformed", "no_block",
                                  "dirty", "root", "pairs_zero", "pairs_negative"])
def test_every_refusal_exits_2_before_measuring_or_writing(workdir, monkeypatch, synthetic_npz, case):
    root, main, _, spec = workdir
    kw, restore, extra = {}, lambda: None, []
    h3_cache = lambda: sorted(str(p.relative_to(root)) for p in (root / "results/m0d/h3/cache").rglob("*"))
    cache_before = h3_cache()
    assert cache_before                                                  # the fixture filled H.3's cache
    if case == "connectome":
        bad = dataclasses.replace(spec, h3=dataclasses.replace(spec.h3, connectome_sha256="0" * 64))
        kw = dict(spec_=bad, summary_spec=bad)
    if case == "digest":
        kw = dict(spec_=dataclasses.replace(spec, pairs_digest="0" * 64))
    if case == "pools":
        restore = _edit_h3(root, lambda b: b.__setitem__("pools", POOLS))
    if case == "measure_key":
        restore = _edit_h3(root, lambda b: b.__setitem__("measure_key", "0" * 64))
    if case == "not_adopted":
        restore = _edit_h3(root, lambda b: b["combos"]["C1"].__setitem__("status", "dropped"))
    if case == "malformed":
        restore = _edit_h3(root, lambda b: b["combos"]["C3"].pop("cells"))
    if case == "no_block":
        (root / "results/summary/m0d.json").write_text("{}")
    if case == "dirty":
        monkeypatch.setattr(run, "git_state", lambda **k: dict(CLEAN, dirty_hashed=["flymon/brain/h4_rules.py"]))
    if case in ("pairs_zero", "pairs_negative"):                          # 0 would run every pair, -1 drop the last
        extra = ["--pairs", "0" if case == "pairs_zero" else "-1"]
    summary_before = (root / "results/summary/m0d.json").read_bytes()
    if case == "root":
        assert run.main(["--npz", str(synthetic_npz)], spec=spec) == 2                  # not at the repository root
    else:
        assert main(*extra, **kw) == 2
    assert (root / "results/summary/m0d.json").read_bytes() == summary_before
    assert h3_cache() == cache_before
    restore()
    assert not (root / "results/m0d/h4").exists()


def test_adopted_restores_a_missing_c3_threshold_file_from_the_committed_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = b"thresholds"
    sha = hashlib.sha256(data).hexdigest()
    Path("results/summary").mkdir(parents=True)
    Path(run.C3_COPY).write_bytes(data)
    p = Params(kc_thresh_mode="homeostatic", kc_thresh_file="results/m0d/h3/thresholds/theta-x.npz", kc_thresh_sha256=sha)
    guard = {"A": {"types": {"MBON13": {"median_delta": 1.0, "zero_share": 0.0}}}, "P": {"types": {}}}
    block = {"combos": {n: dict(status="adopted", adopted=dict(params=json.loads(h3_store.canonical(p))),
                                cells=[dict(status="adopted", stage3=dict(guard=guard))]) for n in SPEC.combos}}
    params, guards = run.adopted({"h3": block}, SPEC)
    assert params["C3"] == p and Path(p.kc_thresh_file).read_bytes() == data
    assert guards["C1"] == {"MBON13": {"median_delta": 1.0, "zero_share": 0.0}}
    Path(p.kc_thresh_file).write_bytes(b"other")
    with pytest.raises(ValueError, match="sha256 differs"):
        run.adopted({"h3": block}, SPEC)
    Path(p.kc_thresh_file).write_bytes(data)
    block["combos"]["C1"]["status"] = "dropped"
    with pytest.raises(ValueError, match="not adopted"):
        run.adopted({"h3": block}, SPEC)
    block["combos"]["C1"]["status"] = "adopted"
    Path(p.kc_thresh_file).unlink(); Path(run.C3_COPY).unlink()
    with pytest.raises(ValueError, match="is missing"):                                  # no copy: a refusal, not a traceback
        run.adopted({"h3": block}, SPEC)
    del block["combos"]["C0"]["cells"]
    with pytest.raises(ValueError, match="malformed"):
        run.adopted({"h3": block}, SPEC)


def test_the_smoke_spec_shrinks_the_samples_but_keeps_the_rules():
    s = run.smoke_spec(SPEC)
    assert len(s.teach_seeds) < len(SPEC.teach_seeds) and len(s.report_seeds) < len(SPEC.report_seeds)
    assert (s.t_b_min, s.f_a_min, s.tie_pairs, s.pairs_digest) == (SPEC.t_b_min, SPEC.f_a_min, SPEC.tie_pairs,
                                                                  SPEC.pairs_digest)


def test_the_key_and_manifest_files_exist_and_the_rules_stay_outside_the_key():
    for f in HASHED_FILES:
        assert (h3_store.ROOT / f).exists(), f
    assert set(MEASURE_FILES) <= set(HASHED_FILES) and len(set(HASHED_FILES)) == len(HASHED_FILES)
    for f in ("flymon/brain/h4_rules.py", "flymon/brain/h4_runner.py", "flymon/brain/h4_spec.py",
              "flymon/brain/h4_pairs.py", "scripts/run_m0d_h4.py"):
        assert f not in MEASURE_FILES and f in HASHED_FILES, f
    for f in ("flymon/brain/h4_formula.py", "flymon/brain/h4_jobs.py", "flymon/brain/plasticity.py",
              "flymon/brain/presentation.py", "flymon/brain/conditioning.py"):
        assert f in MEASURE_FILES, f
