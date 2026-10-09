"""scripts/run_ad.py without Showdown: arguments, the refusals that run before any pool or server (inputs ->
STOP_MANIFEST, stage-1 reuse -> STOP_REUSE, model manifest, run manifest -> STOP_MANIFEST, a used bench directory,
a spent budget), the session window, and the id / name lengths."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.ac import schedules as ac_sched
from flymon.ad import inputs, run_manifest as rm, store
from flymon.ad.spec import LABEL, SPEC, smoke
from tests.ad.test_ad_reuse import fake_stage1

ROOT = Path(__file__).resolve().parents[2]


def load():
    s = importlib.util.spec_from_file_location("rad", ROOT / "scripts/run_ad.py")
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod


def smoke_inputs():
    s = smoke()
    doc_p, man_p = inputs.input_paths(s)
    d = inputs.inputs_doc(s)
    store.write_json(doc_p, d)
    store.write_json(man_p, inputs.input_manifest(d, doc_p))
    return doc_p, man_p


def frozen_model(tmp_path):
    p = tmp_path / "results/summary/ac_model_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "FROZEN", "recovery_per_pulse": 0.001}))


def test_parse_args():
    ra = load()
    a = ra.parse_args(["--smoke"])
    assert a.workers == 4 and ra.spec_for(a).out_root == "results/m4-smoke/ad"
    a = ra.parse_args(["--bench", "2"])
    assert a.workers == 16 and ra.spec_for(a).out_root == "results/m4-bench/ad-2"
    assert ra.spec_for(ra.parse_args([])) == SPEC
    with pytest.raises(SystemExit):
        ra.parse_args(["--bench", "1", "--resume"])
    with pytest.raises(SystemExit):
        ra.parse_args(["--bench", "1", "--smoke"])


def test_refuses_missing_or_changed_inputs(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        ra.main(["--smoke"])
    doc_p, _ = smoke_inputs()
    d = json.loads(doc_p.read_text())
    d["eval"][0]["my_team"] = list(reversed(d["eval"][0]["my_team"]))
    doc_p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        ra.main(["--smoke"])


def test_refuses_without_a_frozen_model(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    smoke_inputs()
    with pytest.raises(SystemExit, match="ac_model_manifest"):
        ra.main(["--smoke"])
    store_p = tmp_path / "results/summary/ac_model_manifest.json"
    store_p.parent.mkdir(parents=True, exist_ok=True)
    store_p.write_text(json.dumps({"status": "STOP_NO_RECOVERY"}))
    with pytest.raises(SystemExit, match="not FROZEN"):
        ra.main(["--smoke"])


def test_resume_refuses_a_changed_run_manifest(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    _, man_p = smoke_inputs()
    frozen_model(tmp_path)
    with pytest.raises(SystemExit, match="run_manifest"):
        ra.main(["--smoke", "--allow-dirty"])
    m = rm.build(smoke(), T=2, new_set=True, budget_path=None, inputs_manifest_path=man_p, commit="0" * 40)
    store.write_json(rm.manifest_path(smoke()), dict(m, points=[0, 2]))
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        ra.main(["--smoke", "--resume", "--allow-dirty"])


def test_run_mode_refuses_a_changed_stage1_record(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    fake_stage1(tmp_path)
    w = importlib.util.spec_from_file_location("wai", ROOT / "scripts/write_ad_inputs.py")
    mod = importlib.util.module_from_spec(w); w.loader.exec_module(mod)
    assert mod.main([]) == 0
    with (tmp_path / "results/m4/stage1/BRAIN/logs/situations_init.jsonl").open("a") as f:
        f.write(json.dumps({"fly": 12, "point": 0, "arm": "COFF", "frozen": True, "label": LABEL}) + "\n")
    with pytest.raises(SystemExit, match="STOP_REUSE"):
        ra.main([])


def test_bench_refuses_a_used_directory(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/m4-bench/ad-1/logs").mkdir(parents=True)
    with pytest.raises(SystemExit, match="earlier bench attempt"):
        ra.fresh_bench(Path("results/m4-bench/ad-1"))
    (tmp_path / "results/m4-bench/ad-2").mkdir(parents=True)
    (tmp_path / "results/m4-bench/ad-2/inputs.json").write_text("{}")
    ra.fresh_bench(Path("results/m4-bench/ad-2"))                 # inputs alone are not an earlier attempt


def test_session_window(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    out = Path("results/m4/ad")
    a = ra.parse_args([])
    cap, left = ra.session_window(SPEC, out, a, 0.0)
    assert cap == 24 * 3600 and left == pytest.approx(SPEC.budget_limit_h() * 3600)
    store.write_json(out / "wall_clock.json", {"sessions": [dict(started_utc="1970-01-01T00:00:00+00:00",
                                                                seconds=47.5 * 3600, status="ok")]})
    with pytest.raises(SystemExit, match="STOP_BUDGET"):
        ra.session_window(SPEC, out, a, 0.0)
    cap, left = ra.session_window(smoke(), Path("results/m4-smoke/ad"), ra.parse_args(["--smoke", "--session-hours",
                                                                                         "0.5"]), 0.0)
    assert cap == 1800.0 and left == float("inf")


def test_run_ids_and_names_fit():
    doc = inputs.inputs_doc(SPEC)
    learn, ev = inputs.schedules_for(SPEC, doc, 120)
    ac_sched.check_names(learn + ev, SPEC.retry_max)
    assert {inputs.account(a, b, g) for a, _, g in SPEC.layout() for b in "LE"} and len(learn) == 2880
