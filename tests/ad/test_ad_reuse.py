"""AD.2 (가) / AD.5 6: the model manifest, the confirmation and evaluation digests and the six C-off point-0 records
reused from stage 1 must be stage 1's; any mismatch is STOP_REUSE. Fake stage-1 trees in tmp_path."""
import importlib.util
import json
from pathlib import Path

from flymon.ad import inputs, reuse, store
from flymon.ad.spec import LABEL, SPEC
from flymon.rescope import blocks

ROOT = Path(__file__).resolve().parents[2]


def fake_stage1(root: Path, *, frozen=True, model_status="FROZEN"):
    (root / "results/summary").mkdir(parents=True, exist_ok=True)
    mp = root / "results/summary/ac_model_manifest.json"
    mp.write_text(json.dumps({"status": model_status, "recovery_per_pulse": 0.001}))
    b = root / "results/m4/stage1/BRAIN"
    (b / "logs").mkdir(parents=True, exist_ok=True)
    (b / "result.json").write_text(json.dumps(dict(model=dict(sha256=store.sha256_file(mp)),
                                                   confirm_digest=reuse.CONFIRM_DIGEST, eval_digest=reuse.EVAL_DIGEST)))
    recs = [dict(kind="situation_eval", fly=g, point=0, arm="FLY" if g < 12 else "COFF", switched=[0] * 20,
                 frozen=frozen or g < 12, label=LABEL) for g in range(18)]
    (b / "logs/situations_init.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def fields():
    return reuse.reuse_fields(reuse.CONFIRM_DIGEST, reuse.EVAL_DIGEST, SPEC)


def test_matching_stage1_has_no_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_stage1(tmp_path)
    f = fields()
    assert reuse.reuse_problems(f) == [] and f["coff_flies"] == [12, 13, 14, 15, 16, 17] and f["coff_arm"] == ["COFF"]


def test_each_mismatch_is_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_stage1(tmp_path, frozen=False)
    assert reuse.reuse_problems(fields()) == ["coff_frozen"]
    fake_stage1(tmp_path)
    (tmp_path / "results/summary/ac_model_manifest.json").write_text("{}")
    assert reuse.reuse_problems(fields()) == ["model_status", "model_sha256"]
    fake_stage1(tmp_path)
    f = reuse.reuse_fields("0" * 64, reuse.EVAL_DIGEST, SPEC)
    assert reuse.reuse_problems(f) == ["confirm_digest"]


def test_changed_since_compares_the_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_stage1(tmp_path)
    old = fields()
    blocks.append_jsonl(tmp_path / "results/m4/stage1/BRAIN/logs/situations_init.jsonl", {"fly": 99, "point": 0})
    assert reuse.changed_since(old, fields()) == ["coff_file_sha256"]


def load_script():
    spec = importlib.util.spec_from_file_location("wai", ROOT / "scripts/write_ad_inputs.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_script_smoke_and_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    w = load_script()
    assert w.main(["--smoke"]) == 0
    man = json.loads((tmp_path / "results/m4-smoke/ad/inputs_manifest.json").read_text())
    assert man["status"] == "OK" and man["reuse"] is None
    fake_stage1(tmp_path)
    assert w.main([]) == 0
    man = json.loads((tmp_path / "results/summary/ad_inputs_manifest.json").read_text())
    assert man["status"] == "OK" and man["reuse"]["coff_content_hash"] and man["reuse_problems"] == []
    fake_stage1(tmp_path, model_status="STOP_NO_RECOVERY")
    assert w.main([]) == 2
    assert json.loads((tmp_path / "results/summary/ad_inputs_manifest.json").read_text())["status"] == "STOP_REUSE"


def test_missing_stage1_is_stop_reuse_and_writes_nothing_frozen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = load_script()
    fake_stage1(tmp_path)
    (tmp_path / "results/m4/stage1/BRAIN/result.json").unlink()
    assert reuse.reuse_problems(fields()) == ["missing_stage1"]
    assert w.main([]) == 2
    man = json.loads((tmp_path / "results/summary/ad_inputs_manifest.json").read_text())
    assert man["status"] == "STOP_REUSE" and man["reuse_problems"] == ["missing_stage1"]
    assert man["inputs_sha256"] is None and not (tmp_path / "results/summary/ad_inputs.json").exists()


def test_digest_drift_after_freeze_is_stop_reuse_not_refusing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = load_script()
    fake_stage1(tmp_path)
    assert w.main([]) == 0
    doc_p = tmp_path / "results/summary/ad_inputs.json"
    frozen = doc_p.read_text()
    real = inputs.inputs_doc

    def drifted(spec):
        d = real(spec)
        return dict(d, digests=dict(d["digests"], confirm="0" * 64))
    monkeypatch.setattr(inputs, "inputs_doc", drifted)
    assert w.main([]) == 2
    man = json.loads((tmp_path / "results/summary/ad_inputs_manifest.json").read_text())
    assert man["status"] == "STOP_REUSE" and man["reuse_problems"] == ["confirm_digest"]
    assert doc_p.read_text() == frozen
