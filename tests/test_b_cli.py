"""Spec J.12's CLIs: the run refuses before measuring; the summary writer derives verdicts from raw records only when
they are this code's complete, clean runs (the runs themselves are controller steps)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "brain"))
from b_fixtures import records  # noqa: E402

NPZ = ROOT / "data/malecns.npz"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RUN, SUM = _load("run_b_test"), _load("write_b_summary")


def test_the_run_refuses_an_unknown_pair_outside_the_root_and_dirty_files(tmp_path, monkeypatch, capsys):
    assert RUN.main(["--pair", "nope"], require_root=False) == 2 and "--pair" in capsys.readouterr().err
    monkeypatch.chdir(tmp_path)
    assert RUN.main(["--pair", "calibration"]) == 2 and "repository root" in capsys.readouterr().err
    monkeypatch.setattr(RUN, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/b_rules.py"],
                                                             dirty_other=[]))
    assert RUN.main(["--pair", "calibration"], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_the_run_refuses_a_foreign_connectome(tmp_path, monkeypatch, capsys):
    npz = tmp_path / "other.npz"
    npz.write_bytes(b"x")
    monkeypatch.setattr(RUN, "git_state", lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[]))
    assert RUN.main(["--pair", "calibration", "--npz", str(npz)], require_root=False) == 2
    assert "declared connectome" in capsys.readouterr().err


def test_the_smoke_spec_is_not_the_declared_one():
    assert RUN.smoke_spec(RUN.SPEC) != RUN.SPEC


def _raw(tmp, pair, recs, key, name=None, **over):
    from flymon.brain.b_spec import SPEC
    from flymon.brain.h3_store import canonical
    d = dict(run_id=f"r-{pair}", pair=pair, smoke=False, git=dict(dirty_hashed=[]), spec=json.loads(canonical(SPEC)),
             measure_key=key, x="b", records=recs)
    d.update(over)
    p = tmp / f"{name or pair}.json"
    p.write_text(json.dumps(d))
    return str(p)


@pytest.mark.skipif(not NPZ.exists(), reason="MaleCNS connectome not built")
def test_the_writer_calibrates_then_judges_and_refuses_what_is_not_this_code_s_clean_run(tmp_path, monkeypatch, capsys):
    key = SUM.code_key(str(NPZ), files=SUM.MEASURE_FILES)["key"]
    monkeypatch.chdir(tmp_path)
    cal = _raw(tmp_path, "calibration", records(reward=10, punish=25, n2=True, noise=4.0), key)
    assert SUM.main(["test", cal, cal], npz=str(NPZ)) == 2 and "results/summary/b_calibration.json" in capsys.readouterr().err
    assert SUM.main(["calibration", _raw(tmp_path, "calibration", [], key, name="smoke", smoke=True)], npz=str(NPZ)) == 2
    assert SUM.main(["calibration", _raw(tmp_path, "calibration", [], "0" * 64, name="foreign")], npz=str(NPZ)) == 2
    assert SUM.main(["calibration", _raw(tmp_path, "exploration", [], key, name="wrongpair")], npz=str(NPZ)) == 2
    flipped = _raw(tmp_path, "calibration", records(reward=10, punish=25, n2=True, noise=4.0), key, name="flipped",
                   x="a")
    assert SUM.main(["calibration", flipped], npz=str(NPZ)) == 2 and "not the rule's" in capsys.readouterr().err
    assert not Path("results/summary/b_calibration.json").exists()
    assert SUM.main(["calibration", cal], npz=str(NPZ)) == 0
    c = json.loads(Path("results/summary/b_calibration.json").read_text())
    assert c["status"] == "CALIBRATED" and set(c["values"]) == {"reward", "punish", "choice"}
    ex = _raw(tmp_path, "exploration", records(reward=10, punish=25, pair="exploration"), key)
    co = _raw(tmp_path, "confirmation", records(reward=10, punish=25, seed=1, pair="confirmation"), key)
    assert SUM.main(["test", ex, co], npz=str(NPZ)) == 0
    t = json.loads(Path("results/summary/b_test.json").read_text())
    assert t["verdict"]["outcome"] == "PASS" and t["thresholds"] == c["values"]
    bad = _raw(tmp_path, "confirmation", records(drift_r=20, drift_p=20, seed=1, pair="confirmation"), key,
               name="drift")
    assert SUM.main(["test", ex, bad], npz=str(NPZ)) == 0
    assert json.loads(Path("results/summary/b_test.json").read_text())["verdict"]["outcome"] == "FAIL"
