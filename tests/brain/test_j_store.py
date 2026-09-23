"""Spec J.10.8 / J.11.6: J's summaries are guarded; block "h3" and C3's threshold copy are read-only to J."""
import json
import shutil
from pathlib import Path

import pytest

from flymon.brain.config import Params
from flymon.brain.j_params import StdParams
from flymon.brain.j_store import guard, load_c3, write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", ["results/m0d/j/std/runs/x.json", "results/summary/std_scan.json",
                                  "results/summary/m2_engine.json"])
def test_allowed_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(path, {"a": 1}, [Params(), StdParams(orn_std=True, receptor_scale=2.0)])
    assert json.loads(Path(path).read_text()) == {"a": 1}


@pytest.mark.parametrize("path", ["results/summary/m0d.json", "results/summary/m0d_h3_c3_thresholds.npz",
                                  "results/summary/m0c.json", "results/m0c/x.json", "results/m2/x.json", "x.json"])
def test_refused_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        guard(path, [StdParams(orn_std=True, receptor_scale=2.0)])
    assert not Path(path).exists()


def test_load_c3_reads_block_h3_and_restores_the_threshold_file_from_the_copy(tmp_path, monkeypatch):
    summary = json.loads((ROOT / "results/summary/m0d.json").read_text())
    monkeypatch.chdir(tmp_path)
    Path("results/summary").mkdir(parents=True)
    shutil.copy(ROOT / "results/summary/m0d_h3_c3_thresholds.npz", "results/summary/m0d_h3_c3_thresholds.npz")
    p, guard_stats, cell = load_c3(summary)
    assert type(p) is Params and p.kc_thresh == 1.65 and p.kc_thresh_mode == "homeostatic"
    assert Path(p.kc_thresh_file).exists() and {"MBON13", "MBON05"} <= set(guard_stats)
    Path("results/summary/m0d_h3_c3_thresholds.npz").unlink()
    load_c3(summary)                                           # the restored file is there now
    Path(p.kc_thresh_file).unlink()
    with pytest.raises(ValueError, match="missing"):
        load_c3(summary)
