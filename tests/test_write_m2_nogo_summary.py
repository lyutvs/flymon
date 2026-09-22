"""scripts/write_m2_nogo_summary.py (spec appendix I): the no-go chain is derived from the recorded summaries and every
link refuses when it does not hold — the H.4a.7 reading at its boundaries, H.4's outcome and bar, G.12's stop, the
ceilings' identity and provenance, the G.8 record, and a learning unit test that ran."""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("write_m2_nogo_summary", Path("scripts/write_m2_nogo_summary.py").resolve())
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ("m2_oracle", "m2_encoders", "m0d", "m2_probe")
FREQ_COMMIT = "ad77490bee47f4e5d744092bace300cd85a4f4e1"      # H.4a.8: the freq ceiling ran at ad77490
FREQ_SHA = "f1589bfe9f9d98ede5fbb1b725b2a001e1fca8c7bdb0bbe4f73d77e3b247ad07"
A_OK = {"condition": "c", "fired": False, "n_presentations": 2624, "n_over": 0, "max_win_spikes": 30,
        "max_win_hz": 150.0, "margin_spikes": 0, "over": []}


def _agg(tb, fa, n_b=21):
    return {"testable_b": tb, "n_b": n_b, "T_b": tb / n_b, "F_a": fa, "testable_a": 2, "n_a": 18, "bar": False}


def _ceiling(family, c3=(2, 0), c0=(2, 1)):
    s = {"commit": FREQ_COMMIT, "script": w.CEILING_SCRIPT, "script_sha256": FREQ_SHA, "smoke": False,
         "h4_run": "results/m0d/h4/runs/20260921T174430Z-7986f4.json",
         "engines": {"C3": {"aggregate": _agg(*c3)}, "C0": {"aggregate": _agg(*c0)}}}
    if family == "all":
        s["family"] = "all"
    return s


@pytest.fixture
def src(monkeypatch):
    """The committed summaries as they are, the two ceilings as H.4a.8 recorded them, and a G.8 verdict stand-in."""
    s = {k: json.loads((ROOT / w.INPUTS[k]).read_text()) for k in COMMITTED}
    s["ceiling_freq"], s["ceiling_all"] = _ceiling("freq", (2, 0), (2, 1)), _ceiling("all", (3, 0), (4, 1))
    s["g8"] = {"stand-in": True}
    monkeypatch.setattr(w.d6a, "judge", lambda raw: dict(A_OK))
    return s


@pytest.mark.parametrize("tb, fa, want", [
    (14, 2, "A"), (21, 4, "A"), (14, 1, "user"), (13, 4, "user"), (11, 2, "user"), (10, 4, "B"), (3, 0, "B"), (0, 0, "B")])
def test_ceiling_reading_at_the_h4a7_boundaries(tb, fa, want):
    assert w.ceiling_reading(tb, 21, fa) == want


def test_ceiling_reading_refuses_another_pair_count():
    with pytest.raises(w.Refused, match="21"):
        w.ceiling_reading(3, 20, 0)


def test_derive_reads_b_and_records_the_chain(src):
    r = w.derive(src, learning_ran=False)
    ch = r["chain"]
    assert r["not_a_d6c_fail"] is True and r["learning_unit_test"]["ran"] is False
    assert (ch["g10_oracle"]["testable"], ch["g10_oracle"]["pairs"], ch["g10_oracle"]["design_pair_testable"]) == (8, 34, True)
    assert ch["g12_encoders"]["winner"] is None and ch["g12_encoders"]["best"] < 0.5
    assert ch["h4"]["outcome"] == "STOP_LOW_T_B" and ch["h4"]["eligible"] == ["C3"]
    assert {c: (a["testable_b"], a["F_a"]) for c, a in ch["h4"]["combos"].items()} == {"C0": (4, 1), "C1": (2, 0), "C3": (7, 2)}
    fams = ch["h4a8_ceiling"]["families"]
    assert {f: (v["C3"]["testable_b"], v["C3"]["F_a"], v["reading"]) for f, v in fams.items()} == \
        {"freq": (2, 0, "B"), "all": (3, 0, "B")}
    assert ch["h4a8_ceiling"]["reading"] == "B"
    assert r["d6"]["c"]["status"] == "not measured" and r["d6"]["b"]["fired"] is True
    assert r["d6"]["a"]["fired"] is False and r["d6"]["a"]["e2_record"]["max_kc_sub_window_hz"] == 150.0
    assert r["std_condition_fired"] is True          # (b) fired in E.2; D.6 is an OR


def test_d6a_met_is_recorded(src, monkeypatch):
    monkeypatch.setattr(w.d6a, "judge", lambda raw: {**A_OK, "fired": True, "n_over": 3, "max_win_spikes": 33})
    assert w.derive(src, learning_ran=False)["d6"]["a"]["fired"] is True


@pytest.mark.parametrize("family, c3", [("freq", (11, 0)), ("all", (11, 0)), ("all", (14, 2)), ("freq", (14, 1))])
def test_refuses_when_a_ceiling_family_does_not_read_b(src, family, c3):
    src[f"ceiling_{family}"]["engines"]["C3"]["aggregate"] = _agg(*c3)
    with pytest.raises(w.Refused, match="does not read B"):
        w.derive(src, learning_ran=False)


def test_ten_of_21_still_reads_b(src):
    src["ceiling_all"]["engines"]["C3"]["aggregate"] = _agg(10, 4)
    assert w.derive(src, learning_ran=False)["chain"]["h4a8_ceiling"]["families"]["all"]["reading"] == "B"


def test_refuses_when_the_learning_unit_test_ran(src):
    with pytest.raises(w.Refused, match="ran"):
        w.derive(src, learning_ran=True)


def test_refuses_when_h4_did_not_stop(src):
    src["m0d"]["h4"]["h4"]["outcome"] = "SELECTED"
    with pytest.raises(w.Refused, match="SELECTED"):
        w.derive(src, learning_ran=False)


def test_refuses_when_a_combination_is_at_the_bar_whatever_the_stored_flag(src):
    agg = src["m0d"]["h4"]["h4"]["combos"]["C3"]["oracle"]["aggregate"]
    agg.update(T_b=0.5, testable_b=11, F_a=2, bar=False)
    with pytest.raises(w.Refused, match="C3"):
        w.derive(src, learning_ran=False)


def test_just_below_the_bar_is_not_at_it(src):
    src["m0d"]["h4"]["h4"]["combos"]["C3"]["oracle"]["aggregate"].update(T_b=10 / 21, F_a=4)
    assert w.derive(src, learning_ran=False)["chain"]["h4"]["outcome"] == "STOP_LOW_T_B"


@pytest.mark.parametrize("winner, score", [("E1", 0.3), (None, 0.5)])
def test_refuses_when_g12_did_not_stop(src, winner, score):
    src["m2_encoders"]["even"]["decision"]["winner"] = winner
    src["m2_encoders"]["even"]["scores"]["E2"]["score"] = score
    with pytest.raises(w.Refused, match="G.12"):
        w.derive(src, learning_ran=False)


@pytest.mark.parametrize("breakage", [
    lambda s: s.update(smoke=True),
    lambda s: s.update(h4_run="results/m0d/h4/runs/other.json"),
    lambda s: s.update(family="freq"),                 # the all-family summary must say all
])
def test_refuses_a_ceiling_that_is_not_h4a8s(src, breakage):
    breakage(src["ceiling_all"])
    with pytest.raises(w.Refused, match="all-family ceiling"):
        w.derive(src, learning_ran=False)


def test_a_g8_problem_refuses(src, monkeypatch):
    def bad(raw):
        raise ValueError("seeds are not 400-463")
    monkeypatch.setattr(w.d6a, "judge", bad)
    with pytest.raises(w.Refused, match="G.8 record: seeds"):
        w.derive(src, learning_ran=False)


def test_ceiling_provenance_binds_to_committed_code():
    w.ceiling_provenance(_ceiling("freq"))                              # ad77490's script hashes to f1589bfe…
    with pytest.raises(w.Refused, match="hash"):
        w.ceiling_provenance({**_ceiling("freq"), "script_sha256": "0" * 64})
    with pytest.raises(w.Refused, match="ancestor"):
        w.ceiling_provenance({**_ceiling("freq"), "commit": "f" * 40})
    with pytest.raises(w.Refused, match="script"):
        w.ceiling_provenance({**_ceiling("freq"), "script": "scripts/other.py"})


def test_main_refuses_a_dirty_tree_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w, "git_state", lambda out: {"git_commit": "abc", "dirty": ["README.md"]})
    out = tmp_path / "m2_nogo.json"
    assert w.main(["--out", str(out)]) == 2 and "dirty" in capsys.readouterr().err and not out.exists()


def test_main_refuses_missing_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w, "git_state", lambda out: {"git_commit": "abc", "dirty": []})
    monkeypatch.chdir(tmp_path)
    assert w.main(["--out", str(tmp_path / "x.json")]) == 2 and "missing inputs" in capsys.readouterr().err


# ---------------------------------------------------------------- the recorded inputs (git-excluded ones present)
REAL = [ROOT / p for p in w.INPUTS.values()]


@pytest.mark.skipif(not all(p.exists() for p in REAL), reason="git-excluded inputs (ceilings, G.8) not present")
def test_the_recorded_inputs_read_b_and_the_committed_summary_matches():
    src = {k: json.loads((ROOT / p).read_text()) for k, p in w.INPUTS.items()}
    body = w.derive(src, learning_ran=(ROOT / w.LEARNING_SUMMARY).exists())
    assert body["chain"]["h4a8_ceiling"]["reading"] == "B"
    committed = ROOT / w.OUT
    if committed.exists():                       # the committed record is what the code derives from its inputs now
        rec = json.loads(committed.read_text())
        assert {k: rec[k] for k in body} == json.loads(json.dumps(body, ensure_ascii=False))
        assert rec["inputs_sha256"] == {p: w._sha256(ROOT / p) for p in w.INPUTS.values()}
