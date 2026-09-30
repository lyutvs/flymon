"""Plan reading 12: the dated N.8a / N.8b paragraphs carry every number N.8.3 / N.8.6 ask the controller to commit and
pass n_cli.spec_note for the block they cite. The fixtures have the keys the stage scripts really write (run_n0f /
run_n2_pilot / n_oc.simulate); every file written here is under tmp_path, never the real spec."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import n_oc
from flymon.brain.n_cli import note_start, spec_note
from flymon.brain.n_measure import HASHED_FILES
from flymon.brain.n_rules import (N2_0_GO, STOP_BUDGET, STOP_DATA_MISMATCH, STOP_NO_OPERATING_POINT, STOP_POWER,
                                  scenario_key, thresholds)
from flymon.brain.n_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[1]
CELL = dict(kc_frac_median=0.05, runaway_share=0.0, A_zero_share=0.1, P_zero_share=0.0, firing_share=0.9,
            apl_out_mean=0.12)
STIM = ("4:1", "1:4", "dDL", "IA", "EB")
N0F = dict(outcome="OPERATING_POINT", run_id="20261001T000000Z-abc123", selected={"g": 0.5, "c_delta": 4.0},
           checks={"0.5|4": dict(mean_on=0.052, dist=0.0034, ok=True)},
           grid={"0.5|4": {s: {c: CELL for c in ("on", "block", "all")} for s in STIM}},
           state_flags=[dict(stimulus="dDL", condition="block", on=0.9, share=0.5)],
           apl_shift={s: {"block": 0.01, "all": -0.1} for s in STIM},
           lin_totals=dict(totals={"signed": {"IA": 2040.0, "EB": 1870.0, "dDL": 296.0},
                                   "positive": {"IA": 2058.0, "EB": 1870.0, "dDL": 296.0}},
                           rel_error={"signed": {"IA": 0.0049, "EB": 0.0054, "dDL": 0.035},
                                      "positive": {"IA": 0.0138, "EB": 0.0054, "dDL": 0.035}}),
           drives={"4:1": dict(clipped_total_hz=4.6, capped=[]), "dDL": dict(clipped_total_hz=0.0, capped=["ORN_DM2"])},
           wall_s_per_step=0.0021, budget_estimate_h={"16": 20.1, "64": 80.4})
N1 = dict(run_id="20261002T000000Z-n1n1n1", pairs={"sim": {"o": -1.6, "p0": -3.1}, "dis": {"o": -2.4, "p0": -4.0}})
SCEN = {scenario_key("same", 1.0): 16, scenario_key("same", 2.0): 24, scenario_key("independent", 1.0): 16,
        scenario_key("independent", 2.0): 32}
ROWS = [dict(pairing=p, sd_mult=m, hyp=h, n=32, p_supported=ps, mc_se=0.002)
        for p in ("same", "independent") for m in (1.0, 2.0) for h, ps in (("null", 0.01), ("alt", 0.9))]
N2_0 = dict(outcome=N2_0_GO, reason=None, run_id="20261002T000000Z-def456", n=32, c1={"sim": 0.4, "dis": 0.6},
            delta_min=0.3, eps=0.2, ell_hat={"sim": 0.6, "dis": 0.8}, pilot_sd={"sim": 0.5, "dis": 0.4},
            alt_D_sim=0.45, budget_h=31.5, pilot_seeds=list(range(21_001_000, 21_001_016)), scenario_n=SCEN,
            oc_boot=1234, judge_boot=56789, upstream={"n1": N1["run_id"]},
            oc=dict(draws=2000, boot=1234, judge_boot=56789, partial=False, n=32, scenario_n=SCEN, rows=ROWS))
DOC = {"n0f": N0F, "n1": N1, "n2_0": N2_0}
SPEC_TEXT = ("# design\n\n## 부록 N. 새 주장\n\n### N.7 추가\nbody\n\n### N.8 개정\n"
             "- 결과는 날짜를 붙인 **N.8a**로 커밋한다. 표는 **N.8b**로 커밋한다.\n\n**N.8.9 멈춤 순서**: a → b.\n")


def _script():
    sp = importlib.util.spec_from_file_location("write_n_notes", ROOT / "scripts" / "write_n_notes.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules["write_n_notes"] = mod
    sp.loader.exec_module(mod)
    return mod


def _files(tmp_path, doc=DOC, spec_text=SPEC_TEXT):
    s, p = tmp_path / "n.json", tmp_path / "spec.md"
    s.write_text(json.dumps(doc))
    p.write_text(spec_text)
    return s, p


def _args(s, p, which, date="2026-10-02"):
    return ["--summary", str(s), "--which", which, "--date", date, "--write", "--spec", str(p)]


# ---------------------------------------------------------------- N.8a
def test_n8a_carries_the_point_states_lin_and_budget_and_passes_the_gate():
    t = _script().n8a(N0F, "2026-10-01")
    assert t.startswith("**N.8a") and N0F["run_id"] in t and "g = 0.5, c_δ = 4" in t
    for s in ("2040", "+3.5%", "0.9 → 0.5", "80.4 h", "2.1 ms", "ORN_DM2", "4.6 Hz", "+0.0100", "-0.1000",
              f"{100 * SPEC.kc_target:.2f}%", f"> {SPEC.state_diff_flag:g}"):
        assert s in t, s
    assert "{" not in t                                  # no raw dict ever reaches the paragraph
    assert spec_note("spec\n" + t, "N.8a", N0F["run_id"]) is None


def test_n8a_of_a_stop_says_so():
    t = _script().n8a(dict(N0F, outcome=STOP_NO_OPERATING_POINT, selected=None), "2026-10-01")
    assert STOP_NO_OPERATING_POINT in t and "작동점 없음" in t


def test_n8a_refuses_a_data_mismatch_and_an_empty_or_short_block():
    m = _script()
    for bad in (dict(outcome=STOP_DATA_MISMATCH, reason="NA", run_id="r"), {}, dict(N0F, run_id=""),
                {k: v for k, v in N0F.items() if k != "grid"}, dict(N0F, selected=None),
                dict(N0F, outcome=STOP_NO_OPERATING_POINT), dict(N0F, outcome="SOMETHING_ELSE")):
        with pytest.raises((ValueError, KeyError)):
            m.n8a(bad, "2026-10-01")


# ---------------------------------------------------------------- N.8b
def test_n8b_carries_the_thresholds_and_the_oc_table_and_passes_the_gate():
    t = _script().n8b(N2_0, N1, "2026-10-02")
    assert t.startswith("**N.8b") and N2_0["run_id"] in t
    for s in ("n = 32", "c₁ = sim 0.4, dis 0.6", "δ_min = 0.3", "ε = 0.2", "o: sim -1.6, dis -2.4", "31.5 h",
              f"한도 {SPEC.budget_h:g} h", "16 시드", "| same | 1 | null | 32 | 0.010 | 0.002 |",
              "| independent | 2 | alt | 32 | 0.900 | 0.002 |"):
        assert s in t, s
    assert spec_note(t, "N.8b", N2_0["run_id"]) is None


def test_n8b_discloses_every_scenario_n_the_joint_rule_the_degenerate_scenario_and_the_bootstrap_approximation():
    t = _script().n8b(N2_0, N1, "2026-10-02")
    for k, n in SCEN.items():                            # (a) all four scenarios, each with its own n
        assert f"{k} n = {n}" in t, k
    assert len(SCEN) == len(SPEC.oc_pairings) * len(SPEC.sd_mults) == 4
    assert "동시에 통과하는 가장 작은 n" in t
    deg = scenario_key("same", 1.0)                      # (b) the same-resample 1x scenario is degenerate
    line = next(ln for ln in t.splitlines() if "퇴화" in ln)
    assert deg in line and "상수" in line and "폭 0" in line and "②" in line and "③" in line
    boot = next(ln for ln in t.splitlines() if "근사" in ln)   # (c) oc_boot vs judge_boot, read from the block
    assert "1234" in boot and "56789" in boot and str(SPEC.oc_boot) not in boot and str(SPEC.boot_draws) not in boot


def test_n8b_shows_a_scenario_without_its_own_n_as_none():
    b = copy.deepcopy(N2_0)
    b["scenario_n"][scenario_key("same", 2.0)] = None
    assert f"{scenario_key('same', 2.0)} n = 없음" in _script().n8b(b, N1, "2026-10-02")


def test_n8b_refuses_a_partial_oc_a_stop_a_missing_scenario_and_another_n1_run():
    m = _script()
    part = copy.deepcopy(N2_0)
    part["oc"]["partial"] = True
    no_flag = copy.deepcopy(N2_0)
    del no_flag["oc"]["partial"]
    short = copy.deepcopy(N2_0)
    del short["scenario_n"][scenario_key("independent", 2.0)]
    for bad in (part, no_flag, short, dict(N2_0, oc=None), dict(N2_0, outcome=STOP_POWER, n=None, reason="no n"),
                dict(N2_0, outcome=STOP_BUDGET, reason="60 h"), dict(N2_0, n=None), dict(N2_0, run_id=None), {},
                dict(N2_0, oc=dict(N2_0["oc"], rows=[])), dict(N2_0, upstream={"n1": "another-run"})):
        with pytest.raises((ValueError, KeyError)):
            m.n8b(bad, N1, "2026-10-02")
    with pytest.raises((ValueError, KeyError)):
        m.n8b(N2_0, {}, "2026-10-02")


def test_n8b_of_what_the_real_oc_writes():
    """The block keys are n_oc.simulate's and n_rules.thresholds' own output, laid out as run_n2_pilot.body does."""
    spec = smoke(SPEC)
    rng = np.random.default_rng(3)
    pilot = {"sim": rng.normal(-1.0, 0.2, 6), "dis": rng.normal(-1.2, 0.2, 6)}
    o = {"sim": -1.6, "dis": -2.4}
    th = thresholds(o, pilot, spec)
    oc = json.loads(json.dumps(n_oc.simulate(pilot, th["c1"], th["delta_min"], th["eps"], spec, spec.oc_pairings)))
    b = dict(outcome=N2_0_GO, reason=None, run_id="rid-real", n=spec.n_grid[0], c1=th["c1"],
             delta_min=th["delta_min"], eps=th["eps"], ell_hat=th["ell_hat"], pilot_sd=th["pilot_sd"],
             alt_D_sim=th["alt_D_sim"], pilot_seeds=list(spec.pilot_seeds), oc=oc, scenario_n=oc["scenario_n"],
             oc_boot=oc["boot"], judge_boot=spec.boot_draws, budget_h=1.25)
    t = _script().n8b(b, dict(pairs={k: {"o": v} for k, v in o.items()}), "2026-10-02")
    assert t.count("\n| ") == 2 + len(oc["rows"])        # header, rule, one line per OC row
    assert f"부트스트랩 {spec.oc_boot}회" in t and f"{spec.boot_draws}회" in t
    assert spec_note(t, "N.8b", "rid-real") is None
    with pytest.raises(ValueError):                      # the same table from a subset of the pairings is partial
        _script().n8b(dict(b, oc=json.loads(json.dumps(n_oc.simulate(
            pilot, th["c1"], th["delta_min"], th["eps"], spec, ["same"])))), dict(pairs={k: {"o": v} for k, v in o.items()}),
            "2026-10-02")


# ---------------------------------------------------------------- main: stdout, and --write into a spec copy
def test_main_reads_the_summary(tmp_path, capsys):
    s, _ = _files(tmp_path)
    m = _script()
    assert m.main(["--summary", str(s), "--which", "n8b", "--date", "2026-10-02"]) == 0
    assert capsys.readouterr().out.startswith("**N.8b")
    assert m.main(["--summary", str(tmp_path / "none.json"), "--which", "n8a"]) == 2
    assert capsys.readouterr().out == ""


def test_write_puts_the_note_at_the_end_of_n8_and_the_real_gate_accepts_it(tmp_path):
    s, p = _files(tmp_path, spec_text=SPEC_TEXT + "\n## 부록 O. 다음\n\n**N.9x 다른 것**\n")
    m = _script()
    assert note_start(p.read_text(), "N.8a") is None and "no N.8a" in spec_note(p.read_text(), "N.8a", N0F["run_id"])
    assert m.main(_args(s, p, "n8a", "2026-10-01")) == 0
    text = p.read_text()
    assert spec_note(text, "N.8a", N0F["run_id"]) is None            # n_cli's gate, as the stages run it
    assert "does not cite" in spec_note(text, "N.8a", "another-run")
    assert "no N.8b" in spec_note(text, "N.8b", N2_0["run_id"])
    assert text.index("**N.8.9") < text.index("\n**N.8a ") < text.index("## 부록 O.")   # inside N.8, after its text
    assert text.replace(m.n8a(N0F, "2026-10-01") + "\n", "") == SPEC_TEXT + "\n## 부록 O. 다음\n\n**N.9x 다른 것**\n"
    assert m.main(_args(s, p, "n8b")) == 0
    text = p.read_text()
    assert spec_note(text, "N.8b", N2_0["run_id"]) is None and spec_note(text, "N.8a", N0F["run_id"]) is None
    assert text.index("\n**N.8a ") < text.index("\n**N.8b ") < text.index("## 부록 O.")


def test_write_is_idempotent_and_never_overwrites_a_different_note(tmp_path, capsys):
    s, p = _files(tmp_path)
    m = _script()
    assert m.main(_args(s, p, "n8b")) == 0
    once = p.read_text()
    assert m.main(_args(s, p, "n8b")) == 0 and p.read_text() == once          # same content: nothing is added
    assert once.count("**N.8b N2.0") == 1
    assert m.main(_args(s, p, "n8b", "2026-10-03")) == 2 and p.read_text() == once   # other content: refused
    assert "already" in capsys.readouterr().err
    hand = SPEC_TEXT + "\n### N.8a 손으로 쓴 것\nrun x\n"
    p.write_text(hand)
    assert m.main(_args(s, p, "n8a")) == 2 and p.read_text() == hand


@pytest.mark.parametrize("which, doc", [
    ("n8a", {}), ("n8a", {"n0f": {}}), ("n8a", {"n0f": dict(outcome=STOP_DATA_MISMATCH, reason="NA", run_id="r")}),
    ("n8b", {"n0f": N0F, "n1": N1}), ("n8b", {"n2_0": N2_0}), ("n8b", dict(DOC, n2_0=dict(N2_0, oc=None))),
    ("n8b", dict(DOC, n2_0=dict(N2_0, oc=dict(N2_0["oc"], partial=True)))),
    ("n8b", dict(DOC, n2_0=dict(N2_0, outcome=STOP_POWER, n=None))), ("n8a", []), ("n8b", "text")])
def test_a_refusal_exits_2_and_writes_nothing(tmp_path, capsys, which, doc):
    s, p = _files(tmp_path, doc)
    assert _script().main(_args(s, p, which)) == 2
    cap = capsys.readouterr()
    assert p.read_text() == SPEC_TEXT and cap.out == "" and cap.err.startswith("refused:")


def test_write_refuses_a_spec_without_the_n8_section_or_no_spec(tmp_path):
    s, p = _files(tmp_path, spec_text="# design\n\n### N.7 추가\nbody\n")
    m = _script()
    assert m.main(_args(s, p, "n8a")) == 2 and p.read_text() == "# design\n\n### N.7 추가\nbody\n"
    assert m.main(_args(s, tmp_path / "none.md", "n8a")) == 2 and not (tmp_path / "none.md").exists()


def test_the_defaults_are_the_repository_s_own_paths_whatever_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _script()
    a = m.parser().parse_args(["--which", "n8a"])
    assert Path(a.summary) == ROOT / "results/summary/n_real_odour.json" and Path(a.spec) == ROOT / SPEC.spec_path
    assert not a.write                                                # printing is the default; writing is asked for


# ---------------------------------------------------------------- the manifest
def test_every_hashed_file_exists():
    assert [f for f in HASHED_FILES if not (ROOT / f).exists()] == []
    assert "scripts/write_n_notes.py" in HASHED_FILES

