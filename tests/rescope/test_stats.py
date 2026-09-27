"""M4 statistics: win tables, the paired fly -> battle bootstrap, the (2a)/(2b) verdict (INVALID on partial or
malformed input, never PASS / FAIL), the recorded information-turn match and the write_rescope_m4 CLI."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.rescope import stats
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]


def arm(rates, n=200, seed=0, invalid=()):
    rng = np.random.default_rng(seed)
    return {"per_fly": [{"fly": k, "invalid": k in invalid,
                         "eval_battles": [{"won": bool(w), "finished": True} for w in rng.random(n) < r]}
                        for k, r in enumerate(rates)]}


def five(n_flies=12, n=200, fly=0.5, coff=0.2, rs=0.5, rnd=0.15, mx=0.47):
    return {"FLY": arm([fly] * n_flies, n, seed=1), "COFF": arm([coff] * n_flies, n, seed=2),
            "RS": arm([rs] * n_flies, n, seed=3), "RND": arm([rnd] * n_flies, n, seed=4),
            "MAX": arm([mx] * n_flies, n, seed=5)}


# ---- the brief's tests ----------------------------------------------------------------------------
def test_clear_difference_detected():
    out = stats.paired_boot(stats.win_table(arm([0.5] * 12, seed=1)), stats.win_table(arm([0.2] * 12, seed=2)), 2000, 0)
    assert out["lo"] > 0 and abs(out["diff"] - 0.3) < 0.05


def test_no_difference_ci_contains_zero():
    out = stats.paired_boot(stats.win_table(arm([0.3] * 12, seed=3)), stats.win_table(arm([0.3] * 12, seed=4)), 2000, 0)
    assert out["lo"] < 0 < out["hi"]


def test_invalid_flies_dropped_pairwise():
    a = stats.win_table(arm([0.5] * 4, invalid=(1,)))
    b = stats.win_table(arm([0.2] * 4))
    assert stats.paired_boot(a, b, 200, 0)["n_pairs"] == 3


def test_2b_needs_both():
    v = stats.m4_verdict(five(), SPEC)
    assert v["status"] == stats.OK
    assert v["2a"]["pass"] is True and v["2a"]["verdict"] == stats.PASS
    assert v["2b"]["vs_coff"]["lo"] > 0 and v["2b"]["pass"] is False and v["2b"]["verdict"] == stats.FAIL


# ---- win table and bootstrap ----------------------------------------------------------------------
def test_unfinished_counts_as_loss():
    res = {"per_fly": [{"fly": 0, "invalid": False, "eval_battles": [
        {"won": True, "finished": True}, {"won": None, "finished": False}, {"won": False, "finished": True}]}]}
    assert stats.win_table(res)[0].tolist() == [1.0, 0.0, 0.0]


def test_boot_is_deterministic_and_paired():
    a, b = stats.win_table(arm([0.5] * 8, 40, seed=1)), stats.win_table(arm([0.3] * 8, 40, seed=2))
    assert stats.paired_boot(a, b, 1000, 7) == stats.paired_boot(a, b, 1000, 7)
    # identical arms: every draw picks the same flies for both, so a zero-variance fly level gives diff 0 exactly
    same = stats.paired_boot(a, a, 500, 0)
    assert same["diff"] == 0.0 and same["lo"] < 0 < same["hi"]      # battle level still resampled per arm


def test_fly_level_resampled_jointly():
    """A big between-fly spread shared by both arms cancels in the paired difference: the CI is tight."""
    rates = np.linspace(0.1, 0.9, 12)
    a = stats.win_table(arm(list(rates + 0.05), 400, seed=1))
    b = stats.win_table(arm(list(rates - 0.05), 400, seed=2))
    out = stats.paired_boot(a, b, 2000, 0)
    assert out["lo"] > 0 and out["hi"] - out["lo"] < 0.1


# ---- invalid pairs and malformed input ------------------------------------------------------------
def test_rs_donor_invalid_drops_the_pair():
    arms = five()
    arms["RS"]["per_fly"][3]["donor_invalid"] = True          # FLY 3 invalid upstream; the RS row itself is not
    arms["RS"]["per_fly"][3]["invalid"] = False
    v = stats.m4_verdict(arms, SPEC)
    assert 3 not in v["2b"]["vs_rs"]["flies"] and v["2b"]["vs_rs"]["n_pairs"] == 11
    assert v["invalid_pairs"]["vs_rs"]["flies"] == [3] and v["invalid_pairs"]["vs_rs"]["n"] == 1
    assert v["invalid_pairs"]["vs_coff"]["n"] == 0 and 3 in v["invalid_flies"]["RS"]


def test_fly_row_invalid_drops_every_fly_pair():
    arms = five()
    arms["FLY"]["per_fly"][5]["invalid"] = True
    v = stats.m4_verdict(arms, SPEC)
    assert all(v["invalid_pairs"][k]["flies"] == [5] for k in ("2a", "vs_coff", "vs_rs"))


def test_rs_residual_over_max_is_invalid_even_if_unflagged():
    arms = five()
    arms["RS"]["per_fly"][0].update(residual_frac=0.06, invalid=False)
    assert stats.m4_verdict(arms, SPEC)["invalid_pairs"]["vs_rs"]["flies"] == [0]


def test_rs_pair_shortage_invalidates_2b_only():
    arms = five(n_flies=8)
    for k in (0, 1, 2):
        arms["RS"]["per_fly"][k]["donor_invalid"] = True       # 5 valid FLY/RS pairs < 6 (6/8)
    v = stats.m4_verdict(arms, SPEC)
    assert v["status"] == stats.OK and v["min_valid_pairs"] == 6
    assert v["2b"]["verdict"] == stats.INVALID and v["2b"]["pass"] is None and v["2b"]["reasons"]
    assert v["2a"]["verdict"] == stats.PASS and v["2a"]["pass"] is True and not v["2a"]["reasons"]
    arms["RS"]["per_fly"][2]["donor_invalid"] = False          # 6 valid pairs: judged
    assert stats.m4_verdict(arms, SPEC)["2b"]["verdict"] == stats.FAIL


def test_coff_pair_shortage_invalidates_2b_only():
    arms = five(n_flies=8)
    for k in (0, 1, 2):
        arms["COFF"]["per_fly"][k]["invalid"] = True
    v = stats.m4_verdict(arms, SPEC)
    assert v["2b"]["verdict"] == stats.INVALID and v["2a"]["verdict"] == stats.PASS


def test_rnd_pair_shortage_invalidates_2a_only():
    arms = five(n_flies=8, rs=0.2)                              # (2b) judged and passes
    for k in (0, 1, 2):
        arms["RND"]["per_fly"][k]["invalid"] = True
    v = stats.m4_verdict(arms, SPEC)
    assert v["status"] == stats.OK
    assert v["2a"]["verdict"] == stats.INVALID and v["2a"]["pass"] is None
    assert v["2b"]["verdict"] == stats.PASS and v["2b"]["pass"] is True


def test_fly_shortage_invalidates_both():
    arms = five(n_flies=8)
    for k in (0, 1, 2):
        arms["FLY"]["per_fly"][k]["invalid"] = True
    v = stats.m4_verdict(arms, SPEC)
    assert v["2a"]["verdict"] == v["2b"]["verdict"] == stats.INVALID


def test_min_valid_pairs():
    assert [stats.min_valid_pairs(f) for f in (1, 2, 4, 6, 8, 12)] == [2, 2, 3, 5, 6, 9]


@pytest.mark.parametrize("bad", ["0.01", None, [0.01], {"x": 1}, True, float("nan"), float("inf")])
def test_non_numeric_residual_is_invalid_not_an_error(bad):
    assert stats.row_invalid({"fly": 0, "invalid": False, "residual_frac": bad}) is True
    arms = five(n_flies=8, n=40)
    arms["RS"]["per_fly"][0]["residual_frac"] = bad
    v = stats.m4_verdict(arms, SPEC)
    assert v["status"] == stats.INVALID and v["2b"]["pass"] is None


@pytest.mark.parametrize("break_it", [
    lambda a: a.pop("MAX"),
    lambda a: a["RS"].__setitem__("per_fly", []),
    lambda a: a["COFF"]["per_fly"].pop(4),
    lambda a: a["COFF"]["per_fly"].append(copy.deepcopy(a["COFF"]["per_fly"][0])),
    lambda a: a["FLY"]["per_fly"][2]["eval_battles"][0].__setitem__("won", float("nan")),
    lambda a: a["FLY"]["per_fly"][2]["eval_battles"][0].__setitem__("won", 1),
    lambda a: a["FLY"]["per_fly"][2]["eval_battles"][0].pop("won"),
    lambda a: a["RND"]["per_fly"][1].__setitem__("eval_battles", a["RND"]["per_fly"][1]["eval_battles"][:-1]),
    lambda a: a["RS"]["per_fly"][1].__setitem__("residual_frac", float("nan")),
    lambda a: a["FLY"].__setitem__("complete", False),
    lambda a: a["FLY"]["per_fly"][0].__setitem__("invalid", "no"),
], ids=["missing-arm", "empty-arm", "missing-row", "duplicate-row", "nan-won", "int-won", "no-won",
        "short-eval", "nan-residual", "incomplete", "non-bool-invalid"])
def test_malformed_input_is_invalid(break_it):
    arms = five(n_flies=8, n=40)
    for d in arms.values():
        d.update(flies=8, eval=40, complete=True)
    assert stats.m4_verdict(arms, SPEC)["status"] == stats.OK
    break_it(arms)
    v = stats.m4_verdict(arms, SPEC)
    assert v["status"] == stats.INVALID and v["reasons"]
    assert v["2a"]["pass"] is None and v["2b"]["pass"] is None
    assert v["2a"]["verdict"] == v["2b"]["verdict"] == stats.INVALID


def test_different_fly_counts_invalid():
    arms = five()
    arms["MAX"] = arm([0.47] * 10, seed=5)
    assert stats.m4_verdict(arms, SPEC)["status"] == stats.INVALID


def test_recorded():
    rec = stats.m4_verdict(five(), SPEC)["recorded"]
    wr = rec["win_rate"]
    assert rec["max_minus_fly"] == pytest.approx(wr["MAX"] - wr["FLY"])
    assert rec["fly_rnd_ratio"] == pytest.approx(wr["FLY"] / wr["RND"])
    assert rec["fly_max_ratio"] == pytest.approx(wr["FLY"] / wr["MAX"])


# ---- information-turn match (recorded only) -------------------------------------------------------
def test_info_turn_match():
    recs = [dict(kind="decision", decider="fly", candidates=["a", "b"], multipliers=[2.0, 1.0], chosen="a"),   # hit
            dict(kind="decision", decider="fly", candidates=["a", "b", "c"], multipliers=[1.0, 0.5, 2.0], chosen="b"),
            dict(kind="decision", decider="fly", candidates=["a", "b"], multipliers=[1.0, 1.0], chosen="b")]   # no info
    assert stats.info_turn_match(recs) == pytest.approx(0.5)
    s = stats.info_turn_stats(recs + [dict(kind="decision", decider="coach", candidates=[], chosen="x"),
                                      dict(kind="decision", decider="fly", candidates=["a", "b"], chosen="a")])
    assert (s["n_info"], s["n_fly_turns"], s["n_without_multipliers"]) == (2, 4, 1)
    assert s["info_frac"] == pytest.approx(2 / 3)
    assert stats.info_turn_match([]) is None


# ---- the CLI ---------------------------------------------------------------------------------------
def load(dirty=False):
    spec = importlib.util.spec_from_file_location("wm4", ROOT / "scripts/write_rescope_m4.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.git_provenance = lambda files=(): {"commit": "c" * 40, "dirty": dirty, "dirty_files": ["x.py"] if dirty else []}
    return mod


def test_cli_refuses_dirty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    with pytest.raises(SystemExit):
        load(dirty=True).main(["--phase", "judge"])
    assert load(dirty=True).main(["--phase", "judge", "--allow-dirty"]) == 0


def donor_log(root, k) -> str:
    """Write FLY k's learning-block log (the RS donor) and return its sha256."""
    import hashlib
    f = root / "FLY" / "logs" / f"fly{k:02d}.jsonl"; f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(kind="reinforce", fly=k, pulses=[["PAM08", 400.0]])) + "\n")
    return hashlib.sha256(f.read_bytes()).hexdigest()


def lay_out(tmp_path, phase="judge", smoke=False, n_flies=8, n=40, **kw):
    root = tmp_path / ("results/rescope-smoke" if smoke else "results/rescope") / phase
    sha = {k: donor_log(root, k) for k in range(n_flies)}
    for name, d in five(n_flies, n, **kw).items():
        d.update(arm=name, phase=phase, smoke=smoke, complete=True, flies=n_flies, eval=n,
                 schedule_digests={"learn": "l" * 64 if name in ("FLY", "RS") else None, "eval": "e" * 64})
        if name == "FLY":
            for r in d["per_fly"]:
                r.update(learn_log_sha256=sha[r["fly"]])
        if name == "RS":
            for r in d["per_fly"]:
                r.update(donor_invalid=False, residual_frac=0.01, donor_sha256=sha[r["fly"]])
        p = root / name / "result.json"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    (root / "eval_schedule.json").write_text(json.dumps({"digest": "e" * 64, "schedule": []}))
    return root


def test_cli_judge_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path)
    log = tmp_path / "results/rescope/judge/FLY/logs/eval/fly00.jsonl"; log.parent.mkdir(parents=True)
    log.write_text(json.dumps(dict(kind="decision", decider="fly", candidates=["a", "b"], multipliers=[2, 1],
                                   chosen="a")) + "\n")
    assert load().main(["--phase", "judge", "--allow-dirty"]) == 0
    s = json.loads((tmp_path / "results/summary/rescope_m4.json").read_text())
    assert s["status"] == "OK" and s["2a"]["verdict"] == "PASS" and s["2b"]["verdict"] == "FAIL"
    assert s["arms"]["RS"]["rs"]["0"]["donor_sha256"] == stats._file_sha(
        tmp_path / "results/rescope/judge/FLY/logs/fly00.jsonl") and s["arms"]["FLY"]["n_eval"] == 40
    assert s["donor_mismatches"] == {}
    assert s["recorded"]["info_turn_match"]["FLY"]["rate"] == 1.0
    rec = s["recorded"]
    assert rec["fly_decision_fraction"]["FLY"]["eval_only"]["frac"] == 1.0
    assert set(rec["fly_decision_fraction"]["FLY"]) >= {"eval_only", "learn_only", "both_blocks"}
    assert set(rec["learn_curve"]) == {"FLY", "RS"} and rec["rs_mismatch"]["per_fly"]["0"]["residual_frac"] == 0.01
    assert set(rec["final_weight_median_ratio"]) == {"FLY", "RS", "COFF"}
    assert s["boot_draws"] == SPEC.boot_draws and s["boot_seed"] == SPEC.boot_seed
    with pytest.raises(SystemExit):                              # the judged file is not overwritten silently
        load().main(["--phase", "judge", "--allow-dirty"])
    assert not (tmp_path / "results/rescope-smoke").exists()


def test_cli_smoke_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, phase="pilot", smoke=True, n_flies=2, n=2)
    load().main(["--smoke", "--phase", "pilot", "--allow-dirty"])
    assert (tmp_path / "results/rescope-smoke/summary/rescope_m4.json").exists()
    assert not (tmp_path / "results/summary").exists()


def test_cli_refuses_pilot_without_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay_out(tmp_path, phase="pilot")
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--allow-dirty"])


@pytest.mark.parametrize("break_it", ["missing", "digest", "schedule-file", "phase", "unreadable", "row-no-fly",
                                      "learn-records-int", "rs-row-no-fly", "row-not-object", "flies-str"])
def test_cli_bad_input_writes_invalid(tmp_path, monkeypatch, break_it):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    p = root / "COFF" / "result.json"
    if break_it == "missing":
        p.unlink()
    elif break_it == "unreadable":
        p.write_text("{not json")
    elif break_it == "schedule-file":
        (root / "eval_schedule.json").write_text(json.dumps({"digest": "f" * 64}))
    elif break_it in ("row-no-fly", "learn-records-int", "rs-row-no-fly", "row-not-object", "flies-str"):
        q = root / ("RS" if break_it == "rs-row-no-fly" else "FLY") / "result.json"
        d = json.loads(q.read_text())
        if break_it in ("row-no-fly", "rs-row-no-fly"):
            d["per_fly"][1].pop("fly")
        elif break_it == "learn-records-int":
            d["per_fly"][1]["learn_records"] = [1]
        elif break_it == "row-not-object":
            d["per_fly"][1] = 1
        else:
            d["flies"] = "8"
        q.write_text(json.dumps(d))
    else:
        d = json.loads(p.read_text())
        if break_it == "digest":
            d["schedule_digests"]["eval"] = "f" * 64
        else:
            d["phase"] = "pilot"
        p.write_text(json.dumps(d))
    assert load().main(["--phase", "judge", "--allow-dirty"]) == 2
    s = json.loads((tmp_path / "results/summary/rescope_m4.json").read_text())
    assert s["status"] == "INVALID" and s["2a"]["pass"] is None and s["2b"]["pass"] is None
    assert s["2a"]["verdict"] == s["2b"]["verdict"] == "INVALID"


# ---- the other recorded items -------------------------------------------------------------------------
def test_learn_curve_rs_mismatch_medians_and_decision_fraction():
    res = {"per_fly": [
        {"fly": 0, "invalid": False, "learn_records": [{"won": True}, {"won": False}, {"won": None}],
         "yoke": {"dropped_bundles": 2, "exhausted_turns": 0, "delivered_bundles": 5, "n_bundles": 7},
         "residual_frac": 0.01, "final_weight_median_ratio": 0.93},
        {"fly": 1, "invalid": False, "learn_records": [{"won": True}, {"won": True}, {"won": False}],
         "yoke": {"dropped_bundles": 0, "exhausted_turns": 3, "delivered_bundles": 4, "n_bundles": 4},
         "residual_frac": 0.0, "final_weight_median_ratio": 1.02},
        {"fly": 2, "invalid": True, "learn_records": [{"won": True}] * 3}]}
    c = stats.learn_curve(res, SPEC, bin_size=2)
    assert c["per_battle"] == [1.0, 0.5, 0.0] and c["bins"] == [0.75, 0.0] and c["n_flies"] == 2
    assert stats.learn_curve({"per_fly": [{"fly": 0, "invalid": False}]}) is None
    m = stats.rs_mismatch(res)
    assert (m["dropped_bundles"], m["exhausted_turns"]) == (2, 3) and m["per_fly"][1]["exhausted_turns"] == 3
    assert stats.weight_medians(res) == {0: 0.93, 1: 1.02, 2: None}
    d = stats.decision_fraction([dict(kind="decision", decider="fly"), dict(kind="decision", decider="coach"),
                                 dict(kind="outcome")])
    assert d == dict(n_decisions=2, n_fly=1, frac=0.5)


def test_arm_per_fly_carries_learn_records_and_medians(tmp_path):
    from flymon.rescope import blocks
    from .test_run_battles import run_fake
    learn, ev = blocks.block_schedule(2, 3, 101, "PL"), blocks.block_schedule(2, 2, 102, "PE")
    run, *_ = run_fake(tmp_path / "FLY", learn=learn, eval_=ev, n=2)
    rows = blocks.arm_per_fly(tmp_path / "FLY", n_flies=2, eval_=ev, run=run, learn=learn, spec=SPEC,
                              weight_medians={0: 0.9, 1: 1.1})
    assert [r["battle_id"] for r in rows[1]["learn_records"]] == blocks.ids_of(learn, 1)
    assert sum(b["won"] is True for b in rows[0]["learn_records"]) == rows[0]["learn_wins"]
    assert [r["final_weight_median_ratio"] for r in rows] == [0.9, 1.1]


def _battle(my_species, opp_species):
    import logging
    from poke_env.battle import Battle
    from flymon.battle.moves import gen1
    from flymon.battle.pool import POOL, by_species
    stats_ = {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}
    my = by_species[my_species]
    bench = [m for m in POOL if m.species != my.species][:2]
    side = [{"ident": f"p1: {m.species}", "details": f"{m.species}, L100", "condition": "300/300", "active": i == 0,
             "stats": stats_, "moves": [gen1(x).id for x in m.attacks + m.support], "baseAbility": "none",
             "item": "", "pokeball": "pokeball"} for i, m in enumerate([my] + bench)]
    active = {"moves": [{"move": x, "id": gen1(x).id, "pp": 16, "maxpp": 16, "target": "normal", "disabled": False}
                        for x in my.attacks + my.support]}
    b = Battle("battle-gen1ou-t", "p1", logging.getLogger("t"), gen=1)
    b.parse_message(["", "switch", f"p2a: {opp_species}", f"{opp_species}, L100", "300/300"])
    b.parse_request({"side": {"name": "p1", "id": "p1", "pokemon": side}, "rqid": 2, "active": [active]})
    return b


def test_eval_player_decision_carries_type_multipliers(tmp_path):
    """On a fake battle the eval player's fly decision record carries each candidate's damage_multiplier against the
    opponent's active Pokemon, still passes logschema, and feeds info_turn_match."""
    import asyncio
    from poke_env.ps_client import AccountConfiguration
    from flymon.agent import logschema
    from flymon.agent.screen import ScreenTable
    from flymon.battle.barrier import BatchBarrier
    from flymon.battle.coach import Coach
    from flymon.battle.router import candidates
    from flymon.rescope import blocks
    from .test_players import FakeEncoder

    async def decide(reqs):
        for r in reqs:
            n = len(r.candidates)
            r.context["detail"] = {"v": [0.0] * n, "a": [0] * n, "p": [0] * n, "kc_active": [0] * n, "tau": 1.0,
                                   "seed": 1}
        return [0] * len(reqs)

    async def batch(reqs):
        return [0] * len(reqs)

    async def go():
        p = blocks.EvalPlayer(fly=0, encoder=FakeEncoder(), table=ScreenTable.allow_all(),
                              rbarrier=BatchBarrier(batch, deadline_ms=5), coach=Coach(),
                              barrier=BatchBarrier(decide, deadline_ms=5), log_path=tmp_path / "fly00.jsonl",
                              account_configuration=AccountConfiguration("fm-eval-mult", None),
                              battle_format="gen1ou", start_listening=False)
        p.start_battle("PE-f00-b000", 0)
        battle = _battle("Blastoise", "Charizard")
        await p.choose_move(battle)
        return battle

    battle = asyncio.run(go())
    recs = blocks.read_jsonl(tmp_path / "fly00.jsonl")
    for r in recs:
        logschema.validate(r)
    d = [r for r in recs if r["kind"] == "decision"][-1]
    assert d["decider"] == "fly"
    opp = battle.opponent_active_pokemon
    want = [float(opp.damage_multiplier(m)) for m in candidates(battle)]
    assert d["multipliers"] == want and len(want) == len(d["candidates"]) >= 2
    assert max(want) == 2.0                                    # Blastoise's water move vs Charizard
    rate = stats.info_turn_match([d])
    assert rate == (1.0 if want[0] == max(want) else 0.0) and rate is not None


def test_cli_pair_shortage_writes_judged_2a_and_invalid_2b(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    p = root / "RS" / "result.json"
    d = json.loads(p.read_text())
    for k in (0, 1, 2):
        d["per_fly"][k]["donor_invalid"] = True
    p.write_text(json.dumps(d))
    assert load().main(["--phase", "judge", "--allow-dirty"]) == 2
    s = json.loads((tmp_path / "results/summary/rescope_m4.json").read_text())
    assert s["status"] == "OK" and s["2a"]["verdict"] == "PASS" and s["2b"]["verdict"] == "INVALID"
    assert s["invalid_pairs"]["vs_rs"]["flies"] == [0, 1, 2]


def test_recorded_helpers_skip_malformed_rows():
    res = {"per_fly": [1, {"invalid": False}, {"fly": "0"}, {"fly": 2, "invalid": False, "learn_records": [1, "x"],
                                                              "yoke": 5, "final_weight_median_ratio": 0.9}]}
    assert stats.learn_curve(res) is None and stats.weight_medians(res) == {2: 0.9}
    assert stats.rs_mismatch(res)["per_fly"][2]["dropped_bundles"] is None
    assert stats.learn_curve({"per_fly": "x"}) is None and stats.fly_rows(None) == []
    assert stats.info_turn_stats([1, None, dict(kind="decision", decider="fly", candidates="ab", multipliers=[1],
                                                chosen="a")])["n_without_multipliers"] == 1
    assert stats.decision_fraction([1, dict(kind="decision", decider="fly")])["n_fly"] == 1


# ---- the FLY <-> RS donor trace (spec 10.8; final-review finding 2) -----------------------------------------
@pytest.mark.parametrize("spoil", ["rs_sha", "fly_sha", "log_changed", "learn_digest"])
def test_cli_donor_mismatch_makes_that_vs_rs_pair_invalid(tmp_path, monkeypatch, spoil):
    monkeypatch.chdir(tmp_path)
    root = lay_out(tmp_path)
    if spoil == "log_changed":                                   # FLY 1 rerun after RS read its log
        (root / "FLY" / "logs" / "fly01.jsonl").write_text("{}\n")
    else:
        arm = "FLY" if spoil == "fly_sha" else "RS"
        p = root / arm / "result.json"
        d = json.loads(p.read_text())
        if spoil == "rs_sha":
            d["per_fly"][1]["donor_sha256"] = "0" * 64
        elif spoil == "fly_sha":
            d["per_fly"][1]["learn_log_sha256"] = "0" * 64
        else:
            d["schedule_digests"]["learn"] = "m" * 64
        p.write_text(json.dumps(d))
    rc = load().main(["--phase", "judge", "--allow-dirty"])
    s = json.loads((tmp_path / "results/summary/rescope_m4.json").read_text())
    if spoil == "learn_digest":                                  # every pair loses its trace: (2b) INVALID
        assert rc == 2 and s["2b"]["verdict"] == "INVALID" and sorted(s["donor_mismatches"]) == [str(k) for k in range(8)]
        assert s["invalid_pairs"]["vs_rs"]["flies"] == list(range(8))
    else:                                                        # one pair out, 7 >= 6 valid: still judged
        assert rc == 0 and list(s["donor_mismatches"]) == ["1"] and "donor mismatch" in s["donor_mismatches"]["1"]
        assert s["invalid_pairs"]["vs_rs"]["flies"] == [1] and s["2b"]["verdict"] in ("PASS", "FAIL")
        assert any("vs_rs pair 1: donor mismatch" in r for r in s["2b"]["reasons"])
    assert s["invalid_pairs"]["vs_coff"]["flies"] == [] and s["2a"]["verdict"] == "PASS"


def test_donor_mismatches_pure(tmp_path):
    root = tmp_path / "r"
    sha = donor_log(root, 0)
    fly = {"per_fly": [{"fly": 0, "learn_log_sha256": sha}], "schedule_digests": {"learn": "l"}}
    rs = {"per_fly": [{"fly": 0, "donor_sha256": sha}], "schedule_digests": {"learn": "l"}}
    assert stats.donor_mismatches(fly, rs, root / "FLY" / "logs") == {}
    assert set(stats.donor_mismatches(fly, dict(rs, schedule_digests={"learn": "x"}), root / "FLY" / "logs")) == {0}
    marked = stats.mark_donor_mismatches(rs, {0: "why"})
    assert stats.row_invalid(marked["per_fly"][0]) and not stats.row_invalid(rs["per_fly"][0])


# ---- residual fix: NoBrainPlayer keeps its _write (fly / battle_id on every record) -------------------------
def test_nobrain_player_defines_write():
    import ast
    import inspect
    from flymon.rescope import blocks
    assert "_write" in vars(blocks.NoBrainPlayer)
    tree = ast.parse(inspect.getsource(blocks))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NoBrainPlayer")
    assert {f.name for f in cls.body if isinstance(f, ast.FunctionDef)} >= {"__init__", "start_battle", "_write"}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rnd_seed")
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn]


@pytest.mark.parametrize("kind", ["RND", "MAX"])
def test_nobrain_decision_record_carries_fly_and_battle_id(tmp_path, kind):
    """Through the real player path (choose_move -> _log -> _write): the record carries fly and battle_id, so
    filter_log keeps it on --resume and the retry rollback can divert it."""
    import asyncio
    from poke_env.ps_client import AccountConfiguration
    from flymon.agent.checkpoint import filter_log
    from flymon.battle.coach import Coach
    from flymon.battle.providers import MaxDamageProvider, RandomProvider
    from flymon.rescope import blocks

    log = tmp_path / "fly02.jsonl"
    p = blocks.NoBrainPlayer(2, phase="judge", provider=RandomProvider(seed=2) if kind == "RND" else MaxDamageProvider(),
                             coach=Coach(), barrier=None, log_path=log,
                             account_configuration=AccountConfiguration(f"fm-nb-{kind}", None),
                             battle_format="gen1ou", start_listening=False)
    p.start_battle("JE-f02-b004", 4)
    asyncio.run(p.choose_move(_battle("Blastoise", "Charizard")))
    recs = blocks.read_jsonl(log)
    assert recs and all(r["fly"] == 2 and r["battle_id"] == "JE-f02-b004" for r in recs)
    assert any(r["kind"] == "decision" for r in recs)
    filter_log(log, {"JE-f02-b004"})                          # a committed battle's records survive a resume
    assert blocks.read_jsonl(log) == recs
