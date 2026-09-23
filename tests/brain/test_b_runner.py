"""Spec J.12.2's protocol on a fake pool: layout, who gets which DAN on which seed, and step-wise resume."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import b_rules as B
from flymon.brain.b_runner import fly_specs, layout, run_pair, steps
from flymon.brain.b_spec import SPEC
from flymon.brain.b_store import Checkpoint
from flymon.brain.config import Params

SMALL = dataclasses.replace(SPEC, n_flies=2, n_probe=2, trials=3)
ODORS = {"a": {"ORN_A": 1.0}, "b": {"ORN_B": 1.0}}
CELLS = {SPEC.a_type: np.array([10, 11]), SPEC.p_type: np.array([20, 21])}


class FakePool:
    """Counts are a deterministic function of the fly's weights, the odour and the seed; a reinforcement with a DAN on
    an enabled fly scales its weights by 0.9 (0.95 for the punishment DAN)."""

    def __init__(self, flies, fail_at=None):
        self.flies = [dict(f) for f in flies]
        self.w = {i: np.ones(4, np.float32) for i in range(len(flies))}
        self.reinforced, self.n_reinforce, self.fail_at = [], 0, fail_at

    def decide_batch(self, reqs, strength, settle_ms, read_ms, idx):
        out = []
        for i, cands, seed in reqs:
            base = int(round(20 * float(self.w[i].sum())))
            out.append(np.array([[base + (seed % 5) + 3 * j + c % 2 for c in range(len(idx))] for j in range(len(cands))]))
        return out

    def reinforce_batch(self, reqs, strength, settle_ms, gap_ms):
        self.n_reinforce += 1
        if self.fail_at is not None and self.n_reinforce == self.fail_at:
            raise RuntimeError("worker lost")
        for i, odor, dan, pulse, seed in reqs:
            self.reinforced.append((i, dan, pulse, seed))
            if dan is not None and self.flies[i]["enabled"]:
                self.w[i] = self.w[i] * np.float32(0.9 if dan == SPEC.reward_dan else 0.95)

    def state(self):
        return {"flies": [dict(enabled=f["enabled"], shuffle_seed=None, w=self.w[i].copy()) for i, f in enumerate(self.flies)]}

    def load_state(self, d):
        for i, e in enumerate(d["flies"]):
            self.w[i] = np.asarray(e["w"], np.float32).copy()


def test_layout_and_steps():
    lay = layout(SMALL, with_n2=True)
    assert lay == [("Rr", 0), ("Rr", 1), ("Rp", 0), ("Rp", 1), ("N", 0), ("N", 1), ("N2", 0), ("N2", 1), ("noplast", 0)]
    assert [f["enabled"] for f in fly_specs(lay)] == [True] * 8 + [False]
    assert steps(SMALL) == ["pre", "train:0", "train:1", "train:2", "S1"]
    assert len(layout(SPEC, False)) == 25 and len(layout(SPEC, True)) == 33


def test_a_run_records_every_declared_cell_and_gives_each_arm_its_dan():
    lay = layout(SMALL, True)
    pool = FakePool(fly_specs(lay))
    out = run_pair(pool, SMALL, "calibration", ODORS, CELLS, True, None, log=lambda s: None)
    seeds = lambda f: SMALL.probe_seeds("calibration", f)
    assert B.check_records(out["records"], ("Rr", "Rp", "N", "N2", "noplast"), SMALL, seeds) == []
    brain = {i: b for i, (b, _) in enumerate(lay)}
    assert {(brain[i], d) for i, d, *_ in pool.reinforced} == {("Rr", "PAM08"), ("Rp", "PPL105"), ("noplast", "PAM08"),
                                                               ("N", None), ("N2", None)}
    seeds_of = lambda b, f: [s for i, d, p, s in pool.reinforced if lay[i] == (b, f)]
    assert seeds_of("Rr", 0) == seeds_of("Rp", 0) == seeds_of("N", 0) == [SMALL.train_seed("calibration", 0, t)
                                                                           for t in range(3)]   # paired presentations
    assert seeds_of("N2", 0) == [SMALL.train_seed("calibration", 0, t, second_null=True) for t in range(3)]
    assert all(p == SMALL.pulse_ms for _, _, p, _ in pool.reinforced)
    assert B.noplast_ok(out["records"])                                                # plasticity off: counts never move


def test_x_follows_the_naive_rule_unless_fixed():
    lay = layout(SMALL, False)
    out = run_pair(FakePool(fly_specs(lay)), SMALL, "exploration", ODORS, CELLS, False, "b", log=lambda s: None)
    assert out["x"] == "b"
    out = run_pair(FakePool(fly_specs(lay)), SMALL, "confirmation", ODORS, CELLS, False, None, log=lambda s: None)
    assert out["x"] in ("a", "b")


def test_an_interrupted_run_resumes_to_the_uninterrupted_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay = layout(SMALL, True)
    whole = FakePool(fly_specs(lay))
    ref = run_pair(whole, SMALL, "calibration", ODORS, CELLS, True, None, log=lambda s: None)
    ck = Checkpoint("results/b/calibration/ck", "key1", [Params()])
    broken = FakePool(fly_specs(lay), fail_at=3)                                     # dies in the last trial
    with pytest.raises(RuntimeError):
        run_pair(broken, SMALL, "calibration", ODORS, CELLS, True, None, checkpoint=ck, log=lambda s: None)
    fresh = FakePool(fly_specs(lay))
    got = run_pair(fresh, SMALL, "calibration", ODORS, CELLS, True, None, checkpoint=ck, log=lambda s: None)
    assert got["records"] == ref["records"] and got["x"] == ref["x"]
    assert all(np.array_equal(fresh.w[i], whole.w[i]) for i in whole.w)
    assert fresh.n_reinforce == 1                                                    # only the trial not yet done


def test_a_checkpoint_from_other_code_is_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay = layout(SMALL, False)
    Checkpoint("results/b/x/ck", "old", [Params()]).save(dict(records=[], x="a", done=["pre"],
                                                              weights=FakePool(fly_specs(lay)).state()))
    pool = FakePool(fly_specs(lay))
    run_pair(pool, SMALL, "exploration", ODORS, CELLS, False, "b", checkpoint=Checkpoint("results/b/x/ck", "new", [Params()]),
             log=lambda s: None)
    assert pool.n_reinforce == 3                                                     # started over


def test_the_store_writes_only_its_declared_paths(tmp_path, monkeypatch):
    from flymon.brain.b_store import write_json
    monkeypatch.chdir(tmp_path)
    for ok in ("results/b/x.json", "results/b-smoke/x.json", "results/summary/b_calibration.json",
               "results/summary/b_test.json"):
        write_json(ok, {"a": 1}, [Params()])
    for bad in ("results/summary/m0d.json", "results/m2/x.json", "x.json"):
        with pytest.raises(SystemExit):
            write_json(bad, {"a": 1}, [Params()])


def test_a_resumed_or_replayed_run_reports_the_provenance_of_the_run_that_measured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay = layout(SMALL, False)
    ck = Checkpoint("results/b/exploration/ck", "key1", [Params()])
    first = dict(git=dict(commit="aaa", dirty_hashed=["flymon/brain/b_runner.py"]), started_utc="2026-09-23T01:00:00+00:00")
    with pytest.raises(RuntimeError):
        run_pair(FakePool(fly_specs(lay), fail_at=2), SMALL, "exploration", ODORS, CELLS, False, "b", checkpoint=ck,
                 log=lambda s: None, provenance=first)
    second = dict(git=dict(commit="bbb", dirty_hashed=[]), started_utc="2026-09-23T02:00:00+00:00")
    got = run_pair(FakePool(fly_specs(lay)), SMALL, "exploration", ODORS, CELLS, False, "b", checkpoint=ck,
                   log=lambda s: None, provenance=second)
    assert got["provenance"] == first and not got["replayed"]                      # the dirty first run measured
    third = dict(git=dict(commit="ccc", dirty_hashed=[]), started_utc="2026-09-23T03:00:00+00:00")
    pool = FakePool(fly_specs(lay))
    again = run_pair(pool, SMALL, "exploration", ODORS, CELLS, False, "b", checkpoint=ck, log=lambda s: None,
                     provenance=third)
    assert again["replayed"] and again["provenance"] == first and pool.n_reinforce == 0
    assert again["records"] == got["records"]
    fresh = run_pair(FakePool(fly_specs(lay)), SMALL, "exploration", ODORS, CELLS, False, "b",
                     checkpoint=Checkpoint("results/b/exploration/ck2", "key1", [Params()]), log=lambda s: None,
                     provenance=third)
    assert fresh["provenance"] == third and not fresh["replayed"]


def test_the_calibration_pair_is_the_first_candidate_with_a_strong_naive_punishment_readout():
    """J.12.7: naive probes only; the rule's X must reach calibration_min_a MBON13 spikes (median)."""
    from b_fixtures import records
    from flymon.brain.b_runner import screen_calibration
    naive = {23: 9, 27: 25, 49: 40}
    measure = lambda seed: [r for r in records(naive=dict(ax=naive[seed], px=40, ay=0, py=40), noise=0.0)
                            if r["brain"] == "Rr" and r["stage"] == "pre"]
    got = screen_calibration(SPEC, measure)
    assert got["selected"] == 27 and got["met"] and [r["seed"] for r in got["rows"]] == [23, 27]
    assert got["rows"][0]["naive_a"] == 9.0 and got["rows"][1]["x"] == "b"
    weak = screen_calibration(SPEC, lambda seed: [r for r in records(naive=dict(ax=5, px=40, ay=0, py=40), noise=0.0)
                                                  if r["brain"] == "Rr" and r["stage"] == "pre"])
    assert weak["selected"] == 23 and not weak["met"] and len(weak["rows"]) == 3
