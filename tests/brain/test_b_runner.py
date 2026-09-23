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
    assert lay == [("R", 0), ("R", 1), ("N", 0), ("N", 1), ("N2", 0), ("N2", 1), ("noplast", 0)]
    assert [f["enabled"] for f in fly_specs(lay)] == [True] * 6 + [False]
    assert steps(SMALL) == ["pre", "reward:0", "reward:1", "reward:2", "S1", "punish:0", "punish:1", "punish:2", "S2"]
    assert len(layout(SPEC, False)) == 17 and len(layout(SPEC, True)) == 25


def test_a_run_records_every_declared_cell_and_gives_the_dan_only_to_r_and_noplast():
    lay = layout(SMALL, True)
    pool = FakePool(fly_specs(lay))
    out = run_pair(pool, SMALL, "calibration", ODORS, CELLS, True, None, log=lambda s: None)
    seeds = lambda f: SMALL.probe_seeds("calibration", f)
    assert B.check_records(out["records"], ("R", "N", "N2", "noplast"), SMALL, seeds) == []
    brain = {i: b for i, (b, _) in enumerate(lay)}
    assert {(brain[i], d) for i, d, *_ in pool.reinforced} == {("R", "PAM08"), ("R", "PPL105"), ("noplast", "PAM08"),
                                                               ("noplast", "PPL105"), ("N", None), ("N2", None)}
    r0 = [s for i, d, p, s in pool.reinforced if i == 0]
    n2 = [s for i, d, p, s in pool.reinforced if brain[i] == "N2" and lay[i][1] == 0]
    assert r0 == [SMALL.train_seed("calibration", 0, t) for t in range(6)]            # punishment trials are 3..5
    assert n2 == [SMALL.train_seed("calibration", 0, t, second_null=True) for t in range(6)]
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
    broken = FakePool(fly_specs(lay), fail_at=5)                                     # dies in the punishment block
    with pytest.raises(RuntimeError):
        run_pair(broken, SMALL, "calibration", ODORS, CELLS, True, None, checkpoint=ck, log=lambda s: None)
    fresh = FakePool(fly_specs(lay))
    got = run_pair(fresh, SMALL, "calibration", ODORS, CELLS, True, None, checkpoint=ck, log=lambda s: None)
    assert got["records"] == ref["records"] and got["x"] == ref["x"]
    assert all(np.array_equal(fresh.w[i], whole.w[i]) for i in whole.w)
    assert fresh.n_reinforce == 6 - 4                                                # only the two trials not yet done


def test_a_checkpoint_from_other_code_is_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lay = layout(SMALL, False)
    Checkpoint("results/b/x/ck", "old", [Params()]).save(dict(records=[], x="a", done=["pre"],
                                                              weights=FakePool(fly_specs(lay)).state()))
    pool = FakePool(fly_specs(lay))
    run_pair(pool, SMALL, "exploration", ODORS, CELLS, False, "b", checkpoint=Checkpoint("results/b/x/ck", "new", [Params()]),
             log=lambda s: None)
    assert pool.n_reinforce == 6                                                     # started over


def test_the_store_writes_only_its_declared_paths(tmp_path, monkeypatch):
    from flymon.brain.b_store import write_json
    monkeypatch.chdir(tmp_path)
    for ok in ("results/b/x.json", "results/b-smoke/x.json", "results/summary/b_calibration.json",
               "results/summary/b_test.json"):
        write_json(ok, {"a": 1}, [Params()])
    for bad in ("results/summary/m0d.json", "results/m2/x.json", "x.json"):
        with pytest.raises(SystemExit):
            write_json(bad, {"a": 1}, [Params()])
