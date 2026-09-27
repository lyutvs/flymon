"""Spec 10.2's primary protocol on a fake pool: layout (no Rp, with N'), DANs and seeds per arm, step-wise resume."""
import dataclasses

import numpy as np
import pytest

from flymon.brain.b_runner import fly_specs
from flymon.rescope import primary, rules
from flymon.rescope.spec import SPEC

SMALL = dataclasses.replace(SPEC, n_flies=2, n_probe=2, trials=3, valid_min=1)
ODORS = {"a": {"ORN_DM1": 1.0}, "b": {"ORN_VA2": 1.0}}
CELLS = {SPEC.a_type: np.array([0, 2]), SPEC.p_type: np.array([1, 3])}
QUIET = lambda s: None


class FakePool:
    """Copied from tests/brain/test_b_runner.py:18 (unchanged). Counts are a deterministic function of the fly's
    weights, the odour and the seed; a reinforcement with a DAN on an enabled fly scales its weights by 0.9 (0.95 for
    the punishment DAN)."""

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


def fake(spec=SMALL, fail_at=None):
    # the copied FakePool takes fly specs (b_runner.fly_specs of the layout), not a count
    return FakePool(fly_specs(primary.layout(spec)), fail_at=fail_at)


def test_layout_has_no_rp_and_has_n2():
    lay = primary.layout(SMALL)
    assert [b for b, _ in lay] == ["Rr", "Rr", "N", "N", "N2", "N2", "noplast"]
    assert [f for _, f in lay] == [0, 1, 0, 1, 0, 1, 0]
    assert [f["enabled"] for f in fly_specs(lay)] == [True] * 6 + [False]
    assert len(primary.layout(SPEC)) == 3 * SPEC.n_flies + 1


def test_run_pair_dans_and_seeds():
    pool = fake()
    out = primary.run_pair(pool, SMALL, "p1000", ODORS, CELLS, log=QUIET)
    lay = out["layout"]
    want = {"Rr": "PAM08", "noplast": "PAM08", "N": None, "N2": None}
    assert all(dan == want[lay[i][0]] for (i, dan, _, _) in pool.reinforced)
    assert {lay[i][0] for (i, _, _, _) in pool.reinforced} == {"Rr", "N", "N2", "noplast"}
    assert all(dan != SPEC.punish_dan for (_, dan, _, _) in pool.reinforced)
    n2_seeds = {s for (i, _, _, s) in pool.reinforced if lay[i][0] == "N2"}
    assert min(n2_seeds) >= SPEC.train_base_n2
    seeds_of = lambda b, f: [s for i, _, _, s in pool.reinforced if lay[i] == (b, f)]
    assert seeds_of("Rr", 1) == seeds_of("N", 1) == [SMALL.train_seed("p1000", 1, t) for t in range(3)]
    assert seeds_of("N2", 1) == [SMALL.train_seed("p1000", 1, t, second_null=True) for t in range(3)]
    assert {r["stage"] for r in out["records"]} == {"pre", "S1"}
    assert {r["seed"] for r in out["records"] if r["fly"] == 1} == set(SMALL.probe_seeds("p1000", 1))
    assert out["x"] == rules.choose_x(out["records"], SMALL)
    assert out["replayed"] is False


def test_resume_gives_same_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from flymon.brain.config import Params
    from flymon.rescope.store import Checkpoint
    whole = fake()
    full = primary.run_pair(whole, SMALL, "p1000", ODORS, CELLS, log=QUIET)
    ck = Checkpoint("results/rescope/p1000", "k", [Params()])
    with pytest.raises(RuntimeError):
        primary.run_pair(fake(fail_at=2), SMALL, "p1000", ODORS, CELLS, checkpoint=ck, log=QUIET,
                         provenance={"commit": "first"})
    fresh = fake()
    resumed = primary.run_pair(fresh, SMALL, "p1000", ODORS, CELLS, checkpoint=ck, log=QUIET,
                               provenance={"commit": "second"})
    assert resumed["records"] == full["records"] and resumed["x"] == full["x"]
    assert all(np.array_equal(fresh.w[i], whole.w[i]) for i in whole.w)
    assert fresh.n_reinforce == 2 and resumed["provenance"] == {"commit": "first"}
    again = primary.run_pair(fake(), SMALL, "p1000", ODORS, CELLS, checkpoint=ck, log=QUIET)
    assert again["replayed"] is True and again["records"] == full["records"]
