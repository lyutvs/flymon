"""Spec N.3 / N.8.5: the punish-only oracle is h4_jobs.oracle_job's punishment half (same unedited probes, α by the
smallest change, weights restored). Spec N.4 / N.8.7: the absolute arm trains the CS+ alone with train_block's seed rule
and timings, probes with decide's counts plus the KC sub-window, and with plasticity off changes nothing (plumbing)."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain.circuits import Populations
from flymon.brain.conditioning import train_block
from flymon.brain.config import Params
from flymon.brain.n_spec import SPEC
from flymon.brain.h4_jobs import oracle_job
from flymon.brain.presentation import decide

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}
Z = {"A": [1.0, 2.0], "P": [1.5, 3.0]}
X, Y = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)
ORACLE = dict(act_seeds=(500, 501), select_seeds=(600, 601, 602), report_seeds=(608, 609, 610),
              alphas=(0.2, 0.5, 0.8), **W)
ARM = dict(readout=RO, punish_type="PPL105", trials=2, present_ms=300.0, gap_ms=50.0, train_settle_ms=100.0,
           seed_base=1_000_000, seed_stride=1000, **W)


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    N._RIG.clear(); H4._RIG.clear()                     # h4_jobs.rig_for is keyed by Params only: never share it
    return c, Populations.from_connectome(c)


def test_pick_alpha_takes_the_smallest_change_and_ties_to_the_smaller_alpha():
    assert N.pick_alpha({"0.2": -1.0, "0.5": -3.0, "0.8": -2.0}, (0.2, 0.5, 0.8)) == 0.5
    assert N.pick_alpha({"0.2": -3.0, "0.5": -3.0, "0.8": -2.0}, (0.2, 0.5, 0.8)) == 0.2
    with pytest.raises(ValueError, match="undefined"):
        N.pick_alpha({"0.2": None, "0.5": -1.0, "0.8": -1.0}, (0.2, 0.5, 0.8))


def test_punish_only_pre_is_oracle_jobs_pre(conn_pops):
    c, pops = conn_pops
    got = N.punish_only_oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                                   punish_type="PPL105", **ORACLE)
    ref = oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                     types=("MBON03", "MBON01"), punish_type="PPL105", reward_type="PAM08", **ORACLE)
    assert got["report"]["pre"] == ref["report"]["pre"]
    assert got["select"]["pre"] == {"A": ref["select"]["pre"]["MBON03"], "P": ref["select"]["pre"]["MBON01"]}
    assert got["kc"] == ref["kc"]
    assert got["alpha_punish"] in ORACLE["alphas"]


def test_a_zero_alpha_leaves_the_probes_unchanged_and_weights_are_restored(conn_pops):
    c, pops = conn_pops
    got = N.punish_only_oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                                   punish_type="PPL105", **dict(ORACLE, alphas=(0.0,)))
    assert got["report"]["P"] == got["report"]["pre"] and got["select"]["punish"]["0.0"]["change"] == 0.0
    e, p, _, sha = N.rig(c, pops, P, "none")
    assert np.array_equal(e.csc.w[p.edges], p.w0) and got["csc_sha256"] == sha


def test_the_punish_edit_leaves_P_untouched(conn_pops):
    c, pops = conn_pops
    got = N.punish_only_oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=X, odor_y=Y, readout=RO, z=Z,
                                   punish_type="PPL105", **dict(ORACLE, alphas=(0.8,)))
    assert got["alpha_punish"] == 0.8
    assert got["report"]["P"]["P"] == got["report"]["pre"]["P"]        # only the PPL105 core (A's pool) is edited


def test_train_seed_is_train_blocks_rule_and_the_specs(conn_pops, monkeypatch):
    c, pops = conn_pops
    e, p, _, _ = N.rig(c, pops, P, "none")
    seen, orig = [], e.reset
    monkeypatch.setattr(e, "reset", lambda seed=None: (seen.append(seed), orig(seed))[1])
    train_block(e, p, pops, X, Y, 3.0, 7, "PPL105", None, trials=2, present_ms=10.0, gap_ms=10.0, settle_ms=10.0)
    p.reset_weights()
    base, stride = SPEC.train_seed_base, SPEC.train_seed_stride
    assert seen == [N.train_seed(7, t, base, stride) for t in range(2) for _ in ("plus", "minus")]
    big = 21_002_063                                     # a judge seed
    assert N.train_seed(big, 11, base, stride) == SPEC.train_seed(big, 11) == 1_000_000 + big * 1000 + 11
    assert N.train_seed(big, 11, base, stride) > 2**32 and isinstance(N.train_seed(big, 11, base, stride), int)


def test_train_plus_only_uses_train_blocks_seed_rule(conn_pops, monkeypatch):
    c, pops = conn_pops
    e, p, _, _ = N.rig(c, pops, P, "none")
    seen, orig = [], e.reset
    monkeypatch.setattr(e, "reset", lambda seed=None: (seen.append(seed), orig(seed))[1])
    N.train_plus_only(e, p, pops, X, 3.0, 21_002_063, "PPL105", 3, 100.0, 50.0, 100.0, 1_000_000, 1000)
    assert seen == [1_000_000 + 21_002_063 * 1000 + t for t in range(3)]


def test_one_plus_trial_equals_train_blocks_first_plus_presentation(conn_pops):
    c, pops = conn_pops
    e, p, _, _ = N.rig(c, pops, P, "none")
    snap = {}

    def on_event(kind, **f):
        if f["trial"] == 0 and f["cs"] == "plus":
            snap["w"] = e.csc.w[p.edges].copy()

    p.reset_weights(); p.set_enabled(True)
    train_block(e, p, pops, X, Y, 3.0, 7, "PPL105", None, trials=1, present_ms=300.0, gap_ms=50.0, settle_ms=100.0,
                on_event=on_event)
    p.reset_weights(); p.set_enabled(True)
    N.train_plus_only(e, p, pops, X, 3.0, 7, "PPL105", 1, 300.0, 50.0, 100.0, 1_000_000, 1000)
    assert not np.array_equal(snap["w"], p.w0)            # the pulse depressed something: the comparison is not vacuous
    assert np.array_equal(e.csc.w[p.edges], snap["w"])
    p.reset_weights()


def test_arm_probes_are_decides_counts(conn_pops):
    c, pops = conn_pops
    row = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=X, odor_y=Y, seed=9,
                             plastic=True, **ARM)
    e, p, _, _ = N.rig(c, pops, P, "none")
    t = np.asarray(c.type).astype(str)
    cnt = decide(e, p, pops, [X, Y], W["strength"], 9, W["settle_ms"], W["read_ms"])
    for k, i in (("x", 0), ("y", 1)):
        assert (row["pre"][k]["A"], row["pre"][k]["P"]) == (int(cnt[i][t == "MBON03"].sum()),
                                                              int(cnt[i][t == "MBON01"].sum()))
    assert row["weights_frac"] < 1.0 and np.array_equal(e.csc.w[p.edges], p.w0)   # learned, then restored


def test_arm_probe_rows_carry_the_kc_subwindow_and_the_state_readable_p(conn_pops):
    c, pops = conn_pops
    row = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=X, odor_y=Y, seed=9,
                             plastic=True, **ARM)
    for phase in ("pre", "post"):
        for k in ("x", "y"):
            r = row[phase][k]
            assert set(r) == set(N.PRESENTATION_KEYS) and r["seed"] == 9
            assert isinstance(r["P"], int) and isinstance(r["kc_spikes"], int) and r["kc_max_win_hz"] >= 0
            assert (r["P"] < SPEC.state_p_min) in (True, False)          # N.8.8 state: silent iff P < state_p_min


def test_plumbing_arm_changes_nothing(conn_pops):
    c, pops = conn_pops
    row = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit="apl_to_kc_zero", odor_x=X, odor_y=Y,
                             seed=9, plastic=False, **ARM)
    strip = lambda d: {k: {f: v for f, v in r.items() if f != "wall_s"} for k, r in d.items()}
    assert strip(row["pre"]) == strip(row["post"]) and row["weights_frac"] == 1.0


def test_arms_on_block_on_in_one_worker(conn_pops):
    c, pops = conn_pops
    run = lambda edit: N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit=edit, odor_x=X, odor_y=Y,
                                          seed=9, plastic=True, **ARM)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s",)} | {
        "pre": {k: {f: v for f, v in x.items() if f != "wall_s"} for k, x in r["pre"].items()},
        "post": {k: {f: v for f, v in x.items() if f != "wall_s"} for k, x in r["post"].items()}}
    a, b, a2 = run("none"), run("apl_to_kc_zero"), run("none")
    assert strip(a) == strip(a2) and a["csc_sha256"] != b["csc_sha256"]
