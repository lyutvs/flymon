"""Spec O.7.3: one O training loop with a punish flag. Flag on is n_jobs.absolute_arm_job bit for bit (regression); flag
off skips drive_dan and recover_pulse (train_block's dan-None rule). The dopamine-zero arm holds the rule's dopamine
trace at exactly 0 every step and leaves DAN spikes untouched, and its swap never outlives the job. The phasic-dopamine
integral is recorded per compartment, weights_frac_by_mbon_set for the A·P cores (comps[punish_type].core,
comps[reward_type].core, as conditioning.run_arm); every arm's pre/post weight sha256 (O.7.4-5, O.7.8)."""
import contextlib
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import n_jobs as N
from flymon.brain import o_jobs as O
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.o_spec import SPEC

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
RO = {"A": "MBON03", "P": "MBON01"}                  # synthetic: PPL105 core = MBON03/04, PAM08 core = MBON01/02
X, Y = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)
ARM = dict(readout=RO, punish_type="PPL105", trials=2, present_ms=300.0, gap_ms=50.0, train_settle_ms=100.0,
           seed_base=1_000_000, seed_stride=1000, **W)
FLAGS = {name: dict(punish=pu, plastic=pl, da_zero=dz) for name, pu, pl, dz in SPEC.o2_arms}


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    t = np.asarray(c.type).astype(str)
    apl, mbon = int(np.flatnonzero(t == "APL")[0]), int(np.flatnonzero(t == "MBON01")[0])
    c = replace(c, pre=np.append(c.pre, np.int32(apl)), post=np.append(c.post, np.int32(mbon)),
                w=np.append(c.w, np.int32(20)))
    O._RIG.clear(); N._RIG.clear(); H4._RIG.clear()
    return c, Populations.from_connectome(c)


def _o(c, pops, arm, edit="none", seed=9, **kw):
    args = {**ARM, **kw}
    return O.arm_job(_Stub(c), None, pops, None, None, params=P, edit=edit, odor_x=X, odor_y=Y, seed=seed, arm=arm,
                     reward_type="PAM08", **FLAGS[arm], **args)


def _rows(d):
    return {k: {f: v for f, v in r.items() if f != "wall_s"} for k, r in d.items()}


@pytest.mark.parametrize("edit", ["none", "apl_to_kc_zero"])
def test_punish_arm_is_bit_identical_to_absolute_arm_job(conn_pops, edit):
    c, pops = conn_pops
    o = _o(c, pops, "punish", edit=edit)
    n = N.absolute_arm_job(_Stub(c), None, pops, None, None, params=P, edit=edit, odor_x=X, odor_y=Y, seed=9,
                           plastic=True, **ARM)
    for k in ("seed", "edit", "plastic", "csc_sha256", "weights_frac"):
        assert o[k] == n[k], k
    assert _rows(o["pre"]) == _rows(n["pre"]) and _rows(o["post"]) == _rows(n["post"])
    assert o["weights_frac"] < 1.0 and o["w_post_sha256"] != o["w0_sha256"]      # not vacuous: weights moved


def test_train_x_with_punish_equals_train_plus_only(conn_pops):
    c, pops = conn_pops
    e, p, _, _ = O.rig(c, pops, P, "none")
    p.reset_weights(); p.set_enabled(True)
    N.train_plus_only(e, p, pops, X, 3.0, 7, "PPL105", 2, 300.0, 50.0, 100.0, 1_000_000, 1000)
    ref = e.csc.w[p.edges].copy()
    p.reset_weights(); p.set_enabled(True)
    O.train_x(e, p, pops, X, 3.0, 7, "PPL105", 2, 300.0, 50.0, 100.0, 1_000_000, 1000)
    assert not np.array_equal(ref, p.w0) and np.array_equal(e.csc.w[p.edges], ref)
    p.reset_weights()


def test_train_x_without_punish_never_drives_or_recovers(conn_pops, monkeypatch):
    c, pops = conn_pops
    e, p, _, _ = O.rig(c, pops, P, "none")
    calls = []
    monkeypatch.setattr(p, "drive_dan", lambda *a: calls.append("drive"))
    monkeypatch.setattr(p, "recover_pulse", lambda: calls.append("recover"))
    seen, orig = [], e.reset
    monkeypatch.setattr(e, "reset", lambda seed=None: (seen.append(seed), orig(seed))[1])
    O.train_x(e, p, pops, X, 3.0, 22_001_005, None, 3, 100.0, 50.0, 100.0, 1_000_000, 1000)
    assert calls == [] and seen == [1_000_000 + 22_001_005 * 1000 + t for t in range(3)]


def test_dopamine_zero_holds_the_trace_at_zero_and_leaves_dan_spikes(conn_pops):
    c, pops = conn_pops
    e, p, _, _ = O.rig(c, pops, P, "none")
    dan = p.types["PPL105"][0]
    before = {k: (cells.copy(), w.copy()) for k, (cells, w) in p.types.items()}

    def run(zero):
        e.reset(5); p.reset_traces(); e.clear_drive(); p.quiet_dan(); p.set_enabled(False)
        p.drive_dan("PPL105", e.p.dan_drive_mv)
        n_dan, da_max = 0, 0.0
        with (O.dopamine_zero(p) if zero else contextlib.nullcontext()):
            for _ in range(200):
                n_dan += int(np.isin(e.step(), dan).sum())
                da_max = max(da_max, float(np.abs(p.da).max()), float(np.abs(p.da_base).max()))
        p.quiet_dan(); p.set_enabled(True)
        return n_dan, da_max

    n_plain, da_plain = run(False)
    n_zero, da_zero = run(True)
    assert n_plain > 0 and n_zero == n_plain                  # DAN firing untouched
    assert da_plain > 0 and da_zero == 0.0                    # the rule's dopamine trace held at exactly 0
    assert p.types["PPL105"][1].any()                         # restored
    assert p.types.keys() == before.keys() and all(
        np.array_equal(p.types[k][0], before[k][0]) and np.array_equal(p.types[k][1], before[k][1]) for k in before)


def test_dopamine_zero_restores_on_error(conn_pops):
    c, pops = conn_pops
    _, p, _, _ = O.rig(c, pops, P, "none")
    saved = p.types
    with pytest.raises(RuntimeError):
        with O.dopamine_zero(p):
            raise RuntimeError
    assert p.types is saved


def test_da_zero_arm_moves_no_weight_and_integrates_no_dopamine(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "da_zero")
    assert r["w_post_sha256"] == r["w0_sha256"] and r["weights_frac"] == 1.0
    assert all(v == 0.0 for v in r["da_integral"].values())


def test_frozen_arm_is_the_plumbing_check(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "frozen")
    assert r["w_post_sha256"] == r["w0_sha256"] and r["weights_frac"] == 1.0
    assert _rows(r["pre"]) == _rows(r["post"])


def test_every_arm_carries_both_weight_hashes(conn_pops):
    c, pops = conn_pops
    _, p, _, _ = O.rig(c, pops, P, "none")
    w0 = O.weights_sha256(p.w0)
    for arm in FLAGS:
        r = _o(c, pops, arm)
        assert r["w0_sha256"] == w0 and len(r["w_post_sha256"]) == 64


def test_punish_arm_records_dopamine_and_readout_weights(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "punish")
    _, _, comps, _ = O.rig(c, pops, P, "none")
    assert set(r["da_integral"]) == {k for k, cp in comps.items() if cp.core.size}
    assert r["da_integral"]["PPL105"] > 0.0
    assert r["weights_frac_A"] < 1.0                          # PPL105's core (the punished compartment) moved
    assert (r["arm"], r["punish"], r["plastic"], r["da_zero"]) == ("punish", True, True, False)


def test_a_p_weights_follow_the_taught_cores_not_the_readout(conn_pops):
    c, pops = conn_pops
    r = _o(c, pops, "punish")
    swapped = _o(c, pops, "punish", readout={"A": "MBON01", "P": "MBON03"})
    assert (swapped["weights_frac_A"], swapped["weights_frac_P"]) == (r["weights_frac_A"], r["weights_frac_P"])
    assert r["weights_frac_A"] < r["weights_frac_P"]          # A = PPL105 core (punished), P = PAM08 core


def test_a_da_zero_arm_does_not_leak_into_the_next_arm(conn_pops):
    c, pops = conn_pops
    _o(c, pops, "da_zero")
    after = _o(c, pops, "punish")
    e, p, _, _ = O.rig(c, pops, P, "none")
    assert e.on_step == p.on_step and all(w.any() for _, w in p.types.values())
    O._RIG.clear()
    fresh = _o(c, pops, "punish")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s", "pre", "post")}
    assert strip(after) == strip(fresh) and _rows(after["post"]) == _rows(fresh["post"])


def test_arms_on_block_on_in_one_worker(conn_pops):
    c, pops = conn_pops
    strip = lambda r: {k: v for k, v in r.items() if k not in ("wall_s", "pre", "post")}
    a, b, a2 = _o(c, pops, "punish"), _o(c, pops, "punish", edit=O.NONKC), _o(c, pops, "punish")
    assert strip(a) == strip(a2) and a["csc_sha256"] != b["csc_sha256"]
