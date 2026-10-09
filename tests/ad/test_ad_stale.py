"""AD.5 2: the stale-pulse race fix. A pulse request carries the attempt token and the fly's weight generation from its
creation; ADSwarm refuses it at submission and LeverFlyPool at write-back unless both are still current. The
deterministic interleavings (a)-(d) are pre-registered (AD.5 2) and are a smoke gate."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.ac.swarm import LVSwarm
from flymon.ad.attempts import AttemptBook
from flymon.ad.swarm import ADSwarm
from flymon.brain.circuits import Populations
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlySpec
from flymon.brain.h4_jobs import type_cells
from flymon.brain.lv_pool import LeverFlyPool
from tests.ac.test_ac_swarm import CFG, P

A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = "DL-f24-b000"


def _parts(npz):
    conn = Connectome.load(str(npz))
    pops = Populations.from_connectome(conn)
    return type_cells(conn, ["MBON03", "MBON01"]), pops.kc


@pytest.fixture
def rig(synthetic_npz):
    cells, kc = _parts(synthetic_npz)
    pool = LeverFlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], edit="none", workers=2, timeout_s=120)
    attempts = AttemptBook()
    sw = ADSwarm(pool, CFG, cells, kc, attempts=attempts, mode="learn")
    yield SimpleNamespace(pool=pool, sw=sw, attempts=attempts)
    sw._exec.shutdown(wait=True)
    pool.terminate()


def _ctx(fly, token, gen, seed, dan="PAM08"):
    return dict(fly=fly, odour=A, dan=dan, ms=400.0, seed=seed, attempt=None if token is None else list(token), gen=gen)


def _req(ctx):
    return SimpleNamespace(player_id=f"p{ctx['fly']}", battle=None, candidates=[ctx["dan"]], context=ctx)


def _rollback(rig, fly, snap):
    rig.attempts.retire(fly)               # ADBattles.reset_player (Task 13)
    rig.pool.w[fly] = snap.copy()          # blocks.rollback_for: bumps the fly's generation


def test_attempt_book():
    book = AttemptBook()
    assert book.current(0) is None and not book.is_current(0, [B, 0]) and not book.is_current(0, None)
    tok = book.begin(0, B, 2)
    assert tok == [B, 2] and book.current(0) == (B, 2) and book.is_current(0, tok)
    book.retire(0)
    assert book.current(0) is None and not book.is_current(0, tok)


async def test_stale_request_made_before_rollback_is_refused_at_submission(rig):
    tok = rig.attempts.begin(0, B, 0)
    snap = np.array(rig.pool.w[0], copy=True)
    c = _ctx(0, tok, rig.sw.gen_of(0), 5)          # (a) made during attempt 0 ...
    _rollback(rig, 0, snap)
    rig.attempts.begin(0, B, 1)
    await rig.sw.reinforce_run_batch([_req(c)])    # ... delivered after the rollback
    assert np.array_equal(rig.pool.w[0], snap)
    assert rig.sw.stale[-1] == dict(fly=0, attempt=[B, 0], gen=c["gen"], at="submit")


def test_stale_request_in_flight_is_refused_at_write_back(rig):
    pool = rig.pool
    tok = rig.attempts.begin(0, B, 0)
    snap = np.array(pool.w[0], copy=True)
    done, release, real = threading.Event(), threading.Event(), pool._map

    def held_map(fn, items):                       # the job ran on the workers; its result is not written yet
        out = real(fn, items)
        done.set(); release.wait(30)
        return out
    pool._map = held_map
    gen0, res = rig.sw.gen_of(0), {}
    th = threading.Thread(target=lambda: res.update(applied=pool.reinforce_batch(
        [(0, A, "PAM08", 400.0, 5)], 1.0, 100.0, expect=[gen0], accept=lambda k: rig.attempts.is_current(0, tok))))
    th.start()
    assert done.wait(60)
    _rollback(rig, 0, snap)                        # (a) the rollback lands while the job is in flight
    release.set(); th.join(60)
    pool._map = real
    assert res["applied"] == [False] and np.array_equal(pool.w[0], snap)


async def test_stale_request_made_after_rollback_is_refused(rig):
    tok = rig.attempts.begin(0, B, 0)
    snap = np.array(rig.pool.w[0], copy=True)
    _rollback(rig, 0, snap)
    c = _ctx(0, tok, rig.sw.gen_of(0), 6)          # (b) made after the rollback: current generation, old token
    await rig.sw.reinforce_run_batch([_req(c)])
    assert np.array_equal(rig.pool.w[0], snap) and rig.sw.stale[-1]["at"] == "submit"
    rig.attempts.begin(0, B, 1)
    await rig.sw.reinforce_run_batch([_req(c)])    # still refused once the next attempt has begun
    assert np.array_equal(rig.pool.w[0], snap)


async def test_new_attempt_pulse_lands(rig):
    rig.attempts.begin(0, B, 0)
    snap = np.array(rig.pool.w[0], copy=True)
    _rollback(rig, 0, snap)
    tok = rig.attempts.begin(0, B, 1)
    await rig.sw.reinforce_run_batch([_req(_ctx(0, tok, rig.sw.gen_of(0), 7))])   # (c)
    assert not np.array_equal(rig.pool.w[0], snap) and rig.sw.stale == []


def test_reinforce_batch_without_expect_keeps_the_old_behaviour(rig):
    applied = rig.pool.reinforce_batch([(0, A, "PAM08", 400.0, 5), (1, A, "PPL105", 400.0, 6)], 1.0, 100.0)
    assert applied == [True, True]
    assert not np.array_equal(rig.pool.w[0], rig.pool.w0[None])


SEQ = [[(0, "PAM08", 11), (1, "PPL105", 12)], [(0, "PPL105", 13), (0, "PAM08", 14), (1, "PAM08", 15)]]


async def test_no_retry_weights_bit_identical_to_the_old_path(synthetic_npz):
    # (d) a battle with no retry: ADSwarm (tokens and generations current) == LVSwarm's BrainSwarm path, bit for bit
    cells, kc = _parts(synthetic_npz)
    with LeverFlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], edit="none", workers=2, timeout_s=120) as old:
        sw = LVSwarm(old, CFG, cells, kc, mode="learn")
        for batch in SEQ:
            await sw.reinforce_run_batch([_req(dict(fly=f, odour=A, dan=d, ms=400.0, seed=s)) for f, d, s in batch])
        want = {f: np.array(old.w[f], copy=True) for f in (0, 1)}
        sw._exec.shutdown(wait=True)
    with LeverFlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], edit="none", workers=2, timeout_s=120) as new:
        att = AttemptBook()
        tok = {f: att.begin(f, f"DL-f{24 + f:02d}-b000", 0) for f in (0, 1)}
        sw = ADSwarm(new, CFG, cells, kc, attempts=att, mode="learn")
        for batch in SEQ:
            await sw.reinforce_run_batch([_req(_ctx(f, tok[f], sw.gen_of(f), s, d)) for f, d, s in batch])
        got = {f: np.array(new.w[f], copy=True) for f in (0, 1)}
        sw._exec.shutdown(wait=True)
    assert all(np.array_equal(got[f], want[f]) for f in (0, 1)) and sw.stale == []


async def test_stale_requests_are_logged(rig, tmp_path):
    rig.sw.stale_path = tmp_path / "stale.jsonl"
    tok = rig.attempts.begin(1, "DL-f25-b003", 0)
    rig.attempts.retire(1)
    await rig.sw.reinforce_run_batch([_req(_ctx(1, tok, rig.sw.gen_of(1), 9))])
    assert '"at": "submit"' in (tmp_path / "stale.jsonl").read_text()


async def test_rollback_between_submit_check_and_pool_is_refused_at_write_back(rig):
    # (f) the 1480b36 gap: the rollback lands after ADSwarm's submission check, before the pool reads the generation
    tok = rig.attempts.begin(0, B, 0)
    snap = np.array(rig.pool.w[0], copy=True)
    c = _ctx(0, tok, rig.sw.gen_of(0), 8)
    real = rig.pool.reinforce_batch

    def late_rollback(*a, **kw):
        _rollback(rig, 0, snap)
        rig.attempts.begin(0, B, 1)
        return real(*a, **kw)
    rig.pool.reinforce_batch = late_rollback
    await rig.sw.reinforce_run_batch([_req(c)])
    del rig.pool.reinforce_batch
    assert np.array_equal(rig.pool.w[0], snap)
    assert rig.sw.stale == [dict(fly=0, attempt=[B, 0], gen=c["gen"], at="apply")]
