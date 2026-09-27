"""The M4 battle driver without Showdown: the eval block's freeze, refusals before any server, retries with rollback
(weights, yoked queue, log), and resume equivalence on a fake pool and fake battles."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.rescope import blocks
from flymon.rescope.spec import SPEC
from flymon.rescope.yoke import YokedQueue

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("rb", ROOT / "scripts/run_rescope_battles.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# ---- the eval block on a real (synthetic) pool ------------------------------------------------------
def make_swarm(synthetic_npz):
    """tests/agent/test_swarm.py's pool and swarm (2 flies, synthetic readout MBON03 / MBON01)."""
    from flymon.agent.config import AgentConfig
    from flymon.agent.swarm import BrainSwarm
    from flymon.brain.circuits import Populations
    from flymon.brain.config import Params
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlyPool, FlySpec
    from flymon.brain.h4_jobs import type_cells
    P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
    cfg = AgentConfig(params=P, z={"A": (5.0, 2.0), "P": (5.0, 2.0)}, readout={"A": "MBON03", "P": "MBON01"},
                      strength=1.0, settle_ms=100.0, read_ms=100.0)
    conn = Connectome.load(str(synthetic_npz))
    pops = Populations.from_connectome(conn)
    pool = FlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], workers=2, timeout_s=120)
    swarm = BrainSwarm(pool, cfg, type_cells(conn, ["MBON03", "MBON01"]), pops.kc)
    return pool, swarm, [{"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}]


def test_eval_block_freezes_weights(synthetic_npz):
    pool, swarm, odours = make_swarm(synthetic_npz)
    try:
        req = lambda b: SimpleNamespace(context=dict(battle_tag="t", turn=1, fly=0, battle_id=b, battle_index=0, k=0,
                                                     odours=odours))
        asyncio.run(swarm.reinforce_run_batch([SimpleNamespace(context=dict(fly=0, odour=odours[0], dan="PAM08",
                                                                            ms=400.0, seed=1))]))
        before = pool.w[0].copy()
        assert not np.array_equal(before, pool.w0[None])            # the learning pulse did change fly 0
        blocks.enter_eval(swarm, pool)
        asyncio.run(swarm.decide_run_batch([req("E-f00-b000")]))
        assert swarm.mode == "eval" and np.array_equal(pool.w[0], before)
        assert all(not f["enabled"] for f in pool.state()["flies"])
    finally:
        swarm._exec.shutdown(wait=True)
        pool.terminate()


def test_eval_player_queues_no_pulse(tmp_path):
    from flymon.battle.attribution import Outcome
    from .test_players import _kw
    p = blocks.EvalPlayer(fly=0, **_kw(tmp_path, "fm-eval-t1", log="ev00.jsonl"))
    p.start_battle("E-f00-b000", 0)
    tag = "battle-gen1ou-1"
    p._choice[tag] = {"turn": 3, "odour": {"ORN_DM1": 1.0}, "move": "surf"}
    p._on_outcome(tag, 3, Outcome(dealt_frac=1.0, effectiveness="super"))   # would be a PAM08 pulse in learning
    assert p.pending_pulses(tag) == [] and tag not in p._choice
    recs = blocks.read_jsonl(tmp_path / "ev00.jsonl")
    assert not [r for r in recs if r.get("kind") == "reinforce"]


def test_eval_reinforcement_barrier_refuses():
    with pytest.raises(RuntimeError, match="no pulse in E"):
        asyncio.run(blocks.refuse_pulses([SimpleNamespace(context={})]))


# ---- refusals before any server -----------------------------------------------------------------------
def _taurec(root, smoke=False, r=0.0, status="SELECTED"):
    t = root / ("results/rescope-smoke/summary" if smoke else "results/summary") / "rescope_taurec.json"
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(json.dumps({"status": status, "recovery_per_pulse": r}))


def test_rs_refuses_without_donor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "RS", "--flies", "2", "--eval", "2",
                     "--out", "results/rescope/pilot/RS", "--allow-dirty"])


def test_rs_refuses_without_donor_result(tmp_path, monkeypatch):
    """FLY's logs exist but its result.json does not (the FLY arm is not complete): RS must not start."""
    monkeypatch.chdir(tmp_path)
    _taurec(tmp_path)
    for k in range(2):
        log = tmp_path / f"results/rescope/pilot/FLY/logs/fly{k:02d}.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("")
    with pytest.raises(SystemExit, match="result.json does not exist"):
        load().main(["--phase", "pilot", "--arm", "RS", "--flies", "2", "--eval", "2", "--learn", "2",
                     "--out", "results/rescope/pilot/RS", "--allow-dirty"])
    assert not (tmp_path / "results/rescope/pilot/eval_schedule.json").exists()


def test_rs_refuses_changed_donor_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _taurec(tmp_path)
    rb = load()
    learn = blocks.block_schedule(2, 2, 101, "L")
    ev = blocks.block_schedule(2, 2, 102, "E")
    rows = []
    for k in range(2):
        log = tmp_path / f"results/rescope/pilot/FLY/logs/fly{k:02d}.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("{}\n")
        rows.append(dict(fly=k, invalid=False, learn_log_sha256=blocks.file_sha(log)))
    donor = dict(arm="FLY", phase="pilot", flies=2, learn=2, eval=2, recovery_per_pulse=0.0, complete=True,
                 schedule_digests=dict(learn=blocks.schedule_digest(learn), eval=blocks.schedule_digest(ev)), per_fly=rows)
    (tmp_path / "results/rescope/pilot/FLY/result.json").write_text(json.dumps(donor))
    a = rb.parse_args(["--phase", "pilot", "--arm", "RS", "--flies", "2", "--eval", "2", "--learn", "2"])
    d = dict(learn=blocks.schedule_digest(learn), eval=blocks.schedule_digest(ev))
    assert rb.check_donor(tmp_path / "results/rescope/pilot/FLY", a, d, 0.0)["arm"] == "FLY"
    with pytest.raises(SystemExit, match="recovery_per_pulse"):
        rb.check_donor(tmp_path / "results/rescope/pilot/FLY", a, d, 0.01)
    (tmp_path / "results/rescope/pilot/FLY/logs/fly01.jsonl").write_text("{}\n{}\n")   # FLY 1 rerun
    with pytest.raises(SystemExit, match="changed after FLY finished"):
        rb.check_donor(tmp_path / "results/rescope/pilot/FLY", a, d, 0.0)


def test_arm_requires_taurec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "FLY", "--flies", "2", "--eval", "2",
                     "--out", "results/rescope/pilot/FLY", "--allow-dirty"])
    assert not (tmp_path / "results/rescope/pilot/eval_schedule.json").exists()


def test_arm_refuses_unselected_taurec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _taurec(tmp_path, status="STOP_NO_RECOVERY")
    with pytest.raises(SystemExit, match="not SELECTED"):
        load().main(["--phase", "pilot", "--arm", "COFF", "--flies", "2", "--eval", "2", "--allow-dirty"])


def test_smoke_reads_the_smoke_taurec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _taurec(tmp_path)                                    # the real one exists, the smoke one does not
    with pytest.raises(SystemExit, match="rescope-smoke/summary/rescope_taurec.json"):
        load().main(["--smoke", "--phase", "pilot", "--arm", "FLY", "--out", "results/rescope-smoke/pilot/FLY",
                     "--allow-dirty"])


def test_smoke_paths_are_self_contained():
    rb = load()
    a = rb.parse_args(["--smoke", "--phase", "pilot", "--arm", "RS", "--out", "results/rescope/pilot/RS"])
    assert a.out == "results/rescope-smoke/pilot/RS" and (a.flies, a.learn, a.eval, a.workers) == (2, 2, 2, 2)
    assert rb.parse_args(["--phase", "judge", "--arm", "MAX", "--flies", "3", "--eval", "4"]).out == \
        "results/rescope/judge/MAX"


def test_judge_requires_sized_power(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["--phase", "judge", "--arm", "RND", "--flies", "2", "--eval", "2", "--allow-dirty"]
    with pytest.raises(SystemExit, match="does not exist"):
        load().main(args)
    p = tmp_path / "results/summary/rescope_power.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "STOP_BUDGET", "F": 2, "E": 2}))
    with pytest.raises(SystemExit, match="not SIZED"):
        load().main(args)
    p.write_text(json.dumps({"status": "SIZED", "F": 8, "E": 2}))
    with pytest.raises(SystemExit, match="sized F/E"):
        load().main(args)


def test_refuses_out_outside_rescope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "RND", "--flies", "2", "--eval", "2", "--out", "results/m3/RND",
                     "--allow-dirty"])


def test_rs_marks_invalid_on_residual():
    q = YokedQueue.from_bundles([[("PAM08", 94.0)], [("PAM08", 6.0)]], donor_sha256="x"); q.pop()
    assert blocks.fly_result(0, [], queue=q, spec=SPEC)["invalid"] is True            # residual 0.06
    q2 = YokedQueue.from_bundles([[("PAM08", 95.0)], [("PAM08", 5.0)]], donor_sha256="x"); q2.pop()
    assert blocks.fly_result(0, [], queue=q2, spec=SPEC)["invalid"] is False          # residual 0.05


def test_retry_limit():
    calls = []
    async def never(sb): calls.append(1); return {"finished": False}
    out = asyncio.run(blocks.play_with_retries(never, None, 3))
    assert out["invalid"] and out["retries"] == 3 and len(calls) == 4
    seq = iter([False, False, True])
    async def third(sb): return {"finished": next(seq)}
    out = asyncio.run(blocks.play_with_retries(third, None, 3))
    assert not out["invalid"] and out["retries"] == 2


def test_eval_schedule_digest_mismatch_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = tmp_path / "results/summary/rescope_taurec.json"; t.parent.mkdir(parents=True)
    t.write_text(json.dumps({"status": "SELECTED", "recovery_per_pulse": 0.0}))
    e = tmp_path / "results/rescope/pilot/eval_schedule.json"; e.parent.mkdir(parents=True)
    e.write_text(json.dumps({"digest": "not-the-one", "schedule": []}))
    with pytest.raises(SystemExit):
        load().main(["--phase", "pilot", "--arm", "RND", "--flies", "2", "--eval", "2",
                     "--out", "results/rescope/pilot/RND", "--allow-dirty"])


def test_eval_schedule_of_another_size_refused(tmp_path):
    rb = load()
    e = tmp_path / "eval_schedule.json"
    other = blocks.block_schedule(3, 2, 102, "E")
    e.write_text(json.dumps({"digest": blocks.schedule_digest(other), "schedule": blocks.schedule_rows(other)}))
    assert rb.check_eval_schedule(e, other) is False
    with pytest.raises(SystemExit, match="another eval schedule"):
        rb.check_eval_schedule(e, blocks.block_schedule(2, 2, 102, "E"))
    assert rb.check_eval_schedule(tmp_path / "absent.json", other) is True


def test_names_fit_showdown():
    rb = load()
    assert rb.account("COFF", "E", 31) == "fm-rCOE-f31" and rb.account("FLY", "L", 0) == "fm-rFLYL-f00"
    assert rb.opponent_name("E-f31-b299", 3) == "fm-h-E-f31-b299-3"


# ---- fake arms: retries, rollback, resume -------------------------------------------------------------
class FakePool:
    def __init__(self, n, enabled=True):
        self.n_flies = n
        self.w0 = {None: np.linspace(1.0, 2.0, 5, dtype=np.float32)}
        self.w = {i: self.w0[None].copy() for i in range(n)}
        self.en = [enabled] * n

    def set_enabled(self, f, on):
        self.en[f] = bool(on)

    def state(self):
        return {"flies": [dict(enabled=self.en[i], shuffle_seed=None, w=self.w[i].copy()) for i in range(self.n_flies)]}

    def load_state(self, d):
        for i, e in enumerate(d["flies"]):
            self.w[i] = np.asarray(e["w"], np.float32).copy()
            self.en[i] = bool(e["enabled"])


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:6], 16)


def fake_attempt_for(out, pool, yoke=None, fail=(), eval_learns=False):
    """A deterministic fake battle: in L the fly learns (weights move by an amount from the battle id) and pops two
    yoked bundles, in E it only decides; `fail` holds (battle_id, attempt) pairs that end unfinished after having
    learned and logged (a mid-battle server error)."""
    def attempt_for(block):
        async def attempt(sb, n):
            f = sb.fly_id
            logs = Path(out) / ("logs" if block == "L" else "logs/eval")
            log = logs / f"fly{f:02d}.jsonl"
            blocks.append_jsonl(log, {"battle_id": sb.battle_id, "fly": f, "kind": "decision"})
            if block == "L" or eval_learns:
                pool.w[f] = pool.w[f] * np.float32(0.99) + np.float32(_h(sb.battle_id) % 7) * np.float32(0.01)
                pulses = []
                if yoke:
                    pulses = [yoke.queues[f].pop(), yoke.queues[f].pop()]
                blocks.append_jsonl(log, {"battle_id": sb.battle_id, "fly": f, "kind": "reinforce",
                                          "pulses": [p for p in pulses if p]})
            await asyncio.sleep(0)
            if (sb.battle_id, n) in fail:
                return {"finished": False, "won": None}
            return {"finished": True, "won": _h(sb.battle_id, "won") % 2 == 0, "fly_turns": 3, "coach_turns": 1}
        return attempt
    return attempt_for


def donor_queues(tmp_path, learn, n, sha_salt=""):
    qs = {}
    for k in range(n):
        log = tmp_path / f"donor{sha_salt}/fly{k:02d}.jsonl"
        if log.exists():
            qs[k] = YokedQueue.from_log(log, blocks.ids_of(learn, k))
            continue
        for b in blocks.ids_of(learn, k):
            blocks.append_jsonl(log, {"kind": "reinforce", "battle_id": b, "pulses": [["PAM08", 100.0 + _h(b) % 5]]})
            blocks.append_jsonl(log, {"kind": "reinforce", "battle_id": b, "pulses": [["PPL105", 50.0]]})
        if sha_salt:
            blocks.append_jsonl(log, {"kind": "note", "salt": sha_salt})
        qs[k] = YokedQueue.from_log(log, blocks.ids_of(learn, k))
    return qs


def run_fake(out, *, learn, eval_, n, resume=False, stop_after=None, fail=(), yoke_queues=None, pool=None,
             eval_learns=False):
    pool = pool or FakePool(n)
    swarm = SimpleNamespace(mode="learn")
    yoke = blocks.YokeBook(Path(out) / "yoke_state.json", yoke_queues) if yoke_queues is not None else None
    resets = []
    run = asyncio.run(blocks.run_arm(out, eval_=eval_, attempt_for=fake_attempt_for(out, pool, yoke, fail, eval_learns),
                                     cfg_hash="h" * 64, retry_max=SPEC.retry_max, resume=resume, n_flies=n,
                                     learn=learn, pool=pool, swarm=swarm, yoke=yoke,
                                     reset_player=lambda b, f: resets.append((b, f)), stop_after=stop_after))
    return run, pool, swarm, yoke, resets


def snapshot(out):
    out = Path(out)
    recs = {}
    for p in ("logs/battles.jsonl", "logs/eval/battles.jsonl"):
        recs[p] = sorted(blocks.read_jsonl(out / p), key=lambda r: r["battle_id"])
    logs = {p.relative_to(out).as_posix(): p.read_text() for p in sorted(out.glob("logs/**/fly*.jsonl"))
            if "retries" not in p.parts}
    return recs, logs


def test_fly_arm_runs_both_blocks_and_freezes_eval(tmp_path):
    learn, ev = blocks.block_schedule(2, 3, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    run, pool, swarm, _, _ = run_fake(tmp_path / "FLY", learn=learn, eval_=ev, n=2)
    assert run["complete"] and swarm.mode == "eval" and pool.en == [False, False]
    assert run["before"] == run["after"] and run["before"][0] != blocks.weights_sha(pool.w0[None])
    rows = blocks.arm_per_fly(tmp_path / "FLY", n_flies=2, eval_=ev, run=run, learn=learn, spec=SPEC)
    assert [len(r["eval_battles"]) for r in rows] == [2, 2] and not any(r["invalid"] for r in rows)
    assert all(r["weights_bit_identical_across_eval"] and r["learn_battles"] == 3 for r in rows)
    assert rows[0]["learn_log_sha256"] == blocks.file_sha(tmp_path / "FLY/logs/fly00.jsonl")


def test_eval_weight_change_is_flagged(tmp_path):
    learn, ev = blocks.block_schedule(2, 2, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    run, *_ = run_fake(tmp_path / "FLY", learn=learn, eval_=ev, n=2, eval_learns=True)
    rows = blocks.arm_per_fly(tmp_path / "FLY", n_flies=2, eval_=ev, run=run, learn=learn, spec=SPEC)
    assert all(r["weights_bit_identical_across_eval"] is False and r["invalid"] for r in rows)


def test_retry_restores_weights_queue_and_log(tmp_path):
    """An unfinished attempt learned, popped yoked bundles and logged; the retry starts from the pre-battle state,
    so the run equals one without the server error (only the retry count differs)."""
    learn, ev = blocks.block_schedule(2, 3, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    clean, pc, _, yc, _ = run_fake(tmp_path / "a", learn=learn, eval_=ev, n=2,
                                   yoke_queues=donor_queues(tmp_path, learn, 2))
    bad = {("L-f00-b001", 0), ("L-f00-b001", 1), ("E-f01-b000", 0)}
    faulty, pf, _, yf, resets = run_fake(tmp_path / "b", learn=learn, eval_=ev, n=2, fail=bad,
                                         yoke_queues=donor_queues(tmp_path, learn, 2))
    assert sorted(resets) == [("E", 1), ("L", 0), ("L", 0)]
    for f in (0, 1):
        assert np.array_equal(pc.w[f], pf.w[f])
        assert yc.queues[f].state() == yf.queues[f].state()
    rc, lc = snapshot(tmp_path / "a")
    rf, lf = snapshot(tmp_path / "b")
    assert lc == lf                                      # failed attempts' records left the fly logs
    strip = lambda rs: [{k: v for k, v in r.items() if k != "retries"} for r in rs]
    assert {p: strip(v) for p, v in rc.items()} == {p: strip(v) for p, v in rf.items()}
    retries = {r["battle_id"]: r["retries"] for p in rf for r in rf[p] if r["retries"]}
    assert retries == {"L-f00-b001": 2, "E-f01-b000": 1}
    moved = blocks.read_jsonl(tmp_path / "b/logs/retries/fly00.jsonl")
    assert {r["attempt"] for r in moved} == {0, 1} and all(r["battle_id"] == "L-f00-b001" for r in moved)
    assert json.loads((tmp_path / "b/yoke_state.json").read_text())["flies"]["0"]["after"]["L-f00-b002"] == \
        yc.queues[0].state()


def test_retries_exhausted_make_the_fly_invalid(tmp_path):
    learn, ev = blocks.block_schedule(2, 2, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    bad = {("L-f01-b000", i) for i in range(SPEC.retry_max + 1)}
    run, pool, *_ = run_fake(tmp_path / "x", learn=learn, eval_=ev, n=2, fail=bad)
    rows = blocks.arm_per_fly(tmp_path / "x", n_flies=2, eval_=ev, run=run, learn=learn, spec=SPEC)
    assert rows[1]["invalid"] and not rows[0]["invalid"]
    rec = next(r for r in blocks.read_jsonl(tmp_path / "x/logs/battles.jsonl") if r["battle_id"] == "L-f01-b000")
    assert rec["retries"] == SPEC.retry_max and rec["invalid"] and not rec["finished"]


@pytest.mark.parametrize("stops", [(1,), (3, 2), (6,), (6, 1, 1), (2, 2, 2, 2)])
def test_resume_gives_the_same_records(tmp_path, stops):
    """Stopped after N battles (in learning, at the block boundary, in evaluation) and resumed: the same battle
    records, fly logs, weights, yoked-queue state and weight hashes as an uninterrupted run."""
    learn, ev = blocks.block_schedule(2, 3, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    full, pfull, _, yfull, _ = run_fake(tmp_path / "full", learn=learn, eval_=ev, n=2,
                                        yoke_queues=donor_queues(tmp_path, learn, 2))
    out = tmp_path / "res"
    for i, s in enumerate(stops):
        run, *_ = run_fake(out, learn=learn, eval_=ev, n=2, resume=i > 0, stop_after=s,
                           yoke_queues=donor_queues(tmp_path, learn, 2))
        assert not run["complete"]
    run, pool, _, yoke, _ = run_fake(out, learn=learn, eval_=ev, n=2, resume=True,
                                     yoke_queues=donor_queues(tmp_path, learn, 2))
    assert run["complete"] and run["before"] == full["before"] and run["after"] == full["after"]
    assert snapshot(out) == snapshot(tmp_path / "full")
    for f in (0, 1):
        assert np.array_equal(pool.w[f], pfull.w[f]) and yoke.queues[f].state() == yfull.queues[f].state()
    rows = lambda o, r, y: blocks.arm_per_fly(o, n_flies=2, eval_=ev, run=r, learn=learn, yoke=y, spec=SPEC)
    assert rows(out, run, yoke) == rows(tmp_path / "full", full, yfull)


def test_restart_without_resume_refused(tmp_path):
    learn, ev = blocks.block_schedule(2, 2, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    run_fake(tmp_path / "x", learn=learn, eval_=ev, n=2, stop_after=1)
    with pytest.raises(SystemExit, match="pass --resume"):
        run_fake(tmp_path / "x", learn=learn, eval_=ev, n=2)


def test_rs_resume_refuses_a_rerun_donor(tmp_path):
    """FLY k rerun (its log's sha256 changed) while RS k is part-way: the RS resume refuses."""
    learn, ev = blocks.block_schedule(2, 3, 101, "L"), blocks.block_schedule(2, 2, 102, "E")
    run_fake(tmp_path / "rs", learn=learn, eval_=ev, n=2, stop_after=3, yoke_queues=donor_queues(tmp_path, learn, 2))
    with pytest.raises(SystemExit, match="must be rerun"):
        run_fake(tmp_path / "rs", learn=learn, eval_=ev, n=2, resume=True,
                 yoke_queues=donor_queues(tmp_path, learn, 2, sha_salt="rerun"))


def test_coff_and_nobrain_arms_play_eval_only(tmp_path):
    ev = blocks.block_schedule(2, 2, 102, "E")
    pool = FakePool(2, enabled=False)
    run, pool, swarm, _, _ = run_fake(tmp_path / "COFF", learn=None, eval_=ev, n=2, pool=pool)
    assert run["complete"] and set(run["played"]) == {"E"} and run["before"] == run["after"]
    assert run["before"][0] == blocks.weights_sha(pool.w0[None])
    nb = asyncio.run(blocks.run_arm(tmp_path / "RND", eval_=ev, attempt_for=fake_attempt_for(tmp_path / "RND", None),
                                    cfg_hash="n" * 64, retry_max=3, resume=False, n_flies=2))
    assert nb["complete"] and nb["after"] == {}
    rows = blocks.arm_per_fly(tmp_path / "RND", n_flies=2, eval_=ev, run=nb, brain=False)
    assert [len(r["eval_battles"]) for r in rows] == [2, 2] and rows[0]["weights_bit_identical_across_eval"] is None


# ---- wall-clock sessions (Task 10 fix round 1) ----------------------------------------------------------
def test_aborted_session_is_recorded_and_refusal_is_not(tmp_path, monkeypatch):
    """An exception after the refusals still ends the session (status aborted, seconds) in the finally; a refused start
    appends nothing."""
    import flymon.agent.config as agent_config
    monkeypatch.chdir(tmp_path)

    def boom():
        raise RuntimeError("engine down")
    monkeypatch.setattr(agent_config, "load_c3_config", boom)
    mod = load()
    mod.git_provenance = lambda files=(): {"commit": "c" * 40, "dirty": False, "dirty_files": []}
    args = ["--phase", "pilot", "--arm", "RND", "--flies", "2", "--eval", "2", "--out", "results/rescope/pilot/RND"]
    with pytest.raises(RuntimeError, match="engine down"):
        mod.main(args)
    with pytest.raises(RuntimeError):
        mod.main(args + ["--resume"])
    wc = tmp_path / "results/rescope/pilot/RND/wall_clock.json"
    ss = json.loads(wc.read_text())["sessions"]
    assert [x["status"] for x in ss] == ["aborted", "aborted"]
    assert all(isinstance(x["seconds"], float) and x["ended_utc"] for x in ss)
    (tmp_path / "results/rescope/pilot/RND/result.json").write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        mod.main(args)                                        # refused: result.json exists, no --resume
    assert len(json.loads(wc.read_text())["sessions"]) == 2
