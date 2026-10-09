"""scripts/run_ad.py without Showdown: arguments, the refusals that run before any pool or server (inputs ->
STOP_MANIFEST, stage-1 reuse -> STOP_REUSE, model manifest, run manifest -> STOP_MANIFEST, a used bench directory,
a spent budget), the session window, and the id / name lengths."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.ac import schedules as ac_sched
from flymon.ad import inputs, run_manifest as rm, store
from flymon.ad.spec import LABEL, SPEC, smoke
from tests.ad.test_ad_reuse import fake_stage1

ROOT = Path(__file__).resolve().parents[2]


def load():
    s = importlib.util.spec_from_file_location("rad", ROOT / "scripts/run_ad.py")
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod


def smoke_inputs():
    s = smoke()
    doc_p, man_p = inputs.input_paths(s)
    d = inputs.inputs_doc(s)
    store.write_json(doc_p, d)
    store.write_json(man_p, inputs.input_manifest(d, doc_p))
    return doc_p, man_p


def frozen_model(tmp_path):
    p = tmp_path / "results/summary/ac_model_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "FROZEN", "recovery_per_pulse": 0.001}))


def test_parse_args():
    ra = load()
    a = ra.parse_args(["--smoke"])
    assert a.workers == 4 and ra.spec_for(a).out_root == "results/m4-smoke/ad"
    a = ra.parse_args(["--bench", "2"])
    assert a.workers == 16 and ra.spec_for(a).out_root == "results/m4-bench/ad-2"
    assert ra.spec_for(ra.parse_args([])) == SPEC
    with pytest.raises(SystemExit):
        ra.parse_args(["--bench", "1", "--resume"])
    with pytest.raises(SystemExit):
        ra.parse_args(["--bench", "1", "--smoke"])


def test_refuses_missing_or_changed_inputs(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        ra.main(["--smoke"])
    doc_p, _ = smoke_inputs()
    d = json.loads(doc_p.read_text())
    d["eval"][0]["my_team"] = list(reversed(d["eval"][0]["my_team"]))
    doc_p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        ra.main(["--smoke"])


def test_refuses_without_a_frozen_model(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    smoke_inputs()
    with pytest.raises(SystemExit, match="ac_model_manifest"):
        ra.main(["--smoke"])
    store_p = tmp_path / "results/summary/ac_model_manifest.json"
    store_p.parent.mkdir(parents=True, exist_ok=True)
    store_p.write_text(json.dumps({"status": "STOP_NO_RECOVERY"}))
    with pytest.raises(SystemExit, match="not FROZEN"):
        ra.main(["--smoke"])


def test_resume_refuses_a_changed_run_manifest(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    _, man_p = smoke_inputs()
    frozen_model(tmp_path)
    with pytest.raises(SystemExit, match="run_manifest"):
        ra.main(["--smoke", "--allow-dirty"])
    m = rm.build(smoke(), T=2, new_set=True, budget_path=None, inputs_manifest_path=man_p, commit="0" * 40)
    store.write_json(rm.manifest_path(smoke()), dict(m, points=[0, 2]))
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        ra.main(["--smoke", "--resume", "--allow-dirty"])


def test_run_mode_refuses_a_changed_stage1_record(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    fake_stage1(tmp_path)
    w = importlib.util.spec_from_file_location("wai", ROOT / "scripts/write_ad_inputs.py")
    mod = importlib.util.module_from_spec(w); w.loader.exec_module(mod)
    assert mod.main([]) == 0
    with (tmp_path / "results/m4/stage1/BRAIN/logs/situations_init.jsonl").open("a") as f:
        f.write(json.dumps({"fly": 12, "point": 0, "arm": "COFF", "frozen": True, "label": LABEL}) + "\n")
    with pytest.raises(SystemExit, match="STOP_REUSE"):
        ra.main([])


def test_bench_refuses_a_used_directory(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/m4-bench/ad-1/logs").mkdir(parents=True)
    with pytest.raises(SystemExit, match="earlier bench attempt"):
        ra.fresh_bench(Path("results/m4-bench/ad-1"))
    (tmp_path / "results/m4-bench/ad-2").mkdir(parents=True)
    (tmp_path / "results/m4-bench/ad-2/inputs.json").write_text("{}")
    ra.fresh_bench(Path("results/m4-bench/ad-2"))                 # inputs alone are not an earlier attempt


def test_session_window(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    out = Path("results/m4/ad")
    a = ra.parse_args([])
    cap, left = ra.session_window(SPEC, out, a, 0.0)
    assert cap == 24 * 3600 and left == pytest.approx(SPEC.budget_limit_h() * 3600)
    store.write_json(out / "wall_clock.json", {"sessions": [dict(started_utc="1970-01-01T00:00:00+00:00",
                                                                seconds=47.5 * 3600, status="ok")]})
    with pytest.raises(SystemExit, match="STOP_BUDGET"):
        ra.session_window(SPEC, out, a, 0.0)
    cap, left = ra.session_window(smoke(), Path("results/m4-smoke/ad"), ra.parse_args(["--smoke", "--session-hours",
                                                                                         "0.5"]), 0.0)
    assert cap == 1800.0 and left == float("inf")


def test_run_ids_and_names_fit():
    doc = inputs.inputs_doc(SPEC)
    learn, ev = inputs.schedules_for(SPEC, doc, 120)
    ac_sched.check_names(learn + ev, SPEC.retry_max)
    names = [inputs.account(a, b, g) for a, _, g in SPEC.layout() for b in "LE"]
    assert len(set(names)) == 48 and max(len(n) for n in names) <= 18 and len(learn) == 2880   # ruling D2


# ---- carry notes (Tasks 2, 5-12 -> 13) ------------------------------------------------------------------------------

def test_run_mode_refuses_a_changed_run_manifest_too(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    _, man_p = smoke_inputs()
    frozen_model(tmp_path)
    m = rm.build(smoke(), T=2, new_set=True, budget_path=None, inputs_manifest_path=man_p, commit="0" * 40)
    store.write_json(rm.manifest_path(smoke()), dict(m, T=3))
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        ra.main(["--smoke", "--allow-dirty"])


def run_inputs(tmp_path):
    fake_stage1(tmp_path)
    w = importlib.util.spec_from_file_location("wai", ROOT / "scripts/write_ad_inputs.py")
    mod = importlib.util.module_from_spec(w); w.loader.exec_module(mod)
    assert mod.main([]) == 0
    return inputs.input_paths(SPEC)


def test_a_changed_confirm_digest_is_stop_reuse_not_stop_manifest(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    doc_p, _ = run_inputs(tmp_path)
    assert ra.load_inputs(SPEC)[0]["digests"]["confirm"]
    d = json.loads(doc_p.read_text())
    d["digests"]["confirm"] = "0" * 64                   # an AD-fixed field too: (가) must be read first
    doc_p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="STOP_REUSE"):
        ra.load_inputs(SPEC)


def test_a_rebuilt_digest_off_stage1_is_stop_reuse(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    run_inputs(tmp_path)
    real = inputs.inputs_doc

    def drifted(spec=SPEC):
        d = real(spec)
        d["digests"] = dict(d["digests"], eval="f" * 64)
        return d
    monkeypatch.setattr(inputs, "inputs_doc", drifted)
    with pytest.raises(SystemExit, match="STOP_REUSE"):
        ra.load_inputs(SPEC)


def test_input_manifest_stop_statuses_map_to_ad_stops(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    doc_p, man_p = smoke_inputs()
    man = json.loads(man_p.read_text())
    for st, code in (("STOP_REUSE", "STOP_REUSE"), ("STOP_SET", "STOP_MANIFEST")):
        man_p.write_text(json.dumps(dict(man, status=st)))
        with pytest.raises(SystemExit, match=code):
            ra.load_inputs(smoke())
    d = json.loads(doc_p.read_text())
    doc_p.write_text(json.dumps(dict(d, confirm=dict(d["confirm"], status="STOP_SET"))))
    with pytest.raises(SystemExit, match="STOP_REUSE"):
        ra.load_inputs(smoke())


def _fake_learn(n_flies=2, n_battles=3):
    from flymon.battle.schedule import ScheduledBattle
    return [ScheduledBattle(f"DL-f{24 + f:02d}-b{b:03d}", f, ["Tauros"], ["Snorlax"], "random")
            for f in range(n_flies) for b in range(n_battles)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_c1_a_lagging_fly_plays_its_opened_batch_after_a_refusal(tmp_path, monkeypatch):
    """Ruling C1: run_ad passes should_stop=None; the gate refuses batch k+1 while a lagging fly still has battle k
    -> battle k is played (fly 0 refused at batch 2 by STOP_BUDGET; fly 1, slow in battle 0, still plays battle 1)."""
    import asyncio
    from types import SimpleNamespace

    from flymon.ad import budget
    from flymon.ad.runner import run_ad_arm
    from tests.ac.ac_fakes import FakePool, fake_attempt_for
    ra = load()
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results/m4/ad"
    clock = Clock()
    gate = budget.BatchGate(SPEC, 120, True, session_end=1e9, budget_end=1000.0, bench_batch_s=100.0,
                            bench_eval_s=0.0, times_path=out / "batch_times.json", now=clock)
    pool = FakePool(2)
    base = fake_attempt_for(out, pool)

    def attempt_for(block):
        att = base(block)

        async def a(sb, n):
            if sb.battle_id == "DL-f25-b000":
                for _ in range(200):
                    await asyncio.sleep(0)                  # fly 1 lags in battle 0
            return await att(sb, n)
        return a

    def after(block, sb, rec):
        gate.finished(block, sb)
        if sb.battle_id == "DL-f24-b001":
            clock.t = 900.0                                 # the budget left (100 s) no longer covers a batch (200 s)
    learn = _fake_learn()
    run = asyncio.run(run_ad_arm(out, learn=learn, eval_=[], attempt_for=attempt_for, cfg_hash="h" * 64, retry_max=3,
                                 resume=False, pool=pool, swarm=SimpleNamespace(mode="learn"),
                                 reset_player=lambda b, f: None, after_battle=after, **ra.gate_kwargs(gate)))
    assert gate.refused == "STOP_BUDGET" and gate.refusal["index"] == 2 and not run["complete"]
    assert sorted(run["played"]["L"]) == ["DL-f24-b000", "DL-f24-b001", "DL-f25-b000", "DL-f25-b001"]


def test_gate_reopens_half_played_batches_on_resume(tmp_path, monkeypatch):
    from flymon.ad import budget
    monkeypatch.chdir(tmp_path)
    clock = Clock()
    clock.t = 950.0
    g = budget.BatchGate(SPEC, 120, True, session_end=1e9, budget_end=1000.0, bench_batch_s=100.0, bench_eval_s=0.0,
                         times_path=tmp_path / "results/m4/ad/batch_times.json", now=clock)
    assert g.reopen("L", ["DL-f24-b000", "DL-f24-b001", "DL-f25-b000"]) == 2
    from types import SimpleNamespace
    assert g("L", SimpleNamespace(battle_id="DL-f25-b001")) is False     # half-played batch 1: finished
    assert g("L", SimpleNamespace(battle_id="DL-f24-b002")) is True and g.refused == "STOP_BUDGET"


def test_on_block_end_learn_fires_once_across_the_futility_split(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from flymon.ad.runner import run_ad_arm
    from tests.ac.ac_fakes import FakePool, fake_attempt_for
    from tests.ad.test_ad_runner import EVAL, LEARN, N, S
    ends = []

    async def on_end(block):
        ends.append(block)
    pool = FakePool(N)
    run = asyncio.run(run_ad_arm(tmp_path, learn=LEARN, eval_=EVAL, attempt_for=fake_attempt_for(tmp_path, pool),
                                 cfg_hash="h" * 64, retry_max=3, resume=False, pool=pool,
                                 swarm=SimpleNamespace(mode="learn"), reset_player=lambda b, f: None,
                                 on_block_end=on_end, futility_point=S.futility_point,
                                 decide_futility=lambda: {"status": "CONTINUE"}))
    assert run["complete"] and ends == ["L", "E"]


def test_checkpoint_checksum_is_stop_infra_and_a_foreign_key_stop_manifest(tmp_path):
    from flymon.ad import hooks
    from flymon.ad.runner import load_checkpoint
    from flymon.agent.checkpoint import CheckpointStore
    import numpy as np
    out = tmp_path / "run"
    st = CheckpointStore(out / "checkpoints" / "learn", "k" * 64)
    st.commit("DL-f24-b000", {"flies": [dict(enabled=True, shuffle_seed=None, w=np.ones(3, np.float32))]}, {})
    assert hooks.committed_all(out, "k" * 64) == {"DL-f24-b000"}
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        hooks.committed_all(out, "j" * 64)
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        load_checkpoint(CheckpointStore(out / "checkpoints" / "learn", "j" * 64))
    for f in (out / "checkpoints" / "learn").glob("state_*.npz"):
        f.write_bytes(b"damaged")
    with pytest.raises(SystemExit, match="STOP_INFRA"):
        hooks.committed_all(out, "k" * 64)
    with pytest.raises(SystemExit, match="STOP_INFRA"):
        load_checkpoint(CheckpointStore(out / "checkpoints" / "learn", "k" * 64))


def test_reset_player_retires_the_attempt_before_the_restore():
    """Carry (hard): the rollback retires the token before the weights come back, and the old room's queue is gone."""
    from types import SimpleNamespace

    from flymon.ad.attempts import AttemptBook
    from flymon.ad.battles import ADBattles
    book = AttemptBook()
    b = ADBattles([("FLYL", 0, 24)], Path("results/m4-smoke/ad"), None, None, {}, book)
    seen = []
    p = SimpleNamespace(_queue={"t": [1]}, _choice={"t": {}})
    b.players["L"][0] = p
    book.begin(0, "DL-f24-b000", 0)
    tok = ["DL-f24-b000", 0]

    def restore():
        seen.append((book.is_current(0, tok), dict(p._queue)))
    b.reset_player("L", 0)
    restore()
    assert seen == [(False, {})] and book.current(0) is None


def test_runner_rollback_calls_reset_player_before_the_weights_come_back(tmp_path):
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from flymon.ad.runner import run_ad_arm
    from tests.ac.ac_fakes import FakePool, fake_attempt_for
    pool = FakePool(2)
    w0 = pool.w[0].copy()
    at_reset = []

    def reset(block, fly):
        at_reset.append(not np.array_equal(pool.w[fly], w0))   # still the failed attempt's weights: not restored yet
    learn = _fake_learn(n_battles=1)
    asyncio.run(run_ad_arm(tmp_path, learn=learn, eval_=[], cfg_hash="h" * 64, retry_max=3, resume=False, pool=pool,
                           attempt_for=fake_attempt_for(tmp_path, pool, fail={("DL-f24-b000", 0)}),
                           swarm=SimpleNamespace(mode="learn"), reset_player=reset))
    assert at_reset == [True]


def test_bench_doc_counts_every_point0_evaluation(tmp_path, monkeypatch):
    import os

    from flymon.ac.invalid import InvalidBook
    from flymon.ad import hooks
    from flymon.ad.spec import bench
    ra = load()
    monkeypatch.chdir(tmp_path)
    spec = bench(1)
    lay = spec.layout()
    arm_of = {i: a for i, (a, _, _) in enumerate(lay)}
    mins = {k: v for k, v in spec.min_valid_of().items() if k in spec.arm_sizes()}
    books = {b: InvalidBook(tmp_path / f"inv{b}.json", arm_of, spec.arm_sizes(), mins) for b in "LE"}
    hk = hooks.ADHooks(spec, spec.learn_battles, True, lay, books, hooks.ADSituationLog(tmp_path / "logs"),
                       lambda i, point, set_, mode: {"frozen": True})
    n = hk.initial()
    assert n == 60                                           # 24 own + 12 paired OS + 24 new-set
    b = ra.bench_doc(spec, dict(t_run_s=1600.0, t_sit_init_s=1200.0, n_init=n, t_sit=None), t0=10.0, t_ended=20.0,
                     workers=16, git={})
    assert b["pid"] == os.getpid() and b["t_started"] == 10.0 and b["t_ended"] == 20.0
    assert b["sit_eval_s_per_eval"] == 20.0 and b["s_per_batch_battle"] == 800.0 and b["n_init"] == 60
    d = tmp_path / "results/m4-bench/ad-1"
    d.mkdir(parents=True)
    (d / "bench.json").write_text(json.dumps(b))
    w = importlib.util.spec_from_file_location("wab", ROOT / "scripts/write_ad_budget.py")
    mod = importlib.util.module_from_spec(w); w.loader.exec_module(mod)
    att = mod.attempts(tmp_path / "results/m4-bench", [])     # every key write_ad_budget reads is there
    assert att[0]["s_per_batch_battle"] == 800.0 and att[0]["sit_eval_s_per_eval"] == 20.0
