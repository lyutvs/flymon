"""AD's situation-pair hooks (AD.2, AD.3, AD.7) on the fake pool: the kinds per arm and point, records keyed by the
global fly, the resume dropping uncommitted records of every kind, the two InvalidBooks, the retry log, the frozen
check, and make_evaluate's OS odours and global seeds."""
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.ac.invalid import InvalidBook
from flymon.ad import hooks, inputs, os_encoder
from flymon.ad.spec import LABEL, smoke
from flymon.agent.policy import derive_seed
from flymon.rescope import blocks
from tests.ac.ac_fakes import FakePool, run_fake, stub_grid
from tests.ac.test_ac_situations import FakePool as SitPool
from tests.ac.test_ac_situations import _counts

S = smoke()
LAY = S.layout()
DOC = {"learn": blocks.schedule_rows(inputs.learn_canonical(S)), "eval": blocks.schedule_rows(inputs.eval_canonical(S))}
LEARN, EVAL = inputs.schedules_for(S, DOC, S.learn_battles)
KEY = "h" * 64


def fake_eval(i, point, set_, mode, frozen=True):
    return dict(kind="situation_eval", fly=i, point=point, rate=0.5, switched=[1, 0], frozen=frozen)


def setup(out, new_set=True, evaluate=fake_eval):
    arm_of = {i: arm for i, (arm, _, _) in enumerate(LAY)}
    books = {"L": InvalidBook(out / "invalid.json", arm_of, S.arm_sizes(), S.min_valid_of()),
             "E": InvalidBook(out / "invalid_eval.json", arm_of, S.arm_sizes(), S.min_valid_of())}
    log = hooks.ADSituationLog(out / "logs")
    return books, log, hooks.ADHooks(S, S.learn_battles, new_set, LAY, books, log, evaluate)


def go(out, books, log, hk, **kw):
    if kw.get("resume"):
        log.filter(hooks.committed_all(out, KEY))
    hk.initial()
    return run_fake(out, learn=LEARN, eval_=EVAL, n=len(LAY), pool=FakePool(len(LAY)), after_battle=hk.after_battle,
                    wrap=lambda af: hooks.skipping(af, books), **kw)


def kinds_of(log):
    return sorted((r["fly"], r["point"], r["set"], r["mode"]) for r in log.records())


def test_kinds_per_arm_and_point(tmp_path):
    _, _, hk = setup(tmp_path)
    assert hk.kinds(0, 0) == [("confirm", "own"), ("confirm", "os_paired"), ("new", "own")]
    assert hk.kinds(2, 1) == [("confirm", "own"), ("new", "own")]
    _, _, hk2 = setup(tmp_path / "b", new_set=False)
    assert hk2.kinds(0, 2) == [("confirm", "own"), ("confirm", "os_paired")]


def test_records_at_every_point_keyed_by_the_global_fly(tmp_path):
    books, log, hk = setup(tmp_path)
    run, _ = go(tmp_path, books, log, hk)
    assert run["complete"]
    recs = log.records()
    assert len(recs) == 3 * (2 * 3 + 2 * 2)                       # points 0 / 1 / 2; FLY-L 3 kinds, FLY-OS 2
    r = next(x for x in recs if x["fly"] == 26 and x["point"] == 2 and x["set"] == "new")
    assert (r["pool_index"], r["arm"], r["battle_id"], r["label"]) == (2, "FLYOS", "XL-f26-b001", LABEL)
    assert {x["fly"] for x in recs} == {24, 25, 26, 27}


def test_resume_drops_uncommitted_records_of_every_kind(tmp_path):
    full = tmp_path / "full"
    b, l, h = setup(full)
    go(full, b, l, h)
    part = tmp_path / "part"
    b, l, h = setup(part)
    run, _ = go(part, b, l, h, stop_after=4)                     # only the four b000 battles are committed
    assert not run["complete"]
    stray = "XL-f27-b001"                                         # a battle that was never committed
    for s, m in (("confirm", "own"), ("new", "own")):
        l.add(dict(fake_eval(3, 2, s, m), fly=27, set=s, mode=m, battle_id=stray, label=LABEL))
    l.add_retry(dict(fly=27, pool_index=3, battle_id=stray, block="L", attempts=2, invalid=False))
    b, l, h = setup(part)
    go(part, b, l, h, resume=True)
    assert kinds_of(l) == kinds_of(setup(full)[1]) and l.retries() == []


def test_failed_learning_battle_marks_the_learning_book_and_skips_eval(tmp_path):
    books, log, hk = setup(tmp_path)
    run, _ = go(tmp_path, books, log, hk, fail={("XL-f24-b000", n) for n in range(4)})
    assert books["L"].invalid[0] == "XL-f24-b000: unfinished after 3 retries"
    assert books["E"].invalid[0] == "learn_invalid" and 0 not in books["L"].stopped
    assert not any(r["fly"] == 24 and r["point"] > 0 for r in log.records())
    assert any(r["fly"] == 24 and r["point"] == 0 for r in log.records())       # point 0 stays readable
    rl = [r for r in log.retries() if r["battle_id"] == "XL-f24-b000"]
    assert rl == [dict(fly=24, pool_index=0, battle_id="XL-f24-b000", block="L", attempts=4, invalid=True)]


def test_failed_eval_battle_marks_only_the_eval_book(tmp_path):
    books, log, hk = setup(tmp_path)
    go(tmp_path, books, log, hk, fail={("XE-f26-b000", n) for n in range(4)})
    assert books["E"].invalid == {2: "XE-f26-b000: unfinished after 3 retries"} and books["L"].invalid == {}


def test_retried_battle_is_logged(tmp_path):
    books, log, hk = setup(tmp_path)
    go(tmp_path, books, log, hk, fail={("XL-f25-b001", 0)})
    assert [r for r in log.retries()] == [dict(fly=25, pool_index=1, battle_id="XL-f25-b001", block="L", attempts=2,
                                               invalid=False)]


def test_unfrozen_record_stops(tmp_path):
    books, log, hk = setup(tmp_path, evaluate=lambda i, p, s, m: fake_eval(i, p, s, m, frozen=p == 0))
    with pytest.raises(SystemExit, match="STOP_SMOKE"):
        go(tmp_path, books, log, hk)


PAIRS = [dict(me="Lapras", o1="Charizard", o2="Exeggutor", cands=["Surf", "Psychic", "Thunderbolt"],
              best1="Surf", best2="Psychic")]


class Pool(SitPool):
    def __init__(self):
        super().__init__({(0, 0): _counts(0), (0, 1): _counts(1)})
        self.w = {f: np.ones(4, np.float32) for f in range(4)}
        self.w0 = {None: np.ones(4, np.float32)}
        self.flies = [SimpleNamespace(shuffle_seed=None)] * 4


def test_make_evaluate_uses_os_odours_and_global_seeds():
    pool, grid = Pool(), stub_grid()
    mask = np.array([True, True, False, False])
    sw = SimpleNamespace(a_idx=np.array([0]), p_idx=np.array([1]), kc=np.array([2, 3]),
                         _edge_masks={"PAM08": mask, "PPL105": mask})
    cfg = SimpleNamespace(z={"A": (0.0, 1.0), "P": (0.0, 1.0)}, strength=1.0, settle_ms=800.0, read_ms=600.0,
                          reward_type="PAM08", punish_type="PPL105")
    ev = hooks.make_evaluate(pool, sw, cfg, S, LAY, {"confirm": PAIRS, "new": PAIRS}, grid)
    r = ev(2, 0, "confirm", "own")                                 # FLY-OS fly 26
    want = [os_encoder.SPECIES[os_encoder.draw_index(derive_seed(307, "os-sit", 26, 0, s))] for s in (0, 1)]
    assert [d["drawn"] for d in r["os"]] == want
    assert [s for _, _, s in pool.calls[-1]] == [derive_seed(305, "sit", 26, 0, s) for s in (0, 1)]
    assert "floor_contact" in r and r["weights"]["n_changed"] == 0
    r = ev(0, 0, "confirm", "own")                                 # FLY-L fly 24: the FLY odours, no draw
    assert "os" not in r and pool.calls[-1][0][1] == [os_encoder.odour_for_species(grid, grid.move_info[m][0], "Charizard")
                                                     for m in PAIRS[0]["cands"]]
    r = ev(0, 0, "confirm", "os_paired")
    assert [d["drawn"] for d in r["os"]] == [
        os_encoder.SPECIES[os_encoder.draw_index(derive_seed(307, "os-sit-paired", 24, 0, s))] for s in (0, 1)]
    assert [s for _, _, s in pool.calls[-1]] == [derive_seed(305, "sit", 24, 0, s) for s in (0, 1)]   # paired seeds
    ev(3, 0, "new", "own")
    assert [s for _, _, s in pool.calls[-1]] == [derive_seed(305, "sit-new", 27, 0, s) for s in (0, 1)]
