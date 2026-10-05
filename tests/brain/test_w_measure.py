"""W's measurement file (W.1, W.3 2, W.9.6 P2-10, W.9.8 H9): the W measurement key (U's files + w_measure.py), the
measurer's cache (one entry per pair · fly · brain, every input in the key, resume with the missing units only, the
budget check between rounds), and on the real connectome: W's job with P's punish-arm settings equals u_arm_job (L_V)
and r_arm_job ("none") field for field; its naive probes equal presentation.decide's counts (the oracle's probe) on
the same seeds; RN1 = R1 bit for bit (counts and weights) while N differs and R2 ≠ RN2; a rig whose edit is not applied
fails the reproduction (measured). RN's independence from R (W.9.6 P2-10) is pinned in test_w_records (the machine
check catches an RN that skipped its own reward phase) and test_w_runner (every RN unit carries both phases)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.brain import u_measure as UM
from flymon.brain import w_measure as WM
from flymon.brain.config import Params
from flymon.brain.r_measure import R_MEASURE_FILES
from flymon.brain.t_measure import T_MEASURE_FILES
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.w_store import WCache

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
READOUT = {"A": "MBON13", "P": "MBON05"}


def test_w_measure_key_is_u_files_plus_w_measure():
    assert WM.W_MEASURE_FILES == ("flymon/brain/w_measure.py",)
    files = set(R_MEASURE_FILES) | set(T_MEASURE_FILES) | set(UM.U_MEASURE_FILES)
    assert not set(WM.W_MEASURE_FILES) & files
    if NPZ.exists():
        k, u = WM.w_measure_key(str(NPZ)), UM.u_measure_key(str(NPZ))
        assert set(k["files"]) == set(u["files"]) | set(WM.W_MEASURE_FILES)
        assert k["key"] != u["key"] and len(k["key"]) == 64


class FakePool:
    def __init__(self, n=2):
        self.n_workers, self.calls = n, []

    def run_jobs(self, fn, kws):
        assert fn is WM.w_learn_job
        self.calls.append(len(kws))
        return [dict(fly=kw["fly"], probe_seeds=kw["probe_seeds"], stages=[], n=len(self.calls)) for kw in kws]


def _unit(fly, brain="R", k=2, trials=20, block_dan="PAM08"):
    return dict(pair="b|306|x|y", idx=0, fly=fly, brain=brain, edit=V.lever_edit, odor_x={"G1": 1.0},
                odor_y={"G2": 1.0}, phases=[[block_dan, trials, 28_000_000], [None, trials, 28_000_020]],
                probe_seeds=W.probe_seeds(0, fly, k), plastic=True)


def _measurer(pool, tmp_path):
    cache = WCache(tmp_path / "results/w/cache", {"key": "w" * 64})
    return WM.WMeasurer(pool, cache, Params(), READOUT, "MBON05", "PAM08", "PPL105",
                        dict(strength=1.0, settle_ms=800.0, read_ms=600.0, window_ms=200),
                        dict(present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, seed_stride=1000))


def test_cache_unit_resume_and_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/w").mkdir(parents=True)
    pool = FakePool(2)
    m = _measurer(pool, tmp_path)
    units = [_unit(f, b) for f in range(2) for b in ("R", "N", "RN")]
    got = m.learn(units[:3], "pilot")
    assert pool.calls == [2, 1] and [g["unit"]["brain"] for g in got] == ["R", "N", "RN"]
    got = m.learn(units, "pilot")                                      # resume: only the 3 missing units
    assert pool.calls == [2, 1, 2, 1] and len(got) == 6 and m.last_jobs == 3
    assert len({g["cache_key"] for g in got}) == 6
    base = m.inputs(units[0], "pilot")
    for change in (dict(probe_seeds=W.probe_seeds(0, 0, 3)), dict(fly=1), dict(brain="N"), dict(edit="none"),
                   dict(phases=[["PAM08", 19, 28_000_000], [None, 20, 28_000_020]]), dict(pair="a|1|x|y")):
        u = dict(units[0], **change)
        assert m.cache.key(WM.KIND, m.inputs(u, "pilot")) != m.cache.key(WM.KIND, base), change
    assert m.cache.key(WM.KIND, m.inputs(units[0], "learn")) != m.cache.key(WM.KIND, base)


def test_check_runs_before_each_round_and_can_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/w").mkdir(parents=True)
    pool = FakePool(2)
    m = _measurer(pool, tmp_path)
    seen = []

    class Stop(Exception):
        pass

    def check(done, todo):
        seen.append((done, todo))
        if done >= 2:
            raise Stop
    with pytest.raises(Stop):
        m.learn([_unit(f) for f in range(5)], "learn", check=check)
    assert seen == [(0, 5), (2, 5)] and pool.calls == [2]
    assert len(m.learn([_unit(f) for f in range(2)], "learn")) == 2 and pool.calls == [2]


# ================================================================ the real connectome
@pytest.fixture(scope="module")
def real():
    if not NPZ.exists():
        pytest.skip("no connectome")
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_spec import make_odors
    conn = Connectome.load(str(NPZ))
    pops = Populations.from_connectome(conn)
    o = make_odors(pops, H3.reference)[0]["strengths"]
    ks = sorted(o)
    return SimpleNamespace(conn=conn, pops=pops, params=load_c3_config(V.m0d_summary).params,
                           eng=SimpleNamespace(conn=conn), ox={k: o[k] for k in ks[: len(ks) // 2]},
                           oy={k: o[k] for k in ks[len(ks) // 2:]})


def _p_kw(real):
    n = V.p.o.n
    h4 = n.h4
    return dict(params=real.params, p_type="MBON05", odor_x=real.ox, odor_y=real.oy, readout=READOUT,
                punish_type="PPL105", reward_type="PAM08", strength=n.h3.strength,
                settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms),
                present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms, train_settle_ms=h4.teach_window.settle_ms)


@pytest.mark.parametrize("edit", [V.lever_edit, "none"])
def test_p_punish_arm_reproduced(real, edit):
    """W.3 2 (i) in miniature (2 trials): phases [[PPL105, 2, base]], probe seed = the arm seed, P's windows —
    every field of u_arm_job's punish arm (r_arm_job's for "none")."""
    from flymon.brain import w_records as WR
    kw, n, seed = _p_kw(real), V.p.o.n, 25_400_000
    ref = UM.u_arm_job(real.eng, None, real.pops, None, None, edit=edit, seed=seed, arm="punish", punish=True,
                       plastic=True, da_zero=False, trials=2, seed_base=n.train_seed_base,
                       seed_stride=n.train_seed_stride, **kw)
    mine = WM.w_learn_job(real.eng, None, real.pops, None, None, edit=edit, phases=[["PPL105", 2, n.train_seed_base]],
                          probe_seeds=[seed], fly=seed, seed_stride=n.train_seed_stride, plastic=True, **kw)
    assert WR.p_repro_diffs(mine, ref, "t") == []
    assert mine["block_edges"] == ({} if edit == "none" else dict(V.contrast_declared()["chain_entry"]))


def test_naive_probe_is_the_oracles_decide(real):
    """W.3 2 (ii) in miniature: W's naive probes (phases []) = presentation.decide's per-type sums on the same seeds."""
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.presentation import decide
    from flymon.brain import w_records as WR
    seeds = [24_600_200, 24_600_201]
    mine = WM.w_learn_job(real.eng, None, real.pops, None, None, params=real.params, edit=V.lever_edit,
                          p_type="MBON05", odor_x=real.ox, odor_y=real.oy, readout=READOUT, reward_type="PAM08",
                          punish_type="PPL105", phases=[], probe_seeds=seeds, strength=1.0, settle_ms=V.settle_ms,
                          read_ms=V.read_ms, window_ms=V.window_ms, present_ms=400.0, gap_ms=200.0,
                          train_settle_ms=800.0, fly=0, seed_stride=1000, plastic=True)
    e, p, *_ = UM.u_rig(real.conn, real.pops, real.params, V.lever_edit, "MBON05")
    cells = type_cells(e.conn, ["MBON13", "MBON05"])
    idx = np.concatenate([cells["MBON13"], cells["MBON05"]])
    n13 = len(cells["MBON13"])
    p.reset_weights()
    ref = []
    for s in seeds:
        c = decide(e, p, real.pops, [real.ox, real.oy], 1.0, s, V.settle_ms, V.read_ms, idx=idx)
        ref.append([[c[0, :n13].sum(), c[1, :n13].sum()], [c[0, n13:].sum(), c[1, n13:].sum()]])
    assert np.array_equal(WR.counts(mine["stages"][0]), np.asarray(ref)) and len(mine["stages"]) == 1


def _learn(real, dans, edit=V.lever_edit, trials=2):
    return WM.w_learn_job(real.eng, None, real.pops, None, None, params=real.params, edit=edit, p_type="MBON05",
                          odor_x=real.ox, odor_y=real.oy, readout=READOUT, reward_type="PAM08", punish_type="PPL105",
                          phases=[[dans[0], trials, 41_000_000], [dans[1], trials, 41_000_020]],
                          probe_seeds=[40_000_000], strength=1.0, settle_ms=V.settle_ms, read_ms=V.read_ms,
                          window_ms=V.window_ms, present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, fly=0,
                          seed_stride=1000, plastic=True)


def test_rn1_equals_r1_bitwise_and_n_differs(real):
    """G.5 / W.9.6 P2-10: RN runs from pre on its own; after the shared reward phase RN1 = R1 (counts and weights);
    N (no DAN) differs; the punishment phase separates R2 from RN2."""
    from flymon.brain import w_records as WR
    r, rn, n = _learn(real, ("PAM08", "PPL105")), _learn(real, ("PAM08", None)), _learn(real, (None, None))
    st = {k: {s["stage"]: s for s in v["stages"]} for k, v in (("R", r), ("RN", rn), ("N", n))}
    assert np.array_equal(WR.counts(st["R"]["S1"]), WR.counts(st["RN"]["S1"]))
    assert st["R"]["S1"]["w_sha256"] == st["RN"]["S1"]["w_sha256"] != st["N"]["S1"]["w_sha256"]
    assert st["R"]["S2"]["w_sha256"] != st["RN"]["S2"]["w_sha256"]
    assert np.array_equal(WR.counts(st["R"]["pre"]), WR.counts(st["N"]["pre"]))


def test_mutation_unapplied_edit_fails_the_reproduction(real, monkeypatch):
    """A rig whose combined edit never reaches the engine measures other counts than the real L_V rig (U.9.1's
    mutation, measured): the path gate would stop."""
    from flymon.brain import w_records as WR
    good = WM.w_learn_job(real.eng, None, real.pops, None, None, **dict(
        _p_kw(real), edit=V.lever_edit, phases=[], probe_seeds=[25_400_000], fly=0, seed_stride=1000, plastic=True))
    real_apply = UM.apply_u_edit
    monkeypatch.setattr(UM, "apply_u_edit", lambda eng, pops, edit, p_type: real_apply(eng, pops, "none", p_type))
    UM._RIG.clear()
    bad = WM.w_learn_job(real.eng, None, real.pops, None, None, **dict(
        _p_kw(real), edit=V.lever_edit, phases=[], probe_seeds=[25_400_000], fly=0, seed_stride=1000, plastic=True))
    UM._RIG.clear()
    assert not np.array_equal(WR.counts(good["stages"][0]), WR.counts(bad["stages"][0]))
