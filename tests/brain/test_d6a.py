"""Spec G.8, the D.6 (a) re-judgement: the sliding-window count against a brute-force reference, the job's plasticity
hygiene, pool = in-process, the 41 odours = E.1's, and judge() at the 30/31-spike boundary and on incomplete records."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import d6a
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair, present

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
E1_META = ROOT / "results/m2/candidate_map_meta.json"          # git-excluded: E.1's encoder and turns
M2_PROBE = ROOT / "docs/superpowers/specs/m2-calibration-g/m2_probe.py"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="data/malecns.npz not present")

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
SYN = dict(strength=6.0, settle_ms=50.0, read_ms=100.0, window_ms=60)   # windows shorter than the presentation


@pytest.fixture
def rig(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, P, seed=0)
    return c, pops, eng, Plasticity(eng, pops, compartments(c, pops, P.core_frac))


def _reference_max_window(eng, pl, pops, odor, seed, strength, settle_ms, read_ms, window_ms):
    """Record every KC spike of the presentation, then take the largest count over all 1 ms-step windows."""
    kc = pops.kc
    pos = np.full(eng.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    pl.set_enabled(False)
    eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
    present(eng, pops, odor, strength)
    n = int(round((settle_ms + read_ms) / eng.p.dt))
    spikes = np.zeros((n, len(kc)), np.int64)
    for step in range(n):
        f = pos[eng.step()]; spikes[step, f[f >= 0]] = 1
    cum = np.vstack([np.zeros((1, len(kc)), np.int64), np.cumsum(spikes, 0)])
    pl.set_enabled(True)
    return max(int((cum[i] - cum[max(0, i - window_ms)]).max()) for i in range(1, n + 1))


def test_window_count_equals_a_brute_force_reference(rig):
    c, pops, eng, pl = rig
    a, b = design_odor_pair(pops, k=2, seed=0)
    got = d6a.d6a_job(eng, pl, pops, None, None, odors=[a, b], seed=7, **SYN)
    ref = [_reference_max_window(eng, pl, pops, o, 7, **SYN) for o in (a, b)]
    assert got["max_win"] == ref
    assert max(ref) > 1, "the synthetic regime must make a KC fire more than once per window"
    short = [_reference_max_window(eng, pl, pops, o, 7, **{**SYN, "window_ms": 20}) for o in (a, b)]
    assert short != ref, "the window length must matter in this regime, or a wrong length would pass"
    assert got["seed"] == 7 and len(got["kc_active_frac"]) == 2 and all(s > 0 for s in got["kc_spikes"])


def test_job_leaves_weights_reset_and_plasticity_on(rig):
    c, pops, eng, pl = rig
    a, b = design_odor_pair(pops, k=2, seed=0)
    pl.set_enabled(True)
    d6a.d6a_job(eng, pl, pops, None, None, odors=[a], seed=3, **SYN)
    assert pl.enabled and pl.weights_frac() == 1.0


def test_pool_equals_in_process(rig, synthetic_npz):
    c, pops, eng, pl = rig
    a, b = design_odor_pair(pops, k=2, seed=0)
    here = d6a.d6a_job(eng, pl, pops, None, None, odors=[a, b], seed=11, **SYN)
    with FlyPool(synthetic_npz, P, [{}], workers=1) as pool:
        there = pool.run_jobs(d6a.d6a_job, [dict(odors=[a, b], seed=11, **SYN)])[0]
    assert there == here


# ---------------------------------------------------------------- judge()
def _fake_odours():
    """41 odours over 16 turns (9 turns of 3 candidates, 7 of 2), like E.1's."""
    rows = []
    for t in range(16):
        for j in range(3 if t < 9 else 2):
            rows.append({"turn": t, "move": f"m{t}_{j}", "odor": {f"ORN_{t}_{j}": 1.0}})
    return rows


def _raw(max_win=30):
    odours = _fake_odours()
    per_turn = {}
    for o in odours:
        per_turn[o["turn"]] = per_turn.get(o["turn"], 0) + 1
    rows = [{"turn": t, "seed": s, "max_win": [max_win] * n, "kc_active_frac": [0.05] * n, "kc_spikes": [100] * n}
            for t, n in per_turn.items() for s in d6a.SEEDS]
    return {"smoke": False, "seeds": list(d6a.SEEDS), "params": d6a.engine_params(), "odours": odours,
            "strength": 0.35, "settle_ms": 800.0, "read_ms": 600.0, "window_ms": 200, "over_spikes": 31,
            "e2_check": {"ok": True, "cells": []}, "rows": rows}


@pytest.fixture
def fake_digest(monkeypatch):
    monkeypatch.setattr(d6a, "ODOURS_DIGEST", d6a.odours_digest(_fake_odours()))


def test_judge_30_spikes_is_not_over(fake_digest):
    v = d6a.judge(_raw(30))
    assert (v["fired"], v["n_over"], v["n_presentations"]) == (False, 0, 41 * 64)
    assert (v["max_win_spikes"], v["max_win_hz"], v["margin_spikes"]) == (30, 150.0, 0)


def test_judge_one_presentation_at_31_spikes_fires(fake_digest):
    raw = _raw(30)
    row = next(r for r in raw["rows"] if r["turn"] == 12 and r["seed"] == 431)
    row["max_win"][1] = 31
    v = d6a.judge(raw)
    assert (v["fired"], v["n_over"], v["max_win_spikes"], v["margin_spikes"]) == (True, 1, 31, -1)
    assert v["over"] == [{"turn": 12, "move": "m12_1", "seed": 431, "max_win": 31}]


@pytest.mark.parametrize("breakage, message", [
    (lambda r: r["rows"].pop(), "cover"),
    (lambda r: r["rows"].append(copy.deepcopy(r["rows"][0])), "duplicate"),
    (lambda r: r.update(smoke=True), "smoke"),
    (lambda r: r.update(seeds=list(range(400, 463))), "400-463"),
    (lambda r: r["params"].update(kc_kc_scale=1.0), "M0c"),
    (lambda r: r.update(window_ms=100), "window_ms"),
    (lambda r: r.update(over_spikes=30), "over_spikes"),
    (lambda r: r.update(e2_check={"ok": False}), "self-check"),
    (lambda r: r.pop("e2_check"), "self-check"),
    (lambda r: r["odours"].pop(), "41"),
    (lambda r: r["rows"][0]["max_win"].pop(), "does not match"),
    (lambda r: r["rows"][0]["max_win"].__setitem__(0, 30.5), "does not match"),
])
def test_judge_refuses_an_incomplete_or_foreign_record(fake_digest, breakage, message):
    raw = _raw(30)
    breakage(raw)
    with pytest.raises(ValueError, match=message):
        d6a.judge(raw)


def test_the_real_digest_rejects_fake_odours():
    with pytest.raises(ValueError, match="41"):
        d6a.judge(_raw(30))


# ---------------------------------------------------------------- the real 41 odours
@needs_npz
def test_candidate_odours_are_e1s_41():
    pops = Populations.from_connectome(Connectome.load(NPZ))
    rows = d6a.candidate_odours(pops)
    assert len(rows) == d6a.N_ODOURS == 41 and len({r["turn"] for r in rows}) == 16
    assert d6a.odours_digest(rows) == d6a.ODOURS_DIGEST
    assert all(6 <= len(r["odor"]) <= 8 for r in rows)
    if E1_META.exists():            # bit for bit, in order, against the committed calibration code on E.1's record
        spec = importlib.util.spec_from_file_location("m2_probe", M2_PROBE)
        m2_probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2_probe)
        meta = json.loads(E1_META.read_text())
        ref = [o for t in meta["turns"] for o in m2_probe.turn_odors(pops, meta["channels"], t)]
        assert [list(r["odor"].items()) for r in rows] == [list(o.items()) for o in ref]


def test_engine_is_h3s_c0():
    """Params() = the M0c engine = H.3's C0 (every H.2 mode off), as recorded in the committed m0d.json."""
    h3 = json.loads((ROOT / "results/summary/m0d.json").read_text())["h3"]
    assert h3["combos"]["C0"]["adopted"]["params"] == d6a.engine_params()
