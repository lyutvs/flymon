"""Stage 1 tau_rec: the pulse plan, the selection rule, the sampling rule (fake pool) and a real trajectory and taught
mask on the synthetic connectome; the battle-encoder odours on the real connectome when it is present."""
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flymon.rescope import taurec
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
# the synthetic-connectome Params tests/brain/test_fly_pool.py uses
P_SYN = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)


def test_pulse_plan():
    plan = taurec.pulse_plan(SPEC)
    assert len(plan) == 1000
    assert plan[0] == ("PAM08", 600.0) and plan[1] == ("PPL105", 400.0)
    assert sum(1 for d, _ in plan if d == "PAM08") == 500
    assert all(p == ("PAM08", 600.0) for p in plan[::2]) and all(p == ("PPL105", 400.0) for p in plan[1::2])


def _alt(ratio, floor):
    return {"alt": {"ratio": list(ratio), "floor_frac_taught": list(floor)},
            "same": {"ratio": [1.0], "floor_frac_taught": [1.0]}}          # same: record-only, never selects


def test_select_smallest_passing():
    res = {0.0: _alt([1.0, 0.45], [0.1]), 0.001: _alt([1.0, 0.6, 0.55], [0.2, 0.3]), 0.002: _alt([1.0, 0.9], [0.1])}
    out = taurec.select(res, SPEC)
    assert out["status"] == "SELECTED" and out["recovery_per_pulse"] == 0.001
    assert out["path_min"]["0.0"] == 0.45 and out["failed"] == {"0.0": ["a"], "0.001": [], "0.002": []}
    assert out["rule"] == "10.6 amendment 2026-09-28"


def test_select_zero_when_zero_passes():
    out = taurec.select({0.0: _alt([1.0, 0.7], [0.1, 0.4]), 0.001: _alt([1.0, 0.9], [0.0])}, SPEC)
    assert out["recovery_per_pulse"] == 0.0


def test_select_r0_like_fails_taught_floor():
    """r = 0.0 as observed: overall median stays 1.0 (vacuous) while 69 % of taught edges end at the floor."""
    res = {0.0: _alt([1.0] * 100, [0.0, 0.3, 0.55, 0.686]), 0.001: _alt([1.0, 0.99], [0.2, 0.45, 0.4]),
           0.002: _alt([1.0], [0.1])}
    out = taurec.select(res, SPEC)
    assert out["failed"]["0.0"] == ["b"] and out["floor_frac_taught_path_max"]["0.0"] == 0.686
    assert out["status"] == "SELECTED" and out["recovery_per_pulse"] == 0.001
    assert out["floor_frac_taught_path_max"]["0.001"] == 0.45


def test_select_uses_path_max_not_last():
    out = taurec.select({0.0: _alt([1.0], [0.1, 0.6, 0.2]), 0.02: _alt([0.8], [0.1])}, SPEC)
    assert out["failed"]["0.0"] == ["b"] and out["recovery_per_pulse"] == 0.02


def test_select_both_conditions_named():
    out = taurec.select({0.0: _alt([0.3], [0.9]), 0.01: _alt([0.6], [0.51]), 0.02: _alt([0.4], [0.2])}, SPEC)
    assert out["failed"] == {"0.0": ["a", "b"], "0.01": ["b"], "0.02": ["a"]}
    assert out["status"] == "STOP_NO_RECOVERY" and out["recovery_per_pulse"] is None


def test_select_boundary_is_inclusive():
    assert SPEC.taurec_taught_floor_max == 0.5 and SPEC.median_floor == 0.5
    assert taurec.select({0.0: _alt([0.5], [0.5])}, SPEC)["recovery_per_pulse"] == 0.0


def test_select_same_trajectory_is_record_only():
    res = {0.0: {"alt": {"ratio": [1.0], "floor_frac_taught": [0.1]}, "same": {"ratio": [0.1], "floor_frac_taught": [1.0]}}}
    assert taurec.select(res, SPEC)["recovery_per_pulse"] == 0.0


def test_select_no_taught_samples_fails_b():
    out = taurec.select({0.0: _alt([1.0], [None, None])}, SPEC)
    assert out["failed"]["0.0"] == ["b"] and out["floor_frac_taught_path_max"]["0.0"] is None


def test_select_stop():
    out = taurec.select({0.0: _alt([0.3], [0.1]), 0.02: _alt([0.4], [0.1])}, SPEC)
    assert out["status"] == "STOP_NO_RECOVERY" and out["recovery_per_pulse"] is None


class _FakePool:
    """Two flies of 4 edges; every reinforce multiplies the fly's weights by 0.9 and logs the request."""

    def __init__(self):
        self.flies = [type("F", (), {"shuffle_seed": None})()] * 2
        self.w0 = {None: np.ones(4, np.float32)}
        self.w = {0: np.ones(4, np.float32), 1: np.ones(4, np.float32)}
        self.calls = []

    def reinforce_batch(self, reqs, strength, settle_ms, gap_ms):
        self.calls.append((reqs, strength, settle_ms, gap_ms))
        for f, *_ in reqs:
            self.w[f] = self.w[f] * np.float32(0.9)


@pytest.mark.parametrize("pulses,every,sampled", [(6, 2, [2, 4, 6]), (5, 2, [2, 4, 5]), (1000, 10, list(range(10, 1001, 10))),
                                                  (20, 5, [5, 10, 15, 20]), (3, 5, [3])])
def test_sampling_rule(pulses, every, sampled):
    small = dataclasses.replace(SPEC, taurec_pulses=pulses, taurec_sample_every=every)
    pool = _FakePool()
    odours = [{"o0": 1.0}, {"o1": 1.0}, {"o2": 1.0}]
    out = taurec.trajectory(pool, odours, taurec.pulse_plan(small), small, np.array([True, True, False, False]))
    for key in ("alt", "same"):
        assert out[key]["pulse"] == sampled
        assert len(out[key]["ratio"]) == len(out[key]["q10_taught"]) == len(out[key]["floor_frac_taught"]) == len(sampled)
        assert out[key]["ratio"] == pytest.approx([0.9 ** p for p in sampled], rel=1e-5)
    assert len(pool.calls) == pulses


def test_trajectory_requests():
    small = dataclasses.replace(SPEC, taurec_pulses=5, taurec_sample_every=2)
    pool = _FakePool()
    odours = [{"o0": 1.0}, {"o1": 1.0}, {"o2": 1.0}]
    taurec.trajectory(pool, odours, taurec.pulse_plan(small), small, np.ones(4, bool))
    for i, (reqs, strength, settle, gap) in enumerate(pool.calls):
        dan, ms = taurec.pulse_plan(small)[i]
        seed = SPEC.taurec_seed_base + i
        assert reqs == [(0, odours[i % 3], dan, ms, seed), (1, odours[0], dan, ms, seed)]
        assert (strength, settle, gap) == (SPEC.strength, SPEC.train_settle_ms, SPEC.train_gap_ms)


def test_trajectory_and_taught_mask_on_synthetic(synthetic_npz):
    from flymon.brain.config import Params
    from flymon.brain.fly_pool import FlyPool, FlySpec
    small = dataclasses.replace(SPEC, taurec_pulses=6, taurec_sample_every=2, taurec_odours=3, strength=3.0,
                                train_settle_ms=50.0, probe_settle_ms=50.0, probe_read_ms=100.0)
    odours = [{"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}, {"ORN_DM6": 1.0}]
    with FlyPool(synthetic_npz, Params(**P_SYN), [FlySpec()], workers=1, timeout_s=120) as one:
        mask, n_active = taurec.taught_mask(one, odours[0], small)
    assert mask.dtype == bool and n_active > 0 and 0 < mask.sum() < mask.size
    with FlyPool(synthetic_npz, Params(**P_SYN), [FlySpec(), FlySpec()], workers=1, timeout_s=120) as pool:
        out = taurec.trajectory(pool, odours, taurec.pulse_plan(small), small, mask)
    for key in ("alt", "same"):
        assert out[key]["pulse"] == [2, 4, 6] and len(out[key]["ratio"]) == 3      # samples after pulses 2, 4, 6
        assert all(0.0 < r <= 1.0 for r in out[key]["ratio"])
        assert all(q is not None and 0.0 < q <= 1.0 for q in out[key]["q10_taught"])
        assert all(0.0 <= f <= 1.0 for f in out[key]["floor_frac_taught"])


needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


@needs_npz
def test_synthetic_odours_real():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    pops = Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))
    a = taurec.synthetic_odours(pops, 5, SPEC.taurec_gen_seed)
    b = taurec.synthetic_odours(pops, 5, SPEC.taurec_gen_seed)
    assert a == b and len(a) == 5 and len({tuple(sorted(o)) for o in a}) > 1
    for o in a:
        assert all(g in pops.receptor_types for g in o) and np.mean(list(o.values())) == pytest.approx(1.0)
        assert len(o) >= 6          # my, opp types (>= 1 each), move, power, my hp, opp hp


def _load():
    spec = importlib.util.spec_from_file_location("run_rescope_taurec", ROOT / "scripts" / "run_rescope_taurec.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_cli_refuses_outside_rescope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _load().main(["--out", "results/b/x", "--allow-dirty"])


def test_cli_smoke_paths_and_spec():
    m = _load()
    out, summ = m.paths("results/rescope/taurec", smoke=True)
    assert str(out).startswith("results/rescope-smoke/") and str(summ) == "results/rescope-smoke/summary/rescope_taurec.json"
    out, summ = m.paths("results/rescope/taurec", smoke=False)
    assert str(out) == "results/rescope/taurec" and str(summ) == "results/summary/rescope_taurec.json"
    s = m.smoke_spec()
    assert (s.taurec_pulses, s.taurec_sample_every, s.taurec_odours, s.recovery_grid) == (20, 5, 4, (0.0, 0.02))


def test_floor_frac_counts_float32_floored_edges():
    """The engine floors in float32 (np.maximum(w, w0 * min_weight_frac)); every such taught edge counts as floored."""
    from types import SimpleNamespace
    from flymon.rescope import taurec
    rng = np.random.default_rng(7)
    w0 = rng.uniform(0.1, 3.0, 2000).astype(np.float32)
    floored = np.maximum(np.float32(1e-6) * w0, w0 * np.float32(0.2)).astype(np.float32)
    assert (floored.astype(float) / w0.astype(float) > 0.2).any()
    w = np.concatenate([floored, (np.float32(0.5) * w0[:1000]).astype(np.float32)])
    pool = SimpleNamespace(w={0: w}, w0={None: np.concatenate([w0, w0[:1000]])}, flies=[SimpleNamespace(shuffle_seed=None)])
    mask = np.zeros(w.size, bool); mask[:2000] = True
    assert taurec._sample(pool, 0, mask)[2] == 1.0
    mask[2000:] = True
    assert taurec._sample(pool, 0, mask)[2] == 2000 / 3000


# ---- --reselect (spec 10.6 amendment 2026-09-28): fake per-r files in tmp dirs only -----------------------------
import json


def _write_grid(root, grid, per_r, sub="results/rescope/taurec"):
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    for r in grid:
        (d / f"r_{r}.json").write_text(json.dumps({"recovery_per_pulse": float(r), "trajectory": per_r[r]}))
    return d


@pytest.fixture
def reselect_env(tmp_path, monkeypatch):
    import flymon.agent.config as agent_config
    from flymon.brain.config import Params
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_config, "load_c3_config", lambda *a, **k: SimpleNamespace(params=Params()))
    m = _load()
    monkeypatch.setattr(m, "git_provenance", lambda files=(): {"commit": "test", "dirty": False, "dirty_files": []})

    def boom(*a, **k):
        raise AssertionError("--reselect must not build a pool")
    import flymon.brain.fly_pool as fp
    monkeypatch.setattr(fp, "FlyPool", boom)
    return tmp_path, m


from types import SimpleNamespace  # noqa: E402

GRID_DATA = {0.0: _alt([1.0, 1.0], [0.3, 0.686]), 0.001: _alt([1.0, 0.98], [0.2, 0.52]),
             0.002: _alt([1.0, 0.97], [0.2, 0.44]), 0.005: _alt([1.0, 0.99], [0.1, 0.3]),
             0.01: _alt([1.0], [0.1]), 0.02: _alt([1.0], [0.05])}


def test_reselect_writes_amended_summary_and_keeps_old(reselect_env):
    root, m = reselect_env
    _write_grid(root, SPEC.recovery_grid, GRID_DATA)
    summ = root / "results/summary/rescope_taurec.json"
    summ.parent.mkdir(parents=True)
    old = {"status": "SELECTED", "recovery_per_pulse": 0.0, "path_min": {"0.0": 1.0}, "provenance": {"x": 1},
           "n_taught_edges": 223, "n_active_kc": 40, "odours_seed": 990000, "trajectories": {}}
    summ.write_text(json.dumps(old))
    assert m.main(["--reselect", "--allow-dirty"]) == 0
    new = json.loads(summ.read_text())
    assert new["status"] == "SELECTED" and new["recovery_per_pulse"] == 0.002
    assert new["rule"] == "10.6 amendment 2026-09-28"
    assert new["failed"]["0.0"] == ["b"] and new["failed"]["0.001"] == ["b"]
    assert new["superseded_rule_v1"] == {"status": "SELECTED", "recovery_per_pulse": 0.0, "path_min": {"0.0": 1.0},
                                         "provenance": {"x": 1}}
    assert new["n_taught_edges"] == 223 and new["provenance"]["reselect"] is True
    assert set(new["provenance"]["per_r_sha256"]) == {f"r_{r}.json" for r in SPEC.recovery_grid}
    assert set(new["trajectories"]) == {str(float(r)) for r in SPEC.recovery_grid}
    # a second --reselect keeps the original old-rule record
    assert m.main(["--reselect", "--allow-dirty"]) == 0
    assert json.loads(summ.read_text())["superseded_rule_v1"]["recovery_per_pulse"] == 0.0


def test_reselect_without_old_summary_and_stop(reselect_env):
    root, m = reselect_env
    bad = {r: _alt([1.0], [0.9]) for r in SPEC.recovery_grid}
    _write_grid(root, SPEC.recovery_grid, bad)
    assert m.main(["--reselect", "--allow-dirty"]) == 0
    new = json.loads((root / "results/summary/rescope_taurec.json").read_text())
    assert new["status"] == "STOP_NO_RECOVERY" and new["recovery_per_pulse"] is None
    assert new["superseded_rule_v1"] is None and new["n_taught_edges"] is None


def test_reselect_refuses_missing_grid_value(reselect_env):
    root, m = reselect_env
    _write_grid(root, SPEC.recovery_grid[:-1], GRID_DATA)
    with pytest.raises(SystemExit, match="r_0.02.json"):
        m.main(["--reselect", "--allow-dirty"])
    assert not (root / "results/summary/rescope_taurec.json").exists()


def test_reselect_refuses_mismatched_value(reselect_env):
    root, m = reselect_env
    d = _write_grid(root, SPEC.recovery_grid, GRID_DATA)
    (d / "r_0.005.json").write_text(json.dumps({"recovery_per_pulse": 0.01, "trajectory": GRID_DATA[0.01]}))
    with pytest.raises(SystemExit, match="r_0.005.json"):
        m.main(["--reselect", "--allow-dirty"])


def test_reselect_smoke_paths(reselect_env):
    root, m = reselect_env
    _write_grid(root, (0.0, 0.02), GRID_DATA, sub="results/rescope-smoke/taurec")
    assert m.main(["--reselect", "--smoke", "--allow-dirty", "--out", "results/rescope-smoke/taurec"]) == 0
    new = json.loads((root / "results/rescope-smoke/summary/rescope_taurec.json").read_text())
    assert new["recovery_per_pulse"] == 0.02 and new["smoke"] is True and not (root / "results/summary").exists()
