"""The qualification and primary CLIs: refusals, the naive / oracle calls the qualification makes, and the primary
driver over qualified pairs on the fake pool (never the real connectome)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.config import Params
from flymon.rescope import primary, rules
from flymon.rescope.spec import SPEC

from .test_primary import CELLS, ODORS, SMALL, fake

ROOT = Path(__file__).resolve().parents[2]
QUIET = lambda s: None


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_primary_refuses_without_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])


def test_primary_refuses_on_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": "STOP_FEW_PAIRS", "control_qualified": True, "qualified": []}))
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])


def test_primary_refuses_on_unqualified_control(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": None, "control_qualified": False, "qualified": ["p1000"] * 4}))
    with pytest.raises(SystemExit):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])


def test_primary_smoke_reads_the_smoke_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": None, "control_qualified": True, "qualified": ["p1000"] * 4}))
    with pytest.raises(SystemExit):                     # the real one exists, the smoke one does not
        load("run_rescope_primary").main(["--smoke", "--allow-dirty"])


def test_qualify_refuses_outside_rescope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load("run_rescope_qualify").main(["--out", "results/b/x", "--allow-dirty"])


def test_smoke_paths():
    q = load("run_rescope_qualify")
    out, summ = q.paths("results/rescope/qualify", smoke=True, stage="qualify")
    assert str(out).startswith("results/rescope-smoke/") and str(summ) == "results/rescope-smoke/summary/rescope_qualify.json"
    out, summ = q.paths("results/rescope/qualify", smoke=False, stage="qualify")
    assert str(out) == "results/rescope/qualify" and str(summ) == "results/summary/rescope_qualify.json"


class NaivePool:
    n_flies = 3

    def __init__(self):
        self.calls = []

    def decide_batch(self, reqs, strength, settle_ms, read_ms, idx):
        self.calls.append(dict(reqs=reqs, strength=strength, settle_ms=settle_ms, read_ms=read_ms, idx=idx))
        return [np.array([[10 + s % 7, 1], [20, 2]]) for _, _, s in reqs]      # a: 11+s%7, b: 22


class OraclePool:
    def __init__(self, good):
        self.good, self.calls = good, []

    def run_jobs(self, fn, kwargs_list):
        self.calls.append((fn, kwargs_list))
        pre = {"A": [[10, 10]] * 8, "P": [[30, 30]] * 8}
        r1 = {"A": [[10, 10]] * 8, "P": [[30 - 12 - (i % 2), 30] for i in range(8)]}
        return [dict(alpha_reward=0.5, report={"pre": pre, "R1": r1 if self.good(kw) else pre}) for kw in kwargs_list]


def test_naive_covers_every_select_and_report_seed():
    q = load("run_rescope_qualify")
    pool = NaivePool()
    idx = np.array([5, 6])
    nv = q.naive_counts(pool, SPEC, "p1001", ODORS, idx)
    seeds = SPEC.qual_seeds("p1001")
    assert len(nv["a"]) == len(nv["b"]) == len(seeds["select"]) + len(seeds["report"])
    got = [s for c in pool.calls for _, _, s in c["reqs"]]
    assert got == seeds["select"] + seeds["report"]
    assert all(cands == [ODORS["a"], ODORS["b"]] for c in pool.calls for _, cands, _ in c["reqs"])
    assert all(c["strength"] == SPEC.strength and c["settle_ms"] == SPEC.probe_settle_ms
               and c["read_ms"] == SPEC.probe_read_ms and np.array_equal(c["idx"], idx) for c in pool.calls)
    assert nv["a"] == [11 + s % 7 for s in got] and nv["b"] == [22] * len(got)


def test_measure_passes_c3_z_and_qualifies():
    from flymon.rescope.oracle import reward_oracle_job
    q = load("run_rescope_qualify")
    names = SPEC.pair_names()
    odors_of = {n: {"a": {f"ORN_{n}_a": 1.0}, "b": {f"ORN_{n}_b": 1.0}} for n in names}
    bad = {"p1004", "p1005"}
    oracle = OraclePool(good=lambda kw: not any(n in next(iter(kw["odor_x"])) for n in bad))
    params = Params()
    pairs = q.measure(NaivePool(), oracle, SPEC, params, odors_of, np.array([1]), log=QUIET)
    fn, kws = oracle.calls[0]
    assert fn is reward_oracle_job and len(kws) == len(names)
    for n, kw in zip(names, kws):
        s = SPEC.qual_seeds(n)
        x = pairs[n]["x"]
        assert x == "b"                                   # naive b (22) > a (<= 17)
        assert kw["odor_x"] == odors_of[n][x] and kw["odor_y"] == odors_of[n]["a"]
        assert kw["z"] == {"A": SPEC.z_a, "P": SPEC.z_p} and kw["readout"] == {"A": SPEC.a_type, "P": SPEC.p_type}
        assert kw["params"] is params and kw["reward_type"] == SPEC.reward_dan
        assert list(kw["act_seeds"]) == s["act"] and list(kw["select_seeds"]) == s["select"]
        assert list(kw["report_seeds"]) == s["report"] and tuple(kw["alphas"]) == SPEC.oracle_alphas
        assert len(pairs[n]["naive"]["a"]) == len(s["select"]) + len(s["report"])
        assert pairs[n]["alpha_reward"] == 0.5
    summ = q.summarize(SPEC, pairs)
    assert summ["qualified"] == ["p1000", "p1001", "p1002", "p1003"] and summ["m"] == 4 and summ["stop"] is None
    assert summ["control_qualified"] is True
    pairs["p1003"]["qualified"] = False
    assert q.summarize(SPEC, pairs)["stop"] == "STOP_FEW_PAIRS"


class XFakePool(type(fake())):
    """The copied FakePool plus what the X-core record needs: w0, run_jobs (xcore_job's answer for 4 plastic edges:
    post cells MBON05 on edges 0, 2, 3; pre KCs 0, 1, 2, 2) and KC reads (decide_batch with 3 KC indices; the fake's
    counts are all > 0, so every KC is active and the mask is edges 0, 2, 3)."""

    def __init__(self, flies, fail_at=None):
        super().__init__(flies, fail_at)
        self.w0 = {None: np.ones(4, np.float32)}
        self.jobs = []

    def run_jobs(self, fn, kwargs_list):
        self.jobs.append((fn, kwargs_list))
        return [dict(post_in=np.array([True, False, True, True]), pre_kc=np.array([0, 1, 2, 2]),
                     kc=np.array([10, 11, 12])) for _ in kwargs_list]


def xfake(spec=SMALL):
    from flymon.brain.b_runner import fly_specs
    return XFakePool(fly_specs(primary.layout(spec)))


def test_primary_driver_runs_control_then_qualified_and_resets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = load("run_rescope_primary")
    qual = {"qualified": ["p1000", "p1001"], "pairs": {"seed0": {"x": "a"}, "p1000": {"x": "b"}, "p1001": {"x": "b"}}}
    pool = xfake()
    res = p.run_all(pool, SMALL, qual, {n: ODORS for n in SMALL.pair_names()}, CELLS, Path("results/rescope/primary"),
                    "abc", [Params()], {"commit": "abc"}, log=QUIET, floor_frac=0.2)
    assert set(res["pairs"]) == {"p1000", "p1001"} and res["control"]["pair"] == "seed0"
    ref = primary.run_pair(fake(), SMALL, "p1001", ODORS, CELLS, log=QUIET)
    raw = json.loads(Path("results/rescope/primary/p1001/records.json").read_text())
    assert raw["records"] == ref["records"]                # the pool was reset to naive between pairs
    assert res["overall"] == rules.overall(res["control"], res["pairs"], SMALL)
    rec = res["recorded"]
    for n in ("seed0", "p1000", "p1001"):
        assert set(rec[n]) == {"level", "sign", "naive", "qual_x_vs_x", "naive_dprime", "choice", "self_change",
                               "xcore", "xcore_mask", "flies", "silent"}
        assert rec[n]["silent"] == rules.silent_states(json.loads(Path(f"results/rescope/primary/{n}/records.json")
                                                                  .read_text())["records"], SMALL)
        xc = rec[n]["xcore"]                               # Rr: 3 reward trials x 0.9; N: no DAN, unchanged
        assert xc["Rr"]["median"] == pytest.approx(0.9 ** SMALL.trials) and xc["N"]["median"] == 1.0
        assert xc["ratio_rr_over_n"] == pytest.approx(0.9 ** SMALL.trials) and xc["n_edges"] == 3
        assert xc["Rr"]["floor_contact"] == 0.0 and xc["floor_frac"] == 0.2
        assert rec[n]["xcore_mask"]["n_edges"] == {"a": 3, "b": 3}
        assert rec[n]["xcore_mask"]["seed"] == SMALL.probe_seeds(n, 0)[0]
        assert rec[n]["qual_x_vs_x"] == (qual["pairs"][n]["x"] == raw["x"] if n == "p1001" else rec[n]["qual_x_vs_x"])
    assert Path("results/rescope/primary/seed0/records.json").exists()


# ---- final-review fixes: X-core mask timing, pairs digest, recovery, qualification stop precedence -------------
def test_xcore_mask_built_naive_and_weights_read_before_reset(tmp_path, monkeypatch):
    """The mask is built on the naive pool before run_pair (no reinforcement yet) and the weights are read after
    run_pair and before the next pair's reset."""
    monkeypatch.chdir(tmp_path)
    p = load("run_rescope_primary")
    pool = xfake()
    seen = {}
    real_masks, real_weights = primary.xcore_masks, primary.xcore_weights

    def masks(pl, *a, **k):
        seen.setdefault("mask_reinforced", []).append(pl.n_reinforce)
        return real_masks(pl, *a, **k)

    def weights(pl, lay, mask):
        seen.setdefault("w_rr0", []).append(float(pl.w[0][0]))
        return real_weights(pl, lay, mask)
    monkeypatch.setattr(primary, "xcore_masks", masks)
    monkeypatch.setattr(primary, "xcore_weights", weights)
    qual = {"qualified": ["p1000"], "pairs": {}}
    p.run_all(pool, SMALL, qual, {n: ODORS for n in SMALL.pair_names()}, CELLS, Path("results/rescope/primary"),
              "abc", [Params()], {"commit": "abc"}, log=QUIET, floor_frac=0.2)
    assert seen["mask_reinforced"] == [0, SMALL.trials]           # before any trial of each pair
    assert seen["w_rr0"] == pytest.approx([0.9 ** SMALL.trials] * 2)
    from flymon.rescope.primary import xcore_job
    assert pool.jobs[0][0] is xcore_job and np.array_equal(pool.jobs[0][1][0]["cells"], CELLS[SPEC.p_type])


def test_primary_checks_pairs_digest():
    p = load("run_rescope_primary")
    good = SPEC.pairs_digest
    assert p.check_pairs_digest(good, SPEC, {"pairs_digest": good}) == good
    with pytest.raises(SystemExit, match="pinned"):
        p.check_pairs_digest("0" * 64, SPEC, {"pairs_digest": "0" * 64})
    with pytest.raises(SystemExit, match="qualification"):
        p.check_pairs_digest(good, SPEC, {"pairs_digest": "1" * 64})
    with pytest.raises(SystemExit, match="qualification"):
        p.check_pairs_digest(good, SPEC, {})


def test_check_recovery():
    import dataclasses
    assert primary.check_recovery(Params()) == 0.0
    with pytest.raises(SystemExit, match="recovery_per_pulse"):
        primary.check_recovery(dataclasses.replace(Params(), recovery_per_pulse=0.01))


def _c3_with_recovery(r):
    import dataclasses
    from types import SimpleNamespace
    return lambda *a, **k: SimpleNamespace(params=dataclasses.replace(Params(), recovery_per_pulse=r), z={}, readout={})


@pytest.mark.parametrize("script", ["run_rescope_qualify", "run_rescope_primary"])
def test_cli_refuses_nonzero_recovery(tmp_path, monkeypatch, script):
    monkeypatch.chdir(tmp_path)
    import flymon.agent.config as agent_config
    import flymon.brain.connectome as connectome
    monkeypatch.setattr(agent_config, "load_c3_config", _c3_with_recovery(0.005))
    monkeypatch.setattr(connectome.Connectome, "load", classmethod(lambda cls, *a: pytest.fail("loaded the connectome")))
    q = tmp_path / "results/summary/rescope_qualify.json"; q.parent.mkdir(parents=True)
    q.write_text(json.dumps({"stop": None, "control_qualified": True, "qualified": ["p1000"] * 4,
                             "pairs_digest": SPEC.pairs_digest}))
    mod = load(script)
    mod.git_provenance = lambda files=(): {"commit": "c" * 40, "dirty": False, "dirty_files": []}
    with pytest.raises(SystemExit, match="recovery_per_pulse is 0.005"):
        mod.main(["--allow-dirty"])


def test_qualify_unqualified_control_stops_control_invalid_first():
    q = load("run_rescope_qualify")
    names = SPEC.pair_names()
    pairs = {n: {"qualified": True} for n in names}
    assert q.summarize(SPEC, pairs)["stop"] is None
    pairs[SPEC.control]["qualified"] = False                     # seed0 fails, 6 pairs qualified
    s = q.summarize(SPEC, pairs)
    assert s["stop"] == "STOP_CONTROL_INVALID" and s["control_qualified"] is False
    for n in names[1:4]:
        pairs[n]["qualified"] = False                            # and too few pairs: the control still wins
    assert q.summarize(SPEC, pairs)["stop"] == "STOP_CONTROL_INVALID"
    pairs[SPEC.control]["qualified"] = True
    assert q.summarize(SPEC, pairs)["stop"] == "STOP_FEW_PAIRS"


def test_primary_refuses_a_control_invalid_qualification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "results/summary/rescope_qualify.json"; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"stop": "STOP_CONTROL_INVALID", "control_qualified": False, "qualified": ["p1000"] * 6}))
    with pytest.raises(SystemExit, match="STOP_CONTROL_INVALID"):
        load("run_rescope_primary").main(["--out", "results/rescope/primary", "--allow-dirty"])
