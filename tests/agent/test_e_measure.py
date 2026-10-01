import pytest

from flymon.agent.e_measure import EMeasurer
from flymon.agent.e_spec import SPEC, smoke
from flymon.agent.e_store import ECache
from tests.agent.e_scripted import ScriptedMeasurer


class FakePool:
    n_workers = 2

    def __init__(self, drop_oracle=False):
        self.calls, self.kws, self.drop_oracle = [], [], drop_oracle

    def run_jobs(self, fn, kws):
        self.calls.append((fn.__name__, len(kws)))
        self.kws.append((fn.__name__, kws))
        if fn.__name__ == "activity_job":
            return [[dict(i=i, seed=s, kc=list(range(int(10 * len(o)))), n=[1] * int(10 * len(o)), max_win=3)
                     for i, o, s in kw["items"]] for kw in kws]
        out = [dict(report={"pre": {}, "R1": {}, "R2": {}}, kc={"jaccard": 0.1}, odor=len(kw["odor_x"])) for kw in kws]
        return out[:-1] if self.drop_oracle else out


class P:  # params stand-in (the store's guard needs a full Params; tests pass guard_params=False)
    kc_kc_scale = 0.0


def _m(pool, spec):
    return EMeasurer(pool, ECache("results/encoder/cache", {"k": 1}), P(), n_kc=100, spec=spec, guard_params=False)


def test_activity_cached_and_resumed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool(); sp = smoke(SPEC)
    m = _m(pool, sp)
    od = {"a": {"ORN_X": 1.0}, "b": {"ORN_X": 1.0, "ORN_Y": 1.0}}
    r1 = m.activity(od, 0.35, sp.strength_seeds)
    n_calls = len(pool.calls)
    r2 = m.activity(od, 0.35, sp.strength_seeds)
    assert r1 == r2 and len(pool.calls) == n_calls
    assert r1["b"]["frac"][0] == 0.2
    assert r1["a"] == {"frac": [0.1, 0.1], "max_win": [3, 3]}
    kw = pool.kws[0][1][0]
    assert (kw["strength"], kw["settle_ms"], kw["read_ms"], kw["window_ms"]) == (0.35, 800.0, 600.0, 200)
    # a new odour alone is measured; the cached one is not re-run
    od["c"] = {"ORN_Z": 1.0}
    m.activity(od, 0.35, sp.strength_seeds)
    assert len(pool.kws[-1][1][0]["items"]) == 2


def test_activity_batches_at_most_64_items_one_job_per_worker_round(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = _m(pool, SPEC)
    od = {f"o{i}": {"ORN_X": 1.0} for i in range(20)}  # 20 x 8 seeds = 160 items -> 3 jobs -> rounds [2, 1]
    out = m.activity(od, 0.5, SPEC.strength_seeds)
    assert [c[1] for c in pool.calls] == [2, 1]
    assert all(len(kw["items"]) <= 64 for _, kws in pool.kws for kw in kws)
    assert len(out) == 20 and all(len(v["frac"]) == 8 for v in out.values())


def test_drive_odour_and_means(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = _m(pool, SPEC)
    out = m.drive(["ORN_A", "ORN_B"], 2.0, {"ORN_A": 4, "ORN_B": 1})
    kws = pool.kws[0][1]
    items = [it for kw in kws for it in kw["items"]]
    assert {it[2] for it in items} == set(SPEC.drive_seeds)
    assert {tuple(it[1].items()) for it in items} == {(("ORN_A", 0.5),), (("ORN_B", 2.0),)}
    assert kws[0]["strength"] == SPEC.drive_strength
    assert out["ORN_A"] == {"mean_spikes": 10.0, "kc_frac_mean": 0.1}
    n = len(pool.calls)
    assert m.drive(["ORN_A", "ORN_B"], 2.0, {"ORN_A": 4, "ORN_B": 1}) == out and len(pool.calls) == n


def test_oracle_rounds_and_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = FakePool()
    m = _m(pool, SPEC)
    rows = [dict(axis="b", turn=i, x=f"x{i}", y=f"y{i}", odor_x={"ORN_X": 1.0}, odor_y={"ORN_Y": 1.0}) for i in range(5)]
    seeds = dict(act=SPEC.even_act_seeds, select=SPEC.even_select_seeds, report=SPEC.even_report_seeds)
    out = m.oracle(rows, 0.35, {"A": "MBON13", "P": "MBON05"}, {"A": (1, 1), "P": (1, 1)}, ["MBON13", "MBON05"], seeds, "t")
    assert len(out) == 5 and [c[1] for c in pool.calls] == [2, 2, 1]
    assert out[3]["turn"] == 3 and out[3]["kc"] == {"jaccard": 0.1}
    kw = pool.kws[0][1][0]
    assert kw["alphas"] == [0.2, 0.5, 0.8] and kw["punish_type"] == "PPL105" and kw["reward_type"] == "PAM08"
    assert kw["act_seeds"] == list(SPEC.even_act_seeds) and kw["params"].__class__ is P
    m.oracle(rows, 0.35, {"A": "MBON13", "P": "MBON05"}, {"A": (1, 1), "P": (1, 1)}, ["MBON13", "MBON05"], seeds, "t")
    assert len(pool.calls) == 3


def test_oracle_never_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _m(FakePool(drop_oracle=True), SPEC)
    rows = [dict(axis="b", turn=i, x=f"x{i}", y=f"y{i}", odor_x={"ORN_X": 1.0 + i}, odor_y={"ORN_Y": 1.0}) for i in range(2)]
    seeds = dict(act=SPEC.even_act_seeds, select=SPEC.even_select_seeds, report=SPEC.even_report_seeds)
    with pytest.raises(RuntimeError):
        m.oracle(rows, 0.35, {"A": "MBON13", "P": "MBON05"}, {"A": (1, 1), "P": (1, 1)}, ["MBON13", "MBON05"], seeds, "t")


def test_scripted_measurer():
    sm = ScriptedMeasurer(testable={("b", 64, "x", "y"): True})
    a = sm.activity({"o": {"ORN_X": 1.0}}, 2.0, SPEC.strength_seeds)
    assert a["o"]["frac"] == [0.06] * 8
    rows = [dict(axis="b", turn=64, x="x", y="y"), dict(axis="b", turn=65, x="x", y="y")]
    out = sm.oracle(rows, 0.35, {}, {}, [], {}, "t")
    assert out[0]["report"]["R1"] != out[0]["report"]["pre"] and out[1]["report"]["R1"] == out[1]["report"]["pre"]
    assert out[0]["kc"] == {"jaccard": 0.2} and [c[0] for c in sm.calls] == ["activity", "oracle"]
