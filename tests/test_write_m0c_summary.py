"""scripts/write_m0c_summary.py composes results/summary/m0c.json from the M0c result files (spec D.4/D.5)."""
import importlib.util
from pathlib import Path

import pytest

from flymon.brain.config import Params

_spec = importlib.util.spec_from_file_location("write_m0c_summary", Path("scripts/write_m0c_summary.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def _row(**kw):
    r = {"kc_thresh": 1.5, "apl_scale": 0.1, "mbon_hold_frac": 0.85, "kc_kc_scale": 0.0, "sparsity_seeds": [100, 101, 102],
         "frac_active_A": 0.064, "frac_active_B": 0.049,
         "jaccard": 0.025, "chance": 0.028, "mbon_hz_A": 15.7, "mbon_hz_B": 18.1, "mbon_hz_rest": 9.3, "mbon_hz_rest_trimmed": 3.3,
         "mbon_hz_rest_trimmed_sd": 0.8, "mbon_hz_rest_trimmed_per_seed": [3.0, 3.6] * 4, "mbon_n_saturated": 2.0,
         "mbon_types_active_rest": 35.0, "rest_seeds": list(range(100, 108)), "rest_ms": 3000.0}
    r.update(kw)
    return r


def test_pick_row_never_matches_an_old_engine_row():
    old = _row(); del old["kc_kc_scale"]          # M0 files predate the key: they are the 1.0 engine
    assert w.pick_row([old], Params(kc_kc_scale=1.0)) is old
    with pytest.raises(SystemExit, match="kc_kc_scale=0.0"):
        w.pick_row([old], Params())
    new = _row()
    assert w.pick_row([old, new], Params()) is new


def _arm(dD):
    return {"mean_dD": dD, "mean_dD_disc": dD}


def _reproduce(seeds, n_rest=8, n_odor=64, odor_over=0, scale=0.0):
    cond = {"n_seeds": len(seeds), "n_flip": 5, "n_flip_disc": 7, "noplast_max_abs_dD": 0.0, "channel_specific_seeds": 7,
            "arms": {"both": _arm(0.0), "reversed": _arm(1.49), "noplast": _arm(0.0), "punish_only": _arm(0.1), "reward_only": _arm(0.2)},
            "per_seed": {}}
    base = [{"mbon_hz": 9.0, "mbon_hz_trimmed": 3.3, "n_saturated": 2, "n_types_active": 35,
             "runaway": {"sat_hz": 100.0, "n_over_sat": 120, "n_kc_over_sat": 0, "spike_share_over_sat": 0.3}} for _ in range(n_rest)]
    odor = [{"seed": 100 + i, "which": "B", "sat_hz": 150.0, "n_over_sat": 130, "n_kc_over_sat": odor_over if i == 0 else 0,
             "n_kc_over_100": 2, "kc_hz_top5": [111.0, 106.0, 95.0, 95.0, 86.0], "frac_active_kc": 0.05, "kc_spikes": 1200} for i in range(n_odor)]
    return {"params": {"kc_kc_scale": scale}, "seeds": seeds, "conditioning": cond,
            "conditioning_match": {"ok": True, "n_results": 40, "n_equal": 40, "n_missing": 0, "missing": [], "max_abs_diff": 0.0},
            "sparsity_match": {"ok": True, "diffs": {"mbon_hz_rest_trimmed": 0.0}, "max_abs_diff": 0.0},
            "baseline": {"mbon_hz_rest": 9.0, "mbon_hz_rest_trimmed": 3.3, "mbon_hz_rest_trimmed_sd": 0.8, "per_seed": base,
                         "runaway": {"n_over_sat": 120.0, "n_kc_over_sat": 0.0, "spike_share_over_sat": 0.3}},
            "odor_runaway": {"which": "B", "sat_hz": 150.0, "n_seeds": n_odor, "per_seed": odor,
                             "seeds_with_kc_over_100": n_odor, "seeds_with_kc_over_sat": int(odor_over > 0), "max_n_kc_over_sat": odor_over},
            "arm_equal": {"pairs": {"8/both": {"equal": True}, "15/reversed": {"equal": True}}, "ok": True},
            "decide_equal": True, "memory": {"worker_rss_GB": 0.6}, "wall_clock_s": 500.0}


def _throughput(scale=0.0):
    """Shaped like results/m0b/throughput.json rows (C.7 numbers): 16 workers is the fastest end to end."""
    return {"kc_kc_scale": scale, "rows": [{"workers": W, "n_flies_batch": W, "s_decide_batch_max": d, "s_reinforce_batch_max": r, "ms_decision_max": m, "ms_reinforce_max": m}
                     for W, d, r, m in ((4, 9.8, 2.8, 1.8), (8, 11.1, 3.5, 2.2), (16, 16.1, 5.3, 4.4))]}


def _old():
    return _reproduce(list(range(8)), scale=1.0)


def test_compose_records_every_gate_term_and_the_seed_policy():
    p = Params()
    out = w.compose(p, {"grid": [_row()]}, _old(), _reproduce(list(range(8, 16))), _reproduce(list(range(8))), _throughput(),
                    provenance={"sparsity": "abc"})
    assert out["params_frozen"]["kc_kc_scale"] == 0.0 and out["kc_kc_scale_old_engine"] == 1.0
    assert out["inputs_sha256"] == {"sparsity": "abc"} and "git_dirty" in out
    assert out["seeds"]["sparsity"] == [100, 101, 102]
    assert out["seeds"]["conditioning_judged"] == list(range(8, 16)) and out["seeds"]["conditioning_reported"] == list(range(8))
    assert out["runaway"]["odor_B_sat_hz"] == 150.0 and out["runaway"]["odor_B_kc_hz_max"] == 111.0
    assert out["runaway"]["rest_n_kc_over_sat_per_seed"] == [0] * 8 and out["runaway"]["odor_B_n_kc_over_100_per_seed"] == [2] * 64
    assert out["equivalence"]["old_conditioning"]["n_results"] == 40 and out["equivalence"]["arm_equal"]["ok"]
    assert out["budget"]["limit_hours"] == 60.0 and out["budget"]["workers"] == 16
    g = out["gate"]
    assert g["passed"] is True and g["conditioning_index_flip_ok"] is False and g["channel_specific_seeds"] == 7
    assert out["conditioning"]["n_flip"] == 5 and out["conditioning_reported_seeds"]["n_seeds"] == 8


def test_compose_writes_a_failed_gate_instead_of_refusing():
    p = Params()
    out = w.compose(p, {"grid": [_row(mbon_hz_rest_trimmed=2.94)]}, _old(),
                    _reproduce(list(range(8, 16)), odor_over=52), None, _throughput())
    g = out["gate"]
    assert g["baseline_ok"] is False and g["runaway_ok"] is False and g["passed"] is False
    assert out["conditioning_reported_seeds"] is None and out["seeds"]["conditioning_reported"] is None
    assert out["runaway"]["odor_B_seeds_with_kc_over_sat"] == 1


@pytest.mark.parametrize("bad, match", [
    (lambda k: dict(old=_reproduce(list(range(8)), scale=0.0)), "old engine"),
    (lambda k: dict(new=_reproduce(list(range(8, 16)), scale=1.0)), "new engine"),
    (lambda k: dict(new=_reproduce(list(range(8, 15)))), "8-15"),
    (lambda k: dict(new=_reproduce(list(range(8, 16)), n_rest=3)), "8 rest seeds"),
    (lambda k: dict(new=_reproduce(list(range(8, 16)), n_odor=3)), "100-163"),
    (lambda k: dict(sp={"grid": [_row(rest_seeds=[100, 101, 102])]}), "100-107"),
    (lambda k: dict(th={"kc_kc_scale": 0.0, "rows": _throughput()["rows"][:2]}), "workers 4, 8, 16"),
    (lambda k: dict(th=_throughput(scale=1.0)), "throughput must be the new engine"),
])
def test_compose_refuses_wrong_identity_or_short_runs(bad, match):
    """Spec D.4: the gate is defined with its sample sizes and engine identity; a short or mismatched run is refused
    with the shortfall named, never scored."""
    kw = dict(sp={"grid": [_row()]}, old=_old(), new=_reproduce(list(range(8, 16))), reported=None, th=_throughput())
    kw.update(bad(kw))
    with pytest.raises(SystemExit, match=match):
        w.compose(Params(), kw["sp"], kw["old"], kw["new"], kw["reported"], kw["th"])


def test_main_refuses_to_write_over_an_old_engine_summary(monkeypatch, tmp_path):
    """Spec D.5: results/summary/m0.json and m0b.json are the old engine's immutable record. The guard runs on the
    raw --out before any input is read, so the refusal costs nothing and does not depend on what exists: with every
    input pointed at a missing file, an old-engine --out still fails on the guard, a new one on the missing input."""
    missing: list[str] = []
    for k in ("sparsity", "reproduce-old", "reproduce", "reproduce-reported", "throughput"):
        missing += [f"--{k}", str(tmp_path / f"{k}.json")]
    for out in ("results/summary/m0.json", "results/summary/m0b.json", "results/m0/sparsity.json"):
        monkeypatch.setattr("sys.argv", ["write_m0c_summary.py", "--out", out, *missing])
        with pytest.raises(SystemExit, match="old engine"):
            w.main()
    monkeypatch.setattr("sys.argv", ["write_m0c_summary.py", "--out", str(tmp_path / "m0c.json"), *missing])
    with pytest.raises(SystemExit, match="missing input"):   # a new path gets past the guard, on to the inputs
        w.main()
