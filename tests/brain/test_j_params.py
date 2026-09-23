"""Spec J.11.3: StdParams.receptor_scale — the receptor out-edge multiplier of the ORN->PN depression scan."""
import dataclasses
import json

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.h3_store import canonical
from flymon.brain.j_params import StdParams, params_from_json, with_std
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide

BASE = dict(noise_mv=0.0, min_weight=1, balance_hemispheres=False, mbon_hold_frac=0.0)


def _csc(synthetic_connectome, params):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    return Engine(c, pops, params, seed=1), pops


def test_params_is_untouched_by_the_new_field():
    """J.10.8: asdict(Params()) is what G.8's judge, the no-go writer and block "h3" compare against."""
    assert "receptor_scale" not in dataclasses.asdict(Params())
    assert StdParams().receptor_scale == 1.0 and isinstance(StdParams(), Params)


def test_scale_one_keeps_the_csc_bit_identical(synthetic_connectome):
    a, _ = _csc(synthetic_connectome, Params(**BASE))
    b, _ = _csc(synthetic_connectome, StdParams(**BASE, receptor_scale=1.0))
    assert np.array_equal(a.csc.w, b.csc.w) and np.array_equal(a.csc.tgt, b.csc.tgt)


def test_scale_multiplies_exactly_the_receptor_out_edges(synthetic_connectome):
    base, pops = _csc(synthetic_connectome, Params(**BASE))
    scaled, _ = _csc(synthetic_connectome, StdParams(**BASE, receptor_scale=8.0))
    from_rec = base.is_receptor[base.csc.pre_of_edge()]
    assert from_rec.any() and (~from_rec).any()
    assert np.array_equal(scaled.csc.w[from_rec], (base.csc.w[from_rec] * np.float32(8.0)).astype(np.float32))
    assert np.array_equal(scaled.csc.w[~from_rec], base.csc.w[~from_rec])


def test_scale_matches_the_c2_diagnostic_edit(synthetic_connectome):
    """c2_std_feasibility.py multiplied the finished CSC's receptor out-edges; the field must agree bit for bit."""
    proto, _ = _csc(synthetic_connectome, Params(**BASE, orn_std=True))
    proto.csc.w[proto.is_receptor[proto.csc.pre_of_edge()]] *= np.float32(3.5)
    impl, _ = _csc(synthetic_connectome, StdParams(**BASE, orn_std=True, receptor_scale=3.5))
    assert np.array_equal(impl.csc.w, proto.csc.w)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_outside_the_range_is_rejected(synthetic_connectome, bad):
    with pytest.raises(ValueError, match="receptor_scale"):
        _csc(synthetic_connectome, StdParams(**BASE, receptor_scale=bad))


def test_json_round_trip_and_replace_keep_the_class():
    p = with_std(Params(apl_mode="graded", kc_thresh=1.65), 0.9, 300.0, 4.0)
    assert (p.orn_std, p.orn_std_f, p.orn_std_tau_ms, p.receptor_scale, p.kc_thresh) == (True, 0.9, 300.0, 4.0, 1.65)
    back = params_from_json(json.loads(json.dumps(dataclasses.asdict(p))))
    assert back == p and type(back) is StdParams
    assert type(dataclasses.replace(p, mbon_hold_frac=0.8)) is StdParams
    plain = params_from_json(json.loads(json.dumps(dataclasses.asdict(Params()))))
    assert plain == Params() and type(plain) is Params


def test_cache_keys_tell_the_classes_apart():
    assert canonical(StdParams()) != canonical(Params())


def test_pool_decide_equals_in_process_with_receptor_scale(synthetic_npz):
    """The field must survive the spawn boundary; the regime is chosen so the scale changes the counts."""
    kw = {**BASE, "orn_std": True, "kc_thresh": 0.2}
    params = StdParams(**kw, receptor_scale=4.0)
    conn = Connectome.load(synthetic_npz)
    pops = Populations.from_connectome(conn)
    ro = sorted(pops.receptor_types)[:2]
    odors = [{ro[0]: 1.0}, {ro[1]: 1.0}]

    def in_process(p):
        eng = Engine(conn, pops, p, seed=7)
        pl = Plasticity(eng, pops, compartments(conn, pops, p.core_frac))
        return np.asarray(decide(eng, pl, pops, odors, strength=1.0, seed=11, settle_ms=50.0, read_ms=300.0))

    scaled = in_process(params)
    assert not np.array_equal(scaled, in_process(StdParams(**kw, receptor_scale=1.0))), \
        "receptor_scale has no effect in this regime, so the pool comparison below cannot fail"
    with FlyPool(synthetic_npz, params, [FlySpec()], workers=1, timeout_s=120) as pool:
        pooled = pool.decide_batch([(0, odors, 11)], strength=1.0, settle_ms=50.0, read_ms=300.0)[0]
    assert np.array_equal(np.asarray(pooled), scaled)
