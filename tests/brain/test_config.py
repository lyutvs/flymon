from flymon.brain.config import Params


def test_shiu_constants_exact():
    p = Params()
    assert p.v_thresh == 7.0
    assert p.v_reset == 0.0
    assert p.tau_m == 20.0
    assert p.tau_syn == 5.0
    assert p.syn_delay_ms == 1.8
    assert p.refractory_ms == 2.2
    assert p.mv_per_synapse == 0.275
    assert p.dt == 1.0


def test_derived_steps():
    p = Params()
    assert p.dly_steps() == 2      # round(1.8 / 1.0)
    assert p.refrac_steps() == 2   # round(2.2 / 1.0)


def test_our_design_defaults():
    p = Params()
    assert p.apl_scale == 0.1
    assert p.mbon_hold_frac == 0.85
    assert p.kc_thresh == 1.5
    assert p.sign_override == (("lLN1", -1), ("lLN2", -1))
    assert p.min_weight == 5
    assert p.kc_kc_scale == 0.0   # KC->KC fast excitation removed (spec appendix D)


def test_params_from_an_older_run_loads_with_apl_input_scale_one():
    """Summaries written before H.3a have no apl_input_scale; they are the unscaled engine."""
    import dataclasses
    from flymon.brain.config import Params
    old = dataclasses.asdict(Params())
    old.pop("apl_input_scale")
    assert Params(**old).apl_input_scale == 1.0
