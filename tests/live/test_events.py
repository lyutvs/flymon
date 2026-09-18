"""Viewer event contract: required fields per type, version, JSON with numpy values."""
import json

import numpy as np
import pytest

from flymon.live.events import REQUIRED, VERSION, make_event, to_json


def test_common_fields_and_version():
    ev = make_event("decision", "fm-a", "battle-gen1ou-1", np.int64(3), decider="fly", coach_kind="attack",
                    candidates=["surf", "earthquake"], chosen="surf")
    assert VERSION == 1
    assert ev == {"v": 1, "fly": "fm-a", "type": "decision", "battle_tag": "battle-gen1ou-1", "turn": 3,
                  "decider": "fly", "coach_kind": "attack", "candidates": ["surf", "earthquake"], "chosen": "surf"}
    assert type(ev["turn"]) is int


@pytest.mark.parametrize("etype", sorted(t for t, req in REQUIRED.items() if req))
def test_each_required_field_is_enforced(etype):
    full = {"decider": "fly", "coach_kind": "attack", "candidates": [], "chosen": None, "outcome": {},
            "phase": "decide", "slot": 0, "series": [], "won": True, "turns": 5, "lines": ["|turn|1"]}
    fields = {k: full[k] for k in REQUIRED[etype]}
    make_event(etype, "fm-a", "b", 1, **fields)
    for k in REQUIRED[etype]:
        with pytest.raises(ValueError, match=k):
            make_event(etype, "fm-a", "b", 1, **{f: v for f, v in fields.items() if f != k})


def test_protocol_lines_must_be_strings():
    make_event("protocol", "fm-a", "b", 1, lines=["|init|battle", "|turn|1"])
    for bad in ("|turn|1", [b"|turn|1"], [["|turn|1"]], None):
        with pytest.raises(ValueError, match="protocol lines"):
            make_event("protocol", "fm-a", "b", 1, lines=bad)


def test_unknown_type_and_bad_trace_are_rejected():
    with pytest.raises(ValueError, match="unknown event type"):
        make_event("spikes", "fm-a", "b", 1)
    ok = {"path": "rate_hz/kc", "kind": "scalar", "points": [[0.0, 1.0]]}
    make_event("trace", "fm-a", "b", 1, phase="reinforce", slot=None, series=[ok])
    with pytest.raises(ValueError, match="phase"):
        make_event("trace", "fm-a", "b", 1, phase="read", slot=0, series=[ok])
    for bad in ({**ok, "kind": "raster"}, {**ok, "path": 3}, {**ok, "points": None}, "rate_hz/kc"):
        with pytest.raises(ValueError, match="bad trace series"):
            make_event("trace", "fm-a", "b", 1, phase="decide", slot=0, series=[bad])


def test_detail_travels_as_given():
    detail = {"V": [0.5, -0.25], "readout": {"mbon": "MBON13"}}
    ev = make_event("decision", "fm-a", "b", 2, decider="fly", coach_kind="attack", candidates=["a", "b"],
                    chosen="a", detail=detail)
    assert ev["detail"] is detail


def test_to_json_accepts_numpy_and_refuses_nan():
    text = to_json({"V": np.array([0.5, 1.0], np.float32), "n": np.int32(4), "x": np.float64(0.25)})
    assert json.loads(text) == {"V": [0.5, 1.0], "n": 4, "x": 0.25}
    with pytest.raises(ValueError):
        to_json({"V": float("nan")})
    with pytest.raises(TypeError):
        to_json({"s": {1, 2}})
