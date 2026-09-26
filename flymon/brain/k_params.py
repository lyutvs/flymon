"""The engine of spec appendix K (K.8.2): C3's `Params` with `kc_kc_scale` = g <= 0. `build_csc` multiplies every
KC->KC edge by g and drops them at 0 (C3's recorded value), so g < 0 is fast inhibition with no engine change and
g = 0 is C3 itself. J's `with_std` is not used: it forces the ORN depression on."""
from __future__ import annotations

import dataclasses
import math

from .config import Params
from .h3_runner import c1_params


def with_kc(base: Params, g: float) -> Params:
    if type(base) is not Params:
        raise TypeError(f"K builds on a plain Params, got {type(base).__name__} (no depression / receptor scale)")
    if base.orn_std:
        raise ValueError("K's engine has no ORN depression (orn_std must be False)")
    if not math.isfinite(g) or g > 0:
        raise ValueError(f"g must be finite and <= 0 (inhibition), got {g!r}")
    return dataclasses.replace(base, kc_kc_scale=float(g) + 0.0)       # + 0.0: -0.0 -> 0.0


def k_make(h3spec, g: float):
    """make_base for j_runner.reconverge: the cell's Params at apl_input_scale 1.0 with g applied."""
    return lambda kc: with_kc(c1_params(h3spec, kc, 1.0), g)
