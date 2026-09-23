"""The engine parameters of spec appendix J (J.11.3): `StdParams` = `Params` plus `receptor_scale`.

`receptor_scale` multiplies every receptor out-edge of the CSC (C2's diagnostic `csc.w` edit made a field), so the
ORN->PN depression scan can restore the overall gain and measure only the shape of the depression. It lives on a
subclass, not on `Params`, because a new `Params` field changes `dataclasses.asdict(Params())`, which G.8's judge
(`d6a.judge`), the M2 no-go writer and block "h3"'s records are compared against (J.10.8: a default-valued new field
must leave every earlier record and comparison unchanged). The engine reads the field with `getattr(p, name, 1.0)`,
so a plain `Params` is the unscaled engine and `StdParams(receptor_scale=1.0)` skips the multiply (bit-identical CSC).
"""
from __future__ import annotations

import dataclasses

from .config import Params


@dataclasses.dataclass(frozen=True)
class StdParams(Params):
    receptor_scale: float = 1.0    # multiplier on every receptor out-edge (spec J.11.3); 1.0 = no multiply


def params_from_json(d: dict) -> Params:
    """A recorded Params / StdParams dict (lists back to tuples) -> the same object; StdParams iff it has the field."""
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return StdParams(**d) if "receptor_scale" in d else Params(**d)


def with_std(base: Params, f: float, tau_ms: float, scale: float) -> StdParams:
    """`base` with ORN->PN depression on at (f, tau) and the receptor out-edges scaled by `scale`."""
    kw = {fl.name: getattr(base, fl.name) for fl in dataclasses.fields(Params)}
    return StdParams(**{**kw, "orn_std": True, "orn_std_f": float(f), "orn_std_tau_ms": float(tau_ms),
                        "receptor_scale": float(scale)})
