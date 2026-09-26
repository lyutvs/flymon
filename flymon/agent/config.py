"""The M3 agent's one configuration object: the C3 engine, its H.4 readout and z constants, the presentation window,
the decision-rule schedule (spec 3.5) and the safety floor (spec 4.3)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from ..brain.config import Params
from ..brain.j_store import load_c3


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    params: Params
    z: dict
    readout: dict
    strength: float = 0.35
    settle_ms: float = 800.0
    read_ms: float = 600.0
    tau_start: float = 1.0
    tau_end: float = 0.2
    tau_battles: int = 20
    reward_type: str = "PAM08"
    punish_type: str = "PPL105"
    median_floor: float = 0.5


def load_c3_config(summary_path: str | Path = "results/summary/m0d.json") -> AgentConfig:
    """C3's adopted Params (block "h3", threshold sha256 checked by load_c3) and its H.4 readout and z (block "h4")."""
    summary = json.loads(Path(summary_path).read_text())
    params, _, _ = load_c3(summary)
    c3 = summary["h4"]["h4"]["combos"]["C3"]
    z = {k: (float(v[0]), float(v[1])) for k, v in c3["z"].items()}
    return AgentConfig(params=params, z=z, readout=dict(c3["readout"]))


def config_hash(cfg: AgentConfig) -> str:
    d = dataclasses.asdict(cfg)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest()
