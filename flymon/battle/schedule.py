"""Deterministic battle schedules shared by every experimental arm."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .pool import POOL


@dataclass
class ScheduledBattle:
    battle_id: str
    fly_id: int
    my_team: list          # 6 species, index 0 leads
    opp_team: list
    opponent: str          # "heuristic" | "random"


def make_schedule(n_flies: int, n_battles: int, opponent: str = "heuristic", seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    species = [m.species for m in POOL]
    out = []
    for fly in range(n_flies):
        for b in range(n_battles):
            mine = [species[i] for i in rng.choice(len(species), 6, replace=False)]
            opp = [species[i] for i in rng.choice(len(species), 6, replace=False)]
            out.append(ScheduledBattle(f"f{fly:02d}-b{b:03d}", fly, mine, opp, opponent))
    return out


def save(path: str | Path, sched: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(s) for s in sched], indent=1))


def load(path: str | Path) -> list:
    return [ScheduledBattle(**d) for d in json.loads(Path(path).read_text())]
