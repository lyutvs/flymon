"""AC.7 2b's smoke gates (any failure -> STOP_SMOKE): the full test suite; the four safety-constraint tests (spec
198); worker == in-process on L_V at the value level (KC counts, A, P, V, choice, the weight change after one pulse)
and the workers' L_V sha; the log schema of every brain-arm record; per-arm micro-smokes (C-off weights unchanged,
FLY-TB's odour gate tests); the smoke run completing across a stop and a resume, the no-brain arms completing, one
evaluation-schedule digest; and the POOL odour response of the naive L_V brain on the confirmation-set candidate
situations (AC plan reading 4: the 79 situations of the 102 candidate pairs, before the seed-301 draw): candidate-V
tie share <= 0.5, A (MBON13) = 0 share <= 0.5, P (MBON05) = 0 share <= 0.5. The share of fly turns whose candidate
KC counts differ by more than 2x is recorded, not gated (AC.0 6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..agent import logschema, policy
from .spec import SPEC

SMOKE = "results/summary/ac_smoke.json"
SAFETY_TESTS = ("tests/agent/test_swarm.py::test_nonplastic_weights_bit_identical",
                "tests/agent/test_policy.py::test_rule_is_stateless",
                "tests/battle/test_agent_player.py::test_coach_turn_never_reinforces",
                "tests/agent/test_swarm.py::test_median_floor_stops_the_run")
EQUALITY_TESTS = ("tests/brain/test_lv_pool.py::test_lever_worker_equals_in_process_values",
                  "tests/brain/test_lv_pool.py::test_lever_sha_matches_v_spec")
TB_TESTS = ("tests/ac/test_ac_tb.py",)


def pool_response(counts, na: int, npp: int, z: dict) -> dict:
    ties = a0 = p0 = n = 0
    for c in counts:
        c = np.asarray(c)
        a, p = c[:, :na].sum(1), c[:, na:na + npp].sum(1)
        v = policy.values(a, p, z)
        ties += int(np.sum(v == v.max()) > 1)
        a0 += int(np.sum(a == 0))
        p0 += int(np.sum(p == 0))
        n += len(a)
    return dict(n_situations=len(counts), n_candidates=n, tie_ratio=ties / len(counts), a0_ratio=a0 / n,
                p0_ratio=p0 / n)


def response_gates(resp: dict, spec=SPEC) -> dict:
    return dict(pool_tie=resp["tie_ratio"] <= spec.smoke_max_tie, pool_a0=resp["a0_ratio"] <= spec.smoke_max_a0,
                pool_p0=resp["p0_ratio"] <= spec.smoke_max_p0)


def log_paths(arm_dir) -> list:
    d = Path(arm_dir)
    return sorted((d / "logs").glob("fly*.jsonl")) + sorted((d / "logs" / "eval").glob("fly*.jsonl"))


def _records(paths):
    for p in paths:
        for i, line in enumerate(Path(p).read_text().splitlines(), 1):
            yield p, i, json.loads(line)


def schema_errors(paths) -> list:
    out = []
    for p, i, rec in _records(paths):
        try:
            logschema.validate(rec)
        except ValueError as e:
            out.append(f"{p}:{i}: {e}")
    return out


def kc_ratio_frac(paths) -> dict:
    flags = [bool(r["kc_ratio_gt2"]) for _, _, r in _records(paths)
             if r.get("kind") == "decision" and r.get("decider") == "fly" and "kc_ratio_gt2" in r]
    return dict(n_fly_turns=len(flags), n_gt2=sum(flags), frac=(sum(flags) / len(flags)) if flags else None)


def resumed(arm_dir) -> bool:
    s = json.loads((Path(arm_dir) / "wall_clock.json").read_text())["sessions"]
    return len(s) >= 2 and s[0].get("complete") is False and s[-1].get("complete") is True


def status(gates: dict) -> str:
    return "OK" if all(gates.values()) else "STOP_SMOKE"
