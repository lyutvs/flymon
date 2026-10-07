"""The L_V agent configuration (spec AC.2): C3's Params (block "h3", threshold sha checked by load_c3) with the
recovery r re-calibrated on L_V (AC.7 2a), readout A = MBON13 / P = MBON05, z_V (W.1), the E-grid k2-norm codebook
with the `norm` dual rule at strength 1.0, presentation 800 / 600 ms, M3's tau schedule (AgentConfig defaults)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..agent.config import AgentConfig
from ..agent.encode_grid import GridEncoder
from ..brain import q_pairs
from ..brain.j_store import load_c3
from ..brain.v_spec import SPEC as V_SPEC
from ..brain.w_spec import SPEC as W_SPEC
from .spec import SPEC
from .store import sha256_file

ENCODER_GRID = "results/summary/encoder_grid.json"
TAUREC_LV = "results/summary/rescope_taurec_lv.json"


def lv_codebook(path=ENCODER_GRID) -> tuple:
    """(Codebook, the block codebook digest) of V's config (k2-norm); q_pairs.codebook checks the digest and cells."""
    if V_SPEC.config != SPEC.codebook_config:
        raise ValueError(f"V's codebook config {V_SPEC.config!r} is not AC's {SPEC.codebook_config!r}")
    enc = json.loads(Path(path).read_text())
    return q_pairs.codebook(enc, V_SPEC), enc["codebook"]["configs"][SPEC.codebook_config]["digest"]


def grid_encoder(pops, path=ENCODER_GRID) -> GridEncoder:
    cb, _ = lv_codebook(path)
    return GridEncoder(pops, cb, SPEC.dual_rule)


def load_recovery(path=TAUREC_LV) -> tuple:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (AC.7 2a: run scripts/run_rescope_taurec.py --lv first)")
    d = json.loads(p.read_text())
    if d.get("status") != "SELECTED":
        raise SystemExit(f"refusing: {p} status {d.get('status')!r} is not SELECTED (STOP_NO_RECOVERY stops AC)")
    if d.get("lever_edit") != SPEC.lever_edit:
        raise SystemExit(f"refusing: {p} was calibrated on lever {d.get('lever_edit')!r}, not {SPEC.lever_edit!r}")
    return float(d["recovery_per_pulse"]), dict(path=str(p), sha256=sha256_file(p))


def load_lv_config(recovery: float, summary_path="results/summary/m0d.json", spec=SPEC) -> AgentConfig:
    params, _, _ = load_c3(json.loads(Path(summary_path).read_text()))
    return AgentConfig(params=dataclasses.replace(params, recovery_per_pulse=float(recovery)), z=W_SPEC.z_v(),
                       readout={"A": spec.a_type, "P": spec.p_type}, strength=spec.strength,
                       settle_ms=spec.settle_ms, read_ms=spec.read_ms, reward_type=spec.reward_dan,
                       punish_type=spec.punish_dan)
