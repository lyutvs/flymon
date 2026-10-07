"""The model manifest (spec AC.1, the 4.4 freeze, part (i)): L_V's CSC sha, the codebook digest, z_V, the strength, the
recovery r chosen by tau_rec on L_V (AC.7 2a) and the seeds - committed after 2a and verified by every M4 run and
resume. STOP_NO_RECOVERY (no r passes) is recorded, not frozen, and stops AC (AC.2)."""
from __future__ import annotations

from ..agent.config import config_hash
from .spec import LABEL, SPEC

MODEL_MANIFEST = "results/summary/ac_model_manifest.json"


def build_model_manifest(taurec: dict, taurec_info: dict, cfg, codebook_digest: str, encoder_grid_sha256: str,
                         spec=SPEC) -> dict:
    if taurec.get("status") != "SELECTED":
        return dict(status="STOP_NO_RECOVERY", taurec=taurec_info, rule=taurec.get("rule"),
                    failed=taurec.get("failed"), label=LABEL)
    if taurec.get("lever_edit") != spec.lever_edit or taurec.get("lever_sha") != spec.lever_sha:
        raise ValueError(f"tau_rec ran on lever {taurec.get('lever_edit')!r} / {str(taurec.get('lever_sha'))[:12]}, "
                         f"not AC's {spec.lever_edit!r} / {spec.lever_sha[:12]}")
    if float(taurec["recovery_per_pulse"]) != float(cfg.params.recovery_per_pulse):
        raise ValueError(f"cfg recovery {cfg.params.recovery_per_pulse} != tau_rec's {taurec['recovery_per_pulse']}")
    return dict(status="FROZEN", lever_edit=spec.lever_edit, lever_sha=taurec["lever_sha"], p_type=spec.p_type,
                readout=dict(cfg.readout), z_V={k: [float(x) for x in v] for k, v in cfg.z.items()},
                codebook_config=spec.codebook_config, dual_rule=spec.dual_rule, codebook_digest=codebook_digest,
                encoder_grid_sha256=encoder_grid_sha256, strength=float(cfg.strength), settle_ms=float(cfg.settle_ms),
                read_ms=float(cfg.read_ms), reward_dan=cfg.reward_type, punish_dan=cfg.punish_type,
                recovery_per_pulse=float(cfg.params.recovery_per_pulse), taurec=taurec_info,
                taurec_rule=taurec.get("rule"), c3_threshold_sha256=cfg.params.kc_thresh_sha256, seeds=spec.seeds(),
                tau={"start": float(cfg.tau_start), "end": float(cfg.tau_end), "battles": int(cfg.tau_battles)},
                config_hash=config_hash(cfg), label=LABEL)


def check_model(doc: dict, *, cfg, codebook_digest: str, spec=SPEC) -> list:
    """Names of the manifest fields this run's configuration does not match ([] = it matches)."""
    if doc.get("status") != "FROZEN":
        return ["status"]
    want = dict(lever_edit=spec.lever_edit, lever_sha=spec.lever_sha, codebook_digest=codebook_digest,
                z_V={k: [float(x) for x in v] for k, v in cfg.z.items()}, strength=float(cfg.strength),
                recovery_per_pulse=float(cfg.params.recovery_per_pulse), seeds=spec.seeds(),
                config_hash=config_hash(cfg))
    return [k for k, v in want.items() if doc.get(k) != v]
