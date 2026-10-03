"""S's gate-② numbers (S.3 ④, S.9.7) that p_rules does not return — records and the ratio the gate reads; no engine.
- vectors: p_rules.group + p_rules.arm_vectors per direction (P.6.2's per-seed dl, ds, dt over arms ① and ③).
- ratio: per direction ℓ_L / ℓ_C on the same seeds (ℓ = mean dl = p_rules' ℓ), with a paired seed bootstrap (one
  o_rules.boot_weights matrix for both conditions, P's draws and seed; the CI at P's level) — the gate reads the
  point estimate (S.3 ④), the CI is a record.
- gate2_oc (S.9.7, recorded before gate ②, never changes it): from P's committed block (no lever, seeds 23_000_xxx),
  per direction ℓ_P and its bootstrap SE; for a true ratio ρ, ℓ_C ≈ ℓ_P and ℓ_L = ρ ℓ_P with both SEs equal to P's
  and independent, so SE(ρ̂) ≈ SE_P √(1 + ρ²) / ℓ_P; P(STOP_PUNISH_WEAKENED) = 1 − Π_d Φ((ρ − p_ratio_min) / SE_d)
  (normal approximation, directions independent; pairing by seed would shrink the SE — the approximation is
  conservative toward STOP)."""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

from .n_rules import _ci
from .o_rules import boot_weights
from .p_rules import arm_vectors, group


def vectors(rows: list, z: dict, pspec) -> dict:
    by = group(rows, pspec)
    return {d: arm_vectors(by[d], z, pspec) for d in pspec.directions}


def ratio(rows_l: list, rows_c: list, z: dict, spec) -> dict:
    """{direction: ell_L, ell_C, ratio, ratio_ci, n}; ValueError when L and C do not hold the same seeds."""
    vl, vc = vectors(rows_l, z, spec.p), vectors(rows_c, z, spec.p_c)
    o = spec.p.o
    out = {}
    for d in spec.p.directions:
        if vl[d]["seeds"] != vc[d]["seeds"]:
            raise ValueError(f"direction {d}: L and C must hold the same seeds in the same order")
        dl_l, dl_c = np.asarray(vl[d]["dl"], float), np.asarray(vc[d]["dl"], float)
        W = boot_weights(dl_l.size, o.boot_draws, o.boot_seed)
        bl, bc = W @ dl_l, W @ dl_c
        out[d] = dict(ell_L=float(dl_l.mean()), ell_C=float(dl_c.mean()), ratio=float(dl_l.mean() / dl_c.mean()),
                      ratio_ci=_ci(bl / bc, o.ci_level), n=int(dl_l.size), seeds=list(vl[d]["seeds"]))
    return out


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def gate2_oc(p_rows: list, z: dict, pspec, spec) -> dict:
    """S.9.7: STOP_PUNISH_WEAKENED's probability at each true ratio in spec.gate2_oc_rhos from P's block (module
    docstring for the model)."""
    v = vectors(p_rows, z, pspec)
    o = pspec.o
    dirs = {}
    for d in pspec.directions:
        dl = np.asarray(v[d]["dl"], float)
        W = boot_weights(dl.size, o.boot_draws, o.boot_seed)
        dirs[d] = dict(ell=float(dl.mean()), se=float(np.std(W @ dl)), n=int(dl.size))
    rows = []
    for rho in spec.gate2_oc_rhos:
        se = {d: s["se"] * sqrt(1.0 + rho * rho) / s["ell"] for d, s in dirs.items()}
        keep = float(np.prod([_phi((rho - spec.p_ratio_min) / se[d]) for d in dirs]))
        rows.append(dict(rho=float(rho), se_ratio=se, p_stop_weakened=1.0 - keep))
    return dict(model="SE(ρ̂) ≈ SE_P √(1 + ρ²) / ℓ_P per direction (ℓ_C ≈ ℓ_P, ℓ_L = ρ ℓ_P, equal independent SEs); "
                      "P(STOP) = 1 − Π_d Φ((ρ − p_ratio_min) / SE_d); normal approximation, directions independent",
                source=dict(seeds=[int(s) for s in pspec.seeds], n_rows=len(p_rows)), directions=dirs, rows=rows)
