"""The configuration object of spec appendix J's ORN->PN depression scan and judgement (J.11, with J.10.4's
re-convergence and J.10.8's records).

H.3a.11's rule carries over: every number the J runner, its rules and its records use is a field of `SPEC` (or of a
`dataclasses.replace` of it, for smoke runs and tests), and no other module restates one. H.4's readout reselection,
oracle, pair list and formula come in unchanged through `h4` (and H.3's measurement table through `h4.h3`).
"""
from __future__ import annotations

from dataclasses import dataclass

from .h4_spec import SPEC as H4_SPEC, H4Spec


@dataclass(frozen=True)
class JSpec:
    h4: H4Spec = H4_SPEC
    # ---- J.11.3: stage 1, the depression scan (calibration: no pairs, readouts or learning) -----------------------
    std_f: tuple = (0.78, 0.90, 0.95)             # resource kept per receptor spike
    std_tau_ms: tuple = (893.0, 300.0, 100.0)     # recovery time constant
    lit_f: float = 0.78                           # the literature constants (Nagel et al. 2015; H.3a.1)
    lit_tau_ms: float = 893.0
    scale_bracket: tuple = (1.0, 64.0)            # receptor_scale searched on log s in [0, log 64]
    scale_bisect_steps: int = 6                   # h3_rules.bisect_log: both ends, then 6 midpoints
    alpn_match_tol: float = 0.05                  # median ALPN within +-5% of the no-depression C3 median
    kc_band_pct: tuple = (3.0, 9.0)               # reference-set KC active median before re-convergence (inclusive)
    tie_tol: float = 0.01                         # relative kc_on log10-variance reduction: ties go to the literature
    count_floor: float = 0.5                      # log10(max(seed-mean count, 0.5)) in every variance
    uni_frac: float = 0.8                         # uniglomerular PN: >= 80% of its ORN input from one glomerulus
    uni_min_syn: int = 20                         #   and >= 20 ORN synapses (m0d-diag/disparity_stages.py)
    # ---- J.11.4 / J.10.4: stage 2, the judgement ----------------------------------------------------------------
    first_kc: float = 1.65                        # C3's adopted kc_thresh; the rest of H.3's grid follows in order
    homeo_target: float = 0.060                   # C3's adopted A0 (the declared 0.062 was never used)
    max_settings: int = 3                         # INVALID_ENGINE -> the next setting in the scan's order, at most 3
    # ---- J.11.4-5: D.6 on the judged engine ------------------------------------------------------------------------
    d6_seed_block: int = 8                        # one cache entry per 8 seeds of d6a.SEEDS (400-463)
    d6b_ratio_limit: float = 2.0                  # E.2: within-turn candidate KC spike ratio > 2
    # ---- where the C3 reference comes from --------------------------------------------------------------------------
    h3_block: str = "h3"
    c3_name: str = "C3"


SPEC = JSpec()
