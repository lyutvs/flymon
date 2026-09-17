# M0d candidate changes — where they land in the engine (code survey, 2026-09-17)

Counts (MaleCNS npz, synapse count >= 5): KC 4064, APL 2, MBON 97, ALPN 682, sensory 13840 (53 receptor types).
ORN->PN 16,234 edges / 469,621 syn; ORN->all 66,725 edges. PN->KC 19,980 / 385,887. KC->APL 4,115 / 209,279.
APL->KC 4,104 / 195,278 (94% of APL output weight); APL->MBON 2.1% (MBON05 1162 syn, MBON13 22).

| Candidate | Engine site | New parameters | Default-off path | Invalidates | Throughput |
|---|---|---|---|---|---|
| Graded APL (global) | `Engine.step` (APL excluded from spike/reset; release r = g*max(0,v) through APL out-edges, same delay) — prototype `m2-calibration-g/graded_apl.py` | gain g (and optional saturation v_half) | `apl_mode="spiking"` -> current step, bit-identical | every M0c/M2 number under the new mode | smoke: no measurable slowdown |
| APL output split KC / non-KC | `build_csc` (APL out-edges scaled by `apl_scale`) | `apl_scale_non_kc` (or cap on non-KC release) | equal to `apl_scale` -> bit-identical CSC | MBON baselines, readout counts | none |
| Local / self APL inhibition | `Engine.step` graded path: per-KC inhibition = global release + beta * own activity trace (stand-in for spatially local release; no synapse coordinates in the npz) | beta (self term), trace tau | beta = 0 -> global graded | as above | small (vector op over 4064 KCs) |
| ORN->PN short-term depression | `Engine.propagate` / `step`: per-ORN resource r_j multiplies its out-edge weights; r_j recovers with tau_rec, drops by U on spike | U, tau_rec (literature) | `orn_std=False` | all odour-driven numbers | per-step vector over 13,840 receptors + repeat over spiking ORN edges |
| ORN->PN divisive normalisation (rate transform stand-in) | `stimuli.present`: transform per-glomerulus rate before Poisson (Olsen 2010 form) | Rmax, gamma, sigma, m (total-drive coefficient) | identity transform | all odour-driven numbers, channel equalisation of spec 3.3 | none |
| KC threshold rule | `Engine.__init__` (threshold = v_thresh * kc_thresh * clip(pn_in/median, 0.5, 3.0)) | rule choice / clip bounds | current rule | sparsity gate, all KC codes | none |
| MBON operating point | `Engine.__init__` `ext0[mbon] = mbon_hold_frac * v_thresh` | per-mode `mbon_hold_frac` (single) — per-type holds would be new | 0.85 | MBON baseline gate (3–4 Hz), readout z constants | none |

Tests pinning the current engine: `tests/brain/test_conditioning*.py`, `test_connectome.py`, `test_measure.py`, `test_pool_bench.py` use `results/m0` / `results/m0c` references bit-exactly — every new mode must default off so they stay green.
