"""What both of spec appendix J's CLIs build before measuring: the connectome's populations and core pools, H.3's
reference set and all51 stimuli, E.1's candidate odours (D.6), the pair list, and the contexts of C3's rule and of J."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from . import d6a
from .circuits import Populations, compartments
from .config import Params
from .connectome import Connectome
from .h3_c3 import ThresholdFiles, rule_thresholds, update_mask
from .h3_runner import Context as H3Context
from .h3_spec import all51_glomeruli, make_odors
from .h4_pairs import even_pairs, pair_key, pairs_digest
from .j_runner import JContext


def pool_types(conn, comps, dan_type) -> list:
    t = np.asarray(conn.type).astype(str)
    return sorted({str(t[int(i)]) for i in comps[dan_type].core})


def build(npz, spec, out, deadline=None, log=print, pairs=None, pools=None) -> dict:
    """pairs / pools replace the E0 pair list and the core-type pools in the tests only (the synthetic connectome has
    neither). Returns conn, pops, pools, odors, all51 items, pairs (+ digest) and the J context."""
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    h3 = spec.h4.h3
    core = dict(A=pool_types(conn, comps, h3.punish_type), P=pool_types(conn, comps, h3.reward_type))
    pools = pools or core
    odors = make_odors(pops, h3.reference)
    all51 = all51_glomeruli(pops, h3.reference.exclude)
    all_pairs = even_pairs(pops) if pairs is None else pairs
    rules = {}

    def rule_for(kc):
        if kc not in rules:
            rules[kc] = rule_thresholds(conn, pops, dataclasses.replace(Params(), kc_thresh=float(kc)))
        return rules[kc]
    h3ctx = H3Context(spec=dataclasses.replace(h3, homeo_target=spec.homeo_target), odors=odors, pools=pools,
                      n_kc=len(pops.kc), deadline=deadline, log=log,
                      extra=dict(update_mask=update_mask(conn, pops, Params()), rule_thresholds=rule_for,
                                 threshold_files=ThresholdFiles(Path(out) / "thresholds", conn.bodyId[pops.kc])))
    ctx = JContext(spec=spec, h3=h3ctx, pools=pools, probe_seeds=[int(s) for o in odors for s in o["seeds"]],
                   expected=[pair_key(p) for p in all_pairs], odours=d6a.candidate_odours(pops), log=log)
    return dict(conn=conn, pops=pops, core=core, pools=pools, odors=odors, all51=all51, pairs=all_pairs,
                pairs_digest=pairs_digest(all_pairs), ctx=ctx)
