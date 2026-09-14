"""Named populations and the dopamine compartment table, derived from annotations and raw edges."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .connectome import Connectome

SENSORY_CLASSES = ("olfactory", "gustatory", "mechanosensory", "mechanosensory_tactile",
                   "mechanosensory_proprioceptive", "thermosensory", "hygrosensory", "visual")


@dataclass
class Populations:
    receptor_types: dict
    receptor_side: dict
    sensory: np.ndarray
    alpn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    dan_types: dict
    pam: np.ndarray
    ppl1: np.ndarray
    apl: np.ndarray
    dn: np.ndarray
    mn: np.ndarray

    @classmethod
    def from_connectome(cls, c: Connectome) -> "Populations":
        t, k, s = c.type, c.cls, c.sc
        is_orn = np.char.startswith(t, "ORN_")
        receptor_types, receptor_side = {}, {}
        for name in sorted(set(t[is_orn])):
            idx = np.flatnonzero(t == name)
            receptor_types[name] = idx
            receptor_side[name] = {sd: idx[c.side[idx] == sd] for sd in ("L", "R", "M")}
        dan = np.flatnonzero(k == "DAN")
        dan_types = {}
        for name in sorted(set(t[dan])):
            if name.startswith("PAM") or name.startswith("PPL1"):
                dan_types[name] = dan[t[dan] == name]
        return cls(
            receptor_types=receptor_types, receptor_side=receptor_side,
            sensory=np.flatnonzero(np.isin(k, SENSORY_CLASSES)),
            alpn=np.flatnonzero(k == "ALPN"), kc=np.flatnonzero(k == "Kenyon_Cell"),
            mbon=np.flatnonzero(k == "MBON"), dan_types=dan_types,
            pam=dan[np.char.startswith(t[dan], "PAM")], ppl1=dan[np.char.startswith(t[dan], "PPL1")],
            apl=np.flatnonzero(t == "APL"),
            dn=np.flatnonzero(s == "descending_neuron"), mn=np.flatnonzero(s == "vnc_motor"),
        )


@dataclass
class Compartment:
    family: str
    cells: np.ndarray
    w_mbon: np.ndarray   # [N], DAN->MBON synapse mass normalised to the type's peak, 0 elsewhere
    core: np.ndarray     # MBON indices with w_mbon >= core_frac


def compartments(c: Connectome, p: Populations, core_frac: float) -> dict:
    """One compartment per DAN type from the RAW DAN->MBON edge list (dopamine modulates, it does
    not transmit, so these edges carry no current in the CSC and must be read from raw arrays)."""
    is_mbon = np.zeros(c.N, bool)
    is_mbon[p.mbon] = True
    is_dan_any = np.zeros(c.N, bool)
    for cells in p.dan_types.values():
        is_dan_any[cells] = True
    # pre-mask once: only DAN->MBON edges can contribute, and they are a tiny slice of the edge list
    keep = np.flatnonzero(is_dan_any[c.pre] & is_mbon[c.post])
    k_pre, k_post, k_w = c.pre[keep], c.post[keep], c.w[keep].astype(np.float32)
    out = {}
    for name, cells in p.dan_types.items():
        m = np.isin(k_pre, cells)
        acc = np.zeros(c.N, np.float32)
        np.add.at(acc, k_post[m], k_w[m])
        peak = float(acc.max())
        if peak <= 0:
            continue
        w = acc / peak
        core = np.flatnonzero(w >= core_frac)
        out[name] = Compartment(family="PAM" if name.startswith("PAM") else "PPL1", cells=cells, w_mbon=w, core=core)
    return out


def validate_populations(conn: Connectome, pops: Populations, comps: dict,
                         punish_type: str = "PPL105", reward_type: str = "PAM08") -> None:
    """Fail fast on a connectome that cannot carry the M0 protocol (called before long runs).

    The two dopamine channels are the ones the conditioning run will use; the defaults are the
    flybrain pair (PPL105 punishment / PAM08 reward)."""
    for name, arr in (("Kenyon cells", pops.kc), ("MBONs", pops.mbon)):
        if len(arr) == 0:
            raise ValueError(f"no {name} in the connectome: check the class annotations")
    if not pops.receptor_types:
        raise ValueError("no olfactory receptor types (ORN_*) in the connectome")
    missing = [t for t in (punish_type, reward_type) if t not in comps]
    if missing:
        raise ValueError(f"missing dopamine compartments {missing}: the M0 gate needs both "
                         f"{punish_type} and {reward_type}")
    overlap = set(comps[punish_type].core.tolist()) & set(comps[reward_type].core.tolist())
    if overlap:
        raise ValueError(f"{punish_type} and {reward_type} core compartments overlap on {len(overlap)} MBONs: "
                         "the readout would not separate approach from avoidance")


def export_compartments(comps: dict, c: Connectome, path: str | Path) -> None:
    d = {name: {"family": cp.family, "n_cells": int(len(cp.cells)),
                "core_mbon_types": sorted(set(c.type[cp.core].tolist()))}
         for name, cp in comps.items()}
    Path(path).write_text(json.dumps(d, indent=2))
