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
    out = {}
    for name, cells in p.dan_types.items():
        m = np.isin(c.pre, cells) & is_mbon[c.post]
        acc = np.zeros(c.N, np.float32)
        np.add.at(acc, c.post[m], c.w[m].astype(np.float32))
        peak = float(acc.max())
        if peak <= 0:
            continue
        w = acc / peak
        core = np.flatnonzero(w >= core_frac)
        out[name] = Compartment(family="PAM" if name.startswith("PAM") else "PPL1", cells=cells, w_mbon=w, core=core)
    return out


def export_compartments(comps: dict, c: Connectome, path: str | Path) -> None:
    d = {name: {"family": cp.family, "n_cells": int(len(cp.cells)),
                "core_mbon_types": sorted(set(c.type[cp.core].tolist()))}
         for name, cp in comps.items()}
    Path(path).write_text(json.dumps(d, indent=2))
