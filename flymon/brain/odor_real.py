"""Spec N.2 as amended by N.8.2: Hallem & Carlson 2006 responses (data/odor/, from DoOR.data) -> the model's ORN drive.

Receptor -> glomerulus from the mapping file. A receptor listed on several glomeruli ("DM5+DM3") is duplicated to each
(co-expression: every listed ORN class carries it); receptors on one glomerulus add. Mixtures are linear in Δ
(4:1 = 0.8 IA + 0.2 EB); δ-DL is multiplied by c_δ. drive_hz = g x max(ΣΔ, 0) per glomerulus: inhibition is clipped and
the clipped amount recorded; no spontaneous rate. present() gets s = drive_hz / (max_rate_hz x strength), for every
model ORN type (0 where nothing maps). Values are read only from the data files; a file whose sha256 differs from its
pin, a missing or non-numeric value, a receptor without a mapping, an unmapped glomerulus or one the model lacks raises
DataMismatch (STOP_DATA_MISMATCH, the only data stop of N0). Channels commanded above the receptors' refractory cap are
listed (a record)."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .h3_store import sha256_file

ODORANTS = ("IA", "EB", "dDL")
TABLE_FILE = "hallem2006_subset.csv"
MAP_FILE = "door_mappings_subset.csv"


class DataMismatch(ValueError):
    """STOP_DATA_MISMATCH (N.8.2)."""


@dataclass(frozen=True)
class Table:
    delta: dict          # receptor -> {"IA", "EB", "dDL": Δ spikes/s}
    glomeruli: dict      # receptor -> tuple of glomerulus names (no "ORN_" prefix)


def load_table(data_dir, sha256: dict | None) -> Table:
    d = Path(data_dir)
    for name in (TABLE_FILE, MAP_FILE):
        if not (d / name).exists():
            raise DataMismatch(f"{d / name} is missing")
        if sha256 is not None and sha256_file(d / name) != sha256.get(name):
            raise DataMismatch(f"{d / name}: sha256 differs from the pin")
    delta: dict = {}
    with open(d / TABLE_FILE, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["receptor", *ODORANTS]:
            raise DataMismatch(f"{TABLE_FILE}: header {r.fieldnames}")
        for row in r:
            rec = row["receptor"]
            if rec in delta:
                raise DataMismatch(f"receptor {rec} listed twice")
            try:
                vals = {k: float(row[k]) for k in ODORANTS}
            except (TypeError, ValueError):
                raise DataMismatch(f"receptor {rec}: a missing or non-numeric value") from None
            if not all(math.isfinite(v) for v in vals.values()):
                raise DataMismatch(f"receptor {rec}: a non-finite value")
            delta[rec] = vals
    glom: dict = {}
    with open(d / MAP_FILE, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["receptor", "glomerulus"]:
            raise DataMismatch(f"{MAP_FILE}: header {r.fieldnames}")
        for row in r:
            parts = tuple(p.strip() for p in (row["glomerulus"] or "").split("+"))
            if any(p in ("", "?") for p in parts):
                raise DataMismatch(f"receptor {row['receptor']} is unmapped ({row['glomerulus']!r})")
            glom[row["receptor"]] = parts
    if set(glom) != set(delta):
        raise DataMismatch(f"receptors without a mapping {sorted(set(delta) - set(glom))}, "
                           f"mappings without values {sorted(set(glom) - set(delta))}")
    return Table(delta=delta, glomeruli=glom)


def glomerular(table: Table, model_types) -> dict:
    """{"ORN_<glom>": {"IA", "EB", "dDL": ΣΔ}} over every model ORN type, in the given order (0 where nothing maps)."""
    out = {str(t): {k: 0.0 for k in ODORANTS} for t in model_types}
    for rec in sorted(table.delta):
        for gl in table.glomeruli[rec]:
            t = "ORN_" + gl
            if t not in out:
                raise DataMismatch(f"receptor {rec} maps to {gl}, which the model has no ORN type for")
            for k in ODORANTS:
                out[t][k] += table.delta[rec][k]
    return out


def receptor_totals(table: Table) -> dict:
    """Sums over the 24 receptors before any glomerular assignment: Σ Δ and Σ max(Δ, 0)."""
    return {"signed": {k: float(sum(v[k] for v in table.delta.values())) for k in ODORANTS},
            "positive": {k: float(sum(max(v[k], 0.0) for v in table.delta.values())) for k in ODORANTS}}


def lin_record(table: Table, lin_totals: dict, tol: float) -> dict:
    tot = receptor_totals(table)
    rel = {c: {k: (tot[c][k] - lin_totals[k]) / lin_totals[k] for k in ODORANTS} for c in tot}
    return dict(totals=tot, lin=dict(lin_totals), rel_error=rel, tol=tol,
                within_tol={c: all(abs(v) <= tol for v in rel[c].values()) for c in rel}, note="record only (N.8.2)")


def mix(glom: dict, weights: dict) -> dict:
    return {t: float(sum(weights.get(k, 0.0) * v[k] for k in ODORANTS)) for t, v in glom.items()}


def drive(delta: dict, g: float) -> tuple:
    hz = {t: float(g) * max(v, 0.0) for t, v in delta.items()}
    clipped = {t: float(g) * -v for t, v in delta.items() if v < 0}
    return hz, clipped


def strengths(drive_hz: dict, max_rate_hz: float, strength: float) -> dict:
    return {t: v / (max_rate_hz * strength) for t, v in drive_hz.items()}


def cap_hz(params) -> float:
    """A receptor fires at most once per refrac_steps() + 1 steps (engine_cpu: receptors honour `free`)."""
    return 1000.0 / (params.dt * (params.refrac_steps() + 1))


def stimuli(glom: dict, names, g: float, c_delta: float, mixtures: dict, max_rate_hz: float, strength: float,
            cap: float) -> dict:
    out = {}
    for n in names:
        w = {k: v * (c_delta if k == "dDL" else 1.0) for k, v in mixtures[n].items()}
        hz, clipped = drive(mix(glom, w), g)
        out[n] = dict(odor=strengths(hz, max_rate_hz, strength), drive_hz=hz, clipped_hz=clipped,
                      clipped_total_hz=float(sum(clipped.values())), weights=w, g=float(g), c_delta=float(c_delta),
                      capped=sorted(t for t, v in hz.items() if v > cap))
    return out
