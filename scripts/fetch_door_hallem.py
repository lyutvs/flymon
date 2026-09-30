#!/usr/bin/env python3
"""Spec N.8.2: fetch Hallem & Carlson 2006 (DoOR.data column `Hallem.2006.EN`) for isopentyl acetate (IA), ethyl
butyrate (EB) and δ-decalactone (δ-DL) on the receptors that carry that column, with their `door_mappings.csv`
glomerulus, from one pinned DoOR.data commit, and write data/odor/.

    uv run python scripts/fetch_door_hallem.py          # network; then: git add -f data/odor

Odorants are matched by CAS, never by name: DoOR calls IA "isopentyl acetate", and δ-decalactone (705-86-2) sits next
to γ-decalactone (706-14-9) in every receptor file. Exit 5 (STOP_DATA_MISMATCH, nothing written) on a missing or NA
value, a receptor count other than 24, a receptor without exactly one mapping row, an unmapped glomerulus ("?" or
empty) or a glomerulus the model has no ORN_* type for. Exit 2 on a network error. data/ is git-ignored: the files are
added with `git add -f` (.gitignore is not edited, N.8.7). Data rows of a DoOR CSV carry a leading row id, so header
column j is field j + 1."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "ropensci/DoOR.data"
SHA = "db323a496577c4b4a72b5c2fcd1859e07521ffb5"           # master as of 2026-09-30 (commit of 2026-07-17)
RAW = f"https://raw.githubusercontent.com/{REPO}/{SHA}/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"
COLUMN = "Hallem.2006.EN"
ODORANTS = (("IA", "123-92-2"), ("EB", "105-54-4"), ("dDL", "705-86-2"))
N_RECEPTORS = 24
OUTPUT_SHA256 = {                                           # the pinned bytes of the two tables (N.8.2); n_spec reads these
    "hallem2006_subset.csv": "d65f2711c73cc7d6e7f547e4e65c0eef5bfaac56b7f08ff4410d69e50c471445",
    "door_mappings_subset.csv": "cacf48b235936086f269ddf2dfc1c4e62f3fa2547a410ad4ccc7e98bc2c60005",
}
NOT_RECEPTOR = ("door_", "odor", "ORs")                     # data/*.csv tables that are not one receptor's file
NOTICE = f"""data/odor/ is derived from DoOR.data (https://github.com/{REPO}), commit {SHA},
licensed CC BY-SA 4.0 (its DESCRIPTION: "License: CC BY-SA 4.0").
Changes: only the {COLUMN} column for isopentyl acetate (CAS 123-92-2), ethyl butyrate (CAS 105-54-4) and
delta-decalactone (CAS 705-86-2) on the 24 receptors that carry it, and those receptors' glomerulus column of
door_mappings.csv, reformatted as comma-separated files. These files are distributed under CC BY-SA 4.0
(LICENSE-CC-BY-SA-4.0.txt). provenance.json records every source file's sha256.
Sources: Hallem & Carlson 2006, Cell 125:143-160; Muench & Galizia 2016, Sci Rep 6:21841 (DoOR 2.0).
"""


class Mismatch(Exception):
    """STOP_DATA_MISMATCH (N.8.2)."""


def parse(text: str) -> tuple:
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    return rows[0], rows[1:]


def receptor_values(files: dict) -> dict:
    """{receptor: {"IA", "EB", "dDL": value text}} for every file whose Hallem.2006.EN column has a non-NA value."""
    out = {}
    for name, text in sorted(files.items()):
        header, rows = parse(text)
        if COLUMN not in header or "CAS" not in header:
            continue
        j, cas_j = header.index(COLUMN) + 1, header.index("CAS") + 1
        if all(len(r) <= j or r[j] in ("NA", "") for r in rows):
            continue
        vals = {}
        for key, cas in ODORANTS:
            hit = [r for r in rows if len(r) > j and r[cas_j] == cas]
            if len(hit) != 1 or hit[0][j] in ("NA", ""):
                raise Mismatch(f"{name}: {key} (CAS {cas}) has no {COLUMN} value")
            float(hit[0][j])                                    # a non-numeric value raises ValueError: not data
            vals[key] = hit[0][j]
        out[name] = vals
    return out


def receptor_glomeruli(mapping_text: str, receptors) -> dict:
    header, rows = parse(mapping_text)
    ri, gi = header.index("receptor") + 1, header.index("glomerulus") + 1
    by: dict = {}
    for r in rows:
        by.setdefault(r[ri], []).append(r[gi])
    out = {}
    for rec in receptors:
        g = by.get(rec) or []
        if len(g) != 1:
            raise Mismatch(f"receptor {rec} has {len(g)} mapping rows (need exactly 1)")
        if any(p.strip() in ("", "?") for p in g[0].split("+")):
            raise Mismatch(f"receptor {rec} is unmapped ({g[0]!r})")
        out[rec] = g[0]
    return out


def check_model(glomeruli: dict, model_types) -> None:
    have = {str(t) for t in model_types}
    miss = sorted({p for v in glomeruli.values() for p in v.split("+") if "ORN_" + p not in have})
    if miss:
        raise Mismatch(f"glomeruli {miss} have no ORN_* type in the model")


def render(values: dict, glomeruli: dict) -> tuple:
    recs = sorted(values)
    table = "receptor,IA,EB,dDL\n" + "".join(
        f"{r},{values[r]['IA']},{values[r]['EB']},{values[r]['dDL']}\n" for r in recs)
    mapping = "receptor,glomerulus\n" + "".join(f"{r},{glomeruli[r]}\n" for r in recs)
    return table, mapping


def extract(files: dict, mapping_text: str, model_types, n_expected: int = N_RECEPTORS) -> tuple:
    values = receptor_values(files)
    if len(values) != n_expected:
        raise Mismatch(f"{len(values)} receptors carry {COLUMN}, expected {n_expected}")
    glomeruli = receptor_glomeruli(mapping_text, values)
    check_model(glomeruli, model_types)
    return render(values, glomeruli)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flymon-fetch-door/1.0"})   # creativecommons.org 403s urllib's default
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _odorant_ids(text: str) -> dict:
    header, rows = parse(text)
    cas_j, name_j, ink_j = (header.index(k) + 1 for k in ("CAS", "Name", "InChIKey"))
    out = {}
    for key, cas in ODORANTS:
        r = next(r for r in rows if r[cas_j] == cas)
        out[key] = dict(cas=cas, door_name=r[name_j], inchikey=r[ink_j])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default="data/odor")
    a = ap.parse_args(argv)
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    types = sorted(Populations.from_connectome(Connectome.load(a.npz)).receptor_types)
    try:
        tree = json.loads(_get(TREE))["tree"]
        paths = sorted(t["path"] for t in tree if t["path"].startswith("data/") and t["path"].endswith(".csv")
                       and not Path(t["path"]).stem.startswith(NOT_RECEPTOR))
        raw = {p: _get(RAW + p) for p in paths + ["data/door_mappings.csv"]}
        license_text = _get(LICENSE_URL)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"refused: network error {e!r}", file=sys.stderr)
        return 2
    files = {Path(p).stem: raw[p].decode() for p in paths}
    try:
        table, mapping = extract(files, raw["data/door_mappings.csv"].decode(), types)
    except (Mismatch, ValueError) as e:
        print(f"STOP_DATA_MISMATCH: {e}", file=sys.stderr)
        return 5
    got = {"hallem2006_subset.csv": table, "door_mappings_subset.csv": mapping}
    bad = sorted(n for n, t in got.items() if hashlib.sha256(t.encode()).hexdigest() != OUTPUT_SHA256[n])
    if bad:                                                     # the pinned commit no longer yields the pinned bytes
        print(f"STOP_DATA_MISMATCH: {bad} differ from OUTPUT_SHA256", file=sys.stderr)
        return 5
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hallem2006_subset.csv").write_text(table)
    (out / "door_mappings_subset.csv").write_text(mapping)
    (out / "LICENSE-CC-BY-SA-4.0.txt").write_bytes(license_text)
    (out / "NOTICE").write_text(NOTICE)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    prov = dict(repo=REPO, commit=SHA, column=COLUMN, retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                odorants=_odorant_ids(files["Or22a"]), sources={p: sha(b) for p, b in sorted(raw.items())},
                license=dict(name="CC BY-SA 4.0", declared_in="DESCRIPTION", text_url=LICENSE_URL,
                             text_sha256=sha(license_text)),
                outputs={n: sha((out / n).read_bytes()) for n in ("hallem2006_subset.csv", "door_mappings_subset.csv",
                                                                  "LICENSE-CC-BY-SA-4.0.txt", "NOTICE")},
                citations=["Hallem EA, Carlson JR (2006) Cell 125:143-160",
                           "Muench D, Galizia CG (2016) Sci Rep 6:21841"])
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: {table.count(chr(10)) - 1} receptors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
