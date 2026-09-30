# tests/test_fetch_door_hallem.py
"""Spec N.8.2: odorants by CAS (never by name; δ-decalactone is not γ), 24 receptors with a Hallem.2006.EN column, one
mapping row each, no "?" glomerulus, every glomerulus a model ORN type; the committed data files are the pinned bytes."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE = """receptor,IA,EB,dDL
Or10a,222,43,3
Or19a,110,102,6
Or22a,236,197,15
Or23a,30,24,7
Or2a,73,20,8
Or33b,31,62,40
Or35a,39,138,7
Or43a,38,43,5
Or43b,98,201,4
Or47a,123,75,1
Or47b,9,31,43
Or49b,19,5,4
Or59b,48,41,4
Or65a,28,10,13
Or67a,140,178,13
Or67c,54,118,6
Or7a,-16,18,2
Or82a,55,43,11
Or85a,-2,139,15
Or85b,244,115,37
Or85f,34,33,6
Or88a,25,17,25
Or98a,260,93,20
Or9a,142,124,1
"""
MAPPING = """receptor,glomerulus
Or10a,DL1
Or19a,DC1
Or22a,DM2
Or23a,DA3
Or2a,DA4m
Or33b,DM5+DM3
Or35a,VC3
Or43a,DA4l
Or43b,VM2
Or47a,DM3
Or47b,VA1v
Or49b,VA5
Or59b,DM4
Or65a,DL3
Or67a,DM6
Or67c,VC4
Or7a,DL5
Or82a,VA6
Or85a,DM5
Or85b,VM5d
Or85f,DL4
Or88a,VA1d
Or98a,VM5v
Or9a,VM3
"""
TABLE_SHA = "d65f2711c73cc7d6e7f547e4e65c0eef5bfaac56b7f08ff4410d69e50c471445"
MAPPING_SHA = "cacf48b235936086f269ddf2dfc1c4e62f3fa2547a410ad4ccc7e98bc2c60005"
HEAD = '"Class";"Name";"InChIKey";"CID";"CAS";"Hallem.2006.EN";"Other.2001"\n'


def _script():
    sp = importlib.util.spec_from_file_location("fetch_door_hallem", ROOT / "scripts" / "fetch_door_hallem.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules["fetch_door_hallem"] = mod
    sp.loader.exec_module(mod)
    return mod


def _receptor(ia="236", eb="197", ddl="15", gamma_only=False):
    rows = [f'"1";NA;"sfr";"SFR";"SFR";"SFR";4;NA',
            f'"2";"ester";"isopentyl acetate";"MLF";"31276";"123-92-2";{ia};NA',
            f'"3";"ester";"ethyl butyrate";"OBN";"7762";"105-54-4";{eb};NA',
            f'"4";"O ring";"gamma-decalactone";"IFY";"12813";"706-14-9";20;NA']
    if not gamma_only:
        rows.append(f'"5";"O ring";"delta-decalactone";"GHB";"12810";"705-86-2";{ddl};NA')
    return HEAD + "\n".join(rows) + "\n"


MAP_HEAD = ('"receptor";"sensillum";"OSN";"glomerulus";"co.receptor";"coexpressing";"related1";"related2";"related3";'
            '"related4";"related5";"related6";"Ors";"sensillum.type";"adult";"larva";"dataset.existing";"comment";"code";'
            '"code.OSN"\n')


def _map_row(i, rec, glom):
    return f'"{i}";"{rec}";"ab";"ab1A";"{glom}";"Orco";"";"";"";"";"";"";"";"{rec}";"x";TRUE;TRUE;TRUE;"";"{glom}";"ab1A"\n'


def test_values_are_matched_by_cas_and_delta_is_not_gamma():
    m = _script()
    got = m.receptor_values({"Or22a": _receptor()})
    assert got == {"Or22a": {"IA": "236", "EB": "197", "dDL": "15"}}
    with pytest.raises(m.Mismatch, match="dDL"):
        m.receptor_values({"Or22a": _receptor(gamma_only=True)})          # only γ present: a stop, never a substitute


def test_an_na_value_is_a_mismatch():
    m = _script()
    with pytest.raises(m.Mismatch, match="EB"):
        m.receptor_values({"Or22a": _receptor(eb="NA")})


def test_a_file_without_the_column_or_all_na_is_not_a_receptor():
    m = _script()
    no_col = '"Class";"Name";"InChIKey";"CID";"CAS";"Other.2001"\n"1";"ester";"x";"1";"123-92-2";4\n'
    all_na = _receptor(ia="NA", eb="NA", ddl="NA").replace(";4;NA", ";NA;NA").replace(";20;NA", ";NA;NA")
    assert m.receptor_values({"a": no_col, "b": all_na}) == {}


def test_mapping_duplicates_are_kept_raw_and_unmapped_receptors_stop():
    m = _script()
    text = MAP_HEAD + _map_row(1, "Or33b", "DM5+DM3") + _map_row(2, "Or22a", "DM2")
    assert m.receptor_glomeruli(text, ["Or33b", "Or22a"]) == {"Or33b": "DM5+DM3", "Or22a": "DM2"}
    with pytest.raises(m.Mismatch, match="unmapped"):
        m.receptor_glomeruli(MAP_HEAD + _map_row(1, "Or22a", "?"), ["Or22a"])
    with pytest.raises(m.Mismatch, match="mapping rows"):
        m.receptor_glomeruli(MAP_HEAD + _map_row(1, "Or22a", "DM2"), ["Or22a", "Or7a"])


def test_a_glomerulus_the_model_lacks_is_a_mismatch():
    m = _script()
    m.check_model({"Or33b": "DM5+DM3"}, ["ORN_DM5", "ORN_DM3"])
    with pytest.raises(m.Mismatch, match="DM3"):
        m.check_model({"Or33b": "DM5+DM3"}, ["ORN_DM5"])


def test_extract_counts_receptors_and_renders_sorted_csv():
    m = _script()
    files = {"Or22a": _receptor(), "Or7a": _receptor(ia="-16", eb="18", ddl="2")}
    mapping = MAP_HEAD + _map_row(1, "Or22a", "DM2") + _map_row(2, "Or7a", "DL5")
    table, mp = m.extract(files, mapping, ["ORN_DM2", "ORN_DL5"], n_expected=2)
    assert table == "receptor,IA,EB,dDL\nOr22a,236,197,15\nOr7a,-16,18,2\n"
    assert mp == "receptor,glomerulus\nOr22a,DM2\nOr7a,DL5\n"
    with pytest.raises(m.Mismatch, match="expected 24"):
        m.extract(files, mapping, ["ORN_DM2", "ORN_DL5"])


def test_render_of_the_24_receptors_is_the_pinned_bytes():
    m = _script()
    rows = [ln.split(",") for ln in TABLE.splitlines()[1:]]
    values = {r[0]: {"IA": r[1], "EB": r[2], "dDL": r[3]} for r in rows}
    glom = dict(ln.split(",") for ln in MAPPING.splitlines()[1:])
    table, mp = m.render(values, glom)
    assert (table, mp) == (TABLE, MAPPING)
    assert hashlib.sha256(table.encode()).hexdigest() == TABLE_SHA
    assert hashlib.sha256(mp.encode()).hexdigest() == MAPPING_SHA


def test_the_committed_files_are_the_pinned_bytes_with_license_and_provenance():
    d = ROOT / "data" / "odor"
    assert (d / "hallem2006_subset.csv").read_text() == TABLE
    assert (d / "door_mappings_subset.csv").read_text() == MAPPING
    assert "CC BY-SA 4.0" in (d / "NOTICE").read_text()
    assert "Attribution-ShareAlike 4.0" in (d / "LICENSE-CC-BY-SA-4.0.txt").read_text()
    import json
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["commit"] == "db323a496577c4b4a72b5c2fcd1859e07521ffb5"
    assert prov["outputs"]["hallem2006_subset.csv"] == TABLE_SHA
    assert {k: v["cas"] for k, v in prov["odorants"].items()} == {"IA": "123-92-2", "EB": "105-54-4",
                                                                  "dDL": "705-86-2"}
