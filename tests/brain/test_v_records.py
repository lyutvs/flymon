"""V's records (V.1, V.3 2-3, V.6, V.9.2, V.9.5): the combined edit's static facts on the real connectome (13 edges
changed by bits — APL->MBON05 2 and the chain entry 7 / 2 / 2 — CSC 2d359b8b…, MBON05->APL untouched) and its
INVALID reasons; a mutation whose chain entry is not applied fails them; full-row comparison with U's rows; the KC input
record (per-odour medians, seed ranges, ORN cap) and the bit-for-bit comparison with U's per_odour_none; the side
record; the two-scale gate-② ratio (L on z_L, C on z_C)."""
from pathlib import Path

import pytest

from flymon.brain import s_records, v_records
from flymon.brain.v_spec import SPEC
from tests.brain.s_fixtures import p_rows
from tests.brain.u_fixtures import ref_rows, rest_rows

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


@pytest.fixture(scope="module")
def real():
    if not NPZ.exists():
        pytest.skip("no connectome")
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    conn = Connectome.load(str(NPZ))
    return dict(conn=conn, pops=Populations.from_connectome(conn),
                params=load_c3_config(str(ROOT / SPEC.m0d_summary)).params)


def test_combined_edit_facts_on_the_real_connectome(real):
    f = v_records.csc_facts(real["conn"], real["pops"], real["params"], SPEC)
    assert f["sha_none"] == SPEC.sha_none and f["csc_sha256"] == SPEC.sha_combined
    assert (f["apl_edges"], f["changed"]) == (2, 13)
    assert f["block_edges"] == {"MBON05->MBON01": 2, "MBON05->MBON09": 7, "MBON05->MBON11": 2}
    assert f["changed_pairs"] == {"APL->MBON05": 2, "MBON05->MBON01": 2, "MBON05->MBON09": 7, "MBON05->MBON11": 2}
    assert f["mbon05_apl"]["n"] == 2 and f["mbon05_apl"]["same_bits"]
    assert sorted(f["mbon05_apl"]["after"]) == pytest.approx([-12.928195, -7.8651862], abs=1e-6)
    assert v_records.csc_reasons(f, SPEC) == []


def test_mutation_an_edit_without_the_chain_entry_fails(real, monkeypatch):
    from flymon.brain import u_measure
    from flymon.brain.u_measure import u_edit
    real_apply = u_measure.apply_u_edit
    monkeypatch.setattr(u_measure, "apply_u_edit",
                        lambda eng, pops, edit, p: real_apply(eng, pops, u_edit(0.0), p))
    f = v_records.csc_facts(real["conn"], real["pops"], real["params"], SPEC)
    bad = v_records.csc_reasons(f, SPEC)
    assert f["changed"] == 2 and any("chain entry" in b for b in bad) and any("CSC" in b for b in bad)


def test_csc_reasons_name_each_break():
    ok = dict(sha_none=SPEC.sha_none, csc_sha256=SPEC.sha_combined, apl_edges=2,
              block_edges={"MBON05->MBON01": 2, "MBON05->MBON09": 7, "MBON05->MBON11": 2}, changed=13,
              mbon05_apl=dict(n=2, before=[-7.8651862, -12.928195], after=[-7.8651862, -12.928195], same_bits=True))
    assert v_records.csc_reasons(ok, SPEC) == []
    for k, v in (("apl_edges", 1), ("changed", 14), ("csc_sha256", "x"), ("sha_none", "y"),
                 ("mbon05_apl", dict(ok["mbon05_apl"], same_bits=False)),
                 ("mbon05_apl", dict(ok["mbon05_apl"], after=[-7.8651862, -12.9302])),
                 ("block_edges", {"MBON05->MBON09": 7})):
        assert len(v_records.csc_reasons(dict(ok, **{k: v}), SPEC)) == 1, k


def test_full_row_diffs_compare_every_field():
    a = ref_rows([1, 2], [3, 4], 2, "sha-V", blocks={"MBON05->MBON09": 7})
    assert v_records.full_row_diffs(a, [dict(r) for r in a], "기준 집합") == []
    b = [dict(r) for r in a]
    b[1] = dict(b[1], mech=dict(b[1]["mech"], apl_out_per_step=0.31))
    assert v_records.full_row_diffs(a, b, "기준 집합") == ["기준 집합: 1 row(s) differ: ['R00/1001']"]
    assert v_records.full_row_diffs(a, b[:1], "x") == ["x: 2 rows, U 1"]
    r = rest_rows(2, 0, 2, "sha-V")
    assert v_records.full_row_diffs(r, [dict(x) for x in r], "휴지") == []


def _act(vals: dict, edges: int, sha: str) -> dict:
    return {o: dict(frac=list(v), max_win=[3] * len(v), edit_edges=[edges], csc_sha256=[sha]) for o, v in vals.items()}


def test_kc_input_record_and_the_u_comparison():
    act = {"none": _act({"A|B": [0.04, 0.05, 0.06], "C|D": [0.02, 0.02, 0.03]}, 0, "sha-C"),
           "lever": _act({"A|B": [0.05, 0.05, 0.05], "C|D": [0.16, 0.2, 0.2]}, 2, "sha-V")}
    rec = v_records.kc_input_record(act, {"A|B": 300.0, "C|D": 200.0}, 1000 / 3, SPEC)
    assert rec["none"]["per_odour"] == {"A|B": 0.05, "C|D": 0.02} and rec["none"]["outside"] == ["C|D"]
    assert rec["lever"]["outside"] == ["C|D"] and rec["none"]["seed_range"]["A|B"] == [0.04, 0.06]
    assert rec["none"]["n_seeds"] == [3] and rec["lever"]["edit_edges"] == [2] and rec["cap"]["all_under"]
    assert rec["band"] == [0.03, 0.15]
    cmp = v_records.kc_u_diffs(rec["none"]["per_odour"], {"A|B": 0.05, "X|Y": 0.1})
    assert (cmp["n_shared"], cmp["differ"], cmp["diffs"]) == (1, [], [])
    cmp = v_records.kc_u_diffs(rec["none"]["per_odour"], {"A|B": 0.0500001})
    assert cmp["differ"] == ["A|B"] and cmp["diffs"] == ["A|B 0.05 ≠ U 0.0500001"]


def test_side_record_ratios():
    s_none = dict(mech=dict(types={"MBON05": dict(mean_stim=26.0)}), z={"A": [10.0, 9.0], "P": [26.0, 19.0]})
    s_v = dict(mech=dict(types={"MBON05": dict(mean_stim=80.0)}), z={"A": [17.0, 12.0], "P": [80.0, 29.0]})
    rec = v_records.side_record(s_v, s_none, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, {"A": "MBON13", "P": "MBON05"})
    assert rec == dict(p_mean_ratio=80.0 / 26.0, sd_ratio={"A": 12.0 / 9.0, "P": 29.0 / 19.0})


def test_ratio_two_z_equals_ss_ratio_on_one_z_and_scales_with_sigma_a():
    rows_l = p_rows(SPEC.p, SPEC.lever_edit, 12, "sha-V", 2)
    rows_c = p_rows(SPEC.p_c, "none", 16)
    z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}
    one = s_records.ratio(rows_l, rows_c, z, SPEC)
    two = v_records.ratio_two_z(rows_l, rows_c, z, z, SPEC)
    assert {d: v["ratio"] for d, v in one.items()} == {d: v["ratio"] for d, v in two.items()}
    zv = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
    two_v = v_records.ratio_two_z(rows_l, rows_c, zv, z, SPEC)
    for d in two_v:
        assert two_v[d]["ratio"] == pytest.approx(3 * one[d]["ratio"]) and two_v[d]["ell_C"] == one[d]["ell_C"]
    with pytest.raises(ValueError):
        v_records.ratio_two_z(rows_l[:-1], rows_c, zv, z, SPEC)
