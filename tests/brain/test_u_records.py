"""U's records: the side (T's z_side + every listed type's guard statistics, KC subtypes, APL release, block edges),
U.9.4 P2-6's per-f ratios, U.9.1's row and oracle comparisons (bit for bit except U's edge label / the edit labels), and
one L_f's even record with S's per-axis net drops."""
from flymon.brain import u_records as UR
from flymon.brain.u_spec import SPEC, at
from tests.brain.r_fixtures import fake_oracle, fake_rows, got_for, key_of
from tests.brain.u_fixtures import ref_rows, rest_rows

RO = {"A": "MBON13", "P": "MBON05"}
TYPES = ["MBON13", "MBON05", "MBON09", "CRE055"]
KC = list(SPEC.kc_types)


def test_side_holds_ts_z_side_and_the_mechanism_record():
    ref = ref_rows([1, 19] * 48, [7, 45] * 48, extra={"MBON09": 4, "CRE055": 0})
    rest = rest_rows(96, 0, extra={"MBON09": 1, "CRE055": 0})
    s = UR.side(ref, rest, RO, TYPES, KC, SPEC)
    assert s["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} and s["n_ref"] == s["n_rest"] == 96
    assert s["guard"]["MBON13"]["median_delta"] == 10.0 and s["edit_edges"] == [2]
    m = s["mech"]
    assert set(m["types"]) == set(TYPES) and m["types"]["MBON09"]["median_delta"] == 3.0
    assert m["types"]["CRE055"]["zero_share"] == 1.0 and not m["types"]["CRE055"]["passes"]
    assert m["kc_sub"]["KCa'b'-ap2"] == dict(stim_median=0.2, rest_median=0.01)
    assert m["kc_active_median"] == dict(stim=0.05, rest=0.002) and m["apl_out_median"] == dict(stim=0.3, rest=0.1)
    assert s["block_edges"] == ["{}"]


def test_f_record_ratios():
    s1 = dict(mech=dict(types={"MBON05": dict(mean_stim=26.25)}), z={"A": [10.78125, 9.4], "P": [26.25, 19.3]})
    sf = dict(mech=dict(types={"MBON05": dict(mean_stim=52.5)}), z={"A": [8.0, 4.7], "P": [52.5, 38.6]})
    r = UR.f_record(sf, s1, {"A": (10.78125, 9.4), "P": (26.25, 19.3)}, RO)
    assert r["p_mean_ratio"] == 2.0 and r["sd_ratio"] == {"A": 0.5, "P": 2.0}
    assert UR.f_record(dict(sf, z=None), s1, {"A": (1.0, 1.0), "P": (1.0, 1.0)}, RO)["sd_ratio"] is None


def test_row_diffs_compare_ts_fields_bit_for_bit_and_ignore_us_edge_label():
    t = [dict(odor="R00", seed=1, types={"MBON13": 2, "MBON05": 19}, kc_active_frac=0.1, csc_sha256="s",
              edit_edges=0)]
    u = [dict(t[0], types={"MBON13": 2, "MBON05": 19, "MBON09": 3}, edit_edges=2, mech={})]
    assert UR.row_diffs(u, t, "기준 집합") == []
    assert UR.row_diffs([dict(u[0], kc_active_frac=0.1000000001)], t, "x") == ["x: 1 row(s) differ: ['R00/1']"]
    assert UR.row_diffs([dict(u[0], types={"MBON13": 3, "MBON05": 19})], t, "x") != []
    assert UR.row_diffs([dict(u[0], csc_sha256="t")], t, "x") != []
    assert UR.row_diffs(u + u, t, "x") == ["x: 2 rows, T 1"]
    tr = [dict(seed=1, types={"MBON13": 0}, csc_sha256="s", edit_edges=0)]
    assert UR.row_diffs([dict(tr[0], edit_edges=2, kc_active_frac=0.0)], tr, "휴지", rest=True) == []


def test_oracle_diffs_ignore_only_the_edit_labels():
    rows = fake_rows(3, 0)
    r = got_for(rows, edit="apl_to_mbon05_zero", edges=2, sha="sha-L")
    u = got_for(rows, edit="u_apl_mbon05_x0.0", edges=2, sha="sha-L")
    assert UR.oracle_diffs(u, r, "R 짝수 L") == []
    u2 = got_for(rows, edit="u_apl_mbon05_x0.0", edges=2, sha="sha-X")
    assert UR.oracle_diffs(u2, r, "R 짝수 L") == [f"R 짝수 L: 3 pair(s) differ: {[key_of(x) for x in rows]}"]
    u3 = got_for(rows, plan={key_of(rows[1]): (True, True, False)}, edit="u_apl_mbon05_x0.0", edges=2, sha="sha-L")
    assert UR.oracle_diffs(u3, r, "L") == [f"L: 1 pair(s) differ: {[key_of(rows[1])]}"]
    assert UR.oracle_diffs(u[:2], r, "L") == ["L: pair lists differ"]
    assert UR.strip_labels(fake_oracle(edit="x", edges=5))["q"].keys() == {"csc_sha256", "fixed", "apl_out", "fx", "reach"}


def test_strip_labels_strips_only_q_edit_labels():
    x = dict(q=dict(edit="u_apl_mbon05_x0.0", edit_edges=2, csc_sha256="s"),
             other=dict(edit="keep", edit_edges=3, inner=[dict(edit="keep too")]), edit="top")
    assert UR.strip_labels(x) == dict(q=dict(csc_sha256="s"),
                                      other=dict(edit="keep", edit_edges=3, inner=[dict(edit="keep too")]), edit="top")


def test_even_record_reads_ss_net_drops():
    from flymon.brain import r_records
    spf = at(SPEC, 0.4)
    rows = fake_rows(21, 18, turn0=0)
    keys = [key_of(r) for r in rows]
    seeds = SPEC.h4_seeds()
    kb = [k for k in keys if k.startswith("b|")]
    ka = [k for k in keys if k.startswith("a|")]
    plan_l = {**{k: (True, True, False) for k in kb[:12]}, **{k: (False, False, False) for k in kb[12:15]},
              **{k: (False, False, False) for k in ka[:2]}}
    L = r_records.cond_summary(got_for(rows, plan_l, edges=2, edit=spf.lever_edit, sha="sha-L", n_rep=8, n_act=8),
                               spf.cond("L"), spf, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, keys, seeds)
    C = r_records.cond_summary(got_for(rows, {k: (True, True, False) for k in kb[:7]}, sha="sha-C"),
                               spf.cond("C"), spf, {"A": (10.0, 9.0), "P": (26.0, 19.0)}, keys, seeds)
    e = UR.even_record(L, C, spf)
    assert e["f"] == 0.4 and e["testable_b"] == 12 and e["edit_edges"] == [2] and e["reasons"] == []
    assert e["drops"]["b"] == dict(pun_L=18, pun_C=21, net_drop=3) and e["drops"]["a"]["net_drop"] == 2
