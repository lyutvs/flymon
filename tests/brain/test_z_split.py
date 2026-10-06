"""Z.3 + Z.9.2 P0-1: the key-only split (fixtures (1)–(6), T5, T6) and the lenient-key reader (plan Reading 3)."""
import dataclasses
import inspect
import json

import numpy as np
import pytest

from flymon.brain import y_rules
from flymon.brain import z_rules as R
from flymon.brain import z_split as S
from flymon.brain.h3_store import sha256_file
from flymon.brain.z_spec import SPEC as Z

FORCED = S.forced_odours(Z.y_admitted)
ITEMS = S.synthetic_items(11, 20, ("Earthquake", "Surf", "Psychic", "Flamethrower"))
WANT_C = [0, 1, 2, 3, 5, 6, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30]
WANT_P = [4, 7, 8, 9, 10, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29]


def test_forced_odours_are_y_admitted_x():
    assert FORCED == {"Rock Slide", "Mega Drain vs Machamp", "Surf vs Slowbro", "Thunderbolt vs Nidoran-M",
                      "Earthquake"}
    assert "Surf" not in FORCED                                   # a|223 Surf|Earthquake was not admitted


def test_pinned_split_and_record():
    sp = S.split(ITEMS, FORCED, Z.split_seed, Z.strata)
    assert (sp["C_c"], sp["P_c"], sp["forced_n"]) == (WANT_C, WANT_P, 5)
    rec = S.record(sp, FORCED, Z.strata)
    assert rec["C"]["by_stratum"] == {"b": 6, "a": 10} and rec["P"]["by_stratum"] == {"b": 5, "a": 10}
    assert rec["shared_with_y"] == 5
    assert rec["forced_pairs"] == 5 and set(rec) == {"C", "P", "forced_pairs", "shared_with_y", "sha256"}
    assert "|" not in json.dumps(rec)                               # no key in the tracked record


def test_fixture1_same_input_same_split_and_fixtures_ok():
    assert S.split(ITEMS, FORCED, Z.split_seed) == S.split(list(ITEMS), FORCED, Z.split_seed)
    assert S.fixtures(Z)["ok"]


def test_fixture2_signature_carries_no_value():
    assert list(inspect.signature(S.split).parameters) == ["items", "forced", "root", "strata"]
    with pytest.raises(ValueError):
        S.split([(k, c, a) for k, c, a in ITEMS] + [("a|99|X|Y", 99, "b")], FORCED, Z.split_seed)


def test_split_axis_error_names_no_key():
    with pytest.raises(ValueError) as e:
        S.split([("a|99|Secret Odour|Other", 99, "b")], FORCED, Z.split_seed)
    msg = str(e.value)
    assert "|" not in msg and "Secret" not in msg and "a|99" not in msg


def test_fixture3_groups_never_split_and_fixture5_forced_in_c():
    sp = S.split(ITEMS, FORCED, Z.split_seed)
    xc, xp = {y_rules.x_odour(k) for k in sp["C"]}, {y_rules.x_odour(k) for k in sp["P"]}
    assert not xc & xp and "Earthquake" in xc


def test_fixture4_balance_bound():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        odours = tuple(rng.choice(["Earthquake", "Rock Slide", "Surf", "Psychic", "Flamethrower", "Mega Drain"],
                                  size=4, replace=False))
        items = S.synthetic_items(int(rng.integers(5, 15)), int(rng.integers(5, 25)), odours)
        assert S.bound_ok(S.split(items, FORCED, Z.split_seed), FORCED)


def test_fixture4_bound_fails_on_an_unbalanced_split():
    keys = [f"b|{i}|xb{i}|yb{i}" for i in range(4)]                # 4 free singletons, all in C
    assert not S.bound_ok(dict(C=keys, P=[]), frozenset())
    assert S.bound_ok(dict(C=keys[:2], P=keys[2:]), frozenset())


def test_free_groups_sorted_by_smallest_c_not_size():
    seq = ["X1", "X2", "X2", "X2", "X3", "X3", "X4", "X5", "X5"]   # sizes 1, 3, 2, 1, 2 — c order ≠ size order
    items = [(f"a|{c}|{x}|y{c}", c, "a") for c, x in enumerate(seq)]
    sp = S.split(items, frozenset(), Z.split_seed)
    assert (sp["C_c"], sp["P_c"]) == ([1, 2, 3, 6], [0, 4, 5, 7, 8])


def test_fixture6_order_and_sorting_decide():
    a = S.split(ITEMS, FORCED, Z.split_seed, ("b", "a"))
    b = S.split(ITEMS, FORCED, Z.split_seed, ("a", "b"))
    assert a != b                                                  # the stratum order matters
    shuffled = list(reversed(ITEMS))                               # input order does not (sorted by c inside)
    assert S.split(shuffled, FORCED, Z.split_seed) == a


def test_t5_singletons_without_forced_c_is_the_larger_half():
    items = S.synthetic_items(11, 20, tuple(f"xa{i}" for i in range(20)))
    sp = S.split(items, frozenset(), Z.split_seed)
    assert len(sp["C"]) - len(sp["P"]) == 1


@pytest.mark.parametrize("mut", ["pairwise", "no_strata", "swap", "root77", "no_tie2"])
def test_mutants_change_the_pinned_split(mut, monkeypatch):
    if mut == "pairwise":                                          # ignore groups: each pair its own group
        monkeypatch.setattr(S.y_rules, "x_odour", lambda k: k)
        got = S.split(ITEMS, frozenset(), Z.split_seed)
    elif mut == "no_strata":
        got = S.split([(k.replace("a|", "b|", 1), c, "b") for k, c, _a in ITEMS], FORCED, Z.split_seed)
    elif mut == "swap":
        sp = S.split(ITEMS, FORCED, Z.split_seed)
        got = dict(sp, C_c=sp["P_c"], P_c=sp["C_c"])
    elif mut == "root77":
        got = S.split(ITEMS, FORCED, Y_ROOT)
    else:                                                          # tie rule (2) dropped: overall count ignored
        src = inspect.getsource(S.split).replace("elif len(C) != len(P):", "elif False:")
        ns = dict(S.__dict__)
        exec(compile(src, "mut", "exec"), ns)
        got = ns["split"](ITEMS, FORCED, Z.split_seed)
    assert (got["C_c"], got["P_c"]) != (WANT_C, WANT_P)


Y_ROOT = 77_000_000
Z_Y = S.Y_SPEC


def _oracle(tmp_path, recs):
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps(dict(pairs=recs)))
    return p, sha256_file(p)


def _rec(key, c, d, la, lp):
    return dict(key=key, c=c, axis=key.split("|")[0], turn=int(key.split("|")[1]), value=True, testable=True,
                d_pre=d, L_A=la, L_P=lp, r=0.1, p=0.2, failure=None)


def test_lenient_items_returns_keys_only_and_is_value_invariant(tmp_path):
    keys = [f"b|{i}|x{i}|y{i}" for i in range(4)] + [f"a|{10 + i}|Surf|y{i}" for i in range(3)]
    recs = [_rec(k, c, 0.1, 30.0, 60.0) for c, k in enumerate(keys)]
    recs[2] = _rec(keys[2], 2, 1.5, 30.0, 60.0)                    # not lenient (|d| ≥ 1.0)
    recs[3] = _rec(keys[3], 3, 0.7, 17.0, 60.0)                    # lenient but not strict (d 0.7, L_A 17)
    assert y_rules.passes(recs[3], Z_Y)["y_lenient"] and not y_rules.passes(recs[3], Z_Y)["y_strict"]
    p, sha = _oracle(tmp_path, recs)
    got = S.lenient_items(p, sha, 6)
    assert got == [(k, c, k[0]) for c, k in enumerate(keys) if c != 2]
    assert all(len(t) == 3 for t in got)
    rng = np.random.default_rng(0)
    recs2 = [_rec(r["key"], r["c"], r["d_pre"] if abs(r["d_pre"]) >= 1.0 else -r["d_pre"] * 0.5,
                  16.0 + rng.uniform(0, 50), 34.4 + rng.uniform(0, 90)) for r in recs]
    (tmp_path / "b").mkdir()
    p2, sha2 = _oracle(tmp_path / "b", recs2)
    assert S.lenient_items(p2, sha2, 6) == got                     # other lenient values, same keys
    assert S.split(got, FORCED, Z.split_seed) == S.split(S.lenient_items(p2, sha2, 6), FORCED, Z.split_seed)


def test_lenient_items_refuses_sha_and_count(tmp_path):
    p, sha = _oracle(tmp_path, [_rec("b|1|x|y", 0, 0.1, 30.0, 60.0)])
    with pytest.raises(S.LenientMismatch):
        S.lenient_items(p, "0" * 64, 1)
    with pytest.raises(S.LenientMismatch):
        S.lenient_items(p, sha, 2)
    with pytest.raises(S.LenientMismatch):
        S.lenient_items(tmp_path / "missing.json", sha, 1)


def test_t6_key_precheck_from_split_keys_only():
    sp = S.split(ITEMS, FORCED, Z.split_seed)
    pre = R.key_precheck(sp["P"], Z)
    assert pre["h"] == 15 and pre["j_max"] == 21
    assert pre["n_sigma_max"] == len(y_rules.merge_groups(list(Z.y_admitted) + sp["P"]))
    assert list(inspect.signature(R.key_precheck).parameters) == ["pilot_keys", "zs"]
