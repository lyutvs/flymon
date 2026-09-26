"""Spec L.4 / L.11.3: the new set — combos not in build_turns or H.5, shuffled once, type-disjoint alternates, odour-
level de-duplication against the old turns and within the set, digests pinned."""
import re
from pathlib import Path

import pytest

from flymon.battle.pool import POOL
from flymon.brain import l_pairs
from flymon.brain.h4_pairs import pool_vocabulary
from flymon.brain.l_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


def test_excluded_combos_are_build_turns_and_h5():
    ex = l_pairs.excluded_combos()
    for i in range(16):
        assert (POOL[i].species, POOL[(5 * i + 3) % 16].species) in ex
    assert (POOL[0].species, POOL[5].species) in ex          # H.5 i = 0: (7*0 + 5) % 16 = 5
    assert all(a != b for a, b in ex) and len(ex) <= 32


def test_new_turns_are_deterministic_disjoint_and_in_order():
    st, mi, _, _ = pool_vocabulary()
    t1 = l_pairs.new_turns(st, mi, 64, SPEC.rng_seed)
    t2 = l_pairs.new_turns(st, mi, 64, SPEC.rng_seed)
    assert t1 == t2 and len(t1) == 64 and [t["pos"] for t in t1] == list(range(64))
    combos = [(t["me"], t["opp"]) for t in t1]
    assert len(set(combos)) == 64 and not set(combos) & l_pairs.excluded_combos()
    assert all(t["me"] != t["opp"] for t in t1)
    hps = [(0.9, 0.9), (0.9, 0.3), (0.5, 0.6), (0.2, 0.8), (0.6, 0.2), (0.3, 0.5)]
    assert all((t["my_hp"], t["opp_hp"]) == hps[t["pos"] % 6] for t in t1)


def test_alternate_is_type_disjoint_and_raises_with_the_turn():
    st, _, _, _ = pool_vocabulary()
    alt = l_pairs.alternate_from("Chansey", "Blastoise", st["Chansey"], st, 3)
    assert alt != "Blastoise" and not set(st[alt]) & set(st["Chansey"])
    with pytest.raises(ValueError, match="turn 7"):
        l_pairs.alternate_from("Chansey", "Blastoise", st["Chansey"], {k: ("NORMAL",) for k in st}, 7)


def test_the_digests_are_pinned():
    assert re.fullmatch(r"[0-9a-f]{64}", SPEC.b_digest) and re.fullmatch(r"[0-9a-f]{64}", SPEC.a_digest)


def test_check_digests_refuses_an_empty_or_different_digest():
    import dataclasses
    from flymon.brain.h4_pairs import pairs_digest
    s = {"b": [], "a": [], "skipped": []}
    with pytest.raises(ValueError, match="b_digest"):
        l_pairs.check_digests(s, dataclasses.replace(SPEC, b_digest=""))
    with pytest.raises(ValueError, match="a_digest"):
        l_pairs.check_digests(s, dataclasses.replace(SPEC, b_digest=pairs_digest([]), a_digest=""))
    with pytest.raises(ValueError, match="differs"):
        l_pairs.check_digests(s, SPEC)


@needs_npz
def test_new_pairs_dedupe_and_digests():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h4_pairs import pairs_digest
    from flymon.brain.k_pairs import odour_key
    pops = Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))
    s = l_pairs.new_pairs(pops, SPEC)
    both = lambda p: frozenset((odour_key(p["odor_x"]), odour_key(p["odor_y"])))
    keys = [both(p) for p in s["b"] + s["a"]]
    assert len(set(keys)) == len(keys)                                   # no in-set duplicate
    assert all(p["turn"] < SPEC.a_turns for p in s["a"]) and s["a"]
    assert all(r["reason"] in ("old_set", "in_set") for r in s["skipped"])
    assert pairs_digest(s["b"]) == SPEC.b_digest and pairs_digest(s["a"]) == SPEC.a_digest
    assert l_pairs.check_digests(s, SPEC) is s
