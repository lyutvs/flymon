"""Z.3 as amended by Z.9.2 P0-1: the pre-declared split of the 31 oracle-lenient pairs into the confirmation half C
and the pilot half P, from keys only.
- lenient_items (plan Reading 3): the ONLY reader of results/y/oracle.json in Z. It checks the file's sha256, keeps the
  records Y's own rule passes (y_rules.passes(x, Y)["y_lenient"] — Y's _lenient path), checks the count, and returns
  (key, c, axis) per pair in c order; every value field (d_pre, L_A, L_P, r, p, testable, value, failure) is dropped
  inside the function and never returned, logged or written. A wrong sha256 or count raises LenientMismatch (the
  stage turns it into a refusal; order 0c has already checked both).
- split(items, forced, root): items = [(key, c, axis)] — the signature carries no value. Groups = pairs sharing the X
  odour (y_rules.x_odour, the key's third field), never split. Forced groups (X odour of a Y-admitted pair) go to C
  first. Strata (b) → (a) (s = 0, 1); the remaining groups of a stratum, sorted by their smallest c, are shuffled by
  numpy.random.default_rng(SeedSequence([root, s])).permutation; each goes (1) to the half with fewer pairs in this
  stratum (forced pairs counted), (2) on a tie to the half with fewer pairs overall, (3) on a tie to C.
- record(sp): the tracked summary — per half and stratum counts, groups per half, largest group, forced pair count,
  the sha256 of each half's key list (canonical JSON, c order); no key appears in it.
- fixtures(zs): Z.6.4's split fixtures (1)–(6) on synthetic keys (stage0 records them; the tests pin them)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import y_rules
from .h3_store import canonical, sha256_file
from .y_spec import SPEC as Y_SPEC


class LenientMismatch(Exception):
    pass


def lenient_items(path, sha256: str, n_len: int, ys=Y_SPEC) -> list:
    p = Path(path)
    if not p.exists() or sha256_file(p) != sha256:
        raise LenientMismatch(f"{path} is missing or its sha256 is not {sha256}")
    per = json.loads(p.read_text())["pairs"]
    out = sorted(((str(x["key"]), int(x["c"]), str(x["axis"])) for x in per if y_rules.passes(x, ys)["y_lenient"]),
                 key=lambda t: t[1])
    del per
    if len(out) != n_len:
        raise LenientMismatch(f"lenient-pass count {len(out)} ≠ {n_len}")
    return out


def forced_odours(admitted) -> frozenset:
    return frozenset(y_rules.x_odour(k) for k in admitted)


def split(items: list, forced: frozenset, root: int, strata=("b", "a")) -> dict:
    for key, c, axis in items:
        if key.split("|")[0] != axis:
            raise ValueError(f"item c={int(c)}: the axis field is not the key's first field")
    by_c = {c: key for key, c, _a in items}
    C, P = [], []
    forced_n = 0
    for s, axis in enumerate(strata):
        groups = {}
        for key, c, a in sorted(items, key=lambda t: t[1]):
            if a == axis:
                groups.setdefault(y_rules.x_odour(key), []).append(c)
        fixed = [v for x, v in groups.items() if x in forced]
        free = sorted((v for x, v in groups.items() if x not in forced), key=min)
        sc = sp_ = 0
        for v in fixed:
            C += v
            sc += len(v)
            forced_n += len(v)
        perm = np.random.default_rng(np.random.SeedSequence([int(root), s])).permutation(len(free))
        for i in perm:
            v = free[int(i)]
            if sc != sp_:
                to_c = sc < sp_
            elif len(C) != len(P):
                to_c = len(C) < len(P)
            else:
                to_c = True
            if to_c:
                C += v
                sc += len(v)
            else:
                P += v
                sp_ += len(v)
    C, P = sorted(C), sorted(P)
    return dict(C=[by_c[c] for c in C], P=[by_c[c] for c in P], C_c=C, P_c=P, forced_n=forced_n)


def _sha(keys: list) -> str:
    return hashlib.sha256(canonical(list(keys)).encode()).hexdigest()


def record(sp: dict, forced: frozenset, strata=("b", "a")) -> dict:
    def per(keys):
        g = y_rules.merge_groups(list(keys))
        return dict(n=len(keys), by_stratum={a: sum(k.split("|")[0] == a for k in keys) for a in strata},
                    n_groups=len(g), largest_group=max((len(x) for x in g), default=0))
    return dict(C=per(sp["C"]), P=per(sp["P"]), forced_pairs=int(sp["forced_n"]),
                shared_with_y=sum(y_rules.x_odour(k) in forced for k in sp["C"] + sp["P"]),
                sha256=dict(C=_sha(sp["C"]), P=_sha(sp["P"])))


def bound_ok(sp: dict, forced: frozenset, strata=("b", "a")) -> bool:
    """Fixture (4) with Z.9.2 P0-1's limit: per stratum |C_s − P_s| ≤ max(largest free group, forced − free pairs)."""
    for a in strata:
        keys = [k for k in sp["C"] + sp["P"] if k.split("|")[0] == a]
        g = {}
        for k in keys:
            g.setdefault(y_rules.x_odour(k), []).append(k)
        free = [len(v) for x, v in g.items() if x not in forced]
        n_forced = sum(len(v) for x, v in g.items() if x in forced)
        cs = sum(k.split("|")[0] == a for k in sp["C"])
        ps = sum(k.split("|")[0] == a for k in sp["P"])
        if abs(cs - ps) > max(max(free, default=0), n_forced - sum(free)):
            return False
    return True


def synthetic_items(n_b: int, n_a: int, a_odours: tuple) -> list:
    """Synthetic keys for the fixtures: (b) one X per pair, (a) cycling through a_odours; c = declared order."""
    out, c = [], 0
    for i in range(n_b):
        out.append((f"b|{i}|xb{i}|yb{i}", c, "b"))
        c += 1
    for i in range(n_a):
        out.append((f"a|{n_b + i}|{a_odours[i % len(a_odours)]}|ya{i}", c, "a"))
        c += 1
    return out


def fixtures(zs) -> dict:
    """Z.6.4 (1)–(6) + Z.9.2 T5 on synthetic keys under the split root (a record; the tests hold the expectations)."""
    forced = forced_odours(zs.y_admitted)
    n = dict(zs.n_len_axes)
    items = synthetic_items(n["b"], n["a"], ("Earthquake", "Surf", "Psychic", "Flamethrower"))
    sp = split(items, forced, zs.split_seed, zs.strata)
    out = {}
    out["f1_same"] = split(items, forced, zs.split_seed, zs.strata) == sp
    out["f3_groups_whole"] = not (set(map(y_rules.x_odour, sp["C"])) & set(map(y_rules.x_odour, sp["P"])))
    out["f4_bound"] = bound_ok(sp, forced, zs.strata)
    out["f5_forced_in_c"] = all(y_rules.x_odour(k) not in forced for k in sp["P"])
    free = synthetic_items(n["b"], n["a"], tuple(f"xa{i}" for i in range(n["a"])))     # singletons, none forced
    spf = split(free, frozenset(), zs.split_seed, zs.strata)
    out["t5_c_larger_without_forced"] = len(spf["C"]) - len(spf["P"]) in (0, 1)
    out["ok"] = all(out.values())
    return out
