# flymon/brain/q_pairs.py
"""Q.3's pairs: the E-grid k2-norm (s 1.0) even (b) 21 pairs — e_pairs.even_situations (H.4's situations and order;
the E0 digest is checked there) re-encoded with the encoder's committed k = 2 codebook (digest checked against block
codebook) — the Q0 file of each pair (block even's manifest basename under results/q/q0_cache, sha256 checked; Q0 never
opens results/encoder), the encoder-key check of the rebuilt oracle inputs, and the record of the cap-limited max s (Q.6.4; since Q.6.9 only a record — (c) lowers s to 0.7/0.8)."""
from __future__ import annotations

from pathlib import Path

from ..agent import e_codebook, e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import Codebook
from .h3_store import sha256_file
from .h4_pairs import pool_vocabulary


def key_str(r) -> str:
    return f"{r['axis']}|{int(r['turn'])}|{r['x']}|{r['y']}"


def check_turns(rows, spec) -> None:
    """Only H.4's even turns 0..even_turns-1; anything else (an odd turn, the L judgement set's 64-103) is refused."""
    bad = sorted({int(r["turn"]) for r in rows if not (0 <= int(r["turn"]) < spec.even_turns and int(r["turn"]) % 2 == 0)})
    if bad:
        raise ValueError(f"Q uses only H.4's even turns 0..{spec.even_turns - 1}; got turns {bad} "
                         f"(the L judgement set is never used)")


def codebook(enc: dict, spec) -> Codebook:
    k = str(E.k_of(spec.config))
    cfg, rec = enc["codebook"]["configs"][spec.config], enc["codebook"]["k"][k]
    book = rec["codebook"]
    if cfg["status"] != "OK" or book is None or e_codebook.digest(book) != cfg["digest"] or rec["digest"] != cfg["digest"]:
        raise ValueError(f"{spec.config}: codebook digest does not match block codebook's {cfg['digest']}")
    _, _, mon, move = pool_vocabulary()
    cells = e_codebook.cells(move, mon)
    if [list(c) for c in cells] != [list(c) for c in rec["cells"]]:
        raise ValueError(f"{spec.config}: codebook cells differ from block codebook's")
    return Codebook(cells, book)


def check_strength(enc: dict, spec) -> None:
    s1 = enc["strength"]["configs"][spec.config]["s"]
    s2 = enc["even"]["configs"][spec.config]["s"]
    if not (s1 == s2 == spec.strength):
        raise ValueError(f"{spec.config}: encoder strength {s1}/{s2} is not Q's s {spec.strength}")


def b_rows(situations: list, receptor_counts: dict, enc: dict, spec) -> list:
    check_turns(situations, spec)
    cb = codebook(enc, spec)
    rows = [r for r in e_pairs.attach_odours(situations, receptor_counts, cb, E.dual_rule(spec.config))
            if r["axis"] == "b"]
    if len(rows) != spec.n_b:
        raise ValueError(f"{len(rows)} (b) rows, not {spec.n_b}")
    return rows


def manifest(enc: dict, spec) -> dict:
    out = {key_str(e): e for e in enc["even"]["configs"][spec.config]["manifest"] if e["axis"] == "b"}
    if len(out) != spec.n_b:
        raise ValueError(f"block even's manifest holds {len(out)} (b) entries, not {spec.n_b}")
    return out


def q0_files(rows: list, enc: dict, spec) -> list:
    man = manifest(enc, spec)
    out = []
    for r in rows:
        e = man[key_str(r)]
        p = Path(spec.q0_cache_dir) / Path(e["cache_file"]).name
        got = sha256_file(p) if p.exists() else None
        out.append(dict(key=key_str(r), path=str(p), cache_key=e["cache_key"], sha256_declared=e["cache_sha256"],
                        sha256=got, ok=bool(got is not None and got == e["cache_sha256"])))
    return out


def encoder_key_match(rows, enc, spec, code: dict, params, readout, z, types, n_kc) -> list:
    """The encoder's oracle cache key of each row's rebuilt inputs (EMeasurer.oracle_key — computed, nothing is read or
    written) equals block even's manifest cache_key: same odours, Params, readout, z, types, H.4 seeds, alphas, windows."""
    from ..agent.e_measure import EMeasurer
    from ..agent.e_store import ECache
    m = EMeasurer(None, ECache(f"{E.raw_dir}/cache", code), params, n_kc, E, guard_params=False)
    man = manifest(enc, spec)
    return [m.oracle_key(r, spec.strength, readout, z, types, spec.repro_seeds())[0] == man[key_str(r)]["cache_key"]
            for r in rows]


def odours_of(rows) -> list:
    return [o for r in rows for o in (r["odor_x"], r["odor_y"])]


def cap_s(odours, max_rate_hz: float, cap_hz: float, spec) -> dict:
    """RECORD only (Q.6.4 as superseded by Q.6.9; this is NOT (c)'s strength — (c) lowers s to 0.7/0.8): the largest s on
    the s_step grid with max_rate_hz * s * v <= cap_hz for every glomerulus value v of every odour (encode_grid.cap_ok's
    inequality); weak <=> |s - strength| < s_weak."""
    vals = [float(v) for o in odours for v in o.values()]

    def ok(s):
        return all(max_rate_hz * s * v <= cap_hz for v in vals)

    if not ok(spec.strength):
        raise ValueError(f"s {spec.strength} is already over the ORN cap")
    n = int(round(spec.strength / spec.s_step))
    while ok(round((n + 1) * spec.s_step, 10)):
        n += 1
    s = round(n * spec.s_step, 10)
    return dict(s=s, weak=bool(abs(round(s - spec.strength, 10)) < spec.s_weak), vmax=max(vals), step=spec.s_step,
                strength=spec.strength, max_rate_hz=float(max_rate_hz), cap_hz=float(cap_hz), n_odours=len(odours))
