"""The measurement layer and pinned inputs of spec appendix L (L.11.1, L.11.3). Measurements: H.4's oracle unchanged
(through h4_measure.H4Measurer, in the scripts) and the naive screening job, cached per pair under
h3_store.MeasureCache keyed by MEASURE_FILES. Inputs: H.4a.8's ceiling raw file (the even pairs' f vectors and
select-seed probes) and K's C3 activity cache (the odd pairs' f vectors), each refused on another sha256, another
engine, other seeds than the declared ones (an empty probe list would pass guard G) or another odour order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import l_jobs
from .h3_store import sha256_file
from .k_measure import HASHED_FILES as K_HASHED_FILES, KMeasurer, MEASURE_FILES as K_MEASURE_FILES
from .k_pairs import odour_key

MEASURE_FILES = tuple(dict.fromkeys(K_MEASURE_FILES + ("flymon/brain/l_jobs.py", "flymon/brain/l_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + K_HASHED_FILES + (
    "flymon/brain/l_spec.py", "flymon/brain/l_store.py", "flymon/brain/l_pairs.py", "flymon/brain/l_screen.py",
    "flymon/brain/l_rules.py", "flymon/brain/l_oc.py", "flymon/brain/l_cli.py", "scripts/run_l_stage0.py",
    "scripts/run_l_oc.py", "scripts/run_l_stage1.py", "scripts/run_l_stage2.py", "scripts/run_l_stage3.py")))


def check_pinned(path, sha: str) -> dict:
    got = sha256_file(path)
    if got != sha:
        raise ValueError(f"{path}: sha256 {got[:12]} is not the pinned {sha[:12]}")
    return json.loads(Path(path).read_text())


def dense(idx, val, n: int) -> np.ndarray:
    v = np.zeros(int(n)); v[np.asarray(idx, np.int64)] = np.asarray(val, float)
    return v


def odours_digest(odours: list) -> str:
    """sha256 of the ordered odour_key list (the refusal names both sides by it)."""
    return hashlib.sha256(json.dumps([odour_key(o) for o in odours]).encode()).hexdigest()


def _seeds(spec) -> tuple:
    return [int(s) for s in spec.j.h4.act_seeds], [int(s) for s in spec.j.h4.select_seeds]


def load_even(spec, c3_json: dict) -> list:
    """The even (b) pairs' fx, fy and select-seed probes from H.4a.8's raw file (rows C3, mode x_only), in H.4 order.
    Refused unless the run is the complete clean one on H.4's pairs, on C3's engine and readout, on exactly the
    declared act / select seeds, with every row carrying one select probe per seed for every type."""
    d = check_pinned(spec.ceiling_path, spec.ceiling_sha256)
    if d.get("pairs_digest") != spec.j.h4.pairs_digest or not d.get("complete") or d.get("smoke") or d.get("dirty_hashed"):
        raise ValueError("ceiling raw file is not the complete, clean run on the declared pairs")
    if d["params"]["C3"] != c3_json:
        raise ValueError("ceiling raw file's C3 engine is not the recorded C3 engine")
    if d["readout"]["C3"] != dict(spec.readout):
        raise ValueError(f"ceiling raw file's C3 readout {d['readout']['C3']} is not {dict(spec.readout)}")
    act, sel = _seeds(spec)
    if [int(s) for s in d["spec"]["act_seeds"]] != act or [int(s) for s in d["spec"]["select_seeds"]] != sel:
        raise ValueError(f"ceiling raw file was measured on act seeds {d['spec']['act_seeds']} / select seeds "
                         f"{d['spec']['select_seeds']}, not the declared {act} / {sel}")
    out = []
    for r in d["rows"]["C3"]:
        if r["mode"] != "x_only" or r["axis"] != "b":
            continue
        key = ("b", int(r["turn"]), r["x"], r["y"])
        pre = r["select"]["pre"]
        for t in spec.guard_types:
            if t not in pre:
                raise ValueError(f"ceiling row {key} has no select probes for {t}")
        if any(len(v) != len(sel) or any(len(xy) != 2 for xy in v) for v in pre.values()):
            raise ValueError(f"ceiling row {key}: select probes are not one [x, y] per declared select seed")
        if any(len(r["kc"][k]["frac"]) != len(act) for k in ("x", "y")):
            raise ValueError(f"ceiling row {key}: KC activity is not one entry per declared act seed")
        out.append(dict(key=key, fx=np.asarray(r["fx"], float), fy=np.asarray(r["fy"], float), pre=pre))
    if len({g["key"] for g in out}) != len(out):
        raise ValueError("ceiling raw file repeats an even (b) pair")
    return out


def load_odd_activity(spec, odd: list, n_kc: int, c3_json: dict) -> list:
    """The odd pairs' fx / fy from K's C3 activity cache, refused unless its engine and presentation are C3's / H.4's,
    its odours are KMeasurer's de-duplicated list for `odd` in the same order, and every odour has each declared act
    seed exactly once."""
    d = check_pinned(spec.k_act_path, spec.k_act_sha256)
    inp = d["inputs"]
    if inp["params"] != c3_json:
        raise ValueError("K activity cache is not the recorded C3 engine")
    h4 = spec.j.h4
    want = dict(strength=h4.h3.strength, settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms,
                window_ms=int(h4.kc_window_ms))
    if {k: inp.get(k) for k in want} != want:
        raise ValueError(f"K activity cache's presentation {({k: inp.get(k) for k in want})} is not H.4's {want}")
    km = KMeasurer(None, None, odd, n_kc, None)
    got_d, want_d = odours_digest(inp["odours"]), odours_digest(km.odours)
    if got_d != want_d:
        raise ValueError(f"K activity cache's odour order differs from odd_pairs' de-duplicated odours: "
                         f"expected {want_d}, got {got_d}")
    seeds, _ = _seeds(spec)
    if [int(s) for s in inp["seeds"]] != seeds:
        raise ValueError(f"K activity cache was measured on act seeds {inp['seeds']}, not the declared {seeds}")
    per = {i: [] for i in range(len(km.odours))}
    fired = np.zeros((len(km.odours), int(n_kc)))
    for r in d["result"]:
        if r["i"] not in per:
            raise ValueError(f"K activity cache has a row for odour {r['i']} outside the {len(km.odours)} odours")
        if r["kc"] and max(r["kc"]) >= int(n_kc):
            raise ValueError(f"K activity cache names KC {max(r['kc'])} outside the {n_kc} KCs")
        per[r["i"]].append(int(r["seed"]))
        fired[r["i"], r["kc"]] += 1
    bad = [i for i, s in per.items() if sorted(s) != sorted(seeds)]
    if bad:
        raise ValueError(f"K activity cache: odours {bad} do not have each declared act seed {seeds} exactly once")
    fired /= len(seeds)
    return [dict(fx=fired[ix], fy=fired[iy]) for ix, iy in km.index]


class _Missing(Exception):
    pass


def _missing(*_):
    raise _Missing                      # h4_measure's pattern: a cache probe that never computes (and writes nothing)


class LMeasurer:
    def __init__(self, pool, spec, types, cache):
        """types: the probed MBON types (H.4's pools, reading 1); spec: LSpec (seeds and windows through spec.j.h4)."""
        self.pool, self.spec, self.types, self.cache = pool, spec, tuple(types), cache
        self.params_seen: list = []

    def naive(self, params, pairs: list) -> list:
        """naive_job per pair, cached per pair, run in rounds of one pair per worker (an interruption loses a round)."""
        if params not in self.params_seen:
            self.params_seen.append(params)
        h4 = self.spec.j.h4
        common = dict(params=params, types=self.types, act_seeds=tuple(h4.act_seeds), select_seeds=tuple(h4.select_seeds),
                      strength=h4.h3.strength, settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms,
                      window_ms=int(h4.kc_window_ms))
        items = [dict(common, odor_x=p["odor_x"], odor_y=p["odor_y"]) for p in pairs]
        inputs = lambda i: dict(items[i], pair=[pairs[i]["axis"], int(pairs[i]["turn"]), pairs[i]["x"], pairs[i]["y"]])
        done = {}
        for i in range(len(items)):
            try:
                done[i] = self.cache.get_or_compute("l_naive", inputs(i), _missing, [params])
            except _Missing:
                pass
        todo = [i for i in range(len(items)) if i not in done]
        n = max(1, self.pool.n_workers)
        for r in range(0, len(todo), n):
            batch = todo[r:r + n]
            for i, out in zip(batch, self.pool.run_jobs(l_jobs.naive_job, [items[i] for i in batch])):
                done[i] = self.cache.get_or_compute("l_naive", inputs(i), lambda out=out: out, [params])
        return [done[i] for i in range(len(items))]
