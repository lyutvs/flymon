"""A process pool of CPU fly brains (spec 3.1, M0b).

W worker processes each hold one Engine + Plasticity per wiring variant (unshuffled, or one per C-shuf
seed) and serve requests for any fly. Membrane state is reset at every presentation, so a fly is only
its plastic KC->MBON weights, its enabled flag and its wiring variant; the parent keeps those (hive =
one weight vector per fly) and ships the weights with each request. Every worker function is
module-level so the spawn start method can pickle it.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import threading

import numpy as np

from .circuits import Populations, compartments, validate_populations
from .conditioning import Readout
from .config import Params
from .connectome import Connectome, shuffle_kc_mbon
from .engine_cpu import Engine
from .plasticity import Plasticity
from .presentation import decide, reinforce


@dataclasses.dataclass(frozen=True)
class FlySpec:
    enabled: bool = True              # plasticity on (C-off: False)
    shuffle_seed: int | None = None   # C-shuf wiring variant


_W = None   # per-process worker state, set by _init_worker


class _WorkerState:
    def __init__(self, npz: str, params: Params, punish_type: str, reward_type: str, max_variants: int):
        self.conn = Connectome.load(npz)
        self.pops = Populations.from_connectome(self.conn)
        self.params, self.punish_type, self.reward_type = params, punish_type, reward_type
        self.max_variants = int(max_variants)
        self.variants: dict = {}

    def get(self, shuffle_seed):
        key = None if shuffle_seed is None else int(shuffle_seed)
        if key not in self.variants:
            if len(self.variants) >= self.max_variants:
                raise ValueError(f"worker already holds {len(self.variants)} wiring variants (max_variants={self.max_variants}); "
                                 f"refusing to build variant {key!r} — each costs a CSC and an engine")
            conn = self.conn if key is None else shuffle_kc_mbon(self.conn, self.pops.kc, self.pops.mbon, key)
            comps = compartments(conn, self.pops, self.params.core_frac)
            validate_populations(conn, self.pops, comps, self.punish_type, self.reward_type)
            eng = Engine(conn, self.pops, self.params, seed=0)
            pl = Plasticity(eng, self.pops, comps)
            ro = Readout.from_compartments(comps, self.punish_type, self.reward_type)
            self.variants[key] = (eng, pl, comps, ro)
        return self.variants[key]


def _init_worker(npz: str, params: Params, punish_type: str, reward_type: str, max_variants: int) -> None:
    global _W
    _W = _WorkerState(npz, params, punish_type, reward_type, max_variants)


def _w0_job(shuffle_seed):
    _, pl, _, _ = _W.get(shuffle_seed)
    return pl.w0.copy()


def _decide_job(req: dict) -> np.ndarray:
    eng, pl, _, _ = _W.get(req["shuffle_seed"])
    eng.csc.w[pl.edges] = req["w"]
    return decide(eng, pl, _W.pops, req["candidates"], req["strength"], req["seed"],
                  req["settle_ms"], req["read_ms"], req["idx"])


def _reinforce_job(req: dict) -> np.ndarray:
    eng, pl, _, _ = _W.get(req["shuffle_seed"])
    eng.csc.w[pl.edges] = req["w"]
    reinforce(eng, pl, _W.pops, req["odor"], req["strength"], req["dan"], req["pulse_ms"], req["seed"],
              req["settle_ms"], req["gap_ms"], req["enabled"])
    return eng.csc.w[pl.edges].copy()


def _job(args):
    fn, shuffle_seed, kw = args
    eng, pl, comps, ro = _W.get(shuffle_seed)
    return fn(eng, pl, _W.pops, comps, ro, **kw)


class FlyPool:
    """W worker processes serving any fly; the parent holds one plastic weight vector per fly.

    Every public call is serialized by an internal lock; from asyncio, run them in a single executor
    thread (`loop.run_in_executor`) so the event loop never blocks on a 10-second batch.
    """

    def __init__(self, npz, params: Params, flies, workers: int = 16, punish_type: str = "PPL105",
                 reward_type: str = "PAM08", timeout_s: float = 3600.0, max_variants: int = 4):
        self.flies = [f if isinstance(f, FlySpec) else FlySpec(**f) for f in flies]
        if not self.flies:
            raise ValueError("FlyPool needs at least one fly")
        variants = sorted({f.shuffle_seed for f in self.flies}, key=lambda v: (v is not None, v if v is not None else 0))
        if len(variants) > max_variants:
            raise ValueError(f"{len(variants)} wiring variants requested but max_variants={max_variants}")
        # Validate the data in the parent first: a worker that dies in its initializer is respawned by
        # multiprocessing.Pool forever, so every data error has to surface here, before any spawn.
        conn = Connectome.load(str(npz))
        pops = Populations.from_connectome(conn)
        validate_populations(conn, pops, compartments(conn, pops, params.core_frac), punish_type, reward_type)
        del conn, pops
        self._lock = threading.Lock()
        self.timeout_s = float(timeout_s)
        self.n_workers = max(1, min(int(workers), len(self.flies)))
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.n_workers, initializer=_init_worker,
                             initargs=(str(npz), params, punish_type, reward_type, int(max_variants)))
        try:
            self.w0 = dict(zip(variants, self._map(_w0_job, variants)))
        except BaseException:
            self.pool.terminate()
            self.pool.join()
            raise
        self.w = {i: self.w0[f.shuffle_seed].copy() for i, f in enumerate(self.flies)}

    # ---- plumbing --------------------------------------------------------------------------
    def _map(self, fn, items):
        if not items:
            return []
        with self._lock:
            return self.pool.map_async(fn, items, chunksize=1).get(self.timeout_s)

    @property
    def n_flies(self) -> int:
        return len(self.flies)

    # ---- the two presentations (spec 2 steps 3 and 5) ----------------------------------------
    def decide_batch(self, requests, strength: float, settle_ms: float = 800.0, read_ms: float = 600.0, idx=None):
        """requests: [(fly, candidates, seed)] -> [counts[n_candidates, N or len(idx)]] in request order."""
        sel = None if idx is None else np.asarray(idx, np.int64)
        jobs = [dict(w=self.w[f], shuffle_seed=self.flies[f].shuffle_seed, candidates=list(c), strength=strength,
                     seed=int(s), settle_ms=settle_ms, read_ms=read_ms, idx=sel) for f, c, s in requests]
        return self._map(_decide_job, jobs)

    def reinforce_batch(self, requests, strength: float, settle_ms: float = 800.0, gap_ms: float = 200.0) -> None:
        """requests: [(fly, odor, dan_or_None, pulse_ms, seed)]; updates the weights of every listed fly.
        A fly may appear once per batch: two reinforcements of one fly would both start from the same
        weights and the second would silently overwrite the first."""
        ids = [int(f) for f, *_ in requests]
        dup = sorted({f for f in ids if ids.count(f) > 1})
        if dup:
            raise ValueError(f"reinforce_batch: fly ids {dup} appear more than once in one batch")
        jobs = [dict(w=self.w[f], shuffle_seed=self.flies[f].shuffle_seed, odor=o, strength=strength, dan=d,
                     pulse_ms=float(pm), seed=int(s), settle_ms=settle_ms, gap_ms=gap_ms, enabled=self.flies[f].enabled)
                for f, o, d, pm, s in requests]
        new = self._map(_reinforce_job, jobs)      # _map takes the lock and releases it; it is not re-entrant
        with self._lock:
            for (f, *_), w in zip(requests, new):
                self.w[f] = w

    # ---- measurements and protocols on the workers --------------------------------------------
    def run_jobs(self, fn, kwargs_list, shuffle_seed=None):
        """fn(engine, plasticity, pops, comps, readout, **kwargs) on any worker; fn must be module-level.
        Results in order. The worker's weights are scratch here: reset them inside fn if it learns."""
        return self._map(_job, [(fn, shuffle_seed, dict(kw)) for kw in kwargs_list])

    # ---- per-fly state -----------------------------------------------------------------------
    def set_enabled(self, fly: int, on: bool) -> None:
        self.flies[fly] = dataclasses.replace(self.flies[fly], enabled=bool(on))

    def weights_frac(self, fly: int) -> float:
        return float(np.mean(self.w[fly] / self.w0[self.flies[fly].shuffle_seed]))

    def state(self) -> dict:
        return {"flies": [dict(enabled=f.enabled, shuffle_seed=f.shuffle_seed, w=self.w[i].copy())
                          for i, f in enumerate(self.flies)]}

    def load_state(self, d: dict) -> None:
        entries = d["flies"]
        if len(entries) != len(self.flies):
            raise ValueError(f"state has {len(entries)} flies, pool has {len(self.flies)}")
        for i, e in enumerate(entries):
            if e["shuffle_seed"] != self.flies[i].shuffle_seed:
                raise ValueError(f"fly {i}: wiring variant {e['shuffle_seed']} != pool's {self.flies[i].shuffle_seed}")
            w = np.asarray(e["w"], np.float32)
            if w.shape != self.w0[self.flies[i].shuffle_seed].shape:
                raise ValueError(f"fly {i}: weight vector shape {w.shape} != {self.w0[self.flies[i].shuffle_seed].shape}")
            if not np.isfinite(w).all() or (w < 0).any():
                raise ValueError(f"fly {i}: weights must be finite and >= 0")
        for i, e in enumerate(entries):
            self.w[i] = np.asarray(e["w"], np.float32).copy()
            self.flies[i] = dataclasses.replace(self.flies[i], enabled=bool(e["enabled"]))

    # ---- lifecycle ---------------------------------------------------------------------------
    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def terminate(self) -> None:
        self.pool.terminate()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        (self.terminate if exc_type else self.close)()
