"""FlyPool on the combined-lever brain L_V (spec appendix AC.2, AC.7 1): every worker builds Engine -> apply_u_edit
(in place on the CSC) -> Plasticity -> Readout, u_measure.u_rig's order, so the plastic KC->MBON edges and w0 are the
edited engine's. fly_pool.py is a measurement-key file (h3_store / h4_measure hash it) and is not edited: the workers
run fly_pool's own job functions on a _WorkerState subclass installed as fly_pool._W (those functions read the module
global at call time)."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import threading

import numpy as np

from . import fly_pool
from .circuits import Populations, compartments, validate_populations
from .conditioning import Readout
from .config import Params
from .connectome import Connectome, shuffle_kc_mbon
from .engine_cpu import Engine
from .fly_pool import FlyPool, FlySpec
from .plasticity import Plasticity
from .q_jobs import NONE
from .u_measure import apply_u_edit, parse_u_edit


class MaskPlasticity(Plasticity):
    """Plasticity whose per-step membership tests use precomputed boolean masks (is_kc[fired], is_type[fired].sum())
    instead of np.isin (speedup brief fix 1; plasticity.py is a measurement-key file and is not edited). Same fired
    selection in the same order, the same integer counts and the same float expressions in the same order, so every
    trace and weight is bit-identical to Plasticity's."""

    def __init__(self, engine: Engine, pops: Populations, comps: dict):
        super().__init__(engine, pops, comps)
        self._is_kc = np.zeros(engine.N, bool); self._is_kc[pops.kc] = True
        self._type_masks = []
        for cells, wvec in self.types.values():
            m = np.zeros(engine.N, bool); m[cells] = True
            self._type_masks.append((m, wvec, len(cells)))

    # copied from flymon/brain/plasticity.py:Plasticity.on_step (np.isin -> boolean masks)
    def on_step(self, engine: Engine, fired: np.ndarray) -> None:
        p, dt = self.p, self.p.dt
        self.kc_trace *= (1.0 - dt / p.kc_trace_ms)
        kf = self.pops.kc
        fired_kc = fired[self._is_kc[fired]]
        if fired_kc.size:
            self.kc_trace[np.searchsorted(kf, fired_kc)] += dt / p.kc_trace_ms
        self.da *= (1.0 - dt / p.da_trace_ms)
        for mask, wvec, n_cells in self._type_masks:
            n = int(mask[fired].sum())
            if n:
                self.da += wvec * (n / n_cells) * (dt / p.da_trace_ms)
        self.da_base += (self.da - self.da_base) * (dt / p.da_baseline_ms)
        if not self.enabled:
            return
        phasic = np.maximum(self.da - self.da_base, 0.0)
        coincide = (self.kc_trace[self.pre_kc] * p.kc_trace_scale) * (phasic[self.post_mb] * p.da_trace_scale)
        if coincide.any():
            w = self.eng.csc.w
            # w[self.edges] is a fancy-index copy: depress and floor it, then write the block back
            we = w[self.edges] * (1.0 - p.learn_rate * np.tanh(coincide)).astype(np.float32)
            np.maximum(we, self.w0 * p.min_weight_frac, out=we)
            w[self.edges] = we


class _LeverWorkerState(fly_pool._WorkerState):
    def __init__(self, npz, params, punish_type, reward_type, max_variants, edit: str, p_type: str,
                 mask_plasticity: bool = True):
        super().__init__(npz, params, punish_type, reward_type, max_variants)
        self.edit, self.p_type = edit, p_type
        self.plasticity_cls = MaskPlasticity if mask_plasticity else Plasticity
        self.edit_info: dict = {}

    # copied from flymon/brain/fly_pool.py:_WorkerState.get (+ apply_u_edit before Plasticity; MaskPlasticity)
    def get(self, shuffle_seed):
        key = None if shuffle_seed is None else int(shuffle_seed)
        if key not in self.variants:
            if len(self.variants) >= self.max_variants:
                raise ValueError(f"worker already holds {len(self.variants)} wiring variants "
                                 f"(max_variants={self.max_variants}); refusing to build variant {key!r}")
            conn = self.conn if key is None else shuffle_kc_mbon(self.conn, self.pops.kc, self.pops.mbon, key)
            comps = compartments(conn, self.pops, self.params.core_frac)
            validate_populations(conn, self.pops, comps, self.punish_type, self.reward_type)
            eng = Engine(conn, self.pops, self.params, seed=0)
            sha, n_apl, blocks = apply_u_edit(eng, self.pops, self.edit, self.p_type)
            pl = self.plasticity_cls(eng, self.pops, comps)
            ro = Readout.from_compartments(comps, self.punish_type, self.reward_type)
            self.variants[key] = (eng, pl, comps, ro)
            self.edit_info[key] = dict(sha=sha, n_apl=n_apl, blocks=blocks)
        return self.variants[key]


def _init_lever_worker(npz, params, punish_type, reward_type, max_variants, edit, p_type,
                       mask_plasticity: bool = True) -> None:
    fly_pool._W = _LeverWorkerState(npz, params, punish_type, reward_type, max_variants, edit, p_type, mask_plasticity)


def lever_sha_job(eng, pl, pops, comps, ro) -> str:
    """sha256 of the worker's CSC weights with the plastic edges at w0 (= apply_u_edit's sha on a fresh engine)."""
    w = eng.csc.w.copy()
    w[pl.edges] = pl.w0
    return hashlib.sha256(w.tobytes()).hexdigest()


class _FlyWeights(dict):
    """The parent's per-fly weights (FlyPool.w) with a generation per fly. Any outside assignment w[f] = ... (rescope
    blocks' rollback of a failed / cancelled attempt, load_state) takes the pool lock and bumps the fly's generation;
    LeverFlyPool.reinforce_batch writes through _set_own under that lock only if the generation it read at submission
    is still current, so a reinforce still running in an executor thread cannot overwrite a rollback."""

    def __init__(self, lock, items):
        super().__init__(items)
        self._lock, self.gen = lock, {}

    def __setitem__(self, fly, w) -> None:
        with self._lock:
            self.gen[fly] = self.gen.get(fly, 0) + 1
            super().__setitem__(fly, w)

    def _set_own(self, fly, w) -> None:       # caller holds the lock
        super().__setitem__(fly, w)


class LeverFlyPool(FlyPool):
    """FlyPool whose workers carry the CSC edit `edit` (u_measure's string; "none" = no edit) on p_type.

    mask_plasticity (default on): workers run MaskPlasticity, bit-identical to Plasticity and cheaper per step;
    False = the plain Plasticity (the old path, kept for the equality tests).
    overlap (default on): _map waits for its jobs without holding the pool lock, so calls from several threads
    (LVSwarm's executor, one batch per thread) share the workers instead of queueing behind the slowest job of the
    batch before. Values are unchanged: each job carries its fly's weights at submission, reinforce_batch replaces
    (never mutates) self.w[f] under the lock, and a fly's own steps stay ordered because its player awaits each one.
    False = FlyPool's _map (lock held until the batch is done).
    reinforce_batch(expect, accept) (AD.5 2): a request created before a rollback is refused at submission and at write-back."""

    # copied from flymon/brain/fly_pool.py:FlyPool.__init__ (+ edit check, lever worker initializer)
    def __init__(self, npz, params: Params, flies, edit: str, p_type: str = "MBON05", workers: int = 16,
                 punish_type: str = "PPL105", reward_type: str = "PAM08", timeout_s: float = 3600.0,
                 max_variants: int = 4, mask_plasticity: bool = True, overlap: bool = True):
        if edit != NONE:
            parse_u_edit(edit)                                   # ValueError before any worker is spawned
        self.edit, self.p_type = edit, p_type
        self.overlap = bool(overlap)
        self.flies = [f if isinstance(f, FlySpec) else FlySpec(**f) for f in flies]
        if not self.flies:
            raise ValueError("LeverFlyPool needs at least one fly")
        variants = sorted({f.shuffle_seed for f in self.flies}, key=lambda v: (v is not None, v if v is not None else 0))
        if len(variants) > max_variants:
            raise ValueError(f"{len(variants)} wiring variants requested but max_variants={max_variants}")
        conn = Connectome.load(str(npz))                         # data errors surface here, not in a respawning worker
        pops = Populations.from_connectome(conn)
        validate_populations(conn, pops, compartments(conn, pops, params.core_frac), punish_type, reward_type)
        del conn, pops
        self._lock = threading.Lock()
        self.timeout_s = float(timeout_s)
        self.n_workers = max(1, min(int(workers), len(self.flies)))
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.n_workers, initializer=_init_lever_worker,
                             initargs=(str(npz), params, punish_type, reward_type, int(max_variants), edit, p_type,
                                       bool(mask_plasticity)))
        try:
            self.w0 = dict(zip(variants, self._map(fly_pool._w0_job, variants)))
        except BaseException:
            self.pool.terminate()
            self.pool.join()
            raise
        self.w = _FlyWeights(self._lock, {i: self.w0[f.shuffle_seed].copy() for i, f in enumerate(self.flies)})

    def _map(self, fn, items):
        if not self.overlap:
            return super()._map(fn, items)
        if not items:
            return []
        # timeout_s now also counts time queued behind other in-flight calls' jobs, not only this call's run time
        return self.pool.map_async(fn, items, chunksize=1).get(self.timeout_s)

    # copied from flymon/brain/fly_pool.py:FlyPool.reinforce_batch (+ stale results after a rollback are dropped;
    # AD.5 2: `expect` - the generation each request was created under - replaces the generation read here, and
    # `accept(k)`, the caller's attempt-token check, is asked at submission and again at write-back)
    def reinforce_batch(self, requests, strength: float, settle_ms: float = 800.0, gap_ms: float = 200.0,
                        expect=None, accept=None) -> list:
        ids = [int(f) for f, *_ in requests]
        dup = sorted({f for f in ids if ids.count(f) > 1})
        if dup:
            raise ValueError(f"reinforce_batch: fly ids {dup} appear more than once in one batch")
        if expect is not None and len(expect) != len(ids):
            raise ValueError(f"reinforce_batch: {len(expect)} expected generations for {len(ids)} requests")
        ok = (lambda k: True) if accept is None else accept
        with self._lock:
            gen = [self.w.gen.get(f, 0) if expect is None else int(expect[k]) for k, f in enumerate(ids)]
            live = [k for k, f in enumerate(ids) if self.w.gen.get(f, 0) == gen[k] and ok(k)]
            jobs = []
            for k in live:
                f, o, d, pm, s = requests[k]
                jobs.append(dict(w=self.w[ids[k]], shuffle_seed=self.flies[ids[k]].shuffle_seed, odor=o,
                                 strength=strength, dan=d, pulse_ms=float(pm), seed=int(s), settle_ms=settle_ms,
                                 gap_ms=gap_ms, enabled=self.flies[ids[k]].enabled))
        new = self._map(fly_pool._reinforce_job, jobs) if jobs else []   # waits outside the lock
        applied = [False] * len(ids)
        with self._lock:
            for k, w in zip(live, new):
                f = ids[k]
                if self.w.gen.get(f, 0) == gen[k] and ok(k):      # rolled back / reloaded / retired meanwhile: stale
                    self.w._set_own(f, w)
                    applied[k] = True
        return applied

    def lever_sha(self) -> str:
        return self.run_jobs(lever_sha_job, [{}])[0]

    def decide_batch(self, requests, strength: float, settle_ms: float = 800.0, read_ms: float = 600.0, idx=None):
        """FlyPool.decide_batch with one worker job per candidate (speedup brief fix 2), reassembled in request order:
        presentation.decide resets the engine from the same seed before every candidate (paired noise) and leaves the
        weights untouched, so decide([a, b, c]) == stack(decide([a]), decide([b]), decide([c])) bit for bit."""
        sel = None if idx is None else np.asarray(idx, np.int64)
        jobs, sizes = [], []
        for f, c, s in requests:
            c = list(c)
            if not c:
                raise ValueError("decide needs at least one candidate odour")
            sizes.append(len(c))
            jobs += [dict(w=self.w[f], shuffle_seed=self.flies[f].shuffle_seed, candidates=[o], strength=strength,
                          seed=int(s), settle_ms=settle_ms, read_ms=read_ms, idx=sel) for o in c]
        flat = self._map(fly_pool._decide_job, jobs)
        out, i = [], 0
        for n in sizes:
            out.append(np.concatenate(flat[i:i + n], axis=0))
            i += n
        return out
