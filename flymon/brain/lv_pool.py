"""FlyPool on the combined-lever brain L_V (spec appendix AC.2, AC.7 1): every worker builds Engine -> apply_u_edit
(in place on the CSC) -> Plasticity -> Readout, u_measure.u_rig's order, so the plastic KC->MBON edges and w0 are the
edited engine's. fly_pool.py is a measurement-key file (h3_store / h4_measure hash it) and is not edited: the workers
run fly_pool's own job functions on a _WorkerState subclass installed as fly_pool._W (those functions read the module
global at call time)."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import threading

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


class _LeverWorkerState(fly_pool._WorkerState):
    def __init__(self, npz, params, punish_type, reward_type, max_variants, edit: str, p_type: str):
        super().__init__(npz, params, punish_type, reward_type, max_variants)
        self.edit, self.p_type = edit, p_type
        self.edit_info: dict = {}

    # copied from flymon/brain/fly_pool.py:_WorkerState.get (+ apply_u_edit before Plasticity)
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
            pl = Plasticity(eng, self.pops, comps)
            ro = Readout.from_compartments(comps, self.punish_type, self.reward_type)
            self.variants[key] = (eng, pl, comps, ro)
            self.edit_info[key] = dict(sha=sha, n_apl=n_apl, blocks=blocks)
        return self.variants[key]


def _init_lever_worker(npz, params, punish_type, reward_type, max_variants, edit, p_type) -> None:
    fly_pool._W = _LeverWorkerState(npz, params, punish_type, reward_type, max_variants, edit, p_type)


def lever_sha_job(eng, pl, pops, comps, ro) -> str:
    """sha256 of the worker's CSC weights with the plastic edges at w0 (= apply_u_edit's sha on a fresh engine)."""
    w = eng.csc.w.copy()
    w[pl.edges] = pl.w0
    return hashlib.sha256(w.tobytes()).hexdigest()


class LeverFlyPool(FlyPool):
    """FlyPool whose workers carry the CSC edit `edit` (u_measure's string; "none" = no edit) on p_type."""

    # copied from flymon/brain/fly_pool.py:FlyPool.__init__ (+ edit check, lever worker initializer)
    def __init__(self, npz, params: Params, flies, edit: str, p_type: str = "MBON05", workers: int = 16,
                 punish_type: str = "PPL105", reward_type: str = "PAM08", timeout_s: float = 3600.0,
                 max_variants: int = 4):
        if edit != NONE:
            parse_u_edit(edit)                                   # ValueError before any worker is spawned
        self.edit, self.p_type = edit, p_type
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
                             initargs=(str(npz), params, punish_type, reward_type, int(max_variants), edit, p_type))
        try:
            self.w0 = dict(zip(variants, self._map(fly_pool._w0_job, variants)))
        except BaseException:
            self.pool.terminate()
            self.pool.join()
            raise
        self.w = {i: self.w0[f.shuffle_seed].copy() for i, f in enumerate(self.flies)}

    def lever_sha(self) -> str:
        return self.run_jobs(lever_sha_job, [{}])[0]
