"""The measurement layer of the M0d H.4 runner: the reference and rest measurements come from H.3's measurer and cache
(the H.3 guard's own entries), the conditioning arms and the oracle from H.4's cache. The procedure (h4_runner) only
sees this interface, so its tests substitute a scripted measurer and never run the engine.

Resume (spec H.3a.12's form, carried over): every measurement is cached under a content key — `h3_store.MeasureCache`
keyed by the files a measurement's result depends on (`MEASURE_FILES`), the NPZ, the versions and the inputs. The
oracle is cached per pair and run in rounds of one pair per worker, so an interrupted run loses at most one round.
"""
from __future__ import annotations

from . import h4_jobs
from .h3_store import MEASURE_FILES as H3_MEASURE_FILES

MEASURE_FILES = ("uv.lock", "flymon/brain/circuits.py", "flymon/brain/config.py", "flymon/brain/connectome.py",
                 "flymon/brain/engine_cpu.py", "flymon/brain/fly_pool.py", "flymon/brain/stimuli.py",
                 "flymon/brain/thresholds.py", "flymon/brain/plasticity.py", "flymon/brain/presentation.py",
                 "flymon/brain/conditioning.py", "flymon/brain/h3_store.py", "flymon/brain/h4_formula.py",
                 "flymon/brain/h4_jobs.py", "flymon/brain/h4_measure.py")
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + H3_MEASURE_FILES + (
    "flymon/brain/h3_rules.py", "flymon/brain/h3_spec.py", "flymon/brain/pool_bench.py", "flymon/brain/h4_spec.py",
    "flymon/brain/h4_pairs.py", "flymon/brain/h4_rules.py", "flymon/brain/h4_runner.py", "scripts/run_m0d_h4.py")))


class _Missing(Exception):
    pass


def _missing(*_):
    raise _Missing


class H4Measurer:
    def __init__(self, pool, spec, pairs: list, pools: dict, cache, h3_measurer):
        """pairs: h4_pairs.even_pairs; pools: {"A": [types], "P": [types]}; h3_measurer: h3_measure.PoolMeasurer on
        H.3's cache (reference / rest)."""
        self.pool, self.spec, self.pairs, self.pools = pool, spec, pairs, pools
        self.cache, self.h3 = cache, h3_measurer
        self.types = [t for k in ("A", "P") for t in pools[k]]
        self.params_seen: list = []

    def _seen(self, params):
        if params not in self.params_seen:
            self.params_seen.append(params)

    def reference(self, params) -> list:
        self._seen(params)
        return self.h3.reference(params)

    def rest(self, params, seeds) -> list:
        self._seen(params)
        return self.h3.rest(params, seeds)

    def teach(self, params) -> list:
        """Both single-channel arms, every declared odour order and seed: one cache entry per combination."""
        s, h3 = self.spec, self.spec.h3
        self._seen(params)
        common = dict(params=params, types=tuple(self.types), punish_type=h3.punish_type, reward_type=h3.reward_type,
                      k=h3.design_k, odor_seed=h3.design_odor_seed, strength=h3.strength, trials=s.teach_trials,
                      present_ms=s.teach_present_ms, gap_ms=s.teach_gap_ms, settle_ms=s.teach_window.settle_ms,
                      read_ms=s.teach_window.read_ms)
        jobs = [dict(common, seed=int(seed), arm=arm, order=order) for order in s.teach_orders
                for arm in ("punish_only", "reward_only") for seed in s.teach_seeds]
        inputs = dict(common, jobs=[(j["seed"], j["arm"], j["order"]) for j in jobs])
        return self.cache.get_or_compute("teach", inputs, lambda: self.pool.run_jobs(h4_jobs.teach_job, jobs), [params])

    def oracle(self, params, readout: dict, z: dict) -> list:
        """One row per pair ({axis, turn, x, y} + oracle_job's result), in pair order."""
        s, h3 = self.spec, self.spec.h3
        self._seen(params)
        common = dict(params=params, readout=dict(readout), z={k: tuple(v) for k, v in z.items()},
                      types=tuple(self.types), act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds),
                      report_seeds=tuple(s.report_seeds), alphas=tuple(s.oracle_alphas), strength=h3.strength,
                      settle_ms=s.oracle_window.settle_ms, read_ms=s.oracle_window.read_ms, window_ms=s.kc_window_ms,
                      punish_type=h3.punish_type, reward_type=h3.reward_type)
        items = [(p, dict(common, odor_x=p["odor_x"], odor_y=p["odor_y"])) for p in self.pairs]
        inputs = lambda p, kw: dict(kw, pair=[p["axis"], int(p["turn"]), p["x"], p["y"]])
        done = {}
        for i, (p, kw) in enumerate(items):
            try:
                done[i] = self.cache.get_or_compute("oracle", inputs(p, kw), _missing, [params])
            except _Missing:
                pass
        todo = [i for i in range(len(items)) if i not in done]
        n = max(1, self.pool.n_workers)
        for r in range(0, len(todo), n):
            batch = todo[r:r + n]
            res = self.pool.run_jobs(h4_jobs.oracle_job, [items[i][1] for i in batch])
            for i, out in zip(batch, res):
                done[i] = self.cache.get_or_compute("oracle", inputs(*items[i]), lambda out=out: out, [params])
        return [done[i] | {k: items[i][0][k] for k in ("axis", "turn", "x", "y")} for i in range(len(items))]
