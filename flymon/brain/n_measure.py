"""The measurement layer of spec appendix N. `NMeasurer` runs n_jobs through a FlyPool and caches every item under
`NCache`: h3_store.MeasureCache's content-addressed entries, but written through n_store's guard under results/n/
(MeasureCache's own write is guarded to results/m0d/).

Items:
- presentations per (edit, odour) over a seed block;
- KC vectors per (edit, odour);
- the punish-only oracle per pair;
- N2's arm per (condition, seed, plastic). This is N.8.7's checkpoint: an interrupted judge loses at most one round of
  one item per worker, and a rerun reads every finished item without starting a job.
Every key carries the Params, the edit, the odour strengths, the seeds and the windows, so an APL-on and an APL-block
result can never share an entry."""
from __future__ import annotations

import json

from . import n_jobs, n_store
from .h3_store import MEASURE_FILES as H3_MEASURE_FILES, MeasureCache, canonical

MEASURE_FILES = tuple(dict.fromkeys(H3_MEASURE_FILES + (
    "flymon/brain/plasticity.py", "flymon/brain/presentation.py", "flymon/brain/h4_jobs.py",
    "flymon/brain/h4_formula.py", "flymon/brain/odor_real.py", "flymon/brain/n_jobs.py", "flymon/brain/n_measure.py",
    "data/odor/hallem2006_subset.csv", "data/odor/door_mappings_subset.csv")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + (
    "flymon/brain/n_spec.py", "flymon/brain/n_store.py", "flymon/brain/n_rules.py", "flymon/brain/n_oc.py",
    "flymon/brain/n_cli.py", "scripts/fetch_door_hallem.py", "scripts/run_n0f.py", "scripts/run_n0.py",
    "scripts/run_n1.py", "scripts/run_n2_pilot.py", "scripts/run_n2_judge.py", "scripts/write_n_notes.py")))


class NCache(MeasureCache):
    """MeasureCache with n_store's guarded write (results/n/ only)."""

    def get_or_compute(self, kind: str, inputs: dict, compute, params_list):
        k = self.key(kind, inputs)
        path = self.root / kind / f"{k[:24]}.json"
        self.used[str(path)] = k
        if path.exists():
            try:
                d = json.loads(path.read_text())
                if d.get("key") == k:
                    self.hits += 1
                    return d["result"]
            except (OSError, ValueError):
                pass
        result = compute()
        n_store.write_json(path, dict(key=k, kind=kind, run_id=self.run_id, inputs=json.loads(canonical(inputs)),
                                      result=result), params_list)
        self.misses += 1
        return json.loads(canonical(result))


class _Missing(Exception):
    pass


def _missing():
    raise _Missing


class NMeasurer:
    def __init__(self, pool, spec, cache):
        self.pool, self.spec, self.cache = pool, spec, cache
        self.params_seen: list = []

    def _items(self, kind, fn, params, items: list, keys: list) -> list:
        """One cache entry per item (keys[i] are its inputs), run in rounds of one item per worker."""
        if params not in self.params_seen:
            self.params_seen.append(params)
        done = {}
        for i, k in enumerate(keys):
            try:
                done[i] = self.cache.get_or_compute(kind, k, _missing, [params])
            except _Missing:
                pass
        todo = [i for i in range(len(items)) if i not in done]
        n = max(1, self.pool.n_workers)
        for r in range(0, len(todo), n):
            batch = todo[r:r + n]
            for i, out in zip(batch, self.pool.run_jobs(fn, [items[i] for i in batch])):
                done[i] = self.cache.get_or_compute(kind, keys[i], lambda out=out: out, [params])
        return [done[i] for i in range(len(items))]

    def _window(self) -> dict:
        h4 = self.spec.h4
        return dict(strength=self.spec.h3.strength, settle_ms=h4.oracle_window.settle_ms,
                    read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms))

    def _pres(self, kind, fn, params, items, seeds, readout) -> list:
        common = dict(params=params, seeds=tuple(int(s) for s in seeds), readout=dict(readout), **self._window())
        jobs = [dict(common, edit=e, odor=dict(o)) for e, o in items]
        return self._items(kind, fn, params, jobs, jobs)

    def presentations(self, params, items, seeds, readout) -> list:
        """items [(edit, odour)] -> one row list per item (presentation_job over `seeds`): scalars only."""
        return self._pres("n_pres", n_jobs.presentation_job, params, items, seeds, readout)

    def kc_vectors(self, params, items, seeds, readout) -> list:
        return self._pres("n_kcv", n_jobs.kc_vectors_job, params, items, seeds, readout)

    def punish_oracle(self, params, pairs, readout, z, punish_type) -> list:
        s = self.spec
        common = dict(params=params, readout=dict(readout), z={k: [float(v) for v in z[k]] for k in ("A", "P")},
                      act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds),
                      report_seeds=tuple(s.report_seeds), alphas=tuple(s.h4.oracle_alphas), punish_type=punish_type,
                      **self._window())
        jobs = [dict(common, odor_x=dict(p["odor_x"]), odor_y=dict(p["odor_y"])) for p in pairs]
        return self._items("n_oracle", n_jobs.punish_only_oracle_job, params, jobs, jobs)

    def arms(self, params, items, readout, punish_type) -> list:
        """items [{"cond", "edit", "odor_x", "odor_y", "seed", "plastic"}] -> absolute_arm_job rows with "cond"."""
        s, h4 = self.spec, self.spec.h4
        common = dict(params=params, readout=dict(readout), punish_type=punish_type, trials=int(h4.teach_trials),
                      present_ms=h4.teach_present_ms, gap_ms=h4.teach_gap_ms,
                      train_settle_ms=h4.teach_window.settle_ms, seed_base=s.train_seed_base,
                      seed_stride=s.train_seed_stride, **self._window())
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     plastic=bool(i["plastic"])) for i in items]
        rows = self._items("n_arm", n_jobs.absolute_arm_job, params, jobs, jobs)
        return [dict(r, cond=i["cond"]) for r, i in zip(rows, items)]
