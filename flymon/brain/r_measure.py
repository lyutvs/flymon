# flymon/brain/r_measure.py
"""R's measurer (R.2, R.3, R.5, R.9.6): the three R jobs behind RCache, one atomic entry per unit, written as soon as
its worker round returns, so an interrupted stage resumes with only the missing units and never returns a partial
list.
- activity: kc_activity_job over (odour, seed) items in jobs of e_measure.MAX_ITEMS, one "r_act" entry per odour
  (EMeasurer.activity's batching and frac = active KCs / n_kc, with the edit, its sha and edge count added).
- oracle: r_oracle_job, one "r_oracle" entry per (pair, condition, block); the kwargs are the encoder's oracle inputs
  (EMeasurer._oracle_kw, test) plus edit, fixed_alphas () and active_fx; the condition picks the odours (E-grid / E0)
  and the strength.
- arms: r_arm_job, one "r_arm" entry per (direction, arm, seed); the kwargs are PMeasurer.p_arms' (test) plus p_type."""
from __future__ import annotations

import sys
import time

from ..agent.e_measure import MAX_ITEMS
from . import p_measure, r_jobs
from .q_measure import Q_MEASURE_FILES
from .r_pairs import cond_odours, row_key

# The cache code key: everything the jobs' results depend on — Q's list (the encoder's engine files, q_jobs, o_jobs,
# n_jobs, n_spec), P's (o_jobs' training, n_measure's windows), and R's jobs, measurer and store.
R_MEASURE_FILES = tuple(dict.fromkeys(Q_MEASURE_FILES + tuple(p_measure.MEASURE_FILES) + (
    "flymon/brain/r_jobs.py", "flymon/brain/r_measure.py", "flymon/brain/r_store.py")))


def _odour(o: dict) -> dict:
    return {str(g): float(v) for g, v in o.items()}


class RMeasurer:
    def __init__(self, pool, cache, spec, params, readout: dict, z: dict, types, n_kc: int):
        self.pool, self.cache, self.spec, self.params = pool, cache, spec, params
        self.readout, self.z, self.types, self.n_kc = dict(readout), dict(z), list(types), int(n_kc)
        self.last_wall_s, self.last_jobs = 0.0, 0

    def _windows(self) -> dict:
        sp = self.spec
        return dict(settle_ms=float(sp.settle_ms), read_ms=float(sp.read_ms), window_ms=int(sp.window_ms))

    # ---- KC activity (gate ①, repro, smoke) -------------------------------------------------------------------------
    def _act_inputs(self, oid, odour, edit, s, seeds, block) -> dict:
        return dict(params=self.params, edit=edit, p_type=self.spec.p_type, odour_id=str(oid), odour=_odour(odour),
                    strength=float(s), seeds=[int(x) for x in seeds], block=block, **self._windows())

    def activity(self, odours: dict, edit: str, s: float, seeds, block: str) -> dict:
        seeds = [int(x) for x in seeds]
        ins = {oid: self._act_inputs(oid, o, edit, s, seeds, block) for oid, o in odours.items()}
        todo = [oid for oid in odours if self.cache.get("r_act", ins[oid]) is None]
        t0 = time.perf_counter()
        if todo:
            print(f"r activity {block}: {len(todo)}/{len(odours)} odours to measure", file=sys.stderr)
            entries = [(oid, odours[oid]) for oid in todo]
            items = [(k * len(seeds) + j, _odour(o), sd) for k, (_, o) in enumerate(entries) for j, sd in enumerate(seeds)]
            jobs = [items[a:a + MAX_ITEMS] for a in range(0, len(items), MAX_ITEMS)]
            common = dict(params=self.params, edit=edit, p_type=self.spec.p_type, strength=float(s), **self._windows())
            got, done = {}, set()
            n_w = max(1, int(self.pool.n_workers))
            for a in range(0, len(jobs), n_w):
                for part in self.pool.run_jobs(r_jobs.kc_activity_job, [dict(common, items=c) for c in jobs[a:a + n_w]]):
                    for r in part:
                        got[int(r["i"])] = r
                for k, (oid, _) in enumerate(entries):
                    idx = range(k * len(seeds), (k + 1) * len(seeds))
                    if k not in done and all(i in got for i in idx):
                        done.add(k)
                        rs = [got[i] for i in idx]
                        self.cache.put("r_act", ins[oid], dict(
                            frac=[len(r["kc"]) / self.n_kc for r in rs], max_win=[int(r["max_win"]) for r in rs],
                            csc_sha256=sorted({r["csc_sha256"] for r in rs}),
                            edit_edges=sorted({int(r["edit_edges"]) for r in rs})), [self.params])
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = {}
        for oid in odours:
            v = self.cache.get("r_act", ins[oid])
            if v is None:
                raise RuntimeError(f"r activity {block}: odour {oid} missing after the run")
            out[oid] = v
        return out

    # ---- the oracle (repro, smoke, even, judgement) --------------------------------------------------------------
    def kwargs(self, row: dict, cond, seeds: dict) -> dict:
        sp = self.spec
        ox, oy = cond_odours(row, cond)
        return dict(params=self.params, odor_x=_odour(ox), odor_y=_odour(oy), readout=dict(self.readout),
                    z={k: [float(v) for v in self.z[k]] for k in self.z}, types=[str(t) for t in self.types],
                    act_seeds=[int(s) for s in seeds["act"]], select_seeds=[int(s) for s in seeds["select"]],
                    report_seeds=[int(s) for s in seeds["report"]], alphas=[float(a) for a in sp.alphas],
                    strength=float(cond.strength), **self._windows(), punish_type=sp.punish_type,
                    reward_type=sp.reward_type, edit=cond.edit, fixed_alphas=[float(a) for a in sp.fixed_alphas],
                    active_fx=float(sp.active_fx))

    def inputs(self, row: dict, cond, block: str, seeds: dict) -> dict:
        return dict(self.kwargs(row, cond, seeds), pair=row_key(row), condition=cond.name, block=block)

    def oracle(self, rows: list, cond, block: str, seeds: dict) -> list:
        ins = [self.inputs(r, cond, block, seeds) for r in rows]
        todo = [i for i, x in enumerate(ins) if self.cache.get("r_oracle", x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        t0 = time.perf_counter()
        if todo:
            print(f"r oracle {cond.name}/{block}: {len(todo)}/{len(rows)} pairs to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            res = self.pool.run_jobs(r_jobs.r_oracle_job, [self.kwargs(rows[i], cond, seeds) for i in batch])
            for i, r in zip(batch, res):
                self.cache.put("r_oracle", ins[i], r, [self.params])
            print(f"r oracle {cond.name}/{block}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for r, x in zip(rows, ins):
            got = self.cache.get("r_oracle", x)
            if got is None:
                raise RuntimeError(f"r oracle {cond.name}/{block}: pair {row_key(r)} missing after the run")
            out.append(dict(key=row_key(r), result=got, cache_key=self.cache.key("r_oracle", x),
                            cache_file=str(self.cache._path("r_oracle", x))))
        return out

    # ---- P arms (repro, smoke, gate ②) ---------------------------------------------------------------------------
    def arm_common(self, nspec, readout: dict, punish_type: str) -> dict:
        """PMeasurer.p_arms' common kwargs (N2's training timings and seed rule, NMeasurer._window) plus p_type."""
        h4 = nspec.h4
        return dict(params=self.params, readout=dict(readout), punish_type=punish_type,
                    reward_type=nspec.h3.reward_type, trials=int(h4.teach_trials), present_ms=h4.teach_present_ms,
                    gap_ms=h4.teach_gap_ms, train_settle_ms=h4.teach_window.settle_ms,
                    seed_base=nspec.train_seed_base, seed_stride=nspec.train_seed_stride,
                    strength=nspec.h3.strength, settle_ms=h4.oracle_window.settle_ms,
                    read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms), p_type=self.spec.p_type)

    def arms(self, items: list, readout: dict, punish_type: str, block: str, nspec) -> list:
        common = self.arm_common(nspec, readout, punish_type)
        jobs = [dict(common, edit=i["edit"], odor_x=dict(i["odor_x"]), odor_y=dict(i["odor_y"]), seed=int(i["seed"]),
                     arm=i["arm"], punish=bool(i["punish"]), plastic=bool(i["plastic"]), da_zero=bool(i["da_zero"]))
                for i in items]
        ins = [dict(j, block=block, direction=i["direction"], x=i["x"], y=i["y"],
                    point=[float(v) for v in i["point"]]) for j, i in zip(jobs, items)]
        todo = [k for k, x in enumerate(ins) if self.cache.get("r_arm", x) is None]
        n_w = max(1, int(self.pool.n_workers)) if todo else 1
        t0 = time.perf_counter()
        if todo:
            print(f"r arms {block}: {len(todo)}/{len(items)} to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            for k, r in zip(batch, self.pool.run_jobs(r_jobs.r_arm_job, [jobs[k] for k in batch])):
                self.cache.put("r_arm", ins[k], r, [self.params])
        self.last_wall_s, self.last_jobs = time.perf_counter() - t0, len(todo)
        out = []
        for x, i in zip(ins, items):
            got = self.cache.get("r_arm", x)
            if got is None:
                raise RuntimeError(f"r arms {block}: {i['direction']}/{i['arm']}/{i['seed']} missing after the run")
            out.append(dict(got, direction=i["direction"], x=i["x"], y=i["y"], point=[float(v) for v in i["point"]]))
        return out
