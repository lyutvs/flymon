"""The encoder track's measurer (spec 4.1, 4.3, 4.4): FlyPool jobs from the brain package (k_jobs.activity_job,
h4_jobs.oracle_job), unchanged, behind the track's content-key cache (e_store.ECache).

Resume: every entry is written as soon as its last item is back from a worker round, so an interrupted run loses at
most one round. Every cache key carries the engine Params. Cache kinds: "drive" (one per glomerulus), "activity" (one per odour, strength and seed batch),
"oracle" (one per pair)."""
from __future__ import annotations

import sys

from ..brain import h4_jobs, k_jobs
from .e_store import ECache

MAX_ITEMS = 64
PUNISH_TYPE = "PPL105"
REWARD_TYPE = "PAM08"


def _odour(o: dict) -> dict:
    return {str(g): float(v) for g, v in o.items()}


class EMeasurer:
    def __init__(self, pool, cache: ECache, params, n_kc: int, spec, guard_params: bool = True):
        self.pool, self.cache, self.params, self.n_kc, self.spec = pool, cache, params, int(n_kc), spec
        self.params_list = [params] if guard_params else []

    # ---- shared --------------------------------------------------------------------------------
    def _windows(self) -> dict:
        sp = self.spec
        return dict(settle_ms=float(sp.settle_ms), read_ms=float(sp.read_ms), window_ms=int(sp.window_ms))

    def _run_items(self, entries: list, strength: float, seeds, finish) -> None:
        """entries: [(entry_id, odour)]; every odour runs on every seed through activity_job in jobs of <= MAX_ITEMS
        items, pool.n_workers jobs per round. finish(entry_id, rows in seed order) is called once all of an entry's
        rows are back (after the round that completes it)."""
        seeds = [int(s) for s in seeds]
        items = [(k * len(seeds) + j, _odour(o), s) for k, (_, o) in enumerate(entries) for j, s in enumerate(seeds)]
        jobs = [items[a:a + MAX_ITEMS] for a in range(0, len(items), MAX_ITEMS)]
        common = dict(params=self.params, strength=float(strength), **self._windows())
        got: dict = {}
        done = set()
        n_w = max(1, int(self.pool.n_workers))
        for a in range(0, len(jobs), n_w):
            parts = self.pool.run_jobs(k_jobs.activity_job, [dict(common, items=c) for c in jobs[a:a + n_w]])
            for part in parts:
                for r in part:
                    got[int(r["i"])] = r
            for k, (eid, _) in enumerate(entries):
                idx = range(k * len(seeds), (k + 1) * len(seeds))
                if k not in done and all(i in got for i in idx):
                    done.add(k)
                    finish(eid, [got[i] for i in idx])
        missing = [eid for k, (eid, _) in enumerate(entries) if k not in done]
        if missing:
            raise RuntimeError(f"activity_job returned no rows for {missing}")

    # ---- drive (4.1) ---------------------------------------------------------------------------
    def _drive_inputs(self, g: str, odour: dict) -> dict:
        return dict(params=self.params, glomerulus=g, odour=odour, strength=float(self.spec.drive_strength),
                    seeds=[int(s) for s in self.spec.drive_seeds], **self._windows())

    def drive(self, glomeruli: list, c_norm: float, receptor_counts: dict) -> dict:
        """Single-glomerulus odour {g: c_norm / receptor_counts[g]} at spec.drive_strength over spec.drive_seeds:
        mean total read-window KC spikes and mean KC active fraction over seeds."""
        odours = {str(g): {str(g): float(c_norm) / float(receptor_counts[g])} for g in glomeruli}
        todo = [(g, o) for g, o in odours.items() if self.cache.get("drive", self._drive_inputs(g, o)) is None]

        def finish(g, rows):
            res = dict(mean_spikes=sum(sum(r["n"]) for r in rows) / len(rows),
                       kc_frac_mean=sum(len(r["kc"]) / self.n_kc for r in rows) / len(rows))
            self.cache.put("drive", self._drive_inputs(g, odours[g]), res, self.params_list)

        if todo:
            print(f"drive: {len(todo)}/{len(odours)} glomeruli to measure", file=sys.stderr)
            self._run_items(todo, self.spec.drive_strength, self.spec.drive_seeds, finish)
        return {g: self.cache.get("drive", self._drive_inputs(g, o)) for g, o in odours.items()}

    # ---- activity (4.3) ------------------------------------------------------------------------
    def _act_inputs(self, oid: str, odour: dict, s: float, seeds) -> dict:
        return dict(params=self.params, odour_id=str(oid), odour=_odour(odour), strength=float(s), seeds=[int(x) for x in seeds],
                    **self._windows())

    def activity(self, odours: dict, s: float, seeds) -> dict:
        """{odour_id: {frac: [KC active fraction per seed], max_win: [per seed]}}, seeds in the given order."""
        todo = [(oid, o) for oid, o in odours.items() if self.cache.get("activity", self._act_inputs(oid, o, s, seeds)) is None]

        def finish(oid, rows):
            res = dict(frac=[len(r["kc"]) / self.n_kc for r in rows], max_win=[int(r["max_win"]) for r in rows])
            self.cache.put("activity", self._act_inputs(oid, odours[oid], s, seeds), res, self.params_list)

        if todo:
            print(f"activity s={s}: {len(todo)}/{len(odours)} odours to measure", file=sys.stderr)
            self._run_items(todo, s, seeds, finish)
        return {oid: self.cache.get("activity", self._act_inputs(oid, o, s, seeds)) for oid, o in odours.items()}

    # ---- oracle (4.4, 5.x) ---------------------------------------------------------------------
    def _oracle_kw(self, row: dict, strength: float, readout: dict, z: dict, types, seeds: dict) -> dict:
        """Both the cache inputs and the oracle_job kwargs (params included)."""
        return dict(params=self.params, odor_x=_odour(row["odor_x"]), odor_y=_odour(row["odor_y"]), readout=dict(readout),
                    z={k: [float(v) for v in z[k]] for k in z}, types=[str(t) for t in types],
                    act_seeds=[int(s) for s in seeds["act"]], select_seeds=[int(s) for s in seeds["select"]],
                    report_seeds=[int(s) for s in seeds["report"]], alphas=[float(a) for a in self.spec.alphas],
                    strength=float(strength), **self._windows(), punish_type=PUNISH_TYPE, reward_type=REWARD_TYPE)

    def oracle(self, rows: list, strength: float, readout: dict, z: dict, types, seeds: dict, tag: str) -> list:
        """[row + oracle result] in row order; one cache entry per pair; missing pairs run in rounds of
        pool.n_workers, each round's results cached before the next. Never returns a partial list."""
        kws = [self._oracle_kw(r, strength, readout, z, types, seeds) for r in rows]
        todo = [k for k, kw in enumerate(kws) if self.cache.get("oracle", kw) is None]
        n_w = max(1, int(self.pool.n_workers))
        if todo:
            print(f"oracle {tag}: {len(todo)}/{len(rows)} pairs to measure", file=sys.stderr)
        for a in range(0, len(todo), n_w):
            batch = todo[a:a + n_w]
            res = self.pool.run_jobs(h4_jobs.oracle_job, [kws[k] for k in batch])
            for k, r in zip(batch, res):
                self.cache.put("oracle", kws[k], r, self.params_list)
            print(f"oracle {tag}: {min(a + n_w, len(todo))}/{len(todo)}", file=sys.stderr)
        out = []
        for row, kw in zip(rows, kws):
            r = self.cache.get("oracle", kw)
            if r is None:
                raise RuntimeError(f"oracle {tag}: pair {row.get('x')}/{row.get('y')} missing after the run")
            clash = sorted(set(row) & set(r))
            if clash:
                raise ValueError(f"oracle {tag}: result keys {clash} collide with row fields")
            out.append({**row, **r})
        return out
