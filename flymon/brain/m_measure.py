"""The measurement layer of spec appendix M (M.10.1, M.10.3). Group readouts are sums over a named cell list (a core
that is part of a type reads its core cells only): reactivity is H.4's rule (h3_rules.mbon_type_stats with
react_med_delta_min / react_zero_share_max) on the group sum, and z = (mean, sd) is valid only when finite with sd > 0.
Per-cell reactivity and the largest per-cell share of the group-sum variance are records. `MMeasurer` caches every
measurement under h3_store.MeasureCache keyed by MEASURE_FILES: reference / rest per odour / seed, pre_job per pair,
edit_job per pair and combination (the key carries the groups' cells, the readout, z and both DAN names), H.4's
teach_job per arm, order and seed — run in rounds of one item per worker, so an interrupted run loses one round."""
from __future__ import annotations

import numpy as np

from . import h4_jobs, m_jobs
from .h3_rules import mbon_type_stats
from .l_measure import HASHED_FILES as L_HASHED_FILES, MEASURE_FILES as L_MEASURE_FILES
from .m_jobs import group_counts

MEASURE_FILES = tuple(dict.fromkeys(L_MEASURE_FILES + ("flymon/brain/h3_jobs.py", "flymon/brain/h4_jobs.py",
                                                         "flymon/brain/m_jobs.py", "flymon/brain/m_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + L_HASHED_FILES + (
    "flymon/brain/m_spec.py", "flymon/brain/m_store.py", "flymon/brain/m_cands.py", "flymon/brain/m_rules.py",
    "flymon/brain/m_oc.py", "flymon/brain/m_cli.py", "scripts/run_m_spec.py", "scripts/run_m_stage0.py",
    "scripts/run_m_oc.py", "scripts/run_m_stage1.py", "scripts/run_m_stage2.py", "scripts/run_m_list.py",
    "scripts/run_m_stage3.py")))


def compose(pre: dict, ed: dict, cells, groups: dict) -> dict:
    """pre_job + edit_job -> oracle_job's row shape, counts summed per group (groups = {name: cells})."""
    gc = lambda counts: group_counts(counts, cells, groups)
    ro = ed["readout"]
    ap = lambda g: {"A": g[ro["A"]], "P": g[ro["P"]]}
    pre_rep, r1, r2 = gc(pre["pre_rep"]), gc(ed["R1_rep"]), gc(ed["R2_rep"])
    return {"select": {"pre": gc(pre["pre_sel"]),
                       "reward": {a: {"R1": gc(v["R"]), "change": v["change"]} for a, v in ed["reward"].items()},
                       "punish": {a: {"R2": gc(v["R"]), "change": v["change"]} for a, v in ed["punish"].items()}},
            "alpha_reward": ed["alpha_reward"], "alpha_punish": ed["alpha_punish"],
            "counts": {"pre": pre_rep, "R1": r1, "R2": r2},
            "report": {"pre": ap(pre_rep), "R1": ap(r1), "R2": ap(r2)}, "kc": pre["kc"]}


def _pos(cells) -> dict:
    return {int(c): k for k, c in enumerate(cells)}


def group_rows(rows: list, cells, groups: dict) -> list:
    """Per-cell rows -> H.3's row shape {"types": {group: sum}, "seed", ("odor")} for h3_rules."""
    pos = _pos(cells)
    out = []
    for r in rows:
        v = r["counts"]
        row = {"types": {g: int(sum(v[pos[int(i)]] for i in idx)) for g, idx in groups.items()}, "seed": int(r["seed"])}
        if "odor" in r:
            row["odor"] = r["odor"]
        out.append(row)
    return out


def group_react(ref_rows: list, rest_rows: list, cells, groups: dict, h4spec) -> dict:
    """H.4's reactivity (h3_rules.mbon_type_stats, react_med_delta_min / react_zero_share_max) on each group's sum."""
    ref, rest = group_rows(ref_rows, cells, groups), group_rows(rest_rows, cells, groups)
    by_seed = {r["seed"]: r for r in rest}
    return {g: mbon_type_stats(ref, by_seed, g, h4spec.react_med_delta_min, h4spec.react_zero_share_max) for g in groups}


def group_z(ref_rows: list, cells, group_cells, ddof: int):
    """(mean, sd) of the group sum over the reference presentations; None unless finite and sd > 0 (M.10.1)."""
    x = np.array([r["types"]["g"] for r in group_rows(ref_rows, cells, {"g": group_cells})], float)
    if len(x) <= ddof:                                                  # no sd defined (numpy would warn and give nan)
        return None
    m, s = float(x.mean()), float(x.std(ddof=ddof))
    return (m, s) if np.isfinite(m) and np.isfinite(s) and s > 0 else None


def cell_records(ref_rows: list, rest_rows: list, cells, group_cells, h4spec) -> dict:
    """Recorded only: each member cell's reactivity, and the largest share of the group-sum variance one cell carries
    (cov(cell, sum) / var(sum), population moments; 0 when the sum does not vary)."""
    groups = {str(int(i)): [int(i)] for i in group_cells}
    react = group_react(ref_rows, rest_rows, cells, groups, h4spec)
    pos = _pos(cells)
    x = np.array([[r["counts"][pos[int(i)]] for i in group_cells] for r in ref_rows], float)
    tot = x.sum(axis=1)
    var = float(tot.var())
    share = [float(np.cov(x[:, k], tot, bias=True)[0, 1] / var) if var > 0 else 0.0 for k in range(x.shape[1])]
    return dict(per_cell=[dict(cell=int(i), **react[str(int(i))], var_share=share[k]) for k, i in enumerate(group_cells)],
                max_var_share=max(share) if share else 0.0)


class _Missing(Exception):
    pass


def _missing(*_):
    raise _Missing


class MMeasurer:
    """Every M measurement through `cache` (h3_store.MeasureCache under results/m0d/m/); `pool` is a FlyPool."""

    def __init__(self, pool, spec, cache):
        self.pool, self.spec, self.cache = pool, spec, cache
        self.params_seen: list = []

    def _seen(self, params):
        if params not in self.params_seen:
            self.params_seen.append(params)

    def _items(self, kind, fn, params, items: list, keys: list) -> list:
        """One cache entry per item (keys[i] are its inputs), run in rounds of one item per worker."""
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

    def _window(self):
        h4 = self.spec.j.h4
        return dict(strength=h4.h3.strength, settle_ms=h4.oracle_window.settle_ms, read_ms=h4.oracle_window.read_ms)

    def reference(self, params, odors: list, cells) -> list:
        """H.3's reference presentations (reference_cells_job), one cache entry per odour; rows in odour order."""
        self._seen(params)
        h3 = self.spec.j.h4.h3
        common = dict(params=params, strength=h3.strength, settle_ms=h3.reference_window.settle_ms,
                      read_steps=int(h3.reference_window.read_ms), cells=[int(c) for c in cells])
        items = [dict(common, odors=[o]) for o in odors]
        return [r for rows in self._items("m_ref", m_jobs.reference_cells_job, params, items, items) for r in rows]

    def rest(self, params, seeds, cells) -> list:
        """H.3's rest window (rest_cells_job), one cache entry per seed."""
        self._seen(params)
        h3 = self.spec.j.h4.h3
        common = dict(params=params, settle_ms=h3.reference_window.settle_ms, read_steps=int(h3.reference_window.read_ms),
                      cells=[int(c) for c in cells])
        items = [dict(common, seeds=[int(s)]) for s in seeds]
        return [r for rows in self._items("m_rest", m_jobs.rest_cells_job, params, items, items) for r in rows]

    def pre(self, params, pairs: list, cells) -> list:
        """pre_job per pair (KC activity and the unedited probes are shared by every candidate: M.10.3's reuse)."""
        self._seen(params)
        h4 = self.spec.j.h4
        common = dict(params=params, cells=[int(c) for c in cells], act_seeds=tuple(h4.act_seeds),
                      select_seeds=tuple(h4.select_seeds), report_seeds=tuple(h4.report_seeds),
                      window_ms=int(h4.kc_window_ms), **self._window())
        items = [dict(common, odor_x=p["odor_x"], odor_y=p["odor_y"]) for p in pairs]
        keys = [dict(it, pair=[p["axis"], int(p["turn"]), p["x"], p["y"]]) for it, p in zip(items, pairs)]
        return self._items("m_pre", m_jobs.pre_job, params, items, keys)

    def edit(self, params, pairs: list, pres: list, cells, groups: dict, readout: dict, z: dict, reward_type: str,
             punish_type: str) -> list:
        """One oracle-shaped row per pair (compose(pre, edit)) plus axis / turn / x / y and edit_job's `edited`
        ({arm: {group, cells}}, the cells edited == the readout group's, logged per candidate). The cache key carries
        the groups' cells, the readout, z, both DAN names and pre's own inputs (act seeds, KC window) (M.10.3)."""
        self._seen(params)
        h4 = self.spec.j.h4
        g = {k: [int(i) for i in v] for k, v in groups.items()}
        common = dict(params=params, cells=[int(c) for c in cells], groups=g, readout=dict(readout),
                      z={k: tuple(v) for k, v in z.items()}, select_seeds=tuple(h4.select_seeds),
                      report_seeds=tuple(h4.report_seeds), alphas=tuple(h4.oracle_alphas), reward_type=reward_type,
                      punish_type=punish_type, **self._window())
        items = [dict(common, odor_x=p["odor_x"], odor_y=p["odor_y"], fx_idx=pr["fx_idx"], fx_val=pr["fx_val"],
                      pre_sel=pr["pre_sel"]) for p, pr in zip(pairs, pres)]
        keys = [dict(common, act_seeds=tuple(h4.act_seeds), window_ms=int(h4.kc_window_ms),       # pre's inputs too
                     pair=[p["axis"], int(p["turn"]), p["x"], p["y"]], odor_x=p["odor_x"], odor_y=p["odor_y"])
                for p in pairs]
        eds = self._items("m_edit", m_jobs.edit_job, params, items, keys)
        return [dict(axis=p["axis"], turn=int(p["turn"]), x=p["x"], y=p["y"], **compose(pr, ed, cells, g),
                     edited=ed["edited"]) for p, pr, ed in zip(pairs, pres, eds)]

    def teach(self, params, types, arm: str, punish_type: str, reward_type: str) -> list:
        """h4_jobs.teach_job unchanged, one arm, both declared orders x teach seeds (a record, reading 9)."""
        self._seen(params)
        s, h3 = self.spec.j.h4, self.spec.j.h4.h3
        common = dict(params=params, types=tuple(types), punish_type=punish_type, reward_type=reward_type,
                      k=h3.design_k, odor_seed=h3.design_odor_seed, strength=h3.strength, trials=s.teach_trials,
                      present_ms=s.teach_present_ms, gap_ms=s.teach_gap_ms, settle_ms=s.teach_window.settle_ms,
                      read_ms=s.teach_window.read_ms)
        jobs = [dict(common, seed=int(seed), arm=arm, order=o) for o in s.teach_orders for seed in s.teach_seeds]
        return self._items("m_teach", h4_jobs.teach_job, params, jobs, jobs)
