"""M0d H.4a.6 diagnostic: the oracle's specificity ceiling — H.4's oracle rerun on C3 and C0 with a weight edit that
depresses only X's Kenyon cells.

Why this exists. H.4 (spec appendix H.4 step 3; result H.4a.5) ended `STOP_LOW_T_B`: C3, the only eligible combination,
had 7 of 21 (b)-axis pairs testable (T_b 0.333 < 0.5). H.4a.6's diagnosis of the recorded run (h4_testability_diag.py;
write-up .superpowers/drafts/2026-09-22-h4-testability-diag.md) put the residual limit at X/Y specificity at the edited
KC->MBON synapses: when the oracle depresses X's KCs, Y's readout moves 54-80% as much as X's. The recorded rows cannot
say how much perfect specificity would buy; this measures it. The job is h4_jobs.oracle_job (G.14.3) with one change,
the per-KC fraction both edits scale alpha by (`edit_fraction`):
  - mode "oracle": f_x, KC i's firing reliability for X over the act seeds — oracle_job itself;
  - mode "x_only": f_x * (f_y == 0) — only KCs that fired for X on some act seed and never for Y are depressed
    (the `freq` family: each such KC's depression scales with its firing reliability f_x);
  - mode "x_only_all": 1[f_x > 0 and f_y == 0] — the same KCs, each depressed by the full alpha regardless of f_x
    (the `all` family, as G.4's O-all; spec H.4a.7's re-read when the freq ceiling is edit-mass-limited).
--family freq (default) measures x_only, --family all measures x_only_all; everything below that says x_only means the
measured mode.
Seeds, alpha selection, probes and return keys are oracle_job's; the rows add f_x, f_y (lists) and the mode. h4_jobs.py
is inside H.4's measurement key, so the job body is copied here, not edited there (`--check` prints the difference).

Measured: C3 and C0 (block "h3"'s adopted Params, block "h4"'s readout and z): x_only on all 39 E0 pairs, oracle on the
first (b) pair, and per engine the KC->readout weights the edits scale. Self-check before any analysis (exit 1): the
oracle row equals H.4's cached row for that pair and engine (the cache entry the run report lists, sha256-checked,
loaded by h4_testability_diag.load), the oracle and x_only rows of that pair share f_x / f_y, and every x_only row's
parts measured before any edit (KC activity, the pre probes on the select and report seeds) equal H.4's.
Analysis (descriptive, post hoc; nothing here is a rule or a verdict): under x_only, h4_rules.combo_stats /
combo_records with H.4's z and SPEC — testable counts, T_b, F_a, naive counts per axis, per pair r / p / m next to
H.4's, the pairs gained and lost; per pair the realised Y/X (h4_testability_diag's `gen`: mean dV(Y) / mean dV(X)
under the reward and under the punishment edit) under oracle (H.4's cache) and x_only; the share of X's KC activity
the mask removes (sum f_x over KCs with f_y > 0 / sum f_x); and the predicted Y/X of the oracle's edit from the edited
KC->MBON05 (PAM08 core: reward) and KC->MBON13 (PPL105 core: punishment) weights w — "briefed" sum w f_x f_y /
sum w f_y and "linear" sum w f_x f_y / sum w f_x^2 (Y's lost drive over X's) — against H.4's realised oracle Y/X
(Spearman).
x_only's predicted Y/X is 0 by construction, so its realised Y/X is the spread a synapse edit cannot remove.

    uv run python docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py [out.json]   # ~40-50 min on 16 workers
        # default results/m0d/diag/h4_specificity_ceiling.json, + _summary.json next to it and
        # .superpowers/drafts/2026-09-22-h4-specificity-ceiling.md
    uv run python docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py --smoke      # minutes: plumbing only
    uv run python docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py --family all # the all-family ceiling
        # results/m0d/diag/h4_specificity_ceiling_all.json, _all_summary.json and
        # .superpowers/drafts/2026-09-22-h4-specificity-ceiling-all.md; with the freq summary present, the findings
        # add the freq ceiling per pair (testable) and per engine (T_b, F_a)
    uv run python docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py --check      # seconds, no engine

Run it from the repository root. --smoke: the first (b) pair per engine in both modes, act / select / report seeds cut
to 2 each (run_m0d_h4.smoke_spec); H.4's identity check is skipped (other seeds) and the two modes' pre-edit parts are
compared instead; output results/m0d/diag/h4_specificity_ceiling_smoke.json and a -smoke.md draft (its numbers are not
measurements; with --family all, _all_smoke.json and -all-smoke.md). --check: edit_fraction on toy arrays, the
output names per family, the job kwargs against h4_measure's, and the job's signature and body against oracle_job's. Outputs go through
h3_store.write_json (results/m0d/ only). Exit codes: 0 done; 1 a self-check failed (the raw rows are kept); 2 refused
before measuring — not at the root, a C3 threshold file missing or off its sha256, block "h4" not the recorded run,
another connectome or pair list, job kwargs other than h4_measure's.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import importlib.util
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

from flymon.brain import h4_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, code_key, git_state, sha256_file, write_json
from flymon.brain.h4_formula import dprime, dv
from flymon.brain.h4_jobs import _present_kc, rig_for, type_cells
from flymon.brain.h4_measure import HASHED_FILES, MEASURE_FILES, H4Measurer
from flymon.brain.h4_pairs import even_pairs, pair_key, pairs_digest
from flymon.brain.h4_rules import combo_records, combo_stats
from flymon.brain.h4_spec import SPEC
from flymon.brain.presentation import decide

HERE = Path(__file__).resolve().parent
NPZ = "data/malecns.npz"
SUMMARY = Path("results/summary/m0d.json")
RUN = Path("results/m0d/h4/runs/20260921T174430Z-7986f4.json")         # H.4's recorded run (H.4a.5)
OUT = Path("results/m0d/diag/h4_specificity_ceiling.json")
OUT_SMOKE = Path("results/m0d/diag/h4_specificity_ceiling_smoke.json")   # = out_paths(True, "freq")[0]
DRAFT = Path(".superpowers/drafts/2026-09-22-h4-specificity-ceiling.md")
ENGINES = ("C3", "C0")                     # C3 first: the only eligible combination
MODES = ("oracle", "x_only", "x_only_all")
FAMILIES = {"freq": "x_only", "all": "x_only_all"}   # --family -> the measured mode (the identity row stays "oracle")
POOL_TIMEOUT_S = 1800.0                    # per round of one pair per worker (~6-8 min), as scripts/run_m0d_h4.py
PRE_EDIT = (("kc",), ("select", "pre"), ("counts", "pre"), ("report", "pre"))   # measured before any weight edit
PAIR_KEYS = ("axis", "turn", "x", "y")
ADDED_KEYS = ("fx", "fy", "mode")
BODY_DIFF = (                              # oracle_job -> ceiling_job, code lines (stripped); --check holds them
    ["wv[rew] = p.w0[rew] * (1.0 - a_r * fx)[p.pre_kc[rew]]",
     "wv[pun] = p.w0[pun] * (1.0 - a_p * fx)[p.pre_kc[pun]]",
     '"jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}'],
    ["fe = edit_fraction(fx, fy, mode)                 # the one change: the per-KC fraction both edits scale by",
     "wv[rew] = p.w0[rew] * (1.0 - a_r * fe)[p.pre_kc[rew]]",
     "wv[pun] = p.w0[pun] * (1.0 - a_p * fe)[p.pre_kc[pun]]",
     '"jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))},',
     '"fx": fx.tolist(), "fy": fy.tolist(), "mode": mode}'])


def out_paths(smoke: bool, family: str, out=None) -> tuple:
    """(raw rows, summary, draft) for a run: freq keeps the original names; all adds "_all" / "-all" before the
    extension (before "_smoke" / "-smoke" under --smoke). An explicit `out` is kept; its summary sits next to it."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {family!r}")
    tag = "" if family == "freq" else "_" + family
    raw = OUT.with_name(OUT.stem + tag + ("_smoke" if smoke else "") + OUT.suffix)
    raw = Path(out) if out else raw
    draft = DRAFT.with_name(DRAFT.stem + tag.replace("_", "-") + ("-smoke" if smoke else "") + DRAFT.suffix)
    return raw, raw.with_name(raw.stem + "_summary.json"), draft


# ================================================================ worker jobs (module level: spawn pickles them)
def edit_fraction(fx, fy, mode: str) -> np.ndarray:
    """The per-KC fraction the oracle's edits scale alpha by: "oracle" -> f_x unchanged (G.14.3); "x_only" -> f_x where
    f_y == 0, else 0 (only KCs that fired for X on some act seed and never for Y); "x_only_all" -> 1 on those same KCs,
    0 elsewhere (the full alpha regardless of f_x). Float arrays of one shape."""
    fx, fy = np.asarray(fx, float), np.asarray(fy, float)
    if fx.shape != fy.shape:
        raise ValueError(f"f_x {fx.shape} and f_y {fy.shape} differ in shape")
    if mode == "oracle":
        return fx
    if mode == "x_only":
        return fx * (fy == 0)
    if mode == "x_only_all":
        return ((fx > 0) & (fy == 0)).astype(np.float64)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def ceiling_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, readout: dict, z: dict, types,
                act_seeds, select_seeds, report_seeds, alphas, strength: float, settle_ms: float, read_ms: float,
                window_ms: int, punish_type: str, reward_type: str, mode: str) -> dict:
    """h4_jobs.oracle_job with both edits scaling alpha by edit_fraction(f_x, f_y, mode) instead of f_x (mode "oracle"
    is oracle_job); the result adds f_x and f_y over the act seeds (lists) and the mode."""
    e, p, c = rig_for(eng.conn, pops, params)
    cells = type_cells(e.conn, types)
    idx = np.concatenate([cells[n] for n in types])
    bounds = np.cumsum([0] + [len(cells[n]) for n in types])
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win = [], [], []
            for s in act_seeds:
                o = _present_kc(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}

        def probe(seeds):
            out = {n: [] for n in types}
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                for j, n in enumerate(types):
                    out[n].append(cnt[:, bounds[j]:bounds[j + 1]].sum(1).tolist())
            return out

        def ap(pr):
            return {"A": pr[readout["A"]], "P": pr[readout["P"]]}

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
        fe = edit_fraction(fx, fy, mode)                 # the one change: the per-KC fraction both edits scale by
        rew = np.isin(p.post_mb, p.mb_local[c[reward_type].core]); pun = np.isin(p.post_mb, p.mb_local[c[punish_type].core])
        w = e.csc.w

        def set_w(a_r, a_p=None):
            wv = p.w0.copy()
            if a_r is not None:
                wv[rew] = p.w0[rew] * (1.0 - a_r * fe)[p.pre_kc[rew]]
            if a_p is not None:
                wv[pun] = p.w0[pun] * (1.0 - a_p * fe)[p.pre_kc[pun]]
            w[p.edges] = wv

        set_w(None); pre_sel = probe(select_seeds)
        reward = {}
        for a in alphas:
            set_w(a); r1 = probe(select_seeds)
            reward[str(a)] = {"R1": r1, "change": dprime(dv(ap(r1), z) - dv(ap(pre_sel), z))}
        a_r = max(alphas, key=lambda a: (reward[str(a)]["change"], -a))
        punish = {}
        for a in alphas:
            set_w(a_r, a); r2 = probe(select_seeds)
            punish[str(a)] = {"R2": r2, "change": dprime(dv(ap(r2), z) - dv(ap(reward[str(a_r)]["R1"]), z))}
        a_p = min(alphas, key=lambda a: (punish[str(a)]["change"], a))
        set_w(None); pre = probe(report_seeds)
        set_w(a_r); R1 = probe(report_seeds)
        set_w(a_r, a_p); R2 = probe(report_seeds)
        w[p.edges] = p.w0
        return {"select": {"pre": pre_sel, "reward": reward, "punish": punish},
                "alpha_reward": a_r, "alpha_punish": a_p,
                "counts": {"pre": pre, "R1": R1, "R2": R2},
                "report": {"pre": ap(pre), "R1": ap(R1), "R2": ap(R2)},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))},
                "fx": fx.tolist(), "fy": fy.tolist(), "mode": mode}
    finally:
        p.reset_weights(); p.set_enabled(True)


def kc_weights_job(eng, pl, pops, comps, ro, params, readout: dict, punish_type: str, reward_type: str) -> dict:
    """Per KC, the summed w0 of its synapses that an edit scales onto the readout type that edit targets: the punishment
    edit (punish_type core) onto readout A, the reward edit (reward_type core) onto readout P."""
    e, p, c = rig_for(eng.conn, pops, params)
    out = {}
    for k, dan in (("A", punish_type), ("P", reward_type)):
        t = readout[k]
        cells = type_cells(e.conn, [t])[t]
        onto = np.isin(p.post_mb, p.mb_local[cells])
        edited = onto & np.isin(p.post_mb, p.mb_local[c[dan].core])
        out[k] = dict(type=t, dan=dan, n_cells=int(len(cells)), n_cells_core=int(np.isin(cells, c[dan].core).sum()),
                      w_unedited=float(p.w0[onto & ~edited].sum()), w_edited=float(p.w0[edited].sum()),
                      w=np.bincount(p.pre_kc[edited], weights=p.w0[edited], minlength=len(pops.kc)).tolist())
    return out


# ================================================================ the job kwargs, checked against h4_measure's
def pool_types(pools: dict) -> list:
    return [t for k in ("A", "P") for t in pools[k]]                     # H4Measurer.types


def oracle_kwargs(spec, params, readout: dict, z: dict, pools: dict, pair: dict, mode: str) -> dict:
    """ceiling_job's kwargs for one pair: H4Measurer.oracle's oracle_job kwargs from `spec` (held equal: --check and
    every run) plus the mode."""
    s, h3 = spec, spec.h3
    return dict(params=params, readout=dict(readout), z={k: tuple(v) for k, v in z.items()},
                types=tuple(pool_types(pools)), act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds),
                report_seeds=tuple(s.report_seeds), alphas=tuple(s.oracle_alphas), strength=h3.strength,
                settle_ms=s.oracle_window.settle_ms, read_ms=s.oracle_window.read_ms, window_ms=s.kc_window_ms,
                punish_type=h3.punish_type, reward_type=h3.reward_type, odor_x=pair["odor_x"], odor_y=pair["odor_y"],
                mode=mode)


class _Recorder:
    """Pool and cache stand-in: H4Measurer.oracle builds its jobs; nothing runs, nothing is cached."""
    n_workers = 1 << 20

    def __init__(self):
        self.fns, self.jobs = [], []

    def run_jobs(self, fn, kwargs_list):
        self.fns.append(fn); self.jobs += [dict(kw) for kw in kwargs_list]
        return [{} for _ in kwargs_list]

    def get_or_compute(self, kind, inputs, compute, params_list):
        return compute()                                                   # H4Measurer's first pass: _missing raises


def h4_measure_kwargs(spec, params, readout: dict, z: dict, pools: dict, pairs: list) -> list:
    """The oracle_job kwargs h4_measure.H4Measurer.oracle builds for `pairs`, in pair order."""
    rec = _Recorder()
    H4Measurer(rec, spec, pairs, pools, rec, None).oracle(params, readout, z)
    if rec.fns != [h4_jobs.oracle_job] or len(rec.jobs) != len(pairs):
        raise AssertionError(f"H4Measurer.oracle ran {rec.fns} on {len(rec.jobs)} jobs, not oracle_job on {len(pairs)}")
    return rec.jobs


def kwargs_mismatch(spec, params, readout: dict, z: dict, pools: dict, pairs: list) -> list:
    """The pairs whose oracle_kwargs (without the mode) differ from h4_measure's, keys or values."""
    ref = h4_measure_kwargs(spec, params, readout, z, pools, pairs)
    bad = []
    for p, r in zip(pairs, ref):
        mine = {k: v for k, v in oracle_kwargs(spec, params, readout, z, pools, p, "oracle").items() if k != "mode"}
        if sorted(mine) != sorted(r) or mine != r:
            bad.append(pair_key(p))
    return bad


# ================================================================ --check (no engine)
def _body(fn) -> list:
    """fn's source lines after its signature and docstring."""
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    first = node.body[1] if ast.get_docstring(node) is not None else node.body[0]
    return src.splitlines()[first.lineno - 1:]


def body_diff() -> tuple:
    """(removed, added) code lines from h4_jobs.oracle_job's body to ceiling_job's, stripped."""
    d = list(difflib.unified_diff(_body(h4_jobs.oracle_job), _body(ceiling_job), lineterm="", n=0))
    rem = [ln[1:].strip() for ln in d if ln.startswith("-") and not ln.startswith("---")]
    add = [ln[1:].strip() for ln in d if ln.startswith("+") and not ln.startswith("+++")]
    return rem, add


def need(ok: bool, what: str, done: list) -> None:
    if not ok:
        raise SystemExit(f"check failed: {what}")
    done.append(what)


def run_check() -> int:
    done = []
    fx = np.array([0.0, 0.125, 0.5, 1.0, 0.25, 0.0, 0.875]); fy = np.array([0.0, 0.0, 0.25, 1.0, 0.0, 0.5, 0.125])
    o = edit_fraction(fx, fy, "oracle")
    need(o.dtype == np.float64 and o.shape == fx.shape and np.array_equal(o, fx), "oracle mode returns f_x unchanged "
         "(float64, same shape)", done)
    need(np.array_equal(1.0 - 0.5 * o, 1.0 - 0.5 * fx), "oracle mode: the edit factor 1 - alpha * fraction is "
         "oracle_job's bit for bit", done)
    x = edit_fraction(fx, fy, "x_only")
    need(x.dtype == np.float64 and x.shape == fx.shape, "x_only returns float64 of f_x's shape", done)
    need(bool((x[fy > 0] == 0).all()) and np.array_equal(x[fy == 0], fx[fy == 0]), "x_only zeroes every KC with "
         "f_y > 0 and keeps f_x elsewhere", done)
    need(np.array_equal(x, [0.0, 0.125, 0.0, 0.0, 0.25, 0.0, 0.0]),
         "x_only on the toy arrays = [0, .125, 0, 0, .25, 0, 0]", done)
    need(np.array_equal(edit_fraction(list(fx), list(fy), "x_only"), x)
         and edit_fraction([], [], "x_only").shape == (0,), "lists and empty inputs give the same arrays", done)
    xa = edit_fraction(fx, fy, "x_only_all")
    need(xa.dtype == x.dtype and xa.shape == x.shape, "x_only_all returns x_only's dtype and shape", done)
    need(bool((xa[(fx > 0) & (fy == 0)] == 1.0).all()) and bool((xa[(fy > 0) | (fx == 0)] == 0.0).all()),
         "x_only_all is 1 exactly where f_x > 0 and f_y == 0, 0 where f_y > 0 or f_x == 0", done)
    need(np.array_equal(xa, [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) and np.array_equal(xa > 0, x > 0),
         "x_only_all on the toy arrays = [0, 1, 0, 0, 1, 0, 0], the KCs x_only edits", done)
    need(np.array_equal(1.0 - 0.5 * xa, np.where(xa > 0, 0.5, 1.0)), "x_only_all: the edit factor is 1 - alpha on "
         "X-only KCs, 1 elsewhere", done)
    need(edit_fraction([], [], "x_only_all").shape == (0,) and edit_fraction([], [], "x_only_all").dtype == x.dtype,
         "x_only_all on empty inputs: an empty float64 array", done)
    need(FAMILIES == {"freq": "x_only", "all": "x_only_all"} and set(FAMILIES.values()) < set(MODES),
         "--family freq measures x_only, --family all measures x_only_all", done)
    names = {(sm, fa): tuple(map(str, out_paths(sm, fa))) for sm in (False, True) for fa in ("freq", "all")}
    need(names[False, "freq"] == ("results/m0d/diag/h4_specificity_ceiling.json",
                                  "results/m0d/diag/h4_specificity_ceiling_summary.json",
                                  ".superpowers/drafts/2026-09-22-h4-specificity-ceiling.md")
         and names[True, "freq"] == ("results/m0d/diag/h4_specificity_ceiling_smoke.json",
                                     "results/m0d/diag/h4_specificity_ceiling_smoke_summary.json",
                                     ".superpowers/drafts/2026-09-22-h4-specificity-ceiling-smoke.md")
         and out_paths(False, "freq")[0] == OUT and out_paths(True, "freq")[0] == OUT_SMOKE,
         "--family freq output names unchanged (full and smoke)", done)
    need(names[False, "all"] == ("results/m0d/diag/h4_specificity_ceiling_all.json",
                                 "results/m0d/diag/h4_specificity_ceiling_all_summary.json",
                                 ".superpowers/drafts/2026-09-22-h4-specificity-ceiling-all.md")
         and names[True, "all"] == ("results/m0d/diag/h4_specificity_ceiling_all_smoke.json",
                                    "results/m0d/diag/h4_specificity_ceiling_all_smoke_summary.json",
                                    ".superpowers/drafts/2026-09-22-h4-specificity-ceiling-all-smoke.md"),
         "--family all output names carry _all / -all (full and smoke)", done)
    need(str(out_paths(False, "all", "x/y.json")[1]) == "x/y_summary.json", "an explicit out keeps its name", done)
    for fam in FAMILIES:
        res = _toy_res(fam)
        md = md_report(res, tables(res), findings(res))
        need(f"({FAMILIES[fam]} edit, {fam} family)" in md and f"Ceiling family: **{fam}**" in md
             and (("freq T" in md and "freq-family" in md) == (fam == "all")),
             f"the {fam}-family report labels its family" + (" and compares with the freq ceiling" if fam == "all"
                                                            else ""), done)
    for bad, what in ((lambda: edit_fraction(fx, fy, "X_only"), "an unknown mode"),
                      (lambda: edit_fraction(fx, fy[:-1], "x_only"), "a shape mismatch")):
        try:
            bad()
            raised = False
        except ValueError:
            raised = True
        need(raised, f"{what} raises ValueError", done)
    cli = h4cli()
    pools = {"A": ["MBON13", "MBON18"], "P": ["MBON05", "MBON21"]}
    readout, z = {"A": "MBON13", "P": "MBON05"}, {"A": (10.5, 9.25), "P": (26.0, 19.5)}
    pair = {"axis": "b", "turn": 0, "x": "Tackle vs A", "y": "Tackle vs B", "odor_x": {"ORN_A": 1.25, "ORN_B": 0.75},
            "odor_y": {"ORN_A": 1.0, "ORN_C": 1.0}}
    for label, spec in (("SPEC", SPEC), ("smoke_spec(SPEC)", cli.smoke_spec(SPEC))):
        need(not kwargs_mismatch(spec, Params(), readout, z, pools, [pair]), f"kwargs = h4_measure's for one pair "
             f"({label}): same keys and values", done)
    kw = oracle_kwargs(SPEC, Params(), readout, z, pools, pair, "x_only")
    ref = h4_measure_kwargs(SPEC, Params(), readout, z, pools, [pair])[0]
    need(sorted(kw) == sorted(list(ref) + ["mode"]), f"kwargs keys = h4_measure's + mode: {sorted(kw)}", done)
    sig_o = list(inspect.signature(h4_jobs.oracle_job).parameters)
    sig_c = list(inspect.signature(ceiling_job).parameters)
    need(sig_c == sig_o + ["mode"], "ceiling_job's parameters = oracle_job's + mode", done)
    inspect.signature(ceiling_job).bind(None, None, None, None, None, **kw)
    done.append("the kwargs bind to ceiling_job")
    rem, add = body_diff()
    need((rem, add) == BODY_DIFF, "ceiling_job's body = oracle_job's except the lines below", done)
    for w in done:
        print(f"ok  {w}")
    print("body diff oracle_job -> ceiling_job:")
    print("\n".join([f"  - {ln}" for ln in rem] + [f"  + {ln}" for ln in add]))
    print(f"check ok ({len(done)} checks)")
    return 0


def _toy_res(family: str) -> dict:
    """A hand-made analysis result (two pairs, one engine) to render the tables, findings and markdown in --check;
    family "all" carries a freq_ceiling."""
    xm = FAMILIES[family]
    agg = dict(testable_a=1, testable_b=1, n_b=1, T_b=1.0, F_a=1, naive_a=1, naive_b=1, bar=False)
    sp = {e: {f: dict(rho=None, n=2) for f in ("briefed", "linear")} for e in ("reward", "punish")}
    yx = {e: {"oracle": 0.5, xm: 0.0} for e in ("reward", "punish")}
    pred = {e: dict(briefed=0.25, linear=0.5) for e in ("reward", "punish")}
    st = lambda t: dict(d_pre=1.0, r=0.5, p=0.5, m=0.5, testable=t, alpha_reward=0.8, alpha_punish=0.8)
    pairs = [dict(axis=ax, turn=0, x="X", y=ax, h4=st(False), yx=yx, predicted=pred,
                  mask=dict(n_x=4, n_y=4, n_removed=2, removed=0.5), **{xm: st(ax == "b")}) for ax in ("a", "b")]
    grp = lambda P: dict(n=len(P), testable_h4=0, **{f"testable_{xm}": 1}, gained=["t0 X / b"], lost=[],
                         yx={e: {"oracle": 0.5, xm: 0.0} for e in ("reward", "punish")}, removed=0.5, n_x=4.0,
                         n_removed=2.0, predicted=pred, spearman=sp)
    res = dict(script="s.py", script_sha256="0" * 64, commit="0" * 40, smoke=False, raw="r.json", h4_run="h.json",
               self_check="toy", engines={"C3": dict(reasons=[], aggregate=agg, records=None, h4_aggregate=agg,
                                                    h4_records=None, pairs=pairs,
                                                    groups={"all": grp(pairs), "a": grp(pairs[:1]),
                                                            "b": grp(pairs[1:])})})
    if family != "freq":
        res.update(family=family, mode=xm, freq_ceiling=dict(source="f.json", engines={
            "C3": dict(aggregate=dict(agg, testable_b=0, T_b=0.0), testable={"t0 X / a": False, "t0 X / b": False})}))
    return res


# ================================================================ loading helpers (parent only)
def _module(name: str, path: Path):
    """A sibling diagnostic or a CLI script, loaded by path (neither directory is a package)."""
    if name not in sys.modules:
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return sys.modules[name]


def diag():
    return _module("h4_testability_diag", HERE / "h4_testability_diag.py")


def h4cli():
    return _module("run_m0d_h4", ROOT / "scripts" / "run_m0d_h4.py")


def refuse(msg: str) -> int:
    print(f"refusing to run: {msg}", file=sys.stderr)
    return 2


def pk(r: dict) -> tuple:
    return (r["axis"], int(r["turn"]), r["x"], r["y"])


def label(k: tuple) -> str:
    return f"t{k[1]} {k[2]} / {k[3]}"


def strip(row: dict, extra=ADDED_KEYS) -> dict:
    return {k: v for k, v in row.items() if k not in PAIR_KEYS + tuple(extra)}


def part(row: dict, path: tuple):
    for k in path:
        row = row[k]
    return row


# ================================================================ self-check
def self_check(dg, rows: dict, h4_rows: dict, smoke: bool, partial: bool = False, xm: str = "x_only") -> list:
    """The reasons the rows cannot be read (empty = fine). Full run: every oracle-mode row = H.4's cached row (minus
    f_x / f_y / mode) and every x_only row's pre-edit parts = H.4's. Smoke (other seeds): x_only's pre-edit parts = the
    oracle-mode row's. Both: the two modes' f_x / f_y agree on the shared pair. partial: missing rows are skipped.
    xm: the measured mode ("x_only" or "x_only_all")."""
    probs = []
    for c, rs in rows.items():
        orc = {pk(r): r for r in rs if r["mode"] == "oracle"}
        xo = {pk(r): r for r in rs if r["mode"] == xm}
        ref = {pk(r): r for r in h4_rows[c]}
        for k, r in orc.items():
            if not smoke:
                a, b = strip(r), strip(ref[k], ())
                diff = sorted(f for f in set(a) | set(b) if dg.canon(a.get(f)) != dg.canon(b.get(f)))
                if diff:
                    probs.append(f"{c} {label(k)}: the oracle-mode row differs from H.4's cached row in {diff}")
            if k not in xo:
                if not partial:
                    probs.append(f"{c} {label(k)}: no {xm} row for the oracle-mode pair")
            elif xo[k]["fx"] != r["fx"] or xo[k]["fy"] != r["fy"]:
                probs.append(f"{c} {label(k)}: {xm}'s f_x / f_y differ from the oracle-mode row's (same act seeds)")
        for k, x in xo.items():
            b = orc.get(k) if smoke else ref[k]
            if b is None:
                continue
            bad = [".".join(p) for p in PRE_EDIT if dg.canon(part(x, p)) != dg.canon(part(b, p))]
            if bad:
                probs.append(f"{c} {label(k)}: {xm}'s pre-edit parts {bad} differ from "
                             f"{'the oracle-mode row' if smoke else 'H.4'}")
    return probs


# ================================================================ analysis
def ratio(a: float, b: float):
    return float(a / b) if abs(b) > 1e-12 else None


def predicted_yx(w, fx, fy) -> dict:
    """The oracle's edit read linearly at the edited synapses (w per KC): Y's lost drive sum w f_x f_y over Y's drive
    sum w f_y (as briefed) and over X's lost drive sum w f_x^2."""
    w, fx, fy = (np.asarray(v, float) for v in (w, fx, fy))
    lost_y = float((w * fx * fy).sum())
    return dict(briefed=ratio(lost_y, float((w * fy).sum())), linear=ratio(lost_y, float((w * fx * fx).sum())))


def rho(dg, a: list, b: list) -> dict:
    keep = [(u, v) for u, v in zip(a, b) if u is not None and v is not None and np.isfinite(u) and np.isfinite(v)]
    return dict(rho=dg.spearman(*zip(*keep)) if len(keep) >= 3 else None, n=len(keep))


def analyse_engine(dg, spec, z: dict, rows: list, h4_rows: list, h4_blk: dict, weights: dict,
                   xm: str = "x_only") -> dict:
    """xm: the measured mode; its per-pair statistics sit under key xm (freq: "x_only", as before)."""
    tmin = SPEC.testable_min
    xo = [r for r in rows if r["mode"] == xm]
    st = combo_stats(xo, z, [pk(r) for r in xo], spec)
    rec = None if st["reasons"] else combo_records(xo, z, spec)
    obs = {pk(r): r for r in h4_blk["oracle"]["pairs"]}                  # H.4's per-pair statistics (run report)
    ref = {pk(r): r for r in h4_rows}
    xs = {pk(r): r for r in st["pairs"]}
    pairs = []
    for r in xo:
        k = pk(r); o, s = obs[k], xs.get(k, {})
        fo, fz = dg.features(ref[k], z, tmin), dg.features(r, z, tmin)
        fx, fy = np.asarray(r["fx"], float), np.asarray(r["fy"], float)
        shared = (fx > 0) & (fy > 0)                                       # the KCs x_only leaves unedited
        pairs.append(dict(
            axis=k[0], turn=k[1], x=k[2], y=k[3],
            h4=dict({f: o[f] for f in ("d_pre", "r", "p", "m", "testable")}, alpha_reward=ref[k]["alpha_reward"],
                    alpha_punish=ref[k]["alpha_punish"]),
            **{xm: dict({f: s.get(f) for f in ("d_pre", "r", "p", "m", "testable")}, alpha_reward=r["alpha_reward"],
                        alpha_punish=r["alpha_punish"])},
            yx={e: {"oracle": fo[e]["gen"], xm: fz[e]["gen"]} for e in ("reward", "punish")},
            mask=dict(n_x=int((fx > 0).sum()), n_y=int((fy > 0).sum()), n_removed=int(shared.sum()),
                      removed=ratio(float(fx[shared].sum()), float(fx.sum()))),
            predicted=dict(reward=predicted_yx(weights["P"]["w"], fx, fy),        # reward edit: readout P's synapses
                           punish=predicted_yx(weights["A"]["w"], fx, fy))))
    groups = {"all": pairs, **{ax: [q for q in pairs if q["axis"] == ax] for ax in ("a", "b")}}
    by = {}
    for g, P in groups.items():
        if not P:
            continue
        m = lambda f: dg.med([f(q) for q in P])
        by[g] = dict(
            n=len(P), testable_h4=sum(bool(q["h4"]["testable"]) for q in P),
            **{f"testable_{xm}": sum(bool(q[xm]["testable"]) for q in P)},
            gained=[label(pk(q)) for q in P if q[xm]["testable"] and not q["h4"]["testable"]],
            lost=[label(pk(q)) for q in P if q["h4"]["testable"] and not q[xm]["testable"]],
            yx={e: {md: m(lambda q: q["yx"][e][md]) for md in ("oracle", xm)} for e in ("reward", "punish")},
            removed=m(lambda q: q["mask"]["removed"]), n_x=m(lambda q: q["mask"]["n_x"]),
            n_removed=m(lambda q: q["mask"]["n_removed"]),
            predicted={e: {f: m(lambda q: q["predicted"][e][f]) for f in ("briefed", "linear")}
                       for e in ("reward", "punish")},
            spearman={e: {f: rho(dg, [q["predicted"][e][f] for q in P], [q["yx"][e]["oracle"] for q in P])
                          for f in ("briefed", "linear")} for e in ("reward", "punish")})
    return dict(reasons=st["reasons"], aggregate=st["aggregate"], records=rec,
                h4_aggregate=h4_blk["oracle"]["aggregate"], h4_records=h4_blk.get("records"), groups=by, pairs=pairs,
                weights={k: {f: v for f, v in w.items() if f != "w"} for k, w in weights.items()})


def f2(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "."
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def tables(res: dict) -> list:
    """[(title, header, body)] for the terminal and the markdown write-up."""
    E = res["engines"]
    xm = res.get("mode", "x_only")
    T = []
    single = lambda rec, k: (str(rec["single_type"][k]["aggregate"]["testable_b"])
                             if rec and rec["single_type"][k]["aggregate"] else "-")
    agg_row = lambda c, edit, a, rec: [c, edit, a["testable_a"], f"{a['testable_b']}/{a['n_b']} ({a['T_b']:.3f})",
                                       a["F_a"], f"{a['naive_a']} / {a['naive_b']}", a["bar"],
                                       f"{single(rec, 'A')} / {single(rec, 'P')}",
                                       " / ".join(str(h["testable_b"]) for h in rec["report_halves"]) if rec else "-"]
    body = []
    for c, e in E.items():
        body.append(agg_row(c, "H.4 oracle", e["h4_aggregate"], e["h4_records"]))
        body.append(agg_row(c, xm, e["aggregate"], e["records"]) if e["aggregate"] else
                    [c, xm, f"INVALID {e['reasons']}", "", "", "", "", "", ""])
    T.append((f"Aggregate: H.4's oracle and the {xm} edit (H.4's z and SPEC)",
              ["engine", "edit", "testable (a)", "testable (b) (T_b)", "F_a", "naive (a) / (b)", "bar",
               "A only / P only testable (b)", "report-seed halves testable (b)"], body))
    T.append((f"Pairs gained / lost by the {xm} edit",
              ["engine", "axis", f"testable H.4 -> {xm}", "gained", "lost"],
              [[c, g, f"{v['testable_h4']} -> {v[f'testable_{xm}']} of {v['n']}", "; ".join(v["gained"]) or "-",
                "; ".join(v["lost"]) or "-"] for c, e in E.items() for g, v in e["groups"].items() if g != "all"]))
    T.append(("Realised Y/X (median of mean dV(Y) / mean dV(X)) and the mask",
              ["engine", "axis", f"reward Y/X oracle -> {xm}", f"punish Y/X oracle -> {xm}",
               "X's KC activity removed", "X KCs / removed (median n)", "predicted reward Y/X briefed / linear",
               "predicted punish Y/X briefed / linear"],
              [[c, g, f"{f2(v['yx']['reward']['oracle'])} -> {f2(v['yx']['reward'][xm])}",
                f"{f2(v['yx']['punish']['oracle'])} -> {f2(v['yx']['punish'][xm])}", f2(v["removed"]),
                f"{v['n_x']:.0f} / {v['n_removed']:.0f}",
                f"{f2(v['predicted']['reward']['briefed'])} / {f2(v['predicted']['reward']['linear'])}",
                f"{f2(v['predicted']['punish']['briefed'])} / {f2(v['predicted']['punish']['linear'])}"]
               for c, e in E.items() for g, v in e["groups"].items()]))
    T.append(("Predicted vs realised oracle Y/X: Spearman rho (n)",
              ["engine", "axis", "reward briefed", "reward linear", "punish briefed", "punish linear"],
              [[c, g, *(f"{f2(v['spearman'][ed][f]['rho'])} ({v['spearman'][ed][f]['n']})"
                        for ed in ("reward", "punish") for f in ("briefed", "linear"))]
               for c, e in E.items() for g, v in e["groups"].items()]))
    for c, e in E.items():
        T.append((f"{c} per pair (T = testable; alpha reward / punish; Y/X oracle -> {xm})",
                  ["pair", "axis", "H.4 r / p / m", "H.4 T", f"{xm} r / p / m", f"{xm} T", f"alphas H.4 -> {xm}",
                   "reward Y/X", "punish Y/X", "removed", "pred. reward b / l", "pred. punish b / l"],
                  [[label(pk(q)), q["axis"], " / ".join(f2(q["h4"][f]) for f in ("r", "p", "m")),
                    f2(q["h4"]["testable"]),
                    " / ".join(f2(q[xm][f]) for f in ("r", "p", "m")), f2(q[xm]["testable"]),
                    f"{q['h4']['alpha_reward']}/{q['h4']['alpha_punish']} -> {q[xm]['alpha_reward']}/"
                    f"{q[xm]['alpha_punish']}",
                    f"{f2(q['yx']['reward']['oracle'])} -> {f2(q['yx']['reward'][xm])}",
                    f"{f2(q['yx']['punish']['oracle'])} -> {f2(q['yx']['punish'][xm])}", f2(q["mask"]["removed"]),
                    f"{f2(q['predicted']['reward']['briefed'])} / {f2(q['predicted']['reward']['linear'])}",
                    f"{f2(q['predicted']['punish']['briefed'])} / {f2(q['predicted']['punish']['linear'])}"]
                   for q in e["pairs"]]))
    fc = res.get("freq_ceiling")
    if fc:
        T.append((f"H.4's oracle, the freq ceiling (x_only, {fc['source']}) and this {xm} ceiling",
                  ["engine", "pair", "axis", "H.4 T", "freq T", f"{xm} T"],
                  [[c, label(pk(q)), q["axis"], f2(q["h4"]["testable"]),
                    f2(fc["engines"][c]["testable"].get(label(pk(q)))) if c in fc["engines"] else "-",
                    f2(q[xm]["testable"])] for c, e in E.items() for q in e["pairs"]]))
    return T


def freq_ceiling(path: Path, engines: list):
    """The freq-family ceiling's per engine aggregate and per pair testable (label -> bool) from its summary, for the
    all-family comparison; None when the summary is absent or does not parse."""
    try:
        d = json.loads(Path(path).read_text())
        return dict(source=str(path), engines={
            c: dict(aggregate=d["engines"][c]["aggregate"],
                    testable={label(pk(q)): q["x_only"]["testable"] for q in d["engines"][c]["pairs"]})
            for c in engines if c in d.get("engines", {})})
    except (OSError, ValueError, KeyError, TypeError):
        return None


def findings(res: dict) -> list:
    xm, fam = res.get("mode", "x_only"), res.get("family", "freq")
    fc = res.get("freq_ceiling")
    out = []
    for c, e in res["engines"].items():
        a, h = e["aggregate"], e["h4_aggregate"]
        b, al = e["groups"].get("b"), e["groups"]["all"]
        if a:
            flips = f"; (b) gained {len(b['gained'])}, lost {len(b['lost'])}" if b else ""
            out.append(f"{c}: {xm} T_b {a['testable_b']}/{a['n_b']} ({a['T_b']:.3f}) vs H.4 "
                       f"{h['testable_b']}/{h['n_b']} ({h['T_b']:.3f}); F_a {a['F_a']} vs {h['F_a']}; bar "
                       f"(T_b >= {SPEC.t_b_min}, F_a >= {SPEC.f_a_min}) {a['bar']}{flips}")
            fq = fc["engines"].get(c) if fc else None
            if fq:
                g, P = fq["aggregate"], e["pairs"]
                up = [label(pk(q)) for q in P if q["axis"] == "b" and q[xm]["testable"]
                      and not fq["testable"].get(label(pk(q)))]
                down = [label(pk(q)) for q in P if q["axis"] == "b" and not q[xm]["testable"]
                        and fq["testable"].get(label(pk(q)))]
                out.append(f"{c}: {fam}-family ceiling T_b {a['testable_b']}/{a['n_b']} ({a['T_b']:.3f}) vs freq-family "
                           f"{g['testable_b']}/{g['n_b']} ({g['T_b']:.3f}); F_a {a['F_a']} vs {g['F_a']}; (b) testable "
                           f"only under {fam}: {'; '.join(up) or '-'}; only under freq: {'; '.join(down) or '-'}")
        else:
            out.append(f"{c}: {xm} rows INVALID for combo_stats: {e['reasons']}")
        out.append(f"{c}: median realised Y/X, reward edit {f2(al['yx']['reward']['oracle'])} -> "
                   f"{f2(al['yx']['reward'][xm])}, punishment edit {f2(al['yx']['punish']['oracle'])} -> "
                   f"{f2(al['yx']['punish'][xm])} (oracle -> {xm}); the mask removed a median "
                   f"{f2(al['removed'])} of X's KC activity")
        sp = al["spearman"]
        out.append(f"{c}: Spearman(predicted, realised oracle Y/X) reward {f2(sp['reward']['briefed']['rho'])} "
                   f"briefed / {f2(sp['reward']['linear']['rho'])} linear, punishment "
                   f"{f2(sp['punish']['briefed']['rho'])} / {f2(sp['punish']['linear']['rho'])} "
                   f"(n {sp['reward']['linear']['n']})")
    return out


def md_table(header: list, body: list) -> list:
    cell = lambda v: str(v).replace("|", "\\|")
    return (["| " + " | ".join(map(cell, header)) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(map(cell, r)) + " |" for r in body])


def md_report(res: dict, T: list, F: list) -> str:
    smoke = " — SMOKE, not a measurement" if res["smoke"] else ""
    xm, fam = res.get("mode", "x_only"), res.get("family", "freq")
    how = ("each X-only KC's depression scales with its firing reliability f_x: w <- w0 (1 - alpha f_x)" if fam == "freq"
           else "each X-only KC is depressed by the full alpha regardless of f_x: w <- w0 (1 - alpha), as G.4's O-all")
    L = [f"# M0d H.4a.6 — the oracle's specificity ceiling ({xm} edit, {fam} family){smoke}", "",
         f"Ceiling family: **{fam}** (mode {xm}; spec H.4a.7) — only KCs that fired for X on some act seed and never "
         f"for Y are edited; {how}.", "",
         f"{res['script']} (sha256 {res['script_sha256'][:12]}), commit {res['commit'][:7]}, raw rows {res['raw']}, "
         f"H.4 run {res['h4_run']}; self-check: {res['self_check']}.", "",
         "Descriptive and post hoc; nothing here is a rule or a verdict. " + xm + " changes the oracle's edit, not the "
         "readout or the engine: its T_b bounds what specificity at the edited synapses could buy, it is not an "
         "achievable T_b. "
         "Y/X = mean dV(Y) / mean dV(X) per edit (h4_testability_diag's `gen`); predicted Y/X from the edited "
         "KC->MBON05 (reward) and KC->MBON13 (punishment) weights: briefed = sum w f_x f_y / sum w f_y, linear = "
         "sum w f_x f_y / sum w f_x^2; " + xm + "'s predicted Y/X is 0 by construction.", "", "## Findings", "",
         *[f"- {s}" for s in F], ""]
    for title, header, body in T:
        L += [f"## {title}", "", *md_table(header, body), ""]
    return "\n".join(L)


# ================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="freq",
                    help="freq: mode x_only (default, the original run); all: mode x_only_all (spec H.4a.7)")
    a = ap.parse_args(argv)
    if a.check:
        return run_check()
    if Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    cli, dg = h4cli(), diag()
    spec = cli.smoke_spec(SPEC) if a.smoke else SPEC
    xm = FAMILIES[a.family]                                                # the measured mode
    out, summary_out, draft = out_paths(a.smoke, a.family, a.out)

    # ---- every input checked before the pool starts
    try:
        summary = json.loads(SUMMARY.read_text())
        h4b, h3b = summary["h4"], summary[SPEC.h3_block]
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable blocks h4 / {SPEC.h3_block} in {SUMMARY}: {e!r}")
    if h4b.get("report") != str(RUN) or sha256_file(RUN) != h4b.get("report_sha256"):
        return refuse(f"block h4 of {SUMMARY} is not the recorded run {RUN} (it names {h4b.get('report')})")
    try:
        run, rspec, combos, z_run, h4_rows = dg.load()                    # cache entries sha256-checked
        dg.check_against_report(run, rspec, combos, z_run, h4_rows)
    except SystemExit as e:
        return refuse(f"H.4's recorded rows do not load: {e}")
    readout = {c: h4b["h4"]["combos"][c]["readout"] for c in ENGINES}
    z = {c: {k: tuple(v) for k, v in h4b["h4"]["combos"][c]["z"].items()} for c in ENGINES}
    if any(z[c] != z_run[c] or readout[c] != run["h4"]["combos"][c]["readout"] for c in ENGINES):
        return refuse("block h4's readout / z differ from the run report's")
    pools = h4b["pools"]
    if pools != h3b.get("pools") or pools != run["pools"]:
        return refuse(f"pools differ: block h4 {pools}, block {SPEC.h3_block} {h3b.get('pools')}, run {run['pools']}")
    for c in ENGINES:                                                      # adopted() would restore a missing file
        p = cli.params_from_json(h3b["combos"][c]["adopted"]["params"])
        if p.kc_thresh_mode == "homeostatic" and not (Path(p.kc_thresh_file).exists()
                                                      and sha256_file(p.kc_thresh_file) == p.kc_thresh_sha256):
            return refuse(f"{c}'s threshold file {p.kc_thresh_file} is missing or off its sha256")
    try:
        params, _ = cli.adopted(summary, dataclasses.replace(SPEC, combos=ENGINES))
    except ValueError as e:
        return refuse(str(e))
    if sha256_file(a.npz) != SPEC.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome")
    pops = Populations.from_connectome(Connectome.load(a.npz))
    pairs = even_pairs(pops)
    if pairs_digest(pairs) != SPEC.pairs_digest:
        return refuse(f"the pair list differs from the declared one ({pairs_digest(pairs)[:12]})")
    if any(sorted(map(pair_key, pairs)) != sorted(map(pk, h4_rows[c])) for c in ENGINES):
        return refuse("the pair list differs from H.4's recorded rows")
    first_b = next(p for p in pairs if p["axis"] == "b")
    run_pairs = [first_b] if a.smoke else pairs
    plan = [("oracle", first_b)] + [(xm, p) for p in run_pairs]           # the identity row first: it fails fast
    for c in ENGINES:
        bad = kwargs_mismatch(spec, params[c], readout[c], z[c], pools, run_pairs)
        if bad:
            return refuse(f"{c}: job kwargs differ from h4_measure's for {bad}")

    guard = [Params()] + [params[c] for c in ENGINES]
    git = git_state(files=HASHED_FILES)
    mkey = code_key(a.npz, files=MEASURE_FILES)["key"]
    raw = dict(script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha256_file(__file__),
               commit=git["commit"], dirty_hashed=git["dirty_hashed"], smoke=a.smoke, family=a.family, mode=xm,
               spec=spec, h4_run=str(RUN),
               h4_measure_key=run["measure_key"], measure_key=mkey, engines=list(ENGINES), pools=pools, readout=readout,
               z=z, params=params, pairs_digest=SPEC.pairs_digest, plan=[(m, pair_key(p)) for m, p in plan],
               weights={}, rows={c: [] for c in ENGINES}, wall_s={}, complete=False)
    same = "unchanged" if mkey == run["measure_key"] else "CHANGED"
    print(f"{a.family} family (mode {xm}): {len(plan)} jobs per engine {list(ENGINES)}; H.4 measurement key {same} since run {run['run_id']}; "
          f"dirty hashed files {git['dirty_hashed'] or 'none'}", flush=True)
    for c in ENGINES:
        t0 = time.time()
        jobs = [oracle_kwargs(spec, params[c], readout[c], z[c], pools, p, mode) for mode, p in plan]
        with FlyPool(a.npz, params[c], [{} for _ in range(a.workers)], workers=a.workers, timeout_s=POOL_TIMEOUT_S,
                     punish_type=SPEC.h3.punish_type, reward_type=SPEC.h3.reward_type) as pool:
            raw["weights"][c] = pool.run_jobs(kc_weights_job, [dict(params=params[c], readout=readout[c],
                                                                    punish_type=SPEC.h3.punish_type,
                                                                    reward_type=SPEC.h3.reward_type)])[0]
            n = pool.n_workers
            for r0 in range(0, len(jobs), n):
                res = pool.run_jobs(ceiling_job, jobs[r0:r0 + n])
                raw["rows"][c] += [dict(o, **{k: p[k] for k in PAIR_KEYS}) for (_, p), o in zip(plan[r0:r0 + n], res)]
                write_json(out, raw, guard)
                print(f"{c}: {len(raw['rows'][c])}/{len(jobs)} jobs, {time.time() - t0:.0f} s", flush=True)
                probs = self_check(dg, {c: raw["rows"][c]}, h4_rows, a.smoke, partial=True, xm=xm)
                if probs:                                                  # stop before the remaining rounds
                    print("self-check failed:\n  " + "\n  ".join(probs), file=sys.stderr)
                    return 1
        raw["wall_s"][c] = time.time() - t0
    raw["complete"] = True
    write_json(out, raw, guard)
    probs = self_check(dg, raw["rows"], h4_rows, a.smoke, xm=xm)
    if probs:
        print("self-check failed:\n  " + "\n  ".join(probs), file=sys.stderr)
        return 1
    check = (f"{xm}'s and the oracle row's pre-edit parts and f_x / f_y agree (smoke: no H.4 identity check)"
             if a.smoke else f"the oracle rows equal H.4's cached rows; every {xm} row's pre-edit parts equal H.4's")
    print(f"self-check ok: {check}", flush=True)

    blk = run["h4"]["combos"]
    res = dict(script=raw["script"], script_sha256=raw["script_sha256"], commit=git["commit"], smoke=a.smoke,
               raw=str(out), h4_run=str(RUN), self_check=check, wall_s=raw["wall_s"],
               engines={c: analyse_engine(dg, spec, z[c], raw["rows"][c], h4_rows[c], blk[c], raw["weights"][c], xm)
                        for c in ENGINES})
    if a.family != "freq":                                                 # labelled; freq's summary keys unchanged
        res.update(family=a.family, mode=xm)
        fc = freq_ceiling(out_paths(a.smoke, "freq")[1], list(ENGINES))  # skipped silently when absent
        if fc:
            res["freq_ceiling"] = fc
    T, F = tables(res), findings(res)
    for title, header, body in T:
        dg.show(title, header, body)
    print("\n## Findings\n" + "\n".join(f"- {s}" for s in F))
    write_json(summary_out, dict(res, findings=F), guard)
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_text(md_report(res, T, F) + "\n")
    print(f"\nwrote {out}, {summary_out}, {draft}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
