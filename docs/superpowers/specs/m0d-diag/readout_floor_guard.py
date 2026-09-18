"""M0d H.3a calibration: readout-floor guard for the two candidate APL operating points. Diagnostic only.

Calibration data, not an H.3 selection: no M2 turns, no candidate odours, no oracle. Plasticity stays off —
the naive response is what is being measured.

The declared disparity rule (apl_cap_disparity.py) selects apl_r_max 1.0, but at that point MBON05's naive
response collapses. The guard fixed before this run:

    The operating point must leave at least one MBON type passing H.4's reactivity rule in EACH of the two
    pools (A = PPL105 core MBON types, P = PAM08 core MBON types). If cap 1.0 passes, take 1.0; else 0.333.

H.4 reactivity rule applied verbatim, per MBON *type* (counts summed over all cells of the type, both
hemispheres): median of (read-window spikes - same-seed unstimulated resting spikes) >= 5 AND the share of
presentations with 0 read-window spikes <= 25%. H.4's second criterion (teachability under conditioning) is
NOT evaluated here.

Three configs: C0 (spiking APL, kc_thresh 1.5, no scaling), C1@0.333 (graded, apl_r_max 0.333, kc_thresh 1.5,
apl_input_scale 0.0799) and C1@1.0 (graded, apl_r_max 1.0, kc_thresh 1.25, apl_input_scale 0.09847).
Stimuli: the H.3 reference odour set (48 odours x 2 seeds, strength 0.35, settle 800 ms, read 600 steps),
generator / APL input scaling / FlyPool patterns imported from apl_input_scale_sweep.py and apl_cap_disparity.py.
Resting runs: the same protocol for each of the 96 seeds with no odour (clear_drive, no present).

Writes results/m0d/diag/readout_floor_guard.json (raw) and .md (report); --smoke runs 1 config x 4 odours
into results/m0d/diag/smoke/.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "docs/superpowers/specs/m0d-diag")

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import present

from apl_input_scale_sweep import NPZ, READ_STEPS, SETTLE_MS, STRENGTH, make_engine, reference_odors
from apl_cap_disparity import CAP0_SCALE

CAP1_SCALE = 0.09847            # H.3a accepted s for apl_r_max 1.0, kc_thresh 1.25 (apl_cap_disparity.py)
CAP1_KC_THRESH = 1.25
MED_DELTA_MIN = 5.0             # H.4 reactivity: median(stimulated - same-seed resting) >= 5
ZERO_SHARE_MAX = 0.25           # H.4 reactivity: share of presentations with 0 read-window spikes <= 25%
PUNISH_TYPE, REWARD_TYPE = "PPL105", "PAM08"
CALLOUT = ("MBON05", "MBON13")
CONFIGS = ("C0", "C1@0.333", "C1@1.0")


def mbon_type_index(conn, pops):
    """{MBON type name: indices of all its cells, both hemispheres} over pops.mbon."""
    t = np.asarray(conn.type).astype(str)
    out = {}
    for i in pops.mbon:
        out.setdefault(t[int(i)], []).append(int(i))
    return {n: np.array(sorted(v), np.int64) for n, v in sorted(out.items())}


def pool_types(conn, comps, dan_type):
    """MBON type names that are core targets of `dan_type` (repo definition: compartments(..., core_frac))."""
    t = np.asarray(conn.type).astype(str)
    return sorted({str(t[int(i)]) for i in comps[dan_type].core})


# ---- jobs (module level so spawn can pickle them) ----------------------------------------------
def job_stim(eng, pl, pops, comps, ro, params, scale, odors):
    """Reference-odour presentations: per MBON type, summed read-window spikes over all its cells."""
    e, _ = make_engine(eng, pops, params, scale)
    mt = mbon_type_index(eng.conn, pops)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            for _ in range(READ_STEPS):
                counts[e.step()] += 1
            out.append(dict(odor=o["name"], seed=int(seed), kc_spikes=int(counts[pops.kc].sum()),
                            types={n: int(counts[idx].sum()) for n, idx in mt.items()}))
    return out


def job_rest(eng, pl, pops, comps, ro, params, scale, seeds):
    """Same protocol with no odour (clear_drive, no present): the same-seed unstimulated baseline."""
    e, _ = make_engine(eng, pops, params, scale)
    mt = mbon_type_index(eng.conn, pops)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); e.run(SETTLE_MS)
        counts = np.zeros(e.N, np.int32)
        for _ in range(READ_STEPS):
            counts[e.step()] += 1
        out.append(dict(seed=int(seed), kc_spikes=int(counts[pops.kc].sum()),
                        types={n: int(counts[idx].sum()) for n, idx in mt.items()}))
    return out


# ---- statistics --------------------------------------------------------------------------------
def type_stats(stim, rest_by_seed, name):
    """The five statistics and the pass flag for one MBON type under one config."""
    s = np.array([p["types"][name] for p in stim], float)
    r = np.array([rest_by_seed[p["seed"]]["types"][name] for p in stim], float)
    d = s - r
    med_delta = float(np.median(d))
    zero_share = float((s == 0).mean())
    return dict(median_delta=med_delta, zero_share=zero_share, median_stim=float(np.median(s)),
                median_rest=float(np.median(r)), mean_stim=float(s.mean()), max_stim=float(s.max()),
                n_pres=len(stim), passes=bool(med_delta >= MED_DELTA_MIN and zero_share <= ZERO_SHARE_MAX))


def run_config(pool, params, scale, odors, seeds, chunk):
    jobs = [dict(params=params, scale=scale, odors=odors[c:c + chunk]) for c in range(0, len(odors), chunk)]
    stim = [p for part in pool.run_jobs(job_stim, jobs) for p in part]
    r_chunk = max(1, 2 * chunk)
    rjobs = [dict(params=params, scale=scale, seeds=seeds[c:c + r_chunk]) for c in range(0, len(seeds), r_chunk)]
    rest = [p for part in pool.run_jobs(job_rest, rjobs) for p in part]
    return stim, rest


def mark(ok):
    return "PASS" if ok else "fail"


def pool_table(L, title, types, res, configs):
    L += ["", f"### Pool {title}", "",
          "| MBON type | " + " | ".join(f"{c}: med Δ / zero / med stim / med rest / flag" for c in configs) + " |",
          "|---|" + "---|" * len(configs)]
    for n in types:
        cells = []
        for c in configs:
            st = res[c]["types"][n]
            cells.append(f"{st['median_delta']:.1f} / {st['zero_share']:.2f} / {st['median_stim']:.1f} / "
                         f"{st['median_rest']:.1f} / **{mark(st['passes'])}**")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    return L


def report(res, pools, mbon_types, guard, configs, wall_s, meta):
    L = ["# M0d H.3a calibration: readout-floor guard for the two candidate APL operating points", "",
         f"Diagnostic only (not an H.3 selection; no M2 turns, candidate odours or oracle; plasticity off — the naive "
         f"response is what is measured). Stimuli: the H.3 reference odour set ({meta['n_odors']} odours, rng 800000, "
         f"6-9 glomeruli, DA1/V excluded), strength {STRENGTH}, 2 seeds each, settle {SETTLE_MS} ms, read {READ_STEPS} "
         f"steps. Resting runs: the same protocol for each of the {meta['n_seeds']} seeds with `clear_drive` and no "
         f"`present`; every presentation is paired with the resting run of its own seed. Counts are per MBON *type*, "
         f"summed over all cells of that type in both hemispheres (`conn.type` over `pops.mbon`).", "",
         "## The rule and the guard", "",
         f"> H.4 reactivity rule (applied verbatim): median of (read-window spikes − same-seed unstimulated resting "
         f"spikes) ≥ {MED_DELTA_MIN:g} **and** the share of presentations with 0 read-window spikes ≤ "
         f"{100 * ZERO_SHARE_MAX:g}%.", "",
         "> Readout-floor guard: the operating point must leave at least one MBON type passing that rule in **each** "
         "of the two pools (A = PPL105 core MBON types, P = PAM08 core MBON types). If cap 1.0 passes, take 1.0; "
         "otherwise take 0.333.", "",
         "**H.4's second criterion (teachability under conditioning) is NOT evaluated here** — this run measures the "
         "naive reactivity criterion only.", "",
         "## Pools", "",
         f"Derived with the repo's own definition: `compartments(conn, pops, Params().core_frac)` with core_frac = "
         f"{meta['core_frac']:g} (DAN→MBON synapse mass ≥ that fraction of the type's peak, read from the raw edge "
         f"list), for the same punish/reward DAN types `{PUNISH_TYPE}` / `{REWARD_TYPE}` that `FlyPool` uses. The MBON "
         f"cell indices in each compartment's `core` are mapped to type names through `conn.type`.", "",
         f"- **Pool A ({PUNISH_TYPE} core, {len(pools['A'])} types):** " + (", ".join(pools["A"]) or "none"),
         f"- **Pool P ({REWARD_TYPE} core, {len(pools['P'])} types):** " + (", ".join(pools["P"]) or "none"),
         f"- Pools are disjoint: {not (set(pools['A']) & set(pools['P']))}. "
         f"{len(mbon_types)} MBON types exist in total; {len(set(pools['A']) | set(pools['P']))} are in a pool.", "",
         "## Configurations", "",
         "| config | apl_mode | apl_r_max | kc_thresh | apl_input_scale | median KC read-window spikes (stimulated) |",
         "|---|---|---|---|---|---|"]
    for c in configs:
        p = res[c]["params"]
        L.append(f"| {c} | {p.get('apl_mode', 'spiking')} | {p.get('apl_r_max', Params().apl_r_max):g} | "
                 f"{p.get('kc_thresh', Params().kc_thresh):g} | {res[c]['scale']:.5g} | {res[c]['median_kc']:.0f} |")
    L += ["", "## Per-type reactivity", "",
          "Each cell is: median(stimulated − same-seed resting) / zero-share of read-window spikes / median "
          "stimulated / median resting / pass flag."]
    pool_table(L, f"A — {PUNISH_TYPE} core", pools["A"], res, configs)
    pool_table(L, f"P — {REWARD_TYPE} core", pools["P"], res, configs)

    L += ["", "## The G.14.8 readout: MBON05 and MBON13", ""]
    for n in CALLOUT:
        where = [k for k in ("A", "P") if n in pools[k]]
        loc = (f"in pool {' and '.join(where)}" if where else
               f"in **neither** pool (not a core target of {PUNISH_TYPE} or {REWARD_TYPE})")
        if n not in mbon_types:
            L.append(f"- **{n}**: not present as an MBON type in this connectome.")
            continue
        L.append(f"- **{n}** ({loc}): " + "; ".join(
            f"{c} median Δ {res[c]['types'][n]['median_delta']:.1f}, zero-share "
            f"{res[c]['types'][n]['zero_share']:.2f}, median stimulated {res[c]['types'][n]['median_stim']:.1f}, "
            f"median resting {res[c]['types'][n]['median_rest']:.1f} → {mark(res[c]['types'][n]['passes'])}"
            for c in configs) + ".")

    L += ["", "## Guard summary", "",
          "| config | passing types in A | passing types in P | guard |", "|---|---|---|---|"]
    for c in configs:
        g = guard["per_config"][c]
        L.append(f"| {c} | {g['n_pass_A']}/{len(pools['A'])}"
                 + (f" ({', '.join(g['pass_A'])})" if g["pass_A"] else "")
                 + f" | {g['n_pass_P']}/{len(pools['P'])}"
                 + (f" ({', '.join(g['pass_P'])})" if g["pass_P"] else "")
                 + f" | {'PASS' if g['passes'] else 'FAIL'} |")
    L += ["", f"- Cap 1.0 (C1@1.0) {'PASSES' if guard['cap10_passes'] else 'does NOT pass'} the readout-floor guard."]
    if "C1@0.333" in guard["per_config"]:
        L.append(f"- Cap 0.333 (C1@0.333) {'passes' if guard['per_config']['C1@0.333']['passes'] else 'does not pass'}"
                 f" it (reported for completeness; the guard only asks about 1.0).")
    if "C0" in guard["per_config"]:
        L.append(f"- C0 (spiking-APL baseline) {'passes' if guard['per_config']['C0']['passes'] else 'does not pass'}"
                 f" it (reference point, not a candidate).")
    L += ["",
          f"- **Verdict: the selected `apl_r_max` is {guard['selected']:g}** — {guard['reason']}.", "",
          f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min). Commit {meta['commit']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    core_frac = Params().core_frac
    comps = compartments(conn, pops, core_frac)
    pools = dict(A=pool_types(conn, comps, PUNISH_TYPE), P=pool_types(conn, comps, REWARD_TYPE))
    mbon_types = sorted(mbon_type_index(conn, pops))
    odors_all = reference_odors(pops)
    del conn

    all_configs = {
        "C0": dict(params=dict(kc_thresh=1.5), scale=1.0),
        "C1@0.333": dict(params=dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), scale=CAP0_SCALE),
        "C1@1.0": dict(params=dict(apl_mode="graded", apl_r_max=1.0, kc_thresh=CAP1_KC_THRESH), scale=CAP1_SCALE),
    }
    if smoke:
        configs, odors = ("C1@1.0",), odors_all[:4]
        chunk, workers, out_dir = 2, 4, "results/m0d/diag/smoke"
    else:
        configs, odors = CONFIGS, odors_all
        chunk, workers, out_dir = 6, 16, "results/m0d/diag"
    seeds = [int(s) for o in odors for s in o["seeds"]]

    print(f"pools: A ({PUNISH_TYPE} core, core_frac {core_frac:g}) = {pools['A']}")
    print(f"pools: P ({REWARD_TYPE} core, core_frac {core_frac:g}) = {pools['P']}")
    print(f"{len(mbon_types)} MBON types; {len(odors)} odours, {len(seeds)} presentations and {len(seeds)} "
          f"resting runs per config; {len(configs)} configs")

    res = {}
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers,
                 punish_type=PUNISH_TYPE, reward_type=REWARD_TYPE) as pool:
        for name in configs:
            c = all_configs[name]
            t1 = time.time()
            stim, rest = run_config(pool, c["params"], c["scale"], odors, seeds, chunk)
            rest_by_seed = {p["seed"]: p for p in rest}
            assert set(rest_by_seed) == set(seeds), "resting runs must cover every presentation seed"
            res[name] = dict(params=c["params"], scale=float(c["scale"]),
                             types={n: type_stats(stim, rest_by_seed, n) for n in mbon_types},
                             median_kc=float(np.median([p["kc_spikes"] for p in stim])),
                             median_kc_rest=float(np.median([p["kc_spikes"] for p in rest])),
                             stim=stim, rest=rest, config_s=time.time() - t1)
            print(f"{name}: {time.time() - t1:.0f} s")

    guard = dict(per_config={})
    for name in configs:
        pa = [n for n in pools["A"] if res[name]["types"][n]["passes"]]
        pp = [n for n in pools["P"] if res[name]["types"][n]["passes"]]
        guard["per_config"][name] = dict(pass_A=pa, pass_P=pp, n_pass_A=len(pa), n_pass_P=len(pp),
                                         passes=bool(pa and pp))
    cap10 = guard["per_config"].get("C1@1.0", dict(passes=False))["passes"]
    guard["cap10_passes"] = bool(cap10)
    guard["selected"] = 1.0 if cap10 else 0.333
    guard["reason"] = ("cap 1.0 leaves at least one passing MBON type in each pool, so the guard takes 1.0"
                       if cap10 else
                       "cap 1.0 leaves a pool with no passing MBON type, so the guard falls back to 0.333")

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(npz=NPZ, smoke=smoke, commit=commit, strength=STRENGTH, settle_ms=SETTLE_MS, read_steps=READ_STEPS,
                n_odors=len(odors), n_seeds=len(seeds), core_frac=float(core_frac), punish_type=PUNISH_TYPE,
                reward_type=REWARD_TYPE, med_delta_min=MED_DELTA_MIN, zero_share_max=ZERO_SHARE_MAX,
                workers=workers, wall_s=wall, teachability_evaluated=False)
    out = dict(meta=meta, pools=pools, mbon_types=mbon_types, configs=res, guard=guard, odors=odors)
    md = report(res, pools, mbon_types, guard, configs, wall, meta)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/readout_floor_guard.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open(f"{out_dir}/readout_floor_guard.md", "w") as f:
        f.write(md)
    print(md)
    print(f"wall {wall:.1f} s; {len(configs) * len(seeds)} stimulated + {len(configs) * len(seeds)} resting "
          f"presentations on {workers} workers")
