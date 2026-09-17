"""Spec appendix G.14 — engine-premise probe. Calibration: no D.6 (c) verdict, even turns only.

Arms: G (graded APL g = 0.05, kc_thresh 1.5), S-match (spiking, kc_thresh k* matched to G's KC activity),
S-ref (spiking, 1.5). Stages, each resumable on its own key: match (G.14.2) -> G -> S-match -> S-ref oracles (G.14.3).
The verdict is not computed here: write_engine_probe_summary.py derives it from the raw files.

Run from the repo root:
  PYTHONPATH=docs/superpowers/specs/m2-calibration-g uv run python docs/superpowers/specs/m2-calibration-g/m2_engine_probe.py
Smoke (G.14.7): ... --smoke --out <scratch dir>   (1 pair per arm, 1 grid point, 2 seeds per block)
"""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, os, time
from collections import deque
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.presentation import decide
from flymon.brain.stimuli import present
import graded_apl, m2_encoder_compare, m2_probe
from engine_probe_verdict import dprime, dv
from graded_apl import make_graded
from m2_encoder_compare import build_encoder, pairs_for
from m2_probe import NPZ, READ_MS, SETTLE_MS, build_turns, pool_vocabulary

HERE = Path(__file__).resolve().parent
STRENGTH, ALPHAS, GAIN = 0.35, [0.2, 0.5, 0.8], 0.05
ACT_SEEDS, SELECT_SEEDS, REPORT_SEEDS = list(range(500, 508)), list(range(600, 608)), list(range(608, 616))
BASE_KC_THRESH = 1.5
GRID = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
MATCH_TOL_PP = 0.5
WINDOW_MS = 200
ARMS = {"G": {"mode": "graded", "kc_thresh": 1.5}, "S-match": {"mode": "spiking", "kc_thresh": None},
        "S-ref": {"mode": "spiking", "kc_thresh": 1.5}}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- worker side ----------------------------------------------------------------------------------------------
def configure(eng, pops, mode: str, kc_thresh: float) -> None:
    """Idempotent per worker: graded conversion (never undone) and KC threshold = base * kc_thresh / 1.5."""
    if mode == "graded":
        make_graded(eng, pops, GAIN)
    elif getattr(eng, "_graded", False):
        raise RuntimeError("spiking job on a worker whose engine was converted to graded APL")
    if not hasattr(eng, "_vth_base"):
        eng._vth_base = eng.v_th.copy()
    eng.v_th[pops.kc] = eng._vth_base[pops.kc] * np.float32(kc_thresh / BASE_KC_THRESH)


def present_kc(eng, pl, pops, odor, seed) -> dict:
    """One presentation (settle + read). KC counts of the read window, and the max 200 ms sliding-window KC count
    over the whole presentation (G.8 definition, descriptive here)."""
    kc = pops.kc
    pos = np.full(eng.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    eng.reset(seed); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
    present(eng, pops, odor, STRENGTH)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32)
    n_settle, n_total = int(round(SETTLE_MS / eng.p.dt)), int(round((SETTLE_MS + READ_MS) / eng.p.dt))
    apl_r, apl_vmax = [], 0.0
    for step in range(n_total):
        fired = eng.step()
        f = pos[fired]; f = f[f >= 0]
        win[f] += 1; hist.append(f)
        if len(hist) > WINDOW_MS:
            win[hist.popleft()] -= 1
        if f.size:
            max_win = max(max_win, int(win.max()))
        if step >= n_settle:
            read[f] += 1
            if getattr(eng, "_graded", False):
                apl_vmax = max(apl_vmax, float(np.max(eng.v[eng._apl_idx])))
    if getattr(eng, "_graded", False):
        apl_r = eng._apl_r_log[n_settle:]
    return {"read": read, "max_win": max_win,
            "apl_hz": float(np.mean(apl_r) * 1000.0) if len(apl_r) else None, "apl_vmax": apl_vmax if apl_r else None}


def activity_job(eng, pl, pops, comps, ro, odor: dict, mode: str, kc_thresh: float, seeds: list) -> dict:
    configure(eng, pops, mode, kc_thresh)
    try:
        pl.reset_weights(); pl.set_enabled(False)
        out = [present_kc(eng, pl, pops, odor, s) for s in seeds]
        return {"frac": [float((o["read"] > 0).mean()) for o in out], "spikes": [int(o["read"].sum()) for o in out],
                "max_win": [o["max_win"] for o in out]}
    finally:
        pl.reset_weights(); pl.set_enabled(True)


def oracle_job(eng, pl, pops, comps, ro, odor_x: dict, odor_y: dict, mode: str, kc_thresh: float,
               act_seeds: list, select_seeds: list, report_seeds: list) -> dict:
    configure(eng, pops, mode, kc_thresh)
    t = eng.conn.type
    c13, c05 = np.flatnonzero(t == "MBON13"), np.flatnonzero(t == "MBON05")
    idx = np.concatenate([c13, c05]); n13 = len(c13)
    try:
        pl.reset_weights(); pl.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win, apl_hz, apl_vmax = [], [], [], [], []
            for s in act_seeds:
                o = present_kc(eng, pl, pops, odor, s)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"]); apl_hz.append(o["apl_hz"]); apl_vmax.append(o["apl_vmax"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win,
                                            "apl_hz": apl_hz, "apl_vmax": apl_vmax}

        def probe(seeds):
            A, P = [], []
            for s in seeds:
                c = decide(eng, pl, pops, [odor_x, odor_y], STRENGTH, s, SETTLE_MS, READ_MS, idx=idx)
                A.append(c[:, :n13].sum(1).tolist()); P.append(c[:, n13:].sum(1).tolist())
            return {"A": A, "P": P}

        fx, kc_x = activity(odor_x)
        _fy, kc_y = activity(odor_y)
        mb = pl.mb_local
        rew = np.isin(pl.post_mb, mb[comps["PAM08"].core]); pun = np.isin(pl.post_mb, mb[comps["PPL105"].core])
        w = eng.csc.w

        def set_w(a_r, a_p=None):
            wv = pl.w0.copy()
            if a_r is not None:
                wv[rew] = pl.w0[rew] * (1.0 - a_r * fx)[pl.pre_kc[rew]]
            if a_p is not None:
                wv[pun] = pl.w0[pun] * (1.0 - a_p * fx)[pl.pre_kc[pun]]
            w[pl.edges] = wv

        # selection on select_seeds: reward change max, then punishment change min (ties -> smaller alpha)
        set_w(None); pre_sel = probe(select_seeds)
        reward = {}
        for a in ALPHAS:
            set_w(a); r1 = probe(select_seeds)
            reward[str(a)] = {"R1": r1, "change": dprime(dv(r1) - dv(pre_sel))}
        a_r = max(ALPHAS, key=lambda a: (reward[str(a)]["change"], -a))
        punish = {}
        for a in ALPHAS:
            set_w(a_r, a); r2 = probe(select_seeds)
            punish[str(a)] = {"R2": r2, "change": dprime(dv(r2) - dv(reward[str(a_r)]["R1"]))}
        a_p = min(ALPHAS, key=lambda a: (punish[str(a)]["change"], a))
        # report on fresh seeds at the chosen depths
        set_w(None); pre = probe(report_seeds)
        set_w(a_r); R1 = probe(report_seeds)
        set_w(a_r, a_p); R2 = probe(report_seeds)
        w[pl.edges] = pl.w0
        return {"select": {"pre": pre_sel, "reward": reward, "punish": punish},
                "alpha_reward": a_r, "alpha_punish": a_p,
                "report": {"pre": pre, "R1": R1, "R2": R2},
                "kc": {"x": kc_x, "y": kc_y,
                       "jaccard": float(((fx > 0) & (_fy > 0)).sum() / max(((fx > 0) | (_fy > 0)).sum(), 1))}}
    finally:
        pl.reset_weights(); pl.set_enabled(True)


# ---- parent side ---------------------------------------------------------------------------------------------
def odor_key(odor: dict) -> str:
    return json.dumps(sorted(odor.items()))


def code_key(extra: dict) -> dict:
    brain = sorted(Path("flymon/brain").glob("*.py"))
    return {"script": sha256(__file__), "graded_apl": sha256(graded_apl.__file__),
            "m2_encoder_compare": sha256(m2_encoder_compare.__file__), "m2_probe": sha256(m2_probe.__file__),
            "flymon_brain": {p.name: sha256(p) for p in brain}, "npz": sha256(NPZ),
            "params": dataclasses.asdict(Params()), **extra}


def load_if(path: Path, key: dict):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d if d.get("key") == key and d.get("complete") else None


def write_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def median_frac(results) -> float:
    return float(np.median([f for r in results for f in r["frac"]]))


def pick_among(medians: dict, target: float, ks) -> float:
    """Closest median to the target; ties -> smaller kc_thresh."""
    return min(ks, key=lambda k: (abs(medians[k] - target), k))


def pick_k_star(medians: dict, target: float, grid) -> tuple[float, tuple | None]:
    """G.14.2: nearest grid point; if it misses by more than the tolerance, also the first pair of neighbouring grid
    points (in kc_thresh order) whose medians straddle the target, else None (target outside the grid)."""
    k = pick_among(medians, target, grid)
    if abs(100 * (medians[k] - target)) <= MATCH_TOL_PP:
        return k, None
    ks = sorted(grid)
    br = next(((lo, hi) for lo, hi in zip(ks, ks[1:]) if (medians[lo] - target) * (medians[hi] - target) <= 0), None)
    return k, br


def run_match(odors: list, out: Path, workers: int, grid: list, seeds: list, npz_key: dict) -> dict:
    key = code_key({"stage": "match", "grid": grid, "seeds": seeds, "gain": GAIN, "odors": [odor_key(o) for o in odors],
                    **npz_key})
    prev = load_if(out / "match.json", key)
    if prev:
        print(f"match resumed: k* {prev['k_star']}  diff {prev['diff_pp']:+.2f} pp", flush=True)
        return prev
    t0 = time.perf_counter()
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers, timeout_s=7200.0) as pool:
        tgt = pool.run_jobs(activity_job, [dict(odor=o, mode="graded", kc_thresh=1.5, seeds=seeds) for o in odors])
    target = median_frac(tgt)
    print(f"match: G target median {100 * target:.2f}%  ({time.perf_counter() - t0:.0f}s)", flush=True)

    def sweep(pool, k):
        res = pool.run_jobs(activity_job, [dict(odor=o, mode="spiking", kc_thresh=k, seeds=seeds) for o in odors])
        med = median_frac(res)
        print(f"match: spiking kc_thresh {k:.4f} median {100 * med:.2f}%", flush=True)
        return {"median": med, "results": res}

    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers, timeout_s=7200.0) as pool:
        points = {k: sweep(pool, k) for k in grid}
        diff = lambda k: 100 * (points[k]["median"] - target)
        k_star, br = pick_k_star({k: v["median"] for k, v in points.items()}, target, grid)
        bisect = None
        if br is not None:
            mid = (br[0] + br[1]) / 2
            points[mid] = sweep(pool, mid)
            bisect = {"bracket": list(br), "mid": mid}
            k_star = pick_among({k: v["median"] for k, v in points.items()}, target, [br[0], br[1], mid])
    res = {"key": key, "complete": True, "target": {"median": target, "results": tgt},
           "points": {str(k): v for k, v in points.items()}, "bisect": bisect, "k_star": k_star, "diff_pp": diff(k_star),
           "seconds": time.perf_counter() - t0}
    write_atomic(out / "match.json", res)
    print(f"match: k* {k_star}  diff {res['diff_pp']:+.2f} pp  ok {abs(res['diff_pp']) <= MATCH_TOL_PP}", flush=True)
    return res


def run_arm(arm: str, kc_thresh: float, pairs: list, out: Path, workers: int, seeds: dict, npz_key: dict) -> dict:
    cfg = {"arm": arm, "mode": ARMS[arm]["mode"], "kc_thresh": kc_thresh, "gain": GAIN if ARMS[arm]["mode"] == "graded" else None}
    key = code_key({"stage": "oracle", **cfg, "seeds": seeds,
                    "pairs": [[p["axis"], p["turn"], p["x"], p["y"], odor_key(p["odor_x"]), odor_key(p["odor_y"])] for p in pairs],
                    **npz_key})
    prev = load_if(out / f"{arm}.json", key)
    if prev:
        print(f"{arm} resumed ({len(prev['rows'])} rows)", flush=True)
        return prev
    t0 = time.perf_counter()
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers, timeout_s=7200.0) as pool:
        res = pool.run_jobs(oracle_job, [dict(odor_x=p["odor_x"], odor_y=p["odor_y"], mode=cfg["mode"], kc_thresh=kc_thresh,
                                              act_seeds=seeds["act"], select_seeds=seeds["select"], report_seeds=seeds["report"])
                                         for p in pairs])
    rows = [{k: p[k] for k in ("axis", "turn", "x", "y")} | r for p, r in zip(pairs, res)]
    d = {"key": key, "complete": True, **cfg, "rows": rows, "seconds": time.perf_counter() - t0}
    write_atomic(out / f"{arm}.json", d)
    print(f"{arm} done {d['seconds']:.0f}s ({len(rows)} pairs)", flush=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m2/calibration/engine_probe")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="G.14.7 smoke: 1 pair per arm, 1 grid point, 2 seeds per block")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    out = Path(a.out)
    if out.resolve() == Path.cwd().resolve():
        raise SystemExit("refusing to write into the repo root")
    out.mkdir(parents=True, exist_ok=True)
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    turns = [t for t in build_turns(species_types, move_info, 16) if t["turn"] % 2 == 0]
    chan = build_encoder("E0", pops, mon_types, move_types)
    pairs = pairs_for("E0", pops, chan, turns, species_types)
    del conn
    if any(p["turn"] % 2 for p in pairs):
        raise SystemExit("odd turn in the pair list")
    seeds = {"act": ACT_SEEDS, "select": SELECT_SEEDS, "report": REPORT_SEEDS}
    grid = GRID
    if a.smoke:
        pairs = [next(p for p in pairs if p["axis"] == "b")]
        seeds = {k: v[:2] for k, v in seeds.items()}
        grid = GRID[-1:]
    print(f"{sum(p['axis'] == 'a' for p in pairs)} (a) + {sum(p['axis'] == 'b' for p in pairs)} (b) pairs, "
          f"smoke={a.smoke}, out={out}", flush=True)
    npz_key = {"smoke": a.smoke}
    odors = list({odor_key(o): o for p in pairs for o in (p["odor_x"], p["odor_y"])}.values())
    match = run_match(odors, out, a.workers, grid, seeds["act"], npz_key)
    if abs(match["diff_pp"]) > MATCH_TOL_PP and not a.smoke:
        print("activity match failed: the verdict will be INVALID; oracle arms still run for the record", flush=True)
    for arm in a.arms:
        k = match["k_star"] if arm == "S-match" else ARMS[arm]["kc_thresh"]
        run_arm(arm, k, pairs, out, a.workers, seeds, npz_key)
    write_atomic(out / "plan.json", {"pairs": [[p["axis"], p["turn"], p["x"], p["y"]] for p in pairs],
                                     "seeds": seeds, "grid": grid, "smoke": a.smoke})
    print("done", flush=True)


if __name__ == "__main__":
    main()
