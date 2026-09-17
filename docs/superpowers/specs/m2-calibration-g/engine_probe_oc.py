"""Operating characteristic of the G.14 improvement test, computed before the probe runs.

Noise structure: axis-(b) reward change r = d'(dV_R1 - dV_pre), max over alpha, clipped to +-10, from the G.12
E0 even-turn raw data (seeds 600-607). One-way random effects by turn (method of moments) gives sigma_turn and
sigma_pair. Simulated paired deltas on the real (b) turn structure:
    delta_i = mu + u_turn + e_i,   u ~ N(0, k * sigma_turn^2),   e ~ N(0, k * sigma_pair^2)
k = 2 for independent arms, k = 1 for arms correlated at 0.5. The false-improvement rate at mu = 0 does not depend
on k (both tests are scale-invariant); power does.

Models: "primary" (estimated ICC) and "icc_x2" (turn variance share doubled, capped at 0.9, or 0.2 if the
estimate is 0; total variance kept).
Switch rule (G.14, declared before this runs): if P(improve | mu = 0) > 0.10 under either model for the bootstrap,
the verdict uses the turn sign-flip test instead.

Usage: python engine_probe_oc.py --raw results/m2/calibration/encoders/E0_even.json --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("engine_probe_verdict", HERE / "engine_probe_verdict.py")
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)

MUS = (0.0, 0.5, 1.0, 2.0)
N_SIM = 2000
N_BOOT_OC = 2000
SIM_SEED = 917
SWITCH_FP = 0.10


def reward_changes(raw_path: str) -> tuple[list[int], list[float]]:
    rows = [r for r in json.load(open(raw_path))["rows"] if r["axis"] == "b"]
    if any(not V.turn_ok(r["turn"]) for r in rows):
        raise ValueError("odd-turn rows in the noise source")
    turns, vals = [], []
    for r in sorted(rows, key=lambda r: (r["turn"], r["x"], r["y"])):
        pre = V.dv(r["pre"])
        best = max(V.dprime(V.dv(v["R1"]) - pre) for v in r["reward"].values())
        turns.append(r["turn"]); vals.append(V.clip(best))
    return turns, vals


def variance_components(turns, vals) -> dict:
    t = np.asarray(turns); x = np.asarray(vals, float)
    groups = [x[t == u] for u in sorted(set(turns))]
    n_i = np.array([len(g) for g in groups], float); N, G = n_i.sum(), len(groups)
    grand = x.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (G - 1), ssw / (N - G)
    n0 = (N - (n_i ** 2).sum() / N) / (G - 1)
    s_turn2 = max((msb - msw) / n0, 0.0)
    return {"sigma_turn": float(np.sqrt(s_turn2)), "sigma_pair": float(np.sqrt(msw)),
            "icc": float(s_turn2 / (s_turn2 + msw)) if s_turn2 + msw > 0 else 0.0, "n_pairs": int(N), "n_turns": G}


def simulate(turns, sigma_turn, sigma_pair, mu, k, rng) -> dict:
    ut = sorted(set(turns)); t_idx = np.array([ut.index(t) for t in turns])
    boot_hits = flip_hits = 0
    for s in range(N_SIM):
        u = rng.normal(0, np.sqrt(k) * sigma_turn, len(ut))
        d = mu + u[t_idx] + rng.normal(0, np.sqrt(k) * sigma_pair, len(turns))
        boot_hits += V.turn_bootstrap(turns, d.tolist(), n_boot=N_BOOT_OC, seed=int(rng.integers(2**31)))["improve"]
        flip_hits += V.turn_signflip(turns, d.tolist())["improve"]
    return {"bootstrap": boot_hits / N_SIM, "signflip": flip_hits / N_SIM}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    turns, vals = reward_changes(a.raw)
    vc = variance_components(turns, vals)
    total2 = vc["sigma_turn"] ** 2 + vc["sigma_pair"] ** 2
    icc2 = min(2 * vc["icc"], 0.9) if vc["icc"] > 0 else 0.2
    models = {"primary": (vc["sigma_turn"], vc["sigma_pair"]),
              "icc_x2": (float(np.sqrt(icc2 * total2)), float(np.sqrt((1 - icc2) * total2)))}
    rng = np.random.default_rng(SIM_SEED)
    oc = {}
    for name, (st, sp) in models.items():
        oc[name] = {f"k{k}": {str(mu): simulate(turns, st, sp, mu, k, rng) for mu in MUS} for k in (2, 1)}
    fp = {name: oc[name]["k2"]["0.0"]["bootstrap"] for name in models}
    chosen = "signflip" if any(v > SWITCH_FP for v in fp.values()) else "bootstrap"
    out = {"source": a.raw, "source_sha256": hashlib.sha256(open(a.raw, "rb").read()).hexdigest(),
           "verdict_sha256": hashlib.sha256((HERE / "engine_probe_verdict.py").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "n_sim": N_SIM, "n_boot": N_BOOT_OC, "sim_seed": SIM_SEED, "switch_fp": SWITCH_FP,
           "variance_components": vc, "models": {k: {"sigma_turn": v[0], "sigma_pair": v[1]} for k, v in models.items()},
           "oc": oc, "bootstrap_false_improve_at_mu0": fp, "chosen_test": chosen}
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"variance_components": vc, "false_improve_mu0": fp, "chosen_test": chosen}, indent=1))


if __name__ == "__main__":
    main()
