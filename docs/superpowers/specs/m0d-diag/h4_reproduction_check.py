"""M0d H.4 reproduction check (plan step R2): the committed H.4 worker jobs against recorded numbers, before the H.4 run.

1. `h4_jobs.teach_job` on C0 (`Params()`), order "ab", both single-channel arms, seeds 8-15: the PPL105-core and
   PAM08-core sums of its per-type counts equal the A/P probe counts recorded in results/m0c/conditioning.json.
2. `h4_jobs.teach_job` on the three adopted engines (block "h3"), both orders, both arms, seeds 8-15 (96 arms): every
   pool type's pre/post counts equal the H.4a.1 diagnostic, results/m0d/diag/h4_teach_odour.json (git-ignored).
3. `h4_jobs.oracle_job` on C0 with readout MBON13 / MBON05 and F.3's frozen constants, the first --pairs pairs of each
   axis: the selection-seed probes (pre, and R1 at every alpha; seeds 600-607) and the KC activity (seeds 500-507)
   equal G.12's E0 rows, results/m2/calibration/encoders/E0_even.json (git-ignored).
4. The same job on G.14's smoke pair and seed blocks (500-501, 600-601, 608-609): the selection probes and changes,
   both alphas, the report pre / R1 / R2 and the KC records equal G.14's S-ref smoke row,
   results/m2/calibration/engine_probe/smoke/S-ref.json (git-ignored; driver sha256 81ad9e82..., spiking, kc_thresh 1.5
   = C0) — the punishment choice and the report seeds, which the verdict reads, against a recorded run.
All bit for bit. Exit 0 only when everything matches. Writes nothing.

    uv run python docs/superpowers/specs/m0d-diag/h4_reproduction_check.py [--workers 16] [--pairs 2]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from flymon.brain import h4_jobs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h4_pairs import even_pairs, pair_key
from flymon.brain.h4_spec import SPEC

NPZ = "data/malecns.npz"
M0C = Path("results/m0c/conditioning.json")
TEACH_DIAG = Path("results/m0d/diag/h4_teach_odour.json")
G12 = Path("results/m2/calibration/encoders/E0_even.json")
SREF = Path("results/m2/calibration/engine_probe/smoke/S-ref.json")
VERDICT = Path("docs/superpowers/specs/m2-calibration-g/engine_probe_verdict.py")
VERDICT_SHA = "c9c81ab97d13e7bc163210564e1114d3caafd60c63431bbef8eb977f8fb334d7"      # spec G.14


def f3_z() -> dict:
    """F.3's frozen constants, from the frozen G.14 verdict module (sha256-checked), not restated."""
    if hashlib.sha256(VERDICT.read_bytes()).hexdigest() != VERDICT_SHA:
        raise SystemExit(f"{VERDICT} is not the frozen G.14 verdict module")
    s = importlib.util.spec_from_file_location("engine_probe_verdict", VERDICT)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return {k: tuple(v) for k, v in m.Z.items()}


def params_from_json(d: dict) -> Params:
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=2)
    a = ap.parse_args()
    s, h3 = SPEC, SPEC.h3
    block = json.loads(Path("results/summary/m0d.json").read_text())["h3"]
    pools = block["pools"]
    types = tuple(pools["A"] + pools["P"])
    combos = {n: params_from_json(block["combos"][n]["adopted"]["params"]) for n in s.combos}
    pops = Populations.from_connectome(Connectome.load(NPZ))
    pairs = even_pairs(pops)
    pick = [p for ax in ("a", "b") for p in [q for q in pairs if q["axis"] == ax][:a.pairs]]
    sref = json.loads(SREF.read_text())["rows"][0]
    sref_pair = next(p for p in pairs if pair_key(p) == (sref["axis"], sref["turn"], sref["x"], sref["y"]))
    teach = dict(types=types, punish_type=h3.punish_type, reward_type=h3.reward_type, k=h3.design_k,
                 odor_seed=h3.design_odor_seed, strength=h3.strength, trials=s.teach_trials,
                 present_ms=s.teach_present_ms, gap_ms=s.teach_gap_ms, settle_ms=s.teach_window.settle_ms,
                 read_ms=s.teach_window.read_ms)
    oracle = dict(params=Params(), readout={"A": "MBON13", "P": "MBON05"}, z=f3_z(), types=types,
                  act_seeds=tuple(s.act_seeds), select_seeds=tuple(s.select_seeds), report_seeds=tuple(s.report_seeds),
                  alphas=tuple(s.oracle_alphas), strength=h3.strength, settle_ms=s.oracle_window.settle_ms,
                  read_ms=s.oracle_window.read_ms, window_ms=s.kc_window_ms, punish_type=h3.punish_type,
                  reward_type=h3.reward_type)
    smoke = dict(oracle, act_seeds=(500, 501), select_seeds=(600, 601), report_seeds=(608, 609))
    t_rows = {}
    with FlyPool(NPZ, Params(), [{} for _ in range(a.workers)], workers=a.workers, timeout_s=3600.0) as pool:
        for name, p in combos.items():                  # one combination at a time: each worker builds one rig per Params
            t_rows[name] = pool.run_jobs(h4_jobs.teach_job, [dict(teach, params=p, seed=seed, arm=arm, order=order)
                                                             for order in ("ab", "ba") for arm in ("punish_only", "reward_only")
                                                             for seed in s.teach_seeds])
        o_rows = pool.run_jobs(h4_jobs.oracle_job, [dict(oracle, odor_x=p["odor_x"], odor_y=p["odor_y"]) for p in pick]
                               + [dict(smoke, odor_x=sref_pair["odor_x"], odor_y=sref_pair["odor_y"])])
    bad = []
    m0c = json.loads(M0C.read_text())["per_seed"]
    for r in t_rows["C0"]:
        if r["order"] != "ab":
            continue
        ref = m0c[str(r["seed"])][r["arm"]]["counts"]
        for ph in ("pre", "post"):
            for cs in ("plus", "minus"):
                got = (sum(r[ph][cs][t] for t in pools["A"]), sum(r[ph][cs][t] for t in pools["P"]))
                if got != (ref[f"{ph}_{cs}"]["A"], ref[f"{ph}_{cs}"]["P"]):
                    bad.append(f"M0c: seed {r['seed']} {r['arm']} {ph}_{cs}: {got}")
    diag = json.loads(TEACH_DIAG.read_text())
    for name, rows in t_rows.items():
        want = {(d["seed"], d["arm"], d["order"]): d for d in diag[name]}
        for r in rows:
            d = want[(r["seed"], r["arm"], r["order"])]
            for ph in ("pre", "post"):
                for cs in ("plus", "minus"):
                    if any(r[ph][cs][t] != d[ph][cs][t] for t in types):
                        bad.append(f"H.4a.1 diagnostic: {name} seed {r['seed']} {r['arm']} {r['order']} {ph}_{cs}")
    print(f"teach: {sum(len(v) for v in t_rows.values())} arms against the H.4a.1 diagnostic, 16 of them against M0c")
    g12 = {(r["axis"], r["turn"], r["x"], r["y"]): r for r in json.loads(G12.read_text())["rows"]}
    for p, r in zip(pick, o_rows[:-1]):
        g, sel = g12[pair_key(p)], r["select"]
        checks = {"pre": sel["pre"]["MBON13"] == g["pre"]["A"] and sel["pre"]["MBON05"] == g["pre"]["P"],
                  "R1": all(sel["reward"][k]["R1"]["MBON13"] == g["reward"][k]["R1"]["A"]
                            and sel["reward"][k]["R1"]["MBON05"] == g["reward"][k]["R1"]["P"] for k in g["reward"]),
                  "kc": r["kc"]["x"]["frac"] == g["kc"]["frac_x"] and r["kc"]["y"]["frac"] == g["kc"]["frac_y"]
                  and r["kc"]["x"]["spikes"] == g["kc"]["spikes_x"] and r["kc"]["y"]["spikes"] == g["kc"]["spikes_y"]}
        bad += [f"G.12 {pair_key(p)}: {k}" for k, ok in checks.items() if not ok]
        print(f"G.12 {pair_key(p)}: {checks}")
    r, ap_ = o_rows[-1], lambda pr: {"A": pr["MBON13"], "P": pr["MBON05"]}
    sel = r["select"]
    checks = {"alphas": (r["alpha_reward"], r["alpha_punish"]) == (sref["alpha_reward"], sref["alpha_punish"]),
              "select pre": ap_(sel["pre"]) == sref["select"]["pre"],
              "select reward": all(ap_(sel["reward"][k]["R1"]) == v["R1"] and sel["reward"][k]["change"] == v["change"]
                                   for k, v in sref["select"]["reward"].items()),
              "select punish": all(ap_(sel["punish"][k]["R2"]) == v["R2"] and sel["punish"][k]["change"] == v["change"]
                                   for k, v in sref["select"]["punish"].items()),
              "report": r["report"] == sref["report"],
              "kc": all(r["kc"][c][f] == sref["kc"][c][f] for c in ("x", "y") for f in ("frac", "spikes", "max_win"))}
    bad += [f"G.14 S-ref: {k}" for k, ok in checks.items() if not ok]
    print(f"G.14 S-ref {pair_key(sref_pair)}: {checks}")
    print("ALL MATCH" if not bad else "MISMATCH\n" + "\n".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
