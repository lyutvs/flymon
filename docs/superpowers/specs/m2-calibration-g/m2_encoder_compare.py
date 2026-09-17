"""Spec appendix G.11 - encoder comparison judged by oracle testability. Calibration: no D.6 (c) verdict.

Four encoders (E0 spec 3.3 receptor-count assignment; E1 same vocabulary, KC-drive assignment; E2 my + opp + move x2;
E3 opp + move x3), two axes (a: same-turn move pairs, b: opponent-type pairs), oracle = freq family over depth
alpha in {0.2, 0.5, 0.8}, punish sweep skipped when the best reward level is below 1.0 (cannot reach testable).
Selection (G.11) uses even turns only; odd turns confirm once.

Run from the repo root:
  PYTHONPATH=docs/superpowers/specs/m2-calibration-g uv run python docs/superpowers/specs/m2-calibration-g/m2_encoder_compare.py --turns even
  ... --turns odd --encoders <winner>
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, itertools, json, math, subprocess, time
from pathlib import Path

import numpy as np

from flymon.battle.pool import POOL
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.presentation import decide
from flymon.brain.stimuli import present
from m2_probe import EXCLUDE, NPZ, READ_MS, SETTLE_MS, assign_channels, build_turns, hp_bin, pool_vocabulary, power_bin

Z = {"A": (21.8293, 18.1032), "P": (40.6433, 24.0891)}
ACT_SEEDS, PROBE_SEEDS = list(range(500, 508)), list(range(600, 608))
STRENGTH, ALPHAS, SKIP_PUNISH_BELOW = 0.35, [0.2, 0.5, 0.8], 1.0
ALL51 = "results/m2/all51_drive.json"
RAW_DIR = Path("results/m2/calibration/encoders")
POWER, HP = ["lt60", "60to89", "ge90"], ["low", "mid", "high"]
PREFERENCE = ["E0", "E1", "E2", "E3"]            # closer to spec 3.3 first (G.11 tie rule)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- encoders -------------------------------------------------------------------------------------
def vocabularies(mon_types, move_types):
    full = ([("my", t, 1) for t in mon_types] + [("opp", t, 1) for t in mon_types] + [("move", t, 1) for t in move_types]
            + [("pow", b, 1) for b in POWER] + [("myhp", b, 1) for b in HP] + [("opphp", b, 1) for b in HP])
    return {"E0": ("count", full), "E1": ("drive", full),
            "E2": ("drive", [("my", t, 1) for t in mon_types] + [("opp", t, 1) for t in mon_types] + [("move", t, 2) for t in move_types]),
            "E3": ("drive", [("opp", t, 1) for t in mon_types] + [("move", t, 3) for t in move_types])}


def band_ranked(need: int) -> list:
    rows = json.load(open(ALL51))["rows"]                       # [receptor, nORN, orn, pn, kc, kc_on, sd]
    cand = sorted(((r[0], r[4]) for r in rows if r[4] > 0),
                  key=lambda rk: (abs(math.log(rk[1]) - math.log(math.sqrt(400 * 3200))), rk[0]))
    if need > len(cand):
        raise ValueError(f"need {need} glomeruli, {len(cand)} have KC drive > 0")
    return [c[0] for c in cand[:need]]


def build_encoder(name, pops, mon_types, move_types):
    """{"group:symbol": [receptor types]}."""
    mode, syms = vocabularies(mon_types, move_types)[name]
    if mode == "count":
        groups = {}
        for g, sym, _k in syms:
            groups.setdefault(g, []).append(sym)
        flat = assign_channels(pops, [(g, v) for g, v in groups.items()])
        return {k: [v] for k, v in flat.items()}
    pool_g = band_ranked(sum(k for *_, k in syms))
    chan, i = {}, 0
    for j in range(max(k for *_, k in syms)):
        for g, sym, k in syms:
            if j < k:
                chan.setdefault(f"{g}:{sym}", []).append(pool_g[i]); i += 1
    return chan


def odour(pops, chan, groups_present, my_types, opp_types, move, bp, my_hp, opp_hp) -> dict:
    keys = []
    if "my" in groups_present: keys += [f"my:{t}" for t in my_types]
    if "opp" in groups_present: keys += [f"opp:{t}" for t in opp_types]
    keys.append(f"move:{move}")
    if "pow" in groups_present: keys.append(f"pow:{power_bin(bp)}")
    if "myhp" in groups_present: keys.append(f"myhp:{hp_bin(my_hp)}")
    if "opphp" in groups_present: keys.append(f"opphp:{hp_bin(opp_hp)}")
    glom = [g for k in keys for g in chan[k]]
    if len(set(glom)) != len(glom):
        raise ValueError(f"glomerulus collision in {keys}")
    inv = np.array([1.0 / len(pops.receptor_types[g]) for g in glom]); inv /= inv.mean()
    return {g: float(s) for g, s in zip(glom, inv)}


def alternate_opponent(turn_index: int, me: str, opp_types, species_types) -> str:
    """G.11: POOL[(5i + 3 + k) mod 16], first k >= 1 with no shared type and not my own species."""
    for k in range(1, len(POOL)):
        cand = POOL[(5 * turn_index + 3 + k) % len(POOL)].species
        if cand != me and not (set(species_types[cand]) & set(opp_types)):
            return cand
    raise ValueError(f"turn {turn_index}: no type-disjoint alternate opponent")


def pairs_for(name, pops, chan, turns, species_types):
    groups_present = {k.split(":")[0] for k in chan}
    out = []
    for t in turns:
        cands = t["candidates"]
        od = lambda c, opp_types: odour(pops, chan, groups_present, t["my_types"], opp_types, c["type"], c["bp"], t["my_hp"], t["opp_hp"])
        for i, j in itertools.combinations(range(len(cands)), 2):
            out.append({"axis": "a", "turn": t["turn"], "x": cands[i]["move"], "y": cands[j]["move"],
                        "odor_x": od(cands[i], t["opp_types"]), "odor_y": od(cands[j], t["opp_types"])})
        alt = alternate_opponent(t["turn"], t["me"], t["opp_types"], species_types)
        for c in cands:
            out.append({"axis": "b", "turn": t["turn"], "x": f"{c['move']} vs {t['opp']}", "y": f"{c['move']} vs {alt}",
                        "odor_x": od(c, t["opp_types"]), "odor_y": od(c, list(species_types[alt]))})
    return out


# ---- oracle job (worker side) --------------------------------------------------------------------------
def dv(c):
    A = np.asarray(c["A"], float); P = np.asarray(c["P"], float)
    V = (A - Z["A"][0]) / Z["A"][1] - (P - Z["P"][0]) / Z["P"][1]
    return V[:, 0] - V[:, 1]


def dprime(x):
    x = np.asarray(x, float); sd = float(x.std(ddof=1)); m = float(x.mean())
    return (0.0 if m == 0 else float(np.copysign(np.inf, m))) if sd == 0 else m / sd


def encoder_oracle_job(eng, pl, pops, comps, ro, odor_x: dict, odor_y: dict) -> dict:
    t = eng.conn.type
    c13, c05 = np.flatnonzero(t == "MBON13"), np.flatnonzero(t == "MBON05")
    idx = np.concatenate([c13, c05]); n13 = len(c13); kc = pops.kc
    try:
        pl.reset_weights(); pl.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(kc)); frac, spikes = [], []
            for s in ACT_SEEDS:
                eng.reset(s); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan()
                present(eng, pops, odor, STRENGTH); eng.run(SETTLE_MS)
                k = eng.run(READ_MS)[kc]
                fired += k > 0; frac.append(float((k > 0).mean())); spikes.append(int(k.sum()))
            return fired / len(ACT_SEEDS), frac, spikes

        def probe():
            A, P = [], []
            for s in PROBE_SEEDS:
                c = decide(eng, pl, pops, [odor_x, odor_y], STRENGTH, s, SETTLE_MS, READ_MS, idx=idx)
                A.append(c[:, :n13].sum(1).tolist()); P.append(c[:, n13:].sum(1).tolist())
            return {"A": A, "P": P}

        fx, frac_x, spk_x = activity(odor_x)
        fy, frac_y, spk_y = activity(odor_y)
        mb = pl.mb_local
        rew = np.isin(pl.post_mb, mb[comps["PAM08"].core]); pun = np.isin(pl.post_mb, mb[comps["PPL105"].core])
        w = eng.csc.w
        pre = probe()
        reward = {}
        for a in ALPHAS:
            wv = pl.w0.copy(); wv[rew] = pl.w0[rew] * (1.0 - a * fx)[pl.pre_kc[rew]]; w[pl.edges] = wv
            R1 = probe(); reward[str(a)] = {"R1": R1, "level": dprime(dv(R1))}
        a_r = max(ALPHAS, key=lambda a: (reward[str(a)]["level"], -a))
        best_r = reward[str(a_r)]["level"]
        punish, a_p, best_p = {}, None, None
        if best_r >= SKIP_PUNISH_BELOW:
            for a in ALPHAS:
                wv = pl.w0.copy()
                wv[rew] = pl.w0[rew] * (1.0 - a_r * fx)[pl.pre_kc[rew]]
                wv[pun] = pl.w0[pun] * (1.0 - a * fx)[pl.pre_kc[pun]]
                w[pl.edges] = wv
                R2 = probe(); punish[str(a)] = {"R2": R2, "drop": dprime(dv(R2) - dv(reward[str(a_r)]["R1"]))}
            a_p = min(ALPHAS, key=lambda a: (punish[str(a)]["drop"], a)); best_p = punish[str(a_p)]["drop"]
        w[pl.edges] = pl.w0
        return {"pre": pre, "reward": reward, "punish": punish, "alpha_reward": a_r, "alpha_punish": a_p,
                "best_reward_level": best_r, "best_punish_drop": best_p,
                "testable": bool(best_r >= 2 and best_p is not None and best_p <= -2),
                "d_pre": dprime(dv(pre)),
                "kc": {"frac_x": frac_x, "frac_y": frac_y, "spikes_x": spk_x, "spikes_y": spk_y,
                       "jaccard": float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))}}
    finally:
        pl.reset_weights(); pl.set_enabled(True)


# ---- driver -----------------------------------------------------------------------------------------------
def score_encoder(rows):
    rate = {ax: (sum(r["testable"] for r in rows if r["axis"] == ax) / max(sum(r["axis"] == ax for r in rows), 1)) for ax in "ab"}
    frac = np.array([f for r in rows for f in r["kc"]["frac_x"] + r["kc"]["frac_y"]])
    return {"rate_a": rate["a"], "rate_b": rate["b"], "score": min(rate.values()),
            "n_a": sum(r["axis"] == "a" for r in rows), "n_b": sum(r["axis"] == "b" for r in rows),
            "testable_a": sum(r["testable"] for r in rows if r["axis"] == "a"),
            "testable_b": sum(r["testable"] for r in rows if r["axis"] == "b"),
            "kc_below_3pct": float((frac < 0.03).mean()), "kc_in_5_9pct": float(((frac >= 0.05) & (frac <= 0.09)).mean()),
            "eligible": bool((frac < 0.03).mean() <= 0.05)}


def select(scores):
    """G.11 rules 1-4 on even-turn scores."""
    elig = {e: s for e, s in scores.items() if s["eligible"]}
    if not elig:
        return {"winner": None, "why": "no encoder passes the sparsity eligibility"}
    ranked = sorted(elig, key=lambda e: -elig[e]["score"])
    top = elig[ranked[0]]["score"]
    if top < 0.5:
        return {"winner": None, "why": f"best score {top:.3f} < 0.5 - stop for a user decision"}
    near = [e for e in ranked if top - elig[e]["score"] <= 0.05]
    return {"winner": min(near, key=PREFERENCE.index), "top_score": top, "within_0.05": near}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", choices=["even", "odd"], required=True)
    ap.add_argument("--encoders", nargs="+", default=PREFERENCE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N pairs per encoder")
    a = ap.parse_args()
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    turns = [t for t in build_turns(species_types, move_info, 16) if (t["turn"] % 2 == 0) == (a.turns == "even")]
    plan = {}
    for e in a.encoders:
        chan = build_encoder(e, pops, mon_types, move_types)
        ps = pairs_for(e, pops, chan, turns, species_types)
        plan[e] = (chan, ps[: a.limit] if a.limit else ps)
        print(f"{e}: {len(chan)} channels, {sum(len(v) for v in chan.values())} glomeruli, "
              f"{sum(p['axis'] == 'a' for p in plan[e][1])} (a) + {sum(p['axis'] == 'b' for p in plan[e][1])} (b) pairs")
    del conn
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{a.turns}{'_smoke' if a.limit else ''}"
    code_sha = sha256(__file__)
    scores, todo = {}, []
    for e, (chan, ps) in plan.items():                   # resume: reuse an encoder only if the same code wrote all its rows
        raw = RAW_DIR / f"{e}_{tag}.json"
        prev = json.loads(raw.read_text()) if raw.exists() else None
        if prev and prev.get("code_sha256") == code_sha and len(prev.get("rows", [])) == len(ps):
            scores[e] = score_encoder(prev["rows"])
            print(f"{e} resumed from {raw} (code sha matches, {len(ps)} rows)", flush=True)
        else:
            todo.append(e)

    def report(e, secs):
        s = scores[e]
        print(f"{e} done {secs:.0f}s  (a) {s['testable_a']}/{s['n_a']}  (b) {s['testable_b']}/{s['n_b']}  "
              f"score {s['score']:.3f}  KC<3% {s['kc_below_3pct']:.3f}  KC 5-9% {s['kc_in_5_9pct']:.3f}  eligible {s['eligible']}", flush=True)

    if todo:
        with FlyPool(NPZ, Params(), [{} for _ in range(a.workers)], workers=a.workers, timeout_s=7200.0) as pool:
            for e in todo:
                chan, ps = plan[e]
                t0 = time.perf_counter()
                res = pool.run_jobs(encoder_oracle_job, [dict(odor_x=p["odor_x"], odor_y=p["odor_y"]) for p in ps])
                rows = [{k: p[k] for k in ("axis", "turn", "x", "y")} | r for p, r in zip(ps, res)]
                tmp = RAW_DIR / f"{e}_{tag}.json.tmp"
                tmp.write_text(json.dumps({"encoder": e, "code_sha256": code_sha, "channels": chan, "rows": rows}, indent=1))
                tmp.rename(RAW_DIR / f"{e}_{tag}.json")          # atomic: a crash mid-write never leaves a partial file
                scores[e] = score_encoder(rows)
                report(e, time.perf_counter() - t0)
    decision = select(scores) if (a.turns == "even" and not a.limit) else {"note": "selection runs on the full even-turn set only"}
    print("\nselection:", json.dumps(decision))
    if not a.limit:
        sha = sha256
        summ = Path("results/summary/m2_encoders.json")
        prev = json.loads(summ.read_text()) if summ.exists() else {}
        prev[a.turns] = {"written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                         "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                         "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
                         "code_sha256": code_sha, "raw": {e: sha(RAW_DIR / f"{e}_{tag}.json") for e in plan},
                         "scores": scores, "decision": decision}
        summ.write_text(json.dumps(prev, indent=1) + "\n")
        print(f"wrote {summ}")


if __name__ == "__main__":
    main()
