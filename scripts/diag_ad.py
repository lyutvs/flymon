#!/usr/bin/env python3
"""Spec AD.11 (record-only, "기록 전용 추정"): why naive FLY-L's right-direction share is 0.196 (FLY-OS 0.45-0.50) and why
FLY-L's share plateaus near 0.40 by 40 battles and ends at 0.362 at 120. Re-reads AD's records (situation records,
turn logs, the end-of-learning checkpoint) and, offline, re-evaluates EXISTING weights (w0 and the 120-battle FLY-L
weights; variants of them) on the confirmation set with AD's own evaluation path (flymon/ac/situations.evaluate, the
same decision seeds as the run's records). No battle, no learning, no verdict, no AD / AC rule change.

    uv run python scripts/diag_ad.py offline --variant map0                      # naive odour map + edge map (~4 min)
    uv run python scripts/diag_ad.py offline --variant amp1 --flies 0            # reproduction check (~1 min)
    uv run python scripts/diag_ad.py offline --variant amp2 [--flies 0-11]       # resumable per fly (~14 min / 12 flies)
    uv run python scripts/diag_ad.py offline --variant amp3.4 | only_PAM08 | only_PPL105
    uv run python scripts/diag_ad.py analyse                                     # -> results/summary/ad_diag.json

Offline records go to results/m4/ad/diag/ (git-excluded raw); a variant is resumable per fly (a fly already recorded
for that variant is skipped). analyse writes whatever offline variants are complete and lists the missing ones."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OUT = "results/summary/ad_diag.json"
ROOT = Path("results/m4/ad")
DIAG = ROOT / "diag"
INPUTS = "results/summary/ad_inputs.json"
NPZ = "data/malecns.npz"
CKPT = ROOT / "checkpoints" / "learn" / "state_002880.npz"
POINTS = (0, 20, 40, 60, 80, 100, 120)
FLY_VARIANTS = ("amp1", "amp2", "amp3.4", "only_PAM08", "only_PPL105")
ARMS = ("FLYL", "FLYOS")
KIND = "기록 전용 추정"


def log(msg: str) -> None:
    print(msg, flush=True)


def jl_read(path) -> list:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def jl_append(path, rec) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_flies(s: str) -> list:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ================================================================ offline re-evaluation (needs the brain)
def brain():
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.ad.spec import SPEC
    from flymon.agent.encode_grid import GridEncoder
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h4_jobs import type_cells
    model = json.loads(Path("results/summary/ac_model_manifest.json").read_text())
    if model.get("status") != "FROZEN":
        raise SystemExit("refusing: the model manifest is not FROZEN")
    cfg = load_lv_config(float(model["recovery_per_pulse"]))
    cb, _ = lv_codebook()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    grid = GridEncoder(pops, cb, SPEC.dual_rule)
    cells = type_cells(conn, [cfg.readout["A"], cfg.readout["P"]])
    return SPEC, model, cfg, pops, grid, cells


def variant_weights(name: str, w, w0, masks, floor: float):
    from flymon.ad import diag
    if name.startswith("amp"):
        return diag.amplify(w, w0, float(name[3:]), floor)
    if name.startswith("only_"):
        return diag.keep_only(w, w0, masks[name[5:]])
    raise ValueError(name)


def offline(a) -> int:
    from flymon.ac import situations
    from flymon.agent.encode_grid import reachable
    from flymon.agent.policy import derive_seed, values
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.lv_pool import LeverFlyPool
    from flymon.ad import diag
    spec, model, cfg, pops, grid, cells = brain()
    pairs = json.loads(Path(INPUTS).read_text())["confirm"]["pairs"]
    lay = spec.layout()
    a_idx, p_idx = cells[cfg.readout["A"]], cells[cfg.readout["P"]]
    DIAG.mkdir(parents=True, exist_ok=True)
    n_pool = 1 if a.variant == "map0" else 12
    with LeverFlyPool(NPZ, cfg.params, [FlySpec() for _ in range(n_pool)], edit=spec.lever_edit, p_type=spec.p_type,
                      workers=a.workers, timeout_s=7200) as pool:
        if pool.lever_sha() != model["lever_sha"]:
            raise SystemExit("refusing: the workers' L_V sha differs from the model manifest's")
        w0 = pool.w0[None]
        emap_p = DIAG / "edge_map.npz"
        if not emap_p.exists():
            em = pool.run_jobs(diag.edge_map_job, [dict(types=[cfg.reward_type, cfg.punish_type])])[0]
            np.savez_compressed(emap_p, pre_kc=em["pre_kc"], tgt=em["tgt"], w0=w0, a_cells=a_idx, p_cells=p_idx,
                                **{f"mask_{t}": m for t, m in em["masks"].items()})
            log(f"wrote {emap_p}")
        em = np.load(emap_p)
        masks = {t: em[f"mask_{t}"] for t in (cfg.reward_type, cfg.punish_type)}
        if a.variant == "map0":
            out = DIAG / "map0.jsonl"
            if jl_read(out):
                log(f"{out} exists: skipped")
                return 0
            mtypes = sorted({v[0] for v in grid.move_info.values()})
            items = reachable(grid.species_types, mtypes)
            idx = np.concatenate([a_idx, p_idx, np.asarray(pops.kc)])
            na, npp = len(a_idx), len(p_idx)
            reqs = [(0, [diag_odour(grid, m, o)], derive_seed(spec.tie_seed, "ad11-map", m, "/".join(o)))
                    for m, o in items]
            log(f"map0: {len(reqs)} odours at w0")
            counts = pool.decide_batch(reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
            for (m, o), cnt in zip(items, counts):
                cnt = np.asarray(cnt)[0]
                av, pv = int(cnt[:na].sum()), int(cnt[na:na + npp].sum())
                jl_append(out, dict(move_type=m, opp_types=list(o), a=av, p=pv,
                                    v=float(values([av], [pv], cfg.z)[0]),
                                    kc=[int(k) for k in np.flatnonzero(cnt[na + npp:] > 0)]))
            log(f"wrote {out}")
            return 0
        if a.variant not in FLY_VARIANTS:
            raise SystemExit(f"unknown variant {a.variant!r}")
        out = DIAG / "offline.jsonl"
        done = {(r["variant"], r["pool_index"]) for r in jl_read(out)}
        flies = [i for i in parse_flies(a.flies) if (a.variant, i) not in done]
        if not flies:
            log(f"{a.variant}: every requested fly already recorded")
            return 0
        z = np.load(CKPT)
        for i in flies:
            if lay[i][0] != "FLYL":
                raise SystemExit(f"pool index {i} is not FLY-L")
            pool.w[i] = variant_weights(a.variant, z[f"w{i}"], w0, masks, spec.floor_ratio)
        ods = situations.pair_odours(grid, pairs)

        def one(i):
            rec = situations.evaluate(pool, i, 120, pairs, ods, cfg, a_idx, p_idx, pops.kc, spec.tie_seed,
                                      seed_fly=lay[i][2], seed_key="sit")
            rec.update(variant=a.variant, pool_index=i, fly=lay[i][2], arm="FLYL", label=spec_label())
            return rec
        log(f"{a.variant}: flies {flies}")
        with ThreadPoolExecutor(max_workers=len(flies)) as ex:
            for rec in ex.map(one, flies):
                jl_append(out, rec)
                log(f"  {a.variant} pool {rec['pool_index']} (fly {rec['fly']}): rate {rec['rate']:.3f}")
    return 0


def diag_odour(grid, move_type, opp_types):
    from flymon.agent.encode_grid import odour
    return odour(grid.rc, grid.cb, move_type, tuple(opp_types), grid.rule)


def spec_label() -> str:
    from flymon.ad.spec import LABEL
    return LABEL


# ================================================================ analysis (records + offline records)
def load_sits():
    recs = jl_read(ROOT / "logs" / "situations_init.jsonl") + jl_read(ROOT / "logs" / "situations.jsonl")
    out = defaultdict(dict)
    for r in recs:
        if r.get("set") != "confirm" or r.get("frozen") is not True:
            continue
        key = (r["arm"], r["mode"], int(r["point"]))
        if r["fly"] in out[key]:
            raise SystemExit(f"duplicate record {key} fly {r['fly']}")
        out[key][r["fly"]] = r
    return out


def share_of(recs: dict, pairs, fn=None) -> dict:
    from flymon.ad import diag
    per = {g: float(np.mean(diag.right_flags(fn(r) if fn else r, pairs))) for g, r in sorted(recs.items())}
    v = list(per.values())
    return dict(share=float(np.mean(v)) if v else None, sd_flies=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                n=len(v))


def flag_units(recs: dict, pairs, fn=None) -> dict:
    from flymon.ad import diag
    return {g: diag.right_flags(fn(r) if fn else r, pairs) for g, r in sorted(recs.items())}


def q1(S, pairs, z, mi, st, mapr) -> dict:
    from flymon.ac import diag as acd
    from flymon.ac.confirm import type_mult
    from flymon.ad import diag
    L0, OS0, LP0 = S[("FLYL", "own", 0)], S[("FLYOS", "own", 0)], S[("FLYL", "os_paired", 0)]
    out = dict(right_share_at_0={"FLYL (FLY odours)": share_of(L0, pairs), "FLYOS (OS odours)": share_of(OS0, pairs),
                                 "FLYL os_paired (w0, OS odours)": share_of(LP0, pairs)})
    # class bias of the naive readout: centred V, z_A and -z_P by class
    comp = {}
    for kind in ("zv", "a_only", "p_only"):
        comp[kind] = diag.class_means([diag.readout_variant(r, z, kind) for r in L0.values()], pairs)
    n_cls = defaultdict(int)
    for s in next(iter(L0.values()))["situations"]:
        for c in range(len(s["v"])):
            n_cls[diag.class_of(pairs[s["pair"]], s["side"], c)] += 1
    out["naive_class_bias"] = dict(centred_v=comp["zv"], centred_z_A=comp["a_only"], centred_minus_z_P=comp["p_only"],
                                   n_candidate_situations_per_fly=dict(n_cls),
                                   note="w0 is shared by every FLY-L fly: the effective n is the 20 pairs' "
                                        "situations, not flies x situations")
    bias = comp["zv"]
    pred = diag.class_model_contrast(pairs, bias)
    obs_flag = np.mean(np.array(list(flag_units(L0, pairs).values()), float), axis=0)
    mean_d = np.mean(np.array([acd.best_contrast(r, pairs) for r in L0.values()]), axis=0)
    rows = []
    for i, p in enumerate(pairs):
        b1, b2 = p["cands"].index(p["best1"]), p["cands"].index(p["best2"])
        rows.append(dict(pair=i, me=p["me"], o1=p["o1"], o2=p["o2"], k=len(p["cands"]),
                         best1=p["best1"], best2=p["best2"],
                         cls_best1_side0=diag.class_of(p, 0, b1), cls_best2_side0=diag.class_of(p, 0, b2),
                         cls_best1_side1=diag.class_of(p, 1, b1), cls_best2_side1=diag.class_of(p, 1, b2),
                         flyl0_right=float(obs_flag[i]), flyl0_mean_d=float(mean_d[i]),
                         class_model_d=float(pred[i]),
                         flyos0_right=float(np.mean([diag.right_flags(r, pairs)[i] for r in OS0.values()]))))
    agree = [int((r["class_model_d"] > 0) == (r["flyl0_right"] >= 0.5)) for r in rows if r["class_model_d"] != 0]
    out["pairs_at_0"] = rows
    out["class_model"] = dict(
        predicted_right_share=float(np.mean([d > 0 for d in pred])), predicted_wrong_share=float(np.mean([d < 0 for d in pred])),
        predicted_zero_share=float(np.mean([d == 0 for d in pred])), sign_agreement_with_observed=diag_mean(agree),
        n_pairs_with_nonzero_prediction=len(agree),
        note="V replaced by the point-0 class mean of centred V (an additive class-only readout); d's sign is the "
             "predicted direction. Agreement = the predicted sign equals the pair's majority flag over the 12 flies")
    # by effectiveness class of the two best moves; by opponent type set (side-level correctness)
    pc = defaultdict(list)
    for r in L0.values():
        for k, v in diag.per_class_contrast(r, pairs).items():
            pc[k] += v
    out["by_best_classes"] = {k: dict(n_fly_pairs=len(v), n_pairs=len(v) // max(len(L0), 1), right=diag_mean(v))
                              for k, v in sorted(pc.items())}
    side = defaultdict(lambda: defaultdict(list))
    for arm, recs in (("FLYL", L0), ("FLYOS", OS0)):
        for r in recs.values():
            for s in r["situations"]:
                p = pairs[s["pair"]]
                ot = "/".join(p["o1_types"] if s["side"] == 0 else p["o2_types"])
                side[ot][arm].append(int(s["correct"]))
    out["by_opponent_types_side_correct"] = {ot: {arm: dict(n=len(v), correct=diag_mean(v)) for arm, v in d.items()}
                                             for ot, d in sorted(side.items())}
    # why FLY-OS sits near 0.5: symmetric contrasts
    sym = {}
    for name, recs in (("FLYOS@0", OS0), ("FLYL_os_paired@0", LP0), ("FLYL@0", L0)):
        sym[name] = diag.contrast_signs(d for r in recs.values() for d in acd.best_contrast(r, pairs))
    sym["FLYOS@all_points"] = diag.contrast_signs(d for p in POINTS for r in S[("FLYOS", "own", p)].values()
                                                  for d in acd.best_contrast(r, pairs))
    out["contrast_symmetry"] = dict(
        cells=sym,
        argument="FLY-OS draws the opponent species for each side independently and uniformly from the same 16 "
                 "(seed keys differ only by side), so the two sides' odours are exchangeable and d = X(side 0) - "
                 "X(side 1) is symmetric about 0 for any fixed weights: P(d > 0) = (1 - P(d = 0)) / 2. FLY-L's d is "
                 "fixed by the pair's two true odours and w0")
    # seed noise at point 0 (same w0 and odours, decision seeds differ)
    out["seed_noise_at_0"] = dict(flag_majority_disagreement=diag.majority_disagreement(list(flag_units(L0, pairs).values())),
                                  **acd.noise_spread(list(L0.values()), len(pairs)))
    # the naive odour map (offline, w0, every reachable (move type, opponent type set) odour)
    if mapr:
        out["naive_map"] = naive_map(mapr, pairs, mi, st, type_mult)
    else:
        out["naive_map"] = dict(missing="run: scripts/diag_ad.py offline --variant map0")
    return out


def diag_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def naive_map(mapr, pairs, mi, st, type_mult) -> dict:
    from flymon.ac import diag as acd
    from flymon.ad import diag
    rows = []
    for r in mapr:
        m = type_mult(r["move_type"], r["opp_types"])
        rows.append(dict(r, mult=m, cls=acd.mult_class(m), dual=len(r["opp_types"]) == 2, n_kc=len(r["kc"])))
    v = np.array([x["v"] for x in rows])
    # centre within move type (a move's V is compared with other moves', and a move type's base level is shared)
    by_mt = defaultdict(list)
    for x in rows:
        by_mt[x["move_type"]].append(x["v"])
    mt_mean = {k: float(np.mean(vv)) for k, vv in by_mt.items()}
    by_ot = defaultdict(list)
    for x in rows:
        by_ot[tuple(x["opp_types"])].append(x["v"])
    ot_mean = {k: float(np.mean(vv)) for k, vv in by_ot.items()}
    za, zp = defaultdict(list), defaultdict(list)
    out_cls = {}
    for c in diag.CLASSES:
        xs = [x for x in rows if x["cls"] == c]
        if not xs:
            continue
        out_cls[c] = dict(n=len(xs), v=diag_mean(x["v"] for x in xs),
                          v_minus_opp_mean=diag_mean(x["v"] - ot_mean[tuple(x["opp_types"])] for x in xs),
                          v_minus_movetype_mean=diag_mean(x["v"] - mt_mean[x["move_type"]] for x in xs),
                          a=diag_mean(x["a"] for x in xs), p=diag_mean(x["p"] for x in xs),
                          n_kc=diag_mean(x["n_kc"] for x in xs), dual_share=diag_mean(int(x["dual"]) for x in xs))
    by_arity = {}
    for dual in (False, True):
        for c in diag.CLASSES:
            xs = [x for x in rows if x["cls"] == c and x["dual"] == dual]
            if xs:
                by_arity[f"{'dual' if dual else 'single'}|{c}"] = dict(
                    n=len(xs), v=diag_mean(x["v"] for x in xs), n_kc=diag_mean(x["n_kc"] for x in xs),
                    a=diag_mean(x["a"] for x in xs), p=diag_mean(x["p"] for x in xs))
    lm = [acd.log2_mult(x["mult"]) for x in rows]
    keyed = {(x["move_type"], tuple(x["opp_types"])): x for x in rows}
    conf = set()
    for p in pairs:
        for c in p["cands"]:
            for ot in (p["o1_types"], p["o2_types"]):
                conf.add((mi[c][0], tuple(sorted(ot))))
    cx = [keyed[k] for k in conf if k in keyed]
    # KC pattern: across the two sides (same move, two opponents) vs within a side (two moves, same opponent)
    jac_side, jac_cand = [], []
    for p in pairs:
        for c in p["cands"]:
            k0, k1 = (mi[c][0], tuple(sorted(p["o1_types"]))), (mi[c][0], tuple(sorted(p["o2_types"])))
            jac_side.append(diag.jaccard(keyed[k0]["kc"], keyed[k1]["kc"]))
        for ot in (p["o1_types"], p["o2_types"]):
            ks = [keyed[(mi[c][0], tuple(sorted(ot)))]["kc"] for c in p["cands"]]
            jac_cand += [diag.jaccard(ks[i], ks[j]) for i in range(len(ks)) for j in range(i + 1, len(ks))]
    # KC overlap between a true-opponent odour and the 16-species draws (FLY-OS): mean over the draw distribution
    from flymon.ad.os_encoder import SPECIES
    jac_os = []
    for p in pairs:
        for c in p["cands"]:
            for ot in (p["o1_types"], p["o2_types"]):
                true = keyed[(mi[c][0], tuple(sorted(ot)))]["kc"]
                jac_os.append(float(np.mean([diag.jaccard(true, keyed[(mi[c][0], tuple(sorted(st[s])))]["kc"])
                                             for s in SPECIES])))
    return dict(
        n_odours=len(rows), by_class=out_cls, by_arity_and_class=by_arity,
        mean_n_kc_single_vs_dual=dict(single=diag_mean(x["n_kc"] for x in rows if not x["dual"]),
                                      dual=diag_mean(x["n_kc"] for x in rows if x["dual"])),
        spearman_v_vs_log2mult=diag.spearman_safe(lm, v),
        spearman_v_vs_log2mult_centred_in_opp=diag.spearman_safe(lm, [x["v"] - ot_mean[tuple(x["opp_types"])] for x in rows]),
        spearman_v_vs_log2mult_centred_in_movetype=diag.spearman_safe(lm, [x["v"] - mt_mean[x["move_type"]] for x in rows]),
        spearman_v_vs_n_kc=diag.spearman_safe([x["n_kc"] for x in rows], v),
        spearman_a_vs_n_kc=diag.spearman_safe([x["n_kc"] for x in rows], [x["a"] for x in rows]),
        spearman_p_vs_n_kc=diag.spearman_safe([x["n_kc"] for x in rows], [x["p"] for x in rows]),
        mean_v_single_vs_dual=dict(single=diag_mean(x["v"] for x in rows if not x["dual"]),
                                   dual=diag_mean(x["v"] for x in rows if x["dual"])),
        confirm_set_odours=dict(n=len(cx), spearman_v_vs_log2mult=diag.spearman_safe(
            [acd.log2_mult(x["mult"]) for x in cx], [x["v"] for x in cx]),
            by_class={c: dict(n=sum(x["cls"] == c for x in cx), v=diag_mean(x["v"] for x in cx if x["cls"] == c))
                      for c in diag.CLASSES}),
        kc_jaccard=dict(same_move_across_sides=diag_mean(jac_side), across_candidates_within_side=diag_mean(jac_cand),
                        true_vs_os_draw=diag_mean(jac_os)),
        method="one decision seed per odour (derive_seed(305, 'ad11-map', move type, opp types)), w0, the run's "
               "settle / read / strength; KC = cells with >= 1 spike in the read window. Seed noise: q1 seed_noise_at_0")


def q2(S, pairs, z, logs, lay, offl, emap, w120, w0, mapr, r_rec) -> dict:
    from flymon.ac import diag as acd
    from flymon.ad import diag
    out = {}
    out["available_weights"] = dict(points=[0, 120], note="checkpoints/learn keeps only the last two generations "
                                    "(state_002879 / state_002880): weights exist at w0 and after battle 120 only; "
                                    "points 20-100 are in the situation records (V, MBON counts, KC counts, the weight "
                                    "summary) but not as weights")
    # right-share curves and readout variants per point (records)
    curves = {}
    for arm in ARMS:
        c = {}
        for p in POINTS:
            recs = S[(arm, "own", p)]
            c[p] = dict(own=share_of(recs, pairs),
                        delta_from_0=share_of(recs, pairs, lambda r, arm=arm: diag.delta_record(
                            r, S[(arm, "own", 0)][r["fly"]])) if p else None,
                        a_only=share_of(recs, pairs, lambda r: diag.readout_variant(r, z, "a_only")),
                        p_only=share_of(recs, pairs, lambda r: diag.readout_variant(r, z, "p_only")),
                        raw_counts=share_of(recs, pairs, lambda r: diag.readout_variant(r, z, "raw")))
        curves[arm] = c
    curves["FLYL_os_paired"] = {p: share_of(S[("FLYL", "os_paired", p)], pairs) for p in POINTS}
    out["right_share_by_readout"] = curves
    # per-pair trajectories and transitions
    traj, trans = {}, {}
    for arm in ARMS:
        units = {p: flag_units(S[(arm, "own", p)], pairs) for p in POINTS}
        flies = sorted(units[0])
        traj[arm] = [dict(pair=i, **{str(p): float(np.mean([units[p][g][i] for g in flies])) for p in POINTS})
                     for i in range(len(pairs))]
        flat = {p: [units[p][g][i] for g in flies for i in range(len(pairs))] for p in POINTS}
        t = diag.transitions(flat)
        t["n_units"] = len(flies) * len(pairs)
        trans[arm] = t
    # per pair (FLY-L): the naive contrast d(0), the learned contrast change (best_contrast of V(t) - V(0)) at each
    # point, flies' mean: a pair turns right when the learned change exceeds the naive deficit
    L0 = S[("FLYL", "own", 0)]
    d0 = np.mean([acd.best_contrast(r, pairs) for r in L0.values()], axis=0)
    dd = {p: np.mean([acd.best_contrast(diag.delta_record(r, L0[g]), pairs) for g, r in S[("FLYL", "own", p)].items()],
                     axis=0) for p in POINTS if p}
    for row in traj["FLYL"]:
        i = row["pair"]
        row["naive_d"] = float(d0[i])
        row["learned_dd"] = {str(p): float(dd[p][i]) for p in dd}
    r120 = [row["120"] for row in traj["FLYL"]]
    out["naive_deficit_vs_learned_change"] = dict(
        spearman_naive_d_vs_right_120=diag.spearman_safe(d0, r120),
        learned_dd_mean_over_pairs={str(p): float(np.mean(dd[p])) for p in dd},
        learned_dd_median_over_pairs={str(p): float(np.median(dd[p])) for p in dd},
        share_pairs_learned_dd_positive={str(p): float(np.mean(dd[p] > 0)) for p in dd},
        pairs_right_at_120_ge_half=int(np.sum(np.array(r120) >= 0.5)),
        pairs_never_right=int(sum(all(row[str(p)] < 0.1 for p in POINTS) for row in traj["FLYL"])),
        note="d = AC.11 best_contrast; learned_dd = best_contrast of the delta record (V(t) - V(0)); with fixed seeds "
             "d(t) = d(0) + dd(t), so a pair is right at t when dd(t) > -d(0)")
    out["pair_trajectories"] = traj
    out["transitions"] = trans
    out["transitions_note"] = ("units = fly x pair (12 x 20 = 240 per arm); decision seeds carry no point, so a flag "
                               "change between points comes from the weights. Reference for a change of the same "
                               "size from spike noise alone: q1 seed_noise_at_0 flag_majority_disagreement")
    # right / wrong at 120 conditioned on the point-0 flag (FLY-L)
    u0, u120 = flag_units(S[("FLYL", "own", 0)], pairs), flag_units(S[("FLYL", "own", 120)], pairs)
    cond = defaultdict(list)
    for g in u0:
        for a_, b_ in zip(u0[g], u120[g]):
            cond[f"naive_{'right' if a_ else 'wrong'}"].append(b_)
    out["flyl_120_given_naive"] = {k: dict(n=len(v), right_at_120=diag_mean(v)) for k, v in cond.items()}
    # pulses by effectiveness class per interval
    out["pulses_by_class"] = {}
    for arm in ARMS:
        rows = []
        for i, recs in logs.items():
            if lay[i][0] == arm:
                rows += diag.pulse_rows(acd.join_turns(recs))
        out["pulses_by_class"][arm] = diag.pulse_class_table(rows, POINTS)
    # compartment means per 10 battles (battle records: compartment_fracs at each battle's end)
    comp = {}
    for arm in ARMS:
        acc = defaultdict(lambda: defaultdict(list))
        for i, recs in logs.items():
            if lay[i][0] != arm:
                continue
            for r in recs:
                if r.get("kind") == "battle" and r.get("compartment_fracs"):
                    from flymon.ad.inputs import battle_index
                    b = battle_index(r["battle_id"])
                    for t, v in r["compartment_fracs"].items():
                        acc[t][b // 10 * 10].append(float(v))
        comp[arm] = {t: {str(b): float(np.mean(v)) for b, v in sorted(d.items())} for t, d in acc.items()}
    out["compartment_mean_w_over_w0_by_battle_bin"] = comp
    # weight summary recorded at the situation evaluations (every point)
    ws = {}
    for arm in ARMS:
        ws[arm] = {p: {k: diag_mean((r.get("weights") or {}).get(k) for r in S[(arm, "own", p)].values())
                       for k in ("frac_changed", "median_ratio_changed", "frac_floor_of_changed", "frac_up_of_changed")}
                   for p in POINTS if p}
    out["weight_summary_by_point"] = ws
    # the weights at 120 vs w0: by compartment x readout cell, and the confirm-set-active KCs
    if emap is not None and w120 is not None:
        out["weights_at_120"] = weights_section(emap, w120, w0, lay, mapr, pairs)
    # recovery
    out["recovery"] = recovery_section(logs, lay, r_rec)
    # offline variants at 120
    out["offline_120"] = offline_section(offl, S, pairs)
    return out


def weights_section(emap, w120, w0, lay, mapr, pairs) -> dict:
    from flymon.ad import diag
    tgt, pre = emap["tgt"], emap["pre_kc"]
    a_c, p_c = set(emap["a_cells"].tolist()), set(emap["p_cells"].tolist())
    is_a, is_p = np.isin(tgt, list(a_c)), np.isin(tgt, list(p_c))
    active = None
    if mapr:
        kcs = set()
        from flymon.brain.h4_pairs import pool_vocabulary
        _, mi, _, _ = pool_vocabulary()
        keyed = {(r["move_type"], tuple(r["opp_types"])): r["kc"] for r in mapr}
        for p in pairs:
            for c in p["cands"]:
                for ot in (p["o1_types"], p["o2_types"]):
                    kcs |= set(keyed[(mi[c][0], tuple(sorted(ot)))])
        active = np.isin(pre, sorted(kcs))
    out = {"n_edges": int(tgt.size), "n_confirm_active_kc": None if active is None else int(len(set(pre[active])))}
    for t in ("PAM08", "PPL105"):
        m = emap[f"mask_{t}"]
        out[f"{t}_core_has_A_cells"] = bool(np.any(m & is_a))
        out[f"{t}_core_has_P_cells"] = bool(np.any(m & is_p))
    sel = {"A_cells": is_a, "P_cells": is_p}
    for t in ("PAM08", "PPL105"):
        sel[f"core_{t}"] = emap[f"mask_{t}"]
    per_arm = {}
    for arm in ARMS:
        cells = {}
        for name, m in sel.items():
            for act_name, am in (("all", None), ("confirm_active_kc", active)):
                if act_name != "all" and am is None:
                    continue
                mm = m if am is None else (m & am)
                rs = [diag.ratio_summary(w120[i], w0, mm) for i in range(len(lay)) if lay[i][0] == arm]
                cells[f"{name}/{act_name}"] = {k: diag_mean(r.get(k) for r in rs) for k in rs[0]}
        per_arm[arm] = cells
    out["per_arm_mean_over_flies"] = per_arm
    out["note"] = ("ratio w/w0; floor = share <= 0.2 (the plasticity floor min_weight_frac); at_w0 = unchanged (the "
                   "rule depresses only and recovers toward w0, so w0 is the ceiling); confirm_active_kc = edges whose "
                   "KC spiked for any confirmation-set FLY odour in the naive map")
    return out


def recovery_section(logs, lay, r) -> dict:
    from flymon.ad import diag
    from flymon.ad.inputs import battle_index
    from flymon.brain.h4_pairs import pool_vocabulary
    st, _, _, _ = pool_vocabulary()
    out = {}
    for arm in ARMS:
        gaps_odour, gaps_mt, per_battle = [], [], []
        for i, recs in logs.items():
            if lay[i][0] != arm:
                continue
            dec = {}
            for x in recs:
                if x.get("kind") == "decision" and x.get("decider") == "fly" and x.get("candidates"):
                    dec[(x["battle_id"], x["turn"])] = x
            k_od, k_mt, nb = [], [], defaultdict(int)
            for x in recs:
                if x.get("kind") != "reinforce" or not x.get("pulses"):
                    continue
                d = dec.get((x["battle_id"], x["turn"]))
                if d is None or x["odour_move"] not in d["candidates"]:
                    continue
                mt, ot = d["egrid"][d["candidates"].index(x["odour_move"])]
                if arm == "FLYOS" and d.get("os"):
                    ot = sorted(st[d["os"]["drawn"]])
                key = (mt, tuple(ot))
                for _ in x["pulses"]:
                    k_od.append(key)
                    k_mt.append(mt)
                nb[battle_index(x["battle_id"])] += len(x["pulses"])
            gaps_odour += diag.use_gaps(k_od)
            gaps_mt += diag.use_gaps(k_mt)
            per_battle.append(np.mean(list(nb.values())))
        ppb = float(np.mean(per_battle))
        med_o, med_m = float(np.median(gaps_odour)), float(np.median(gaps_mt))
        out[arm] = dict(pulses_per_battle=ppb, recovery_time_constant_battles=1.0 / (r * ppb),
                        retention_after_20_battles=diag.retention(r, 20 * ppb),
                        retention_after_40_battles=diag.retention(r, 40 * ppb),
                        retention_after_120_battles=diag.retention(r, 120 * ppb),
                        same_odour_gap_pulses=dict(median=med_o, q90=float(np.percentile(gaps_odour, 90)),
                                                   n=len(gaps_odour)),
                        retention_between_same_odour_uses_median=diag.retention(r, med_o),
                        same_move_type_gap_pulses=dict(median=med_m, n=len(gaps_mt)),
                        no_recovery_deficit_factor_at_40=diag.no_recovery_factor(r, 40 * ppb),
                        no_recovery_deficit_factor_at_120=diag.no_recovery_factor(r, 120 * ppb))
    out["r"] = r
    out["note"] = ("recover_pulse: after each DAN pulse every plastic edge moves r of the way back to w0 "
                   "(plasticity.py). Odour key = (move type, opponent type set the brain saw: FLY-L the true one, "
                   "FLY-OS the drawn one). no_recovery factor = r n / (1 - exp(-r n)), the linear-regime ratio of the "
                   "deficit without recovery to the deficit with it for a constant depression per pulse")
    return out


def offline_section(offl, S, pairs) -> dict:
    from flymon.ac import diag as acd
    from flymon.ad import diag
    from flymon.ad.spec import SPEC
    by = defaultdict(dict)
    for r in offl:
        by[r["variant"]][r["fly"]] = r
    actual = S[("FLYL", "own", 120)]
    base = flag_units(actual, pairs)
    out = {"actual_120": share_of(actual, pairs)}
    for v in FLY_VARIANTS:
        recs = by.get(v, {})
        if len(recs) < len(actual) and v != "amp1":
            out[v] = dict(missing=f"{len(recs)}/{len(actual)} flies recorded",
                          run=f"scripts/diag_ad.py offline --variant {v}")
            continue
        if not recs:
            out[v] = dict(missing="not run")
            continue
        cell = dict(share_of(recs, pairs), switch_rate=diag_mean(r["rate"] for r in recs.values()),
                    centred_v_by_class=diag.class_means(recs.values(), pairs))
        if v == "amp1":
            cell["reproduces_recorded_v_max_abs_diff"] = max(
                abs(x - y) for g, r in recs.items() for s1, s2 in zip(r["situations"], actual[g]["situations"])
                for x, y in zip(s1["v"], s2["v"]))
        else:
            units = flag_units(recs, pairs)
            cell["vs_actual_120_crossed_paired"] = acd.crossed_boot(units, base, SPEC.boot_draws, SPEC.boot_seed,
                                                                    paired=True)
        out[v] = cell
    out["note"] = ("FLY-L's 12 checkpoint weight vectors after battle 120, modified, re-evaluated on the confirmation "
                   "set with the run's own path and decision seeds (point 120 records); amp<k> = w0 + k (w120 - w0) "
                   "held at the 0.2 w0 floor; only_<DAN> = the learned change kept only on that DAN type's "
                   "core-compartment edges (w0 elsewhere). amp1 = the unmodified weights (reproduction check)")
    return out


def candidates(q1o, q2o) -> list:
    rs = q2o["right_share_by_readout"]
    off = q2o["offline_120"]
    pick = lambda v: off.get(v, {}).get("share")   # noqa: E731
    dl = {p: rs["FLYL"][p]["delta_from_0"]["share"] for p in POINTS if p}
    out = [
        dict(cause="naive w0 class bias (immune / not-very candidates read higher than super-effective ones)",
             question=1, evidence=dict(naive_class_bias=q1o["naive_class_bias"]["centred_v"],
                                       class_model=q1o["class_model"]),
             estimate=dict(metric="FLY-L right-direction share with the naive V removed (V(t) - V(0), same odours "
                                  "and seeds)", by_point=dl, actual=dict((p, rs["FLYL"][p]["own"]["share"]) for p in POINTS)),
             method="records only (delta_record)",
             caveat="a readout that subtracts each odour's naive V is not a mechanism the fly has; it measures the "
                    "direction of the learned change alone. Point 0 is undefined (V - V = 0)"),
        dict(cause="learned change too small: depression balanced by recovery r = 0.001 (equilibrium by ~40 battles)",
             question=2, evidence=dict(recovery=q2o["recovery"], compartments=q2o["compartment_mean_w_over_w0_by_battle_bin"]),
             estimate=dict(metric="FLY-L right-direction share at 120 with the learned change scaled",
                           amp2=pick("amp2"), amp3_4=pick("amp3.4"), actual=off.get("actual_120", {}).get("share")),
             method="offline re-evaluation of the 120-battle weights, w0 + k (w120 - w0)",
             caveat="k = 3.4 is the linear-regime no-recovery equivalent at ~3 200 pulses; it assumes the change's "
                    "direction is the same over the run (with recovery, late pulses weigh more) and ignores tanh "
                    "saturation of the rule. Real no-recovery learning would need new learning runs"),
        dict(cause="which reinforcement carries the right direction (PAM08 reward vs PPL105 punishment compartments)",
             question=2, evidence=dict(pulses_by_class=q2o["pulses_by_class"]["FLYL"]),
             estimate=dict(metric="FLY-L right-direction share at 120 keeping only one compartment's change",
                           only_PAM08=pick("only_PAM08"), only_PPL105=pick("only_PPL105"),
                           actual=off.get("actual_120", {}).get("share")),
             method="offline re-evaluation, change kept on one DAN type's core edges",
             caveat="the two compartments' changes are not independent in the run (each shapes later choices and "
                    "rewards); a different reward / punishment balance needs new learning runs"),
        dict(cause="readout channel (z_A vs z_P carries the bias or the learning)", question=1,
             evidence=dict(naive=dict(centred_z_A=q1o["naive_class_bias"]["centred_z_A"],
                                      centred_minus_z_P=q1o["naive_class_bias"]["centred_minus_z_P"])),
             estimate=dict(metric="right-direction share with one channel only (records)",
                           a_only={p: rs["FLYL"][p]["a_only"]["share"] for p in POINTS},
                           p_only={p: rs["FLYL"][p]["p_only"]["share"] for p in POINTS}),
             method="records only (MBON counts re-read with another readout)",
             caveat="one-channel readouts drop half the signal; z parameters unchanged"),
        dict(cause="opponent channel removed (FLY-L weights on OS odours)", question=1,
             evidence=dict(contrast_symmetry=q1o["contrast_symmetry"]["cells"]),
             estimate=dict(metric="FLY-L os_paired right-direction share",
                           by_point={p: rs["FLYL_os_paired"][p]["share"] for p in POINTS}),
             method="records only (the run's paired OS evaluation)",
             caveat="removing the channel makes the share chance (~0.5 by symmetry) — not a gain in opponent "
                    "conditioning; it only removes the naive bias's sign"),
        dict(cause="plasticity floor / saturation", question=2,
             evidence=dict(floor_contact=q2o["weight_summary_by_point"], weights_at_120=q2o.get("weights_at_120")),
             estimate=dict(metric="gain from removing the floor", value=0.0),
             method="records: no edge reached the 0.2 w0 floor at any point, so removing it changes nothing",
             caveat="the rule's tanh per-step saturation is not examined"),
        dict(cause="point-to-point flips of the per-pair direction (the 100 -> 120 drop)", question=2,
             evidence=dict(transitions=q2o["transitions"]["FLYL"],
                           seed_noise_flag_disagreement_at_0=q1o["seed_noise_at_0"]["flag_majority_disagreement"]),
             estimate=dict(metric="none (a description of turnover, not a fixable cause)"),
             method="records only", caveat="decision seeds are fixed across points; flips come from weight changes "
                                           "amplified by spiking noise"),
    ]
    return out


def check_words(doc) -> None:
    from flymon.ad.spec import FORBIDDEN
    text = json.dumps(doc, ensure_ascii=False)
    bad = [w for w in FORBIDDEN if w in text]
    if bad:
        raise ValueError(f"forbidden words {bad} in the diagnosis output (AC.8)")


def analyse(a) -> int:
    from flymon.ac import store
    from flymon.ac.config import load_lv_config
    from flymon.ad.spec import LABEL, SPEC
    from flymon.brain.h4_pairs import pool_vocabulary
    from flymon.rescope import blocks
    model = json.loads(Path("results/summary/ac_model_manifest.json").read_text())
    r_rec = float(model["recovery_per_pulse"])
    z = load_lv_config(r_rec).z
    pairs = json.loads(Path(INPUTS).read_text())["confirm"]["pairs"]
    st, mi, _, _ = pool_vocabulary()
    lay = SPEC.layout()
    log("reading situation records")
    S = load_sits()
    log("reading turn logs")
    logs = {int(p.stem[3:]): blocks.read_jsonl(p) for p in sorted((ROOT / "logs").glob("fly*.jsonl"))}
    mapr = jl_read(DIAG / "map0.jsonl")
    offl = jl_read(DIAG / "offline.jsonl")
    emap_p = DIAG / "edge_map.npz"
    emap = np.load(emap_p) if emap_p.exists() else None
    z120 = np.load(CKPT)
    w120 = {i: z120[f"w{i}"] for i in range(len(lay))}
    w0 = emap["w0"] if emap is not None else None
    log("question 1")
    q1o = q1(S, pairs, z, mi, st, mapr)
    log("question 2")
    q2o = q2(S, pairs, z, logs, lay, offl, emap, w120 if w0 is not None else None, w0, mapr, r_rec)
    cands = candidates(q1o, q2o)
    missing = [v for v in FLY_VARIANTS if "missing" in q2o["offline_120"].get(v, {})]
    if not mapr:
        missing.append("map0")
    srcs = [ROOT / "logs" / "situations_init.jsonl", ROOT / "logs" / "situations.jsonl", CKPT,
            Path("results/summary/ad_m4.json"), Path(INPUTS), Path("results/summary/ac_model_manifest.json")]
    srcs += sorted((ROOT / "logs").glob("fly*.jsonl"))
    if mapr:
        srcs.append(DIAG / "map0.jsonl")
    if offl:
        srcs.append(DIAG / "offline.jsonl")
    if emap is not None:
        srcs.append(emap_p)
    ad = json.loads(Path("results/summary/ad_m4.json").read_text())
    doc = dict(appendix="AD", section="AD.11", kind="record-only", kind_text=KIND,
               note="기록 전용 추정: no verdict, no AD / AC rule change, no new battle or learning; offline brain "
                    "evaluations of existing weights only",
               label=LABEL, label_matches_ad_m4=(LABEL == ad["label"]), complete=not missing, missing=missing,
               question1=q1o, question2=q2o, candidate_causes=cands,
               sources=[dict(path=str(p), sha256=store.sha256_file(p)) for p in srcs])
    check_words(doc)
    from flymon.ad import store as adstore
    adstore.write_json(a.out, doc)
    log(f"wrote {a.out} (complete={not missing}; missing {missing})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offline")
    o.add_argument("--variant", required=True, choices=("map0",) + FLY_VARIANTS)
    o.add_argument("--flies", default="0-11", help="FLY-L pool indices (0-11)")
    o.add_argument("--workers", type=int, default=16)
    an = sub.add_parser("analyse")
    an.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    return offline(a) if a.cmd == "offline" else analyse(a)


if __name__ == "__main__":
    sys.exit(main())
