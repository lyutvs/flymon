#!/usr/bin/env python3
"""Spec AC.11 (record-only): why criterion 1's FLY - FLY-TB difference is small while FLY - C-off is not. Re-reads the
existing M4 stage-1 records (situation evaluations, learning / evaluation decision logs, reinforce / outcome records,
the end-of-learning checkpoint) and writes results/summary/ac_m4_diag.json. No battle is run, no brain is evaluated,
no verdict is made; every number is descriptive and carries the AC.5 label.

    uv run python scripts/diag_ac_m4.py"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from flymon.ac import boot, diag, stage1, store
from flymon.ac.spec import FORBIDDEN, LABEL, SPEC

OUT = "results/summary/ac_m4_diag.json"
INPUTS = "results/summary/ac_inputs.json"
ARMS = ("FLY", "COFF", "TB")


def log(msg: str) -> None:
    print(msg, flush=True)


def vocab():
    from poke_env.data.normalize import to_id_str

    from flymon.battle.pool import POOL
    from flymon.brain.h4_pairs import pool_vocabulary
    st, mi, _, _ = pool_vocabulary()
    ids = {to_id_str(a): a for m in POOL for a in m.attacks}
    return st, mi, ids


def read_logs(root: Path, sub: str = "") -> dict:
    from flymon.rescope import blocks
    out = {}
    for p in sorted((root / "BRAIN" / "logs" / sub).glob("fly*.jsonl")):
        out[int(p.stem[3:])] = blocks.read_jsonl(p)
    return out


def arm_of(lay) -> dict:
    return {g: arm for g, (arm, _) in enumerate(lay)}


# ---------------------------------------------------------------- (i) opponent-type channel
def section_i(sits, pairs, cb, mi) -> dict:
    n = len(pairs)
    chance = diag.chance_switch(pairs)
    by = defaultdict(list)
    for r in sits:
        by[(r["arm"], r["point"])].append(r)
    cells = {}
    for (arm, point), recs in sorted(by.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1])):
        contrasts = [d for r in recs for d in diag.best_contrast(r, pairs)]
        cells[f"{arm}@{point}"] = dict(
            arm=arm, point=point, n_flies=len(recs),
            switch_rate=diag.mean(r["rate"] for r in recs),
            side_correct=diag.mean(x for r in recs for x in diag.side_correct(r)),
            pick_changed_across_sides=diag.mean(x for r in recs for x in diag.pick_changed(r, n)),
            best_contrast_mean=diag.mean(contrasts),
            best_contrast_pos_share=diag.mean(int(d > 0) for d in contrasts),
            margin_mean=diag.mean(diag.margin(s["v"]) for r in recs for s in r["situations"]),
            abs_dv_centred_across_sides=diag.mean(x for r in recs for x in diag.abs_side_delta(r, n, "v", True)),
            abs_da_across_sides=diag.mean(x for r in recs for x in diag.abs_side_delta(r, n, "a")),
            abs_dp_across_sides=diag.mean(x for r in recs for x in diag.abs_side_delta(r, n, "p")),
            abs_dkc_across_sides=diag.mean(x for r in recs for x in diag.abs_side_delta(r, n, "kc_active")),
            mean_a=diag.mean(x for r in recs for s in r["situations"] for x in s["a"]),
            mean_p=diag.mean(x for r in recs for s in r["situations"] for x in s["p"]),
            mean_kc=diag.mean(x for r in recs for s in r["situations"] for x in s["kc_active"]),
            ties=sum(r.get("ties", 0) for r in recs))
    # point-0 noise: within an arm group the flies share w0 and the odours; only the decision seed (fly id) differs
    noise_fly = diag.noise_spread(by[("FLY", 0)] + by[("COFF", 0)], n)
    noise_tb = diag.noise_spread(by[("TB", 0)], n)
    jac = [diag.glom_jaccard(cb, mi[c][0], p["o1_types"], p["o2_types"]) for p in pairs for c in p["cands"]]
    return dict(
        chance_switch_mean=float(np.mean(chance)),
        chance_note="1/k^2 per pair (uniform independent pick on each side); a fixed opponent-blind pick switches 0",
        n_pairs_by_k={str(k): sum(len(p["cands"]) == k for p in pairs) for k in (2, 3)},
        cells=cells,
        point0_noise=dict(fly_and_coff=noise_fly, tb=noise_tb,
                          note="point 0: FLY 0-11 + C-off 12-17 (FLY odours) and FLY-TB 18-23 (TB odours) each share "
                               "w0 and the odours; only the decision seed differs -> the spread is simulation noise"),
        fly_odour_glom_jaccard_across_sides=dict(mean=float(np.mean(jac)), share_zero=float(np.mean(np.array(jac) == 0)),
                                                 note="FLY-TB: 1.0 by construction (odour of move type only)"),
        tb_design="FLY-TB odour = mean over the 16 POOL type sets of the E-grid odour of the move type, rescaled to the "
                  "FLY mean total strength (flymon/ac/tb.py): the opponent-type channel is removed; the two sides of a "
                  "pair present identical odours, so any TB pick change across sides comes from simulation noise")


# ---------------------------------------------------------------- (ii) what each arm learned
def situation_items(rec, pairs, mi):
    sd = diag.sides(rec)
    for i, p in enumerate(pairs):
        pw = [mi[c][1] for c in p["cands"]]
        for side, mults in ((0, p["mults1"]), (1, p["mults2"])):
            yield sd[(i, side)]["pick"], diag.rule_picks(pw, mults)


def decision_items(recs, st, mi, ids):
    from flymon.ac.confirm import type_mult
    for r in recs:
        if r.get("kind") != "decision" or r.get("decider") != "fly" or not r.get("candidates"):
            continue
        cands = [ids[c] for c in r["candidates"]]
        if r["chosen"] not in r["candidates"]:
            continue
        mults = r.get("multipliers") or [type_mult(mt, ot) for mt, ot in r["egrid"]]
        yield r, cands, r["candidates"].index(r["chosen"]), [mi[c][1] for c in cands], [float(m) for m in mults]


def section_ii(sits, pairs, mi, st, ids, learn_logs, eval_logs, arms, ckpt) -> dict:
    out = {}
    # rule agreement on the confirmation situations, per arm and point
    by = defaultdict(list)
    for r in sits:
        by[(r["arm"], r["point"])].append(r)
    out["situation_rules"] = {f"{a}@{p}": diag.rule_agreement(x for r in rs for x in situation_items(r, pairs, mi))
                              for (a, p), rs in sorted(by.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1]))}
    # per-move centred V shift 0 -> 40 and centred V by multiplier class
    shift, by_class = {}, {}
    for arm in ("FLY", "TB"):
        acc = defaultdict(lambda: defaultdict(list))
        cls = defaultdict(lambda: defaultdict(list))
        for r in sits:
            if r["arm"] != arm or r["point"] not in (0, 40):
                continue
            for s in r["situations"]:
                p = pairs[s["pair"]]
                mults = p["mults1"] if s["side"] == 0 else p["mults2"]
                for c, v, m in zip(p["cands"], diag.centred(s["v"]), mults):
                    acc[c][r["point"]].append(v)
                    cls[diag.mult_class(m)][r["point"]].append(v)
        shift[arm] = {c: dict(v0=diag.mean(d[0]), v40=diag.mean(d[40]), shift=diag.mean(d[40]) - diag.mean(d[0]),
                              n=len(d[40])) for c, d in sorted(acc.items())}
        by_class[arm] = {k: dict(v0=diag.mean(d[0]), v40=diag.mean(d[40])) for k, d in sorted(cls.items())}
    out["centred_v_shift_by_move"] = shift
    out["centred_v_by_mult_class"] = by_class
    # decisions: learning by quarter (battles 0-9, ..., 30-39) and evaluation, per arm
    dec = {}
    for name, logs in (("learn", learn_logs), ("eval", eval_logs)):
        groups = defaultdict(list)
        for g, recs in logs.items():
            for r, cands, ci, pw, mu in decision_items(recs, st, mi, ids):
                key = arms[g]
                if name == "learn":
                    key = f"{key}@q{int(r['battle_id'].rsplit('-b', 1)[1]) // 10}"
                groups[key].append((ci, diag.rule_picks(pw, mu)))
        dec[name] = {k: diag.rule_agreement(v) for k, v in sorted(groups.items())}
    out["decision_rules"] = dec
    # evaluation: multiplier-best choice by opponent type set (info turns only), FLY vs TB vs C-off
    opp = defaultdict(lambda: defaultdict(list))
    for g, recs in eval_logs.items():
        for r, cands, ci, pw, mu in decision_items(recs, st, mi, ids):
            if len(set(mu)) < 2:
                continue
            ot = "/".join(r["egrid"][0][1])
            opp[ot][arms[g]].append(int(mu[ci] == max(mu)))
    out["eval_mult_best_by_opp_types"] = {ot: {a: dict(n=len(v), share=diag.mean(v)) for a, v in sorted(d.items())}
                                          for ot, d in sorted(opp.items())}
    # weights: end of learning vs naive (C-off flies keep w0; checked)
    if ckpt is not None:
        z = np.load(ckpt)
        ws = {g: z[f"w{g}"] for g in arms}
        coff = [g for g in arms if arms[g] == "COFF"]
        w0 = ws[coff[0]]
        same = all(np.array_equal(ws[g], w0) for g in coff)
        per = defaultdict(list)
        for g, w in ws.items():
            per[arms[g]].append(diag.weight_summary(w, w0, SPEC.floor_ratio))
        out["weights"] = dict(checkpoint=str(ckpt), naive_from_coff=coff[0], coff_all_equal=bool(same),
                              per_arm={a: {k: diag.mean(x[k] for x in v) for k in v[0]} for a, v in per.items()})
    # floor contact recorded at the situation evaluations
    fc = defaultdict(list)
    for r in sits:
        fc[f"{r['arm']}@{r['point']}"].append(r.get("floor_contact"))
    out["floor_contact"] = {k: diag.mean(v) for k, v in fc.items()}
    return out


# ---------------------------------------------------------------- (iii) reward / punishment vs matchup
def section_iii(learn_logs, arms, st, mi, ids) -> dict:
    from flymon.ac.confirm import type_mult
    out = {}
    conf = defaultdict(lambda: dict(both=0, agree=0, power_vals=[], mult_vals=[]))
    for arm in ARMS:
        rows = []
        for g, recs in learn_logs.items():
            if arms[g] != arm:
                continue
            for t in diag.join_turns(recs):
                d = t["decision"]
                if not d.get("candidates") or d["chosen"] not in d["candidates"]:
                    continue
                cands = [ids[c] for c in d["candidates"]]
                mults = [type_mult(mt, ot) for mt, ot in d["egrid"]]
                pw = [mi[c][1] for c in cands]
                ci = d["candidates"].index(d["chosen"])
                rp = diag.rule_picks(pw, mults)
                c = conf[arm]
                if len(set(mults)) > 1 and rp["power"] is not None:
                    c["both"] += 1
                    c["agree"] += int(rp["power"] == rp["mult"])
                c["power_vals"].extend(pw)
                c["mult_vals"].extend(diag.log2_mult(m) for m in mults)
                if "reinforce" not in t:
                    continue
                r, p = diag.pulse_ms(t["reinforce"]["pulses"])
                oc = t.get("outcome", {}).get("outcome", {})
                rows.append(dict(move=cands[ci], mult=mults[ci], power=pw[ci], reward=r, punish=p, net=r - p,
                                 dealt=oc.get("dealt_frac")))
        by_cls = defaultdict(list)
        for x in rows:
            by_cls[diag.mult_class(x["mult"])].append(x)
        by_move = defaultdict(list)
        for x in rows:
            by_move[x["move"]].append(x)
        lm, pw = [diag.log2_mult(x["mult"]) for x in rows], [x["power"] for x in rows]
        out[arm] = dict(
            n_turns=len(rows),
            by_mult_class={k: dict(n=len(v), share=len(v) / len(rows), reward_ms=diag.mean(x["reward"] for x in v),
                                   punish_ms=diag.mean(x["punish"] for x in v), net_ms=diag.mean(x["net"] for x in v))
                           for k, v in sorted(by_cls.items())},
            by_move={k: dict(n=len(v), net_ms=diag.mean(x["net"] for x in v), reward_ms=diag.mean(x["reward"] for x in v))
                     for k, v in sorted(by_move.items())},
            spearman_net_vs_log2mult=diag.spearman(lm, [x["net"] for x in rows]),
            spearman_net_vs_power=diag.spearman(pw, [x["net"] for x in rows]),
            spearman_reward_vs_log2mult=diag.spearman(lm, [x["reward"] for x in rows]),
            spearman_reward_vs_power=diag.spearman(pw, [x["reward"] for x in rows]),
            spearman_power_vs_log2mult_chosen=diag.spearman(pw, lm))
        c = conf[arm]
        out[arm]["offered_confound"] = dict(
            info_turns_with_unique_power=c["both"],
            power_pick_equals_mult_pick=(c["agree"] / c["both"] if c["both"] else None),
            spearman_power_vs_log2mult_all_candidates=diag.spearman(c["power_vals"], c["mult_vals"]))
    out["pulse_rule"] = ("agent/pulses.py: reward PAM08 400 ms x dealt_frac (+200 ms on a faint); punishment PPL105 "
                         "200 ms resisted, 400 ms immune; super-effective has no separate term (only via damage)")
    return out


# ---------------------------------------------------------------- (iv) per-pair overlap
def section_iv(sits, pairs, mi) -> dict:
    last = max(SPEC.sit_points)
    pick = {("FLY", last): [], ("TB", last): [], ("COFF", 0): [], ("FLY", 0): [], ("TB", 0): []}
    chg = defaultdict(list)
    for r in sits:
        k = (r["arm"], r["point"])
        if k in pick:
            pick[k].append(r["switched"])
            chg[k].append(diag.pick_changed(r, len(pairs)))
    frac = {f"{a}@{p}": np.mean(np.array(v, float), axis=0) for (a, p), v in pick.items()}
    chgf = {f"{a}@{p}": np.mean(np.array(v, float), axis=0) for (a, p), v in chg.items()}
    chance = diag.chance_switch(pairs)
    rows = []
    for i, p in enumerate(pairs):
        pw = [mi[c][1] for c in p["cands"]]
        pp = diag.unique_argmax(pw)
        pname = p["cands"][pp] if pp is not None else None
        power_side = [s for s, b in ((0, p["best1"]), (1, p["best2"])) if pname == b]
        rows.append(dict(pair=i, me=p["me"], o1=p["o1"], o2=p["o2"], cands=p["cands"], best1=p["best1"],
                         best2=p["best2"], k=len(p["cands"]), chance=chance[i], power_pick=pname,
                         power_right_side=power_side[0] if power_side else None,
                         **{f"switch_{k}": float(v[i]) for k, v in frac.items()},
                         **{f"pick_changed_{k}": float(v[i]) for k, v in chgf.items()},
                         gap_fly_coff_contrib=float(frac[f"FLY@{last}"][i] - frac["COFF@0"][i]) / len(pairs),
                         gap_fly_tb_contrib=float(frac[f"FLY@{last}"][i] - frac[f"TB@{last}"][i]) / len(pairs)))
    f40, t40, c0 = (np.array([r[f"switch_{k}"] for r in rows]) for k in (f"FLY@{last}", f"TB@{last}", "COFF@0"))
    ch = np.array(chance)
    groups = {}
    for name, sel in (("power_right_one_side", [r["power_right_side"] is not None for r in rows]),
                      ("power_right_neither", [r["power_right_side"] is None for r in rows]),
                      ("k2", [r["k"] == 2 for r in rows]), ("k3", [r["k"] == 3 for r in rows])):
        m = np.array(sel)
        groups[name] = dict(n=int(m.sum()), fly40=float(f40[m].mean()), tb40=float(t40[m].mean()),
                            coff0=float(c0[m].mean()), chance=float(ch[m].mean()))
    top = sorted(rows, key=lambda r: -r["gap_fly_coff_contrib"])[:6]
    return dict(
        pairs=rows, groups=groups,
        pairs_driving_fly_coff_gap=[dict(pair=r["pair"], me=r["me"], contrib=r["gap_fly_coff_contrib"]) for r in top],
        both_any=int(np.sum((f40 > 0) & (t40 > 0))), only_fly_any=int(np.sum((f40 > 0) & (t40 == 0))),
        only_tb_any=int(np.sum((f40 == 0) & (t40 > 0))), neither_any=int(np.sum((f40 == 0) & (t40 == 0))),
        fly_ge_half=int(np.sum(f40 >= 0.5)), tb_ge_half=int(np.sum(t40 >= 0.5)),
        pearson_pairs_fly40_vs_tb40=diag.pearson(f40, t40), pearson_pairs_tb40_vs_chance=diag.pearson(t40, ch),
        pearson_pairs_fly40_vs_chance=diag.pearson(f40, ch),
        note="'any' = at least one fly of the arm switched on that pair (FLY 12 flies, FLY-TB 6: unequal chance)")


# ---------------------------------------------------------------- bootstrap (descriptive)
def boot_section(sits, pairs) -> dict:
    last = max(SPEC.sit_points)
    units = defaultdict(dict)
    for r in sits:
        k = f"{r['arm']}@{r['point']}"
        d = diag.best_contrast(r, pairs)
        units[("contrast_pos", k)][r["fly"]] = [int(x > 0) for x in d]
        units[("pick_changed", k)][r["fly"]] = diag.pick_changed(r, len(pairs))
    out = []
    for metric in ("contrast_pos", "pick_changed"):
        for a, b in ((f"FLY@{last}", f"TB@{last}"), (f"FLY@{last}", "FLY@0"), (f"TB@{last}", "TB@0"),
                     ("FLY@0", "COFF@0")):
            log(f"  bootstrap {metric} {a} - {b}")
            res = boot.two_stage_diff(units[(metric, a)], units[(metric, b)], SPEC.boot_draws, SPEC.boot_seed)
            out.append(dict(metric=metric, a=a, b=b, **res, kind="descriptive"))
    return dict(seed=SPEC.boot_seed, draws=SPEC.boot_draws, method="flymon/ac/boot.two_stage_diff (descriptive)",
                contrasts=out)


def check_words(doc) -> None:
    text = json.dumps(doc, ensure_ascii=False)
    bad = [w for w in FORBIDDEN if w in text]
    if bad:
        raise ValueError(f"forbidden words {bad} in the diagnosis output (AC.8)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=SPEC.out_root)
    ap.add_argument("--inputs", default=INPUTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-weights", action="store_true", help="skip the checkpoint weight summary")
    a = ap.parse_args(argv)
    root = Path(a.root)
    from flymon.ac.config import lv_codebook
    pairs = json.loads(Path(a.inputs).read_text())["confirm"]["pairs"]
    arms = arm_of(stage1.layout(SPEC))
    st, mi, ids = vocab()
    cb, _ = lv_codebook()
    log("reading situation records")
    sits = stage1.SituationLog(root / "BRAIN" / "logs").records()
    for r in sits:
        if r.get("arm") != arms[r["fly"]]:
            raise SystemExit(f"fly {r['fly']}: arm {r.get('arm')} != layout {arms[r['fly']]}")
    log("reading decision logs")
    learn_logs, eval_logs = read_logs(root), read_logs(root, "eval")
    log("section (i)")
    s1 = section_i(sits, pairs, cb, mi)
    log("section (ii)")
    ckpt = None if a.no_weights else root / "BRAIN" / "checkpoints" / "learn" / "state_000960.npz"
    s2 = section_ii(sits, pairs, mi, st, ids, learn_logs, eval_logs, arms, ckpt)
    log("section (iii)")
    s3 = section_iii(learn_logs, arms, st, mi, ids)
    log("section (iv)")
    s4 = section_iv(sits, pairs, mi)
    log("bootstrap")
    bs = boot_section(sits, pairs)
    c1 = json.loads(Path("results/summary/ac_m4_stage1.json").read_text())["criteria"][:2]
    doc = dict(appendix="AC", section="AC.11", kind="record-only diagnosis (no verdict, no rule change)", label=LABEL,
               criterion1=[{k: c[k] for k in ("a", "b", "diff", "lo", "hi", "n_a", "n_b")} for c in c1],
               layout={arm: [g for g in arms if arms[g] == arm] for arm in ARMS},
               opponent_channel=s1, learned=s2, reinforcement=s3, pairs=s4, bootstrap=bs,
               brain_evaluation="none (records only)",
               sources=dict(stage1_result=dict(path=str(root / "BRAIN" / "result.json"),
                                               sha256=store.sha256_file(root / "BRAIN" / "result.json")),
                            inputs=dict(path=a.inputs, sha256=store.sha256_file(a.inputs))))
    check_words(doc)
    store.write_json(a.out, doc)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
