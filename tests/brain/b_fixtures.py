"""Synthetic probe records with a known answer for spec J.12's rules (F.6's method), J.12.7's independent arms. Every fly
shares the naive state; per probe seed there is Gaussian noise shared by all brains of the fly at that seed (paired
presentations), so the arm - N contrasts see only what the fixture puts into an arm."""
import numpy as np

from flymon.brain.b_spec import SPEC

A, P = SPEC.a_type, SPEC.p_type


def records(x="b", naive=dict(ax=30, px=40, ay=20, py=40), reward=0, punish=0, drift_r=0, drift_p=0, spill=0.0,
            n2=False, noise=3.0, brain_noise=1.0, seed=0, n_flies=None, fly_scale=None, noplast_move=False,
            pair="calibration"):
    """reward: spikes the reward arm's X P-type readout loses; punish: spikes the punishment arm's X A-type readout loses;
    drift_r / drift_p: P- / A-type losses of X in every brain that saw the block (Rr, Rp, N, N2) — presentation-induced,
    no DAN; spill: the share of an arm's loss that Y also takes. fly_scale(f) scales one fly's learning. After the naive
    stage every brain adds its own noise (brain_noise): trained weights make each brain's spikes its own. pair: whose
    declared probe seeds the records carry."""
    rng = np.random.default_rng(seed)
    y = "a" if x == "b" else "b"
    n = n_flies or SPEC.n_flies
    out = []
    brains = ["Rr", "Rp", "N"] + (["N2"] if n2 else [])
    for f in range(n):
        k = 1.0 if fly_scale is None else fly_scale(f)
        for s in SPEC.probe_seeds(pair, f):
            e = rng.normal(0, noise, size=4)
            base = {x: {A: naive["ax"] + e[0], P: naive["px"] + e[1]}, y: {A: naive["ay"] + e[2], P: naive["py"] + e[3]}}
            for b in brains:
                for st in ("pre", "S1"):
                    c = {o: dict(v) for o, v in base.items()}
                    if st == "S1":
                        c[x][P] -= drift_r
                        c[x][A] -= drift_p
                        if b == "Rr":
                            c[x][P] -= k * reward
                            c[y][P] -= k * reward * spill
                        if b == "Rp":
                            c[x][A] -= k * punish
                            c[y][A] -= k * punish * spill
                        if brain_noise:
                            c = {o: {t: v + rng.normal(0, brain_noise) for t, v in c[o].items()} for o in c}
                    out.append(dict(brain=b, fly=f, stage=st, seed=int(s),
                                    counts={o: {t: int(max(0, round(v))) for t, v in c[o].items()} for o in ("a", "b")}))
    s0 = SPEC.probe_seeds(pair, 0)
    pre0 = {int(r["seed"]): r["counts"] for r in out if r["brain"] == "Rr" and r["fly"] == 0 and r["stage"] == "pre"}
    for s in s0:
        for st in ("pre", "S1"):
            c = {o: dict(v) for o, v in pre0[int(s)].items()}
            if noplast_move and st == "S1" and s == s0[0]:
                c[x][A] += 1
            out.append(dict(brain="noplast", fly=0, stage=st, seed=int(s), counts=c))
    return out
