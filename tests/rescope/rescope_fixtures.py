"""Synthetic probe records for the re-scope primary test. V moves through the P (MBON05) count: lowering P raises V."""
import numpy as np
from flymon.rescope.spec import SPEC


def records(x="a", p_naive=30, a_naive=10, dx=0.0, dy=0.0, null=0.0, noise=0.0, seed=0, pair="p1000",
            n_flies=None, noplast_move=0, drop=None, x_bias=1):
    """dx / dy: drop of MBON05 count on X / Y in Rr at S1 relative to N (positive = learning).
    null: extra P drop of N2 on X relative to N (the null's association).
    x_bias: added to X's MBON05 count in every record (all brains, both stages) so choose_x returns x; 0 gives a tie."""
    rng = np.random.default_rng(seed)
    n_flies = SPEC.n_flies if n_flies is None else n_flies
    y = "b" if x == "a" else "a"
    out = []
    def rec(brain, fly, stage, s, px, py):
        c = {o: {SPEC.a_type: a_naive, SPEC.p_type: int(max(0, round(v + rng.normal(0, noise))))}
             for o, v in ((x, px + x_bias), (y, py))}
        out.append(dict(brain=brain, fly=fly, stage=stage, seed=s, counts=c))
    for f in range(n_flies):
        for s in SPEC.probe_seeds(pair, f):
            for b in ("Rr", "N", "N2"):
                rec(b, f, "pre", s, p_naive, p_naive)
            rec("Rr", f, "S1", s, p_naive - dx, p_naive - dy)
            rec("N", f, "S1", s, p_naive, p_naive)
            rec("N2", f, "S1", s, p_naive - null, p_naive)
    for s in SPEC.probe_seeds(pair, 0):
        rec("noplast", 0, "pre", s, p_naive, p_naive)
        # plasticity off: S1 is the same probe as pre (no fresh noise draw), moved only by noplast_move on X's MBON05
        c = {o: dict(v) for o, v in out[-1]["counts"].items()}
        c[x][SPEC.p_type] += noplast_move
        out.append(dict(brain="noplast", fly=0, stage="S1", seed=s, counts=c))
    if drop is not None:
        out = [r for i, r in enumerate(out) if i != drop]
    return out
