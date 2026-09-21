"""A scripted stand-in for h4_measure.H4Measurer: every row is a simple function of the combination, so the H.4
procedure's branches can be steered one at a time without running the engine. Combinations are told apart by their
Params' kc_thresh (C0 1.5, C1 1.6, C3 1.7)."""
from flymon.brain.config import Params
from flymon.brain.h4_spec import SPEC

POOLS = {"A": ["MA1", "MA2"], "P": ["MP1", "MP2"]}
TYPES = POOLS["A"] + POOLS["P"]
COMBOS = {"C0": Params(), "C1": Params(kc_thresh=1.6), "C3": Params(kc_thresh=1.7)}
NAME = {p: n for n, p in COMBOS.items()}
SEEDS = [1000 + j for j in range(8)]                     # the reference set's seeds (8 presentations here)
PAIRS = ([("a", 2 * (j % 4), f"ax{j}", f"ay{j}") for j in range(4)]
         + [("b", 2 * (j % 4), f"bx{j}", f"by{j}") for j in range(5)])


def report(testable=True, n=8):
    """d_pre 0; testable: R1 lowers P(X) by 20 + (s % 2), R2 lowers A(X) the same (m ~ 29); else no change (m = 0)."""
    step = [20 + (s % 2) for s in range(n)] if testable else [0] * n
    pre = {"A": [[20 + (1 if s % 2 else -1), 20] for s in range(n)], "P": [[40, 40]] * n}
    r1 = {"A": pre["A"], "P": [[40 - d, 40] for d in step]}
    r2 = {"A": [[a - d, b] for (a, b), d in zip(pre["A"], step)], "P": r1["P"]}
    return {"pre": pre, "R1": r1, "R2": r2}


def guard_of(scripted, name):
    """The H.3 guard statistics the scripted reference/rest rows give (what block "h3" would hold)."""
    from flymon.brain.h3_rules import mbon_type_stats
    p = COMBOS[name]
    ref, rest = scripted.reference(p), {r["seed"]: r for r in scripted.rest(p, SEEDS)}
    scripted.calls.clear()
    return {t: mbon_type_stats(ref, rest, t, 5.0, 0.25) for t in TYPES}


class Scripted:
    def __init__(self, react=None, teach=None, testable=None, rows=None):
        """react(name) -> set of reactive types; teach(name) -> set of teachable types; testable(name) -> (n_b of 5
        testable, n_a of 4 testable and naive); rows(name, rows) -> rows (to corrupt them)."""
        self.react_fn = react or (lambda n: {"MA1", "MP1"})
        self.teach_fn = teach or (lambda n: set(TYPES))
        self.testable_fn = testable or (lambda n: (3, 2))
        self.rows_fn = rows or (lambda n, r: r)
        self.calls = []

    def reference(self, params):
        self.calls.append(("reference", NAME[params]))
        on = self.react_fn(NAME[params])
        return [dict(odor=f"R{j:02d}", seed=s, types={t: (10 + j if t in on else 0) for t in TYPES})
                for j, s in enumerate(SEEDS)]

    def rest(self, params, seeds):
        self.calls.append(("rest", NAME[params]))
        return [dict(seed=int(s), types={t: 1 for t in TYPES}) for s in seeds]

    def teach(self, params):
        self.calls.append(("teach", NAME[params]))
        ok = self.teach_fn(NAME[params])
        rows = []
        for order in ("ab", "ba"):
            for arm in ("punish_only", "reward_only"):
                for s in SPEC.teach_seeds:
                    pre = {cs: {t: 10 for t in TYPES} for cs in ("plus", "minus")}
                    post = {cs: {t: (5 if t in ok else 10) for t in TYPES} for cs in ("plus", "minus")}
                    rows.append(dict(seed=s, arm=arm, order=order, pre=pre, post=post))
        return rows

    def oracle(self, params, readout, z):
        name = NAME[params]
        self.calls.append(("oracle", name, tuple(sorted(readout.items()))))
        n_b, n_a = self.testable_fn(name)
        rows, ia, ib = [], 0, 0
        for ax, t, x, y in PAIRS:
            ok = (ia < n_a) if ax == "a" else (ib < n_b)
            ia, ib = ia + (ax == "a"), ib + (ax == "b")
            rows.append(dict(axis=ax, turn=t, x=x, y=y, report=report(ok)))
        return self.rows_fn(name, rows)
