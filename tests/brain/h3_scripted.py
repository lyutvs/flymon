"""A scripted stand-in for h3_measure.PoolMeasurer: rows are simple functions of Params and seed, so the H.3 state
machine's branches can be steered one at a time without running the engine."""
import math

from flymon.brain.h3_spec import SPEC

N_KC = 100
POOLS = {"A": ["MA1", "MA2"], "P": ["MP1", "MP2"]}
ODORS = [dict(name=f"R{j:02d}", types=["X"], strengths={"X": 1.0}, seeds=[800_100 + 2 * j, 800_101 + 2 * j])
         for j in range(48)]


def default_mv(p):
    """Membrane rises with apl_input_scale and falls slightly with the hold (mimics the recalibration shift)."""
    return 50.0 * p.apl_input_scale * (1.0 + 0.1 * (0.85 - p.mbon_hold_frac)) + (p.kc_thresh - 1.6)


def default_hz(p, seed):
    return 28.0 * (p.mbon_hold_frac - 0.725) + 0.3 * math.sin(seed)


class Scripted:
    def __init__(self, mv=default_mv, kc=lambda p: 0.06, margin=lambda p, seed: 0.003 + 0.001 * math.sin(seed),
                 pct=lambda p: (0.058, 0.045), type_count=lambda p, name, j, s: 10, hz=default_hz,
                 rest_kc_over=lambda p, seed: 0, odor_kc_over=lambda p, seed: 0, fired=None):
        self.mv, self.kc, self.margin, self.pct, self.type_count = mv, kc, margin, pct, type_count
        self.hz, self.rest_kc_over, self.odor_kc_over = hz, rest_kc_over, odor_kc_over
        self.fired = fired or (lambda p, j, s: list(range(int(round(self.kc(p) * N_KC)))))
        self.calls = []

    def reference(self, params, which="reference", csc_edit=None):
        self.calls.append(("reference", params, which, csc_edit))
        rows = []
        for j, o in enumerate(ODORS):
            for s, seed in enumerate(o["seeds"]):
                rows.append(dict(odor=o["name"], seed=seed, apl_v_mean=self.mv(params), kc_active_frac=self.kc(params),
                                 release_frac=0.5, release_mean=0.5 * params.apl_r_max, fired=self.fired(params, j, s),
                                 apl_v_quantiles=[self.mv(params)] * len(SPEC.apl_v_quantiles),
                                 apl_input_by_class=dict(kc=1.0, alpn=0.1, mbon=0.05, other=0.0),
                                 apl_to_type_input={"MBON05": 0.2, "MBON13": 0.1},
                                 types={n: self.type_count(params, n, j, s) for k in POOLS for n in POOLS[k]}))
        return rows

    def rest(self, params, seeds, csc_edit=None):
        self.calls.append(("rest", params, tuple(seeds), csc_edit))
        return [dict(seed=int(s), types={n: 0 for k in POOLS for n in POOLS[k]}) for s in seeds]

    def design(self, params, seeds):
        self.calls.append(("design", params, tuple(seeds)))
        fa, fb = self.pct(params)
        return [dict(seed=int(s), frac_active_A=fa, frac_active_B=fb, jaccard=0.02, chance=0.02 + self.margin(params, s),
                     kc_hz_A=1.0, kc_hz_B=1.0, mbon_hz_A=15.0, mbon_hz_B=16.0) for s in seeds]

    def baseline(self, params, seeds, sat_hz=None):
        self.calls.append(("baseline", params, tuple(seeds), sat_hz))
        return [dict(seed=int(s), trimmed=self.hz(params, s), raw=self.hz(params, s),
                     n_kc_over_sat=self.rest_kc_over(params, s)) for s in seeds]

    def odor_runaway(self, params, seeds):
        self.calls.append(("odor_runaway", params, tuple(seeds)))
        return [dict(seed=int(s), n_kc_over_sat=self.odor_kc_over(params, s), n_kc_over_record=0,
                     kc_hz_top5=[90.0, 80.0, 70.0, 60.0, 50.0], band=[]) for s in seeds]

    def all51(self, params):
        self.calls.append(("all51", params))
        return [dict(g=f"G{g}", seed=200, kc=10 * g, kc_on=g, pn=5) for g in range(5)]

    def seeds_asked(self, method):
        return {tuple(c[2]) for c in self.calls if c[0] == method}
