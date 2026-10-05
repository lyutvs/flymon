"""A scripted W world for the runner tests (no engine, no pool): a fake ctx (V's summary with the reused blocks, V's
git facts, 16 pilot rows, a 249-row main set honouring block set, P's stimuli, V's judgement rows), a FakePool whose
w_learn_job is a deterministic count model (paired noise per probe seed; PAM08 lowers MBON05(X) by `a`, PPL105 lowers
MBON13(X) by `b`, no effect with plasticity off; the plastic weights' "sha" names the DANs applied), run through the
real WMeasurer and WCache under results/w/cache, an oracle measurer whose results are r_fixtures.fake_oracle (testable
per plan) written as cache files, fast fakes for w_oc.run / synthetic_validation, V's own reuse decision patched to
PASS (V's tests own it), and V's gate-② rows / judgement raw computed from the same count model so the path gate
reproduces unless a test breaks it."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from flymon.brain import w_measure as WM
from flymon.brain import w_runner as WR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.r_measure import RMeasurer
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W_SPEC
from flymon.brain.w_store import WCache
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle
from tests.brain.s_fixtures import stimuli

SPEC = dataclasses.replace(W_SPEC, workers=4)
CODE, TCODE, UCODE = {"key": SPEC.r_shared_key}, {"key": SPEC.t_measure_key_t}, {"key": SPEC.u_measure_key_u}
WCODE, PIPE = {"key": "w" * 64}, {"key": "p" * 64}
ZV = {"A": [16.916666666666668, 12.483878492769072], "P": [80.16666666666667, 29.775432639827233]}


def rows(n_b, n_a, turn0, tag):
    out = []
    for i in range(n_b + n_a):
        ax = "b" if i < n_b else "a"
        out.append(dict(axis=ax, turn=turn0 + i, x=f"{tag}x{i}", y=f"{tag}y{i}", odor_x={f"{tag}G": 1.0 + i},
                        odor_y={f"{tag}H": 1.0 + i}, move_x="M", opp_x=["T"], move_y="M", opp_y=["U"]))
    return out


PILOT = rows(15, 1, 0, "p")
MAIN = [dict(r, c=i) for i, r in enumerate(rows(167, 82, 306, "m"))]
VJ = rows(3, 0, 0, "v")


class Model:
    """The count model: effects[pair] = (a, b); naive offset of MBON13(X) per pair (an unbalanced pair)."""

    def __init__(self):
        self.effects, self.offset, self.c_scale, self.train_s, self.probe_s = {}, {}, 0.5, 0.01, 0.001
        self.share_state, self.state = False, {}     # mutation: RN continues on R's trained brain (same pair, fly)

    def probe(self, pair, seed, edit, applied, plastic):
        r = np.random.default_rng(int(seed))
        n = r.normal(0, 3, 4)
        a, b = self.effects.get(pair, (30.0, 10.0))
        s = 1.0 if edit == V.lever_edit else self.c_scale
        ax = 30 + self.offset.get(pair, 0.0) + n[0]
        px, ay, py = 60 + n[1], 30 + n[2], 60 + n[3]
        if plastic:
            if "PAM08" in applied:
                px -= a * s
            if "PPL105" in applied:
                ax -= b * s
        pr = lambda A, P: dict(seed=int(seed), A=int(max(0, round(A))), P=int(max(0, round(P))), kc_frac=0.05,  # noqa
                               kc_spikes=10, kc_max_win_hz=10.0, apl_out_per_step=0.1, wall_s=0.0, steps=1400)
        return pr(ax, px), pr(ay, py)

    def job(self, kw, pair):
        edit, plastic = kw["edit"], kw["plastic"]
        lever = edit == V.lever_edit
        dans = [p[0] for p in kw["phases"]]
        rn = self.share_state and dans == ["PAM08", None]
        stages, applied = [], list(self.state.get((pair, kw["fly"]), [])) if rn else []
        sha = lambda: hashlib.sha256(f"{pair}|{kw['fly']}|{applied}|{plastic}".encode()).hexdigest()[:16]  # noqa

        def snap(name, da):
            xs, ys = zip(*(self.probe(pair, s, edit, applied, plastic) for s in kw["probe_seeds"]))
            return dict(stage=name, x=list(xs), y=list(ys), w_sha256="w0" if not applied or not plastic else sha(),
                        weights_frac=1.0, weights_frac_A=1.0, weights_frac_P=0.9 if "PAM08" in applied else 1.0,
                        da_integral=da)
        stages.append(snap("pre", None))
        for i, (dan, _trials, _base) in enumerate(kw["phases"]):
            applied.append(dan)
            stages.append(snap(f"S{i + 1}", {"PAM08": 0.1}))
        if self.share_state and dans == ["PAM08", "PPL105"]:
            self.state[(pair, kw["fly"])] = list(applied)
        n_tr = sum(int(p[1]) for p in kw["phases"])
        return dict(edit=edit, csc_sha256=V.sha_combined if lever else V.sha_none, edit_edges=2 if lever else 0,
                    block_edges=dict(V.contrast_declared()["chain_entry"]) if lever else {}, w0_sha256="w0",
                    fly=kw["fly"], plastic=plastic, probe_seeds=list(kw["probe_seeds"]), stages=stages,
                    wall_s=n_tr * self.train_s + len(stages) * self.probe_s, train_s=n_tr * self.train_s,
                    probe_s=2 * len(kw["probe_seeds"]) * len(stages) * self.probe_s)


class FakePool:
    def __init__(self, model, n=4):
        self.model, self.n_workers, self.jobs = model, n, 0
        self.pairs = {}

    def run_jobs(self, fn, kws):
        assert fn is WM.w_learn_job
        self.jobs += len(kws)
        return [self.model.job(kw, self.pairs.get(json.dumps([kw["odor_x"], kw["odor_y"]], sort_keys=True), "?"))
                for kw in kws]


class FakeOracle:
    def __init__(self, world, z, smoke):
        self.w, self.z, self.smoke = world, dict(z), smoke
        self._rm = RMeasurer(None, None, V, Params(), READOUT, self.z, TYPES, 100)
        self.last_wall_s, self.last_jobs = 1.0, 1

    def oracle(self, rs, cond, block, seeds):
        self.w.oracle_calls.append((block, cond.name, len(rs), tuple(self.z["A"])))
        root = Path(SPEC.smoke_cache_dir if self.smoke else SPEC.cache_dir) / "r_oracle"
        root.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rs:
            k = row_key(r)
            ok = self.w.testable.get(k, False)
            res = fake_oracle(ok, ok, False, sha=self.w.oracle_sha, edges=2, edit=cond.edit,
                              n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ins = json.loads(canonical(self._rm.inputs(r, cond, block, seeds)))
            ck = hashlib.sha256(canonical(ins).encode()).hexdigest()
            f = root / f"{ck[:24]}.json"
            f.write_text(json.dumps(dict(key=ck, kind="r_oracle", inputs=ins, result=res)))
            out.append(dict(key=k, result=res, cache_key=ck, cache_file=str(f)))
        return out


def fake_oc(pilot, z, spec, cost, n_boot=None, n_rep=None, n_boot_rep=None, log=None):
    sel = dict(q=0.75, K=8, F=8, cost_h=float(cost(8, 8)))
    alt = dict(q=0.75, K=8, F=9, cost_h=float(cost(8, 9)))
    return dict(selected=None if fake_oc.none else sel, ranking=[] if fake_oc.none else [sel, alt], reachable=True,
                calibration={}, theta=dict(n_pairs=len(pilot)), records=dict(mixed_flies={"g0.0": [0.1] * 5}),
                drift_dprime=dict(population={"reward_level": 0.1}, pilot_pairs=[], level_gates=["reward_level"]),
                power_lo=[[[0.5] * 25] * 2] * 3, false_hi=[[[0.01] * 25] * 2] * 3,
                power_lo_by_k=[[[[0.5] * 5] * 25] * 2] * 3, false_hi_by_k=[[[[0.01] * 5] * 25] * 2] * 3,
                axes=dict(q=list(spec.q_grid), K=list(spec.k_grid), F=list(range(spec.f_min, spec.f_max + 1)),
                          k=list(range(spec.k_min, spec.k_cap + 1)), g=list(spec.cluster_grid)),
                k_values=list(range(spec.k_min, spec.k_cap + 1)),
                timing=dict(point_s=1.0, boot_s=2.0, records_s=1.0, total_s=4.0), n_pilot=len(pilot))


fake_oc.none = False


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.tmp = tmp_path
        self.model = Model()
        self.testable = {row_key(r): (r["c"] % 3 == 0) for r in MAIN}
        self.oracle_calls, self.judge_commits, self.oracle_sha = [], [], V.sha_combined
        self.v = self.v_doc()
        self.v_git = dict(tracked=True, dirty=False, judge_commits=["a279a56"])
        self.facts = dict(last="a279a56" + "0" * 33, ancestors={c: True for _, c in SPEC.v_commits})
        fake_oc.none = False
        monkeypatch.setattr(WR, "summary_git", lambda p: dict(tracked=True, dirty=False,
                                                              judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(WR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(WR.v_rules, "reuse", lambda *a, **k: dict(outcome="PASS", reasons=[]))
        monkeypatch.setattr(WR.w_oc, "run", fake_oc)
        self.synth_calls = []

        def synth(spec, z, n_rep=None):
            self.synth_calls.append(int(spec.synth_reps if n_rep is None else n_rep))
            return dict(ok=True, zero_effect=dict(ok=True))
        monkeypatch.setattr(WR.w_oc, "synthetic_validation", synth)
        self.archive = tmp_path / "archive"
        self.pool = FakePool(self.model)
        self.st = stimuli(V.p)
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100,
                        p_inputs=lambda pspec, smoke_: (dict(c1=0.6), self.st),
                        r_doc=dict, r_git=dict, t_doc=dict, t_git=dict, u_doc=dict, u_git=dict,
                        v_doc=lambda: json.loads(json.dumps(self.v)), v_git=lambda: dict(self.v_git),
                        v_facts=lambda: json.loads(json.dumps(self.facts)), pilot_rows=lambda: list(PILOT),
                        w_set=self.w_set, main_rows=self.main_rows, v_gate2_rows=self.v_gate2_rows, v_jm=self.v_jm,
                        clusters=lambda rs: {row_key(r): "X" for r in rs},
                        c3_params=dict(learn_rate=3e-4, recovery_per_pulse=0.0, kc_kc_scale=0.0))
        for r in PILOT + MAIN + VJ:
            self.pool.pairs[json.dumps([r["odor_x"], r["odor_y"]], sort_keys=True)] = row_key(r)
        for d, (x, y) in V.p.pairs().items():
            self.pool.pairs[json.dumps([self.st[x]["odor"], self.st[y]["odor"]], sort_keys=True)] = f"P{d}"

    # ---- V's summary --------------------------------------------------------------------------------------------
    def v_doc(self) -> dict:
        keys = dict(u_measure_key=SPEC.u_measure_key_u, code_key=SPEC.r_shared_key, t_measure_key=SPEC.t_measure_key_t)
        return dict(z=dict(keys, z_V=ZV, outcome="PASS"), kc_input=dict(keys), set=dict(keys, set={}),
                    judge=dict(keys, band="SELECTED"), even=dict(keys, pairs=[]),
                    **{"jm:L": dict(keys, wall_s=1800.0, jobs=64)})

    # ---- the set callables ---------------------------------------------------------------------------------------
    def js(self):
        keys = [row_key(r) for r in MAIN]
        return dict(rows=MAIN, n_b=167, n_a=82, n=249, first_turn=306, last_turn=MAIN[-1]["turn"],
                    skipped=dict(used=1), clusters_b=[], clusters_a=[], digest_keys="k" * 64, digest_e0_b="b" * 64,
                    digest_e0_a="a" * 64, n_odours=10, all_off_pool=True, keys=keys)

    def w_set(self):
        return self.js()

    def main_rows(self, blk):
        from flymon.brain.w_pairs import check_w_set
        bad = check_w_set(self.js(), blk)
        if bad:
            raise ValueError("; ".join(bad))
        return [dict(r) for r in MAIN]

    # ---- V's raw (computed from the same model) ----------------------------------------------------------------
    def v_gate2_rows(self, items):
        n = V.p.o.n
        out = []
        for i in items:
            kw = dict(edit=i["edit"], plastic=True, fly=i["seed"], probe_seeds=[i["seed"]],
                      phases=[[n.h3.punish_type, int(n.h4.teach_trials), n.train_seed_base]])
            res = self.model.job(kw, f"P{i['direction']}")
            st = {s["stage"]: s for s in res["stages"]}
            out.append(dict(pre=dict(x=st["pre"]["x"][0], y=st["pre"]["y"][0]),
                            post=dict(x=st["S1"]["x"][0], y=st["S1"]["y"][0]), w0_sha256="w0",
                            w_post_sha256=st["S1"]["w_sha256"], weights_frac=1.0, weights_frac_A=1.0,
                            weights_frac_P=st["S1"]["weights_frac_P"], da_integral=st["S1"]["da_integral"],
                            csc_sha256=res["csc_sha256"]))
        return out

    def v_jm(self, name, n):
        edit = V.cond(name).edit
        out = []
        for r in VJ[:n]:
            kw = dict(edit=edit, plastic=True, fly=0, probe_seeds=V.judge_seeds()["report"], phases=[])
            st = self.model.job(kw, row_key(r))["stages"][0]
            res = {"report": {"pre": {"A": [[x["A"], y["A"]] for x, y in zip(st["x"], st["y"])],
                                      "P": [[x["P"], y["P"]] for x, y in zip(st["x"], st["y"])]}}}
            out.append((r, res))
        return out

    # ---- measurers ------------------------------------------------------------------------------------------------
    def measure(self, z, smoke=False):
        return FakeOracle(self, z, smoke)

    def learner(self, smoke=False):
        cache = WCache(SPEC.smoke_cache_dir if smoke else SPEC.cache_dir, WCODE)
        return WM.WMeasurer(self.pool, cache, Params(), READOUT, "MBON05", "PAM08", "PPL105",
                            dict(strength=1.0, settle_ms=800.0, read_ms=600.0, window_ms=200),
                            dict(present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, seed_stride=1000))

    def runner(self, spec=None, code=None, wcode=None):
        return WR.Runner(self.measure, self.learner, self.ctx, spec or SPEC, code=code or CODE, tcode=TCODE,
                         ucode=UCODE, wcode=wcode or WCODE, pipeline=PIPE, archive_root=self.archive)


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through(w, last, smoke_spec=None):
    from flymon.brain.w_spec import smoke
    for s in WR.ORDER[:WR.ORDER.index(last) + 1]:
        r = w.runner(smoke(SPEC) if s == "smoke" else None)
        out = getattr(r, f"stage_{s}")()
        if s in WR.GATES:
            assert out["outcome"] == "PASS", (s, out)
        if s == "smoke":
            assert out["problems"] == [], out["problems"]
    return out
