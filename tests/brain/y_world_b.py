"""A scripted Y world for phase B's runner tests (no engine, no FlyPool): the committed phase-A blocks digest …
precheck written as a summary (block precheck's calibration is the real calibrate_y on the world's θ̂ so the
precheck-reproduction check holds), results/y/oracle.json for W's fake 249-row main set (every third row lenient),
W's count model and FakePool (tests/brain/w_world) behind the real WMeasurer and YCache, a smoke oracle writing
r_fixtures.fake_oracle cache files under results/y/smoke/cache, fast fakes for the bootstrap / reconfirmation draws
(module-level, so the pool path can pickle them) and git facts patched clean."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from flymon.brain import w_measure as WM
from flymon.brain import w_oc, x_oc
from flymon.brain import w_runner as WR
from flymon.brain import y_oc
from flymon.brain import y_runner as YR
from flymon.brain import y_store as YS
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical, sha256_file
from flymon.brain.r_measure import RMeasurer
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle
from tests.brain.w_world import MAIN, ZV, FakePool, Model, rows

YSPEC = dataclasses.replace(Y, workers=1, boot_draws=4, boot_reps=2, reconfirm_reps=2, oc_reps=2, oc_chunk=2,
                            precheck_reps=2)
COSTS = dict(trial_s=0.001, presentation_s=0.0001, oracle_round_s=1.0, workers=16)
SMOKE_ROWS = rows(4, 0, 0, "s")


def lenient(c: int) -> bool:
    return c % 3 == 0


def _cal_ok():
    knob = dict(status="ok", coarse_step=False, stage=None)
    return dict(ok=True, failure=None, corners=[dict(a=0.0, b=0.0, w=1.0)], a=knob, b=[knob])


def fake_boot(theta, z, ys, r, bi, n_rep=None, power=1.0, false=0.0):
    """A bootstrap draw with P(PASS) = power at d′ 1.5 and false at 0.5 everywhere (counts of n_rep)."""
    n = ys.boot_reps if n_rep is None else n_rep
    shp = x_oc.grid_shape(ys)
    G, S = len(ys.cluster_grid), len(y_oc.SCENARIOS)
    hits = np.zeros((G, 2, S) + shp, np.uint16)
    hits[:, 0], hits[:, 1] = int(round(power * n)), int(round(false * n))
    fills = np.zeros((G, 2, S, shp[-1]))
    return dict(arrays=dict(hits=hits, fills=fills), meta=dict(cal=dict(min=_cal_ok(), max=_cal_ok()), n_rep=n))


def fake_boot_fail(theta, z, ys, r, bi, n_rep=None):
    return fake_boot(theta, z, ys, r, bi, n_rep, power=0.0, false=0.0)


def fake_reconfirm(theta, z, ys, r, d, rank, bi, n_rep=None):
    """Passes unless the design's rank is in fake_reconfirm.fail."""
    yd = y_oc.reconfirm_spec(ys, d)
    ok = int(rank) not in fake_reconfirm.fail
    return fake_boot(theta, z, yd, r, bi, ys.reconfirm_reps, power=1.0 if ok else 0.0)


fake_reconfirm.fail = set()


class SmokeOracle:
    def __init__(self, world):
        self.w, self.z = world, dict(W.z_v())
        self._rm = RMeasurer(None, None, V, Params(), READOUT, self.z, TYPES, 100)

    def oracle(self, rs, cond, block, seeds):
        root = Path(YSPEC.smoke_cache_dir) / "r_oracle"
        root.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rs:
            res = fake_oracle(True, True, False, sha=self.w.oracle_sha, edges=V.lever_edges, edit=cond.edit,
                              n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ins = json.loads(canonical(self._rm.inputs(r, cond, block, seeds)))
            ck = hashlib.sha256(canonical(ins).encode()).hexdigest()
            f = root / f"{ck[:24]}.json"
            f.write_text(json.dumps(dict(key=ck, kind="r_oracle", inputs=ins, result=res)))
            out.append(dict(key=row_key(r), result=res, cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch, ys=YSPEC, elapsed_s=3600.0):
        monkeypatch.chdir(tmp_path)
        self.ys = dataclasses.replace(ys, archive_root=str(tmp_path / "arch"))
        self.model, self.oracle_sha = Model(), V.sha_combined
        self.pool = FakePool(self.model)
        for r in MAIN + SMOKE_ROWS:
            self.pool.pairs[json.dumps([r["odor_x"], r["odor_y"]], sort_keys=True)] = row_key(r)
        kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
        self.wpilot = w_oc.synthetic_pilot(np.random.default_rng(5), n_pair=16, **kw)
        self.wkeys = list(Y.pilot_w_pairs) + [f"b|{40 + i}|w{i}|y{i}" for i in range(13)]
        self.theta_w = w_oc.fit(self.wpilot)
        self.vdata = dict(zip(Y.pilot_v_pairs, w_oc.synthetic_pilot(np.random.default_rng(6), n_pair=4, **kw)))
        self.keys = list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs)[:3]
        self.cands = dict(zip(self.wkeys, self.wpilot))
        self.cands.update(self.vdata)
        self.r = y_oc.r_v0(self.theta_w)
        self.theta = y_oc.fit_y([self.cands[k] for k in self.keys], self.keys, self.r)
        self.z = {k: (float(v[0]), float(v[1])) for k, v in ZV.items()}
        self.w_doc = dict(reuse=dict(z_V=ZV))
        self.why = []
        self.judge_commits = []
        self.facts = dict(x_doc=dict(stage0=dict(decision_files={"x.py": "1"}), precheck=dict(outcome="PASS")),
                          ancestors={c: True for _, c in Y.x_commits}, last=Y.x_commits[-1][1] + "0" * 33,
                          git=dict(tracked=True, dirty=False), diag_sha=Y.x_precheck_diag_sha256,
                          x_files={"x.py": "1"}, w_ancestors={c: True for _, c in Y.w_commits},
                          w_blocks={b: Y.w_measure_key for b, _ in Y.w_commits})
        monkeypatch.setattr(YR, "summary_git", lambda p: dict(tracked=True, dirty=False,
                                                              judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(YR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(YR.y_oc, "boot_draw", fake_boot)
        monkeypatch.setattr(YR.y_oc, "reconfirm_draw", fake_reconfirm)
        fake_reconfirm.fail = set()
        self.wctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, main_rows=self.main_rows)
        self.ctx = dict(keys=lambda: dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key),
                        params=lambda: Params(), x_reuse=self.x_reuse, x_facts=lambda ys: dict(self.facts),
                        pilot_back=lambda man: (dict(self.vdata), []),
                        v_candidates=lambda ys: [(r, {}) for r in SMOKE_ROWS], main_rows=self.main_rows,
                        w_runner=self.w_runner, smoke_oracle=lambda pool: SmokeOracle(self))
        self._write_oracle()
        self._write_summary(elapsed_s)

    # ---- the context ------------------------------------------------------------------------------------------------
    def x_reuse(self):
        return list(self.why), self.w_doc, dict(self.cands), self.theta_w

    def main_rows(self, blk):
        if blk != {"digest_keys": "k" * 64}:
            raise ValueError("block digest's set differs")
        return [dict(r) for r in MAIN]

    def w_runner(self, pool, smoke=False):
        yw = YR.y_w_spec(self.ys, smoke)
        code = {"key": Y.w_measure_key}
        cache = (YS.YCache(self.ys.smoke_cache_dir, code, smoke_seeds=yw.smoke_seed_set()) if smoke
                 else YS.YCache(self.ys.cache_dir, code))
        wm = WM.WMeasurer(self.pool, cache, Params(), READOUT, "MBON05", "PAM08", "PPL105",
                          dict(strength=1.0, settle_ms=800.0, read_ms=600.0, window_ms=200),
                          dict(present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, seed_stride=1000))
        return WR.Runner(None, lambda smoke_=False: wm, self.wctx, yw), wm

    # ---- the phase-A record --------------------------------------------------------------------------------------------
    def _write_oracle(self):
        per = [dict(key=row_key(r), c=r["c"], axis=r["axis"], turn=r["turn"], value=True, failure=False,
                    testable=True, d_pre=0.1 if lenient(r["c"]) else 3.0, L_A=30.0, L_P=60.0, r=3.0, p=-3.0)
               for r in MAIN]
        p = Path(self.ys.oracle_detail)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(pairs=per, manifest=[])))
        self.n_len = sum(lenient(r["c"]) for r in MAIN)

    def _write_summary(self, elapsed_s: float):
        ys = self.ys
        idx = y_oc.rng(ys.precheck_seed, y_oc.TAG_CAL).integers(0, len(self.theta["resid"]), ys.cal_reps)
        cal = {m: y_oc.calibrate_y(self.theta, getattr(ys, f), m, idx, self.z, ys) for m, f in y_oc.MODES}
        cal = {m: dict(ok=c["ok"], failure=c["failure"], corners=c["corners"], a=c["a"], b=c["b"])
               for m, c in cal.items()}
        env = YR.env_hashes()
        pd = Path(ys.pilot_detail)
        pd.write_text(json.dumps(dict(manifest=[])))
        doc = dict(
            digest=dict(outcome="PASS", set={"digest_keys": "k" * 64}),
            oracle=dict(outcome="PASS", n=len(MAIN), k_ranges=[[4, 8], [6, 10]], detail_path=ys.oracle_detail,
                        detail_sha256=sha256_file(ys.oracle_detail),
                        derived=dict(counts=dict(all=dict(testable=len(MAIN), y_lenient=self.n_len)),
                                     yield_rule=dict(n_len=self.n_len))),
            stage0=dict(outcome="PASS"), reuse=dict(outcome="PASS"),
            pilot=dict(outcome="PASS", admitted=self.keys, theta=json.loads(canonical(w_oc.summary(self.theta))),
                       z_V=ZV, costs=dict(COSTS)),
            precheck=dict(outcome="PASS", calibration=y_store_json(cal), env=env),
            budget=dict(ledger=[dict(stage="oracle", wall_s=float(elapsed_s))]))
        Path(ys.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(ys.summary).write_text(json.dumps(doc))

    def runner(self):
        return YR.Runner(self.ctx, self.ys, pool=self.pool)


def y_store_json(o):
    return YS.to_json(o)


def doc():
    return json.loads(Path(Y.summary).read_text())


def through(w, last):
    out = None
    for s in YR.PHASE_B[:YR.PHASE_B.index(last) + 1]:
        out = getattr(w.runner(), f"stage_{s}")()
        assert out["outcome"] == "PASS", (s, out.get("reasons"), out.get("sentence"))
    return out
