"""A synthetic AA world (no engine): 249 main rows (167 b · 82 a; the (a) X odours drawn from 7 names so X-odour groups
exist), 31 lenient pairs (b: c 0, 15, …, 150; a: c 167, 171, …, 243), w_world's count Model and FakePool behind the
REAL WMeasurer and AACache, Y files in tmp written by Y's OWN path (results/y/oracle.json with lenient flags,
y_learning.json blocks digest … oc, results/y/pilot.json + results/y/cache from w_runner.Runner.units on Y's seed spec
and WMeasurer → YCache, exactly as Y's pilot_measure), a Z summary, git facts all-true, a passing tests log."""
import dataclasses
import json
from pathlib import Path

from flymon.brain import aa_runner as AR
from flymon.brain import aa_store
from flymon.brain import w_measure as WM
from flymon.brain import w_records
from flymon.brain import w_runner as WR
from flymon.brain import y_rules, y_store, z_split
from flymon.brain import y_runner as YR
from flymon.brain.aa_spec import SPEC as AA
from flymon.brain.config import Params
from flymon.brain.h3_store import sha256_file
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y
from tests.brain.r_fixtures import READOUT
from tests.brain.w_world import FakePool, Model

ZV = {"A": [16.916666666666668, 12.483878492769072], "P": [80.16666666666667, 29.775432639827233]}
A_ODOURS = ("Earthquake", "Surf", "Rock Slide", "Psychic", "Flamethrower", "Ice Beam", "Thunderbolt")
LENIENT_C = tuple(range(0, 151, 15)) + tuple(range(167, 244, 4))           # 11 (b) + 20 (a)
WINDOWS = dict(strength=1.0, settle_ms=800.0, read_ms=600.0, window_ms=200)
TIMING = dict(present_ms=400.0, gap_ms=200.0, train_settle_ms=800.0, seed_stride=1000)
CODE = {"key": Y.w_measure_key}


def main_rows():
    out = []
    for i in range(249):
        ax = "b" if i < 167 else "a"
        x = f"mx{i}" if ax == "b" else A_ODOURS[i % 7]
        out.append(dict(axis=ax, turn=306 + i, x=x, y=f"my{i}", odor_x={f"G{i}": 1.0}, odor_y={f"H{i}": 1.0},
                        move_x="M", opp_x=["T"], move_y="M", opp_y=["U"], c=i))
    return out


MAIN = main_rows()
PILOT = [dict(axis=k.split("|")[0], turn=int(k.split("|")[1]), x=k.split("|")[2], y=k.split("|")[3],
              odor_x={f"PG{j}": 1.0}, odor_y={f"PH{j}": 1.0}, move_x="M", opp_x=["T"], move_y="M", opp_y=["U"])
         for j, k in enumerate(Y.pilot_v_pairs)]


class World:
    def __init__(self, tmp_path, monkeypatch, s=None, n_lenient_pass=None):
        monkeypatch.chdir(tmp_path)
        self.tmp = tmp_path
        self.s = s or dataclasses.replace(AA, archive_root=str(tmp_path / "arch"), boot_b=200, boot_chunk=100,
                                          cov_reps=6, cov_b=40, synth_reps=4, workers=4)
        self.model = Model()
        self.pool = FakePool(self.model)
        for r in MAIN + PILOT:
            self.pool.pairs[json.dumps([r["odor_x"], r["odor_y"]], sort_keys=True)] = row_key(r)
        self.facts_ok = True
        self.z = {k: (float(v[0]), float(v[1])) for k, v in ZV.items()}
        self.wz = dict(self.z)                                  # W block reuse z_V (AA.2)
        self._write_y()
        self._write_z()
        Path(self.s.tests_log).parent.mkdir(parents=True, exist_ok=True)
        Path(self.s.tests_log).write_text("1234 passed in 600.00s\nexit 0\n")
        monkeypatch.setattr(AR, "summary_git", lambda p: dict(tracked=True, dirty=False))
        monkeypatch.setattr(AR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.ctx = dict(
            keys=lambda: dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key), params=lambda: Params(),
            main_rows=self.main_rows, pilot_rows=lambda: [dict(r) for r in PILOT], measurer=self.measurer,
            y_reader=lambda: WM.WMeasurer(None, aa_store.ReadCache(Y.cache_dir, CODE), Params(), READOUT, "MBON05",
                                          "PAM08", "PPL105", WINDOWS, TIMING),
            y_doc=lambda: json.loads(Path(Y.summary).read_text()), y_z=lambda: dict(self.z),
            y_theta=lambda where: dict(why=["Y θ̂ 없음(시험 세계)"]),
            y_pilot_detail=lambda: json.loads(Path(Y.pilot_detail).read_text()), y_pilot_back=self.pilot_back,
            z_doc=lambda: json.loads(Path(AA.z_summary).read_text()),
            git_facts=lambda path, commits: dict(last="x", ancestors={c: self.facts_ok for c in commits}),
            decl_sha=lambda files, commit: AR.files_sha(files),
            lenient=lambda: z_split.lenient_items(Y.oracle_detail, self.s.oracle_sha256, self.s.n_len),
            w_z=lambda: {k: list(v) for k, v in self.wz.items()})

    # ---- ctx pieces ---------------------------------------------------------------------------------------------
    def main_rows(self, blk):
        if blk.get("digest_keys") != Y.main_digest_keys:
            raise ValueError("block digest's set differs")
        return [dict(r) for r in MAIN]

    def measurer(self, pool, smoke=False):
        s = self.s
        cache = aa_store.AACache(s.smoke_cache_dir if smoke else s.cache_dir, CODE, smoke_seeds=AR.smoke_seed_set(s))
        return WM.WMeasurer(pool, cache, Params(), READOUT, "MBON05", "PAM08", "PPL105", WINDOWS, TIMING)

    def pilot_back(self, man):
        got, bad = aa_store_load(man)
        if bad:                                         # as y_runner.build_ctx.pilot_back
            return None, bad
        by = {}
        for g in got:
            pair, fly, brain = g["key"].rsplit("|", 2)
            by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
        return {k: w_records.pair_data(v, list(range(Y.pilot_flies))) for k, v in by.items()}, []

    # ---- Y and Z files in tmp (Y's own code paths) -------------------------------------------------------------------
    def _write_y(self):
        per = [dict(key=row_key(r), c=r["c"], axis=r["axis"], turn=r["turn"], value=True, failure=False, testable=True,
                    d_pre=0.1 if r["c"] in LENIENT_C else 3.0, L_A=30.0, L_P=60.0, r=3.0, p=-3.0) for r in MAIN]
        Path(Y.oracle_detail).parent.mkdir(parents=True, exist_ok=True)
        Path(Y.oracle_detail).write_text(json.dumps(dict(pairs=per, manifest=[])))
        self.s = dataclasses.replace(self.s, oracle_sha256=sha256_file(Y.oracle_detail))
        # Y's pilot through Y's own unit / measurer path (y_runner.build_ctx.pilot_measure, read at plan time)
        yw = dataclasses.replace(YR.y_w_spec(Y), workers=4)
        wm = WM.WMeasurer(self.pool, y_store.YCache(Y.cache_dir, CODE), Params(), READOUT, V.p_type, "PAM08", "PPL105",
                          WINDOWS, TIMING)
        r = WR.Runner(None, lambda smoke=False: wm, dict(params=Params()), yw)
        units = r.units(PILOT, "pilot", Y.pilot_probes, Y.pilot_flies, V.lever_edit)
        got = wm.learn(units, "pilot")
        pairs = {k: w_records.pair_data([g for g in got if g["unit"]["pair"] == k], list(range(Y.pilot_flies)))
                 for k in Y.pilot_v_pairs}
        rec = w_records.pilot_record(pairs, self.z, {}, W)
        Path(Y.pilot_detail).write_text(json.dumps(dict(
            admission=y_rules.admission({k: d["pre"] for k, d in pairs.items()}, self.z, Y), record_all=rec,
            manifest=r._manifest(got))))
        self.pool_jobs_after_y = self.pool.jobs
        keys = [row_key(r) for r in MAIN]
        doc = dict(digest=dict(outcome="PASS", set=dict(digest_keys=Y.main_digest_keys, n_b=167, n_a=82,
                                                        last_turn=Y.main_last_turn, keys=keys)),
                   oracle=dict(outcome="PASS", detail_sha256=self.s.oracle_sha256,
                               derived=dict(yield_rule=dict(n_len=31))),
                   stage0=dict(outcome="PASS"), reuse=dict(outcome="PASS"), pilot=dict(outcome="PASS", z_V=ZV),
                   precheck=dict(outcome="PASS"), oc=dict(outcome="STOP_OC_UNREACHABLE"))
        Path(Y.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(Y.summary).write_text(json.dumps(doc))

    def _write_z(self):
        Path(AA.z_summary).write_text(json.dumps({b: dict(outcome=o) for b, o in AA.z_outcomes}))

    def runner(self):
        return AR.Runner(self.ctx, self.s, pool=self.pool)

    def chain(self, last):
        out = None
        for st in self.s.stages[:self.s.stages.index(last) + 1]:
            out = self.runner().run(st)
            assert out["outcome"] == "PASS", (st, out.get("reasons"), out.get("sentence"))
        return out


def aa_store_load(man):
    from flymon.brain.r_store import load_manifest
    return load_manifest(man)


def doc():
    return json.loads(Path(AA.summary).read_text())
