"""A synthetic AB world (no engine, no connectome) in tmp_path:
- W's main set (48 rows with c; AA's eight learned pairs among them), even rows, V's set (4 rows), W / Y / V / AA
  summaries, results/v/kc_input.json (its sha256 is the world spec's kc_input_sha256);
- AA's learn raw: aa_runner.units on AA's main-set seeds → the REAL WMeasurer over aa_store.AACache with w_world's
  count Model / FakePool into results/aa/cache, results/aa/learn.json (manifest with real sha256), records.json /
  estimate.json / aa_learning `records` computed by aa_estimate from the same raw (AA's own path; min_groups 3 so the
  six-pair S1 has a two-stage primary), so the differential test and the 0f source (fut_check) reproduce;
- AA S1 = 6 pairs over 3 X labels (sizes 3, 2, 1), plus 2 screened-out learned pairs (192 learn units, 144 in S1);
- lv_sources is the REAL ab_pairs function over the world's rows; base / selftest / generate are scripted, the
  generated KC-pre set has 40 rows ((a) 30 over 7 X labels and 9 type sets, (b) 10), and every AB.3 declared value of
  the world spec equals the scripted generation (decl_fields);
- git facts all true, summary_git / git_state clean, HEAD untracked, a passing tests log."""
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path

from flymon.brain import aa_estimate, aa_runner, aa_store, ab_pairs, w_records
from flymon.brain import ab_runner as R
from flymon.brain import ab_store
from flymon.brain import w_measure as WM
from flymon.brain.aa_spec import SPEC as AA
from flymon.brain.ab_spec import SPEC, small
from flymon.brain.config import Params
from flymon.brain.r_pairs import row_key
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y
from tests.brain.r_fixtures import READOUT
from tests.brain.w_world import FakePool, Model

ZV = {"A": [16.916666666666668, 12.483878492769072], "P": [80.16666666666667, 29.775432639827233]}
CODE = {"key": Y.w_measure_key}
S1_C = (3, 7, 11, 20, 25, 40)                                  # AA S1, c order
S1_X = ("Earthquake", "Earthquake", "Surf", "Earthquake", "Surf", "Psychic")       # X-label groups 3 · 2 · 1
OUT_C = (5, 30)                                                # learned, screened out
A_X = ("Earthquake", "Surf", "Psychic", "Flamethrower", "Rock Slide", "Mega Drain", "Thunderbolt")
TSETS = (("DRAGON", "WATER"), ("FIRE", "ROCK"), ("FLYING", "GRASS"), ("FLYING", "GROUND"), ("FLYING", "PSYCHIC"),
         ("GROUND", "ICE"), ("GROUND", "WATER"), ("NORMAL", "PSYCHIC"), ("ROCK",))


def _row(axis, turn, x, y, mx, ox, my, oy, tag):
    return dict(axis=axis, turn=int(turn), x=x, y=y, odor_x={f"{tag}G{turn}": 1.0}, odor_y={f"{tag}H{turn}": 1.0},
                move_x=mx, opp_x=list(ox), move_y=my, opp_y=list(oy))


def main_rows():
    out = []
    for c in range(48):
        if c in S1_C:
            x = S1_X[S1_C.index(c)]
        else:
            x = f"wx{c}"
        mv = "GROUND" if c % 2 else "WATER"
        out.append(dict(_row("b" if c < 30 else "a", 306 + c, x, f"wy{c}", mv, ("FIRE",), "ICE", ("NORMAL",), "m"),
                        c=c))
    return out


MAIN = main_rows()
EVEN = [_row("b", 10 + j, f"ex{j}", f"ey{j}", "FIRE", ("GRASS",), "ROCK", ("WATER",), "e") for j in range(5)]
VROWS = [_row("b", 17 + 2 * j, f"vx{j}", f"vy{j}", "ELECTRIC", ("FLYING",), "ICE", ("DRAGON",), "v")
         for j in range(4)]
AA_ROWS = sorted([r for r in MAIN if r["c"] in S1_C + OUT_C], key=lambda r: r["c"])


def gen_doc():
    """The scripted 0c output: a 40-row KC-pre set and every value AB.3 declares (in ab_pairs.generate's form)."""
    rows = []
    for i in range(30):
        rows.append(_row("a", 3 * i, f"{A_X[i % 7]}", f"ay{i}", f"M{i % 7}", TSETS[i % 9], "ELECTRIC", ("ICE",), "g"))
    for i in range(10):
        rows.append(_row("b", 3 * i + 1, f"bx{i}", f"by{i}", f"N{i % 3}", TSETS[i % 4], "ELECTRIC", ("ROCK",), "g"))
    rows = ab_pairs.declared_order(rows)
    ids = sorted({f"{m}|{'+'.join(o)}" for r in rows for m, o in ((r["move_x"], r["opp_x"]), (r["move_y"], r["opp_y"]))})
    known = ids[:10]
    a_rows = [r for r in rows if r["axis"] == "a"]
    b_rows = [r for r in rows if r["axis"] == "b"]
    by = lambda rs: {"a": sum(r["axis"] == "a" for r in rs), "b": sum(r["axis"] == "b" for r in rs)}   # noqa: E731
    return dict(
        rows=rows, n_opp=11, n_opp_type_out=4, opponents=[f"O{i}" for i in range(11)], n_new_type_species=3,
        new_type_sets=["FIRE+ROCK", "ROCK"], n_combos=132,
        skipped_inorder=dict(cap=5, collision=0, e1=0, used=40, glom_dup=0, in_set=12, pool_only=3, lv_odour=7,
                             kc_input=0),
        skipped_declared=dict(used=40, in_set=15, pool_only=3, cap=5, collision=0, e1=0, glom_dup=0),
        lv={"a": 2, "b": 3}, lv_kc_out={"a": 1, "b": 1}, pre_kc=by(rows),
        turns=[min(r["turn"] for r in rows), max(r["turn"] for r in rows)],
        last_turn={ax: max(r["turn"] for r in rows if r["axis"] == ax) for ax in ("a", "b")},
        odours=dict(all=len(ids), v_cache=len(known), new=len(ids) - len(known)), odour_ids=ids,
        new_odours=ids[10:], v_cache_odours=known, v_cache_out=known[4:6], vcache_drop={"a": 1, "b": 2},
        post_kc_max={"a": 29, "b": 8}, needs_kc={"a": 29, "b": 6}, vcache_pass={"a": 0, "b": 2},
        a_x_groups=dict(Counter(r["x"] for r in a_rows)), a_t_groups=dict(Counter(map(ab_pairs.ts_key, a_rows))),
        b_x_groups=len({r["x"] for r in b_rows}), b_t_groups=len({ab_pairs.ts_key(r) for r in b_rows}),
        gen1_same_keys=2, digest_keys=ab_pairs.digest_keys(rows), keys=[row_key(r) for r in rows])


def decl_fields(g: dict) -> dict:
    """The world spec's AB.3 declared values = the scripted generation."""
    return dict(n_opp=g["n_opp"], n_opp_type_out=g["n_opp_type_out"], n_new_type_species=g["n_new_type_species"],
                new_type_sets=tuple(g["new_type_sets"]), n_combos=g["n_combos"],
                decl_skips=tuple(g["skipped_declared"].items()), decl_lv=tuple(g["lv"].items()),
                decl_lv_kc_out=tuple(g["lv_kc_out"].items()), decl_pre_kc=tuple(g["pre_kc"].items()),
                decl_turns=tuple(g["turns"]), decl_last_turn=tuple(g["last_turn"].items()),
                decl_odours=tuple(g["odours"].items()), decl_vcache_drop=tuple(g["vcache_drop"].items()),
                decl_vcache_drop_odours=len(g["v_cache_out"]), decl_post_kc_max=tuple(g["post_kc_max"].items()),
                decl_needs_kc=tuple(g["needs_kc"].items()), decl_vcache_pass=tuple(g["vcache_pass"].items()),
                decl_a_x_groups=tuple(g["a_x_groups"].items()), decl_a_t_groups=tuple(g["a_t_groups"].items()),
                decl_b_x_groups=g["b_x_groups"], decl_b_t_groups=g["b_t_groups"],
                decl_gen1_same_keys=g["gen1_same_keys"], kc_repro=tuple(g["v_cache_odours"][:4]))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj))


class World:
    def __init__(self, tmp_path, monkeypatch, s=None):
        monkeypatch.chdir(tmp_path)
        self.tmp = tmp_path
        self.gen = gen_doc()
        base = small(SPEC, archive_root=str(tmp_path / "arch"), boot_b=200, boot_chunk=100, n_sel=30, n_ver=40,
                     truth_pairs=2_000, truth_chunk=1_000, cal_chunk=7, bench_reps=2, synth_reps=2, workers=2,
                     fut_src_pairs=len(S1_C), fut_src_units=len(S1_C) * 8 * 3, fut_reps=6, fut_grid_reps=2,
                     fut_grid_g=(5, 6), fut_grid_k=(8, 12), aa_trained=len(AA_ROWS), w_pilot_pairs=3,
                     y_pilot_pairs=3, y_pilot_split=(("even", 2), ("v_set", 1)), v_set_rows=len(VROWS),
                     **decl_fields(self.gen))
        self.s = s or base
        self.model = Model()
        self.pool = FakePool(self.model)
        for r in MAIN:
            self.pool.pairs[json.dumps([r["odor_x"], r["odor_y"]], sort_keys=True)] = row_key(r)
        self.z = {k: (float(v[0]), float(v[1])) for k, v in ZV.items()}
        self.wz = dict(self.z)
        self.facts_ok = lambda c: True
        self.w_digest = dict(digest_keys=self.s.w_digest_keys, n_b=self.s.w_n_b, n_a=self.s.w_n_a,
                             last_turn=self.s.w_last_turn)
        self.base_error = None
        self.gen_calls = []
        self._write_summaries()
        self._write_aa()
        lv = ab_pairs.lv_sources(*self._lv_args(), dataclasses.replace(self.s, lv_odour_n=0))
        self.s = dataclasses.replace(self.s, lv_odour_n=lv["n"])
        kc = Path(V.kc_input_detail)
        write_json(kc, dict(world="kc_input"))
        self.s = dataclasses.replace(self.s, kc_input_sha256=hashlib.sha256(kc.read_bytes()).hexdigest())
        self.write_log()
        monkeypatch.setattr(R, "summary_git", lambda p: dict(tracked=True, dirty=False))
        monkeypatch.setattr(R, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(R, "_head_summary", lambda path: None)
        detail = dict(learn=SPEC.aa_learn_detail, records=SPEC.aa_records_detail, estimate=SPEC.aa_estimate_detail)
        self.ctx = dict(
            keys=lambda: dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key), params=lambda: Params(),
            docs=self.docs, base=self.base, lv=lambda: ab_pairs.lv_sources(*self._lv_args(), self.s),
            generate=self.generate, selftest=self.selftest, even_rows=lambda: [dict(r) for r in EVEN],
            aa_reader=lambda: WM.WMeasurer(None, ab_store.ReadCache(SPEC.aa_cache_dir, CODE), Params(), READOUT,
                                           "MBON05", "PAM08", "PPL105", R.WINDOWS, R.TIMING),
            aa_rows=lambda: [dict(r) for r in MAIN], w_z=lambda: {k: tuple(v) for k, v in self.wz.items()},
            git_facts=lambda path, commits: dict(last="x", ancestors={c: self.facts_ok(c) for c in commits}),
            decl_sha=lambda files, commit: R.files_sha(files),
            aa_detail=lambda name: json.loads(Path(detail[name]).read_text()),
            fut_source=lambda: R.read_fut_source(self.s, self.docs()["aa_learning"], self.docs()["y_learning"]))

    # ---- ctx pieces ---------------------------------------------------------------------------------------------
    def docs(self):
        return {n: json.loads(Path(p).read_text()) for n, p in (("v_lever", W.v_summary), ("w_learning", W.summary),
                                                                ("y_learning", Y.summary),
                                                                ("aa_learning", SPEC.aa_summary))}

    def _lv_args(self):
        d = self.docs()
        return d["aa_learning"], d["w_learning"], d["y_learning"], [dict(r) for r in MAIN], EVEN, VROWS

    def base(self):
        if self.base_error:
            raise ValueError(self.base_error)
        return None, [dict(r) for r in VROWS], dict(self.w_digest, rows=[dict(r) for r in MAIN])

    def selftest(self):
        s = self.s
        return dict(ok=True, n_b=s.w_n_b, n_a=s.w_n_a, last_turn=s.w_last_turn, digest_keys=s.w_digest_keys,
                    skipped={})

    def generate(self, lv):
        self.gen_calls.append(list(lv))
        return json.loads(json.dumps(self.gen))

    # ---- the summaries and AA's raw in tmp ---------------------------------------------------------------------------
    def _write_summaries(self):
        write_json(W.v_summary, dict(set=dict(set={}), kc_input=dict(outcome="PASS")))
        man = [dict(key=f"{row_key(r)}|{f}|R") for r in EVEN[:3] for f in range(2)]
        write_json(W.summary, dict(reuse=dict(outcome="PASS", z_V=ZV), pilot=dict(outcome="PASS", manifest=man)))
        adm = {row_key(r): dict(d=0.1) for r in EVEN[:2] + VROWS[:1]}
        write_json(Y.summary, dict(digest=dict(outcome="PASS", set=dict(digest_keys=Y.main_digest_keys)),
                                   pilot=dict(outcome="PASS", z_V=ZV, admission=adm)))

    def _write_aa(self):
        s = self.s
        U = aa_runner.units(AA_ROWS, "main", s.probes, s.flies, V.lever_edit, ws=aa_runner.aa_w_spec())
        wm = WM.WMeasurer(self.pool, aa_store.AACache(SPEC.aa_cache_dir, CODE), Params(), READOUT, "MBON05", "PAM08",
                          "PPL105", R.WINDOWS, R.TIMING)
        got = wm.learn(U, "learn")
        self.pool_jobs_after_aa = self.pool.jobs
        write_json(SPEC.aa_learn_detail, dict(manifest=aa_runner.manifest(got)))
        flies = list(range(s.flies))
        data = {k: w_records.pair_data(v, flies) for k, v in aa_runner.by_pair(got).items()}
        aas = dataclasses.replace(AA, boot_b=s.boot_b, boot_chunk=s.boot_chunk, min_groups=3)
        keys = [row_key(r) for r in AA_ROWS if r["c"] in S1_C]
        arrs = {k: aa_estimate.pair_arrays(data[k], self.z, aas) for k in data}
        pairs = aa_store.to_json({k: aa_estimate.pair_record(data[k], self.z, aas) for k in data})
        est = aa_estimate.estimate_set(arrs, keys, "S1", aas)
        assert est["plan"]["primary"] == "two_stage"
        s1 = aa_store.to_json({k: v for k, v in est.items() if not k.startswith("_")})
        write_json(SPEC.aa_records_detail, dict(block=dict(pairs=pairs, s1_records=s1)))
        prim = [dict(key=k, c=int(r["c"]), **{g: dict(mean=float(arrs[k]["w"][:, aa_estimate.GI[g]].mean()))
                                             for g in aa_estimate.PRIMARY})
                for r in AA_ROWS if (k := row_key(r)) in keys]
        pooled = {g: dict(point=est["point"][g], ci=est["ci"]["two_stage"][g], method="two_stage")
                  for g in aa_estimate.PRIMARY}
        write_json(SPEC.aa_estimate_detail, dict(primary=dict(pairs=prim, pooled=pooled)))
        doc = {b: dict(outcome="PASS") for b, _c in SPEC.aa_blocks}
        doc["screen"]["pairs"] = [dict(key=row_key(r), c=int(r["c"]), passed=r["c"] in S1_C) for r in AA_ROWS]
        doc["records"].update(s1_records=s1, pairs=pairs)
        doc["candidates"] = [dict(key=row_key(r), c=int(r["c"]), state="trained" if r in AA_ROWS else "untouched")
                             for r in MAIN]
        write_json(SPEC.aa_summary, doc)

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def write_log(self, text="1234 passed in 600.00s\nexit 0\n"):
        Path(self.s.tests_log).parent.mkdir(parents=True, exist_ok=True)
        Path(self.s.tests_log).write_text(text)

    def runner(self):
        return R.Runner(self.ctx, self.s)

    def chain(self, last):
        out = None
        for st in self.s.stages[:self.s.stages.index(last) + 1]:
            out = self.runner().run(st)
            assert out["outcome"] == "PASS", (st, out.get("reasons"), out.get("sentence"))
        return out


def doc():
    return json.loads(Path(SPEC.summary).read_text())
