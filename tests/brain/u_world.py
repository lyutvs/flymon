"""A scripted U world for the runner tests (no engine, no pool): a fake ctx (R's and T's summaries as the reuse source,
T's z rows for the endpoint gate, T's set as a declared-value dict, the reference set as 48 fake odours, 112 fake
calibration odours, R's even raw as fabricated entries, P's block as fixture rows), a ZScripted reference / rest
measurer whose counts depend on the edit's f (f = 1: block h4's z of this world, A 10 / 9, P 26 / 19, guard Δ 10;
f = 0: T's lever shape, MBON13 Δ 1 and zero share 0.5 — fails; f 0.1-0.4: MBON13 Δ 3 — fails; f 0.5-0.9: z_f A 6 / 3,
P 80 / 20, Δ 6 — passes), and a Scripted measurer built per z (`at(z)`) whose oracle writes real cache-like files under
results/u/cache keyed by (block, edit) plans, whose P arms come from s_fixtures and whose KC activity is scripted per
edit. Default even plans make f* = 0.6: f 0.5 testable_b 12, f 0.6 13, f 0.7 13 but (a) net drop 2 (filtered)."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

from flymon.brain import u_rules
from flymon.brain import u_runner as UR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.u_measure import is_u_edit, parse_u_edit, u_edit
from flymon.brain.u_spec import SPEC as U_SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli
from tests.brain.u_fixtures import ref_rows, rest_rows

SPEC = dataclasses.replace(U_SPEC, z_h4=(("A", Z["A"]), ("P", Z["P"])), sha_none="sha-C", sha_zero="sha-L0",
                           t_measure_key_t="t" * 64, t_z_detail_sha256="d" * 64)
CODE = {"key": SPEC.r_shared_key}
TCODE = {"key": "t" * 64}
UCODE = {"key": "u" * 64}
PIPE = {"key": "p" * 64}
N_REF = 96
ZF = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
REC_TYPES = list(dict.fromkeys(TYPES + list(SPEC.chain_types)))
EXTRA = {t: 1 for t in REC_TYPES if t not in ("MBON13", "MBON05")}
PASS_COUNTS = ([3, 9] * 48, [60, 100] * 48)                         # z A 6 / 3, P 80 / 20; MBON13 Δ 6
COUNTS = {1.0: ([1, 19] * 48, [7, 45] * 48),                          # block h4's z of this world; Δ 10
          0.0: ([0, 2] * 48, [60, 100] * 48),                         # MBON13 Δ 1, zero share 0.5 (T's lever shape)
          **{f: ([0, 6] * 48, [60, 100] * 48) for f in (0.1, 0.2, 0.3, 0.4)},          # MBON13 Δ 3: fails
          **{f: PASS_COUNTS for f in (0.5, 0.6, 0.7, 0.8, 0.9)}}
BLOCK_COUNTS = {(1.0, "out_block"): ([1, 19] * 48, [7, 45] * 48), (0.0, "out_block"): ([0, 4] * 48, [60, 100] * 48),
                (0.0, "chain_entry"): ([0, 2] * 48, [60, 100] * 48)}
_CLUSTER = {}


def sha_of(edit: str) -> str:
    if not is_u_edit(edit):
        return "sha-C"
    f, block = parse_u_edit(edit)
    if block is None and f == 1.0:
        return "sha-C"
    if block is None and f == 0.0:
        return "sha-L0"
    return f"sha-{edit}"


def cluster_oc(spec):
    if "v" not in _CLUSTER:
        _CLUSTER["v"] = REAL_OC_CLUSTER(spec)
    return _CLUSTER["v"]


REAL_OC_CLUSTER = u_rules.oc_cluster


def r_doc() -> dict:
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L0"}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def t_doc() -> dict:
    g = dict(passes=True, median_delta=10.0, zero_share=0.0)
    return dict(z=dict(outcome="STOP_Z_DEGENERATE", t_measure_key="t" * 64, code_key=SPEC.r_shared_key,
                       detail_sha256="d" * 64, lever=dict(z={"A": [0.5, 1.0], "P": [80.0, 20.0]}),
                       none=dict(z={"A": [10.0, 9.0], "P": [26.0, 19.0]}, csc_sha256=["sha-C"], edit_edges=[0],
                                 guard={"MBON13": g, "MBON05": g})))


def declared_set(**kw) -> dict:
    js = dict(status="OK", n_b=21, n_a=43, last_turn=103, last_turn_b=17, last_turn_a=103, n_opp=127, n_combos=1986,
              n_odours=55, skipped=dict(SPEC.skipped_declared), e1_clashes=[], cap_fails=[], all_off_pool=True,
              clusters_b=sorted(list(c) for c in SPEC.clusters_b), clusters_a=sorted(list(c) for c in SPEC.clusters_a),
              digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, b=[], a=[])
    return dict(js, **kw)


def t_rows() -> dict:
    """T's z rows of this world: "none" = f = 1's counts, T's lever = f = 0's (T's rows hold edges 0 / 2)."""
    out = {}
    for edit, f in ((SPEC.t_none_edit, 1.0), (SPEC.t_lever_edit, 0.0)):
        a, p = COUNTS[f]
        e, sha = (0, "sha-C") if f == 1.0 else (2, "sha-L0")
        out[edit] = dict(ref=[dict(r, types={"MBON13": r["types"]["MBON13"], "MBON05": r["types"]["MBON05"]})
                              for r in ref_rows(a, p, e, sha)], rest=rest_rows(N_REF, 0, e, sha))
    return out


class ZScripted:
    """u_measure.UZMeasurer's interface; counts[f] / block_counts[(f, block)] = (MBON13 counts, MBON05 counts);
    edges / blocks override the edge labels per edit."""

    def __init__(self, counts=None, block_counts=None, edges=None, blocks=None, n=None):
        self.counts = {**COUNTS, **(counts or {})}
        self.block_counts = {**BLOCK_COUNTS, **(block_counts or {})}
        self.edges, self.blocks, self.n = dict(edges or {}), dict(blocks or {}), n
        self.calls = []

    def _rows(self, edit):
        f, block = parse_u_edit(edit)
        a, p = self.counts[f] if block is None else self.block_counts[(f, block)]
        bl = self.blocks.get(edit, SPEC.contrast_declared().get(block, {}))
        return a, p, self.edges.get(edit, SPEC.lever_edges), bl

    def reference(self, edit, odors, strength, settle_ms, read_steps):
        self.calls.append(("reference", edit, len(odors)))
        a, p, e, bl = self._rows(edit)
        rows = ref_rows(a, p, e, sha_of(edit), extra=EXTRA, blocks=bl)
        return rows[:self.n] if self.n else rows

    def rest(self, edit, seeds, settle_ms, read_steps):
        self.calls.append(("rest", edit, len(seeds)))
        _, _, e, bl = self._rows(edit)
        return rest_rows(len(seeds), 0, e, sha_of(edit), extra=EXTRA, blocks=bl)


class Scripted:
    """RMeasurer's interface, built per z through at(z). plan[(block, edit)][pair key] = (r_ok, p_ok, bal) (default:
    punish passes only); override[(block, cond name)] = dict(edges=, edit=, sha=) changes one block's condition only;
    drop = {edit: P drop}; arm_edges = {edit: edges}; kc = {edit: {odour: frac}}; st["fail_once"] makes the next oracle
    call raise like an interrupted measurement."""

    def __init__(self, plan=None, override=None, drop=None, arm_edges=None, kc=None):
        self.plan, self.override = plan or {}, override or {}
        self.drop, self.arm_edges, self.kc = dict(drop or {}), dict(arm_edges or {}), kc or {}
        self.st, self.calls, self.last_wall_s, self.last_jobs = {"fail_once": False}, [], 2.0, 1
        self.z = dict(Z)
        self._rm = RMeasurer(None, None, SPEC, Params(), READOUT, self.z, TYPES, 100)

    def at(self, z):
        v = copy.copy(self)
        v.z = {k: tuple(x) for k, x in z.items()}
        v._rm = RMeasurer(None, None, SPEC, Params(), READOUT, v.z, TYPES, 100)
        return v

    def inputs(self, row, cond, block, seeds):
        return self._rm.inputs(row, cond, block, seeds)

    def _drop(self, edit):
        return self.drop.get(edit, 12 if is_u_edit(edit) else 16)

    def _edges(self, edit):
        return self.arm_edges.get(edit, SPEC.lever_edges if is_u_edit(edit) else 0)

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self._drop(i["edit"]), sha_of(i["edit"]), self._edges(i["edit"])) for i in items]

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, edit, len(odours)))
        fr = self.kc.get(edit, {})
        return {o: dict(frac=[fr.get(o, 0.05)] * len(seeds), max_win=[3] * len(seeds),
                        edit_edges=[self._edges(edit)], csc_sha256=[sha_of(edit)]) for o in odours}

    def oracle(self, rows, cond, block, seeds):
        self.calls.append(("oracle", block, cond.name, cond.edit, len(rows), tuple(self.z["A"])))
        if self.st["fail_once"]:
            self.st["fail_once"] = False
            raise RuntimeError("worker died")
        o = self.override.get((block, cond.name), {})
        out = []
        for r in rows:
            k = key_of(r)
            flags = self.plan.get((block, cond.edit), {}).get(k, (False, True, False))
            res = fake_oracle(*flags, sha=o.get("sha", sha_of(cond.edit)),
                              edges=o.get("edges", self._edges(cond.edit)), edit=o.get("edit", cond.edit),
                              n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ck = hashlib.sha256(f"{block}|{cond.name}|{cond.edit}|{k}|{self.z}".encode()).hexdigest()
            f = Path(SPEC.cache_dir) / "r_oracle" / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(UR.u_rules, "oc_cluster", cluster_oc)
        self.judge_commits, self.judgement_calls, self.set_calls, self.odour_calls = [], 0, 0, 0
        self.r, self.r_git, self.js = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"]), declared_set()
        self.t, self.t_git = t_doc(), dict(tracked=True, dirty=False, judge_commits=[])
        self.t_rows = t_rows()
        monkeypatch.setattr(UR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(UR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=104)
        eb, ea = self.keys(self.even, "b"), self.keys(self.even, "a")
        self.r_even_plan = {"C": {k: (True, True, False) for k in eb[:7]},
                            "L": {k: (True, True, False) for k in eb[:16]}}
        self.even_plan = {("even", u_edit(0.5)): {k: (True, True, False) for k in eb[:12]},
                          ("even", u_edit(0.6)): {k: (True, True, False) for k in eb[:13]},
                          ("even", u_edit(0.7)): {**{k: (True, True, False) for k in eb[:13]},
                                                  **{k: (False, False, False) for k in ea[:2]}},
                          ("path", u_edit(0.0)): dict(self.r_even_plan["L"]),
                          ("path", u_edit(1.0)): dict(self.r_even_plan["C"])}
        p_ref = {(r["direction"], r["arm"], r["seed"]): {k: v for k, v in r.items()
                                                         if k not in ("direction", "x", "y", "point", "r")}
                 for r in p_rows(P_SPEC, "none", 16)}
        self.archive = tmp_path / "archive"
        odors = [dict(name=f"R{j:02d}", seeds=[1000 + 2 * j, 1001 + 2 * j], strengths={}) for j in range(N_REF // 2)]
        calib = {f"C{i}|T": {"G": 1.0} for i in range(112)}
        ids = list(calib)
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, rec_types=REC_TYPES,
                        even_rows=self.even, ref_odors=odors, probe_seeds=[s for o in odors for s in o["seeds"]],
                        ref_strength=0.35, ref_window=(800.0, 600), calib=(calib, ids[:56], ids[56:]), cap_ok=True,
                        enc_per={o: 0.05 for o in ids}, t_set=self._t_set, set_odours=self._set_odours,
                        judgement_rows=self._judgement_rows, clusters=self._clusters, r_even=self._r_even,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git),
                        t_doc=lambda: json.loads(json.dumps(self.t)), t_git=lambda: dict(self.t_git),
                        t_rows=lambda: json.loads(json.dumps(self.t_rows)))

    def _t_set(self):
        self.set_calls += 1
        return dict(self.js)

    def _set_odours(self):
        self.odour_calls += 1
        return {f"M{i}|T": {"G": 1.0} for i in range(55)}

    def _judgement_rows(self):
        self.judgement_calls += 1
        return self.judge

    def _clusters(self, rows):
        return {key_of(r): ("GROUND* 대 NORMAL" if r["axis"] == "b" else "FIRE*") for r in rows}

    def _r_even(self, rows, name):
        lever = name == "L"
        return got_for(rows, self.r_even_plan[name], sha="sha-L0" if lever else "sha-C", edges=2 if lever else 0,
                       edit=SPEC.t_lever_edit if lever else SPEC.no_edit)

    def runner(self, m, zm=None, code=None, tcode=None, ucode=None, pipeline=None):
        return UR.Runner(m.at, zm or ZScripted(), self.ctx, SPEC, code=code or CODE, tcode=tcode or TCODE,
                         ucode=ucode or UCODE, pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def scripted(self, **kw):
        m = Scripted(**kw)
        m.plan = {**self.even_plan, **m.plan}
        return m

    def pass_plan(self, f=0.6, n=14, c=8, f_a=3):
        """judgement: L_f n / C c (b) testable and L_f f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", u_edit(f)): {**{k: (True, True, False) for k in jb[:n]},
                                       **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "none"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through(w, m, last="gate2", r=None, zm=None):
    r = r or w.runner(m, zm)
    for s in UR.ORDER[:UR.ORDER.index(last) + 1]:
        out = getattr(r, f"stage_{s}")()
        if s in UR.GATES:
            assert out["outcome"] == "PASS", (s, out)
        if s == "smoke":
            assert out["problems"] == [], out["problems"]
    return r
