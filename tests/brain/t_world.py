"""A scripted T world for the runner tests (no engine, no pool): a fake ctx (R's summary as the reuse source, T's set
as a declared-value dict, the reference set as 48 fake odours, R's even raw as fabricated entries, P's block as
fixture rows), a ZScripted reference / rest measurer whose unedited counts give exactly block h4's z of this world
(A 10 / 9, P 26 / 19 — the world's spec carries that z_h4) and whose lever counts give z_lever A 6 / 3, P 80 / 20, and
a Scripted measurer that is built per z (`at(z)`, as the runner's measure(z) is), whose P arms come from s_fixtures
(real p_rules.p_judge reads them) and whose oracle writes real cache-like files under results/t/cache (the seal and the
judge re-read them; the stored inputs carry the view's z). Pair statistics of r_fixtures' fake oracle do not depend on
z (each phase moves one readout type only), so the same flags read the same under both z."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

from flymon.brain import t_rules
from flymon.brain import t_runner as TR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.t_spec import SPEC as T_SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli
from tests.brain.t_fixtures import ref_rows, rest_rows

SPEC = dataclasses.replace(T_SPEC, z_h4=(("A", Z["A"]), ("P", Z["P"])))
SHA = {"L": "sha-L", "C": "sha-C", "E0": "sha-C"}
CODE = {"key": SPEC.r_shared_key}
TCODE = {"key": "t" * 64}
PIPE = {"key": "p" * 64}
N_REF = 96
ZL = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
COUNTS = {SPEC.no_edit: ([1, 19] * 48, [7, 45] * 48),          # mean 10 / SD 9, mean 26 / SD 19: block h4's z here
          SPEC.lever_edit: ([3, 9] * 48, [60, 100] * 48)}      # z_lever A 6 / 3, P 80 / 20; guard Δ 6 / 80
_CLUSTER = {}


def cluster_oc(spec):
    """t_rules.oc_cluster once per test session (10⁵ draws take about a second)."""
    if "v" not in _CLUSTER:
        _CLUSTER["v"] = REAL_OC_CLUSTER(spec)
    return _CLUSTER["v"]


REAL_OC_CLUSTER = t_rules.oc_cluster


def r_doc() -> dict:
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L"}}),
                gate1=dict(outcome="PASS", code_key=k, record={"median": 0.0457}, written_at="r-g1"),
                gate2=dict(outcome="PASS", code_key=k, label="LEARNS_CONFIRMATORY",
                           p_judgement={"directions": {"r1": {"ell": 1.087}, "r2": {"ell": 1.792}}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def declared_set(**kw) -> dict:
    js = dict(status="OK", n_b=21, n_a=43, last_turn=103, last_turn_b=17, last_turn_a=103, n_opp=127, n_combos=1986,
              n_odours=55, skipped=dict(SPEC.skipped_declared), e1_clashes=[], cap_fails=[], all_off_pool=True,
              clusters_b=sorted(list(c) for c in SPEC.clusters_b), clusters_a=sorted(list(c) for c in SPEC.clusters_a),
              digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, b=[], a=[])
    return dict(js, **kw)


class ZScripted:
    """t_measure.ZMeasurer's interface; counts[edit] = (MBON13 counts, MBON05 counts) over the 96 presentations."""

    def __init__(self, counts=None, edges=None):
        self.counts = dict(COUNTS, **(counts or {}))
        self.edges = dict({SPEC.no_edit: 0, SPEC.lever_edit: SPEC.lever_edges}, **(edges or {}))
        self.calls = []

    def _sha(self, edit):
        return "sha-C" if edit == SPEC.no_edit else "sha-L"

    def reference(self, edit, odors, strength, settle_ms, read_steps):
        self.calls.append(("reference", edit, len(odors)))
        a, p = self.counts[edit]
        return ref_rows(a, p, self.edges[edit], self._sha(edit))

    def rest(self, edit, seeds, settle_ms, read_steps):
        self.calls.append(("rest", edit, len(seeds)))
        return rest_rows(len(seeds), 0, self.edges[edit], self._sha(edit))


class Scripted:
    """RMeasurer's interface, built per z through at(z). plan[(block, cond)][pair key] = (r_ok, p_ok, bal) (default:
    punish passes only); override[(block, cond)] = dict(edges=, edit=, sha=) changes one block's condition only;
    drop = {edit: P drop} sets ℓ per condition; arm_edges = {edit: edges}; kc = {edit: {odour: frac}} sets KC
    activity; st["fail_once"] makes the next oracle call raise like an interrupted measurement."""

    def __init__(self, plan=None, override=None, drop=None, arm_edges=None, kc=None):
        self.plan, self.override = plan or {}, override or {}
        self.drop = dict({SPEC.lever_edit: 12, SPEC.no_edit: 16}, **(drop or {}))
        self.arm_edges = dict({SPEC.lever_edit: SPEC.lever_edges, SPEC.no_edit: 0}, **(arm_edges or {}))
        self.kc = kc or {}
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

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self.drop[i["edit"]], "sha-L" if i["edit"] == SPEC.lever_edit else "sha-C",
                        self.arm_edges[i["edit"]]) for i in items]

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, edit, len(odours)))
        lever = edit == SPEC.lever_edit
        fr = self.kc.get(edit, {})
        return {o: dict(frac=[fr.get(o, 0.05)] * len(seeds), max_win=[3] * len(seeds),
                        edit_edges=[self.arm_edges[edit]], csc_sha256=["sha-L" if lever else "sha-C"]) for o in odours}

    def oracle(self, rows, cond, block, seeds):
        self.calls.append(("oracle", block, cond.name, len(rows), tuple(self.z["A"])))
        if self.st["fail_once"]:
            self.st["fail_once"] = False
            raise RuntimeError("worker died")
        o = self.override.get((block, cond.name), {})
        out = []
        for r in rows:
            k = key_of(r)
            flags = self.plan.get((block, cond.name), {}).get(k, (False, True, False))
            res = fake_oracle(*flags, sha=o.get("sha", SHA[cond.name]),
                              edges=o.get("edges", SPEC.lever_edges if cond.name == "L" else 0),
                              edit=o.get("edit", cond.edit), n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ck = hashlib.sha256(f"{block}|{cond.name}|{k}|{self.z}".encode()).hexdigest()
            f = Path(SPEC.cache_dir) / "r_oracle" / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(TR.t_rules, "oc_cluster", cluster_oc)
        self.judge_commits, self.judgement_calls, self.set_calls, self.odour_calls = [], 0, 0, 0
        self.r, self.r_git, self.js = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"]), declared_set()
        self.s = {}
        monkeypatch.setattr(TR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(TR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=104)
        eb = self.keys(self.even, "b")
        self.r_even_plan = {"C": {k: (True, True, False) for k in eb[:7]},
                            "L": {k: (True, True, False) for k in eb[:16]}}
        self.even_plan = {("even", "L"): {k: (True, True, False) for k in eb[:12]}}
        p_ref = {(r["direction"], r["arm"], r["seed"]): {k: v for k, v in r.items()
                                                         if k not in ("direction", "x", "y", "point", "r")}
                 for r in p_rows(P_SPEC, "none", 16)}
        self.archive = tmp_path / "archive"
        odors = [dict(name=f"R{j:02d}", seeds=[1000 + 2 * j, 1001 + 2 * j], strengths={}) for j in range(N_REF // 2)]
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, even_rows=self.even,
                        ref_odors=odors, probe_seeds=[s for o in odors for s in o["seeds"]], ref_strength=0.35,
                        ref_window=(800.0, 600), t_set=self._t_set, set_odours=self._set_odours,
                        judgement_rows=self._judgement_rows, clusters=self._clusters, r_even=self._r_even,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git),
                        s_doc=lambda: json.loads(json.dumps(self.s)))

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
        return got_for(rows, self.r_even_plan[name], sha="sha-L" if lever else "sha-C",
                       edges=SPEC.lever_edges if lever else 0, edit=SPEC.lever_edit if lever else SPEC.no_edit)

    def runner(self, m, zm=None, code=None, tcode=None, pipeline=None):
        return TR.Runner(m.at, zm or ZScripted(), self.ctx, SPEC, code=code or CODE, tcode=tcode or TCODE,
                         pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def scripted(self, **kw):
        m = Scripted(**kw)
        m.plan = {**self.even_plan, **m.plan}
        return m

    def pass_plan(self, n=14, c=8, f_a=3):
        """judgement: L n / C c (b) testable and L f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", "L"): {**{k: (True, True, False) for k in jb[:n]},
                                 **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "C"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through_gate3(w, m, r=None, zm=None):
    r = r or w.runner(m, zm)
    assert r.stage_reuse()["outcome"] == "PASS"
    assert r.stage_set()["outcome"] == "PASS"
    assert r.stage_z()["outcome"] == "PASS"
    assert r.stage_gate1s()["outcome"] == "PASS"
    assert r.stage_smoke()["problems"] == []
    r.stage_oc()
    r.stage_gate2_oc()
    assert r.stage_gate2()["outcome"] == "PASS"
    assert r.stage_gate3()["outcome"] == "PASS"
    return r
