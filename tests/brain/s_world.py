"""A scripted S world for the runner tests (no engine, no pool): a fake ctx (R's summary as the reuse source, S's set
as a declared-value dict, P's block as fixture rows), a Scripted measurer whose P arms come from s_fixtures (real
p_rules.p_judge reads them: ℓ ≈ (drop + 1) / 9) and whose oracle writes real cache-like files under results/s/cache
(the seal and the judge re-read them), and helpers to run the chain."""
import hashlib
import json
from pathlib import Path

from flymon.brain import s_runner as SR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.s_spec import SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli

SHA = {"L": "sha-L", "C": "sha-C", "E0": "sha-C"}
CODE = {"key": SPEC.r_shared_key}
PIPE = {"key": "p" * 64}


def r_doc() -> dict:
    """R's summary as S reads it (S.3 ①): the three reused blocks plus what S records from R."""
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L"}}),
                gate1=dict(outcome="PASS", code_key=k, record={"median": 0.0457}, written_at="r-g1"),
                gate2=dict(outcome="PASS", code_key=k, label="LEARNS_CONFIRMATORY",
                           p_judgement={"directions": {"r1": {"ell": 1.087}, "r2": {"ell": 1.792}}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def declared_set(**kw) -> dict:
    js = dict(status="OK", n_b=SPEC.n_b, n_a=SPEC.n_a, last_turn=SPEC.last_turn, skipped={"used": 1},
              digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, b=[], a=[])
    return dict(js, **kw)


class Scripted:
    """RMeasurer's interface. plan[(block, cond)][pair key] = (r_ok, p_ok, bal) (default: punish passes only);
    override[(block, cond)] = dict(edges=, edit=, sha=) changes one block's condition only; drop = {edit: P drop}
    sets ℓ per condition; arm_edges = {edit: edges}; fail_once makes the next oracle call raise like an
    interrupted measurement."""

    def __init__(self, plan=None, override=None, drop=None, arm_edges=None):
        self.plan, self.override = plan or {}, override or {}
        self.drop = dict({SPEC.lever_edit: 12, SPEC.no_edit: 16}, **(drop or {}))
        self.arm_edges = dict({SPEC.lever_edit: SPEC.lever_edges, SPEC.no_edit: 0}, **(arm_edges or {}))
        self.fail_once, self.calls, self.last_wall_s, self.last_jobs = False, [], 2.0, 1
        self._rm = RMeasurer(None, None, SPEC, Params(), READOUT, Z, TYPES, 100)

    def inputs(self, row, cond, block, seeds):
        return self._rm.inputs(row, cond, block, seeds)

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self.drop[i["edit"]], "sha-L" if i["edit"] == SPEC.lever_edit else "sha-C",
                        self.arm_edges[i["edit"]]) for i in items]

    def oracle(self, rows, cond, block, seeds):
        self.calls.append(("oracle", block, cond.name, len(rows)))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("worker died")
        o = self.override.get((block, cond.name), {})
        out = []
        for r in rows:
            k = key_of(r)
            flags = self.plan.get((block, cond.name), {}).get(k, (False, True, False))
            res = fake_oracle(*flags, sha=o.get("sha", SHA[cond.name]),
                              edges=o.get("edges", SPEC.lever_edges if cond.name == "L" else 0),
                              edit=o.get("edit", cond.edit), n_rep=len(seeds["report"]), n_act=len(seeds["act"]))
            ck = hashlib.sha256(f"{block}|{cond.name}|{k}".encode()).hexdigest()
            f = Path(SPEC.cache_dir) / "r_oracle" / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.judge_commits, self.judgement_calls, self.set_calls = [], 0, 0
        self.r, self.r_git, self.js = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"]), declared_set()
        monkeypatch.setattr(SR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(SR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=104)
        p_ref = {(r["direction"], r["arm"], r["seed"]): {k: v for k, v in r.items()
                                                         if k not in ("direction", "x", "y", "point", "r")}
                 for r in p_rows(P_SPEC, "none", 16)}
        self.archive = tmp_path / "archive"
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, even_rows=self.even,
                        s_set=self._s_set, judgement_rows=self._judgement_rows,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git))

    def _s_set(self):
        self.set_calls += 1
        return dict(self.js)

    def _judgement_rows(self):
        self.judgement_calls += 1
        return self.judge

    def runner(self, m, code=None, pipeline=None):
        return SR.Runner(m, self.ctx, SPEC, code=code or CODE, pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def pass_plan(self, n=14, c=8, f_a=3):
        """judgement: L n / C c (b) testable and L f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", "L"): {**{k: (True, True, False) for k in jb[:n]}, **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "C"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through_gate2(w, m, r=None):
    r = r or w.runner(m)
    assert r.stage_reuse()["outcome"] == "PASS"
    assert r.stage_set()["outcome"] == "PASS"
    assert r.stage_smoke()["problems"] == []
    r.stage_oc()
    r.stage_gate2_oc()
    assert r.stage_gate2()["outcome"] == "PASS"
    return r
