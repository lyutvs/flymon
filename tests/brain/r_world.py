# tests/brain/r_world.py
"""A scripted R world for the runner tests (no engine, no pool): a fake ctx, a Scripted measurer that writes real
cache-like files (the seal and the judge re-read them), the reproduction gate's reference files, and a P judgement
stub (r_runner.p_judge is monkeypatched)."""
import hashlib
import json
from pathlib import Path

from flymon.brain import r_runner as RR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.r_measure import RMeasurer
from flymon.brain.r_spec import SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, key_of

SHA = {"L": "sha-L", "C": "sha-C", "E0": "sha-C"}


def calib():
    single = [SPEC.kc_repro_odours[0]] + [f"S{i}|T" for i in range(31)]
    dual = list(SPEC.kc_repro_odours[1:]) + [f"D{i}|T+U" for i in range(78)]
    return {o: {f"ORN_{i}": 1.0} for i, o in enumerate(single + dual)}, single, dual


def ref_name(oid):
    return oid.replace("|", "_").replace("+", "_") + ".json"


def arm_result(direction, arm, seed, edit="none"):
    probe = {"A": 10, "P": 11, "kc_frac": 0.05, "wall_s": 0.1}
    return dict(seed=int(seed), edit=edit, arm=arm, punish=arm == "punish", plastic=arm != "frozen", da_zero=False,
                csc_sha256="sha-L" if edit != "none" else "sha-C", pre={"x": dict(probe), "y": dict(probe)},
                post={"x": dict(probe), "y": dict(probe)}, weights_frac=0.0, weights_frac_A=0.0, weights_frac_P=0.0,
                w0_sha256="w", w_post_sha256="w", da_integral={}, wall_s=1.0)


class Scripted:
    """RMeasurer's interface. plan[(block, cond)][pair key] = (r_ok, p_ok, bal) (default: punish passes only);
    override[(block, cond)] = dict(edges=, edit=, sha=) changes one block's condition only; fail_once makes the next
    oracle call raise like an interrupted measurement. inputs() is a real RMeasurer's (Params()), and every oracle
    entry stores it the way RCache.put does, so the seal's stored-inputs check (r_records.judge_inputs) sees what a
    real run would write."""

    def __init__(self, plan=None, kc=0.045, override=None):
        self.plan, self.kc, self.override = plan or {}, kc, override or {}
        self.kc_override, self.fail_once = {}, False
        self.calls, self.last_wall_s, self.last_jobs = [], 2.0, 1
        self._rm = RMeasurer(None, None, SPEC, Params(), READOUT, Z, TYPES, 100)

    def inputs(self, row, cond, block, seeds):
        return self._rm.inputs(row, cond, block, seeds)

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, len(odours)))
        lever = edit == SPEC.lever_edit
        v = self.kc if lever else 0.05
        return {o: dict(frac=list(self.kc_override.get(o, [v] * len(seeds))), max_win=[5] * len(seeds),
                        csc_sha256=["sha-L" if lever else "sha-C"], edit_edges=[SPEC.lever_edges if lever else 0])
                for o in odours}

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [dict(arm_result(i["direction"], i["arm"], i["seed"], i["edit"]),
                     r=dict(edit_edges=SPEC.lever_edges if i["edit"] == SPEC.lever_edit else 0, p_type="MBON05"),
                     direction=i["direction"], x=i["x"], y=i["y"], point=[float(v) for v in i["point"]])
                for i in items]

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
            f = Path("results/r/cache/r_oracle") / f"{ck[:24]}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            ins = json.loads(canonical(self.inputs(r, cond, block, seeds)))
            f.write_text(json.dumps({"key": ck, "kind": "r_oracle", "inputs": ins, "result": res}))
            out.append(dict(key=k, result=json.loads(json.dumps(res)), cache_key=ck, cache_file=str(f)))
        return out


class World:
    def __init__(self, tmp_path, monkeypatch, p_label="LEARNS_CONFIRMATORY"):
        monkeypatch.chdir(tmp_path)
        self.judge_commits, self.p_label, self.judgement_calls = [], p_label, 0
        monkeypatch.setattr(RR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(RR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        monkeypatch.setattr(RR, "p_judge", self._p_judge)
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 32, turn0=64)
        odours, single, dual = calib()
        self.enc = {"strength": {"configs": {SPEC.config: {"table": {str(SPEC.strength): {
            "per_odour": {o: 0.05 for o in odours}}}}}}}
        d = tmp_path / SPEC.ref_dir
        d.mkdir(parents=True)
        for o in SPEC.kc_repro_odours:
            (d / ref_name(o)).write_text(json.dumps({"kind": "activity",
                                                     "result": {"frac": [0.05] * 8, "max_win": [5] * 8}}))
        q0 = tmp_path / "q0" / "0.json"
        q0.parent.mkdir()
        ref = fake_oracle(n_rep=8, n_act=8)
        q0.write_text(json.dumps({"kind": "oracle", "result": {k: v for k, v in ref.items() if k not in ("q", "r")}}))
        self.archive = tmp_path / "archive"
        self.ctx = dict(params=Params(), readout=READOUT, z=Z, types=TYPES, n_kc=100, enc=self.enc,
                        calib=(odours, single, dual), cap_ok=True, even_rows=self.even,
                        judgement_rows=self._judgement_rows, kc_ref=ref_name, p_block={"measure_key": "K"},
                        p_ref=lambda wanted: {k: arm_result(*k) for k in wanted}, p_inputs=self._p_inputs,
                        q0_ref=lambda row: dict(path=str(q0), ok=True))

    def _judgement_rows(self):
        self.judgement_calls += 1
        return self.judge

    def _p_inputs(self, pspec, smoke_):
        names = {s for xy in pspec.pairs().values() for s in xy}
        return dict(c1=0.608), {n: {"odor": {f"G_{n}": 1.0}} for n in names}

    def _p_judge(self, rows, z, c1, spec):
        if self.p_label == "INVALID":
            return dict(outcome="INVALID", label="INVALID", reasons=["broken"], n_rows=len(rows))
        return dict(outcome="JUDGED", label=self.p_label, reasons=[], n_rows=len(rows),
                    directions={"r1": {"ell": 1.79}, "r2": {"ell": 2.16}})

    def runner(self, m, code="r" * 64):
        return RR.Runner(m, self.ctx, SPEC, code={"key": code}, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def pass_plan(self, n=14, c=8, f_a=3, even_l=12, even_c=7):
        """even: L even_l / C even_c (b) testable; judgement: L n / C c (b) testable and L f_a (a) testable and naive;
        punishment passes everywhere (no G_fail)."""
        eb, jb, ja = self.keys(self.even, "b"), self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("even", "L"): {k: (True, True, False) for k in eb[:even_l]},
                ("even", "C"): {k: (True, True, False) for k in eb[:even_c]},
                ("judge", "L"): {**{k: (True, True, False) for k in jb[:n]}, **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "C"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through_gate3(w, m, r=None):
    r = r or w.runner(m)
    assert r.stage_repro()["passed"]
    assert r.stage_smoke()["problems"] == []
    r.stage_oc()
    assert r.stage_gate1()["outcome"] == "PASS"
    assert r.stage_gate2()["outcome"] == "PASS"
    r.stage_even()
    assert r.stage_gate3()["outcome"] == "PASS"
    return r
