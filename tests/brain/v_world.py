"""A scripted V world for the runner tests (no engine, no pool): a fake ctx (R's, T's and U's summaries as the reuse
source, T's z rows and U's entry_f0 rows for the path gate, the static edit facts, 20 fake KC candidate odours of which
5 are U's, a set callable that honours block kc_input's values and block set, the reference set as 48 fake odours, 112
fake calibration odours, R's even raw as fabricated entries, P's block as fixture rows), a ZScripted reference / rest
measurer whose counts depend on the edit (none: block h4's z of this world, A 10 / 9, P 26 / 19; APL->MBON05 only: T's
lever shape, MBON13 Δ 1 — fails the guard; L_V: z_V A 6 / 3, P 80 / 20, Δ 6 — passes), and a Scripted measurer built
per z (`at(z)`) whose oracle writes real cache-like files under results/v/cache keyed by (block, edit) plans, whose P
arms come from s_fixtures and whose KC activity is scripted per (block, edit). Default even plan: L_V testable_b 13,
punishment passes everywhere (net drops 0)."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path

from flymon.brain import v_rules
from flymon.brain import v_runner as VR
from flymon.brain.config import Params
from flymon.brain.h3_store import canonical
from flymon.brain.p_spec import SPEC as P_SPEC
from flymon.brain.r_measure import RMeasurer
from flymon.brain.u_measure import is_u_edit, u_edit
from flymon.brain.v_spec import LEVER_V
from flymon.brain.v_spec import SPEC as V_SPEC
from tests.brain.r_fixtures import READOUT, TYPES, Z, fake_oracle, fake_rows, got_for, key_of
from tests.brain.s_fixtures import C1, arm_row, p_rows, stimuli
from tests.brain.u_fixtures import ref_rows, rest_rows
from tests.brain.v_fixtures import CANDS, U_IDS, r_doc, t_doc, u_doc  # noqa: F401  (re-exported)

SPEC = dataclasses.replace(V_SPEC, z_h4=(("A", Z["A"]), ("P", Z["P"])), sha_none="sha-C", sha_zero="sha-L0",
                           sha_combined="sha-V", t_measure_key_t="t" * 64, t_z_detail_sha256="d" * 64,
                           u_measure_key_u="u" * 64, u_path_detail_sha256="e" * 64, u_scan_detail_sha256="f" * 64)
CODE = {"key": SPEC.r_shared_key}
TCODE = {"key": "t" * 64}
UCODE = {"key": "u" * 64}
PIPE = {"key": "p" * 64}
N_REF = 96
APL = u_edit(0.0)
ZV = {"A": (6.0, 3.0), "P": (80.0, 20.0)}
REC_TYPES = list(dict.fromkeys(TYPES + list(SPEC.chain_types)))
EXTRA = {t: 1 for t in REC_TYPES if t not in ("MBON13", "MBON05")}
CHAIN = dict(SPEC.contrast_declared()["chain_entry"])
COUNTS = {"none": ([1, 19] * 48, [7, 45] * 48),                       # block h4's z of this world; Δ 10
          APL: ([0, 2] * 48, [60, 100] * 48),                          # MBON13 Δ 1, zero share 0.5 (T's lever shape)
          LEVER_V: ([3, 9] * 48, [60, 100] * 48)}                      # z_V A 6 / 3, P 80 / 20; MBON13 Δ 6
_OC = {}


def sha_of(edit: str) -> str:
    return {"none": "sha-C", APL: "sha-L0", LEVER_V: "sha-V"}.get(edit, f"sha-{edit}")


def edges_of(edit: str) -> int:
    return SPEC.lever_edges if is_u_edit(edit) else 0


def cached_oc_cluster(cb, ca):
    k = json.dumps([cb, ca])
    if k not in _OC:
        _OC[k] = REAL_OC_CLUSTER(cb, ca)
    return _OC[k]


REAL_OC_CLUSTER = v_rules.oc_cluster


def t_rows() -> dict:
    """T's z rows of this world: "none" = the unedited counts, T's lever = the APL->MBON05-only counts."""
    out = {}
    for t_edit, edit, e, sha in ((SPEC.t_none_edit, "none", 0, "sha-C"), (SPEC.t_lever_edit, APL, 2, "sha-L0")):
        a, p = COUNTS[edit]
        out[t_edit] = dict(ref=[dict(r, types={"MBON13": r["types"]["MBON13"], "MBON05": r["types"]["MBON05"]})
                                for r in ref_rows(a, p, e, sha)], rest=rest_rows(N_REF, 0, e, sha))
    return out


def csc_facts(**kw) -> dict:
    f = dict(sha_none="sha-C", csc_sha256="sha-V", apl_edges=2, block_edges=dict(sorted(CHAIN.items())), changed=13,
             changed_pairs={}, mbon05_apl=dict(n=2, before=[-7.8651862, -12.928195], after=[-7.8651862, -12.928195],
                                               same_bits=True))
    return dict(f, **kw)


class ZScripted:
    """u_measure.UZMeasurer's interface; counts[edit] = (MBON13 counts, MBON05 counts); edges / blocks override the
    edge labels per edit."""

    def __init__(self, counts=None, edges=None, blocks=None, n=None):
        self.counts = {**COUNTS, **(counts or {})}
        self.edges, self.blocks, self.n = dict(edges or {}), dict(blocks or {}), n
        self.calls = []

    def _rows(self, edit):
        a, p = self.counts[edit]
        bl = self.blocks.get(edit, CHAIN if edit == LEVER_V else {})
        return a, p, self.edges.get(edit, edges_of(edit)), bl

    def reference(self, edit, odors, strength, settle_ms, read_steps):
        self.calls.append(("reference", edit, len(odors)))
        a, p, e, bl = self._rows(edit)
        rows = ref_rows(a, p, e, sha_of(edit), extra=EXTRA, blocks=bl)
        return rows[:self.n] if self.n else rows

    def rest(self, edit, seeds, settle_ms, read_steps):
        self.calls.append(("rest", edit, len(seeds)))
        _, _, e, bl = self._rows(edit)
        return rest_rows(len(seeds), 0, e, sha_of(edit), extra=EXTRA, blocks=bl)


def u_rows() -> dict:
    zm = ZScripted()
    return dict(edit=LEVER_V, ref=zm.reference(LEVER_V, [None] * 48, 0.35, 800.0, 600),
                rest=zm.rest(LEVER_V, list(range(N_REF)), 800.0, 600))


class Scripted:
    """RMeasurer's interface, built per z through at(z). plan[(block, edit)][pair key] = (r_ok, p_ok, bal) (default:
    punish passes only); override[(block, cond name)] = dict(edges=, edit=, sha=) changes one block's condition only;
    drop = {edit: P drop}; arm_edges = {edit: edges}; kc = {edit or (block, edit): {odour: frac}};
    st["fail_once"] makes the next oracle call raise like an interrupted measurement."""

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
        return self.arm_edges.get(edit, edges_of(edit))

    def arms(self, items, readout, punish_type, block, nspec):
        self.calls.append(("arms", block, len(items)))
        return [arm_row(i, self._drop(i["edit"]), sha_of(i["edit"]), self._edges(i["edit"])) for i in items]

    def activity(self, odours, edit, s, seeds, block):
        self.calls.append(("activity", block, edit, len(odours)))
        fr = self.kc.get((block, edit), self.kc.get(edit, {}))
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
        monkeypatch.setattr(VR.v_rules, "oc_cluster", cached_oc_cluster)
        self.judge_commits, self.judgement_calls, self.set_calls, self.odour_calls = [], 0, 0, 0
        self.r, self.r_git = r_doc(), dict(tracked=True, dirty=False, judge_commits=["r"])
        self.t, self.t_git = t_doc(), dict(tracked=True, dirty=False, judge_commits=[])
        self.u, self.u_git = u_doc(), dict(tracked=True, dirty=False, judge_commits=[])
        self.t_rows, self.u_rows, self.facts, self.short = t_rows(), u_rows(), csc_facts(), False
        monkeypatch.setattr(VR, "summary_git",
                            lambda p: dict(tracked=True, dirty=False, judge_commits=list(self.judge_commits)))
        monkeypatch.setattr(VR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=[], dirty_other=[]))
        self.even, self.judge = fake_rows(21, 18, turn0=0), fake_rows(21, 43, turn0=300)
        eb = self.keys(self.even, "b")
        self.r_even_plan = {"C": {k: (True, True, False) for k in eb[:7]},
                            "L": {k: (True, True, False) for k in eb[:16]}}
        self.even_plan = {("even", LEVER_V): {k: (True, True, False) for k in eb[:13]},
                          ("path", APL): dict(self.r_even_plan["L"]),
                          ("path", "none"): dict(self.r_even_plan["C"])}
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
                        enc_per={o: 0.05 for o in ids}, r_even=self._r_even,
                        p_ref=lambda wanted: {k: v for k, v in p_ref.items() if k in wanted},
                        p_inputs=lambda pspec, smoke_: (dict(c1=C1), stimuli(pspec)),
                        r_doc=lambda: json.loads(json.dumps(self.r)), r_git=lambda: dict(self.r_git),
                        t_doc=lambda: json.loads(json.dumps(self.t)), t_git=lambda: dict(self.t_git),
                        t_rows=lambda: json.loads(json.dumps(self.t_rows)),
                        u_doc=lambda: json.loads(json.dumps(self.u)), u_git=lambda: dict(self.u_git),
                        u_none=lambda: SPEC.u_kc_none(self.u), u_rows=lambda: json.loads(json.dumps(self.u_rows)),
                        csc_facts=lambda: dict(self.facts), kc_candidates=self._cands, v_set=self._v_set,
                        kc_record=lambda kc: dict(outside={"none": [], "lever": []}, dropped=[],
                                                  candidate_rows_dropped=0, set_rows={}),
                        set_odours=self._set_odours, set_e0_odours=self._set_e0_odours,
                        judgement_rows=self._judgement_rows, clusters=self._clusters)

    # ---- the set callables (V.2): honour block kc_input's values and block set --------------------------------------
    def _cands(self):
        return dict(odours={o: {"G": 1.0} for o in CANDS}, cap={o: 200.0 for o in CANDS}, cap_hz=1000 / 3,
                    n_rows=40, n_turns=1986)

    def _js(self, kc):
        lo, hi = SPEC.valid_band
        used = [o for o in CANDS if all(lo <= kc[e][o] <= hi for e in ("none", "lever"))][:10]
        b, a = (self.judge[:21], self.judge[21:]) if not self.short else ([], [])
        return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=296, last_turn_b=30, last_turn_a=296,
                    status="STOP_SET_SHORT" if self.short else "OK",
                    skipped=dict(cap=16, collision=0, e1=0, used=722, glom_dup=0, in_set=87, pool_only=171,
                                 kc_input=sum(not all(lo <= kc[e][o] <= hi for e in ("none", "lever"))
                                              for o in CANDS)),
                    n_opp=127, n_combos=1986, n_odours=len(used), e1_clashes=[], cap_fails=[], all_off_pool=True,
                    clusters_b=[["GROUND*", "NORMAL", 21]], clusters_a=[["FIRE*", 43]], digest_e0_b="b" * 64,
                    digest_e0_a="a" * 64, digest_keys="k" * 64, odour_ids=used)

    def _v_set(self, kc):
        self.set_calls += 1
        return self._js(kc)

    def _checked(self, kc, blk):
        from flymon.brain.v_pairs import check_v_set
        js = self._js(kc)
        bad = check_v_set(js, blk)
        if bad:
            raise ValueError("; ".join(bad))
        return js

    def _set_odours(self, kc, blk):
        self.odour_calls += 1
        return {o: {"G": 1.0} for o in self._checked(kc, blk)["odour_ids"]}

    def _set_e0_odours(self, kc, blk):
        return {f"{key_of(r)}|x": {"E": 1.0} for r in self._checked(kc, blk)["b"][:3]}

    def _judgement_rows(self, kc, blk):
        self._checked(kc, blk)
        self.judgement_calls += 1
        return self.judge

    def _clusters(self, rows):
        return {key_of(r): ("GROUND* 대 NORMAL" if r["axis"] == "b" else "FIRE*") for r in rows}

    def _r_even(self, rows, name):
        lever = name == "L"
        return got_for(rows, self.r_even_plan[name], sha="sha-L0" if lever else "sha-C", edges=2 if lever else 0,
                       edit=SPEC.t_lever_edit if lever else SPEC.no_edit)

    def runner(self, m, zm=None, code=None, tcode=None, ucode=None, pipeline=None):
        return VR.Runner(m.at, zm or ZScripted(), self.ctx, SPEC, code=code or CODE, tcode=tcode or TCODE,
                         ucode=ucode or UCODE, pipeline=pipeline or PIPE, archive_root=self.archive)

    def keys(self, rows, axis):
        return [key_of(r) for r in rows if r["axis"] == axis]

    def scripted(self, **kw):
        m = Scripted(**kw)
        m.plan = {**self.even_plan, **m.plan}
        return m

    def pass_plan(self, n=14, c=8, f_a=3):
        """judgement: L_V n / C c (b) testable and L_V f_a (a) testable and naive; punishment passes everywhere."""
        jb, ja = self.keys(self.judge, "b"), self.keys(self.judge, "a")
        return {("judge", LEVER_V): {**{k: (True, True, False) for k in jb[:n]},
                                     **{k: (True, True, True) for k in ja[:f_a]}},
                ("judge", "none"): {k: (True, True, False) for k in jb[:c]}}


def doc():
    return json.loads(Path(SPEC.summary).read_text())


def through(w, m, last="gate2", r=None, zm=None):
    r = r or w.runner(m, zm)
    for s in VR.ORDER[:VR.ORDER.index(last) + 1]:
        out = getattr(r, f"stage_{s}")()
        if s in VR.GATES:
            assert out["outcome"] == "PASS", (s, out)
        if s == "smoke":
            assert out["problems"] == [], out["problems"]
    return r
