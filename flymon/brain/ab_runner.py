"""Spec AB's stage chain (AB.7 6784–6870; plan Reading 1): stage0 (0a) → reuse (0b) → generate (0c) → seal_code (0d)
→ cal_gate (0e) → futility (0f) → kc_input (3) → set (4) → smoke (5) → oracle (6) → screen (7) → calibrate (7b) →
learn (8) → verdict (9) → records (10), one block each in results/summary/ab_learning.json, written only through
ab_store. AB.2 (6648–6652): the assembly is AB's own; aa_runner is used for its pure helpers only (units, pair_counts,
by_pair, manifest, the seed spec of AA's main set), W's ctx (w_runner.build_ctx, read only) for params / readout /
main_rows / even_rows; nothing writes through aa_store / y_store / w_store / v_store and nothing assigns to another
module (plan Global Constraints).
- Part A (plan Task 7, Stage I): env, closure, seal, ctx, chain, cells, stage0 (bench and the AA differential), reuse,
  generate, seal_code, restart_preseal. A stage without a `stage_<name>` method refuses (exit 2, "not implemented
  yet"), so nothing past the built stages can start (plan section "Staged build").
- Part B1 (plan Task 8a, Stage I): the pooled checkpointing cell executor (CalExec / _cal_job; spawn processes,
  checkpoints under results/ab/cal/ and results/ab/fut/, plan Reading 9), cal_gate (0e: reservation incl. the 0f
  estimate, AB.5 on g 5 · 6 · 7, measured cap between cells, STOP_CALIBRATION, the synthetic validation) and futility
  (0f, AB.9.3: reservation, the AA source bit check, P̂ < 0.5 → STOP_FUTILE, the record grid on the records ledger).
- Chain: refusals (exit 2); non-stage keys (budget, candidates, invalid, restart_preseal_<n>, defect_<n>, reseal_<n>,
  <stage>_v<n>) are not stages. Stages after reuse also refuse when ctx's measurement keys differ from Y's (W.8 form).
- Env (AB.7 0a): env_now() at every start / resume vs every committed block and the stage's first start
  (progress/<stage>.env.json); a difference → core-ledger entry {stage, env_mismatch}, exit EXIT_ENV.
- Seal (AB.7 0d): the stages after seal_code compare seal_now() with block seal_code (or the newest reseal_<n>);
  a difference → exit EXIT_SEAL.
- Archive (AB.7 머리): every non-INVALID block's files are copied once before the block is written; for the learn
  stage an archive that holds other bytes is STOP_MACHINE〈8〉 (AB.7 8 "같은 sha면 건너뜀, 다르면 STOP_MACHINE"),
  for every other stage it refuses (ab_store.archive, exit 2).
- restart_preseal (AB.7 0d, plan Reading 15): before the first 2nd-gen measurement only."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from ..agent.e_runner import summary_git
from . import aa_estimate, aa_runner, ab_estimate, ab_pairs, ab_rules, ab_store, w_records, w_verdict, y_rules
from .ab_spec import SPEC as AB
from .ab_spec import ABSpec
from .h3_store import ROOT, canonical, sha256_file
from .h3_store import git_state as _h3_git_state
from .r_store import load_manifest
from .v_spec import SPEC as V_SPEC
from .w_measure import KIND
from .r_pairs import row_key
from .w_spec import BRAINS
from .w_spec import SPEC as W_SPEC
from .y_spec import SPEC as Y_SPEC

EXIT_ENV, EXIT_SEAL = 6, 7
WINDOWS = aa_runner.WINDOWS
TIMING = aa_runner.TIMING
# aa_runner's pure helpers (AB.2: imported, never edited); module names here so the runner tests can inject mutants
units = aa_runner.units
pair_counts = aa_runner.pair_counts
by_pair = aa_runner.by_pair
manifest = aa_runner.manifest
aa_seeds = aa_runner.aa_w_spec                    # AA's main-set seeds (Y's 62M / 64M) — the differential test only
MANIFEST_STAGES = ("smoke", "oracle", "screen", "learn", "records")   # blocks whose raw files are archived with them
RECOMPUTE_STAGES = ("set", "oracle", "screen", "calibrate", "learn", "verdict", "records")   # plan Reading 16


# ================================================================ files, env, closure, seal (AB.7 0a / 0d)
def ab_files() -> list:
    out = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "flymon/brain").glob("ab_*.py"))
    return out + (["scripts/run_ab.py"] if (ROOT / "scripts/run_ab.py").exists() else [])


def files_sha(files) -> dict:
    return {f: sha256_file(ROOT / f) for f in files}


def closure(files) -> list:
    """Every flymon file reachable from `files` through relative imports (levels 1 and 2), recursively; the files
    themselves included. A missing target is skipped."""
    import ast
    seen, todo = set(), [str(f) for f in files]
    while todo:
        p = todo.pop()
        if p in seen or not (ROOT / p).exists():
            continue
        seen.add(p)
        base = Path(p).parent
        for n in ast.walk(ast.parse((ROOT / p).read_text())):
            if not isinstance(n, ast.ImportFrom) or n.level not in (1, 2):
                continue
            pkg = base if n.level == 1 else base.parent
            if n.module:
                mod = pkg.joinpath(*n.module.split("."))
                todo.append(f"{mod.as_posix()}.py")
                todo += [f"{mod.as_posix()}/{a.name}.py" for a in n.names]      # `from ..agent import e_pairs`
            else:
                todo += [f"{pkg.as_posix()}/{a.name}.py" for a in n.names]      # `from . import x`
    return sorted(seen)


def imported_files() -> list:
    """The flymon files the ab_* files import (closure), without the ab_* files themselves (a patch changes ab_files
    only, so the defect procedure can rebase on that one field)."""
    own = set(ab_files())
    return [f for f in closure(ab_files()) if f not in own]


def env_now() -> dict:
    import importlib.metadata

    import scipy
    return dict(uv_lock_sha256=sha256_file(ROOT / "uv.lock"), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, poke_env=importlib.metadata.version("poke-env"), platform=platform.platform(),
                ab_files=files_sha(ab_files()), imported=files_sha(imported_files()))


def reuse_files(s=AB) -> list:
    """AB.2 0b: t_* · v_* · w_* · y_* · aa_* (flymon/brain) and z_split.py (plan Reading 4)."""
    return sorted(p.relative_to(ROOT).as_posix() for g in s.reuse_globs for p in (ROOT / "flymon/brain").glob(g)) + \
        list(s.reuse_extra)


def hashed_files(s=AB) -> list:
    return ab_files() + reuse_files(s) + [s.aa_summary, Y_SPEC.summary, W_SPEC.summary, W_SPEC.v_summary]


def git_state() -> dict:
    return _h3_git_state(files=hashed_files())


def seal_files(s=AB) -> list:
    """AB.7 0d: the sealed AB rule files and every flymon file they import."""
    return sorted(set(s.seal_files) | set(closure(s.seal_files)))


seal_closure = seal_files                         # the name ab_spec's docstring uses


def seal_now(s=AB) -> dict:
    files = files_sha(seal_files(s))
    nums = json.loads(canonical(s.seal_fields()))
    return dict(files=files, numbers=nums,
                key=hashlib.sha256(canonical(dict(files=files, numbers=nums)).encode()).hexdigest())


def stream_vectors(s=AB) -> list:
    """AB.2: the first three integers(0, 2^63) of every sealed test stream, recomputed with aa_estimate.stream."""
    hi = int(np.iinfo(np.int64).max) + 1
    out = []
    for (root, *tags), _want in s.stream_vectors:
        r = aa_estimate.stream(s.root(root), *tags)
        out.append([[root, *tags], [int(v) for v in r.integers(0, hi, size=len(_want))]])
    return out


# ================================================================ seeds (AB.2 AB 시드 블록)
def ab_w_spec(s=AB):
    """W's seed methods on AB's blocks: judged probe 88_100_000 + c·4_000 + f·100 + k, training 89_500_000 + c·40_000 +
    f·1_000 + t, AB's smoke and oracle seeds (c = the AB candidate number)."""
    o = s.oracle_seeds()
    return dataclasses.replace(W_SPEC, probe_seed0=s.probe_seed0, train_seed0=s.train_seed0,
                               smoke_probe_seed0=s.smoke_probe_seed0, smoke_train_seed0=s.smoke_train_seed0,
                               smoke_oracle_seed0=s.smoke_oracle_seed0, oracle_act_seeds=tuple(o["act"]),
                               oracle_select_seeds=tuple(o["select"]), oracle_report_seeds=tuple(o["report"]))


def smoke_seed_set(s=AB) -> frozenset:
    probe = {s.smoke_probe_seed0 + f * s.smoke_probe_span + k for f in range(s.smoke_fly_span)
             for k in range(s.smoke_probe_span)}
    return frozenset(probe | {int(x) for v in s.smoke_oracle_seeds().values() for x in v})


def _same(a, b) -> bool:
    """Bit equality through the canonical JSON text."""
    return canonical(ab_store.to_json(a)) == canonical(ab_store.to_json(b))


_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")                    # ANSI CSI escape sequences (colours, cursor moves)


# ================================================================ the differential test (AB.7 0 6783)
def differential(ctx, s=AB) -> dict:
    """AA S1's first `diff_pairs` pairs in c order, assembled here with AA's main-set seeds (aa_runner.aa_w_spec():
    62_000_000 / 64_000_000), read from results/aa/cache by W measurement key (cache hits only — a miss is a failure,
    never a measurement). Bit-equal: unit cache keys and inputs vs AA's learn manifest, count arrays vs AA's manifest
    raw, gate_stats (fly labels) / winsorized fly means / the two raw-contrast means vs results/aa/records.json, the
    association means vs estimate.json; plus the TS path: ab_estimate.ts_reps + ts_limits at α p_grid[0] (= AA's
    2.5 / 97.5) on all S1 pairs with AA's stream (87_000_000, "pool", "S1") = AA's pooled CIs."""
    aa = ctx["docs"]()["aa_learning"]
    s1 = [p for p in aa["screen"]["pairs"] if p["passed"]]
    first = sorted(s1, key=lambda p: int(p["c"]))[:s.diff_pairs]
    rows = {int(r["c"]): r for r in ctx["aa_rows"]()}
    learn = ctx["aa_detail"]("learn")["manifest"]
    man = {m["key"]: m for m in learn}
    wm = ctx["aa_reader"]()
    z = {k: (float(v[0]), float(v[1])) for k, v in ctx["w_z"]().items()}
    rec = ctx["aa_detail"]("records")["block"]["pairs"]
    est = ctx["aa_detail"]("estimate")["primary"]
    why, per = [], {}
    if len(first) != s.diff_pairs:
        why.append(f"AA S1 쌍 {len(first)}개 < 차등 시험 쌍 수 {s.diff_pairs}")
    missing = [int(p["c"]) for p in first if int(p["c"]) not in rows]
    if missing:
        why.append(f"AA S1 c {missing}가 AA 주 세트 행에 없음")
    U = units([rows[int(p["c"])] for p in first if int(p["c"]) in rows], "main", s.probes, s.flies,
              V_SPEC.lever_edit, ws=aa_seeds())
    ukeys = [f"{u['pair']}|{u['fly']}|{u['brain']}" for u in U]
    dup = sorted({k for k in ukeys if ukeys.count(k) > 1})
    if dup:
        why.append(f"AB 단위 키 중복 {len(dup)}개: {dup[:3]}")
    want_pairs = {row_key(rows[int(p["c"])]) for p in first if int(p["c"]) in rows}
    n_man = sum(1 for k in man if k.rsplit("|", 2)[0] in want_pairs)
    if len(U) != n_man:
        why.append(f"단위 수 {len(U)} ≠ AA learn 매니페스트의 그 쌍 단위 {n_man}")
    got = []
    for u, k in zip(U, ukeys):
        ins = wm.inputs(u, "learn")
        m = man.get(k)
        if m is None:
            why.append(f"{k}: AA learn 매니페스트에 없음")
        elif wm.cache.key(KIND, ins) != m["cache_key"]:
            why.append(f"{k}: 캐시 키 ≠ AA learn 매니페스트")
        res = wm.cache.get(KIND, ins)
        if res is None:
            why.append(f"{k}: 캐시 적중 실패(새 측정 없음)")
            continue
        if m is not None and not _same(ins, json.loads(Path(m["cache_file"]).read_text())["inputs"]):
            why.append(f"{k}: 단위 입력 ≠ AA")
        got.append(dict(unit=u, result=res))
    flies = list(range(s.flies))
    aa_got, bad = load_manifest(learn)
    why += [f"AA learn 원자료: {b}" for b in bad[:3]]
    aa_by = {}
    for g in aa_got:
        pair, fly, brain = g["key"].rsplit("|", 2)
        aa_by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=g["result"]))
    data = {}
    for k, v in aa_by.items():
        try:
            data[k] = w_records.pair_data(v, flies)
        except KeyError as e:                                   # a unit of AA's manifest is unreadable or absent
            why.append(f"AA learn 원자료 {k}: 단위 {e} 없음")
    n_pair = len(flies) * len(BRAINS)
    for k, rows_k in by_pair(got).items():
        if len(rows_k) != n_pair or k not in data or k not in rec:
            why.append(f"{k}: 적중 단위 {len(rows_k)} · AA 자료 {'있음' if k in data else '없음'}")
            continue
        mine = pair_counts(rows_k, flies)
        try:
            st = ab_estimate.pair_stats(mine, z, s)
        except ValueError as e:
            why.append(f"{k}: {e}")
            continue
        r_ = rec[k]
        e_ = next((p for p in est["pairs"] if p["key"] == k), None)
        ok = dict(
            counts=all(np.array_equal(mine[x], data[k][x]) for x in w_verdict.STAGES),
            means=all(float(st["D"][:, i].mean()) == r_["gates"][g]["mean"] for i, g in enumerate(w_verdict.GATES)),
            flies=all([aa_estimate._lab(v) for v in st["raw"][:, i]] == r_["gates"][g]["fly"]
                      for i, g in enumerate(w_verdict.GATES)),
            raw=all(float(st["R"][:, aa_estimate.GI[g]].mean()) == r_["raw"][g]["mean"] for g in aa_estimate.RAW),
            estimate=e_ is not None and all(float(st["D"][:, aa_estimate.GI[g]].mean()) == e_[g]["mean"]
                                            for g in aa_estimate.PRIMARY))
        per[k] = ok
        why += [f"{k}: {n} 불일치" for n, v in ok.items() if not v]
    why += [f"{k}: 대조 안 됨" for k in sorted(want_pairs) if k not in per]
    keys = [p["key"] for p in s1]                          # AA's screen order = c order (AA estimate_set's keys)
    ts_ok = {}
    if all(k in data for k in keys) and keys:
        try:
            X = np.stack([ab_estimate.pair_stats(data[k], z, s)["D"] for k in keys])
            reps = ab_estimate.ts_reps(X, y_rules.merge_groups(keys), aa_estimate.stream(s.aa_boot_seed, "pool", "S1"),
                                       s)
            lo, hi = ab_estimate.ts_limits(reps, [s.p_grid[0]], s)      # AA's (2.5, 97.5)
            ts_ok = {g: [float(lo[0][aa_estimate.GI[g]]), float(hi[0][aa_estimate.GI[g]])] == est["pooled"][g]["ci"]
                     for g in aa_estimate.PRIMARY}
        except (ValueError, KeyError) as e:
            why.append(f"TS 경로: {type(e).__name__}: {e}")
    else:
        why.append("TS 경로: AA S1 원자료가 모두 있지 않음")
    why += [f"TS {g}: AA pooled CI와 다름" for g, v in ts_ok.items() if not v]
    return dict(ok=not why, reasons=why, n_units=len(U), n_hits=len(got), pairs=per, ts=ts_ok)


# ================================================================ the 0f source (AB.2 6655@248cd25, AB.7 0f)
def read_fut_source(s, aa_doc: dict, y_doc: dict) -> dict:
    """AA S1's learn units (records.s1_records.keys) read from AA's learn manifest, read only: the stored file bytes'
    sha256 against the manifest's (→ bad_sha; a stored key other than the manifest's cache_key counts there too), a
    missing file → missing; z = Y pilot's full-precision z_V (AA's)."""
    rec = aa_doc.get("records") or {}
    s1 = rec.get("s1_records") or {}
    keys = list(s1.get("keys") or [])
    want = set(keys)
    p = Path(s.aa_learn_detail)
    man = (json.loads(p.read_text()).get("manifest") or []) if p.exists() else []
    by, missing, bad, n = {}, [], [], 0
    for m in man:
        pair, fly, brain = m["key"].rsplit("|", 2)
        if pair not in want:
            continue
        n += 1
        f = Path(m["cache_file"])
        if not f.exists():
            missing.append(m["key"])
            continue
        b = f.read_bytes()
        if hashlib.sha256(b).hexdigest() != m["sha256"]:
            bad.append(m["key"])
            continue
        d = json.loads(b)
        if d.get("key") != m["cache_key"]:
            bad.append(m["key"])
            continue
        by.setdefault(pair, []).append(dict(unit=dict(pair=pair, fly=int(fly), brain=brain), result=d["result"]))
    zv = (y_doc.get("pilot") or {}).get("z_V") or {}
    return dict(keys=keys, by_pair=by, z={k: (float(v[0]), float(v[1])) for k, v in zv.items()}, s1=s1,
                pairs=rec.get("pairs") or {}, n_units=n, n_pairs=len(keys), missing=missing, bad_sha=bad)


def refuse(msg: str, code: int = 2):
    ab_store.refuse(msg, code)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _head_summary(path):
    """The summary as committed at HEAD (bytes), or None when HEAD does not track it (or there is no repository)."""
    import subprocess
    r = subprocess.run(["git", "show", f"HEAD:./{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def _base(stage: str) -> str:
    """`set_v1` → `set` (a defect-procedure block follows its stage's ledger and files)."""
    return re.sub(r"_v\d+$", "", stage)


def archive_clash(files: list, dest: str, s) -> list:
    """Reasons the archive <archive_root>/<dest>/ holds another file set or other bytes (ab_store.archive's own
    test, without refusing); [] when it is absent or holds exactly these files byte-identically."""
    d = Path(os.path.expanduser(s.archive_root)) / dest
    if not d.exists():
        return []
    want = {d / Path(f).parent.name / Path(f).name: f for f in files}
    have = {p for p in d.rglob("*") if p.is_file()}
    if have != set(want):
        return [f"보관 사본 {d}의 파일 집합이 다름"]
    return [f"보관 사본 {dst} sha256 ≠ {f}" for dst, f in want.items() if sha256_file(dst) != sha256_file(f)]


# ================================================================ the engine context (read only)
def build_ctx(npz: str, s=AB) -> dict:
    """W's context (w_runner.build_ctx, built once, lazily: params, readout, types, n_kc, enc, even_rows, main_rows),
    the populations, the four summaries read only, and the AB pieces (plan Task 7 table); nothing writes."""
    import subprocess

    from . import w_runner
    from .r_measure import RMeasurer
    from .u_measure import UPool, u_measure_key
    from .v_runner import kc_values
    from .w_measure import WMeasurer, w_measure_key
    cache = {}

    def once(name, fn):
        if name not in cache:
            cache[name] = fn()
        return cache[name]

    def wctx():
        return once("w", lambda: w_runner.build_ctx(W_SPEC, npz))

    def pops():
        from .circuits import Populations
        from .connectome import Connectome
        return once("pops", lambda: Populations.from_connectome(Connectome.load(npz)))

    def wkey():
        return once("wk", lambda: w_measure_key(npz))

    def keys():
        return dict(w_measure_key=wkey()["key"], u_measure_key=once("uk", lambda: u_measure_key(npz))["key"])

    def docs():
        return {n: ab_store.read_summary(p) for n, p in (("v_lever", W_SPEC.v_summary), ("w_learning", W_SPEC.summary),
                                                         ("y_learning", Y_SPEC.summary), ("aa_learning", s.aa_summary))}

    def kc_v():
        return kc_values(docs()["v_lever"])

    def base():
        c = wctx()
        return once("base", lambda: ab_pairs.base_used_rows(pops(), c["enc"], c["params"], kc_v(),
                                                            docs()["v_lever"]["set"]["set"]))

    def lv():
        d = docs()
        b = base()
        return ab_pairs.lv_sources(d["aa_learning"], d["w_learning"], d["y_learning"], b[2]["rows"],
                                   wctx()["even_rows"], b[1], s)

    def generate(lv_ids):
        b = base()
        return ab_pairs.generate(b[0], b[1], b[2]["rows"], list(lv_ids), kc_v(), s)

    def attach(rows):
        return ab_pairs.attach(rows, pops(), wctx()["enc"])

    def smoke_row():
        got = [r for r in base()[1] if row_key(r) == s.smoke_pair]
        if len(got) != 1:
            raise ValueError(f"스모크 쌍 {s.smoke_pair}가 V 세트 행에 {len(got)}개")
        return attach(got)[0]

    def wm(pool, cache_obj):
        c = wctx()
        return WMeasurer(pool, cache_obj, c["params"], c["readout"], V_SPEC.p_type, W_SPEC.reward_dan,
                         W_SPEC.punish_dan, WINDOWS, TIMING)

    def rm(pool, root, smoke):
        c = wctx()
        return RMeasurer(UPool(pool), ab_store.ABCache(root, wkey(), smoke_seeds=smoke_seed_set(s)), V_SPEC,
                         c["params"], c["readout"], W_SPEC.z_v(), c["types"], c["n_kc"])

    def decl_sha(files: list, commit: str) -> dict:
        out = {}
        for f in files:
            r = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{f}"], capture_output=True)
            out[f] = hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None
        return out

    detail = dict(learn=s.aa_learn_detail, records=s.aa_records_detail, estimate=s.aa_estimate_detail)
    return dict(
        keys=keys, params=lambda: wctx()["params"], docs=docs, kc_v=kc_v, base=base,
        even_rows=lambda: wctx()["even_rows"], lv=lv, generate=generate,
        selftest=lambda: (lambda b: ab_pairs.selftest(b[0], b[1], kc_v(), s))(base()), attach=attach,
        odour_inputs=lambda ids: ab_pairs.new_odour_inputs(base()[0], ids), smoke_row=smoke_row,
        kc_measurer=lambda pool: rm(pool, s.cache_dir, False),
        oracle_measurer=lambda pool, smoke=False: rm(pool, s.smoke_cache_dir if smoke else s.cache_dir, smoke),
        learn_measurer=lambda pool, smoke=False: wm(pool, ab_store.ABCache(
            s.smoke_cache_dir if smoke else s.cache_dir, wkey(), smoke_seeds=smoke_seed_set(s))),
        aa_reader=lambda: wm(None, ab_store.ReadCache(s.aa_cache_dir, wkey())),
        aa_rows=lambda: wctx()["main_rows"](docs()["y_learning"]["digest"]["set"]),
        w_z=lambda: {k: (float(v[0]), float(v[1])) for k, v in docs()["w_learning"]["reuse"]["z_V"].items()},
        git_facts=w_runner.git_facts, decl_sha=decl_sha,
        aa_detail=lambda name: json.loads(Path(detail[name]).read_text()),
        fut_source=lambda: once("fut", lambda: (lambda d: read_fut_source(s, d["aa_learning"], d["y_learning"]))(
            docs())))


# ================================================================ the runner
class Runner:
    def __init__(self, ctx: dict, s=AB, summary_path=None, pool=None):
        self.ctx, self.s, self.pool = ctx, s, pool
        self.summary_path = str(summary_path or s.summary)
        self.workers = None
        self._s = dict(core=0.0, records=0.0)

    @property
    def plist(self) -> list:
        return [self.ctx["params"]()]

    # ---- chain ---------------------------------------------------------------------------------------------------
    def _doc(self) -> dict:
        return ab_store.read_summary(self.summary_path)

    def _clean(self, stage: str) -> None:
        if Path(self.summary_path).exists():
            g = summary_git(self.summary_path)
            if not g["tracked"] or g["dirty"]:
                refuse(f"stage {stage}: {self.summary_path} has uncommitted changes; commit the previous block first")
        gs = git_state()
        if gs["dirty_hashed"]:
            refuse(f"stage {stage}: hashed files are dirty: {gs['dirty_hashed']}")

    def _require(self, stage: str) -> dict:
        s = self.s
        if stage not in s.stages:
            refuse(f"{stage} is not an AB stage")
        self._clean(stage)
        doc = self._doc()
        order = list(s.stages)
        i = order.index(stage)
        missing = [b for b in order[:i] if self._eff(doc, b) is None]
        if missing:
            refuse(f"stage {stage} needs block(s) {missing}")
        later = [b for b in order[i + 1:] if self._eff(doc, b) is not None]
        if later:
            refuse(f"stage {stage}: later block(s) {later} exist; AB never rewrites an earlier block")
        if stage in doc:
            refuse(f"stage {stage}: block {stage} exists; AB never rewrites a recorded block")
        if self._eff_name(doc, stage) != stage:
            refuse(f"stage {stage}: block {self._eff_name(doc, stage)} exists (defect procedure); run() never writes "
                   f"a stage that has a recomputed block")
        stopped = [b for b in order[:i] if self._eff(doc, b).get("outcome") != ab_rules.PASS]
        if stopped:
            refuse(f"stage {stage}: {self._eff_name(doc, stopped[0])} outcome "
                   f"{self._eff(doc, stopped[0]).get('outcome')} — AB stops there")
        if i > order.index("reuse"):
            k = self.ctx["keys"]()
            bad = [n for n in ("w_measure_key", "u_measure_key") if k.get(n) != getattr(Y_SPEC, n)]
            if bad:
                refuse(f"stage {stage}: measurement key(s) {bad} differ from Y's (W.8)")
        if stage in s.sealed_stages():
            self._seal_check(doc)
        self._env(stage, doc)
        return doc

    def _seal_check(self, doc: dict) -> None:
        """AB.7 0d: seal_now() against block seal_code, or against the newest reseal_<n> (the defect procedure)."""
        _n, rs = self._latest(doc, "reseal")
        blk = doc[rs] if rs else doc.get("seal_code")
        want = ((blk or {}).get("seal") or {}).get("key")
        if want is None or seal_now(self.s)["key"] != want:
            refuse(f"the sealed code differs from {rs or 'seal_code'} (AB.7 0d); restart_preseal before the first "
                   "measurement, the defect procedure after it", EXIT_SEAL)

    @staticmethod
    def _latest(doc: dict, prefix: str) -> tuple:
        """(n, name) of the newest `<prefix>_<n>` block, or (0, None)."""
        got = sorted((int(m.group(1)), k) for k in doc if (m := re.fullmatch(rf"{prefix}_(\d+)", k)))
        return got[-1] if got else (0, None)

    @staticmethod
    def _eff_name(doc: dict, stage: str) -> str:
        """The block that stands for `stage`: the newest `<stage>_v<n>` (defect procedure) or `stage`."""
        got = sorted((int(m.group(1)), k) for k in doc if (m := re.fullmatch(rf"{stage}_v(\d+)", k)))
        return got[-1][1] if got else stage

    def _eff(self, doc: dict, stage: str):
        return doc.get(self._eff_name(doc, stage))

    def _env(self, stage: str, doc: dict, drop_ab: bool = False) -> dict:
        """AB.7 0a at the start and at every resume. Once a reseal_<n> exists, the blocks (and first-start records)
        written before the newest reseal are compared without `ab_files` (the patch changes them by design);
        `drop_ab` (the reseal itself) drops `ab_files` everywhere; every other field is always compared."""
        now = env_now()
        _n, rs = self._latest(doc, "reseal")
        cut = doc[rs].get("written_at") if rs else None

        def base(env: dict, at) -> dict:
            if drop_ab or (cut is not None and (at is None or at < cut)):
                return {k: v for k, v in env.items() if k != "ab_files"}
            return env
        blocks = [(b, base(v["env"], v.get("written_at"))) for b, v in doc.items()
                  if isinstance(v, dict) and isinstance(v.get("env"), dict)]
        p = Path(self.s.progress_dir) / f"{stage}.env.json"
        if p.exists():
            blocks.append((f"{stage} 첫 시작", base(json.loads(p.read_text()), None)))
        why = ab_rules.env_reasons(blocks, now)
        if why:
            ab_store.append_ledger(self.summary_path, dict(stage=stage, env_mismatch=why, at=_now(), wall_s=0.0),
                                   self.plist)
            refuse(f"stage {stage}: environment differs (AB.7 0a): {why[:5]}", EXIT_ENV)
        if not p.exists():                  # recorded only once it agrees with every committed block
            ab_store.write_json(str(p), now, self.plist)
        return now

    def _detail(self, stage: str) -> str:
        return f"{self.s.raw_dir}/{stage}.json"

    def _files(self, stage: str) -> list:
        """The stage's detail file, plus the raw files of its manifest (AB.7 "사본은 블록마다")."""
        p = Path(self._detail(stage))
        if not p.exists():
            return []
        out = [str(p)]
        if _base(stage) in MANIFEST_STAGES:
            d = json.loads(p.read_text())
            out += [m["cache_file"] for m in d.get("manifest") or []]
        return list(dict.fromkeys(out))

    # ---- candidates (AB.3 6691, plan Readings 17, 31) ---------------------------------------------------------------
    def _cands_path(self, stage: str) -> Path:
        return Path(self.s.progress_dir) / f"{stage}.candidates.json"

    def _cands(self, doc: dict, stage: str | None = None) -> list:
        """The candidate ledger: the running stage's progress file, else the tracked block; empty before order 4
        (the AB set does not exist yet — plan Reading 31)."""
        if stage is not None and self._cands_path(stage).exists():
            return json.loads(self._cands_path(stage).read_text())
        return [dict(x) for x in doc.get("candidates") or []]

    def _counts(self, doc: dict, stage: str | None = None) -> dict:
        return ab_rules.counts(self._cands(doc, stage))

    def _cands_set(self, stage: str, keys: list, state: str) -> list:
        """States only move forward: untouched → oracled → screened → training → trained."""
        rank = {x: i for i, x in enumerate(ab_rules.STATES)}
        want = set(keys)
        cur = self._cands(self._doc(), stage)
        for x in cur:
            if x["key"] in want and rank[state] > rank[x["state"]]:
                x["state"] = state
        ab_store.write_json(str(self._cands_path(stage)), cur, self.plist)
        return cur

    # ---- blocks ------------------------------------------------------------------------------------------------------
    def _write(self, stage: str, body: dict, core_s=None, records_s=None, ledger: bool = True,
               top: dict | None = None) -> dict:
        """One block (and its ledger line) through ab_store; the files are archived first unless INVALID. Learn: an
        archive holding other bytes is STOP_MACHINE〈8〉 (AB.7 8), never an overwrite."""
        s, extra = self.s, {}
        if body.get("outcome") != ab_rules.INVALID:
            files = self._files(stage)
            clash = archive_clash(files, stage, s) if _base(stage) == "learn" else []
            if clash:
                stop = ab_rules.machine_stop("학습", "; ".join(clash[:3]), self._counts(self._doc(), stage),
                                             s.stage_label("learn"))
                body = dict(body, **stop, archive_clash=clash)
            else:
                extra["archive"] = ab_store.archive(files, stage, s)
        k = self.ctx["keys"]()
        prog = self._prog(stage)
        block = ab_store.to_json(dict(body, **extra, stage=stage, order=s.stage_label(_base(stage)), env=env_now(),
                                      git=git_state(), written_at=_now(), w_measure_key=k.get("w_measure_key"),
                                      u_measure_key=k.get("u_measure_key")))
        core = float(prog["core"] if core_s is None else core_s)
        rec = float(prog["records"] if records_s is None else records_s)
        at = block["written_at"]
        if not ledger:
            led = rled = None
        elif _base(stage) in s.records_stages:
            led, rled = None, dict(stage=stage, wall_s=rec, at=at)
        else:
            led, rled = dict(stage=stage, wall_s=core, at=at), (dict(stage=stage, wall_s=rec, at=at) if rec else None)
        doc = self._doc()
        ab_store.write_summary_block(self.summary_path, stage, block, self.plist, led, rled,
                                     candidates=self._cands(doc, stage), extra=top)
        return block

    def archive(self, stage: str) -> list:
        blk = self._doc().get(stage)
        if blk is None:
            refuse(f"archive {stage}: no block {stage}")
        if blk.get("outcome") == ab_rules.INVALID:
            refuse(f"archive {stage}: block {stage} is INVALID; INVALID is never archived")
        return ab_store.archive(self._files(stage), stage, self.s)

    def run(self, stage: str) -> dict:
        if stage not in self.s.stages:
            refuse(f"{stage} is not an AB stage")
        fn = getattr(self, f"stage_{stage}", None)
        if fn is None:
            refuse(f"stage {stage} is not implemented yet")
        return fn()

    # ---- spend and cells -------------------------------------------------------------------------------------------
    def _prog_path(self, stage: str) -> Path:
        return Path(self.s.progress_dir) / f"{stage}.json"

    def _prog(self, stage: str) -> dict:
        p = self._prog_path(stage)
        d = json.loads(p.read_text()) if p.exists() else {}
        return dict(core=float(d.get("core", 0.0)), records=float(d.get("records", 0.0)))

    def _add(self, stage: str, sec: float, which: str = "core") -> None:
        d = self._prog(stage)
        d[which] += float(sec)
        ab_store.write_json(str(self._prog_path(stage)), dict(d, at=_now()), self.plist)

    def _finish(self, stage: str, t0: float) -> None:
        """The stage's own (non-cell) seconds go to its ledger."""
        which = "records" if _base(stage) in self.s.records_stages else "core"
        self._add(stage, (time.perf_counter() - t0) - self._s["core"] - self._s["records"], which)

    def _key(self, tag: str, key_obj) -> str:
        spec_sha = hashlib.sha256(canonical(self.s).encode()).hexdigest()
        return hashlib.sha256(canonical(dict(tag=tag, key=key_obj, s=spec_sha, ab=files_sha(ab_files()))).encode()
                              ).hexdigest()

    def _cell(self, stage: str, tag: str, key_obj, fn, which: str = "core"):
        """One resumable JSON cell progress/<stage>/<tag>.json (key = tag, key_obj, the spec and the ab_* files)."""
        key = self._key(tag, key_obj)
        p = Path(self.s.progress_dir) / stage / (_safe(tag) + ".json")
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except ValueError:
                d = {}
            if d.get("key") == key:
                return d["value"]
        t = time.perf_counter()
        v = ab_store.to_json(fn())
        ab_store.write_json(str(p), dict(key=key, value=v, at=_now()), self.plist)
        dt = time.perf_counter() - t
        self._add(stage, dt, which)
        self._s[which] += dt
        return v

    def _workers(self) -> int:
        return max(1, int(self.workers or getattr(self.pool, "n_workers", 0) or self.s.workers))

    def _core_h(self, doc: dict) -> float:
        """The core ledger so far (AB.7 예산), hours."""
        return sum(float(e.get("wall_s") or 0.0) for e in (doc.get("budget") or {}).get("ledger", [])) / self.s.s_per_h

    def _ticker(self, stage: str, which: str = "core"):
        """Every call adds the seconds since the previous call to progress/<stage>.json (they survive a kill); seconds
        that _cell already added to the same ledger since the previous call are left out (no double count)."""
        mark = [time.perf_counter(), self._s[which]]

        def tick(*_a):
            now = time.perf_counter()
            cells = self._s[which] - mark[1]
            self._add(stage, max(0.0, now - mark[0] - cells), which)
            mark[0], mark[1] = now, self._s[which]
        return tick

    def _tests_log(self) -> tuple:
        """The order-0 tests log (AB.7 0a): the last line `exit 0`, an "N passed" line, no failed / error summary — all
        read after stripping ANSI CSI escape sequences (pytest colours its summary line on a terminal)."""
        s = self.s
        log = Path(s.tests_log)
        text = _CSI.sub("", log.read_text()) if log.exists() else ""
        lines = text.strip().splitlines()
        passed = [ln for ln in lines if re.search(r"\d+ passed", ln)]
        tests = dict(path=s.tests_log, log_sha256=sha256_file(log) if log.exists() else None,
                     last_line=lines[-1] if lines else None, passed_line=passed[-1] if passed else None)
        bad = []
        if tests["last_line"] != s.tests_ok_line:
            bad.append(f"시험 로그 {s.tests_log}의 마지막 줄 {tests['last_line']!r} ≠ {s.tests_ok_line!r}")
        if tests["passed_line"] is None:
            bad.append(f"시험 로그 {s.tests_log}에 통과 개수 줄(\"N passed\")이 없음")
        fails = [ln for ln in lines if re.search(r"\b\d+ (failed|errors?)\b", ln)]
        if fails:
            bad.append(f"시험 로그 {s.tests_log}에 실패 · 오류 요약 줄: {fails[-1]!r}")
        return tests, bad

    # ================================================================ 0a: stage0 (AB.7 0a)
    def stage_stage0(self) -> dict:
        self._require("stage0")
        s, t0 = self.s, time.perf_counter()
        tests, bad = self._tests_log()
        bench = self._cell("stage0", "bench", s.seal_fields(), lambda: ab_estimate.bench(s))
        diff = differential(self.ctx, s)
        if not diff["ok"]:
            bad += [f"차등 시험: {x}" for x in diff["reasons"]]
        dec = dict(outcome=ab_rules.INVALID, reasons=bad) if bad else dict(outcome=ab_rules.PASS, reasons=[])
        numbers = ab_store.to_json(s)
        p = ab_store.write_json(self._detail("stage0"), dict(tests=tests, differential=diff, bench=bench,
                                                             numbers=numbers), self.plist)
        self._finish("stage0", t0)
        dsum = dict(ok=diff["ok"], n_units=diff["n_units"], n_hits=diff["n_hits"], pairs=diff["pairs"], ts=diff["ts"],
                    n_reasons=len(diff["reasons"]))
        return self._write("stage0", dict(
            dec, tests=tests, bench=bench, differential=dsum, decision_files=files_sha(ab_files() + seal_files(s)),
            seal=seal_now(s), numbers=numbers, numbers_sha256=hashlib.sha256(canonical(numbers).encode()).hexdigest(),
            detail_path=self._detail("stage0"), detail_sha256=sha256_file(p),
            note="AB.7 0a: 시험 로그, bench, AA 차등 시험(캐시 적중만, 비트 단위), 결정 파일, 수치 표, 환경 해시"
                 "(env — 이후 모든 단계 시작 · 재개에서 대조)."))

    # ================================================================ 0b: reuse (AB.2, AB.7 0b)
    def stage_reuse(self) -> dict:
        doc = self._require("reuse")
        s, t0, ctx = self.s, time.perf_counter(), self.ctx
        d = ctx["docs"]()
        v_check, w_digest, lv = [], {}, None
        try:
            js_w = ctx["base"]()[2]
            w_digest = {k: js_w.get(k) for k in ("digest_keys", "n_b", "n_a", "last_turn")}
            lv = ctx["lv"]()
        except ValueError as e:                              # base_used_rows: V's set does not reproduce block set
            v_check = [str(e)]
        kc = Path(V_SPEC.kc_input_detail)
        src = ctx["fut_source"]()
        fut_src = {k: src[k] for k in ("n_units", "n_pairs", "missing", "bad_sha")}
        rf = reuse_files(s)
        facts = dict(
            v_anc=(ctx["git_facts"](W_SPEC.v_summary, [c for _, c in s.v_commits]) or {}).get("ancestors"),
            aa_anc=(ctx["git_facts"](s.aa_summary, [c for _, c in s.aa_blocks]) or {}).get("ancestors"),
            kc_sha=sha256_file(kc) if kc.exists() else None, w_digest=w_digest, aa_doc=d["aa_learning"], lv=lv,
            v_check=v_check, keys=ctx["keys"](),
            want_keys=dict(w_measure_key=Y_SPEC.w_measure_key, u_measure_key=Y_SPEC.u_measure_key),
            decl_sha=ctx["decl_sha"](rf, s.decl_commit), now_sha=files_sha(rf), fut_src=fut_src)
        why = ab_rules.reuse_reasons(facts, s)
        body = ab_rules.reuse_stop(why, self._counts(doc)) if why else dict(
            outcome=ab_rules.PASS, reasons=[],
            lv_odour=dict(odours=lv["odours"], sha256=lv["sha256"], n=lv["n"], per_source=lv["per_source"]),
            lv_keys=lv["keys"])
        body = dict(body, fut_source=fut_src, w_digest=w_digest, kc_sha256=facts["kc_sha"], decl_commit=s.decl_commit,
                    n_reuse_files=len(rf),
                    note="AB.2 · AB.7 0b: V 블록 다섯 · AA 블록 아홉(결과 · 커밋), kc_input.json sha256, W 주 세트 "
                         "digest, AA trained 31 · lv_odour 출처(108냄새, 목록과 sha256 — 0c는 이 목록만 쓴다), W · U "
                         "측정 키, t/v/w/y/aa · z_split 불변, 가망 관문 원천(AA S1 16쌍 384단위, 캐시 sha256).")
        ab_store.write_json(self._detail("reuse"), body, self.plist)
        self._finish("reuse", t0)
        return self._write("reuse", body)

    # ================================================================ 0c: generate (AB.3, AB.7 0c)
    def stage_generate(self) -> dict:
        doc = self._require("generate")
        s, t0 = self.s, time.perf_counter()
        lv = doc["reuse"]["lv_odour"]["odours"]                      # 0b's list only (6647)
        st = self.ctx["selftest"]()
        gen = self.ctx["generate"](lv)
        why = ab_pairs.declared_reasons(gen, s)
        if not st["ok"]:
            why.append(f"생성기 자가 시험: digest {st['digest_keys']} ≠ {s.w_digest_keys}")
        rec = {k: gen[k] for k in gen if k not in ("rows",)}
        p = ab_store.write_json(self._detail("generate"), dict(rec, selftest=st), self.plist)
        self._finish("generate", t0)
        if why:
            # AB.7 0c (해석 9): the code follows AB.3's wording (its tests pass), so a difference is a declared-value
            # defect → STOP_SET_MISMATCH〈0c〉. A code defect is found by the order-0 tests before this point; then
            # nothing is written (refuse, exit 2) and the fix goes through restart_preseal.
            if self.ctx.get("refuse_generate_mismatch"):
                refuse(f"stage generate: declared values differ: {why[:3]}")
            body = ab_rules.set_mismatch_stop("0c", "; ".join(why[:3]), self._counts(doc))
        else:
            body = dict(outcome=ab_rules.PASS, reasons=[])
        return self._write("generate", dict(body, **rec, selftest=st, n_reasons=len(why),
                                            detail_path=self._detail("generate"), detail_sha256=sha256_file(p),
                                            note="AB.3 · AB.7 0c: 측정 없음, 봉인 앞 — 선언값과 대조(선언 형식 계수, "
                                                 "plan Reading 2); 순서대로 센 계수는 skipped_inorder에 기록."))

    # ================================================================ 0d: seal_code (AB.7 0d)
    def stage_seal_code(self) -> dict:
        self._require("seal_code")
        t0 = time.perf_counter()
        s = self.s
        vec = stream_vectors(s)
        want = [[list(tags), list(v)] for tags, v in s.stream_vectors]
        if json.loads(json.dumps(vec)) != json.loads(json.dumps(want)):
            refuse(f"stage seal_code: the stream test vectors differ from ab_spec.stream_vectors: {vec} ≠ {want}")
        seal = seal_now(s)
        return self._write("seal_code", dict(
            outcome=ab_rules.PASS, reasons=[], seal=seal, stream_vectors=vec,
            note="AB.7 0d: 2세대 첫 측정 전 봉인 — ab_estimate(가망 관문 futility_inputs · futility · futility_grid 포함) · "
                 "ab_rules · ab_pairs + import 닫힘 + ab_spec 수치(0f 모형 · 회수 · 문턱 · 격자 포함) + 흐름 시험 벡터 다섯."),
            core_s=time.perf_counter() - t0)

    # ================================================================ restart_preseal (AB.7 0d, plan Reading 15)
    def restart_preseal(self, note_path) -> dict:
        """Before the first 2nd-gen measurement only: results/ab → results/ab.invalid-<n>, the archive root →
        <archive_root>.invalid-<n>, the summary copied into results/ab.invalid-<n>/ and then rewritten to
        {restart_preseal_<k> blocks, budget} with block restart_preseal_<n> = {symptom, clause, cause, old_seal,
        invalidated (every earlier stage block, outcome INVALID), at} — the core ledger carried, so spent time still
        counts — or removed when HEAD does not track it and no block exists. Refuses (exit 2) when the working or HEAD
        summary holds kc_input or any later block (or a defect_<n> / reseal_<n>), a STOP block (AB.9 closure rule),
        the raw cache has a file, or the note lacks a non-empty symptom / clause / cause."""
        s = self.s
        try:
            note = json.loads(Path(note_path).read_text()) if note_path else None
        except (OSError, ValueError) as e:
            refuse(f"restart_preseal: cannot read the note {note_path}: {e}")
        need = ("symptom", "clause", "cause")
        bad = [k for k in need if not isinstance(note, dict) or not note.get(k)]
        if bad:
            refuse(f"restart_preseal: the note needs non-empty {list(need)}; missing {bad}")
        doc = self._doc()
        head = _head_summary(self.summary_path)
        try:
            hdoc = json.loads(head) if head else {}
        except ValueError:
            refuse(f"restart_preseal: HEAD's {self.summary_path} is not JSON")
        order = list(s.stages)
        post = set(order[order.index(s.first_measure):])
        late = sorted({k for d in (doc, hdoc) for k in d
                       if _base(k) in post or re.fullmatch(r"(defect|reseal)_\d+", k)})
        if late:
            refuse(f"restart_preseal: block(s) {late} exist — after the first measurement a fix goes through the "
                   "defect procedure (AB.7 0d), never a restart")
        stops = sorted({f"{k} {v.get('outcome')}" for d in (doc, hdoc) for k, v in d.items()
                        if k in order and isinstance(v, dict)
                        and v.get("outcome") not in (ab_rules.PASS, ab_rules.INVALID)})
        if stops:
            refuse(f"restart_preseal: {stops} — a STOP is final (AB.9 닫힘 규칙)")
        cache = Path(s.cache_dir)
        if cache.exists() and any(p.is_file() for p in cache.rglob("*")):
            refuse(f"restart_preseal: {cache} holds 2nd-gen raw; the first measurement happened")
        summ = Path(self.summary_path)
        if not (Path(s.raw_dir).exists() or Path(os.path.expanduser(s.archive_root)).exists() or summ.exists()):
            refuse("restart_preseal: nothing to restart (no results, archive or summary)")
        n = ab_store.invalid_slot(s.raw_dir, s.archive_root)
        raw = ab_store.move_aside(s.raw_dir, n)
        arch = ab_store.move_aside(s.archive_root, n, archive=True)
        inv = f"{s.raw_dir}.invalid-{n}"
        copy = str(ab_store.write_invalid(s.raw_dir, n, summ.name, summ.read_bytes())) if summ.exists() else None
        blocks = {b: (doc.get(b) or hdoc.get(b)) for b in order if b in doc or b in hdoc}
        invalidated = {b: dict(v, outcome_was=v.get("outcome"), outcome=ab_rules.INVALID) for b, v in blocks.items()}
        old_seal = ((blocks.get("seal_code") or {}).get("seal") or {}).get("key")
        name = f"restart_preseal_{n}"
        rblk = dict(outcome=ab_rules.PASS, reasons=[], n=n, **{k: note[k] for k in need}, old_seal=old_seal,
                    invalidated=invalidated, at=_now(), invalid_dir=inv, raw=raw, archive=arch, summary_copy=copy)
        if head is None and not blocks:
            action = "removed (untracked at HEAD, no block)" if ab_store.remove_summary(self.summary_path, self.plist) \
                else "absent"
        else:
            prev = {k: v for d in (hdoc, doc) for k, v in d.items() if re.fullmatch(r"restart_preseal_\d+", k)}
            new = dict(prev, **{name: ab_store.to_json(rblk)})
            budget = doc.get("budget") or hdoc.get("budget")
            if budget:
                new["budget"] = budget
            ab_store.write_json(self.summary_path, new, self.plist)
            action = "rewritten to the restart_preseal blocks + budget (commit it before stage0)"
        entry = dict(n=n, at=rblk["at"], invalid_dir=inv, raw=raw, archive=arch, summary_copy=copy, summary=action,
                     invalidated=sorted(invalidated), old_seal=old_seal)
        note_p = ab_store.write_invalid(s.raw_dir, n, "restart_preseal.json",
                                        (canonical(ab_store.to_json(dict(entry, note=note))) + "\n").encode())
        return dict(outcome=ab_rules.PASS, reasons=[], restart=entry, note_path=str(note_p),
                    sentence=f"사전 측정 재시작 {n}: 블록 {sorted(invalidated)} INVALID, {inv}로 옮김, 요약 {action}; "
                             "다음은 §2 시험 로그 → stage0.")


    # ================================================================ budget terms (AB.7 예산, plan Reading 18)
    def _bench_s(self, doc: dict) -> float:
        """The sealed code's per-replicate cost (0a `bench`, g 7, seconds)."""
        return float(doc["stage0"]["bench"]["max_s"])

    def _rest_terms(self, doc: dict) -> dict:
        """The remaining core estimate of orders 3–9 (AB.7 0e "남은 핵심 추정"; the same terms before 0e and before
        0f), hours, from the AB.7 table's prior unit costs: KC (the declared new odours + the KC repro odours), the
        fixed 0a / smoke share, the oracle on the KC-after maximum, the screen on the lenient 40, 7b at k = smoke_k
        (bench × smoke_k / k of the g 7 bench structure), learn at k = smoke_k, the verdict."""
        s, h = self.s, self.s.s_per_h
        n_kc = dict(s.decl_odours)["new"] + len(s.kc_repro)
        n_or = sum(dict(s.decl_post_kc_max).values())
        k_bench = sum(s.rep_sizes("g7"))                         # ab_estimate.bench runs the g 7 structure
        cal7b = (len(s.cells) * (s.n_sel + s.n_ver) * self._bench_s(doc) * s.smoke_k / k_bench / self._workers() / h)
        out = dict(kc=n_kc * s.prior_kc_s_per_odour / h, fixed=float(s.prior_fixed_h),
                   oracle=n_or * s.prior_oracle_s_per_row / h, screen=s.smoke_lenient * s.prior_screen_s_per_pair / h,
                   calibrate=cal7b, learn=s.smoke_k * s.prior_learn_s_per_pair / h, verdict=float(s.prior_verdict_h))
        return dict(out, total=sum(out.values()))

    def _rest_core_h(self, doc: dict) -> float:
        return self._rest_terms(doc)["total"]

    def _fut_est_h(self, doc: dict) -> float:
        """AB.7 0f "0f 추정은 bench 단가 × 회수": rows × fut_reps × bench / workers (the record grid is not in it)."""
        s = self.s
        return len(s.fut_tags) * len(s.fut_allocs) * s.fut_reps * self._bench_s(doc) / self._workers() / s.s_per_h

    def _cal_est(self, doc: dict) -> dict:
        """The 0e reservation terms (AB.7 0e 6818@248cd25): 0e (structures × cells × (n_sel + n_ver) × bench /
        workers), the synthetic validation (configs × cells × synth_reps × statistics × bench / workers), the 0f
        estimate, the remaining core; total = their sum (hours)."""
        s, b, w, h = self.s, self._bench_s(doc), self._workers(), self.s.s_per_h
        cal = len(s.rep_structures) * len(s.cells) * (s.n_sel + s.n_ver) * b / w / h
        syn = (len(ab_estimate.synth_configs(s)) * len(s.synth_cells) * s.synth_reps * len(ab_estimate.STATS) * b
               / w / h)
        fut, rest = self._fut_est_h(doc), self._rest_terms(doc)
        return dict(cal=cal, synth=syn, futility=fut, rest=rest["total"], rest_terms=rest,
                    total=cal + syn + fut + rest["total"])

    def _rec_spent_h(self, doc: dict, name: str) -> float:
        """The records ledger so far + this stage's records progress, hours (aa_runner.Runner's form)."""
        prev = sum(float(e.get("wall_s") or 0.0) for e in (doc.get("budget") or {}).get("records_ledger", []))
        return (prev + self._prog(name)["records"]) / self.s.s_per_h

    def _spent_core_h(self, doc: dict, stage: str) -> float:
        """The measured core: the ledger so far + this stage's core progress (seconds of earlier killed runs too)."""
        return self._core_h(doc) + self._prog(stage)["core"] / self.s.s_per_h

    # ================================================================ 0e: cal_gate (AB.5, AB.7 0e)
    def stage_cal_gate(self) -> dict:
        """AB.7 0e 측정 전 보정 관문: the seal check (in _require), the reservation (core ledger + 2.0 × (0e + synthetic
        validation + 0f + the remaining core) ≤ 24 h, else STOP_BUDGET〈0e 전〉), AB.5 on the representative structures
        g 5 · 6 · 7 through the pooled checkpointing executor (the measured cap between cells: STOP_BUDGET〈0e〉, the
        checkpoints stay), STOP_CALIBRATION〈0e〉 when any structure × statistic has no verified PASS-side α, the
        synthetic validation (record) when g 7's three PASS-side α are verified. No Gen-2 value is read or measured."""
        doc = self._require("cal_gate")
        s = self.s
        tick = self._ticker("cal_gate")
        est = self._cal_est(doc)
        core = self._core_h(doc)
        cnt = self._counts(doc)
        if not ab_rules.core_ok(core, est["total"], s):
            tick()
            return self._write("cal_gate", dict(
                ab_rules.budget_stop(s.stage_label("cal_gate") + " 전", ab_rules.budget_text(core, est["total"], s),
                                     "2세대 측정 없음", cnt), estimate_h=est))
        items = [(ab_estimate.rep_structure(sizes, s), tag) for tag, sizes in s.rep_structures]
        ex = CalExec(self, "cal_gate", s.cal_dir, tick=tick, cap=True)
        synth, synth_reason = None, None
        try:
            out = ab_estimate.calibrate_many(items, s, ex)
            summ = {o["tag"]: ab_store.to_json(o) for o in out}
            g7 = summ.get(s.synth_struct)
            if g7 is not None and g7["pass_ok"]:
                synth = self._synth(next(st for st, t in items if t == s.synth_struct), g7, ex)
            else:
                synth_reason = (f"{s.synth_struct}의 PASS 쪽 α가 셋 모두 검증되지 않아 합성 검증을 하지 않음"
                                "(AB.5, plan Reading 10)")
        except _BudgetStop:
            tick()
            spent = self._spent_core_h(self._doc(), "cal_gate")
            return self._write("cal_gate", dict(
                ab_rules.budget_stop(s.stage_label("cal_gate"), ab_rules.spent_text(spent, s),
                                     "2세대 측정 없음", cnt),
                estimate_h=est, checkpoints=s.cal_dir))
        for tag, sm in summ.items():
            sm["sizes"] = list(s.rep_sizes(tag))
            sm["fail_flag"] = None if sm["fail_ok"] else "FAIL 쪽 미도달 위험"
            sm["streams"] = {ph: ["cal", tag, ph] for ph in ("truth", "sel", "ver")}
        fails = [(tag, stat) for tag, _sz in s.rep_structures for stat in ab_estimate.STATS
                 if summ[tag]["alpha"][stat]["P"] is None]
        why = [self._cal_failure(summ[tag], stat) for tag, stat in fails]
        p = ab_store.write_json(self._detail("cal_gate"), dict(structures=summ, synth=synth), self.plist)
        common = dict(
            structures=summ, synth=synth, synth_reason=synth_reason, estimate_h=est,
            roots=dict(cal=s.cal_seed, synth=s.synth_seed), seal_key=doc["seal_code"]["seal"]["key"],
            checkpoints=s.cal_dir, detail_path=self._detail("cal_gate"), detail_sha256=sha256_file(p),
            note="AB.5 · AB.7 0e: 측정 전 보정 관문 — 대표 구조 g 5 · 6 · 7, 칸 24, 선택 · 검증 흐름 분리, CP 97.5% 상한; "
                 "구조 · 통계 하나라도 PASS 쪽 α 미검증이면 STOP_CALIBRATION. FAIL 쪽 미도달은 기록(fail_flag). "
                 "합성 검증은 기록 — 방법 · 수치를 바꾸지 않는다. 2세대 측정 없음.")
        tick()
        if not why:
            return self._write("cal_gate", dict(common, outcome=ab_rules.PASS, reasons=[]))
        first = why[0]
        tag = first["tag"]
        stop = ab_rules.calibration_stop(tag, s.rep_sizes(tag), first["stat"], first["phase"], first["cell"],
                                         first["cp"], first["x"], first["n"], first["alpha"], cnt)
        stop["reasons"] = [f"{w['tag']} {w['stat']} {w['phase']} 칸 {w['cell']}" for w in why]
        return self._write("cal_gate", dict(common, **stop, failures=why))

    def _cal_failure(self, sm: dict, stat: str) -> dict:
        """The STOP_CALIBRATION slots of one (structure, statistic): "sel" when no grid level passes the selection
        (the grid end p_grid[-1], the cell with the largest CP bound there — ties: the lowest cell number), else "ver"
        (the chosen α's worst verification cell)."""
        s = self.s
        base = dict(tag=sm["tag"], stat=stat)
        if sm["chosen"][stat]["P"] is None:
            i = len(s.p_grid) - 1
            cps = {int(ci): float(v[i]) for ci, v in sm["sel_cp"][stat]["P"].items()}
            w = max(sorted(cps), key=lambda c: cps[c])
            return dict(base, phase="sel", cell=w, cp=cps[w], x=int(sm["sel_counts"][str(w)][stat]["P"][i]),
                        n=int(sm["n_sel"]), alpha=None)
        v = sm["verified"][stat]["P"]["worst"]
        return dict(base, phase="ver", cell=int(v["cell"]), cp=float(v["cp"]), x=int(v["x"]), n=int(sm["n_ver"]),
                    alpha=sm["chosen"][stat]["P"])

    def _synth(self, st, g7: dict, ex) -> dict:
        """AB.5 합성 검증 (record): configs × synth_cells, synth_reps each on stream ("synth", config, ci) under the
        synth root, g 7's verified levels (α_F only when reached)."""
        s = self.s
        a = g7["alpha"]
        alpha = dict(D=a["D"]["P"], F=a["D"]["F"], Dfin=a["Dfin"]["P"], R=a["R"]["P"])
        jobs, idx = [], []
        for name, eff in ab_estimate.synth_configs(s):
            for ci in s.synth_cells:
                jobs.append(("synth", (st, name, int(ci), tuple(eff), alpha, s)))
                idx.append((name, int(ci)))
        got = ex.many(jobs)
        out = {}
        for (name, ci), v in zip(idx, got):
            out.setdefault(name, {})[str(ci)] = v
        return dict(structure=s.synth_struct, alpha=alpha, reps=s.synth_reps, cells=list(s.synth_cells),
                    configs={n: list(e) for n, e in ab_estimate.synth_configs(s)}, results=out,
                    stream=["synth", "<config>", "<cell>"], label="기록 — 방법 · 수치를 바꾸지 않는다")

    # ================================================================ 0f: futility (AB.7 0f, AB.9.3)
    def stage_futility(self) -> dict:
        """AB.7 0f 가망 관문: no Gen-2 value is read or measured. The reservation (core ledger + 2.0 × (0f + the
        remaining core) ≤ 24 h, else STOP_BUDGET〈0f 전〉); the AA source must equal 0b's facts and reproduce AA bit for
        bit (else exit 2: a sealed-code defect → restart_preseal); the judgement row (fut_judge) decides on the point
        P̂; the record rows, CP intervals and blocked shares are records; the grid runs only on STOP_FUTILE (records
        ledger, the records cap before each scenario). No measured cap during 0f (AB.7 예산)."""
        doc = self._require("futility")                       # sealed stage: the seal check first (exit 7)
        s = self.s
        tick = self._ticker("futility")
        cal = self._eff(doc, "cal_gate")["structures"]
        core = self._core_h(doc)
        fut_h, rest = self._fut_est_h(doc), self._rest_terms(doc)
        est = dict(futility=fut_h, rest=rest["total"], rest_terms=rest, total=fut_h + rest["total"])
        if not ab_rules.core_ok(core, est["total"], s):
            tick()
            return self._write("futility", dict(
                ab_rules.budget_stop(s.stage_label("futility") + " 전", ab_rules.budget_text(core, est["total"], s),
                                     "2세대 측정 없음", self._counts(doc)), estimate_h=est))
        src = self.ctx["fut_source"]()
        facts = {k: src[k] for k in ("n_units", "n_pairs", "missing", "bad_sha")}
        if not _same(facts, doc["reuse"]["fut_source"]):
            tick()
            refuse(f"stage futility: the AA source differs from the reuse block ({facts}) — the user's case")
        try:
            arr = ab_estimate.futility_inputs(src["by_pair"], src["keys"], src["z"], s)
        except (KeyError, ValueError) as e:
            tick()
            refuse(f"stage futility: the AA source cannot be read through the judgement path ({type(e).__name__}: "
                   f"{e}) — a sealed-code defect (AB.7 0f): restart_preseal")
        why = ab_estimate.fut_check(arr, src["pairs"], src["s1"], s)
        if why:
            tick()
            refuse(f"stage futility: the AA source is not bit-equal to AA ({why[:3]}) — a sealed-code defect "
                   "(AB.7 0f): restart_preseal")
        inp = ab_estimate.fut_model(arr, src["s1"], s)
        alpha = {t: cal[t]["alpha"] for t in s.fut_tags}
        out = ab_estimate.futility(inp, alpha, s, CalExec(self, "futility", s.fut_dir, tick=tick, cap=False))
        rows, j = out["rows"], out["judge"]
        grid = None
        if j["futile"]:
            tick()                                            # the core part ends here; the grid is records
            rtick = self._ticker("futility", "records")
            try:
                grid = ab_estimate.futility_grid(
                    inp, {t: cal[t]["sel_counts"] for t in s.fut_tags}, s,
                    CalExec(self, "futility", s.fut_dir, tick=rtick, which="records", cap=False),
                    stop=lambda: round(self._rec_spent_h(self._doc(), "futility") - s.records_cap_h,
                                       s.round_digits) >= 0)
            finally:
                rtick()
            tick = self._ticker("futility")                   # a fresh core mark: the grid's seconds stay records
        model = {k: inp[k] for k in ("keys", "mu", "tau", "tau_src", "rho", "C", "x_sizes")}
        p = ab_store.write_json(s.fut_detail, dict(model=model, rows=rows, judge=j, grid=grid), self.plist)
        icc, pair, group = s.fut_allocs
        common = dict(
            source=dict(facts, keys=src["keys"]), model=model,
            alpha={t: ab_estimate.fut_alpha(alpha[t]) for t in s.fut_tags}, rows=rows, judge=j,
            grid=None if grid is None else {
                sc: dict(pareto=v["pareto"], cells=len(v["cells"]), reason=v.get("reason"),
                         done=sum(c.get("p_hat") is not None for c in v["cells"])) for sc, v in grid.items()},
            grid_reason=None if j["futile"] else "STOP_FUTILE이 아니므로 계산하지 않음(AB.7 0f)",
            streams=dict(rows=["fut", "<tag>", "<alloc>"], grid=["fut", "grid", "<scenario>", "<g>", "<k>"],
                         root=s.synth_seed),
            estimate_h=est, seal_key=doc["seal_code"]["seal"]["key"], checkpoints=s.fut_dir,
            detail_path=s.fut_detail, detail_sha256=sha256_file(p),
            note="AB.7 0f: 측정 없음 — 2세대 세트를 읽지도 재지도 않는다. 판단 = g 6 · ρ 배분 행의 점 추정 P̂ "
                 "(CP · 기록 행 · 막힘 몫은 기록). 기록 격자는 STOP_FUTILE일 때만(records 원장), AB를 바꾸지 않는다.")
        tick()
        if not j["futile"]:
            return self._write("futility", dict(common, outcome=ab_rules.PASS, reasons=[]))
        jr, g6 = rows[j["row"]], alpha[s.fut_judge[0]]
        fill = dict(aD=g6["D"]["P"], aF=g6["Dfin"]["P"], aR=g6["R"]["P"], p_hat=jr["p_hat"], x=jr["passed"], n=jr["n"],
                    cp=jr["cp95"], g5=rows[f"{s.fut_tags[0]}|{icc}"]["p_hat"],
                    g7=rows[f"{s.fut_tags[-1]}|{icc}"]["p_hat"],
                    pair=[rows[f"{t}|{pair}"]["p_hat"] for t in s.fut_tags],
                    group=[rows[f"{t}|{group}"]["p_hat"] for t in s.fut_tags], worst=jr["worst"],
                    pareto={sc: v["pareto"] for sc, v in grid.items()})
        return self._write("futility", dict(common, **ab_rules.futile_stop(fill, self._counts(doc))))


# ================================================================ the pooled checkpointing cell executor (Reading 9)
class _BudgetStop(Exception):
    """The measured core cap was reached between cells (AB.7 0e); the checkpoints written so far stay."""


def _synth_run(st, config: str, ci: int, eff: tuple, alpha: dict, s) -> dict:
    """AB.5 합성 검증, one (config, cell): synth_reps × ab_estimate.synth_rep on stream ("synth", config, ci) under the
    synth root; the label counts and P(PASS) · P(FAIL)."""
    r = aa_estimate.stream(s.synth_seed, "synth", config, int(ci))
    cnt = {lab: 0 for lab in (ab_rules.PASS, ab_rules.FAIL, ab_rules.UNDECIDED)}
    for _ in range(s.synth_reps):
        cnt[ab_estimate.synth_rep(st, s.cell(ci), eff, alpha, r, s)] += 1
    n = int(s.synth_reps)
    return dict(counts=cnt, n=n, p_pass=cnt[ab_rules.PASS] / n, p_fail=cnt[ab_rules.FAIL] / n)


def _cal_job(kind: str, args: tuple, path: str, key: str, plist) -> dict:
    """One calibration / synthetic / futility job (module level: a spawn worker pickles it by name). Resumes from its
    checkpoint when the key matches; checkpoints after every chunk (ab_store atomic write)."""
    p = Path(path)
    try:
        old = json.loads(p.read_text()) if p.exists() else None
    except ValueError:
        old = None
    if old is not None and old.get("key") == key and old.get("final"):
        return old["value"]
    t = time.perf_counter()
    if kind in ("truth", "synth"):
        v = ab_estimate.truth(*args) if kind == "truth" else _synth_run(*args)
        ab_store.write_json(path, dict(key=key, final=True, value=v, wall_s=time.perf_counter() - t), plist)
        return ab_store.to_json(v)
    resume = old["value"] if old is not None and old.get("key") == key else None

    def save(pay):
        ab_store.write_json(path, dict(key=key, final=pay["done"] >= pay["n"], value=pay,
                                       wall_s=time.perf_counter() - t), plist)
    if kind == "fut":                                         # 0f rows (6 args) and grid cells (11 args)
        a = tuple(args) + (None,) * (len(_FUT_ARGS) - len(args))
        v = ab_estimate.fut_run(*a[:_FUT_ARGS.index("resume")], resume=resume, on_chunk=save,
                                B=a[_FUT_ARGS.index("B")], stream=a[_FUT_ARGS.index("stream")])
    elif kind in ("sel", "ver"):
        a = tuple(args) + (None,) * (len(_CAL_ARGS) - len(args))
        v = ab_estimate.cal_run(*a[:_CAL_ARGS.index("resume")], resume=resume, on_chunk=save,
                                B=a[_CAL_ARGS.index("B")], only=a[_CAL_ARGS.index("only")])
    else:
        raise ValueError(f"unknown job kind {kind!r}")
    if v["done"] < v["n"] or not p.exists():                  # n = 0 never calls on_chunk
        save(v)
    return ab_store.to_json(v)


_CAL_ARGS = ("st", "tag", "phase", "ci", "tru", "s", "n", "resume", "on_chunk", "B", "only")      # ab_estimate.cal_run
_FUT_ARGS = ("st", "tag", "alloc", "inp", "alpha", "s", "n", "resume", "on_chunk", "B", "stream")  # ab_estimate.fut_run


def _cal_star(item):
    i, a = item
    return i, _cal_job(*a)


def job_path(root: str, kind: str, args: tuple) -> str:
    """Calibration <root>/<tag>/<kind>_<ci>.json; synthetic <root>/synth/<config>_<ci>.json; futility
    <root>/<"_".join(stream)>.json with stream = args[10] or ("fut", tag, alloc)."""
    if kind == "fut":
        a = tuple(args) + (None,) * (len(_FUT_ARGS) - len(args))
        stream = a[_FUT_ARGS.index("stream")] or ("fut", a[_FUT_ARGS.index("tag")], a[_FUT_ARGS.index("alloc")])
        return f"{root}/{_safe('_'.join(str(x) for x in stream))}.json"
    if kind == "synth":
        return f"{root}/synth/{_safe(str(args[1]))}_{int(args[2])}.json"
    if kind == "truth":
        return f"{root}/{_safe(str(args[1]))}/truth_{int(args[2])}.json"
    return f"{root}/{_safe(str(args[1]))}/{kind}_{int(args[3])}.json"


def job_key(kind: str, args: tuple, seal_key: str) -> str:
    """sha256 of canonical(kind, the args without any ABSpec, the seal key)."""
    a = [x for x in args if not isinstance(x, ABSpec)]
    return hashlib.sha256(canonical(ab_store.to_json(dict(kind=kind, args=a, seal=seal_key))).encode()).hexdigest()


class CalExec:
    """The cell executor of ab_estimate.calibrate_many / futility / futility_grid: __call__(kind, args) and
    many(jobs). Jobs whose checkpoint is final are read back; the rest run in a spawn process pool of the runner's
    workers (in process when one worker or one job), each checkpointing after every chunk. After every finished job
    the stage ticker `tick` adds the seconds to ledger `which`; with cap=True the measured core (ledger + this stage's
    progress) is checked before the first job and after every job — past core_cap_h the pool is terminated and
    _BudgetStop raised (the checkpoints stay)."""

    def __init__(self, runner, stage: str, root: str, tick=None, which: str = "core", cap: bool = True):
        self.r, self.stage, self.root, self.which, self.cap = runner, stage, root, which, cap
        self.tick = tick or runner._ticker(stage, which)
        self.seal_key = seal_now(runner.s)["key"]
        self.ran = 0                                          # jobs computed (not read back) by this executor

    def __call__(self, kind: str, args: tuple):
        return self.many([(kind, args)])[0]

    def _check(self) -> None:
        self.tick()
        if self.cap and not ab_rules.spent_ok(self.r._spent_core_h(self.r._doc(), self.stage), self.r.s):
            raise _BudgetStop()

    def many(self, jobs: list) -> list:
        r = self.r
        out, todo = [None] * len(jobs), []
        for i, (kind, args) in enumerate(jobs):
            path, key = job_path(self.root, kind, args), job_key(kind, args, self.seal_key)
            p = Path(path)
            try:
                old = json.loads(p.read_text()) if p.exists() else None
            except ValueError:
                old = None
            if old is not None and old.get("key") == key and old.get("final"):
                out[i] = old["value"]
            else:
                todo.append((i, (kind, args, path, key, r.plist)))
        self._check()
        if not todo:
            return out
        n_done = len(jobs) - len(todo)
        workers = r._workers()
        if workers == 1 or len(todo) == 1:
            for i, a in todo:
                out[i] = _cal_job(*a)
                self.ran += 1
                n_done += 1
                _log(f"ab {self.stage}: {n_done}/{len(jobs)} cells")
                self._check()
            return out
        for v in r.s.thread_env:
            os.environ.setdefault(v, "1")
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            try:
                for i, v in pool.imap_unordered(_cal_star, todo):
                    out[i] = v
                    self.ran += 1
                    n_done += 1
                    _log(f"ab {self.stage}: {n_done}/{len(jobs)} cells")
                    self._check()
            except BaseException:
                pool.terminate()
                raise
        return out
