"""FlyPool worker jobs of spec appendix N (N.8.3, N.3 / N.8.5, N.4 / N.8.7). Signature fn(engine, plasticity, pops,
comps, readout, **kwargs), module-level so the spawn pool can pickle them; the worker's default engine only lends its
connectome.

The rig (Engine, Plasticity, compartments) is cached per worker under (Params, edit), edit in EDITS:
- "none";
- "apl_to_kc_zero": every APL -> KC edge 0, N.7's primary block;
- "apl_all_zero": every APL out-edge 0, the record arm, value-equal to Params(apl_scale=0).
The edit is made in place on a freshly built engine's CSC (the graded-APL out-edge views follow:
h3_jobs.apply_csc_edit's precedent). The sha256 of the weight vector goes with every row. No existing module changes
(N.8.7).

presentation_job: one presentation per seed (reset, clear_drive, present, settle, read), plasticity off. It returns
scalars only:
- KC active fraction and spikes of the read window;
- the largest 200 ms sliding-window KC count over the whole presentation, in Hz (h4_jobs._present_kc's window);
- the readout A / P read counts, the APL output per step and the wall time.
It never returns a KC vector (N.8.3 blinding). kc_vectors_job adds the fired KC positions; N0 calls it with APL on, and
N2's judge only after its verdict."""
from __future__ import annotations

import hashlib
import time
from collections import deque

import numpy as np

from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .h4_formula import dprime, dv
from .h4_jobs import _present_kc, type_cells
from .n_spec import train_seed
from .plasticity import Plasticity
from .presentation import decide
from .stimuli import present

EDITS = ("none", "apl_to_kc_zero", "apl_all_zero")
PRESENTATION_KEYS = ("seed", "A", "P", "kc_frac", "kc_spikes", "kc_max_win_hz", "apl_out_per_step", "wall_s", "steps")
_RIG: dict = {}          # (Params, edit) -> (Engine, Plasticity, comps, csc sha256); one entry per worker


def apply_edit(eng, pops, edit: str) -> str:
    """Zero the edit's APL out-edges in the engine's CSC in place; the sha256 of the weight vector."""
    if edit not in EDITS:
        raise ValueError(f"unknown edit {edit!r}; known: {EDITS}")
    if edit != "none":
        is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
        m = is_apl[edge_sources(eng.csc)]
        if edit == "apl_to_kc_zero":
            is_kc = np.zeros(eng.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
            m &= is_kc[eng.csc.tgt.astype(np.int64)]
        eng.csc.w[m] = np.float32(0.0)
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


def rig(conn, pops, params, edit: str):
    """The worker's rig for (params, edit), built once and reused while jobs keep asking for it (one entry: a switch of
    arm rebuilds, it never edits a cached engine)."""
    key = (params, edit)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha = apply_edit(eng, pops, edit)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha)
    return _RIG[key]


def readout_cells(conn, readout: dict) -> tuple:
    cells = type_cells(conn, (readout["A"], readout["P"]))
    return cells[readout["A"]], cells[readout["P"]]


def _present(e, p, pops, odor, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells) -> tuple:
    """One presentation (presentation._fresh, then settle + read steps, as decide and _present_kc step them)."""
    kc = pops.kc
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    apl = np.asarray(pops.apl, np.int64)
    graded = e.p.apl_mode == "graded"
    t0 = time.perf_counter()
    e.reset(int(seed)); p.reset_traces(); e.clear_drive(); p.quiet_dan()
    present(e, pops, odor, strength)
    win = np.zeros(len(kc), np.int32); hist = deque(); max_win = 0
    read = np.zeros(len(kc), np.int32); counts = np.zeros(e.N, np.int32); apl_out = 0.0
    n_settle, n_total = int(round(settle_ms / e.p.dt)), int(round((settle_ms + read_ms) / e.p.dt))
    for step in range(n_total):
        fired = e.step()
        f = pos[fired]; f = f[f >= 0]
        win[f] += 1; hist.append(f)
        if len(hist) > window_ms:
            win[hist.popleft()] -= 1
        if f.size:
            max_win = max(max_win, int(win.max()))
        if step >= n_settle:
            read[f] += 1
            counts[fired] += 1
            if graded:                                  # reading the membrane does not change the engine state
                apl_out += float(e.apl_release(e.v[apl]).sum())
    n_read = n_total - n_settle
    if not graded:                                      # spiking APL: its read-window spikes
        apl_out = float(counts[apl].sum())
    row = dict(seed=int(seed), A=int(counts[a_cells].sum()), P=int(counts[p_cells].sum()),
               kc_frac=float((read > 0).mean()), kc_spikes=int(read.sum()),
               kc_max_win_hz=max_win * 1000.0 / (window_ms * e.p.dt),
               apl_out_per_step=apl_out / (n_read * max(apl.size, 1)), wall_s=time.perf_counter() - t0,
               steps=int(n_total))
    return row, read


def _rows(eng, pops, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms, keep_kc: bool):
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = readout_cells(e.conn, readout)
    p.reset_weights()
    was = p.enabled
    p.set_enabled(False)
    out = []
    try:
        for s in seeds:
            row, read = _present(e, p, pops, odor, s, strength, settle_ms, read_ms, window_ms, a_cells, p_cells)
            row = dict(row, edit=edit, csc_sha256=sha)
            if keep_kc:
                row.update(n_kc=int(len(pops.kc)), kc_fired=np.flatnonzero(read > 0).astype(int).tolist())
            out.append(row)
    finally:
        p.set_enabled(was)
    return out


def presentation_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                     settle_ms: float, read_ms: float, window_ms: int) -> list:
    """N0f's presentations: scalars only, never a KC vector (N.8.3)."""
    return _rows(eng, pops, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms, False)


def kc_vectors_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                   settle_ms: float, read_ms: float, window_ms: int) -> list:
    """The same presentations with the fired KC positions of the read window (N0 with APL on; the judge after its
    verdict)."""
    return _rows(eng, pops, params, edit, odor, seeds, readout, strength, settle_ms, read_ms, window_ms, True)


# ================================================================ N1: the punish-only oracle (N.3, N.8.5)
def pick_alpha(changes: dict, alphas) -> float:
    """The punishment alpha with the smallest change (the most negative d'), ties -> the smaller alpha."""
    if any(changes.get(str(a)) is None for a in alphas):
        raise ValueError("an undefined change (fewer than 2 select seeds)")
    return min(alphas, key=lambda a: (changes[str(a)], a))


def punish_only_oracle_job(eng, pl, pops, comps, ro, params, odor_x: dict, odor_y: dict, readout: dict, z: dict,
                           act_seeds, select_seeds, report_seeds, alphas, strength: float, settle_ms: float,
                           read_ms: float, window_ms: int, punish_type: str) -> dict:
    """h4_jobs.oracle_job's punishment half:
    - KC activity f of X (and Y, for the Jaccard record) on act_seeds;
    - the punish core's KC -> MBON weights set to w0 (1 - alpha f) for X's KCs, alpha picked on select_seeds by the
      change d'(dV_edit - dV_pre) alone (no reward edit; pick_alpha);
    - pre and P on report_seeds.
    APL on only (N.3 blinding)."""
    e, p, c, sha = rig(eng.conn, pops, params, "none")
    a_cells, p_cells = readout_cells(e.conn, readout)
    idx = np.concatenate([a_cells, p_cells]); n_a = len(a_cells)
    try:
        p.reset_weights(); p.set_enabled(False)

        def activity(odor):
            fired = np.zeros(len(pops.kc)); frac, spikes, max_win = [], [], []
            for s in act_seeds:
                o = _present_kc(e, p, pops, odor, int(s), strength, settle_ms, read_ms, window_ms)
                fired += o["read"] > 0; frac.append(float((o["read"] > 0).mean())); spikes.append(int(o["read"].sum()))
                max_win.append(o["max_win"])
            return fired / len(act_seeds), {"frac": frac, "spikes": spikes, "max_win": max_win}

        def probe(seeds):
            A, Pc = [], []
            for s in seeds:
                cnt = decide(e, p, pops, [odor_x, odor_y], strength, int(s), settle_ms, read_ms, idx=idx)
                A.append(cnt[:, :n_a].sum(1).tolist()); Pc.append(cnt[:, n_a:].sum(1).tolist())
            return {"A": A, "P": Pc}

        fx, kc_x = activity(odor_x)
        fy, kc_y = activity(odor_y)
        pun = np.isin(p.post_mb, p.mb_local[c[punish_type].core])
        w = e.csc.w

        def set_w(a_p):
            wv = p.w0.copy()
            if a_p is not None:
                wv[pun] = p.w0[pun] * (1.0 - a_p * fx)[p.pre_kc[pun]]
            w[p.edges] = wv

        set_w(None); pre_sel = probe(select_seeds)
        punish = {}
        for a in alphas:
            set_w(a); r = probe(select_seeds)
            punish[str(a)] = {"P": r, "change": dprime(dv(r, z) - dv(pre_sel, z))}
        a_p = pick_alpha({k: v["change"] for k, v in punish.items()}, alphas)
        set_w(None); pre = probe(report_seeds)
        set_w(a_p); post = probe(report_seeds)
        w[p.edges] = p.w0
        jac = float(((fx > 0) & (fy > 0)).sum() / max(((fx > 0) | (fy > 0)).sum(), 1))
        return {"alpha_punish": a_p, "select": {"pre": pre_sel, "punish": punish}, "report": {"pre": pre, "P": post},
                "kc": {"x": kc_x, "y": kc_y, "jaccard": jac}, "csc_sha256": sha}
    finally:
        p.reset_weights(); p.set_enabled(True)


# ================================================================ N2: absolute punishment conditioning (N.4, N.8.7)
def train_plus_only(e, p, pops, cs_plus, strength: float, seed: int, punish: str, trials: int, present_ms: float,
                    gap_ms: float, settle_ms: float, seed_base: int, seed_stride: int) -> None:
    """conditioning.train_block's CS+ half alone (no CS- during training). Per trial:
    - reset to train_seed(seed, trial);
    - settle with the weights frozen;
    - the punishment DAN for present_ms;
    - the gap and one recovery step."""
    for t in range(int(trials)):
        e.reset(train_seed(seed, t, seed_base, seed_stride))
        p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, cs_plus, strength)
        was = p.enabled
        p.set_enabled(False); e.run(settle_ms); p.set_enabled(was)
        p.drive_dan(punish, e.p.dan_drive_mv)
        e.run(present_ms)
        p.quiet_dan(); e.clear_drive(); e.run(gap_ms)
        p.recover_pulse()


def absolute_arm_job(eng, pl, pops, comps, ro, params, edit: str, odor_x: dict, odor_y: dict, seed: int,
                     plastic: bool, readout: dict, punish_type: str, strength: float, settle_ms: float, read_ms: float,
                     window_ms: int, trials: int, present_ms: float, gap_ms: float, train_settle_ms: float,
                     seed_base: int, seed_stride: int) -> dict:
    """One (condition, seed) of N2, in the edit's rig for the whole arm (pre-test, training and post-test: N.4):
    - probes of X and Y at `seed`, plasticity off (PRESENTATION_KEYS rows: A, P, KC fraction and spikes, KC sub-window;
      the per-presentation state P < SPEC.state_p_min is read off each row's P, N.8.8);
    - training of X alone with the punishment DAN;
    - the probes again.
    With plastic False the training runs with the weights frozen (the plumbing check, N.8.7). Weights are restored."""
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = readout_cells(e.conn, readout)
    t0 = time.perf_counter()

    def probes():
        was = p.enabled
        p.set_enabled(False)
        try:
            return {k: _present(e, p, pops, o, seed, strength, settle_ms, read_ms, window_ms, a_cells, p_cells)[0]
                    for k, o in (("x", odor_x), ("y", odor_y))}
        finally:
            p.set_enabled(was)

    try:
        p.reset_weights()
        pre = probes()
        p.set_enabled(bool(plastic))
        train_plus_only(e, p, pops, odor_x, strength, int(seed), punish_type, trials, present_ms, gap_ms,
                        train_settle_ms, seed_base, seed_stride)
        p.set_enabled(True)
        post = probes()
        wf = float(p.weights_frac())
    finally:
        p.reset_weights(); p.set_enabled(True)
    return dict(seed=int(seed), edit=edit, plastic=bool(plastic), csc_sha256=sha, pre=pre, post=post,
                weights_frac=wf, wall_s=time.perf_counter() - t0)
