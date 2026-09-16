"""M2 measurement A (scratch, throwaway): candidate-odour MBON response map.

Provisional encoder = spec 3.3 as written (channel per receptor type, drive equalised by 1/receptor
count, presentation strength 0.35). Nothing here is a design decision; it exists to answer, on the
REAL candidate odours rather than the M0 designed pair:

  (a) D.6: any KC above 150 Hz for a 200 ms sub-window of the presentation
  (b) D.6: within-turn ratio of candidate KC spike counts
  (c) D.6: candidate-odour KC sparsity against the 3% floor
  (d) per-MBON spike counts for all MBONs, per candidate
  (e) MBON13 / MBON05 / MBON11 / MBON19 in particular
  (f) whether the candidate ordering an MBON gives is stable over paired-noise seeds
  (g) resting and odour-evoked firing of every DAN type (can PPL102/104/106 carry a phasic pulse?)

Run: uv run python <this file> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from poke_env.battle import Move
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from flymon.battle.pool import POOL
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import present

NPZ = "data/malecns.npz"
EXCLUDE = ("ORN_DA1", "ORN_V")          # spec 3.3: cVA and CO2 channels are not free to reassign
SUB_MS = 200.0                          # D.6 (a) asks for "above 150 Hz for 200 ms"
SETTLE_MS, READ_MS = 800.0, 600.0       # presentation.decide's window
SAT_HZ = 150.0
POWER_EDGES = (60, 90)                  # spec 3.3: <60, 60-89, >=90
HP_EDGES = (0.34, 0.67)                 # three HP bins


# ---- provisional encoder (spec 3.3) ------------------------------------------------------------
def pool_vocabulary():
    """The channel vocabulary the real pool actually needs: species types, move types, bins."""
    gd = GenData.from_gen(1)
    species_types, move_info = {}, {}
    for m in POOL:
        species_types[m.species] = tuple(t.upper() for t in gd.pokedex[to_id_str(m.species)]["types"])
        for a in m.attacks:
            mv = Move(to_id_str(a), gen=1)
            move_info[a] = (mv.type.name, int(mv.base_power))
    mon_types = sorted({t for v in species_types.values() for t in v})
    move_types = sorted({v[0] for v in move_info.values()})
    return species_types, move_info, mon_types, move_types


def assign_channels(pops: Populations, groups: list) -> dict:
    """Map every channel to one receptor type.

    Provisional and deterministic: drop DA1/V, sort the rest by receptor count, trim the extreme
    ends symmetrically down to the number of channels needed, then give each channel GROUP a set of
    positions spread evenly over that sorted range (smallest group first, nearest free position).
    Spreading each group over the whole range matters: within a turn the candidates differ only in
    the move-type and power channels, so those must not all land on tiny or huge glomeruli.
    """
    cand = sorted((t for t in pops.receptor_types if t not in EXCLUDE),
                  key=lambda t: (len(pops.receptor_types[t]), str(t)))
    need = sum(len(names) for _, names in groups)
    if need > 45:
        raise ValueError(f"spec 3.3 allows at most 45 channels, this vocabulary needs {need}")
    if need > len(cand):
        raise ValueError(f"need {need} receptor types, have {len(cand)}")
    drop = len(cand) - need
    lo = drop // 2
    band = cand[lo:lo + need]
    free = set(range(need))
    out = {}
    for gname, names in sorted(groups, key=lambda g: (len(g[1]), g[0])):
        q = len(names)
        for j, nm in enumerate(names):
            ideal = (j + 0.5) * need / q
            pos = min(free, key=lambda p: (abs(p + 0.5 - ideal), p))
            free.discard(pos)
            out[f"{gname}:{nm}"] = str(band[pos])
    return out


def channel_odor(pops: Populations, channels: dict, keys: list) -> dict:
    """Spec 3.3 drive equalisation: strength proportional to 1/receptor count, mean 1."""
    types = [channels[k] for k in keys]
    if len(set(types)) != len(types):
        raise ValueError(f"channel collision in {keys}")
    inv = np.array([1.0 / len(pops.receptor_types[t]) for t in types])
    inv = inv / inv.mean()
    return {t: float(s) for t, s in zip(types, inv)}


def power_bin(bp: int) -> str:
    return "lt60" if bp < POWER_EDGES[0] else ("60to89" if bp < POWER_EDGES[1] else "ge90")


def hp_bin(frac: float) -> str:
    return "low" if frac <= HP_EDGES[0] else ("mid" if frac <= HP_EDGES[1] else "high")


def build_turns(species_types, move_info, n_turns: int) -> list:
    """Deterministic turns: every pool member takes a turn as the active mon against a rotating
    opponent, with HP bins cycled so the shared channels are not constant across turns."""
    hps = [(0.9, 0.9), (0.9, 0.3), (0.5, 0.6), (0.2, 0.8), (0.6, 0.2), (0.3, 0.5)]
    turns = []
    for i in range(n_turns):
        me, opp = POOL[i % len(POOL)], POOL[(i * 5 + 3) % len(POOL)]
        my_hp, op_hp = hps[i % len(hps)]
        cands = [{"move": a, "type": move_info[a][0], "bp": move_info[a][1]} for a in me.attacks][:4]
        turns.append({"turn": i, "me": me.species, "opp": opp.species,
                      "my_types": list(species_types[me.species]), "opp_types": list(species_types[opp.species]),
                      "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands})
    return turns


def turn_odors(pops: Populations, channels: dict, turn: dict) -> list:
    shared = ([f"my:{t}" for t in turn["my_types"]] + [f"opp:{t}" for t in turn["opp_types"]]
              + [f"myhp:{hp_bin(turn['my_hp'])}", f"opphp:{hp_bin(turn['opp_hp'])}"])
    out = []
    for c in turn["candidates"]:
        keys = shared + [f"move:{c['type']}", f"pow:{power_bin(c['bp'])}"]
        out.append(channel_odor(pops, channels, keys))
    return out


# ---- worker jobs (module level so spawn can pickle them) ---------------------------------------
def candidate_map_job(eng, pl, pops, comps, ro, odors, seed: int, strength: float,
                      settle_ms: float = SETTLE_MS, read_ms: float = READ_MS, sub_ms: float = SUB_MS) -> dict:
    """One turn, one paired-noise seed: present every candidate from the same reset seed and report
    the MBON counts of the read window plus KC statistics over 200 ms sub-windows of the WHOLE
    presentation (settle included - the M0c clique ignited during the odour, not only in the read
    window). Plasticity off, weights reset, so the map is the naive brain."""
    pl.reset_weights()
    pl.set_enabled(False)
    n_sub_settle = int(round(settle_ms / sub_ms))
    n_sub_read = int(round(read_ms / sub_ms))
    kc, mbon = pops.kc, pops.mbon
    dan_idx = {name: cells for name, cells in pops.dan_types.items()}
    sec_sub, sec_read = sub_ms / 1000.0, read_ms / 1000.0
    cands = []
    for odor in odors:
        eng.reset(seed)
        pl.reset_traces()
        eng.clear_drive()
        pl.quiet_dan()
        present(eng, pops, odor, strength)
        sub_max_kc_hz = 0.0
        n_kc_sub_over_sat = 0
        offenders = {}
        read_counts = np.zeros(eng.conn.N, np.int64)
        for w in range(n_sub_settle + n_sub_read):
            c = eng.run(sub_ms)
            if w >= n_sub_settle:
                read_counts += c
            khz = c[kc] / sec_sub
            sub_max_kc_hz = max(sub_max_kc_hz, float(khz.max()) if khz.size else 0.0)
            over = np.flatnonzero(khz > SAT_HZ)
            n_kc_sub_over_sat = max(n_kc_sub_over_sat, int(over.size))
            for j in over.tolist():
                offenders[int(kc[j])] = max(offenders.get(int(kc[j]), 0.0), float(khz[j]))
        kc_read_hz = read_counts[kc] / sec_read
        cands.append({
            "mbon_counts": read_counts[mbon].astype(int).tolist(),
            "kc_active_frac": float((kc_read_hz > 0).mean()),
            "kc_spikes": int(read_counts[kc].sum()),
            "kc_max_hz_read": float(kc_read_hz.max()) if kc_read_hz.size else 0.0,
            "kc_max_hz_sub": sub_max_kc_hz,
            "n_kc_sub_over_150": n_kc_sub_over_sat,
            "n_kc_read_over_100": int((kc_read_hz > 100.0).sum()),
            "kc_offenders": {str(k): v for k, v in sorted(offenders.items(), key=lambda kv: -kv[1])[:10]},
            "dan_hz": {name: float(read_counts[cells].sum() / len(cells) / sec_read)
                       for name, cells in dan_idx.items()},
        })
    pl.set_enabled(True)
    return {"seed": seed, "candidates": cands}


def dan_rest_job(eng, pl, pops, comps, ro, seed: int, ms: float = 3000.0) -> dict:
    """Resting rate of every DAN type and of the MBONs we care about (spec A.4 rejected PPL101 at
    ~119 Hz; the question is whether any other PPL1 type is quiet enough to carry a phasic pulse)."""
    pl.reset_weights()
    pl.set_enabled(False)
    eng.reset(seed)
    eng.clear_drive()
    pl.quiet_dan()
    counts = eng.run(ms)
    pl.set_enabled(True)
    sec = ms / 1000.0
    t = eng.conn.type
    mb = {}
    for m in ("MBON13", "MBON18", "MBON05", "MBON21", "MBON11", "MBON19"):
        cells = np.flatnonzero(t == m)
        if cells.size:
            mb[m] = float(counts[cells].sum() / len(cells) / sec)
    return {"seed": seed,
            "dan_hz": {name: float(counts[cells].sum() / len(cells) / sec)
                       for name, cells in pops.dan_types.items()},
            "mbon_hz": mb}


# ---- driver -------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.35)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    groups = [("my", mon_types), ("opp", mon_types), ("move", move_types),
              ("pow", ["lt60", "60to89", "ge90"]), ("myhp", ["low", "mid", "high"]),
              ("opphp", ["low", "mid", "high"])]
    channels = assign_channels(pops, groups)
    turns = build_turns(species_types, move_info, a.turns)
    seeds = [a.seed_start + i for i in range(a.seeds)]
    mbon_types = conn.type[pops.mbon].astype(str).tolist()

    meta = {
        "encoder": "provisional, spec 3.3 (per-receptor-type channels, 1/n equalisation, strength "
                   f"{a.strength}); channel->receptor assignment is deterministic and throwaway",
        "n_channels": len(channels), "channels": channels,
        "channel_receptor_counts": {k: int(len(pops.receptor_types[v])) for k, v in channels.items()},
        "mon_types": mon_types, "move_types": move_types,
        "turns": turns, "seeds": seeds,
        "params": {"kc_kc_scale": Params().kc_kc_scale, "kc_thresh": Params().kc_thresh,
                   "strength": a.strength, "settle_ms": SETTLE_MS, "read_ms": READ_MS, "sub_ms": SUB_MS},
        "glomeruli_per_candidate": sorted({len(turn_odors(pops, channels, t)[0]) for t in turns}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"channels {len(channels)}  turns {len(turns)}  seeds {len(seeds)}  "
          f"glomeruli/candidate {meta['glomeruli_per_candidate']}")

    jobs, index = [], []
    for t in turns:
        odors = turn_odors(pops, channels, t)
        for s in seeds:
            jobs.append(dict(odors=odors, seed=s, strength=a.strength))
            index.append((t["turn"], s))
    del conn

    t0 = time.perf_counter()
    with FlyPool(NPZ, Params(), [{} for _ in range(a.workers)], workers=a.workers) as pool:
        print(f"pool up in {time.perf_counter() - t0:.0f}s; {len(jobs)} candidate-map jobs")
        t1 = time.perf_counter()
        res = pool.run_jobs(candidate_map_job, jobs)
        print(f"candidate map {time.perf_counter() - t1:.0f}s")
        t1 = time.perf_counter()
        rest = pool.run_jobs(dan_rest_job, [dict(seed=s) for s in seeds])
        print(f"dan rest {time.perf_counter() - t1:.0f}s")

    (out / "raw.json").write_text(json.dumps(
        {"index": index, "results": res, "rest": rest, "mbon_types": mbon_types}, indent=1))
    print(f"wrote {out/'raw.json'}  total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
