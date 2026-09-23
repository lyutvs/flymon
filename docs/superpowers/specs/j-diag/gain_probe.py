"""J.10 pre-amendment probe (scratch, throwaway): on the C3 adopted engine, present each all51 glomerulus alone at
ORN gain g and record realized receptor rate, ALPN spikes, KC spikes and active KCs. Answers (1) whether a
receptor-stage gain can reach the KC-drive equalization J.2 declares, (2) whether weak glomeruli are weak at the
PN rate or downstream of it. Not an oracle run: no pairs, no readouts, no learning."""
import json
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[4].as_posix()
sys.path.insert(0, ROOT)

GAINS = (0.25, 1.0, 2.0, 4.0, 16.0)
SEEDS = (200, 201)
STRENGTH, SETTLE_MS, READ_STEPS = 0.35, 800.0, 600
_W = {}


def params_from_json(d):
    from flymon.brain.config import Params
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def init():
    import os
    os.chdir(ROOT)
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.engine_cpu import Engine
    conn = Connectome.load("data/malecns.npz")
    pops = Populations.from_connectome(conn)
    s = json.load(open("results/summary/m0d.json"))
    p = params_from_json(s["h3"]["combos"]["C3"]["adopted"]["params"])
    _W.update(pops=pops, eng=Engine(conn, pops, p, seed=0))


def job(item):
    from flymon.brain.stimuli import present
    t, g, seed, c_norm = item
    e, pops = _W["eng"], _W["pops"]
    e.reset(int(seed)); e.clear_drive()
    present(e, pops, {t: c_norm / len(pops.receptor_types[t]) * g}, STRENGTH)
    cmd_hz = float(e.drive_hz[pops.receptor_types[t]][0])
    e.run(SETTLE_MS)
    counts = np.zeros(e.N, np.int32)
    for _ in range(READ_STEPS):
        counts[e.step()] += 1
    rec = np.asarray(pops.receptor_types[t], np.int64)
    kc = counts[pops.kc]
    return dict(g_type=t, gain=g, seed=int(seed), cmd_hz=cmd_hz,
                orn_hz=float(counts[rec].sum() / rec.size / (READ_STEPS / 1000.0)),
                pn=int(counts[pops.alpn].sum()), kc=int(kc.sum()), kc_on=int((kc > 0).sum()))


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h3_spec import all51_glomeruli
    pops = Populations.from_connectome(Connectome.load("data/malecns.npz"))
    gloms, c_norm = all51_glomeruli(pops)
    items = [(t, g, s, c_norm) for g in GAINS for t in gloms for s in SEEDS]
    print(f"{len(gloms)} glomeruli, c_norm {c_norm}, {len(items)} presentations", flush=True)
    t0, rows = time.time(), []
    with get_context("spawn").Pool(12, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(job, items, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    out = sys.argv[1]
    json.dump(dict(gains=GAINS, seeds=SEEDS, strength=STRENGTH, c_norm=c_norm, rows=rows), open(out, "w"))
    print(f"done {time.time() - t0:.0f}s -> {out}", flush=True)
