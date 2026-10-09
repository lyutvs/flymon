"""Load logging around AD's re-bench and run (spec AD.4). Every minute a sample: uptime's three load averages and every
process above busy_cpu % CPU (ps -A -o pid=,ppid=,pcpu=,command=) with its ancestor pid chain, so a bench's own
processes (the bench python, its spawn workers, its Showdown node) are recognised by ancestry. A bench is invalid if a
process outside it is busy in busy_samples (3) consecutive samples inside the bench window - with one sample a minute
this calls a 2-to-4-minute burst invalid, the conservative side of AD.4's "3 minutes". A bench window with no sample
is invalid (the logger was not running). The start condition is the last 3 samples free of busy processes (any
process, not only python, AD.4). During the run the samples are recorded; the result carries load1 median / max and
the outside offenders (not a STOP)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

UPTIME = re.compile(r"load averages?:\s*([\d.]+),?\s+([\d.]+),?\s+([\d.]+)")


def parse_uptime(text: str):
    m = UPTIME.search(text or "")
    return [float(m.group(i)) for i in (1, 2, 3)] if m else None


def parse_ps(text: str) -> list:
    rows = []
    for line in (text or "").splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            rows.append(dict(pid=int(parts[0]), ppid=int(parts[1]), cpu=float(parts[2]), cmd=parts[3].strip()))
        except ValueError:
            continue
    return rows


def ancestors(rows, pid: int) -> list:
    parent = {r["pid"]: r["ppid"] for r in rows}
    out, cur = [], parent.get(int(pid))
    while cur and cur not in out:
        out.append(cur)
        cur = parent.get(cur)
    return out


def sample(uptime_text: str, ps_text: str, t: float, busy_cpu: float) -> dict:
    rows = parse_ps(ps_text)
    busy = [dict(r, cmd=r["cmd"][:160], ancestors=ancestors(rows, r["pid"])) for r in rows if r["cpu"] > busy_cpu]
    return dict(kind="sample", t=float(t), load=parse_uptime(uptime_text), uptime=(uptime_text or "").strip(),
                busy=busy)


def take(busy_cpu: float) -> dict:
    import time
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout
    ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pcpu=,command="], capture_output=True, text=True).stdout
    return sample(up, ps, time.time(), busy_cpu)


def read(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") in ("sample", "wait"):
            out.append(rec)
    return sorted(out, key=lambda r: r["t"])


def window(samples, start: float, end: float) -> list:
    return [x for x in samples if float(start) <= x["t"] <= float(end)]


def _own(proc: dict, own_pids) -> bool:
    own = {int(p) for p in own_pids}
    return proc["pid"] in own or bool(own & set(proc.get("ancestors", [])))


def offenders(samples, own_pids, busy_samples: int) -> list:
    runs: dict = {}
    best: dict = {}
    for x in samples:
        now = {b["pid"]: b for b in x["busy"] if not _own(b, own_pids)}
        for pid in list(runs):
            if pid not in now:
                runs.pop(pid)
        for pid, b in now.items():
            r = runs.setdefault(pid, dict(pid=pid, cmd=b["cmd"], n_consecutive=0, first_t=x["t"], last_t=x["t"]))
            r["n_consecutive"] += 1
            r["last_t"] = x["t"]
            if r["n_consecutive"] > best.get(pid, {}).get("n_consecutive", 0):
                best[pid] = dict(r)
    return [best[pid] for pid in sorted(best) if best[pid]["n_consecutive"] >= int(busy_samples)]


def bench_validity(samples, start: float, end: float, own_pid: int, busy_samples: int) -> dict:
    w = window(samples, start, end)
    if not w:
        return dict(valid=False, reason="no load samples in the bench window", offenders=[], n_samples=0)
    off = offenders(w, {own_pid}, busy_samples)
    return dict(valid=not off, reason=None if not off else "a process outside the bench above the CPU limit",
                offenders=off, n_samples=len(w))


def quiet(samples, exclude_pids, busy_samples: int) -> bool:
    last = list(samples)[-int(busy_samples):]
    return len(last) == int(busy_samples) and all(not [b for b in x["busy"] if not _own(b, exclude_pids)] for x in last)


def summary(samples, start: float, end: float, own_pid: int, busy_samples: int) -> dict:
    w = window(samples, start, end)
    l1 = [x["load"][0] for x in w if x.get("load")]
    return dict(n_samples=len(w), load1_median=float(np.median(l1)) if l1 else None,
                load1_max=float(max(l1)) if l1 else None, offenders=offenders(w, {own_pid}, busy_samples))
