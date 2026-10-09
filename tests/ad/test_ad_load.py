"""Load logging (AD.4): uptime / ps parsing, the ancestry of busy processes, a bench invalidated by an outside process
above 50% CPU for 3 consecutive samples, the quiet start condition, the run summary, and write_ad_budget's statuses."""
import importlib.util
import json
from pathlib import Path

from flymon.ad import load, store

ROOT = Path(__file__).resolve().parents[2]
MAC = "10:00  up 1 day, 10:13, 32 users, load averages: 4.01 5.33 5.80"
PS = """    1     0   0.0 /sbin/launchd
   10     1  10.0 python scripts/run_ad.py --bench 1
   11    10 180.0 python -c spawn_main
   12    10  60.0 node pokemon-showdown
   99     1  95.0 /usr/bin/other-job
"""


def s(t, busy):
    return dict(kind="sample", t=float(t), load=[1.0 + t, 1.0, 1.0], uptime="", busy=busy)


def p(pid, anc=(1,), cpu=90.0):
    return dict(pid=pid, ppid=anc[0] if anc else 0, cpu=cpu, cmd=f"p{pid}", ancestors=list(anc))


def test_parse_uptime_and_ps():
    assert load.parse_uptime(MAC) == [4.01, 5.33, 5.80]
    assert load.parse_uptime(" 10:00:00 up 3 days,  load average: 0.10, 0.20, 0.30") == [0.10, 0.20, 0.30]
    rows = load.parse_ps(PS)
    assert rows[2] == dict(pid=11, ppid=10, cpu=180.0, cmd="python -c spawn_main")
    assert load.ancestors(rows, 11) == [10, 1]


def test_sample_lists_busy_processes_with_ancestry():
    x = load.sample(MAC, PS, 100.0, 50.0)
    assert x["load"] == [4.01, 5.33, 5.80] and [b["pid"] for b in x["busy"]] == [11, 12, 99]
    assert x["busy"][0]["ancestors"] == [10, 1]


def test_outside_process_three_consecutive_samples_invalidates():
    run = [s(0, []), s(60, [p(99)]), s(120, [p(99)]), s(180, [p(99)]), s(240, [])]
    v = load.bench_validity(run, 0, 300, own_pid=10, busy_samples=3)
    assert not v["valid"] and v["offenders"][0]["pid"] == 99 and v["offenders"][0]["n_consecutive"] == 3
    gap = [s(0, []), s(60, [p(99)]), s(120, [p(99)]), s(180, []), s(240, [p(99)])]
    assert load.bench_validity(gap, 0, 300, own_pid=10, busy_samples=3)["valid"]


def test_the_bench_own_processes_never_count():
    run = [s(t, [p(11, (10, 1), 180.0), p(12, (10, 1))]) for t in (0, 60, 120, 180)]
    assert load.bench_validity(run, 0, 300, own_pid=10, busy_samples=3)["valid"]


def test_only_samples_inside_the_window_count_and_none_is_invalid():
    run = [s(t, [p(99)]) for t in (0, 60, 120)] + [s(400, [])]
    assert load.bench_validity(run, 300, 500, own_pid=10, busy_samples=3)["valid"]
    v = load.bench_validity(run, 1000, 2000, own_pid=10, busy_samples=3)
    assert not v["valid"] and v["reason"] == "no load samples in the bench window"


def test_quiet_needs_three_clean_samples():
    assert not load.quiet([s(0, []), s(60, [])], set(), 3)
    assert load.quiet([s(0, [p(99)]), s(60, []), s(120, []), s(180, [])], set(), 3)
    assert not load.quiet([s(0, []), s(60, [p(99)]), s(120, [])], set(), 3)
    assert load.quiet([s(0, [p(5)]), s(60, [p(5)]), s(120, [p(5)])], {5}, 3)


def test_summary():
    run = [s(0, []), s(60, [p(99)]), s(120, [p(99)]), s(180, [p(99)])]
    x = load.summary(run, 0, 200, own_pid=10, busy_samples=3)
    assert x["load1_median"] == 91.0 and x["load1_max"] == 181.0       # load1 = 1 + t in these samples
    assert x["n_samples"] == 4 and [o["pid"] for o in x["offenders"]] == [99]


def load_script():
    spec = importlib.util.spec_from_file_location("wab", ROOT / "scripts/write_ad_budget.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def bench_dir(tmp_path, n, start, b, e):
    store.write_json(tmp_path / f"results/m4-bench/ad-{n}/bench.json",
                     dict(s_per_batch_battle=b, sit_eval_s_per_eval=e, pid=10, t_started=start, t_ended=start + 500))


def test_write_ad_budget_statuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "load.log"
    lines = [s(t, [p(99)] if 1000 <= t < 1500 else []) for t in range(0, 4000, 60)]
    log.write_text("".join(json.dumps(x) + "\n" for x in lines))
    w = load_script()
    bench_dir(tmp_path, 1, 0, 830.0, 27.0)
    bench_dir(tmp_path, 2, 1000, 2900.0, 90.0)                     # an outside process: invalid
    assert w.main(["--load-log", str(log)]) == 3                   # one valid so far
    bench_dir(tmp_path, 3, 2000, 850.0, 28.0)
    bench_dir(tmp_path, 4, 3000, 840.0, 26.0)
    assert w.main(["--load-log", str(log)]) == 0
    d = json.loads((tmp_path / "results/summary/ad_budget.json").read_text())
    assert d["status"] == "OK" and d["T"] == 120 and d["new_set"] is True and d["bench"]["used"] == [1, 3, 4]
    assert d["bench"]["s_per_batch_battle"] == 840.0 and [a["valid"] for a in d["attempts"]] == [True, False, True, True]
    assert d["label"].startswith("M2 정식 판정 없음") and d["load_log_sha256"]
