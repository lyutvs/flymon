"""M0d H.3 reproduction check (plan step R2): the runner's C1 at kc_thresh 1.6 against the committed diagnostics.

H.3a.7 says the H.3 run re-derives the adopted operating point rather than discovering it. This compares one runner
report (`scripts/run_m0d_h3.py --combos C1 --kc-cells 1.6`, records on) with the values the diagnostics recorded —
tests/brain/fixtures/h3_recorded.json (candidate_feasibility.py 90c36f9, align_inputs.py 19188ab) and the git-ignored
results/m0d/diag/{slim_inputs,candidate_feasibility}.json — every value bit for bit. Exit 0 only when all match.

    uv run python docs/superpowers/specs/m0d-diag/h3_reproduction_check.py results/m0d/h3/runs/<run id>.json
"""
import json
import sys
from pathlib import Path

FIXTURE = Path("tests/brain/fixtures/h3_recorded.json")
SLIM = Path("results/m0d/diag/slim_inputs.json")
FEAS = Path("results/m0d/diag/candidate_feasibility.json")


def main(report_path) -> int:
    rep = json.loads(Path(report_path).read_text())
    fx = json.loads(FIXTURE.read_text())
    slim = json.loads(SLIM.read_text())
    gc_rec = json.loads(FEAS.read_text())["step5"]["gc"]
    bad = []

    def check(name, got, want):
        ok = got == want
        print(f"{'OK ' if ok else 'BAD'} {name}: got {got!r}" + ("" if ok else f" want {want!r}"))
        if not ok:
            bad.append(name)

    cell = [c for c in rep["combos"]["C1"]["cells"] if c["kc"] == 1.6][0]
    s = cell["search"]
    check("stage-1 trace (s, median mV)",
          [[e["x"], e["median_mv"]] for e in s["endpoints"]] + [[t["x"], t["median_mv"]] for t in s["trace"]],
          fx["stage1_kc1_6"]["points"])
    check("stage-1 accepted s", s["accepted"]["x"], fx["stage1_kc1_6"]["accepted_x"])
    h = cell["stage2"]
    check("stage-2 trace (hold, Hz)",
          [[e["x"], e["mean_hz"]] for e in h["endpoints"]] + [[t["x"], t["mean_hz"]] for t in h["trace"]],
          fx["hold_G1_6"]["points"])
    check("stage-2 hold", h["accepted"], fx["hold_G1_6"]["accepted"])
    st3 = cell["stage3"]
    ref = slim["item3"]["reference"]
    check("stage-3 reference medians", [st3["reference"]["median_mv"], st3["reference"]["median_kc_pct"]],
          [ref["median_mv"], ref["median_kc_pct"]])
    check("D.4 per-seed overlap margins (100-107)", st3["d4"]["overlap_margin_per_seed"],
          [r["chance"] - r["jaccard"] for r in fx["d4"]["rows"][:8]])
    check("D.4 pct A/B", [st3["d4"]["pct_A"], st3["d4"]["pct_B"]], [fx["d4"]["by8"]["pct_A"], fx["d4"]["by8"]["pct_B"]])
    check("D.4 overlap CI (8)", st3["overlap"]["ci8"]["ci"], fx["d4"]["ci8"])
    for pool, n in (("A", "MBON13"), ("P", "MBON05")):
        t = st3["guard"][pool]["types"][n]
        check(f"guard {n}", [t["median_delta"], t["zero_share"]],
              [fx["guard"]["expected"][n]["median_delta"], fx["guard"]["expected"][n]["zero_share"]])
    check("guard passing", [st3["guard"]["A"]["passing"], st3["guard"]["P"]["passing"]],
          [fx["guard"]["pass_A"], fx["guard"]["pass_P"]])
    b = cell["stage4"]["baseline"]
    check("gate baseline per seed (116-147)", b["per_seed"], fx["baseline_gate32"]["per_seed"])
    check("gate baseline mean / SE", [b["mean_hz"], b["se_hz"]], [fx["baseline_gate32"]["mean"], fx["baseline_gate32"]["se"]])
    run = cell["stage4"]["runaway"]
    check("runaway (rest 8, odour B 64, all zero)", [run["ok"], len(run["rest_kc_over_per_seed"]),
                                                     len(run["odor_kc_over_per_seed"])], [True, 8, 64])
    check("C1 status", [cell["status"], rep["combos"]["C1"]["status"]], ["adopted", "adopted"])

    rec = rep["records"]["C1"]
    gc = rec["gain_control"]
    check("gain control R-bar, B apl_r_max", [gc["r_bar"], gc["r_max_b"]], [gc_rec["r_bar"], gc_rec["r_max_b"]])
    check("gain control release A / B", [gc["release_a"], gc["release_b"]], [gc_rec["release_a"], gc_rec["release_b"]])
    metrics = {mm["name"]: mm for mm in gc_rec["metrics"]}
    iqr = metrics["odour-to-odour IQR of KC active %"]
    check("gain control IQR A / B / diff / CI", [gc["iqr_a"], gc["iqr_b"], gc["iqr_diff"]["diff"], gc["iqr_diff"]["ci"]],
          [iqr["a"], iqr["b"], iqr["diff"], iqr["ci"]])
    lv = metrics["all51 log10 variance of per-glomerulus KC drive"]
    a51 = rec["all51"]
    check("all51 log10 variance A / B / diff / CI",
          [a51["log10_var_a"], a51["log10_var_b"], a51["log10_var_diff"]["diff"], a51["log10_var_diff"]["ci"]],
          [lv["a"], lv["b"], lv["diff"], lv["ci"]])
    zc = metrics["all51 zero-drive glomerulus count"]
    check("all51 zero-drive count A / B", [a51["zero_count_a"], a51["zero_count_b"]], [zc["a"], zc["b"]])
    print("ALL MATCH" if not bad else f"MISMATCH: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
