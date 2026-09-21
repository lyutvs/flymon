"""Provenance of h3_recorded.json: the values the committed M0d diagnostics recorded for the adopted operating point
(spec H.3a.7). Run once from the repository root; it reads the git-ignored run outputs
results/m0d/diag/{candidate_feasibility,align_inputs}.json (candidate_feasibility.py 90c36f9, align_inputs.py 19188ab)
and writes the fixture the H.3 rule tests replay:

    uv run python tests/brain/fixtures/make_h3_recorded.py tests/brain/fixtures/h3_recorded.json
"""
import json
import sys
from pathlib import Path

MAIN = Path("results/m0d/diag")
OUT = Path(sys.argv[1])

feas = json.loads((MAIN / "candidate_feasibility.json").read_text())
align = json.loads((MAIN / "align_inputs.json").read_text())

b16 = [b for b in feas["step1"]["bisections"] if b["kc_thresh"] == 1.6][0]
pts = [[float(k), v["median_mv"]] for k, v in b16["endpoints"].items()] + [[s["scale"], s["median_mv"]] for s in b16["trace"]]
acc = b16["accepted"]
ref_rows = [dict(odor=p["odor"], seed=p["seed"], apl_v_mean=p["apl_v_mean"], kc_active_frac=p["kc_active_frac"],
                 release_frac=p["release_frac"]) for p in acc["presentations"]]

row3 = [r for r in feas["step3"]["rows"] if r["label"].startswith("G(1.6")][0]
hpts = [[float(k), v["hz"]] for k, v in row3["cal"]["endpoints"].items()] + [[s["hold"], s["hz"]] for s in row3["cal"]["trace"]]

g = align["item1"]["configs"]["G(1.6, 0.1186)"]
rows16 = g["per_seed"]
row4 = [r for r in feas["step4"]["rows"] if r["label"].startswith("G(1.6")][0]
types = ["MBON13", "MBON18", "MBON05", "MBON21"]
stim = [dict(odor=p["odor"], seed=p["seed"], types={n: p["types"][n] for n in types}) for p in row4["guard"]["stim"]]
rest = [dict(seed=p["seed"], types={n: p["types"][n] for n in types}) for p in row4["guard"]["rest"]]
pooled = align["item2"]["configs"]["G(1.6, 0.1186)"]["pooled"]

fx = dict(
    source=dict(candidate_feasibility=feas["meta"]["commit"], align_inputs=align["meta"].get("commit"),
                note="values recorded by the committed diagnostics (spec H.3a.7); regenerate only from those runs"),
    stage1_kc1_6=dict(points=pts, accepted_x=b16["accepted_scale"], accepted_mv=b16["median_mv"],
                      final_bracket=b16["final_bracket"]),
    reference_kc1_6_hold085=dict(rows=ref_rows, median_mv=acc["median_mv"], median_kc_pct=acc["median_kc_pct"],
                                 q1_mv=acc["q1_mv"], q3_mv=acc["q3_mv"], median_release_frac=acc["median_release_frac"],
                                 rel_share_in=acc["rel_share_in"]),
    hold_G1_6=dict(points=hpts, accepted=row3["cal"]["hold"]),
    d4=dict(rows=rows16, by8=dict((k, g["by_count"]["8"][k]) for k in ("pct_A", "pct_B", "margin_pp", "overlap_margin")),
            ci8=g["by_count"]["8"]["boot"]["ci"], ci16=g["by_count"]["16"]["boot"]["ci"],
            c0_rows=align["item1"]["configs"]["C0"]["per_seed"], c0_ci8=align["item1"]["configs"]["C0"]["by_count"]["8"]["boot"]["ci"]),
    baseline_gate32=dict(per_seed=pooled["per_seed"], mean=pooled["mean_hz"], se=pooled["se_hz"]),
    guard=dict(pools=feas["pools"], stim=stim, rest=rest,
               expected={n: {k: row4["guard"]["types"][n][k] for k in ("median_delta", "zero_share", "passes")} for n in types},
               pass_A=row4["pass_A"], pass_P=row4["pass_P"]),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(fx, indent=1) + "\n")
print(OUT, OUT.stat().st_size, "bytes")
