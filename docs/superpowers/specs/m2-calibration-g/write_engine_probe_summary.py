"""Spec G.14 summary: derive the verdict from the raw probe files with engine_probe_verdict.py, plus descriptive records
(G.14.6). Refuses a dirty git tree when writing results/summary (G.14.7).

Run from the repo root:
  uv run python docs/superpowers/specs/m2-calibration-g/write_engine_probe_summary.py
Smoke: ... --raw-dir <scratch>/probe --out <scratch>/summary.json --smoke   (no git check, never into results/summary)
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, importlib.util, json, os, subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("engine_probe_verdict", HERE / "engine_probe_verdict.py")
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)

ARMS = ("G", "S-match", "S-ref")
RUNAWAY_SPIKES = 31          # > 150 Hz in 200 ms (G.8), descriptive here


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single_cell_dprimes(report: dict) -> dict:
    """r and p with only the MBON13 term (A) or only the MBON05 term (P) of V."""
    out = {}
    for name, (za, zp) in {"MBON13": (1.0, 0.0), "MBON05": (0.0, 1.0)}.items():
        def dv1(c):
            A = np.asarray(c["A"], float); P = np.asarray(c["P"], float)
            Vv = za * (A - V.Z["A"][0]) / V.Z["A"][1] - zp * (P - V.Z["P"][0]) / V.Z["P"][1]
            return Vv[:, 0] - Vv[:, 1]
        pre, r1, r2 = dv1(report["pre"]), dv1(report["R1"]), dv1(report["R2"])
        out[name] = {"r": V.dprime(r1 - pre), "p": V.dprime(r2 - r1)}
    return out


def describe_arm(rows: list) -> dict:
    fr = np.array([f for r in rows for side in ("x", "y") for f in r["kc"][side]["frac"]])
    mw = np.array([m for r in rows for side in ("x", "y") for m in r["kc"][side]["max_win"]])
    a_rows = [r for r in rows if r["axis"] == "a"]
    ratios = []
    for r in a_rows:
        sx, sy = np.asarray(r["kc"]["x"]["spikes"], float), np.asarray(r["kc"]["y"]["spikes"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.maximum(sx / sy, sy / sx)
        ratios.append(float(np.nanmax(rr)))
    pre_A = np.array([a for r in rows for a in np.asarray(r["report"]["pre"]["A"], float).ravel()])
    pre_P = np.array([p for r in rows for p in np.asarray(r["report"]["pre"]["P"], float).ravel()])
    apl_hz = [h for r in rows for side in ("x", "y") for h in r["kc"][side]["apl_hz"] if h is not None]
    apl_v = [v for r in rows for side in ("x", "y") for v in r["kc"][side]["apl_vmax"] if v is not None]
    return {"kc_frac_median": float(np.median(fr)), "kc_below_3pct": float((fr < 0.03).mean()),
            "kc_in_5_9pct": float(((fr >= 0.05) & (fr <= 0.09)).mean()),
            "max_window_spikes": int(mw.max()), "presentations_over_150hz": float((mw >= RUNAWAY_SPIKES).mean()),
            "same_turn_kc_ratio_max": float(max(ratios)) if ratios else None,
            "same_turn_pairs_ratio_over_2": int(sum(x > 2 for x in ratios)),
            "naive_mbon13_mean": float(pre_A.mean()), "naive_mbon05_mean": float(pre_P.mean()),
            "naive_mbon13_zero_frac": float((pre_A == 0).mean()), "naive_mbon05_zero_frac": float((pre_P == 0).mean()),
            "apl_hz_mean": float(np.mean(apl_hz)) if apl_hz else None, "apl_vmax_max": float(np.max(apl_v)) if apl_v else None,
            "alpha_reward": {str(a): sum(r["alpha_reward"] == a for r in rows) for a in (0.2, 0.5, 0.8)},
            "alpha_punish": {str(a): sum(r["alpha_punish"] == a for r in rows) for a in (0.2, 0.5, 0.8)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/m2/calibration/engine_probe")
    ap.add_argument("--oc", default="results/m2/calibration/engine_probe/oc.json")
    ap.add_argument("--out", default="results/summary/m2_engine_probe.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    raw_dir, out = Path(a.raw_dir), Path(a.out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if a.smoke:
        if out.resolve().parent == Path("results/summary").resolve():
            raise SystemExit("--smoke never writes into results/summary")
    elif dirty:
        raise SystemExit("git tree is dirty: commit first (G.14.7)")
    plan = json.loads((raw_dir / "plan.json").read_text())
    if plan["smoke"] != a.smoke:
        raise SystemExit(f"plan.json smoke={plan['smoke']} but --smoke={a.smoke}")
    match = json.loads((raw_dir / "match.json").read_text())
    arms = {arm: json.loads((raw_dir / f"{arm}.json").read_text()) for arm in ARMS if (raw_dir / f"{arm}.json").exists()}
    oc = json.loads(Path(a.oc).read_text())
    raw_by_arm = {arm: [{k: r[k] for k in ("axis", "turn", "x", "y", "report")} for r in d["rows"]] for arm, d in arms.items()}
    expected = [tuple(p) for p in plan["pairs"]]
    verdict = V.verdict(raw_by_arm, expected, {"diff_pp": match["diff_pp"]}, oc["chosen_test"])
    pairs = {arm: [{"axis": r["axis"], "turn": r["turn"], "x": r["x"], "y": r["y"],
                    "alpha_reward": r["alpha_reward"], "alpha_punish": r["alpha_punish"],
                    "stats": V.pair_stats(r["report"]), "single_cell": single_cell_dprimes(r["report"]),
                    "kc_jaccard": r["kc"]["jaccard"]} for r in d["rows"]] for arm, d in arms.items()}
    summary = {"spec": "G.14", "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
               "git_commit": commit, "git_dirty": dirty, "smoke": a.smoke,
               "code_sha256": {"verdict": sha256(HERE / "engine_probe_verdict.py"), "summary": sha256(__file__)},
               "inputs_sha256": {p.name: sha256(p) for p in sorted(raw_dir.glob("*.json")) if p.name != "oc.json"}
                                | {"oc.json": sha256(a.oc)},
               "test": oc["chosen_test"],
               "match": {k: match[k] for k in ("k_star", "diff_pp", "bisect")} | {"target_median": match["target"]["median"],
                         "points": {k: v["median"] for k, v in match["points"].items()}},
               "arm_seconds": {arm: d["seconds"] for arm, d in arms.items()},
               "verdict": verdict, "describe": {arm: describe_arm(d["rows"]) for arm, d in arms.items()},
               "pairs": pairs}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"outcome": verdict["outcome"], "reasons": verdict["reasons"], "test": oc["chosen_test"],
                      "contrast": verdict.get("contrast"), "arms": verdict.get("arms"), "match": summary["match"]}, indent=1))


if __name__ == "__main__":
    main()
