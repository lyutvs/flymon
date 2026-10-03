"""U's numbers that no R / S / T module returns — records and the values U's gates read; no engine.
- side (U.3 3 (a), U.6, U.9.4 P2-6): one engine variant's reference-set side — T's z_side (z over the readout types, H.4's
  guard per readout type, edges, CSC, counts) plus the mechanism record: every listed type's guard statistics
  (h3_rules.mbon_type_stats on any type's summed count), the α′β′ KC subtypes' and all KCs' active fraction and the APL
  release (medians over the 96 presentations and over the same-seed rest), and the contrast block's edge counts.
- f_record (U.9.4 P2-6): MBON05's mean under L_f over its mean unedited, and σA_f / σA_h4 (σP too).
- row_diffs (U.9.1): U's reference / rest rows against T's, bit for bit on what T's rows hold (odour, seed, every T type
  count, KC active fraction, CSC) — the edge count is U's own label (2 for every f, T's 0 for "none") and not compared.
- oracle_diffs (U.9.1): U's oracle results against R's even raw, bit for bit except the edit labels (q.edit,
  q.edit_edges).
- even_record (U.3 4-5): one L_f's even summary against C — testable_b, F_a, naive_a, per-axis counts, S's net drops."""
from __future__ import annotations

import numpy as np

from . import s_rules, t_records
from .h3_rules import mbon_type_stats
from .h3_store import canonical

LABELS = ("edit", "edit_edges")


def _med(x) -> float | None:
    return float(np.median(x)) if len(x) else None


def side(ref: list, rest: list, readout: dict, types, kc_types, spec) -> dict:
    out = t_records.z_side(ref, rest, readout, spec)
    rb = {int(r["seed"]): r for r in rest}
    out["mech"] = dict(
        types={t: mbon_type_stats(ref, rb, t, spec.z_guard_med_min, spec.z_guard_zero_max) for t in types},
        kc_sub={k: dict(stim_median=_med([r["mech"]["kc_sub"][k] for r in ref]),
                        rest_median=_med([r["mech"]["kc_sub"][k] for r in rest])) for k in kc_types},
        kc_active_median=dict(stim=_med([r["kc_active_frac"] for r in ref]),
                              rest=_med([r["kc_active_frac"] for r in rest])),
        apl_out_median=dict(stim=_med([r["mech"]["apl_out_per_step"] for r in ref]),
                            rest=_med([r["mech"]["apl_out_per_step"] for r in rest])))
    out["block_edges"] = sorted({canonical(r["block_edges"]) for r in ref + rest})
    return out


def f_record(side_f: dict, side_1: dict, z_h4: dict, readout: dict) -> dict:
    """U.9.4 P2-6: MBON05 mean ratio L_f / unedited (U.9.1's f = 1 side) and σ_f / σ_h4 per readout (None without z)."""
    p = readout["P"]
    m1 = side_1["mech"]["types"][p]["mean_stim"]
    z = side_f.get("z")
    return dict(p_mean_ratio=(side_f["mech"]["types"][p]["mean_stim"] / m1) if m1 else None,
                sd_ratio=None if z is None else {k: float(z[k][1]) / float(z_h4[k][1]) for k in z_h4})


def row_diffs(u_rows: list, t_rows: list, label: str, rest: bool = False) -> list:
    """[] when every U row equals T's row in order on T's fields; else the differing items (at most 3 named)."""
    if len(u_rows) != len(t_rows):
        return [f"{label}: {len(u_rows)} rows, T {len(t_rows)}"]
    bad = []
    for u, t in zip(u_rows, t_rows):
        keys = ("seed", "csc_sha256") if rest else ("odor", "seed", "kc_active_frac", "csc_sha256")
        same = all(u.get(k) == t.get(k) for k in keys) and {k: u["types"].get(k) for k in t["types"]} == t["types"]
        if not same:
            bad.append(f"{t.get('odor', 'rest')}/{t['seed']}")
    return [f"{label}: {len(bad)} row(s) differ: {bad[:3]}"] if bad else []


def strip_labels(result: dict) -> dict:
    """Plan Reading 3: drop only result["q"]["edit"] and result["q"]["edit_edges"]; same-named keys elsewhere stay."""
    if not isinstance(result, dict) or not isinstance(result.get("q"), dict):
        return result
    return dict(result, q={k: v for k, v in result["q"].items() if k not in LABELS})


def oracle_diffs(u_got: list, r_got: list, label: str) -> list:
    if [g["key"] for g in u_got] != [g["key"] for g in r_got]:
        return [f"{label}: pair lists differ"]
    bad = [u["key"] for u, r in zip(u_got, r_got)
           if canonical(strip_labels(u["result"])) != canonical(strip_labels(r["result"]))]
    return [f"{label}: {len(bad)} pair(s) differ: {bad[:3]}"] if bad else []


def even_record(L: dict, C: dict, spec_f) -> dict:
    """One qualified f's even values (U.3 5 "모든 자격 f의 짝수 값") and S's per-axis net drops against C."""
    a = L["aggregate"] or {}
    gf = s_rules.g_fail_s(L["pairs"], C["pairs"], spec_f)
    return dict(f=spec_f.f, testable_b=a.get("testable_b"), F_a=a.get("F_a"), naive_a=a.get("naive_a"),
                counts=L["counts"], drops={ax: dict(pun_L=v["pun_L"], pun_C=v["pun_C"], net_drop=v["net_drop"])
                                           for ax, v in gf["axes"].items()},
                edit_edges=L["edit_edges"], csc_sha256=L["csc_sha256"], reasons=L["reasons"])
