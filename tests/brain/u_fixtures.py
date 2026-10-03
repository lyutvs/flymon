"""Fabricated reference-set rows for U's tests (no engine): ref_rows / rest_rows give u_ref_job / u_rest_job's shape —
T's row (types, KC active fraction, CSC, edges) plus the mechanism reads (KC subtypes, APL release) and the contrast
block's edge counts. With base counts a / p per presentation and a rest of `rest` spikes, the guard reads
median(read − rest) and the zero share of the read counts; z is the counts' mean and population SD."""
from flymon.brain.u_spec import SPEC

KC = SPEC.kc_types


def ref_rows(a: list, p: list, edges: int = 2, sha: str = "sha-L", seed0: int = 1000, extra=None, kc=0.05, sub=0.2,
             apl=0.3, blocks=None) -> list:
    return [dict(odor=f"R{i // 2:02d}", seed=seed0 + i, types={"MBON13": int(x), "MBON05": int(y), **(extra or {})},
                 kc_active_frac=kc, csc_sha256=sha, edit_edges=edges, block_edges=dict(blocks or {}),
                 mech=dict(kc_sub={k: sub for k in KC}, apl_out_per_step=apl))
            for i, (x, y) in enumerate(zip(a, p))]


def rest_rows(n: int, rest: int = 0, edges: int = 2, sha: str = "sha-L", seed0: int = 1000, extra=None, kc=0.002,
              sub=0.01, apl=0.1, blocks=None) -> list:
    return [dict(seed=seed0 + i, types={"MBON13": rest, "MBON05": rest, **(extra or {})}, kc_active_frac=kc,
                 csc_sha256=sha, edit_edges=edges, block_edges=dict(blocks or {}),
                 mech=dict(kc_sub={k: sub for k in KC}, apl_out_per_step=apl)) for i in range(n)]
