"""Fabricated reference-set rows for T's z tests (no engine): ref_rows / rest_rows give t_ref_job / t_rest_job's
shape. With base counts a / p per presentation and a rest of `rest` spikes, the readout guard reads median(read − rest)
and the zero share of the read counts; z is the counts' mean and population SD."""

SHA_NONE, SHA_L = "sha-C", "sha-L"


def ref_rows(a: list, p: list, edges: int = 0, sha: str = SHA_NONE, seed0: int = 1000) -> list:
    return [dict(odor=f"R{i // 2:02d}", seed=seed0 + i, types={"MBON13": int(x), "MBON05": int(y)},
                 kc_active_frac=0.05, csc_sha256=sha, edit_edges=edges) for i, (x, y) in enumerate(zip(a, p))]


def rest_rows(n: int, rest: int = 0, edges: int = 0, sha: str = SHA_NONE, seed0: int = 1000) -> list:
    return [dict(seed=seed0 + i, types={"MBON13": rest, "MBON05": rest}, csc_sha256=sha, edit_edges=edges)
            for i in range(n)]


def counts(n: int, base: int, zeros: int = 0) -> list:
    """n counts: `zeros` zeros first, the rest base + (i % 5)."""
    return [0] * zeros + [base + (i % 5) for i in range(n - zeros)]
