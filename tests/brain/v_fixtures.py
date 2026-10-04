"""Fabricated R, T and U summaries for V's reuse tests (no engine): R's repro and gate ③, T's z block (its unedited side
on block h4's z of this world, A 10 / 9, P 26 / 19), U's reuse / path / scan / kc blocks under U's measurement key ("u"
× 64) with the detail shas "e" × 64 / "f" × 64, the entry_f0 point on CSC "sha-V" and per_odour_none on U_IDS + one
odour V never lists; CANDS are the world's 20 KC candidate odours (5 of them U's)."""
from flymon.brain.v_spec import SPEC

U_IDS = [f"U{i}|T" for i in range(5)]
CANDS = U_IDS + [f"K{i}|T" for i in range(15)]


def r_doc() -> dict:
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none="sha-C", written_at="r-repro"),
                smoke=dict(code_key=k, oracle={"L": {"csc_sha256": "sha-L0"}}),
                gate3=dict(outcome="PASS", code_key=k, testable_b=16, c_even=7, written_at="r-g3"))


def t_doc() -> dict:
    g = dict(passes=True, median_delta=10.0, zero_share=0.0)
    return dict(z=dict(outcome="STOP_Z_DEGENERATE", t_measure_key="t" * 64, code_key=SPEC.r_shared_key,
                       detail_sha256="d" * 64, lever=dict(z={"A": [0.5, 1.0], "P": [80.0, 20.0]}),
                       none=dict(z={"A": [10.0, 9.0], "P": [26.0, 19.0]}, csc_sha256=["sha-C"], edit_edges=[0],
                                 guard={"MBON13": g, "MBON05": g})))


def u_doc() -> dict:
    keys = dict(u_measure_key="u" * 64, code_key=SPEC.r_shared_key, t_measure_key="t" * 64)
    pts = {k: dict(record_set=dict(per_odour_none={o: 0.05 for o in U_IDS + ["Z|T"]})) for k in SPEC.u_kc_points}
    return dict(reuse=dict(outcome="PASS", **keys), path=dict(outcome="PASS", detail_sha256="e" * 64, **keys),
                scan=dict(outcome="PASS", detail_sha256="f" * 64, contrast=dict(
                    entry_f0=dict(invalid=[], side=dict(csc_sha256=["sha-V"], z={"A": [16.9, 12.5]})),
                    readings=dict(entry=dict(reading="사슬 지지"))), **keys),
                kc=dict(outcome="STOP_NO_QUALIFIED_F", points=pts, **keys))
