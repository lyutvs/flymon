"""Fabricated P arm rows for S's gate-② tests (no engine): arm_row gives r_arm_job + RMeasurer.arms' shape for one
item; only the punish arm (③) lowers X's A after training, by drop + seed % 3, so per seed dl = (drop + seed % 3) /
z_A's SD and ℓ ≈ (drop + 1) / 9 under Z — p_rules.p_judge confirms a direction when ℓ ≥ c₁ (0.608, drop ≥ 5) and its
s / t conditions hold (X falls, Y does not move). p_rows builds a whole P grid (two directions × three arms × seeds)."""
from flymon.brain.r_runner import p_items

Z = {"A": (10.0, 9.0), "P": (26.0, 19.0)}
C1 = 0.608


def arm_row(item: dict, drop: float, sha: str, edges: int) -> dict:
    k = float(drop + int(item["seed"]) % 3) if item["punish"] else 0.0
    probe = {"A": 30.0, "P": 26.0, "kc_frac": 0.05, "kc_spikes": 100}
    post_x = dict(probe, A=30.0 - k)
    return dict(seed=int(item["seed"]), edit=item["edit"], arm=item["arm"], punish=bool(item["punish"]),
                plastic=bool(item["plastic"]), da_zero=bool(item["da_zero"]), csc_sha256=sha,
                pre={"x": dict(probe), "y": dict(probe)}, post={"x": post_x, "y": dict(probe)}, weights_frac=0.0,
                weights_frac_A=0.0, weights_frac_P=0.0, w0_sha256="w0",
                w_post_sha256="w0" if not item["plastic"] else "w1", da_integral={}, wall_s=1.0,
                r=dict(edit_edges=int(edges), p_type="MBON05"), direction=item["direction"], x=item["x"], y=item["y"],
                point=[float(v) for v in item["point"]])


def stimuli(pspec) -> dict:
    names = {s for xy in pspec.pairs().values() for s in xy}
    return {n: {"odor": {f"G_{n}": 1.0}} for n in names}


def p_rows(pspec, edit: str, drop: float, sha: str = "sha-C", edges: int = 0) -> list:
    return [arm_row(i, drop, sha, edges) for i in p_items(pspec, stimuli(pspec), edit)]
