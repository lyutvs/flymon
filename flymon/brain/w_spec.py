"""Every number of spec appendix W as amended by W.9 (W.9.10 > W.9.9 > W.9.8 > W.9.1-W.9.7 > W.0-W.8): the F v4 learning
test (spec 5's M2 learning unit, F.2's sequential R / N brains plus G.5's state-matched RN) on V's combined lever L_V,
read with V's z_V, judged by per-fly joint satisfaction with a design (q, K, F; k cap 8) chosen by W's own operating
characteristic (w_oc) from a POOL even pilot.
- WSpec is a plain frozen dataclass, not a VSpec: every field whose name holds "seed" carries a W block only, so the
  repository's seed collectors (tests/brain/test_p_spec.py MODULES) see W's blocks and nothing of V's. V's numbers
  (the lever string, the oracle's windows, alphas, seeds and z rule, the readout, the gate-② P spec) are read from
  v_spec.SPEC where they are used (w_runner, w_pairs), never restated here.
- Seeds (W.9.6 P2-11, W.9.8 H9): candidate c of the main set (declared order), fly f, probe k, trial t —
  probe 26_000_000 + c·4_000 + f·100 + k, training 28_000_000 + c·40_000 + f·1_000 + t (t < 40: reward / first
  phase t < 20, punishment / second phase 20 ≤ t < 40); pilot pair j — 40_000_000 + j·4_000 + f·100 + k and
  41_000_000 + j·40_000 + f·1_000 + t; smoke inside 42_100_000-42_199_999; w_oc's generator root 42_000_000; the
  oracle screen 24_700_000+i / 24_700_100+i / 24_700_200+i (i < 8).
- The protocol table (W.9.6 P2-12, "W.1 부록 표"): C3's Params unchanged (block h4: kc_kc_scale 0, recovery 0, learn
  rate 3e-4 …, read from the C3 config at run time), o_jobs.train_x's settle 800 ms and gap 200 ms, F.2's pulse 400 ms
  × 20 per phase, PAM08 (reward) / PPL105 (punishment), the probe windows of V's oracle (settle 800 · read 600 ·
  window 200 at the E-grid strength 1.0 — the same probe path as V's oracle, W.3 2 (ii)); training at the same
  strength 1.0.
- Paths: results/w/ (git-ignored), results/summary/w_learning.json (tracked), ~/flymon-archive/w."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

BRAINS = ("R", "N", "RN")                          # F.2 R and N, G.5's RN (W.9.6 P2-10: RN runs from pre)


@dataclass(frozen=True)
class WSpec:
    # ---- engine and readout (W.1) ------------------------------------------------------------------------------------
    z_v_declared: tuple = (("A", (16.917, 12.484)), ("P", (80.167, 29.775)))   # W.1, V block z 928eaad (3 decimals)
    z_v_digits: int = 3
    # ---- the protocol table (F.2, W.9.6 P2-12) -----------------------------------------------------------------------
    trials: int = 20                               # per phase
    pulse_ms: float = 400.0
    train_settle_ms: float = 800.0                 # o_jobs.train_x's settle (h4 teach_window)
    gap_ms: float = 200.0                          # o_jobs.train_x's gap (h4 teach_gap_ms)
    strength: float = 1.0                          # E-grid k2-norm s 1.0: probes and training
    reward_dan: str = "PAM08"
    punish_dan: str = "PPL105"
    # ---- the verdict (F.5, W.9.1, W.9.2, W.9.8 H2 / H5 / H7) --------------------------------------------------------
    bar: float = 1.0
    band_width: float = 0.2
    naive_max: float = 0.5
    mech_min: float = 0.75
    round_digits: int = 9
    min_gate_pairs: int = 4
    # ---- the designs (W.9.8 H1, W.9.9 P2-7) --------------------------------------------------------------------------
    q_grid: tuple = (0.5, 0.625, 0.75)
    k_grid: tuple = (8, 16)                        # probes K; BAND re-measure 2K
    f_min: int = 8
    f_max: int = 32
    k_min: int = 4                                 # gate pairs: the OC's k runs k_min .. k_cap
    k_cap: int = 8
    envelope: int = 3                              # F, F+1, F+2, F+3
    envelope_solo_from: int = 29                   # F 29-32 alone (W.9.9 P1-3)
    # ---- G.6's targets and w_oc (W.9.3, W.9.8 H4, W.9.9) -------------------------------------------------------------
    d_power: float = 1.5
    p_power: float = 0.80
    d_false: float = 0.5
    p_false: float = 0.05
    oc_reps: int = 4000
    oc_seed: int = 42_000_000
    cal_reps: int = 2000
    cal_tol: float = 0.02
    cal_iter: int = 40
    boot_draws: int = 200
    boot_reps: int = 400                           # experiments per bootstrap draw (Reading 11)
    boot_level: float = 0.95
    cluster_grid: tuple = (0.0, 0.5, 1.0)
    record_dprimes: tuple = (0.5, 1.0, 1.5, 2.0)
    oc_chunk: int = 50
    cal_floor_rule: str = "zero"                   # W.9.10 2: "zero" (null = zero DAN injection); "unreachable" = literal
    synth_reps: int = 1000                         # stage 0: experiments per synthetic-validation check (P2-11)
    synth_null_max: float = 0.02                   # P2-11: zero-effect / one-gate / negative-correlation P(PASS) ≤
    synth_big_min: float = 0.98                    # P2-11: big-effect P(PASS) ≥
    synth_big_dprime: float = 4.0                  # P2-11: the big effect's calibrated true d′ (min gate)
    synth_drift_dprime_min: float = 1.5            # P2-11: the one-gate fixture's drifting gate has true d′ ≥
    simple_normal_fs: tuple = (8, 16, 24, 32)      # P0-1: the F ladder P(PASS) must not rise along
    simple_normal_reps: int = 2000                 # P0-1: experiments per F
    # ---- the pilot (W.3 3, W.9.4, W.9.8 H6) --------------------------------------------------------------------------
    pilot_flies: int = 8
    pilot_probes: int = 8
    pilot_extra: tuple = ("a|4|Rock Slide|Strength",)    # + every L_V-testable (b) even pair (V block even)
    no_effect_d: float = 0.5
    exploratory_q: float = 0.75                    # the pilot's exploratory (label "탐색") judge only — no gate
    no_effect_share: float = 0.5
    floor_share_max: float = 0.5
    naive_floor_spikes: int = 2                    # W.9.2's pilot pre-check record (< 2 spikes)
    # ---- the main set (W.2) ------------------------------------------------------------------------------------------
    first_turn: int = 306
    last_turn: int = 1985
    n_b_expected: int = 167                        # W.0 fact check
    n_a_expected: int = 82
    n_cand_max: int = 300
    # ---- seeds (W.9.6 P2-11; Reading 6) ------------------------------------------------------------------------------
    probe_seed0: int = 26_000_000
    train_seed0: int = 28_000_000
    pilot_probe_seed0: int = 40_000_000
    pilot_train_seed0: int = 41_000_000
    smoke_probe_seed0: int = 42_100_000
    smoke_train_seed0: int = 42_110_000
    smoke_oracle_seed0: int = 42_150_000
    cand_probe_stride: int = 4_000
    cand_train_stride: int = 40_000
    fly_probe_stride: int = 100
    fly_train_stride: int = 1_000
    phase_trial_offset: int = 20                   # the second phase's t starts at 20
    n_pilot_max: int = 25
    oracle_act_seeds: tuple = tuple(range(24_700_000, 24_700_008))
    oracle_select_seeds: tuple = tuple(range(24_700_100, 24_700_108))
    oracle_report_seeds: tuple = tuple(range(24_700_200, 24_700_208))
    # ---- the path gate (W.3 2, W.9.9 P2-9) ---------------------------------------------------------------------------
    repro_p_count: int = 2                         # V gate ②'s first 2 seeds per direction
    repro_p_arm: str = "punish"
    repro_naive_rows: int = 3                      # V's judgement rows 0-2, both conditions
    # ---- the plasticity-off control and C (W.3 9) --------------------------------------------------------------------
    noplast_pairs: int = 2
    noplast_flies: int = 2
    # ---- the cost model (W.9.6 F): one learning job = 2 × trials training trials + job_stages probe stages ----------
    job_stages: int = 3                            # pre, after the first phase, after the second
    # ---- the budget (W.9.6 F, W.9.9 P1-4 / P1-5) ---------------------------------------------------------------------
    budget_h: float = 24.0
    # ---- V's reuse (W.3 1) -------------------------------------------------------------------------------------------
    r_shared_key: str = "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    t_measure_key_t: str = "7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374"
    u_measure_key_u: str = "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    v_commits: tuple = (("z", "928eaad"), ("kc_input", "7dc199d"), ("set", "cf0b3b2"), ("judge", "a279a56"))
    v_band: str = "SELECTED"
    v_summary: str = "results/summary/v_lever.json"
    # ---- smoke and the pool ------------------------------------------------------------------------------------------
    smoke: bool = False
    smoke_flies: int = 1
    workers: int = 16
    run_workers: int = 0                           # the real run's pool size for cost estimates; 0 = workers (smoke()
                                                   # keeps the real one, so the smoke's 4 workers never enter a cost)
    pool_timeout_s: float = 3600.0
    # ---- paths -------------------------------------------------------------------------------------------------------
    summary: str = "results/summary/w_learning.json"
    raw_dir: str = "results/w"
    cache_dir: str = "results/w/cache"
    smoke_cache_dir: str = "results/w/smoke/cache"
    progress_dir: str = "results/w/progress"
    oc_detail: str = "results/w/oc.json"
    path_detail: str = "results/w/path.json"
    archive_root: str = "~/flymon-archive/w"

    # ---- seeds -------------------------------------------------------------------------------------------------------
    def _probe(self, root: int, stride: int, n: int, f: int, k_n: int) -> list:
        if not 0 <= f < self.f_max or not 0 < k_n <= self.fly_probe_stride:
            raise ValueError(f"fly {f} / {k_n} probes outside the block layout")
        return [root + n * stride + f * self.fly_probe_stride + k for k in range(k_n)]

    def probe_seeds(self, c: int, f: int, k_n: int, k0: int = 0) -> list:
        """Main-set candidate c, fly f: probes k0 .. k_n - 1 (W.9.6 P2-11)."""
        if not 0 <= c < self.n_cand_max:
            raise ValueError(f"candidate {c} outside 0..{self.n_cand_max - 1}")
        return self._probe(self.probe_seed0, self.cand_probe_stride, c, f, k_n)[k0:]

    def pilot_probe_seeds(self, j: int, f: int, k_n: int, k0: int = 0) -> list:
        if not 0 <= j < self.n_pilot_max:
            raise ValueError(f"pilot pair {j} outside 0..{self.n_pilot_max - 1}")
        return self._probe(self.pilot_probe_seed0, self.cand_probe_stride, j, f, k_n)[k0:]

    def smoke_probe_seeds(self, f: int, k_n: int, k0: int = 0) -> list:
        return self._probe(self.smoke_probe_seed0, 0, 0, f, k_n)[k0:]

    def train_base(self, c: int, phase: int) -> int:
        """train_x's seed_base of phase 0 / 1 for candidate c (train seed = base + f·1_000 + t′)."""
        if not 0 <= c < self.n_cand_max:
            raise ValueError(f"candidate {c} outside 0..{self.n_cand_max - 1}")
        return self.train_seed0 + c * self.cand_train_stride + phase * self.phase_trial_offset

    def pilot_train_base(self, j: int, phase: int) -> int:
        if not 0 <= j < self.n_pilot_max:
            raise ValueError(f"pilot pair {j} outside 0..{self.n_pilot_max - 1}")
        return self.pilot_train_seed0 + j * self.cand_train_stride + phase * self.phase_trial_offset

    def smoke_train_base(self, phase: int) -> int:
        return self.smoke_train_seed0 + phase * self.phase_trial_offset

    def oracle_seeds(self) -> dict:
        if self.smoke:
            s = self.smoke_oracle_seed0
            return dict(act=list(range(s, s + 8)), select=list(range(s + 100, s + 108)),
                        report=list(range(s + 200, s + 208)))
        return dict(act=list(self.oracle_act_seeds), select=list(self.oracle_select_seeds),
                    report=list(self.oracle_report_seeds))

    def smoke_seed_set(self) -> frozenset:
        """Every seed a smoke entry may carry (WCache's smoke scope)."""
        s = set()
        for f in range(self.f_max):
            s |= set(self._probe(self.smoke_probe_seed0, 0, 0, f, self.fly_probe_stride))
        o = self.smoke_oracle_seed0
        return frozenset(s | set(range(o, o + 8)) | set(range(o + 100, o + 108)) | set(range(o + 200, o + 208)))

    # ---- phases (F.2, G.5) -------------------------------------------------------------------------------------------
    def phases(self, brain: str, base0: int, base1: int) -> list:
        """[[dan or None, trials, seed_base]] of R (PAM08 then PPL105), N (none, none) and RN (PAM08, then none)."""
        dans = {"R": (self.reward_dan, self.punish_dan), "N": (None, None), "RN": (self.reward_dan, None)}[brain]
        return [[dans[0], int(self.trials), int(base0)], [dans[1], int(self.trials), int(base1)]]

    def designs(self) -> list:
        return [(q, k, f) for q in self.q_grid for k in self.k_grid for f in range(self.f_min, self.f_max + 1)]

    def z_v(self) -> dict:
        return {k: tuple(v) for k, v in self.z_v_declared}

    def cost_workers(self) -> int:
        """The worker count every cost estimate rounds by: the real run's pool, not the smoke pool (W.9.6 F)."""
        return int(self.run_workers or self.workers)


SPEC = WSpec()


def smoke(spec: WSpec = SPEC) -> WSpec:
    """Scale only: the smoke flag (oracle seeds 42_150_xxx, inside the smoke block) and 4 workers; run_workers keeps
    the real run's pool size for the cost ledger. Every threshold stays."""
    return dataclasses.replace(spec, smoke=True, workers=4, run_workers=spec.cost_workers())
