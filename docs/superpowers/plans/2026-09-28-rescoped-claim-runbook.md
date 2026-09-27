# Re-scoped claim — runbook (controller only)

Plan: `docs/superpowers/plans/2026-09-28-rescoped-claim.md`. Ledger: `docs/superpowers/plans/2026-09-28-rescoped-claim-ledger.md`.
All commands run from the worktree root on a committed tree (every CLI refuses a dirty `flymon/rescope/` or its own
script without `--allow-dirty`; do not pass `--allow-dirty` to a real stage). Every command below runs in the
background (`run_in_background`); read only the exit code and the error lines (`tail` / `grep`) of its log.
Scripts that build a FlyPool keep all work under `if __name__ == "__main__":` (spawn re-import) — never import them
into a long-lived process to run them.

## Smoke (after all tasks; controller runs these in the background, reads exit code + error lines only)

Order respects the dependencies: the brain arms of step 6 read `results/rescope-smoke/summary/rescope_taurec.json`
(step 3, status SELECTED); primary (step 5) reads `results/rescope-smoke/summary/rescope_qualify.json` (step 4); RS
needs FLY's complete `result.json` (FLY first, and RS only after FLY exits 0); power (7) and M4 (8) read the pilot arms.

1. `uv run pytest -q -rfE -o addopts="" tests/rescope tests/agent tests/battle > <log> 2>&1`   (redirect; piping to tail hangs)
2. `uv run python scripts/rescope_selfcheck.py`   → `results/rescope/selfcheck.json`, exit 0 only if `all_ok`
   (four checks: pairs_digest, oracle_copy_real, pool_equivalence_c3, naive_vs_oracle_pre; exit 1 = a check failed —
   read that check's entry, it carries `error` / `traceback` if it raised)
3. `uv run python scripts/run_rescope_taurec.py --smoke --out results/rescope-smoke/taurec`
   then `uv run python scripts/run_rescope_taurec.py --smoke --reselect --out results/rescope-smoke/taurec` (amended selection, spec 10.6 2026-09-28)
   (no `--workers` flag: one 1-worker pool, then a 2-worker pool per recovery value; smoke = 20 pulses every 5,
   4 odours, grid (0.0, 0.02); exit 0 even on STOP_NO_RECOVERY, but then step 6's brain arms refuse — record it)
4. `uv run python scripts/run_rescope_qualify.py --smoke --workers 4 --out results/rescope-smoke/qualify`
   (smoke spec: 2 qualification seeds; STOP_FEW_PAIRS in the smoke summary is not an error, exit 0)
   (floor rule B, spec 10.3 amendment 2026-09-28: the smoke has 4 seeds per odour, so 1/8 allows no silent seed —
   the smoke seed0 Y `[45, 49, 1, 52]` still fails there and STOP_CONTROL_INVALID on smoke is expected, not a code
   failure; check that each pair entry carries `floor` and `silent_share`)
5. `uv run python scripts/run_rescope_primary.py --smoke --workers 4 --out results/rescope-smoke/primary`
   (refuses unless step 4's smoke summary has stop None and control_qualified; if the 2-seed smoke qualification
   stops, record it and skip 5 — it is not a code failure; when it runs, each `recorded.<pair>` carries `silent`)
6. Pilot arms, one after the other (each starts its own Showdown server; `--smoke` forces flies 2, learn 2, eval 2,
   workers 2 — `--workers` / `--flies` / `--learn` / `--eval` are ignored):
   `for ARM in FLY RS COFF RND MAX; do uv run python scripts/run_rescope_battles.py --smoke --phase pilot --arm $ARM --out results/rescope-smoke/pilot/$ARM || break; done`
   (FLY must exit 0 before RS starts: RS refuses without `results/rescope-smoke/pilot/FLY/result.json`)
7. `uv run python scripts/write_rescope_power.py --smoke --m-overlap <yes|no>`
   → `results/rescope-smoke/summary/rescope_power.json`. Exit 0 = SIZED, exit 2 = STOP_BUDGET / STOP_POWER (both are
   fine on 2x2 smoke data; the check is that it sized without a refusal / traceback). `--workers` defaults to 16.
8. `uv run python scripts/write_rescope_m4.py --smoke --phase pilot`
   → `results/rescope-smoke/summary/rescope_m4.json` (the CLI accepts `--phase pilot` only with `--smoke`; it reads all
   five smoke arms incl. MAX). Exit 0 = PASS / FAIL verdicts written; exit 2 = INVALID (read `reasons`; with 2 flies a
   single INVALID fly already makes a verdict INVALID via the pair floor of 2). `recorded.silent_decisions` holds
   FLY / RS / COFF (record-only, spec 10.3 amendment 2026-09-28).

Record wall-clock of each smoke step in the ledger (feeds the cost estimates of spec 5).

## Notes for the real stages

- Primary checkpoint key = the git commit (+ pair name). Committing anything between an interrupt and the resume
  changes the key, so the resume restarts every pair from scratch. Do not commit while S3 is interrupted.
- Battle runs record every session (aborted ones too) in `<out>/wall_clock.json`; a hard-killed session (no end
  record) makes `write_rescope_power` refuse — rerun the arm rather than editing the file. Pass the judge run's
  `--workers` to `write_rescope_power.py --workers` (the hours estimate scales by ceil(F / workers)) and then use that
  same `--workers` for every S5 battle arm.
- Smoke cannot force a server-error retry. Watch the first real pilot arm (S4 FLY) for retry events
  (a `retries/` directory under the arm's logs, `retries` > 0 in the battle records): the Task 8 review flagged a possible late pulse after a
  rollback. If any retry happened, check that fly's pulse count against its committed learning battles before S4 RS.
- Resuming an interrupted battle arm: rerun the same command with `--resume`.

## Real stages — NOT started without the user's go

Before each: controller: check `orca worktree ps` for M's oracle/scan and report to the user first
(is M's oracle / scan running in guillemot? report it and whether the stage would share the CPU; wait for the go;
record the answer as `--m-overlap-note` / `--m-overlap`).

S1  controller: check `orca worktree ps` for M's oracle/scan and report to the user first
    `uv run python scripts/run_rescope_taurec.py --out results/rescope/taurec`
    then, only after that process has exited (it writes the old-rule summary at the end; spec 10.6 amendment 2026-09-28):
    `uv run python scripts/run_rescope_taurec.py --reselect --out results/rescope/taurec`
    (no trajectories; rewrites `results/summary/rescope_taurec.json` with `rule: "10.6 amendment 2026-09-28"` and the
    old selection under `superseded_rule_v1`; report every r's `path_min`, `floor_frac_taught_path_max` and `failed`;
    stop if status STOP_NO_RECOVERY)
S2  controller: check `orca worktree ps` for M's oracle/scan and report to the user first
    `uv run python scripts/run_rescope_qualify.py --workers 16 --out results/rescope/qualify`
    (stop if STOP_FEW_PAIRS / control not qualified; the floor is rule B — spec 10.3 amendment 2026-09-28; report
    every pair's `silent_share` with the result)
S3  controller: check `orca worktree ps` for M's oracle/scan and report to the user first
    `uv run python scripts/run_rescope_primary.py --workers 16 --out results/rescope/primary`
S4  controller: check `orca worktree ps` for M's oracle/scan and report to the user first
    pilot (6 flies, L 40, E 20 for every arm): FLY, then RS (after FLY exits 0), then COFF, RND
      `uv run python scripts/run_rescope_battles.py --phase pilot --arm <ARM> --flies 6 --eval 20 --learn 40 --workers 16 --m-overlap-note <yes|no> --out results/rescope/pilot/<ARM>`
    then `uv run python scripts/write_rescope_power.py --m-overlap <yes|no> --workers <judge workers, 16>`
S5  controller: check `orca worktree ps` for M's oracle/scan and report to the user first
    judge (only if `results/summary/rescope_power.json` status SIZED): FLY, RS (after FLY exits 0), COFF, RND, MAX with
      `uv run python scripts/run_rescope_battles.py --phase judge --arm <ARM> --flies F --eval E --learn 40 --workers <same as power's --workers> --m-overlap-note <yes|no> --out results/rescope/judge/<ARM>`
    then `uv run python scripts/write_rescope_m4.py --phase judge`
