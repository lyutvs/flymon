"""AD's situation-pair evaluations and per-battle hooks (spec AD.2, AD.3, AD.7). Records carry the global fly (`fly`),
the pool index, the arm, the point, the set ("confirm" | "new") and the mode ("own" | "os_paired"), floor contact,
the plastic-weight summary, FLY-OS's draws (`os`) and AC.5's label.
- Kinds per point: (confirm, own) every fly - FLY-L on the FLY odours, FLY-OS on OS odours keyed "os-sit";
  (confirm, os_paired) FLY-L flies on OS odours keyed "os-sit-paired" with the SAME decision seeds as (confirm, own)
  (within-fly pairing, AD.2); (new, own) every fly at the new-set points (0 / 40 / T), decision key "sit-new" (FLY-OS
  OS key "os-sit-new"), only if the run manifest includes the set.
- Point 0: initial() evaluates every fly lacking a point-0 record of a kind on w0, into logs/situations_init.jsonl, kept
  across resumes (deterministic from w0 and the seeds).
- Points 20 / 40 / ...: after_battle evaluates a fly right after its learning battle point - 1 settles and before that
  battle's checkpoint commit, into logs/situations.jsonl keyed by that battle id; a resume filters the file (and the
  retry log) to committed battles first, so an evaluation whose battle was not committed is redone.
- A battle that needed a retry is written to logs/retry_log.jsonl (global fly, battle id, attempts, invalid); a battle
  still unfinished after the retries marks the fly in the learning or the evaluation InvalidBook (AD.7: a learning
  failure drops the fly from the later points and from the evaluation block; an evaluation failure drops it from
  criteria 2-3 only). A fly skipped in the evaluation block because it is invalid in learning is "learn_invalid".
- A record whose weights changed during the evaluation stops the run: STOP_SMOKE in the smoke, STOP_INFRA otherwise."""
from __future__ import annotations

from pathlib import Path

from ..ac import diag, situations
from ..ac.stage1 import SituationLog
from ..agent.checkpoint import CheckpointStore, filter_log
from ..rescope import blocks
from . import os_encoder
from .inputs import battle_index
from .spec import LABEL

OS_KEY = {("confirm", "own"): "os-sit", ("confirm", "os_paired"): "os-sit-paired", ("new", "own"): "os-sit-new"}
SEED_KEY = {"confirm": "sit", "new": "sit-new"}


class ADSituationLog(SituationLog):
    def __init__(self, logs_dir):
        super().__init__(logs_dir)
        self.retry_path = Path(logs_dir) / "retry_log.jsonl"

    def has_init(self, fly: int, set_: str = "confirm", mode: str = "own") -> bool:
        return any(r.get("fly") == int(fly) and r.get("set") == set_ and r.get("mode") == mode
                   for r in blocks.read_jsonl(self.init_path))

    def add_retry(self, rec: dict) -> None:
        blocks.append_jsonl(self.retry_path, rec)

    def retries(self) -> list:
        return blocks.read_jsonl(self.retry_path)

    def filter(self, completed: set) -> int:
        return super().filter(completed) + filter_log(self.retry_path, set(completed))


def committed_all(out, key: str) -> set:
    done = set()
    for block in ("learn", "eval"):
        try:
            ck = CheckpointStore(Path(out) / "checkpoints" / block, key).load()
        except ValueError as e:
            raise SystemExit(f"STOP_MANIFEST: {out}/checkpoints/{block}: {e} (another run manifest, AD.4)")
        if ck:
            done |= set(ck["completed"])
    return done


def make_evaluate(pool, swarm, cfg, spec, lay, sets: dict, grid):
    cache: dict = {}

    def odours(i: int, set_: str, mode: str):
        arm, _, g = lay[i]
        os_kind = arm == "FLYOS" or mode == "os_paired"
        key = (i if os_kind else arm, set_, mode)
        if key not in cache:
            if os_kind:
                cache[key] = os_encoder.os_pair_odours(grid, sets[set_], g, spec.os_seed, OS_KEY[(set_, mode)])
            else:
                cache[key] = (situations.pair_odours(grid, sets[set_]), None)
        return cache[key]

    def evaluate(i: int, point: int, set_: str, mode: str) -> dict:
        ods, draws = odours(i, set_, mode)
        rec = situations.evaluate(pool, i, point, sets[set_], ods, cfg, swarm.a_idx, swarm.p_idx, swarm.kc,
                                  spec.tie_seed, seed_fly=lay[i][2], seed_key=SEED_KEY[set_])
        rec["floor_contact"] = situations.floor_contact(pool, i, swarm._edge_masks, (cfg.reward_type, cfg.punish_type),
                                                        spec.floor_ratio)
        rec["weights"] = diag.weight_summary(pool.w[i], pool.w0[pool.flies[i].shuffle_seed], spec.floor_ratio)
        if draws is not None:
            rec["os"] = draws
        return rec
    return evaluate


class ADHooks:
    def __init__(self, spec, T: int, new_set: bool, lay, books: dict, sitlog: ADSituationLog, evaluate):
        self.spec, self.T, self.new_set, self.lay = spec, int(T), bool(new_set), list(lay)
        self.books, self.sitlog, self.evaluate = books, sitlog, evaluate

    def kinds(self, i: int, point: int) -> list:
        out = [("confirm", "own")]
        if self.lay[i][0] == "FLYL":
            out.append(("confirm", "os_paired"))
        if self.new_set and int(point) in self.spec.new_points(self.T):
            out.append(("new", "own"))
        return out

    def _record(self, i: int, point: int, set_: str, mode: str, battle_id) -> dict:
        arm, _, g = self.lay[i]
        rec = self.evaluate(i, point, set_, mode)
        if not rec.get("frozen", False):
            code = "STOP_SMOKE" if self.spec.mode == "smoke" else "STOP_INFRA"
            raise SystemExit(f"{code}: fly {g}: weights changed during the point-{point} {set_}/{mode} situation "
                             f"evaluation (AD.7)")
        return dict(rec, fly=g, pool_index=i, arm=arm, point=int(point), set=set_, mode=mode, battle_id=battle_id,
                    label=LABEL)

    def initial(self) -> int:
        self.sitlog.repair_init()
        n = 0
        for i, (_, _, g) in enumerate(self.lay):
            if self.books["L"].skip(i):
                continue
            for set_, mode in self.kinds(i, 0):
                if not self.sitlog.has_init(g, set_, mode):
                    self.sitlog.add_init(self._record(i, 0, set_, mode, None))
                    n += 1
        return n

    def after_battle(self, block: str, sb, rec: dict) -> None:
        i = int(sb.fly_id)
        g = self.lay[i][2]
        book = self.books[block]
        skipped = book.skip(i) or (block == "E" and self.books["L"].skip(i))
        if int(rec.get("retries", 0)) > 0 and not skipped:
            self.sitlog.add_retry(dict(fly=g, pool_index=i, battle_id=sb.battle_id, block=block,
                                       attempts=int(rec["retries"]) + 1, invalid=bool(rec["invalid"])))
        if rec["invalid"]:
            if block == "E" and self.books["L"].skip(i):
                book.mark(i, "learn_invalid")
            else:
                book.mark(i, "arm_stopped" if book.skip(i) else
                          f"{sb.battle_id}: unfinished after {rec['retries']} retries")
            return
        if block != "L" or self.books["L"].skip(i):
            return
        point = battle_index(sb.battle_id) + 1
        if point in self.spec.points(self.T):
            for set_, mode in self.kinds(i, point):
                self.sitlog.add(self._record(i, point, set_, mode, sb.battle_id))


def skipping(attempt_for, books: dict):
    def wrapped(block):
        attempt = attempt_for(block)

        async def att(sb, n):
            i = int(sb.fly_id)
            if books["L"].skip(i) or (block == "E" and books["E"].skip(i)):
                return {"finished": False, "won": None, "skipped": True}
            return await attempt(sb, n)
        return att
    return wrapped
