"""AD.3's contrasts (spec AD.3, AD.7). Nothing here is PASS / FAIL.
- Units: per fly the 20 confirmation pairs (switch = both sides picked the best move, AC.5); a fly enters a point iff its
  record at that point passed the frozen check (validity.records - point-wise validity).
- The decision rule (boot.two_stage_diff, B 10 000, seed 304; "충족" iff diff >= 0.15 and lower end > 0) is applied
  only to the two primaries (FLY-L - FLY-OS at the final point and at 40) and to the floor-contact companion row; every
  other row is "기술". Every situation contrast also carries diag.crossed_boot's CI (one pair draw shared by both
  arms; descriptive, never changes a status).
- Reference row FLY-L(0) - FLY-OS(0); difference of differences per fly = switch(t) - switch(0) pair by pair at t = 40
  and the final point; FLY-L - C-off (stage-1 point-0 records, one column used at both points); FLY-L(40) - stage-1
  FLY(40); the new 20 pairs at 0 / 40 / final; the paired OS evaluation FLY-L own - FLY-L on OS odours (within fly,
  stats.two_stage_mean, crossed CI paired); the sensitivity rows = the two primaries without the flies that needed a
  retry in learning.
- Minimum valid flies per item: FLY-L 9, FLY-OS 9, C-off 4, stage-1 FLY 9, RND / MAX 12; fewer is "측정 불성립".
- Criteria 2-3 (descriptive): FLY-L - FLY-OS information-turn type match and win rate in the evaluation battles; FLY-L -
  RND and MAX - FLY-L against stage 1's RND / MAX (same evaluation-schedule digest). No evaluation block (a budget stop)
  -> "측정 불성립" for these only."""
from __future__ import annotations

from ..ac import boot, diag
from ..ac.analysis import c2_units, c3_units
from . import stats, validity


def switch_units(sits, arm: str, point, *, set_: str = "confirm", mode: str = "own", exclude=()) -> dict:
    if point is None:
        return {}
    ex = set(exclude)
    recs = validity.records(sits, point=point, arm=arm, set_=set_, mode=mode)
    return {g: [int(x) for x in r["switched"]] for g, r in recs.items() if g not in ex}


def did_units(sits, arm: str, point, exclude=()) -> dict:
    u0, ut = switch_units(sits, arm, 0, exclude=exclude), switch_units(sits, arm, point, exclude=exclude)
    return {g: [b - a for a, b in zip(u0[g], ut[g])] for g in ut if g in u0}


def stage1_units(recs) -> dict:
    return {r["fly"]: [int(x) for x in r["switched"]] for r in recs if r.get("frozen") is True}


def contrast(name: str, a: str, b: str, ua: dict, ub: dict, spec, *, point=None, rule: bool = False,
             crossed: bool = True) -> dict:
    mins = spec.min_valid_of()
    base = dict(name=name, a=a, b=b, point=point, n_a=len(ua), n_b=len(ub), min_a=mins[a], min_b=mins[b])
    if not ua or not ub or len(ua) < mins[a] or len(ub) < mins[b]:
        return dict(base, status="측정 불성립")
    res = boot.two_stage_diff(ua, ub, spec.boot_draws, spec.boot_seed)
    out = dict(base, diff=res["diff"], lo=res["lo"], hi=res["hi"])
    if crossed:
        cr = diag.crossed_boot(ua, ub, spec.boot_draws, spec.boot_seed)
        out["crossed"] = dict(lo=cr["lo"], hi=cr["hi"])
    out["status"] = ("충족" if boot.meets(res, spec.min_effect) else "미충족") if rule else "기술"
    return out


def paired_os(sits, spec, point: int) -> dict:
    own = switch_units(sits, "FLYL", point)
    os_ = switch_units(sits, "FLYL", point, mode="os_paired")
    common = sorted(set(own) & set(os_))
    n, m = len(common), spec.min_valid_of()["FLYL"]
    base = dict(name=f"짝지은 OS 평가 FLY-L({point}) − FLY-L-OS평가({point})", a="FLYL", b="FLYL_OS", point=point,
                n_a=n, n_b=n, min_a=m, min_b=m)
    if n < m:
        return dict(base, status="측정 불성립")
    own, os_ = {g: own[g] for g in common}, {g: os_[g] for g in common}
    res = stats.two_stage_mean({g: [x - y for x, y in zip(own[g], os_[g])] for g in common}, spec.boot_draws,
                               spec.boot_seed)
    cr = diag.crossed_boot(own, os_, spec.boot_draws, spec.boot_seed, paired=True)
    return dict(base, status="기술", diff=res["mean"], lo=res["lo"], hi=res["hi"], crossed=dict(lo=cr["lo"], hi=cr["hi"]))


def situation_readings(sits, spec, *, final, new_points, coff, fly1, retried) -> dict:
    fp = spec.futility_point
    u = lambda arm, t, **kw: switch_units(sits, arm, t, **kw)
    prim = [contrast("1차 1", "FLYL", "FLYOS", u("FLYL", final), u("FLYOS", final), spec, point=final, rule=True),
            contrast("1차 2", "FLYL", "FLYOS", u("FLYL", fp), u("FLYOS", fp), spec, point=fp, rule=True)]
    ref = contrast("기준 행 FLY-L(0) − FLY-OS(0)", "FLYL", "FLYOS", u("FLYL", 0), u("FLYOS", 0), spec, point=0)
    pts = [p for p in dict.fromkeys((fp, final)) if p is not None]
    did = [contrast(f"차이의 차이 {t}", "FLYL", "FLYOS", did_units(sits, "FLYL", t), did_units(sits, "FLYOS", t), spec,
                    point=t) for t in pts]
    cu, fu = stage1_units(coff), stage1_units(fly1)
    aux = [contrast(f"FLY-L({t}) − C-off", "FLYL", "COFF", u("FLYL", t), cu, spec, point=t) for t in pts]
    aux.append(contrast(f"FLY-L({fp}) − 1단계 FLY({fp})", "FLYL", "FLY1", u("FLYL", fp), fu, spec, point=fp))
    new = [contrast(f"새 20쌍 FLY-L({t}) − FLY-OS({t})", "FLYL", "FLYOS", u("FLYL", t, set_="new"),
                    u("FLYOS", t, set_="new"), spec, point=t) for t in new_points]
    paired = [paired_os(sits, spec, t) for t in sorted({int(r["point"]) for r in sits if r.get("mode") == "os_paired"})]
    ex = set(retried)
    sens = [contrast(f"민감도(재시도 마리 제외) {c['name']}", "FLYL", "FLYOS", u("FLYL", c["point"], exclude=ex),
                     u("FLYOS", c["point"], exclude=ex), spec, point=c["point"]) for c in prim]
    return dict(primary=prim, reference=ref, did=did, aux=aux, new_set=new, paired_os=paired, sensitivity=sens)


EVAL_ITEMS = ((2, "FLYL", "FLYOS"), (3, "FLYL", "FLYOS"), (3, "FLYL", "RND"), (3, "MAX", "FLYL"))
EVAL_TEXT = {2: "정보 있는 턴 일치율", 3: "승률"}


def eval_readings(result: dict, eval_records, nobrain_rows, spec) -> list:
    rows = result.get("per_fly") if result.get("status") == "COMPLETE" else None
    name = lambda c, a, b: f"평가 배틀 기준 {c}({EVAL_TEXT[c]}) {a} − {b}"
    if not rows:
        return [dict(criterion=c, name=name(c, a, b), a=a, b=b, status="측정 불성립",
                     reason="평가 블록 없음(실행 중 예산 소진 또는 미완료)") for c, a, b in EVAL_ITEMS]
    u2 = c2_units(eval_records, rows)
    u3 = c3_units(list(rows) + list(nobrain_rows))
    out = []
    for c, a, b in EVAL_ITEMS:
        units = u2 if c == 2 else u3
        out.append(dict(contrast(name(c, a, b), a, b, units.get(a, {}), units.get(b, {}), spec, crossed=False),
                        criterion=c))
    return out
