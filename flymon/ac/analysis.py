"""The stage-1 reading of M4 under AC (spec AC.5, AC.8). Nothing here is PASS / FAIL.
- Criterion 1 (primary): the switch rate at battle 40 (C-off: its point-0 rate, AC.2), FLY - C-off and FLY - FLY-TB,
  each "충족" iff the difference >= 0.15 and the 95% CI's lower end > 0. Units: the 20 pairs.
- Criterion 2: FLY - C-off information-turn type match in the evaluation battles (a fly-decided turn whose candidate
  multipliers are not all equal; 1 if the chosen move has the largest multiplier), "충족" iff > 0 with the CI above 0.
- Criterion 3: FLY - RND evaluation win rate, same rule; MAX - FLY is reported as the upper bound (descriptive, "기술",
  never judged); WEAK arms are the M1 pilot's values, copied.
- Criteria 4 and 5 need stage-2 arms: "미측정" (criterion 4 would be descriptive only, AC.5; never judged here).
- Every contrast uses boot.two_stage_diff (B = 10 000, seed 304) on valid flies only; an arm under its minimum of valid
  flies (AC.8), with no valid fly at all, or stopped (STOP_INFRA) makes the contrast "측정 불성립" before any bootstrap.
  No multiple-comparison correction (AC.0 8): criterion 1 is primary, everything else is descriptive.
- Fly ids: brain arms carry global ids on one pool (FLY 0-11, C-off 12-17, FLY-TB 18-23); RND / MAX carry arm-local
  ids 0..n-1, so criterion-3 units are keyed by (arm, k), never by the bare fly id.
- Invalid accounting: battle failures are counted by reason, excluding "arm_stopped" (a fly skipped because its arm
  had already stopped); a stopped arm's valid count is not read - its n_invalid at the stop is reported instead."""
from __future__ import annotations

from . import boot
from .spec import FORBIDDEN, LABEL, SPEC

RESULT = "results/summary/ac_m4_stage1.json"
ARM_TEXT = {"FLY": "FLY", "COFF": "C-off", "TB": "FLY-TB", "RND": "RND", "MAX": "MAX"}
SMALL_NOTE = "(6마리 팔의 CI는 불안정할 수 있다)"
MULTI_NOTE = "(기술 통계, 다중 비교 미보정)"


def _valid(rows) -> dict:
    return {r["fly"]: r["arm"] for r in rows if not r["invalid"]}


def c1_units(sit_records, rows, spec=SPEC) -> dict:
    valid, last, out = _valid(rows), max(spec.sit_points), {}
    for rec in sit_records:
        f = rec["fly"]
        if f not in valid:
            continue
        arm = valid[f]
        if rec["point"] == (0 if arm == "COFF" else last):
            out.setdefault(arm, {})[f] = list(rec["switched"])
    return out


def c2_units(eval_records, rows) -> dict:
    valid, out = _valid(rows), {}
    for rec in eval_records:
        f = rec.get("fly")
        if f not in valid or rec.get("kind") != "decision" or rec.get("decider") != "fly":
            continue
        cands, m, chosen = rec.get("candidates"), rec.get("multipliers"), rec.get("chosen")
        if not isinstance(cands, list) or not isinstance(m, list) or len(m) != len(cands) or chosen not in cands:
            continue
        m = [float(x) for x in m]
        if len(set(m)) < 2:
            continue
        out.setdefault(valid[f], {}).setdefault(f, []).append(int(m[cands.index(chosen)] == max(m)))
    return out


def c3_units(rows) -> dict:
    out = {}
    for r in rows:
        if not r["invalid"]:
            out.setdefault(r["arm"], {})[(r["arm"], r["k"])] = [int(b["won"] is True) for b in r["eval_battles"]]
    return out


def kc_ratio_by_arm(records, rows) -> dict:
    arm_of = {r["fly"]: r["arm"] for r in rows if not r["invalid"]}
    flags: dict = {a: [] for a in dict.fromkeys(arm_of.values())}
    for rec in records:
        f = rec.get("fly")
        if f in arm_of and rec.get("kind") == "decision" and rec.get("decider") == "fly" and "kc_ratio_gt2" in rec:
            flags[arm_of[f]].append(bool(rec["kc_ratio_gt2"]))
    return {a: (sum(v) / len(v) if v else None) for a, v in flags.items()}


def contrast(criterion: int, kind: str, a: str, b: str, units: dict, spec=SPEC, min_effect: float = 0.0,
             stopped=None) -> dict:
    mins, stopped = spec.min_valid_of(), dict(stopped or {})
    ua = {f: u for f, u in units.get(a, {}).items() if len(u)}
    ub = {f: u for f, u in units.get(b, {}).items() if len(u)}
    base = dict(criterion=criterion, kind=kind, a=a, b=b, n_a=len(ua), n_b=len(ub), min_a=mins[a], min_b=mins[b],
                min_effect=min_effect, stopped_a=a in stopped, stopped_b=b in stopped)
    if (a in stopped or b in stopped or not ua or not ub or len(ua) < mins[a] or len(ub) < mins[b]):
        return dict(base, status="측정 불성립")
    res = boot.two_stage_diff(ua, ub, spec.boot_draws, spec.boot_seed)
    res = {k: v for k, v in res.items() if k not in ("n_a", "n_b")}
    if kind == "upper":
        status = "기술"
    else:
        status = "충족" if boot.meets(res, min_effect) else "미충족"
    return dict(base, status=status, **res)


def _small(c, spec=SPEC) -> bool:
    sizes = spec.arm_sizes()
    return min(sizes[c["a"]], sizes[c["b"]]) <= 6


def sentence(c: dict, kc_fly, spec=SPEC) -> str:
    kc = "기록 없음" if kc_fly is None else f"{kc_fly:.3f}"
    tail = f" {LABEL} 같은 턴 후보 KC 비율 > 2 턴 비율(FLY): {kc}."
    if c["status"] == "미측정":
        return f"4.3 기준 {c['criterion']}: — 탐색 기준 미측정(2단계 팔이 필요하다).{tail}"
    who = f"{ARM_TEXT[c['a']]} − {ARM_TEXT[c['b']]}"
    if c["status"] == "측정 불성립":
        why = [f"{ARM_TEXT[c[s]]} STOP_INFRA" for s in ("a", "b") if c.get(f"stopped_{s}")]
        stop = f" ({', '.join(why)})" if why else ""
        body = (f"{who}에서 4.3 기준 {c['criterion']}: 유효 마리 {c['n_a']}/{c['min_a']} · {c['n_b']}/{c['min_b']}{stop} — "
                f"탐색 기준 측정 불성립.")
    else:
        head = f"{who}에서 4.3 기준 {c['criterion']}: {c['diff']:.3f} [95% CI {c['lo']:.3f}, {c['hi']:.3f}]"
        if c["status"] == "기술":
            body = f"{head} — 상한 보고(기술)."
        else:
            body = f"{head} — 탐색 기준 {c['status']}."
            if c["criterion"] != 1 and c["status"] == "충족":
                body += f" {MULTI_NOTE}"
        if _small(c, spec):
            body += f" {SMALL_NOTE}"
    return body + tail


def c1_summary(cs: list) -> str:
    st = [c["status"] for c in cs]
    if "측정 불성립" in st:
        s = "기준 1은 측정 불성립이다(최소 유효 마리 미달 또는 팔 STOP_INFRA)."
    elif all(x == "충족" for x in st):
        s = "배틀로 학습한 마리에서 상성 조건부 선택이 관찰됐다(탐색, 학습 중 같은 냄새를 봤을 수 있음)."
    else:
        s = "배틀로 학습한 마리의 상성 조건부 선택이 탐색 기준에 못 미쳤다."
    return f"{s} 확인 세트 상황은 학습 중에도 나왔을 수 있다(AC.4). {LABEL}"


def check_digests(results: dict) -> None:
    if any(r.get("mode") != "run" for r in results.values()):
        raise SystemExit("refusing: every stage-1 result must be mode 'run' (not smoke / bench)")
    if len({r.get("eval_digest") for r in results.values()}) != 1:
        raise SystemExit("refusing: the arms ran on different evaluation schedules (AC.4 digest gate)")


def _reason_kind(reason: str) -> str:
    return "unfinished_after_retries" if "unfinished after" in reason else reason


def invalid_accounting(rows, stopped: dict) -> tuple:
    """(n_valid, invalid_reasons, n_arm_stopped) per arm. A stopped arm's n_valid is None (read stopped[arm])."""
    n_valid, reasons, skipped = {}, {}, {}
    for r in rows:
        arm = r["arm"]
        n_valid.setdefault(arm, 0)
        reasons.setdefault(arm, {})
        skipped.setdefault(arm, 0)
        if not r["invalid"]:
            n_valid[arm] += 1
            continue
        reason = r.get("invalid_reason") or "unknown"
        if reason == "arm_stopped":
            skipped[arm] += 1
        else:
            k = _reason_kind(reason)
            reasons[arm][k] = reasons[arm].get(k, 0) + 1
    for arm in stopped:
        n_valid[arm] = None
    return n_valid, reasons, skipped


def analyse(results: dict, sit_records, eval_records, all_records, spec=SPEC, weak=None) -> dict:
    rows = [dict(r) for res in results.values() for r in res["per_fly"]]
    brain_rows = results["BRAIN"]["per_fly"]
    stopped = {}
    for res in results.values():
        stopped.update(res.get("book", {}).get("stopped", {}))
    u1, u2, u3 = c1_units(sit_records, brain_rows, spec), c2_units(eval_records, brain_rows), c3_units(rows)
    cs = [contrast(1, "primary", "FLY", "COFF", u1, spec, spec.min_effect, stopped),
          contrast(1, "primary", "FLY", "TB", u1, spec, spec.min_effect, stopped),
          contrast(2, "secondary", "FLY", "COFF", u2, spec, 0.0, stopped),
          contrast(3, "secondary", "FLY", "RND", u3, spec, 0.0, stopped),
          contrast(3, "upper", "MAX", "FLY", u3, spec, 0.0, stopped),
          dict(criterion=4, kind="specificity", status="미측정"),
          dict(criterion=5, kind="reversal", status="미측정")]
    kc = kc_ratio_by_arm(all_records, brain_rows)
    sentences = [sentence(c, kc.get("FLY"), spec) for c in cs]
    summary = c1_summary(cs[:2])
    for s in sentences + [summary]:
        bad = [w for w in FORBIDDEN if w in s]
        if bad:
            raise ValueError(f"forbidden words {bad} in a result sentence (AC.8): {s}")
    n_valid, reasons, skipped = invalid_accounting(rows, stopped)
    return dict(appendix="AC", stage=1, label=LABEL, criteria=cs, sentences=sentences, c1_summary=summary,
                kc_ratio_gt2=kc, n_valid=n_valid, invalid_reasons=reasons, n_arm_stopped=skipped, stopped=stopped,
                weak_pilot=weak, boot=dict(draws=spec.boot_draws, seed=spec.boot_seed),
                eval_digest=next(iter(results.values())).get("eval_digest"),
                notes=["다중 비교 보정 없음(AC.0 8): 기준 1만 1차, 나머지는 기술 통계.",
                       "6마리 팔의 CI는 불안정할 수 있다(AC.5).",
                       "기준 3은 학습의 확인 수단이 아니다(AC.0 5 · 7, 검정력 낮음).",
                       "기준 4 · 5는 2단계 팔이 필요해 1단계에서 미측정이다(기준 4는 측정되더라도 기술로만 적는다, AC.5).",
                       "MAX − FLY는 상한 보고(기술)이며 판정에 쓰지 않는다.",
                       "WEAK 팔은 M1 파일럿 값을 옮겨 적은 것이다(AC.2).",
                       "KC 폭주(D.6 (a))는 대응 없이 수용하고 재기만 했다(AC.0 6)."])
