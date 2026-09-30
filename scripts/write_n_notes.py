#!/usr/bin/env python3
"""Spec N.8.3 / N.8.6 / N.8.9 (plan reading 12): the dated N.8a paragraph (after N0f, from block n0f) and N.8b paragraph
(after N2.0, from blocks n2_0 and n1) of results/summary/n_real_odour.json. N0 / N1 / N2.0 (N.8a) and the N2 judge
(N.8b) refuse until the committed spec has the paragraph citing the block's run id (n_cli.spec_note).

    uv run python scripts/write_n_notes.py --which n8a [--date 2026-10-01]      # prints the paragraph
    uv run python scripts/write_n_notes.py --which n8a --write                  # puts it at the end of spec N.8
    uv run python scripts/write_n_notes.py --which n8b --write

Every number is the recorded block's or n_spec's. Refusals (exit 2, nothing printed or written):
- no summary, no block, a block without a run id or without a field the paragraph states (never a note from empty input);
- N.8a of STOP_DATA_MISMATCH (no measurement to state); STOP_NO_OPERATING_POINT is stated as such;
- N.8b of anything but N2_0_GO (no design n to fix), of a partial OC, of an OC without every declared scenario, or of a
  block n2_0 produced on another n1 run than the summary's;
- --write: a spec without the N.8 section, or one that already has the paragraph with other content (never overwritten;
  the same content again changes nothing)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from flymon.brain import n_store
from flymon.brain.n_cli import note_start, refuse
from flymon.brain.n_rules import N2_0_GO, OPERATING_POINT, STOP_NO_OPERATING_POINT, point_key, scenario_key
from flymon.brain.n_spec import SPEC

ROOT = Path(__file__).resolve().parents[1]
SECTION = "N.8"                                                  # the notes go at the end of this spec section
NOTES = {"n8a": ("N.8a", "n0f"), "n8b": ("N.8b", "n2_0")}        # which -> (marker, source block): the stages' `note`
COND_LABEL = {"on": "켬", "block": "APL→KC 차단", "all": "전 출력 차단"}
PAIRING_LABEL = {"same": "켬 팔과 같은 재표본", "independent": "독립 재표본"}


class Refusal(ValueError):
    """The block cannot carry the paragraph; main() turns it into exit 2."""


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def _block(b, what: str) -> dict:
    if not isinstance(b, dict) or not b:
        raise Refusal(f"block {what} is missing or empty")
    if not b.get("run_id"):
        raise Refusal(f"block {what} has no run_id to cite")
    return b


def n8a(b: dict, date: str, spec=SPEC) -> str:
    b = _block(b, "n0f")
    out, sel = b["outcome"], b.get("selected")
    if out not in (OPERATING_POINT, STOP_NO_OPERATING_POINT):
        raise Refusal(f"block n0f ended {out}: there is no N0f measurement for an N.8a paragraph")
    if (out == OPERATING_POINT) != bool(sel):
        raise Refusal(f"block n0f's outcome {out} and its selected point {sel} disagree")
    conds = [c for c, _ in spec.conditions]
    base = conds[0]                                              # the first declared condition is the APL-on one
    lab = lambda c: COND_LABEL.get(c, c)                         # noqa: E731
    lines = [f"**N.8a N0f 결과 ({date}, run `{b['run_id']}`, 기록 전용)** — outcome `{out}`."]
    if sel:
        key = point_key(sel["g"], sel["c_delta"])
        chk = b["checks"][key]
        lines.append(f"- 작동점: g = {sel['g']:g}, c_δ = {sel['c_delta']:g} (켬 판정 자극 KC 활성 중앙값 평균 "
                     f"{100 * chk['mean_on']:.2f}%, 목표 {100 * spec.kc_target:.2f}%).")
        for s, by in b["grid"][key].items():
            parts = [f"{lab(c)} KC {100 * by[c]['kc_frac_median']:.1f}% · >{spec.runaway_hz:g} Hz "
                     f"{by[c]['runaway_share']:.3f} · A0 {by[c]['A_zero_share']:.2f} · P0 {by[c]['P_zero_share']:.2f} · "
                     f"방출 상태 {by[c]['firing_share']:.2f} · APL {by[c]['apl_out_mean']:.4f}" for c in conds]
            lines.append(f"- {s}: " + " / ".join(parts))
        lines.append(f"- 상태 몫 플래그(켬과의 차이 > {spec.state_diff_flag:g}, 멈춤 아님): " + (", ".join(
            f"{f['stimulus']} {lab(f['condition'])} {f['on']:g} → {f['share']:g}" for f in b["state_flags"]) or "없음"))
        lines.append("- APL 출력 이동(조건 − 켬): " + "; ".join(
            f"{s} " + ", ".join(f"{lab(c)} {v[c]:+.4f}" for c in conds if c != base) for s, v in b["apl_shift"].items()))
        lines.append("- 잘린 억제량 / 불응기 상한 초과 채널: " + "; ".join(
            f"{s} {d['clipped_total_hz']:.1f} Hz / {', '.join(d['capped']) or '없음'}" for s, d in b["drives"].items()))
    else:
        lines.append(f"- 작동점 없음: 켬 대역·차단 폭주·판독 바닥 조건을 모두 만족하는 (g, c_δ)가 없다 ({out}). 격자 "
                     "전체는 블록 n0f의 grid·checks에 있다. 사용자 판단으로 넘긴다.")
    lt = b["lin_totals"]
    lines.append("- Lin 2014 총합 대조(기록): " + "; ".join(
        f"{conv} " + ", ".join(f"{k} {tot[k]:g} ({_pct(lt['rel_error'][conv][k])})" for k, _ in spec.lin_totals)
        for conv, tot in lt["totals"].items()))
    lines.append(f"- 벽시계: step당 {1000 * b['wall_s_per_step']:.1f} ms; 판정 블록 예상 "
                 + ", ".join(f"n={n} {h:.1f} h" for n, h in b["budget_estimate_h"].items())
                 + f" (한도 {spec.budget_h:g} h).")
    return "\n".join(lines) + "\n"


def n8b(b: dict, n1: dict, date: str, spec=SPEC) -> str:
    b = _block(b, "n2_0")
    if b["outcome"] != N2_0_GO:
        raise Refusal(f"block n2_0 ended {b['outcome']}: N.8b fixes the design n of a {N2_0_GO}; a STOP goes to the user")
    oc = b["oc"]
    if not isinstance(oc, dict) or not oc.get("rows"):
        raise Refusal("block n2_0 has no OC table")
    if oc["partial"]:
        raise Refusal("block n2_0's OC is partial (a subset of the declared pairings): it implies no design n")
    if not isinstance(b["n"], int):
        raise Refusal(f"block n2_0 has no design n ({b['n']!r})")
    if b["oc_boot"] != oc["boot"]:
        raise Refusal(f"block n2_0's oc_boot {b['oc_boot']} is not its OC's {oc['boot']}")
    up = (b.get("upstream") or {}).get("n1")
    if up and n1.get("run_id") and up != n1["run_id"]:
        raise Refusal(f"block n2_0 was produced on n1 run {up}; the summary holds {n1['run_id']}")
    if not b["pilot_seeds"]:
        raise Refusal("block n2_0 records no pilot seed")
    o = {k: n1["pairs"][k]["o"] for k, _, _ in spec.pairs}
    scen = [(p, m, scenario_key(p, m)) for p in spec.oc_pairings for m in spec.sd_mults]
    per = {k: b["scenario_n"][k] for _, _, k in scen}            # KeyError: a declared scenario is not in the block
    pairs = [k for k, _, _ in spec.pairs]
    both = lambda d, f="g": ", ".join(f"{k} {d[k]:{f}}" for k in pairs)  # noqa: E731
    lines = [f"**N.8b N2.0 보정 결과 ({date}, run `{b['run_id']}`)** — outcome `{b['outcome']}`.",
             f"- 파일럿(켬, {len(b['pilot_seeds'])} 시드): ℓ̂_on = {both(b['ell_hat'], '.4g')} "
             f"(sd {both(b['pilot_sd'], '.4g')}).",
             f"- N1 원단위 효과 o: {both(o)} → c₁ = {both(b['c1'])}; δ_min = {b['delta_min']:.4g}; "
             f"ε = {b['eps']:.4g}; 대립 D_sim = {b['alt_D_sim']:.4g}.",
             f"- n = {b['n']}; 판정 블록 예상 {b['budget_h']:.1f} h (한도 {spec.budget_h:g} h).",
             f"- 시나리오별 최소 n (귀무 ≤ {spec.null_max:g}, 검정력 ≥ {spec.power_min:g}): "
             + "; ".join(f"{k} n = {per[k] if per[k] is not None else '없음'} ({PAIRING_LABEL.get(p, p)}, 편차 {m:g}배)"
                         for p, m, k in scen)
             + f". 설계 n은 이 {len(scen)}개 시나리오를 동시에 통과하는 가장 작은 n이다."]
    for k in (k for p, m, k in scen if p == "same" and m == 1):
        lines.append(f"- 퇴화 시나리오 {k}: 차단 팔이 켬 팔에 상수를 더한 것이어서 D_sim의 CI가 폭 0이고, 조항 ②와 ③이 "
                     "이 시나리오에서는 구속하지 않는다.")
    lines += [f"- 부트스트랩 근사: 운영 특성은 모의 실험마다 부트스트랩 {b['oc_boot']}회를 썼고 판정은 "
              f"{b['judge_boot']}회를 쓴다. 판정 규칙의 CI에 대한 근사다.",
              f"- 운영 특성 (칸마다 모의 실험 {oc['draws']}회, P = P(SUPPORTED)):", "",
              "| 짝짓기 | 편차 배수 | 가설 | n | P | MC SE |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['pairing']} | {r['sd_mult']:g} | {r['hyp']} | {r['n']} | {r['p_supported']:.3f} | {r['mc_se']:.3f} |"
              for r in oc["rows"]]
    return "\n".join(lines) + "\n"


def write_note(path: Path, marker: str, text: str) -> bool:
    """Put `text` at the end of the spec's N.8 section. False (nothing written) if the spec already holds exactly this
    text; Refusal if it holds a `marker` paragraph with other content, or has no N.8 section."""
    doc = path.read_text(encoding="utf-8")
    if note_start(doc, marker) is not None:
        if text in doc:
            return False
        raise Refusal(f"{path} already has a {marker} paragraph with other content: it is not overwritten "
                      "(remove it by hand to write a new one)")
    m = re.search(rf"^###\s+{re.escape(SECTION)}(?![\w.])", doc, re.M)
    if not m:
        raise Refusal(f"{path} has no '### {SECTION}' section to put the {marker} paragraph in")
    nxt = re.compile(r"^#{1,3}\s", re.M).search(doc, m.end())
    end = nxt.start() if nxt else len(doc)
    rest = doc[end:]
    path.write_text(doc[:end].rstrip("\n") + "\n\n" + text + ("\n" + rest if rest else ""), encoding="utf-8")
    return True


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(ROOT / n_store.SUMMARY))
    ap.add_argument("--which", choices=tuple(NOTES), required=True)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--write", action="store_true", help="put the paragraph into --spec instead of printing it")
    ap.add_argument("--spec", default=str(ROOT / SPEC.spec_path))
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    marker, block = NOTES[a.which]
    try:
        dt.date.fromisoformat(a.date)
        doc = json.loads(Path(a.summary).read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise Refusal("the summary is not a JSON object")
        text = n8a(doc[block], a.date) if a.which == "n8a" else n8b(doc[block], _block(doc["n1"], "n1"), a.date)
        wrote = write_note(Path(a.spec), marker, text) if a.write else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return refuse(f"no {marker} paragraph from block {block} of {a.summary}: {e!r}")
    if wrote is None:
        print(text, end="")
    else:
        print(f"{marker}: " + (f"written into {a.spec}" if wrote else f"already in {a.spec} with this content; unchanged"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
