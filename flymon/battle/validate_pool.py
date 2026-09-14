"""Validate the pool: allow-list rules, distinct attack types, speed, and Showdown gen1ou legality.
Usage: uv run python -m flymon.battle.validate_pool [--write-docs]"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .moves import attack_allowed, gen1, support_allowed
from .pool import POOL, export_text
from .server import NODE_BIN, ShowdownServer

DOCS_POOL = Path(__file__).resolve().parents[2] / "docs" / "pool.md"

# Roster substitutions made because a proposed attack failed `validate-team gen1ou`
# (species, removed move, replacement move, reason). Regenerated into docs/pool.md.
SUBSTITUTIONS: list[tuple[str, str, str, str]] = []


def _legality_cell(legal: bool | None, msg: str) -> str:
    """Three states: checked-ok, checked-FAIL, and "node absent, nothing was checked"."""
    if legal is None:
        return "unchecked"
    return "ok" if legal else "FAIL: " + msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-docs", action="store_true")
    a = ap.parse_args(argv)
    node_ok = NODE_BIN.exists()
    if a.write_docs and not node_ok:
        print(f"refusing --write-docs: {NODE_BIN} is missing, so gen1ou legality cannot be checked "
              "(run scripts/install_showdown.sh first)")
        return 2
    srv = ShowdownServer()
    rows, failed = [], 0
    for mon in POOL:
        verdicts = []
        types = Counter(gen1(m).type for m in mon.attacks)
        for m in mon.attacks:
            ok, why = attack_allowed(gen1(m))
            if types[gen1(m).type] > 1:
                ok, why = False, "duplicate attack type"
            verdicts.append((m, "attack", ok, why)); failed += not ok
        for s in mon.support:
            ok, why = support_allowed(gen1(s)); verdicts.append((s, "support", ok, why)); failed += not ok
        if mon.base_speed > 100:
            verdicts.append(("(speed)", "rule", False, f"base speed {mon.base_speed} > 100")); failed += 1
        legal, msg = (None, "node missing: legality not checked")
        if node_ok:
            legal, msg = srv.validate_team(srv.pack_team(export_text(mon)))
            failed += not legal
        rows.append((mon, verdicts, legal, msg))
    lines = ["# 팀 풀 검증", "", "| 종족 | 속도 | 기술 | 역할 | 규칙 | 사유 | gen1ou 합법 |", "|---|---|---|---|---|---|---|"]
    for mon, verdicts, legal, msg in rows:
        for m, role, ok, why in verdicts:
            lines.append(f"| {mon.species} | {mon.base_speed} | {m} | {role} | {'ok' if ok else 'FAIL'} | {why} | {_legality_cell(legal, msg)} |")
    lines += ["", "## 변경 기록", ""]
    if SUBSTITUTIONS:
        lines += ["| 종족 | 제외된 기술 | 대체 기술 | 사유 |", "|---|---|---|---|"]
        lines += [f"| {sp} | {old} | {new} | {why} |" for sp, old, new, why in SUBSTITUTIONS]
    else:
        lines.append("없음 — 제안 로스터의 모든 기술이 `validate-team gen1ou`를 그대로 통과했다.")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.write_docs:
        DOCS_POOL.write_text(text)
    if not node_ok:
        print(f"WARNING: {NODE_BIN} is missing: gen1ou legality was not checked (reported as `unchecked`)")
    print(f"{'PASS' if failed == 0 else 'FAIL'}: {failed} problem(s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
