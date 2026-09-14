# FlyMon M1 — 배틀 환경 구현 계획 (Showdown 서버, poke-env 플레이어, 코치·라우터, 귀속 파서, 팀 풀, 뇌 없는 파일럿과 풀 게이트)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로컬 Pokémon Showdown 서버 위에서 poke-env 플레이어가 1세대 OU 규칙 6대6 배틀을 돌리되, 공격기 선택은 교체 가능한 "결정 제공자"(무작위·최대데미지, 나중에 초파리 뇌)가 하고, 코치가 전략을 맡으며, 원시 프로토콜에서 강화 신호를 귀속하는 파이프라인을 만든다. 뇌 없이 파일럿 4팔을 돌려 풀 게이트(MAX − RND 승률 ≥ 0.15)를 통과한다.

**Architecture:** Node `pokemon-showdown@0.11.11`(npm, MIT) 서버를 loopback에 띄우고 poke-env 0.16.1 `Player` 서브클래스 `FlyCoachPlayer`가 웹소켓으로 붙는다. 한 턴: 코치(`SimpleHeuristicsPlayer` 규칙 래핑)가 주문을 내고 → 라우터가 "공격 & 후보 ≥ 2"면 결정 제공자에게 후보를 넘기고 → 선택된 주문을 보낸다. `_handle_battle_message` 훅이 원시 프로토콜 줄을 배틀별로 모아 귀속 파서가 내 기술의 직접 피해·상성·기절을 계산한다. 배치 배리어는 여러 배틀의 초파리 차례를 모아 한 번에 결정 제공자를 호출한다(M3에서 뇌 스웜이 붙는다). 팀 풀은 Showdown 기술 데이터 필드로 허용 목록을 검증하고 Showdown의 `validate-team gen1ou`로 합법성을 확인한다.

**Tech Stack:** Python 3.13(uv), poke-env 0.16.1, Node 24 + pokemon-showdown 0.11.11(npm), pytest(+ pytest-asyncio), numpy.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` (v4, 부록 A 포함). 이 계획은 스펙 2절(아키텍처·배치 배리어), 3.2(배틀 환경), 3.4(귀속 파서), 4.1(RND/MAX/WEAK 팔), 5절 M1을 구현한다. M0 결과(부분 통과)는 스펙 부록 A와 `docs/handoffs/2026-09-14-m1-handoff.md`에 있다.

## Global Constraints

- Python 3.13, uv. 새 의존성: `poke-env>=0.16.1,<0.17`, dev `pytest-asyncio>=0.24`. Node 24. `flymon/battle/package.json`에 `pokemon-showdown` **0.11.11 고정**(`npm ci`).
- 서버는 항상 `bindaddress 127.0.0.1`, `--no-security`, 포트는 인자(기본 8000). 시작 스크립트가 `config/config.js`를 만들고 `logs/repl` 디렉터리를 만든다(npm 패키지는 이 디렉터리가 없어 시작 시 죽는다 — 확인됨).
- 포맷 `gen1ou`. 팀은 Showdown export 텍스트로 넘긴다(`Player(team=str)`). 1세대는 아이템·특성이 없다.
- 초파리 후보 공격기 허용 규칙(스펙 3.2, `poke_env.battle.Move(id, gen=1).entry`의 Showdown 필드로 판정): `category ∈ {Physical, Special}`, `basePower > 0`, `accuracy ≥ 90`(True는 100으로), `critRatio` 없음 또는 1, `flags`에 `charge`·`recharge` 없음, `recoil` 없음, `selfdestruct` 없음, `volatileStatus` 없음(부분 구속·잠금·Bide 제외), `damage` 필드 없음, `secondary`가 없거나 `{status ∈ {par, brn, psn}, chance ≤ 10}` 또는 `{boosts(하락), chance ≤ 33}`; `secondary`에 `volatileStatus`(혼란·풀죽음)나 `status ∈ {frz, slp}`가 있으면 거부. 같은 마리의 공격기는 타입이 서로 다르다. 종족 속도 ≤ 100.
- 코치용 보조기 허용: Thunder Wave, Reflect, Light Screen, Recover, Soft-Boiled, Amnesia, Swords Dance, Agility. 거부: Substitute, Toxic, 혼란·잠듦·구속·폭발·Rest.
- 팀 합법성의 최종 권위는 `node pokemon-showdown validate-team gen1ou`(종료 코드 0). 계획의 로스터는 **제안**이며, 검증에 실패한 기술은 허용 목록 안의 다른 기술로 바꾼다(변경 내역을 `docs/pool.md`에 기록).
- 배틀 일정은 결정적(시드)이지만 서버 난수는 고정되지 않는다(스펙 0절). 팔 간 비교는 같은 일정 파일 기준.
- 결과는 `results/m1/`(git-ignored)에, 게이트 요약은 `results/summary/m1_pilot.json`(커밋)에 쓴다. **서브에이전트는 results/ 아래에 쓰지 않는다**(파일럿 실행은 컨트롤러가 한다).
- 커밋은 태스크마다. 메시지 `feat(battle): …`, `test(battle): …`, `chore: …`.

---

## 검증된 API 사실 (2026-09-14, poke-env 0.16.1 / pokemon-showdown 0.11.11)

- `Player.__init__(account_configuration, avatar, battle_format, log_level, max_concurrent_battles, accept_open_team_sheet, save_replays, server_configuration, start_timer_on_battle_start, start_listening, open_timeout, ping_interval, ping_timeout, loop, team, strict_battle_tracking)`. `choose_move`는 코루틴이어도 된다(`_handle_battle_request`가 await). `battle_against(opponent, n_battles)`, `send_challenges(opponent, n_challenges)`, `accept_challenges(opponent, n_challenges, packed_team=None)`.
- 원시 프로토콜: `Player._handle_battle_message(self, split_messages: List[List[str]])`를 오버라이드해 가로챈다. `split_messages[0][0]`가 배틀 태그, 이후 각 항목은 `|`로 나눈 한 줄(예: `['', '-damage', 'p2a: Venusaur', '99/300']`). poke-env 0.16에는 `Battle.observations`가 **없다**.
- `Battle(battle_tag, username, logger, gen=1)`을 오프라인으로 만들고 `parse_message([...])`·`parse_request(dict)`로 상태를 채울 수 있다(테스트에 사용). `battle.available_moves`(Move: `id, base_power, type, category, accuracy, entry`), `available_switches`, `active_pokemon`, `opponent_active_pokemon`(`species, types, current_hp_fraction, damage_multiplier(move)`), `force_switch`, `turn`, `finished`, `won`.
- `SimpleHeuristicsPlayer.choose_singles_move(battle) -> (order, score)`는 정적 메서드라 어느 Battle에나 쓸 수 있다. 공격 점수 = `base_power × 자속 1.5 × (물리/특수 스탯비) × accuracy × 상성`. 교체 판단 `_should_switch_out(battle)`.
- `Move(id, gen=1).entry`가 Showdown 기술 dict(`flags`, `secondary`, `selfdestruct`, `volatileStatus`, `damage`, `critRatio`, `accuracy`)를 그대로 준다. 예: Explosion `selfdestruct: 'always'`, Wrap `volatileStatus: 'partiallytrapped'`, Thrash `self.volatileStatus: 'lockedmove'`(→ `self` 안도 검사), Hyper Beam `flags.recharge`, Fly `flags.charge`, Slash `critRatio 2`, Body Slam `secondary {chance 30, status par}`, Seismic Toss `damage: 'level'`.
- Showdown CLI: `node pokemon-showdown pack-team < export.txt`(패킹), `node pokemon-showdown validate-team gen1ou < packed`(종료 코드 0/1, 위반 사유 stdout).
- 서버: `node node_modules/pokemon-showdown/pokemon-showdown start --no-security --port N`; 첫 실행에 `config/config.js`를 자동 생성하지만 `logs/repl`이 없으면 죽는다. 기동 후 `http://127.0.0.1:N/`이 HTML을 준다. `LocalhostServerConfiguration = ('ws://localhost:8000/showdown/websocket', 'https://play.pokemonshowdown.com/action.php?')`; 다른 포트는 `ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")`.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `flymon/battle/package.json`, `flymon/battle/package-lock.json` | pokemon-showdown 0.11.11 고정 |
| `scripts/install_showdown.sh` | `npm ci`, `logs/repl` 생성 |
| `flymon/battle/server.py` | `ShowdownServer`: config.js 생성(loopback, port), 기동·대기·종료, `pack_team`, `validate_team` |
| `flymon/battle/moves.py` | 허용 규칙 `attack_allowed(move)`, `support_allowed(move)`, `is_attack(move)` |
| `flymon/battle/pool.py` | `PoolMon`, `POOL`(16마리), `export_text(mon)`, `team_export(mons)` |
| `flymon/battle/validate_pool.py` | CLI: 규칙 + Showdown 검증, 기술별 사유 출력, `docs/pool.md` 갱신용 표 |
| `flymon/battle/schedule.py` | 배틀 일정 생성/저장/로드 |
| `flymon/battle/coach.py` | `Coach`(v1 SimpleHeuristics 규칙, `weak`), `CoachDecision` |
| `flymon/battle/router.py` | 후보 추출, 결정 주체 판정 |
| `flymon/battle/providers.py` | `DecisionProvider` 인터페이스, `RandomProvider`, `MaxDamageProvider` |
| `flymon/battle/attribution.py` | 원시 프로토콜 블록 → `Outcome` |
| `flymon/battle/barrier.py` | `BatchBarrier` |
| `flymon/battle/fly_coach_player.py` | `FlyCoachPlayer` |
| `flymon/battle/opponents.py` | 상대 플레이어 팩토리 |
| `scripts/pilot_no_brain.py` | 파일럿 4팔 실행, `results/m1/*.json`, `results/summary/m1_pilot.json`, 게이트 |
| `tests/battle/*.py` | 태스크별 테스트; `tests/battle/conftest.py`에 오프라인 Battle 빌더와 서버 픽스처 |
| `docs/pool.md` | 로스터와 검증 결과 |

---

### Task 1: Showdown 서버 도구와 의존성

**Files:**
- Create: `flymon/battle/__init__.py`, `flymon/battle/package.json`, `scripts/install_showdown.sh`, `flymon/battle/server.py`
- Modify: `pyproject.toml`(poke-env, pytest-asyncio), `.gitignore`(`flymon/battle/node_modules/`, `flymon/battle/logs/`, `flymon/battle/config/config.js`)
- Test: `tests/battle/test_server.py`

**Interfaces:**
- Produces `flymon.battle.server.ShowdownServer(root: Path = PKG_ROOT, port: int = 8000)` with `install()`(npm ci + logs/repl), `write_config()`, `start(timeout_s=30)`, `stop()`, `is_up() -> bool`, `url_ws`, `pack_team(export_text) -> str`, `validate_team(packed, format_id="gen1ou") -> tuple[bool, str]`, context manager. `PKG_ROOT = Path(__file__).parent` (package.json 위치). `NODE_BIN = PKG_ROOT / "node_modules/pokemon-showdown/pokemon-showdown"`.

- [ ] **Step 1: 의존성과 package.json**

`pyproject.toml` dependencies에 `"poke-env>=0.16.1,<0.17"`, dev에 `"pytest-asyncio>=0.24"` 추가, `[tool.pytest.ini_options]`에 `asyncio_mode = "auto"` 추가. `uv sync`.

`flymon/battle/package.json`:
```json
{
  "name": "flymon-battle",
  "private": true,
  "description": "Pinned Pokémon Showdown server for FlyMon (MIT)",
  "dependencies": { "pokemon-showdown": "0.11.11" }
}
```
`cd flymon/battle && npm install --no-audit --no-fund`로 lock을 만들고 `package-lock.json`을 커밋한다. `.gitignore`에 위 세 경로 추가.

- [ ] **Step 2: 설치 스크립트** — `scripts/install_showdown.sh`

```bash
#!/usr/bin/env bash
# Install the pinned Pokémon Showdown server for FlyMon (loopback only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)/flymon/battle"
cd "$ROOT"
npm ci --no-audit --no-fund
mkdir -p node_modules/pokemon-showdown/logs/repl
echo "installed pokemon-showdown $(node -e "console.log(require('./node_modules/pokemon-showdown/package.json').version)")"
```

- [ ] **Step 3: 실패하는 테스트** — `tests/battle/test_server.py`

```python
import socket
from pathlib import Path

import pytest

from flymon.battle.server import NODE_BIN, ShowdownServer

pytestmark = pytest.mark.skipif(not NODE_BIN.exists(), reason="run scripts/install_showdown.sh first")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_write_config_is_loopback(tmp_path):
    srv = ShowdownServer(port=8123)
    text = srv.write_config()
    assert "exports.bindaddress = '127.0.0.1'" in text
    assert "exports.port = 8123" in text


def test_start_stop_and_health():
    port = _free_port()
    with ShowdownServer(port=port) as srv:
        assert srv.is_up()
        assert srv.url_ws == f"ws://127.0.0.1:{port}/showdown/websocket"
    assert not srv.is_up()


def test_pack_and_validate_team():
    srv = ShowdownServer(port=_free_port())
    packed = srv.pack_team("Blastoise\n- Surf\n- Earthquake\n- Strength\n- Reflect\n")
    assert packed.startswith("Blastoise||||Surf,Earthquake,Strength,Reflect")
    ok, msg = srv.validate_team(packed)
    assert ok, msg
    bad = srv.pack_team("Blastoise\n- Surf\n- Fire Blast\n")
    ok, msg = srv.validate_team(bad)
    assert not ok and "Fire Blast" in msg
```

- [ ] **Step 4: 실패 확인** — `uv run pytest tests/battle/test_server.py -v` → ImportError (설치 전에는 skip).

- [ ] **Step 5: server.py 구현**

```python
"""Local Pokémon Showdown server (pinned npm package), loopback only."""
from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

PKG_ROOT = Path(__file__).parent
NODE_BIN = PKG_ROOT / "node_modules" / "pokemon-showdown" / "pokemon-showdown"
CONFIG_DIR = PKG_ROOT / "node_modules" / "pokemon-showdown" / "config"
LOGS_REPL = PKG_ROOT / "node_modules" / "pokemon-showdown" / "logs" / "repl"

CONFIG_TEMPLATE = """'use strict';
exports.port = {port};
exports.bindaddress = '127.0.0.1';
exports.workers = 1;
exports.ssl = null;
exports.proxyip = false;
exports.loginserver = 'https://play.pokemonshowdown.com/';
exports.serverid = 'flymon';
exports.servertoken = '';
exports.repl = false;
exports.crashguard = true;
exports.reportjoins = false;
exports.noipchecks = true;
exports.emergency = false;
exports.potd = '';
exports.allowrequestingties = true;
exports.forcetimer = false;
exports.simulatorprocesses = 1;
exports.validatorprocesses = 1;
"""


class ShowdownServer:
    def __init__(self, root: Path = PKG_ROOT, port: int = 8000):
        self.root, self.port = Path(root), int(port)
        self.proc: subprocess.Popen | None = None

    @property
    def url_ws(self) -> str:
        return f"ws://127.0.0.1:{self.port}/showdown/websocket"

    @property
    def url_http(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def install(self) -> None:
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=self.root, check=True)
        LOGS_REPL.mkdir(parents=True, exist_ok=True)

    def write_config(self) -> str:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        text = CONFIG_TEMPLATE.format(port=self.port)
        (CONFIG_DIR / "config.js").write_text(text)
        return text

    def is_up(self) -> bool:
        try:
            with urllib.request.urlopen(self.url_http, timeout=1) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, timeout_s: float = 30.0) -> "ShowdownServer":
        if not NODE_BIN.exists():
            raise FileNotFoundError(f"{NODE_BIN} missing: run scripts/install_showdown.sh")
        LOGS_REPL.mkdir(parents=True, exist_ok=True)
        self.write_config()
        self.proc = subprocess.Popen(
            ["node", str(NODE_BIN), "start", "--no-security", "--port", str(self.port)],
            cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.is_up():
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"showdown server exited with {self.proc.returncode}")
            time.sleep(0.25)
        self.stop()
        raise TimeoutError("showdown server did not come up")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- team tooling (no server needed) --------------------------------------------
    def pack_team(self, export_text: str) -> str:
        out = subprocess.run(["node", str(NODE_BIN), "pack-team"], input=export_text, text=True,
                             capture_output=True, check=True)
        return out.stdout.strip()

    def validate_team(self, packed: str, format_id: str = "gen1ou") -> tuple[bool, str]:
        out = subprocess.run(["node", str(NODE_BIN), "validate-team", format_id], input=packed + "\n", text=True,
                             capture_output=True)
        return out.returncode == 0, (out.stdout + out.stderr).strip()
```

`exports.repl = false`가 무시되는 버전을 대비해 `LOGS_REPL.mkdir`를 항상 한다. 서버 stdout/stderr는 버린다(문제가 있으면 `stdout=None`으로 바꿔 본다).

- [ ] **Step 6: 설치·테스트** — `bash scripts/install_showdown.sh && uv run pytest tests/battle/test_server.py -v` → 3 passed.

- [ ] **Step 7: 커밋** — `chore(battle): pinned Showdown server tooling, poke-env dependency`

---

### Task 2: 기술 허용 규칙과 팀 풀

**Files:**
- Create: `flymon/battle/moves.py`, `flymon/battle/pool.py`, `flymon/battle/validate_pool.py`, `docs/pool.md`
- Test: `tests/battle/test_moves.py`, `tests/battle/test_pool.py`

**Interfaces:**
- `moves.attack_allowed(move: Move) -> tuple[bool, str]`, `moves.support_allowed(move) -> tuple[bool, str]`, `moves.is_attack(move) -> bool`(카테고리 Physical/Special & basePower > 0 & damage 없음), `moves.gen1(move_id) -> Move`.
- `pool.PoolMon(species: str, attacks: tuple[str, ...], support: tuple[str, ...], base_speed: int)`, `pool.POOL: list[PoolMon]`, `pool.export_text(mon) -> str`, `pool.team_export(mons: list[PoolMon]) -> str`, `pool.by_species: dict`.
- `validate_pool.main()`: 규칙·유일성·속도·Showdown 검증을 돌리고 마리·기술별 표를 stdout과 `docs/pool.md`에 쓴다. 종료 코드 1이면 하나라도 실패.

- [ ] **Step 1: 실패하는 테스트** — `tests/battle/test_moves.py`

```python
import pytest

from flymon.battle.moves import attack_allowed, gen1, is_attack, support_allowed


@pytest.mark.parametrize("mid,ok,why", [
    ("surf", True, ""), ("earthquake", True, ""), ("strength", True, ""),
    ("thunderbolt", True, ""),            # par 10% — mild secondary allowed
    ("psychic", True, ""),                # 33% special drop — stat-drop secondary allowed
    ("bodyslam", False, "secondary"),     # par 30%
    ("blizzard", False, "secondary"),     # frz
    ("explosion", False, "selfdestruct"),
    ("wrap", False, "volatileStatus"),
    ("thrash", False, "volatileStatus"),
    ("hyperbeam", False, "recharge"),
    ("fly", False, "charge"),
    ("slash", False, "critRatio"),
    ("seismictoss", False, "damage"),
    ("doubleedge", False, "recoil"),
    ("megakick", False, "accuracy"),
    ("stomp", False, "secondary"),        # flinch
    ("thunderwave", False, "category"),
])
def test_attack_allowed(mid, ok, why):
    allowed, reason = attack_allowed(gen1(mid))
    assert allowed is ok, (mid, reason)
    if not ok:
        assert why in reason


@pytest.mark.parametrize("mid,ok", [("thunderwave", True), ("reflect", True), ("softboiled", True), ("amnesia", True),
                                    ("swordsdance", True), ("agility", True), ("substitute", False), ("toxic", False),
                                    ("rest", False), ("confuseray", False), ("sleeppowder", False), ("surf", False)])
def test_support_allowed(mid, ok):
    assert support_allowed(gen1(mid))[0] is ok


def test_is_attack():
    assert is_attack(gen1("surf")) and not is_attack(gen1("thunderwave")) and not is_attack(gen1("seismictoss"))
```

`tests/battle/test_pool.py`:

```python
from collections import Counter

from flymon.battle.moves import attack_allowed, gen1, support_allowed
from flymon.battle.pool import POOL, export_text, team_export


def test_pool_size_and_shape():
    assert len(POOL) == 16
    for mon in POOL:
        assert 2 <= len(mon.attacks) <= 3 and 0 <= len(mon.support) <= 2
        assert 3 <= len(mon.attacks) + len(mon.support) <= 4
        assert mon.base_speed <= 100


def test_every_attack_allowed_and_types_distinct():
    for mon in POOL:
        types = [gen1(m).type for m in mon.attacks]
        assert len(set(types)) == len(types), (mon.species, types)
        for m in mon.attacks:
            ok, why = attack_allowed(gen1(m))
            assert ok, (mon.species, m, why)
        for s in mon.support:
            assert support_allowed(gen1(s))[0], (mon.species, s)


def test_attack_type_budget():
    types = {gen1(m).type for mon in POOL for m in mon.attacks}
    assert len(types) <= 12


def test_export_text_format():
    mon = POOL[0]
    text = export_text(mon)
    assert text.splitlines()[0] == mon.species
    assert all(line.startswith("- ") for line in text.splitlines()[1:])
    team = team_export(POOL[:6])
    assert team.count("\n\n") == 5
```

- [ ] **Step 2: 실패 확인** — ImportError.

- [ ] **Step 3: moves.py 구현**

```python
"""Which Gen 1 moves the fly may choose among (attacks) and which the coach may use (support)."""
from __future__ import annotations

from functools import lru_cache

from poke_env.battle import Move

SUPPORT_ALLOWED = {"thunderwave", "reflect", "lightscreen", "recover", "softboiled", "amnesia", "swordsdance", "agility"}
BAD_SECONDARY_STATUS = {"frz", "slp"}
MILD_STATUS = {"par", "brn", "psn"}


@lru_cache(maxsize=None)
def gen1(move_id: str) -> Move:
    return Move(move_id, gen=1)


def is_attack(move: Move) -> bool:
    e = move.entry
    return e.get("category") in ("Physical", "Special") and e.get("basePower", 0) > 0 and "damage" not in e


def attack_allowed(move: Move) -> tuple[bool, str]:
    e = move.entry
    if e.get("category") not in ("Physical", "Special"):
        return False, "category: not an attack"
    if e.get("basePower", 0) <= 0 or "damage" in e:
        return False, "damage: fixed/level damage or zero power"
    acc = e.get("accuracy", 100)
    acc = 100 if acc is True else acc
    if acc < 90:
        return False, f"accuracy {acc} < 90"
    if e.get("critRatio", 1) != 1:
        return False, f"critRatio {e['critRatio']}"
    flags = e.get("flags", {})
    for f in ("charge", "recharge"):
        if f in flags:
            return False, f"flags: {f}"
    if e.get("recoil"):
        return False, "recoil"
    if e.get("selfdestruct"):
        return False, "selfdestruct"
    if e.get("volatileStatus") or e.get("self", {}).get("volatileStatus"):
        return False, "volatileStatus: trapping/locking/bide"
    sec = e.get("secondary")
    if sec:
        if sec.get("volatileStatus"):
            return False, f"secondary volatileStatus {sec['volatileStatus']}"
        st = sec.get("status")
        if st in BAD_SECONDARY_STATUS:
            return False, f"secondary status {st}"
        if st in MILD_STATUS and sec.get("chance", 100) > 10:
            return False, f"secondary status {st} chance {sec.get('chance')} > 10"
        if "boosts" in sec and sec.get("chance", 100) > 33:
            return False, f"secondary boosts chance {sec.get('chance')} > 33"
        if st not in MILD_STATUS and "boosts" not in sec:
            return False, f"secondary: {sec}"
    return True, ""


def support_allowed(move: Move) -> tuple[bool, str]:
    if is_attack(move):
        return False, "attack"
    return (move.id in SUPPORT_ALLOWED), ("" if move.id in SUPPORT_ALLOWED else "not in support allow-list")
```

- [ ] **Step 4: pool.py 구현 (제안 로스터 — 검증기가 최종 권위)**

```python
"""The 16-Pokémon Gen 1 OU-legal team pool. Attacks are the fly's candidates; support is coach-only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoolMon:
    species: str
    attacks: tuple
    support: tuple
    base_speed: int


POOL = [
    PoolMon("Blastoise", ("Surf", "Earthquake", "Strength"), ("Reflect",), 78),
    PoolMon("Snorlax", ("Earthquake", "Surf", "Strength"), ("Amnesia",), 30),
    PoolMon("Exeggutor", ("Psychic", "Mega Drain"), ("Reflect",), 55),
    PoolMon("Chansey", ("Thunderbolt", "Psychic"), ("Soft-Boiled", "Thunder Wave"), 50),
    PoolMon("Rhydon", ("Earthquake", "Rock Slide", "Strength"), (), 40),
    PoolMon("Slowbro", ("Surf", "Psychic"), ("Amnesia", "Thunder Wave"), 30),
    PoolMon("Lapras", ("Surf", "Psychic", "Thunderbolt"), (), 60),
    PoolMon("Venusaur", ("Mega Drain", "Cut"), ("Swords Dance",), 80),
    PoolMon("Charizard", ("Flamethrower", "Earthquake", "Strength"), ("Swords Dance",), 100),
    PoolMon("Nidoking", ("Earthquake", "Thunderbolt", "Surf"), (), 85),
    PoolMon("Kangaskhan", ("Earthquake", "Surf", "Strength"), (), 90),
    PoolMon("Hypno", ("Psychic", "Tri Attack"), ("Thunder Wave", "Reflect"), 67),
    PoolMon("Dragonite", ("Surf", "Thunderbolt"), ("Agility",), 80),
    PoolMon("Machamp", ("Earthquake", "Rock Slide", "Strength"), (), 55),
    PoolMon("Tentacruel", ("Surf", "Mega Drain"), ("Swords Dance",), 100),
    PoolMon("Poliwrath", ("Surf", "Earthquake", "Psychic"), ("Amnesia",), 70),
]
by_species = {m.species: m for m in POOL}


def export_text(mon: PoolMon) -> str:
    return "\n".join([mon.species] + [f"- {m}" for m in mon.attacks + mon.support])


def team_export(mons: list) -> str:
    return "\n\n".join(export_text(m) for m in mons)
```

- [ ] **Step 5: validate_pool.py 구현**

```python
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-docs", action="store_true")
    a = ap.parse_args()
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
        legal, msg = (True, "node missing: legality not checked")
        if NODE_BIN.exists():
            legal, msg = srv.validate_team(srv.pack_team(export_text(mon)))
            failed += not legal
        rows.append((mon, verdicts, legal, msg))
    lines = ["# 팀 풀 검증", "", "| 종족 | 속도 | 기술 | 역할 | 규칙 | 사유 | gen1ou 합법 |", "|---|---|---|---|---|---|---|"]
    for mon, verdicts, legal, msg in rows:
        for m, role, ok, why in verdicts:
            lines.append(f"| {mon.species} | {mon.base_speed} | {m} | {role} | {'ok' if ok else 'FAIL'} | {why} | {'ok' if legal else 'FAIL: ' + msg} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.write_docs:
        Path("docs/pool.md").write_text(text)
    print(f"{'PASS' if failed == 0 else 'FAIL'}: {failed} problem(s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 실행·수정 루프** — `uv run pytest tests/battle/test_moves.py tests/battle/test_pool.py -v`가 통과할 때까지 규칙/로스터를 맞추고, `uv run python -m flymon.battle.validate_pool --write-docs`가 `PASS`가 될 때까지 검증 실패 기술을 허용 목록 안의 다른 기술로 교체한다(예: Hypno의 Tri Attack이 불법이면 Psychic + Thunder Wave + Reflect 3기술로). 교체 내역은 `docs/pool.md` 아래 "변경 기록" 절에 적는다.

- [ ] **Step 7: 커밋** — `feat(battle): move allow-list, 16-mon team pool, validator (docs/pool.md)`

---

### Task 3: 배틀 일정 생성기

**Files:**
- Create: `flymon/battle/schedule.py`
- Test: `tests/battle/test_schedule.py`

**Interfaces:**
- `Battle plan` dataclass `ScheduledBattle(battle_id: str, fly_id: int, my_team: list[str], opp_team: list[str], opponent: str)`; `make_schedule(n_flies, n_battles, opponent="heuristic", seed=0) -> list[ScheduledBattle]`; `save(path, sched)`, `load(path)`. 팀 6마리는 풀에서 비복원 추출, 순서 포함(선봉 = 첫 원소). `battle_id = f"f{fly:02d}-b{idx:03d}"`.

- [ ] **Step 1: 실패하는 테스트**

```python
from flymon.battle.pool import by_species
from flymon.battle.schedule import load, make_schedule, save


def test_schedule_is_deterministic_and_well_formed(tmp_path):
    a = make_schedule(n_flies=3, n_battles=4, seed=7)
    b = make_schedule(n_flies=3, n_battles=4, seed=7)
    assert [x.__dict__ for x in a] == [x.__dict__ for x in b]
    assert len(a) == 12
    for sb in a:
        assert len(sb.my_team) == 6 and len(set(sb.my_team)) == 6 and all(s in by_species for s in sb.my_team)
        assert len(sb.opp_team) == 6 and len(set(sb.opp_team)) == 6
        assert sb.opponent == "heuristic"
    assert make_schedule(3, 4, seed=8)[0].my_team != a[0].my_team
    p = tmp_path / "s.json"; save(p, a)
    assert [x.__dict__ for x in load(p)] == [x.__dict__ for x in a]
```

- [ ] **Step 2: 실패 확인** — ImportError.

- [ ] **Step 3: schedule.py 구현**

```python
"""Deterministic battle schedules shared by every experimental arm."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .pool import POOL


@dataclass
class ScheduledBattle:
    battle_id: str
    fly_id: int
    my_team: list          # 6 species, index 0 leads
    opp_team: list
    opponent: str          # "heuristic" | "random"


def make_schedule(n_flies: int, n_battles: int, opponent: str = "heuristic", seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    species = [m.species for m in POOL]
    out = []
    for fly in range(n_flies):
        for b in range(n_battles):
            mine = [species[i] for i in rng.choice(len(species), 6, replace=False)]
            opp = [species[i] for i in rng.choice(len(species), 6, replace=False)]
            out.append(ScheduledBattle(f"f{fly:02d}-b{b:03d}", fly, mine, opp, opponent))
    return out


def save(path: str | Path, sched: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(s) for s in sched], indent=1))


def load(path: str | Path) -> list:
    return [ScheduledBattle(**d) for d in json.loads(Path(path).read_text())]
```

- [ ] **Step 4: 통과 확인·커밋** — `feat(battle): deterministic battle schedule`.

---

### Task 4: 코치와 결정 제공자

**Files:**
- Create: `flymon/battle/coach.py`, `flymon/battle/providers.py`, `tests/battle/conftest.py`
- Test: `tests/battle/test_coach.py`, `tests/battle/test_providers.py`

**Interfaces:**
- `coach.CoachDecision(order: BattleOrder, kind: str ∈ {"attack", "support", "switch", "default"}, move: Move | None)`.
- `coach.Coach(weak: bool = False).decide(battle) -> CoachDecision`: v1 = `SimpleHeuristicsPlayer.choose_singles_move(battle)`의 주문을 분류. `weak=True`: 강제 교체(`battle.force_switch`)가 아니면 절대 교체하지 않고 보조기도 쓰지 않는다 — `available_moves` 중 공격기 점수(코치와 같은 식) 최대를 고른다.
- `providers.DecisionProvider` 프로토콜: `async def decide(self, battle, candidates: list[Move], context: dict) -> int`(후보 인덱스). `RandomProvider(seed)`, `MaxDamageProvider()`(코치와 같은 점수식 = MAX 팔).
- `tests/battle/conftest.py`: `make_battle(my: PoolMon, opp_species: str, opp_types_known=True, turn=1) -> Battle` — `Battle("battle-gen1ou-t", "p1", logger, gen=1)` + `parse_message(["", "switch", f"p2a: {opp}", f"{opp}, L100", "300/300"])` + `parse_request(request dict built from the PoolMon)`.

- [ ] **Step 1: 실패하는 테스트** — `tests/battle/test_coach.py`

```python
from flymon.battle.coach import Coach
from flymon.battle.pool import by_species


def test_coach_attacks_with_best_move_in_good_matchup(make_battle):
    b = make_battle(by_species["Blastoise"], "Charizard")
    d = Coach().decide(b)
    assert d.kind == "attack" and d.move.id == "surf"      # water vs fire


def test_weak_coach_never_switches_or_supports(make_battle):
    b = make_battle(by_species["Chansey"], "Machamp")       # bad matchup, switches available
    d = Coach(weak=True).decide(b)
    assert d.kind == "attack"
```

`tests/battle/test_providers.py`:

```python
import asyncio

from flymon.battle.moves import gen1
from flymon.battle.pool import by_species
from flymon.battle.providers import MaxDamageProvider, RandomProvider


def test_random_provider_is_seeded(make_battle):
    b = make_battle(by_species["Lapras"], "Venusaur")
    cands = [gen1(m) for m in ("Surf", "Psychic", "Thunderbolt")]
    a = [asyncio.run(RandomProvider(3).decide(b, cands, {})) for _ in range(10)]
    c = [asyncio.run(RandomProvider(3).decide(b, cands, {})) for _ in range(10)]
    assert a == c and set(a) <= {0, 1, 2}


def test_max_damage_provider_prefers_super_effective(make_battle):
    b = make_battle(by_species["Lapras"], "Venusaur")
    cands = [gen1(m) for m in ("Surf", "Psychic", "Thunderbolt")]
    assert asyncio.run(MaxDamageProvider().decide(b, cands, {})) == 1   # psychic vs grass/poison
```

- [ ] **Step 2: conftest 빌더 구현**

```python
import logging

import pytest
from poke_env.battle import Battle

from flymon.battle.moves import gen1
from flymon.battle.pool import POOL, PoolMon

_STATS = {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}


def _request(my: PoolMon, bench: list) -> dict:
    ids = [gen1(m).id for m in my.attacks + my.support]
    active = {"moves": [{"move": m, "id": gen1(m).id, "pp": 16, "maxpp": 16, "target": "normal", "disabled": False}
                        for m in my.attacks + my.support]}
    side = [{"ident": f"p1: {my.species}", "details": f"{my.species}, L100", "condition": "300/300", "active": True,
             "stats": _STATS, "moves": ids, "baseAbility": "none", "item": "", "pokeball": "pokeball"}]
    for b in bench:
        side.append({"ident": f"p1: {b.species}", "details": f"{b.species}, L100", "condition": "300/300", "active": False,
                     "stats": _STATS, "moves": [gen1(m).id for m in b.attacks + b.support], "baseAbility": "none",
                     "item": "", "pokeball": "pokeball"})
    return {"active": [active], "side": {"name": "p1", "id": "p1", "pokemon": side}, "rqid": 2}


@pytest.fixture
def make_battle():
    def _make(my: PoolMon, opp_species: str, bench: list | None = None) -> Battle:
        bench = bench if bench is not None else [m for m in POOL if m.species != my.species][:2]
        b = Battle("battle-gen1ou-t", "p1", logging.getLogger("t"), gen=1)
        b.parse_message(["", "switch", f"p2a: {opp_species}", f"{opp_species}, L100", "300/300"])
        b.parse_request(_request(my, bench))
        return b
    return _make
```

- [ ] **Step 3: coach.py / providers.py 구현**

```python
# coach.py
"""Coach v1: SimpleHeuristicsPlayer's singles rules, classified so the router knows what it chose."""
from __future__ import annotations

from dataclasses import dataclass

from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder

from .moves import is_attack


@dataclass
class CoachDecision:
    order: BattleOrder
    kind: str
    move: Move | None


def attack_score(battle: AbstractBattle, move: Move) -> float:
    """The coach's own attack scoring (SimpleHeuristicsPlayer): bp x STAB x stat ratio x accuracy x type."""
    active, opp = battle.active_pokemon, battle.opponent_active_pokemon
    phys = SimpleHeuristicsPlayer._stat_estimation(active, "atk") / SimpleHeuristicsPlayer._stat_estimation(opp, "def")
    spec = SimpleHeuristicsPlayer._stat_estimation(active, "spa") / SimpleHeuristicsPlayer._stat_estimation(opp, "spd")
    ratio = phys if move.category == MoveCategory.PHYSICAL else spec
    stab = 1.5 if move.type in active.types else 1.0
    return move.base_power * stab * ratio * move.accuracy * opp.damage_multiplier(move)


class Coach:
    def __init__(self, weak: bool = False):
        self.weak = weak

    def decide(self, battle: AbstractBattle) -> CoachDecision:
        if self.weak and not battle.force_switch:
            attacks = [m for m in battle.available_moves if is_attack(m)]
            if attacks:
                best = max(attacks, key=lambda m: attack_score(battle, m))
                return CoachDecision(Player.create_order(best), "attack", best)
        order, _ = SimpleHeuristicsPlayer.choose_singles_move(battle)
        target = getattr(order, "order", None)
        if isinstance(target, Move):
            return CoachDecision(order, "attack" if is_attack(target) else "support", target)
        if isinstance(target, Pokemon):
            return CoachDecision(order, "switch", None)
        return CoachDecision(order, "default", None)
```

```python
# providers.py
"""Decision providers: who picks among the fly's attack candidates. The brain provider arrives in M3."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from poke_env.battle import AbstractBattle, Move

from .coach import attack_score


class DecisionProvider(Protocol):
    async def decide(self, battle: AbstractBattle, candidates: list[Move], context: dict) -> int: ...


class RandomProvider:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    async def decide(self, battle, candidates, context) -> int:
        return int(self.rng.integers(len(candidates)))


class MaxDamageProvider:
    async def decide(self, battle, candidates, context) -> int:
        scores = [attack_score(battle, m) for m in candidates]
        return int(np.argmax(scores))
```

- [ ] **Step 4: 통과 확인·커밋** — `feat(battle): coach v1 (heuristic, weak) and decision providers`

---

### Task 5: 라우터와 후보 추출

**Files:** Create `flymon/battle/router.py`; Test `tests/battle/test_router.py`

**Interfaces:**
- `router.candidates(battle) -> list[Move]`: `available_moves` 중 `is_attack`이고 `attack_allowed`인 기술(풀 검증을 통과했으므로 사실상 전부), 요청 순서 유지.
- `router.route(coach_decision, cands) -> tuple[str, list[Move]]`: `("fly", cands)` iff `coach_decision.kind == "attack" and len(cands) >= 2`, else `("coach", [])`.

- [ ] **Step 1: 테스트**

```python
from flymon.battle.coach import Coach, CoachDecision
from flymon.battle.pool import by_species
from flymon.battle.router import candidates, route


def test_candidates_are_attacks_only(make_battle):
    b = make_battle(by_species["Chansey"], "Machamp")
    assert [m.id for m in candidates(b)] == ["thunderbolt", "psychic"]


def test_route_fly_when_attack_and_two_plus(make_battle):
    b = make_battle(by_species["Lapras"], "Venusaur")
    d = Coach().decide(b)
    who, cands = route(d, candidates(b))
    assert who == "fly" and len(cands) == 3


def test_route_coach_when_single_candidate_or_non_attack(make_battle):
    b = make_battle(by_species["Venusaur"], "Charizard")
    only = candidates(b)[:1]
    assert route(CoachDecision(None, "attack", only[0]), only)[0] == "coach"
    assert route(CoachDecision(None, "switch", None), candidates(b))[0] == "coach"
```

- [ ] **Step 2–4: 구현·통과·커밋** — `feat(battle): decision router and candidate extraction`

---

### Task 6: 귀속 파서

**Files:** Create `flymon/battle/attribution.py`; Test `tests/battle/test_attribution.py` (고정 프로토콜 로그 6개는 테스트 파일 안에 문자열 상수로).

**Interfaces:**
- `attribution.Outcome(move_id: str | None, direct_damage: int, target_hp_before: int, target_max_hp: int, dealt_frac: float, effectiveness: str ∈ {"super","neutral","resisted","immune","unknown"}, target_fainted_by_me: bool, missed: bool, no_action: bool, uncertain: bool, notes: list[str])`.
- `attribution.TurnAttributor(my_side: str = "p1")` with `feed(split_message: list[str])` for every raw line in order, `end_turn() -> Outcome`, `reset()`. 규칙(스펙 3.4): 내 활성이 쓴 `|move|` 뒤 다음 `|move|`/`|turn|`/`|upkeep|` 전까지가 내 행동 블록; 블록 안 `|-damage|<대상>|HP` 중 `[from]`이 없는 것만 직접 피해(대상 HP 값은 `cur/max` 또는 `cur/100` — 상대는 퍼센트이므로 `target_max_hp=100`으로 정규화해 `dealt_frac = (before − after)/max`); `[from] brn/psn/…`은 잔여로 무시; `|-activate|<대상>|Substitute|[damage]`는 직접 피해 0·무신호; `|faint|<대상>`이 직접 피해 바로 뒤에 오면 내 기절 귀속; `|-miss|`→missed; `|cant|<내 활성>`→no_action; `-supereffective/-resisted/-immune` 태그를 effectiveness에; 순서가 어긋나거나 대상 식별이 안 되면 `uncertain=True`.

- [ ] **Step 1: 테스트 (6개 사례)**

```python
from flymon.battle.attribution import TurnAttributor

def _lines(text):
    return [l.split("|") for l in text.strip().splitlines()]

CASE_DIRECT_SE_KO = """
|turn|3
|move|p1a: Lapras|Psychic|p2a: Venusaur
|-supereffective|p2a: Venusaur
|-damage|p2a: Venusaur|0 fnt
|faint|p2a: Venusaur
|upkeep
"""
CASE_RESIDUAL_KO = """
|turn|4
|move|p1a: Charizard|Flamethrower|p2a: Exeggutor
|-supereffective|p2a: Exeggutor
|-damage|p2a: Exeggutor|12/100
|move|p2a: Exeggutor|Psychic|p1a: Charizard
|-damage|p1a: Charizard|150/300
|-damage|p2a: Exeggutor|0 fnt|[from] brn
|faint|p2a: Exeggutor
|upkeep
"""
CASE_SUBSTITUTE = """
|turn|5
|move|p1a: Snorlax|Earthquake|p2a: Chansey
|-activate|p2a: Chansey|Substitute|[damage]
|upkeep
"""
CASE_MISS = """
|turn|6
|move|p1a: Rhydon|Rock Slide|p2a: Dragonite|[miss]
|-miss|p1a: Rhydon|p2a: Dragonite
|upkeep
"""
CASE_CANT = """
|turn|7
|cant|p1a: Slowbro|par
|move|p2a: Machamp|Earthquake|p1a: Slowbro
|-damage|p1a: Slowbro|100/300
|upkeep
"""
CASE_SWITCH_TURN = """
|turn|8
|switch|p1a: Chansey|Chansey, L100|300/300
|move|p2a: Machamp|Earthquake|p1a: Chansey
|-damage|p1a: Chansey|150/300
|upkeep
"""

def run(case):
    a = TurnAttributor("p1")
    for l in _lines(case):
        a.feed(l)
    return a.end_turn()

def test_direct_super_effective_ko():
    o = run(CASE_DIRECT_SE_KO)
    assert o.move_id == "psychic" and o.effectiveness == "super" and o.target_fainted_by_me and o.dealt_frac == 1.0 and not o.uncertain

def test_residual_ko_not_attributed():
    o = run(CASE_RESIDUAL_KO)
    assert o.move_id == "flamethrower" and o.dealt_frac == 0.88 and not o.target_fainted_by_me and "residual" in " ".join(o.notes)

def test_substitute_gives_no_signal():
    o = run(CASE_SUBSTITUTE)
    assert o.direct_damage == 0 and o.dealt_frac == 0.0 and o.uncertain

def test_miss_and_cant_are_no_signal():
    assert run(CASE_MISS).missed and run(CASE_CANT).no_action

def test_switch_turn_has_no_move():
    o = run(CASE_SWITCH_TURN)
    assert o.move_id is None and o.no_action
```

- [ ] **Step 2–4: 구현·통과·커밋** — `feat(battle): turn attribution parser for reinforcement signals`. HP 파싱: `"0 fnt"` → 0; `"12/100"` → (12, 100); `"150/300"` → (150, 300). 대상의 "타격 전 HP"는 그 대상의 마지막으로 본 HP(블록 이전 줄에서 추적; 없으면 max로 가정하고 `notes`에 기록).

---

### Task 7: 배치 배리어

**Files:** Create `flymon/battle/barrier.py`; Test `tests/battle/test_barrier.py`

**Interfaces:**
- `BatchBarrier(run_batch: Callable[[list[Request]], Awaitable[list[int]]], deadline_ms: float = 100.0)`; `register(player_id)`, `unregister(player_id)`(활성 집합); `async submit(player_id, battle, candidates, context) -> int`(대기 후 결과); 실행 조건: 첫 요청 도착 후 `deadline_ms` 경과 **또는** 활성 플레이어 전원 대기. 실행 시 대기 요청만 모아 `run_batch`를 한 번 호출. `cancel(player_id)`는 그 플레이어의 대기 요청을 `asyncio.CancelledError`로 깨운다. 예외는 대기자 전원에게 전파.

- [ ] **Step 1: 실패하는 테스트** — `tests/battle/test_barrier.py`

```python
import asyncio

import pytest

from flymon.battle.barrier import BatchBarrier, Request


def _batcher(calls):
    async def run_batch(reqs):
        calls.append([r.player_id for r in reqs])
        return [0] * len(reqs)
    return run_batch


async def test_runs_immediately_when_all_active_players_wait():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=10_000)   # long deadline: must not be what triggers
    bar.register("a"); bar.register("b")
    t0 = asyncio.get_event_loop().time()
    ra, rb = await asyncio.gather(bar.submit("a", None, ["x", "y"], {}), bar.submit("b", None, ["x", "y"], {}))
    assert (ra, rb) == (0, 0) and calls == [["a", "b"]] or calls == [["b", "a"]]
    assert asyncio.get_event_loop().time() - t0 < 1.0


async def test_single_remaining_player_runs_after_deadline():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=50)
    bar.register("a"); bar.register("b")
    bar.unregister("b")                                  # b's battle ended
    t0 = asyncio.get_event_loop().time()
    assert await bar.submit("a", None, ["x", "y"], {}) == 0
    assert 0.04 <= asyncio.get_event_loop().time() - t0 < 1.0 and calls == [["a"]]


async def test_batch_exception_propagates_to_all_waiters():
    async def boom(reqs):
        raise RuntimeError("swarm failed")
    bar = BatchBarrier(boom, deadline_ms=20)
    bar.register("a"); bar.register("b")
    with pytest.raises(RuntimeError):
        await asyncio.gather(bar.submit("a", None, ["x"], {}), bar.submit("b", None, ["x"], {}))


async def test_cancel_wakes_waiter():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=10_000)
    bar.register("a"); bar.register("b")
    task = asyncio.create_task(bar.submit("a", None, ["x"], {}))
    await asyncio.sleep(0.01)
    bar.cancel("a")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []
```

- [ ] **Step 2: 실패 확인** — ImportError.

- [ ] **Step 3: barrier.py 구현**

```python
"""Collect the fly's decisions across concurrent battles and run them as one batch (the swarm in M3)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Request:
    player_id: str
    battle: Any
    candidates: list
    context: dict
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class BatchBarrier:
    def __init__(self, run_batch: Callable[[list], Awaitable[list]], deadline_ms: float = 100.0):
        self.run_batch, self.deadline = run_batch, deadline_ms / 1000.0
        self.active: set = set()
        self.pending: dict = {}
        self._timer: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def register(self, player_id: str) -> None:
        self.active.add(player_id)

    def unregister(self, player_id: str) -> None:
        self.active.discard(player_id)
        self.cancel(player_id)
        self._maybe_flush_soon()

    def cancel(self, player_id: str) -> None:
        req = self.pending.pop(player_id, None)
        if req and not req.future.done():
            req.future.cancel()

    async def submit(self, player_id: str, battle, candidates: list, context: dict) -> int:
        req = Request(player_id, battle, candidates, context)
        self.pending[player_id] = req
        if self._timer is None:
            self._timer = asyncio.create_task(self._deadline())
        self._maybe_flush_soon()
        return await req.future

    def _maybe_flush_soon(self) -> None:
        if self.pending and self.active <= set(self.pending):      # every active player is waiting
            asyncio.create_task(self._flush())

    async def _deadline(self) -> None:
        await asyncio.sleep(self.deadline)
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if self._timer and not self._timer.done() and asyncio.current_task() is not self._timer:
                self._timer.cancel()
            self._timer = None
            reqs = [r for r in self.pending.values() if not r.future.done()]
            self.pending.clear()
            if not reqs:
                return
            try:
                results = await self.run_batch(reqs)
            except Exception as e:                                   # propagate to every waiter
                for r in reqs:
                    if not r.future.done():
                        r.future.set_exception(e)
                return
            for r, idx in zip(reqs, results):
                if not r.future.done():
                    r.future.set_result(int(idx))
```

- [ ] **Step 4: 통과 확인·커밋** — `uv run pytest tests/battle/test_barrier.py -v` → 4 passed. 커밋 `feat(battle): batch barrier with wall-clock deadline`.

---

### Task 8: FlyCoachPlayer, 상대, 통합 테스트, 파일럿 러너와 M1 게이트

**Files:** Create `flymon/battle/fly_coach_player.py`, `flymon/battle/opponents.py`, `scripts/pilot_no_brain.py`; Test `tests/battle/test_fly_coach_player.py`(통합, 서버 없으면 skip)

**Interfaces:**
- `FlyCoachPlayer(provider: DecisionProvider, coach: Coach, barrier: BatchBarrier | None, log_path: Path | None, **player_kwargs)`: `_handle_battle_message`를 오버라이드해 각 줄을 배틀별 `TurnAttributor.feed`에 넣고 `|turn|`/`|upkeep|`에서 `end_turn()` 결과를 `self.outcomes[battle_tag].append(...)`; `async choose_move(battle)`: `coach.decide` → `route` → fly면 `provider.decide`(배리어가 있으면 `barrier.submit`) → 주문. 턴마다 JSONL 로그 `{battle_tag, turn, decider, coach_kind, candidates, chosen, outcome}`. `battle_stats(battle_tag) -> {"fly_turns", "coach_turns", "won"}`.
- `opponents.make_opponent(kind: str, **kw)`: `"random"` → `RandomPlayer`, `"heuristic"` → `SimpleHeuristicsPlayer`; 계정 이름은 `f"fm-{kind}-{n}"`.
- `scripts/pilot_no_brain.py --arm {RND,MAX,WEAK-RND,WEAK-MAX} --flies 16 --battles 100 --port 8000 --schedule results/m1/schedule.json`: 서버를 띄우고(또는 기존 사용), 일정을 만들거나 로드하고, 마리마다 `FlyCoachPlayer`(RND 또는 MAX 제공자, 약한 코치 여부)와 상대 `SimpleHeuristicsPlayer`를 만들어 `battle_against`를 돌린다. 결과 `results/m1/pilot_<arm>.json` = 팔·승률·초파리 결정 턴 비율·후보 수 분포·배틀 로그 경로. `--gate`: 네 파일을 읽어 `MAX − RND ≥ 0.15`를 판정하고 `results/summary/m1_pilot.json`을 쓴다(`{"arms": {...win_rate...}, "gap_max_minus_rnd", "gate_ok"}`).

- [ ] **Step 1: 통합 테스트** — `tests/battle/test_fly_coach_player.py`

```python
import pytest
from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from flymon.battle.coach import Coach
from flymon.battle.fly_coach_player import FlyCoachPlayer
from flymon.battle.pool import POOL, team_export
from flymon.battle.providers import RandomProvider
from flymon.battle.server import NODE_BIN, ShowdownServer

pytestmark = pytest.mark.skipif(not NODE_BIN.exists(), reason="run scripts/install_showdown.sh first")


@pytest.fixture(scope="module")
def server():
    with ShowdownServer(port=8790) as s:
        yield s


async def test_two_battles_end_and_log_turns(server, tmp_path):
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    me = FlyCoachPlayer(provider=RandomProvider(1), coach=Coach(), barrier=None, log_path=tmp_path / "log.jsonl",
                        account_configuration=AccountConfiguration("fm-test-me", None), battle_format="gen1ou",
                        server_configuration=cfg, team=team_export(POOL[:6]), max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    await me.battle_against(opp, n_battles=2)
    assert me.n_finished_battles == 2
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) > 10
    stats = [me.battle_stats(t) for t in me.battles]
    assert all(s["fly_turns"] + s["coach_turns"] > 0 for s in stats)
    await me.ps_client.stop_listening(); await opp.ps_client.stop_listening()
```

- [ ] **Step 2: 구현** — `flymon/battle/fly_coach_player.py`

```python
"""poke-env player: coach decides strategy, a DecisionProvider picks among attack candidates."""
from __future__ import annotations

import json
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import Player

from .attribution import TurnAttributor
from .barrier import BatchBarrier
from .coach import Coach
from .providers import DecisionProvider
from .router import candidates, route


class FlyCoachPlayer(Player):
    def __init__(self, provider: DecisionProvider, coach: Coach, barrier: BatchBarrier | None = None,
                 log_path: Path | None = None, **player_kwargs):
        super().__init__(**player_kwargs)
        self.provider, self.coach, self.barrier = provider, coach, barrier
        self.log_path = Path(log_path) if log_path else None
        self.attributors: dict = {}
        self.outcomes: dict = {}
        self.turn_log: dict = {}
        self.player_id = self.username
        if barrier is not None:
            barrier.register(self.player_id)

    # ---- raw protocol -> attribution ----------------------------------------------------
    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        att = self.attributors.setdefault(tag, TurnAttributor(self._my_side(tag)))
        for line in split_messages[1:]:
            if len(line) > 1 and line[1] in ("turn", "upkeep"):
                if att.has_block():
                    self.outcomes.setdefault(tag, []).append(att.end_turn())
            att.feed(line)
        await super()._handle_battle_message(split_messages)

    def _my_side(self, tag: str) -> str:
        b = self._battles.get(tag)
        return b.player_role if b is not None and b.player_role else "p1"

    # ---- decision -------------------------------------------------------------------------
    async def choose_move(self, battle: AbstractBattle):
        decision = self.coach.decide(battle)
        cands = candidates(battle)
        who, cands = route(decision, cands)
        chosen = None
        if who == "fly":
            ctx = {"battle_tag": battle.battle_tag, "turn": battle.turn}
            idx = (await self.barrier.submit(self.player_id, battle, cands, ctx)) if self.barrier else \
                  (await self.provider.decide(battle, cands, ctx))
            chosen = cands[idx]
            order = self.create_order(chosen)
        else:
            order = decision.order
        self._log(battle, who, decision, cands, chosen)
        return order

    def _log(self, battle, who, decision, cands, chosen) -> None:
        rec = {"battle_tag": battle.battle_tag, "turn": battle.turn, "decider": who, "coach_kind": decision.kind,
               "candidates": [m.id for m in cands], "chosen": chosen.id if chosen else (decision.move.id if decision.move else None)}
        self.turn_log.setdefault(battle.battle_tag, []).append(rec)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def battle_stats(self, battle_tag: str) -> dict:
        recs = self.turn_log.get(battle_tag, [])
        b = self._battles.get(battle_tag)
        return {"fly_turns": sum(r["decider"] == "fly" for r in recs), "coach_turns": sum(r["decider"] == "coach" for r in recs),
                "won": bool(b.won) if b is not None else None}

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self.barrier is not None:
            self.barrier.unregister(self.player_id)
```

`TurnAttributor`에 `has_block() -> bool`(현재 턴에 내 행동 블록이 있었는지)을 추가한다(T6 인터페이스 보강). `player_role`이 없으면 `battle_tag`의 순서 규칙 대신 요청의 `side.id`를 쓴다 — poke-env `Battle.player_role`은 요청 수신 후 채워지므로 첫 턴 전에는 "p1"로 두고 `end_turn` 전에 다시 확인한다.

`flymon/battle/opponents.py`:

```python
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration


def make_opponent(kind: str, n: int, server_configuration, team: str):
    cls = {"random": RandomPlayer, "heuristic": SimpleHeuristicsPlayer}[kind]
    return cls(account_configuration=AccountConfiguration(f"fm-{kind}-{n}", None), battle_format="gen1ou",
               server_configuration=server_configuration, team=team, max_concurrent_battles=1)
```

`scripts/pilot_no_brain.py` 핵심 루프(팔마다 실행; 마리 `fly`마다 플레이어 쌍을 만들고 일정의 배틀을 순서대로 `battle_against(opp, 1)`로 돌리되 팀은 배틀마다 `player.update_team(team_export(...))`으로 교체 — poke-env `Player.update_team(team)` 존재 여부를 T8 Step 1 전에 `inspect`로 확인하고, 없으면 배틀마다 플레이어를 새로 만든다):

```python
ARMS = {"RND": ("random", False), "MAX": ("max", False), "WEAK-RND": ("random", True), "WEAK-MAX": ("max", True)}
# provider = RandomProvider(seed=fly) | MaxDamageProvider(); coach = Coach(weak=weak)
# per battle: my team = team_export([by_species[s] for s in sb.my_team]); opp team likewise; opponent = make_opponent(sb.opponent, ...)
# record: {"battle_id", "won", "turns", "fly_turns", "coach_turns", "n_candidates_mean"} -> results/m1/pilot_<ARM>.json with win_rate
# --gate: read the four files; gap = win(MAX) - win(RND); gate_ok = gap >= 0.15; write results/summary/m1_pilot.json
```

- [ ] **Step 3: 통합 테스트 통과** — `uv run pytest tests/battle -v` 전부 통과(서버 테스트 포함, 약 1~2분).

- [ ] **Step 4: 파일럿 실행(컨트롤러)** — 서브에이전트가 아니라 컨트롤러가 돌린다:
```bash
uv run python scripts/pilot_no_brain.py --arm RND --flies 16 --battles 100
uv run python scripts/pilot_no_brain.py --arm MAX --flies 16 --battles 100
uv run python scripts/pilot_no_brain.py --arm WEAK-RND --flies 16 --battles 100
uv run python scripts/pilot_no_brain.py --arm WEAK-MAX --flies 16 --battles 100
uv run python scripts/pilot_no_brain.py --gate
```
**M1 게이트**: `MAX − RND 승률 ≥ 0.15`. 미달이면 풀 재설계(마리당 상성 대비 타입 다양화, 코치 교체 조건 완화)를 `docs/pool.md`에 기록하고 재실행. 통과하면 `results/summary/m1_pilot.json`과 `results/summary/schedule_m1.json`(일정)을 커밋.

- [ ] **Step 5: 커밋** — `feat(battle): FlyCoachPlayer, opponents, no-brain pilot runner and M1 gate`

---

## 자체 검토

- 스펙 커버리지: 서버(3.2 서버·loopback) T1; 팀 풀·허용 목록·검증(3.2) T2; 일정(3.2) T3; 코치·약한 코치·MAX(3.2) T4; 라우터·후보(2절, 3.2) T5; 귀속 파서(3.4, 테스트 6개) T6; 배치 배리어(2절) T7; 플레이어·상대·파일럿·게이트(4.1, 5절 M1) T8. 초파리 결정 제공자(뇌)는 M3.
- 플레이스홀더 없음. 로스터는 "제안 + 검증기 권위" 규칙으로 명시.
- 타입 일관성: `CoachDecision.kind` 값, `route` 반환, `DecisionProvider.decide` 시그니처, `Outcome` 필드가 T4~T8에서 동일.
