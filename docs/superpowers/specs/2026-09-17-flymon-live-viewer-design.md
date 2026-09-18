# FlyMon 라이브 뷰어 설계 — 뇌 없는 배틀 뷰, 뇌를 바로 붙일 수 있게

- 작성일: 2026-09-17
- 상태: 설계 승인(사용자, 대화에서 네 부분 각각 승인) → 이 문서 검토 대기
- 브랜치: `live-viewer`(워크트리 `fruit-fly/live-viewer`, `open-fly-brain-connectome` `8a9865c`에서 분기)
- 관계: 본 스펙 `2026-09-14-flymon-design.md`의 3.7 뷰어(M5, 공개용·정적)와 **별개**다. 이것은 로컬에서 실행 중에 보는 도구다.
  M0d·M2의 어떤 판정에도 쓰지 않는다.

## 1. 목표와 범위

**목표**: 배틀이 도는 동안 한 마리의 배틀 화면과 그 턴의 결정·귀속 기록을 한 페이지에서 본다. 뇌(M3)가 준비되면
**뷰어를 고치지 않고** 뇌 쪽이 데이터를 보내는 것만으로 결정 상세와 뉴런 활동이 화면에 나타나게 한다.

**범위 밖**: 되감기(리플레이 기반 사후 뷰), 여러 마리 동시 보기, 강화 펄스 표시(귀속 → PAM08/PPL105 변환은 M3에서 생긴다 —
뷰어가 3.4 표를 따로 구현하면 실제 구현과 어긋날 수 있다), 공개용 꾸밈(M5), JSONL 로그 형식 변경(M3의 로그 스키마 결정).

## 2. 확인된 사실 (2026-09-17 버리는 스파이크, 뇌 없음)

- 공식 Showdown 클라이언트로 로컬 배틀을 실시간 관전할 수 있다: `http://localhost:<포트>/<battle_tag>` →
  `https://localhost--<포트>.psim.us/<battle_tag>`로 넘어가 진행 중인 배틀이 보였다. 클라이언트 코드·스프라이트는 인터넷에서 받는다.
- **iframe 관전은 불가**(2026-09-17 확인): 응답에 `X-Frame-Options`·CSP는 없지만 공식 클라이언트가 스스로 "Loading client… IN FRAME.
  Please visit Showdown directly."로 멈춘다.
- **대신 배틀 화면을 우리 페이지에서 직접 그린다**(2026-09-18 확인): 리플레이 렌더러 `https://play.pokemonshowdown.com/js/replay-embed.js`를
  우리 페이지에 싣고 `Replays.init()` → `Replays.battle.add(line)`로 프로토콜을 흘려 넣으면 배틀이 실시간으로 그려진다.
  배틀을 바꿀 때는 `script.battle-log-data`를 비우고 `Replays.init()`을 다시 부른다. 애니메이션이 밀리면 `seekTurn(최신 턴)`으로 따라잡는다.
  - **함정**: 이 렌더러는 전역 `$`(jQuery)를 쓴다. 우리 스크립트가 `const $ = ...`를 선언하면 전역을 가려 `Replays.init()`이
    `Cannot read properties of null` 로 죽는다. 페이지 코드에서 `$`라는 이름을 쓰지 않는다.
  - **위험**: `add`·`instantAdd`·`init`은 공개 계약이 아니고 스크립트는 매일 Smogon에서 새로 받는다. 없으면 기능 검사로 감지해 안내 문구를 띄운다.
- 뇌 없는 배틀은 한 판에 약 1초 → 실시간으로 보려면 보는 마리를 일부러 늦춰야 한다.
- `FlyCoachPlayer` JSONL 레코드는 모두 `battle_tag`·`turn`을 가진다. 강제 교체 턴은 같은 (battle_tag, turn)에 `decision`이 둘이다.
- 결정 provider는 후보 인덱스(int)만 돌려준다. `context` dict는 직접 호출(`provider.decide`)과 배리어(`Request.context`) 경로를 모두 거친다.
- `flymon/brain/viz.py`의 싱크 인터페이스는 `scalar(path, t_ms, value)`, `bars(path, t_ms, values)`, `text(path, t_ms, msg)`이고
  `SpikeTap`은 관찰이 결과를 바꾸지 않는다(`tests/brain/test_conditioning_events.py`).
- 의존성: `websockets`만 설치돼 있다. 이 설계는 표준 라이브러리만 쓴다.

## 3. 화면

```
┌───────────────────────────────────────────────────────────────────────────┐
│ FlyMon Live │ 보는 마리 [fly 03 ▾] │ battle-gen1ou-14 · 턴 10 │ 뷰어 연결 ● │
├──────────────────────────────┬────────────────────────────────────────────┤
│                              │ ① 이번 턴 결정                              │
│  공식 Showdown 배틀 화면      │   [초파리]  코치 주문: 공격                   │
│  (우리 페이지 안, 프로토콜 실시간) │   후보: 10만볼트 · 사이코키네시스 ✔          │
│                              │   ┄ detail이 오면: 후보별 막대 ┄             │
│                              │ ② 이번 턴 결과 (귀속)                        │
│                              │   사이코키네시스 → 직접 피해 34% · 효과 굉장함 │
│                              │   기절 ✗ · 빗나감 ✗ · 불확실 ✗               │
├──────────────────────────────┴────────────────────────────────────────────┤
│ ③ 뇌 활동    뇌 미연결 (trace가 오면 path별 차트가 자동으로 생김)            │
├───────────────────────────────────────────────────────────────────────────┤
│ ④ 턴 타임라인  1 코치 │ 2 초파리 ✔34% │ 3 초파리 ✗빗나감 │ 4 코치 │ …       │
└───────────────────────────────────────────────────────────────────────────┘
```

- **마리 선택**: 이벤트가 들어온 마리 목록. 기본값은 URL의 `?fly=<계정 이름>`, 없으면 처음 이벤트를 보낸 마리.
  `live_battles.py`는 `--watch-fly` 마리의 `?fly=`가 붙은 주소를 출력한다. 보는 마리나 배틀이 바뀌면 렌더러를 다시 초기화하고
  그 배틀의 프로토콜을 `instantAdd`로 한 번에 따라잡은 뒤 이어서 그린다.
- **① ②**: 해당 마리의 가장 최근 턴. 강제 교체로 `decision`이 둘이면 둘 다 보인다.
- **③**: 선택한 턴의 `trace`를 `phase`·`slot`별로 묶고, `kind`별로 그린다 — `scalar` 선 그래프, `bars` 막대(마지막 점), `text` 목록.
  trace가 없으면 "뇌 미연결".
- **④**: 턴을 누르면 그 턴의 ①②③을 보여준다. 배틀 화면은 실시간 그대로다(되감지 않음). "최신 따라가기"로 돌아간다.
- **렌더러를 못 쓸 때**: 기능 검사가 실패하면 배틀 화면 자리에 안내 문구만 두고 ①②③④는 그대로 동작한다.
- 외부 라이브러리 없음. 차트는 SVG를 직접 그린다.

**확인 결과 (2026-09-18, `live-viewer` 브랜치, 뇌 없음)**: 완료 기준 4개 모두 통과.
① 가짜 뇌 2마리×3배틀에서 배틀이 페이지 안에 그려지고 렌더러 턴 = 헤더 턴(13/13), 후보별 막대 3개, trace 차트 15개, 턴 칩 13개,
드롭다운으로 다른 마리 배틀로 교체됨(스크린샷 `results/live-viewer/criterion1-fake-brain.png`, git 제외).
② `--provider rnd`에서 "뇌 미연결", 막대 0개, decision에 detail 없음.
③ 실행 중 뷰어를 `kill -TERM` 하니 배틀은 끝까지 진행, `exit 0`, `dropped_events=68`.
④ 렌더러를 못 쓰게 만들면 안내 문구가 뜨고 ①②④ 패널은 계속 동작.
- 운영 메모: 배경 프로세스는 SIGINT를 무시하므로 뷰어를 멈출 때는 `kill -TERM`을 쓴다.

## 4. 이벤트 형식 (뇌 접속 계약)

모든 이벤트는 JSON 객체 하나이고 공통 필드를 가진다.

```json
{"v": 1, "fly": "fm-rnd-f03", "type": "decision", "battle_tag": "battle-gen1ou-14", "turn": 10}
```

| type | 시점 | 추가 필드 |
|---|---|---|
| `battle_start` | 플레이어가 새 배틀 방을 처음 볼 때 | — |
| `protocol` | 배틀 메시지가 올 때마다 | `lines`(원시 프로토콜 줄, `|request|` 제외) |
| `decision` | JSONL `decision` 레코드를 쓸 때 | `decider`, `coach_kind`, `candidates`, `chosen`, `detail`(선택) |
| `outcome` | JSONL `outcome` 레코드를 쓸 때 | `outcome`(귀속 `Outcome` 전체), `detail`(선택) |
| `trace` | 냄새 제시 하나가 끝날 때 | `phase`(`"decide"`·`"reinforce"`), `slot`(후보 번호 또는 `null`), `series` |
| `battle_end` | 배틀 종료 콜백 | `won`(true/false/null), `turns` |

- `protocol`은 페이지가 공식 렌더러로 배틀을 그리는 데 쓴다. `|request|`는 내 팀 시트라 뺀다(렌더러도 쓰지 않는다).
  `battle_end` 뒤에도 서버가 방 메시지를 더 보내면 `protocol`이 더 올 수 있다.
- `fly`는 플레이어 계정 이름이다(실행 안에서 유일). `turn`은 `battle_start`·`battle_end`에서 그때의 마지막 턴 값이다.
- `decision`·`outcome`의 필드 값은 같은 순간의 JSONL 레코드와 같다(테스트로 고정).
- **`detail`**: provider 또는 배치 실행기가 `context["detail"]`에 dict를 넣으면 플레이어가 `decision` 이벤트에 싣는다. 뷰어의 일반 규칙:
  숫자 리스트이고 길이 = 후보 수 → 후보별 막대, 스칼라(숫자·문자·불리언) → `이름: 값`, dict → 소제목 + 안쪽에 같은 규칙, 그 외 → JSON 문자열.
- **`series`**: `[{"path": str, "kind": "scalar"|"bars"|"text", "points": [[t_ms, 값], ...]}]`. `t_ms`는 시뮬레이션 시간이다.
  `SpikeTap`의 시계는 `engine.reset()`을 넘어 계속 흐르므로, 뷰어는 trace 하나 안의 최소 `t_ms`를 0으로 맞춰 그린다.
- **호환성**: `v`가 1이 아니거나 모르는 `type`이면 뷰어는 무시하고 개수만 센다. 모르는 필드는 무시한다. 필드 추가는 `v`를 올리지 않는다.

**M3에서 뇌가 할 일(이 작업의 범위 밖, 계약으로만 고정)**: (1) 배치 실행기가 각 요청의 `context["detail"]`을 채운다.
(2) 워커가 `SpikeTap(engine, groups, WebSink(...))`로 제시를 관찰하고, 제시가 끝나면 `WebSink.flush(meta)`를 부른다.

## 5. 구성요소와 파일

```
flymon/live/
  events.py        make_event(type, fly, battle_tag, turn, **fields) — type별 필수 필드 검사, v=1
  sink.py          EventSink 프로토콜(emit(event) -> None), NullSink, RecordingSink(테스트용),
                   HttpEventSink(url, max_queue) — 큐 + 데몬 스레드가 POST /events로 묶어 보냄,
                   dropped 카운터, close(timeout)
  trace.py         WebSink(event_sink, fly) — scalar/bars/text 시그니처가 RerunSink와 같다.
                   flush(battle_tag, turn, phase, slot)가 모은 점을 trace 이벤트 하나로 보내고 비운다
  server.py        ViewerServer(host="127.0.0.1", port, keep_battles=3) — ThreadingHTTPServer
                   POST /events(JSON 배열 또는 객체), GET /stream(SSE), GET /state(마리별 보관 이벤트), GET /(web/live 정적 파일)
  fake_brain.py    FakeBrainProvider(inner, event_sink, seed) — inner의 선택을 그대로 쓰고, context["detail"]에
                   합성 V·p를 넣고, 후보마다 합성 trace를 WebSink로 보낸다. 실험에 쓰지 않는다
web/live/          index.html(렌더러 wrapper + replay-embed 로드) · app.js(`$` 이름 금지) · style.css
scripts/
  live_viewer.py   ViewerServer 실행(--port 8765), 주소 출력
  live_battles.py  Showdown 서버(빈 포트) + FlyCoachPlayer N마리, --flies --battles --provider rnd|max|fake-brain
                   --watch-fly --turn-delay --viewer URL(없으면 NullSink). 끝에 버린 이벤트 수 출력
수정: flymon/battle/fly_coach_player.py
  __init__(..., event_sink=None, turn_delay_s=0.0) — None이면 emit 호출 없음, 0이면 대기 없음
  _log·_close_turn에서 JSONL 레코드와 같은 내용으로 emit, 새 방이면 battle_start, _battle_finished_callback에서 battle_end
  choose_move에서 context를 만든 뒤 provider/배리어가 채운 context.get("detail")을 decision 이벤트에 싣는다
  _handle_battle_message에서 원시 프로토콜 줄을 protocol 이벤트로 내보낸다(`|request|` 제외)
  turn_delay_s > 0이면 주문을 돌려주기 전에 asyncio.sleep
```

- `pilot_no_brain.py`(M1 게이트)는 바꾸지 않는다.
- 플레이어는 `emit`을 try/except로 감싼다 — 싱크가 예외를 던져도 배틀은 계속된다(싱크 쪽 계약도 "예외 없음"이지만 이중으로 막는다).
- 서버와 Showdown 서버 모두 127.0.0.1에만 바인딩한다.

## 6. 관찰 불변 계약

- `event_sink=None`, `turn_delay_s=0`이면 `FlyCoachPlayer`의 동작과 JSONL은 이 작업 전과 같다.
- 뷰어 서버가 없거나 느려도 `emit`은 즉시 돌아오고(큐 가득 → 버림), 배틀 결과·JSONL에 영향이 없다.
- `WebSink`를 단 `SpikeTap`은 스파이크를 바꾸지 않는다.
- `turn_delay_s`는 시간만 늦춘다(서버 난수는 원래 고정되지 않으므로 배틀 재현성 주장에 영향 없음).

## 7. 테스트

| 파일 | 확인 |
|---|---|
| `tests/live/test_events.py` | type별 필수 필드 누락 시 `ValueError`, `v` = 1 |
| `tests/live/test_sink.py` | 서버 없음: emit 1,000회가 짧은 시간 안에 끝나고 예외 없음, 큐 가득 → dropped 증가, 서버 있음: 보낸 순서대로 도착 |
| `tests/live/test_trace.py` | `WebSink`와 `RerunSink`의 메서드 시그니처 동일, flush 1회 = trace 이벤트 1개·점 보존·비움, 작은 합성 망에서 `SpikeTap`+`WebSink` 유무로 스파이크 비트 동일 |
| `tests/live/test_viewer_server.py` | 127.0.0.1 바인딩, POST → `/state`·SSE 순서, 마리별 최근 배틀 `keep_battles`개 보관, 깨진 JSON → 400 이후에도 정상 |
| `tests/battle/test_fly_coach_player.py`(추가) | 기본값: emit 0회, `RecordingSink`: JSONL 레코드마다 같은 내용 이벤트, `battle_start`가 맨 앞·`battle_end` 뒤에 결정/결과 없음, 예외 던지는 싱크로도 배틀 완료, `protocol`이 `|init|battle`로 시작하고 `|request|`를 담지 않음 |
| `tests/battle/test_barrier.py`(추가) | 배치 실행기가 채운 `context["detail"]`이 submit한 쪽 context에 보인다 |
| `tests/live/test_live_battles.py` | 1마리·1배틀·`fake-brain`, 인프로세스 ViewerServer의 `/state`에 5개 type 모두 존재 |

## 8. 완료 기준 (관찰 가능, 스크린샷을 `results/live-viewer/`(git 제외)에 남기고 경로를 원장에 기록)

1. `live_viewer.py` + `live_battles.py --flies 2 --battles 2 --provider fake-brain --watch-fly 0 --turn-delay 2`:
   페이지 안에 fly 0 배틀이 그려짐(렌더러 턴이 헤더 턴보다 2턴 이상 뒤처지지 않음), ①② 채워짐, ③ 차트 생성, ④ 늘어남, 드롭다운 전환 시 그 마리의 배틀로 교체.
2. `--provider rnd`: ③ "뇌 미연결", ①에 막대 없음.
3. 실행 도중 뷰어 서버 종료: 배틀 정상 종료, exit 0, 버린 이벤트 수 출력.
4. 렌더러 기능 검사 실패를 흉내 냈을 때 안내 문구가 뜨고 나머지 패널이 동작.
5. 전체 `uv run pytest` 통과.

## 9. 진행

- 태스크(계획서에서 확정): 1 events·sink → 2 trace → 3 server → 4 플레이어 훅(detail·protocol) → 5 fake_brain·실행 스크립트 →
  6 web/live → 7 완료 기준 확인.
- SDD 루프. 구현자는 이 워크트리의 오케스트레이션 워커(다른 세션이 `open-fly-brain-connectome`에 M0d를 커밋 중이라 격리), 리뷰·원장은 컨트롤러.
- 확인용 배틀은 몇 판만 — 같은 기계의 M0d 실데이터 실행 부하를 늘리지 않는다.
- `live-viewer` → `open-fly-brain-connectome` 병합은 M0d 세션이 태스크 경계에 있을 때 사용자에게 묻고 한다.
