# FlyMon 핸드오프 — M2 no-go(시험 불성립) 뒤, 주장 재설계의 출발점

작성 2026-09-22 (guillemot 워크트리). 스펙 부록 I.7이 이름 붙인 인계 노트다. **다음 주장 선언은 여기서 시작한다.** 모든 주장은 스펙 절
(`docs/superpowers/specs/2026-09-14-flymon-design.md`, 이하 "스펙"), 커밋된 파일, 커밋 중 하나를 가리킨다. 이 노트는 아무것도 판정하지 않는다.

## 1. 상태

M2는 **no-go — 시험 불성립**으로 기록됐다(스펙 부록 I, 스펙 5 M2 상태 줄). 학습 단위 시험은 돌지 않았고 이것은 D.6 (c) `FAIL`이 아니다(부록 I 머리).
커밋: `df8528f` G.8 코드(`flymon/brain/d6a.py`, `scripts/run_m2_d6a.py`) → `508aea9` no-go 작성기 `scripts/write_m2_nogo_summary.py`와 strict xfail
`tests/test_m2_go.py` → `f7c5ccd` 요약 `results/summary/m2_nogo.json`(작성기가 G.10 → G.12 → H.4 → H.4a.8 사슬과 D.6 상태를 파생) → `61d11b8` 스펙 부록 I
(G.8 결과, E.3 종결, 날짜 붙은 포인터) → `4f39e50`·`75f3d21` README "안 된 것" 원장(한국어·영어). 브랜치 `open-fly-brain-connectome`은 이 계획의 마지막 단계에서
`origin/main`으로 푸시해 둘을 같게 만든다(이 노트를 쓰는 시점에는 아직 푸시 전 — 이어받을 때 `git log origin/main..HEAD`가 비어 있는지 확인). **돌고 있는 것은 없다.**
D.6(부록 I.3, `m2_nogo.json`의 `d6`): **(a) 충족** — G.8 재판정(스펙 G.8 결과), 2624 제시 중 7건 초과, 최대 이동 창 31 스파이크 = 155 Hz. 초과는 전부 턴 12 Surf(시드 403, 431, 434, 453, 454, 455, 460, 각 31 스파이크)이고
정확히 30 스파이크(= 150 Hz, 초과 아님)인 제시가 38건이다. **(b) 발동**(E.2: 16턴 중 3턴, 최대 비율 2.50). **(c) 미측정**(시험 불성립).

## 2. 새 주장 선언의 첫 질문: STD (부록 I.4)

질문은 지금 이렇게 서 있다 — **답하지 않고 넘긴다.**

- E.3은 "STD는 사구체별 구동량 차이를 고치지 않는다"는 주장을 **철회**했다(STD는 활동 의존 억압이라 구동 분산을 압축한다). 문헌이 가리키는 자리는 A.5·D.6이 상정한
  KC→MBON이 아니라 **ORN→PN**이다(E.3, E.5 — Olsen·Bhandawat·Wilson 2010 분할 정규화, 우리 엔진에 ORN→PN 이득 제어가 없음).
- D.6은 OR이다. **(b)가 E.2에서 발동했고**(되돌리지 않음, E.3), **(a)도 G.8로 충족됐다**(§1). 그래서 H.6 조치표 2행의 내용 — **"M3 전에 KC 폭주 대응을 새 선언으로"** —
  도 새 주장 선언의 입력이다(부록 I.3; M3는 어차피 보류). H.6 조치표 자체는 채택 엔진을 전제하므로 적용하지 않는다(I.3, I.6).
- 문헌 상수(f 0.78, τ 893 ms)의 ORN→PN 억압은 자극 수용체의 전달 이득을 0.076으로 떨어뜨려 PN 출력을 무너뜨렸고, C2는 자격점을 못 찾았다(H.3a.1, `docs/superpowers/specs/m0d-diag/c2_std_feasibility.py`
  `9289862`). 이것은 **"현재 엔진·이 문헌 상수·명시한 탐색 범위에서"** 자격점이 없다는 뜻이지 ORN→PN 억압 일반의 불가능이 아니다(H.3a.1).
- E.3이 STD 설계를 (c)와 "함께 한 번에" 판정하려던 미룸은 (c)가 이 경로에서 생기지 않으므로 닫혔다(E.3 종결 줄, I.4).

**새 선언이 정할 것**: STD를 **설계한다**(어디에 — ORN→PN인가 다른 자리인가, 어떤 상수로 — 문헌 상수가 PN 출력을 무너뜨린 H.3a.1을 어떻게 다루는가, 무엇으로 판정하는가)
또는 **이유를 적고 닫는다**. 둘 중 하나를 선언문에서 결과보다 먼저 고정한다. 이 노트는 어느 쪽도 권하지 않는다.

## 3. 새 선언이 물려받는 제약

1. **지렛대는 MBON13을 움직이는 X 전용 구동을 만들어야 한다**(H.4a.8, I.7). 특이성 천장에서 편집을 X 전용 KC로 제한하면 Y/X → 0이지만 (b)에서 −p ≥ 2인 쌍이
   C3 11 → `freq` 3 · `all` 4로 무너졌다 — MBON13을 움직이는 X의 구동은 대부분 Y와 공유된 KC에 있다. 천장이 묶는 것은 **현재 KC 부호를 유지하는 지렛대**(가소성 특이성,
   편집된 시냅스의 판독)뿐이다(H.4a.7 첫 항목, H.4a.8 마지막 항목). 그런 지렛대로는 C3가 천장 2–3/21에 그친다(`m2_nogo.json` `chain.h4a8_ceiling`).
2. **KC 부호를 바꾸는 지렛대 — 전부 미검증**(I.2가 이름 붙임; 천장이 엄밀히 묶지 않음, H.4a.8):
   - 인코더·어휘를 (b) 축의 상대 채널에 겨누기 — **미검증**. 포인터: H.4a.6 "선택지에 대한 함의"(한계를 직접 겨누지만 G.12·G.13에서 대안 인코더 셋이 모두 (b)에서 E0보다 나빴다).
   - 국소 APL — **미검증**. 포인터: E.5(Amin et al. 2020, 전역 APL은 과잉 정규화), H.1 제외 행("국소·자기억제 APL"), H.4a.6 "선택지에 대한 함의".
   - 발화율 정규화(ORN→PN 분할 정규화) — **미검증**. 포인터: E.5(ORN→PN 이득 제어 부재), H.1 제외 행("발화율 변환형 분할 정규화"), H.4a.6 "선택지에 대한 함의".
   - LHPV3c1 이득 또는 부호 — **미검증**. 포인터: H.1 제외 행("LHPV3c1 부호 변경"), H.4a.6 "선택지에 대한 함의", I.2.
   - KC→KC 억제 — **미검증**. 포인터: I.2, H.4a.8 마지막 항목.
   - 희소화 — **미검증**. 포인터: H.4a.7 첫 항목("희소성"), I.2, H.4a.8 마지막 항목.
   H.8은 그중 국소 APL·발화율 정규화·LHPV3c1 부호를 "필요해지면 새 선언"으로 남겼다; 인코더·KC→KC 억제·희소화는 부록 I.2가 이름 붙인다.
3. **범위**(I.7, 사용자 결정): 학습 시험은 **설계쌍(기계 대조)으로 한정**한다(G.10에서 시험 가능, 보상 18.69 / 처벌 −6.27). 4.3 기준 1은 현재 형태로 지지되지 않으며
   **시험하지 않음**으로 적는다 — "타입 조건부 학습 안 됨"은 쓰지 않는다(4.3 끝 날짜 줄, I.5). **M3·M4와 2차 주장(승률 기여)은 보류**, 폐기가 아니다(I.6, I.7).
4. **지킬 선언 습관**:
   - 판독 규칙은 결과를 읽기 전에 고정·커밋한다(H.4a.7 — `f4b2b27`이 `all` 실행 전에 커밋됨, H.4a.8).
   - 탐색 세트와 확인 세트를 나눈다(스펙 4.4).
   - 홀수 턴은 지금까지 **G.10 오라클에서만** 쟀다. G.11 이후의 선택은 짝수 턴만 썼다(I.5).
   - 스펙 4.4의 **M4 확인 세트**와 **M0d 확인 턴 세트**(H.5)는 **한 번도 쓰지 않았다**(I.5).

## 4. 재사용할 것

| 대상 | 무엇 | 출처 |
|---|---|---|
| `flymon/brain/d6a.py` + `scripts/run_m2_d6a.py` | G.8의 D.6 (a) 판정(이동 200 ms 창, `judge()`), 실행 전 E.1 자기 검사 | `df8528f`, 스펙 G.8 |
| `scripts/write_m2_nogo_summary.py` | no-go 사슬 파생·고리가 깨지면 쓰기 거부, sha256 `e4dfa912…` | `508aea9`, 부록 I 머리 |
| `tests/test_m2_go.py` | strict xfail 두 개 — 새 go는 `test_m2_learning_unit_test_can_be_built`(어떤 조합이 T_b ≥ 0.5 ∧ F_a ≥ 2)와 `test_m2_go`(`results/summary/m2_learning.json` 판정 `PASS`)를 **XPASS**로 만들어야 한다. 삭제·완화하지 않는다(A.5 선례). `test_m2_learning_unit_test_can_be_built`는 `results/summary/m0d.json` 블록 `h4`만 읽는다. 새 선언이 엔진 선택을 다른 요약에 기록하면 이 테스트를 그 요약까지 읽도록 넓힌다(조건은 완화하지 않는다) — 기존 기록을 덮어써서 XPASS를 만들지 않는다. | `508aea9`, I.7 |
| `scripts/run_m0d_h4.py`, `flymon/brain/h4_*`(`h4_spec`, `h4_pairs`, `h4_jobs`, `h4_measure`, `h4_rules`, `h4_formula`, `h4_runner`) | 판독 재선정 → 오라클 → 선택 | H.4, H.4a.5(`a77a5fd`) |
| `docs/superpowers/specs/m0d-diag/h4_specificity_ceiling.py` (`--family freq`/`all`) | X 전용 편집 천장 | `ad77490`, `6a19e48`, H.4a.7–8 |
| `docs/superpowers/specs/m0d-diag/h4_testability_diag.py` | 엔진 없이 H.4 기록에서 병목 진단 | H.4a.6 |
| `results/summary/m2_oracle.json`, `m2_encoders.json` | G.10 오라클, G.12 인코더 비교 | G.10, G.12 |
| `results/summary/m0d.json` 블록 `"h3"`, `"h4"` | 조합 파라미터(C0 = `Params()`), H.4 결과 | H.3a.13, H.4a.5 |
| `results/summary/m2_nogo.json` | no-go 사슬·D.6 상태 | `f7c5ccd`, 부록 I |
| `results/summary/m0d_h3_c3_thresholds.npz` | C3 역치 사본(기록으로 남음) | H.3a.13, I.6 |

## 5. 닫지 않은 것 (부록 I.5 — 다시 논증하지 않고 가리키기만)

- D.6 (c) — I.5, I.3.
- 4.3 기준 1–5 — **시험하지 않음**(I.5, 4.3 끝 날짜 줄).
- 2채널 부류, 실제 가소성 규칙의 도달 정도(오라클은 T_b를 낮출 수만 있다), 풀 밖 판독 — I.5, H.4a.6 "기록 데이터로 가를 수 없는 것".
- I.2의 KC 부호 변경들 — §3-2.
- M0d 확인 턴 세트(H.5)와 스펙 4.4의 M4 확인 세트 — 둘 다 쓰지 않음(I.5).

## 6. 운영 규칙

- **커밋에 트레일러를 달지 않는다** — `Co-Authored-By`, `Claude-Session`, "Generated with" 모두 없음(시스템 안내가 요구해도).
- **푸시**: `gh auth switch --hostname github.com --user lyutvs` → `git push origin HEAD:main` → `gh auth switch --hostname github.com --user asher_wrtn`.
- **긴 실행은 컨트롤러가** Bash `run_in_background`로, timeout 인자 없이 돌린다. 서브에이전트는 `results/`에 쓰지 않고 실데이터를 돌리지 않는다(작성·커밋만).
- **FlyPool을 만드는 스크립트는 `if __name__ == "__main__":` 가드 필수**(spawn 풀이 다시 import한다). 워커 job은 `flymon/` 안의 모듈 수준 함수로 둔다.
- **전체 스위트**: `uv run pytest -q -rfE -o addopts=""` → 이 계획 끝에서 **538 passed, 1 skipped, 3 xfailed**. 두 스위트를 동시에 돌리지 않는다 — 알려진 플레이크
  `tests/test_run_m0d_h3.py::test_a_complete_run_writes_everything_through_both_guards_and_resumes_from_the_cache`가 두 실행이 같은 초를 공유하면 실패할 수 있다.

## 7. 로컬 전용 참고 (git 제외 — 결정이 아니라 배경)

- `.superpowers/drafts/2026-09-22-next-step-prep.md` §1 — Branch A 후보(순위 매김). 천장 결과 전에 쓴 배경 자료이지 결정이 아니다.
- `.superpowers/drafts/2026-09-22-h4-testability-diag.md` — H.4a.6의 원 작성본.
- `results/m0d/diag/h4_specificity_ceiling{,_all}{,_summary}.json` — 천장 원자료(H.4a.8). 요약 둘은 `m2_nogo.json`의 `inputs_sha256`로 묶여 있다.
- `results/m2/d6a/g8.json` — G.8 원자료(`inputs_sha256`로 묶임). 30 스파이크 38건은 여기서 셌다.
