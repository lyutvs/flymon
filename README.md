# FlyMon

MaleCNS v1.0 초파리 뇌 커넥톰의 LIF 시뮬레이션이 포켓몬 1세대 OU **규칙** 위의 제한 과제(16종·제약 기술 풀)에서 공격기 선택을 배우게 한다.
M0: 엔진과 flybrain 측정값 재현. M1: Showdown 배틀 환경과 뇌 없는 파일럿.
M2: 인코더·판독·학습 단위 시험 — no-go(시험 불성립). STD 지렛대 — 닫힘(부록 J.13). 빠른 KC→KC 억제 — 닫힘(부록 K.9). 순진 판독 선별 — 닫힘(부록 L.12). 판독 확장(가르치는 구획 재선택) — 닫힘(부록 M.11). 실제 냄새(Lin 2014 패턴) — N1에서 닫힘(부록 N.9). 주장 재범위(기계 대조 + 승률) — 두 층 모두 시험하지 않고 닫힘(재범위 스펙 11.3·11.4). C3 두 상태·제시 유발 억압 특성화(학습 주장 아님) — 기록(부록 O.8·O.9). 실제 냄새 다른 쌍의 처벌 학습 — **학습함(확인 시험, 부록 P.7)**. 포켓몬 인코더 재설계(사구체 서로소 결합 부호 E-grid) — 닫힘(`STOP_EVEN_LOW`: KC 겹침은 사라졌으나 보상 판독이 막아 testable_b 7/21, 인코더 스펙 13절). M3 인프라만 완료(판정 없음), M4는 보류.

**M0 결과: 부분 통과 (희소성·기저 발화·채널별 냄새 특이 억제 통과, 합성 지수 반전 미달)**

**M1 결과: 통과 (풀 게이트 MAX − RND 승률 0.319 ≥ 0.15, 95% CI [0.291, 0.344])**

**M2 결과: no-go — 시험 불성립 (2026-09-22). 학습 단위 시험을 세울 쌍이 선언된 기준에 못 미쳐 시험이 돌지 않았다. "학습 안 됨"이 아니다 — 스펙 부록 I.**

## 측정된 것 / 우리가 정한 것 / 안 된 것

- **측정된 것(데이터)**: 뉴런 연결, 시냅스 수, 신경전달물질 예측(MaleCNS v1.0).
- **실제 데이터 M0 측정**(`kc_thresh` 1.5, `apl_scale` 0.1, `mbon_hold_frac` 0.85):
  - 설계된 8-사구체 냄새쌍의 KC 희소성 6.2% / 5.8%, Jaccard 0.023 < 우연 수준 0.031 —
    두 냄새 표현이 오히려 탈상관된다. `kc_thresh` 1.0에서는 15.9% / 10.8%에 Jaccard가 우연 수준을 넘는다.
  - `mbon_hold_frac` 0이면 뇌 전체가 침묵한다(2초 동안 0 스파이크). 0.85에서는 MBON 활동이 자기지속적인
    콜린성 clique로 번진다: FR1 18개(class CX)가 상호 시냅스 8,694개(세포당 483개)로 ~260–300 Hz에 포화하고,
    110–165개 뉴런(FR1, 날개 동력 운동뉴런 DLMn/DVMn, 상행뉴런, 간혹 KCab-p 일부)을 100 Hz 위로 끌어올린다.
    여기에 MBON 97개 중 2–5개가 포함된다(FR1→MBON30 시냅스 809개를 통한 MBON30, 때때로 MBON06/07).
    포화한 MBON 중 PPL105(MBON13/18)·PAM08(MBON05/21) core 집합(서로 disjoint)에 속하는 것은 없다.
  - 따라서 MBON 기저의 raw 평균은 포화 세포에 지배되어 불안정하다(5–12 Hz, 시드 간 sd 최대 4 Hz).
    100 Hz 초과 세포를 제외하면 hold 0.85에서 기저 평균은 3시드 × 3초에 걸쳐 3.52 ± 1.03 Hz,
    활성 MBON 타입 34–36개로 flybrain 보고값(3.4 Hz / 33타입)과 일치한다.
  - 휴지 상태 엔진 속도: 스텝당 0.83 ms(뉴런 162,517개, CSC 엣지 6.0 M).
  - DAN 내인성 발화(세포당 800 ms 기준): **PPL101은 상시 활성**이다 — 냄새가 없는 휴지 상태에서 ~119 Hz,
    냄새 제시 중 104–166 Hz이고 70 mV로 구동해도 194–236 Hz까지밖에 오르지 않아 위상성 교사 신호로 쓸 수 없다.
    **PPL105는 조용하고 냄새 특이적**이다(휴지 0 Hz, 냄새 A 0 Hz, 냄새 B 14 Hz). **PAM08은 내인성으로 침묵**하며
    (50개 세포에서 0–2 스파이크) 구동하면 세포당 165 Hz다.
  - **희소성 게이트 통과 수치**: KC 활성 6.2% / 5.8%(게이트 3–7%), Jaccard 0.023 ≤ 우연 0.031,
    MBON 기저 절사 평균 3.52 ± 1.03 Hz(게이트 3–4 Hz), 활성 MBON 타입 35개.
  - **채널별 냄새 특이 억제(원시 프로브 카운트, 600 ms 기준, 8/8 시드)**: 보상 채널 PAM08을 냄새 B와
    짝지으면 MBON05의 B 반응이 사라지고(35 → 0) A 반응은 일부만 줄어든다(33 → 22). 냄새 A와 짝지으면
    A 반응이 사라지고(33 → 0) B는 일부만 준다(35 → 15). 처벌 채널 PPL105를 냄새 B와 짝지으면 MBON13의
    B 반응이 사라지고(31 → 0–1), 냄새 A와 짝지은 팔에서는 같은 B 반응이 절반만 준다(31 → 12–19).
    MBON13의 A 반응은 학습 전에도 0이라 억압할 것이 없다. 즉 두 채널 모두 **짝지은 냄새를 따라가는
    냄새 특이적 억압**을 보인다. 이것은 아래 "안 된 것"의 사전 등록 기준을 대체하지 않는다 —
    실패한 기준은 실패한 채로 둔다.
  - 고전적 판별 지수 `(x+ − x−)/(x+ + x−)`는 한쪽 카운트가 0이면 ±1에 포화한다. 희소 영역에서 PPL105 core의
    유일한 반응 세포(MBON13)는 두 냄새 중 하나에만 답하므로(600 ms에 0 대 27–41 스파이크) disc_A는 학습 전부터
    ±1에 고정되고, 반응하던 냄새가 실제로 억압되어도(A− 33 → 10) disc_A = −1 그대로다 — 학습이 없는 것이 아니라
    측정 방식의 한계다.
- **우리가 정한 것**: LIF 상수(Shiu et al. 2024), lLN1/lLN2 억제 재지정, APL 출력 스케일, MBON 기저 구동,
  KC 역치 정규화, 반구 보정, 가소성 규칙과 상수, 불응기 회계(스파이크 후 `refrac_steps()` 스텝만큼 더 지나야 다시
  발화 가능 — dt = 1 ms에서 실제 3 ms, 2.2 ms보다 한 스텝 길다), 막전위 하한 `v >= -v_thresh`.
  값은 `results/summary/m0.json`의 `params_frozen`.
- **우리가 정한 것(게이트 통계)**: MBON 기저 게이트 통계는 3시드 × 3초의 **절사 평균**(100 Hz 이하 세포만)이다 —
  자기지속적인 FR1 clique가 MBON 몇 개를 포화시켜 raw 평균이 시드마다 흔들리기 때문이다. `mbon_hold_frac`은
  0.85로 두고(0이면 뇌가 침묵한다), `kc_thresh`는 1.5로 고정한다(1.0은 KC 겹침이 우연 수준을 넘는다).
  이런 clique를 안정화할 기전인 단기 시냅스 억압(short-term depression)은 M0에 없으며 후속 과제로 남긴다.
- **우리가 정한 것(판별 지수)**: 게이트의 판별 지수는 **graded(비포화) 지수**다 —
  `disc_graded(x+, x−, norm) = (x+ − x−) / max(norm, 1)`이고 `norm`은 **같은 시드·같은 팔의 학습 전(naive) 합계**
  (`A+_pre + A−_pre`, `P+_pre + P−_pre`)다. 고전 지수는 현재 합계로 나누기 때문에 한쪽이 0이면 ±1에 포화해
  희소 영역에서 실제 학습을 전혀 보여주지 못한다(위 "측정된 것" 참조). 고정된 학습 전 합계로 정규화하면
  같은 변화가 등급으로 나타난다. 고전 지수도 계속 계산해 `D_pre_disc`/`D_post_disc`/`dD_disc`와 요약의
  `n_flip_disc`·팔별 `mean_dD_disc`로 함께 보고하며, 네 번의 프로브 원시 카운트는 `counts`에 남는다.
  `noplast` 팔은 두 지수 모두에서 정확히 0.0이다. 다만 graded 지수로도 게이트 기준(부호 반전)은
  **통과하지 못했다** — 아래 "안 된 것" 참조.
- **우리가 정한 것(도파민 채널)**: 실제 데이터 조건화 게이트의 처벌 채널은 **PPL105**(core MBON13/18/23,
  flybrain과 같은 선택), 보상 채널은 **PAM08**(core MBON05/21)이다(두 core는 disjoint). PPL105는 내인성으로
  조용하고 냄새 특이적이라 위상성 교사 신호를 실을 수 있다. **PPL101**(γ1pedc, core MBON11)도 시험했으나
  **기각**했다 — 휴지 상태에서 이미 ~119 Hz로 상시 발화해 구동해도 위상성 대비가 거의 생기지 않기 때문이다
  (Aso et al. 2014의 표준 혐오 구획이지만 우리 엔진에서는 교사 채널로 못 쓴다). PPL105 core 판독이
  냄새 하나에만 반응해 지수가 ±1에 붙던 **측정 방식의 포화**는 위의 graded 지수로 걷어냈다. 그러나 판독이
  한 냄새에만 답한다는 사실 자체는 그대로여서, 채널을 바꾸든 지수를 바꾸든 사전 등록한 부호 반전 기준은
  통과하지 못했다("안 된 것" 참조).
  CLI 기본값은 PPL105/PAM08이고(`conditioning --punish-type`으로 바꾼다), 사용한 채널은 M0 요약의
  `conditioning` 블록에 기록된다.
- **우리가 정한 것(위상성 도파민)**: 조건화 프로토콜은 냄새를 켠 뒤 **800 ms settle 구간**(DAN 구동 없음,
  가중치 동결, 트레이스와 `da_base`는 계속 적분)을 두어 도파민 기저선이 *냄새로 유발된* DAN 수준에 적응한
  다음에 펄스를 준다(800 ms = `da_baseline_ms`의 4배, 기저선이 냄새 유발 수준의 ~98%까지 따라온다.
  3배인 600 ms에서는 적응된 상시 DAN이 여전히 위상성 펄스의 52%만큼 억압한다 — 합성 픽스처 측정, 800 ms에서는 22%). `da_baseline_ms`도 1000 → **200 ms**로 줄였다. 이유: 실제 커넥톰에서 DAN(특히 PPL101)은
  냄새 제시 중 스스로 발화하는데, 매 제시마다 트레이스를 0으로 리셋하면 이 상시 수준이 통째로 "놀라움"으로
  계산된다. 그 결과 12 trial 뒤 PPL101을 한 번도 구동하지 않는 `reward_only`를 포함해 **모든** 가소성 팔이
  PPL101 core 시냅스를 ~8% 억압했고(weights 0.918) MBON11이 두 냄새 모두에 침묵했다(A+ = A− = 0). 보상 채널은
  특이적이었다 — PAM08을 냄새 A와 짝지은 팔만 P+를 없앴다(34 → 0). 기저선이 펄스 전에 적응하면 위상성 성분만
  학습에 쓰인다.
- **재현 결과**: `results/summary/m0.json` — KC 희소성/겹침, MBON 기저, 조건화 8시드 결과와
  `gate` 블록(`sparsity_ok` / `conditioning_index_flip_ok` / `conditioning_channel_specific_ok` /
  `passed` / `partial`). 현재 실행은 `passed: false`, `partial: true`다.
- **안 된 것(사전 등록 기준 실패)**: 조건화 게이트의 사전 등록 기준 —
  *graded 지수 D가 `both`와 `reversed` 사이에서 부호를 뒤집고(8/8 시드), 팔별 |mean dD| ≥ 0.3* —
  은 **통과하지 못했다**. 동결 조건에서 graded n_flip은 **0/8**이고 두 팔의 mean dD는 모두 양수다
  (both +0.37, reversed +1.44). 고전 포화 지수로는 8/8이 나오지만 이는 퇴화한 값이다 —
  PPL105 core에서 유일하게 반응하는 세포 MBON13이 냄새 A에는 아예 답하지 않아 지수가 ±1에 고정되기
  때문이며, 학습의 증거로 쓸 수 없다. 짝지음 자체는 검증됐다(`noplast` 팔 dD가 모든 실행에서 정확히 0.0).
  원인은 합성 지수의 접근(approach) 항이 이 엔진에서 한쪽으로만 열려 있다는 것이다: 처벌 채널 판독이
  한 냄새에만 반응하므로 억압은 한 방향으로만 나타나고, 두 채널을 합산한 지수는 부호를 뒤집을 수 없다.
  시도한 실행은 `results/m0/gate_runs.md`에 표로 남겼다 — PPL105 구 프로토콜 2/8, PPL101 구 프로토콜 5/8
  (PPL101은 휴지 ~119 Hz 상시 발화로 기각), PPL105 새 프로토콜 0/8(disc 8/8),
  trace 스케일 10/5 실행 2/8(disc 4/8, 기록용으로만 보관, 채택 안 함). 후속 과제는 단기 시냅스 억압
  (Tsodyks–Markram STD)이며 M0에는 없다. M0는 이 기준 실패를 안은 채 **부분 통과**로 닫는다.
- **실제 M1 파일럿 측정**(뇌 없음, 팔당 16마리 × 100배틀, 상대 SimpleHeuristicsPlayer, 서버 난수 비고정):
  승률 RND 0.152 · MAX 0.471 · WEAK-RND 0.145 · WEAK-MAX 0.485, 동점·미완료 0. 마리 간 SD는 이항 하한과 같다
  (마리 수준 추가 분산 ≈ 0). 약한 코치(교체·보조기 없음)와 강한 코치의 승률이 같다 — 이 풀에서 승률 차이는 공격기 선택이
  만든다. 결정 제공자가 결정 레코드의 82–83%를 맡고 후보는 평균 2.47개다. 요약 `results/summary/m1_pilot.json`,
  일정 `results/summary/schedule_m1.json`, 자세한 것은 스펙 부록 B.
- **우리가 정한 것(M1)**: 16마리 풀과 기술 허용 목록(`docs/pool.md`, Showdown `validate-team gen1ou`가 최종 권위), 코치 v1 =
  poke-env SimpleHeuristicsPlayer 규칙(스탯 추정이 3세대 이후 공식이라 1세대에서는 근사), MAX = 코치 자체의 공격 점수,
  라우터(공격기 & 후보 ≥ 2 → 제공자), 귀속 규칙(직접 피해만, 잔여·대타·빗나감·불확실은 무신호), 배리어(100 ms 마감 또는 활성 전원 대기).
- **실제 M0b 측정**(CPU 프로세스 풀, 실제 커넥톰): 16워커 엔드투엔드 예산 42.2시간(학습 결정 52,000 + 평가 결정 81,600; 학습만 19.3시간) ≤ 60시간 게이트 **통과**. 풀로 재실행한 M0 조건화 40건(5팔 × 8시드)이 `results/m0/conditioning.json`과 비트 동일, 희소성·기저도 차이 0.0, 풀의 결정이 인프로세스 엔진과 동일. 워커 RSS 0.57 GB(최대 0.79), C-shuf 배선 변형 하나 +0.15 GB. 휴지 3초에서 100 Hz 초과 뉴런 156개 중 KC 17개, 스파이크의 29% — 스펙 A.5의 STD 재검토 조건 충족 → M0c에서 해소(스펙 부록 D). 요약 `results/summary/m0b.json`, 자세한 것은 스펙 부록 C.7.
- **우리가 정한 것(M0b)**: MPS 배치 엔진 대신 기존 CPU 엔진의 프로세스 풀(레드팀 실측, 스펙 C.6). 마리별로 남는 상태는 KC→MBON 가중치·켬/끔·배선 변형뿐이고 부모가 보관한다. 결정은 후보를 같은 시드로 순차 제시(짝지은 잡음), 강화는 M0 `train_block`의 한 프레젠테이션. 등가성 게이트는 통계 일치가 아니라 비트 동일 재현. STD는 여전히 보류(A.5).
- **실제 M0c 측정**(KC→KC 제거 엔진, 실제 커넥톰, 사전 등록 게이트 스펙 D.4): **통과**. KC 희소성 6.5% / 4.6%, Jaccard 0.025 ≤ 우연 0.028; MBON 기저 절사 평균 8시드 3.14 ± 0.86 Hz(3–4 Hz); 폭주 — 휴지 8시드 100 Hz 초과 KC 0개, 냄새 B 결정 창 64시드 150 Hz 초과 KC 0개(최대 KC 발화율 118 Hz, PN 구동 후각 KC); 등가성 — 옛 엔진(`kc_kc_scale` 1.0)이 `results/m0` 조건화 40/40 비트 동일·희소성 차이 0.0, 새 엔진 풀 = 인프로세스(팔 2/2, decide 동일); 처리량 16워커 엔드투엔드 43.2시간 ≤ 60시간(M0b 42.2). 조건화 사전 등록 기준은 예측대로 **실패**(판정 시드 8–15: graded n_flip 6/8, both -0.29, reversed +1.24, 채널 특이 억제 7/8) — MBON13의 한쪽 냄새 무반응은 KC→KC와 무관하다. 요약 `results/summary/m0c.json`, 자세한 것은 스펙 부록 D.8.
- **우리가 정한 것(M0c, KC→KC 제거)**: KC→KC 엣지는 기본 제거한다(`kc_kc_scale` 0.0). M0b가 잰 "휴지 100 Hz 초과 집합의 KC 17개"는 시드 평균(0 / 52 / 0)이었고,
  실체는 KCab-p 우반구 62개의 KC→KC 재귀 흥분 clique였다(냄새 B가 64시드 중 54개에서 점화, 판독 core에 직접 시냅스 0). KC 축삭간 접촉은
  mAChR-B 매개 억제라(Manoim et al. 2022) 빠른 흥분으로 두는 것이 근거 없는 선택이었다. 이는 lLN1/lLN2와 같은 급의 데이터 라벨 재정의다.
  `kc_kc_scale=1.0`은 M0·M0b 엔진이며 `results/m0/`·`results/summary/m0.json`·`m0b.json`은 옛 엔진의 기록으로 불변이다. 게이트는 실행 전에
  사전 등록했고(스펙 부록 D.4: 희소성 ∧ 기저 8시드 ∧ 폭주 ∧ 등가성 ∧ 처리량; 조건화 기준은 그대로 두고 시드 8–15로 판정·기록), 파라미터는
  게이트 결과를 보고 조정하지 않는다. STD는 재보류(재검토 조건은 D.6의 관측치). 알려진 한계: PN 입력이 0인 KC 297개(αβp·γd)의 역치 정규화는
  정의되지 않아 클립 하한을 받는다(KC→KC 제거 후 실질 영향 없음).
- **실제 M2 첫 측정**(2026-09-16, 스펙 부록 E, 게이트 아님): 잠정 인코더 = 스펙 3.3 그대로(41채널), 팀 풀 16턴 × 짝지은 잡음 8시드, 328 프레젠테이션. **D.6 (a) 미충족(단 최대 서브창이 정확히 150.0 Hz로 여유 0, 격자 정렬 타일·8시드 — 이동 창과 64시드로 재판정 필요), (b) 충족**(16턴 중 3턴이 비율 2 초과, 최대 2.50), **(c) 미측정**(학습 단위 시험은 아직 안 돌았다). 후보 냄새 희소성은 3% 하한 위반 0건(최소 3.08%)이지만 스펙 5의 5~9% 대역은 328건 중 220건(67%)만 만족. MBON13은 실제 후보 냄새에 반응하나 16턴 중 4~5턴에서 무발화, MBON18·MBON21은 전 턴 무발화. 동일 총 ORN 구동에서 51 사구체의 KC 구동이 2~5017 스파이크(2867배)로 퍼지고 `corr(ORN, PN) = −0.015` — 스펙 3.3의 1/수용체수 균등화는 KC 수준에서 작동하지 않는다. 요약 `results/summary/m2_probe.json`, 원자료 `results/m2/`(git 제외).
- **안 된 것(M2 첫 측정)**: 이 측정으로는 인코더 어휘를 고를 수 없다. 레드팀 2회(외부 모델 `gpt-5.6-sol`, `gpt-6-astra`) 모두 RETHINK. 네 가지가 각각 단독으로 근거를 무너뜨린다 — (1) 사전 등록 1차 기준(4.3 #1)이 판정하는 **상대 타입만 다른 상황 쌍이 데이터에 0개**, (2) 탐색/확인 세트 분리(4.4)가 소진됨(팀 풀 16마리 전부 사용, 홀드아웃 없음), (3) 어휘 비교가 교락됨(사구체 선택 규칙·어휘 구조·채널 복제 수·강도를 동시 변경), (4) 엔진 기각과 인코더 기각에 서로 다른 잣대(평균 vs 최소·실패 턴). 정정 두 건: D.6 (c)를 희소성 검사와 혼동해 "통과"라 적은 것, 그리고 "STD는 구동량 차이를 고치지 않는다"는 주장 — STD는 활동 의존 억압이라 오히려 구동 분산을 압축한다(스펙 E.3에서 철회). 엔진 근사 두 개는 M0d 후보로 기록만 했다: APL을 스파이킹 LIF로 둔 것(실제로는 비발화, Amin et al. 2020; 설계 강도에서 LIF 상한의 88.8%로 포화)과 ORN→PN 이득 제어 부재(Olsen et al. 2010).
- **실제 M2 보정 — 오라클 상한**(2026-09-16, 스펙 G.10, D.6 (c) 판정 아님): 학습 대신 KC→MBON 가중치를 직접 깎아 "가르친 냄새에 최대로 특이적인 감소가 일어났다면
  판독이 기준(보상 d′ ≥ +2, 처벌 하락 d′ ≤ −2)에 닿는가"를 쟀다. 후보쌍 **8/34**만 시험 가능했고(1채널 2/19, 2채널 6/15), 시험 가능한 쌍은 NORMAL 기술이 끼는 쌍에
  몰렸다(7/8 — KC를 거의 구동하지 못하는 사구체가 만든 비대칭일 수 있다, 기전 미확인). F v3의 게이트 3쌍은 전부 시험 불가였다 — v3를 돌렸다면 학습 실패가 아니라
  구조 때문에 FAIL이 났을 것이다. 순진 d′ ≈ 0 ∧ 시험 가능 ∧ 바닥 가드 통과인 홀수 턴 게이트 후보는 0쌍. 설계쌍은 시험 가능(보상 18.69 / 처벌 −6.27).
  요약 `results/summary/m2_oracle.json`.
- **실제 M2 인코더 비교**(2026-09-17, 스펙 G.12·G.13, 짝수 턴): 인코더 네 개(E0 스펙 3.3, E1 KC 구동 기준 배정, E2·E3 기술 채널 복제)의 점수 = min(턴 내 기술 축,
  상대 타입 축의 시험 가능 비율)이 E0 0.222 · E1 0.143 · E2 0.143 · E3 0.190으로 모두 0.5 미만 → 사전 선언한 규칙대로 멈췄다. 처벌 판독(MBON13)이 늘 마지막 관문이고,
  강한 사구체를 두 축이 나눠 갖는 제로섬이라 어느 하나만 고쳐서는 0.5에 닿지 않는다. 요약 `results/summary/m2_encoders.json`.
- **안 된 것(엔진 전제 프로브, 스펙 G.14.8)**: 등급 APL(E.5)이 상대 타입 축 병목을 줄이는지 보려던 프로브는 스모크에서 설계 결함이 드러나 전체 실행 전에 중단했다.
  등급 APL은 보상 판독 MBON05를 직접 억제하고(APL의 MBON 표적 1위), 활성을 맞춘 스파이킹 팔에서는 처벌 판독 MBON13이 바닥이라, 대비가 편차 완화가 아니라
  판독 세포의 바닥 효과를 쟀을 것이다. 판정은 내지 않았다. 등급 APL을 그대로 바꿔 끼우는 것은 엔진 교정이 아니다 → M0d(스펙 부록 H).
- **우리가 정한 것(M0d, 스펙 H.1–H.3a)**: 조합 C0 현재 엔진(대조) · C1 포화형 등급 APL + MBON 재조정 · C3 C1 + KC 활성 항상성 역치. C2(C1 + ORN→PN STD)는
  문헌 상수(f 0.78, τ 893 ms)에서 ALPN 스파이크가 C1의 0.08–0.11배로 무너져 자격점이 없어 탈락했다(이 엔진·이 상수·이 탐색 범위에서). 모든 변경은 기본값 꺼짐이고,
  엔진 기본값은 M0c 엔진 그대로다.
- **실제 M0d 측정 — 작동점 고정**(2026-09-21, 스펙 H.3a.13): 세 조합 모두 채택. C1 `kc_thresh` 1.65 · `apl_input_scale` 0.137 · `mbon_hold_frac` 0.853
  (기준 KC 활성 5.95%, 게이트 32시드 기저 3.66 Hz), C3 같은 `kc_thresh` + 항상성 역치 · `apl_input_scale` 0.609 · `mbon_hold_frac` 0.849(5.54%, 3.08 Hz),
  세 조합 모두 폭주 0. 요약 `results/summary/m0d.json` 블록 `"h3"`.
- **실제 M0d 측정 — 조합 선택**(2026-09-22, 스펙 H.4a.5, 짝수 턴 39쌍): 세 조합 모두 판독 A = MBON13, P = MBON05. 오라클로 시험 가능한 상대 타입 축 쌍
  C3 **7/21**(F_a 2) · C0 4/21(F_a 1) · C1 2/21(F_a 0) → 자격은 C3 하나, T_b 0.333 < 0.5 → `STOP_LOW_T_B`. 요약 `results/summary/m0d.json` 블록 `"h4"`.
- **실제 M0d 진단과 특이성 천장**(2026-09-22, 스펙 H.4a.6–H.4a.8, 선택에 쓴 짝수 턴 데이터의 사후 분석): 학습 세기·시드 잡음·판독 가중은 원인이 아니고, 판독 바닥은
  C3가 대부분 고쳤다(바닥 통과 14쌍 중 시험 가능 7). 남은 한계는 편집된 시냅스의 X/Y 특이성 — 가르치지 않은 Y의 판독이 X의 54–80%만큼 같이 움직인다.
  편집을 X에만 활성인 KC로 제한한 천장(판독 규칙을 결과 전에 고정: C3 ≥ 14/21이면 새 선언, ≤ 10/21이면 no-go)은 특이성을 얻었으나(Y/X → 0) 처벌 조건이 무너져
  (−p ≥ 2인 쌍 C3 11 → 3 · 4) C3 `freq` 2/21 · `all` 3/21 → **no-go**. MBON13(처벌 판독)을 움직이는 X의 구동은 대부분 Y와 공유된 KC에 있다.
- **안 된 것(M2 no-go — 시험 불성립, 2026-09-22, 스펙 부록 I)**: 스펙 5의 M2 학습 단위 시험(순진 d′ ≈ 0 → 보상 20회 뒤 d′ ≥ 1 → 처벌 20회 뒤 하락)은
  **한 번도 돌지 않았다.** 시험한 인코더(E0–E3)와 엔진(C0·C1·C3)에서, 이상화한 특이적 가중치 감소로도 시험을 세울 수 있는 쌍이 선언된 기준(엔진은 상대 타입 축 시험 가능
  비율 ≥ 0.5 ∧ 순진 균형 턴 내 쌍 ≥ 2, 인코더는 두 축 중 낮은 시험 가능 비율 ≥ 0.5)에 못 미쳤다. 그래서 이것은 "학습 안 됨"(D.6 (c) 실패)이 **아니라** 시험 불성립이다. 사용자 결정으로 스펙이 이름 붙인
  3번 경로(범위 축소 / M2 no-go 기록 + 주장 재설계)를 택했다.
  - 학습 시험은 설계쌍(기계 대조)으로 한정한다. 0절의 1차 주장(상대 타입 조건부 선호 학습)은 4.3 기준 1이 현재 형태로 **지지되지 않는다** — 시험하지 않았다.
    M3(에이전트 루프)·M4(실험)와 2차 주장(승률 기여)은 **보류**(폐기 아님).
  - D.6: (a) 충족 — G.8 재판정(41냄새 × 64시드, 1 ms씩 움직이는 200 ms 창)에서 2624 제시 중 7건이 150 Hz 초과(최대 31 스파이크) · (b) 발동(E.2: 16턴 중 3턴이 비율 2 초과) · (c) 미측정. STD 재설계 여부는 새 주장 선언의 첫 질문으로 넘겼다.
  - 원인을 하나로 귀속하지 않는다: 판독이 두 타입뿐, 오라클은 이상화된 편집이고 천장은 현재 KC 부호의 상한(인코더·국소 APL·발화율 정규화 같은 KC 부호 변경은
    시험하지 않음), 짝수 턴 21쌍뿐, 채널→사구체 배정이 임의적(2867배), C0 대 C1·C3 기저 교락.
  - 판정은 `scripts/write_m2_nogo_summary.py`가 기록된 요약에서 파생한다(한 고리라도 성립하지 않으면 쓰기를 거부). 요약 `results/summary/m2_nogo.json`.
    M2 go 조건은 strict xfail 테스트 `tests/test_m2_go.py`로 남는다 — 조건이 충족되는 날 XPASS로 깨진다.
- **안 된 것(STD 지렛대, 2026-09-25, 스펙 부록 J.13)**: M2 no-go 뒤의 새 주장 선언(부록 J)은 ORN→PN 단기 억압(STD)이 병목을 푸는지 직접 쟀다. f × τ 9점 스캔에서 전체 이득을 되돌렸고,
  KC 부호 분산은 최대 26.9% 줄었다. 그러나 순위 상위 두 설정은 C3 규칙 재수렴에서 자격점이 없었다. 3위 설정(f 0.95, τ 100 ms)의 재수렴 엔진에서는 상대 타입 축 시험 가능 쌍이
  **4/21**로 C3(7/21)보다도 낮았다. 사전 선언 구간 B_Tb에 따라 **STD는 이 병목의 지렛대로 닫는다.** D.6: (a) 미발동(최대 110 Hz), (b) 발동(16턴 중 5턴).
  축소 주장의 학습 시험(B)은 보정할 수 없어 닫혔으므로(J.12.8) 축소 주장은 시험하지 않음이다. 다음은 새 주장 선언이다.
- **안 된 것(빠른 KC→KC 억제, 2026-09-27, 스펙 부록 K.9)**: J 다음 선언(부록 K)은 PN 아래, KC 부호에서 X/Y 겹침을 줄이려고 KC→KC 엣지를 빠른 억제로 되살렸다
  (`kc_kc_scale` g ∈ {−0.05, −0.1, −0.2, −0.4, −0.8}, 설정마다 C3 규칙 재수렴). 홀수 턴 (b) 20쌍으로 잰 X 전용 MBON13 구동은 C3 대비 최대 +3.5%에 그쳤고(−0.4는 자격점 없음),
  선택된 g = −0.1(+3.3%)의 엔진에서 짝수 턴 상대 타입 축 시험 가능 쌍은 **4/21**(C3 7/21)이었다. 사전 선언 구간 B_Tb에 따라 **빠른 KC→KC 억제는 이 병목의 지렛대로 닫는다.**
  KC별 항상성 재수렴이 억제의 효과를 거의 되돌렸다. D.6: (a) 미발동(최대 105 Hz), (b) 발동(16턴 중 6턴). 다음은 새 주장 선언이다.
- **안 된 것(순진 판독 선별, 2026-09-27, 스펙 부록 L.12)**: K 다음 선언(부록 L)은 엔진을 바꾸지 않고 주장을 줄였다 — C3 엔진에서, 학습 전 판독으로 미리 고른 상대 타입 축 쌍만 시험한다.
  짝수 21쌍 + 홀수 20쌍(C3 오라클로 새로 라벨, 시험 가능 2/20)으로 "MBON13 바닥 가드 + 지표 하나의 문턱"을 맞췄고, 최종 규칙 G ∧ S > 0.186의 턴 단위 LOO 정밀도가
  **7/15 = 0.467**로 게이트 0.6에 못 미쳤다(`SCREEN_IMPRECISE`). 사전 선언대로 **순진 판독 선별 주장을 닫고**, 새 세트는 돌리지 않았다.
  C3의 D.6: (a) 미발동(최대 100 Hz — 옛 기록의 (a) 충족은 C0 엔진의 값이었다), (b) 발동(16턴 중 7턴). M3 인프라(에이전트 루프·체크포인트·로그·스모크)는 판정 없이 병행 완료했다. 다음은 새 주장 선언이다.
- **안 된 것(판독 확장, 2026-09-28, 스펙 부록 M.11)**: 판독이 MBON13·MBON05 두 타입뿐이라는 대안 설명을 겨눠, 가르치는 DAN 구획(PPL1 처벌 × PAM 보상)을 다시 고르고
  그 core MBON 세포 집단으로 읽었다. core KC 입력의 X 전용 몫이 현 조합보다 큰 후보(보상 5·처벌 2) 가운데 반응 가드를 통과한 것을 팔별로 스캔하고 상위 조합을 함께 편집했더니
  최선 PPL106·PAM12도 짝수 턴 시험 가능 **7/21**(C3와 같음, F_a 1)로 M2 기준에 못 미쳤다(`STOP_NO_GAIN`). 판정 세트는 쓰지 않았다. 이것으로 이번 라운드의 지렛대 탐색을 끝낸다.
- **안 된 것(실제 냄새, 2026-09-30, 스펙 부록 N.9)**: 포켓몬 쌍 대신 Hallem 2006 실측 반응(DoOR 매핑)으로 만든 IA:EB 4:1 대 1:4(비슷한 쌍)·4:1 대 δ-DL(다른 쌍)을 C3 엔진에 넣어
  Lin 2014의 APL 의존 미세 변별 패턴을 겨눴다. 작동점은 잡혔고(g 0.25, c_δ 8, KC 활성 5.19%) 유사도 순서도 섰지만(Δr 0.336, 95% CI [0.296, 0.356]),
  처벌만 오라클에서 비슷한 쌍이 p0 **−1.955**로 기준 −2에 못 미쳤다(다른 쌍 −3.734는 통과, `STOP_UNTESTABLE`). 사전 선언대로 N1에서 닫았고 학습 판정(N2)은 재지 않았다.
  상한이 아니라 "사전 지정 편집 프로토콜에서 시험 불성립"이다.
- **안 된 것(주장 재범위, 2026-09-30, 재범위 스펙 11.3·11.4)**: 1차 주장을 "보상 쪽 기계 대조(C3, 설계쌍)"와 "배틀 승률(FLY 대 RND·가소성 끈 초파리·yoked 강화)" 두 층으로 다시 잡았다
  (`docs/superpowers/specs/2026-09-28-rescoped-claim-design.md`). 1차는 양성 대조 seed0의 Y가 16시드 중 4시드에서 조용한 상태라 자격을 얻지 못해(`STOP_CONTROL_INVALID`) 시험하지 않았다.
  2차는 검정력 파일럿(팔마다 6마리 × 평가 20배틀)의 분산으로 계산한 (2b) 동시 검정력이 사전 고정 격자 어디에서도 0.8에 닿지 않았다 — 60시간 안 최대 **0.140**(F 32·E 40),
  격자 전체 최대 0.186(F 32·E 300, 234.8시간) (`STOP_POWER`). 판정 실행은 돌리지 않았고 (2a)·(2b)는 재지 않았다. 결과를 본 뒤 규칙·수치는 바꾸지 않았다.
  "승률 효과 없음"이 아니라 "이 규모에서 시험 불성립"이다.
- **특성화(C3 두 상태·제시 유발 억압, 2026-10-01, 스펙 부록 O.8·O.9 — 학습 주장 아님)**: 실제 냄새 rig에서 두 현상을 사전 선언 판정으로 쟀다.
  O1(제시 7 680회): 정지 상태는 낮은 구동(g 0.125–0.25)에만 있고(혼합 칸 6개) 판독이 두 봉우리로 갈리지 않았다(`GRADED`, 중간대 조건 0/6). APL→비KC 출력 차단이 정지를 61% 줄였지만
  사전 기준(75%)에 못 미쳐 경로는 `NEITHER_PATH`, 구동이 클수록 정지가 준다(4자극 `DECREASING`). O2(256팔): 처벌 없는 제시만으로 MBON13이 약 5스파이크(14%) 줄었으나 기준량 안(`NO_DEPRESSION`),
  처벌 효과는 그 위에서 크게 갈라졌다(`SEPARABLE`, dV −2.1·−2.4). 포켓몬 냄새에서 본 "망 쌍안정"과 "처벌을 억압과 가를 수 없음"은 실제 냄새의 이 절차에서는 나타나지 않았다.
- **된 것(첫 학습 판정, 2026-10-01, 스펙 부록 P.7)**: 실제 냄새 다른 쌍(IA:EB 4:1 대 δ-DL)의 처벌 쪽 절대 조건화를 두 방향(각 냄새를 번갈아 처벌)·새 시드 32개로 판정했다.
  처벌이 처벌 없는 같은 제시보다 판독을 더 낮춘 양 ℓ이 4:1 처벌 **1.79 [1.60, 1.97]**, δ-DL 처벌 **2.16 [1.92, 2.38]**로 사전 기준량 c₁ 0.61을 두 방향 모두 넘었고, 변화는 짝 냄새보다 처벌받은 냄새에 몰렸다 →
  **`LEARNS_CONFIRMATORY`**. O2를 본 뒤 선언한 확인 시험이라 발견이 아니라 재현이며, 범위는 C3·이 쌍·처벌 쪽까지다(포켓몬 배틀·비슷한 쌍·보상 쪽 주장이 아니다). 다음은 포켓몬 인코더 재설계 트랙과의 합류다.
- **안 된 것(인코더 재설계, 2026-10-02, 인코더 스펙 `2026-10-01-encoder-redesign-design.md` 13절)**: 포켓몬 상황을 (기술 타입 × 상대 타입) 결합마다 고유한 사구체 패턴으로 바꾸고,
  같은 기술·같은 상대 타입을 공유하는 결합끼리는 사구체를 하나도 공유하지 않게 했다(E-grid, 4설정). 판정 규칙과 새 판정 세트(L 생성기 턴 64–103, 결합 키 중복 제외)는 실행 전에 고정했다.
  KC 활성만으로 강도를 맞추는 단계에서 자격을 얻은 설정은 k2-norm(s 1.0) 하나였고(나머지는 ORN 발화율 상한 아래에서 꼬리·대역 조건에 근소하게 미달), 짝수 턴 오라클에서
  시험 가능 (b) **7/21**·F_a 0으로 M2 기준(11·2)에 못 미쳐 사전 선언대로 멈췄다(`STOP_EVEN_LOW`). 판정 세트는 측정하지 않았다.
  기록(판정 아님): X·Y의 KC Jaccard는 약 0.04로 겹침이 거의 사라졌고 처벌 쪽 통과는 늘었지만((b) 11 → 15), 보상 쪽(MBON05)이 늘지 않아 시험 가능 수가 E0와 같다 —
  이 인코더에서 병목은 KC 겹침에서 보상 판독으로 옮겨 갔다(원인은 재지 않았다). 이 트랙을 이 결과로 닫는다(사용자 결정). 다음은 사용자 결정이며, 보상 쪽 새 선언은 P 트랙 결과 뒤에 검토한다.
- **기록(보상 판독 병목 진단, 2026-10-02, 부록 Q.7 — 특성화, 판정 아님)**: E-grid k2-norm 짝수 (b) 21쌍을 실패 F 12 / 통과 S 9로 고정하고, 다섯 후보를 결과 전 고정 규칙으로 MBON05 단독 r_P에서 봤다.
  ④ 작동점 **일치**(전역 이득 ×1.25에서 F의 r_P 중앙값 +3.1, 순진 P_X와 무관; mv_scale은 입력 구동도 바꾸므로 ①과 완전히 갈린 것은 아니다), ③ 편집 도달·⑤ 변동 **불일치**,
  ① 바닥 **판단 불가**(순진 P_X는 F에서 낮지만(AUC 0.87) s 하향의 차등 악화가 반대 방향, ρ −0.60), ② APL 억제 **판단 불가**(APL→MBON05 2간선 차단이 F의 r_P를 +18 올렸으나 순진 P_X도 함께 올라 편집 비율은 줄었다).
  보상 편집은 F에서 P_X를 거의 움직이지 못한다(ΔP_X 중앙값 −3 대 S −26). 지렛대 선택(부록 R)은 사용자 결정이다.
- **판정(지렛대 APL→MBON05 제거, 2026-10-03, 부록 R.10 — B_처벌가드)**: Q.7의 ②를 지렛대로 선언하고, 쓰지 않은 판정 세트(L 생성기 턴 64–103, (b) 21·(a) 32)에서 M2 기준으로 한 번 판정했다.
  무편집 재현 관문과 관문 ①(KC 0.0457, 대역 안)·②(P 재현 `LEARNS_CONFIRMATORY`)·③(짝수 testable_b 16/21, 지렛대 없음 7/21)을 통과했다.
  판정 세트에서 지렛대 **12/21** 대 지렛대 없음 **6/21**(기준 11·여유 2 통과), F_a 2/32였으나, (a)에서 처벌 통과 3쌍이 바뀌어(17 대 17, pass→fail 3) 처벌 가드가 켜졌다 — **B_처벌가드**.
  지렛대는 보상 통과를 거의 전부 열었다((b) 21/21). 이 결과로 지렛대를 닫지도 선택하지도 않으며, 다음은 사용자 결정이다.
- **판정(같은 지렛대, 마지막 세트, 2026-10-03, 부록 S.10 — B_Fa)**: R 결과를 본 뒤 처벌 가드를 방향성 있게(축마다 처벌 통과 순감소 ≥ 3) 바꾸고, 관문 ②에 같은 시드 L/C 처벌 학습 비(≥ 0.5)를 넣어, 마지막 미사용 세트(L 생성기 턴 104–177, (b) 21·(a) 43)에서 한 번 더 판정했다.
  R의 재현·관문 ①·③을 같은 측정 코드 키로 재사용했고, 관문 ②는 지렛대 아래 처벌 학습이 같은 시드에서 약 1/3 줄었지만(비 0.64 / 0.68) 기준을 넘었다.
  판정 세트에서 지렛대 **13/21** 대 지렛대 없음 **6/21**, 처벌 가드 통과((b) 순감소 2, (a) −1), 그러나 F_a **1/43**(naive_a 3)으로 **B_Fa** — 다음 병목은 순진 균형 (a) 쌍이다.
  이 지렛대의 M2 판정 세트는 더 남지 않으며, 다음은 사용자 결정이다.
- **관문 STOP(엔진별 기준 집합 z, 2026-10-04, 부록 T.10 — STOP_Z_DEGENERATE, 판정 없음)**: z 정규화를 바로잡는 규칙(엔진 변형마다 H.3 기준 집합에서 자기 z)을 결과 전에 고정하고, 상대를 1세대 기본 폼으로 넓힌 새 세트에서 같은 지렛대를 판정하려 했다.
  편집 없는 엔진의 z는 블록 h4 값을 비트 단위로 재현했지만, 지렛대 아래에서 MBON13이 기준 집합 반응성 가드를 넘지 못했다(Δ 중앙값 6.0 → **1.0**, 0 비율 0.146 → **0.302**; MBON05 평균 26.25 → 77.5, σ 비 A 0.62 · P 1.58).
  기록(판정 아님): 이 지렛대는 MBON05를 약 3배 켜고 MBON13의 기준 집합 반응을 끈다 — R·S의 h4 z 판독은 이것을 가렸을 수 있다. T 세트는 측정하지 않아 미사용으로 남는다. 다음은 '부분 제거' 지렛대의 새 선언(부록 U, 사용자 결정)이다.
- **관문 STOP(APL→MBON05 부분 제거 f, 2026-10-04, 부록 U.10 — STOP_NO_QUALIFIED_F, 판정 없음)**: APL→MBON05 2간선 가중치를 f배로 줄이는 지렛대를 엔진별 z로 T 세트에서 판정하려 했다.
  끝점 재현(f = 1 = T 편집 없음, f = 0 = T 지렛대, 비트 일치)은 통과했고, 가드 스캔 9점에서 f 0.6–0.9가 MBON13 가드를 경계값(Δ 5.0)으로 넘었다(f ≤ 0.5는 Δ 2–4).
  검사한 f {0.6, 0.7, 0.8} 모두 T 세트 냄새 4–5개의 KC 활성 중앙값이 0.03 아래(최소 0.019)라 KC 대역에서 떨어졌다 — **편집 없는 엔진도 같은 냄새 4개가 대역 밖**이라 f가 아니라 T 세트 냄새 입력에서 온 STOP이다.
  기록(판정 아님): 기전 대조의 고정 읽기는 출력 차단 '사슬 비지지'(D_block 6.5 대 D_none 5.0), 사슬 진입부 차단 '사슬 지지'(MBON13 Δ 16.0)로 엇갈렸다. 판정 세트의 오라클은 재지 않아 미사용으로 남는다. 다음은 사용자 결정이다.
- **판정 SELECTED(조합 지렛대, 2026-10-05, 부록 V.10)**: APL→MBON05 2간선 제거와 MBON05→MBON09/MBON11/MBON01 11간선 제거를 함께 한 모델(커넥톰 간선 13개 제거)을, 편집 없는 엔진과 조합 엔진 둘 다에서 KC 입력으로 거른 넓힌 풀 새 세트(생성원 턴 0–305)에서 엔진별 z로 한 번 판정했다.
  지렛대 **16/21** 대 지렛대 없음 **6/21**, F_a **2/43**(기준 경계값), 처벌 가드 통과((b) 순감소 −2, (a) 0), 관문 ② 비 h4 z 1.019 · 0.942, z_V 0.773 · 0.704, 짝수 15/21 → **SELECTED**. 오른 것은 주로 보상 통과((b) 8 → 21)이고 처벌 통과는 줄지 않았다.
  귀결은 넓힌 풀·엔진별 z에서 M2 시험 가능성이 섰다는 것까지다 — 지렛대는 U 기전 기록을 보고 골랐고(다섯 번째 시도), 결과는 이 64쌍 조건부이며(군집 모형 null 0.374), POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다. 다음은 사용자 결정이다.
- **관문 STOP(F v4 학습 시험, 2026-10-05, 부록 W.10 — STOP_OC_UNREACHABLE, 판정 없음)**: 조합 지렛대 엔진(V와 같은 13간선, z_V)에서 실제 학습 규칙(F.2 순차 R·N·RN)으로 M2 학습 단위를 판정하려 했다.
  재사용·끝점 재현(V 관문 ② 처벌 행 8개·순진 프로브 6개 비트 일치, 보상 경로 양성)과 파일럿(16쌍 × 8마리, 보상 쌍 비율 1.0, 처벌 0.875, 부호 반대 0.0, X·Y 바닥 0.033)은 통과했다(탐색 판정 11 PASS / 5 FAIL).
  작동 특성에서 어떤 설계(q 0.5–0.75, K 8/16, F 8–32, k 4–8)도 G.6을 맞추지 못했다 — 참 d′ 1.5의 검정력 점 추정 최대 **0.449**(k 4), F에 대해 평평(약 0.82^k: 쌍마다 약 18% 실패, 파일럿 모형의 쌍 수준 이질성과 맞는 해석), d′ 0.5의 거짓 통과 0.0; 부트스트랩 재보정 29/200 실패로 한계 0.000 / 1.000. 드리프트만의 보상 수준 d′ −2.35.
  주 세트(턴 306–1985)는 쓰지 않아 남아 있다. 다음은 사용자 판단이다.
- **관문 STOP(F v4 학습 시험 "절반 이상", 2026-10-06, 부록 X.10 — STOP_OC_UNREACHABLE, 판정 없음)**: W와 같은 지렛대·프로토콜·관문 쌍으로 세트 판정만 "판정 가능 관문 쌍 중 PASS 비율 ≥ p_set(최소 3쌍)"으로 바꾸고 보정 실패를 상태별로 처리하는 X를 선언했고, 레드팀 뒤 점 θ 사전 점검 관문을 앞에 두었다.
  순서 0(W 보정 진단 재현: W의 29회 실패는 모두 처벌 손잡이 b의 미수렴이고 반복 160으로도 회복 0, X의 브래킷 ×4 규칙에서는 b 바닥이 검정력 쪽 79/200 · 거짓 통과 쪽 44/200; 합성 검증·P2-6 통과)는 PASS였다.
  사전 점검에서 750 설계 어디서도 k 4–8 모두 점 검정력 ≥ 0.80에 닿지 않았다 — 최대 p_set 0.5 · q 0.5 · K 16 · F 8의 **0.608 / 0.730 / 0.795 / 0.735 / 0.790**(군집 최악 g 1.0; g 0에서는 0.827–0.991), 거짓 통과는 모든 곳에서 0.000.
  기록 전용 진단: 쌍당 PASS는 F 8–64에서 평평(천장 0.68–0.82)이고, g 1.0 실패는 보상 수준의 순진 쌍 효과 약 42%, 처벌 단계 기계 대조 약 40%, 처벌 바닥 약 12%, 잡음 약 2%다; 균형 쌍 θ(F1, 파일럿 3쌍뿐이라 낙관적일 수 있음)는 0.974–0.993, 순진 반응 하한 거름(F2)은 수용률 0.95 → 0.13에서도 ≤ 0.889, 최소 변경 순위 1은 F1 하나(k 4–8, F 8).
  주 세트(턴 306–1985)는 쓰지 않아 남아 있다. 사용자 사전 승인(2026-10-06)에 따라 다음은 순진 측정 기반 관문 쌍 거름(부록 Y) 선언이다.
- **안 된 것(운영)**: 실패한 튜닝 실행은 스크립트를 `--out results/m0/<태그>.json`으로 다시 돌려 보관한다.
  `results/`는 `results/summary/`를 빼고 git에서 제외되며, 채택한 실행만 `results/summary/m0.json`에 요약된다.

## In English

FlyMon runs a leaky integrate-and-fire simulation of the whole MaleCNS v1.0 fruit-fly connectome as the attacking-move
chooser in a restricted task on Pokémon Showdown's Gen 1 OU rules (16 species, a constrained move pool), next to a
heuristic coach. The ledger above separates what was measured, what we chose, and what did not work; this is its summary.

- **M0 (engine, reproducing flybrain's measurements): partial pass.** Kenyon-cell sparsity, the MBON baseline and
  channel-specific odour depression pass; the pre-registered composite-index flip does not (graded n_flip 0/8).
- **M0c (KC→KC fast excitation removed): pass** on sparsity, baseline, runaway, equivalence and throughput; the
  conditioning criterion still fails, as predicted.
- **M1 (battle environment, brain-free pilot): pass** — MAX − RND win rate 0.319 ≥ 0.15.
- **M2 (encoder, readout, learning unit test): no-go — the test could not be built (2026-09-22).** Spec 5's learning
  unit test (naive d′ ≈ 0, d′ ≥ 1 after 20 rewards, a drop after 20 punishments) never ran. On every encoder (E0–E3) and
  engine (C0, C1, C3) we tried, too few candidate pairs are testable even under an idealised, maximally specific weight
  edit. The encoder comparison's best score was 0.222 against a bar of 0.5 (E0: 4 of 18 within-turn and 9 of 21
  opponent-type pairs; the score is the smaller rate). The engine comparison's best was 7 of 21 opponent-type pairs
  (C3), against a bar of half the pairs plus at least two testable within-turn pairs with a naive d′ near 0. Confining
  the edit to Kenyon cells active only for the taught odour left C3 with 2–3 of 21, because the punishment readout
  (MBON13) is driven mostly by Kenyon cells the two odours share. This is **not** a "no learning" result.
  - Learning tests are limited to the designed odour pair (a mechanism control). The primary claim (type-conditional
    move preference, spec 4.3 criterion 1) is not supported in its current form and is reported as untested. M3 (agent
    loop), M4 (experiment) and the secondary claim (win-rate contribution) are on hold, not discarded. Whether to design
    short-term depression (ORN→PN) is the first question of the next claim declaration.
  - Runaway and ratio conditions (spec D.6): (a) met — 7 of 2,624 presentations had a Kenyon cell above 150 Hz in some 200 ms window (maximum 31 spikes); (b) met — the within-turn candidate KC spike ratio exceeds
    2 on 3 of 16 turns; (c) the learning unit test itself: not measured.
  - We do not attribute the no-go to one cause: only two readout types, an idealised oracle whose ceiling bounds only the
    current Kenyon-cell code (encoder and coding changes such as local APL or rate normalisation were not tested), 21
    even-turn pairs, an arbitrary channel-to-glomerulus assignment, and a baseline confound between C0 and C1/C3.
  - Derived by `scripts/write_m2_nogo_summary.py` into `results/summary/m2_nogo.json`; the M2 go condition stays in the
    code as strict expected failures (`tests/test_m2_go.py`). Details: spec appendix I (in Korean).
- **Short-term depression (ORN→PN) as the lever: closed (2026-09-25, spec J.13).** The next claim declaration
  (appendix J) measured STD directly. A 9-point f × τ scan with the total gain restored compressed the Kenyon-cell
  code's spread by up to 26.9%. The two top-ranked settings had no operating point under C3's re-convergence rule. The
  third (f 0.95, τ 100 ms), re-converged, had 4 of 21 testable opponent-type pairs, fewer than C3's 7. That falls in the
  pre-declared band B_Tb, so STD is closed as the lever for this bottleneck. D.6: (a) not met (max 110 Hz); (b) met on
  5 of 16 turns. The reduced claim's learning test could not be calibrated (J.12.8), so that claim is untested. Next: a
  new claim declaration.
- **Fast KC→KC inhibition as the lever: closed (2026-09-27, spec K.9).** The next declaration (appendix K) went below
  the PN layer: KC→KC edges restored as fast inhibition (`kc_kc_scale` g in {−0.05, −0.1, −0.2, −0.4, −0.8}), every
  setting re-converged by C3's rule. The X-only drive onto MBON13, measured on 20 odd-turn opponent-type pairs, rose by
  at most 3.5% over C3 (g = −0.4 had no operating point). The selected g = −0.1 (+3.3%) had 4 of 21 testable even-turn
  opponent-type pairs (C3: 7). That falls in the pre-declared band B_Tb, so fast KC→KC inhibition is closed as the lever
  for this bottleneck; per-KC homeostatic re-convergence undid most of the inhibition's effect. D.6: (a) not met (max
  105 Hz); (b) met on 6 of 16 turns. Next: a new claim declaration.
- **Naive-readout screening: closed (2026-09-27, spec L.12).** The next declaration (appendix L) left the engine alone
  and narrowed the claim: on the C3 engine, test only opponent-type pairs picked in advance from the untrained brain's
  readout. A "MBON13 floor guard + one-feature threshold" screen was fitted on 21 even-turn + 20 odd-turn pairs (the odd
  ones newly labelled with the C3 oracle; 2 of 20 testable). The final rule, G and S > 0.186, had a leave-one-turn-out
  precision of 7/15 = 0.467, below the 0.6 gate (`SCREEN_IMPRECISE`), so the screening claim is closed as pre-declared
  and the new pair set was never run. D.6 on C3: (a) not met (max 100 Hz; the earlier "met" record was the C0 engine's),
  (b) met on 7 of 16 turns. The M3 infrastructure (agent loop, checkpoints, logs, smoke) was finished in parallel with
  no judgement attached. Next: a new claim declaration.
- **Readout expansion: closed (2026-09-28, spec M.11).** Aimed at the alternative explanation "the readout is only
  MBON13 and MBON05": the taught DAN compartments (PPL1 punish x PAM reward) were re-chosen and read as their core MBON
  cell populations. Among candidates whose core KC input had a larger X-only share than the incumbents (5 reward, 2
  punish) and that passed the reactivity guard, a per-arm scan and a joint check of the top combinations gave at best
  PPL106·PAM12 with 7 of 21 testable even-turn pairs (same as C3, F_a 1), below the M2 bar (`STOP_NO_GAIN`); the
  judgement set was never used. This ends this round's lever search.
- **Real odours: closed at N1 (2026-09-30, spec N.9).** Instead of Pokémon pairs, measured Hallem 2006 responses (DoOR
  mapping) drove the C3 engine with IA:EB 4:1 vs 1:4 (similar pair) and 4:1 vs δ-DL (dissimilar pair), aiming at the
  APL-dependent fine discrimination of Lin 2014. An operating point was found (g 0.25, c_δ 8, KC activity 5.19%) and the
  similarity order held (Δr 0.336, 95% CI [0.296, 0.356]), but in the punish-only oracle the similar pair reached
  p0 = −1.955 against the bar of −2 (the dissimilar pair passed at −3.734; `STOP_UNTESTABLE`). As pre-declared, the
  claim was closed at N1 and the learning judgement (N2) was never measured. This is not an upper bound, only "not
  testable under the pre-specified edit protocol".
- **Claim re-scope: closed, neither layer tested (2026-09-30, re-scope spec 11.3, 11.4).** The primary claim was
  re-cut into two layers (`docs/superpowers/specs/2026-09-28-rescoped-claim-design.md`): a reward-side mechanism
  contrast on C3 design pairs, and battle win rate (FLY vs RND, vs plasticity-off flies, vs yoked reinforcement). Layer 1
  was not tested: the positive control seed0 did not qualify (its Y odour was in the silent state in 4 of 16 seeds,
  `STOP_CONTROL_INVALID`). Layer 2 stopped at the power pilot (6 flies x 20 evaluation battles per arm): with the pilot's
  variance components the joint (2b) power never reached 0.8 anywhere on the pre-fixed grid — at most 0.140 within the
  60 h budget (F 32, E 40) and 0.186 on the whole grid (F 32, E 300, 234.8 h) (`STOP_POWER`). The judgement run was
  never started and (2a)/(2b) were not measured; no rule or number was changed after seeing the result. This is not
  "no win-rate effect", only "not testable at this scale".
- **Characterisation of C3's two states and presentation-evoked depression (2026-10-01, spec O.8, O.9; not a learning
  claim).** Both were measured on the real-odour rig with pre-declared judgements. O1 (7,680 presentations): the silent
  state appears only at low drive (g 0.125-0.25, 6 mixed cells) and the readout is not bimodal (`GRADED`, mid-band rule
  met in 0 of 6 cells); blocking APL's non-KC output cut silence by 61%, short of the pre-set 75% (`NEITHER_PATH`); more
  drive means less silence (`DECREASING` for 4 stimuli). O2 (256 arms): presentation alone lowered MBON13 by about 5
  spikes (14%), inside the pre-set band (`NO_DEPRESSION`), and punishment separated clearly on top of it (`SEPARABLE`,
  dV -2.1 / -2.4). The "network bistability" and "punishment inseparable from depression" seen with Pokémon odours did
  not appear with real odours under this protocol.
- **First learning judgement: learns, confirmatory (2026-10-01, spec P.7).** Punishment-side absolute conditioning on
  one real-odour dissimilar pair (IA:EB 4:1 vs δ-DL) was judged in both directions (each odour punished in turn) on 32
  new seeds. The extra drop in the readout caused by punishment over the same unpunished presentations was 1.79 [1.60,
  1.97] with 4:1 punished and 2.16 [1.92, 2.38] with δ-DL punished, both above the pre-set c1 = 0.61, and the change
  concentrated on the punished odour rather than its partner: `LEARNS_CONFIRMATORY`. It was declared after O2, so it is
  a replication, not a discovery; the scope is C3, this pair and the punishment side only (not Pokémon battles, the
  similar pair, or reward). Next: join the Pokémon encoder-redesign track.
- **Encoder redesign: stopped at the even-turn selection (2026-10-02, encoder spec 2026-10-01-encoder-redesign-design.md
  section 13).** Each (move type × opponent type) conjunction got its own glomerulus pattern, and conjunctions sharing a
  move type or an opponent type share no glomerulus (E-grid, 4 configurations); the rules and a fresh judgement set (L
  generator turns 64-103, deduplicated by conjunction key) were fixed before any run. Strength calibration on KC activity
  alone left one eligible configuration, k2-norm at s 1.0 (the others narrowly missed a tail or band condition under the
  ORN rate cap), and its even-turn oracle gave 7 of 21 testable (b) pairs and F_a 0, below the M2 bar (11 and 2), so the
  run stopped as declared (`STOP_EVEN_LOW`); the judgement set was not measured. Recorded, not judged: the X/Y KC Jaccard
  fell to about 0.04 and punishment passes rose ((b) 11 to 15), but reward passes (MBON05) did not, so testable pairs equal
  E0's: with this encoder the bottleneck moved from KC overlap to the reward readout (the cause was not measured).
  The track is closed on this result (user decision); next is the user's call, and a reward-side declaration will be
  considered after track P's result.
- **Reward-readout bottleneck diagnosis: recorded (2026-10-02, appendix Q.7; a characterisation, not a verdict).** The 21
  even (b) pairs of E-grid k2-norm were fixed as 12 failing (F) / 9 passing (S), and five candidates were read on the
  MBON05-only r_P with rules fixed before any run. Operating point **matches** (global gain x1.25 raises F's median r_P by
  3.1, not concentrated on low naive P_X; mv_scale also changes input drive, so this is not fully separate from the floor
  candidate); edit reach and seed variation **do not match**; floor is **undecided** (naive P_X is lower in F, AUC 0.87,
  but lowering s hurt high-P_X pairs more, rho -0.60); APL inhibition is **undecided** (zeroing the 2 APL->MBON05 edges
  raised F's r_P by +18 but also naive P_X, so the edit ratio fell). The reward edit barely moves P_X in F (median -3 vs
  -26 in S). Choosing the lever (appendix R) is the user's decision.
- **Lever APL->MBON05 removal: judged B_처벌가드 (punishment guard) (2026-10-03, appendix R.10).** Q.7's candidate 2 was
  declared the lever and judged once, on M2's bar, on the unused judgement set (L generator turns 64-103, (b) 21, (a) 32).
  The no-edit reproduction gate and gates 1 (KC 0.0457, in band), 2 (P replication LEARNS_CONFIRMATORY) and 3 (even
  testable_b 16/21, no lever 7/21) passed. On the judgement set the lever reached **12/21** against **6/21** without it
  (bar 11 and margin 2 met), F_a 2/32, but on the (a) side 3 punishment passes swapped (17 vs 17, pass->fail 3), which
  trips the punishment guard. The lever opened nearly every reward pass ((b) 21/21). The lever is neither closed nor
  selected; next is the user's decision.
- **Same lever on the last set: judged B_Fa (2026-10-03, appendix S.10).** After R's result the punishment guard was made
  directional (net drop in punishment passes >= 3 on either axis) and gate 2 gained a same-seed L/C punishment-learning
  ratio (>= 0.5); the lever was then judged once more on the last unused set (L generator turns 104-177, (b) 21, (a) 43).
  R's reproduction gate and gates 1 and 3 were reused on the same measurement code key; at gate 2 the lever cut
  punishment learning by about a third on the same seeds (ratios 0.64 / 0.68) but passed. On the judgement set the lever
  reached **13/21** against **6/21** without it and passed the punishment guard ((b) net drop 2, (a) -1), but F_a was
  **1/43** (naive_a 3): **B_Fa**, the next bottleneck being naively balanced (a) pairs. No M2 judgement set remains for
  this lever; next is the user's decision.
- **Per-engine reference-set z: gate stop STOP_Z_DEGENERATE, no judgement (2026-10-04, appendix T.10).** A rule fixing
  z per engine variant (each variant's own H.3 reference-set z) was declared before any result, to judge the same lever on
  a new set with opponents widened to Gen-1 base forms. The unedited engine reproduced block h4's z bit for bit, but under
  the lever MBON13 failed the reference-set readout guard (median delta 6.0 -> **1.0**, zero share 0.146 -> **0.302**;
  MBON05 mean 26.25 -> 77.5; SD ratios A 0.62, P 1.58). Recorded, not judged: the lever turns MBON05 up about threefold
  and switches off MBON13's reference-set response, which R's and S's h4-z readings may have hidden. T's set was not
  measured and stays unused. Next is a 'partial removal' lever declaration (appendix U, the user's decision).
- **Partial APL->MBON05 removal f: gate stop STOP_NO_QUALIFIED_F, no judgement (2026-10-04, appendix U.10).** A lever
  scaling the two APL->MBON05 edges by f was to be judged on T's set with per-engine z. The endpoint gate passed (f=1 =
  T's unedited rows, f=0 = T's lever rows, bit for bit); in the 9-point guard scan f 0.6-0.9 passed the MBON13 guard at
  the boundary (median delta 5.0; f <= 0.5 gave 2-4). All checked f {0.6, 0.7, 0.8} failed the KC band: 4-5 of T's set
  odours had median KC activity below 0.03 (minimum 0.019) - and **the unedited engine has the same 4 odours outside
  the band**, so the stop comes from T's set odour inputs, not from f. Recorded, not judged: the fixed mechanism readings
  disagree (output block 'chain not supported', D_block 6.5 vs D_none 5.0; chain-entry cut 'chain supported', MBON13
  delta 16.0). The judgement set's oracle was not measured and stays unused. Next is the user's decision.
- **Combined lever: judgement SELECTED (2026-10-05, appendix V.10).** A model with the two APL->MBON05 edges and the
  eleven MBON05->MBON09/MBON11/MBON01 edges removed (13 connectome edges) was judged once, with per-engine z, on a new
  widened-pool set filtered by KC input on both the unedited and the combined engine (generator turns 0-305). Lever
  **16/21** vs no lever **6/21**, F_a **2/43** (at the bar), punishment guard passed ((b) net drop -2, (a) 0), gate-2
  ratios h4 z 1.019 / 0.942 and z_V 0.773 / 0.704, even 15/21 -> **SELECTED**. The gain is mostly reward passes ((b)
  8 -> 21); punishment passes did not fall. The consequence stops at M2 testability on the widened pool with per-engine
  z: the lever was chosen after seeing U's mechanism records (fifth attempt), the result is conditional on these 64
  pairs (cluster-model null 0.374), and it does not by itself justify a POOL F v4 learning test. Next is the user's
  decision.
- **F v4 learning test: gate stop STOP_OC_UNREACHABLE, no judgement (2026-10-05, appendix W.10).** The M2 learning
  unit was to be judged with the real learning rule (F.2 sequential R/N/RN) on the combined-lever engine (V's 13 edges,
  z_V). Reuse, the endpoint gate (8 V gate-2 punish rows and 6 naive probe rows bit for bit, reward path positive) and
  the pilot (16 pairs x 8 flies; reward share 1.0, punishment 0.875, reversed 0.0, X+Y floor 0.033) passed (exploratory
  verdict 11 PASS / 5 FAIL). In the operating characteristic no design (q 0.5-0.75, K 8/16, F 8-32, k 4-8) met G.6:
  point power at true d' 1.5 is at most **0.449** (k 4) and flat in F (about 0.82^k: each pair fails ~18%, read as
  pair-level heterogeneity in the pilot model), false pass at d' 0.5 is 0.0, and 29/200 bootstrap recalibrations
  failed so the limits are 0.000 / 1.000. Drift-only reward-level d' is -2.35. The main set (turns 306-1985) was not
  used and stays available. Next is the user's decision.
- **F v4 learning test ("at least half"): gate stop STOP_OC_UNREACHABLE, no judgement (2026-10-06, appendix
  X.10).** X kept W's lever, protocol and gate pairs, changed only the set verdict to "PASS share of judgeable gate
  pairs >= p_set (at least 3 pairs)" plus per-status calibration-failure handling, and after the red team put a
  point-theta precheck gate first. Stage 0 passed (W calibration diagnosis reproduced: all 29 W failures are
  punishment-handle b non-convergence and 160 steps recover none; under X's x4 bracket b hits the floor on 79/200
  power-side and 44/200 false-side draws; synthetic checks and P2-6 pass). In the precheck no design of 750 reached
  point power >= 0.80 at every k 4-8: the best, p_set 0.5 q 0.5 K 16 F 8, gives **0.608 / 0.730 / 0.795 / 0.735 /
  0.790** (worst cluster g 1.0; 0.827-0.991 at g 0), with false pass 0.000 everywhere. Record-only diagnostics: per-pair
  PASS is flat in F 8-64 (ceiling 0.68-0.82); at g 1.0 failures are ~42% reward-level naive pair effect, ~40%
  punishment-stage mechanism control, ~12% punishment floor, ~2% noise; a balanced-pair theta (F1, only 3 pilot pairs,
  likely optimistic) gives 0.974-0.993, naive-response floors (F2) stay <= 0.889 even at acceptance 0.95 -> 0.13, and
  the minimum-change rank 1 is F1 alone (k 4-8, F 8). The main set (turns 306-1985) was not used and stays available.
  Per the user's standing approval (2026-10-06), next is a naive-measurement pair-filter declaration (appendix Y).

## 실행

    uv sync
    uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
    uv run python scripts/reproduce_flybrain_measurements.py sparsity --kc-kc-scale 1.0                # M0 = 옛 엔진; 기본값(0.0)으로는 results/m0/ 쓰기를 거부한다
    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8 --kc-kc-scale 1.0
    uv run python scripts/write_m0_summary.py
    uv run pytest

`results/summary/m0.json`은 위 두 게이트 실행이 끝난 뒤 `scripts/write_m0_summary.py`가 만든다
(입력 파일이 없거나 현재 `Params()` 기본값에 해당하는 격자 행이 희소성 게이트를 통과하지 못하면 이유를
출력하고 종료 코드 2로 끝나며 아무것도 쓰지 않는다. 조건화 기준 실패는 쓰기를 막지 않는다 —
PASS / PARTIAL / FAIL 한 줄을 찍고 `gate` 블록에 그대로 기록한다).

- MBON 기저 게이트 구간은 스펙의 **3–4 Hz**(절사 평균 `mbon_hz_rest_trimmed`, `--rest-seeds` × `--rest-ms`)이며,
  이 구간에 들어가도록 `sparsity`가 `--mbon-hold`
  (`mbon_hold_frac`)를 `--kc-thresh`/`--apl-scale`과 함께 세 번째 격자 축으로 훑는다.
- `sparsity`는 도파민 구획 표를 `results/summary/compartments.json`으로 내보낸다(PPL105·PAM08의 core 행 포함).
- `conditioning`의 워커는 각자 커넥톰 전체(약 1–2 GB)를 올린다. `--jobs`는 단일 워커 실행으로 메모리를 잰 뒤 정한다
  (기본값 `min(4, CPU 수)`).

### M1 배틀 환경

    bash scripts/install_showdown.sh                      # pokemon-showdown 0.11.11 고정 설치(npm ci)
    uv run python -m flymon.battle.validate_pool --write-docs
    uv run python scripts/pilot_no_brain.py --arm RND --flies 16 --battles 100
    uv run python scripts/pilot_no_brain.py --arm MAX --flies 16 --battles 100
    uv run python scripts/pilot_no_brain.py --arm WEAK-RND --flies 16 --battles 100
    uv run python scripts/pilot_no_brain.py --arm WEAK-MAX --flies 16 --battles 100
    uv run python scripts/pilot_no_brain.py --gate         # MAX − RND ≥ 0.15 → results/summary/m1_pilot.json

- 서버는 항상 127.0.0.1에만 바인딩된다(`flymon/battle/server.py`). 포트 충돌은 `lsof -nP -iTCP:<port>`로 확인한다.
- 팔은 한 번에 하나씩 돌린다(상대 계정 이름이 팔 사이에 겹친다). 팔당 1,600배틀은 수 분이다.
- `tests/battle/test_server.py`·`test_fly_coach_player.py`는 서버가 설치돼 있지 않으면 skip된다.

### M0b 프로세스 풀 스웜

    uv run python scripts/bench_pool.py throughput            # 워커 4/8/16, 결정·강화 단계 ms/step 3회 → results/m0b/throughput.json
    uv run python scripts/bench_pool.py reproduce             # M0 조건화 5팔×8시드를 풀로 재실행(비트 동일 검사), 희소성·기저·폭주 집합 → results/m0b/reproduce.json (16워커, 약 10분)
    uv run python scripts/bench_pool.py summary               # 예산표(스펙 C.6)·정확 일치·게이트 → results/summary/m0b.json

- 스웜은 `flymon/brain/fly_pool.py`의 워커 프로세스 풀이다. 워커마다 CPU 엔진 하나, 마리별로는 KC→MBON 가중치·켬/끔·배선 변형만 남고 부모가 보관한다.
- 결정은 후보를 같은 시드로 순차 제시해 잡음을 짝짓는다(`flymon/brain/presentation.py`). 강화는 M0 `train_block`의 한 프레젠테이션과 같다.
- MPS 배치 엔진은 스파이크와 레드팀 뒤 채택하지 않았다(스펙 부록 C.6).

### M0c KC→KC 제거와 재게이트

    uv run python scripts/reproduce_flybrain_measurements.py sparsity --kc-thresh 1.5 --apl-scale 0.1 --rest-seeds 8 --out results/m0c/sparsity.json
    uv run python scripts/bench_pool.py reproduce --kc-kc-scale 1.0 --out results/m0c/reproduce_old.json                       # 옛 엔진 = results/m0 비트 동일(등가성)
    uv run python scripts/bench_pool.py reproduce --seed-start 8 --sparsity-seeds 3 --rest-seeds 8 --odor-runaway-seeds 64 --arm-equal \
        --m0-conditioning "" --m0-sparsity results/m0c/sparsity.json --conditioning-out results/m0c/conditioning.json --out results/m0c/reproduce.json
    uv run python scripts/bench_pool.py reproduce --seed-start 0 --rest-seeds 0 --sparsity-seeds 0 --m0-conditioning "" --m0-sparsity "" \
        --conditioning-out results/m0c/conditioning_seeds0-7.json --out results/m0c/reproduce_seeds0-7.json                    # 보고용(판정 아님)
    uv run python scripts/bench_pool.py throughput --out results/m0c/throughput.json
    uv run python scripts/write_m0c_summary.py                                                                                  # results/summary/m0c.json

- 새 엔진(`Params()` 기본값, `kc_kc_scale` 0.0)의 M0 게이트를 스펙 부록 D.4의 사전 등록대로 다시 잰다. 옛 결과 파일은 덮어쓰지 않는다.
- `--kc-kc-scale 1.0`은 옛 엔진이다: `reproduce_old.json`의 `conditioning_match`·`sparsity_match`가 `results/m0`와 비트 동일해야 한다.
- `--arm-equal`은 (seed 8, both)·(seed 15, reversed)를 인프로세스로 다시 돌려 풀 행과 비트 동일한지 본다. 폭주 검사 임계는 휴지 100 Hz, 냄새 창 150 Hz(D.4).

### 학습 중 보기

    uv sync --extra viz
    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 2 --jobs 2 --viz --out results/m0/viz.json

- `--viz`는 Rerun 뷰어를 띄우고 워커마다 `seed<N>/` 아래로 보낸다. 시간축 `sim`은 시뮬레이션 시간이고
  `engine.reset()`을 넘어 계속 흐른다.
  - `rate_hz/{alpn,kc,mbon,apl,dan/<타입>}`: `--viz-every` 스텝(기본 100) 창의 집단 평균 발화율
  - `cell_hz/mbon`: MBON별 발화율 막대(포화 세포가 바로 보인다)
  - `weights_frac/<타입>`, `dan_pulse/<타입>`: 제시마다 core 구획의 KC→MBON 가중치 비율과 도파민 펄스
  - `probe/{pre,post}/{plus,minus}/{A,P}`, `D_pre`·`D_post`·`dD/<팔>`, `events`: 탐침 발화 수와 팔 결과
- 관찰은 결과를 바꾸지 않는다. 탭은 가소성 훅을 먼저 부른 뒤 스파이크만 센다(`tests/brain/test_conditioning_events.py`).
- `--out`을 주지 않으면 게이트 결과 `results/m0/conditioning.json`을 덮어쓴다. 그냥 `uv sync`를 하면 `viz` extra가 빠진다.

데이터 출처와 감사: `docs/acknowledgments.md`.
