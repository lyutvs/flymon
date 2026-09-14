# FlyMon

MaleCNS v1.0 초파리 뇌 커넥톰의 LIF 시뮬레이션. M0 단계: 엔진과 flybrain 측정값 재현.

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
- **우리가 정한 것**: LIF 상수(Shiu et al. 2024), lLN1/lLN2 억제 재지정, APL 출력 스케일, MBON 기저 구동,
  KC 역치 정규화, 반구 보정, 가소성 규칙과 상수, 불응기 회계(스파이크 후 `refrac_steps()` 스텝만큼 더 지나야 다시
  발화 가능 — dt = 1 ms에서 실제 3 ms, 2.2 ms보다 한 스텝 길다), 막전위 하한 `v >= -v_thresh`.
  값은 `results/summary/m0.json`의 `params_frozen`.
- **우리가 정한 것(게이트 통계)**: MBON 기저 게이트 통계는 3시드 × 3초의 **절사 평균**(100 Hz 이하 세포만)이다 —
  자기지속적인 FR1 clique가 MBON 몇 개를 포화시켜 raw 평균이 시드마다 흔들리기 때문이다. `mbon_hold_frac`은
  0.85로 두고(0이면 뇌가 침묵한다), `kc_thresh`는 1.5로 고정한다(1.0은 KC 겹침이 우연 수준을 넘는다).
  이런 clique를 안정화할 기전인 단기 시냅스 억압(short-term depression)은 M0에 없으며 후속 과제로 남긴다.
- **우리가 정한 것(도파민 채널)**: 실제 데이터 조건화 게이트는 처벌 채널로 **PPL101**(γ1pedc, core MBON11,
  Aso et al. 2014의 표준 혐오 기억 구획), 보상 채널로 **PAM08**(core MBON05/21)을 쓴다(두 core는 disjoint).
  flybrain은 PPL105를 썼지만 우리 엔진에서 PPL105의 core 판독은 퇴화한다 — MBON18은 어떤 hold에서도 침묵하고
  (flybrain 보고와 동일), MBON23은 0–3 스파이크, MBON13은 시험한 설계 냄새쌍 16개 모두에서 두 냄새 중
  **하나에만** 반응해(600 ms에 0 대 27–41 스파이크) 판별 지수가 학습 전부터 ±1에 붙는다. 실제로 PPL105/PAM08
  8시드 실행은 noplast = 0.0, `reversed` dD = +2.28(8/8 양수)이지만 `both` dD ≈ 0, n_flip 2/8이었다.
  MBON11은 모든 쌍에서 두 냄새에 모두 반응한다(설계 시드 0에서 41/107, 26/107 스파이크/600 ms).
  CLI 기본값은 PPL105/PAM08 그대로라 합성 테스트와 flybrain 비교는 바뀌지 않으며
  (`conditioning --punish-type PPL101`로 바꾼다), 사용한 채널은 M0 요약의 `conditioning` 블록에 기록된다.
- **재현 결과**: `results/summary/m0.json` — KC 희소성/겹침, MBON 기저, 조건화 반전(8시드).
- **안 된 것**: 실패한 튜닝 실행은 스크립트를 `--out results/m0/<태그>.json`으로 다시 돌려 보관한다.
  `results/`는 `results/summary/`를 빼고 git에서 제외되며, 채택한 실행만 `results/summary/m0.json`에 요약된다.

## 실행

    uv sync
    uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
    uv run python scripts/reproduce_flybrain_measurements.py sparsity
    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8
    uv run python scripts/write_m0_summary.py
    uv run pytest

`results/summary/m0.json`은 위 두 게이트 실행이 끝난 뒤 `scripts/write_m0_summary.py`가 만든다
(입력 파일이 없거나 현재 `Params()` 기본값에 해당하는 격자 행이 게이트를 통과하지 못하면 이유를 출력하고
종료 코드 2로 끝나며 아무것도 쓰지 않는다).

- MBON 기저 게이트 구간은 스펙의 **3–4 Hz**(절사 평균 `mbon_hz_rest_trimmed`, `--rest-seeds` × `--rest-ms`)이며,
  이 구간에 들어가도록 `sparsity`가 `--mbon-hold`
  (`mbon_hold_frac`)를 `--kc-thresh`/`--apl-scale`과 함께 세 번째 격자 축으로 훑는다.
- `sparsity`는 도파민 구획 표를 `results/summary/compartments.json`으로 내보낸다(PPL105·PAM08의 core 행 포함).
- `conditioning`의 워커는 각자 커넥톰 전체(약 1–2 GB)를 올린다. `--jobs`는 단일 워커 실행으로 메모리를 잰 뒤 정한다
  (기본값 `min(4, CPU 수)`).

데이터 출처와 감사: `docs/acknowledgments.md`.
