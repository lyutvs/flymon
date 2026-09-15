# FlyMon

MaleCNS v1.0 초파리 뇌 커넥톰의 LIF 시뮬레이션이 포켓몬 1세대 OU **규칙** 위의 제한 과제(16종·제약 기술 풀)에서 공격기 선택을 배우게 한다.
M0: 엔진과 flybrain 측정값 재현. M1: Showdown 배틀 환경과 뇌 없는 파일럿.

**M0 결과: 부분 통과 (희소성·기저 발화·채널별 냄새 특이 억제 통과, 합성 지수 반전 미달)**

**M1 결과: 통과 (풀 게이트 MAX − RND 승률 0.319 ≥ 0.15, 95% CI [0.291, 0.344])**

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
- **실제 M0b 측정**(CPU 프로세스 풀, 실제 커넥톰): 16워커 엔드투엔드 예산 42.2시간(학습 결정 52,000 + 평가 결정 81,600; 학습만 19.3시간) ≤ 60시간 게이트 **통과**. 풀로 재실행한 M0 조건화 40건(5팔 × 8시드)이 `results/m0/conditioning.json`과 비트 동일, 희소성·기저도 차이 0.0, 풀의 결정이 인프로세스 엔진과 동일. 워커 RSS 0.60 GB(최대 0.69). 휴지 3초에서 100 Hz 초과 뉴런 156개 중 KC 17개, 스파이크의 29% — 스펙 A.5의 STD 재검토 조건 충족. 요약 `results/summary/m0b.json`, 자세한 것은 스펙 부록 C.7.
- **우리가 정한 것(M0b)**: MPS 배치 엔진 대신 기존 CPU 엔진의 프로세스 풀(레드팀 실측, 스펙 C.6). 마리별로 남는 상태는 KC→MBON 가중치·켬/끔·배선 변형뿐이고 부모가 보관한다. 결정은 후보를 같은 시드로 순차 제시(짝지은 잡음), 강화는 M0 `train_block`의 한 프레젠테이션. 등가성 게이트는 통계 일치가 아니라 비트 동일 재현. STD는 여전히 보류(A.5).
- **안 된 것(운영)**: 실패한 튜닝 실행은 스크립트를 `--out results/m0/<태그>.json`으로 다시 돌려 보관한다.
  `results/`는 `results/summary/`를 빼고 git에서 제외되며, 채택한 실행만 `results/summary/m0.json`에 요약된다.

## 실행

    uv sync
    uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
    uv run python scripts/reproduce_flybrain_measurements.py sparsity
    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8
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
