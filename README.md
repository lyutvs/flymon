# FlyMon

MaleCNS v1.0 초파리 뇌 커넥톰의 LIF 시뮬레이션. M0 단계: 엔진과 flybrain 측정값 재현.

## 측정된 것 / 우리가 정한 것 / 안 된 것

- **측정된 것(데이터)**: 뉴런 연결, 시냅스 수, 신경전달물질 예측(MaleCNS v1.0).
- **우리가 정한 것**: LIF 상수(Shiu et al. 2024), lLN1/lLN2 억제 재지정, APL 출력 스케일, MBON 기저 구동,
  KC 역치 정규화, 반구 보정, 가소성 규칙과 상수. 값은 `results/summary/m0.json`의 `params_frozen`.
- **재현 결과**: `results/summary/m0.json` — KC 희소성/겹침, MBON 기저, 조건화 반전(8시드).
- **안 된 것**: 이 단계에서 시도했다가 실패한 설정은 `results/m0/*_*.json`에 남긴다.

## 실행

    uv sync
    uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
    uv run python scripts/reproduce_flybrain_measurements.py sparsity
    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8 --jobs 8
    uv run python scripts/write_m0_summary.py
    uv run pytest

`results/summary/m0.json`은 위 두 게이트 실행이 끝난 뒤 `scripts/write_m0_summary.py`가 만든다
(입력 파일이 없으면 빠진 경로를 출력하고 종료 코드 2로 끝난다).

데이터 출처와 감사: `docs/acknowledgments.md`.
