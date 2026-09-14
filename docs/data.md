# 데이터

MaleCNS v1.0 (HHMI Janelia FlyEM, Cambridge Connectomics, Google Research), CC-BY 4.0.
https://male-cns.janelia.org/download/ (neuPrint 계정 필요)

`data/raw/`에 세 파일을 받은 뒤:

    uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz

`data/malecns.manifest.json`에 입력 파일 sha256과 뉴런·엣지 수가 기록된다. 실제 데이터로
빌드하면 이 파일의 해시를 아래 표에 옮겨 적는다(아래 표는 실제 빌드 2026-09-14T17:09:36 기준,
뉴런 162,517개 · 엣지 25,120,209개).

| 파일 | sha256 | bytes |
|---|---|---|
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` | 1,051,241,946 |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` | 14,483,314 |
| body-neurotransmitters-male-cns-v1.0.feather | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` | 43,282,834 |
