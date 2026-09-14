# 데이터

MaleCNS v1.0 (HHMI Janelia FlyEM, Cambridge Connectomics, Google Research), CC-BY 4.0.
https://male-cns.janelia.org/download/ (neuPrint 계정 필요)

`data/raw/`에 세 파일을 받은 뒤:

    uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz

`data/malecns.manifest.json`에 입력 파일 sha256과 뉴런·엣지 수가 기록된다. 실제 데이터로
빌드하면 이 파일의 해시를 아래 표에 옮겨 적는다.

| 파일 | sha256 | bytes |
|---|---|---|
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | (빌드 후 기입) | |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | (빌드 후 기입) | |
| body-neurotransmitters-male-cns-v1.0.feather | (빌드 후 기입) | |
