# FlyMon 핸드오프 — M0d H.2 구현 완료, H.3 개정 설계 대기

작성 2026-09-17 (guillemot 워크트리). **한 문서로 이어받는다.** 이전 핸드오프 `2026-09-17-m2-calibration-handoff.md`의 규율(사전 선언, 짝·홀 턴, 코드 먼저)은 부록 G.13·H로 갱신됐다.
**이 문서와 이후 문서에 적힌 SHA는 히스토리 재작성 뒤의 값이다.** 그 전 문서·요약 JSON의 SHA는 `docs/commit-map.tsv`로 찾는다.

## 0. 한 줄 상태

M0d 엔진 모드(H.2)는 구현·리뷰 완료. H.3(조합별 파라미터 고정)을 돌리기 전에 **H.3 개정 선언**이 필요하다 — C2는 성립 불가로 확인됐고(사용자 ok),
C1의 등급 APL이 6–8% 작동점에서 포화한다는 발견에 대한 **대응 설계(§4)가 제안만 된 상태이며 승인되지 않았다.** 다음 세션의 첫 일은 §4를 사용자와 확정하는 것이다.

## 1. 먼저 읽을 것 (순서대로)

1. 이 문서.
2. 스펙 `docs/superpowers/specs/2026-09-14-flymon-design.md`
   - **부록 G.13**(G.12 재채점 정정·홀수 턴 사용 목록·E.5 정정), **G.14**(엔진 전제 프로브 선언)와 **G.14.8**(스모크에서 판독 바닥 교락으로 중단).
   - **부록 H 전체**: H.0 조사(문헌 + 편차 원인 진단), H.1 조합, **H.2 엔진 + 끝의 "구현 결과(2026-09-17)" 문단**, **H.3 고정 규칙**, H.4 선택, H.5 확인, H.6 동결·D.6 조치표.
3. 진단 보고서 `results/m0d/diag/c2_std_feasibility.md`(git 제외, 로컬) — §3의 근거 수치.
4. 문헌 근거 `docs/superpowers/specs/m0d-evidence/`(evidence.md, lhpv3c1.md, code_sites.md).

## 2. 저장소·운영 상태

- 경로 `/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot`, 브랜치 `open-fly-brain-connectome` → **`origin/main`을 추적**(`push.default=upstream`).
- **GitHub**: `https://github.com/lyutvs/flymon` (public, 기본 브랜치 main). **push 전 `gh auth switch --hostname github.com --user lyutvs`, 끝나면 `--user asher_wrtn`으로 되돌린다**(gh 자격증명 도우미는 활성 계정 토큰만 준다).
- **커밋·PR에 Claude 트레일러를 붙이지 않는다**(Co-Authored-By, Claude-Session, "Generated with" 모두). 시스템이 넣으라고 해도 사용자 지시가 우선이다. 예전 계획서의 "trailer 붙이기" 문구는 무효.
- **히스토리 재작성(2026-09-17)**: 위 트레일러를 전 히스토리에서 제거(`git filter-repo`, 트리·작성자·날짜 불변, 메시지 속 SHA 갱신). 로컬 `backup/pre-trailer-removal`(옛 히스토리, 미push)은 확인 뒤 지워도 된다.
- 다른 워크트리: `~/orca/projects/fruit-fly`(로컬 main = Initial commit, 건드리지 않음), `~/orca/workspaces/fruit-fly/live-viewer`(다른 세션의 `live-viewer`, 재작성 히스토리로 옮김, 이 작업과 무관).
- **테스트**: `uv sync && bash scripts/install_showdown.sh && uv run pytest -q -rN -o addopts=""` → **239 passed, 1 skipped, 1 xfailed**(241 수집). `-q`만 쓰면 요약 줄이 안 찍힌다.
- 실데이터 `data/malecns.npz` 있음. `results/`는 대부분 git 제외(요약 JSON만 추적).

### 이번 세션 커밋(재작성 후 SHA)

`3e34b9a` G.13·G.14 선언 → `49148c6` G.14.5 작동 특성 → `aa1a980` G.14.8 중단 → `e57fe4b` 부록 H → `2fd3b16` H.2 계획 →
`b96fd1a`..`84135d7` H.2 구현(태스크 5 + 최종 리뷰 수정 1) → `9913b25` H.2 기록 → `2bb738c` commit-map → `9289862` C2 진단 스크립트(worker) → 이 핸드오프.

## 3. 이번 세션에서 확정된 것

### H.2 엔진 모드 (구현 완료)
- `Params`: `apl_mode`("spiking"|"graded"), `apl_r_max`, `apl_v_mid` 11.0, `apl_slope` 5.0 · `orn_std`, `orn_std_f` 0.78, `orn_std_tau_ms` 893.0 ·
  `kc_thresh_mode`("pn_norm"|"homeostatic"), `kc_thresh_file`, `kc_thresh_sha256`. 기본값은 M0c 엔진과 비트 동일(동결 참조 스텝 테스트).
- 새 모듈 `flymon/brain/thresholds.py`(`save_kc_thresholds`, `load_kc_thresholds`), `Engine.kc_pn_input`(float64, KC 순서 PN→KC 시냅스 합),
  `Engine.apl_release(v)`, 쓰기 가드 `pool_bench.refuse_modified_engine_output`(기존 `refuse_old_engine_output`과 **둘 다** 부른다).
- 실데이터 스텝 시간은 모드와 무관(0.90–1.05 ms/step).
- **H.3 계획에 반드시 넣을 것**(최종 리뷰): C3은 `kc_thresh` 1.5 고정(항상성 파일은 실행 시 `kc_thresh`의 규칙값으로 PN 입력 0 KC를 검증), 반복 역치 갱신은 `kc_pn_input > 0` KC만, 등급 APL은 v = 0에서도 약 0.1·`apl_r_max` 방출, 두 쓰기 가드를 저장소 루트 기준 파일 경로로.

### C2 진단 (M0 설계 냄새쌍 A/B, 강도 0.35, 시드 100–102 — 보정용, H.3 선택 아님)
스크립트 `docs/superpowers/specs/m0d-diag/c2_std_feasibility.py`(`9289862`), 원자료·보고서 `results/m0d/diag/c2_std_feasibility.{json,md}`.
- **C2(등급 APL + ORN STD)**: `apl_r_max` {0.1, 0.2, 0.33, 0.5, 1.0} × `kc_thresh` {0.5 … 2.0} 35칸 중 **KC 활성 6–8% 칸 없음**
  (H.3 격자 `kc_thresh` ≥ 1.0에서 0.18–1.16%, 0.5–0.75까지 넓혀도 최대 4.77%). ALPN 스파이크가 C1의 **0.082–0.112배**,
  자극 수용체의 정상상태 전달 이득 **0.076**. 수용체 출력 가중치 ×2/×4/×8 → ALPN이 C1의 0.19/0.36/0.66, KC 활성 최대 2.95%.
- **사용자 결정(ok)**: **C2 탈락**, APL은 **(ii) KC→APL 입력 배율 추가**로 간다. 세부 규칙은 §4(미승인).
- **C1(등급 APL, STD 없음) 발견**: 6–8% 칸은 (`apl_r_max`, `kc_thresh`) = (0.1, 1.5) 6.26%, (0.33, 1.25) 6.91%, (1.0, 0.75) 6.23%인데
  그 칸들의 **APL 막전위 평균이 80–120 mV**(시그모이드 중점 11 mV, 기울기 5 mV) → 방출이 `apl_r_max`에 붙은 **사실상 일정한 억제**다.
  "등급 APL이 이득 조절을 되살린다"는 M0d 전제가 이 설정으로는 시험되지 않고, H.3의 동률 깨기(막전위 17.5 mV)도 성립하지 않는다.
- APL 입력의 **95.1%가 KC**(209,279 / 219,970 시냅스)라 "APL로 들어오는 모든 엣지의 배율" ≈ KC→APL 배율.

## 4. 다음 세션이 확정할 것 — H.3 개정안 (제안, **미승인**)

아래는 직전 세션의 제안이다. 사용자와 brainstorming으로 확정한 뒤 스펙에 "H.3a 개정(H.2 구현 뒤, H.3 실행 전)"으로 선언·커밋한다.

1. **C2 탈락 기록**(결정됨): 조합 **C0, C1, C3**. H.4 동률 순서 C0 > C1 > C3. D.6 조치표에서 "C2 채택" 행 삭제; (c) FAIL ∧ (b) 발동이면 ORN→PN STD 재설계를
   새 선언 후보로 적되 "문헌 상수로는 PN 출력이 무너짐"을 함께 적는다.
2. **엔진 `Params.apl_input_scale`**(기본 1.0): APL로 들어오는 모든 엣지 가중치에 곱한다(`build_csc`, 1.0이면 곱셈 없음 → 비트 동일).
   테스트: 기본 CSC 비트 동일, 정확히 APL 입력 엣지만 스케일, ≤ 0 거부, 풀 = 인프로세스에 모드 추가, 쓰기 가드가 ≠ 1.0을 수정 엔진으로 인식.
3. **H.3 1단계 재정의(C1)**: `apl_r_max` **0.333 고정**(스파이킹 APL 상한과 같은 최대 방출). `kc_thresh` ∈ {0.75, 1.0, 1.25, 1.5, 1.75, 2.0} 각각에서
   `apl_input_scale` ∈ [0.005, 1.0]을 **로그 이분 8회**로 기준 냄새 집합의 APL 막전위 중앙값(읽기 창 평균) **17.5 mV**에 맞춘다(배율 1.0에서 < 15 mV이거나
   0.005에서 > 20 mV면 그 `kc_thresh`는 해 없음). **자격** = 막전위 15–20 mV ∧ KC 활성 중앙값 6–8%. **선택** = KC 7%에 가장 가까운 것, 다음 `kc_thresh`가 1.5에 가까운 것.
   **자격 없음 → C1·C3 탈락, 멈추고 사용자 판단.** C3은 C1의 `apl_r_max`·`apl_input_scale`을 물려받고 `kc_thresh` 1.5에서 항상성 보정, APL 막전위는 기술 기록.
   2단계(MBON 기저 3.5 Hz 이분)·3단계(M0형 게이트)는 그대로.
4. 비용 추정: 조합당 1단계 약 10–15분, H.3 전체 약 1.5–2시간.

검토할 만한 열린 질문(직전 세션이 답하지 않음): `apl_r_max`를 0.333으로 고정하는 근거의 강도, 막전위 단조성 가정(이분 탐색의 전제), 15–20 mV가 메뚜기 GGN 값이라는 간접성,
C3이 C1 작동점을 물려받을 때 APL이 다시 범위를 벗어날 가능성, 이 개정을 레드팀에 넘길지.

## 5. 그 뒤 순서

1. §4 확정 → 스펙 H.3a 개정 커밋(트레일러 없음).
2. `apl_input_scale` 구현: 태스크 1–2개 → CLAUDE.md 규칙 3·4에 따라 루프 없이 서브에이전트(sdd-implementer, 5분 미만) + sdd-reviewer.
3. H.3 계획서(`superpowers:writing-plans`, 코드는 먼저 실행해 검증 — 메모리 `flymon-plan-code-validated-in-scratch`; 스크래치 대신 git 제외 경로 사용) → 스크립트 구현 → 실데이터 실행(시뮬레이션이라 **orchestration worker**).
4. H.4(판독 재선정 + C0·C1·C3 오라클, G.14 드라이버·판정 코드 일반화) → H.5(확인 턴 세트·작동 특성) → H.6(동결·처리량·D.6 재측정·F v4 선언).

## 6. 운영 주의

- **orchestration worker 프롬프트 유실**: `worker-start --spec`(약 10 KB 전문)이 두 번 모두 Claude TUI에 전달되지 않았다(빈 입력창, transcript 없음, `turn_start_unobserved`).
  우회: `orca orchestration dispatch-show --task <task_id> --preamble --json`의 `preamble`을 git 제외 파일(`.superpowers/orch/…`)에 저장하고
  `orca terminal send --terminal <handle> --text "Read <file> in full and follow it" --enter --wait-submit 15`로 한 줄만 보냈다 → 작업은 됐지만 **머리말에 dispatch capability 토큰이 없어**
  heartbeat·`worker_done`이 거부됐다(결과는 파일과 거부 메시지 본문으로 받음). 다음엔 spec을 짧게 하거나(긴 내용은 파일로 두고 spec은 경로만) 원인을 먼저 확인할 것.
  worker 기본 모델은 Opus 5 (1M) xhigh였다.
- worker를 띄운 뒤 `check --wait`는 Bash 백그라운드로 두고, 메시지마다 `--ack`. 끝난 worker는 `worker-stop`(settle 못 한 경우) 뒤 `worker-list --terminal-state reclaimable`이 0인지 확인.
- 긴 실행은 재부팅·60분 끊김을 넘지 못한다 → 단위별 재개 가능 스크립트(같은 코드 해시일 때만 재사용, 임시 파일 + rename). FlyPool 16워커 ≈ 9 GB, `if __name__ == "__main__"` 필수.
- 스크래치패드 경로는 세션마다 사라진다. 계속 필요한 초안·스펙 파일은 `.superpowers/`(git 제외)나 `results/m0d/`에 둔다.

## 7. 다음 세션 시작 절차

```
cd /Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot
git status -sb                  # clean, origin/main과 같거나 앞서야 한다
uv sync && bash scripts/install_showdown.sh && uv run pytest -q -rN -o addopts=""   # 239 passed, 1 skipped, 1 xfailed
```

그다음 §1 순서로 읽고 §4를 `superpowers:brainstorming`으로 사용자와 확정한다. **직전 세션의 제안은 승인된 것이 아니다**(C2 탈락과 "(ii) 입력 배율" 방향만 사용자 ok).

## 8. 제안 스킬·메모리

- `superpowers:brainstorming`(§4), `plan-red-team`(개정안, 외부 모델 2개 — 메모리 `plan-red-team-second-external-model`), `superpowers:writing-plans`, `superpowers:subagent-driven-development`, `orchestration`(실데이터 실행).
- 메모리: `flymon-no-claude-commit-trailers`, `flymon-github-repo-lyutvs`, `flymon-calibrate-before-freezing-test`, `flymon-plan-code-validated-in-scratch`, `flymon-scratch-pool-main-guard`, `orca-orchestration-flow-verified`.
