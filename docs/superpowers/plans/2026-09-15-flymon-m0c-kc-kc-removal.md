# FlyMon M0c — KC→KC 빠른 흥분 제거와 재게이트 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KC→KC 엣지를 CSC에서 제거하는 파라미터(`kc_kc_scale`, 기본 0.0)를 엔진에 넣고, 옛 엔진(`kc_kc_scale=1.0`)이 `results/m0`을 비트 동일 재현함을 테스트로 고정한 뒤, 스펙 부록 D.4에 사전 등록한 M0c 게이트(희소성 ∧ 기저 8시드 ∧ 폭주 ∧ 등가성 ∧ 처리량; 조건화는 시드 8–15로 기록)를 잴 수 있는 스크립트와 요약 작성기를 만든다. 실제 데이터 실행과 `results/` 쓰기는 컨트롤러가 마지막 절에서 한다.

**Architecture:** `Params.kc_kc_scale`이 `connectome.build_csc(conn, params, apl_idx, kc_idx)`에서 pre∈KC ∧ post∈KC 엣지에 적용된다(0이면 엣지를 CSC에서 뺀다, 1.0이면 곱셈 없이 옛 CSC와 비트 동일). 워커 잡 `pool_jobs.odor_runaway_job`이 냄새 창 폭주 검사를, `pool_bench.m0c_gate`가 게이트 합성을 맡는다. `scripts/reproduce_flybrain_measurements.py`(희소성·인프로세스 조건화)와 `scripts/bench_pool.py`(풀 조건화·기저·폭주·등가성·처리량)에 `--kc-kc-scale`·`--seed-start`·`--odor-runaway-seeds`·`--arm-equal`·`--conditioning-out`이 붙고, 새 `scripts/write_m0c_summary.py`가 결과 파일들을 `results/summary/m0c.json`으로 합친다. 옛 결과 파일과 옛 스크립트 경로 기본값은 그대로다.

**Tech Stack:** Python 3.13(uv), numpy, multiprocessing(spawn), pytest. 새 의존성 없음. 실제 데이터 `data/malecns.npz`(git 제외).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` — 부록 D(D.1 조사, D.2 반사실, D.3 결정, **D.4 사전 등록 게이트(150 Hz 수정 포함)**, D.5 기록 규칙, D.6 STD 조건, D.7 한계), A.6, 3.1 엔진 항목, 5절 M0c, 6절 파일 목록. 이 계획은 D.3–D.5를 구현한다.

## Global Constraints

- Python 3.13, uv. **새 의존성 추가 금지.** numpy 2.x.
- `kc_kc_scale` 기본값 **0.0**(스펙 D.3). `kc_kc_scale=1.0`은 M0·M0b 엔진이며 CSC가 옛 코드와 **비트 동일**해야 한다(곱셈을 아예 하지 않는다). 0.0이면 KC→KC 엣지를 CSC에서 **제거**한다(가중치 0으로 남기지 않는다). 0과 1 사이 값은 `mv`에 곱한다.
- `hemisphere_scale`(원시 w, KC→KC 포함)과 `shuffle_kc_mbon`(KC→MBON만)은 바꾸지 않는다. `mbon_hold_frac` 등 다른 파라미터는 손대지 않는다(D.3).
- `results/m0/`, `results/summary/m0.json`, `results/summary/m0b.json`은 **불변**이다. 스크립트의 기본 출력 경로는 옛 것 그대로지만, `kc_kc_scale ≠ 1.0`으로 그 경로에 쓰려 하면 **거부**한다(`pool_bench.refuse_old_engine_output`, 레드팀 P0). M0c 실행은 명시 경로(`results/m0c/…`, `results/summary/m0c.json`)로만 쓴다(D.5). 옛 엔진의 소비자(`write_m0_summary.py`, `bench_pool.py summary`, M0 실제 데이터 테스트)는 `kc_kc_scale` 1.0 행만 고른다 — 기존 테스트의 수치·구간은 바꾸지 않는다.
- 요약 작성기는 엔진 정체와 사전 등록 표본 수(판정 시드 8–15, 희소성 100–102, 휴지 100–107, 냄새 B 100–163, 임계 100/150 Hz, 등가성 40건, 워커 (4, 8, 16))를 검증해 부족 항목을 이름 붙여 거부한다(레드팀 P0). `m0c_gate`의 폭주 항은 표본 수를 정의에 포함한다.
- 폭주 임계: 휴지 100 Hz(`baseline_job`, 그대로), 냄새 창 **150 Hz**(`odor_runaway_job` 기본값, D.4 수정). 100 Hz 초과 수와 상위 5개 KC 발화율도 기록한다.
- **서브에이전트는 `results/` 아래에 쓰지 않고 실제 데이터 벤치·재현을 돌리지 않는다**(컨트롤러 실행, 마지막 절). 스크립트 스모크는 `--out <tmp>`로만 하며 이 계획에서는 요구하지 않는다(코드는 이미 실제 데이터로 검증됐다, 아래).
- 커밋은 태스크마다. 메시지 `feat(brain): …`, `test(brain): …`, `feat(scripts): …`, `docs: …`.
- 계획의 코드는 세션 스크래치(작업 트리 초안)에서 **실행해 검증한 것**이고 계획 레드팀(Codex gpt-5.6-sol + 호스트, RETHINK → P0 2·P1 3·P2 3·P3 4 반영)을 거쳤다: 전체 스위트 초록(203 passed, 1 xfailed, 2 skipped), 실제 커넥톰에서 `kc_kc_scale=1.0`이 `results/m0/conditioning.json` 시드 0의 noplast·both 팔과 비트 동일(CSC 엣지 6,005,611; 0.0이면 5,972,364 = 정확히 33,247 감소), `bench_pool.py reproduce --seed-start 8 --odor-runaway-seeds 3 --arm-equal` 실행(팔 등가성 참, 결정 등가성 참, `--conditioning-out` 작성), `--kc-kc-scale 1.0` 재현(시드 0 조건화 비트 동일), `--sparsity-seeds` 분리 뒤 풀·인프로세스 희소성 차이 0.0, `throughput` CLI, `write_m0c_summary.py` 합성과 짧은 입력 거부(부족 항목 7개 이름), 옛 경로 쓰기 가드 5개 명령 전부 거부(참조 파일 mtime 불변), 옛 엔진 `bench_pool.py summary`가 기존 `m0b.json`의 게이트·예산과 동일 재작성. 실행 중 편차가 생기면 보고하고 판정을 받는다.

---

## 검증된 사실 (2026-09-15, 실제 커넥톰)

- `build_csc`의 KC→KC 마스크는 원시 엣지 배열에서 `is_kc[pre] & is_kc[post]`로 만들고, 0이면 `keep`에서 빼며, 0·1이 아닐 때만 `mv`에 곱한다. 1.0에서 옛 함수 본문과 CSC(`ptr`, `tgt`, `w`)가 비트 동일하다(합성 테스트 `_build_csc_m0` 참조 구현으로 고정).
- 새 엔진에서 냄새 B 결정 창(settle 800 + read 600 ms)에 100 Hz를 넘는 KC는 PN 구동 후각 KC 두 개(KCγ-d 우반구 106–108 Hz, KCαβ-s 우반구 110–113 Hz)이며 옛 엔진에서도 같은 세포가 110–115 Hz다. clique(KCab-p 우반구 62개)는 옛 엔진에서 160–250 Hz. 그래서 D.4의 냄새 창 임계는 150 Hz다(실행 전 수정, D.4에 기록).
- 풀 `reproduce`(6워커, 시드 1개, 휴지 2시드, 폭주 3시드, 팔 등가성 2쌍)는 275–289초. 팔 등가성은 인프로세스 `run_arm` 두 번(약 160초)이 지배한다.
- 희소성 스크립트의 grid 행에 `kc_kc_scale`, `sparsity_seeds`, `mbon_hz_rest_trimmed_per_seed`가 들어간다. 풀 `reproduce`의 희소성 시드 수는 `--sparsity-seeds`(기본 3)로 `--rest-seeds`와 분리돼 있다 — 묶여 있으면 `--rest-seeds 8`에서 풀 희소성이 8시드 평균이 되어 인프로세스(3시드)와 어긋난다(레드팀 P1). 옛 `results/m0/sparsity.json` 행에는 두 키가 없다 — 행 선택은 `g.get("kc_kc_scale", 1.0)`으로 옛 행을 1.0 엔진으로 읽는다.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `flymon/brain/config.py` | `Params.kc_kc_scale: float = 0.0` |
| `flymon/brain/connectome.py` | `build_csc(conn, params, apl_idx, kc_idx)`: KC→KC 스케일·제거 |
| `flymon/brain/engine_cpu.py` | `build_csc` 호출에 `pops.kc` 전달 |
| `flymon/brain/measure.py` | `mbon_baseline_multi`에 `mbon_hz_rest_trimmed_per_seed` |
| `flymon/brain/pool_jobs.py` | `odor_runaway_job` |
| `flymon/brain/pool_bench.py` | `refuse_old_engine_output`, `m0c_gate`(표본 수 포함) |
| `scripts/reproduce_flybrain_measurements.py` | `--kc-kc-scale`(둘 다), `--seed-start`(conditioning), grid 행 키 |
| `scripts/bench_pool.py` | `--kc-kc-scale`, `--seed-start`, `--odor-runaway-seeds`, `--odor-sat-hz`, `--conditioning-out`, `--arm-equal`, 빈 참조 허용, `rest-seeds 0` 허용 |
| `scripts/write_m0_summary.py` | M0(옛 엔진) 요약: `kc_kc_scale` 1.0 행만 선택 |
| `scripts/write_m0c_summary.py` | `results/summary/m0c.json` 합성(`pick_row`, `validate_inputs`, `compose`; git 커밋·dirty·입력 sha256) |
| `tests/brain/test_connectome.py`, `test_config.py`, `test_shuffle_kc_mbon.py` | Task 1 테스트와 호출부 |
| `tests/brain/test_fly_pool.py`, `test_pool_bench.py` | Task 2 테스트 |
| `tests/test_write_m0c_summary.py`, `tests/test_summary.py`, `tests/brain/test_measure.py` | Task 3 테스트(작성기 단위·거부, m0c.json 검사, M0c 희소성 게이트 파일 검사) |
| `README.md`, 스펙 6절 | Task 4 문서(M0c 절, 결정 항목, M0 명령에 `--kc-kc-scale 1.0`) |

---

### Task 1: `kc_kc_scale` — 파라미터, `build_csc`, 엔진, 합성 테스트

**Files:**
- Modify: `flymon/brain/config.py`(`balance_hemispheres` 다음 줄), `flymon/brain/connectome.py`(`build_csc`), `flymon/brain/engine_cpu.py:31`
- Test: `tests/brain/test_connectome.py`(`test_csc_matches_dense` 교체 + 테스트 2개 추가 + 호출부 2곳), `tests/brain/test_config.py`, `tests/brain/test_shuffle_kc_mbon.py`(호출부 2곳)

**Interfaces:**
- Produces: `Params.kc_kc_scale: float` (기본 0.0); `build_csc(conn: Connectome, params: Params, apl_idx: np.ndarray, kc_idx: np.ndarray) -> CSC` — `kc_idx`는 필수 위치 인자(빠뜨리면 `TypeError`, 옛 엔진으로 조용히 떨어지지 않는다). Task 2·3은 `Params(kc_kc_scale=…)`만 쓴다.

- [ ] **Step 1: 실패하는 테스트 — `tests/brain/test_connectome.py`**

파일 머리의 import를 다음으로 바꾼다(`dataclasses`, `Path` 추가).

```python
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.config import Params
```

기존 `test_csc_matches_dense`를 지우고 그 자리에 다음 블록(헬퍼 3개 + 테스트 3개)을 넣는다. 나머지 테스트(`test_csc_drops_below_threshold_and_zero_sign`, `test_load_rejects_out_of_range_edge_index`, `test_sign_override_counts_only_actual_changes`)는 그대로 둔다.

```python
def _dense_reference(c, p, apl, kc):
    """M[post, pre] = sign[pre] * w * mv_per_synapse, APL rows scaled by apl_scale, KC->KC entries
    scaled by kc_kc_scale (0 drops them), edges below min_weight or from sign-0 cells skipped."""
    M = np.zeros((c.N, c.N), np.float64)
    sign, _ = apply_sign_override(c, p)
    apl_set, kc_set = set(apl.tolist()), set(kc.tolist())
    for a, b, w in zip(c.pre, c.post, c.w):
        s = sign[a]
        if s == 0 or w < p.min_weight:
            continue
        scale = p.apl_scale if a in apl_set else 1.0
        if a in kc_set and b in kc_set:
            scale *= p.kc_kc_scale
        M[b, a] += s * w * p.mv_per_synapse * scale
    return M


def _with_kc_kc_edges(c, n_edges=30, seed=0):
    """The synthetic fixture has no KC->KC edges; append `n_edges` of them (weights 1..9) so the
    kc_kc_scale rule has something to act on."""
    rng = np.random.default_rng(seed)
    kc = np.flatnonzero(c.cls == "Kenyon_Cell")
    a, b = rng.choice(kc, n_edges), rng.choice(kc, n_edges)
    keep = a != b
    return dataclasses.replace(c, pre=np.concatenate([c.pre, a[keep]]).astype(c.pre.dtype),
                               post=np.concatenate([c.post, b[keep]]).astype(c.post.dtype),
                               w=np.concatenate([c.w, rng.integers(1, 10, keep.sum())]).astype(c.w.dtype))


def test_csc_matches_dense(synthetic_connectome):
    c = _with_kc_kc_edges(synthetic_connectome())
    apl, kc = np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell")
    for scale in (0.0, 0.5, 1.0):
        p = Params(min_weight=1, balance_hemispheres=False, kc_kc_scale=scale)
        csc = build_csc(c, p, apl, kc)
        M = _dense_reference(c, p, apl, kc)
        for i in range(c.N):
            col = np.zeros(c.N)
            np.add.at(col, csc.tgt[csc.ptr[i]:csc.ptr[i + 1]], csc.w[csc.ptr[i]:csc.ptr[i + 1]])
            np.testing.assert_allclose(col, M[:, i], atol=1e-6)


def test_kc_kc_scale_zero_drops_exactly_the_kc_kc_edges(synthetic_connectome):
    """kc_kc_scale=0 removes the KC->KC edges from the CSC (not merely zeroes them): the edge count
    falls by exactly the number of KC->KC edges that pass min_weight, and nothing else moves."""
    c = _with_kc_kc_edges(synthetic_connectome())
    apl, kc = np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell")
    p1, p0 = Params(min_weight=3), Params(min_weight=3, kc_kc_scale=0.0)
    assert p1.kc_kc_scale == 0.0 and Params(kc_kc_scale=1.0).kc_kc_scale == 1.0   # default is off
    csc1, csc0 = build_csc(c, Params(min_weight=3, kc_kc_scale=1.0), apl, kc), build_csc(c, p0, apl, kc)
    is_kc = np.zeros(c.N, bool); is_kc[kc] = True
    sign, _ = apply_sign_override(c, p1)
    n_kc_kc = int((is_kc[c.pre] & is_kc[c.post] & (c.w >= 3) & (sign[c.pre] != 0)).sum())
    assert n_kc_kc > 0
    assert len(csc0.w) == len(csc1.w) - n_kc_kc
    pre0 = csc0.pre_of_edge()
    assert not (is_kc[pre0] & is_kc[csc0.tgt]).any()
    # every non-KC->KC edge is untouched: same (pre, tgt, w) multiset
    pre1 = csc1.pre_of_edge()
    keep = ~(is_kc[pre1] & is_kc[csc1.tgt])
    np.testing.assert_array_equal(pre1[keep], pre0)
    np.testing.assert_array_equal(csc1.tgt[keep], csc0.tgt)
    np.testing.assert_array_equal(csc1.w[keep], csc0.w)


def _build_csc_m0(conn, params, apl_idx):
    """The M0/M0b build_csc body, verbatim (no KC->KC rule): the reference for kc_kc_scale=1.0."""
    sign, _ = apply_sign_override(conn, params)
    keep = (conn.w >= params.min_weight) & (sign[conn.pre] != 0)
    pre, post = conn.pre[keep], conn.post[keep]
    mv = conn.w[keep].astype(np.float32)
    mv *= sign[pre].astype(np.float32)
    mv *= np.float32(params.mv_per_synapse)
    if params.balance_hemispheres:
        m = conn.side[post] == "R"
        mv[m] *= np.float32(hemisphere_scale(conn))
    is_apl = np.zeros(conn.N, bool)
    is_apl[apl_idx] = True
    m = is_apl[pre]
    mv[m] *= np.float32(params.apl_scale)
    order = np.argsort(pre, kind="stable")
    pre, post, mv = pre[order], post[order], mv[order]
    counts = np.bincount(pre, minlength=conn.N)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ptr, post.astype(np.int32), mv


def test_kc_kc_scale_one_is_the_m0_csc_bit_for_bit(synthetic_connectome):
    """kc_kc_scale=1.0 must reproduce the M0/M0b CSC exactly (same float32 operation order, no extra
    multiply), so the old result files stay bit-exact reproduction references."""
    c = _with_kc_kc_edges(synthetic_connectome())
    apl, kc = np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell")
    p = Params(min_weight=1, kc_kc_scale=1.0)
    csc = build_csc(c, p, apl, kc)
    ptr, tgt, w = _build_csc_m0(c, p, apl)
    assert np.array_equal(csc.ptr, ptr) and np.array_equal(csc.tgt, tgt) and np.array_equal(csc.w, w)
```

같은 파일의 `test_csc_drops_below_threshold_and_zero_sign` 안 호출을 바꾼다.

```python
    csc = build_csc(c, p, np.flatnonzero(c.type == "APL"), np.flatnonzero(c.cls == "Kenyon_Cell"))
```

`tests/brain/test_shuffle_kc_mbon.py`의 두 호출:

```python
    csc0 = build_csc(c, p, pops.apl, pops.kc)
    csc1 = build_csc(shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3), p, pops.apl, pops.kc)
```

`tests/brain/test_config.py::test_our_design_defaults` 끝에 한 줄:

```python
    assert p.kc_kc_scale == 0.0   # KC->KC fast excitation removed (spec appendix D)
```

`tests/brain/test_connectome.py` 파일 끝에 실제 데이터 엣지 수 테스트(`data/malecns.npz` 없으면 skip):

```python
def test_real_kc_kc_scale_edge_counts():
    """On MaleCNS the M0 engine (kc_kc_scale=1.0) has 6,005,611 CSC edges and the M0c engine (0.0) exactly 33,247 fewer:
    the KC->KC edges at min_weight 5 (spec D.1). Fixed numbers so a later change to the rule cannot pass unnoticed."""
    from flymon.brain.circuits import Populations
    c = Connectome.load("data/malecns.npz")
    pops = Populations.from_connectome(c)
    n_old = len(build_csc(c, Params(kc_kc_scale=1.0), pops.apl, pops.kc).w)
    n_new = len(build_csc(c, Params(), pops.apl, pops.kc).w)
    assert (n_old, n_new) == (6_005_611, 5_972_364)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest -q tests/brain/test_connectome.py tests/brain/test_config.py tests/brain/test_shuffle_kc_mbon.py`
Expected: FAIL — `TypeError: build_csc() takes 3 positional arguments but 4 were given`, `Params.__init__() got an unexpected keyword argument 'kc_kc_scale'`, `AttributeError: 'Params' object has no attribute 'kc_kc_scale'`.

- [ ] **Step 3: 구현 — `flymon/brain/config.py`**

`balance_hemispheres: bool = True` 바로 아래에:

```python
    balance_hemispheres: bool = True
    kc_kc_scale: float = 0.0       # multiplier on KC->KC edge weights; 0 removes them from the CSC.
                                   # KC axo-axonic contacts act through mAChR-B and suppress neighbouring
                                   # KCs (Manoim et al. 2022), so fast excitation is the wrong sign; 0 is
                                   # the first-order stand-in. 1.0 reproduces the M0/M0b engine (spec D).
```

- [ ] **Step 4: 구현 — `flymon/brain/connectome.py`의 `build_csc` 전체 교체**

파일 머리 `from dataclasses import dataclass, replace` 위에 `import math`를 추가하고, `build_csc`를 다음으로 바꾼다. 1.0이면 마스크를 만들지 않는다(비트 동일성과 메모리, 레드팀 P3).

```python
def build_csc(conn: Connectome, params: Params, apl_idx: np.ndarray, kc_idx: np.ndarray) -> CSC:
    """Signed mV out-edges. Edges below `min_weight` synapses or from sign-0 cells are dropped; APL
    out-edges are scaled by `apl_scale`; KC->KC edges are scaled by `kc_kc_scale` and, when that is 0,
    dropped from the CSC altogether (spec appendix D). Right-hemisphere inputs get the hemisphere factor."""
    if not math.isfinite(params.kc_kc_scale):
        raise ValueError(f"kc_kc_scale must be finite, got {params.kc_kc_scale!r}")
    sign, _ = apply_sign_override(conn, params)
    keep = (conn.w >= params.min_weight) & (sign[conn.pre] != 0)
    kc_kc = None
    if params.kc_kc_scale != 1.0:      # 1.0 is the M0/M0b engine: no mask, no multiply, bit-identical CSC
        is_kc = np.zeros(conn.N, bool)
        is_kc[np.asarray(kc_idx, dtype=np.int64)] = True
        kc_kc = is_kc[conn.pre] & is_kc[conn.post]
        if params.kc_kc_scale == 0.0:
            keep &= ~kc_kc
    pre, post = conn.pre[keep], conn.post[keep]
    # float32 end to end and in-place masked multiplies: at 100M+ edges each float64 temporary
    # costs ~1 GB, and np.where would allocate a second full-length array per correction
    mv = conn.w[keep].astype(np.float32)
    mv *= sign[pre].astype(np.float32)
    mv *= np.float32(params.mv_per_synapse)
    if params.balance_hemispheres:
        m = conn.side[post] == "R"
        mv[m] *= np.float32(hemisphere_scale(conn))
    is_apl = np.zeros(conn.N, bool)
    is_apl[apl_idx] = True
    m = is_apl[pre]
    mv[m] *= np.float32(params.apl_scale)
    if kc_kc is not None and params.kc_kc_scale != 0.0:
        m = kc_kc[keep]
        mv[m] *= np.float32(params.kc_kc_scale)
    order = np.argsort(pre, kind="stable")
    pre, post, mv = pre[order], post[order], mv[order]
    counts = np.bincount(pre, minlength=conn.N)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return CSC(ptr=ptr, tgt=post.astype(np.int32), w=mv)
```

- [ ] **Step 5: 구현 — `flymon/brain/engine_cpu.py:31`**

```python
        self.csc = build_csc(conn, params, pops.apl, pops.kc)
```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest -q tests/brain/test_connectome.py tests/brain/test_config.py tests/brain/test_shuffle_kc_mbon.py`
Expected: PASS (연결체 11개 — 실제 데이터 없으면 그중 1개 skip, 설정 3개, 셔플 전부).

- [ ] **Step 7: 전체 스위트**

Run: `uv run pytest -q`
Expected: 전부 PASS(1 xfailed, 실제 데이터 없으면 skip 몇 개). 합성 픽스처에는 KC→KC 엣지가 없어 기존 동역학 테스트의 수치는 변하지 않는다.

- [ ] **Step 8: 커밋**

```bash
git add flymon/brain/config.py flymon/brain/connectome.py flymon/brain/engine_cpu.py tests/brain/test_connectome.py tests/brain/test_config.py tests/brain/test_shuffle_kc_mbon.py
git commit -m "feat(brain): kc_kc_scale — KC->KC edges scaled or dropped in build_csc (default 0, spec D.3); 1.0 is the M0 CSC bit for bit"
```

---

### Task 2: 냄새 창 폭주 잡, 시드별 기저, M0c 게이트 함수

**Files:**
- Modify: `flymon/brain/pool_jobs.py`(import `numpy`, `odor_runaway_job` 추가, 모듈 docstring 한 구절), `flymon/brain/measure.py`(`mbon_baseline_multi` 반환 키 추가), `flymon/brain/pool_bench.py`(`m0c_gate` 추가)
- Test: `tests/brain/test_fly_pool.py`(import + `test_odor_runaway_job`), `tests/brain/test_pool_bench.py`(import + `_m0c_inputs` + `test_m0c_gate_terms_and_composition`)

**Interfaces:**
- Consumes: `FlyPool.run_jobs(fn, kwargs_list)` 워커 함수 규약 `fn(eng, pl, pops, comps, ro, **kw)`(M0b); `stimuli.design_odor_pair`, `stimuli.present`.
- Produces: `refuse_old_engine_output(out: str, kc_kc_scale: float) -> None` (옛 경로 + `kc_kc_scale != 1.0`이면 `SystemExit`); `odor_runaway_job(eng, pl, pops, comps, ro, seed, which="B", strength=0.35, k=8, odor_seed=0, settle_ms=800.0, read_ms=600.0, sat_hz=150.0) -> dict` with keys `seed, which, sat_hz, n_over_sat, n_kc_over_sat, n_kc_over_100, kc_hz_top5, frac_active_kc, kc_spikes`; `m0c_gate(sparsity_row, baseline, runaway, equivalence, budget, conditioning, limit_hours=60.0) -> dict` with keys `sparsity_ok, baseline_ok, runaway_ok, equivalence_ok, throughput_ok, conditioning_index_flip_ok, channel_specific_seeds, passed`; `mbon_baseline_multi(...)["mbon_hz_rest_trimmed_per_seed"]: list[float]`.

- [ ] **Step 1: 실패하는 테스트 — `tests/brain/test_fly_pool.py`**

import 줄에 `odor_runaway_job`을 넣는다(알파벳순 유지).

```python
from flymon.brain.pool_jobs import (baseline_job, conditioning_arm_job, odor_runaway_job, phase_timing_job, rss_job,
```

`test_conditioning_arm_job_leaves_the_worker_weights_reset` 바로 앞에:

```python
def test_odor_runaway_job(pool):
    """One seed of the odour-window runaway check (spec D.4): counts over the read window, KC subset, and
    the odour actually presented (A or B of the designed pair)."""
    rb = pool.run_jobs(odor_runaway_job, [dict(seed=100, which="B", strength=1.0, k=2, odor_seed=0, settle_ms=200.0, read_ms=600.0)])[0]
    ra = pool.run_jobs(odor_runaway_job, [dict(seed=100, which="A", strength=1.0, k=2, odor_seed=0, settle_ms=200.0, read_ms=600.0)])[0]
    assert set(rb) == {"seed", "which", "sat_hz", "n_over_sat", "n_kc_over_sat", "n_kc_over_100", "kc_hz_top5", "frac_active_kc", "kc_spikes"}
    assert rb["which"] == "B" and ra["which"] == "A" and rb["seed"] == 100 and rb["sat_hz"] == 150.0
    assert 0 <= rb["n_kc_over_sat"] <= rb["n_kc_over_100"] and rb["n_kc_over_sat"] <= rb["n_over_sat"]
    assert 0.0 <= rb["frac_active_kc"] <= 1.0 and len(rb["kc_hz_top5"]) == 5 and rb["kc_hz_top5"] == sorted(rb["kc_hz_top5"], reverse=True)
    assert rb["n_kc_over_sat"] == int(sum(h > 150.0 for h in rb["kc_hz_top5"])) or rb["n_kc_over_sat"] > 5
    with pytest.raises(ValueError, match="which"):
        pool.run_jobs(odor_runaway_job, [dict(seed=100, which="C", strength=1.0, k=2)])
    assert pool.run_jobs(weights_frac_job, [{}])[0] == 1.0     # the job leaves the worker's weights reset
```

- [ ] **Step 2: 실패하는 테스트 — `tests/brain/test_pool_bench.py`**

import에 `m0c_gate, refuse_old_engine_output`을 추가하고(`m0b_gate, m0c_gate, refuse_old_engine_output,`), 파일 끝에:

```python
def _m0c_inputs():
    row = {"frac_active_A": 0.064, "frac_active_B": 0.049, "jaccard": 0.025, "chance": 0.028}
    baseline = {"mbon_hz_rest_trimmed": 3.2}
    runaway = {"rest_n_kc_over_sat_per_seed": [0] * 8, "odor_B_n_kc_over_sat_per_seed": [0] * 64}
    equivalence = {"old_conditioning": {"ok": True}, "old_sparsity": {"ok": True}, "arm_equal": {"ok": True}, "decide_equal": True}
    budget = {"gate_ok": True, "limit_hours": 60.0}
    cond = {"n_seeds": 8, "n_flip": 5, "channel_specific_seeds": 7,
            "arms": {"both": {"mean_dD": 0.0}, "reversed": {"mean_dD": 1.49}}}
    return row, baseline, runaway, equivalence, budget, cond


def test_m0c_gate_terms_and_composition():
    row, baseline, runaway, equivalence, budget, cond = _m0c_inputs()
    g = m0c_gate(row, baseline, runaway, equivalence, budget, cond)
    assert set(g) == {"sparsity_ok", "baseline_ok", "runaway_ok", "equivalence_ok", "throughput_ok",
                      "conditioning_index_flip_ok", "channel_specific_seeds", "passed"}
    assert g["passed"] is True and g["conditioning_index_flip_ok"] is False and g["channel_specific_seeds"] == 7
    # the conditioning criterion is recorded, never gated on
    cond8 = dict(cond, n_flip=8, arms={"both": {"mean_dD": -0.4}, "reversed": {"mean_dD": 1.0}})
    assert m0c_gate(row, baseline, runaway, equivalence, budget, cond8)["conditioning_index_flip_ok"] is True
    # every gate term fails the composite on its own
    assert not m0c_gate(dict(row, frac_active_B=0.02), baseline, runaway, equivalence, budget, cond)["passed"]
    assert not m0c_gate(dict(row, jaccard=0.03), baseline, runaway, equivalence, budget, cond)["passed"]
    assert not m0c_gate(row, {"mbon_hz_rest_trimmed": 2.94}, runaway, equivalence, budget, cond)["baseline_ok"]
    assert not m0c_gate(row, baseline, dict(runaway, odor_B_n_kc_over_sat_per_seed=[0] * 63 + [52]), equivalence, budget, cond)["runaway_ok"]
    assert not m0c_gate(row, baseline, dict(runaway, rest_n_kc_over_sat_per_seed=[0, 52, 0]), equivalence, budget, cond)["runaway_ok"]
    # the sample sizes are part of the pre-registered definition: a short run cannot pass
    assert not m0c_gate(row, baseline, dict(runaway, rest_n_kc_over_sat_per_seed=[0] * 3), equivalence, budget, cond)["runaway_ok"]
    assert not m0c_gate(row, baseline, dict(runaway, odor_B_n_kc_over_sat_per_seed=[0] * 32), equivalence, budget, cond)["runaway_ok"]
    assert not m0c_gate(row, baseline, runaway, dict(equivalence, arm_equal={"ok": False}), budget, cond)["equivalence_ok"]
    assert not m0c_gate(row, baseline, runaway, equivalence, {"gate_ok": False, "limit_hours": 60.0}, cond)["throughput_ok"]
    assert not m0c_gate(row, baseline, runaway, equivalence, {"gate_ok": True, "limit_hours": 80.0}, cond)["throughput_ok"]


def test_refuse_old_engine_output_guards_the_immutable_references():
    """Spec D.5: results/m0*, results/summary/m0.json and m0b.json are the old engine's (kc_kc_scale 1.0) bit-exact
    references and git-ignored; no script writes there with another engine."""
    for out in ("results/m0/sparsity.json", "results/m0/x.json", "results/m0b/throughput.json", "results/summary/m0.json", "results/summary/m0b.json"):
        with pytest.raises(SystemExit, match="old engine"):
            refuse_old_engine_output(out, 0.0)
        refuse_old_engine_output(out, 1.0)                       # the old engine may write its own files
    for out in ("results/m0c/sparsity.json", "results/summary/m0c.json", "/tmp/x.json", ""):
        refuse_old_engine_output(out, 0.0)                       # new paths, empty (unused) paths: fine
```

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest -q tests/brain/test_fly_pool.py tests/brain/test_pool_bench.py`
Expected: FAIL — `ImportError: cannot import name 'odor_runaway_job'`, `cannot import name 'm0c_gate'` / `'refuse_old_engine_output'`.

- [ ] **Step 4: 구현 — `flymon/brain/pool_jobs.py`**

`import time` 아래에 `import numpy as np`를 추가하고(빈 줄로 분리), 모듈 docstring의 "the resting baseline with the runaway set, the peak RSS"를 "the resting baseline with the runaway set, the odour-window runaway check, the peak RSS"로 고친 뒤, `rss_job` 바로 앞에:

```python
def odor_runaway_job(eng, pl, pops, comps, ro, seed: int, which: str = "B", strength: float = 0.35, k: int = 8,
                     odor_seed: int = 0, settle_ms: float = 800.0, read_ms: float = 600.0, sat_hz: float = 150.0) -> dict:
    """The M0c runaway check under odour (spec D.4): present odour A or B of the designed pair for a decision
    window (settle, then read), plasticity off, and count the neurons - and the Kenyon cells among them -
    whose rate over the read window exceeds sat_hz. The KCab-p clique (spec D.1) fires at 160-250 Hz when odour
    B ignites it while PN-driven Kenyon cells top out near 113 Hz, so sat_hz is 150 (the pre-registered
    amendment in D.4); the count above 100 Hz and the five highest KC rates are reported alongside."""
    if which not in ("A", "B"):
        raise ValueError(f"which must be 'A' or 'B', got {which!r}")
    pl.reset_weights()
    pl.set_enabled(False)
    pl.quiet_dan()
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    eng.reset(seed)
    eng.clear_drive()
    present(eng, pops, a if which == "A" else b, strength)
    eng.run(settle_ms)
    counts = eng.run(read_ms)
    pl.set_enabled(True)
    hz = counts / (read_ms / 1000.0)
    over = hz > sat_hz
    kc_hz = hz[pops.kc]
    return {"seed": seed, "which": which, "sat_hz": sat_hz, "n_over_sat": int(over.sum()),
            "n_kc_over_sat": int(over[pops.kc].sum()), "n_kc_over_100": int((kc_hz > 100.0).sum()),
            "kc_hz_top5": [float(x) for x in np.sort(kc_hz)[::-1][:5]],
            "frac_active_kc": float((kc_hz > 0).mean()), "kc_spikes": int(counts[pops.kc].sum())}
```

- [ ] **Step 5: 구현 — `flymon/brain/measure.py`의 `mbon_baseline_multi` 반환 dict**

`"mbon_hz_rest_trimmed_sd": float(trimmed.std()),` 다음 줄에:

```python
        "mbon_hz_rest_trimmed_per_seed": [float(x) for x in trimmed],
```

- [ ] **Step 6: 구현 — `flymon/brain/pool_bench.py`**

`FLOAT_KEYS = (...)` 바로 아래에 옛 엔진 참조 파일의 쓰기 가드(스펙 D.5, 레드팀 P0):

```python
OLD_ENGINE_DIRS = ("results/m0/", "results/m0b/")
OLD_ENGINE_FILES = ("results/summary/m0.json", "results/summary/m0b.json", "results/summary/compartments.json")


def refuse_old_engine_output(out: str, kc_kc_scale: float) -> None:
    """The M0/M0b result files are the old engine's (kc_kc_scale 1.0) immutable, bit-exact references
    (spec D.5) and results/m0* is git-ignored, so an overwrite is unrecoverable. Any script that would write
    under those paths with another engine refuses (SystemExit 2); pass an explicit --out under results/m0c/."""
    if kc_kc_scale == 1.0 or not out:
        return
    norm = str(out).replace("\\", "/")
    if norm.startswith(OLD_ENGINE_DIRS) or norm in OLD_ENGINE_FILES:
        raise SystemExit(f"refusing to write {out} with kc_kc_scale={kc_kc_scale}: that path holds the old engine's "
                         f"(kc_kc_scale=1.0) immutable reference (spec D.5); use --out results/m0c/... or --kc-kc-scale 1.0")


def exact_match(pool_per_seed: dict, m0_per_seed: dict) -> dict:
    """Bit-for-bit comparison of the pool's conditioning results with results/m0/conditioning.json:
    every seed and arm, the four raw probe count pairs and the float indices."""
    n, n_equal, worst = 0, 0, 0.0
    missing = []
    for seed, arms in m0_per_seed.items():
        for arm, ref in arms.items():
            n += 1
            got = pool_per_seed.get(str(seed), pool_per_seed.get(int(seed) if str(seed).isdigit() else seed, {})).get(arm)
            if got is None:
                missing.append(f"{seed}/{arm}")
                continue
            same = got["counts"] == ref["counts"] and all(got[k] == ref[k] for k in FLOAT_KEYS)
            worst = max(worst, max(abs(float(got[k]) - float(ref[k])) for k in FLOAT_KEYS))
            n_equal += int(same)
    return {"n_results": n, "n_equal": n_equal, "n_missing": len(missing), "missing": missing,
            "max_abs_diff": worst, "ok": n > 0 and n_equal == n}


def m0b_gate(budget: dict, conditioning_match: dict, sparsity_match: dict, decide_equal: bool) -> dict:
    return {"throughput_ok": bool(budget["gate_ok"]), "conditioning_exact_ok": bool(conditioning_match["ok"]),
            "sparsity_exact_ok": bool(sparsity_match["ok"]), "decide_equal_ok": bool(decide_equal),
            "passed": bool(budget["gate_ok"] and conditioning_match["ok"] and sparsity_match["ok"] and decide_equal)}
```

파일 끝에 게이트(폭주 항은 표본 수 8·64를 정의에 포함한다):

```python
def m0c_gate(sparsity_row: dict, baseline: dict, runaway: dict, equivalence: dict, budget: dict,
             conditioning: dict, limit_hours: float = 60.0) -> dict:
    """The M0c gate (spec D.4), every term pre-registered. `sparsity_row` is the results/m0c sparsity grid row
    for Params(); `baseline` has "mbon_hz_rest_trimmed" over the 8 rest seeds; `runaway` has the per-seed KC
    counts at rest and under odour B; `equivalence` has the three checks against the old engine and the pool;
    `budget` is budget_table(); `conditioning` is the judged (seeds 8-15) summary. PASS = sparsity and baseline
    and runaway and equivalence and throughput; the conditioning criterion is recorded, not gated on. The runaway
    term includes its pre-registered sample sizes (8 rest seeds, 64 odour seeds): a shorter run cannot pass."""
    sparsity_ok = (0.03 <= sparsity_row["frac_active_A"] <= 0.07 and 0.03 <= sparsity_row["frac_active_B"] <= 0.07
                   and sparsity_row["jaccard"] <= sparsity_row["chance"])
    baseline_ok = 3.0 <= baseline["mbon_hz_rest_trimmed"] <= 4.0
    rest, odor = runaway["rest_n_kc_over_sat_per_seed"], runaway["odor_B_n_kc_over_sat_per_seed"]
    runaway_ok = (len(rest) >= 8 and len(odor) >= 64 and max(rest) == 0 and max(odor) == 0)   # sample sizes are part of the definition
    equivalence_ok = bool(equivalence["old_conditioning"]["ok"] and equivalence["old_sparsity"]["ok"]
                          and equivalence["arm_equal"]["ok"] and equivalence["decide_equal"])
    throughput_ok = bool(budget["gate_ok"]) and budget["limit_hours"] == limit_hours
    both, rev = conditioning["arms"]["both"]["mean_dD"], conditioning["arms"]["reversed"]["mean_dD"]
    flip_ok = (conditioning["n_flip"] == conditioning["n_seeds"] and abs(both) >= 0.3 and abs(rev) >= 0.3
               and (both > 0) != (rev > 0))
    return {"sparsity_ok": bool(sparsity_ok), "baseline_ok": bool(baseline_ok), "runaway_ok": bool(runaway_ok),
            "equivalence_ok": equivalence_ok, "throughput_ok": throughput_ok,
            "conditioning_index_flip_ok": bool(flip_ok), "channel_specific_seeds": int(conditioning["channel_specific_seeds"]),
            "passed": bool(sparsity_ok and baseline_ok and runaway_ok and equivalence_ok and throughput_ok)}
```

- [ ] **Step 7: 통과 확인**

Run: `uv run pytest -q tests/brain/test_fly_pool.py tests/brain/test_pool_bench.py tests/brain/test_measure.py`
Expected: PASS.

- [ ] **Step 8: 커밋**

```bash
git add flymon/brain/pool_jobs.py flymon/brain/measure.py flymon/brain/pool_bench.py tests/brain/test_fly_pool.py tests/brain/test_pool_bench.py
git commit -m "feat(brain): odour-window runaway job (150 Hz), per-seed trimmed baseline, m0c_gate with sample sizes, old-engine output guard (spec D.4/D.5)"
```

---

### Task 3: 스크립트 — `reproduce_flybrain_measurements.py`, `bench_pool.py`, 새 `write_m0c_summary.py`

**Files:**
- Modify: `scripts/reproduce_flybrain_measurements.py`, `scripts/bench_pool.py`, `scripts/write_m0_summary.py`(옛 엔진 행만 선택)
- Create: `scripts/write_m0c_summary.py`
- Test: `tests/test_write_m0c_summary.py`(새 파일), `tests/test_summary.py`(끝에 m0c 검사 추가), `tests/brain/test_measure.py`(옛 행 선택 헬퍼 + M0c 실제 데이터 게이트 테스트)

**Interfaces:**
- Consumes: Task 1 `Params(kc_kc_scale=…)`; Task 2 `odor_runaway_job`, `m0c_gate`; 기존 `pool_bench.budget_table`, `exact_match`, `FLOAT_KEYS`, `conditioning.Readout/run_arm/summarise/channel_specific_seeds`.
- Produces: CLI 옵션(README의 M0c 명령이 그대로 돈다; 옛 경로 + 새 엔진은 거부); `write_m0c_summary.pick_row(grid, p) -> dict`, `validate_inputs(p, sp, old, new, reported, th) -> None`(부족 항목을 이름 붙여 `SystemExit`), `compose(p, sp, old, new, reported, th, limit_hours=60.0, provenance=None) -> dict`; `results/summary/m0c.json` 스키마 = `compose`의 반환(키: `params_frozen, git_commit, git_dirty, inputs_sha256, kc_kc_scale_old_engine, seeds, sparsity, baseline, runaway, conditioning, conditioning_reported_seeds, equivalence, throughput, budget, memory, reproduce_wall_clock_s, generated_at, gate`).

- [ ] **Step 1: 실패하는 테스트 — `tests/test_write_m0c_summary.py`(새 파일, 전문)**

```python
"""scripts/write_m0c_summary.py composes results/summary/m0c.json from the M0c result files (spec D.4/D.5)."""
import importlib.util
from pathlib import Path

import pytest

from flymon.brain.config import Params

_spec = importlib.util.spec_from_file_location("write_m0c_summary", Path("scripts/write_m0c_summary.py"))
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


def _row(**kw):
    r = {"kc_thresh": 1.5, "apl_scale": 0.1, "mbon_hold_frac": 0.85, "kc_kc_scale": 0.0, "sparsity_seeds": [100, 101, 102],
         "frac_active_A": 0.064, "frac_active_B": 0.049,
         "jaccard": 0.025, "chance": 0.028, "mbon_hz_A": 15.7, "mbon_hz_B": 18.1, "mbon_hz_rest": 9.3, "mbon_hz_rest_trimmed": 3.3,
         "mbon_hz_rest_trimmed_sd": 0.8, "mbon_hz_rest_trimmed_per_seed": [3.0, 3.6] * 4, "mbon_n_saturated": 2.0,
         "mbon_types_active_rest": 35.0, "rest_seeds": list(range(100, 108)), "rest_ms": 3000.0}
    r.update(kw)
    return r


def test_pick_row_never_matches_an_old_engine_row():
    old = _row(); del old["kc_kc_scale"]          # M0 files predate the key: they are the 1.0 engine
    assert w.pick_row([old], Params(kc_kc_scale=1.0)) is old
    with pytest.raises(SystemExit, match="kc_kc_scale=0.0"):
        w.pick_row([old], Params())
    new = _row()
    assert w.pick_row([old, new], Params()) is new


def _arm(dD):
    return {"mean_dD": dD, "mean_dD_disc": dD}


def _reproduce(seeds, n_rest=8, n_odor=64, odor_over=0, scale=0.0):
    cond = {"n_seeds": len(seeds), "n_flip": 5, "n_flip_disc": 7, "noplast_max_abs_dD": 0.0, "channel_specific_seeds": 7,
            "arms": {"both": _arm(0.0), "reversed": _arm(1.49), "noplast": _arm(0.0), "punish_only": _arm(0.1), "reward_only": _arm(0.2)},
            "per_seed": {}}
    base = [{"mbon_hz": 9.0, "mbon_hz_trimmed": 3.3, "n_saturated": 2, "n_types_active": 35,
             "runaway": {"sat_hz": 100.0, "n_over_sat": 120, "n_kc_over_sat": 0, "spike_share_over_sat": 0.3}} for _ in range(n_rest)]
    odor = [{"seed": 100 + i, "which": "B", "sat_hz": 150.0, "n_over_sat": 130, "n_kc_over_sat": odor_over if i == 0 else 0,
             "n_kc_over_100": 2, "kc_hz_top5": [111.0, 106.0, 95.0, 95.0, 86.0], "frac_active_kc": 0.05, "kc_spikes": 1200} for i in range(n_odor)]
    return {"params": {"kc_kc_scale": scale}, "seeds": seeds, "conditioning": cond,
            "conditioning_match": {"ok": True, "n_results": 40, "n_equal": 40, "n_missing": 0, "missing": [], "max_abs_diff": 0.0},
            "sparsity_match": {"ok": True, "diffs": {"mbon_hz_rest_trimmed": 0.0}, "max_abs_diff": 0.0},
            "baseline": {"mbon_hz_rest": 9.0, "mbon_hz_rest_trimmed": 3.3, "mbon_hz_rest_trimmed_sd": 0.8, "per_seed": base,
                         "runaway": {"n_over_sat": 120.0, "n_kc_over_sat": 0.0, "spike_share_over_sat": 0.3}},
            "odor_runaway": {"which": "B", "sat_hz": 150.0, "n_seeds": n_odor, "per_seed": odor,
                             "seeds_with_kc_over_100": n_odor, "seeds_with_kc_over_sat": int(odor_over > 0), "max_n_kc_over_sat": odor_over},
            "arm_equal": {"pairs": {"8/both": {"equal": True}, "15/reversed": {"equal": True}}, "ok": True},
            "decide_equal": True, "memory": {"worker_rss_GB": 0.6}, "wall_clock_s": 500.0}


def _throughput(scale=0.0):
    """Shaped like results/m0b/throughput.json rows (C.7 numbers): 16 workers is the fastest end to end."""
    return {"kc_kc_scale": scale, "rows": [{"workers": W, "n_flies_batch": W, "s_decide_batch_max": d, "s_reinforce_batch_max": r, "ms_decision_max": m, "ms_reinforce_max": m}
                     for W, d, r, m in ((4, 9.8, 2.8, 1.8), (8, 11.1, 3.5, 2.2), (16, 16.1, 5.3, 4.4))]}


def _old():
    return _reproduce(list(range(8)), scale=1.0)


def test_compose_records_every_gate_term_and_the_seed_policy():
    p = Params()
    out = w.compose(p, {"grid": [_row()]}, _old(), _reproduce(list(range(8, 16))), _reproduce(list(range(8))), _throughput(),
                    provenance={"sparsity": "abc"})
    assert out["params_frozen"]["kc_kc_scale"] == 0.0 and out["kc_kc_scale_old_engine"] == 1.0
    assert out["inputs_sha256"] == {"sparsity": "abc"} and "git_dirty" in out
    assert out["seeds"]["sparsity"] == [100, 101, 102]
    assert out["seeds"]["conditioning_judged"] == list(range(8, 16)) and out["seeds"]["conditioning_reported"] == list(range(8))
    assert out["runaway"]["odor_B_sat_hz"] == 150.0 and out["runaway"]["odor_B_kc_hz_max"] == 111.0
    assert out["runaway"]["rest_n_kc_over_sat_per_seed"] == [0] * 8 and out["runaway"]["odor_B_n_kc_over_100_per_seed"] == [2] * 64
    assert out["equivalence"]["old_conditioning"]["n_results"] == 40 and out["equivalence"]["arm_equal"]["ok"]
    assert out["budget"]["limit_hours"] == 60.0 and out["budget"]["workers"] == 16
    g = out["gate"]
    assert g["passed"] is True and g["conditioning_index_flip_ok"] is False and g["channel_specific_seeds"] == 7
    assert out["conditioning"]["n_flip"] == 5 and out["conditioning_reported_seeds"]["n_seeds"] == 8


def test_compose_writes_a_failed_gate_instead_of_refusing():
    p = Params()
    out = w.compose(p, {"grid": [_row(mbon_hz_rest_trimmed=2.94)]}, _old(),
                    _reproduce(list(range(8, 16)), odor_over=52), None, _throughput())
    g = out["gate"]
    assert g["baseline_ok"] is False and g["runaway_ok"] is False and g["passed"] is False
    assert out["conditioning_reported_seeds"] is None and out["seeds"]["conditioning_reported"] is None
    assert out["runaway"]["odor_B_seeds_with_kc_over_sat"] == 1


@pytest.mark.parametrize("bad, match", [
    (lambda k: dict(old=_reproduce(list(range(8)), scale=0.0)), "old engine"),
    (lambda k: dict(new=_reproduce(list(range(8, 16)), scale=1.0)), "new engine"),
    (lambda k: dict(new=_reproduce(list(range(8, 15)))), "8-15"),
    (lambda k: dict(new=_reproduce(list(range(8, 16)), n_rest=3)), "8 rest seeds"),
    (lambda k: dict(new=_reproduce(list(range(8, 16)), n_odor=3)), "100-163"),
    (lambda k: dict(sp={"grid": [_row(rest_seeds=[100, 101, 102])]}), "100-107"),
    (lambda k: dict(th={"kc_kc_scale": 0.0, "rows": _throughput()["rows"][:2]}), "workers 4, 8, 16"),
    (lambda k: dict(th=_throughput(scale=1.0)), "throughput must be the new engine"),
])
def test_compose_refuses_wrong_identity_or_short_runs(bad, match):
    """Spec D.4: the gate is defined with its sample sizes and engine identity; a short or mismatched run is refused
    with the shortfall named, never scored."""
    kw = dict(sp={"grid": [_row()]}, old=_old(), new=_reproduce(list(range(8, 16))), reported=None, th=_throughput())
    kw.update(bad(kw))
    with pytest.raises(SystemExit, match=match):
        w.compose(Params(), kw["sp"], kw["old"], kw["new"], kw["reported"], kw["th"])
```

`tests/test_summary.py` 끝에:

```python
P_M0C = Path("results/summary/m0c.json")


@pytest.mark.skipif(not P_M0C.exists(), reason="M0c summary not written yet")
def test_m0c_summary_records_the_pre_registered_gate():
    """M0c (spec D.4): PASS = sparsity and baseline and runaway and equivalence and throughput; the conditioning
    criterion is recorded alongside. The summary is the new engine (kc_kc_scale 0.0) and names the old one."""
    d = json.loads(P_M0C.read_text())
    g = d["gate"]
    assert set(g) == {"sparsity_ok", "baseline_ok", "runaway_ok", "equivalence_ok", "throughput_ok",
                      "conditioning_index_flip_ok", "channel_specific_seeds", "passed"}
    assert g["passed"] == all(g[k] for k in ("sparsity_ok", "baseline_ok", "runaway_ok", "equivalence_ok", "throughput_ok"))
    assert d["params_frozen"]["kc_kc_scale"] == Params().kc_kc_scale == 0.0 and d["kc_kc_scale_old_engine"] == 1.0
    assert d["sparsity"]["kc_kc_scale"] == 0.0
    assert d["seeds"]["conditioning_judged"] == list(range(8, 16)) and d["seeds"]["sparsity"] == [100, 101, 102]
    assert d["runaway"]["rest_seeds"] == list(range(100, 108)) and len(d["runaway"]["rest_n_kc_over_sat_per_seed"]) == 8
    assert d["runaway"]["odor_B_n_seeds"] == 64 and d["runaway"]["odor_B_sat_hz"] == 150.0
    assert d["equivalence"]["old_conditioning"]["n_results"] == 40
    # the terms the milestone exists for, predicted PASS in spec D.4: asserted, not merely recorded
    assert d["equivalence"]["old_conditioning"]["ok"] is True and d["gate"]["equivalence_ok"] is True
    assert d["gate"]["runaway_ok"] is True
    assert d["budget"]["limit_hours"] == 60.0 and {r["workers"] for r in d["throughput"]} >= {4, 8, 16}
    assert d["git_commit"] and set(d["inputs_sha256"]) >= {"sparsity", "reproduce_old", "reproduce", "throughput"}
```

`tests/brain/test_measure.py`(diff 그대로: 옛 테스트는 1.0 행만 고르고 수치는 그대로, M0c 게이트 테스트 추가 — `results/m0c/sparsity.json`이 없으면 skip):

```diff
diff --git a/tests/brain/test_measure.py b/tests/brain/test_measure.py
index 8c30553..5cec67e 100644
--- a/tests/brain/test_measure.py
+++ b/tests/brain/test_measure.py
@@ -92,16 +92,31 @@ def test_mbon_baseline_is_measured_per_params(synthetic_connectome):
     assert len(out) == 2
 
 
+def _gate_row(path: str, p: Params) -> dict:
+    d = json.loads(Path(path).read_text())
+    pick = [g for g in d["grid"] if g["kc_thresh"] == p.kc_thresh and g["apl_scale"] == p.apl_scale
+            and g.get("mbon_hold_frac") == p.mbon_hold_frac and g.get("kc_kc_scale", 1.0) == p.kc_kc_scale]
+    assert pick, f"no sparsity grid row in {path} for {p.kc_thresh}/{p.apl_scale}/{p.mbon_hold_frac}/kc_kc_scale={p.kc_kc_scale}"
+    return pick[0]
+
+
 @pytest.mark.skipif(not Path("results/m0/sparsity.json").exists(), reason="gate not yet run on real data")
 def test_real_sparsity_gate_recorded():
-    """The grid row that the frozen Params() defaults select must itself meet the spec 5 gate
-    (KC sparsity 3-7%, overlap at or below chance, trimmed MBON baseline 3-4 Hz)."""
-    d = json.loads(Path("results/m0/sparsity.json").read_text())
-    p = Params()
-    pick = [g for g in d["grid"] if g["kc_thresh"] == p.kc_thresh and g["apl_scale"] == p.apl_scale
-            and g.get("mbon_hold_frac") == p.mbon_hold_frac]
-    assert pick, "no sparsity grid row for the current Params() defaults"
-    g = pick[0]
+    """M0 (the old engine, kc_kc_scale=1.0; its rows predate the key): the grid row for the frozen defaults
+    must itself meet the spec 5 gate (KC sparsity 3-7%, overlap at or below chance, trimmed MBON baseline 3-4 Hz)."""
+    g = _gate_row("results/m0/sparsity.json", Params(kc_kc_scale=1.0))
+    assert 0.03 <= g["frac_active_A"] <= 0.07 and 0.03 <= g["frac_active_B"] <= 0.07
+    assert g["jaccard"] <= g["chance"]
+    assert 3.0 <= g["mbon_hz_rest_trimmed"] <= 4.0
+
+
+@pytest.mark.skipif(not Path("results/m0c/sparsity.json").exists(), reason="M0c gate not yet run on real data")
+def test_real_m0c_sparsity_gate_recorded():
+    """M0c (spec D.4, the new engine = Params() defaults): sparsity 3-7% for both odours, overlap at or below chance,
+    trimmed MBON baseline over 8 rest seeds within 3-4 Hz. Pre-registered; if the run fails a term this test is kept
+    and marked strict xfail with the recorded reason, never relaxed (D.4)."""
+    g = _gate_row("results/m0c/sparsity.json", Params())
+    assert g["kc_kc_scale"] == 0.0 and g["sparsity_seeds"] == [100, 101, 102] and len(g["rest_seeds"]) >= 8
     assert 0.03 <= g["frac_active_A"] <= 0.07 and 0.03 <= g["frac_active_B"] <= 0.07
     assert g["jaccard"] <= g["chance"]
     assert 3.0 <= g["mbon_hz_rest_trimmed"] <= 4.0
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest -q tests/test_write_m0c_summary.py tests/test_summary.py`
Expected: `tests/test_write_m0c_summary.py` — 모듈 로드에서 `FileNotFoundError`(스크립트 없음); `tests/test_summary.py`의 m0c 검사는 파일이 없어 skip.

- [ ] **Step 3: 구현 — `scripts/reproduce_flybrain_measurements.py`(diff 그대로 적용)**

```diff
diff --git a/scripts/reproduce_flybrain_measurements.py b/scripts/reproduce_flybrain_measurements.py
index 7178996..841a11d 100644
--- a/scripts/reproduce_flybrain_measurements.py
+++ b/scripts/reproduce_flybrain_measurements.py
@@ -21,6 +21,7 @@ from flymon.brain.connectome import Connectome
 from flymon.brain.engine_cpu import Engine
 from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline_multi
 from flymon.brain.plasticity import Plasticity
+from flymon.brain.pool_bench import refuse_old_engine_output
 from flymon.brain.stimuli import design_odor_pair, total_drive
 
 
@@ -28,6 +29,7 @@ COMPARTMENTS_OUT = Path("results/summary/compartments.json")
 
 
 def cmd_sparsity(a):
+    refuse_old_engine_output(a.out, a.kc_kc_scale)
     conn = Connectome.load(a.npz)
     pops = Populations.from_connectome(conn)
     comps = compartments(conn, pops, Params().core_frac)
@@ -37,7 +39,7 @@ def cmd_sparsity(a):
     odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
     grid = []
     for kc_thresh, apl, hold in itertools.product(a.kc_thresh, a.apl_scale, a.mbon_hold):
-        p = Params(kc_thresh=kc_thresh, apl_scale=apl, mbon_hold_frac=hold)
+        p = Params(kc_thresh=kc_thresh, apl_scale=apl, mbon_hold_frac=hold, kc_kc_scale=a.kc_kc_scale)
         eng = Engine(conn, pops, p, seed=0)
         rows = []
         for s in range(a.seeds):
@@ -51,9 +53,10 @@ def cmd_sparsity(a):
         # baseline depends on every Params in the grid; several seeds because the raw mean is
         # unstable when the FR1 clique saturates a few MBONs (the gate reads the trimmed mean)
         rest = mbon_baseline_multi(eng, pops, seeds=range(100, 100 + a.rest_seeds), ms=a.rest_ms)
-        grid.append({"kc_thresh": kc_thresh, "apl_scale": apl, "mbon_hold_frac": hold, **mean, **rest})
+        grid.append({"kc_thresh": kc_thresh, "apl_scale": apl, "mbon_hold_frac": hold, "kc_kc_scale": a.kc_kc_scale,
+                     "sparsity_seeds": [100 + s for s in range(a.seeds)], **mean, **rest})
         print(json.dumps(grid[-1]), flush=True)
-    d = Params()
+    d = Params(kc_kc_scale=a.kc_kc_scale)
     default_row = [g for g in grid if (g["kc_thresh"], g["apl_scale"], g["mbon_hold_frac"])
                    == (d.kc_thresh, d.apl_scale, d.mbon_hold_frac)]
     base = (default_row[0] if default_row else
@@ -98,9 +101,10 @@ def _cond_worker(args):
 
 def cmd_conditioning(a):
     from multiprocessing import Pool
+    refuse_old_engine_output(a.out, a.kc_kc_scale)
     params_dict = dict(learn_rate=a.learn_rate, kc_trace_scale=a.kc_trace_scale, da_trace_scale=a.da_trace_scale,
                        recovery_per_pulse=a.recovery, kc_thresh=a.kc_thresh, apl_scale=a.apl_scale,
-                       da_baseline_ms=a.da_baseline_ms)
+                       da_baseline_ms=a.da_baseline_ms, kc_kc_scale=a.kc_kc_scale)
     kw = dict(trials=a.trials, present_ms=a.present_ms, settle_ms=a.settle_ms)
     viz = None
     if a.viz:
@@ -112,7 +116,7 @@ def cmd_conditioning(a):
         viz = (str(uuid.uuid4()), a.viz_every)
         rr.init("flymon-conditioning", recording_id=viz[0], spawn=True)   # opens the viewer; workers join this recording
     jobs = [(a.npz, params_dict, s, a.strength, a.k, a.odor_seed, a.punish_type, a.reward_type, kw, viz)
-            for s in range(a.seeds)]
+            for s in range(a.seed_start, a.seed_start + a.seeds)]
     with Pool(a.jobs) as pool:
         per_seed = dict(pool.map(_cond_worker, jobs))
     params_out = dict(params_dict, punish_type=a.punish_type, reward_type=a.reward_type, settle_ms=a.settle_ms)
@@ -132,10 +136,12 @@ def main():
     s.add_argument("--strength", type=float, default=0.35); s.add_argument("--seeds", type=int, default=3)
     s.add_argument("--k", type=int, default=8); s.add_argument("--odor-seed", type=int, default=0)
     s.add_argument("--rest-seeds", type=int, default=3); s.add_argument("--rest-ms", type=float, default=3000.0)
+    s.add_argument("--kc-kc-scale", type=float, default=Params().kc_kc_scale)   # 1.0 = the M0/M0b engine
     s.set_defaults(fn=cmd_sparsity)
     c = sub.add_parser("conditioning")
     c.add_argument("--npz", default="data/malecns.npz"); c.add_argument("--out", default="results/m0/conditioning.json")
-    c.add_argument("--seeds", type=int, default=8)
+    c.add_argument("--seeds", type=int, default=8); c.add_argument("--seed-start", type=int, default=0)   # M0c judges on seeds 8-15
+    c.add_argument("--kc-kc-scale", type=float, default=Params().kc_kc_scale)
     c.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))   # each worker holds a full connectome
     c.add_argument("--trials", type=int, default=12); c.add_argument("--present-ms", type=float, default=800.0)
     c.add_argument("--strength", type=float, default=0.35); c.add_argument("--k", type=int, default=8); c.add_argument("--odor-seed", type=int, default=0)
```

- [ ] **Step 4: 구현 — `scripts/bench_pool.py`(diff 그대로 적용)**

```diff
diff --git a/scripts/bench_pool.py b/scripts/bench_pool.py
index 2825fa6..24273a8 100755
--- a/scripts/bench_pool.py
+++ b/scripts/bench_pool.py
@@ -19,15 +19,15 @@ from pathlib import Path
 import numpy as np
 
 from flymon.brain.circuits import Populations, compartments
-from flymon.brain.conditioning import channel_specific_seeds, summarise
+from flymon.brain.conditioning import Readout, channel_specific_seeds, run_arm, summarise
 from flymon.brain.config import Params
 from flymon.brain.connectome import Connectome
 from flymon.brain.engine_cpu import Engine
 from flymon.brain.plasticity import Plasticity
 from flymon.brain.stimuli import design_odor_pair
 from flymon.brain.fly_pool import FlyPool, FlySpec
-from flymon.brain.pool_bench import budget_table, exact_match, m0b_gate, throughput_row
-from flymon.brain.pool_jobs import baseline_job, conditioning_arm_job, rss_job, sparsity_job
+from flymon.brain.pool_bench import FLOAT_KEYS, budget_table, exact_match, m0b_gate, refuse_old_engine_output, throughput_row
+from flymon.brain.pool_jobs import baseline_job, conditioning_arm_job, odor_runaway_job, rss_job, sparsity_job
 from flymon.brain.presentation import decide
 
 ARMS = ("both", "reversed", "noplast", "punish_only", "reward_only")
@@ -47,16 +47,18 @@ def _write(path: str, obj: dict) -> None:
 
 
 def cmd_throughput(a):
+    refuse_old_engine_output(a.out, a.kc_kc_scale)
     npz = _npz(a)
     conn = Connectome.load(npz)
     pops = Populations.from_connectome(conn)
     odor_a, _ = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
     rows = []
     for W in a.workers:
-        with FlyPool(npz, Params(), [FlySpec()] * W, workers=W) as pool:
+        with FlyPool(npz, Params(kc_kc_scale=a.kc_kc_scale), [FlySpec()] * W, workers=W) as pool:
             rows.append(throughput_row(pool, odor_a, a.strength, a.steps, a.warm, a.repeats, idx=pops.mbon))
         print(json.dumps(rows[-1]), flush=True)
-    _write(a.out, {"steps": a.steps, "warm": a.warm, "repeats": a.repeats, "strength": a.strength, "odor_seed": a.odor_seed, "rows": rows})
+    _write(a.out, {"steps": a.steps, "warm": a.warm, "repeats": a.repeats, "strength": a.strength, "odor_seed": a.odor_seed,
+                   "kc_kc_scale": a.kc_kc_scale, "rows": rows})
 
 
 def _rss_by_pid(samples) -> dict:
@@ -68,23 +70,32 @@ def _rss_by_pid(samples) -> dict:
 
 
 def cmd_reproduce(a):
+    refuse_old_engine_output(a.out, a.kc_kc_scale)
+    refuse_old_engine_output(a.conditioning_out, a.kc_kc_scale)
     npz = _npz(a)
     t0 = time.perf_counter()
-    m0_cond = json.loads(Path(a.m0_conditioning).read_text()) if Path(a.m0_conditioning).exists() else None
-    m0_sp = json.loads(Path(a.m0_sparsity).read_text()) if Path(a.m0_sparsity).exists() else None
-    p = Params()
+    m0_cond = json.loads(Path(a.m0_conditioning).read_text()) if a.m0_conditioning and Path(a.m0_conditioning).exists() else None
+    m0_sp = json.loads(Path(a.m0_sparsity).read_text()) if a.m0_sparsity and Path(a.m0_sparsity).exists() else None
+    p = Params(kc_kc_scale=a.kc_kc_scale)
+    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
     with FlyPool(npz, p, [FlySpec()] * a.workers, workers=a.workers) as pool:
         jobs = [dict(seed=s, arm=arm, strength=a.strength, k=a.k, odor_seed=a.odor_seed, trials=a.trials,
                      present_ms=a.present_ms, settle_ms=a.settle_ms, punish_type=a.punish_type, reward_type=a.reward_type)
-                for s in range(a.seeds) for arm in ARMS]
+                for s in seeds for arm in ARMS]
         res = pool.run_jobs(conditioning_arm_job, jobs)
         per_seed: dict = {}
         for job, r in zip(jobs, res):
             per_seed.setdefault(job["seed"], {})[job["arm"]] = r
         cond = summarise(per_seed)
         cond["channel_specific_seeds"] = channel_specific_seeds(cond["per_seed"])
-        sp_rows = pool.run_jobs(sparsity_job, [dict(seed=100 + s, strength=a.strength, k=a.k, odor_seed=a.odor_seed) for s in range(a.rest_seeds)])
+        if a.conditioning_out:   # the M0-format conditioning file (spec D.5: the M0c reference is made by the pool)
+            _write(a.conditioning_out, {"params": {**dataclasses.asdict(p), "punish_type": a.punish_type, "reward_type": a.reward_type,
+                                                    "settle_ms": a.settle_ms}, "strength": a.strength, "trials": a.trials,
+                                        "seeds": seeds, "workers": a.workers, **cond})
+        sp_rows = pool.run_jobs(sparsity_job, [dict(seed=100 + s, strength=a.strength, k=a.k, odor_seed=a.odor_seed) for s in range(a.sparsity_seeds)])
         base_rows = pool.run_jobs(baseline_job, [dict(seed=100 + s, ms=a.rest_ms) for s in range(a.rest_seeds)])
+        runaway_rows = pool.run_jobs(odor_runaway_job, [dict(seed=100 + s, which="B", strength=a.strength, k=a.k, odor_seed=a.odor_seed,
+                                                              settle_ms=a.settle_ms, read_ms=600.0, sat_hz=a.odor_sat_hz) for s in range(a.odor_runaway_seeds)])
         rss_before = _rss_by_pid(pool.run_jobs(rss_job, [{}] * (4 * a.workers)))
         rss_after = _rss_by_pid(pool.run_jobs(rss_job, [{}] * (4 * a.workers), shuffle_seed=1_000_003))  # builds one C-shuf variant on every worker it reaches
         both = sorted(set(rss_before) & set(rss_after))
@@ -98,38 +109,58 @@ def cmd_reproduce(a):
         pl = Plasticity(eng, pops, compartments(conn, pops, p.core_frac))
         local = decide(eng, pl, pops, [odor_a, odor_b], a.strength, seed=7, settle_ms=200.0, read_ms=600.0, idx=pops.mbon)
         remote = pool.decide_batch([(0, [odor_a, odor_b], 7)], a.strength, settle_ms=200.0, read_ms=600.0, idx=pops.mbon)[0]
+        # pool == in-process for whole conditioning arms (spec D.5): two (seed, arm) pairs re-run here, bit for bit
+        arm_pairs = [(seeds[0], "both"), (seeds[-1], "reversed")] if a.arm_equal else []
+        ro = Readout.from_compartments(compartments(conn, pops, p.core_frac), a.punish_type, a.reward_type)
+        arm_rows = {}
+        for sd, arm in arm_pairs:
+            got = run_arm(eng, pl, pops, ro, odor_a, odor_b, a.strength, sd, arm, trials=a.trials, present_ms=a.present_ms,
+                          settle_ms=a.settle_ms, punish_type=a.punish_type, reward_type=a.reward_type)
+            pl.reset_weights()
+            arm_rows[f"{sd}/{arm}"] = {"equal": got["counts"] == per_seed[sd][arm]["counts"] and all(got[k] == per_seed[sd][arm][k] for k in FLOAT_KEYS)}
+    arm_equal = {"pairs": arm_rows, "ok": bool(arm_rows) and all(v["equal"] for v in arm_rows.values())}
     decide_equal = bool(np.array_equal(local, remote))
-    sparsity = {k: float(np.mean([r[k] for r in sp_rows])) for k in sp_rows[0]}
+    sparsity = ({k: float(np.mean([r[k] for r in sp_rows])) for k in sp_rows[0]} if sp_rows else {})
     sparsity["per_seed"] = sp_rows
     trimmed = np.array([r["mbon_hz_trimmed"] for r in base_rows])
-    baseline = {"mbon_hz_rest": float(np.mean([r["mbon_hz"] for r in base_rows])), "mbon_hz_rest_trimmed": float(trimmed.mean()),
-                "mbon_hz_rest_trimmed_sd": float(trimmed.std()), "per_seed": base_rows,
-                "runaway": {k: float(np.mean([r["runaway"][k] for r in base_rows])) for k in ("n_over_sat", "n_kc_over_sat", "spike_share_over_sat")}}
-    cond_match = exact_match(per_seed, {k: v for k, v in m0_cond["per_seed"].items() if int(k) < a.seeds}) if m0_cond else {"ok": False, "note": "no M0 file"}
-    sp_match = {"ok": False, "note": "no M0 file"}
-    if m0_sp:
-        row = [g for g in m0_sp["grid"] if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac")) == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac)][0]
+    baseline = {"mbon_hz_rest": float(np.mean([r["mbon_hz"] for r in base_rows])) if base_rows else None,
+                "mbon_hz_rest_trimmed": float(trimmed.mean()) if base_rows else None,
+                "mbon_hz_rest_trimmed_sd": float(trimmed.std()) if base_rows else None, "per_seed": base_rows,
+                "runaway": {k: float(np.mean([r["runaway"][k] for r in base_rows])) for k in ("n_over_sat", "n_kc_over_sat", "spike_share_over_sat")} if base_rows else None}
+    odor_runaway = {"which": "B", "sat_hz": a.odor_sat_hz, "n_seeds": len(runaway_rows), "per_seed": runaway_rows,
+                    "seeds_with_kc_over_100": int(sum(r["n_kc_over_100"] > 0 for r in runaway_rows)),
+                    "seeds_with_kc_over_sat": int(sum(r["n_kc_over_sat"] > 0 for r in runaway_rows)),
+                    "max_n_kc_over_sat": max([r["n_kc_over_sat"] for r in runaway_rows], default=0)}
+    cond_match = exact_match(per_seed, {k: v for k, v in m0_cond["per_seed"].items() if int(k) in seeds}) if m0_cond else {"ok": False, "note": "no reference file"}
+    sp_match = {"ok": False, "note": "no reference file"}
+    if m0_sp and sp_rows:
+        row = [g for g in m0_sp["grid"] if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac"), g.get("kc_kc_scale", 1.0))
+               == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac, p.kc_kc_scale)][0]   # rows without the key are the M0 engine (1.0)
         keys = ("frac_active_A", "frac_active_B", "jaccard", "chance", "mbon_hz_A", "mbon_hz_B")
         diffs = {k: abs(sparsity[k] - row[k]) for k in keys}
         diffs["mbon_hz_rest_trimmed"] = abs(baseline["mbon_hz_rest_trimmed"] - row["mbon_hz_rest_trimmed"])
         sp_match = {"diffs": diffs, "max_abs_diff": max(diffs.values()), "ok": max(diffs.values()) == 0.0, "cpu_row": {k: row[k] for k in keys + ("mbon_hz_rest_trimmed",)}}
     out = {"params": {**dataclasses.asdict(p), "punish_type": a.punish_type, "reward_type": a.reward_type, "settle_ms": a.settle_ms,
                       "present_ms": a.present_ms, "trials": a.trials}, "strength": a.strength, "odor_seed": a.odor_seed,
-           "workers": a.workers, "conditioning": cond, "conditioning_match": cond_match, "sparsity": sparsity, "sparsity_match": sp_match,
-           "baseline": baseline, "memory": memory, "decide_equal": decide_equal, "wall_clock_s": time.perf_counter() - t0}
+           "workers": a.workers, "seeds": seeds, "conditioning": cond, "conditioning_match": cond_match, "sparsity": sparsity, "sparsity_match": sp_match,
+           "baseline": baseline, "odor_runaway": odor_runaway, "arm_equal": arm_equal, "memory": memory, "decide_equal": decide_equal,
+           "wall_clock_s": time.perf_counter() - t0}
     print(json.dumps({"n_seeds": cond["n_seeds"], "n_flip": cond["n_flip"], "channel_specific": cond["channel_specific_seeds"],
                       "conditioning_exact": cond_match.get("ok"), "sparsity_exact": sp_match.get("ok"), "decide_equal": decide_equal,
-                      "runaway": baseline["runaway"], "memory": memory, "wall_s": round(out["wall_clock_s"])}), flush=True)
+                      "arm_equal": arm_equal["ok"], "runaway": baseline["runaway"], "odor_runaway": {k: odor_runaway[k] for k in ("n_seeds", "seeds_with_kc_over_sat", "max_n_kc_over_sat")},
+                      "memory": memory, "wall_s": round(out["wall_clock_s"])}), flush=True)
     _write(a.out, out)
 
 
 def cmd_summary(a):
     th = json.loads(Path(a.throughput).read_text())
     rp = json.loads(Path(a.reproduce).read_text())
+    scale = th.get("kc_kc_scale", 1.0)      # files that predate the key are the old engine
+    refuse_old_engine_output(a.out, scale)
     budget = budget_table(th["rows"], decisions=a.decisions, eval_decisions=a.eval_decisions, limit_hours=a.limit_hours)
     gate = m0b_gate(budget, rp["conditioning_match"], rp["sparsity_match"], rp["decide_equal"])
     co = rp["conditioning"]
-    out = {"params_frozen": dataclasses.asdict(Params()), "throughput": th["rows"], "budget": budget,
+    out = {"params_frozen": dataclasses.asdict(Params(kc_kc_scale=scale)), "throughput": th["rows"], "budget": budget,
            "conditioning": {k: co[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")},
            "conditioning_match": rp["conditioning_match"], "sparsity_match": rp["sparsity_match"], "decide_equal": rp["decide_equal"],
            "runaway": rp["baseline"]["runaway"], "memory": rp.get("memory"),
@@ -147,12 +178,19 @@ def main():
     t.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16]); t.add_argument("--steps", type=int, default=300)
     t.add_argument("--warm", type=int, default=100); t.add_argument("--repeats", type=int, default=3)
     t.add_argument("--strength", type=float, default=0.35); t.add_argument("--k", type=int, default=8); t.add_argument("--odor-seed", type=int, default=0)
+    t.add_argument("--kc-kc-scale", type=float, default=Params().kc_kc_scale)   # 1.0 = the M0/M0b engine
     t.set_defaults(fn=cmd_throughput)
     r = sub.add_parser("reproduce")
     r.add_argument("--npz", default="data/malecns.npz"); r.add_argument("--out", default="results/m0b/reproduce.json")
     r.add_argument("--m0-conditioning", default="results/m0/conditioning.json"); r.add_argument("--m0-sparsity", default="results/m0/sparsity.json")
-    r.add_argument("--workers", type=int, default=16); r.add_argument("--seeds", type=int, default=8)
+    r.add_argument("--workers", type=int, default=16); r.add_argument("--seeds", type=int, default=8); r.add_argument("--seed-start", type=int, default=0)
     r.add_argument("--rest-seeds", type=int, default=3); r.add_argument("--rest-ms", type=float, default=3000.0)
+    r.add_argument("--sparsity-seeds", type=int, default=3)          # pool sparsity seeds 100.. (spec D.4: 100-102), independent of --rest-seeds
+    r.add_argument("--odor-runaway-seeds", type=int, default=0)      # M0c: odour-B decision-window runaway check, seeds 100.. (spec D.4: 64)
+    r.add_argument("--odor-sat-hz", type=float, default=150.0)        # spec D.4 amendment: PN-driven KCs reach ~113 Hz, the clique 160-250 Hz
+    r.add_argument("--kc-kc-scale", type=float, default=Params().kc_kc_scale)
+    r.add_argument("--conditioning-out", default="")                # also write the pool's conditioning in the M0 file format
+    r.add_argument("--arm-equal", action="store_true")             # re-run two (seed, arm) pairs in-process and compare bit for bit
     r.add_argument("--trials", type=int, default=12); r.add_argument("--present-ms", type=float, default=800.0); r.add_argument("--settle-ms", type=float, default=800.0)
     r.add_argument("--strength", type=float, default=0.35); r.add_argument("--k", type=int, default=8); r.add_argument("--odor-seed", type=int, default=0)
     r.add_argument("--punish-type", default="PPL105"); r.add_argument("--reward-type", default="PAM08")
```

- [ ] **Step 5: 구현 — `scripts/write_m0_summary.py`(diff 그대로: M0는 옛 엔진의 기록)**

```diff
diff --git a/scripts/write_m0_summary.py b/scripts/write_m0_summary.py
index 83c4bee..89c9dc5 100755
--- a/scripts/write_m0_summary.py
+++ b/scripts/write_m0_summary.py
@@ -2,7 +2,8 @@
 """Write results/summary/m0.json from the M0 gate outputs.
 
 Reads the data manifest and the two gate result files, picks the sparsity grid
-row matching the current Params() defaults, checks that row against the M0 gate
+row matching the M0 engine (Params() defaults with kc_kc_scale=1.0: M0 predates the KC->KC
+removal of spec appendix D and its files are that engine's immutable record), checks that row against the M0 gate
 (spec 5: KC sparsity 3-7%, overlap at or below chance, trimmed MBON baseline 3-4 Hz) and
 freezes those defaults into the summary. A row that fails the gate is never
 frozen. The conditioning block carries the graded gate statistics (`n_flip`, per-arm `mean_dD`) and,
@@ -54,9 +55,9 @@ def main() -> None:
     man = _load(MANIFEST)
     sp = _load(SPARSITY)
     co = _load(CONDITIONING)
-    p = Params()
+    p = Params(kc_kc_scale=1.0)       # M0 is the old engine's record; rows without the key predate it (spec D.5)
     pick = [g for g in sp["grid"] if g["kc_thresh"] == p.kc_thresh and g["apl_scale"] == p.apl_scale
-            and g.get("mbon_hold_frac") == p.mbon_hold_frac]
+            and g.get("mbon_hold_frac") == p.mbon_hold_frac and g.get("kc_kc_scale", 1.0) == p.kc_kc_scale]
     if not pick:
         print(f"no sparsity grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} "
               f"mbon_hold_frac={p.mbon_hold_frac} in {SPARSITY}")
```

- [ ] **Step 6: 구현 — `scripts/write_m0c_summary.py`(새 파일, 전문)**

```python
#!/usr/bin/env python3
"""Write results/summary/m0c.json: the M0c gate (spec appendix D.4) from the M0c result files.

Inputs (all written by the controller; none is overwritten here):
  results/m0c/sparsity.json           reproduce_flybrain_measurements.py sparsity --rest-seeds 8  (default Params row)
  results/m0c/reproduce_old.json      bench_pool.py reproduce --kc-kc-scale 1.0  vs results/m0  (old-engine equivalence)
  results/m0c/reproduce.json          bench_pool.py reproduce --seed-start 8 --rest-seeds 8 --odor-runaway-seeds 64 --arm-equal
  results/m0c/reproduce_seeds0-7.json bench_pool.py reproduce --seed-start 0 (reported alongside, not judged)
  results/m0c/throughput.json         bench_pool.py throughput
PASS = sparsity and baseline and runaway and equivalence and throughput. The conditioning criterion (pre-registered,
unchanged) is recorded as it comes out; it does not gate M0c. A gate that fails is written as failed, never refused —
but inputs of the wrong engine, the wrong seeds or a short run are refused with the shortfalls named (validate_inputs).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from flymon.brain.config import Params
from flymon.brain.pool_bench import budget_table, m0c_gate


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing input {p}: run the M0c commands in README first")
    return json.loads(p.read_text())


def _git_state() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"git_commit": head, "git_dirty": dirty}
    except Exception:   # not a git checkout: still write the summary
        return {"git_commit": "", "git_dirty": None}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_inputs(p: Params, sp: dict, old: dict, new: dict, reported: dict | None, th: dict) -> None:
    """The pre-registered identity and sample sizes (spec D.4): refuse to compose a gate from the wrong engine,
    a short run, or the wrong seeds, naming every shortfall."""
    problems = []
    row = pick_row(sp["grid"], p)
    if old["params"].get("kc_kc_scale", 1.0) != 1.0:
        problems.append(f"reproduce_old must be the old engine (kc_kc_scale=1.0), got {old['params'].get('kc_kc_scale')}")
    for name, d in (("reproduce", new), ("reproduce_reported", reported)):
        if d is not None and d["params"].get("kc_kc_scale", 1.0) != p.kc_kc_scale:
            problems.append(f"{name} must be the new engine (kc_kc_scale={p.kc_kc_scale}), got {d['params'].get('kc_kc_scale', 1.0)}")
    if th.get("kc_kc_scale", 1.0) != p.kc_kc_scale:
        problems.append(f"throughput must be the new engine (kc_kc_scale={p.kc_kc_scale}), got {th.get('kc_kc_scale', 1.0)}")
    if new["seeds"] != list(range(8, 16)):
        problems.append(f"judged conditioning seeds must be 8-15, got {new['seeds']}")
    if reported is not None and reported["seeds"] != list(range(8)):
        problems.append(f"reported conditioning seeds must be 0-7, got {reported['seeds']}")
    if old["conditioning_match"].get("n_results") != 40:
        problems.append(f"old-engine equivalence must cover 40 M0 results, got {old['conditioning_match'].get('n_results')}")
    if row.get("sparsity_seeds") != [100, 101, 102]:
        problems.append(f"sparsity seeds must be 100-102, got {row.get('sparsity_seeds')}")
    if list(row["rest_seeds"]) != list(range(100, 108)):
        problems.append(f"rest seeds must be 100-107, got {row['rest_seeds']}")
    if len(new["baseline"]["per_seed"]) != 8:
        problems.append(f"pool baseline must have 8 rest seeds, got {len(new['baseline']['per_seed'])}")
    odor_seeds = [r["seed"] for r in new["odor_runaway"]["per_seed"]]
    if odor_seeds != list(range(100, 164)):
        problems.append(f"odour-B runaway seeds must be 100-163, got {len(odor_seeds)} seeds")
    if new["odor_runaway"].get("sat_hz") != 150.0 or any(r["runaway"]["sat_hz"] != 100.0 for r in new["baseline"]["per_seed"]):
        problems.append("runaway thresholds must be 150 Hz (odour window) and 100 Hz (rest)")
    if not new["arm_equal"].get("pairs"):
        problems.append("reproduce must be run with --arm-equal")
    if not {r["workers"] for r in th["rows"]} >= {4, 8, 16}:
        problems.append(f"throughput must cover workers 4, 8, 16, got {sorted(r['workers'] for r in th['rows'])}")
    if problems:
        raise SystemExit("M0c summary refused (spec D.4 identity / sample sizes):\n  - " + "\n  - ".join(problems))


def pick_row(grid: list, p: Params) -> dict:
    """The sparsity grid row for these Params. Rows without `kc_kc_scale` are the M0 engine (1.0)."""
    rows = [g for g in grid if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac"), g.get("kc_kc_scale", 1.0))
            == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac, p.kc_kc_scale)]
    if not rows:
        raise SystemExit(f"no sparsity grid row for kc_thresh={p.kc_thresh} apl_scale={p.apl_scale} "
                         f"mbon_hold_frac={p.mbon_hold_frac} kc_kc_scale={p.kc_kc_scale}")
    return rows[0]


def compose(p: Params, sp: dict, old: dict, new: dict, reported: dict | None, th: dict, limit_hours: float = 60.0,
            provenance: dict | None = None) -> dict:
    validate_inputs(p, sp, old, new, reported, th)
    row = pick_row(sp["grid"], p)
    baseline = {"mbon_hz_rest_trimmed": row["mbon_hz_rest_trimmed"], "mbon_hz_rest_trimmed_sd": row["mbon_hz_rest_trimmed_sd"],
                "per_seed": row.get("mbon_hz_rest_trimmed_per_seed"), "rest_seeds": row["rest_seeds"], "rest_ms": row["rest_ms"],
                "pool_diff": new["sparsity_match"]["diffs"]["mbon_hz_rest_trimmed"] if new["sparsity_match"].get("diffs") else None}
    runaway = {"rest_seeds": [r.get("seed", 100 + i) for i, r in enumerate(new["baseline"]["per_seed"])],
               "rest_n_kc_over_sat_per_seed": [r["runaway"]["n_kc_over_sat"] for r in new["baseline"]["per_seed"]],
               "rest_n_over_sat_per_seed": [r["runaway"]["n_over_sat"] for r in new["baseline"]["per_seed"]],
               "rest_spike_share_over_sat": new["baseline"]["runaway"]["spike_share_over_sat"],
               "odor_B_n_seeds": new["odor_runaway"]["n_seeds"], "odor_B_sat_hz": new["odor_runaway"]["sat_hz"],
               "odor_B_n_kc_over_sat_per_seed": [r["n_kc_over_sat"] for r in new["odor_runaway"]["per_seed"]],
               "odor_B_n_kc_over_100_per_seed": [r["n_kc_over_100"] for r in new["odor_runaway"]["per_seed"]],
               "odor_B_kc_hz_max": max(r["kc_hz_top5"][0] for r in new["odor_runaway"]["per_seed"]),
               "odor_B_seeds_with_kc_over_sat": new["odor_runaway"]["seeds_with_kc_over_sat"]}
    equivalence = {"old_conditioning": old["conditioning_match"], "old_sparsity": old["sparsity_match"],
                   "old_decide_equal": old["decide_equal"], "arm_equal": new["arm_equal"], "decide_equal": new["decide_equal"],
                   "new_sparsity_vs_in_process": new["sparsity_match"]}
    budget = budget_table(th["rows"], limit_hours=limit_hours)
    cond = new["conditioning"]
    judged = {k: cond[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")}
    judged["seeds"] = new["seeds"]
    out = {"params_frozen": dataclasses.asdict(p), **_git_state(), "inputs_sha256": provenance or {}, "kc_kc_scale_old_engine": 1.0,
           "seeds": {"sparsity": row["sparsity_seeds"], "rest": row["rest_seeds"], "odor_runaway": [r["seed"] for r in new["odor_runaway"]["per_seed"]],
                     "conditioning_judged": new["seeds"], "conditioning_reported": reported["seeds"] if reported else None},
           "sparsity": row, "baseline": baseline, "runaway": runaway, "conditioning": judged,
           "conditioning_reported_seeds": ({k: reported["conditioning"][k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")}
                                           if reported else None),
           "equivalence": equivalence, "throughput": th["rows"], "budget": budget, "memory": new.get("memory"),
           "reproduce_wall_clock_s": {"old": old.get("wall_clock_s"), "new": new.get("wall_clock_s")},
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    out["gate"] = m0c_gate(row, baseline, runaway, equivalence, budget, judged, limit_hours)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", default="results/m0c/sparsity.json")
    ap.add_argument("--reproduce-old", default="results/m0c/reproduce_old.json")
    ap.add_argument("--reproduce", default="results/m0c/reproduce.json")
    ap.add_argument("--reproduce-reported", default="results/m0c/reproduce_seeds0-7.json")
    ap.add_argument("--throughput", default="results/m0c/throughput.json")
    ap.add_argument("--out", default="results/summary/m0c.json")
    ap.add_argument("--limit-hours", type=float, default=60.0)
    a = ap.parse_args()
    reported = json.loads(Path(a.reproduce_reported).read_text()) if Path(a.reproduce_reported).exists() else None
    paths = {"sparsity": a.sparsity, "reproduce_old": a.reproduce_old, "reproduce": a.reproduce, "throughput": a.throughput,
             **({"reproduce_reported": a.reproduce_reported} if reported is not None else {})}
    out = compose(Params(), _load(a.sparsity), _load(a.reproduce_old), _load(a.reproduce), reported, _load(a.throughput), a.limit_hours,
                  provenance={k: _sha256(v) for k, v in paths.items()})
    g = out["gate"]
    print(f"M0c gate {'PASS' if g['passed'] else 'FAIL'}: sparsity={g['sparsity_ok']} baseline={g['baseline_ok']} "
          f"({out['baseline']['mbon_hz_rest_trimmed']:.2f} Hz) runaway={g['runaway_ok']} equivalence={g['equivalence_ok']} "
          f"throughput={g['throughput_ok']} ({out['budget']['baseline_hours']:.1f} h); conditioning index flip "
          f"{'ok' if g['conditioning_index_flip_ok'] else 'FAIL (recorded)'}, channel-specific {g['channel_specific_seeds']}/{out['conditioning']['n_seeds']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 통과 확인**

Run: `uv run pytest -q tests/test_write_m0c_summary.py tests/test_summary.py tests/brain/test_measure.py && uv run python scripts/bench_pool.py --help && uv run python scripts/reproduce_flybrain_measurements.py conditioning --help | grep -c "seed-start"`
Expected: 테스트 PASS(m0c 파일 검사들은 skip), `--help` 정상, `1`.

- [ ] **Step 8: 전체 스위트**

Run: `uv run pytest -q`
Expected: 전부 PASS.

- [ ] **Step 9: 커밋**

```bash
git add scripts/reproduce_flybrain_measurements.py scripts/bench_pool.py scripts/write_m0_summary.py scripts/write_m0c_summary.py tests/test_write_m0c_summary.py tests/test_summary.py tests/brain/test_measure.py
git commit -m "feat(scripts): M0c runs — kc-kc-scale/seed-start/sparsity-seeds, old-path guard, odour runaway check, arm equivalence, write_m0c_summary with input validation"
```

---

### Task 4: 문서 — README M0c 절·"우리가 정한 것(M0c)"·M0 명령의 `--kc-kc-scale 1.0`, 스펙 6절

**Files:**
- Modify: `README.md`(diff 그대로), `docs/superpowers/specs/2026-09-14-flymon-design.md` 6절 `scripts/` 줄

- [ ] **Step 1: README(diff 그대로 적용)**

```diff
diff --git a/README.md b/README.md
index f6c73b0..74d2b7d 100644
--- a/README.md
+++ b/README.md
@@ -99,6 +99,13 @@ M0: 엔진과 flybrain 측정값 재현. M1: Showdown 배틀 환경과 뇌 없
   라우터(공격기 & 후보 ≥ 2 → 제공자), 귀속 규칙(직접 피해만, 잔여·대타·빗나감·불확실은 무신호), 배리어(100 ms 마감 또는 활성 전원 대기).
 - **실제 M0b 측정**(CPU 프로세스 풀, 실제 커넥톰): 16워커 엔드투엔드 예산 42.2시간(학습 결정 52,000 + 평가 결정 81,600; 학습만 19.3시간) ≤ 60시간 게이트 **통과**. 풀로 재실행한 M0 조건화 40건(5팔 × 8시드)이 `results/m0/conditioning.json`과 비트 동일, 희소성·기저도 차이 0.0, 풀의 결정이 인프로세스 엔진과 동일. 워커 RSS 0.57 GB(최대 0.79), C-shuf 배선 변형 하나 +0.15 GB. 휴지 3초에서 100 Hz 초과 뉴런 156개 중 KC 17개, 스파이크의 29% — 스펙 A.5의 STD 재검토 조건 충족. 요약 `results/summary/m0b.json`, 자세한 것은 스펙 부록 C.7.
 - **우리가 정한 것(M0b)**: MPS 배치 엔진 대신 기존 CPU 엔진의 프로세스 풀(레드팀 실측, 스펙 C.6). 마리별로 남는 상태는 KC→MBON 가중치·켬/끔·배선 변형뿐이고 부모가 보관한다. 결정은 후보를 같은 시드로 순차 제시(짝지은 잡음), 강화는 M0 `train_block`의 한 프레젠테이션. 등가성 게이트는 통계 일치가 아니라 비트 동일 재현. STD는 여전히 보류(A.5).
+- **우리가 정한 것(M0c, KC→KC 제거)**: KC→KC 엣지는 기본 제거한다(`kc_kc_scale` 0.0). M0b가 잰 "휴지 100 Hz 초과 집합의 KC 17개"는 시드 평균(0 / 52 / 0)이었고,
+  실체는 KCab-p 우반구 62개의 KC→KC 재귀 흥분 clique였다(냄새 B가 64시드 중 54개에서 점화, 판독 core에 직접 시냅스 0). KC 축삭간 접촉은
+  mAChR-B 매개 억제라(Manoim et al. 2022) 빠른 흥분으로 두는 것이 근거 없는 선택이었다. 이는 lLN1/lLN2와 같은 급의 데이터 라벨 재정의다.
+  `kc_kc_scale=1.0`은 M0·M0b 엔진이며 `results/m0/`·`results/summary/m0.json`·`m0b.json`은 옛 엔진의 기록으로 불변이다. 게이트는 실행 전에
+  사전 등록했고(스펙 부록 D.4: 희소성 ∧ 기저 8시드 ∧ 폭주 ∧ 등가성 ∧ 처리량; 조건화 기준은 그대로 두고 시드 8–15로 판정·기록), 파라미터는
+  게이트 결과를 보고 조정하지 않는다. STD는 재보류(재검토 조건은 D.6의 관측치). 알려진 한계: PN 입력이 0인 KC 297개(αβp·γd)의 역치 정규화는
+  정의되지 않아 클립 하한을 받는다(KC→KC 제거 후 실질 영향 없음).
 - **안 된 것(운영)**: 실패한 튜닝 실행은 스크립트를 `--out results/m0/<태그>.json`으로 다시 돌려 보관한다.
   `results/`는 `results/summary/`를 빼고 git에서 제외되며, 채택한 실행만 `results/summary/m0.json`에 요약된다.
 
@@ -106,8 +113,8 @@ M0: 엔진과 flybrain 측정값 재현. M1: Showdown 배틀 환경과 뇌 없
 
     uv sync
     uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
-    uv run python scripts/reproduce_flybrain_measurements.py sparsity
-    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8
+    uv run python scripts/reproduce_flybrain_measurements.py sparsity --kc-kc-scale 1.0                # M0 = 옛 엔진; 기본값(0.0)으로는 results/m0/ 쓰기를 거부한다
+    uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8 --kc-kc-scale 1.0
     uv run python scripts/write_m0_summary.py
     uv run pytest
 
@@ -147,6 +154,21 @@ PASS / PARTIAL / FAIL 한 줄을 찍고 `gate` 블록에 그대로 기록한다)
 - 결정은 후보를 같은 시드로 순차 제시해 잡음을 짝짓는다(`flymon/brain/presentation.py`). 강화는 M0 `train_block`의 한 프레젠테이션과 같다.
 - MPS 배치 엔진은 스파이크와 레드팀 뒤 채택하지 않았다(스펙 부록 C.6).
 
+### M0c KC→KC 제거와 재게이트
+
+    uv run python scripts/reproduce_flybrain_measurements.py sparsity --kc-thresh 1.5 --apl-scale 0.1 --rest-seeds 8 --out results/m0c/sparsity.json
+    uv run python scripts/bench_pool.py reproduce --kc-kc-scale 1.0 --out results/m0c/reproduce_old.json                       # 옛 엔진 = results/m0 비트 동일(등가성)
+    uv run python scripts/bench_pool.py reproduce --seed-start 8 --rest-seeds 8 --odor-runaway-seeds 64 --arm-equal \
+        --m0-conditioning "" --m0-sparsity results/m0c/sparsity.json --conditioning-out results/m0c/conditioning.json --out results/m0c/reproduce.json
+    uv run python scripts/bench_pool.py reproduce --seed-start 0 --rest-seeds 0 --m0-conditioning "" --m0-sparsity "" \
+        --conditioning-out results/m0c/conditioning_seeds0-7.json --out results/m0c/reproduce_seeds0-7.json                    # 보고용(판정 아님)
+    uv run python scripts/bench_pool.py throughput --out results/m0c/throughput.json
+    uv run python scripts/write_m0c_summary.py                                                                                  # results/summary/m0c.json
+
+- 새 엔진(`Params()` 기본값, `kc_kc_scale` 0.0)의 M0 게이트를 스펙 부록 D.4의 사전 등록대로 다시 잰다. 옛 결과 파일은 덮어쓰지 않는다.
+- `--kc-kc-scale 1.0`은 옛 엔진이다: `reproduce_old.json`의 `conditioning_match`·`sparsity_match`가 `results/m0`와 비트 동일해야 한다.
+- `--arm-equal`은 (seed 8, both)·(seed 15, reversed)를 인프로세스로 다시 돌려 풀 행과 비트 동일한지 본다. 폭주 검사 임계는 휴지 100 Hz, 냄새 창 150 Hz(D.4).
+
 ### 학습 중 보기
 
     uv sync --extra viz
```

- [ ] **Step 2: 스펙 6절의 `scripts/` 줄을 다음으로**

```
scripts/    install_showdown.sh  reproduce_flybrain_measurements.py  write_m0_summary.py  write_m0c_summary.py  bench_pool.py  pilot_no_brain.py  serve_human_challenge.py
```

- [ ] **Step 3: 커밋**

```bash
git add README.md docs/superpowers/specs/2026-09-14-flymon-design.md
git commit -m "docs: README M0c commands and decision entry; spec file list"
```

---

## 컨트롤러 실행 (서브에이전트 금지) — 실제 데이터 게이트, 부록 D.8, README 측정 항목

순서대로. 처리량을 잴 때는 다른 무거운 프로세스를 돌리지 않는다.

0. 백업: `mkdir -p <scratch>/m0-backup && cp results/m0/conditioning.json results/m0/sparsity.json results/summary/m0.json results/summary/m0b.json <scratch>/m0-backup/`; `sha256sum` 기록. 컨트롤러 스크래치 진단 스크립트(`diag_*.py`, `m0_no_kckc.py`)를 `results/m0c/scratch/`에 복사한다(D.2 표의 출처, HEAD 재현 불가).
1. `uv run python scripts/reproduce_flybrain_measurements.py sparsity --kc-thresh 1.5 --apl-scale 0.1 --rest-seeds 8 --out results/m0c/sparsity.json` (약 3분)
2. `uv run python scripts/bench_pool.py reproduce --kc-kc-scale 1.0 --out results/m0c/reproduce_old.json` (약 7분) — 기대: `conditioning_exact true`(40/40), `sparsity_exact true`, `decide_equal true`.
3. `uv run python scripts/bench_pool.py reproduce --seed-start 8 --sparsity-seeds 3 --rest-seeds 8 --odor-runaway-seeds 64 --arm-equal --m0-conditioning "" --m0-sparsity results/m0c/sparsity.json --conditioning-out results/m0c/conditioning.json --out results/m0c/reproduce.json` (약 12분) — 기대: `arm_equal true`, `decide_equal true`, `sparsity_exact true`(풀 희소성 3시드 = 인프로세스 3시드, 기저 8시드 = 8시드), 폭주 `seeds_with_kc_over_sat 0`.
4. `uv run python scripts/bench_pool.py reproduce --seed-start 0 --rest-seeds 0 --m0-conditioning "" --m0-sparsity "" --conditioning-out results/m0c/conditioning_seeds0-7.json --out results/m0c/reproduce_seeds0-7.json` (약 7분)
5. `uv run python scripts/bench_pool.py throughput --out results/m0c/throughput.json` (약 4분)
6. `uv run python scripts/write_m0c_summary.py` → `results/summary/m0c.json`, 한 줄 판정(입력 정체·표본 수가 어긋나면 거부하고 부족 항목을 찍는다 — 그때는 해당 단계를 다시 돈다). `uv run pytest -q tests/test_summary.py tests/brain/test_measure.py` 확인.
7. `results/m0c/gate_runs.md`: D.2의 변형 비교 표(KC→KC 제거, 클립 하한 1.0, 전 KC 총 흥분 정규화, PN 없는 KC 폴백, 스케일 0.5/0.25, 반구 보정 끔)에 **출처 열**("스크래치 조사 2026-09-15, `results/m0c/scratch/`, HEAD에서 재현 불가")을 붙여 남긴다. 저장소에는 `results/`가 제외되므로 표의 사본을 스펙 D.8에 넣는다.
8. 스펙 **D.8 M0c 결과**: 게이트 표(항목별 예측 대 결과), 기저 8시드 평균 ± 표준편차와 시드별 값, 폭주(휴지 8시드; 냄새 B 64시드: 150 Hz 초과 0/64인지, **100–150 Hz 구간 KC 수의 분포 표와 그 KC들의 정체(PN 구동 vs KCab-p)**, 최대 KC 발화율), 조건화 시드 8–15 n_flip·팔별 mean dD·채널별 특이 억제와 시드 0–7 보고, 등가성 셋, 처리량 표와 예산(옛 42.2시간 대비; 메모리는 1.0에서 마스크를 만들지 않으므로 옛 값과 비교 가능), 최종 커밋 해시. 판정 한 줄.
9. README "측정된 것" M0c 항목(수치), M0b 항목 끝의 "STD 재검토 조건 충족" 문구 뒤에 "→ M0c에서 해소(스펙 D)" 추가.
10. **FAIL 분기**(D.4/D.5): 기저가 3–4 Hz 밖이면 `tests/brain/test_measure.py::test_real_m0c_sparsity_gate_recorded`를 `@pytest.mark.xfail(strict=True, reason=…)`로 표시(삭제·완화 없음)하고 파라미터는 조정하지 않는다. 폭주·등가성·처리량이 예측(PASS)을 벗어나면 `test_m0c_summary_records_the_pre_registered_gate`가 빨간불이 되는데, 이것도 같은 방식으로 strict xfail + D.8 기록이다. 어느 경우든 기본값 0.0은 유지하고 D.8·README·`m0c.json`에 FAIL을 기록한 뒤 사용자 판정을 받는다; 되돌린다면 `kc_kc_scale` 기본값을 1.0으로 바꾸는 한 커밋이다. 조건화 기준은 예측대로 FAIL이어도 M0c 판정과 무관하다. 컨트롤러 2단계의 `decide_equal`(옛 엔진)이 참이 아니면 편차로 보고한다.
11. 문서 커밋 뒤 `write_m0c_summary.py`를 한 번 더 돌려 `git_commit`·`git_dirty=false`가 최종 트리를 가리키게 하고(입력 sha256은 변하지 않는다) 커밋한다. `docs/handoffs/2026-09-15-m2-handoff.md`의 STD 항목을 D 기준으로 갱신하고 M2 첫 측정 목록(D.6)을 적는다.

## 자체 검토

- 스펙 커버리지: D.3(파라미터·제거·1.0 등가) → Task 1; D.4 폭주 정의(휴지 100 / 냄새 창 150, 64시드)·기저 8시드·조건화 시드 8–15·등가성 세 가지·처리량 → Task 2·3과 컨트롤러 1–6; D.5(파일 경로·불변·grid 키·테스트) → Task 3·4와 컨트롤러 7–9; D.6·D.7은 문서(컨트롤러 8·11). 5절 M0c 항목의 게이트 합성은 `m0c_gate`.
- 레드팀 반영: P0 옛 경로 가드(Task 2 함수, Task 3 호출·README M0 명령, 컨트롤러 0), P0 입력 검증·표본 수(Task 2 `m0c_gate`, Task 3 `validate_inputs`와 거부 테스트 8건), P1 `--sparsity-seeds`(Task 3, 컨트롤러 3), P1 실제 데이터 테스트 3종(Task 1 엣지 수, Task 3 희소성 게이트·요약 단언), P1 출처(컨트롤러 0·7), P2 FAIL 분기(컨트롤러 10), P2 옛 행 선택(Task 3 `write_m0_summary.py`·`test_measure.py`, `bench_pool.py summary`), P2 git dirty·sha256(Task 3, 컨트롤러 11), P3 100–150 Hz 표(컨트롤러 8), 1.0 마스크 생략·`isfinite`(Task 1).
- 플레이스홀더 없음. 타입 일관성: `build_csc`의 네 번째 인자 이름 `kc_idx`(Task 1) = 엔진·테스트 호출부; `odor_runaway_job` 반환 키 = `bench_pool.py`의 `odor_runaway` 집계 = `write_m0c_summary.compose`의 `runaway` 블록 = `m0c_gate`의 `runaway` 키(`rest_n_kc_over_sat_per_seed`, `odor_B_n_kc_over_sat_per_seed`); `compose`의 `equivalence` 키(`old_conditioning`, `old_sparsity`, `arm_equal`, `decide_equal`) = `m0c_gate`.
