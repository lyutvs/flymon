# FlyMon M0b — CPU 프로세스 풀 스웜 구현 계획 (프레젠테이션 프로토콜, 워커 풀, 짝지은 잡음, hive, 마리별 팔 플래그, 정확 재현, 처리량과 60시간 게이트)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M0의 CPU LIF 엔진을 워커 프로세스 풀로 돌려 마리 F개의 결정·강화를 병렬로 처리하는 스웜을 만들고, 풀의 결정이 인프로세스 엔진과 카운트가 같고 M0 조건화·희소성이 풀에서 비트 동일하게 재현됨을 보이며, 처리량을 재어 학습 결정 52,000회와 평가 결정 81,600회로 외삽한 예산이 60시간 이하임을 기록한다.

**Architecture:** `presentation.py`가 한 마리의 결정(후보 ≤4개를 같은 시드로 순차 제시, 가소성 끔)과 강화(안정화 → DAN 펄스 → 간격 → 회복)를 정의한다. `fly_pool.py`의 `FlyPool`은 `multiprocessing` spawn 풀의 워커 W개에 각각 `Engine`+`Plasticity`(배선 변형별)를 한 번 올려 두고, 부모가 마리별 KC→MBON 가중치(hive)와 켬/끔 플래그를 들고 요청마다 가중치를 실어 보낸다. `pool_jobs.py`는 워커에서 도는 측정·프로토콜 함수(처리량, M0 조건화 팔, 희소성, 기저·폭주 집합)이고, `pool_bench.py`가 예산 공식·정확 일치 검사·게이트를, `scripts/bench_pool.py`가 CLI를 맡아 `results/summary/m0b.json`을 쓴다. C-shuf는 `connectome.shuffle_kc_mbon`으로 KC→MBON 배선을 섞은 커넥톰 변형이다.

**Tech Stack:** Python 3.13(uv), numpy, multiprocessing(spawn), pytest. 새 의존성 없음. 실제 데이터 `data/malecns.npz`(git 제외, 없으면 skip).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` (v4 + 부록 A·B·C). 이 계획은 2절의 3·5단계(결정·강화 프레젠테이션; 배치 배리어는 M1이 만들었고 M3가 `decide_batch`에 얹는다), 3.1의 프로세스 풀 스웜 항목, 3.6, 5절 M0b, 8절 통계, 부록 C.6을 구현한다. 부록 C.1–C.5의 MPS 설계는 레드팀 뒤 철회됐다(C.6). STD는 보류 상태이며 재검토 조건이 A.5에 있다.

## Global Constraints

- Python 3.13, uv. **새 의존성 추가 금지**(torch 없음). numpy 2.x.
- 결정: 후보를 같은 시드로 `reset`하며 순차 제시(짝지은 잡음), 가소성 끔, 가중치 불변. 강화: `conditioning.train_block`의 한 프레젠테이션과 같은 순서(안정화 중 가소성 끔 → 마리 플래그대로 켬 → DAN 구동 → 펄스 → 간격 → `recover_pulse`). DAN을 구동할지는 호출자가 정한다(코치 턴은 호출 자체가 없다, 스펙 4.3 안전 제약).
- 풀의 워커 함수는 전부 모듈 수준(spawn이 피클한다). 워커의 가중치는 스크래치이며 요청이 실어 온 값으로 덮인다. 부모가 마리별 가중치·플래그의 유일한 보관자다.
- 실제 데이터 테스트는 `data/malecns.npz`가 없으면 skip. 스크립트는 기본 경로가 없으면 `SKIP:` 한 줄과 종료 코드 2. **서브에이전트는 `results/` 아래에 쓰지 않고 실제 데이터 벤치·재현을 돌리지 않는다**(컨트롤러 실행, 마지막 절). 스모크는 `--out <tmp>`로만.
- 커밋은 태스크마다. 메시지 `feat(brain): …`, `test(brain): …`, `refactor(brain): …`, `docs: …`.
- 계획의 코드는 세션 스크래치에서 **실행해 검증한 것**이다(합성 망 테스트 20개, 실제 커넥톰에서 M0 조건화 시드 0의 5팔 비트 동일 재현·희소성 3시드 정확 일치·decide 동일·엔드투엔드 배치 타이밍·변형 RSS). 계획 레드팀(Codex + 호스트)의 P1 4건이 반영돼 있다. 실행 중 편차가 생기면 보고하고 판정을 받는다.

---

## 검증된 API 사실 (2026-09-15, Python 3.13.13 / numpy 2 / Apple M5 Pro 6P+12E 코어, 통합 메모리 48 GB)

- `multiprocessing.get_context("spawn").Pool(W, initializer=…, initargs=…)`로 워커마다 커넥톰과 엔진을 한 번 올리는 패턴은 M0의 `reproduce_flybrain_measurements.py conditioning`이 이미 쓴다. `map_async(fn, items, chunksize=1).get(timeout)`은 워커의 예외를 부모에서 같은 타입으로 다시 던지고 풀은 계속 살아 있다(초안 테스트로 확인). `Params`(frozen dataclass)와 numpy 배열은 그대로 피클된다.
- 실제 커넥톰 16워커 동시 실행: 결정 단계(냄새 켬, 가소성 끔) 1.87–1.89 ms/step, 강화 단계(가소성 켬, PAM08 구동) 2.59 ms/step(최대 3.01), 워커당 최고 RSS 0.75 GB. 8워커 1.35–1.43 ms/step. 1워커 1.19 ms/step.
- 풀로 돌린 M0 조건화 시드 0(5팔, 12 trial, settle 800)은 `results/m0/conditioning.json`의 `per_seed["0"]`과 카운트·지수·가중치 비율까지 **비트 동일**했다. 희소성 시드 100–102의 평균(`frac_active_A/B`, `jaccard`, `chance`, `mbon_hz_A/B`)과 3초 휴지 절사 기저 평균은 `results/m0/sparsity.json`의 기본값 격자 행과 차이 0.0. 풀의 `decide`는 인프로세스와 카운트 동일. 5워커로 이 전부에 110초.
- 엔드투엔드(부모 쪽 벽시계, 4워커): `decide_batch`(마리 4 × 후보 4 × 안정화 800 + 읽기 600) 9.4초, `reinforce_batch`(마리 4 × 800+600+200) 2.9초. 워커 RSS 0.78 GB(최대 0.84), C-shuf 배선 변형 하나를 더 올리면 워커당 +0.115 GB. `max_variants=4`면 워커당 최대 1.3 GB, 16워커 21 GB.
- 휴지 3초(시드 100–102 평균)에서 100 Hz 초과 뉴런 156개 중 **KC 17.3개**, 전체 스파이크의 29%가 이 집합에서 난다. 부록 A.5의 STD 재검토 조건(휴지 폭주 집합에 KC 포함)이 **충족**된다 — 컨트롤러가 요약과 핸드오프에 적는다.
- `Connectome`은 `@dataclass`라 `dataclasses.replace(conn, pre=…)`로 배열 하나만 바꾼 사본을 만들 수 있다. `Connectome.save/load`는 합성 커넥톰도 왕복한다(기존 `test_roundtrip_npz`).

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `flymon/brain/presentation.py` | `decide(engine, pl, pops, candidates, strength, seed, settle_ms, read_ms, idx)`, `reinforce(engine, pl, pops, odor, strength, dan, pulse_ms, seed, settle_ms, gap_ms, enabled)` |
| `flymon/brain/connectome.py` | `shuffle_kc_mbon(conn, kc, mbon, seed)` 추가 |
| `flymon/brain/fly_pool.py` | `FlySpec`, `FlyPool`(spawn 풀, 워커 상태, `decide_batch`, `reinforce_batch`, `run_jobs`, `state`/`load_state`) |
| `flymon/brain/pool_jobs.py` | 워커 함수: `phase_timing_job`, `conditioning_arm_job`, `sparsity_job`, `baseline_job` |
| `flymon/brain/conditioning.py` | `channel_specific_seeds()` 이동(scripts와 테스트의 중복 제거) |
| `flymon/brain/pool_bench.py` | `throughput_row`, `budget_hours`, `budget_table`, `exact_match`, `m0b_gate` |
| `scripts/bench_pool.py` | CLI `throughput` / `reproduce` / `summary` |
| `scripts/write_m0_summary.py` | `channel_specific_seeds`를 import로 교체 |
| `tests/conftest.py` | `synthetic_npz` 픽스처(모듈 범위) |
| `tests/brain/test_presentation.py`, `test_shuffle_kc_mbon.py`, `test_fly_pool.py`, `test_pool_bench.py` | 태스크별 테스트 |
| `tests/brain/test_conditioning.py` | 중복 헬퍼를 import로 교체 |
| `tests/test_summary.py` | `results/summary/m0b.json` 검사(있을 때만) |
| `README.md`, 스펙 6절 | M0b 실행 절, 파일 목록에 `pool_jobs.py` |

---
### Task 1: 프레젠테이션 프로토콜과 KC→MBON 셔플

**Files:**
- Create: `flymon/brain/presentation.py`
- Modify: `flymon/brain/connectome.py`(import 줄과 `shuffle_kc_mbon` 추가)
- Test: `tests/brain/test_presentation.py`, `tests/brain/test_shuffle_kc_mbon.py`

**Interfaces:**
- Consumes `engine_cpu.Engine`(`reset`, `clear_drive`, `run`, `p.dan_drive_mv`), `plasticity.Plasticity`(`enabled`, `set_enabled`, `reset_traces`, `quiet_dan`, `drive_dan`, `recover_pulse`, `types`), `stimuli.present`.
- Produces `presentation.decide(engine, pl, pops, candidates, strength, seed, settle_ms=800.0, read_ms=600.0, idx=None) -> np.ndarray[n_candidates, N or len(idx)] int32`: 후보마다 `reset(seed)` → `present` → settle → read 카운트. 가소성 끔, 끝나면 `enabled` 복원. 빈 후보는 `ValueError`.
- Produces `presentation.reinforce(engine, pl, pops, odor, strength, dan: str | None, pulse_ms, seed, settle_ms=800.0, gap_ms=200.0, enabled=True) -> None`. 모르는 DAN 타입은 `ValueError`. 끝난 뒤 `pl.enabled == enabled`.
- Produces `connectome.shuffle_kc_mbon(conn, kc, mbon, seed) -> Connectome`: KC→MBON 엣지의 `pre`만 KC 순열로 바꾼 새 `Connectome`(다른 배열은 공유). 엣지 수·`post`·`w` 불변, KC별 MBON 출력 가중치 합은 순열. KC→MBON 엣지가 없으면 `ValueError`.

- [ ] **Step 1: 실패하는 테스트** — `tests/brain/test_presentation.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide, reinforce

A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}


def _setup(synthetic_connectome, **kw):
    c = synthetic_connectome(disjoint_kc=True)
    p = Params(**{"noise_mv": 0.15, "min_weight": 1, "balance_hemispheres": False, "kc_thresh": 0.5, "learn_rate": 0.05, **kw})
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=0)
    pl = Plasticity(eng, pops, compartments(c, pops, p.core_frac))
    return c, pops, eng, pl


def test_decide_pairs_noise_across_candidates(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    counts = decide(eng, pl, pops, [A, A, B], 1.0, seed=5, settle_ms=50, read_ms=300)
    assert counts.shape == (3, c.N)
    np.testing.assert_array_equal(counts[0], counts[1])          # same odour, same seed -> identical spikes
    assert (counts[2] != counts[0]).any()                         # a different odour differs
    assert counts[:, pops.kc].sum() > 0


def test_decide_is_deterministic_and_leaves_weights_and_enable_flag(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    pl.drive_dan("PAM08", 70.0)                                    # a stale drive must not leak into a decision
    a = decide(eng, pl, pops, [A, B], 1.0, seed=3, settle_ms=50, read_ms=300)
    b = decide(eng, pl, pops, [A, B], 1.0, seed=3, settle_ms=50, read_ms=300)
    np.testing.assert_array_equal(a, b)
    assert pl.weights_frac() == pytest.approx(1.0)
    assert pl.enabled is True
    pl.set_enabled(False)
    decide(eng, pl, pops, [A], 1.0, seed=3, settle_ms=10, read_ms=10)
    assert pl.enabled is False


def test_decide_idx_subset_and_empty(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    counts = decide(eng, pl, pops, [A, B], 1.0, seed=1, settle_ms=10, read_ms=50, idx=pops.mbon)
    assert counts.shape == (2, len(pops.mbon))
    with pytest.raises(ValueError, match="at least one candidate"):
        decide(eng, pl, pops, [], 1.0, seed=1)


def test_reinforce_depresses_only_with_dan_and_when_enabled(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=300, seed=1, settle_ms=50, gap_ms=20)
    assert pl.weights_frac() < 1.0
    pl.reset_weights()
    reinforce(eng, pl, pops, A, 1.0, None, pulse_ms=300, seed=1, settle_ms=50, gap_ms=20)
    assert pl.weights_frac() == pytest.approx(1.0)
    reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=300, seed=1, settle_ms=50, gap_ms=20, enabled=False)
    assert pl.weights_frac() == pytest.approx(1.0)
    assert pl.enabled is False


def test_reinforce_recovery_and_unknown_dan(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, recovery_per_pulse=1.0)
    reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=300, seed=1, settle_ms=50, gap_ms=20)
    assert pl.weights_frac() == pytest.approx(1.0)               # full recovery after the pulse
    with pytest.raises(ValueError, match="unknown DAN"):
        reinforce(eng, pl, pops, A, 1.0, "PAM99", pulse_ms=10, seed=1)


def test_reinforce_cleans_up_when_the_pulse_fails(synthetic_connectome, monkeypatch):
    c, pops, eng, pl = _setup(synthetic_connectome)
    cells = pl.types["PAM08"][0]
    calls = {"n": 0}
    real_run = eng.run

    def failing_run(ms, count_idx=None):
        calls["n"] += 1
        if calls["n"] == 2:                     # the pulse window (1 = settle)
            raise RuntimeError("boom")
        return real_run(ms, count_idx)

    monkeypatch.setattr(eng, "run", failing_run)
    with pytest.raises(RuntimeError, match="boom"):
        reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=100, seed=1, settle_ms=20, gap_ms=10)
    np.testing.assert_array_equal(eng.ext[cells], eng.ext0[cells])     # DAN quiet again
    assert eng.drive_hz.max() == 0.0                                    # odour off
    assert pl.weights_frac() == pytest.approx(1.0)
```

`tests/brain/test_shuffle_kc_mbon.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.connectome import shuffle_kc_mbon


def test_shuffle_permutes_kc_side_of_kc_mbon_edges_only(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    s = shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3)
    assert s.E == c.E and s.N == c.N
    np.testing.assert_array_equal(s.post, c.post)
    np.testing.assert_array_equal(s.w, c.w)
    m = np.isin(c.pre, pops.kc) & np.isin(c.post, pops.mbon)
    np.testing.assert_array_equal(s.pre[~m], c.pre[~m])            # every other edge untouched
    assert (s.pre[m] != c.pre[m]).any() and np.isin(s.pre[m], pops.kc).all()
    out_before = np.bincount(c.pre[m], weights=c.w[m], minlength=c.N)[pops.kc]
    out_after = np.bincount(s.pre[m], weights=s.w[m], minlength=c.N)[pops.kc]
    np.testing.assert_array_equal(np.sort(out_before), np.sort(out_after))   # KC out-weights are permuted, not changed
    np.testing.assert_array_equal(shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=3).pre, s.pre)
    assert (shuffle_kc_mbon(c, pops.kc, pops.mbon, seed=4).pre != s.pre).any()


def test_shuffle_rejects_connectome_without_kc_mbon_edges(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    with pytest.raises(ValueError, match="no KC->MBON edges"):
        shuffle_kc_mbon(c, pops.kc, pops.mbon[:0], seed=1)
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/brain/test_presentation.py tests/brain/test_shuffle_kc_mbon.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.presentation'` / `ImportError: cannot import name 'shuffle_kc_mbon'`.

- [ ] **Step 3: 구현** — `flymon/brain/presentation.py`

```python
"""Decision and reinforcement presentations for one fly on one CPU engine (spec 2, steps 3 and 5).

decide: every candidate odour is presented from the same reset seed, so the candidates see identical
membrane noise and receptor spike trains (paired noise) and differ only through the odour; plasticity
is off and the weights are untouched. reinforce: one presentation of conditioning.train_block — settle
with the weights frozen so the dopamine baseline adapts to the odour-evoked DAN level, then the DAN
pulse, the gap, and one recovery step (spec 3.1). Whether a DAN is driven at all is the caller's
decision (spec 3.4 table; C-PAM / C-PPL arms simply never pass that channel).
"""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .plasticity import Plasticity
from .stimuli import present


def _fresh(engine: Engine, pl: Plasticity, pops: Populations, odor, strength: float, seed: int) -> None:
    engine.reset(seed)
    pl.reset_traces()
    engine.clear_drive()
    pl.quiet_dan()
    present(engine, pops, odor, strength)


def decide(engine: Engine, pl: Plasticity, pops: Populations, candidates, strength: float, seed: int,
           settle_ms: float = 800.0, read_ms: float = 600.0, idx=None) -> np.ndarray:
    """Spike counts of the read window for each candidate: [n_candidates, N] or [n_candidates, len(idx)]."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("decide needs at least one candidate odour")
    sel = None if idx is None else np.asarray(idx, np.int64)
    was = pl.enabled
    pl.set_enabled(False)
    try:
        out = []
        for odor in candidates:
            _fresh(engine, pl, pops, odor, strength, seed)
            engine.run(settle_ms)
            c = engine.run(read_ms)
            out.append(c if sel is None else c[sel])
    finally:
        pl.set_enabled(was)
    return np.stack(out)


def reinforce(engine: Engine, pl: Plasticity, pops: Populations, odor, strength: float, dan: str | None,
              pulse_ms: float, seed: int, settle_ms: float = 800.0, gap_ms: float = 200.0, enabled: bool = True) -> None:
    """One reinforcement presentation. `dan` is the DAN type to drive for `pulse_ms` (None = no signal;
    the odour is still presented so the call is uniform). Learning happens only if `enabled`."""
    if dan is not None and dan not in pl.types:
        raise ValueError(f"unknown DAN type {dan!r}; known: {sorted(pl.types)}")
    _fresh(engine, pl, pops, odor, strength, seed)
    pl.set_enabled(False)                       # settle: baseline adapts, weights frozen
    engine.run(settle_ms)
    try:
        pl.set_enabled(bool(enabled))
        if dan is not None:
            pl.drive_dan(dan, engine.p.dan_drive_mv)
        engine.run(pulse_ms)
    finally:                                    # a failing pulse never leaves a DAN driven or an odour on
        pl.quiet_dan()
        engine.clear_drive()
    engine.run(gap_ms)
    if dan is not None:
        pl.recover_pulse()
```

`flymon/brain/connectome.py`: 4행 `from dataclasses import dataclass`를 `from dataclasses import dataclass, replace`로 바꾸고 파일 끝에 추가:

```python
def shuffle_kc_mbon(conn: Connectome, kc: np.ndarray, mbon: np.ndarray, seed: int) -> Connectome:
    """C-shuf control (spec 4.1): permute the presynaptic Kenyon cell of every KC->MBON edge with one
    permutation over the KC population, so the odour code reaches the MBONs through scrambled wiring.
    Edge count, the weight multiset and every MBON's total KC input are preserved; every other edge is
    untouched. Returns a new Connectome (arrays other than `pre` are shared)."""
    is_kc = np.zeros(conn.N, bool); is_kc[kc] = True
    is_mb = np.zeros(conn.N, bool); is_mb[mbon] = True
    m = is_kc[conn.pre] & is_mb[conn.post]
    if not m.any():
        raise ValueError("no KC->MBON edges to shuffle")
    perm = np.random.default_rng(int(seed)).permutation(len(kc))
    mapping = np.arange(conn.N, dtype=conn.pre.dtype)
    mapping[kc] = kc[perm]
    pre = conn.pre.copy()
    pre[m] = mapping[pre[m]]
    return replace(conn, pre=pre)
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/brain/test_presentation.py tests/brain/test_shuffle_kc_mbon.py tests/brain/test_connectome.py -v` — Expected: 8 새 테스트 + 기존 connectome 테스트 PASS.

- [ ] **Step 5: 커밋**

```bash
git add flymon/brain/presentation.py flymon/brain/connectome.py tests/brain/test_presentation.py tests/brain/test_shuffle_kc_mbon.py
git commit -m "feat(brain): decision/reinforcement presentations (paired noise via shared seed) and the C-shuf KC->MBON wiring shuffle"
```

---
### Task 2: `FlyPool` 워커 풀과 워커 함수

**Files:**
- Create: `flymon/brain/fly_pool.py`, `flymon/brain/pool_jobs.py`
- Modify: `tests/conftest.py`(`synthetic_npz` 픽스처)
- Test: `tests/brain/test_fly_pool.py`

**Interfaces:**
- Consumes Task 1의 `decide`/`reinforce`/`shuffle_kc_mbon`, `circuits.{Populations, compartments, validate_populations}`, `conditioning.{Readout, run_arm}`, `measure.{kc_sparsity, jaccard, chance_jaccard}`, `stimuli.{design_odor_pair, present}`.
- Produces `fly_pool.FlySpec(enabled: bool = True, shuffle_seed: int | None = None)`(frozen dataclass).
- Produces `fly_pool.FlyPool(npz, params, flies, workers=16, punish_type="PPL105", reward_type="PAM08", timeout_s=3600.0, max_variants=4)`: 생성자는 **부모에서 먼저** npz를 열어 `validate_populations`를 통과시키고(워커 initializer가 죽으면 `Pool`이 무한 재생성하므로 데이터 오류는 spawn 전에 나야 한다), 배선 변형 수가 `max_variants`를 넘으면 `ValueError`, 풀 생성 뒤 초기화가 실패하면 `terminate()`·`join()` 후 재던진다. 속성 `flies: list[FlySpec]`, `n_flies`, `n_workers`(= min(workers, n_flies)), `w0: dict[variant -> np.ndarray]`(배선 변형별 초기 가중치), `w: dict[fly -> np.ndarray]`(현재 가중치, 길이 = 가소성 엣지 수); 메서드 `decide_batch(requests=[(fly, candidates, seed)], strength, settle_ms=800.0, read_ms=600.0, idx=None) -> list[np.ndarray]`, `reinforce_batch(requests=[(fly, odor, dan_or_None, pulse_ms, seed)], strength, settle_ms=800.0, gap_ms=200.0) -> None`(마리 가중치 갱신; 같은 마리가 두 번 들어오면 `ValueError`), `run_jobs(fn, kwargs_list, shuffle_seed=None) -> list`(`fn(engine, pl, pops, comps, ro, **kwargs)`, 모듈 수준 함수), `set_enabled(fly, on)`, `weights_frac(fly) -> float`, `state() -> {"flies": [{"enabled", "shuffle_seed", "w"}]}`, `load_state(d)`(길이·배선 변형·가중치 형태·유한성·비음수 검사 뒤 전부 적용, 하나라도 틀리면 아무것도 바꾸지 않음), `close()`, `terminate()`, 컨텍스트 매니저(예외 시 terminate).
- Produces `pool_jobs.phase_timing_job(eng, pl, pops, comps, ro, odor, strength, steps, warm, reward_type="PAM08") -> {"ms_decision", "ms_reinforce"}`, `conditioning_arm_job(…, seed, arm, strength=0.35, k=8, odor_seed=0, trials=12, present_ms=800.0, settle_ms=800.0, punish_type, reward_type) -> run_arm 결과`, `sparsity_job(…, seed, strength=0.35, k=8, odor_seed=0) -> {"frac_active_A", "frac_active_B", "jaccard", "chance", "mbon_hz_A", "mbon_hz_B"}`, `baseline_job(…, seed, ms=3000.0, sat_hz=100.0) -> {"mbon_hz", "mbon_hz_trimmed", "n_saturated", "n_types_active", "runaway": {"sat_hz", "n_over_sat", "n_kc_over_sat", "spike_share_over_sat"}}`, `rss_job(…) -> float`(워커 최고 RSS GB; `run_jobs(rss_job, [{}]*W, shuffle_seed=s)`로 변형 하나의 메모리를 잰다).
- Produces `tests/conftest.py`의 `synthetic_npz`(모듈 범위, `_build(disjoint_kc=True).save(tmp)` 경로).

- [ ] **Step 1: 픽스처** — `tests/conftest.py` 끝에 추가:

```python
@pytest.fixture(scope="module")
def synthetic_npz(tmp_path_factory):
    """The disjoint-KC synthetic connectome saved as an npz, for pool workers that load from disk."""
    path = tmp_path_factory.mktemp("conn") / "synthetic.npz"
    _build(disjoint_kc=True).save(path)
    return path
```

- [ ] **Step 2: 실패하는 테스트** — `tests/brain/test_fly_pool.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, shuffle_kc_mbon
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_jobs import phase_timing_job
from flymon.brain.presentation import decide

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
FLIES = [FlySpec(), FlySpec(shuffle_seed=3), FlySpec(enabled=False)]


@pytest.fixture(scope="module")
def pool(synthetic_npz):
    with FlyPool(synthetic_npz, P, FLIES, workers=2, timeout_s=120) as p:
        yield p


def _in_process(npz, shuffle_seed=None):
    c = Connectome.load(npz)
    pops = Populations.from_connectome(c)
    if shuffle_seed is not None:
        c = shuffle_kc_mbon(c, pops.kc, pops.mbon, shuffle_seed)
    eng = Engine(c, pops, P, seed=0)
    pl = Plasticity(eng, pops, compartments(c, pops, P.core_frac))
    return pops, eng, pl


def test_pool_layout(pool):
    assert pool.n_flies == 3 and pool.n_workers == 2
    assert set(pool.w0) == {None, 3} and pool.w[0].shape == pool.w0[None].shape
    assert pool.weights_frac(0) == pytest.approx(1.0)


def test_decide_batch_equals_in_process_engine(pool, synthetic_npz):
    pops, eng, pl = _in_process(synthetic_npz)
    expected = decide(eng, pl, pops, [A, B], 1.0, seed=7, settle_ms=50, read_ms=300)
    got = pool.decide_batch([(0, [A, B], 7), (0, [A], 7)], 1.0, settle_ms=50, read_ms=300)
    np.testing.assert_array_equal(got[0], expected)
    np.testing.assert_array_equal(got[1], expected[:1])
    pops, eng2, pl2 = _in_process(synthetic_npz, shuffle_seed=3)
    expected2 = decide(eng2, pl2, pops, [A, B], 1.0, seed=7, settle_ms=50, read_ms=300)
    got2 = pool.decide_batch([(1, [A, B], 7)], 1.0, settle_ms=50, read_ms=300)[0]
    np.testing.assert_array_equal(got2, expected2)
    assert (expected2[:, pops.mbon] != expected[:, pops.mbon]).any()      # scrambled wiring reaches the MBONs differently
    sub = pool.decide_batch([(0, [A], 7)], 1.0, settle_ms=50, read_ms=300, idx=pops.mbon)[0]
    np.testing.assert_array_equal(sub, expected[:1][:, pops.mbon])


def test_reinforce_batch_isolates_flies_and_respects_enabled(pool, synthetic_npz):
    pops, eng, pl = _in_process(synthetic_npz)
    naive = decide(eng, pl, pops, [A], 1.0, seed=9, settle_ms=50, read_ms=300)
    pool.reinforce_batch([(0, A, "PAM08", 300, 1), (2, A, "PAM08", 300, 1)], 1.0, settle_ms=50, gap_ms=20)
    assert pool.weights_frac(0) < 1.0
    assert pool.weights_frac(2) == pytest.approx(1.0)              # disabled fly: driven but does not learn
    assert pool.weights_frac(1) == pytest.approx(1.0)              # not in the batch
    learned = pool.decide_batch([(0, [A], 9)], 1.0, settle_ms=50, read_ms=300)[0]
    assert (learned[:, pops.mbon] != naive[:, pops.mbon]).any()    # the learned weights travelled with the request


def test_state_roundtrip_and_validation(pool):
    s = pool.state()
    w0_fly0 = pool.w[0].copy()
    pool.w[0][:] = pool.w0[None]
    pool.set_enabled(2, True)
    pool.load_state(s)
    np.testing.assert_array_equal(pool.w[0], w0_fly0)
    assert pool.flies[2].enabled is False
    bad = {"flies": s["flies"][:2]}
    with pytest.raises(ValueError, match="pool has 3"):
        pool.load_state(bad)
    bad = {"flies": [dict(e, shuffle_seed=99) if i == 1 else e for i, e in enumerate(s["flies"])]}
    with pytest.raises(ValueError, match="wiring variant"):
        pool.load_state(bad)
    np.testing.assert_array_equal(pool.w[0], w0_fly0)             # a rejected state changes nothing


def test_run_jobs_and_worker_errors_propagate(pool):
    res = pool.run_jobs(phase_timing_job, [dict(odor=A, strength=1.0, steps=5, warm=2)] * 2)
    assert len(res) == 2 and all(r["ms_decision"] > 0 and r["ms_reinforce"] > 0 for r in res)
    with pytest.raises(ValueError, match="unknown DAN"):
        pool.reinforce_batch([(0, A, "PAM99", 10, 1)], 1.0, settle_ms=5, gap_ms=5)
    assert pool.decide_batch([(0, [A], 1)], 1.0, settle_ms=5, read_ms=5)[0].shape[0] == 1   # still alive


def test_reinforce_batch_rejects_a_fly_listed_twice(pool):
    with pytest.raises(ValueError, match="more than once"):
        pool.reinforce_batch([(0, A, "PAM08", 10, 1), (0, B, "PAM08", 10, 2)], 1.0, settle_ms=5, gap_ms=5)


def test_load_state_rejects_non_finite_or_negative_weights(pool):
    s = pool.state()
    before = pool.w[0].copy()
    bad = {"flies": [dict(e, w=np.where(np.arange(e["w"].size) == 0, np.nan, e["w"])) if i == 0 else e for i, e in enumerate(s["flies"])]}
    with pytest.raises(ValueError, match="finite"):
        pool.load_state(bad)
    bad = {"flies": [dict(e, w=-e["w"]) if i == 0 else e for i, e in enumerate(s["flies"])]}
    with pytest.raises(ValueError, match="finite"):
        pool.load_state(bad)
    np.testing.assert_array_equal(pool.w[0], before)


def test_constructor_validates_before_spawning_and_caps_variants(synthetic_npz, tmp_path):
    import multiprocessing as mp
    children_before = len(mp.active_children())        # the module-scoped pool's workers may be alive
    with pytest.raises(FileNotFoundError):
        FlyPool(tmp_path / "missing.npz", P, [FlySpec()], workers=1, timeout_s=30)
    with pytest.raises(ValueError, match="max_variants"):
        FlyPool(synthetic_npz, P, [FlySpec(shuffle_seed=s) for s in range(5)], workers=1, timeout_s=30, max_variants=4)
    assert len(mp.active_children()) == children_before  # neither failure spawned (and leaked) a worker
```

- [ ] **Step 3: 실패 확인** — Run: `uv run pytest tests/brain/test_fly_pool.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.fly_pool'`.

- [ ] **Step 4: 구현** — `flymon/brain/pool_jobs.py`

```python
"""Functions that run on a FlyPool worker (`FlyPool.run_jobs`): phase timing, the M0 conditioning arm,
the M0 sparsity seed, the resting baseline with the runaway set, and the peak RSS of the worker. Module-level so the pool can pickle them;
signature fn(engine, plasticity, pops, comps, readout, **kwargs); every job leaves the worker's weights reset."""
from __future__ import annotations

import time

from .conditioning import run_arm
from .measure import chance_jaccard, jaccard, kc_sparsity
from .stimuli import design_odor_pair, present


# ---- worker-side jobs (module-level so the pool can pickle them) --------------------------------
def phase_timing_job(eng, pl, pops, comps, ro, odor, strength: float, steps: int, warm: int,
                     reward_type: str = "PAM08") -> dict:
    """ms per step of the decision phase (odour on, plasticity off) and the reinforcement phase
    (plasticity on, the reward DAN driven). Leaves the worker's weights reset."""
    dt = eng.p.dt
    pl.reset_weights()
    pl.set_enabled(False)
    eng.reset(0)
    eng.clear_drive()
    pl.quiet_dan()
    present(eng, pops, odor, strength)
    eng.run(warm * dt)
    t0 = time.perf_counter()
    eng.run(steps * dt)
    ms_dec = (time.perf_counter() - t0) / steps * 1000.0
    pl.set_enabled(True)
    pl.drive_dan(reward_type, eng.p.dan_drive_mv)
    eng.run(warm * dt)
    t0 = time.perf_counter()
    eng.run(steps * dt)
    ms_rein = (time.perf_counter() - t0) / steps * 1000.0
    pl.quiet_dan()
    pl.reset_weights()
    return {"ms_decision": ms_dec, "ms_reinforce": ms_rein}


def conditioning_arm_job(eng, pl, pops, comps, ro, seed: int, arm: str, strength: float = 0.35, k: int = 8,
                         odor_seed: int = 0, trials: int = 12, present_ms: float = 800.0, settle_ms: float = 800.0,
                         punish_type: str = "PPL105", reward_type: str = "PAM08") -> dict:
    """One (seed, arm) of the M0 conditioning, exactly as scripts/reproduce_flybrain_measurements.py runs it."""
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    return run_arm(eng, pl, pops, ro, a, b, strength, seed, arm, trials=trials, present_ms=present_ms,
                   settle_ms=settle_ms, punish_type=punish_type, reward_type=reward_type)


def sparsity_job(eng, pl, pops, comps, ro, seed: int, strength: float = 0.35, k: int = 8, odor_seed: int = 0) -> dict:
    """One seed of the M0 sparsity row (measure.kc_sparsity for both odours), plasticity off."""
    pl.reset_weights()
    pl.set_enabled(False)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    ra = kc_sparsity(eng, pops, a, strength, seed=seed)
    rb = kc_sparsity(eng, pops, b, strength, seed=seed)
    pl.set_enabled(True)
    return {"frac_active_A": ra["frac_active"], "frac_active_B": rb["frac_active"],
            "jaccard": jaccard(ra["active"], rb["active"]), "chance": chance_jaccard(ra["frac_active"], rb["frac_active"]),
            "mbon_hz_A": ra["mbon_hz"], "mbon_hz_B": rb["mbon_hz"]}


def baseline_job(eng, pl, pops, comps, ro, seed: int, ms: float = 3000.0, sat_hz: float = 100.0) -> dict:
    """Resting MBON rates (measure.mbon_baseline) plus the runaway set: neurons above sat_hz, how many
    are Kenyon cells, and their share of all spikes (spec 5 M0b; STD revisit condition, A.5)."""
    pl.reset_weights()
    pl.set_enabled(False)
    eng.reset(seed)
    eng.clear_drive()
    pl.quiet_dan()
    counts = eng.run(ms)
    pl.set_enabled(True)
    sec = ms / 1000.0
    hz = counts[pops.mbon] / sec
    keep = hz <= sat_hz
    over = (counts / sec) > sat_hz
    types = eng.conn.type[pops.mbon]
    return {"mbon_hz": float(hz.mean()), "mbon_hz_trimmed": float(hz[keep].mean()) if keep.any() else 0.0,
            "n_saturated": int((~keep).sum()), "n_types_active": int(len(set(types[hz > 0].tolist()))),
            "runaway": {"sat_hz": sat_hz, "n_over_sat": int(over.sum()), "n_kc_over_sat": int(over[pops.kc].sum()),
                        "spike_share_over_sat": float(counts[over].sum() / max(int(counts.sum()), 1))}}


def rss_job(eng, pl, pops, comps, ro) -> float:
    """Peak resident set size of this worker process in GB; the engine for the requested wiring variant
    is built before the job runs, so calling it before and after a new variant measures that variant."""
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9      # macOS reports bytes, Linux kB
```

`flymon/brain/fly_pool.py`

```python
"""A process pool of CPU fly brains (spec 3.1, M0b).

W worker processes each hold one Engine + Plasticity per wiring variant (unshuffled, or one per C-shuf
seed) and serve requests for any fly. Membrane state is reset at every presentation, so a fly is only
its plastic KC->MBON weights, its enabled flag and its wiring variant; the parent keeps those (hive =
one weight vector per fly) and ships the weights with each request. Every worker function is
module-level so the spawn start method can pickle it.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp

import numpy as np

from .circuits import Populations, compartments, validate_populations
from .conditioning import Readout
from .config import Params
from .connectome import Connectome, shuffle_kc_mbon
from .engine_cpu import Engine
from .plasticity import Plasticity
from .presentation import decide, reinforce


@dataclasses.dataclass(frozen=True)
class FlySpec:
    enabled: bool = True              # plasticity on (C-off: False)
    shuffle_seed: int | None = None   # C-shuf wiring variant


_W = None   # per-process worker state, set by _init_worker


class _WorkerState:
    def __init__(self, npz: str, params: Params, punish_type: str, reward_type: str, max_variants: int):
        self.conn = Connectome.load(npz)
        self.pops = Populations.from_connectome(self.conn)
        self.params, self.punish_type, self.reward_type = params, punish_type, reward_type
        self.max_variants = int(max_variants)
        self.variants: dict = {}

    def get(self, shuffle_seed):
        key = None if shuffle_seed is None else int(shuffle_seed)
        if key not in self.variants:
            if len(self.variants) >= self.max_variants:
                raise ValueError(f"worker already holds {len(self.variants)} wiring variants (max_variants={self.max_variants}); "
                                 f"refusing to build variant {key!r} — each costs a CSC and an engine")
            conn = self.conn if key is None else shuffle_kc_mbon(self.conn, self.pops.kc, self.pops.mbon, key)
            comps = compartments(conn, self.pops, self.params.core_frac)
            validate_populations(conn, self.pops, comps, self.punish_type, self.reward_type)
            eng = Engine(conn, self.pops, self.params, seed=0)
            pl = Plasticity(eng, self.pops, comps)
            ro = Readout.from_compartments(comps, self.punish_type, self.reward_type)
            self.variants[key] = (eng, pl, comps, ro)
        return self.variants[key]


def _init_worker(npz: str, params: Params, punish_type: str, reward_type: str, max_variants: int) -> None:
    global _W
    _W = _WorkerState(npz, params, punish_type, reward_type, max_variants)


def _w0_job(shuffle_seed):
    _, pl, _, _ = _W.get(shuffle_seed)
    return pl.w0.copy()


def _decide_job(req: dict) -> np.ndarray:
    eng, pl, _, _ = _W.get(req["shuffle_seed"])
    eng.csc.w[pl.edges] = req["w"]
    return decide(eng, pl, _W.pops, req["candidates"], req["strength"], req["seed"],
                  req["settle_ms"], req["read_ms"], req["idx"])


def _reinforce_job(req: dict) -> np.ndarray:
    eng, pl, _, _ = _W.get(req["shuffle_seed"])
    eng.csc.w[pl.edges] = req["w"]
    reinforce(eng, pl, _W.pops, req["odor"], req["strength"], req["dan"], req["pulse_ms"], req["seed"],
              req["settle_ms"], req["gap_ms"], req["enabled"])
    return eng.csc.w[pl.edges].copy()


def _job(args):
    fn, shuffle_seed, kw = args
    eng, pl, comps, ro = _W.get(shuffle_seed)
    return fn(eng, pl, _W.pops, comps, ro, **kw)


class FlyPool:
    def __init__(self, npz, params: Params, flies, workers: int = 16, punish_type: str = "PPL105",
                 reward_type: str = "PAM08", timeout_s: float = 3600.0, max_variants: int = 4):
        self.flies = [f if isinstance(f, FlySpec) else FlySpec(**f) for f in flies]
        if not self.flies:
            raise ValueError("FlyPool needs at least one fly")
        variants = sorted({f.shuffle_seed for f in self.flies}, key=lambda v: (v is not None, v if v is not None else 0))
        if len(variants) > max_variants:
            raise ValueError(f"{len(variants)} wiring variants requested but max_variants={max_variants}")
        # Validate the data in the parent first: a worker that dies in its initializer is respawned by
        # multiprocessing.Pool forever, so every data error has to surface here, before any spawn.
        conn = Connectome.load(str(npz))
        pops = Populations.from_connectome(conn)
        validate_populations(conn, pops, compartments(conn, pops, params.core_frac), punish_type, reward_type)
        del conn, pops
        self.timeout_s = float(timeout_s)
        self.n_workers = max(1, min(int(workers), len(self.flies)))
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.n_workers, initializer=_init_worker,
                             initargs=(str(npz), params, punish_type, reward_type, int(max_variants)))
        try:
            self.w0 = dict(zip(variants, self._map(_w0_job, variants)))
        except BaseException:
            self.pool.terminate()
            self.pool.join()
            raise
        self.w = {i: self.w0[f.shuffle_seed].copy() for i, f in enumerate(self.flies)}

    # ---- plumbing --------------------------------------------------------------------------
    def _map(self, fn, items):
        if not items:
            return []
        return self.pool.map_async(fn, items, chunksize=1).get(self.timeout_s)

    @property
    def n_flies(self) -> int:
        return len(self.flies)

    # ---- the two presentations (spec 2 steps 3 and 5) ----------------------------------------
    def decide_batch(self, requests, strength: float, settle_ms: float = 800.0, read_ms: float = 600.0, idx=None):
        """requests: [(fly, candidates, seed)] -> [counts[n_candidates, N or len(idx)]] in request order."""
        sel = None if idx is None else np.asarray(idx, np.int64)
        jobs = [dict(w=self.w[f], shuffle_seed=self.flies[f].shuffle_seed, candidates=list(c), strength=strength,
                     seed=int(s), settle_ms=settle_ms, read_ms=read_ms, idx=sel) for f, c, s in requests]
        return self._map(_decide_job, jobs)

    def reinforce_batch(self, requests, strength: float, settle_ms: float = 800.0, gap_ms: float = 200.0) -> None:
        """requests: [(fly, odor, dan_or_None, pulse_ms, seed)]; updates the weights of every listed fly.
        A fly may appear once per batch: two reinforcements of one fly would both start from the same
        weights and the second would silently overwrite the first."""
        ids = [int(f) for f, *_ in requests]
        dup = sorted({f for f in ids if ids.count(f) > 1})
        if dup:
            raise ValueError(f"reinforce_batch: fly ids {dup} appear more than once in one batch")
        jobs = [dict(w=self.w[f], shuffle_seed=self.flies[f].shuffle_seed, odor=o, strength=strength, dan=d,
                     pulse_ms=float(pm), seed=int(s), settle_ms=settle_ms, gap_ms=gap_ms, enabled=self.flies[f].enabled)
                for f, o, d, pm, s in requests]
        for (f, *_), w in zip(requests, self._map(_reinforce_job, jobs)):
            self.w[f] = w

    # ---- measurements and protocols on the workers --------------------------------------------
    def run_jobs(self, fn, kwargs_list, shuffle_seed=None):
        """fn(engine, plasticity, pops, comps, readout, **kwargs) on any worker; fn must be module-level.
        Results in order. The worker's weights are scratch here: reset them inside fn if it learns."""
        return self._map(_job, [(fn, shuffle_seed, dict(kw)) for kw in kwargs_list])

    # ---- per-fly state -----------------------------------------------------------------------
    def set_enabled(self, fly: int, on: bool) -> None:
        self.flies[fly] = dataclasses.replace(self.flies[fly], enabled=bool(on))

    def weights_frac(self, fly: int) -> float:
        return float(np.mean(self.w[fly] / self.w0[self.flies[fly].shuffle_seed]))

    def state(self) -> dict:
        return {"flies": [dict(enabled=f.enabled, shuffle_seed=f.shuffle_seed, w=self.w[i].copy())
                          for i, f in enumerate(self.flies)]}

    def load_state(self, d: dict) -> None:
        entries = d["flies"]
        if len(entries) != len(self.flies):
            raise ValueError(f"state has {len(entries)} flies, pool has {len(self.flies)}")
        for i, e in enumerate(entries):
            if e["shuffle_seed"] != self.flies[i].shuffle_seed:
                raise ValueError(f"fly {i}: wiring variant {e['shuffle_seed']} != pool's {self.flies[i].shuffle_seed}")
            w = np.asarray(e["w"], np.float32)
            if w.shape != self.w0[self.flies[i].shuffle_seed].shape:
                raise ValueError(f"fly {i}: weight vector shape {w.shape} != {self.w0[self.flies[i].shuffle_seed].shape}")
            if not np.isfinite(w).all() or (w < 0).any():
                raise ValueError(f"fly {i}: weights must be finite and >= 0")
        for i, e in enumerate(entries):
            self.w[i] = np.asarray(e["w"], np.float32).copy()
            self.flies[i] = dataclasses.replace(self.flies[i], enabled=bool(e["enabled"]))

    # ---- lifecycle ---------------------------------------------------------------------------
    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def terminate(self) -> None:
        self.pool.terminate()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        (self.terminate if exc_type else self.close)()
```

- [ ] **Step 5: 통과 확인** — Run: `uv run pytest tests/brain/test_fly_pool.py -v` — Expected: 8 PASS(풀 하나를 모듈 범위로 공유, 수 초). 테스트가 걸리면 워커가 죽은 것이다: `timeout_s=120`이 지나면 `multiprocessing.TimeoutError`가 난다.

- [ ] **Step 6: 전체 스위트** — Run: `uv run pytest -q` — Expected: 전부 PASS, 경고 없음.

- [ ] **Step 7: 커밋**

```bash
git add flymon/brain/fly_pool.py flymon/brain/pool_jobs.py tests/conftest.py tests/brain/test_fly_pool.py
git commit -m "feat(brain): FlyPool — spawn worker pool with per-fly weights (hive), per-fly enable and wiring variants, worker jobs"
```

---
### Task 3: 예산·정확 일치·게이트, `bench_pool.py` CLI, `channel_specific_seeds` 이동

**Files:**
- Create: `flymon/brain/pool_bench.py`, `scripts/bench_pool.py`
- Modify: `flymon/brain/conditioning.py`(`channel_specific_seeds` 추가), `scripts/write_m0_summary.py`(로컬 정의 삭제, import), `tests/brain/test_conditioning.py`(`_channel_specific_seeds` 삭제, import), `tests/test_summary.py`(m0b 검사)
- Test: `tests/brain/test_pool_bench.py`

**Interfaces:**
- Produces `conditioning.channel_specific_seeds(per_seed) -> int`(`scripts/write_m0_summary.py`의 정의 그대로).
- Produces `pool_bench.DECISIONS = 52_000`, `EVAL_DECISIONS = 81_600`, `WINDOWS`, `throughput_row(pool, odor, strength, steps, warm, repeats=3, idx=None, settle_decision_ms=800.0, read_ms=600.0, settle_reinforce_ms=800.0, pulse_ms=600.0, gap_ms=200.0, reward_type="PAM08") -> dict`(워커 안 스텝 시간: `ms_decision_*`, `ms_reinforce_*`, `agg_slot_steps_per_ms`; 부모 쪽 엔드투엔드: `s_decide_batch_*`, `s_reinforce_batch_*`, `n_flies_batch`, `windows`; 배치 학습은 되돌린다), `budget_hours_e2e(s_decide_batch, s_reinforce_batch, n_flies_batch, decisions, eval_decisions) -> float`(게이트 기준), `budget_hours(ms_decision, ms_reinforce, workers, decisions, eval_decisions, n_candidates=4, settle_decision_ms=800.0, read_ms=600.0, settle_reinforce_ms=800.0, pulse_max_ms=600.0, gap_ms=200.0, dt_ms=1.0) -> float`, `budget_table(rows, decisions, eval_decisions, limit_hours=60.0) -> {"workers", "n_flies_batch", "decisions", "eval_decisions", "windows", "by_workers": [{"workers", "hours_e2e", "hours_e2e_training_only", "hours_step_estimate"}], "rows": [{"assumption", "hours"}], "limit_hours", "baseline_hours", "gate_ok"}`(워커 수마다 엔드투엔드 최대값으로 시간을 내고 **가장 빠른 구성**을 게이트에 쓴다; 스텝 기반 추정은 진단용), `exact_match(pool_per_seed, m0_per_seed) -> {"n_results", "n_equal", "n_missing", "missing", "max_abs_diff", "ok"}`, `m0b_gate(budget, conditioning_match, sparsity_match, decide_equal) -> {"throughput_ok", "conditioning_exact_ok", "sparsity_exact_ok", "decide_equal_ok", "passed"}`.
- Produces CLI `scripts/bench_pool.py throughput|reproduce|summary`(기본 경로 `results/m0b/throughput.json`, `results/m0b/reproduce.json`, `results/summary/m0b.json`; 실행은 컨트롤러). `results/summary/m0b.json` 스키마: `params_frozen, throughput[rows], budget, memory{worker_rss_GB, worker_rss_GB_max, variant_rss_GB}, conditioning{n_seeds, n_flip, n_flip_disc, noplast_max_abs_dD, channel_specific_seeds, arms}, conditioning_match, sparsity_match{diffs, max_abs_diff, ok, cpu_row}, decide_equal, runaway{n_over_sat, n_kc_over_sat, spike_share_over_sat}, gate, generated_at`.

- [ ] **Step 1: `channel_specific_seeds` 이동** — `flymon/brain/conditioning.py` 끝에 추가:

```python
def _drop(pre: float, post: float) -> float:
    """Fraction of a naive probe response lost after training (0 when there was nothing to lose)."""
    return 0.0 if pre == 0 else (pre - post) / pre


def channel_specific_seeds(per_seed: dict) -> int:
    """Seeds where both dopamine channels depress the odour they were actually paired with.

    Read off the raw probe counts, not the composite index. Per seed, all three must hold:
      - `both` (reward PAM08 on the CS-): the approach core's minus-odour response drops more
        than its plus-odour response;
      - `reversed` (reward on the CS+): the other way round;
      - `reversed` (punishment PPL105 on the CS-): the aversive core's minus-odour response drops
        more than it does in `both`, where the same channel was paired with the other odour.
    """
    n = 0
    for rec in per_seed.values():
        both, rev = rec["both"]["counts"], rec["reversed"]["counts"]
        p_minus_both = _drop(both["pre_minus"]["P"], both["post_minus"]["P"])
        p_plus_both = _drop(both["pre_plus"]["P"], both["post_plus"]["P"])
        p_plus_rev = _drop(rev["pre_plus"]["P"], rev["post_plus"]["P"])
        p_minus_rev = _drop(rev["pre_minus"]["P"], rev["post_minus"]["P"])
        a_minus_both = _drop(both["pre_minus"]["A"], both["post_minus"]["A"])
        a_minus_rev = _drop(rev["pre_minus"]["A"], rev["post_minus"]["A"])
        if p_minus_both > p_plus_both and p_plus_rev > p_minus_rev and a_minus_rev > a_minus_both:
            n += 1
    return n
```

`scripts/write_m0_summary.py`: 39–66행의 `_drop`과 `channel_specific_seeds` 정의를 지우고, 24행 뒤에 `from flymon.brain.conditioning import channel_specific_seeds`를 더한다. `index_flip_ok`, `gate_ok`, `main`은 그대로.

`tests/brain/test_conditioning.py`: 9–10행 import를
```python
from flymon.brain.conditioning import (D, D_graded, Readout, arms, channel_specific_seeds, disc, disc_graded,
                                       probe, run_arm, train_block)
```
로 바꾸고, `def _channel_specific_seeds(per_seed: dict) -> int:` 정의 전체(docstring 포함, `return n`까지)를 지운 뒤 `test_real_conditioning_channel_specificity`의 호출을 `channel_specific_seeds(d["per_seed"])`로 바꾼다.

Run: `uv run pytest tests/brain/test_conditioning.py tests/test_summary.py -q && uv run python -c "import importlib.util; s=importlib.util.spec_from_file_location('w','scripts/write_m0_summary.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.channel_specific_seeds.__module__)"` — Expected: PASS, 그리고 `flymon.brain.conditioning`.

- [ ] **Step 2: 실패하는 테스트** — `tests/brain/test_pool_bench.py`

```python
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_bench import (DECISIONS, EVAL_DECISIONS, budget_hours, budget_hours_e2e, budget_table, exact_match, m0b_gate,
                        throughput_row)

A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}


def test_budget_formulas():
    # step-based: 16 workers, 1.9 / 2.6 ms: training 52,000 x (5,600 x 1.9 + 1,600 x 2.6) ms + eval 81,600 x 5,600 x 1.9 ms, / 16
    h = budget_hours(1.9, 2.6, 16)
    train = DECISIONS * (4 * 1400 * 1.9 + 1600 * 2.6)
    ev = EVAL_DECISIONS * 4 * 1400 * 1.9
    assert h == pytest.approx((train + ev) / 16 / 3.6e6)
    assert budget_hours(1.9, 2.6, 16, eval_decisions=0) < h
    assert budget_hours(1.9, 2.6, 8) == pytest.approx(2 * h)
    # end to end: a 16-fly batch taking 11 s to decide and 4 s to reinforce
    e = budget_hours_e2e(11.0, 4.0, 16)
    assert e == pytest.approx((DECISIONS * 15.0 + EVAL_DECISIONS * 11.0) / 16 / 3600)
    assert budget_hours_e2e(11.0, 4.0, 16, eval_decisions=0) < e


def _row(workers, s_dec, s_rein, ms_dec=1.9, ms_rein=2.6):
    return {"workers": workers, "n_flies_batch": workers, "ms_decision_median": ms_dec, "ms_decision_max": ms_dec,
            "ms_reinforce_median": ms_rein, "ms_reinforce_max": ms_rein,
            "s_decide_batch_median": s_dec, "s_decide_batch_max": s_dec, "s_reinforce_batch_median": s_rein, "s_reinforce_batch_max": s_rein}


def test_budget_table_picks_the_best_configuration_and_gates_on_it():
    rows = [_row(8, 8.0, 3.0), _row(16, 11.0, 4.0)]
    t = budget_table(rows)
    assert [b["workers"] for b in t["by_workers"]] == [8, 16]
    assert t["workers"] == 16 and t["gate_ok"] is True and 20 < t["baseline_hours"] < 40
    assert t["rows"][0]["hours"] == pytest.approx(budget_hours_e2e(11.0, 4.0, 16))
    assert t["rows"][1]["hours"] < t["rows"][0]["hours"]
    contended = budget_table([_row(8, 8.0, 3.0), _row(16, 30.0, 10.0)])       # 16 workers thrash: 8 is the better configuration
    assert contended["workers"] == 8
    slow = budget_table([_row(16, 60.0, 20.0)])
    assert slow["gate_ok"] is False


def test_throughput_row_on_a_synthetic_pool(synthetic_npz):
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
    pops = Populations.from_connectome(Connectome.load(synthetic_npz))
    with FlyPool(synthetic_npz, p, [FlySpec(), FlySpec()], workers=2, timeout_s=120) as pool:
        row = throughput_row(pool, A, 1.0, steps=5, warm=2, repeats=2, idx=pops.mbon, settle_decision_ms=5, read_ms=5,
                             settle_reinforce_ms=5, pulse_ms=5, gap_ms=5)
        assert row["workers"] == 2 and row["n_flies_batch"] == 2 and row["repeats"] == 2
        assert row["ms_decision_max"] >= row["ms_decision_median"] > 0
        assert row["s_decide_batch_max"] >= row["s_decide_batch_median"] > 0 and row["s_reinforce_batch_max"] > 0
        assert all(pool.weights_frac(f) == pytest.approx(1.0) for f in range(2))     # learning undone


def _res(dD, a=0):
    return {"D_pre": -0.9, "D_post": -0.9 + dD, "dD": dD, "D_pre_disc": -1.0, "D_post_disc": -1.0, "dD_disc": 0.0,
            "weights_frac": 0.98, "w_frac_a_core": 0.9, "w_frac_p_core": 0.99,
            "counts": {"pre_plus": {"A": a, "P": 33}, "pre_minus": {"A": 31, "P": 35}, "post_plus": {"A": 0, "P": 22}, "post_minus": {"A": 14, "P": 0}}}


def test_exact_match_and_gate():
    ref = {"0": {"both": _res(0.2), "reversed": _res(1.4)}, "1": {"both": _res(0.3), "reversed": _res(1.2)}}
    same = {0: {"both": _res(0.2), "reversed": _res(1.4)}, 1: {"both": _res(0.3), "reversed": _res(1.2)}}
    m = exact_match(same, ref)
    assert m["ok"] and m["n_results"] == 4 and m["n_equal"] == 4 and m["max_abs_diff"] == 0.0
    off = {0: {"both": _res(0.2 + 1e-9), "reversed": _res(1.4)}, 1: {"both": _res(0.3, a=1), "reversed": _res(1.2)}}
    m = exact_match(off, ref)
    assert not m["ok"] and m["n_equal"] == 2 and m["max_abs_diff"] == pytest.approx(1e-9)
    m = exact_match({0: {"both": _res(0.2)}}, ref)
    assert m["n_missing"] == 3 and not m["ok"]
    budget = {"gate_ok": True}
    assert m0b_gate(budget, {"ok": True}, {"ok": True}, True)["passed"] is True
    assert m0b_gate(budget, {"ok": False}, {"ok": True}, True)["passed"] is False
    assert m0b_gate({"gate_ok": False}, {"ok": True}, {"ok": True}, True)["passed"] is False
```

`tests/test_summary.py`에 추가:

```python
P_M0B = Path("results/summary/m0b.json")


@pytest.mark.skipif(not P_M0B.exists(), reason="M0b summary not written yet")
def test_m0b_summary_records_gate_and_budget():
    """M0b gate = budget <= 60 h AND exact reproduction of M0 (conditioning, sparsity) AND pool decide == in-process."""
    d = json.loads(P_M0B.read_text())
    assert set(d["gate"]) == {"throughput_ok", "conditioning_exact_ok", "sparsity_exact_ok", "decide_equal_ok", "passed"}
    assert d["budget"]["limit_hours"] == 60.0
    assert d["gate"]["throughput_ok"] == (d["budget"]["baseline_hours"] <= 60.0)
    assert d["gate"]["passed"] == all(d["gate"][k] for k in ("throughput_ok", "conditioning_exact_ok", "sparsity_exact_ok", "decide_equal_ok"))
    assert d["params_frozen"]["kc_thresh"] == Params().kc_thresh
    assert d["runaway"]["n_kc_over_sat"] >= 0.0
    assert {r["workers"] for r in d["throughput"]} >= {4, 8, 16}
```

- [ ] **Step 3: 실패 확인** — Run: `uv run pytest tests/brain/test_pool_bench.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.pool_bench'`.

- [ ] **Step 4: 구현** — `flymon/brain/pool_bench.py`

```python
"""M0b budget extrapolation (spec C.6), throughput rows, exact-reproduction checks against the M0 result files, and the gate."""
from __future__ import annotations

import statistics
import time

from .pool_jobs import phase_timing_job

DECISIONS = 52_000            # spec 4.1: 42,000 training decisions + 10,000 exploration / secondary
EVAL_DECISIONS = 81_600       # spec 4.2: (42 flies x 4 evals + 12 REV x 1 + 6 WEAK-FLY x 4) x 20 battles x 20 fly decisions
WINDOWS = {"n_candidates": 4, "settle_decision_ms": 800.0, "read_ms": 600.0,
           "settle_reinforce_ms": 800.0, "pulse_max_ms": 600.0, "gap_ms": 200.0}


# ---- parent-side ---------------------------------------------------------------------------------
def throughput_row(pool, odor, strength: float, steps: int, warm: int, repeats: int = 3, idx=None,
                   settle_decision_ms: float = 800.0, read_ms: float = 600.0, settle_reinforce_ms: float = 800.0,
                   pulse_ms: float = 600.0, gap_ms: float = 200.0, reward_type: str = "PAM08") -> dict:
    """Two measurements per repeat. (1) In-worker step times of the two phases, all workers at once (the
    slowest worker bounds a batch, so the max over workers is kept). (2) End to end from the parent: one
    `decide_batch` (one fly per worker, four candidates) and one `reinforce_batch` at the real window
    lengths — this includes weight shipping, resets, presentation and result transfer, and is what the
    budget uses. The learning from (2) is undone afterwards."""
    W = pool.n_workers
    F = min(pool.n_flies, W)
    dec, rein, e2e_dec, e2e_rein = [], [], [], []
    for r in range(repeats):
        res = pool.run_jobs(phase_timing_job, [dict(odor=odor, strength=strength, steps=steps, warm=warm, reward_type=reward_type)] * W)
        dec.append(max(x["ms_decision"] for x in res))
        rein.append(max(x["ms_reinforce"] for x in res))
        t0 = time.perf_counter()
        pool.decide_batch([(f, [odor] * 4, 1000 + r) for f in range(F)], strength, settle_decision_ms, read_ms, idx)
        e2e_dec.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pool.reinforce_batch([(f, odor, reward_type, pulse_ms, 2000 + r) for f in range(F)], strength, settle_reinforce_ms, gap_ms)
        e2e_rein.append(time.perf_counter() - t0)
        for f in range(F):
            pool.w[f] = pool.w0[pool.flies[f].shuffle_seed].copy()
    return {"workers": W, "n_flies_batch": F, "repeats": repeats, "steps": steps, "warm": warm,
            "ms_decision_median": statistics.median(dec), "ms_decision_max": max(dec),
            "ms_reinforce_median": statistics.median(rein), "ms_reinforce_max": max(rein),
            "agg_slot_steps_per_ms": W / statistics.median(dec),
            "s_decide_batch_median": statistics.median(e2e_dec), "s_decide_batch_max": max(e2e_dec),
            "s_reinforce_batch_median": statistics.median(e2e_rein), "s_reinforce_batch_max": max(e2e_rein),
            "windows": {"settle_decision_ms": settle_decision_ms, "read_ms": read_ms, "settle_reinforce_ms": settle_reinforce_ms,
                        "pulse_ms": pulse_ms, "gap_ms": gap_ms}}


def budget_hours(ms_decision: float, ms_reinforce: float, workers: int, decisions: int = DECISIONS,
                 eval_decisions: int = EVAL_DECISIONS, n_candidates: int = 4, settle_decision_ms: float = 800.0,
                 read_ms: float = 600.0, settle_reinforce_ms: float = 800.0, pulse_max_ms: float = 600.0,
                 gap_ms: float = 200.0, dt_ms: float = 1.0) -> float:
    """Step-based estimate (diagnostic): training decisions cost n_candidates decision runs + one
    reinforcement each; evaluation decisions (spec 4.2, plasticity off) cost the decision runs only."""
    dec_steps = n_candidates * (settle_decision_ms + read_ms) / dt_ms
    rein_steps = (settle_reinforce_ms + pulse_max_ms + gap_ms) / dt_ms
    train_ms = decisions * (dec_steps * ms_decision + rein_steps * ms_reinforce)
    eval_ms = eval_decisions * dec_steps * ms_decision
    return (train_ms + eval_ms) / workers / 3.6e6


def budget_hours_e2e(s_decide_batch: float, s_reinforce_batch: float, n_flies_batch: int, decisions: int = DECISIONS,
                     eval_decisions: int = EVAL_DECISIONS) -> float:
    """Gate estimate from measured batch wall-clock: a batch serves n_flies_batch decisions at once."""
    return (decisions * (s_decide_batch + s_reinforce_batch) + eval_decisions * s_decide_batch) / n_flies_batch / 3600.0


def budget_table(rows: list, decisions: int = DECISIONS, eval_decisions: int = EVAL_DECISIONS,
                 limit_hours: float = 60.0) -> dict:
    """Hours for every worker count measured (end-to-end max over repeats, and the step-based estimate);
    the gate reads the best end-to-end configuration, not simply the largest one."""
    by_workers = []
    for r in rows:
        by_workers.append({"workers": r["workers"], "n_flies_batch": r["n_flies_batch"],
                           "hours_e2e": budget_hours_e2e(r["s_decide_batch_max"], r["s_reinforce_batch_max"], r["n_flies_batch"], decisions, eval_decisions),
                           "hours_e2e_training_only": budget_hours_e2e(r["s_decide_batch_max"], r["s_reinforce_batch_max"], r["n_flies_batch"], decisions, 0),
                           "hours_step_estimate": budget_hours(r["ms_decision_max"], r["ms_reinforce_max"], r["workers"], decisions, eval_decisions)})
    best = min(by_workers, key=lambda b: b["hours_e2e"])
    table = [
        {"assumption": f"baseline: end-to-end batches, {best['workers']} workers, training + evaluation", "hours": best["hours_e2e"]},
        {"assumption": "training decisions only", "hours": best["hours_e2e_training_only"]},
        {"assumption": "step-based estimate (diagnostic, excludes pool overhead)", "hours": best["hours_step_estimate"]},
    ]
    return {"workers": best["workers"], "n_flies_batch": best["n_flies_batch"], "decisions": decisions, "eval_decisions": eval_decisions,
            "windows": WINDOWS, "by_workers": by_workers, "rows": table, "limit_hours": limit_hours,
            "baseline_hours": table[0]["hours"], "gate_ok": table[0]["hours"] <= limit_hours}


FLOAT_KEYS = ("D_pre", "D_post", "dD", "D_pre_disc", "D_post_disc", "dD_disc", "weights_frac", "w_frac_a_core", "w_frac_p_core")


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

- [ ] **Step 5: CLI** — `scripts/bench_pool.py`

```python
#!/usr/bin/env python3
"""M0b: pool throughput, the M0 measurements reproduced through the pool (bit-for-bit), and the gate summary.

  throughput  per-worker ms/step of the decision and reinforcement phases at 4/8/16 workers, 3 repeats
  reproduce   M0 conditioning 5 arms x 8 seeds as pool jobs (== results/m0/conditioning.json per seed),
              sparsity seeds 100-102 (== results/m0/sparsity.json default row), resting baseline + runaway
              set, and one in-process vs pool decide() equality check
  summary     results/summary/m0b.json: budget table (spec C.6), exact-match results, the M0b gate
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import channel_specific_seeds, summarise
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_bench import budget_table, exact_match, m0b_gate, throughput_row
from flymon.brain.pool_jobs import baseline_job, conditioning_arm_job, rss_job, sparsity_job
from flymon.brain.presentation import decide

ARMS = ("both", "reversed", "noplast", "punish_only", "reward_only")


def _npz(a) -> str:
    if not Path(a.npz).exists():
        print(f"SKIP: {a.npz} not found (real data is user-downloaded, see docs/data.md)")
        raise SystemExit(2)
    return a.npz


def _write(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def cmd_throughput(a):
    npz = _npz(a)
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    odor_a, _ = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    rows = []
    for W in a.workers:
        with FlyPool(npz, Params(), [FlySpec()] * W, workers=W) as pool:
            rows.append(throughput_row(pool, odor_a, a.strength, a.steps, a.warm, a.repeats, idx=pops.mbon))
        print(json.dumps(rows[-1]), flush=True)
    _write(a.out, {"steps": a.steps, "warm": a.warm, "repeats": a.repeats, "strength": a.strength, "odor_seed": a.odor_seed, "rows": rows})


def cmd_reproduce(a):
    npz = _npz(a)
    t0 = time.perf_counter()
    m0_cond = json.loads(Path(a.m0_conditioning).read_text()) if Path(a.m0_conditioning).exists() else None
    m0_sp = json.loads(Path(a.m0_sparsity).read_text()) if Path(a.m0_sparsity).exists() else None
    p = Params()
    with FlyPool(npz, p, [FlySpec()] * a.workers, workers=a.workers) as pool:
        jobs = [dict(seed=s, arm=arm, strength=a.strength, k=a.k, odor_seed=a.odor_seed, trials=a.trials,
                     present_ms=a.present_ms, settle_ms=a.settle_ms, punish_type=a.punish_type, reward_type=a.reward_type)
                for s in range(a.seeds) for arm in ARMS]
        res = pool.run_jobs(conditioning_arm_job, jobs)
        per_seed: dict = {}
        for job, r in zip(jobs, res):
            per_seed.setdefault(job["seed"], {})[job["arm"]] = r
        cond = summarise(per_seed)
        cond["channel_specific_seeds"] = channel_specific_seeds(cond["per_seed"])
        sp_rows = pool.run_jobs(sparsity_job, [dict(seed=100 + s, strength=a.strength, k=a.k, odor_seed=a.odor_seed) for s in range(a.rest_seeds)])
        base_rows = pool.run_jobs(baseline_job, [dict(seed=100 + s, ms=a.rest_ms) for s in range(a.rest_seeds)])
        rss_before = pool.run_jobs(rss_job, [{}] * a.workers)
        rss_after = pool.run_jobs(rss_job, [{}] * a.workers, shuffle_seed=1_000_003)      # builds one C-shuf variant per worker
        memory = {"worker_rss_GB": float(np.mean(rss_before)), "worker_rss_GB_max": float(max(rss_before)),
                  "variant_rss_GB": float(np.mean(np.subtract(rss_after, rss_before)))}
        conn = Connectome.load(npz)
        pops = Populations.from_connectome(conn)
        odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
        eng = Engine(conn, pops, p, seed=0)
        pl = Plasticity(eng, pops, compartments(conn, pops, p.core_frac))
        local = decide(eng, pl, pops, [odor_a, odor_b], a.strength, seed=7, settle_ms=200.0, read_ms=600.0, idx=pops.mbon)
        remote = pool.decide_batch([(0, [odor_a, odor_b], 7)], a.strength, settle_ms=200.0, read_ms=600.0, idx=pops.mbon)[0]
    decide_equal = bool(np.array_equal(local, remote))
    sparsity = {k: float(np.mean([r[k] for r in sp_rows])) for k in sp_rows[0]}
    sparsity["per_seed"] = sp_rows
    trimmed = np.array([r["mbon_hz_trimmed"] for r in base_rows])
    baseline = {"mbon_hz_rest": float(np.mean([r["mbon_hz"] for r in base_rows])), "mbon_hz_rest_trimmed": float(trimmed.mean()),
                "mbon_hz_rest_trimmed_sd": float(trimmed.std()), "per_seed": base_rows,
                "runaway": {k: float(np.mean([r["runaway"][k] for r in base_rows])) for k in ("n_over_sat", "n_kc_over_sat", "spike_share_over_sat")}}
    cond_match = exact_match(per_seed, {k: v for k, v in m0_cond["per_seed"].items() if int(k) < a.seeds}) if m0_cond else {"ok": False, "note": "no M0 file"}
    sp_match = {"ok": False, "note": "no M0 file"}
    if m0_sp:
        row = [g for g in m0_sp["grid"] if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac")) == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac)][0]
        keys = ("frac_active_A", "frac_active_B", "jaccard", "chance", "mbon_hz_A", "mbon_hz_B")
        diffs = {k: abs(sparsity[k] - row[k]) for k in keys}
        diffs["mbon_hz_rest_trimmed"] = abs(baseline["mbon_hz_rest_trimmed"] - row["mbon_hz_rest_trimmed"])
        sp_match = {"diffs": diffs, "max_abs_diff": max(diffs.values()), "ok": max(diffs.values()) == 0.0, "cpu_row": {k: row[k] for k in keys + ("mbon_hz_rest_trimmed",)}}
    out = {"params": {**dataclasses.asdict(p), "punish_type": a.punish_type, "reward_type": a.reward_type, "settle_ms": a.settle_ms,
                      "present_ms": a.present_ms, "trials": a.trials}, "strength": a.strength, "odor_seed": a.odor_seed,
           "workers": a.workers, "conditioning": cond, "conditioning_match": cond_match, "sparsity": sparsity, "sparsity_match": sp_match,
           "baseline": baseline, "memory": memory, "decide_equal": decide_equal, "wall_clock_s": time.perf_counter() - t0}
    print(json.dumps({"n_seeds": cond["n_seeds"], "n_flip": cond["n_flip"], "channel_specific": cond["channel_specific_seeds"],
                      "conditioning_exact": cond_match.get("ok"), "sparsity_exact": sp_match.get("ok"), "decide_equal": decide_equal,
                      "runaway": baseline["runaway"], "memory": memory, "wall_s": round(out["wall_clock_s"])}), flush=True)
    _write(a.out, out)


def cmd_summary(a):
    th = json.loads(Path(a.throughput).read_text())
    rp = json.loads(Path(a.reproduce).read_text())
    budget = budget_table(th["rows"], decisions=a.decisions, eval_decisions=a.eval_decisions, limit_hours=a.limit_hours)
    gate = m0b_gate(budget, rp["conditioning_match"], rp["sparsity_match"], rp["decide_equal"])
    co = rp["conditioning"]
    out = {"params_frozen": dataclasses.asdict(Params()), "throughput": th["rows"], "budget": budget,
           "conditioning": {k: co[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")},
           "conditioning_match": rp["conditioning_match"], "sparsity_match": rp["sparsity_match"], "decide_equal": rp["decide_equal"],
           "runaway": rp["baseline"]["runaway"], "memory": rp.get("memory"), "gate": gate, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    print(f"M0b gate {'PASS' if gate['passed'] else 'FAIL'}: baseline {budget['baseline_hours']:.1f} h (limit {budget['limit_hours']}), "
          f"conditioning_exact={gate['conditioning_exact_ok']} sparsity_exact={gate['sparsity_exact_ok']} decide_equal={gate['decide_equal_ok']}")
    _write(a.out, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("throughput")
    t.add_argument("--npz", default="data/malecns.npz"); t.add_argument("--out", default="results/m0b/throughput.json")
    t.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16]); t.add_argument("--steps", type=int, default=300)
    t.add_argument("--warm", type=int, default=100); t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--strength", type=float, default=0.35); t.add_argument("--k", type=int, default=8); t.add_argument("--odor-seed", type=int, default=0)
    t.set_defaults(fn=cmd_throughput)
    r = sub.add_parser("reproduce")
    r.add_argument("--npz", default="data/malecns.npz"); r.add_argument("--out", default="results/m0b/reproduce.json")
    r.add_argument("--m0-conditioning", default="results/m0/conditioning.json"); r.add_argument("--m0-sparsity", default="results/m0/sparsity.json")
    r.add_argument("--workers", type=int, default=16); r.add_argument("--seeds", type=int, default=8)
    r.add_argument("--rest-seeds", type=int, default=3); r.add_argument("--rest-ms", type=float, default=3000.0)
    r.add_argument("--trials", type=int, default=12); r.add_argument("--present-ms", type=float, default=800.0); r.add_argument("--settle-ms", type=float, default=800.0)
    r.add_argument("--strength", type=float, default=0.35); r.add_argument("--k", type=int, default=8); r.add_argument("--odor-seed", type=int, default=0)
    r.add_argument("--punish-type", default="PPL105"); r.add_argument("--reward-type", default="PAM08")
    r.set_defaults(fn=cmd_reproduce)
    s = sub.add_parser("summary")
    s.add_argument("--throughput", default="results/m0b/throughput.json"); s.add_argument("--reproduce", default="results/m0b/reproduce.json")
    s.add_argument("--out", default="results/summary/m0b.json"); s.add_argument("--decisions", type=int, default=52_000)
    s.add_argument("--eval-decisions", type=int, default=81_600); s.add_argument("--limit-hours", type=float, default=60.0)
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 통과 확인** — Run: `uv run pytest tests/brain/test_pool_bench.py tests/test_summary.py -v` — Expected: PASS(m0b 요약 테스트는 파일이 없어 skip).

- [ ] **Step 7: CLI 스모크(결과 디렉터리 금지)** — 실제 데이터가 있을 때만, 임시 경로로:

```bash
if test -f data/malecns.npz; then
  T=$(mktemp -d)
  uv run python scripts/bench_pool.py throughput --workers 4 --steps 50 --warm 20 --repeats 2 --out $T/throughput.json
  uv run python scripts/bench_pool.py reproduce --seeds 1 --workers 5 --out $T/reproduce.json
  uv run python scripts/bench_pool.py summary --throughput $T/throughput.json --reproduce $T/reproduce.json --out $T/m0b.json
fi
```

Expected: `reproduce`가 `"conditioning_exact": true, "sparsity_exact": true, "decide_equal": true`와 `memory`를 찍고(약 2분), `summary`가 `M0b gate FAIL: baseline … h`(4워커라 예산 초과가 정상)를 찍는다. `git status`에 `results/` 변화가 없어야 한다. 데이터가 없으면 `SKIP:` 한 줄과 종료 코드 2.

- [ ] **Step 8: 전체 스위트 + 커밋**

```bash
uv run pytest -q
git add flymon/brain/conditioning.py flymon/brain/pool_bench.py scripts/bench_pool.py scripts/write_m0_summary.py tests/brain/test_conditioning.py tests/brain/test_pool_bench.py tests/test_summary.py
git commit -m "feat(brain): pool budget/exact-reproduction/gate (spec C.6) and the bench_pool CLI; channel_specific_seeds lives in conditioning.py"
```

---
### Task 4: 문서 — README M0b 절, 스펙 6절 파일 목록

**Files:**
- Modify: `README.md`(`### 학습 중 보기` 앞에 `### M0b 프로세스 풀 스웜` 절), `docs/superpowers/specs/2026-09-14-flymon-design.md`(6절 `brain/` 줄에 `pool_jobs.py`)

- [ ] **Step 1: README** — `### 학습 중 보기` 앞에 추가:

```markdown
### M0b 프로세스 풀 스웜

    uv run python scripts/bench_pool.py throughput            # 워커 4/8/16, 결정·강화 단계 ms/step 3회 → results/m0b/throughput.json
    uv run python scripts/bench_pool.py reproduce             # M0 조건화 5팔×8시드를 풀로 재실행(비트 동일 검사), 희소성·기저·폭주 집합 → results/m0b/reproduce.json (16워커, 약 10분)
    uv run python scripts/bench_pool.py summary               # 예산표(스펙 C.6)·정확 일치·게이트 → results/summary/m0b.json

- 스웜은 `flymon/brain/fly_pool.py`의 워커 프로세스 풀이다. 워커마다 CPU 엔진 하나, 마리별로는 KC→MBON 가중치·켬/끔·배선 변형만 남고 부모가 보관한다.
- 결정은 후보를 같은 시드로 순차 제시해 잡음을 짝짓는다(`flymon/brain/presentation.py`). 강화는 M0 `train_block`의 한 프레젠테이션과 같다.
- MPS 배치 엔진은 스파이크와 레드팀 뒤 채택하지 않았다(스펙 부록 C.6).
```

- [ ] **Step 2: 스펙 6절** — `brain/` 줄을 `brain/    data_build.py  engine_cpu.py  presentation.py  fly_pool.py  pool_jobs.py  pool_bench.py  circuits.py  odors.py  plasticity.py  readout.py`로 바꾼다.

- [ ] **Step 3: 커밋**

```bash
git add README.md docs/superpowers/specs/2026-09-14-flymon-design.md
git commit -m "docs: README M0b section; spec repo layout names pool_jobs.py"
```

---

## 컨트롤러 실행 (서브에이전트 금지) — 실제 데이터 벤치·재현·게이트, 부록 C.7

1. `uv run pytest -q` 초록 확인.
2. `uv run python scripts/bench_pool.py throughput` (워커 4/8/16, 300스텝 × 3회 × 2단계; 약 3분). 실행 중 다른 무거운 프로세스를 돌리지 않는다.
3. `uv run python scripts/bench_pool.py reproduce` (조건화 40작업을 16워커로 3라운드 + 희소성·기저 3작업 + decide 검사; 약 10분).
4. `uv run python scripts/bench_pool.py summary` → `results/summary/m0b.json`, 한 줄 판정. `uv run pytest tests/test_summary.py -q`.
5. 판정에 따라:
   - PASS: `results/summary/m0b.json` 커밋(`results: M0b pool bench, exact M0 reproduction, gate`). 부록 C에 C.7 "M0b 측정 결과"를 추가한다 — 처리량 표(워커별 스텝 시간과 엔드투엔드 배치 시간), `by_workers` 예산과 게이트에 쓴 구성, 세 정확 일치 결과, 메모리(워커·변형), 폭주 집합(KC 수·스파이크 비율). README "측정된 것"에 M0b 한 문단.
   - 예산 > 60시간: 스펙 4.1 축소 순서(배틀 40 → 30, 그다음 2차 팔)를 적용한 예산을 C.7에 함께 적고 판정을 기록.
   - 정확 일치 실패: 워커의 엔진 구성(`Engine(conn, pops, params, seed=0)`, `compartments`, `Readout`)과 M0 스크립트 `_cond_worker`의 차이를 찾는다. 시드는 프레젠테이션마다 명시적으로 리셋되므로 엔진 초기 시드는 결과에 영향이 없다.
6. **STD 재검토 조건(A.5)이 충족됐다**(휴지 폭주 집합에 KC 17개). 요약의 `runaway`와 함께 핸드오프에 적고, M2 브레인스토밍의 첫 질문에 "STD 도입 여부(조건 충족)"를 올린다. M0b에서는 결정하지 않는다.
7. 최종 브랜치 리뷰(sdd-reviewer, 이 계획 범위만), 핸드오프 `docs/handoffs/2026-09-15-m2-handoff.md`. 핸드오프의 M3 항목 두 가지: `decide_batch`/`reinforce_batch`는 동기 호출이라 배리어의 `run_batch`는 `loop.run_in_executor`로 감싸야 poke-env 웹소켓이 멈추지 않는다; 스펙 3.6의 체크포인트 가중치 fp16은 정확 재개를 깨므로 float32로 고친다.

## 자체 검토

- 스펙 커버리지: 2절 3·5단계 = Task 1 `decide`/`reinforce`; 3.1 프로세스 풀 항목(워커·가중치 스왑·hive·플래그·셔플·상태 왕복) = Task 1–2; 3.6 = Task 2; 5절 M0b(등가성 세 가지·처리량 반복·예산 학습/평가·폭주 집합) = Task 2–3 + 컨트롤러; 8절 통계 = Task 2(decide 동일)·Task 3(비트 동일 재현); C.6 결정 전부 반영. 배리어 `run_batch`는 M3에서 `decide_batch`에 얹는다(스펙 2절).
- 플레이스홀더: 없음. 모든 코드 블록은 스크래치에서 실행·통과했다. 레드팀 반영: 엔드투엔드 배치 타이밍이 게이트 기준, 변형 상한과 RSS 측정, 중복 마리 거부, 생성자의 부모 측 검증과 정리, `load_state` 유한성, `reinforce`의 finally 정리, 워커 수별 예산.
- 타입 일관성: `decide(...) -> np.ndarray`, `reinforce(...) -> None`, `FlyPool.decide_batch -> list[np.ndarray]`, `run_jobs(fn, kwargs_list, shuffle_seed=None)`, `fn(engine, pl, pops, comps, ro, **kw)`, `throughput_row(pool, odor, strength, steps, warm, repeats)`, `exact_match(pool_per_seed, m0_per_seed)`, `m0b_gate(budget, conditioning_match, sparsity_match, decide_equal)`가 Task 1·2·3과 CLI에서 같은 이름·시그니처로 쓰인다.
