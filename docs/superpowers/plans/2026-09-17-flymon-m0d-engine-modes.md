# FlyMon M0d H.2 — 엔진 모드(등급 APL · ORN 억압 · KC 항상성 역치) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙 부록 H.2의 세 엔진 모드를 `flymon/brain`에 기본값 꺼짐으로 넣는다 — 꺼져 있으면 M0c 엔진과 비트 동일하고, 켜면 H.2의 식대로 동작하며, 풀 경로와 인프로세스 경로가 모든 모드에서 같은 카운트를 내고, 기본값이 아닌 모드는 M0/M0b/M0c 참조 경로에 쓰지 못한다.

**Architecture:** `Params`에 모드 필드 10개를 더하고 `Engine.__init__`이 `_validate_modes`로 검증한다. 등급 APL은 `Engine.step` 안에서 APL의 스파이크를 막고 갱신된 막전위로 계산한 방출을 지연선(`_apl_release`)에 넣어 APL 출력 엣지로 전달한다. ORN 억압은 수용체별 자원 `_std_r`과 이득 지연선(`_std_delay`)을 두고 `propagate(src, gain)`이 출처별 이득을 곱한다. 항상성 역치는 새 모듈 `flymon/brain/thresholds.py`가 검증한 npz 파일을 `v_th[kc]`에 쓴다. 쓰기 가드는 `pool_bench.refuse_modified_engine_output`. 모든 분기는 플래그가 꺼져 있으면 실행되지 않는다.

**Tech Stack:** Python 3.13(uv), numpy 2.x, multiprocessing(spawn), pytest. 새 의존성 없음.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` — **부록 H**(H.0 조사, H.1 조합, **H.2 엔진 구현**, H.3 고정 규칙, H.7 순서). 이 계획은 H.2만 구현한다. H.3 이후(조합별 고정·선택·확인)는 이 구현의 결과를 쓰는 별도 계획이다.

## Global Constraints

- Python 3.13, uv. **새 의존성 추가 금지.**
- **기본 `Params()`는 M0c 엔진과 비트 동일**해야 한다: `apl_mode="spiking"`, `orn_std=False`, `kc_thresh_mode="pn_norm"`. 기존 테스트의 수치·구간은 바꾸지 않는다. `results/m0*` 참조를 읽는 기존 실데이터 테스트가 그대로 통과해야 한다.
- 문헌 고정값(H.2/H.3): `apl_v_mid` = 11.0 mV, `apl_slope` = 5.0 mV, `orn_std_f` = 0.78, `orn_std_tau_ms` = 893.0. `apl_r_max` 기본 0.333(스파이킹 상한 333 Hz의 스텝 등가) — 실제 값은 H.3이 정한다.
- 등급 APL 방출: r = `apl_r_max` / (1 + exp(−(v − `apl_v_mid`)/`apl_slope`)), **갱신된** 막전위에서 계산해 지연선에 넣고 `dly_steps` 뒤 APL 출력 엣지 가중치 × r로 전달. APL은 스파이크 판정·리셋·불응기에서 빠진다. 막전위 바닥(≥ −v_thresh)은 그대로.
- ORN 억압: 수용체마다 자원 R(리셋 시 1). 수용체 스파이크는 **감소 전** R을 곱해 전달, 그다음 R ← f·R, 매 스텝 모든 수용체 R ← R + (1 − R)·dt/τ. 수용체가 아닌 출처는 이득 1.
- 항상성 역치 파일: npz(`kc_body_ids` int64, `v_th` float32), sha256이 `kc_thresh_sha256`과 일치해야 하고, body id가 커넥톰의 KC 순서와 같아야 하며, 값은 유한·양수, **PN 입력이 0인 KC는 PN 정규화 규칙의 값을 유지**해야 한다(허용오차 1e-6). 어기면 `ValueError`.
- 기본값이 아닌 모드는 `results/m0/`, `results/m0b/`, `results/m0c/`, `results/summary/{m0,m0b,m0c,compartments}.json`에 쓰지 못한다(SystemExit 2, stderr에 "M0d"). 결과는 `results/m0d/`.
- **서브에이전트는 `results/` 아래에 쓰지 않고 실데이터 실행을 하지 않는다.** 이 계획의 태스크는 합성 커넥톰 테스트만 돈다(`tests/conftest.py`의 `synthetic_connectome`, `synthetic_npz`). 실데이터 확인은 마지막 절에서 컨트롤러가 한다.
- 커밋은 태스크마다. 메시지 `feat(brain): …`, `test(brain): …`. 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- 계획의 코드는 세션 스크래치(저장소 사본)에서 **실행해 검증한 것**이다: 전체 스위트 240개(238 passed, 1 skipped, 1 xfailed; 기존 223 + 새 17), 변이 8종 전부 사멸(등급 APL 스파이크 허용, 가중치 없는 전달, 감소 후 이득 전달, 회복 제거, `propagate`가 이득 무시, PN 입력 0 규칙 검사 제거, body id 순서 검사 제거, 갱신 전 막전위로 방출). 실행 중 편차가 생기면 멈추고 보고한다.

---

## 검증된 사실 (2026-09-17, 실제 커넥톰, 인프로세스)

- 실데이터 CSC에서 APL 두 세포의 출력 엣지는 2261·2237개이고 **대상이 모두 서로 다르다**(연결체 전체의 중복 (pre, post) 쌍 0). 등급 전달의 `g[tgt] += w * r`은 반복 대상을 한 번만 더하므로, 엔진은 초기화 때 반복 대상을 거부한다(태스크 2).
- 스텝당 시간(설계 냄새 B, 강도 0.35, 600 스텝): 기본 0.95 ms · 등급 0.96 · ORN 억압 0.89 · 둘 다 1.03 — 모드가 처리량을 떨어뜨리지 않는다.
- 참고(H.3 입력, 이 계획의 판정 아님): 문헌값 ORN 억압을 켜면 `kc_thresh` 1.5에서 설계 냄새 B의 KC 활성이 0.0%가 된다(시드 100 한 번).

## 파일 구조

| 파일 | 책임 | 태스크 |
|---|---|---|
| `flymon/brain/config.py` | 모드 필드 10개 | 1 |
| `flymon/brain/engine_cpu.py` | 검증(1), 등급 APL(2), ORN 억압(3), 항상성 역치 적용(4) | 1–4 |
| `flymon/brain/thresholds.py` (새) | 역치 파일 저장·검증 로드 | 4 |
| `flymon/brain/pool_bench.py` | `refuse_modified_engine_output` | 5 |
| `tests/brain/test_engine_modes.py` (새) | 모드 테스트 전부 | 1–5 |
| `tests/brain/test_pool_bench.py` | 가드 테스트 | 5 |

---

### Task 1: 모드 파라미터, 검증, 기본값 비트 동일성

**Files:**
- Modify: `flymon/brain/config.py` (kc_kc_scale 주석 블록 바로 뒤)
- Modify: `flymon/brain/engine_cpu.py` (모듈 docstring, import 아래 상수, `Engine.__init__` 첫 줄, 파일 끝 함수)
- Create: `tests/brain/test_engine_modes.py`

**Interfaces:**
- Produces: `Params.apl_mode: str = "spiking"`, `apl_r_max: float = 0.333`, `apl_v_mid: float = 11.0`, `apl_slope: float = 5.0`, `orn_std: bool = False`, `orn_std_f: float = 0.78`, `orn_std_tau_ms: float = 893.0`, `kc_thresh_mode: str = "pn_norm"`, `kc_thresh_file: str = ""`, `kc_thresh_sha256: str = ""`; `engine_cpu.APL_MODES = ("spiking", "graded")`, `engine_cpu.KC_THRESH_MODES = ("pn_norm", "homeostatic")`; `engine_cpu._validate_modes(p: Params) -> None`(잘못된 값이면 `ValueError`).

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_engine_modes.py`를 만든다(이후 태스크가 이 파일에 테스트를 더한다):

```python
"""M0d engine modes (spec appendix H.2): graded APL, ORN->PN depression, homeostatic KC thresholds.
Every default must keep the M0c engine bit-identical."""
import hashlib

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide

BASE = dict(noise_mv=0.0, min_weight=1, balance_hemispheres=False, mbon_hold_frac=0.0)


def _engine(synthetic_connectome, **kw):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    return Engine(c, pops, Params(**{**BASE, **kw}), seed=1), c, pops


def _reference_step(eng):
    """The M0c Engine.step, frozen verbatim before the M0d modes were added."""
    p = eng.p
    arrived = eng.delay.popleft()
    if arrived.size:
        eng.g += eng.propagate(arrived)
    dt_m = p.dt / p.tau_m
    dv = (-eng.v + eng.g + eng.ext) * dt_m
    if p.noise_mv:
        dv += eng.rng.standard_normal(eng.N, dtype=np.float32) * p.noise_mv
    free = eng.refrac <= 0
    eng.v = np.where(free, eng.v + dv, eng.v)
    eng.refrac = np.where(free, eng.refrac, eng.refrac - 1)
    np.maximum(eng.v, -p.v_thresh, out=eng.v)
    eng.g *= (1.0 - p.dt / p.tau_syn)
    spk = (eng.v >= eng.v_th) & free
    ri = eng.receptor_idx
    hz = eng.drive_hz[ri]
    pois = (eng.rng.random(ri.size) < hz * (p.dt / 1000.0)) & free[ri]
    spk[ri] = pois
    fired = np.flatnonzero(spk)
    eng.v[fired] = p.v_reset
    eng.refrac[fired] = p.refrac_steps()
    eng.last = fired
    eng.delay.append(fired)
    eng.t_ms += p.dt
    if eng.on_step is not None:
        eng.on_step(eng, fired)
    return fired


# ---- defaults -----------------------------------------------------------------------------------------------
def test_mode_defaults_are_the_m0c_engine():
    p = Params()
    assert (p.apl_mode, p.orn_std, p.kc_thresh_mode) == ("spiking", False, "pn_norm")
    assert (p.apl_v_mid, p.apl_slope, p.orn_std_f, p.orn_std_tau_ms) == (11.0, 5.0, 0.78, 893.0)


def test_default_step_is_bit_identical_to_the_m0c_step(synthetic_connectome):
    kw = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    a, b = Engine(c, pops, Params(**kw), seed=3), Engine(c, pops, Params(**kw), seed=3)
    orn = np.flatnonzero(c.cls == "olfactory")
    for e in (a, b):
        e.set_drive_hz(orn, 180.0)
    for _ in range(400):
        np.testing.assert_array_equal(a.step(), _reference_step(b))
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.g, b.g)


def test_unknown_modes_are_rejected(synthetic_connectome):
    with pytest.raises(ValueError, match="apl_mode"):
        _engine(synthetic_connectome, apl_mode="local")
    with pytest.raises(ValueError, match="kc_thresh_mode"):
        _engine(synthetic_connectome, kc_thresh_mode="adaptive")
    with pytest.raises(ValueError, match="orn_std_f"):
        _engine(synthetic_connectome, orn_std=True, orn_std_f=1.5)
    with pytest.raises(ValueError, match="apl_r_max"):
        _engine(synthetic_connectome, apl_mode="graded", apl_r_max=0.0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py -v`
Expected: FAIL — `TypeError: Params.__init__() got an unexpected keyword argument 'apl_mode'`(또는 `AttributeError: 'Params' object has no attribute 'apl_mode'`). `test_default_step_is_bit_identical_to_the_m0c_step`는 이 시점에 이미 통과해도 된다(이후 태스크의 회귀 방지용이다).

- [ ] **Step 3: `config.py`에 필드 추가** — 다음 블록을 찾아

```python
                                   # the first-order stand-in. 1.0 reproduces the M0/M0b engine (spec D).
```

바로 뒤(빈 줄 앞)에 넣어 아래처럼 만든다:

```python
                                   # the first-order stand-in. 1.0 reproduces the M0/M0b engine (spec D).

    # --- M0d engine modes (spec appendix H.2); every default keeps the M0c engine bit-identical ---
    apl_mode: str = "spiking"      # "graded": APL never spikes and releases apl_release(v) per step (Amin et al. 2020)
    apl_r_max: float = 0.333       # graded release at saturation, spike equivalents per step (spiking cap 333 Hz)
    apl_v_mid: float = 11.0        # sigmoid midpoint, mV above rest (GGN model, Ray et al. 2020: -40 mV mid, -51 mV rest)
    apl_slope: float = 5.0         # sigmoid slope, mV
    orn_std: bool = False          # presynaptic depression of receptor out-edges (Nagel et al. 2015)
    orn_std_f: float = 0.78        # resource kept per spike
    orn_std_tau_ms: float = 893.0  # recovery time constant
    kc_thresh_mode: str = "pn_norm"   # "homeostatic": per-KC thresholds from kc_thresh_file (spec H.3)
    kc_thresh_file: str = ""
    kc_thresh_sha256: str = ""
```

- [ ] **Step 4: `engine_cpu.py` 검증 추가**

(a) 모듈 docstring에서 `Receptors (sensory classes) ignore the membrane and fire as Poisson sources at drive_hz.` 줄 바로 뒤에 넣는다:

```
M0d modes (spec appendix H.2, all off by default):
  apl_mode "graded"          APL never spikes; each step it queues r = apl_r_max / (1 + exp(-(v - apl_v_mid) / apl_slope))
                             from its updated membrane, delivered through its out-edges after the synaptic delay.
  orn_std                    each receptor carries a resource R (1 at reset); a receptor spike delivers its out-edges
                             scaled by R, then R <- orn_std_f * R; every step R <- R + (1 - R) * dt / orn_std_tau_ms.
  kc_thresh_mode "homeostatic"  KC thresholds come from a validated file (flymon.brain.thresholds).
```

(b) `from .connectome import Connectome, build_csc` 바로 뒤에:

```python

APL_MODES = ("spiking", "graded")
KC_THRESH_MODES = ("pn_norm", "homeostatic")
```

(c) `Engine.__init__`의 첫 줄 `self.p = params` 앞에 `_validate_modes(params)`를 넣는다:

```python
    def __init__(self, conn: Connectome, pops: Populations, params: Params, seed: int = 0):
        _validate_modes(params)
        self.p = params
```

(d) 파일 끝에:

```python


def _validate_modes(p: Params) -> None:
    if p.apl_mode not in APL_MODES:
        raise ValueError(f"apl_mode must be one of {APL_MODES}, got {p.apl_mode!r}")
    if p.kc_thresh_mode not in KC_THRESH_MODES:
        raise ValueError(f"kc_thresh_mode must be one of {KC_THRESH_MODES}, got {p.kc_thresh_mode!r}")
    if p.apl_mode == "graded" and not (p.apl_r_max > 0 and p.apl_slope > 0):
        raise ValueError(f"graded APL needs apl_r_max > 0 and apl_slope > 0, got {p.apl_r_max}, {p.apl_slope}")
    if p.orn_std and not (0 < p.orn_std_f <= 1 and p.orn_std_tau_ms > 0):
        raise ValueError(f"orn_std needs 0 < orn_std_f <= 1 and orn_std_tau_ms > 0, got {p.orn_std_f}, {p.orn_std_tau_ms}")
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py tests/brain/test_config.py tests/brain/test_engine_cpu.py -v`
Expected: PASS (새 3개 + 기존 전부)

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/config.py flymon/brain/engine_cpu.py tests/brain/test_engine_modes.py
git commit -m "feat(brain): M0d mode parameters and validation, defaults pinned to the M0c step"
```

---

### Task 2: 등급 APL

**Files:**
- Modify: `flymon/brain/engine_cpu.py` (`__init__`의 `drive_hz` 뒤, `reset` 끝, `step`, `run` 앞에 메서드)
- Test: `tests/brain/test_engine_modes.py`

**Interfaces:**
- Consumes: Task 1의 `Params.apl_mode`, `apl_r_max`, `apl_v_mid`, `apl_slope`.
- Produces: `Engine.apl_idx: np.ndarray[int64]`(모든 모드), `Engine.apl_release(v: np.ndarray) -> np.ndarray[float32]`, 등급 모드일 때 `Engine._apl_release: deque[np.ndarray]`(길이 `dly_steps`, 원소 크기 = APL 수), `Engine._graded_apl: bool`.

- [ ] **Step 1: 실패하는 테스트 추가** — `tests/brain/test_engine_modes.py` 끝에:

```python


# ---- graded APL ---------------------------------------------------------------------------------------------
def test_graded_apl_never_spikes_and_the_spiking_control_does(synthetic_connectome):
    for mode, expect_spikes in (("spiking", True), ("graded", False)):
        eng, c, pops = _engine(synthetic_connectome, apl_mode=mode)
        eng.set_ext(pops.apl, 1000.0)
        fired_apl = any(np.isin(pops.apl, eng.step()).any() for _ in range(30))
        assert fired_apl is expect_spikes, mode


def test_apl_release_is_the_declared_sigmoid(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded", apl_r_max=0.4)
    v = np.array([-7.0, 0.0, 11.0, 30.0], np.float32)
    np.testing.assert_allclose(eng.apl_release(v), 0.4 / (1.0 + np.exp(-(v - 11.0) / 5.0)), rtol=1e-6)
    assert eng.apl_release(np.array([11.0], np.float32))[0] == pytest.approx(0.2)


def test_graded_release_is_queued_from_the_updated_membrane_and_delivered_after_the_delay(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded", apl_r_max=0.5)
    apl = int(pops.apl[0])
    kc = int(pops.kc[0])
    ptr, tgt, w = eng.csc.ptr, eng.csc.tgt, eng.csc.w
    w_apl_kc = float(w[ptr[apl]:ptr[apl + 1]][tgt[ptr[apl]:ptr[apl + 1]] == kc].sum())
    assert w_apl_kc < 0                                       # GABA, scaled by apl_scale
    eng.set_ext([apl], 100.0)
    eng.step()                                                # step 1: v_apl = 5 mV, release queued
    r1 = float(eng.apl_release(eng.v[[apl]])[0])
    assert eng._apl_release[-1][0] == pytest.approx(r1)
    assert eng.g[kc] == 0.0
    eng.step()                                                # step 2: nothing has arrived yet
    assert eng.g[kc] == 0.0
    eng.step()                                                # step 3: release of step 1 arrives, then tau_syn decay
    assert eng.g[kc] == pytest.approx(w_apl_kc * r1 * (1 - 1 / 5.0), rel=1e-5)


def test_graded_apl_rejects_repeated_out_edge_targets(synthetic_connectome):
    c0 = synthetic_connectome()
    pops = Populations.from_connectome(c0)
    apl, kc = int(pops.apl[0]), int(pops.kc[0])
    c = Connectome(bodyId=c0.bodyId, type=c0.type, cls=c0.cls, sc=c0.sc, nt=c0.nt, sign=c0.sign, side=c0.side,
                   pre=np.append(c0.pre, np.int32(apl)), post=np.append(c0.post, np.int32(kc)), w=np.append(c0.w, np.int32(5)))
    Engine(c, pops, Params(**BASE), seed=1)                   # the spiking engine sums repeated edges in propagate
    with pytest.raises(ValueError, match="unique out-edge targets"):
        Engine(c, pops, Params(**BASE, apl_mode="graded"), seed=1)


def test_graded_reset_clears_the_release_line(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded")
    eng.set_ext(pops.apl, 100.0)
    for _ in range(5):
        eng.step()
    eng.reset(seed=2)
    assert len(eng._apl_release) == eng.p.dly_steps()
    assert all((r == 0).all() for r in eng._apl_release)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py -k graded -v`
Expected: FAIL — `AttributeError: 'Engine' object has no attribute 'apl_release'` 등(스파이킹 대조는 통과, 등급은 APL이 스파이크하므로 실패).

- [ ] **Step 3: `__init__`** — 다음 두 줄을 찾아

```python
        self.drive_hz = np.zeros(self.N, np.float32)

        self._seed = seed
```

이렇게 바꾼다:

```python
        self.drive_hz = np.zeros(self.N, np.float32)

        self._graded_apl = params.apl_mode == "graded"
        self.apl_idx = np.asarray(pops.apl, np.int64)
        if self._graded_apl:                        # views into the CSC: APL out-edges are never plastic
            ptr, tgt, w = self.csc.ptr, self.csc.tgt, self.csc.w
            self._apl_edges = [(tgt[ptr[i]:ptr[i + 1]], w[ptr[i]:ptr[i + 1]]) for i in self.apl_idx]
            if any(np.unique(t).size != t.size for t, _ in self._apl_edges):   # g[tgt] += adds a repeated target once
                raise ValueError("graded APL needs unique out-edge targets per APL cell")

        self._seed = seed
```

- [ ] **Step 4: `reset`** — 끝의 두 줄

```python
        self.last = np.zeros(0, np.int64)
        self.t_ms = 0.0
```

뒤에 넣는다:

```python
        if self._graded_apl:
            self._apl_release = deque([np.zeros(self.apl_idx.size, np.float32) for _ in range(self.p.dly_steps())],
                                      maxlen=self.p.dly_steps())
```

- [ ] **Step 5: `step`** — (a) 시작부

```python
        arrived = self.delay.popleft()
        if arrived.size:
            self.g += self.propagate(arrived)
        dt_m = p.dt / p.tau_m
```

를

```python
        arrived = self.delay.popleft()
        if arrived.size:
            self.g += self.propagate(arrived)
        if self._graded_apl:
            r_in = self._apl_release.popleft()
            for k in np.flatnonzero(r_in):
                tgt, w = self._apl_edges[k]
                self.g[tgt] += w * r_in[k]
        dt_m = p.dt / p.tau_m
```

로. (b) 스파이크 처리부

```python
        spk[ri] = pois
        fired = np.flatnonzero(spk)
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refrac_steps()
        self.last = fired
        self.delay.append(fired)          # delivered dly_steps steps from now
        self.t_ms += p.dt
```

를

```python
        spk[ri] = pois
        if self._graded_apl:
            spk[self.apl_idx] = False     # non-spiking: no threshold, reset or refractory
        fired = np.flatnonzero(spk)
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refrac_steps()
        self.last = fired
        self.delay.append(fired)          # delivered dly_steps steps from now
        if self._graded_apl:
            self._apl_release.append(self.apl_release(self.v[self.apl_idx]))
        self.t_ms += p.dt
```

로. (c) `def run(` 바로 앞에 메서드:

```python
    def apl_release(self, v: np.ndarray) -> np.ndarray:
        """Graded APL release per step (spike equivalents) at membrane v, mV above rest."""
        p = self.p
        return (p.apl_r_max / (1.0 + np.exp(-(np.asarray(v, np.float32) - p.apl_v_mid) / p.apl_slope))).astype(np.float32)

```

- [ ] **Step 6: 통과 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py tests/brain/test_engine_cpu.py -v`
Expected: PASS (기본값 비트 동일성 포함)

- [ ] **Step 7: 커밋**

```bash
git add flymon/brain/engine_cpu.py tests/brain/test_engine_modes.py
git commit -m "feat(brain): graded non-spiking APL with saturating release (spec H.2)"
```

---

### Task 3: ORN→PN 억압

**Files:**
- Modify: `flymon/brain/engine_cpu.py` (`__init__`, `reset`, `propagate`, `step`)
- Test: `tests/brain/test_engine_modes.py`

**Interfaces:**
- Consumes: Task 1의 `Params.orn_std`, `orn_std_f`, `orn_std_tau_ms`; Task 2가 만든 `step`의 형태.
- Produces: `Engine.propagate(src: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray`(gain은 출처별, `src`와 같은 길이), 억압 모드일 때 `Engine._std_r: np.ndarray[float32, N]`(비수용체는 늘 1), `Engine._std_delay: deque[np.ndarray[float32]]`(각 원소는 그 스텝 `fired`와 같은 길이의 이득), `Engine._orn_std: bool`.

- [ ] **Step 1: 실패하는 테스트 추가** — 파일 끝에:

```python


# ---- ORN->PN depression -------------------------------------------------------------------------------------
def test_propagate_gain_scales_each_source(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome)
    src = np.array([0, 5, 17])
    gain = np.array([0.5, 1.0, 0.25], np.float32)
    expect = sum(g * eng.propagate(np.array([s])) for s, g in zip(src, gain))
    np.testing.assert_allclose(eng.propagate(src, gain), expect, atol=1e-5)


def test_orn_depression_follows_the_discrete_rule_and_reaches_its_steady_state(synthetic_connectome):
    f, tau = 0.78, 893.0
    eng, c, pops = _engine(synthetic_connectome, orn_std=True, orn_std_f=f, orn_std_tau_ms=tau)
    o = int(np.flatnonzero(c.cls == "olfactory")[0])
    eng.set_drive_hz([o], 1e6)                                # p = 1 whenever not refractory: every 3rd step
    k = 1.0 / tau
    r, gains = 1.0, []
    for _ in range(3000):
        fired = eng.step()
        if o in fired:
            gains.append(float(eng._std_delay[-1][fired == o][0]))
            assert gains[-1] == pytest.approx(r, rel=1e-5)
            r *= f
        r += (1.0 - r) * k
        assert eng._std_r[o] == pytest.approx(r, rel=1e-5)
    a = (1.0 - k) ** 3
    assert gains[-1] == pytest.approx((1.0 - a) / (1.0 - f * a), rel=1e-4)   # delivered gain at steady state


def test_orn_depression_leaves_other_sources_at_full_weight_and_reset_restores_it(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, orn_std=True)
    o = int(np.flatnonzero(c.cls == "olfactory")[0])
    kc = int(pops.kc[0])
    eng.set_drive_hz([o], 1e6)
    eng.set_ext([kc], 1000.0)
    for _ in range(30):
        fired = eng.step()
        if kc in fired:
            assert eng._std_delay[-1][fired == kc][0] == 1.0
    assert eng._std_r[o] < 1.0
    eng.reset(seed=4)
    assert (eng._std_r == 1.0).all()
    assert all(x.size == 0 for x in eng._std_delay)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py -k "depression or gain" -v`
Expected: FAIL — `TypeError: Engine.propagate() takes 2 positional arguments but 3 were given`, `AttributeError: ... '_std_delay'`.

- [ ] **Step 3: `__init__`** — Task 2가 만든

```python
                raise ValueError("graded APL needs unique out-edge targets per APL cell")

        self._seed = seed
```

를

```python
                raise ValueError("graded APL needs unique out-edge targets per APL cell")
        self._orn_std = bool(params.orn_std)

        self._seed = seed
```

로.

- [ ] **Step 4: `reset`** — Task 2가 넣은 블록

```python
        if self._graded_apl:
            self._apl_release = deque([np.zeros(self.apl_idx.size, np.float32) for _ in range(self.p.dly_steps())],
                                      maxlen=self.p.dly_steps())
```

바로 뒤에:

```python
        if self._orn_std:
            self._std_r = np.ones(self.N, np.float32)          # stays 1 for every non-receptor
            self._std_delay = deque([np.zeros(0, np.float32) for _ in range(self.p.dly_steps())], maxlen=self.p.dly_steps())
```

- [ ] **Step 5: `propagate`** — 시그니처·docstring과 `val` 줄을 바꾼다. 다음을

```python
    def propagate(self, src: np.ndarray) -> np.ndarray:
        """Sum signed mV of every out-edge of the spiking sources into a dense [N] vector."""
```

```python
    def propagate(self, src: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        """Sum signed mV of every out-edge of the spiking sources into a dense [N] vector; `gain` scales each
        source's out-edges (ORN depression)."""
```

로 바꾸고, `val = np.concatenate([w[a:b] for a, b in zip(starts, ends)])` 바로 뒤에:

```python
        if gain is not None:
            val = val * np.repeat(np.asarray(gain, np.float32), ends - starts)
```

- [ ] **Step 6: `step`** — (a) 시작부

```python
        arrived = self.delay.popleft()
        if arrived.size:
            self.g += self.propagate(arrived)
```

를

```python
        arrived = self.delay.popleft()
        arrived_gain = self._std_delay.popleft() if self._orn_std else None
        if arrived.size:
            self.g += self.propagate(arrived, arrived_gain)
```

로. (b) Task 2가 만든

```python
        self.delay.append(fired)          # delivered dly_steps steps from now
        if self._graded_apl:
            self._apl_release.append(self.apl_release(self.v[self.apl_idx]))
```

를

```python
        self.delay.append(fired)          # delivered dly_steps steps from now
        if self._orn_std:
            self._std_delay.append(self._std_r[fired].copy())
            rf = fired[self.is_receptor[fired]]
            self._std_r[rf] *= np.float32(p.orn_std_f)
            self._std_r[ri] += (1.0 - self._std_r[ri]) * np.float32(p.dt / p.orn_std_tau_ms)
        if self._graded_apl:
            self._apl_release.append(self.apl_release(self.v[self.apl_idx]))
```

로.

- [ ] **Step 7: 통과 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py tests/brain/test_engine_cpu.py -v`
Expected: PASS

- [ ] **Step 8: 커밋**

```bash
git add flymon/brain/engine_cpu.py tests/brain/test_engine_modes.py
git commit -m "feat(brain): ORN presynaptic depression with per-source gain (spec H.2)"
```

---

### Task 4: KC 항상성 역치 파일

**Files:**
- Create: `flymon/brain/thresholds.py`
- Modify: `flymon/brain/engine_cpu.py` (import, `__init__`의 역치 계산 뒤)
- Test: `tests/brain/test_engine_modes.py`

**Interfaces:**
- Consumes: Task 1의 `Params.kc_thresh_mode`, `kc_thresh_file`, `kc_thresh_sha256`.
- Produces: `thresholds.save_kc_thresholds(path, kc_body_ids, v_th) -> tuple[Path, str]`(경로, 파일 sha256), `thresholds.load_kc_thresholds(path: str, sha256: str, kc_body_ids, rule_v_th, has_pn_input) -> np.ndarray[float32]`, `Engine.kc_pn_input: np.ndarray`(KC 순서의 PN→KC 시냅스 합, 모든 모드). H.3의 보정 스크립트가 `save_kc_thresholds`와 `Engine.kc_pn_input`을 쓴다.

- [ ] **Step 1: 실패하는 테스트 추가** — (a) 파일 상단 import 블록의 `from flymon.brain.presentation import decide` 뒤에:

```python
from flymon.brain.thresholds import load_kc_thresholds, save_kc_thresholds
```

(b) 파일 끝에:

```python


# ---- homeostatic KC thresholds ------------------------------------------------------------------------------
def _theta_file(tmp_path, eng, c, pops, scale=1.2):
    v_th = eng.v_th[pops.kc].astype(np.float32).copy()
    movable = eng.kc_pn_input > 0
    v_th[movable] *= np.float32(scale)
    path = tmp_path / "theta.npz"
    return save_kc_thresholds(path, c.bodyId[pops.kc], v_th), v_th


def test_homeostatic_thresholds_load_into_the_kcs_only(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    (path, sha), v_th = _theta_file(tmp_path, ref, c, pops)
    eng, _, _ = _engine(synthetic_connectome, kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256=sha)
    np.testing.assert_array_equal(eng.v_th[pops.kc], v_th)
    others = np.setdiff1d(np.arange(c.N), pops.kc)
    np.testing.assert_array_equal(eng.v_th[others], ref.v_th[others])


def test_homeostatic_file_is_validated(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    (path, sha), v_th = _theta_file(tmp_path, ref, c, pops)
    ids = c.bodyId[pops.kc]
    mk = lambda **kw: _engine(synthetic_connectome, kc_thresh_mode="homeostatic", **kw)
    with pytest.raises(ValueError, match="kc_thresh_file"):
        mk()
    with pytest.raises(ValueError, match="sha256"):
        mk(kc_thresh_file=str(path), kc_thresh_sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        mk(kc_thresh_file=str(path))
    bad = tmp_path / "bad_ids.npz"
    p2, s2 = save_kc_thresholds(bad, ids[::-1], v_th)
    with pytest.raises(ValueError, match="body ids"):
        mk(kc_thresh_file=str(p2), kc_thresh_sha256=s2)
    neg = v_th.copy(); neg[0] = -1.0
    p3, s3 = save_kc_thresholds(tmp_path / "neg.npz", ids, neg)
    with pytest.raises(ValueError, match="positive"):
        mk(kc_thresh_file=str(p3), kc_thresh_sha256=s3)
    short = tmp_path / "short.npz"
    p4, s4 = save_kc_thresholds(short, ids[:-1], v_th[:-1])
    with pytest.raises(ValueError, match="KCs"):
        mk(kc_thresh_file=str(p4), kc_thresh_sha256=s4)


def test_homeostatic_file_must_keep_the_rule_for_kcs_without_pn_input(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    zero = np.flatnonzero(ref.kc_pn_input == 0)
    if zero.size == 0:                                        # make one: drop every PN->KC edge onto the first KC
        c0 = synthetic_connectome()
        k0 = int(Populations.from_connectome(c0).kc[0])
        keep = ~(np.isin(c0.pre, Populations.from_connectome(c0).alpn) & (c0.post == k0))
        c = Connectome(bodyId=c0.bodyId, type=c0.type, cls=c0.cls, sc=c0.sc, nt=c0.nt, sign=c0.sign, side=c0.side,
                       pre=c0.pre[keep], post=c0.post[keep], w=c0.w[keep])
        pops = Populations.from_connectome(c)
        ref = Engine(c, pops, Params(**BASE), seed=1)
        zero = np.flatnonzero(ref.kc_pn_input == 0)
    v_th = ref.v_th[pops.kc].astype(np.float32).copy()
    v_th[zero[0]] *= np.float32(2.0)
    path, sha = save_kc_thresholds(tmp_path / "moved.npz", c.bodyId[pops.kc], v_th)
    with pytest.raises(ValueError, match="no PN input"):
        Engine(c, pops, Params(**BASE, kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256=sha))


def test_load_kc_thresholds_returns_float32_in_kc_order(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    (path, sha), v_th = _theta_file(tmp_path, ref, c, pops)
    got = load_kc_thresholds(str(path), sha, c.bodyId[pops.kc], ref.v_th[pops.kc], ref.kc_pn_input > 0)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, v_th)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py -v`
Expected: 수집 단계 ERROR — `ModuleNotFoundError: No module named 'flymon.brain.thresholds'`.

- [ ] **Step 3: `flymon/brain/thresholds.py` 작성**

```python
"""Per-KC threshold files for the homeostatic threshold mode (spec appendix H.2, H.3).

A file is an npz with `kc_body_ids` (int64, the connectome's KC order) and `v_th` (float32, mV above rest).
The engine only accepts a file whose sha256 matches `Params.kc_thresh_sha256`, whose body ids match its KC
population in order, and which leaves every KC without PN input at the PN-normalisation rule's value.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def save_kc_thresholds(path, kc_body_ids, v_th) -> tuple[Path, str]:
    """Write the file and return (path, sha256 of its bytes)."""
    path = Path(path)
    ids = np.asarray(kc_body_ids, np.int64)
    th = np.asarray(v_th, np.float32)
    if ids.shape != th.shape:
        raise ValueError(f"kc_body_ids {ids.shape} and v_th {th.shape} differ in shape")
    with open(path, "wb") as f:                     # a file handle: np.savez would append .npz to a str path
        np.savez(f, kc_body_ids=ids, v_th=th)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_kc_thresholds(path: str, sha256: str, kc_body_ids, rule_v_th, has_pn_input) -> np.ndarray:
    """Validated thresholds in KC order (float32). `rule_v_th` is the PN-normalisation rule's value per KC and
    `has_pn_input` marks KCs with any PN input; KCs without PN input must keep the rule's value."""
    if not path:
        raise ValueError("kc_thresh_mode='homeostatic' needs kc_thresh_file")
    if not sha256:
        raise ValueError("kc_thresh_mode='homeostatic' needs kc_thresh_sha256")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"kc_thresh_file sha256 {digest} != kc_thresh_sha256 {sha256}")
    with np.load(path) as d:
        ids, th = d["kc_body_ids"].astype(np.int64), d["v_th"].astype(np.float32)
    want = np.asarray(kc_body_ids, np.int64)
    if ids.shape != want.shape:
        raise ValueError(f"threshold file holds {ids.size} KCs, the connectome has {want.size}")
    if not np.array_equal(ids, want):
        raise ValueError("threshold file body ids differ from the connectome's KC order")
    if not np.isfinite(th).all() or (th <= 0).any():
        raise ValueError("thresholds must be finite and positive")
    fixed = ~np.asarray(has_pn_input, bool)
    if not np.allclose(th[fixed], np.asarray(rule_v_th, np.float32)[fixed], rtol=0, atol=1e-6):
        raise ValueError("thresholds of KCs with no PN input must keep the PN-normalisation rule's value")
    return th
```

- [ ] **Step 4: `engine_cpu.py`** — (a) `from .connectome import Connectome, build_csc` 뒤(Task 1의 상수 앞)에:

```python
from .thresholds import load_kc_thresholds
```

(b) `__init__`의

```python
            self.v_th[pops.kc] = params.v_thresh * params.kc_thresh * np.clip(pn_in[pops.kc] / med, lo, hi)
```

바로 뒤(같은 `if med > 0:` 블록 **밖**, 들여쓰기 8칸)에:

```python
        self.kc_pn_input = pn_in[pops.kc]          # total PN->KC synapses per KC, in KC order
        if params.kc_thresh_mode == "homeostatic":
            self.v_th[pops.kc] = load_kc_thresholds(params.kc_thresh_file, params.kc_thresh_sha256,
                                                    conn.bodyId[pops.kc], self.v_th[pops.kc], self.kc_pn_input > 0)
```

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest tests/brain/test_engine_modes.py tests/brain/test_engine_cpu.py -v`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/thresholds.py flymon/brain/engine_cpu.py tests/brain/test_engine_modes.py
git commit -m "feat(brain): validated per-KC homeostatic threshold files (spec H.2)"
```

---

### Task 5: 풀 등가성과 쓰기 가드

**Files:**
- Modify: `flymon/brain/pool_bench.py` (`refuse_old_engine_output` 뒤)
- Modify: `tests/brain/test_pool_bench.py` (import, 가드 테스트)
- Test: `tests/brain/test_engine_modes.py` (풀 등가성)

**Interfaces:**
- Consumes: Task 1–4의 모든 모드, `FlyPool(npz, params, flies, workers, timeout_s)`와 `decide_batch`(기존).
- Produces: `pool_bench.PRE_M0D_DIRS`, `pool_bench.PRE_M0D_FILES`, `pool_bench.refuse_modified_engine_output(out: str, params) -> None`(M0d 모드가 켜져 있고 경로가 참조 트리면 stderr에 "M0d"를 포함해 쓰고 `SystemExit(2)`). H.3 스크립트가 출력 경로마다 부른다.

- [ ] **Step 1: 실패하는 테스트 추가** — (a) `tests/brain/test_engine_modes.py` 끝에:

```python


# ---- pool = in-process in every new mode --------------------------------------------------------------------
def test_pool_decide_equals_in_process_in_every_new_mode(synthetic_npz, tmp_path):
    c = Connectome.load(synthetic_npz)
    pops = Populations.from_connectome(c)
    base = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
    ref = Engine(c, pops, Params(**base), seed=0)
    v_th = ref.v_th[pops.kc].astype(np.float32).copy()
    v_th[ref.kc_pn_input > 0] *= np.float32(0.9)
    path, sha = save_kc_thresholds(tmp_path / "theta.npz", c.bodyId[pops.kc], v_th)
    A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
    B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
    for extra in (dict(apl_mode="graded", apl_r_max=0.3), dict(orn_std=True),
                  dict(kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256=sha),
                  dict(apl_mode="graded", apl_r_max=0.3, orn_std=True)):
        p = Params(**base, **extra)
        eng = Engine(c, pops, p, seed=0)
        pl = Plasticity(eng, pops, compartments(c, pops, p.core_frac))
        expected = decide(eng, pl, pops, [A, B], 1.0, seed=7, settle_ms=50, read_ms=300)
        with FlyPool(synthetic_npz, p, [FlySpec()], workers=1, timeout_s=120) as pool:
            got = pool.decide_batch([(0, [A, B], 7)], 1.0, settle_ms=50, read_ms=300)[0]
        np.testing.assert_array_equal(got, expected, err_msg=str(extra))
```

(b) `tests/brain/test_pool_bench.py`의 import를

```python
from flymon.brain.pool_bench import (DECISIONS, EVAL_DECISIONS, budget_hours, budget_hours_e2e, budget_table, exact_match,
                        m0b_gate, m0c_gate, match_sparsity_row, refuse_modified_engine_output, refuse_old_engine_output,
                        throughput_row)
```

로 바꾸고, `def _sp_row(**kw):` 바로 앞에:

```python
def test_refuse_modified_engine_output_guards_every_pre_m0d_reference(capsys):
    """Spec H.2: an engine with any M0d mode on never writes under the M0/M0b/M0c reference trees or summaries."""
    modified = (Params(apl_mode="graded"), Params(orn_std=True),
                Params(kc_thresh_mode="homeostatic", kc_thresh_file="x.npz", kc_thresh_sha256="0" * 64))
    for p in modified:
        for out in ("results/m0/x.json", "results/m0b/x.json", "results/m0c/sparsity.json", "./results/m0c/x.json",
                    "results/summary/m0.json", "results/summary/m0b.json", "results/summary/m0c.json",
                    "results/summary/compartments.json", os.path.abspath("results/summary/m0c.json")):
            with pytest.raises(SystemExit) as e:
                refuse_modified_engine_output(out, p)
            assert e.value.code == 2
            assert "M0d" in capsys.readouterr().err
        for out in ("results/m0d/sparsity.json", "results/summary/m0d.json", "/tmp/x.json", ""):
            refuse_modified_engine_output(out, p)
    for out in ("results/m0c/sparsity.json", "results/summary/m0c.json"):
        refuse_modified_engine_output(out, Params())         # the M0c engine may still write its own files


```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/brain/test_pool_bench.py tests/brain/test_engine_modes.py -k "refuse_modified or pool_decide" -v`
Expected: 수집 ERROR — `ImportError: cannot import name 'refuse_modified_engine_output'`. (풀 등가성 테스트는 Task 1–4 구현만으로 통과할 수 있다 — 등가성의 회귀 방지 테스트다.)

- [ ] **Step 3: `pool_bench.py`** — `refuse_old_engine_output` 함수 끝(`raise SystemExit(2)`)과 `def exact_match(` 사이에 넣는다(앞뒤 빈 줄 두 개씩):

```python
PRE_M0D_DIRS = ("results/m0/", "results/m0b/", "results/m0c/")
PRE_M0D_FILES = ("results/summary/m0.json", "results/summary/m0b.json", "results/summary/m0c.json",
                 "results/summary/compartments.json")


def refuse_modified_engine_output(out: str, params) -> None:
    """Spec H.2: an engine with any M0d mode on (graded APL, ORN depression, homeostatic thresholds) never writes
    under the M0/M0b/M0c reference trees or summaries (SystemExit 2); its results go under results/m0d/."""
    modified = params.apl_mode != "spiking" or params.orn_std or params.kc_thresh_mode != "pn_norm"
    if not modified or not out:
        return
    rel = os.path.relpath(os.path.abspath(str(out)), os.getcwd()).replace(os.sep, "/")
    if rel.startswith(PRE_M0D_DIRS) or rel in PRE_M0D_FILES:
        print(f"refusing to write {out} with M0d modes on (apl_mode={params.apl_mode}, orn_std={params.orn_std}, "
              f"kc_thresh_mode={params.kc_thresh_mode}): that path holds a pre-M0d engine's reference; use results/m0d/",
              file=sys.stderr)
        raise SystemExit(2)
```

- [ ] **Step 4: 통과 확인 — 전체 스위트**

Run: `uv sync && bash scripts/install_showdown.sh && uv run pytest -q`
Expected: **240 수집 — 238 passed, 1 skipped, 1 xfailed**(기존 221 passed + 새 17). 실패가 있으면 멈추고 보고한다.

- [ ] **Step 5: 커밋**

```bash
git add flymon/brain/pool_bench.py tests/brain/test_pool_bench.py tests/brain/test_engine_modes.py
git commit -m "feat(brain): pool = in-process in every M0d mode, and a write guard for pre-M0d references"
```

---

## 컨트롤러 실행 (서브에이전트 아님)

1. **스크래치 검증본과 대조**: 이 세션의 스크래치 사본(`…/scratchpad/m0d-impl/`)이 남아 있으면 `diff -u flymon/brain/engine_cpu.py <scratch>/flymon/brain/engine_cpu.py` 등으로 `engine_cpu.py`·`config.py`·`thresholds.py`·`pool_bench.py`·두 테스트 파일이 일치하는지 본다(주석·공백 차이는 보고만).
2. **실데이터 스텝 시간 스모크**(인프로세스, `results/` 쓰기 없음): 기본·등급·억압·둘 다에서 설계 냄새 B(강도 0.35, 시드 100)로 100 스텝 워밍업 뒤 600 스텝의 ms/step과 KC 활성 %를 잰다. 기준(검증된 사실 절)보다 2배 이상 느리면 보고한다.
3. 결과를 스펙 H.2 끝에 한 문단으로 덧붙이고(구현 커밋 해시, 스위트 수, 스텝 시간) 커밋한다. H.3 계획은 그다음이다.
