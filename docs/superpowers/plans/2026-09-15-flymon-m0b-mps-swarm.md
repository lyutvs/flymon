# FlyMon M0b — MPS 스웜 엔진 구현 계획 (배치 LIF, 짝지은 잡음, hive, 마리별 팔 플래그, CPU 일치, 처리량과 60시간 게이트)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M0의 CPU LIF 엔진과 같은 동역학을 torch MPS 위에서 B = 4F 슬롯으로 동시에 돌리는 스웜 엔진을 만들고, 합성 망에서는 CPU와 래스터가 정확히 같음을, 실제 커넥톰에서는 M0 측정값(희소성·MBON 기저·채널별 특이 억제)이 재현됨을 보이며, 처리량을 재어 4.1의 결정 수로 외삽한 예산이 60시간 이하임을 기록한다.

**Architecture:** `SwarmEngine`(`flymon/brain/engine_mps.py`)은 커넥톰을 공유하고 막 상태 [B, N]만 슬롯별로 둔다. 전파는 두 갈래다 — KC→MBON을 뺀 공유 CSC는 도착한 (슬롯, 뉴런) 쌍의 out-edge를 펼쳐 `index_add_`로 더하고, KC→MBON 33,496개 엣지는 마리별 밀집 행렬 W[F, n_KC, n_MBON]에 `bmm`한다(hive = 한 마리의 4슬롯이 W 하나를 읽는다). 잡음은 [F, ·]로 뽑아 4배 반복해 짝짓고, 잡음 소스를 주입할 수 있어 테스트는 CPU `Engine`의 numpy 난수를 그대로 먹인다. `SwarmPlasticity`(`plasticity_mps.py`)는 같은 3인자 규칙을 외적 형태로 마리별 슬롯 0에서만 적용하고 마리별 켬/끔·DAN 구동·회복을 둔다. `conditioning_mps.py`는 M0 측정(희소성·기저·폭주 집합·5팔 조건화)을 스웜 한 번으로 돌리고, `swarm_bench.py` + `scripts/bench_mps.py`가 처리량·예산표·게이트를 `results/summary/m0b.json`에 쓴다.

**Tech Stack:** Python 3.13(uv), numpy, **torch 2.14(MPS; `device="cpu"` 대체)**, pytest. 실제 데이터 `data/malecns.npz`(git 제외, 없으면 skip).

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` (v4 + 부록 A·B·C). 이 계획은 스펙 3.1의 MPS 배치 엔진 항목, 5절 M0b, 8절 통계 항목, **부록 C(설계 결정·엔진 설계·스파이크 측정·예산 외삽·테스트 목록)**를 구현한다. STD는 보류됐다(부록 A.5, 2026-09-15 추가).

## Global Constraints

- Python 3.13, uv. 새 의존성: `torch>=2.14,<3`(주 의존성; `uv sync`가 잠금 파일을 갱신한다). numpy 2.x. 다른 의존성 추가 금지.
- 엔진 기본 장치는 `"mps"`, `"cpu"`로도 동작해야 한다. **모든 스웜 테스트는 `device` 픽스처로 두 장치에서 돈다**(MPS가 없는 기계에서는 cpu만). MPS는 float64가 없으므로 float 배열은 항상 `dtype=torch.float32`로 옮긴다.
- B = 4F. 마리 f의 슬롯은 4f..4f+3, 강화 슬롯은 4f(슬롯 0). 잡음은 [F, ·]를 4배 반복(짝지은 잡음). 가소성 자취는 마리별 슬롯 0에서만 쌓인다(스펙 2절 5단계, 부록 C.2 4항).
- 방정식·지연선·불응기 회계·막전위 하한은 `engine_cpu.py`와 같다. torch 상수는 `float(np.float32(c))`로 미리 반올림해 numpy와 같은 값을 곱한다(그래야 합성 망 래스터가 비트 동일하다).
- 실제 데이터 테스트·스크립트는 `data/malecns.npz`가 없으면 skip. **서브에이전트는 `results/` 아래에 쓰지 않고 실제 데이터 벤치·재현을 돌리지 않는다**(컨트롤러 실행, 마지막 절). 스크립트 스모크는 `--out <tmp>`로만.
- 커밋은 태스크마다. 메시지 `feat(brain): …`, `test(brain): …`, `refactor(brain): …`, `docs: …`.
- 계획의 코드는 세션 스크래치에서 **실제로 실행해 검증한 것**이다(합성 망 테스트 48개, cpu·mps 두 장치; 실제 커넥톰 CLI 스모크). 그래도 실행 중 편차가 생기면 보고하고 판정을 받는다.

---

## 검증된 API 사실 (2026-09-15, torch 2.14.0 / Python 3.13.13 / Apple M5 Pro 20코어 GPU, 통합 메모리 48 GB)

- `uv pip install torch`가 Python 3.13에서 `torch==2.14.0`을 해결한다(의존: sympy, networkx, typing-extensions, setuptools). `torch.backends.mps.is_available()`은 True.
- `torch.Generator(device="mps")` + `manual_seed`, `get_state()`(uint8 텐서 44바이트, CPU) + `set_state()`가 동작한다. `torch.randn(..., generator=g)`, `torch.rand(...)`도 MPS 생성기를 받는다.
- MPS에서 동작 확인: `index_add_(0, idx, src)`(view 위에서), `repeat_interleave(x, repeats_tensor)`, `nonzero()`, `bmm`, `bincount`, `torch.where`, `clamp_(min=float)`, `tanh`, `torch.maximum(..., out=)`, 고급 인덱싱 대입 `t[rows[:, None], cols[None, :]] = v`, `t[b, n] += 1`(int32), `torch.as_tensor(np_float64, dtype=torch.float32, device="mps")`. `torch.mps.synchronize()`, `torch.mps.current_allocated_memory()`, `torch.mps.empty_cache()`.
- MPS에 없는 것: float64 텐서(변환 시 dtype 지정 필수). `np.asarray(mps_tensor)`는 TypeError — 텐서는 `.cpu().numpy()`.
- 실제 커넥톰(N 162,517, CSC 6,005,611 엣지, 그중 KC→MBON 33,496): 스파이크 측정은 부록 C.3. 가소성 켬 슬롯당 0.21–0.29 ms/step, B=256에서 GPU 메모리 858 MB.
- 합성 망에서 CPU `Engine`과 래스터가 300스텝 동안 정확히 같으려면 `mv_per_synapse=0.25`, `apl_scale=0.125`, `balance_hemispheres=False`가 필요하다(이진 정확 가중치 → float64 bincount와 float32 index_add_/bmm의 합이 비트 동일). 기본값 0.275/0.1로는 마지막 비트가 달라 래스터가 갈라질 수 있다.
- 합성 망 5팔 × 2시드 스웜 조건화는 `trials=5, learn_rate=0.1`에서 두 장치 모두 2/2 시드 부호 반전, `noplast` dD 정확히 0.0. CPU 테스트의 3/0.05로는 한 시드가 0으로 양자화된다(실행 확인).
- 실제 데이터 CLI 스모크(seeds 1, trials 1): 희소성 KC 0.059/0.065, Jaccard 0.021 ≤ 우연 0.032, MBON 절사 기저 3.86 Hz(M0: 0.062/0.058/0.023/3.52), 휴지 300 ms 100 Hz 초과 15개 중 KC 0, 스파이크 비율 2.5%.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | `torch>=2.14,<3` 추가 |
| `flymon/brain/engine_cpu.py` | `kc_thresholds(conn, pops, params)`, `tonic_ext(pops, params, n)`를 모듈 함수로 추출(동작 불변) |
| `flymon/brain/engine_mps.py` | `SLOTS`, `DeviceNoise`, `NumpyNoise`, `SwarmEngine`(생성·전파·스텝·run·드라이브·present·state) |
| `flymon/brain/plasticity_mps.py` | `SwarmPlasticity`(외적 규칙, 슬롯 0 자취, 마리별 켬/끔·DAN 구동·회복·state) |
| `flymon/brain/conditioning.py` | `channel_specific_seeds()` 이동(scripts와 테스트의 중복 제거) |
| `flymon/brain/conditioning_mps.py` | `fly_arms`, `swarm_probe`, `swarm_train_block`, `swarm_reversal_test`, `swarm_sparsity`, `swarm_baseline` |
| `flymon/brain/swarm_bench.py` | `throughput_row`, `budget_hours`, `budget_table`, `agreement`, `m0b_gate` |
| `scripts/bench_mps.py` | CLI `throughput` / `reproduce` / `summary` |
| `scripts/write_m0_summary.py` | `channel_specific_seeds`를 import로 교체 |
| `tests/conftest.py` | `device` 픽스처(cpu + mps) |
| `tests/brain/test_engine_mps.py`, `test_plasticity_mps.py`, `test_conditioning_mps.py`, `test_swarm_bench.py` | 태스크별 테스트 |
| `tests/brain/test_conditioning.py` | 중복 헬퍼를 import로 교체 |
| `tests/test_summary.py` | `results/summary/m0b.json` 검사(있을 때만) |
| `README.md`, 스펙 6절 | M0b 실행 절, 파일 목록 |

---
### Task 1: torch 의존성, CPU 엔진 리팩터, `SwarmEngine` 핵심

**Files:**
- Modify: `pyproject.toml`(dependencies), `flymon/brain/engine_cpu.py`(`Engine.__init__` 역치·기저 블록 → 모듈 함수), `tests/conftest.py`(`device` 픽스처)
- Create: `flymon/brain/engine_mps.py`
- Test: `tests/brain/test_engine_mps.py`

**Interfaces:**
- Produces `flymon.brain.engine_cpu.kc_thresholds(conn, pops, params) -> np.ndarray[N] float32`, `tonic_ext(pops, params, n) -> np.ndarray[N] float32`. `Engine`은 이 둘을 써서 `v_th`, `ext0`를 만든다(값 불변 — 기존 `test_engine_cpu.py`가 지킨다).
- Produces `flymon.brain.engine_mps.SLOTS = 4`, `DeviceNoise(device, seed)`(`seed(s)`, `normal(F, N)`, `uniform(F, R)`, `get_state()`, `set_state(s)`), `SwarmEngine(conn, pops, params, n_flies, seed=0, device="mps", noise=None, shuffle_seeds=None)`:
  속성 `F, B, N, n_kc, n_mb, n_plastic, device, v(B,N) g(B,N) refrac(B,N) ext(B,N) drive_hz(B,R) W(F,n_kc,n_mb) W0(F,n_kc,n_mb) kc_local(N) mb_local(N, numpy) mbon_idx receptor_idx(numpy) R ri ext0 v_th noise on_step t_ms last`;
  메서드 `reset(seed=None)`, `set_ext(slots, idx, mv)`, `clear_ext(slots=None)`, `set_drive_hz(slots, idx, hz)`, `clear_drive(slots=None)`, `present(slot, odor, strength)`, `propagate(b, n)`, `step() -> (b, n)`, `run(ms, idx=None) -> np.ndarray[B, N or len(idx)] int32`. `slots=None`은 전체. `idx`는 numpy 배열이나 장치 텐서. `on_step(engine, b, n)` 훅은 스텝 끝에 호출된다(Task 3의 가소성이 잡는다).
- `state()`/`load_state()`와 `NumpyNoise`는 Task 2.

- [ ] **Step 1: 의존성** — `pyproject.toml` dependencies에 `"torch>=2.14,<3",` 추가 후 `uv sync`. 확인: `uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"` → `2.14.0 True`(MPS 없는 기계면 False여도 진행).

- [ ] **Step 2: CPU 엔진 리팩터(동작 불변)** — `flymon/brain/engine_cpu.py`의 `class Engine:` 바로 앞에 두 함수를 추가한다.

```python
def kc_thresholds(conn: Connectome, pops: Populations, params: Params) -> np.ndarray:
    """[N] float32 thresholds: v_thresh everywhere, KC thresholds normalised by their PN input (our design decision)."""
    v_th = np.full(conn.N, params.v_thresh, np.float32)
    pn_in = np.zeros(conn.N, np.float64)
    is_alpn = np.zeros(conn.N, bool); is_alpn[pops.alpn] = True
    is_kc = np.zeros(conn.N, bool); is_kc[pops.kc] = True
    m = is_alpn[conn.pre] & is_kc[conn.post]
    np.add.at(pn_in, conn.post[m], conn.w[m])
    med = float(np.median(pn_in[pops.kc])) if len(pops.kc) else 1.0
    if med > 0:
        lo, hi = params.kc_norm_clip
        v_th[pops.kc] = params.v_thresh * params.kc_thresh * np.clip(pn_in[pops.kc] / med, lo, hi)
    return v_th


def tonic_ext(pops: Populations, params: Params, n: int) -> np.ndarray:
    """[N] float32 tonic drive: MBON hold (our design decision), 0 elsewhere."""
    ext0 = np.zeros(n, np.float32)
    ext0[pops.mbon] = params.mbon_hold_frac * params.v_thresh
    return ext0
```

`Engine.__init__`에서 아래 블록을

```python
        # thresholds: KC thresholds normalised by their PN input (our design decision)
        self.v_th = np.full(self.N, params.v_thresh, np.float32)
        pn_in = np.zeros(self.N, np.float64)
        is_alpn = np.zeros(self.N, bool); is_alpn[pops.alpn] = True
        is_kc = np.zeros(self.N, bool); is_kc[pops.kc] = True
        m = is_alpn[conn.pre] & is_kc[conn.post]   # one boolean mask, not two np.isin scans
        np.add.at(pn_in, conn.post[m], conn.w[m])
        med = float(np.median(pn_in[pops.kc])) if len(pops.kc) else 1.0
        if med > 0:
            lo, hi = params.kc_norm_clip
            self.v_th[pops.kc] = params.v_thresh * params.kc_thresh * np.clip(pn_in[pops.kc] / med, lo, hi)

        # tonic drive: MBON hold (our design decision); everything else 0 until set_ext
        self.ext0 = np.zeros(self.N, np.float32)
        self.ext0[pops.mbon] = params.mbon_hold_frac * params.v_thresh
        self.ext = self.ext0.copy()
```

이렇게 바꾼다.

```python
        self.v_th = kc_thresholds(conn, pops, params)      # KC thresholds normalised by PN input (our design decision)
        self.ext0 = tonic_ext(pops, params, self.N)        # MBON hold; everything else 0 until set_ext
        self.ext = self.ext0.copy()
```

Run: `uv run pytest tests/brain -q` — Expected: 기존 테스트 전부 PASS(변경 전과 같은 수).

- [ ] **Step 3: `device` 픽스처** — `tests/conftest.py`: 상단 import에 `import torch`를 더하고 파일 끝에 추가.

```python
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    """torch device for the swarm tests: always the CPU backend, plus MPS when the machine has it."""
    return request.param
```

- [ ] **Step 4: 실패하는 테스트** — `tests/brain/test_engine_mps.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine, kc_thresholds, tonic_ext
from flymon.brain.engine_mps import SLOTS, SwarmEngine


def _params(**kw):
    return Params(**{"noise_mv": 0.0, "min_weight": 1, "balance_hemispheres": False, "mbon_hold_frac": 0.0, **kw})


def _swarm(synthetic_connectome, device, n_flies=1, seed=1, noise=None, shuffle_seeds=None, **kw):
    c = synthetic_connectome()
    p = _params(**kw)
    pops = Populations.from_connectome(c)
    eng = SwarmEngine(c, pops, p, n_flies=n_flies, seed=seed, device=device, noise=noise, shuffle_seeds=shuffle_seeds)
    return eng, c, pops


# ---- engine --------------------------------------------------------------------------------
def test_thresholds_and_tonic_drive_match_cpu_engine(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(min_weight=1, balance_hemispheres=False, kc_thresh=1.5)
    pops = Populations.from_connectome(c)
    cpu = Engine(c, pops, p)
    np.testing.assert_array_equal(kc_thresholds(c, pops, p), cpu.v_th)
    np.testing.assert_array_equal(tonic_ext(pops, p, c.N), cpu.ext0)


def test_edge_split_and_dense_block(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device)
    cpu = Engine(c, pops, eng.p)
    assert eng.n_plastic == 40 * 4
    assert eng.tgt.numel() + eng.n_plastic == cpu.csc.tgt.size
    pre = cpu.csc.pre_of_edge()
    m = np.isin(pre, pops.kc) & np.isin(cpu.csc.tgt, pops.mbon)
    assert float(eng.W0[0].sum()) == pytest.approx(float(cpu.csc.w[m].sum()), rel=1e-6)
    assert int((eng.W0[0] > 0).sum()) == 40 * 4


def test_constant_drive_follows_discrete_rc_and_fires_at_step_24(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[0])
    eng.set_ext(0, [i], 10.0)
    v_prev, fired_at = 0.0, None
    for n in range(1, 40):
        b, spk = eng.step()
        if i in spk.tolist():
            fired_at = n
            break
        expected = v_prev + (10.0 - v_prev) * (1.0 / 20.0)
        assert float(eng.v[0, i]) == pytest.approx(expected, abs=1e-5)
        v_prev = expected
    assert fired_at == 24


def test_kc_spike_reaches_mbon_through_dense_block_after_delay(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device)
    a, b = int(pops.kc[0]), int(pops.mbon[0])
    w_ab = float(eng.W0[0, 0, 0])
    assert w_ab > 0
    eng.set_ext(0, [a], 1000.0)
    _, s1 = eng.step()
    assert a in s1.tolist()
    assert float(eng.g[0, b]) == 0.0
    eng.step()
    assert float(eng.g[0, b]) == 0.0
    eng.step()
    assert float(eng.g[0, b]) == pytest.approx(w_ab * (1 - 1 / 5.0), rel=1e-6)


def test_refractory_blocks_immediate_refire(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[1])
    eng.set_ext(0, [i], 1000.0)
    fired = [n for n in range(6) if i in eng.step()[1].tolist()]
    assert fired == [0, 3]


def test_poisson_receptor_rate(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device)
    orn = np.flatnonzero(c.cls == "olfactory")
    eng.set_drive_hz(0, orn, 100.0)
    counts = eng.run(5000)
    rate = counts[0][orn].mean() / 5.0
    assert 75 <= rate <= 92
    assert counts[1][orn].sum() == 0          # only slot 0 was driven


def test_reset_with_seed_is_deterministic(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device, noise_mv=0.15)
    eng.set_drive_hz(None, pops.sensory, 150.0)
    eng.reset(seed=7); a = eng.run(300)
    eng.reset(seed=7); b = eng.run(300)
    np.testing.assert_array_equal(a, b)


def test_paired_slots_spike_identically_and_flies_differ(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device, n_flies=2, noise_mv=0.15)
    eng.set_drive_hz(None, pops.sensory, 150.0)
    counts = eng.run(300)
    for s in range(1, SLOTS):
        np.testing.assert_array_equal(counts[s], counts[0])
        np.testing.assert_array_equal(counts[SLOTS + s], counts[SLOTS])
    assert (counts[SLOTS] != counts[0]).any()


def test_shuffle_seed_permutes_kc_rows_of_that_fly_only(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device, n_flies=2, shuffle_seeds={1: 3})
    w0, w1 = eng.W0[0].cpu().numpy(), eng.W0[1].cpu().numpy()
    assert not np.array_equal(w0, w1)
    np.testing.assert_array_equal(np.sort(w0.ravel()), np.sort(w1.ravel()))
    np.testing.assert_allclose(w0.sum(0), w1.sum(0), rtol=1e-5) # each MBON keeps its total KC input
```

- [ ] **Step 5: 실패 확인** — Run: `uv run pytest tests/brain/test_engine_mps.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.engine_mps'`.

- [ ] **Step 6: 구현** — `flymon/brain/engine_mps.py`

```python
"""Batched leaky integrate-and-fire swarm on torch (MPS or CPU) over one shared MaleCNS connectome.

B = 4F slots: fly f owns slots 4f..4f+3 (spec 2: four candidate odours per decision). The connectome
is shared; membrane state is per slot; the plastic KC->MBON block is per fly (hive: a fly's four
slots read one weight matrix). Two propagation paths per step:
  shared CSC (every edge except KC->MBON): out-edges of the arrived (slot, neuron) pairs are
      expanded and scatter-added into the [B*N] conductance vector;
  plastic block: arrived KC spikes form S[F, 4, n_kc] and torch.bmm(S, W[F, n_kc, n_mbon]) is
      added to the MBON conductances.
Noise is paired: the membrane jitter [F, N] and the receptor uniforms [F, R] are drawn once per fly
and repeated over its four slots, so slots that see the same odour spike identically. The noise
source is injectable (`NumpyNoise`) so a test can feed the CPU `Engine`'s exact random stream.
Equations, delay line, refractory accounting and the membrane floor are those of engine_cpu.py.
"""
from __future__ import annotations

import numpy as np
import torch

from .circuits import Populations
from .config import Params
from .connectome import Connectome, build_csc
from .engine_cpu import kc_thresholds, tonic_ext

SLOTS = 4


class DeviceNoise:
    """Default noise source: one torch generator on the engine's device."""

    def __init__(self, device: torch.device, seed: int):
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def seed(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def normal(self, f: int, n: int) -> torch.Tensor:
        return torch.randn(f, n, device=self.device, generator=self.gen)

    def uniform(self, f: int, r: int) -> torch.Tensor:
        return torch.rand(f, r, device=self.device, generator=self.gen)

    def get_state(self) -> np.ndarray:
        return self.gen.get_state().cpu().numpy()

    def set_state(self, state) -> None:
        self.gen.set_state(torch.as_tensor(np.asarray(state, dtype=np.uint8)))


def _f32(x: float) -> float:
    """A Python float holding the float32 rounding of x, so torch and numpy multiply by the same constant."""
    return float(np.float32(x))


class SwarmEngine:
    def __init__(self, conn: Connectome, pops: Populations, params: Params, n_flies: int, seed: int = 0,
                 device: str = "mps", noise=None, shuffle_seeds: dict | None = None):
        if n_flies < 1:
            raise ValueError("n_flies must be >= 1")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("torch MPS backend not available: pass device='cpu'")
        self.p, self.conn, self.pops = params, conn, pops
        self.F, self.B, self.N = int(n_flies), SLOTS * int(n_flies), conn.N
        self.device = torch.device(device)
        dev = self.device
        csc = build_csc(conn, params, pops.apl)
        pre = csc.pre_of_edge()
        is_kc = np.zeros(self.N, bool); is_kc[pops.kc] = True
        is_mb = np.zeros(self.N, bool); is_mb[pops.mbon] = True
        plastic = is_kc[pre] & is_mb[csc.tgt]
        if not plastic.any():
            raise ValueError("no plastic KC->MBON edges: check min_weight and the KC/MBON populations")
        if (csc.w[plastic] <= 0).any():
            raise ValueError("non-positive KC->MBON weight: Kenyon cells are cholinergic, so a "
                             "w <= 0 means a sign/transmitter problem in the connectome")
        self.n_plastic = int(plastic.sum())
        keep = ~plastic
        counts = np.bincount(pre[keep], minlength=self.N)
        self.ptr = torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]).astype(np.int64), device=dev)
        self.tgt = torch.as_tensor(csc.tgt[keep].astype(np.int64), device=dev)
        self.wgt = torch.as_tensor(csc.w[keep].astype(np.float32), device=dev)
        self.deg = self.ptr[1:] - self.ptr[:-1]
        # dense plastic block, per fly
        self.n_kc, self.n_mb = len(pops.kc), len(pops.mbon)
        kc_local = np.full(self.N, -1, np.int64); kc_local[pops.kc] = np.arange(self.n_kc)
        mb_local = np.full(self.N, -1, np.int64); mb_local[pops.mbon] = np.arange(self.n_mb)
        w0 = np.zeros((self.n_kc, self.n_mb), np.float32)
        np.add.at(w0, (kc_local[pre[plastic]], mb_local[csc.tgt[plastic]]), csc.w[plastic])
        w0f = np.repeat(w0[None], self.F, axis=0)
        for fly, s in (shuffle_seeds or {}).items():      # C-shuf: permute the KC rows of that fly's block
            w0f[fly] = w0[np.random.default_rng(int(s)).permutation(self.n_kc)]
        self.W0 = torch.as_tensor(w0f, device=dev)
        self.W = self.W0.clone()
        self.kc_local = torch.as_tensor(kc_local, device=dev)
        self.mb_local = mb_local
        self.mbon_idx = torch.as_tensor(pops.mbon.astype(np.int64), device=dev)
        # thresholds, tonic drive, receptors (same arrays as the CPU engine)
        self.v_th = torch.as_tensor(kc_thresholds(conn, pops, params), device=dev)
        self.ext0 = torch.as_tensor(tonic_ext(pops, params, self.N), device=dev)
        self.receptor_idx = pops.sensory.astype(np.int64)
        self.R = int(self.receptor_idx.size)
        self.ri = torch.as_tensor(self.receptor_idx, device=dev)
        recep_local = np.full(self.N, -1, np.int64); recep_local[self.receptor_idx] = np.arange(self.R)
        self.recep_local = recep_local
        self._type_local = {t: torch.as_tensor(recep_local[cells], device=dev) for t, cells in pops.receptor_types.items()}
        self._orn_local = torch.as_tensor(np.concatenate([recep_local[c] for c in pops.receptor_types.values()])
                                          if pops.receptor_types else np.zeros(0, np.int64), device=dev)
        # float32-rounded constants so torch and numpy multiply by identical values
        p = params
        self._dt_m, self._syn_keep = _f32(p.dt / p.tau_m), _f32(1.0 - p.dt / p.tau_syn)
        self._noise_mv, self._dt_s = _f32(p.noise_mv), _f32(p.dt / 1000.0)
        self.ext = self.ext0.expand(self.B, self.N).clone()
        self.drive_hz = torch.zeros(self.B, self.R, dtype=torch.float32, device=dev)
        self.noise = noise if noise is not None else DeviceNoise(dev, seed)
        self.on_step = None
        self._seed = seed
        self.reset(seed)

    # ---- state -------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Zero membrane state and the delay line, reseed the noise source. Drives and weights persist (as in Engine)."""
        if seed is not None:
            self._seed = seed
        self.noise.seed(self._seed)
        dev = self.device
        self.v = torch.zeros(self.B, self.N, dtype=torch.float32, device=dev)
        self.g = torch.zeros(self.B, self.N, dtype=torch.float32, device=dev)
        self.refrac = torch.zeros(self.B, self.N, dtype=torch.int32, device=dev)
        e = torch.zeros(0, dtype=torch.int64, device=dev)
        self.delay = [(e, e) for _ in range(self.p.dly_steps())]
        self.last = (e, e)
        self.t_ms = 0.0

    def _rows(self, slots) -> torch.Tensor:
        if slots is None:
            return torch.arange(self.B, device=self.device)
        return torch.as_tensor(np.atleast_1d(np.asarray(slots, np.int64)), device=self.device)

    def _cols(self, idx) -> torch.Tensor:
        if torch.is_tensor(idx):
            return idx.to(self.device, torch.int64)
        return torch.as_tensor(np.asarray(idx, np.int64), device=self.device)

    def set_ext(self, slots, idx, mv: float) -> None:
        rows, cols = self._rows(slots), self._cols(idx)
        self.ext[rows[:, None], cols[None, :]] = float(mv)

    def clear_ext(self, slots=None) -> None:
        self.ext[self._rows(slots)] = self.ext0

    def set_drive_hz(self, slots, idx, hz: float) -> None:
        loc = self.recep_local[np.asarray(idx, np.int64)]
        if (loc < 0).any():
            raise ValueError("set_drive_hz: index is not a receptor (sensory) neuron")
        rows = self._rows(slots)
        cols = torch.as_tensor(loc, device=self.device)
        self.drive_hz[rows[:, None], cols[None, :]] = float(hz)

    def clear_drive(self, slots=None) -> None:
        self.drive_hz[self._rows(slots)] = 0.0

    def present(self, slot: int, odor: dict, strength: float) -> None:
        """stimuli.present for one slot: zero every olfactory receptor type, then drive the odour's types."""
        self.drive_hz[slot, self._orn_local] = 0.0
        for t, s in odor.items():
            self.drive_hz[slot, self._type_local[t]] = float(np.float32(self.p.max_rate_hz * strength * s))

    # ---- dynamics ----------------------------------------------------------------------------
    def propagate(self, b: torch.Tensor, n: torch.Tensor) -> None:
        """Add the signed mV of every out-edge of the arrived (slot, neuron) spikes to g."""
        d = self.deg[n]
        total = int(d.sum())
        if total:
            bb = torch.repeat_interleave(b, d)
            base = torch.repeat_interleave(self.ptr[n], d)
            off = torch.arange(total, device=self.device) - torch.repeat_interleave(torch.cumsum(d, 0) - d, d)
            e = base + off
            self.g.view(-1).index_add_(0, bb * self.N + self.tgt[e], self.wgt[e])
        kl = self.kc_local[n]
        sel = kl >= 0
        if bool(sel.any()):
            s = torch.zeros(self.B, self.n_kc, dtype=torch.float32, device=self.device)
            s[b[sel], kl[sel]] = 1.0
            out = torch.bmm(s.view(self.F, SLOTS, self.n_kc), self.W)
            self.g[:, self.mbon_idx] += out.reshape(self.B, self.n_mb)

    def step(self):
        p = self.p
        b, n = self.delay.pop(0)
        if n.numel():
            self.propagate(b, n)
        dv = (-self.v + self.g + self.ext) * self._dt_m
        if p.noise_mv:
            noise = torch.as_tensor(self.noise.normal(self.F, self.N), dtype=torch.float32, device=self.device)
            dv += noise.repeat_interleave(SLOTS, 0) * self._noise_mv
        free = self.refrac <= 0
        self.v = torch.where(free, self.v + dv, self.v)
        self.refrac = torch.where(free, self.refrac, self.refrac - 1)
        self.v.clamp_(min=-p.v_thresh)
        self.g *= self._syn_keep
        spk = (self.v >= self.v_th) & free
        u = torch.as_tensor(self.noise.uniform(self.F, self.R), dtype=torch.float32, device=self.device)
        pois = (u.repeat_interleave(SLOTS, 0) < self.drive_hz * self._dt_s) & free[:, self.ri]
        spk[:, self.ri] = pois
        idx = spk.nonzero()
        b, n = idx[:, 0], idx[:, 1]
        self.v[b, n] = p.v_reset
        self.refrac[b, n] = p.refrac_steps()
        self.last = (b, n)
        self.delay.append((b, n))
        self.t_ms += p.dt
        if self.on_step is not None:
            self.on_step(self, b, n)
        return b, n

    def run(self, ms: float, idx=None) -> np.ndarray:
        """Step for `ms` and return spike counts [B, N] (or [B, len(idx)]) as numpy int32."""
        counts = torch.zeros(self.B, self.N, dtype=torch.int32, device=self.device)
        for _ in range(int(round(ms / self.p.dt))):
            b, n = self.step()
            counts[b, n] += 1
        if idx is not None:
            counts = counts[:, torch.as_tensor(np.asarray(idx, np.int64), device=self.device)]
        return counts.cpu().numpy()
```

- [ ] **Step 7: 통과 확인** — Run: `uv run pytest tests/brain/test_engine_mps.py -v` — Expected: 9 테스트 × 장치 수 PASS(포아송 테스트는 5,000스텝이라 장치당 수 초).

- [ ] **Step 8: 전체 스위트** — Run: `uv run pytest -q` — Expected: 기존 171 passed + 새 테스트, 1 skipped, 1 xfailed, 경고 없음.

- [ ] **Step 9: 커밋**

```bash
git add pyproject.toml uv.lock flymon/brain/engine_cpu.py flymon/brain/engine_mps.py tests/conftest.py tests/brain/test_engine_mps.py
git commit -m "feat(brain): SwarmEngine batched LIF on torch (shared CSC scatter-add + per-fly dense KC->MBON), paired noise; engine_cpu thresholds/tonic drive as module functions"
```

---
### Task 2: 잡음 주입과 CPU 래스터 동일성, 상태 왕복

**Files:**
- Modify: `flymon/brain/engine_mps.py`(`NumpyNoise` 클래스, `SwarmEngine.state`/`load_state`)
- Test: `tests/brain/test_engine_mps.py`(테스트 2개 추가)

**Interfaces:**
- Consumes Task 1의 `SwarmEngine`, `DeviceNoise` 프로토콜(`seed/normal/uniform/get_state/set_state`).
- Produces `NumpyNoise(seed)`: 같은 프로토콜, numpy `Generator` 하나를 `standard_normal((F, N), float32)` → `random((F, R))` 순서로 소비한다(= `Engine.step`의 소비 순서; `noise_mv == 0`이면 엔진이 `normal`을 부르지 않으므로 순서가 유지된다). `get_state()`는 `bit_generator.state` dict.
- Produces `SwarmEngine.state() -> {"W", "W0", "noise_state", "t_ms", "seed"}`(numpy/기본형), `load_state(d)`(형태 검사 후 복원). 막 상태는 포함하지 않는다 — M3의 체크포인트는 배틀 사이(프레젠테이션 사이)에 찍고 각 프레젠테이션은 `reset(seed)`로 시작한다(스펙 3.6).

- [ ] **Step 1: 실패하는 테스트** — `tests/brain/test_engine_mps.py`에 추가(import 줄을 `from flymon.brain.engine_mps import SLOTS, NumpyNoise, SwarmEngine`으로 바꾼다).

```python
def test_slot0_raster_equals_cpu_engine_with_the_same_noise(synthetic_connectome, device):
    """Exact equality needs weights that are exact in binary: 0.25 mV/synapse, APL scale 0.125, no
    hemisphere scaling. Then float64 (numpy bincount) and float32 (index_add_/bmm) sums agree bit for bit."""
    c = synthetic_connectome()
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, mv_per_synapse=0.25, apl_scale=0.125,
               kc_thresh=1.0)
    pops = Populations.from_connectome(c)
    cpu = Engine(c, pops, p, seed=11)
    sw = SwarmEngine(c, pops, p, n_flies=1, seed=11, device=device, noise=NumpyNoise(11))
    cpu.set_drive_hz(pops.sensory, 150.0)
    sw.set_drive_hz(0, pops.sensory, 150.0)
    total = 0
    for t in range(300):
        a = cpu.step()
        b, n = sw.step()
        n0 = n[b == 0]                        # the undriven slots 1-3 still spike (MBON hold + noise)
        assert sorted(n0.tolist()) == sorted(a.tolist()), f"rasters diverge at step {t}"
        total += a.size
    assert total > 200                        # the network actually spiked


def test_state_roundtrip_continues_identically(synthetic_connectome, device):
    eng, c, pops = _swarm(synthetic_connectome, device, n_flies=2, noise_mv=0.15)
    eng.set_drive_hz(None, pops.sensory, 150.0)
    eng.run(100)
    eng.W[1] *= 0.5                           # something to carry across
    s = eng.state()
    eng.reset(seed=5); a = eng.run(200)
    eng2, _, _ = _swarm(synthetic_connectome, device, n_flies=2, noise_mv=0.15, seed=99)
    eng2.set_drive_hz(None, pops.sensory, 150.0)
    eng2.load_state(s)
    eng2.reset(seed=5); b = eng2.run(200)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(eng2.W.cpu().numpy(), eng.W.cpu().numpy())
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/brain/test_engine_mps.py -q -k "raster or roundtrip"` — Expected: ImportError(`NumpyNoise`).

- [ ] **Step 3: 구현** — `engine_mps.py`의 `DeviceNoise` 클래스 뒤(`_f32` 앞)에 추가:

```python
class NumpyNoise:
    """Test noise source: one numpy Generator consumed in exactly the order engine_cpu.Engine.step
    consumes its own (standard_normal(N, float32) when noise_mv != 0, then random(R))."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def normal(self, f: int, n: int) -> np.ndarray:
        return self.rng.standard_normal((f, n), dtype=np.float32)

    def uniform(self, f: int, r: int) -> np.ndarray:
        return self.rng.random((f, r))

    def get_state(self):
        return self.rng.bit_generator.state

    def set_state(self, state) -> None:
        self.rng.bit_generator.state = state
```

`SwarmEngine` 클래스 끝(`run` 뒤)에 추가:

```python
    # ---- checkpoint --------------------------------------------------------------------------
    def state(self) -> dict:
        return {"W": self.W.cpu().numpy(), "W0": self.W0.cpu().numpy(), "noise_state": self.noise.get_state(),
                "t_ms": float(self.t_ms), "seed": int(self._seed)}

    def load_state(self, d: dict) -> None:
        w, w0 = np.asarray(d["W"], np.float32), np.asarray(d["W0"], np.float32)
        if w.shape != (self.F, self.n_kc, self.n_mb) or w0.shape != w.shape:
            raise ValueError(f"state W shape {w.shape} != {(self.F, self.n_kc, self.n_mb)}")
        self.W = torch.as_tensor(w, device=self.device)
        self.W0 = torch.as_tensor(w0, device=self.device)
        self.noise.set_state(d["noise_state"])
        self.t_ms = float(d["t_ms"])
        self._seed = int(d["seed"])
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/brain/test_engine_mps.py -v` — Expected: 11 × 장치 수 PASS. 래스터 테스트가 어느 스텝에서 갈라지면 메시지에 스텝이 찍힌다 — 그 경우 상수 반올림(`_f32`)이나 연산 순서가 CPU와 달라진 것이므로 테스트 파라미터를 바꾸지 말고 원인을 보고한다.

- [ ] **Step 5: 커밋**

```bash
git add flymon/brain/engine_mps.py tests/brain/test_engine_mps.py
git commit -m "feat(brain): injectable numpy noise (exact CPU raster equality on the synthetic net) and swarm state round-trip"
```

---
### Task 3: `SwarmPlasticity` — 외적 규칙, 슬롯 0 자취, 마리별 플래그

**Files:**
- Create: `flymon/brain/plasticity_mps.py`
- Test: `tests/brain/test_plasticity_mps.py`

**Interfaces:**
- Consumes `SwarmEngine`(`W`, `W0`, `kc_local`, `mb_local`, `ext`, `ext0`, `set_ext`, `on_step`, `device`, `F`, `B`, `N`, `n_kc`, `n_mb`), `circuits.compartments()`의 `Compartment(cells, w_mbon, core)`.
- Produces `SwarmPlasticity(engine, pops, comps)`: 속성 `kc_trace[F, n_kc]`, `da[F, n_mb]`, `da_base[F, n_mb]`, `enabled`(numpy bool [F]); 메서드 `drive_dan(fly, type_name, mv)`(마리의 슬롯 0에만), `quiet_dan(fly=None)`, `on_step(engine, b, n)`, `weights_frac(fly=None)`, `weights_frac_by_mbon_set(mbon_idx, fly=None)`, `recover_pulse(fly=None)`, `reset_traces()`, `reset_weights(fly=None)`, `set_enabled(on, fly=None)`(`on`은 bool, 또는 fly=None일 때 [F] bool 배열), `state()`/`load_state(d)`(enabled·자취). 생성자가 `engine.on_step`을 잡는다(CPU `Plasticity`와 같다).
- 규칙: CPU와 같은 `w *= 1 − lr·tanh(kc_trace·kc_scale × phasic·da_scale)`, 바닥 `min_weight_frac·W0`, 마스크(W0 = 0인 칸은 0 유지). 도파민 자취는 DAN 타입별 발화 수를 `bincount`로 세어 `[F, T] @ wt[T, n_mb]` 한 번에 더한다(CPU의 타입별 루프와 같은 값, float32 합 순서만 다름). 켜진 마리가 없으면 갱신 계산을 건너뛴다(결정 단계 비용 절감, 부록 C.4).

- [ ] **Step 1: 실패하는 테스트** — `tests/brain/test_plasticity_mps.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.engine_mps import SLOTS, NumpyNoise, SwarmEngine
from flymon.brain.plasticity_mps import SwarmPlasticity


def _params(**kw):
    return Params(**{"noise_mv": 0.0, "min_weight": 1, "balance_hemispheres": False, "mbon_hold_frac": 0.0, **kw})


def _swarm(synthetic_connectome, device, n_flies=1, seed=1, noise=None, shuffle_seeds=None, **kw):
    c = synthetic_connectome()
    p = _params(**kw)
    pops = Populations.from_connectome(c)
    eng = SwarmEngine(c, pops, p, n_flies=n_flies, seed=seed, device=device, noise=noise, shuffle_seeds=shuffle_seeds)
    return eng, c, pops


def _setup(synthetic_connectome, device, n_flies=2, **kw):
    eng, c, pops = _swarm(synthetic_connectome, device, n_flies=n_flies, seed=3, **{"learn_rate": 0.05, **kw})
    comps = compartments(c, pops, eng.p.core_frac)
    pl = SwarmPlasticity(eng, pops, comps)
    return c, pops, eng, pl, comps


def _drive_kcs(eng, pops, fly=0, k=10, mv=60.0):
    eng.set_ext(SLOTS * fly, pops.kc[:k], mv)


def test_no_change_without_dopamine(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device)
    _drive_kcs(eng, pops)
    eng.run(300)
    assert pl.weights_frac() == pytest.approx(1.0)


def test_no_change_without_kc_activity(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device)
    pl.drive_dan(0, "PAM08", 70.0)
    eng.run(300)
    assert pl.weights_frac() == pytest.approx(1.0)


def test_coincidence_depresses_only_taught_compartment_of_that_fly(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device)
    _drive_kcs(eng, pops, fly=0)
    pl.drive_dan(0, "PAM08", 70.0)
    eng.run(400)
    assert pl.weights_frac_by_mbon_set(comps["PAM08"].core, fly=0) < 0.99
    assert pl.weights_frac_by_mbon_set(comps["PPL105"].core, fly=0) == pytest.approx(1.0)
    assert pl.weights_frac(fly=1) == pytest.approx(1.0)
    np.testing.assert_array_equal(eng.W[1].cpu().numpy(), eng.W0[1].cpu().numpy())


def test_traces_only_from_slot0(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device)
    eng.set_ext(1, pops.kc[:10], 60.0)         # slot 1 of fly 0: KCs fire, but the trace must not see them
    pl.drive_dan(0, "PAM08", 70.0)
    eng.run(400)
    assert float(pl.kc_trace.max()) == 0.0
    assert pl.weights_frac() == pytest.approx(1.0)


def test_floor_and_per_fly_disable(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device, learn_rate=0.5, min_weight_frac=0.2)
    _drive_kcs(eng, pops, fly=0, k=40, mv=80.0)
    pl.drive_dan(0, "PAM08", 70.0)
    eng.run(2000)
    w, w0 = eng.W[0].cpu().numpy(), eng.W0[0].cpu().numpy()
    m = w0 > 0
    assert (w[m] / w0[m] >= 0.2 - 1e-6).all()
    assert pl.weights_frac_by_mbon_set(comps["PAM08"].core, fly=0) == pytest.approx(0.2, abs=1e-3)
    assert pl.weights_frac(fly=0) < 0.7
    pl.set_enabled(False, fly=0)
    before = eng.W.cpu().numpy().copy()
    eng.run(200)
    np.testing.assert_array_equal(eng.W.cpu().numpy(), before)
    pl.set_enabled(True, fly=0)
    pl.set_enabled(False, fly=1)
    _drive_kcs(eng, pops, fly=1, k=40, mv=80.0)
    pl.drive_dan(1, "PAM08", 70.0)
    eng.run(400)
    assert pl.weights_frac(fly=1) == pytest.approx(1.0)           # fly 1 disabled: no learning despite drive


def test_recover_pulse_moves_toward_w0(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device, learn_rate=0.5, recovery_per_pulse=0.5)
    _drive_kcs(eng, pops, fly=0, k=40, mv=80.0)
    pl.drive_dan(0, "PAM08", 70.0)
    eng.run(1000)
    f0 = pl.weights_frac(fly=0)
    pl.recover_pulse(fly=0)
    f1 = pl.weights_frac(fly=0)
    assert f1 == pytest.approx(f0 + (1.0 - f0) * 0.5, abs=1e-6)
    assert pl.weights_frac(fly=1) == pytest.approx(1.0)


def test_reset_traces_and_reset_weights(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device, learn_rate=0.5)
    _drive_kcs(eng, pops, fly=0, k=40, mv=80.0)
    pl.drive_dan(0, "PPL105", 70.0)
    eng.run(500)
    assert float(pl.kc_trace.max()) > 0 and float(pl.da.max()) > 0
    assert pl.weights_frac(fly=0) < 1.0
    w = eng.W.cpu().numpy().copy()
    pl.reset_traces()
    assert float(pl.kc_trace.max()) == 0 and float(pl.da.max()) == 0 and float(pl.da_base.max()) == 0
    np.testing.assert_array_equal(eng.W.cpu().numpy(), w)
    pl.reset_weights(fly=0)
    assert pl.weights_frac(fly=0) == pytest.approx(1.0)


def test_quiet_dan_restores_tonic_drive(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device, mbon_hold_frac=0.85)
    cells = comps["PAM08"].cells
    pl.drive_dan(0, "PAM08", 70.0)
    assert (eng.ext[0, cells].cpu().numpy() == 70.0).all()
    pl.quiet_dan(0)
    np.testing.assert_array_equal(eng.ext[0, cells].cpu().numpy(), eng.ext0[cells].cpu().numpy())
    assert (eng.ext[1].cpu().numpy() == eng.ext0.cpu().numpy()).all()


def test_plasticity_state_roundtrip(synthetic_connectome, device):
    c, pops, eng, pl, comps = _setup(synthetic_connectome, device, learn_rate=0.5)
    pl.set_enabled(False, fly=1)
    _drive_kcs(eng, pops, fly=0, k=40, mv=80.0)
    pl.drive_dan(0, "PAM08", 70.0)
    eng.run(300)
    s = pl.state()
    c2, pops2, eng2, pl2, _ = _setup(synthetic_connectome, device, learn_rate=0.5)
    pl2.load_state(s)
    assert pl2.enabled.tolist() == [True, False]
    np.testing.assert_array_equal(pl2.kc_trace.cpu().numpy(), pl.kc_trace.cpu().numpy())
    np.testing.assert_array_equal(pl2.da_base.cpu().numpy(), pl.da_base.cpu().numpy())


def test_plasticity_tracks_cpu_rule_on_a_deterministic_run(synthetic_connectome, device):
    """Noise off + binary-exact weights: rasters are identical, so traces and the learned block must
    agree with the CPU Plasticity up to float32 summation order."""
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False, mv_per_synapse=0.25, apl_scale=0.125,
               learn_rate=0.05)
    pops = Populations.from_connectome(c)
    comps = compartments(c, pops, p.core_frac)
    cpu = Engine(c, pops, p, seed=2); cpl = Plasticity(cpu, pops, comps)
    sw = SwarmEngine(c, pops, p, n_flies=1, seed=2, device=device, noise=NumpyNoise(2)); spl = SwarmPlasticity(sw, pops, comps)
    cpu.set_ext(pops.kc[:10], 60.0); sw.set_ext(0, pops.kc[:10], 60.0)
    cpl.drive_dan("PAM08", 70.0); spl.drive_dan(0, "PAM08", 70.0)
    for t in range(400):
        a = cpu.step(); b, n = sw.step()
        assert sorted(n.tolist()) == sorted(a.tolist()), f"step {t}"
    np.testing.assert_allclose(spl.kc_trace[0].cpu().numpy(), cpl.kc_trace, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(spl.da[0].cpu().numpy(), cpl.da, rtol=1e-5, atol=1e-6)
    assert cpl.weights_frac() < 0.99
    assert spl.weights_frac() == pytest.approx(cpl.weights_frac(), rel=1e-4)
    pre, post = cpu.csc.pre_of_edge()[cpl.edges], cpu.csc.tgt[cpl.edges]
    kl = np.full(c.N, -1); kl[pops.kc] = np.arange(len(pops.kc))
    ml = np.full(c.N, -1); ml[pops.mbon] = np.arange(len(pops.mbon))
    np.testing.assert_allclose(sw.W[0].cpu().numpy()[kl[pre], ml[post]], cpu.csc.w[cpl.edges], rtol=1e-4)
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/brain/test_plasticity_mps.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.plasticity_mps'`.

- [ ] **Step 3: 구현** — `flymon/brain/plasticity_mps.py`

```python
"""Dopamine-gated depression of the per-fly KC->MBON block (three-factor rule), compartment by DAN type.

Same rule as plasticity.py in outer-product form: per fly, kc_trace [n_kc] and the phasic dopamine
[n_mbon] at the core MBONs of each DAN type multiply into a [n_kc, n_mbon] coincidence and the block
W[f] *= 1 - lr * tanh(coincidence), floored at min_weight_frac * W0[f]. Traces accumulate from each
fly's slot 0 only (the reinforcement slot, spec 2 step 5). Plasticity, DAN drive and recovery are
per fly so control arms (C-off, C-PAM, C-shuf) share one swarm.
"""
from __future__ import annotations

import numpy as np
import torch

from .circuits import Populations
from .engine_mps import SLOTS, SwarmEngine


class SwarmPlasticity:
    def __init__(self, engine: SwarmEngine, pops: Populations, comps: dict):
        self.eng, self.pops, self.p = engine, pops, engine.p
        dev, F = engine.device, engine.F
        self.type_names = list(comps)
        dan_type = np.full(engine.N, -1, np.int64)
        n_cells = np.zeros(len(comps), np.float32)
        wt = np.zeros((len(comps), engine.n_mb), np.float32)
        for k, (name, cp) in enumerate(comps.items()):
            dan_type[cp.cells] = k
            n_cells[k] = len(cp.cells)
            wt[k, engine.mb_local[cp.core]] = cp.w_mbon[cp.core]
        self.T = len(comps)
        self.dan_type = torch.as_tensor(dan_type, device=dev)
        self.n_cells = torch.as_tensor(n_cells, device=dev)
        self.wt = torch.as_tensor(wt, device=dev)                       # [T, n_mb] core-restricted weights
        self.cells = {name: torch.as_tensor(cp.cells.astype(np.int64), device=dev) for name, cp in comps.items()}
        self.kc_trace = torch.zeros(F, engine.n_kc, dtype=torch.float32, device=dev)
        self.da = torch.zeros(F, engine.n_mb, dtype=torch.float32, device=dev)
        self.da_base = torch.zeros(F, engine.n_mb, dtype=torch.float32, device=dev)
        self.enabled = np.ones(F, bool)
        self._enabled_t = torch.ones(F, dtype=torch.bool, device=dev)
        self._any_enabled = True
        self.slot0 = torch.arange(0, engine.B, SLOTS, device=dev)
        p = self.p
        self._kc_decay, self._kc_inc = float(np.float32(1.0 - p.dt / p.kc_trace_ms)), float(np.float32(p.dt / p.kc_trace_ms))
        self._da_decay, self._da_inc = float(np.float32(1.0 - p.dt / p.da_trace_ms)), float(np.float32(p.dt / p.da_trace_ms))
        self._base_rate = float(np.float32(p.dt / p.da_baseline_ms))
        engine.on_step = self.on_step

    # ---- dopamine drive (slot 0 of a fly) ---------------------------------------------------
    def drive_dan(self, fly: int, type_name: str, mv: float) -> None:
        self.eng.set_ext(SLOTS * int(fly), self.cells[type_name], mv)

    def quiet_dan(self, fly: int | None = None) -> None:
        rows = self.slot0 if fly is None else self.slot0[int(fly):int(fly) + 1]
        for cells in self.cells.values():
            self.eng.ext[rows[:, None], cells[None, :]] = self.eng.ext0[cells]

    # ---- rule ----------------------------------------------------------------------------
    def on_step(self, engine: SwarmEngine, b: torch.Tensor, n: torch.Tensor) -> None:
        p, F = self.p, engine.F
        self.kc_trace *= self._kc_decay
        s0 = (b % SLOTS) == 0
        kl = engine.kc_local[n]
        sel = (kl >= 0) & s0
        if bool(sel.any()):
            self.kc_trace.view(-1).index_add_(0, (b[sel] // SLOTS) * engine.n_kc + kl[sel],
                                              torch.full((int(sel.sum()),), self._kc_inc, device=engine.device))
        self.da *= self._da_decay
        tid = self.dan_type[n]
        dsel = (tid >= 0) & s0
        if bool(dsel.any()):
            key = (b[dsel] // SLOTS) * self.T + tid[dsel]
            cnt = torch.bincount(key, minlength=F * self.T).to(torch.float32).view(F, self.T)
            self.da += ((cnt / self.n_cells) @ self.wt) * self._da_inc
        self.da_base += (self.da - self.da_base) * self._base_rate
        if not self._any_enabled:
            return
        phasic = torch.clamp(self.da - self.da_base, min=0.0)
        if not (bool((phasic > 0).any()) and bool((self.kc_trace > 0).any())):
            return
        coincide = (self.kc_trace * p.kc_trace_scale)[:, :, None] * (phasic * p.da_trace_scale)[:, None, :]
        fac = 1.0 - p.learn_rate * torch.tanh(coincide)
        fac = torch.where(self._enabled_t[:, None, None], fac, torch.ones_like(fac))
        w = engine.W
        w.mul_(fac)
        torch.maximum(w, engine.W0 * p.min_weight_frac, out=w)

    # ---- bookkeeping ---------------------------------------------------------------------
    def _sel(self, fly):
        return slice(None) if fly is None else slice(int(fly), int(fly) + 1)

    def weights_frac(self, fly: int | None = None) -> float:
        w, w0 = self.eng.W[self._sel(fly)], self.eng.W0[self._sel(fly)]
        m = w0 > 0
        if not bool(m.any()):
            raise ValueError("no plastic edges selected")
        return float((w[m] / w0[m]).mean())

    def weights_frac_by_mbon_set(self, mbon_idx, fly: int | None = None) -> float:
        cols = self.eng.mb_local[np.asarray(mbon_idx, np.int64)]
        cols = torch.as_tensor(cols[cols >= 0], device=self.eng.device)
        w, w0 = self.eng.W[self._sel(fly)][:, :, cols], self.eng.W0[self._sel(fly)][:, :, cols]
        m = w0 > 0
        if not bool(m.any()):
            raise ValueError("no plastic edges selected")
        return float((w[m] / w0[m]).mean())

    def recover_pulse(self, fly: int | None = None) -> None:
        r = self.p.recovery_per_pulse
        if r <= 0:
            return
        s = self._sel(fly)
        self.eng.W[s] += (self.eng.W0[s] - self.eng.W[s]) * float(np.float32(r))

    def reset_traces(self) -> None:
        self.kc_trace.zero_(); self.da.zero_(); self.da_base.zero_()

    def reset_weights(self, fly: int | None = None) -> None:
        s = self._sel(fly)
        self.eng.W[s] = self.eng.W0[s]
        self.reset_traces()

    def set_enabled(self, on, fly: int | None = None) -> None:
        """`on` is a bool, or with fly=None also a [F] bool array (one flag per fly)."""
        if fly is None:
            self.enabled[:] = np.asarray(on, bool)
        else:
            self.enabled[int(fly)] = bool(on)
        self._enabled_t = torch.as_tensor(self.enabled, device=self.eng.device)
        self._any_enabled = bool(self.enabled.any())

    def state(self) -> dict:
        return {"enabled": self.enabled.copy(), "kc_trace": self.kc_trace.cpu().numpy(),
                "da": self.da.cpu().numpy(), "da_base": self.da_base.cpu().numpy()}

    def load_state(self, d: dict) -> None:
        dev = self.eng.device
        self.enabled = np.asarray(d["enabled"], bool).copy()
        self._enabled_t = torch.as_tensor(self.enabled, device=dev)
        self._any_enabled = bool(self.enabled.any())
        self.kc_trace = torch.as_tensor(np.asarray(d["kc_trace"], np.float32), device=dev)
        self.da = torch.as_tensor(np.asarray(d["da"], np.float32), device=dev)
        self.da_base = torch.as_tensor(np.asarray(d["da_base"], np.float32), device=dev)
```

- [ ] **Step 4: 통과 확인** — Run: `uv run pytest tests/brain/test_plasticity_mps.py -v` — Expected: 11 × 장치 수 PASS. `test_plasticity_tracks_cpu_rule_on_a_deterministic_run`이 자취에서 어긋나면 상수(`_kc_inc` 등)의 float32 반올림을 먼저 본다.

- [ ] **Step 5: 전체 스위트** — Run: `uv run pytest -q` — Expected: 전부 PASS, 경고 없음.

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/plasticity_mps.py tests/brain/test_plasticity_mps.py
git commit -m "feat(brain): SwarmPlasticity — per-fly outer-product three-factor rule from slot 0, per-fly enable/DAN drive/recovery"
```

---
### Task 4: 스웜 위의 M0 측정 — `conditioning_mps.py`, `channel_specific_seeds` 이동

**Files:**
- Create: `flymon/brain/conditioning_mps.py`
- Modify: `flymon/brain/conditioning.py`(`channel_specific_seeds` 추가), `scripts/write_m0_summary.py`(로컬 정의 삭제, import), `tests/brain/test_conditioning.py`(`_channel_specific_seeds` 삭제, import)
- Test: `tests/brain/test_conditioning_mps.py`

**Interfaces:**
- Consumes `SwarmEngine`, `SwarmPlasticity`, `conditioning.{D, D_graded, Readout, arms, summarise}`, `measure.{jaccard, chance_jaccard}`.
- Produces `conditioning.channel_specific_seeds(per_seed) -> int`(`scripts/write_m0_summary.py`의 정의 그대로).
- Produces `conditioning_mps.PROBE_SEED = 100`, `TRAIN_SEED_BASE = 1_000_000`, `fly_arms(n_seeds, punish_type, reward_type) -> [(seed, arm)]`(F = 5·n_seeds, 시드 바깥·팔 안쪽), `present_slot0(eng, odor, strength)`, `swarm_probe(eng, pl, ro, odor, strength, seed, settle_ms, read_ms) -> {"A": [F], "P": [F]}`, `swarm_train_block(eng, pl, cs_plus, cs_minus, strength, dan_plus, dan_minus, plastic, trials, present_ms, gap_ms, settle_ms)`, `swarm_reversal_test(eng, pl, ro, cs_plus, cs_minus, strength, n_seeds, punish_type, reward_type, trials, present_ms, gap_ms, settle_ms, read_ms) -> per_seed[seed][arm]`(`run_arm`과 같은 키 + `"fly"`; `summarise`에 그대로 넣는다), `swarm_sparsity(eng, pops, odor_a, odor_b, strength, seed, settle_ms, read_ms) -> dict`, `swarm_baseline(eng, pops, seed, ms, sat_hz) -> dict`(`runaway` 블록 포함: `n_over_sat`, `n_kc_over_sat`, `spike_share_over_sat`).
- CPU 프로토콜과의 의도된 차이(모듈 docstring): 프레젠테이션마다 스웜 전체를 한 번 `reset(TRAIN_SEED_BASE + t)`(CPU는 `seed*1000 + t`), 프로브는 항상 `reset(PROBE_SEED)` → `noplast` 마리의 dD는 정확히 0.0.

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

`scripts/write_m0_summary.py`: `_drop`과 `channel_specific_seeds` 정의(`def _drop` 부터 `return n` 까지)를 지우고 상단 import에 `from flymon.brain.conditioning import channel_specific_seeds`를 더한다. `index_flip_ok`, `gate_ok`, `main`은 그대로.

`tests/brain/test_conditioning.py`: `_channel_specific_seeds` 헬퍼(정의 전체)를 지우고 상단 import를 `from flymon.brain.conditioning import (D, D_graded, Readout, arms, channel_specific_seeds, disc, disc_graded, probe, run_arm, train_block)`으로 바꾼 뒤 `test_real_conditioning_channel_specificity`의 호출을 `channel_specific_seeds(d["per_seed"])`로 바꾼다.

Run: `uv run pytest tests/brain/test_conditioning.py tests/test_summary.py -q && uv run python scripts/write_m0_summary.py --help >/dev/null || true` — Expected: 테스트 PASS(실제 데이터 테스트는 `results/m0/conditioning.json`이 있으면 실행된다). `write_m0_summary.py`는 인자를 받지 않으므로 import 오류만 없으면 된다: `uv run python -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('w','scripts/write_m0_summary.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.channel_specific_seeds)"`.

- [ ] **Step 2: 실패하는 테스트** — `tests/brain/test_conditioning_mps.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import Readout, summarise
from flymon.brain.config import Params
from flymon.brain.engine_mps import SwarmEngine
from flymon.brain.plasticity_mps import SwarmPlasticity
from flymon.brain.conditioning_mps import fly_arms, swarm_baseline, swarm_reversal_test, swarm_sparsity


def test_fly_arms_layout():
    fa = fly_arms(2)
    assert len(fa) == 10 and fa[0] == (0, "both") and fa[5] == (1, "both") and fa[4] == (0, "reward_only")


def test_swarm_reversal_flips_sign_and_noplast_is_exactly_zero(synthetic_connectome, device):
    c = synthetic_connectome(disjoint_kc=True)
    # trials 5 / learn_rate 0.1 (not the CPU test's 3 / 0.05): with 10 flies on the tiny synthetic net a
    # weaker protocol leaves one arm's dD quantised to exactly 0 for some device noise streams
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.1, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    comps = compartments(c, pops, p.core_frac)
    n_seeds = 2
    eng = SwarmEngine(c, pops, p, n_flies=5 * n_seeds, seed=0, device=device)
    pl = SwarmPlasticity(eng, pops, comps)
    ro = Readout.from_compartments(comps)
    a = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
    b = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
    per_seed = swarm_reversal_test(eng, pl, ro, a, b, 1.0, n_seeds, trials=5, present_ms=150, gap_ms=20,
                                   settle_ms=50, read_ms=600)
    s = summarise(per_seed)
    assert s["n_seeds"] == 2 and s["noplast_max_abs_dD"] == 0.0
    for seed in range(n_seeds):
        arms = per_seed[seed]
        assert arms["noplast"]["dD"] == 0.0 and arms["noplast"]["weights_frac"] == pytest.approx(1.0)
        assert arms["both"]["dD"] != 0.0
        assert np.sign(arms["both"]["dD"]) == -np.sign(arms["reversed"]["dD"])
        assert arms["both"]["fly"] == seed * 5
    assert s["n_flip"] == n_seeds


def test_swarm_sparsity_and_baseline_shapes(synthetic_connectome, device):
    c = synthetic_connectome(disjoint_kc=True)      # the non-disjoint build leaves KCs silent at this drive
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    eng = SwarmEngine(c, pops, p, n_flies=2, seed=0, device=device)
    a = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
    b = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
    sp = swarm_sparsity(eng, pops, a, b, 1.0, settle_ms=50, read_ms=600)
    assert 0.0 < sp["frac_active_A"] <= 1.0 and 0.0 < sp["frac_active_B"] <= 1.0
    assert len(sp["per_fly"]) == 2 and sp["chance"] >= 0.0 and sp["jaccard"] <= sp["chance"]
    base = swarm_baseline(eng, pops, ms=200)
    assert base["mbon_hz_rest"] >= 0.0 and base["runaway"]["n_kc_over_sat"] >= 0.0
    assert 0.0 <= base["runaway"]["spike_share_over_sat"] <= 1.0
```

- [ ] **Step 3: 실패 확인** — Run: `uv run pytest tests/brain/test_conditioning_mps.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.conditioning_mps'`.

- [ ] **Step 4: 구현** — `flymon/brain/conditioning_mps.py`

```python
"""The M0 measurements on the swarm: sparsity/baseline with odour A and B in paired slots, and the
five-arm conditioning with every (seed, arm) as one fly, all in lockstep.

Differences from the CPU protocol (conditioning.py), by design:
  - one reset per presentation for the whole swarm (seed_base + trial, not seed*1000 + trial);
    flies still see different noise because the paired draw is [F, .];
  - probes reseed the swarm with PROBE_SEED before every presentation, so a no-plasticity fly
    returns exactly its pre-training counts (dD == 0.0), as on the CPU.
Results have the shape conditioning.run_arm returns, so conditioning.summarise applies unchanged.
"""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .conditioning import D, D_graded, Readout, arms as arm_table
from .measure import chance_jaccard, jaccard
from .engine_mps import SLOTS, SwarmEngine
from .plasticity_mps import SwarmPlasticity

PROBE_SEED = 100
TRAIN_SEED_BASE = 1_000_000


def fly_arms(n_seeds: int, punish_type: str = "PPL105", reward_type: str = "PAM08") -> list[tuple[int, str]]:
    """Fly f -> (seed, arm): seeds outer, the five arms inner, so F = 5 * n_seeds."""
    names = list(arm_table(punish_type, reward_type))
    return [(s, a) for s in range(n_seeds) for a in names]


def present_slot0(eng: SwarmEngine, odor, strength: float) -> None:
    eng.clear_drive()
    for f in range(eng.F):
        eng.present(SLOTS * f, odor, strength)


def swarm_probe(eng: SwarmEngine, pl: SwarmPlasticity, ro: Readout, odor, strength: float,
                seed: int = PROBE_SEED, settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    """Spike counts on the approach (A) and avoidance (P) cores of slot 0 of every fly: {"A": [F], "P": [F]}."""
    was = pl.enabled.copy()
    pl.set_enabled(False)
    eng.reset(seed)
    pl.reset_traces()
    pl.quiet_dan()
    present_slot0(eng, odor, strength)
    eng.run(settle_ms)
    counts = eng.run(read_ms, idx=eng.pops.mbon)[::SLOTS]          # [F, n_mbon], slot 0 of each fly
    pl.set_enabled(was)
    return {"A": counts[:, eng.mb_local[ro.a_core]].sum(1).astype(int),
            "P": counts[:, eng.mb_local[ro.p_core]].sum(1).astype(int)}


def swarm_train_block(eng: SwarmEngine, pl: SwarmPlasticity, cs_plus, cs_minus, strength: float,
                      dan_plus: list, dan_minus: list, plastic: np.ndarray, trials: int = 12,
                      present_ms: float = 800.0, gap_ms: float = 200.0, settle_ms: float = 800.0) -> None:
    """conditioning.train_block for the swarm: fly f is paired with dan_plus[f] on CS+ and dan_minus[f]
    on CS- (None = unpaired), and learns only if plastic[f]."""
    p = eng.p
    for t in range(trials):
        for odor, dans in ((cs_plus, dan_plus), (cs_minus, dan_minus)):
            eng.reset(TRAIN_SEED_BASE + t)
            pl.reset_traces()
            pl.quiet_dan()
            present_slot0(eng, odor, strength)
            pl.set_enabled(False)                    # settle: baseline adapts, weights frozen
            eng.run(settle_ms)
            pl.set_enabled(plastic)
            for f, dan in enumerate(dans):
                if dan is not None:
                    pl.drive_dan(f, dan, p.dan_drive_mv)
            eng.run(present_ms)
            pl.quiet_dan()
            eng.clear_drive()
            eng.run(gap_ms)
            for f, dan in enumerate(dans):
                if dan is not None:
                    pl.recover_pulse(fly=f)


def swarm_reversal_test(eng: SwarmEngine, pl: SwarmPlasticity, ro: Readout, cs_plus, cs_minus, strength: float,
                        n_seeds: int, punish_type: str = "PPL105", reward_type: str = "PAM08", trials: int = 12,
                        present_ms: float = 800.0, gap_ms: float = 200.0, settle_ms: float = 800.0,
                        read_ms: float = 600.0) -> dict:
    """All five arms x n_seeds as one swarm (F = 5 * n_seeds). Returns per_seed[seed][arm] in the
    shape of conditioning.run_arm, for conditioning.summarise."""
    fa = fly_arms(n_seeds, punish_type, reward_type)
    if eng.F != len(fa):
        raise ValueError(f"swarm has {eng.F} flies, the protocol needs {len(fa)} (5 arms x {n_seeds} seeds)")
    table = arm_table(punish_type, reward_type)
    dan_plus = [table[a][0] for _, a in fa]
    dan_minus = [table[a][1] for _, a in fa]
    plastic = np.array([table[a][2] for _, a in fa], bool)
    pl.reset_weights()
    pre = {"plus": swarm_probe(eng, pl, ro, cs_plus, strength, read_ms=read_ms),
           "minus": swarm_probe(eng, pl, ro, cs_minus, strength, read_ms=read_ms)}
    swarm_train_block(eng, pl, cs_plus, cs_minus, strength, dan_plus, dan_minus, plastic, trials,
                      present_ms, gap_ms, settle_ms)
    post = {"plus": swarm_probe(eng, pl, ro, cs_plus, strength, read_ms=read_ms),
            "minus": swarm_probe(eng, pl, ro, cs_minus, strength, read_ms=read_ms)}
    per_seed: dict = {}
    for f, (s, arm) in enumerate(fa):
        pick = lambda d: {cs: {"A": int(d[cs]["A"][f]), "P": int(d[cs]["P"][f])} for cs in ("plus", "minus")}
        pre_f, post_f = pick(pre), pick(post)
        norm_a = pre_f["plus"]["A"] + pre_f["minus"]["A"]
        norm_p = pre_f["plus"]["P"] + pre_f["minus"]["P"]
        g_pre = D_graded(pre_f["plus"], pre_f["minus"], norm_a, norm_p)
        g_post = D_graded(post_f["plus"], post_f["minus"], norm_a, norm_p)
        d_pre, d_post = D(ro, pre_f["plus"], pre_f["minus"]), D(ro, post_f["plus"], post_f["minus"])
        per_seed.setdefault(s, {})[arm] = {
            "arm": arm, "seed": s, "fly": f,
            "D_pre": g_pre, "D_post": g_post, "dD": g_post - g_pre,
            "D_pre_disc": d_pre, "D_post_disc": d_post, "dD_disc": d_post - d_pre,
            "counts": {"pre_plus": pre_f["plus"], "pre_minus": pre_f["minus"],
                       "post_plus": post_f["plus"], "post_minus": post_f["minus"]},
            "weights_frac": pl.weights_frac(fly=f),
            "w_frac_a_core": pl.weights_frac_by_mbon_set(ro.a_core, fly=f),
            "w_frac_p_core": pl.weights_frac_by_mbon_set(ro.p_core, fly=f)}
    return per_seed


def swarm_sparsity(eng: SwarmEngine, pops: Populations, odor_a, odor_b, strength: float, seed: int = PROBE_SEED,
                   settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    """measure.kc_sparsity for A and B at once: A on slot 0 and B on slot 1 of every fly (paired noise
    plays the role of the CPU's shared seed per odour pair); per-fly rows and their means."""
    eng.clear_drive()
    for f in range(eng.F):
        eng.present(SLOTS * f, odor_a, strength)
        eng.present(SLOTS * f + 1, odor_b, strength)
    eng.reset(seed)
    eng.run(settle_ms)
    counts = eng.run(read_ms)
    sec = read_ms / 1000.0
    rows = []
    for f in range(eng.F):
        ca, cb = counts[SLOTS * f], counts[SLOTS * f + 1]
        act_a, act_b = ca[pops.kc] > 0, cb[pops.kc] > 0
        fa, fb = float(act_a.mean()), float(act_b.mean())
        rows.append({"frac_active_A": fa, "frac_active_B": fb, "jaccard": jaccard(act_a, act_b),
                     "chance": chance_jaccard(fa, fb),
                     "mbon_hz_A": float(ca[pops.mbon].mean() / sec), "mbon_hz_B": float(cb[pops.mbon].mean() / sec)})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return {**mean, "per_fly": rows, "n_flies": eng.F, "seed": seed, "read_ms": read_ms}


def swarm_baseline(eng: SwarmEngine, pops: Populations, seed: int = PROBE_SEED, ms: float = 3000.0,
                   sat_hz: float = 100.0) -> dict:
    """measure.mbon_baseline on slot 0 of every fly at rest, plus the runaway set (spec 5 / A.5):
    neurons above sat_hz, how many of them are Kenyon cells, and their share of all spikes."""
    eng.clear_drive()
    eng.reset(seed)
    counts = eng.run(ms)[::SLOTS]
    sec = ms / 1000.0
    types = eng.conn.type[pops.mbon]
    rows = []
    for f in range(eng.F):
        c = counts[f]
        hz = c[pops.mbon] / sec
        keep = hz <= sat_hz
        over = (c / sec) > sat_hz
        rows.append({"mbon_hz": float(hz.mean()), "mbon_hz_trimmed": float(hz[keep].mean()) if keep.any() else 0.0,
                     "n_saturated": int((~keep).sum()), "n_types_active": int(len(set(types[hz > 0].tolist()))),
                     "n_over_sat": int(over.sum()), "n_kc_over_sat": int(over[pops.kc].sum()),
                     "spike_share_over_sat": float(c[over].sum() / max(int(c.sum()), 1))})
    trimmed = np.array([r["mbon_hz_trimmed"] for r in rows])
    return {"mbon_hz_rest": float(np.mean([r["mbon_hz"] for r in rows])),
            "mbon_hz_rest_trimmed": float(trimmed.mean()), "mbon_hz_rest_trimmed_sd": float(trimmed.std()),
            "mbon_n_saturated": float(np.mean([r["n_saturated"] for r in rows])),
            "mbon_types_active_rest": float(np.mean([r["n_types_active"] for r in rows])),
            "runaway": {"sat_hz": sat_hz, "n_over_sat": float(np.mean([r["n_over_sat"] for r in rows])),
                        "n_kc_over_sat": float(np.mean([r["n_kc_over_sat"] for r in rows])),
                        "spike_share_over_sat": float(np.mean([r["spike_share_over_sat"] for r in rows]))},
            "per_fly": rows, "n_flies": eng.F, "seed": seed, "rest_ms": ms}
```

- [ ] **Step 5: 통과 확인** — Run: `uv run pytest tests/brain/test_conditioning_mps.py -v` — Expected: 3 × 장치 수 PASS(반전 테스트는 장치당 약 10초).

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/conditioning.py flymon/brain/conditioning_mps.py scripts/write_m0_summary.py tests/brain/test_conditioning.py tests/brain/test_conditioning_mps.py
git commit -m "feat(brain): M0 measurements on the swarm (sparsity/baseline/runaway set, 5-arm conditioning as one swarm); channel_specific_seeds lives in conditioning.py"
```

---
### Task 5: 처리량·예산·게이트 — `swarm_bench.py`, `scripts/bench_mps.py`, 요약 테스트

**Files:**
- Create: `flymon/brain/swarm_bench.py`, `scripts/bench_mps.py`
- Modify: `tests/test_summary.py`(`results/summary/m0b.json` 검사 추가)
- Test: `tests/brain/test_swarm_bench.py`

**Interfaces:**
- Consumes Task 1–4 전부, `results/summary/m0.json`의 `sparsity` 행(CPU 기준).
- Produces `swarm_bench.DECISIONS = 52_000`, `WINDOWS`, `time_phase(eng, steps, warm) -> (ms, counts)`, `throughput_row(conn, pops, comps, params, B, odor, strength, steps, warm, device, reward_type="PAM08") -> dict`(`B, n_flies, ms_decision, ms_reinforce, fired_per_step_per_slot, kc_frac_slot0, mbon_hz_slot0, paired_slots_identical, weights_frac_after[, mem_alloc_MB]`), `budget_hours(ms_decision, ms_reinforce, n_flies, decisions, settle_decision_ms, read_ms, settle_reinforce_ms, pulse_max_ms, gap_ms, reinforce_slot_fraction) -> float`(부록 C.4 공식), `budget_table(rows, decisions, limit_hours) -> dict`(가장 큰 B 행, 가정 4줄, `baseline_hours`, `gate_ok`), `agreement(cpu_row, sw_sparsity, sw_baseline, kc_tol=0.01) -> dict`, `m0b_gate(budget, agree, n_channel_specific, n_seeds) -> dict`.
- Produces CLI `scripts/bench_mps.py throughput|reproduce|summary`(기본 경로 `results/m0b/throughput.json`, `results/m0b/reproduce.json`, `results/summary/m0b.json`; 실행은 컨트롤러).
- `results/summary/m0b.json` 스키마: `device, torch, params_frozen, throughput[rows], budget{B, n_flies, decisions, windows, rows[{assumption, hours}], limit_hours, baseline_hours, gate_ok}, agreement{kc_frac_A{cpu,swarm,abs_diff}, kc_frac_B, jaccard, chance, mbon_hz_rest_trimmed{cpu,swarm}, kc_tol, ok}, conditioning{n_seeds, n_flip, n_flip_disc, noplast_max_abs_dD, channel_specific_seeds, arms}, runaway{sat_hz, n_over_sat, n_kc_over_sat, spike_share_over_sat}, gate{throughput_ok, agreement_ok, channel_specific_ok, channel_specific_seeds, passed}, generated_at`.

- [ ] **Step 1: 실패하는 테스트** — `tests/brain/test_swarm_bench.py`

```python
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.swarm_bench import agreement, budget_hours, budget_table, m0b_gate, throughput_row


def test_budget_formula_and_gate():
    # 52,000 decisions, 32 flies: (1400 x 28.9 + 1600 x 28.9) ms per batch = 86.7 s -> 1625 batches -> 39.1 h
    h = budget_hours(28.9, 28.9, 32)
    assert h == pytest.approx(52_000 / 32 * (1400 + 1600) * 28.9 / 3.6e6)
    assert budget_hours(28.9, 28.9, 32, reinforce_slot_fraction=0.25) < h
    rows = [{"B": 32, "n_flies": 8, "ms_decision": 6.5, "ms_reinforce": 9.4},
            {"B": 128, "n_flies": 32, "ms_decision": 21.6, "ms_reinforce": 28.9}]
    t = budget_table(rows)
    assert t["B"] == 128 and t["gate_ok"] is True and 30 < t["baseline_hours"] < 45
    assert [r["hours"] for r in t["rows"]] == sorted([r["hours"] for r in t["rows"]], reverse=True)
    slow = budget_table([{"B": 128, "n_flies": 32, "ms_decision": 60.0, "ms_reinforce": 60.0}])
    assert slow["gate_ok"] is False
    ag = agreement({"frac_active_A": 0.062, "frac_active_B": 0.058, "mbon_hz_rest_trimmed": 3.5},
                   {"frac_active_A": 0.055, "frac_active_B": 0.06, "jaccard": 0.02, "chance": 0.03},
                   {"mbon_hz_rest_trimmed": 3.6})
    assert ag["ok"] is True
    assert m0b_gate(t, ag, 8, 8)["passed"] is True
    assert m0b_gate(t, ag, 7, 8)["passed"] is False


def test_throughput_row_runs_on_synthetic(synthetic_connectome, device):
    c = synthetic_connectome(disjoint_kc=True)
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    comps = compartments(c, pops, p.core_frac)
    row = throughput_row(c, pops, comps, p, 8, {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, 1.0, steps=20, warm=5, device=device)
    assert row["B"] == 8 and row["n_flies"] == 2
    assert row["ms_decision"] > 0 and row["ms_reinforce"] > 0
    assert row["paired_slots_identical"] is True
    assert 0.0 < row["weights_frac_after"] <= 1.0
```

`tests/test_summary.py`에 추가:

```python
P_M0B = Path("results/summary/m0b.json")


@pytest.mark.skipif(not P_M0B.exists(), reason="M0b summary not written yet")
def test_m0b_summary_records_gate_and_budget():
    """M0b gate = budget <= 60 h AND CPU-vs-swarm agreement AND channel-specific depression on every seed."""
    d = json.loads(P_M0B.read_text())
    assert set(d["gate"]) == {"throughput_ok", "agreement_ok", "channel_specific_ok", "channel_specific_seeds", "passed"}
    assert d["budget"]["limit_hours"] == 60.0
    assert d["gate"]["throughput_ok"] == (d["budget"]["baseline_hours"] <= 60.0)
    assert d["gate"]["passed"] == (d["gate"]["throughput_ok"] and d["gate"]["agreement_ok"] and d["gate"]["channel_specific_ok"])
    assert d["params_frozen"]["kc_thresh"] == Params().kc_thresh
    assert d["runaway"]["n_kc_over_sat"] >= 0.0
    assert {r["B"] for r in d["throughput"]} >= {32, 64, 128}
```

- [ ] **Step 2: 실패 확인** — Run: `uv run pytest tests/brain/test_swarm_bench.py -q` — Expected: `ModuleNotFoundError: No module named 'flymon.brain.swarm_bench'`.

- [ ] **Step 3: 구현** — `flymon/brain/swarm_bench.py`

```python
"""M0b throughput timing, the budget extrapolation of spec C.4, and the M0b gate."""
from __future__ import annotations

import time

import numpy as np
import torch

from .circuits import Populations
from .config import Params
from .connectome import Connectome
from .engine_mps import SLOTS, SwarmEngine
from .plasticity_mps import SwarmPlasticity

DECISIONS = 52_000          # spec 4.1: 42,000 brain decisions + 10,000 exploration/secondary
WINDOWS = {"settle_decision_ms": 800.0, "read_ms": 600.0, "settle_reinforce_ms": 800.0,
           "pulse_max_ms": 600.0, "gap_ms": 200.0}


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def time_phase(eng: SwarmEngine, steps: int, warm: int) -> tuple[float, np.ndarray]:
    """Mean ms per step over `steps` after `warm` steps; returns (ms, counts over the timed window)."""
    eng.run(warm)
    _sync(eng.device)
    t0 = time.perf_counter()
    counts = eng.run(steps)
    _sync(eng.device)
    return (time.perf_counter() - t0) / steps * 1000.0, counts


def throughput_row(conn: Connectome, pops: Populations, comps: dict, params: Params, B: int, odor, strength: float,
                   steps: int, warm: int, device: str, reward_type: str = "PAM08") -> dict:
    """One swarm of B slots: the decision phase (odour on every slot, plasticity off) and the
    reinforcement phase (odour on slot 0, plasticity on, the reward DAN driven on slot 0)."""
    if B % SLOTS:
        raise ValueError(f"B must be a multiple of {SLOTS}")
    eng = SwarmEngine(conn, pops, params, n_flies=B // SLOTS, seed=0, device=device)
    pl = SwarmPlasticity(eng, pops, comps)
    sec = steps / 1000.0
    for s in range(eng.B):
        eng.present(s, odor, strength)
    pl.set_enabled(False)
    ms_dec, c = time_phase(eng, steps, warm)
    row = {"B": B, "n_flies": eng.F, "ms_decision": ms_dec,
           "fired_per_step_per_slot": float(c.sum() / steps / eng.B),
           "kc_frac_slot0": float((c[0][pops.kc] > 0).mean()), "mbon_hz_slot0": float(c[0][pops.mbon].mean() / sec),
           "paired_slots_identical": bool(np.array_equal(c[0], c[1]))}
    eng.clear_drive()
    for f in range(eng.F):
        eng.present(SLOTS * f, odor, strength)
        pl.drive_dan(f, reward_type, params.dan_drive_mv)
    pl.set_enabled(True)
    ms_rein, _ = time_phase(eng, steps, warm)
    row["ms_reinforce"] = ms_rein
    row["weights_frac_after"] = pl.weights_frac()
    if eng.device.type == "mps":
        row["mem_alloc_MB"] = torch.mps.current_allocated_memory() / 1e6
    del pl, eng
    if device == "mps":
        torch.mps.empty_cache()
    return row


def budget_hours(ms_decision: float, ms_reinforce: float, n_flies: int, decisions: int = DECISIONS,
                 settle_decision_ms: float = 800.0, read_ms: float = 600.0, settle_reinforce_ms: float = 800.0,
                 pulse_max_ms: float = 600.0, gap_ms: float = 200.0, reinforce_slot_fraction: float = 1.0) -> float:
    """Spec C.4: T = decisions / F x (decision steps x ms_decision + reinforcement steps x ms_reinforce).
    `reinforce_slot_fraction` < 1 models skipping the idle slots in the reinforcement phase."""
    s_dec = settle_decision_ms + read_ms
    s_rein = settle_reinforce_ms + pulse_max_ms + gap_ms
    ms = s_dec * ms_decision + s_rein * ms_reinforce * reinforce_slot_fraction
    return decisions / n_flies * ms / 3.6e6


def budget_table(rows: list[dict], decisions: int = DECISIONS, limit_hours: float = 60.0) -> dict:
    """Budget rows for the largest B measured, under the C.4 assumptions, and the gate on the baseline."""
    row = max(rows, key=lambda r: r["B"])
    F = row["n_flies"]
    base = dict(decisions=decisions, n_flies=F, ms_decision=row["ms_decision"], ms_reinforce=row["ms_reinforce"])
    table = [
        {"assumption": "baseline (settle 800 / read 600 / reinforce 800+600+200)", "hours": budget_hours(**base)},
        {"assumption": "decision settle 200 ms", "hours": budget_hours(**base, settle_decision_ms=200.0)},
        {"assumption": "baseline + idle reinforcement slots skipped (1/4 cost)",
         "hours": budget_hours(**base, reinforce_slot_fraction=0.25)},
        {"assumption": "decision settle 200 ms + idle reinforcement slots skipped",
         "hours": budget_hours(**base, settle_decision_ms=200.0, reinforce_slot_fraction=0.25)},
    ]
    return {"B": row["B"], "n_flies": F, "decisions": decisions, "windows": WINDOWS, "rows": table,
            "limit_hours": limit_hours, "baseline_hours": table[0]["hours"],
            "gate_ok": table[0]["hours"] <= limit_hours}


def agreement(cpu_row: dict, sw_sparsity: dict, sw_baseline: dict, kc_tol: float = 0.01) -> dict:
    """CPU (results/summary/m0.json sparsity row) vs swarm: KC fraction within kc_tol, overlap at or
    below chance, trimmed MBON baseline inside the spec band 3-4 Hz."""
    d_a = abs(sw_sparsity["frac_active_A"] - cpu_row["frac_active_A"])
    d_b = abs(sw_sparsity["frac_active_B"] - cpu_row["frac_active_B"])
    trimmed = sw_baseline["mbon_hz_rest_trimmed"]
    ok = d_a <= kc_tol and d_b <= kc_tol and sw_sparsity["jaccard"] <= sw_sparsity["chance"] and 3.0 <= trimmed <= 4.0
    return {"kc_frac_A": {"cpu": cpu_row["frac_active_A"], "swarm": sw_sparsity["frac_active_A"], "abs_diff": d_a},
            "kc_frac_B": {"cpu": cpu_row["frac_active_B"], "swarm": sw_sparsity["frac_active_B"], "abs_diff": d_b},
            "jaccard": sw_sparsity["jaccard"], "chance": sw_sparsity["chance"],
            "mbon_hz_rest_trimmed": {"cpu": cpu_row["mbon_hz_rest_trimmed"], "swarm": trimmed},
            "kc_tol": kc_tol, "ok": bool(ok)}


def m0b_gate(budget: dict, agree: dict, n_channel_specific: int, n_seeds: int) -> dict:
    return {"throughput_ok": bool(budget["gate_ok"]), "agreement_ok": bool(agree["ok"]),
            "channel_specific_ok": n_channel_specific == n_seeds, "channel_specific_seeds": n_channel_specific,
            "passed": bool(budget["gate_ok"] and agree["ok"] and n_channel_specific == n_seeds)}
```

- [ ] **Step 4: CLI** — `scripts/bench_mps.py`

```python
#!/usr/bin/env python3
"""M0b: swarm throughput, the M0 measurements reproduced on the swarm (real data), and the gate summary.

  throughput  ms/step for B in {32,64,128,256}: decision phase (odour on every slot, plasticity off)
              and reinforcement phase (odour + reward DAN on slot 0, plasticity on)
  reproduce   sparsity + MBON baseline + runaway set on a 3-fly swarm, then the 5-arm x 8-seed
              conditioning as ONE swarm of 40 flies (every (seed, arm) is a fly)
  summary     results/summary/m0b.json: budget table (spec C.4), CPU-vs-swarm agreement, conditioning
              channel specificity (A.2 definition), the M0b gate
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path

import torch

from flymon.brain.circuits import Populations, compartments, validate_populations
from flymon.brain.conditioning import Readout, channel_specific_seeds, summarise
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.stimuli import design_odor_pair
from flymon.brain.engine_mps import SwarmEngine
from flymon.brain.plasticity_mps import SwarmPlasticity
from flymon.brain.conditioning_mps import swarm_baseline, swarm_reversal_test, swarm_sparsity
from flymon.brain.swarm_bench import agreement, budget_table, m0b_gate, throughput_row


def _load(npz: str, punish_type: str = "PPL105", reward_type: str = "PAM08"):
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    validate_populations(conn, pops, comps, punish_type, reward_type)
    return conn, pops, comps


def _write(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def cmd_throughput(a):
    conn, pops, comps = _load(a.npz)
    odor_a, _ = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    rows = []
    for B in a.B:
        rows.append(throughput_row(conn, pops, comps, Params(), B, odor_a, a.strength, a.steps, a.warm, a.device))
        print(json.dumps(rows[-1]), flush=True)
    _write(a.out, {"device": a.device, "torch": torch.__version__, "N": conn.N, "warm": a.warm, "steps": a.steps,
                   "strength": a.strength, "odor_seed": a.odor_seed, "rows": rows})


def cmd_reproduce(a):
    conn, pops, comps = _load(a.npz, a.punish_type, a.reward_type)
    p = Params()
    odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    t0 = time.perf_counter()
    eng = SwarmEngine(conn, pops, p, n_flies=a.rest_seeds, seed=0, device=a.device)
    sp = swarm_sparsity(eng, pops, odor_a, odor_b, a.strength)
    base = swarm_baseline(eng, pops, ms=a.rest_ms)
    print(json.dumps({k: sp[k] for k in ("frac_active_A", "frac_active_B", "jaccard", "chance")}),
          json.dumps({k: base[k] for k in ("mbon_hz_rest", "mbon_hz_rest_trimmed")}), json.dumps(base["runaway"]), flush=True)
    del eng
    if a.device == "mps":
        torch.mps.empty_cache()
    eng = SwarmEngine(conn, pops, p, n_flies=5 * a.seeds, seed=0, device=a.device)
    pl = SwarmPlasticity(eng, pops, comps)
    ro = Readout.from_compartments(comps, a.punish_type, a.reward_type)
    per_seed = swarm_reversal_test(eng, pl, ro, odor_a, odor_b, a.strength, a.seeds, a.punish_type, a.reward_type,
                                   trials=a.trials, present_ms=a.present_ms, settle_ms=a.settle_ms)
    s = summarise(per_seed)
    s["channel_specific_seeds"] = channel_specific_seeds(s["per_seed"])
    out = {"device": a.device, "torch": torch.__version__,
           "params": {**dataclasses.asdict(p), "punish_type": a.punish_type, "reward_type": a.reward_type,
                      "settle_ms": a.settle_ms, "present_ms": a.present_ms, "trials": a.trials},
           "strength": a.strength, "odor_seed": a.odor_seed, "sparsity": sp, "baseline": base, "conditioning": s,
           "wall_clock_s": time.perf_counter() - t0}
    print(json.dumps({k: s[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds")}),
          f"wall {out['wall_clock_s']:.0f}s", flush=True)
    _write(a.out, out)


def cmd_summary(a):
    th = json.loads(Path(a.throughput).read_text())
    rp = json.loads(Path(a.reproduce).read_text())
    m0 = json.loads(Path(a.m0).read_text())
    budget = budget_table(th["rows"], decisions=a.decisions, limit_hours=a.limit_hours)
    agree = agreement(m0["sparsity"], rp["sparsity"], rp["baseline"])
    co = rp["conditioning"]
    gate = m0b_gate(budget, agree, co["channel_specific_seeds"], co["n_seeds"])
    out = {"device": th["device"], "torch": th["torch"], "params_frozen": dataclasses.asdict(Params()),
           "throughput": th["rows"], "budget": budget, "agreement": agree,
           "conditioning": {k: co[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD",
                                               "channel_specific_seeds", "arms")},
           "runaway": rp["baseline"]["runaway"], "gate": gate,
           "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    print(f"M0b gate {'PASS' if gate['passed'] else 'FAIL'}: baseline {budget['baseline_hours']:.1f} h "
          f"(limit {budget['limit_hours']}), agreement_ok={agree['ok']}, "
          f"channel_specific={gate['channel_specific_seeds']}/{co['n_seeds']}")
    _write(a.out, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("throughput")
    t.add_argument("--npz", default="data/malecns.npz"); t.add_argument("--out", default="results/m0b/throughput.json")
    t.add_argument("--B", type=int, nargs="+", default=[32, 64, 128, 256]); t.add_argument("--device", default="mps")
    t.add_argument("--steps", type=int, default=200); t.add_argument("--warm", type=int, default=100)
    t.add_argument("--strength", type=float, default=0.35); t.add_argument("--k", type=int, default=8); t.add_argument("--odor-seed", type=int, default=0)
    t.set_defaults(fn=cmd_throughput)
    r = sub.add_parser("reproduce")
    r.add_argument("--npz", default="data/malecns.npz"); r.add_argument("--out", default="results/m0b/reproduce.json")
    r.add_argument("--device", default="mps"); r.add_argument("--seeds", type=int, default=8)
    r.add_argument("--rest-seeds", type=int, default=3); r.add_argument("--rest-ms", type=float, default=3000.0)
    r.add_argument("--trials", type=int, default=12); r.add_argument("--present-ms", type=float, default=800.0)
    r.add_argument("--settle-ms", type=float, default=800.0)
    r.add_argument("--strength", type=float, default=0.35); r.add_argument("--k", type=int, default=8); r.add_argument("--odor-seed", type=int, default=0)
    r.add_argument("--punish-type", default="PPL105"); r.add_argument("--reward-type", default="PAM08")
    r.set_defaults(fn=cmd_reproduce)
    s = sub.add_parser("summary")
    s.add_argument("--throughput", default="results/m0b/throughput.json"); s.add_argument("--reproduce", default="results/m0b/reproduce.json")
    s.add_argument("--m0", default="results/summary/m0.json"); s.add_argument("--out", default="results/summary/m0b.json")
    s.add_argument("--decisions", type=int, default=52_000); s.add_argument("--limit-hours", type=float, default=60.0)
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 통과 확인** — Run: `uv run pytest tests/brain/test_swarm_bench.py tests/test_summary.py -v` — Expected: PASS(m0b 요약 테스트는 파일이 없어 skip).

- [ ] **Step 6: CLI 스모크(결과 디렉터리 금지)** — 실제 데이터가 있으면 임시 경로로만:

```bash
T=$(mktemp -d)
uv run python scripts/bench_mps.py throughput --B 32 --steps 50 --warm 20 --out $T/throughput.json
uv run python scripts/bench_mps.py reproduce --seeds 1 --trials 1 --present-ms 100 --settle-ms 100 --rest-ms 300 --out $T/reproduce.json
uv run python scripts/bench_mps.py summary --throughput $T/throughput.json --reproduce $T/reproduce.json --out $T/m0b.json
```

Expected: 세 파일이 생기고 `summary`가 `M0b gate FAIL: baseline … h` 한 줄을 찍는다(작은 B와 1 trial이라 FAIL이 정상). 약 1분. `git status`에 `results/` 변화가 없어야 한다.

- [ ] **Step 7: 전체 스위트 + 커밋**

```bash
uv run pytest -q
git add flymon/brain/swarm_bench.py scripts/bench_mps.py tests/brain/test_swarm_bench.py tests/test_summary.py
git commit -m "feat(brain): swarm throughput/budget/gate (spec C.4) and the bench_mps CLI (throughput, reproduce, summary)"
```

---
### Task 6: 문서 — README M0b 절, 스펙 6절 파일 목록

**Files:**
- Modify: `README.md`(`### 학습 중 보기` 앞에 `### M0b MPS 스웜` 절), `docs/superpowers/specs/2026-09-14-flymon-design.md`(6절 `brain/` 줄)

- [ ] **Step 1: README** — `### 학습 중 보기` 앞에 추가:

```markdown
### M0b MPS 스웜

    uv run python scripts/bench_mps.py throughput            # B=32/64/128/256, 결정·강화 단계 ms/step → results/m0b/throughput.json
    uv run python scripts/bench_mps.py reproduce             # 희소성·기저·폭주 집합(3마리) + 조건화 5팔×8시드를 40마리 스웜 하나로 → results/m0b/reproduce.json
    uv run python scripts/bench_mps.py summary               # 예산표(스펙 C.4)·CPU 일치·채널 특이성·게이트 → results/summary/m0b.json

- 스웜 엔진 `flymon/brain/engine_mps.py`는 torch MPS가 기본이고 `--device cpu`로도 돈다(느리다). 스웜 테스트는 cpu와 mps 두 장치에서 실행된다.
- 슬롯 규칙(마리당 4슬롯, 강화는 슬롯 0), 마리별 플래그(가소성·셔플·DAN), 잡음 주입, 예산 가정은 스펙 부록 C.
- `reproduce`의 조건화는 CPU와 프로토콜이 조금 다르다(프레젠테이션마다 스웜 전체를 한 번 리셋, 프로브는 고정 시드): `flymon/brain/conditioning_mps.py` docstring.
```

- [ ] **Step 2: 스펙 6절** — `brain/` 줄을 `brain/    data_build.py  engine_cpu.py  engine_mps.py  plasticity_mps.py  conditioning_mps.py  swarm_bench.py  circuits.py  odors.py  plasticity.py  readout.py`로 바꾼다. `scripts/` 줄은 그대로(`bench_mps.py` 이미 있음).

- [ ] **Step 3: 커밋**

```bash
git add README.md docs/superpowers/specs/2026-09-14-flymon-design.md
git commit -m "docs: README M0b section; spec repo layout names the swarm modules"
```

---

## 컨트롤러 실행 (서브에이전트 금지) — 실제 데이터 벤치·재현·게이트, 부록 C 갱신

1. `uv run pytest -q` 초록 확인.
2. `uv run python scripts/bench_mps.py throughput` (B=32/64/128/256, 각 100+200스텝 × 2단계; 약 2분). `results/m0b/throughput.json`.
3. `uv run python scripts/bench_mps.py reproduce` (희소성 3마리 + 조건화 40마리 B=160, 약 46,400스텝 × ~35 ms ≈ 25–30분). 실행 중 다른 MPS 작업을 돌리지 않는다. `results/m0b/reproduce.json`.
4. `uv run python scripts/bench_mps.py summary` → `results/summary/m0b.json`, 한 줄 판정. `uv run pytest tests/test_summary.py -q`.
5. 판정에 따라:
   - PASS: `results/summary/m0b.json` 커밋(`results: M0b bench, swarm reproduction, gate`). 부록 C.3의 스파이크 표 아래에 "M0b 측정(bench_mps.py)" 표를 추가하고 C.4의 예산표를 측정값으로 바꾸며 잠정 판정을 확정 판정으로 고친다. C.5 6번의 결과(KC 활성 Δ, Jaccard, 절사 기저, 채널 특이 n/8, 폭주 집합의 KC 수·스파이크 비율)를 적는다. README "측정된 것"에 M0b 한 문단.
   - 예산 > 60시간: 스펙 4.1 축소 순서(배틀 40 → 30, 그다음 2차 팔)를 적용한 예산을 C.4에 함께 적고 판정을 기록. 유휴 슬롯 건너뜀(C.4)을 M3 필수 과제로 올린다.
   - 채널 특이성 < 8/8 또는 일치 실패: 스웜 프로토콜 차이(리셋 시드, 프로브 시드)를 먼저 의심하고 `--seeds 8`로 CPU와 같은 수치 정의를 다시 확인한 뒤 판정.
6. STD 재검토 조건(A.5): `runaway.n_kc_over_sat > 0`이면 핸드오프에 기록.
7. 최종 브랜치 리뷰(sdd-reviewer, 이 계획 범위만), 핸드오프 `docs/handoffs/2026-09-15-m2-handoff.md`(M2 브레인스토밍 입력: 판독 집합 선정과 후보 냄새에 대한 MBON13 반응 여부가 첫 질문).

## 자체 검토

- 스펙 커버리지: 3.1 MPS 항목(전파 두 갈래, 슬롯/마리 상태, 짝지은 잡음·주입, 플래그, 상태 왕복) = Task 1–3; 5절 M0b(배치·잡음·hive·플래그·API·CPU 일치·처리량·예산·폭주 집합) = Task 1–5 + 컨트롤러; 8절 통계 = Task 2(합성 래스터)·Task 4/5(실제 통계); C.5 테스트 1–7 = Task 2(1), Task 1(2), Task 3(3, 4), Task 2(5), Task 4/5·컨트롤러(6, 7). C.2 5항의 API 이름은 Task 1–3의 Interfaces와 같다.
- 플레이스홀더: 없음. 모든 코드 블록은 스크래치에서 실행·통과한 것이다.
- 타입 일관성: `SwarmEngine.run(ms, idx=None) -> np.ndarray`, `step() -> (b, n)`, `on_step(engine, b, n)`, `set_enabled(on, fly=None)`, `weights_frac(fly=None)`, `weights_frac_by_mbon_set(mbon_idx, fly=None)`, `swarm_probe -> {"A","P"}`, `throughput_row -> dict`가 Task 1·3·4·5에서 같은 이름·시그니처로 쓰인다.
