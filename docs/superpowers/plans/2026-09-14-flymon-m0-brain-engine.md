# FlyMon M0 — 뇌 엔진과 flybrain 측정값 재현 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MaleCNS v1.0 커넥톰을 우리 포맷으로 빌드하고, 자체 CPU LIF 엔진과 KC→MBON 가소성을 구현한 뒤, flybrain이 기록한 세 측정값(케니언세포 희소성, MBON 기저 발화, 조건화 반전 8/8 시드)을 재현해 엔진 합격 게이트를 통과한다.

**Architecture:** 데이터 빌드(feather → npz) → `Connectome`(배열 + 부호·반구 보정 + CSC) → `Populations`/구획 표 → `Engine`(이벤트 구동 LIF, 포아송 수용체, MBON 기저 구동, KC 역치 정규화) → `Plasticity`(3인자 규칙, 구획별 도파민, 바닥, 회복) → 재현 스크립트가 JSON 결과를 남기고 테스트가 범위를 검사한다. M0b(MPS 스웜), M1(배틀 환경)은 별도 계획이며 M1은 이 계획과 병렬로 진행 가능하다.

**Tech Stack:** Python 3.13(uv), numpy, scipy, pyarrow, pandas, pytest. torch는 M0에서 쓰지 않는다.

**Spec:** `docs/superpowers/specs/2026-09-14-flymon-design.md` (v4). 이 계획은 스펙 3.1, 5절 M0, 8절 단위·통계 테스트를 구현한다.

## Global Constraints

- Python 3.13, uv 프로젝트. 의존성은 pyproject.toml에 고정.
- flybrain(TheMrRaGe) 코드를 복사하지 않는다. 설계 결정과 측정값만 재현 목표로 쓴다. 파일 헤더 주석에 "reproduction target: flybrain FINDINGS.md"로 출처만 적는다.
- Shiu et al. 2024 상수 그대로: 휴지 0 mV 기준으로 역치 7.0 mV(−52 → −45), 리셋 0, τm 20 ms, τsyn 5 ms, 시냅스 지연 1.8 ms, 불응 2.2 ms, 시냅스당 0.275 mV, dt 1.0 ms.
- 부호: acetylcholine +1, gaba −1, glutamate −1(GluCl-α), histamine −1, dopamine·octopamine·serotonin·unknown 0.
- 우리 설계 결정(스펙 3.1): lLN1/lLN2 억제 재지정, APL 출력 스케일 0.1, MBON 기저 구동 0.85×역치, KC 역치 정규화, 반구 후시냅스 스케일 보정, 가중치 임계 ≥ 5.
- 데이터: MaleCNS v1.0(CC-BY 4.0). `data/`와 `results/`는 git-ignore, `results/summary/*.json`만 커밋.
- 실제 데이터가 필요한 테스트는 `data/malecns.npz`가 없으면 `pytest.skip`. 합성 커넥톰 테스트는 항상 돈다.
- 각 태스크 끝에 커밋. 커밋 메시지는 `feat(brain): ...`, `test(brain): ...`, `chore: ...` 형식.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | uv 프로젝트, 의존성, pytest 설정 |
| `.gitignore` | data/, results/(summary 제외), 캐시 |
| `flymon/__init__.py`, `flymon/brain/__init__.py` | 패키지 |
| `flymon/brain/config.py` | `Params` dataclass: Shiu 상수 + 우리 결정값 + 가소성 상수 |
| `flymon/brain/connectome.py` | npz 스키마, `Connectome` 로드·검증, 부호 재지정, 반구 보정, CSC(out-edge) 빌드 |
| `flymon/brain/data_build.py` | Janelia feather 3개 → `data/malecns.npz` + `data/malecns.manifest.json` |
| `flymon/brain/circuits.py` | `Populations`(수용체 타입, KC, MBON, DAN 타입별, APL, DN, MN), 도파민 구획 표 |
| `flymon/brain/stimuli.py` | 냄새(수용체 타입 → Hz) 제시, 드라이브 균등화, 설계된 냄새 쌍 |
| `flymon/brain/engine_cpu.py` | `Engine`: 이벤트 구동 LIF, 포아송 수용체, 외부 구동, 잡음, 기록 |
| `flymon/brain/plasticity.py` | `Plasticity`: KC→MBON 가소성 엣지, 자취, 구획별 도파민, 규칙, 바닥, 회복 |
| `scripts/reproduce_flybrain_measurements.py` | KC 희소성·MBON 기저·조건화 반전 측정, JSON 출력 |
| `tests/conftest.py` | 합성 커넥톰 픽스처 |
| `tests/brain/test_*.py` | 태스크별 테스트 |
| `results/summary/m0.json` | 게이트 결과(커밋) |
| `docs/acknowledgments.md` | 데이터·선행 연구 표기 |

---

### Task 1: 프로젝트 스캐폴드와 파라미터

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `flymon/__init__.py`, `flymon/brain/__init__.py`, `flymon/brain/config.py`
- Test: `tests/brain/test_config.py`

**Interfaces:**
- Produces: `flymon.brain.config.Params` (dataclass, 아래 필드 전부), `Params.dly_steps()`, `Params.refrac_steps()`.

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "flymon"
version = "0.0.1"
description = "MaleCNS fly-brain LIF simulation learning Pokémon attack choice via KC->MBON plasticity"
requires-python = ">=3.13,<3.14"
dependencies = [
  "numpy>=2.1",
  "scipy>=1.14",
  "pyarrow>=17",
  "pandas>=2.2",
]

[dependency-groups]
dev = ["pytest>=8", "pytest-timeout>=2.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["flymon"]

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 600
addopts = "-q"
```

- [ ] **Step 2: .gitignore 작성**

```
data/
!data/.gitkeep
results/
!results/summary/
!results/summary/*.json
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

`data/.gitkeep`와 `results/summary/.gitkeep`를 빈 파일로 만든다.

- [ ] **Step 3: 실패하는 테스트 작성** — `tests/brain/test_config.py`

```python
from flymon.brain.config import Params


def test_shiu_constants_exact():
    p = Params()
    assert p.v_thresh == 7.0
    assert p.v_reset == 0.0
    assert p.tau_m == 20.0
    assert p.tau_syn == 5.0
    assert p.syn_delay_ms == 1.8
    assert p.refractory_ms == 2.2
    assert p.mv_per_synapse == 0.275
    assert p.dt == 1.0


def test_derived_steps():
    p = Params()
    assert p.dly_steps() == 2      # round(1.8 / 1.0)
    assert p.refrac_steps() == 2   # round(2.2 / 1.0)


def test_our_design_defaults():
    p = Params()
    assert p.apl_scale == 0.1
    assert p.mbon_hold_frac == 0.85
    assert p.kc_thresh == 1.0
    assert p.sign_override == (("lLN1", -1), ("lLN2", -1))
    assert p.min_weight == 5
```

- [ ] **Step 4: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: flymon.brain.config`

- [ ] **Step 5: config.py 구현**

```python
"""Model constants.

Shiu et al. 2024 (Nature 634:210) LIF constants are used at their published
values. Everything under "our design decisions" is a modelling choice made in
this project; reproduction target for those choices: flybrain FINDINGS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Params:
    # --- Shiu et al. 2024, membrane and synapse (mV above rest, ms) ---
    v_thresh: float = 7.0          # -52 -> -45 mV
    v_reset: float = 0.0
    tau_m: float = 20.0
    tau_syn: float = 5.0
    syn_delay_ms: float = 1.8
    refractory_ms: float = 2.2
    mv_per_synapse: float = 0.275
    dt: float = 1.0
    noise_mv: float = 0.15         # per-step Gaussian membrane jitter
    max_rate_hz: float = 200.0     # receptor Poisson rate at strength 1.0

    # --- our design decisions (spec 3.1) ---
    min_weight: int = 5            # synapse-count threshold when building CSC
    sign_override: tuple = (("lLN1", -1), ("lLN2", -1))
    apl_scale: float = 0.1         # multiplier on APL out-edge weights
    mbon_hold_frac: float = 0.85   # tonic drive on MBONs as fraction of v_thresh
    kc_thresh: float = 1.0         # multiplier on KC threshold after PN-input normalisation
    kc_norm_clip: tuple = (0.5, 3.0)
    balance_hemispheres: bool = True

    # --- plasticity (spec 3.1, tuned in Task 9 within these defaults) ---
    learn_rate: float = 3e-4
    kc_trace_ms: float = 200.0
    da_trace_ms: float = 100.0
    da_baseline_ms: float = 1000.0
    kc_trace_scale: float = 40.0   # normalise a strongly driven KC trace to ~1
    da_trace_scale: float = 20.0   # normalise phasic dopamine at a core MBON to ~1
    min_weight_frac: float = 0.2
    core_frac: float = 0.2         # DAN->MBON weight >= this fraction of type peak = core
    dan_drive_mv: float = 70.0
    recovery_per_pulse: float = 0.0  # fraction of (w0 - w) restored after each pulse; 0 = off

    def dly_steps(self) -> int:
        return max(1, round(self.syn_delay_ms / self.dt))

    def refrac_steps(self) -> int:
        return max(1, round(self.refractory_ms / self.dt))
```

- [ ] **Step 6: 패키지 초기화 파일 생성**

`flymon/__init__.py`와 `flymon/brain/__init__.py`는 각각 한 줄 docstring만 둔다.

- [ ] **Step 7: 테스트 통과 확인**

Run: `uv sync && uv run pytest tests/brain/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: 커밋**

```bash
git add pyproject.toml .gitignore uv.lock data/.gitkeep results/summary/.gitkeep flymon tests
git commit -m "chore: scaffold flymon package with Shiu LIF constants"
```

---

### Task 2: npz 스키마, Connectome 로드, 부호·반구 보정, CSC

**Files:**
- Create: `flymon/brain/connectome.py`, `tests/conftest.py`
- Test: `tests/brain/test_connectome.py`

**Interfaces:**
- Consumes: `Params`.
- Produces:
  - `Connectome` dataclass: `bodyId:int64[N]`, `type:str[N]`, `cls:str[N]`, `sc:str[N]`, `nt:str[N]`, `sign:int8[N]`, `side:str[N]`, `pre:int32[E]`, `post:int32[E]`, `w:int32[E]`, `N:int`, `E:int`.
  - `Connectome.load(path) -> Connectome`, `Connectome.save(path)`.
  - `apply_sign_override(conn, params) -> (sign:int8[N], n_overridden:int)`.
  - `hemisphere_scale(conn) -> float` (inL/inR of synaptic weight onto L vs R postsynaptic).
  - `build_csc(conn, params, apl_idx:int32[]) -> CSC` where `CSC` has `ptr:int64[N+1]`, `tgt:int32[nnz]`, `w:float32[nnz]` (signed mV, min_weight applied, zero-sign edges dropped, APL scaled, hemisphere balanced), plus `pre_of_edge()`.
- `tests/conftest.py`의 `synthetic_connectome(n_kc=40, n_mbon=4, n_dan=4, n_orn=20, seed=0) -> Connectome` 픽스처 빌더.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_connectome.py`

```python
import numpy as np
import pytest

from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, apply_sign_override, build_csc, hemisphere_scale


def test_roundtrip_npz(tmp_path, synthetic_connectome):
    c = synthetic_connectome()
    path = tmp_path / "c.npz"
    c.save(path)
    d = Connectome.load(path)
    assert d.N == c.N and d.E == c.E
    np.testing.assert_array_equal(d.pre, c.pre)
    assert d.type.dtype.kind == "U"


def test_load_rejects_bad_schema(tmp_path):
    np.savez(tmp_path / "bad.npz", bodyId=np.arange(3))
    with pytest.raises(ValueError, match="missing"):
        Connectome.load(tmp_path / "bad.npz")


def test_sign_override_counts_prefix_matches(synthetic_connectome):
    c = synthetic_connectome()
    # synthetic builder labels two cells 'lLN1_a' and one 'lLN2P_a' as acetylcholine (+1)
    sign, n = apply_sign_override(c, Params())
    assert n == 3
    assert (sign[np.char.startswith(c.type, "lLN")] == -1).all()
    # untouched elsewhere
    other = ~np.char.startswith(c.type, "lLN")
    np.testing.assert_array_equal(sign[other], c.sign[other])


def test_hemisphere_scale_is_inL_over_inR(synthetic_connectome):
    c = synthetic_connectome()
    inL = c.w[c.side[c.post] == "L"].sum()
    inR = c.w[c.side[c.post] == "R"].sum()
    assert hemisphere_scale(c) == pytest.approx(inL / inR)


def test_csc_matches_dense(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(min_weight=1, balance_hemispheres=False)
    apl = np.flatnonzero(c.type == "APL")
    csc = build_csc(c, p, apl)
    # dense reference: M[post, pre] = sign[pre] * w * mv_per_synapse, APL rows scaled
    M = np.zeros((c.N, c.N), np.float64)
    sign, _ = apply_sign_override(c, p)
    for a, b, w in zip(c.pre, c.post, c.w):
        s = sign[a]
        if s == 0 or w < p.min_weight:
            continue
        scale = p.apl_scale if a in set(apl.tolist()) else 1.0
        M[b, a] += s * w * p.mv_per_synapse * scale
    for i in range(c.N):
        col = np.zeros(c.N)
        np.add.at(col, csc.tgt[csc.ptr[i]:csc.ptr[i + 1]], csc.w[csc.ptr[i]:csc.ptr[i + 1]])
        np.testing.assert_allclose(col, M[:, i], atol=1e-6)


def test_csc_drops_below_threshold_and_zero_sign(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(min_weight=5)
    csc = build_csc(c, p, np.flatnonzero(c.type == "APL"))
    pre = csc.pre_of_edge()
    sign, _ = apply_sign_override(c, p)
    assert (sign[pre] != 0).all()
    assert (np.abs(csc.w) >= 5 * p.mv_per_synapse * min(1.0, p.apl_scale) - 1e-9).all()
```

- [ ] **Step 2: 합성 커넥톰 픽스처 작성** — `tests/conftest.py`

```python
"""Synthetic connectome builder shared by brain tests.

Layout (indices are contiguous blocks, in this order):
  ORN receptors (class 'olfactory', types ORN_<k>), ALPN, KC, MBON, DAN (PAM/PPL1 types),
  APL (one), lLN1/lLN2 (three, labelled acetylcholine on purpose), DN, MN.
Edges: ORN->ALPN, ALPN->KC (random 3 glomeruli each), KC->MBON (all pairs), DAN->MBON,
  APL->KC, KC->APL, lLN->ALPN, MBON->DN, DN->MN. Weights are synapse counts >= 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from flymon.brain.connectome import Connectome


def _build(n_orn=20, n_alpn=10, n_kc=40, n_mbon=4, n_dan=4, n_dn=6, n_mn=4, seed=0):
    rng = np.random.default_rng(seed)
    types, cls, sc, nt, side = [], [], [], [], []

    def add(n, t, c, s, n_t, sd=None):
        for i in range(n):
            types.append(t(i) if callable(t) else t)
            cls.append(c); sc.append(s); nt.append(n_t)
            side.append(sd if sd else ("L" if i % 2 == 0 else "R"))

    add(n_orn, lambda i: f"ORN_{'DM1 DA1 VA2 DM6 VC1'.split()[i % 5]}", "olfactory", "cb_sensory", "acetylcholine", "M")
    add(n_alpn, lambda i: f"ALPN{i}", "ALPN", "cb_intrinsic", "acetylcholine")
    add(n_kc, lambda i: "KCab-m", "Kenyon_Cell", "cb_intrinsic", "acetylcholine")
    add(n_mbon, lambda i: f"MBON{i + 1:02d}", "MBON", "cb_intrinsic", "acetylcholine" if i % 2 else "glutamate")
    add(n_dan, lambda i: ["PAM08", "PAM08", "PPL105", "PPL105"][i], "DAN", "cb_intrinsic", "dopamine")
    add(1, "APL", "", "cb_intrinsic", "gaba", "L")
    add(3, lambda i: ["lLN1_a", "lLN1_a", "lLN2P_a"][i], "ALLN", "cb_intrinsic", "acetylcholine")
    add(n_dn, lambda i: f"DNg{i:03d}", "", "descending_neuron", "acetylcholine")
    add(n_mn, lambda i: f"MNleg{i}", "", "vnc_motor", "acetylcholine")

    types = np.array(types); cls = np.array(cls); sc = np.array(sc); nt = np.array(nt); side = np.array(side)
    N = len(types)
    SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
    sign = np.array([SIGN.get(x, 0) for x in nt], np.int8)

    def idx(mask):
        return np.flatnonzero(mask)

    orn, alpn, kc, mbon = idx(cls == "olfactory"), idx(cls == "ALPN"), idx(cls == "Kenyon_Cell"), idx(cls == "MBON")
    dan, apl, lln = idx(cls == "DAN"), idx(types == "APL"), idx(np.char.startswith(types, "lLN"))
    dn, mn = idx(sc == "descending_neuron"), idx(sc == "vnc_motor")

    pre, post, w = [], [], []

    def edge(a, b, weight):
        pre.append(a); post.append(b); w.append(int(weight))

    glom = {t: i for i, t in enumerate(sorted(set(types[orn])))}
    for o in orn:                                  # each glomerulus -> two ALPNs
        g = glom[types[o]]
        for b in (alpn[(2 * g) % n_alpn], alpn[(2 * g + 1) % n_alpn]):
            edge(o, b, 12)
    for k in kc:                                   # 3 random ALPN inputs, 6-20 synapses
        for a in rng.choice(alpn, 3, replace=False):
            edge(a, k, rng.integers(6, 21))
        edge(k, apl[0], 4); edge(apl[0], k, 30)    # APL loop
    for k in kc:
        for m in mbon:
            edge(k, m, rng.integers(5, 15))
    for d in dan:                                  # PAM08 -> MBON01/02 strongly, PPL105 -> MBON03/04
        targets = mbon[:2] if types[d].startswith("PAM") else mbon[2:]
        for m in targets:
            edge(d, m, 60)
        edge(d, mbon[(d % 2) + 2] if types[d].startswith("PAM") else mbon[d % 2], 3)  # stray, below core
    for l in lln:
        for a in alpn:
            edge(l, a, 8)
    for m in mbon:
        for d_ in dn[:3]:
            edge(m, d_, 9)
    for d_ in dn:
        edge(d_, mn[d_ % n_mn], 7)

    return Connectome(
        bodyId=np.arange(10_000, 10_000 + N, dtype=np.int64), type=types, cls=cls, sc=sc, nt=nt,
        sign=sign, side=side, pre=np.array(pre, np.int32), post=np.array(post, np.int32), w=np.array(w, np.int32),
    )


@pytest.fixture
def synthetic_connectome():
    return _build
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_connectome.py -v`
Expected: FAIL, `ImportError` on `flymon.brain.connectome`

- [ ] **Step 4: connectome.py 구현**

```python
"""Connectome arrays, our npz schema, sign/hemisphere corrections, CSC out-edge build."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Params

SCHEMA = {
    "bodyId": np.int64, "type": None, "cls": None, "sc": None, "nt": None,
    "sign": np.int8, "side": None, "pre": np.int32, "post": np.int32, "w": np.int32,
}
SIGN_OF_NT = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}


@dataclass
class Connectome:
    bodyId: np.ndarray
    type: np.ndarray
    cls: np.ndarray
    sc: np.ndarray
    nt: np.ndarray
    sign: np.ndarray
    side: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    w: np.ndarray

    @property
    def N(self) -> int:
        return int(self.bodyId.shape[0])

    @property
    def E(self) -> int:
        return int(self.pre.shape[0])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in SCHEMA})

    @classmethod
    def load(cls, path: str | Path) -> "Connectome":
        d = np.load(path, allow_pickle=False)
        missing = [k for k in SCHEMA if k not in d.files]
        if missing:
            raise ValueError(f"npz missing arrays: {missing}")
        arrs = {}
        for k, dt in SCHEMA.items():
            a = d[k]
            arrs[k] = a.astype(dt) if dt is not None else a.astype(str)
        c = cls(**arrs)
        if c.pre.max(initial=-1) >= c.N or c.post.max(initial=-1) >= c.N:
            raise ValueError("edge index out of range")
        return c


def apply_sign_override(conn: Connectome, params: Params) -> tuple[np.ndarray, int]:
    """Re-sign cell types by prefix (spec 3.1: lLN1/lLN2 are inhibitory). Returns (sign, n_changed)."""
    sign = conn.sign.astype(np.int8).copy()
    n = 0
    for prefix, s in params.sign_override:
        m = np.char.startswith(conn.type, prefix)
        n += int(m.sum())
        sign[m] = s
    return sign, n


def hemisphere_scale(conn: Connectome) -> float:
    """inL / inR: total synapse count onto left vs right postsynaptic cells. Applied to R inputs."""
    in_l = float(conn.w[conn.side[conn.post] == "L"].sum())
    in_r = float(conn.w[conn.side[conn.post] == "R"].sum())
    return in_l / in_r if in_l > 0 and in_r > 0 else 1.0


@dataclass
class CSC:
    """Out-edges grouped by presynaptic neuron: targets tgt[ptr[i]:ptr[i+1]] with signed mV weights."""
    ptr: np.ndarray
    tgt: np.ndarray
    w: np.ndarray

    @property
    def N(self) -> int:
        return int(self.ptr.shape[0] - 1)

    def pre_of_edge(self) -> np.ndarray:
        return np.repeat(np.arange(self.N, dtype=np.int32), np.diff(self.ptr))


def build_csc(conn: Connectome, params: Params, apl_idx: np.ndarray) -> CSC:
    sign, _ = apply_sign_override(conn, params)
    keep = (conn.w >= params.min_weight) & (sign[conn.pre] != 0)
    pre, post, w = conn.pre[keep], conn.post[keep], conn.w[keep].astype(np.float64)
    mv = sign[pre] * w * params.mv_per_synapse
    if params.balance_hemispheres:
        scale = hemisphere_scale(conn)
        mv = np.where(conn.side[post] == "R", mv * scale, mv)
    is_apl = np.zeros(conn.N, bool)
    is_apl[apl_idx] = True
    mv = np.where(is_apl[pre], mv * params.apl_scale, mv)
    order = np.argsort(pre, kind="stable")
    pre, post, mv = pre[order], post[order], mv[order]
    counts = np.bincount(pre, minlength=conn.N)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return CSC(ptr=ptr, tgt=post.astype(np.int32), w=mv.astype(np.float32))
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_connectome.py -v`
Expected: 6 passed

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/connectome.py tests/conftest.py tests/brain/test_connectome.py
git commit -m "feat(brain): connectome schema, sign override, hemisphere balance, CSC build"
```

---

### Task 3: 데이터 빌드 (Janelia feather → npz + manifest)

**Files:**
- Create: `flymon/brain/data_build.py`, `docs/data.md`
- Test: `tests/brain/test_data_build.py`

**Interfaces:**
- Consumes: `Connectome`, `SIGN_OF_NT`.
- Produces: CLI `uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz [--min-weight 1]`, 함수 `build(data_dir, out_path, min_weight=1) -> dict` (manifest). 빌드는 **모든** 엣지(w ≥ 1)를 저장하고, 임계 적용은 `build_csc`가 한다(구획 표는 원시 가중치가 필요).
- `data/malecns.manifest.json`: `{"inputs": {파일명: {"sha256", "bytes"}}, "n_neurons", "n_edges", "min_weight", "counts": {"Kenyon_Cell", "MBON", "DAN", "ALPN", "olfactory", "APL", "lLN"}, "built_at"}`.

**사용자 액션(이 태스크 전에 필요)**: https://male-cns.janelia.org/download/ 에서 neuPrint 계정으로 로그인해 아래 세 파일을 `data/raw/`에 받는다. 없으면 실제 데이터 테스트는 skip된다.
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.1 GB)
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (13 MB)
- `body-neurotransmitters-male-cns-v1.0.feather` (42 MB)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_data_build.py`

```python
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flymon.brain.connectome import Connectome
from flymon.brain.data_build import build

RAW = Path("data/raw")
FILES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
]


def _write_fake_dataset(d: Path):
    d.mkdir(parents=True)
    ann = pa.table({
        "bodyId": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "type": ["ORN_DM1", "KCab-m", "MBON01", None, "lLN1_a"],
        "class": ["olfactory", "Kenyon_Cell", "MBON", None, "ALLN"],
        "superclass": ["cb_sensory", "cb_intrinsic", "cb_intrinsic", None, "cb_intrinsic"],
        "somaSide": ["M", "L", "R", "L", "L"],
        "rootSide": ["L", None, None, None, None],
        "status": ["Traced", "Traced", "Traced", "Traced", "Orphan"],
    })
    feather.write_feather(ann, d / FILES[1])
    nt = pa.table({"body": pa.array([1, 2, 3, 3], pa.int64()),
                   "consensus_nt": ["acetylcholine", "acetylcholine", "glutamate", "glutamate"]})
    feather.write_feather(nt, d / FILES[2])
    wt = pa.table({"body_pre": pa.array([1, 2, 2, 3, 9], pa.int64()),
                   "body_post": pa.array([2, 3, 3, 1, 1], pa.int64()),
                   "weight": pa.array([12, 7, 2, 1, 5], pa.int64())})
    # write as IPC file so record-batch streaming works
    with pa.OSFile(str(d / FILES[0]), "wb") as sink:
        with pa.ipc.new_file(sink, wt.schema) as writer:
            writer.write_table(wt, max_chunksize=2)


def test_build_from_fake_feathers(tmp_path):
    raw = tmp_path / "raw"
    _write_fake_dataset(raw)
    out = tmp_path / "c.npz"
    manifest = build(raw, out, min_weight=1)
    c = Connectome.load(out)
    # neuron 4 (no type) and neuron 5 (not Traced) dropped
    assert c.N == 3 and list(c.bodyId) == [1, 2, 3]
    assert list(c.type) == ["ORN_DM1", "KCab-m", "MBON01"]
    assert list(c.sign) == [1, 1, -1]
    assert list(c.side) == ["L", "L", "R"]           # ORN side from rootSide
    # edges: 1->2 (12), 2->3 (7), 2->3 (2), 3->1 (1); 9->1 dropped (unknown body)
    assert c.E == 4
    assert manifest["n_neurons"] == 3 and manifest["n_edges"] == 4
    assert set(manifest["inputs"]) == set(FILES)
    assert all(len(v["sha256"]) == 64 for v in manifest["inputs"].values())
    assert json.loads((out.with_suffix(".manifest.json")).read_text())["n_edges"] == 4


@pytest.mark.skipif(not all((RAW / f).exists() for f in FILES), reason="MaleCNS raw files not downloaded")
def test_build_real_dataset_counts(tmp_path):
    out = tmp_path / "malecns.npz"
    m = build(RAW, out, min_weight=1)
    assert 160_000 <= m["n_neurons"] <= 170_000
    assert m["counts"]["Kenyon_Cell"] == 4064
    assert m["counts"]["MBON"] == 97
    assert m["counts"]["APL"] == 2
    assert m["counts"]["lLN"] == 151
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_data_build.py -v`
Expected: FAIL, `ImportError` on `flymon.brain.data_build`

- [ ] **Step 3: data_build.py 구현**

```python
"""Build data/malecns.npz from the three MaleCNS v1.0 feather files (CC-BY 4.0, HHMI Janelia/Google).

Usage: uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from .connectome import SIGN_OF_NT, Connectome

WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOT = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = "body-neurotransmitters-male-cns-v1.0.feather"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _side(ann) -> np.ndarray:
    """Sensory somas sit outside the volume (somaSide 'M'); use rootSide when it is L/R."""
    soma = ann["somaSide"].fillna("").astype(str)
    root = ann["rootSide"].fillna("").astype(str)
    side = root.where(root.isin(["L", "R"]), soma)
    return side.where(side.isin(["L", "R"]), "M").to_numpy().astype(str)


def build(data_dir: str | Path, out_path: str | Path, min_weight: int = 1) -> dict:
    data_dir, out_path = Path(data_dir), Path(out_path)
    ann = feather.read_table(
        data_dir / ANNOT,
        columns=["bodyId", "type", "class", "superclass", "somaSide", "rootSide", "status"],
    ).to_pandas()
    ann = ann[(ann["status"] == "Traced") & ann["type"].notna()].sort_values("bodyId").reset_index(drop=True)
    ids = ann["bodyId"].to_numpy(dtype=np.int64)

    nt = feather.read_table(data_dir / NT, columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    ntv = nt.reindex(ids).fillna("unknown").to_numpy().astype(str)
    sign = np.array([SIGN_OF_NT.get(x, 0) for x in ntv], dtype=np.int8)

    # streaming edge read: the weights file is ~1.1 GB, so go batch by batch
    lookup = np.full(ids.max() + 1, -1, dtype=np.int64) if ids.max() < 50_000_000 else None
    if lookup is not None:
        lookup[ids] = np.arange(len(ids))

    def to_index(b: np.ndarray) -> np.ndarray:
        if lookup is not None:
            ok = (b >= 0) & (b < len(lookup))
            out = np.full(b.shape, -1, np.int64)
            out[ok] = lookup[b[ok]]
            return out
        i = np.searchsorted(ids, b)
        i[i >= len(ids)] = 0
        return np.where(ids[i] == b, i, -1)

    P, Q, W = [], [], []
    reader = pa.ipc.open_file(pa.memory_map(str(data_dir / WEIGHTS)))
    for bi in range(reader.num_record_batches):
        bt = reader.get_batch(bi)
        w = bt.column("weight").to_numpy(zero_copy_only=False).astype(np.int64)
        k = w >= min_weight
        if not k.any():
            continue
        pre = to_index(bt.column("body_pre").to_numpy(zero_copy_only=False)[k].astype(np.int64))
        post = to_index(bt.column("body_post").to_numpy(zero_copy_only=False)[k].astype(np.int64))
        ok = (pre >= 0) & (post >= 0)
        P.append(pre[ok].astype(np.int32)); Q.append(post[ok].astype(np.int32)); W.append(w[k][ok].astype(np.int32))
    pre = np.concatenate(P) if P else np.zeros(0, np.int32)
    post = np.concatenate(Q) if Q else np.zeros(0, np.int32)
    w = np.concatenate(W) if W else np.zeros(0, np.int32)

    conn = Connectome(
        bodyId=ids,
        type=ann["type"].to_numpy().astype(str),
        cls=ann["class"].fillna("").to_numpy().astype(str),
        sc=ann["superclass"].fillna("").to_numpy().astype(str),
        nt=ntv, sign=sign, side=_side(ann), pre=pre, post=post, w=w,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn.save(out_path)

    types = conn.type
    manifest = {
        "inputs": {f: {"sha256": _sha256(data_dir / f), "bytes": (data_dir / f).stat().st_size} for f in (WEIGHTS, ANNOT, NT)},
        "n_neurons": conn.N, "n_edges": conn.E, "min_weight": min_weight,
        "counts": {
            "Kenyon_Cell": int((conn.cls == "Kenyon_Cell").sum()),
            "MBON": int((conn.cls == "MBON").sum()),
            "DAN": int((conn.cls == "DAN").sum()),
            "ALPN": int((conn.cls == "ALPN").sum()),
            "olfactory": int((conn.cls == "olfactory").sum()),
            "APL": int((types == "APL").sum()),
            "lLN": int(np.char.startswith(types, "lLN1").sum() + np.char.startswith(types, "lLN2").sum()),
        },
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="data/malecns.npz")
    ap.add_argument("--min-weight", type=int, default=1)
    a = ap.parse_args()
    m = build(a.data, a.out, a.min_weight)
    print(json.dumps({k: m[k] for k in ("n_neurons", "n_edges", "counts")}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: docs/data.md 작성**

```markdown
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
```

- [ ] **Step 5: 합성 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_data_build.py -v`
Expected: `test_build_from_fake_feathers` PASS, `test_build_real_dataset_counts` SKIPPED(파일 없을 때) 또는 PASS

- [ ] **Step 6: 실제 데이터가 있으면 빌드 실행**

Run: `uv run python -m flymon.brain.data_build --data data/raw --out data/malecns.npz`
Expected: 뉴런 수 약 162,000, `counts.Kenyon_Cell == 4064`, `counts.MBON == 97`. `docs/data.md` 표에 manifest의 해시를 기입한다.

- [ ] **Step 7: 커밋**

```bash
git add flymon/brain/data_build.py tests/brain/test_data_build.py docs/data.md
git commit -m "feat(brain): build MaleCNS npz from Janelia feathers with manifest"
```

---

### Task 4: 인구 정의와 도파민 구획 표

**Files:**
- Create: `flymon/brain/circuits.py`
- Test: `tests/brain/test_circuits.py`

**Interfaces:**
- Consumes: `Connectome`, `Params`.
- Produces:
  - `Populations` dataclass: `receptor_types: dict[str, np.ndarray]` (53 ORN 타입 → 인덱스), `receptor_side: dict[str, dict[str, np.ndarray]]` ('L'/'R'/'M'), `sensory: np.ndarray` (포아송 소스 전체: class ∈ {olfactory, gustatory, mechanosensory, mechanosensory_tactile, mechanosensory_proprioceptive, thermosensory, hygrosensory, visual}), `alpn`, `kc`, `mbon`, `dan_types: dict[str, np.ndarray]` (예: 'PAM08' → 세포들), `pam`, `ppl1`, `apl`, `dn`, `mn`.
  - `Populations.from_connectome(conn) -> Populations`.
  - `compartments(conn, pops, core_frac) -> dict[str, Compartment]`; `Compartment(family:str, cells:np.ndarray, w_mbon:np.ndarray[N] (0..1, 타입 최대로 정규화), core:np.ndarray (MBON 인덱스 중 w ≥ core_frac))`.
  - `export_compartments(comps, conn, path)` → JSON `{type: {family, n_cells, core_mbon_types: [...]}}`.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_circuits.py`

```python
import json

import numpy as np

from flymon.brain.circuits import Populations, compartments, export_compartments


def test_populations_from_synthetic(synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    assert len(p.kc) == 40 and len(p.mbon) == 4 and len(p.apl) == 1
    assert set(p.dan_types) == {"PAM08", "PPL105"}
    assert len(p.pam) == 2 and len(p.ppl1) == 2
    assert set(p.receptor_types) == {"ORN_DM1", "ORN_DA1", "ORN_VA2", "ORN_DM6", "ORN_VC1"}
    assert sum(len(v) for v in p.receptor_types.values()) == 20
    assert len(p.sensory) == 20 and len(p.dn) == 6 and len(p.mn) == 4
    # receptor sides: synthetic ORNs are 'M'
    assert len(p.receptor_side["ORN_DM1"]["M"]) == 4


def test_compartments_core_excludes_strays(synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    comps = compartments(c, p, core_frac=0.2)
    pam, ppl = comps["PAM08"], comps["PPL105"]
    assert pam.family == "PAM" and ppl.family == "PPL1"
    # synthetic: PAM08 -> MBON01/02 (60 synapses), stray 3 synapses -> not core
    assert set(c.type[pam.core]) == {"MBON01", "MBON02"}
    assert set(c.type[ppl.core]) == {"MBON03", "MBON04"}
    assert pam.w_mbon.max() == 1.0
    assert (pam.w_mbon[ppl.core] < 0.2).all()


def test_export_compartments(tmp_path, synthetic_connectome):
    c = synthetic_connectome()
    p = Populations.from_connectome(c)
    comps = compartments(c, p, 0.2)
    path = tmp_path / "comps.json"
    export_compartments(comps, c, path)
    d = json.loads(path.read_text())
    assert d["PAM08"]["core_mbon_types"] == ["MBON01", "MBON02"]
    assert d["PPL105"]["n_cells"] == 2
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_circuits.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: circuits.py 구현**

```python
"""Named populations and the dopamine compartment table, derived from annotations and raw edges."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .connectome import Connectome

SENSORY_CLASSES = ("olfactory", "gustatory", "mechanosensory", "mechanosensory_tactile",
                   "mechanosensory_proprioceptive", "thermosensory", "hygrosensory", "visual")


@dataclass
class Populations:
    receptor_types: dict
    receptor_side: dict
    sensory: np.ndarray
    alpn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    dan_types: dict
    pam: np.ndarray
    ppl1: np.ndarray
    apl: np.ndarray
    dn: np.ndarray
    mn: np.ndarray

    @classmethod
    def from_connectome(cls, c: Connectome) -> "Populations":
        t, k, s = c.type, c.cls, c.sc
        is_orn = np.char.startswith(t, "ORN_")
        receptor_types, receptor_side = {}, {}
        for name in sorted(set(t[is_orn])):
            idx = np.flatnonzero(t == name)
            receptor_types[name] = idx
            receptor_side[name] = {sd: idx[c.side[idx] == sd] for sd in ("L", "R", "M")}
        dan = np.flatnonzero(k == "DAN")
        dan_types = {}
        for name in sorted(set(t[dan])):
            if name.startswith("PAM") or name.startswith("PPL1"):
                dan_types[name] = dan[t[dan] == name]
        return cls(
            receptor_types=receptor_types, receptor_side=receptor_side,
            sensory=np.flatnonzero(np.isin(k, SENSORY_CLASSES)),
            alpn=np.flatnonzero(k == "ALPN"), kc=np.flatnonzero(k == "Kenyon_Cell"),
            mbon=np.flatnonzero(k == "MBON"), dan_types=dan_types,
            pam=dan[np.char.startswith(t[dan], "PAM")], ppl1=dan[np.char.startswith(t[dan], "PPL1")],
            apl=np.flatnonzero(t == "APL"),
            dn=np.flatnonzero(s == "descending_neuron"), mn=np.flatnonzero(s == "vnc_motor"),
        )


@dataclass
class Compartment:
    family: str
    cells: np.ndarray
    w_mbon: np.ndarray   # [N], DAN->MBON synapse mass normalised to the type's peak, 0 elsewhere
    core: np.ndarray     # MBON indices with w_mbon >= core_frac


def compartments(c: Connectome, p: Populations, core_frac: float) -> dict:
    """One compartment per DAN type from the RAW DAN->MBON edge list (dopamine modulates, it does
    not transmit, so these edges carry no current in the CSC and must be read from raw arrays)."""
    is_mbon = np.zeros(c.N, bool)
    is_mbon[p.mbon] = True
    out = {}
    for name, cells in p.dan_types.items():
        m = np.isin(c.pre, cells) & is_mbon[c.post]
        acc = np.zeros(c.N, np.float32)
        np.add.at(acc, c.post[m], c.w[m].astype(np.float32))
        peak = float(acc.max())
        if peak <= 0:
            continue
        w = acc / peak
        core = np.flatnonzero(w >= core_frac)
        out[name] = Compartment(family="PAM" if name.startswith("PAM") else "PPL1", cells=cells, w_mbon=w, core=core)
    return out


def export_compartments(comps: dict, c: Connectome, path: str | Path) -> None:
    d = {name: {"family": cp.family, "n_cells": int(len(cp.cells)),
                "core_mbon_types": sorted(set(c.type[cp.core].tolist()))}
         for name, cp in comps.items()}
    Path(path).write_text(json.dumps(d, indent=2))
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_circuits.py -v`
Expected: 3 passed

- [ ] **Step 5: 실제 데이터가 있으면 구획 표 확인**

Run:
```bash
uv run python -c "
from flymon.brain.connectome import Connectome
from flymon.brain.circuits import Populations, compartments, export_compartments
c = Connectome.load('data/malecns.npz'); p = Populations.from_connectome(c)
comps = compartments(c, p, 0.2); export_compartments(comps, c, 'results/summary/compartments.json')
print(len(p.receptor_types), 'receptor types;', 'PPL105 core:', sorted(set(c.type[comps['PPL105'].core])), 'PAM08 core:', sorted(set(c.type[comps['PAM08'].core])))
"
```
Expected: 53 receptor types. PPL105 core에 MBON13/18/23, PAM08 core에 MBON05/21이 포함되고 두 core가 겹치지 않는다(flybrain 측정과 일치). 다르면 `results/summary/compartments.json`에 그대로 남기고 README의 "측정된 것"에 적는다. 이 값 자체는 게이트가 아니다.

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/circuits.py tests/brain/test_circuits.py results/summary/compartments.json
git commit -m "feat(brain): populations and DAN->MBON compartment table"
```

---

### Task 5: CPU LIF 엔진 핵심

**Files:**
- Create: `flymon/brain/engine_cpu.py`
- Test: `tests/brain/test_engine_cpu.py`

**Interfaces:**
- Consumes: `Connectome`, `Populations`, `Params`, `build_csc`.
- Produces `Engine`:
  - `Engine(conn, pops, params, seed=0)`; 속성 `N`, `csc`, `v_th:float32[N]`, `ext:float32[N]`(mV, 지속 구동), `drive_hz:float32[N]`, `v`, `g`, `t_ms`.
  - `reset(seed=None)`: 상태·지연 링·RNG 초기화(같은 seed → 같은 잡음·포아송 흐름).
  - `set_ext(idx, mv)`, `clear_ext()`(MBON hold는 유지), `set_drive_hz(idx, hz)`, `clear_drive()`.
  - `step() -> np.ndarray` 이번 ms에 발화한 뉴런 인덱스.
  - `run(ms, count_idx=None) -> np.ndarray[N] 스파이크 수` (count_idx가 있으면 그 부분만이 아니라 전체 N 배열을 돌려주되 계산은 동일).
  - `propagate(src_idx) -> float32[N]` 이벤트 구동 시냅스 입력(테스트용 공개).
  - 훅: `on_step: callable(engine, spikes) | None` (가소성이 붙는다).

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_engine_cpu.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine


def _engine(synthetic_connectome, **kw):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False, mbon_hold_frac=0.0, **kw)
    return Engine(c, Populations.from_connectome(c), p, seed=1), c


def test_constant_drive_follows_discrete_rc_and_fires_at_step_24(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[0])   # a DN: no inputs fire in a quiet net
    eng.set_ext([i], 10.0)
    v_prev, fired_at = 0.0, None
    for n in range(1, 40):
        spk = eng.step()
        if i in spk:
            fired_at = n
            break
        expected = v_prev + (10.0 - v_prev) * (1.0 / 20.0)   # Euler, dt=1, tau_m=20
        assert eng.v[i] == pytest.approx(expected, abs=1e-5)
        v_prev = expected
    assert fired_at == 24        # 10*(1-0.95^n) >= 7  ->  n = 24


def test_spike_arrives_after_delay_with_alpha_kernel(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    kc = np.flatnonzero(c.cls == "Kenyon_Cell")
    mbon = np.flatnonzero(c.cls == "MBON")
    a, b = int(kc[0]), int(mbon[0])
    w_ab = c.w[(c.pre == a) & (c.post == b)].sum() * eng.p.mv_per_synapse
    eng.set_ext([a], 1000.0)     # dv = 50 mV on step 1 -> fires on step 1
    s1 = eng.step()
    assert a in s1
    assert eng.g[b] == 0.0
    eng.step()                   # delay = 2 steps: arrives on the 3rd step
    assert eng.g[b] == 0.0
    eng.step()                   # arrival is added, then the step's tau_syn decay (x0.8) applies
    assert eng.g[b] == pytest.approx(w_ab * (1 - 1 / 5.0), rel=1e-6)
    g_after_arrival = eng.g[b]
    eng.set_ext([a], 0.0)
    eng.step()
    assert eng.g[b] == pytest.approx(g_after_arrival * (1 - 1 / 5.0), rel=1e-6)   # tau_syn decay


def test_propagate_matches_dense_matrix(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    csc = eng.csc
    M = np.zeros((c.N, c.N))
    for i in range(c.N):
        np.add.at(M[:, i], csc.tgt[csc.ptr[i]:csc.ptr[i + 1]], csc.w[csc.ptr[i]:csc.ptr[i + 1]])
    src = np.array([0, 5, 17, 33])
    np.testing.assert_allclose(eng.propagate(src), M[:, src].sum(axis=1), atol=1e-5)


def test_refractory_blocks_immediate_refire(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[1])
    eng.set_ext([i], 1000.0)
    fired = [n for n in range(6) if i in eng.step()]
    assert fired == [0, 3]       # fires, 2 refractory steps, fires again


def test_poisson_receptor_rate(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    orn = np.flatnonzero(c.cls == "olfactory")
    eng.set_drive_hz(orn, 100.0)
    counts = eng.run(5000)
    rate = counts[orn].mean() / 5.0
    assert 75 <= rate <= 92      # p=0.1/step with 2 refractory steps -> 0.1/(1+0.2) = 83 Hz


def test_reset_with_seed_is_deterministic(synthetic_connectome):
    eng, c = _engine(synthetic_connectome, noise_mv=0.15)
    orn = np.flatnonzero(c.cls == "olfactory")
    eng.set_drive_hz(orn, 150.0)
    eng.reset(seed=7); a = eng.run(300)
    eng.reset(seed=7); b = eng.run(300)
    np.testing.assert_array_equal(a, b)


def test_mbon_hold_sets_tonic_ext(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False)
    eng = Engine(c, Populations.from_connectome(c), p)
    kc = np.flatnonzero(c.cls == "Kenyon_Cell")
    assert (eng.ext[np.flatnonzero(c.cls == "MBON")] == pytest.approx(0.85 * 7.0)).all()
    assert (eng.ext[kc] == 0).all()


def test_kc_threshold_normalised_by_pn_input(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False, kc_thresh=1.5)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p)
    pn_in = np.zeros(c.N)
    m = np.isin(c.pre, pops.alpn) & np.isin(c.post, pops.kc)
    np.add.at(pn_in, c.post[m], c.w[m])
    med = np.median(pn_in[pops.kc])
    expect = 7.0 * 1.5 * np.clip(pn_in[pops.kc] / med, 0.5, 3.0)
    np.testing.assert_allclose(eng.v_th[pops.kc], expect, rtol=1e-6)
    assert (eng.v_th[pops.mbon] == 7.0).all()
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_engine_cpu.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: engine_cpu.py 구현**

```python
"""Event-driven leaky integrate-and-fire engine over the MaleCNS connectome (reference CPU implementation).

Membrane (mV above rest):  v <- v + (-v + g + ext) * dt/tau_m + noise, while not refractory
Alpha synapse:             g <- g + W on arrival of a spike emitted syn_delay ago;  g <- g * (1 - dt/tau_syn)
Threshold:                 spike when v >= v_th; v <- v_reset; refractory for refrac_steps
Receptors (sensory classes) ignore the membrane and fire as Poisson sources at drive_hz.
Reproduction target for the design decisions (MBON hold, KC threshold normalisation, APL scale):
flybrain FINDINGS.md; constants: Shiu et al. 2024.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .circuits import Populations
from .config import Params
from .connectome import Connectome, build_csc


class Engine:
    def __init__(self, conn: Connectome, pops: Populations, params: Params, seed: int = 0):
        self.p = params
        self.conn, self.pops = conn, pops
        self.N = conn.N
        self.csc = build_csc(conn, params, pops.apl)
        self.on_step = None

        # thresholds: KC thresholds normalised by their PN input (our design decision)
        self.v_th = np.full(self.N, params.v_thresh, np.float32)
        pn_in = np.zeros(self.N, np.float64)
        m = np.isin(conn.pre, pops.alpn) & np.isin(conn.post, pops.kc)
        np.add.at(pn_in, conn.post[m], conn.w[m])
        med = float(np.median(pn_in[pops.kc])) if len(pops.kc) else 1.0
        if med > 0:
            lo, hi = params.kc_norm_clip
            self.v_th[pops.kc] = params.v_thresh * params.kc_thresh * np.clip(pn_in[pops.kc] / med, lo, hi)

        # tonic drive: MBON hold (our design decision); everything else 0 until set_ext
        self.ext0 = np.zeros(self.N, np.float32)
        self.ext0[pops.mbon] = params.mbon_hold_frac * params.v_thresh
        self.ext = self.ext0.copy()

        self.is_receptor = np.zeros(self.N, bool)
        self.is_receptor[pops.sensory] = True
        self.receptor_idx = pops.sensory.astype(np.int64)
        self.drive_hz = np.zeros(self.N, np.float32)

        self._seed = seed
        self.reset(seed)

    # ---- state -------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.v = np.zeros(self.N, np.float32)
        self.g = np.zeros(self.N, np.float32)
        self.refrac = np.zeros(self.N, np.int32)
        # delay line: spikes emitted at step t are popped (delivered) at step t + dly_steps.
        # The deque holds dly_steps entries; step() pops one at the start and appends this step's
        # spikes at the end, so a 2-step delay means: fire on step 1 -> arrive on step 3.
        self.delay = deque([np.zeros(0, np.int64) for _ in range(self.p.dly_steps())], maxlen=self.p.dly_steps())
        self.last = np.zeros(0, np.int64)
        self.t_ms = 0.0

    def set_ext(self, idx, mv: float) -> None:
        self.ext[np.asarray(idx, dtype=np.int64)] = np.float32(mv)

    def clear_ext(self) -> None:
        self.ext = self.ext0.copy()

    def set_drive_hz(self, idx, hz: float) -> None:
        self.drive_hz[np.asarray(idx, dtype=np.int64)] = np.float32(hz)

    def clear_drive(self) -> None:
        self.drive_hz[:] = 0.0

    # ---- dynamics ----------------------------------------------------------------------------
    def propagate(self, src: np.ndarray) -> np.ndarray:
        """Sum signed mV of every out-edge of the spiking sources into a dense [N] vector."""
        out = np.zeros(self.N, np.float32)
        if src.size == 0:
            return out
        ptr, tgt, w = self.csc.ptr, self.csc.tgt, self.csc.w
        starts, ends = ptr[src], ptr[src + 1]
        total = int((ends - starts).sum())
        if total == 0:
            return out
        idx = np.concatenate([tgt[a:b] for a, b in zip(starts, ends)])
        val = np.concatenate([w[a:b] for a, b in zip(starts, ends)])
        np.add.at(out, idx, val)
        return out

    def step(self) -> np.ndarray:
        p = self.p
        arrived = self.delay.popleft()
        if arrived.size:
            self.g += self.propagate(arrived)
        dt_m = p.dt / p.tau_m
        dv = (-self.v + self.g + self.ext) * dt_m
        if p.noise_mv:
            dv += self.rng.standard_normal(self.N, dtype=np.float32) * p.noise_mv
        free = self.refrac <= 0
        self.v = np.where(free, self.v + dv, self.v)
        self.refrac = np.where(free, self.refrac, self.refrac - 1)
        np.maximum(self.v, -p.v_thresh, out=self.v)
        self.g *= (1.0 - p.dt / p.tau_syn)
        spk = (self.v >= self.v_th) & free
        # receptors: Poisson at commanded rate, membrane ignored
        ri = self.receptor_idx
        hz = self.drive_hz[ri]
        pois = (self.rng.random(ri.size) < hz * (p.dt / 1000.0)) & free[ri]
        spk[ri] = pois
        fired = np.flatnonzero(spk)
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refrac_steps()
        self.last = fired
        self.delay.append(fired)          # delivered dly_steps steps from now
        self.t_ms += p.dt
        if self.on_step is not None:
            self.on_step(self, fired)
        return fired

    def run(self, ms: float, count_idx=None) -> np.ndarray:
        counts = np.zeros(self.N, np.int32)
        for _ in range(int(round(ms / self.p.dt))):
            counts[self.step()] += 1
        return counts
```

한 step의 순서는 "지연 큐에서 도착분 꺼내 g에 더함 → 적분·역치 → g 감쇠 → 이번 step 스파이크를 큐 끝에 넣음"이다. 이 순서라야 지연이 정확히 `dly_steps`가 되고, 도착 step의 g가 `W × 0.8`이 된다(테스트 2가 이를 고정한다).

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_engine_cpu.py -v`
Expected: 8 passed

- [ ] **Step 5: 실제 데이터 성능 스모크(있을 때)**

Run:
```bash
uv run python -c "
import time, numpy as np
from flymon.brain.connectome import Connectome
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
c = Connectome.load('data/malecns.npz'); p = Populations.from_connectome(c)
e = Engine(c, p, Params(), seed=0)
e.set_drive_hz(p.receptor_types['ORN_DM1'], 70.0)
t = time.time(); n = e.run(200); dt = (time.time() - t) / 200 * 1000
print(f'{dt:.1f} ms/step, spikes/ms {n.sum()/200:.0f}, KC active {(n[p.kc] > 0).mean():.3f}')
"
```
Expected: 10 ms/step 이하(M5 Pro). 20 ms/step을 넘으면 `propagate`의 리스트 컴프리헨션을 `np.repeat` 기반 벡터화로 바꾼다(테스트 `test_propagate_matches_dense_matrix`가 동등성을 보장).

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/engine_cpu.py tests/brain/test_engine_cpu.py
git commit -m "feat(brain): event-driven LIF engine with Poisson receptors, MBON hold, KC threshold normalisation"
```

---

### Task 6: 냄새 제시와 설계된 냄새 쌍

**Files:**
- Create: `flymon/brain/stimuli.py`
- Test: `tests/brain/test_stimuli.py`

**Interfaces:**
- Consumes: `Engine`, `Populations`, `Params`.
- Produces:
  - `Odor = dict[str, float]` (수용체 타입 → 채널 강도 0..1).
  - `channel_strengths(pops, types, equalize=True) -> Odor`: 강도 ∝ 1/수용체 수, 평균이 1이 되도록 정규화(균등화 끄면 전부 1.0).
  - `present(engine, pops, odor, strength) -> None`: `drive_hz[receptors of type] = max_rate_hz × strength × channel`. 이전 후각 구동은 지운다.
  - `design_odor_pair(pops, k=8, exclude=("ORN_DA1", "ORN_V"), seed=0) -> tuple[Odor, Odor]`: 제외 타입을 뺀 뒤 수용체 수 오름차순으로 정렬해 번갈아 A/B에 배정(드라이브 매칭), 각 k개, 서로 겹치지 않음.
  - `total_drive(pops, odor) -> float` = Σ 채널 강도 × 수용체 수.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_stimuli.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.stimuli import channel_strengths, design_odor_pair, present, total_drive


def test_channel_strengths_equalise_by_receptor_count(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome())
    od = channel_strengths(p, ["ORN_DM1", "ORN_DA1"], equalize=True)
    n = {t: len(p.receptor_types[t]) for t in od}
    assert od["ORN_DM1"] * n["ORN_DM1"] == pytest.approx(od["ORN_DA1"] * n["ORN_DA1"])
    assert np.mean(list(od.values())) == pytest.approx(1.0)


def test_present_sets_rates_and_clears_previous(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False))
    present(eng, pops, {"ORN_DM1": 1.0}, strength=0.35)
    assert (eng.drive_hz[pops.receptor_types["ORN_DM1"]] == pytest.approx(200 * 0.35)).all()
    present(eng, pops, {"ORN_DA1": 0.5}, strength=1.0)
    assert (eng.drive_hz[pops.receptor_types["ORN_DM1"]] == 0).all()
    assert (eng.drive_hz[pops.receptor_types["ORN_DA1"]] == pytest.approx(100.0)).all()


def test_design_pair_disjoint_and_drive_matched(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome())
    a, b = design_odor_pair(p, k=2, exclude=("ORN_DA1",), seed=0)
    assert len(a) == 2 and len(b) == 2 and not set(a) & set(b)
    assert "ORN_DA1" not in a and "ORN_DA1" not in b
    assert abs(total_drive(p, a) - total_drive(p, b)) / total_drive(p, a) < 0.25
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_stimuli.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: stimuli.py 구현**

```python
"""Odours are defined over olfactory receptor TYPES (glomeruli), never over individual receptors."""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine

Odor = dict


def channel_strengths(pops: Populations, types, equalize: bool = True) -> Odor:
    types = list(types)
    if not equalize:
        return {t: 1.0 for t in types}
    inv = np.array([1.0 / len(pops.receptor_types[t]) for t in types])
    inv = inv / inv.mean()
    return {t: float(s) for t, s in zip(types, inv)}


def total_drive(pops: Populations, odor: Odor) -> float:
    return float(sum(s * len(pops.receptor_types[t]) for t, s in odor.items()))


def present(engine: Engine, pops: Populations, odor: Odor, strength: float) -> None:
    for t in pops.receptor_types:
        engine.drive_hz[pops.receptor_types[t]] = 0.0
    for t, s in odor.items():
        engine.drive_hz[pops.receptor_types[t]] = np.float32(engine.p.max_rate_hz * strength * s)


def design_odor_pair(pops: Populations, k: int = 8, exclude=("ORN_DA1", "ORN_V"), seed: int = 0):
    """Two disjoint k-glomerulus odours with matched total receptor drive.
    Sort candidate types by receptor count, take the 2k smallest-variance middle band, alternate A/B."""
    cand = [t for t in pops.receptor_types if t not in exclude]
    cand.sort(key=lambda t: len(pops.receptor_types[t]))
    if len(cand) < 2 * k:
        raise ValueError(f"need {2 * k} receptor types, have {len(cand)}")
    mid = len(cand) // 2
    band = cand[max(0, mid - k):mid + k]
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        band = band[::-1]
    a_types, b_types = band[0::2], band[1::2]
    return channel_strengths(pops, a_types), channel_strengths(pops, b_types)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_stimuli.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add flymon/brain/stimuli.py tests/brain/test_stimuli.py
git commit -m "feat(brain): odour presentation over receptor types and drive-matched odour pairs"
```

---

### Task 7: KC→MBON 가소성

**Files:**
- Create: `flymon/brain/plasticity.py`
- Test: `tests/brain/test_plasticity.py`

**Interfaces:**
- Consumes: `Engine`, `Populations`, `compartments()` 결과, `Params`.
- Produces `Plasticity`:
  - `Plasticity(engine, pops, comps)`: CSC 안의 KC→MBON 엣지 인덱스 `edges`, `w0`(복사), `pre_kc_local`, `post_mbon`, 자취 `kc_trace[n_kc]`, `da[n_mbon]`, `da_base[n_mbon]`. 생성 시 `engine.on_step = self.on_step`을 건다.
  - `enabled: bool` (False면 자취는 갱신하되 가중치를 바꾸지 않는다).
  - `drive_dan(type_name, mv)` / `quiet_dan()`: 해당 DAN 타입 세포의 `engine.ext`를 mv로 / ext0으로.
  - `on_step(engine, fired)`: 규칙 한 번.
  - `weights_frac() -> float` 평균 w/w0, `weights_frac_by_mbon_set(mbon_idx) -> float`.
  - `recover_pulse()`: `w += (w0 − w) × recovery_per_pulse`, 바닥 유지.
  - `reset_weights()`, `set_enabled(bool)`.
  - 규칙(스펙 3.1):
    ```
    kc_trace *= 1 − dt/kc_trace_ms;  kc_trace[fired KC] += dt/kc_trace_ms
    da *= 1 − dt/da_trace_ms;  for each DAN type with n spikes this step: da += w_mbon[type][mbon] × n / n_cells × dt/da_trace_ms
    da_base += (da − da_base) × dt/da_baseline_ms;  phasic = max(da − da_base, 0)
    coincide = (kc_trace[pre] × kc_trace_scale) × (phasic[post] × da_trace_scale)
    w *= 1 − learn_rate × tanh(coincide);  w = max(w, w0 × min_weight_frac)
    ```
    구획 표 `w_mbon`은 core 밖 MBON을 0으로 자른 버전을 쓴다(핵심 구획만 가르친다, 스펙 1절 conditioning4c).

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_plasticity.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity


def _setup(synthetic_connectome, **kw):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False, learn_rate=0.05, **kw)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=3)
    comps = compartments(c, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    return c, pops, eng, pl


def _drive_kcs(eng, pops, k=10, mv=60.0):
    eng.set_ext(pops.kc[:k], mv)


def test_plastic_edges_are_exactly_kc_to_mbon(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    pre = eng.csc.pre_of_edge()[pl.edges]
    post = eng.csc.tgt[pl.edges]
    assert np.isin(pre, pops.kc).all() and np.isin(post, pops.mbon).all()
    assert len(pl.edges) == 40 * 4


def test_no_change_without_dopamine(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    _drive_kcs(eng, pops)
    eng.run(300)
    assert pl.weights_frac() == pytest.approx(1.0)


def test_no_change_without_kc_activity(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    pl.drive_dan("PAM08", 70.0)
    eng.run(300)
    assert pl.weights_frac() == pytest.approx(1.0)


def test_coincidence_depresses_only_taught_compartment(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    _drive_kcs(eng, pops)
    pl.drive_dan("PAM08", 70.0)
    eng.run(400)
    pam_core = compartments(c, pops, 0.2)["PAM08"].core
    ppl_core = compartments(c, pops, 0.2)["PPL105"].core
    assert pl.weights_frac_by_mbon_set(pam_core) < 0.99
    assert pl.weights_frac_by_mbon_set(ppl_core) == pytest.approx(1.0)


def test_floor_and_disabled(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5, min_weight_frac=0.2)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PAM08", 70.0)
    eng.run(2000)
    w = eng.csc.w[pl.edges]
    assert (w / pl.w0 >= 0.2 - 1e-6).all()
    pam_core = compartments(c, pops, 0.2)["PAM08"].core
    assert pl.weights_frac_by_mbon_set(pam_core) == pytest.approx(0.2, abs=1e-3)   # taught edges hit the floor
    assert pl.weights_frac() < 0.7                                                  # untaught PPL core edges stay at 1.0
    pl.set_enabled(False)
    before = eng.csc.w[pl.edges].copy()
    eng.run(200)
    np.testing.assert_array_equal(eng.csc.w[pl.edges], before)


def test_recover_pulse_moves_toward_w0(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5, recovery_per_pulse=0.5)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PAM08", 70.0)
    eng.run(1000)
    f0 = pl.weights_frac()
    pl.recover_pulse()
    f1 = pl.weights_frac()
    assert f1 == pytest.approx(f0 + (1.0 - f0) * 0.5, abs=1e-6)


def test_reset_weights_restores_w0(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PPL105", 70.0)
    eng.run(500)
    assert pl.weights_frac() < 1.0
    pl.reset_weights()
    assert pl.weights_frac() == pytest.approx(1.0)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_plasticity.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: plasticity.py 구현**

```python
"""Dopamine-gated depression of KC->MBON synapses (three-factor rule), compartment by DAN type."""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine


class Plasticity:
    def __init__(self, engine: Engine, pops: Populations, comps: dict):
        self.eng, self.pops, self.p = engine, pops, engine.p
        csc = engine.csc
        pre = csc.pre_of_edge()
        is_kc = np.zeros(engine.N, bool); is_kc[pops.kc] = True
        is_mbon = np.zeros(engine.N, bool); is_mbon[pops.mbon] = True
        self.edges = np.flatnonzero(is_kc[pre] & is_mbon[csc.tgt])
        self.w0 = csc.w[self.edges].copy()
        kc_local = np.full(engine.N, -1, np.int64); kc_local[pops.kc] = np.arange(len(pops.kc))
        mb_local = np.full(engine.N, -1, np.int64); mb_local[pops.mbon] = np.arange(len(pops.mbon))
        self.pre_kc = kc_local[pre[self.edges]]
        self.post_mb = mb_local[csc.tgt[self.edges]]
        self.mb_local = mb_local
        # per DAN type: cells and MBON weight vector restricted to the core compartment
        self.types = {}
        for name, cp in comps.items():
            w = np.zeros(len(pops.mbon), np.float32)
            w[mb_local[cp.core]] = cp.w_mbon[cp.core]
            self.types[name] = (cp.cells.astype(np.int64), w)
        self.kc_trace = np.zeros(len(pops.kc), np.float32)
        self.da = np.zeros(len(pops.mbon), np.float32)
        self.da_base = np.zeros(len(pops.mbon), np.float32)
        self.enabled = True
        engine.on_step = self.on_step

    # ---- dopamine drive ------------------------------------------------------------------
    def drive_dan(self, type_name: str, mv: float) -> None:
        cells, _ = self.types[type_name]
        self.eng.set_ext(cells, mv)

    def quiet_dan(self) -> None:
        for cells, _ in self.types.values():
            self.eng.ext[cells] = self.eng.ext0[cells]

    # ---- rule ----------------------------------------------------------------------------
    def on_step(self, engine: Engine, fired: np.ndarray) -> None:
        p, dt = self.p, self.p.dt
        self.kc_trace *= (1.0 - dt / p.kc_trace_ms)
        kf = self.pops.kc
        fired_kc = fired[np.isin(fired, kf)]
        if fired_kc.size:
            self.kc_trace[np.searchsorted(kf, fired_kc)] += dt / p.kc_trace_ms
        self.da *= (1.0 - dt / p.da_trace_ms)
        for cells, wvec in self.types.values():
            n = int(np.isin(fired, cells).sum())
            if n:
                self.da += wvec * (n / len(cells)) * (dt / p.da_trace_ms)
        self.da_base += (self.da - self.da_base) * (dt / p.da_baseline_ms)
        if not self.enabled:
            return
        phasic = np.maximum(self.da - self.da_base, 0.0)
        coincide = (self.kc_trace[self.pre_kc] * p.kc_trace_scale) * (phasic[self.post_mb] * p.da_trace_scale)
        if coincide.any():
            w = self.eng.csc.w
            w[self.edges] *= (1.0 - p.learn_rate * np.tanh(coincide)).astype(np.float32)
            np.maximum(w[self.edges], self.w0 * p.min_weight_frac, out=w[self.edges])

    # ---- bookkeeping ---------------------------------------------------------------------
    def weights_frac(self) -> float:
        return float(np.mean(self.eng.csc.w[self.edges] / self.w0))

    def weights_frac_by_mbon_set(self, mbon_idx) -> float:
        sel = np.isin(self.post_mb, self.mb_local[np.asarray(mbon_idx)])
        return float(np.mean(self.eng.csc.w[self.edges][sel] / self.w0[sel]))

    def recover_pulse(self) -> None:
        r = self.p.recovery_per_pulse
        if r <= 0:
            return
        w = self.eng.csc.w
        w[self.edges] += (self.w0 - w[self.edges]) * np.float32(r)

    def reset_weights(self) -> None:
        self.eng.csc.w[self.edges] = self.w0
        self.kc_trace[:] = 0; self.da[:] = 0; self.da_base[:] = 0

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_plasticity.py -v`
Expected: 7 passed. `test_coincidence_depresses_only_taught_compartment`가 "< 0.99"를 못 넘으면 kc_trace_scale·da_trace_scale이 너무 작은 것이므로 Params 기본값을 2배씩 올려 재실행하고, 바뀐 값을 config.py에 반영한다.

- [ ] **Step 5: 커밋**

```bash
git add flymon/brain/plasticity.py tests/brain/test_plasticity.py flymon/brain/config.py
git commit -m "feat(brain): dopamine-gated KC->MBON depression with per-type core compartments, floor, recovery"
```

---

### Task 8: 재현 1 — 케니언세포 희소성과 MBON 기저 발화

**Files:**
- Create: `scripts/reproduce_flybrain_measurements.py` (`sparsity` 서브커맨드), `flymon/brain/measure.py`
- Test: `tests/brain/test_measure.py`

**Interfaces:**
- Consumes: 전부.
- Produces `flymon/brain/measure.py`:
  - `kc_sparsity(engine, pops, odor, strength, seed, settle_ms=200, read_ms=600) -> dict` = `{"frac_active": KC 중 read 구간에 1회 이상 발화한 비율, "active": bool[n_kc], "mbon_hz": MBON 평균 Hz, "kc_hz"}`. 제시 전 `engine.reset(seed)`, `present(...)`, settle 뒤 read 구간만 센다.
  - `jaccard(a: bool[], b: bool[]) -> float`, `chance_jaccard(pa, pb) -> float` = `pa·pb / (pa + pb − pa·pb)`.
  - `mbon_baseline(engine, pops, seed, ms=1000) -> dict` = 무자극 상태의 MBON 평균 Hz와 발화하는 MBON 타입 수.
- 스크립트 `uv run python scripts/reproduce_flybrain_measurements.py sparsity --npz data/malecns.npz --out results/m0/sparsity.json --kc-thresh 1.0 1.5 --apl-scale 0.1 0.2 --strength 0.35 --seeds 3`: 설계된 냄새 쌍(A, B)에 대해 격자마다 `frac_active_A/B`, `jaccard`, `chance`, `mbon_hz_A/B`, 기저 `mbon_hz_rest`를 기록.
- **합격 범위(스펙 5절 M0)**: 어떤 격자점에서 A와 B 모두 `frac_active ∈ [0.03, 0.07]`, `jaccard ≤ chance`, `mbon_hz_rest ∈ [2, 6]`. 그 격자점의 kc_thresh·apl_scale을 `Params` 기본값으로 채택한다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_measure.py`

```python
import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline
from flymon.brain.stimuli import design_odor_pair


def test_jaccard_and_chance():
    a = np.array([1, 1, 0, 0], bool); b = np.array([1, 0, 1, 0], bool)
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert chance_jaccard(0.05, 0.05) == pytest.approx(0.0025 / 0.0975)


def test_kc_sparsity_runs_on_synthetic(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, Params(min_weight=1, balance_hemispheres=False), seed=0)
    a, b = design_odor_pair(pops, k=2, exclude=("ORN_DA1",))
    r = kc_sparsity(eng, pops, a, strength=1.0, seed=0, settle_ms=50, read_ms=100)
    assert 0.0 <= r["frac_active"] <= 1.0 and r["active"].shape == (40,)
    base = mbon_baseline(eng, pops, seed=0, ms=100)
    assert "mbon_hz" in base and "n_types_active" in base


@pytest.mark.skipif(not Path("data/malecns.npz").exists(), reason="real connectome not built")
def test_real_sparsity_gate_recorded():
    """Passes only after Task 8 Step 5 found a grid point in range and wrote results/m0/sparsity.json."""
    d = json.loads(Path("results/m0/sparsity.json").read_text())
    ok = [g for g in d["grid"] if 0.03 <= g["frac_active_A"] <= 0.07 and 0.03 <= g["frac_active_B"] <= 0.07
          and g["jaccard"] <= g["chance"] and 2.0 <= d["mbon_hz_rest"] <= 6.0]
    assert ok, "no grid point met the KC-sparsity / MBON-baseline gate"
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_measure.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: measure.py 구현**

```python
"""Measurements used by the M0 gate: Kenyon-cell sparsity/overlap and MBON baseline."""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .stimuli import present


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def chance_jaccard(pa: float, pb: float) -> float:
    return float(pa * pb / (pa + pb - pa * pb)) if (pa + pb - pa * pb) > 0 else 0.0


def kc_sparsity(engine: Engine, pops: Populations, odor, strength: float, seed: int,
                settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    engine.reset(seed)
    engine.clear_drive()
    present(engine, pops, odor, strength)
    engine.run(settle_ms)
    counts = engine.run(read_ms)
    active = counts[pops.kc] > 0
    sec = read_ms / 1000.0
    return {
        "frac_active": float(active.mean()),
        "active": active,
        "kc_hz": float(counts[pops.kc].mean() / sec),
        "mbon_hz": float(counts[pops.mbon].mean() / sec),
    }


def mbon_baseline(engine: Engine, pops: Populations, seed: int, ms: float = 1000.0) -> dict:
    engine.reset(seed)
    engine.clear_drive()
    counts = engine.run(ms)
    hz = counts[pops.mbon] / (ms / 1000.0)
    types = engine.conn.type[pops.mbon]
    return {"mbon_hz": float(hz.mean()), "n_types_active": int(len(set(types[hz > 0].tolist())))}
```

- [ ] **Step 4: 스크립트 `sparsity` 서브커맨드 구현** — `scripts/reproduce_flybrain_measurements.py`

```python
#!/usr/bin/env python3
"""Reproduce the three flybrain measurements that gate the M0 engine.

  sparsity      Kenyon-cell sparsity/overlap for a designed odour pair + MBON baseline
  conditioning  paired-seed olfactory conditioning with reversal (Task 9)
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline
from flymon.brain.stimuli import design_odor_pair, total_drive


def cmd_sparsity(a):
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    grid = []
    for kc_thresh, apl in itertools.product(a.kc_thresh, a.apl_scale):
        p = Params(kc_thresh=kc_thresh, apl_scale=apl)
        eng = Engine(conn, pops, p, seed=0)
        rows = []
        for s in range(a.seeds):
            ra = kc_sparsity(eng, pops, odor_a, a.strength, seed=100 + s)
            rb = kc_sparsity(eng, pops, odor_b, a.strength, seed=100 + s)
            rows.append({"frac_active_A": ra["frac_active"], "frac_active_B": rb["frac_active"],
                         "jaccard": jaccard(ra["active"], rb["active"]),
                         "chance": chance_jaccard(ra["frac_active"], rb["frac_active"]),
                         "mbon_hz_A": ra["mbon_hz"], "mbon_hz_B": rb["mbon_hz"]})
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        grid.append({"kc_thresh": kc_thresh, "apl_scale": apl, **mean})
        print(json.dumps(grid[-1]), flush=True)
    base = mbon_baseline(Engine(conn, pops, Params(), seed=0), pops, seed=100)
    out = {"odor_A": odor_a, "odor_B": odor_b, "drive_A": total_drive(pops, odor_a), "drive_B": total_drive(pops, odor_b),
           "strength": a.strength, "grid": grid, "mbon_hz_rest": base["mbon_hz"], "mbon_types_active_rest": base["n_types_active"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sparsity")
    s.add_argument("--npz", default="data/malecns.npz"); s.add_argument("--out", default="results/m0/sparsity.json")
    s.add_argument("--kc-thresh", type=float, nargs="+", default=[1.0, 1.5]); s.add_argument("--apl-scale", type=float, nargs="+", default=[0.1, 0.2])
    s.add_argument("--strength", type=float, default=0.35); s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--k", type=int, default=8); s.add_argument("--odor-seed", type=int, default=0)
    s.set_defaults(fn=cmd_sparsity)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 합성 테스트 통과 확인, 실제 데이터로 격자 실행**

Run: `uv run pytest tests/brain/test_measure.py -v` → 2 passed, 실제 데이터 테스트는 파일 없으면 skip.
Run(실제 데이터): `uv run python scripts/reproduce_flybrain_measurements.py sparsity`
Expected: 격자 4점 × 3시드 × 2냄새 × 0.8 s ≈ 24 제시, 약 10분. 합격 격자점이 있으면 그 `kc_thresh`, `apl_scale`을 `Params` 기본값으로 바꾸고 `test_real_sparsity_gate_recorded`가 통과하는지 확인한다. 합격점이 없으면 격자를 넓힌다: `--kc-thresh 0.8 1.0 1.25 1.5 2.0 --apl-scale 0.05 0.1 0.2 0.3`. 그래도 없으면 `Engine`의 KC 정규화 클립(`kc_norm_clip`)을 (0.5, 2.0)으로 좁혀 재시도하고, 시도한 격자와 결과를 전부 `results/m0/sparsity_*.json`으로 남긴다.

- [ ] **Step 6: 커밋**

```bash
git add flymon/brain/measure.py scripts/reproduce_flybrain_measurements.py tests/brain/test_measure.py flymon/brain/config.py
git commit -m "feat(brain): KC sparsity / MBON baseline measurement and reproduction script"
```

---

### Task 9: 재현 2 — 짝지은 시드 조건화와 반전 (엔진 합격 게이트)

**Files:**
- Create: `flymon/brain/conditioning.py`
- Modify: `scripts/reproduce_flybrain_measurements.py` (`conditioning` 서브커맨드 추가)
- Test: `tests/brain/test_conditioning.py`

**Interfaces:**
- Consumes: `Engine`, `Plasticity`, `compartments`, `design_odor_pair`, `present`.
- Produces `flymon/brain/conditioning.py`:
  - `Readout(a_core: np.ndarray, p_core: np.ndarray)`: PPL105 core(접근 MBON)와 PAM08 core(회피 MBON) 인덱스.
  - `probe(engine, pl, pops, readout, odor, strength, seed, settle_ms=200, read_ms=600) -> dict[str,int]`: `{"A": Σ a_core 스파이크, "P": Σ p_core 스파이크}`. 가소성은 probe 동안 `enabled=False`로 내렸다가 원래 값으로 되돌린다.
  - `disc(x_plus, x_minus) -> float` = `(x+ − x−)/(x+ + x− + 1e-9)`.
  - `D(readout, plus_counts, minus_counts) -> float` = `disc(A+, A−) − disc(P+, P−)`.
  - `train_block(engine, pl, pops, cs_plus, cs_minus, strength, seed, punish: str|None, reward: str|None, trials=12, present_ms=800, gap_ms=200)`: 각 trial마다 `reset(seed+trial)` → cs_plus 제시 + (punish면 `drive_dan(punish, dan_drive_mv)`) present_ms → quiet → gap → cs_minus 제시 + (reward면 `drive_dan(reward, ...)`) present_ms → quiet → gap. 가소성 `enabled=True`.
  - `run_arm(engine, pl, pops, readout, cs_plus, cs_minus, strength, seed, arm: str) -> dict`: `arm ∈ {"both", "reversed", "noplast", "punish_only", "reward_only"}`. 절차: `pl.reset_weights()` → pre = probe(cs+), probe(cs−) (같은 seed) → train_block (arm에 따라 punish/reward 설정; noplast는 `enabled=False`로 같은 자극) → post = probe(cs+), probe(cs−) (같은 seed) → `{"D_pre", "D_post", "dD", "weights_frac"}`.
  - `reversal_test(engine, pl, pops, readout, cs_plus, cs_minus, strength, seeds: list[int]) -> dict`: 시드별 5팔 결과, `n_flip`(both와 reversed의 dD 부호가 반대인 시드 수), `noplast_max_abs_dD`, 팔별 평균·표준편차.
- 스크립트 `conditioning --npz data/malecns.npz --out results/m0/conditioning.json --seeds 8 --jobs 8 [--learn-rate ... --kc-trace-scale ... --da-trace-scale ... --recovery 0]`: 시드를 `multiprocessing` 프로세스로 병렬 실행(각 프로세스가 자기 Engine을 만든다).
- **합격(스펙 5절 M0 엔진 게이트)**: `n_flip == 8/8`, `noplast_max_abs_dD == 0.0`, `mean |dD_both| ≥ 0.3`, `mean |dD_reversed| ≥ 0.3`, `punish_only`와 `reward_only`의 dD 합이 `both`의 dD와 같은 부호. 이 게이트를 통과한 파라미터를 `Params` 기본값으로 동결한다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/brain/test_conditioning.py`

```python
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import D, Readout, disc, probe, run_arm, train_block
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair


def _setup(synthetic_connectome):
    c = synthetic_connectome()
    # kc_thresh 0.5 so the tiny synthetic olfactory pathway reliably drives Kenyon cells
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=0)
    comps = compartments(c, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    ro = Readout(a_core=comps["PPL105"].core, p_core=comps["PAM08"].core)
    a, b = design_odor_pair(pops, k=2, exclude=("ORN_DA1",))
    return c, pops, eng, pl, ro, a, b


def test_disc_and_D():
    assert disc(3, 1) == pytest.approx(0.5)
    assert disc(0, 0) == pytest.approx(0.0)
    ro = Readout(a_core=np.array([0]), p_core=np.array([1]))
    assert D(ro, {"A": 3, "P": 1}, {"A": 1, "P": 3}) == pytest.approx(0.5 - (-0.5))


def test_probe_is_paired_by_seed_and_leaves_weights(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    r1 = probe(eng, pl, pops, ro, a, 1.0, seed=5, settle_ms=50, read_ms=100)
    r2 = probe(eng, pl, pops, ro, a, 1.0, seed=5, settle_ms=50, read_ms=100)
    assert r1 == r2
    assert pl.weights_frac() == pytest.approx(1.0)
    assert pl.enabled is True     # restored after the probe


def test_noplast_arm_gives_exactly_zero(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    r = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=2, arm="noplast", trials=2, present_ms=100, gap_ms=20,
                settle_ms=50, read_ms=100)
    assert r["dD"] == 0.0 and r["weights_frac"] == pytest.approx(1.0)


def test_train_block_changes_weights_when_dan_driven(synthetic_connectome):
    c, pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish="PPL105", reward="PAM08", trials=2, present_ms=200, gap_ms=20)
    assert pl.weights_frac() < 1.0
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `uv run pytest tests/brain/test_conditioning.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: conditioning.py 구현**

```python
"""Paired-seed olfactory conditioning with reversal: the engine acceptance gate (spec 5, M0).

Protocol (reproduction target: flybrain conditioning4c):
  pre-test  : probe CS+ and CS- with the same noise seed (plasticity off)
  training  : N trials of CS+ paired with one DAN type (punishment PPL105 or reward PAM08),
              then CS- paired with the other
  post-test : same probes, same seed -> the no-plasticity arm is exactly 0.0
Readout D = disc over the PPL105 core (approach MBONs) minus disc over the PAM08 core (avoidance MBONs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .plasticity import Plasticity
from .stimuli import present


@dataclass
class Readout:
    a_core: np.ndarray
    p_core: np.ndarray


def disc(x_plus: float, x_minus: float) -> float:
    return float((x_plus - x_minus) / (x_plus + x_minus + 1e-9))


def D(ro: Readout, plus: dict, minus: dict) -> float:
    return disc(plus["A"], minus["A"]) - disc(plus["P"], minus["P"])


def probe(engine: Engine, pl: Plasticity, pops: Populations, ro: Readout, odor, strength: float, seed: int,
          settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    was = pl.enabled
    pl.set_enabled(False)
    engine.reset(seed)
    engine.clear_drive()
    pl.quiet_dan()
    present(engine, pops, odor, strength)
    engine.run(settle_ms)
    counts = engine.run(read_ms)
    pl.set_enabled(was)
    return {"A": int(counts[ro.a_core].sum()), "P": int(counts[ro.p_core].sum())}


def train_block(engine: Engine, pl: Plasticity, pops: Populations, cs_plus, cs_minus, strength: float, seed: int,
                punish: str | None, reward: str | None, trials: int = 12, present_ms: float = 800.0,
                gap_ms: float = 200.0) -> None:
    p = engine.p
    for t in range(trials):
        for odor, dan in ((cs_plus, punish), (cs_minus, reward)):
            engine.reset(seed * 1000 + t)
            engine.clear_drive()
            pl.quiet_dan()
            present(engine, pops, odor, strength)
            if dan is not None:
                pl.drive_dan(dan, p.dan_drive_mv)
            engine.run(present_ms)
            pl.quiet_dan()
            engine.clear_drive()
            engine.run(gap_ms)
            pl.recover_pulse()


ARMS = {
    # arm: (DAN type paired with the odour in the CS+ slot, DAN type paired with the CS- slot, plasticity on)
    "both": ("PPL105", "PAM08", True),
    # reversed = same odours, same dopamine amounts, channels exchanged; sign of dD must flip
    "reversed": ("PAM08", "PPL105", True),
    "noplast": ("PPL105", "PAM08", False),
    "punish_only": ("PPL105", None, True),
    "reward_only": (None, "PAM08", True),
}


def run_arm(engine: Engine, pl: Plasticity, pops: Populations, ro: Readout, cs_plus, cs_minus, strength: float,
            seed: int, arm: str, trials: int = 12, present_ms: float = 800.0, gap_ms: float = 200.0,
            settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    punish, reward, plastic = ARMS[arm]
    pl.reset_weights()
    pre = D(ro, probe(engine, pl, pops, ro, cs_plus, strength, seed, settle_ms, read_ms),
            probe(engine, pl, pops, ro, cs_minus, strength, seed, settle_ms, read_ms))
    pl.set_enabled(plastic)
    train_block(engine, pl, pops, cs_plus, cs_minus, strength, seed, punish, reward, trials, present_ms, gap_ms)
    pl.set_enabled(True)
    post = D(ro, probe(engine, pl, pops, ro, cs_plus, strength, seed, settle_ms, read_ms),
             probe(engine, pl, pops, ro, cs_minus, strength, seed, settle_ms, read_ms))
    return {"arm": arm, "seed": seed, "D_pre": pre, "D_post": post, "dD": post - pre, "weights_frac": pl.weights_frac()}


def reversal_test(engine, pl, pops, ro, cs_plus, cs_minus, strength, seeds, **kw) -> dict:
    per_seed = {s: {arm: run_arm(engine, pl, pops, ro, cs_plus, cs_minus, strength, s, arm, **kw) for arm in ARMS} for s in seeds}
    return summarise(per_seed)


def summarise(per_seed: dict) -> dict:
    seeds = list(per_seed)
    flips = sum(1 for s in seeds if np.sign(per_seed[s]["both"]["dD"]) == -np.sign(per_seed[s]["reversed"]["dD"])
                and per_seed[s]["both"]["dD"] != 0)
    arms = {arm: {"mean_dD": float(np.mean([per_seed[s][arm]["dD"] for s in seeds])),
                  "sd_dD": float(np.std([per_seed[s][arm]["dD"] for s in seeds])),
                  "mean_weights_frac": float(np.mean([per_seed[s][arm]["weights_frac"] for s in seeds]))}
            for arm in ARMS}
    return {"n_seeds": len(seeds), "n_flip": flips,
            "noplast_max_abs_dD": float(max(abs(per_seed[s]["noplast"]["dD"]) for s in seeds)),
            "arms": arms, "per_seed": {str(s): per_seed[s] for s in seeds}}
```

주의: `run_arm`의 "reversed" 팔은 냄새 정체성과 판독 기준(readout frame)을 그대로 두고 **어느 도파민 종류가 어느 냄새와 짝지어지는지**만 바꿔서 반전을 구현한다. 냄새를 바꿔 끼우면 D(a, b) = −D(b, a)라는 판독의 반대칭성 때문에 학습이 없어도 부호가 뒤집혀 보이므로 절대 그렇게 하지 않는다. 두 팔은 **같은 양의 도파민**을 받고, 같은 냄새로 프로브하며, 부호만 뒤집혀야 한다.

- [ ] **Step 4: 스크립트에 `conditioning` 서브커맨드 추가** — `scripts/reproduce_flybrain_measurements.py`

```python
def _cond_worker(args):
    npz, params_dict, seed, strength, k, odor_seed, kw = args
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    p = Params(**params_dict)
    eng = Engine(conn, pops, p, seed=seed)
    comps = compartments(conn, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    ro = Readout(a_core=comps["PPL105"].core, p_core=comps["PAM08"].core)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    return seed, {arm: run_arm(eng, pl, pops, ro, a, b, strength, seed, arm, **kw) for arm in ARMS}


def cmd_conditioning(a):
    from multiprocessing import Pool
    params_dict = dict(learn_rate=a.learn_rate, kc_trace_scale=a.kc_trace_scale, da_trace_scale=a.da_trace_scale,
                       recovery_per_pulse=a.recovery, kc_thresh=a.kc_thresh, apl_scale=a.apl_scale)
    kw = dict(trials=a.trials, present_ms=a.present_ms)
    jobs = [(a.npz, params_dict, s, a.strength, a.k, a.odor_seed, kw) for s in range(a.seeds)]
    with Pool(a.jobs) as pool:
        per_seed = dict(pool.map(_cond_worker, jobs))
    out = {"params": params_dict, "strength": a.strength, "trials": a.trials, **summarise(per_seed)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_seeds", "n_flip", "noplast_max_abs_dD")}), json.dumps(out["arms"], indent=1))
```

`main()`에 서브파서를 추가한다:

```python
    c = sub.add_parser("conditioning")
    c.add_argument("--npz", default="data/malecns.npz"); c.add_argument("--out", default="results/m0/conditioning.json")
    c.add_argument("--seeds", type=int, default=8); c.add_argument("--jobs", type=int, default=8)
    c.add_argument("--trials", type=int, default=12); c.add_argument("--present-ms", type=float, default=800.0)
    c.add_argument("--strength", type=float, default=0.35); c.add_argument("--k", type=int, default=8); c.add_argument("--odor-seed", type=int, default=0)
    c.add_argument("--learn-rate", type=float, default=Params().learn_rate)
    c.add_argument("--kc-trace-scale", type=float, default=Params().kc_trace_scale)
    c.add_argument("--da-trace-scale", type=float, default=Params().da_trace_scale)
    c.add_argument("--recovery", type=float, default=0.0)
    c.add_argument("--kc-thresh", type=float, default=Params().kc_thresh); c.add_argument("--apl-scale", type=float, default=Params().apl_scale)
    c.set_defaults(fn=cmd_conditioning)
```

상단 import에 `from flymon.brain.conditioning import ARMS, Readout, run_arm, summarise`와 `from flymon.brain.plasticity import Plasticity`를 추가한다.

- [ ] **Step 5: 합성 테스트 통과 확인**

Run: `uv run pytest tests/brain/test_conditioning.py -v`
Expected: 4 passed

- [ ] **Step 6: 실제 데이터로 게이트 실행과 튜닝**

Run: `uv run python scripts/reproduce_flybrain_measurements.py conditioning --seeds 8 --jobs 8`
Expected 소요: 시드당 5팔 × (4 probe + 24 학습 제시) × 약 1 s(0.8 s 제시 + gap) × CPU 속도(약 5 ms/step → 제시당 5 s) ≈ 시드당 12분, 8프로세스 병렬로 약 15분.

합격 조건: `n_flip == 8`, `noplast_max_abs_dD == 0.0`, `arms.both.mean_dD`와 `arms.reversed.mean_dD`의 절댓값이 0.3 이상이고 부호가 반대, `punish_only.mean_dD + reward_only.mean_dD`가 `both.mean_dD`와 같은 부호.

불합격 시 순서대로, 한 번에 하나만 바꾸고 각 실행 결과를 `results/m0/conditioning_<태그>.json`으로 보관한다:
1. `noplast_max_abs_dD != 0.0`이면 버그다(짝지은 시드가 깨짐). `Engine.reset(seed)`가 RNG·상태·지연 링을 전부 초기화하는지, `probe`가 `enabled=False`로 도는지 확인.
2. `both`가 거의 0이면 학습이 약하다: `--kc-trace-scale`과 `--da-trace-scale`을 2배씩(최대 8배), 그다음 `--learn-rate 1e-3`.
3. `both`와 `reversed`가 같은 부호면 냄새별 학습이 없거나(두 냄새가 같은 KC를 쓴다) 가소성이 포화된 것이다: Task 8 격자에서 Jaccard가 가장 낮은 `kc_thresh`/`apl_scale`을 `--kc-thresh --apl-scale`로 지정해 겹침을 낮춘다.
4. 포화(weights_frac < 0.3)면 `--learn-rate 1e-4`, `--trials 6`.

합격한 파라미터를 `Params` 기본값으로 바꾸고 결과 파일을 `results/m0/conditioning.json`으로 둔다.

- [ ] **Step 7: 게이트 테스트 추가** — `tests/brain/test_conditioning.py` 끝에 추가

```python
@pytest.mark.skipif(not Path("results/m0/conditioning.json").exists(), reason="gate not yet run on real data")
def test_real_conditioning_gate():
    d = json.loads(Path("results/m0/conditioning.json").read_text())
    assert d["n_seeds"] == 8 and d["n_flip"] == 8
    assert d["noplast_max_abs_dD"] == 0.0
    both, rev = d["arms"]["both"]["mean_dD"], d["arms"]["reversed"]["mean_dD"]
    assert abs(both) >= 0.3 and abs(rev) >= 0.3 and np.sign(both) == -np.sign(rev)
    assert np.sign(d["arms"]["punish_only"]["mean_dD"] + d["arms"]["reward_only"]["mean_dD"]) == np.sign(both)
```

파일 상단에 `import json`과 `from pathlib import Path`를 추가한다.

- [ ] **Step 8: 전체 테스트와 커밋**

Run: `uv run pytest -v`
Expected: 모두 통과(실제 데이터 없는 환경에서는 3개 skip).

```bash
git add flymon/brain/conditioning.py scripts/reproduce_flybrain_measurements.py tests/brain/test_conditioning.py flymon/brain/config.py
git commit -m "feat(brain): paired-seed conditioning with reversal; engine gate script"
```

---

### Task 10: M0 게이트 결과 기록과 표기

**Files:**
- Create: `results/summary/m0.json`, `docs/acknowledgments.md`, `README.md`
- Test: `tests/test_summary.py`

**Interfaces:**
- Produces: `results/summary/m0.json` = `{"data_manifest": manifest 요약, "params_frozen": Params 값 전체, "sparsity": 채택 격자점, "conditioning": {"n_flip", "arms 평균"}, "engine_ms_per_step": Task 5 Step 5 측정값}`.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_summary.py`

```python
import json
from pathlib import Path

import pytest

from flymon.brain.config import Params

P = Path("results/summary/m0.json")


@pytest.mark.skipif(not P.exists(), reason="M0 summary not written yet")
def test_summary_matches_frozen_params():
    d = json.loads(P.read_text())
    frozen = d["params_frozen"]
    live = Params().__dict__
    for k in ("kc_thresh", "apl_scale", "learn_rate", "kc_trace_scale", "da_trace_scale", "mbon_hold_frac"):
        assert frozen[k] == live[k], f"{k}: summary {frozen[k]} != Params default {live[k]}"
    assert d["conditioning"]["n_flip"] == 8
```

- [ ] **Step 2: 요약 파일 생성 스크립트 실행**

```bash
uv run python -c "
import json, dataclasses
from pathlib import Path
from flymon.brain.config import Params
man = json.loads(Path('data/malecns.manifest.json').read_text())
sp = json.loads(Path('results/m0/sparsity.json').read_text())
co = json.loads(Path('results/m0/conditioning.json').read_text())
p = Params()
pick = [g for g in sp['grid'] if g['kc_thresh'] == p.kc_thresh and g['apl_scale'] == p.apl_scale][0]
out = {'data_manifest': {k: man[k] for k in ('n_neurons', 'n_edges', 'counts')},
       'inputs_sha256': {k: v['sha256'] for k, v in man['inputs'].items()},
       'params_frozen': dataclasses.asdict(p), 'sparsity': pick, 'mbon_hz_rest': sp['mbon_hz_rest'],
       'conditioning': {'n_flip': co['n_flip'], 'noplast_max_abs_dD': co['noplast_max_abs_dD'], 'arms': co['arms']}}
Path('results/summary/m0.json').write_text(json.dumps(out, indent=2)); print('ok')
"
```

- [ ] **Step 3: docs/acknowledgments.md 작성**

```markdown
# 출처와 감사

- **데이터**: Male CNS Connectome v1.0 — HHMI Janelia FlyEM Project Team, Cambridge Drosophila
  Connectomics Group (MRC LMB), Google Research Connectomics. CC-BY 4.0. https://male-cns.janelia.org/
- **모델 상수**: Shiu et al. 2024, *Nature* 634:210, "A Drosophila computational brain model reveals
  sensorimotor processing".
- **재현 목표**: TheMrRaGe/flybrain의 FINDINGS.md에 기록된 설계 결정(lLN1/lLN2 억제 재지정, APL 스케일,
  MBON 기저 구동, 구획별 도파민, 짝지은 시드 조건화 프로토콜)과 측정값. 이 저장소는 그 코드를 복사하지
  않았고, 문서화된 결정을 독립 구현해 같은 측정을 다시 했다.
- **생물학**: Aso et al. 2014 (eLife), Owald et al. 2015 (Neuron), Lin et al. 2014 (Nat Neurosci),
  Olsen & Wilson 2008 (Nature), Tully & Quinn 1985.
```

- [ ] **Step 4: README.md 초안 작성** (M0 범위만)

```markdown
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
    uv run pytest

데이터 출처와 감사: `docs/acknowledgments.md`.
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `uv run pytest tests/test_summary.py -v`
Expected: PASS(요약 파일이 있을 때)

- [ ] **Step 6: 커밋**

```bash
git add results/summary/m0.json docs/acknowledgments.md README.md tests/test_summary.py
git commit -m "docs: M0 gate summary, acknowledgments, README (measured vs chosen)"
```

---

## 자체 검토

**스펙 커버리지(3.1, 5절 M0, 8절)**:
- 데이터 빌드·해시 → Task 3. CPU 엔진·Shiu 상수·이벤트 구동·포아송·APL·MBON hold·KC 정규화·반구 보정 → Task 2, 5. 회로·구획 표 JSON → Task 4. 가소성·바닥·회복(펄스 기준) → Task 7. 냄새 쌍·균등화 → Task 6. 재현 세 측정과 게이트 → Task 8, 9. 요약·표기·README → Task 10. flybrain 감사·라이선스 이슈는 Task 10의 acknowledgments로 문서화하고, GitHub 이슈 작성 자체는 사용자 액션으로 남긴다.
- 스펙 8절의 "엔진 적분(작은 회로 해석 비교)" → Task 5 테스트 1, 2. "구획 표" → Task 4. "M0 재현 수치 범위" → Task 8, 9 게이트 테스트.
- MPS 엔진·짝지은 잡음·hive는 M0b 계획(별도). 인코더 `odors.py`(배틀 상태 → 냄새)는 M2 계획(별도); M0의 `stimuli.py`는 그 기반이 된다.

**플레이스홀더**: 없음. 튜닝 절차는 순서와 값이 명시됨.

**타입 일관성**: `Engine.csc.w`를 `Plasticity`가 제자리에서 수정하고, `probe`/`run_arm`는 `Plasticity`를 인자로 받는다(테스트와 스크립트 모두 같은 시그니처). `compartments()` 반환 dict의 키는 DAN 타입 문자열, `Readout`은 core 인덱스 배열. `Params`의 튜닝 대상 필드명은 스크립트 CLI 옵션명과 1:1.
