# MBON 보상/처벌 시소: 외부 논문 조사

작성일 2026-10-04. 읽기 전용 세션이며 저장소는 수정하거나 커밋하지 않았다.
상태: **완료**.
인용 원칙: 논문 내용은 Europe PMC 전문(XML)이나 eLife XML에서 해당 문장을 직접 확인한 뒤 적었다. 커넥톰 수치는 저장소의 maleCNS 원본 파일에서 직접 셌다. 확인하지 못한 항목에는 **미확인**을 붙였다.

---

## (a) 한 줄 결론

실제 커넥톰에는 MBON05→MBON13 직접 억제가 없다(1 시냅스).
시소를 설명할 가장 유력한 후보는 문헌에 기록된 MBON 간 피드포워드 억제 사슬이다. MBON05가 GABA/Glu 계열인 MBON11·MBON09·MBON01을 누르면, 이들이 누르던 MBON03(β′2mp, Glu)과 CRE055(GABA)가 풀려난다. 그 둘이 MBON13을 억제한다. 부호로 보면 억제가 세 번 이어지므로 순효과는 음이다.
반대 방향(탈억제) 경로도 구조적으로는 더 많다. 그러므로 부분 제거 f에 따른 MBON13 반응은 선형으로 줄기보다 문턱 근처에서 급하게 꺾일 가능성이 높다. 이 부분은 추정이다.

---

## §0 로컬 커넥톰 수치 (직접 계산)

- 입력: `data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather`. 모델이 쓰는 malecns.npz의 원본이다.
- 신경전달물질: `body-neurotransmitters-male-cns-v1.0.feather`의 `consensus_nt`.
- 모델 부호는 `flymon/brain/connectome.py:16`을 따른다. ACh +1, GABA/Glu/His −1, 도파민 0.
- 간선 문턱은 `min_weight=5`다(`flymon/brain/config.py`).

| 연결 | 시냅스 (좌우 합) | 비고 |
|---|---|---|
| APL → MBON05 | **1162** (APL_R→MBON05_L 594, APL_L→MBON05_R 566) | 지렛대로 지운 2간선. APL은 GABA |
| APL → MBON13 | 22 (12 + 10) | 약하다. 'readout' 제거(아래)에서는 이것도 함께 지워진다 |
| MBON05 → MBON13 | **1** | min_weight 5 아래라 모델에 없다 |
| MBON13 → MBON05 | 1 | 없음 |
| MBON05 → APL | 131 (84 + 47) | Glu(−)이므로 MBON05가 오르면 APL이 내려간다 |
| DPM → MBON05 / → MBON13 | 1253 / 161 | 도파민, 모델 부호 0 |
| MBON05 → MBON30 / MBON11 / MBON20 / MBON09 / MBON01 | 1400 / 406 / 325 / 261 / 132 | MBON05의 MBON 출력 상위 |
| MBON03 → MBON13 | **395** | MBON13이 받는 MBON 입력 중 가장 크다. MBON03은 Glu |
| CRE055 → MBON13 | 497 | GABA |
| LHMB1 → MBON13 | 356 | Glu |
| MBON13 입력 중 KC | KCa′b′-ap2 5271, KCa′b′-m 3876, KCa′b′-ap1 1047 | 시냅스 기준 대부분이 α′β′ KC |

신경전달물질: MBON05는 consensus glutamate(신경세포 단위 예측은 "unclear"), MBON13은 acetylcholine, APL은 GABA, CRE052/055/057은 GABA, MBON03과 LHMB1은 glutamate.

### 경로 부호 분석

질문은 "MBON05가 오르면 MBON13에 순효과가 어느 쪽인가"이다. min_weight 5, 신경세포 단위로 세고, 각 경로의 병목(최소) 시냅스를 합했다.

- **2-hop**: 양(탈억제) 183, 음 5, 0(도파민 경유) 64. MBON05가 CRE052/057/055와 APL을 억제하고 이들이 MBON13을 억제하는 경로는 모두 탈억제다. 즉 시소와 **반대 방향**이다.
- **3-hop**: 양 14145, 음 3472, 0 7429. 양 쪽의 대부분은 MBON05 ⊣ APL ⊣ KCα′β′ → MBON13 경로다. APL을 누르면 α′β′ KC가 풀리는 방향이다.
- **음(시소 방향) 3-hop 상위**:
  - MBON05 ⊣ MBON09 ⊣ CRE055 ⊣ MBON13: 860
  - MBON05 ⊣ MBON09 ⊣ MBON03 ⊣ MBON13: 450
  - MBON05 ⊣ MBON11 ⊣ MBON03 ⊣ MBON13: 257
  - MBON05 ⊣ APL ⊣ MBON03 ⊣ MBON13: 229
  - MBON05 ⊣ MBON01 ⊣ MBON03 ⊣ MBON13: 197
  - MBON05 ⊣ MBON09 ⊣ CRE054·CRE052 ⊣ MBON13
- **저장소 진단과의 대조**: `results/m0d/diag/h4_apl_ablation.json`의 "readout" 조건은 APL→MBON05와 APL→MBON13을 함께 지운다. 여기서도 MBON13은 무너진다(C0: 중앙값 Δ 17.5→0.0, 0 비율 0.094→0.427). APL→MBON13 직접 억제가 사라져도 MBON13이 꺼지므로, 시소는 MBON05를 거치는 네트워크 효과라는 해석과 맞는다. KC 활성 %도 6.80→6.36으로 소폭 내려간다.
- **해석(추정, 측정 안 함)**: 모델은 모든 MBON에 `mbon_hold_frac=0.85` 강장 구동을 준다. 그래서 MBON09·MBON11·MBON01이 쉴 때도 문턱 가까이 있고, 이들을 매개로 한 사슬이 살아 있다. 구조적으로는 양 경로가 더 많은데도 음 사슬이 동역학적으로 이긴 것으로 보인다. 다만 이것은 아직 검증되지 않았다.

---

## (b) 논문별 요약

### 1. Aso et al. 2014a, eLife 3:e04577: MB 해부
- "The neuronal architecture of the mushroom body provides a logic for associative learning." DOI 10.7554/eLife.04577 (PMC4273437)
- **핵심(원문 확인)**
  - "A glutamatergic output (MBON-γ4>γ1γ2) from γ4 sends axons back into the lobes that terminate in γ1 and γ2."
  - MBON-γ4>γ1γ2, MBON-β1>α, MBON-γ1pedc>α/β 세 종이 다른 구획으로 투사해 "multi-layered feedforward network"를 이룬다.
  - "Most of the MBONs having dendrites in the α′/β′ and γ lobes provide such a single-layer readout." 광학현미경 수준에서 α′/β′ MBON은 다른 MBON 입력을 받지 않는 단층 판독으로 분류되었다.
- **관련성**: MBON05(=γ4>γ1γ2)는 원래부터 "다른 구획 MBON을 누르는" 피드포워드 억제 뉴런으로 정의되었다. 다만 표적은 γ1·γ2 구획(MBON11, MBON12=γ2α′1)이고 α′2가 아니다.

### 2. Aso et al. 2014b, eLife 3:e04580: MBON 가치와 행동
- "Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila." DOI 10.7554/eLife.04580 (PMC4273436)
- **핵심(원문 확인)**
  - 여러 MBON을 함께 활성화하면 효과가 더해진다(additive). 같은 부호끼리는 반응이 강해지고 반대 부호끼리는 반응이 줄어든다.
  - "the avoidance-mediating MBON-γ4>γ1γ2 targets the compartments of attraction-mediating MBON-γ2α′1 and MBON-γ1pedc>α/β." 구획 간 연결은 "once local modulation breaks the balance between MBONs… could amplify the differential level of activity of MBONs for opposing effects."
  - MBON-γ4>γ1γ2를 광유전 활성화하면 회피가 나타난다(MB434B).
  - MBON-γ4>γ1γ2와 MBON-α′2는 에탄올 24시간 기억에 선택적으로 필요하다.
- **관련성**
  - 구획 간 억제가 반대 가치 MBON 사이의 차이를 증폭한다는 생각, 즉 시소의 개념적 선례다.
  - **주의**: 문헌에서 MBON05는 활성화하면 **회피**를 일으키는 뉴런이다. 모델은 MBON05를 보상 판독 P로 쓴다. P를 "보상 학습 뒤 반응 감소"로 읽는지, "보상 쪽 증가"로 읽는지 확인이 필요하다. 저장소 정의는 이 세션에서 확인하지 않았다.

### 3. Li et al. 2020, eLife 9:e62576: hemibrain MB 커넥톰
- "The connectome of the adult Drosophila mushroom body provides insights into function." DOI 10.7554/eLife.62576 (PMC7909955, eLife XML v2로 확인)
- **핵심(원문 확인)**
  - MBON05, MBON06, MBON11은 "putatively inhibitory based on their use of GABA or glutamate"이며 피드포워드 MBON 네트워크를 이룬다.
  - MBON05→MBON01 시냅스는 MBON01 수상돌기 뿌리 쪽으로 몰려 있다(분로 억제에 유리한 위치).
  - "MBON30 receives strong input from MBON05 … (260 + 140 + 283 synapses)."
  - MBON11은 문턱 10 시냅스 기준으로 17개 MBON과 연결된다. MBON11 억제가 풀리면 MBON01·MBON03의 CS+ 반응이 커진다. 또한 "analogous feedforward inhibitory connections from MBON09 (γ3β′1) to MBON01 and MBON03."
  - Figure 23 보충 2D: "Localized axo-axonal connections from **MBON03 (β′2mp) to MBON13 (α′2)**."
  - MBON09는 MBON03·MBON01과 MB 밖에서 축삭-축삭으로 서로 연결된다.
  - MBON05는 γ1·γ2 구획에서 PPL1 1개와 PAM 5개에 156 시냅스를 준다.
  - 단서: 글루탐산 MBON30에 대해 "could therefore be either inhibitory or excitatory, depending on the re[ceptor]."
- **관련성**: §0의 음 사슬에 들어가는 고리 둘이 모두 hemibrain에서 기술되어 있다. MBON11/MBON09 ⊣ MBON03, 그리고 MBON03 → MBON13. **미확인**: 논문 Figure 24 화살표에 적힌 시냅스 수치는 그림이라 읽지 못했다. MBON05→MBON11 hemibrain 수치도 마찬가지다.

### 4. Perisse et al. 2016, Neuron 90:1086: MVP2(MBON11)의 피드포워드 억제
- "Aversive learning and appetitive motivation toggle feed-forward inhibition in the Drosophila mushroom body." DOI 10.1016/j.neuron.2016.04.034 (PMC4893166)
- **핵심(원문 확인)**
  - GABA성 MVP2(=MBON-γ1pedc>α/β)는 "feed-forward inhibition selectively inhibits avoidance-directing neural pathways"이다.
  - Figure 4 제목: "MVP2 Neurons Inhibit Odor-Evoked Responses in M4/6, but Not V2αV2α′ MBONs." M4β′는 MBON-β′2mp, M6은 MBON-γ5β′2a다.
  - 혐오 학습은 MVP2의 CS+ 구동을 줄인다(피드포워드 억제가 감소). 배고픔은 이 억제를 키운다.
- **관련성**
  - MBON11 ⊣ MBON03(M4β′)이 생리 실험으로 확인되었다. §0의 MBON05 ⊣ MBON11 ⊣ MBON03 고리 중 뒤쪽 절반의 직접 증거다.
  - MVP2가 V2α′를 억제하지 않는다는 결과는 MBON11 ⊣ MBON13 직접 경로가 약하다는 점과 맞는다(maleCNS MBON11→MBON13은 1). **미확인**: V2α′가 hemibrain의 어느 유형(α′2인지 α′3인지)에 해당하는지는 확인하지 못했다.

### 5. Felsenberg et al. 2018, Cell 175:709: 반대 기억의 병렬 통합(소거)
- DOI 10.1016/j.cell.2018.08.021 (PMC6198041)
- **핵심(원문 확인)**
  - 소거 기억은 처벌이 생략된 경험을 보상으로 저장한 것이다.
  - 혐오 기억 흔적과 새로 생긴 보상(소거) 기억 흔적이 서로 다른 MBON에 공존한다. 둘은 회피를 이끄는 MBON 안에서 합쳐진다. "extinction-evoked plasticity in a pair of these neurons neutralizes the potentiated odor response."
  - 혐오 학습은 접근형 MBON(V2α, MVP2)의 CS+ 반응을 낮추고 회피형 M4β′/M6의 반응을 높인다.
- **관련성**: 보상 신호와 처벌 신호가 같은 MBON 집단 안에서 **상쇄**된다는 점을 보여 준다. 다만 회로가 한쪽을 켜면 다른 쪽이 꺼지는 배타적 시소라는 근거는 아니다.

### 6. Liu & Davis 2009, Nat Neurosci 12:53: APL
- DOI 10.1038/nn.2235 (PMID 19043409). 전문을 받지 못해 초록으로 확인했다.
- **핵심**
  - APL에서 GABA 합성을 줄이면 후각 학습이 좋아진다.
  - APL은 냄새와 전기충격 모두에 반응한다.
  - 냄새와 충격을 짝지으면 APL의 학습 냄새 반응이 줄어든다. 저자들은 이를 "mutual suppression"이라 부른다.
- **관련성**: APL 억제를 **부분적으로** 줄였을 때 행동이 좋아진 선례다. 부분 감소가 반드시 해롭지는 않다는 근거가 된다.

### 7. Lin et al. 2014, Nat Neurosci 17:559: APL 피드백과 희소 부호화
- DOI 10.1038/nn.3660 (PMC4000970)
- **핵심(원문 확인)**
  - APL 출력을 막으면(shibire) KC 반응이 덜 희소해지고 냄새 간 상관이 커진다. 비슷한 냄새의 변별 학습은 깨지지만 다른 냄새의 변별은 유지된다.
  - **부분 조작**: APL>GAD RNAi는 shibire보다 효과가 "significantly smaller"했다. α′ 엽에서만 KC 반응이 소폭 올랐다.
  - "Negative feedback systems such as the Kenyon cell–APL circuit may be especially robust to partial perturbations."
- **관련성**: 부분 억제 감소(graded manipulation)의 실험 선례다. 다만 이 논문이 다룬 것은 APL→KC 피드백 고리다. 우리 지렛대 APL→MBON05는 피드백 고리 밖이라, 같은 "보상(compensation)"이 생긴다고 기대할 수 없다.

### 8. Amin et al. 2020, eLife 9:e56954: APL 국소 억제
- "Localized inhibition in the Drosophila mushroom body." DOI 10.7554/eLife.56954 (PMC7541083)
- **핵심(원문 확인)**
  - APL은 활동전위를 내지 않는(non-spiking) 뉴런이다.
  - APL의 활성과 억제 효과 모두 공간적으로 국소화되어 있어, 구획마다 다르게 억제할 수 있다.
  - "Local inhibition of Kenyon cell output predicts that activity of MBONs near the site of APL activation would be more strongly inhibited than MBONs far away."
- **관련성**: 모델의 APL은 점 뉴런이라 한 번에 전역으로 억제한다. 실제로는 γ4 구획(MBON05)과 α′2 구획(MBON13)의 APL 억제가 따로 움직일 수 있다. 저장소 config 주석에 따르면 graded APL 모드가 이 논문을 근거로 한다.

### 9. Prisco et al. 2021, eLife 10:e74172: APL의 calyx 정규화
- DOI 10.7554/eLife.74172 (PMID 34964714). 초록으로 확인했다.
- **핵심**: APL은 PN 부톤과 KC 수상돌기 양쪽에 억제·상호 시냅스를 만들어 냄새 반응을 정규화한다. APL 반응 크기는 PN 입력 세기에 비례한다.
- **관련성**: 간접적이다. APL이 하는 일이 MBON 억제가 아니라 KC 입력 정규화라는 점을 뒷받침한다.

### 10. Gkanias, McCurdy, Nitabach, Webb 2022, eLife 11:e75611: incentive circuit 모델
- DOI 10.7554/eLife.75611 (PMC8975552). 코드는 github.com/InsectRobotics/IncentiveCircuit.
- **핵심(원문 확인)**
  - MBON 6개와 DAN 6개로 회로를 구성하고, MBON-MBON 연결과 MBON→DAN 연결을 포함했다.
  - "MBON-γ4>γ1γ2 as the avoidance-driving susceptible MBON; and MBON-γ2α′1 as the attraction-driving restrained MBON."
  - "Susceptible MBONs [MBON-γ1pedc>α/β and MBON-γ4>γ1γ2] convulsively break the balance between attraction and avoidance."
- **관련성**: MBON05를 가치 균형을 깨는 역할로 놓은 계산 모델 선례다. 여기서도 MBON05는 **회피** 쪽에 놓여 있다.

### 11. McCurdy et al. 2021, Nat Commun 12:1115: 처벌 생략을 보상으로 부호화
- DOI 10.1038/s41467-021-21388-w (PMC7893153)
- **핵심(원문 확인)**: "MBON-α′2 increases CS+ odor response during acquisition and decreases during reversal." 처벌 DAN의 활성이 줄면 콜린성 MBON 입력의 억압이 풀리고, 그 MBON이 보상 DAN을 흥분시킨다.
- **관련성**: MBON13(α′2)을 혐오 학습 쪽 판독 A로 쓰는 모델 선택과 맞는다(혐오 획득 시 CS+ 반응 증가).

### 12. Eschbach et al. 2020 (Nat Neurosci 23:544–555) / 2021 (eLife 10:e62567): 유충 커넥톰
- DOI 10.1038/s41593-020-0607-9 (PMC7145459), 10.7554/eLife.62567 (PMC8616581)
- **핵심(확인)**
  - 2020: 많은 조절 뉴런이 수상돌기 입력의 50% 이상을 MBON 피드백 경로에서 받는다.
  - 2021: 수렴 뉴런들이 양의 가치 MB·LH 경로에서 흥분 입력을, 음의 가치 MB 경로에서 억제 입력을 받는다.
- **관련성**: 가치가 반대인 MBON 경로가 하류에서 E/I 균형으로 합쳐진다는 구조 선례다. 다만 유충이라 성충 MBON05/13에 직접 대응하지는 않는다.

### 13. Otto et al. 2020, Curr Biol 30:3200: DAN 입력 이질성
- DOI 10.1016/j.cub.2020.05.077 (PMC7443709). 초록으로 확인했다.
- **핵심**: MBON-γ5β′2a가 일부 PAM-γ5 아형에 직접 되먹임을 준다. PPL1-γ1pedc의 입력은 γ5 DAN과 대부분 다르다.
- **관련성**: 간접적이다(MBON→DAN 경로). 이 시소 문제와는 약하게 관련된다.

### 14. Jacob & Waddell 2020, Neuron (권·쪽 미확인): 반대 가치 장기기억
- DOI 10.1016/j.neuron.2020.03.013 (PMC7302427). 초록으로 확인했다.
- **핵심**: 간격 훈련을 하면 CS+ 혐오 기억과 CS− "안전 기억"이 함께 생긴다. 두 출력의 조합이 상대적 안전을 신호한다.
- **관련성**: 보상과 처벌 기억이 서로 지우지 않고 **공존·조합**되는 예다. 우리 시소처럼 한쪽이 다른 쪽을 끄는 현상은 생물학적으로 기본값이 아니라는 근거다.

### 15. 글루탐산 부호 근거
- **Liu & Wilson 2013**, PNAS 110:10294, DOI 10.1073/pnas.1220560110: 글루탐산이 GluClα를 통해 더듬이엽 뉴런을 과분극시킨다(초록 확인).
- **Shiu et al. 2024**, Nature 634:210–219, DOI 10.1038/s41586-024-07763-9: 전뇌 LIF 모델이 GABA와 Glu를 억제성으로 둔다. 이 내용은 검색 요약으로만 봤고 원문 문장은 **미확인**이다.
- **관련성**: 모델의 Glu=−1은 관행과 같다. 다만 Li 2020이 말하듯 MBON 표적에서 수용체가 무엇인지는 대부분 측정되지 않았다. MBON05→MBON11/09/01과 MBON03→MBON13의 실제 부호는 미측정이다.

### 16. 기타
- **Shuai et al. 2015**, PNAS 112:E6663, DOI 10.1073/pnas.1512792112(초록 확인): MBON-γ4>γ1γ2는 혐오 기억의 망각을 매개하는 글루탐산 뉴런 한 쌍이다. 혐오 기억을 약화하는 쪽의 역할이다.
- **Aso & Rubin 2016**, eLife 5:e16135(초록 확인): 같은 DAN이라도 냄새에 대한 타이밍에 따라 혐오 기억을 쓰거나 줄이거나, 보상 기억을 쓴다.

---

## (c) 부분 제거 선언(부록 U)에 주는 함의

1. **비율 f에 대한 예상 모양(추정)**
   - 시소가 "억제 3단 사슬"에서 나온다면, MBON05 출력이 커지면서 강장 구동 중인 MBON09·MBON11·MBON01을 문턱 아래로 누르는 시점이 있다. 이 시점에서 MBON03·CRE055가 풀리고 MBON13이 꺼진다.
   - 그래서 MBON13의 Δ는 f에 대해 **문턱형**으로 떨어질 가능성이 높다. 반면 MBON05 증가는 f에 대해 더 매끄러울 것이다.
   - 따라서 f 격자는 등간격 ×0.25/0.5/0.75만으로는 부족할 수 있다. 꺾이는 구간을 더 촘촘히 보거나, 이분 탐색할 여지를 남기는 편이 안전하다.
   - 반대로 APL 경유 탈억제(MBON05 ⊣ APL ⊣ KCα′β′ → MBON13)도 함께 커진다. 그래서 작은 f에서는 MBON13이 약간 **오를** 수도 있다. 이 경우 단조성 가정이 깨진다.
   - Lin 2014가 말한 "부분 섭동에 대한 강건성"은 APL→KC 피드백 고리에 대한 이야기다. 그 고리 밖에 있는 APL→MBON05 간선에는 적용되지 않는다고 보는 편이 맞다.
2. **가드에 넣을 만한 것**
   - 기존 MBON13 판독 가드(중앙값 Δ ≥ 5, 0 비율 ≤ 0.25; h4_apl_ablation.py의 `mbon_type_stats(..., 5.0, 0.25)`)에 더해, 각 f에서 **중간 고리**의 반응을 진단으로만 기록하는 것을 고려할 수 있다. 대상은 MBON09, MBON11, MBON01, MBON03, CRE055다.
   - 그러면 MBON13이 무너졌을 때 그 원인이 사슬인지 바로 판정할 수 있다.
   - 메커니즘 대조군(선택): APL→MBON05를 전부 지운 상태에서 MBON03→MBON13(395)이나 CRE055→MBON13을 추가로 끈다. 그때 MBON13이 돌아오면 사슬 가설이 확인된다.
3. **예상되는 MBON13 반응**
   - 전부 제거했을 때(f=0) MBON13은 C0 기준 평균 반응 20.9→3.0, 0 비율 0.094→0.427이다. 이미 측정된 값이다.
   - MBON05가 ×1.5~2 정도로만 오르는 f 구간에서는 MBON13이 가드를 통과할 수 있다. 그 구간이 실제로 존재하는지가 부록 U의 핵심 질문이 된다. **이것은 예측일 뿐 근거는 없다.**
4. **해석상 주의**
   - 문헌상 MBON05는 활성화 시 **회피**를 일으키고 혐오 기억의 망각에 관여한다(Aso 2014b, Shuai 2015, Gkanias 2022). 모델이 MBON05를 보상 판독 P로 쓰는 논리가 문헌과 맞는지 선언문에서 한 줄로 근거를 대 두는 것이 좋다. 저장소의 근거 문서는 이 세션에서 확인하지 않았다.
   - 모델의 APL→MBON05 억제(1162 시냅스 × apl_scale 0.1)는 실제 커넥톰을 반영한 것이다. 그러나 Amin 2020에 따르면 실제 APL 억제는 구획마다 국소적이다. 점 뉴런 APL이 γ4와 α′2를 묶어 버리는 것도 시소에 기여할 수 있다. 다만 'readout' 진단에서 APL→MBON13 제거가 MBON13을 살리지 못했으므로 주된 원인은 아닐 것이다.

---

## (d) 확인하지 못한 것

- 시소 메커니즘의 **동역학적 확인**: §0 분석은 구조와 부호만 봤다. 실제 시뮬레이션에서 MBON09/11/01/03과 CRE055의 활성 변화는 재지 않았다(저장소 실행 금지).
- Li 2020 Figure 24의 MBON-MBON 화살표 수치, 그리고 hemibrain에서 MBON05→MBON11, MBON03→MBON13 시냅스 수. 그림이라 텍스트로 읽지 못했다. 위 수치는 모두 maleCNS 기준이다. FlyWire 수치도 조회하지 않았다.
- Perisse 2016의 V2α′가 hemibrain MBON13(α′2)과 같은 유형인지.
- MBON05 표적(MBON11/09/01)과 MBON03→MBON13 시냅스에서 글루탐산 수용체 종류(GluCl인지 흥분성 수용체인지).
- MBON-MBON 억제를 분수 단위로 줄인(×0.25/0.5/0.75) **계산 모델 선례**. 찾은 graded 선례는 실험(Lin 2014 GAD RNAi와 shibire 비교, Liu & Davis 2009 GAD RNAi)뿐이다. Abdelrahman 2021, Bennett 2021, Eschbach 2020 모델은 이 축을 다루지 않았다.
- Felsenberg 2017 (Nature, "Re-evaluation of learned information"), Hige 2015, Cohn 2015는 이번에 확인하지 않았다.
- Shiu 2024의 신경전달물질 부호 규칙 원문 문장.

## 재현

- 커넥톰 집계 스크립트와 출력은 이 세션 scratchpad에 있다: `.../scratchpad/syn.txt`, `.../scratchpad/paths.txt`.
- 논문 전문 텍스트는 `.../scratchpad/ft/*.txt`에 있다.
- 저장소는 읽기만 했다.
