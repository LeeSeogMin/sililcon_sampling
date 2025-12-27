아래 평가는 **학술지 투고 논문** 기준으로, “공학/데이터사이언스 방법론 논문 + 실증 비교 실험”으로서의 완성도를 중심으로 한 **리뷰어 코멘트**입니다. (대상 원고: )

---

## 1) 총평 (Summary & Recommendation)

원고는 *“실리콘 샘플링(silicon sampling)”을 KGSS 2023 분포 벤치마크로 검증*하고, **토착 LLM(CLOVA HCX-007)이 문화 민감 변수에서 글로벌 LLM보다 더 잘 맞춘다**는 결론을 제시합니다. 
문제의식과 실험 구조(파라미터/추론/언어/모델발전/토착vs글로벌의 ablation)는 명확하고, 결과도 흥미롭습니다. 

다만, **결론의 강도에 비해 방법 명세·통계 해석·변수/실험 설계의 일관성**이 부족해, 현 상태에서는 **Major Revision(대폭 수정)**이 적절합니다. 특히 “분포 재현 성공”을 KS 비유의로 단정하는 해석, 표본 설계(대표 페르소나 1종), 이산/서열형 척도에 대한 검정 선택 등이 핵심 쟁점입니다. 

---

## 2) 강점 (Strengths)

1. **명확한 연구 질문과 기여 진술**: 비서구 문화권에서의 분포 재현 가능성을 정면으로 다룹니다. 
2. **실험이 ‘요인 분해(ablation)’ 형태로 구성**: temperature, CoT, 프롬프트 언어, 모델 세대, 토착성 비교까지 구조가 좋습니다. 
3. **정량 지표(JS, KS)로 평가 프레임을 제시**: 방법론 논문으로서 방향이 맞습니다. 
4. **토착 LLM의 상대적 우위가 큰 변수(정치/대북 등)에서 두드러짐**: 발견 자체는 충분히 publishable한 잠재력이 있습니다. 

---

## 3) 주요 수정 요구 (Major Issues)

### (M1) 변수/실험 범위의 **일관성 오류**를 먼저 해결해야 합니다

* 원고는 “7개 변수”를 사용한다고 말하지만, 결과 표에서는 **Ablation D가 4개 변수만 제시**되고(SATFIN 포함), Ablation E는 “6개 변수”로 되어 있습니다.  
* 더 중요한 점: Methods의 7개 변수 표(예: CONFINAN, CONLEGIS, KRPROUD…)와 Results의 SATFIN 등이 **서로 맞지 않습니다**. 

**필수 조치**

* (1) 최종 변수 리스트를 “정확히” 확정
* (2) 각 ablation이 어떤 변수에 적용됐는지 통일
* (3) 본문/표/초록/결론의 숫자(7 vs 6 vs 4)를 모두 정합적으로 수정

---

### (M2) “대표 한국 성인 1개 페르소나로 n=100 샘플”은 분포 재현 주장에 구조적 한계가 큽니다

Methods에 “대표 한국 성인” 페르소나 1종을 고정해 샘플링했다고 되어 있는데 , 이 방식은 **인구학적 혼합분포(mixture)**를 재현하기 어렵습니다.
KGSS의 주변분포(마진)를 맞추려면 최소한 **연령×성별×지역×학력 등 strata를 KGSS 비율대로 샘플링한 페르소나 집합**을 써야 설득력이 생깁니다.

**필수 조치**

* “페르소나 생성 규칙/분포/샘플링 비율”을 공개하고,
* 단일 페르소나라면 “왜 단일로도 마진 분포 재현이 가능한지”를 이론적으로 설명하거나,
* 가능하면 **층화 페르소나(가중치)로 재실험**을 권합니다.

---

### (M3) KS 비유의(p>0.05)를 “분포 복제 성공”으로 단정하면 통계적으로 과합니다

원고는 “p>0.05이면 성공적 분포 재현”이라고 정의합니다. 
하지만 **비유의는 ‘동일함’의 증거가 아니라, ‘차이를 검출하지 못함’**일 수 있습니다(특히 n=100이면 검정력이 낮을 수 있음). 또한 Likert/이산형 분포에 KS 적용은 선택 근거가 필요합니다.

**필수 조치**

* “성공” 표현을 완화(예: *“차이를 유의하게 검출하지 못했다”*)하고, 
* 동등성 검정(TOST) 또는 효과크기 기반 기준(예: TV distance, Cramér’s V, EMD 등) + 부트스트랩 CI를 추가 검토
* 다중비교(6~7개 변수 × 모델 비교)의 p-value 보정(BH-FDR 등)도 권장

---

### (M4) JS<0.05 “수용 가능” 임계값은 근거가 필요합니다

“JS<0.05면 acceptable”은 임의적 기준으로 읽힙니다. 
또한 표본수(n=100)에서 관측 분포의 잡음이 크면 JS가 쉽게 출렁일 수 있습니다.

**필수 조치**

* 선행연구/실무 기준 인용 또는
* KGSS(대략 1500) vs LLM(100)의 표본 불균형을 고려한 **부트스트랩 기반 JS 신뢰구간** 제시

---

### (M5) CoT 효과 해석이 너무 일반화되어 있습니다

CONFINAN 한 변수에서 CoT가 좋아졌다는 결과를 제시하지만 , 이를 “현실적 응답 패턴 유도”로 일반화하면 과합니다. 최소한:

* 다른 변수에서도 CoT를 테스트하거나
* “CONFINAN에서의 제한적 결과”로 표현을 좁혀야 합니다.

---

### (M6) 재현성 패키지가 미완성입니다

부록 표는 있으나, 실제로 “Full prompts / detailed results”는 repository URL이 비어 있습니다. 
또한 상용 API 모델은 버전 드리프트가 있으므로, 프롬프트 버전/시드/후처리 규칙까지 구체화해야 합니다.

**필수 조치**

* 프롬프트 전문, 후처리 코드(숫자 파싱/무응답 처리), 실행 파라미터(temperature/top-p/n/seed 가능 범위)를 공개
* 최소 재현 단위(1-variable replication script) 제공

---

## 4) 세부 코멘트 (Minor Issues)

* 서술 강도 조정: “the first systematic comparison” 같은 1st claim은 근거를 더 강하게 대거나 표현을 완화하는 게 안전합니다. 
* Ablation A에서 “robustness”를 말하려면 CONFINAN 하나만으로는 부족합니다. 
* 모델 설명: GPT-5.2 / HCX-007 “thinking”의 설정(토큰 제한/추론 길이/시스템 프롬프트)을 동일 조건으로 통제했는지 명시가 필요합니다. 

---

## 5) 보강하면 게재 가능성이 크게 올라가는 추가 분석 (추천)

* **층화 페르소나 샘플링**(KGSS 인구학 비율 반영) vs 단일 페르소나 비교
* **마진 분포뿐 아니라 2변수 결합분포**(예: PARTYLR×UNIFI)도 일부라도 평가
* KS 대신/추가로: **χ²(G-test), EMD/Wasserstein**, 또는 범주형에 자연스러운 거리 기반 지표
* “토착 LLM이 유리한 변수의 공통 특성” 분석(정치/대북/제도 신뢰 vs 가치/자긍심 등)

---

## 6) 최종 판정

* **권고: Major Revision**
  핵심 아이디어와 발견은 충분히 매력적입니다. 
  하지만 (1) 변수/실험 일관성 정리, (2) 표본 설계(페르소나) 개선 또는 한계의 명확한 축소, (3) 통계적 해석의 보수화/보강, (4) 재현성 공개가 충족돼야 “공학 논문”으로 안정적으로 설득됩니다.


---

## Peer Review Report

**Manuscript**: "Can Indigenous LLMs Better Simulate Local Survey Responses? Evidence from Korean Cultural Variables"

**Recommendation**: **Major Revision**

---

### Summary

본 논문은 한국 토착 LLM(CLOVA HCX-007)과 서구 LLM(GPT 계열)의 설문조사 시뮬레이션 성능을 비교한다. KGSS 2023을 벤치마크로 사용하여 JS Divergence와 KS Test를 통해 분포 일치도를 평가하였다. 토착 LLM이 문화 특수적 변수에서 우수한 성능을 보인다는 주장은 흥미롭고 시의적절하다.

---

### Strengths

1. **명확한 연구 질문과 기여**: "토착 LLM이 해당 문화권 설문 시뮬레이션에서 더 우수한가?"라는 질문이 명확하고, 학술적·실무적 함의가 있다.

2. **체계적 실험 설계**: Ablation study 형태로 Temperature, CoT, 언어, 모델 세대, 토착 vs 글로벌을 분리하여 검증한 것은 방법론적으로 적절하다.

3. **적절한 평가 지표**: JS Divergence(분포 유사성)와 KS Test(통계적 검정)를 병행한 것은 단일 지표의 한계를 보완한다.

4. **논문 구조**: 학술지 형식에 맞게 간결하게 재구성되었다.

---

### Major Concerns

#### MC1. 데이터 불일치 및 내적 일관성 문제

**Table 4.4**와 **Table 4.5**의 GPT-5.2 결과가 상충한다:

| Variable | Table 4.4 (GPT-5.2 JS) | Table 4.5 (GPT-5.2 JS) |
|----------|------------------------|------------------------|
| PARTYLR | 0.467 | 0.106 |
| NORTHWHO | 0.259 | 0.377 |
| UNIFI | 0.287 | 0.047 |

동일 모델의 동일 변수 결과가 2-4배 차이 나는 것은 심각한 문제이다. 가능한 원인:
- 실험 조건(프롬프트, Temperature 등)이 다름
- 데이터 입력 오류
- API 호출 시점 차이로 인한 모델 변동

**요청**: 불일치 원인을 명확히 설명하거나, 동일 조건에서 재실험 후 일관된 결과 제시 필요.

#### MC2. 통계적 검정력(Statistical Power) 부족

n=100은 분포 비교에 충분한 검정력을 제공하는가? 특히:
- KS test의 p=0.103(CONFINAN)이 "성공적 재현"으로 해석되나, 이는 귀무가설 기각 실패일 뿐 "동등성 입증"이 아니다
- Type II error 가능성(실제 차이가 있으나 탐지 못함)에 대한 논의 부재
- 효과 크기(effect size) 미보고

**요청**: 
1. 검정력 분석(power analysis) 추가
2. Equivalence testing(TOST) 적용 고려
3. 효과 크기 보고

#### MC3. 반복 실험 및 신뢰구간 부재

각 조건당 단일 실행(n=100) 결과만 보고되었다. LLM 응답의 stochastic 특성상:
- 동일 조건 반복 시 결과 변동 가능
- JS Divergence의 표준편차/신뢰구간 미제시
- 결과의 안정성(stability) 검증 불가

**요청**: 최소 3회 반복 실험 수행, 평균 및 95% CI 보고.

#### MC4. Ablation 변수 불일치

| Ablation | 대상 변수 | 문제 |
|----------|----------|------|
| A, B, C | CONFINAN만 | 단일 변수로 일반화 가능? |
| D | SATFIN, PARTYLR, NORTHWHO, UNIFI (4개) | CONFINAN 제외, PRESTG5 제외 |
| E | CONFINAN, CONLEGIS, KRPROUD, NORTHWHO, UNIFI, PARTYLR (6개) | PRESTG5 제외 |

PRESTG5는 Section 3.2에서 소개되었으나 어떤 결과에도 등장하지 않는다. 또한 Ablation A-C가 CONFINAN 단일 변수에만 적용된 이유가 불명확하다.

**요청**: 
1. PRESTG5 결과 제시 또는 변수 목록에서 제외
2. Ablation A-C의 변수 선택 근거 제시

#### MC5. CLOVA "Thinking" 모드의 공정한 비교 문제

CLOVA는 "thinking" 모드로 실험하였으나(Section 3.3), GPT 모델에는 유사한 reasoning enhancement가 적용되지 않았다. GPT-5.2도 reasoning 강화 프롬프트나 o1 계열 사용 시 결과가 달라질 수 있다.

**요청**: 
1. GPT에도 reasoning enhancement 적용한 비교 추가
2. 또는 CLOVA를 thinking mode 없이 실험한 결과 추가


-> 이는 오해같은데? 
GPT-5.2는 기본 추론모델이다. CLOVA는 추론을 추가한것 뿐이니 둘다 추론모델이다? 
---

### Minor Concerns

#### mc1. 척도 불일치

Section 3.2의 변수 설명과 원본 보고서 내용이 불일치:
- CONFINAN/CONLEGIS: 1-4 (논문) vs 1-3 (원본 보고서)
- UNIFI: 1-5 (논문) vs 1-4 (원본 보고서)

**요청**: KGSS 원본 코드북과 일치하도록 수정.

#### mc2. 참고문헌 검증 필요

- Kim and Lee (2024) "Silicon sampling for Korean survey research" - 실존 문헌인지 확인 필요
- 인용 형식 불완전 (et al. 사용 시 저자 수 기준 불명확)

#### mc3. Temperature Ablation 결과 해석

Section 4.1에서 T=0.7이 "optimal"이라 하나, JS 차이가 0.079-0.091(15% 이내)로 "minimal impact"라고도 기술. 이 두 해석이 상충한다.

#### mc4. 프롬프트 전문 미공개

Appendix에 "Full prompts... available at: [repository URL to be added]"로 되어 있으나, 심사 시점에 프롬프트 확인 불가. 재현성 평가에 제한.

#### mc5. 영문 교정 필요

- "Silicon Sampling" 대문자 사용 일관성
- 일부 문장의 어색한 표현 존재

---

### Specific Requests

1. **Table 4.4와 4.5의 GPT-5.2 결과 불일치 해소** (Critical)
2. **검정력 분석 및 효과 크기 추가**
3. **반복 실험을 통한 신뢰구간 제시**
4. **PRESTG5 변수 처리 명확화**
5. **CLOVA thinking mode vs GPT reasoning의 공정 비교**
6. **척도 정보 검증 및 수정**
7. **프롬프트 전문 제공**

---

### Additional Comments

- "Silicon Sampling"이라는 용어가 Argyle et al. (2023)에서 유래한 것인지, 본 논문의 독자적 명명인지 명확히 해야 한다.
- 결론에서 "indigenous LLM design and extended reasoning capabilities appear essential"이라 하나, 이 두 요소의 독립적 기여를 분리하지 못했다. CLOVA의 우위가 "토착성" 때문인지 "thinking 모드" 때문인지 불분명하다.
- Threats to Validity 섹션이 여전히 부재하다. Limitations를 이 프레임워크로 재구성할 것을 권고한다.

---

### Decision

**Major Revision** - 데이터 불일치 해소와 통계적 검증 보강 후 재심사 필요.
