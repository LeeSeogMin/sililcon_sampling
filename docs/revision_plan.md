# Journal Paper Revision Plan
## 목표: academic_paper.md 5개 가설 구조를 journal_paper.md에 반영

---

## 현재 상태 비교

### academic_paper.md (한국어 - 참조용)
**5개 가설 구조** (모델 발전 제외 후):
1. 인구학적 페르소나 기반 시뮬레이션 (Argyle et al., 2023)
2. Temperature 파라미터 최적화
3. Chain-of-Thought(CoT) 추론 (Dillion et al., 2023)
4. 프롬프트 엔지니어링 (Ornstein et al., 2024)
5. 문화적 맥락과 토착 LLM의 우위

**삭제 대상**:
- ~~주장 5: 모델 발전에 따른 성능 향상 (GPT-4o-mini vs GPT-5.2)~~

### journal_paper.md (영어 - 수정 대상)
**현재 4개 Ablation Study 구조**:
- Ablation A: Sampling Parameters (Temperature)
- Ablation B: Reasoning Strategy (CoT)
- Ablation C: Prompt Language (Korean vs English)
- Ablation D: Indigenous vs. Global LLM (CLOVA vs GPT-5.2)

---

## 수정 계획

### 1. Abstract 수정
**현재**: CLOVA vs GPT-5.2 비교 중심, 4개 ablation 언급
**수정**: 5개 가설 검증 프레임워크 반영
- "four ablation studies" → "five experiments"
- 각 가설 검증 결과 요약 추가

### 2. Section 1 (Introduction) 수정
**수정 사항**:
- 5개 가설 명시적 나열
- 연구 목적을 가설 검증으로 재구성

### 3. Section 3 (Methods) 수정
**Section 3.4 Experimental Design 재구성**:

| 기존 | 수정 후 |
|------|---------|
| Ablation A: Sampling Parameters | Experiment 1: Baseline Simulation |
| Ablation B: Reasoning Strategy | Experiment 2: Temperature Optimization |
| Ablation C: Prompt Language | Experiment 3: Chain-of-Thought |
| Ablation D: Indigenous vs Global | Experiment 4: Prompt Engineering |
| (없음) | Experiment 5: Cultural Context (Indigenous LLM) |

### 4. Section 4 (Results) 재구성
**수정 사항**:
- 4.1-4.4 → 5개 실험 결과로 재구성
- 각 실험이 해당 가설을 검증하는 구조로 정리
- 4.5 Stratified Persona Analysis 유지

### 5. Section 5 (Discussion) 수정
**수정 사항**:
- 5개 가설 검증 결과 요약
- 모델 발전 관련 내용 제거 (있다면)

### 6. Section 6 (Conclusion) 수정
**수정 사항**:
- 5개 가설 검증 결론
- 학술적 기여 재정리

---

## 수정 순서

1. [x] Abstract 수정 ✅
2. [x] Section 1 Introduction 수정 ✅
3. [x] Section 3.4 Experimental Design 재구성 ✅
4. [x] Section 4 Results 재구성 ✅
5. [x] Section 5 Discussion 수정 ✅
6. [x] Section 6 Conclusion 수정 ✅
7. [x] References 확인 ✅ (Dillion et al. 저자명 수정 완료)
8. [x] Appendix 확인 ✅

---

## 논문 심사 완료 (3회 반복)

### 1차 심사 결과 및 수정
- Abstract 수치 불일치 수정 (59.8% → 26.5%, p=0.103 → all p<0.05)
- 저자명 오타 발견 (Dillion vs Dillon - 2차에서 재검토)

### 2차 심사 결과 및 수정
- 저자명 확인: "Dillion" (L 2개)이 정확함 (References 포함 수정)
- 구어체 표현 수정: "essentially a statistical tie" → "indicating an inconclusive advantage pattern"
- "This counter-intuitive finding" → "This unexpected result"
- "catastrophic failure" → "response concentration"
- "Notably" 제거

### 3차 심사 결과 및 수정
- Abstract-Results 논리적 일관성 수정
  - CoT: "Inconsistent results" → "19.1% improvement, limited generalizability"
  - Conclusion: 과장된 표현 완화 ("essential" → "key consideration")
- 참고문헌 검증 완료:
  - Argyle et al. (2023): DOI 10.1017/pan.2023.2 ✅
  - Dillion et al. (2023): DOI 10.1016/j.tics.2023.04.008 ✅
  - Ornstein et al. (2024): DOI 10.1017/psrm.2024.64 ✅
- 수치 검증 완료: Baseline JS=0.397, CLOVA 26.5% improvement ✅

---

## 핵심 변경 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 실험 수 | 4개 ablation | 5개 experiments |
| 구조 | Ablation study | Hypothesis testing |
| 모델 발전 | 언급 없음 | 유지 (없었음) |
| 가설 체계 | 암묵적 | 명시적 5개 가설 |

---
## journal_paper.md 수정 

당신은 ai 연구자로서 journal_paper.md 논문을 아래의 기준으로 심사한다. 

1. 논문은 이공계 저널 논문의 형식을 따르고 있는가? 
2. 논문은 이공계 저널 논문의 전형적인 문장과 문체를 따르고 있는가? ai가 작성한 흔적이 있는가? 
3. 논문은 내용의 타당성과 논리성이 있는가?
4. 논문은 내용에서 논리적인 모순, 수치 불일치가 있는가? 
5. 논문에서 사용한 참고문헌이 사실인가? 적절하게 인용되었는가? 

위의 결과를 반영하여 스스로 수정한다.

이러한 수정절차를 세번 반복한다. 

---

