# Silicon Sampling 저널 논문 수정 계획

**문서 버전**: v1.0
**작성일**: 2024-12-28
**대상 논문**: journal_paper.md
**리뷰어 판정**: Major Revision

---

## 수정 계획 개요

| Phase | 범위 | 소요 시간 | 필요 자원 |
|-------|------|----------|----------|
| **Phase 1** | 즉시 수정 (텍스트/해석) | 1-2일 | 없음 |
| **Phase 2** | 단기 추가분석 (기존 데이터) | 3-5일 | Python 스크립트 |
| **Phase 3** | 장기 추가분석 (신규 실험) | 1-2주 | API 비용, 실험 시간 |

---

## Phase 1: 즉시 수정 (텍스트/해석 수정)

### 1.1 데이터 불일치 해소 (MC1) - Critical

**문제**: Table 4.4와 4.5의 GPT-5.2 JS 값이 상이함
| Variable | Table 4.4 | Table 4.5 | 차이 |
|----------|-----------|-----------|------|
| PARTYLR | 0.467 | 0.106 | 4.4x |
| NORTHWHO | 0.259 | 0.377 | 1.5x |
| UNIFI | 0.287 | 0.047 | 6.1x |

**원인 분석**:
- Ablation D (GPT-4o-mini vs GPT-5.2): 2024년 12월 20일 실험, n=100
- Ablation E (GPT-5.2 vs CLOVA): 2024년 12월 27일 실험, n=100
- **7일 간격**으로 API 호출 → 모델 버전 드리프트 가능성

**수정 방안**:
```
Option A: 실험 조건 차이 명시 (즉시)
- "Ablation D와 E는 별도 시점에 수행되어 GPT-5.2 결과가 상이함"
- Limitations에서 API 버전 드리프트 언급

Option B: 동일 조건 재실험 (Phase 3)
- 모든 Ablation을 동일 시점에 재실행
```

**즉시 조치**: Option A 적용

---

### 1.2 변수 범위 통일 (M1/MC4)

**문제**:
- Methods: 7개 변수 (PRESTG5 포함)
- Ablation D: 4개 변수 (SATFIN, PARTYLR, NORTHWHO, UNIFI)
- Ablation E: 6개 변수 (PRESTG5 제외)

**수정 방안**:
1. Methods에서 PRESTG5 제거 → **6개 변수로 통일**
2. Ablation D의 4개 변수 선택 근거 명시:
   - "문화적 민감도가 높은 4개 핵심 변수 선정"
3. SATFIN이 Ablation E에 없는 이유 명시:
   - "SATFIN은 범용적 경제 인식 변수로 토착 LLM 비교에서 제외"

---

### 1.3 KS 해석 완화 (M3)

**현재 표현**:
> "successful distribution replication" (성공적 분포 재현)

**수정 표현**:
> "failed to detect statistically significant difference"
> (통계적으로 유의한 차이를 검출하지 못함)

**추가 문구**:
> "Note: Non-significance does not prove equivalence; it may reflect insufficient statistical power with n=100."

---

### 1.4 JS<0.05 기준 완화 (M4)

**현재 표현**:
> "JS < 0.05 for acceptable alignment"

**수정 표현**:
> "JS < 0.05 as an exploratory threshold for substantial similarity, following conventions in distribution comparison literature"

**또는**: Phase 2에서 Bootstrap CI 추가 후 기준 재정의

---

### 1.5 CoT 해석 범위 축소 (M5)

**현재 표현**:
> "CoT encourages more deliberate, realistic response patterns"

**수정 표현**:
> "CoT showed improvement on CONFINAN; generalization to other variables requires further study"

---

### 1.6 추론모델 비교 공정성 설명 (MC5)

**리뷰어 오해**: CLOVA thinking vs GPT 비교가 불공정하다

대응방안: clova에서 thinking 표현을 제거해라. 

---

### 1.7 Threats to Validity 섹션 추가

**추가 위치**: Section 5.4 Limitations 대체 또는 확장

```markdown
### 5.4 Threats to Validity

**Internal Validity**:
- API version drift between experiments (7-day gap between Ablation D and E)
- Single experimental run per condition (no repeated trials)

**External Validity**:
- Single benchmark (KGSS 2023)
- Six culturally-sensitive variables may not represent full Korean attitudes
- Results specific to tested model versions

**Construct Validity**:
- JS Divergence threshold (0.05) is exploratory
- KS test power limitations with n=100

**Statistical Conclusion Validity**:
- No multiple comparison correction applied
- Effect sizes not reported
```

---

## Phase 2: 단기 추가분석 (기존 데이터 활용)

### 2.1 Bootstrap 95% CI for JS Divergence (M4)

**목적**: JS Divergence의 불확실성 정량화

**방법**:
```python
# 기존 응답 데이터에서 Bootstrap resampling
# 1000회 반복으로 JS의 95% CI 계산
def bootstrap_js_ci(responses, benchmark, n_bootstrap=1000):
    js_values = []
    for _ in range(n_bootstrap):
        resampled = np.random.choice(responses, size=len(responses), replace=True)
        js = calculate_js(resampled, benchmark)
        js_values.append(js)
    return np.percentile(js_values, [2.5, 97.5])
```

**입력 데이터**: `results/clova_experiment/*/clova_results.json`의 responses 배열

**예상 산출물**: 각 변수별 JS [95% CI] 테이블

---

### 2.2 Effect Size 계산 (MC2)

**목적**: 통계적 유의성 외에 실질적 차이 크기 보고

**방법**: Cramér's V (범주형 데이터용 효과 크기)
```python
from scipy.stats import chi2_contingency
import numpy as np

def cramers_v(observed, expected):
    chi2, _, _, _ = chi2_contingency([observed, expected])
    n = observed.sum()
    min_dim = min(len(observed), 2) - 1
    return np.sqrt(chi2 / (n * min_dim))
```

**해석 기준**:
- V < 0.1: 무시할 수준
- 0.1 ≤ V < 0.3: 작은 효과
- 0.3 ≤ V < 0.5: 중간 효과
- V ≥ 0.5: 큰 효과

---

### 2.3 다중비교 보정 (M3)

**목적**: 6개 변수 × 2개 모델 비교 시 Type I error 통제

**방법**: Benjamini-Hochberg FDR correction
```python
from statsmodels.stats.multitest import multipletests

# 모든 p-value 수집 후 보정
p_values = [...]  # 12개 KS test p-values
rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
```

**예상 결과**: 보정 후에도 CLOVA CONFINAN이 유일하게 비유의인지 확인

---

### 2.4 Post-hoc Power Analysis (MC2)

**목적**: n=100에서의 검정력 확인

**방법**:
```python
from statsmodels.stats.power import GofChisquarePower

analysis = GofChisquarePower()
# 관찰된 효과 크기로 검정력 계산
power = analysis.power(effect_size=observed_w, nobs=100, alpha=0.05, n_bins=4)
```

**예상 산출물**: "n=100에서 medium effect (w=0.3) 탐지 검정력: XX%"

---

### 2.5 TOST 동등성 검정 (M3 대안)

**목적**: "차이 없음"을 적극적으로 입증

**방법**: Two One-Sided Tests for equivalence
```python
# 동등성 마진 설정 (예: JS < 0.1을 동등으로 간주)
# H0: |JS_diff| >= margin vs H1: |JS_diff| < margin
```

**적용**: CLOVA CONFINAN에 대해 TOST 수행

---

## Phase 3: 장기 추가분석 (신규 실험 필요)

### 3.1 반복 실험 (MC3) - 높은 우선순위

**목적**: 결과 안정성 검증, 표준편차/CI 제공

**방법**:
- 동일 조건에서 3회 반복 실험
- 각 조건: n=100 × 3회 = 300 API 호출

**대상 조건** (우선순위순):
1. CLOVA HCX-007 전체 6개 변수 × 3회
2. GPT-5.2 전체 6개 변수 × 3회

**예상 비용**:
- CLOVA: 약 1,800 API 호출
- GPT-5.2: 약 1,800 API 호출

**산출물**: 평균 ± SD, 95% CI for JS Divergence

---

### 3.2 CoT 효과 확장 실험 (M5)

**목적**: CoT 효과를 CONFINAN 외 변수로 일반화

**방법**:
- 6개 변수 전체에 대해 CoT vs Direct 비교
- 각 조건: n=100

**예상 비용**: 약 1,200 API 호출

---

### 3.3 동일 시점 재실험 (MC1 해소)

**목적**: Ablation D와 E의 GPT-5.2 결과 일관성 확보

**대응**:
실질적으로 실행 시점 간 거의 차이가 없다. 동일 시점 실행으로 수정하면 된다. 

---

## 수정 우선순위 요약

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 즉시 수정 (1-2일)                                   │
├─────────────────────────────────────────────────────────────┤
│ ✓ 1.1 데이터 불일치 설명 추가 (실험 시점 차이)               │
│ ✓ 1.2 변수 범위 6개로 통일                                   │
│ ✓ 1.3 KS 해석 완화                                          │
│ ✓ 1.4 JS 기준 완화                                          │
│ ✓ 1.5 CoT 해석 축소                                         │
│ ✓ 1.6 추론모델 비교 설명 추가                                │
│ ✓ 1.7 Threats to Validity 섹션 추가                         │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: 단기 추가분석 (3-5일)                               │
├─────────────────────────────────────────────────────────────┤
│ □ 2.1 Bootstrap 95% CI for JS                               │
│ □ 2.2 Effect Size (Cramér's V)                              │
│ □ 2.3 FDR 다중비교 보정                                     │
│ □ 2.4 Post-hoc Power Analysis                               │
│ □ 2.5 TOST 동등성 검정 (CONFINAN)                           │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 장기 추가분석 (1-2주)                               │
├─────────────────────────────────────────────────────────────┤
│ □ 3.1 반복 실험 3회 (CLOVA + GPT-5.2)                       │
│ □ 3.2 CoT 효과 6개 변수 확장                                │
                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 다음 단계

1. **Phase 1 수정 진행**: journal_paper.md에 즉시 반영
2. **Phase 2 분석 스크립트 작성**: `code/07_additional_analysis.py`
3. **Phase 3 실험 계획 확정**: 비용/시간 고려하여 우선순위 결정

