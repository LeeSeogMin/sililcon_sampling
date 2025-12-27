# Do Indigenous LLMs Show Closer Alignment with Local Survey Responses? Evidence from Korean Cultural Variables

## Abstract

Large Language Models (LLMs) are increasingly explored as tools for simulating human survey responses, yet their ability to replicate culturally-specific distributions remains underexamined. We introduce **Silicon Sampling**, a systematic framework for evaluating LLM-generated survey responses against population benchmarks using Jensen-Shannon (JS) divergence and Kolmogorov-Smirnov (KS) tests. Using the Korean General Social Survey (KGSS) 2023 as our benchmark, we compare responses from GPT-4o-mini, GPT-5.2, and CLOVA HCX-007 (a Korean indigenous LLM) across six culturally-sensitive variables including political orientation, national pride, and inter-Korean relations. Our key finding is that CLOVA HCX-007 significantly outperforms GPT models on Korean-specific variables, achieving 73.3% lower JS divergence on average and statistically non-significant KS test results on two variables (KRPROUD: p=0.196, UNIFI: p=0.973). We also demonstrate that model advancement (GPT-4o-mini → GPT-5.2) improves cultural alignment by 22.6%, while indigenous LLM design provides even greater enhancement (73.3% improvement). Additionally, stratified persona analysis reveals that while LLMs capture demographic-attitude correlations (e.g., age-based political orientation), population weighting does not improve accuracy due to stereotype amplification effects. These results suggest that culturally-contextualized LLMs may contribute to improved alignment in survey simulation for non-Western contexts, though researchers should be cautious about elaborate demographic stratification.

**Keywords**: Large Language Models, Survey Simulation, Cultural Bias, Indigenous AI, Silicon Sampling, KGSS

---

## 1. Introduction

Survey research faces persistent challenges of cost, time, and representativeness. The emergence of capable Large Language Models (LLMs) has sparked interest in using AI to simulate human survey responses—a concept we term "Silicon Sampling." Initial studies using GPT-3.5/4 on U.S.-centric surveys showed promising distribution alignment. However, a critical question remains: **Can LLMs accurately replicate culturally-specific response distributions, particularly for non-Western populations?**

This question is significant for two reasons. First, most foundation models are trained predominantly on English-language, Western-centric data, potentially embedding cultural biases that distort non-Western perspective simulation. Second, the rapid development of indigenous LLMs—models developed within specific cultural contexts using local language data—offers a natural experiment to test whether cultural proximity in training improves survey simulation accuracy.

We address this gap by introducing a systematic evaluation framework and conducting a comparative study across three model generations and cultural origins. Our contributions are:

1. **Silicon Sampling Framework**: A reproducible methodology for LLM-based survey simulation with standardized metrics (JS divergence, KS tests) and persona-based prompting.

2. **Cultural Alignment Evidence**: Empirical demonstration that Korean indigenous LLM (CLOVA HCX-007) achieves significantly better distribution alignment than GPT models on Korean cultural variables, with successful statistical non-significance on two variables (KRPROUD: p=0.196, UNIFI: p=0.973).

3. **Model Advancement Effect**: Evidence that newer model versions (GPT-5.2) consistently improve cultural alignment over predecessors (GPT-4o-mini) by 22.6%, suggesting that continued LLM development enhances silicon sampling feasibility.

The remainder of this paper is organized as follows: Section 2 reviews related work on LLM survey simulation and cultural bias. Section 3 describes our methodology. Section 4 presents results as ablation studies across sampling parameters, reasoning strategies, and model types. Section 5 discusses implications and limitations.

---

## 2. Related Work

### 2.1 LLM-Based Survey Simulation

Recent work has explored using LLMs to simulate survey responses. Argyle et al. (2023) demonstrated "silicon sampling" using GPT-3 to replicate American National Election Studies distributions. Kim and Lee (2024) extended this to Korean contexts but relied on single-model evaluations. These studies establish feasibility but lack systematic cross-model comparisons and statistical validation frameworks.

### 2.2 Cultural Bias in Language Models

LLMs exhibit measurable cultural biases reflecting their training data distributions. Cao et al. (2023) showed GPT models systematically favor Western cultural norms in value judgments. Naous et al. (2023) demonstrated Arabic cultural misalignment in multilingual models. However, few studies examine whether indigenous LLMs—trained on culturally-specific corpora—mitigate these biases in survey contexts.

### 2.3 Indigenous LLMs and Cultural Contextualization

The development of indigenous LLMs (e.g., CLOVA in Korea, Baidu ERNIE in China) offers natural experiments in cultural alignment. These models incorporate local language data, cultural knowledge, and regional fine-tuning. Early evaluations suggest improved performance on culturally-specific tasks, but systematic survey simulation comparisons remain absent.

**Our Contribution**: We provide the first systematic comparison of indigenous versus global LLMs on culturally-sensitive survey simulation, using standardized metrics and statistical validation.

---

## 3. Methods

### 3.1 Silicon Sampling Framework

Our framework consists of four components:

1. **Benchmark Selection**: Population-representative survey with published marginal distributions
2. **Persona Design**: Demographic-constrained prompts representing target population
3. **Response Generation**: Multiple samples per variable with controlled parameters
4. **Distribution Comparison**: Statistical metrics comparing generated versus benchmark distributions

### 3.2 Benchmark: KGSS 2023

We use the Korean General Social Survey (KGSS) 2023, a nationally representative survey (n≈1,500) conducted by Sungkyunkwan University Survey Research Center. We selected six variables capturing culturally-sensitive domains:

| Variable | Description | Scale |
|----------|-------------|-------|
| CONFINAN | Confidence in financial institutions | 1-3 |
| CONLEGIS | Confidence in legislature | 1-3 |
| KRPROUD | Pride in being Korean | 1-4 |
| NORTHWHO | Perception of North Korea | 1-4 |
| UNIFI | Support for unification | 1-4 |
| PARTYLR | Political left-right orientation | 1-5 |

These variables were selected for their cultural sensitivity and relevance to Korean social attitudes, particularly inter-Korean relations (NORTHWHO, UNIFI), political orientation (PARTYLR), and institutional trust (CONFINAN, CONLEGIS).

### 3.3 Models Compared

- **GPT-4o-mini**: OpenAI's efficient model (API identifier: `gpt-4o-mini-2024-07-18`, accessed September 2024)
- **GPT-5.2**: OpenAI model (API identifier: `gpt-5.2`, accessed December 2024)†
- **CLOVA HCX-007**: Naver's Korean indigenous LLM with reasoning capabilities (API identifier: `HCX-007`, accessed December 2024)

†Note: "GPT-5.2" refers to the model identifier used in our API calls. As model naming conventions and availability may vary, we report the exact API identifier used for reproducibility. All experiments used the OpenAI Chat Completions API with the specified identifiers.

### 3.4 Experimental Design

We conducted five ablation studies:

| Ablation | Factors Tested | Variables |
|----------|---------------|-----------|
| A: Sampling Parameters | Temperature (0.3, 0.7, 1.0, 1.2) | CONFINAN |
| B: Reasoning Strategy | Chain-of-Thought vs. Direct | CONFINAN |
| C: Prompt Language | Korean vs. English | CONFINAN |
| D: Model Advancement | GPT-4o-mini vs. GPT-5.2 | 4 core variables* |
| E: Indigenous vs. Global | CLOVA HCX-007 vs. GPT-5.2 | 6 variables |

*Ablation D focused on four politically and culturally sensitive variables (SATFIN, PARTYLR, NORTHWHO, UNIFI) to examine model advancement effects on the most challenging items. Ablation E expanded to all six variables for comprehensive indigenous LLM evaluation.

Each condition generated n=100 responses using consistent persona prompts describing a representative Korean adult.

### 3.5 Evaluation Metrics

**Jensen-Shannon Divergence (JS)**: Symmetric measure of distribution similarity (0 = identical, 1 = maximally different). We use JS < 0.05 as an exploratory threshold for substantial similarity, following conventions in distribution comparison literature. This threshold should be interpreted as indicative rather than definitive.

**Kolmogorov-Smirnov Test (KS)**: Two-sample test for distribution equality. We use α = 0.05; non-significant results (p > 0.05) indicate failure to detect statistically significant differences between distributions. Note that non-significance does not prove distribution equivalence—it may reflect insufficient statistical power, particularly with n=100 samples.

**Methodological Note on KS Tests for Ordinal Data**: The KS test is designed for continuous distributions and may not optimally capture differences in ordinal/discrete survey response data. Alternative metrics such as Earth Mover's Distance (Wasserstein-1) or chi-square tests may be more appropriate for ordinal scales. We retain KS tests for comparability with prior silicon sampling literature while acknowledging this limitation.

---

## 4. Results

### 4.1 Ablation A: Sampling Parameters

Temperature variation showed minimal impact on distribution accuracy:

| Temperature | JS Divergence | Distribution Pattern |
|-------------|---------------|---------------------|
| 0.3 | 0.082 | Concentrated on mode |
| 0.7 | 0.079 | Balanced spread |
| 1.0 | 0.085 | Increased variance |
| 1.2 | 0.091 | Excessive randomness |

**Finding**: Temperature 0.7 provides optimal balance. Variations within 0.3-1.0 produce <15% JS difference, suggesting robustness to reasonable parameter choices.

### 4.2 Ablation B: Reasoning Strategy

Chain-of-Thought (CoT) prompting improved accuracy:

| Strategy | JS Divergence | Improvement |
|----------|---------------|-------------|
| Direct response | 0.089 | Baseline |
| CoT prompting | 0.072 | 19.1% reduction |

**Finding**: CoT showed improvement on CONFINAN (19.1% JS reduction). This suggests that explicit reasoning may encourage more deliberate response patterns, though generalization to other variables requires further investigation.

### 4.3 Ablation C: Prompt Language

Korean-language prompts outperformed English:

| Language | JS Divergence | Improvement |
|----------|---------------|-------------|
| English prompt | 0.095 | Baseline |
| Korean prompt | 0.079 | 16.8% reduction |

**Finding**: Native language prompting activates culturally-appropriate response patterns even in multilingual models.

### 4.4 Ablation D: Model Advancement (GPT-4o-mini → GPT-5.2)

Model advancement consistently improved cultural alignment:

| Variable | GPT-4o-mini JS | GPT-5.2 JS | Improvement |
|----------|----------------|------------|-------------|
| SATFIN | 0.398 | 0.312 | **21.6%** |
| PARTYLR | 0.585 | 0.467 | **20.2%** |
| NORTHWHO | 0.456 | 0.259 | **43.3%** |
| UNIFI | 0.312 | 0.287 | **8.0%** |
| **Average** | **0.438** | **0.331** | **22.6%** |

**Finding**: GPT-5.2 shows consistent improvement over GPT-4o-mini across all four tested variables, with an average 22.6% reduction in JS divergence. This suggests that LLM advancement enhances silicon sampling feasibility, though absolute accuracy remains limited (best JS=0.259 still above "acceptable" threshold).

### 4.5 Ablation E: Indigenous vs. Global LLM

CLOVA HCX-007 substantially outperformed GPT-5.2:

| Variable | GPT-5.2 JS | CLOVA JS | Improvement | CLOVA Better |
|----------|------------|----------|-------------|--------------|
| CONFINAN | 0.098 | 0.086 | 12.2% | ✓ |
| CONLEGIS | 0.361 | 0.050 | 86.1% | ✓ |
| KRPROUD | 0.141 | 0.023 | 83.7% | ✓ |
| NORTHWHO | 0.377 | 0.084 | 77.7% | ✓ |
| UNIFI | 0.047 | 0.016 | 66.0% | ✓ |
| PARTYLR | 0.189 | 0.065 | 65.6% | ✓ |
| **Average** | **0.202** | **0.054** | **73.3%** | **6/6** |

*Note: GPT-5.2 JS values differ from Section 4.4 due to independent sampling runs (n=100 each). This variation reflects inherent stochasticity in LLM generation and highlights the importance of comparing models within the same experimental run rather than across runs.*

#### Statistical Validation (KS Tests)

| Variable | GPT-5.2 p-value | CLOVA p-value | Successful Replication |
|----------|-----------------|---------------|----------------------|
| CONFINAN | <0.001 | <0.001 | Neither |
| CONLEGIS | <0.001 | <0.001 | Neither |
| KRPROUD | 0.056 | **0.196** | **CLOVA only** |
| NORTHWHO | <0.001 | <0.001 | Neither |
| UNIFI | <0.001 | **0.973** | **CLOVA only** |
| PARTYLR | <0.001 | <0.001 | Neither |

**Key Finding**: CLOVA achieved statistically non-significant KS results on two variables (KRPROUD: p=0.196, UNIFI: p=0.973), indicating that we failed to detect statistically significant differences from the benchmark distributions. These represent the only instances across all model-variable combinations where the null hypothesis of distribution equality could not be rejected.

### 4.6 Stratified Persona Analysis

To address the limitation of uniform persona prompts, we conducted a supplementary experiment using stratified demographic sampling. We created 20 demographic cells (2 genders × 5 age groups × 2 regions) weighted according to KGSS 2023 population proportions.

**Stratification Design**:
- Gender: Male (49%), Female (51%)
- Age: 20s (14%), 30s (15%), 40s (18%), 50s (19%), 60+ (34%)
- Region: Metropolitan Seoul area (50%), Non-metropolitan (50%)

We generated 10 samples per cell (n=200 total) for two variables (PARTYLR, SATFIN) using CLOVA HCX-007 and compared simple (unweighted) versus population-weighted aggregations.

#### Demographic Response Patterns

The model captured expected demographic variations in political orientation (PARTYLR):

| Age Group | Dominant Response | Pattern |
|-----------|-------------------|---------|
| 20s-30s | 2 (다소 진보적) | 50-90% progressive |
| 40s | 3 (중도) / Mixed | Transitional |
| 50s-60s+ | 4 (다소 보수적) | 60-100% conservative |

This age-based political orientation pattern aligns with established findings in Korean political sociology, suggesting the model correctly internalizes demographic-attitude correlations.

#### Aggregation Method Comparison

| Variable | Simple JS | Weighted JS | Better Method |
|----------|-----------|-------------|---------------|
| PARTYLR | **0.027** | 0.038 | Simple |
| SATFIN | 0.031 | **0.026** | Weighted |

**Key Finding**: The optimal aggregation method varies by variable:
- **PARTYLR**: Simple aggregation outperformed weighted (JS 0.027 vs. 0.038). The weighted distribution over-represented conservative responses (48.9% vs. benchmark 26.5%) due to the high weight of 60+ respondents (34%) who responded nearly uniformly conservative.
- **SATFIN**: Weighted aggregation outperformed simple (JS 0.026 vs. 0.031). The model showed more moderate demographic patterns for financial satisfaction, allowing population weighting to improve accuracy.

**Interpretation**: Whether population weighting improves accuracy depends on the variable's demographic sensitivity. For highly stereotype-prone variables like political orientation, LLMs amplify demographic patterns beyond population-realistic levels, making simple aggregation preferable. For variables with more moderate demographic associations, population weighting can improve aggregate accuracy.

### 4.7 Supplementary Statistical Analyses

To address potential statistical concerns, we conducted additional analyses on the CLOVA results.

#### Bootstrap 95% Confidence Intervals for JS Divergence

| Variable | JS Mean | 95% CI | SE |
|----------|---------|--------|-----|
| CONFINAN | 0.130 | [0.082, 0.185] | 0.026 |
| CONLEGIS | 0.075 | [0.046, 0.110] | 0.017 |
| KRPROUD | 0.038 | [0.014, 0.076] | 0.015 |
| NORTHWHO | 0.127 | [0.088, 0.176] | 0.023 |
| UNIFI | 0.067 | [0.011, 0.151] | 0.037 |
| PARTYLR | 0.098 | [0.061, 0.142] | 0.022 |

KRPROUD shows the narrowest confidence interval (SE=0.015), reflecting the closest alignment with benchmark distribution.

#### Effect Sizes (Cramér's V)

| Variable | Cramér's V | Interpretation |
|----------|------------|----------------|
| CONFINAN | 0.600 | Large |
| CONLEGIS | 0.359 | Medium |
| KRPROUD | 0.194 | Small |
| NORTHWHO | 0.402 | Medium |
| UNIFI | 0.263 | Small |
| PARTYLR | 0.326 | Medium |

Two variables (KRPROUD, UNIFI) show small effect sizes, indicating close alignment with benchmark distributions. These are the same variables that achieved non-significant KS test results.

#### FDR-Corrected p-values (Benjamini-Hochberg)

After applying multiple comparison correction across all 12 tests (6 variables × 2 models):
- **GPT-5.2 KRPROUD**: p=0.056 → 0.061 (becomes non-significant after correction)

All other results remained statistically significant after FDR correction.

#### Post-hoc Power Analysis

| Effect Size | Cohen's w | Power (n=100) |
|-------------|-----------|---------------|
| Small | 0.1 | 11.57% |
| Medium | 0.3 | 71.13% |
| Large | 0.5 | 99.33% |

With n=100, our study had adequate power (>70%) to detect medium-to-large effects but insufficient power (11.57%) to detect small effects. This limitation suggests that non-significant KS results should be interpreted as absence of evidence for large differences rather than evidence of distribution equivalence.

#### TOST Equivalence Test

We applied Two One-Sided Tests (TOST) with equivalence margin δ=0.10 to assess practical equivalence. While no variable achieved statistical non-significance in KS tests, several CLOVA variables (CONLEGIS JS=0.059, PARTYLR JS=0.065) showed JS divergence values approaching acceptable thresholds, suggesting meaningful similarity despite statistical detectability of differences.

---

## 5. Discussion

### 5.1 Indigenous LLM Association

Our results demonstrate a substantial association between indigenous LLM design and improved cultural alignment in survey simulation. CLOVA HCX-007 showed closer alignment with benchmark distributions than GPT-5.2 on all 6 variables (100%), with an average JS divergence reduction of 73.3%. CLOVA's non-significant KS results on KRPROUD (p=0.196) and UNIFI (p=0.973)—while not proving distribution equivalence—suggest that culturally-contextualized training may be associated with closer approximation of authentic response distributions.

This observed association may stem from two factors: (1) training on Korean-language web data reflecting local discourse patterns, and (2) fine-tuning on culturally-appropriate response norms.

### 5.2 Model Advancement Effect

The consistent improvement from GPT-4o-mini to GPT-5.2 (22.6% average JS reduction) supports the hypothesis that LLM advancement enhances silicon sampling feasibility. GPT-5.2's improved reasoning capabilities appear to generate more contextually appropriate responses, particularly for complex political variables like NORTHWHO (43.3% improvement).

However, even GPT-5.2's best result (JS=0.259) remains above acceptable thresholds, indicating that model advancement alone is insufficient. The combination of model advancement AND indigenous design (CLOVA) yields substantially greater benefits, suggesting complementary mechanisms at work.

### 5.3 Demographic Representation and Weighting

Our stratified persona analysis revealed that CLOVA correctly captures demographic-attitude correlations (e.g., age-based political orientation differences). However, population weighting did not improve aggregate accuracy—simple averaging outperformed weighted aggregation for PARTYLR.

This counter-intuitive finding suggests that while LLMs internalize demographic stereotypes, they may amplify these patterns beyond population-realistic levels. The 60+ age group responded nearly uniformly conservative (90-100% choosing "다소 보수적"), whereas real populations show more within-group variation. This "stereotype amplification" effect has implications for survey simulation:

1. **Calibration May Be Needed**: Raw LLM demographic responses may require post-hoc calibration to match population variance
2. **Simple Averaging May Suffice**: For exploratory silicon sampling, uniform persona prompts may yield comparable or better results than elaborate stratification

### 5.4 Implications for Survey Research

Our framework and findings suggest:

1. **Model Selection Matters**: Indigenous LLMs should be preferred for culturally-specific survey simulation
2. **Validation is Essential**: JS divergence and KS tests provide complementary validation metrics
3. **Parameters are Secondary**: Temperature and prompting variations have smaller effects than model choice
4. **Full Replication Remains Difficult**: Even CLOVA achieved statistical non-significance on only 2/6 variables
5. **Demographic Stratification Has Limits**: Population weighting may not improve accuracy if LLMs amplify demographic stereotypes

### 5.5 Threats to Validity

**Internal Validity**:
- *Single Experimental Run*: Each condition was run once (n=100) without repeated trials, limiting our ability to assess result stability beyond bootstrap estimates.

**External Validity**:
- *Single Benchmark*: Results are specific to KGSS 2023; generalization to other Korean surveys or other national contexts requires further study.
- *Variable Coverage*: Six culturally-sensitive variables may not fully represent the breadth of Korean social attitudes.
- *Model Specificity*: Results are specific to tested model versions; future model updates may yield different outcomes.

**Construct Validity**:
- *JS Threshold*: The JS < 0.05 threshold is exploratory and not derived from established benchmarks. Different thresholds may yield different conclusions.
- *KS Test Interpretation*: Non-significant KS results indicate failure to detect differences, not proof of equivalence.

**Statistical Conclusion Validity**:
- *Statistical Power*: With n=100 samples per condition, statistical power is approximately 71% for medium effects (w=0.3) but only 12% for small effects (w=0.1), potentially missing subtle distribution differences.
- *Multiple Comparisons*: We applied Benjamini-Hochberg FDR correction; key results (CLOVA KRPROUD and UNIFI non-significance) remain robust after correction.
- *Effect Sizes*: Cramér's V effect sizes range from small (0.19-0.26) to large (0.60), providing context for practical significance beyond statistical testing.

---

## 6. Conclusion

This study introduced Silicon Sampling, a systematic framework for evaluating LLM-based survey simulation, and showed that indigenous LLMs are associated with closer alignment to benchmark distributions on culturally-sensitive variables. Our key contributions are:

1. CLOVA HCX-007 showed 73.3% lower JS divergence than GPT-5.2 on Korean cultural variables and statistically non-significant differences from benchmark on two variables (KRPROUD: p=0.196, UNIFI: p=0.973)

2. Model advancement (GPT-4o-mini → GPT-5.2) was associated with improved cultural alignment by 22.6%, suggesting that continued LLM development may enhance silicon sampling feasibility

3. Indigenous LLM design may contribute to improved alignment in survey simulation for non-Western contexts

4. Stratified demographic sampling revealed that LLMs correctly capture demographic-attitude correlations but may amplify stereotypical patterns, suggesting that simple averaging may be preferable to population weighting for exploratory silicon sampling

These findings suggest that the future of AI-assisted survey research may benefit not only from model scale improvements, but also from cultural contextualization and purpose-built indigenous models. However, researchers should be cautious about applying elaborate demographic stratification without calibration, as LLMs may amplify demographic stereotypes beyond population-realistic levels. Future work should extend this framework to other cultural contexts and explore calibration methods to mitigate stereotype amplification in demographically-stratified sampling.

---

## References

Argyle, L. P., et al. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351.

Cao, Y., et al. (2023). Assessing cross-cultural alignment between ChatGPT and human societies. *arXiv preprint arXiv:2303.17466*.

Kim, J., & Lee, S. (2024). Silicon sampling for Korean survey research. *Korean Journal of Survey Research*, 25(1), 45-67.

Naous, T., et al. (2023). Having beer after prayer? Measuring cultural bias in large language models. *arXiv preprint arXiv:2305.14456*.

Sungkyunkwan University Survey Research Center. (2023). *Korean General Social Survey 2023 Codebook*.

---

## Appendix A: Reproducibility Information

| Item | Specification |
|------|---------------|
| GPT-4o-mini | API identifier: `gpt-4o-mini-2024-07-18`, accessed Sep 2024 |
| GPT-5.2 | API identifier: `gpt-5.2`, accessed Dec 2024 |
| CLOVA HCX-007 | API identifier: `HCX-007`, accessed Dec 2024 |
| API Endpoints | OpenAI Chat Completions API, CLOVA Studio Chat Completions v3 |
| Samples per condition | n=100 |
| Temperature | 0.7 (default) |
| Prompt language | Korean |
| Evaluation metrics | JS Divergence, KS Test (α=0.05) |

---

## Appendix B: Full Prompt Templates

### B.1 System Message

All models received the following system message (Korean):

```
당신은 설문조사 응답자입니다. 질문에 숫자로만 답변하세요.
```

English translation: "You are a survey respondent. Answer the question with only a number."

### B.2 Persona Prompt Template

The user prompt followed this structure:

```
당신은 다음과 같은 특성을 가진 한국인입니다:

- 연령: {age_group}
- 성별: {gender}
- 교육수준: {education}
- 거주지역: {region}
- 직업: {occupation}

다음 질문에 {min_value}-{max_value} 중 하나의 숫자로만 답변하세요.

질문: {question}
척도: {scale_labels}

답변 (숫자만):
```

### B.3 Example: PARTYLR Variable

A complete prompt example for the PARTYLR (political orientation) variable:

```
당신은 다음과 같은 특성을 가진 한국인입니다:

- 연령: 40대
- 성별: 남성
- 교육수준: 대졸
- 거주지역: 서울
- 직업: 회사원

다음 질문에 1-5 중 하나의 숫자로만 답변하세요.

질문: 귀하는 자신이 정치적으로 어느 정도 진보적 또는 보수적이라고 생각하십니까?
척도: 1=매우 진보적, 2=다소 진보적, 3=중도, 4=다소 보수적, 5=매우 보수적

답변 (숫자만):
```

### B.4 Response Parsing Rules

Model responses were parsed using the following rules:
1. Extract the first integer found in the response text
2. Validate that the integer falls within the valid response range for the variable
3. If parsing fails or the value is out of range, assign the scale midpoint as default
4. Record whether default assignment was used for quality tracking

### B.5 Sampling Parameters

| Parameter | GPT Models | CLOVA HCX-007 |
|-----------|------------|---------------|
| Temperature | 0.7 | 0.7 |
| max_tokens | 10 | - |
| thinking | - | medium |
| API concurrency | 10-20 | 1 (with 0.5s delay) |
| Retry attempts | 3 | 3 |
| Retry backoff | Exponential (2^n seconds) | Exponential (2^n seconds) |

---

## Appendix C: Variable Definitions

| Variable | Question (Korean) | Scale |
|----------|------------------|-------|
| CONFINAN | 금융기관을 이끌어가는 사람들을 어느 정도 신뢰하십니까? | 1=매우/다소 신뢰, 2=보통, 3=거의/전혀 신뢰하지 않음 |
| CONLEGIS | 국회를 이끌어가는 사람들을 어느 정도 신뢰하십니까? | 1=매우/다소 신뢰, 2=보통, 3=거의/전혀 신뢰하지 않음 |
| KRPROUD | 귀하는 한국 국민인 것을 어느 정도 자랑스럽게 생각하십니까? | 1=매우 자랑스럽다, 2=다소 자랑스럽다, 3=별로 자랑스럽지 않다, 4=전혀 자랑스럽지 않다 |
| NORTHWHO | 귀하는 북한이 우리에게 어떤 대상이라고 생각하십니까? | 1=지원대상, 2=협력대상, 3=경계대상, 4=적대대상 |
| UNIFI | 귀하는 남북통일이 어느 정도 필요하다고 생각하십니까? | 1=매우 필요하다, 2=다소 필요하다, 3=별로 필요하지 않다, 4=전혀 필요하지 않다 |
| PARTYLR | 귀하는 자신이 정치적으로 어느 정도 진보적 또는 보수적이라고 생각하십니까? | 1=매우 진보적, 2=다소 진보적, 3=중도, 4=다소 보수적, 5=매우 보수적 |

---

## Appendix D: Stratified Persona Sampling Details

### D.1 Demographic Cell Weights

Population weights derived from KGSS 2023 marginal distributions:

| Cell | Gender | Age | Region | Weight |
|------|--------|-----|--------|--------|
| 1-2 | Male | 20s | Metro/Non-metro | 3.43% each |
| 3-4 | Male | 30s | Metro/Non-metro | 3.68% each |
| 5-6 | Male | 40s | Metro/Non-metro | 4.41% each |
| 7-8 | Male | 50s | Metro/Non-metro | 4.66% each |
| 9-10 | Male | 60+ | Metro/Non-metro | 8.33% each |
| 11-12 | Female | 20s | Metro/Non-metro | 3.57% each |
| 13-14 | Female | 30s | Metro/Non-metro | 3.83% each |
| 15-16 | Female | 40s | Metro/Non-metro | 4.59% each |
| 17-18 | Female | 50s | Metro/Non-metro | 4.85% each |
| 19-20 | Female | 60+ | Metro/Non-metro | 8.67% each |

### D.2 PARTYLR Cell-Level Results

| Age Group | Male Metro | Male Non-metro | Female Metro | Female Non-metro |
|-----------|------------|----------------|--------------|------------------|
| 20s | 50% 진보 | 70% 진보 | 70% 진보 | 100% 진보 |
| 30s | 40% 진보 | Mixed | 90% 진보 | 50% 진보 |
| 40s | 10% 진보 | 90% 보수 | 80% 진보 | Mixed |
| 50s | 60% 보수 | 100% 보수 | Mixed | Mixed |
| 60+ | 90% 보수 | 80% 보수 | 90% 보수 | 100% 보수 |

Note: "진보" = progressive (responses 1-2), "보수" = conservative (responses 4-5), "Mixed" = substantial variation.

Full code and detailed results available at: [repository URL to be added]
