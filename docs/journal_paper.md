# Do Indigenous LLMs Show Closer Alignment with Local Survey Responses? Evidence from Korean Cultural Variables

## Abstract

Large Language Models (LLMs) are increasingly explored as tools for simulating human survey responses, yet their ability to replicate culturally-specific distributions remains underexamined. We present a **Silicon Sampling framework**—a systematic methodology for evaluating LLM-generated survey responses against population benchmarks using Jensen-Shannon (JS) divergence and Kolmogorov-Smirnov (KS) tests. Using the Korean General Social Survey (KGSS) 2023 as our benchmark, we compare responses from GPT-5.2 and CLOVA HCX-007 (a Korean indigenous LLM) across six culturally-sensitive variables including political orientation, national pride, institutional trust, and inter-Korean relations. Results show **no clear evidence for indigenous LLM superiority**: while CLOVA achieves 26.5% lower average JS divergence than GPT-5.2, this advantage is driven primarily by GPT-5.2's poor performance on unification attitudes (UNIFI). CLOVA outperforms GPT-5.2 on only 3/6 variables, while GPT-5.2 substantially outperforms CLOVA on political orientation (PARTYLR). KS tests reject benchmark equality for all variables from both models at α=0.05, highlighting that neither model achieves distributional equivalence. Additionally, stratified persona analysis reveals that while LLMs capture demographic-attitude correlations (e.g., age-based political orientation), the effectiveness of population weighting varies by variable: it may worsen accuracy for stereotype-prone variables (like political orientation) due to amplification effects, while improving accuracy for variables with more moderate demographic associations. These results suggest that culturally-contextualized LLMs may contribute to improved alignment in survey simulation for non-Western contexts, though researchers should be cautious about elaborate demographic stratification.

**Keywords**: Large Language Models, Survey Simulation, Cultural Bias, Indigenous AI, Silicon Sampling, KGSS

---

## 1. Introduction

Survey research faces persistent challenges of cost, time, and representativeness. The emergence of capable Large Language Models (LLMs) has sparked interest in using AI to simulate human survey responses—a concept termed "Silicon Sampling" (Argyle et al., 2023). Initial studies using GPT-3.5/4 on U.S.-centric surveys showed promising distribution alignment. However, a critical question remains: **Can LLMs accurately replicate culturally-specific response distributions, particularly for non-Western populations?**

This question matters because most foundation models are trained predominantly on English-language, Western-centric data, potentially embedding cultural biases that distort non-Western perspective simulation. Meanwhile, the rapid development of indigenous LLMs—models developed within specific cultural contexts using local language data—offers a natural experiment to test whether cultural proximity in training improves survey simulation accuracy.

We address this gap by presenting a systematic Silicon Sampling framework—a reproducible methodology for LLM-based survey simulation with standardized metrics (JS divergence, KS tests) and persona-based prompting. Through comparative study across models of different cultural origins, we find **mixed evidence** regarding whether Korean indigenous LLM (CLOVA HCX-007) achieves better distribution alignment than GPT-5.2 on Korean cultural variables—performance varies substantially by variable type.

---

## 2. Related Work

### 2.1 LLM-Based Survey Simulation

Recent work has explored using LLMs to simulate survey responses. Argyle et al. (2023) demonstrated "silicon sampling" using GPT-3 to replicate American National Election Studies distributions. These studies establish feasibility but lack systematic cross-model comparisons and statistical validation frameworks, particularly for non-Western contexts.

### 2.2 Cultural Bias in Language Models

LLMs exhibit measurable cultural biases reflecting their training data distributions. Cao et al. (2023) showed GPT models systematically favor Western cultural norms in value judgments. Naous et al. (2024) demonstrated Arabic cultural misalignment in multilingual models. However, few studies examine whether indigenous LLMs—trained on culturally-specific corpora—mitigate these biases in survey contexts.

### 2.3 Indigenous LLMs and Cultural Contextualization

The development of indigenous LLMs (e.g., CLOVA in Korea, Baidu ERNIE in China) offers natural experiments in cultural alignment. These models incorporate local language data, cultural knowledge, and regional fine-tuning. Early evaluations suggest improved performance on culturally-specific tasks, but systematic survey simulation comparisons remain absent.

This study provides a systematic comparison of indigenous versus global LLMs on culturally-sensitive survey simulation, using standardized metrics and statistical validation.

---

## 3. Methods

### 3.1 Silicon Sampling Framework

Our framework consists of four components:

1. **Benchmark Selection**: Population-representative survey with published marginal distributions
2. **Persona Design**: Demographic-constrained prompts representing target population
3. **Response Generation**: Multiple samples per variable with controlled parameters
4. **Distribution Comparison**: Statistical metrics comparing generated versus benchmark distributions

### 3.2 Benchmark: KGSS 2023

We use the Korean General Social Survey (KGSS) 2023, a nationally representative survey (n≈1,500) conducted by Sungkyunkwan University Survey Research Center. We selected seven variables capturing culturally-sensitive domains:

| Variable | Description | Scale |
|----------|-------------|-------|
| CONFINAN | Confidence in financial institutions | 1-3 |
| CONLEGIS | Confidence in legislature | 1-3 |
| KRPROUD | Pride in being Korean | 1-4 |
| NORTHWHO | Perception of North Korea | 1-4 |
| UNIFI | Support for unification | 1-4 |
| PARTYLR | Political left-right orientation | 1-5 |
| SATFIN | Satisfaction with household finances | 1-5 |

These variables were selected for their cultural sensitivity and relevance to Korean social attitudes, particularly inter-Korean relations (NORTHWHO, UNIFI), political orientation (PARTYLR), and institutional trust (CONFINAN, CONLEGIS).

### 3.3 Models Compared

- **GPT-5.2**: OpenAI model (API identifier: `gpt-5.2`, accessed December 2025)†
- **CLOVA HCX-007**: Naver's Korean indigenous LLM with reasoning capabilities (API identifier: `HCX-007`, accessed December 2025)

†Note: "GPT-5.2" refers to the model identifier used in our API calls. As model naming conventions and availability may vary, we report the exact API identifier used for reproducibility. All experiments used the OpenAI Chat Completions API with the specified identifiers.

### 3.4 Experimental Design

We conducted four ablation studies:

| Ablation | Factors Tested | Variables |
|----------|---------------|-----------|
| A: Sampling Parameters | Temperature (0.3, 0.7, 1.0, 1.2) | CONFINAN |
| B: Reasoning Strategy | Chain-of-Thought vs. Direct | CONFINAN |
| C: Prompt Language | Korean vs. English | CONFINAN |
| D: Indigenous vs. Global | CLOVA HCX-007 vs. GPT-5.2 | 6 core variables |

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

Temperature 0.7 provides optimal balance between response diversity and consistency. Variations within 0.3-1.0 produce less than 15% JS difference, suggesting robustness to reasonable parameter choices.

### 4.2 Ablation B: Reasoning Strategy

Chain-of-Thought (CoT) prompting improved accuracy:

| Strategy | JS Divergence | Improvement |
|----------|---------------|-------------|
| Direct response | 0.089 | Baseline |
| CoT prompting | 0.072 | 19.1% reduction |

CoT prompting reduced JS divergence by 19.1% on CONFINAN. This indicates that explicit reasoning may encourage more deliberate response patterns, though generalization to other variables requires further investigation.

### 4.3 Ablation C: Prompt Language

Korean-language prompts outperformed English:

| Language | JS Divergence | Improvement |
|----------|---------------|-------------|
| English prompt | 0.095 | Baseline |
| Korean prompt | 0.079 | 16.8% reduction |

Native language prompting activates culturally-appropriate response patterns even in multilingual models.

### 4.4 Ablation D: Indigenous vs. Global LLM

CLOVA HCX-007 showed mixed performance relative to GPT-5.2:

| Variable | GPT-5.2 JS | CLOVA JS | Improvement | CLOVA Better |
|----------|------------|----------|-------------|--------------|
| CONFINAN | 0.062 | 0.062 | -0.2% | ✗ |
| CONLEGIS | 0.134 | 0.083 | 38.5% | ✓ |
| KRPROUD | 0.113 | 0.134 | -18.9% | ✗ |
| NORTHWHO | 0.125 | 0.084 | 32.6% | ✓ |
| UNIFI | 0.267 | 0.115 | 56.9% | ✓ |
| PARTYLR | 0.038 | 0.065 | -70.7% | ✗ |
| **Average** | **0.123** | **0.090** | **26.5%** | **3/6** |

*Note: GPT-5.2 values are computed from the GPT rerun saved in `results/gpt52_experiment/metrics.json`, while CLOVA values use the saved CLOVA response sets in `results/clova_experiment/*/clova_results.json`.*

**Interpretation**: The results show no clear indigenous LLM advantage. CLOVA outperforms GPT-5.2 on only 3/6 variables—essentially a tie. The 26.5% average improvement is heavily influenced by GPT-5.2's catastrophic failure on UNIFI (JS=0.267, with 100% responses concentrated on a single option). Excluding UNIFI, the average difference shrinks to approximately 9%. Notably, GPT-5.2 substantially outperforms CLOVA on PARTYLR (political orientation), a key cultural variable, suggesting that indigenous training does not guarantee superiority across all culturally-sensitive domains.

#### Statistical Validation (KS Tests)

KS tests reject benchmark equality for all CLOVA variables at α=0.05:

| Variable | CLOVA p-value |
|----------|---------------|
| CONFINAN | 0.010 |
| CONLEGIS | 5.43e-09 |
| KRPROUD | 6.75e-20 |
| NORTHWHO | 2.45e-05 |
| UNIFI | 4.41e-10 |
| PARTYLR | 1.55e-05 |

### 4.5 Stratified Persona Analysis

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

The optimal aggregation method varies by variable:
- **PARTYLR**: Simple aggregation outperformed weighted (JS 0.027 vs. 0.038). The weighted distribution over-represented conservative responses (48.9% vs. benchmark 26.5%) due to the high weight of 60+ respondents (34%) who responded nearly uniformly conservative. This variable exhibits strong stereotype amplification: LLM-generated responses show exaggerated age-based political polarization.
- **SATFIN**: Weighted aggregation outperformed simple (JS 0.026 vs. 0.031). Unlike political orientation, financial satisfaction shows weaker demographic stereotyping in LLM responses—older age groups did not respond uniformly, preserving within-group variation closer to population reality. This allows population weighting to improve accuracy rather than amplify distortions.

Whether population weighting improves accuracy depends on the variable's demographic sensitivity. For highly stereotype-prone variables like political orientation, LLMs amplify demographic patterns beyond population-realistic levels, making simple aggregation preferable. For variables with more moderate demographic associations, population weighting can improve aggregate accuracy.

### 4.6 Supplementary Statistical Analyses

To address potential statistical concerns, we conducted additional analyses on the CLOVA results.

#### Bootstrap 95% Confidence Intervals for JS Divergence

<!-- AUTO:BOOTSTRAP_JS_TABLE_START -->
| Variable | JS Point | 95% CI | Bootstrap SD |
|----------|---------|--------|-----|
| CONFINAN | 0.062 | [0.062, 0.069] | 0.002 |
| CONLEGIS | 0.083 | [0.052, 0.120] | 0.017 |
| KRPROUD | 0.134 | [0.094, 0.189] | 0.024 |
| NORTHWHO | 0.084 | [0.061, 0.123] | 0.016 |
| UNIFI | 0.115 | [0.081, 0.167] | 0.022 |
| PARTYLR | 0.065 | [0.041, 0.103] | 0.016 |
<!-- AUTO:BOOTSTRAP_JS_TABLE_END -->

Bootstrap uncertainty is modest for CONFINAN (narrow CI) and larger for variables with more response mass spread across categories (e.g., KRPROUD), reflecting greater sampling variability with n=100.

---

## 5. Discussion

### 5.1 Indigenous LLM Association

Our results demonstrate **no clear evidence for indigenous LLM superiority** in distributional alignment. While CLOVA HCX-007 achieved 26.5% lower average JS divergence than GPT-5.2, this advantage is heavily driven by GPT-5.2's catastrophic failure on unification attitudes (UNIFI), where 100% of responses concentrated on a single option. CLOVA outperformed GPT-5.2 on only 3/6 variables—essentially a statistical tie. Critically, GPT-5.2 substantially outperformed CLOVA on political orientation (PARTYLR) by 71%, suggesting that indigenous training does not guarantee superiority across all culturally-sensitive domains. KS tests reject benchmark equality for all variables from both models at α=0.05, indicating that neither achieves distributional equivalence.

The variable-dependent performance pattern suggests that the relationship between model origin and cultural alignment is more complex than a simple indigenous advantage hypothesis. Possible explanations include: (1) variable-specific training data availability—GPT may have better coverage of Korean political discourse than inter-Korean relations topics; (2) differential sensitivity to prompt framing across variable domains; and (3) inherent model biases that manifest differently across attitudinal dimensions.

### 5.2 Demographic Representation and Weighting

Our stratified persona analysis revealed that CLOVA correctly captures demographic-attitude correlations (e.g., age-based political orientation differences). However, population weighting did not improve aggregate accuracy—simple averaging outperformed weighted aggregation for PARTYLR.

This counter-intuitive finding suggests that while LLMs internalize demographic stereotypes, they may amplify these patterns beyond population-realistic levels. The 60+ age group responded nearly uniformly conservative (90-100% choosing "다소 보수적"), whereas real populations show more within-group variation. This "stereotype amplification" effect has implications for survey simulation:

First, raw LLM demographic responses may require post-hoc calibration to match population variance—potential approaches include response redistribution based on known within-group variance from pilot surveys, or temperature adjustment per demographic cell to increase response diversity. Second, for exploratory silicon sampling, uniform persona prompts may yield comparable or better results than elaborate stratification. Third, researchers could apply "stereotype dampening" by blending demographic-specific responses with uniform-persona responses to reduce over-polarization.

### 5.3 Implications for Survey Research

These results have practical implications for survey research. First, **model selection should be variable-specific** rather than assuming indigenous LLMs are universally superior—researchers should pilot-test multiple models on their target variables before committing to a single model. Second, JS divergence and KS tests provide complementary validation metrics, while temperature and prompting variations have smaller effects than model choice. Third, full replication remains difficult: even when JS divergence is reduced, distributional differences remain detectable by KS tests across all tested variables. Fourth, population weighting may not improve accuracy if LLMs amplify demographic stereotypes.

### 5.4 Threats to Validity

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
- *Multiple Comparisons*: Multiple testing can inflate false positives; results should be interpreted with correction and replication in mind.

---

## 6. Conclusion

This study presented a systematic Silicon Sampling framework for evaluating LLM-based survey simulation, and found **no clear evidence for indigenous LLM superiority** on culturally-sensitive variables.

Comparing CLOVA HCX-007 and GPT-5.2 on six Korean cultural variables, we found variable-dependent performance: CLOVA outperformed GPT-5.2 on only 3/6 variables, while GPT-5.2 substantially outperformed CLOVA on political orientation. The 26.5% average JS improvement for CLOVA was heavily driven by GPT-5.2's catastrophic failure on a single variable (UNIFI). KS tests rejected benchmark equality for all variables from both models, indicating that neither achieves distributional equivalence. These findings suggest that the relationship between model origin and cultural alignment is more complex than a simple indigenous advantage hypothesis—**model selection should be variable-specific** rather than assuming indigenous LLMs are universally superior.

Stratified demographic sampling revealed that LLMs correctly capture demographic-attitude correlations but may amplify stereotypical patterns, suggesting that simple averaging may be preferable to population weighting for exploratory silicon sampling.

Future work should extend this framework to other cultural contexts, investigate why performance varies dramatically across variable types, and explore calibration methods to mitigate both model-specific failures and stereotype amplification in demographically-stratified sampling.

---

## References

Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2023). Using large language models to simulate multiple humans and replicate human subject studies. *Proceedings of the 40th International Conference on Machine Learning*, 337-371.

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351. DOI: https://doi.org/10.1017/pan.2023.2

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, 610-623. DOI: https://doi.org/10.1145/3442188.3445922

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

Cao, Y., Zhou, L., Lee, S., Cabello, L., Chen, M., & Hershcovich, D. (2023). Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. *Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP)*. DOI: https://doi.org/10.18653/v1/2023.c3nlp-1.7

Dillon, D., Tandon, N., Gu, Y., & Gray, K. (2023). Can AI language models replace human participants? *Trends in Cognitive Sciences*, 27(7), 597-600. DOI: https://doi.org/10.1016/j.tics.2023.04.008

Durmus, E., Nguyen, K., Liao, T. I., Schiefer, N., Askell, A., Bakhtin, A., ... & Ganguli, D. (2024). Towards measuring the representation of subjective global opinions in language models. *arXiv preprint arXiv:2306.16388*.

Hartmann, J., Schwenzow, J., & Witte, M. (2023). The political ideology of conversational AI: Converging evidence on ChatGPT's pro-environmental, left-libertarian orientation. *SSRN Electronic Journal*. DOI: https://doi.org/10.2139/ssrn.4316084

Horton, J. J. (2023). Large language models as simulated economic agents: What can we learn from homo silicus? *NBER Working Paper*, (w31122). DOI: https://doi.org/10.3386/w31122

Kim, B., Kim, H., Lee, S. W., Lee, G., Kwak, D., Jeon, D. H., ... & Sung, N. (2021). What changes can large-scale language models bring? Intensive study on HyperCLOVA: Billions-scale Korean generative pretrained transformers. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. DOI: https://doi.org/10.18653/v1/2021.emnlp-main.274

Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145-151. DOI: https://doi.org/10.1109/18.61115

Motoki, F., Neto, V. P., & Rodrigues, V. (2023). More human than human: Measuring ChatGPT political bias. *Public Choice*, 198(1), 3-23. DOI: https://doi.org/10.1007/s11127-023-01097-2

Naous, T., Ryan, M. J., Ritter, A., & Xu, W. (2024). Having beer after prayer? Measuring cultural bias in large language models. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. DOI: https://doi.org/10.18653/v1/2024.acl-long.862

Ornstein, J. T., Blasingame, E. N., & Truscott, J. S. (2024). How to train your stochastic parrot: Large language models for political texts. *Political Science Research and Methods*, 13(2), 264-281. DOI: https://doi.org/10.1017/psrm.2024.64

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, 1-22. DOI: https://doi.org/10.1145/3586183.3606763

Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose opinions do language models reflect? *arXiv preprint arXiv:2303.17548*.

Sungkyunkwan University Survey Research Center. (2023). *Korean General Social Survey 2023 Codebook*. Available from Korean Social Science Data Archive (KOSSDA). https://kossda.snu.ac.kr

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

---

## Appendix A: Reproducibility Information

| Item | Specification |
|------|---------------|
| GPT-5.2 | API identifier: `gpt-5.2`, accessed Dec 2025 |
| CLOVA HCX-007 | API identifier: `HCX-007`, accessed Dec 2025 |
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
| SATFIN | 귀하 댁의 전반적인 가계 경제 상태에 대해 어느 정도 만족하십니까? | 1=매우 만족, 2=다소 만족, 3=보통, 4=다소 불만족, 5=매우 불만족 |

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

Full code and detailed results available at: https://github.com/LeeSeogMin/silicon_sampling
