# Do Indigenous LLMs Show Closer Alignment with Local Survey Responses? Evidence from Korean Cultural Variables

## Abstract

Large language models (LLMs) are increasingly used to simulate survey responses (“silicon sampling”), but it remains unclear whether locally trained (“indigenous”) models better reproduce local public opinion in a distributional sense. Using marginal distributions from the Korean General Social Survey (KGSS) 2023 as benchmarks, we compare a global model (GPT-5.2) with a Korean indigenous model (CLOVA HCX-007) on six culturally salient variables spanning institutional trust, national pride, inter-Korean attitudes, and political orientation. We quantify alignment using Jensen–Shannon (JS) divergence (natural log) and report Kolmogorov–Smirnov (KS) tests as a secondary check for distributional differences. In the baseline persona-prompting setting, GPT-5.2 shows heterogeneous alignment across variables (average JS=0.123), including extreme response concentration on UNIFI. In the cultural-context comparison, CLOVA exhibits mixed performance relative to GPT-5.2 (average JS=0.090 vs. 0.123; 3/6 variables improved), with large gains on UNIFI (56.9%) but worse performance on PARTYLR (−70.7%). Neither model achieves distributional equivalence with KGSS under the reported KS tests (all p<0.05 for CLOVA). Overall, cultural context effects appear variable-dependent rather than a uniform “local model advantage,” motivating variable-level pilot evaluation before substantive use of silicon sampling outputs.

**Keywords**: Silicon Sampling, Large Language Models, Korean General Social Survey, Indigenous LLM, CLOVA, Cultural Context, GPT-5.2

---

## 1. Introduction

### 1.1 Research Background: Proliferation of Silicon Sampling Methodology Claims

Survey research using Large Language Models (LLMs) for survey simulation, termed "Silicon Sampling," has been proposed as an innovative methodology to overcome the cost, time, and accessibility limitations of traditional social surveys. Since Argyle et al. (2023) reported high accuracy in simulating U.S. GSS using GPT-3, researchers have proposed various methodological improvements.

However, these methodological claims have mostly been validated independently in individual studies, with insufficient integrated and systematic comparative research. Moreover, most research has focused on English-speaking (particularly U.S.) data, leaving generalizability to non-Western contexts unverified.

### 1.2 Study Focus

This paper evaluates silicon sampling in a setting where cultural context plausibly matters: Korean public-opinion items from KGSS 2023. We focus on two practical questions that frequently arise in applied work: (i) whether a straightforward demographic-persona prompting setup produces marginal distributions close to survey benchmarks, and (ii) whether an indigenous model trained primarily on Korean data exhibits closer alignment than a global model. We treat “alignment” as a distributional property rather than a single-response accuracy notion, and report both divergence metrics and hypothesis-test results with clear caveats for ordinal/discrete outcomes.

### 1.3 Research Questions

- **RQ1 (Baseline validity)**: Under a standard persona-based prompting protocol, how closely do LLM-generated response distributions match KGSS 2023 marginal distributions?
- **RQ2 (Cultural context)**: Does a Korean indigenous model (CLOVA HCX-007) yield closer alignment than a global model (GPT-5.2), and is any advantage variable-dependent?

### 1.4 Research Significance

**Academic Contributions**:
1. Integrated empirical evaluation of multiple silicon sampling claims
2. Evidence from a non-Western (Korean) benchmark survey
3. A reproducible evaluation framework (distributional metrics and tests)
4. Empirical evidence on the scope and limits of cultural-context effects in survey simulation

**Practical Contributions**:
1. Practical guidance for model and prompt selection via variable-level piloting
2. A transparent reporting template for silicon sampling studies (metrics, tests, prompts)
3. Precautions regarding distributional mismatch and stereotype amplification risks

---

## 2. Related Work

### 2.1 LLM-Based Survey Simulation

Recent work has explored using LLMs to simulate survey responses. Argyle et al. (2023) and Aher et al. (2022) evaluate whether persona-conditioned LLM outputs can reproduce survey patterns in controlled settings. These studies establish feasibility but leave open questions about distributional validity across cultures and models, particularly in non-Western contexts.

### 2.2 Cultural Bias in Language Models

LLMs exhibit measurable cultural biases reflecting their training data distributions. Cao et al. (2023) showed GPT models systematically favor Western cultural norms in value judgments. Naous et al. (2024) demonstrated Arabic cultural misalignment in multilingual models. However, few studies examine whether indigenous LLMs—trained on culturally-specific corpora—mitigate these biases in survey contexts.

### 2.3 Indigenous LLMs and Cultural Contextualization

The development of indigenous LLMs (e.g., HyperCLOVA in Korea) offers natural experiments in cultural alignment. These models incorporate local language data, cultural knowledge, and regional fine-tuning (Kim et al., 2021). However, systematic survey-simulation comparisons that directly benchmark distributional alignment remain limited.

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

We use the Korean General Social Survey (KGSS) 2023, a nationally representative survey (n≈1,500) conducted by Sungkyunkwan University Survey Research Center (2023). We selected six variables capturing culturally sensitive domains:

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

- **GPT-5.2**: OpenAI model (API identifier: `gpt-5.2`, accessed December 2025)†
- **CLOVA HCX-007**: Naver's Korean indigenous LLM with reasoning capabilities (API identifier: `HCX-007`, accessed December 2025)

†Note: "GPT-5.2" refers to the model identifier used in our API calls. As model naming conventions and availability may vary, we report the exact API identifier used for reproducibility. All experiments used the OpenAI Chat Completions API with the specified identifiers.

### 3.4 Experimental Design

We ran two experiments using a consistent persona-prompting protocol:

- **Experiment 1 (Baseline simulation)**: Generate n=100 persona-conditioned responses per variable using GPT-5.2 and compare the resulting marginal distributions to KGSS 2023 benchmarks.
- **Experiment 2 (Cultural-context comparison)**: Repeat the same procedure with CLOVA HCX-007 and compare model–benchmark alignment (CLOVA vs. KGSS) as well as relative alignment (CLOVA vs. GPT-5.2) on the same variables.

For both models, we used a fixed set of 100 personas (stratified to match KGSS 2023 demographic marginals) and a shared Korean prompt template (Appendix B). The GPT-5.2 run configuration and metric outputs are stored under `results/gpt52_experiment/`, and the CLOVA response sets are stored under `results/clova_experiment/`.

### 3.5 Evaluation Metrics

**Jensen–Shannon Divergence (JS)**: Symmetric measure of distribution similarity. We report JS divergence computed with the natural logarithm (Lin, 1991), which is bounded in \[0, ln 2\] (0 indicates identical distributions). For convenience, readers can obtain a \[0,1\] normalized value by dividing by ln 2.

**Kolmogorov-Smirnov Test (KS)**: Two-sample test for distribution equality. We use α = 0.05; non-significant results (p > 0.05) indicate failure to detect statistically significant differences between distributions. Note that non-significance does not prove distribution equivalence—it may reflect insufficient statistical power, particularly with n=100 samples.

**Methodological Note on KS Tests for Ordinal Data**: The KS test is designed for continuous distributions and may not optimally capture differences in ordinal/discrete survey response data. Alternative metrics such as Earth Mover's Distance (Wasserstein-1) or chi-square tests may be more appropriate for ordinal scales. We retain KS tests for comparability with prior silicon sampling literature while acknowledging this limitation.

---

## 4. Results

This section reports results for baseline distributional alignment and for the cultural-context comparison.

### 4.1 Experiment 1: Baseline Simulation

Using GPT-5.2 (T=0.7) with demographic persona prompts (age, gender, education, region, occupation), we generated n=100 responses per variable.

| Variable | JS Divergence (ln) |
|----------|---------------------|
| CONFINAN | 0.062 |
| CONLEGIS | 0.134 |
| KRPROUD | 0.113 |
| NORTHWHO | 0.125 |
| UNIFI | 0.267 |
| PARTYLR | 0.038 |
| **Average** | **0.123** |

The baseline setting yields variable-dependent alignment. Some items show strong response concentration (e.g., UNIFI concentrates on a single option in this run), whereas others (e.g., PARTYLR) are closer to the KGSS marginals. Overall, this motivates evaluating alignment at the variable level rather than assuming uniform performance across survey domains.

### 4.2 Experiment 2: Cultural Context

**Claim tested**: Indigenous LLMs trained on local cultural data outperform global models in reproducing culturally-specific response patterns.

CLOVA HCX-007 (Korean indigenous LLM) showed mixed performance relative to GPT-5.2 (Western global LLM):

| Variable | GPT-5.2 JS | CLOVA JS | Improvement | CLOVA Better |
|----------|------------|----------|-------------|--------------|
| CONFINAN | 0.062 | 0.062 | -0.2% | ✗ |
| CONLEGIS | 0.134 | 0.083 | 38.5% | ✓ |
| KRPROUD | 0.113 | 0.134 | -18.9% | ✗ |
| NORTHWHO | 0.125 | 0.084 | 32.6% | ✓ |
| UNIFI | 0.267 | 0.115 | 56.9% | ✓ |
| PARTYLR | 0.038 | 0.065 | -70.7% | ✗ |
| **Average** | **0.123** | **0.090** | **26.5%** | **3/6** |

*Note: GPT-5.2 values are computed from `results/gpt52_experiment/metrics.json`, while CLOVA values use the saved CLOVA response sets in `results/clova_experiment/*/clova_results.json`.*

**Interpretation**: The results show no clear indigenous LLM advantage. CLOVA outperforms GPT-5.2 on only 3/6 variables, indicating an inconclusive advantage pattern. The 26.5% average improvement is heavily influenced by GPT-5.2's response concentration on UNIFI (JS=0.267, with 100% responses on a single option). Excluding UNIFI, the average difference shrinks to approximately 9%. GPT-5.2 substantially outperforms CLOVA on PARTYLR (political orientation), a key cultural variable, suggesting that indigenous training does not guarantee superiority across all culturally-sensitive domains.

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

**Conclusion**: The cultural-context hypothesis receives mixed support. CLOVA outperforms GPT-5.2 on 3/6 variables with 26.5% lower average JS divergence, but the aggregate advantage is heavily influenced by GPT-5.2’s extreme concentration on UNIFI. Neither model achieves distributional equivalence with KGSS benchmarks under the reported KS tests (α=0.05).

### 4.3 Supplementary Statistical Analyses

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

### 5.1 Baseline Distributional Validity

In the baseline persona-prompting setting, alignment is heterogeneous across variables (Table 4.1). Even within a single model run, some items exhibit strong response concentration (e.g., UNIFI), while others are closer to the KGSS marginals (e.g., PARTYLR). This motivates reporting silicon sampling performance at the variable level rather than assuming a stable “survey simulation quality” across domains.

### 5.2 Cultural Context and Variable Dependence

Our results show no uniform evidence for indigenous LLM superiority in distributional alignment. While CLOVA HCX-007 achieved 26.5% lower average JS divergence than GPT-5.2, this advantage is heavily driven by GPT-5.2’s extreme response concentration on unification attitudes (UNIFI), where 100% of responses fall on a single option. CLOVA outperformed GPT-5.2 on 3/6 variables, whereas GPT-5.2 substantially outperformed CLOVA on political orientation (PARTYLR) by 71%, suggesting that any “indigenous advantage” may be variable-dependent rather than general. KS tests reject benchmark equality for all CLOVA variables at α=0.05, indicating that CLOVA does not achieve distributional equivalence under this test.

The variable-dependent performance pattern suggests that the relationship between model origin and cultural alignment is more complex than a simple indigenous advantage hypothesis. Possible explanations include: (1) variable-specific coverage in training data (e.g., differential exposure to Korean political discourse vs. inter-Korean issues), (2) prompt-framing sensitivity that varies by domain, and (3) model-specific priors that manifest differently across attitudinal dimensions.

### 5.3 Response-Format Compliance as a Measurement Issue

Survey simulation depends on strict response-format compliance (e.g., “answer with a single integer”), and post-processing rules (Appendix B.4) can meaningfully affect distributions. For interpretability and fair comparisons, silicon sampling studies should report response validity rates (e.g., the fraction of responses requiring default assignment) alongside divergence metrics.

### 5.4 Implications and Limitations

Practically, these findings support variable-level pilot testing and transparent reporting (prompt templates, parsing rules, and validity rates) before using silicon sampling outputs for substantive inference. Methodologically, our evaluation is limited by focusing on marginal distributions (not joint or conditional distributions), by a modest sample size (n=100 per variable), and by the use of the KS test on ordinal/discrete outcomes. Future work should (i) evaluate joint distributions and subgroup patterns, (ii) repeat runs to quantify stochastic variability, and (iii) compare ordinal-appropriate goodness-of-fit measures (e.g., chi-square or Wasserstein-1 on ordered categories).

---

## 6. Conclusion

This study evaluated distributional alignment in silicon sampling using KGSS 2023 marginals and compared a global model (GPT-5.2) with a Korean indigenous model (CLOVA HCX-007) on six culturally salient variables.

In the baseline persona-prompting setting, GPT-5.2 shows heterogeneous alignment (average JS=0.123), including extreme response concentration on UNIFI. In the cultural-context comparison, CLOVA yields a lower average divergence (JS=0.090) but improves only 3/6 variables relative to GPT-5.2, indicating that any “indigenous advantage” is variable-dependent rather than uniform. Under the reported KS tests, CLOVA distributions differ significantly from KGSS benchmarks for all variables (p<0.05), underscoring that lower divergence does not necessarily imply distributional equivalence.

Overall, these results recommend (i) variable-level pilot evaluation across models, (ii) transparent reporting of prompts and parsing rules, and (iii) explicit reporting of response-format compliance when using LLMs for survey simulation. Future work should broaden cultural settings, evaluate joint/conditional distributions, and assess robustness across repeated runs and alternative goodness-of-fit metrics for ordinal outcomes.

---

## References

Aher, G. V., Arriaga, R. I., & Kalai, A. T. (2022). Using large language models to simulate multiple humans and replicate human subject studies. *arXiv preprint arXiv:2208.10264*. https://arxiv.org/abs/2208.10264

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351. https://doi.org/10.1017/pan.2023.2

Cao, Y., Zhou, L., Lee, S., Cabello, L., Chen, M., & Hershcovich, D. (2023). Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. *Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP)*. https://doi.org/10.18653/v1/2023.c3nlp-1.7

Kim, B., Kim, H., Lee, S. W., Lee, G., Kwak, D., Jeon, D. H., ... & Sung, N. (2021). What changes can large-scale language models bring? Intensive study on HyperCLOVA: Billions-scale Korean generative pretrained transformers. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*. https://doi.org/10.18653/v1/2021.emnlp-main.274

Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145-151. https://doi.org/10.1109/18.61115

Naous, T., Ryan, M. J., Ritter, A., & Xu, W. (2024). Having beer after prayer? Measuring cultural bias in large language models. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. https://doi.org/10.18653/v1/2024.acl-long.862

Sungkyunkwan University Survey Research Center. (2023). *Korean General Social Survey 2023 Codebook*. Korean Social Science Data Archive (KOSSDA). https://kossda.snu.ac.kr

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

---
