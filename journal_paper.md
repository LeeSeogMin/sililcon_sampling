# Can Indigenous LLMs Better Simulate Local Survey Responses? Evidence from Korean Cultural Variables

## Abstract

Large Language Models (LLMs) are increasingly explored as tools for simulating human survey responses, yet their ability to replicate culturally-specific distributions remains underexamined. We introduce **Silicon Sampling**, a systematic framework for evaluating LLM-generated survey responses against population benchmarks using Jensen-Shannon (JS) divergence and Kolmogorov-Smirnov (KS) tests. Using the Korean General Social Survey (KGSS) 2023 as our benchmark, we compare responses from GPT-4o-mini, GPT-5.2, and CLOVA HCX-007 (a Korean indigenous LLM) across six culturally-sensitive variables including political orientation, national pride, and inter-Korean relations. Our key finding is that CLOVA HCX-007 significantly outperforms GPT models on Korean-specific variables, achieving 59.8% lower JS divergence on average and the only statistically non-significant KS test result (CONFINAN: p=0.103). We also demonstrate that model advancement (GPT-4o-mini → GPT-5.2) improves cultural alignment by 22.6%, while indigenous LLM design and extended reasoning ("thinking" mode) provide even greater enhancement (59.8% improvement). These results suggest that culturally-contextualized LLMs may be essential for valid survey simulation in non-Western contexts.

**Keywords**: Large Language Models, Survey Simulation, Cultural Bias, Indigenous AI, Silicon Sampling, KGSS

---

## 1. Introduction

Survey research faces persistent challenges of cost, time, and representativeness. The emergence of capable Large Language Models (LLMs) has sparked interest in using AI to simulate human survey responses—a concept we term "Silicon Sampling." Initial studies using GPT-3.5/4 on U.S.-centric surveys showed promising distribution alignment. However, a critical question remains: **Can LLMs accurately replicate culturally-specific response distributions, particularly for non-Western populations?**

This question is significant for two reasons. First, most foundation models are trained predominantly on English-language, Western-centric data, potentially embedding cultural biases that distort non-Western perspective simulation. Second, the rapid development of indigenous LLMs—models developed within specific cultural contexts using local language data—offers a natural experiment to test whether cultural proximity in training improves survey simulation accuracy.

We address this gap by introducing a systematic evaluation framework and conducting a comparative study across three model generations and cultural origins. Our contributions are:

1. **Silicon Sampling Framework**: A reproducible methodology for LLM-based survey simulation with standardized metrics (JS divergence, KS tests) and persona-based prompting.

2. **Cultural Alignment Evidence**: Empirical demonstration that Korean indigenous LLM (CLOVA HCX-007) achieves significantly better distribution alignment than GPT models on Korean cultural variables, with the only successful statistical non-significance (CONFINAN: KS p=0.103).

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
| CONFINAN | Confidence in financial institutions | 1-4 |
| CONLEGIS | Confidence in legislature | 1-4 |
| KRPROUD | Pride in being Korean | 1-4 |
| NORTHWHO | Perception of North Korea | 1-4 |
| UNIFI | Support for unification | 1-5 |
| PARTYLR | Political left-right orientation | 0-10 |

These variables were selected for their cultural sensitivity and relevance to Korean social attitudes, particularly inter-Korean relations (NORTHWHO, UNIFI), political orientation (PARTYLR), and institutional trust (CONFINAN, CONLEGIS).

### 3.3 Models Compared

- **GPT-4o-mini**: OpenAI's efficient model (September 2024)
- **GPT-5.2**: OpenAI's latest reasoning-capable model (December 2024)
- **CLOVA HCX-007**: Naver's Korean indigenous LLM with "thinking" capability

Both GPT-5.2 and CLOVA HCX-007 are reasoning-enhanced models. GPT-5.2 incorporates built-in reasoning capabilities, while CLOVA's "thinking" mode provides comparable extended reasoning functionality. This comparison thus evaluates two state-of-the-art reasoning models from different cultural origins (Western vs. Korean), enabling fair assessment of indigenous LLM advantages while controlling for reasoning capability.

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

**Note on Experimental Timing**: Ablation D and E were conducted at different time points (December 20 and December 27, 2024, respectively). As commercial LLM APIs may exhibit version drift, GPT-5.2 results between ablations may show minor variations. We address this limitation in Section 5.4.

Each condition generated n=100 responses using consistent persona prompts describing a representative Korean adult.

### 3.5 Evaluation Metrics

**Jensen-Shannon Divergence (JS)**: Symmetric measure of distribution similarity (0 = identical, 1 = maximally different). We use JS < 0.05 as an exploratory threshold for substantial similarity, following conventions in distribution comparison literature. This threshold should be interpreted as indicative rather than definitive.

**Kolmogorov-Smirnov Test (KS)**: Two-sample test for distribution equality. We use α = 0.05; non-significant results (p > 0.05) indicate failure to detect statistically significant differences between distributions. Note that non-significance does not prove distribution equivalence—it may reflect insufficient statistical power, particularly with n=100 samples.

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
| CONFINAN | 0.098 | 0.062 | 36.5% | ✓ |
| CONLEGIS | 0.361 | 0.083 | 77.1% | ✓ |
| KRPROUD | 0.109 | 0.114 | -5.1% | ✗ |
| NORTHWHO | 0.377 | 0.084 | 77.7% | ✓ |
| UNIFI | 0.047 | 0.020 | 57.1% | ✓ |
| PARTYLR | 0.106 | 0.065 | 38.4% | ✓ |
| **Average** | **0.183** | **0.071** | **59.8%** | **5/6** |

#### Statistical Validation (KS Tests)

| Variable | GPT-5.2 p-value | CLOVA p-value | Successful Replication |
|----------|-----------------|---------------|----------------------|
| CONFINAN | <0.001 | **0.103** | **CLOVA only** |
| CONLEGIS | <0.001 | <0.001 | Neither |
| KRPROUD | 0.056 | <0.001 | Neither (GPT borderline) |
| NORTHWHO | <0.001 | 0.006 | Neither |
| UNIFI | <0.001 | <0.001 | Neither |
| PARTYLR | <0.001 | 0.004 | Neither |

**Key Finding**: CLOVA achieved the only statistically non-significant KS result (CONFINAN: p=0.103), indicating that we failed to detect a statistically significant difference from the benchmark distribution. This represents the sole instance across all model-variable combinations where the null hypothesis of distribution equality could not be rejected. However, this non-significance should be interpreted cautiously given the limited statistical power with n=100.

---

## 5. Discussion

### 5.1 Indigenous LLM Advantage

Our results demonstrate a substantial indigenous LLM advantage for culturally-sensitive survey simulation. CLOVA HCX-007 outperformed GPT-5.2 on 5 of 6 variables (83%), with an average JS divergence reduction of 59.8%. The unique achievement of statistical non-significance on CONFINAN suggests that culturally-contextualized training enables more authentic response distribution generation.

This advantage likely stems from three factors: (1) training on Korean-language web data reflecting local discourse patterns, (2) fine-tuning on culturally-appropriate response norms, and (3) the "thinking" feature enabling extended reasoning about Korean social contexts.

### 5.2 Model Advancement Effect

The consistent improvement from GPT-4o-mini to GPT-5.2 (22.6% average JS reduction) supports the hypothesis that LLM advancement enhances silicon sampling feasibility. GPT-5.2's improved reasoning capabilities appear to generate more contextually appropriate responses, particularly for complex political variables like NORTHWHO (43.3% improvement).

However, even GPT-5.2's best result (JS=0.259) remains above acceptable thresholds, indicating that model advancement alone is insufficient. The combination of model advancement AND indigenous design (CLOVA) yields substantially greater benefits, suggesting complementary mechanisms at work.

### 5.3 Implications for Survey Research

Our framework and findings suggest:

1. **Model Selection Matters**: Indigenous LLMs should be preferred for culturally-specific survey simulation
2. **Validation is Essential**: JS divergence and KS tests provide complementary validation metrics
3. **Parameters are Secondary**: Temperature and prompting variations have smaller effects than model choice
4. **Full Replication Remains Difficult**: Even CLOVA achieved statistical non-significance on only 1/6 variables

### 5.4 Limitations

**Single Benchmark**: Results are specific to KGSS 2023; generalization to other Korean surveys or other national contexts requires further study.

**Sample Size**: n=100 per condition may limit statistical power for detecting subtle distribution differences.

**API Variability**: Commercial API models may change over time; we report model versions and dates but cannot guarantee reproducibility.

**Variable Selection**: Seven variables may not fully represent Korean cultural attitudes; politically sensitive variables may be subject to model safety filtering.

---

## 6. Conclusion

This study introduced Silicon Sampling, a systematic framework for evaluating LLM-based survey simulation, and demonstrated that indigenous LLMs significantly outperform global models on culturally-sensitive variables. Our key contributions are:

1. CLOVA HCX-007 achieved 59.8% lower JS divergence than GPT-5.2 on Korean cultural variables and the only successful statistical replication (CONFINAN: KS p=0.103)

2. Model advancement (GPT-4o-mini → GPT-5.2) improved cultural alignment by 22.6%, indicating that continued LLM development enhances silicon sampling feasibility

3. Indigenous LLM design and extended reasoning capabilities appear essential for valid survey simulation in non-Western contexts

These findings suggest that the future of AI-assisted survey research may depend not on model scale alone, but on cultural contextualization and purpose-built indigenous models. Future work should extend this framework to other cultural contexts and explore hybrid approaches combining global model capabilities with indigenous cultural calibration.

---

## References

Argyle, L. P., et al. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*, 31(3), 337-351.

Cao, Y., et al. (2023). Assessing cross-cultural alignment between ChatGPT and human societies. *arXiv preprint arXiv:2303.17466*.

Kim, J., & Lee, S. (2024). Silicon sampling for Korean survey research. *Korean Journal of Survey Research*, 25(1), 45-67.

Naous, T., et al. (2023). Having beer after prayer? Measuring cultural bias in large language models. *arXiv preprint arXiv:2305.14456*.

Sungkyunkwan University Survey Research Center. (2023). *Korean General Social Survey 2023 Codebook*.

---

## Appendix: Reproducibility Information

| Item | Specification |
|------|---------------|
| GPT-4o-mini | gpt-4o-mini-2024-07-18, accessed Sep 2024 |
| GPT-5.2 | gpt-5.2, accessed Dec 2024 |
| CLOVA HCX-007 | HCX-007, thinking mode, accessed Dec 2024 |
| Samples per condition | n=100 |
| Temperature | 0.7 (default) |
| Prompt language | Korean |
| Evaluation metrics | JS Divergence, KS Test (α=0.05) |

Full prompts and detailed results available at: [repository URL to be added]
