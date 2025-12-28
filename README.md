# Silicon Sampling: Korean LLM Survey Simulation

**Do Indigenous LLMs Show Closer Alignment with Local Survey Responses?**

Replication code for systematic validation of Silicon Sampling methodology using Korean General Social Survey (KGSS) 2023 data.

## Paper Abstract

This study systematically validates Silicon Sampling through two research axes: (1) methodological effectiveness validation, and (2) cultural context effects. We test whether indigenous LLMs (CLOVA HCX-007) outperform Western LLMs (GPT-5.2) in simulating Korean cultural contexts.

**Key Findings:**
- Baseline simulation shows significant differences (mean JS=0.397)
- Temperature optimization: T=0.7 optimal
- Chain-of-Thought: 19.1% improvement
- Prompt engineering: Korean prompts 16.8% better than English
- **Cultural Context**: CLOVA wins 3/6 variables with 26.5% average JS improvement, but no clear indigenous LLM advantage

## Repository Structure

```
silicon_sampling/
├── code/                          # Experiment scripts
│   ├── 01_generate_personas.py    # Generate 100 stratified personas
│   ├── 02_run_main_experiment.py  # GPT experiments (Exp 1-4)
│   ├── 04_statistical_analysis.py # KS test, Chi-square, JS divergence
│   ├── 05_report_experiments.py   # Generate reports
│   ├── 06_clova_experiment.py     # CLOVA HCX-007 experiment (Exp 5)
│   └── 06_update_paper_bootstrap_table.py
├── config/
│   └── kgss_variables_2023.json   # Variable definitions & prompts
├── data/
│   └── kgss_benchmarks_2023.json  # KGSS 2023 benchmark distributions
├── results/                       # Experiment results
│   ├── clova_experiment/          # CLOVA HCX-007 results (Exp 5)
│   ├── gpt52_experiment/          # GPT-5.2 results
│   └── gpt35turbo_experiment/     # GPT-3.5-turbo baseline
├── docs/
│   └── journal_paper.md           # Full paper manuscript
├── ss_utils.py                    # Shared utilities (JS divergence, etc.)
├── clova_client.py                # CLOVA Studio API client
├── requirements.txt               # Python dependencies
└── .env.example                   # API key template
```

## Five Experiments

| Exp | Claim Tested | Script | Description |
|-----|--------------|--------|-------------|
| 1 | Baseline Simulation | `02_run_main_experiment.py` | Demographic persona-based reproduction |
| 2 | Temperature Optimization | `02_run_main_experiment.py --temperature` | T=0.3-1.1 comparison |
| 3 | Chain-of-Thought | `02_run_main_experiment.py` + CoT prompt | CoT vs Direct response |
| 4 | Prompt Engineering | `02_run_main_experiment.py` | Korean vs English prompts |
| 5 | Cultural Context | `06_clova_experiment.py` | CLOVA HCX-007 vs GPT-5.2 |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/LeeSeogMin/silicon_sampling.git
cd silicon_sampling
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `OPENAI_API_KEY`: For GPT experiments
- `CLOVA_STUDIO_API_KEY`: For CLOVA experiments

### 3. Run Experiments

```bash
# Step 1: Generate 100 stratified personas
python code/01_generate_personas.py --seed 42 --out outputs/personas/personas_100.json

# Step 2: Run GPT experiment (Experiments 1-4)
python code/02_run_main_experiment.py \
  --model gpt-5.2 \
  --temperature 0.7 \
  --n-samples 100

# Step 3: Run CLOVA experiment (Experiment 5)
python code/06_clova_experiment.py \
  --thinking medium \
  --n-samples 100

# Step 4: Statistical analysis
python code/04_statistical_analysis.py \
  --sim-csv outputs/runs/<timestamp>/persona_responses.csv
```

## Variables (KGSS 2023)

| Variable | Description | Scale |
|----------|-------------|-------|
| CONFINAN | Confidence in financial institutions | 1-3 |
| CONLEGIS | Confidence in legislature | 1-3 |
| KRPROUD | Pride in being Korean | 1-4 |
| NORTHWHO | Perception of North Korea | 1-4 |
| UNIFI | Support for unification | 1-4 |
| PARTYLR | Political orientation | 1-5 |
| SATFIN | Financial satisfaction | 1-5 |

## Key Results

### Experiment 5: Cultural Context (GPT-5.2 vs CLOVA HCX-007)

| Variable | GPT-5.2 JS | CLOVA JS | Improvement | Winner |
|----------|------------|----------|-------------|--------|
| CONFINAN | 0.062 | 0.062 | -0.2% | Tie |
| CONLEGIS | 0.134 | 0.083 | 38.5% | CLOVA |
| KRPROUD | 0.113 | 0.134 | -18.9% | GPT |
| NORTHWHO | 0.125 | 0.084 | 32.6% | CLOVA |
| UNIFI | 0.267 | 0.115 | 56.9% | CLOVA |
| PARTYLR | 0.038 | 0.065 | -70.7% | GPT |
| **Average** | **0.123** | **0.090** | **26.5%** | **3/6** |

**Conclusion**: Indigenous LLM training does not guarantee superiority across all culturally-sensitive domains.

## Reproducibility

### System Requirements
- Python 3.10+
- OpenAI API access (GPT-5.2 or similar)
- CLOVA Studio API access (HCX-007)

### API Costs (Approximate)
- GPT-5.2: ~$10-15 per 700 calls
- CLOVA HCX-007: ~$5-10 per 600 calls

### Random Seed
All experiments use `seed=42` for reproducibility.

## Citation

```bibtex
@article{silicon_sampling_korean_2025,
  title={Do Indigenous LLMs Show Closer Alignment with Local Survey Responses?
         Evidence from Korean Cultural Variables},
  author={[Author]},
  journal={[Journal]},
  year={2025},
  note={GitHub: https://github.com/LeeSeogMin/silicon_sampling}
}
```

## References

- Argyle et al. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis*
- Dillion et al. (2023). Can AI language models replace human participants? *Trends in Cognitive Sciences*
- Ornstein et al. (2024). How to train your stochastic parrot. *Political Science Research and Methods*
- KGSS (2023). Korean General Social Survey 2023. Sungkyunkwan University Survey Research Center

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Last Updated**: December 2025
