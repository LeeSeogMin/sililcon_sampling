# Silicon Sampling for Korean Social Surveys

**Validating LLM-based Survey Simulation with KGSS 2023 Data**

This repository contains the experimental code for validating Silicon Sampling methodology in the Korean social survey context.

## 📋 Paper

**Title**: Validating Silicon Sampling for Korean Social Surveys: A Large-Scale Comparison with KGSS 2023 Data

**Abstract**: This study empirically validates whether LLM-based Silicon Sampling can accurately reproduce real Korean social survey data. Using 100 stratified personas and GPT-5.2, we collected responses for 7 key KGSS 2023 variables and compared distributions using statistical tests.

**Note**: Manuscript drafts and internal research materials are stored separately and are not included in the public GitHub release. See `PUBLIC_RELEASE.md`.

## 🎯 Research Questions

1. **RQ1**: How accurately does LLM-based Silicon Sampling reproduce KGSS 2023 response distributions?
2. **RQ2**: What systematic biases exist in Silicon Sampling, and what patterns do they show?
3. **RQ3**: For which types of variables is Silicon Sampling suitable/unsuitable?

## 📊 Key Findings

### Accuracy (RQ1)
- All 7 variables showed statistically significant differences (KS test p<.001)
- Average Jensen-Shannon divergence: **0.226** (moderate similarity, insufficient for replacement)

### Systematic Biases (RQ2)
1. **Positivity Bias**: 96% positive responses for UNIFI (actual: 26%)
2. **Extremity Avoidance**: 14.5%p lower extreme response rate on average
3. **Negativity Omission**: 0% "North Korea responsible" for NORTHWHO (actual: 45%)

### Variable-Specific Performance (RQ3)
| Variable | JS Divergence | Assessment | Usage |
|----------|---------------|------------|-------|
| CONFINAN (Financial trust) | 0.104 | Best | ⚠️ Limited use |
| SATFIN (Economic satisfaction) | 0.134 | Good | ⚠️ Limited use |
| KRPROUD (National pride) | 0.158 | Moderate | ⚠️ Caution |
| NORTHWHO (NK responsibility) | 0.190 | Moderate | ⚠️ Caution |
| PARTYLR (Political orientation) | 0.296 | Poor | ❌ Avoid |
| UNIFI (Unification necessity) | 0.339 | Very Poor | ❌ Prohibited |
| CONLEGIS (Parliamentary trust) | 0.361 | Critical Failure | ❌ Prohibited |

## 🔬 Experimental Design

### Variables (7 KGSS 2023 items)
- **SATFIN**: Household economic satisfaction (1-5 scale)
- **CONFINAN**: Trust in financial institutions (recoded; see `data/kgss_benchmarks_2023.json`)
- **CONLEGIS**: Trust in National Assembly (recoded; see `data/kgss_benchmarks_2023.json`)
- **PARTYLR**: Political orientation (0-10 scale)
- **NORTHWHO**: Responsibility for NK relations deterioration (see benchmark)
- **UNIFI**: Necessity of unification (see benchmark)
- **KRPROUD**: Pride in being Korean (see benchmark)

### Personas
- **Sample size**: n=100
- **Stratification**: Age, gender, education, region, occupation
- **Distribution**: Matched to KGSS 2023 demographic proportions

### LLM Setup
- **Primary model**: GPT-5.2 (OpenAI, 2025)
- **Comparison model**: GPT-4o-mini (for model improvement analysis)
- **Temperature**: 0.7
- **API calls**: 1,000 total (700 per model)

## 📁 Repository Structure

```
silicon_sampling/
├── README.md                  # This file
├── PUBLIC_RELEASE.md          # What to publish (and not)
├── requirements.txt           # Python dependencies
├── data/                      # Benchmark data
│   └── kgss_benchmarks_2023.json
├── config/                    # Variable definitions for prompting
├── code/                      # Experimental code
│   ├── 01_generate_personas.py
│   ├── 02_run_main_experiment.py
│   ├── 03_gpt5_comparison.py
│   └── 04_statistical_analysis.py
└── outputs/                   # Generated artifacts (gitignored)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/silicon_sampling.git
cd silicon_sampling

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Key

Create `.env` file (never commit this):
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Experiments

```bash
# Step 1: Generate personas (KGSS 2023 demographic distribution)
python code/01_generate_personas.py

# Step 2: Run main experiment (GPT-5.2, n=100, 7 variables)
python code/02_run_main_experiment.py

# Step 3: Run GPT-5.2 vs GPT-4o-mini comparison
python code/03_gpt5_comparison.py

# Step 4: Statistical analysis
python code/04_statistical_analysis.py
```

### 4. View Results

Run artifacts are saved under `outputs/` (ignored by git).

## 📈 Reproducibility

### System Requirements
- Python 3.8+
- OpenAI API access (GPT-4 or GPT-5 models)
- ~$15 USD for full experiment replication (1,000 API calls)

### Expected Runtime
- Persona generation: <1 minute
- Main experiment (700 calls): 25-30 minutes
- Model comparison (300 calls): 10-15 minutes
- Statistical analysis: <1 minute

### Verification
This public repo includes **aggregate KGSS benchmarks** only (no microdata). Analysis scripts are designed to reproduce comparisons against these benchmark distributions.

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{silicon_sampling_korean_2025,
  title={Validating Silicon Sampling for Korean Social Surveys: A Large-Scale Comparison with KGSS 2023 Data},
  author={[Author Name]},
  journal={[Journal Name]},
  year={2025},
  note={GitHub: https://github.com/YOUR_USERNAME/silicon_sampling}
}
```

## 🔍 Key Results Summary

### Main Conclusion
> Silicon Sampling **cannot replace** traditional social surveys as of 2025, but shows **limited potential** for pilot studies and exploratory analysis.

### Practical Guidelines

**✅ Acceptable Use** (with caution):
- Pilot testing of questionnaires
- Exploratory hypothesis generation
- Policy scenario simulations (reference only)
- Educational training

**❌ Prohibited Use**:
- Policy decision-making
- Primary data for academic research
- Political polling
- Sensitive topics (discrimination, ethics)

### Validation Requirements
Before using Silicon Sampling for any variable:
1. Run pilot test (n=30)
2. Compare with KGSS or similar survey data
3. Verify JS divergence < 0.15
4. Re-validate after model updates

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or collaboration:
- **Email**: [your_email@example.com]
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/silicon_sampling/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **KGSS (Korean General Social Survey)**: Data source for benchmarking
- **OpenAI**: GPT models for experiments
- **Research funding**: [If applicable]

## 📚 Related Work

- Argyle et al. (2023): "Out of One, Many: Using Language Models to Simulate Human Samples"
- Horton (2023): "Large Language Models as Simulated Economic Agents"
- Park et al. (2023): "Generative Agents: Interactive Simulacra of Human Behavior"

---

**Last Updated**: December 20, 2025
**Repository Status**: Active Development
**Version**: 1.0.0
