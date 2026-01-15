# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Silicon Sampling research codebase for validating LLM-based survey simulation methodology. Tests whether indigenous Korean LLM (CLOVA HCX-007) outperforms Western LLMs (GPT-5.2) in simulating Korean cultural survey responses using KGSS 2023 benchmark data.

**Research Question**: Do indigenous LLMs show closer alignment with local survey responses?

**Key Findings**: CLOVA wins 3/6 variables with 26.5% average JS improvement, but no clear indigenous LLM advantage across all culturally-sensitive domains.

## Core Architecture

### Experiment Pipeline

The codebase implements a 5-experiment pipeline executed sequentially:

1. **Persona Generation** (`01_generate_personas.py`)
   - Creates 100 stratified demographic personas matching KGSS 2023 distribution
   - Stratification dimensions: age-gender, education, region, occupation
   - Output: `outputs/personas/personas_100.json`

2. **GPT Experiments** (`02_run_main_experiment.py`)
   - Runs Experiments 1-4: Baseline, Temperature, CoT, Prompt Engineering
   - Async OpenAI API calls with retry logic
   - Default values assigned for non-compliant responses
   - Output: `outputs/runs/<timestamp>/persona_responses.csv`

3. **CLOVA Experiment** (`06_clova_experiment.py`)
   - Experiment 5: Cultural Context comparison
   - Uses CLOVA Studio Chat Completions v3 API
   - HCX-007 reasoning model with `thinking` parameter (short/medium/deep)
   - Output: `results/clova_experiment/`

4. **Statistical Analysis** (`04_statistical_analysis.py`)
   - Compares simulated vs. benchmark distributions
   - Metrics: KS test, Chi-square test, Jensen-Shannon divergence
   - Note: KS test has known limitations for ordinal data (acknowledged in paper)

5. **Reporting** (`05_report_experiments.py` and `06_update_paper_bootstrap_table.py`)
   - Generates markdown reports and tables for paper

### Key Data Flow

```
config/kgss_variables_2023.json → defines 7 survey variables + prompts
data/kgss_benchmarks_2023.json → real KGSS 2023 distributions (ground truth)
outputs/personas/ → generated personas
outputs/runs/ or results/ → experiment results (CSV + JSON)
```

### Core Utilities (`ss_utils.py`)

- **Benchmark**: Dataclass for KGSS ground truth distributions
- **js_divergence()**: Natural log Jensen-Shannon divergence (0 to ln(2))
- **parse_first_int()**: Extracts first integer from LLM response text
- **normalize_probabilities()**: Converts counts to probability distributions
- API credential loaders: OpenAI, CLOVA Studio, NCP API Gateway

### CLOVA Studio Integration (`clova_client.py`)

- Wrapper for CLOVA Studio Chat Completions v3 API
- HCX-007 is a reasoning model requiring `thinking` parameter for CoT
- Thinking levels: "short" (fast), "medium" (balanced), "deep" (thorough)
- Auth: Bearer token via `CLOVA_STUDIO_API_KEY`

## Common Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, CLOVA_STUDIO_API_KEY)
```

### Running Experiments

**IMPORTANT**: All Python scripts must be run from the repository root directory.

```bash
# 1. Generate personas (always run first)
python code/01_generate_personas.py --seed 42 --out outputs/personas/personas_100.json

# 2. Run GPT experiments (Exp 1-4)
python code/02_run_main_experiment.py \
  --model gpt-5.2 \
  --temperature 0.7 \
  --n-samples 100 \
  --concurrency 5

# Run specific variables only
python code/02_run_main_experiment.py \
  --model gpt-5.2 \
  --variables CONFINAN CONLEGIS \
  --n-samples 50

# 3. Run CLOVA experiment (Exp 5)
python code/06_clova_experiment.py \
  --thinking medium \
  --n-samples 100 \
  --delay 0.5

# Run with different thinking levels
python code/06_clova_experiment.py --thinking short --n-samples 50    # Fast
python code/06_clova_experiment.py --thinking deep --n-samples 50     # Thorough
python code/06_clova_experiment.py --thinking none --n-samples 50     # No CoT

# 4. Statistical analysis (requires prior experiment results)
python code/04_statistical_analysis.py --sim-csv outputs/runs/<timestamp>/persona_responses.csv

# 5. Generate reports
python code/05_report_experiments.py --exp-dir outputs/runs/<timestamp>
```

### Analysis Utilities

```bash
# Quick JS divergence calculation
python -c "from ss_utils import js_divergence; print(js_divergence([50, 30, 20], [40, 40, 20]))"

# Load and inspect benchmark data
python -c "from ss_utils import load_benchmark; b = load_benchmark(); print(b.analyzable_variables)"

# Test API connectivity
python -c "from ss_utils import load_clova_studio_api_key; print('CLOVA key loaded:', load_clova_studio_api_key() is not None)"
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OpenAI key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```

## Key Configuration Files

- **config/kgss_variables_2023.json**: Defines 7 KGSS variables
  - Each variable: `question`, `valid_responses`, `scale_labels`
  - Variables: SATFIN, CONFINAN, CONLEGIS, PARTYLR, NORTHWHO, UNIFI, KRPROUD

- **data/kgss_benchmarks_2023.json**: Ground truth distributions
  - Structure: `{"year": 2023, "analyzable_variables": [...], "distributions": {...}}`
  - Distributions stored as percentage maps: `{"SATFIN": {"1": 5.2, "2": 23.4, ...}}`

## Environment Variables

Required in `.env`:

```bash
# OpenAI (GPT experiments)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.2  # or gpt-3.5-turbo

# CLOVA Studio (Korean LLM experiments)
CLOVA_STUDIO_API_KEY=your_token
CLOVA_STUDIO_MODEL=HCX-007
CLOVA_STUDIO_BASE_URL=https://clovastudio.stream.ntruss.com

# Optional: NCP API Gateway (if using NCP services)
NCP_ACCESS_KEY_ID=...
NCP_SECRET_KEY=...
```

## Important Implementation Notes

### Response Parsing and Default Values

Both GPT and CLOVA experiments use `parse_first_int()` to extract integers from model responses. If parsing fails after retries, a **default value** (median of valid scale) is assigned. This is a methodological choice documented in the paper and affects distributional alignment results.

### Reproducibility

- All experiments use `seed=42` for persona generation
- Single sampling run per experiment (acknowledged limitation in paper)
- Sample size: n=100 per variable (choice requires justification per reviewer comments)

### Metric Interpretation Priority

Per reviewer feedback, when reporting results:
1. **Primary metric**: Jensen-Shannon divergence (0 to ~0.693)
2. **Secondary metrics**: Chi-square test, KS test
3. KS test results should be interpreted cautiously for ordinal data

### Async API Patterns

- `02_run_main_experiment.py`: Uses `asyncio.Semaphore` for rate limiting OpenAI calls
- `06_clova_experiment.py`: Synchronous CLOVA API calls (no official async client)
- Both implement retry logic with exponential backoff

## Output Structure

```
outputs/
├── personas/
│   └── personas_100.json              # 100 stratified personas
└── runs/
    └── <timestamp>/
        ├── persona_responses.csv      # Raw LLM responses
        ├── metrics.json               # JS divergence, KS test, chi-square
        ├── run_config.json            # Experiment parameters
        └── summary.md                 # Human-readable report

results/
├── gpt52_experiment/                  # GPT-5.2 results
├── gpt35turbo_experiment/             # GPT-3.5 baseline
└── clova_experiment/                  # CLOVA HCX-007 results
```

## Development Notes

### Code Organization

- All Python scripts use UTF-8 encoding for Korean text
- Korean comments and docstrings are intentional (primary research language)
- CSV files use `utf-8-sig` encoding for Excel compatibility
- Scripts in [code/](code/) directory add parent to sys.path for imports
- The [ms-word/](ms-word/) directory contains Node.js dependencies for Word doc generation (not core to experiments)

### Working with the Codebase

- **Always run scripts from repository root**: `python code/script_name.py`
- **Import structure**: Core utilities in [ss_utils.py](ss_utils.py), CLOVA client in [clova_client.py](clova_client.py)
- **Personas must exist before running experiments**: Run [code/01_generate_personas.py](code/01_generate_personas.py) first
- **Default sample size is 100**: Use `--n-samples` to adjust (must not exceed 100)
- **Temperature defaults to 0.7**: Optimal value found in Experiment 2
- **CLOVA thinking defaults to "medium"**: Balance between speed and reasoning quality

### Troubleshooting

**ModuleNotFoundError: No module named 'ss_utils'**
- Ensure you're running scripts from repository root, not from [code/](code/) directory
- Example: `python code/02_run_main_experiment.py` (correct) not `cd code && python 02_run_main_experiment.py`

**API Authentication Errors**
- Verify `.env` file exists in repository root (copy from `.env.example`)
- Check API keys are set: `OPENAI_API_KEY` and `CLOVA_STUDIO_API_KEY`
- CLOVA Studio API key format: Bearer token (no prefix needed in .env)
- OpenAI API key format: `sk-...`

**Missing Personas File**
- Generate personas first: `python code/01_generate_personas.py --seed 42 --out outputs/personas/personas_100.json`
- Default path: [outputs/personas/personas_100.json](outputs/personas/personas_100.json)

**Statistical Analysis Fails**
- Ensure experiment results exist in [outputs/runs/](outputs/runs/) or [results/](results/)
- CSV file must contain all required columns: `persona_id`, `variable`, `response`
- Verify benchmark distributions exist in [data/kgss_benchmarks_2023.json](data/kgss_benchmarks_2023.json)

### Testing Changes

When modifying core utilities:

```bash
# Test JS divergence calculation
python -c "from ss_utils import js_divergence; assert abs(js_divergence([0.5, 0.5], [0.5, 0.5])) < 1e-10"

# Test response parsing
python -c "from ss_utils import parse_first_int; assert parse_first_int('답: 3') == 3"

# Test benchmark loading
python -c "from ss_utils import load_benchmark; b = load_benchmark(); assert len(b.analyzable_variables) == 7"
```
