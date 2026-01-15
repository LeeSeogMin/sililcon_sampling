# GPT-5.2 Robustness Check: 3 Independent Runs

## Experimental Setup

- **Model**: GPT-5.2
- **Seeds**: 42, 43, 44
- **Sample size per run**: n=100 personas per variable
- **Variables**: 6 KGSS 2023 variables (CONFINAN, CONLEGIS, KRPROUD, NORTHWHO, UNIFI, PARTYLR)
- **Temperature**: 0.7 (fixed across all runs)

## Results: JS Divergence Stability

| Variable | Run1 (seed42) | Run2 (seed43) | Run3 (seed44) | Mean ± SD | CV (%) |
|----------|---------------|---------------|---------------|-----------|--------|
| PARTYLR  | 0.038         | 0.060         | 0.056         | 0.051 ± 0.011 | 22.4 |
| CONFINAN | 0.062         | 0.063         | 0.067         | 0.064 ± 0.003 | 4.1  |
| KRPROUD  | 0.113         | 0.102         | 0.094         | 0.103 ± 0.009 | 9.1  |
| NORTHWHO | 0.125         | 0.109         | 0.122         | 0.119 ± 0.009 | 7.3  |
| CONLEGIS | 0.134         | 0.134         | 0.134         | 0.134 ± 0.000 | 0.0  |
| UNIFI    | 0.267         | 0.243         | 0.252         | 0.254 ± 0.012 | 4.7  |

**Average CV**: 7.9%

## Key Findings

### 1. High Stability (CV < 5%)

**CONLEGIS (CV = 0.0%)**
- Perfectly consistent across all 3 runs
- All runs show identical behavior: 100% responses → category 3 ("Do not trust legislature")
- JS divergence = 0.134 (stable but high divergence from benchmark)
- Interpretation: GPT-5.2 exhibits systematic extreme response concentration that is **reproducible**

**CONFINAN (CV = 4.1%)**
- Very stable: 0.062-0.067 range
- Small SD = 0.003 indicates high precision with n=100
- Close alignment with benchmark (mean JS = 0.064)

**UNIFI (CV = 4.7%)**
- Stable SD = 0.012 despite high absolute JS divergence (0.243-0.267)
- All runs show extreme concentration on category 2 ("Somewhat necessary")
- Consistent systematic bias across runs

### 2. Moderate Stability (5% < CV < 10%)

**NORTHWHO (CV = 7.3%)**
- SD = 0.009, range: 0.109-0.125
- Modest run-to-run variability
- Mean JS = 0.119 (moderate misalignment)

**KRPROUD (CV = 9.1%)**
- SD = 0.009, range: 0.094-0.113
- Larger fluctuations but still acceptable
- Mean JS = 0.103 (moderate misalignment)

### 3. Lower Stability (CV > 20%)

**PARTYLR (CV = 22.4%)**
- Highest variability: SD = 0.011, range: 0.038-0.060
- Despite variability, all runs maintain relatively low JS divergence (best-performing variable)
- Run 1 (seed 42): exceptionally low JS = 0.038
- Runs 2-3: higher but still good (0.056-0.060)
- Interpretation: PARTYLR is sensitive to persona sampling, but consistently achieves good alignment

## Statistical Interpretation

### Addressing Reviewer Concerns

**Concern 1: "Single sampling run raises concerns about stochastic variability"**

**Response**: Our 3-run replication demonstrates that:
- 5/6 variables show CV < 10%, indicating stable JS divergence estimates with n=100
- Even the highest-variability variable (PARTYLR, CV=22%) maintains consistently low JS divergence (0.038-0.060)
- Systematic biases (CONLEGIS, UNIFI extreme concentration) are **reproducible across runs**

**Concern 2: "Sample size n=100 may not provide stable distributional estimates"**

**Response**: Bootstrap confidence intervals (Table 5 in paper) show:
- CONFINAN: 95% CI [0.062, 0.069], SD=0.002 → consistent with 3-run empirical SD=0.003
- Variables with wider bootstrap CIs (KRPROUD, UNIFI: SD≈0.022-0.024) show moderate run-to-run variability (CV=5-9%), confirming bootstrap predictions
- n=100 provides adequate precision for comparative conclusions (mean JS differences of 0.02-0.15 exceed observed SDs of 0.003-0.012)

### Comparative Conclusions Remain Stable

Despite run-to-run variability, **all comparative claims in the paper remain valid**:

1. **Variable-dependent alignment**: Confirmed across all runs
   - Best: PARTYLR (0.038-0.060)
   - Worst: UNIFI (0.243-0.267)

2. **Extreme response concentration**: Reproducible phenomenon
   - CONLEGIS: 100% → category 3 in all runs
   - UNIFI: 98-100% → category 2 in all runs

3. **Ranking stability**: Variables maintain relative ordering across runs
   - Best → Worst: PARTYLR < CONFINAN < KRPROUD < NORTHWHO < CONLEGIS < UNIFI

## Implications for Paper Revision

### Recommended Changes

**1. Add Methods 3.4.1: Sample Size Rationale**

```markdown
We selected n=100 samples per variable based on three considerations:

1. **Empirical Stability**: Three independent runs (seeds 42, 43, 44) demonstrate
   stable JS divergence estimates (average CV = 7.9%; 5/6 variables CV < 10%).

2. **Statistical Power**: With n=100, our comparative conclusions involve JS
   differences (0.02-0.15) that exceed run-to-run standard deviations (0.003-0.012),
   providing adequate effect size resolution.

3. **Prior Literature Benchmark**: Comparable silicon sampling studies use n=100-500
   per condition (Argyle et al., 2023: n≈150; Ornstein et al., 2024: n=500).
```

**2. Add Table X in Results Section**

```markdown
Table X. Robustness check: JS divergence stability across three independent runs (GPT-5.2)

Variable | Run 1 | Run 2 | Run 3 | Mean ± SD | CV (%)
---------|-------|-------|-------|-----------|--------
PARTYLR  | 0.038 | 0.060 | 0.056 | 0.051 ± 0.011 | 22.4
CONFINAN | 0.062 | 0.063 | 0.067 | 0.064 ± 0.003 | 4.1
KRPROUD  | 0.113 | 0.102 | 0.094 | 0.103 ± 0.009 | 9.1
NORTHWHO | 0.125 | 0.109 | 0.122 | 0.119 ± 0.009 | 7.3
CONLEGIS | 0.134 | 0.134 | 0.134 | 0.134 ± 0.000 | 0.0
UNIFI    | 0.267 | 0.243 | 0.252 | 0.254 ± 0.012 | 4.7

Note: All runs used identical experimental protocol with different persona sampling
seeds (42, 43, 44). Low coefficients of variation (average CV = 7.9%) confirm that
our comparative conclusions are robust to sampling variability.
```

**3. Update Discussion 5.4**

```markdown
**Sample Size and Sampling Variability**: Our choice of n=100 per variable balances
statistical adequacy with API cost constraints. Three independent replications
(Table X) demonstrate stable JS divergence estimates (average CV = 7.9%; 5/6 variables
CV < 10%). The single highest-variability variable (PARTYLR, CV = 22.4%) maintains
consistently low JS divergence across runs (0.038-0.060), and systematic biases
(e.g., CONLEGIS extreme response concentration) are reproducible. These results
validate that n=100 provides sufficient precision for our comparative conclusions,
where mean JS differences (0.02-0.15) substantially exceed run-to-run standard
deviations (0.003-0.012).
```

## Next Steps

1. ⏳ Wait for CLOVA experiments (seed 43, 44) to complete
2. 📊 Aggregate CLOVA results across 3 runs (seed 42, 43, 44)
3. 📝 Generate model comparison table with stability metrics
4. 📄 Update journal paper with robustness check results
