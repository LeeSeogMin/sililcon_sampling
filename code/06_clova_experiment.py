"""
CLOVA HCX-007 experiment runner for Korean LLM comparison.

Compares CLOVA's Korean-native LLM (HCX-007) against GPT models
for Korean survey simulation tasks.

Outputs (default: results/clova_experiment/):
  - clova_results.json: Raw experiment results
  - comparison_summary.json: GPT-5.2 vs CLOVA comparison
  - summary.md: Human-readable summary
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv

from clova_client import clova_chat_completions_v3
from ss_utils import (
    Benchmark,
    format_scale_labels,
    js_divergence,
    js_distance,
    load_benchmark,
    parse_first_int,
    read_json,
    utc_timestamp,
    write_json,
)


def load_personas(path: str, n_samples: int) -> List[Dict[str, Any]]:
    personas = read_json(path)
    if not isinstance(personas, list) or not personas:
        raise ValueError(f"Invalid personas file: {path}")
    if n_samples > len(personas):
        raise ValueError(f"--n-samples {n_samples} exceeds personas count {len(personas)}")
    return personas[:n_samples]


def load_variable_defs(path: str) -> Dict[str, Dict[str, Any]]:
    raw = read_json(path)
    variables = raw.get("variables", {})
    if not isinstance(variables, dict) or not variables:
        raise ValueError(f"Invalid variables config: {path}")
    return variables


def build_prompt(persona: Dict[str, Any], question: str, scale_text: str, valid_values: List[int]) -> str:
    valid_values_sorted = sorted(valid_values)
    if valid_values_sorted == list(range(valid_values_sorted[0], valid_values_sorted[-1] + 1)):
        instruction = f"다음 질문에 {valid_values_sorted[0]}-{valid_values_sorted[-1]} 중 하나의 숫자로만 답변하세요."
    else:
        instruction = f"다음 질문에 {', '.join(map(str, valid_values_sorted))} 중 하나의 숫자로만 답변하세요."

    return f"""당신은 다음과 같은 특성을 가진 한국인입니다:

- 연령: {persona.get('age_group')}
- 성별: {persona.get('gender')}
- 교육수준: {persona.get('education')}
- 거주지역: {persona.get('region')}
- 직업: {persona.get('occupation')}

{instruction}

질문: {question}
척도: {scale_text}

답변 (숫자만):"""


def fetch_clova_response(
    prompt: str,
    valid_values: List[int],
    temperature: float = 0.7,
    thinking: str | None = "medium",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Fetch response from CLOVA HCX-007.

    Args:
        prompt: Survey prompt
        valid_values: Valid response values
        temperature: Sampling temperature
        thinking: Thinking effort level ("short", "medium", "deep", or None)
        max_retries: Maximum retry attempts
    """
    default_value = sorted(valid_values)[len(valid_values) // 2]
    last_error: str | None = None
    last_content: str = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = clova_chat_completions_v3(
                messages=[
                    {"role": "system", "content": "당신은 설문조사 응답자입니다. 질문에 숫자로만 답변하세요."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                thinking=thinking,
            )

            # Extract content from CLOVA response format
            content = response.get("result", {}).get("message", {}).get("content", "").strip()
            last_content = content

            parsed = parse_first_int(content)
            if parsed is None or parsed not in set(valid_values):
                return {
                    "answer": default_value,
                    "raw": content,
                    "parsed": parsed,
                    "used_default": True,
                    "error": "invalid_response",
                    "attempts": attempt,
                }
            return {
                "answer": parsed,
                "raw": content,
                "parsed": parsed,
                "used_default": False,
                "error": None,
                "attempts": attempt,
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                return {
                    "answer": default_value,
                    "raw": last_content,
                    "parsed": None,
                    "used_default": True,
                    "error": last_error,
                    "attempts": attempt,
                }
            time.sleep(2 ** (attempt - 1))

    return {
        "answer": default_value,
        "raw": last_content,
        "parsed": None,
        "used_default": True,
        "error": last_error,
        "attempts": max_retries,
    }


def compute_metrics(benchmark: Benchmark, variable: str, responses: List[int]) -> Dict[str, Any]:
    categories = benchmark.categories(variable)
    counts = {c: 0 for c in categories}
    for r in responses:
        if r in counts:
            counts[r] += 1

    total = max(1, sum(counts.values()))
    obs_prob = np.array([counts[c] / total for c in categories], dtype=float)
    bench_prob = benchmark.distribution_prob(variable)

    return {
        "n": int(total),
        "categories": categories,
        "observed_counts": counts,
        "observed_pct": {str(c): (counts[c] / total) * 100 for c in categories},
        "benchmark_pct": {str(c): benchmark.distributions_pct[variable][c] for c in categories},
        "js_divergence": float(js_divergence(obs_prob, bench_prob)),
        "js_distance": float(js_distance(obs_prob, bench_prob)),
    }


def load_gpt5_results(path: str) -> Dict[str, Any]:
    """Load GPT-5.2 results for comparison."""
    if not os.path.exists(path):
        return {}

    data = read_json(path)
    results_by_var = {}
    for item in data.get("results", []):
        if item.get("model") == "gpt-5.2":
            var = item.get("variable")
            results_by_var[var] = {
                "js_divergence": item.get("js_divergence"),
                "distribution": item.get("distribution"),
                "responses": item.get("responses"),
            }
    return results_by_var


def run(args: argparse.Namespace) -> int:
    load_dotenv()

    benchmark = load_benchmark(args.benchmark)
    variable_defs = load_variable_defs(args.variables_config)

    variables = args.variables or benchmark.analyzable_variables
    variables = [v for v in variables if v in benchmark.distributions_pct]
    if not variables:
        raise ValueError("No variables to run (check benchmark / --variables).")

    personas = load_personas(args.personas, args.n_samples)

    out_dir = args.out_dir or "results/clova_experiment"
    os.makedirs(out_dir, exist_ok=True)

    # Load GPT-5.2 results for comparison
    gpt5_results = load_gpt5_results(args.gpt5_results)

    results: List[Dict[str, Any]] = []
    comparison: Dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "configuration": {
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "variables": variables,
            "clova_model": os.getenv("CLOVA_STUDIO_MODEL", "HCX-007"),
        },
        "variables": {},
    }

    total_calls = len(variables) * len(personas)
    completed = 0

    for variable in variables:
        if variable not in variable_defs:
            raise ValueError(f"Missing variable definition for {variable} in {args.variables_config}")

        var_def = variable_defs[variable]
        question = str(var_def["question"])
        valid_values = [int(v) for v in var_def["valid_responses"]]
        scale_text = format_scale_labels(var_def.get("scale_labels"))

        print(f"\n[{variable}] Running {len(personas)} samples...")
        responses: List[int] = []
        raw_responses: List[Dict[str, Any]] = []

        for i, persona in enumerate(personas):
            prompt = build_prompt(persona, question, scale_text, valid_values)
            result = fetch_clova_response(prompt, valid_values, args.temperature, args.thinking)
            responses.append(result["answer"])
            raw_responses.append({
                "persona_id": persona.get("persona_id"),
                **result,
            })

            completed += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(personas)} ({completed}/{total_calls} total)")

            # Rate limiting
            time.sleep(args.delay)

        metrics = compute_metrics(benchmark, variable, responses)

        results.append({
            "variable": variable,
            "model": "HCX-007",
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "distribution": metrics["observed_pct"],
            "js_divergence": metrics["js_divergence"],
            "responses": responses,
            "raw_responses": raw_responses,
        })

        # Comparison with GPT-5.2
        gpt5_var = gpt5_results.get(variable, {})
        gpt5_js = gpt5_var.get("js_divergence")
        clova_js = metrics["js_divergence"]

        comparison["variables"][variable] = {
            "clova_hcx007": {
                "js_divergence": clova_js,
                "distribution": metrics["observed_pct"],
            },
            "gpt_5_2": {
                "js_divergence": gpt5_js,
                "distribution": gpt5_var.get("distribution"),
            } if gpt5_js else None,
            "delta": float(gpt5_js - clova_js) if gpt5_js else None,
            "clova_better": clova_js < gpt5_js if gpt5_js else None,
        }

        print(f"  CLOVA JS: {clova_js:.4f}" + (f" | GPT-5.2 JS: {gpt5_js:.4f} | Delta: {gpt5_js - clova_js:+.4f}" if gpt5_js else ""))

        # Incremental save after each variable
        write_json(os.path.join(out_dir, "clova_results_partial.json"), {
            "timestamp": utc_timestamp(),
            "configuration": comparison["configuration"],
            "status": "in_progress",
            "completed_variables": [r["variable"] for r in results],
            "results": results,
        })
        print(f"  [Saved: {variable} → clova_results_partial.json]")

    # Save final results
    write_json(os.path.join(out_dir, "clova_results.json"), {
        "timestamp": utc_timestamp(),
        "configuration": comparison["configuration"],
        "results": results,
    })

    write_json(os.path.join(out_dir, "comparison_summary.json"), comparison)

    # Generate summary
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# CLOVA HCX-007 vs GPT-5.2 Comparison\n\n")
        f.write(f"- **Date**: {utc_timestamp()}\n")
        f.write(f"- **CLOVA Model**: {comparison['configuration']['clova_model']}\n")
        f.write(f"- **Thinking**: {args.thinking or 'off'}\n")
        f.write(f"- **Temperature**: {args.temperature}\n")
        f.write(f"- **Samples**: {args.n_samples}\n\n")
        f.write("## Results\n\n")
        f.write("| Variable | CLOVA HCX-007 JS | GPT-5.2 JS | Delta | Better |\n")
        f.write("|----------|-----------------|------------|-------|--------|\n")

        clova_wins = 0
        gpt_wins = 0
        total_delta = 0.0
        valid_comparisons = 0

        for var in variables:
            comp = comparison["variables"][var]
            clova_js = comp["clova_hcx007"]["js_divergence"]
            gpt_js = comp["gpt_5_2"]["js_divergence"] if comp["gpt_5_2"] else None
            delta = comp["delta"]
            better = comp["clova_better"]

            if gpt_js is not None:
                f.write(f"| {var} | {clova_js:.4f} | {gpt_js:.4f} | {delta:+.4f} | {'CLOVA' if better else 'GPT'} |\n")
                if better:
                    clova_wins += 1
                else:
                    gpt_wins += 1
                total_delta += delta
                valid_comparisons += 1
            else:
                f.write(f"| {var} | {clova_js:.4f} | N/A | N/A | N/A |\n")

        if valid_comparisons > 0:
            f.write(f"\n## Summary\n\n")
            f.write(f"- **CLOVA wins**: {clova_wins}/{valid_comparisons}\n")
            f.write(f"- **GPT-5.2 wins**: {gpt_wins}/{valid_comparisons}\n")
            f.write(f"- **Average delta (GPT - CLOVA)**: {total_delta / valid_comparisons:+.4f}\n")
            f.write(f"- **Interpretation**: Positive delta means CLOVA is better (lower JS divergence)\n")

    print(f"\n✅ Saved: {out_dir}")
    print(f"  - clova_results.json")
    print(f"  - comparison_summary.json")
    print(f"  - summary.md")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CLOVA HCX-007 experiment")
    parser.add_argument("--personas", type=str, default="results/personas/personas_100.json")
    parser.add_argument("--benchmark", type=str, default="data/kgss_benchmarks_2023.json")
    parser.add_argument("--variables-config", dest="variables_config", type=str, default="config/kgss_variables_2023.json")
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    parser.add_argument("--thinking", type=str, default="medium", choices=["short", "medium", "deep", "none"],
                        help="HCX-007 thinking effort level (default: medium)")
    parser.add_argument("--gpt5-results", type=str, default="results/gpt5_full_comparison/results_20251220_012830.json")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    # Convert "none" string to None for thinking parameter
    if args.thinking == "none":
        args.thinking = None

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
