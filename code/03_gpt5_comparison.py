"""
Model comparison runner (public reproducibility version)

Compares two models against the same personas/variables and reports metric deltas.

Outputs (default: outputs/comparisons/<timestamp>/):
  - comparison.json
  - summary.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ss_utils import (
    Benchmark,
    format_scale_labels,
    js_distance,
    js_divergence,
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


async def fetch_response(
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    prompt: str,
    valid_values: List[int],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> int:
    default_value = sorted(valid_values)[len(valid_values) // 2]

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                params: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "당신은 설문조사 응답자입니다. 질문에 숫자로만 답변하세요."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }

                if "gpt-5" in model.lower():
                    params["max_completion_tokens"] = 10
                else:
                    params["max_tokens"] = 10

                response = await client.chat.completions.create(**params)

            content = (response.choices[0].message.content or "").strip()
            parsed = parse_first_int(content)
            if parsed is None or parsed not in set(valid_values):
                return default_value
            return parsed

        except Exception:
            if attempt >= max_retries:
                return default_value
            await asyncio.sleep(2 ** (attempt - 1))

    return default_value


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
        "js_divergence_ln": js_divergence(obs_prob, bench_prob),
        "js_distance_ln": js_distance(obs_prob, bench_prob),
    }


def load_responses_from_csv(csv_path: str, personas: List[Dict[str, Any]], variables: List[str]) -> Dict[str, List[int]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "persona_id" not in df.columns:
        raise ValueError(f"CSV must contain persona_id column: {csv_path}")

    df = df.set_index("persona_id")
    responses: Dict[str, List[int]] = {}
    for var in variables:
        if var not in df.columns:
            raise ValueError(f"CSV missing variable column {var}: {csv_path}")
        ordered = []
        for p in personas:
            ordered.append(int(df.loc[p["persona_id"], var]))
        responses[var] = ordered
    return responses


async def run(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Create a .env file based on .env.example.")

    benchmark = load_benchmark(args.benchmark)
    variable_defs = load_variable_defs(args.variables_config)

    variables = args.variables or benchmark.analyzable_variables
    variables = [v for v in variables if v in benchmark.distributions_pct]
    if not variables:
        raise ValueError("No variables to run (check benchmark / --variables).")

    personas = load_personas(args.personas, args.n_samples)

    out_dir = args.out_dir or os.path.join("outputs", "comparisons", utc_timestamp())
    os.makedirs(out_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(api_key=api_key)

    model_a_responses: Optional[Dict[str, List[int]]] = None
    if args.model_a_csv:
        model_a_responses = load_responses_from_csv(args.model_a_csv, personas, variables)

    model_b_responses: Optional[Dict[str, List[int]]] = None
    if args.model_b_csv:
        model_b_responses = load_responses_from_csv(args.model_b_csv, personas, variables)

    results: Dict[str, Any] = {
        "config": {
            "timestamp_utc": utc_timestamp(),
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "variables": variables,
            "personas": args.personas,
            "benchmark": args.benchmark,
            "variables_config": args.variables_config,
            "model_a": {"name": args.model_a, "csv": args.model_a_csv},
            "model_b": {"name": args.model_b, "csv": args.model_b_csv},
        },
        "variables": {},
    }

    for var in variables:
        if var not in variable_defs:
            raise ValueError(f"Missing variable definition for {var} in {args.variables_config}")

        valid_values = [int(v) for v in variable_defs[var]["valid_responses"]]
        question = str(variable_defs[var]["question"])
        scale_text = format_scale_labels(variable_defs[var].get("scale_labels"))

        # Model A
        if model_a_responses is None:
            tasks = []
            for persona in personas:
                prompt = build_prompt(persona, question, scale_text, valid_values)
                tasks.append(fetch_response(client, args.model_a, args.temperature, prompt, valid_values, semaphore))
            a_resps = await asyncio.gather(*tasks)
        else:
            a_resps = model_a_responses[var]

        # Model B
        if model_b_responses is None:
            tasks = []
            for persona in personas:
                prompt = build_prompt(persona, question, scale_text, valid_values)
                tasks.append(fetch_response(client, args.model_b, args.temperature, prompt, valid_values, semaphore))
            b_resps = await asyncio.gather(*tasks)
        else:
            b_resps = model_b_responses[var]

        a_metrics = compute_metrics(benchmark, var, a_resps)
        b_metrics = compute_metrics(benchmark, var, b_resps)
        delta = a_metrics["js_divergence_ln"] - b_metrics["js_divergence_ln"]

        results["variables"][var] = {
            "model_a": {"model": args.model_a, "metrics": a_metrics},
            "model_b": {"model": args.model_b, "metrics": b_metrics},
            "delta_js_divergence_ln": float(delta),
        }

    write_json(os.path.join(out_dir, "comparison.json"), results)

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Model Comparison Summary\n\n")
        f.write(f"- Model A: `{args.model_a}`\n")
        f.write(f"- Model B: `{args.model_b}`\n")
        f.write(f"- Temperature: `{args.temperature}`\n")
        f.write(f"- Samples: `{args.n_samples}`\n\n")
        f.write("| Variable | JS(A) | JS(B) | A-B |\n")
        f.write("|---|---:|---:|---:|\n")
        for var in variables:
            a = results["variables"][var]["model_a"]["metrics"]["js_divergence_ln"]
            b = results["variables"][var]["model_b"]["metrics"]["js_divergence_ln"]
            d = results["variables"][var]["delta_js_divergence_ln"]
            f.write(f"| {var} | {a:.4f} | {b:.4f} | {d:+.4f} |\n")

    print(f"✅ Saved: {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two models for Silicon Sampling")
    parser.add_argument("--personas", type=str, default="outputs/personas/personas_100.json")
    parser.add_argument("--benchmark", type=str, default="data/kgss_benchmarks_2023.json")
    parser.add_argument("--variables-config", dest="variables_config", type=str, default="config/kgss_variables_2023.json")
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")))
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--model-a", dest="model_a", type=str, default="gpt-4o-mini")
    parser.add_argument("--model-b", dest="model_b", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--model-a-csv", dest="model_a_csv", type=str, default=None, help="Reuse model A responses from CSV")
    parser.add_argument("--model-b-csv", dest="model_b_csv", type=str, default=None, help="Reuse model B responses from CSV")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

