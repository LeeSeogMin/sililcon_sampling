"""
Main experiment runner (public reproducibility version)

- Inputs:
  - Personas JSON (default: outputs/personas/personas_100.json)
  - KGSS benchmark distributions (data/kgss_benchmarks_2023.json)
  - Variable prompt definitions (config/kgss_variables_2023.json)

- Outputs (default: outputs/runs/<timestamp>/):
  - persona_responses.csv
  - metrics.json
  - run_config.json
  - summary.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List

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


def load_personas(path: str) -> List[Dict[str, Any]]:
    personas = read_json(path)
    if not isinstance(personas, list) or not personas:
        raise ValueError(f"Invalid personas file: {path}")
    return personas


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

    prompt = f"""당신은 다음과 같은 특성을 가진 한국인입니다:

- 연령: {persona.get('age_group')}
- 성별: {persona.get('gender')}
- 교육수준: {persona.get('education')}
- 거주지역: {persona.get('region')}
- 직업: {persona.get('occupation')}

{instruction}

질문: {question}
척도: {scale_text}

답변 (숫자만):"""
    return prompt


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
        "variable": variable,
        "n": int(total),
        "categories": categories,
        "observed_counts": counts,
        "observed_pct": {str(c): (counts[c] / total) * 100 for c in categories},
        "benchmark_pct": {str(c): benchmark.distributions_pct[variable][c] for c in categories},
        "js_divergence_ln": js_divergence(obs_prob, bench_prob),
        "js_distance_ln": js_distance(obs_prob, bench_prob),
    }


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

    personas = load_personas(args.personas)
    if args.n_samples > len(personas):
        raise ValueError(f"--n-samples {args.n_samples} exceeds personas count {len(personas)}")
    personas = personas[: args.n_samples]

    out_dir = args.out_dir or os.path.join("outputs", "runs", utc_timestamp())
    os.makedirs(out_dir, exist_ok=True)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}

    for variable in variables:
        if variable not in variable_defs:
            raise ValueError(f"Missing variable definition for {variable} in {args.variables_config}")

        var_def = variable_defs[variable]
        question = str(var_def["question"])
        valid_values = [int(v) for v in var_def["valid_responses"]]
        scale_text = format_scale_labels(var_def.get("scale_labels"))

        tasks = []
        for persona in personas:
            prompt = build_prompt(persona, question, scale_text, valid_values)
            tasks.append(fetch_response(client, args.model, args.temperature, prompt, valid_values, semaphore))

        responses = await asyncio.gather(*tasks)
        metrics[variable] = compute_metrics(benchmark, variable, responses)

        for persona, answer in zip(personas, responses):
            rows.append(
                {
                    "persona_id": persona.get("persona_id"),
                    "age_group": persona.get("age_group"),
                    "gender": persona.get("gender"),
                    "education": persona.get("education"),
                    "region": persona.get("region"),
                    "occupation": persona.get("occupation"),
                    "variable": variable,
                    "response": int(answer),
                }
            )

    df_long = pd.DataFrame(rows)
    df_wide = df_long.pivot_table(
        index=["persona_id", "age_group", "gender", "education", "region", "occupation"],
        columns="variable",
        values="response",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None

    csv_path = os.path.join(out_dir, "persona_responses.csv")
    df_wide.to_csv(csv_path, index=False, encoding="utf-8-sig")

    run_config = {
        "timestamp_utc": utc_timestamp(),
        "model": args.model,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "variables": variables,
        "paths": {
            "personas": args.personas,
            "benchmark": args.benchmark,
            "variables_config": args.variables_config,
            "out_dir": out_dir,
        },
        "concurrency": args.concurrency,
    }

    write_json(os.path.join(out_dir, "run_config.json"), run_config)
    write_json(os.path.join(out_dir, "metrics.json"), metrics)

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Silicon Sampling Run Summary\n\n")
        f.write(f"- Model: `{args.model}`\n")
        f.write(f"- Temperature: `{args.temperature}`\n")
        f.write(f"- Samples: `{args.n_samples}`\n")
        f.write(f"- Variables: `{', '.join(variables)}`\n\n")
        f.write("## Metrics\n\n")
        f.write("| Variable | JS divergence (ln) | JS distance (ln) |\n")
        f.write("|---|---:|---:|\n")
        for v in variables:
            f.write(f"| {v} | {metrics[v]['js_divergence_ln']:.4f} | {metrics[v]['js_distance_ln']:.4f} |\n")

    print(f"✅ Saved: {out_dir}")
    print(f"  - {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Silicon Sampling main experiment")
    parser.add_argument("--personas", type=str, default="outputs/personas/personas_100.json")
    parser.add_argument("--benchmark", type=str, default="data/kgss_benchmarks_2023.json")
    parser.add_argument("--variables-config", type=str, default="config/kgss_variables_2023.json")
    parser.add_argument("--variables", nargs="*", default=None, help="Subset of variables to run")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: outputs/runs/<timestamp>)")
    args = parser.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

