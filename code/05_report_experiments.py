"""
Run experiments described in papers/final_report.md using local configs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ss_utils import (
    Benchmark,
    format_scale_labels,
    js_divergence,
    load_benchmark,
    parse_first_int,
    read_json,
    utc_timestamp,
    write_json,
)


@dataclass(frozen=True)
class PromptStrategy:
    name: str
    extra_line: str | None = None


BASELINE = PromptStrategy("baseline")
PERSONA_REINFORCED = PromptStrategy("persona_reinforced", "당신의 특성을 고려하여 솔직하게 답변하세요.")
EXTREME_ALLOWED = PromptStrategy("extreme_allowed", "극단적인 의견도 괜찮습니다. 솔직하게 답변하세요.")


def load_personas(path: str, n_samples: int | None = None) -> List[Dict[str, Any]]:
    personas = read_json(path)
    if not isinstance(personas, list) or not personas:
        raise ValueError(f"Invalid personas file: {path}")
    if n_samples is not None:
        personas = personas[:n_samples]
    return personas


def load_variable_defs(path: str) -> Dict[str, Dict[str, Any]]:
    raw = read_json(path)
    variables = raw.get("variables", {})
    if not isinstance(variables, dict) or not variables:
        raise ValueError(f"Invalid variables config: {path}")
    return variables


def mode_with_tiebreak(values: Iterable[int]) -> int:
    counts = Counter(values)
    if not counts:
        raise ValueError("Empty values for mode")
    max_count = max(counts.values())
    candidates = sorted([v for v, c in counts.items() if c == max_count])
    return candidates[len(candidates) // 2]


def parse_last_int(text: str) -> int | None:
    matches = list(re.finditer(r"-?\d+", text.strip()))
    if not matches:
        return None
    try:
        return int(matches[-1].group(0))
    except ValueError:
        return None


def build_prompt(
    persona: Dict[str, Any],
    question: str,
    scale_text: str,
    valid_values: List[int],
    strategy: PromptStrategy,
) -> str:
    values_text = ", ".join(map(str, valid_values))
    lines = [
        "당신은 다음과 같은 특성을 가진 한국인입니다:",
        f"- 나이: {persona.get('age_group')}",
        f"- 성별: {persona.get('gender')}",
        f"- 학력: {persona.get('education')}",
        f"- 거주 지역: {persona.get('region')}",
        f"- 직업: {persona.get('occupation')}",
        "",
    ]
    if strategy.extra_line:
        lines.append(strategy.extra_line)
        lines.append("")
    lines.append("이 사람의 입장에서 다음 질문에 답해주세요.")
    lines.append(f"질문: {question}")
    if scale_text:
        lines.append(f"응답 범주: {scale_text}")
    else:
        lines.append(f"응답 범주: {values_text}")
    lines.append("숫자만 답하세요. 설명은 필요 없습니다.")
    return "\n".join(lines)


def build_cot_prompt(
    persona: Dict[str, Any],
    question: str,
    scale_text: str,
    valid_values: List[int],
) -> str:
    values_text = ", ".join(map(str, valid_values))
    range_text = (
        f"{min(valid_values)}-{max(valid_values)}"
        if valid_values == list(range(min(valid_values), max(valid_values) + 1))
        else values_text
    )
    return f"""당신은 다음과 같은 특성을 가진 한국인입니다:
- 나이: {persona.get('age_group')}
- 성별: {persona.get('gender')}
- 학력: {persona.get('education')}
- 거주 지역: {persona.get('region')}
- 직업: {persona.get('occupation')}

다음 질문에 답변하기 전에, 먼저 당신의 생각을 단계별로 설명하세요:

1. 당신의 개인적 경험이나 주변 사례를 떠올려보세요
2. 해당 기관에 대한 신뢰를 결정하는 요인들을 생각해보세요
3. 당신의 특성(연령, 교육수준, 직업 등)이 어떻게 영향을 미치는지 고려하세요
4. 위의 생각을 종합하여 {range_text} 중 하나의 숫자로 답변하세요

질문: {question}
응답 범주: {scale_text if scale_text else values_text}

[생각 과정]

[최종 답변]
({range_text} 중 하나의 숫자만):"""


async def fetch_response(
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    prompt: str,
    valid_values: List[int],
    semaphore: asyncio.Semaphore,
    parse_last: bool = False,
    max_retries: int = 3,
) -> int:
    default_value = sorted(valid_values)[len(valid_values) // 2]
    parse_fn = parse_last_int if parse_last else parse_first_int

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
                    params["max_completion_tokens"] = 200
                else:
                    params["max_tokens"] = 200

                response = await client.chat.completions.create(**params)

            content = (response.choices[0].message.content or "").strip()
            parsed = parse_fn(content)
            if parsed is None or parsed not in set(valid_values):
                return default_value
            return parsed

        except Exception:
            if attempt >= max_retries:
                return default_value
            await asyncio.sleep(2 ** (attempt - 1))

    return default_value


def compute_js(benchmark: Benchmark, variable: str, responses: List[int]) -> float:
    categories = benchmark.categories(variable)
    counts = {c: 0 for c in categories}
    for r in responses:
        if r in counts:
            counts[r] += 1

    total = max(1, sum(counts.values()))
    obs_prob = np.array([counts[c] / total for c in categories], dtype=float)
    bench_prob = benchmark.distribution_prob(variable)
    return float(js_divergence(obs_prob, bench_prob))


async def run_variable(
    client: AsyncOpenAI,
    personas: List[Dict[str, Any]],
    variable: str,
    var_def: Dict[str, Any],
    model: str,
    temperature: float,
    benchmark: Benchmark,
    semaphore: asyncio.Semaphore,
    repeats: int = 1,
    strategy: PromptStrategy = BASELINE,
    use_cot: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    valid_values = [int(v) for v in var_def["valid_responses"]]
    scale_text = format_scale_labels(var_def.get("scale_labels"))
    tasks: List[Tuple[str, asyncio.Task[int]]] = []

    for persona in personas:
        prompt = (
            build_cot_prompt(persona, str(var_def["question"]), scale_text, valid_values)
            if use_cot
            else build_prompt(persona, str(var_def["question"]), scale_text, valid_values, strategy)
        )
        for _ in range(repeats):
            task = asyncio.create_task(
                fetch_response(
                    client,
                    model,
                    temperature,
                    prompt,
                    valid_values,
                    semaphore,
                    parse_last=use_cot,
                )
            )
            tasks.append((persona["persona_id"], task))

    responses_by_persona: Dict[str, List[int]] = {p["persona_id"]: [] for p in personas}
    for persona_id, task in tasks:
        responses_by_persona[persona_id].append(await task)

    rows: List[Dict[str, Any]] = []
    final_responses: List[int] = []
    for persona in personas:
        pid = persona["persona_id"]
        responses = responses_by_persona[pid]
        final_answer = mode_with_tiebreak(responses)
        final_responses.append(final_answer)
        rows.append(
            {
                "persona_id": pid,
                "age_group": persona.get("age_group"),
                "gender": persona.get("gender"),
                "education": persona.get("education"),
                "region": persona.get("region"),
                "occupation": persona.get("occupation"),
                "variable": variable,
                "responses": ",".join(map(str, responses)),
                "response": int(final_answer),
            }
        )

    js_value = compute_js(benchmark, variable, final_responses)
    return rows, js_value


async def run_experiment(
    out_dir: str,
    personas: List[Dict[str, Any]],
    benchmark: Benchmark,
    variable_defs: Dict[str, Dict[str, Any]],
    variables: List[str],
    model: str,
    temperature: float,
    repeats: int,
    strategy: PromptStrategy = BASELINE,
    use_cot: bool = False,
    concurrency: int = 20,
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    semaphore = asyncio.Semaphore(concurrency)

    rows: List[Dict[str, Any]] = []
    metrics: Dict[str, float] = {}

    for variable in variables:
        if variable not in variable_defs:
            raise ValueError(f"Missing variable definition for {variable}")
        var_rows, js_value = await run_variable(
            client,
            personas,
            variable,
            variable_defs[variable],
            model,
            temperature,
            benchmark,
            semaphore,
            repeats=repeats,
            strategy=strategy,
            use_cot=use_cot,
        )
        rows.extend(var_rows)
        metrics[variable] = js_value

    df_long = pd.DataFrame(rows)
    df_wide = df_long.pivot_table(
        index=["persona_id", "age_group", "gender", "education", "region", "occupation"],
        columns="variable",
        values="response",
        aggfunc="first",
    ).reset_index()
    df_wide.columns.name = None

    df_long.to_csv(os.path.join(out_dir, "persona_responses_long.csv"), index=False, encoding="utf-8-sig")
    df_wide.to_csv(os.path.join(out_dir, "persona_responses.csv"), index=False, encoding="utf-8-sig")

    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    return metrics


def ensure_api_key() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Create a .env file based on .env.example.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run report experiments")
    subparsers = parser.add_subparsers(dest="task", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--personas", type=str, default="outputs/personas/personas_100.json")
    common.add_argument("--benchmark", type=str, default="results/kgss_benchmarks_2023.json")
    common.add_argument("--variables-config", type=str, default="config/kgss_variables_2023_report.json")
    common.add_argument("--model", type=str, default="gpt-4o-mini")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--n-samples", type=int, default=100)
    common.add_argument("--concurrency", type=int, default=20)
    common.add_argument("--out-dir", type=str, default=None)

    subparsers.add_parser("main", parents=[common], help="Main experiment (n=100, 3 repeats)")
    subparsers.add_parser("prompt", parents=[common], help="Prompt engineering experiment (n=5)")
    subparsers.add_parser("temp_opt", parents=[common], help="Task 1 temperature optimization (n=20)")
    subparsers.add_parser("temp_opt_re", parents=[common], help="Task 1 reexperiment (n=100, 3 reps)")
    subparsers.add_parser("phase3", parents=[common], help="Phase 3 rerun (n=100)")
    subparsers.add_parser("cot", parents=[common], help="Task 4 CoT experiment")

    args = parser.parse_args()
    ensure_api_key()

    if args.task == "main":
        personas = load_personas(args.personas, n_samples=args.n_samples)
        benchmark = load_benchmark(args.benchmark)
        variable_defs = load_variable_defs(args.variables_config)
        variables = benchmark.analyzable_variables
        out_dir = args.out_dir or os.path.join("outputs", "report", "main_experiment", utc_timestamp())

        metrics = asyncio.run(
            run_experiment(
                out_dir,
                personas,
                benchmark,
                variable_defs,
                variables,
                model=args.model,
                temperature=args.temperature,
                repeats=3,
                concurrency=args.concurrency,
            )
        )
        write_json(os.path.join(out_dir, "summary.json"), metrics)
        print(f"✅ Main experiment saved: {out_dir}")
        return 0

    if args.task == "prompt":
        personas = load_personas(args.personas, n_samples=5)
        benchmark = load_benchmark(args.benchmark)
        variable_defs = load_variable_defs(args.variables_config)
        variables = benchmark.analyzable_variables
        base_dir = args.out_dir or os.path.join("outputs", "report", "prompt_experiment", utc_timestamp())

        results: List[Dict[str, Any]] = []
        summary: Dict[str, Dict[str, float]] = {"mean": {}, "std": {}, "min": {}, "max": {}}

        for strategy in [BASELINE, PERSONA_REINFORCED, EXTREME_ALLOWED]:
            out_dir = os.path.join(base_dir, strategy.name)
            metrics = asyncio.run(
                run_experiment(
                    out_dir,
                    personas,
                    benchmark,
                    variable_defs,
                    variables,
                    model=args.model,
                    temperature=args.temperature,
                    repeats=1,
                    strategy=strategy,
                    concurrency=args.concurrency,
                )
            )
            for variable, js_value in metrics.items():
                results.append(
                    {
                        "variable": variable,
                        "strategy": strategy.name,
                        "js_divergence": js_value,
                    }
                )

        df = pd.DataFrame(results)
        for strategy in df["strategy"].unique():
            values = df[df["strategy"] == strategy]["js_divergence"].values
            summary["mean"][strategy] = float(np.mean(values))
            summary["std"][strategy] = float(np.std(values))
            summary["min"][strategy] = float(np.min(values))
            summary["max"][strategy] = float(np.max(values))

        write_json(os.path.join(base_dir, "summary.json"), {"summary": summary, "results": results})
        print(f"✅ Prompt experiment saved: {base_dir}")
        return 0

    if args.task in {"temp_opt", "temp_opt_re"}:
        n_samples = 20 if args.task == "temp_opt" else 100
        repeats = 3
        temps = [0.3, 0.5, 0.7, 0.9, 1.1]
        variables = ["SATFIN", "PARTYLR", "CONFINAN", "CONLEGIS"]

        personas = load_personas(args.personas, n_samples=n_samples)
        benchmark = load_benchmark(args.benchmark)
        variable_defs = load_variable_defs(args.variables_config)

        base_dir = args.out_dir or os.path.join(
            "outputs",
            "report",
            "temperature_optimization" if args.task == "temp_opt" else "temperature_optimization_re",
            utc_timestamp(),
        )
        os.makedirs(base_dir, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for temperature in temps:
            metrics = asyncio.run(
                run_experiment(
                    os.path.join(base_dir, f"t_{temperature}"),
                    personas,
                    benchmark,
                    variable_defs,
                    variables,
                    model=args.model,
                    temperature=temperature,
                    repeats=repeats,
                    concurrency=args.concurrency,
                )
            )
            for variable, js_value in metrics.items():
                rows.append(
                    {
                        "variable": variable,
                        "temperature": temperature,
                        "js_divergence": js_value,
                        "n_samples": n_samples,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(base_dir, "final_results.csv"), index=False, encoding="utf-8-sig")
        write_json(os.path.join(base_dir, "full_results.json"), rows)
        print(f"✅ Temperature optimization saved: {base_dir}")
        return 0

    if args.task == "phase3":
        personas = load_personas(args.personas, n_samples=100)
        benchmark = load_benchmark(args.benchmark)
        variable_defs = load_variable_defs(args.variables_config)
        variables = ["SATFIN", "PARTYLR", "CONFINAN", "CONLEGIS"]

        out_dir = args.out_dir or os.path.join("outputs", "report", "phase3_rerun", utc_timestamp())
        metrics = asyncio.run(
            run_experiment(
                out_dir,
                personas,
                benchmark,
                variable_defs,
                variables,
                model=args.model,
                temperature=args.temperature,
                repeats=1,
                concurrency=args.concurrency,
            )
        )
        write_json(os.path.join(out_dir, "summary.json"), metrics)
        print(f"✅ Phase 3 rerun saved: {out_dir}")
        return 0

    if args.task == "cot":
        personas = load_personas(args.personas, n_samples=100)
        benchmark = load_benchmark(args.benchmark)
        variable_defs = load_variable_defs(args.variables_config)
        variables = ["CONFINAN", "CONLEGIS"]

        base_dir = args.out_dir or os.path.join("outputs", "report", "cot_experiment", utc_timestamp())
        os.makedirs(base_dir, exist_ok=True)

        for label, use_cot in [("baseline", False), ("cot", True)]:
            out_dir = os.path.join(base_dir, label)
            metrics = asyncio.run(
                run_experiment(
                    out_dir,
                    personas,
                    benchmark,
                    variable_defs,
                    variables,
                    model=args.model,
                    temperature=args.temperature,
                    repeats=1,
                    use_cot=use_cot,
                    concurrency=args.concurrency,
                )
            )
            write_json(os.path.join(out_dir, "summary.json"), metrics)

        print(f"✅ CoT experiment saved: {base_dir}")
        return 0

    raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    raise SystemExit(main())
