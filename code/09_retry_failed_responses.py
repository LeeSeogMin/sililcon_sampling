"""
실패한 CLOVA 응답 재시도 스크립트

실패 케이스(used_default=True)를 찾아 해당 persona만 다시 실행하고
성공하면 기존 결과에 병합합니다.

Usage:
    python code/09_retry_failed_responses.py --seed 44 --variable CONFINAN --batch-size 10
    python code/09_retry_failed_responses.py --seed 44 --all --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv

from clova_client import clova_chat_completions_v3
from ss_utils import (
    Benchmark,
    format_scale_labels,
    js_divergence,
    load_benchmark,
    parse_first_int,
    read_json,
    write_json,
)


def load_personas_by_ids(path: str, persona_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """특정 persona_id들만 로드"""
    personas = read_json(path)
    return {p['persona_id']: p for p in personas if p['persona_id'] in persona_ids}


def get_failed_persona_ids(results_path: str, variable: str) -> List[str]:
    """실패한 persona_id 목록 추출"""
    data = read_json(results_path)

    for result in data.get('results', []):
        if result.get('variable') == variable:
            raw_responses = result.get('raw_responses', [])
            failed_ids = [
                r['persona_id'] for r in raw_responses
                if r.get('used_default', False)
            ]
            return failed_ids

    return []


def build_prompt(persona: Dict[str, Any], question: str, scale_text: str, valid_values: List[int]) -> str:
    """프롬프트 생성"""
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
    """CLOVA API 호출"""
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


def retry_variable(
    seed: int,
    variable: str,
    batch_size: int = 10,
    temperature: float = 0.7,
    thinking: str = "medium",
    delay: float = 0.5,
) -> Dict[str, Any]:
    """특정 변수의 실패 케이스 재시도"""

    # 경로 설정
    results_path = f'results/clova_experiment_seed{seed}/clova_results_partial.json'
    personas_path = f'outputs/personas/personas_100_seed{seed}.json'
    variables_config = read_json('config/kgss_variables_2023.json')

    if not os.path.exists(results_path):
        return {'error': f'Results file not found: {results_path}'}

    # 실패한 persona_id 추출
    failed_ids = get_failed_persona_ids(results_path, variable)

    if not failed_ids:
        return {'variable': variable, 'retried': 0, 'succeeded': 0, 'message': 'No failed cases'}

    # batch_size만큼만 선택 (0이면 전체)
    if batch_size <= 0:
        target_ids = failed_ids
    else:
        target_ids = failed_ids[:batch_size]

    # persona 로드
    personas = load_personas_by_ids(personas_path, target_ids)

    # 변수 정의
    var_def = variables_config['variables'].get(variable)
    if not var_def:
        return {'error': f'Variable not found: {variable}'}

    question = str(var_def['question'])
    valid_values = [int(v) for v in var_def['valid_responses']]
    scale_text = format_scale_labels(var_def.get('scale_labels'))

    print(f"\n[{variable}] 재시도: {len(target_ids)}개 (총 실패: {len(failed_ids)}개)")

    # 재시도 실행
    new_responses = []
    succeeded = 0

    for i, persona_id in enumerate(target_ids):
        persona = personas.get(persona_id)
        if not persona:
            print(f"  {persona_id}: persona not found")
            continue

        prompt = build_prompt(persona, question, scale_text, valid_values)
        result = fetch_clova_response(prompt, valid_values, temperature, thinking)

        new_response = {
            'persona_id': persona_id,
            **result,
        }
        new_responses.append(new_response)

        if not result['used_default']:
            succeeded += 1
            print(f"  [{i+1}/{len(target_ids)}] {persona_id}: ✅ {result['answer']}")
        else:
            print(f"  [{i+1}/{len(target_ids)}] {persona_id}: ❌ (default)")

        time.sleep(delay)

    # 결과 병합
    data = read_json(results_path)

    for result in data.get('results', []):
        if result.get('variable') == variable:
            raw_responses = result.get('raw_responses', [])

            # 새 응답으로 교체
            new_responses_dict = {r['persona_id']: r for r in new_responses}
            updated_responses = []
            updated_answers = []

            for r in raw_responses:
                pid = r['persona_id']
                if pid in new_responses_dict:
                    updated_responses.append(new_responses_dict[pid])
                    updated_answers.append(new_responses_dict[pid]['answer'])
                else:
                    updated_responses.append(r)
                    updated_answers.append(r['answer'])

            result['raw_responses'] = updated_responses
            result['responses'] = updated_answers

            # JS divergence 재계산 (유효 응답만)
            benchmark = load_benchmark('data/kgss_benchmarks_2023.json')
            valid_responses = [r['answer'] for r in updated_responses if not r.get('used_default', False)]

            if valid_responses:
                categories = sorted(benchmark.categories(variable))
                counts = {c: valid_responses.count(c) for c in categories}
                total = len(valid_responses)
                obs_prob = np.array([counts.get(c, 0) / total for c in categories], dtype=float)
                bench_prob = benchmark.distribution_prob(variable)
                result['js_divergence'] = float(js_divergence(obs_prob, bench_prob))
                result['valid_n'] = total

            break

    # 저장
    write_json(results_path, data)

    return {
        'variable': variable,
        'retried': len(new_responses),
        'succeeded': succeeded,
        'remaining_failed': len(failed_ids) - succeeded,
    }


def main():
    parser = argparse.ArgumentParser(description='Retry failed CLOVA responses')
    parser.add_argument('--seed', type=int, required=True, help='Seed number (44, 45, 46)')
    parser.add_argument('--variable', type=str, default=None, help='Variable to retry')
    parser.add_argument('--all', action='store_true', help='Retry all variables')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size per variable (0 = all failed)')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--thinking', type=str, default='medium')
    parser.add_argument('--delay', type=float, default=0.5)

    args = parser.parse_args()

    load_dotenv()

    variables = ['CONFINAN', 'CONLEGIS', 'PARTYLR', 'NORTHWHO', 'UNIFI', 'KRPROUD']

    if args.all:
        target_vars = variables
    elif args.variable:
        target_vars = [args.variable]
    else:
        print("Error: Specify --variable or --all")
        return 1

    print(f"=== Seed{args.seed} 실패 케이스 재시도 ===")
    print(f"변수: {target_vars}")
    print(f"배치 크기: {args.batch_size}")

    for var in target_vars:
        result = retry_variable(
            seed=args.seed,
            variable=var,
            batch_size=args.batch_size,
            temperature=args.temperature,
            thinking=args.thinking,
            delay=args.delay,
        )

        if 'error' in result:
            print(f"\n❌ {var}: {result['error']}")
        else:
            print(f"\n✅ {var}: {result['retried']}개 재시도, {result['succeeded']}개 성공, {result.get('remaining_failed', 'N/A')}개 남음")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
