#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap CI analysis for robustness check
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from ss_utils import load_benchmark

def load_clova_results(seed):
    """Load CLOVA results for a seed"""
    # Seed42는 다른 경로 사용
    if seed == 42:
        # Seed42는 폴더별 구조 - 각 변수별 JSON 파일에서 로드
        results = {}
        base = 'results/clova_experiment'
        for var in ['CONFINAN', 'CONLEGIS', 'PARTYLR', 'NORTHWHO', 'UNIFI', 'KRPROUD']:
            json_path = f'{base}/{var}/clova_results.json'
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'results' in data and len(data['results']) > 0:
                    r = data['results'][0]
                    results[var] = r.get('js_divergence')
        return results

    # Seed43은 partial 파일 사용
    if seed == 43:
        result_file = f'results/clova_experiment_seed{seed}/clova_results_partial.json'
    else:
        result_file = f'results/clova_experiment_seed{seed}/clova_results.json'

    if not os.path.exists(result_file):
        print(f"⚠️  File not found: {result_file}")
        return {}

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        results = {}
        for var_result in data.get('results', []):
            var = var_result.get('variable')
            if var:
                results[var] = var_result.get('js_divergence')
        return results

def load_gpt_results(seed):
    """Load GPT-5.2 results for a seed"""
    # Seed42는 다른 경로 사용
    if seed == 42:
        result_file = 'results/gpt52_experiment/metrics.json'
    else:
        result_file = f'results/gpt52_experiment_seed{seed}/metrics.json'

    if not os.path.exists(result_file):
        print(f"⚠️  File not found: {result_file}")
        return {}

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        results = {}
        # 직접 변수별 구조 (js_divergence_ln 키 사용)
        for var, metrics in data.items():
            if isinstance(metrics, dict):
                js_val = metrics.get('js_divergence') or metrics.get('js_divergence_ln')
                if js_val is not None:
                    results[var] = js_val
        return results

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Calculate bootstrap confidence interval"""
    if len(data) < 2:
        return np.mean(data), np.nan, np.nan

    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    mean = np.mean(data)
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

    return mean, lower, upper

def aggregate_with_bootstrap():
    """Aggregate results with bootstrap CI"""
    seeds = [42, 43, 44, 45, 46]
    variables = ['CONFINAN', 'CONLEGIS', 'KRPROUD', 'NORTHWHO', 'PARTYLR', 'UNIFI']

    clova_by_var = defaultdict(list)
    gpt_by_var = defaultdict(list)

    print("\n" + "="*80)
    print("LOADING 5-SEED RESULTS FOR BOOTSTRAP CI ANALYSIS")
    print("="*80)

    # Load all CLOVA results
    print("\n📊 CLOVA HCX-007 Results:")
    print("-" * 80)
    clova_loaded = 0
    for seed in seeds:
        results = load_clova_results(seed)
        if results:
            clova_loaded += 1
            print(f"\nSeed {seed}:")
            for var in variables:
                if var in results and results[var] is not None:
                    clova_by_var[var].append(results[var])
                    print(f"  {var:10s}: {results[var]:.4f}")

    # Load all GPT-5.2 results
    print("\n📊 GPT-5.2 Results:")
    print("-" * 80)
    gpt_loaded = 0
    for seed in seeds:
        results = load_gpt_results(seed)
        if results:
            gpt_loaded += 1
            print(f"\nSeed {seed}:")
            for var in variables:
                if var in results and results[var] is not None:
                    gpt_by_var[var].append(results[var])
                    print(f"  {var:10s}: {results[var]:.4f}")

    print(f"\n✅ Loaded {clova_loaded} CLOVA results and {gpt_loaded} GPT results")

    # Calculate statistics and bootstrap CI
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS (95% CI)")
    print("="*80)

    stats_dict = {}
    for var in variables:
        if clova_by_var[var]:
            clova_vals = np.array(clova_by_var[var])
            gpt_vals = np.array(gpt_by_var[var]) if gpt_by_var[var] else None

            clova_mean, clova_lower, clova_upper = bootstrap_ci(clova_vals)
            gpt_mean, gpt_lower, gpt_upper = bootstrap_ci(gpt_vals) if gpt_vals is not None else (None, None, None)

            stats_dict[var] = {
                'clova': {
                    'mean': float(clova_mean),
                    'lower': float(clova_lower) if not np.isnan(clova_lower) else None,
                    'upper': float(clova_upper) if not np.isnan(clova_upper) else None,
                    'n': len(clova_vals)
                },
                'gpt': {
                    'mean': float(gpt_mean) if gpt_mean is not None else None,
                    'lower': float(gpt_lower) if gpt_lower is not None and not np.isnan(gpt_lower) else None,
                    'upper': float(gpt_upper) if gpt_upper is not None and not np.isnan(gpt_upper) else None,
                    'n': len(gpt_vals) if gpt_vals is not None else 0
                }
            }

    # Print table
    print(f"\n{'Variable':<10} | {'CLOVA (95% CI)':<35} | {'GPT-5.2 (95% CI)':<35} | {'Winner':<10}")
    print("-" * 95)

    clova_wins = 0
    gpt_wins = 0

    for var in variables:
        if var in stats_dict:
            s = stats_dict[var]
            c = s['clova']
            g = s['gpt']

            clova_str = f"{c['mean']:.4f} [{c['lower']:.4f}, {c['upper']:.4f}]"

            if g['mean'] is not None:
                gpt_str = f"{g['mean']:.4f} [{g['lower']:.4f}, {g['upper']:.4f}]"
                winner = "CLOVA" if c['mean'] < g['mean'] else "GPT-5.2"
                if winner == "CLOVA":
                    clova_wins += 1
                else:
                    gpt_wins += 1
            else:
                gpt_str = "N/A"
                winner = "N/A"

            print(f"{var:<10} | {clova_str:<35} | {gpt_str:<35} | {winner:<10}")

    # Summary
    clova_means = [stats_dict[v]['clova']['mean'] for v in variables if v in stats_dict and stats_dict[v]['clova']['mean'] is not None]
    gpt_means = [stats_dict[v]['gpt']['mean'] for v in variables if v in stats_dict and stats_dict[v]['gpt']['mean'] is not None]

    if clova_means and gpt_means:
        avg_clova = np.mean(clova_means)
        avg_gpt = np.mean(gpt_means)
        improvement = (avg_gpt - avg_clova) / avg_gpt * 100

        print("-" * 95)
        print(f"{'AVERAGE':<10} | {avg_clova:.4f} {'':>25} | {avg_gpt:.4f} {'':>25} | CLOVA {clova_wins}/6")
        print(f"\n🎯 CLOVA Improvement: {improvement:.1f}% (wins {clova_wins}/6 variables)")

    # Save results
    output = {
        'metadata': {
            'seeds': seeds,
            'variables': variables,
            'model_clova': 'HCX-007',
            'model_gpt': 'gpt-5.2',
            'ci_level': 95,
            'bootstrap_iterations': 10000
        },
        'statistics': stats_dict,
        'summary': {
            'clova_average': float(np.mean(clova_means)) if clova_means else None,
            'gpt_average': float(np.mean(gpt_means)) if gpt_means else None,
            'clova_wins': int(clova_wins),
            'gpt_wins': int(gpt_wins),
            'improvement_percent': float(improvement) if clova_means and gpt_means else None
        }
    }

    output_file = 'results/aggregated_5seed_bootstrap_results.json'
    os.makedirs('results/aggregated', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Bootstrap results saved: {output_file}")

    return stats_dict

if __name__ == '__main__':
    aggregate_with_bootstrap()
