#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate all CLOVA results from seeds 42-46 and create robustness comparison table
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from ss_utils import load_benchmark

def load_clova_results(seed):
    """Load CLOVA results for a seed from aggregated JSON"""
    result_file = f'results/clova_experiment_seed{seed}/clova_results.json'
    if not os.path.exists(result_file):
        print(f"⚠️  File not found: {result_file}")
        return {}
    
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_gpt_results(seed):
    """Load GPT-5.2 results for a seed"""
    result_file = f'results/gpt52_experiment_seed{seed}/metrics.json'
    if not os.path.exists(result_file):
        print(f"⚠️  File not found: {result_file}")
        return {}
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Extract JS divergence from metrics
        results = {}
        if 'metrics' in data:
            for var, metrics in data['metrics'].items():
                results[var] = metrics.get('js_divergence', None)
        return results

def aggregate_all_results():
    """Aggregate all results from 5 seeds"""
    seeds = [42, 43, 44, 45, 46]
    variables = ['CONFINAN', 'CONLEGIS', 'KRPROUD', 'NORTHWHO', 'PARTYLR', 'UNIFI']
    
    # Store results by variable
    clova_by_var = defaultdict(list)
    gpt_by_var = defaultdict(list)
    
    print("\n" + "="*80)
    print("AGGREGATING 5-SEED RESULTS")
    print("="*80)
    
    # Load all CLOVA results
    print("\n📊 CLOVA HCX-007 Results:")
    print("-" * 80)
    for seed in seeds:
        results = load_clova_results(seed)
        if results:
            print(f"\nSeed {seed}:")
            for var in variables:
                if var in results:
                    js_val = results[var].get('js_divergence')
                    if js_val is not None:
                        clova_by_var[var].append(js_val)
                        print(f"  {var:10s}: {js_val:.4f}")
    
    # Load all GPT-5.2 results
    print("\n📊 GPT-5.2 Results:")
    print("-" * 80)
    for seed in seeds:
        results = load_gpt_results(seed)
        if results:
            print(f"\nSeed {seed}:")
            for var in variables:
                if var in results and results[var] is not None:
                    js_val = results[var]
                    gpt_by_var[var].append(js_val)
                    print(f"  {var:10s}: {js_val:.4f}")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    stats = {}
    for var in variables:
        if clova_by_var[var]:
            clova_vals = np.array(clova_by_var[var])
            gpt_vals = np.array(gpt_by_var[var]) if gpt_by_var[var] else None
            
            stats[var] = {
                'clova_mean': float(np.mean(clova_vals)),
                'clova_std': float(np.std(clova_vals, ddof=1)) if len(clova_vals) > 1 else 0.0,
                'clova_n': len(clova_vals),
                'gpt_mean': float(np.mean(gpt_vals)) if gpt_vals is not None else None,
                'gpt_std': float(np.std(gpt_vals, ddof=1)) if gpt_vals is not None and len(gpt_vals) > 1 else None,
                'gpt_n': len(gpt_vals) if gpt_vals is not None else 0,
            }
    
    # Print table
    print("\n" + "="*80)
    print("ROBUSTNESS CHECK: JS DIVERGENCE STABILITY ACROSS 5 INDEPENDENT RUNS")
    print("="*80)
    print(f"\n{'Variable':<10} | {'CLOVA (n=5)':<20} | {'GPT-5.2 (n=5)':<20} | {'Better':<12}")
    print("-" * 75)
    
    clova_wins = 0
    gpt_wins = 0
    
    for var in variables:
        if var in stats:
            s = stats[var]
            clova_str = f"{s['clova_mean']:.4f} ± {s['clova_std']:.4f}"
            if s['gpt_mean'] is not None:
                gpt_str = f"{s['gpt_mean']:.4f} ± {s['gpt_std']:.4f}"
                better = "CLOVA" if s['clova_mean'] < s['gpt_mean'] else "GPT-5.2" if s['gpt_mean'] < s['clova_mean'] else "TIE"
                if better == "CLOVA":
                    clova_wins += 1
                elif better == "GPT-5.2":
                    gpt_wins += 1
            else:
                gpt_str = "N/A"
                better = "N/A"
            
            print(f"{var:<10} | {clova_str:<20} | {gpt_str:<20} | {better:<12}")
    
    # Summary
    avg_clova = np.mean([stats[v]['clova_mean'] for v in variables if v in stats])
    avg_gpt = np.mean([stats[v]['gpt_mean'] for v in variables if v in stats and stats[v]['gpt_mean'] is not None])
    
    print("-" * 75)
    print(f"{'AVERAGE':<10} | {avg_clova:.4f} {'':>13} | {avg_gpt:.4f} {'':>13} | CLOVA {clova_wins}/6")
    
    improvement = (avg_gpt - avg_clova) / avg_gpt * 100 if avg_gpt > 0 else 0
    print(f"\n🎯 CLOVA Improvement: {improvement:.1f}% (wins {clova_wins}/6 variables)")
    
    # Save aggregated results
    output = {
        'metadata': {
            'seeds': seeds,
            'variables': variables,
            'model_clova': 'HCX-007',
            'model_gpt': 'gpt-5.2'
        },
        'statistics': stats,
        'summary': {
            'clova_average': float(avg_clova),
            'gpt_average': float(avg_gpt),
            'clova_wins': int(clova_wins),
            'gpt_wins': int(gpt_wins),
            'improvement_percent': float(improvement)
        }
    }
    
    output_file = 'results/aggregated_5seed_results.json'
    os.makedirs('results/aggregated', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Aggregated results saved: {output_file}")
    
    return stats

if __name__ == '__main__':
    aggregate_all_results()
