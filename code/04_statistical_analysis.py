"""
Silicon Sampling 통계 분석 - 재현성 검증 (벤치마크 분포 기반)

목적: LLM 시뮬레이션 데이터와 KGSS 벤치마크 분포 비교
방법: KS test, Chi-square test, Jensen-Shannon divergence
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ss_utils import load_benchmark, utc_timestamp


N_VALID_2023 = {
    "SATFIN": 1230,
    "CONFINAN": 1230,
    "CONLEGIS": 1230,
    "PARTYLR": 1230,
    "NORTHWHO": 1230,
    "UNIFI": 1230,
    "KRPROUD": 1230,
}


# ============================================================================
# 1. 데이터 로드
# ============================================================================

def load_simulation_data(filepath: str) -> pd.DataFrame:
    """
    LLM 시뮬레이션 결과 로드

    Args:
        filepath: CSV 파일 경로

    Returns:
        pd.DataFrame: 시뮬레이션 데이터
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    print(f"시뮬레이션 데이터 로드: {len(df)}행")
    return df


# ============================================================================
# 2. 분포 비교 통계
# ============================================================================

def counts_from_percentages(pct_map: Dict[int, float], categories: List[int], total: int) -> np.ndarray:
    """퍼센트 분포를 정수 카운트로 변환 (합계는 total로 맞춤)."""
    raw = np.array([(pct_map[c] / 100.0) * total for c in categories], dtype=float)
    floors = np.floor(raw).astype(int)
    remainder = int(total - floors.sum())
    if remainder > 0:
        frac = raw - floors
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            floors[idx] += 1
    return floors


def distribution_from_counts(counts: np.ndarray) -> np.ndarray:
    total = max(1, int(counts.sum()))
    return counts.astype(float) / total


def kolmogorov_smirnov_test(sim_counts: np.ndarray, real_counts: np.ndarray, categories: List[int]) -> Dict:
    sim_sample = np.repeat(categories, sim_counts)
    real_sample = np.repeat(categories, real_counts)

    statistic, p_value = stats.ks_2samp(sim_sample, real_sample)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "interpretation": "유의하지 않음 (유사)" if p_value > 0.05 else "유의함 (차이 있음)",
    }


def chi_square_test(sim_counts: np.ndarray, real_counts: np.ndarray) -> Dict:
    table = np.vstack([sim_counts, real_counts])
    statistic, p_value, _, _ = stats.chi2_contingency(table)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "interpretation": "유의하지 않음 (유사)" if p_value > 0.05 else "유의함 (차이 있음)",
    }


def jensen_shannon_divergence(sim_counts: np.ndarray, real_counts: np.ndarray) -> Dict:
    sim_dist = distribution_from_counts(sim_counts)
    real_dist = distribution_from_counts(real_counts)

    js_div = jensenshannon(sim_dist, real_dist, base=2)

    return {
        "divergence": float(js_div),
        "interpretation": "매우 유사" if js_div < 0.1 else ("유사" if js_div < 0.3 else "차이 있음"),
    }


# ============================================================================
# 3. 변수별 종합 분석
# ============================================================================

def describe_real_distribution(categories: List[int], probs: np.ndarray) -> Tuple[float, float]:
    mean = float(np.sum(np.array(categories, dtype=float) * probs))
    var = float(np.sum(((np.array(categories, dtype=float) - mean) ** 2) * probs))
    return mean, float(np.sqrt(var))


def analyze_variable(
    variable: str,
    sim_df: pd.DataFrame,
    categories: List[int],
    benchmark_pct: Dict[int, float],
    n_valid: int,
) -> Dict:
    print(f"\n분석 중: {variable}")
    print("-" * 40)

    sim_data = sim_df[variable].dropna().astype(int)
    sim_counts = sim_data.value_counts().reindex(categories, fill_value=0).values
    real_counts = counts_from_percentages(benchmark_pct, categories, n_valid)

    sim_mean = float(sim_data.mean())
    sim_std = float(sim_data.std())
    real_probs = distribution_from_counts(real_counts)
    real_mean, real_std = describe_real_distribution(categories, real_probs)

    print(f"  시뮬레이션: 평균={sim_mean:.2f}, 표준편차={sim_std:.2f}, N={len(sim_data)}")
    print(f"  실제 분포: 평균={real_mean:.2f}, 표준편차={real_std:.2f}, N={n_valid}")

    ks_result = kolmogorov_smirnov_test(sim_counts, real_counts, categories)
    chi_result = chi_square_test(sim_counts, real_counts)
    js_result = jensen_shannon_divergence(sim_counts, real_counts)

    print(f"  KS test: statistic={ks_result['statistic']:.3f}, p={ks_result['p_value']:.3f} → {ks_result['interpretation']}")
    print(f"  Chi-square: statistic={chi_result['statistic']:.3f}, p={chi_result['p_value']:.3f} → {chi_result['interpretation']}")
    print(f"  JS divergence: {js_result['divergence']:.3f} → {js_result['interpretation']}")

    return {
        "variable": variable,
        "descriptive": {
            "sim_mean": sim_mean,
            "sim_std": sim_std,
            "sim_n": int(len(sim_data)),
            "real_mean": real_mean,
            "real_std": real_std,
            "real_n": int(n_valid),
        },
        "ks_test": ks_result,
        "chi_square": chi_result,
        "js_divergence": js_result,
    }


def analyze_all_variables(
    sim_df: pd.DataFrame,
    benchmark,
    n_valid_map: Dict[str, int],
    variables: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    results = []

    target_vars = variables or benchmark.analyzable_variables
    for variable in target_vars:
        if variable not in sim_df.columns:
            print(f"\n[건너뜀] {variable}: 시뮬레이션 결과에 없음")
            continue
        if variable not in benchmark.distributions_pct:
            print(f"\n[건너뜀] {variable}: 벤치마크 분포 없음")
            continue

        categories = benchmark.categories(variable)
        benchmark_pct = benchmark.distributions_pct[variable]
        n_valid = n_valid_map.get(variable, len(sim_df))

        result = analyze_variable(variable, sim_df, categories, benchmark_pct, n_valid)
        results.append(result)

    summary_data = []
    for r in results:
        summary_data.append(
            {
                "변수": r["variable"],
                "시뮬_평균": r["descriptive"]["sim_mean"],
                "실제_평균": r["descriptive"]["real_mean"],
                "평균_차이": abs(r["descriptive"]["sim_mean"] - r["descriptive"]["real_mean"]),
                "KS_p값": r["ks_test"]["p_value"],
                "Chi2_p값": r["chi_square"]["p_value"],
                "JS_divergence": r["js_divergence"]["divergence"],
                "KS_결론": r["ks_test"]["interpretation"],
                "JS_결론": r["js_divergence"]["interpretation"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    return summary_df, results


# ============================================================================
# 4. 메인 실행
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Silicon Sampling 통계 분석 (벤치마크 분포 기반)")
    parser.add_argument("--sim-csv", required=True, help="시뮬레이션 결과 CSV 경로")
    parser.add_argument("--benchmark", default="data/kgss_benchmarks_2023.json")
    parser.add_argument("--out-dir", default=None, help="결과 저장 디렉토리")
    parser.add_argument("--variables", nargs="*", default=None, help="분석할 변수 목록")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Silicon Sampling 통계 분석 시작")
    print("=" * 60 + "\n")

    sim_df = load_simulation_data(args.sim_csv)
    benchmark = load_benchmark(args.benchmark)

    summary_df, detailed_results = analyze_all_variables(sim_df, benchmark, N_VALID_2023, args.variables)

    out_dir = args.out_dir or os.path.join("outputs", "analysis", utc_timestamp())
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "analysis_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    details_path = os.path.join(out_dir, "detailed_results.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {out_dir}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
