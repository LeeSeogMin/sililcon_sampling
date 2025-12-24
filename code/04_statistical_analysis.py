"""
Silicon Sampling 통계 분석 - 재현성 검증

목적: LLM 시뮬레이션 데이터와 KGSS 실제 데이터 비교
방법: KS test, Chi-square test, Jensen-Shannon divergence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple
import json

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


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
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"시뮬레이션 데이터 로드: {len(df)}행")
    return df


def load_kgss_data(filepath: str) -> pd.DataFrame:
    """
    KGSS 실제 데이터 로드

    Args:
        filepath: CSV 파일 경로

    Returns:
        pd.DataFrame: KGSS 데이터
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"KGSS 데이터 로드: {len(df)}행")
    return df


# ============================================================================
# 2. 분포 비교 통계
# ============================================================================

def calculate_distribution(data: pd.Series, bins: List[int]) -> np.ndarray:
    """
    응답 분포 계산 (정규화)

    Args:
        data: 응답 데이터
        bins: 유효한 응답 범위

    Returns:
        np.ndarray: 정규화된 분포
    """
    counts = data.value_counts().reindex(bins, fill_value=0)
    distribution = counts / counts.sum()
    return distribution.values


def kolmogorov_smirnov_test(sim_data: pd.Series, real_data: pd.Series,
                            bins: List[int]) -> Dict:
    """
    Kolmogorov-Smirnov 검정

    Args:
        sim_data: 시뮬레이션 응답
        real_data: 실제 응답
        bins: 유효한 응답 범위

    Returns:
        Dict: KS 통계량, p-value
    """
    sim_dist = calculate_distribution(sim_data, bins)
    real_dist = calculate_distribution(real_data, bins)

    # KS test (누적 분포 비교)
    statistic, p_value = stats.ks_2samp(sim_data.dropna(), real_data.dropna())

    return {
        "statistic": statistic,
        "p_value": p_value,
        "interpretation": "유의하지 않음 (유사)" if p_value > 0.05 else "유의함 (차이 있음)"
    }


def chi_square_test(sim_data: pd.Series, real_data: pd.Series,
                   bins: List[int]) -> Dict:
    """
    Chi-square 검정 (범주형 변수)

    Args:
        sim_data: 시뮬레이션 응답
        real_data: 실제 응답
        bins: 유효한 응답 범위

    Returns:
        Dict: Chi-square 통계량, p-value
    """
    sim_counts = sim_data.value_counts().reindex(bins, fill_value=0).values
    real_counts = real_data.value_counts().reindex(bins, fill_value=0).values

    # 기대 빈도가 5 미만인 셀이 20% 이상이면 경고
    expected = (sim_counts + real_counts) / 2
    if (expected < 5).sum() / len(expected) > 0.2:
        print("  [경고] 기대 빈도가 5 미만인 셀이 20% 이상입니다.")

    statistic, p_value = stats.chisquare(sim_counts, real_counts)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "interpretation": "유의하지 않음 (유사)" if p_value > 0.05 else "유의함 (차이 있음)"
    }


def jensen_shannon_divergence(sim_data: pd.Series, real_data: pd.Series,
                              bins: List[int]) -> float:
    """
    Jensen-Shannon Divergence 계산 (0에 가까울수록 유사)

    Args:
        sim_data: 시뮬레이션 응답
        real_data: 실제 응답
        bins: 유효한 응답 범위

    Returns:
        float: JS divergence (0~1)
    """
    sim_dist = calculate_distribution(sim_data, bins)
    real_dist = calculate_distribution(real_data, bins)

    # Jensen-Shannon divergence
    js_div = jensenshannon(sim_dist, real_dist)

    return {
        "divergence": js_div,
        "interpretation": "매우 유사" if js_div < 0.1 else ("유사" if js_div < 0.3 else "차이 있음")
    }


# ============================================================================
# 3. 변수별 종합 분석
# ============================================================================

def analyze_variable(variable: str, sim_df: pd.DataFrame, real_df: pd.DataFrame,
                     bins: List[int]) -> Dict:
    """
    단일 변수에 대한 종합 분석

    Args:
        variable: 변수명
        sim_df: 시뮬레이션 데이터
        real_df: KGSS 데이터
        bins: 유효한 응답 범위

    Returns:
        Dict: 분석 결과
    """
    print(f"\n분석 중: {variable}")
    print("-" * 40)

    sim_data = sim_df[variable]
    real_data = real_df[variable]

    # 결측치 제거
    sim_data = sim_data.dropna()
    real_data = real_data.dropna()

    # 기술통계
    sim_mean = sim_data.mean()
    real_mean = real_data.mean()
    sim_std = sim_data.std()
    real_std = real_data.std()

    print(f"  시뮬레이션: 평균={sim_mean:.2f}, 표준편차={sim_std:.2f}, N={len(sim_data)}")
    print(f"  실제 데이터: 평균={real_mean:.2f}, 표준편차={real_std:.2f}, N={len(real_data)}")

    # 통계 검정
    ks_result = kolmogorov_smirnov_test(sim_data, real_data, bins)
    chi_result = chi_square_test(sim_data, real_data, bins)
    js_result = jensen_shannon_divergence(sim_data, real_data, bins)

    print(f"  KS test: statistic={ks_result['statistic']:.3f}, p={ks_result['p_value']:.3f} → {ks_result['interpretation']}")
    print(f"  Chi-square: statistic={chi_result['statistic']:.3f}, p={chi_result['p_value']:.3f} → {chi_result['interpretation']}")
    print(f"  JS divergence: {js_result['divergence']:.3f} → {js_result['interpretation']}")

    return {
        "variable": variable,
        "descriptive": {
            "sim_mean": sim_mean,
            "sim_std": sim_std,
            "sim_n": len(sim_data),
            "real_mean": real_mean,
            "real_std": real_std,
            "real_n": len(real_data)
        },
        "ks_test": ks_result,
        "chi_square": chi_result,
        "js_divergence": js_result
    }


def analyze_all_variables(sim_df: pd.DataFrame, real_df: pd.DataFrame,
                         variable_bins: Dict[str, List[int]]) -> pd.DataFrame:
    """
    모든 변수에 대한 종합 분석

    Args:
        sim_df: 시뮬레이션 데이터
        real_df: KGSS 데이터
        variable_bins: 변수별 유효 응답 범위

    Returns:
        pd.DataFrame: 분석 결과 요약
    """
    results = []

    for variable, bins in variable_bins.items():
        if variable not in sim_df.columns or variable not in real_df.columns:
            print(f"\n[건너뜀] {variable}: 데이터에 존재하지 않음")
            continue

        result = analyze_variable(variable, sim_df, real_df, bins)
        results.append(result)

    # 결과 DataFrame으로 변환
    summary_data = []
    for r in results:
        summary_data.append({
            "변수": r["variable"],
            "시뮬_평균": r["descriptive"]["sim_mean"],
            "실제_평균": r["descriptive"]["real_mean"],
            "평균_차이": abs(r["descriptive"]["sim_mean"] - r["descriptive"]["real_mean"]),
            "KS_p값": r["ks_test"]["p_value"],
            "Chi2_p값": r["chi_square"]["p_value"],
            "JS_divergence": r["js_divergence"]["divergence"],
            "KS_결론": r["ks_test"]["interpretation"],
            "JS_결론": r["js_divergence"]["interpretation"]
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df, results


# ============================================================================
# 4. 시각화
# ============================================================================

def plot_distribution_comparison(variable: str, sim_df: pd.DataFrame,
                                 real_df: pd.DataFrame, bins: List[int],
                                 save_path: str = None):
    """
    시뮬레이션 vs 실제 데이터 분포 비교 시각화

    Args:
        variable: 변수명
        sim_df: 시뮬레이션 데이터
        real_df: KGSS 데이터
        bins: 유효한 응답 범위
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 시뮬레이션 분포
    sim_counts = sim_df[variable].value_counts().reindex(bins, fill_value=0)
    sim_pct = (sim_counts / sim_counts.sum() * 100)

    axes[0].bar(bins, sim_pct, color='steelblue', alpha=0.7)
    axes[0].set_title(f'{variable} - 시뮬레이션 분포', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('응답 범주', fontsize=12)
    axes[0].set_ylabel('비율 (%)', fontsize=12)
    axes[0].set_ylim(0, max(sim_pct.max(), 50))
    axes[0].grid(axis='y', alpha=0.3)

    # 실제 데이터 분포
    real_counts = real_df[variable].value_counts().reindex(bins, fill_value=0)
    real_pct = (real_counts / real_counts.sum() * 100)

    axes[1].bar(bins, real_pct, color='coral', alpha=0.7)
    axes[1].set_title(f'{variable} - KGSS 실제 분포', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('응답 범주', fontsize=12)
    axes[1].set_ylabel('비율 (%)', fontsize=12)
    axes[1].set_ylim(0, max(real_pct.max(), 50))
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  그래프 저장: {save_path}")

    plt.show()


def plot_summary_heatmap(summary_df: pd.DataFrame, save_path: str = None):
    """
    변수별 유사성 히트맵

    Args:
        summary_df: 분석 결과 요약 DataFrame
        save_path: 저장 경로
    """
    # p-value와 JS divergence 기반 유사성 점수 계산
    summary_df['유사성_점수'] = (
        (summary_df['KS_p값'] > 0.05).astype(int) +
        (summary_df['Chi2_p값'] > 0.05).astype(int) +
        (summary_df['JS_divergence'] < 0.2).astype(int)
    )

    fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.5 + 2))

    # 히트맵 데이터 준비
    heatmap_data = summary_df[['변수', 'KS_p값', 'Chi2_p값', 'JS_divergence']].set_index('변수')

    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.05, vmin=0, vmax=1, cbar_kws={'label': '값'},
                linewidths=0.5, ax=ax)

    ax.set_title('변수별 재현성 평가 (녹색=유사, 빨강=차이)', fontsize=14, fontweight='bold')
    ax.set_xlabel('변수', fontsize=12)
    ax.set_ylabel('지표', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  히트맵 저장: {save_path}")

    plt.show()


# ============================================================================
# 5. 메인 실행
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Silicon Sampling 통계 분석 시작")
    print("="*60 + "\n")

    # 변수별 유효 범위 정의
    VARIABLE_BINS = {
        "SATFIN": [1, 2, 3, 4, 5],
        "TRUST": [1, 2, 3],
        "FAIR": [1, 2, 3],
        "HELPFUL": [1, 2, 3],
        "CONFINAN": [1, 2, 3],
        "CONLEGIS": [1, 2, 3],
        "PARTYLR": list(range(0, 11)),
        "NORTHWHO": [1, 2, 3, 4],
        "UNIFI": [1, 2, 3, 4],
        "KRPROUD": [1, 2, 3, 4]
    }

    # 데이터 로드 (경로는 실제 환경에 맞게 수정)
    try:
        sim_df = load_simulation_data("research_agent/output/simulation_results/final_results.csv")
        real_df = load_kgss_data("research_agent/data/kgss_2023.csv")

        # 종합 분석
        print("\n" + "="*60)
        print("변수별 분석 시작")
        print("="*60)

        summary_df, detailed_results = analyze_all_variables(sim_df, real_df, VARIABLE_BINS)

        # 결과 저장
        output_dir = "research_agent/output/statistical_results"
        import os
        os.makedirs(output_dir, exist_ok=True)

        summary_df.to_csv(f"{output_dir}/analysis_summary.csv", index=False, encoding='utf-8-sig')

        with open(f"{output_dir}/detailed_results.json", "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장 완료: {output_dir}/")

        # 요약 통계 출력
        print("\n" + "="*60)
        print("분석 결과 요약")
        print("="*60)
        print(summary_df.to_string())

        # 전체 재현성 평가
        n_similar_ks = (summary_df['KS_p값'] > 0.05).sum()
        n_similar_js = (summary_df['JS_divergence'] < 0.2).sum()
        total_vars = len(summary_df)

        print(f"\n재현성 평가:")
        print(f"  KS test 통과 (p>0.05): {n_similar_ks}/{total_vars} ({n_similar_ks/total_vars*100:.1f}%)")
        print(f"  JS divergence 통과 (<0.2): {n_similar_js}/{total_vars} ({n_similar_js/total_vars*100:.1f}%)")

        # 시각화
        print("\n시각화 생성 중...")
        for variable in ["SATFIN", "TRUST", "PARTYLR", "NORTHWHO"]:
            plot_distribution_comparison(
                variable, sim_df, real_df, VARIABLE_BINS[variable],
                save_path=f"{output_dir}/{variable}_comparison.png"
            )

        plot_summary_heatmap(summary_df, save_path=f"{output_dir}/similarity_heatmap.png")

    except FileNotFoundError as e:
        print(f"\n[오류] 파일을 찾을 수 없습니다: {e}")
        print("\n다음 파일이 필요합니다:")
        print("  1. research_agent/output/simulation_results/final_results.csv (LLM 시뮬레이션 결과)")
        print("  2. research_agent/data/kgss_2023.csv (KGSS 실제 데이터)")
        print("\n시뮬레이션을 먼저 실행하세요: python research_agent/silicon_sampling_simulation.py")
