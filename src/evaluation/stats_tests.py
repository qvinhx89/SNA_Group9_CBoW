"""
Statistical Tests Module (Enhanced)
===================================
Advanced statistical testing with multiple testing correction and effect sizes.

Includes:
- Mann-Whitney U with Benjamini-Hochberg correction
- Cliff's Delta
- Rank-biserial correlation
- Kruskal-Wallis with post-hoc Dunn test

CHANGE-7: Output columns per implementation plan:
- effect_size_r: rank-biserial correlation
- p_raw: raw p-value
- p_corrected_bh: Benjamini-Hochberg corrected p-value
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Tuple, Dict, Optional
import warnings


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cliff's Delta effect size for ordinal/ranked data.

    Cliff's Delta = (# x > y - # x < y) / (n1 * n2)

    Parameters
    ----------
    x : np.ndarray
        First group values
    y : np.ndarray
        Second group values

    Returns
    -------
    Tuple[float, str]
        (delta value, interpretation)

    References
    ----------
    Cliff (1993) "Dominance statistics: Ordinal analyses to answer ordinal questions"
    """
    n1, n2 = len(x), len(y)

    # Count dominance
    greater = 0
    less = 0

    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1

    delta = (greater - less) / (n1 * n2)

    # Interpretation (Romano et al., 2006)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return delta, interpretation


def rank_biserial_correlation(U: float, n1: int, n2: int) -> float:
    """
    Calculate rank-biserial correlation from Mann-Whitney U.

    r_rb = 1 - 2U / (n1 * n2)

    Parameters
    ----------
    U : float
        Mann-Whitney U statistic
    n1 : int
        Size of first group
    n2 : int
        Size of second group

    Returns
    -------
    float
        Rank-biserial correlation [-1, 1]
    """
    return 1 - (2 * U) / (n1 * n2)


def mann_whitney_with_effect(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict:
    """
    Mann-Whitney U test with effect size measures.

    Parameters
    ----------
    x : np.ndarray
        First group values
    y : np.ndarray
        Second group values
    alternative : str
        'two-sided', 'less', 'greater'

    Returns
    -------
    dict
        Test results with effect sizes
    """
    # Mann-Whitney U test
    U, p_value = stats.mannwhitneyu(x, y, alternative=alternative)

    # Effect sizes
    n1, n2 = len(x), len(y)
    r_rb = rank_biserial_correlation(U, n1, n2)
    delta, delta_interp = cliffs_delta(x, y)

    return {
        "U_statistic": float(U),
        "p_value": float(p_value),
        "n1": n1,
        "n2": n2,
        "rank_biserial": float(r_rb),
        "cliffs_delta": float(delta),
        "cliffs_delta_interpretation": delta_interp,
        "median_x": float(np.median(x)),
        "median_y": float(np.median(y))
    }


def multiple_comparisons_correction(
    p_values: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Dict:
    """
    Apply multiple testing correction.

    Parameters
    ----------
    p_values : List[float]
        Raw p-values
    method : str
        Correction method: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg),
        'fdr_by', 'holm', 'hommel'
    alpha : float
        Significance level

    Returns
    -------
    dict
        Corrected results
    """
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)

    return {
        "original_p_values": list(p_values),
        "corrected_p_values": list(pvals_corrected),
        "reject_null": list(reject),
        "method": method,
        "alpha": alpha,
        "n_significant": int(np.sum(reject))
    }


def pairwise_comparisons_with_correction(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh'
) -> pd.DataFrame:
    """
    Perform pairwise Mann-Whitney comparisons with FDR correction.

    Parameters
    ----------
    groups : Dict[str, np.ndarray]
        Dictionary of group name -> values
    alpha : float
        Significance level
    correction_method : str
        Multiple testing correction method

    Returns
    -------
    pd.DataFrame
        Pairwise comparison results
    """
    group_names = list(groups.keys())
    n_groups = len(group_names)

    results = []
    raw_p_values = []

    # Pairwise comparisons
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name1, name2 = group_names[i], group_names[j]
            x, y = groups[name1], groups[name2]

            mw_result = mann_whitney_with_effect(x, y)

            results.append({
                "group1": name1,
                "group2": name2,
                "U_statistic": mw_result["U_statistic"],
                "p_raw": mw_result["p_value"],  # CHANGE-7: renamed from p_value_raw
                "effect_size_r": mw_result["rank_biserial"],  # CHANGE-7: renamed from rank_biserial
                "cliffs_delta": mw_result["cliffs_delta"],
                "effect_interpretation": mw_result["cliffs_delta_interpretation"],
                "median_group1": mw_result["median_x"],
                "median_group2": mw_result["median_y"]
            })
            raw_p_values.append(mw_result["p_value"])

    # Apply correction
    if raw_p_values:
        correction = multiple_comparisons_correction(raw_p_values, method=correction_method, alpha=alpha)

        for i, result in enumerate(results):
            result["p_corrected_bh"] = correction["corrected_p_values"][i]  # CHANGE-7: renamed
            result["significant"] = correction["reject_null"][i]

    return pd.DataFrame(results)


def kruskal_wallis_with_posthoc(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict:
    """
    Kruskal-Wallis H-test with Dunn's post-hoc test.

    Parameters
    ----------
    groups : Dict[str, np.ndarray]
        Dictionary of group name -> values
    alpha : float
        Significance level

    Returns
    -------
    dict
        Test results with post-hoc comparisons
    """
    group_values = list(groups.values())
    group_names = list(groups.keys())

    # Kruskal-Wallis omnibus test
    H, p_value = stats.kruskal(*group_values)

    result = {
        "H_statistic": float(H),
        "p_value": float(p_value),
        "n_groups": len(groups),
        "significant": p_value < alpha
    }

    # Post-hoc Dunn test if significant
    if p_value < alpha:
        try:
            from scikit_posthocs import posthoc_dunn
            posthoc = pairwise_comparisons_with_correction(groups, alpha=alpha)
            result["posthoc_comparisons"] = posthoc.to_dict('records')
        except ImportError:
            warnings.warn("scikit-posthocs not installed. Using manual pairwise comparisons.")
            posthoc = pairwise_comparisons_with_correction(groups, alpha=alpha)
            result["posthoc_comparisons"] = posthoc.to_dict('records')

    return result


def cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size (parametric alternative).

    Parameters
    ----------
    x : np.ndarray
        First group
    y : np.ndarray
        Second group

    Returns
    -------
    Tuple[float, str]
        (d value, interpretation)
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    d = (np.mean(x) - np.mean(y)) / pooled_std

    # Interpretation (Cohen, 1988)
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return d, interpretation


# Convenience function for RQ2 analysis
def compare_typology_groups(
    data: pd.DataFrame,
    metric_col: str,
    typology_col: str = 'typology',
    groups_to_compare: List[str] = None
) -> pd.DataFrame:
    """
    Compare metric across typology groups with proper statistical testing.

    Parameters
    ----------
    data : pd.DataFrame
        Data with typology labels and metric
    metric_col : str
        Column name for metric to compare
    typology_col : str
        Column name for typology labels
    groups_to_compare : List[str]
        Specific groups to compare (default: all)

    Returns
    -------
    pd.DataFrame
        Comparison results with effect sizes
    """
    if groups_to_compare is None:
        groups_to_compare = data[typology_col].unique().tolist()

    groups = {
        name: data[data[typology_col] == name][metric_col].values
        for name in groups_to_compare
    }

    return pairwise_comparisons_with_correction(groups)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate data
    hidden = np.random.normal(100, 20, 50)
    overrated = np.random.normal(80, 25, 50)
    true_inf = np.random.normal(120, 15, 50)

    groups = {
        "Hidden": hidden,
        "Overrated": overrated,
        "True_Influencer": true_inf
    }

    print("Kruskal-Wallis with post-hoc:")
    result = kruskal_wallis_with_posthoc(groups)
    print(f"H = {result['H_statistic']:.2f}, p = {result['p_value']:.4f}")

    print("\nPairwise comparisons:")
    df = pairwise_comparisons_with_correction(groups)
    print(df[['group1', 'group2', 'effect_size_r', 'cliffs_delta', 'effect_interpretation', 'p_raw', 'p_corrected_bh', 'significant']])
