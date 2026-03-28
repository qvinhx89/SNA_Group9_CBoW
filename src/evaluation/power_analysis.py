"""
Statistical Power Analysis Module
=================================
Compute required sample sizes and statistical power for hypothesis tests.

Used to determine appropriate sampling for RQ2 validation experiments.

References:
- Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
import yaml

from scipy import stats

try:
    from statsmodels.stats.power import TTestIndPower, NormalIndPower
    STATSMODELS_POWER = True
except ImportError:
    STATSMODELS_POWER = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cohens_d_from_groups(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size from two groups.

    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values

    Returns
    -------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d magnitude.

    Parameters
    ----------
    d : float
        Cohen's d value

    Returns
    -------
    str
        Interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_required_sample_size(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    ratio: float = 1.0
) -> int:
    """
    Compute required sample size per group for two-sample t-test.

    Parameters
    ----------
    effect_size : float
        Expected Cohen's d
    power : float
        Desired statistical power (default 0.80)
    alpha : float
        Significance level (default 0.05)
    ratio : float
        Ratio of group sizes n2/n1 (default 1.0 for equal)

    Returns
    -------
    int
        Required sample size per group (assuming equal groups)
    """
    if not STATSMODELS_POWER:
        # Fallback formula for equal groups
        # n ≈ 2 * ((z_alpha + z_beta) / d)^2
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        ratio=ratio,
        alternative='two-sided'
    )
    return int(np.ceil(n))


def compute_achieved_power(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05
) -> float:
    """
    Compute achieved power given sample size.

    Parameters
    ----------
    effect_size : float
        Observed or expected Cohen's d
    n_per_group : int
        Sample size per group
    alpha : float
        Significance level

    Returns
    -------
    float
        Achieved power
    """
    if not STATSMODELS_POWER:
        # Fallback approximation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z = effect_size * np.sqrt(n_per_group / 2) - z_alpha
        return float(stats.norm.cdf(z))

    analysis = TTestIndPower()
    power = analysis.solve_power(
        effect_size=effect_size,
        nobs1=n_per_group,
        alpha=alpha,
        ratio=1.0,
        alternative='two-sided'
    )
    return float(power)


def run_power_analysis(
    typology_path: str = "data/processed/typology_labels.parquet",
    ic_results_path: str = "outputs/stage4_single_seed/rq2_validation.csv",
    output_dir: str = "outputs/stage3",
    config_path: str = "src/config/base.yaml",
    target_power: float = 0.80,
    alpha: float = 0.05
) -> Dict:
    """
    Run comprehensive power analysis for RQ2 comparisons.

    Parameters
    ----------
    typology_path : str
        Path to typology labels (for group sizes)
    ic_results_path : str
        Path to IC validation results (for effect size estimation)
    output_dir : str
        Output directory
    config_path : str
        Config file path
    target_power : float
        Target statistical power
    alpha : float
        Significance level

    Returns
    -------
    Dict
        Power analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "target_power": target_power,
        "alpha": alpha,
        "analyses": {}
    }

    # Load typology data for group sizes
    try:
        typology_df = pd.read_parquet(typology_path)
        group_counts = typology_df['typology_label'].value_counts().to_dict()
        results["population_sizes"] = group_counts
        logger.info(f"Population sizes: {group_counts}")
    except FileNotFoundError:
        logger.warning("Typology file not found. Using hypothetical analysis.")
        group_counts = None

    # Standard effect size scenarios
    effect_sizes = {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8
    }

    # Compute required sample sizes for each effect size
    results["sample_size_requirements"] = {}
    for name, d in effect_sizes.items():
        n_required = compute_required_sample_size(d, power=target_power, alpha=alpha)
        results["sample_size_requirements"][name] = {
            "effect_size": d,
            "n_per_group": n_required,
            "n_total_two_groups": n_required * 2
        }
        logger.info(f"{name.capitalize()} effect (d={d}): n={n_required} per group")

    # If IC results available, estimate actual effect size
    try:
        ic_df = pd.read_csv(ic_results_path)
        if 'typology' in ic_df.columns and 'mean_reach' in ic_df.columns:
            # Compute effect size between Hidden and Overrated
            hidden_reach = ic_df[ic_df['typology'] == 'hidden']['mean_reach'].values
            overrated_reach = ic_df[ic_df['typology'] == 'overrated']['mean_reach'].values

            if len(hidden_reach) > 0 and len(overrated_reach) > 0:
                d_observed = cohens_d_from_groups(hidden_reach, overrated_reach)
                interpretation = interpret_cohens_d(d_observed)

                results["observed_effect_size"] = {
                    "hidden_vs_overrated": {
                        "cohens_d": float(d_observed),
                        "interpretation": interpretation,
                        "n_hidden": len(hidden_reach),
                        "n_overrated": len(overrated_reach)
                    }
                }

                # Achieved power with current samples
                min_n = min(len(hidden_reach), len(overrated_reach))
                achieved_power = compute_achieved_power(abs(d_observed), min_n, alpha)
                results["observed_effect_size"]["hidden_vs_overrated"]["achieved_power"] = achieved_power

                logger.info(f"Observed effect size (Hidden vs Overrated): d={d_observed:.3f} ({interpretation})")
                logger.info(f"Achieved power with n={min_n}: {achieved_power:.3f}")

    except FileNotFoundError:
        logger.info("IC results not yet available. Providing prospective analysis only.")

    # Recommendations
    recommendations = []

    # Minimum recommended: detect medium effect with 80% power
    n_medium = results["sample_size_requirements"]["medium"]["n_per_group"]
    recommendations.append(f"Minimum recommended: {n_medium} samples per typology group (for medium effect)")

    # Conservative: detect small effect
    n_small = results["sample_size_requirements"]["small"]["n_per_group"]
    recommendations.append(f"Conservative (small effect): {n_small} samples per group")

    # Practical recommendation
    if group_counts:
        min_group = min(group_counts.values())
        if min_group >= n_medium:
            recommendations.append(f"Population supports medium effect detection (min group: {min_group})")
        elif min_group >= n_small:
            recommendations.append(f"Population supports small effect detection (min group: {min_group})")
        else:
            recommendations.append(f"WARNING: Smallest group ({min_group}) may be insufficient for reliable detection")

    results["recommendations"] = recommendations

    # Save results
    output_path = output_dir / "power_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved power analysis to {output_path}")

    # Summary table
    summary_df = pd.DataFrame([
        {
            "effect_size_name": name,
            "cohens_d": d,
            "n_per_group_required": results["sample_size_requirements"][name]["n_per_group"],
            "target_power": target_power,
            "alpha": alpha
        }
        for name, d in effect_sizes.items()
    ])
    summary_path = output_dir / "power_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    logger.info("Power analysis complete")
    return results


if __name__ == "__main__":
    run_power_analysis()
