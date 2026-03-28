"""
SHAP Analysis Module
====================
Compute SHAP (SHapley Additive exPlanations) values to interpret
why Hidden Influencers are hard to detect from surface metrics.

References:
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import yaml
import pickle
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "src/config/base.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}


def compute_shap_values(
    model_path: str = "outputs/stage6_ml/rf_model.pkl",
    typology_path: str = "data/processed/typology_labels.parquet",
    centrality_path: str = "data/processed/centrality_table.parquet",
    output_dir: str = "outputs/stage6_ml",
    figures_dir: str = "reports/figures",
    config_path: str = "src/config/base.yaml",
    n_samples: int = 500,
    seed: int = 42
) -> Dict:
    """
    Compute SHAP values for model interpretation.

    Parameters
    ----------
    model_path : str
        Path to trained model (pickle with 'model' and 'scaler')
    typology_path : str
        Path to typology labels
    centrality_path : str
        Path to centrality features
    output_dir : str
        Output directory for SHAP values
    figures_dir : str
        Output directory for figures
    config_path : str
        Config file path
    n_samples : int
        Number of samples for SHAP computation
    seed : int
        Random seed

    Returns
    -------
    Dict
        SHAP analysis results
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP not available. Install with: pip install shap")
        return {"error": "SHAP not installed"}

    np.random.seed(seed)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return {"error": "Model file not found"}

    # Load data
    logger.info("Loading data...")
    typology_df = pd.read_parquet(typology_path)
    centrality_df = pd.read_parquet(centrality_path)

    data = typology_df.merge(centrality_df, on='node_id', how='inner')

    # Features
    feature_cols = ['views', 'degree']
    X = data[feature_cols].values
    y = data['typology_label'].values

    # Scale
    X_scaled = scaler.transform(X)

    # Sample for SHAP (for computational efficiency)
    if n_samples < len(X_scaled):
        sample_idx = np.random.choice(len(X_scaled), size=n_samples, replace=False)
        X_sample = X_scaled[sample_idx]
        y_sample = y[sample_idx]
        X_sample_original = X[sample_idx]
    else:
        X_sample = X_scaled
        y_sample = y
        X_sample_original = X

    # Create SHAP explainer
    logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "n_samples": len(X_sample),
        "features": feature_cols,
        "expected_value": explainer.expected_value.tolist() if hasattr(explainer.expected_value, 'tolist') else list(explainer.expected_value),
        "class_labels": model.classes_.tolist()
    }

    # Mean absolute SHAP values per feature (global importance)
    shap_importance = {}
    for i, cls in enumerate(model.classes_):
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        shap_importance[cls] = {
            feature_cols[j]: float(mean_abs_shap[j])
            for j in range(len(feature_cols))
        }
    results["shap_importance_by_class"] = shap_importance

    # Overall importance
    all_shap = np.concatenate([np.abs(sv) for sv in shap_values], axis=0)
    overall_importance = all_shap.mean(axis=0)
    results["overall_importance"] = {
        feature_cols[i]: float(overall_importance[i])
        for i in range(len(feature_cols))
    }

    # Save SHAP values
    shap_df = pd.DataFrame(X_sample_original, columns=feature_cols)
    shap_df['typology'] = y_sample
    for i, cls in enumerate(model.classes_):
        for j, feat in enumerate(feature_cols):
            shap_df[f'shap_{feat}_{cls}'] = shap_values[i][:, j]

    shap_csv_path = output_dir / "shap_values.csv"
    shap_df.to_csv(shap_csv_path, index=False)
    logger.info(f"Saved SHAP values to {shap_csv_path}")

    # Generate plots
    logger.info("Generating SHAP plots...")

    # 1. Summary plot (beeswarm)
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # For binary Hidden vs others analysis
        if 'hidden' in model.classes_:
            hidden_idx = list(model.classes_).index('hidden')
            shap.summary_plot(
                shap_values[hidden_idx],
                X_sample_original,
                feature_names=feature_cols,
                show=False
            )
        else:
            # Multi-class summary
            shap.summary_plot(
                shap_values,
                X_sample_original,
                feature_names=feature_cols,
                class_names=model.classes_.tolist(),
                show=False
            )

        plt.tight_layout()
        beeswarm_path = figures_dir / "fig_shap_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved beeswarm plot to {beeswarm_path}")
        results["plots"] = {"beeswarm": str(beeswarm_path)}
    except Exception as e:
        logger.warning(f"Failed to generate beeswarm plot: {e}")

    # 2. Bar plot (feature importance)
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_values,
            X_sample_original,
            feature_names=feature_cols,
            plot_type="bar",
            class_names=model.classes_.tolist(),
            show=False
        )
        plt.tight_layout()
        bar_path = figures_dir / "fig_shap_importance.png"
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved importance plot to {bar_path}")
        results["plots"]["importance_bar"] = str(bar_path)
    except Exception as e:
        logger.warning(f"Failed to generate bar plot: {e}")

    # Interpretation for Hidden Influencers
    if 'hidden' in model.classes_:
        hidden_idx = list(model.classes_).index('hidden')
        hidden_mask = y_sample == 'hidden'
        if hidden_mask.sum() > 0:
            hidden_shap = shap_values[hidden_idx][hidden_mask]
            results["hidden_interpretation"] = {
                "n_hidden_samples": int(hidden_mask.sum()),
                "mean_shap_views": float(hidden_shap[:, 0].mean()),
                "mean_shap_degree": float(hidden_shap[:, 1].mean()),
                "interpretation": (
                    "Hidden Influencers have low views (negative SHAP for views) "
                    "but their structural position (degree) provides some signal. "
                    "The difficulty in detection comes from relying primarily on views."
                )
            }

    # Save results
    results_path = output_dir / "shap_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis results to {results_path}")

    logger.info("SHAP analysis complete")
    return results


if __name__ == "__main__":
    compute_shap_values()
