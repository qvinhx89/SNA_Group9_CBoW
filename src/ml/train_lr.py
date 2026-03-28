"""
Logistic Regression Training Module
===================================
Train ML models for typology detectability (RQ4).

FORBIDDEN-2: Only LR baselines per proposal Section 5:
- Majority class baseline
- LR (views only)
- LR (degree only)
- LR (views + degree)

NO RandomForest, NO SHAP - the point of RQ4 is to show surface metrics
are insufficient, so LR is intentionally simple.

NUANCED-2: 5-fold CV for variance estimation (supplementary only).
Primary result is the 70/10/20 split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)

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


def train_and_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    model_name: str
) -> Dict:
    """
    Train model and evaluate on all splits.

    Parameters
    ----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    X_test, y_test : Test data
    model : sklearn model
    model_name : str

    Returns
    -------
    Dict : Evaluation results
    """
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Evaluate
    results = {
        "model": model_name,
        "train": {
            "accuracy": float(accuracy_score(y_train, y_train_pred)),
            "f1_macro": float(f1_score(y_train, y_train_pred, average='macro')),
            "f1_weighted": float(f1_score(y_train, y_train_pred, average='weighted'))
        },
        "val": {
            "accuracy": float(accuracy_score(y_val, y_val_pred)),
            "f1_macro": float(f1_score(y_val, y_val_pred, average='macro')),
            "f1_weighted": float(f1_score(y_val, y_val_pred, average='weighted'))
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "f1_macro": float(f1_score(y_test, y_test_pred, average='macro')),
            "f1_weighted": float(f1_score(y_test, y_test_pred, average='weighted')),
            "f1_per_class": {
                label: float(f1_score(y_test, y_test_pred, labels=[label], average='macro'))
                for label in np.unique(y_test)
            },
            "precision_macro": float(precision_score(y_test, y_test_pred, average='macro')),
            "recall_macro": float(recall_score(y_test, y_test_pred, average='macro')),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
        }
    }

    return results


def run_cv_variance_estimation(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_folds: int = 5
) -> Dict:
    """
    NUANCED-2: Run stratified CV on train+val to estimate variance.

    Parameters
    ----------
    X : Features (train + val combined)
    y : Labels
    model : sklearn model
    n_folds : Number of CV folds

    Returns
    -------
    Dict : CV results with mean ± std
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Get CV scores
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

    return {
        "n_folds": n_folds,
        "f1_scores": f1_scores.tolist(),
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "note": "Supplementary CV for variance estimation. Primary result is 70/10/20 split."
    }


def train_ml_detectability(
    typology_path: str = "data/processed/typology_labels.parquet",
    centrality_path: str = "data/processed/centrality_table.parquet",
    output_dir: str = "outputs/stage6_ml",
    config_path: str = "src/config/base.yaml",
    seed: int = 42
) -> pd.DataFrame:
    """
    Train ML models for RQ4 detectability analysis.

    FORBIDDEN-2: Only trains LR baselines per proposal.
    NUANCED-2: Additionally runs 5-fold CV for variance estimation.

    Parameters
    ----------
    typology_path : str
        Path to typology labels
    centrality_path : str
        Path to centrality table (for features)
    output_dir : str
        Output directory
    config_path : str
        Configuration file
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Results summary
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)
    train_size = config.get('ml', {}).get('split', {}).get('train', 0.70)
    val_size = config.get('ml', {}).get('split', {}).get('val', 0.10)
    test_size = config.get('ml', {}).get('split', {}).get('test', 0.20)
    cv_folds = config.get('ml', {}).get('cv_folds', 5)

    # Load data
    logger.info("Loading data...")
    typology_df = pd.read_parquet(typology_path)
    centrality_df = pd.read_parquet(centrality_path)

    # Merge data
    data = typology_df.merge(centrality_df, on='node_id', how='inner')
    logger.info(f"Loaded {len(data)} samples")

    # Check required columns
    required_cols = ['node_id', 'typology_label', 'views', 'degree']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare features
    X_views = data[['views']].values
    X_degree = data[['degree']].values
    X_combined = data[['views', 'degree']].values
    y = data['typology_label'].values

    # Scale features
    scaler_views = StandardScaler()
    scaler_degree = StandardScaler()
    scaler_combined = StandardScaler()

    # 70/10/20 stratified split
    logger.info(f"Creating {train_size}/{val_size}/{test_size} stratified split...")

    # First split: train+val vs test
    X_views_trainval, X_views_test, y_trainval, y_test = train_test_split(
        X_views, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Second split: train vs val
    val_ratio = val_size / (train_size + val_size)
    X_views_train, X_views_val, y_train, y_val = train_test_split(
        X_views_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=seed
    )

    # Same splits for other feature sets (using indices)
    indices = np.arange(len(data))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_ratio, stratify=y[idx_trainval], random_state=seed
    )

    X_degree_train, X_degree_val, X_degree_test = X_degree[idx_train], X_degree[idx_val], X_degree[idx_test]
    X_combined_train, X_combined_val, X_combined_test = X_combined[idx_train], X_combined[idx_val], X_combined[idx_test]

    logger.info(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # Scale
    X_views_train_scaled = scaler_views.fit_transform(X_views_train)
    X_views_val_scaled = scaler_views.transform(X_views_val)
    X_views_test_scaled = scaler_views.transform(X_views_test)

    X_degree_train_scaled = scaler_degree.fit_transform(X_degree_train)
    X_degree_val_scaled = scaler_degree.transform(X_degree_val)
    X_degree_test_scaled = scaler_degree.transform(X_degree_test)

    X_combined_train_scaled = scaler_combined.fit_transform(X_combined_train)
    X_combined_val_scaled = scaler_combined.transform(X_combined_val)
    X_combined_test_scaled = scaler_combined.transform(X_combined_test)

    # Train models (FORBIDDEN-2: only LR baselines per proposal)
    all_results = []

    # 1. Majority class baseline
    logger.info("Training Majority Class baseline...")
    majority_model = DummyClassifier(strategy='most_frequent', random_state=seed)
    results = train_and_evaluate_model(
        X_views_train_scaled, y_train,
        X_views_val_scaled, y_val,
        X_views_test_scaled, y_test,
        majority_model, "majority_class"
    )
    all_results.append(results)

    # 2. LR (views only)
    logger.info("Training LR (views only)...")
    lr_views = LogisticRegression(random_state=seed, max_iter=1000)
    results = train_and_evaluate_model(
        X_views_train_scaled, y_train,
        X_views_val_scaled, y_val,
        X_views_test_scaled, y_test,
        lr_views, "lr_views_only"
    )
    all_results.append(results)

    # 3. LR (degree only)
    logger.info("Training LR (degree only)...")
    lr_degree = LogisticRegression(random_state=seed, max_iter=1000)
    results = train_and_evaluate_model(
        X_degree_train_scaled, y_train,
        X_degree_val_scaled, y_val,
        X_degree_test_scaled, y_test,
        lr_degree, "lr_degree_only"
    )
    all_results.append(results)

    # 4. LR (views + degree)
    logger.info("Training LR (views + degree)...")
    lr_combined = LogisticRegression(random_state=seed, max_iter=1000)
    results = train_and_evaluate_model(
        X_combined_train_scaled, y_train,
        X_combined_val_scaled, y_val,
        X_combined_test_scaled, y_test,
        lr_combined, "lr_views_degree"
    )
    all_results.append(results)

    # NUANCED-2: 5-fold CV for variance estimation (supplementary)
    logger.info(f"Running {cv_folds}-fold CV for variance estimation (supplementary)...")
    cv_results = {}

    # CV on train+val combined
    X_views_trainval_scaled = scaler_views.fit_transform(X_views[idx_trainval])
    X_combined_trainval_scaled = scaler_combined.fit_transform(X_combined[idx_trainval])

    cv_results["lr_views_only"] = run_cv_variance_estimation(
        X_views_trainval_scaled, y[idx_trainval],
        LogisticRegression(random_state=seed, max_iter=1000),
        n_folds=cv_folds
    )

    cv_results["lr_views_degree"] = run_cv_variance_estimation(
        X_combined_trainval_scaled, y[idx_trainval],
        LogisticRegression(random_state=seed, max_iter=1000),
        n_folds=cv_folds
    )

    # Add CV results to main results
    for result in all_results:
        model_name = result["model"]
        if model_name in cv_results:
            result["cv_variance_estimation"] = cv_results[model_name]

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "split": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_test": len(idx_test)
        },
        "models": all_results,
        "note": "FORBIDDEN-2: Only LR baselines per proposal Section 5. NO RandomForest, NO SHAP.",
        "cv_note": "NUANCED-2: CV variance estimation is supplementary. Primary result is 70/10/20 split."
    }

    output_path = output_dir / "rq4_detectability.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    # Create summary CSV
    summary_rows = []
    for result in all_results:
        row = {
            "model": result["model"],
            "test_accuracy": result["test"]["accuracy"],
            "test_f1_macro": result["test"]["f1_macro"],
            "test_f1_weighted": result["test"]["f1_weighted"],
            "test_precision_macro": result["test"]["precision_macro"],
            "test_recall_macro": result["test"]["recall_macro"]
        }
        if "cv_variance_estimation" in result:
            cv = result["cv_variance_estimation"]
            row["cv_f1_mean"] = cv["mean_f1"]
            row["cv_f1_std"] = cv["std_f1"]
            row["cv_f1_display"] = f"{cv['mean_f1']:.3f} +/- {cv['std_f1']:.3f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "rq4_detectability.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    logger.info("ML detectability training complete")
    return summary_df


if __name__ == "__main__":
    train_ml_detectability()
