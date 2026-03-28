"""
Random Forest Training Module
=============================
Train RandomForest classifier for typology detection as upper-bound comparison.

This provides a non-linear baseline to compare against LR models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import yaml
import pickle

from sklearn.ensemble import RandomForestClassifier
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


def train_random_forest(
    typology_path: str = "data/processed/typology_labels.parquet",
    centrality_path: str = "data/processed/centrality_table.parquet",
    output_dir: str = "outputs/stage6_ml",
    config_path: str = "src/config/base.yaml",
    seed: int = 42
) -> Dict:
    """
    Train RandomForest classifier for typology detection.

    Parameters
    ----------
    typology_path : str
        Path to typology labels
    centrality_path : str
        Path to centrality features
    output_dir : str
        Output directory
    config_path : str
        Config file path
    seed : int
        Random seed

    Returns
    -------
    Dict
        Training results
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

    # Merge
    data = typology_df.merge(centrality_df, on='node_id', how='inner')
    logger.info(f"Loaded {len(data)} samples")

    # Features: views + degree (surface metrics only, for fair comparison with LR)
    feature_cols = ['views', 'degree']
    missing_cols = [c for c in feature_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X = data[feature_cols].values
    y = data['typology_label'].values

    # Stratified split: 70/10/20
    logger.info(f"Creating {train_size}/{val_size}/{test_size} stratified split...")

    # First: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Second: train vs val
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=seed
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    logger.info("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = rf.predict(X_train_scaled)
    y_val_pred = rf.predict(X_val_scaled)
    y_test_pred = rf.predict(X_test_scaled)

    # Metrics
    results = {
        "model": "random_forest",
        "features": feature_cols,
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
            "precision_macro": float(precision_score(y_test, y_test_pred, average='macro')),
            "recall_macro": float(recall_score(y_test, y_test_pred, average='macro')),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True)
        }
    }

    # Feature importance
    results["feature_importance"] = {
        feature_cols[i]: float(rf.feature_importances_[i])
        for i in range(len(feature_cols))
    }

    # Cross-validation for variance estimation
    logger.info(f"Running {cv_folds}-fold CV for variance estimation...")
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1),
        X_trainval_scaled, y_trainval,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        scoring='f1_macro'
    )
    results["cv"] = {
        "n_folds": cv_folds,
        "f1_scores": cv_scores.tolist(),
        "mean_f1": float(np.mean(cv_scores)),
        "std_f1": float(np.std(cv_scores))
    }

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        },
        "results": results
    }

    output_path = output_dir / "rf_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    # Save model
    model_path = output_dir / "rf_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': rf, 'scaler': scaler}, f)
    logger.info(f"Saved model to {model_path}")

    # Summary
    logger.info(f"RandomForest Test F1 (macro): {results['test']['f1_macro']:.4f}")
    logger.info(f"RandomForest CV F1: {results['cv']['mean_f1']:.4f} +/- {results['cv']['std_f1']:.4f}")

    return results


if __name__ == "__main__":
    train_random_forest()
