"""Probability-of-default (PD) model: train, evaluate, score.

A gradient-boosted classifier on the Master Analytical Record predicts TARGET (default). The
output score feeds the monitoring side of the engine: PSI on the score (development vs scoring
population) and model-health checks (discrimination and calibration) in the diagnostic agent.

Home Credit has no calendar dates, so the hold-out is a stratified random split, not
out-of-time. Reported performance is therefore an optimistic bound versus a true
out-of-time validation. Performance can only be re-measured on data that has labels.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET = "TARGET"
ID = "SK_ID_CURR"
MODEL_FILE, METRICS_FILE = "pd_model.joblib", "model_metrics.json"
DEV_SCORES, TEST_SCORES = "model_scores_development.parquet", "model_scores_test.parquet"
SEED = 42


def feature_columns(schema: dict[str, pl.DataType]) -> list[str]:
    """Everything except the identifier and the label (never feed the target to the model)."""
    return [c for c in schema if c not in (ID, TARGET)]


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def calibration_table(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Predicted vs observed default rate by score decile (a well-calibrated model has them equal)."""
    order = np.argsort(score, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append(
            {"accounts": int(len(chunk)), "mean_score": float(score[chunk].mean()), "observed_rate": float(y_true[chunk].mean())}
        )
    return rows


def build_pipeline(frame: pl.DataFrame, features: list[str], seed: int = SEED) -> Pipeline:
    """Ordinal-encode string columns (unseen/missing -> NaN, which boosting treats as missing); numerics pass through."""
    categorical = [c for c in features if frame.schema[c] == pl.String]
    encoder = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan, dtype=float), categorical)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=40,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=seed,
        categorical_features=list(range(len(categorical))) if categorical else None,
    )
    return Pipeline([("encode", encoder), ("model", model)])


def _to_pandas(frame: pl.DataFrame, features: list[str]):
    return frame.select(features).to_pandas()


def train_model(
    mar_path: str | Path = "data/processed/master_analytical_record.parquet",
    processed_dir: str | Path = "data/processed",
    model_dir: str | Path = "data/models",
    holdout_fraction: float = 0.2,
    seed: int = SEED,
    importance_sample: int = 10_000,
) -> dict[str, Any]:
    """Train on a stratified split, evaluate on the hold-out, and write the model, metrics and hold-out scores."""
    start = time.perf_counter()
    # Sort first: the MAR's physical row order is not guaranteed (parallel DuckDB writes), and a positional
    # split on it would pick a different hold-out each run.
    frame = pl.read_parquet(mar_path).sort(ID)
    features = feature_columns(frame.schema)
    y = frame[TARGET].to_numpy()
    train_idx, hold_idx = train_test_split(np.arange(frame.height), test_size=holdout_fraction, stratify=y, random_state=seed)
    train, hold = frame[train_idx], frame[hold_idx]

    pipeline = build_pipeline(frame, features, seed)
    pipeline.fit(_to_pandas(train, features), train[TARGET].to_numpy())
    X_hold, y_hold = _to_pandas(hold, features), hold[TARGET].to_numpy()
    p_hold = pipeline.predict_proba(X_hold)[:, 1]

    auc = float(roc_auc_score(y_hold, p_hold))
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(y_hold), size=min(importance_sample, len(y_hold)), replace=False)
    imp = permutation_importance(pipeline, X_hold.iloc[sample], y_hold[sample], scoring="roc_auc", n_repeats=2, random_state=seed, n_jobs=1)
    top = sorted(zip(features, imp.importances_mean), key=lambda t: t[1], reverse=True)[:15]

    metrics = {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_hold, p_hold),
        "brier": float(brier_score_loss(y_hold, p_hold)),
        "base_rate": float(y_hold.mean()),
        "mean_predicted": float(p_hold.mean()),
        "calibration": calibration_table(y_hold, p_hold),
        "top_features": [{"feature": f, "auc_drop": float(v)} for f, v in top],
        "n_train": int(train.height),
        "n_holdout": int(hold.height),
        "n_features": len(features),
        "boosting_iterations": int(pipeline.named_steps["model"].n_iter_),
        "validation": "stratified random hold-out (no dates available; not out-of-time)",
        "seed": seed,
        "sklearn": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(time.perf_counter() - start, 1),
    }

    model_dir, processed_dir = Path(model_dir), Path(processed_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features, "dtypes": {c: str(frame.schema[c]) for c in features}}, model_dir / MODEL_FILE)
    (model_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    pl.DataFrame({ID: hold[ID], TARGET: hold[TARGET], "score": p_hold}).write_parquet(processed_dir / DEV_SCORES)
    logger.info("PD model: AUC=%.4f Gini=%.4f KS=%.4f on %d hold-out rows (%.1fs)", auc, metrics["gini"], metrics["ks"], hold.height, metrics["train_seconds"])
    return metrics


def score_frame(frame: pl.DataFrame, model_dir: str | Path = "data/models") -> pl.DataFrame:
    """Score any frame with the saved model. Missing feature columns are an error; extra ones are ignored."""
    bundle = joblib.load(Path(model_dir) / MODEL_FILE)
    missing = [c for c in bundle["features"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Cannot score: {len(missing)} feature columns missing, e.g. {missing[:5]}")
    scores = bundle["pipeline"].predict_proba(_to_pandas(frame, bundle["features"]))[:, 1]
    return pl.DataFrame({ID: frame[ID], "score": scores})


def score_test(
    mar_test_path: str | Path = "data/processed/master_analytical_record_test.parquet",
    processed_dir: str | Path = "data/processed",
    model_dir: str | Path = "data/models",
) -> pl.DataFrame:
    """Score the application_test population and write model_scores_test.parquet."""
    scores = score_frame(pl.read_parquet(mar_test_path), model_dir)
    scores.write_parquet(Path(processed_dir) / TEST_SCORES)
    logger.info("Scored %d test applications (mean score %.4f)", scores.height, scores["score"].mean())
    return scores


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in train_model().items() if k not in ("calibration", "top_features")}, indent=2))
