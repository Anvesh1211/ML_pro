from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from src.common import (
    DATA_DIR,
    MODEL_DIR,
    MODEL_PATH,
    REQUIRED_TRAINING_COLUMNS,
    atomic_write_json,
    build_model_text,
    require_columns,
)


TRAINING_PATH = DATA_DIR / "training.csv"
METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "classification_metrics.json"
RANDOM_STATE = 42
MODEL_VERSION = "1.0"


def build_classifier() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, sublinear_tf=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate(
    training_path: Path = TRAINING_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> tuple[Pipeline, dict]:
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_path}")
    frame = pd.read_csv(training_path, low_memory=False)
    require_columns(frame, REQUIRED_TRAINING_COLUMNS, "Training data")
    frame = frame.dropna(subset=["category"]).copy()
    if frame.empty:
        raise ValueError("Training data has no rows with a category label.")
    frame["category"] = frame["category"].astype(str).str.strip()
    frame = frame.loc[frame["category"].ne("")].copy()
    if frame["category"].nunique() < 2:
        raise ValueError("Training requires at least two distinct category labels.")
    text = build_model_text(frame)
    labels = frame["category"]
    groups = (
        frame["merchant"]
        .fillna("UNKNOWN_MERCHANT")
        .astype(str)
        .map(lambda value: " ".join(value.casefold().split()))
    )
    if groups.nunique() < 2:
        raise ValueError("Merchant-disjoint evaluation requires at least two merchants.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_indices, test_indices = next(splitter.split(text, labels, groups=groups))
    train_merchants = set(groups.iloc[train_indices])
    test_merchants = set(groups.iloc[test_indices])
    if train_merchants & test_merchants:
        raise RuntimeError("Merchant leakage detected in grouped evaluation split.")

    model = build_classifier()
    model.fit(text.iloc[train_indices], labels.iloc[train_indices])
    started = time.perf_counter()
    predictions = model.predict(text.iloc[test_indices])
    inference_ms = (time.perf_counter() - started) * 1000 / len(test_indices)
    y_true = labels.iloc[test_indices]
    supported_categories = sorted(labels.unique())
    evaluated_categories = sorted(y_true.unique())
    report = classification_report(
        y_true,
        predictions,
        labels=supported_categories,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true, predictions, labels=supported_categories
    )
    metrics = {
        "model_version": MODEL_VERSION,
        "dataset": training_path.name,
        "dataset_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "rows_used": int(len(frame)),
        "categories": supported_categories,
        "random_state": RANDOM_STATE,
        "split": "GroupShuffleSplit(test_size=0.20, groups=normalized merchant)",
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_merchants": int(len(train_merchants)),
        "test_merchants": int(len(test_merchants)),
        "merchant_overlap": 0,
        "categories_absent_from_test": sorted(
            set(supported_categories).difference(evaluated_categories)
        ),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "mean_inference_ms_per_transaction": float(inference_ms),
        "per_category": {
            category: {
                key: float(report[category][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for category in supported_categories
        },
        "confusion_matrix_labels": supported_categories,
        "confusion_matrix": matrix.astype(int).tolist(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    try:
        joblib.dump(
            {
                "model": model,
                "model_version": MODEL_VERSION,
                "categories": supported_categories,
                "training_sha256": metrics["dataset_sha256"],
                "random_state": RANDOM_STATE,
            },
            temporary_path,
        )
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    atomic_write_json(metrics, metrics_path)
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate FinTwin's classifier.")
    parser.add_argument("--training", type=Path, default=TRAINING_PATH)
    args = parser.parse_args()
    _, result = train_and_evaluate(training_path=args.training)
    print(
        f"Rows: {result['train_rows']} train / {result['test_rows']} test; "
        f"merchant overlap: {result['merchant_overlap']}"
    )
    print(
        f"Accuracy={result['accuracy']:.4f} "
        f"macro-F1={result['macro_f1']:.4f} "
        f"weighted-F1={result['weighted_f1']:.4f}"
    )
    print(f"Metrics: {METRICS_PATH}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
