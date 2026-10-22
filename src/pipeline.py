from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.common import (
    DATA_DIR,
    MODEL_PATH,
    REPORT_DIR,
    atomic_write_csv,
    atomic_write_json,
)
from src.financial_intelligence import (
    add_expense_flexibility,
    cluster_spending,
    detect_anomalies,
    detect_recurring_transactions,
)
from src.predict_phonepe import (
    FINAL_RESULTS_PATH,
    FEEDBACK_PATH,
    PROCESSED_PATH,
    PREDICTIONS_PATH,
    run_prediction,
)
from src.preprocess_phonepe import RAW_PATH, preprocess_phonepe
from src.train_model import (
    METRICS_PATH,
    MODEL_VERSION,
    TRAINING_PATH,
    train_and_evaluate,
)


def _model_matches_training() -> bool:
    if not MODEL_PATH.exists():
        return False
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        return False
    current_hash = hashlib.sha256(TRAINING_PATH.read_bytes()).hexdigest()
    return (
        artifact.get("training_sha256") == current_hash
        and artifact.get("model_version") == MODEL_VERSION
    )


def refresh_derived_outputs(predictions: pd.DataFrame) -> dict:
    spending, cluster_metrics = cluster_spending(predictions)
    anomalies = detect_anomalies(predictions)
    flexibility = add_expense_flexibility(predictions)
    recurring = detect_recurring_transactions(predictions)

    atomic_write_csv(spending, DATA_DIR / "spending_behavior.csv")
    atomic_write_csv(anomalies, DATA_DIR / "anomaly_results.csv")
    atomic_write_csv(flexibility, DATA_DIR / "expense_flexibility.csv")
    atomic_write_csv(recurring, DATA_DIR / "recurring_transactions.csv")
    atomic_write_json(cluster_metrics, REPORT_DIR / "spending_clusters.json")
    return {
        "anomaly_rows": len(anomalies),
        "recurring_patterns": len(recurring),
        "cluster_metrics": cluster_metrics,
    }


def run_pipeline(retrain: bool = False) -> dict:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"PhonePe statement not found: {RAW_PATH}")
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_PATH}")
    processed = preprocess_phonepe()
    if retrain or not _model_matches_training():
        _, metrics = train_and_evaluate()
    else:
        if not METRICS_PATH.exists():
            raise FileNotFoundError(
                f"Classifier evaluation report is missing: {METRICS_PATH}. "
                "Run `python -m src.train_model` to recreate it."
            )
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    predictions = run_prediction()
    analytic_results = refresh_derived_outputs(predictions)

    verified = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0",
        "model_version": MODEL_VERSION,
        "source": str(RAW_PATH),
        "source_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        "training_source": str(TRAINING_PATH),
        "training_sha256": hashlib.sha256(TRAINING_PATH.read_bytes()).hexdigest(),
        "feedback_source": str(FEEDBACK_PATH),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "processed_rows": len(processed),
        "prediction_rows": len(predictions),
        "anomaly_rows": analytic_results["anomaly_rows"],
        "recurring_patterns": analytic_results["recurring_patterns"],
        "model_accuracy": metrics["accuracy"],
        "model_macro_f1": metrics["macro_f1"],
        "model_weighted_f1": metrics["weighted_f1"],
        "outputs": [
            str(PROCESSED_PATH),
            str(PREDICTIONS_PATH),
            str(FINAL_RESULTS_PATH),
            str(DATA_DIR / "spending_behavior.csv"),
            str(DATA_DIR / "anomaly_results.csv"),
            str(DATA_DIR / "expense_flexibility.csv"),
            str(DATA_DIR / "recurring_transactions.csv"),
        ],
    }
    atomic_write_json(verified, REPORT_DIR / "pipeline_run.json")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FinTwin local data pipeline.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain even when the saved model matches the training CSV.",
    )
    args = parser.parse_args()
    result = run_pipeline(retrain=args.retrain)
    print("Pipeline completed and outputs were read-back validated.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
