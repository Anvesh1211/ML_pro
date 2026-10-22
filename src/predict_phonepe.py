from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from src.common import (
    BASE_CATEGORIES,
    DATA_DIR,
    LEGACY_MODEL_PATH,
    MODEL_PATH,
    REQUIRED_PROCESSED_COLUMNS,
    atomic_write_csv,
    build_model_text,
    normalize_merchant,
    require_columns,
)
from src.preprocess_phonepe import classify_description


PROCESSED_PATH = DATA_DIR / "phonepe_processed.csv"
PREDICTIONS_PATH = DATA_DIR / "phonepe_predictions.csv"
FINAL_RESULTS_PATH = DATA_DIR / "final_phonepe_results.csv"
FEEDBACK_PATH = DATA_DIR / "user_feedback.csv"
LOW_CONFIDENCE_THRESHOLD = 0.30


def load_feedback(path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["merchant", "final_category", "scope", "transaction_id"]
        )
    feedback = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"merchant", "final_category"}
    missing = sorted(required.difference(feedback.columns))
    if missing:
        raise ValueError(
            f"Feedback file is missing required columns: {', '.join(missing)}"
        )
    for column, default in (
        ("scope", "merchant"),
        ("transaction_id", ""),
    ):
        if column not in feedback:
            feedback[column] = default
    feedback["merchant"] = feedback["merchant"].fillna("").str.strip()
    feedback["final_category"] = feedback["final_category"].fillna("").str.strip()
    feedback["scope"] = feedback["scope"].fillna("merchant").str.strip().str.casefold()
    feedback["transaction_id"] = (
        feedback["transaction_id"].fillna("").str.strip()
    )
    allowed_categories = set(BASE_CATEGORIES)
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            allowed_categories = set(artifact.get("categories", BASE_CATEGORIES))
    invalid = sorted(set(feedback["final_category"]) - allowed_categories)
    if invalid:
        raise ValueError(
            "Feedback contains categories not present in the training labels: "
            + ", ".join(invalid)
        )
    invalid_scope = sorted(
        set(feedback["scope"]).difference({"merchant", "transaction"})
    )
    if invalid_scope:
        raise ValueError("Feedback scope must be merchant or transaction.")
    if ((feedback["scope"] == "merchant") & feedback["merchant"].eq("")).any():
        raise ValueError("Merchant-scoped feedback requires a merchant value.")
    if (
        (feedback["scope"] == "transaction")
        & feedback["transaction_id"].eq("")
    ).any():
        raise ValueError("Transaction-scoped feedback requires a transaction_id.")
    return feedback


def _extract_model(artifact):
    return artifact["model"] if isinstance(artifact, dict) else artifact


def _load_model(path: Path | None = None):
    model_path = path or (MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained classifier found. Run `python -m src.train_model` first."
        )
    artifact = joblib.load(model_path)
    return _extract_model(artifact)


def _apply_feedback(
    frame: pd.DataFrame, feedback: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    corrected = frame["final_category"].copy()
    source = frame["category_source"].copy()
    seen: dict[tuple[str, str], str] = {}
    for scope in ("merchant", "transaction"):
        for _, row in feedback.loc[feedback["scope"].eq(scope)].iterrows():
            value = row["final_category"]
            if scope == "transaction":
                key = ("transaction", str(row["transaction_id"]))
                mask = frame["transaction_id"].astype(str).eq(key[1])
            else:
                merchant_key = normalize_merchant(row["merchant"])
                key = ("merchant", merchant_key)
                mask = frame["merchant"].map(normalize_merchant).eq(merchant_key)
            if not mask.any():
                continue
            if key in seen and seen[key] != value:
                raise ValueError(
                    f"Conflicting saved feedback for {scope} key {key[1]!r}."
                )
            seen[key] = value
            corrected.loc[mask] = value
            source.loc[mask] = (
                "User transaction correction"
                if scope == "transaction"
                else "User merchant feedback"
            )
    return corrected, source


def predict_transactions(
    processed: pd.DataFrame,
    model=None,
    feedback: pd.DataFrame | None = None,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    require_columns(processed, REQUIRED_PROCESSED_COLUMNS, "Processed transactions")
    if processed.empty:
        raise ValueError("There are no processed transactions to classify.")
    classifier = model or _load_model()
    result = processed.copy()
    texts = build_model_text(result)
    result["text"] = texts
    result["ml_category"] = classifier.predict(texts)
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("The configured classifier must provide predict_proba().")
    probabilities = classifier.predict_proba(texts)
    result["ml_confidence"] = probabilities.max(axis=1)
    result["final_category"] = result["ml_category"].astype(str)
    result["category_source"] = "ML"

    for index, row in result.iterrows():
        rule_category = classify_description(
            row["transaction_details"], row["transaction_type"]
        )
        if row["transaction_type"] == "CREDIT" and rule_category == "Needs Review":
            description = str(row["transaction_details"]).casefold()
            if "received from" in description:
                rule_category = "Personal Transfer"
        if rule_category != "Needs Review":
            result.at[index, "final_category"] = rule_category
            result.at[index, "category_source"] = "Deterministic rule"
        elif result.at[index, "ml_confidence"] < confidence_threshold:
            result.at[index, "final_category"] = "Needs Review"
            result.at[index, "category_source"] = "Low ML confidence"
        if (
            row["transaction_type"] == "DEBIT"
            and result.at[index, "final_category"] == "Salary/Income"
        ):
            result.at[index, "final_category"] = "Needs Review"
            result.at[index, "category_source"] = "Direction/category conflict"

    if feedback is not None:
        result["final_category"], result["category_source"] = _apply_feedback(
            result, feedback
        )
    return result


def run_prediction(
    processed_path: Path = PROCESSED_PATH,
    output_paths: tuple[Path, ...] = (PREDICTIONS_PATH, FINAL_RESULTS_PATH),
    feedback_path: Path = FEEDBACK_PATH,
    model_path: Path | None = None,
) -> pd.DataFrame:
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed transactions not found: {processed_path}")
    processed = pd.read_csv(
        processed_path,
        dtype={"transaction_id": "string", "utr": "string"},
        low_memory=False,
    )
    feedback = load_feedback(feedback_path)
    model = _load_model(model_path)
    result = predict_transactions(processed, model=model, feedback=feedback)
    required_result = {"final_category", "category_source", "ml_category"}
    if not required_result.issubset(result.columns):
        raise RuntimeError("Prediction did not produce the required result fields.")
    for output_path in output_paths:
        atomic_write_csv(result, output_path)
        check = pd.read_csv(output_path, low_memory=False)
        if len(check) != len(processed) or not required_result.issubset(check.columns):
            raise IOError(f"Prediction output failed read-back validation: {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify processed PhonePe transactions.")
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_prediction(processed_path=args.processed)
    print(f"Classified {len(result)} transactions in {time.perf_counter() - started:.3f}s.")
    print(result["category_source"].value_counts().to_string())
    print(f"Saved and validated: {PREDICTIONS_PATH}")
    print(f"Saved and validated: {FINAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
