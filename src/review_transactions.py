from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import DATA_DIR, atomic_write_csv, normalize_merchant
from src.predict_phonepe import (
    FEEDBACK_PATH,
    FINAL_RESULTS_PATH,
    PREDICTIONS_PATH,
    load_feedback,
    run_prediction,
)


def save_correction(
    merchant: str,
    category: str,
    *,
    transaction_id: str | None = None,
    feedback_path: Path = FEEDBACK_PATH,
) -> None:
    merchant_value = merchant.strip()
    if not category.strip():
        raise ValueError("A correction category is required.")
    if transaction_id is None and not merchant_value:
        raise ValueError("A merchant is required for merchant-wide feedback.")
    if transaction_id is not None and not transaction_id.strip():
        raise ValueError("A transaction ID is required for transaction feedback.")
    feedback = load_feedback(feedback_path)
    scope = "transaction" if transaction_id is not None else "merchant"
    if scope == "transaction":
        matches = feedback["scope"].eq(scope) & feedback["transaction_id"].eq(
            transaction_id.strip()
        )
    else:
        key = normalize_merchant(merchant_value)
        matches = feedback["scope"].eq(scope) & feedback["merchant"].map(
            normalize_merchant
        ).eq(key)
    row = {
        "merchant": merchant_value,
        "final_category": category.strip(),
        "scope": scope,
        "transaction_id": transaction_id.strip() if transaction_id else "",
    }
    if matches.any():
        first = matches[matches].index[0]
        feedback.loc[first, list(row)] = list(row.values())
        feedback = feedback.drop(index=matches[matches].index[1:])
    else:
        feedback = pd.concat([feedback, pd.DataFrame([row])], ignore_index=True)
    atomic_write_csv(feedback, feedback_path)


def apply_feedback_to_existing_outputs() -> pd.DataFrame:
    if not (DATA_DIR / "phonepe_processed.csv").exists():
        raise FileNotFoundError("Run `python -m src.pipeline` before reviewing transactions.")
    result = run_prediction()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and save category corrections.")
    parser.add_argument("--merchant")
    parser.add_argument("--transaction-id")
    parser.add_argument("--category")
    args = parser.parse_args()
    if not args.category:
        parser.error("--category is required.")
    if args.transaction_id is None and args.merchant is None:
        parser.error("Provide --transaction-id or --merchant.")
    if args.transaction_id is None and args.merchant:
        transaction_id = None
    else:
        transaction_id = args.transaction_id
    save_correction(
        args.merchant or "",
        args.category,
        transaction_id=transaction_id,
    )
    result = apply_feedback_to_existing_outputs()
    print(
        f"Saved correction and reclassified {len(result)} transactions. "
        f"Updated {PREDICTIONS_PATH} and {FINAL_RESULTS_PATH}."
    )


if __name__ == "__main__":
    main()
