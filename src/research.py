from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.common import (
    DATA_DIR,
    REPORT_DIR,
    atomic_write_json,
    build_model_text,
    normalize_merchant,
    require_columns,
)
from src.financial_intelligence import (
    Scenario,
    add_expense_flexibility,
    build_recommendations,
    calculate_scenario,
    forecast_monthly_expenses,
    monthly_summary,
)
from src.train_model import build_classifier, TRAINING_PATH


def run_personalization_experiment(
    training_path=TRAINING_PATH,
) -> dict:
    data = pd.read_csv(training_path, low_memory=False)
    require_columns(
        data,
        {
            "transaction_details", "merchant", "subcategory", "category",
            "transaction_id", "transaction_date",
        },
        "Personalization experiment training data",
    )
    data = data.dropna(subset=["category"]).copy()
    data["category"] = data["category"].astype(str).str.strip()
    data["_merchant_key"] = data["merchant"].map(normalize_merchant)
    data["_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")
    eligible_groups = {
        merchant_key: group
        for merchant_key, group in data.groupby("_merchant_key")
        if len(group) >= 2
    }
    if not eligible_groups:
        return {
            "available": False,
            "reason": "No merchant has two or more labeled examples.",
        }
    merchant_keys = sorted(eligible_groups)
    random.Random(42).shuffle(merchant_keys)
    evaluation_count = max(1, int(round(len(merchant_keys) * 0.20)))
    evaluation_merchants = set(merchant_keys[:evaluation_count])

    support_indices = []
    test_indices = []
    merchant_feedback = {}
    conflicting_merchants = 0
    for merchant_key in sorted(evaluation_merchants):
        group = eligible_groups[merchant_key]
        ordered = group.sort_values(
            ["_date", "transaction_id"], na_position="last", kind="stable"
        )
        support = ordered.iloc[0]
        support_indices.append(support.name)
        later = ordered.iloc[1:]
        test_indices.extend(later.index.tolist())
        merchant_feedback[merchant_key] = support["category"]
        if group["category"].nunique() > 1:
            conflicting_merchants += 1

    test_frame = data.loc[test_indices].copy()
    training_frame = data.loc[
        ~data["_merchant_key"].isin(evaluation_merchants)
    ].copy()
    if training_frame.empty or test_frame.empty or training_frame["category"].nunique() < 2:
        return {
            "available": False,
            "reason": "The available merchant groups cannot produce independent support and evaluation sets.",
        }
    model = build_classifier()
    model.fit(build_model_text(training_frame), training_frame["category"])
    baseline = model.predict(build_model_text(test_frame))
    personalized = [
        merchant_feedback[merchant_key]
        for merchant_key in test_frame["_merchant_key"]
    ]
    y_true = test_frame["category"]
    evaluated_categories = sorted(y_true.unique())
    baseline_report = classification_report(
        y_true,
        baseline,
        labels=evaluated_categories,
        output_dict=True,
        zero_division=0,
    )
    personalized_report = classification_report(
        y_true,
        personalized,
        labels=evaluated_categories,
        output_dict=True,
        zero_division=0,
    )
    return {
        "available": True,
        "experiment": "offline merchant-memory simulation on labeled training data",
        "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "support_policy": "Deterministically hold out 20% of repeated merchants (seed 42); use the earliest labeled row per held-out merchant as feedback.",
        "evaluation_policy": (
            "Evaluate only later rows from those merchants. Exclude all corrected merchants "
            "from the classifier fit, so the support label is not used to train the model."
        ),
        "training_rows": int(len(training_frame)),
        "support_corrections": int(len(support_indices)),
        "evaluation_rows": int(len(test_frame)),
        "evaluation_merchants": int(test_frame["_merchant_key"].nunique()),
        "merchants_with_mixed_ground_truth_labels": int(conflicting_merchants),
        "categories_absent_from_evaluation": sorted(
            set(data["category"]).difference(evaluated_categories)
        ),
        "baseline": {
            "accuracy": float(accuracy_score(y_true, baseline)),
            "macro_f1": float(
                f1_score(y_true, baseline, labels=evaluated_categories, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, baseline, labels=evaluated_categories, average="weighted", zero_division=0)
            ),
        },
        "merchant_memory": {
            "accuracy": float(accuracy_score(y_true, personalized)),
            "macro_f1": float(
                f1_score(y_true, personalized, labels=evaluated_categories, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, personalized, labels=evaluated_categories, average="weighted", zero_division=0)
            ),
        },
        "delta_macro_f1": float(
            f1_score(y_true, personalized, labels=evaluated_categories, average="macro", zero_division=0)
            - f1_score(y_true, baseline, labels=evaluated_categories, average="macro", zero_division=0)
        ),
        "personalized_per_category": {
            category: {
                key: float(personalized_report[category][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for category in evaluated_categories
        },
        "baseline_per_category": {
            category: {
                key: float(baseline_report[category][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for category in evaluated_categories
        },
        "limitations": [
            "This is an offline simulation using existing labeled training examples, not the user's actual feedback outcomes.",
            "The merchant-memory policy can misclassify mixed-use merchants; such merchants are counted and exposed.",
            "The chosen support row and later evaluation rows are disjoint, but only one deterministic support row per merchant is evaluated.",
            "PhonePe predictions have no independently verified ground-truth labels.",
        ],
    }


def run_all_experiments() -> dict:
    results_path = DATA_DIR / "final_phonepe_results.csv"
    transactions = pd.read_csv(
        results_path,
        dtype={"transaction_id": "string"},
        low_memory=False,
    )
    personalization = run_personalization_experiment()
    recommendations = build_recommendations(transactions)
    summary = monthly_summary(transactions)
    flexibility_rows = add_expense_flexibility(transactions)
    eligible_observations = int(
        flexibility_rows["flexibility_class"]
        .eq("Potentially adjustable (category proxy)")
        .sum()
    )
    observed_months = int(summary["month"].nunique())
    reduction_rate = 0.10
    all_consumption_per_month = (
        float(summary["consumption_expenses"].sum() / observed_months)
        if observed_months
        else 0.0
    )
    flexibility_adjusted_estimate = (
        float(recommendations["estimated_monthly_difference"].sum())
        if not recommendations.empty
        else 0.0
    )
    flexibility = {
        "assumption": f"{reduction_rate:.0%} reduction; hypothetical, not an observed saving.",
        "observed_months": observed_months,
        "eligible_observations": eligible_observations,
        "flexibility_adjusted_estimated_monthly_difference": flexibility_adjusted_estimate,
        "no_flexibility_layer_estimated_monthly_difference": (
            all_consumption_per_month * reduction_rate
        ),
        "estimated_difference_from_layer": (
            flexibility_adjusted_estimate - all_consumption_per_month * reduction_rate
        ),
        "average_all_consumption_expenses_per_observed_month": all_consumption_per_month,
        "interpretation": (
            "The flexibility layer limits the hypothetical reduction base to categories "
            "marked potentially adjustable; results remain category proxies."
        ),
    }
    forecast = forecast_monthly_expenses(transactions)
    stress = calculate_scenario(
        Scenario(
            monthly_income=50_000,
            essential_expenses=20_000,
            flexible_expenses=10_000,
            other_expenses=5_000,
            income_reduction_rate=0.20,
            essential_increase_rate=0.10,
            one_time_emergency_expense=6_000,
            new_monthly_emi=2_000,
            flexible_spending_reduction_rate=0.20,
        )
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": json.loads(
            (REPORT_DIR / "classification_metrics.json").read_text(encoding="utf-8")
        ),
        "personalization": personalization,
        "flexibility_comparison": flexibility,
        "expense_forecast": forecast,
        "spending_cluster_comparison": json.loads(
            (REPORT_DIR / "spending_clusters.json").read_text(encoding="utf-8")
        ),
        "stress_test_arithmetic": stress,
        "data_boundary": "Existing data/training.csv and current predictions derived from existing data/testing.csv only.",
    }
    atomic_write_json(result, REPORT_DIR / "experiments.json")
    return result


if __name__ == "__main__":
    output = run_all_experiments()
    print(json.dumps(output, indent=2, ensure_ascii=False))
