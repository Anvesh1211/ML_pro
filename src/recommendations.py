from __future__ import annotations

import calendar
import math
import re
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd

from src.financial_intelligence import (
    FLEXIBLE_CATEGORIES,
    build_recommendations,
    canonicalize_transactions,
    detect_recurring_transactions,
)


def _period_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(frame["transaction_date"], errors="coerce").dropna()
    if dates.empty:
        today = date.today().isoformat()
        return today, today
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _is_full_calendar_month(month: pd.DataFrame) -> bool:
    dates = pd.to_datetime(month["transaction_date"], errors="coerce").dropna()
    if dates.empty:
        return False
    first = dates.min().date()
    last = dates.max().date()
    month_end = calendar.monthrange(first.year, first.month)[1]
    return first.day == 1 and last.day == month_end and first.month == last.month


def _to_minor_units(value: float) -> int:
    return int(
        (Decimal(str(max(0.0, value))) * 100).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def _make_suggestion(
    kind: str,
    title: str,
    explanation: str,
    evidence: dict[str, Any],
    start: str,
    end: str,
    saving: float | None = None,
    priority: str = "medium",
    action: str = "",
) -> dict[str, Any]:
    return {
        "period_start": start,
        "period_end": end,
        "category": kind,
        "title": title,
        "explanation": explanation,
        "evidence": evidence,
        "estimated_potential_saving_minor": (
            _to_minor_units(saving) if saving is not None else None
        ),
        "priority": priority,
        "recommended_action": action,
        "estimate_is_hypothetical": saving is not None,
    }


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype(str).str.lower()


def _category_totals(expenses: pd.DataFrame) -> pd.Series:
    if expenses.empty:
        return pd.Series(dtype=float)
    return expenses.groupby("final_category")["amount"].sum().sort_values(ascending=False)


def _merchant_text(frame: pd.DataFrame) -> pd.Series:
    candidates = ["merchant", "description", "transaction_note", "narration", "details"]
    result = pd.Series("", index=frame.index, dtype="string")
    for column in candidates:
        if column in frame.columns:
            result = result + " " + frame[column].fillna("").astype(str).str.lower()
    return result.str.strip()


def build_personalized_suggestions(
    frame: pd.DataFrame,
    goals: pd.DataFrame | None = None,
    *,
    assumed_reduction_rate: float = 0.10,
    category_share_threshold: float = 0.30,
    increase_rate_threshold: float = 0.25,
    minimum_increase_amount: float = 500.0,
    small_purchase_limit: float = 500.0,
    minimum_small_purchases: int = 5,
    max_suggestions: int = 12,
) -> list[dict[str, Any]]:
    """
    Generate transparent, rule-based financial suggestions from transaction data.

    Savings amounts are hypothetical scenarios, not guarantees. The function expects
    canonicalize_transactions() to provide transaction_date, amount, final_category,
    and is_consumption_expense. Merchant/narration fields are optional.
    """
    if not math.isfinite(assumed_reduction_rate) or not 0 <= assumed_reduction_rate <= 1:
        raise ValueError("assumed_reduction_rate must be between 0 and 1.")
    if not 0 <= category_share_threshold <= 1:
        raise ValueError("category_share_threshold must be between 0 and 1.")
    if not 0 <= increase_rate_threshold:
        raise ValueError("increase_rate_threshold must be non-negative.")
    if minimum_increase_amount < 0 or small_purchase_limit < 0:
        raise ValueError("Amount thresholds must be non-negative.")
    if minimum_small_purchases < 1 or max_suggestions < 1:
        raise ValueError("Suggestion limits must be positive.")
    if frame is None or frame.empty:
        return []

    ledger = canonicalize_transactions(frame).copy()
    if ledger.empty:
        return []
    ledger["transaction_date"] = pd.to_datetime(
        ledger["transaction_date"], errors="coerce"
    )
    ledger["amount"] = pd.to_numeric(ledger["amount"], errors="coerce")
    ledger = ledger.dropna(subset=["transaction_date", "amount"])
    if ledger.empty:
        return []

    if "final_category" not in ledger:
        ledger["final_category"] = "Uncategorized"
    ledger["final_category"] = (
        ledger["final_category"].fillna("Uncategorized").astype(str)
    )
    if "is_consumption_expense" not in ledger:
        raise ValueError(
            "canonicalize_transactions() must provide 'is_consumption_expense'."
        )

    start, end = _period_bounds(ledger)
    expenses = ledger.loc[
        ledger["is_consumption_expense"].fillna(False).astype(bool)
        & ledger["amount"].gt(0)
    ].copy()
    if expenses.empty:
        return []

    results: list[dict[str, Any]] = []
    total_spend = float(expenses["amount"].sum())
    if total_spend <= 0:
        return []

    categories = _category_totals(expenses)
    merchant_text = _merchant_text(expenses)
    expenses["_merchant_text"] = merchant_text

    # 1. Large share of spending concentrated in one category.
    for category, amount in categories.items():
        share = float(amount) / total_spend
        if share >= category_share_threshold and amount >= 1000:
            flexible = str(category) in FLEXIBLE_CATEGORIES
            saving = float(amount) * assumed_reduction_rate if flexible else None
            results.append(
                _make_suggestion(
                    "category_budget_share",
                    f"Review your {category} spending share",
                    "This category accounts for a large share of recorded consumption spending. "
                    "A large share is not automatically bad; check whether it matches your needs.",
                    {
                        "category": str(category),
                        "category_total": round(float(amount), 2),
                        "total_consumption_spend": round(total_spend, 2),
                        "share_percent": round(share * 100, 1),
                        "threshold_percent": round(category_share_threshold * 100, 1),
                    },
                    start,
                    end,
                    saving=saving,
                    priority="high" if share >= 0.40 else "medium",
                    action="Set a category limit for the next month and review the largest transactions.",
                )
            )

    # 2. Fast food, cafes, food delivery and dining keywords in merchant/narration.
    food_keywords = (
        "fast food", "burger", "pizza", "cafe", "coffee", "restaurant",
        "food delivery", "swiggy", "zomato", "dominos", "mcdonald",
        "kfc", "subway", "starbucks", "eatery", "bakery"
    )
    food_match = expenses["_merchant_text"].str.contains(
        "|".join(re.escape(word) for word in food_keywords), regex=True, na=False
    )
    if not food_match.any():
        food_match = expenses["final_category"].str.lower().str.contains(
            r"food|dining|restaurant|fast.?food|cafe", regex=True, na=False
        )
    food_spend = expenses.loc[food_match]
    if len(food_spend) >= 3:
        food_total = float(food_spend["amount"].sum())
        saving = food_total * assumed_reduction_rate
        results.append(
            _make_suggestion(
                "food_and_dining",
                "Find an easy saving in food and dining",
                "Several food, cafe, restaurant or food-delivery transactions were found. "
                "Choose a realistic limit for convenience meals rather than assuming every meal is optional.",
                {
                    "matching_transactions": int(len(food_spend)),
                    "observed_food_dining_total": round(food_total, 2),
                    "assumed_reduction_rate": assumed_reduction_rate,
                    "matched_by": "merchant/narration keywords or category fallback",
                },
                start,
                end,
                saving=saving,
                priority="high" if food_total >= 2000 else "medium",
                action="Try planning meals, comparing delivery fees, and setting a weekly takeout budget.",
            )
        )

    # 3. Frequent lower-value purchases in potentially flexible categories.
    small = expenses.loc[
        expenses["final_category"].isin(FLEXIBLE_CATEGORIES)
        & expenses["amount"].between(0, small_purchase_limit, inclusive="both")
    ]
    if len(small) >= minimum_small_purchases:
        small_total = float(small["amount"].sum())
        results.append(
            _make_suggestion(
                "frequent_small_purchases",
                "Review frequent small purchases",
                "Repeated lower-value expenses can add up. Review their purpose before deciding which, if any, to reduce.",
                {
                    "observations": int(len(small)),
                    "categories": sorted(small["final_category"].unique().tolist()),
                    "amount_limit": small_purchase_limit,
                    "observed_total": round(small_total, 2),
                    "assumed_reduction_rate": assumed_reduction_rate,
                },
                start,
                end,
                saving=small_total * assumed_reduction_rate,
                priority="medium",
                action="Pick one adjustable habit and test a weekly spending cap.",
            )
        )

    # 4. Monthly category increases, using only observed full calendar months.
    expenses["month"] = expenses["transaction_date"].dt.to_period("M").astype(str)
    full_months = []
    for _, month_frame in expenses.groupby("month", sort=True):
        if _is_full_calendar_month(month_frame):
            full_months.append(month_frame)
    if len(full_months) >= 2:
        month_totals = (
            pd.concat(full_months)
            .groupby(["month", "final_category"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )
        previous, current = month_totals.iloc[-2], month_totals.iloc[-1]
        for category in current.index.union(previous.index):
            before = float(previous.get(category, 0))
            now = float(current.get(category, 0))
            difference = now - before
            if before > 0 and now > before * (1 + increase_rate_threshold) and difference >= minimum_increase_amount:
                flexible = str(category) in FLEXIBLE_CATEGORIES
                results.append(
                    _make_suggestion(
                        "category_increase",
                        f"{category} spending increased",
                        "Recorded spending rose compared with the previous eligible complete month. "
                        "Check whether the change came from a one-off purchase or a repeated habit.",
                        {
                            "category": str(category),
                            "current_month": str(month_totals.index[-1]),
                            "current_amount": round(now, 2),
                            "baseline_month": str(month_totals.index[-2]),
                            "baseline_amount": round(before, 2),
                            "change_amount": round(difference, 2),
                            "change_percent": round(difference / before * 100, 1),
                        },
                        start,
                        end,
                        saving=difference * assumed_reduction_rate if flexible else None,
                        priority="high" if difference >= 2000 else "medium",
                        action="Inspect the largest transactions in this category and set a next-month limit if appropriate.",
                    )
                )

    # 5. Three-month rising trend.
    if len(full_months) >= 3:
        month_totals = (
            pd.concat(full_months)
            .groupby(["month", "final_category"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )
        last_three = month_totals.tail(3)
        for category in last_three.columns:
            values = last_three[category].astype(float).tolist()
            if values[0] < values[1] < values[2] and values[2] >= values[0] * 1.25:
                flexible = str(category) in FLEXIBLE_CATEGORIES
                results.append(
                    _make_suggestion(
                        "sustained_trend",
                        f"{category} spending rose for three months",
                        "Spending increased in each of the last three eligible complete months. "
                        "This is a short observed trend, not a prediction.",
                        {
                            "category": str(category),
                            "months": last_three.index.tolist(),
                            "monthly_amounts": [round(v, 2) for v in values],
                            "increase_from_first_percent": round(
                                (values[2] - values[0]) / values[0] * 100, 1
                            ) if values[0] else None,
                        },
                        start,
                        end,
                        saving=(values[2] - values[0]) * assumed_reduction_rate if flexible else None,
                        priority="high",
                        action="Compare the transactions across all three months and decide whether the trend is sustainable.",
                    )
                )

    # 6. Unusually high-spend day using the 1.5 x IQR rule.
    daily = expenses.assign(
        day=expenses["transaction_date"].dt.date.astype(str)
    ).groupby("day")["amount"].sum()
    if len(daily) >= 14:
        q1, q3 = np.quantile(daily.to_numpy(), [0.25, 0.75])
        iqr = q3 - q1
        high_day = str(daily.idxmax())
        high_amount = float(daily.loc[high_day])
        threshold = float(q3 + 1.5 * iqr)
        if high_amount > threshold:
            results.append(
                _make_suggestion(
                    "high_spend_day",
                    "Review an unusually high-spending day",
                    "One day's total is unusually high compared with the observed daily distribution. "
                    "Check for a one-off bill, travel, a large purchase, or duplicate records.",
                    {
                        "date": high_day,
                        "amount": round(high_amount, 2),
                        "daily_median": round(float(daily.median()), 2),
                        "outlier_threshold": round(threshold, 2),
                        "observed_days": int(len(daily)),
                        "method": "1.5 x IQR",
                    },
                    start,
                    end,
                    priority="medium",
                    action="Review transactions on this date before treating the amount as avoidable spending.",
                )
            )

    # 7. Potential recurring payments.
    recurring = detect_recurring_transactions(ledger)
    if recurring is not None and not recurring.empty:
        recurring_rows = recurring.loc[
            recurring["status"].isin(
                {"Regular cadence observed (3+ dates)", "Probable pattern"}
            )
        ].head(5)
        for row in recurring_rows.to_dict(orient="records"):
            results.append(
                _make_suggestion(
                    "repeated_merchant",
                    f"Review repeated payments to {row.get('merchant', 'this merchant')}",
                    "A repeated payment pattern was detected. Confirm whether it is a subscription, bill, transfer, or another legitimate recurring expense.",
                    {
                        "merchant": row.get("merchant"),
                        "observations": row.get("observations"),
                        "first_seen": row.get("first_seen"),
                        "last_seen": row.get("last_seen"),
                        "median_interval_days": row.get("median_interval_days"),
                        "pattern_status": row.get("status"),
                    },
                    start,
                    end,
                    priority="medium",
                    action="Check whether the service is still needed and review the actual cancellation terms before changing it.",
                )
            )

    # 8. Shopping / discretionary merchant review.
    shopping_keywords = (
        "amazon", "flipkart", "myntra", "ajio", "shopping", "mall",
        "fashion", "apparel", "electronics", "meesho"
    )
    shopping_match = expenses["_merchant_text"].str.contains(
        "|".join(re.escape(word) for word in shopping_keywords), regex=True, na=False
    )
    if not shopping_match.any():
        shopping_match = expenses["final_category"].str.lower().str.contains(
            r"shopping|fashion|apparel|personal purchase", regex=True, na=False
        )
    shopping = expenses.loc[shopping_match]
    if len(shopping) >= 2:
        shopping_total = float(shopping["amount"].sum())
        results.append(
            _make_suggestion(
                "shopping_review",
                "Set a planned shopping budget",
                "Several transactions matched shopping-related merchants or categories. The records do not establish whether purchases were necessary.",
                {
                    "matching_transactions": int(len(shopping)),
                    "observed_total": round(shopping_total, 2),
                    "share_of_consumption_percent": round(shopping_total / total_spend * 100, 1),
                    "assumed_reduction_rate": assumed_reduction_rate,
                },
                start,
                end,
                saving=shopping_total * assumed_reduction_rate,
                priority="medium",
                action="Use a 24-hour pause for non-urgent purchases and compare the total with your own budget.",
            )
        )

    # 9. Transport spending review.
    transport = expenses.loc[
        expenses["final_category"].str.lower().str.contains(
            r"transport|travel|fuel|cab|taxi|commut", regex=True, na=False
        )
        | expenses["_merchant_text"].str.contains(
            r"uber|ola|rapido|petrol|fuel|metro|bus pass", regex=True, na=False
        )
    ]
    if len(transport) >= 3:
        transport_total = float(transport["amount"].sum())
        results.append(
            _make_suggestion(
                "transport_review",
                "Compare your transport costs",
                "Multiple transport-related transactions were found. Some trips may be essential, so review convenience trips separately from necessary travel.",
                {
                    "matching_transactions": int(len(transport)),
                    "observed_total": round(transport_total, 2),
                    "share_of_consumption_percent": round(transport_total / total_spend * 100, 1),
                },
                start,
                end,
                saving=transport_total * assumed_reduction_rate,
                priority="low",
                action="Where practical, compare public transport, shared rides, and trip grouping.",
            )
        )

    # 10. Category-specific flexible recommendations from the existing module.
    try:
        flexible = build_recommendations(ledger, assumed_reduction_rate)
        if flexible is not None and not flexible.empty:
            for row in flexible.head(3).to_dict(orient="records"):
                estimated = float(row.get("estimated_monthly_difference", 0) or 0)
                results.append(
                    _make_suggestion(
                        "flexible_category",
                        f"Review {row.get('category', 'flexible category')} spending",
                        "This category is treated as potentially adjustable only as a broad proxy. "
                        "Review the transaction purpose before changing your spending.",
                        {
                            "category": row.get("category"),
                            "observed_total": round(float(row.get("observed_total", 0) or 0), 2),
                            "observed_months": int(row.get("observed_months", 0) or 0),
                            "assumed_reduction_rate": assumed_reduction_rate,
                            "estimated_monthly_difference": round(estimated, 2),
                        },
                        start,
                        end,
                        saving=estimated,
                        priority="medium",
                        action="Choose a small, realistic reduction and compare the next month's results.",
                    )
                )
    except (KeyError, TypeError, ValueError):
        pass

    # 11. Goal contribution calculation.
    if goals is not None and not goals.empty:
        required_columns = {
            "id", "title", "target_date", "target_amount_minor", "current_saved_minor"
        }
        if required_columns.issubset(goals.columns):
            for goal in goals.to_dict(orient="records"):
                target = pd.to_datetime(goal["target_date"], errors="coerce")
                if pd.isna(target) or target.date() <= date.today():
                    continue
                months = max(
                    (target.to_period("M") - pd.Timestamp(date.today()).to_period("M")).n + 1,
                    1,
                )
                remaining_minor = max(
                    int(goal["target_amount_minor"]) - int(goal["current_saved_minor"]), 0
                )
                required_minor = math.ceil(remaining_minor / months)
                results.append(
                    _make_suggestion(
                        "goal_contribution",
                        f"Plan contributions for {goal['title']}",
                        "This is the arithmetic monthly amount required by the stored target and deadline. Affordability is not assessed.",
                        {
                            "goal_id": str(goal["id"]),
                            "goal_name": str(goal["title"]),
                            "target_amount": int(goal["target_amount_minor"]) / 100,
                            "current_saved": int(goal["current_saved_minor"]) / 100,
                            "remaining_amount": remaining_minor / 100,
                            "months_remaining": months,
                            "required_monthly_contribution": required_minor / 100,
                        },
                        start,
                        end,
                        priority="high",
                        action="Compare the required contribution with your actual available monthly surplus before committing.",
                    )
                )

    # Rank suggestions so high-impact, actionable items appear first.
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    results.sort(
        key=lambda item: (
            priority_rank.get(item.get("priority", "medium"), 1),
            -(item.get("estimated_potential_saving_minor") or 0),
            item["title"],
        )
    )

    # Avoid duplicate titles and return a manageable set for the UI.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        key = item["title"].casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:max_suggestions]
