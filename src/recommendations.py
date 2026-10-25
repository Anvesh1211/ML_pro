from __future__ import annotations

import calendar
import math
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
    dates = pd.to_datetime(frame["transaction_date"], errors="coerce")
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _is_full_calendar_month(month: pd.DataFrame) -> bool:
    dates = pd.to_datetime(month["transaction_date"], errors="coerce")
    first = dates.min().date()
    last = dates.max().date()
    month_end = calendar.monthrange(first.year, first.month)[1]
    return first.day == 1 and last.day == month_end and first.month == last.month


def _recommendation(
    category: str,
    title: str,
    explanation: str,
    evidence: dict[str, Any],
    *,
    start: str,
    end: str,
    saving_minor: int | None = None,
) -> dict[str, Any]:
    return {
        "period_start": start,
        "period_end": end,
        "category": category,
        "title": title,
        "explanation": explanation,
        "evidence": evidence,
        "estimated_potential_saving_minor": saving_minor,
    }


def _to_minor_units(value: float) -> int:
    return int(
        (Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def build_personalized_suggestions(
    frame: pd.DataFrame,
    goals: pd.DataFrame | None = None,
    *,
    assumed_reduction_rate: float = 0.10,
) -> list[dict[str, Any]]:
    if not math.isfinite(assumed_reduction_rate) or not 0 <= assumed_reduction_rate <= 1:
        raise ValueError("assumed_reduction_rate must be between 0 and 1.")
    if frame.empty:
        return []

    ledger = canonicalize_transactions(frame)
    start, end = _period_bounds(ledger)
    expenses = ledger.loc[ledger["is_consumption_expense"]].copy()
    results: list[dict[str, Any]] = []
    if not expenses.empty:
        expenses["month"] = expenses["transaction_date"].dt.to_period("M").astype(str)
        full_months = [
            month
            for _period, month in expenses.groupby("month", sort=True)
            if _is_full_calendar_month(month)
        ]
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
                if before > 0 and now > before * 1.25 and difference >= 500:
                    results.append(
                        _recommendation(
                            "category_increase",
                            f"{category} spending increased",
                            "Recorded spending was higher than the previous complete "
                            "calendar month. This is a descriptive comparison, not a "
                            "judgment about whether the spending was necessary.",
                            {
                                "category": str(category),
                                "current_month": str(month_totals.index[-1]),
                                "current_amount": round(now, 2),
                                "baseline_month": str(month_totals.index[-2]),
                                "baseline_amount": round(before, 2),
                                "change_amount": round(difference, 2),
                                "change_percent": round(difference / before * 100, 1),
                                "comparison": "two complete observed calendar months",
                            },
                            start=start,
                            end=end,
                        )
                    )

        small_flexible = expenses.loc[
            expenses["final_category"].isin(FLEXIBLE_CATEGORIES)
            & expenses["amount"].between(0, 500, inclusive="both")
        ]
        if len(small_flexible) >= 5:
            observed_amount = float(small_flexible["amount"].sum())
            estimated = _to_minor_units(observed_amount * assumed_reduction_rate)
            results.append(
                _recommendation(
                    "frequent_small_purchases",
                    "Frequent smaller purchases observed",
                    "Several lower-value debits occurred in categories that can include "
                    "flexible purchases. Review their actual purpose before deciding "
                    "whether any are adjustable.",
                    {
                        "observations": int(len(small_flexible)),
                        "categories": sorted(
                            small_flexible["final_category"].unique().tolist()
                        ),
                        "amount_threshold": 500,
                        "observed_total": round(observed_amount, 2),
                        "assumed_reduction_rate": assumed_reduction_rate,
                        "estimate_basis": "hypothetical reduction of this observed total",
                    },
                    start=start,
                    end=end,
                    saving_minor=estimated,
                )
            )

        daily = expenses.assign(
            day=expenses["transaction_date"].dt.date.astype(str)
        ).groupby("day")["amount"].sum()
        if len(daily) >= 14:
            q1, q3 = np.quantile(daily.to_numpy(), [0.25, 0.75])
            high_day = daily.idxmax()
            high_amount = float(daily.loc[high_day])
            if high_amount > q3 + 1.5 * (q3 - q1):
                results.append(
                    _recommendation(
                        "high_spend_day",
                        "One day had unusually high recorded spending",
                        "This date is an outlier relative to the user's observed daily "
                        "spending distribution. Check the transactions and context; a "
                        "single high day is not evidence of a recurring issue.",
                        {
                            "date": high_day,
                            "amount": round(high_amount, 2),
                            "daily_median": round(float(daily.median()), 2),
                            "observed_days": int(len(daily)),
                            "method": "1.5×IQR on observed daily consumption totals",
                        },
                        start=start,
                        end=end,
                    )
                )

        recurring = detect_recurring_transactions(ledger)
        for row in recurring.loc[
            recurring["status"].isin(
                {"Regular cadence observed (3+ dates)", "Probable pattern"}
            )
        ].head(5).to_dict(orient="records"):
            results.append(
                _recommendation(
                    "repeated_merchant",
                    f"Repeated payments to {row['merchant']}",
                    "A repeated date pattern appears in this statement. It is only a "
                    "candidate for review and is not a confirmed subscription or future bill.",
                    {
                        "merchant": row["merchant"],
                        "observations": row["observations"],
                        "first_seen": row["first_seen"],
                        "last_seen": row["last_seen"],
                        "median_interval_days": row["median_interval_days"],
                        "pattern_status": row["status"],
                    },
                    start=start,
                    end=end,
                )
            )

        flexible = build_recommendations(ledger, assumed_reduction_rate)
        for row in flexible.head(3).to_dict(orient="records"):
            results.append(
                _recommendation(
                    "flexible_category",
                    f"Review recorded {row['category']} spending",
                    "The category is treated as potentially adjustable only as a broad "
                    "proxy. The estimate is hypothetical; confirm the transaction purpose "
                    "before changing spending.",
                    {
                        "category": row["category"],
                        "observed_total": round(float(row["observed_total"]), 2),
                        "observed_months": int(row["observed_months"]),
                        "assumed_reduction_rate": assumed_reduction_rate,
                        "estimated_difference": round(
                            float(row["estimated_monthly_difference"]), 2
                        ),
                    },
                    start=start,
                    end=end,
                    saving_minor=int(
                        _to_minor_units(
                            float(row["estimated_monthly_difference"])
                        )
                    ),
                )
            )

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
                    results.append(
                        _recommendation(
                            "sustained_trend",
                            f"{category} spending rose across three months",
                            "Recorded amounts increased in each of the last three "
                            "complete months. This short trend may not continue and does "
                            "not establish a cause.",
                            {
                                "category": str(category),
                                "months": last_three.index.tolist(),
                                "monthly_amounts": [round(value, 2) for value in values],
                                "evidence": "three complete observed calendar months",
                            },
                            start=start,
                            end=end,
                        )
                    )

    if goals is not None and not goals.empty:
        for goal in goals.to_dict(orient="records"):
            target = pd.to_datetime(goal["target_date"], errors="coerce")
            if pd.isna(target) or target.date() <= date.today():
                continue
            months = max(
                (target.to_period("M") - pd.Timestamp(date.today()).to_period("M")).n
                + 1,
                1,
            )
            remaining_minor = max(
                int(goal["target_amount_minor"]) - int(goal["current_saved_minor"]), 0
            )
            required_minor = math.ceil(remaining_minor / months)
            results.append(
                _recommendation(
                    "goal_contribution",
                    f"Plan contributions for {goal['title']}",
                    "This is the arithmetic monthly amount needed under the stored "
                    "goal and deadline. It does not assess affordability.",
                    {
                        "goal_id": str(goal["id"]),
                        "goal_name": str(goal["title"]),
                        "target_amount": int(goal["target_amount_minor"]) / 100,
                        "current_saved": int(goal["current_saved_minor"]) / 100,
                        "remaining_amount": remaining_minor / 100,
                        "months_remaining": months,
                        "required_monthly_contribution": required_minor / 100,
                        "affordability": "not assessed",
                    },
                    start=start,
                    end=end,
                )
            )

    return results
