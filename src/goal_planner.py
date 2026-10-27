from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any

import pandas as pd

from src.common import BASE_CATEGORIES
from src.financial_intelligence import (
    ESSENTIAL_PROXY_CATEGORIES,
    FLEXIBLE_CATEGORIES,
    calculate_goal,
    canonicalize_transactions,
)


MINIMUM_HISTORY_MONTHS = 2
MINIMUM_CLASSIFICATION_CONFIDENCE = 0.65
CATEGORY_REDUCTION_LIMITS = {
    "Food & Dining": (0.10, 1.0),
    "Shopping": (0.08, 1.5),
    "Entertainment": (0.10, 1.0),
    "Personal Care": (0.05, 2.0),
}


def rupees_to_minor(value: str | int | float | Decimal) -> int:
    try:
        amount = Decimal(str(value).replace(",", "").replace("₹", "").strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Enter a valid rupee amount.") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("Amounts must be finite and non-negative.")
    if amount.as_tuple().exponent < -2:
        raise ValueError("Enter no more than two decimal places.")
    return int((amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def minor_to_rupees(amount_minor: int | None) -> float | None:
    return None if amount_minor is None else int(amount_minor) / 100


def _amount_minor(row: pd.Series) -> int:
    if "amount_minor" in row and pd.notna(row["amount_minor"]):
        amount = int(row["amount_minor"])
        if amount < 0:
            raise ValueError("Transaction amounts must be non-negative.")
        return amount
    value = Decimal(str(row["amount"]))
    if not value.is_finite() or value < 0:
        raise ValueError("Transaction amounts must be finite and non-negative.")
    return int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def estimate_monthly_spending(frame: pd.DataFrame) -> dict[str, Any]:
    """Estimate monthly consumption in paise; observed dates are not coverage proof."""
    if frame is None or frame.empty:
        return {
            "estimated_monthly_expenses_minor": None,
            "observed_consumption_minor": 0,
            "observed_month_count": 0,
            "months_with_transactions": 0,
            "observed_period_start": None,
            "observed_period_end": None,
            "coverage_verified": False,
            "coverage_status": "No transaction history is available.",
            "unclassified_debit_count": 0,
            "low_confidence_count": 0,
            "assessment_ready": False,
            "categories": {},
        }

    ledger = canonicalize_transactions(frame).copy()
    ledger["amount_minor_for_plan"] = ledger.apply(_amount_minor, axis=1)
    if "transaction_semantics" in ledger.columns:
        semantics = ledger["transaction_semantics"].fillna("").astype(str)
        consumption = semantics.isin({"consumption", "unknown", ""})
    else:
        consumption = ledger["is_consumption_expense"]
    consumption &= ledger["is_consumption_expense"]
    non_consumption_category = ledger["final_category"].str.casefold().str.contains(
        r"transfer|refund|reversal|cashback|salary|income|account movement",
        regex=True,
        na=False,
    )
    description_columns = [
        column
        for column in ("description", "transaction_details")
        if column in ledger.columns
    ]
    if description_columns:
        refund_text = (
            ledger[description_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.casefold()
            .str.contains(r"\brefund\b|\breversal\b|\bcashback\b", regex=True)
        )
        non_consumption_category |= refund_text
    consumption &= ~non_consumption_category

    debit = ledger["transaction_type"].eq("DEBIT")
    unclassified = debit & (
        ledger["final_category"].isin({"Needs Review", "Salary/Income"})
        | (
            ledger["transaction_semantics"].fillna("").eq("unclassified")
            if "transaction_semantics" in ledger.columns
            else False
        )
    )
    expenses = ledger.loc[consumption & ledger["amount_minor_for_plan"].gt(0)].copy()
    dates = pd.to_datetime(expenses["transaction_date"], errors="coerce")
    month_periods = dates.dt.to_period("M")
    all_dates = pd.to_datetime(ledger["transaction_date"], errors="coerce")
    observed_periods = all_dates.dt.to_period("M").dropna()
    observed_month_count = (
        int(observed_periods.max().ordinal - observed_periods.min().ordinal + 1)
        if not observed_periods.empty
        else 0
    )
    months_with_transactions = int(observed_periods.nunique())
    observed_total = int(expenses["amount_minor_for_plan"].sum())
    starts = all_dates.dropna()
    period_start = starts.min().date().isoformat() if not starts.empty else None
    period_end = starts.max().date().isoformat() if not starts.empty else None

    low_confidence_mask = pd.Series(False, index=expenses.index)
    if "ml_confidence" in expenses:
        confidence = pd.to_numeric(expenses["ml_confidence"], errors="coerce")
        low_confidence_mask = confidence.isna() | confidence.lt(
            MINIMUM_CLASSIFICATION_CONFIDENCE
        )
        if "category_source" in expenses:
            corrected = expenses["category_source"].astype(str).str.contains(
                "user", case=False, na=False
            )
            low_confidence_mask &= ~corrected
    else:
        low_confidence_mask[:] = True
    low_confidence_count = int(low_confidence_mask.sum())

    monthly_total = (
        (observed_total + observed_month_count - 1) // observed_month_count
        if observed_month_count >= MINIMUM_HISTORY_MONTHS
        else None
    )
    category_results: dict[str, dict[str, Any]] = {}
    if not expenses.empty:
        expenses["_month"] = month_periods.astype(str)
        for category, rows in expenses.groupby("final_category", sort=True):
            category_months = int(rows["_month"].nunique())
            category_total = int(rows["amount_minor_for_plan"].sum())
            baseline = (
                (category_total + observed_month_count - 1) // observed_month_count
                if observed_month_count >= MINIMUM_HISTORY_MONTHS
                else None
            )
            category_confidence: float | None = None
            if "ml_confidence" in rows:
                values = pd.to_numeric(rows["ml_confidence"], errors="coerce").dropna()
                if not values.empty:
                    category_confidence = float(values.mean())
            merchant_samples: list[str] = []
            merchant_column = next(
                (column for column in ("merchant", "description", "transaction_details")
                 if column in rows),
                None,
            )
            if merchant_column is not None:
                merchant_samples = sorted(
                    {
                        " ".join(str(value).split())[:100]
                        for value in rows[merchant_column].dropna().tolist()
                        if str(value).strip()
                    }
                )[:5]
            reliable = (
                not bool(low_confidence_mask.loc[rows.index].any())
                and category_months >= MINIMUM_HISTORY_MONTHS
            )
            category_results[str(category)] = {
                "observed_total_minor": category_total,
                "transaction_count": int(len(rows)),
                "active_month_count": category_months,
                "baseline_monthly_spend_minor": baseline,
                "mean_classification_confidence": category_confidence,
                "classification_reliable": reliable,
                "merchant_samples": merchant_samples,
                "period_start": rows["transaction_date"].min().date().isoformat(),
                "period_end": rows["transaction_date"].max().date().isoformat(),
            }

    unclassified_count = int(unclassified.sum())
    has_unverified_coverage = not bool(
        "coverage_validated" in frame.columns
        and frame["coverage_validated"].fillna(False).astype(bool).all()
    )
    history_sufficient = observed_month_count >= MINIMUM_HISTORY_MONTHS
    coverage_verified = not has_unverified_coverage
    assessment_ready = (
        history_sufficient
        and monthly_total is not None
        and unclassified_count == 0
        and coverage_verified
    )
    if not history_sufficient:
        coverage_status = (
            f"Only {observed_month_count} calendar month(s) in the observed date range; "
            f"at least {MINIMUM_HISTORY_MONTHS} are required for a monthly estimate."
        )
    elif unclassified_count:
        coverage_status = (
            f"{unclassified_count} debit transaction(s) are unclassified; "
            "correct their categories before assessing affordability."
        )
    elif has_unverified_coverage:
        coverage_status = (
            f"{observed_month_count} calendar month(s) in the observed date range, "
            "but the statement "
            "coverage is unverified. Observed first/last dates do not prove a complete month."
        )
    else:
        coverage_status = (
            f"{observed_month_count} calendar month(s) in the observed date range "
            "with user-confirmed coverage."
        )
    return {
        "estimated_monthly_expenses_minor": monthly_total,
        "observed_consumption_minor": observed_total,
        "observed_month_count": observed_month_count,
        "months_with_transactions": months_with_transactions,
        "observed_period_start": period_start,
        "observed_period_end": period_end,
        "coverage_verified": coverage_verified,
        "coverage_status": coverage_status,
        "unclassified_debit_count": unclassified_count,
        "low_confidence_count": low_confidence_count,
        "assessment_ready": assessment_ready,
        "categories": category_results,
    }


def calculate_goal_affordability(
    goal: dict[str, Any],
    profile: dict[str, Any],
    spending: dict[str, Any],
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Assess one saved goal using paise and the existing inclusive-month convention."""
    as_of = as_of or date.today()
    target_minor = int(goal["target_amount_minor"])
    saved_minor = int(goal["current_saved_minor"])
    if target_minor < 0 or saved_minor < 0:
        raise ValueError("Goal amounts must be non-negative.")
    for field in (
        "monthly_income_minor",
        "emergency_balance_minor",
        "emergency_target_minor",
        "emergency_contribution_minor",
        "additional_commitments_minor",
        "protected_buffer_contribution_minor",
    ):
        value = profile.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{field} must be a non-negative whole number of paise.")
    remaining_minor = max(target_minor - saved_minor, 0)
    target_date = pd.to_datetime(goal["target_date"], errors="coerce")
    if pd.isna(target_date):
        raise ValueError("The goal deadline is invalid.")
    deadline = target_date.date()
    if remaining_minor == 0:
        months_remaining = 0
        required_minor = 0
        goal_state = "complete"
    elif deadline <= as_of:
        months_remaining = 0
        required_minor = None
        goal_state = "expired"
    else:
        months_remaining = (
            (deadline.year - as_of.year) * 12 + deadline.month - as_of.month + 1
        )
        # Reuse the repository's existing goal arithmetic; round the result upward
        # to a whole paise for the planner's integer monetary representation.
        existing_calculation = calculate_goal(
            target_minor / 100,
            saved_minor / 100,
            months_remaining,
        )
        remaining_minor = int(
            (Decimal(str(existing_calculation["remaining_amount"])) * 100).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        required_minor = (remaining_minor + months_remaining - 1) // months_remaining
        goal_state = "active"

    income = profile.get("monthly_income_minor")
    expenses = spending.get("estimated_monthly_expenses_minor")
    commitments = profile.get("additional_commitments_minor")
    emergency_contribution = profile.get("emergency_contribution_minor")
    buffer_contribution = profile.get("protected_buffer_contribution_minor")
    required_inputs = (
        income,
        expenses,
        commitments,
        emergency_contribution,
        buffer_contribution,
    )
    assessment_ready = bool(spending.get("assessment_ready")) and all(
        value is not None for value in required_inputs
    )
    safe_surplus = (
        max(
            0,
            int(income)
            - int(expenses)
            - int(commitments)
            - int(emergency_contribution)
            - int(buffer_contribution),
        )
        if assessment_ready
        else None
    )
    funding_gap = (
        max(required_minor - safe_surplus, 0)
        if required_minor is not None and safe_surplus is not None
        else None
    )
    emergency_balance = profile.get("emergency_balance_minor")
    emergency_target = profile.get("emergency_target_minor")
    emergency_shortfall = (
        max(int(emergency_target) - int(emergency_balance), 0)
        if emergency_target is not None and emergency_balance is not None
        else None
    )
    return {
        "goal_state": goal_state,
        "target_amount_minor": target_minor,
        "current_saved_minor": saved_minor,
        "remaining_goal_amount_minor": remaining_minor,
        "target_date": deadline.isoformat(),
        "months_remaining": months_remaining,
        "required_monthly_contribution_minor": required_minor,
        "monthly_income_minor": income,
        "estimated_monthly_expenses_minor": expenses,
        "additional_commitments_minor": commitments,
        "emergency_contribution_minor": emergency_contribution,
        "protected_buffer_contribution_minor": buffer_contribution,
        "safe_surplus_minor": safe_surplus,
        "funding_gap_minor": funding_gap,
        "emergency_balance_minor": emergency_balance,
        "emergency_target_minor": emergency_target,
        "emergency_reserve_shortfall_minor": emergency_shortfall,
        "assessment_ready": assessment_ready,
        "assessment_missing_inputs": [
            name
            for name, value in (
                ("confirmed monthly take-home income", income),
                ("estimated monthly expenses with sufficient history", expenses),
                ("additional monthly commitments", commitments),
                ("planned emergency contribution", emergency_contribution),
                ("protected monthly buffer contribution", buffer_contribution),
            )
            if value is None
        ],
    }


def _allocation(
    candidates: list[dict[str, Any]], funding_gap_minor: int
) -> tuple[list[dict[str, Any]], int]:
    remaining = max(funding_gap_minor, 0)
    available = [
        item for item in candidates if item.get("status") != "rejected"
    ]
    total_capacity = sum(int(item["max_reduction_minor"]) for item in available)
    for item in sorted(
        available,
        key=lambda candidate: (
            float(candidate["disruption_weight"]),
            candidate["category"],
        ),
    ):
        planned = min(int(item["max_reduction_minor"]), remaining)
        item["recommended_reduction_minor"] = planned
        baseline = int(item["baseline_monthly_spend_minor"])
        item["reduction_rate"] = planned / baseline if baseline else 0.0
        remaining -= planned
    for item in candidates:
        item.setdefault("recommended_reduction_minor", 0)
        baseline = int(item["baseline_monthly_spend_minor"])
        item["reduction_rate"] = (
            item["recommended_reduction_minor"] / baseline if baseline else 0.0
        )
    return candidates, total_capacity


def build_goal_savings_plan(
    goal: dict[str, Any],
    profile: dict[str, Any],
    spending: dict[str, Any],
    suggestions: list[dict[str, Any]] | None = None,
    preferences: dict[str, str] | None = None,
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    assessment = calculate_goal_affordability(
        goal, profile, spending, as_of=as_of
    )
    preferences = preferences or {}
    if any(value not in {"approved", "rejected"} for value in preferences.values()):
        raise ValueError("Goal-plan preferences must be approved or rejected.")
    user_protected = profile.get("protected_categories", [])
    if (
        not isinstance(user_protected, list)
        or not all(isinstance(category, str) for category in user_protected)
        or not set(user_protected).issubset(BASE_CATEGORIES)
    ):
        raise ValueError("Choose supported categories to protect.")
    suggestion_context = suggestions or []
    protected = set(ESSENTIAL_PROXY_CATEGORIES)
    protected.update(user_protected)
    candidates: list[dict[str, Any]] = []
    for category, facts in spending.get("categories", {}).items():
        rule = CATEGORY_REDUCTION_LIMITS.get(category)
        baseline = facts.get("baseline_monthly_spend_minor")
        if (
            rule is None
            or baseline is None
            or int(baseline) <= 0
            or category in protected
            or category not in FLEXIBLE_CATEGORIES
            or not facts.get("classification_reliable")
        ):
            continue
        limit_rate, weight = rule
        maximum = int(
            (Decimal(int(baseline)) * Decimal(str(limit_rate))).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        related_suggestions = [
            str(item.get("title", ""))
            for item in suggestion_context
            if (
                str(item.get("evidence", {}).get("category", "")) == category
                or (
                    category == "Food & Dining"
                    and item.get("category") == "food_and_dining"
                )
                or (
                    category == "Shopping"
                    and "shopping" in str(item.get("title", "")).casefold()
                )
            )
        ]
        candidates.append(
            {
                "category": category,
                "observed_total_minor": int(facts["observed_total_minor"]),
                "observed_month_count": int(spending["observed_month_count"]),
                "transaction_count": int(facts["transaction_count"]),
                "baseline_monthly_spend_minor": int(baseline),
                "max_reduction_minor": maximum,
                "max_reduction_rate": limit_rate,
                "eligibility_reason": (
                    f"{category} is in the planner's explicitly configured flexible "
                    "category rules; essential and user-protected categories are excluded."
                ),
                "recommended_reduction_minor": 0,
                "reduction_rate": 0.0,
                "disruption_weight": weight,
                "classification_confidence": facts.get(
                    "mean_classification_confidence"
                ),
                "merchant_samples": facts["merchant_samples"],
                "status": preferences.get(category, "undecided"),
                "period_start": facts["period_start"],
                "period_end": facts["period_end"],
                "evidence": (
                    f"{facts['transaction_count']} classified debit transaction(s), "
                    f"active in {facts['active_month_count']} of "
                    f"{spending['observed_month_count']} calendar month(s), "
                    f"INR {facts['observed_total_minor'] / 100:,.2f} observed across "
                    f"the analyzed date range; merchant/narration samples: "
                    f"{', '.join(facts['merchant_samples']) or 'unavailable'}. "
                    "The monthly baseline is normalized over the date range. "
                    + (
                        "Statement coverage was user-confirmed."
                        if spending["coverage_verified"]
                        else "Statement coverage is not verified."
                    )
                ),
                "uncertainty": (
                    "Category flexibility and the reduction cap are conservative "
                    "planning rules, not learned preferences. Confirm each purchase "
                    "is adjustable. Category-level activity does not prove a specific "
                    "merchant payment is recurring or non-essential."
                ),
                "related_suggestions": related_suggestions,
            }
        )

    funding_gap = assessment["funding_gap_minor"]
    if funding_gap is not None:
        candidates, total_capacity = _allocation(candidates, int(funding_gap))
    else:
        total_capacity = sum(int(item["max_reduction_minor"]) for item in candidates)
    potential_total = sum(
        int(item["recommended_reduction_minor"]) for item in candidates
    )
    approved_total = sum(
        int(item["recommended_reduction_minor"])
        for item in candidates
        if item["status"] == "approved"
    )
    required_minor = assessment["required_monthly_contribution_minor"]
    safe_surplus = assessment["safe_surplus_minor"]
    if assessment["goal_state"] == "complete":
        status = "Goal already funded"
    elif assessment["goal_state"] == "expired":
        status = "Expired goal deadline"
    elif not assessment["assessment_ready"] or funding_gap is None:
        status = "Not assessable because important financial information is missing"
    elif int(funding_gap) == 0:
        status = "Affordable under the stated assumptions"
    elif total_capacity >= int(funding_gap):
        status = "Potentially affordable after feasible spending adjustments"
    else:
        status = "Partially feasible, with a remaining monthly shortfall"
    after_approval = (
        max(int(required_minor) - int(safe_surplus) - approved_total, 0)
        if required_minor is not None and safe_surplus is not None
        else None
    )
    plan = {
        **assessment,
        "status": status,
        "candidates": candidates,
        "protected_categories": sorted(protected),
        "maximum_eligible_adjustments_minor": total_capacity,
        "proposed_adjustments_minor": potential_total,
        "approved_adjustments_minor": approved_total,
        "planned_goal_contribution_minor": (
            int(safe_surplus) + approved_total if safe_surplus is not None else None
        ),
        "remaining_shortfall_after_approval_minor": after_approval,
        "remaining_shortfall_if_all_proposed_minor": (
            max(int(funding_gap) - potential_total, 0)
            if funding_gap is not None
            else None
        ),
        "optimisation_method": (
            "Deterministic greedy allocation in ascending disruption-weight order; "
            "ties are broken by category name. Category caps are explicit heuristics."
        ),
        "coverage_status": spending.get("coverage_status", "Unknown coverage."),
    }
    validate_plan_constraints(plan)
    return plan


def validate_plan_constraints(plan: dict[str, Any]) -> bool:
    candidates = plan.get("candidates", [])
    protected = set(plan.get("protected_categories", []))
    categories = [item["category"] for item in candidates]
    if len(categories) != len(set(categories)):
        raise ValueError("A spending category appears more than once in the plan.")
    for item in candidates:
        maximum = int(item["max_reduction_minor"])
        reduction = int(item["recommended_reduction_minor"])
        baseline = int(item["baseline_monthly_spend_minor"])
        if maximum < 0 or reduction < 0 or reduction > maximum:
            raise ValueError(f"Invalid allocation for {item['category']}.")
        if reduction > baseline:
            raise ValueError(f"Allocation exceeds observed baseline for {item['category']}.")
        rule = CATEGORY_REDUCTION_LIMITS.get(item["category"])
        if rule is None:
            raise ValueError(f"Category {item['category']} has no configured reduction limit.")
        permitted_maximum = int(
            (Decimal(baseline) * Decimal(str(rule[0]))).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        if maximum > permitted_maximum:
            raise ValueError(f"Configured cap exceeded for {item['category']}.")
        if item["category"] in ESSENTIAL_PROXY_CATEGORIES or item["category"] in protected:
            raise ValueError(f"Protected category {item['category']} received an allocation.")
    proposed = sum(int(item["recommended_reduction_minor"]) for item in candidates)
    if proposed != int(plan["proposed_adjustments_minor"]):
        raise ValueError("Reported proposed adjustments do not match the allocation.")
    approved = sum(
        int(item["recommended_reduction_minor"])
        for item in candidates
        if item.get("status") == "approved"
    )
    if approved != int(plan["approved_adjustments_minor"]):
        raise ValueError("Reported approved adjustments do not match the allocation.")
    expected_capacity = sum(
        int(item["max_reduction_minor"])
        for item in candidates
        if item.get("status") != "rejected"
    )
    if expected_capacity != int(plan["maximum_eligible_adjustments_minor"]):
        raise ValueError("Reported category capacity does not match the permitted caps.")
    if any(
        item.get("status") == "rejected"
        and int(item["recommended_reduction_minor"]) != 0
        for item in candidates
    ):
        raise ValueError("Rejected categories cannot receive a planned reduction.")
    gap = plan.get("funding_gap_minor")
    if gap == 0 and any(int(item["recommended_reduction_minor"]) for item in candidates):
        raise ValueError("A zero funding gap must not generate a spending reduction.")
    if gap is not None and plan["remaining_shortfall_if_all_proposed_minor"] != max(
        int(gap) - proposed, 0
    ):
        raise ValueError("Reported shortfall does not reconcile with the allocation.")
    required = plan["required_monthly_contribution_minor"]
    safe = plan["safe_surplus_minor"]
    if required is not None and safe is not None:
        expected_contribution = int(safe) + approved
        expected_shortfall = max(int(required) - expected_contribution, 0)
        if plan["planned_goal_contribution_minor"] != expected_contribution:
            raise ValueError("Reported planned contribution does not reconcile.")
        if plan["remaining_shortfall_after_approval_minor"] != expected_shortfall:
            raise ValueError("Reported after-approval shortfall does not reconcile.")
    expected_status = (
        "Goal already funded"
        if plan["goal_state"] == "complete"
        else "Expired goal deadline"
        if plan["goal_state"] == "expired"
        else "Not assessable because important financial information is missing"
        if not plan["assessment_ready"] or gap is None
        else "Affordable under the stated assumptions"
        if int(gap) == 0
        else "Potentially affordable after feasible spending adjustments"
        if int(plan["maximum_eligible_adjustments_minor"]) >= int(gap)
        else "Partially feasible, with a remaining monthly shortfall"
    )
    if plan["status"] != expected_status:
        raise ValueError("Reported feasibility status does not match the plan totals.")
    return True


def reconcile_savings_plan(
    previous_plan: dict[str, Any] | None, current_plan: dict[str, Any]
) -> dict[str, Any]:
    """Compare plan baselines without treating changed spending as achieved savings."""
    if not previous_plan:
        return {
            "comparison_available": False,
            "message": "No prior saved plan is available for comparison.",
            "category_changes": [],
        }
    previous = {
        item["category"]: int(item["baseline_monthly_spend_minor"])
        for item in previous_plan.get("candidates", [])
    }
    current = {
        item["category"]: int(item["baseline_monthly_spend_minor"])
        for item in current_plan.get("candidates", [])
    }
    changes = [
        {
            "category": category,
            "previous_baseline_minor": previous.get(category),
            "current_baseline_minor": current.get(category),
            "change_minor": (
                current.get(category, 0) - previous.get(category, 0)
            ),
        }
        for category in sorted(set(previous) | set(current))
        if previous.get(category) != current.get(category)
    ]
    return {
        "comparison_available": True,
        "message": (
            "Observed baselines changed. This does not prove savings: statement "
            "coverage, one-off spending, and classification may also have changed."
        ),
        "category_changes": changes,
    }
