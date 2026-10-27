from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.auth import register_user
from src.database import connect, initialize_database
from src.finance_store import (
    get_goal_plan_preferences,
    get_latest_goal_plan_snapshot,
    get_user_financial_profile,
    list_goal_plan_feedback,
    list_user_goals,
    record_goal_plan_feedback,
    save_goal_plan_snapshot,
    save_user_financial_profile,
    save_user_goal,
    set_goal_plan_adjustment_status,
)
from src.goal_planner import (
    build_goal_savings_plan,
    calculate_goal_affordability,
    estimate_monthly_spending,
    reconcile_savings_plan,
    rupees_to_minor,
    validate_plan_constraints,
)


def transaction(
    index: int,
    month: int,
    category: str = "Food & Dining",
    amount_minor: int = 10_000,
    *,
    confidence: float = 0.9,
    transaction_type: str = "DEBIT",
    semantics: str = "consumption",
    coverage_validated: bool = True,
) -> dict[str, object]:
    return {
        "transaction_id": f"tx-{index}",
        "transaction_date": f"2026-{month:02d}-15",
        "transaction_type": transaction_type,
        "amount": amount_minor / 100,
        "amount_minor": amount_minor,
        "merchant": f"merchant-{index}",
        "final_category": category,
        "category_source": "ML",
        "ml_confidence": confidence,
        "transaction_semantics": semantics,
        "coverage_validated": coverage_validated,
    }


def two_month_spending(
    rows: list[dict[str, object]] | None = None,
    *,
    coverage_validated: bool = True,
) -> dict[str, object]:
    if rows is None:
        rows = [
            transaction(1, 1),
            transaction(2, 2),
        ]
    frame = pd.DataFrame(rows)
    if "coverage_validated" not in frame:
        frame["coverage_validated"] = coverage_validated
    return estimate_monthly_spending(frame)


def complete_profile(**overrides: object) -> dict[str, object]:
    profile: dict[str, object] = {
        "monthly_income_minor": 1_000_000,
        "emergency_balance_minor": 50_000,
        "emergency_target_minor": 100_000,
        "emergency_contribution_minor": 10_000,
        "additional_commitments_minor": 20_000,
        "protected_buffer_contribution_minor": 5_000,
        "protected_categories": [],
    }
    profile.update(overrides)
    return profile


def goal(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": "goal-1",
        "title": "Trip",
        "target_amount_minor": 300_000,
        "current_saved_minor": 0,
        "target_date": "2026-03-20",
        "status": "active",
    }
    result.update(overrides)
    return result


class GoalPlannerCalculationTests(unittest.TestCase):
    def setUp(self):
        self.as_of = date(2026, 1, 15)
        self.spending = two_month_spending()

    def test_goal_already_fully_funded_has_no_required_contribution(self):
        result = calculate_goal_affordability(
            goal(target_amount_minor=10_000, current_saved_minor=15_000),
            complete_profile(),
            self.spending,
            as_of=self.as_of,
        )
        self.assertEqual(result["goal_state"], "complete")
        self.assertEqual(result["remaining_goal_amount_minor"], 0)
        self.assertEqual(result["required_monthly_contribution_minor"], 0)

    def test_future_deadline_counts_current_and_target_month_inclusively(self):
        result = calculate_goal_affordability(
            goal(), complete_profile(), self.spending, as_of=self.as_of
        )
        self.assertEqual(result["months_remaining"], 3)
        self.assertEqual(result["required_monthly_contribution_minor"], 100_000)

    def test_expired_deadline_does_not_divide_by_zero(self):
        result = calculate_goal_affordability(
            goal(target_date="2026-01-14"),
            complete_profile(),
            self.spending,
            as_of=self.as_of,
        )
        self.assertEqual(result["goal_state"], "expired")
        self.assertIsNone(result["required_monthly_contribution_minor"])

    def test_missing_income_prevents_affordability_claim(self):
        result = calculate_goal_affordability(
            goal(),
            complete_profile(monthly_income_minor=None),
            self.spending,
            as_of=self.as_of,
        )
        self.assertFalse(result["assessment_ready"])
        self.assertIsNone(result["safe_surplus_minor"])
        self.assertIn("confirmed monthly take-home income", result["assessment_missing_inputs"])

    def test_negative_financial_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calculate_goal_affordability(
                goal(),
                complete_profile(additional_commitments_minor=-1),
                self.spending,
                as_of=self.as_of,
            )

    def test_zero_financial_inputs_are_valid_explicit_values(self):
        result = calculate_goal_affordability(
            goal(),
            complete_profile(
                monthly_income_minor=0,
                emergency_contribution_minor=0,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            self.spending,
            as_of=self.as_of,
        )
        self.assertTrue(result["assessment_ready"])
        self.assertEqual(result["safe_surplus_minor"], 0)

    def test_emergency_fund_balance_is_not_monthly_income(self):
        result = calculate_goal_affordability(
            goal(),
            complete_profile(
                monthly_income_minor=100_000,
                emergency_balance_minor=900_000,
                emergency_contribution_minor=10_000,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            two_month_spending(
                [
                    transaction(1, 1, amount_minor=10_000),
                    transaction(2, 2, amount_minor=10_000),
                ]
            ),
            as_of=self.as_of,
        )
        self.assertEqual(result["safe_surplus_minor"], 80_000)
        self.assertEqual(result["emergency_reserve_shortfall_minor"], 0)

    def test_baseline_and_additional_commitments_are_deducted_once(self):
        spending = two_month_spending(
            [
                transaction(1, 1, amount_minor=20_000),
                transaction(2, 2, amount_minor=20_000),
            ]
        )
        result = calculate_goal_affordability(
            goal(),
            complete_profile(
                monthly_income_minor=100_000,
                additional_commitments_minor=5_000,
                emergency_contribution_minor=10_000,
                protected_buffer_contribution_minor=5_000,
            ),
            spending,
            as_of=self.as_of,
        )
        self.assertEqual(result["estimated_monthly_expenses_minor"], 20_000)
        self.assertEqual(result["safe_surplus_minor"], 60_000)

    def test_partial_statement_coverage_is_not_treated_as_complete(self):
        spending = two_month_spending(
            [
                transaction(1, 1, coverage_validated=False),
                transaction(2, 2, coverage_validated=False),
            ]
        )
        result = calculate_goal_affordability(
            goal(), complete_profile(), spending, as_of=self.as_of
        )
        self.assertIsNotNone(spending["estimated_monthly_expenses_minor"])
        self.assertFalse(spending["coverage_verified"])
        self.assertFalse(result["assessment_ready"])
        self.assertIsNone(result["safe_surplus_minor"])

    def test_calendar_gaps_are_included_in_month_normalization(self):
        spending = two_month_spending(
            [
                transaction(1, 1, amount_minor=10_000),
                transaction(2, 3, amount_minor=10_000),
            ]
        )
        self.assertEqual(spending["observed_month_count"], 3)
        self.assertEqual(spending["months_with_transactions"], 2)
        self.assertEqual(spending["estimated_monthly_expenses_minor"], 6_667)

    def test_transfers_credits_refunds_and_unclassified_rows_are_not_spend_cuts(self):
        rows = [
            transaction(1, 1, amount_minor=10_000),
            transaction(2, 2, amount_minor=10_000),
            transaction(
                3, 1, "Personal Transfer", 50_000, semantics="transfer"
            ),
            transaction(
                4,
                2,
                "Food & Dining",
                30_000,
                transaction_type="CREDIT",
                semantics="other_credit",
            ),
            transaction(5, 1, "Refund", 2_000),
            transaction(6, 2, "Needs Review", 4_000, semantics="unclassified"),
        ]
        spending = two_month_spending(rows)
        self.assertEqual(spending["observed_consumption_minor"], 20_000)
        self.assertEqual(spending["unclassified_debit_count"], 1)
        self.assertNotIn("Refund", spending["categories"])
        self.assertNotIn("Personal Transfer", spending["categories"])

    def test_essential_and_user_protected_categories_are_not_candidates(self):
        rows = [
            transaction(1, 1, "Grocery"),
            transaction(2, 2, "Grocery"),
            transaction(3, 1, "Shopping"),
            transaction(4, 2, "Shopping"),
        ]
        spending = two_month_spending(rows)
        plan = build_goal_savings_plan(
            goal(),
            complete_profile(protected_categories=["Shopping"]),
            spending,
            as_of=self.as_of,
        )
        self.assertEqual([item["category"] for item in plan["candidates"]], [])
        self.assertTrue(validate_plan_constraints(plan))

    def test_uncertain_classification_is_not_reduction_evidence(self):
        spending = two_month_spending(
            [
                transaction(1, 1, confidence=0.4),
                transaction(2, 2, confidence=0.9),
            ]
        )
        plan = build_goal_savings_plan(
            goal(), complete_profile(), spending, as_of=self.as_of
        )
        self.assertEqual(plan["candidates"], [])
        self.assertGreater(spending["low_confidence_count"], 0)

    def test_category_cap_rounds_down_to_a_whole_paise(self):
        spending = two_month_spending(
            [
                transaction(1, 1, amount_minor=6),
                transaction(2, 2, amount_minor=6),
            ]
        )
        plan = build_goal_savings_plan(
            goal(target_amount_minor=1_000),
            complete_profile(
                monthly_income_minor=0,
                emergency_contribution_minor=0,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            spending,
            as_of=self.as_of,
        )
        self.assertEqual(plan["candidates"][0]["max_reduction_minor"], 0)
        self.assertEqual(plan["candidates"][0]["recommended_reduction_minor"], 0)

    def test_overlapping_suggestions_still_create_one_category_candidate(self):
        suggestions = [
            {
                "category": "food_and_dining",
                "title": "Dining transactions",
                "evidence": {},
            },
            {
                "category": "frequent_small_purchases",
                "title": "Repeated dining purchases",
                "evidence": {"category": "Food & Dining"},
            },
        ]
        plan = build_goal_savings_plan(
            goal(),
            complete_profile(),
            self.spending,
            suggestions,
            as_of=self.as_of,
        )
        self.assertEqual(
            [item["category"] for item in plan["candidates"]], ["Food & Dining"]
        )

    def test_feasible_gap_is_closed_without_exceeding_category_cap(self):
        spending = two_month_spending(
            [
                transaction(1, 1, amount_minor=100_000),
                transaction(2, 2, amount_minor=100_000),
            ]
        )
        plan = build_goal_savings_plan(
            goal(target_amount_minor=30_000),
            complete_profile(
                monthly_income_minor=100_000,
                emergency_contribution_minor=0,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            spending,
            as_of=self.as_of,
        )
        self.assertEqual(plan["funding_gap_minor"], 10_000)
        self.assertEqual(plan["proposed_adjustments_minor"], 10_000)
        self.assertEqual(
            plan["status"], "Potentially affordable after feasible spending adjustments"
        )
        self.assertTrue(validate_plan_constraints(plan))
        approved_plan = build_goal_savings_plan(
            goal(target_amount_minor=30_000),
            complete_profile(
                monthly_income_minor=100_000,
                emergency_contribution_minor=0,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            spending,
            preferences={"Food & Dining": "approved"},
            as_of=self.as_of,
        )
        self.assertEqual(approved_plan["approved_adjustments_minor"], 10_000)
        self.assertEqual(approved_plan["remaining_shortfall_after_approval_minor"], 0)

    def test_infeasible_gap_reports_shortfall(self):
        spending = two_month_spending(
            [
                transaction(1, 1, amount_minor=10_000),
                transaction(2, 2, amount_minor=10_000),
            ]
        )
        plan = build_goal_savings_plan(
            goal(target_amount_minor=1_000_000),
            complete_profile(
                monthly_income_minor=10_000,
                emergency_contribution_minor=0,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            spending,
            as_of=self.as_of,
        )
        self.assertEqual(
            plan["status"], "Partially feasible, with a remaining monthly shortfall"
        )
        self.assertGreater(plan["remaining_shortfall_if_all_proposed_minor"], 0)

    def test_zero_gap_does_not_recommend_spending_cuts(self):
        plan = build_goal_savings_plan(
            goal(target_amount_minor=10_000),
            complete_profile(),
            self.spending,
            as_of=self.as_of,
        )
        self.assertEqual(plan["funding_gap_minor"], 0)
        self.assertEqual(plan["proposed_adjustments_minor"], 0)
        self.assertTrue(
            all(item["recommended_reduction_minor"] == 0 for item in plan["candidates"])
        )

    def test_rejected_candidate_is_zero_and_remainder_is_reallocated(self):
        spending = two_month_spending(
            [
                transaction(1, 1, "Food & Dining", 100_000),
                transaction(2, 2, "Food & Dining", 100_000),
                transaction(3, 1, "Shopping", 100_000),
                transaction(4, 2, "Shopping", 100_000),
            ]
        )
        plan = build_goal_savings_plan(
            goal(target_amount_minor=190_000),
            complete_profile(
                monthly_income_minor=100_000,
                emergency_contribution_minor=0,
                additional_commitments_minor=0,
                protected_buffer_contribution_minor=0,
            ),
            spending,
            preferences={"Food & Dining": "rejected"},
            as_of=self.as_of,
        )
        rejected = next(
            item for item in plan["candidates"] if item["category"] == "Food & Dining"
        )
        self.assertEqual(rejected["recommended_reduction_minor"], 0)
        self.assertGreater(
            next(
                item["recommended_reduction_minor"]
                for item in plan["candidates"]
                if item["category"] == "Shopping"
            ),
            0,
        )

    def test_new_spending_baseline_is_compared_without_claiming_achievement(self):
        previous = build_goal_savings_plan(
            goal(),
            complete_profile(),
            self.spending,
            as_of=self.as_of,
        )
        changed = two_month_spending(
            [
                transaction(3, 1, amount_minor=20_000),
                transaction(4, 2, amount_minor=20_000),
            ]
        )
        current = build_goal_savings_plan(
            goal(),
            complete_profile(),
            changed,
            as_of=self.as_of,
        )
        reconciliation = reconcile_savings_plan(previous, current)
        self.assertTrue(reconciliation["comparison_available"])
        self.assertIn("does not prove savings", reconciliation["message"])
        self.assertTrue(reconciliation["category_changes"])

    def test_paise_rounding_and_rupee_conversion(self):
        self.assertEqual(rupees_to_minor("1,234.56"), 123_456)
        self.assertEqual(rupees_to_minor("0.01"), 1)
        with self.assertRaisesRegex(ValueError, "decimal places"):
            rupees_to_minor("1.001")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            rupees_to_minor("-1")


class GoalPlannerPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "fintwin.sqlite3"
        self.environment = patch.dict(
            os.environ, {"FINTWIN_DATABASE_PATH": str(self.database_path)}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.directory.cleanup)
        self.user_a = register_user("User A", "a@example.com", "safe-password-123")
        self.user_b = register_user("User B", "b@example.com", "safe-password-123")
        self.goal_a = save_user_goal(
            self.user_a["id"], "Emergency", 100_000, 10_000, "2026-12-01"
        )
        self.goal_b = save_user_goal(
            self.user_b["id"], "Education", 200_000, 20_000, "2027-01-01"
        )

    def test_financial_profiles_and_goal_preferences_are_user_scoped(self):
        save_user_financial_profile(
            self.user_a["id"],
            {
                "monthly_income_minor": 1_000_000,
                "emergency_balance_minor": 20_000,
                "emergency_target_minor": 100_000,
                "emergency_contribution_minor": 5_000,
                "additional_commitments_minor": 0,
                "protected_buffer_contribution_minor": 0,
                "protected_categories": ["Shopping"],
                "statement_coverage_confirmed": True,
            },
        )
        self.assertEqual(
            get_user_financial_profile(self.user_a["id"])["monthly_income_minor"],
            1_000_000,
        )
        self.assertIsNone(
            get_user_financial_profile(self.user_b["id"])["monthly_income_minor"]
        )
        set_goal_plan_adjustment_status(
            self.user_a["id"], self.goal_a, "Food & Dining", "approved"
        )
        self.assertEqual(
            get_goal_plan_preferences(self.user_a["id"], self.goal_a),
            {"Food & Dining": "approved"},
        )
        self.assertEqual(
            get_goal_plan_preferences(self.user_b["id"], self.goal_a), {}
        )
        with self.assertRaisesRegex(ValueError, "not available"):
            set_goal_plan_adjustment_status(
                self.user_b["id"], self.goal_a, "Food & Dining", "approved"
            )

    def test_plan_snapshots_are_versioned_and_user_scoped(self):
        first = {"status": "Not assessable", "candidates": []}
        inputs = {"monthly_income_minor": None}
        self.assertEqual(
            save_goal_plan_snapshot(
                self.user_a["id"], self.goal_a, inputs, first
            ),
            1,
        )
        self.assertEqual(
            save_goal_plan_snapshot(
                self.user_a["id"], self.goal_a, inputs, first
            ),
            1,
        )
        self.assertEqual(
            save_goal_plan_snapshot(
                self.user_a["id"],
                self.goal_a,
                {"monthly_income_minor": 100},
                first,
            ),
            2,
        )
        latest = get_latest_goal_plan_snapshot(self.user_a["id"], self.goal_a)
        self.assertEqual(latest["version"], 2)
        self.assertEqual(
            get_latest_goal_plan_snapshot(self.user_b["id"], self.goal_a), None
        )
        with self.assertRaisesRegex(ValueError, "not available"):
            save_goal_plan_snapshot(
                self.user_b["id"], self.goal_a, inputs, first
            )

    def test_outcome_feedback_requires_evidence_and_is_user_scoped(self):
        with self.assertRaisesRegex(ValueError, "positive reported monthly reduction"):
            record_goal_plan_feedback(
                self.user_a["id"],
                self.goal_a,
                "Food & Dining",
                "achieved_confirmed",
                "Observed by user",
            )
        record_goal_plan_feedback(
            self.user_a["id"],
            self.goal_a,
            "Food & Dining",
            "achieved_confirmed",
            "Compared with the prior fully covered statement.",
            2_500,
        )
        record_goal_plan_feedback(
            self.user_a["id"],
            self.goal_a,
            "Shopping",
            "not_applicable",
            "These purchases are necessary.",
        )
        self.assertEqual(len(list_goal_plan_feedback(self.user_a["id"], self.goal_a)), 2)
        self.assertEqual(list_goal_plan_feedback(self.user_b["id"], self.goal_a), [])
        with self.assertRaisesRegex(ValueError, "not available"):
            record_goal_plan_feedback(
                self.user_b["id"],
                self.goal_a,
                "Food & Dining",
                "not_applicable",
                "Not applicable.",
            )

    def test_existing_goals_and_recommendations_survive_schema_migration(self):
        with connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO recommendations (
                        id, user_id, category, title, explanation, evidence_json,
                        status, created_at
                    ) VALUES ('rec-a', ?, 'food', 'Existing', 'Saved', '{}', 'active', 'now')
                    """,
                    (self.user_a["id"],),
                )
                connection.execute("PRAGMA user_version = 3")
        initialize_database(self.database_path)
        self.assertEqual(len(list_user_goals(self.user_a["id"])), 1)
        with connect() as connection:
            recommendation = connection.execute(
                "SELECT title FROM recommendations WHERE id = 'rec-a' AND user_id = ?",
                (self.user_a["id"],),
            ).fetchone()
        self.assertEqual(recommendation["title"], "Existing")


if __name__ == "__main__":
    unittest.main()
