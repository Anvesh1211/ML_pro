from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.common import DATA_DIR, MODEL_PATH
from src.financial_intelligence import (
    Scenario,
    add_expense_flexibility,
    calculate_goal,
    calculate_scenario,
    canonicalize_transactions,
    build_recommendations,
    cluster_spending,
    clear_application_state,
    delete_goal,
    detect_anomalies,
    detect_recurring_transactions,
    forecast_monthly_expenses,
    get_anomaly_review_statuses,
    load_goals,
    monthly_summary,
    save_goal,
    set_anomaly_review_status,
)
from src.predict_phonepe import (
    FEEDBACK_PATH,
    PROCESSED_PATH,
    _apply_feedback,
    load_feedback,
    run_prediction,
)
from src.preprocess_phonepe import RAW_PATH, load_phonepe_statement
from src.review_transactions import save_correction
from src.train_model import train_and_evaluate


class ProjectDatasetTests(unittest.TestCase):
    def test_statement_footer_is_audited_not_silently_treated_as_transactions(self):
        processed, audit = load_phonepe_statement(RAW_PATH)
        self.assertEqual(len(processed), 312)
        self.assertEqual(len(audit), 7)
        self.assertEqual(int(audit.iloc[0]["source_line"]), 317)
        self.assertEqual(int(audit.iloc[-1]["source_line"]), 323)
        self.assertTrue(audit["reason"].str.contains("statement_footer").all())
        self.assertTrue(processed["source_line"].is_unique)

    def test_original_phonepe_input_is_not_modified_by_loading(self):
        original = RAW_PATH.read_bytes()
        load_phonepe_statement(RAW_PATH)
        self.assertEqual(RAW_PATH.read_bytes(), original)

    def test_model_evaluation_is_merchant_disjoint_and_reports_real_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            report_path = Path(directory) / "metrics.json"
            model, metrics = train_and_evaluate(
                model_path=model_path, metrics_path=report_path
            )
            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())
        self.assertEqual(metrics["merchant_overlap"], 0)
        self.assertEqual(
            metrics["train_rows"] + metrics["test_rows"], metrics["rows_used"]
        )
        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(0 <= metrics["macro_f1"] <= 1)
        self.assertTrue(0 <= metrics["weighted_f1"] <= 1)
        self.assertIn("categories_absent_from_test", metrics)
        self.assertTrue(
            set(metrics["categories_absent_from_test"]).issubset(metrics["categories"])
        )
        self.assertGreater(len(model.classes_), 1)


class FinancialCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transactions = pd.read_csv(
            DATA_DIR / "final_phonepe_results.csv",
            dtype={"transaction_id": "string"},
            low_memory=False,
        )

    def test_monthly_totals_reconcile_with_observed_transaction_types(self):
        ledger = canonicalize_transactions(self.transactions)
        summary = monthly_summary(self.transactions)
        self.assertAlmostEqual(
            summary["all_debits"].sum(),
            float(ledger.loc[ledger["transaction_type"].eq("DEBIT"), "amount"].sum()),
        )
        self.assertAlmostEqual(
            summary["all_debits"].sum(),
            summary["consumption_expenses"].sum()
            + summary["unclassified_debits"].sum()
            + summary["debit_transfers"].sum(),
        )
        self.assertAlmostEqual(
            summary["salary_income_category_credits"].sum(),
            float(
                ledger.loc[
                    ledger["transaction_type"].eq("CREDIT")
                    & ledger["final_category"].eq("Salary/Income"),
                    "amount",
                ].sum()
            ),
        )
        self.assertTrue((summary["debit_transfers"] <= summary["all_debits"]).all())

    def test_invalid_ledger_values_raise_instead_of_becoming_zero(self):
        invalid = self.transactions.head(1).copy()
        invalid["amount"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "invalid values"):
            canonicalize_transactions(invalid)

    def test_duplicate_transaction_ids_are_rejected_to_prevent_double_counting(self):
        duplicated = self.transactions.head(2).copy()
        duplicated.iloc[1, duplicated.columns.get_loc("transaction_id")] = duplicated.iloc[0]["transaction_id"]
        with self.assertRaisesRegex(ValueError, "duplicate transaction IDs"):
            canonicalize_transactions(duplicated)

    def test_financial_goal_arithmetic_and_infeasibility(self):
        result = calculate_goal(100_000, 20_000, 8, 9_000)
        self.assertEqual(result["required_monthly_contribution"], 10_000)
        self.assertFalse(result["feasible_under_assumption"])
        self.assertEqual(result["monthly_shortfall"], 1_000)
        complete = calculate_goal(10_000, 12_000, 3)
        self.assertEqual(complete["remaining_amount"], 0)
        self.assertIsNone(complete["feasible_under_assumption"])

    def test_scenario_arithmetic_matches_hand_calculation(self):
        result = calculate_scenario(
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
        self.assertEqual(result["baseline_monthly_cash_flow"], 15_000)
        self.assertEqual(result["scenario_monthly_cash_flow"], 3_000)
        self.assertEqual(result["first_month_cash_flow_after_emergency"], -3_000)
        self.assertEqual(result["monthly_change"], -12_000)

    def test_scenario_rejects_invalid_percentages_and_negative_inputs(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_scenario(
                Scenario(1_000, 200, 200, income_reduction_rate=1.2)
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            calculate_scenario(Scenario(1_000, 200, -1))

    def test_flexibility_is_explicitly_a_category_proxy(self):
        result = add_expense_flexibility(self.transactions)
        self.assertEqual(len(result), len(self.transactions))
        self.assertTrue(result["flexibility_reason"].str.len().gt(0).all())
        self.assertTrue(
            result["flexibility_class"].str.contains("proxy", case=False).any()
        )

    def test_cluster_anomaly_recurring_and_forecast_use_current_ledger_schema(self):
        clusters, metrics = cluster_spending(self.transactions)
        anomalies = detect_anomalies(self.transactions)
        recurring = detect_recurring_transactions(self.transactions)
        forecast = forecast_monthly_expenses(self.transactions)
        debit_count = int(self.transactions["transaction_type"].eq("DEBIT").sum())
        canonical = canonicalize_transactions(self.transactions)
        self.assertEqual(len(anomalies), debit_count)
        self.assertEqual(len(clusters), int(canonical["is_consumption_expense"].sum()))
        self.assertIn("silhouette_score", metrics)
        self.assertIn("amount_only_baseline", metrics)
        self.assertIsNotNone(metrics["amount_only_baseline"]["silhouette_score"])
        self.assertTrue({"anomaly_score", "anomaly_status", "review_status"}.issubset(anomalies))
        self.assertTrue({"observations", "evidence", "status"}.issubset(recurring))
        self.assertTrue(forecast["available"])
        self.assertEqual(forecast["history_months"], len(monthly_summary(self.transactions)))
        self.assertEqual(forecast["last_observed_month"], forecast["observed_months"][-1])
        self.assertEqual(
            forecast["is_current_forecast"],
            forecast["forecast_month"] >= forecast["as_of_month"],
        )

    def test_recommendation_estimate_uses_all_statement_months_and_shows_assumptions(self):
        recommendations = build_recommendations(self.transactions, 0.10)
        self.assertFalse(recommendations.empty)
        observed_months = monthly_summary(self.transactions)["month"].nunique()
        for _, row in recommendations.iterrows():
            self.assertEqual(row["observed_months"], observed_months)
            self.assertAlmostEqual(
                row["estimated_monthly_difference"],
                row["observed_total"] / observed_months * 0.10,
            )
        self.assertTrue(recommendations["suggested_action"].str.len().gt(0).all())


class FeedbackAndPersistenceTests(unittest.TestCase):
    def test_existing_feedback_is_loaded_and_legacy_scope_is_merchant(self):
        feedback = load_feedback(FEEDBACK_PATH)
        self.assertEqual(len(feedback), 8)
        self.assertTrue(feedback["scope"].eq("merchant").all())

    def test_feedback_overrides_the_prediction_for_exact_transaction(self):
        processed = pd.read_csv(PROCESSED_PATH, dtype={"transaction_id": "string"})
        if not MODEL_PATH.exists():
            self.skipTest("Verified project model has not been trained.")
        result = run_prediction(
            processed_path=PROCESSED_PATH,
            output_paths=(),
            feedback_path=FEEDBACK_PATH,
        )
        source = result.iloc[0]
        feedback = pd.DataFrame(
            [{
                "merchant": source["merchant"],
                "final_category": "Personal Transfer",
                "scope": "transaction",
                "transaction_id": source["transaction_id"],
            }]
        )
        updated, sources = _apply_feedback(result, feedback)
        self.assertEqual(updated.iloc[0], "Personal Transfer")
        self.assertEqual(sources.iloc[0], "User transaction correction")
        self.assertEqual(
            result.iloc[0]["ml_category"],
            run_prediction(
                processed_path=PROCESSED_PATH,
                output_paths=(),
                feedback_path=FEEDBACK_PATH,
            ).iloc[0]["ml_category"],
        )

    def test_transaction_correction_has_precedence_over_merchant_memory(self):
        result = run_prediction(
            processed_path=PROCESSED_PATH,
            output_paths=(),
            feedback_path=FEEDBACK_PATH,
        )
        row = result.iloc[0]
        feedback = pd.DataFrame(
            [
                {
                    "merchant": row["merchant"],
                    "final_category": "Grocery",
                    "scope": "transaction",
                    "transaction_id": row["transaction_id"],
                },
                {
                    "merchant": row["merchant"],
                    "final_category": "Food & Dining",
                    "scope": "merchant",
                    "transaction_id": "",
                },
            ]
        )
        updated, sources = _apply_feedback(result, feedback)
        self.assertEqual(updated.iloc[0], "Grocery")
        self.assertEqual(sources.iloc[0], "User transaction correction")

    def test_saving_correction_preserves_existing_feedback_and_applies_it(self):
        processed = pd.read_csv(PROCESSED_PATH, dtype={"transaction_id": "string"})
        row = processed.iloc[0]
        with tempfile.TemporaryDirectory() as directory:
            feedback_path = Path(directory) / "feedback.csv"
            output_path = Path(directory) / "predictions.csv"
            pd.read_csv(FEEDBACK_PATH).to_csv(feedback_path, index=False)
            save_correction(
                str(row["merchant"]),
                "Grocery",
                transaction_id=str(row["transaction_id"]),
                feedback_path=feedback_path,
            )
            saved = load_feedback(feedback_path)
            result = run_prediction(
                processed_path=PROCESSED_PATH,
                output_paths=(output_path,),
                feedback_path=feedback_path,
            )
        self.assertEqual(len(saved), 9)
        corrected = result.loc[
            result["transaction_id"].astype(str).eq(str(row["transaction_id"]))
        ].iloc[0]
        self.assertEqual(corrected["final_category"], "Grocery")
        self.assertEqual(corrected["category_source"], "User transaction correction")

    def test_predictions_are_persisted_and_read_back(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            results = run_prediction(output_paths=(output,))
            persisted = pd.read_csv(output, low_memory=False)
        self.assertEqual(len(results), 312)
        self.assertEqual(len(persisted), 312)
        self.assertTrue(
            {"ml_category", "final_category", "category_source"}.issubset(
                persisted.columns
            )
        )

    def test_goals_persist_and_delete_in_temporary_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            save_goal(
                {
                    "goal_name": "Test goal",
                    "goal_amount": 12_000,
                    "current_saved": 2_000,
                    "target_date": "2027-10-01",
                },
                database,
            )
            goals = load_goals(database)
            self.assertEqual(len(goals), 1)
            self.assertEqual(goals.iloc[0]["goal_amount"], 12_000)
            delete_goal("Test goal", database)
            self.assertTrue(load_goals(database).empty)

    def test_anomaly_review_state_persists_and_clear_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            transaction_id = str(
                pd.read_csv(
                    DATA_DIR / "final_phonepe_results.csv",
                    dtype={"transaction_id": "string"},
                    usecols=["transaction_id"],
                ).iloc[0, 0]
            )
            set_anomaly_review_status(transaction_id, "Reviewed - expected", database)
            self.assertEqual(
                get_anomaly_review_statuses(database)[transaction_id],
                "Reviewed - expected",
            )
            with self.assertRaisesRegex(ValueError, "confirmation"):
                clear_application_state(confirmed=False, database_path=database)
            clear_application_state(confirmed=True, database_path=database)
            self.assertEqual(get_anomaly_review_statuses(database), {})


if __name__ == "__main__":
    unittest.main()
