from __future__ import annotations

import csv
import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.auth import (
    AuthenticationError,
    authenticate,
    get_session_user,
    hash_password,
    normalize_email,
    register_user,
    revoke_session,
    verify_password,
)
from src.database import SCHEMA_VERSION, connect, initialize_database
from src.common import BASE_CATEGORIES, DATA_DIR, MODEL_PATH
from src.finance_store import (
    get_import_history,
    get_user_financial_profile,
    get_user_transactions,
    list_user_goals,
    list_user_recommendations,
    save_import,
    save_user_financial_profile,
    save_user_correction,
    save_user_goal,
    save_recommendations,
    set_recommendation_status,
)
from src.phonepe_upload import PhonePeUploadError, parse_phonepe_csv
from src.predict_phonepe import _load_model, predict_transactions
from src.recommendations import build_personalized_suggestions


class UserWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.directory.name) / "fintwin.sqlite3"
        self.environment = patch.dict(
            os.environ, {"FINTWIN_DATABASE_PATH": str(self.database_path)}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.directory.cleanup)
        source_path = DATA_DIR / "testing.csv"
        self.phonepe_source = parse_phonepe_csv(
            source_path.read_bytes(), source_path.name
        )

    def register(self, name: str, email: str):
        return register_user(name, email, "a-safe-passphrase-123")

    def statement(self, row_index: int = 0, filename: str = "statement.csv"):
        source_row = self.phonepe_source.source_rows[row_index]
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(self.phonepe_source.source_columns)
        writer.writerow(
            [source_row.get(column, "") for column in self.phonepe_source.source_columns]
        )
        return parse_phonepe_csv(output.getvalue().encode("utf-8"), filename)

    def predictions_for(self, parsed):
        frame = parsed.processed.copy()
        frame["ml_category"] = "Food & Dining"
        frame["ml_confidence"] = 0.91
        frame["final_category"] = "Food & Dining"
        frame["category_source"] = "ML"
        return frame

    def test_email_normalization_password_hash_and_registration(self):
        user = self.register("  Example User ", "Person@Example.COM")
        self.assertEqual(user["name"], "Example User")
        self.assertEqual(user["email"], "person@example.com")
        self.assertEqual(normalize_email(" Person@Example.com "), "person@example.com")
        hashed = hash_password("another-safe-passphrase")
        self.assertNotIn("another-safe-passphrase", hashed)
        self.assertTrue(verify_password("another-safe-passphrase", hashed))
        self.assertFalse(verify_password("wrong-passphrase", hashed))
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.register("Duplicate", "person@example.com")

    def test_authentication_session_login_logout_and_bad_password(self):
        registered = self.register("Example User", "person@example.com")
        with self.assertRaises(AuthenticationError):
            authenticate("PERSON@example.com", "incorrect password")
        user, token, _expires = authenticate(
            " PERSON@EXAMPLE.COM ", "a-safe-passphrase-123"
        )
        self.assertEqual(user["id"], registered["id"])
        self.assertEqual(get_session_user(token)["id"], registered["id"])
        revoke_session(token)
        self.assertIsNone(get_session_user(token))

    def test_expired_session_is_rejected(self):
        user = self.register("Example User", "person@example.com")
        _authenticated, token, _expires = authenticate(
            user["email"], "a-safe-passphrase-123"
        )
        with connect() as connection:
            with connection:
                connection.execute(
                    "UPDATE auth_sessions SET expires_at = '2000-01-01T00:00:00+00:00' "
                    "WHERE user_id = ?",
                    (user["id"],),
                )
        self.assertIsNone(get_session_user(token))

    def test_phonepe_header_detection_validation_and_optional_ids(self):
        source = self.phonepe_source.source_rows[0]
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            [" date ", "TIME", " transaction details ", "Transaction Type", "Amount"]
        )
        writer.writerow(
            [
                source["Date"],
                source["Time"],
                source["Transaction Details"],
                source["Transaction Type"],
                source["Amount"],
            ]
        )
        content = ("\ufeff" + output.getvalue()).encode("utf-8")
        parsed = parse_phonepe_csv(content, "phonepe.csv")
        self.assertEqual(parsed.total_rows, 1)
        self.assertEqual(parsed.valid_rows, 1)
        self.assertEqual(len(parsed.rejected), 0)
        self.assertEqual(
            parsed.processed.iloc[0]["amount_minor"],
            self.phonepe_source.processed.iloc[0]["amount_minor"],
        )
        self.assertEqual(parsed.processed.iloc[0]["transaction_id"], "")
        self.assertEqual(
            parsed.date_start, self.phonepe_source.processed.iloc[0]["transaction_date"]
        )

    def test_existing_statement_footer_rows_are_reported_as_rejections(self):
        source_path = DATA_DIR / "testing.csv"
        parsed = parse_phonepe_csv(source_path.read_bytes(), source_path.name)
        self.assertEqual(parsed.total_rows, 1594)
        self.assertEqual(parsed.valid_rows, 1587)
        self.assertEqual(len(parsed.rejected), 7)

    def test_csv_rejects_unsupported_schema_size_and_extension(self):
        with self.assertRaises(PhonePeUploadError):
            parse_phonepe_csv(b"a,b\n1,2\n", "transactions.csv")
        with self.assertRaises(PhonePeUploadError):
            parse_phonepe_csv(b"Date,Amount\n1,1\n", "transactions.csv", max_bytes=5)
        with self.assertRaises(PhonePeUploadError):
            parse_phonepe_csv(b"Date,Amount\n1,1\n", "transactions.txt")

    def test_oversized_amount_is_rejected_without_aborting_the_statement(self):
        source = self.phonepe_source.source_rows[0]
        oversized = dict(source)
        oversized["Amount"] = "9" * 500
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(self.phonepe_source.source_columns)
        writer.writerow(
            [source.get(column, "") for column in self.phonepe_source.source_columns]
        )
        writer.writerow(
            [
                oversized.get(column, "")
                for column in self.phonepe_source.source_columns
            ]
        )
        parsed = parse_phonepe_csv(output.getvalue().encode(), "amounts.csv")
        self.assertEqual(parsed.valid_rows, 1)
        self.assertIn("supported precision", parsed.rejected[0].reason)

    def test_uploaded_statement_can_use_existing_model_without_shared_file_writes(self):
        if not MODEL_PATH.exists():
            self.skipTest("The existing shared classifier has not been trained.")
        raw_path = DATA_DIR / "testing.csv"
        before = {
            path: path.read_bytes()
            for path in (
                raw_path,
                DATA_DIR / "phonepe_processed.csv",
                DATA_DIR / "final_phonepe_results.csv",
            )
        }
        parsed = parse_phonepe_csv(raw_path.read_bytes(), raw_path.name)
        processed = parsed.processed.head(8).drop(
            columns=["amount_minor", "duplicate_type"], errors="ignore"
        )
        feedback = pd.DataFrame(
            columns=["merchant", "final_category", "scope", "transaction_id"]
        )
        result = predict_transactions(
            processed, model=_load_model(), feedback=feedback
        )
        self.assertEqual(len(result), 8)
        self.assertTrue(
            set(result["final_category"]).issubset(
                set(BASE_CATEGORIES) | {"Needs Review"}
            )
        )
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_suggestions_use_existing_statement_data_and_explain_assumptions(self):
        results_path = DATA_DIR / "final_phonepe_results.csv"
        transactions = pd.read_csv(
            results_path,
            dtype={"transaction_id": "string"},
            low_memory=False,
        )
        suggestions = build_personalized_suggestions(transactions)
        self.assertTrue(suggestions)
        for suggestion in suggestions:
            self.assertIn("title", suggestion)
            self.assertIn("explanation", suggestion)
            self.assertIn("evidence", suggestion)
            self.assertIn("period_start", suggestion)
            self.assertIn("period_end", suggestion)
            self.assertIn("category", suggestion)
            if suggestion["estimated_potential_saving_minor"] is not None:
                self.assertGreaterEqual(
                    suggestion["estimated_potential_saving_minor"], 0
                )

    def test_statement_duplicate_ids_and_candidate_signals(self):
        first = self.phonepe_source.source_rows[0].copy()
        second = self.phonepe_source.source_rows[1].copy()
        second["Transaction ID"] = ""
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(self.phonepe_source.source_columns)
        for row in (first, first, second, second):
            writer.writerow(
                [row.get(column, "") for column in self.phonepe_source.source_columns]
            )
        parsed = parse_phonepe_csv(output.getvalue().encode(), "duplicates.csv")
        self.assertEqual(parsed.duplicate_candidates, 2)
        self.assertEqual(
            parsed.processed["duplicate_type"].tolist(),
            ["", "exact", "", "possible"],
        )

    def test_import_is_user_scoped_and_preserves_exact_minor_units(self):
        user_a = self.register("A", "a@example.com")
        user_b = self.register("B", "b@example.com")
        parsed_a = self.statement(0)
        parsed_b = self.statement(1, "statement-2.csv")
        imported_a = save_import(user_a["id"], parsed_a, self.predictions_for(parsed_a))
        imported_b = save_import(user_b["id"], parsed_b, self.predictions_for(parsed_b))
        self.assertEqual(imported_a["accepted_rows"], 1)
        self.assertEqual(imported_b["accepted_rows"], 1)
        a_rows = get_user_transactions(user_a["id"])
        b_rows = get_user_transactions(user_b["id"])
        self.assertEqual(
            a_rows["transaction_id"].tolist(),
            [self.phonepe_source.processed.iloc[0]["transaction_id"]],
        )
        self.assertEqual(
            b_rows["transaction_id"].tolist(),
            [self.phonepe_source.processed.iloc[1]["transaction_id"]],
        )
        self.assertEqual(
            int(b_rows.iloc[0]["amount_minor"]),
            int(self.phonepe_source.processed.iloc[1]["amount_minor"]),
        )
        self.assertEqual(len(get_import_history(user_a["id"])), 1)
        with self.assertRaisesRegex(ValueError, "not available"):
            save_user_correction(
                user_b["id"], int(a_rows.iloc[0]["record_id"]), "Shopping"
            )
        self.assertEqual(get_user_transactions(user_a["id"]).iloc[0]["final_category"], "Food & Dining")

    def test_reimport_skips_exact_ids_and_tracks_possible_matches(self):
        user = self.register("A", "a@example.com")
        statement = self.statement(0)
        predictions = self.predictions_for(statement)
        first = save_import(user["id"], statement, predictions)
        second = save_import(user["id"], statement, predictions)
        self.assertEqual(first["accepted_rows"], 1)
        self.assertEqual(second["accepted_rows"], 0)
        self.assertEqual(second["duplicate_rows"], 1)
        history = get_import_history(user["id"])
        self.assertEqual(len(history), 2)
        duplicate_import = history.loc[
            history["id"].eq(second["import_id"])
        ].iloc[0]
        self.assertEqual(duplicate_import["total_rows"], 1)
        self.assertEqual(duplicate_import["duplicate_rows"], 1)

    def test_transaction_feedback_goals_and_recommendations_are_scoped(self):
        user_a = self.register("A", "a@example.com")
        user_b = self.register("B", "b@example.com")
        parsed = self.statement(0)
        save_import(user_a["id"], parsed, self.predictions_for(parsed))
        record_id = int(get_user_transactions(user_a["id"]).iloc[0]["record_id"])
        save_user_correction(user_a["id"], record_id, "Shopping")
        self.assertEqual(
            get_user_transactions(user_a["id"]).iloc[0]["final_category"], "Shopping"
        )
        self.assertTrue(list_user_goals(user_a["id"]).empty)
        goal_id = save_user_goal(
            user_a["id"], "Emergency fund", 1000000, 100000, "2030-01-01"
        )
        self.assertEqual(list_user_goals(user_a["id"]).iloc[0]["id"], goal_id)
        self.assertTrue(list_user_goals(user_b["id"]).empty)
        recommendation = {
            "period_start": "2026-10-01",
            "period_end": "2026-10-31",
            "category": "spending",
            "title": "Review small purchases",
            "explanation": "Repeated transactions were observed.",
            "evidence": {"observations": 3},
            "estimated_potential_saving_minor": 2500,
        }
        save_recommendations(user_a["id"], [recommendation])
        a_recs = list_user_recommendations(user_a["id"])
        self.assertEqual(len(a_recs), 1)
        self.assertTrue(list_user_recommendations(user_b["id"]).empty)
        set_recommendation_status(user_a["id"], a_recs.iloc[0]["id"], "Helpful")
        self.assertEqual(list_user_recommendations(user_a["id"]).iloc[0]["status"], "Helpful")
        with self.assertRaisesRegex(ValueError, "not available"):
            set_recommendation_status(user_b["id"], a_recs.iloc[0]["id"], "Dismissed")

    def test_new_import_requires_statement_coverage_to_be_reconfirmed(self):
        user = self.register("Coverage User", "coverage@example.com")
        save_user_financial_profile(
            user["id"],
            {
                "monthly_income_minor": 100_000,
                "emergency_balance_minor": 0,
                "emergency_target_minor": 0,
                "emergency_contribution_minor": 0,
                "additional_commitments_minor": 0,
                "protected_buffer_contribution_minor": 0,
                "protected_categories": [],
                "statement_coverage_confirmed": True,
            },
        )
        parsed = self.statement(0)
        save_import(user["id"], parsed, self.predictions_for(parsed))
        self.assertFalse(
            get_user_financial_profile(user["id"])["statement_coverage_confirmed"]
        )

    def test_legacy_database_migration_preserves_records_and_creates_backup(self):
        legacy = Path(self.directory.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy)
        connection.execute(
            "CREATE TABLE goals (goal_name TEXT PRIMARY KEY, goal_amount REAL NOT NULL, "
            "current_saved REAL NOT NULL, target_date TEXT NOT NULL, "
            "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO goals (goal_name, goal_amount, current_saved, target_date) "
            "VALUES ('Existing goal', 12000, 2000, '2030-01-01')"
        )
        connection.execute(
            "CREATE TABLE anomaly_reviews (transaction_id TEXT PRIMARY KEY, "
            "review_status TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO anomaly_reviews (transaction_id, review_status) "
            "VALUES ('old-id', 'Reviewed - expected')"
        )
        connection.commit()
        connection.close()
        initialize_database(legacy)
        migrated = sqlite3.connect(legacy)
        self.assertEqual(
            migrated.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
        )
        self.assertEqual(migrated.execute("SELECT count(*) FROM goals").fetchone()[0], 1)
        self.assertEqual(
            migrated.execute("SELECT count(*) FROM anomaly_reviews").fetchone()[0], 1
        )
        migrated.close()
        self.assertTrue(list(Path(self.directory.name).glob("legacy.sqlite3.pre-migration-*.bak")))

    def test_version_two_import_history_migration_backfills_counts(self):
        legacy = Path(self.directory.name) / "version-two.sqlite3"
        connection = sqlite3.connect(legacy)
        connection.execute(
            """
            CREATE TABLE imports (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                original_filename TEXT NOT NULL, file_hash TEXT NOT NULL,
                period_start TEXT, period_end TEXT, uploaded_at TEXT NOT NULL,
                status TEXT NOT NULL, accepted_rows INTEGER NOT NULL DEFAULT 0,
                rejected_rows INTEGER NOT NULL DEFAULT 0,
                duplicate_rows INTEGER NOT NULL DEFAULT 0,
                error_summary TEXT NOT NULL DEFAULT '',
                source_columns_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO imports (
                id, user_id, original_filename, file_hash, uploaded_at, status,
                accepted_rows, rejected_rows, duplicate_rows, source_columns_json
            ) VALUES ('import-1', 'user-1', 'statement.csv', 'hash-1',
                      '2026-10-01T00:00:00+00:00', 'imported', 4, 2, 1, '[]')
            """
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
        connection.close()

        initialize_database(legacy)

        migrated = sqlite3.connect(legacy)
        self.assertEqual(
            migrated.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
        )
        self.assertEqual(
            migrated.execute(
                "SELECT total_rows, possible_duplicate_rows FROM imports "
                "WHERE id = 'import-1'"
            ).fetchone(),
            (7, 0),
        )
        migrated.close()
        self.assertTrue(
            list(Path(self.directory.name).glob(
                "version-two.sqlite3.pre-migration-*.bak"
            ))
        )

    def test_failed_import_rolls_back_import_and_rejections(self):
        parsed = self.statement(0)
        with self.assertRaisesRegex(ValueError, "not available"):
            save_import("missing-user", parsed, self.predictions_for(parsed))
        initialize_database(self.database_path)
        with connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM imports").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM transactions").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()