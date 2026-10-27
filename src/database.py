from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "fintwin_state.sqlite3"
SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS auth_attempts (
    email_key TEXT PRIMARY KEY,
    window_started REAL NOT NULL,
    failures INTEGER NOT NULL,
    locked_until REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS imports (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT,
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL,
    accepted_rows INTEGER NOT NULL DEFAULT 0,
    rejected_rows INTEGER NOT NULL DEFAULT 0,
    duplicate_rows INTEGER NOT NULL DEFAULT 0,
    total_rows INTEGER NOT NULL DEFAULT 0,
    possible_duplicate_rows INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT NOT NULL DEFAULT '',
    source_columns_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    import_id TEXT NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    transaction_date TEXT NOT NULL,
    transaction_time TEXT,
    description TEXT NOT NULL,
    merchant TEXT NOT NULL,
    transaction_id TEXT,
    utr TEXT,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('DEBIT', 'CREDIT')),
    instrument TEXT NOT NULL DEFAULT '',
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    predicted_category TEXT NOT NULL,
    final_category TEXT NOT NULL,
    category_source TEXT NOT NULL,
    prediction_confidence REAL,
    review_status TEXT NOT NULL DEFAULT 'Needs review',
    transaction_semantics TEXT NOT NULL DEFAULT 'unknown',
    fingerprint TEXT NOT NULL,
    source_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, fingerprint)
);
CREATE TABLE IF NOT EXISTS import_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    import_id TEXT NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_line INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    transaction_record_id INTEGER REFERENCES transactions(id) ON DELETE CASCADE,
    transaction_id TEXT,
    merchant_key TEXT,
    original_category TEXT NOT NULL,
    corrected_category TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('transaction', 'merchant')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (scope = 'transaction' AND transaction_record_id IS NOT NULL)
        OR (scope = 'merchant' AND merchant_key IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    transaction_record_id INTEGER NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    anomaly_score REAL NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Unreviewed',
    detected_at TEXT NOT NULL,
    reviewed_at TEXT,
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    UNIQUE(user_id, transaction_record_id)
);
CREATE TABLE IF NOT EXISTS financial_goals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    target_amount_minor INTEGER NOT NULL CHECK (target_amount_minor >= 0),
    current_saved_minor INTEGER NOT NULL CHECK (current_saved_minor >= 0),
    target_date TEXT NOT NULL,
    monthly_contribution_minor INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS financial_profiles (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    monthly_income_minor INTEGER CHECK (monthly_income_minor >= 0),
    emergency_balance_minor INTEGER CHECK (emergency_balance_minor >= 0),
    emergency_target_minor INTEGER CHECK (emergency_target_minor >= 0),
    emergency_contribution_minor INTEGER CHECK (emergency_contribution_minor >= 0),
    additional_commitments_minor INTEGER CHECK (additional_commitments_minor >= 0),
    protected_buffer_contribution_minor INTEGER
        CHECK (protected_buffer_contribution_minor >= 0),
    protected_categories_json TEXT NOT NULL DEFAULT '[]',
    statement_coverage_confirmed INTEGER NOT NULL DEFAULT 0
        CHECK (statement_coverage_confirmed IN (0, 1)),
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    period_start TEXT,
    period_end TEXT,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    explanation TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    estimated_potential_saving_minor INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    UNIQUE(user_id, category, period_start, period_end, title)
);
CREATE TABLE IF NOT EXISTS goal_plan_preferences (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_id TEXT NOT NULL REFERENCES financial_goals(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('approved', 'rejected')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(user_id, goal_id, category)
);
CREATE TABLE IF NOT EXISTS goal_plan_feedback (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_id TEXT NOT NULL REFERENCES financial_goals(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('not_applicable', 'achieved_confirmed')),
    reported_monthly_reduction_minor INTEGER
        CHECK (reported_monthly_reduction_minor >= 0),
    evidence_note TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goal_plans (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal_id TEXT NOT NULL REFERENCES financial_goals(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    signature TEXT NOT NULL,
    input_json TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, goal_id, version),
    UNIQUE(user_id, goal_id, signature)
);
CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    setting_key TEXT NOT NULL,
    setting_value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(user_id, setting_key)
);
CREATE TABLE IF NOT EXISTS goals (
    goal_name TEXT PRIMARY KEY,
    goal_amount REAL NOT NULL CHECK(goal_amount >= 0),
    current_saved REAL NOT NULL CHECK(current_saved >= 0),
    target_date TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS anomaly_reviews (
    transaction_id TEXT PRIMARY KEY,
    review_status TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_expiry
    ON auth_sessions(user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_imports_user_uploaded
    ON imports(user_id, uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_user_date
    ON transactions(user_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_user_import
    ON transactions(user_id, import_id);
CREATE INDEX IF NOT EXISTS idx_transactions_user_category
    ON transactions(user_id, final_category);
CREATE INDEX IF NOT EXISTS idx_rejections_user_import
    ON import_rejections(user_id, import_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_transaction
    ON user_feedback(user_id, transaction_record_id);
CREATE INDEX IF NOT EXISTS idx_anomalies_user_status
    ON anomaly_alerts(user_id, status, is_current);
CREATE INDEX IF NOT EXISTS idx_goals_user_status
    ON financial_goals(user_id, status);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_period
    ON recommendations(user_id, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_goal_plans_user_goal_version
    ON goal_plans(user_id, goal_id, version DESC);
CREATE INDEX IF NOT EXISTS idx_goal_plan_feedback_user_goal
    ON goal_plan_feedback(user_id, goal_id, created_at DESC);
"""
_INITIALIZED_DATABASES: set[str] = set()


def database_path(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("FINTWIN_DATABASE_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_DATABASE_PATH


def _backup_before_migration(
    connection: sqlite3.Connection, path: Path
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.pre-migration-{timestamp}.bak")
    suffix = 1
    while backup.exists():
        backup = path.with_name(
            f"{path.name}.pre-migration-{timestamp}-{suffix}.bak"
        )
        suffix += 1
    backup_connection = sqlite3.connect(backup)
    try:
        connection.backup(backup_connection)
    finally:
        backup_connection.close()


def initialize_database(path: str | Path | None = None) -> Path:
    resolved = database_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    existed = resolved.is_file() and resolved.stat().st_size > 0
    connection = sqlite3.connect(resolved, timeout=15)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema {version} is newer than supported schema "
                f"{SCHEMA_VERSION}."
            )
        if existed and version < SCHEMA_VERSION:
            _backup_before_migration(connection, resolved)
        connection.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                connection.execute(statement)
        import_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(imports)").fetchall()
        }
        if "total_rows" not in import_columns:
            connection.execute(
                "ALTER TABLE imports ADD COLUMN total_rows INTEGER NOT NULL DEFAULT 0"
            )
        if "possible_duplicate_rows" not in import_columns:
            connection.execute(
                "ALTER TABLE imports ADD COLUMN possible_duplicate_rows "
                "INTEGER NOT NULL DEFAULT 0"
            )
        connection.execute(
            """
            UPDATE imports
            SET total_rows = accepted_rows + rejected_rows + duplicate_rows
            WHERE total_rows = 0
            """
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(
                "Database migration completed with foreign-key violations."
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    _INITIALIZED_DATABASES.add(str(resolved.resolve()))
    return resolved


@contextmanager
def connect(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    resolved = database_path(path)
    resolved_key = str(resolved.resolve())
    if resolved_key in _INITIALIZED_DATABASES:
        pass
    elif not resolved.exists():
        initialize_database(resolved)
    else:
        probe = sqlite3.connect(resolved, timeout=15)
        try:
            version = int(probe.execute("PRAGMA user_version").fetchone()[0])
        finally:
            probe.close()
        if version < SCHEMA_VERSION:
            initialize_database(resolved)
        elif version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema {version} is newer than supported schema "
                f"{SCHEMA_VERSION}."
            )
    connection = sqlite3.connect(resolved, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    try:
        yield connection
    finally:
        connection.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
