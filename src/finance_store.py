from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import BASE_CATEGORIES, normalize_merchant
from src.database import connect, utc_now
from src.phonepe_upload import ParsedStatement


def _transaction_fingerprint(transaction_id: str, fallback: str) -> str:
    seed = f"id:{transaction_id}" if transaction_id else f"fallback:{fallback}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def save_import(
    user_id: str,
    statement: ParsedStatement,
    predictions: pd.DataFrame,
) -> dict[str, int | str]:
    if len(statement.processed) != len(predictions):
        raise ValueError("Prediction results do not match the imported row count.")
    required = {"ml_category", "ml_confidence", "final_category", "category_source"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Predictions are missing required fields: " + ", ".join(sorted(missing))
        )

    import_id = str(uuid.uuid4())
    now = utc_now()
    inserted = 0
    duplicate_count = 0
    possible_duplicate_count = 0
    row_rejections = 0
    seen_ids: set[str] = set()
    try:
        with connect() as connection:
            with connection:
                owner = connection.execute(
                    "SELECT 1 FROM users WHERE id = ? AND is_active = 1", (user_id,)
                ).fetchone()
                if owner is None:
                    raise ValueError("The signed-in account is not available.")
                connection.execute(
                    """
                    INSERT INTO imports (
                        id, user_id, original_filename, file_hash, period_start,
                        period_end, uploaded_at, status, accepted_rows, rejected_rows,
                        duplicate_rows, total_rows, possible_duplicate_rows,
                        error_summary, source_columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, ?, 0, ?, ?)
                    """,
                    (
                        import_id,
                        user_id,
                        statement.filename,
                        statement.file_sha256,
                        statement.date_start,
                        statement.date_end,
                        now,
                        "processing",
                        len(statement.rejected),
                        statement.total_rows,
                        "",
                        json.dumps(statement.source_columns, ensure_ascii=False),
                    ),
                )
                for rejected in statement.rejected:
                    connection.execute(
                        """
                        INSERT INTO import_rejections
                            (import_id, user_id, source_line, reason, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            import_id,
                            user_id,
                            rejected.source_line,
                            rejected.reason[:250],
                            now,
                        ),
                    )
                for position, (row, prediction) in enumerate(
                    zip(
                        statement.processed.to_dict(orient="records"),
                        predictions.to_dict(orient="records"),
                        strict=True,
                    )
                ):
                    external_id = str(row.get("transaction_id") or "").strip()
                    duplicate_type = row.get("duplicate_type", "")
                    if external_id and external_id in seen_ids:
                        duplicate_type = "exact"
                    if external_id:
                        seen_ids.add(external_id)
                    existing = None
                    if external_id:
                        existing = connection.execute(
                            """
                            SELECT id FROM transactions
                            WHERE user_id = ? AND transaction_id = ?
                            LIMIT 1
                            """,
                            (user_id, external_id),
                        ).fetchone()
                    if existing is not None or duplicate_type == "exact":
                        duplicate_count += 1
                        continue

                    description = str(row["transaction_details"])
                    merchant = str(row.get("merchant") or "")
                    date_value = str(row["transaction_date"])
                    tx_type = str(row["transaction_type"])
                    amount_minor = int(row["amount_minor"])
                    if not external_id:
                        possible = connection.execute(
                            """
                            SELECT id FROM transactions
                            WHERE user_id = ? AND transaction_date = ?
                              AND transaction_type = ? AND amount_minor = ?
                              AND lower(trim(merchant)) = lower(trim(?))
                            LIMIT 1
                            """,
                            (user_id, date_value, tx_type, amount_minor, merchant),
                        ).fetchone()
                        if possible is not None:
                            duplicate_type = "possible"
                    if duplicate_type == "possible":
                        possible_duplicate_count += 1

                    fallback = json.dumps(
                        [import_id, position, date_value, tx_type, amount_minor, merchant],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    fingerprint = _transaction_fingerprint(external_id, fallback)
                    final_category = str(prediction["final_category"])
                    if final_category == "Personal Transfer":
                        semantics = "transfer"
                    elif tx_type == "CREDIT":
                        semantics = (
                            "salary_income"
                            if final_category == "Salary/Income"
                            else "other_credit"
                        )
                    elif final_category == "Needs Review":
                        semantics = "unclassified"
                    else:
                        semantics = "consumption"
                    review_status = (
                        "Needs review"
                        if final_category == "Needs Review" or duplicate_type == "possible"
                        else "Reviewed"
                    )
                    confidence = float(prediction["ml_confidence"])
                    if not 0 <= confidence <= 1:
                        raise ValueError("Prediction confidence must be between 0 and 1.")
                    connection.execute(
                        """
                        INSERT INTO transactions (
                            user_id, import_id, transaction_date, transaction_time,
                            description, merchant, transaction_id, utr,
                            transaction_type, instrument, amount_minor, currency,
                            predicted_category, final_category, category_source,
                            prediction_confidence, review_status, transaction_semantics,
                            fingerprint, source_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'INR',
                                  ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            import_id,
                            date_value,
                            str(row.get("transaction_time") or ""),
                            description,
                            merchant,
                            external_id or None,
                            str(row.get("utr") or "") or None,
                            tx_type,
                            str(row.get("payment_method") or ""),
                            amount_minor,
                            str(prediction["ml_category"]),
                            final_category,
                            str(prediction["category_source"]),
                            confidence,
                            review_status,
                            semantics,
                            fingerprint,
                            json.dumps(
                                statement.source_rows[position],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            now,
                        ),
                    )
                    inserted += 1
                row_rejections = len(statement.rejected)
                connection.execute(
                    """
                    UPDATE imports SET status = ?, accepted_rows = ?,
                        duplicate_rows = ?, possible_duplicate_rows = ?,
                        error_summary = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (
                        "imported" if inserted else "duplicate",
                        inserted,
                        duplicate_count,
                        possible_duplicate_count,
                        f"{row_rejections} invalid row(s) excluded"
                        if row_rejections
                        else "",
                        import_id,
                        user_id,
                    ),
                )
    except sqlite3.IntegrityError as error:
        raise RuntimeError(
            "The import could not be committed; no partial transactions were saved."
        ) from error
    return {
        "import_id": import_id,
        "accepted_rows": inserted,
        "rejected_rows": row_rejections,
        "duplicate_rows": duplicate_count,
        "possible_duplicate_rows": possible_duplicate_count,
        "total_rows": statement.total_rows,
    }


def get_import_history(user_id: str) -> pd.DataFrame:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, original_filename, uploaded_at, period_start, period_end,
                   total_rows, accepted_rows, rejected_rows, duplicate_rows,
                   possible_duplicate_rows, status, error_summary
            FROM imports
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            """,
            (user_id,),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def count_import_duplicate_candidates(
    user_id: str, statement: ParsedStatement
) -> dict[str, int]:
    transaction_ids = [
        str(value).strip()
        for value in statement.processed["transaction_id"].tolist()
        if str(value).strip()
    ]
    existing_ids: set[str] = set()
    with connect() as connection:
        for offset in range(0, len(transaction_ids), 500):
            batch = transaction_ids[offset : offset + 500]
            placeholders = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"""
                SELECT transaction_id FROM transactions
                WHERE user_id = ? AND transaction_id IN ({placeholders})
                """,
                [user_id, *batch],
            ).fetchall()
            existing_ids.update(str(row["transaction_id"]) for row in rows)
        existing_without_ids = connection.execute(
            """
            SELECT transaction_date, transaction_type, amount_minor, merchant
            FROM transactions WHERE user_id = ? AND transaction_id IS NULL
            """,
            (user_id,),
        ).fetchall()
    possible_keys = {
        (
            str(row["transaction_date"]),
            str(row["transaction_type"]),
            int(row["amount_minor"]),
            normalize_merchant(row["merchant"]),
        )
        for row in existing_without_ids
    }
    exact = 0
    possible = 0
    for row in statement.processed.to_dict(orient="records"):
        transaction_id = str(row.get("transaction_id") or "").strip()
        if transaction_id and transaction_id in existing_ids:
            exact += 1
        elif not transaction_id and (
            str(row["transaction_date"]),
            str(row["transaction_type"]),
            int(row["amount_minor"]),
            normalize_merchant(row.get("merchant", "")),
        ) in possible_keys:
            possible += 1
    return {"exact": exact, "possible": possible}


def get_user_transactions(
    user_id: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    merchant: str | None = None,
    transaction_type: str | None = None,
    import_id: str | None = None,
    review_status: str | None = None,
) -> pd.DataFrame:
    clauses = ["user_id = ?"]
    parameters: list[Any] = [user_id]
    for column, value, operator in (
        ("transaction_date", start_date, ">="),
        ("transaction_date", end_date, "<="),
        ("final_category", category, "="),
        ("transaction_type", transaction_type, "="),
        ("import_id", import_id, "="),
        ("review_status", review_status, "="),
    ):
        if value:
            clauses.append(f"{column} {operator} ?")
            parameters.append(value)
    query = """
        SELECT t.id AS record_id, t.import_id, t.transaction_date,
               t.transaction_time, t.description AS transaction_details,
               t.transaction_id, t.utr, t.transaction_type, t.instrument AS payment_method,
               t.amount_minor, t.merchant, t.predicted_category AS ml_category,
               t.final_category, t.category_source,
               t.prediction_confidence AS ml_confidence, t.currency,
               t.review_status, t.transaction_semantics
        FROM transactions AS t
        WHERE """ + " AND ".join(clauses) + """
        ORDER BY t.transaction_date DESC, t.id DESC
    """
    with connect() as connection:
        rows = connection.execute(query, parameters).fetchall()
    if not rows:
        return pd.DataFrame(
            columns=[
                "record_id", "import_id", "transaction_date", "transaction_time",
                "transaction_details", "transaction_id", "utr", "transaction_type",
                "payment_method", "amount_minor", "amount", "merchant", "ml_category",
                "final_category", "category_source", "ml_confidence", "currency",
                "review_status", "transaction_semantics",
            ]
        )
    frame = pd.DataFrame([dict(row) for row in rows])
    if merchant:
        frame = frame.loc[
            frame["merchant"].map(normalize_merchant).eq(normalize_merchant(merchant))
        ].copy()
    frame["amount"] = frame["amount_minor"].astype("int64") / 100
    frame["transaction_id"] = frame["transaction_id"].fillna("").astype(str)
    missing_ids = frame["transaction_id"].eq("")
    frame.loc[missing_ids, "transaction_id"] = (
        "record:" + frame.loc[missing_ids, "record_id"].astype(str)
    )
    return frame


def get_user_feedback(user_id: str) -> pd.DataFrame:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT f.scope, f.transaction_id, f.merchant_key,
                   f.corrected_category, t.merchant
            FROM user_feedback AS f
            LEFT JOIN transactions AS t
              ON t.id = f.transaction_record_id AND t.user_id = f.user_id
            WHERE f.user_id = ?
            ORDER BY f.updated_at, f.id
            """,
            (user_id,),
        ).fetchall()
    return pd.DataFrame(
        [
            {
                "scope": row["scope"],
                "transaction_id": row["transaction_id"] or "",
                "merchant": row["merchant"] or row["merchant_key"] or "",
                "final_category": row["corrected_category"],
            }
            for row in rows
        ],
        columns=["scope", "transaction_id", "merchant", "final_category"],
    )


def save_user_correction(
    user_id: str,
    record_id: int,
    category: str,
    *,
    merchant_wide: bool = False,
) -> None:
    if category.strip() not in BASE_CATEGORIES:
        raise ValueError("Choose a supported transaction category.")
    now = utc_now()
    with connect() as connection:
        with connection:
            selected = connection.execute(
                """
                SELECT id, transaction_id, merchant, final_category
                FROM transactions WHERE id = ? AND user_id = ?
                """,
                (record_id, user_id),
            ).fetchone()
            if selected is None:
                raise ValueError("That transaction is not available to this account.")
            scope = "merchant" if merchant_wide else "transaction"
            external_id = selected["transaction_id"]
            if not external_id:
                external_id = f"record:{selected['id']}"
            merchant_key = normalize_merchant(selected["merchant"])
            if scope == "merchant" and not merchant_key:
                raise ValueError("Merchant-wide correction requires a merchant.")
            if scope == "merchant":
                previous = connection.execute(
                    """
                    SELECT id FROM user_feedback
                    WHERE user_id = ? AND scope = 'merchant' AND merchant_key = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (user_id, merchant_key),
                ).fetchone()
                if previous is None:
                    connection.execute(
                        """
                        INSERT INTO user_feedback (
                            user_id, merchant_key, original_category,
                            corrected_category, scope, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'merchant', ?, ?)
                        """,
                        (
                            user_id, merchant_key, selected["final_category"],
                            category.strip(), now, now,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE user_feedback
                        SET corrected_category = ?, updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (category.strip(), now, previous["id"], user_id),
                    )
                records = connection.execute(
                    "SELECT id, merchant FROM transactions WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
                matching_ids = [
                    int(row["id"])
                    for row in records
                    if normalize_merchant(row["merchant"]) == merchant_key
                ]
                for matching_id in matching_ids:
                    connection.execute(
                        """
                        UPDATE transactions SET final_category = ?,
                            category_source = 'User merchant feedback',
                            review_status = 'Reviewed'
                        WHERE id = ? AND user_id = ?
                          AND NOT EXISTS (
                              SELECT 1 FROM user_feedback AS f
                              WHERE f.user_id = transactions.user_id
                                AND f.transaction_record_id = transactions.id
                                AND f.scope = 'transaction'
                          )
                        """,
                        (category.strip(), matching_id, user_id),
                    )
            else:
                previous = connection.execute(
                    """
                    SELECT id FROM user_feedback
                    WHERE user_id = ? AND transaction_record_id = ? AND scope = 'transaction'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (user_id, record_id),
                ).fetchone()
                if previous is None:
                    connection.execute(
                        """
                        INSERT INTO user_feedback (
                            user_id, transaction_record_id, transaction_id, merchant_key,
                            original_category, corrected_category, scope, created_at, updated_at
                        ) VALUES (?, ?, ?, NULL, ?, ?, 'transaction', ?, ?)
                        """,
                        (
                            user_id, record_id, external_id,
                            selected["final_category"], category.strip(), now, now,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE user_feedback SET corrected_category = ?, updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (category.strip(), now, previous["id"], user_id),
                    )
                connection.execute(
                    """
                    UPDATE transactions SET final_category = ?,
                        category_source = 'User transaction correction',
                        review_status = 'Reviewed'
                    WHERE id = ? AND user_id = ?
                    """,
                    (category.strip(), record_id, user_id),
                )


def list_user_goals(user_id: str) -> pd.DataFrame:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, title, target_amount_minor, current_saved_minor,
                   target_date, monthly_contribution_minor, status
            FROM financial_goals WHERE user_id = ? ORDER BY target_date
            """,
            (user_id,),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def save_user_goal(
    user_id: str,
    title: str,
    target_amount_minor: int,
    current_saved_minor: int,
    target_date: str,
    monthly_contribution_minor: int | None = None,
    goal_id: str | None = None,
) -> str:
    goal_title = " ".join(title.split())
    if not goal_title:
        raise ValueError("Goal name must not be empty.")
    if target_amount_minor < 0 or current_saved_minor < 0:
        raise ValueError("Goal amounts must be non-negative.")
    if monthly_contribution_minor is not None and monthly_contribution_minor < 0:
        raise ValueError("Monthly contribution must be non-negative.")
    goal_id = goal_id or str(uuid.uuid4())
    now = utc_now()
    with connect() as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO financial_goals (
                    id, user_id, title, target_amount_minor, current_saved_minor,
                    target_date, monthly_contribution_minor, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    target_amount_minor=excluded.target_amount_minor,
                    current_saved_minor=excluded.current_saved_minor,
                    target_date=excluded.target_date,
                    monthly_contribution_minor=excluded.monthly_contribution_minor,
                    updated_at=excluded.updated_at
                WHERE financial_goals.user_id = excluded.user_id
                """,
                (
                    goal_id, user_id, goal_title, target_amount_minor,
                    current_saved_minor, target_date, monthly_contribution_minor,
                    now, now,
                ),
            )
    return goal_id


def delete_user_goal(user_id: str, goal_id: str) -> None:
    with connect() as connection:
        with connection:
            connection.execute(
                "DELETE FROM financial_goals WHERE id = ? AND user_id = ?",
                (goal_id, user_id),
            )


def get_user_anomaly_statuses(user_id: str) -> dict[int, str]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT transaction_record_id, status FROM anomaly_alerts
            WHERE user_id = ? AND is_current = 1
            """,
            (user_id,),
        ).fetchall()
    return {int(row["transaction_record_id"]): str(row["status"]) for row in rows}


def set_user_anomaly_status(
    user_id: str,
    record_id: int,
    status: str,
    *,
    anomaly_score: float,
    explanation: str,
) -> None:
    allowed = {"Unreviewed", "Reviewed - expected", "Reviewed - investigate"}
    if status not in allowed:
        raise ValueError("Choose a valid anomaly review status.")
    now = utc_now()
    with connect() as connection:
        with connection:
            transaction = connection.execute(
                "SELECT id FROM transactions WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            ).fetchone()
            if transaction is None:
                raise ValueError("That transaction is not available to this account.")
            alert_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{record_id}"))
            connection.execute(
                """
                INSERT INTO anomaly_alerts (
                    id, user_id, transaction_record_id, anomaly_score, reason,
                    status, detected_at, reviewed_at, is_current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(user_id, transaction_record_id) DO UPDATE SET
                    anomaly_score=excluded.anomaly_score,
                    reason=excluded.reason,
                    status=excluded.status,
                    reviewed_at=excluded.reviewed_at,
                    is_current=1
                """,
                (
                    alert_id, user_id, record_id, anomaly_score,
                    explanation[:1000], status, now,
                    now if status != "Unreviewed" else None,
                ),
            )


def save_recommendations(
    user_id: str, recommendations: list[dict[str, Any]]
) -> None:
    now = utc_now()
    with connect() as connection:
        with connection:
            for item in recommendations:
                recommendation_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{user_id}:{item.get('category')}:{item.get('period_start')}:"
                        f"{item.get('period_end')}:{item.get('title')}",
                    )
                )
                existing = connection.execute(
                    """
                    SELECT id FROM recommendations WHERE id = ? AND user_id = ?
                    """,
                    (recommendation_id, user_id),
                ).fetchone()
                values = (
                    item.get("period_start"),
                    item.get("period_end"),
                    str(item["category"]),
                    str(item["title"]),
                    str(item["explanation"]),
                    json.dumps(item.get("evidence", {}), ensure_ascii=False),
                    item.get("estimated_potential_saving_minor"),
                )
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO recommendations (
                            id, user_id, period_start, period_end, category, title,
                            explanation, evidence_json, estimated_potential_saving_minor,
                            status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                        """,
                        (recommendation_id, user_id, *values, now),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE recommendations SET
                            period_start = ?, period_end = ?, category = ?, title = ?,
                            explanation = ?, evidence_json = ?,
                            estimated_potential_saving_minor = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (*values, recommendation_id, user_id),
                    )


def list_user_recommendations(user_id: str) -> pd.DataFrame:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, period_start, period_end, category, title, explanation,
                   evidence_json, estimated_potential_saving_minor, status, created_at
            FROM recommendations WHERE user_id = ?
            ORDER BY created_at DESC, title
            """,
            (user_id,),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def set_recommendation_status(
    user_id: str, recommendation_id: str, status: str
) -> None:
    if status not in {"active", "Helpful", "Dismissed", "Addressed"}:
        raise ValueError("Choose a valid recommendation status.")
    with connect() as connection:
        with connection:
            cursor = connection.execute(
                "UPDATE recommendations SET status = ? WHERE id = ? AND user_id = ?",
                (status, recommendation_id, user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("That recommendation is not available to this account.")


def minor_to_decimal(amount_minor: int) -> Decimal:
    return Decimal(int(amount_minor)) / Decimal(100)
