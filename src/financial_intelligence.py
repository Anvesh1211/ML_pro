from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import DATA_DIR, normalize_merchant, require_columns


TRANSACTION_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "transaction_type",
    "amount",
    "merchant",
    "final_category",
}
ESSENTIAL_PROXY_CATEGORIES = {
    "Bills & Utilities",
    "Education",
    "Grocery",
    "Health",
    "Mobile Recharge",
    "Rent",
    "Transport",
}
FLEXIBLE_CATEGORIES = {
    "Entertainment",
    "Food & Dining",
    "Personal Care",
    "Shopping",
}


def canonicalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, TRANSACTION_COLUMNS, "Transaction ledger")
    ledger = frame.copy()
    ledger["transaction_date"] = pd.to_datetime(
        ledger["transaction_date"], errors="coerce"
    )
    ledger["amount"] = pd.to_numeric(ledger["amount"], errors="coerce")
    ledger["transaction_type"] = (
        ledger["transaction_type"].fillna("").astype(str).str.strip().str.upper()
    )
    invalid_date = ledger["transaction_date"].isna()
    invalid_amount = ledger["amount"].isna() | ~np.isfinite(ledger["amount"])
    invalid_type = ~ledger["transaction_type"].isin({"DEBIT", "CREDIT"})
    invalid = invalid_date | invalid_amount | invalid_type
    if invalid.any():
        raise ValueError(
            "Transaction ledger contains invalid values: "
            f"{int(invalid_date.sum())} dates, {int(invalid_amount.sum())} amounts, "
            f"{int(invalid_type.sum())} transaction types."
        )
    if (ledger["amount"] < 0).any():
        raise ValueError(
            "Amounts must be non-negative; represent direction in transaction_type."
        )
    transaction_ids = ledger["transaction_id"].fillna("").astype(str).str.strip()
    duplicates = transaction_ids.ne("") & transaction_ids.duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            f"Transaction ledger contains {int(duplicates.sum())} rows with duplicate transaction IDs."
        )
    ledger["merchant"] = ledger["merchant"].fillna("").astype(str).str.strip()
    ledger["merchant_key"] = ledger["merchant"].map(normalize_merchant)
    ledger["final_category"] = (
        ledger["final_category"].fillna("Needs Review").astype(str).str.strip()
    )
    ledger["month_period"] = ledger["transaction_date"].dt.to_period("M")
    ledger["signed_cash_flow"] = np.where(
        ledger["transaction_type"].eq("CREDIT"),
        ledger["amount"],
        -ledger["amount"],
    )
    ledger["is_transfer"] = ledger["final_category"].eq("Personal Transfer")
    ledger["is_salary_income"] = (
        ledger["transaction_type"].eq("CREDIT")
        & ledger["final_category"].eq("Salary/Income")
    )
    ledger["is_consumption_expense"] = (
        ledger["transaction_type"].eq("DEBIT")
        & ~ledger["is_transfer"]
        & ~ledger["final_category"].isin({"Needs Review", "Salary/Income"})
    )
    ledger["is_unclassified_debit"] = (
        ledger["transaction_type"].eq("DEBIT")
        & ~ledger["is_transfer"]
        & ~ledger["is_consumption_expense"]
    )
    return ledger


def monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ledger = canonicalize_transactions(frame)
    rows = []
    for period, month in ledger.groupby("month_period", sort=True):
        credits = month["transaction_type"].eq("CREDIT")
        debits = month["transaction_type"].eq("DEBIT")
        classified_consumption = month["is_consumption_expense"]
        salary = credits & month["final_category"].eq("Salary/Income")
        rows.append(
            {
                "month": str(period),
                "salary_income_category_credits": float(
                    month.loc[salary, "amount"].sum()
                ),
                "other_credits": float(month.loc[credits & ~salary, "amount"].sum()),
                "consumption_expenses": float(month.loc[classified_consumption, "amount"].sum()),
                "unclassified_debits": float(
                    month.loc[month["is_unclassified_debit"], "amount"].sum()
                ),
                "debit_transfers": float(month.loc[debits & month["is_transfer"], "amount"].sum()),
                "all_debits": float(month.loc[debits, "amount"].sum()),
                "net_observed_cash_flow": float(month["signed_cash_flow"].sum()),
                "transaction_count": int(len(month)),
                "salary_income_category_credit_count": int(salary.sum()),
                "coverage_start": month["transaction_date"].min().date().isoformat(),
                "coverage_end": month["transaction_date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def add_expense_flexibility(frame: pd.DataFrame) -> pd.DataFrame:
    ledger = canonicalize_transactions(frame)
    classes = []
    reasons = []
    for _, row in ledger.iterrows():
        category = row["final_category"]
        if row["is_transfer"]:
            label, reason = "Transfer (excluded from consumption)", "Personal transfers are separated from spending."
        elif row["transaction_type"] == "CREDIT":
            label, reason = "Income or other credit", "Credit; confirm income versus transfer from its category."
        elif category in ESSENTIAL_PROXY_CATEGORIES:
            label, reason = "Potentially essential (category proxy)", "Category is commonly essential; individual transaction purpose is unverified."
        elif category in FLEXIBLE_CATEGORIES:
            label, reason = "Potentially adjustable (category proxy)", "Category may include flexible purchases; individual purpose is unverified."
        else:
            label, reason = "Unknown / review", "Available description and category do not support a flexibility claim."
        classes.append(label)
        reasons.append(reason)
    ledger["flexibility_class"] = classes
    ledger["flexibility_reason"] = reasons
    return ledger


def detect_recurring_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    ledger = canonicalize_transactions(frame)
    ledger = ledger.loc[
        ledger["transaction_type"].eq("DEBIT")
        & ~ledger["is_transfer"]
        & ledger["merchant_key"].ne("")
    ].copy()
    observations = []
    for merchant_key, group in ledger.groupby("merchant_key"):
        group = group.sort_values("transaction_date")
        if len(group) < 2:
            continue
        dates = group["transaction_date"].drop_duplicates().sort_values()
        intervals = dates.diff().dropna().dt.days.astype(float)
        if intervals.empty:
            continue
        median_interval = float(intervals.median())
        relative_spread = (
            float(intervals.std(ddof=0) / median_interval)
            if median_interval > 0 and len(intervals) > 1
            else 0.0
        )
        recurring_cadence = (
            5 <= median_interval <= 9 or 20 <= median_interval <= 40
        )
        consistent = relative_spread <= 0.25
        if len(dates) >= 3 and recurring_cadence and consistent:
            status = "Regular cadence observed (3+ dates)"
        elif recurring_cadence:
            status = "Probable pattern"
        else:
            status = "Repeated; cadence unclear"
        amounts = group["amount"].astype(float)
        observations.append(
            {
                "merchant": group["merchant"].iloc[0],
                "observations": int(len(dates)),
                "first_seen": dates.min().date().isoformat(),
                "last_seen": dates.max().date().isoformat(),
                "median_interval_days": median_interval,
                "interval_variation": relative_spread,
                "median_amount": float(amounts.median()),
                "amount_variation": (
                    float(amounts.std(ddof=0) / amounts.mean())
                    if amounts.mean() > 0
                    else None
                ),
                "evidence": f"{len(dates)} distinct dates; observed intervals {', '.join(str(int(x)) for x in intervals)} days.",
                "status": status,
            }
        )
    return pd.DataFrame(observations).sort_values(
        ["observations", "merchant"], ascending=[False, True]
    ).reset_index(drop=True) if observations else pd.DataFrame(
        columns=[
            "merchant", "observations", "first_seen", "last_seen",
            "median_interval_days", "interval_variation", "median_amount",
            "amount_variation", "evidence", "status",
        ]
    )


def cluster_spending(frame: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, dict]:
    ledger = canonicalize_transactions(frame)
    expenses = ledger.loc[ledger["is_consumption_expense"]].copy()
    if len(expenses) < max(n_clusters, 3):
        raise ValueError("Not enough non-transfer debit transactions for clustering.")
    expenses["log_amount"] = np.log1p(expenses["amount"].astype(float))
    expenses["weekday"] = expenses["transaction_date"].dt.dayofweek.astype(str)
    expenses["category_feature"] = expenses["final_category"].replace("", "Unknown")
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["log_amount"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["category_feature", "weekday"],
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("features", preprocessor),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ]
    )
    clusters = pipeline.fit_predict(expenses)
    matrix = pipeline.named_steps["features"].transform(expenses)
    silhouette = (
        float(silhouette_score(matrix, clusters))
        if len(set(clusters)) > 1 and len(set(clusters)) < len(expenses)
        else None
    )
    amount_only_matrix = StandardScaler().fit_transform(
        expenses[["amount"]].astype(float)
    )
    amount_only_clusters = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    ).fit_predict(amount_only_matrix)
    amount_only_silhouette = (
        float(silhouette_score(amount_only_matrix, amount_only_clusters))
        if len(set(amount_only_clusters)) > 1
        and len(set(amount_only_clusters)) < len(expenses)
        else None
    )
    expenses["spending_cluster"] = clusters
    summary = (
        expenses.groupby("spending_cluster")
        .agg(
            transaction_count=("amount", "size"),
            mean_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            category_mix=("final_category", lambda values: values.value_counts().index[0]),
        )
        .reset_index()
    )
    return expenses, {
        "method": "KMeans on log amount, category, and weekday; one-hot/scaled features",
        "random_state": 42,
        "silhouette_score": silhouette,
        "amount_only_baseline": {
            "method": "KMeans on standardized raw transaction amount only",
            "random_state": 42,
            "silhouette_score": amount_only_silhouette,
        },
        "summary": summary.to_dict(orient="records"),
    }


def detect_anomalies(frame: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be greater than 0 and less than 0.5.")
    ledger = canonicalize_transactions(frame)
    eligible = ledger.loc[ledger["transaction_type"].eq("DEBIT")].copy()
    if len(eligible) < 3:
        raise ValueError("At least three debit transactions are required for anomaly analysis.")
    log_amount_series = np.log1p(eligible["amount"].astype(float))
    log_amount = log_amount_series.to_numpy().reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(log_amount)
    decision = model.decision_function(log_amount)
    eligible["anomaly"] = labels
    eligible["anomaly_score"] = -decision
    eligible["anomaly_status"] = np.where(labels == -1, "Unusual; review", "Normal")
    eligible["review_priority"] = np.where(labels == -1, "Review", "Routine")
    global_q1, global_q3 = np.quantile(log_amount_series, [0.25, 0.75])
    global_iqr = global_q3 - global_q1
    robust_flags = []
    for index, row in eligible.iterrows():
        category_values = log_amount_series.loc[
            eligible["final_category"].eq(row["final_category"])
        ]
        if len(category_values) < 5:
            category_values = log_amount_series
        q1, q3 = np.quantile(category_values, [0.25, 0.75])
        spread = q3 - q1
        if spread == 0:
            q1, q3, spread = global_q1, global_q3, global_iqr
        value = log_amount_series.loc[index]
        robust_flags.append(bool(value < q1 - 1.5 * spread or value > q3 + 1.5 * spread))
    eligible["robust_amount_outlier"] = robust_flags
    explanations = []
    for _, row in eligible.iterrows():
        category_values = eligible.loc[
            eligible["final_category"].eq(row["final_category"]), "amount"
        ]
        if len(category_values) >= 3:
            median = float(category_values.median())
            explanations.append(
                f"Amount ₹{row['amount']:.2f}; category median ₹{median:.2f}; "
                "flag is exploratory and not a fraud determination."
            )
        else:
            explanations.append(
                f"Amount ₹{row['amount']:.2f}; limited category history; "
                "flag is exploratory and not a fraud determination."
            )
    eligible["anomaly_explanation"] = explanations
    eligible["robust_baseline"] = "1.5×IQR on log amount; category group when >=5 rows, otherwise all debits"
    eligible["review_status"] = "Unreviewed"
    return eligible.reset_index(drop=True)


def forecast_monthly_expenses(frame: pd.DataFrame) -> dict:
    summary = monthly_summary(frame)
    if summary.empty:
        return {
            "available": False,
            "reason": "No valid monthly transaction history.",
            "history_months": 0,
        }
    values = summary["consumption_expenses"].astype(float).tolist()
    months = summary["month"].tolist()
    if len(values) < 2:
        return {
            "available": False,
            "reason": "At least two observed calendar months are required for a baseline forecast.",
            "history_months": len(values),
            "observed_months": months,
        }
    target_period = pd.Period(months[-1], freq="M") + 1
    as_of_period = pd.Timestamp.now().to_period("M")
    mean_errors = []
    moving_errors = []
    for index in range(2, len(values)):
        actual = values[index]
        mean_errors.append((actual, float(np.mean(values[:index]))))
        moving_errors.append((actual, float(np.mean(values[max(0, index - 3):index]))))
    mean_mae = (
        float(mean_absolute_error(
            [actual for actual, _ in mean_errors],
            [prediction for _, prediction in mean_errors],
        ))
        if mean_errors else None
    )
    moving_mae = (
        float(mean_absolute_error(
            [actual for actual, _ in moving_errors],
            [prediction for _, prediction in moving_errors],
        ))
        if moving_errors else None
    )
    return {
        "available": True,
        "forecast_month": str(target_period),
        "last_observed_month": months[-1],
        "as_of_month": str(as_of_period),
        "data_age_months": max(int(as_of_period.ordinal - pd.Period(months[-1], freq="M").ordinal), 0),
        "is_current_forecast": target_period >= as_of_period,
        "forecast_basis": "Month immediately after last observed month; may be a historical period.",
        "forecast_expense_historical_mean": float(np.mean(values)),
        "forecast_expense_moving_average_3_month": float(np.mean(values[-3:])),
        "history_months": len(values),
        "observed_months": months,
        "chronological_validation_origins": max(len(values) - 2, 0),
        "historical_mean_mae": mean_mae,
        "moving_average_mae": moving_mae,
        "limitation": (
            "Calendar months are based on statement coverage, which is incomplete at its first "
            "and last month. One-step historical errors are not a guarantee of future accuracy; "
            "seasonality is not estimated."
        ),
    }


def calculate_goal(
    goal_amount: float,
    current_saved: float,
    months_remaining: int,
    available_monthly_contribution: float | None = None,
) -> dict:
    values = (goal_amount, current_saved)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Goal amount and current savings must be finite and non-negative.")
    if months_remaining <= 0:
        raise ValueError("months_remaining must be a positive integer.")
    remaining = max(goal_amount - current_saved, 0.0)
    required = remaining / months_remaining
    feasible = (
        None
        if available_monthly_contribution is None
        else available_monthly_contribution >= required
    )
    return {
        "goal_amount": float(goal_amount),
        "current_saved": float(current_saved),
        "remaining_amount": float(remaining),
        "months_remaining": int(months_remaining),
        "required_monthly_contribution": float(required),
        "available_monthly_contribution": available_monthly_contribution,
        "feasible_under_assumption": feasible,
        "monthly_shortfall": (
            None
            if available_monthly_contribution is None
            else max(required - available_monthly_contribution, 0.0)
        ),
    }


@dataclass(frozen=True)
class Scenario:
    monthly_income: float
    essential_expenses: float
    flexible_expenses: float
    other_expenses: float = 0.0
    income_reduction_rate: float = 0.0
    essential_increase_rate: float = 0.0
    one_time_emergency_expense: float = 0.0
    new_monthly_emi: float = 0.0
    flexible_spending_reduction_rate: float = 0.0


def calculate_scenario(scenario: Scenario) -> dict:
    numeric_values = asdict(scenario)
    if any(not math.isfinite(value) or value < 0 for value in numeric_values.values()):
        raise ValueError("Scenario values must be finite and non-negative.")
    for key in (
        "income_reduction_rate",
        "essential_increase_rate",
        "flexible_spending_reduction_rate",
    ):
        if numeric_values[key] > 1:
            raise ValueError(f"{key} must be between 0 and 1.")
    baseline_monthly = (
        scenario.monthly_income
        - scenario.essential_expenses
        - scenario.flexible_expenses
        - scenario.other_expenses
    )
    scenario_income = scenario.monthly_income * (1 - scenario.income_reduction_rate)
    scenario_essential = (
        scenario.essential_expenses * (1 + scenario.essential_increase_rate)
    )
    scenario_flexible = scenario.flexible_expenses * (
        1 - scenario.flexible_spending_reduction_rate
    )
    scenario_monthly = (
        scenario_income
        - scenario_essential
        - scenario_flexible
        - scenario.other_expenses
        - scenario.new_monthly_emi
    )
    return {
        "baseline_monthly_cash_flow": float(baseline_monthly),
        "scenario_monthly_cash_flow": float(scenario_monthly),
        "monthly_change": float(scenario_monthly - baseline_monthly),
        "one_time_emergency_expense": float(scenario.one_time_emergency_expense),
        "first_month_cash_flow_after_emergency": float(
            scenario_monthly - scenario.one_time_emergency_expense
        ),
        "income_after_reduction": float(scenario_income),
        "essential_expenses_after_increase": float(scenario_essential),
        "flexible_expenses_after_reduction": float(scenario_flexible),
        "assumptions": asdict(scenario),
        "disclaimer": "Hypothetical deterministic estimate; not an observed outcome or guarantee.",
    }


def build_recommendations(
    frame: pd.DataFrame, assumed_reduction_rate: float = 0.10
) -> pd.DataFrame:
    if not 0 <= assumed_reduction_rate <= 1:
        raise ValueError("assumed_reduction_rate must be between 0 and 1.")
    ledger = add_expense_flexibility(frame)
    eligible = ledger.loc[
        ledger["flexibility_class"].eq("Potentially adjustable (category proxy)")
    ]
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "category", "observed_total", "observed_months", "average_per_observed_month",
                "assumed_reduction_rate", "estimated_monthly_difference", "evidence",
            ]
        )
    statement_months = int(ledger["month_period"].nunique())
    grouped = eligible.groupby("final_category").agg(
        observed_total=("amount", "sum"),
        months_with_activity=("month_period", "nunique"),
        observations=("amount", "size"),
    )
    grouped["observed_months"] = statement_months
    grouped["average_per_observed_month"] = grouped["observed_total"] / statement_months
    grouped["assumed_reduction_rate"] = assumed_reduction_rate
    grouped["estimated_monthly_difference"] = (
        grouped["average_per_observed_month"] * assumed_reduction_rate
    )
    grouped["evidence"] = grouped.apply(
        lambda row: (
            f"{int(row['observations'])} observed transactions across "
            f"{int(row['observed_months'])} calendar month(s); "
            "category is a flexibility proxy; reduction is hypothetical."
        ),
        axis=1,
    )
    result = grouped.reset_index(names="category").sort_values(
        "estimated_monthly_difference", ascending=False
    )
    result["confidence"] = "Limited: category-level proxy and short statement history"
    result["suggested_action"] = result["category"].map(
        lambda category: (
            f"Review {category} transactions and test the stated reduction assumption; "
            "confirm purchases are genuinely adjustable first."
        )
    )
    result["priority_rank"] = range(1, len(result) + 1)
    return result


def save_goal(
    goal: dict, database_path: Path = DATA_DIR / "fintwin_state.sqlite3"
) -> None:
    required = {"goal_name", "goal_amount", "current_saved", "target_date"}
    missing = required.difference(goal)
    if missing:
        raise ValueError(f"Goal is missing fields: {', '.join(sorted(missing))}")
    if float(goal["goal_amount"]) < 0 or float(goal["current_saved"]) < 0:
        raise ValueError("Goal values must be non-negative.")
    if not str(goal["goal_name"]).strip():
        raise ValueError("Goal name must not be empty.")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    goal_name TEXT PRIMARY KEY,
                    goal_amount REAL NOT NULL CHECK(goal_amount >= 0),
                    current_saved REAL NOT NULL CHECK(current_saved >= 0),
                    target_date TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT INTO goals (goal_name, goal_amount, current_saved, target_date)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(goal_name) DO UPDATE SET
                    goal_amount=excluded.goal_amount,
                    current_saved=excluded.current_saved,
                    target_date=excluded.target_date,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    str(goal["goal_name"]).strip(),
                    float(goal["goal_amount"]),
                    float(goal["current_saved"]),
                    str(goal["target_date"]),
                ),
            )


def delete_goal(goal_name: str, database_path: Path = DATA_DIR / "fintwin_state.sqlite3") -> None:
    if not goal_name.strip():
        raise ValueError("A goal name is required.")
    if not database_path.exists():
        return
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                "DELETE FROM goals WHERE goal_name = ?", (goal_name.strip(),)
            )


def set_anomaly_review_status(
    transaction_id: str,
    status: str,
    database_path: Path = DATA_DIR / "fintwin_state.sqlite3",
) -> None:
    allowed = {"Unreviewed", "Reviewed - expected", "Reviewed - investigate"}
    if status not in allowed:
        raise ValueError(f"Review status must be one of: {', '.join(sorted(allowed))}.")
    if not transaction_id.strip():
        raise ValueError("A transaction ID is required to save review status.")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS anomaly_reviews (
                    transaction_id TEXT PRIMARY KEY,
                    review_status TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT INTO anomaly_reviews (transaction_id, review_status)
                VALUES (?, ?)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    review_status=excluded.review_status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (transaction_id.strip(), status),
            )


def clear_application_state(
    *,
    confirmed: bool,
    database_path: Path = DATA_DIR / "fintwin_state.sqlite3",
) -> None:
    if not confirmed:
        raise ValueError("Explicit confirmation is required to clear local app state.")
    if not database_path.exists():
        return
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            for table in ("goals", "anomaly_reviews"):
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if exists:
                    connection.execute(f"DELETE FROM {table}")


def get_anomaly_review_statuses(
    database_path: Path = DATA_DIR / "fintwin_state.sqlite3",
) -> dict[str, str]:
    if not database_path.exists():
        return {}
    with closing(sqlite3.connect(database_path)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='anomaly_reviews'"
        ).fetchone()
        if exists is None:
            return {}
        return dict(
            connection.execute(
                "SELECT transaction_id, review_status FROM anomaly_reviews"
            ).fetchall()
        )


def load_goals(database_path: Path = DATA_DIR / "fintwin_state.sqlite3") -> pd.DataFrame:
    if not database_path.exists():
        return pd.DataFrame(
            columns=["goal_name", "goal_amount", "current_saved", "target_date", "updated_at"]
        )
    with closing(sqlite3.connect(database_path)) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='goals'",
            connection,
        )
        if tables.empty:
            return pd.DataFrame(
                columns=["goal_name", "goal_amount", "current_saved", "target_date", "updated_at"]
            )
        return pd.read_sql_query(
            "SELECT goal_name, goal_amount, current_saved, target_date, updated_at "
            "FROM goals ORDER BY target_date",
            connection,
        )
