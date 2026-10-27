from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import streamlit as st

from src.common import BASE_CATEGORIES
from src.auth import (
    AuthenticationError,
    authenticate,
    get_session_user,
    register_user,
    revoke_session,
)
from src.finance_store import (
    count_import_duplicate_candidates,
    delete_user_goal,
    get_import_history,
    get_goal_plan_preferences,
    get_latest_goal_plan_snapshot,
    list_goal_plan_feedback,
    get_user_financial_profile,
    get_user_anomaly_statuses,
    get_user_feedback,
    get_user_transactions,
    list_user_goals,
    list_user_recommendations,
    save_import,
    save_goal_plan_snapshot,
    record_goal_plan_feedback,
    save_recommendations,
    save_user_financial_profile,
    save_user_correction,
    save_user_goal,
    set_recommendation_status,
    set_goal_plan_adjustment_status,
    set_user_anomaly_status,
)
from src.financial_intelligence import (
    Scenario,
    add_expense_flexibility,
    build_recommendations,
    calculate_goal,
    calculate_scenario,
    canonicalize_transactions,
    cluster_spending,
    detect_anomalies,
    detect_recurring_transactions,
    forecast_monthly_expenses,
    monthly_summary,
)
from src.phonepe_upload import (
    MAX_UPLOAD_BYTES,
    ParsedStatement,
    PhonePeUploadError,
    parse_phonepe_csv,
)
from src.predict_phonepe import _load_model, predict_transactions
from src.recommendations import build_personalized_suggestions
from src.goal_planner import (
    build_goal_savings_plan,
    estimate_monthly_spending,
    minor_to_rupees,
    reconcile_savings_plan,
    rupees_to_minor,
)


st.set_page_config(
    page_title="FinTwin AI | Personal Finance Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ft-bg: #F8F7FC;
      --ft-surface: #FFFFFF;
      --ft-navy: #594481;
      --ft-blue: #8064D8;
      --ft-soft: #EDE5FF;
      --ft-border: #E3DDF0;
      --ft-text: #393047;
      --ft-muted: #77708A;
      --ft-green: #78BFA5;
      --ft-amber: #E8BF78;
      --ft-red: #DE8F9B;
    }

    .stApp {
      background: var(--ft-bg);
      color: var(--ft-text);
    }

    [data-testid="stMain"] {
      background: transparent;
    }

    [data-testid="stSidebar"] {
      background: #EDE5FF;
      border-right: 1px solid var(--ft-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      color: var(--ft-text);
    }

    h1, h2, h3 {
      color: var(--ft-navy);
      letter-spacing: -0.02em;
      font-weight: 700;
    }

    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.05rem !important; }

    div[data-testid="stMetric"] {
      background: #FFFFFF;
      border: 1px solid var(--ft-border);
      padding: 18px 16px;
      border-radius: 10px;
      box-shadow: 0 3px 12px rgba(89, 68, 129, 0.05);
    }

    div[data-testid="stMetric"] label {
      color: var(--ft-muted) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: var(--ft-navy) !important;
      font-weight: 700;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
      border-radius: 8px;
      border: 1px solid #D6C9EF;
      background: #EDE5FF;
      color: #594481;
      font-weight: 600;
      min-height: 2.5rem;
      transition: background-color 0.15s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
      background: #E3D5FF;
      border-color: #B69AF7;
      color: #40305F;
    }

    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
      background: #8064D8;
      color: #FFFFFF;
      border: none;
    }

    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
      background: #594481;
      color: #FFFFFF;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
      border-radius: 8px;
      border-color: #D6C9EF;
      background: #FFFFFF;
    }

    [data-testid="stDataFrame"] {
      border: 1px solid var(--ft-border);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: none;
    }

    [data-testid="stAlert"] {
      border-radius: 8px;
      border: 1px solid var(--ft-border);
    }

    [data-testid="stHeader"] {
      background: transparent;
    }

    #MainMenu, footer {
      visibility: hidden;
    }

    .ft-kicker {
      color: var(--ft-blue);
      font-size: 0.78rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      font-weight: 700;
    }

    .ft-note {
      color: var(--ft-muted);
      font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_classifier() -> object:
    return _load_model()


def render_public_page() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ft-bg: #FFF6F2;
          --ft-surface: #FFFFFF;
          --ft-navy: #80566B;
          --ft-blue: #B87591;
          --ft-soft: #FFE0D5;
          --ft-border: #F2D7DF;
          --ft-text: #493642;
          --ft-muted: #947D89;
          --ft-green: #78BFA5;
          --ft-amber: #E8BF78;
          --ft-red: #DE8F83;
        }

        .stApp {
          background: #FFF6F2;
          color: var(--ft-text);
        }

        [data-testid="stMain"] {
          background: transparent;
        }

        [data-testid="stMainBlockContainer"] {
          padding-top: 3rem;
          padding-bottom: 3rem;
          max-width: 1250px;
        }

        h1, h2, h3 {
          color: var(--ft-navy);
          letter-spacing: -0.03em;
          font-weight: 700;
        }

        h1 {
          font-size: 2.8rem !important;
          line-height: 1.2 !important;
        }

        h2 {
          font-size: 1.6rem !important;
        }

        h3 {
          font-size: 1.1rem !important;
        }

        .ft-kicker {
          color: #B87591;
          font-size: 0.78rem;
          letter-spacing: .12em;
          text-transform: uppercase;
          font-weight: 700;
          margin-bottom: 1rem;
        }

        [data-testid="stMainBlockContainer"] > div > div > div > div[data-testid="stVerticalBlock"] {
          gap: 1rem;
        }

        div[data-testid="stForm"] {
          background: #FFFFFF;
          border: 1px solid #F2D7DF;
          border-radius: 18px;
          padding: 1.5rem;
          box-shadow: 0 8px 28px rgba(128, 86, 107, 0.08);
        }

        div[data-testid="stForm"] label {
          color: #80566B;
          font-weight: 500;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
          border-radius: 9px;
          border-color: #F0CFDA;
          background: #FFFCFB;
        }

        div[data-baseweb="input"] input {
          color: #493642;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
          border-radius: 9px;
          border: 1px solid #E9BFD0;
          background: #FFE0D5;
          color: #80566B;
          font-weight: 600;
          min-height: 2.8rem;
          transition: all 0.2s ease;
        }

        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
          background: #B87591;
          color: #FFFFFF;
          border: 1px solid #B87591;
        }

        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
          background: #965C78;
          border-color: #965C78;
          color: #FFFFFF;
        }

        div.stButton > button:hover {
          background: #F7BFAF;
          border-color: #DE8F83;
          color: #493642;
        }

        div[data-testid="stRadio"] label {
          color: #80566B;
        }

        div[data-testid="stRadio"] [role="radiogroup"] {
          gap: 0.5rem;
        }

        div[data-testid="stCaptionContainer"] {
          color: #947D89;
        }

        [data-testid="stAlert"] {
          border-radius: 10px;
          border: 1px solid #F2D7DF;
        }

        [data-testid="stHeader"] {
          background: transparent;
        }

        #MainMenu, footer {
          visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ft-kicker">FinTwin AI · Personal finance intelligence</div>',
        unsafe_allow_html=True,
    )
    st.title("FinTwin AI — Your Personal Financial Intelligence Beyond Spending, Into Insights...")
    st.write(
        "Turn your PhonePe transaction history into meaningful financial insights. Discover where your money goes, understand your spending habits, identify unusual transactions, and explore smarter ways to save — all through one intelligent dashboard."
    )
    left, right = st.columns([1.3, 0.7])
    with left:
        st.subheader("A clearer view of your recorded finances")
        st.write(
            "From category-wise spending and recurring payments to cash-flow forecasts",
            "financial goals, and personalized savings scenarios, FinTwin helps you turn"
            "recorded financial activity into clearer decisions."
        )
        st.caption(
            "FinTwin is a local decision-support prototype. It is not a bank, "
            "financial adviser, verified fraud detector, or source of account balances."
        )
    with right:
        mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
        if mode == "Create account":
            with st.form("register_form", clear_on_submit=False):
                name = st.text_input("Full name")
                email = st.text_input("Email address")
                password = st.text_input(
                    "Password", type="password",
                    help="Use at least 12 characters.",
                )
                confirmation = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", type="primary")
            if submitted:
                if password != confirmation:
                    st.error("The passwords do not match.")
                else:
                    try:
                        register_user(name, email, password)
                        user, token, _expires = authenticate(email, password)
                        st.session_state.clear()
                        st.session_state["_fintwin_session_token"] = token
                        st.session_state["_fintwin_user_id"] = user["id"]
                        st.rerun()
                    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
                        st.error(str(error))
        else:
            with st.form("login_form"):
                email = st.text_input("Email address")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary")
            if submitted:
                try:
                    user, token, _expires = authenticate(email, password)
                    st.session_state.clear()
                    st.session_state["_fintwin_session_token"] = token
                    st.session_state["_fintwin_user_id"] = user["id"]
                    st.rerun()
                except AuthenticationError as error:
                    st.error(str(error))
                except (RuntimeError, OSError, sqlite3.Error) as error:
                    st.error(f"Sign-in is unavailable: {error}")

def _get_authenticated_user() -> dict[str, str] | None:
    token = str(st.session_state.get("_fintwin_session_token", ""))
    user = get_session_user(token)
    if user is None:
        if token:
            st.session_state.clear()
        return None
    st.session_state["_fintwin_user_id"] = user["id"]
    return user


def _logout() -> None:
    token = str(st.session_state.get("_fintwin_session_token", ""))
    try:
        revoke_session(token)
    finally:
        st.session_state.clear()
    st.rerun()


def _filter_transactions(frame: pd.DataFrame, user_id: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    period = st.sidebar.selectbox(
        "Analysis period",
        ["All history", "Month", "Year", "Custom dates"],
        key="analysis_period",
    )
    dates = pd.to_datetime(filtered["transaction_date"], errors="coerce")
    if period == "Month":
        months = sorted(dates.dt.to_period("M").astype(str).dropna().unique())
        selected = st.sidebar.selectbox("Month", months, key="analysis_month")
        filtered = filtered.loc[dates.dt.to_period("M").astype(str).eq(selected)]
    elif period == "Year":
        years = sorted(dates.dt.year.dropna().astype(int).unique().tolist())
        year = st.sidebar.selectbox("Year", years, key="analysis_year")
        filtered = filtered.loc[dates.dt.year.eq(year)]
    elif period == "Custom dates":
        min_date = dates.min().date()
        max_date = dates.max().date()
        value = st.sidebar.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
            key="analysis_custom_dates",
        )
        if isinstance(value, tuple) and len(value) == 2:
            start, end = value
            filtered = filtered.loc[
                dates.dt.date.between(start, end, inclusive="both")
            ]

    category_options = ["All"] + sorted(filtered["final_category"].dropna().unique())
    category = st.sidebar.selectbox("Category", category_options, key="analysis_category")
    if category != "All":
        filtered = filtered.loc[filtered["final_category"].eq(category)]
    merchant_options = ["All"] + sorted(filtered["merchant"].dropna().unique())
    merchant = st.sidebar.selectbox("Merchant", merchant_options, key="analysis_merchant")
    if merchant != "All":
        filtered = filtered.loc[filtered["merchant"].eq(merchant)]
    tx_type = st.sidebar.selectbox(
        "Direction", ["All", "DEBIT", "CREDIT"], key="analysis_type"
    )
    if tx_type != "All":
        filtered = filtered.loc[filtered["transaction_type"].eq(tx_type)]
    review_options = ["All"] + sorted(filtered["review_status"].dropna().unique())
    review = st.sidebar.selectbox("Review status", review_options, key="analysis_review")
    if review != "All":
        filtered = filtered.loc[filtered["review_status"].eq(review)]
    history = get_import_history(user_id)
    if not history.empty:
        import_labels = {
            f"{row['original_filename']} · {row['uploaded_at']}": row["id"]
            for row in history.to_dict(orient="records")
        }
        selected_import = st.sidebar.selectbox(
            "Statement import", ["All"] + list(import_labels), key="analysis_import"
        )
        if selected_import != "All":
            filtered = filtered.loc[
                filtered["import_id"].eq(import_labels[selected_import])
            ]
    return filtered.copy()


def upload_page(user: dict[str, str]) -> None:
    page_header(
        "Statement upload & history",
        "Upload PhonePe CSV statements. Preview and validation happen before user-scoped records are committed.",
    )
    st.write(f"Welcome, {user['name']}. Your imported transactions are private to this account.")
    try:
        configured_limit = int(
            os.environ.get(
                "FINTWIN_MAX_UPLOAD_MB",
                str(MAX_UPLOAD_BYTES // (1024 * 1024)),
            )
        )
    except ValueError:
        st.error("FINTWIN_MAX_UPLOAD_MB must be a whole-number file size.")
        return
    if not 1 <= configured_limit <= 100:
        st.error("FINTWIN_MAX_UPLOAD_MB must be between 1 and 100.")
        return
    max_bytes = configured_limit * 1024 * 1024
    uploads = st.file_uploader(
        "Choose one or more PhonePe CSV statements",
        type=["csv"],
        accept_multiple_files=True,
        help=f"Maximum {configured_limit} MB per file. Files are parsed in memory and are not saved as public uploads.",
    )
    statements: list[ParsedStatement] = []
    cache_key = f"_fintwin_upload_cache_{user['id']}"
    imported_key = f"_fintwin_imported_hashes_{user['id']}"
    cached_statements = st.session_state.setdefault(cache_key, {})
    imported_hashes = st.session_state.setdefault(imported_key, set())
    if uploads:
        active_hashes: set[str] = set()
        for upload in uploads:
            try:
                content = upload.getvalue()
                content_hash = hashlib.sha256(content).hexdigest()
                if content_hash in imported_hashes:
                    st.caption(
                        f"{upload.name}: this exact file was already imported into this account."
                    )
                    continue
                if content_hash in active_hashes:
                    st.caption(
                        f"{upload.name}: identical file content is already selected."
                    )
                    continue
                active_hashes.add(content_hash)
                statement = cached_statements.get(content_hash)
                if statement is None:
                    statement = parse_phonepe_csv(
                        content, upload.name, max_bytes=max_bytes
                    )
                    cached_statements[content_hash] = statement
                statements.append(statement)
            except PhonePeUploadError as error:
                st.error(f"{upload.name}: {error}")
        for inactive_hash in set(cached_statements).difference(active_hashes):
            del cached_statements[inactive_hash]
        for statement in statements:
            st.subheader(statement.filename)
            existing_duplicates = count_import_duplicate_candidates(
                user["id"], statement
            )
            metrics = st.columns(5)
            metrics[0].metric("Total rows", statement.total_rows)
            metrics[1].metric("Valid rows", statement.valid_rows)
            metrics[2].metric("Rejected rows", len(statement.rejected))
            metrics[3].metric(
                "Duplicate candidates",
                statement.duplicate_candidates
                + existing_duplicates["exact"]
                + existing_duplicates["possible"],
            )
            metrics[4].metric(
                "Date coverage",
                f"{statement.date_start or '—'} – {statement.date_end or '—'}",
            )
            preview_columns = [
                "transaction_date", "transaction_time", "merchant",
                "transaction_type", "amount", "category", "duplicate_type",
            ]
            st.dataframe(
                statement.processed[preview_columns].head(12),
                hide_index=True,
                width="stretch",
            )
            if existing_duplicates["exact"] or existing_duplicates["possible"]:
                st.caption(
                    f"Already in this account: {existing_duplicates['exact']} exact "
                    f"transaction-ID match(es); {existing_duplicates['possible']} "
                    "possible matches without IDs. Exact matches are skipped; possible "
                    "matches remain reviewable."
                )
            if statement.rejected:
                st.warning(f"{len(statement.rejected)} row(s) will be excluded and recorded.")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"source_line": row.source_line, "reason": row.reason}
                            for row in statement.rejected
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
    else:
        cached_statements.clear()
        st.info(
            "No statement has been selected. Your existing financial workspaces "
            "remain available after an import."
        )
    last_import_notice = st.session_state.pop("_fintwin_last_import", None)
    if last_import_notice:
        if last_import_notice["imported"]:
            st.success(
                "Completed imports: " + " · ".join(last_import_notice["imported"])
            )
        if last_import_notice["failures"]:
            st.error(
                "Some files could not be imported; each failed import rolled back. "
                + " · ".join(last_import_notice["failures"])
            )
        overview_page(user_transactions(user), user)
    if statements and st.button("Import valid rows", type="primary"):
        try:
            classifier = load_classifier()
            imported = []
            failures = []
            for statement in statements:
                try:
                    processed = statement.processed.drop(
                        columns=["amount_minor", "duplicate_type"], errors="ignore"
                    ).copy()
                    predictions = predict_transactions(
                        processed,
                        model=classifier,
                        feedback=get_user_feedback(user["id"]),
                    )
                    result = save_import(user["id"], statement, predictions)
                    imported_hashes.add(statement.file_sha256)
                    imported.append(
                        f"{statement.filename}: {result['accepted_rows']} saved, "
                        f"{result['rejected_rows']} rejected, "
                        f"{result['duplicate_rows']} exact duplicates skipped, "
                        f"{result['possible_duplicate_rows']} possible matches retained for review"
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    failures.append(f"{statement.filename}: {error}")
            for statement in statements:
                cached_statements.pop(statement.file_sha256, None)
            if imported:
                st.session_state["_fintwin_last_import"] = {
                    "imported": imported,
                    "failures": failures,
                }
            if failures:
                if not imported:
                    st.error(
                        "Some files could not be imported; each failed import rolled back. "
                        + " · ".join(failures)
                    )
            if imported:
                st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Import failed; the affected file was not partially saved: {error}")

    st.subheader("Your import history")
    history = get_import_history(user["id"])
    if history.empty:
        st.caption("Imports you make will appear here.")
    else:
        st.dataframe(
            history[
                [
                    "original_filename", "uploaded_at", "period_start", "period_end",
                    "total_rows", "accepted_rows", "rejected_rows", "duplicate_rows",
                    "possible_duplicate_rows", "status", "error_summary",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        import_id = st.selectbox(
            "Inspect transactions from an import",
            history["id"].tolist(),
            format_func=lambda selected_id: history.loc[
                history["id"].eq(selected_id), "original_filename"
            ].iloc[0],
        )
        imported_transactions = get_user_transactions(
            user["id"], import_id=str(import_id)
        )
        if imported_transactions.empty:
            st.info("This import has no saved transaction rows.")
        else:
            columns = [
                column
                for column in (
                    "transaction_date", "transaction_time", "merchant", "amount",
                    "transaction_type", "final_category", "category_source",
                    "review_status",
                )
                if column in imported_transactions
            ]
            st.dataframe(
                imported_transactions[columns],
                hide_index=True,
                width="stretch",
            )


def user_transactions(user: dict[str, str]) -> pd.DataFrame:
    return get_user_transactions(user["id"])


def money(value: float | int) -> str:
    return f"₹{value:,.2f}"


def month_picker(frame: pd.DataFrame, key: str) -> str:
    if frame.empty:
        return ""
    months = sorted(
        pd.to_datetime(frame["transaction_date"]).dt.to_period("M").astype(str).unique()
    )
    default = months[-1]
    return st.selectbox("Statement month", months, index=months.index(default), key=key)


def page_header(title: str, description: str) -> None:
    st.markdown('<div class="ft-kicker">FinTwin AI · Adaptive personal finance intelligence</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def selected_month_records(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["transaction_date"], errors="coerce")
    return frame.loc[dates.dt.to_period("M").astype(str).eq(month)].copy()


def overview_page(frame: pd.DataFrame, user: dict[str, str]) -> None:
    st.markdown(
        """
        <style>
        :root {
          --ft-page-accent: #48A77A;
          --ft-page-soft: #EFFBF5;
          --ft-page-border: #D0F2E0;
        }

        div[data-testid="stMetric"] {
          background: #FFFFFF;
          border: 1px solid var(--ft-page-border);
          border-top: 3px solid var(--ft-page-accent);
          padding: 18px 16px;
          border-radius: 10px;
          box-shadow: 0 3px 12px rgba(72, 167, 122, 0.06);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
          color: #327C5B !important;
          font-weight: 700;
        }

        div[data-testid="stMetric"] label {
          color: #6D897A !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
          color: #48A77A !important;
        }

        [data-testid="stDataFrame"] {
          border: 1px solid #D0F2E0;
          border-radius: 10px;
          overflow: hidden;
        }

        div[data-baseweb="select"] > div {
          border-color: #B9E4CB;
          border-radius: 8px;
        }

        [data-testid="stAlert"] {
          border-radius: 9px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* Main application background */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #EFFBF5 !important;
        }

        /* Sidebar background */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background-color: #E3F5E9 !important;
        }

        /* Main content area */
        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D0F2E0;
            border-top: 3px solid #48A77A;
            padding: 18px 16px;
            border-radius: 10px;
            box-shadow: 0 3px 12px rgba(72, 167, 122, 0.06);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #327C5B !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #D0F2E0;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page_header(
        "Financial overview",
        "Observed transaction activity from your PhonePe statement. Missing inputs are not treated as zero.",
    )
    if frame.empty:
        st.info("No imported transactions are available for this account and period.")
        return
    month = month_picker(frame, "overview_month")
    latest_date = pd.to_datetime(frame["transaction_date"]).max().date()
    if latest_date < date.today():
        st.warning(
            f"The statement ends on {latest_date.isoformat()}; figures are historical "
            f"and may not represent your finances today ({date.today().isoformat()})."
        )
    selected = selected_month_records(frame, month)
    summary = monthly_summary(selected)
    if summary.empty:
        st.info("There are no valid transactions in this period.")
        return
    values = summary.iloc[0]
    ledger = canonicalize_transactions(selected)
    income_value = (
        money(values["salary_income_category_credits"])
        if values["salary_income_category_credit_count"] > 0
        else "Not categorized"
    )
    metrics = st.columns(5)
    metrics[0].metric("Credits categorized as salary/income", income_value)
    metrics[1].metric("Other credits", money(values["other_credits"]))
    metrics[2].metric("Classified consumption debits", money(values["consumption_expenses"]))
    metrics[3].metric("Personal-transfer debits", money(values["debit_transfers"]))
    metrics[4].metric("Net observed cash flow", money(values["net_observed_cash_flow"]))
    st.caption(
        f"Coverage: {values['coverage_start']} to {values['coverage_end']} · "
        f"{int(values['transaction_count'])} transactions. This can be a partial calendar month."
    )
    if values["unclassified_debits"] > 0:
        st.info(
            f"Unclassified debits kept outside consumption totals: "
            f"{money(values['unclassified_debits'])}. These may include transfers or other unknown purposes."
        )

    debit_total = float(
        ledger.loc[ledger["transaction_type"].eq("DEBIT"), "amount"].sum()
    )
    credit_total = float(
        ledger.loc[ledger["transaction_type"].eq("CREDIT"), "amount"].sum()
    )
    largest_debit = ledger.loc[ledger["transaction_type"].eq("DEBIT"), "amount"]
    daily_expenses = ledger.loc[ledger["is_consumption_expense"]].groupby(
        ledger.loc[ledger["is_consumption_expense"], "transaction_date"].dt.date
    )["amount"].sum()
    daily_average = float(daily_expenses.mean()) if not daily_expenses.empty else 0.0
    second_metrics = st.columns(4)
    second_metrics[0].metric("Total recorded debits", money(debit_total))
    second_metrics[1].metric("Total recorded credits", money(credit_total))
    second_metrics[2].metric(
        "Largest recorded debit",
        money(float(largest_debit.max())) if not largest_debit.empty else "—",
    )
    second_metrics[3].metric(
        "Average daily consumption",
        money(daily_average),
        help="Average over dates with classified consumption transactions in this period.",
    )
    category_totals = (
        ledger.loc[ledger["is_consumption_expense"]]
        .groupby("final_category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    try:
        anomaly_count = int(detect_anomalies(selected)["anomaly"].eq(-1).sum())
    except ValueError:
        anomaly_count = 0
    goals = list_user_goals(user["id"])
    goal_progress = (
        float(
            (
                goals["current_saved_minor"].astype(float)
                / goals["target_amount_minor"].replace(0, pd.NA).astype(float)
            )
            .dropna()
            .clip(upper=1)
            .mean()
            * 100
        )
        if not goals.empty else None
    )
    third_metrics = st.columns(4)
    third_metrics[0].metric(
        "Highest spending category",
        str(category_totals.index[0]) if not category_totals.empty else "—",
        money(float(category_totals.iloc[0])) if not category_totals.empty else None,
    )
    third_metrics[1].metric(
        "Transactions needing review",
        int(ledger["final_category"].eq("Needs Review").sum()),
    )
    third_metrics[2].metric("Anomaly signals to review", anomaly_count)
    third_metrics[3].metric(
        "User-entered goal progress",
        f"{goal_progress:.1f}%" if goal_progress is not None else "No saved goals",
        help="Average across saved goals, based on user-entered saved amounts.",
    )

    col_a, col_b = st.columns([1.1, 0.9])
    with col_a:
        st.subheader("Recorded spending by category")
        selected_ledger = add_expense_flexibility(selected)
        expenses = selected_ledger.loc[
            selected_ledger["is_consumption_expense"]
        ]
        category_totals = (
            expenses.groupby("final_category")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        if category_totals.empty:
            st.info("No categorized consumption debits were observed.")
        else:
            st.bar_chart(category_totals, horizontal=True, color="#78BFA5")
    with col_b:
        st.subheader("Recent transactions")
        recent_columns = [
            name for name in (
                "transaction_date", "merchant", "amount", "transaction_type",
                "final_category", "category_source",
            ) if name in selected.columns
        ]
        recent = selected.sort_values("transaction_date", ascending=False)[
            recent_columns
        ].head(8)
        st.dataframe(recent, hide_index=True, width="stretch")
    st.warning(
        "This statement has transaction history but does not establish account balances, "
        "net worth, or a complete monthly income history."
    )


def transaction_page(frame: pd.DataFrame, user: dict[str, str]) -> None:
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #F5F0FF !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background-color: #EAE0FA !important;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: #FFFFFF !important;
            border-color: #D8C8F0 !important;
            border-radius: 9px !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #DDD0F3;
            border-radius: 10px;
            overflow: hidden;
            background-color: #FFFFFF;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #DDD0F3;
            border-top: 3px solid #9878CC;
            padding: 16px;
            border-radius: 10px;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        div[data-testid="stForm"] {
            background-color: #FCFAFF;
            border: 1px solid #DDD0F3;
            border-radius: 12px;
            padding: 18px;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #9878CC;
            color: #FFFFFF;
            border: 1px solid #9878CC;
            border-radius: 9px;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #8261B7;
            color: #FFFFFF;
            border-color: #8261B7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page_header(
        "Transaction intelligence",
        "Search and review classifications. Corrections are saved locally and applied on the next classification pass.",
    )
    if frame.empty:
        st.info("Import a statement to search or review your transactions.")
        return

    month = month_picker(frame, "tx_month")
    visible = selected_month_records(frame, month)

    filter_cols = st.columns([1.3, 1, 1])
    query = filter_cols[0].text_input("Search merchant or description")
    category = filter_cols[1].selectbox(
        "Final category", ["All"] + sorted(visible["final_category"].dropna().unique())
    )
    tx_type = filter_cols[2].selectbox(
        "Transaction type", ["All", "DEBIT", "CREDIT"]
    )

    if query.strip():
        needle = query.strip().casefold()
        text = (
            visible["merchant"].fillna("").astype(str)
            + " "
            + visible["transaction_details"].fillna("").astype(str)
        ).str.casefold()
        visible = visible.loc[text.str.contains(needle, regex=False)]

    if category != "All":
        visible = visible.loc[visible["final_category"].eq(category)]

    if tx_type != "All":
        visible = visible.loc[visible["transaction_type"].eq(tx_type)]

    columns = [
        name for name in (
            "transaction_date", "transaction_time", "merchant", "transaction_details",
            "amount", "transaction_type", "ml_category", "final_category",
            "ml_confidence", "category_source",
        ) if name in visible.columns
    ]

    st.dataframe(
        visible[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "amount": st.column_config.NumberColumn("Amount (INR)", format="₹%.2f"),
            "ml_confidence": st.column_config.NumberColumn("ML confidence (0–1)", format="%.3f"),
        },
    )
    st.caption(f"{len(visible)} matching transaction(s). Transaction and UTR identifiers are hidden.")

    needs_review = frame.loc[frame["final_category"].eq("Needs Review")]
    st.subheader("Correct a category")

    if frame.empty:
        st.info("No transactions are available to correct.")
        return

    options = frame.reset_index(drop=True)
    selected_index = st.selectbox(
        "Choose a transaction",
        range(len(options)),
        format_func=lambda index: (
            f"{options.at[index, 'transaction_date']} · "
            f"{options.at[index, 'merchant']} · "
            f"{money(float(options.at[index, 'amount']))} · "
            f"{options.at[index, 'final_category']}"
        ),
    )

    row = options.iloc[selected_index]

    with st.form("transaction_correction"):
        correction_scope = st.radio(
            "Apply correction to",
            ["This transaction only", "All transactions with this exact merchant"],
            help="Merchant-wide feedback may be inappropriate for a mixed-use merchant. "
            "Use transaction-only unless you have confirmed the merchant's purpose.",
        )
        new_category = st.selectbox(
            "Correct category", BASE_CATEGORIES
        )
        confirm_merchant = False

        if correction_scope.startswith("All transactions"):
            confirm_merchant = st.checkbox(
                "I confirm this merchant consistently represents the selected category."
            )

        submitted = st.form_submit_button(
            "Save correction and refresh predictions", type="primary"
        )

    if submitted:
        if correction_scope.startswith("All transactions") and not confirm_merchant:
            st.error("Confirm the merchant-wide correction or select transaction-only.")
        else:
            try:
                save_user_correction(
                    user["id"],
                    int(row["record_id"]),
                    new_category,
                    merchant_wide=correction_scope.startswith("All transactions"),
                )
                st.success("Correction saved for this account.")
                st.rerun()
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Could not save the correction: {error}")

    # st.caption(f"{len(needs_review)} transactions currently need category review.")
def spending_page(frame: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #FFF4EC !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {
            background-color: #FBE5D7 !important;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border-color: #F0CCB5 !important;
            border-radius: 9px !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #F0D1BE;
            border-radius: 10px;
            overflow: hidden;
            background-color: #FFFFFF;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #F0D1BE;
            border-top: 3px solid #D99068;
            padding: 16px;
            border-radius: 10px;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #D99068;
            color: #FFFFFF;
            border: 1px solid #D99068;
            border-radius: 9px;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #BF7953;
            color: #FFFFFF;
            border-color: #BF7953;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page_header(
        "Spending analytics",
        "Observed spending, transparent category proxies, recurrence evidence, and exploratory clusters.",
    )

    summary = monthly_summary(frame)
    st.subheader("Monthly observed totals")

    if summary.empty:
        st.info("No monthly activity is available.")
    else:
        chart = summary.set_index("month")[
            ["consumption_expenses", "debit_transfers", "net_observed_cash_flow"]
        ]
        st.line_chart(chart, color=["#D99068", "#C49A6C", "#78A98B"])
        st.dataframe(summary, hide_index=True, width="stretch")

    flexibility = add_expense_flexibility(frame)
    st.subheader("Expense flexibility (category-level proxy)")
    st.caption(
        "A category is not proof of a transaction's purpose. These labels are rules-based estimates, not user-verified facts."
    )

    st.dataframe(
        flexibility.loc[
            flexibility["transaction_type"].eq("DEBIT"),
            ["transaction_date", "merchant", "amount", "final_category", "flexibility_class", "flexibility_reason"],
        ],
        hide_index=True,
        width="stretch",
    )

    recommendations = build_recommendations(frame)
    st.subheader("Illustrative adjustment opportunities")
    st.caption("Estimated impact assumes a 10% reduction in observed potentially adjustable categories; it is not a promised saving.")

    if recommendations.empty:
        st.info("The available transactions do not support an adjustable-category estimate.")
    else:
        st.dataframe(recommendations, hide_index=True, width="stretch")

    st.subheader("Recurring payment signals")
    recurring = detect_recurring_transactions(frame)

    if recurring.empty:
        st.info("No repeated merchant transactions provide a recurrence signal.")
    else:
        st.dataframe(recurring, hide_index=True, width="stretch")
        st.caption("Observed patterns cover only the statement period; they do not establish a future commitment.")

    if len(frame) >= 3:
        try:
            clusters, metrics = cluster_spending(frame)
            st.subheader("Exploratory spending clusters")
            st.dataframe(
                metrics["summary"],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"{len(clusters)} transactions grouped using this account's selected "
                "history. Cluster IDs are descriptive, not financial-health ratings."
            )
        except ValueError as error:
            st.caption(f"Clustering unavailable for this selection: {error}")

def twin_page(frame: pd.DataFrame) -> None:
    page_header(
        "Financial twin",
        "A derived state from the available ledger. It refreshes when transactions or corrections change.",
    )
    summaries = monthly_summary(frame)
    if summaries.empty:
        st.info("No ledger history is available.")
        return
    month = month_picker(frame, "twin_month")
    selected = selected_month_records(frame, month)
    row = monthly_summary(selected).iloc[0]
    st.subheader(f"Observed state · {month}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Consumption debits", money(row["consumption_expenses"]))
    col2.metric("Transfers", money(row["debit_transfers"]))
    col3.metric("Recorded net cash flow", money(row["net_observed_cash_flow"]))
    st.metric("Other credits (source not verified)", money(row["other_credits"]))
    if row["salary_income_category_credit_count"]:
        st.metric(
            "Credits categorized as salary/income",
            money(row["salary_income_category_credits"]),
        )
    else:
        st.info(
            "No credits were categorized as salary/income in this period. "
            "Other credits are not assumed to be income."
        )
    st.markdown(
        "- **Available balance / net worth:** not supplied by the statement.\n"
        "- **Savings progress:** cannot be inferred without a verified starting balance or contribution history.\n"
        "- **Debt and EMI obligations:** not established by the available transaction schema.\n"
        "- **Health score:** withheld because income, balances, and obligations are incomplete."
    )
    st.subheader("State coverage")
    st.dataframe(summaries, hide_index=True, width="stretch")
    st.caption("No missing balance, income, or savings value is imputed as zero.")


def forecasting_page(frame: pd.DataFrame) -> None:
    page_header(
        "Cash-flow forecasting",
        "Chronological, simple expense baselines. Predictions are estimates and no salary series is inferred.",
    )
    forecast = forecast_monthly_expenses(frame)
    history = monthly_summary(frame)
    st.subheader("Observed history")
    st.dataframe(history, hide_index=True, width="stretch")
    if not forecast["available"]:
        st.warning(f"Forecast unavailable: {forecast['reason']}")
        return
    if not forecast["is_current_forecast"]:
        st.warning(
            f"The next month in the data ({forecast['forecast_month']}) has already passed. "
            f"The statement ends in {forecast['last_observed_month']}; this is a historical "
            "baseline demonstration, not a current forecast."
        )
    st.subheader(f"Baseline estimate after last observation · {forecast['forecast_month']}")
    cols = st.columns(2)
    cols[0].metric("Historical mean estimate", money(forecast["forecast_expense_historical_mean"]))
    cols[1].metric("Three-month moving-average estimate", money(forecast["forecast_expense_moving_average_3_month"]))
    if forecast["historical_mean_mae"] is None:
        st.info("There are too few monthly observations for a meaningful holdout error calculation.")
    else:
        st.write(
            f"Expanding chronological validation across "
            f"{forecast['chronological_validation_origins']} origin(s): "
            f"historical-mean MAE {money(forecast['historical_mean_mae'])}; "
            f"moving-average MAE {money(forecast['moving_average_mae'])}."
        )
    st.warning(forecast["limitation"])
    st.info(
        "Income forecasting is unavailable because this statement does not contain a validated recurring income series."
    )


def scenario_page(frame: pd.DataFrame) -> None:
    page_header(
        "Scenario lab",
        "Test explicit what-if assumptions against observed spending. Results are deterministic estimates, not predictions or guarantees.",
    )
    if frame.empty:
        st.info("Import transactions before estimating expense assumptions.")
        return
    month = month_picker(frame, "scenario_month")
    st.session_state.setdefault("scenario_comparison", [])
    selected = selected_month_records(frame, month)
    ledger = add_expense_flexibility(selected)
    debits = ledger.loc[ledger["transaction_type"].eq("DEBIT")]
    consumption = debits.loc[~debits["is_transfer"]]
    essential = float(
        consumption.loc[
            consumption["flexibility_class"].eq("Potentially essential (category proxy)"),
            "amount",
        ].sum()
    )
    flexible = float(
        consumption.loc[
            consumption["flexibility_class"].eq("Potentially adjustable (category proxy)"),
            "amount",
        ].sum()
    )
    other = float(
        consumption.loc[
            consumption["flexibility_class"].eq("Unknown / review"), "amount"
        ].sum()
    )
    st.caption(
        f"Observed categories in {month}: potentially essential {money(essential)}, "
        f"potentially adjustable {money(flexible)}, unclassified debit outflow "
        f"{money(other)} (included as an unchanged other outflow in this scenario). "
        "Statement coverage may be partial."
    )
    latest_date = pd.to_datetime(frame["transaction_date"]).max().date()
    if latest_date < date.today():
        st.warning(
            f"Scenario expenses are based on statement data ending {latest_date.isoformat()}, "
            "not verified current spending."
        )
    with st.form("scenario_inputs"):
        income_text = st.text_input(
            "Monthly take-home income assumption (INR; enter explicitly)",
            help="No income is inferred if this is blank.",
        )
        reduction = st.slider("Income reduction", 0, 100, 0, format="%d%%")
        essential_increase = st.slider("Increase in essential costs", 0, 100, 0, format="%d%%")
        flexible_cut = st.slider("Reduction in potentially adjustable expenses", 0, 100, 0, format="%d%%")
        new_emi = st.number_input("Additional monthly EMI assumption (INR)", min_value=0.0, value=0.0, step=500.0)
        emergency = st.number_input("One-time emergency expense assumption (INR)", min_value=0.0, value=0.0, step=500.0)
        submitted = st.form_submit_button("Calculate and compare", type="primary")
    if submitted:
        try:
            if not income_text.strip():
                raise ValueError("Enter a monthly income assumption to calculate cash flow.")
            income = float(income_text.replace(",", "").replace("₹", "").strip())
            if not math.isfinite(income) or income < 0:
                raise ValueError("Monthly income must be a finite, non-negative amount.")
            result = calculate_scenario(
                Scenario(
                    monthly_income=income,
                    essential_expenses=essential,
                    flexible_expenses=flexible,
                    other_expenses=other,
                    income_reduction_rate=reduction / 100,
                    essential_increase_rate=essential_increase / 100,
                    one_time_emergency_expense=emergency,
                    new_monthly_emi=new_emi,
                    flexible_spending_reduction_rate=flexible_cut / 100,
                )
            )
            st.subheader("Scenario result")
            cols = st.columns(3)
            cols[0].metric("Baseline monthly cash flow", money(result["baseline_monthly_cash_flow"]))
            cols[1].metric("Scenario monthly cash flow", money(result["scenario_monthly_cash_flow"]))
            cols[2].metric("Change per month", money(result["monthly_change"]))
            st.write(
                "First-month cash flow after the one-time emergency assumption: "
                f"**{money(result['first_month_cash_flow_after_emergency'])}**"
            )
            st.json(result["assumptions"])
            st.caption(result["disclaimer"])
            st.session_state["scenario_comparison"].append(
                {
                    "scenario": f"Scenario {len(st.session_state['scenario_comparison']) + 1}",
                    "baseline_monthly_cash_flow": result["baseline_monthly_cash_flow"],
                    "scenario_monthly_cash_flow": result["scenario_monthly_cash_flow"],
                    "monthly_change": result["monthly_change"],
                    "first_month_after_emergency": result[
                        "first_month_cash_flow_after_emergency"
                    ],
                }
            )
        except (ValueError, TypeError) as error:
            st.error(str(error))
    st.subheader("Scenario comparison")
    st.caption("Results are held in this browser session only; assumptions are not historical transactions.")
    if st.session_state["scenario_comparison"]:
        st.dataframe(
            pd.DataFrame(st.session_state["scenario_comparison"]),
            hide_index=True,
            width="stretch",
        )
        if st.button("Clear scenario comparison"):
            st.session_state["scenario_comparison"] = []
            st.rerun()
    else:
        st.info("Calculate a scenario to add a comparison row.")


def goals_page(user: dict[str, str]) -> None:
    page_header(
        "Financial goals",
        "Goals are stored privately for this account. Contribution feasibility is calculated only from values you provide.",
    )
    goals = list_user_goals(user["id"])
    goal_options = {"Create a new goal": None}
    if not goals.empty:
        goal_options.update(
            {
                f"{row['title']} · {row['target_date']}": row["id"]
                for row in goals.to_dict(orient="records")
            }
        )
    selected_label = st.selectbox("Goal to edit", list(goal_options))
    selected_id = goal_options[selected_label]
    selected_goal = (
        goals.loc[goals["id"].eq(selected_id)].iloc[0]
        if selected_id is not None
        else None
    )
    with st.form("goal_form"):
        name = st.text_input(
            "Goal name",
            value=str(selected_goal["title"]) if selected_goal is not None else "",
        )
        amount = st.number_input(
            "Target amount (INR)",
            min_value=0.0,
            value=(
                int(selected_goal["target_amount_minor"]) / 100
                if selected_goal is not None else 0.0
            ),
            step=1000.0,
        )
        saved = st.number_input(
            "Already saved (INR; user supplied)",
            min_value=0.0,
            value=(
                int(selected_goal["current_saved_minor"]) / 100
                if selected_goal is not None else 0.0
            ),
            step=1000.0,
        )
        target = st.date_input(
            "Target date",
            value=(
                pd.Timestamp(selected_goal["target_date"]).date()
                if selected_goal is not None else date.today()
            ),
        )
        submitted = st.form_submit_button("Save or update goal", type="primary")
    if submitted:
        try:
            if not name.strip() or amount <= 0 or target <= date.today():
                raise ValueError("Enter a goal name, positive target amount, and a future target date.")
            amount_minor = int(
                (Decimal(str(amount)) * 100).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            saved_minor = int(
                (Decimal(str(saved)) * 100).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            save_user_goal(
                user["id"],
                name,
                amount_minor,
                saved_minor,
                target.isoformat(),
                goal_id=str(selected_id) if selected_id is not None else None,
            )
            st.success("Goal saved for this account.")
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))
    goals = list_user_goals(user["id"])
    st.subheader("Saved goals")
    if goals.empty:
        st.info("No goals have been saved yet.")
        return
    display_goals = goals.copy()
    display_goals["target_amount"] = display_goals["target_amount_minor"] / 100
    display_goals["current_saved"] = display_goals["current_saved_minor"] / 100
    st.dataframe(
        display_goals[
            ["title", "target_amount", "current_saved", "target_date", "status"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "target_amount": st.column_config.NumberColumn("Target (INR)", format="₹%.2f"),
            "current_saved": st.column_config.NumberColumn("Saved (INR)", format="₹%.2f"),
        },
    )
    goal_ids = goals["id"].tolist()
    selected_goal_id = st.selectbox(
        "Evaluate goal",
        goal_ids,
        format_func=lambda goal_id: goals.loc[
            goals["id"].eq(goal_id), "title"
        ].iloc[0],
    )
    goal = goals.loc[goals["id"].eq(selected_goal_id)].iloc[0]
    target_date = pd.Timestamp(goal["target_date"]).date()
    if target_date <= date.today():
        st.warning("This goal's target date has passed; edit it to reassess the deadline.")
    else:
        months = max(
            (pd.Period(pd.Timestamp(target_date), freq="M") - pd.Period(date.today(), freq="M")).n + 1,
            1,
        )
        contribution_text = st.text_input(
            "Monthly contribution available for this goal (INR; optional)",
            key="goal_contribution",
        )
        try:
            contribution = (
                float(contribution_text.replace(",", "").replace("₹", ""))
                if contribution_text.strip()
                else None
            )
            if contribution is not None and (not math.isfinite(contribution) or contribution < 0):
                raise ValueError("Contribution must be finite and non-negative.")
            result = calculate_goal(
                int(goal["target_amount_minor"]) / 100,
                int(goal["current_saved_minor"]) / 100,
                months,
                contribution,
            )
        except ValueError as error:
            st.error(str(error))
            result = None
        if result is not None:
            st.write(
                f"Required contribution: **{money(result['required_monthly_contribution'])}/month** "
                f"for {months} month(s)."
            )
            alternate_months = st.number_input(
                "Compare an alternative deadline (months from now)",
                min_value=1,
                value=max(months + 3, 1),
                step=1,
            )
            alternate_result = calculate_goal(
                int(goal["target_amount_minor"]) / 100,
                int(goal["current_saved_minor"]) / 100,
                int(alternate_months),
            )
            st.write(
                f"At {int(alternate_months)} months, required contribution would be "
                f"**{money(alternate_result['required_monthly_contribution'])}/month**."
            )
            if result["feasible_under_assumption"] is None:
                st.info("Feasibility is unknown until you provide an available monthly contribution.")
            elif result["feasible_under_assumption"]:
                st.success("Feasible under the contribution assumption entered.")
            else:
                st.warning(
                    f"Shortfall under that assumption: {money(result['monthly_shortfall'])}/month."
                )
    if st.button("Delete selected goal"):
        delete_user_goal(user["id"], selected_goal_id)
        st.rerun()


def _profile_amount_input_value(value: int | None) -> str:
    rupees = minor_to_rupees(value)
    return "" if rupees is None else f"{rupees:.2f}"


def _parse_optional_profile_amount(value: str) -> int | None:
    return rupees_to_minor(value) if value.strip() else None


def savings_page(frame: pd.DataFrame, user: dict[str, str]) -> None:
    page_header(
        "Savings suggestions",
        "Transaction-based observations and an account-specific, evidence-aware goal planner.",
    )
    reduction = st.slider(
        "Illustrative reduction assumption for general suggestions",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        format="%d%%",
    )
    st.caption(
        "This slider changes only hypothetical general suggestions. Goal affordability "
        "uses your saved financial inputs and the planner's separate conservative category limits."
    )

    try:
        profile = get_user_financial_profile(user["id"])
        goals = list_user_goals(user["id"])
    except (OSError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Could not load this account's savings-planning data: {error}")
        return

    st.subheader("Financial profile")
    st.caption(
        "Enter values you have confirmed. Blank fields remain unavailable; use 0 only "
        "when you explicitly confirm there is no amount. Income is never inferred from credits."
    )
    with st.form(f"goalpilot_profile_{user['id']}"):
        income_column, balance_column, target_column = st.columns(3)
        income_text = income_column.text_input(
            "Monthly take-home income (INR)",
            value=_profile_amount_input_value(profile["monthly_income_minor"]),
            help="Confirmed net income provided by you, not inferred from transaction credits.",
        )
        emergency_balance_text = balance_column.text_input(
            "Emergency-fund balance (INR)",
            value=_profile_amount_input_value(profile["emergency_balance_minor"]),
            help="Current reserve balance; it is not counted as new monthly savings.",
        )
        emergency_target_text = target_column.text_input(
            "Emergency-fund target (INR)",
            value=_profile_amount_input_value(profile["emergency_target_minor"]),
        )
        contribution_column, commitments_column, buffer_column = st.columns(3)
        emergency_contribution_text = contribution_column.text_input(
            "Planned monthly emergency contribution (INR)",
            value=_profile_amount_input_value(profile["emergency_contribution_minor"]),
            help="This flow is protected and deducted from the monthly surplus.",
        )
        commitments_text = commitments_column.text_input(
            "Additional monthly commitments (INR)",
            value=_profile_amount_input_value(
                profile["additional_commitments_minor"]
            ),
            help="Include only commitments not already represented in imported expenses.",
        )
        buffer_text = buffer_column.text_input(
            "Protected monthly cash-buffer contribution (INR)",
            value=_profile_amount_input_value(
                profile["protected_buffer_contribution_minor"]
            ),
            help="A monthly set-aside flow, not a minimum account balance.",
        )
        protected_categories = st.multiselect(
            "Categories to protect from proposed reductions",
            BASE_CATEGORIES,
            default=profile["protected_categories"],
            help="Essential proxy categories are protected automatically as well.",
        )
        coverage_confirmed = st.checkbox(
            "I have checked that every calendar month between this account's earliest "
            "and latest transaction is fully covered",
            value=profile["statement_coverage_confirmed"],
            help=(
                "Observed transaction dates alone do not prove statement coverage. "
                "Only confirm this after checking your statements; unverified coverage "
                "prevents an affordability conclusion."
            ),
        )
        profile_submitted = st.form_submit_button(
            "Save financial profile", type="primary"
        )
    if profile_submitted:
        try:
            save_user_financial_profile(
                user["id"],
                {
                    "monthly_income_minor": _parse_optional_profile_amount(income_text),
                    "emergency_balance_minor": _parse_optional_profile_amount(
                        emergency_balance_text
                    ),
                    "emergency_target_minor": _parse_optional_profile_amount(
                        emergency_target_text
                    ),
                    "emergency_contribution_minor": _parse_optional_profile_amount(
                        emergency_contribution_text
                    ),
                    "additional_commitments_minor": _parse_optional_profile_amount(
                        commitments_text
                    ),
                    "protected_buffer_contribution_minor": _parse_optional_profile_amount(
                        buffer_text
                    ),
                    "protected_categories": protected_categories,
                    "statement_coverage_confirmed": coverage_confirmed,
                },
            )
            st.success("Financial profile saved to this account.")
            st.rerun()
        except ValueError as error:
            st.error(str(error))

    st.divider()
    st.subheader("GoalPilot affordability and plan")
    try:
        planning_frame = user_transactions(user)
        if not planning_frame.empty:
            planning_frame = planning_frame.copy()
            planning_frame["coverage_validated"] = bool(
                profile["statement_coverage_confirmed"]
            )
        spending = estimate_monthly_spending(planning_frame)
        import_history = get_import_history(user["id"])
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Could not estimate this account's spending: {error}")
        return
    if spending["coverage_verified"]:
        st.success(spending["coverage_status"])
    else:
        st.warning(spending["coverage_status"])
    if not import_history.empty and import_history["rejected_rows"].fillna(0).gt(0).any():
        st.warning(
            "One or more imported statements contained rejected rows. Review import "
            "history and verify that those rows do not leave transaction coverage incomplete."
        )
    if spending["low_confidence_count"]:
        st.warning(
            f"{spending['low_confidence_count']} consumption transaction(s) have "
            "uncertain classifications. They are not eligible for reduction proposals."
        )
    if spending["unclassified_debit_count"]:
        st.warning(
            "Unclassified debits are present. Correct or review those transactions "
            "before the planner assesses affordability."
        )

    active_goals = (
        goals.loc[goals["status"].eq("active")].copy()
        if not goals.empty
        else goals
    )
    if active_goals.empty:
        st.info("Create an active goal in Financial goals to build a goal-specific plan.")
    else:
        selected_goal_id = st.selectbox(
            "Select a saved goal",
            active_goals["id"].tolist(),
            format_func=lambda goal_id: active_goals.loc[
                active_goals["id"].eq(goal_id), "title"
            ].iloc[0],
            key="goalpilot_selected_goal",
        )
        goal = active_goals.loc[active_goals["id"].eq(selected_goal_id)].iloc[0]
        goal_record = goal.to_dict()
        all_goal_suggestions: list[dict[str, object]] = []
        try:
            all_goal_suggestions = build_personalized_suggestions(
                planning_frame,
                goals,
                assumed_reduction_rate=reduction / 100,
            )
            preferences = get_goal_plan_preferences(user["id"], selected_goal_id)
            plan = build_goal_savings_plan(
                goal_record,
                profile,
                spending,
                all_goal_suggestions,
                preferences,
            )
            previous_snapshot = get_latest_goal_plan_snapshot(
                user["id"], selected_goal_id
            )
            reconciliation = reconcile_savings_plan(
                previous_snapshot["plan"] if previous_snapshot else None, plan
            )
            snapshot_plan = dict(plan)
            input_snapshot = {
                "goal": {
                    "id": str(goal_record["id"]),
                    "target_amount_minor": int(goal_record["target_amount_minor"]),
                    "current_saved_minor": int(goal_record["current_saved_minor"]),
                    "target_date": str(goal_record["target_date"]),
                },
                "profile": {
                    key: profile.get(key)
                    for key in (
                        "monthly_income_minor",
                        "emergency_balance_minor",
                        "emergency_target_minor",
                        "emergency_contribution_minor",
                        "additional_commitments_minor",
                        "protected_buffer_contribution_minor",
                        "protected_categories",
                        "statement_coverage_confirmed",
                    )
                },
                "spending": {
                    key: spending.get(key)
                    for key in (
                        "estimated_monthly_expenses_minor",
                        "observed_month_count",
                        "months_with_transactions",
                        "observed_period_start",
                        "observed_period_end",
                        "coverage_verified",
                    )
                },
                "preferences": preferences,
            }
            plan_version = save_goal_plan_snapshot(
                user["id"], selected_goal_id, input_snapshot, snapshot_plan
            )
            plan["reconciliation"] = reconciliation
        except (ValueError, RuntimeError, sqlite3.Error) as error:
            st.error(f"Could not build or save this goal plan: {error}")
            plan = None
            plan_version = None

        if plan is not None:
            st.caption(
                f"Plan version {plan_version}. Required contribution counts the "
                "current calendar month through the goal month, inclusive."
            )
            target_minor = plan["target_amount_minor"]
            saved_minor = plan["current_saved_minor"]
            st.write(
                f"Target **{money(target_minor / 100)}** · "
                f"Saved **{money(saved_minor / 100)}** · "
                f"Remaining **{money(plan['remaining_goal_amount_minor'] / 100)}** · "
                f"Deadline **{plan['target_date']}**"
            )
            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            expense_minor = plan["estimated_monthly_expenses_minor"]
            required_minor = plan["required_monthly_contribution_minor"]
            gap_minor = plan["funding_gap_minor"]
            safe_minor = plan["safe_surplus_minor"]
            reserve_shortfall = plan["emergency_reserve_shortfall_minor"]
            metric_one.metric(
                "Estimated monthly expenses",
                money(expense_minor / 100) if expense_minor is not None else "Unavailable",
            )
            metric_two.metric(
                "Safe monthly surplus",
                money(safe_minor / 100) if safe_minor is not None else "Unavailable",
            )
            metric_three.metric(
                "Required goal contribution",
                money(required_minor / 100)
                if required_minor is not None
                else "Deadline passed",
            )
            metric_four.metric(
                "Funding gap before approval",
                money(gap_minor / 100) if gap_minor is not None else "Unavailable",
            )
            with st.expander("Monthly cash-flow assumptions", expanded=True):
                assumptions = [
                    ("Confirmed take-home income", plan["monthly_income_minor"]),
                    ("Estimated transaction expenses", expense_minor),
                    ("Additional commitments", plan["additional_commitments_minor"]),
                    ("Emergency contribution", plan["emergency_contribution_minor"]),
                    (
                        "Protected cash-buffer contribution",
                        plan["protected_buffer_contribution_minor"],
                    ),
                    ("Safe surplus", safe_minor),
                    ("Emergency reserve shortfall", reserve_shortfall),
                ]
                for label, amount in assumptions:
                    value = "Unavailable" if amount is None else money(amount / 100)
                    st.write(f"**{label}:** {value}")
                st.caption(
                    "Transaction expenses include categorized consumption debits. "
                    "Additional commitments must not repeat expenses already included "
                    "in that estimate. Emergency balances are not counted as income."
                )
            st.info(f"Plan status: **{plan['status']}**")
            if plan["assessment_missing_inputs"]:
                st.warning(
                    "Affordability is not assessed until these values are available: "
                    + ", ".join(plan["assessment_missing_inputs"])
                    + "."
                )
            if reserve_shortfall is not None and reserve_shortfall > 0:
                st.warning(
                    f"Your emergency reserve is {money(reserve_shortfall / 100)} "
                    "below its target. The planner preserves your stated emergency "
                    "contribution; it does not divert that reserve to this goal. "
                    "Increasing the monthly emergency contribution would reduce "
                    "the surplus available to the goal by the same monthly amount."
                )
            if required_minor is not None:
                st.caption(
                    f"Required monthly contribution: {money(required_minor / 100)} "
                    f"for {plan['months_remaining']} month(s), counting the current "
                    "month through the goal month inclusively."
                )

            if plan["candidates"]:
                st.markdown("#### Spending adjustments and evidence")
                st.caption(
                    "Candidates are hypothetical. Only approved reductions count in "
                    "the after-approval total; approval does not mean savings occurred."
                )
                display_candidates = []
                for candidate in plan["candidates"]:
                    confidence = candidate["classification_confidence"]
                    display_candidates.append(
                        {
                            "Category": candidate["category"],
                            "Monthly baseline": money(
                                candidate["baseline_monthly_spend_minor"] / 100
                            ),
                            "Proposed reduction": money(
                                candidate["recommended_reduction_minor"] / 100
                            ),
                            "Maximum permitted": money(
                                candidate["max_reduction_minor"] / 100
                            ),
                            "Reduction": f"{candidate['reduction_rate'] * 100:.1f}%",
                            "Disruption weight": candidate["disruption_weight"],
                            "Eligibility": candidate["eligibility_reason"],
                            "Evidence": (
                                f"{candidate['evidence']} "
                                f"Period: {candidate['period_start']} to "
                                f"{candidate['period_end']}."
                            ),
                            "Confidence": (
                                (
                                    "Limited; statement coverage is unverified"
                                    if not spending["coverage_verified"]
                                    else "Classification confidence unavailable"
                                )
                                if confidence is None
                                else f"Category confidence {confidence:.0%}; "
                                + (
                                    "coverage was user-confirmed"
                                    if spending["coverage_verified"]
                                    else "coverage is unverified"
                                )
                            ),
                            "Uncertainty": candidate["uncertainty"],
                            "Related general observations": (
                                ", ".join(candidate["related_suggestions"])
                                or "No matching suggestion"
                            ),
                            "Status": candidate["status"].title(),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(display_candidates),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(plan["optimisation_method"])
                with st.form(
                    f"goalpilot_decisions_{user['id']}_{selected_goal_id}"
                ):
                    selected_statuses: dict[str, str] = {}
                    for candidate in plan["candidates"]:
                        category = candidate["category"]
                        options = ["Not decided", "Approved", "Rejected"]
                        stored = candidate["status"]
                        index = (
                            1 if stored == "approved"
                            else 2 if stored == "rejected"
                            else 0
                        )
                        selected_statuses[category] = st.selectbox(
                            f"{category} proposed adjustment",
                            options,
                            index=index,
                            key=f"goalpilot_status_{selected_goal_id}_{category}",
                        )
                    decisions_submitted = st.form_submit_button(
                        "Save adjustment decisions"
                    )
                if decisions_submitted:
                    try:
                        for category, selected_status in selected_statuses.items():
                            status_value = {
                                "Not decided": "undecided",
                                "Approved": "approved",
                                "Rejected": "rejected",
                            }[selected_status]
                            set_goal_plan_adjustment_status(
                                user["id"],
                                selected_goal_id,
                                category,
                                status_value,
                            )
                        st.success("Adjustment decisions saved for this goal.")
                        st.rerun()
                    except (ValueError, sqlite3.Error) as error:
                        st.error(f"Could not save adjustment decisions: {error}")
                st.write(
                    f"Potential adjustments: **"
                    f"{money(plan['proposed_adjustments_minor'] / 100)}** · "
                    f"Approved adjustments: **"
                    f"{money(plan['approved_adjustments_minor'] / 100)}** · "
                    f"Goal contribution after approval: **"
                    f"{money(plan['planned_goal_contribution_minor'] / 100)}** · "
                    f"Remaining shortfall after approval: **"
                    f"{money(plan['remaining_shortfall_after_approval_minor'] / 100)}**"
                    if plan["remaining_shortfall_after_approval_minor"] is not None
                    else "Potential and approved adjustments are estimates; the goal contribution is unavailable until affordability inputs are complete."
                )
            else:
                st.info(
                    "No eligible, sufficiently supported flexible-category reductions "
                    "are available under the current evidence and protection rules."
                )
                planned = (
                    money(safe_minor / 100) if safe_minor is not None else "Unavailable"
                )
                shortfall = (
                    money(plan["remaining_shortfall_after_approval_minor"] / 100)
                    if plan["remaining_shortfall_after_approval_minor"] is not None
                    else "Unavailable"
                )
                st.write(
                    f"Safe surplus: **{planned}** · Potential adjustments: **"
                    f"{money(0)}** · Approved adjustments: **{money(0)}** · "
                    f"Planned goal contribution: **{planned}** · Remaining shortfall: "
                    f"**{shortfall}**"
                )
            feedback_categories = sorted(
                {candidate["category"] for candidate in plan["candidates"]}
            )
            if feedback_categories:
                st.markdown("#### User-reported outcome")
                st.caption(
                    "An outcome is recorded only from your explicit feedback and note. "
                    "It is not inferred from an Addressed recommendation or from a "
                    "change in transaction totals."
                )
                with st.form(
                    f"goalpilot_feedback_{user['id']}_{selected_goal_id}"
                ):
                    feedback_category = st.selectbox(
                        "Category",
                        feedback_categories,
                        key=f"goalpilot_feedback_category_{selected_goal_id}",
                    )
                    feedback_status = st.selectbox(
                        "Outcome",
                        ["Not applicable", "Achieved (user confirmed)"],
                        key=f"goalpilot_feedback_status_{selected_goal_id}",
                    )
                    reported_reduction_text = st.text_input(
                        "Reported monthly reduction (INR; required if achieved)",
                        key=f"goalpilot_feedback_amount_{selected_goal_id}",
                    )
                    evidence_note = st.text_input(
                        "Supporting note",
                        key=f"goalpilot_feedback_note_{selected_goal_id}",
                    )
                    feedback_submitted = st.form_submit_button(
                        "Save outcome feedback"
                    )
                if feedback_submitted:
                    try:
                        feedback_amount = (
                            _parse_optional_profile_amount(reported_reduction_text)
                            if feedback_status == "Achieved (user confirmed)"
                            else None
                        )
                        record_goal_plan_feedback(
                            user["id"],
                            selected_goal_id,
                            feedback_category,
                            (
                                "achieved_confirmed"
                                if feedback_status == "Achieved (user confirmed)"
                                else "not_applicable"
                            ),
                            evidence_note,
                            feedback_amount,
                        )
                        st.success("User-reported outcome saved to this goal.")
                        st.rerun()
                    except (ValueError, sqlite3.Error) as error:
                        st.error(f"Could not save outcome feedback: {error}")
            feedback_history = list_goal_plan_feedback(
                user["id"], selected_goal_id
            )
            if feedback_history:
                display_feedback = pd.DataFrame(feedback_history)
                display_feedback["status"] = display_feedback["status"].map(
                    {
                        "not_applicable": "Not applicable",
                        "achieved_confirmed": "Achieved (user confirmed)",
                    }
                )
                display_feedback["reported_monthly_reduction_minor"] = (
                    display_feedback["reported_monthly_reduction_minor"].map(
                        lambda value: (
                            "—" if pd.isna(value) else money(int(value) / 100)
                        )
                    )
                )
                st.dataframe(
                    display_feedback.rename(
                        columns={
                            "category": "Category",
                            "status": "Feedback",
                            "reported_monthly_reduction_minor":
                                "Reported monthly reduction",
                            "evidence_note": "Supporting note",
                            "created_at": "Recorded",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            if reconciliation["comparison_available"]:
                st.caption(reconciliation["message"])
                if reconciliation["category_changes"]:
                    st.dataframe(
                        pd.DataFrame(reconciliation["category_changes"]),
                        hide_index=True,
                        width="stretch",
                    )

    st.divider()
    st.subheader("General savings suggestions")
    try:
        suggestions = build_personalized_suggestions(
            frame, goals, assumed_reduction_rate=reduction / 100
        )
        save_recommendations(user["id"], suggestions)
    except (ValueError, RuntimeError) as error:
        st.error(f"Could not calculate suggestions: {error}")
        return
    all_recommendations = list_user_recommendations(user["id"])
    if not suggestions:
        st.info(
            "There is not enough evidence in the selected period for a personalized "
            "suggestion. A saved goal can still produce a contribution-planning estimate."
        )
    if all_recommendations.empty:
        return
    st.caption(
        "Any estimated saving is hypothetical arithmetic, not money already saved or "
        "a recommendation to reduce essential spending."
    )
    st.dataframe(
        all_recommendations[
            [
                "title", "explanation", "period_start", "period_end",
                "estimated_potential_saving_minor", "status",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "estimated_potential_saving_minor": st.column_config.NumberColumn(
                "Hypothetical saving (paise)", format="%d"
            )
        },
    )
    selected_id = st.selectbox(
        "Inspect a suggestion",
        all_recommendations["id"].tolist(),
        format_func=lambda suggestion_id: all_recommendations.loc[
            all_recommendations["id"].eq(suggestion_id), "title"
        ].iloc[0],
    )
    selected = all_recommendations.loc[
        all_recommendations["id"].eq(selected_id)
    ].iloc[0]
    st.write(str(selected["explanation"]))
    st.caption(
        f"Period: {selected['period_start'] or 'unspecified'} to "
        f"{selected['period_end'] or 'unspecified'}."
    )
    try:
        st.json(json.loads(selected["evidence_json"]))
    except (TypeError, json.JSONDecodeError):
        st.error("Saved suggestion evidence could not be read.")
        return
    status = st.selectbox(
        "Your feedback",
        ["Helpful", "Dismissed", "Addressed"],
        key=f"recommendation_status_{selected_id}",
    )
    if st.button("Save suggestion feedback", type="primary"):
        try:
            set_recommendation_status(user["id"], selected_id, status)
            st.success("Feedback saved to this account.")
            st.rerun()
        except ValueError as error:
            st.error(str(error))


def anomaly_page(frame: pd.DataFrame, user: dict[str, str]) -> None:
    page_header(
        "Anomaly review",
        "Exploratory unusual-amount flags for review. An anomaly is not a fraud determination.",
    )
    if frame.empty:
        st.info("Import transactions to explore unusual debit amounts.")
        return
    try:
        anomalies = detect_anomalies(frame)
    except ValueError as error:
        st.info(str(error))
        return
    if anomalies.empty:
        st.info("There are not enough debit transactions for anomaly analysis.")
        return
    stored_status = get_user_anomaly_statuses(user["id"])
    anomalies["review_status"] = anomalies["record_id"].map(stored_status).fillna(
        "Unreviewed"
    )
    flagged = anomalies.loc[anomalies["anomaly"].eq(-1)].copy()
    st.metric("Transactions flagged for review", len(flagged))
    display_columns = [
        col for col in [
            "transaction_date", "merchant", "amount", "final_category",
            "anomaly_score", "robust_amount_outlier",
            "anomaly_explanation", "review_status",
        ] if col in flagged.columns
    ]
    st.dataframe(flagged[display_columns], hide_index=True, width="stretch")
    if flagged.empty or "record_id" not in flagged:
        return
    row_index = st.selectbox(
        "Choose flagged transaction",
        range(len(flagged)),
        format_func=lambda index: (
            f"{flagged.iloc[index]['transaction_date']} · "
            f"{flagged.iloc[index].get('merchant', '')} · "
            f"{money(float(flagged.iloc[index]['amount']))}"
        ),
    )
    chosen = flagged.iloc[row_index]
    status = st.selectbox(
        "Review result",
        ["Reviewed - expected", "Reviewed - investigate", "Unreviewed"],
    )
    if st.button("Save review status", type="primary"):
        try:
            set_user_anomaly_status(
                user["id"],
                int(chosen["record_id"]),
                status,
                anomaly_score=float(chosen["anomaly_score"]),
                explanation=str(chosen["anomaly_explanation"]),
            )
            st.success("Review status saved for this account.")
            st.rerun()
        except (ValueError, RuntimeError) as error:
            st.error(str(error))


# def report_page(frame: pd.DataFrame, user: dict[str, str]) -> None:
#     page_header(
#         "Reports and settings",
#         "Export this account's calculations and inspect local data-handling settings.",
#     )
#     summary = monthly_summary(frame)
#     forecast = forecast_monthly_expenses(frame)
#     recommendations = list_user_recommendations(user["id"])
#     report = {
#         "project": "FinTwin AI — Adaptive Personal Finance Intelligence",
#         "data_source": "Transactions imported by the authenticated account",
#         "statement_coverage": (
#             {
#                 "first_date": str(pd.to_datetime(frame["transaction_date"]).min().date()),
#                 "last_date": str(pd.to_datetime(frame["transaction_date"]).max().date()),
#                 "transactions": int(len(frame)),
#             }
#             if not frame.empty else None
#         ),
#         "monthly_summary": summary.to_dict(orient="records"),
#         "expense_forecast": forecast,
#         "illustrative_recommendations": (
#             recommendations.drop(columns=["evidence_json"], errors="ignore")
#             .to_dict(orient="records")
#             if not recommendations.empty else []
#         ),
#         "limitations": [
#             "PhonePe categories are not verified ground truth.",
#             "The statement does not provide account balances or complete income.",
#             "Forecasts are simple baselines on limited, potentially partial months.",
#             "Category flexibility is a heuristic and needs user confirmation.",
#             "Anomaly flags are exploratory and are not fraud labels.",
#         ],
#     }
#     report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)
#     st.download_button(
#         "Download calculated report (JSON)",
#         report_json,
#         file_name="fintwin_report.json",
#         mime="application/json",
#     )
#     st.subheader("Local data and privacy")
#     st.write(
#         "- Processing and persistence run locally; no external financial API or cloud AI service is used.\n"
#         "- Uploaded transactions, corrections, goals, and review statuses are scoped to this account in SQLite.\n"
#         "- Shared model training data and model artifacts are not changed by uploads or corrections.\n"
#         "- Transaction identifiers and UTRs are omitted from the default transaction export."
#     )
#     include_descriptions = st.checkbox(
#         "Include merchant names and transaction descriptions in the CSV export",
#         value=False,
#         help="These fields may reveal personal financial activity.",
#     )
#     if include_descriptions:
#         export_columns = [
#             col for col in frame.columns
#             if col not in {
#                 "transaction_id", "utr", "text", "payment_method",
#                 "record_id", "import_id",
#             }
#         ]
#     else:
#         export_columns = [
#             col for col in frame.columns
#             if col not in {
#                 "transaction_id", "utr", "text", "transaction_details",
#                 "merchant", "payment_method", "record_id", "import_id",
#             }
#         ]
#     csv_bytes = frame[export_columns].to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "Download transactions (privacy-conscious)",
#         csv_bytes,
#         file_name="fintwin_transactions.csv",
#         mime="text/csv",
#     )
#     st.subheader("Refresh account suggestions")
#     if st.button("Recalculate suggestions for this account", type="primary"):
#         try:
#             goals = list_user_goals(user["id"])
#             suggestions = build_personalized_suggestions(frame, goals)
#             save_recommendations(user["id"], suggestions)
#             st.success(f"Updated {len(suggestions)} account-specific suggestions.")
#             st.rerun()
#         except (ValueError, RuntimeError) as error:
#             st.error(f"Could not refresh suggestions: {error}")


def main() -> None:
    try:
        user = _get_authenticated_user()
    except (OSError, RuntimeError, sqlite3.Error) as error:
        st.error(f"FinTwin could not access its local account database: {error}")
        return
    if user is None:
        render_public_page()
        return

    st.sidebar.markdown("## FinTwin AI")
    st.sidebar.caption(f"Signed in as {user['name']}")
    pages = [
        "Statement Upload & History",
        "Financial overview",
        "Transaction intelligence",
        "Spending analytics",
        # "Financial twin",
        # "Cash-flow forecasting",
        "Scenario lab",
        "Financial goals",
        "Anomaly review",
        # "Reports and settings",
        "Savings Suggestions",
    ]
    selected_page = st.sidebar.selectbox("Workspace", pages)
    if st.sidebar.button("Sign out"):
        _logout()
    st.sidebar.divider()
    st.sidebar.caption("Local processing · Account-scoped data")
    try:
        frame = user_transactions(user)
        filtered = _filter_transactions(frame, user["id"])
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        st.error(f"Could not load this account's transaction data: {error}")
        return
    if selected_page == "Statement Upload & History":
        upload_page(user)
    elif selected_page == "Financial overview":
        overview_page(filtered, user)
    elif selected_page == "Transaction intelligence":
        transaction_page(filtered, user)
    elif selected_page == "Spending analytics":
        spending_page(filtered)
    elif selected_page == "Financial twin":
        twin_page(filtered)
    elif selected_page == "Cash-flow forecasting":
        forecasting_page(filtered)
    elif selected_page == "Scenario lab":
        scenario_page(filtered)
    elif selected_page == "Financial goals":
        goals_page(user)
    elif selected_page == "Anomaly review":
        anomaly_page(filtered, user)
    # elif selected_page == "Reports and settings":
    #     report_page(filtered, user)
    elif selected_page == "Savings Suggestions":
        savings_page(filtered, user)


if __name__ == "__main__":
    main()
