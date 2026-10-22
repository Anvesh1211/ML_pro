from __future__ import annotations

import json
import math
from datetime import date

import pandas as pd
import streamlit as st

from src.common import (
    BASE_CATEGORIES,
    DATA_DIR,
    REPORT_DIR,
    atomic_write_csv,
)
from src.financial_intelligence import (
    Scenario,
    add_expense_flexibility,
    build_recommendations,
    calculate_goal,
    calculate_scenario,
    clear_application_state,
    delete_goal,
    detect_recurring_transactions,
    forecast_monthly_expenses,
    get_anomaly_review_statuses,
    load_goals,
    monthly_summary,
    save_goal,
    set_anomaly_review_status,
)
from src.pipeline import run_pipeline
from src.predict_phonepe import (
    FEEDBACK_PATH,
    FINAL_RESULTS_PATH,
    PREDICTIONS_PATH,
    run_prediction,
)
from src.review_transactions import save_correction


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
      --ft-bg: #F5F0FF;
      --ft-surface: #FFFFFF;
      --ft-navy: #594481;
      --ft-blue: #9B8AE6;
      --ft-soft: #EDE5FF;
      --ft-border: #DCD0F5;
      --ft-text: #332B45;
      --ft-muted: #77708A;
      --ft-green: #78BFA5;
      --ft-amber: #E8BF78;
      --ft-red: #E69BA8;
    }

    .stApp {
      background: linear-gradient(135deg, #F5F0FF 0%, #FFF4F7 50%, #EDF7FF 100%);
      color: var(--ft-text);
    }

    [data-testid="stMain"] {
      background: transparent;
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #EDE5FF 0%, #F8EDFF 55%, #EAF5FF 100%);
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
      background: linear-gradient(145deg, #FFFFFF, #F3EDFF);
      border: 1px solid #DCD0F5;
      padding: 18px 16px;
      border-radius: 14px;
      box-shadow: 0 4px 14px rgba(89, 68, 129, 0.07);
    }

    div[data-testid="stMetric"] label {
      color: #77708A !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: #594481 !important;
      font-weight: 700;
    }

    div.stButton > button,
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
      border-radius: 10px;
      border: 1px solid #CFC0F0;
      background: #F0E9FF;
      color: #594481;
      font-weight: 600;
      min-height: 2.5rem;
      transition: all 0.2s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
      background: #DFD3FF;
      border-color: #9B8AE6;
      color: #40305F;
    }

    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
      background: linear-gradient(135deg, #7962B8, #A08BE5);
      color: #FFFFFF;
      border: none;
    }

    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
      background: linear-gradient(135deg, #6850A7, #8C76D4);
      color: #FFFFFF;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
      border-radius: 9px;
      border-color: #D6C9EF;
      background: #FFFFFF;
    }

    [data-testid="stDataFrame"] {
      border: 1px solid #DCD0F5;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 3px 12px rgba(89, 68, 129, 0.05);
    }

    [data-testid="stAlert"] {
      border-radius: 12px;
      border: 1px solid #E5D9F7;
    }

    [data-testid="stHeader"] {
      background: transparent;
    }

    #MainMenu, footer {
      visibility: hidden;
    }

    .ft-kicker {
      color: #8B78B8;
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

@st.cache_data(show_spinner=False)
def load_results(path: str, modified: float) -> pd.DataFrame:
    del modified
    frame = pd.read_csv(
        path,
        dtype={"transaction_id": "string", "utr": "string"},
        low_memory=False,
    )
    required = {
        "transaction_date",
        "transaction_type",
        "amount",
        "merchant",
        "final_category",
        "category_source",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The final transaction file is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return frame


def current_transactions() -> pd.DataFrame:
    if not FINAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "No verified prediction output exists. Run `python -m src.pipeline` first."
        )
    return load_results(
        str(FINAL_RESULTS_PATH),
        FINAL_RESULTS_PATH.stat().st_mtime,
    )


def money(value: float | int) -> str:
    return f"₹{value:,.2f}"


def month_picker(frame: pd.DataFrame, key: str) -> str:
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


def overview_page(frame: pd.DataFrame) -> None:
    page_header(
        "Financial overview",
        "Observed transaction activity from your PhonePe statement. Missing inputs are not treated as zero.",
    )
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
            st.bar_chart(category_totals, horizontal=True, color="#386FA4")
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


def transaction_page(frame: pd.DataFrame) -> None:
    page_header(
        "Transaction intelligence",
        "Search and review classifications. Corrections are saved locally and applied on the next classification pass.",
    )
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
        submitted = st.form_submit_button("Save correction and refresh predictions", type="primary")
    if submitted:
        if correction_scope.startswith("All transactions") and not confirm_merchant:
            st.error("Confirm the merchant-wide correction or select transaction-only.")
        else:
            try:
                save_correction(
                    str(row.get("merchant", "")),
                    new_category,
                    transaction_id=(
                        None
                        if correction_scope.startswith("All transactions")
                        else str(row["transaction_id"])
                    ),
                )
                updated = run_prediction()
                from src.pipeline import refresh_derived_outputs

                refresh_derived_outputs(updated)
                st.cache_data.clear()
                st.success("Correction saved; predictions and dependent analytics refreshed.")
                st.rerun()
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Could not save and apply this correction: {error}")
    st.caption(
        f"{len(needs_review)} transactions currently need review. "
        f"Saved feedback file: {FEEDBACK_PATH.name}."
    )


def spending_page(frame: pd.DataFrame) -> None:
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
        st.line_chart(chart, color=["#386FA4", "#A16207", "#18794E"])
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
    cluster_path = REPORT_DIR / "spending_clusters.json"
    if cluster_path.exists():
        st.subheader("Exploratory spending clusters")
        st.json(json.loads(cluster_path.read_text(encoding="utf-8")))
        st.caption("K-Means cluster IDs are descriptive groups, not financial-health ratings.")


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


def goals_page() -> None:
    page_header(
        "Financial goals",
        "Store goals locally and estimate required contributions. Feasibility requires an explicit contribution assumption.",
    )
    with st.form("goal_form"):
        name = st.text_input("Goal name")
        amount = st.number_input("Target amount (INR)", min_value=0.0, value=0.0, step=1000.0)
        saved = st.number_input("Already saved (INR; user supplied)", min_value=0.0, value=0.0, step=1000.0)
        target = st.date_input("Target date", value=date.today())
        submitted = st.form_submit_button("Save or update goal", type="primary")
    if submitted:
        try:
            if not name.strip() or amount <= 0 or target <= date.today():
                raise ValueError("Enter a goal name, positive target amount, and a future target date.")
            save_goal(
                {
                    "goal_name": name.strip(),
                    "goal_amount": amount,
                    "current_saved": saved,
                    "target_date": target.isoformat(),
                }
            )
            st.success("Goal saved locally.")
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))
    goals = load_goals()
    st.subheader("Saved goals")
    if goals.empty:
        st.info("No goals have been saved yet.")
        return
    st.dataframe(goals, hide_index=True, width="stretch")
    options = goals["goal_name"].tolist()
    selected_goal = st.selectbox("Evaluate goal", options)
    goal = goals.loc[goals["goal_name"].eq(selected_goal)].iloc[0]
    target_date = pd.Timestamp(goal["target_date"]).date()
    if target_date <= date.today():
        st.warning("This goal's target date has passed; choose a new future target date to reassess it.")
        return
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
            float(goal["goal_amount"]),
            float(goal["current_saved"]),
            months,
            contribution,
        )
    except ValueError as error:
        st.error(str(error))
        return
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
        float(goal["goal_amount"]),
        float(goal["current_saved"]),
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
        delete_goal(selected_goal)
        st.rerun()


def anomaly_page(frame: pd.DataFrame) -> None:
    page_header(
        "Anomaly review",
        "Exploratory unusual-amount flags for review. An anomaly is not a fraud determination.",
    )
    result_path = DATA_DIR / "anomaly_results.csv"
    if not result_path.exists():
        st.info("Run the pipeline to create anomaly results.")
        return
    anomalies = pd.read_csv(result_path, dtype={"transaction_id": "string"}, low_memory=False)
    if anomalies.empty:
        st.info("There are not enough debit transactions for anomaly analysis.")
        return
    stored_status = get_anomaly_review_statuses()
    if "transaction_id" in anomalies:
        anomalies["review_status"] = anomalies["transaction_id"].map(stored_status).fillna(
            anomalies.get("review_status", "Unreviewed")
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
    if flagged.empty or "transaction_id" not in flagged:
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
        set_anomaly_review_status(str(chosen["transaction_id"]), status)
        st.success("Review status saved locally.")
        st.rerun()


def report_page(frame: pd.DataFrame) -> None:
    page_header(
        "Reports and settings",
        "Export current calculations, inspect data provenance, and refresh from the existing local statement.",
    )
    summary = monthly_summary(frame)
    forecast = forecast_monthly_expenses(frame)
    recommendations = build_recommendations(frame)
    report = {
        "project": "FinTwin AI — Adaptive Personal Finance Intelligence",
        "data_source": "Existing local data/testing.csv and data/training.csv",
        "statement_coverage": (
            {
                "first_date": str(pd.to_datetime(frame["transaction_date"]).min().date()),
                "last_date": str(pd.to_datetime(frame["transaction_date"]).max().date()),
                "transactions": int(len(frame)),
            }
            if not frame.empty else None
        ),
        "monthly_summary": summary.to_dict(orient="records"),
        "expense_forecast": forecast,
        "illustrative_recommendations": recommendations.to_dict(orient="records"),
        "limitations": [
            "PhonePe categories are not verified ground truth.",
            "The statement does not provide account balances or complete income.",
            "Forecasts are simple baselines on limited, potentially partial months.",
            "Category flexibility is a heuristic and needs user confirmation.",
            "Anomaly flags are exploratory and are not fraud labels.",
        ],
    }
    report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    st.download_button(
        "Download calculated report (JSON)",
        report_json,
        file_name="fintwin_report.json",
        mime="application/json",
    )
    st.subheader("Local data and privacy")
    st.write(
        "- Processing and persistence run locally; no external financial API or cloud AI service is used.\n"
        "- Original statement and training files are read-only inputs to the pipeline.\n"
        "- Derived predictions, audit, analytics, user corrections, and goals are stored under `data/`.\n"
        "- Transaction IDs and UTRs are omitted from the default transaction export."
    )
    include_descriptions = st.checkbox(
        "Include merchant names and transaction descriptions in the CSV export",
        value=False,
        help="These fields may reveal personal financial activity.",
    )
    if include_descriptions:
        export_columns = [
            col for col in frame.columns
            if col not in {"transaction_id", "utr", "text", "payment_method"}
        ]
    else:
        export_columns = [
            col for col in frame.columns
            if col not in {
                "transaction_id", "utr", "text", "transaction_details",
                "merchant", "payment_method",
            }
        ]
    csv_bytes = frame[export_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download transactions (privacy-conscious)",
        csv_bytes,
        file_name="fintwin_transactions.csv",
        mime="text/csv",
    )
    audit_path = DATA_DIR / "preprocessing_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        st.write(f"Statement footer/invalid rows explicitly excluded: {len(audit)}")
        if not audit.empty:
            st.dataframe(audit, hide_index=True, width="stretch")
    st.subheader("Refresh existing data")
    if st.button("Re-run verified local pipeline", type="primary"):
        try:
            with st.spinner("Processing local data and validating outputs..."):
                result = run_pipeline()
            st.cache_data.clear()
            st.success(
                f"Processed {result['processed_rows']} statements rows using the existing training dataset."
            )
            st.rerun()
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            st.error(f"Pipeline failed; previous derived files were not intentionally removed: {error}")
    st.subheader("Delete locally stored application state")
    st.caption("This clears saved goals and anomaly-review statuses only; it does not delete transaction files or user feedback.")
    confirm_clear = st.checkbox("I understand and want to clear goals and review statuses.")
    if st.button("Clear local goals and review statuses"):
        try:
            clear_application_state(confirmed=confirm_clear)
            st.success("Local application state cleared.")
        except ValueError as error:
            st.error(str(error))


def main() -> None:
    st.sidebar.markdown("## FinTwin AI")
    st.sidebar.caption("Personal finance intelligence")
    pages = [
        "Financial overview",
        "Transaction intelligence",
        "Spending analytics",
        "Financial twin",
        "Cash-flow forecasting",
        "Scenario lab",
        "Financial goals",
        "Anomaly review",
        "Reports and settings",
    ]
    selected_page = st.sidebar.selectbox("Workspace", pages)
    st.sidebar.divider()
    st.sidebar.caption("Local processing · Existing datasets only")
    try:
        frame = current_transactions()
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.info("From the project folder, run `python -m src.pipeline`, then refresh this page.")
        return
    {
        "Financial overview": overview_page,
        "Transaction intelligence": transaction_page,
        "Spending analytics": spending_page,
        "Financial twin": twin_page,
        "Cash-flow forecasting": forecasting_page,
        "Scenario lab": scenario_page,
        "Financial goals": lambda _: goals_page(),
        "Anomaly review": anomaly_page,
        "Reports and settings": report_page,
    }[selected_page](frame)


if __name__ == "__main__":
    main()
