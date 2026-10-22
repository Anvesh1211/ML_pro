from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from src.common import DATA_DIR, atomic_write_csv


RAW_PATH = DATA_DIR / "testing.csv"
OUTPUT_PATH = DATA_DIR / "phonepe_processed.csv"
AUDIT_PATH = DATA_DIR / "preprocessing_audit.csv"
RAW_COLUMNS = {
    "Date",
    "Time",
    "Transaction Details",
    "Transaction ID",
    "UTR",
    "Transaction Type",
    "Credit/debit instrument",
    "Amount",
}

CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Salary/Income", ("salary", "stipend", "payroll")),
    ("Personal Transfer", ("received from", "family transfer", "friend transfer")),
    (
        "Food & Dining",
        ("swiggy", "zomato", "restaurant", "dining", "food", "biryani", "cafe"),
    ),
    (
        "Grocery",
        ("super market", "supermarket", "grocery", "kirana", "fruit", "vegetable"),
    ),
    ("Mobile Recharge", ("mobile recharge", "recharge", "prepaid", "postpaid")),
    ("Shopping", ("meesho", "amazon", "flipkart", "myntra", "shopping")),
    (
        "Transport",
        ("uber", "ola", "rapido", "irctc", "railway", "metro", "petrol", "fuel"),
    ),
    (
        "Bills & Utilities",
        ("electricity", "water bill", "gas bill", "broadband", "internet bill"),
    ),
    ("Health", ("hospital", "medical", "medicine", "pharmacy", "clinic")),
    ("Personal Care", ("salon", "parlour", "barber", "haircut")),
    ("Education", ("college fees", "school fees", "tuition", "education")),
    ("Entertainment", ("cinema", "movie", "netflix", "spotify", "bookmyshow")),
    ("Rent", ("house rent", "room rent", "rent payment")),
)

CATEGORY_SUBCATEGORY = {
    "Food & Dining": "Restaurant",
    "Grocery": "Groceries",
    "Mobile Recharge": "Mobile Recharge",
    "Shopping": "Shopping",
    "Transport": "Transport",
    "Salary/Income": "Income",
    "Personal Transfer": "Transfer",
    "Bills & Utilities": "Utilities",
    "Health": "Healthcare",
    "Personal Care": "Personal Care",
    "Education": "Education",
    "Entertainment": "Entertainment",
    "Rent": "Rent",
}
ESSENTIAL_CATEGORIES = {
    "Grocery",
    "Bills & Utilities",
    "Health",
    "Transport",
    "Rent",
    "Education",
    "Mobile Recharge",
}
DISCRETIONARY_CATEGORIES = {
    "Food & Dining",
    "Shopping",
    "Personal Care",
    "Entertainment",
}


def classify_description(
    description: object, transaction_type: object | None = None
) -> str:
    normalized = "" if pd.isna(description) else str(description).casefold()
    kind = "" if pd.isna(transaction_type) else str(transaction_type).upper()
    for category, phrases in CATEGORY_RULES:
        if any(
            re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
            for phrase in phrases
        ):
            if category == "Salary/Income" and kind == "DEBIT":
                continue
            return category
    return "Needs Review"


def _statement_record_start_lines(
    path: Path, expected_columns: set[str], expected_rows: int
) -> list[int]:
    starts = []
    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.reader(source_file)
        for _ in range(2):
            next(reader, None)
        previous_line = reader.line_num

        while True:
            start_line = previous_line + 1
            row = next(reader, None)
            if row is None:
                raise ValueError(
                    f"Could not find the PhonePe CSV header in {path}."
                )
            previous_line = reader.line_num
            if not row:
                continue
            if set(column.strip() for column in row) == expected_columns:
                break

        for row in reader:
            start_line = previous_line + 1
            previous_line = reader.line_num
            if row:
                starts.append(start_line)

    if len(starts) != expected_rows:
        raise ValueError(
            f"Could not map all parsed statement rows to source lines in {path}: "
            f"CSV parser found {len(starts)} data records but pandas parsed "
            f"{expected_rows}."
        )
    return starts


def load_phonepe_statement(path=RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"PhonePe statement not found: {path}")
    raw = pd.read_csv(
        path,
        skiprows=2,
        dtype={"Transaction ID": "string", "UTR": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    missing = sorted(RAW_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(
            f"PhonePe statement is missing required columns: {', '.join(missing)}"
        )

    source_lines = pd.Series(
        _statement_record_start_lines(path, set(raw.columns), len(raw)),
        index=raw.index,
        dtype="int64",
    )
    dates = pd.to_datetime(raw["Date"], errors="coerce", format="mixed")
    amounts = pd.to_numeric(raw["Amount"], errors="coerce")
    transaction_types = raw["Transaction Type"].str.strip().str.upper()
    has_identifier = raw["Transaction ID"].astype(str).str.strip().ne("")
    has_description = raw["Transaction Details"].astype(str).str.strip().ne("")
    valid_types = transaction_types.isin({"DEBIT", "CREDIT"})
    valid = (
        dates.notna()
        & amounts.notna()
        & valid_types
        & has_identifier
        & has_description
    )

    audit_rows = []
    for row_index in raw.index[~valid]:
        reasons = []
        if pd.isna(dates.loc[row_index]):
            reasons.append("invalid_date_or_statement_footer")
        if pd.isna(amounts.loc[row_index]):
            reasons.append("invalid_amount_or_statement_footer")
        if not valid_types.loc[row_index]:
            reasons.append("invalid_transaction_type_or_statement_footer")
        if not has_identifier.loc[row_index]:
            reasons.append("missing_transaction_id")
        if not has_description.loc[row_index]:
            reasons.append("missing_transaction_details")
        audit_rows.append(
            {
                "source_file": path.name,
                "source_line": int(source_lines.loc[row_index]),
                "disposition": "excluded",
                "reason": ";".join(reasons),
            }
        )

    clean = raw.loc[valid].copy()
    clean_dates = dates.loc[valid]
    clean_amounts = amounts.loc[valid]
    clean_types = transaction_types.loc[valid]
    details = clean["Transaction Details"].astype(str).str.strip()
    merchants = details.str.replace(
        r"^(Paid to|Received from)\s+", "", regex=True, flags=re.IGNORECASE
    ).str.strip()
    categories = pd.Series(
        [
            classify_description(description, tx_type)
            for description, tx_type in zip(details, clean_types, strict=True)
        ],
        index=details.index,
        dtype="string",
    )
    expense_nature = []
    flexibility = []
    for tx_type, category in zip(clean_types, categories, strict=True):
        if tx_type == "CREDIT":
            nature, level = (
                ("Income", "Not Applicable")
                if category == "Salary/Income"
                else ("Transfer", "Not Applicable")
            )
        elif category == "Personal Transfer":
            nature, level = "Transfer", "Not Applicable"
        elif category in ESSENTIAL_CATEGORIES:
            nature, level = "Essential", "Low"
        elif category in DISCRETIONARY_CATEGORIES:
            nature, level = "Discretionary", "High"
        else:
            nature, level = "Unknown", "Unknown"
        expense_nature.append(nature)
        flexibility.append(level)

    processed = pd.DataFrame(
        {
            "transaction_id": clean["Transaction ID"].astype("string").str.strip(),
            "transaction_date": clean_dates.dt.strftime("%Y-%m-%d"),
            "transaction_time": clean["Time"].astype(str).str.strip(),
            "transaction_details": details,
            "utr": clean["UTR"].astype("string").str.strip(),
            "transaction_type": clean_types,
            "payment_method": clean["Credit/debit instrument"].astype(str).str.strip(),
            "amount": clean_amounts.astype(float),
            "currency": "INR",
            "merchant": merchants,
            "subcategory": categories.map(CATEGORY_SUBCATEGORY).fillna("Unknown"),
            "category": categories,
            "cashflow_type": clean_types,
            "expense_nature": expense_nature,
            "flexibility_level": flexibility,
            "flexibility_score": [
                0.3 if level == "Low" else 0.8 if level == "High" else pd.NA
                for level in flexibility
            ],
            "transaction_status": "Unknown",
            "notes": "",
            "city": pd.NA,
            "is_expense": (clean_types == "DEBIT").astype(int),
            "amount_band": pd.cut(
                clean_amounts,
                bins=[-1, 100, 500, 1000, 5000, float("inf")],
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
            ).astype("string"),
            "day_of_week": clean_dates.dt.day_name(),
            "month": clean_dates.dt.month,
            "year": clean_dates.dt.year,
            "monthly_budget": pd.NA,
            "remaining_budget": pd.NA,
            "savings_goal": pd.NA,
            "source_line": source_lines.loc[clean.index].to_numpy(),
        },
        index=clean.index,
    ).reset_index(drop=True)
    return processed, pd.DataFrame(
        audit_rows, columns=["source_file", "source_line", "disposition", "reason"]
    )


def preprocess_phonepe(
    raw_path=RAW_PATH,
    output_path=OUTPUT_PATH,
    audit_path=AUDIT_PATH,
) -> pd.DataFrame:
    processed, audit = load_phonepe_statement(raw_path)
    atomic_write_csv(processed, output_path)
    atomic_write_csv(audit, audit_path)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the existing PhonePe statement.")
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    result = preprocess_phonepe(raw_path=args.input)
    print(f"Processed {len(result)} transactions -> {OUTPUT_PATH}")
    print(f"Excluded rows and reasons -> {AUDIT_PATH}")


if __name__ == "__main__":
    main()
