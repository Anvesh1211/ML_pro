from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocess_phonepe import CATEGORY_SUBCATEGORY, classify_description


CORE_COLUMNS = {"date", "transaction details", "transaction type", "amount"}
COLUMN_NAMES = {
    "date": "Date",
    "time": "Time",
    "transaction details": "Transaction Details",
    "transaction id": "Transaction ID",
    "utr": "UTR",
    "transaction type": "Transaction Type",
    "credit/debit instrument": "Credit/debit instrument",
    "amount": "Amount",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_MERCHANT_PREFIX = re.compile(r"^(Paid to|Received from)\s+", re.IGNORECASE)


class PhonePeUploadError(ValueError):
    pass


@dataclass(frozen=True)
class RejectedRow:
    source_line: int
    reason: str


@dataclass
class ParsedStatement:
    filename: str
    file_sha256: str
    source_columns: list[str]
    processed: pd.DataFrame
    source_rows: list[dict[str, str]]
    rejected: list[RejectedRow]
    total_rows: int
    duplicate_candidates: int

    @property
    def valid_rows(self) -> int:
        return len(self.processed)

    @property
    def date_start(self) -> str | None:
        return (
            str(self.processed["transaction_date"].min())
            if not self.processed.empty
            else None
        )

    @property
    def date_end(self) -> str | None:
        return (
            str(self.processed["transaction_date"].max())
            if not self.processed.empty
            else None
        )


def _normalize_header(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _decode_csv(content: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise PhonePeUploadError("The CSV encoding is not supported.")


def _find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]]:
    for row_index, row in enumerate(rows):
        normalized = [_normalize_header(value) for value in row]
        if not CORE_COLUMNS.issubset(set(normalized)):
            continue
        positions: dict[str, int] = {}
        duplicate_names: set[str] = set()
        for index, column in enumerate(normalized):
            if column in COLUMN_NAMES:
                if column in positions:
                    duplicate_names.add(column)
                positions[column] = index
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise PhonePeUploadError(f"CSV contains repeated column names: {names}.")
        return row_index, positions
    raise PhonePeUploadError(
        "Could not find a PhonePe header. Required columns are Date, "
        "Transaction Details, Transaction Type, and Amount."
    )


def _parse_amount(value: str) -> int:
    cleaned = value.strip().replace(",", "").replace("₹", "").replace("INR", "")
    try:
        amount = Decimal(cleaned.strip())
    except InvalidOperation as error:
        raise ValueError("invalid amount") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("amount must be a finite non-negative value")
    try:
        quantized = amount.quantize(Decimal("0.01"))
    except InvalidOperation as error:
        raise ValueError("amount exceeds the supported precision") from error
    if quantized != amount:
        raise ValueError("amount has more than two decimal places")
    amount_minor = int(quantized * 100)
    if amount_minor > 2**63 - 1:
        raise ValueError("amount exceeds the supported range")
    return amount_minor


def _candidate_key(
    transaction_date: str,
    transaction_type: str,
    amount_minor: int,
    merchant: str,
    description: str,
) -> tuple[str, str, int, str, str]:
    return (
        transaction_date,
        transaction_type,
        amount_minor,
        " ".join(merchant.casefold().split()),
        " ".join(description.casefold().split()),
    )


def parse_phonepe_csv(
    content: bytes,
    filename: str,
    *,
    max_bytes: int | None = None,
) -> ParsedStatement:
    limit = max_bytes if max_bytes is not None else MAX_UPLOAD_BYTES
    if not filename.casefold().endswith(".csv"):
        raise PhonePeUploadError("Upload a CSV file with a .csv extension.")
    if not content:
        raise PhonePeUploadError("The uploaded CSV is empty.")
    if len(content) > limit:
        raise PhonePeUploadError(
            f"File exceeds the configured {limit // (1024 * 1024)} MB upload limit."
        )

    text, _encoding = _decode_csv(content)
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except csv.Error as error:
        raise PhonePeUploadError("The CSV has malformed quoting or row structure.") from error
    if not rows:
        raise PhonePeUploadError("The uploaded CSV has no rows.")
    header_row_index, positions = _find_header(rows)
    headers = rows[header_row_index]
    header_line = header_row_index + 1

    valid_records: list[dict[str, Any]] = []
    source_records: list[dict[str, str]] = []
    rejected: list[RejectedRow] = []
    total_rows = 0
    seen_transaction_ids: set[str] = set()
    seen_candidate_keys: set[tuple[str, str, int, str, str]] = set()
    candidate_count = 0

    for row_index, values in enumerate(rows[header_row_index + 1 :], start=header_line + 1):
        if not values or all(not value.strip() for value in values):
            continue
        total_rows += 1
        if len(values) != len(headers):
            rejected.append(RejectedRow(row_index, "row has a different number of columns"))
            continue
        raw = {header: value.strip() for header, value in zip(headers, values, strict=True)}
        source = {
            COLUMN_NAMES[column]: values[index].strip()
            for column, index in positions.items()
        }
        details = source.get("Transaction Details", "").strip()
        transaction_type = source.get("Transaction Type", "").strip().upper()
        transaction_id = source.get("Transaction ID", "").strip()
        if not details:
            rejected.append(RejectedRow(row_index, "missing transaction details"))
            continue
        if transaction_type not in {"DEBIT", "CREDIT"}:
            rejected.append(RejectedRow(row_index, "transaction type must be DEBIT or CREDIT"))
            continue
        transaction_date = pd.to_datetime(
            source.get("Date", ""), errors="coerce", format="mixed"
        )
        if pd.isna(transaction_date):
            rejected.append(RejectedRow(row_index, "invalid transaction date"))
            continue
        try:
            amount_minor = _parse_amount(source.get("Amount", ""))
        except ValueError as error:
            rejected.append(RejectedRow(row_index, str(error)))
            continue

        merchant = _MERCHANT_PREFIX.sub("", details).strip()
        date_value = transaction_date.strftime("%Y-%m-%d")
        candidate_key = _candidate_key(
            date_value, transaction_type, amount_minor, merchant, details
        )
        duplicate_type = ""
        if transaction_id:
            if transaction_id in seen_transaction_ids:
                duplicate_type = "exact"
            seen_transaction_ids.add(transaction_id)
        elif candidate_key in seen_candidate_keys:
            duplicate_type = "possible"
        seen_candidate_keys.add(candidate_key)
        if duplicate_type:
            candidate_count += 1
        category = classify_description(details, transaction_type)
        valid_records.append(
            {
                "transaction_id": transaction_id,
                "transaction_date": date_value,
                "transaction_time": source.get("Time", ""),
                "transaction_details": details,
                "utr": source.get("UTR", ""),
                "transaction_type": transaction_type,
                "payment_method": source.get("Credit/debit instrument", ""),
                "amount": amount_minor / 100,
                "amount_minor": amount_minor,
                "currency": "INR",
                "merchant": merchant,
                "subcategory": CATEGORY_SUBCATEGORY.get(category, "Unknown"),
                "category": category,
                "cashflow_type": transaction_type,
                "source_line": row_index,
                "duplicate_type": duplicate_type,
            }
        )
        source_records.append(raw)

    if total_rows == 0:
        raise PhonePeUploadError("The CSV contains a header but no transaction records.")
    processed = pd.DataFrame(valid_records)
    if processed.empty:
        raise PhonePeUploadError(
            f"No valid transactions were found; {len(rejected)} row(s) need correction."
        )
    return ParsedStatement(
        filename=Path(filename).name,
        file_sha256=hashlib.sha256(content).hexdigest(),
        source_columns=[header.strip() for header in headers],
        processed=processed,
        source_rows=source_records,
        rejected=rejected,
        total_rows=total_rows,
        duplicate_candidates=candidate_count,
    )


def source_json(row: dict[str, str]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))