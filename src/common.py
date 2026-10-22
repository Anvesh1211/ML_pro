from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "category_model.joblib"
LEGACY_MODEL_PATH = PROJECT_ROOT / "src" / "category_model.pkl"

REQUIRED_TRAINING_COLUMNS = {
    "transaction_details",
    "merchant",
    "subcategory",
    "category",
}
REQUIRED_PROCESSED_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "transaction_details",
    "transaction_type",
    "amount",
    "merchant",
    "subcategory",
}
BASE_CATEGORIES = (
    "Bills & Utilities",
    "Education",
    "Entertainment",
    "Food & Dining",
    "Grocery",
    "Health",
    "Mobile Recharge",
    "Personal Care",
    "Personal Transfer",
    "Rent",
    "Salary/Income",
    "Shopping",
    "Transport",
)


def require_columns(
    frame: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{path.stem}-",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            frame.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{path.stem}-",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def normalize_merchant(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def build_model_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["transaction_details"].fillna("").astype(str)
        + " "
        + frame["merchant"].fillna("").astype(str)
        + " "
        + frame["subcategory"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()
