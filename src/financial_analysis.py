from __future__ import annotations

import argparse

import pandas as pd

from src.common import DATA_DIR
from src.financial_intelligence import monthly_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show month-wise observed transaction cash flows."
    )
    parser.add_argument(
        "--input", default=str(DATA_DIR / "final_phonepe_results.csv")
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    summary = monthly_summary(frame)
    if summary.empty:
        print("No transaction months are available.")
        return
    print("Observed month summaries (partial statement months are not extrapolated):")
    print(summary.to_string(index=False))
    if summary["salary_income_category_credits"].sum() == 0:
        print(
            "\nNo credits were categorized as salary/income. "
            "Income and savings feasibility cannot be inferred from these transactions."
        )


if __name__ == "__main__":
    main()
