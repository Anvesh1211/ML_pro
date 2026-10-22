from __future__ import annotations

import argparse

import pandas as pd

from src.common import DATA_DIR, atomic_write_csv
from src.financial_intelligence import detect_anomalies


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore unusual debit transactions.")
    parser.add_argument(
        "--input", default=str(DATA_DIR / "final_phonepe_results.csv")
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    results = detect_anomalies(frame)
    output = DATA_DIR / "anomaly_results.csv"
    atomic_write_csv(results, output)
    print(
        f"Flagged {int(results['anomaly'].eq(-1).sum())} of {len(results)} debits "
        "for exploratory review; this is not fraud detection."
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
