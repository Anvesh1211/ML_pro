from __future__ import annotations

import argparse

from src.common import DATA_DIR, atomic_write_csv
from src.financial_intelligence import cluster_spending


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster observed transaction spending.")
    parser.add_argument(
        "--input", default=str(DATA_DIR / "final_phonepe_results.csv")
    )
    args = parser.parse_args()
    import pandas as pd

    frame = pd.read_csv(args.input, low_memory=False)
    clusters, metrics = cluster_spending(frame)
    output = DATA_DIR / "spending_behavior.csv"
    atomic_write_csv(clusters, output)
    print(f"Saved {len(clusters)} observed debit transactions to {output}")
    print(f"Silhouette score: {metrics['silhouette_score']}")


if __name__ == "__main__":
    main()
