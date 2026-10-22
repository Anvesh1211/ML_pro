# FinTwin AI — Reproducible Experiments

All runs below were generated from the existing `data/training.csv`,
`data/testing.csv`, and predictions derived from that statement. No external
dataset or synthetic transaction record was added. Exact machine-readable
outputs are in `reports/classification_metrics.json`,
`reports/spending_clusters.json`, and `reports/experiments.json`.

## Experiment A — Merchant-disjoint category classification

- Source: the existing 900-row `data/training.csv`, 13 labels.
- Method: word and character TF-IDF + class-balanced Logistic Regression.
- Split: fixed `GroupShuffleSplit`, seed 42, 723 training rows/86 merchants and
  177 test rows/22 merchants; 0 merchant overlap.
- Accuracy: **0.9435**.
- Macro-F1: **0.9197**.
- Weighted-F1: **0.9417**.
- Mean inference time: **0.226 ms/transaction** on the current local environment.
- The held-out set has no `Personal Care` or `Shopping` examples. Those
  classes were not empirically evaluated in this split. Per-class metrics and
  the confusion matrix are saved in the JSON report.
- Limitation: this evaluates labels in the supplied training file, whose
  collection/annotation process and ground-truth status are undocumented. It
  does not measure accuracy on the PhonePe statement.

## Experiment B — Offline merchant-memory simulation

- Source: labeled examples from the same existing training file.
- Protocol: deterministic seed 42; hold out 20% of repeated merchants; use the
  earliest labeled transaction for each held-out merchant as the support
  correction; evaluate only later rows from those merchants. Held-out
  merchants are excluded from model fitting.
- Model-fitting rows: 730. Support corrections: 22. Later evaluation rows:
  148 across 22 merchants.
- Baseline: accuracy **0.9797**, macro-F1 **0.9416**, weighted-F1 **0.9813**.
- Merchant memory: accuracy **1.0000**, macro-F1 **1.0000**, weighted-F1
  **1.0000**; macro-F1 difference **+0.0584** on this selected subset.
- `Salary/Income` is absent from the evaluation set. No held-out merchant in
  this sample has mixed category labels, which makes the merchant rule unusually
  favorable and limits generalization.
- This is an offline simulation, not an experiment on the eight real saved
  user corrections and not independently verified PhonePe accuracy. Do not
  claim a user-level accuracy improvement from it.

## Experiment C — Expense-flexibility comparison

- Period: four observed calendar months; boundary months are partial.
- Actual eligible data: 24 debit transactions in the potentially adjustable
  category proxy.
- Identical hypothetical assumption: reduce eligible outlay by **10%**.
- Without the flexibility filter, 10% of observed average monthly
  classified consumption outlay is **₹163.77**.
- With the flexibility proxy, the estimated amount is **₹86.34/month**, a
  **₹77.43/month lower** estimated reduction base than the unfiltered
  calculation.
- These values are arithmetic estimates, not observed savings or a
  recommendation that the user should reduce those expenses. Category-level
  flexibility has not been user-verified.

## Experiment D — Expense forecast baseline

- Target: next-month recorded consumption debits (transfers excluded).
- Observed coverage: May–August 2026 (four months, with partial May/August).
- Baselines: historical mean **₹1,637.71** and three-month moving average
  **₹1,211.42** for September 2026.
- Chronological one-step evaluation: only **2 origins** are available.
  Both approaches produced MAE **₹1,150.20** on these origins.
- No seasonal forecast or income forecast is justified. The shared MAE and
  tiny evaluation set are not evidence of forecast reliability. As of the
  current October 2026 runtime, September 2026 is already past; this is a
  historical next-observation estimate, not a current financial forecast.

## Experiment E — Flexible clustering versus amount-only baseline

- Input: 92 classified consumption debit transactions; transfers and unknown
  categories are excluded.
- Amount-only baseline: K-Means on standardized amount, silhouette **0.7631**.
- Richer feature comparison: K-Means on log amount, category, and weekday,
  silhouette **0.2129**.
- The richer feature set did **not** improve silhouette on this dataset. The
  cluster output is exploratory; cluster labels are not health scores and
  should not be presented as a validated behavioral improvement.

## Experiment F — Anomaly exploration

- Input: 278 debit rows.
- Isolation Forest at a fixed 5% contamination flagged **8** transactions.
- A robust 1.5×IQR baseline on log amount flagged **4** rows.
- There are no verified fraud/anomaly outcome labels, so precision, recall,
  and fraud accuracy cannot be calculated. These flags are review leads only.

## Experiment G — Hand-checked scenario arithmetic

The unit test passes explicit hypothetical inputs (not transaction data):
monthly income ₹50,000; essential outlay ₹20,000; flexible outlay ₹10,000;
other outlay ₹5,000; income reduction 20%; essential-cost increase 10%;
flexible-spending reduction 20%; new EMI ₹2,000; one-time emergency
expense ₹6,000.

- Baseline monthly cash flow: **₹15,000**.
- Scenario monthly cash flow: **₹3,000**.
- Change: **−₹12,000/month**.
- First month after emergency expense: **−₹3,000**.

This verifies arithmetic only; it is not a forecast or an observed user
outcome.
