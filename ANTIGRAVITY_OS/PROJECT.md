# FinTwin AI — Verified Project State

## Identity and scope

FinTwin AI is a local-first Python/Streamlit personal-finance prototype. It
uses the existing labeled training file and PhonePe statement; it does not
download or generate financial data. It is decision support, not a bank,
official credit score, confirmed fraud detector, or personalized financial
adviser.

## Current architecture

- `src/preprocess_phonepe.py`: validates the PhonePe CSV, preserves transaction
  identifiers as strings, normalizes valid transaction rows, and audits
  excluded statement/footer rows.
- `src/train_model.py`: word + character TF-IDF and Logistic Regression with
  reproducible merchant-grouped holdout metrics.
- `src/predict_phonepe.py`: consistent model text features, deterministic
  description rules, saved merchant/transaction feedback precedence, and
  atomic prediction persistence.
- `src/financial_intelligence.py`: canonical ledger, monthly cash-flow
  arithmetic, category-proxy flexibility, recurring evidence, clustering,
  exploratory anomalies, historical expense baselines, goal math, scenario
  arithmetic, and local persistence.
- `src/pipeline.py`: runs processing, model freshness checks, classification,
  and derived analytics.
- `app.py`: local Streamlit interface.
- `data/*.csv`: source datasets and derived results; original inputs are
  preserved.
- `data/fintwin_state.sqlite3`: locally persisted goals and anomaly-review
  statuses; does not store a second transaction ledger.
- `reports/`: actual model metrics and pipeline/cluster metadata.

## Environment and run state

Verified interpreter: project virtual environment at
`D:\FinTwin\venv\Scripts\python.exe`, Python 3.13.12. Installed versions
recorded in `requirements.txt`. The previous `pyvenv.cfg` path was stale, but
the interpreter now runs successfully in this workspace.

Run from the root:

```powershell
.\venv\Scripts\python.exe -m src.pipeline
.\venv\Scripts\python.exe -m streamlit run app.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Data inventory and provenance

| File | Rows/columns | Purpose and status |
|---|---:|---|
| `data/training.csv` | 900 × 27 | Sole supervised source; 13 labels, 108 merchants; no missing labels. Category counts: Grocery 120, Food & Dining 110, Personal Transfer 100, Shopping 85, Transport 80, Bills & Utilities 75, Mobile Recharge 65, Health 55, Personal Care 50, Entertainment 45, Education 45, Salary/Income 35, Rent 35. |
| `data/testing.csv` | 1,594 CSV records × 8 (including footer records) | Current supplied PhonePe statement; valid dates span 2025-10-26–2026-10-22. |
| `data/phonepe_processed.csv` | 1,587 × 28 | Valid PhonePe rows plus physical source-line traceability. |
| `data/phonepe_predictions.csv` | 1,587 × 33 | Current predictions, confidence, final category and source. |
| `data/final_phonepe_results.csv` | 1,587 × 33 | Same current result as the predictions file. |
| `data/user_feedback.csv` | 9 × 4 | Preserved merchant corrections with scope and transaction-ID fields. |
| `data/spending_behavior.csv` | 507 × 44 | Classified consumption debits and richer KMeans cluster features. |
| `data/anomaly_results.csv` | 1,487 × 48 | Debit rows with Isolation Forest, robust IQR comparison and review fields. |
| `data/preprocessing_audit.csv` | 7 × 4 | Footer/non-transaction exclusion reasons and physical source lines 1593–1600. |
| `data/expense_flexibility.csv` | 1,587 × 42 | Category-proxy flexibility class and explanation per row. |
| `data/recurring_transactions.csv` | 117 × 10 | Repeated merchant/date evidence, not a promise of future recurrence. |

### Other supplied datasets and duplicates

| Existing file | Schema/size | Provenance and use |
|---|---|---|
| `miscellaneous/FinTwin_Indian_PhonePe_Compatible_Training .csv` | 1,000 × 27 | Supplemental historical training draft; not used by the active model. |
| `miscellaneous/training(wrong).csv` | 1,000 × 27 | Byte-identical duplicate of the supplemental training draft; not additional examples. |
| `miscellaneous/PhonePe_testing dataset.csv` | 319 × 8 | Older May–August 2026 PhonePe statement; not byte-identical to the current testing file and not used separately. |
| `miscellaneous/archive (5).zip` → `personal_finance_tracking.csv` | 1,000 × 16 | Separate USD synthetic budgeting dataset (archive README calls it synthetic). Not merged because it is not the PhonePe history and uses another schema/currency. |
| `miscellaneous/archive (5).zip` → `personal_finance_tracking.xlsx` | 1 sheet, 1,000 × 16 | Spreadsheet copy of the same archived finance table; not used. |

The archived table's 16 fields are transaction_id, transaction_date,
transaction_type, category, subcategory, amount, currency, payment_method,
merchant, city, account_type, monthly_budget, remaining_budget, savings_goal,
transaction_status, and notes.

### Active CSV schemas

- `data/training.csv` fields (27): `transaction_id`, `transaction_date`,
  `transaction_time`, `transaction_details`, `utr`, `transaction_type`,
  `payment_method`, `amount`, `currency`, `merchant`, `subcategory`,
  `category`, `cashflow_type`, `expense_nature`, `flexibility_level`,
  `flexibility_score`, `transaction_status`, `notes`, `city`, `is_expense`,
  `amount_band`, `day_of_week`, `month`, `year`, `monthly_budget`,
  `remaining_budget`, `savings_goal`.
- `data/testing.csv` fields (8): `Date`, `Time`, `Transaction Details`,
  `Transaction ID`, `UTR`, `Transaction Type`, `Credit/debit instrument`,
  `Amount`. The file has a two-line statement preamble and a blank line before
  the header. Date values use mixed month-name formats; parsing uses mixed
  format recognition. Seven footer records are audited, not treated as
  transactions.
- `data/phonepe_processed.csv` has the training schema plus `source_line`
  (28 fields). Budget, city, and transaction status are unknown in the raw
  statement and are not fabricated.
- Prediction files add `text`, `ml_category`, `ml_confidence`,
  `final_category`, and `category_source` (33 fields).
- `data/user_feedback.csv` retains the original `merchant`,
  `final_category` columns; new corrections add `scope` and `transaction_id`
  to distinguish transaction-specific from exact-merchant memory.
- Spending, anomaly, and flexibility files begin with the 33 prediction
  fields above, then canonical ledger fields `merchant_key`, `month_period`,
  `signed_cash_flow`, `is_transfer`, `is_salary_income`,
  `is_consumption_expense`, and `is_unclassified_debit`.
- `data/spending_behavior.csv` adds `log_amount`, `weekday`,
  `category_feature`, and `spending_cluster` to those canonical fields.
- `data/anomaly_results.csv` adds `anomaly`, `anomaly_score`,
  `anomaly_status`, `review_priority`, `robust_amount_outlier`,
  `anomaly_explanation`, `robust_baseline`, and `review_status`.
- `data/expense_flexibility.csv` adds `flexibility_class` and
  `flexibility_reason`.
- `data/recurring_transactions.csv` fields (10): `merchant`, `observations`,
  `first_seen`, `last_seen`, `median_interval_days`, `interval_variation`,
  `median_amount`, `amount_variation`, `evidence`, `status`.
- `data/preprocessing_audit.csv` fields (4): `source_file`, `source_line`,
  `disposition`, `reason`. It intentionally omits private transaction
  descriptions and identifiers.

Training-file annotation, collection method, source license, and ground-truth
verification are not documented in the supplied project. Its labels are used
for the required supervised experiment, but the measured scores do not prove
performance on independently verified real-world PhonePe labels.

### Input SHA-256 integrity baseline

| Source file | SHA-256 |
|---|---|
| `data/training.csv` | `a45c74d5e2cf785a81944f8a238b3397d4957e840afd64be1a9296a42e5817da` |
| `data/testing.csv` | `29287c81046712951cfc3596ed24b684444820d3fa48d485f52312e8304f86af` |
| `data/user_feedback.csv` | `3992430c9f10e717ac381a82928561de1a39c951804186d19f51bc58db68bece` |
| Supplemental training pair | `efe3c376593c19583a5eba64ddb6cc6b08c0f212c54ef2e4428cb8d8f47e5195` |
| Older raw PhonePe duplicate | `53db118cf92f8c565d9bc626dbb661ef109cad485be3f04c07b72ced014555ae` |
| Archive ZIP | `7fe3d71da660647eb16b127b7421cdada008524bab2cfb3886d72a0cf2ffe611` |

The active result source counts include 971 `Needs Review` rows (low ML
confidence), 512 deterministic-rule rows, 42 saved merchant-feedback rows,
and 62 direct ML rows. This substantial review burden is exposed rather than
hidden by broad person-name rules. The 1,587-row active statement has 1,487
debits and 100 credits. Only classified, non-transfer debits enter consumption
totals; unclassified debits and transfers are kept separate. The ledger's
all-debit and cash-flow totals still reconcile.

`miscellaneous/FinTwin_Indian_PhonePe_Compatible_Training .csv` and
`miscellaneous/training(wrong).csv` are byte-identical historical copies and
are excluded from training. `miscellaneous/PhonePe_testing dataset.csv` is an
older statement and is excluded from pipeline input.
`miscellaneous/archive (5).zip` includes a separate USD personal-finance
dataset; it is not merged or used.

## Evaluation and limitations

The live, reproducible classification metrics are in
`reports/classification_metrics.json`; this file is created by the actual
training run, not hardcoded. It includes merchant overlap, held-out class
coverage, per-class metrics, confusion matrix, and latency.

PhonePe labels are predictions/user corrections, not independently verified
ground truth. Statement boundaries are partial calendar months. No account
balances, net worth, complete income history, debt schedule, or verified
fraud labels exist in the supplied statement. Forecasts, recurrence,
flexibility, and anomaly results are limited accordingly.
The statement ends in August 2026; as of October 2026, its September
next-observation estimate is retrospective and is not a current forecast.
The app labels this data gap explicitly. Rebuild the offline experiment report
after a pipeline run with `python -m src.research`.

## Integrity

Only the existing `data/training.csv` is used for supervised training.
PhonePe predictions are never treated as training labels. Original source
files are preserved; transformations are written to derived outputs. See
`EXPERIMENTS.md` for actual measured results and `ISSUES.md` for remaining
limitations.
