# FinTwin AI — Debug and Validation Log

## 2026-10-22 — Earlier baseline verification and initial repair

- Confirmed Python 3.13.12 works from `venv\Scripts\python.exe`; it currently
  provides pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, NumPy 2.5.4, and
  Streamlit 1.66.1.
- At that time, the PhonePe statement had 312 valid debit/credit transactions
  and seven footer rows. Those outputs were superseded by the user-requested
  October 2025–October 2026 statement replacement below.
- Ran the revised pipeline using `data/training.csv` and the real statement.
- Merchant-disjoint classifier holdout produced 0.9435 accuracy, 0.9197
  macro-F1, and 0.9417 weighted-F1 (details in
  `reports/classification_metrics.json`).
- Pipeline wrote and read back 312 predictions and 278 debit anomaly rows.

## 2026-10-22 — User-requested PhonePe statement replacement

- Validated the newly supplied statement before replacement: expected
  eight-column schema; 1,587 valid transactions, 7 audited footer records,
  unique transaction IDs, and mixed-format dates spanning 2025-10-26 through
  2026-10-22.
- Updated date parsing to handle mixed month-name formats and mapped each CSV
  record to its physical source line, including the footer audit.
- Replaced `data/testing.csv` and ran `python -m src.pipeline`; all 1,587
  predictions were written and read back. Derived outputs now contain 1,487
  debit anomaly rows and 117 repeated-merchant patterns.
- Refreshed `reports/experiments.json`. The classifier training source and
  model remain the existing `data/training.csv`; the new statement was not
  added to training.
- Updated statements, schemas, hashes, and measured result documentation.
- The first test pass identified hardcoded old row counts and feedback counts;
  assertions now derive row counts from the active statement and current
  feedback file.
- Final compile and test run passed: all 19 unittest checks. The parser test
  independently calculates expected valid rows using mixed-format dates.
- Streamlit AppTest rendered all nine workspaces without exceptions, completed
  a scenario with explicit income, and verified both report downloads.
- Read-back checks confirm 1,587 rows in the processed, prediction, final
  result, and flexibility files; 1,487 anomaly rows; 507 spending-cluster
  rows; 117 recurrence summaries; and seven audited footer records.
- Active statement SHA-256: `29287c81046712951cfc3596ed24b684444820d3fa48d485f52312e8304f86af`.
