# FinTwin AI — Debug and Validation Log

## 2026-10-22 — Baseline verification and initial repair

- Confirmed Python 3.13.12 works from `venv\Scripts\python.exe`; it currently
  provides pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0, NumPy 2.5.4, and
  Streamlit 1.66.1.
- Confirmed PhonePe statement read as 319 post-header rows; 312 valid debit/
  credit transactions and seven footer rows. The footer exclusions are
  classified by reason in `data/preprocessing_audit.csv`.
- Preserved `data/testing.csv`, `data/training.csv`, `data/user_feedback.csv`,
  and all supplemental originals.
- Ran the revised pipeline using `data/training.csv` and the real statement.
- Merchant-disjoint classifier holdout produced 0.9435 accuracy, 0.9197
  macro-F1, and 0.9417 weighted-F1 (details in
  `reports/classification_metrics.json`).
- Pipeline wrote and read back 312 predictions and 278 debit anomaly rows.

## Final validation — 2026-10-22

- `python -m compileall -q src app.py tests`: passed.
- `python -m unittest discover -s tests -v`: all 19 tests passed. A prior run
  caught a stale test assumption that all non-transfer debits enter clustering;
  the assertion now uses the canonical `is_consumption_expense` definition.
- Re-ran `python -m src.pipeline`: completed and read-back validated; 312
  processed/predicted rows, 278 debit anomaly rows, 34 repeated-merchant
  patterns, and the expected merchant-disjoint metrics.
- Streamlit AppTest exercised all nine workspaces without exceptions, ran a
  scenario with an explicit income assumption, and confirmed both report
  download controls render.
- Local Streamlit health endpoint returned HTTP 200 (`ok`).
- Browser rendered the live overview, including the stale-statement warning,
  data-coverage note, distinct credits/debits, unclassified debit disclosure,
  and chart. At 390px viewport width, document and body widths were also
  390px (no horizontal overflow).
- SHA-256 checks confirm training data, raw PhonePe statement, feedback,
  supplemental duplicate pair, duplicate statement, and archive are unchanged.
