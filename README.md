# FinTwin AI — Adaptive Personal Finance Intelligence

FinTwin is a local-first personal-finance prototype. It classifies PhonePe
transactions with a merchant-disjoint-evaluated text model, applies explicit
user corrections, separates transfers from consumption, and provides
explainable spending, recurrence, baseline forecasting, goal, and scenario
calculations.

It is not a bank connection, credit score, fraud detector, or financial
adviser. PhonePe categories have not been independently verified as ground
truth. The current statement has no account balances and does not establish a
complete income history.

## Data policy and provenance

The project uses existing files only:

- `data/training.csv` is the sole supervised classification source.
- `data/testing.csv` is the latest supplied PhonePe statement. Its metadata,
  blank separator, and CSV header are retained; the importer supports the
  statement's mixed month-name date formats.
- The current file has 1,587 valid transactions from 2025-10-26 through
  2026-10-22. Seven trailing statement-footer records are excluded and
  recorded with their physical source lines in `data/preprocessing_audit.csv`.
- `data/user_feedback.csv` contains existing user merchant corrections.
- Other existing prediction/analysis CSVs are regenerated derived outputs, not
  training labels.
- Identical duplicate files in `miscellaneous/` and the unrelated archived
  USD personal-finance dataset are not added to training or analytics.
- The original training and supplemental files are preserved. The active
  `testing.csv` was replaced with the newly supplied October 2025–October 2026
  statement; the older duplicate in `miscellaneous/` is not used.
- The training table has 900 rows, 27 fields, 13 categories, and 108 merchants;
  the active PhonePe statement has 1,587 valid transactions after auditing
  seven non-transaction footer records.
- Exact field lists, output schemas, duplicate hashes, archive details, and
  dataset limitations are documented in
  [`ANTIGRAVITY_OS/PROJECT.md`](ANTIGRAVITY_OS/PROJECT.md).

Processing is local. No external financial API or cloud AI service is called.
UTRs and transaction identifiers are kept in the local ledger to retain
traceability, but omitted from the default report export.

## Environment

Tested with Python 3.13 and the versions pinned in `requirements.txt`.

```powershell
cd D:\FinTwin
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The existing `venv/` may be used if it is valid:

```powershell
.\venv\Scripts\python.exe -m src.pipeline
```

## Run the application

From the project root:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively substitute `.\venv\Scripts\python.exe` if using the existing
environment. Streamlit prints the local URL in the terminal. Keep the server
bound to localhost unless you intentionally configure and secure another
deployment.

## Pipeline commands

```powershell
# Process the existing statement, train only if the model is missing/stale,
# predict, then refresh analytics.
python -m src.pipeline

# Force a deterministic model retrain and merchant-disjoint evaluation.
python -m src.pipeline --retrain

# Run only the training evaluation and save the model/metrics.
python -m src.train_model

# Reprocess the PhonePe statement and write an exclusion audit.
python -m src.preprocess_phonepe

# Refresh the offline personalization, flexibility, forecasting, clustering,
# and hypothetical scenario evaluations using verified project outputs.
python -m src.research
```

Run commands from `D:\FinTwin`; modules resolve data paths relative to the
project root. The model is trained using the original 900-row
`data/training.csv`, word and character TF-IDF, Logistic Regression, and a
fixed merchant-grouped 80/20 holdout. `reports/classification_metrics.json`
records the actual run metrics and any categories missing from the held-out
set. PhonePe predictions are not used for model training.

## Feedback and persistence

Use the **Transaction intelligence** page to correct either one transaction
or, after explicit confirmation, all current/future transactions with the
exact merchant name. The app saves feedback in `data/user_feedback.csv`,
preserves the model prediction, and refreshes dependent predictions and
analytics. Legacy two-column feedback rows are interpreted as the
merchant-level corrections recorded by the prior review script. SQLite stores
user-created goals and anomaly-review status in `data/fintwin_state.sqlite3`;
it does not duplicate the transaction ledger.

## Website

The Streamlit app includes:

- Financial overview and transaction search/review.
- Spending and category-flexibility analysis.
- Financial-state coverage, with unavailable balances/income disclosed.
- Historical-mean and moving-average expense baselines with chronological
  validation where possible.
- Explicit what-if income, expense, EMI, and emergency scenarios.
- Persisted goals and contribution-feasibility arithmetic.
- Exploratory anomaly review with saved review status.
- JSON report and identifier-redacted transaction export.

## Tests and evaluation

```powershell
python -m unittest discover -s tests -v
```

The unit tests use the project statement/training data for data/model checks
and hand-calculated **scenario parameters** for deterministic arithmetic
checks. They do not create synthetic training or financial transaction data.
The current statement covers October 2025–October 2026 with partial boundary
months. This supports a next-month expense baseline but not a verified income
forecast or reliable seasonal claim. Anomaly outputs have no verified fraud
labels and must be treated as review signals only.

See `ANTIGRAVITY_OS/PROJECT.md`, `ISSUES.md`, `DECISIONS.md`,
`DEBUG_LOGS.md`, and `EXPERIMENTS.md` for the recorded state, decisions,
limitations, and measured runs.

.\venv\Scripts\python.exe -m src.pipeline
.\venv\Scripts\python.exe -m src.research
.\venv\Scripts\python.exe -m streamlit run app.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v