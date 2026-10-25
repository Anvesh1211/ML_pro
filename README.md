# FinTwin AI — Adaptive Personal Finance Intelligence

run this exact in the terminal  fintwin to run the project plz do not share any of the dataset to anyone else 
.\venv\Scripts\python.exe -m src.pipeline
.\venv\Scripts\python.exe -m src.research
.\venv\Scripts\python.exe -m streamlit run app.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v 


git add README.md
git commit -m "Update"
git push -u origin main
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

The standalone prediction pipeline supports the existing
`data/user_feedback.csv` corrections. In the signed-in website, corrections
are stored per account in SQLite, separately from the original model
prediction. A transaction correction affects only that transaction; a
merchant-level correction requires explicit confirmation and is scoped to the
signed-in user's imported transactions. Website goals, anomaly-review status,
recommendations, imports, and transactions are also account-scoped.

## Website

The Streamlit app includes:

- Local account registration, sign-in, session expiry, and logout.
- Private PhonePe CSV upload, validation preview, import history, and duplicate
  review.
- Financial overview and transaction search/review.
- Spending and category-flexibility analysis.
- Financial-state coverage, with unavailable balances/income disclosed.
- Historical-mean and moving-average expense baselines with chronological
  validation where possible.
- Explicit what-if income, expense, EMI, and emergency scenarios.
- Persisted goals and contribution-feasibility arithmetic.
- Exploratory anomaly review with saved review status.
- JSON report and identifier-redacted transaction export.

## Accounts, imports, and local storage

The app uses a single local SQLite database for account records. Its default
location is `data/fintwin_state.sqlite3`; set `FINTWIN_DATABASE_PATH` before
starting Streamlit to use another path. It does not load `.env` files
automatically. The optional settings in `.env.example` can be set as process
environment variables:

```powershell
$env:FINTWIN_DATABASE_PATH = "data\fintwin_state.sqlite3"
$env:FINTWIN_SESSION_TTL_MINUTES = "480"
$env:FINTWIN_MAX_UPLOAD_MB = "10"
.\venv\Scripts\python.exe -m streamlit run app.py
```

Passwords are stored as salted scrypt hashes. Browser session tokens are
random, opaque values; only their SHA-256 digests are persisted in SQLite.
Sessions expire after the configured number of minutes (480 by default), and
logging out revokes the current session. Authentication does not include
email verification or password recovery, so this prototype should remain on
localhost unless a deployment adds appropriate account-recovery and
operational controls.

The upload screen accepts `.csv` PhonePe statements with date, transaction
details, transaction type, and amount columns. Transaction identifiers,
timestamps, payment method, and UTR are optional. Files are validated and
previewed in memory before import; invalid rows are recorded with their source
line and reason. Exact transaction-ID matches are skipped, while uncertain
matches are retained and marked for review. Imported rows are stored in minor
currency units and associated with the signed-in account. The existing
`data/testing.csv` is not automatically imported into any account. Uploaded
data and predictions do not retrain or overwrite the shared classifier or its
global output CSVs.

Database initialization is additive. It preserves legacy goals and anomaly
reviews and does not assign those unscoped records to a newly registered
account. Before upgrading an existing database, the app creates a
timestamped `.pre-migration-*.bak` backup next to it, then applies schema
version 3 and checks foreign-key integrity. Backups are local and ignored by
Git. Keep a separate protected backup for important financial data.

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