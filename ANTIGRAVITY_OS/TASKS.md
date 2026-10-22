# FinTwin AI — Implementation Checklist

## Dataset and pipeline foundation
- [x] Inspect project, interpreter, dependencies, source and datasets.
- [x] Verify duplicates and archive; exclude unrelated/duplicate files.
- [x] Validate statement columns, preserve identifiers as strings and audit footer rows.
- [x] Preserve training and supplemental sources; replace the PhonePe input only when explicitly requested and regenerate traceable derived outputs.
- [x] Reproducible grouped model evaluation and model freshness check.
- [x] Persist predictions and validate output by reading it back.
- [x] Load and apply scoped saved feedback without overwriting model prediction.

## Financial intelligence
- [x] Canonical ledger validation and month-wise sums.
- [x] Separate personal transfers from consumption spending.
- [x] Category-proxy flexibility with caveats.
- [x] Recurrence evidence based only on observed dates/amounts.
- [x] Richer baseline clusters and exploratory anomaly outputs.
- [x] Chronological simple expense forecasts with baseline errors.
- [x] Deterministic scenario/goal arithmetic and local goal persistence.
- [x] Explicit missing income/balance/debt warnings; no fabricated health score.

## Website and quality
- [x] Local Streamlit application with nine functional workspaces.
- [x] Transaction search, single-transaction correction, confirmed exact-merchant memory.
- [x] Scenario inputs, goal persistence, anomaly review status and report downloads.
- [x] Local-only privacy notes and identifier-redacted default export.
- [x] Add unit/integration checks using existing datasets and scenario parameters.
- [x] Compile all Python modules and pass all 19 unittest checks after final edits.
- [x] Start the local app, verify the health endpoint and exercise all nine workspaces, scenario calculation, and report downloads.
- [x] Complete measured experiment report and document remaining limitations.
- [x] Check the rendered overview at a 390px mobile viewport; no horizontal overflow.
