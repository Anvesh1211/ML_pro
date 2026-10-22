# FinTwin AI — Engineering Decisions

1. **Use only existing project data.** `data/training.csv` is the sole
   supervised source; the PhonePe statement is inference/analysis input.
   Duplicate historical files and the separate archived USD dataset are not
   merged.
2. **Keep CSV as the transaction source of truth.** SQLite stores only goals
   and anomaly review state; it does not duplicate the transaction history.
3. **Retain merchant-disjoint model evaluation.** This better tests
   generalization beyond seen merchants than a random row split.
4. **Use deterministic preprocessing and a single feature builder.** Model
   text uses the same fields at train and inference time.
5. **Prefer explicit uncertainty.** Unknown categories are sent for review;
   missing income, balance, and savings are not silently zero-filled.
6. **Treat feedback scopes separately.** Transaction-specific correction is
   default in the UI. Exact merchant memory requires user confirmation.
   Existing legacy feedback stays merchant-scoped to preserve its prior
   semantics.
7. **Separate transfers from consumption.** Credits and debits remain
   distinct; personal-transfer debits are disclosed but excluded from
   consumption totals.
8. **Use interpretable baselines where history is short.** Expense forecasts
   are historical-mean/moving-average baselines with chronological errors;
   no seasonal model is justified by the available history.
9. **Keep processing local.** No external financial API, cloud AI service, or
   external dataset is used.
10. **Preserve source files and write derived outputs atomically.** Footer
    exclusions are recorded by source line and reason.
